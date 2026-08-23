//! Immutable CUDA Graph capture and replay ownership.

use crate::{CudaContext, CudaStream, GpuError, GpuResult};
use cuda_core::{IntoResult, sys};
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

/// One immutable CUDA Graph and its executable instance.
///
/// Captured operations retain device addresses, so every allocation named by
/// the recording must outlive this graph and all of its launches.
pub struct CudaGraph {
    context: Arc<CudaContext>,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
}

impl CudaGraph {
    /// Captures operations submitted by `record` and instantiates them.
    pub fn capture<F>(stream: &CudaStream, record: F) -> GpuResult<Self>
    where
        F: FnOnce() -> GpuResult<()>,
    {
        let context = stream.context().clone();
        context
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
        // SAFETY: the stream is live and its context is current on this thread.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
        }
        .map_err(|source| GpuError::driver("beginning CUDA Graph capture", source))?;
        let capture = ActiveCapture {
            context: context.clone(),
            stream,
            active: true,
        };

        record()?;

        let graph = capture.finish()?;
        if graph.is_null() {
            return Err(GpuError::graph("CUDA stream capture returned a null graph"));
        }

        let mut executable = ptr::null_mut();
        // SAFETY: `graph` is the live graph returned by the completed capture.
        let instantiate = unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0) }
            .result()
            .map_err(|source| GpuError::driver("instantiating a CUDA Graph", source));
        if let Err(error) = instantiate {
            // SAFETY: instantiation failed without transferring ownership of `graph`.
            context.record_err(unsafe { sys::cuGraphDestroy(graph).result() });
            return Err(error);
        }
        if executable.is_null() {
            // SAFETY: a null executable leaves `graph` owned by this function.
            context.record_err(unsafe { sys::cuGraphDestroy(graph).result() });
            return Err(GpuError::graph(
                "CUDA Graph instantiation returned a null executable",
            ));
        }

        Ok(Self {
            context,
            graph,
            executable,
        })
    }

    /// Enqueues this immutable graph on a stream from the same CUDA context.
    pub fn launch(&self, stream: &CudaStream) -> GpuResult<()> {
        if self.context.as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "CUDA Graph and launch stream belong to different contexts",
            ));
        }

        self.context
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
        // SAFETY: both handles are live and belong to `self.context`.
        unsafe { sys::cuGraphLaunch(self.executable, stream.cu_stream()).result() }
            .map_err(|source| GpuError::driver("launching a CUDA Graph", source))
    }

    /// Writes CUDA's structural graph inventory for an out-of-band profiling artifact.
    pub fn debug_dot(&self, path: &Path) -> GpuResult<()> {
        self.context
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| GpuError::graph("CUDA Graph DOT path contains an interior NUL"))?;
        let flags = sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
            | sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
            | sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO;
        // SAFETY: the graph is live, the path is NUL-terminated, and CUDA writes the output file
        // before returning without retaining either borrowed value.
        unsafe { sys::cuGraphDebugDotPrint(self.graph, path.as_ptr(), flags) }
            .result()
            .map_err(|source| GpuError::driver("writing CUDA Graph DOT inventory", source))
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
        // SAFETY: this owner destroys each live handle exactly once.
        self.context
            .record_err(unsafe { sys::cuGraphExecDestroy(self.executable).result() });
        self.context
            .record_err(unsafe { sys::cuGraphDestroy(self.graph).result() });
    }
}

struct ActiveCapture<'a> {
    context: Arc<CudaContext>,
    stream: &'a CudaStream,
    active: bool,
}

impl ActiveCapture<'_> {
    fn finish(mut self) -> GpuResult<sys::CUgraph> {
        let mut graph = ptr::null_mut();
        // SAFETY: this guard owns the active capture on its borrowed stream.
        let result =
            unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph).result() };
        self.active = false;

        if let Err(source) = result {
            if !graph.is_null() {
                // SAFETY: a failed end returned this graph without transferring ownership.
                self.context
                    .record_err(unsafe { sys::cuGraphDestroy(graph).result() });
            }
            return Err(GpuError::driver("ending CUDA Graph capture", source));
        }

        Ok(graph)
    }
}

impl Drop for ActiveCapture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        self.active = false;
        self.context.record_err(self.context.bind_to_thread());
        let mut graph = ptr::null_mut();
        // SAFETY: this guard still owns the active capture on its borrowed stream.
        self.context.record_err(unsafe {
            sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph).result()
        });
        if !graph.is_null() {
            // SAFETY: cleanup owns any graph returned while ending the abandoned capture.
            self.context
                .record_err(unsafe { sys::cuGraphDestroy(graph).result() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CudaGraph;
    use crate::{ArenaLayout, CudaContext, DeviceArena, GpuError};

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn captured_fill_replays_over_a_stable_arena_address() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let prefix = layout.reserve::<u8>(13, 1).unwrap();
        let values = layout.reserve::<u32>(4, 256).unwrap();
        let suffix = layout.reserve::<u8>(7, 1).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();
        let base_address = arena.base_address();
        let values_address = arena.address(values).unwrap();
        let graph = CudaGraph::capture(&stream, || arena.fill(&stream, values, 0x5a)).unwrap();

        graph.launch(&stream).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(arena.base_address(), base_address);
        assert_eq!(arena.address(values).unwrap(), values_address);
        let host = arena.to_host_vec(&stream).unwrap();
        assert!(
            host[prefix.offset_bytes()..prefix.offset_bytes() + prefix.byte_len()]
                .iter()
                .all(|byte| *byte == 0)
        );
        assert!(
            host[values.offset_bytes()..values.offset_bytes() + values.byte_len()]
                .iter()
                .all(|byte| *byte == 0x5a)
        );
        assert!(
            host[suffix.offset_bytes()..suffix.offset_bytes() + suffix.byte_len()]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn failed_recording_ends_capture_and_leaves_the_stream_reusable() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let region = layout.reserve::<u8>(16, 256).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();

        let error = CudaGraph::capture(&stream, || Err(GpuError::graph("recording failed")))
            .err()
            .unwrap();

        assert!(error.to_string().contains("recording failed"));
        arena.fill(&stream, region, 0xa5).unwrap();
        stream.synchronize().unwrap();
        assert!(
            arena
                .to_host_vec(&stream)
                .unwrap()
                .iter()
                .all(|byte| *byte == 0xa5)
        );
    }
}

//! Immutable CUDA Graph capture and replay ownership.

use crate::{CudaContext, CudaStream, GpuError, GpuResult};
use cuda_core::{IntoResult, sys};
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

/// One captured CUDA Graph definition without an executable instance.
pub struct CudaGraphDefinition {
    context: Arc<CudaContext>,
    graph: sys::CUgraph,
}

/// Exact topology-compatible graph definitions sharing one executable instance.
///
/// The first launch pins its stream: every later launch must pass the same
/// stream, so the shared executable is only ever updated or re-enqueued behind
/// the launches already ordered on that one stream.
///
/// The definitions retain captured device addresses without keeping the
/// allocations behind them alive; [`CudaGraphVariants::launch`] is `unsafe`
/// over that obligation.
pub struct CudaGraphVariants<const N: usize> {
    context: Arc<CudaContext>,
    definitions: [CudaGraphDefinition; N],
    executable: sys::CUgraphExec,
    selected: Mutex<SelectedLaunch>,
}

/// Serialized launch state: the resident variant and the pinned launch stream.
struct SelectedLaunch {
    index: usize,
    /// Raw handle of the stream every launch has used, or `None` before the
    /// first launch. Compared by handle only and never dereferenced; the owner
    /// keeps that stream alive across launches.
    stream: Option<sys::CUstream>,
}

/// One immutable CUDA Graph and its executable instance.
///
/// Captured operations retain device addresses, so every allocation named by
/// the recording must outlive this graph and all of its launches. The graph
/// holds no handle to those allocations; [`CudaGraph::launch`] is `unsafe`
/// over exactly that obligation.
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
        let definition = CudaGraphDefinition::capture(stream, record)?;
        let executable = instantiate(&definition)?;
        let (context, graph) = definition.into_parts();

        Ok(Self {
            context,
            graph,
            executable,
        })
    }

    /// Enqueues this immutable graph on a stream from the same CUDA context.
    ///
    /// # Safety
    ///
    /// Replay re-issues the captured operations against the raw addresses they
    /// recorded, and this graph does not keep those allocations alive. The
    /// caller must guarantee that every allocation the recording captured —
    /// device arenas, TMA descriptor maps, pinned host staging buffers, and
    /// loaded module code — is still alive at its captured address when this
    /// call enqueues the graph, and stays alive and unmoved until `stream`
    /// completes the replayed work.
    pub unsafe fn launch(&self, stream: &CudaStream) -> GpuResult<()> {
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
        debug_dot(&self.context, self.graph, path)
    }
}

impl CudaGraphDefinition {
    /// Captures operations without allocating an executable graph instance.
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
        Ok(Self { context, graph })
    }

    /// Writes this definition's structural inventory for out-of-band inspection.
    pub fn debug_dot(&self, path: &Path) -> GpuResult<()> {
        debug_dot(&self.context, self.graph, path)
    }

    fn into_parts(self) -> (Arc<CudaContext>, sys::CUgraph) {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` will not run Drop; ownership of both fields moves to the caller.
        let context = unsafe { ptr::read(&this.context) };
        (context, this.graph)
    }
}

impl<const N: usize> CudaGraphVariants<N> {
    /// Instantiates the first of an exact non-empty definition inventory.
    pub fn new(definitions: [CudaGraphDefinition; N]) -> GpuResult<Self> {
        let first = definitions
            .first()
            .ok_or_else(|| GpuError::graph("CUDA Graph variant inventory cannot be empty"))?;
        if definitions
            .iter()
            .any(|definition| definition.context.as_ref() != first.context.as_ref())
        {
            return Err(GpuError::context(
                "CUDA Graph variants must belong to one context",
            ));
        }
        let context = first.context.clone();
        let executable = instantiate(first)?;
        Ok(Self {
            context,
            definitions,
            executable,
            selected: Mutex::new(SelectedLaunch {
                index: 0,
                stream: None,
            }),
        })
    }

    /// Updates the shared executable when the variant changes and enqueues one variant.
    ///
    /// The first launch pins its stream; every later launch must pass that same stream and is
    /// rejected otherwise. Pinning keeps all launches of the shared executable ordered on one
    /// stream, so a variant switch only needs to drain that stream before the update, and the
    /// same-stream re-launch path needs no synchronization at all.
    ///
    /// # Safety
    ///
    /// Replay re-issues the selected definition's captured operations against
    /// the raw addresses they recorded, and neither the definitions nor the
    /// shared executable keep those allocations alive. The caller must
    /// guarantee that every allocation captured by the definition at `index` —
    /// device arenas, TMA descriptor maps, pinned host staging buffers, and
    /// loaded module code — is still alive at its captured address when this
    /// call updates and enqueues the shared executable, and stays alive and
    /// unmoved until `stream` completes the replayed work.
    pub unsafe fn launch(&self, stream: &CudaStream, index: usize) -> GpuResult<()> {
        if self.context.as_ref() != stream.context().as_ref() {
            return Err(GpuError::context(
                "CUDA Graph variants and launch stream belong to different contexts",
            ));
        }
        let definition = self.definitions.get(index).ok_or_else(|| {
            GpuError::graph(format!(
                "CUDA Graph variant {index} is outside an exact {N}-entry inventory"
            ))
        })?;
        let mut selected = self
            .selected
            .lock()
            .map_err(|_| GpuError::graph("CUDA Graph variant selection lock is poisoned"))?;
        if let Some(pinned) = selected.stream
            && pinned != stream.cu_stream()
        {
            return Err(GpuError::context(
                "CUDA Graph variants are pinned to their first launch stream",
            ));
        }
        if selected.index != index {
            // Every prior launch was enqueued on this pinned stream, so draining it proves no
            // launch of the shared executable is still executing during the update below.
            stream.synchronize().map_err(|source| {
                GpuError::driver("synchronizing before CUDA Graph update", source)
            })?;
            self.context
                .bind_to_thread()
                .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
            let mut result = sys::CUgraphExecUpdateResultInfo {
                result: sys::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR,
                errorNode: ptr::null_mut(),
                errorFromNode: ptr::null_mut(),
            };
            // SAFETY: the executable and definition are live, context-matched handles. Every
            // earlier launch of this serialized owner was enqueued on the pinned stream just
            // synchronized, so the executable is idle for the update.
            unsafe { sys::cuGraphExecUpdate_v2(self.executable, definition.graph, &mut result) }
                .result()
                .map_err(|source| GpuError::driver("updating a CUDA Graph executable", source))?;
            if result.result != sys::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_SUCCESS {
                return Err(GpuError::graph(format!(
                    "CUDA Graph executable update returned result {}",
                    result.result
                )));
            }
            selected.index = index;
        }

        self.context
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
        // SAFETY: the selected executable is live and belongs to the launch stream's context;
        // the pin check above serialized this launch behind every prior one on this stream.
        unsafe { sys::cuGraphLaunch(self.executable, stream.cu_stream()).result() }
            .map_err(|source| GpuError::driver("launching a CUDA Graph variant", source))?;
        selected.stream = Some(stream.cu_stream());
        Ok(())
    }

    /// Writes one exact definition's structural inventory for out-of-band inspection.
    pub fn debug_dot(&self, index: usize, path: &Path) -> GpuResult<()> {
        self.definitions
            .get(index)
            .ok_or_else(|| {
                GpuError::graph(format!(
                    "CUDA Graph variant {index} is outside an exact {N}-entry inventory"
                ))
            })?
            .debug_dot(path)
    }

    /// Exact route-definition count.
    pub const fn route_count(&self) -> usize {
        N
    }

    /// One shared executable instance.
    pub const fn executable_count(&self) -> usize {
        1
    }
}

fn instantiate(definition: &CudaGraphDefinition) -> GpuResult<sys::CUgraphExec> {
    let mut executable = ptr::null_mut();
    // SAFETY: `definition.graph` is live in the current context.
    unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, definition.graph, 0) }
        .result()
        .map_err(|source| GpuError::driver("instantiating a CUDA Graph", source))?;
    if executable.is_null() {
        return Err(GpuError::graph(
            "CUDA Graph instantiation returned a null executable",
        ));
    }
    Ok(executable)
}

fn debug_dot(context: &CudaContext, graph: sys::CUgraph, path: &Path) -> GpuResult<()> {
    context
        .bind_to_thread()
        .map_err(|source| GpuError::driver("binding the graph CUDA context", source))?;
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| GpuError::graph("CUDA Graph DOT path contains an interior NUL"))?;
    let flags = sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
        | sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
        | sys::CUgraphDebugDot_flags_enum_CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO;
    // SAFETY: the graph is live, and CUDA consumes the NUL-terminated path before returning.
    unsafe { sys::cuGraphDebugDotPrint(graph, path.as_ptr(), flags) }
        .result()
        .map_err(|source| GpuError::driver("writing CUDA Graph DOT inventory", source))
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

impl Drop for CudaGraphDefinition {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
        // SAFETY: this owner destroys its captured definition exactly once.
        self.context
            .record_err(unsafe { sys::cuGraphDestroy(self.graph).result() });
    }
}

impl<const N: usize> Drop for CudaGraphVariants<N> {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
        // SAFETY: this owner destroys its one executable before definitions drop.
        self.context
            .record_err(unsafe { sys::cuGraphExecDestroy(self.executable).result() });
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
    use super::{CudaGraph, CudaGraphDefinition, CudaGraphVariants};
    use crate::{ArenaLayout, CudaContext, DeviceArena, GpuError, device_memory_info};

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

        // SAFETY: `arena`, the only allocation the recording captured, lives
        // past the synchronize below.
        unsafe {
            graph.launch(&stream).unwrap();
            graph.launch(&stream).unwrap();
        }
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

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn topology_compatible_variants_share_one_allocation_stable_executable() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut layout = ArenaLayout::new();
        let values = layout.reserve::<u32>(4, 256).unwrap();
        let arena = DeviceArena::zeroed(&stream, &layout).unwrap();
        let first =
            CudaGraphDefinition::capture(&stream, || arena.fill(&stream, values, 0x11)).unwrap();
        let second =
            CudaGraphDefinition::capture(&stream, || arena.fill(&stream, values, 0x22)).unwrap();
        let variants = CudaGraphVariants::new([first, second]).unwrap();
        assert_eq!(variants.route_count(), 2);
        assert_eq!(variants.executable_count(), 1);

        // SAFETY: `arena`, the only allocation either definition captured,
        // lives past the final synchronize below.
        unsafe {
            variants.launch(&stream, 0).unwrap();
            variants.launch(&stream, 1).unwrap();
            variants.launch(&stream, 0).unwrap();
        }
        stream.synchronize().unwrap();
        let before = device_memory_info(&context).unwrap();
        for index in [1, 0, 1, 1, 0] {
            // SAFETY: `arena`, the only captured allocation, lives past the
            // synchronize after this loop.
            unsafe { variants.launch(&stream, index) }.unwrap();
        }
        stream.synchronize().unwrap();
        let after = device_memory_info(&context).unwrap();

        assert_eq!(
            arena.copy_to_host(&stream, values).unwrap(),
            [0x1111_1111; 4]
        );
        assert_eq!(before.free_bytes, after.free_bytes);
    }
}

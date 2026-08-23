//! Explicit profiler capture boundaries for diagnostic processes.

use crate::{CudaContext, GpuError, GpuResult};
use cuda_core::{IntoResult, sys};

unsafe extern "C" {
    fn cuProfilerStart() -> sys::CUresult;
    fn cuProfilerStop() -> sys::CUresult;
}

/// Starts an attached CUDA profiler after binding the owning context.
pub fn profiler_start(context: &CudaContext) -> GpuResult<()> {
    context
        .bind_to_thread()
        .map_err(|source| GpuError::driver("binding the profiler CUDA context", source))?;
    // SAFETY: the CUDA driver is initialized by the live context and retains no Rust value.
    unsafe { cuProfilerStart() }
        .result()
        .map_err(|source| GpuError::driver("starting CUDA profiler capture", source))
}

/// Stops an attached CUDA profiler after all captured device work completes.
pub fn profiler_stop(context: &CudaContext) -> GpuResult<()> {
    context
        .bind_to_thread()
        .map_err(|source| GpuError::driver("binding the profiler CUDA context", source))?;
    // SAFETY: the CUDA driver is initialized by the live context and retains no Rust value.
    unsafe { cuProfilerStop() }
        .result()
        .map_err(|source| GpuError::driver("stopping CUDA profiler capture", source))
}

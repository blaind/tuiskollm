//! Checked device-memory observations.

use crate::{CudaContext, GpuError, GpuResult};
use cuda_core::{IntoResult, sys};

/// Free and total bytes visible to one CUDA context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMemoryInfo {
    /// Currently free device bytes.
    pub free_bytes: usize,
    /// Total device bytes.
    pub total_bytes: usize,
}

/// Reads the CUDA driver's device-memory counters.
pub fn device_memory_info(context: &CudaContext) -> GpuResult<DeviceMemoryInfo> {
    context
        .bind_to_thread()
        .map_err(|source| GpuError::driver("binding the memory-query CUDA context", source))?;
    let mut free_bytes = 0usize;
    let mut total_bytes = 0usize;
    // SAFETY: both output pointers are valid for writes and the context is current.
    unsafe { sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes).result() }
        .map_err(|source| GpuError::driver("querying device memory", source))?;

    Ok(DeviceMemoryInfo {
        free_bytes,
        total_bytes,
    })
}

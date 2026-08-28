//! CUDA ownership primitives shared by the runtime and SM120 kernel crate.

mod arena;
mod error;
mod graph;
mod memory;
mod profiler;
mod streaming;
mod timer;
mod vmm_manifest;

pub use arena::{ArenaLayout, ArenaRegion, DeviceArena, LoadingDeviceArena};
pub use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig1D,
    LaunchConfig2D, LaunchContractError, PinnedHostBuffer, PreparedLaunch, SyncPolicy,
};
pub use error::{GpuError, GpuErrorCode, GpuResult};
pub use graph::{CudaGraph, CudaGraphDefinition, CudaGraphVariants};
pub use memory::{DeviceMemoryInfo, device_memory_info};
pub use profiler::{profiler_start, profiler_stop};
pub use streaming::{
    ABSENT_SLOT, DeviceSlotPool, INDIRECTION_TABLE_GENERATIONS, PinnedBounceRing, PinnedHostPool,
    RECLAIM_FENCE_GENERATIONS, TransferStream,
};
pub use timer::{GpuTimer, GpuTiming};
pub use vmm_manifest::{VmmSegment, VmmSegmentClass, VmmSegmentManifest};

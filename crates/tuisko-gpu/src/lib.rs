//! CUDA ownership primitives shared by the runtime and SM120 kernel crate.

mod arena;
mod error;
mod graph;

pub use arena::{ArenaLayout, ArenaRegion, DeviceArena};
pub use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig1D,
    LaunchContractError, PinnedHostBuffer, PreparedLaunch, SyncPolicy,
};
pub use error::{GpuError, GpuErrorCode, GpuResult};
pub use graph::CudaGraph;

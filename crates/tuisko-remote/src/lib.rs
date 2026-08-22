//! Runs prebuilt qualification and benchmark executables on ephemeral RunPod GPU pods.

mod error;
mod gpu;
mod key;
mod run;
mod sentry;
mod ssh;
mod v2;

pub use error::RemoteError;
pub use gpu::GpuTarget;
pub use run::{
    BenchmarkOptions, ProbeOptions, QualificationOptions, check, check_credentials, run_benchmark,
    run_probe, run_qualification, sweep_stale,
};
pub use sentry::{mark_keep, run_sentry, spawn_sentry};

/// Default image for remote qualification pods.
pub const DEFAULT_IMAGE: &str = v2::DEFAULT_IMAGE;

/// The result type for remote gate operations.
pub type RemoteResult<T> = Result<T, RemoteError>;

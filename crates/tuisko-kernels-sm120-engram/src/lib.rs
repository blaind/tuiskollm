//! Exact Qwen3.8-Flash-Next PLE operators for SM120.

mod engram;
mod state_snapshot;

pub use engram::{
    Qwen38FlashNextEngramOp, Qwen38FlashNextEngramSources, Qwen38FlashNextEngramWorkspace,
};
pub use state_snapshot::Qwen38FlashNextPleStateSnapshotOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    engram::engram_ptx_names()
        .into_iter()
        .chain(state_snapshot::ple_state_snapshot_ptx_names())
        .collect()
}

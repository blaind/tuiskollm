//! Gated-delta-network operators for SM120.

mod device;
mod prepare;
mod qwen35_prepare;
mod qwen35_recurrence;
mod recurrence;
mod state_snapshot;

pub use prepare::GdnPrepareOp;
pub use qwen35_prepare::{Qwen35GdnPrepareOp, Qwen36GdnPrepareOp};
pub use qwen35_recurrence::{Qwen35GdnRecurrenceOp, Qwen36GdnRecurrenceOp};
pub use recurrence::GdnRecurrenceOp;
pub use state_snapshot::GdnStateSnapshotOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    prepare::gdn_prepare_ptx_names()
        .into_iter()
        .chain(recurrence::gdn_recurrence_ptx_names())
        .chain([state_snapshot::gdn_state_snapshot_ptx_name()])
        .chain(qwen35_prepare::qwen35_gdn_prepare_ptx_names())
        .chain(qwen35_recurrence::qwen35_gdn_recurrence_ptx_names())
        .collect()
}

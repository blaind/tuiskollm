//! Reusable mechanics of the standard qualification lifecycle.
//!
//! These helpers carry machinery only: captured-graph replay, address stability, the post-warmup
//! device-heap gate, sentinel and immutability scans, and layout byte accounting. Acceptance
//! thresholds, expected values, route sets, sentinel bytes, repetition counts, and failure
//! messages stay in each suite as parameters — see `docs/architecture-refactoring.md` Part V §3.F
//! and §3.G. A suite that deviates from the lifecycle keeps its bespoke form.

pub(crate) mod graph_replay;
pub(crate) mod immutable_sentinel;
#[cfg(feature = "engine")]
pub(crate) mod layout_audit;

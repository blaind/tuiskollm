//! Reusable mechanics of the standard qualification and benchmark lifecycles.
//!
//! These helpers carry machinery only: captured-graph replay, address stability, the post-warmup
//! device-heap gate, sentinel and immutability scans, layout byte accounting, and the paired
//! benchmark session. Acceptance thresholds, expected values, route sets, sentinel bytes,
//! repetition counts, metric names, suite identity, and refusal texts stay in each suite because
//! they are measurement identity. A suite that deviates from a lifecycle keeps its bespoke form.

#[cfg(feature = "device")]
pub(crate) mod benchmark_session;
pub(crate) mod graph_replay;
pub(crate) mod immutable_sentinel;
#[cfg(feature = "engine")]
pub(crate) mod layout_audit;

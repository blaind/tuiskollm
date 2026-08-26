//! Reusable mechanics of the standard qualification and benchmark lifecycles.
//!
//! These helpers carry machinery only: captured-graph replay, address stability, the post-warmup
//! device-heap gate, sentinel and immutability scans, layout byte accounting, and the paired
//! benchmark session. Acceptance thresholds, expected values, route sets, sentinel bytes,
//! warmup and repetition counts, metric route names, suite identity, and refusal texts stay in
//! each suite and reach the harness as bound parameters — see `docs/architecture-refactoring.md`
//! Part V §3.B, §3.F and §3.G, and `AGENTS.md`, which treat those as measurement identity. A
//! suite that deviates from a lifecycle keeps its bespoke form rather than being bent to fit.

#[cfg(feature = "device")]
pub(crate) mod benchmark_session;
pub(crate) mod graph_replay;
pub(crate) mod immutable_sentinel;
#[cfg(feature = "engine")]
pub(crate) mod layout_audit;

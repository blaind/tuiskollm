//! Centralized independent reference oracles.
//!
//! Every item here is a value-identical extraction of per-suite host oracles: the numerical
//! semantics, accumulation order, and represented-value handling are transcribed, never
//! rewritten. The module depends on `std` alone so it stays structurally independent of the
//! device implementations it checks.

pub(crate) mod attention;
pub(crate) mod codecs;
pub(crate) mod norm;

#[cfg(test)]
mod diff_tests;

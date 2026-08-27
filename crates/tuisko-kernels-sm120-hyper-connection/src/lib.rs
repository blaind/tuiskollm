//! Exact Qwen3.8-Flash-Next hyper-connection (gated-residual) operators for SM120.

mod hyper_connection;

pub use hyper_connection::Qwen38FlashNextHyperConnectionOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    hyper_connection::hyper_connection_ptx_names()
}

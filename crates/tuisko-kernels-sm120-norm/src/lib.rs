//! Exact-batch zero-centered RMSNorm operators for SM120.

mod residual_norm;

pub use residual_norm::{Qwen35ResidualNormOp, Qwen36ResidualNormOp, ResidualNormOp};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    residual_norm::residual_norm_ptx_names()
        .into_iter()
        .chain(residual_norm::qwen35_residual_norm_ptx_names())
        .chain(residual_norm::qwen36_residual_norm_ptx_names())
        .collect()
}

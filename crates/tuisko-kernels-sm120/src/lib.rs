//! Exact-target SM120 operator kernels and their prepared host launchers.

mod residual_norm;

pub use residual_norm::{ResidualNormOp, residual_norm_ptx_names};

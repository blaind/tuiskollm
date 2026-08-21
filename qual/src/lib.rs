//! Independent operator qualification for the exact SM120 target.

mod residual_norm;
pub use residual_norm::{
    ResidualNormQualification, ResidualNormQualificationError, qualify_residual_norm,
};

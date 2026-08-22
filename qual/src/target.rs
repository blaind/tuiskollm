#[cfg(feature = "sm86-residual")]
pub(crate) use tuisko_kernels_sm86::ResidualNormOp;
#[cfg(feature = "sm89-residual")]
pub(crate) use tuisko_kernels_sm89::ResidualNormOp;
#[cfg(feature = "device")]
pub(crate) use tuisko_kernels_sm120::ResidualNormOp;

#[cfg(feature = "device")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY: (i32, i32) = (12, 0);
#[cfg(feature = "sm89-residual")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY: (i32, i32) = (8, 9);
#[cfg(feature = "sm86-residual")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY: (i32, i32) = (8, 6);

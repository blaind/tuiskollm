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

#[cfg(feature = "device")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY_TEXT: &str = "12.0";
#[cfg(feature = "sm89-residual")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY_TEXT: &str = "8.9";
#[cfg(feature = "sm86-residual")]
pub(crate) const EXPECTED_COMPUTE_CAPABILITY_TEXT: &str = "8.6";

#[cfg(feature = "device")]
pub(crate) const EXPECTED_DEVICE_NAME: &str = "NVIDIA GeForce RTX 5090";
#[cfg(feature = "sm89-residual")]
pub(crate) const EXPECTED_DEVICE_NAME: &str = "NVIDIA GeForce RTX 4090";
#[cfg(feature = "sm86-residual")]
pub(crate) const EXPECTED_DEVICE_NAME: &str = "NVIDIA GeForce RTX 3090";

#[cfg(feature = "device")]
pub(crate) const CLOCK_LOCK_COMMAND: Option<&str> = Some(
    "sudo nvidia-smi -i 0 --lock-gpu-clocks=2200,2200 && sudo nvidia-smi -i 0 --lock-memory-clocks=14001,14001",
);
#[cfg(any(feature = "sm89-residual", feature = "sm86-residual"))]
pub(crate) const CLOCK_LOCK_COMMAND: Option<&str> = None;

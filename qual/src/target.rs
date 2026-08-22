#[cfg(feature = "sm86-residual")]
pub(crate) use tuisko_kernels_sm86::ResidualNormOp;
#[cfg(feature = "sm89")]
pub(crate) use tuisko_kernels_sm89::{Nvfp4SwiGluOp, ResidualNormOp};
#[cfg(feature = "device")]
pub(crate) use tuisko_kernels_sm120::ResidualNormOp;

#[cfg(feature = "device")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm120;
#[cfg(feature = "sm89")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm89;
#[cfg(feature = "sm86-residual")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm86;

pub(crate) const EXPECTED_COMPUTE_CAPABILITY: (i32, i32) = TARGET_PROFILE.compute_capability();
pub(crate) const EXPECTED_COMPUTE_CAPABILITY_TEXT: &str = TARGET_PROFILE.compute_capability_text();
pub(crate) const EXPECTED_DEVICE_NAME: &str = TARGET_PROFILE.device_name();

pub(crate) const CLOCK_LOCK_COMMAND: Option<&str> = match TARGET_PROFILE {
    tuisko_targets::TargetProfile::Sm120 => Some(
        "sudo nvidia-smi -i 0 --lock-gpu-clocks=2200,2200 && sudo nvidia-smi -i 0 --lock-memory-clocks=14001,14001",
    ),
    tuisko_targets::TargetProfile::Sm89 | tuisko_targets::TargetProfile::Sm86 => None,
};

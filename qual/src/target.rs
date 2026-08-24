#[cfg(feature = "sm86")]
pub(crate) use tuisko_kernels_sm86::{Nvfp4SwiGluOp, ResidualNormOp};
#[cfg(feature = "sm89")]
pub(crate) use tuisko_kernels_sm89::{
    FullAttentionQkvOp, Nvfp4DownOp, Nvfp4SwiGluOp, ResidualNormOp,
};
#[cfg(feature = "device")]
pub(crate) use tuisko_kernels_sm120::{
    FullAttentionQkvOp, MtpBf16AttentionOutputOp, MtpBf16FusionOp, MtpBf16MlpOp,
    MtpBf16QkPrepareOp, MtpBf16QkvOp, Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4DownOp,
    Qwen35Bf16LmHeadOp, Qwen35GdnPrepareOp, Qwen35GdnRecurrenceOp, Qwen35Nvfp4GdnInputOp,
    Qwen35Nvfp4GdnOutputOp,
    Qwen35Nvfp4QkvOp, Qwen35Nvfp4SwiGluOp, Qwen35ResidualNormOp, Qwen36Fp8QkvOp, Qwen36GdnInputOp,
    Qwen36GdnOutputOp, Qwen36MoeExpertsOp, Qwen36MoeRouterOp, Qwen36ResidualNormOp, ResidualNormOp,
};

#[cfg(feature = "device")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm120;
#[cfg(feature = "sm89")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm89;
#[cfg(feature = "sm86")]
const TARGET_PROFILE: tuisko_targets::TargetProfile = tuisko_targets::TargetProfile::Sm86;

pub(crate) const EXPECTED_COMPUTE_CAPABILITY: (i32, i32) = TARGET_PROFILE.compute_capability();
pub(crate) const EXPECTED_COMPUTE_CAPABILITY_TEXT: &str = TARGET_PROFILE.compute_capability_text();
pub(crate) const EXPECTED_DEVICE_NAME: &str = TARGET_PROFILE.device_name();

pub(crate) const MAX_SM_CLOCK_SPREAD_MHZ: u32 = match TARGET_PROFILE {
    // The RTX 5090 moves between 2,160 and 2,197 MHz under measured light loads.
    tuisko_targets::TargetProfile::Sm120 => 50,
    tuisko_targets::TargetProfile::Sm89 | tuisko_targets::TargetProfile::Sm86 => 30,
};
pub(crate) const MAX_MEMORY_CLOCK_SPREAD_MHZ: u32 = match TARGET_PROFILE {
    // The RTX 5090 moves between 14,001 and 13,801 MHz under measured light loads.
    tuisko_targets::TargetProfile::Sm120 => 250,
    tuisko_targets::TargetProfile::Sm89 | tuisko_targets::TargetProfile::Sm86 => 100,
};

pub(crate) const CLOCK_LOCK_COMMAND: Option<&str> = match TARGET_PROFILE {
    tuisko_targets::TargetProfile::Sm120 => Some(
        "sudo nvidia-smi -i 0 --lock-gpu-clocks=2200,2200 && sudo nvidia-smi -i 0 --lock-memory-clocks=14001,14001",
    ),
    tuisko_targets::TargetProfile::Sm89 | tuisko_targets::TargetProfile::Sm86 => None,
};

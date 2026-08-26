//! Build and qualification metadata layered on shared GPU identities.

pub(crate) use tuisko_targets::TargetProfile as GpuTarget;

/// Repository-specific build metadata for one shared target profile.
pub(crate) trait BuildTargetProfile {
    /// Cargo package that owns this target's kernels. Only the remote gate
    /// path names it; the device build takes `device_codegen_crates`.
    #[cfg_attr(not(any(feature = "remote", test)), allow(dead_code))]
    fn kernel_crate(self) -> &'static str;
    fn device_codegen_crates(self) -> &'static str;
    fn qualification_feature(self) -> &'static str;
    fn oxide_test_target(self) -> &'static str;
    fn oxide_build_target(self) -> &'static str;
    /// Path of this target's single PTX module, when it has one.
    ///
    /// SM120 splits its device code across per-family crates, so it emits one
    /// module per family instead and has no single path.
    fn ptx_path(self) -> Option<&'static str>;
    fn residual_resource_baseline(self) -> &'static str;
    fn nvfp4_swiglu_resource_baseline(self) -> Option<&'static str>;
    fn nvfp4_down_resource_baseline(self) -> Option<&'static str>;
    fn fp8_qkv_resource_baseline(self) -> Option<&'static str>;
}

impl BuildTargetProfile for GpuTarget {
    fn kernel_crate(self) -> &'static str {
        match self {
            Self::Sm120 => "tuisko-kernels-sm120",
            Self::Sm89 => "tuisko-kernels-sm89",
            Self::Sm86 => "tuisko-kernels-sm86",
        }
    }

    /// Owners cuda-oxide compiles device code for, as one comma-separated list.
    ///
    /// SM120 splits its kernels across per-family crates, so its owner list is
    /// not its facade package name.
    fn device_codegen_crates(self) -> &'static str {
        match self {
            Self::Sm120 => crate::SM120_DEVICE_CODEGEN_CRATES,
            Self::Sm89 => "tuisko-kernels-sm89",
            Self::Sm86 => "tuisko-kernels-sm86",
        }
    }

    fn qualification_feature(self) -> &'static str {
        match self {
            Self::Sm120 => "device",
            Self::Sm89 => "sm89",
            Self::Sm86 => "sm86",
        }
    }

    fn oxide_test_target(self) -> &'static str {
        match self {
            Self::Sm120 => "target/cuda-oxide-test",
            Self::Sm89 => "target/cuda-oxide-test-sm89",
            Self::Sm86 => "target/cuda-oxide-test-sm86",
        }
    }

    fn oxide_build_target(self) -> &'static str {
        match self {
            Self::Sm120 => "target/cuda-oxide-build-sm120",
            Self::Sm89 => "target/cuda-oxide-build-sm89",
            Self::Sm86 => "target/cuda-oxide-build-sm86",
        }
    }

    fn ptx_path(self) -> Option<&'static str> {
        match self {
            Self::Sm120 => None,
            Self::Sm89 => Some("target/cuda/tuisko_kernels_sm89.ptx"),
            Self::Sm86 => Some("target/cuda/tuisko_kernels_sm86.ptx"),
        }
    }

    fn residual_resource_baseline(self) -> &'static str {
        match self {
            Self::Sm120 => "qual/baselines/residual-norm-sm120.txt",
            Self::Sm89 => "qual/baselines/residual-norm-sm89.txt",
            Self::Sm86 => "qual/baselines/residual-norm-sm86.txt",
        }
    }

    fn nvfp4_swiglu_resource_baseline(self) -> Option<&'static str> {
        match self {
            Self::Sm120 => Some("qual/baselines/nvfp4-swiglu-sm120.txt"),
            Self::Sm89 => Some("qual/baselines/nvfp4-swiglu-sm89.txt"),
            Self::Sm86 => Some("qual/baselines/nvfp4-swiglu-sm86.txt"),
        }
    }

    fn nvfp4_down_resource_baseline(self) -> Option<&'static str> {
        match self {
            Self::Sm120 => Some("qual/baselines/nvfp4-down-sm120.txt"),
            Self::Sm89 => Some("qual/baselines/nvfp4-down-sm89.txt"),
            Self::Sm86 => None,
        }
    }

    fn fp8_qkv_resource_baseline(self) -> Option<&'static str> {
        match self {
            Self::Sm120 => Some("qual/baselines/fp8-qkv-sm120.txt"),
            Self::Sm89 => Some("qual/baselines/fp8-qkv-sm89.txt"),
            Self::Sm86 => None,
        }
    }
}

#[cfg(feature = "remote")]
pub(crate) const fn has_full_kernel_inventory(target: GpuTarget) -> bool {
    matches!(target, GpuTarget::Sm120)
}

#[cfg(test)]
mod tests {
    use super::{BuildTargetProfile, GpuTarget};

    #[test]
    fn build_target_table_is_exact_and_complete() {
        assert_eq!(
            GpuTarget::ALL.map(BuildTargetProfile::device_codegen_crates),
            [
                crate::SM120_DEVICE_CODEGEN_CRATES,
                "tuisko-kernels-sm89",
                "tuisko-kernels-sm86",
            ]
        );
        let rows = GpuTarget::ALL.map(|target| {
            (
                target,
                target.kernel_crate(),
                target.qualification_feature(),
                target.ptx_path(),
                target.residual_resource_baseline(),
                target.oxide_build_target(),
            )
        });

        assert_eq!(
            rows,
            [
                (
                    GpuTarget::Sm120,
                    "tuisko-kernels-sm120",
                    "device",
                    None,
                    "qual/baselines/residual-norm-sm120.txt",
                    "target/cuda-oxide-build-sm120",
                ),
                (
                    GpuTarget::Sm89,
                    "tuisko-kernels-sm89",
                    "sm89",
                    Some("target/cuda/tuisko_kernels_sm89.ptx"),
                    "qual/baselines/residual-norm-sm89.txt",
                    "target/cuda-oxide-build-sm89",
                ),
                (
                    GpuTarget::Sm86,
                    "tuisko-kernels-sm86",
                    "sm86",
                    Some("target/cuda/tuisko_kernels_sm86.ptx"),
                    "qual/baselines/residual-norm-sm86.txt",
                    "target/cuda-oxide-build-sm86",
                ),
            ]
        );

        assert_eq!(
            GpuTarget::ALL.map(BuildTargetProfile::nvfp4_swiglu_resource_baseline),
            [
                Some("qual/baselines/nvfp4-swiglu-sm120.txt"),
                Some("qual/baselines/nvfp4-swiglu-sm89.txt"),
                Some("qual/baselines/nvfp4-swiglu-sm86.txt"),
            ]
        );
        assert_eq!(
            GpuTarget::ALL.map(BuildTargetProfile::nvfp4_down_resource_baseline),
            [
                Some("qual/baselines/nvfp4-down-sm120.txt"),
                Some("qual/baselines/nvfp4-down-sm89.txt"),
                None,
            ]
        );
        assert_eq!(
            GpuTarget::ALL.map(BuildTargetProfile::fp8_qkv_resource_baseline),
            [
                Some("qual/baselines/fp8-qkv-sm120.txt"),
                Some("qual/baselines/fp8-qkv-sm89.txt"),
                None,
            ]
        );
    }
}

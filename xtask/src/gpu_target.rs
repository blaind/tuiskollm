//! GPU targets owned by the build and qualification driver.

use std::error::Error;

/// A concrete GPU and code-generation target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuTarget {
    /// Blackwell GeForce target.
    Rtx5090,
    /// Ada GeForce target.
    Rtx4090,
    /// Ampere GeForce target.
    Rtx3090,
}

impl GpuTarget {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 3] = [Self::Rtx5090, Self::Rtx4090, Self::Rtx3090];

    pub(crate) fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "5090" => Ok(Self::Rtx5090),
            "4090" => Ok(Self::Rtx4090),
            "3090" => Ok(Self::Rtx3090),
            _ => Err(format!("unknown GPU target `{value}`; expected 5090, 4090, or 3090").into()),
        }
    }

    pub(crate) const fn key(self) -> &'static str {
        match self {
            Self::Rtx5090 => "5090",
            Self::Rtx4090 => "4090",
            Self::Rtx3090 => "3090",
        }
    }

    pub(crate) const fn device_name(self) -> &'static str {
        match self {
            Self::Rtx5090 => "NVIDIA GeForce RTX 5090",
            Self::Rtx4090 => "NVIDIA GeForce RTX 4090",
            Self::Rtx3090 => "NVIDIA GeForce RTX 3090",
        }
    }

    pub(crate) const fn compute_capability(self) -> &'static str {
        match self {
            Self::Rtx5090 => "12.0",
            Self::Rtx4090 => "8.9",
            Self::Rtx3090 => "8.6",
        }
    }

    pub(crate) const fn oxide_arch(self) -> &'static str {
        match self {
            Self::Rtx5090 => "sm_120a",
            Self::Rtx4090 => "sm_89",
            Self::Rtx3090 => "sm_86",
        }
    }

    pub(crate) const fn kernel_crate(self) -> &'static str {
        match self {
            Self::Rtx5090 => "tuisko-kernels-sm120",
            Self::Rtx4090 => "tuisko-kernels-sm89",
            Self::Rtx3090 => "tuisko-kernels-sm86",
        }
    }

    pub(crate) const fn qualification_feature(self) -> &'static str {
        match self {
            Self::Rtx5090 => "device",
            Self::Rtx4090 => "sm89-residual",
            Self::Rtx3090 => "sm86-residual",
        }
    }

    #[cfg_attr(not(any(feature = "remote", test)), allow(dead_code))]
    pub(crate) const fn has_full_kernel_inventory(self) -> bool {
        matches!(self, Self::Rtx5090)
    }

    pub(crate) const fn oxide_test_target(self) -> &'static str {
        match self {
            Self::Rtx5090 => "target/cuda-oxide-test",
            Self::Rtx4090 => "target/cuda-oxide-test-sm89",
            Self::Rtx3090 => "target/cuda-oxide-test-sm86",
        }
    }

    pub(crate) const fn oxide_build_target(self) -> &'static str {
        match self {
            Self::Rtx5090 => "target/cuda-oxide-build-sm120",
            Self::Rtx4090 => "target/cuda-oxide-build-sm89",
            Self::Rtx3090 => "target/cuda-oxide-build-sm86",
        }
    }

    pub(crate) const fn ptx_path(self) -> &'static str {
        match self {
            Self::Rtx5090 => "target/cuda/tuisko_kernels_sm120.ptx",
            Self::Rtx4090 => "target/cuda/tuisko_kernels_sm89.ptx",
            Self::Rtx3090 => "target/cuda/tuisko_kernels_sm86.ptx",
        }
    }

    pub(crate) const fn residual_resource_baseline(self) -> &'static str {
        match self {
            Self::Rtx5090 => "qual/baselines/residual-norm-sm120.txt",
            Self::Rtx4090 => "qual/baselines/residual-norm-sm89.txt",
            Self::Rtx3090 => "qual/baselines/residual-norm-sm86.txt",
        }
    }

    #[cfg(feature = "remote")]
    pub(crate) const fn remote_gpu(self) -> tuisko_remote::GpuTarget {
        tuisko_remote::GpuTarget::new(self.device_name(), self.compute_capability())
    }
}

#[cfg(test)]
mod tests {
    use super::GpuTarget;

    #[test]
    fn target_table_is_exact_and_complete() {
        let rows = GpuTarget::ALL.map(|target| {
            (
                target.key(),
                target.device_name(),
                target.compute_capability(),
                target.oxide_arch(),
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
                    "5090",
                    "NVIDIA GeForce RTX 5090",
                    "12.0",
                    "sm_120a",
                    "tuisko-kernels-sm120",
                    "device",
                    "target/cuda/tuisko_kernels_sm120.ptx",
                    "qual/baselines/residual-norm-sm120.txt",
                    "target/cuda-oxide-build-sm120",
                ),
                (
                    "4090",
                    "NVIDIA GeForce RTX 4090",
                    "8.9",
                    "sm_89",
                    "tuisko-kernels-sm89",
                    "sm89-residual",
                    "target/cuda/tuisko_kernels_sm89.ptx",
                    "qual/baselines/residual-norm-sm89.txt",
                    "target/cuda-oxide-build-sm89",
                ),
                (
                    "3090",
                    "NVIDIA GeForce RTX 3090",
                    "8.6",
                    "sm_86",
                    "tuisko-kernels-sm86",
                    "sm86-residual",
                    "target/cuda/tuisko_kernels_sm86.ptx",
                    "qual/baselines/residual-norm-sm86.txt",
                    "target/cuda-oxide-build-sm86",
                ),
            ]
        );
    }

    #[test]
    fn target_parser_rejects_untracked_hardware() {
        assert_eq!(GpuTarget::parse("4090").unwrap(), GpuTarget::Rtx4090);
        assert!(GpuTarget::parse("A100").is_err());
    }
}

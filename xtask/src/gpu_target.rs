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

    pub(crate) const fn kernel_crate(self) -> Option<&'static str> {
        match self {
            Self::Rtx5090 => Some("tuisko-kernels-sm120"),
            Self::Rtx4090 | Self::Rtx3090 => None,
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
                    Some("tuisko-kernels-sm120"),
                ),
                ("4090", "NVIDIA GeForce RTX 4090", "8.9", "sm_89", None,),
                ("3090", "NVIDIA GeForce RTX 3090", "8.6", "sm_86", None,),
            ]
        );
    }

    #[test]
    fn target_parser_rejects_untracked_hardware() {
        assert_eq!(GpuTarget::parse("4090").unwrap(), GpuTarget::Rtx4090);
        assert!(GpuTarget::parse("A100").is_err());
    }
}

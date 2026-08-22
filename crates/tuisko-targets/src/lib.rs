//! Exact GPU target identities shared by build, qualification, and startup selection.

/// One compiled GPU architecture and its admitted physical product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetProfile {
    /// Blackwell GeForce RTX 5090.
    Sm120,
    /// Ada GeForce RTX 4090.
    Sm89,
    /// Ampere GeForce RTX 3090.
    Sm86,
}

impl TargetProfile {
    /// Every target tracked by the build and qualification inventory.
    pub const ALL: [Self; 3] = [Self::Sm120, Self::Sm89, Self::Sm86];

    /// Parses the stable CLI product key.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "5090" => Ok(Self::Sm120),
            "4090" => Ok(Self::Sm89),
            "3090" => Ok(Self::Sm86),
            _ => Err(format!(
                "unknown GPU target `{value}`; expected 5090, 4090, or 3090"
            )),
        }
    }

    /// Selects an exact target from the CUDA device identity.
    pub fn from_device(name: &str, capability: (i32, i32)) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| {
            profile.device_name() == name && profile.compute_capability() == capability
        })
    }

    /// Stable product key used by commands and report paths.
    pub const fn key(self) -> &'static str {
        match self {
            Self::Sm120 => "5090",
            Self::Sm89 => "4090",
            Self::Sm86 => "3090",
        }
    }

    /// Exact CUDA-visible device name admitted by this target.
    pub const fn device_name(self) -> &'static str {
        match self {
            Self::Sm120 => "NVIDIA GeForce RTX 5090",
            Self::Sm89 => "NVIDIA GeForce RTX 4090",
            Self::Sm86 => "NVIDIA GeForce RTX 3090",
        }
    }

    /// Exact CUDA compute capability admitted by this target.
    pub const fn compute_capability(self) -> (i32, i32) {
        match self {
            Self::Sm120 => (12, 0),
            Self::Sm89 => (8, 9),
            Self::Sm86 => (8, 6),
        }
    }

    /// Display spelling of the exact compute capability.
    pub const fn compute_capability_text(self) -> &'static str {
        match self {
            Self::Sm120 => "12.0",
            Self::Sm89 => "8.9",
            Self::Sm86 => "8.6",
        }
    }

    /// cuda-oxide architecture spelling for this compiled artifact.
    pub const fn oxide_arch(self) -> &'static str {
        match self {
            Self::Sm120 => "sm_120a",
            Self::Sm89 => "sm_89",
            Self::Sm86 => "sm_86",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TargetProfile;

    #[test]
    fn exact_device_table_round_trips() {
        for profile in TargetProfile::ALL {
            assert_eq!(TargetProfile::parse(profile.key()).unwrap(), profile);
            assert_eq!(
                TargetProfile::from_device(profile.device_name(), profile.compute_capability()),
                Some(profile)
            );
        }
    }

    #[test]
    fn capability_without_the_exact_product_name_is_not_admitted() {
        assert_eq!(TargetProfile::from_device("NVIDIA L40", (8, 9)), None);
        assert!(TargetProfile::parse("A100").is_err());
    }
}

use serde::{Deserialize, Deserializer, de};
use std::fmt::{self, Display, Formatter};

/// Element representation stored by the admitted checkpoint shards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// Two-byte bfloat16 source words.
    Bf16,
    /// Four-byte IEEE 754 source values.
    F32,
    /// One-byte FP8 E4M3 source codes.
    Fp8E4M3,
    /// Raw bytes, including packed E2M1 NVFP4 weights.
    U8,
}

impl DType {
    /// Number of source bytes occupied by one stored element.
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::Fp8E4M3 | Self::U8 => 1,
        }
    }

    /// Exact dtype spelling used in the safetensors header.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::Fp8E4M3 => "F8_E4M3",
            Self::U8 => "U8",
        }
    }
}

impl Display for DType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;

        match name.as_str() {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            "F8_E4M3" => Ok(Self::Fp8E4M3),
            "U8" => Ok(Self::U8),
            _ => Err(de::Error::custom(format_args!(
                "unsupported checkpoint dtype `{name}`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn parses_exact_source_dtype_names() {
        for (name, expected, width) in [
            ("BF16", DType::Bf16, 2),
            ("F32", DType::F32, 4),
            ("F8_E4M3", DType::Fp8E4M3, 1),
            ("U8", DType::U8, 1),
        ] {
            let dtype: DType = serde_json::from_str(&format!("\"{name}\"")).unwrap();

            assert_eq!(dtype, expected);
            assert_eq!(dtype.as_str(), name);
            assert_eq!(dtype.byte_width(), width);
        }
    }

    #[test]
    fn rejects_dtype_outside_checkpoint_contract() {
        let error = serde_json::from_str::<DType>("\"F16\"")
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("unsupported checkpoint dtype `F16`"));
    }
}

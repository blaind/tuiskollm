mod bindings;
mod config;
mod dtype;
mod error;
mod inventory;
mod safetensors;
mod views;

pub use bindings::{Nvfp4DownBindings, Nvfp4GateUpBindings, TextEndpointBindings};
pub use config::validate_config;
pub use dtype::DType;
pub use error::{CheckpointError, CheckpointErrorCode, CheckpointResult};
pub use inventory::CheckpointSnapshot;
pub use safetensors::{SafeTensorFile, TensorView};
pub use views::{Bf16View, F32View, Fp8E4M3View, U8View};

/// Compile-time identity and geometry of an admitted model target.
pub trait Arch: Copy + 'static {
    /// Hugging Face repository identifier.
    const MODEL_ID: &'static str;
    /// Immutable Hugging Face snapshot revision.
    const REVISION: &'static str;
    /// Residual stream width.
    const HIDDEN: usize;
    /// Dense MLP intermediate width.
    const INTERMEDIATE: usize;
    /// Token vocabulary size.
    const VOCAB: usize;
    /// Number of decoder layers.
    const LAYERS: usize;
    /// Distance between full-attention layers.
    const FULL_ATTENTION_INTERVAL: usize;
    /// Number of full-attention query heads.
    const NUM_ATTENTION_HEADS: usize;
    /// Number of full-attention key/value heads.
    const NUM_KV_HEADS: usize;
    /// Width of each full-attention head.
    const HEAD_DIM: usize;
    /// Number of GDN query/key heads.
    const LINEAR_KEY_HEADS: usize;
    /// Number of GDN value heads.
    const LINEAR_VALUE_HEADS: usize;
    /// Width of each GDN head.
    const LINEAR_HEAD_DIM: usize;
    /// Width of the GDN causal convolution.
    const LINEAR_CONV_KERNEL_DIM: usize;

    /// Rows in the fused full-attention query and gate plane.
    const ATTENTION_QUERY_ROWS: usize = 2 * Self::NUM_ATTENTION_HEADS * Self::HEAD_DIM;
    /// Rows in one full-attention key or value plane.
    const ATTENTION_KV_ROWS: usize = Self::NUM_KV_HEADS * Self::HEAD_DIM;
    /// Rows in the fused full-attention query/gate, key, and value projection.
    const ATTENTION_QKV_ROWS: usize = Self::ATTENTION_QUERY_ROWS + 2 * Self::ATTENTION_KV_ROWS;
    /// Rows in one GDN query or key plane.
    const GDN_QK_ROWS: usize = Self::LINEAR_KEY_HEADS * Self::LINEAR_HEAD_DIM;
    /// Rows in one GDN value or Z plane.
    const GDN_VALUE_ROWS: usize = Self::LINEAR_VALUE_HEADS * Self::LINEAR_HEAD_DIM;
    /// Rows in the fused GDN query, key, and value projection.
    const GDN_QKV_ROWS: usize = 2 * Self::GDN_QK_ROWS + Self::GDN_VALUE_ROWS;
    /// Rows in the fused GDN query, key, value, and Z projection.
    const GDN_INPUT_ROWS: usize = Self::GDN_QKV_ROWS + Self::GDN_VALUE_ROWS;
    /// Per-value-head GDN control width.
    const GDN_CONTROL_ROWS: usize = Self::LINEAR_VALUE_HEADS;
}

/// Geometry and pinned identity of `unsloth/Qwen3.8-27B-NVFP4`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38_27B;

impl Arch for Qwen38_27B {
    const MODEL_ID: &'static str = "unsloth/Qwen3.8-27B-NVFP4";
    const REVISION: &'static str = "16b6615af3548b88e2d8e382457bc705b00479cf";
    const HIDDEN: usize = 5_120;
    const INTERMEDIATE: usize = 17_408;
    const VOCAB: usize = 248_320;
    const LAYERS: usize = 64;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const NUM_ATTENTION_HEADS: usize = 24;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const LINEAR_KEY_HEADS: usize = 16;
    const LINEAR_VALUE_HEADS: usize = 48;
    const LINEAR_HEAD_DIM: usize = 128;
    const LINEAR_CONV_KERNEL_DIM: usize = 4;
}

#[cfg(test)]
mod tests {
    use super::{Arch, Qwen38_27B};

    #[test]
    fn qwen38_profile_matches_checkpoint_geometry() {
        type A = Qwen38_27B;

        for (field, actual, expected) in [
            ("hidden", A::HIDDEN, 5_120),
            ("intermediate", A::INTERMEDIATE, 17_408),
            ("vocab", A::VOCAB, 248_320),
            ("layers", A::LAYERS, 64),
            ("full_attention_interval", A::FULL_ATTENTION_INTERVAL, 4),
            ("num_attention_heads", A::NUM_ATTENTION_HEADS, 24),
            ("num_kv_heads", A::NUM_KV_HEADS, 4),
            ("head_dim", A::HEAD_DIM, 256),
            ("linear_key_heads", A::LINEAR_KEY_HEADS, 16),
            ("linear_value_heads", A::LINEAR_VALUE_HEADS, 48),
            ("linear_head_dim", A::LINEAR_HEAD_DIM, 128),
            ("linear_conv_kernel_dim", A::LINEAR_CONV_KERNEL_DIM, 4),
            ("attention_query_rows", A::ATTENTION_QUERY_ROWS, 12_288),
            ("attention_kv_rows", A::ATTENTION_KV_ROWS, 1_024),
            ("attention_qkv_rows", A::ATTENTION_QKV_ROWS, 14_336),
            ("gdn_qk_rows", A::GDN_QK_ROWS, 2_048),
            ("gdn_value_rows", A::GDN_VALUE_ROWS, 6_144),
            ("gdn_qkv_rows", A::GDN_QKV_ROWS, 10_240),
            ("gdn_input_rows", A::GDN_INPUT_ROWS, 16_384),
            ("gdn_control_rows", A::GDN_CONTROL_ROWS, 48),
        ] {
            assert_eq!(actual, expected, "{field}");
        }
    }

    #[test]
    fn qwen38_profile_matches_checkpoint_identity() {
        assert_eq!(Qwen38_27B::MODEL_ID, "unsloth/Qwen3.8-27B-NVFP4");
        assert_eq!(
            Qwen38_27B::REVISION,
            "16b6615af3548b88e2d8e382457bc705b00479cf"
        );
    }
}

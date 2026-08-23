//! Exact-target checkpoint admission and source-layout ownership.

mod bindings;
mod config;
mod dtype;
mod error;
mod inventory;
mod materialize;
mod safetensors;
mod views;

pub use bindings::{
    DenseFp8DownBindings, DenseFp8GateUpBindings, DenseFp8MlpBindings, FullAttentionPostBindings,
    FullAttentionQkvBindings, GdnBindings, MtpBindings, Nvfp4DownBindings, Nvfp4GateUpBindings,
    Nvfp4MlpBindings, TextEndpointBindings, VisionBindings, VisionBlockBindings,
};
pub use config::validate_config;
pub use dtype::DType;
pub use error::{CheckpointError, CheckpointErrorCode, CheckpointResult};
pub use inventory::CheckpointSnapshot;
pub use materialize::{
    MaterializedFullAttentionQkv, MaterializedMtpQkv, MaterializedNvfp4Down,
    MaterializedNvfp4GateUp,
};
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
    /// Pinned `text_config.rms_norm_eps` used by RMSNorm.
    const RMS_NORM_EPSILON: f32;
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
    /// Number of draft layers in the MTP checkpoint shard.
    const MTP_LAYERS: usize;
    /// Whether MTP owns embeddings separate from the base model.
    const MTP_USES_DEDICATED_EMBEDDINGS: bool;
    /// Number of transformer blocks in the Vision encoder.
    const VISION_DEPTH: usize;
    /// Vision encoder channel width.
    const VISION_HIDDEN: usize;
    /// Vision encoder MLP intermediate width.
    const VISION_INTERMEDIATE: usize;
    /// Number of Vision attention heads.
    const VISION_NUM_HEADS: usize;
    /// Number of learned Vision position embeddings.
    const VISION_POSITIONS: usize;
    /// Width projected from Vision into the text residual stream.
    const VISION_OUTPUT_HIDDEN: usize;
    /// Number of image channels consumed by the patch projection.
    const VISION_INPUT_CHANNELS: usize;
    /// Spatial extent of one Vision patch.
    const VISION_PATCH_SIZE: usize;
    /// Spatial patch-grid reduction performed by the merger.
    const VISION_SPATIAL_MERGE_SIZE: usize;
    /// Frames grouped by one temporal patch.
    const VISION_TEMPORAL_PATCH_SIZE: usize;

    /// Rows in the fused full-attention query and gate plane.
    const ATTENTION_QUERY_ROWS: usize = 2 * Self::NUM_ATTENTION_HEADS * Self::HEAD_DIM;
    /// Width returned by full attention before its output projection.
    const ATTENTION_OUTPUT_COLUMNS: usize = Self::NUM_ATTENTION_HEADS * Self::HEAD_DIM;
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
    const RMS_NORM_EPSILON: f32 = 1.0e-6;
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
    const MTP_LAYERS: usize = 1;
    const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
    const VISION_DEPTH: usize = 27;
    const VISION_HIDDEN: usize = 1_152;
    const VISION_INTERMEDIATE: usize = 4_304;
    const VISION_NUM_HEADS: usize = 16;
    const VISION_POSITIONS: usize = 2_304;
    const VISION_OUTPUT_HIDDEN: usize = 5_120;
    const VISION_INPUT_CHANNELS: usize = 3;
    const VISION_PATCH_SIZE: usize = 16;
    const VISION_SPATIAL_MERGE_SIZE: usize = 2;
    const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
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
            ("mtp_layers", A::MTP_LAYERS, 1),
            ("vision_depth", A::VISION_DEPTH, 27),
            ("vision_hidden", A::VISION_HIDDEN, 1_152),
            ("vision_intermediate", A::VISION_INTERMEDIATE, 4_304),
            ("vision_num_heads", A::VISION_NUM_HEADS, 16),
            ("vision_positions", A::VISION_POSITIONS, 2_304),
            ("vision_output_hidden", A::VISION_OUTPUT_HIDDEN, 5_120),
            ("vision_input_channels", A::VISION_INPUT_CHANNELS, 3),
            ("vision_patch_size", A::VISION_PATCH_SIZE, 16),
            ("vision_spatial_merge_size", A::VISION_SPATIAL_MERGE_SIZE, 2),
            (
                "vision_temporal_patch_size",
                A::VISION_TEMPORAL_PATCH_SIZE,
                2,
            ),
            ("attention_query_rows", A::ATTENTION_QUERY_ROWS, 12_288),
            (
                "attention_output_columns",
                A::ATTENTION_OUTPUT_COLUMNS,
                6_144,
            ),
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

        const {
            assert!(!Qwen38_27B::MTP_USES_DEDICATED_EMBEDDINGS);
        }
        assert_eq!(Qwen38_27B::RMS_NORM_EPSILON, 1.0e-6);
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

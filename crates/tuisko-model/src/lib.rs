//! Exact-target checkpoint admission and source-layout ownership.

mod common;
mod dtype;
mod error;
mod qwen35;
mod qwen36;
mod qwen38;
mod qwen38_flash_next;
mod safetensors;
mod views;

pub use common::inventory::CheckpointSnapshot;
pub use common::materialized::MaterializedMemory;
pub use common::modelopt_codec::{MaterializedModelOptNvfp4Linear, ModelOptNvfp4LinearBindings};
pub use common::mtp::MaterializedMtpQkv;
pub use common::nvfp4::{
    MaterializedNvfp4Down, MaterializedNvfp4GateUp, Nvfp4DownBindings, Nvfp4GateUpBindings,
};
pub use common::routes::NVFP4_MLP_LAYER_END;
pub use common::scale_swizzle::nvfp4_scale_materialization_workers;
pub use common::schema::validate_config;
pub use common::source_binding::SourceLayerBinding;
pub use common::vision::{VisionBindings, VisionBlockBindings};
pub use dtype::DType;
pub use error::{CheckpointError, CheckpointErrorCode, CheckpointResult};
pub use qwen35::bindings::{
    Bf16TextEndpointBindings, ModelOptNvfp4AttentionBindings, ModelOptNvfp4GdnBindings,
    ModelOptNvfp4MlpBindings,
};
pub use qwen35::materialize::{
    MaterializedModelOptNvfp4Attention, MaterializedModelOptNvfp4Gdn, MaterializedModelOptNvfp4Mlp,
};
pub use qwen36::bindings::{
    Qwen36Fp8LinearBindings, Qwen36FullAttentionBindings, Qwen36GdnBindings,
    Qwen36MoeExpertBindings, Qwen36MoeLayerBindings, Qwen36MtpBindings, Qwen36TextEndpointBindings,
};
pub use qwen36::materialize::{
    MaterializedQwen36Fp8Linear, MaterializedQwen36FullAttention, MaterializedQwen36Gdn,
    MaterializedQwen36MoeExperts, MaterializedQwen36MoeLayer, MaterializedQwen36TextEndpoint,
};
pub use qwen38::bindings::{
    DenseFp8DownBindings, DenseFp8GateUpBindings, DenseFp8MlpBindings, FullAttentionPostBindings,
    FullAttentionQkvBindings, GdnBindings, MtpBindings, Nvfp4MlpBindings, TextEndpointBindings,
};
pub use qwen38::materialize::MaterializedFullAttentionQkv;
pub use qwen38_flash_next::bindings::{
    Qwen38FlashNextEngramBindings, Qwen38FlashNextExpertBindings, Qwen38FlashNextGdnBindings,
    Qwen38FlashNextHyperConnectionBindings, Qwen38FlashNextIndexerBindings,
    Qwen38FlashNextLayerHyperConnections, Qwen38FlashNextMoeBindings,
    Qwen38FlashNextSharedExpertBindings, Qwen38FlashNextSparseAttentionBindings,
    Qwen38FlashNextTextEndpointBindings,
};
pub use qwen38_flash_next::engram::{
    Qwen38FlashNextEngramConstantBindings, Qwen38FlashNextEngramHashConstants,
};
pub use qwen38_flash_next::engram_hash::{
    QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN, QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN,
    QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN, Qwen38FlashNextEngramCarry,
    Qwen38FlashNextEngramRowHasher, Qwen38FlashNextEngramTable,
    admit_qwen38_flash_next_engram_token,
};
pub use qwen38_flash_next::materialize::{
    MaterializedQwen38FlashNextEngram, MaterializedQwen38FlashNextExpert,
    MaterializedQwen38FlashNextExpertPool, MaterializedQwen38FlashNextGdn,
    MaterializedQwen38FlashNextHyperConnections, MaterializedQwen38FlashNextMoe,
    MaterializedQwen38FlashNextSparseAttention, MaterializedQwen38FlashNextTextEndpoint,
    Qwen38FlashNextPlaneExtent,
};
pub use safetensors::{SafeTensorFile, TensorView};
pub use views::{Bf16View, F32View, Fp8E4M3View, I64View, U8View};

/// Source checkpoint contract selected before config and inventory admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointContract {
    /// Mixed compressed-tensors FP8/NVFP4 checkpoint.
    CompressedTensors,
    /// ModelOpt NVFP4 checkpoint with static source scales.
    ModelOptNvfp4,
    /// ModelOpt mixed FP8/NVFP4 checkpoint with routed experts.
    ModelOptNvfp4Moe,
    /// ModelOpt NVFP4 checkpoint whose routed experts are sharded per layer and expert block,
    /// with an FP8 engram table beside a BF16 non-routed remainder.
    Qwen38FlashNextModelOptNvfp4,
}

/// Compile-time identity and geometry of an admitted model target.
pub trait Arch: Copy + 'static {
    /// Hugging Face repository identifier.
    const MODEL_ID: &'static str;
    /// Immutable Hugging Face snapshot revision.
    const REVISION: &'static str;
    /// Exact config, quantization, and inventory convention.
    const CHECKPOINT_CONTRACT: CheckpointContract = CheckpointContract::CompressedTensors;
    /// Residual stream width.
    const HIDDEN: usize;
    /// Pinned `text_config.rms_norm_eps` used by RMSNorm.
    const RMS_NORM_EPSILON: f32;
    /// Target feed-forward intermediate width.
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
    const CHECKPOINT_CONTRACT: CheckpointContract = CheckpointContract::CompressedTensors;
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

/// Geometry and pinned identity of `AxionML/Qwen3.5-9B-NVFP4`.
///
/// The profile admits host-side geometry checks only. Device crates remain
/// sealed to their explicitly qualified targets.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen35_9B;

impl Arch for Qwen35_9B {
    const MODEL_ID: &'static str = "AxionML/Qwen3.5-9B-NVFP4";
    const REVISION: &'static str = "97aef92393f126bf649f310cd40861be8dad3279";
    const CHECKPOINT_CONTRACT: CheckpointContract = CheckpointContract::ModelOptNvfp4;
    const HIDDEN: usize = 4_096;
    const RMS_NORM_EPSILON: f32 = 1.0e-6;
    const INTERMEDIATE: usize = 12_288;
    const VOCAB: usize = 248_320;
    const LAYERS: usize = 32;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const NUM_ATTENTION_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const LINEAR_KEY_HEADS: usize = 16;
    const LINEAR_VALUE_HEADS: usize = 32;
    const LINEAR_HEAD_DIM: usize = 128;
    const LINEAR_CONV_KERNEL_DIM: usize = 4;
    const MTP_LAYERS: usize = 1;
    const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
    const VISION_DEPTH: usize = 27;
    const VISION_HIDDEN: usize = 1_152;
    const VISION_INTERMEDIATE: usize = 4_304;
    const VISION_NUM_HEADS: usize = 16;
    const VISION_POSITIONS: usize = 2_304;
    const VISION_OUTPUT_HIDDEN: usize = 4_096;
    const VISION_INPUT_CHANNELS: usize = 3;
    const VISION_PATCH_SIZE: usize = 16;
    const VISION_SPATIAL_MERGE_SIZE: usize = 2;
    const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
}

/// Geometry and pinned identity of `nvidia/Qwen3.6-35B-A3B-NVFP4`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen36Moe35B;

impl Qwen36Moe35B {
    /// Unit cache scale selected by the checkpoint's constant-amax E4M3 cast recipe.
    pub const FP8_CACHE_SCALE: f32 = 1.0;
    /// Maximum text position admitted by the pinned config.
    pub const MAX_POSITION_EMBEDDINGS: usize = 262_144;
    /// Routed experts owned by every decoder layer.
    pub const NUM_EXPERTS: usize = 256;
    /// Experts selected and normalized for every represented token.
    pub const NUM_EXPERTS_PER_TOKEN: usize = 8;
    /// Intermediate width of the always-active shared expert.
    pub const SHARED_EXPERT_INTERMEDIATE: usize = 512;
}

impl Arch for Qwen36Moe35B {
    const MODEL_ID: &'static str = "nvidia/Qwen3.6-35B-A3B-NVFP4";
    const REVISION: &'static str = "491c2f1ea524c639598bf8fa787a93fed5a6fbce";
    const CHECKPOINT_CONTRACT: CheckpointContract = CheckpointContract::ModelOptNvfp4Moe;
    const HIDDEN: usize = 2_048;
    const RMS_NORM_EPSILON: f32 = 1.0e-6;
    const INTERMEDIATE: usize = 512;
    const VOCAB: usize = 248_320;
    const LAYERS: usize = 40;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const NUM_ATTENTION_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const LINEAR_KEY_HEADS: usize = 16;
    const LINEAR_VALUE_HEADS: usize = 32;
    const LINEAR_HEAD_DIM: usize = 128;
    const LINEAR_CONV_KERNEL_DIM: usize = 4;
    const MTP_LAYERS: usize = 1;
    const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
    const VISION_DEPTH: usize = 27;
    const VISION_HIDDEN: usize = 1_152;
    const VISION_INTERMEDIATE: usize = 4_304;
    const VISION_NUM_HEADS: usize = 16;
    const VISION_POSITIONS: usize = 2_304;
    const VISION_OUTPUT_HIDDEN: usize = 2_048;
    const VISION_INPUT_CHANNELS: usize = 3;
    const VISION_PATCH_SIZE: usize = 16;
    const VISION_SPATIAL_MERGE_SIZE: usize = 2;
    const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
}

/// Geometry and pinned identity of `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
///
/// Family-specific geometry stays here instead of widening `Arch` for unrelated targets.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38FlashNext;

impl Qwen38FlashNext {
    /// Routed experts owned by every decoder layer and by the MTP layer.
    pub const NUM_EXPERTS: usize = 512;
    /// Experts selected and renormalized for every represented token.
    pub const NUM_EXPERTS_PER_TOKEN: usize = 10;
    /// Intermediate width of the always-active shared expert.
    pub const SHARED_EXPERT_INTERMEDIATE: usize = 640;
    /// Maximum text position admitted by the pinned config.
    pub const MAX_POSITION_EMBEDDINGS: usize = 262_144;
    /// Fraction of every attention head rotated by MRoPE.
    pub const PARTIAL_ROTARY_FACTOR: f32 = 0.25;
    /// Pinned MRoPE base period.
    pub const ROPE_THETA: u64 = 10_000_000;
    /// Interleaved MRoPE temporal, height, and width section widths.
    pub const MROPE_SECTION: [usize; 3] = [11, 11, 10];

    /// Parallel residual branches carried by the hyper-connection stream.
    pub const HC_COUNT: usize = 4;
    /// Rank of every hyper-connection read gate.
    pub const HC_LOWRANK: usize = 320;
    /// Width of the widened residual stream: one `HIDDEN`-wide branch per `HC_COUNT`.
    pub const HC_WIDTH: usize = Self::HC_COUNT * <Self as Arch>::HIDDEN;

    /// Query heads owned by one sparse-attention indexer.
    pub const INDEXER_HEADS: usize = 4;
    /// Key/value heads owned by one sparse-attention indexer.
    pub const INDEXER_KV_HEADS: usize = 1;
    /// Width of every indexer head.
    pub const INDEXER_HEAD_DIM: usize = 128;
    /// Tokens one query may select through the indexer.
    pub const INDEXER_BUDGET: usize = 2_048;
    /// Tokens pooled into one indexer micro-block.
    pub const INDEXER_COMPRESS_RATIO: usize = 4;
    /// Rows in the fused indexer query and key projection.
    pub const INDEXER_ROWS: usize =
        (Self::INDEXER_HEADS + Self::INDEXER_KV_HEADS) * Self::INDEXER_HEAD_DIM;

    /// EOS token and engram segment boundary.
    pub const EOS_TOKEN_ID: u32 = 248_044;

    /// Zero-based decoder layer carrying the single engram injection.
    ///
    /// The config spells this one-indexed as `ple_layer_ids: [2]`.
    pub const PLE_LAYER: usize = 1;
    /// Width of the engram embedding injected at `PLE_LAYER`.
    pub const PLE_EMBED_DIM: usize = 2_560;
    /// Width of the engram short convolution.
    pub const PLE_CONV_KERNEL: usize = 4;
    /// Longest n-gram hashed by the engram table.
    pub const NGRAM_SIZE: usize = 3;
    /// Tap spacing of the engram short convolution: the n-gram width.
    pub const PLE_CONV_DILATION: usize = Self::NGRAM_SIZE;
    /// Columns of engram convolution history one sequence carries.
    ///
    /// `(kernel - 1) * dilation`, so the four taps land at `t-9, t-6, t-3, t`.
    pub const PLE_CONV_STATE_LEN: usize = (Self::PLE_CONV_KERNEL - 1) * Self::PLE_CONV_DILATION;
    /// Cache states the engram layer owns: GDN conv, PLE conv, then token history.
    pub const LAYER_CONV_STATES: usize = 3;
    /// Cache slot holding the engram convolution history.
    pub const PLE_CONV_STATE_SLOT: usize = 1;
    /// Cache slot holding the two-token engram hash carry.
    pub const PLE_TOKEN_HISTORY_SLOT: usize = 2;
    /// Tokens of hash carry the engram layer keeps: `ngram_size - 1`.
    pub const PLE_CONTEXT_LEN: usize = Self::NGRAM_SIZE - 1;
    /// Magnitude floor for the engram gate's signed square root.
    pub const PLE_GATE_FLOOR: f32 = 1.0e-6;
    /// Independently hashed lookups per n-gram order.
    pub const HEADS_PER_NGRAM: usize = 8;
    /// Engram lookup heads: `HEADS_PER_NGRAM` per hashed n-gram order.
    pub const NGRAM_HEADS: usize = (Self::NGRAM_SIZE - 1) * Self::HEADS_PER_NGRAM;
    /// Width contributed by one engram head.
    pub const NGRAM_HEAD_DIM: usize = Self::PLE_EMBED_DIM / Self::NGRAM_HEADS;
    /// Base vocabulary searched for each engram head's prime modulus.
    pub const NGRAM_VOCAB_BASE: usize = 20_000_000;
    /// Alignment applied to the summed engram vocabulary.
    pub const NGRAM_VOCAB_DIVISOR: usize = 128;
    /// Checkpoint shards the engram table is concatenated from, in shard-index order.
    pub const NGRAM_SHARDS: usize = 128;
    /// Rows held by one engram table shard.
    pub const NGRAM_SHARD_ROWS: usize = 2_500_012;

    /// Activation applied to the GDN output gate.
    pub const GDN_OUTPUT_GATE: &'static str = "sigmoid";
    /// Element representation of the engram table in this NVFP4 release.
    pub const PLE_EMBEDDING_DTYPE: &'static str = "float8_e4m3fn";
}

impl Arch for Qwen38FlashNext {
    const MODEL_ID: &'static str = "RadixArk/Qwen3.8-Flash-Next-NVFP4";
    const REVISION: &'static str = "7b719225242aacd3dbd3f9407468c2ee9a9d2594";
    const CHECKPOINT_CONTRACT: CheckpointContract =
        CheckpointContract::Qwen38FlashNextModelOptNvfp4;
    const HIDDEN: usize = 2_560;
    const RMS_NORM_EPSILON: f32 = 1.0e-6;
    const INTERMEDIATE: usize = 640;
    const VOCAB: usize = 248_320;
    const LAYERS: usize = 48;
    const FULL_ATTENTION_INTERVAL: usize = 4;
    const NUM_ATTENTION_HEADS: usize = 24;
    const NUM_KV_HEADS: usize = 2;
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
    const VISION_OUTPUT_HIDDEN: usize = 2_560;
    const VISION_INPUT_CHANNELS: usize = 3;
    const VISION_PATCH_SIZE: usize = 16;
    const VISION_SPATIAL_MERGE_SIZE: usize = 2;
    const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
}

#[cfg(test)]
mod tests {
    use super::{Arch, CheckpointContract, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

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
        assert_eq!(
            Qwen38_27B::CHECKPOINT_CONTRACT,
            CheckpointContract::CompressedTensors
        );
    }

    #[test]
    fn qwen35_profile_matches_checkpoint_geometry() {
        type A = Qwen35_9B;

        for (field, actual, expected) in [
            ("hidden", A::HIDDEN, 4_096),
            ("intermediate", A::INTERMEDIATE, 12_288),
            ("vocab", A::VOCAB, 248_320),
            ("layers", A::LAYERS, 32),
            ("full_attention_interval", A::FULL_ATTENTION_INTERVAL, 4),
            ("num_attention_heads", A::NUM_ATTENTION_HEADS, 16),
            ("num_kv_heads", A::NUM_KV_HEADS, 4),
            ("head_dim", A::HEAD_DIM, 256),
            ("linear_key_heads", A::LINEAR_KEY_HEADS, 16),
            ("linear_value_heads", A::LINEAR_VALUE_HEADS, 32),
            ("linear_head_dim", A::LINEAR_HEAD_DIM, 128),
            ("linear_conv_kernel_dim", A::LINEAR_CONV_KERNEL_DIM, 4),
            ("mtp_layers", A::MTP_LAYERS, 1),
            ("vision_depth", A::VISION_DEPTH, 27),
            ("vision_hidden", A::VISION_HIDDEN, 1_152),
            ("vision_intermediate", A::VISION_INTERMEDIATE, 4_304),
            ("vision_num_heads", A::VISION_NUM_HEADS, 16),
            ("vision_positions", A::VISION_POSITIONS, 2_304),
            ("vision_output_hidden", A::VISION_OUTPUT_HIDDEN, 4_096),
            ("vision_input_channels", A::VISION_INPUT_CHANNELS, 3),
            ("vision_patch_size", A::VISION_PATCH_SIZE, 16),
            ("vision_spatial_merge_size", A::VISION_SPATIAL_MERGE_SIZE, 2),
            (
                "vision_temporal_patch_size",
                A::VISION_TEMPORAL_PATCH_SIZE,
                2,
            ),
            ("attention_query_rows", A::ATTENTION_QUERY_ROWS, 8_192),
            (
                "attention_output_columns",
                A::ATTENTION_OUTPUT_COLUMNS,
                4_096,
            ),
            ("attention_kv_rows", A::ATTENTION_KV_ROWS, 1_024),
            ("attention_qkv_rows", A::ATTENTION_QKV_ROWS, 10_240),
            ("gdn_qk_rows", A::GDN_QK_ROWS, 2_048),
            ("gdn_value_rows", A::GDN_VALUE_ROWS, 4_096),
            ("gdn_qkv_rows", A::GDN_QKV_ROWS, 8_192),
            ("gdn_input_rows", A::GDN_INPUT_ROWS, 12_288),
            ("gdn_control_rows", A::GDN_CONTROL_ROWS, 32),
        ] {
            assert_eq!(actual, expected, "{field}");
        }

        const {
            assert!(!Qwen35_9B::MTP_USES_DEDICATED_EMBEDDINGS);
        }
        assert_eq!(Qwen35_9B::RMS_NORM_EPSILON, 1.0e-6);
    }

    #[test]
    fn qwen35_profile_matches_checkpoint_identity() {
        assert_eq!(Qwen35_9B::MODEL_ID, "AxionML/Qwen3.5-9B-NVFP4");
        assert_eq!(
            Qwen35_9B::REVISION,
            "97aef92393f126bf649f310cd40861be8dad3279"
        );
        assert_eq!(
            Qwen35_9B::CHECKPOINT_CONTRACT,
            CheckpointContract::ModelOptNvfp4
        );
    }

    #[test]
    fn qwen36_profile_matches_checkpoint_geometry() {
        type A = Qwen36Moe35B;

        for (field, actual, expected) in [
            ("hidden", A::HIDDEN, 2_048),
            ("expert_intermediate", A::INTERMEDIATE, 512),
            (
                "shared_expert_intermediate",
                A::SHARED_EXPERT_INTERMEDIATE,
                512,
            ),
            ("vocab", A::VOCAB, 248_320),
            ("layers", A::LAYERS, 40),
            ("full_attention_interval", A::FULL_ATTENTION_INTERVAL, 4),
            ("num_attention_heads", A::NUM_ATTENTION_HEADS, 16),
            ("num_kv_heads", A::NUM_KV_HEADS, 2),
            ("head_dim", A::HEAD_DIM, 256),
            ("linear_key_heads", A::LINEAR_KEY_HEADS, 16),
            ("linear_value_heads", A::LINEAR_VALUE_HEADS, 32),
            ("linear_head_dim", A::LINEAR_HEAD_DIM, 128),
            ("linear_conv_kernel_dim", A::LINEAR_CONV_KERNEL_DIM, 4),
            ("experts", A::NUM_EXPERTS, 256),
            ("experts_per_token", A::NUM_EXPERTS_PER_TOKEN, 8),
            (
                "max_position_embeddings",
                A::MAX_POSITION_EMBEDDINGS,
                262_144,
            ),
            ("attention_query_rows", A::ATTENTION_QUERY_ROWS, 8_192),
            (
                "attention_output_columns",
                A::ATTENTION_OUTPUT_COLUMNS,
                4_096,
            ),
            ("attention_kv_rows", A::ATTENTION_KV_ROWS, 512),
            ("attention_qkv_rows", A::ATTENTION_QKV_ROWS, 9_216),
            ("gdn_qk_rows", A::GDN_QK_ROWS, 2_048),
            ("gdn_value_rows", A::GDN_VALUE_ROWS, 4_096),
            ("gdn_qkv_rows", A::GDN_QKV_ROWS, 8_192),
            ("gdn_input_rows", A::GDN_INPUT_ROWS, 12_288),
            ("gdn_control_rows", A::GDN_CONTROL_ROWS, 32),
            ("vision_output_hidden", A::VISION_OUTPUT_HIDDEN, 2_048),
        ] {
            assert_eq!(actual, expected, "{field}");
        }

        const {
            assert!(!A::MTP_USES_DEDICATED_EMBEDDINGS);
        }
        assert_eq!(A::RMS_NORM_EPSILON, 1.0e-6);
        assert_eq!(A::FP8_CACHE_SCALE.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn qwen36_profile_matches_checkpoint_identity() {
        assert_eq!(Qwen36Moe35B::MODEL_ID, "nvidia/Qwen3.6-35B-A3B-NVFP4");
        assert_eq!(
            Qwen36Moe35B::REVISION,
            "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
        );
        assert_eq!(
            Qwen36Moe35B::CHECKPOINT_CONTRACT,
            CheckpointContract::ModelOptNvfp4Moe
        );
    }

    #[test]
    fn qwen38_flash_next_profile_matches_checkpoint_geometry() {
        type A = Qwen38FlashNext;

        for (field, actual, expected) in [
            ("hidden", A::HIDDEN, 2_560),
            ("expert_intermediate", A::INTERMEDIATE, 640),
            (
                "shared_expert_intermediate",
                A::SHARED_EXPERT_INTERMEDIATE,
                640,
            ),
            ("vocab", A::VOCAB, 248_320),
            ("layers", A::LAYERS, 48),
            ("full_attention_interval", A::FULL_ATTENTION_INTERVAL, 4),
            ("num_attention_heads", A::NUM_ATTENTION_HEADS, 24),
            ("num_kv_heads", A::NUM_KV_HEADS, 2),
            ("head_dim", A::HEAD_DIM, 256),
            ("linear_key_heads", A::LINEAR_KEY_HEADS, 16),
            ("linear_value_heads", A::LINEAR_VALUE_HEADS, 48),
            ("linear_head_dim", A::LINEAR_HEAD_DIM, 128),
            ("linear_conv_kernel_dim", A::LINEAR_CONV_KERNEL_DIM, 4),
            ("experts", A::NUM_EXPERTS, 512),
            ("experts_per_token", A::NUM_EXPERTS_PER_TOKEN, 10),
            (
                "max_position_embeddings",
                A::MAX_POSITION_EMBEDDINGS,
                262_144,
            ),
            ("hc_count", A::HC_COUNT, 4),
            ("hc_lowrank", A::HC_LOWRANK, 320),
            ("hc_width", A::HC_WIDTH, 10_240),
            ("indexer_heads", A::INDEXER_HEADS, 4),
            ("indexer_kv_heads", A::INDEXER_KV_HEADS, 1),
            ("indexer_head_dim", A::INDEXER_HEAD_DIM, 128),
            ("indexer_budget", A::INDEXER_BUDGET, 2_048),
            ("indexer_compress_ratio", A::INDEXER_COMPRESS_RATIO, 4),
            ("indexer_rows", A::INDEXER_ROWS, 640),
            ("ple_layer", A::PLE_LAYER, 1),
            ("ple_embed_dim", A::PLE_EMBED_DIM, 2_560),
            ("ple_conv_kernel", A::PLE_CONV_KERNEL, 4),
            ("ple_conv_dilation", A::PLE_CONV_DILATION, 3),
            ("ple_conv_state_len", A::PLE_CONV_STATE_LEN, 9),
            ("layer_conv_states", A::LAYER_CONV_STATES, 3),
            ("ple_conv_state_slot", A::PLE_CONV_STATE_SLOT, 1),
            ("ple_token_history_slot", A::PLE_TOKEN_HISTORY_SLOT, 2),
            ("ple_context_len", A::PLE_CONTEXT_LEN, 2),
            ("ngram_size", A::NGRAM_SIZE, 3),
            ("heads_per_ngram", A::HEADS_PER_NGRAM, 8),
            ("ngram_heads", A::NGRAM_HEADS, 16),
            ("ngram_head_dim", A::NGRAM_HEAD_DIM, 160),
            ("ngram_vocab_base", A::NGRAM_VOCAB_BASE, 20_000_000),
            ("ngram_vocab_divisor", A::NGRAM_VOCAB_DIVISOR, 128),
            ("ngram_shards", A::NGRAM_SHARDS, 128),
            ("ngram_shard_rows", A::NGRAM_SHARD_ROWS, 2_500_012),
            ("vision_depth", A::VISION_DEPTH, 27),
            ("vision_hidden", A::VISION_HIDDEN, 1_152),
            ("vision_intermediate", A::VISION_INTERMEDIATE, 4_304),
            ("vision_num_heads", A::VISION_NUM_HEADS, 16),
            ("vision_positions", A::VISION_POSITIONS, 2_304),
            ("vision_output_hidden", A::VISION_OUTPUT_HIDDEN, 2_560),
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
            ("attention_kv_rows", A::ATTENTION_KV_ROWS, 512),
            ("attention_qkv_rows", A::ATTENTION_QKV_ROWS, 13_312),
            ("gdn_qk_rows", A::GDN_QK_ROWS, 2_048),
            ("gdn_value_rows", A::GDN_VALUE_ROWS, 6_144),
            ("gdn_qkv_rows", A::GDN_QKV_ROWS, 10_240),
            ("gdn_input_rows", A::GDN_INPUT_ROWS, 16_384),
            ("gdn_control_rows", A::GDN_CONTROL_ROWS, 48),
        ] {
            assert_eq!(actual, expected, "{field}");
        }

        const {
            assert!(!A::MTP_USES_DEDICATED_EMBEDDINGS);
        }
        assert_eq!(A::MTP_LAYERS, 1);
        assert_eq!(A::RMS_NORM_EPSILON, 1.0e-6);
        assert_eq!(A::PARTIAL_ROTARY_FACTOR, 0.25);
        assert_eq!(A::ROPE_THETA, 10_000_000);
        assert_eq!(A::MROPE_SECTION, [11, 11, 10]);
        assert_eq!(A::GDN_OUTPUT_GATE, "sigmoid");
        assert_eq!(A::PLE_EMBEDDING_DTYPE, "float8_e4m3fn");
        assert_eq!(A::PLE_GATE_FLOOR, 1.0e-6);
    }

    /// The engram convolution shares only its kernel width with the GDN one.
    ///
    /// Dilation 3 makes the state nine columns rather than three and moves every
    /// tap, so a route that reused the GDN convolution here would be wrong.
    #[test]
    fn flashnext_engram_convolution_is_dilated_and_gdn_is_not() {
        type A = Qwen38FlashNext;

        assert_eq!(A::PLE_CONV_KERNEL, A::LINEAR_CONV_KERNEL_DIM);
        assert_eq!(A::PLE_CONV_DILATION, 3);
        assert_eq!(A::PLE_CONV_STATE_LEN, 9);
        assert_ne!(A::PLE_CONV_STATE_LEN, A::LINEAR_CONV_KERNEL_DIM - 1);
        assert_eq!(A::PLE_CONV_STATE_SLOT, 1);
        assert_eq!(A::PLE_TOKEN_HISTORY_SLOT, A::PLE_CONV_STATE_SLOT + 1);
        const {
            assert!(A::PLE_TOKEN_HISTORY_SLOT < A::LAYER_CONV_STATES);
        }
    }

    #[test]
    fn qwen38_flash_next_profile_matches_checkpoint_identity() {
        assert_eq!(
            Qwen38FlashNext::MODEL_ID,
            "RadixArk/Qwen3.8-Flash-Next-NVFP4"
        );
        assert_eq!(
            Qwen38FlashNext::REVISION,
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
        );
        assert_eq!(
            Qwen38FlashNext::CHECKPOINT_CONTRACT,
            CheckpointContract::Qwen38FlashNextModelOptNvfp4
        );
    }

    /// The engram table is 128 equal shards concatenated in shard-index order; the padded
    /// row count must reproduce the checkpoint's shard geometry exactly.
    #[test]
    fn qwen38_flash_next_engram_table_geometry_closes() {
        type A = Qwen38FlashNext;

        let rows = A::NGRAM_SHARDS * A::NGRAM_SHARD_ROWS;

        assert_eq!(rows, 320_001_536);
        assert!(rows.is_multiple_of(A::NGRAM_VOCAB_DIVISOR));
        assert_eq!(A::NGRAM_HEADS * A::NGRAM_HEAD_DIM, A::PLE_EMBED_DIM);
    }
}

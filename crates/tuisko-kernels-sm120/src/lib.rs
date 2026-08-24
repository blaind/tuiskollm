//! Exact-target SM120 operator kernels and their prepared host launchers.

mod arch;
mod attention;
mod bf16_lm_head;
mod device;
mod fp8;
mod gdn;
mod inventory;
mod moe;
mod mtp_bf16_attention_output;
mod mtp_bf16_fusion;
mod mtp_bf16_mlp;
mod mtp_bf16_paged_gqa;
mod mtp_bf16_qk_prepare;
mod mtp_bf16_qkv;
mod nvfp4_down;
mod nvfp4_gdn_input;
mod nvfp4_qkv;
mod nvfp4_swiglu;
mod qwen36_fp8_qkv;
mod qwen36_gdn_input;
mod qwen36_gdn_output;
mod residual_norm;

pub use arch::Sm120Arch;
pub use attention::{
    ATTENTION_PAGE_SIZE, AttentionOutputOp, AttentionQkPrepareOp, LONG_CONTEXT_GQA_MAX_PARTITIONS,
    LONG_CONTEXT_GQA_MAX_TOKENS, LONG_CONTEXT_GQA_PARTITION_BUCKETS,
    LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp,
    Qwen35AttentionQkPrepareOp, Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4GdnOutputOp,
    Qwen35PagedGqaOp, Qwen36AttentionQkPrepareOp, paged_gqa_prefill_partitions,
};
pub use bf16_lm_head::Qwen35Bf16LmHeadOp;
pub use fp8::{
    DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp, DenseFp8SwiGluTmaMaps,
    FullAttentionQkvOp, GdnInputProjectionOp, GdnOutputProjectionOp, LmHeadOp,
};
pub use gdn::{
    GdnPrepareOp, GdnRecurrenceOp, GdnStateSnapshotOp, Qwen35GdnPrepareOp, Qwen35GdnRecurrenceOp,
    Qwen36GdnPrepareOp, Qwen36GdnRecurrenceOp,
};
pub use inventory::kernel_ptx_names;
pub use moe::{Qwen36MoeExpertsOp, Qwen36MoeRouterOp};
pub use mtp_bf16_attention_output::MtpBf16AttentionOutputOp;
pub use mtp_bf16_fusion::MtpBf16FusionOp;
pub use mtp_bf16_mlp::MtpBf16MlpOp;
pub use mtp_bf16_paged_gqa::MtpBf16PagedGqaOp;
pub use mtp_bf16_qk_prepare::MtpBf16QkPrepareOp;
pub use mtp_bf16_qkv::MtpBf16QkvOp;
pub use nvfp4_down::{Nvfp4DownOp, Qwen35Nvfp4DownOp};
pub use nvfp4_gdn_input::Qwen35Nvfp4GdnInputOp;
pub use nvfp4_qkv::Qwen35Nvfp4QkvOp;
pub use nvfp4_swiglu::{Nvfp4SwiGluOp, Qwen35Nvfp4SwiGluOp};
pub use qwen36_fp8_qkv::Qwen36Fp8QkvOp;
pub use qwen36_gdn_input::Qwen36GdnInputOp;
pub use qwen36_gdn_output::Qwen36GdnOutputOp;
pub use residual_norm::{Qwen35ResidualNormOp, Qwen36ResidualNormOp, ResidualNormOp};

#[cfg(test)]
pub(crate) mod test_arch {
    use tuisko_model::Arch;

    #[derive(Clone, Copy)]
    pub(crate) struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/sm120-arch";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 1_024;
        const RMS_NORM_EPSILON: f32 = 1.0e-5;
        const INTERMEDIATE: usize = 512;
        const VOCAB: usize = 512;
        const LAYERS: usize = 4;
        const FULL_ATTENTION_INTERVAL: usize = 2;
        const NUM_ATTENTION_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const LINEAR_KEY_HEADS: usize = 2;
        const LINEAR_VALUE_HEADS: usize = 4;
        const LINEAR_HEAD_DIM: usize = 32;
        const LINEAR_CONV_KERNEL_DIM: usize = 4;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 2;
        const VISION_HIDDEN: usize = 64;
        const VISION_INTERMEDIATE: usize = 128;
        const VISION_NUM_HEADS: usize = 4;
        const VISION_POSITIONS: usize = 16;
        const VISION_OUTPUT_HIDDEN: usize = 1_024;
        const VISION_INPUT_CHANNELS: usize = 3;
        const VISION_PATCH_SIZE: usize = 8;
        const VISION_SPATIAL_MERGE_SIZE: usize = 2;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
    }
}

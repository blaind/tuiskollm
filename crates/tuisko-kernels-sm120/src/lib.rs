//! Exact-target SM120 operator kernels and their prepared host launchers.

mod arch;
mod attention;
mod device;
mod fp8;
mod gdn;
mod inventory;
mod nvfp4_down;
mod nvfp4_qkv;
mod nvfp4_swiglu;
mod residual_norm;

pub use arch::Sm120Arch;
pub use attention::{
    ATTENTION_PAGE_SIZE, AttentionOutputOp, AttentionQkPrepareOp, LONG_CONTEXT_GQA_MAX_PARTITIONS,
    LONG_CONTEXT_GQA_MAX_TOKENS, LONG_CONTEXT_GQA_PARTITION_BUCKETS,
    LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp,
    Qwen35AttentionQkPrepareOp, Qwen35Nvfp4AttentionOutputOp, Qwen35PagedGqaOp,
    paged_gqa_prefill_partitions,
};
pub use fp8::{
    DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp, DenseFp8SwiGluTmaMaps,
    FullAttentionQkvOp, GdnInputProjectionOp, GdnOutputProjectionOp, LmHeadOp,
};
pub use gdn::{GdnPrepareOp, GdnRecurrenceOp};
pub use inventory::kernel_ptx_names;
pub use nvfp4_down::{Nvfp4DownOp, Qwen35Nvfp4DownOp};
pub use nvfp4_qkv::Qwen35Nvfp4QkvOp;
pub use nvfp4_swiglu::{Nvfp4SwiGluOp, Qwen35Nvfp4SwiGluOp};
pub use residual_norm::{Qwen35ResidualNormOp, ResidualNormOp};

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

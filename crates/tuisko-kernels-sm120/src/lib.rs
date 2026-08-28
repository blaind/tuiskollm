//! Exact-target SM120 operator kernels and their prepared host launchers.
//!
//! The operators live in per-family crates so that editing one family re-runs
//! cuda-oxide device codegen only for that family. This crate is the stable
//! facade over them: it re-exports the same public surface the engine has
//! always consumed and owns the aggregate entry inventory.

mod inventory;

pub use inventory::kernel_ptx_names;
pub use tuisko_kernels_sm120_attention::{
    ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, IndexerCompressArgs, IndexerPrepareArgs,
    IndexerSelectionArgs, LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp,
    Qwen35AttentionQkPrepareOp, Qwen35PagedGqaOp, Qwen36AttentionQkPrepareOp,
    Qwen36Fp8AttentionQkPrepareOp, Qwen36Fp8PagedGqaOp, Qwen36PagedGqaOp,
    Qwen38FlashNextAttentionQkPrepareOp, Qwen38FlashNextIndexerPrepareOp,
    Qwen38FlashNextIndexerSelectionOp, Qwen38FlashNextPagedGqaOp,
    Qwen38FlashNextSelectedPagedGqaOp, SELECTION_BLOCK_BUCKETS, SELECTION_BLOCKS_PER_PAGE,
    SELECTION_MAX_BATCH, SELECTION_MAX_BLOCKS, SELECTION_MAX_SELECTED, SELECTION_PREFILL_TOKENS,
    SELECTION_ROW_TILE, SelectedAttentionArgs, paged_gqa_prefill_partitions,
    selection_block_bucket, selection_round_blocks, selection_round_rows,
};
pub use tuisko_kernels_sm120_common::Sm120Arch;
pub use tuisko_kernels_sm120_engram::{
    Qwen38FlashNextEngramOp, Qwen38FlashNextEngramSources, Qwen38FlashNextEngramWorkspace,
    Qwen38FlashNextPleStateSnapshotOp,
};
pub use tuisko_kernels_sm120_fp8_mlp::{
    DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp, DenseFp8SwiGluTmaMaps,
};
pub use tuisko_kernels_sm120_fp8_projection::{
    AttentionOutputOp, DenseFp8GdnInputTmaMaps, FullAttentionQkvOp, GdnInputProjectionOp,
    GdnOutputProjectionOp, LmHeadOp, Qwen36AttentionOutputOp, Qwen36Fp8QkvOp, Qwen36GdnInputOp,
    Qwen36GdnOutputOp, Qwen38FlashNextAttentionGateOp,
};
pub use tuisko_kernels_sm120_gdn::{
    GdnPrepareOp, GdnRecurrenceOp, GdnStateSnapshotOp, Qwen35GdnPrepareOp, Qwen35GdnRecurrenceOp,
    Qwen36GdnPrepareOp, Qwen36GdnRecurrenceOp, Qwen38FlashNextGdnPrepareOp,
    Qwen38FlashNextGdnRecurrenceOp, Qwen38FlashNextGdnStateSnapshotOp,
};
pub use tuisko_kernels_sm120_hyper_connection::Qwen38FlashNextHyperConnectionOp;
pub use tuisko_kernels_sm120_lm_head::{Qwen35Bf16LmHeadOp, Qwen36Nvfp4LmHeadOp};
pub use tuisko_kernels_sm120_moe::{
    QWEN38_FLASH_NEXT_ABSENT_SLOT, QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, Qwen36MoeExpertsOp,
    Qwen36MoeRouterOp, Qwen36MtpBf16MoeOp, Qwen38FlashNextExpertDispatch,
    Qwen38FlashNextMoeExpertsOp, Qwen38FlashNextMoeRouterOp, Qwen38FlashNextSlotPlane,
    qwen38_flash_next_expert_slot_plane,
};
pub use tuisko_kernels_sm120_mtp::{
    MtpBf16AttentionOutputOp, MtpBf16FusionOp, MtpBf16MlpOp, MtpBf16PagedGqaOp, MtpBf16QkPrepareOp,
    MtpBf16QkvOp, Qwen35MtpBf16AttentionOutputOp, Qwen35MtpBf16FusionOp, Qwen35MtpBf16MlpOp,
    Qwen35MtpBf16PagedGqaOp, Qwen35MtpBf16QkPrepareOp, Qwen35MtpBf16QkvOp,
    Qwen36MtpBf16AttentionOutputOp, Qwen36MtpBf16FusionOp, Qwen36MtpBf16QkvOp,
};
pub use tuisko_kernels_sm120_norm::{Qwen35ResidualNormOp, Qwen36ResidualNormOp, ResidualNormOp};
pub use tuisko_kernels_sm120_nvfp4::{
    Nvfp4DownOp, Nvfp4SwiGluOp, Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4DownOp,
    Qwen35Nvfp4GdnInputOp, Qwen35Nvfp4GdnOutputOp, Qwen35Nvfp4QkvOp, Qwen35Nvfp4SwiGluOp,
};

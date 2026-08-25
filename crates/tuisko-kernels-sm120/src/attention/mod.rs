//! Exact full-attention operators.

mod long_context_paged_gqa;
mod nvfp4_output;
mod output;
mod paged_gqa;
mod qk_prepare;

pub(crate) use long_context_paged_gqa::long_context_paged_gqa_ptx_names;
pub use long_context_paged_gqa::{
    LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
};
pub(crate) use nvfp4_output::qwen35_nvfp4_attention_output_ptx_names;
pub use nvfp4_output::{Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4GdnOutputOp};
pub use output::AttentionOutputOp;
pub(crate) use output::attention_output_ptx_names;
pub use paged_gqa::{
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp, Qwen35PagedGqaOp,
    Qwen36PagedGqaOp, paged_gqa_prefill_partitions,
};
pub(crate) use paged_gqa::{
    paged_gqa_ptx_names, qwen35_paged_gqa_ptx_names, qwen36_paged_gqa_ptx_names,
};
pub use qk_prepare::{
    ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, Qwen35AttentionQkPrepareOp,
    Qwen36AttentionQkPrepareOp, Qwen36Fp8AttentionQkPrepareOp,
};
pub(crate) use qk_prepare::{
    attention_qk_prepare_ptx_names, qwen35_attention_qk_prepare_ptx_names,
    qwen36_attention_qk_prepare_ptx_names, qwen36_fp8_attention_qk_prepare_ptx_names,
};

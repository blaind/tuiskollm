//! Exact full-attention operators for SM120.

mod device;
mod long_context_paged_gqa;
mod paged_gqa;
mod qk_prepare;

/// Device bodies the MTP BF16 family reuses.
///
/// The MTP kernels launch their own entries around these bodies, so they are
/// re-exported rather than copied into a second crate.
pub mod shared_device {
    pub use crate::device::attention_qk_prepare::bf16_attention_qk_prepare;
    pub use crate::device::paged_gqa::{
        DECODE_RING_SHARED_BYTES, DECODE_SHARED_VALUES, DECODE_THREADS, bf16_paged_gqa,
        bf16_paged_gqa_partitioned,
    };
}

pub use long_context_paged_gqa::{
    LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
};
pub use paged_gqa::{
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp, Qwen35PagedGqaOp,
    Qwen36Fp8PagedGqaOp, Qwen36PagedGqaOp, paged_gqa_prefill_partitions,
};
pub use qk_prepare::{
    ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, Bf16Cache, CacheFormat, CacheScales, Fp8Cache,
    PreparedPrefillRoute, PreparedQwen35PrefillRoute, PreparedQwen35Route,
    PreparedQwen36Fp8PrefillRoute, PreparedQwen36Fp8Route, PreparedQwen36PrefillRoute,
    PreparedQwen36Route, PreparedRoute, QkPrepareArgs, QkPrepareEntries, QkPrepareRoute,
    Qwen35AttentionQkPrepareOp, Qwen35QkPrepareEntries, Qwen36AttentionQkPrepareOp,
    Qwen36Fp8AttentionQkPrepareOp, Qwen36Fp8QkPrepareEntries, Qwen36QkPrepareEntries,
    Qwen38QkPrepareEntries, UnadmittedRoute,
};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    qk_prepare::attention_qk_prepare_ptx_names()
        .into_iter()
        .chain(qk_prepare::qwen35_attention_qk_prepare_ptx_names())
        .chain(qk_prepare::qwen36_attention_qk_prepare_ptx_names())
        .chain(qk_prepare::qwen36_fp8_attention_qk_prepare_ptx_names())
        .chain(paged_gqa::paged_gqa_ptx_names())
        .chain(paged_gqa::qwen35_paged_gqa_ptx_names())
        .chain(paged_gqa::qwen36_paged_gqa_ptx_names())
        .chain(paged_gqa::qwen36_fp8_paged_gqa_ptx_names())
        .chain(long_context_paged_gqa::long_context_paged_gqa_ptx_names())
        .collect()
}

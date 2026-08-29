//! Exact full-attention operators for SM120.

mod device;
mod long_context_mtp_paged_gqa;
mod long_context_paged_gqa;
mod paged_gqa;
mod qk_prepare;
mod qsa_selection;

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

pub use long_context_mtp_paged_gqa::LongContextMtpPagedGqaOp;
pub use long_context_paged_gqa::{
    LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
};
pub use paged_gqa::{
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PAGED_GQA_PREFILL_MAX_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES, PagedGqaOp, Qwen35PagedGqaOp,
    Qwen36Fp8PagedGqaOp, Qwen36PagedGqaOp, Qwen38FlashNextPagedGqaOp, paged_gqa_prefill_partitions,
};
pub use qk_prepare::{
    ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, Bf16Cache, CacheFormat, CacheScales, Fp8Cache,
    PreparedPrefillRoute, PreparedQwen35PrefillRoute, PreparedQwen35Route,
    PreparedQwen36Fp8PrefillRoute, PreparedQwen36Fp8Route, PreparedQwen36PrefillRoute,
    PreparedQwen36Route, PreparedRoute, QkPrepareArgs, QkPrepareEntries, QkPrepareRoute,
    Qwen35AttentionQkPrepareOp, Qwen35QkPrepareEntries, Qwen36AttentionQkPrepareOp,
    Qwen36Fp8AttentionQkPrepareOp, Qwen36Fp8QkPrepareEntries, Qwen36QkPrepareEntries,
    Qwen38FlashNextAttentionQkPrepareOp, Qwen38FlashNextQkPrepareEntries, Qwen38QkPrepareEntries,
    UnadmittedRoute,
};
pub use qsa_selection::{
    IndexerCompressArgs, IndexerPrepareArgs, IndexerSelectionArgs, Qwen38FlashNextIndexerPrepareOp,
    Qwen38FlashNextIndexerSelectionOp, Qwen38FlashNextSelectedPagedGqaOp, SELECTION_BLOCK_BUCKETS,
    SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_BATCH, SELECTION_MAX_BLOCKS,
    SELECTION_MAX_CTAS_PER_ROW, SELECTION_MAX_SELECTED, SELECTION_PREFILL_TOKENS,
    SELECTION_RADIX_PASSES, SELECTION_RING_SLOTS, SELECTION_ROW_TILE, SELECTION_SCRATCH_WORDS,
    SelectedAttentionArgs, selection_block_bucket, selection_ctas_per_row, selection_round_blocks,
    selection_round_rows,
};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    qk_prepare::attention_qk_prepare_ptx_names()
        .into_iter()
        .chain(qk_prepare::qwen35_attention_qk_prepare_ptx_names())
        .chain(qk_prepare::qwen36_attention_qk_prepare_ptx_names())
        .chain(qk_prepare::qwen36_fp8_attention_qk_prepare_ptx_names())
        .chain(qk_prepare::qwen38_flash_next_attention_qk_prepare_ptx_names())
        .chain(paged_gqa::paged_gqa_ptx_names())
        .chain(paged_gqa::qwen35_paged_gqa_ptx_names())
        .chain(paged_gqa::qwen36_paged_gqa_ptx_names())
        .chain(paged_gqa::qwen36_fp8_paged_gqa_ptx_names())
        .chain(paged_gqa::qwen38_flash_next_paged_gqa_ptx_names())
        .chain(long_context_paged_gqa::long_context_paged_gqa_ptx_names())
        .chain(long_context_mtp_paged_gqa::long_context_mtp_paged_gqa_ptx_names())
        .chain(qsa_selection::qwen38_flash_next_indexer_ptx_names())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::kernel_ptx_names;
    use std::collections::{BTreeMap, BTreeSet};

    /// A generic specialization is exported as `base_TID_<type hash>`, and that
    /// hash is only reproducible inside the compilation that emitted it. The
    /// base name is the part a host build and the device build agree on.
    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    /// The declared count `tuisko-kernels-sm120` pins for this family, split
    /// per entry. Adding or dropping a specialization moves one row, so an
    /// owner merge from silently changing the emitted artifact.
    #[test]
    fn family_inventory_is_pinned_per_base_name() {
        let names = kernel_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), names.len());

        let mut counts = BTreeMap::new();
        for name in names {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("attention_qk_prepare_exact", 8),
                ("attention_qk_prepare_prefill_exact", 4),
                ("long_context_mtp_paged_gqa_partial_exact", 3),
                ("long_context_mtp_paged_gqa_reduce_exact", 3),
                ("long_context_paged_gqa_partial_exact", 8),
                ("long_context_paged_gqa_reduce_exact", 8),
                ("paged_gqa_exact", 8),
                ("paged_gqa_prefill_flash_macro_exact", 1),
                ("paged_gqa_prefill_flash_p16_exact", 1),
                ("paged_gqa_prefill_flash_p8_exact", 1),
                ("paged_gqa_prefill_macro_reduce_exact", 5),
                ("paged_gqa_prefill_partitioned_reduce_exact", 2),
                ("paged_gqa_prefill_shared_exact", 3),
                ("qwen35_attention_qk_prepare_exact", 8),
                ("qwen35_attention_qk_prepare_prefill_exact", 4),
                ("qwen35_paged_gqa_exact", 8),
                ("qwen35_paged_gqa_prefill_shared_exact", 3),
                ("qwen36_attention_qk_prepare_exact", 8),
                ("qwen36_attention_qk_prepare_prefill_exact", 3),
                ("qwen36_fp8_attention_qk_prepare_exact", 8),
                ("qwen36_fp8_attention_qk_prepare_prefill_exact", 3),
                ("qwen36_fp8_paged_gqa_exact", 8),
                ("qwen36_fp8_paged_gqa_prefill_shared_exact", 3),
                ("qwen36_paged_gqa_exact", 8),
                ("qwen36_paged_gqa_prefill_shared_exact", 3),
                ("qwen38_flash_next_attention_qk_prepare_exact", 8),
                ("qwen38_flash_next_attention_qk_prepare_prefill_exact", 4),
                ("qwen38_flash_next_indexer_block_compress_exact", 12),
                ("qwen38_flash_next_indexer_prepare_exact", 8),
                ("qwen38_flash_next_indexer_prepare_prefill_exact", 4),
                ("qwen38_flash_next_indexer_score_exact", 10),
                ("qwen38_flash_next_indexer_select_expand_exact", 10),
                ("qwen38_flash_next_indexer_select_pass_exact", 10),
                ("qwen38_flash_next_paged_gqa_exact", 8),
                ("qwen38_flash_next_paged_gqa_prefill_selected_exact", 4),
                ("qwen38_flash_next_paged_gqa_prefill_shared_exact", 4),
                ("qwen38_flash_next_paged_gqa_selected_exact", 8),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 212);
    }
}

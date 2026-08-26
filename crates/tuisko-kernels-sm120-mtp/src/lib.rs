//! MTP BF16 speculative-decode operators for SM120.
//!
//! Two of these kernels launch entries around device bodies owned by the
//! full-attention crate, and the fusion operator composes the RMSNorm owner,
//! so this crate depends on both rather than copying either.

mod mtp_bf16_attention_output;
mod mtp_bf16_fusion;
mod mtp_bf16_mlp;
mod mtp_bf16_paged_gqa;
mod mtp_bf16_qk_prepare;
mod mtp_bf16_qkv;

pub use mtp_bf16_attention_output::{
    MtpAttentionOutputEntries, MtpAttentionOutputRoute, MtpBf16AttentionOutputOp,
    Qwen35MtpAttentionOutputEntries, Qwen35MtpBf16AttentionOutputOp,
    Qwen36MtpAttentionOutputEntries, Qwen36MtpBf16AttentionOutputOp,
    Qwen38MtpAttentionOutputEntries,
};
pub use mtp_bf16_fusion::{
    MtpBf16FusionOp, MtpFusionEntries, MtpFusionRoute, Qwen35MtpBf16FusionOp,
    Qwen35MtpFusionEntries, Qwen36MtpBf16FusionOp, Qwen36MtpFusionEntries, Qwen38MtpFusionEntries,
    UnadmittedFusionRoute,
};
pub use mtp_bf16_mlp::{
    MtpBf16MlpOp, MtpMlpEntries, MtpMlpRoute, Qwen35MtpBf16MlpOp, Qwen35MtpMlpEntries,
    Qwen38MtpMlpEntries,
};
pub use mtp_bf16_paged_gqa::{
    MtpBf16PagedGqaOp, MtpPagedGqaEntries, MtpPagedGqaRoute, Qwen35MtpBf16PagedGqaOp,
    Qwen35MtpPagedGqaEntries, Qwen38MtpPagedGqaEntries,
};
pub use mtp_bf16_qk_prepare::{
    MtpBf16QkPrepareOp, MtpQkPrepareEntries, MtpQkPrepareRoute, Qwen35MtpBf16QkPrepareOp,
    Qwen35MtpQkPrepareEntries, Qwen38MtpQkPrepareEntries, UnadmittedQkPrepareRoute,
};
pub use mtp_bf16_qkv::{
    MtpBf16QkvOp, MtpQkvEntries, MtpQkvRoute, Qwen35MtpBf16QkvOp, Qwen35MtpQkvEntries,
    Qwen36MtpBf16QkvOp, Qwen36MtpQkvEntries, Qwen38MtpQkvEntries, UnadmittedQkvRoute,
};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    mtp_bf16_fusion::mtp_bf16_fusion_ptx_names()
        .into_iter()
        .chain(mtp_bf16_fusion::mtp_bf16_fusion_prefill_ptx_names())
        .chain(mtp_bf16_fusion::qwen35_mtp_bf16_fusion_ptx_names())
        .chain(mtp_bf16_fusion::qwen36_mtp_bf16_fusion_ptx_names())
        .chain(mtp_bf16_mlp::mtp_bf16_mlp_ptx_names())
        .chain(mtp_bf16_mlp::qwen35_mtp_bf16_mlp_ptx_names())
        .chain(mtp_bf16_attention_output::mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_attention_output::qwen35_mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_attention_output::qwen36_mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_qkv::mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qkv::mtp_bf16_qkv_prefill_ptx_names())
        .chain(mtp_bf16_qkv::qwen35_mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qkv::qwen36_mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qk_prepare::mtp_bf16_qk_prepare_ptx_names())
        .chain(mtp_bf16_qk_prepare::mtp_bf16_qk_prepare_prefill_ptx_names())
        .chain(mtp_bf16_qk_prepare::qwen35_mtp_bf16_qk_prepare_ptx_names())
        .chain(mtp_bf16_paged_gqa::mtp_bf16_paged_gqa_ptx_names())
        .chain(mtp_bf16_paged_gqa::qwen35_mtp_bf16_paged_gqa_ptx_names())
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
    /// per entry. A wrapper change that instantiates one more specialization —
    /// or drops one — moves exactly one of these rows, which is what keeps an
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
                ("mtp_bf16_attention_gate", 8),
                ("mtp_bf16_attention_output", 8),
                ("mtp_bf16_down", 8),
                ("mtp_bf16_fusion", 8),
                ("mtp_bf16_fusion_prefill", 4),
                ("mtp_bf16_paged_gqa", 8),
                ("mtp_bf16_qk_prepare", 8),
                ("mtp_bf16_qk_prepare_prefill", 4),
                ("mtp_bf16_qkv", 8),
                ("mtp_bf16_qkv_prefill", 4),
                ("mtp_bf16_swiglu", 8),
                ("qwen35_mtp_bf16_attention_gate", 8),
                ("qwen35_mtp_bf16_attention_output", 8),
                ("qwen35_mtp_bf16_down", 8),
                ("qwen35_mtp_bf16_fusion", 8),
                ("qwen35_mtp_bf16_fusion_prefill", 3),
                ("qwen35_mtp_bf16_paged_gqa", 8),
                ("qwen35_mtp_bf16_qk_prepare", 8),
                ("qwen35_mtp_bf16_qk_prepare_prefill", 3),
                ("qwen35_mtp_bf16_qkv", 8),
                ("qwen35_mtp_bf16_qkv_prefill", 3),
                ("qwen35_mtp_bf16_swiglu", 8),
                ("qwen36_mtp_bf16_attention_gate", 8),
                ("qwen36_mtp_bf16_attention_output", 8),
                ("qwen36_mtp_bf16_fusion", 8),
                ("qwen36_mtp_bf16_fusion_prefill", 3),
                ("qwen36_mtp_bf16_qkv", 8),
                ("qwen36_mtp_bf16_qkv_prefill", 3),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 187);
    }
}

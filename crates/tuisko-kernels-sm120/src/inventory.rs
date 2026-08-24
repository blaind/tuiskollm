use crate::attention::{
    attention_output_ptx_names, attention_qk_prepare_ptx_names, long_context_paged_gqa_ptx_names,
    paged_gqa_ptx_names, qwen35_attention_qk_prepare_ptx_names,
    qwen35_nvfp4_attention_output_ptx_names, qwen35_paged_gqa_ptx_names,
    qwen36_attention_qk_prepare_ptx_names, qwen36_paged_gqa_ptx_names,
};
use crate::bf16_lm_head::qwen35_bf16_lm_head_ptx_names;
use crate::fp8::gdn_output_ptx_names;
use crate::fp8::{
    fp8_down_ptx_names, fp8_gdn_input_ptx_names, fp8_lm_head_ptx_names, fp8_qkv_ptx_names,
    fp8_swiglu_ptx_names,
};
use crate::gdn::{
    gdn_prepare_ptx_names, gdn_recurrence_ptx_names, gdn_state_snapshot_ptx_name,
    qwen35_gdn_prepare_ptx_names, qwen35_gdn_recurrence_ptx_names,
};
use crate::moe::{qwen36_moe_experts_ptx_names, qwen36_moe_router_ptx_names};
use crate::mtp_bf16_attention_output::mtp_bf16_attention_output_ptx_names;
use crate::mtp_bf16_fusion::{mtp_bf16_fusion_prefill_ptx_names, mtp_bf16_fusion_ptx_names};
use crate::mtp_bf16_mlp::mtp_bf16_mlp_ptx_names;
use crate::mtp_bf16_paged_gqa::mtp_bf16_paged_gqa_ptx_names;
use crate::mtp_bf16_qk_prepare::{
    mtp_bf16_qk_prepare_prefill_ptx_names, mtp_bf16_qk_prepare_ptx_names,
};
use crate::mtp_bf16_qkv::{mtp_bf16_qkv_prefill_ptx_names, mtp_bf16_qkv_ptx_names};
use crate::nvfp4_down::{nvfp4_down_ptx_names, qwen35_nvfp4_down_ptx_names};
use crate::nvfp4_gdn_input::qwen35_nvfp4_gdn_input_ptx_names;
use crate::nvfp4_qkv::qwen35_nvfp4_qkv_ptx_names;
use crate::nvfp4_swiglu::{nvfp4_swiglu_ptx_names, qwen35_nvfp4_swiglu_ptx_names};
use crate::qwen36_attention_output::qwen36_attention_output_ptx_names;
use crate::qwen36_fp8_qkv::qwen36_fp8_qkv_ptx_names;
use crate::qwen36_gdn_input::qwen36_gdn_input_ptx_names;
use crate::qwen36_gdn_output::qwen36_gdn_output_ptx_names;
use crate::qwen36_nvfp4_lm_head::qwen36_nvfp4_lm_head_ptx_names;
use crate::residual_norm::{
    qwen35_residual_norm_ptx_names, qwen36_residual_norm_ptx_names, residual_norm_ptx_names,
};

/// Stable semantic inventory of every admitted SM120 entry.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    residual_norm_ptx_names()
        .into_iter()
        .chain(qwen35_residual_norm_ptx_names())
        .chain(qwen36_residual_norm_ptx_names())
        .chain(attention_qk_prepare_ptx_names())
        .chain(qwen35_attention_qk_prepare_ptx_names())
        .chain(qwen36_attention_qk_prepare_ptx_names())
        .chain(paged_gqa_ptx_names())
        .chain(qwen35_paged_gqa_ptx_names())
        .chain(qwen36_paged_gqa_ptx_names())
        .chain(qwen35_nvfp4_attention_output_ptx_names())
        .chain(long_context_paged_gqa_ptx_names())
        .chain(attention_output_ptx_names())
        .chain(fp8_qkv_ptx_names())
        .chain(fp8_gdn_input_ptx_names())
        .chain(fp8_lm_head_ptx_names())
        .chain(fp8_swiglu_ptx_names())
        .chain(fp8_down_ptx_names())
        .chain(gdn_output_ptx_names())
        .chain(gdn_prepare_ptx_names())
        .chain(gdn_recurrence_ptx_names())
        .chain([gdn_state_snapshot_ptx_name()])
        .chain(mtp_bf16_fusion_ptx_names())
        .chain(mtp_bf16_fusion_prefill_ptx_names())
        .chain(mtp_bf16_mlp_ptx_names())
        .chain(mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qkv_prefill_ptx_names())
        .chain(mtp_bf16_qk_prepare_ptx_names())
        .chain(mtp_bf16_qk_prepare_prefill_ptx_names())
        .chain(mtp_bf16_paged_gqa_ptx_names())
        .chain(nvfp4_swiglu_ptx_names())
        .chain(qwen35_nvfp4_swiglu_ptx_names())
        .chain(nvfp4_down_ptx_names())
        .chain(qwen35_nvfp4_down_ptx_names())
        .chain(qwen35_nvfp4_qkv_ptx_names())
        .chain(qwen35_nvfp4_gdn_input_ptx_names())
        .chain(qwen35_gdn_prepare_ptx_names())
        .chain(qwen35_gdn_recurrence_ptx_names())
        .chain(qwen35_bf16_lm_head_ptx_names())
        .chain(qwen36_moe_router_ptx_names())
        .chain(qwen36_moe_experts_ptx_names())
        .chain(qwen36_gdn_input_ptx_names())
        .chain(qwen36_gdn_output_ptx_names())
        .chain(qwen36_fp8_qkv_ptx_names())
        .chain(qwen36_attention_output_ptx_names())
        .chain(qwen36_nvfp4_lm_head_ptx_names())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::kernel_ptx_names;
    use std::collections::BTreeSet;

    #[test]
    fn inventory_has_no_missing_or_duplicate_entries() {
        let names = kernel_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 656);
        assert_eq!(unique.len(), names.len());
    }
}

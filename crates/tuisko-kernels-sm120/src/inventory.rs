use crate::attention::{
    attention_output_ptx_names, attention_qk_prepare_ptx_names, long_context_paged_gqa_ptx_names,
    paged_gqa_ptx_names,
};
use crate::fp8::gdn_output_ptx_names;
use crate::fp8::{
    fp8_down_ptx_names, fp8_gdn_input_ptx_names, fp8_lm_head_ptx_names, fp8_qkv_ptx_names,
    fp8_swiglu_ptx_names,
};
use crate::gdn::{gdn_prepare_ptx_names, gdn_recurrence_ptx_names};
use crate::nvfp4_down::nvfp4_down_ptx_names;
use crate::nvfp4_swiglu::nvfp4_swiglu_ptx_names;
use crate::residual_norm::residual_norm_ptx_names;

/// Stable semantic inventory of every admitted SM120 entry.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    residual_norm_ptx_names()
        .into_iter()
        .chain(attention_qk_prepare_ptx_names())
        .chain(paged_gqa_ptx_names())
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
        .chain(nvfp4_swiglu_ptx_names())
        .chain(nvfp4_down_ptx_names())
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

        assert_eq!(names.len(), 170);
        assert_eq!(unique.len(), names.len());
    }
}

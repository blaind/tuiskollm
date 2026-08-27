//! Source-BF16 backbone projection operators for the Qwen3.8-Flash-Next target on SM120.
//!
//! Every decoder layer moves activations between four widths through plain
//! `nn.Linear`s. This crate owns the three shapes that sit inside a layer; the
//! vocabulary projection that closes the stack belongs to the LM-head family
//! beside the other heads. All four instantiate one device body, so an edit to
//! the projection arithmetic reaches them together.

mod backbone;

pub use backbone::{
    Qwen38FlashNextBlockOutputProjectionOp, Qwen38FlashNextGdnInputProjectionOp,
    Qwen38FlashNextQsaQkvProjectionOp,
};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    backbone::qwen38_flash_next_gdn_input_projection_ptx_names()
        .into_iter()
        .chain(backbone::qwen38_flash_next_qsa_qkv_projection_ptx_names())
        .chain(backbone::qwen38_flash_next_block_output_projection_ptx_names())
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
    /// per entry. Adding or dropping a specialization moves exactly one row.
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
                ("qwen38_flash_next_block_output_projection", 8),
                ("qwen38_flash_next_block_output_projection_prefill", 4),
                ("qwen38_flash_next_gdn_input_projection", 8),
                ("qwen38_flash_next_gdn_input_projection_prefill", 4),
                ("qwen38_flash_next_qsa_qkv_projection", 8),
                ("qwen38_flash_next_qsa_qkv_projection_prefill", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 36);
    }
}

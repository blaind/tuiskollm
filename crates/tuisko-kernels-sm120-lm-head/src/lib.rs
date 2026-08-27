//! Exact language-model head operators for SM120.
//!
//! Qwen3.8-Flash-Next shares its plain BF16 projection body with the backbone
//! family. The other heads retain their target-specific device bodies.

mod bf16_lm_head;
mod qwen36_nvfp4_lm_head;
mod qwen38_flash_next_bf16_lm_head;

pub use bf16_lm_head::Qwen35Bf16LmHeadOp;
pub use qwen36_nvfp4_lm_head::Qwen36Nvfp4LmHeadOp;
pub use qwen38_flash_next_bf16_lm_head::Qwen38FlashNextBf16LmHeadOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    bf16_lm_head::qwen35_bf16_lm_head_ptx_names()
        .into_iter()
        .chain(qwen38_flash_next_bf16_lm_head::qwen38_flash_next_bf16_lm_head_ptx_names())
        .chain(qwen36_nvfp4_lm_head::qwen36_nvfp4_lm_head_ptx_names())
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
    /// per entry, so adding a head moves exactly one of these rows.
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
                ("qwen35_bf16_lm_head", 8),
                ("qwen36_nvfp4_lm_head_a16", 8),
                ("qwen38_flash_next_bf16_lm_head", 8),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 24);
    }
}

/// One device-codegen crate: its PTX module stem and its family inventory.
type Family = (&'static str, fn() -> Vec<&'static str>);

/// Every SM120 device-codegen crate.
///
/// Each entry owns exactly one PTX module, so the aggregate inventory is the
/// concatenation of the family inventories and never an independent list.
const FAMILIES: &[Family] = &[
    (
        "tuisko_kernels_sm120_norm",
        tuisko_kernels_sm120_norm::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_attention",
        tuisko_kernels_sm120_attention::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_nvfp4",
        tuisko_kernels_sm120_nvfp4::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_fp8_projection",
        tuisko_kernels_sm120_fp8_projection::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_fp8_mlp",
        tuisko_kernels_sm120_fp8_mlp::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_qwen38_flash_next_projection",
        tuisko_kernels_sm120_qwen38_flash_next_projection::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_gdn",
        tuisko_kernels_sm120_gdn::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_mtp",
        tuisko_kernels_sm120_mtp::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_hyper_connection",
        tuisko_kernels_sm120_hyper_connection::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_lm_head",
        tuisko_kernels_sm120_lm_head::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_moe",
        tuisko_kernels_sm120_moe::kernel_ptx_names,
    ),
    (
        "tuisko_kernels_sm120_engram",
        tuisko_kernels_sm120_engram::kernel_ptx_names,
    ),
];

/// Stable semantic inventory of every admitted SM120 entry.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    FAMILIES
        .iter()
        .flat_map(|(_, names)| names())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::{FAMILIES, kernel_ptx_names};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    /// Directory `xtask` compiles the pinned SM120 device build's modules to.
    const PTX_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/cuda");

    /// Declared entry count of every family, and their sum.
    ///
    /// The gate contract is three-way: emitted == declared == the sum of these
    /// counts. A family moving an entry to another crate has to change this
    /// table, which is what makes the move visible in review.
    const FAMILY_COUNTS: &[(&str, usize)] = &[
        ("tuisko_kernels_sm120_norm", 68),
        ("tuisko_kernels_sm120_attention", 196),
        ("tuisko_kernels_sm120_nvfp4", 140),
        ("tuisko_kernels_sm120_fp8_projection", 160),
        ("tuisko_kernels_sm120_fp8_mlp", 26),
        ("tuisko_kernels_sm120_qwen38_flash_next_projection", 36),
        ("tuisko_kernels_sm120_gdn", 134),
        ("tuisko_kernels_sm120_mtp", 187),
        ("tuisko_kernels_sm120_hyper_connection", 60),
        ("tuisko_kernels_sm120_lm_head", 24),
        ("tuisko_kernels_sm120_moe", 196),
        ("tuisko_kernels_sm120_engram", 66),
    ];

    /// A generic specialization is exported as `base_TID_<type hash>`, and that
    /// hash is only reproducible inside the compilation that emitted it; a
    /// host build of this crate hashes the same kernel differently. The base
    /// name is the part both compilations agree on.
    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    fn counts_by_base_name<'a>(
        names: impl IntoIterator<Item = &'a str>,
    ) -> BTreeMap<&'a str, usize> {
        let mut counts = BTreeMap::new();
        for name in names {
            *counts.entry(base_name(name)).or_default() += 1;
        }

        counts
    }

    fn emitted_ptx_names(ptx: &str) -> Vec<&str> {
        ptx.split(".visible .entry ")
            .skip(1)
            .filter_map(|entry| entry.split_once('(').map(|(name, _)| name.trim()))
            .collect()
    }

    #[test]
    fn inventory_has_no_missing_or_duplicate_entries() {
        let names = kernel_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        // Derived, never memorized: a free-standing total lagged the emitted
        // artifact across five merges (798 -> 847 -> 880 -> 883 -> 886) because
        // nothing tied it to the family whose entries moved.
        assert_eq!(
            names.len(),
            FAMILY_COUNTS.iter().map(|(_, count)| count).sum::<usize>()
        );
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn every_family_declares_its_pinned_entry_count() {
        let declared = FAMILIES
            .iter()
            .map(|(module, names)| (*module, names().len()))
            .collect::<Vec<_>>();
        let pinned = FAMILY_COUNTS.to_vec();

        assert_eq!(declared, pinned);
        assert_eq!(
            pinned.iter().map(|(_, count)| count).sum::<usize>(),
            kernel_ptx_names().len()
        );
    }

    /// Reconciles every family's inventory against the entries its own device
    /// module actually emitted, so a kernel cannot enter the build undeclared,
    /// a declaration cannot outlive its kernel, and an entry cannot silently
    /// migrate to another family's module. `gate_sm120_resources` runs it.
    #[test]
    #[ignore = "requires the pinned SM120 device build's PTX"]
    fn inventory_matches_emitted_entries() {
        for (module, names) in FAMILIES {
            let path = PathBuf::from(PTX_DIRECTORY).join(format!("{module}.ptx"));
            let ptx = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!("could not read {}: {error}", path.display());
            });
            let emitted = counts_by_base_name(emitted_ptx_names(&ptx));
            let declared = counts_by_base_name(names());
            let drift = declared
                .keys()
                .chain(emitted.keys())
                .map(|name| {
                    (
                        *name,
                        declared.get(name).copied(),
                        emitted.get(name).copied(),
                    )
                })
                .filter(|(_, declared, emitted)| declared != emitted)
                .collect::<BTreeSet<_>>();

            assert!(
                drift.is_empty(),
                "{module} and its emitted PTX disagree on (name, declared, emitted): {drift:?}"
            );
        }
    }
}

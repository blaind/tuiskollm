//! Byte-accounting and arena-bounds audits over `tuisko_engine::LayerMemoryLayout`.
//!
//! The trait reports four totals per owner and never constructs a layout or touches the device,
//! so every audit here is pure host arithmetic.

use tuisko_engine::LayerMemoryLayout;
use tuisko_gpu::ArenaRegion;

/// The four byte totals one owner reports through `LayerMemoryLayout`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayoutAudit {
    /// Complete allocation size including alignment padding.
    pub(crate) arena_bytes: usize,
    /// Source-backed resident device weight bytes.
    pub(crate) resident_weight_bytes: usize,
    /// Quantized key/value cache bytes.
    pub(crate) cache_bytes: usize,
    /// Address-stable activation and transient workspace bytes.
    pub(crate) workspace_bytes: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl LayoutAudit {
    /// Reads the four totals from one layout.
    pub(crate) fn read<L: LayerMemoryLayout + ?Sized>(layout: &L) -> Self {
        Self {
            arena_bytes: layout.arena_bytes(),
            resident_weight_bytes: layout.resident_weight_bytes(),
            cache_bytes: layout.cache_bytes(),
            workspace_bytes: layout.workspace_bytes(),
        }
    }

    /// Bytes the three named roles attribute.
    pub(crate) const fn attributed_bytes(self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Arena bytes no named role claims, or `None` when the attribution overruns the arena.
    pub(crate) const fn unattributed_bytes(self) -> Option<usize> {
        self.arena_bytes.checked_sub(self.attributed_bytes())
    }
}

/// Requires that one owner's arena covers everything its three named roles attribute.
///
/// This is an inequality and never an equality. For most owners the remainder is alignment
/// padding, but Qwen3.8's `ResidentModelLayout` also owns recurrent history, recurrent state, and
/// KV table bytes that the four trait accessors do not name; asserting equality would report that
/// single owner as an accounting defect.
///
/// An owner that allocates more than one arena while reporting only the first through
/// `arena_bytes` is rejected here — audit that owner against its complete allocation instead.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn require_arena_covers_attribution<L: LayerMemoryLayout + ?Sized>(
    name: &str,
    layout: &L,
) -> Result<LayoutAudit, String> {
    let audit = LayoutAudit::read(layout);
    if audit.unattributed_bytes().is_none() {
        return Err(format!(
            "{name} attributes {} bytes to weights, cache, and workspace but owns a {}-byte arena",
            audit.attributed_bytes(),
            audit.arena_bytes
        ));
    }

    Ok(audit)
}

/// One arena region erased to the bytes an audit compares.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegionSpan {
    /// Region name as the suite reserves it.
    pub(crate) name: &'static str,
    /// Byte offset from the arena base address.
    pub(crate) offset_bytes: usize,
    /// Bytes the region occupies.
    pub(crate) byte_len: usize,
    /// Alignment used when the region was reserved.
    pub(crate) alignment: usize,
}

/// Erases one typed region into an auditable span.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn span<T: Copy>(name: &'static str, region: ArenaRegion<T>) -> RegionSpan {
    RegionSpan {
        name,
        offset_bytes: region.offset_bytes(),
        byte_len: region.byte_len(),
        alignment: region.alignment(),
    }
}

/// Exact payload and alignment-padding totals for one arena partition.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArenaPartitionAudit {
    /// Bytes occupied by reported regions.
    pub(crate) payload_bytes: usize,
    /// Gaps required to align each reported region.
    pub(crate) padding_bytes: usize,
}

/// Requires every span to be `alignment`-aligned, inside `arena_bytes`, and pairwise disjoint.
///
/// `alignment` stays caller-supplied: it is the reserving suite's own contract, not a harness
/// default. `spans` is sorted in place by offset, matching the per-suite audits this replaces.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn require_spans_aligned_disjoint_and_bounded(
    spans: &mut [RegionSpan],
    alignment: usize,
    arena_bytes: usize,
) -> Result<(), String> {
    spans.sort_unstable_by_key(|span| span.offset_bytes);
    for span in spans.iter() {
        if !span.offset_bytes.is_multiple_of(alignment) {
            return Err(format!(
                "region {} begins at {}, which is not {alignment}-byte aligned",
                span.name, span.offset_bytes
            ));
        }
        let end = span
            .offset_bytes
            .checked_add(span.byte_len)
            .ok_or_else(|| format!("region {} overflows the address space", span.name))?;
        if end > arena_bytes {
            return Err(format!(
                "region {} ends at {end}, past the {arena_bytes}-byte arena",
                span.name
            ));
        }
    }
    for adjacent in spans.windows(2) {
        if adjacent[0].offset_bytes + adjacent[0].byte_len > adjacent[1].offset_bytes {
            return Err(format!(
                "regions {} and {} overlap",
                adjacent[0].name, adjacent[1].name
            ));
        }
    }

    Ok(())
}

/// Requires the reported regions and only their required alignment gaps to tile the arena.
///
/// A missing middle region leaves a gap larger than the next region's alignment requires; a
/// missing final region leaves an unclaimed tail. Both fail independently of suite byte totals.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn require_spans_exactly_tile_arena(
    spans: &mut [RegionSpan],
    arena_bytes: usize,
) -> Result<ArenaPartitionAudit, String> {
    spans.sort_unstable_by_key(|span| span.offset_bytes);
    let mut cursor = 0usize;
    let mut payload_bytes = 0usize;
    let mut padding_bytes = 0usize;

    for span in spans {
        if !span.alignment.is_power_of_two() {
            return Err(format!(
                "region {} has invalid alignment {}",
                span.name, span.alignment
            ));
        }
        let mask = span.alignment - 1;
        let expected_offset = cursor
            .checked_add(mask)
            .map(|value| value & !mask)
            .ok_or_else(|| format!("alignment before region {} overflows", span.name))?;
        if span.offset_bytes != expected_offset {
            return Err(format!(
                "region {} begins at {}, expected {} after aligning the preceding end {}",
                span.name, span.offset_bytes, expected_offset, cursor
            ));
        }
        let end = span
            .offset_bytes
            .checked_add(span.byte_len)
            .ok_or_else(|| format!("region {} overflows the address space", span.name))?;
        if end > arena_bytes {
            return Err(format!(
                "region {} ends at {end}, past the {arena_bytes}-byte arena",
                span.name
            ));
        }
        padding_bytes = padding_bytes
            .checked_add(span.offset_bytes - cursor)
            .ok_or_else(|| "arena padding total overflows".to_string())?;
        payload_bytes = payload_bytes
            .checked_add(span.byte_len)
            .ok_or_else(|| "arena payload total overflows".to_string())?;
        cursor = end;
    }

    if cursor != arena_bytes {
        return Err(format!(
            "reported regions end at {cursor}, leaving {} unclaimed arena bytes",
            arena_bytes - cursor
        ));
    }

    Ok(ArenaPartitionAudit {
        payload_bytes,
        padding_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        LayoutAudit, RegionSpan, require_arena_covers_attribution,
        require_spans_aligned_disjoint_and_bounded, require_spans_exactly_tile_arena, span,
    };
    use tuisko_engine::{
        DenseFp8GdnLayerLayout, DenseFp8MlpLayout, EndpointLayout, FullAttentionLayerLayout,
        KvCacheCodec, LayerMemoryLayout, MtpLayerLayout, MtpPromptPrimeLayout, Nvfp4MlpLayout,
        Qwen35FullAttentionLayerLayout, Qwen35GdnLayerLayout, Qwen35LongContextKvLayout,
        Qwen35MtpLayerLayout, Qwen35ResidentModelLayout, Qwen35ResidentMtpLayout,
        Qwen35TextEndpointLayout, Qwen36FullAttentionLayerLayout, Qwen36GdnMoeLayerLayout,
        Qwen36LongContextKvLayout, Qwen36MtpLayerLayout, Qwen36ResidentModelLayout,
        Qwen36TextEndpointLayout, ResidentModelLayout, ResidentMtpLayout, SharedPagedKvLayout,
    };
    use tuisko_gpu::ArenaLayout;
    use tuisko_model::{Qwen35_9B, Qwen38_27B};

    const ALIGNMENT: usize = 256;

    /// The one exported owner whose `arena_bytes` reports a single allocation while `cache_bytes`
    /// counts a second, separately allocated arena.
    const SPLIT_ARENA_OWNER: &str = "qwen38/resident_mtp";

    /// Every owner `tuisko-engine` exports through the trait, built with the arch its production
    /// program uses.
    fn exported_layouts() -> Vec<(&'static str, Box<dyn LayerMemoryLayout>)> {
        vec![
            (
                "qwen38/endpoint",
                Box::new(EndpointLayout::build::<Qwen38_27B>().unwrap()),
            ),
            (
                "qwen38/full_attention_layer",
                Box::new(FullAttentionLayerLayout::build::<Qwen38_27B>().unwrap()),
            ),
            (
                "qwen38/dense_fp8_gdn_layer",
                Box::new(DenseFp8GdnLayerLayout::build::<Qwen38_27B>().unwrap()),
            ),
            (
                "qwen38/dense_fp8_mlp",
                Box::new(DenseFp8MlpLayout::build::<Qwen38_27B>().unwrap()),
            ),
            (
                "qwen38/nvfp4_mlp",
                Box::new(Nvfp4MlpLayout::build::<Qwen35_9B>().unwrap()),
            ),
            (
                "qwen38/mtp_layer",
                Box::new(MtpLayerLayout::build::<Qwen38_27B>().unwrap()),
            ),
            (
                "qwen38/mtp_prompt_prime",
                Box::new(MtpPromptPrimeLayout::build().unwrap()),
            ),
            (
                "qwen38/shared_paged_kv/e4m3",
                Box::new(SharedPagedKvLayout::build(KvCacheCodec::E4m3).unwrap()),
            ),
            (
                "qwen38/shared_paged_kv/nvfp4_e2m1",
                Box::new(SharedPagedKvLayout::build(KvCacheCodec::Nvfp4E2m1).unwrap()),
            ),
            (
                "qwen38/resident_model",
                Box::new(ResidentModelLayout::build().unwrap()),
            ),
            (
                "qwen38/resident_mtp",
                Box::new(ResidentMtpLayout::build().unwrap()),
            ),
            (
                "qwen35/full_attention_layer",
                Box::new(Qwen35FullAttentionLayerLayout::build().unwrap()),
            ),
            (
                "qwen35/gdn_layer",
                Box::new(Qwen35GdnLayerLayout::build().unwrap()),
            ),
            (
                "qwen35/long_context_kv",
                Box::new(Qwen35LongContextKvLayout::build().unwrap()),
            ),
            (
                "qwen35/mtp_layer",
                Box::new(Qwen35MtpLayerLayout::build().unwrap()),
            ),
            (
                "qwen35/resident_mtp",
                Box::new(Qwen35ResidentMtpLayout::build().unwrap()),
            ),
            (
                "qwen35/text_endpoint",
                Box::new(Qwen35TextEndpointLayout::build().unwrap()),
            ),
            (
                "qwen35/resident_model",
                Box::new(Qwen35ResidentModelLayout::build().unwrap()),
            ),
            (
                "qwen36/full_attention_layer",
                Box::new(Qwen36FullAttentionLayerLayout::build().unwrap()),
            ),
            (
                "qwen36/gdn_moe_layer",
                Box::new(Qwen36GdnMoeLayerLayout::build().unwrap()),
            ),
            (
                "qwen36/long_context_kv",
                Box::new(Qwen36LongContextKvLayout::build().unwrap()),
            ),
            (
                "qwen36/mtp_layer",
                Box::new(Qwen36MtpLayerLayout::build().unwrap()),
            ),
            (
                "qwen36/text_endpoint",
                Box::new(Qwen36TextEndpointLayout::build().unwrap()),
            ),
            (
                "qwen36/resident_model",
                Box::new(Qwen36ResidentModelLayout::build().unwrap()),
            ),
        ]
    }

    #[test]
    fn every_exported_owner_arena_covers_its_attributed_bytes() {
        let layouts = exported_layouts();
        assert_eq!(layouts.len(), 24);
        let partial = layouts
            .iter()
            .filter(|(_, layout)| {
                LayoutAudit::read(layout.as_ref())
                    .unattributed_bytes()
                    .is_none()
            })
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        assert_eq!(partial, [SPLIT_ARENA_OWNER]);

        for (name, layout) in layouts
            .iter()
            .filter(|(name, _)| *name != SPLIT_ARENA_OWNER)
        {
            let audit = require_arena_covers_attribution(name, layout.as_ref()).unwrap();
            assert!(audit.arena_bytes > 0, "{name} owns an empty arena");
            assert_eq!(
                audit.unattributed_bytes().unwrap(),
                audit.arena_bytes - audit.attributed_bytes()
            );
        }
    }

    #[test]
    fn the_qwen38_resident_mtp_reports_only_its_first_allocation() {
        let layout = ResidentMtpLayout::build().unwrap();
        let audit = LayoutAudit::read(&layout);

        // This owner allocates two arenas and `arena_bytes` reports only the first, so the
        // covering invariant holds against both allocations rather than against `arena_bytes`
        // alone. Its Qwen3.5 sibling sums every sub-arena, so this is an engine-side reporting
        // asymmetry, not an accounting defect in the layout itself.
        assert_eq!(audit.unattributed_bytes(), None);
        assert_eq!(
            layout.arena_bytes() + layout.cache_arena_bytes() - audit.attributed_bytes(),
            layout.padding_bytes()
        );
    }

    #[test]
    fn a_single_role_owner_leaves_only_alignment_padding_unattributed() {
        let layout = FullAttentionLayerLayout::build::<Qwen38_27B>().unwrap();
        let audit =
            require_arena_covers_attribution("qwen38/full_attention_layer", &layout).unwrap();

        assert_eq!(audit.attributed_bytes(), layout.owner_bytes());
        assert_eq!(
            audit.unattributed_bytes().unwrap(),
            layout.arena_bytes() - layout.owner_bytes()
        );
    }

    #[test]
    fn the_qwen38_resident_model_owns_bytes_the_trait_does_not_name() {
        let layout = ResidentModelLayout::build().unwrap();
        let audit = require_arena_covers_attribution("qwen38/resident_model", &layout).unwrap();

        // The equality that holds for every other owner fails here: recurrent history, recurrent
        // state, and the KV tables are owned but unnamed by the four accessors.
        assert!(audit.attributed_bytes() < layout.owner_bytes());
        assert_eq!(
            audit.unattributed_bytes().unwrap(),
            layout.history_bytes()
                + layout.state_bytes()
                + layout.kv_table_bytes()
                + layout.padding_bytes()
        );
    }

    #[test]
    fn an_attribution_larger_than_the_arena_is_reported() {
        struct Overrun;

        impl LayerMemoryLayout for Overrun {
            fn arena_bytes(&self) -> usize {
                1_024
            }

            fn resident_weight_bytes(&self) -> usize {
                512
            }

            fn cache_bytes(&self) -> usize {
                512
            }

            fn workspace_bytes(&self) -> usize {
                256
            }
        }

        let audit = LayoutAudit::read(&Overrun);
        assert_eq!(audit.attributed_bytes(), 1_280);
        assert_eq!(audit.unattributed_bytes(), None);
        assert!(require_arena_covers_attribution("overrun", &Overrun).is_err());
    }

    #[test]
    fn reserved_regions_are_aligned_disjoint_and_inside_the_arena() {
        let mut builder = ArenaLayout::new();
        let input = builder.reserve::<u16>(1_024, ALIGNMENT).unwrap();
        let weight = builder.reserve::<u16>(96, ALIGNMENT).unwrap();
        let output = builder.reserve::<u16>(1_024, ALIGNMENT).unwrap();
        let mut spans = vec![
            span("output", output),
            span("input", input),
            span("weight", weight),
        ];

        require_spans_aligned_disjoint_and_bounded(&mut spans, ALIGNMENT, builder.byte_len())
            .unwrap();
        assert_eq!(spans[0].name, "input");
        assert_eq!(spans[2].name, "output");
        assert_eq!(
            require_spans_exactly_tile_arena(&mut spans, builder.byte_len()).unwrap(),
            super::ArenaPartitionAudit {
                payload_bytes: 4_288,
                padding_bytes: 64,
            }
        );
    }

    #[test]
    fn overlapping_misaligned_and_escaping_regions_are_reported() {
        let aligned = |name, offset_bytes, byte_len| RegionSpan {
            name,
            offset_bytes,
            byte_len,
            alignment: ALIGNMENT,
        };

        let mut misaligned = [aligned("head", 128, 128)];
        assert!(
            require_spans_aligned_disjoint_and_bounded(&mut misaligned, ALIGNMENT, 4_096).is_err()
        );

        let mut escaping = [aligned("tail", 3_840, 512)];
        assert!(
            require_spans_aligned_disjoint_and_bounded(&mut escaping, ALIGNMENT, 4_096).is_err()
        );

        let mut overlapping = [aligned("first", 0, 512), aligned("second", 256, 256)];
        assert!(
            require_spans_aligned_disjoint_and_bounded(&mut overlapping, ALIGNMENT, 4_096).is_err()
        );
    }

    #[test]
    fn exact_partition_rejects_missing_middle_and_final_regions() {
        let mut builder = ArenaLayout::new();
        let head = builder.reserve::<u8>(13, 1).unwrap();
        let middle = builder.reserve::<u32>(4, ALIGNMENT).unwrap();
        let tail = builder.reserve::<u16>(3, 8).unwrap();

        let mut complete = vec![
            span("tail", tail),
            span("head", head),
            span("middle", middle),
        ];
        assert_eq!(
            require_spans_exactly_tile_arena(&mut complete, builder.byte_len()).unwrap(),
            super::ArenaPartitionAudit {
                payload_bytes: 35,
                padding_bytes: 243,
            }
        );

        let mut missing_middle = vec![span("head", head), span("tail", tail)];
        assert!(require_spans_exactly_tile_arena(&mut missing_middle, builder.byte_len()).is_err());

        let mut missing_tail = vec![span("head", head), span("middle", middle)];
        assert!(require_spans_exactly_tile_arena(&mut missing_tail, builder.byte_len()).is_err());
    }
}

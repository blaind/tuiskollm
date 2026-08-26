//! Byte-accounting and arena-bounds audits over `tuisko_engine::LayerMemoryLayout`.
//!
//! The trait reports four totals per owner and never constructs a layout or touches the device,
//! so every audit here is pure host arithmetic.

use tuisko_engine::LayerMemoryLayout;
#[cfg(feature = "device")]
use tuisko_gpu::{ArenaLayout, ArenaRegion, DeviceCopy, GpuResult};

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

/// Arena byte totals derived while each region is reserved.
#[cfg(feature = "device")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArenaPartitionAudit {
    padding_bytes: usize,
    categories: Vec<(&'static str, usize)>,
}

#[cfg(feature = "device")]
impl ArenaPartitionAudit {
    /// Bytes inserted by the layout's alignment rules.
    pub(crate) const fn padding_bytes(&self) -> usize {
        self.padding_bytes
    }

    /// Bytes attributed to `category`.
    pub(crate) fn category_bytes(&self, category: &str) -> usize {
        self.categories
            .iter()
            .find(|(name, _)| *name == category)
            .map_or(0, |(_, bytes)| *bytes)
    }
}

/// A completed layout paired with its reservation-derived accounting.
#[cfg(feature = "device")]
#[derive(Debug)]
pub(crate) struct AccountedArenaLayout {
    layout: ArenaLayout,
    audit: ArenaPartitionAudit,
}

#[cfg(feature = "device")]
impl AccountedArenaLayout {
    /// Immutable layout used to allocate the arena.
    pub(crate) const fn layout(&self) -> &ArenaLayout {
        &self.layout
    }

    /// Consumes the sealed layout and returns its accounting.
    pub(crate) fn into_audit(self) -> ArenaPartitionAudit {
        self.audit
    }
}

/// Builds an arena while attributing every successful reservation.
///
/// The inner layout is never exposed mutably, so every reservation passes through `reserve`.
#[cfg(feature = "device")]
#[derive(Debug, Default)]
pub(crate) struct ArenaPartitionBuilder {
    layout: ArenaLayout,
    payload_bytes: usize,
    categories: Vec<(&'static str, usize)>,
}

#[cfg(feature = "device")]
impl ArenaPartitionBuilder {
    /// Starts an empty accounted layout.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Reserves one typed region and attributes its bytes to `category`.
    pub(crate) fn reserve<T: DeviceCopy>(
        &mut self,
        category: &'static str,
        len: usize,
        alignment: usize,
    ) -> GpuResult<ArenaRegion<T>> {
        let region = self.layout.reserve(len, alignment)?;
        self.payload_bytes += region.byte_len();
        match self
            .categories
            .iter_mut()
            .find(|(name, _)| *name == category)
        {
            Some((_, bytes)) => *bytes += region.byte_len(),
            None => self.categories.push((category, region.byte_len())),
        }
        Ok(region)
    }

    /// Finishes the layout and its reservation-derived accounting.
    pub(crate) fn finish(self) -> AccountedArenaLayout {
        let arena_bytes = self.layout.byte_len();
        let audit = ArenaPartitionAudit {
            padding_bytes: arena_bytes - self.payload_bytes,
            categories: self.categories,
        };
        AccountedArenaLayout {
            layout: self.layout,
            audit,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "device")]
    use super::ArenaPartitionBuilder;
    use super::{LayoutAudit, require_arena_covers_attribution};
    use tuisko_engine::{
        DenseFp8GdnLayerLayout, DenseFp8MlpLayout, EndpointLayout, FullAttentionLayerLayout,
        KvCacheCodec, LayerMemoryLayout, MtpLayerLayout, MtpPromptPrimeLayout, Nvfp4MlpLayout,
        Qwen35FullAttentionLayerLayout, Qwen35GdnLayerLayout, Qwen35LongContextKvLayout,
        Qwen35MtpLayerLayout, Qwen35ResidentModelLayout, Qwen35ResidentMtpLayout,
        Qwen35TextEndpointLayout, Qwen36FullAttentionLayerLayout, Qwen36GdnMoeLayerLayout,
        Qwen36LongContextKvLayout, Qwen36MtpLayerLayout, Qwen36ResidentModelLayout,
        Qwen36TextEndpointLayout, ResidentModelLayout, ResidentMtpLayout, SharedPagedKvLayout,
    };
    use tuisko_model::{Qwen35_9B, Qwen38_27B};

    #[cfg(feature = "device")]
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

    #[cfg(feature = "device")]
    #[test]
    fn every_reservation_contributes_to_one_category() {
        let mut builder = ArenaPartitionBuilder::new();
        let input = builder
            .reserve::<u16>("workspace", 1_024, ALIGNMENT)
            .unwrap();
        let weight = builder.reserve::<u16>("weights", 96, ALIGNMENT).unwrap();
        let output = builder
            .reserve::<u16>("workspace", 1_024, ALIGNMENT)
            .unwrap();
        let layout = builder.finish();
        let arena_bytes = layout.layout().byte_len();
        let audit = layout.into_audit();

        assert_eq!(input.offset_bytes(), 0);
        assert_eq!(weight.offset_bytes(), 2_048);
        assert_eq!(output.offset_bytes(), 2_304);
        assert_eq!(arena_bytes, 4_352);
        assert_eq!(audit.padding_bytes, 64);
        assert_eq!(audit.category_bytes("workspace"), 4_096);
        assert_eq!(audit.category_bytes("weights"), 192);
        assert_eq!(audit.categories, [("workspace", 4_096), ("weights", 192)]);
    }

    #[cfg(feature = "device")]
    #[test]
    fn a_reservation_hidden_inside_alignment_padding_is_accounted() {
        let mut builder = ArenaPartitionBuilder::new();
        builder.reserve::<u8>("workspace", 1, 1).unwrap();
        builder.reserve::<u8>("state", 1, 1).unwrap();
        builder.reserve::<u8>("weights", 1, ALIGNMENT).unwrap();
        let layout = builder.finish();
        let arena_bytes = layout.layout().byte_len();
        let audit = layout.into_audit();

        assert_eq!(arena_bytes, 257);
        assert_eq!(audit.padding_bytes, 254);
        assert_eq!(audit.category_bytes("state"), 1);
        assert_eq!(
            audit.category_bytes("workspace")
                + audit.category_bytes("state")
                + audit.category_bytes("weights"),
            3
        );
    }
}

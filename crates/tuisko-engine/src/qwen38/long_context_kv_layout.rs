//! Exact shared paged-KV ownership for the admitted long-context text geometry.
//!
//! The NVFP4 variant reserves the future represented planes only. Selecting that
//! layout does not admit an append or attention execution route.

use crate::common::math::{checked_sum, product};
use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen38_27B};

const ALIGNMENT: usize = 256;
const ATTENTION_LAYERS: usize = Qwen38_27B::LAYERS / Qwen38_27B::FULL_ATTENTION_INTERVAL;

/// Maximum admitted logical context for one request.
pub const MAX_CONTEXT_TOKENS: usize = 220_000;
/// Pages in the shared pool, including the final partially used logical page.
pub const LONG_CONTEXT_PHYSICAL_PAGES: usize = MAX_CONTEXT_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);

/// Exact represented formats prepared by the shared KV owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheCodec {
    /// One represented E4M3 byte per key or value element.
    E4m3,
    /// Packed E2M1 data with one represented E4M3 scale per 16 elements.
    Nvfp4E2m1,
}

/// Plane geometry for one exact KV represented format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheCodecDescriptor {
    data_bits: usize,
    scale_bits: usize,
    scale_group: Option<usize>,
}

impl KvCacheCodecDescriptor {
    /// Represented data bits per key or value element.
    pub const fn data_bits(self) -> usize {
        self.data_bits
    }

    /// Represented scale bits per scale element, or zero when no scale plane exists.
    pub const fn scale_bits(self) -> usize {
        self.scale_bits
    }

    /// Number of data elements represented by one scale.
    pub const fn scale_group(self) -> Option<usize> {
        self.scale_group
    }
}

impl KvCacheCodec {
    /// Exact data and separate-scale plane description.
    pub const fn descriptor(self) -> KvCacheCodecDescriptor {
        match self {
            Self::E4m3 => KvCacheCodecDescriptor {
                data_bits: 8,
                scale_bits: 0,
                scale_group: None,
            },
            Self::Nvfp4E2m1 => KvCacheCodecDescriptor {
                data_bits: 4,
                scale_bits: 8,
                scale_group: Some(16),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KvPlaneRegions {
    pub(crate) data: ArenaRegion<u8>,
    pub(crate) scales: Option<ArenaRegion<u8>>,
}

impl KvPlaneRegions {
    fn push_spans(self, spans: &mut Vec<(usize, usize)>) {
        spans.push((self.data.offset_bytes(), self.data.byte_len()));
        if let Some(scales) = self.scales {
            spans.push((scales.offset_bytes(), scales.byte_len()));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerKvRegions {
    pub(crate) key: KvPlaneRegions,
    pub(crate) value: KvPlaneRegions,
}

impl LayerKvRegions {
    fn push_spans(self, spans: &mut Vec<(usize, usize)>) {
        self.key.push_spans(spans);
        self.value.push_spans(spans);
    }
}

/// One address-stable shared page pool and per-slot page tables for all attention layers.
#[derive(Clone, Debug)]
pub struct SharedPagedKvLayout {
    builder: ArenaLayout,
    codec: KvCacheCodec,
    physical_pages: usize,
    block_tables: ArenaRegion<u32>,
    layers: Vec<LayerKvRegions>,
    data_bytes: usize,
    scale_bytes: usize,
}

impl SharedPagedKvLayout {
    /// Plans the full 220K shared pool for the selected exact represented format.
    pub fn build(codec: KvCacheCodec) -> EngineResult<Self> {
        Self::build_for_pages(codec, LONG_CONTEXT_PHYSICAL_PAGES)
    }

    fn build_for_pages(codec: KvCacheCodec, physical_pages: usize) -> EngineResult<Self> {
        if physical_pages > LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "shared KV page count {physical_pages} exceeds {LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }

        require_exact_geometry()?;
        let descriptor = codec.descriptor();
        let values_per_plane = product(
            "shared KV values per plane",
            physical_pages,
            values_per_page()?,
        )?;
        let data_bytes_per_plane = bit_plane_bytes(
            "shared KV data plane",
            values_per_plane,
            descriptor.data_bits,
        )?;
        let scale_bytes_per_plane = match descriptor.scale_group {
            Some(group) => {
                if !values_per_plane.is_multiple_of(group) {
                    return Err(EngineError::layout(format!(
                        "shared KV plane has {values_per_plane} values, not a multiple of scale group {group}"
                    )));
                }
                bit_plane_bytes(
                    "shared KV scale plane",
                    values_per_plane / group,
                    descriptor.scale_bits,
                )?
            }
            None => 0,
        };

        let mut builder = ArenaLayout::new();
        let block_tables = builder.reserve(
            product(
                "shared KV block-table entries",
                MAX_BATCH,
                LONG_CONTEXT_PHYSICAL_PAGES,
            )?,
            ALIGNMENT,
        )?;
        let mut layers = Vec::with_capacity(ATTENTION_LAYERS);
        for _ in 0..ATTENTION_LAYERS {
            layers.push(LayerKvRegions {
                key: reserve_plane(&mut builder, data_bytes_per_plane, scale_bytes_per_plane)?,
                value: reserve_plane(&mut builder, data_bytes_per_plane, scale_bytes_per_plane)?,
            });
        }

        let plane_count = product("shared KV data/scale plane count", ATTENTION_LAYERS, 2)?;
        let data_bytes = product(
            "shared KV represented data bytes",
            data_bytes_per_plane,
            plane_count,
        )?;
        let scale_bytes = product(
            "shared KV represented scale bytes",
            scale_bytes_per_plane,
            plane_count,
        )?;
        let layout = Self {
            builder,
            codec,
            physical_pages,
            block_tables,
            layers,
            data_bytes,
            scale_bytes,
        };
        layout.validate_regions()?;

        Ok(layout)
    }

    /// Exact represented format of the data and optional scale planes.
    pub const fn codec(&self) -> KvCacheCodec {
        self.codec
    }

    /// Number of physical pages shared across all active slots.
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    /// Allocated token positions, including final-page rounding.
    pub const fn rounded_token_capacity(&self) -> usize {
        self.physical_pages * ATTENTION_PAGE_SIZE
    }

    /// Per-slot page-table stride capable of addressing the entire shared pool.
    pub const fn block_table_stride(&self) -> usize {
        LONG_CONTEXT_PHYSICAL_PAGES
    }

    /// Exact page-table bytes across all eight stable slot rows.
    pub const fn block_table_bytes(&self) -> usize {
        self.block_tables.byte_len()
    }

    /// Represented key/value data-plane bytes across all attention layers.
    pub const fn data_bytes(&self) -> usize {
        self.data_bytes
    }

    /// Separate represented scale-plane bytes across all attention layers.
    pub const fn scale_bytes(&self) -> usize {
        self.scale_bytes
    }

    /// Key/value data and scale bytes without block tables or padding.
    pub const fn cache_bytes(&self) -> usize {
        self.data_bytes + self.scale_bytes
    }

    /// Block-table and represented cache bytes without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.block_table_bytes() + self.cache_bytes()
    }

    /// Complete single-allocation byte count, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Alignment bytes not owned by a typed page table or cache plane.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes() - self.owner_bytes()
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn block_tables(&self) -> ArenaRegion<u32> {
        self.block_tables
    }

    pub(crate) fn layers(&self) -> &[LayerKvRegions] {
        &self.layers
    }

    fn validate_regions(&self) -> EngineResult<()> {
        let mut spans = vec![(
            self.block_tables.offset_bytes(),
            self.block_tables.byte_len(),
        )];
        for layer in &self.layers {
            layer.push_spans(&mut spans);
        }
        spans.sort_unstable_by_key(|&(offset, _)| offset);

        for &(offset, bytes) in &spans {
            if !offset.is_multiple_of(ALIGNMENT) {
                return Err(EngineError::layout(format!(
                    "shared KV region offset {offset} is not {ALIGNMENT}-byte aligned"
                )));
            }
            let end = checked_sum("shared KV region end", offset, bytes)?;
            if end > self.arena_bytes() {
                return Err(EngineError::layout(format!(
                    "shared KV region {offset}..{end} exceeds arena {}",
                    self.arena_bytes()
                )));
            }
        }
        for pair in spans.windows(2) {
            let first_end = checked_sum("shared KV region end", pair[0].0, pair[0].1)?;
            if first_end > pair[1].0 {
                return Err(EngineError::layout(format!(
                    "shared KV regions {}..{first_end} and {}..{} overlap",
                    pair[0].0,
                    pair[1].0,
                    checked_sum("shared KV region end", pair[1].0, pair[1].1)?
                )));
            }
        }

        Ok(())
    }
}

/// Exact resident memory envelope and fair-share context for an active-slot count.
#[derive(Clone, Debug)]
pub struct ResidentKvCapacityPlan {
    layout: SharedPagedKvLayout,
    device_budget_bytes: usize,
    required_headroom_bytes: usize,
    fixed_resident_bytes: usize,
    active_slots: usize,
}

impl ResidentKvCapacityPlan {
    /// Shared page-pool layout selected by the capacity calculation.
    pub const fn layout(&self) -> &SharedPagedKvLayout {
        &self.layout
    }

    /// Exact owner budget supplied by product admission.
    pub const fn device_budget_bytes(&self) -> usize {
        self.device_budget_bytes
    }

    /// Bytes deliberately left outside exact resident owners.
    pub const fn required_headroom_bytes(&self) -> usize {
        self.required_headroom_bytes
    }

    /// Resident weight, GDN history/state, and workspace bytes outside this KV arena.
    pub const fn fixed_resident_bytes(&self) -> usize {
        self.fixed_resident_bytes
    }

    /// Number of simultaneously active stable slot rows used for fair-share admission.
    pub const fn active_slots(&self) -> usize {
        self.active_slots
    }

    /// Logical context guaranteed to every active slot under an even page split.
    pub const fn context_tokens_per_slot(&self) -> usize {
        let pages_per_slot = self.layout.physical_pages / self.active_slots;
        let rounded_tokens = pages_per_slot * ATTENTION_PAGE_SIZE;
        if rounded_tokens > MAX_CONTEXT_TOKENS {
            MAX_CONTEXT_TOKENS
        } else {
            rounded_tokens
        }
    }

    /// Exact bytes occupied by fixed owners and the selected KV arena.
    pub const fn accounted_resident_bytes(&self) -> usize {
        self.fixed_resident_bytes + self.layout.arena_bytes()
    }

    /// Remaining bytes after owners and required headroom.
    pub const fn spare_bytes(&self) -> usize {
        self.device_budget_bytes - self.required_headroom_bytes - self.accounted_resident_bytes()
    }
}

/// Selects the largest shared Qwen KV pool fitting one exact resident owner envelope.
///
/// `fixed_resident_bytes` includes weights, GDN history/state, workspaces, and their
/// alignment padding. `required_headroom_bytes` covers CUDA/driver allocations that
/// are observed but not owned by the resident arenas.
pub fn plan_resident_kv_capacity(
    codec: KvCacheCodec,
    device_budget_bytes: usize,
    required_headroom_bytes: usize,
    fixed_resident_bytes: usize,
    active_slots: usize,
) -> EngineResult<ResidentKvCapacityPlan> {
    if !(1..=MAX_BATCH).contains(&active_slots) {
        return Err(EngineError::layout(format!(
            "active slot count {active_slots} is outside 1..={MAX_BATCH}"
        )));
    }

    let non_cache_bytes = checked_sum(
        "resident fixed bytes plus headroom",
        fixed_resident_bytes,
        required_headroom_bytes,
    )?;
    let cache_budget = device_budget_bytes.checked_sub(non_cache_bytes).ok_or_else(|| {
        EngineError::layout(format!(
            "device budget {device_budget_bytes} is smaller than fixed resident bytes {fixed_resident_bytes} plus headroom {required_headroom_bytes}"
        ))
    })?;
    let empty_layout = SharedPagedKvLayout::build_for_pages(codec, 0)?;
    let bytes_per_page = cache_bytes_per_physical_page(codec)?;
    let page_bytes = cache_budget
        .checked_sub(empty_layout.arena_bytes())
        .ok_or_else(|| {
            EngineError::layout(format!(
                "KV budget {cache_budget} cannot hold the {}-byte block-table arena",
                empty_layout.arena_bytes()
            ))
        })?;
    let physical_pages = (page_bytes / bytes_per_page).min(LONG_CONTEXT_PHYSICAL_PAGES);
    if physical_pages < active_slots {
        return Err(EngineError::layout(format!(
            "KV budget admits {physical_pages} shared pages, fewer than {active_slots} active slots"
        )));
    }
    let layout = SharedPagedKvLayout::build_for_pages(codec, physical_pages)?;

    Ok(ResidentKvCapacityPlan {
        layout,
        device_budget_bytes,
        required_headroom_bytes,
        fixed_resident_bytes,
        active_slots,
    })
}

fn reserve_plane(
    builder: &mut ArenaLayout,
    data_bytes: usize,
    scale_bytes: usize,
) -> EngineResult<KvPlaneRegions> {
    Ok(KvPlaneRegions {
        data: builder.reserve(data_bytes, ALIGNMENT)?,
        scales: if scale_bytes == 0 {
            None
        } else {
            Some(builder.reserve(scale_bytes, ALIGNMENT)?)
        },
    })
}

fn values_per_page() -> EngineResult<usize> {
    product(
        "shared KV values per page",
        product(
            "shared KV page head-tokens",
            Qwen38_27B::NUM_KV_HEADS,
            ATTENTION_PAGE_SIZE,
        )?,
        Qwen38_27B::HEAD_DIM,
    )
}

fn cache_bytes_per_physical_page(codec: KvCacheCodec) -> EngineResult<usize> {
    let descriptor = codec.descriptor();
    let values = values_per_page()?;
    let data = bit_plane_bytes("shared KV page data", values, descriptor.data_bits)?;
    let scales = match descriptor.scale_group {
        Some(group) => bit_plane_bytes(
            "shared KV page scales",
            values / group,
            descriptor.scale_bits,
        )?,
        None => 0,
    };
    product(
        "shared KV bytes per physical page",
        checked_sum("shared KV data plus scales per page", data, scales)?,
        ATTENTION_LAYERS * 2,
    )
}

fn bit_plane_bytes(name: &str, elements: usize, bits: usize) -> EngineResult<usize> {
    let total_bits = product(name, elements, bits)?;
    if !total_bits.is_multiple_of(8) {
        return Err(EngineError::layout(format!(
            "{name} has {total_bits} bits, not a whole number of bytes"
        )));
    }

    Ok(total_bits / 8)
}

fn require_exact_geometry() -> EngineResult<()> {
    if Qwen38_27B::LAYERS != 64
        || Qwen38_27B::FULL_ATTENTION_INTERVAL != 4
        || ATTENTION_LAYERS != 16
        || Qwen38_27B::NUM_KV_HEADS != 4
        || Qwen38_27B::HEAD_DIM != 256
        || ATTENTION_PAGE_SIZE != 64
        || MAX_BATCH != 8
    {
        return Err(EngineError::layout(
            "shared KV layout requires exact 64-layer/16-attention-layer, 4-KV-head, 256-wide, page-64, B=1..8 geometry",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_LAYERS, KvCacheCodec, LONG_CONTEXT_PHYSICAL_PAGES, MAX_CONTEXT_TOKENS,
        SharedPagedKvLayout, plan_resident_kv_capacity,
    };

    const PAGE_VALUES: usize = 4 * 64 * 256;
    const PLANE_COUNT: usize = 16 * 2;
    const BLOCK_TABLE_BYTES: usize = 8 * 3_438 * size_of::<u32>();

    #[test]
    fn exact_codec_descriptors_name_separate_plane_geometry() {
        let fp8 = KvCacheCodec::E4m3.descriptor();
        assert_eq!((fp8.data_bits(), fp8.scale_bits()), (8, 0));
        assert_eq!(fp8.scale_group(), None);

        let nvfp4 = KvCacheCodec::Nvfp4E2m1.descriptor();
        assert_eq!((nvfp4.data_bits(), nvfp4.scale_bits()), (4, 8));
        assert_eq!(nvfp4.scale_group(), Some(16));
    }

    #[test]
    fn full_fp8_pool_matches_independent_represented_value_oracle() {
        let layout = SharedPagedKvLayout::build(KvCacheCodec::E4m3).unwrap();
        let values_per_plane = 3_438 * PAGE_VALUES;
        let data_bytes = values_per_plane * PLANE_COUNT;

        assert_eq!(ATTENTION_LAYERS, 16);
        assert_eq!(LONG_CONTEXT_PHYSICAL_PAGES, 3_438);
        assert_eq!(layout.rounded_token_capacity(), 220_032);
        assert_eq!(layout.block_table_stride(), 3_438);
        assert_eq!(layout.block_table_bytes(), BLOCK_TABLE_BYTES);
        assert_eq!(layout.data_bytes(), data_bytes);
        assert_eq!(layout.data_bytes(), 7_210_008_576);
        assert_eq!(layout.scale_bytes(), 0);
        assert_eq!(layout.owner_bytes(), 7_210_118_592);
        assert_eq!(layout.padding_bytes(), 64);
        assert_eq!(layout.arena_bytes(), 7_210_118_656);
    }

    #[test]
    fn full_nvfp4_pool_matches_independent_data_and_scale_oracle() {
        let layout = SharedPagedKvLayout::build(KvCacheCodec::Nvfp4E2m1).unwrap();
        let values_per_plane = 3_438 * PAGE_VALUES;
        let packed_data_bytes = values_per_plane / 2 * PLANE_COUNT;
        let scale_bytes = values_per_plane / 16 * PLANE_COUNT;

        assert_eq!(layout.data_bytes(), packed_data_bytes);
        assert_eq!(layout.data_bytes(), 3_605_004_288);
        assert_eq!(layout.scale_bytes(), scale_bytes);
        assert_eq!(layout.scale_bytes(), 450_625_536);
        assert_eq!(layout.cache_bytes(), 4_055_629_824);
        assert_eq!(layout.owner_bytes(), 4_055_739_840);
        assert_eq!(layout.padding_bytes(), 64);
        assert_eq!(layout.arena_bytes(), 4_055_739_904);
    }

    #[test]
    fn capacity_function_accounts_fixed_owners_headroom_and_all_slots_once() {
        let fixed = 19_500_000_000;
        let headroom = 1_000_000_000;
        let full_layout = SharedPagedKvLayout::build(KvCacheCodec::E4m3).unwrap();
        let budget = fixed + headroom + full_layout.arena_bytes();

        let singleton =
            plan_resident_kv_capacity(KvCacheCodec::E4m3, budget, headroom, fixed, 1).unwrap();
        assert_eq!(singleton.layout().physical_pages(), 3_438);
        assert_eq!(singleton.context_tokens_per_slot(), MAX_CONTEXT_TOKENS);
        assert_eq!(
            singleton.accounted_resident_bytes(),
            fixed + full_layout.arena_bytes()
        );
        assert_eq!(singleton.spare_bytes(), 0);

        let eight =
            plan_resident_kv_capacity(KvCacheCodec::E4m3, budget, headroom, fixed, 8).unwrap();
        assert_eq!(eight.layout().physical_pages(), 3_438);
        assert_eq!(eight.context_tokens_per_slot(), 27_456);
        assert_eq!(eight.active_slots(), 8);
    }

    #[test]
    fn capacity_function_selects_only_whole_shared_pages() {
        let empty = SharedPagedKvLayout::build_for_pages(KvCacheCodec::E4m3, 0).unwrap();
        let one_page_all_layers = PAGE_VALUES * PLANE_COUNT;
        let budget = empty.arena_bytes() + 9 * one_page_all_layers + one_page_all_layers - 1;
        let plan = plan_resident_kv_capacity(KvCacheCodec::E4m3, budget, 0, 0, 8).unwrap();

        assert_eq!(plan.layout().physical_pages(), 9);
        assert_eq!(plan.layout().rounded_token_capacity(), 576);
        assert_eq!(plan.context_tokens_per_slot(), 64);
        assert_eq!(plan.spare_bytes(), one_page_all_layers - 1);
    }

    #[test]
    fn capacity_function_refuses_invalid_or_unfunded_slot_inventory() {
        let error = plan_resident_kv_capacity(KvCacheCodec::E4m3, usize::MAX, 0, 0, 0).unwrap_err();
        assert!(error.to_string().contains("outside 1..=8"));

        let empty = SharedPagedKvLayout::build_for_pages(KvCacheCodec::E4m3, 0).unwrap();
        let error = plan_resident_kv_capacity(KvCacheCodec::E4m3, empty.arena_bytes(), 0, 0, 8)
            .unwrap_err();
        assert!(error.to_string().contains("fewer than 8 active slots"));
    }
}

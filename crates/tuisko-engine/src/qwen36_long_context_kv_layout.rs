//! Shared E4M3 KV ownership for the exact Qwen3.6 text geometry.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen36Moe35B};

const ALIGNMENT: usize = 256;
const ATTENTION_LAYERS: usize = Qwen36Moe35B::LAYERS / Qwen36Moe35B::FULL_ATTENTION_INTERVAL;

/// Maximum logical context admitted by the pinned Qwen3.6 config.
pub const QWEN36_MAX_CONTEXT_TOKENS: usize = Qwen36Moe35B::MAX_POSITION_EMBEDDINGS;
/// Physical pages in the complete shared Qwen3.6 pool.
pub const QWEN36_LONG_CONTEXT_PHYSICAL_PAGES: usize =
    QWEN36_MAX_CONTEXT_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36LayerKvRegions {
    pub(crate) key: ArenaRegion<u8>,
    pub(crate) value: ArenaRegion<u8>,
}

/// One address-stable E4M3 page pool shared by eight persistent slots.
#[derive(Clone, Debug)]
pub struct Qwen36LongContextKvLayout {
    builder: ArenaLayout,
    physical_pages: usize,
    block_tables: ArenaRegion<u32>,
    layers: Vec<Qwen36LayerKvRegions>,
    cache_bytes: usize,
}

impl Qwen36LongContextKvLayout {
    /// Plans the complete 262,144-position E4M3 KV owner.
    pub fn build() -> EngineResult<Self> {
        Self::build_for_pages(QWEN36_LONG_CONTEXT_PHYSICAL_PAGES)
    }

    pub(crate) fn build_for_pages(physical_pages: usize) -> EngineResult<Self> {
        if physical_pages > QWEN36_LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "Qwen3.6 shared KV page count {physical_pages} exceeds {QWEN36_LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        require_geometry()?;

        let values_per_plane = product(
            "Qwen3.6 shared KV values per plane",
            physical_pages,
            values_per_page()?,
        )?;
        let mut builder = ArenaLayout::new();
        let block_tables = builder.reserve(
            product(
                "Qwen3.6 shared KV block-table entries",
                MAX_BATCH,
                QWEN36_LONG_CONTEXT_PHYSICAL_PAGES,
            )?,
            ALIGNMENT,
        )?;
        let mut layers = Vec::with_capacity(ATTENTION_LAYERS);
        for _ in 0..ATTENTION_LAYERS {
            layers.push(Qwen36LayerKvRegions {
                key: builder.reserve(values_per_plane, ALIGNMENT)?,
                value: builder.reserve(values_per_plane, ALIGNMENT)?,
            });
        }
        let cache_bytes = product(
            "Qwen3.6 shared E4M3 KV bytes",
            values_per_plane,
            ATTENTION_LAYERS * 2,
        )?;
        let layout = Self {
            builder,
            physical_pages,
            block_tables,
            layers,
            cache_bytes,
        };
        layout.validate_regions()?;

        Ok(layout)
    }

    /// Physical pages shared across active and retained slots.
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    /// Allocated positions, including final-page rounding.
    pub const fn rounded_token_capacity(&self) -> usize {
        self.physical_pages * ATTENTION_PAGE_SIZE
    }

    /// Page-table columns reserved for every stable slot.
    pub const fn block_table_stride(&self) -> usize {
        QWEN36_LONG_CONTEXT_PHYSICAL_PAGES
    }

    /// Exact page-table bytes across all stable slots.
    pub const fn block_table_bytes(&self) -> usize {
        self.block_tables.byte_len()
    }

    /// Represented E4M3 key and value bytes across all attention layers.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Typed page-table and cache bytes without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.block_table_bytes() + self.cache_bytes
    }

    /// Complete single-allocation bytes including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Bytes not owned by a typed region.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes() - self.owner_bytes()
    }

    fn validate_regions(&self) -> EngineResult<()> {
        let mut spans = Vec::with_capacity(1 + self.layers.len() * 2);
        spans.push((
            self.block_tables.offset_bytes(),
            self.block_tables.byte_len(),
        ));
        for layer in &self.layers {
            spans.push((layer.key.offset_bytes(), layer.key.byte_len()));
            spans.push((layer.value.offset_bytes(), layer.value.byte_len()));
        }
        spans.sort_unstable_by_key(|&(offset, _)| offset);

        for &(offset, bytes) in &spans {
            if !offset.is_multiple_of(ALIGNMENT) {
                return Err(EngineError::layout(format!(
                    "Qwen3.6 shared KV region offset {offset} is not {ALIGNMENT}-byte aligned"
                )));
            }
            let end = sum("Qwen3.6 shared KV region end", offset, bytes)?;
            if end > self.arena_bytes() {
                return Err(EngineError::layout(format!(
                    "Qwen3.6 shared KV region {offset}..{end} exceeds arena {}",
                    self.arena_bytes()
                )));
            }
        }
        for pair in spans.windows(2) {
            let first_end = sum("Qwen3.6 shared KV region end", pair[0].0, pair[0].1)?;
            if first_end > pair[1].0 {
                return Err(EngineError::layout(format!(
                    "Qwen3.6 shared KV regions {}..{first_end} and {} overlap",
                    pair[0].0, pair[1].0
                )));
            }
        }

        Ok(())
    }
}

fn values_per_page() -> EngineResult<usize> {
    product(
        "Qwen3.6 shared KV values per page",
        product(
            "Qwen3.6 shared KV page head-tokens",
            Qwen36Moe35B::NUM_KV_HEADS,
            ATTENTION_PAGE_SIZE,
        )?,
        Qwen36Moe35B::HEAD_DIM,
    )
}

fn require_geometry() -> EngineResult<()> {
    if Qwen36Moe35B::LAYERS != 40
        || Qwen36Moe35B::FULL_ATTENTION_INTERVAL != 4
        || ATTENTION_LAYERS != 10
        || Qwen36Moe35B::NUM_KV_HEADS != 2
        || Qwen36Moe35B::HEAD_DIM != 256
        || ATTENTION_PAGE_SIZE != 64
        || MAX_BATCH != 8
    {
        return Err(EngineError::layout(
            "Qwen3.6 shared KV layout requires exact 40-layer/10-attention-layer, 2-KV-head, 256-wide, page-64, B=1..8 geometry",
        ));
    }

    Ok(())
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn sum(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_LAYERS, QWEN36_LONG_CONTEXT_PHYSICAL_PAGES, QWEN36_MAX_CONTEXT_TOKENS,
        Qwen36LongContextKvLayout,
    };

    const PAGE_VALUES: usize = 2 * 64 * 256;

    #[test]
    fn exact_full_pool_matches_an_independent_e4m3_oracle() {
        let layout = Qwen36LongContextKvLayout::build().unwrap();
        let plane_bytes = 4_096 * PAGE_VALUES;
        let cache_bytes = plane_bytes * 2 * 10;
        let block_table_bytes = 8 * 4_096 * size_of::<u32>();

        assert_eq!(ATTENTION_LAYERS, 10);
        assert_eq!(QWEN36_MAX_CONTEXT_TOKENS, 262_144);
        assert_eq!(QWEN36_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(layout.rounded_token_capacity(), 262_144);
        assert_eq!(layout.block_table_stride(), 4_096);
        assert_eq!(layout.block_table_bytes(), block_table_bytes);
        assert_eq!(layout.cache_bytes(), cache_bytes);
        assert_eq!(layout.cache_bytes(), 2_684_354_560);
        assert_eq!(layout.owner_bytes(), 2_684_485_632);
        assert_eq!(layout.padding_bytes(), 0);
        assert_eq!(layout.arena_bytes(), 2_684_485_632);
    }

    #[test]
    fn partial_pool_preserves_full_slot_table_stride() {
        let layout = Qwen36LongContextKvLayout::build_for_pages(9).unwrap();

        assert_eq!(layout.physical_pages(), 9);
        assert_eq!(layout.rounded_token_capacity(), 576);
        assert_eq!(layout.block_table_stride(), 4_096);
        assert_eq!(layout.block_table_bytes(), 131_072);
        assert_eq!(layout.cache_bytes(), 5_898_240);
    }

    #[test]
    fn rejects_more_than_the_pinned_context() {
        let error = Qwen36LongContextKvLayout::build_for_pages(4_097).unwrap_err();

        assert!(error.to_string().contains("exceeds 4096"));
    }
}

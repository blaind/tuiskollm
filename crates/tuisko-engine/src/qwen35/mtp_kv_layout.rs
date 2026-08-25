//! Long-context BF16 cache mirror for the exact Qwen3.5 MTP layer.

use crate::common::math::product;
use crate::common::paged_kv::{PagedKvLayout, PagedKvPlanes, sealed};
use crate::{
    EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES,
    QWEN35_MAX_CONTEXT_TOKENS,
};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;

/// One address-stable BF16 K/V mirror and eight stable table rows.
#[derive(Clone, Debug)]
pub(crate) struct Qwen35MtpKvLayout {
    builder: ArenaLayout,
    block_tables: ArenaRegion<u32>,
    planes: [PagedKvPlanes<u16>; 1],
}

impl Qwen35MtpKvLayout {
    /// Plans the complete 262,144-position MTP mirror.
    pub(crate) fn build() -> EngineResult<Self> {
        Self::build_for_pages(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES)
    }

    pub(crate) fn build_for_pages(physical_pages: usize) -> EngineResult<Self> {
        if physical_pages > QWEN35_LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP KV page count {physical_pages} exceeds {QWEN35_LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        let plane_values = product(
            "Qwen3.5 MTP KV plane values",
            physical_pages,
            page_values()?,
        )?;
        let mut builder = ArenaLayout::new();
        let block_tables = builder.reserve(
            product(
                "Qwen3.5 MTP block-table entries",
                MAX_BATCH,
                QWEN35_LONG_CONTEXT_PHYSICAL_PAGES,
            )?,
            ALIGNMENT,
        )?;
        let planes = [PagedKvPlanes {
            key: builder.reserve(plane_values, ALIGNMENT)?,
            value: builder.reserve(plane_values, ALIGNMENT)?,
        }];

        Ok(Self {
            builder,
            block_tables,
            planes,
        })
    }

    #[cfg(test)]
    fn physical_pages(&self) -> usize {
        self.planes[0].key.len() / page_values().unwrap()
    }

    /// Represented BF16 K/V bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.planes[0].key.byte_len() + self.planes[0].value.byte_len()
    }

    /// Device block-table bytes.
    pub const fn block_table_bytes(&self) -> usize {
        self.block_tables.byte_len()
    }

    /// Complete single-allocation bytes.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    #[cfg(test)]
    fn owner_bytes(&self) -> usize {
        self.cache_bytes() + self.block_table_bytes()
    }
}

impl sealed::Sealed for Qwen35MtpKvLayout {}

impl PagedKvLayout for Qwen35MtpKvLayout {
    type Value = u16;

    const NAME: &'static str = "Qwen3.5 MTP";
    const FULL_PHYSICAL_PAGES: usize = QWEN35_LONG_CONTEXT_PHYSICAL_PAGES;
    const MAX_CONTEXT_TOKENS: usize = QWEN35_MAX_CONTEXT_TOKENS;
    // The mirror writes only its accepted positions, so a reused page keeps the
    // previous owner's represented values unless the reservation clears it.
    const CLEARS_REUSED_PAGES: bool = true;

    fn build_for_pages(physical_pages: usize) -> EngineResult<Self> {
        Qwen35MtpKvLayout::build_for_pages(physical_pages)
    }

    fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    fn block_tables(&self) -> ArenaRegion<u32> {
        self.block_tables
    }

    fn planes(&self) -> &[PagedKvPlanes<u16>] {
        &self.planes
    }
}

impl LayerMemoryLayout for Qwen35MtpKvLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    // A paged cache owner carries no source-backed weights.
    fn resident_weight_bytes(&self) -> usize {
        0
    }

    fn cache_bytes(&self) -> usize {
        self.cache_bytes()
    }

    // Block tables are the owner's only address-stable non-cache bytes.
    fn workspace_bytes(&self) -> usize {
        self.block_table_bytes()
    }
}

fn page_values() -> EngineResult<usize> {
    product(
        "Qwen3.5 MTP KV page values",
        product(
            "Qwen3.5 MTP KV page head positions",
            Qwen35_9B::NUM_KV_HEADS,
            ATTENTION_PAGE_SIZE,
        )?,
        Qwen35_9B::HEAD_DIM,
    )
}

#[cfg(test)]
mod tests {
    use super::{PagedKvLayout, Qwen35MtpKvLayout, page_values};

    #[test]
    fn qwen35_mtp_full_mirror_accounting_is_exact() {
        let layout = Qwen35MtpKvLayout::build().unwrap();

        assert_eq!(page_values().unwrap(), 65_536);
        assert_eq!(layout.physical_pages(), 4_096);
        assert_eq!(Qwen35MtpKvLayout::FULL_PHYSICAL_PAGES, 4_096);
        assert_eq!(layout.cache_bytes(), 1_073_741_824);
        assert_eq!(layout.block_table_bytes(), 131_072);
        assert_eq!(layout.owner_bytes(), 1_073_872_896);
        assert_eq!(layout.arena_bytes(), 1_073_872_896);
    }
}

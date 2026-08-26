//! Long-context BF16 cache mirror for the exact Qwen3.5 MTP layer.

use crate::common::math::product;
use crate::{EngineError, EngineResult, MAX_BATCH, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35MtpKvRegions {
    pub(crate) block_tables: ArenaRegion<u32>,
    pub(crate) key_pages: ArenaRegion<u16>,
    pub(crate) value_pages: ArenaRegion<u16>,
}

/// One address-stable BF16 K/V mirror and eight stable table rows.
#[derive(Clone, Debug)]
pub(crate) struct Qwen35MtpKvLayout {
    builder: ArenaLayout,
    regions: Qwen35MtpKvRegions,
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
        let regions = Qwen35MtpKvRegions {
            block_tables: builder.reserve(
                product(
                    "Qwen3.5 MTP block-table entries",
                    MAX_BATCH,
                    QWEN35_LONG_CONTEXT_PHYSICAL_PAGES,
                )?,
                ALIGNMENT,
            )?,
            key_pages: builder.reserve(plane_values, ALIGNMENT)?,
            value_pages: builder.reserve(plane_values, ALIGNMENT)?,
        };

        Ok(Self { builder, regions })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn regions(&self) -> Qwen35MtpKvRegions {
        self.regions
    }

    #[cfg(test)]
    fn physical_pages(&self) -> usize {
        self.regions.key_pages.len() / page_values().unwrap()
    }

    /// Fixed columns in every stable block-table row.
    pub const fn table_stride(&self) -> usize {
        QWEN35_LONG_CONTEXT_PHYSICAL_PAGES
    }

    /// Represented BF16 K/V bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.regions.key_pages.byte_len() + self.regions.value_pages.byte_len()
    }

    /// Device block-table bytes.
    pub const fn block_table_bytes(&self) -> usize {
        self.regions.block_tables.byte_len()
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
    use super::{Qwen35MtpKvLayout, page_values};

    #[test]
    fn qwen35_mtp_full_mirror_accounting_is_exact() {
        let layout = Qwen35MtpKvLayout::build().unwrap();

        assert_eq!(page_values().unwrap(), 65_536);
        assert_eq!(layout.physical_pages(), 4_096);
        assert_eq!(layout.table_stride(), 4_096);
        assert_eq!(layout.cache_bytes(), 1_073_741_824);
        assert_eq!(layout.block_table_bytes(), 131_072);
        assert_eq!(layout.owner_bytes(), 1_073_872_896);
        assert_eq!(layout.arena_bytes(), 1_073_872_896);
    }
}

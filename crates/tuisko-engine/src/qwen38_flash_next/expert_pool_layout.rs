//! Address-stable expert slots and their expert-to-slot table.
//!
//! Captured graphs retain both addresses while table contents may change. Qualification reserves
//! every expert; production may share a smaller streaming pool.

use crate::common::math::product;
use crate::{EngineError, EngineResult};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES;
use tuisko_model::Qwen38FlashNext;

const ALIGNMENT: usize = 256;

/// The two planes an expert dispatch resolves its weights through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextExpertPoolRegions {
    /// Expert id to slot index, `NUM_EXPERTS` entries.
    pub(crate) slot_table: ArenaRegion<u32>,
    /// The sealed slot arena every resolved slot indexes into.
    pub(crate) slot_pool: ArenaRegion<u8>,
}

/// Checked slot-pool allocation for one composed layer's routed experts.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextExpertPoolLayout {
    builder: ArenaLayout,
    regions: Qwen38FlashNextExpertPoolRegions,
    slot_count: usize,
}

impl Qwen38FlashNextExpertPoolLayout {
    /// Reserves `slot_count` slots plus the full-width indirection table.
    ///
    /// `slot_count` may be smaller than `NUM_EXPERTS`; the caller keeps table entries in range.
    pub fn build(slot_count: usize) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        if slot_count == 0 || slot_count > A::NUM_EXPERTS {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next expert slot count {slot_count} is outside 1..={}",
                A::NUM_EXPERTS
            )));
        }

        let slot_pool = product(
            "Qwen3.8-Flash-Next expert slot pool",
            slot_count,
            QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES,
        )?;
        let mut builder = ArenaLayout::new();
        let regions = Qwen38FlashNextExpertPoolRegions {
            slot_table: builder.reserve(A::NUM_EXPERTS, ALIGNMENT)?,
            slot_pool: builder.reserve(slot_pool, ALIGNMENT)?,
        };

        Ok(Self {
            builder,
            regions,
            slot_count,
        })
    }

    /// Reserves one slot per routed expert, the posture a layer qualification needs.
    pub fn resident() -> EngineResult<Self> {
        Self::build(Qwen38FlashNext::NUM_EXPERTS)
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn regions(&self) -> Qwen38FlashNextExpertPoolRegions {
        self.regions
    }

    /// Slots this pool holds.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact routed-expert bytes this pool addresses.
    pub const fn slot_pool_bytes(&self) -> usize {
        self.regions.slot_pool.byte_len()
    }

    /// Exact indirection-table bytes.
    pub const fn table_bytes(&self) -> usize {
        self.regions.slot_table.byte_len()
    }
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, Qwen38FlashNextExpertPoolLayout};
    use tuisko_model::Qwen38FlashNext;

    type A = Qwen38FlashNext;

    #[test]
    fn a_resident_pool_holds_every_routed_expert() {
        let layout = Qwen38FlashNextExpertPoolLayout::resident().unwrap();

        assert_eq!(layout.slot_count(), A::NUM_EXPERTS);
        assert_eq!(QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, 2_764_800);
        assert_eq!(layout.slot_pool_bytes(), 1_415_577_600);
        assert_eq!(layout.table_bytes(), 2_048);
        assert_eq!(layout.arena_bytes(), 1_415_579_648);
    }

    #[test]
    fn the_table_stays_full_width_when_the_pool_is_not() {
        // The production posture: a cache far smaller than the expert count, addressed by a
        // table that still has one entry per expert.
        let layout = Qwen38FlashNextExpertPoolLayout::build(16).unwrap();

        assert_eq!(layout.slot_count(), 16);
        assert_eq!(layout.table_bytes(), A::NUM_EXPERTS * 4);
        assert_eq!(
            layout.slot_pool_bytes(),
            16 * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES
        );
    }

    #[test]
    fn regions_are_aligned_and_disjoint() {
        let layout = Qwen38FlashNextExpertPoolLayout::resident().unwrap();
        let regions = layout.regions();

        assert_eq!(regions.slot_table.offset_bytes() % ALIGNMENT, 0);
        assert_eq!(regions.slot_pool.offset_bytes() % ALIGNMENT, 0);
        assert!(
            regions.slot_table.offset_bytes() + regions.slot_table.byte_len()
                <= regions.slot_pool.offset_bytes()
        );
        assert!(
            regions.slot_pool.offset_bytes() + regions.slot_pool.byte_len() <= layout.arena_bytes()
        );
    }

    #[test]
    fn an_impossible_slot_count_is_refused() {
        assert!(Qwen38FlashNextExpertPoolLayout::build(0).is_err());
        assert!(Qwen38FlashNextExpertPoolLayout::build(A::NUM_EXPERTS + 1).is_err());
        assert!(Qwen38FlashNextExpertPoolLayout::build(A::NUM_EXPERTS).is_ok());
    }
}

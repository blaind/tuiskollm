//! Shared page ownership behind eight Qwen3.8 Flash-Next block-table rows.
//!
//! Attention masks positions at or beyond the committed length, so recycled page tails are
//! unreachable and need no scrub. Lifecycle operations update only changed table entries.

use crate::qwen38_flash_next::qsa_moe_layer_layout::QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;
use crate::qwen38_flash_next::resident_model_layout::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES;
use crate::{EngineError, EngineResult, MAX_BATCH};
use std::ops::Range;

/// Sentinel for an unowned block-table entry.
pub const QWEN38_FLASH_NEXT_UNMAPPED_PAGE: u32 = u32::MAX;

/// Tokens one physical Flash-Next attention page holds.
pub const QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS: usize = QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;

/// Where one slot is in its lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextSlotState {
    /// Owns no pages and carries no sequence.
    Free,
    /// Owns pages and is serving a request.
    Active,
    /// Owns pages and carries a completed sequence's prefix, held for reuse.
    Retained,
}

/// One slot's page ownership and committed length.
#[derive(Clone, Debug)]
struct Qwen38FlashNextSlot {
    state: Qwen38FlashNextSlotState,
    /// Changes whenever prior sequence snapshots become invalid.
    sequence: u64,
    /// Physical page ids in logical order; entry `p` maps logical page `p`.
    pages: Vec<u32>,
    /// Tokens this slot's sequence has committed to its cache.
    tokens: usize,
    /// Block-table entries whose device copy no longer matches `pages`.
    dirty: Option<Range<usize>>,
}

impl Qwen38FlashNextSlot {
    const fn new() -> Self {
        Self {
            state: Qwen38FlashNextSlotState::Free,
            sequence: 0,
            pages: Vec::new(),
            tokens: 0,
            dirty: None,
        }
    }

    fn mark(&mut self, entries: Range<usize>) {
        if entries.is_empty() {
            return;
        }
        self.dirty = Some(match self.dirty.take() {
            Some(existing) => existing.start.min(entries.start)..existing.end.max(entries.end),
            None => entries,
        });
    }
}

/// Mapping changes produced by one lifecycle call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextSlotChange {
    /// Pages taken from the shared free list.
    pub acquired_pages: usize,
    /// Pages returned to the shared free list.
    pub released_pages: usize,
    /// Tokens the slot's cache covers after the call.
    pub tokens: usize,
}

/// Pages and retained-prefix metadata released by recycling a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextSlotRelease {
    /// Pages returned to the shared free list.
    pub released_pages: usize,
    /// Tokens the released retained prefix covered, zero unless it was retained.
    pub retained_tokens: usize,
}

/// Eight block-table rows drawing from one shared pool of funded pages.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextSlotPool {
    /// Free physical page ids. Popped from the end, so a recycled page is reused first.
    free: Vec<u32>,
    slots: [Qwen38FlashNextSlot; MAX_BATCH],
    /// Host mirror of the whole `[MAX_BATCH, 4096]` device block table.
    table: Vec<u32>,
    funded_pages: usize,
}

impl Qwen38FlashNextSlotPool {
    /// Builds an empty pool over `funded_pages` physical pages.
    pub fn new(funded_pages: usize) -> EngineResult<Self> {
        if funded_pages == 0 {
            return Err(EngineError::layout(
                "the Flash-Next slot pool needs at least one funded page",
            ));
        }
        if funded_pages > MAX_BATCH * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "the Flash-Next slot pool was funded {funded_pages} pages, more than the \
                 {} the block table can address",
                MAX_BATCH * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
            )));
        }

        Ok(Self {
            // Pop page zero first for deterministic mappings.
            free: (0..funded_pages as u32).rev().collect(),
            slots: [const { Qwen38FlashNextSlot::new() }; MAX_BATCH],
            table: vec![
                QWEN38_FLASH_NEXT_UNMAPPED_PAGE;
                MAX_BATCH * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
            ],
            funded_pages,
        })
    }

    /// Physical pages the plan funded.
    pub const fn funded_pages(&self) -> usize {
        self.funded_pages
    }

    /// Pages no slot currently owns.
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// Pages every slot owns together.
    pub fn allocated_pages(&self) -> usize {
        self.funded_pages - self.free.len()
    }

    /// Longest sequence one slot could reach if it took every free page it does not already own.
    pub fn reachable_tokens(&self, slot: usize) -> EngineResult<usize> {
        let owned = self.slot(slot)?.pages.len();
        let addressable =
            QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES.min(owned + self.free.len());

        Ok(addressable * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS)
    }

    /// Lifecycle position of one slot.
    pub fn state(&self, slot: usize) -> EngineResult<Qwen38FlashNextSlotState> {
        Ok(self.slot(slot)?.state)
    }

    /// Tokens one slot's cache currently covers.
    pub fn tokens(&self, slot: usize) -> EngineResult<usize> {
        Ok(self.slot(slot)?.tokens)
    }

    /// Physical pages one slot owns, in logical order.
    pub fn pages(&self, slot: usize) -> EngineResult<&[u32]> {
        Ok(&self.slot(slot)?.pages)
    }

    /// The host mirror of one slot's block-table row.
    pub fn table_row(&self, slot: usize) -> EngineResult<&[u32]> {
        self.require_slot(slot)?;
        let begin = slot * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES;

        Ok(&self.table[begin..begin + QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES])
    }

    /// Grows one slot's mapping atomically to cover `tokens`.
    pub fn reserve(
        &mut self,
        slot: usize,
        tokens: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        self.require_slot(slot)?;
        let committed = self.slots[slot].tokens;
        if tokens < committed {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} cannot reserve {tokens} tokens below its {committed} \
                 committed tokens; truncate with a matching recurrent snapshot first"
            )));
        }
        let needed = pages_for(tokens)?;
        if needed > QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::route(format!(
                "a Flash-Next slot reservation of {tokens} tokens needs {needed} pages, more \
                 than the {QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES} one block-table row addresses"
            )));
        }
        let owned = self.slots[slot].pages.len();
        if needed > owned + self.free.len() {
            return Err(EngineError::capacity(format!(
                "a Flash-Next slot reservation of {tokens} tokens needs {needed} pages; slot \
                 {slot} owns {owned} and the shared pool has {} free, so the request is refused \
                 rather than served short",
                self.free.len()
            )));
        }
        if self.slots[slot].state == Qwen38FlashNextSlotState::Free {
            self.advance_sequence(slot)?;
        }

        let acquired = needed.saturating_sub(owned);
        for logical in owned..needed {
            let page = self
                .free
                .pop()
                .ok_or_else(|| EngineError::layout("the Flash-Next page pool underflowed"))?;
            self.slots[slot].pages.push(page);
            self.table[slot * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES + logical] = page;
        }
        if acquired != 0 {
            self.slots[slot].mark(owned..needed);
        }
        self.slots[slot].state = Qwen38FlashNextSlotState::Active;

        Ok(Qwen38FlashNextSlotChange {
            acquired_pages: acquired,
            released_pages: 0,
            tokens: self.slots[slot].tokens,
        })
    }

    /// Records that a slot's sequence now covers `tokens`.
    pub fn commit(&mut self, slot: usize, tokens: usize) -> EngineResult<()> {
        self.require_slot(slot)?;
        if self.slots[slot].state != Qwen38FlashNextSlotState::Active {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} cannot commit while {:?}",
                self.slots[slot].state
            )));
        }
        if tokens < self.slots[slot].tokens {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} cannot commit {tokens} tokens below its existing {}",
                self.slots[slot].tokens
            )));
        }
        let mapped = self.slots[slot].pages.len() * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS;
        if tokens > mapped {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} committed {tokens} tokens against {mapped} mapped, so a \
                 round would have attended through an unmapped page"
            )));
        }
        self.slots[slot].tokens = tokens;
        self.slots[slot].state = Qwen38FlashNextSlotState::Active;

        Ok(())
    }

    /// Truncates append-only cache state and returns released pages.
    pub fn truncate(
        &mut self,
        slot: usize,
        tokens: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        self.require_slot(slot)?;
        if tokens > self.slots[slot].tokens {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} cannot truncate to {tokens} tokens from {}, which would \
                 extend rather than roll back",
                self.slots[slot].tokens
            )));
        }
        let needed = pages_for(tokens)?;
        let owned = self.slots[slot].pages.len();
        if tokens < self.slots[slot].tokens {
            self.advance_sequence(slot)?;
        }
        self.truncate_mapping(slot, tokens, needed, owned)
    }

    /// Rolls back within one sequence without invalidating its snapshot epoch.
    pub(crate) fn rollback(
        &mut self,
        slot: usize,
        tokens: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        self.require_slot(slot)?;
        if tokens > self.slots[slot].tokens {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} cannot roll back to {tokens} tokens from {}",
                self.slots[slot].tokens
            )));
        }
        let needed = pages_for(tokens)?;
        let owned = self.slots[slot].pages.len();
        self.truncate_mapping(slot, tokens, needed, owned)
    }

    fn truncate_mapping(
        &mut self,
        slot: usize,
        tokens: usize,
        needed: usize,
        owned: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        let released = owned.saturating_sub(needed);
        for logical in (needed..owned).rev() {
            let page = self.slots[slot]
                .pages
                .pop()
                .ok_or_else(|| EngineError::layout("the Flash-Next page pool underflowed"))?;
            self.table[slot * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES + logical] =
                QWEN38_FLASH_NEXT_UNMAPPED_PAGE;
            self.free.push(page);
        }
        if released != 0 {
            self.slots[slot].mark(needed..owned);
        }
        self.slots[slot].tokens = tokens;

        Ok(Qwen38FlashNextSlotChange {
            acquired_pages: 0,
            released_pages: released,
            tokens,
        })
    }

    /// Holds one slot's committed prefix after its request finishes.
    pub fn retain(&mut self, slot: usize) -> EngineResult<usize> {
        self.require_slot(slot)?;
        if self.slots[slot].tokens == 0 {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} has no committed tokens to retain"
            )));
        }
        self.slots[slot].state = Qwen38FlashNextSlotState::Retained;

        Ok(self.slots[slot].tokens)
    }

    /// Restarts one logical sequence without releasing its reserved pages.
    pub(crate) fn restart(&mut self, slot: usize) -> EngineResult<usize> {
        self.require_slot(slot)?;
        let tokens = self.slots[slot].tokens;
        if self.slots[slot].state != Qwen38FlashNextSlotState::Free {
            self.advance_sequence(slot)?;
            self.slots[slot].state = Qwen38FlashNextSlotState::Active;
        }
        self.slots[slot].tokens = 0;

        Ok(tokens)
    }

    /// Returns every page one slot owns to the shared pool and clears its sequence.
    pub fn recycle(&mut self, slot: usize) -> EngineResult<Qwen38FlashNextSlotRelease> {
        self.require_slot(slot)?;
        let retained = match self.slots[slot].state {
            Qwen38FlashNextSlotState::Retained => self.slots[slot].tokens,
            Qwen38FlashNextSlotState::Active | Qwen38FlashNextSlotState::Free => 0,
        };
        if self.slots[slot].state != Qwen38FlashNextSlotState::Free {
            self.advance_sequence(slot)?;
        }
        let owned = self.slots[slot].pages.len();
        let released = self.truncate_to_zero(slot)?;
        debug_assert_eq!(released, owned);
        self.slots[slot].state = Qwen38FlashNextSlotState::Free;

        Ok(Qwen38FlashNextSlotRelease {
            released_pages: released,
            retained_tokens: retained,
        })
    }

    /// Longest common prefix of retained and requested token ids.
    pub fn reusable_prefix(committed: &[u32], prompt: &[u32]) -> usize {
        committed
            .iter()
            .zip(prompt)
            .take_while(|(left, right)| left == right)
            .count()
    }

    /// Block-table entries one slot needs uploaded.
    pub(crate) fn dirty_range(&self, slot: usize) -> EngineResult<Option<Range<usize>>> {
        self.require_slot(slot)?;

        Ok(self.slots[slot].dirty.clone())
    }

    /// Sequence epoch used to bind recurrent snapshots to their K/V prefix.
    pub(crate) fn sequence(&self, slot: usize) -> EngineResult<u64> {
        Ok(self.slot(slot)?.sequence)
    }

    /// Clears a dirty range after its upload was accepted.
    pub(crate) fn clear_dirty(&mut self, slot: usize) -> EngineResult<()> {
        self.require_slot(slot)?;
        self.slots[slot].dirty = None;

        Ok(())
    }

    /// Whether any slot's mapping has moved since its last flush.
    pub fn has_dirty(&self) -> bool {
        self.slots.iter().any(|slot| slot.dirty.is_some())
    }

    /// Clears every slot and returns the pool to the state `new` built.
    pub fn reset(&mut self) -> EngineResult<()> {
        if self
            .slots
            .iter()
            .any(|slot| slot.state != Qwen38FlashNextSlotState::Free && slot.sequence == u64::MAX)
        {
            return Err(EngineError::layout(
                "a Flash-Next slot sequence epoch overflowed",
            ));
        }
        for slot in 0..MAX_BATCH {
            if self.slots[slot].state != Qwen38FlashNextSlotState::Free {
                self.slots[slot].sequence += 1;
            }
            self.truncate_to_zero(slot)?;
            self.slots[slot].state = Qwen38FlashNextSlotState::Free;
        }
        self.free = (0..self.funded_pages as u32).rev().collect();
        self.table.fill(QWEN38_FLASH_NEXT_UNMAPPED_PAGE);
        for slot in &mut self.slots {
            slot.dirty = Some(0..QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES);
        }

        Ok(())
    }

    fn advance_sequence(&mut self, slot: usize) -> EngineResult<()> {
        self.slots[slot].sequence = self.slots[slot].sequence.checked_add(1).ok_or_else(|| {
            EngineError::layout(format!("Flash-Next slot {slot} sequence epoch overflowed"))
        })?;

        Ok(())
    }

    fn truncate_to_zero(&mut self, slot: usize) -> EngineResult<usize> {
        self.slots[slot].tokens = 0;
        let owned = self.slots[slot].pages.len();
        for logical in (0..owned).rev() {
            let page = self.slots[slot]
                .pages
                .pop()
                .ok_or_else(|| EngineError::layout("the Flash-Next page pool underflowed"))?;
            self.table[slot * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES + logical] =
                QWEN38_FLASH_NEXT_UNMAPPED_PAGE;
            self.free.push(page);
        }
        if owned != 0 {
            self.slots[slot].mark(0..owned);
        }

        Ok(owned)
    }

    fn slot(&self, slot: usize) -> EngineResult<&Qwen38FlashNextSlot> {
        self.slots.get(slot).ok_or_else(|| {
            EngineError::route(format!("Flash-Next slot {slot} is outside 0..{MAX_BATCH}"))
        })
    }

    fn require_slot(&self, slot: usize) -> EngineResult<()> {
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }

        Ok(())
    }
}

/// Pages a sequence of `tokens` needs, rounding the partial last page up.
pub fn pages_for(tokens: usize) -> EngineResult<usize> {
    tokens
        .checked_add(QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS - 1)
        .map(|rounded| rounded / QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS)
        .ok_or_else(|| EngineError::route("a Flash-Next page count overflows its token count"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineErrorCode;

    const PAGES: usize = 3_672;

    #[test]
    fn a_page_holds_sixty_four_tokens_and_partial_pages_round_up() {
        assert_eq!(QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS, 64);
        assert_eq!(pages_for(0).unwrap(), 0);
        assert_eq!(pages_for(1).unwrap(), 1);
        assert_eq!(pages_for(64).unwrap(), 1);
        assert_eq!(pages_for(65).unwrap(), 2);
        assert_eq!(pages_for(2_051).unwrap(), 33);
    }

    #[test]
    fn a_fresh_pool_owns_every_funded_page_and_no_slot_owns_one() {
        let pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();

        assert_eq!(pool.funded_pages(), PAGES);
        assert_eq!(pool.free_pages(), PAGES);
        assert_eq!(pool.allocated_pages(), 0);
        for slot in 0..MAX_BATCH {
            assert_eq!(pool.state(slot).unwrap(), Qwen38FlashNextSlotState::Free);
            assert_eq!(pool.tokens(slot).unwrap(), 0);
            assert!(pool.pages(slot).unwrap().is_empty());
            assert!(
                pool.table_row(slot)
                    .unwrap()
                    .iter()
                    .all(|&entry| entry == QWEN38_FLASH_NEXT_UNMAPPED_PAGE)
            );
        }
    }

    #[test]
    fn one_slot_can_borrow_the_whole_pool_which_the_partition_forbade() {
        // The identity partition gave every slot 459 pages and no way to exceed them. A single
        // request that needs more than an eighth of the cache is the case the partition could
        // not serve at all, so it is the case this test pins.
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        let partition = PAGES / MAX_BATCH;
        assert_eq!(partition, 459);

        let tokens = (partition + 1) * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS;
        let change = pool.reserve(0, tokens).unwrap();

        assert_eq!(change.acquired_pages, partition + 1);
        assert_eq!(pool.pages(0).unwrap().len(), partition + 1);
        assert_eq!(pool.free_pages(), PAGES - partition - 1);
    }

    #[test]
    fn a_reservation_the_pool_cannot_satisfy_is_refused_without_taking_a_page() {
        let mut pool = Qwen38FlashNextSlotPool::new(64).unwrap();
        pool.reserve(0, 60 * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS)
            .unwrap();
        let free_before = pool.free_pages();

        let error = pool
            .reserve(1, 8 * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("refused rather than served short"),
            "{error}"
        );
        assert_eq!(error.code(), Some(EngineErrorCode::Capacity));
        assert_eq!(pool.free_pages(), free_before);
        assert!(pool.pages(1).unwrap().is_empty());
    }

    #[test]
    fn a_reservation_cannot_discard_committed_tokens() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 200).unwrap();
        pool.commit(0, 200).unwrap();
        let pages = pool.pages(0).unwrap().to_vec();
        let free = pool.free_pages();

        let error = pool.reserve(0, 128).unwrap_err().to_string();

        assert!(error.contains("below its 200 committed tokens"), "{error}");
        assert_eq!(pool.tokens(0).unwrap(), 200);
        assert_eq!(pool.pages(0).unwrap(), pages);
        assert_eq!(pool.free_pages(), free);
    }

    #[test]
    fn truncation_returns_exactly_the_pages_that_fell_away_and_keeps_the_rest() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 640).unwrap();
        pool.commit(0, 640).unwrap();
        let kept = pool.pages(0).unwrap()[..3].to_vec();

        let change = pool.truncate(0, 129).unwrap();

        assert_eq!(change.released_pages, 10 - 3);
        assert_eq!(change.tokens, 129);
        assert_eq!(pool.pages(0).unwrap(), kept);
        assert_eq!(pool.free_pages(), PAGES - 3);
    }

    #[test]
    fn truncation_never_extends() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 640).unwrap();
        pool.commit(0, 100).unwrap();

        let error = pool.truncate(0, 101).unwrap_err().to_string();

        assert!(error.contains("extend rather than roll back"), "{error}");
    }

    #[test]
    fn a_recycled_page_is_handed_out_again_before_an_untouched_one() {
        // Reuse-first is what makes a long-running server touch a bounded set of pages rather
        // than walking the whole pool, and it is also what makes the stale-tail argument worth
        // gating: the next sequence lands on bytes the previous one wrote.
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 128).unwrap();
        let first = pool.pages(0).unwrap().to_vec();
        pool.commit(0, 128).unwrap();
        pool.recycle(0).unwrap();

        pool.reserve(1, 128).unwrap();

        assert_eq!(pool.pages(1).unwrap(), first);
    }

    #[test]
    fn recycling_returns_every_page_and_reports_a_retained_prefix_only_when_retained() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 200).unwrap();
        pool.commit(0, 200).unwrap();

        let plain = pool.recycle(0).unwrap();
        assert_eq!(plain.released_pages, 4);
        assert_eq!(plain.retained_tokens, 0);
        assert_eq!(pool.free_pages(), PAGES);

        pool.reserve(0, 200).unwrap();
        pool.commit(0, 200).unwrap();
        assert_eq!(pool.retain(0).unwrap(), 200);
        assert_eq!(pool.state(0).unwrap(), Qwen38FlashNextSlotState::Retained);

        let held = pool.recycle(0).unwrap();
        assert_eq!(held.retained_tokens, 200);
        assert_eq!(pool.free_pages(), PAGES);
        assert_eq!(pool.state(0).unwrap(), Qwen38FlashNextSlotState::Free);
    }

    #[test]
    fn a_slots_table_row_maps_its_pages_and_leaves_the_rest_unmapped() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(3, 130).unwrap();
        let pages = pool.pages(3).unwrap().to_vec();
        let row = pool.table_row(3).unwrap();

        assert_eq!(pages.len(), 3);
        assert_eq!(&row[..3], pages.as_slice());
        assert!(
            row[3..]
                .iter()
                .all(|&entry| entry == QWEN38_FLASH_NEXT_UNMAPPED_PAGE)
        );
    }

    #[test]
    fn only_the_entries_that_moved_are_marked_dirty() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 128).unwrap();
        assert_eq!(pool.dirty_range(0).unwrap(), Some(0..2));
        assert_eq!(pool.dirty_range(0).unwrap(), Some(0..2));
        pool.clear_dirty(0).unwrap();
        assert_eq!(pool.dirty_range(0).unwrap(), None);

        pool.reserve(0, 256).unwrap();
        assert_eq!(pool.dirty_range(0).unwrap(), Some(2..4));
        pool.clear_dirty(0).unwrap();

        pool.reserve(0, 200).unwrap();
        assert_eq!(pool.dirty_range(0).unwrap(), None);
    }

    #[test]
    fn a_committed_length_can_never_exceed_the_mapping() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 128).unwrap();

        let error = pool.commit(0, 129).unwrap_err().to_string();

        assert!(error.contains("unmapped page"), "{error}");
    }

    #[test]
    fn commits_are_active_and_append_only() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        assert!(pool.commit(0, 0).is_err());
        pool.reserve(0, 128).unwrap();
        pool.commit(0, 100).unwrap();

        let error = pool.commit(0, 99).unwrap_err().to_string();

        assert!(error.contains("below its existing 100"), "{error}");
        assert_eq!(pool.tokens(0).unwrap(), 100);
    }

    #[test]
    fn restart_keeps_pages_but_clears_the_logical_sequence() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 128).unwrap();
        pool.commit(0, 100).unwrap();
        let pages = pool.pages(0).unwrap().to_vec();

        assert_eq!(pool.restart(0).unwrap(), 100);
        assert_eq!(pool.tokens(0).unwrap(), 0);
        assert_eq!(pool.pages(0).unwrap(), pages);
        assert_eq!(pool.state(0).unwrap(), Qwen38FlashNextSlotState::Active);

        assert_eq!(pool.restart(1).unwrap(), 0);
        assert_eq!(pool.state(1).unwrap(), Qwen38FlashNextSlotState::Free);
    }

    #[test]
    fn sequence_epochs_invalidate_snapshots_only_when_prefix_ownership_changes() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        pool.reserve(0, 128).unwrap();
        pool.commit(0, 100).unwrap();
        let first = pool.sequence(0).unwrap();

        pool.reserve(0, 256).unwrap();
        pool.commit(0, 180).unwrap();
        assert_eq!(pool.sequence(0).unwrap(), first);

        pool.rollback(0, 100).unwrap();
        assert_eq!(pool.sequence(0).unwrap(), first);

        pool.truncate(0, 80).unwrap();
        let truncated = pool.sequence(0).unwrap();
        assert_ne!(truncated, first);

        pool.restart(0).unwrap();
        let restarted = pool.sequence(0).unwrap();
        assert_ne!(restarted, truncated);

        pool.recycle(0).unwrap();
        let recycled = pool.sequence(0).unwrap();
        assert_ne!(recycled, restarted);

        pool.reserve(0, 64).unwrap();
        assert_ne!(pool.sequence(0).unwrap(), recycled);
        let reserved = pool.sequence(0).unwrap();
        pool.reset().unwrap();
        assert_ne!(pool.sequence(0).unwrap(), reserved);
        assert_eq!(pool.state(0).unwrap(), Qwen38FlashNextSlotState::Free);
    }

    #[test]
    fn reachable_tokens_counts_the_slots_own_pages_plus_everything_free() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        assert_eq!(
            pool.reachable_tokens(0).unwrap(),
            PAGES * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS
        );

        pool.reserve(1, 640).unwrap();
        assert_eq!(
            pool.reachable_tokens(0).unwrap(),
            (PAGES - 10) * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS
        );
        assert_eq!(
            pool.reachable_tokens(1).unwrap(),
            PAGES * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS
        );
    }

    #[test]
    fn a_retained_prefix_stops_at_the_first_token_that_differs() {
        let committed = [1u32, 2, 3, 4, 5];
        assert_eq!(
            Qwen38FlashNextSlotPool::reusable_prefix(&committed, &[1, 2, 3]),
            3
        );
        assert_eq!(
            Qwen38FlashNextSlotPool::reusable_prefix(&committed, &[1, 2, 9, 4]),
            2
        );
        assert_eq!(
            Qwen38FlashNextSlotPool::reusable_prefix(&committed, &[9]),
            0
        );
        assert_eq!(
            Qwen38FlashNextSlotPool::reusable_prefix(&committed, &[1, 2, 3, 4, 5, 6]),
            5
        );
    }

    #[test]
    fn every_page_the_pool_hands_out_is_accounted_across_a_long_random_lifecycle() {
        // The invariant that matters over a session's lifetime: pages are conserved. A leak
        // shows up as a pool that slowly stops being able to admit anything, which is the
        // failure a short test never reaches.
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for round in 0..2_000u64 {
            let slot = (next() % MAX_BATCH as u64) as usize;
            match round % 4 {
                0 => {
                    let tokens = (next() % 4_000) as usize;
                    if pool.reserve(slot, tokens).is_ok() {
                        pool.commit(slot, tokens).unwrap();
                    }
                }
                1 => {
                    let tokens = pool.tokens(slot).unwrap();
                    let target = if tokens == 0 {
                        0
                    } else {
                        (next() as usize) % (tokens + 1)
                    };
                    pool.truncate(slot, target).unwrap();
                }
                2 => {
                    let _ = pool.retain(slot);
                }
                _ => {
                    pool.recycle(slot).unwrap();
                }
            }

            let owned = (0..MAX_BATCH)
                .map(|slot| pool.pages(slot).unwrap().len())
                .sum::<usize>();
            assert_eq!(owned + pool.free_pages(), PAGES);
        }

        for slot in 0..MAX_BATCH {
            pool.recycle(slot).unwrap();
        }
        let mut every = (0..MAX_BATCH)
            .flat_map(|slot| pool.pages(slot).unwrap().to_vec())
            .chain(pool.free.iter().copied())
            .collect::<Vec<_>>();
        every.sort_unstable();
        every.dedup();
        assert_eq!(every.len(), PAGES);
    }

    #[test]
    fn a_slot_outside_the_batch_is_refused_by_every_operation() {
        let mut pool = Qwen38FlashNextSlotPool::new(PAGES).unwrap();
        assert!(pool.reserve(MAX_BATCH, 1).is_err());
        assert!(pool.truncate(MAX_BATCH, 0).is_err());
        assert!(pool.retain(MAX_BATCH).is_err());
        assert!(pool.recycle(MAX_BATCH).is_err());
        assert!(pool.tokens(MAX_BATCH).is_err());
        assert!(pool.table_row(MAX_BATCH).is_err());
    }

    #[test]
    fn a_pool_wider_than_the_block_table_is_refused() {
        assert!(Qwen38FlashNextSlotPool::new(0).is_err());
        assert!(
            Qwen38FlashNextSlotPool::new(
                MAX_BATCH * QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES + 1
            )
            .is_err()
        );
    }
}

//! Allocation-free shared-page ownership for eight stable KV slot rows.

use crate::{
    EngineError, EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, MAX_CONTEXT_TOKENS,
};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;

const FREE_PAGE_OWNER: u8 = u8::MAX;
const UNUSED_PAGE: u32 = u32::MAX;

/// Lifecycle of one stable page-table row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PagedKvSlotState {
    /// No live or retained tokens own pages in this row.
    Vacant,
    /// One admitted request owns this row.
    Active,
    /// A completed or cancelled request retains an exact reusable prefix.
    Retained,
}

/// Physical page and in-page token offset for one exact cached position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvRoute {
    physical_page: u32,
    page_offset: usize,
}

impl PagedKvRoute {
    /// Physical page shared by every full-attention layer for this logical page.
    pub const fn physical_page(self) -> u32 {
        self.physical_page
    }

    /// Token offset inside the 64-position physical page.
    pub const fn page_offset(self) -> usize {
        self.page_offset
    }
}

/// Contiguous page-table entries changed by one token-capacity reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvTableUpdate {
    slot: usize,
    first_entry: usize,
    entry_count: usize,
}

impl PagedKvTableUpdate {
    /// Stable slot row whose page table changed.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// First changed page-table column.
    pub const fn first_entry(self) -> usize {
        self.first_entry
    }

    /// Number of newly assigned contiguous table entries.
    pub const fn entry_count(self) -> usize {
        self.entry_count
    }

    /// Whether no new physical page was required.
    pub const fn is_empty(self) -> bool {
        self.entry_count == 0
    }
}

/// Host owner for one shared physical-page inventory and eight stable table rows.
///
/// Both backing allocations are fixed at construction. Activating, retaining,
/// extending, truncating, and recycling slots perform no heap allocation.
#[derive(Debug)]
pub struct PagedKvSlotPool {
    physical_pages: usize,
    table_stride: usize,
    max_context_tokens: usize,
    page_tables: Box<[u32]>,
    page_owners: Box<[u8]>,
    states: [PagedKvSlotState; MAX_BATCH],
    token_counts: [usize; MAX_BATCH],
    page_counts: [usize; MAX_BATCH],
    free_pages: usize,
    next_free_hint: usize,
}

impl PagedKvSlotPool {
    /// Creates the host owner for an exact shared pool selected by capacity admission.
    pub fn new(physical_pages: usize) -> EngineResult<Self> {
        if physical_pages > LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "paged KV slot pool has {physical_pages} pages, maximum is {LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }

        Self::new_with_limits(
            physical_pages,
            LONG_CONTEXT_PHYSICAL_PAGES,
            MAX_CONTEXT_TOKENS,
        )
    }

    pub(crate) fn new_with_limits(
        physical_pages: usize,
        table_stride: usize,
        max_context_tokens: usize,
    ) -> EngineResult<Self> {
        if table_stride == 0 {
            return Err(EngineError::layout("paged KV table stride must be nonzero"));
        }
        let table_capacity = table_stride
            .checked_mul(ATTENTION_PAGE_SIZE)
            .ok_or_else(|| EngineError::layout("paged KV table capacity overflows"))?;
        if max_context_tokens == 0 || max_context_tokens > table_capacity {
            return Err(EngineError::layout(format!(
                "paged KV maximum context {max_context_tokens} is outside 1..={table_capacity}"
            )));
        }
        if physical_pages > table_stride {
            return Err(EngineError::layout(format!(
                "paged KV pool has {physical_pages} physical pages, table stride is {table_stride}"
            )));
        }
        let table_entries = MAX_BATCH
            .checked_mul(table_stride)
            .ok_or_else(|| EngineError::layout("paged KV page-table entries overflow"))?;

        Ok(Self {
            physical_pages,
            table_stride,
            max_context_tokens,
            page_tables: vec![UNUSED_PAGE; table_entries].into_boxed_slice(),
            page_owners: vec![FREE_PAGE_OWNER; physical_pages].into_boxed_slice(),
            states: [PagedKvSlotState::Vacant; MAX_BATCH],
            token_counts: [0; MAX_BATCH],
            page_counts: [0; MAX_BATCH],
            free_pages: physical_pages,
            next_free_hint: 0,
        })
    }

    /// Number of physical pages shared across all active and retained slots.
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    /// Number of pages not owned by any slot.
    pub const fn free_pages(&self) -> usize {
        self.free_pages
    }

    /// Fixed bytes in the two host backing allocations.
    pub const fn host_allocation_bytes(&self) -> usize {
        self.page_tables.len() * size_of::<u32>() + self.page_owners.len() * size_of::<u8>()
    }

    #[cfg(feature = "qualification")]
    /// Stable host backing addresses for post-warmup allocation checks.
    pub fn qualification_addresses(&self) -> [usize; 2] {
        [
            self.page_tables.as_ptr().addr(),
            self.page_owners.as_ptr().addr(),
        ]
    }

    /// Current lifecycle state for one stable slot row.
    pub fn state(&self, slot: usize) -> EngineResult<PagedKvSlotState> {
        require_slot(slot)?;
        Ok(self.states[slot])
    }

    /// Exact processed token count whose KV state is owned by one slot.
    pub fn token_count(&self, slot: usize) -> EngineResult<usize> {
        require_slot(slot)?;
        Ok(self.token_counts[slot])
    }

    /// Number of logical pages currently assigned to one slot.
    pub fn page_count(&self, slot: usize) -> EngineResult<usize> {
        require_slot(slot)?;
        Ok(self.page_counts[slot])
    }

    /// Activates a vacant row cold or resumes a retained row without moving pages.
    pub fn activate(&mut self, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        match self.states[slot] {
            PagedKvSlotState::Vacant | PagedKvSlotState::Retained => {
                self.states[slot] = PagedKvSlotState::Active;
                Ok(())
            }
            PagedKvSlotState::Active => Err(EngineError::generation(format!(
                "paged KV slot {slot} is already active"
            ))),
        }
    }

    /// Marks one active row as an exact retained prefix without changing ownership.
    pub fn retain(&mut self, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        if self.states[slot] != PagedKvSlotState::Active {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} cannot be retained from state {:?}",
                self.states[slot]
            )));
        }
        self.states[slot] = PagedKvSlotState::Retained;
        Ok(())
    }

    /// Extends one active row to own every page needed for `token_count` positions.
    pub fn reserve_tokens(
        &mut self,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        require_slot(slot)?;
        require_active(self.states[slot], slot)?;
        if token_count > self.max_context_tokens {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} requires {token_count} tokens, maximum is {}",
                self.max_context_tokens
            )));
        }
        if token_count < self.token_counts[slot] {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} cannot reserve backwards from {} to {token_count} tokens",
                self.token_counts[slot]
            )));
        }

        let required_pages = token_count.div_ceil(ATTENTION_PAGE_SIZE);
        let existing_pages = self.page_counts[slot];
        let additional_pages = required_pages - existing_pages;
        if additional_pages > self.free_pages {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} requires {additional_pages} additional pages, only {} are free",
                self.free_pages
            )));
        }

        let table_begin = self.table_row_begin(slot)?;
        let mut search = self.next_free_hint;
        for entry in existing_pages..required_pages {
            while search < self.physical_pages && self.page_owners[search] != FREE_PAGE_OWNER {
                search += 1;
            }
            if search == self.physical_pages {
                self.release_partial_pages(table_begin, existing_pages, entry);
                return Err(EngineError::generation(
                    "paged KV free-page accounting disagrees with its owner inventory",
                ));
            }
            self.page_owners[search] = u8::try_from(slot)
                .map_err(|_| EngineError::layout("paged KV slot exceeds owner tag width"))?;
            self.page_tables[table_begin + entry] = u32::try_from(search)
                .map_err(|_| EngineError::layout("paged KV page exceeds table entry width"))?;
            search += 1;
        }
        self.next_free_hint = search;
        self.free_pages -= additional_pages;
        self.page_counts[slot] = required_pages;
        self.token_counts[slot] = token_count;

        Ok(PagedKvTableUpdate {
            slot,
            first_entry: existing_pages,
            entry_count: additional_pages,
        })
    }

    /// Releases trailing pages and retains exactly `token_count` processed positions.
    pub fn truncate_tokens(&mut self, slot: usize, token_count: usize) -> EngineResult<usize> {
        require_slot(slot)?;
        if self.states[slot] == PagedKvSlotState::Vacant {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} is vacant"
            )));
        }
        if token_count > self.token_counts[slot] {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} cannot truncate forwards from {} to {token_count} tokens",
                self.token_counts[slot]
            )));
        }

        let retained_pages = token_count.div_ceil(ATTENTION_PAGE_SIZE);
        let existing_pages = self.page_counts[slot];
        let table_begin = self.table_row_begin(slot)?;
        for entry in retained_pages..existing_pages {
            let table_index = table_begin + entry;
            let physical_page = usize::try_from(self.page_tables[table_index])
                .map_err(|_| EngineError::layout("paged KV table entry exceeds host width"))?;
            if physical_page >= self.physical_pages || self.page_owners[physical_page] != slot as u8
            {
                return Err(EngineError::generation(format!(
                    "paged KV slot {slot} table entry {entry} has inconsistent ownership"
                )));
            }
            self.page_owners[physical_page] = FREE_PAGE_OWNER;
            self.page_tables[table_index] = UNUSED_PAGE;
            self.next_free_hint = self.next_free_hint.min(physical_page);
        }
        let released = existing_pages - retained_pages;
        self.free_pages += released;
        self.page_counts[slot] = retained_pages;
        self.token_counts[slot] = token_count;

        Ok(released)
    }

    /// Releases every page and returns one active or retained row to vacant.
    pub fn recycle(&mut self, slot: usize) -> EngineResult<usize> {
        require_slot(slot)?;
        if self.states[slot] == PagedKvSlotState::Vacant {
            return Ok(0);
        }
        let released = self.truncate_tokens(slot, 0)?;
        self.states[slot] = PagedKvSlotState::Vacant;
        Ok(released)
    }

    /// Full stable device-table row, with unused columns set to `u32::MAX`.
    pub fn page_table(&self, slot: usize) -> EngineResult<&[u32]> {
        require_slot(slot)?;
        let begin = self.table_row_begin(slot)?;
        Ok(&self.page_tables[begin..begin + self.table_stride])
    }

    /// Physical route for one already-owned cached position.
    pub fn route(&self, slot: usize, position: usize) -> EngineResult<PagedKvRoute> {
        require_slot(slot)?;
        if position >= self.token_counts[slot] {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} position {position} is outside {} owned tokens",
                self.token_counts[slot]
            )));
        }
        let logical_page = position / ATTENTION_PAGE_SIZE;
        let physical_page = self.page_table(slot)?[logical_page];
        if physical_page == UNUSED_PAGE || physical_page as usize >= self.physical_pages {
            return Err(EngineError::generation(format!(
                "paged KV slot {slot} logical page {logical_page} is not assigned"
            )));
        }

        Ok(PagedKvRoute {
            physical_page,
            page_offset: position % ATTENTION_PAGE_SIZE,
        })
    }

    /// Returns the pages assigned by an aborted reservation, leaving it atomic.
    fn release_partial_pages(&mut self, table_begin: usize, first_entry: usize, end_entry: usize) {
        for entry in first_entry..end_entry {
            let table_index = table_begin + entry;
            let physical_page = self.page_tables[table_index] as usize;
            if physical_page < self.physical_pages {
                self.page_owners[physical_page] = FREE_PAGE_OWNER;
            }
            self.page_tables[table_index] = UNUSED_PAGE;
        }
    }

    fn table_row_begin(&self, slot: usize) -> EngineResult<usize> {
        slot.checked_mul(self.table_stride)
            .ok_or_else(|| EngineError::layout("paged KV table row offset overflows"))
    }
}

fn require_slot(slot: usize) -> EngineResult<()> {
    if slot >= MAX_BATCH {
        return Err(EngineError::route(format!(
            "paged KV slot {slot} is outside 0..{MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_active(state: PagedKvSlotState, slot: usize) -> EngineResult<()> {
    if state != PagedKvSlotState::Active {
        return Err(EngineError::generation(format!(
            "paged KV slot {slot} cannot reserve from state {state:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PagedKvSlotPool, PagedKvSlotState, UNUSED_PAGE};
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;

    #[test]
    fn one_slot_owns_the_exact_220k_shared_pool() {
        let mut pool = PagedKvSlotPool::new(3_438).unwrap();
        pool.activate(0).unwrap();
        let update = pool.reserve_tokens(0, 220_000).unwrap();

        assert_eq!((update.slot(), update.first_entry()), (0, 0));
        assert_eq!(update.entry_count(), 3_438);
        assert_eq!(pool.page_count(0).unwrap(), 3_438);
        assert_eq!(pool.token_count(0).unwrap(), 220_000);
        assert_eq!(pool.free_pages(), 0);
        assert_eq!(pool.route(0, 0).unwrap().physical_page(), 0);
        assert_eq!(pool.route(0, 63).unwrap().page_offset(), 63);
        assert_eq!(pool.route(0, 64).unwrap().physical_page(), 1);
        assert_eq!(pool.route(0, 219_999).unwrap().physical_page(), 3_437);
        assert_eq!(pool.route(0, 219_999).unwrap().page_offset(), 31);
    }

    #[test]
    fn eight_stable_rows_share_one_page_inventory() {
        let mut pool = PagedKvSlotPool::new(3_438).unwrap();
        for slot in 0..8 {
            pool.activate(slot).unwrap();
            pool.reserve_tokens(slot, 27_456).unwrap();
        }

        assert_eq!(pool.free_pages(), 6);
        for slot in 0..8 {
            assert_eq!(pool.page_count(slot).unwrap(), 429);
            assert_eq!(
                pool.route(slot, 0).unwrap().physical_page(),
                (slot * 429) as u32
            );
        }
        pool.reserve_tokens(0, 27_457).unwrap();
        assert_eq!(pool.free_pages(), 5);
    }

    #[test]
    fn exhaustion_is_atomic_and_recycling_reuses_the_released_page() {
        let mut pool = PagedKvSlotPool::new(8).unwrap();
        pool.activate(0).unwrap();
        pool.activate(1).unwrap();
        pool.reserve_tokens(0, 8 * 64).unwrap();
        let table_address = pool.page_table(1).unwrap().as_ptr();

        let error = pool.reserve_tokens(1, 1).unwrap_err();
        assert!(error.to_string().contains("only 0 are free"));
        assert_eq!(pool.page_count(1).unwrap(), 0);
        assert_eq!(pool.token_count(1).unwrap(), 0);
        assert_eq!(pool.page_table(1).unwrap().as_ptr(), table_address);

        pool.truncate_tokens(0, 7 * 64).unwrap();
        pool.reserve_tokens(1, 1).unwrap();
        assert_eq!(pool.route(1, 0).unwrap().physical_page(), 7);
        assert_eq!(pool.page_table(1).unwrap().as_ptr(), table_address);
    }

    #[test]
    fn retained_prefix_keeps_routes_until_explicit_recycle() {
        let mut pool = PagedKvSlotPool::new(16).unwrap();
        pool.activate(3).unwrap();
        pool.reserve_tokens(3, 130).unwrap();
        let routes = [
            pool.route(3, 0).unwrap(),
            pool.route(3, 64).unwrap(),
            pool.route(3, 129).unwrap(),
        ];
        pool.retain(3).unwrap();

        assert_eq!(pool.state(3).unwrap(), PagedKvSlotState::Retained);
        assert_eq!(pool.free_pages(), 13);
        pool.activate(3).unwrap();
        assert_eq!(pool.route(3, 0).unwrap(), routes[0]);
        assert_eq!(pool.route(3, 64).unwrap(), routes[1]);
        assert_eq!(pool.route(3, 129).unwrap(), routes[2]);

        assert_eq!(pool.recycle(3).unwrap(), 3);
        assert_eq!(pool.state(3).unwrap(), PagedKvSlotState::Vacant);
        assert_eq!(pool.free_pages(), 16);
        assert!(
            pool.page_table(3)
                .unwrap()
                .iter()
                .all(|&page| page == UNUSED_PAGE)
        );
    }

    #[test]
    fn failed_admission_rollback_restores_the_exact_retained_prefix() {
        let mut pool = PagedKvSlotPool::new(6).unwrap();
        pool.activate(0).unwrap();
        pool.reserve_tokens(0, 3 * 64).unwrap();
        pool.activate(1).unwrap();
        pool.reserve_tokens(1, 70).unwrap();
        pool.retain(1).unwrap();
        let routes = [pool.route(1, 0).unwrap(), pool.route(1, 69).unwrap()];

        pool.activate(1).unwrap();
        let error = pool.reserve_tokens(1, 200).unwrap_err();
        assert!(error.to_string().contains("only 1 are free"));
        assert_eq!(pool.truncate_tokens(1, 70).unwrap(), 0);
        pool.retain(1).unwrap();

        assert_eq!(pool.state(1).unwrap(), PagedKvSlotState::Retained);
        assert_eq!(pool.token_count(1).unwrap(), 70);
        assert_eq!(pool.page_count(1).unwrap(), 2);
        assert_eq!(pool.free_pages(), 1);
        assert_eq!(pool.route(1, 0).unwrap(), routes[0]);
        assert_eq!(pool.route(1, 69).unwrap(), routes[1]);

        pool.truncate_tokens(0, 64).unwrap();
        pool.activate(1).unwrap();
        pool.reserve_tokens(1, 200).unwrap();
        assert_eq!(pool.route(1, 0).unwrap(), routes[0]);
        assert_eq!(pool.route(1, 69).unwrap(), routes[1]);
        assert_eq!(pool.token_count(1).unwrap(), 200);
    }

    #[test]
    fn partial_page_truncation_preserves_only_the_exact_processed_length() {
        let mut pool = PagedKvSlotPool::new(8).unwrap();
        pool.activate(0).unwrap();
        pool.reserve_tokens(0, 193).unwrap();
        assert_eq!(pool.page_count(0).unwrap(), 4);

        assert_eq!(pool.truncate_tokens(0, 65).unwrap(), 2);
        assert_eq!(pool.page_count(0).unwrap(), 2);
        assert_eq!(pool.token_count(0).unwrap(), 65);
        assert_eq!(pool.route(0, 64).unwrap().page_offset(), 0);
        assert!(pool.route(0, 65).is_err());
        assert_eq!(pool.page_table(0).unwrap()[2], UNUSED_PAGE);
    }

    #[test]
    fn backing_addresses_and_allocation_bytes_are_stable_after_warmup() {
        let mut pool = PagedKvSlotPool::new(3_438).unwrap();
        let table_address = pool.page_table(0).unwrap().as_ptr();
        let allocation_bytes = 8 * 3_438 * size_of::<u32>() + 3_438;
        assert_eq!(pool.host_allocation_bytes(), allocation_bytes);

        pool.activate(0).unwrap();
        for tokens in [1, 64, 65, 4_096, 220_000] {
            pool.reserve_tokens(0, tokens).unwrap();
            assert_eq!(pool.page_table(0).unwrap().as_ptr(), table_address);
            assert_eq!(pool.host_allocation_bytes(), allocation_bytes);
        }
        pool.retain(0).unwrap();
        pool.recycle(0).unwrap();
        assert_eq!(pool.page_table(0).unwrap().as_ptr(), table_address);
        assert_eq!(pool.host_allocation_bytes(), allocation_bytes);
    }

    #[test]
    fn exact_qwen35_limits_reuse_the_allocation_free_owner() {
        let mut pool = PagedKvSlotPool::new_with_limits(4_096, 4_096, 262_144).unwrap();
        pool.activate(7).unwrap();
        pool.reserve_tokens(7, 262_144).unwrap();

        assert_eq!(pool.page_table(7).unwrap().len(), 4_096);
        assert_eq!(pool.page_count(7).unwrap(), 4_096);
        assert_eq!(pool.route(7, 262_143).unwrap().physical_page(), 4_095);
        assert_eq!(pool.route(7, 262_143).unwrap().page_offset(), 63);
        assert_eq!(pool.host_allocation_bytes(), 8 * 4_096 * 4 + 4_096);
    }

    #[test]
    fn retained_page_reclamation_closes_the_observed_admission_shortfall() {
        let mut pool = PagedKvSlotPool::new(3_438).unwrap();
        pool.activate(3).unwrap();
        pool.reserve_tokens(3, 1_236 * ATTENTION_PAGE_SIZE).unwrap();
        pool.retain(3).unwrap();
        pool.activate(0).unwrap();
        pool.reserve_tokens(0, 46 * ATTENTION_PAGE_SIZE).unwrap();
        pool.retain(0).unwrap();
        pool.activate(1).unwrap();
        pool.reserve_tokens(1, 1_678 * ATTENTION_PAGE_SIZE).unwrap();
        assert_eq!(pool.free_pages(), 478);

        pool.activate(3).unwrap();
        let error = pool
            .reserve_tokens(3, 1_760 * ATTENTION_PAGE_SIZE)
            .unwrap_err();
        assert!(error.to_string().contains("requires 524 additional pages"));
        assert_eq!(pool.page_count(3).unwrap(), 1_236);
        pool.retain(3).unwrap();

        assert_eq!(pool.recycle(0).unwrap(), 46);
        pool.activate(3).unwrap();
        pool.reserve_tokens(3, 1_760 * ATTENTION_PAGE_SIZE).unwrap();
        assert_eq!(pool.free_pages(), 0);
    }
}

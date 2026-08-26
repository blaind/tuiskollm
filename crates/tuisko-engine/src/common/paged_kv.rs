//! Generic paged K/V cache storage for every exact paged cache family.
//!
//! One storage owns exactly one device allocation and one logical page
//! lifecycle. The arena is built here from the family's own plan and is never
//! accepted from or handed to a caller, so two storages — a target cache and a
//! separate cache mirror — occupy disjoint device allocations by construction
//! and can never alias each other's represented cache bytes. A mirror shares
//! the target's *logical* page lifecycle by being driven through the identical
//! lifecycle calls; `require_mirror` proves the two mappings stayed equal.

use crate::{
    EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH, PagedKvRoute, PagedKvSlotPool,
    PagedKvSlotState, PagedKvTableUpdate,
};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaStream, DeviceArena, DeviceCopy, GpuError,
};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;

/// One attention layer's key and value page planes inside a cache arena.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvPlanes<V: DeviceCopy> {
    /// Physical key pages for this layer.
    pub key: ArenaRegion<V>,
    /// Physical value pages for this layer.
    pub value: ArenaRegion<V>,
}

/// Stable device addresses one attention route binds for a single cache plane.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvBinding {
    /// Shared page table covering every stable slot row.
    pub block_tables: u64,
    /// Physical key pages for the bound plane.
    pub key_pages: u64,
    /// Physical value pages for the bound plane.
    pub value_pages: u64,
    /// Page-table columns reserved for every stable slot.
    pub table_stride: usize,
    /// Maximum logical context admitted by the pinned checkpoint.
    pub context_capacity: usize,
}

/// Seals `PagedKvLayout` against out-of-crate implementations.
pub mod sealed {
    /// Restricts `PagedKvLayout` to the in-crate exact cache families.
    pub trait Sealed {}
}

/// Host plan of one exact paged K/V cache family.
///
/// Sealed and monomorphized so only admitted cache families are constructible. It publishes page
/// geometry and arena regions, never launch topology, graph order, or accumulation order.
pub trait PagedKvLayout: LayerMemoryLayout + Sized + sealed::Sealed {
    /// Represented cache element admitted by this family's kernels.
    type Value: DeviceCopy;

    /// Error-message identity of this cache family.
    const NAME: &'static str;
    /// Physical pages in the complete pool, which is also the fixed number of
    /// page-table columns every stable slot row keeps in a partial pool.
    const FULL_PHYSICAL_PAGES: usize;
    /// Maximum logical context admitted by the pinned checkpoint.
    const MAX_CONTEXT_TOKENS: usize;
    /// Whether reserving a physical page must clear its represented values.
    ///
    /// A cache mirror writes only the positions its own route accepts, so a
    /// reused page would otherwise expose the previous owner's values. A target
    /// cache is fully written by its prefill route and never clears.
    const CLEARS_REUSED_PAGES: bool;

    /// Plans the family's single allocation over an exact physical page count.
    fn build_for_pages(physical_pages: usize) -> EngineResult<Self>;

    /// Single-allocation plan for the complete owner.
    fn builder(&self) -> &ArenaLayout;

    /// Device page table shared by every stable slot.
    fn block_tables(&self) -> ArenaRegion<u32>;

    /// Key and value page planes in exact attention-layer inventory order.
    fn planes(&self) -> &[PagedKvPlanes<Self::Value>];
}

/// One paged K/V cache family's device storage and allocation-free page lifecycle.
pub struct PagedKvCacheStorage<L: PagedKvLayout> {
    // The arena drops before the slot owner and the context it was cut from.
    arena: DeviceArena,
    slots: PagedKvSlotPool,
    context: Arc<CudaContext>,
    layout: L,
    physical_pages: usize,
}

impl<L: PagedKvLayout> PagedKvCacheStorage<L> {
    /// Allocates the family's complete physical page pool.
    pub fn new(context: &Arc<CudaContext>) -> EngineResult<Self> {
        Self::new_for_pages(context, L::FULL_PHYSICAL_PAGES)
    }

    fn new_for_pages(context: &Arc<CudaContext>, physical_pages: usize) -> EngineResult<Self> {
        let layout = L::build_for_pages(physical_pages)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let slots = PagedKvSlotPool::new_with_limits(
            physical_pages,
            L::FULL_PHYSICAL_PAGES,
            L::MAX_CONTEXT_TOKENS,
        )?;
        arena.fill(&stream, layout.block_tables(), u8::MAX)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(Self {
            arena,
            slots,
            context: Arc::clone(context),
            layout,
            physical_pages,
        })
    }

    /// Marks one vacant or retained physical slot active.
    pub fn activate_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.slots.activate(slot)
    }

    /// Reserves all pages needed by one exact processed-token capacity.
    pub fn reserve_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        let update = self.slots.reserve_tokens(slot, token_count)?;
        if L::CLEARS_REUSED_PAGES {
            self.clear_reserved_pages(stream, slot, update)?;
        }
        self.upload_update(stream, update)?;

        Ok(update)
    }

    /// Releases trailing pages and retains exactly `token_count` positions.
    pub fn truncate_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<usize> {
        let old_pages = self.slots.page_count(slot)?;
        let released = self.slots.truncate_tokens(slot, token_count)?;
        let first_entry = self.slots.page_count(slot)?;
        self.upload_entries(stream, slot, first_entry, old_pages - first_entry)?;

        Ok(released)
    }

    /// Retains one active slot's exact page ownership for prefix reuse.
    pub fn retain_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.slots.retain(slot)
    }

    /// Releases every page owned by one active or retained slot.
    pub fn recycle_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        let old_pages = self.slots.page_count(slot)?;
        let released = self.slots.recycle(slot)?;
        self.upload_entries(stream, slot, 0, old_pages)?;

        Ok(released)
    }

    /// Clears all page ownership and represented cache values.
    pub fn reset(&mut self, stream: &CudaStream) -> EngineResult<()> {
        self.reset_ownership(stream)?;
        for plane in self.layout.planes() {
            self.arena.fill(stream, plane.key, 0)?;
            self.arena.fill(stream, plane.value, 0)?;
        }

        Ok(())
    }

    /// Releases every slot's pages and clears the shared device page table.
    pub(crate) fn reset_ownership(&mut self, stream: &CudaStream) -> EngineResult<()> {
        for slot in 0..MAX_BATCH {
            self.slots.recycle(slot)?;
        }
        self.arena
            .fill(stream, self.layout.block_tables(), u8::MAX)?;

        Ok(())
    }

    /// Host lifecycle state for one stable physical slot.
    pub fn slot_state(&self, slot: usize) -> EngineResult<PagedKvSlotState> {
        self.slots.state(slot)
    }

    /// Exact processed token count owned by one slot.
    pub fn slot_token_count(&self, slot: usize) -> EngineResult<usize> {
        self.slots.token_count(slot)
    }

    /// Exact physical route for one already-owned position.
    pub fn route(&self, slot: usize, position: usize) -> EngineResult<PagedKvRoute> {
        self.slots.route(slot, position)
    }

    /// Maximum logical context admitted by the pinned checkpoint.
    pub const fn context_capacity(&self) -> usize {
        L::MAX_CONTEXT_TOKENS
    }

    /// Number of physical pages shared by every active slot.
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    /// Complete device allocation bytes.
    pub const fn arena_bytes(&self) -> usize {
        self.arena.byte_len()
    }

    /// Fixed host page-table and owner-map bytes.
    pub const fn host_allocation_bytes(&self) -> usize {
        self.slots.host_allocation_bytes()
    }

    /// Stable base address of the single device allocation.
    pub fn base_address(&self) -> u64 {
        self.arena.base_address()
    }

    /// CUDA context shared by the arena and all consuming graphs.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Checked page-pool layout.
    pub const fn layout(&self) -> &L {
        &self.layout
    }

    /// Stable device addresses one attention route binds for a single plane.
    pub(crate) fn plane_binding(&self, plane: usize) -> EngineResult<PagedKvBinding> {
        let regions = self.layout.planes().get(plane).ok_or_else(|| {
            EngineError::layout(format!(
                "{} attention layer {plane} is outside the shared KV inventory",
                L::NAME
            ))
        })?;

        Ok(PagedKvBinding {
            block_tables: self.arena.address(self.layout.block_tables())?.addr() as u64,
            key_pages: self.arena.address(regions.key)?.addr() as u64,
            value_pages: self.arena.address(regions.value)?.addr() as u64,
            table_stride: L::FULL_PHYSICAL_PAGES,
            context_capacity: L::MAX_CONTEXT_TOKENS,
        })
    }

    /// Requires that `mirror` still shares this storage's logical page lifecycle.
    ///
    /// The two storages keep disjoint device allocations, so this only compares
    /// the host page mapping: lifecycle state, processed tokens, and the
    /// physical page selected for every owned logical page.
    pub(crate) fn require_mirror<M: PagedKvLayout>(
        &self,
        mirror: &PagedKvCacheStorage<M>,
        slot: usize,
    ) -> EngineResult<()> {
        let state = self.slot_state(slot)?;
        let mirror_state = mirror.slot_state(slot)?;
        let tokens = self.slot_token_count(slot)?;
        let mirror_tokens = mirror.slot_token_count(slot)?;
        if state != mirror_state || tokens != mirror_tokens {
            return Err(EngineError::generation(format!(
                "{} and {} slot {slot} lifecycle differs: {state:?}/{mirror_state:?}, {tokens}/{mirror_tokens} tokens",
                L::NAME,
                M::NAME
            )));
        }
        for position in (0..tokens).step_by(ATTENTION_PAGE_SIZE) {
            self.require_mirrored_route(mirror, slot, position)?;
        }

        Ok(())
    }

    /// Requires that both storages map one owned position to the same page.
    pub(crate) fn require_mirrored_route<M: PagedKvLayout>(
        &self,
        mirror: &PagedKvCacheStorage<M>,
        slot: usize,
        position: usize,
    ) -> EngineResult<()> {
        if self.route(slot, position)? != mirror.route(slot, position)? {
            return Err(EngineError::generation(format!(
                "{} and {} slot {slot} map position {position} differently",
                L::NAME,
                M::NAME
            )));
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Builds a smaller physical pool while preserving the production table stride.
    pub fn qualification_for_pages(
        context: &Arc<CudaContext>,
        physical_pages: usize,
    ) -> EngineResult<Self> {
        Self::new_for_pages(context, physical_pages)
    }

    #[cfg(feature = "qualification")]
    /// Reads every stable device page-table row.
    pub fn qualification_block_tables(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        Ok(self
            .arena
            .copy_to_host(stream, self.layout.block_tables())?)
    }

    #[cfg(feature = "qualification")]
    /// Stable device and host backing addresses.
    pub fn qualification_addresses(&self) -> [usize; 3] {
        let host = self.slots.qualification_addresses();
        [self.arena.base_address() as usize, host[0], host[1]]
    }

    #[cfg(feature = "qualification")]
    /// Reads one complete physical K/V page from every attention layer.
    pub fn qualification_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<L::Value>, Vec<L::Value>)> {
        if physical_page >= self.physical_pages {
            return Err(EngineError::route(format!(
                "{} physical KV page {physical_page} is outside 0..{}",
                L::NAME,
                self.physical_pages
            )));
        }
        let mut key = Vec::new();
        let mut value = Vec::new();
        for plane in self.layout.planes() {
            let page_values = plane.key.len() / self.physical_pages;
            let start = physical_page
                .checked_mul(page_values)
                .ok_or_else(|| Self::overflow("KV page offset"))?;
            key.extend(
                self.arena
                    .copy_slice_to_host(stream, plane.key, start, page_values)?,
            );
            value.extend(
                self.arena
                    .copy_slice_to_host(stream, plane.value, start, page_values)?,
            );
        }
        Ok((key, value))
    }

    #[cfg(feature = "qualification")]
    /// Reads a leading represented-value prefix of the first cache plane.
    pub(crate) fn qualification_cache_prefix(
        &self,
        stream: &CudaStream,
        values: usize,
    ) -> EngineResult<(Vec<L::Value>, Vec<L::Value>)> {
        let plane = self
            .layout
            .planes()
            .first()
            .ok_or_else(|| EngineError::layout(format!("{} has no cache plane", L::NAME)))?;
        Ok((
            self.arena.copy_prefix_to_host(stream, plane.key, values)?,
            self.arena
                .copy_prefix_to_host(stream, plane.value, values)?,
        ))
    }

    fn clear_reserved_pages(
        &self,
        stream: &CudaStream,
        slot: usize,
        update: PagedKvTableUpdate,
    ) -> EngineResult<()> {
        for logical_page in update.first_entry()..update.first_entry() + update.entry_count() {
            let position = logical_page
                .checked_mul(ATTENTION_PAGE_SIZE)
                .ok_or_else(|| Self::overflow("KV page position"))?;
            let physical_page = usize::try_from(self.slots.route(slot, position)?.physical_page())
                .map_err(|_| {
                    EngineError::layout(format!("{} physical page exceeds host width", L::NAME))
                })?;
            for plane in self.layout.planes() {
                let page_values = plane
                    .key
                    .len()
                    .checked_div(self.physical_pages)
                    .ok_or_else(|| Self::overflow("KV page size"))?;
                let start = physical_page
                    .checked_mul(page_values)
                    .ok_or_else(|| Self::overflow("KV page offset"))?;
                self.arena
                    .fill_slice(stream, plane.key, start, page_values, 0)?;
                self.arena
                    .fill_slice(stream, plane.value, start, page_values, 0)?;
            }
        }

        Ok(())
    }

    fn upload_update(&self, stream: &CudaStream, update: PagedKvTableUpdate) -> EngineResult<()> {
        self.upload_entries(
            stream,
            update.slot(),
            update.first_entry(),
            update.entry_count(),
        )
    }

    fn upload_entries(
        &self,
        stream: &CudaStream,
        slot: usize,
        first_entry: usize,
        entry_count: usize,
    ) -> EngineResult<()> {
        if entry_count == 0 {
            return Ok(());
        }
        let destination = slot
            .checked_mul(L::FULL_PHYSICAL_PAGES)
            .and_then(|begin| begin.checked_add(first_entry))
            .ok_or_else(|| Self::overflow("KV table upload offset"))?;
        let end = first_entry
            .checked_add(entry_count)
            .ok_or_else(|| Self::overflow("KV table upload range"))?;
        self.arena.copy_slice_from_host(
            stream,
            self.layout.block_tables(),
            destination,
            &self.slots.page_table(slot)?[first_entry..end],
        )?;

        Ok(())
    }

    fn overflow(what: &str) -> EngineError {
        EngineError::layout(format!("{} {what} overflows", L::NAME))
    }
}

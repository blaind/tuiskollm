//! Address-stable device and host ownership for Qwen3.5 BF16 KV pages.

use crate::{
    EngineError, EngineResult, MAX_BATCH, PagedKvRoute, PagedKvSlotPool, PagedKvSlotState,
    PagedKvTableUpdate, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS,
    Qwen35LongContextKvLayout,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, DeviceArena, GpuError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35AttentionKvBinding {
    pub(crate) block_tables: u64,
    pub(crate) key_pages: u64,
    pub(crate) value_pages: u64,
    pub(crate) table_stride: usize,
    pub(crate) context_capacity: usize,
}

/// Fixed shared BF16 KV allocation and its allocation-free page lifecycle.
pub struct Qwen35LongContextKvProgram {
    arena: DeviceArena,
    slots: PagedKvSlotPool,
    context: Arc<CudaContext>,
    layout: Qwen35LongContextKvLayout,
}

impl Qwen35LongContextKvProgram {
    /// Allocates the complete 262,144-position page pool.
    pub fn new(context: &Arc<CudaContext>) -> EngineResult<Self> {
        Self::new_for_pages(context, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES)
    }

    fn new_for_pages(context: &Arc<CudaContext>, physical_pages: usize) -> EngineResult<Self> {
        let layout = Qwen35LongContextKvLayout::build_for_pages(physical_pages)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let slots = PagedKvSlotPool::new_with_limits(
            physical_pages,
            QWEN35_LONG_CONTEXT_PHYSICAL_PAGES,
            QWEN35_MAX_CONTEXT_TOKENS,
        )?;
        arena.fill(&stream, layout.block_tables(), u8::MAX)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(Self {
            arena,
            slots,
            context: Arc::clone(context),
            layout,
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
        for layer in self.layout.layers() {
            self.arena.fill(stream, layer.key, 0)?;
            self.arena.fill(stream, layer.value, 0)?;
        }

        Ok(())
    }

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
        QWEN35_MAX_CONTEXT_TOKENS
    }

    /// Number of physical pages shared by every active slot.
    pub const fn physical_pages(&self) -> usize {
        self.layout.physical_pages()
    }

    /// Complete device allocation bytes.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
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
    pub const fn layout(&self) -> &Qwen35LongContextKvLayout {
        &self.layout
    }

    pub(crate) fn layer_binding(
        &self,
        attention_layer: usize,
    ) -> EngineResult<Qwen35AttentionKvBinding> {
        let regions = self.layout.layers().get(attention_layer).ok_or_else(|| {
            EngineError::layout(format!(
                "Qwen3.5 attention layer {attention_layer} is outside the shared KV inventory"
            ))
        })?;

        Ok(Qwen35AttentionKvBinding {
            block_tables: self.arena.address(self.layout.block_tables())?.addr() as u64,
            key_pages: self.arena.address(regions.key)?.addr() as u64,
            value_pages: self.arena.address(regions.value)?.addr() as u64,
            table_stride: self.layout.block_table_stride(),
            context_capacity: QWEN35_MAX_CONTEXT_TOKENS,
        })
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
            .checked_mul(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES)
            .and_then(|begin| begin.checked_add(first_entry))
            .ok_or_else(|| EngineError::layout("Qwen3.5 KV table upload offset overflows"))?;
        let end = first_entry
            .checked_add(entry_count)
            .ok_or_else(|| EngineError::layout("Qwen3.5 KV table upload range overflows"))?;
        self.arena.copy_slice_from_host(
            stream,
            self.layout.block_tables(),
            destination,
            &self.slots.page_table(slot)?[first_entry..end],
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS};

    #[test]
    fn exact_context_and_page_inventory_are_consistent() {
        assert_eq!(QWEN35_MAX_CONTEXT_TOKENS, 262_144);
        assert_eq!(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES * 64, 262_144);
    }
}

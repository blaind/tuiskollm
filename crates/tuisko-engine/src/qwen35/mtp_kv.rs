//! Address-stable Qwen3.5 MTP cache storage with mirrored page ownership.

use crate::qwen35::mtp_kv_layout::Qwen35MtpKvLayout;
#[cfg(feature = "qualification")]
use crate::qwen35::mtp_kv_layout::Qwen35MtpKvRegions;
use crate::qwen35::mtp_layer::Qwen35MtpKvBinding;
use crate::{
    EngineError, EngineResult, PagedKvRoute, PagedKvSlotPool, PagedKvSlotState, PagedKvTableUpdate,
    QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, DeviceArena, GpuError};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen35_9B};

/// Separate BF16 MTP cache whose logical page ownership mirrors the target.
pub(crate) struct Qwen35MtpKvProgram {
    arena: DeviceArena,
    slots: PagedKvSlotPool,
    layout: Qwen35MtpKvLayout,
}

impl Qwen35MtpKvProgram {
    /// Allocates the complete 262,144-position MTP cache mirror.
    pub(crate) fn new(context: &Arc<CudaContext>) -> EngineResult<Self> {
        Self::new_for_pages(context, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES)
    }

    fn new_for_pages(context: &Arc<CudaContext>, physical_pages: usize) -> EngineResult<Self> {
        let layout = Qwen35MtpKvLayout::build_for_pages(physical_pages)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let slots = PagedKvSlotPool::new_with_limits(
            physical_pages,
            QWEN35_LONG_CONTEXT_PHYSICAL_PAGES,
            QWEN35_MAX_CONTEXT_TOKENS,
        )?;
        arena.fill(&stream, layout.regions().block_tables, u8::MAX)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(Self {
            arena,
            slots,
            layout,
        })
    }

    pub(crate) fn binding(&self) -> EngineResult<Qwen35MtpKvBinding> {
        let regions = self.layout.regions();
        Ok(Qwen35MtpKvBinding {
            block_tables: self.arena.address(regions.block_tables)?.addr() as u64,
            key_pages: self.arena.address(regions.key_pages)?.addr() as u64,
            value_pages: self.arena.address(regions.value_pages)?.addr() as u64,
            table_stride: self.layout.table_stride(),
            context_capacity: QWEN35_MAX_CONTEXT_TOKENS,
        })
    }

    pub(crate) fn activate_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.slots.activate(slot)
    }

    pub(crate) fn reserve_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        let update = self.slots.reserve_tokens(slot, token_count)?;
        let regions = self.layout.regions();
        let page_values = Qwen35_9B::NUM_KV_HEADS
            .checked_mul(ATTENTION_PAGE_SIZE)
            .and_then(|values| values.checked_mul(Qwen35_9B::HEAD_DIM))
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP page size overflows"))?;
        for logical_page in update.first_entry()..update.first_entry() + update.entry_count() {
            let position = logical_page
                .checked_mul(ATTENTION_PAGE_SIZE)
                .ok_or_else(|| EngineError::layout("Qwen3.5 MTP page position overflows"))?;
            let physical_page = usize::try_from(self.slots.route(slot, position)?.physical_page())
                .map_err(|_| EngineError::layout("Qwen3.5 MTP physical page exceeds host width"))?;
            let start = physical_page
                .checked_mul(page_values)
                .ok_or_else(|| EngineError::layout("Qwen3.5 MTP page offset overflows"))?;
            self.arena
                .fill_slice(stream, regions.key_pages, start, page_values, 0)?;
            self.arena
                .fill_slice(stream, regions.value_pages, start, page_values, 0)?;
        }
        self.upload_update(stream, update)?;

        Ok(update)
    }

    pub(crate) fn truncate_slot_tokens(
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

    pub(crate) fn retain_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.slots.retain(slot)
    }

    pub(crate) fn recycle_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        let old_pages = self.slots.page_count(slot)?;
        let released = self.slots.recycle(slot)?;
        self.upload_entries(stream, slot, 0, old_pages)?;

        Ok(released)
    }

    pub(crate) fn reset(&mut self, stream: &CudaStream) -> EngineResult<()> {
        for slot in 0..crate::MAX_BATCH {
            self.slots.recycle(slot)?;
        }
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.block_tables, u8::MAX)?;
        self.arena.fill(stream, regions.key_pages, 0)?;
        self.arena.fill(stream, regions.value_pages, 0)?;

        Ok(())
    }

    pub(crate) fn state(&self, slot: usize) -> EngineResult<PagedKvSlotState> {
        self.slots.state(slot)
    }

    pub(crate) fn token_count(&self, slot: usize) -> EngineResult<usize> {
        self.slots.token_count(slot)
    }

    pub(crate) fn route(&self, slot: usize, position: usize) -> EngineResult<PagedKvRoute> {
        self.slots.route(slot, position)
    }

    /// Fixed host owner-map and page-table bytes.
    pub const fn host_owner_bytes(&self) -> usize {
        self.slots.host_allocation_bytes()
    }

    /// Stable device base address.
    pub fn base_address(&self) -> u64 {
        self.arena.base_address()
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
            .checked_mul(self.layout.table_stride())
            .and_then(|start| start.checked_add(first_entry))
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP table offset overflows"))?;
        let end = first_entry
            .checked_add(entry_count)
            .ok_or_else(|| EngineError::layout("Qwen3.5 MTP table range overflows"))?;
        self.arena.copy_slice_from_host(
            stream,
            self.layout.regions().block_tables,
            destination,
            &self.slots.page_table(slot)?[first_entry..end],
        )?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    pub(crate) fn qualification_block_tables(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        Ok(self
            .arena
            .copy_to_host(stream, self.layout.regions().block_tables)?)
    }

    #[cfg(feature = "qualification")]
    pub(crate) fn qualification_cache_prefix(
        &self,
        stream: &CudaStream,
        values: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        let Qwen35MtpKvRegions {
            key_pages,
            value_pages,
            ..
        } = self.layout.regions();
        Ok((
            self.arena.copy_prefix_to_host(stream, key_pages, values)?,
            self.arena
                .copy_prefix_to_host(stream, value_pages, values)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::MAX_BATCH;

    #[test]
    fn qwen35_mtp_mirror_has_eight_stable_rows() {
        assert_eq!(MAX_BATCH, 8);
    }
}

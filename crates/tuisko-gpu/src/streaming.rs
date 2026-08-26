//! Model-independent primitives for a streaming-resident weight tier.
//!
//! [`PinnedHostPool`] owns pooled host bytes, [`TransferStream`] fences uploads,
//! [`PinnedBounceRing`] stages mapped extents, and [`DeviceSlotPool`] owns stable
//! device slots plus their item-to-slot table.
//!
//! None of these know a model geometry. The item count, extent size, and slot
//! budget are parameters; residency policy lives above them in the engine.

use crate::{
    ArenaLayout, ArenaRegion, CudaContext, CudaEvent, CudaStream, DeviceArena, GpuError, GpuResult,
    PinnedHostBuffer,
};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Indirection-table entry for an item that owns no slot.
pub const ABSENT_SLOT: u32 = u32::MAX;

/// Device address alignment shared by every slot extent and the table.
const ALIGNMENT: usize = 256;

/// Pinned staging generations the indirection table rotates through.
///
/// Each generation has a fence that prevents reuse during an in-flight copy.
pub const INDIRECTION_TABLE_GENERATIONS: usize = 4;

/// Local spelling of the staging ring depth.
const TABLE_GENERATIONS: usize = INDIRECTION_TABLE_GENERATIONS;

/// Reclaim fences the transfer stream keeps, one per released replay generation.
///
/// A round waits on the oldest recorded fence that releases every slot it will
/// overwrite. Four generations preserve useful overlap with an in-flight replay.
pub const RECLAIM_FENCE_GENERATIONS: usize = 4;

/// One page-locked host allocation with its measured pinning cost.
///
/// Allocation records its pinning time for residency accounting.
pub struct PinnedHostPool {
    storage: PinnedHostBuffer<u8>,
    bytes: usize,
    pin_duration: Duration,
}

impl PinnedHostPool {
    /// Allocates and zeroes `bytes` of page-locked host memory.
    pub fn allocate(context: &Arc<CudaContext>, bytes: usize) -> GpuResult<Self> {
        context
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the pinned-pool CUDA context", source))?;
        let started = Instant::now();
        let storage = PinnedHostBuffer::zeroed(context, bytes)
            .map_err(|source| GpuError::driver("allocating a pinned host pool", source))?;
        let pin_duration = started.elapsed();

        Ok(Self {
            storage,
            bytes,
            pin_duration,
        })
    }

    /// Page-locked bytes owned by this pool.
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Whether this pool owns no host bytes.
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Wall time the allocation and its page locking took.
    pub const fn pin_duration(&self) -> Duration {
        self.pin_duration
    }

    /// CUDA context this allocation is page-locked against.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.storage.context()
    }

    /// Stable host base address of the page-locked allocation.
    pub fn base_address(&self) -> usize {
        self.storage.as_ptr().addr()
    }

    /// Copies `source` into the pool at `offset`.
    ///
    /// Callers must not overwrite bytes an enqueued upload has not yet read;
    /// [`TransferStream::stall_for_publication`] proves that for a whole round.
    pub fn write(&mut self, offset: usize, source: &[u8]) -> GpuResult<()> {
        let end = offset
            .checked_add(source.len())
            .ok_or_else(|| GpuError::arena("pinned host pool write range overflows"))?;
        if end > self.bytes {
            return Err(GpuError::arena(format!(
                "pinned host pool write {offset}..{end} exceeds {} bytes",
                self.bytes
            )));
        }
        self.storage.as_mut_slice()[offset..end].copy_from_slice(source);

        Ok(())
    }

    /// Borrows one checked byte range of the pool.
    pub fn slice(&self, range: Range<usize>) -> GpuResult<&[u8]> {
        if range.start > range.end || range.end > self.bytes {
            return Err(GpuError::arena(format!(
                "pinned host pool range {}..{} exceeds {} bytes",
                range.start, range.end, self.bytes
            )));
        }

        Ok(&self.storage.as_slice()[range])
    }

    fn source_pointer(&self, offset: usize, bytes: usize) -> GpuResult<*const u8> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| GpuError::arena("pinned host pool upload range overflows"))?;
        if end > self.bytes {
            return Err(GpuError::arena(format!(
                "pinned host pool upload {offset}..{end} exceeds {} bytes",
                self.bytes
            )));
        }
        // SAFETY: the checked offset is inside the live page-locked allocation.
        Ok(unsafe { self.storage.as_ptr().add(offset) })
    }
}

/// A dedicated upload stream with explicit publication and reclaim fences.
///
/// A publication event orders uploads before consumers. A ring of reclaim
/// events orders slot reuse after the relevant consumer replay without
/// serializing unrelated uploads.
pub struct TransferStream {
    stream: Arc<CudaStream>,
    publication: CudaEvent,
    reclaim: Vec<CudaEvent>,
    released: [u64; RECLAIM_FENCE_GENERATIONS],
    recorded: usize,
    published: bool,
}

impl TransferStream {
    /// Creates a non-blocking upload stream and its fence events.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let stream = context
            .new_stream()
            .map_err(|source| GpuError::driver("creating the slot transfer stream", source))?;
        let publication = context
            .new_event(None)
            .map_err(|source| GpuError::driver("creating the publication fence event", source))?;
        let reclaim = (0..RECLAIM_FENCE_GENERATIONS)
            .map(|_| {
                context
                    .new_event(None)
                    .map_err(|source| GpuError::driver("creating a reclaim fence event", source))
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            stream,
            publication,
            reclaim,
            released: [0; RECLAIM_FENCE_GENERATIONS],
            recorded: 0,
            published: false,
        })
    }

    /// The dedicated upload stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Records the publication fence after every upload enqueued so far.
    pub fn record_publication(&mut self) -> GpuResult<()> {
        self.publication
            .record(&self.stream)
            .map_err(|source| GpuError::driver("recording the publication fence", source))?;
        self.published = true;

        Ok(())
    }

    /// Enqueues `consumer`'s wait on the latest publication fence.
    ///
    /// This is the only ordering a replay may rely on: without it a captured
    /// graph could read a slot whose upload has not landed.
    pub fn wait_publication(&self, consumer: &CudaStream) -> GpuResult<()> {
        if !self.published {
            return Ok(());
        }
        if consumer.context().as_ref() != self.stream.context().as_ref() {
            return Err(GpuError::context(
                "transfer stream and consumer stream belong to different contexts",
            ));
        }
        consumer
            .wait(&self.publication)
            .map_err(|source| GpuError::driver("waiting on the publication fence", source))
    }

    /// Whether the latest publication fence has completed. Never blocks.
    ///
    /// This is diagnostic only; consumers still need an explicit wait.
    pub fn publication_completed(&self) -> GpuResult<bool> {
        if !self.published {
            return Ok(true);
        }
        self.publication
            .query()
            .map_err(|source| GpuError::driver("querying the publication fence", source))
    }

    /// Blocks the calling thread until the latest publication fence completes.
    ///
    /// This is the stall a miss costs. It is never optional: an unsatisfied
    /// miss must stall, never skip, reroute, or recompute.
    pub fn stall_for_publication(&self) -> GpuResult<()> {
        if !self.published {
            return Ok(());
        }
        self.publication
            .synchronize()
            .map_err(|source| GpuError::driver("stalling on the publication fence", source))
    }

    /// Records a reclaim fence on `consumer`, releasing replay `generation`.
    ///
    /// The consumer must be in order, so this fence also releases every earlier
    /// generation. A generation older than the latest release is refused.
    pub fn record_reclaim(&mut self, consumer: &CudaStream, generation: u64) -> GpuResult<()> {
        if consumer.context().as_ref() != self.stream.context().as_ref() {
            return Err(GpuError::context(
                "transfer stream and consumer stream belong to different contexts",
            ));
        }
        if let Some(newest) = self.newest_released()
            && generation < newest
        {
            return Err(GpuError::arena(format!(
                "reclaim fence for generation {generation} follows a release of {newest}"
            )));
        }
        let index = self.recorded % RECLAIM_FENCE_GENERATIONS;
        self.reclaim[index]
            .record(consumer)
            .map_err(|source| GpuError::driver("recording a reclaim fence", source))?;
        self.released[index] = generation;
        self.recorded = self.recorded.wrapping_add(1);

        Ok(())
    }

    /// Enqueues this stream's wait on the cheapest fence that releases
    /// `generation`.
    ///
    /// Generation zero has never been read and needs no wait. A nonzero
    /// generation without a recorded release is refused.
    pub fn wait_reclaim(&self, generation: u64) -> GpuResult<()> {
        if generation == 0 {
            return Ok(());
        }
        let live = self.live_reclaims();
        if live == 0 {
            return Err(GpuError::arena(format!(
                "reclaim generation {generation} has no recorded release"
            )));
        }
        let newest = self.released[self.reclaim_index(live - 1)];
        if generation > newest {
            return Err(GpuError::arena(format!(
                "reclaim generation {generation} exceeds latest release {newest}"
            )));
        }
        let index = (0..live)
            .map(|age| self.reclaim_index(age))
            .find(|&index| self.released[index] >= generation)
            .expect("the newest recorded fence releases this generation");
        self.stream
            .wait(&self.reclaim[index])
            .map_err(|source| GpuError::driver("waiting on a reclaim fence", source))
    }

    /// Enqueues this stream's wait on every reclaim fence recorded so far.
    ///
    /// The barrier form, for a caller that is about to invalidate residency
    /// wholesale rather than overwrite a known set of slots.
    pub fn wait_all_reclaims(&self) -> GpuResult<()> {
        let live = self.live_reclaims();
        if live == 0 {
            return Ok(());
        }
        self.stream
            .wait(&self.reclaim[self.reclaim_index(live - 1)])
            .map_err(|source| GpuError::driver("waiting on every reclaim fence", source))
    }

    /// Newest replay generation a recorded reclaim fence releases.
    pub fn released_generation(&self) -> Option<u64> {
        self.newest_released()
    }

    /// Reclaim fences recorded so far, saturating at the ring depth.
    fn live_reclaims(&self) -> usize {
        self.recorded.min(RECLAIM_FENCE_GENERATIONS)
    }

    /// Ring index of the entry `age` steps newer than the oldest held one.
    fn reclaim_index(&self, age: usize) -> usize {
        (self.recorded - self.live_reclaims() + age) % RECLAIM_FENCE_GENERATIONS
    }

    fn newest_released(&self) -> Option<u64> {
        let live = self.live_reclaims();
        (live > 0).then(|| self.released[self.reclaim_index(live - 1)])
    }

    /// Drains every upload enqueued on this stream.
    pub fn synchronize(&self) -> GpuResult<()> {
        self.stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing the slot transfer stream", source))
    }

    /// Enqueues one raw host-to-device copy on the upload stream.
    ///
    /// # Safety
    ///
    /// `destination` must address at least `bytes` of live device memory, and
    /// the selected pinned source range must stay allocated and immutable until
    /// this stream reaches the copy.
    pub(crate) unsafe fn enqueue_upload(
        &self,
        destination: u64,
        source: *const u8,
        bytes: usize,
    ) -> GpuResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|source| GpuError::driver("binding the transfer CUDA context", source))?;
        // SAFETY: the caller owns the destination validity and pinned-source
        // lifetime contract documented above; both ranges cover `bytes`.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                destination,
                source,
                bytes,
                self.stream.cu_stream(),
            )
        }
        .map_err(|source| GpuError::driver("enqueueing a slot upload", source))
    }
}

impl Drop for TransferStream {
    /// Drains the upload stream so no enqueued copy can outlive the pinned host
    /// pool or device arena this owner was streaming between.
    fn drop(&mut self) {
        let context = self.stream.context().clone();
        context.record_err(context.bind_to_thread());
        context.record_err(self.stream.synchronize());
    }
}

/// A fixed pinned ring that carries borrowed extents onto the transfer stream.
///
/// A target that leaves part of an item in its checkpoint mapping instead of the
/// pinned pool cannot hand those bytes to the driver directly. This bounded
/// page-locked ring gives each slot its own upload fence.
///
/// The fence is the whole safety argument. A ring slot is rewritten only after
/// the host has synchronized on that slot's own fence, recorded on the transfer
/// stream immediately after its copy was enqueued, so the driver never reads a
/// slot a later host copy is tearing. The ring is allocation-free after
/// construction; wrapping stalls only on this pool's own upload rate.
pub struct PinnedBounceRing {
    storage: PinnedHostPool,
    fences: Vec<CudaEvent>,
    fenced: Box<[bool]>,
    slot_count: usize,
    slot_bytes: usize,
    cursor: usize,
    waits: u64,
}

impl PinnedBounceRing {
    /// Allocates `slot_count` page-locked slots of `slot_bytes` and their fences.
    pub fn allocate(
        context: &Arc<CudaContext>,
        slot_count: usize,
        slot_bytes: usize,
    ) -> GpuResult<Self> {
        if slot_count == 0 || slot_bytes == 0 {
            return Err(GpuError::arena(
                "a pinned bounce ring needs a nonzero slot count and slot size",
            ));
        }
        if !slot_bytes.is_multiple_of(ALIGNMENT) {
            return Err(GpuError::arena(format!(
                "bounce slot extent {slot_bytes} is not a multiple of {ALIGNMENT} bytes"
            )));
        }
        let bytes = slot_count
            .checked_mul(slot_bytes)
            .ok_or_else(|| GpuError::arena("pinned bounce ring byte count overflows"))?;
        let storage = PinnedHostPool::allocate(context, bytes)?;
        let fences = (0..slot_count)
            .map(|_| {
                context.new_event(None).map_err(|source| {
                    GpuError::driver("creating a bounce-ring upload fence", source)
                })
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            storage,
            fences,
            fenced: vec![false; slot_count].into_boxed_slice(),
            slot_count,
            slot_bytes,
            cursor: 0,
            waits: 0,
        })
    }

    /// Page-locked bytes this ring holds.
    pub const fn byte_len(&self) -> usize {
        self.slot_count * self.slot_bytes
    }

    /// Slots the ring rotates through.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Bytes one ring slot carries.
    pub const fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Times a wrap made the host wait for a ring slot's upload to land.
    ///
    /// Diagnostic, and the direct evidence that the wraparound fence is on the
    /// live path rather than a branch a test never reaches.
    pub const fn wraparound_waits(&self) -> u64 {
        self.waits
    }

    /// Wall time the ring's page-locked allocation took.
    pub const fn pin_duration(&self) -> Duration {
        self.storage.pin_duration()
    }

    /// Copies `source` through the next ring slot and enqueues its upload.
    ///
    /// The slot stride is aligned, so a source shorter than the stride is
    /// admitted and exactly its own length is copied and uploaded; the padding
    /// tail is never read and never written.
    ///
    /// # Safety
    ///
    /// `destination` must address at least `source.len()` bytes of live device
    /// memory that stays live until `transfer` reaches the copy.
    pub unsafe fn upload(
        &mut self,
        transfer: &TransferStream,
        destination: u64,
        source: &[u8],
    ) -> GpuResult<()> {
        if source.is_empty() || source.len() > self.slot_bytes {
            return Err(GpuError::arena(format!(
                "bounce ring staged {} bytes into a {}-byte slot",
                source.len(),
                self.slot_bytes
            )));
        }
        if self.storage.context().as_ref() != transfer.stream.context().as_ref() {
            return Err(GpuError::context(
                "bounce ring and transfer stream belong to different contexts",
            ));
        }
        let slot = self.cursor % self.slot_count;
        if self.fenced[slot] {
            // The wraparound rule: this slot's previous upload must have landed
            // before the host may overwrite the bytes the driver was reading.
            self.fences[slot].synchronize().map_err(|source| {
                GpuError::driver("waiting on a bounce-ring upload fence", source)
            })?;
            self.waits = self.waits.wrapping_add(1);
        }
        let offset = slot * self.slot_bytes;
        self.storage.write(offset, source)?;
        let pointer = self.storage.source_pointer(offset, source.len())?;
        // SAFETY: the caller owns `destination`'s validity and lifetime;
        // `pointer` names this live page-locked ring slot, and the fence
        // recorded below is what proves the copy finished before the slot is
        // written again.
        unsafe { transfer.enqueue_upload(destination, pointer, source.len())? };
        self.fences[slot]
            .record(&transfer.stream)
            .map_err(|source| GpuError::driver("recording a bounce-ring upload fence", source))?;
        self.fenced[slot] = true;
        self.cursor = self.cursor.wrapping_add(1);

        Ok(())
    }
}

/// One sealed arena of address-stable slot extents and its indirection table.
///
/// The slot region and the item-to-slot table share a single allocation, so
/// every slot address and the table address are fixed for the owner's lifetime
/// and a captured graph may bake them. Only slot contents and table entries
/// change between replays, exactly as paged K/V pages do today.
pub struct DeviceSlotPool {
    arena: DeviceArena,
    staging: PinnedHostBuffer<u32>,
    staged: [bool; TABLE_GENERATIONS],
    fences: Vec<CudaEvent>,
    generation: usize,
    slots: ArenaRegion<u8>,
    table: ArenaRegion<u32>,
    item_count: usize,
    slot_count: usize,
    slot_bytes: usize,
    context: Arc<CudaContext>,
}

impl DeviceSlotPool {
    /// Allocates `slot_count` extents of `slot_bytes` and an `item_count` table.
    ///
    /// Returns only after the zeroing and the absent-sentinel table fill have
    /// completed, so no consumer can observe uninitialized bytes.
    pub fn new(
        context: &Arc<CudaContext>,
        item_count: usize,
        slot_count: usize,
        slot_bytes: usize,
    ) -> GpuResult<Self> {
        if item_count == 0 || slot_count == 0 || slot_bytes == 0 {
            return Err(GpuError::arena(
                "a device slot pool needs a nonzero item count, slot count, and slot size",
            ));
        }
        if slot_count > item_count {
            return Err(GpuError::arena(format!(
                "device slot pool has {slot_count} slots for only {item_count} items"
            )));
        }
        if u32::try_from(slot_count).is_err() || slot_count as u32 == ABSENT_SLOT {
            return Err(GpuError::arena(format!(
                "device slot pool of {slot_count} slots collides with the absent sentinel"
            )));
        }
        if !slot_bytes.is_multiple_of(ALIGNMENT) {
            return Err(GpuError::arena(format!(
                "slot extent {slot_bytes} is not a multiple of {ALIGNMENT} bytes"
            )));
        }
        let slot_region_bytes = slot_count
            .checked_mul(slot_bytes)
            .ok_or_else(|| GpuError::arena("device slot region byte count overflows"))?;
        let staging_entries = item_count
            .checked_mul(TABLE_GENERATIONS)
            .ok_or_else(|| GpuError::arena("indirection-table staging count overflows"))?;

        let mut layout = ArenaLayout::new();
        let slots = layout.reserve::<u8>(slot_region_bytes, ALIGNMENT)?;
        let table = layout.reserve::<u32>(item_count, ALIGNMENT)?;
        let stream = context
            .new_stream()
            .map_err(|source| GpuError::driver("creating the slot-pool setup stream", source))?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        arena.fill(&stream, table, u8::MAX)?;
        let staging = PinnedHostBuffer::zeroed(context, staging_entries).map_err(|source| {
            GpuError::driver("allocating the indirection-table staging ring", source)
        })?;
        let fences = (0..TABLE_GENERATIONS)
            .map(|_| {
                context.new_event(None).map_err(|source| {
                    GpuError::driver("creating an indirection-table generation fence", source)
                })
            })
            .collect::<GpuResult<Vec<_>>>()?;
        stream
            .synchronize()
            .map_err(|source| GpuError::driver("synchronizing slot-pool initialization", source))?;

        Ok(Self {
            arena,
            staging,
            staged: [false; TABLE_GENERATIONS],
            fences,
            generation: 0,
            slots,
            table,
            item_count,
            slot_count,
            slot_bytes,
            context: Arc::clone(context),
        })
    }

    /// Items the indirection table addresses.
    pub const fn item_count(&self) -> usize {
        self.item_count
    }

    /// Address-stable slot extents in the pool.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Bytes in one slot extent.
    pub const fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Complete device allocation bytes including alignment padding.
    pub const fn device_bytes(&self) -> usize {
        self.arena.byte_len()
    }

    /// Bytes covered by the slot extents alone.
    pub const fn slot_region_bytes(&self) -> usize {
        self.slots.byte_len()
    }

    /// Bytes covered by the device-side indirection table.
    pub const fn table_bytes(&self) -> usize {
        self.table.byte_len()
    }

    /// Page-locked bytes held by the table staging ring.
    pub fn staging_bytes(&self) -> usize {
        self.staging.num_bytes()
    }

    /// CUDA context shared by the arena, the staging ring, and every consumer.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of the single device allocation.
    pub fn base_address(&self) -> u64 {
        self.arena.base_address()
    }

    /// Stable device address of the item-to-slot indirection table.
    pub fn table_address(&self) -> GpuResult<u64> {
        Ok(self.arena.address(self.table)?.addr() as u64)
    }

    /// Stable device address of one slot extent.
    pub fn slot_address(&self, slot: usize) -> GpuResult<u64> {
        self.require_slot(slot)?;
        let offset = slot
            .checked_mul(self.slot_bytes)
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| GpuError::arena("slot device offset overflows"))?;
        (self.arena.address(self.slots)?.addr() as u64)
            .checked_add(offset)
            .ok_or_else(|| GpuError::arena("slot device address overflows"))
    }

    /// Enqueues one complete slot upload on the dedicated transfer stream.
    ///
    /// # Safety
    ///
    /// `source` must stay allocated and its selected range immutable until
    /// `transfer` reaches the copy; the caller proves that with a publication
    /// fence before reusing those host bytes.
    pub unsafe fn enqueue_slot_upload(
        &self,
        transfer: &TransferStream,
        slot: usize,
        source: &PinnedHostPool,
        source_offset: usize,
    ) -> GpuResult<()> {
        // SAFETY: forwarded verbatim; the whole-extent range is the widest this
        // slot admits and the caller's pinned-source contract is unchanged.
        unsafe {
            self.enqueue_slot_extent_upload(
                transfer,
                slot,
                0,
                source,
                source_offset,
                self.slot_bytes,
            )
        }
    }

    /// Enqueues one checked byte range of a slot on the transfer stream.
    ///
    /// The split-extent form, for a target whose slot is filled from more than
    /// one host source, such as a pooled secondary beside a borrowed primary
    /// that arrives through a [`PinnedBounceRing`], for instance. The device
    /// slot layout is unchanged either way: a consuming kernel still reads one
    /// contiguous extent at one stable address.
    ///
    /// # Safety
    ///
    /// `source` must stay allocated and its selected range immutable until
    /// `transfer` reaches the copy; the caller proves that with a publication
    /// fence before reusing those host bytes.
    pub unsafe fn enqueue_slot_extent_upload(
        &self,
        transfer: &TransferStream,
        slot: usize,
        slot_offset: usize,
        source: &PinnedHostPool,
        source_offset: usize,
        bytes: usize,
    ) -> GpuResult<()> {
        self.require_slot(slot)?;
        self.require_context(source.context(), "uploading into a device slot")?;
        self.require_context(transfer.stream.context(), "uploading into a device slot")?;
        let end = slot_offset
            .checked_add(bytes)
            .ok_or_else(|| GpuError::arena("slot upload range overflows"))?;
        if end > self.slot_bytes {
            return Err(GpuError::arena(format!(
                "slot upload {slot_offset}..{end} exceeds a {}-byte slot extent",
                self.slot_bytes
            )));
        }
        let destination = self
            .slot_address(slot)?
            .checked_add(slot_offset as u64)
            .ok_or_else(|| GpuError::arena("slot device address overflows"))?;
        let pointer = source.source_pointer(source_offset, bytes)?;
        // SAFETY: `destination` names a checked range inside this live arena's
        // slot extent and `pointer` names `bytes` of the caller's live pinned
        // allocation.
        unsafe { transfer.enqueue_upload(destination, pointer, bytes) }
    }

    /// Stages the complete item-to-slot table and enqueues one publication copy.
    ///
    /// The copy is enqueued on the same transfer stream as the round's slot
    /// uploads and after them, so no observer can see a table entry whose slot
    /// upload has not been issued. Cross-stream visibility still requires the
    /// publication fence.
    pub fn publish_table(&mut self, transfer: &TransferStream, table: &[u32]) -> GpuResult<()> {
        if table.len() != self.item_count {
            return Err(GpuError::arena(format!(
                "indirection table has {} entries for {} items",
                table.len(),
                self.item_count
            )));
        }
        self.require_context(transfer.stream.context(), "publishing an indirection table")?;
        if let Some(&slot) = table
            .iter()
            .find(|&&slot| slot != ABSENT_SLOT && slot as usize >= self.slot_count)
        {
            return Err(GpuError::arena(format!(
                "indirection table names slot {slot} outside a {}-slot pool",
                self.slot_count
            )));
        }

        let generation = self.generation % TABLE_GENERATIONS;
        if self.staged[generation] {
            self.fences[generation].synchronize().map_err(|source| {
                GpuError::driver("waiting on an indirection-table generation fence", source)
            })?;
        }
        let start = generation * self.item_count;
        self.staging.as_mut_slice()[start..start + self.item_count].copy_from_slice(table);
        let destination = self.table_address()?;
        // SAFETY: `destination` names this live arena's checked table region and
        // the staging generation is page-locked for the pool's whole lifetime;
        // its fence below proves the copy finished before the generation is
        // written again.
        unsafe {
            transfer.enqueue_upload(
                destination,
                self.staging.as_ptr().add(start).cast::<u8>(),
                self.table.byte_len(),
            )?;
        }
        self.fences[generation]
            .record(&transfer.stream)
            .map_err(|source| {
                GpuError::driver("recording an indirection-table generation fence", source)
            })?;
        self.staged[generation] = true;
        self.generation = self.generation.wrapping_add(1);

        Ok(())
    }

    /// Reads one complete slot extent back to the host.
    pub fn read_slot(&self, stream: &CudaStream, slot: usize) -> GpuResult<Vec<u8>> {
        self.require_slot(slot)?;
        let start = slot
            .checked_mul(self.slot_bytes)
            .ok_or_else(|| GpuError::arena("slot readback offset overflows"))?;
        self.arena
            .copy_slice_to_host(stream, self.slots, start, self.slot_bytes)
    }

    /// Reads the complete device-side indirection table back to the host.
    pub fn read_table(&self, stream: &CudaStream) -> GpuResult<Vec<u32>> {
        self.arena.copy_to_host(stream, self.table)
    }

    fn require_slot(&self, slot: usize) -> GpuResult<()> {
        if slot >= self.slot_count {
            return Err(GpuError::arena(format!(
                "slot {slot} is outside a {}-slot pool",
                self.slot_count
            )));
        }

        Ok(())
    }

    fn require_context(&self, other: &Arc<CudaContext>, operation: &str) -> GpuResult<()> {
        if other.as_ref() != self.context.as_ref() {
            return Err(GpuError::context(format!(
                "{operation} requires one shared CUDA context"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ABSENT_SLOT, DeviceSlotPool, PinnedBounceRing, PinnedHostPool, RECLAIM_FENCE_GENERATIONS,
        TABLE_GENERATIONS, TransferStream,
    };
    use crate::{CudaContext, GpuErrorCode};

    #[test]
    fn absent_sentinel_and_generation_count_are_stable() {
        assert_eq!(ABSENT_SLOT, u32::MAX);
        assert_eq!(TABLE_GENERATIONS, 4);
        assert_eq!(RECLAIM_FENCE_GENERATIONS, 4);
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn the_reclaim_ring_picks_the_oldest_fence_that_releases_a_generation() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let consumer = context.new_stream().unwrap();
        let mut transfer = TransferStream::new(&context).unwrap();

        // Nothing recorded: nothing has been replayed, so nothing is reclaimed.
        assert_eq!(transfer.released_generation(), None);
        assert_eq!(transfer.live_reclaims(), 0);
        transfer.wait_reclaim(0).unwrap();
        assert_eq!(
            transfer.wait_reclaim(9).unwrap_err().code(),
            GpuErrorCode::Arena
        );

        for generation in 1..=6u64 {
            transfer.record_reclaim(&consumer, generation).unwrap();
        }
        // The ring holds the newest four releases, oldest first.
        assert_eq!(transfer.live_reclaims(), RECLAIM_FENCE_GENERATIONS);
        assert_eq!(transfer.released_generation(), Some(6));
        assert_eq!(
            (0..RECLAIM_FENCE_GENERATIONS)
                .map(|age| transfer.released[transfer.reclaim_index(age)])
                .collect::<Vec<_>>(),
            [3, 4, 5, 6]
        );
        // A slot last read long ago resolves to the oldest held fence, which is
        // the one most likely already retired; a slot read last round resolves
        // to the newest.
        for (required, expected) in [(1u64, 3u64), (3, 3), (4, 4), (6, 6)] {
            let live = transfer.live_reclaims();
            let index = (0..live)
                .map(|age| transfer.reclaim_index(age))
                .find(|&index| transfer.released[index] >= required)
                .unwrap_or_else(|| transfer.reclaim_index(live - 1));
            assert_eq!(transfer.released[index], expected, "required {required}");
            transfer.wait_reclaim(required).unwrap();
        }
        assert_eq!(
            transfer.wait_reclaim(7).unwrap_err().code(),
            GpuErrorCode::Arena
        );
        transfer.wait_all_reclaims().unwrap();

        // A release that walks the timeline backwards is refused.
        assert_eq!(
            transfer.record_reclaim(&consumer, 5).unwrap_err().code(),
            GpuErrorCode::Arena
        );
        consumer.synchronize().unwrap();
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn the_bounce_ring_fences_every_slot_before_it_wraps_onto_it() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let transfer = TransferStream::new(&context).unwrap();
        let mut pool = DeviceSlotPool::new(&context, 8, 8, 256).unwrap();
        let mut ring = PinnedBounceRing::allocate(&context, 2, 256).unwrap();

        assert_eq!(ring.byte_len(), 512);
        assert_eq!((ring.slot_count(), ring.slot_bytes()), (2, 256));
        assert_eq!(ring.wraparound_waits(), 0);

        // Eight uploads through two ring slots wrap the ring three times; each
        // item's pattern is distinct, so a slot rewritten under an in-flight
        // copy would surface the wrong item's bytes.
        for slot in 0..8usize {
            let source = vec![0x10u8 + slot as u8; 256];
            // SAFETY: `pool` outlives the publication stall below.
            unsafe {
                ring.upload(&transfer, pool.slot_address(slot).unwrap(), &source)
                    .unwrap();
            }
        }
        pool.publish_table(&transfer, &[0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        assert_eq!(ring.wraparound_waits(), 6);

        for slot in 0..8usize {
            assert_eq!(
                pool.read_slot(&stream, slot).unwrap(),
                vec![0x10u8 + slot as u8; 256],
                "bounce slot {slot}"
            );
        }
        assert_eq!(
            PinnedBounceRing::allocate(&context, 2, 300)
                .err()
                .unwrap()
                .code(),
            GpuErrorCode::Arena
        );
        // SAFETY: both calls are rejected by the checked slot width before any
        // enqueue: a source wider than the slot, and an empty one.
        unsafe {
            assert_eq!(
                ring.upload(&transfer, pool.slot_address(0).unwrap(), &[0u8; 512])
                    .unwrap_err()
                    .code(),
                GpuErrorCode::Arena
            );
            assert_eq!(
                ring.upload(&transfer, pool.slot_address(0).unwrap(), &[])
                    .unwrap_err()
                    .code(),
                GpuErrorCode::Arena
            );
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn slot_pool_publishes_a_fenced_table_over_stable_addresses() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let stream = context.new_stream().unwrap();
        let mut transfer = TransferStream::new(&context).unwrap();
        let mut host = PinnedHostPool::allocate(&context, 3 * 256).unwrap();
        host.write(0, &[0x11; 256]).unwrap();
        host.write(256, &[0x22; 256]).unwrap();
        host.write(512, &[0x33; 256]).unwrap();
        let mut pool = DeviceSlotPool::new(&context, 3, 2, 256).unwrap();
        let addresses = [
            pool.slot_address(0).unwrap(),
            pool.slot_address(1).unwrap(),
            pool.table_address().unwrap(),
        ];

        assert_eq!(pool.read_table(&stream).unwrap(), [ABSENT_SLOT; 3]);

        // SAFETY: `host` outlives every copy; the publication stall below
        // proves each upload completed before its source is reused.
        unsafe {
            pool.enqueue_slot_upload(&transfer, 0, &host, 0).unwrap();
            pool.enqueue_slot_upload(&transfer, 1, &host, 512).unwrap();
        }
        pool.publish_table(&transfer, &[0, ABSENT_SLOT, 1]).unwrap();
        transfer.record_publication().unwrap();
        transfer.stall_for_publication().unwrap();

        assert_eq!(pool.read_slot(&stream, 0).unwrap(), [0x11; 256]);
        assert_eq!(pool.read_slot(&stream, 1).unwrap(), [0x33; 256]);
        assert_eq!(pool.read_table(&stream).unwrap(), [0, ABSENT_SLOT, 1]);

        // Rotating the whole staging ring must keep publishing exact tables.
        for round in 0..2 * TABLE_GENERATIONS {
            let table = if round % 2 == 0 {
                [ABSENT_SLOT, 0, 1]
            } else {
                [1, 0, ABSENT_SLOT]
            };
            pool.publish_table(&transfer, &table).unwrap();
            transfer.record_publication().unwrap();
            transfer.stall_for_publication().unwrap();
            assert_eq!(pool.read_table(&stream).unwrap(), table);
        }
        assert_eq!(
            [
                pool.slot_address(0).unwrap(),
                pool.slot_address(1).unwrap(),
                pool.table_address().unwrap()
            ],
            addresses
        );
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn slot_pool_rejects_unaligned_extents_and_out_of_range_entries() {
        let context = CudaContext::new(0).unwrap();
        assert_eq!(context.compute_capability().unwrap(), (12, 0));
        let transfer = TransferStream::new(&context).unwrap();
        let host = PinnedHostPool::allocate(&context, 256).unwrap();

        for rejected in [(4, 2, 300), (2, 4, 256), (4, 0, 256)] {
            let error = DeviceSlotPool::new(&context, rejected.0, rejected.1, rejected.2)
                .err()
                .unwrap();
            assert_eq!(error.code(), GpuErrorCode::Arena);
        }

        let mut pool = DeviceSlotPool::new(&context, 4, 2, 256).unwrap();
        assert_eq!(
            pool.publish_table(&transfer, &[0, 1, 0])
                .unwrap_err()
                .code(),
            GpuErrorCode::Arena
        );
        assert_eq!(
            pool.publish_table(&transfer, &[0, 1, 2, 0])
                .unwrap_err()
                .code(),
            GpuErrorCode::Arena
        );
        // SAFETY: both calls are rejected by checked bounds before any enqueue.
        unsafe {
            assert_eq!(
                pool.enqueue_slot_upload(&transfer, 2, &host, 0)
                    .unwrap_err()
                    .code(),
                GpuErrorCode::Arena
            );
            assert_eq!(
                pool.enqueue_slot_upload(&transfer, 0, &host, 128)
                    .unwrap_err()
                    .code(),
                GpuErrorCode::Arena
            );
        }
        assert_eq!(
            pool.slot_address(2).unwrap_err().code(),
            GpuErrorCode::Arena
        );
        assert!(host.slice(0..257).is_err());
    }
}

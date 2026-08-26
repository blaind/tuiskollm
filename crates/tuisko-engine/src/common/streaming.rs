//! Streaming-resident weight ownership: pinned host pool plus VRAM slot cache.
//!
//! [`StreamingWeightPool`] owns pinned or mapped host extents, a device slot
//! cache, and the item-to-slot table. Geometry and slot budget are parameters;
//! residency remains an admission-time partition.
//!
//! # Publication ordering law
//!
//! Cache state must not change produced bits. Each admitted round has a monotone
//! generation, and each slot records its latest reader generation.
//!
//! 1. Before overwriting slots, wait on the oldest consumer release covering
//!    their newest reader. Generation zero needs no wait; a missing release is
//!    refused.
//! 2. Enqueue slot uploads and then the complete table on one transfer stream.
//! 3. Record one publication event after the table copy.
//! 4. Make consumers wait on that event before replay.
//! 5. Fence each pinned bounce slot before host reuse.
//! 6. Plan against unchanged residency, validate sources, enqueue every upload,
//!    and only then commit. A pre-enqueue failure is retryable; a later failure
//!    poisons the pool.
//!
//! Rule 1 also protects table rewrites: newly admitted entries were absent,
//! evicted entries name reclaimed slots, and every other entry is unchanged.
//! The consumer owes one in-order stream and records a release after the last
//! replay of a round, before the pool's next round.
//!
//! [`StreamingWeightPool::require`] adds the stall: it blocks the host on the
//! publication fence, so every requested item is resident before it returns. A
//! miss it cannot satisfy is a stall, never a skip, a reroute, or a recompute.
//! [`StreamingWeightPool::prefetch`] is the same round without the stall and is
//! a performance lever only; both publish identical bytes.
//!
//! # Host postures
//!
//! [`StreamingPrimarySource::Pinned`] pools the whole extent.
//! [`StreamingPrimarySource::Mapped`] pools only the secondary extent and
//! stages the borrowed primary through a bounce ring. Both postures publish the
//! same device layout and report their host classes separately.

use crate::common::math::{checked_sum, product};
use crate::{EngineError, EngineResult, StreamingResidencyAccounting};
use std::sync::Arc;
use std::time::Duration;
use tuisko_gpu::{
    ABSENT_SLOT, CudaContext, CudaStream, DeviceSlotPool, INDIRECTION_TABLE_GENERATIONS,
    PinnedBounceRing, PinnedHostPool, TransferStream,
};

/// Indirection-table entry for an item that currently owns no slot.
pub const STREAMING_ABSENT_SLOT: u32 = ABSENT_SLOT;

/// Slot-ownership entry for a slot that currently holds no item.
pub const STREAMING_ABSENT_ITEM: u32 = u32::MAX;

/// Device address alignment shared by every slot extent.
const ALIGNMENT: usize = 256;

/// Where an item's primary extent lives on the host, as an admission constant.
///
/// The choice is arithmetic, not preference: a pool whose pinned bytes exceed
/// the box's usable RAM cannot be allocated at all. It is resolved once, at
/// admission, and pinned by accounting tests for both values; nothing infers it
/// at runtime and no fully-resident target may grow a runtime residency choice
/// from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingPrimarySource {
    /// The primary extent is copied into the pinned pool at admission, so one
    /// upload fills a slot and the pinned class carries the whole item pool.
    Pinned,
    /// The primary extent stays in the target's borrowed file mapping and
    /// reaches the device through the pinned bounce ring, so it counts against
    /// `host_mapped_bytes` and the pinned pool carries only secondary extents.
    Mapped,
}

impl StreamingPrimarySource {
    /// Whether this posture pools the primary extent.
    pub const fn is_pinned(self) -> bool {
        matches!(self, Self::Pinned)
    }
}

/// Borrowed, file-backed source of primary extents in the mapped posture.
///
/// The implementer owns the mapping, a checkpoint `mmap` in production, and
/// hands back one item's contiguous primary extent. Those bytes must be the
/// represented source words, byte-identical to what the pinned posture would
/// have staged, and the mapping must outlive the pool that borrows from it.
pub trait StreamingMappedPrimary: Send + Sync {
    /// One item's contiguous primary extent, exactly the admitted length.
    fn primary_extent(&self, item: usize) -> EngineResult<&[u8]>;
}

/// Host plan of one streaming-resident item pool.
///
/// Every byte count is fixed at admission and pinned by accounting tests. An
/// item's admitted extent is one contiguous primary plane set plus an optional
/// second plane set (staged scales, for instance); the host pool and the device
/// slots both use that extent as their stride, so one item is one contiguous
/// read and one contiguous upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingWeightLayout {
    item_count: usize,
    primary_extent_bytes: usize,
    secondary_extent_bytes: usize,
    stride_bytes: usize,
    slot_count: usize,
    slot_region_bytes: usize,
    table_bytes: usize,
    table_staging_bytes: usize,
    device_bytes: usize,
    host_pool_bytes: usize,
    host_stride_bytes: usize,
    primary_source: StreamingPrimarySource,
    bounce_slot_count: usize,
    bounce_slot_bytes: usize,
    bounce_ring_bytes: usize,
    host_pinned_bytes: usize,
    host_mapped_bytes: usize,
}

impl StreamingWeightLayout {
    /// Plans `item_count` items of one primary and one optional second extent
    /// over an exact device slot budget, with the whole extent pinned.
    ///
    /// The fully-pinned posture: `host_pool_bytes` is the complete item pool and
    /// nothing is file-backed.
    pub fn build(
        item_count: usize,
        primary_extent_bytes: usize,
        secondary_extent_bytes: Option<usize>,
        slot_count: usize,
    ) -> EngineResult<Self> {
        Self::plan(
            item_count,
            primary_extent_bytes,
            secondary_extent_bytes,
            slot_count,
            StreamingPrimarySource::Pinned,
            0,
        )
    }

    /// Plans the same inventory with the primary extent borrowed from a mapping.
    ///
    /// `bounce_slot_count` sizes the pinned ring. The device plan is identical
    /// to [`Self::build`]; only the host classes move.
    pub fn build_mapped_primary(
        item_count: usize,
        primary_extent_bytes: usize,
        secondary_extent_bytes: Option<usize>,
        slot_count: usize,
        bounce_slot_count: usize,
    ) -> EngineResult<Self> {
        if bounce_slot_count == 0 {
            return Err(EngineError::layout(
                "a mapped-primary streaming pool needs at least one bounce slot",
            ));
        }
        Self::plan(
            item_count,
            primary_extent_bytes,
            secondary_extent_bytes,
            slot_count,
            StreamingPrimarySource::Mapped,
            bounce_slot_count,
        )
    }

    fn plan(
        item_count: usize,
        primary_extent_bytes: usize,
        secondary_extent_bytes: Option<usize>,
        slot_count: usize,
        primary_source: StreamingPrimarySource,
        bounce_slot_count: usize,
    ) -> EngineResult<Self> {
        if item_count == 0 {
            return Err(EngineError::layout(
                "a streaming weight pool needs at least one item",
            ));
        }
        if primary_extent_bytes == 0 {
            return Err(EngineError::layout(
                "a streaming weight item needs a nonzero primary extent",
            ));
        }
        if slot_count == 0 || slot_count > item_count {
            return Err(EngineError::layout(format!(
                "streaming slot budget {slot_count} is outside 1..={item_count}"
            )));
        }
        let secondary_extent_bytes = secondary_extent_bytes.unwrap_or(0);
        let extent = checked_sum(
            "streaming item extent",
            primary_extent_bytes,
            secondary_extent_bytes,
        )?;
        let stride_bytes = extent
            .checked_next_multiple_of(ALIGNMENT)
            .ok_or_else(|| EngineError::layout("streaming item stride overflows"))?;
        let slot_region_bytes = product("streaming slot region", slot_count, stride_bytes)?;
        let table_bytes = product("streaming indirection table", item_count, size_of::<u32>())?;
        // The pool publishes the complete table out of a rotating pinned ring,
        // so those page-locked bytes belong to the host-pinned class too.
        let table_staging_bytes = product(
            "streaming table staging",
            table_bytes,
            INDIRECTION_TABLE_GENERATIONS,
        )?;
        let device_bytes = checked_sum("streaming device bytes", slot_region_bytes, table_bytes)?;
        // The pinned pool carries the whole extent when the primary is pooled,
        // and only the secondary extent when the primary is borrowed. Both
        // strides are aligned so an item is still one contiguous pinned read.
        let host_stride_bytes = if primary_source.is_pinned() {
            stride_bytes
        } else {
            secondary_extent_bytes
                .checked_next_multiple_of(ALIGNMENT)
                .ok_or_else(|| EngineError::layout("streaming pinned item stride overflows"))?
        };
        let host_pool_bytes = product("streaming host pool", item_count, host_stride_bytes)?;
        let (bounce_slot_bytes, bounce_ring_bytes, host_mapped_bytes) =
            if primary_source.is_pinned() {
                (0, 0, 0)
            } else {
                let slot = primary_extent_bytes
                    .checked_next_multiple_of(ALIGNMENT)
                    .ok_or_else(|| EngineError::layout("streaming bounce slot overflows"))?;
                (
                    slot,
                    product("streaming bounce ring", bounce_slot_count, slot)?,
                    // The mapped class counts the file's own bytes: the mapping
                    // is the checkpoint, so it carries no alignment padding.
                    product("streaming mapped pool", item_count, primary_extent_bytes)?,
                )
            };
        let host_pinned_bytes = checked_sum(
            "streaming pinned host bytes",
            checked_sum(
                "streaming pooled and table bytes",
                host_pool_bytes,
                table_staging_bytes,
            )?,
            bounce_ring_bytes,
        )?;

        Ok(Self {
            item_count,
            primary_extent_bytes,
            secondary_extent_bytes,
            stride_bytes,
            slot_count,
            slot_region_bytes,
            table_bytes,
            table_staging_bytes,
            device_bytes,
            host_pool_bytes,
            host_stride_bytes,
            primary_source,
            bounce_slot_count,
            bounce_slot_bytes,
            bounce_ring_bytes,
            host_pinned_bytes,
            host_mapped_bytes,
        })
    }

    /// Items the extent table describes.
    pub const fn item_count(&self) -> usize {
        self.item_count
    }

    /// Contiguous primary plane-set bytes per item.
    pub const fn primary_extent_bytes(&self) -> usize {
        self.primary_extent_bytes
    }

    /// Second plane-set bytes per item, zero when the target stages none.
    pub const fn secondary_extent_bytes(&self) -> usize {
        self.secondary_extent_bytes
    }

    /// Admitted extent stride shared by the host pool and every device slot.
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Extent bytes an item actually occupies inside its stride.
    pub const fn extent_bytes(&self) -> usize {
        self.primary_extent_bytes + self.secondary_extent_bytes
    }

    /// Stride bytes an item leaves unused for alignment.
    pub const fn extent_padding_bytes(&self) -> usize {
        self.stride_bytes - self.extent_bytes()
    }

    /// Address-stable device slots in the cache.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Device bytes covered by the slot extents alone.
    pub const fn slot_region_bytes(&self) -> usize {
        self.slot_region_bytes
    }

    /// Device bytes covered by the item-to-slot indirection table.
    pub const fn table_bytes(&self) -> usize {
        self.table_bytes
    }

    /// Page-locked bytes covered by the item pool alone.
    pub const fn host_pool_bytes(&self) -> usize {
        self.host_pool_bytes
    }

    /// Fraction of the item pool the slot budget keeps device-resident.
    pub fn resident_fraction(&self) -> f64 {
        self.slot_count as f64 / self.item_count as f64
    }

    /// Page-locked bytes covered by the indirection-table staging ring.
    pub const fn table_staging_bytes(&self) -> usize {
        self.table_staging_bytes
    }

    /// Admitted host posture of the primary extent.
    pub const fn primary_source(&self) -> StreamingPrimarySource {
        self.primary_source
    }

    /// Pinned pool stride per item: the whole extent, or the secondary alone.
    pub const fn host_stride_bytes(&self) -> usize {
        self.host_stride_bytes
    }

    /// Slots the pinned bounce ring rotates through, zero when it has none.
    pub const fn bounce_slot_count(&self) -> usize {
        self.bounce_slot_count
    }

    /// Bytes one bounce slot carries, zero when the pool has no ring.
    pub const fn bounce_slot_bytes(&self) -> usize {
        self.bounce_slot_bytes
    }

    /// Page-locked bytes covered by the primary-extent bounce ring.
    pub const fn bounce_ring_bytes(&self) -> usize {
        self.bounce_ring_bytes
    }

    /// Bytes this item's pinned extents occupy inside the host stride.
    pub const fn host_extent_bytes(&self) -> usize {
        if self.primary_source.is_pinned() {
            self.extent_bytes()
        } else {
            self.secondary_extent_bytes
        }
    }

    fn require_measured_allocations(
        &self,
        device_bytes: usize,
        staging_bytes: usize,
        bounce_bytes: usize,
    ) -> EngineResult<()> {
        if device_bytes != self.device_bytes
            || staging_bytes != self.table_staging_bytes
            || bounce_bytes != self.bounce_ring_bytes
        {
            return Err(EngineError::layout(format!(
                "streaming pool allocated {device_bytes} device, {staging_bytes} staging and {bounce_bytes} bounce bytes for a plan of {}, {} and {}",
                self.device_bytes, self.table_staging_bytes, self.bounce_ring_bytes
            )));
        }

        Ok(())
    }

    fn item_offset(&self, item: usize) -> EngineResult<usize> {
        if item >= self.item_count {
            return Err(EngineError::route(format!(
                "streaming item {item} is outside 0..{}",
                self.item_count
            )));
        }
        product("streaming item offset", item, self.host_stride_bytes)
    }
}

impl crate::LayerMemoryLayout for StreamingWeightLayout {
    fn arena_bytes(&self) -> usize {
        self.device_bytes
    }

    fn resident_weight_bytes(&self) -> usize {
        self.slot_region_bytes
    }

    fn cache_bytes(&self) -> usize {
        0
    }

    fn workspace_bytes(&self) -> usize {
        self.table_bytes
    }
}

impl StreamingResidencyAccounting for StreamingWeightLayout {
    fn device_resident_bytes(&self) -> usize {
        self.device_bytes
    }

    fn host_pinned_bytes(&self) -> usize {
        self.host_pinned_bytes
    }

    fn host_mapped_bytes(&self) -> usize {
        self.host_mapped_bytes
    }
}

/// One item admitted into a slot, and the item that lost it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingSlotAssignment {
    item: u32,
    slot: u32,
    evicted: u32,
}

impl StreamingSlotAssignment {
    /// Item this round admitted.
    pub const fn item(self) -> u32 {
        self.item
    }

    /// Slot the item now owns.
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Item the assignment evicted, or [`STREAMING_ABSENT_ITEM`].
    pub const fn evicted(self) -> u32 {
        self.evicted
    }
}

/// What one `require` or `prefetch` round did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamingRound {
    hits: usize,
    misses: usize,
    uploaded_bytes: usize,
    reclaim_generation: u64,
    stalled: bool,
}

impl StreamingRound {
    /// Requested items already resident.
    pub const fn hits(self) -> usize {
        self.hits
    }

    /// Requested items this round uploaded.
    pub const fn misses(self) -> usize {
        self.misses
    }

    /// Host-to-device bytes this round enqueued.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }

    /// Newest replay generation this round must reclaim before it may overwrite.
    ///
    /// Rule 1 of the publication ordering law: the largest last-reader
    /// generation over the slots the round evicts, and zero when it overwrites
    /// only slots no round has ever requested. A round whose reclaim generation
    /// is older than the replay in flight does not wait for that replay.
    pub const fn reclaim_generation(self) -> u64 {
        self.reclaim_generation
    }

    /// Whether the round blocked the host on the publication fence.
    pub const fn stalled(self) -> bool {
        self.stalled
    }
}

/// One step of a planned round: an admission, or a hit that touches a slot.
///
/// Planning records the whole round as a sequence of these before any of it is
/// committed, so committing is a replay rather than a re-derivation and cannot
/// disagree with the uploads the plan produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedStep {
    item: u32,
    slot: u32,
    evicted: u32,
    admitted: bool,
}

/// Global plain-LRU slot ownership over one item inventory.
///
/// Allocation-free after construction and completely host-side: it decides
/// which slot an item occupies and which item loses a slot, and nothing else.
/// Eviction order has no numerical authority. This type never touches device
/// state, but its deterministic order is pinned by test.
///
/// # Plan then commit
///
/// A round is resolved in two steps. [`Self::plan_round`] decides every hit and
/// every admission against the *current* residency and writes the answer into
/// this type's plan scratch, leaving residency, recency, the reader generations
/// and the counters untouched. [`Self::commit_planned_round`] replays that plan
/// after the owner resolves sources and enqueues uploads.
#[derive(Debug)]
pub struct StreamingSlotCache {
    item_slot: Box<[u32]>,
    slot_item: Box<[u32]>,
    older: Box<[u32]>,
    newer: Box<[u32]>,
    pinned_round: Box<[u64]>,
    least_recent: u32,
    most_recent: u32,
    round: u64,
    hits: u64,
    misses: u64,
    /// The round decided but not yet committed. Scratch, never residency: a
    /// plan that is replaced or dropped changes nothing an observer can see.
    ///
    /// One step per *slot* the round touches, not one per request: a repeated
    /// item is a hit on a slot the plan already holds, so it adds to the round's
    /// hit count without adding a step. A round touches at most `slot_count`
    /// distinct slots, which is what keeps planning allocation-free.
    plan: Box<[PlannedStep]>,
    plan_len: usize,
    plan_hits: usize,
    plan_misses: usize,
    planned: bool,
}

impl StreamingSlotCache {
    /// Creates an empty cache over `item_count` items and `slot_count` slots.
    pub fn new(item_count: usize, slot_count: usize) -> EngineResult<Self> {
        if item_count == 0 || slot_count == 0 || slot_count > item_count {
            return Err(EngineError::layout(format!(
                "streaming slot cache of {slot_count} slots is outside 1..={item_count}"
            )));
        }
        if u32::try_from(item_count).is_err() || item_count as u32 == STREAMING_ABSENT_ITEM {
            return Err(EngineError::layout(format!(
                "streaming item inventory of {item_count} collides with the absent sentinel"
            )));
        }

        Ok(Self {
            item_slot: vec![STREAMING_ABSENT_SLOT; item_count].into_boxed_slice(),
            slot_item: vec![STREAMING_ABSENT_ITEM; slot_count].into_boxed_slice(),
            older: vec![STREAMING_ABSENT_SLOT; slot_count].into_boxed_slice(),
            newer: vec![STREAMING_ABSENT_SLOT; slot_count].into_boxed_slice(),
            pinned_round: vec![0; slot_count].into_boxed_slice(),
            least_recent: STREAMING_ABSENT_SLOT,
            most_recent: STREAMING_ABSENT_SLOT,
            round: 0,
            hits: 0,
            misses: 0,
            // A round admits at most `slot_count` distinct items, so one plan
            // never outgrows this and planning stays allocation-free.
            plan: vec![
                PlannedStep {
                    item: STREAMING_ABSENT_ITEM,
                    slot: STREAMING_ABSENT_SLOT,
                    evicted: STREAMING_ABSENT_ITEM,
                    admitted: false,
                };
                slot_count
            ]
            .into_boxed_slice(),
            plan_len: 0,
            plan_hits: 0,
            plan_misses: 0,
            planned: false,
        })
    }

    /// Items this cache can address.
    pub const fn item_count(&self) -> usize {
        self.item_slot.len()
    }

    /// Slots this cache owns.
    pub const fn slot_count(&self) -> usize {
        self.slot_item.len()
    }

    /// Slot holding `item`, when it is resident.
    pub fn slot_of(&self, item: usize) -> EngineResult<Option<usize>> {
        let slot = *self.item_slot.get(item).ok_or_else(|| {
            EngineError::route(format!(
                "streaming item {item} is outside 0..{}",
                self.item_slot.len()
            ))
        })?;
        Ok((slot != STREAMING_ABSENT_SLOT).then_some(slot as usize))
    }

    /// Item holding `slot`, when it is occupied.
    pub fn item_of(&self, slot: usize) -> EngineResult<Option<usize>> {
        let item = *self.slot_item.get(slot).ok_or_else(|| {
            EngineError::route(format!(
                "streaming slot {slot} is outside 0..{}",
                self.slot_item.len()
            ))
        })?;
        Ok((item != STREAMING_ABSENT_ITEM).then_some(item as usize))
    }

    /// Complete item-to-slot table, with [`STREAMING_ABSENT_SLOT`] for absent items.
    pub fn table(&self) -> &[u32] {
        &self.item_slot
    }

    /// Cumulative resident requests.
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Cumulative admitted requests.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Generation of the most recent admitted round.
    ///
    /// The monotone round counter rule 1 reads as a replay timeline. It never
    /// restarts. [`Self::clear`] drops residency, not the timeline, because a
    /// generation that went backwards would let a stale release look as though
    /// it covered a fresh round.
    pub const fn round(&self) -> u64 {
        self.round
    }

    /// Generation of the last round that requested the item `slot` holds.
    ///
    /// Zero for a slot no round has ever requested. A replay fenced against
    /// generation `g` reads exactly the items round `g` requested, so this is
    /// the newest replay that could still be reading the slot.
    pub fn slot_reader_round(&self, slot: usize) -> EngineResult<u64> {
        self.pinned_round.get(slot).copied().ok_or_else(|| {
            EngineError::route(format!(
                "streaming slot {slot} is outside 0..{}",
                self.slot_item.len()
            ))
        })
    }

    /// Fixed host bookkeeping bytes.
    pub const fn host_allocation_bytes(&self) -> usize {
        self.item_slot.len() * size_of::<u32>()
            + self.slot_item.len() * size_of::<u32>()
            + self.older.len() * size_of::<u32>()
            + self.newer.len() * size_of::<u32>()
            + self.pinned_round.len() * size_of::<u64>()
            + self.plan.len() * size_of::<PlannedStep>()
    }

    /// Admits every item of one round, appending the assignments it needs.
    ///
    /// Items are processed in the given order. A resident item is a hit and is
    /// touched to most-recent; a missing item takes the lowest free slot, or
    /// else the least-recently-used slot no earlier item of this same round
    /// claimed. Requesting more distinct items than the pool has slots is a
    /// contract failure, not a silent eviction of this round's own work.
    ///
    /// Every request is validated before any state changes, so a refused round
    /// leaves residency, recency, and the counters exactly as it found them.
    pub fn admit_round(
        &mut self,
        items: &[u32],
        assignments: &mut Vec<StreamingSlotAssignment>,
    ) -> EngineResult<StreamingRound> {
        let round = self.plan_round(items, assignments)?;
        self.commit_planned_round();

        Ok(round)
    }

    /// Decides one round without committing any of it.
    ///
    /// Resolves every requested item, in request order, to a hit or to the slot
    /// a miss would take, appending one [`StreamingSlotAssignment`] per
    /// admission to `assignments` (which it clears first). Residency, recency,
    /// the reader generations, and the counters are all left exactly as they
    /// were, so a plan the caller never commits is invisible and a retry plans
    /// again from unchanged state.
    ///
    /// Rule 6 of the publication ordering law: the caller resolves every
    /// borrowed source this plan will read and reaches the transfer stream with
    /// all of its uploads before [`Self::commit_planned_round`] makes any of it
    /// residency. Planning twice without committing simply replaces the plan.
    pub fn plan_round(
        &mut self,
        items: &[u32],
        assignments: &mut Vec<StreamingSlotAssignment>,
    ) -> EngineResult<StreamingRound> {
        if self.round == u64::MAX {
            return Err(EngineError::layout(
                "streaming round generation is exhausted",
            ));
        }
        self.require_admissible(items)?;
        assignments.clear();
        self.plan_len = 0;
        self.planned = true;
        let mut round = StreamingRound::default();

        for &item in items {
            if let Some(slot) = self.planned_slot_of(item) {
                // A repeat inside one round hits a slot the plan already holds,
                // so it counts but adds no step.
                if !self.is_planned(slot) {
                    self.push_plan(PlannedStep {
                        item,
                        slot,
                        evicted: STREAMING_ABSENT_ITEM,
                        admitted: false,
                    });
                }
                round.hits += 1;
                continue;
            }

            let slot = self.plan_claim_slot()?;
            // Rule 1: the round may not overwrite this slot until every replay
            // that could still be reading it has been released, and the newest
            // such replay is the last round that requested the slot's item.
            round.reclaim_generation = round
                .reclaim_generation
                .max(self.pinned_round[slot as usize]);
            let evicted = self.slot_item[slot as usize];
            self.push_plan(PlannedStep {
                item,
                slot,
                evicted,
                admitted: true,
            });
            assignments.push(StreamingSlotAssignment {
                item,
                slot,
                evicted,
            });
            round.misses += 1;
        }
        let counters_fit = u64::try_from(round.hits)
            .ok()
            .and_then(|hits| self.hits.checked_add(hits))
            .is_some()
            && u64::try_from(round.misses)
                .ok()
                .and_then(|misses| self.misses.checked_add(misses))
                .is_some();
        if !counters_fit {
            assignments.clear();
            self.abandon_planned_round();
            return Err(EngineError::layout(
                "streaming cache request counters overflow",
            ));
        }
        self.plan_hits = round.hits;
        self.plan_misses = round.misses;

        Ok(round)
    }

    /// Applies the round [`Self::plan_round`] decided. Cannot fail.
    ///
    /// A replay of the plan's own steps rather than a second derivation, so the
    /// residency it commits is exactly the residency whose uploads the caller
    /// enqueued. Committing with no plan is a no-op.
    pub fn commit_planned_round(&mut self) {
        if !self.planned {
            return;
        }
        self.round += 1;

        for index in 0..self.plan_len {
            let step = self.plan[index];
            if step.admitted {
                if step.evicted != STREAMING_ABSENT_ITEM {
                    self.item_slot[step.evicted as usize] = STREAMING_ABSENT_SLOT;
                }
                self.slot_item[step.slot as usize] = step.item;
                self.item_slot[step.item as usize] = step.slot;
            }
            self.touch(step.slot);
            self.pinned_round[step.slot as usize] = self.round;
        }
        // The counters follow the round's requests, not its steps, so a repeat
        // inside one round still counts as the hit it was reported as.
        self.hits += self.plan_hits as u64;
        self.misses += self.plan_misses as u64;
        self.abandon_planned_round();
    }

    /// Whether a planned round is waiting to be committed.
    pub const fn has_planned_round(&self) -> bool {
        self.planned
    }

    /// Drops the planned round without committing any of it.
    pub fn abandon_planned_round(&mut self) {
        self.plan_len = 0;
        self.plan_hits = 0;
        self.plan_misses = 0;
        self.planned = false;
    }

    fn push_plan(&mut self, step: PlannedStep) {
        self.plan[self.plan_len] = step;
        self.plan_len += 1;
    }

    /// Slot `item` would read this round, or `None` when it would be a miss.
    ///
    /// Earlier steps may have admitted or evicted the item, so committed
    /// residency alone cannot answer this query.
    fn planned_slot_of(&self, item: u32) -> Option<u32> {
        // Admission wins over eviction whichever order they appear in: a plan
        // admits one item at most once, because the slot it lands in is then
        // planned and no later step of the same round may claim it back.
        if let Some(step) = self.plan[..self.plan_len]
            .iter()
            .find(|step| step.admitted && step.item == item)
        {
            return Some(step.slot);
        }
        if self.plan[..self.plan_len]
            .iter()
            .any(|step| step.admitted && step.evicted == item)
        {
            return None;
        }
        let committed = self.item_slot[item as usize];
        (committed != STREAMING_ABSENT_SLOT).then_some(committed)
    }

    /// Whether an earlier step of this plan already claimed or touched `slot`.
    ///
    /// The planning counterpart of [`Self::claim_slot`]'s reader-generation
    /// test: a round never evicts a slot it has itself already used, whether it
    /// used it by admitting into it or by hitting the item it holds.
    fn is_planned(&self, slot: u32) -> bool {
        self.plan[..self.plan_len]
            .iter()
            .any(|step| step.slot == slot)
    }

    /// The slot a miss would take, without committing anything.
    fn plan_claim_slot(&self) -> EngineResult<u32> {
        // The lowest free slot this plan has not already taken. Committing an
        // admission fills the slot it claims, so this reproduces exactly the
        // free-list order a committing walk would have seen.
        let free = (0..self.slot_item.len() as u32).find(|&slot| {
            self.slot_item[slot as usize] == STREAMING_ABSENT_ITEM && !self.is_planned(slot)
        });
        if let Some(slot) = free {
            return Ok(slot);
        }

        let mut candidate = self.least_recent;
        while candidate != STREAMING_ABSENT_SLOT {
            if !self.is_planned(candidate) {
                return Ok(candidate);
            }
            candidate = self.newer[candidate as usize];
        }

        // `require_admissible` already proved the round fits, so an exhausted
        // recency list means the list itself lost a slot.
        Err(EngineError::layout(format!(
            "streaming recency list holds fewer than the {} owned slots",
            self.slot_item.len()
        )))
    }

    /// Drops every residency record without touching device state.
    ///
    /// The round counter deliberately survives: it is rule 1's replay timeline,
    /// and restarting it would let a release recorded before the clear look as
    /// though it covered a round admitted after it.
    pub fn clear(&mut self) {
        self.item_slot.fill(STREAMING_ABSENT_SLOT);
        self.slot_item.fill(STREAMING_ABSENT_ITEM);
        self.older.fill(STREAMING_ABSENT_SLOT);
        self.newer.fill(STREAMING_ABSENT_SLOT);
        self.pinned_round.fill(0);
        self.least_recent = STREAMING_ABSENT_SLOT;
        self.most_recent = STREAMING_ABSENT_SLOT;
        self.hits = 0;
        self.misses = 0;
        self.abandon_planned_round();
    }

    /// Rejects an out-of-range item and a round wider than the slot budget
    /// before the round changes anything.
    fn require_admissible(&self, items: &[u32]) -> EngineResult<()> {
        let mut distinct = 0;
        for (position, &item) in items.iter().enumerate() {
            if item as usize >= self.item_slot.len() {
                return Err(EngineError::route(format!(
                    "streaming item {item} is outside 0..{}",
                    self.item_slot.len()
                )));
            }
            if !items[..position].contains(&item) {
                distinct += 1;
            }
        }
        if distinct > self.slot_item.len() {
            return Err(EngineError::route(format!(
                "one streaming round requested {distinct} distinct items for {} slots",
                self.slot_item.len()
            )));
        }

        Ok(())
    }

    fn touch(&mut self, slot: u32) {
        if self.most_recent == slot {
            return;
        }
        self.unlink(slot);
        self.older[slot as usize] = self.most_recent;
        self.newer[slot as usize] = STREAMING_ABSENT_SLOT;
        if self.most_recent != STREAMING_ABSENT_SLOT {
            self.newer[self.most_recent as usize] = slot;
        }
        self.most_recent = slot;
        if self.least_recent == STREAMING_ABSENT_SLOT {
            self.least_recent = slot;
        }
    }

    fn unlink(&mut self, slot: u32) {
        let older = self.older[slot as usize];
        let newer = self.newer[slot as usize];
        if older != STREAMING_ABSENT_SLOT {
            self.newer[older as usize] = newer;
        } else if self.least_recent == slot {
            self.least_recent = newer;
        }
        if newer != STREAMING_ABSENT_SLOT {
            self.older[newer as usize] = older;
        } else if self.most_recent == slot {
            self.most_recent = older;
        }
        self.older[slot as usize] = STREAMING_ABSENT_SLOT;
        self.newer[slot as usize] = STREAMING_ABSENT_SLOT;
    }
}

/// One streaming-resident item pool: pinned host bytes, device slots, and the
/// device-visible indirection table a consuming kernel reads.
pub struct StreamingWeightPool {
    // Field order is the drop order: the transfer stream drains first, so no
    // enqueued upload can outlive the slot arena, the pinned pool, or the
    // bounce ring it reads.
    transfer: TransferStream,
    bounce: Option<PinnedBounceRing>,
    slots: DeviceSlotPool,
    host: PinnedHostPool,
    mapped: Option<Box<dyn StreamingMappedPrimary>>,
    cache: StreamingSlotCache,
    assignments: Vec<StreamingSlotAssignment>,
    layout: StreamingWeightLayout,
    staged: Box<[bool]>,
    uploaded_bytes: u64,
    /// Why this pool refuses every further round, once one has failed past its
    /// first upload. See [`StreamingWeightPool::poisoned`].
    poisoned: Option<String>,
    context: Arc<CudaContext>,
}

impl StreamingWeightPool {
    /// Allocates the pinned item pool, the device slot cache, and the table.
    ///
    /// The fully-pinned posture. A mapped-primary layout is refused here rather
    /// than silently constructed without the mapping it borrows from.
    pub fn new(context: &Arc<CudaContext>, layout: StreamingWeightLayout) -> EngineResult<Self> {
        if !layout.primary_source.is_pinned() {
            return Err(EngineError::layout(
                "a mapped-primary streaming layout needs `new_with_mapped_primary` and its mapping",
            ));
        }
        Self::allocate(context, layout, None)
    }

    /// Allocates the same pool with the primary extents borrowed from `mapped`.
    ///
    /// The mapped posture: the pinned pool carries only the secondary extents
    /// and the bounce ring carries the primary ones. A fully-pinned layout is
    /// refused here, because it has no ring for the mapping to travel through.
    pub fn new_with_mapped_primary(
        context: &Arc<CudaContext>,
        layout: StreamingWeightLayout,
        mapped: Box<dyn StreamingMappedPrimary>,
    ) -> EngineResult<Self> {
        if layout.primary_source.is_pinned() {
            return Err(EngineError::layout(
                "a fully-pinned streaming layout takes no mapped primary source",
            ));
        }
        Self::allocate(context, layout, Some(mapped))
    }

    fn allocate(
        context: &Arc<CudaContext>,
        layout: StreamingWeightLayout,
        mapped: Option<Box<dyn StreamingMappedPrimary>>,
    ) -> EngineResult<Self> {
        let transfer = TransferStream::new(context)?;
        let bounce = if layout.primary_source.is_pinned() {
            None
        } else {
            Some(PinnedBounceRing::allocate(
                context,
                layout.bounce_slot_count,
                layout.bounce_slot_bytes,
            )?)
        };
        let slots = DeviceSlotPool::new(
            context,
            layout.item_count,
            layout.slot_count,
            layout.stride_bytes,
        )?;
        let host = PinnedHostPool::allocate(context, layout.host_pool_bytes)?;
        let cache = StreamingSlotCache::new(layout.item_count, layout.slot_count)?;
        layout.require_measured_allocations(
            slots.device_bytes(),
            slots.staging_bytes(),
            bounce.as_ref().map_or(0, PinnedBounceRing::byte_len),
        )?;

        Ok(Self {
            transfer,
            bounce,
            slots,
            host,
            mapped,
            cache,
            assignments: Vec::with_capacity(layout.slot_count),
            layout,
            staged: vec![false; layout.item_count].into_boxed_slice(),
            uploaded_bytes: 0,
            poisoned: None,
            context: Arc::clone(context),
        })
    }

    /// Checked byte plan of this pool.
    pub const fn layout(&self) -> &StreamingWeightLayout {
        &self.layout
    }

    /// Host LRU ownership of the slot cache.
    pub const fn cache(&self) -> &StreamingSlotCache {
        &self.cache
    }

    /// CUDA context shared by the pinned pool, the slot arena, and consumers.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Wall time the pinned item pool took to allocate and page-lock.
    pub const fn host_pin_duration(&self) -> Duration {
        self.host.pin_duration()
    }

    /// Cumulative host-to-device bytes this pool has enqueued.
    pub const fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// Why this pool refuses every further round, when it does.
    ///
    /// Set only by a failure past a round's first upload, where the device holds
    /// bytes no published table describes. Earlier failures leave the pool
    /// untouched and retryable.
    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    /// Writes one item's *pinned* extents into the pinned host pool.
    ///
    /// Admission-time only: the item's bytes are the represented source words,
    /// copied once and never decoded. `secondary` must be empty when the layout
    /// admits no second plane set.
    ///
    /// The pinned posture takes both extents. The mapped posture requires an
    /// empty `primary` and stores only `secondary`.
    ///
    /// The transfer stream is drained first, because these page-locked bytes
    /// are the source of every slot upload and an enqueued copy must never read
    /// a half-rewritten extent. At admission that stream is idle, so the drain
    /// costs nothing; it is what makes a late restage safe rather than a race.
    pub fn stage_item(
        &mut self,
        item: usize,
        primary: &[u8],
        secondary: &[u8],
    ) -> EngineResult<()> {
        self.require_live()?;
        let offset = self.layout.item_offset(item)?;
        if self.staged[item] {
            return Err(EngineError::layout(format!(
                "streaming item {item} was already staged"
            )));
        }
        let admitted_primary = if self.layout.primary_source.is_pinned() {
            self.layout.primary_extent_bytes
        } else {
            0
        };
        if primary.len() != admitted_primary
            || secondary.len() != self.layout.secondary_extent_bytes
        {
            return Err(EngineError::layout(format!(
                "streaming item {item} staged {}+{} bytes for an admitted {}+{} pinned extent under {:?} primaries",
                primary.len(),
                secondary.len(),
                admitted_primary,
                self.layout.secondary_extent_bytes,
                self.layout.primary_source
            )));
        }
        self.transfer.synchronize()?;
        self.host.write(offset, primary)?;
        self.host.write(offset + admitted_primary, secondary)?;
        self.staged[item] = true;

        Ok(())
    }

    /// Whether every item of the inventory has been staged.
    pub fn is_fully_staged(&self) -> bool {
        self.staged.iter().all(|&staged| staged)
    }

    /// Reads one item's staged host bytes back.
    ///
    /// The pool's own bytes, so in the mapped posture this is the secondary
    /// extent alone; the primary extent is the mapping's, not the pool's.
    pub fn staged_item(&self, item: usize) -> EngineResult<&[u8]> {
        let offset = self.layout.item_offset(item)?;
        Ok(self
            .host
            .slice(offset..offset + self.layout.host_extent_bytes())?)
    }

    /// Times the bounce ring made the host wait for a slot's upload to land.
    ///
    /// Zero in the fully-pinned posture, which has no ring. Diagnostic, and the
    /// direct evidence that rule 5's wraparound fence is on the live path.
    pub fn bounce_wraparound_waits(&self) -> u64 {
        self.bounce
            .as_ref()
            .map_or(0, PinnedBounceRing::wraparound_waits)
    }

    /// Wall time the pinned bounce ring took to allocate and page-lock.
    pub fn bounce_pin_duration(&self) -> Duration {
        self.bounce
            .as_ref()
            .map_or(Duration::ZERO, PinnedBounceRing::pin_duration)
    }

    /// Enqueues the round's uploads and table without stalling the host.
    ///
    /// The overlap route. The caller must call [`Self::fence_replay`] on its
    /// consumer stream before any replay that reads these slots.
    pub fn prefetch(&mut self, items: &[u32]) -> EngineResult<StreamingRound> {
        self.round(items, false)
    }

    /// Enqueues the round and stalls the host until every item is resident.
    ///
    /// The correctness route. On return the transfer stream has drained, so any
    /// stream in this context observes the round's slots and table.
    pub fn require(&mut self, items: &[u32]) -> EngineResult<StreamingRound> {
        self.round(items, true)
    }

    /// Enqueues `consumer`'s wait on the latest publication fence.
    ///
    /// Rule 4 of the publication ordering law. Cheap and idempotent: call it
    /// before every replay that reads slots or the indirection table.
    pub fn fence_replay(&self, consumer: &CudaStream) -> EngineResult<()> {
        Ok(self.transfer.wait_publication(consumer)?)
    }

    /// Records the reclaim fence on `consumer` after a replay that read slots.
    ///
    /// Rule 1 of the publication ordering law. Without it a later round may
    /// overwrite a slot the recorded replay is still reading. The fence is
    /// tagged with the current round generation, the newest it could have read.
    /// The consumer owes the pool the production order:
    /// one in-order stream, the release after the last replay that read the
    /// round and before the pool's next round.
    pub fn record_replay_release(&mut self, consumer: &CudaStream) -> EngineResult<()> {
        Ok(self.transfer.record_reclaim(consumer, self.cache.round())?)
    }

    /// Newest replay generation a recorded reclaim fence releases.
    pub fn released_generation(&self) -> Option<u64> {
        self.transfer.released_generation()
    }

    /// Whether the latest published round's transfer has completed.
    ///
    /// Diagnostic only; consumers still owe [`Self::fence_replay`].
    pub fn publication_completed(&self) -> EngineResult<bool> {
        Ok(self.transfer.publication_completed()?)
    }

    /// Drains the transfer stream.
    pub fn synchronize(&self) -> EngineResult<()> {
        Ok(self.transfer.synchronize()?)
    }

    /// Slot holding `item`, when it is resident.
    pub fn slot_of(&self, item: usize) -> EngineResult<Option<usize>> {
        self.cache.slot_of(item)
    }

    /// Stable device address of one slot extent.
    pub fn slot_address(&self, slot: usize) -> EngineResult<u64> {
        Ok(self.slots.slot_address(slot)?)
    }

    /// Stable device address of the item-to-slot indirection table.
    pub fn table_address(&self) -> EngineResult<u64> {
        Ok(self.slots.table_address()?)
    }

    /// Evicts every item and republishes an all-absent indirection table.
    ///
    /// Rule 1 still applies: the newest slot reader must have a recorded release
    /// before the all-absent table is published.
    pub fn reset(&mut self) -> EngineResult<()> {
        self.require_live()?;
        let reclaim_generation = self.cache.pinned_round.iter().copied().max().unwrap_or(0);
        self.transfer.wait_reclaim(reclaim_generation)?;
        self.cache.clear();
        if let Err(error) = self
            .slots
            .publish_table(&self.transfer, self.cache.table())
            .and_then(|()| self.transfer.record_publication())
        {
            return Err(self.poison("publishing a reset table", error.into()));
        }
        if let Err(error) = self.transfer.stall_for_publication() {
            return Err(self.poison("stalling for a reset table", error.into()));
        }

        Ok(())
    }

    /// Reads one slot extent back to the host.
    pub fn read_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<Vec<u8>> {
        Ok(self.slots.read_slot(stream, slot)?)
    }

    /// Reads the complete device-side indirection table back to the host.
    pub fn read_table(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        Ok(self.slots.read_table(stream)?)
    }

    /// Stable device and host backing addresses for post-warmup checks.
    pub fn allocation_addresses(&self) -> [usize; 2] {
        [self.slots.base_address() as usize, self.host.base_address()]
    }

    fn round(&mut self, items: &[u32], stall: bool) -> EngineResult<StreamingRound> {
        self.require_live()?;
        // Refuse an unstaged item before the cache plans anything, so a rejected
        // round can never leave host residency claiming a slot whose device
        // bytes were never uploaded.
        for &item in items {
            let index = item as usize;
            let staged = *self.staged.get(index).ok_or_else(|| {
                EngineError::route(format!(
                    "streaming item {item} is outside 0..{}",
                    self.layout.item_count
                ))
            })?;
            if !staged {
                return Err(EngineError::layout(format!(
                    "streaming item {item} was requested before it was staged"
                )));
            }
        }

        // Rule 6, first half: decide the round without committing any of it, and
        // resolve every borrowed source it will read before the first byte is
        // enqueued. A source that refuses here costs nothing: residency, the
        // table, and the counters are all still exactly as this call found them,
        // so a retry plans again and genuinely re-fetches.
        let mut round = self.cache.plan_round(items, &mut self.assignments)?;
        round.uploaded_bytes = match self.assignments.len().checked_mul(self.layout.stride_bytes) {
            Some(bytes) => bytes,
            None => {
                self.cache.abandon_planned_round();
                return Err(EngineError::layout("streamed byte counter overflows"));
            }
        };
        let next_uploaded_bytes = match u64::try_from(round.uploaded_bytes)
            .ok()
            .and_then(|bytes| self.uploaded_bytes.checked_add(bytes))
        {
            Some(bytes) => bytes,
            None => {
                self.cache.abandon_planned_round();
                return Err(EngineError::layout("streamed byte counter overflows"));
            }
        };
        if let Err(error) = self.require_planned_sources() {
            self.cache.abandon_planned_round();
            return Err(error);
        }

        if self.assignments.is_empty() {
            // A pure-hit round moves recency and nothing else; there is no
            // upload to fail between the plan and the commit.
            self.cache.commit_planned_round();
        } else {
            // Rule 1: nothing is overwritten while a replay that could read it
            // is unreleased. The round waits on the cheapest fence that covers
            // the newest reader among the slots it evicts, so a round evicting
            // only long-idle slots does not wait for the replay in flight.
            if let Err(error) = self.transfer.wait_reclaim(round.reclaim_generation) {
                self.cache.abandon_planned_round();
                return Err(error.into());
            }
            // Rule 6, second half: past this point the device holds bytes no
            // published table describes, so a failure can no longer be undone
            // by forgetting the plan, so it poisons instead.
            self.enqueue_planned_round()?;
            // Every upload reached the transfer stream, so the round is real:
            // commit it, and only now. This cannot fail.
            self.cache.commit_planned_round();
            // Rules 2 and 3: the table publication rides the same stream, last,
            // and one fence covers the whole round.
            self.publish_committed_round()?;
            self.uploaded_bytes = next_uploaded_bytes;
        }

        if stall {
            // The stall a miss costs. It also covers an earlier prefetch of a
            // requested item that is a hit here but has not landed yet.
            self.transfer.stall_for_publication()?;
            round.stalled = true;
        }

        Ok(round)
    }

    /// Resolves and length-checks every borrowed extent the plan will upload.
    ///
    /// The fallible half of the mapped posture, lifted out of the enqueue loop
    /// so a source that refuses cannot leave a half-uploaded round behind. In
    /// the pinned posture there is nothing to resolve.
    fn require_planned_sources(&self) -> EngineResult<()> {
        if self.layout.primary_source.is_pinned() {
            return Ok(());
        }
        let mapped = self.mapped.as_ref().ok_or_else(|| {
            EngineError::layout("a mapped-primary streaming pool lost its mapping")
        })?;
        if self.bounce.is_none() {
            return Err(EngineError::layout(
                "a mapped-primary streaming pool lost its bounce ring",
            ));
        }
        for assignment in &self.assignments {
            let item = assignment.item as usize;
            let primary = mapped.primary_extent(item)?;
            if primary.len() != self.layout.primary_extent_bytes {
                return Err(EngineError::layout(format!(
                    "streaming item {item} borrowed {} bytes for an admitted {}-byte primary extent",
                    primary.len(),
                    self.layout.primary_extent_bytes
                )));
            }
        }

        Ok(())
    }

    /// Enqueues every planned assignment's uploads, poisoning on any failure.
    fn enqueue_planned_round(&mut self) -> EngineResult<()> {
        for index in 0..self.assignments.len() {
            let assignment = self.assignments[index];
            if let Err(error) = self.enqueue_assignment(assignment) {
                return Err(self.poison("enqueueing a round's slot uploads", error));
            }
        }

        Ok(())
    }

    /// Publishes the committed table and fences it, poisoning on any failure.
    fn publish_committed_round(&mut self) -> EngineResult<()> {
        if let Err(error) = self
            .slots
            .publish_table(&self.transfer, self.cache.table())
            .and_then(|()| self.transfer.record_publication())
        {
            return Err(self.poison("publishing a round's indirection table", error.into()));
        }

        Ok(())
    }

    /// Refuses every later round after an unrecoverable mid-round failure.
    ///
    /// Reached only past a round's first upload, where the device already holds
    /// bytes no published table describes. No later round could prove what those
    /// slots contain, and inventing residency for them is exactly what the
    /// cache-state exactness law forbids, so the pool stops instead. The owner
    /// rebuilds it; the original failure is returned unchanged.
    fn poison(&mut self, operation: &str, error: EngineError) -> EngineError {
        self.cache.abandon_planned_round();
        if self.poisoned.is_none() {
            self.poisoned = Some(format!("{operation} failed mid-round: {error}"));
        }
        error
    }

    fn require_live(&self) -> EngineResult<()> {
        match &self.poisoned {
            Some(reason) => Err(EngineError::layout(format!(
                "streaming pool refuses every round after {reason}"
            ))),
            None => Ok(()),
        }
    }

    /// Enqueues the uploads that fill one admitted slot.
    ///
    /// One copy in the fully-pinned posture, two in the mapped one: the
    /// borrowed primary extent through the bounce ring, then the pooled
    /// secondary extent behind it. The device slot is byte-identical either way:
    /// the two copies write the same contiguous extent the single copy would
    /// have written, at the same stable address.
    fn enqueue_assignment(&mut self, assignment: StreamingSlotAssignment) -> EngineResult<()> {
        let item = assignment.item as usize;
        let slot = assignment.slot as usize;
        let offset = self.layout.item_offset(item)?;
        let Self {
            transfer,
            bounce,
            slots,
            host,
            mapped,
            layout,
            ..
        } = self;

        if layout.primary_source.is_pinned() {
            // SAFETY: the pinned pool outlives this transfer stream, which
            // drains on drop, and staging never rewrites an item's bytes
            // after admission.
            unsafe { slots.enqueue_slot_upload(transfer, slot, host, offset)? };

            return Ok(());
        }

        let mapped = mapped.as_ref().ok_or_else(|| {
            EngineError::layout("a mapped-primary streaming pool lost its mapping")
        })?;
        let bounce = bounce.as_mut().ok_or_else(|| {
            EngineError::layout("a mapped-primary streaming pool lost its bounce ring")
        })?;
        let primary = mapped.primary_extent(item)?;
        if primary.len() != layout.primary_extent_bytes {
            return Err(EngineError::layout(format!(
                "streaming item {item} borrowed {} bytes for an admitted {}-byte primary extent",
                primary.len(),
                layout.primary_extent_bytes
            )));
        }
        let destination = slots.slot_address(slot)?;
        // SAFETY: `destination` names this live arena's checked slot extent, and
        // the ring's own fence proves the bounce slot the copy reads is not
        // rewritten before that copy lands.
        unsafe { bounce.upload(transfer, destination, primary)? };
        if layout.secondary_extent_bytes != 0 {
            // SAFETY: the pinned pool outlives this transfer stream, which
            // drains on drop, and staging never rewrites an item's bytes after
            // admission.
            unsafe {
                slots.enqueue_slot_extent_upload(
                    transfer,
                    slot,
                    layout.primary_extent_bytes,
                    host,
                    offset,
                    layout.secondary_extent_bytes,
                )?;
            }
        }

        Ok(())
    }
}

impl crate::LayerMemoryLayout for StreamingWeightPool {
    fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }
}

impl StreamingResidencyAccounting for StreamingWeightPool {
    fn device_resident_bytes(&self) -> usize {
        self.layout.device_resident_bytes()
    }

    fn host_pinned_bytes(&self) -> usize {
        self.layout.host_pinned_bytes()
    }

    fn host_mapped_bytes(&self) -> usize {
        self.layout.host_mapped_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        STREAMING_ABSENT_ITEM, STREAMING_ABSENT_SLOT, StreamingPrimarySource,
        StreamingSlotAssignment, StreamingSlotCache, StreamingWeightLayout,
    };
    use crate::{EngineErrorCode, LayerMemoryLayout, StreamingResidencyAccounting};

    fn assignment(item: u32, slot: u32, evicted: u32) -> StreamingSlotAssignment {
        StreamingSlotAssignment {
            item,
            slot,
            evicted,
        }
    }

    const QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS: usize = 8;

    #[test]
    fn qwen38_flash_next_layout_pins_stride_padding_and_every_residency_class() {
        let layout = StreamingWeightLayout::build(24_576, 2_457_600, Some(307_200), 6_144).unwrap();

        assert_eq!(layout.extent_bytes(), 2_764_800);
        assert_eq!(layout.stride_bytes(), 2_764_800);
        assert_eq!(layout.extent_padding_bytes(), 0);
        assert_eq!(layout.slot_region_bytes(), 16_986_931_200);
        assert_eq!(layout.table_bytes(), 98_304);
        assert_eq!(layout.table_staging_bytes(), 393_216);
        assert_eq!(layout.host_pool_bytes(), 67_947_724_800);
        assert_eq!(layout.arena_bytes(), 16_987_029_504);
        assert_eq!(layout.resident_weight_bytes(), 16_986_931_200);
        assert_eq!(layout.cache_bytes(), 0);
        assert_eq!(layout.workspace_bytes(), 98_304);
        assert_eq!(layout.device_resident_bytes(), 16_987_029_504);
        assert_eq!(layout.host_pinned_bytes(), 67_948_118_016);
        assert_eq!(layout.host_mapped_bytes(), 0);
        assert!((layout.resident_fraction() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn the_mapped_posture_moves_the_primary_extent_between_the_host_classes() {
        // The exact product inventory must have one device plan under both host
        // postures.
        let pinned = StreamingWeightLayout::build(24_576, 2_457_600, Some(307_200), 6_144).unwrap();
        let mapped = StreamingWeightLayout::build_mapped_primary(
            24_576,
            2_457_600,
            Some(307_200),
            6_144,
            QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS,
        )
        .unwrap();

        assert_eq!(pinned.primary_source(), StreamingPrimarySource::Pinned);
        assert_eq!(mapped.primary_source(), StreamingPrimarySource::Mapped);
        assert_eq!(mapped.stride_bytes(), pinned.stride_bytes());
        assert_eq!(mapped.slot_region_bytes(), pinned.slot_region_bytes());
        assert_eq!(mapped.table_bytes(), pinned.table_bytes());
        assert_eq!(
            mapped.device_resident_bytes(),
            pinned.device_resident_bytes()
        );

        // Pinned: the whole 2,764,800 B stride is pooled and nothing is mapped.
        assert_eq!(pinned.host_stride_bytes(), 2_764_800);
        assert_eq!(pinned.host_pool_bytes(), 67_947_724_800);
        assert_eq!(pinned.bounce_slot_count(), 0);
        assert_eq!(pinned.bounce_ring_bytes(), 0);
        assert_eq!(pinned.host_pinned_bytes(), 67_948_118_016);
        assert_eq!(pinned.host_mapped_bytes(), 0);

        // Mapped: only the 307,200 B swizzled scale extent is pooled, the
        // 2,457,600 B packed codes are borrowed, and eight bounce slots carry
        // them. 7,549,747,200 + 393,216 + 19,660,800 = 7,569,801,216.
        assert_eq!(mapped.host_stride_bytes(), 307_200);
        assert_eq!(mapped.host_extent_bytes(), 307_200);
        assert_eq!(mapped.host_pool_bytes(), 7_549_747_200);
        assert_eq!(mapped.bounce_slot_count(), 8);
        assert_eq!(mapped.bounce_slot_bytes(), 2_457_600);
        assert_eq!(mapped.bounce_ring_bytes(), 19_660_800);
        assert_eq!(mapped.table_staging_bytes(), 393_216);
        assert_eq!(mapped.host_pinned_bytes(), 7_569_801_216);
        assert_eq!(mapped.host_mapped_bytes(), 60_397_977_600);

        // The mapped posture trades 60,377,922,800 pinned bytes for the same
        // codes held file-backed, which is the whole reason it exists: the
        // pinned pool alone is 63.28 GiB against 59.2 GiB of usable RAM.
        assert_eq!(
            pinned.host_pinned_bytes() - mapped.host_pinned_bytes(),
            60_378_316_800
        );
        assert_eq!(
            mapped.host_mapped_bytes() + mapped.host_pool_bytes(),
            pinned.host_pool_bytes()
        );
        // A mapped posture with no bounce slot has no path for the primary.
        assert_eq!(
            StreamingWeightLayout::build_mapped_primary(8, 256, Some(256), 2, 0)
                .unwrap_err()
                .code(),
            Some(EngineErrorCode::Layout)
        );
    }

    #[test]
    fn a_mapped_layout_with_no_secondary_extent_pins_only_its_rings() {
        let layout = StreamingWeightLayout::build_mapped_primary(16, 1_024, None, 4, 2).unwrap();

        assert_eq!(layout.stride_bytes(), 1_024);
        assert_eq!(layout.host_stride_bytes(), 0);
        assert_eq!(layout.host_pool_bytes(), 0);
        assert_eq!(layout.bounce_ring_bytes(), 2_048);
        assert_eq!(layout.table_staging_bytes(), 4 * 16 * 4);
        assert_eq!(layout.host_pinned_bytes(), 2_048 + 256);
        assert_eq!(layout.host_mapped_bytes(), 16 * 1_024);
    }

    #[test]
    fn layout_rounds_an_unaligned_extent_up_to_one_stride() {
        let layout = StreamingWeightLayout::build(8, 300, None, 2).unwrap();

        assert_eq!(layout.extent_bytes(), 300);
        assert_eq!(layout.stride_bytes(), 512);
        assert_eq!(layout.extent_padding_bytes(), 212);
        assert_eq!(layout.host_pool_bytes(), 4_096);
        assert_eq!(layout.slot_region_bytes(), 1_024);
        assert_eq!(layout.table_bytes(), 32);
    }

    #[test]
    fn layout_rejects_an_empty_inventory_extent_or_slot_budget() {
        for rejected in [(0, 256, 1), (4, 0, 1), (4, 256, 0), (4, 256, 5)] {
            let error = StreamingWeightLayout::build(rejected.0, rejected.1, None, rejected.2)
                .err()
                .unwrap();
            assert_eq!(error.code(), Some(EngineErrorCode::Layout));
        }
    }

    #[test]
    fn mapped_layout_rejects_total_pinned_byte_overflow() {
        let secondary = (usize::MAX / 2) / 256 * 256;
        let error =
            StreamingWeightLayout::build_mapped_primary(2, 256, Some(secondary), 1, 2).unwrap_err();

        assert_eq!(error.code(), Some(EngineErrorCode::Layout));
    }

    #[test]
    fn plain_lru_fills_free_slots_in_order_then_evicts_least_recently_used() {
        let mut cache = StreamingSlotCache::new(6, 3).unwrap();
        let mut admitted = Vec::new();

        let round = cache.admit_round(&[0, 1, 2], &mut admitted).unwrap();
        assert_eq!((round.hits(), round.misses()), (0, 3));
        assert_eq!(
            admitted,
            [
                assignment(0, 0, STREAMING_ABSENT_ITEM),
                assignment(1, 1, STREAMING_ABSENT_ITEM),
                assignment(2, 2, STREAMING_ABSENT_ITEM),
            ]
        );

        // Touch 0 so 1 becomes the least-recently-used entry.
        admitted.clear();
        let round = cache.admit_round(&[0], &mut admitted).unwrap();
        assert_eq!((round.hits(), round.misses()), (1, 0));
        assert!(admitted.is_empty());

        admitted.clear();
        cache.admit_round(&[3], &mut admitted).unwrap();
        assert_eq!(admitted, [assignment(3, 1, 1)]);
        assert_eq!(cache.slot_of(1).unwrap(), None);
        assert_eq!(cache.slot_of(3).unwrap(), Some(1));

        // 2 is now least-recent, then 0, then 3.
        admitted.clear();
        cache.admit_round(&[4, 5], &mut admitted).unwrap();
        assert_eq!(admitted, [assignment(4, 2, 2), assignment(5, 0, 0)]);
        assert_eq!(
            cache.table(),
            [
                STREAMING_ABSENT_SLOT,
                STREAMING_ABSENT_SLOT,
                STREAMING_ABSENT_SLOT,
                1,
                2,
                0
            ]
        );
        assert_eq!((cache.hits(), cache.misses()), (1, 6));
    }

    #[test]
    fn a_round_never_evicts_an_item_it_already_admitted() {
        let mut cache = StreamingSlotCache::new(8, 3).unwrap();
        let mut admitted = Vec::new();
        cache.admit_round(&[0, 1, 2], &mut admitted).unwrap();

        admitted.clear();
        cache.admit_round(&[3, 4, 5], &mut admitted).unwrap();
        assert_eq!(
            admitted,
            [
                assignment(3, 0, 0),
                assignment(4, 1, 1),
                assignment(5, 2, 2)
            ]
        );

        // A round wider than the budget is refused before it changes anything.
        let table = cache.table().to_vec();
        let counters = (cache.hits(), cache.misses());
        admitted.clear();
        let error = cache.admit_round(&[0, 1, 2, 3], &mut admitted).unwrap_err();
        assert_eq!(error.code(), Some(EngineErrorCode::Route));
        assert!(error.to_string().contains("4 distinct items for 3 slots"));
        assert!(admitted.is_empty());
        assert_eq!(cache.table(), table);
        assert_eq!((cache.hits(), cache.misses()), counters);
    }

    #[test]
    fn a_repeated_item_inside_one_round_is_a_hit_after_its_admission() {
        let mut cache = StreamingSlotCache::new(4, 2).unwrap();
        let mut admitted = Vec::new();

        let round = cache.admit_round(&[3, 3, 3], &mut admitted).unwrap();

        assert_eq!((round.hits(), round.misses()), (2, 1));
        assert_eq!(admitted, [assignment(3, 0, STREAMING_ABSENT_ITEM)]);
    }

    #[test]
    fn a_round_reclaims_only_the_newest_reader_of_the_slots_it_overwrites() {
        let mut cache = StreamingSlotCache::new(8, 4).unwrap();
        let mut admitted = Vec::new();

        // Generations 1..4 each take one free slot, so nothing is reclaimed:
        // a slot no round has ever requested has reader generation zero.
        for (generation, item) in [(1u64, 0u32), (2, 1), (3, 2), (4, 3)] {
            admitted.clear();
            let round = cache.admit_round(&[item], &mut admitted).unwrap();
            assert_eq!(cache.round(), generation);
            assert_eq!(round.reclaim_generation(), 0, "generation {generation}");
        }
        assert_eq!(
            (0..4)
                .map(|slot| cache.slot_reader_round(slot).unwrap())
                .collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );

        // Generation 5 evicts the least-recently-used slot, whose item was last
        // requested at generation 1, so it must reclaim exactly generation 1,
        // not the replay of generation 4 that is still in flight.
        admitted.clear();
        let round = cache.admit_round(&[4], &mut admitted).unwrap();
        assert_eq!(round.reclaim_generation(), 1);
        assert_eq!(cache.slot_reader_round(0).unwrap(), 5);

        // A round is reclaimed at the *newest* reader among the slots it takes.
        admitted.clear();
        let round = cache.admit_round(&[5, 6], &mut admitted).unwrap();
        assert_eq!(round.reclaim_generation(), 3);

        // A pure-hit round overwrites nothing and reclaims nothing.
        admitted.clear();
        let round = cache.admit_round(&[4, 5], &mut admitted).unwrap();
        assert_eq!((round.hits(), round.misses()), (2, 0));
        assert_eq!(round.reclaim_generation(), 0);
        assert!(admitted.is_empty());

        // A refused round advances neither the timeline nor the reclaim record.
        let generation = cache.round();
        admitted.clear();
        assert!(cache.admit_round(&[0, 1, 2, 3, 4], &mut admitted).is_err());
        assert_eq!(cache.round(), generation);
        assert_eq!(
            cache.slot_reader_round(4).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
    }

    #[test]
    fn a_planned_round_that_is_never_committed_changes_nothing() {
        let mut cache = StreamingSlotCache::new(8, 3).unwrap();
        let mut admitted = Vec::new();
        cache.admit_round(&[0, 1, 2], &mut admitted).unwrap();

        let table = cache.table().to_vec();
        let counters = (cache.hits(), cache.misses());
        let generation = cache.round();
        let readers = (0..3)
            .map(|slot| cache.slot_reader_round(slot).unwrap())
            .collect::<Vec<_>>();

        // Planning includes the eviction while observable state stays unchanged.
        admitted.clear();
        let planned = cache.plan_round(&[3, 4], &mut admitted).unwrap();
        assert!(cache.has_planned_round());
        assert_eq!((planned.hits(), planned.misses()), (0, 2));
        assert_eq!(planned.reclaim_generation(), 1);
        assert_eq!(
            admitted,
            [assignment(3, 0, 0), assignment(4, 1, 1)],
            "the plan must name the slots the uploads will fill"
        );
        assert_eq!(cache.table(), table);
        assert_eq!((cache.hits(), cache.misses()), counters);
        assert_eq!(cache.round(), generation);
        assert_eq!(cache.slot_of(3).unwrap(), None);
        assert_eq!(cache.slot_of(0).unwrap(), Some(0));

        // Abandoning it leaves the same state a second time: a round whose
        // uploads never reached the transfer stream never happened.
        cache.abandon_planned_round();
        assert!(!cache.has_planned_round());
        assert_eq!(cache.table(), table);
        assert_eq!((cache.hits(), cache.misses()), counters);
        assert_eq!(cache.round(), generation);
        assert_eq!(
            (0..3)
                .map(|slot| cache.slot_reader_round(slot).unwrap())
                .collect::<Vec<_>>(),
            readers
        );
        // Committing with nothing planned is a no-op, not a phantom round.
        cache.commit_planned_round();
        assert_eq!(cache.round(), generation);
        assert_eq!(cache.table(), table);

        // Re-planning the same round and committing it reaches exactly the
        // residency the abandoned plan described.
        admitted.clear();
        let retried = cache.plan_round(&[3, 4], &mut admitted).unwrap();
        assert_eq!(retried, planned);
        cache.commit_planned_round();
        assert!(!cache.has_planned_round());
        assert_eq!(cache.round(), generation + 1);
        assert_eq!(cache.slot_of(3).unwrap(), Some(0));
        assert_eq!(cache.slot_of(4).unwrap(), Some(1));
        assert_eq!(cache.slot_of(0).unwrap(), None);
        assert_eq!((cache.hits(), cache.misses()), (counters.0, counters.1 + 2));
    }

    #[test]
    fn planning_twice_without_committing_replaces_the_plan() {
        let mut cache = StreamingSlotCache::new(8, 3).unwrap();
        let mut admitted = Vec::new();
        cache.admit_round(&[0, 1, 2], &mut admitted).unwrap();
        let table = cache.table().to_vec();

        admitted.clear();
        cache.plan_round(&[5, 6, 7], &mut admitted).unwrap();
        admitted.clear();
        let second = cache.plan_round(&[0, 4], &mut admitted).unwrap();

        // The second plan is decided against the *committed* state, not against
        // the first plan's, so item 0 is still the hit it always was.
        assert_eq!((second.hits(), second.misses()), (1, 1));
        assert_eq!(admitted, [assignment(4, 1, 1)]);
        assert_eq!(cache.table(), table);

        cache.commit_planned_round();
        assert_eq!(cache.slot_of(0).unwrap(), Some(0));
        assert_eq!(cache.slot_of(4).unwrap(), Some(1));
        assert_eq!(cache.slot_of(5).unwrap(), None);
    }

    #[test]
    fn the_round_generation_never_restarts_across_a_clear() {
        let mut cache = StreamingSlotCache::new(6, 2).unwrap();
        let mut admitted = Vec::new();
        for item in 0..4u32 {
            admitted.clear();
            cache.admit_round(&[item], &mut admitted).unwrap();
        }
        assert_eq!(cache.round(), 4);

        cache.clear();

        // Residency is gone but the replay timeline is not: a release recorded
        // before the clear must never look as though it covered a later round.
        assert_eq!(cache.round(), 4);
        assert!((0..2).all(|slot| cache.slot_reader_round(slot).unwrap() == 0));
        admitted.clear();
        let round = cache.admit_round(&[5], &mut admitted).unwrap();
        assert_eq!(cache.round(), 5);
        assert_eq!(round.reclaim_generation(), 0);
    }

    #[test]
    fn exhausted_generations_and_counters_are_refused_before_planning() {
        let mut cache = StreamingSlotCache::new(4, 2).unwrap();
        let mut admitted = Vec::new();

        cache.round = u64::MAX;
        assert!(cache.plan_round(&[0], &mut admitted).is_err());
        assert!(!cache.has_planned_round());
        assert!(admitted.is_empty());

        cache.round = 0;
        cache.hits = u64::MAX;
        cache.admit_round(&[0], &mut admitted).unwrap();
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), u64::MAX);
        assert!(cache.plan_round(&[0], &mut admitted).is_err());
        assert!(!cache.has_planned_round());
        assert!(admitted.is_empty());
    }

    #[test]
    fn clearing_restores_an_empty_cache_without_reallocating() {
        let mut cache = StreamingSlotCache::new(6, 3).unwrap();
        let bytes = cache.host_allocation_bytes();
        let mut admitted = Vec::new();
        cache.admit_round(&[0, 1, 2], &mut admitted).unwrap();
        admitted.clear();
        cache.admit_round(&[3], &mut admitted).unwrap();

        cache.clear();

        assert_eq!(cache.host_allocation_bytes(), bytes);
        assert_eq!(cache.table(), [STREAMING_ABSENT_SLOT; 6]);
        assert!((0..3).all(|slot| cache.item_of(slot).unwrap().is_none()));
        admitted.clear();
        cache.admit_round(&[5], &mut admitted).unwrap();
        assert_eq!(admitted, [assignment(5, 0, STREAMING_ABSENT_ITEM)]);
    }

    #[test]
    fn cache_rejects_an_out_of_range_item_and_an_oversized_slot_budget() {
        assert_eq!(
            StreamingSlotCache::new(2, 3).unwrap_err().code(),
            Some(EngineErrorCode::Layout)
        );
        let mut cache = StreamingSlotCache::new(4, 2).unwrap();
        let mut admitted = Vec::new();
        assert_eq!(
            cache.admit_round(&[4], &mut admitted).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
        assert_eq!(
            cache.slot_of(9).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
        assert_eq!(
            cache.item_of(9).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
    }
}

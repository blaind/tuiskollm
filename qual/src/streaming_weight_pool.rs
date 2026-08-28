//! Device qualification for the streaming-resident weight pool.
//!
//! The subsystem adds no kernel route. Its contract is cache-state exactness
//! and publication ordering. Device readback must match staged bytes exactly.
//!
//! Single-round suites cover upload, replay, release, eviction, and retry.
//! Window suites cover Qwen3.8-Flash-Next's 48 rounds per token and prove
//! overlap as an ordering fact, never as a timing baseline.

#[cfg(test)]
mod tests {
    use crate::device_benchmark;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tuisko_engine::{
        EngineError, EngineErrorCode, EngineResult, STREAMING_ABSENT_ITEM, STREAMING_ABSENT_SLOT,
        StreamingMappedPrimary, StreamingResidencyAccounting, StreamingSlotCache,
        StreamingWeightLayout, StreamingWeightPool,
    };
    use tuisko_gpu::{
        ArenaLayout, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, device_memory_info,
    };

    /// Small enough to keep the pinned pool far below the shared-RAM budget, big
    /// enough that a slot spans several 256-byte alignment units.
    const ITEM_COUNT: usize = 12;
    const SLOT_COUNT: usize = 4;
    const PRIMARY_BYTES: usize = 3_072;
    const SECONDARY_BYTES: usize = 1_024;
    const STRIDE_BYTES: usize = 4_096;

    /// Streaming rounds one Flash-Next decode token issues: one `require` per
    /// MoE layer. The shape the single-round qualification could not see.
    const WINDOW_ROUNDS: usize = 48;

    /// Distinct experts one Flash-Next round requests: the router's top-10.
    const WINDOW_REQUESTS: usize = 10;

    /// Window-pool inventory. Slots are sized so the 48-round window evicts only
    /// slots the warm-up phase read and released: 49 warm rounds fill 490 of
    /// 500 slots, the window opener takes the 10 free ones, and the window's 480
    /// admissions consume exactly the warm-up's first 48 generations.
    const WINDOW_ITEMS: usize = 1_024;
    const WINDOW_SLOTS: usize = 500;
    const WINDOW_WARM_ROUNDS: usize = 49;

    /// Elements one captured copy moves, and how many copies make the window's
    /// replay long enough to still be running when the window's uploads land.
    /// This sizes an ordering observation, not a measurement.
    const WINDOW_COPY_ELEMENTS: usize = 4 * 1_024 * 1_024;
    const WINDOW_COPIES_PER_GRAPH: usize = 64;
    const WINDOW_GRAPH_LAUNCHES: usize = 32;

    /// Bounce slots the mapped-posture suite admits, chosen to wrap the ring
    /// several times inside a single round rather than to model production.
    const BOUNCE_SLOTS: usize = 2;

    type Suite = Result<(), Box<dyn std::error::Error>>;

    fn qualification_layout() -> StreamingWeightLayout {
        StreamingWeightLayout::build(ITEM_COUNT, PRIMARY_BYTES, Some(SECONDARY_BYTES), SLOT_COUNT)
            .unwrap()
    }

    /// One item's sentinel pattern: distinct per item, per plane, and per byte
    /// position, so a stale slot, a swapped plane, and a short copy all differ.
    fn item_bytes(item: usize) -> (Vec<u8>, Vec<u8>) {
        let primary = (0..PRIMARY_BYTES)
            .map(|index| (index as u8) ^ (0xa5u8.wrapping_add(item as u8 * 17)))
            .collect();
        let secondary = (0..SECONDARY_BYTES)
            .map(|index| (index as u8).wrapping_mul(3) ^ (0x5au8.wrapping_sub(item as u8 * 11)))
            .collect();
        (primary, secondary)
    }

    fn staged_pool(
        context: &Arc<CudaContext>,
    ) -> Result<StreamingWeightPool, Box<dyn std::error::Error>> {
        let mut pool = StreamingWeightPool::new(context, qualification_layout())?;
        for item in 0..ITEM_COUNT {
            let (primary, secondary) = item_bytes(item);
            pool.stage_item(item, &primary, &secondary)?;
        }
        assert!(pool.is_fully_staged());
        Ok(pool)
    }

    fn expected_slot(item: usize) -> Vec<u8> {
        let (primary, secondary) = item_bytes(item);
        let mut slot = Vec::with_capacity(STRIDE_BYTES);
        slot.extend_from_slice(&primary);
        slot.extend_from_slice(&secondary);
        slot.resize(STRIDE_BYTES, 0);
        slot
    }

    /// The window pool's sentinel pattern, distinct across all 1,024 items
    /// rather than the small pool's per-byte-of-`item` pattern: with 500 slots
    /// live, two items 256 apart must not be able to alias each other.
    fn window_item_bytes(item: usize) -> (Vec<u8>, Vec<u8>) {
        let key = (item as u32).wrapping_mul(2_654_435_761);
        let primary = (0..PRIMARY_BYTES)
            .map(|index| (index as u8) ^ key.to_le_bytes()[index % 4])
            .collect();
        let secondary = (0..SECONDARY_BYTES)
            .map(|index| (index as u8).wrapping_mul(3) ^ key.to_be_bytes()[index % 4])
            .collect();
        (primary, secondary)
    }

    fn window_expected_slot(item: usize) -> Vec<u8> {
        let (primary, secondary) = window_item_bytes(item);
        let mut slot = Vec::with_capacity(STRIDE_BYTES);
        slot.extend_from_slice(&primary);
        slot.extend_from_slice(&secondary);
        slot.resize(STRIDE_BYTES, 0);
        slot
    }

    fn staged_window_pool(
        context: &Arc<CudaContext>,
    ) -> Result<StreamingWeightPool, Box<dyn std::error::Error>> {
        let layout = StreamingWeightLayout::build(
            WINDOW_ITEMS,
            PRIMARY_BYTES,
            Some(SECONDARY_BYTES),
            WINDOW_SLOTS,
        )?;
        let mut pool = StreamingWeightPool::new(context, layout)?;
        for item in 0..WINDOW_ITEMS {
            let (primary, secondary) = window_item_bytes(item);
            pool.stage_item(item, &primary, &secondary)?;
        }
        assert!(pool.is_fully_staged());
        Ok(pool)
    }

    /// Asserts every resident item of the window pool holds exactly its staged
    /// bytes and that the published table agrees with the host cache.
    fn require_exact_window_residency(
        pool: &StreamingWeightPool,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let table = pool.read_table(stream)?;
        assert_eq!(table.len(), WINDOW_ITEMS);
        let mut resident = 0;
        for (item, &entry) in table.iter().enumerate() {
            match pool.slot_of(item)? {
                Some(slot) => {
                    assert_eq!(entry, slot as u32, "item {item} table entry");
                    assert_eq!(
                        pool.read_slot(stream, slot)?,
                        window_expected_slot(item),
                        "item {item} slot {slot} bytes"
                    );
                    resident += 1;
                }
                None => assert_eq!(entry, STREAMING_ABSENT_SLOT, "item {item} is absent"),
            }
        }
        assert_eq!(resident, WINDOW_SLOTS, "the window pool left a slot empty");
        Ok(())
    }

    fn exclusive_context() -> Result<Arc<CudaContext>, Box<dyn std::error::Error>> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err(
                "streaming weight pool qualification requires compute capability 12.0".into(),
            );
        }
        Ok(context)
    }

    /// Asserts every resident item's slot holds exactly its staged bytes and
    /// that the published table agrees with the host cache.
    fn require_exact_residency(
        pool: &StreamingWeightPool,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let table = pool.read_table(stream)?;
        assert_eq!(table.len(), ITEM_COUNT);
        for (item, &entry) in table.iter().enumerate() {
            match pool.slot_of(item)? {
                Some(slot) => {
                    assert_eq!(entry, slot as u32, "item {item} table entry");
                    assert_eq!(
                        pool.read_slot(stream, slot)?,
                        expected_slot(item),
                        "item {item} slot {slot} bytes"
                    );
                }
                None => assert_eq!(entry, STREAMING_ABSENT_SLOT, "item {item} is absent"),
            }
        }
        Ok(())
    }

    #[test]
    fn streaming_weight_pool_suite_byte_accounting_is_exact() {
        let layout = qualification_layout();

        assert_eq!(layout.item_count(), ITEM_COUNT);
        assert_eq!(layout.slot_count(), SLOT_COUNT);
        assert_eq!(layout.extent_bytes(), PRIMARY_BYTES + SECONDARY_BYTES);
        assert_eq!(layout.stride_bytes(), STRIDE_BYTES);
        assert_eq!(layout.extent_padding_bytes(), 0);
        assert_eq!(layout.slot_region_bytes(), SLOT_COUNT * STRIDE_BYTES);
        assert_eq!(layout.table_bytes(), ITEM_COUNT * 4);
        assert_eq!(layout.table_staging_bytes(), 4 * ITEM_COUNT * 4);
        assert_eq!(layout.host_pool_bytes(), ITEM_COUNT * STRIDE_BYTES);
        assert_eq!(
            layout.device_resident_bytes(),
            SLOT_COUNT * STRIDE_BYTES + ITEM_COUNT * 4
        );
        assert_eq!(
            layout.host_pinned_bytes(),
            ITEM_COUNT * STRIDE_BYTES + 4 * ITEM_COUNT * 4
        );
        assert_eq!(layout.host_mapped_bytes(), 0);
    }

    /// Structurally independent plain-LRU model: an ordered recency list of
    /// slot indices, least-recent first, and one item per slot. It shares no
    /// data structure with the engine's linked-list cache.
    struct LruModel {
        slots: Vec<Option<u32>>,
        recency: Vec<usize>,
        pinned: Vec<u64>,
        round: u64,
    }

    impl LruModel {
        fn new(slot_count: usize) -> Self {
            Self {
                slots: vec![None; slot_count],
                recency: Vec::with_capacity(slot_count),
                pinned: vec![0; slot_count],
                round: 0,
            }
        }

        fn touch(&mut self, slot: usize) {
            self.recency.retain(|&entry| entry != slot);
            self.recency.push(slot);
        }

        /// Returns `(item, slot, evicted)` for every admission of the round.
        fn admit_round(&mut self, items: &[u32]) -> Vec<(u32, u32, u32)> {
            self.round += 1;
            let mut admitted = Vec::new();
            for &item in items {
                if let Some(slot) = self.slots.iter().position(|held| *held == Some(item)) {
                    self.touch(slot);
                    self.pinned[slot] = self.round;
                    continue;
                }
                let slot = match self.slots.iter().position(Option::is_none) {
                    Some(free) => free,
                    None => *self
                        .recency
                        .iter()
                        .find(|&&slot| self.pinned[slot] != self.round)
                        .expect("a round never requests more items than the pool has slots"),
                };
                let evicted = self.slots[slot].unwrap_or(STREAMING_ABSENT_ITEM);
                self.slots[slot] = Some(item);
                self.touch(slot);
                self.pinned[slot] = self.round;
                admitted.push((item, slot as u32, evicted));
            }
            admitted
        }

        fn table(&self, item_count: usize) -> Vec<u32> {
            (0..item_count)
                .map(|item| {
                    self.slots
                        .iter()
                        .position(|held| *held == Some(item as u32))
                        .map_or(STREAMING_ABSENT_SLOT, |slot| slot as u32)
                })
                .collect()
        }
    }

    #[test]
    fn streaming_weight_pool_suite_lru_eviction_order_is_deterministic() {
        // Eviction order carries no numerical authority, but it is deterministic
        // and it is pinned here twice: against an explicit table for a short
        // sequence, and against an independent model over a long one.
        let mut cache = StreamingSlotCache::new(6, 3).unwrap();
        let mut admitted = Vec::new();
        let mut order = Vec::new();
        for round in [
            &[0u32, 1, 2][..],
            &[0][..],
            &[3][..],
            &[4, 5][..],
            &[0, 3][..],
            &[1][..],
        ] {
            admitted.clear();
            let report = cache.admit_round(round, &mut admitted).unwrap();
            order.push((
                report.hits(),
                report.misses(),
                admitted
                    .iter()
                    .map(|admission| (admission.item(), admission.slot(), admission.evicted()))
                    .collect::<Vec<_>>(),
            ));
        }

        assert_eq!(
            order,
            [
                (
                    0,
                    3,
                    vec![
                        (0, 0, STREAMING_ABSENT_ITEM),
                        (1, 1, STREAMING_ABSENT_ITEM),
                        (2, 2, STREAMING_ABSENT_ITEM),
                    ]
                ),
                (1, 0, vec![]),
                (0, 1, vec![(3, 1, 1)]),
                (0, 2, vec![(4, 2, 2), (5, 0, 0)]),
                (0, 2, vec![(0, 1, 3), (3, 2, 4)]),
                (0, 1, vec![(1, 0, 5)]),
            ]
        );
        assert_eq!((cache.hits(), cache.misses()), (1, 9));
        assert_eq!(cache.host_allocation_bytes(), 132);

        // Long deterministic sequence against the independent model. The
        // generator is a fixed LCG so the trace is reproducible byte for byte.
        const ITEMS: usize = 32;
        const SLOTS: usize = 8;
        const ROUNDS: usize = 512;
        let mut cache = StreamingSlotCache::new(ITEMS, SLOTS).unwrap();
        let mut model = LruModel::new(SLOTS);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };
        let mut requests = Vec::new();
        for _ in 0..ROUNDS {
            requests.clear();
            let width = 1 + next() % SLOTS;
            while requests.len() < width {
                let item = (next() % ITEMS) as u32;
                if !requests.contains(&item) {
                    requests.push(item);
                }
            }
            admitted.clear();
            cache.admit_round(&requests, &mut admitted).unwrap();
            assert_eq!(
                admitted
                    .iter()
                    .map(|admission| (admission.item(), admission.slot(), admission.evicted()))
                    .collect::<Vec<_>>(),
                model.admit_round(&requests),
                "admission stream diverged on requests {requests:?}"
            );
            assert_eq!(cache.table(), model.table(ITEMS));
        }
        assert!(cache.misses() > 0 && cache.hits() > 0);
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_every_cache_state_holds_identical_bits() -> Suite {
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut pool = staged_pool(&context)?;
        let addresses = pool.allocation_addresses();
        let slot_addresses = (0..SLOT_COUNT)
            .map(|slot| pool.slot_address(slot))
            .collect::<Result<Vec<_>, _>>()?;
        let table_address = pool.table_address()?;

        let (primary, secondary) = item_bytes(0);
        assert!(
            pool.stage_item(0, &primary, &secondary).is_err(),
            "a staged source extent was mutable"
        );

        // A cold pool publishes an all-absent table and no slot content.
        assert_eq!(
            pool.read_table(&stream)?,
            [STREAMING_ABSENT_SLOT; ITEM_COUNT]
        );

        // Cold-miss round: every item is uploaded and the table names its slot.
        let cold = pool.require(&[0, 1, 2, 3])?;
        assert_eq!((cold.hits(), cold.misses(), cold.stalled()), (0, 4, true));
        assert_eq!(cold.uploaded_bytes(), 4 * STRIDE_BYTES);
        require_exact_residency(&pool, &stream)?;
        let cold_slots = (0..SLOT_COUNT)
            .map(|slot| pool.read_slot(&stream, slot))
            .collect::<Result<Vec<_>, _>>()?;
        pool.record_replay_release(&stream)?;

        // Hit path: the identical request uploads nothing and changes no byte.
        let warm = pool.require(&[0, 1, 2, 3])?;
        assert_eq!((warm.hits(), warm.misses()), (4, 0));
        assert_eq!(warm.uploaded_bytes(), 0);
        for (slot, expected) in cold_slots.iter().enumerate() {
            assert_eq!(
                &pool.read_slot(&stream, slot)?,
                expected,
                "hit path slot {slot}"
            );
        }
        pool.record_replay_release(&stream)?;

        // Pathological eviction: a sweep wider than the pool, so every round
        // evicts, then the original items are reloaded into different slots.
        let before = device_memory_info(&context)?;
        for chunk in [
            &[4u32, 5, 6, 7][..],
            &[8, 9, 10, 11][..],
            &[0, 1, 2, 3][..],
            &[11, 0, 10, 1][..],
            &[5, 5, 5, 5][..],
        ] {
            let round = pool.require(chunk)?;
            assert!(round.stalled());
            require_exact_residency(&pool, &stream)?;
            pool.record_replay_release(&stream)?;
        }
        let after = device_memory_info(&context)?;
        assert_eq!(before, after, "streaming rounds allocated after warmup");

        // Post-eviction reload must reproduce the cold-miss bytes exactly.
        pool.reset()?;
        assert_eq!(
            pool.read_table(&stream)?,
            [STREAMING_ABSENT_SLOT; ITEM_COUNT]
        );
        let reloaded = pool.require(&[0, 1, 2, 3])?;
        assert_eq!(reloaded.misses(), 4);
        for (slot, expected) in cold_slots.iter().enumerate() {
            assert_eq!(
                &pool.read_slot(&stream, slot)?,
                expected,
                "post-eviction reload slot {slot}"
            );
        }
        pool.record_replay_release(&stream)?;

        assert_eq!(pool.allocation_addresses(), addresses);
        assert_eq!(
            (0..SLOT_COUNT)
                .map(|slot| pool.slot_address(slot))
                .collect::<Result<Vec<_>, _>>()?,
            slot_addresses
        );
        assert_eq!(pool.table_address()?, table_address);
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool cache-state exactness passed: {} items, {} slots, {} device bytes, {} pinned bytes, {} streamed bytes",
            ITEM_COUNT,
            SLOT_COUNT,
            pool.device_resident_bytes(),
            pool.host_pinned_bytes(),
            pool.uploaded_bytes()
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_a_miss_stalls_instead_of_serving_a_stale_slot() -> Suite {
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut pool = staged_pool(&context)?;

        // Fill every slot, then force a full-pool eviction sweep. Each readback
        // below runs on a stream that never waits for the transfer stream, so a
        // missing stall would surface the previous occupant's bytes.
        pool.require(&[0, 1, 2, 3])?;
        let table = pool.read_table(&stream)?;
        let counters = (pool.cache().hits(), pool.cache().misses());
        assert!(
            pool.reset().is_err(),
            "a reset without the prior replay release was admitted"
        );
        assert!(
            pool.require(&[4]).is_err(),
            "an eviction without the prior replay release was admitted"
        );
        assert_eq!(pool.read_table(&stream)?, table);
        assert_eq!((pool.cache().hits(), pool.cache().misses()), counters);
        pool.record_replay_release(&stream)?;
        for (round, victim) in [(&[4u32][..], 0usize), (&[5][..], 1), (&[6][..], 2)] {
            let stale = expected_slot(victim);
            let report = pool.require(round)?;
            assert_eq!((report.misses(), report.stalled()), (1, true));
            let admitted = usize::try_from(round[0]).unwrap();
            let slot = pool.slot_of(admitted)?.expect("admitted item is resident");
            let observed = pool.read_slot(&stream, slot)?;
            assert_ne!(observed, stale, "slot {slot} still holds the evicted item");
            assert_eq!(observed, expected_slot(admitted), "slot {slot} bytes");
            assert_eq!(pool.slot_of(victim)?, None);
            assert_eq!(pool.read_table(&stream)?[victim], STREAMING_ABSENT_SLOT);
            pool.record_replay_release(&stream)?;
        }

        // The overlap route produces the identical residency once fenced: the
        // eviction policy and the prefetch order carry no numerical authority.
        pool.reset()?;
        let prefetched = pool.prefetch(&[7, 8, 9, 10])?;
        assert!(!prefetched.stalled());
        pool.fence_replay(&stream)?;
        pool.synchronize()?;
        require_exact_residency(&pool, &stream)?;
        pool.record_replay_release(&stream)?;

        // A `require` over items a prefetch has not landed yet still stalls.
        pool.reset()?;
        pool.prefetch(&[0, 1, 2, 3])?;
        let required = pool.require(&[0, 1, 2, 3])?;
        assert_eq!((required.hits(), required.misses()), (4, 0));
        assert!(required.stalled());
        require_exact_residency(&pool, &stream)?;
        pool.record_replay_release(&stream)?;

        // A round wider than the slot budget, an unstaged item, and an
        // out-of-range item are all refused before anything changes, never
        // silently truncated into a partial residency.
        let table = pool.read_table(&stream)?;
        let counters = (pool.cache().hits(), pool.cache().misses());
        let streamed = pool.uploaded_bytes();
        for refused in [&[0u32, 1, 2, 3, 4][..], &[ITEM_COUNT as u32][..]] {
            assert!(pool.require(refused).is_err(), "admitted {refused:?}");
        }
        assert_eq!(pool.read_table(&stream)?, table);
        assert_eq!((pool.cache().hits(), pool.cache().misses()), counters);
        assert_eq!(pool.uploaded_bytes(), streamed);
        require_exact_residency(&pool, &stream)?;
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool stall-on-miss passed: {} hits, {} misses",
            pool.cache().hits(),
            pool.cache().misses()
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_replay_coexists_with_slot_streaming() -> Suite {
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let compute = context.new_stream().map_err(GpuError::from)?;
        let mut pool = staged_pool(&context)?;

        // A captured graph over stable addresses represents the consumer. This
        // subsystem adds no device kernel of its own.
        let mut layout = ArenaLayout::new();
        let source = layout.reserve::<u32>(1_024, 256)?;
        let destination = layout.reserve::<u32>(1_024, 256)?;
        let arena = DeviceArena::zeroed(&compute, &layout)?;
        arena.fill(&compute, source, 0x3c)?;
        compute.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&compute, || {
            arena.fill(&compute, destination, 0x00)?;
            // SAFETY: both prefixes name disjoint ranges of `arena`, which
            // outlives every replay below.
            unsafe {
                arena.copy_prefix_from_arena_async(&compute, destination, &arena, source, 1_024)
            }
        })?;

        // Isolated replay establishes the reference bits.
        // SAFETY: `arena` is the only allocation the recording captured and it
        // outlives the synchronize below.
        unsafe { graph.launch(&compute)? };
        compute.synchronize().map_err(GpuError::from)?;
        let isolated = arena.copy_to_host(&compute, destination)?;
        assert_eq!(isolated, vec![0x3c3c_3c3c; 1_024]);

        // Concurrent replay: the pool streams on its own transfer stream while
        // the graph replays on the compute stream, fenced by the publication
        // and reclaim events on every round.
        let rounds: [&[u32]; 8] = [
            &[0, 1, 2, 3],
            &[4, 5, 6, 7],
            &[8, 9, 10, 11],
            &[0, 4, 8, 1],
            &[2, 6, 10, 3],
            &[7, 11, 5, 9],
            &[0, 1, 2, 3],
            &[11, 10, 9, 8],
        ];
        for round in rounds {
            pool.prefetch(round)?;
            pool.fence_replay(&compute)?;
            // SAFETY: `arena` outlives the final synchronize below.
            unsafe { graph.launch(&compute)? };
            pool.record_replay_release(&compute)?;
        }
        compute.synchronize().map_err(GpuError::from)?;
        pool.synchronize()?;

        assert_eq!(
            arena.copy_to_host(&compute, destination)?,
            isolated,
            "graph replay bits changed under concurrent slot streaming"
        );
        require_exact_residency(&pool, &compute)?;
        assert_eq!(pool.uploaded_bytes() % STRIDE_BYTES as u64, 0);
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool replay coexistence passed: {} rounds, {} streamed bytes over {} replays",
            rounds.len(),
            pool.uploaded_bytes(),
            rounds.len() + 1
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_a_round_window_holds_identical_bits() -> Suite {
        // The production round shape at its most hostile: 48 rounds against one
        // pool of four slots, each round requesting four items the previous
        // round did not, so every round evicts everything the replay it is
        // fenced against just read. That is the tight end of the reclaim
        // discipline: `reclaim_generation` is always the previous generation,
        // so every round waits for its predecessor's replay. The point of
        // the suite is that refining rule 1 did not weaken it there.
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let compute = context.new_stream().map_err(GpuError::from)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut pool = staged_pool(&context)?;

        let mut layout = ArenaLayout::new();
        let source = layout.reserve::<u32>(1_024, 256)?;
        let destination = layout.reserve::<u32>(1_024, 256)?;
        let arena = DeviceArena::zeroed(&compute, &layout)?;
        arena.fill(&compute, source, 0x3c)?;
        compute.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&compute, || {
            arena.fill(&compute, destination, 0x00)?;
            // SAFETY: both prefixes name disjoint ranges of `arena`, which
            // outlives every replay below.
            unsafe {
                arena.copy_prefix_from_arena_async(&compute, destination, &arena, source, 1_024)
            }
        })?;
        // SAFETY: `arena` outlives the synchronize below.
        unsafe { graph.launch(&compute)? };
        compute.synchronize().map_err(GpuError::from)?;
        let isolated = arena.copy_to_host(&compute, destination)?;
        assert_eq!(isolated, vec![0x3c3c_3c3c; 1_024]);

        let before = device_memory_info(&context)?;
        let mut requests = Vec::with_capacity(SLOT_COUNT);
        for round in 0..WINDOW_ROUNDS {
            requests.clear();
            requests.extend(
                (0..SLOT_COUNT).map(|lane| ((round * SLOT_COUNT + lane) % ITEM_COUNT) as u32),
            );

            // The five ordering points, in the production order.
            let report = pool.require(&requests)?;
            assert_eq!(
                (report.hits(), report.misses(), report.stalled()),
                (0, SLOT_COUNT, true),
                "round {round} was not a full-eviction round"
            );
            // Round 0 takes free slots no replay has read; every later round
            // takes back exactly the slots the previous generation requested.
            assert_eq!(
                report.reclaim_generation(),
                round as u64,
                "round {round} reclaim generation"
            );
            pool.fence_replay(&compute)?;
            // SAFETY: `arena` outlives the final synchronize below.
            unsafe { graph.launch(&compute)? };
            pool.record_replay_release(&compute)?;
            assert_eq!(pool.released_generation(), Some(round as u64 + 1));

            // `require` stalled on the publication fence, so this readback on a
            // stream that never waits for the transfer stream must already see
            // the round's bytes and none of the evicted occupants'.
            require_exact_residency(&pool, &stream)?;
        }
        compute.synchronize().map_err(GpuError::from)?;
        pool.synchronize()?;
        let after = device_memory_info(&context)?;

        assert_eq!(before, after, "the round window allocated after warmup");
        assert_eq!(
            arena.copy_to_host(&compute, destination)?,
            isolated,
            "graph replay bits changed across the round window"
        );
        require_exact_residency(&pool, &stream)?;
        assert_eq!(
            pool.uploaded_bytes(),
            (WINDOW_ROUNDS * SLOT_COUNT * STRIDE_BYTES) as u64
        );
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool round window passed: {WINDOW_ROUNDS} rounds x {SLOT_COUNT} evictions, {} streamed bytes, released through generation {:?}",
            pool.uploaded_bytes(),
            pool.released_generation()
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_uploads_overlap_an_in_flight_replay() -> Suite {
        // The loose end of the same discipline, and the reason it was refined:
        // 48 rounds issued against *one* consumer replay window. Every one of
        // them evicts slots the window's replay never read, so rule 1 resolves
        // each to a fence that retired before the window opened and none of
        // them waits for the window. The proof is an ordering observation:
        // the window's uploads are seen complete while the replay fence is
        // still unsignalled, not a duration.
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let compute = context.new_stream().map_err(GpuError::from)?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut pool = staged_window_pool(&context)?;
        let replay_complete = context.new_event(None).map_err(GpuError::from)?;

        let mut layout = ArenaLayout::new();
        let source = layout.reserve::<u32>(WINDOW_COPY_ELEMENTS, 256)?;
        let destination = layout.reserve::<u32>(WINDOW_COPY_ELEMENTS, 256)?;
        let arena = DeviceArena::zeroed(&compute, &layout)?;
        arena.fill(&compute, source, 0x5a)?;
        compute.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&compute, || {
            for _ in 0..WINDOW_COPIES_PER_GRAPH {
                // SAFETY: both prefixes name disjoint ranges of `arena`, which
                // outlives every replay below.
                unsafe {
                    arena.copy_prefix_from_arena_async(
                        &compute,
                        destination,
                        &arena,
                        source,
                        WINDOW_COPY_ELEMENTS,
                    )?;
                }
            }
            Ok(())
        })?;
        // SAFETY: `arena` outlives the synchronize below.
        unsafe { graph.launch(&compute)? };
        compute.synchronize().map_err(GpuError::from)?;
        let isolated = arena.copy_slice_to_host(&compute, destination, 0, 1_024)?;
        assert_eq!(isolated, vec![0x5a5a_5a5a; 1_024]);

        // Warm-up: 49 rounds in the production order, each released, all drained.
        // They leave 490 slots occupied at generations 1..49 and 10 slots free.
        let mut item = 0u32;
        let mut requests = Vec::with_capacity(WINDOW_REQUESTS);
        let next_round = |requests: &mut Vec<u32>, item: &mut u32| {
            requests.clear();
            for _ in 0..WINDOW_REQUESTS {
                requests.push(*item);
                *item += 1;
            }
        };
        for round in 0..WINDOW_WARM_ROUNDS {
            next_round(&mut requests, &mut item);
            let report = pool.require(&requests)?;
            assert_eq!(report.misses(), WINDOW_REQUESTS, "warm round {round}");
            assert_eq!(report.reclaim_generation(), 0, "warm round {round} evicted");
            pool.fence_replay(&compute)?;
            // SAFETY: `arena` outlives the final synchronize below.
            unsafe { graph.launch(&compute)? };
            pool.record_replay_release(&compute)?;
        }
        compute.synchronize().map_err(GpuError::from)?;
        pool.synchronize()?;
        assert_eq!(
            pool.released_generation(),
            Some(WINDOW_WARM_ROUNDS as u64),
            "the warm-up did not release its whole timeline"
        );

        // Open the window: one round onto the last free slots, then a replay
        // window long enough to outlive everything the window enqueues, and one
        // release covering the whole window.
        next_round(&mut requests, &mut item);
        let opener = pool.require(&requests)?;
        assert_eq!(opener.reclaim_generation(), 0);
        pool.fence_replay(&compute)?;
        for _ in 0..WINDOW_GRAPH_LAUNCHES {
            // SAFETY: `arena` outlives the final synchronize below.
            unsafe { graph.launch(&compute)? };
        }
        replay_complete.record(&compute).map_err(GpuError::from)?;
        pool.record_replay_release(&compute)?;
        let window_generation = WINDOW_WARM_ROUNDS as u64 + 1;
        assert_eq!(pool.released_generation(), Some(window_generation));

        // The window: 48 rounds whose evictions all predate the open window.
        assert!(
            !replay_complete.query().map_err(GpuError::from)?,
            "the replay window retired before the streaming window opened"
        );
        for round in 0..WINDOW_ROUNDS {
            next_round(&mut requests, &mut item);
            let report = pool.prefetch(&requests)?;
            assert_eq!(report.misses(), WINDOW_REQUESTS, "window round {round}");
            assert!(!report.stalled());
            // Round `r` of the window takes back the slots warm round `r + 1`
            // filled, so its reclaim generation is always older than the
            // generation the in-flight replay is reading.
            assert_eq!(
                report.reclaim_generation(),
                round as u64 + 1,
                "window round {round} reclaim generation"
            );
            assert!(
                report.reclaim_generation() < window_generation,
                "window round {round} would have to wait for the open window"
            );
        }
        // The ordering proof. The window's uploads were issued *after* the
        // replay window was enqueued; observing the last round's publication
        // fence complete while the replay fence is still unsignalled is only
        // possible if those uploads did not queue behind the replay. Under one
        // global reclaim event they necessarily would have.
        let mut observed_overlap = false;
        let mut polls = 0u64;
        loop {
            let uploaded = pool.publication_completed()?;
            let replayed = replay_complete.query().map_err(GpuError::from)?;
            polls += 1;
            if uploaded && !replayed {
                observed_overlap = true;
                break;
            }
            if replayed {
                break;
            }
        }
        assert!(
            observed_overlap,
            "the window's uploads never completed before the replay fence released, after {polls} polls"
        );

        // The stall law inside the window. The next round takes back the slots
        // warm round 49 filled, released, and retired, so it still does not
        // wait; the round after it takes back the slots the window *opener*
        // filled, whose only release is the window fence itself, so it must
        // block the host until the whole replay window has retired.
        next_round(&mut requests, &mut item);
        let unblocked = pool.require(&requests)?;
        assert_eq!(unblocked.reclaim_generation(), WINDOW_WARM_ROUNDS as u64);
        let replay_still_running = !replay_complete.query().map_err(GpuError::from)?;
        assert!(
            replay_still_running,
            "the replay window retired before the stall-law round could be issued"
        );
        next_round(&mut requests, &mut item);
        let blocked = pool.require(&requests)?;
        assert_eq!(blocked.reclaim_generation(), window_generation);
        assert!(blocked.stalled());
        assert!(
            replay_complete.query().map_err(GpuError::from)?,
            "a round reclaiming the open window's own generation returned before it retired"
        );

        compute.synchronize().map_err(GpuError::from)?;
        pool.synchronize()?;
        assert_eq!(
            arena.copy_slice_to_host(&compute, destination, 0, 1_024)?,
            isolated,
            "graph replay bits changed under the streaming window"
        );
        require_exact_window_residency(&pool, &stream)?;
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool upload/replay overlap passed: {WINDOW_ROUNDS} rounds x {WINDOW_REQUESTS} uploads issued against one {WINDOW_GRAPH_LAUNCHES}-replay window, overlap observed after {polls} polls, {} streamed bytes over {WINDOW_SLOTS} slots",
            pool.uploaded_bytes()
        );
        println!(
            "streaming weight pool overlap evidence is an ordering observation and never blesses a performance baseline"
        );

        Ok(())
    }

    /// A borrowed primary-extent source over owned host bytes.
    ///
    /// Production hands the pool a checkpoint `mmap`; the qualification hands it
    /// an owned vector, because what has to be proved is that the bounce path
    /// reproduces the pinned path byte for byte, not that `mmap` works.
    struct MappedPrimaries {
        extents: Vec<Vec<u8>>,
    }

    impl StreamingMappedPrimary for MappedPrimaries {
        fn primary_extent(&self, item: usize) -> EngineResult<&[u8]> {
            self.extents
                .get(item)
                .map(Vec::as_slice)
                .ok_or_else(|| EngineError::Contract {
                    code: EngineErrorCode::Route,
                    message: format!("mapped item {item} is outside the map"),
                })
        }
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_the_bounce_path_reproduces_the_pinned_path() -> Suite {
        // Both admitted host postures over one inventory, side by side. The
        // mapped posture fills a slot with two uploads: the borrowed primary
        // through a two-slot bounce ring, then the pooled secondary,
        // and the slot it produces has to be the slot the single pinned upload
        // would have produced, byte for byte, in every cache state.
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let stream = context.new_stream().map_err(GpuError::from)?;

        let mapped_layout = StreamingWeightLayout::build_mapped_primary(
            ITEM_COUNT,
            PRIMARY_BYTES,
            Some(SECONDARY_BYTES),
            SLOT_COUNT,
            BOUNCE_SLOTS,
        )?;
        // The device plan is the pinned plan; only the host classes moved.
        assert_eq!(mapped_layout.stride_bytes(), STRIDE_BYTES);
        assert_eq!(
            mapped_layout.device_resident_bytes(),
            qualification_layout().device_resident_bytes()
        );
        assert_eq!(mapped_layout.host_stride_bytes(), SECONDARY_BYTES);
        assert_eq!(
            mapped_layout.host_pool_bytes(),
            ITEM_COUNT * SECONDARY_BYTES
        );
        assert_eq!(
            mapped_layout.bounce_ring_bytes(),
            BOUNCE_SLOTS * PRIMARY_BYTES
        );
        assert_eq!(
            mapped_layout.host_pinned_bytes(),
            ITEM_COUNT * SECONDARY_BYTES + 4 * ITEM_COUNT * 4 + BOUNCE_SLOTS * PRIMARY_BYTES
        );
        assert_eq!(
            mapped_layout.host_mapped_bytes(),
            ITEM_COUNT * PRIMARY_BYTES
        );

        // The postures are admission constants: neither constructor accepts the
        // other's layout, so nothing can infer a posture at runtime.
        assert!(StreamingWeightPool::new(&context, mapped_layout).is_err());
        assert!(
            StreamingWeightPool::new_with_mapped_primary(
                &context,
                qualification_layout(),
                Box::new(MappedPrimaries {
                    extents: Vec::new()
                }),
            )
            .is_err()
        );

        let extents = (0..ITEM_COUNT).map(|item| item_bytes(item).0).collect();
        let mut mapped = StreamingWeightPool::new_with_mapped_primary(
            &context,
            mapped_layout,
            Box::new(MappedPrimaries { extents }),
        )?;
        for item in 0..ITEM_COUNT {
            let (primary, secondary) = item_bytes(item);
            // The mapped posture never pools the primary extent, so handing it
            // one is refused rather than silently copied twice.
            assert!(mapped.stage_item(item, &primary, &secondary).is_err());
            mapped.stage_item(item, &[], &secondary)?;
            assert_eq!(mapped.staged_item(item)?, secondary.as_slice());
        }
        assert!(mapped.is_fully_staged());
        let mut pinned = staged_pool(&context)?;

        // Identical round sequences, including a full-pool eviction sweep and a
        // post-eviction reload, checked slot by slot after every round.
        let rounds: [&[u32]; 8] = [
            &[0, 1, 2, 3],
            &[4, 5, 6, 7],
            &[8, 9, 10, 11],
            &[0, 4, 8, 1],
            &[2, 6, 10, 3],
            &[7, 11, 5, 9],
            &[5, 5, 5, 5],
            &[0, 1, 2, 3],
        ];
        for (index, round) in rounds.iter().enumerate() {
            let pinned_round = pinned.require(round)?;
            let mapped_round = mapped.require(round)?;
            assert_eq!(
                (pinned_round.hits(), pinned_round.misses()),
                (mapped_round.hits(), mapped_round.misses()),
                "round {index} admission differed between the postures"
            );
            assert_eq!(
                pinned_round.reclaim_generation(),
                mapped_round.reclaim_generation(),
                "round {index} reclaim generation differed between the postures"
            );
            require_exact_residency(&mapped, &stream)?;
            for slot in 0..SLOT_COUNT {
                assert_eq!(
                    mapped.read_slot(&stream, slot)?,
                    pinned.read_slot(&stream, slot)?,
                    "round {index} slot {slot} differed between the postures"
                );
            }
            pinned.record_replay_release(&stream)?;
            mapped.record_replay_release(&stream)?;
        }

        // Ring-wraparound safety. Twenty-four admissions ran through two ring
        // slots, so the ring was reused twenty-two times and every reuse had to
        // wait on that slot's own upload fence first. Every slot above still
        // held its own item's bytes, which a torn bounce slot could not.
        let admissions = pinned.uploaded_bytes() / STRIDE_BYTES as u64;
        assert_eq!(mapped.uploaded_bytes(), pinned.uploaded_bytes());
        assert_eq!(
            mapped.bounce_wraparound_waits(),
            admissions - BOUNCE_SLOTS as u64,
            "the bounce ring did not fence every reuse"
        );
        assert_eq!(pinned.bounce_wraparound_waits(), 0);

        // The stall law through the ring: a readback on a stream that never
        // waits for the transfer stream must not see the evicted occupant.
        for (round, victim) in [(&[4u32][..], 0usize), (&[5][..], 1), (&[6][..], 2)] {
            let stale = expected_slot(victim);
            let report = mapped.require(round)?;
            assert_eq!((report.misses(), report.stalled()), (1, true));
            let admitted = usize::try_from(round[0]).unwrap();
            let slot = mapped
                .slot_of(admitted)?
                .expect("admitted item is resident");
            let observed = mapped.read_slot(&stream, slot)?;
            assert_ne!(observed, stale, "slot {slot} still holds the evicted item");
            assert_eq!(observed, expected_slot(admitted), "slot {slot} bytes");
            assert_eq!(mapped.read_table(&stream)?[victim], STREAMING_ABSENT_SLOT);
            mapped.record_replay_release(&stream)?;
        }

        // A mapping that hands back the wrong extent length is refused at the
        // upload, not padded into a slot.
        let mut short = StreamingWeightPool::new_with_mapped_primary(
            &context,
            StreamingWeightLayout::build_mapped_primary(
                ITEM_COUNT,
                PRIMARY_BYTES,
                Some(SECONDARY_BYTES),
                SLOT_COUNT,
                BOUNCE_SLOTS,
            )?,
            Box::new(MappedPrimaries {
                extents: (0..ITEM_COUNT)
                    .map(|_| vec![0u8; PRIMARY_BYTES - 1])
                    .collect(),
            }),
        )?;
        for item in 0..ITEM_COUNT {
            short.stage_item(item, &[], &item_bytes(item).1)?;
        }
        assert!(short.require(&[0]).is_err());

        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool bounce path passed: {} admissions through a {BOUNCE_SLOTS}-slot ring, {} wraparound fences, {} pinned bytes vs {} mapped bytes",
            admissions,
            mapped.bounce_wraparound_waits(),
            mapped.host_pinned_bytes(),
            mapped.host_mapped_bytes()
        );

        Ok(())
    }

    /// A borrowed source that can be made to fail on one chosen item.
    ///
    /// Cheap to clone and shared with the pool it is handed to, so a suite can
    /// arm a failure, observe the round that hits it, and then repair the source
    /// and retry, which is the whole shape of the commit-ordering law.
    #[derive(Clone)]
    struct FailingPrimaries {
        extents: Arc<Vec<Vec<u8>>>,
        /// Item whose extent the source refuses outright.
        refuses: Arc<AtomicUsize>,
        /// Item whose extent the source returns short.
        truncates: Arc<AtomicUsize>,
        /// Item whose extent is whole on its first read and short afterwards,
        /// which is the one way a caller can drive a failure *past* validation.
        tears: Arc<AtomicUsize>,
        reads: Arc<AtomicUsize>,
    }

    /// Sentinel for "this source refuses nothing".
    const NO_FAILURE: usize = usize::MAX;

    impl FailingPrimaries {
        fn new() -> Self {
            Self {
                extents: Arc::new((0..ITEM_COUNT).map(|item| item_bytes(item).0).collect()),
                refuses: Arc::new(AtomicUsize::new(NO_FAILURE)),
                truncates: Arc::new(AtomicUsize::new(NO_FAILURE)),
                tears: Arc::new(AtomicUsize::new(NO_FAILURE)),
                reads: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn repair(&self) {
            self.refuses.store(NO_FAILURE, Ordering::Relaxed);
            self.truncates.store(NO_FAILURE, Ordering::Relaxed);
            self.tears.store(NO_FAILURE, Ordering::Relaxed);
        }
    }

    impl StreamingMappedPrimary for FailingPrimaries {
        fn primary_extent(&self, item: usize) -> EngineResult<&[u8]> {
            let extent =
                self.extents
                    .get(item)
                    .map(Vec::as_slice)
                    .ok_or_else(|| EngineError::Contract {
                        code: EngineErrorCode::Route,
                        message: format!("mapped item {item} is outside the map"),
                    })?;
            if item == self.refuses.load(Ordering::Relaxed) {
                return Err(EngineError::Contract {
                    code: EngineErrorCode::Route,
                    message: format!("mapped item {item} is unreadable"),
                });
            }
            if item == self.truncates.load(Ordering::Relaxed) {
                return Ok(&extent[..extent.len() - 1]);
            }
            if item == self.tears.load(Ordering::Relaxed)
                && self.reads.fetch_add(1, Ordering::Relaxed) > 0
            {
                return Ok(&extent[..extent.len() - 1]);
            }

            Ok(extent)
        }
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn streaming_weight_pool_suite_a_failed_upload_never_commits_residency() -> Suite {
        // Rule 6. A round is decided against unchanged state and every borrowed
        // source it will read is resolved before the first byte is enqueued, so
        // a source that refuses leaves no residency behind. The failure mode
        // this forecloses is a *phantom hit*: host residency claiming a slot
        // whose device bytes were never uploaded, which a retry would then serve
        // as a hit while the device table still said the item was absent.
        let _preflight = device_benchmark::preflight()?;
        let context = exclusive_context()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let source = FailingPrimaries::new();
        let mut pool = StreamingWeightPool::new_with_mapped_primary(
            &context,
            StreamingWeightLayout::build_mapped_primary(
                ITEM_COUNT,
                PRIMARY_BYTES,
                Some(SECONDARY_BYTES),
                SLOT_COUNT,
                BOUNCE_SLOTS,
            )?,
            Box::new(source.clone()),
        )?;
        for item in 0..ITEM_COUNT {
            pool.stage_item(item, &[], &item_bytes(item).1)?;
        }

        // Warm the pool so the failing round has real residency to corrupt.
        pool.require(&[0, 1, 2, 3])?;
        require_exact_residency(&pool, &stream)?;
        pool.record_replay_release(&stream)?;

        let table = pool.read_table(&stream)?;
        let residency = (0..ITEM_COUNT)
            .map(|item| pool.slot_of(item))
            .collect::<Result<Vec<_>, _>>()?;
        let counters = (pool.cache().hits(), pool.cache().misses());
        let streamed = pool.uploaded_bytes();
        let waits = pool.bounce_wraparound_waits();

        // Two ways a borrowed source can refuse a round: outright, and by
        // handing back an extent of the wrong length. Both must be caught before
        // anything is committed, and both must leave the pool retryable.
        for (armed, flag) in [(4usize, &source.refuses), (5usize, &source.truncates)] {
            flag.store(armed, Ordering::Relaxed);
            let error = pool
                .require(&[armed as u32, 6, 7])
                .expect_err("a refusing source must fail the round");
            flag.store(NO_FAILURE, Ordering::Relaxed);

            // No phantom hit: the failing item owns no slot, and neither do the
            // items that shared its round.
            for item in [armed, 6, 7] {
                assert_eq!(
                    pool.slot_of(item)?,
                    None,
                    "item {item} took a slot from a round that failed: {error}"
                );
            }
            // Nothing the round would have evicted lost its slot either.
            assert_eq!(
                (0..ITEM_COUNT)
                    .map(|item| pool.slot_of(item))
                    .collect::<Result<Vec<_>, _>>()?,
                residency,
                "a failed round moved committed residency"
            );
            // The device table never held a stale entry across the failure, and
            // no byte was streamed for it.
            assert_eq!(pool.read_table(&stream)?, table);
            assert_eq!((pool.cache().hits(), pool.cache().misses()), counters);
            assert_eq!(pool.uploaded_bytes(), streamed);
            assert_eq!(pool.bounce_wraparound_waits(), waits);
            assert!(
                pool.poisoned().is_none(),
                "a failure before the first upload must stay retryable"
            );
            require_exact_residency(&pool, &stream)?;
        }

        // The retry. With the source repaired the same round must be a genuine
        // set of misses, real uploads rather than a hit against phantom residency,
        // and only then does residency and the table commit.
        source.repair();
        let retried = pool.require(&[4, 6, 7])?;
        assert_eq!(
            (retried.hits(), retried.misses()),
            (0, 3),
            "the retry served a phantom hit instead of re-uploading"
        );
        assert_eq!(retried.uploaded_bytes(), 3 * STRIDE_BYTES);
        assert_eq!(
            pool.uploaded_bytes(),
            streamed + 3 * STRIDE_BYTES as u64,
            "the retry did not re-fetch"
        );
        assert!(pool.bounce_wraparound_waits() > waits);
        for item in [4usize, 6, 7] {
            let slot = pool.slot_of(item)?.expect("the retry must be resident");
            assert_eq!(pool.read_table(&stream)?[item], slot as u32);
            assert_eq!(pool.read_slot(&stream, slot)?, expected_slot(item));
        }
        require_exact_residency(&pool, &stream)?;

        // The irreducible window: a source that validates and then tears fails
        // *past* the first upload, where the device already holds bytes no
        // published table describes. The pool cannot prove what those slots hold,
        // so it refuses every later round rather than inventing residency.
        let torn = FailingPrimaries::new();
        torn.tears.store(2, Ordering::Relaxed);
        let mut poisoned = StreamingWeightPool::new_with_mapped_primary(
            &context,
            StreamingWeightLayout::build_mapped_primary(
                ITEM_COUNT,
                PRIMARY_BYTES,
                Some(SECONDARY_BYTES),
                SLOT_COUNT,
                BOUNCE_SLOTS,
            )?,
            Box::new(torn.clone()),
        )?;
        for item in 0..ITEM_COUNT {
            poisoned.stage_item(item, &[], &item_bytes(item).1)?;
        }
        assert!(poisoned.poisoned().is_none());
        assert!(
            poisoned.require(&[2]).is_err(),
            "a source that tears after validation must fail its round"
        );
        let reason = poisoned
            .poisoned()
            .expect("a failure past the first upload must poison the pool");
        assert!(reason.contains("mid-round"), "{reason}");
        assert_eq!(
            poisoned.slot_of(2)?,
            None,
            "the torn round left a phantom hit"
        );
        assert!(
            poisoned.require(&[0]).is_err(),
            "a poisoned pool admitted a round"
        );
        assert!(
            poisoned.prefetch(&[0]).is_err(),
            "a poisoned pool admitted a prefetch"
        );
        assert!(
            poisoned.reset().is_err(),
            "a poisoned pool admitted a reset"
        );
        assert!(
            poisoned.stage_item(0, &[], &item_bytes(0).1).is_err(),
            "a poisoned pool admitted new staging"
        );

        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool failed-upload ordering passed: 2 pre-upload refusals left {} table bytes and {} streamed bytes untouched, the retry re-uploaded {} bytes, and a post-upload tear poisoned the pool",
            table.len() * 4,
            streamed,
            3 * STRIDE_BYTES
        );

        Ok(())
    }
}

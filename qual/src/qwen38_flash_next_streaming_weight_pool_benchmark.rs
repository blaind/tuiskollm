//! Streamed-byte accounting and the opt-in large-pinned-pool timing report.
//!
//! The pool adds no device route, so there is no production graph to time and
//! no performance baseline to bless here. What this sibling does own is the
//! byte accounting the exact expert route is measured against: the admitted
//! extent, the pool and cache totals, and the streamed bytes one
//! decode token costs at a given hit rate, plus an opt-in report of what a
//! multi-gibibyte pinned allocation actually costs on this box.

#[cfg(test)]
mod tests {
    use crate::device_benchmark;
    use std::time::{Duration, Instant};
    use tuisko_engine::{StreamingResidencyAccounting, StreamingWeightLayout, StreamingWeightPool};
    use tuisko_gpu::{CudaContext, GpuError};

    /// Exact Qwen3.8-Flash-Next expert-pool geometry.
    const QWEN38_FLASH_NEXT_LAYERS: usize = 48;
    const QWEN38_FLASH_NEXT_EXPERTS_PER_LAYER: usize = 512;
    const QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES: usize = 2_457_600;
    const QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES: usize = 307_200;
    const QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS: usize = 8;
    /// Routed experts per token per layer, plus the one shared expert.
    const QWEN38_FLASH_NEXT_ROUTED_EXPERTS: usize = 10;
    const QWEN38_FLASH_NEXT_SHARED_EXPERTS: usize = 1;

    /// Environment variable that admits the multi-gibibyte pinned-pool report.
    const BIG_POOL_ENVIRONMENT: &str = "TUISKO_STREAMING_BIG_POOL";

    /// Largest pinned pool the report will allocate, in gibibytes.
    ///
    /// Pinned pages are evicted from the page cache and this box shares its RAM
    /// with a checkpoint download and sibling agents, so the report refuses to
    /// pin more than this even when asked for more.
    const BIG_POOL_MAX_GIB: usize = 8;

    /// Pool size when the environment variable names no usable size.
    const BIG_POOL_DEFAULT_GIB: usize = 2;

    const _: () = assert!(BIG_POOL_DEFAULT_GIB <= BIG_POOL_MAX_GIB);

    /// Items in the complete Qwen3.8-Flash-Next expert pool.
    const fn expert_pool_items() -> usize {
        QWEN38_FLASH_NEXT_LAYERS * QWEN38_FLASH_NEXT_EXPERTS_PER_LAYER
    }

    /// Host-to-device bytes one fully-missing decode token would stream.
    ///
    /// One expert is one contiguous admitted extent, so the token cost is the
    /// selection count times the stride and nothing else.
    const fn token_streamed_bytes(stride_bytes: usize, shared: bool) -> usize {
        let experts = if shared {
            QWEN38_FLASH_NEXT_ROUTED_EXPERTS + QWEN38_FLASH_NEXT_SHARED_EXPERTS
        } else {
            QWEN38_FLASH_NEXT_ROUTED_EXPERTS
        };
        QWEN38_FLASH_NEXT_LAYERS * experts * stride_bytes
    }

    /// Host-to-device bytes one decode token streams at hit rate `hit_rate`.
    fn streamed_bytes_at_hit_rate(stride_bytes: usize, hit_rate: f64) -> f64 {
        token_streamed_bytes(stride_bytes, true) as f64 * (1.0 - hit_rate)
    }

    /// Seconds per gibibyte a measured pinning duration implies.
    fn seconds_per_gibibyte(bytes: usize, elapsed: Duration) -> f64 {
        elapsed.as_secs_f64() / (bytes as f64 / (1u64 << 30) as f64)
    }

    /// Gigabytes per second a measured transfer implies.
    fn gigabytes_per_second(bytes: u64, elapsed: Duration) -> f64 {
        bytes as f64 / elapsed.as_secs_f64() / 1e9
    }

    #[test]
    fn streaming_weight_pool_suite_benchmark_inventory_and_accounting_are_exact() {
        let items = expert_pool_items();
        // One quarter residency is the trace study's floor: global-budget plain
        // LRU reaches 88% at 33%, and PCIe stops binding above roughly 25%.
        let layout = StreamingWeightLayout::build(
            items,
            QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES,
            Some(QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES),
            items / 4,
        )
        .unwrap();

        assert_eq!(items, 24_576);
        assert_eq!(layout.slot_count(), 6_144);
        assert_eq!(layout.extent_bytes(), 2_764_800);
        assert_eq!(layout.stride_bytes(), 2_764_800);
        assert_eq!(layout.extent_padding_bytes(), 0);
        assert_eq!(layout.host_pool_bytes(), 67_947_724_800);
        assert_eq!(layout.slot_region_bytes(), 16_986_931_200);
        assert_eq!(layout.table_bytes(), 98_304);
        assert_eq!(layout.table_staging_bytes(), 393_216);
        assert_eq!(layout.device_resident_bytes(), 16_987_029_504);
        assert_eq!(layout.host_pinned_bytes(), 67_948_118_016);
        assert_eq!(layout.host_mapped_bytes(), 0);

        // Routed-only is 1.24 GiB/token; the shared expert adds one selection
        // per layer.
        assert_eq!(
            token_streamed_bytes(layout.stride_bytes(), false),
            1_327_104_000
        );
        assert_eq!(
            token_streamed_bytes(layout.stride_bytes(), true),
            1_459_814_400
        );
        // Two bases, both pinned, so the hit-rate row can never be read against
        // the wrong one again. Only the routed selections stream; the shared
        // expert is device-resident. At h = 0.85 this is 199,065,600 B.
        // Applying the same hit rate to the routed-plus-shared base
        // gives 218,972,160 B, which is h = 0.835 against the routed base.
        let planned = streamed_bytes_at_hit_rate(layout.stride_bytes(), 0.85);
        assert!(
            (planned - 218_972_160.0).abs() < 1.0,
            "planned streamed bytes at h=0.85 over routed+shared were {planned}"
        );
        let routed = token_streamed_bytes(layout.stride_bytes(), false) as f64 * 0.15;
        assert!(
            (routed - 199_065_600.0).abs() < 1.0,
            "planned streamed bytes at h=0.85 over routed only were {routed}"
        );
        assert_eq!(
            (218_972_160.0 / token_streamed_bytes(layout.stride_bytes(), false) as f64 * 1_000.0)
                .round() as u64,
            165,
            "218,972,160 B is a 0.835 hit rate against the routed base"
        );

        // The reporting helpers must convert exactly, not approximately.
        assert!((seconds_per_gibibyte(1 << 30, Duration::from_millis(180)) - 0.18).abs() < 1e-9);
        assert!((gigabytes_per_second(57_000_000_000, Duration::from_secs(1)) - 57.0).abs() < 1e-9);
        assert_eq!(BIG_POOL_ENVIRONMENT, "TUISKO_STREAMING_BIG_POOL");
        assert_eq!((BIG_POOL_DEFAULT_GIB, BIG_POOL_MAX_GIB), (2, 8));
    }

    #[test]
    fn streaming_weight_pool_suite_benchmark_both_host_postures_are_exact() {
        // The two host postures at the exact Qwen3.8-Flash-Next inventory.
        // Which one a product ships is an admission constant resolved by
        // arithmetic: the fully-pinned pool is 67,947,724,800 B (63.28 GiB) and
        // the 64 GB box has 59.2 GiB usable, so on that box only the mapped
        // posture is constructible at all.
        let items = expert_pool_items();
        let pinned = StreamingWeightLayout::build(
            items,
            QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES,
            Some(QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES),
            items / 4,
        )
        .unwrap();
        let mapped = StreamingWeightLayout::build_mapped_primary(
            items,
            QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES,
            Some(QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES),
            items / 4,
            QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS,
        )
        .unwrap();

        // Nothing device-side moves between the postures.
        assert_eq!(mapped.stride_bytes(), 2_764_800);
        assert_eq!(mapped.slot_count(), 6_144);
        assert_eq!(mapped.slot_region_bytes(), 16_986_931_200);
        assert_eq!(mapped.table_bytes(), 98_304);
        assert_eq!(mapped.device_resident_bytes(), 16_987_029_504);
        assert_eq!(
            mapped.device_resident_bytes(),
            pinned.device_resident_bytes()
        );

        // 128 GB posture: the whole item pool is pinned, nothing is file-backed.
        assert_eq!(pinned.host_pool_bytes(), 67_947_724_800);
        assert_eq!(pinned.table_staging_bytes(), 393_216);
        assert_eq!(pinned.bounce_ring_bytes(), 0);
        assert_eq!(pinned.host_pinned_bytes(), 67_948_118_016);
        assert_eq!(pinned.host_mapped_bytes(), 0);

        // 64 GB posture: 24,576 swizzled 307,200 B scale extents stay pinned,
        // the 2,457,600 B packed code extents are borrowed from the checkpoint,
        // and eight bounce slots carry them onto the wire.
        // 7,549,747,200 + 393,216 + 19,660,800 = 7,569,801,216.
        assert_eq!(mapped.host_pool_bytes(), 7_549_747_200);
        assert_eq!(mapped.table_staging_bytes(), 393_216);
        assert_eq!(mapped.bounce_slot_count(), 8);
        assert_eq!(mapped.bounce_slot_bytes(), 2_457_600);
        assert_eq!(mapped.bounce_ring_bytes(), 19_660_800);
        assert_eq!(mapped.host_pinned_bytes(), 7_569_801_216);
        assert_eq!(mapped.host_mapped_bytes(), 60_397_977_600);

        // Neither posture loses or duplicates a byte of the inventory: what
        // leaves the pinned class is exactly what enters the mapped one.
        assert_eq!(
            mapped.host_pool_bytes() + mapped.host_mapped_bytes(),
            pinned.host_pool_bytes()
        );
        assert_eq!(
            pinned.host_pinned_bytes() - mapped.host_pinned_bytes(),
            60_378_316_800
        );
        // The 63.28 GiB pinned pool against 59.2 GiB of usable host RAM.
        assert!(pinned.host_pinned_bytes() > 63 * (1u64 << 30) as usize);
        assert!(mapped.host_pinned_bytes() < 8 * (1u64 << 30) as usize);
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090 and TUISKO_STREAMING_BIG_POOL"]
    fn streaming_weight_pool_suite_benchmark_big_pool_reports_pinning_and_upload_rate()
    -> Result<(), Box<dyn std::error::Error>> {
        let Some(requested) = std::env::var_os(BIG_POOL_ENVIRONMENT) else {
            println!(
                "SKIPPED streaming weight pool big-pool report: set {BIG_POOL_ENVIRONMENT}=<GiB> to admit it"
            );
            return Ok(());
        };
        let gibibytes = requested
            .to_string_lossy()
            .trim()
            .parse::<usize>()
            .unwrap_or(BIG_POOL_DEFAULT_GIB)
            .max(BIG_POOL_DEFAULT_GIB);
        if gibibytes > BIG_POOL_MAX_GIB {
            return Err(format!(
                "{BIG_POOL_ENVIRONMENT}={gibibytes} exceeds {BIG_POOL_MAX_GIB} GiB; pinned pages evict this box's shared page cache"
            )
            .into());
        }

        let _preflight = device_benchmark::preflight()?;
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err("streaming weight pool report requires compute capability 12.0".into());
        }
        let stride =
            QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES + QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES;
        let items = (gibibytes << 30) / stride;
        // Sixteen slots keep the device side small: this report measures the
        // host pinning cost and the achieved upload rate, not cache behavior.
        let slots = 16.min(items);
        let layout = StreamingWeightLayout::build(
            items,
            QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES,
            Some(QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES),
            slots,
        )?;
        let mut pool = StreamingWeightPool::new(&context, layout)?;
        let pinned = pool.host_pin_duration();

        let primary = vec![0xa5u8; QWEN38_FLASH_NEXT_PRIMARY_EXTENT_BYTES];
        let secondary = vec![0x5au8; QWEN38_FLASH_NEXT_SECONDARY_EXTENT_BYTES];
        let staging = Instant::now();
        for item in 0..items {
            pool.stage_item(item, &primary, &secondary)?;
        }
        let staged = staging.elapsed();

        // A full-miss sweep: every round after the first evicts, so the rate is
        // measured against real slot traffic rather than a warm cache.
        let requests = (0..items as u32).collect::<Vec<_>>();
        let sweep = Instant::now();
        for chunk in requests.chunks(slots) {
            pool.require(chunk)?;
        }
        let swept = sweep.elapsed();

        device_benchmark::require_current_process_exclusive()?;
        println!(
            "streaming weight pool big-pool report: {gibibytes} GiB requested, {items} items x {stride} B, {} pinned bytes in {:.3} s ({:.3} s/GiB), staged in {:.3} s, swept {} bytes in {:.3} s ({:.1} GB/s)",
            pool.host_pinned_bytes(),
            pinned.as_secs_f64(),
            seconds_per_gibibyte(pool.host_pinned_bytes(), pinned),
            staged.as_secs_f64(),
            pool.uploaded_bytes(),
            swept.as_secs_f64(),
            gigabytes_per_second(pool.uploaded_bytes(), swept),
        );
        println!(
            "streaming weight pool big-pool report is diagnostic only and never blesses a performance baseline"
        );

        Ok(())
    }
}

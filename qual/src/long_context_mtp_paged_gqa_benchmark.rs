//! Production-shape timings for exact-K long-context target-MTP GQA.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_PARTITION_BUCKETS,
    LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextMtpPagedGqaOp,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_K: usize = 4;
const ALIGNMENT: usize = 256;
const CONTEXT_TOKENS: usize = 182_111;
const LENGTHS: [u32; MAX_K] = [182_108, 182_109, 182_110, 182_111];
const PHYSICAL_PAGES: usize = CONTEXT_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const QUERY_VALUES: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_CODES: [u8; 8] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20];
const VALUE_CODES: [u8; 8] = [0x38, 0xb8, 0x30, 0xb0, 0x28, 0xa8, 0x20, 0xa0];

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_table: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
    partial_maximum: ArenaRegion<f32>,
    partial_denominator: ArenaRegion<f32>,
    partial_numerator: ArenaRegion<f32>,
    output: ArenaRegion<f32>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.block_table.byte_len()
            + self.table_rows.byte_len()
            + self.lengths.byte_len()
            + self.partial_maximum.byte_len()
            + self.partial_denominator.byte_len()
            + self.partial_numerator.byte_len()
            + self.output.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }

    fn partition_workspace_bytes(self) -> usize {
        self.partial_maximum.byte_len()
            + self.partial_denominator.byte_len()
            + self.partial_numerator.byte_len()
    }
}

struct Addresses {
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_table: *const u32,
    table_rows: *const u32,
    lengths: *const u32,
    partial_maximum: *mut f32,
    partial_denominator: *mut f32,
    partial_numerator: *mut f32,
    output: *mut f32,
}

struct RouteGraphs {
    tokens: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: LongContextMtpPagedGqaOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        load_fixture(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = LongContextMtpPagedGqaOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = (2..=MAX_K)
            .map(|tokens| capture_route(&op, &stream, &addresses, tokens, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _op: op,
            arena,
            regions,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: Session owns the graph and every captured address.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let mut workload = BenchmarkWorkload::warm_operator_decode(route.tokens as u32);
                workload.context_tokens = Some(u64::from(LENGTHS[route.tokens - 1]));
                ExactDeviceCase::new(
                    "long_context_mtp_paged_gqa/represented_kv_tile_reuse",
                    format!("K={}", route.tokens),
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.tokens),
                        route.tokens as u64,
                        "provisional_row",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn scalar_elements() -> usize {
    MAX_K * Qwen38_27B::NUM_ATTENTION_HEADS * LONG_CONTEXT_GQA_MAX_PARTITIONS
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_K * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_table = layout.reserve(PHYSICAL_PAGES, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_K, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_K, ALIGNMENT)?;
    let partial_maximum = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_denominator = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_numerator = layout.reserve(scalar_elements() * Qwen38_27B::HEAD_DIM, ALIGNMENT)?;
    let output = layout.reserve(MAX_K * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_table,
            table_rows,
            lengths,
            partial_maximum,
            partial_denominator,
            partial_numerator,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let query = (0..MAX_K * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_VALUES[(index + index / Qwen38_27B::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = (0..plane_bytes)
        .map(|index| KEY_CODES[(index + index / Qwen38_27B::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let value_pages = (0..plane_bytes)
        .map(|index| VALUE_CODES[(index * 3 + index / Qwen38_27B::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let block_table = (0..PHYSICAL_PAGES)
        .map(|page| u32::try_from(page).expect("physical page fits u32"))
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_table, &block_table)?;
    arena.copy_from_host(stream, regions.table_rows, &[0u32; MAX_K])?;
    arena.copy_from_host(stream, regions.lengths, &LENGTHS)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_table: arena.address(regions.block_table)?,
        table_rows: arena.address(regions.table_rows)?,
        lengths: arena.address(regions.lengths)?,
        partial_maximum: arena.address(regions.partial_maximum)?,
        partial_denominator: arena.address(regions.partial_denominator)?,
        partial_numerator: arena.address(regions.partial_numerator)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &LongContextMtpPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    tokens: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, tokens))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, tokens)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        tokens,
        leaf,
        repeated,
    })
}

fn launch(
    op: &LongContextMtpPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: exact K rows select one complete table row and consecutive lengths.
    unsafe {
        op.launch(
            stream,
            tokens,
            LENGTHS[tokens - 1] as usize,
            addresses.query,
            addresses.key_pages,
            addresses.value_pages,
            addresses.block_table,
            addresses.table_rows,
            PHYSICAL_PAGES,
            addresses.lengths,
            addresses.partial_maximum,
            addresses.partial_denominator,
            addresses.partial_numerator,
            addresses.output,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn logical_bytes(tokens: usize) -> usize {
    let maximum_length = LENGTHS[tokens - 1] as usize;
    let active_partitions = maximum_length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
    let launched_partitions = LONG_CONTEXT_GQA_PARTITION_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| bucket >= active_partitions)
        .expect("admitted context has a partition bucket");
    let row_partials = LENGTHS[..tokens]
        .iter()
        .map(|&length| (length as usize).div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE))
        .sum::<usize>()
        * Qwen38_27B::NUM_ATTENTION_HEADS;
    let query = row_partials * Qwen38_27B::HEAD_DIM * size_of::<f32>();
    let represented_cache = 2
        * Qwen38_27B::NUM_KV_HEADS
        * maximum_length.div_ceil(ATTENTION_PAGE_SIZE)
        * ATTENTION_PAGE_SIZE
        * Qwen38_27B::HEAD_DIM;
    let metadata = Qwen38_27B::NUM_KV_HEADS
        * (launched_partitions * tokens + active_partitions)
        * size_of::<u32>();
    let partial = row_partials
        * (2 * size_of::<f32>()
            + Qwen38_27B::HEAD_DIM * size_of::<f32>()
            + 3 * size_of::<f32>()
            + Qwen38_27B::HEAD_DIM * size_of::<f32>());
    let output = tokens * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();

    query + represented_cache + metadata + partial + output
}

/// Measures exact `K=2..4` with the production 182K represented-cache geometry.
pub fn benchmark_long_context_mtp_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    let cache_bytes = session.regions.cache_bytes();
    let partition_workspace_bytes = session.regions.partition_workspace_bytes();
    let boundary_workspace_bytes =
        session.arena.byte_len() - cache_bytes - partition_workspace_bytes - padding_bytes;
    memory.register_owned(
        "long_context_mtp_paged_gqa/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "2,846 physical pages, four KV heads, 64 positions, represented E4M3 K/V",
    )?;
    memory.register_owned(
        "long_context_mtp_paged_gqa/partition_workspace",
        BenchmarkMemoryKind::Workspace,
        partition_workspace_bytes,
        "K=4, 24 query heads, 860 maximum/denominator/numerator FP32 partials",
    )?;
    memory.register_owned(
        "long_context_mtp_paged_gqa/address_stable_boundary",
        BenchmarkMemoryKind::Workspace,
        boundary_workspace_bytes,
        "K=4 query/output and one complete page table",
    )?;
    memory.register_owned(
        "long_context_mtp_paged_gqa/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-long-context-mtp-paged-gqa",
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production two-stage graph, and repeated-operation graph at exact K=2..4 and 182K context",
        },
        preflight,
        baseline_sha256,
        options,
        metrics,
        energy_metrics,
        telemetry,
        memory,
    )
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_TOKENS, MAX_K, PHYSICAL_PAGES, layout, logical_bytes, scalar_elements};
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;

    #[test]
    fn accounting_exposes_the_kv_reuse_boundary() {
        let k4_cache = 2 * 4 * PHYSICAL_PAGES * ATTENTION_PAGE_SIZE * 256;
        let generic_k4_cache = 2 * MAX_K * 24 * CONTEXT_TOKENS * 256;

        assert!(logical_bytes(2) < logical_bytes(3));
        assert!(logical_bytes(3) < logical_bytes(4));
        assert!(k4_cache * 23 < generic_k4_cache);
        assert!(generic_k4_cache < k4_cache * 25);
    }

    #[test]
    fn arena_accounting_covers_every_production_owner() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(CONTEXT_TOKENS, 182_111);
        assert_eq!(PHYSICAL_PAGES, 2_846);
        assert_eq!(scalar_elements(), 82_560);
        assert_eq!(regions.cache_bytes(), 373_030_912);
        assert_eq!(regions.partition_workspace_bytes(), 85_201_920);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 616);
    }
}

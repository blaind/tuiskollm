//! Direct timings for exact represented-BF16 paged-GQA graphs.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use std::{mem::size_of, sync::Arc};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, MtpBf16PagedGqaOp, Qwen35PagedGqaOp, Qwen36PagedGqaOp,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 24;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const CONTEXT_TOKENS: usize = 130;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const LENGTHS: [u32; MAX_BATCH] = [CONTEXT_TOKENS as u32; MAX_BATCH];
const BLOCK_TABLES: [u32; TABLE_ROWS * TABLE_STRIDE] = [
    17, 2, 21, 4, 15, 0, 23, 7, 12, 1, 18, 9, 14, 5, 22, 8, 19, 3, 20, 6, 13, 10, 16, 11,
];
const QUERY_VALUES: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_VALUES: [f32; 8] = [
    0.0, 0.03125, 0.0625, 0.125, -0.03125, -0.0625, -0.125, 0.015625,
];
const VALUE_VALUES: [f32; 8] = [
    0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125, 0.015625, -0.015625,
];

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u16>,
    value_pages: ArenaRegion<u16>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
    output: ArenaRegion<f32>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.block_tables.byte_len()
            + self.table_rows.byte_len()
            + self.lengths.byte_len()
            + self.output.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }
}

struct Addresses {
    query: *const f32,
    key_pages: *const u16,
    value_pages: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    lengths: *const u32,
    output: *mut f32,
}

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

trait BenchPagedGqaOp {
    type Target: Arch;
    const ROUTE: &'static str;
    const SUITE: &'static str;
    const CACHE_OWNER: &'static str;
    const WORKSPACE_OWNER: &'static str;
    const PADDING_OWNER: &'static str;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self>
    where
        Self: Sized;
    fn workload(batch: usize) -> BenchmarkWorkload;
    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()>;
}

impl BenchPagedGqaOp for Qwen35PagedGqaOp {
    type Target = Qwen35_9B;
    const ROUTE: &'static str = "qwen35_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-qwen35-paged-gqa";
    const CACHE_OWNER: &'static str = "qwen35_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "qwen35_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "qwen35_paged_gqa/alignment_padding";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen35PagedGqaOp::new(context)
    }

    fn workload(batch: usize) -> BenchmarkWorkload {
        BenchmarkWorkload::warm_operator_decode(batch as u32)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables,
                addresses.table_rows,
                TABLE_STRIDE,
                addresses.lengths,
                addresses.output,
            )
        }
    }
}

impl BenchPagedGqaOp for Qwen36PagedGqaOp {
    type Target = Qwen36Moe35B;
    const ROUTE: &'static str = "qwen36_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-qwen36-paged-gqa";
    const CACHE_OWNER: &'static str = "qwen36_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "qwen36_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "qwen36_paged_gqa/alignment_padding";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen36PagedGqaOp::new(context)
    }

    fn workload(batch: usize) -> BenchmarkWorkload {
        BenchmarkWorkload::warm_operator_decode(batch as u32)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables,
                addresses.table_rows,
                TABLE_STRIDE,
                addresses.lengths,
                addresses.output,
            )
        }
    }
}

impl BenchPagedGqaOp for MtpBf16PagedGqaOp {
    type Target = Qwen38_27B;
    const ROUTE: &'static str = "mtp_bf16_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-mtp-bf16-paged-gqa";
    const CACHE_OWNER: &'static str = "mtp_bf16_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "mtp_bf16_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "mtp_bf16_paged_gqa/alignment_padding";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        MtpBf16PagedGqaOp::new(context)
    }

    fn workload(batch: usize) -> BenchmarkWorkload {
        BenchmarkWorkload::warm_operator_mtp(batch as u64)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables,
                addresses.table_rows,
                TABLE_STRIDE,
                addresses.lengths,
                addresses.output,
            )
        }
    }
}

struct Session<O> {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: O,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl<O: BenchPagedGqaOp> Session<O> {
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
        let (layout, regions) = layout::<O::Target>()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        load_fixture::<O::Target>(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = O::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| capture_route(&op, &stream, &addresses, batch, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;
        let timer = GpuTimer::new(&context)?;

        Ok(Self {
            routes,
            timer,
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
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let mut workload = O::workload(route.batch);
                workload.context_tokens = Some(CONTEXT_TOKENS as u64);
                ExactDeviceCase::new(
                    O::ROUTE,
                    format!("B={}", route.batch),
                    workload,
                    OperationAccounting::new(
                        logical_bytes::<O::Target>(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn layout<A: Arch>() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_elements = PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM;
    let key_pages = layout.reserve(plane_elements, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_elements, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            output,
        },
    ))
}

fn load_fixture<A: Arch>(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    let query = (0..MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_VALUES[(index + index / A::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let plane_elements = PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM;
    let key_pages = (0..plane_elements)
        .map(|index| f32_to_bf16(KEY_VALUES[(index + index / A::HEAD_DIM) & 7]))
        .collect::<Vec<_>>();
    let value_pages = (0..plane_elements)
        .map(|index| f32_to_bf16(VALUE_VALUES[(index * 3 + index / A::HEAD_DIM) & 7]))
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &BLOCK_TABLES)?;
    arena.copy_from_host(stream, regions.table_rows, &TABLE_ROW_IDS)?;
    arena.copy_from_host(stream, regions.lengths, &LENGTHS)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_tables: arena.address(regions.block_tables)?,
        table_rows: arena.address(regions.table_rows)?,
        lengths: arena.address(regions.lengths)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route<O: BenchPagedGqaOp>(
    op: &O,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, batch))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, batch)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        batch,
        leaf,
        repeated,
    })
}

fn launch<O: BenchPagedGqaOp>(
    op: &O,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    op.launch(stream, batch, addresses)
}

fn logical_bytes<A: Arch>(batch: usize) -> usize {
    let query = A::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let output = A::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let scanned_positions = batch * CONTEXT_TOKENS;
    let cache = 2 * A::NUM_ATTENTION_HEADS * scanned_positions * A::HEAD_DIM * size_of::<u16>();
    let metadata = 2 * batch * size_of::<u32>()
        + A::NUM_ATTENTION_HEADS * scanned_positions * size_of::<u32>();

    batch * (query + output) + cache + metadata
}

fn benchmark_target<O: BenchPagedGqaOp>(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::<O>::new(options.launches_per_sample)?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    let cache_bytes = session.regions.cache_bytes();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - padding_bytes;
    memory.register_owned(
        O::CACHE_OWNER,
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "24 physical pages, four KV heads, 64 positions, represented BF16 K/V",
    )?;
    memory.register_owned(
        O::WORKSPACE_OWNER,
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "B=8 query/output plus page-table, row, and length metadata",
    )?;
    memory.register_owned(
        O::PADDING_OWNER,
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
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: O::SUITE,
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph at a 130-token decode context",
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

/// Measures every exact Qwen3.5 BF16 paged-GQA graph at 130 context tokens.
pub fn benchmark_qwen35_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen35PagedGqaOp>(options)
}

/// Measures every exact Qwen3.6 represented-BF16 paged-GQA batch.
pub fn benchmark_qwen36_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen36PagedGqaOp>(options)
}

/// Measures every exact Qwen3.8 MTP BF16 paged-GQA graph at 130 context tokens.
pub fn benchmark_mtp_bf16_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<MtpBf16PagedGqaOp>(options)
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_TOKENS, MAX_BATCH, layout, logical_bytes};
    use std::mem::size_of;
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

    #[test]
    fn qwen35_bf16_byte_accounting_covers_every_query_head_cache_read() {
        let per_token = 2 * Qwen35_9B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen35_9B::NUM_ATTENTION_HEADS
                * CONTEXT_TOKENS
                * Qwen35_9B::HEAD_DIM
                * size_of::<u16>()
            + 2 * size_of::<u32>()
            + Qwen35_9B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();

        assert_eq!(logical_bytes::<Qwen35_9B>(1), per_token);
        assert_eq!(logical_bytes::<Qwen35_9B>(MAX_BATCH), MAX_BATCH * per_token);
    }

    #[test]
    fn qwen35_bf16_arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout::<Qwen35_9B>().unwrap();

        assert_eq!(regions.cache_bytes(), 6_291_456);
        assert_eq!(regions.payload_bytes(), 6_553_760);
        assert_eq!(layout.byte_len(), 6_554_368);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }

    #[test]
    fn qwen36_bf16_byte_and_arena_accounting_cover_the_two_head_cache() {
        let per_token = 2 * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen36Moe35B::NUM_ATTENTION_HEADS
                * CONTEXT_TOKENS
                * Qwen36Moe35B::HEAD_DIM
                * size_of::<u16>()
            + 2 * size_of::<u32>()
            + Qwen36Moe35B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();

        assert_eq!(logical_bytes::<Qwen36Moe35B>(1), per_token);
        assert_eq!(
            logical_bytes::<Qwen36Moe35B>(MAX_BATCH),
            MAX_BATCH * per_token
        );

        let (layout, regions) = layout::<Qwen36Moe35B>().unwrap();
        assert_eq!(regions.cache_bytes(), 3_145_728);
        assert_eq!(regions.payload_bytes(), 3_408_032);
        assert_eq!(layout.byte_len(), 3_408_640);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }

    #[test]
    fn mtp_bf16_paged_gqa_byte_accounting_covers_every_query_head_cache_read() {
        let per_token = 2 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen38_27B::NUM_ATTENTION_HEADS
                * CONTEXT_TOKENS
                * Qwen38_27B::HEAD_DIM
                * size_of::<u16>()
            + 2 * size_of::<u32>()
            + Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();

        assert_eq!(logical_bytes::<Qwen38_27B>(1), per_token);
        assert_eq!(
            logical_bytes::<Qwen38_27B>(MAX_BATCH),
            MAX_BATCH * per_token
        );
    }

    #[test]
    fn mtp_bf16_paged_gqa_arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout::<Qwen38_27B>().unwrap();

        assert_eq!(regions.cache_bytes(), 6_291_456);
        assert_eq!(regions.payload_bytes(), 6_684_832);
        assert_eq!(layout.byte_len(), 6_685_440);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }
}

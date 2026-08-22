//! Direct timings for exact short-context paged GQA graph routes.

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
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, PagedGqaOp};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 24;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const CONTEXT_TOKENS: usize = 130;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const LENGTHS: [u32; MAX_BATCH] = [CONTEXT_TOKENS as u32; MAX_BATCH];
const BLOCK_TABLES: [u32; TABLE_ROWS * TABLE_STRIDE] = [
    17, 2, 21, 4, 15, 0, 23, 7, 12, 1, 18, 9, 14, 5, 22, 8, 19, 3, 20, 6, 13, 10, 16, 11,
];
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
    key_pages: *const u8,
    value_pages: *const u8,
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

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: PagedGqaOp,
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
        let op = PagedGqaOp::new(&context)?;
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
                route.leaf.launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let mut workload = BenchmarkWorkload::warm_operator_decode(route.batch as u32);
                workload.context_tokens = Some(CONTEXT_TOKENS as u64);
                ExactDeviceCase::new(
                    "paged_gqa/online_softmax_e4m3_kv",
                    format!("B={}", route.batch),
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.batch),
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

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

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

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let query = (0..MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
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

fn capture_route(
    op: &PagedGqaOp,
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

fn launch(
    op: &PagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: the benchmark owns all 24 physical pages and each row's three
    // table entries cover the fixed 130-token context.
    unsafe {
        op.launch(
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
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let cache = 2 * Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * Qwen38_27B::HEAD_DIM;
    let page_metadata =
        2 * size_of::<u32>() + Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();

    batch * (query + cache + page_metadata + output)
}

/// Measures every exact paged GQA batch at a 130-token, three-page context.
pub fn benchmark_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    let cache_bytes = session.regions.cache_bytes();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - padding_bytes;
    memory.register_owned(
        "paged_gqa/kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "24 physical pages, four KV heads, 64 positions, represented E4M3 K/V",
    )?;
    memory.register_owned(
        "paged_gqa/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_batch=8 query/output and three-page metadata",
    )?;
    memory.register_owned(
        "paged_gqa/alignment_padding",
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
            suite: "bench-paged-gqa",
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph at context=130",
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
    use super::{CONTEXT_TOKENS, MAX_BATCH, layout, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_every_query_head_cache_scan() {
        let per_token = 2 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * 4
            + 2 * Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * Qwen38_27B::HEAD_DIM
            + 2 * 4
            + Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * 4;

        assert_eq!(logical_bytes(1), per_token);
        assert_eq!(logical_bytes(MAX_BATCH), MAX_BATCH * per_token);
    }

    #[test]
    fn arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(layout.byte_len(), 3_539_712);
        assert_eq!(regions.payload_bytes(), 3_539_104);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }
}

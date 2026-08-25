//! Direct timings for exact represented-cache paged-GQA graphs.

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
    ATTENTION_PAGE_SIZE, MtpBf16PagedGqaOp, Qwen35PagedGqaOp, Qwen36Fp8PagedGqaOp, Qwen36PagedGqaOp,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const DECODE_ROUTES: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];
const QWEN35_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const QWEN35_MAX_TOKENS: usize = 128;
const QWEN36_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const QWEN36_MAX_TOKENS: usize = 128;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 24;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const CONTEXT_TOKENS: usize = 130;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
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

trait BenchPagedGqaOp {
    type Target: Arch;
    const CACHE_ELEMENT_BYTES: usize;
    const ROUTES: &'static [usize];
    const MAX_TOKENS: usize;
    const ROUTE: &'static str;
    const SUITE: &'static str;
    const CACHE_OWNER: &'static str;
    const WORKSPACE_OWNER: &'static str;
    const PADDING_OWNER: &'static str;
    const CACHE_DESCRIPTION: &'static str;
    const WORKSPACE_DESCRIPTION: &'static str;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self>
    where
        Self: Sized;
    fn workload(batch: usize) -> BenchmarkWorkload;
    fn cache_fixture(plane_elements: usize) -> (Vec<u8>, Vec<u8>);
    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()>;
}

impl BenchPagedGqaOp for Qwen35PagedGqaOp {
    type Target = Qwen35_9B;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const ROUTES: &'static [usize] = &QWEN35_ROUTES;
    const MAX_TOKENS: usize = QWEN35_MAX_TOKENS;
    const ROUTE: &'static str = "qwen35_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-qwen35-paged-gqa";
    const CACHE_OWNER: &'static str = "qwen35_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "qwen35_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "qwen35_paged_gqa/alignment_padding";
    const CACHE_DESCRIPTION: &'static str =
        "24 physical pages, four KV heads, 64 positions, represented BF16 K/V";
    const WORKSPACE_DESCRIPTION: &'static str =
        "T=128 query/output plus page-table, row, and length metadata";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen35PagedGqaOp::new(context)
    }

    fn workload(tokens: usize) -> BenchmarkWorkload {
        if tokens <= MAX_BATCH {
            BenchmarkWorkload::warm_operator_decode(tokens as u32)
        } else {
            BenchmarkWorkload::warm_operator_prefill(tokens as u64)
        }
    }

    fn cache_fixture(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
        bf16_cache_fixture::<Self::Target>(plane_elements)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages.cast(),
                addresses.value_pages.cast(),
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
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const ROUTES: &'static [usize] = &QWEN36_ROUTES;
    const MAX_TOKENS: usize = QWEN36_MAX_TOKENS;
    const ROUTE: &'static str = "qwen36_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-qwen36-paged-gqa";
    const CACHE_OWNER: &'static str = "qwen36_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "qwen36_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "qwen36_paged_gqa/alignment_padding";
    const CACHE_DESCRIPTION: &'static str =
        "24 physical pages, two KV heads, 64 positions, represented BF16 K/V";
    const WORKSPACE_DESCRIPTION: &'static str =
        "T=128 query/output plus page-table, row, and length metadata";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen36PagedGqaOp::new(context)
    }

    fn workload(tokens: usize) -> BenchmarkWorkload {
        if tokens <= MAX_BATCH {
            BenchmarkWorkload::warm_operator_decode(tokens as u32)
        } else {
            BenchmarkWorkload::warm_operator_prefill(tokens as u64)
        }
    }

    fn cache_fixture(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
        bf16_cache_fixture::<Self::Target>(plane_elements)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages.cast(),
                addresses.value_pages.cast(),
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
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const ROUTES: &'static [usize] = &DECODE_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
    const ROUTE: &'static str = "mtp_bf16_paged_gqa/online_softmax_bf16_kv";
    const SUITE: &'static str = "bench-mtp-bf16-paged-gqa";
    const CACHE_OWNER: &'static str = "mtp_bf16_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "mtp_bf16_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "mtp_bf16_paged_gqa/alignment_padding";
    const CACHE_DESCRIPTION: &'static str =
        "24 physical pages, four KV heads, 64 positions, source-native BF16 K/V";
    const WORKSPACE_DESCRIPTION: &'static str =
        "B=8 MTP query/output plus page-table, row, and length metadata";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        MtpBf16PagedGqaOp::new(context)
    }

    fn workload(batch: usize) -> BenchmarkWorkload {
        BenchmarkWorkload::warm_operator_mtp(batch as u64)
    }

    fn cache_fixture(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
        bf16_cache_fixture::<Self::Target>(plane_elements)
    }

    fn launch(&self, stream: &CudaStream, batch: usize, addresses: &Addresses) -> GpuResult<()> {
        // SAFETY: the session owns all 24 pages and metadata rows for the
        // 130-position context throughout every captured graph replay.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages.cast(),
                addresses.value_pages.cast(),
                addresses.block_tables,
                addresses.table_rows,
                TABLE_STRIDE,
                addresses.lengths,
                addresses.output,
            )
        }
    }
}

impl BenchPagedGqaOp for Qwen36Fp8PagedGqaOp {
    type Target = Qwen36Moe35B;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u8>();
    const ROUTES: &'static [usize] = &QWEN36_ROUTES;
    const MAX_TOKENS: usize = QWEN36_MAX_TOKENS;
    const ROUTE: &'static str = "qwen36_fp8_paged_gqa/online_softmax_e4m3_kv";
    const SUITE: &'static str = "bench-qwen36-fp8-paged-gqa";
    const CACHE_OWNER: &'static str = "qwen36_fp8_paged_gqa/kv_cache";
    const WORKSPACE_OWNER: &'static str = "qwen36_fp8_paged_gqa/address_stable_workspace";
    const PADDING_OWNER: &'static str = "qwen36_fp8_paged_gqa/alignment_padding";
    const CACHE_DESCRIPTION: &'static str =
        "24 physical pages, two KV heads, 64 positions, represented E4M3 K/V";
    const WORKSPACE_DESCRIPTION: &'static str =
        "T=128 query/output plus page-table, row, and length metadata";

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen36Fp8PagedGqaOp::new(context)
    }

    fn workload(tokens: usize) -> BenchmarkWorkload {
        if tokens <= MAX_BATCH {
            BenchmarkWorkload::warm_operator_decode(tokens as u32)
        } else {
            BenchmarkWorkload::warm_operator_prefill(tokens as u64)
        }
    }

    fn cache_fixture(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
        fp8_cache_fixture::<Self::Target>(plane_elements)
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
                KEY_SCALE,
                VALUE_SCALE,
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
        let (layout, regions) = layout::<O::Target>(O::MAX_TOKENS, O::CACHE_ELEMENT_BYTES)?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        load_fixture::<O>(&arena, &stream, regions, O::MAX_TOKENS)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = O::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = O::ROUTES
            .iter()
            .map(|&tokens| capture_route(&op, &stream, &addresses, tokens, repeated_operations))
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
        for route in &self.routes {
            // SAFETY: this Session owns the repeated route graph and every captured
            // allocation until after this synchronized replay.
            unsafe { route.repeated.launch(&self.stream) }?;
        }
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
                    if route.batch <= MAX_BATCH {
                        format!("B={}", route.batch)
                    } else {
                        format!("T={}", route.batch)
                    },
                    workload,
                    OperationAccounting::new(
                        logical_bytes::<O::Target>(route.batch, O::CACHE_ELEMENT_BYTES),
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

fn layout<A: Arch>(
    max_tokens: usize,
    cache_element_bytes: usize,
) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_elements = PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM;
    let key_pages = layout.reserve(plane_elements * cache_element_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_elements * cache_element_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(max_tokens, ALIGNMENT)?;
    let lengths = layout.reserve(max_tokens, ALIGNMENT)?;
    let output = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

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

fn load_fixture<O: BenchPagedGqaOp>(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    max_tokens: usize,
) -> GpuResult<()> {
    let query = (0..max_tokens * O::Target::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_VALUES[(index + index / O::Target::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let plane_elements =
        PHYSICAL_PAGES * O::Target::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * O::Target::HEAD_DIM;
    let (key_pages, value_pages) = O::cache_fixture(plane_elements);

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &BLOCK_TABLES)?;
    let table_rows = (0..max_tokens)
        .map(|token| TABLE_ROW_IDS[token % MAX_BATCH])
        .collect::<Vec<_>>();
    let lengths = vec![CONTEXT_TOKENS as u32; max_tokens];
    arena.copy_from_host(stream, regions.table_rows, &table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &lengths)
}

fn bf16_cache_fixture<A: Arch>(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
    let key_pages = (0..plane_elements)
        .flat_map(|index| f32_to_bf16(KEY_VALUES[(index + index / A::HEAD_DIM) & 7]).to_le_bytes())
        .collect();
    let value_pages = (0..plane_elements)
        .flat_map(|index| {
            f32_to_bf16(VALUE_VALUES[(index * 3 + index / A::HEAD_DIM) & 7]).to_le_bytes()
        })
        .collect();
    (key_pages, value_pages)
}

fn fp8_cache_fixture<A: Arch>(plane_elements: usize) -> (Vec<u8>, Vec<u8>) {
    let key_pages = (0..plane_elements)
        .map(|index| KEY_CODES[(index + index / A::HEAD_DIM) & 7])
        .collect();
    let value_pages = (0..plane_elements)
        .map(|index| VALUE_CODES[(index * 3 + index / A::HEAD_DIM) & 7])
        .collect();
    (key_pages, value_pages)
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

fn logical_bytes<A: Arch>(batch: usize, cache_element_bytes: usize) -> usize {
    let query = A::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let output = A::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let scanned_positions = batch * CONTEXT_TOKENS;
    let cache_heads = if batch <= MAX_BATCH {
        A::NUM_ATTENTION_HEADS
    } else {
        A::NUM_KV_HEADS
    };
    let cache = 2 * cache_heads * scanned_positions * A::HEAD_DIM * cache_element_bytes;
    let metadata =
        2 * batch * size_of::<u32>() + cache_heads * scanned_positions * size_of::<u32>();

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
        O::CACHE_DESCRIPTION,
    )?;
    memory.register_owned(
        O::WORKSPACE_OWNER,
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        O::WORKSPACE_DESCRIPTION,
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
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph at a 130-token context",
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

/// Measures every exact Qwen3.5 represented-BF16 paged-GQA decode and prompt route.
pub fn benchmark_qwen35_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen35PagedGqaOp>(options)
}

/// Measures every exact Qwen3.6 represented-BF16 paged-GQA decode and prompt route.
pub fn benchmark_qwen36_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen36PagedGqaOp>(options)
}

/// Measures every exact Qwen3.6 represented-E4M3 paged-GQA decode and prompt route.
pub fn benchmark_qwen36_fp8_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen36Fp8PagedGqaOp>(options)
}

/// Measures every exact Qwen3.8 MTP BF16 paged-GQA graph at 130 context tokens.
pub fn benchmark_mtp_bf16_paged_gqa(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<MtpBf16PagedGqaOp>(options)
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_TOKENS, MAX_BATCH, QWEN35_MAX_TOKENS, QWEN35_ROUTES, QWEN36_MAX_TOKENS,
        QWEN36_ROUTES, layout, logical_bytes,
    };
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

        assert_eq!(logical_bytes::<Qwen35_9B>(1, size_of::<u16>()), per_token);
        assert_eq!(
            logical_bytes::<Qwen35_9B>(MAX_BATCH, size_of::<u16>()),
            MAX_BATCH * per_token
        );

        let prompt_per_token = 2 * Qwen35_9B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen35_9B::NUM_KV_HEADS * CONTEXT_TOKENS * Qwen35_9B::HEAD_DIM * size_of::<u16>()
            + 2 * size_of::<u32>()
            + Qwen35_9B::NUM_KV_HEADS * CONTEXT_TOKENS * size_of::<u32>();
        for tokens in [32, 64, 128] {
            assert_eq!(
                logical_bytes::<Qwen35_9B>(tokens, size_of::<u16>()),
                tokens * prompt_per_token
            );
        }
        assert_eq!(QWEN35_ROUTES.len(), 11);
    }

    #[test]
    fn qwen35_bf16_arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout::<Qwen35_9B>(QWEN35_MAX_TOKENS, size_of::<u16>()).unwrap();

        assert_eq!(regions.cache_bytes(), 6_291_456);
        assert_eq!(regions.payload_bytes(), 10_486_880);
        assert_eq!(layout.byte_len(), 10_487_040);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 160);
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

        assert_eq!(
            logical_bytes::<Qwen36Moe35B>(1, size_of::<u16>()),
            per_token
        );
        assert_eq!(
            logical_bytes::<Qwen36Moe35B>(MAX_BATCH, size_of::<u16>()),
            MAX_BATCH * per_token
        );
        let prompt_per_token = 2 * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen36Moe35B::NUM_KV_HEADS
                * CONTEXT_TOKENS
                * Qwen36Moe35B::HEAD_DIM
                * size_of::<u16>()
            + 2 * size_of::<u32>()
            + Qwen36Moe35B::NUM_KV_HEADS * CONTEXT_TOKENS * size_of::<u32>();
        for tokens in [32, 64, 128] {
            assert_eq!(
                logical_bytes::<Qwen36Moe35B>(tokens, size_of::<u16>()),
                tokens * prompt_per_token
            );
        }
        assert_eq!(QWEN36_ROUTES.len(), 11);

        let (layout, regions) =
            layout::<Qwen36Moe35B>(QWEN36_MAX_TOKENS, size_of::<u16>()).unwrap();
        assert_eq!(regions.cache_bytes(), 3_145_728);
        assert_eq!(regions.payload_bytes(), 7_341_152);
        assert_eq!(layout.byte_len(), 7_341_312);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 160);
    }

    #[test]
    fn qwen36_fp8_byte_and_arena_accounting_cover_the_two_head_cache() {
        let per_token = 2 * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen36Moe35B::NUM_ATTENTION_HEADS
                * CONTEXT_TOKENS
                * Qwen36Moe35B::HEAD_DIM
                * size_of::<u8>()
            + 2 * size_of::<u32>()
            + Qwen36Moe35B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>();

        assert_eq!(logical_bytes::<Qwen36Moe35B>(1, size_of::<u8>()), per_token);
        assert_eq!(
            logical_bytes::<Qwen36Moe35B>(MAX_BATCH, size_of::<u8>()),
            MAX_BATCH * per_token
        );
        let prompt_per_token = 2 * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen36Moe35B::NUM_KV_HEADS
                * CONTEXT_TOKENS
                * Qwen36Moe35B::HEAD_DIM
                * size_of::<u8>()
            + 2 * size_of::<u32>()
            + Qwen36Moe35B::NUM_KV_HEADS * CONTEXT_TOKENS * size_of::<u32>();
        for tokens in [32, 64, 128] {
            assert_eq!(
                logical_bytes::<Qwen36Moe35B>(tokens, size_of::<u8>()),
                tokens * prompt_per_token
            );
        }

        let (layout, regions) = layout::<Qwen36Moe35B>(QWEN36_MAX_TOKENS, size_of::<u8>()).unwrap();
        assert_eq!(regions.cache_bytes(), 1_572_864);
        assert_eq!(regions.payload_bytes(), 5_768_288);
        assert_eq!(layout.byte_len(), 5_768_448);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 160);
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

        assert_eq!(logical_bytes::<Qwen38_27B>(1, size_of::<u16>()), per_token);
        assert_eq!(
            logical_bytes::<Qwen38_27B>(MAX_BATCH, size_of::<u16>()),
            MAX_BATCH * per_token
        );
    }

    #[test]
    fn mtp_bf16_paged_gqa_arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout::<Qwen38_27B>(MAX_BATCH, size_of::<u16>()).unwrap();

        assert_eq!(regions.cache_bytes(), 6_291_456);
        assert_eq!(regions.payload_bytes(), 6_684_832);
        assert_eq!(layout.byte_len(), 6_685_440);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }
}

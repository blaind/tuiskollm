//! Direct timings for exact decode and admitted paged-GQA prefill graph routes.

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
    ATTENTION_PAGE_SIZE, PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT,
    PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS, PagedGqaOp,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const TAIL_TOKENS: usize = 128;
const MAX_TOKENS: usize = PAGED_GQA_PREFILL_MACRO_TOKENS;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const PARTITIONED_PREFILL_ROUTES: [(usize, usize); 2] = [
    (PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT - 1, 8),
    (98_304, 16),
];
const MACRO_PREFILL_ROUTES: [(usize, usize); 2] = [(32_768, 4), (98_304, 4)];
const PREFIX_TOKENS: usize = 2;
const ALIGNMENT: usize = 256;
const SHARED_PHYSICAL_PAGES: usize = 24;
const PARTITIONED_PHYSICAL_PAGES: usize = PARTITIONED_PREFILL_ROUTES[1]
    .0
    .div_ceil(ATTENTION_PAGE_SIZE);
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const CONTEXT_TOKENS: usize = 130;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const DECODE_TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const DECODE_LENGTHS: [u32; MAX_BATCH] = [CONTEXT_TOKENS as u32; MAX_BATCH];
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
    decode_table_rows: ArenaRegion<u32>,
    decode_lengths: ArenaRegion<u32>,
    prefill_table_rows: ArenaRegion<u32>,
    prefill_lengths: ArenaRegion<u32>,
    partitioned_block_table: ArenaRegion<u32>,
    partitioned_table_rows: ArenaRegion<u32>,
    partitioned_short_lengths: ArenaRegion<u32>,
    partitioned_long_lengths: ArenaRegion<u32>,
    macro_short_lengths: ArenaRegion<u32>,
    macro_long_lengths: ArenaRegion<u32>,
    partitioned_partials: ArenaRegion<f32>,
    output: ArenaRegion<f32>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.block_tables.byte_len()
            + self.decode_table_rows.byte_len()
            + self.decode_lengths.byte_len()
            + self.prefill_table_rows.byte_len()
            + self.prefill_lengths.byte_len()
            + self.partitioned_block_table.byte_len()
            + self.partitioned_table_rows.byte_len()
            + self.partitioned_short_lengths.byte_len()
            + self.partitioned_long_lengths.byte_len()
            + self.macro_short_lengths.byte_len()
            + self.macro_long_lengths.byte_len()
            + self.partitioned_partials.byte_len()
            + self.output.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }

    fn partial_bytes(self) -> usize {
        self.partitioned_partials.byte_len()
    }
}

struct Addresses {
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    decode_table_rows: *const u32,
    decode_lengths: *const u32,
    prefill_table_rows: *const u32,
    prefill_lengths: *const u32,
    partitioned_block_table: *const u32,
    partitioned_table_rows: *const u32,
    partitioned_short_lengths: *const u32,
    partitioned_long_lengths: *const u32,
    macro_short_lengths: *const u32,
    macro_long_lengths: *const u32,
    partitioned_partials: *mut f32,
    output: *mut f32,
}

#[derive(Clone, Copy)]
enum Route {
    Decode {
        batch: usize,
    },
    SharedPrefill {
        tokens: usize,
    },
    PartitionedPrefill {
        context_tokens: usize,
        partitions: usize,
    },
    MacroPrefill {
        context_tokens: usize,
        partitions: usize,
    },
}

struct RouteGraphs {
    route: Route,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
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
        let routes =
            (1..=MAX_BATCH)
                .map(|batch| Route::Decode { batch })
                .chain(
                    PREFILL_ROUTES
                        .into_iter()
                        .map(|tokens| Route::SharedPrefill { tokens }),
                )
                .chain(PARTITIONED_PREFILL_ROUTES.into_iter().map(
                    |(context_tokens, partitions)| Route::PartitionedPrefill {
                        context_tokens,
                        partitions,
                    },
                ))
                .chain(
                    MACRO_PREFILL_ROUTES
                        .into_iter()
                        .map(|(context_tokens, partitions)| Route::MacroPrefill {
                            context_tokens,
                            partitions,
                        }),
                )
                .map(|route| capture_route(&op, &stream, &addresses, route, repeated_operations))
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
                let (shape, mut workload, context_tokens, logical_bytes, tokens) = match route.route
                {
                    Route::Decode { batch } => (
                        format!("B={batch}"),
                        BenchmarkWorkload::warm_operator_decode(batch as u32),
                        CONTEXT_TOKENS,
                        decode_logical_bytes(batch),
                        batch,
                    ),
                    Route::SharedPrefill { tokens } => (
                        format!("T={tokens}"),
                        BenchmarkWorkload::warm_operator_prefill(tokens as u64),
                        PREFIX_TOKENS + tokens,
                        shared_prefill_logical_bytes(tokens),
                        tokens,
                    ),
                    Route::PartitionedPrefill {
                        context_tokens,
                        partitions,
                    } => (
                        format!("T={TAIL_TOKENS}/P={partitions}/context={context_tokens}"),
                        BenchmarkWorkload::warm_operator_prefill(TAIL_TOKENS as u64),
                        context_tokens,
                        partitioned_prefill_logical_bytes(context_tokens, partitions),
                        TAIL_TOKENS,
                    ),
                    Route::MacroPrefill {
                        context_tokens,
                        partitions,
                    } => (
                        format!("T={MAX_TOKENS}/P={partitions}/context={context_tokens}"),
                        BenchmarkWorkload::warm_operator_prefill(MAX_TOKENS as u64),
                        context_tokens,
                        macro_prefill_logical_bytes(context_tokens, partitions),
                        MAX_TOKENS,
                    ),
                };
                workload.context_tokens = Some(context_tokens as u64);
                ExactDeviceCase::new(
                    "paged_gqa/online_softmax_e4m3_kv",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes, tokens as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes = PARTITIONED_PHYSICAL_PAGES
        * Qwen38_27B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let decode_table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let decode_lengths = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let prefill_table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let prefill_lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let partitioned_block_table = layout.reserve(PARTITIONED_PHYSICAL_PAGES, ALIGNMENT)?;
    let partitioned_table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let partitioned_short_lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let partitioned_long_lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let macro_short_lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let macro_long_lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let partitioned_partials = layout.reserve(
        PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES / size_of::<f32>(),
        ALIGNMENT,
    )?;
    let output = layout.reserve(MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_tables,
            decode_table_rows,
            decode_lengths,
            prefill_table_rows,
            prefill_lengths,
            partitioned_block_table,
            partitioned_table_rows,
            partitioned_short_lengths,
            partitioned_long_lengths,
            macro_short_lengths,
            macro_long_lengths,
            partitioned_partials,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    debug_assert_eq!(BLOCK_TABLES.len(), SHARED_PHYSICAL_PAGES);
    let query = (0..MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_VALUES[(index + index / Qwen38_27B::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let plane_bytes = PARTITIONED_PHYSICAL_PAGES
        * Qwen38_27B::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38_27B::HEAD_DIM;
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
    let prefill_table_rows = (0..MAX_TOKENS)
        .map(|token| ((token / 2) % TABLE_ROWS) as u32)
        .collect::<Vec<_>>();
    let prefill_lengths = (0..MAX_TOKENS)
        .map(|token| (PREFIX_TOKENS + token + 1) as u32)
        .collect::<Vec<_>>();
    let partitioned_block_table = (0..PARTITIONED_PHYSICAL_PAGES)
        .map(|page| ((page * 17) % PARTITIONED_PHYSICAL_PAGES) as u32)
        .collect::<Vec<_>>();
    let partitioned_table_rows = vec![0u32; MAX_TOKENS];
    let route_lengths = |context_tokens: usize, tokens: usize| {
        let first = context_tokens - tokens + 1;
        (0..MAX_TOKENS)
            .map(|token| (first + token.min(tokens - 1)) as u32)
            .collect::<Vec<_>>()
    };
    arena.copy_from_host(stream, regions.decode_table_rows, &DECODE_TABLE_ROW_IDS)?;
    arena.copy_from_host(stream, regions.decode_lengths, &DECODE_LENGTHS)?;
    arena.copy_from_host(stream, regions.prefill_table_rows, &prefill_table_rows)?;
    arena.copy_from_host(stream, regions.prefill_lengths, &prefill_lengths)?;
    arena.copy_from_host(
        stream,
        regions.partitioned_block_table,
        &partitioned_block_table,
    )?;
    arena.copy_from_host(
        stream,
        regions.partitioned_table_rows,
        &partitioned_table_rows,
    )?;
    arena.copy_from_host(
        stream,
        regions.partitioned_short_lengths,
        &route_lengths(PARTITIONED_PREFILL_ROUTES[0].0, TAIL_TOKENS),
    )?;
    arena.copy_from_host(
        stream,
        regions.partitioned_long_lengths,
        &route_lengths(PARTITIONED_PREFILL_ROUTES[1].0, TAIL_TOKENS),
    )?;
    arena.copy_from_host(
        stream,
        regions.macro_short_lengths,
        &route_lengths(MACRO_PREFILL_ROUTES[0].0, MAX_TOKENS),
    )?;
    arena.copy_from_host(
        stream,
        regions.macro_long_lengths,
        &route_lengths(MACRO_PREFILL_ROUTES[1].0, MAX_TOKENS),
    )
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_tables: arena.address(regions.block_tables)?,
        decode_table_rows: arena.address(regions.decode_table_rows)?,
        decode_lengths: arena.address(regions.decode_lengths)?,
        prefill_table_rows: arena.address(regions.prefill_table_rows)?,
        prefill_lengths: arena.address(regions.prefill_lengths)?,
        partitioned_block_table: arena.address(regions.partitioned_block_table)?,
        partitioned_table_rows: arena.address(regions.partitioned_table_rows)?,
        partitioned_short_lengths: arena.address(regions.partitioned_short_lengths)?,
        partitioned_long_lengths: arena.address(regions.partitioned_long_lengths)?,
        macro_short_lengths: arena.address(regions.macro_short_lengths)?,
        macro_long_lengths: arena.address(regions.macro_long_lengths)?,
        partitioned_partials: arena.address(regions.partitioned_partials)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &PagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    route: Route,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, route))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, route)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        route,
        leaf,
        repeated,
    })
}

fn launch(
    op: &PagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    route: Route,
) -> GpuResult<()> {
    // SAFETY: the benchmark owns every referenced page and the maximum P=16
    // partial workspace. Each metadata row covers its exact route context.
    unsafe {
        match route {
            Route::Decode { batch } => op.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables,
                addresses.decode_table_rows,
                TABLE_STRIDE,
                addresses.decode_lengths,
                addresses.output,
                KEY_SCALE,
                VALUE_SCALE,
            ),
            Route::SharedPrefill { tokens } => op.launch_prefill_shared(
                stream,
                tokens,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables,
                addresses.prefill_table_rows,
                TABLE_STRIDE,
                addresses.prefill_lengths,
                addresses.output,
                KEY_SCALE,
                VALUE_SCALE,
            ),
            Route::PartitionedPrefill { context_tokens, .. } => {
                let lengths = if context_tokens == PARTITIONED_PREFILL_ROUTES[0].0 {
                    addresses.partitioned_short_lengths
                } else {
                    addresses.partitioned_long_lengths
                };

                op.launch_prefill_partitioned(
                    stream,
                    context_tokens,
                    addresses.query,
                    addresses.key_pages,
                    addresses.value_pages,
                    addresses.partitioned_block_table,
                    addresses.partitioned_table_rows,
                    PARTITIONED_PHYSICAL_PAGES,
                    lengths,
                    addresses.partitioned_partials,
                    addresses.output,
                    KEY_SCALE,
                    VALUE_SCALE,
                )
            }
            Route::MacroPrefill {
                context_tokens,
                partitions,
            } => {
                let lengths = if context_tokens == MACRO_PREFILL_ROUTES[0].0 {
                    addresses.macro_short_lengths
                } else {
                    addresses.macro_long_lengths
                };

                op.launch_prefill_macro(
                    stream,
                    partitions,
                    addresses.query,
                    addresses.key_pages,
                    addresses.value_pages,
                    addresses.partitioned_block_table,
                    addresses.partitioned_table_rows,
                    PARTITIONED_PHYSICAL_PAGES,
                    lengths,
                    addresses.partitioned_partials,
                    addresses.output,
                    KEY_SCALE,
                    VALUE_SCALE,
                )
            }
        }
    }
}

fn decode_logical_bytes(batch: usize) -> usize {
    let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let scanned_positions = batch * CONTEXT_TOKENS;
    let cache = 2 * Qwen38_27B::NUM_ATTENTION_HEADS * scanned_positions * Qwen38_27B::HEAD_DIM;
    let page_metadata = 2 * batch * size_of::<u32>()
        + Qwen38_27B::NUM_ATTENTION_HEADS * scanned_positions * size_of::<u32>();

    batch * (query + output) + cache + page_metadata
}

fn shared_prefill_logical_bytes(tokens: usize) -> usize {
    let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let shared_positions = (0..tokens)
        .step_by(2)
        .map(|first_token| PREFIX_TOKENS + first_token + 2)
        .sum::<usize>();
    let cache = 2 * Qwen38_27B::NUM_KV_HEADS * shared_positions * Qwen38_27B::HEAD_DIM;
    let page_metadata =
        Qwen38_27B::NUM_KV_HEADS * (tokens + tokens / 2 + shared_positions) * size_of::<u32>();

    tokens * (query + output) + cache + page_metadata
}

fn partitioned_prefill_logical_bytes(context_tokens: usize, partitions: usize) -> usize {
    let key_tile = match partitions {
        8 => 64,
        16 => 32,
        _ => unreachable!(),
    };
    flash_prefill_logical_bytes(TAIL_TOKENS, context_tokens, partitions, key_tile)
}

fn macro_prefill_logical_bytes(context_tokens: usize, partitions: usize) -> usize {
    debug_assert!(matches!(partitions, 1 | 2 | 4 | 8 | 16));
    flash_prefill_logical_bytes(MAX_TOKENS, context_tokens, partitions, 32)
}

fn flash_prefill_logical_bytes(
    tokens: usize,
    context_tokens: usize,
    partitions: usize,
    key_tile: usize,
) -> usize {
    let first_length = context_tokens - tokens + 1;
    let mut query = 0usize;
    let mut cache = 0usize;
    let mut metadata = 0usize;
    for first_token in (0..tokens).step_by(32) {
        let group_length = first_length + first_token + 31;
        let key_tiles = group_length.div_ceil(key_tile);
        let tiles_per_partition = key_tiles.div_ceil(partitions);
        let active_partitions = key_tiles.div_ceil(tiles_per_partition);
        query += active_partitions
            * 32
            * Qwen38_27B::NUM_ATTENTION_HEADS
            * Qwen38_27B::HEAD_DIM
            * size_of::<f32>();
        cache += 2 * Qwen38_27B::NUM_ATTENTION_HEADS * group_length * Qwen38_27B::HEAD_DIM;
        metadata += Qwen38_27B::NUM_ATTENTION_HEADS
            * (partitions + active_partitions * (1 + 32) + key_tiles)
            * size_of::<u32>();
    }
    let partials = tokens
        * Qwen38_27B::NUM_ATTENTION_HEADS
        * partitions
        * (Qwen38_27B::HEAD_DIM + 2)
        * size_of::<f32>();
    let output = tokens * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();

    query + cache + metadata + 2 * partials + output
}

/// Measures exact decode, tail, and macro paged-GQA graph routes.
pub fn benchmark_paged_gqa(
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
    let partial_bytes = session.regions.partial_bytes();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - partial_bytes - padding_bytes;
    memory.register_owned(
        "paged_gqa/kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "1,536 physical pages, four KV heads, 64 positions, represented E4M3 K/V",
    )?;
    memory.register_owned(
        "paged_gqa/partition_partials",
        BenchmarkMemoryKind::Workspace,
        partial_bytes,
        "T=1024, 24 query heads, maximum P=16, complete FP32 max/denominator/numerator states",
    )?;
    memory.register_owned(
        "paged_gqa/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_tokens=1024 query/output plus separate decode, tail-prefill, and macro metadata",
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
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-paged-gqa",
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph at decode, shared-tail, partitioned-tail, or macro-prefill context",
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
    use super::{
        CONTEXT_TOKENS, MACRO_PREFILL_ROUTES, MAX_BATCH, MAX_TOKENS, PARTITIONED_PHYSICAL_PAGES,
        PARTITIONED_PREFILL_ROUTES, PREFIX_TOKENS, SHARED_PHYSICAL_PAGES, TABLE_ROWS, TABLE_STRIDE,
        TAIL_TOKENS, decode_logical_bytes, layout, macro_prefill_logical_bytes,
        partitioned_prefill_logical_bytes, shared_prefill_logical_bytes,
    };
    use tuisko_kernels_sm120::{
        PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, paged_gqa_prefill_partitions,
    };
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn paged_gqa_suite_byte_accounting_covers_every_admitted_route_family() {
        let per_token = 2 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * 4
            + 2 * Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * Qwen38_27B::HEAD_DIM
            + 2 * 4
            + Qwen38_27B::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * 4;

        assert_eq!(decode_logical_bytes(1), per_token);
        assert_eq!(decode_logical_bytes(MAX_BATCH), MAX_BATCH * per_token);

        let shared_positions = (0..TAIL_TOKENS)
            .step_by(2)
            .map(|first_token| PREFIX_TOKENS + first_token + 2)
            .sum::<usize>();
        let prefill = 2 * TAIL_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * 4
            + 2 * Qwen38_27B::NUM_KV_HEADS * shared_positions * Qwen38_27B::HEAD_DIM
            + Qwen38_27B::NUM_KV_HEADS * (TAIL_TOKENS + TAIL_TOKENS / 2 + shared_positions) * 4;
        assert_eq!(shared_prefill_logical_bytes(TAIL_TOKENS), prefill);

        for (context_tokens, partitions) in PARTITIONED_PREFILL_ROUTES {
            assert_eq!(
                paged_gqa_prefill_partitions(context_tokens).unwrap(),
                partitions
            );
            let logical = partitioned_prefill_logical_bytes(context_tokens, partitions);
            let first_length = context_tokens - TAIL_TOKENS + 1;
            let key_tile = if partitions == 8 { 64 } else { 32 };
            let groups = (0..TAIL_TOKENS)
                .step_by(32)
                .map(|first_token| first_length + first_token + 31)
                .collect::<Vec<_>>();
            let cache = 2
                * Qwen38_27B::NUM_ATTENTION_HEADS
                * groups.iter().sum::<usize>()
                * Qwen38_27B::HEAD_DIM;
            let query =
                TAIL_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * partitions * size_of::<f32>();
            let partials = TAIL_TOKENS
                * Qwen38_27B::NUM_ATTENTION_HEADS
                * partitions
                * (Qwen38_27B::HEAD_DIM + 2)
                * 4;
            assert!(logical > query + cache + 2 * partials);
            assert!(
                groups
                    .iter()
                    .all(|group_length| group_length.div_ceil(key_tile) >= partitions)
            );
            assert!(logical > 2 * partials);
            assert!(logical > shared_prefill_logical_bytes(TAIL_TOKENS));
        }
        for (context_tokens, partitions) in MACRO_PREFILL_ROUTES {
            let logical = macro_prefill_logical_bytes(context_tokens, partitions);
            let first_length = context_tokens - MAX_TOKENS + 1;
            let groups = (0..MAX_TOKENS)
                .step_by(32)
                .map(|first_token| first_length + first_token + 31)
                .collect::<Vec<_>>();
            let cache = 2
                * Qwen38_27B::NUM_ATTENTION_HEADS
                * groups.iter().sum::<usize>()
                * Qwen38_27B::HEAD_DIM;
            let query =
                MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * partitions * size_of::<f32>();
            let partials = MAX_TOKENS
                * Qwen38_27B::NUM_ATTENTION_HEADS
                * partitions
                * (Qwen38_27B::HEAD_DIM + 2)
                * size_of::<f32>();
            assert!(logical > query + cache + 2 * partials);
            assert!(
                groups
                    .iter()
                    .all(|group_length| group_length.div_ceil(32) >= partitions)
            );
            assert!(logical > partitioned_prefill_logical_bytes(98_304, 16));
        }
        assert_eq!(SHARED_PHYSICAL_PAGES, TABLE_ROWS * TABLE_STRIDE);
        assert_eq!(PARTITIONED_PHYSICAL_PAGES, 1_536);
    }

    #[test]
    fn paged_gqa_suite_arena_accounting_exposes_every_padding_byte() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(
            regions.partitioned_partials.byte_len(),
            PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES
        );
        assert!(layout.byte_len() > regions.payload_bytes());
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 608);
    }
}

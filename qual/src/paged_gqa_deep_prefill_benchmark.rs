//! Controlled shared-versus-partitioned crossover timings for deep exact prefill tails.

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

const TOKEN_WIDTHS: [usize; 2] = [32, 64];
const CONTEXTS: [usize; 7] = [8_192, 32_768, 65_536, 98_304, 108_095, 131_073, 220_000];
const PARTITIONS: [usize; 2] = [8, 16];
const MAX_TOKENS: usize = 64;
const MAX_CONTEXT: usize = 220_000;
const PHYSICAL_PAGES: usize = MAX_CONTEXT.div_ceil(ATTENTION_PAGE_SIZE);
const LENGTH_ROWS: usize = TOKEN_WIDTHS.len() * CONTEXTS.len();
const MAX_PARTITIONS: usize = 16;
const PARTIAL_VALUES: usize = Qwen38_27B::HEAD_DIM + 2;
const PARTIAL_FLOATS: usize =
    MAX_TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * MAX_PARTITIONS * PARTIAL_VALUES;
const ALIGNMENT: usize = 256;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const QUERY_VALUES: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_CODES: [u8; 8] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20];
const VALUE_CODES: [u8; 8] = [0x38, 0xb8, 0x30, 0xb0, 0x28, 0xa8, 0x20, 0xa0];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RouteKind {
    Shared,
    Partitioned(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Route {
    tokens: usize,
    context_tokens: usize,
    lengths_row: usize,
    kind: RouteKind,
}

fn routes() -> Vec<Route> {
    TOKEN_WIDTHS
        .into_iter()
        .enumerate()
        .flat_map(|(width_index, tokens)| {
            CONTEXTS
                .into_iter()
                .enumerate()
                .flat_map(move |(context_index, context_tokens)| {
                    let lengths_row = width_index * CONTEXTS.len() + context_index;
                    std::iter::once(Route {
                        tokens,
                        context_tokens,
                        lengths_row,
                        kind: RouteKind::Shared,
                    })
                    .chain(PARTITIONS.into_iter().map(move |partitions| {
                        Route {
                            tokens,
                            context_tokens,
                            lengths_row,
                            kind: RouteKind::Partitioned(partitions),
                        }
                    }))
                })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_table: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
    partials: ArenaRegion<f32>,
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
            + self.partials.byte_len()
            + self.output.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }
}

#[derive(Clone, Copy)]
struct Addresses {
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_table: *const u32,
    table_rows: *const u32,
    lengths: *const u32,
    partials: *mut f32,
    output: *mut f32,
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
        let routes = routes()
            .into_iter()
            .map(|route| capture_route(&op, &stream, addresses, route, repeated_operations))
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
                // SAFETY: the session owns the graph and every captured address.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let shape = match route.route.kind {
                    RouteKind::Shared => format!(
                        "T={}/shared/context={}",
                        route.route.tokens, route.route.context_tokens
                    ),
                    RouteKind::Partitioned(partitions) => format!(
                        "T={}/P={partitions}/context={}",
                        route.route.tokens, route.route.context_tokens
                    ),
                };
                let logical_bytes = match route.route.kind {
                    RouteKind::Shared => {
                        deep_shared_logical_bytes(route.route.tokens, route.route.context_tokens)
                    }
                    RouteKind::Partitioned(partitions) => partitioned_logical_bytes(
                        route.route.tokens,
                        route.route.context_tokens,
                        partitions,
                    ),
                };
                let mut workload =
                    BenchmarkWorkload::warm_operator_prefill(route.route.tokens as u64);
                workload.context_tokens = Some(route.route.context_tokens as u64);
                ExactDeviceCase::new(
                    "paged_gqa/deep_prefill_crossover",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes, route.route.tokens as u64, "token"),
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
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_table = layout.reserve(PHYSICAL_PAGES, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(LENGTH_ROWS * MAX_TOKENS, ALIGNMENT)?;
    let partials = layout.reserve(PARTIAL_FLOATS, ALIGNMENT)?;
    let output = layout.reserve(MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_table,
            table_rows,
            lengths,
            partials,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let query = (0..MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
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
        .map(|page| ((page * 17) % PHYSICAL_PAGES) as u32)
        .collect::<Vec<_>>();
    let table_rows = vec![0u32; MAX_TOKENS];
    let mut lengths = vec![0u32; LENGTH_ROWS * MAX_TOKENS];
    for route in routes()
        .into_iter()
        .filter(|route| route.kind == RouteKind::Shared)
    {
        let first = route.context_tokens - route.tokens + 1;
        let row =
            &mut lengths[route.lengths_row * MAX_TOKENS..(route.lengths_row + 1) * MAX_TOKENS];
        for (token, length) in row.iter_mut().enumerate() {
            *length = (first + token.min(route.tokens - 1)) as u32;
        }
    }

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_table, &block_table)?;
    arena.copy_from_host(stream, regions.table_rows, &table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &lengths)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_table: arena.address(regions.block_table)?,
        table_rows: arena.address(regions.table_rows)?,
        lengths: arena.address(regions.lengths)?,
        partials: arena.address(regions.partials)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &PagedGqaOp,
    stream: &CudaStream,
    addresses: Addresses,
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
    addresses: Addresses,
    route: Route,
) -> GpuResult<()> {
    // SAFETY: every route uses one complete table row, an exact 32/64-row
    // metadata slice, the maximum P=16 partial workspace, and a 220K cache.
    unsafe {
        let lengths = addresses.lengths.add(route.lengths_row * MAX_TOKENS);
        match route.kind {
            RouteKind::Shared => op.launch_prefill_shared(
                stream,
                route.tokens,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_table,
                addresses.table_rows,
                PHYSICAL_PAGES,
                lengths,
                addresses.output,
                KEY_SCALE,
                VALUE_SCALE,
            ),
            RouteKind::Partitioned(partitions) => op.launch_prefill_partitioned(
                stream,
                route.tokens,
                partitions,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_table,
                addresses.table_rows,
                PHYSICAL_PAGES,
                lengths,
                addresses.partials,
                addresses.output,
                KEY_SCALE,
                VALUE_SCALE,
            ),
        }
    }
}

fn deep_shared_logical_bytes(tokens: usize, context_tokens: usize) -> usize {
    let query = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let first_length = context_tokens - tokens + 1;
    let shared_positions = (0..tokens)
        .step_by(2)
        .map(|first_token| first_length + first_token + 1)
        .sum::<usize>();
    let cache = 2 * Qwen38_27B::NUM_KV_HEADS * shared_positions * Qwen38_27B::HEAD_DIM;
    let metadata =
        Qwen38_27B::NUM_KV_HEADS * (tokens + tokens / 2 + shared_positions) * size_of::<u32>();
    tokens * (query + output) + cache + metadata
}

fn partitioned_logical_bytes(tokens: usize, context_tokens: usize, partitions: usize) -> usize {
    let key_tile = match partitions {
        8 => 64,
        16 => 32,
        _ => unreachable!(),
    };
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
            * (partitions + active_partitions * 33 + key_tiles)
            * size_of::<u32>();
    }
    let partials =
        tokens * Qwen38_27B::NUM_ATTENTION_HEADS * partitions * PARTIAL_VALUES * size_of::<f32>();
    let output = tokens * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    query + cache + metadata + 2 * partials + output
}

/// Measures the 42 controlled T32/T64 shared/P8/P16 crossover cells.
pub fn benchmark_paged_gqa_deep_prefill(
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
    let partial_bytes = session.regions.partials.byte_len();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - partial_bytes - padding_bytes;
    memory.register_owned(
        "paged_gqa_deep_prefill/kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "3,438 physical pages, four KV heads, 64 positions, represented E4M3 K/V",
    )?;
    memory.register_owned(
        "paged_gqa_deep_prefill/partition_partials",
        BenchmarkMemoryKind::Workspace,
        partial_bytes,
        "T=64, 24 query heads, maximum P=16, complete FP32 partition states",
    )?;
    memory.register_owned(
        "paged_gqa_deep_prefill/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "T=64 query/output plus fourteen exact context-length rows and one block table",
    )?;
    memory.register_owned(
        "paged_gqa_deep_prefill/alignment_padding",
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
            suite: "bench-paged-gqa-deep-prefill",
            classification: "optimization_candidate_exact_leaf_crossover",
            timing_scope: "paired Rust completion and CUDA-event time for exact T32/T64 shared, P8, and P16 production leaf graphs at seven controlled context depths",
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
        CONTEXTS, LENGTH_ROWS, MAX_TOKENS, PARTITIONS, RouteKind, TOKEN_WIDTHS,
        deep_shared_logical_bytes, layout, partitioned_logical_bytes, routes,
    };
    use std::collections::BTreeSet;

    #[test]
    fn paged_gqa_suite_deep_prefill_crossover_inventory_is_exact_and_independent() {
        let routes = routes();
        let identities = routes
            .iter()
            .map(|route| {
                (
                    route.tokens,
                    route.context_tokens,
                    route.lengths_row,
                    route.kind,
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(routes.len(), TOKEN_WIDTHS.len() * CONTEXTS.len() * 3);
        assert_eq!(identities.len(), routes.len());
        assert_eq!(LENGTH_ROWS, TOKEN_WIDTHS.len() * CONTEXTS.len());
        for tokens in TOKEN_WIDTHS {
            for context_tokens in CONTEXTS {
                assert!(routes.iter().any(|route| {
                    route.tokens == tokens
                        && route.context_tokens == context_tokens
                        && route.kind == RouteKind::Shared
                }));
                for partitions in PARTITIONS {
                    assert!(routes.iter().any(|route| {
                        route.tokens == tokens
                            && route.context_tokens == context_tokens
                            && route.kind == RouteKind::Partitioned(partitions)
                    }));
                }
            }
        }
    }

    #[test]
    fn paged_gqa_suite_deep_prefill_accounting_binds_width_context_and_partition() {
        for tokens in TOKEN_WIDTHS {
            let mut previous_shared = 0;
            for context_tokens in CONTEXTS {
                let shared = deep_shared_logical_bytes(tokens, context_tokens);
                assert!(shared > previous_shared);
                previous_shared = shared;
                let mut partitioned = Vec::new();
                for partitions in PARTITIONS {
                    partitioned.push(partitioned_logical_bytes(
                        tokens,
                        context_tokens,
                        partitions,
                    ));
                }
                assert!(partitioned.iter().all(|&bytes| bytes > 0));
                assert_ne!(partitioned[0], partitioned[1]);
            }
        }
        let (arena, regions) = layout().unwrap();
        assert!(arena.byte_len() > regions.payload_bytes());
        assert_eq!(regions.lengths.len(), LENGTH_ROWS * MAX_TOKENS);
    }
}

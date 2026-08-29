//! Direct timings for exact MTP BF16 split-KV paged GQA graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE, MtpBf16PagedGqaOp,
    MtpBf16SplitKvPagedGqaOp,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const CONTEXT_TOKENS: usize = LONG_CONTEXT_GQA_MAX_TOKENS;
const PHYSICAL_PAGES: usize = CONTEXT_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);
const BATCH_ROUTE_METADATA_ELEMENTS: usize = MAX_BATCH * (MAX_BATCH + 1) / 2;
const CROSSOVER_LENGTHS: [usize; 12] = [
    1, 32, 128, 257, 1_024, 1_025, 1_280, 1_536, 1_792, 2_048, 3_072, 4_096,
];
const ROUTE_METADATA_ELEMENTS: usize = BATCH_ROUTE_METADATA_ELEMENTS + CROSSOVER_LENGTHS.len();
const QUERY_VALUES: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_VALUES: [f32; 8] = [0.0, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0, 0.125];
const VALUE_VALUES: [f32; 8] = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125];

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u16>,
    value_pages: ArenaRegion<u16>,
    block_tables: ArenaRegion<u32>,
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
            + self.block_tables.byte_len()
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
    key_pages: *const u16,
    value_pages: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    lengths: *const u32,
    partial_maximum: *mut f32,
    partial_denominator: *mut f32,
    partial_numerator: *mut f32,
    output: *mut f32,
}

struct RouteGraphs {
    spec: RouteSpec,
    kind: RouteKind,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

#[derive(Clone, Copy)]
struct RouteSpec {
    batch: usize,
    context_tokens: usize,
    table_stride: usize,
    table_offset: usize,
    metadata_offset: usize,
}

#[derive(Clone, Copy)]
enum RouteKind {
    Control,
    SplitKv,
}

impl RouteKind {
    fn route(self) -> &'static str {
        match self {
            Self::Control => "mtp_bf16_paged_gqa/warp_partitioned_control",
            Self::SplitKv => "mtp_bf16_paged_gqa/cross_cta_split_kv",
        }
    }
}

struct Session {
    routes: Vec<RouteGraphs>,
    _control: MtpBf16PagedGqaOp,
    _split_kv: MtpBf16SplitKvPagedGqaOp,
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
        let control = MtpBf16PagedGqaOp::new(&context)?;
        let split_kv = MtpBf16SplitKvPagedGqaOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let batch_routes = (1..=MAX_BATCH).map(|batch| RouteSpec {
            batch,
            context_tokens: route_length(batch),
            table_stride: route_pages(batch),
            table_offset: route_table_offset(batch),
            metadata_offset: route_metadata_offset(batch),
        });
        let crossover_routes =
            CROSSOVER_LENGTHS
                .into_iter()
                .enumerate()
                .map(|(index, context_tokens)| RouteSpec {
                    batch: 1,
                    context_tokens,
                    table_stride: route_pages(1),
                    table_offset: route_table_offset(1),
                    metadata_offset: BATCH_ROUTE_METADATA_ELEMENTS + index,
                });
        let routes = batch_routes
            .chain(crossover_routes)
            .flat_map(|batch| {
                [RouteKind::Control, RouteKind::SplitKv].map(move |kind| (batch, kind))
            })
            .map(|(spec, kind)| {
                capture_route(
                    &control,
                    &split_kv,
                    &stream,
                    &addresses,
                    spec,
                    kind,
                    repeated_operations,
                )
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _control: control,
            _split_kv: split_kv,
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
                let mut workload = BenchmarkWorkload::warm_operator_decode(route.spec.batch as u32);
                let context_tokens = route.spec.context_tokens;
                workload.context_tokens = Some(context_tokens as u64);
                ExactDeviceCase::new(
                    route.kind.route(),
                    format!("B={},context={context_tokens}", route.spec.batch),
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.spec.batch, context_tokens, route.kind),
                        route.spec.batch as u64,
                        "token",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn scalar_elements() -> usize {
    MAX_BATCH * Qwen38_27B::NUM_ATTENTION_HEADS * LONG_CONTEXT_GQA_MAX_PARTITIONS
}

fn route_pages(batch: usize) -> usize {
    PHYSICAL_PAGES / batch
}

fn route_length(batch: usize) -> usize {
    (route_pages(batch) * ATTENTION_PAGE_SIZE).min(CONTEXT_TOKENS)
}

fn route_table_offset(batch: usize) -> usize {
    (1..batch).map(|prior| prior * route_pages(prior)).sum()
}

fn route_metadata_offset(batch: usize) -> usize {
    batch * (batch - 1) / 2
}

fn table_elements() -> usize {
    (1..=MAX_BATCH)
        .map(|batch| batch * route_pages(batch))
        .sum()
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_elements =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_elements, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_elements, ALIGNMENT)?;
    let block_tables = layout.reserve(table_elements(), ALIGNMENT)?;
    let table_rows = layout.reserve(ROUTE_METADATA_ELEMENTS, ALIGNMENT)?;
    let lengths = layout.reserve(ROUTE_METADATA_ELEMENTS, ALIGNMENT)?;
    let partial_maximum = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_denominator = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_numerator = layout.reserve(scalar_elements() * Qwen38_27B::HEAD_DIM, ALIGNMENT)?;
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
            partial_maximum,
            partial_denominator,
            partial_numerator,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let query = (0..MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_VALUES[(index + index / Qwen38_27B::HEAD_DIM) & 7])
        .collect::<Vec<_>>();
    let plane_elements =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = (0..plane_elements)
        .map(|index| f32_to_bf16(KEY_VALUES[(index + index / Qwen38_27B::HEAD_DIM) & 7]))
        .collect::<Vec<_>>();
    let value_pages = (0..plane_elements)
        .map(|index| f32_to_bf16(VALUE_VALUES[(index * 3 + index / Qwen38_27B::HEAD_DIM) & 7]))
        .collect::<Vec<_>>();
    let mut block_tables = Vec::with_capacity(table_elements());
    let mut table_rows = Vec::with_capacity(ROUTE_METADATA_ELEMENTS);
    let mut lengths = Vec::with_capacity(ROUTE_METADATA_ELEMENTS);
    for batch in 1..=MAX_BATCH {
        let pages = route_pages(batch);
        let length = u32::try_from(route_length(batch)).expect("context fits u32");
        for token in 0..batch {
            table_rows.push(u32::try_from(token).expect("batch fits u32"));
            lengths.push(length);
            block_tables.extend(
                (0..pages)
                    .map(|page| u32::try_from(token * pages + page).expect("page ID fits u32")),
            );
        }
    }
    table_rows.extend(std::iter::repeat_n(0, CROSSOVER_LENGTHS.len()));
    lengths.extend(
        CROSSOVER_LENGTHS
            .into_iter()
            .map(|length| u32::try_from(length).expect("crossover context fits u32")),
    );

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &block_tables)?;
    arena.copy_from_host(stream, regions.table_rows, &table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &lengths)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_tables: arena.address(regions.block_tables)?,
        table_rows: arena.address(regions.table_rows)?,
        lengths: arena.address(regions.lengths)?,
        partial_maximum: arena.address(regions.partial_maximum)?,
        partial_denominator: arena.address(regions.partial_denominator)?,
        partial_numerator: arena.address(regions.partial_numerator)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    control: &MtpBf16PagedGqaOp,
    split_kv: &MtpBf16SplitKvPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    spec: RouteSpec,
    kind: RouteKind,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || {
        launch(control, split_kv, stream, addresses, spec, kind)
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(control, split_kv, stream, addresses, spec, kind)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        spec,
        kind,
        leaf,
        repeated,
    })
}

fn launch(
    control: &MtpBf16PagedGqaOp,
    split_kv: &MtpBf16SplitKvPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    spec: RouteSpec,
    kind: RouteKind,
) -> GpuResult<()> {
    let batch = spec.batch;
    // SAFETY: each exact route partitions the one 3,438-page production pool
    // into disjoint slot rows; all scratch planes own the B=8 maximum.
    unsafe {
        match kind {
            RouteKind::Control => control.launch(
                stream,
                batch,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables.add(spec.table_offset),
                addresses.table_rows.add(spec.metadata_offset),
                spec.table_stride,
                addresses.lengths.add(spec.metadata_offset),
                addresses.output,
            ),
            RouteKind::SplitKv => split_kv.launch(
                stream,
                batch,
                spec.context_tokens,
                addresses.query,
                addresses.key_pages,
                addresses.value_pages,
                addresses.block_tables.add(spec.table_offset),
                addresses.table_rows.add(spec.metadata_offset),
                spec.table_stride,
                addresses.lengths.add(spec.metadata_offset),
                addresses.partial_maximum,
                addresses.partial_denominator,
                addresses.partial_numerator,
                addresses.output,
            ),
        }
    }
}

fn logical_bytes(batch: usize, context_tokens: usize, kind: RouteKind) -> usize {
    let partitions = context_tokens.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
    let launched_partitions = LONG_CONTEXT_GQA_PARTITION_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| bucket >= partitions)
        .expect("admitted context has a partition bucket");
    let partials = Qwen38_27B::NUM_ATTENTION_HEADS * partitions;
    let query_repetitions = match kind {
        RouteKind::Control => 8 * Qwen38_27B::NUM_ATTENTION_HEADS,
        RouteKind::SplitKv => partials,
    };
    let query = query_repetitions * Qwen38_27B::HEAD_DIM * size_of::<f32>();
    let cache = 2
        * Qwen38_27B::NUM_ATTENTION_HEADS
        * context_tokens
        * Qwen38_27B::HEAD_DIM
        * size_of::<u16>();
    let block_table = Qwen38_27B::NUM_ATTENTION_HEADS * context_tokens * size_of::<u32>();
    let metadata =
        Qwen38_27B::NUM_ATTENTION_HEADS * (launched_partitions + partitions + 1) * size_of::<u32>();
    let partial = match kind {
        RouteKind::Control => 0,
        RouteKind::SplitKv => 2 * partials * (2 + Qwen38_27B::HEAD_DIM) * size_of::<f32>(),
    };
    let output = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();

    batch * (query + cache + block_table + metadata + partial + output)
}

/// Measures every exact batch with the complete 3,438-page pool divided among active slots.
pub fn benchmark_mtp_bf16_split_kv_paged_gqa(
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
    let workspace_bytes = session.arena.byte_len() - cache_bytes - padding_bytes;
    let partition_workspace_bytes = session.regions.partition_workspace_bytes();
    let boundary_workspace_bytes = workspace_bytes - partition_workspace_bytes;
    memory.register_owned(
        "mtp_bf16_split_kv_paged_gqa/represented_kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "3,438 physical pages, four KV heads, 64 positions, represented BF16 K/V",
    )?;
    memory.register_owned(
        "mtp_bf16_split_kv_paged_gqa/partition_workspace",
        BenchmarkMemoryKind::Workspace,
        partition_workspace_bytes,
        "max_batch=8, 24 query heads, 860 maximum/denominator/numerator FP32 partials",
    )?;
    memory.register_owned(
        "mtp_bf16_split_kv_paged_gqa/address_stable_boundary",
        BenchmarkMemoryKind::Workspace,
        boundary_workspace_bytes,
        "max_batch=8 query/output and complete 3,438-page table metadata",
    )?;
    memory.register_owned(
        "mtp_bf16_split_kv_paged_gqa/alignment_padding",
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
            suite: "bench-mtp-bf16-split-kv-paged-gqa",
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production two-stage graph, and repeated-operation graph with the complete 3,438-page pool divided among active slots",
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
        CONTEXT_TOKENS, MAX_BATCH, PHYSICAL_PAGES, RouteKind, layout, logical_bytes, route_length,
        route_pages, scalar_elements, table_elements,
    };

    #[test]
    fn mtp_bf16_paged_gqa_split_kv_byte_accounting_covers_partition_and_reduction_traffic() {
        assert_eq!(
            logical_bytes(1, route_length(1), RouteKind::Control),
            5_428_226_400
        );
        assert_eq!(
            logical_bytes(1, route_length(1), RouteKind::SplitKv),
            5_491_766_112
        );
        assert_eq!(
            logical_bytes(MAX_BATCH, route_length(MAX_BATCH), RouteKind::Control),
            5_421_205_248
        );
        assert_eq!(
            logical_bytes(MAX_BATCH, route_length(MAX_BATCH), RouteKind::SplitKv),
            5_483_665_152
        );
    }

    #[test]
    fn mtp_bf16_paged_gqa_split_kv_each_batch_uses_only_the_shared_resident_page_pool() {
        assert_eq!(
            (1..=MAX_BATCH).map(route_length).collect::<Vec<_>>(),
            [
                220_000, 110_016, 73_344, 54_976, 43_968, 36_672, 31_424, 27_456
            ]
        );
        for batch in 1..=MAX_BATCH {
            assert!(batch * route_pages(batch) <= PHYSICAL_PAGES);
        }
        assert_eq!(table_elements(), 27_492);
    }

    #[test]
    fn mtp_bf16_paged_gqa_split_kv_arena_accounting_exposes_every_owner_and_padding_byte() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(CONTEXT_TOKENS, 220_000);
        assert_eq!(scalar_elements(), 165_120);
        assert_eq!(regions.cache_bytes(), 901_251_072);
        assert_eq!(regions.partition_workspace_bytes(), 170_403_840);
        assert_eq!(
            layout.byte_len() - regions.cache_bytes() - regions.partition_workspace_bytes() - 288,
            503_472
        );
        assert_eq!(layout.byte_len(), 1_072_158_720);
        assert_eq!(regions.payload_bytes(), 1_072_158_432);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 288);
    }
}

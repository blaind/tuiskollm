//! Direct timings for exact full-attention Q/K preparation graph routes.

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
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, Qwen35AttentionQkPrepareOp};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const QWEN35_ROUTES: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 16;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 16;
const ROTARY_PAIRS: usize = 32;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const DECODE_TABLE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const DECODE_CACHE_POSITIONS: [u32; MAX_BATCH] = [63, 64, 1, 126, 2, 65, 127, 0];
const VALUES: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];

#[derive(Clone, Copy)]
struct Regions {
    qkv: ArenaRegion<u16>,
    query_norm: ArenaRegion<u16>,
    key_norm: ArenaRegion<u16>,
    rope_cos: ArenaRegion<f32>,
    rope_sin: ArenaRegion<f32>,
    block_tables: ArenaRegion<u32>,
    decode_table_rows: ArenaRegion<u32>,
    decode_cache_positions: ArenaRegion<u32>,
    prefill_table_rows: ArenaRegion<u32>,
    prefill_cache_positions: ArenaRegion<u32>,
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.qkv.byte_len()
            + self.query_norm.byte_len()
            + self.key_norm.byte_len()
            + self.rope_cos.byte_len()
            + self.rope_sin.byte_len()
            + self.block_tables.byte_len()
            + self.decode_table_rows.byte_len()
            + self.decode_cache_positions.byte_len()
            + self.prefill_table_rows.byte_len()
            + self.prefill_cache_positions.byte_len()
            + self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
    }

    fn weight_bytes(self) -> usize {
        self.query_norm.byte_len() + self.key_norm.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }
}

struct Addresses {
    qkv: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    decode_table_rows: *const u32,
    decode_cache_positions: *const u32,
    prefill_table_rows: *const u32,
    prefill_cache_positions: *const u32,
    query: *mut f32,
    key_pages: *mut u8,
    value_pages: *mut u8,
}

struct RouteGraphs {
    tokens: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

trait BenchQkPrepareOp {
    type Target: Arch;
    const ROUTES: &'static [usize];
    const MAX_TOKENS: usize;
    const CACHE_ELEMENT_BYTES: usize;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self>
    where
        Self: Sized;

    fn launch(&self, stream: &CudaStream, addresses: &Addresses, batch: usize) -> GpuResult<()>;
}

macro_rules! impl_bench_op {
    ($op:ty, $target:ty, $routes:expr, $max_tokens:expr) => {
        impl BenchQkPrepareOp for $op {
            type Target = $target;
            const ROUTES: &'static [usize] = $routes;
            const MAX_TOKENS: usize = $max_tokens;
            const CACHE_ELEMENT_BYTES: usize = size_of::<u8>();

            fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
                <$op>::new(context)
            }

            fn launch(
                &self,
                stream: &CudaStream,
                addresses: &Addresses,
                batch: usize,
            ) -> GpuResult<()> {
                let (table_rows, cache_positions) = if batch <= MAX_BATCH {
                    (
                        addresses.decode_table_rows,
                        addresses.decode_cache_positions,
                    )
                } else {
                    (
                        addresses.prefill_table_rows,
                        addresses.prefill_cache_positions,
                    )
                };
                // SAFETY: the session owns maximum-batch regions and valid
                // page metadata for the lifetime of every captured graph.
                unsafe {
                    <$op>::launch(
                        self,
                        stream,
                        batch,
                        addresses.qkv,
                        addresses.query_norm,
                        addresses.key_norm,
                        addresses.rope_cos,
                        addresses.rope_sin,
                        addresses.block_tables,
                        table_rows,
                        TABLE_STRIDE,
                        cache_positions,
                        addresses.query,
                        addresses.key_pages,
                        addresses.value_pages,
                        KEY_SCALE,
                        VALUE_SCALE,
                    )
                }
            }
        }
    };
}

impl_bench_op!(AttentionQkPrepareOp, Qwen38_27B, &ROUTES, MAX_TOKENS);

impl BenchQkPrepareOp for Qwen35AttentionQkPrepareOp {
    type Target = Qwen35_9B;
    const ROUTES: &'static [usize] = &QWEN35_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen35AttentionQkPrepareOp::new(context)
    }

    fn launch(&self, stream: &CudaStream, addresses: &Addresses, batch: usize) -> GpuResult<()> {
        // SAFETY: the arena reserves aligned BF16 cache planes and owns every
        // pointer through completion of the captured graph.
        unsafe {
            self.launch(
                stream,
                batch,
                addresses.qkv,
                addresses.query_norm,
                addresses.key_norm,
                addresses.rope_cos,
                addresses.rope_sin,
                addresses.block_tables,
                addresses.decode_table_rows,
                TABLE_STRIDE,
                addresses.decode_cache_positions,
                addresses.query,
                addresses.key_pages.cast(),
                addresses.value_pages.cast(),
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

impl<O: BenchQkPrepareOp> Session<O> {
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
        load_fixture::<O::Target>(&arena, &stream, regions, O::MAX_TOKENS)?;
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
        for _ in 0..launches {
            for route in &self.routes {
                route.leaf.launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64, route: &'static str) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route_graph| {
                let (shape, workload) = if route_graph.tokens <= MAX_BATCH {
                    (
                        format!("B={}", route_graph.tokens),
                        BenchmarkWorkload::warm_operator_decode(route_graph.tokens as u32),
                    )
                } else {
                    (
                        format!("T={}", route_graph.tokens),
                        BenchmarkWorkload::warm_operator_prefill(route_graph.tokens as u64),
                    )
                };
                ExactDeviceCase::new(
                    route,
                    shape,
                    workload,
                    OperationAccounting::new(
                        logical_bytes::<O::Target>(route_graph.tokens, O::CACHE_ELEMENT_BYTES),
                        route_graph.tokens as u64,
                        "token",
                    ),
                    &route_graph.leaf,
                    Some(RepeatedGraph::new(
                        &route_graph.repeated,
                        repeated_operations,
                    )),
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
    let qkv = layout.reserve(max_tokens * A::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let query_norm = layout.reserve(A::HEAD_DIM, ALIGNMENT)?;
    let key_norm = layout.reserve(A::HEAD_DIM, ALIGNMENT)?;
    let rope_cos = layout.reserve(max_tokens * ROTARY_PAIRS, ALIGNMENT)?;
    let rope_sin = layout.reserve(max_tokens * ROTARY_PAIRS, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let decode_table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let decode_cache_positions = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let prefill_table_rows = layout.reserve(max_tokens, ALIGNMENT)?;
    let prefill_cache_positions = layout.reserve(max_tokens, ALIGNMENT)?;
    let query = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM * cache_element_bytes;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            qkv,
            query_norm,
            key_norm,
            rope_cos,
            rope_sin,
            block_tables,
            decode_table_rows,
            decode_cache_positions,
            prefill_table_rows,
            prefill_cache_positions,
            query,
            key_pages,
            value_pages,
        },
    ))
}

fn load_fixture<A: Arch>(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    max_tokens: usize,
) -> GpuResult<()> {
    let qkv = (0..max_tokens * A::ATTENTION_QKV_ROWS)
        .map(|index| f32_to_bf16(VALUES[(index + index / A::ATTENTION_QKV_ROWS) & 7]))
        .collect::<Vec<_>>();
    let query_norm = (0..A::HEAD_DIM)
        .map(|index| f32_to_bf16(VALUES[(index + 3) & 7] * 0.25))
        .collect::<Vec<_>>();
    let key_norm = (0..A::HEAD_DIM)
        .map(|index| f32_to_bf16(VALUES[(index + 5) & 7] * 0.25))
        .collect::<Vec<_>>();
    let mut rope_cos = vec![0.0f32; max_tokens * ROTARY_PAIRS];
    let mut rope_sin = vec![0.0f32; max_tokens * ROTARY_PAIRS];
    for token in 0..max_tokens {
        for pair in 0..ROTARY_PAIRS {
            let angle = (token * 3 + pair) as f32 / 128.0;
            rope_cos[token * ROTARY_PAIRS + pair] = angle.cos();
            rope_sin[token * ROTARY_PAIRS + pair] = angle.sin();
        }
    }
    let block_tables = (0..TABLE_ROWS)
        .flat_map(|row| {
            (0..TABLE_STRIDE).map(move |page| ((2 * row + page) % PHYSICAL_PAGES) as u32)
        })
        .collect::<Vec<_>>();
    let prefill_table_rows = vec![0u32; max_tokens];
    let prefill_cache_positions = (0..max_tokens as u32).collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.qkv, &qkv)?;
    arena.copy_from_host(stream, regions.query_norm, &query_norm)?;
    arena.copy_from_host(stream, regions.key_norm, &key_norm)?;
    arena.copy_from_host(stream, regions.rope_cos, &rope_cos)?;
    arena.copy_from_host(stream, regions.rope_sin, &rope_sin)?;
    arena.copy_from_host(stream, regions.block_tables, &block_tables)?;
    arena.copy_from_host(stream, regions.decode_table_rows, &DECODE_TABLE_ROWS)?;
    arena.copy_from_host(
        stream,
        regions.decode_cache_positions,
        &DECODE_CACHE_POSITIONS,
    )?;
    arena.copy_from_host(stream, regions.prefill_table_rows, &prefill_table_rows)?;
    arena.copy_from_host(
        stream,
        regions.prefill_cache_positions,
        &prefill_cache_positions,
    )
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        qkv: arena.address(regions.qkv)?,
        query_norm: arena.address(regions.query_norm)?,
        key_norm: arena.address(regions.key_norm)?,
        rope_cos: arena.address(regions.rope_cos)?,
        rope_sin: arena.address(regions.rope_sin)?,
        block_tables: arena.address(regions.block_tables)?,
        decode_table_rows: arena.address(regions.decode_table_rows)?,
        decode_cache_positions: arena.address(regions.decode_cache_positions)?,
        prefill_table_rows: arena.address(regions.prefill_table_rows)?,
        prefill_cache_positions: arena.address(regions.prefill_cache_positions)?,
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
    })
}

fn capture_route(
    op: &impl BenchQkPrepareOp,
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
    op: &impl BenchQkPrepareOp,
    stream: &CudaStream,
    addresses: &Addresses,
    tokens: usize,
) -> GpuResult<()> {
    op.launch(stream, addresses, tokens)
}

fn logical_bytes<A: Arch>(tokens: usize, cache_element_bytes: usize) -> usize {
    let heads = A::NUM_ATTENTION_HEADS + A::NUM_KV_HEADS;
    let source_values = A::ATTENTION_OUTPUT_COLUMNS + 2 * A::ATTENTION_KV_ROWS;
    let norm_values = heads * A::HEAD_DIM;
    let rotary_values = heads * ROTARY_PAIRS * 2;
    let metadata_values = 3;
    let query_output = A::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>();
    let cache_output = 2 * A::ATTENTION_KV_ROWS * cache_element_bytes;
    let per_token = (source_values + norm_values) * size_of::<u16>()
        + rotary_values * size_of::<f32>()
        + metadata_values * size_of::<u32>()
        + query_output
        + cache_output;

    tokens * per_token
}

#[derive(Clone, Copy)]
struct BenchmarkLabels {
    suite: &'static str,
    route: &'static str,
    weights: &'static str,
    cache: &'static str,
    cache_description: &'static str,
    workspace: &'static str,
    padding: &'static str,
}

const QWEN38_LABELS: BenchmarkLabels = BenchmarkLabels {
    suite: "bench-attention-qk-prepare",
    route: "attention_qk_prepare/norm_mrope_cache_append",
    weights: "attention_qk_prepare/weights",
    cache: "attention_qk_prepare/kv_cache",
    cache_description: "16 physical pages, four KV heads, 64 positions, E4M3 K/V",
    workspace: "attention_qk_prepare/address_stable_workspace",
    padding: "attention_qk_prepare/alignment_padding",
};

const QWEN35_LABELS: BenchmarkLabels = BenchmarkLabels {
    suite: "bench-qwen35-attention-qk-prepare",
    route: "qwen35_attention_qk_prepare/norm_mrope_cache_append",
    weights: "qwen35_attention_qk_prepare/weights",
    cache: "qwen35_attention_qk_prepare/kv_cache",
    cache_description: "16 physical pages, four KV heads, 64 positions, BF16 K/V",
    workspace: "qwen35_attention_qk_prepare/address_stable_workspace",
    padding: "qwen35_attention_qk_prepare/alignment_padding",
};

fn benchmark_target<O: BenchQkPrepareOp>(
    options: DeviceBenchmarkOptions,
    labels: BenchmarkLabels,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::<O>::new(options.launches_per_sample)?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    let weight_bytes = session.regions.weight_bytes();
    let cache_bytes = session.regions.cache_bytes();
    let workspace_bytes = session.arena.byte_len() - weight_bytes - cache_bytes - padding_bytes;
    memory.register_owned(
        labels.weights,
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "BF16 zero-centered query/key norm weights",
    )?;
    memory.register_owned(
        labels.cache,
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        labels.cache_description,
    )?;
    memory.register_owned(
        labels.workspace,
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_tokens=1024 QKV, rotary, separate compact-decode/contiguous-prefill metadata, and prepared query",
    )?;
    memory.register_owned(
        labels.padding,
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample, labels.route);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: labels.suite,
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph",
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

/// Measures every exact Qwen3.8 attention Q/K decode and prefill route.
pub fn benchmark_attention_qk_prepare(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<AttentionQkPrepareOp>(options, QWEN38_LABELS)
}

/// Measures every exact Qwen3.5 attention Q/K preparation batch.
pub fn benchmark_qwen35_attention_qk_prepare(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen35AttentionQkPrepareOp>(options, QWEN35_LABELS)
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, MAX_TOKENS, ROUTES, layout, logical_bytes};
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_norm_mrope_query_and_cache_append() {
        let heads = Qwen38_27B::NUM_ATTENTION_HEADS + Qwen38_27B::NUM_KV_HEADS;
        let per_token = (Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + 2 * Qwen38_27B::ATTENTION_KV_ROWS
            + heads * Qwen38_27B::HEAD_DIM)
            * 2
            + heads * 32 * 2 * 4
            + 3 * 4
            + Qwen38_27B::ATTENTION_OUTPUT_COLUMNS * 4
            + 2 * Qwen38_27B::ATTENTION_KV_ROWS;

        assert_eq!(logical_bytes::<Qwen38_27B>(1, size_of::<u8>()), per_token);
        for tokens in ROUTES {
            assert_eq!(
                logical_bytes::<Qwen38_27B>(tokens, size_of::<u8>()),
                tokens * per_token
            );
        }

        let qwen35_heads = Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS;
        let qwen35_per_token = (Qwen35_9B::ATTENTION_OUTPUT_COLUMNS
            + 2 * Qwen35_9B::ATTENTION_KV_ROWS
            + qwen35_heads * Qwen35_9B::HEAD_DIM)
            * 2
            + qwen35_heads * 32 * 2 * 4
            + 3 * 4
            + Qwen35_9B::ATTENTION_OUTPUT_COLUMNS * 4
            + 2 * Qwen35_9B::ATTENTION_KV_ROWS * size_of::<u16>();
        assert_eq!(
            logical_bytes::<Qwen35_9B>(1, size_of::<u16>()),
            qwen35_per_token
        );
    }

    #[test]
    fn arena_accounting_exposes_every_padding_byte() {
        let (qwen38_layout, regions) = layout::<Qwen38_27B>(MAX_TOKENS, size_of::<u8>()).unwrap();
        assert_eq!(qwen38_layout.byte_len(), 56_895_488);
        assert_eq!(regions.payload_bytes(), 56_895_040);
        assert_eq!(qwen38_layout.byte_len() - regions.payload_bytes(), 448);
        assert_eq!(MAX_TOKENS, 1_024);

        let (qwen35_layout, regions) = layout::<Qwen35_9B>(MAX_BATCH, size_of::<u16>()).unwrap();
        assert_eq!(qwen35_layout.byte_len(), 4_493_824);
        assert_eq!(regions.payload_bytes(), 4_492_928);
        assert_eq!(qwen35_layout.byte_len() - regions.payload_bytes(), 896);
    }

    #[test]
    fn qwen35_bf16_cache_accounting_matches_the_production_route() {
        let heads = Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS;
        let per_token = (Qwen35_9B::ATTENTION_OUTPUT_COLUMNS
            + 2 * Qwen35_9B::ATTENTION_KV_ROWS
            + heads * Qwen35_9B::HEAD_DIM)
            * size_of::<u16>()
            + heads * 32 * 2 * size_of::<f32>()
            + 3 * size_of::<u32>()
            + Qwen35_9B::ATTENTION_OUTPUT_COLUMNS * size_of::<f32>()
            + 2 * Qwen35_9B::ATTENTION_KV_ROWS * size_of::<u16>();
        for batch in 1..=MAX_BATCH {
            assert_eq!(
                logical_bytes::<Qwen35_9B>(batch, size_of::<u16>()),
                batch * per_token
            );
        }

        let (layout, regions) = layout::<Qwen35_9B>(MAX_BATCH, size_of::<u16>()).unwrap();
        assert_eq!(layout.byte_len(), 4_493_824);
        assert_eq!(regions.payload_bytes(), 4_492_928);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 896);
        assert_eq!(regions.cache_bytes(), 4_194_304);
    }
}

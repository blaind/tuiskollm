//! Direct timings for Qwen3.8-Flash-Next dense QSA attention and its sigmoid gate.
//!
//! Every case uses a fixed 256-token span inside the 2,051-token dense route.

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
    ATTENTION_PAGE_SIZE, Qwen38FlashNextAttentionGateOp, Qwen38FlashNextPagedGqaOp,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
/// Every width the Qwen3.8-Flash-Next QSA attention entries admit.
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_TOKENS];
const ALIGNMENT: usize = 256;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 16;
/// Sixteen pages of 64 positions cover the widest admitted prompt exactly.
const PHYSICAL_PAGES: usize = 16;
/// Visible span every timed route attends.
///
/// Four whole pages: the widest span the QSA qualification's decode fixture
/// exercises, held flat across decode and prompt widths so one route's timing
/// differs from another's only in its token count.
const CONTEXT_TOKENS: usize = 256;
const _: () = assert!(CONTEXT_TOKENS <= PHYSICAL_PAGES * ATTENTION_PAGE_SIZE);
const _: () = assert!(CONTEXT_TOKENS <= 2_051);
const KEY_SCALE: f32 = 32.0;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];

/// Represented E4M3 codes the cache planes are filled from.
const CACHE_CODES: [u8; 12] = [
    0x38, 0xb0, 0x30, 0x28, 0xa8, 0x20, 0xb8, 0x34, 0xac, 0x24, 0x3c, 0xa0,
];

/// FP32 query pattern. Sixteen values, mixed sign, all exactly representable.
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];

/// Gate values the packed `q_proj` half is filled from.
const GATE_PATTERN: [f32; 8] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0];

/// Query-half filler for the packed projection, deliberately unrelated to
/// [`GATE_PATTERN`] so the two halves never carry the same value.
const PACKED_QUERY_PATTERN: [f32; 8] = [3.0, 0.75, -2.5, -0.5, 1.5, -3.5, 0.125, -1.25];

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
    qkv: ArenaRegion<u16>,
    attention: ArenaRegion<f32>,
    activation: ArenaRegion<u16>,
}

impl Regions {
    /// Every byte the layout reserves, so the arena remainder is padding alone.
    fn payload_bytes(self) -> usize {
        self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.block_tables.byte_len()
            + self.table_rows.byte_len()
            + self.lengths.byte_len()
            + self.qkv.byte_len()
            + self.attention.byte_len()
            + self.activation.byte_len()
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
    qkv: *const u16,
    attention: *mut f32,
    activation: *mut u16,
}

struct RouteGraphs {
    tokens: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _attention_op: Qwen38FlashNextPagedGqaOp,
    _gate_op: Qwen38FlashNextAttentionGateOp,
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
        let attention_op = Qwen38FlashNextPagedGqaOp::new(&context)?;
        let gate_op = Qwen38FlashNextAttentionGateOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = ROUTES
            .into_iter()
            .map(|tokens| {
                capture_route(
                    &attention_op,
                    &gate_op,
                    &stream,
                    &addresses,
                    tokens,
                    repeated_operations,
                )
            })
            .collect::<GpuResult<Vec<_>>>()?;
        Ok(Self {
            routes,
            _attention_op: attention_op,
            _gate_op: gate_op,
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
                let (shape, mut workload) = if route.tokens <= MAX_BATCH {
                    (
                        format!("B={}", route.tokens),
                        BenchmarkWorkload::warm_operator_decode(route.tokens as u32),
                    )
                } else {
                    (
                        format!("T={}", route.tokens),
                        BenchmarkWorkload::warm_operator_prefill(route.tokens as u64),
                    )
                };
                workload.context_tokens = Some(CONTEXT_TOKENS as u64);
                ExactDeviceCase::new(
                    "qwen38_flash_next/qsa_attention/paged_gqa_sigmoid_gate",
                    shape,
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.tokens),
                        route.tokens as u64,
                        "token",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }

    fn padding_bytes(&self) -> usize {
        self.arena.byte_len() - self.regions.payload_bytes()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(
        MAX_TOKENS * Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS,
        ALIGNMENT,
    )?;
    let plane_bytes = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_TOKENS * Qwen38FlashNext::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let attention = layout.reserve(
        MAX_TOKENS * Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS,
        ALIGNMENT,
    )?;
    let activation = layout.reserve(
        MAX_TOKENS * Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS,
        ALIGNMENT,
    )?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            qkv,
            attention,
            activation,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let query = (0..MAX_TOKENS * columns)
        .map(|index| {
            let token = index / columns;
            QUERY_PATTERN[(index + 5 * token) & 15] * (1.0 - (token & 7) as f32 / 16.0)
        })
        .collect::<Vec<_>>();
    let plane_bytes = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let key_pages = (0..plane_bytes)
        .map(|index| CACHE_CODES[(index * 7 + 3) % CACHE_CODES.len()])
        .collect::<Vec<_>>();
    let value_pages = (0..plane_bytes)
        .map(|index| CACHE_CODES[(index * 5 + 1) % CACHE_CODES.len()])
        .collect::<Vec<_>>();
    let block_tables = (0..TABLE_ROWS)
        .flat_map(|row| {
            (0..TABLE_STRIDE).map(move |page| ((2 * row + page) % PHYSICAL_PAGES) as u32)
        })
        .collect::<Vec<_>>();
    // Eight distinct block-table rows, each a permutation of all sixteen
    // pages, so every token's visible span is mapped at any admitted width.
    let table_rows = (0..MAX_TOKENS)
        .map(|token| TABLE_ROW_IDS[token % MAX_BATCH])
        .collect::<Vec<_>>();
    let lengths = vec![CONTEXT_TOKENS as u32; MAX_TOKENS];
    // Different query and gate patterns expose a wrong packed-half read.
    let qkv = (0..MAX_TOKENS * Qwen38FlashNext::ATTENTION_QKV_ROWS)
        .map(|index| {
            let row = index % Qwen38FlashNext::ATTENTION_QKV_ROWS;
            if row >= Qwen38FlashNext::ATTENTION_QUERY_ROWS {
                // Key and value rows are unread by the gate; keep them inert.
                return f32_to_bf16(0.0);
            }
            let within_head = row % (2 * Qwen38FlashNext::HEAD_DIM);
            if within_head < Qwen38FlashNext::HEAD_DIM {
                f32_to_bf16(PACKED_QUERY_PATTERN[within_head & 7])
            } else {
                f32_to_bf16(GATE_PATTERN[(within_head - Qwen38FlashNext::HEAD_DIM) & 7])
            }
        })
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &block_tables)?;
    arena.copy_from_host(stream, regions.table_rows, &table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &lengths)?;
    arena.copy_from_host(stream, regions.qkv, &qkv)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        query: arena.address(regions.query)?,
        key_pages: arena.address(regions.key_pages)?,
        value_pages: arena.address(regions.value_pages)?,
        block_tables: arena.address(regions.block_tables)?,
        table_rows: arena.address(regions.table_rows)?,
        lengths: arena.address(regions.lengths)?,
        qkv: arena.address(regions.qkv)?,
        attention: arena.address(regions.attention)?,
        activation: arena.address(regions.activation)?,
    })
}

fn capture_route(
    attention_op: &Qwen38FlashNextPagedGqaOp,
    gate_op: &Qwen38FlashNextAttentionGateOp,
    stream: &CudaStream,
    addresses: &Addresses,
    tokens: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || {
        launch(attention_op, gate_op, stream, addresses, tokens)
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(attention_op, gate_op, stream, addresses, tokens)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        tokens,
        leaf,
        repeated,
    })
}

/// Launches the production order: paged E4M3 GQA, then the packed sigmoid gate
/// over the FP32 seam it just wrote.
fn launch(
    attention_op: &Qwen38FlashNextPagedGqaOp,
    gate_op: &Qwen38FlashNextAttentionGateOp,
    stream: &CudaStream,
    addresses: &Addresses,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: the session owns every plane this pair reads or writes; each is
    // aligned, disjoint, and outlives every captured graph replay, and the
    // block-table rows map the flat visible span of every timed width.
    unsafe {
        attention_op.launch(
            stream,
            tokens,
            addresses.query,
            addresses.key_pages,
            addresses.value_pages,
            addresses.block_tables,
            addresses.table_rows,
            TABLE_STRIDE,
            addresses.lengths,
            addresses.attention,
            KEY_SCALE,
            VALUE_SCALE,
        )?;
        gate_op.launch(
            stream,
            tokens,
            addresses.attention,
            addresses.qkv,
            addresses.activation,
        )
    }
}

fn logical_bytes(tokens: usize) -> usize {
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    // The partitioned decode entry gives every query head its own cache scan;
    // the shared-tile prompt entry feeds the twelve query heads one KV head
    // serves from a single 64-position tile.
    let cache_heads = if tokens <= MAX_BATCH {
        Qwen38FlashNext::NUM_ATTENTION_HEADS
    } else {
        Qwen38FlashNext::NUM_KV_HEADS
    };
    let attention_per_token = columns * size_of::<f32>()
        + columns * size_of::<f32>()
        + 2 * cache_heads * CONTEXT_TOKENS * Qwen38FlashNext::HEAD_DIM
        + 2 * size_of::<u32>()
        + cache_heads * CONTEXT_TOKENS * size_of::<u32>();
    // The gate reads the FP32 seam and the packed gate half, then republishes
    // the seam in place beside the BF16 activation.
    let gate_per_token = 2 * columns * size_of::<f32>() + 2 * columns * size_of::<u16>();

    tokens * (attention_per_token + gate_per_token)
}

/// Measures every exact Qwen3.8-Flash-Next QSA dense attention decode and prompt route.
pub fn benchmark_qwen38_flash_next_qsa_attention(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let padding_bytes = session.padding_bytes();
    let cache_bytes = session.regions.cache_bytes();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - padding_bytes;
    memory.register_owned(
        "qwen38_flash_next_qsa_attention/kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "16 physical pages, two KV heads, 64 positions, represented E4M3 K/V",
    )?;
    memory.register_owned(
        "qwen38_flash_next_qsa_attention/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_tokens=1024 query, packed projection, FP32 attention seam, BF16 activation, and page metadata",
    )?;
    memory.register_owned(
        "qwen38_flash_next_qsa_attention/alignment_padding",
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
            suite: "bench-qwen38-flash-next-qsa-attention",
            classification: "performance_sensitive_leaf_pair",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph at a 256-token context",
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
        CONTEXT_TOKENS, MAX_BATCH, MAX_TOKENS, PHYSICAL_PAGES, ROUTES, TABLE_ROWS, TABLE_STRIDE,
        layout, logical_bytes,
    };
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
    use tuisko_model::{Arch, Qwen38FlashNext};

    /// One token reads its FP32 query row and its share of the E4M3 cache,
    /// writes the FP32 seam, and the gate then reads that seam and the packed
    /// gate half to republish the seam beside a BF16 activation.
    #[test]
    fn qwen38_flash_next_qsa_attention_byte_accounting_covers_the_cache_scan_and_packed_gate() {
        let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
        let gate_per_token = 2 * columns * size_of::<f32>() + 2 * columns * size_of::<u16>();
        let decode_per_token = 2 * columns * size_of::<f32>()
            + 2 * Qwen38FlashNext::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * Qwen38FlashNext::HEAD_DIM
            + 2 * size_of::<u32>()
            + Qwen38FlashNext::NUM_ATTENTION_HEADS * CONTEXT_TOKENS * size_of::<u32>()
            + gate_per_token;
        let prompt_per_token = 2 * columns * size_of::<f32>()
            + 2 * Qwen38FlashNext::NUM_KV_HEADS * CONTEXT_TOKENS * Qwen38FlashNext::HEAD_DIM
            + 2 * size_of::<u32>()
            + Qwen38FlashNext::NUM_KV_HEADS * CONTEXT_TOKENS * size_of::<u32>()
            + gate_per_token;

        assert_eq!(gate_per_token, 73_728);
        assert_eq!(decode_per_token, 3_293_192);
        assert_eq!(prompt_per_token, 387_080);
        for tokens in ROUTES {
            let per_token = if tokens <= MAX_BATCH {
                decode_per_token
            } else {
                prompt_per_token
            };
            assert_eq!(logical_bytes(tokens), tokens * per_token);
        }
    }

    /// Pins route, arena, and byte-accounting inventories.
    #[test]
    fn qwen38_flash_next_qsa_attention_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(MAX_TOKENS, 1_024);
        assert_eq!(PHYSICAL_PAGES * ATTENTION_PAGE_SIZE, MAX_TOKENS);
        assert_eq!(TABLE_STRIDE, PHYSICAL_PAGES);
        assert_eq!(TABLE_ROWS * TABLE_STRIDE, 128);
        // The flat visible span is four whole pages, is mapped by every
        // block-table row, and stays inside the 2,051 dense-equivalence band.
        assert_eq!(CONTEXT_TOKENS, 4 * ATTENTION_PAGE_SIZE);

        let (layout, regions) = layout().unwrap();
        assert_eq!(regions.cache_bytes(), 1_048_576);
        assert_eq!(regions.query.byte_len(), 25_165_824);
        assert_eq!(regions.qkv.byte_len(), 27_262_976);
        assert_eq!(regions.attention.byte_len(), 25_165_824);
        assert_eq!(regions.activation.byte_len(), 12_582_912);
        assert_eq!(regions.payload_bytes(), 91_234_816);
        assert_eq!(layout.byte_len(), 91_234_816);
        // Every reserved plane is a whole multiple of the 256-byte alignment,
        // so the arena holds no padding at all: a nonzero remainder here would
        // mean `payload_bytes` had missed a region.
        for bytes in [
            regions.query.byte_len(),
            regions.key_pages.byte_len(),
            regions.value_pages.byte_len(),
            regions.block_tables.byte_len(),
            regions.table_rows.byte_len(),
            regions.lengths.byte_len(),
            regions.qkv.byte_len(),
            regions.attention.byte_len(),
            regions.activation.byte_len(),
        ] {
            assert_eq!(bytes % super::ALIGNMENT, 0);
        }
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }
}

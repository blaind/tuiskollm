//! Direct timings for the exact Qwen3.8-Flash-Next QSA selection graph routes.
//!
//! The four indexer stages are timed **separately**, not as one composed graph,
//! because they answer different questions and scale differently. The scorer
//! reads every candidate block and so grows linearly with the context; the
//! selection reads only the score plane; and the gather attention reads at most
//! 2,051 positions however long the context is, which is the whole point of the
//! route. Timing them together would hide exactly the shape a reader wants.
//!
//! Every route is driven at a context **above** the dense-equivalent ceiling,
//! so the radix select's outer passes run rather than its fast path. The ladder
//! is 32,768 / 131,072 / 262,144; the last is the pinned config's own
//! `max_position_embeddings`.

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
    ATTENTION_PAGE_SIZE, IndexerCompressArgs, IndexerPrepareArgs, IndexerSelectionArgs,
    Qwen38FlashNextIndexerPrepareOp, Qwen38FlashNextIndexerSelectionOp,
    Qwen38FlashNextSelectedPagedGqaOp, SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_SELECTED,
    SELECTION_RADIX_PASSES, SELECTION_ROW_TILE, SELECTION_SCRATCH_WORDS, SelectedAttentionArgs,
    selection_block_bucket, selection_ctas_per_row,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
const ALIGNMENT: usize = 256;
const RATIO: usize = Qwen38FlashNext::INDEXER_COMPRESS_RATIO;
const ROTARY_PAIRS: usize = 32;

/// Table rows the benchmark maps.
const TABLE_ROWS: usize = MAX_BATCH;
/// Entries between two block-table rows.
///
/// Row zero owns the whole 262,144-position span the pinned config admits; the
/// seven batch rows own a 32,768-position slot each, past the deep row so no
/// two rows address the same physical page. That keeps the batched shapes a
/// real cache regime rather than an aliased one.
const TABLE_STRIDE: usize = Qwen38FlashNext::MAX_POSITION_EMBEDDINGS / ATTENTION_PAGE_SIZE;
/// Pages the deep row owns.
const DEEP_PAGES: usize = TABLE_STRIDE;
/// Pages each batch row owns.
const BATCH_PAGES: usize = 32_768 / ATTENTION_PAGE_SIZE;
const PHYSICAL_PAGES: usize = DEEP_PAGES + (TABLE_ROWS - 1) * BATCH_PAGES;

const _: () = assert!(TABLE_STRIDE == 4_096);
const _: () = assert!(BATCH_PAGES == 512);
const _: () = assert!(PHYSICAL_PAGES == 7_680);

/// The context ladder every deep route walks.
const DEEP_CONTEXTS: [usize; 3] = [32_768, 131_072, 262_144];
/// Context the batched and prompt shapes hold, which every row can map.
const WIDE_CONTEXT: usize = 32_768;
/// Widths timed at [`WIDE_CONTEXT`] beside the deep `B=1` ladder.
const WIDE_ROUTES: [usize; 2] = [MAX_BATCH, MAX_TOKENS];

const KEY_SCALE: f32 = 32.0;
const VALUE_SCALE: f32 = 0.0625;

/// Represented E4M3 codes the cache planes are filled from.
const CACHE_CODES: [u8; 12] = [
    0x38, 0xb0, 0x30, 0x28, 0xa8, 0x20, 0xb8, 0x34, 0xac, 0x24, 0x3c, 0xa0,
];
const VALUE_CODES: [u8; 7] = [0x38, 0x30, 0x28, 0x34, 0x3c, 0x20, 0x24];
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
const INDEXER_PATTERN: [f32; 11] = [
    0.5, -0.25, 0.125, -0.75, 0.375, -0.125, 0.25, -0.5, 0.0625, -0.375, 0.75,
];
const NORM_PATTERN: [f32; 5] = [0.0, 0.25, -0.125, 0.5, -0.375];

/// One timed stage of the selection route.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Indexer query norm and rotation, plus the raw key append.
    Prepare,
    /// One round's block compression.
    Compress,
    /// The ReLU-sum scoring over every candidate block.
    Score,
    /// The radix select, tie-break and position expansion.
    Select,
    /// The gather attention over the selected positions.
    Attention,
}

impl Stage {
    const ALL: [Stage; 5] = [
        Stage::Prepare,
        Stage::Compress,
        Stage::Score,
        Stage::Select,
        Stage::Attention,
    ];

    fn operation(self) -> &'static str {
        match self {
            Stage::Prepare => "qwen38_flash_next/qsa_selection/indexer_prepare",
            Stage::Compress => "qwen38_flash_next/qsa_selection/block_compress",
            Stage::Score => "qwen38_flash_next/qsa_selection/block_score",
            Stage::Select => "qwen38_flash_next/qsa_selection/block_select",
            Stage::Attention => "qwen38_flash_next/qsa_selection/gather_attention",
        }
    }
}

#[derive(Clone, Copy)]
struct Regions {
    indexer_qk: ArenaRegion<u16>,
    query_norm: ArenaRegion<u16>,
    key_norm: ArenaRegion<u16>,
    rope_cos: ArenaRegion<f32>,
    rope_sin: ArenaRegion<f32>,
    block_rope_cos: ArenaRegion<f32>,
    block_rope_sin: ArenaRegion<f32>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    cache_positions: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
    block_counts: ArenaRegion<u32>,
    first_blocks: ArenaRegion<u32>,
    indexer_query: ArenaRegion<f32>,
    /// Per-sequence raw-key ring: one open micro-block per table row.
    raw_ring: ArenaRegion<u16>,
    /// Round-local raw keys, one row per position a prompt width carries.
    raw_round: ArenaRegion<u16>,
    block_keys: ArenaRegion<u16>,
    scores: ArenaRegion<f32>,
    select_scratch: ArenaRegion<u32>,
    selected: ArenaRegion<u32>,
    selected_counts: ArenaRegion<u32>,
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    attention: ArenaRegion<f32>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.indexer_qk.byte_len()
            + self.query_norm.byte_len()
            + self.key_norm.byte_len()
            + self.rope_cos.byte_len()
            + self.rope_sin.byte_len()
            + self.block_rope_cos.byte_len()
            + self.block_rope_sin.byte_len()
            + self.block_tables.byte_len()
            + self.table_rows.byte_len()
            + self.cache_positions.byte_len()
            + self.lengths.byte_len()
            + self.block_counts.byte_len()
            + self.first_blocks.byte_len()
            + self.indexer_query.byte_len()
            + self.raw_ring.byte_len()
            + self.raw_round.byte_len()
            + self.block_keys.byte_len()
            + self.scores.byte_len()
            + self.select_scratch.byte_len()
            + self.selected.byte_len()
            + self.selected_counts.byte_len()
            + self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.attention.byte_len()
    }

    /// Bytes the paged cache classes occupy: K, V, and the compressed block
    /// keys.
    ///
    /// The raw indexer keys are deliberately absent. They live in a four-slot
    /// per-sequence ring and a round-local plane, neither of which scales with
    /// the page pool, so counting them here would report a page cost the cache
    /// does not have.
    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len() + self.block_keys.byte_len()
    }

    /// Bytes the raw indexer keys occupy, which is not a per-page class.
    fn raw_key_bytes(self) -> usize {
        self.raw_ring.byte_len() + self.raw_round.byte_len()
    }
}

#[derive(Clone, Copy)]
struct Addresses {
    indexer_qk: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_rope_cos: *const f32,
    block_rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    block_counts: *const u32,
    first_blocks: *const u32,
    indexer_query: *mut f32,
    raw_ring: *mut u16,
    raw_round: *mut u16,
    block_keys: *mut u16,
    scores: *mut f32,
    select_scratch: *mut u32,
    selected: *mut u32,
    selected_counts: *mut u32,
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    attention: *mut f32,
}

/// One timed configuration: a width, a visible span, and the stage it drives.
///
/// `metadata` selects one of the four published metadata sets. Captured graphs
/// bake in addresses and not values, so a per-context ladder needs a resident
/// metadata set per context rather than one plane republished between cases.
#[derive(Clone, Copy)]
struct Route {
    tokens: usize,
    context: usize,
    stage: Stage,
    metadata: usize,
}

/// Metadata sets the fixture publishes: the three deep contexts a single row
/// spans, then the batched set whose eight rows each hold [`WIDE_CONTEXT`].
const METADATA_SETS: usize = DEEP_CONTEXTS.len() + 1;
/// The set index the batched and prompt shapes read.
const BATCH_METADATA: usize = DEEP_CONTEXTS.len();

struct RouteGraphs {
    route: Route,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _prepare_op: Qwen38FlashNextIndexerPrepareOp,
    _selection_op: Qwen38FlashNextIndexerSelectionOp,
    _attention_op: Qwen38FlashNextSelectedPagedGqaOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

/// Every timed configuration, deep ladder first.
fn routes() -> Vec<Route> {
    let mut routes = Vec::new();
    for stage in Stage::ALL {
        for (metadata, context) in DEEP_CONTEXTS.into_iter().enumerate() {
            routes.push(Route {
                tokens: 1,
                context,
                stage,
                metadata,
            });
        }
        for tokens in WIDE_ROUTES {
            routes.push(Route {
                tokens,
                context: WIDE_CONTEXT,
                stage,
                metadata: BATCH_METADATA,
            });
        }
    }

    routes
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
        let prepare_op = Qwen38FlashNextIndexerPrepareOp::new(&context)?;
        let selection_op = Qwen38FlashNextIndexerSelectionOp::new(&context)?;
        let attention_op = Qwen38FlashNextSelectedPagedGqaOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = routes()
            .into_iter()
            .map(|route| {
                capture_route(
                    &prepare_op,
                    &selection_op,
                    &attention_op,
                    &stream,
                    &addresses,
                    route,
                    repeated_operations,
                )
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _prepare_op: prepare_op,
            _selection_op: selection_op,
            _attention_op: attention_op,
            arena,
            regions,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: this Session owns every route graph and everything
                // they captured, and drops the graphs before the arena.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|graphs| {
                let route = graphs.route;
                let (width, mut workload) = if route.tokens <= MAX_BATCH {
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
                workload.context_tokens = Some(route.context as u64);
                ExactDeviceCase::new(
                    route.stage.operation(),
                    format!("{width} ctx={}", route.context),
                    workload,
                    OperationAccounting::new(logical_bytes(route), route.tokens as u64, "token"),
                    &graphs.leaf,
                    Some(RepeatedGraph::new(&graphs.repeated, repeated_operations)),
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
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let cache_plane = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let attention_plane = MAX_TOKENS * Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let maximum_blocks = DEEP_CONTEXTS[DEEP_CONTEXTS.len() - 1] / RATIO;

    let indexer_qk = layout.reserve(MAX_TOKENS * Qwen38FlashNext::INDEXER_ROWS, ALIGNMENT)?;
    let query_norm = layout.reserve(indexer_dim, ALIGNMENT)?;
    let key_norm = layout.reserve(indexer_dim, ALIGNMENT)?;
    let rope_cos = layout.reserve(MAX_TOKENS * ROTARY_PAIRS, ALIGNMENT)?;
    let rope_sin = layout.reserve(MAX_TOKENS * ROTARY_PAIRS, ALIGNMENT)?;
    let block_rope_cos = layout.reserve(257 * ROTARY_PAIRS, ALIGNMENT)?;
    let block_rope_sin = layout.reserve(257 * ROTARY_PAIRS, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(METADATA_SETS * MAX_TOKENS, ALIGNMENT)?;
    let cache_positions = layout.reserve(METADATA_SETS * MAX_TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(METADATA_SETS * MAX_TOKENS, ALIGNMENT)?;
    let block_counts = layout.reserve(METADATA_SETS * MAX_TOKENS, ALIGNMENT)?;
    let first_blocks = layout.reserve(METADATA_SETS * MAX_TOKENS, ALIGNMENT)?;
    let indexer_query = layout.reserve(
        MAX_TOKENS * Qwen38FlashNext::INDEXER_HEADS * indexer_dim,
        ALIGNMENT,
    )?;
    let raw_ring = layout.reserve(TABLE_ROWS * RATIO * indexer_dim, ALIGNMENT)?;
    let raw_round = layout.reserve(MAX_TOKENS.max(257 * RATIO) * indexer_dim, ALIGNMENT)?;
    let block_keys = layout.reserve(
        PHYSICAL_PAGES * SELECTION_BLOCKS_PER_PAGE * indexer_dim,
        ALIGNMENT,
    )?;
    let scores = layout.reserve(SELECTION_ROW_TILE * maximum_blocks, ALIGNMENT)?;
    let select_scratch = layout.reserve(SELECTION_SCRATCH_WORDS, ALIGNMENT)?;
    let selected = layout.reserve(MAX_TOKENS * SELECTION_MAX_SELECTED, ALIGNMENT)?;
    let selected_counts = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let query = layout.reserve(attention_plane, ALIGNMENT)?;
    let key_pages = layout.reserve(cache_plane, ALIGNMENT)?;
    let value_pages = layout.reserve(cache_plane, ALIGNMENT)?;
    let attention = layout.reserve(attention_plane, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            indexer_qk,
            query_norm,
            key_norm,
            rope_cos,
            rope_sin,
            block_rope_cos,
            block_rope_sin,
            block_tables,
            table_rows,
            cache_positions,
            lengths,
            block_counts,
            first_blocks,
            indexer_query,
            raw_ring,
            raw_round,
            block_keys,
            scores,
            select_scratch,
            selected,
            selected_counts,
            query,
            key_pages,
            value_pages,
            attention,
        },
    ))
}

/// Physical page a table row's logical page maps to.
///
/// Row zero owns the deep span; the batch rows follow it, disjointly.
fn physical_page(row: usize, page: usize) -> u32 {
    if row == 0 {
        page as u32
    } else {
        (DEEP_PAGES + (row - 1) * BATCH_PAGES + page.min(BATCH_PAGES - 1)) as u32
    }
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let indexer_rows = Qwen38FlashNext::INDEXER_ROWS;
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;

    let indexer_qk = (0..MAX_TOKENS * indexer_rows)
        .map(|index| f32_to_bf16(INDEXER_PATTERN[index % INDEXER_PATTERN.len()] * 0.5))
        .collect::<Vec<_>>();
    let norm = |offset: usize| {
        (0..indexer_dim)
            .map(|index| f32_to_bf16(NORM_PATTERN[(index + offset) % NORM_PATTERN.len()]))
            .collect::<Vec<_>>()
    };
    let rope = |scale: f64| {
        (0..MAX_TOKENS * ROTARY_PAIRS)
            .map(|index| (index as f64 * scale).cos() as f32)
            .collect::<Vec<_>>()
    };
    let block_rope = |scale: f64| {
        (0..257 * ROTARY_PAIRS)
            .map(|index| (index as f64 * scale).cos() as f32)
            .collect::<Vec<_>>()
    };
    let block_tables = (0..TABLE_ROWS * TABLE_STRIDE)
        .map(|index| physical_page(index / TABLE_STRIDE, index % TABLE_STRIDE))
        .collect::<Vec<_>>();
    let raw_ring = (0..TABLE_ROWS * RATIO * indexer_dim)
        .map(|index| f32_to_bf16(INDEXER_PATTERN[index % INDEXER_PATTERN.len()] * 0.25))
        .collect::<Vec<_>>();
    let raw_round = (0..MAX_TOKENS.max(257 * RATIO) * indexer_dim)
        .map(|index| f32_to_bf16(INDEXER_PATTERN[index % INDEXER_PATTERN.len()] * 0.25))
        .collect::<Vec<_>>();
    // The block-key plane is filled directly: this suite times the stages, and
    // the compression stage is one of them, so seeding it here keeps the scorer
    // and the gather independent of whether the compression has run.
    let block_keys = (0..PHYSICAL_PAGES * SELECTION_BLOCKS_PER_PAGE * indexer_dim)
        .map(|index| f32_to_bf16(INDEXER_PATTERN[(index + 3) % INDEXER_PATTERN.len()] * 0.25))
        .collect::<Vec<_>>();
    let query = (0..MAX_TOKENS * columns)
        .map(|index| QUERY_PATTERN[index & 15])
        .collect::<Vec<_>>();
    let cache_plane = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let key_pages = (0..cache_plane)
        .map(|index| CACHE_CODES[index % CACHE_CODES.len()])
        .collect::<Vec<_>>();
    let value_pages = (0..cache_plane)
        .map(|index| VALUE_CODES[index % VALUE_CODES.len()])
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.indexer_qk, &indexer_qk)?;
    arena.copy_from_host(stream, regions.query_norm, &norm(0))?;
    arena.copy_from_host(stream, regions.key_norm, &norm(2))?;
    arena.copy_from_host(stream, regions.rope_cos, &rope(0.013))?;
    arena.copy_from_host(stream, regions.rope_sin, &rope(0.017))?;
    arena.copy_from_host(stream, regions.block_rope_cos, &block_rope(0.013))?;
    arena.copy_from_host(stream, regions.block_rope_sin, &block_rope(0.017))?;
    arena.copy_from_host(stream, regions.block_tables, &block_tables)?;
    arena.copy_from_host(stream, regions.raw_ring, &raw_ring)?;
    arena.copy_from_host(stream, regions.raw_round, &raw_round)?;
    arena.copy_from_host(stream, regions.block_keys, &block_keys)?;
    arena.copy_from_host(stream, regions.query, &query)?;
    arena.copy_from_host(stream, regions.key_pages, &key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &value_pages)?;
    let (rows, positions, lengths, blocks, first) = metadata_planes();
    arena.copy_from_host(stream, regions.table_rows, &rows)?;
    arena.copy_from_host(stream, regions.cache_positions, &positions)?;
    arena.copy_from_host(stream, regions.lengths, &lengths)?;
    arena.copy_from_host(stream, regions.block_counts, &blocks)?;
    arena.copy_from_host(stream, regions.first_blocks, &first)?;

    Ok(())
}

/// Builds every metadata set the ladder reads.
///
/// Each set holds a flat visible span, so the widths are directly comparable
/// and the deep ladder measures context scaling alone. The three deep sets put
/// every token on the row that owns the 262,144-position span; the batched set
/// spreads them over the eight rows, each of which owns [`WIDE_CONTEXT`].
fn metadata_planes() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rows = Vec::with_capacity(METADATA_SETS * MAX_TOKENS);
    let mut positions = Vec::with_capacity(METADATA_SETS * MAX_TOKENS);
    let mut lengths = Vec::with_capacity(METADATA_SETS * MAX_TOKENS);
    let mut blocks = Vec::with_capacity(METADATA_SETS * MAX_TOKENS);
    for (set, context) in DEEP_CONTEXTS.into_iter().chain([WIDE_CONTEXT]).enumerate() {
        for token in 0..MAX_TOKENS {
            rows.push(if set == BATCH_METADATA {
                (token % TABLE_ROWS) as u32
            } else {
                0
            });
            positions.push((context - MAX_TOKENS + token) as u32);
            lengths.push(context as u32);
            blocks.push((context / RATIO) as u32);
        }
    }

    (
        rows,
        positions,
        lengths,
        blocks,
        vec![0u32; METADATA_SETS * MAX_TOKENS],
    )
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        indexer_qk: arena.address(regions.indexer_qk)?.cast_const(),
        query_norm: arena.address(regions.query_norm)?.cast_const(),
        key_norm: arena.address(regions.key_norm)?.cast_const(),
        rope_cos: arena.address(regions.rope_cos)?.cast_const(),
        rope_sin: arena.address(regions.rope_sin)?.cast_const(),
        block_rope_cos: arena.address(regions.block_rope_cos)?.cast_const(),
        block_rope_sin: arena.address(regions.block_rope_sin)?.cast_const(),
        block_tables: arena.address(regions.block_tables)?.cast_const(),
        table_rows: arena.address(regions.table_rows)?.cast_const(),
        cache_positions: arena.address(regions.cache_positions)?.cast_const(),
        lengths: arena.address(regions.lengths)?.cast_const(),
        block_counts: arena.address(regions.block_counts)?.cast_const(),
        first_blocks: arena.address(regions.first_blocks)?.cast_const(),
        indexer_query: arena.address(regions.indexer_query)?,
        raw_ring: arena.address(regions.raw_ring)?,
        raw_round: arena.address(regions.raw_round)?,
        block_keys: arena.address(regions.block_keys)?,
        scores: arena.address(regions.scores)?,
        select_scratch: arena.address(regions.select_scratch)?,
        selected: arena.address(regions.selected)?,
        selected_counts: arena.address(regions.selected_counts)?,
        query: arena.address(regions.query)?.cast_const(),
        key_pages: arena.address(regions.key_pages)?.cast_const(),
        value_pages: arena.address(regions.value_pages)?.cast_const(),
        attention: arena.address(regions.attention)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn capture_route(
    prepare_op: &Qwen38FlashNextIndexerPrepareOp,
    selection_op: &Qwen38FlashNextIndexerSelectionOp,
    attention_op: &Qwen38FlashNextSelectedPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    route: Route,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || {
        launch(
            prepare_op,
            selection_op,
            attention_op,
            stream,
            addresses,
            route,
        )
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(
                prepare_op,
                selection_op,
                attention_op,
                stream,
                addresses,
                route,
            )?;
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
    prepare_op: &Qwen38FlashNextIndexerPrepareOp,
    selection_op: &Qwen38FlashNextIndexerSelectionOp,
    attention_op: &Qwen38FlashNextSelectedPagedGqaOp,
    stream: &CudaStream,
    addresses: &Addresses,
    route: Route,
) -> GpuResult<()> {
    let blocks = route.context / RATIO;
    let set = route.metadata * MAX_TOKENS;
    // SAFETY: every metadata plane is `METADATA_SETS * MAX_TOKENS` wide, so a
    // set offset stays inside its own region.
    let table_rows = unsafe { addresses.table_rows.add(set) };
    let cache_positions = unsafe { addresses.cache_positions.add(set) };
    let lengths = unsafe { addresses.lengths.add(set) };
    let block_counts = unsafe { addresses.block_counts.add(set) };
    let first_blocks = unsafe { addresses.first_blocks.add(set) };
    // SAFETY: the session owns every plane these stages read or write; each is
    // aligned, disjoint, and outlives every captured graph replay, and every
    // table row maps the flat visible span of every timed configuration.
    unsafe {
        match route.stage {
            Stage::Prepare => prepare_op.launch_prepare(
                stream,
                route.tokens,
                IndexerPrepareArgs {
                    indexer_qk: addresses.indexer_qk,
                    query_norm: addresses.query_norm,
                    rope_cos: addresses.rope_cos,
                    rope_sin: addresses.rope_sin,
                    table_rows,
                    cache_positions,
                    query: addresses.indexer_query,
                    raw_keys: if route.tokens <= MAX_BATCH {
                        addresses.raw_ring
                    } else {
                        addresses.raw_round
                    },
                },
            ),
            Stage::Compress => prepare_op.launch_compress(
                stream,
                route.tokens,
                IndexerCompressArgs {
                    raw_keys: if route.tokens <= MAX_BATCH {
                        addresses.raw_ring.cast_const()
                    } else {
                        addresses.raw_round.cast_const()
                    },
                    key_norm: addresses.key_norm,
                    block_rope_cos: addresses.block_rope_cos,
                    block_rope_sin: addresses.block_rope_sin,
                    block_tables: addresses.block_tables,
                    table_rows,
                    table_stride: TABLE_STRIDE as u32,
                    first_blocks,
                    block_counts,
                    block_keys: addresses.block_keys,
                },
            ),
            Stage::Score | Stage::Select => {
                let tile = if route.tokens <= MAX_BATCH {
                    route.tokens
                } else {
                    SELECTION_ROW_TILE
                };
                let mut offset = 0usize;
                while offset < route.tokens {
                    let rows = tile.min(route.tokens - offset);
                    let args = IndexerSelectionArgs {
                        query: addresses.indexer_query.cast_const(),
                        block_keys: addresses.block_keys.cast_const(),
                        block_tables: addresses.block_tables,
                        table_rows,
                        table_stride: TABLE_STRIDE as u32,
                        visible_lengths: lengths,
                        block_counts,
                        scores: addresses.scores,
                        score_stride: (DEEP_CONTEXTS[DEEP_CONTEXTS.len() - 1] / RATIO) as u32,
                        selected: addresses.selected,
                        selected_counts: addresses.selected_counts,
                        scratch: addresses.select_scratch,
                    };
                    // The two stages are timed apart because only one of them
                    // grows with the context. A composed route calls `launch`,
                    // which runs the pair in the only admitted order.
                    if route.stage == Stage::Score {
                        selection_op.launch_score(stream, rows, offset, blocks, args)?;
                    } else {
                        selection_op.launch_select(stream, rows, offset, blocks, args)?;
                    }
                    offset += rows;
                }

                Ok(())
            }
            Stage::Attention => attention_op.launch(
                stream,
                route.tokens,
                SelectedAttentionArgs {
                    query: addresses.query,
                    key_pages: addresses.key_pages,
                    value_pages: addresses.value_pages,
                    block_tables: addresses.block_tables,
                    table_rows,
                    table_stride: TABLE_STRIDE as u32,
                    selected: addresses.selected.cast_const(),
                    selected_counts: addresses.selected_counts.cast_const(),
                    output: addresses.attention,
                    key_scale: KEY_SCALE,
                    value_scale: VALUE_SCALE,
                },
            ),
        }
    }
}

/// Rows one launch of the scoring and selection pair owns.
fn tile_rows(route: Route) -> usize {
    if route.tokens <= MAX_BATCH {
        route.tokens
    } else {
        SELECTION_ROW_TILE
    }
}

/// Bytes the partial-histogram plane of one row moves across the whole select.
///
/// Every pass publishes 256 bins per CTA of the row, and every reduction - the
/// three later passes and the expansion - reads that plane back. Each CTA of
/// the row reads the whole plane so that all of them derive the same digit from
/// the same integers; that re-read is 64 KiB of L2-resident traffic per CTA and
/// is left out here for the same reason a cache-served re-read is left out of
/// every other stage's model.
fn partial_bytes(rows: usize, blocks: usize) -> usize {
    let ctas = selection_ctas_per_row(rows, selection_block_bucket(blocks).unwrap_or(blocks));
    let plane = ctas * 256 * size_of::<u32>();

    2 * SELECTION_RADIX_PASSES * plane
}

/// Bytes one timed configuration moves, per stage.
///
/// The four stages have genuinely different byte models, and the point of the
/// route is that only one of them grows with the context while the attention
/// does not: writing one shared model would have hidden that.
fn logical_bytes(route: Route) -> usize {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let head_dim = Qwen38FlashNext::HEAD_DIM;
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let blocks = route.context / RATIO;
    let selected = SELECTION_MAX_SELECTED.min(route.context);

    let per_token = match route.stage {
        // The fused projection row in, the four prepared query heads and one
        // raw cached key out.
        Stage::Prepare => {
            Qwen38FlashNext::INDEXER_ROWS * size_of::<u16>()
                + Qwen38FlashNext::INDEXER_HEADS * indexer_dim * size_of::<f32>()
                + indexer_dim * size_of::<u16>()
        }
        // One round closes at most a quarter of its own positions, each pooling
        // four raw keys into one published block key.
        Stage::Compress => {
            (RATIO * indexer_dim * size_of::<u16>() + indexer_dim * size_of::<u16>()) / RATIO
        }
        // Every candidate block key, once per row, plus the four query heads.
        Stage::Score => {
            blocks * indexer_dim * size_of::<u16>()
                + Qwen38FlashNext::INDEXER_HEADS * indexer_dim * size_of::<f32>()
                + blocks * size_of::<f32>()
        }
        // Four radix passes over the score plane, then one expansion pass,
        // plus the partial histograms the split publishes and reduces.
        Stage::Select => {
            (SELECTION_RADIX_PASSES + 1) * blocks * size_of::<f32>()
                + partial_bytes(tile_rows(route), blocks)
                + selected * size_of::<u32>()
        }
        // The selected positions only: this is the number that does *not* grow
        // with the context, which is what the route buys.
        Stage::Attention => {
            2 * columns * size_of::<f32>()
                + 2 * Qwen38FlashNext::NUM_ATTENTION_HEADS * selected * head_dim
                + selected * size_of::<u32>()
        }
    };

    route.tokens * per_token
}

/// Times every exact Qwen3.8-Flash-Next QSA selection stage over its context ladder.
pub fn benchmark_qwen38_flash_next_qsa_selection(
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
    let raw_key_bytes = session.regions.raw_key_bytes();
    let workspace_bytes = session.arena.byte_len() - cache_bytes - padding_bytes - raw_key_bytes;
    memory.register_owned(
        "qwen38_flash_next_qsa_selection/kv_cache",
        BenchmarkMemoryKind::KvCache,
        cache_bytes,
        "7,680 physical pages: represented E4M3 K/V and BF16 block keys",
    )?;
    memory.register_owned(
        "qwen38_flash_next_qsa_selection/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_tokens=1024 query and attention planes, the indexer query plane, the 65,536-block score scratch, the selected-position plane, the raw-key ring and its round-local plane, and page metadata",
    )?;
    memory.register_owned(
        "qwen38_flash_next_qsa_selection/raw_indexer_keys",
        BenchmarkMemoryKind::Workspace,
        raw_key_bytes,
        "the four-slot per-sequence ring and one prompt round's raw keys, neither of which scales with the page pool",
    )?;
    memory.register_owned(
        "qwen38_flash_next_qsa_selection/alignment_padding",
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
            suite: "bench-qwen38_flash_next-qsa-selection",
            classification: "performance_sensitive_leaf_pair",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph, per indexer stage over a 32K/131K/262K context ladder",
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
        BATCH_PAGES, DEEP_CONTEXTS, DEEP_PAGES, PHYSICAL_PAGES, RATIO, Route, Stage, TABLE_ROWS,
        TABLE_STRIDE, WIDE_CONTEXT, WIDE_ROUTES, layout, logical_bytes, physical_page, routes,
    };
    use std::collections::BTreeSet;
    use tuisko_kernels_sm120::{
        ATTENTION_PAGE_SIZE, SELECTION_MAX_SELECTED, selection_block_bucket, selection_ctas_per_row,
    };
    use tuisko_model::{Arch, Qwen38FlashNext};

    #[test]
    fn qwen38_flash_next_qsa_selection_byte_accounting_separates_the_four_stages() {
        let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
        let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;

        // Derived a second way rather than read back from the function, so a
        // throughput number that quietly changed shape fails here.
        for context in DEEP_CONTEXTS {
            let blocks = context / RATIO;
            let route = |stage| Route {
                tokens: 1,
                context,
                stage,
                metadata: 0,
            };
            assert_eq!(
                logical_bytes(route(Stage::Prepare)),
                640 * 2 + 4 * 128 * 4 + 128 * 2
            );
            assert_eq!(
                logical_bytes(route(Stage::Score)),
                blocks * indexer_dim * 2 + 4 * indexer_dim * 4 + blocks * 4
            );
            // Stated rather than read back: every rung of the deep ladder maps
            // one row onto the widest prepared split, so all three publish and
            // reduce the same 64 KiB partial plane four times over.
            assert_eq!(
                selection_ctas_per_row(1, selection_block_bucket(blocks).unwrap()),
                64
            );
            assert_eq!(
                logical_bytes(route(Stage::Select)),
                5 * blocks * 4 + 8 * 64 * 1_024 + SELECTION_MAX_SELECTED * 4
            );
            assert_eq!(
                logical_bytes(route(Stage::Attention)),
                2 * columns * 4
                    + 2 * 24 * SELECTION_MAX_SELECTED * 256
                    + SELECTION_MAX_SELECTED * 4
            );
        }

        // The claim the route exists for: the scorer grows with the context and
        // the gather attention does not.
        let scan = |context: usize, stage| {
            logical_bytes(Route {
                tokens: 1,
                context,
                stage,
                metadata: 0,
            })
        };
        assert_eq!(
            scan(262_144, Stage::Score),
            8 * scan(32_768, Stage::Score) - 7 * (4 * indexer_dim * 4)
        );
        assert_eq!(
            scan(262_144, Stage::Attention),
            scan(32_768, Stage::Attention)
        );
    }

    #[test]
    fn qwen38_flash_next_qsa_selection_benchmark_inventory_and_accounting_are_exact() {
        assert_eq!(TABLE_STRIDE, Qwen38FlashNext::MAX_POSITION_EMBEDDINGS / 64);
        assert_eq!(DEEP_PAGES * ATTENTION_PAGE_SIZE, 262_144);
        assert_eq!(BATCH_PAGES * ATTENTION_PAGE_SIZE, WIDE_CONTEXT);
        assert_eq!(PHYSICAL_PAGES, DEEP_PAGES + (TABLE_ROWS - 1) * BATCH_PAGES);
        assert_eq!(DEEP_CONTEXTS, [32_768, 131_072, 262_144]);
        assert_eq!(WIDE_ROUTES, [8, 1_024]);

        // Every timed configuration sits above the dense-equivalent ceiling, so
        // the selection is doing real work in every one of them.
        let ceiling = Qwen38FlashNext::INDEXER_BUDGET + RATIO - 1;
        let all = routes();
        assert_eq!(
            all.len(),
            Stage::ALL.len() * (DEEP_CONTEXTS.len() + WIDE_ROUTES.len())
        );
        for route in &all {
            assert!(route.context > ceiling, "ctx {} is dense", route.context);
        }

        // No two table rows address the same physical page.
        let mut pages = BTreeSet::new();
        for row in 0..TABLE_ROWS {
            let span = if row == 0 { DEEP_PAGES } else { BATCH_PAGES };
            for page in 0..span {
                assert!(
                    pages.insert(physical_page(row, page)),
                    "row {row} page {page} aliases another row"
                );
            }
        }
        assert_eq!(pages.len(), PHYSICAL_PAGES);

        let (layout, regions) = layout().expect("the benchmark layout fits");
        // The two block-rotary planes are 257 rows of 32 pairs, which is 128
        // bytes short of the 256-byte region alignment each; every other region
        // is an exact multiple. Pinning the figure keeps a region that quietly
        // stopped being counted from hiding inside the remainder.
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 256);
        assert!(regions.cache_bytes() < layout.byte_len());

        // The raw indexer keys are not a page class. Their size is the ring plus
        // one prompt round, whatever the pool holds, which is the whole reason
        // `cache_bytes` no longer counts them.
        assert_eq!(
            regions.raw_key_bytes(),
            (TABLE_ROWS * RATIO + 257 * RATIO) * Qwen38FlashNext::INDEXER_HEAD_DIM * 2
        );
        assert!(regions.raw_key_bytes() < regions.cache_bytes() / 1_000);
    }
}

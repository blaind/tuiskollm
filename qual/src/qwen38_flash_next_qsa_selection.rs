//! Numerical and graph qualification for Qwen3.8-Flash-Next QSA selection.
//!
//! The suite proves dense-band bit identity through 2,051 positions, FP64
//! prepare/compress/score agreement above that band, exact top-512 selection,
//! per-sequence page-table isolation, and eager/CUDA Graph replay agreement.
//! Score ties select the lowest block index.

use crate::fp8_projection_oracle::{
    BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, f32_to_bf16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, IndexerCompressArgs, IndexerPrepareArgs, IndexerSelectionArgs,
    Qwen38FlashNextIndexerPrepareOp, Qwen38FlashNextIndexerSelectionOp, Qwen38FlashNextPagedGqaOp,
    Qwen38FlashNextSelectedPagedGqaOp, SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_SELECTED,
    SELECTION_ROW_TILE, SelectedAttentionArgs,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
const ALIGNMENT: usize = 256;
const TABLE_ROWS: usize = 8;
/// Eighty-eight pages of 64 positions give every row 5,632 cached positions.
///
/// That is past twice the dense-equivalent ceiling, so the radix select's outer
/// passes run rather than only its fast path, and it leaves room for the
/// append landing zone below without overlapping any scored block.
const TABLE_STRIDE: usize = 88;
const PHYSICAL_PAGES: usize = TABLE_ROWS * TABLE_STRIDE;
/// Cached positions one table row owns.
const ROW_CONTEXT: usize = TABLE_STRIDE * ATTENTION_PAGE_SIZE;
/// Tokens pooled into one micro-block.
const RATIO: usize = Qwen38FlashNext::INDEXER_COMPRESS_RATIO;
/// Blocks one table row can present.
const ROW_BLOCKS: usize = ROW_CONTEXT / RATIO;
/// Values between two rows of the score scratch.
const SCORE_STRIDE: usize = ROW_BLOCKS;
/// Blocks one compression launch closes, matching the widest prompt entry.
const COMPRESS_CHUNK: usize = 257;
/// Width whose compression entry owns one sequence and [`COMPRESS_CHUNK`] blocks.
const COMPRESS_ROUTE: usize = 1_024;

/// First position every `indexer_prepare` launch appends its raw key at.
///
/// The prepare stage writes the raw indexer key of the token it is given, which
/// would rewrite the synthetic history the block keys were pooled from. Landing
/// every append in the tail of the row keeps the history the selection stages
/// read untouched, and the const assertion below is what makes that a checked
/// property rather than an arrangement.
const APPEND_BASE: usize = ROW_CONTEXT - MAX_TOKENS;
/// Largest visible count any stage drives.
const MAX_VISIBLE: usize = 4_096;

// No block the selection ever scores may share a raw key with the landing zone.
const _: () = assert!(MAX_VISIBLE / RATIO < APPEND_BASE / RATIO);
const _: () = assert!(ROW_CONTEXT == 5_632);
const _: () = assert!(ROW_BLOCKS == 1_408);
const _: () = assert!(APPEND_BASE == 4_608);

const KEY_SCALE: f32 = 32.0;
const VALUE_SCALE: f32 = 0.0625;
/// `head_dim**-0.5` for head_dim 256.
const SOFTMAX_SCALE: f64 = 0.0625;
const RMS_EPSILON: f64 = 1.0e-6;
/// `1 / sqrt(128)`, applied after the ReLU sum.
const SCORE_SCALE: f64 = 0.088_388_347_648_318_45;

/// Contexts inside the dense band, where the two routes must agree bitwise.
///
/// `2050` and `2051` are the last two dense-equivalent counts; `2051` is the
/// sharp one, because `2052` is the first count whose `n_blocks` exceeds the
/// budget.
const DENSE_BAND_CONTEXTS: [usize; 8] = [1, 4, 63, 64, 65, 127, 2_050, 2_051];
/// Contexts above the band, where only the FP64 reference can judge.
///
/// `2052..2055` complete the residue class: the unconditional tail is
/// `V mod 4`, so all four residues have to appear above the ceiling or the tail
/// append is only ever tested at one width.
const SELECTED_CONTEXTS: [usize; 8] = [2_052, 2_053, 2_054, 2_055, 2_100, 3_000, 3_500, 4_096];
/// Prompt widths the gather attention admits.
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
/// First position of the above-budget prompt sweep.
///
/// Row `t` then sees `2048 + t + 1` visible positions, so the sweep starts one
/// row inside the dense band and crosses out of it at `t = 3`. One prompt tile
/// therefore carries both regimes, which is the case the composed route meets.
const PREFILL_SELECTED_BASE: usize = 2_048;

/// Token whose indexer projection is zero, so every score of its row ties at
/// `+0.0` and the pinned rule alone decides the whole selection.
const ZERO_QUERY_TOKEN: usize = 6;
/// Table row whose cached indexer keys are zero past [`PARTIAL_TIE_POSITION`].
const PARTIAL_TIE_ROW: usize = 7;
/// First position of [`PARTIAL_TIE_ROW`] whose cached indexer key is zero.
///
/// A multiple of the compression ratio, so no block straddles the boundary. At
/// 1,200 positions only 300 blocks carry a nonzero key, against a 512-block
/// budget, so at least 212 selected blocks come from the tie group and the
/// pinned rule decides which.
const PARTIAL_TIE_POSITION: usize = 1_200;
/// The two rows the isolation proof uses; their page maps are disjoint.
const ISOLATION_ROWS: [usize; 2] = [0, 1];
/// Acceptance tolerance for one published block score.
///
/// Deliberately loose: this comparison is a sanity check on the scoring
/// arithmetic, not the proof of the selection. The selection is judged exactly
/// against the device's own scores, so no tolerance sits anywhere near the
/// radix select, the tie-break, or the tail append.
const SCORE_TOLERANCE: f64 = 2.0e-3;

/// Rotary frequency pairs the model's 64-wide rotation carries.
const ROTARY_PAIRS: usize = 32;
/// Indexer dimensions the rotation touches; the rest pass through.
///
/// The width is the *model's*, not the indexer's: `partial_rotary_factor` is
/// `0.25` of the 256-wide attention head, and the indexer inherits the 64-wide
/// `cos`/`sin` unchanged, so half of its 128-wide vector rotates.
const ROTARY_DIM: usize = 64;

/// Represented E4M3 codes the key plane is filled from, mixed sign so the
/// scores spread rather than tracking the query norm.
const CACHE_CODES: [u8; 12] = [
    0x38, 0xb0, 0x30, 0x28, 0xa8, 0x20, 0xb8, 0x34, 0xac, 0x24, 0x3c, 0xa0,
];
/// Represented E4M3 codes the value plane is filled from, positive-biased so
/// the attention output does not cancel toward the acceptance floor.
const VALUE_CODES: [u8; 7] = [0x38, 0x30, 0x28, 0x34, 0x3c, 0x20, 0x24];
/// FP32 attention query pattern, mixed sign and exactly representable.
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
/// Indexer projection pattern.
///
/// Mixed sign, so ReLU actually discards heads: a same-sign pattern would make
/// the four-head sum a plain sum and the ReLU would never be observed. Eleven
/// entries against the 128-wide head and the 640-wide row keeps the pattern
/// from aligning with either stride.
const INDEXER_PATTERN: [f32; 11] = [
    0.5, -0.25, 0.125, -0.75, 0.375, -0.125, 0.25, -0.5, 0.0625, -0.375, 0.75,
];
/// Indexer RMSNorm weight pattern, in the `(1 + w)` form.
const NORM_PATTERN: [f32; 5] = [0.0, 0.25, -0.125, 0.5, -0.375];

/// Acceptance tolerance for a BF16-published indexer value.
const BF16_TOLERANCE: f64 = 8.0e-3;
/// Acceptance tolerance for an FP32 attention seam.
const FP32_TOLERANCE: f64 = 1.0e-3;

/// Failure of the Qwen3.8-Flash-Next QSA selection gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextQsaSelectionQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next QSA selection qualification failed: {0}")]
    Mismatch(String),
}

use Qwen38FlashNextQsaSelectionQualificationError as Failure;

type Outcome<T> = Result<T, Qwen38FlashNextQsaSelectionQualificationError>;

/// Counters the Qwen3.8-Flash-Next QSA selection gate proves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextQsaSelectionQualification {
    /// Block-key values compared against the FP64 pooling reference.
    pub block_key_values: usize,
    /// Block scores compared against the FP64 reference.
    pub score_values: usize,
    /// Selected positions compared entry for entry against the exact host
    /// selection over the device's own scores.
    pub selected_positions: usize,
    /// Rows whose selected list was the whole visible list.
    pub identity_rows: usize,
    /// Rows whose selection dropped at least one block.
    pub selective_rows: usize,
    /// Attention values proved bit-identical between the dense and the
    /// selection route inside the dense band.
    pub bit_identical_values: usize,
    /// Attention values compared against the FP64 reference above the band.
    pub attention_values: usize,
    /// Values proved unchanged when a foreign sequence's history was rewritten.
    pub isolated_values: usize,
    /// Values proved untouched outside the launched rows.
    pub untouched_values: usize,
    /// Input values proved unchanged across every launch.
    pub immutable_input_values: usize,
    /// Values compared between the eager launch and the graph replay.
    pub graph_replay_values: usize,
    /// Selected blocks the pinned tie-break admitted rather than a strictly
    /// greater score.
    pub tie_broken_blocks: usize,
    /// Blocks where the FP64 and device rankings disagreed, each proved to sit
    /// within the acceptance tolerance of the threshold.
    pub threshold_ambiguous_blocks: usize,
    /// Arena bytes the suite allocated.
    pub arena_bytes: usize,
    /// Alignment padding inside the arena.
    pub padding_bytes: usize,
    /// Largest absolute error against the FP64 reference.
    pub maximum_absolute_error: f32,
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
    indexer_pages: ArenaRegion<u16>,
    block_keys: ArenaRegion<u16>,
    scores: ArenaRegion<f32>,
    selected: ArenaRegion<u32>,
    selected_counts: ArenaRegion<u32>,
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    dense_attention: ArenaRegion<f32>,
    selected_attention: ArenaRegion<f32>,
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
            + self.indexer_pages.byte_len()
            + self.block_keys.byte_len()
            + self.scores.byte_len()
            + self.selected.byte_len()
            + self.selected_counts.byte_len()
            + self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
            + self.dense_attention.byte_len()
            + self.selected_attention.byte_len()
    }
}

struct Fixture {
    indexer_qk: Vec<u16>,
    query_norm: Vec<u16>,
    key_norm: Vec<u16>,
    /// Rotary rows at every absolute position, from which both the per-round
    /// query rows and the per-block rows are sliced.
    full_cos: Vec<f32>,
    full_sin: Vec<f32>,
    block_tables: Vec<u32>,
    /// Raw indexer keys the cache holds, `[pages, 64, 128]` BF16.
    indexer_pages: Vec<u16>,
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    /// The two cache planes decoded once in the reference's working precision.
    /// Every code is exactly representable, so this is a lossless restatement.
    key_values: Vec<f64>,
    value_values: Vec<f64>,
    /// The FP64 block key of every block of every table row, computed once
    /// because the score reference reads them per context and recomputing
    /// would dominate the suite.
    block_keys: Vec<Option<Vec<Vec<f64>>>>,
}

/// The ops one launch helper drives for both the eager path and the graph.
struct Ops {
    prepare: Qwen38FlashNextIndexerPrepareOp,
    selection: Qwen38FlashNextIndexerSelectionOp,
    dense: Qwen38FlashNextPagedGqaOp,
    selected: Qwen38FlashNextSelectedPagedGqaOp,
}

/// One round's per-row metadata.
#[derive(Clone)]
struct Round {
    tokens: usize,
    table_rows: Vec<u32>,
    positions: Vec<u32>,
    lengths: Vec<u32>,
}

impl Round {
    /// One decode step per row, each row its own sequence at its own context.
    fn decode(rows: &[usize], visible: &[usize]) -> Self {
        Self {
            tokens: rows.len(),
            table_rows: rows.iter().map(|&row| row as u32).collect(),
            positions: (0..rows.len())
                .map(|token| (APPEND_BASE + token) as u32)
                .collect(),
            lengths: visible.iter().map(|&length| length as u32).collect(),
        }
    }

    /// One prompt tile over a single sequence starting at `base`.
    fn prompt(row: usize, tokens: usize, base: usize) -> Self {
        Self {
            tokens,
            table_rows: vec![row as u32; tokens],
            positions: (0..tokens)
                .map(|token| (APPEND_BASE + token) as u32)
                .collect(),
            lengths: (0..tokens).map(|token| (base + token + 1) as u32).collect(),
        }
    }

    fn blocks(&self) -> Vec<u32> {
        self.lengths
            .iter()
            .map(|&length| length / RATIO as u32)
            .collect()
    }

    fn maximum_blocks(&self) -> usize {
        self.blocks().iter().copied().max().unwrap_or(0) as usize
    }
}

/// Qualifies the Qwen3.8-Flash-Next QSA selection route at every admitted width.
pub fn qualify_qwen38_flash_next_qsa_selection() -> Outcome<Qwen38FlashNextQsaSelectionQualification>
{
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Failure::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let ops = Ops {
        prepare: Qwen38FlashNextIndexerPrepareOp::new(&context)?,
        selection: Qwen38FlashNextIndexerSelectionOp::new(&context)?,
        dense: Qwen38FlashNextPagedGqaOp::new(&context)?,
        selected: Qwen38FlashNextSelectedPagedGqaOp::new(&context)?,
    };
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen38FlashNextQsaSelectionQualification {
        block_key_values: 0,
        score_values: 0,
        selected_positions: 0,
        identity_rows: 0,
        selective_rows: 0,
        bit_identical_values: 0,
        attention_values: 0,
        isolated_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        tie_broken_blocks: 0,
        threshold_ambiguous_blocks: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    // Stage one closes every micro-block the fixture's history completes and
    // proves each published block key against the FP64 pooling reference.
    compress_history(&ops, &arena, &stream, regions, &fixture, &mut report)?;

    // Stage two proves the raw append lands where the page mapping says and
    // carries the untouched projection row.
    verify_prepare(&ops, &arena, &stream, regions, &fixture, &mut report)?;

    // Stage three is the identity claim: inside the dense band the two routes
    // must agree bit for bit and the selected list must be the visible list.
    let rows = (0..TABLE_ROWS).collect::<Vec<_>>();
    run_round(
        &ops,
        &arena,
        &stream,
        regions,
        &fixture,
        &Round::decode(&rows, &DENSE_BAND_CONTEXTS),
        &mut report,
    )?;
    for &tokens in &PREFILL_ROUTES {
        let round = Round::prompt(0, tokens, 0);
        run_round(
            &ops,
            &arena,
            &stream,
            regions,
            &fixture,
            &round,
            &mut report,
        )?;
    }

    // Stage four is above the band, where the FP64 reference judges the scores
    // and the host's exact selection judges the list.
    run_round(
        &ops,
        &arena,
        &stream,
        regions,
        &fixture,
        &Round::decode(&rows, &SELECTED_CONTEXTS),
        &mut report,
    )?;
    for &tokens in &PREFILL_ROUTES {
        let round = Round::prompt(1, tokens, PREFILL_SELECTED_BASE);
        run_round(
            &ops,
            &arena,
            &stream,
            regions,
            &fixture,
            &round,
            &mut report,
        )?;
    }

    // Stage five: two sequences, one pool, disjoint page maps.
    verify_isolation(&ops, &arena, &stream, regions, &fixture, &mut report)?;

    if addresses(&arena, regions)? != stable_addresses {
        return Err(Failure::Mismatch(
            "device addresses changed while qualifying the selection route".into(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let cache_plane = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let attention_plane = MAX_TOKENS * Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;

    let indexer_qk = layout.reserve(MAX_TOKENS * Qwen38FlashNext::INDEXER_ROWS, ALIGNMENT)?;
    let query_norm = layout.reserve(indexer_dim, ALIGNMENT)?;
    let key_norm = layout.reserve(indexer_dim, ALIGNMENT)?;
    let rope_cos = layout.reserve(MAX_TOKENS * ROTARY_PAIRS, ALIGNMENT)?;
    let rope_sin = layout.reserve(MAX_TOKENS * ROTARY_PAIRS, ALIGNMENT)?;
    let block_rope_cos = layout.reserve(COMPRESS_CHUNK * ROTARY_PAIRS, ALIGNMENT)?;
    let block_rope_sin = layout.reserve(COMPRESS_CHUNK * ROTARY_PAIRS, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let cache_positions = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let block_counts = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let first_blocks = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let indexer_query = layout.reserve(
        MAX_TOKENS * Qwen38FlashNext::INDEXER_HEADS * indexer_dim,
        ALIGNMENT,
    )?;
    let indexer_pages = layout.reserve(
        PHYSICAL_PAGES * ATTENTION_PAGE_SIZE * indexer_dim,
        ALIGNMENT,
    )?;
    let block_keys = layout.reserve(
        PHYSICAL_PAGES * SELECTION_BLOCKS_PER_PAGE * indexer_dim,
        ALIGNMENT,
    )?;
    let scores = layout.reserve(SELECTION_ROW_TILE * SCORE_STRIDE, ALIGNMENT)?;
    let selected = layout.reserve(MAX_TOKENS * SELECTION_MAX_SELECTED, ALIGNMENT)?;
    let selected_counts = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let query = layout.reserve(attention_plane, ALIGNMENT)?;
    let key_pages = layout.reserve(cache_plane, ALIGNMENT)?;
    let value_pages = layout.reserve(cache_plane, ALIGNMENT)?;
    let dense_attention = layout.reserve(attention_plane, ALIGNMENT)?;
    let selected_attention = layout.reserve(attention_plane, ALIGNMENT)?;

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
            indexer_pages,
            block_keys,
            scores,
            selected,
            selected_counts,
            query,
            key_pages,
            value_pages,
            dense_attention,
            selected_attention,
        },
    ))
}

/// Rotary rows at one absolute position.
///
/// Three different position rows feed the `[11, 11, 10]` MRoPE partition, so a
/// kernel that collapsed the temporal, height and width rows would disagree.
fn rotary_row(position: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(ROTARY_PAIRS);
    let mut sin = Vec::with_capacity(ROTARY_PAIRS);
    for pair in 0..ROTARY_PAIRS {
        let row = match pair % 3 {
            1 => 3 * position + 17,
            2 if pair < 30 => 5 * position + 29,
            _ => position,
        } as f64;
        let inverse = (1.0e7f64).powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
        let angle = row * inverse;
        cos.push(angle.cos() as f32);
        sin.push(angle.sin() as f32);
    }

    (cos, sin)
}

/// Raw indexer key of one cached position, before the append landing zone.
fn history_key(row: usize, position: usize, dimension: usize) -> u16 {
    if row == PARTIAL_TIE_ROW && position >= PARTIAL_TIE_POSITION {
        return 0;
    }

    f32_to_bf16(
        INDEXER_PATTERN[(position + 7 * dimension + 5 * row) % INDEXER_PATTERN.len()] * 0.25,
    )
}

fn fixture() -> Fixture {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let indexer_rows = Qwen38FlashNext::INDEXER_ROWS;

    let mut full_cos = Vec::with_capacity(ROW_CONTEXT * ROTARY_PAIRS);
    let mut full_sin = Vec::with_capacity(ROW_CONTEXT * ROTARY_PAIRS);
    for position in 0..ROW_CONTEXT {
        let (cos, sin) = rotary_row(position);
        full_cos.extend_from_slice(&cos);
        full_sin.extend_from_slice(&sin);
    }

    let indexer_qk = (0..MAX_TOKENS * indexer_rows)
        .map(|index| {
            let token = index / indexer_rows;
            let column = index - token * indexer_rows;
            // The designated token's four query heads are exactly zero, so its
            // row's every block score is `+0.0` and the whole selection is the
            // tie-break's. Its key half stays populated: the append it drives
            // still has to carry a real projection row.
            if token % MAX_BATCH == ZERO_QUERY_TOKEN
                && column < Qwen38FlashNext::INDEXER_HEADS * indexer_dim
            {
                return 0;
            }
            f32_to_bf16(INDEXER_PATTERN[(index + 3 * token) % INDEXER_PATTERN.len()] * 0.5)
        })
        .collect::<Vec<_>>();
    let query_norm = (0..indexer_dim)
        .map(|index| f32_to_bf16(NORM_PATTERN[index % NORM_PATTERN.len()]))
        .collect::<Vec<_>>();
    let key_norm = (0..indexer_dim)
        .map(|index| f32_to_bf16(NORM_PATTERN[(index + 2) % NORM_PATTERN.len()]))
        .collect::<Vec<_>>();

    // Identity page partition: row `r` owns pages `[88r, 88r + 88)`, so no two
    // rows can address the same physical page.
    let block_tables = (0..TABLE_ROWS * TABLE_STRIDE)
        .map(|index| index as u32)
        .collect::<Vec<_>>();

    let mut indexer_pages = vec![0u16; PHYSICAL_PAGES * ATTENTION_PAGE_SIZE * indexer_dim];
    for row in 0..TABLE_ROWS {
        for position in 0..ROW_CONTEXT {
            let page = row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
            let base = indexer_dim * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * page);
            for dimension in 0..indexer_dim {
                indexer_pages[base + dimension] = history_key(row, position, dimension);
            }
        }
    }

    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let query = (0..MAX_TOKENS * columns)
        .map(|index| {
            let token = index / columns;
            QUERY_PATTERN[(index + 5 * token) & 15] * (1.0 - (token & 7) as f32 / 16.0)
        })
        .collect();
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
    let key_values = key_pages
        .iter()
        .map(|&code| {
            f64::from(decode_e4m3fn(code).expect("every CACHE_CODES entry is finite E4M3"))
                * KEY_SCALE as f64
        })
        .collect();
    let value_values = value_pages
        .iter()
        .map(|&code| {
            f64::from(decode_e4m3fn(code).expect("every VALUE_CODES entry is finite E4M3"))
                * VALUE_SCALE as f64
        })
        .collect();

    let mut partial = Fixture {
        indexer_qk,
        query_norm,
        key_norm,
        full_cos,
        full_sin,
        block_tables,
        indexer_pages,
        query,
        key_pages,
        value_pages,
        key_values,
        value_values,
        block_keys: vec![None; TABLE_ROWS],
    };
    for row in 0..TABLE_ROWS {
        let keys = (0..ROW_BLOCKS)
            .map(|block| reference_block_key(&partial, row, block))
            .collect::<Vec<_>>();
        partial.block_keys[row] = Some(keys);
    }

    partial
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.indexer_qk, &fixture.indexer_qk)?;
    arena.copy_from_host(stream, regions.query_norm, &fixture.query_norm)?;
    arena.copy_from_host(stream, regions.key_norm, &fixture.key_norm)?;
    arena.copy_from_host(stream, regions.block_tables, &fixture.block_tables)?;
    arena.copy_from_host(stream, regions.indexer_pages, &fixture.indexer_pages)?;
    arena.copy_from_host(stream, regions.query, &fixture.query)?;
    arena.copy_from_host(stream, regions.key_pages, &fixture.key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &fixture.value_pages)?;
    stream.synchronize().map_err(GpuError::from)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 24]> {
    Ok([
        arena.address(regions.indexer_qk)?.addr(),
        arena.address(regions.query_norm)?.addr(),
        arena.address(regions.key_norm)?.addr(),
        arena.address(regions.rope_cos)?.addr(),
        arena.address(regions.rope_sin)?.addr(),
        arena.address(regions.block_rope_cos)?.addr(),
        arena.address(regions.block_rope_sin)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.cache_positions)?.addr(),
        arena.address(regions.lengths)?.addr(),
        arena.address(regions.block_counts)?.addr(),
        arena.address(regions.first_blocks)?.addr(),
        arena.address(regions.indexer_query)?.addr(),
        arena.address(regions.indexer_pages)?.addr(),
        arena.address(regions.block_keys)?.addr(),
        arena.address(regions.scores)?.addr(),
        arena.address(regions.selected)?.addr(),
        arena.address(regions.selected_counts)?.addr(),
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.dense_attention)?.addr(),
        arena.address(regions.selected_attention)?.addr(),
    ])
}

/// Uploads one full-width metadata plane, padding the tail with a valid filler.
fn load_plane(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: ArenaRegion<u32>,
    values: &[u32],
    filler: u32,
) -> GpuResult<()> {
    let mut plane = vec![filler; region.len()];
    plane[..values.len()].copy_from_slice(values);
    arena.copy_from_host(stream, region, &plane)
}

fn f64_bf16(value: f64) -> f64 {
    bf16_to_f32(f32_to_bf16(value as f32)) as f64
}

/// Applies the partial NeoX rotation to a 128-wide indexer vector.
fn rotate(values: &[f64], fixture: &Fixture, position: usize) -> Vec<f64> {
    let half = ROTARY_DIM / 2;
    values
        .iter()
        .enumerate()
        .map(|(dimension, &value)| {
            if dimension >= ROTARY_DIM {
                return value;
            }
            let pair = dimension % half;
            let cosine = fixture.full_cos[position * ROTARY_PAIRS + pair] as f64;
            let sine = fixture.full_sin[position * ROTARY_PAIRS + pair] as f64;
            let rotated = if dimension < half {
                value * cosine - values[dimension + half] * sine
            } else {
                values[dimension - half] * sine + value * cosine
            };
            f64_bf16(rotated)
        })
        .collect()
}

/// The `(1 + w)` RMSNorm over one 128-wide head, rounded where the reference
/// materializes BF16.
fn normalize(values: &[f64], weights: &[u16]) -> Vec<f64> {
    let width = values.len();
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / width as f64;
    let inverse_rms = 1.0 / (mean_square + RMS_EPSILON).sqrt();
    values
        .iter()
        .enumerate()
        .map(|(dimension, value)| {
            let weight = bf16_to_f32(weights[dimension]) as f64;
            f64_bf16(value * inverse_rms * (1.0 + weight))
        })
        .collect()
}

/// The reference block key: FP32 mean of four raw keys, `(1 + w)` RMSNorm, and
/// the partial rotation at the block's first token.
fn reference_block_key(fixture: &Fixture, table_row: usize, block: usize) -> Vec<f64> {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let mut pooled = vec![0.0f64; indexer_dim];
    for member in 0..RATIO {
        let position = block * RATIO + member;
        let page = table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
        let base = indexer_dim * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * page);
        for (dimension, value) in pooled.iter_mut().enumerate() {
            *value += bf16_to_f32(fixture.indexer_pages[base + dimension]) as f64;
        }
    }
    for value in pooled.iter_mut() {
        *value = f64_bf16(*value * 0.25);
    }

    let normalized = normalize(&pooled, &fixture.key_norm);

    rotate(&normalized, fixture, block * RATIO)
}

/// The reference indexer query row for one head at one absolute position.
fn reference_query(fixture: &Fixture, token: usize, head: usize, position: usize) -> Vec<f64> {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let base = token * Qwen38FlashNext::INDEXER_ROWS + head * indexer_dim;
    let values = (0..indexer_dim)
        .map(|dimension| bf16_to_f32(fixture.indexer_qk[base + dimension]) as f64)
        .collect::<Vec<_>>();
    let normalized = normalize(&values, &fixture.query_norm);

    rotate(&normalized, fixture, position)
}

/// The reference block scores of one row.
fn reference_scores(
    fixture: &Fixture,
    table_row: usize,
    token: usize,
    visible: usize,
    position: usize,
) -> Vec<f64> {
    let blocks = visible / RATIO;
    let keys = fixture.block_keys[table_row]
        .as_ref()
        .expect("the reference rows precompute their block keys");
    let queries = (0..Qwen38FlashNext::INDEXER_HEADS)
        .map(|head| reference_query(fixture, token, head, position))
        .collect::<Vec<_>>();

    (0..blocks)
        .map(|block| {
            let key = &keys[block];
            let total = queries
                .iter()
                .map(|query| {
                    query
                        .iter()
                        .zip(key.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        .max(0.0)
                })
                .sum::<f64>();
            total * SCORE_SCALE
        })
        .collect()
}

/// The exact block-granular selection: top-512 under the pinned tie-break,
/// expanded to ascending positions, then the unconditional tail.
struct Selection {
    positions: Vec<u32>,
    /// Blocks admitted at the threshold rather than strictly above it.
    tie_broken: usize,
    /// The threshold score, when the selection was actually selective.
    threshold: Option<f64>,
}

fn select(scores: &[f64], visible: usize) -> Selection {
    let budget = Qwen38FlashNext::INDEXER_BUDGET / RATIO;
    let blocks = visible / RATIO;
    let mut positions = Vec::with_capacity(SELECTION_MAX_SELECTED);
    if blocks <= budget {
        positions.extend(0..visible as u32);
        return Selection {
            positions,
            tie_broken: 0,
            threshold: None,
        };
    }

    let mut order = (0..blocks).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        scores[right]
            .partial_cmp(&scores[left])
            .expect("scores are finite")
            .then(left.cmp(&right))
    });
    let threshold = scores[order[budget - 1]];
    let mut selected = order[..budget].to_vec();
    let tie_broken = selected
        .iter()
        .filter(|&&block| scores[block] == threshold)
        .count();
    selected.sort_unstable();
    for block in selected {
        positions.extend((0..RATIO).map(|member| (block * RATIO + member) as u32));
    }
    positions.extend((blocks * RATIO..visible).map(|position| position as u32));

    Selection {
        positions,
        tie_broken,
        threshold: Some(threshold),
    }
}

fn require_close(
    role: &str,
    index: usize,
    actual: f64,
    expected: f64,
    tolerance: f64,
    maximum: &mut f32,
) -> Outcome<()> {
    let error = (actual - expected).abs();
    *maximum = maximum.max(error as f32);
    let bound = tolerance.max(expected.abs() * tolerance);
    if error > bound {
        return Err(Failure::Mismatch(format!(
            "{role} value {index} was {actual} against {expected}, error {error} exceeds {bound}"
        )));
    }

    Ok(())
}

/// Closes every micro-block the fixture's cached history completes, then proves
/// each published block key against the FP64 pooling reference.
fn compress_history(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    for table_row in 0..TABLE_ROWS {
        let mut first = 0usize;
        while first < ROW_BLOCKS {
            let count = COMPRESS_CHUNK.min(ROW_BLOCKS - first);
            let mut cos = vec![0.0f32; COMPRESS_CHUNK * ROTARY_PAIRS];
            let mut sin = vec![0.0f32; COMPRESS_CHUNK * ROTARY_PAIRS];
            for slot in 0..count {
                let position = (first + slot) * RATIO;
                let source = position * ROTARY_PAIRS..(position + 1) * ROTARY_PAIRS;
                cos[slot * ROTARY_PAIRS..(slot + 1) * ROTARY_PAIRS]
                    .copy_from_slice(&fixture.full_cos[source.clone()]);
                sin[slot * ROTARY_PAIRS..(slot + 1) * ROTARY_PAIRS]
                    .copy_from_slice(&fixture.full_sin[source]);
            }
            arena.copy_from_host(stream, regions.block_rope_cos, &cos)?;
            arena.copy_from_host(stream, regions.block_rope_sin, &sin)?;
            load_plane(arena, stream, regions.table_rows, &[table_row as u32], 0)?;
            load_plane(arena, stream, regions.first_blocks, &[first as u32], 0)?;
            load_plane(arena, stream, regions.block_counts, &[count as u32], 0)?;
            stream.synchronize().map_err(GpuError::from)?;
            launch_compress(ops, arena, stream, regions)?;
            stream.synchronize().map_err(GpuError::from)?;
            first += count;
        }
    }

    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let observed = arena.copy_to_host(stream, regions.block_keys)?;
    for table_row in 0..TABLE_ROWS {
        let keys = fixture.block_keys[table_row]
            .as_ref()
            .expect("every row precomputes its block keys");
        let mut block = 0usize;
        while block < ROW_BLOCKS {
            let expected = &keys[block];
            let page = table_row * TABLE_STRIDE + block / SELECTION_BLOCKS_PER_PAGE;
            let base = indexer_dim
                * (page * SELECTION_BLOCKS_PER_PAGE + block % SELECTION_BLOCKS_PER_PAGE);
            for (dimension, want) in expected.iter().enumerate() {
                require_close(
                    "block key",
                    base + dimension,
                    bf16_to_f32(observed[base + dimension]) as f64,
                    *want,
                    BF16_TOLERANCE,
                    &mut report.maximum_absolute_error,
                )?;
                report.block_key_values += 1;
            }
            block += 1;
        }
    }

    Ok(())
}

/// Proves the raw append lands where the shared page mapping says, carries the
/// untouched projection row, and writes nothing else.
fn verify_prepare(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let rows = (0..MAX_BATCH).collect::<Vec<_>>();
    let round = Round::decode(&rows, &[64usize; MAX_BATCH]);
    load_round(arena, stream, regions, fixture, &round)?;
    arena.fill(stream, regions.indexer_query, BYTE_SENTINEL)?;
    stream.synchronize().map_err(GpuError::from)?;
    launch_prepare(ops, arena, stream, regions, round.tokens)?;
    stream.synchronize().map_err(GpuError::from)?;

    let pages = arena.copy_to_host(stream, regions.indexer_pages)?;
    for token in 0..round.tokens {
        let table_row = round.table_rows[token] as usize;
        let position = round.positions[token] as usize;
        let page = table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
        let base = indexer_dim * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * page);
        let source =
            token * Qwen38FlashNext::INDEXER_ROWS + Qwen38FlashNext::INDEXER_HEADS * indexer_dim;
        for dimension in 0..indexer_dim {
            // Bitwise: the cached vector is the raw projection row, so a norm
            // or a rotation anywhere on this path would move the bits.
            if pages[base + dimension] != fixture.indexer_qk[source + dimension] {
                return Err(Failure::Mismatch(format!(
                    "the appended indexer key at row {table_row} position {position} \
                     dimension {dimension} was not the raw projection row"
                )));
            }
            report.immutable_input_values += 1;
        }
    }

    // Everything outside the landing zone still holds the uploaded history.
    let mut untouched = 0usize;
    for (index, (&observed, &expected)) in
        pages.iter().zip(fixture.indexer_pages.iter()).enumerate()
    {
        let position_in_page = (index / indexer_dim) % ATTENTION_PAGE_SIZE;
        let page = index / (indexer_dim * ATTENTION_PAGE_SIZE);
        let position = (page % TABLE_STRIDE) * ATTENTION_PAGE_SIZE + position_in_page;
        if position < APPEND_BASE && observed != expected {
            return Err(Failure::Mismatch(format!(
                "the indexer prepare wrote outside its landing zone at value {index}"
            )));
        }
        if position < APPEND_BASE {
            untouched += 1;
        }
    }
    report.untouched_values += untouched;

    // The published query row is the norm then the rotation at the query's own
    // position, which is `visible - 1`.
    let query = arena.copy_to_host(stream, regions.indexer_query)?;
    for token in 0..round.tokens {
        let position = round.lengths[token] as usize - 1;
        for head in 0..Qwen38FlashNext::INDEXER_HEADS {
            let expected = reference_query(fixture, token, head, position);
            let base = (token * Qwen38FlashNext::INDEXER_HEADS + head) * indexer_dim;
            for (dimension, want) in expected.iter().enumerate() {
                require_close(
                    "indexer query",
                    base + dimension,
                    query[base + dimension] as f64,
                    *want,
                    BF16_TOLERANCE,
                    &mut report.maximum_absolute_error,
                )?;
            }
        }
    }

    // Restore the history so the block keys the later stages read stay the ones
    // this suite already proved.
    arena.copy_from_host(stream, regions.indexer_pages, &fixture.indexer_pages)?;
    stream.synchronize().map_err(GpuError::from)?;

    Ok(())
}

/// Publishes one round's metadata and its rotary rows.
fn load_round(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    round: &Round,
) -> GpuResult<()> {
    let mut cos = vec![0.0f32; MAX_TOKENS * ROTARY_PAIRS];
    let mut sin = vec![0.0f32; MAX_TOKENS * ROTARY_PAIRS];
    for token in 0..round.tokens {
        // The indexer query is rotated at the query's own absolute position,
        // which is one before its visible count.
        let position = round.lengths[token] as usize - 1;
        let source = position * ROTARY_PAIRS..(position + 1) * ROTARY_PAIRS;
        cos[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS]
            .copy_from_slice(&fixture.full_cos[source.clone()]);
        sin[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS]
            .copy_from_slice(&fixture.full_sin[source]);
    }
    arena.copy_from_host(stream, regions.rope_cos, &cos)?;
    arena.copy_from_host(stream, regions.rope_sin, &sin)?;
    load_plane(arena, stream, regions.table_rows, &round.table_rows, 0)?;
    load_plane(
        arena,
        stream,
        regions.cache_positions,
        &round.positions,
        APPEND_BASE as u32,
    )?;
    load_plane(arena, stream, regions.lengths, &round.lengths, 1)?;
    load_plane(arena, stream, regions.block_counts, &round.blocks(), 0)?;
    stream.synchronize().map_err(GpuError::from)
}

fn launch_prepare(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: the qualification arena establishes the complete pointer
    // contract; every plane is aligned, disjoint, and outlives the stream.
    unsafe {
        ops.prepare.launch_prepare(
            stream,
            tokens,
            IndexerPrepareArgs {
                indexer_qk: arena.address(regions.indexer_qk)?.cast_const(),
                query_norm: arena.address(regions.query_norm)?.cast_const(),
                rope_cos: arena.address(regions.rope_cos)?.cast_const(),
                rope_sin: arena.address(regions.rope_sin)?.cast_const(),
                block_tables: arena.address(regions.block_tables)?.cast_const(),
                table_rows: arena.address(regions.table_rows)?.cast_const(),
                table_stride: TABLE_STRIDE as u32,
                cache_positions: arena.address(regions.cache_positions)?.cast_const(),
                query: arena.address(regions.indexer_query)?,
                indexer_pages: arena.address(regions.indexer_pages)?,
            },
        )
    }
}

fn launch_compress(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    // SAFETY: as `launch_prepare`.
    unsafe {
        ops.prepare.launch_compress(
            stream,
            COMPRESS_ROUTE,
            IndexerCompressArgs {
                indexer_pages: arena.address(regions.indexer_pages)?.cast_const(),
                key_norm: arena.address(regions.key_norm)?.cast_const(),
                block_rope_cos: arena.address(regions.block_rope_cos)?.cast_const(),
                block_rope_sin: arena.address(regions.block_rope_sin)?.cast_const(),
                block_tables: arena.address(regions.block_tables)?.cast_const(),
                table_rows: arena.address(regions.table_rows)?.cast_const(),
                table_stride: TABLE_STRIDE as u32,
                first_blocks: arena.address(regions.first_blocks)?.cast_const(),
                block_counts: arena.address(regions.block_counts)?.cast_const(),
                block_keys: arena.address(regions.block_keys)?,
            },
        )
    }
}

/// Runs the whole selection pipeline for one round, exactly as a composed
/// route would: prepare, then one scoring and selection pair per row tile,
/// then both attention routes over the same planes.
fn launch_round(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    round: &Round,
) -> GpuResult<()> {
    launch_prepare(ops, arena, stream, regions, round.tokens)?;

    let maximum_blocks = round.maximum_blocks();
    let tile = if round.tokens <= MAX_BATCH {
        round.tokens
    } else {
        SELECTION_ROW_TILE.min(round.tokens)
    };
    let mut offset = 0usize;
    while offset < round.tokens {
        let rows = tile.min(round.tokens - offset);
        // SAFETY: as `launch_prepare`.
        unsafe {
            ops.selection.launch(
                stream,
                rows,
                offset,
                maximum_blocks,
                IndexerSelectionArgs {
                    query: arena.address(regions.indexer_query)?.cast_const(),
                    block_keys: arena.address(regions.block_keys)?.cast_const(),
                    block_tables: arena.address(regions.block_tables)?.cast_const(),
                    table_rows: arena.address(regions.table_rows)?.cast_const(),
                    table_stride: TABLE_STRIDE as u32,
                    visible_lengths: arena.address(regions.lengths)?.cast_const(),
                    block_counts: arena.address(regions.block_counts)?.cast_const(),
                    scores: arena.address(regions.scores)?,
                    score_stride: SCORE_STRIDE as u32,
                    selected: arena.address(regions.selected)?,
                    selected_counts: arena.address(regions.selected_counts)?,
                },
            )?;
        }
        offset += rows;
    }

    // SAFETY: as `launch_prepare`.
    unsafe {
        ops.dense.launch(
            stream,
            round.tokens,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.lengths)?,
            arena.address(regions.dense_attention)?,
            KEY_SCALE,
            VALUE_SCALE,
        )?;
        ops.selected.launch(
            stream,
            round.tokens,
            SelectedAttentionArgs {
                query: arena.address(regions.query)?.cast_const(),
                key_pages: arena.address(regions.key_pages)?.cast_const(),
                value_pages: arena.address(regions.value_pages)?.cast_const(),
                block_tables: arena.address(regions.block_tables)?.cast_const(),
                table_rows: arena.address(regions.table_rows)?.cast_const(),
                table_stride: TABLE_STRIDE as u32,
                selected: arena.address(regions.selected)?.cast_const(),
                selected_counts: arena.address(regions.selected_counts)?.cast_const(),
                output: arena.address(regions.selected_attention)?,
                key_scale: KEY_SCALE,
                value_scale: VALUE_SCALE,
            },
        )
    }
}

struct Observed {
    scores: Vec<f32>,
    selected: Vec<u32>,
    selected_counts: Vec<u32>,
    dense_attention: Vec<f32>,
    selected_attention: Vec<f32>,
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> Outcome<Observed> {
    Ok(Observed {
        scores: arena.copy_to_host(stream, regions.scores)?,
        selected: arena.copy_to_host(stream, regions.selected)?,
        selected_counts: arena.copy_to_host(stream, regions.selected_counts)?,
        dense_attention: arena.copy_to_host(stream, regions.dense_attention)?,
        selected_attention: arena.copy_to_host(stream, regions.selected_attention)?,
    })
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.indexer_query, BYTE_SENTINEL)?;
    arena.fill(stream, regions.scores, BYTE_SENTINEL)?;
    arena.fill(stream, regions.selected, BYTE_SENTINEL)?;
    arena.fill(stream, regions.selected_counts, BYTE_SENTINEL)?;
    arena.fill(stream, regions.dense_attention, BYTE_SENTINEL)?;
    arena.fill(stream, regions.selected_attention, BYTE_SENTINEL)
}

/// Drives one round through the whole pipeline, eagerly and then as a captured
/// graph, and proves every observable the round exposes.
fn run_round(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    round: &Round,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    load_round(arena, stream, regions, fixture, round)?;
    reset_outputs(arena, stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    launch_round(ops, arena, stream, regions, round)?;
    let eager = observe(arena, stream, regions)?;
    verify_round(fixture, round, &eager, report)?;
    verify_inputs(arena, stream, regions, fixture, report)?;

    reset_outputs(arena, stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    let graph = CudaGraph::capture(stream, || launch_round(ops, arena, stream, regions, round))?;
    // SAFETY: every allocation this graph captured is owned by the caller's
    // scope and outlives the replay and the synchronize that follows.
    unsafe { graph.launch(stream) }?;
    let replay = observe(arena, stream, regions)?;
    verify_replay(round, &eager, &replay, report)?;
    verify_inputs(arena, stream, regions, fixture, report)?;

    Ok(())
}

/// Compares the device's selected list against the exact host selection.
///
/// Below the budget the expectation is the whole visible list and the check is
/// unconditional. Above it the two lists must agree entry for entry, except
/// where the FP64 and device rankings straddle the threshold; every such block
/// is required to sit inside the acceptance tolerance of it, and is counted so
/// the allowance is visible rather than silent.
fn compare_selection(
    observed: &[u32],
    expected: &Selection,
    scores: &[f64],
    visible: usize,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    if observed.len() != expected.positions.len() {
        return Err(Failure::Mismatch(format!(
            "visible {visible} selected {} positions against {}",
            observed.len(),
            expected.positions.len()
        )));
    }
    if observed == expected.positions {
        return Ok(());
    }

    let Some(threshold) = expected.threshold else {
        return Err(Failure::Mismatch(format!(
            "visible {visible} is inside the dense band but its selected list was not the \
             visible list"
        )));
    };
    let blocks = visible / RATIO;
    let tail = blocks * RATIO;
    let observed_tail = observed.iter().copied().filter(|&p| p as usize >= tail);
    let expected_tail = expected
        .positions
        .iter()
        .copied()
        .filter(|&p| p as usize >= tail);
    if !observed_tail.eq(expected_tail) {
        return Err(Failure::Mismatch(format!(
            "visible {visible} did not append the unconditional tail"
        )));
    }

    let block_set = |list: &[u32]| {
        let mut set = vec![false; blocks];
        for &position in list.iter().filter(|&&p| (p as usize) < tail) {
            set[position as usize / RATIO] = true;
        }
        set
    };
    let got = block_set(observed);
    let want = block_set(&expected.positions);
    for block in 0..blocks {
        if got[block] == want[block] {
            continue;
        }
        let distance = (scores[block] - threshold).abs();
        let bound = SCORE_TOLERANCE.max(threshold.abs() * SCORE_TOLERANCE);
        if distance > bound {
            return Err(Failure::Mismatch(format!(
                "visible {visible} disagreed on block {block}, whose score {} is {distance} \
                 from the threshold {threshold} and so is not an admissible ranking difference",
                scores[block]
            )));
        }
        report.threshold_ambiguous_blocks += 1;
    }

    Ok(())
}

/// The FP64 attention over one row's selected positions.
///
/// A plain two-pass maximum-subtracted softmax, sharing no code with the
/// entries' online recurrence, which is what makes it independent.
fn reference_attention(
    fixture: &Fixture,
    table_row: usize,
    token: usize,
    head: usize,
    positions: &[u32],
) -> Vec<f64> {
    let head_dim = Qwen38FlashNext::HEAD_DIM;
    let kv_head = head / (Qwen38FlashNext::NUM_ATTENTION_HEADS / Qwen38FlashNext::NUM_KV_HEADS);
    let query_base = (token * Qwen38FlashNext::NUM_ATTENTION_HEADS + head) * head_dim;
    let element = |position: u32| {
        let position = position as usize;
        let page = table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
        head_dim
            * (position % ATTENTION_PAGE_SIZE
                + ATTENTION_PAGE_SIZE * (kv_head + Qwen38FlashNext::NUM_KV_HEADS * page))
    };

    let scores = positions
        .iter()
        .map(|&position| {
            let base = element(position);
            (0..head_dim)
                .map(|dimension| {
                    fixture.query[query_base + dimension] as f64
                        * fixture.key_values[base + dimension]
                })
                .sum::<f64>()
                * SOFTMAX_SCALE
        })
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = weights.iter().sum::<f64>();

    let mut output = vec![0.0f64; head_dim];
    for (index, &position) in positions.iter().enumerate() {
        let base = element(position);
        for (dimension, value) in output.iter_mut().enumerate() {
            *value += weights[index] * fixture.value_values[base + dimension];
        }
    }
    for value in output.iter_mut() {
        *value /= denominator;
    }

    output
}

/// Attention heads the FP64 reference compares above the dense band.
///
/// Two per KV head, at both ends of each group, so the `12:1` grouping is
/// exercised rather than assumed. A full 24-head comparison over 2,051
/// positions is quadratic in the fixture and buys no new coverage: the entry's
/// work on one head is independent of the others.
const VERIFIED_HEADS: [usize; 4] = [0, 11, 12, 23];
/// Rows the FP64 attention reference compares per above-band round.
const VERIFIED_ROWS: usize = 2;

fn verify_round(
    fixture: &Fixture,
    round: &Round,
    observed: &Observed,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let budget = Qwen38FlashNext::INDEXER_BUDGET / RATIO;
    let mut above_band = Vec::new();

    for token in 0..round.tokens {
        let visible = round.lengths[token] as usize;
        let blocks = visible / RATIO;
        let table_row = round.table_rows[token] as usize;
        let count = observed.selected_counts[token] as usize;
        if count == 0 || count > SELECTION_MAX_SELECTED {
            return Err(Failure::Mismatch(format!(
                "token {token} published a selected count of {count}"
            )));
        }
        let base = token * SELECTION_MAX_SELECTED;
        let list = &observed.selected[base..base + count];
        if list.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Failure::Mismatch(format!(
                "token {token} published a selected list that is not strictly ascending"
            )));
        }
        if list.iter().any(|&position| position as usize >= visible) {
            return Err(Failure::Mismatch(format!(
                "token {token} selected a position at or beyond its visible count {visible}"
            )));
        }

        let scores = reference_scores(fixture, table_row, token, visible, visible - 1);
        let expected = select(&scores, visible);
        compare_selection(list, &expected, &scores, visible, report)?;
        report.selected_positions += count;
        report.tie_broken_blocks += expected.tie_broken;
        if blocks <= budget {
            report.identity_rows += 1;
        } else {
            report.selective_rows += 1;
            above_band.push(token);
        }

        // The entry writes exactly `count` entries and nothing beyond them.
        for index in count..SELECTION_MAX_SELECTED {
            if observed.selected[base + index].to_le_bytes() != [BYTE_SENTINEL; 4] {
                return Err(Failure::Mismatch(format!(
                    "token {token} wrote past its selected count at entry {index}"
                )));
            }
            report.untouched_values += 1;
        }
    }

    // The identity claim. Inside the dense band the two routes read the same
    // planes and must land on the same bits; above it they must not, or the
    // selection would be doing nothing.
    let mut identical = 0usize;
    let mut differing_rows = 0usize;
    for token in 0..round.tokens {
        let span = token * columns..(token + 1) * columns;
        let dense = &observed.dense_attention[span.clone()];
        let selected = &observed.selected_attention[span];
        let same = dense
            .iter()
            .zip(selected.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits());
        if above_band.contains(&token) {
            if !same {
                differing_rows += 1;
            }
            continue;
        }
        if !same {
            let index = dense
                .iter()
                .zip(selected.iter())
                .position(|(left, right)| left.to_bits() != right.to_bits())
                .unwrap_or_default();
            return Err(Failure::Mismatch(format!(
                "token {token} at visible {} is inside the dense band, but the selection route \
                 published {} where the dense route published {} at column {index}",
                round.lengths[token], selected[index], dense[index]
            )));
        }
        identical += columns;
    }
    report.bit_identical_values += identical;
    if !above_band.is_empty() && differing_rows == 0 {
        return Err(Failure::Mismatch(
            "every above-budget row reproduced the dense route exactly, so the selection \
             dropped nothing and the fixture does not discriminate"
                .into(),
        ));
    }

    // Above the band the FP64 reference is the only judge, over the positions
    // the device itself selected.
    for &token in above_band.iter().take(VERIFIED_ROWS) {
        let table_row = round.table_rows[token] as usize;
        let base = token * SELECTION_MAX_SELECTED;
        let count = observed.selected_counts[token] as usize;
        let list = &observed.selected[base..base + count];
        for &head in &VERIFIED_HEADS {
            let expected = reference_attention(fixture, table_row, token, head, list);
            let output = token * columns + head * Qwen38FlashNext::HEAD_DIM;
            for (dimension, want) in expected.iter().enumerate() {
                require_close(
                    "selected attention",
                    output + dimension,
                    observed.selected_attention[output + dimension] as f64,
                    *want,
                    FP32_TOLERANCE,
                    &mut report.maximum_absolute_error,
                )?;
                report.attention_values += 1;
            }
        }
    }

    // The scoring itself, against the FP64 reference, for the tile the score
    // plane still holds.
    let tile = if round.tokens <= MAX_BATCH {
        round.tokens
    } else {
        SELECTION_ROW_TILE
    };
    let first = round.tokens - round.tokens.min(tile);
    for token in first..round.tokens {
        let visible = round.lengths[token] as usize;
        let table_row = round.table_rows[token] as usize;
        let scores = reference_scores(fixture, table_row, token, visible, visible - 1);
        let row = token - first;
        for (block, want) in scores.iter().enumerate() {
            require_close(
                "block score",
                row * SCORE_STRIDE + block,
                observed.scores[row * SCORE_STRIDE + block] as f64,
                *want,
                SCORE_TOLERANCE,
                &mut report.maximum_absolute_error,
            )?;
            report.score_values += 1;
        }
    }

    // Nothing outside the launched rows moved.
    for index in round.tokens * columns..MAX_TOKENS * columns {
        if observed.dense_attention[index].to_bits() != F32_SENTINEL_BITS
            || observed.selected_attention[index].to_bits() != F32_SENTINEL_BITS
        {
            return Err(Failure::Mismatch(format!(
                "a route wrote past its own rows at attention value {index}"
            )));
        }
        report.untouched_values += 2;
    }

    Ok(())
}

fn verify_replay(
    round: &Round,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    macro_rules! same {
        ($field:ident, $role:literal, $bits:expr) => {{
            if eager.$field.len() != replay.$field.len() {
                return Err(Failure::Mismatch(
                    concat!($role, " changed width on replay").into(),
                ));
            }
            for (index, (left, right)) in eager.$field.iter().zip(replay.$field.iter()).enumerate()
            {
                #[allow(clippy::redundant_closure_call)]
                if ($bits)(left) != ($bits)(right) {
                    return Err(Failure::Mismatch(format!(
                        concat!($role, " diverged on graph replay at value {} of tokens={}"),
                        index, round.tokens
                    )));
                }
                report.graph_replay_values += 1;
            }
        }};
    }

    same!(scores, "the score plane", |value: &f32| value.to_bits());
    same!(selected, "the selected plane", |value: &u32| *value);
    same!(selected_counts, "the selected counts", |value: &u32| *value);
    same!(dense_attention, "the dense attention", |value: &f32| value
        .to_bits());
    same!(
        selected_attention,
        "the selected attention",
        |value: &f32| { value.to_bits() }
    );

    Ok(())
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    macro_rules! unchanged {
        ($region:ident, $expected:expr, $role:literal) => {{
            let observed = arena.copy_to_host(stream, regions.$region)?;
            let expected: &[_] = $expected;
            if observed.len() != expected.len() || observed != expected {
                return Err(Failure::Mismatch(
                    concat!($role, " changed while the route ran").into(),
                ));
            }
            report.immutable_input_values += observed.len();
        }};
    }

    unchanged!(indexer_qk, &fixture.indexer_qk, "the indexer projection");
    unchanged!(query_norm, &fixture.query_norm, "the indexer query norm");
    unchanged!(key_norm, &fixture.key_norm, "the indexer key norm");
    unchanged!(block_tables, &fixture.block_tables, "the block table");
    unchanged!(query, &fixture.query, "the attention query");
    unchanged!(key_pages, &fixture.key_pages, "the key plane");
    unchanged!(value_pages, &fixture.value_pages, "the value plane");

    Ok(())
}

/// Proves that two sequences sharing one pool cannot reach each other.
///
/// Row one's cached history is rewritten wholesale; row zero's scores,
/// selection and attention output must not move by a bit. Their page maps are
/// disjoint, so a block key resolved inside the sequence's own mapping cannot
/// see the change, while an absolute-position lookup across the pool
/// would.
fn verify_isolation(
    ops: &Ops,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextQsaSelectionQualification,
) -> Outcome<()> {
    let indexer_dim = Qwen38FlashNext::INDEXER_HEAD_DIM;
    let round = Round::decode(&ISOLATION_ROWS, &[3_000, 3_000]);
    load_round(arena, stream, regions, fixture, &round)?;
    reset_outputs(arena, stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    launch_round(ops, arena, stream, regions, &round)?;
    let before = observe(arena, stream, regions)?;

    // Rewrite the foreign row's raw history, then close its blocks again so the
    // block-key plane follows. Both planes now differ everywhere row one owns.
    let foreign = ISOLATION_ROWS[1];
    let mut pages = fixture.indexer_pages.clone();
    for position in 0..ROW_CONTEXT {
        let page = foreign * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE;
        let base = indexer_dim * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * page);
        for dimension in 0..indexer_dim {
            pages[base + dimension] = f32_to_bf16(
                -INDEXER_PATTERN[(position + 3 * dimension) % INDEXER_PATTERN.len()] * 0.5,
            );
        }
    }
    arena.copy_from_host(stream, regions.indexer_pages, &pages)?;
    stream.synchronize().map_err(GpuError::from)?;
    let mut first = 0usize;
    while first < ROW_BLOCKS {
        let count = COMPRESS_CHUNK.min(ROW_BLOCKS - first);
        let mut cos = vec![0.0f32; COMPRESS_CHUNK * ROTARY_PAIRS];
        let mut sin = vec![0.0f32; COMPRESS_CHUNK * ROTARY_PAIRS];
        for slot in 0..count {
            let position = (first + slot) * RATIO;
            let source = position * ROTARY_PAIRS..(position + 1) * ROTARY_PAIRS;
            cos[slot * ROTARY_PAIRS..(slot + 1) * ROTARY_PAIRS]
                .copy_from_slice(&fixture.full_cos[source.clone()]);
            sin[slot * ROTARY_PAIRS..(slot + 1) * ROTARY_PAIRS]
                .copy_from_slice(&fixture.full_sin[source]);
        }
        arena.copy_from_host(stream, regions.block_rope_cos, &cos)?;
        arena.copy_from_host(stream, regions.block_rope_sin, &sin)?;
        load_plane(arena, stream, regions.table_rows, &[foreign as u32], 0)?;
        load_plane(arena, stream, regions.first_blocks, &[first as u32], 0)?;
        load_plane(arena, stream, regions.block_counts, &[count as u32], 0)?;
        stream.synchronize().map_err(GpuError::from)?;
        launch_compress(ops, arena, stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        first += count;
    }

    load_round(arena, stream, regions, fixture, &round)?;
    reset_outputs(arena, stream, regions)?;
    stream.synchronize().map_err(GpuError::from)?;
    launch_round(ops, arena, stream, regions, &round)?;
    let after = observe(arena, stream, regions)?;

    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let blocks = 3_000 / RATIO;
    for block in 0..blocks {
        if before.scores[block].to_bits() != after.scores[block].to_bits() {
            return Err(Failure::Mismatch(format!(
                "rewriting row {foreign}'s history moved row {}'s score at block {block}",
                ISOLATION_ROWS[0]
            )));
        }
        report.isolated_values += 1;
    }
    for index in 0..SELECTION_MAX_SELECTED {
        if before.selected[index] != after.selected[index] {
            return Err(Failure::Mismatch(format!(
                "rewriting row {foreign}'s history moved row {}'s selection at entry {index}",
                ISOLATION_ROWS[0]
            )));
        }
        report.isolated_values += 1;
    }
    if before.selected_counts[0] != after.selected_counts[0] {
        return Err(Failure::Mismatch(
            "rewriting a foreign row's history moved the observed row's selected count".into(),
        ));
    }
    for index in 0..columns {
        if before.selected_attention[index].to_bits() != after.selected_attention[index].to_bits() {
            return Err(Failure::Mismatch(format!(
                "rewriting row {foreign}'s history moved row {}'s attention at column {index}",
                ISOLATION_ROWS[0]
            )));
        }
        report.isolated_values += 1;
    }

    // The foreign row must actually have moved, or the fixture proved nothing.
    let foreign_base = SELECTION_MAX_SELECTED;
    let moved = (0..SELECTION_MAX_SELECTED)
        .any(|index| before.selected[foreign_base + index] != after.selected[foreign_base + index]);
    if !moved {
        return Err(Failure::Mismatch(
            "rewriting the foreign row's whole history left its own selection unchanged, so the \
             isolation fixture does not discriminate"
                .into(),
        ));
    }

    arena.copy_from_host(stream, regions.indexer_pages, &fixture.indexer_pages)?;
    stream.synchronize().map_err(GpuError::from)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        APPEND_BASE, DENSE_BAND_CONTEXTS, MAX_VISIBLE, PARTIAL_TIE_POSITION, PREFILL_ROUTES,
        PREFILL_SELECTED_BASE, RATIO, ROTARY_DIM, ROTARY_PAIRS, ROW_BLOCKS, ROW_CONTEXT,
        SCORE_STRIDE, SELECTED_CONTEXTS, TABLE_ROWS, TABLE_STRIDE, VERIFIED_HEADS, layout,
        rotary_row, select,
    };
    use tuisko_kernels_sm120::{
        ATTENTION_PAGE_SIZE, SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_SELECTED, SELECTION_ROW_TILE,
    };
    use tuisko_model::{Arch, Qwen38FlashNext};

    /// The block budget the reference's `indexer_budget // compress_ratio` gives.
    const BUDGET: usize = Qwen38FlashNext::INDEXER_BUDGET / RATIO;

    /// The dense-equivalent ceiling, derived rather than written down.
    const CEILING: usize = Qwen38FlashNext::INDEXER_BUDGET + RATIO - 1;

    #[test]
    fn the_selection_is_the_identity_exactly_up_to_the_ceiling() {
        assert_eq!(CEILING, 2_051);
        assert_eq!(BUDGET, 512);

        for visible in [1usize, 3, 4, 2_047, 2_048, 2_050, CEILING] {
            let scores = vec![0.0f64; visible / RATIO];
            let selection = select(&scores, visible);
            assert_eq!(
                selection.positions,
                (0..visible as u32).collect::<Vec<_>>(),
                "visible {visible} is inside the band and must select every position"
            );
            assert_eq!(selection.threshold, None);
            assert_eq!(selection.tie_broken, 0);
        }

        // One past the ceiling the top-k stops naming the whole axis, and the
        // count becomes the budget plus the tail rather than the visible span.
        for visible in [CEILING + 1, CEILING + 2, CEILING + 3, CEILING + 4] {
            let scores = vec![0.0f64; visible / RATIO];
            let selection = select(&scores, visible);
            assert_eq!(
                selection.positions.len(),
                Qwen38FlashNext::INDEXER_BUDGET + visible % RATIO,
                "visible {visible} selects the budget plus its own tail"
            );
            assert!(selection.positions.len() <= SELECTION_MAX_SELECTED);
        }
    }

    #[test]
    fn the_pinned_tie_break_takes_the_lowest_block_index() {
        // Every score tied: the ranking carries no information at all, so the
        // selection is entirely the tie-break's, and it must be the first 512
        // blocks and nothing else.
        let visible = MAX_VISIBLE;
        let blocks = visible / RATIO;
        let tied = select(&vec![0.0f64; blocks], visible);
        assert_eq!(tied.tie_broken, BUDGET);
        assert_eq!(tied.threshold, Some(0.0));
        assert_eq!(
            tied.positions,
            (0..(BUDGET * RATIO) as u32).collect::<Vec<_>>()
        );

        // A tie group straddling the cut: 300 blocks score, the rest tie at
        // zero, so 212 of the 512 selected blocks are the rule's choice.
        let mut scores = vec![0.0f64; blocks];
        let scored = (PARTIAL_TIE_POSITION / RATIO).min(blocks);
        assert_eq!(scored, 300);
        for (block, score) in scores.iter_mut().enumerate().take(scored) {
            *score = 1.0 + block as f64;
        }
        let straddling = select(&scores, visible);
        assert_eq!(straddling.tie_broken, BUDGET - scored);
        assert_eq!(straddling.tie_broken, 212);
        assert_eq!(
            straddling.positions,
            (0..(BUDGET * RATIO) as u32).collect::<Vec<_>>(),
            "the scored blocks and the lowest ties are both prefixes, so the union is one"
        );

        // And the rule is observable: a run that took the highest index instead
        // would have selected block `blocks - 1`, which this one does not.
        let last_block_first = ((blocks - 1) * RATIO) as u32;
        assert!(!straddling.positions.contains(&last_block_first));
    }

    #[test]
    fn a_selected_list_is_ascending_and_appends_its_own_tail() {
        let visible = 2_055usize;
        let blocks = visible / RATIO;
        let scores = (0..blocks)
            .map(|block| ((block * 37) % 1_009) as f64)
            .collect::<Vec<_>>();
        let selection = select(&scores, visible);

        assert!(selection.positions.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(selection.positions.len(), 2_048 + visible % RATIO);
        let tail = (blocks * RATIO..visible)
            .map(|p| p as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            selection.positions[selection.positions.len() - tail.len()..],
            tail[..],
            "the trailing tokens of the incomplete block are unconditional"
        );
        // Whole blocks only: a selected position implies its three siblings.
        for &position in &selection.positions {
            if (position as usize) < blocks * RATIO {
                let first = (position as usize / RATIO * RATIO) as u32;
                for member in 0..RATIO as u32 {
                    assert!(selection.positions.contains(&(first + member)));
                }
            }
        }
    }

    #[test]
    fn the_fixture_geometry_covers_every_admitted_width() {
        assert_eq!(ATTENTION_PAGE_SIZE, 64);
        assert_eq!(ROW_CONTEXT, TABLE_STRIDE * ATTENTION_PAGE_SIZE);
        assert_eq!(ROW_BLOCKS, ROW_CONTEXT / RATIO);
        assert_eq!(SCORE_STRIDE, ROW_BLOCKS);
        assert_eq!(SELECTION_BLOCKS_PER_PAGE, ATTENTION_PAGE_SIZE / RATIO);
        assert_eq!(SELECTION_ROW_TILE, 64);
        assert_eq!(PREFILL_ROUTES, [32, 64, 128, 1_024]);

        // Every context the suite drives is mapped, and every scored block sits
        // below the append landing zone.
        for context in DENSE_BAND_CONTEXTS.into_iter().chain(SELECTED_CONTEXTS) {
            assert!(
                context <= MAX_VISIBLE,
                "context {context} exceeds the sweep"
            );
            assert!(context / RATIO < APPEND_BASE / RATIO);
        }
        assert!(PREFILL_SELECTED_BASE + PREFILL_ROUTES[3] <= MAX_VISIBLE);
        assert!(DENSE_BAND_CONTEXTS.contains(&CEILING));
        assert_eq!(SELECTED_CONTEXTS[0], CEILING + 1);
        // All four tail residues appear above the ceiling.
        let residues = SELECTED_CONTEXTS
            .iter()
            .map(|context| context % RATIO)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(residues.len(), RATIO);

        // The verified heads straddle both KV groups.
        let group = Qwen38FlashNext::NUM_ATTENTION_HEADS / Qwen38FlashNext::NUM_KV_HEADS;
        let kv = VERIFIED_HEADS
            .iter()
            .map(|head| head / group)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(kv.len(), Qwen38FlashNext::NUM_KV_HEADS);

        let (layout, regions) = layout().expect("the fixture layout fits");
        // The two block-rotary planes are 257 rows of 32 pairs, which is 128
        // bytes short of the 256-byte region alignment each; every other region
        // is an exact multiple. Pinning the figure keeps a region that quietly
        // stopped being counted from hiding inside the remainder.
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 256);
        assert_eq!(TABLE_ROWS, 8);
    }

    #[test]
    fn the_rotary_fixture_carries_three_distinct_mrope_rows() {
        // `[11, 11, 10]` means slots 1, 4, .. carry the height row and 2, 5, ..
        // the width row; a fixture that collapsed them would not discriminate a
        // kernel that read the wrong one.
        let (cos, sin) = rotary_row(11);
        assert_eq!(cos.len(), ROTARY_PAIRS);
        assert_eq!(sin.len(), ROTARY_PAIRS);
        assert_eq!(ROTARY_DIM, 2 * ROTARY_PAIRS);
        let (base, _) = rotary_row(0);
        assert_ne!(cos[0].to_bits(), base[0].to_bits());
        assert_ne!(cos[1].to_bits(), cos[0].to_bits());
        assert_ne!(cos[2].to_bits(), cos[1].to_bits());
        // The temporal slots stay temporal.
        assert_eq!(cos[0].to_bits(), rotary_row(11).0[0].to_bits());
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_qsa_selection_matches_its_oracles_and_graph_replay()
    -> Result<(), super::Qwen38FlashNextQsaSelectionQualificationError> {
        let report = super::qualify_qwen38_flash_next_qsa_selection()?;
        println!("qwen38-flash-next-qsa-selection: {report:#?}");

        // Every block of every row is pooled, normed, rotated and compared.
        assert_eq!(
            report.block_key_values,
            TABLE_ROWS * ROW_BLOCKS * Qwen38FlashNext::INDEXER_HEAD_DIM
        );
        assert!(report.identity_rows > 0, "the dense band was never driven");
        assert!(report.selective_rows > 0, "the budget was never exceeded");
        assert!(
            report.bit_identical_values > 0,
            "no row proved the dense identity"
        );
        assert!(report.attention_values > 0);
        assert!(report.score_values > 0);
        assert!(report.selected_positions > 0);
        assert!(report.isolated_values > 0);
        assert!(report.untouched_values > 0);
        assert!(report.immutable_input_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(
            report.tie_broken_blocks > 0,
            "no selection was decided by the pinned tie-break, so the rule is untested"
        );
        assert!(
            report.maximum_absolute_error <= 0.05,
            "maximum absolute error {} against the FP64 reference",
            report.maximum_absolute_error
        );
        assert_eq!(
            report.arena_bytes - report.padding_bytes,
            layout().expect("the fixture layout fits").1.payload_bytes()
        );

        Ok(())
    }
}

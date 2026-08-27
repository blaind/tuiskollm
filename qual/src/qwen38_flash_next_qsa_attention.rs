//! Qualification for Qwen3.8-Flash-Next dense QSA attention and its sigmoid gate.
//!
//! The composed route must enforce a total visible length of at most 2,051,
//! where the QSA selection mask is the identity.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, f32_to_bf16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, Qwen38FlashNextAttentionGateOp, Qwen38FlashNextPagedGqaOp,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
/// Every width the Qwen3.8-Flash-Next QSA attention entries admit.
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 16;
/// Sixteen pages of 64 positions cover the widest admitted prompt exactly.
const PHYSICAL_PAGES: usize = 16;
/// Produces a measured 15x softmax-weight spread in the fixture.
const KEY_SCALE: f32 = 32.0;
const VALUE_SCALE: f32 = 0.0625;
/// `head_dim.powf(-0.5)` for head dimension 256.
const SOFTMAX_SCALE: f64 = 0.0625;
const DECODE_TABLE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
/// Decode context lengths that straddle every page boundary the cache has.
const DECODE_LENGTHS: [u32; MAX_BATCH] = [1, 64, 65, 127, 128, 129, 191, 256];
/// Widest prompt whose every row the FP64 oracle compares.
///
/// Above it the oracle samples rows instead; see [`verified_tokens`].
const FULLY_VERIFIED_TOKENS: usize = 128;
/// Row stride of the sampled comparison above [`FULLY_VERIFIED_TOKENS`].
const VERIFIED_TOKEN_STRIDE: usize = 32;

/// Mixed-sign represented E4M3 keys that spread attention scores.
const CACHE_CODES: [u8; 12] = [
    0x38, 0xb0, 0x30, 0x28, 0xa8, 0x20, 0xb8, 0x34, 0xac, 0x24, 0x3c, 0xa0,
];

/// Positive-biased represented values keep gate differences above tolerance.
const VALUE_CODES: [u8; 7] = [0x38, 0x30, 0x28, 0x34, 0x3c, 0x20, 0x24];

/// FP32 query pattern. Sixteen values, mixed sign, all exactly representable.
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];

/// Separates sigmoid from a dropped gate and SiLU, including the `z = 1` coincidence.
const GATE_PATTERN: [f32; 8] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0];

/// Query-half filler for the packed projection.
///
/// It is deliberately unrelated to [`GATE_PATTERN`] so an entry that read the
/// query half where it should read the gate half produces a different sigmoid.
const PACKED_QUERY_PATTERN: [f32; 8] = [3.0, 0.75, -2.5, -0.5, 1.5, -3.5, 0.125, -1.25];

/// Failure of the Qwen3.8-Flash-Next QSA dense attention gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextQsaAttentionQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next QSA attention qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact decode and prompt route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextQsaAttentionQualification {
    /// Attention output values compared with the independent FP64 formula.
    pub attention_values: usize,
    /// Gated FP32 and BF16 values compared with the independent formula.
    pub gated_values: usize,
    /// Inactive output words proved untouched.
    pub untouched_values: usize,
    /// Read-only input and metadata values proved unchanged.
    pub immutable_input_values: usize,
    /// Complete output state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Gated values that differ from the ungated attention beyond tolerance.
    ///
    /// A route whose gate was dropped, or replaced by a multiply by one, would
    /// leave this at zero.
    pub gate_separated_values: usize,
    /// Gated values a SiLU gate would have placed outside tolerance.
    ///
    /// `silu` and `sigmoid` agree only at `z = 1`, so a SiLU epilogue could
    /// not reach this count.
    pub silu_separated_values: usize,
    /// Values whose gate source is exactly zero, where `sigmoid` must halve.
    pub half_gated_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute attention or gated error.
    pub maximum_absolute_error: f32,
    /// Token rows compared value by value with the FP64 oracle.
    pub verified_tokens: usize,
    /// Minimum softmax-weight spread over rows with at least one visible page.
    pub minimum_softmax_ratio: f32,
}

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
}

struct Fixture {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    /// Lossless decoded cache values reused by the FP64 oracle.
    key_values: Vec<f64>,
    value_values: Vec<f64>,
    block_tables: Vec<u32>,
    decode_table_rows: Vec<u32>,
    decode_lengths: Vec<u32>,
    prefill_table_rows: Vec<u32>,
    prefill_lengths: Vec<u32>,
    qkv: Vec<u16>,
}

struct Observed {
    attention: Vec<f32>,
    activation: Vec<u16>,
}

/// Qualifies the Qwen3.8-Flash-Next QSA dense attention route at every admitted width.
pub fn qualify_qwen38_flash_next_qsa_attention()
-> Result<Qwen38FlashNextQsaAttentionQualification, Qwen38FlashNextQsaAttentionQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let attention_op = Qwen38FlashNextPagedGqaOp::new(&context)?;
    let gate_op = Qwen38FlashNextAttentionGateOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen38FlashNextQsaAttentionQualification {
        attention_values: 0,
        gated_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        gate_separated_values: 0,
        silu_separated_values: 0,
        half_gated_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
        minimum_softmax_ratio: f32::INFINITY,
        verified_tokens: 0,
    };

    for &tokens in &ROUTES {
        // Table rows and lengths are the only per-width operands, so they are
        // republished before each route and left addressable for the replay.
        load_metadata(&arena, &stream, regions, &fixture, tokens)?;
        reset_outputs(&arena, &stream, regions)?;
        launch(&attention_op, &gate_op, &arena, &stream, regions, tokens)?;
        let eager = observe(&arena, &stream, regions)?;
        let expected = oracle(tokens, &fixture)?;
        verify(tokens, &expected, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&attention_op, &gate_op, &arena, &stream, regions, tokens)
        })?;
        // SAFETY: every allocation this graph captured is owned by this scope
        // and outlives the replay and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
                format!("device addresses changed while qualifying tokens={tokens}"),
            ));
        }
    }

    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
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

fn fixture() -> Fixture {
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let query = (0..MAX_TOKENS * columns)
        .map(|index| {
            let token = index / columns;
            QUERY_PATTERN[(index + 5 * token) & 15] * (1.0 - (token & 7) as f32 / 16.0)
        })
        .collect();
    let plane_bytes = PHYSICAL_PAGES
        * Qwen38FlashNext::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * Qwen38FlashNext::HEAD_DIM;
    let key_pages = (0..plane_bytes)
        .map(|index| CACHE_CODES[(index * 7 + 3) % CACHE_CODES.len()])
        .collect();
    let value_pages = (0..plane_bytes)
        .map(|index| VALUE_CODES[(index * 5 + 1) % VALUE_CODES.len()])
        .collect();

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
        .collect();

    let decode_plane = |plane: &Vec<u8>| {
        plane
            .iter()
            .map(|&code| {
                f64::from(decode_e4m3fn(code).expect("every CACHE_CODES entry is finite E4M3"))
            })
            .collect::<Vec<_>>()
    };
    let key_values = decode_plane(&key_pages);
    let value_values = decode_plane(&value_pages);

    Fixture {
        query,
        key_pages,
        value_pages,
        key_values,
        value_values,
        block_tables: (0..TABLE_ROWS)
            .flat_map(|row| {
                (0..TABLE_STRIDE).map(move |page| ((2 * row + page) % PHYSICAL_PAGES) as u32)
            })
            .collect(),
        // Full-width valid metadata satisfies the arena copy contract.
        decode_table_rows: (0..MAX_TOKENS)
            .map(|token| DECODE_TABLE_ROWS[token.min(MAX_BATCH - 1)])
            .collect(),
        decode_lengths: (0..MAX_TOKENS)
            .map(|token| {
                if token < MAX_BATCH {
                    DECODE_LENGTHS[token]
                } else {
                    1
                }
            })
            .collect(),
        // Every prompt token shares block-table row zero and sees exactly the
        // positions at or before its own: dense causal visibility.
        prefill_table_rows: vec![0; MAX_TOKENS],
        prefill_lengths: (1..=MAX_TOKENS as u32).collect(),
        qkv,
    }
}

/// Compares every row through T=128 and samples T=1024 at stride 32, including
/// the last row, to bound the quadratic FP64 oracle.
fn verified_tokens(tokens: usize) -> Vec<usize> {
    if tokens <= FULLY_VERIFIED_TOKENS {
        return (0..tokens).collect();
    }

    (0..tokens)
        .filter(|token| token % VERIFIED_TOKEN_STRIDE == VERIFIED_TOKEN_STRIDE - 1)
        .chain(std::iter::once(tokens - 1))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn metadata(fixture: &Fixture, tokens: usize) -> (&[u32], &[u32]) {
    if tokens <= MAX_BATCH {
        (&fixture.decode_table_rows, &fixture.decode_lengths)
    } else {
        (&fixture.prefill_table_rows, &fixture.prefill_lengths)
    }
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.query, &fixture.query)?;
    arena.copy_from_host(stream, regions.key_pages, &fixture.key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &fixture.value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &fixture.block_tables)?;
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    stream.synchronize().map_err(GpuError::from)
}

/// Publishes the metadata one width reads, then leaves it addressable.
fn load_metadata(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    tokens: usize,
) -> GpuResult<()> {
    let (table_rows, lengths) = metadata(fixture, tokens);
    arena.copy_from_host(stream, regions.table_rows, table_rows)?;
    arena.copy_from_host(stream, regions.lengths, lengths)?;
    stream.synchronize().map_err(GpuError::from)
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.attention, BYTE_SENTINEL)?;
    arena.fill(stream, regions.activation, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 9]> {
    Ok([
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.lengths)?.addr(),
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.attention)?.addr(),
        arena.address(regions.activation)?.addr(),
    ])
}

fn launch(
    attention_op: &Qwen38FlashNextPagedGqaOp,
    gate_op: &Qwen38FlashNextAttentionGateOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: the qualification arena establishes the complete pointer
    // contract; every plane is aligned, disjoint, and outlives the stream.
    unsafe {
        attention_op.launch(
            stream,
            tokens,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.lengths)?,
            arena.address(regions.attention)?,
            KEY_SCALE,
            VALUE_SCALE,
        )?;
        gate_op.launch(
            stream,
            tokens,
            arena.address(regions.attention)?,
            arena.address(regions.qkv)?,
            arena.address(regions.activation)?,
        )
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<Observed, Qwen38FlashNextQsaAttentionQualificationError> {
    Ok(Observed {
        attention: arena.copy_to_host(stream, regions.attention)?,
        activation: arena.copy_to_host(stream, regions.activation)?,
    })
}

/// One route's independent expectation, over the compared token rows only.
struct Expected {
    /// Token rows this expectation covers, ascending.
    tokens: Vec<usize>,
    /// Ungated attention output, `[tokens.len(), 6144]`.
    attention: Vec<f64>,
    /// Gate source per output column, `[tokens.len(), 6144]`.
    gate: Vec<f64>,
    /// Smallest largest-to-smallest softmax weight ratio over all rows.
    softmax_ratio: f64,
}

fn cache_offset(physical_page: usize, kv_head: usize, position: usize, dimension: usize) -> usize {
    Qwen38FlashNext::HEAD_DIM
        * ((position & (ATTENTION_PAGE_SIZE - 1))
            + ATTENTION_PAGE_SIZE * (kv_head + Qwen38FlashNext::NUM_KV_HEADS * physical_page))
        + dimension
}

/// Independent FP64 dense causal attention plus the packed gate source.
///
/// Nothing here shares code with the device: the softmax is a plain two-pass
/// maximum-subtracted sum in `f64`, not the entries' online recurrence.
fn oracle(
    tokens: usize,
    fixture: &Fixture,
) -> Result<Expected, Qwen38FlashNextQsaAttentionQualificationError> {
    let heads = Qwen38FlashNext::NUM_ATTENTION_HEADS;
    let head_dim = Qwen38FlashNext::HEAD_DIM;
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let group = heads / Qwen38FlashNext::NUM_KV_HEADS;
    let (table_rows, lengths) = metadata(fixture, tokens);
    let compared = verified_tokens(tokens);
    let mut attention = vec![0.0f64; compared.len() * columns];
    let mut gate = vec![0.0f64; compared.len() * columns];
    let mut softmax_ratio = f64::INFINITY;

    for (row, &token) in compared.iter().enumerate() {
        let table_row = table_rows[token] as usize;
        let length = lengths[token] as usize;
        for query_head in 0..heads {
            // Each KV head serves 12 consecutive query heads.
            let kv_head = query_head / group;
            let query_base = (token * heads + query_head) * head_dim;
            let query = &fixture.query[query_base..query_base + head_dim];

            let mut scores = Vec::with_capacity(length);
            for position in 0..length {
                let physical_page = fixture.block_tables
                    [table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE]
                    as usize;
                let base = cache_offset(physical_page, kv_head, position, 0);
                let keys = &fixture.key_values[base..base + head_dim];
                let mut score = 0.0f64;
                for (&q, &key) in query.iter().zip(keys) {
                    score += f64::from(q) * key * f64::from(KEY_SCALE);
                }
                scores.push(score * SOFTMAX_SCALE);
            }

            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            // Only rows with a page or more of competition say anything about
            // the fixture's ability to spread the softmax.
            if weights.len() >= ATTENTION_PAGE_SIZE {
                let largest = weights.iter().copied().fold(0.0f64, f64::max);
                let smallest = weights.iter().copied().fold(f64::INFINITY, f64::min);
                if smallest > 0.0 {
                    softmax_ratio = softmax_ratio.min(largest / smallest);
                }
            }

            // Accumulate position-major so each position's 256 contiguous
            // decoded values are walked once, rather than restriding the plane
            // per dimension.
            let mut sums = vec![0.0f64; head_dim];
            for (position, &weight) in weights.iter().enumerate() {
                let physical_page = fixture.block_tables
                    [table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE]
                    as usize;
                let base = cache_offset(physical_page, kv_head, position, 0);
                let values = &fixture.value_values[base..base + head_dim];
                for (sum, &value) in sums.iter_mut().zip(values) {
                    *sum += weight * value * f64::from(VALUE_SCALE);
                }
            }

            // The gate follows the query in each packed 512-row head.
            let gate_base =
                token * Qwen38FlashNext::ATTENTION_QKV_ROWS + query_head * 2 * head_dim + head_dim;
            let gates = &fixture.qkv[gate_base..gate_base + head_dim];
            let base = row * columns + query_head * head_dim;
            for (dimension, (&sum, &packed)) in sums.iter().zip(gates).enumerate() {
                attention[base + dimension] = sum / denominator;
                gate[base + dimension] = f64::from(bf16_to_f32(packed));
            }
        }
    }

    Ok(Expected {
        tokens: compared,
        attention,
        gate,
        softmax_ratio,
    })
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f64) -> f64 {
    value * sigmoid(value)
}

fn verify(
    tokens: usize,
    expected: &Expected,
    observed: &Observed,
    report: &mut Qwen38FlashNextQsaAttentionQualification,
) -> Result<(), Qwen38FlashNextQsaAttentionQualificationError> {
    let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
    let active = tokens * columns;
    if expected.softmax_ratio.is_finite() {
        report.minimum_softmax_ratio = report
            .minimum_softmax_ratio
            .min(expected.softmax_ratio as f32);
    }

    for (row, &token) in expected.tokens.iter().enumerate() {
        for column in 0..columns {
            let index = token * columns + column;
            let ungated = expected.attention[row * columns + column];
            let gate = expected.gate[row * columns + column];
            let gated = ungated * sigmoid(gate);

            let device = observed.attention[index];
            require_close("gated attention", index, device, gated, report)?;
            let device_activation = bf16_to_f32(observed.activation[index]);
            require_bf16_close("gated activation", index, device_activation, gated, report)?;
            report.gated_values += 2;

            // Count alternatives rejected by the device output's tolerance.
            let tolerance = acceptance_tolerance(gated);
            if (gated - ungated).abs() > tolerance {
                report.gate_separated_values += 1;
            }
            if (gated - ungated * silu(gate)).abs() > tolerance {
                report.silu_separated_values += 1;
            }
            if gate == 0.0 {
                // Sigmoid at zero must halve the ungated attention.
                require_close("half-gated attention", index, device, ungated * 0.5, report)?;
                report.half_gated_values += 1;
            }
            report.attention_values += 1;
        }
        report.verified_tokens += 1;
    }

    for index in active..MAX_TOKENS * columns {
        if observed.attention[index].to_bits() != F32_SENTINEL_BITS
            || observed.activation[index] != BF16_SENTINEL
        {
            return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
                format!("tokens={tokens} wrote past its own rows at value {index}"),
            ));
        }
        report.untouched_values += 2;
    }

    Ok(())
}

/// FP32 online-softmax tolerance against the two-pass FP64 oracle.
fn acceptance_tolerance(expected: f64) -> f64 {
    0.000_5f64.max(expected.abs() * 0.001)
}

fn require_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    report: &mut Qwen38FlashNextQsaAttentionQualification,
) -> Result<(), Qwen38FlashNextQsaAttentionQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    // The entries run an FP32 online softmax over `ex2.approx`; the oracle is a
    // two-pass FP64 softmax with a true exponential, so the two agree to a
    // relative bound rather than exactly.
    let tolerance = 0.000_5f32.max(expected.abs() as f32 * 0.001);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
            format!(
                "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            ),
        ));
    }
    Ok(())
}

fn require_bf16_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    report: &mut Qwen38FlashNextQsaAttentionQualification,
) -> Result<(), Qwen38FlashNextQsaAttentionQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    // BF16 keeps eight significand bits, so the published activation carries a
    // rounding step the FP32 seam beside it does not.
    let tolerance = 0.008f32.max(expected.abs() as f32 * 0.008);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
            format!(
                "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            ),
        ));
    }
    Ok(())
}

fn verify_replay(
    tokens: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen38FlashNextQsaAttentionQualification,
) -> Result<(), Qwen38FlashNextQsaAttentionQualificationError> {
    for (index, (&eager, &replay)) in eager
        .attention
        .iter()
        .zip(replay.attention.iter())
        .enumerate()
    {
        if eager.to_bits() != replay.to_bits() {
            return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
                format!(
                    "graph replay changed the attention output at tokens={tokens} value {index}"
                ),
            ));
        }
        report.graph_replay_values += 1;
    }
    for (index, (&eager, &replay)) in eager
        .activation
        .iter()
        .zip(replay.activation.iter())
        .enumerate()
    {
        if eager != replay {
            return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
                format!(
                    "graph replay changed the gated activation at tokens={tokens} value {index}"
                ),
            ));
        }
        report.graph_replay_values += 1;
    }

    Ok(())
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextQsaAttentionQualification,
) -> Result<(), Qwen38FlashNextQsaAttentionQualificationError> {
    macro_rules! unchanged {
        ($region:ident, $expected:expr, $role:literal) => {{
            let observed = arena.copy_to_host(stream, regions.$region)?;
            let expected: &[_] = $expected;
            if observed.len() != expected.len() || observed != expected {
                return Err(Qwen38FlashNextQsaAttentionQualificationError::Mismatch(
                    format!(concat!($role, " changed while the route ran")),
                ));
            }
            report.immutable_input_values += observed.len();
        }};
    }

    unchanged!(query, &fixture.query, "query");
    unchanged!(key_pages, &fixture.key_pages, "key cache");
    unchanged!(value_pages, &fixture.value_pages, "value cache");
    unchanged!(block_tables, &fixture.block_tables, "block table");
    unchanged!(qkv, &fixture.qkv, "packed projection");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CACHE_CODES, VALUE_CODES};
    use super::{
        Fixture, GATE_PATTERN, PACKED_QUERY_PATTERN, PHYSICAL_PAGES, Qwen38FlashNext, ROUTES,
        SOFTMAX_SCALE, TABLE_ROWS, TABLE_STRIDE, cache_offset, fixture, layout,
        qualify_qwen38_flash_next_qsa_attention, sigmoid, silu, verified_tokens,
    };
    use crate::fp8_projection_oracle::{bf16_to_f32, decode_e4m3fn};
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
    use tuisko_model::Arch;

    /// The gate fixture has to separate `sigmoid` from both a dropped gate and
    /// a SiLU gate, and `z = 1` has to be the *only* coincidence between the
    /// two activations, so the fixture is never mistaken for one that
    /// discriminates everywhere.
    #[test]
    fn qwen38_flash_next_qsa_attention_gate_fixture_is_decisive() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert_eq!(silu(0.0), 0.0);

        let coincidences = GATE_PATTERN
            .iter()
            .filter(|&&z| (sigmoid(f64::from(z)) - silu(f64::from(z))).abs() < 1e-9)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(coincidences, vec![1.0]);

        // Every other entry separates the two activations, and the fixture
        // spans both signs.
        assert!(GATE_PATTERN.iter().any(|&z| z < 0.0));
        assert!(GATE_PATTERN.iter().any(|&z| z > 0.0));
        assert!(GATE_PATTERN.contains(&0.0));
        // `silu(z) = z * sigmoid(z)`, so at `z = -4` a SiLU gate's multiplier
        // is exactly four times sigmoid's in magnitude and opposite in sign.
        assert!((silu(-4.0) / sigmoid(-4.0) + 4.0).abs() < 1e-12);
        assert!((silu(4.0) / sigmoid(4.0) - 4.0).abs() < 1e-12);

        // Away from the coincidence point the two multipliers separate by at
        // least 0.7x the sigmoid multiplier, since
        // `|sigmoid(z) - silu(z)| = sigmoid(z) * |1 - z|` and the fixture's
        // smallest nonzero `|1 - z|` is 0.75 at `z = 0.25`.
        for &z in &GATE_PATTERN {
            let z = f64::from(z);
            if (z - 1.0).abs() < 1e-12 {
                continue;
            }
            assert!(
                (sigmoid(z) - silu(z)).abs() > 0.7 * sigmoid(z).abs(),
                "gate value {z} does not separate sigmoid from SiLU"
            );
        }

        // A dropped gate is a multiply by one; no fixture entry has
        // `sigmoid(z) == 1`.
        assert!(
            GATE_PATTERN
                .iter()
                .all(|&z| (sigmoid(f64::from(z)) - 1.0).abs() > 0.01)
        );
    }

    /// The packed `[query|gate]` split has to be observable: the gate half and
    /// the query half beside it must never carry the same value, or an entry
    /// that slices the wrong half would still agree with the oracle.
    #[test]
    fn qwen38_flash_next_qsa_attention_fixture_separates_the_packed_halves() {
        let Fixture { qkv, .. } = fixture();
        let head_dim = Qwen38FlashNext::HEAD_DIM;
        let mut checked = 0_usize;
        for token in 0..4 {
            let base = token * Qwen38FlashNext::ATTENTION_QKV_ROWS;
            for head in 0..Qwen38FlashNext::NUM_ATTENTION_HEADS {
                let query = base + head * 2 * head_dim;
                for dimension in 0..head_dim {
                    let query_value = bf16_to_f32(qkv[query + dimension]);
                    let gate_value = bf16_to_f32(qkv[query + head_dim + dimension]);
                    assert_ne!(
                        query_value, gate_value,
                        "packed halves collide at head {head} dimension {dimension}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 4 * Qwen38FlashNext::NUM_ATTENTION_HEADS * head_dim);
        assert!(PACKED_QUERY_PATTERN.iter().all(|value| value.is_finite()));
    }

    /// Proves the oracle's decoded cache planes preserve the source codes.
    #[test]
    fn qwen38_flash_next_qsa_attention_decoded_cache_matches_its_codes() {
        for &code in CACHE_CODES.iter().chain(&VALUE_CODES) {
            let value = decode_e4m3fn(code).expect("fixture codes are finite E4M3");
            assert!(value.is_finite());
        }
        // The value plane must not cancel: a same-sign table is what keeps the
        // attention magnitude clear of the acceptance floor.
        assert!(
            VALUE_CODES
                .iter()
                .all(|&code| decode_e4m3fn(code).expect("finite") > 0.0)
        );
        // ... while the key plane must span both signs to spread the scores.
        assert!(
            CACHE_CODES
                .iter()
                .any(|&code| decode_e4m3fn(code).expect("finite") < 0.0)
        );

        let fixture = fixture();
        assert_eq!(fixture.key_values.len(), fixture.key_pages.len());
        assert_eq!(fixture.value_values.len(), fixture.value_pages.len());
        for (index, (&code, &decoded)) in fixture
            .key_pages
            .iter()
            .zip(&fixture.key_values)
            .enumerate()
            .step_by(97)
        {
            let expected = f64::from(decode_e4m3fn(code).expect("fixture codes are finite E4M3"));
            assert_eq!(decoded, expected, "key plane diverged at {index}");
        }
        for (index, (&code, &decoded)) in fixture
            .value_pages
            .iter()
            .zip(&fixture.value_values)
            .enumerate()
            .step_by(97)
        {
            let expected = f64::from(decode_e4m3fn(code).expect("fixture codes are finite E4M3"));
            assert_eq!(decoded, expected, "value plane diverged at {index}");
        }
    }

    /// The fixture's page geometry has to cover the widest admitted prompt
    /// exactly, or `T=1024` would read a page the block table never mapped.
    #[test]
    fn qwen38_flash_next_qsa_attention_layout_covers_every_admitted_width() {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(PHYSICAL_PAGES * ATTENTION_PAGE_SIZE, 1_024);
        assert_eq!(TABLE_STRIDE, PHYSICAL_PAGES);
        assert_eq!(TABLE_ROWS * TABLE_STRIDE, 128);
        assert_eq!(SOFTMAX_SCALE, 0.0625);
        assert_eq!(
            SOFTMAX_SCALE,
            f64::from(Qwen38FlashNext::HEAD_DIM as u32).powf(-0.5)
        );
        assert_eq!(
            Qwen38FlashNext::NUM_ATTENTION_HEADS / Qwen38FlashNext::NUM_KV_HEADS,
            12
        );
        // Distinct (page, head, position, dimension) tuples must not alias.
        assert_ne!(cache_offset(0, 0, 0, 0), cache_offset(0, 1, 0, 0));
        assert_ne!(cache_offset(0, 0, 0, 0), cache_offset(1, 0, 0, 0));
        assert_ne!(cache_offset(0, 0, 0, 0), cache_offset(0, 0, 1, 0));
        let (layout, regions) = layout().expect("layout fits");
        assert_eq!(
            layout.byte_len() - (layout.byte_len() - regions.payload_bytes()),
            regions.payload_bytes()
        );
    }

    /// Qwen3.8-Flash-Next QSA attends 24 query heads over 2 KV heads at head_dim 256,
    /// so one token carries 6,144 attention values and 6,144 gated values.
    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_qsa_attention_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::Qwen38FlashNextQsaAttentionQualificationError> {
        let report = qualify_qwen38_flash_next_qsa_attention()?;
        let columns = Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS;
        let active_tokens = ROUTES.iter().sum::<usize>();
        let compared_tokens = ROUTES
            .iter()
            .map(|&tokens| verified_tokens(tokens).len())
            .sum::<usize>();
        let total_observable = ROUTES.len() * 2 * super::MAX_TOKENS * columns;

        assert_eq!(columns, 6_144);
        // Every row is compared through T=128; T=1024 contributes a
        // deterministic stride-32 sample of 32 rows including the last.
        assert_eq!(verified_tokens(128).len(), 128);
        assert_eq!(verified_tokens(1_024).len(), 32);
        assert!(verified_tokens(1_024).contains(&1_023));
        assert_eq!(compared_tokens, active_tokens - 1_024 + 32);

        assert_eq!(report.verified_tokens, compared_tokens);
        assert_eq!(report.attention_values, compared_tokens * columns);
        assert_eq!(report.gated_values, 2 * compared_tokens * columns);
        // The untouched proof is not sampled: it covers every value past each
        // route's own rows at every width.
        assert_eq!(
            report.untouched_values,
            total_observable - 2 * active_tokens * columns
        );
        assert_eq!(report.graph_replay_values, total_observable);

        // One gate value in eight is exactly zero, and `sigmoid(0)` halves it.
        assert_eq!(
            report.half_gated_values,
            compared_tokens * columns / GATE_PATTERN.len()
        );

        // The gate must be observable on a large majority of values, and a
        // SiLU epilogue must be excluded on a large majority too, so neither
        // count can pass on rounding alone.
        assert!(report.gate_separated_values * 4 > report.attention_values * 3);
        assert!(report.silu_separated_values * 4 > report.attention_values * 3);

        assert!(report.immutable_input_values > 0);
        assert!(report.maximum_absolute_error <= 0.01);
        // The softmax has to have been exercised: a fixture whose scores
        // coincide would leave this at one and the online recurrence's
        // maximum tracking would never have run.
        assert!(
            report.minimum_softmax_ratio > 4.0,
            "softmax weights barely spread ({}x) on full-page rows; the fixture does not test the softmax",
            report.minimum_softmax_ratio
        );
        assert_eq!(report.arena_bytes - report.padding_bytes, {
            let (_, regions) = layout()?;
            regions.payload_bytes()
        });
        Ok(())
    }
}

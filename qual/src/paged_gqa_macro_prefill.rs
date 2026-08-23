//! Exact T=1024 macro-prefill qualification for paged GQA.

use crate::fp8_projection_oracle::{
    BYTE_SENTINEL, F32_SENTINEL_BITS, decode_e4m3fn, encode_e4m3fn, f16_to_f32, f32_to_f16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS, PAGED_GQA_PREFILL_MACRO_TOKENS,
    PagedGqaOp,
};
use tuisko_model::{Arch, Qwen38_27B};

const TOKENS: usize = PAGED_GQA_PREFILL_MACRO_TOKENS;
const CONTEXT_TOKENS: usize = 2_016;
const PHYSICAL_PAGES: usize = CONTEXT_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);
const TABLE_STRIDE: usize = PHYSICAL_PAGES;
const ALIGNMENT: usize = 256;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const KEY_TILE: usize = 32;
const PARTIAL_VALUES: usize = Qwen38_27B::HEAD_DIM + 2;
const MAX_PARTITIONS: usize = PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS;
const PARTIAL_FLOATS: usize =
    TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * MAX_PARTITIONS * PARTIAL_VALUES;
const PARTITIONS: [usize; 5] = [1, 2, 4, 8, 16];
const QUERY_PATTERN: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_CODES: [u8; 8] = [0x28, 0x38, 0x48, 0x58, 0xa8, 0xb8, 0xc8, 0xd8];
const VALUE_CODES: [u8; 8] = [0x58, 0xc8, 0x38, 0xa8, 0x48, 0xd8, 0x28, 0xb8];

/// Failure of exact T=1024 paged-GQA qualification.
#[derive(Debug, thiserror::Error)]
pub enum PagedGqaMacroPrefillQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent represented-value oracle.
    #[error("macro paged GQA prefill qualification failed: {0}")]
    Mismatch(String),
}

/// Complete observable accounting for every admitted macro partition route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedGqaMacroPrefillQualification {
    /// Final FP32 output values checked against the oracle.
    pub output_values: usize,
    /// Active FP32 partition fields checked against the oracle.
    pub partial_values: usize,
    /// Inactive maximum-workspace words proved untouched.
    pub untouched_partial_values: usize,
    /// Read-only input values proved unchanged.
    pub immutable_input_values: usize,
    /// Eager values reproduced bit-exactly by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest final-output absolute error observed.
    pub maximum_absolute_error: f32,
    /// Largest active-partial absolute error observed.
    pub maximum_partial_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_tables: ArenaRegion<u32>,
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
            + self.block_tables.byte_len()
            + self.table_rows.byte_len()
            + self.lengths.byte_len()
            + self.partials.byte_len()
            + self.output.byte_len()
    }
}

struct Fixture {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    block_tables: Vec<u32>,
    table_rows: Vec<u32>,
    lengths: Vec<u32>,
}

#[derive(Clone, Copy)]
struct RepresentedState {
    maximum: f64,
    denominator: f64,
    numerator: [f64; 8],
}

struct PrefixHistograms {
    snapshots: Vec<[[u32; 64]; Qwen38_27B::NUM_KV_HEADS]>,
}

impl PrefixHistograms {
    fn interval(&self, kv_head: usize, begin: usize, end: usize) -> [u32; 64] {
        if begin >= end {
            return [0; 64];
        }
        let mut counts = [0u32; 64];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = self.snapshots[end][kv_head][index] - self.snapshots[begin][kv_head][index];
        }
        counts
    }
}

/// Qualifies every exact T=1024 `P=1/2/4/8/16` route and public FP32 seam.
pub fn qualify_paged_gqa_macro_prefill()
-> Result<PagedGqaMacroPrefillQualification, PagedGqaMacroPrefillQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = PagedGqaOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let histograms = prefix_histograms(&fixture)?;
    let scores = score_classes(&fixture)?;
    let values = value_classes();
    let mut report = PagedGqaMacroPrefillQualification {
        output_values: 0,
        partial_values: 0,
        untouched_partial_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
        maximum_partial_absolute_error: 0.0,
    };

    for partitions in PARTITIONS {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, partitions)?;
        let eager_output = arena.copy_to_host(&stream, regions.output)?;
        let eager_partials = arena.copy_to_host(&stream, regions.partials)?;
        verify_oracle(
            partitions,
            &eager_output,
            &eager_partials,
            &fixture,
            &histograms,
            &scores,
            &values,
            &mut report,
        )?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &arena, &stream, regions, partitions)
        })?;
        graph.launch(&stream)?;
        let replay_output = arena.copy_to_host(&stream, regions.output)?;
        let replay_partials = arena.copy_to_host(&stream, regions.partials)?;
        verify_replay(
            partitions,
            &eager_output,
            &eager_partials,
            &replay_output,
            &replay_partials,
            &mut report,
        )?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
                "device addresses changed for P={partitions}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(TOKENS, ALIGNMENT)?;
    let partials = layout.reserve(PARTIAL_FLOATS, ALIGNMENT)?;
    let output = layout.reserve(TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            partials,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    let query = (0..TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let token = index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let head_dimension = index % Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let head = head_dimension / Qwen38_27B::HEAD_DIM;
            let dimension = head_dimension % Qwen38_27B::HEAD_DIM;
            let amplitude = 1.0 + (token & 3) as f32 * 0.125;
            QUERY_PATTERN[(dimension + head * 3 + token) & 7] * amplitude
        })
        .collect::<Vec<_>>();
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let mut key_pages = vec![0u8; plane_bytes];
    let mut value_pages = vec![0u8; plane_bytes];
    for physical in 0..PHYSICAL_PAGES {
        for head in 0..Qwen38_27B::NUM_KV_HEADS {
            for position in 0..ATTENTION_PAGE_SIZE {
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let offset = cache_offset(physical, head, position, dimension);
                    key_pages[offset] =
                        KEY_CODES[(physical * 3 + head * 5 + position * 7 + dimension) & 7];
                    value_pages[offset] =
                        VALUE_CODES[(physical * 5 + head * 3 + position * 7 + dimension) & 7];
                }
            }
        }
    }
    let block_tables = (0..PHYSICAL_PAGES)
        .map(|page| ((page * 17) % PHYSICAL_PAGES) as u32)
        .collect::<Vec<_>>();
    let table_rows = vec![0u32; TOKENS];
    let first_length = CONTEXT_TOKENS - TOKENS + 1;
    let lengths = (0..TOKENS)
        .map(|token| (first_length + token) as u32)
        .collect::<Vec<_>>();

    Fixture {
        query,
        key_pages,
        value_pages,
        block_tables,
        table_rows,
        lengths,
    }
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.query, &fixture.query)?;
    arena.copy_from_host(stream, regions.key_pages, &fixture.key_pages)?;
    arena.copy_from_host(stream, regions.value_pages, &fixture.value_pages)?;
    arena.copy_from_host(stream, regions.block_tables, &fixture.block_tables)?;
    arena.copy_from_host(stream, regions.table_rows, &fixture.table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &fixture.lengths)
}

fn reset_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.partials, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 8]> {
    Ok([
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.lengths)?.addr(),
        arena.address(regions.partials)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &PagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    partitions: usize,
) -> GpuResult<()> {
    // SAFETY: the arena owns all 32 cache pages and the maximum P16 workspace.
    unsafe {
        op.launch_prefill_macro(
            stream,
            partitions,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.lengths)?,
            arena.address(regions.partials)?,
            arena.address(regions.output)?,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn flash_partition_interval(
    token: usize,
    lengths: &[u32],
    partitions: usize,
    partition: usize,
) -> (usize, usize) {
    let first_token = token / 32 * 32;
    let group_length = lengths[first_token + 31] as usize;
    let key_tiles = group_length.div_ceil(KEY_TILE);
    let tiles_per_partition = key_tiles.div_ceil(partitions);
    let begin = partition * tiles_per_partition * KEY_TILE;
    let end = ((partition + 1) * tiles_per_partition * KEY_TILE)
        .min(group_length)
        .min(lengths[token] as usize);
    (begin, end)
}

fn prefix_histograms(
    fixture: &Fixture,
) -> Result<PrefixHistograms, PagedGqaMacroPrefillQualificationError> {
    let mut counts = [[0u32; 64]; Qwen38_27B::NUM_KV_HEADS];
    let mut snapshots = Vec::with_capacity(CONTEXT_TOKENS + 1);
    snapshots.push(counts);
    for position in 0..CONTEXT_TOKENS {
        let physical = fixture.block_tables[position / ATTENTION_PAGE_SIZE] as usize;
        let page_offset = position & (ATTENTION_PAGE_SIZE - 1);
        for kv_head in 0..Qwen38_27B::NUM_KV_HEADS {
            let offset = cache_offset(physical, kv_head, page_offset, 0);
            let key = code_class(&KEY_CODES, fixture.key_pages[offset], "key")?;
            let value = code_class(&VALUE_CODES, fixture.value_pages[offset], "value")?;
            counts[kv_head][key * 8 + value] += 1;
        }
        snapshots.push(counts);
    }
    Ok(PrefixHistograms { snapshots })
}

fn score_classes(
    fixture: &Fixture,
) -> Result<Vec<[f64; 8]>, PagedGqaMacroPrefillQualificationError> {
    let mut scores = Vec::with_capacity(TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS);
    for head_token in 0..TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS {
        let query_base = head_token * Qwen38_27B::HEAD_DIM;
        let query = &fixture.query[query_base..query_base + Qwen38_27B::HEAD_DIM];
        let maximum = query
            .iter()
            .fold(0.0f32, |current, value| current.max(value.abs()));
        let scale = if maximum > 0.0 { maximum / 448.0 } else { 1.0 };
        let represented_query = query
            .iter()
            .map(|&value| {
                let code = encode_e4m3fn(value / scale)
                    .map_err(PagedGqaMacroPrefillQualificationError::Mismatch)?;
                decode_e4m3fn(code)
                    .map(|represented| f64::from(represented * scale))
                    .map_err(PagedGqaMacroPrefillQualificationError::Mismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut row_scores = [0.0f64; 8];
        for (class, score) in row_scores.iter_mut().enumerate() {
            *score = represented_query
                .iter()
                .enumerate()
                .map(|(dimension, &query)| {
                    query * decode_e4m3(KEY_CODES[(class + dimension) & 7]) * f64::from(KEY_SCALE)
                })
                .sum::<f64>()
                * 0.0625;
        }
        scores.push(row_scores);
    }
    Ok(scores)
}

fn value_classes() -> [[f64; 8]; 8] {
    let mut values = [[0.0f64; 8]; 8];
    for (class, class_values) in values.iter_mut().enumerate() {
        for (dimension, value) in class_values.iter_mut().enumerate() {
            let unrounded =
                decode_e4m3(VALUE_CODES[(class + dimension) & 7]) * f64::from(VALUE_SCALE);
            *value = f64::from(f16_to_f32(f32_to_f16(unrounded as f32)));
        }
    }
    values
}

fn represented_state(
    counts: &[u32; 64],
    scores: &[f64; 8],
    values: &[[f64; 8]; 8],
) -> RepresentedState {
    let mut maximum = f64::NEG_INFINITY;
    for key in 0..8 {
        if counts[key * 8..key * 8 + 8].iter().any(|&count| count != 0) {
            maximum = maximum.max(scores[key]);
        }
    }
    if !maximum.is_finite() {
        return RepresentedState {
            maximum: -1.0e30,
            denominator: 0.0,
            numerator: [0.0; 8],
        };
    }

    let mut denominator = 0.0f64;
    let mut value_weights = [0.0f64; 8];
    for key in 0..8 {
        let weight = (scores[key] - maximum).exp();
        let represented_weight = f64::from(f16_to_f32(f32_to_f16(weight as f32)));
        for value in 0..8 {
            let count = f64::from(counts[key * 8 + value]);
            denominator += count * weight;
            value_weights[value] += count * represented_weight;
        }
    }
    let mut numerator = [0.0f64; 8];
    for (dimension, sum) in numerator.iter_mut().enumerate() {
        for value in 0..8 {
            *sum += value_weights[value] * values[value][dimension];
        }
    }

    RepresentedState {
        maximum,
        denominator,
        numerator,
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_oracle(
    partitions: usize,
    output: &[f32],
    partials: &[f32],
    fixture: &Fixture,
    histograms: &PrefixHistograms,
    scores: &[[f64; 8]],
    values: &[[f64; 8]; 8],
    report: &mut PagedGqaMacroPrefillQualification,
) -> Result<(), PagedGqaMacroPrefillQualificationError> {
    for token in 0..TOKENS {
        let length = fixture.lengths[token] as usize;
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let head_token = token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head;
            let kv_head = query_head / (Qwen38_27B::NUM_ATTENTION_HEADS / Qwen38_27B::NUM_KV_HEADS);
            let final_counts = histograms.interval(kv_head, 0, length);
            let final_state = represented_state(&final_counts, &scores[head_token], values);
            let output_base = head_token * Qwen38_27B::HEAD_DIM;
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let actual = output[output_base + dimension];
                let truth = (final_state.numerator[dimension & 7] / final_state.denominator) as f32;
                let error = (actual - truth).abs();
                report.maximum_absolute_error = report.maximum_absolute_error.max(error);
                let tolerance = 0.003f32.max(truth.abs() * 0.005);
                if !actual.is_finite() || error > tolerance {
                    return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
                        "output at P={partitions}, token={token}, head={query_head}, dimension={dimension}: device={actual}, oracle={truth}, tolerance={tolerance}"
                    )));
                }
            }

            for partition in 0..partitions {
                let (begin, end) =
                    flash_partition_interval(token, &fixture.lengths, partitions, partition);
                let counts = histograms.interval(kv_head, begin, end);
                let state = represented_state(&counts, &scores[head_token], values);
                let base = (head_token * partitions + partition) * PARTIAL_VALUES;
                verify_partial_value(
                    partitions,
                    token,
                    query_head,
                    partition,
                    "maximum",
                    partials[base],
                    state.maximum as f32,
                    report,
                )?;
                verify_partial_value(
                    partitions,
                    token,
                    query_head,
                    partition,
                    "denominator",
                    partials[base + 1],
                    state.denominator as f32,
                    report,
                )?;
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    verify_partial_value(
                        partitions,
                        token,
                        query_head,
                        partition,
                        "numerator",
                        partials[base + 2 + dimension],
                        state.numerator[dimension & 7] as f32,
                        report,
                    )?;
                }
            }
        }
    }

    let active_partials = TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * partitions * PARTIAL_VALUES;
    for (index, value) in partials[active_partials..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
                "P={partitions} modified inactive partial word {}",
                active_partials + index
            )));
        }
    }

    report.output_values += output.len();
    report.partial_values += active_partials;
    report.untouched_partial_values += partials.len() - active_partials;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_partial_value(
    partitions: usize,
    token: usize,
    query_head: usize,
    partition: usize,
    field: &str,
    actual: f32,
    truth: f32,
    report: &mut PagedGqaMacroPrefillQualification,
) -> Result<(), PagedGqaMacroPrefillQualificationError> {
    let error = (actual - truth).abs();
    report.maximum_partial_absolute_error = report.maximum_partial_absolute_error.max(error);
    let tolerance = 0.01f32.max(truth.abs() * 0.005);
    if !actual.is_finite() || error > tolerance {
        return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
            "partial {field} at P={partitions}, token={token}, head={query_head}, partition={partition}: device={actual}, oracle={truth}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut PagedGqaMacroPrefillQualification,
) -> Result<(), PagedGqaMacroPrefillQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual.iter().zip($expected).position(|(a, e)| a != e) {
                return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
                    "read-only {} changed at index {index}",
                    $name
                )));
            }
            report.immutable_input_values += actual.len();
        }};
    }

    check!(regions.query, &fixture.query, "query");
    check!(regions.key_pages, &fixture.key_pages, "key cache");
    check!(regions.value_pages, &fixture.value_pages, "value cache");
    check!(regions.block_tables, &fixture.block_tables, "block tables");
    check!(regions.table_rows, &fixture.table_rows, "table rows");
    check!(regions.lengths, &fixture.lengths, "lengths");
    Ok(())
}

fn verify_replay(
    partitions: usize,
    eager_output: &[f32],
    eager_partials: &[f32],
    replay_output: &[f32],
    replay_partials: &[f32],
    report: &mut PagedGqaMacroPrefillQualification,
) -> Result<(), PagedGqaMacroPrefillQualificationError> {
    let mismatch = replay_output
        .iter()
        .chain(replay_partials)
        .map(|value| value.to_bits())
        .zip(
            eager_output
                .iter()
                .chain(eager_partials)
                .map(|value| value.to_bits()),
        )
        .position(|(actual, expected)| actual != expected);
    if let Some(index) = mismatch {
        return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
            "P={partitions} graph word {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay_output.len() + replay_partials.len();
    Ok(())
}

fn code_class(
    codes: &[u8; 8],
    code: u8,
    plane: &str,
) -> Result<usize, PagedGqaMacroPrefillQualificationError> {
    codes
        .iter()
        .position(|&candidate| candidate == code)
        .ok_or_else(|| {
            PagedGqaMacroPrefillQualificationError::Mismatch(format!(
                "represented {plane} code 0x{code:02x} is outside the oracle alphabet"
            ))
        })
}

fn cache_offset(physical: usize, head: usize, position: usize, dimension: usize) -> usize {
    Qwen38_27B::HEAD_DIM
        * (position + ATTENTION_PAGE_SIZE * (head + Qwen38_27B::NUM_KV_HEADS * physical))
        + dimension
}

fn decode_e4m3(code: u8) -> f64 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let fraction = code & 0x07;
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0.0,
        (0, fraction) => f64::from(fraction) * 2.0f64.powi(-9),
        (15, 7) => f64::NAN,
        (exponent, fraction) => {
            (1.0 + f64::from(fraction) / 8.0) * 2.0f64.powi(i32::from(exponent) - 7)
        }
    };
    sign * magnitude
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &PagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), PagedGqaMacroPrefillQualificationError> {
    let graphs = PARTITIONS
        .into_iter()
        .map(|partitions| {
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, partitions))
        })
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            graph.launch(stream)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(PagedGqaMacroPrefillQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_TOKENS, MAX_PARTITIONS, PARTIAL_FLOATS, PARTIAL_VALUES, PARTITIONS, TOKENS,
        fixture, flash_partition_interval, layout, qualify_paged_gqa_macro_prefill,
    };
    use tuisko_kernels_sm120::PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES;

    #[test]
    fn paged_gqa_suite_macro_partition_boundaries_are_k32_query_group_exact() {
        let fixture = fixture();

        for partitions in PARTITIONS {
            let group_tiles = (fixture.lengths[31] as usize).div_ceil(32);
            let tiles_per_partition = group_tiles.div_ceil(partitions);
            for partition in 0..partitions {
                let (begin, end) =
                    flash_partition_interval(31, &fixture.lengths, partitions, partition);
                assert_eq!(begin, partition * tiles_per_partition * 32);
                assert_eq!(end, ((partition + 1) * tiles_per_partition * 32).min(1_024));
            }
        }
        assert_eq!(fixture.lengths[0], 993);
        assert_eq!(fixture.lengths[TOKENS - 1] as usize, CONTEXT_TOKENS);
        assert_eq!(
            flash_partition_interval(0, &fixture.lengths, 16, 15),
            (960, 993)
        );
        assert_eq!(
            flash_partition_interval(31, &fixture.lengths, 16, 15),
            (960, 1_024)
        );
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn paged_gqa_suite_macro_prefill_matches_every_oracle_seam_and_graph_replay()
    -> Result<(), super::PagedGqaMacroPrefillQualificationError> {
        let report = qualify_paged_gqa_macro_prefill()?;
        let output_values = TOKENS * 6_144;
        let active = PARTITIONS
            .into_iter()
            .map(|partitions| TOKENS * 24 * partitions * PARTIAL_VALUES)
            .sum::<usize>();

        assert_eq!(report.output_values, PARTITIONS.len() * output_values);
        assert_eq!(report.partial_values, active);
        assert_eq!(
            report.untouched_partial_values,
            PARTITIONS.len() * PARTIAL_FLOATS - active
        );
        assert_eq!(
            report.graph_replay_values,
            PARTITIONS.len() * (output_values + PARTIAL_FLOATS)
        );
        assert!(report.maximum_absolute_error.is_finite());
        assert!(report.maximum_partial_absolute_error.is_finite());
        assert_eq!(
            PARTIAL_FLOATS * size_of::<f32>(),
            PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES
        );
        assert_eq!(MAX_PARTITIONS, 16);
        let (arena, regions) = layout()?;
        assert_eq!(report.arena_bytes, 460_333_312);
        assert_eq!(report.padding_bytes, 128);
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(
            report.padding_bytes,
            arena.byte_len() - regions.payload_bytes()
        );
        assert_eq!(
            report.arena_bytes - report.padding_bytes,
            regions.payload_bytes()
        );
        Ok(())
    }
}

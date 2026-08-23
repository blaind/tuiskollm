//! Deep-context qualification for partitioned T=128 paged GQA.

use crate::fp8_projection_oracle::{
    BYTE_SENTINEL, F32_SENTINEL_BITS, decode_e4m3fn, encode_e4m3fn, f16_to_f32, f32_to_f16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use std::collections::BTreeSet;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PagedGqaOp,
    paged_gqa_prefill_partitions,
};
use tuisko_model::{Arch, Qwen38_27B};

const TOKENS: usize = 128;
const SHORT_CONTEXT: usize = 257;
const LONG_CONTEXT: usize = PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT;
const PHYSICAL_PAGES: usize = LONG_CONTEXT.div_ceil(ATTENTION_PAGE_SIZE);
const TABLE_STRIDE: usize = PHYSICAL_PAGES;
const ALIGNMENT: usize = 256;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const PARTIAL_VALUES: usize = Qwen38_27B::HEAD_DIM + 2;
const MAX_PARTITIONS: usize = 16;
const PARTIAL_FLOATS: usize =
    TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * MAX_PARTITIONS * PARTIAL_VALUES;
const QUERY_PATTERN: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_CODES: [u8; 8] = [0x28, 0x38, 0x48, 0x58, 0xa8, 0xb8, 0xc8, 0xd8];
const VALUE_CODES: [u8; 8] = [0x58, 0xc8, 0x38, 0xa8, 0x48, 0xd8, 0x28, 0xb8];

/// Failure of partitioned T=128 paged-GQA qualification.
#[derive(Debug, thiserror::Error)]
pub enum PagedGqaPartitionedPrefillQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("partitioned paged GQA prefill qualification failed: {0}")]
    Mismatch(String),
}

/// Complete observable accounting for both exact partition bands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedGqaPartitionedPrefillQualification {
    /// Final FP32 output values compared with the independent oracle.
    pub output_values: usize,
    /// Active FP32 partial fields compared with independent partition states.
    pub partial_values: usize,
    /// Inactive maximum-workspace words proved untouched.
    pub untouched_partial_values: usize,
    /// Read-only inputs proved unchanged.
    pub immutable_input_values: usize,
    /// Output and partial words reproduced exactly by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest final-output absolute error.
    pub maximum_absolute_error: f32,
    /// Largest active-partial absolute error.
    pub maximum_partial_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    short_lengths: ArenaRegion<u32>,
    long_lengths: ArenaRegion<u32>,
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
            + self.short_lengths.byte_len()
            + self.long_lengths.byte_len()
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
    short_lengths: Vec<u32>,
    long_lengths: Vec<u32>,
}

struct Oracle {
    output: Vec<f32>,
    partials: Vec<f32>,
    active_partials: usize,
}

struct PrefixHistograms {
    boundaries: Vec<usize>,
    snapshots: Vec<[[u32; 64]; Qwen38_27B::NUM_KV_HEADS]>,
}

impl PrefixHistograms {
    fn interval(&self, kv_head: usize, begin: usize, end: usize) -> [u32; 64] {
        if begin >= end {
            return [0; 64];
        }
        let begin = self.boundaries.binary_search(&begin).unwrap();
        let end = self.boundaries.binary_search(&end).unwrap();
        let mut counts = [0u32; 64];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = self.snapshots[end][kv_head][index] - self.snapshots[begin][kv_head][index];
        }
        counts
    }
}

fn flash_partition_interval(
    token: usize,
    lengths: &[u32],
    partitions: usize,
    partition: usize,
) -> (usize, usize) {
    let key_tile = if partitions == 8 { 64 } else { 32 };
    let first_token = token / 32 * 32;
    let group_length = lengths[first_token + 31] as usize;
    let key_tiles = group_length.div_ceil(key_tile);
    let tiles_per_partition = key_tiles.div_ceil(partitions);
    let begin = partition * tiles_per_partition * key_tile;
    let end = ((partition + 1) * tiles_per_partition * key_tile)
        .min(group_length)
        .min(lengths[token] as usize);
    (begin, end)
}

/// Qualifies P=8 and P=16 eager/graph routes, including their full FP32 seams.
pub fn qualify_paged_gqa_partitioned_prefill()
-> Result<PagedGqaPartitionedPrefillQualification, PagedGqaPartitionedPrefillQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
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
    let op = PagedGqaOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = PagedGqaPartitionedPrefillQualification {
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

    for context_tokens in [SHORT_CONTEXT, LONG_CONTEXT] {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, context_tokens)?;
        let eager_output = arena.copy_to_host(&stream, regions.output)?;
        let eager_partials = arena.copy_to_host(&stream, regions.partials)?;
        let expected = oracle(context_tokens, &fixture)?;
        verify_oracle(
            context_tokens,
            &eager_output,
            &eager_partials,
            &expected,
            &mut report,
        )?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &arena, &stream, regions, context_tokens)
        })?;
        graph.launch(&stream)?;
        let replay_output = arena.copy_to_host(&stream, regions.output)?;
        let replay_partials = arena.copy_to_host(&stream, regions.partials)?;
        verify_replay(
            context_tokens,
            &eager_output,
            &eager_partials,
            &replay_output,
            &replay_partials,
            &mut report,
        )?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
                format!("device addresses changed at context={context_tokens}"),
            ));
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
    let short_lengths = layout.reserve(TOKENS, ALIGNMENT)?;
    let long_lengths = layout.reserve(TOKENS, ALIGNMENT)?;
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
            short_lengths,
            long_lengths,
            partials,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    let query = (0..TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| QUERY_PATTERN[(index + index / Qwen38_27B::HEAD_DIM) & 7])
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
    let short_first = SHORT_CONTEXT - TOKENS + 1;
    let long_first = LONG_CONTEXT - TOKENS + 1;
    let short_lengths = (0..TOKENS)
        .map(|token| (short_first + token) as u32)
        .collect();
    let long_lengths = (0..TOKENS)
        .map(|token| (long_first + token) as u32)
        .collect();

    Fixture {
        query,
        key_pages,
        value_pages,
        block_tables,
        table_rows,
        short_lengths,
        long_lengths,
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
    arena.copy_from_host(stream, regions.short_lengths, &fixture.short_lengths)?;
    arena.copy_from_host(stream, regions.long_lengths, &fixture.long_lengths)
}

fn reset_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.partials, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 9]> {
    Ok([
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.short_lengths)?.addr(),
        arena.address(regions.long_lengths)?.addr(),
        arena.address(regions.partials)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &PagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    context_tokens: usize,
) -> GpuResult<()> {
    let lengths = if context_tokens == SHORT_CONTEXT {
        regions.short_lengths
    } else {
        regions.long_lengths
    };
    // SAFETY: the maximum arena owns all 513 pages and the P=16 workspace.
    unsafe {
        op.launch_prefill_partitioned(
            stream,
            context_tokens,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(lengths)?,
            arena.address(regions.partials)?,
            arena.address(regions.output)?,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn oracle(
    context_tokens: usize,
    fixture: &Fixture,
) -> Result<Oracle, PagedGqaPartitionedPrefillQualificationError> {
    let partitions = paged_gqa_prefill_partitions(context_tokens)?;
    let lengths = if context_tokens == SHORT_CONTEXT {
        &fixture.short_lengths
    } else {
        &fixture.long_lengths
    };
    let active_partials = TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * partitions * PARTIAL_VALUES;
    let mut partials = vec![f32::from_bits(F32_SENTINEL_BITS); PARTIAL_FLOATS];
    let mut output = vec![0.0f32; TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS];
    let histograms = prefix_histograms(context_tokens, lengths, partitions, fixture)?;
    let scores = score_classes(fixture)?;
    let values = value_classes();

    for token in 0..TOKENS {
        let length = lengths[token] as usize;
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let kv_head = query_head / (Qwen38_27B::NUM_ATTENTION_HEADS / Qwen38_27B::NUM_KV_HEADS);
            let final_counts = histograms.interval(kv_head, 0, length);
            let final_state = represented_state(&final_counts, &scores[query_head], &values);
            let output_base =
                (token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head) * Qwen38_27B::HEAD_DIM;
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                output[output_base + dimension] = (final_state.2[dimension] / final_state.1) as f32;
            }

            for partition in 0..partitions {
                let (begin, end) = flash_partition_interval(token, lengths, partitions, partition);
                let counts = histograms.interval(kv_head, begin, end);
                let state = represented_state(&counts, &scores[query_head], &values);
                let base = ((token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head) * partitions
                    + partition)
                    * PARTIAL_VALUES;
                partials[base] = state.0 as f32;
                partials[base + 1] = state.1 as f32;
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    partials[base + 2 + dimension] = state.2[dimension] as f32;
                }
            }
        }
    }

    Ok(Oracle {
        output,
        partials,
        active_partials,
    })
}

fn prefix_histograms(
    context_tokens: usize,
    lengths: &[u32],
    partitions: usize,
    fixture: &Fixture,
) -> Result<PrefixHistograms, PagedGqaPartitionedPrefillQualificationError> {
    let mut boundaries = BTreeSet::from([0usize, context_tokens]);
    for token in 0..TOKENS {
        let length = lengths[token] as usize;
        boundaries.insert(length);
        for partition in 0..partitions {
            let (begin, end) = flash_partition_interval(token, lengths, partitions, partition);
            if begin < end {
                boundaries.insert(begin);
                boundaries.insert(end);
            }
        }
    }
    let boundaries = boundaries.into_iter().collect::<Vec<_>>();
    let mut counts = [[0u32; 64]; Qwen38_27B::NUM_KV_HEADS];
    let mut snapshots = Vec::with_capacity(boundaries.len());
    let mut boundary = 0usize;

    for position in 0..=context_tokens {
        if boundaries[boundary] == position {
            snapshots.push(counts);
            boundary += 1;
            if boundary == boundaries.len() {
                break;
            }
        }

        let physical = fixture.block_tables[position / ATTENTION_PAGE_SIZE] as usize;
        let page_offset = position & (ATTENTION_PAGE_SIZE - 1);
        for kv_head in 0..Qwen38_27B::NUM_KV_HEADS {
            let offset = cache_offset(physical, kv_head, page_offset, 0);
            let key = code_class(&KEY_CODES, fixture.key_pages[offset], "key")?;
            let value = code_class(&VALUE_CODES, fixture.value_pages[offset], "value")?;
            counts[kv_head][key * 8 + value] += 1;
        }
    }

    Ok(PrefixHistograms {
        boundaries,
        snapshots,
    })
}

fn code_class(
    codes: &[u8; 8],
    code: u8,
    plane: &str,
) -> Result<usize, PagedGqaPartitionedPrefillQualificationError> {
    codes
        .iter()
        .position(|&candidate| candidate == code)
        .ok_or_else(|| {
            PagedGqaPartitionedPrefillQualificationError::Mismatch(format!(
                "represented {plane} code 0x{code:02x} is outside the oracle alphabet"
            ))
        })
}

fn score_classes(
    fixture: &Fixture,
) -> Result<[[f64; 8]; Qwen38_27B::NUM_ATTENTION_HEADS], PagedGqaPartitionedPrefillQualificationError>
{
    let mut scores = [[0.0f64; 8]; Qwen38_27B::NUM_ATTENTION_HEADS];
    for (query_head, head_scores) in scores.iter_mut().enumerate() {
        let query_base = query_head * Qwen38_27B::HEAD_DIM;
        let query = &fixture.query[query_base..query_base + Qwen38_27B::HEAD_DIM];
        let maximum = query
            .iter()
            .fold(0.0f32, |current, value| current.max(value.abs()));
        let scale = if maximum > 0.0 { maximum / 448.0 } else { 1.0 };
        let represented_query = query
            .iter()
            .map(|&value| {
                let code = encode_e4m3fn(value / scale)
                    .map_err(PagedGqaPartitionedPrefillQualificationError::Mismatch)?;
                decode_e4m3fn(code)
                    .map(|represented| f64::from(represented * scale))
                    .map_err(PagedGqaPartitionedPrefillQualificationError::Mismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (class, score) in head_scores.iter_mut().enumerate() {
            *score = represented_query
                .iter()
                .enumerate()
                .map(|(dimension, &query)| {
                    query * decode_e4m3(KEY_CODES[(class + dimension) & 7]) * f64::from(KEY_SCALE)
                })
                .sum::<f64>()
                * 0.0625;
        }
    }
    Ok(scores)
}

fn value_classes() -> [[f64; Qwen38_27B::HEAD_DIM]; 8] {
    let mut values = [[0.0f64; Qwen38_27B::HEAD_DIM]; 8];
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
    values: &[[f64; Qwen38_27B::HEAD_DIM]; 8],
) -> (f64, f64, [f64; Qwen38_27B::HEAD_DIM]) {
    let mut maximum = f64::NEG_INFINITY;
    for key in 0..8 {
        if counts[key * 8..key * 8 + 8].iter().any(|&count| count != 0) {
            maximum = maximum.max(scores[key]);
        }
    }
    if !maximum.is_finite() {
        return (-1.0e30, 0.0, [0.0; Qwen38_27B::HEAD_DIM]);
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
    let mut numerator = [0.0f64; Qwen38_27B::HEAD_DIM];
    for (dimension, sum) in numerator.iter_mut().enumerate() {
        for value in 0..8 {
            *sum += value_weights[value] * values[value][dimension];
        }
    }

    (maximum, denominator, numerator)
}

fn verify_oracle(
    context_tokens: usize,
    output: &[f32],
    partials: &[f32],
    expected: &Oracle,
    report: &mut PagedGqaPartitionedPrefillQualification,
) -> Result<(), PagedGqaPartitionedPrefillQualificationError> {
    for (index, (&actual, &truth)) in output.iter().zip(&expected.output).enumerate() {
        let error = (actual - truth).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = 0.003f32.max(truth.abs() * 0.005);
        if !actual.is_finite() || error > tolerance {
            return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
                format!(
                    "output at context={context_tokens}, index={index}: device={actual}, oracle={truth}, tolerance={tolerance}"
                ),
            ));
        }
    }
    for (index, (&actual, &truth)) in partials[..expected.active_partials]
        .iter()
        .zip(&expected.partials[..expected.active_partials])
        .enumerate()
    {
        let error = (actual - truth).abs();
        report.maximum_partial_absolute_error = report.maximum_partial_absolute_error.max(error);
        let tolerance = 0.01f32.max(truth.abs() * 0.005);
        if !actual.is_finite() || error > tolerance {
            return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
                format!(
                    "partial at context={context_tokens}, index={index}: device={actual}, oracle={truth}, tolerance={tolerance}"
                ),
            ));
        }
    }
    for (index, value) in partials[expected.active_partials..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
                format!(
                    "context={context_tokens} modified inactive partial word {}",
                    expected.active_partials + index
                ),
            ));
        }
    }

    report.output_values += output.len();
    report.partial_values += expected.active_partials;
    report.untouched_partial_values += partials.len() - expected.active_partials;
    Ok(())
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut PagedGqaPartitionedPrefillQualification,
) -> Result<(), PagedGqaPartitionedPrefillQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual.iter().zip($expected).position(|(a, e)| a != e) {
                return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
                    format!("read-only {} changed at index {index}", $name),
                ));
            }
            report.immutable_input_values += actual.len();
        }};
    }

    check!(regions.query, &fixture.query, "query");
    check!(regions.key_pages, &fixture.key_pages, "key cache");
    check!(regions.value_pages, &fixture.value_pages, "value cache");
    check!(regions.block_tables, &fixture.block_tables, "block tables");
    check!(regions.table_rows, &fixture.table_rows, "table rows");
    check!(
        regions.short_lengths,
        &fixture.short_lengths,
        "short lengths"
    );
    check!(regions.long_lengths, &fixture.long_lengths, "long lengths");
    Ok(())
}

fn verify_replay(
    context_tokens: usize,
    eager_output: &[f32],
    eager_partials: &[f32],
    replay_output: &[f32],
    replay_partials: &[f32],
    report: &mut PagedGqaPartitionedPrefillQualification,
) -> Result<(), PagedGqaPartitionedPrefillQualificationError> {
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
        return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
            format!("context={context_tokens} graph word {index} differs from eager"),
        ));
    }
    report.graph_replay_values += replay_output.len() + replay_partials.len();
    Ok(())
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
) -> Result<(), PagedGqaPartitionedPrefillQualificationError> {
    let graphs = [SHORT_CONTEXT, LONG_CONTEXT]
        .into_iter()
        .map(|context_tokens| {
            CudaGraph::capture(stream, || {
                launch(op, arena, stream, regions, context_tokens)
            })
        })
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        graphs[1].launch(stream)?;
        graphs[0].launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(PagedGqaPartitionedPrefillQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LONG_CONTEXT, MAX_PARTITIONS, PARTIAL_FLOATS, PARTIAL_VALUES, SHORT_CONTEXT, TOKENS,
        decode_e4m3fn, encode_e4m3fn, f16_to_f32, f32_to_f16, fixture, flash_partition_interval,
        layout, qualify_paged_gqa_partitioned_prefill,
    };
    use tuisko_kernels_sm120::{PAGED_GQA_PREFILL_PARTIAL_BYTES, paged_gqa_prefill_partitions};

    #[test]
    fn paged_gqa_suite_flash_partition_boundaries_follow_exact_query_groups_and_key_tiles() {
        let fixture = fixture();

        assert_eq!(
            flash_partition_interval(0, &fixture.short_lengths, 8, 0),
            (0, 64)
        );
        assert_eq!(
            flash_partition_interval(0, &fixture.short_lengths, 8, 2),
            (128, 130)
        );
        assert_eq!(
            flash_partition_interval(31, &fixture.short_lengths, 8, 2),
            (128, 161)
        );
        assert_eq!(
            flash_partition_interval(31, &fixture.short_lengths, 8, 3),
            (192, 161)
        );

        assert_eq!(
            flash_partition_interval(0, &fixture.long_lengths, 16, 0),
            (0, 2_048)
        );
        assert_eq!(
            flash_partition_interval(0, &fixture.long_lengths, 16, 15),
            (30_720, 32_642)
        );
        assert_eq!(
            flash_partition_interval(31, &fixture.long_lengths, 16, 15),
            (30_720, 32_673)
        );
    }

    #[test]
    fn paged_gqa_suite_flash_representation_codecs_pin_rounding() {
        let scale = 0.5 / 448.0;

        assert_eq!(encode_e4m3fn(-0.375 / scale).unwrap(), 0xfa);
        assert_eq!(decode_e4m3fn(0xfa).unwrap() * scale, -0.357_142_87);
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
        assert_eq!(f16_to_f32(0x3c01), 1.0 + 2.0f32.powi(-10));
        assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn paged_gqa_suite_partitioned_prefill_matches_complete_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaPartitionedPrefillQualificationError> {
        let report = qualify_paged_gqa_partitioned_prefill()?;
        let output_values = TOKENS * 6_144;
        let active = [SHORT_CONTEXT, LONG_CONTEXT]
            .into_iter()
            .map(|context| {
                TOKENS * 24 * paged_gqa_prefill_partitions(context).unwrap() * PARTIAL_VALUES
            })
            .sum::<usize>();

        assert_eq!(report.output_values, 2 * output_values);
        assert_eq!(report.partial_values, active);
        assert_eq!(report.untouched_partial_values, 2 * PARTIAL_FLOATS - active);
        assert_eq!(
            report.graph_replay_values,
            2 * (output_values + PARTIAL_FLOATS)
        );
        assert!(report.maximum_absolute_error <= 0.003);
        assert!(report.maximum_partial_absolute_error <= 0.01);
        assert_eq!(
            PARTIAL_FLOATS * size_of::<f32>(),
            PAGED_GQA_PREFILL_PARTIAL_BYTES
        );
        assert_eq!(MAX_PARTITIONS, 16);
        let (arena, regions) = layout()?;
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

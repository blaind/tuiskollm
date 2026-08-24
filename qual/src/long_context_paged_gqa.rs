//! Numerical and graph qualification for partitioned long-context paged GQA.

use crate::fp8_projection_oracle::{BYTE_SENTINEL, F32_SENTINEL_BITS};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_SIZE, LongContextPagedGqaOp,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = LONG_CONTEXT_GQA_MAX_TOKENS.div_ceil(ATTENTION_PAGE_SIZE);
const TABLE_ROWS: usize = MAX_BATCH;
const TABLE_STRIDE: usize = PHYSICAL_PAGES;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const LENGTHS: [u32; MAX_BATCH] = [257, 1_025, 4_097, 16_385, 65_537, 131_073, 219_999, 220_000];
const QUERY_PATTERN: [f32; 8] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125,
];
const KEY_CODES: [u8; 9] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20, 0xa0];
const VALUE_CODES: [u8; 9] = [0x38, 0xb8, 0x30, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];

/// Failure of the partitioned long-context paged GQA gate.
#[derive(Debug, thiserror::Error)]
pub enum LongContextPagedGqaQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("long-context paged GQA qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongContextPagedGqaQualification {
    /// Active final FP32 outputs compared with the independent FP64 softmax.
    pub output_values: usize,
    /// Active partition maximum and denominator values compared.
    pub partial_scalar_values: usize,
    /// Active partition numerator values compared.
    pub partial_numerator_values: usize,
    /// Inactive scratch and output words proved untouched.
    pub untouched_values: usize,
    /// Read-only query, cache, and page metadata values proved unchanged.
    pub immutable_input_values: usize,
    /// Complete scratch and output state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact bytes owned by the three partition scratch planes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute final-output error.
    pub maximum_output_error: f32,
    /// Largest absolute partition-maximum error.
    pub maximum_partial_maximum_error: f32,
    /// Largest absolute partition-denominator error.
    pub maximum_partial_denominator_error: f32,
    /// Largest absolute partition-numerator error.
    pub maximum_partial_numerator_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
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

    fn workspace_bytes(self) -> usize {
        self.partial_maximum.byte_len()
            + self.partial_denominator.byte_len()
            + self.partial_numerator.byte_len()
    }
}

struct Fixture {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    block_tables: Vec<u32>,
}

struct Oracle {
    partial_maximum: Vec<f32>,
    partial_denominator: Vec<f32>,
    partial_numerator: Vec<f32>,
    output: Vec<f32>,
}

struct Observed {
    partial_maximum: Vec<f32>,
    partial_denominator: Vec<f32>,
    partial_numerator: Vec<f32>,
    output: Vec<f32>,
}

/// Qualifies eager and captured partitioned GQA routes at exact `B=1..=8`.
pub fn qualify_long_context_paged_gqa()
-> Result<LongContextPagedGqaQualification, LongContextPagedGqaQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let oracle = oracle(&fixture)?;
    let op = LongContextPagedGqaOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = LongContextPagedGqaQualification {
        output_values: 0,
        partial_scalar_values: 0,
        partial_numerator_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        arena_bytes: layout.byte_len(),
        workspace_bytes: regions.workspace_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_output_error: 0.0,
        maximum_partial_maximum_error: 0.0,
        maximum_partial_denominator_error: 0.0,
        maximum_partial_numerator_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_observed(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = copy_observed(&arena, &stream, regions)?;
        verify_oracle(batch, &oracle, &eager, &mut report)?;

        reset_observed(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = copy_observed(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn scalar_elements() -> usize {
    MAX_BATCH * Qwen38_27B::NUM_ATTENTION_HEADS * LONG_CONTEXT_GQA_MAX_PARTITIONS
}

fn numerator_elements() -> usize {
    scalar_elements() * Qwen38_27B::HEAD_DIM
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let partial_maximum = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_denominator = layout.reserve(scalar_elements(), ALIGNMENT)?;
    let partial_numerator = layout.reserve(numerator_elements(), ALIGNMENT)?;
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

fn fixture() -> Fixture {
    let query = (0..MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let token = index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let query_head = index / Qwen38_27B::HEAD_DIM % Qwen38_27B::NUM_ATTENTION_HEADS;
            let class = index & 7;
            QUERY_PATTERN[(class + query_head * 3 + token * 5) & 7] * (1.0 - token as f32 / 16.0)
        })
        .collect::<Vec<_>>();
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let mut key_pages = vec![0u8; plane_bytes];
    let mut value_pages = vec![0u8; plane_bytes];
    for physical in 0..PHYSICAL_PAGES {
        for head in 0..Qwen38_27B::NUM_KV_HEADS {
            for position in 0..ATTENTION_PAGE_SIZE {
                for class in 0..8 {
                    let key = KEY_CODES[(physical * 5 + head * 3 + position + class * 2) % 9];
                    let value = VALUE_CODES[(physical * 7 + head + position * 2 + class * 3) % 9];
                    let mut dimension = class;
                    while dimension < Qwen38_27B::HEAD_DIM {
                        let offset = cache_offset(physical, head, position, dimension);
                        key_pages[offset] = key;
                        value_pages[offset] = value;
                        dimension += 8;
                    }
                }
            }
        }
    }
    let block_tables = (0..TABLE_ROWS)
        .flat_map(|row| {
            (0..TABLE_STRIDE)
                .map(move |page| u32::try_from((page * 5 + row * 17) % PHYSICAL_PAGES).unwrap())
        })
        .collect::<Vec<_>>();

    Fixture {
        query,
        key_pages,
        value_pages,
        block_tables,
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
    arena.copy_from_host(stream, regions.table_rows, &TABLE_ROW_IDS)?;
    arena.copy_from_host(stream, regions.lengths, &LENGTHS)
}

fn reset_observed(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.partial_maximum, BYTE_SENTINEL)?;
    arena.fill(stream, regions.partial_denominator, BYTE_SENTINEL)?;
    arena.fill(stream, regions.partial_numerator, BYTE_SENTINEL)?;
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 10]> {
    Ok([
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.lengths)?.addr(),
        arena.address(regions.partial_maximum)?.addr(),
        arena.address(regions.partial_denominator)?.addr(),
        arena.address(regions.partial_numerator)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn maximum_length(batch: usize) -> usize {
    LENGTHS[..batch]
        .iter()
        .copied()
        .max()
        .expect("exact batch is nonzero") as usize
}

fn launch(
    op: &LongContextPagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: all allocations cover the exact maximum geometry. Every table
    // row is a permutation of the complete 3,438-page resident pool.
    unsafe {
        op.launch(
            stream,
            batch,
            maximum_length(batch),
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.lengths)?,
            arena.address(regions.partial_maximum)?,
            arena.address(regions.partial_denominator)?,
            arena.address(regions.partial_numerator)?,
            arena.address(regions.output)?,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn copy_observed(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        partial_maximum: arena.copy_to_host(stream, regions.partial_maximum)?,
        partial_denominator: arena.copy_to_host(stream, regions.partial_denominator)?,
        partial_numerator: arena.copy_to_host(stream, regions.partial_numerator)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn oracle(fixture: &Fixture) -> Result<Oracle, LongContextPagedGqaQualificationError> {
    let sentinel = f32::from_bits(F32_SENTINEL_BITS);
    let mut result = Oracle {
        partial_maximum: vec![sentinel; scalar_elements()],
        partial_denominator: vec![sentinel; scalar_elements()],
        partial_numerator: vec![sentinel; numerator_elements()],
        output: vec![sentinel; MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS],
    };

    for token in 0..MAX_BATCH {
        let table_row = TABLE_ROW_IDS[token] as usize;
        let length = LENGTHS[token] as usize;
        let partitions = length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let kv_head = query_head / (Qwen38_27B::NUM_ATTENTION_HEADS / Qwen38_27B::NUM_KV_HEADS);
            let head_token = token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head;
            let query_base = head_token * Qwen38_27B::HEAD_DIM;
            let mut partition_maxima = Vec::with_capacity(partitions);
            let mut partition_denominators = Vec::with_capacity(partitions);
            let mut partition_numerators = Vec::with_capacity(partitions);
            for partition in 0..partitions {
                let first = partition * LONG_CONTEXT_GQA_PARTITION_SIZE;
                let end = (first + LONG_CONTEXT_GQA_PARTITION_SIZE).min(length);
                let mut maximum = f64::NEG_INFINITY;
                let mut denominator = 0.0f64;
                let mut numerator = [0.0f64; 8];
                for position in first..end {
                    let physical = fixture.block_tables
                        [table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE]
                        as usize;
                    let page_position = position & (ATTENTION_PAGE_SIZE - 1);
                    let mut score = 0.0f64;
                    let mut class = 0usize;
                    while class < 8 {
                        let key = fixture.key_pages
                            [cache_offset(physical, kv_head, page_position, class)];
                        score += 32.0
                            * f64::from(fixture.query[query_base + class])
                            * decode_e4m3(key)
                            * f64::from(KEY_SCALE);
                        class += 1;
                    }
                    score *= 0.0625;
                    let weight = if score > maximum {
                        let old_scale = (maximum - score).exp();
                        denominator = denominator * old_scale + 1.0;
                        for value in &mut numerator {
                            *value *= old_scale;
                        }
                        maximum = score;
                        1.0
                    } else {
                        let weight = (score - maximum).exp();
                        denominator += weight;
                        weight
                    };
                    class = 0;
                    while class < 8 {
                        let value = fixture.value_pages
                            [cache_offset(physical, kv_head, page_position, class)];
                        numerator[class] += weight * decode_e4m3(value) * f64::from(VALUE_SCALE);
                        class += 1;
                    }
                }
                let partial = head_token * LONG_CONTEXT_GQA_MAX_PARTITIONS + partition;
                result.partial_maximum[partial] = maximum as f32;
                result.partial_denominator[partial] = denominator as f32;
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    result.partial_numerator[partial * Qwen38_27B::HEAD_DIM + dimension] =
                        numerator[dimension & 7] as f32;
                }
                partition_maxima.push(maximum);
                partition_denominators.push(denominator);
                partition_numerators.push(numerator);
            }

            let maximum = partition_maxima
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut denominator = 0.0f64;
            let mut numerator = [0.0f64; 8];
            for partition in 0..partitions {
                let weight = (partition_maxima[partition] - maximum).exp();
                denominator += weight * partition_denominators[partition];
                for class in 0..8 {
                    numerator[class] += weight * partition_numerators[partition][class];
                }
            }
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                result.output[query_base + dimension] =
                    (numerator[dimension & 7] / denominator) as f32;
            }
        }
    }

    Ok(result)
}

fn verify_oracle(
    batch: usize,
    oracle: &Oracle,
    observed: &Observed,
    report: &mut LongContextPagedGqaQualification,
) -> Result<(), LongContextPagedGqaQualificationError> {
    verify_scalar_plane(
        batch,
        "partial maximum",
        &observed.partial_maximum,
        &oracle.partial_maximum,
        0.004,
        0.004,
        &mut report.partial_scalar_values,
        &mut report.untouched_values,
        &mut report.maximum_partial_maximum_error,
    )?;
    verify_scalar_plane(
        batch,
        "partial denominator",
        &observed.partial_denominator,
        &oracle.partial_denominator,
        0.02,
        0.008,
        &mut report.partial_scalar_values,
        &mut report.untouched_values,
        &mut report.maximum_partial_denominator_error,
    )?;
    verify_numerator_plane(batch, observed, oracle, report)?;
    verify_output(batch, observed, oracle, report)
}

#[allow(clippy::too_many_arguments)]
fn verify_scalar_plane(
    batch: usize,
    seam: &str,
    observed: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
    active_count: &mut usize,
    untouched_count: &mut usize,
    maximum_error: &mut f32,
) -> Result<(), LongContextPagedGqaQualificationError> {
    for (token, &length) in LENGTHS.iter().enumerate() {
        let partitions = (length as usize).div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let base = (token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head)
                * LONG_CONTEXT_GQA_MAX_PARTITIONS;
            for partition in 0..LONG_CONTEXT_GQA_MAX_PARTITIONS {
                let index = base + partition;
                if token < batch && partition < partitions {
                    compare_value(
                        batch,
                        seam,
                        index,
                        observed[index],
                        expected[index],
                        absolute_tolerance,
                        relative_tolerance,
                        maximum_error,
                    )?;
                    *active_count += 1;
                } else {
                    require_sentinel(batch, seam, index, observed[index])?;
                    *untouched_count += 1;
                }
            }
        }
    }

    Ok(())
}

fn verify_numerator_plane(
    batch: usize,
    observed: &Observed,
    oracle: &Oracle,
    report: &mut LongContextPagedGqaQualification,
) -> Result<(), LongContextPagedGqaQualificationError> {
    for (token, &length) in LENGTHS.iter().enumerate() {
        let partitions = (length as usize).div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let partial_base = (token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head)
                * LONG_CONTEXT_GQA_MAX_PARTITIONS;
            for partition in 0..LONG_CONTEXT_GQA_MAX_PARTITIONS {
                let base = (partial_base + partition) * Qwen38_27B::HEAD_DIM;
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let index = base + dimension;
                    if token < batch && partition < partitions {
                        compare_value(
                            batch,
                            "partial numerator",
                            index,
                            observed.partial_numerator[index],
                            oracle.partial_numerator[index],
                            0.02,
                            0.008,
                            &mut report.maximum_partial_numerator_error,
                        )?;
                        report.partial_numerator_values += 1;
                    } else {
                        require_sentinel(
                            batch,
                            "partial numerator",
                            index,
                            observed.partial_numerator[index],
                        )?;
                        report.untouched_values += 1;
                    }
                }
            }
        }
    }

    Ok(())
}

fn verify_output(
    batch: usize,
    observed: &Observed,
    oracle: &Oracle,
    report: &mut LongContextPagedGqaQualification,
) -> Result<(), LongContextPagedGqaQualificationError> {
    let active = batch * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    for index in 0..active {
        compare_value(
            batch,
            "output",
            index,
            observed.output[index],
            oracle.output[index],
            0.004,
            0.006,
            &mut report.maximum_output_error,
        )?;
        report.output_values += 1;
    }
    for index in active..observed.output.len() {
        require_sentinel(batch, "output", index, observed.output[index])?;
        report.untouched_values += 1;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compare_value(
    batch: usize,
    seam: &str,
    index: usize,
    actual: f32,
    expected: f32,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    maximum_error: &mut f32,
) -> Result<(), LongContextPagedGqaQualificationError> {
    let error = (actual - expected).abs();
    *maximum_error = (*maximum_error).max(error);
    let tolerance = absolute_tolerance.max(expected.abs() * relative_tolerance);
    if !actual.is_finite() || error > tolerance {
        return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
            "{seam} at B={batch}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

fn require_sentinel(
    batch: usize,
    seam: &str,
    index: usize,
    actual: f32,
) -> Result<(), LongContextPagedGqaQualificationError> {
    if actual.to_bits() != F32_SENTINEL_BITS {
        return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
            "B={batch} modified inactive {seam} word {index}"
        )));
    }

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut LongContextPagedGqaQualification,
) -> Result<(), LongContextPagedGqaQualificationError> {
    macro_rules! check {
        ($name:literal, $eager:expr, $replay:expr) => {{
            let eager = $eager;
            let replay = $replay;
            if let Some(index) = replay
                .iter()
                .map(|value| value.to_bits())
                .zip(eager.iter().map(|value| value.to_bits()))
                .position(|(actual, expected)| actual != expected)
            {
                return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
                    "B={batch} graph {} word {index} differs from eager",
                    $name
                )));
            }
            report.graph_replay_values += replay.len();
        }};
    }

    check!(
        "partial maximum",
        &eager.partial_maximum,
        &replay.partial_maximum
    );
    check!(
        "partial denominator",
        &eager.partial_denominator,
        &replay.partial_denominator
    );
    check!(
        "partial numerator",
        &eager.partial_numerator,
        &replay.partial_numerator
    );
    check!("output", &eager.output, &replay.output);

    Ok(())
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut LongContextPagedGqaQualification,
) -> Result<(), LongContextPagedGqaQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
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
    check!(regions.table_rows, &TABLE_ROW_IDS, "table rows");
    check!(regions.lengths, &LENGTHS, "lengths");

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &LongContextPagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), LongContextPagedGqaQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graphs[batch - 1].launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(LongContextPagedGqaQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
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

fn cache_offset(physical: usize, head: usize, position: usize, dimension: usize) -> usize {
    Qwen38_27B::HEAD_DIM
        * (position + ATTENTION_PAGE_SIZE * (head + Qwen38_27B::NUM_KV_HEADS * physical))
        + dimension
}

#[cfg(test)]
mod tests {
    use super::{
        LENGTHS, MAX_BATCH, PHYSICAL_PAGES, Qwen38_27B, TABLE_ROWS, TABLE_STRIDE, layout,
        numerator_elements, qualify_long_context_paged_gqa, scalar_elements,
    };
    use tuisko_kernels_sm120::{LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_PARTITION_SIZE};
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::LongContextPagedGqaQualificationError> {
        let report = qualify_long_context_paged_gqa()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let weighted_partitions = LENGTHS
            .iter()
            .enumerate()
            .map(|(token, &length)| {
                (MAX_BATCH - token) * (length as usize).div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE)
            })
            .sum::<usize>();
        let active_partials = weighted_partitions * Qwen38_27B::NUM_ATTENTION_HEADS;
        let complete_observed = 2 * scalar_elements()
            + numerator_elements()
            + MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;

        assert_eq!(
            report.output_values,
            active_tokens * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
        );
        assert_eq!(report.partial_scalar_values, 2 * active_partials);
        assert_eq!(
            report.partial_numerator_values,
            active_partials * Qwen38_27B::HEAD_DIM
        );
        let active_observed =
            report.output_values + report.partial_scalar_values + report.partial_numerator_values;
        assert_eq!(
            report.untouched_values,
            MAX_BATCH * complete_observed - active_observed
        );
        assert_eq!(report.untouched_values, 306_149_712);
        assert_eq!(report.immutable_input_values, 450_702_208);
        assert_eq!(report.graph_replay_values, MAX_BATCH * complete_observed);
        assert_eq!(TABLE_ROWS, MAX_BATCH);
        assert_eq!(TABLE_STRIDE, PHYSICAL_PAGES);
        assert_eq!(LONG_CONTEXT_GQA_MAX_PARTITIONS, 860);
        assert!(report.maximum_output_error <= 0.004);
        assert!(report.maximum_partial_maximum_error <= 0.004);
        let (arena, regions) = layout()?;
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(report.workspace_bytes, regions.workspace_bytes());
        assert_eq!(
            report.padding_bytes,
            arena.byte_len() - regions.payload_bytes()
        );
        assert_eq!(report.workspace_bytes, 170_403_840);
        assert_eq!(report.arena_bytes, 621_533_184);
        assert_eq!(report.padding_bytes, 512);

        Ok(())
    }
}

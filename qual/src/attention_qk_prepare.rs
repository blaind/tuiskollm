//! Numerical and graph qualification for full-attention Q/K preparation.

use crate::fp8_projection_oracle::{
    BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, encode_e4m3fn, f32_to_bf16,
};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, AttentionQkPrepareOp};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 16;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 2;
const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const CACHE_POSITIONS: [u32; MAX_BATCH] = [63, 64, 1, 126, 2, 65, 127, 0];
const INPUT_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
const NORM_PATTERN: [f32; 8] = [-0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25, -0.1875];

/// Failure of the exact attention Q/K preparation gate.
#[derive(Debug, thiserror::Error)]
pub enum AttentionQkPrepareQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("attention Q/K preparation qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors across every exact decode route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionQkPrepareQualification {
    /// Active FP32 query values compared with the independent formula.
    pub query_values: usize,
    /// Appended represented E4M3 key and value codes compared bit-exactly.
    pub appended_cache_codes: usize,
    /// Cache bytes and inactive query words proved untouched.
    pub untouched_values: usize,
    /// Read-only input and metadata values proved unchanged.
    pub immutable_input_values: usize,
    /// Complete output/cache state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute prepared-query error.
    pub maximum_query_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    qkv: ArenaRegion<u16>,
    query_norm: ArenaRegion<u16>,
    key_norm: ArenaRegion<u16>,
    rope_cos: ArenaRegion<f32>,
    rope_sin: ArenaRegion<f32>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    cache_positions: ArenaRegion<u32>,
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
            + self.table_rows.byte_len()
            + self.cache_positions.byte_len()
            + self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
    }
}

struct Fixture {
    qkv: Vec<u16>,
    query_norm: Vec<u16>,
    key_norm: Vec<u16>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    block_tables: Vec<u32>,
}

struct Observed {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
}

/// Qualifies eager and captured Q/K preparation routes at exact `B=1..=8`.
pub fn qualify_attention_qk_prepare()
-> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = AttentionQkPrepareOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = AttentionQkPrepareQualification {
        query_values: 0,
        appended_cache_codes: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_query_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle(batch, &fixture, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &AttentionQkPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), AttentionQkPrepareQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        graph.launch(stream)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            graphs[batch - 1].launch(stream)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let query_norm = layout.reserve(Qwen38_27B::HEAD_DIM, ALIGNMENT)?;
    let key_norm = layout.reserve(Qwen38_27B::HEAD_DIM, ALIGNMENT)?;
    let rope_cos = layout.reserve(MAX_BATCH * ROTARY_PAIRS, ALIGNMENT)?;
    let rope_sin = layout.reserve(MAX_BATCH * ROTARY_PAIRS, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let cache_positions = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let query = layout.reserve(MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
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
            table_rows,
            cache_positions,
            query,
            key_pages,
            value_pages,
        },
    ))
}

fn fixture() -> Fixture {
    let qkv = (0..MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::ATTENTION_QKV_ROWS;
            f32_to_bf16(INPUT_PATTERN[(index + 3 * token) & 15] * (1.0 - token as f32 / 16.0))
        })
        .collect();
    let query_norm = (0..Qwen38_27B::HEAD_DIM)
        .map(|index| f32_to_bf16(NORM_PATTERN[(index + 3) & 7]))
        .collect();
    let key_norm = (0..Qwen38_27B::HEAD_DIM)
        .map(|index| f32_to_bf16(NORM_PATTERN[(index + 5) & 7]))
        .collect();
    let positions = [
        [0u32, 1, 63, 64, 127, 128, 511, 512],
        [0u32, 2, 17, 33, 49, 65, 81, 97],
        [0u32, 3, 19, 35, 51, 67, 83, 99],
    ];
    let (rope_cos, rope_sin) = make_mrope_coefficients(&positions);

    Fixture {
        qkv,
        query_norm,
        key_norm,
        rope_cos,
        rope_sin,
        block_tables: (0..TABLE_ROWS * TABLE_STRIDE)
            .map(|page| page as u32)
            .collect(),
    }
}

fn make_mrope_coefficients(positions: &[[u32; MAX_BATCH]; 3]) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0f32; MAX_BATCH * ROTARY_PAIRS];
    let mut sine = vec![0.0f32; MAX_BATCH * ROTARY_PAIRS];
    for token in 0..MAX_BATCH {
        for pair in 0..ROTARY_PAIRS {
            // Consecutive pairs cycle temporal/height/width, yielding the
            // checkpoint's exact 32-pair [11, 11, 10] MRoPE partition.
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
            let angle = f64::from(positions[pair % 3][token]) * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[token * ROTARY_PAIRS + pair] = cos as f32;
            sine[token * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn load_fixture(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.qkv, &fixture.qkv)?;
    arena.copy_from_host(stream, regions.query_norm, &fixture.query_norm)?;
    arena.copy_from_host(stream, regions.key_norm, &fixture.key_norm)?;
    arena.copy_from_host(stream, regions.rope_cos, &fixture.rope_cos)?;
    arena.copy_from_host(stream, regions.rope_sin, &fixture.rope_sin)?;
    arena.copy_from_host(stream, regions.block_tables, &fixture.block_tables)?;
    arena.copy_from_host(stream, regions.table_rows, &TABLE_ROW_IDS)?;
    arena.copy_from_host(stream, regions.cache_positions, &CACHE_POSITIONS)
}

fn reset_outputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.query, BYTE_SENTINEL)?;
    arena.fill(stream, regions.key_pages, BYTE_SENTINEL)?;
    arena.fill(stream, regions.value_pages, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 11]> {
    Ok([
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.query_norm)?.addr(),
        arena.address(regions.key_norm)?.addr(),
        arena.address(regions.rope_cos)?.addr(),
        arena.address(regions.rope_sin)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.cache_positions)?.addr(),
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
    ])
}

fn launch(
    op: &AttentionQkPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: all regions cover the maximum batch, metadata selects valid
    // table rows/pages, and both cache planes own all sixteen physical pages.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.qkv)?,
            arena.address(regions.query_norm)?,
            arena.address(regions.key_norm)?,
            arena.address(regions.rope_cos)?,
            arena.address(regions.rope_sin)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.cache_positions)?,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn observe(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<Observed> {
    Ok(Observed {
        query: arena.copy_to_host(stream, regions.query)?,
        key_pages: arena.copy_to_host(stream, regions.key_pages)?,
        value_pages: arena.copy_to_host(stream, regions.value_pages)?,
    })
}

fn verify_oracle(
    batch: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut AttentionQkPrepareQualification,
) -> Result<(), AttentionQkPrepareQualificationError> {
    let (query, key_pages, value_pages) = oracle(batch, fixture)?;
    let active_query = batch * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    for (index, (&actual, &expected)) in observed.query[..active_query]
        .iter()
        .zip(&query[..active_query])
        .enumerate()
    {
        let error = (actual - expected).abs();
        report.maximum_query_error = report.maximum_query_error.max(error);
        let tolerance = 0.002f32.max(expected.abs() * 0.003);
        if !actual.is_finite() || error > tolerance {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "query at B={batch}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    for (index, value) in observed.query[active_query..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "B={batch} modified inactive query word {}",
                active_query + index
            )));
        }
    }
    compare_cache(batch, "key", &observed.key_pages, &key_pages)?;
    compare_cache(batch, "value", &observed.value_pages, &value_pages)?;

    let appended = batch * 2 * Qwen38_27B::ATTENTION_KV_ROWS;
    report.query_values += active_query;
    report.appended_cache_codes += appended;
    report.untouched_values +=
        observed.query.len() - active_query + observed.key_pages.len() + observed.value_pages.len()
            - appended;

    Ok(())
}

fn compare_cache(
    batch: usize,
    name: &str,
    actual: &[u8],
    expected: &[u8],
) -> Result<(), AttentionQkPrepareQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "{name} cache at B={batch}, byte={index}: device={:#04x}, oracle={:#04x}",
            actual[index], expected[index]
        )));
    }
    Ok(())
}

fn oracle(
    batch: usize,
    fixture: &Fixture,
) -> Result<(Vec<f32>, Vec<u8>, Vec<u8>), AttentionQkPrepareQualificationError> {
    let mut query =
        vec![f32::from_bits(F32_SENTINEL_BITS); MAX_BATCH * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS];
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let mut key_pages = vec![BYTE_SENTINEL; plane_bytes];
    let mut value_pages = vec![BYTE_SENTINEL; plane_bytes];

    for token in 0..batch {
        let token_base = token * Qwen38_27B::ATTENTION_QKV_ROWS;
        let cosine = &fixture.rope_cos[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        let sine = &fixture.rope_sin[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        for head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let source = token_base + head * 2 * Qwen38_27B::HEAD_DIM;
            let destination =
                (token * Qwen38_27B::NUM_ATTENTION_HEADS + head) * Qwen38_27B::HEAD_DIM;
            normalize_rotate_oracle(
                &fixture.qkv[source..source + Qwen38_27B::HEAD_DIM],
                &fixture.query_norm,
                cosine,
                sine,
                &mut query[destination..destination + Qwen38_27B::HEAD_DIM],
            );
        }

        let table_row = TABLE_ROW_IDS[token] as usize;
        let position = CACHE_POSITIONS[token] as usize;
        let physical_page = fixture.block_tables
            [table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE]
            as usize;
        let key_source = token_base + Qwen38_27B::ATTENTION_QUERY_ROWS;
        let value_source = key_source + Qwen38_27B::ATTENTION_KV_ROWS;
        for head in 0..Qwen38_27B::NUM_KV_HEADS {
            let mut normalized = vec![0.0f32; Qwen38_27B::HEAD_DIM];
            let source = key_source + head * Qwen38_27B::HEAD_DIM;
            normalize_rotate_oracle(
                &fixture.qkv[source..source + Qwen38_27B::HEAD_DIM],
                &fixture.key_norm,
                cosine,
                sine,
                &mut normalized,
            );
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let destination = cache_offset(physical_page, head, position, dimension);
                key_pages[destination] = encode_e4m3fn(normalized[dimension] / KEY_SCALE)
                    .map_err(AttentionQkPrepareQualificationError::Mismatch)?;
                value_pages[destination] = encode_e4m3fn(
                    bf16_to_f32(
                        fixture.qkv[value_source + head * Qwen38_27B::HEAD_DIM + dimension],
                    ) / VALUE_SCALE,
                )
                .map_err(AttentionQkPrepareQualificationError::Mismatch)?;
            }
        }
    }

    Ok((query, key_pages, value_pages))
}

fn normalize_rotate_oracle(
    source: &[u16],
    norm: &[u16],
    cosine: &[f32],
    sine: &[f32],
    output: &mut [f32],
) {
    let sum = source
        .iter()
        .map(|&bits| {
            let value = f64::from(bf16_to_f32(bits));
            value * value
        })
        .sum::<f64>();
    let inverse_rms =
        1.0 / (sum / Qwen38_27B::HEAD_DIM as f64 + f64::from(Qwen38_27B::RMS_NORM_EPSILON)).sqrt();
    let normalized = source
        .iter()
        .zip(norm)
        .map(|(&value, &weight)| {
            f64::from(bf16_to_f32(value)) * inverse_rms * (1.0 + f64::from(bf16_to_f32(weight)))
        })
        .collect::<Vec<_>>();
    for dimension in 0..Qwen38_27B::HEAD_DIM {
        output[dimension] = if dimension < ROTARY_PAIRS {
            (normalized[dimension] * f64::from(cosine[dimension])
                - normalized[dimension + ROTARY_PAIRS] * f64::from(sine[dimension]))
                as f32
        } else if dimension < ROTARY_DIM {
            let pair = dimension - ROTARY_PAIRS;
            (normalized[pair] * f64::from(sine[pair])
                + normalized[dimension] * f64::from(cosine[pair])) as f32
        } else {
            normalized[dimension] as f32
        };
    }
}

fn cache_offset(physical_page: usize, head: usize, position: usize, dimension: usize) -> usize {
    Qwen38_27B::HEAD_DIM
        * ((position & (ATTENTION_PAGE_SIZE - 1))
            + ATTENTION_PAGE_SIZE * (head + Qwen38_27B::NUM_KV_HEADS * physical_page))
        + dimension
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut AttentionQkPrepareQualification,
) -> Result<(), AttentionQkPrepareQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                    "read-only {} changed at index {index}",
                    $name
                )));
            }
            report.immutable_input_values += actual.len();
        }};
    }

    check!(regions.qkv, &fixture.qkv, "QKV");
    check!(regions.query_norm, &fixture.query_norm, "query norm");
    check!(regions.key_norm, &fixture.key_norm, "key norm");
    check!(regions.rope_cos, &fixture.rope_cos, "rotary cosine");
    check!(regions.rope_sin, &fixture.rope_sin, "rotary sine");
    check!(regions.block_tables, &fixture.block_tables, "block table");
    check!(regions.table_rows, &TABLE_ROW_IDS, "table rows");
    check!(regions.cache_positions, &CACHE_POSITIONS, "cache positions");

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut AttentionQkPrepareQualification,
) -> Result<(), AttentionQkPrepareQualificationError> {
    if let Some(index) = replay
        .query
        .iter()
        .map(|value| value.to_bits())
        .zip(eager.query.iter().map(|value| value.to_bits()))
        .position(|(actual, expected)| actual != expected)
    {
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "B={batch} graph query word {index} differs from eager"
        )));
    }
    compare_cache(batch, "graph key", &replay.key_pages, &eager.key_pages)?;
    compare_cache(
        batch,
        "graph value",
        &replay.value_pages,
        &eager.value_pages,
    )?;
    report.graph_replay_values +=
        replay.query.len() + replay.key_pages.len() + replay.value_pages.len();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, PHYSICAL_PAGES, Qwen38_27B, TABLE_ROWS, TABLE_STRIDE,
        qualify_attention_qk_prepare,
    };
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::AttentionQkPrepareQualificationError> {
        let report = qualify_attention_qk_prepare()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let query_per_token = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
        let cache_per_token = 2 * Qwen38_27B::ATTENTION_KV_ROWS;
        let plane_bytes =
            PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
        let replay_per_route = MAX_BATCH * query_per_token + 2 * plane_bytes;

        assert_eq!(report.query_values, active_tokens * query_per_token);
        assert_eq!(report.appended_cache_codes, active_tokens * cache_per_token);
        assert_eq!(report.untouched_values, 16_875_520);
        assert_eq!(report.immutable_input_values, 1_851_904);
        assert_eq!(report.graph_replay_values, MAX_BATCH * replay_per_route);
        assert_eq!(TABLE_ROWS * TABLE_STRIDE, PHYSICAL_PAGES);
        assert_eq!(report.arena_bytes - report.padding_bytes, 2_526_336);
        assert!(report.maximum_query_error <= 0.003);

        Ok(())
    }
}

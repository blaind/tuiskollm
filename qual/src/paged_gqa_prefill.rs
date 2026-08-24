//! Numerical and graph qualification for exact shared paged-GQA prefill tails.

use crate::fp8_projection_oracle::{BYTE_SENTINEL, F32_SENTINEL_BITS};
use crate::{DeviceBenchmarkError, device_benchmark};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, PagedGqaOp};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_TOKENS: usize = 128;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const PREFIX_TOKENS: usize = 2;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 24;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const BLOCK_TABLES: [u32; TABLE_ROWS * TABLE_STRIDE] = [
    17, 2, 21, 4, 15, 0, 23, 7, 12, 1, 18, 9, 14, 5, 22, 8, 19, 3, 20, 6, 13, 10, 16, 11,
];
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
const KEY_CODES: [u8; 9] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20, 0xa0];
const VALUE_CODES: [u8; 9] = [0x38, 0xb8, 0x30, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];

/// Failure of the exact shared prefill paged-GQA qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum PagedGqaPrefillQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("paged GQA prefill qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error across every exact shared prefill route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedGqaPrefillQualification {
    /// Active FP32 outputs compared with the independent FP64 softmax.
    pub output_values: usize,
    /// Inactive FP32 output words proved untouched.
    pub untouched_values: usize,
    /// Read-only query, cache, and page metadata values proved unchanged.
    pub immutable_input_values: usize,
    /// Complete output state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute output error.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    query: ArenaRegion<f32>,
    key_pages: ArenaRegion<u8>,
    value_pages: ArenaRegion<u8>,
    block_tables: ArenaRegion<u32>,
    table_rows: ArenaRegion<u32>,
    lengths: ArenaRegion<u32>,
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
            + self.output.byte_len()
    }
}

struct Fixture {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    table_rows: Vec<u32>,
    lengths: Vec<u32>,
}

/// Qualifies eager and captured shared paged-GQA routes at exact `T=32/64/128`.
pub fn qualify_paged_gqa_prefill()
-> Result<PagedGqaPrefillQualification, PagedGqaPrefillQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
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
    let mut report = PagedGqaPrefillQualification {
        output_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for tokens in PREFILL_ROUTES {
        reset_output(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, tokens)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_oracle(tokens, &fixture, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_output(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, tokens))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(tokens, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
                "device addresses changed while qualifying T={tokens}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_TOKENS, ALIGNMENT)?;
    let output = layout.reserve(MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            output,
        },
    ))
}

fn fixture() -> Fixture {
    let query = (0..MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let token = index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            let head = index / Qwen38_27B::HEAD_DIM % Qwen38_27B::NUM_ATTENTION_HEADS;
            QUERY_PATTERN[(index + head * 3 + token * 5) & 15] * (1.0 - (token & 15) as f32 / 32.0)
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
                        KEY_CODES[(physical * 5 + head * 3 + position + dimension) % 9];
                    value_pages[offset] =
                        VALUE_CODES[(physical * 7 + head + position * 2 + dimension * 3) % 9];
                }
            }
        }
    }
    let table_rows = (0..MAX_TOKENS)
        .map(|token| ((token / 2) % TABLE_ROWS) as u32)
        .collect();
    let lengths = (0..MAX_TOKENS)
        .map(|token| (PREFIX_TOKENS + token + 1) as u32)
        .collect();

    Fixture {
        query,
        key_pages,
        value_pages,
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
    arena.copy_from_host(stream, regions.block_tables, &BLOCK_TABLES)?;
    arena.copy_from_host(stream, regions.table_rows, &fixture.table_rows)?;
    arena.copy_from_host(stream, regions.lengths, &fixture.lengths)
}

fn reset_output(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> GpuResult<()> {
    arena.fill(stream, regions.output, BYTE_SENTINEL)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 7]> {
    Ok([
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.table_rows)?.addr(),
        arena.address(regions.lengths)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &PagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    // SAFETY: all allocations cover T=128. Adjacent tokens share a table row,
    // every causal length is nonzero and fits three entries, and all pages are owned.
    unsafe {
        op.launch_prefill_shared(
            stream,
            tokens,
            arena.address(regions.query)?,
            arena.address(regions.key_pages)?,
            arena.address(regions.value_pages)?,
            arena.address(regions.block_tables)?,
            arena.address(regions.table_rows)?,
            TABLE_STRIDE,
            arena.address(regions.lengths)?,
            arena.address(regions.output)?,
            KEY_SCALE,
            VALUE_SCALE,
        )
    }
}

fn verify_oracle(
    tokens: usize,
    fixture: &Fixture,
    observed: &[f32],
    report: &mut PagedGqaPrefillQualification,
) -> Result<(), PagedGqaPrefillQualificationError> {
    let expected = oracle(tokens, fixture);
    let active = tokens * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    for (index, (&actual, &expected)) in observed[..active]
        .iter()
        .zip(&expected[..active])
        .enumerate()
    {
        let error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = 0.002f32.max(expected.abs() * 0.003);
        if !actual.is_finite() || error > tolerance {
            return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
                "output at T={tokens}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    for (index, value) in observed[active..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
                "T={tokens} modified inactive output word {}",
                active + index
            )));
        }
    }
    report.output_values += active;
    report.untouched_values += observed.len() - active;

    Ok(())
}

fn oracle(tokens: usize, fixture: &Fixture) -> Vec<f32> {
    let mut output =
        vec![f32::from_bits(F32_SENTINEL_BITS); MAX_TOKENS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS];
    for token in 0..tokens {
        let row = fixture.table_rows[token] as usize;
        let length = fixture.lengths[token] as usize;
        for query_head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            let kv_head = query_head / (Qwen38_27B::NUM_ATTENTION_HEADS / Qwen38_27B::NUM_KV_HEADS);
            let query_base =
                (token * Qwen38_27B::NUM_ATTENTION_HEADS + query_head) * Qwen38_27B::HEAD_DIM;
            let mut scores = Vec::with_capacity(length);
            for position in 0..length {
                let physical =
                    BLOCK_TABLES[row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE] as usize;
                let key_base =
                    cache_offset(physical, kv_head, position & (ATTENTION_PAGE_SIZE - 1), 0);
                let score = fixture.query[query_base..query_base + Qwen38_27B::HEAD_DIM]
                    .iter()
                    .zip(&fixture.key_pages[key_base..key_base + Qwen38_27B::HEAD_DIM])
                    .map(|(&query, &code)| {
                        f64::from(query) * decode_e4m3(code) * f64::from(KEY_SCALE)
                    })
                    .sum::<f64>()
                    * 0.0625;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let mut numerator = 0.0f64;
                for (position, &weight) in weights.iter().enumerate() {
                    let physical =
                        BLOCK_TABLES[row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE] as usize;
                    let offset = cache_offset(
                        physical,
                        kv_head,
                        position & (ATTENTION_PAGE_SIZE - 1),
                        dimension,
                    );
                    numerator +=
                        weight * decode_e4m3(fixture.value_pages[offset]) * f64::from(VALUE_SCALE);
                }
                output[query_base + dimension] = (numerator / denominator) as f32;
            }
        }
    }

    output
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

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut PagedGqaPrefillQualification,
) -> Result<(), PagedGqaPrefillQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
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
    check!(regions.block_tables, &BLOCK_TABLES, "block tables");
    check!(regions.table_rows, &fixture.table_rows, "table rows");
    check!(regions.lengths, &fixture.lengths, "lengths");

    Ok(())
}

fn verify_replay(
    tokens: usize,
    eager: &[f32],
    replay: &[f32],
    report: &mut PagedGqaPrefillQualification,
) -> Result<(), PagedGqaPrefillQualificationError> {
    if let Some(index) = replay
        .iter()
        .map(|value| value.to_bits())
        .zip(eager.iter().map(|value| value.to_bits()))
        .position(|(actual, expected)| actual != expected)
    {
        return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
            "T={tokens} graph output word {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &PagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), PagedGqaPrefillQualificationError> {
    let graphs = PREFILL_ROUTES
        .iter()
        .map(|&tokens| CudaGraph::capture(stream, || launch(op, arena, stream, regions, tokens)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..8 {
        for &route in &[2usize, 0, 1] {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graphs[route].launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(PagedGqaPrefillQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_TOKENS, PREFILL_ROUTES, Qwen38_27B, layout, qualify_paged_gqa_prefill};
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn paged_gqa_suite_exact_prefill_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaPrefillQualificationError> {
        let report = qualify_paged_gqa_prefill()?;
        let active_tokens = PREFILL_ROUTES.iter().sum::<usize>();
        let output_per_token = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(
            report.untouched_values,
            PREFILL_ROUTES
                .iter()
                .map(|tokens| (MAX_TOKENS - tokens) * output_per_token)
                .sum::<usize>()
        );
        assert_eq!(
            report.graph_replay_values,
            PREFILL_ROUTES.len() * MAX_TOKENS * output_per_token
        );
        assert_eq!(report.arena_bytes - report.padding_bytes, 9_438_304);
        assert_eq!(report.immutable_input_values, 23_594_640);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout()?;
        assert_eq!(arena.byte_len(), 9_438_464);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 160);

        Ok(())
    }
}

//! Numerical and graph qualification for exact paged GQA decode.

use crate::fp8_projection_oracle::{BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16};
use crate::{DeviceBenchmarkError, device_benchmark};
use std::{mem::size_of, sync::Arc};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, PagedGqaOp, Qwen35PagedGqaOp};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 24;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 3;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const TABLE_ROW_IDS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const LENGTHS: [u32; MAX_BATCH] = [1, 63, 64, 65, 97, 127, 128, 130];
const BLOCK_TABLES: [u32; TABLE_ROWS * TABLE_STRIDE] = [
    17, 2, 21, 4, 15, 0, 23, 7, 12, 1, 18, 9, 14, 5, 22, 8, 19, 3, 20, 6, 13, 10, 16, 11,
];
const QUERY_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
const KEY_CODES: [u8; 9] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20, 0xa0];
const VALUE_CODES: [u8; 9] = [0x38, 0xb8, 0x30, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];

/// Failure of the exact paged GQA qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum PagedGqaQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("paged GQA qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error across every exact route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedGqaQualification {
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
}

#[derive(Clone, Copy)]
enum CacheFormat {
    E4m3,
    Bf16,
}

trait QualifiedPagedGqaOp {
    type Target: Arch;
    const CACHE_ELEMENT_BYTES: usize;
    const CACHE_FORMAT: CacheFormat;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self>
    where
        Self: Sized;

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()>;
}

impl QualifiedPagedGqaOp for PagedGqaOp {
    type Target = Qwen38_27B;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u8>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::E4m3;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        PagedGqaOp::new(context)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        // SAFETY: the qualification arena establishes the complete pointer
        // contract before dispatching through this safe seam.
        unsafe {
            self.launch(
                stream,
                batch,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            )
        }
    }
}

impl QualifiedPagedGqaOp for Qwen35PagedGqaOp {
    type Target = Qwen35_9B;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen35PagedGqaOp::new(context)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        _key_scale: f32,
        _value_scale: f32,
    ) -> GpuResult<()> {
        // SAFETY: the qualification layout reserves aligned BF16 cache
        // planes; byte typing is confined to this shared qualification seam.
        unsafe {
            self.launch(
                stream,
                batch,
                query,
                key_pages.cast(),
                value_pages.cast(),
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            )
        }
    }
}

/// Qualifies eager and captured paged GQA routes at exact `B=1..=8`.
pub fn qualify_paged_gqa() -> Result<PagedGqaQualification, PagedGqaQualificationError> {
    qualify_target::<PagedGqaOp>()
}

/// Qualifies Qwen3.5 eager and captured BF16 paged GQA at exact `B=1..=8`.
pub fn qualify_qwen35_paged_gqa() -> Result<PagedGqaQualification, PagedGqaQualificationError> {
    qualify_target::<Qwen35PagedGqaOp>()
}

fn qualify_target<O: QualifiedPagedGqaOp>()
-> Result<PagedGqaQualification, PagedGqaQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(PagedGqaQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout::<O::Target>(O::CACHE_ELEMENT_BYTES)?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture::<O::Target>(O::CACHE_FORMAT);
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = O::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = PagedGqaQualification {
        output_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        reset_output(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_oracle::<O::Target>(batch, O::CACHE_FORMAT, &fixture, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_output(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        graph.launch(&stream)?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(PagedGqaQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout<A: Arch>(cache_element_bytes: usize) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM * cache_element_bytes;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let lengths = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

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

fn fixture<A: Arch>(cache_format: CacheFormat) -> Fixture {
    let query = (0..MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let token = index / A::ATTENTION_OUTPUT_COLUMNS;
            let head = index / A::HEAD_DIM % A::NUM_ATTENTION_HEADS;
            QUERY_PATTERN[(index + head * 3 + token * 5) & 15] * (1.0 - token as f32 / 16.0)
        })
        .collect::<Vec<_>>();
    let plane_elements = PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM;
    let plane_bytes = plane_elements * cache_element_bytes(cache_format);
    let mut key_pages = vec![0u8; plane_bytes];
    let mut value_pages = vec![0u8; plane_bytes];
    for physical in 0..PHYSICAL_PAGES {
        for head in 0..A::NUM_KV_HEADS {
            for position in 0..ATTENTION_PAGE_SIZE {
                for dimension in 0..A::HEAD_DIM {
                    let offset = cache_offset::<A>(physical, head, position, dimension);
                    let key_code = KEY_CODES[(physical * 5 + head * 3 + position + dimension) % 9];
                    let value_code =
                        VALUE_CODES[(physical * 7 + head + position * 2 + dimension * 3) % 9];
                    write_cache_value(&mut key_pages, offset, cache_format, key_code, KEY_SCALE);
                    write_cache_value(
                        &mut value_pages,
                        offset,
                        cache_format,
                        value_code,
                        VALUE_SCALE,
                    );
                }
            }
        }
    }

    Fixture {
        query,
        key_pages,
        value_pages,
    }
}

fn write_cache_value(
    plane: &mut [u8],
    element: usize,
    format: CacheFormat,
    e4m3_code: u8,
    scale: f32,
) {
    match format {
        CacheFormat::E4m3 => plane[element] = e4m3_code,
        CacheFormat::Bf16 => {
            let byte = element * size_of::<u16>();
            let value = (decode_e4m3(e4m3_code) * f64::from(scale)) as f32;
            plane[byte..byte + size_of::<u16>()].copy_from_slice(&f32_to_bf16(value).to_le_bytes());
        }
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
    arena.copy_from_host(stream, regions.table_rows, &TABLE_ROW_IDS)?;
    arena.copy_from_host(stream, regions.lengths, &LENGTHS)
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
    op: &impl QualifiedPagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    op.launch(
        stream,
        batch,
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

fn verify_oracle<A: Arch>(
    batch: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
    observed: &[f32],
    report: &mut PagedGqaQualification,
) -> Result<(), PagedGqaQualificationError> {
    let expected = oracle::<A>(batch, cache_format, fixture)?;
    let active = batch * A::ATTENTION_OUTPUT_COLUMNS;
    for (index, (&actual, &expected)) in observed[..active]
        .iter()
        .zip(&expected[..active])
        .enumerate()
    {
        let error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        let tolerance = 0.002f32.max(expected.abs() * 0.003);
        if !actual.is_finite() || error > tolerance {
            return Err(PagedGqaQualificationError::Mismatch(format!(
                "output at B={batch}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    for (index, value) in observed[active..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(PagedGqaQualificationError::Mismatch(format!(
                "B={batch} modified inactive output word {}",
                active + index
            )));
        }
    }
    report.output_values += active;
    report.untouched_values += observed.len() - active;

    Ok(())
}

fn oracle<A: Arch>(
    batch: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
) -> Result<Vec<f32>, PagedGqaQualificationError> {
    let mut output =
        vec![f32::from_bits(F32_SENTINEL_BITS); MAX_BATCH * A::ATTENTION_OUTPUT_COLUMNS];
    for token in 0..batch {
        let row = TABLE_ROW_IDS[token] as usize;
        let length = LENGTHS[token] as usize;
        for query_head in 0..A::NUM_ATTENTION_HEADS {
            let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
            let query_base = (token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM;
            let mut scores = Vec::with_capacity(length);
            for position in 0..length {
                let physical =
                    BLOCK_TABLES[row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE] as usize;
                let score = (0..A::HEAD_DIM)
                    .map(|dimension| {
                        let offset = cache_offset::<A>(
                            physical,
                            kv_head,
                            position & (ATTENTION_PAGE_SIZE - 1),
                            dimension,
                        );
                        f64::from(fixture.query[query_base + dimension])
                            * cache_value(&fixture.key_pages, offset, cache_format, KEY_SCALE)
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
            for dimension in 0..A::HEAD_DIM {
                let mut numerator = 0.0f64;
                for (position, &weight) in weights.iter().enumerate() {
                    let physical =
                        BLOCK_TABLES[row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE] as usize;
                    let offset = cache_offset::<A>(
                        physical,
                        kv_head,
                        position & (ATTENTION_PAGE_SIZE - 1),
                        dimension,
                    );
                    numerator += weight
                        * cache_value(&fixture.value_pages, offset, cache_format, VALUE_SCALE);
                }
                output[query_base + dimension] = (numerator / denominator) as f32;
            }
        }
    }

    Ok(output)
}

fn cache_value(plane: &[u8], element: usize, format: CacheFormat, scale: f32) -> f64 {
    match format {
        CacheFormat::E4m3 => decode_e4m3(plane[element]) * f64::from(scale),
        CacheFormat::Bf16 => {
            let byte = element * size_of::<u16>();
            let bits = u16::from_le_bytes([plane[byte], plane[byte + 1]]);
            f64::from(bf16_to_f32(bits))
        }
    }
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

const fn cache_element_bytes(format: CacheFormat) -> usize {
    match format {
        CacheFormat::E4m3 => size_of::<u8>(),
        CacheFormat::Bf16 => size_of::<u16>(),
    }
}

fn cache_offset<A: Arch>(physical: usize, head: usize, position: usize, dimension: usize) -> usize {
    A::HEAD_DIM * (position + ATTENTION_PAGE_SIZE * (head + A::NUM_KV_HEADS * physical)) + dimension
}

fn verify_inputs(
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut PagedGqaQualification,
) -> Result<(), PagedGqaQualificationError> {
    macro_rules! check {
        ($region:expr, $expected:expr, $name:literal) => {{
            let actual = arena.copy_to_host(stream, $region)?;
            if let Some(index) = actual
                .iter()
                .zip($expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(PagedGqaQualificationError::Mismatch(format!(
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
    check!(regions.table_rows, &TABLE_ROW_IDS, "table rows");
    check!(regions.lengths, &LENGTHS, "lengths");

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[f32],
    replay: &[f32],
    report: &mut PagedGqaQualification,
) -> Result<(), PagedGqaQualificationError> {
    if let Some(index) = replay
        .iter()
        .map(|value| value.to_bits())
        .zip(eager.iter().map(|value| value.to_bits()))
        .position(|(actual, expected)| actual != expected)
    {
        return Err(PagedGqaQualificationError::Mismatch(format!(
            "B={batch} graph output word {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &impl QualifiedPagedGqaOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), PagedGqaQualificationError> {
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
        return Err(PagedGqaQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, PHYSICAL_PAGES, Qwen35_9B, Qwen38_27B, TABLE_ROWS, TABLE_STRIDE, layout,
        qualify_paged_gqa, qualify_qwen35_paged_gqa,
    };
    use std::mem::size_of;
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn paged_gqa_suite_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaQualificationError> {
        let report = qualify_paged_gqa()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let output_per_token = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(report.untouched_values, 172_032);
        assert_eq!(
            report.graph_replay_values,
            MAX_BATCH * MAX_BATCH * output_per_token
        );
        assert_eq!(TABLE_ROWS * TABLE_STRIDE, PHYSICAL_PAGES);
        assert_eq!(report.arena_bytes - report.padding_bytes, 3_539_104);
        assert_eq!(report.immutable_input_values, 51_118_720);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout::<Qwen38_27B>(size_of::<u8>())?;
        assert_eq!(arena.byte_len(), 3_539_712);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 608);

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen35_bf16_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaQualificationError> {
        let report = qualify_qwen35_paged_gqa()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let output_per_token = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
        let plane_bytes = PHYSICAL_PAGES
            * Qwen35_9B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen35_9B::HEAD_DIM
            * size_of::<u16>();
        let immutable_per_check = MAX_BATCH * output_per_token
            + 2 * plane_bytes
            + TABLE_ROWS * TABLE_STRIDE
            + 2 * MAX_BATCH;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(report.untouched_values, 114_688);
        assert_eq!(report.graph_replay_values, 262_144);
        assert_eq!(
            report.immutable_input_values,
            2 * MAX_BATCH * immutable_per_check
        );
        assert_eq!(report.arena_bytes - report.padding_bytes, 6_553_760);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout::<Qwen35_9B>(size_of::<u16>())?;
        assert_eq!(arena.byte_len(), 6_554_368);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 608);

        Ok(())
    }
}

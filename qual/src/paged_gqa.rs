//! Numerical and graph qualification for exact paged GQA routes.

use crate::fp8_projection_oracle::{BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16};
use crate::{DeviceBenchmarkError, device_benchmark};
use std::{mem::size_of, sync::Arc};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, MtpBf16PagedGqaOp, PagedGqaOp, Qwen35PagedGqaOp, Qwen36PagedGqaOp,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const DECODE_ROUTES: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];
const QWEN35_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const QWEN35_MAX_TOKENS: usize = 128;
const QWEN36_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const QWEN36_MAX_TOKENS: usize = 128;
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
    /// Exact represented cache bytes in the qualification arena.
    pub cache_bytes: usize,
    /// Exact query, output, and metadata bytes in the qualification arena.
    pub workspace_bytes: usize,
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

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }

    fn workspace_bytes(self) -> usize {
        self.payload_bytes() - self.cache_bytes()
    }
}

struct Fixture {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
    table_rows: Vec<u32>,
    lengths: Vec<u32>,
}

#[derive(Clone, Copy)]
enum CacheFormat {
    E4m3,
    Bf16,
}

trait QualifiedPagedGqaOp {
    type Target: Arch;
    const ROUTES: &'static [usize];
    const MAX_TOKENS: usize;
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
    const ROUTES: &'static [usize] = &DECODE_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
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
    const ROUTES: &'static [usize] = &QWEN35_ROUTES;
    const MAX_TOKENS: usize = QWEN35_MAX_TOKENS;
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

impl QualifiedPagedGqaOp for Qwen36PagedGqaOp {
    type Target = Qwen36Moe35B;
    const ROUTES: &'static [usize] = &QWEN36_ROUTES;
    const MAX_TOKENS: usize = QWEN36_MAX_TOKENS;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen36PagedGqaOp::new(context)
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
        // planes for the complete Qwen3.6 target geometry.
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

impl QualifiedPagedGqaOp for MtpBf16PagedGqaOp {
    type Target = Qwen38_27B;
    const ROUTES: &'static [usize] = &DECODE_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        MtpBf16PagedGqaOp::new(context)
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

/// Qualifies Qwen3.5 eager and captured BF16 paged GQA at exact `B=1..=8`
/// and `T=32,64,128`.
pub fn qualify_qwen35_paged_gqa() -> Result<PagedGqaQualification, PagedGqaQualificationError> {
    qualify_target::<Qwen35PagedGqaOp>()
}

/// Qualifies Qwen3.6 eager and captured BF16 paged GQA at exact `B=1..=8`
/// and `T=32,64,128`.
pub fn qualify_qwen36_paged_gqa() -> Result<PagedGqaQualification, PagedGqaQualificationError> {
    qualify_target::<Qwen36PagedGqaOp>()
}

/// Qualifies Qwen3.8 MTP eager and captured BF16 paged GQA at exact `B=1..=8`.
pub fn qualify_mtp_bf16_paged_gqa() -> Result<PagedGqaQualification, PagedGqaQualificationError> {
    qualify_target::<MtpBf16PagedGqaOp>()
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
    let (layout, regions) = layout::<O::Target>(O::MAX_TOKENS, O::CACHE_ELEMENT_BYTES)?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture::<O::Target>(O::MAX_TOKENS, O::CACHE_FORMAT);
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = O::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = PagedGqaQualification {
        output_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        cache_bytes: regions.cache_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for &tokens in O::ROUTES {
        reset_output(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, tokens)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_oracle::<O::Target>(tokens, O::CACHE_FORMAT, &fixture, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_output(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, tokens))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replay and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(tokens, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(PagedGqaQualificationError::Mismatch(format!(
                "device addresses changed while qualifying tokens={tokens}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout<A: Arch>(
    max_tokens: usize,
    cache_element_bytes: usize,
) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let query = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM * cache_element_bytes;
    let key_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let value_pages = layout.reserve(plane_bytes, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let table_rows = layout.reserve(max_tokens, ALIGNMENT)?;
    let lengths = layout.reserve(max_tokens, ALIGNMENT)?;
    let output = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;

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

fn fixture<A: Arch>(max_tokens: usize, cache_format: CacheFormat) -> Fixture {
    let query = (0..max_tokens * A::ATTENTION_OUTPUT_COLUMNS)
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
        table_rows: (0..max_tokens)
            .map(|token| {
                TABLE_ROW_IDS
                    .get(token)
                    .copied()
                    .unwrap_or((token % TABLE_ROWS) as u32)
            })
            .collect(),
        lengths: (0..max_tokens)
            .map(|token| LENGTHS.get(token).copied().unwrap_or((token + 1) as u32))
            .collect(),
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
    tokens: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
    observed: &[f32],
    report: &mut PagedGqaQualification,
) -> Result<(), PagedGqaQualificationError> {
    let expected = oracle::<A>(tokens, cache_format, fixture)?;
    let active = tokens * A::ATTENTION_OUTPUT_COLUMNS;
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
                "output at tokens={tokens}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    for (index, value) in observed[active..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(PagedGqaQualificationError::Mismatch(format!(
                "tokens={tokens} modified inactive output word {}",
                active + index
            )));
        }
    }
    report.output_values += active;
    report.untouched_values += observed.len() - active;

    Ok(())
}

fn oracle<A: Arch>(
    tokens: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
) -> Result<Vec<f32>, PagedGqaQualificationError> {
    let mut output = vec![f32::from_bits(F32_SENTINEL_BITS); fixture.query.len()];
    for token in 0..tokens {
        let row = fixture.table_rows[token] as usize;
        let length = fixture.lengths[token] as usize;
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
    check!(regions.table_rows, &fixture.table_rows, "table rows");
    check!(regions.lengths, &fixture.lengths, "lengths");

    Ok(())
}

fn verify_replay(
    tokens: usize,
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
            "tokens={tokens} graph output word {index} differs from eager"
        )));
    }
    report.graph_replay_values += replay.len();

    Ok(())
}

fn verify_no_post_warmup_allocation<O: QualifiedPagedGqaOp>(
    context: &CudaContext,
    op: &O,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), PagedGqaQualificationError> {
    let graphs = O::ROUTES
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
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: every allocation this graph captured is owned by this scope or
            // its caller and outlives the replays and the synchronize that follows.
            unsafe { graph.launch(stream) }?;
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
        MAX_BATCH, PHYSICAL_PAGES, QWEN35_MAX_TOKENS, QWEN35_ROUTES, QWEN36_MAX_TOKENS,
        QWEN36_ROUTES, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, TABLE_ROWS, TABLE_STRIDE, layout,
        qualify_mtp_bf16_paged_gqa, qualify_paged_gqa, qualify_qwen35_paged_gqa,
        qualify_qwen36_paged_gqa,
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
        assert_eq!(report.cache_bytes, 3_145_728);
        assert_eq!(report.workspace_bytes, 393_376);
        assert_eq!(report.immutable_input_values, 51_118_720);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout::<Qwen38_27B>(MAX_BATCH, size_of::<u8>())?;
        assert_eq!(arena.byte_len(), 3_539_712);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 608);

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen35_bf16_exact_decode_and_prompt_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaQualificationError> {
        let report = qualify_qwen35_paged_gqa()?;
        let active_tokens = QWEN35_ROUTES.iter().sum::<usize>();
        let output_per_token = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
        let plane_bytes = PHYSICAL_PAGES
            * Qwen35_9B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen35_9B::HEAD_DIM
            * size_of::<u16>();
        let immutable_per_check = QWEN35_MAX_TOKENS * output_per_token
            + 2 * plane_bytes
            + TABLE_ROWS * TABLE_STRIDE
            + 2 * QWEN35_MAX_TOKENS;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(report.untouched_values, 4_702_208);
        assert_eq!(report.graph_replay_values, 5_767_168);
        assert_eq!(
            report.immutable_input_values,
            2 * QWEN35_ROUTES.len() * immutable_per_check
        );
        assert_eq!(report.arena_bytes - report.padding_bytes, 10_486_880);
        assert_eq!(report.cache_bytes, 6_291_456);
        assert_eq!(report.workspace_bytes, 4_195_424);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout::<Qwen35_9B>(QWEN35_MAX_TOKENS, size_of::<u16>())?;
        assert_eq!(arena.byte_len(), 10_487_040);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 160);

        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen36_bf16_exact_decode_and_prompt_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaQualificationError> {
        let report = qualify_qwen36_paged_gqa()?;
        let active_tokens = QWEN36_ROUTES.iter().sum::<usize>();
        let output_per_token = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
        let plane_bytes = PHYSICAL_PAGES
            * Qwen36Moe35B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen36Moe35B::HEAD_DIM
            * size_of::<u16>();
        let immutable_per_check = QWEN36_MAX_TOKENS * output_per_token
            + 2 * plane_bytes
            + TABLE_ROWS * TABLE_STRIDE
            + 2 * QWEN36_MAX_TOKENS;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(report.untouched_values, 4_702_208);
        assert_eq!(report.graph_replay_values, 5_767_168);
        assert_eq!(
            report.immutable_input_values,
            2 * QWEN36_ROUTES.len() * immutable_per_check
        );
        assert_eq!(report.arena_bytes - report.padding_bytes, 7_341_152);
        assert!(report.maximum_absolute_error <= 0.003);
        let (arena, regions) = layout::<Qwen36Moe35B>(QWEN36_MAX_TOKENS, size_of::<u16>())?;
        assert_eq!(arena.byte_len(), 7_341_312);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 160);

        Ok(())
    }

    #[test]
    fn mtp_bf16_paged_gqa_suite_arena_accounting_exposes_every_byte() {
        let (arena, regions) = layout::<Qwen38_27B>(MAX_BATCH, size_of::<u16>()).unwrap();

        assert_eq!(regions.cache_bytes(), 6_291_456);
        assert_eq!(regions.workspace_bytes(), 393_376);
        assert_eq!(regions.payload_bytes(), 6_684_832);
        assert_eq!(arena.byte_len(), 6_685_440);
        assert_eq!(arena.byte_len() - regions.payload_bytes(), 608);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn mtp_bf16_paged_gqa_suite_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::PagedGqaQualificationError> {
        let report = qualify_mtp_bf16_paged_gqa()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let output_per_token = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
        let plane_bytes = PHYSICAL_PAGES
            * Qwen38_27B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen38_27B::HEAD_DIM
            * size_of::<u16>();
        let immutable_per_check = MAX_BATCH * output_per_token
            + 2 * plane_bytes
            + TABLE_ROWS * TABLE_STRIDE
            + 2 * MAX_BATCH;

        assert_eq!(report.output_values, active_tokens * output_per_token);
        assert_eq!(report.untouched_values, 172_032);
        assert_eq!(report.graph_replay_values, 393_216);
        assert_eq!(
            report.immutable_input_values,
            2 * MAX_BATCH * immutable_per_check
        );
        assert_eq!(report.cache_bytes, 6_291_456);
        assert_eq!(report.workspace_bytes, 393_376);
        assert_eq!(report.arena_bytes - report.padding_bytes, 6_684_832);
        assert_eq!(report.arena_bytes, 6_685_440);
        assert_eq!(report.padding_bytes, 608);
        assert!(report.maximum_absolute_error <= 0.003);

        Ok(())
    }
}

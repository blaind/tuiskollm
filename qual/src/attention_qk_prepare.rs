//! Numerical and graph qualification for full-attention Q/K preparation.

use crate::fp8_projection_oracle::{
    BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, encode_e4m3fn, f32_to_bf16,
};
use crate::target::MtpBf16QkPrepareOp;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::{mem::size_of, path::Path, sync::Arc};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, AttentionQkPrepareOp, Qwen35AttentionQkPrepareOp,
    Qwen36AttentionQkPrepareOp,
};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MtpBindings, Qwen35_9B, Qwen36Moe35B, Qwen38_27B,
};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 1_024;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const BF16_ROUTES: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];
const ALIGNMENT: usize = 256;
const PHYSICAL_PAGES: usize = 16;
const TABLE_ROWS: usize = 8;
const TABLE_STRIDE: usize = 16;
const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const KEY_SCALE: f32 = 0.03125;
const VALUE_SCALE: f32 = 0.0625;
const DECODE_TABLE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const DECODE_CACHE_POSITIONS: [u32; MAX_BATCH] = [63, 64, 1, 126, 2, 65, 127, 0];
const INPUT_PATTERN: [f32; 16] = [
    0.5, -0.375, 0.25, -0.1875, 0.125, -0.09375, 0.0625, -0.03125, -0.5, 0.375, -0.25, 0.1875,
    -0.125, 0.09375, -0.0625, 0.03125,
];
const NORM_PATTERN: [f32; 8] = [-0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25, -0.1875];

/// Failure of the exact attention Q/K preparation gate.
#[derive(Debug, thiserror::Error)]
pub enum AttentionQkPrepareQualificationError {
    /// Snapshot admission or MTP source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

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

/// Observable counts and worst errors across every exact decode and prefill route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionQkPrepareQualification {
    /// Active FP32 query values compared with the independent formula.
    pub query_values: usize,
    /// Appended represented key and value elements compared with independent formulas.
    pub appended_cache_values: usize,
    /// Cache bytes and inactive query words proved untouched.
    pub untouched_values: usize,
    /// Read-only input and metadata values proved unchanged.
    pub immutable_input_values: usize,
    /// Complete output/cache state reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Exact represented query/key norm bytes.
    pub weight_bytes: usize,
    /// Exact BF16 or E4M3 cache bytes owned by the fixture.
    pub cache_bytes: usize,
    /// Exact address-stable non-weight, non-cache bytes.
    pub workspace_bytes: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Largest absolute prepared-query or BF16-key error.
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
    decode_table_rows: ArenaRegion<u32>,
    decode_cache_positions: ArenaRegion<u32>,
    prefill_table_rows: ArenaRegion<u32>,
    prefill_cache_positions: ArenaRegion<u32>,
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
            + self.decode_table_rows.byte_len()
            + self.decode_cache_positions.byte_len()
            + self.prefill_table_rows.byte_len()
            + self.prefill_cache_positions.byte_len()
            + self.query.byte_len()
            + self.key_pages.byte_len()
            + self.value_pages.byte_len()
    }

    fn weight_bytes(self) -> usize {
        self.query_norm.byte_len() + self.key_norm.byte_len()
    }

    fn cache_bytes(self) -> usize {
        self.key_pages.byte_len() + self.value_pages.byte_len()
    }

    fn workspace_bytes(self) -> usize {
        self.payload_bytes() - self.weight_bytes() - self.cache_bytes()
    }
}

struct Fixture {
    qkv: Vec<u16>,
    query_norm: Vec<u16>,
    key_norm: Vec<u16>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    block_tables: Vec<u32>,
    prefill_table_rows: Vec<u32>,
    prefill_cache_positions: Vec<u32>,
}

struct SourceNorms {
    query: Vec<u16>,
    key: Vec<u16>,
}

struct Observed {
    query: Vec<f32>,
    key_pages: Vec<u8>,
    value_pages: Vec<u8>,
}

#[derive(Clone, Copy)]
enum CacheFormat {
    E4m3,
    Bf16Exact,
    Bf16Math,
}

trait QualifiedQkPrepareOp {
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
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()>;
}

macro_rules! impl_qualified_op {
    ($op:ty, $target:ty, $routes:expr, $max_tokens:expr) => {
        impl QualifiedQkPrepareOp for $op {
            type Target = $target;
            const ROUTES: &'static [usize] = $routes;
            const MAX_TOKENS: usize = $max_tokens;
            const CACHE_ELEMENT_BYTES: usize = size_of::<u8>();
            const CACHE_FORMAT: CacheFormat = CacheFormat::E4m3;

            fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
                <$op>::new(context)
            }

            #[allow(clippy::too_many_arguments)]
            fn launch(
                &self,
                stream: &tuisko_gpu::CudaStream,
                batch: usize,
                qkv: *const u16,
                query_norm: *const u16,
                key_norm: *const u16,
                rope_cos: *const f32,
                rope_sin: *const f32,
                block_tables: *const u32,
                table_rows: *const u32,
                table_stride: usize,
                cache_positions: *const u32,
                query: *mut f32,
                key_pages: *mut u8,
                value_pages: *mut u8,
                key_scale: f32,
                value_scale: f32,
            ) -> GpuResult<()> {
                // SAFETY: the qualification arena establishes the complete
                // pointer contract before dispatching through this safe seam.
                unsafe {
                    <$op>::launch(
                        self,
                        stream,
                        batch,
                        qkv,
                        query_norm,
                        key_norm,
                        rope_cos,
                        rope_sin,
                        block_tables,
                        table_rows,
                        table_stride,
                        cache_positions,
                        query,
                        key_pages,
                        value_pages,
                        key_scale,
                        value_scale,
                    )
                }
            }
        }
    };
}

impl_qualified_op!(AttentionQkPrepareOp, Qwen38_27B, &ROUTES, MAX_TOKENS);

impl QualifiedQkPrepareOp for Qwen35AttentionQkPrepareOp {
    type Target = Qwen35_9B;
    const ROUTES: &'static [usize] = &BF16_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16Exact;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen35AttentionQkPrepareOp::new(context)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        _key_scale: f32,
        _value_scale: f32,
    ) -> GpuResult<()> {
        // SAFETY: the qualification layout reserves aligned BF16 cache planes;
        // this byte-typed seam is shared only to keep the Qwen3.8 oracle path.
        unsafe {
            self.launch(
                stream,
                batch,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages.cast(),
                value_pages.cast(),
            )
        }
    }
}

impl QualifiedQkPrepareOp for Qwen36AttentionQkPrepareOp {
    type Target = Qwen36Moe35B;
    const ROUTES: &'static [usize] = &BF16_ROUTES;
    const MAX_TOKENS: usize = MAX_BATCH;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16Exact;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        Qwen36AttentionQkPrepareOp::new(context)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        _key_scale: f32,
        _value_scale: f32,
    ) -> GpuResult<()> {
        // SAFETY: the qualification layout reserves aligned BF16 cache planes
        // for the complete Qwen3.6 target geometry.
        unsafe {
            self.launch(
                stream,
                batch,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages.cast(),
                value_pages.cast(),
            )
        }
    }
}

impl QualifiedQkPrepareOp for MtpBf16QkPrepareOp {
    type Target = Qwen38_27B;
    const ROUTES: &'static [usize] = &ROUTES;
    const MAX_TOKENS: usize = MAX_TOKENS;
    const CACHE_ELEMENT_BYTES: usize = size_of::<u16>();
    const CACHE_FORMAT: CacheFormat = CacheFormat::Bf16Math;

    fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        MtpBf16QkPrepareOp::new(context)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &tuisko_gpu::CudaStream,
        batch: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        _key_scale: f32,
        _value_scale: f32,
    ) -> GpuResult<()> {
        // SAFETY: the qualification layout reserves aligned BF16 cache planes
        // and establishes the complete exact-batch pointer contract.
        unsafe {
            self.launch(
                stream,
                batch,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages.cast(),
                value_pages.cast(),
            )
        }
    }
}

/// Qualifies eager and captured Q/K preparation routes at exact `B=1..=8`
/// and `T=32,64,128,1024`.
pub fn qualify_attention_qk_prepare()
-> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError> {
    qualify_target::<AttentionQkPrepareOp>(None)
}

/// Qualifies Qwen3.5 eager and captured Q/K preparation at exact `B=1..=8`.
pub fn qualify_qwen35_attention_qk_prepare()
-> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError> {
    qualify_target::<Qwen35AttentionQkPrepareOp>(None)
}

/// Qualifies Qwen3.6 eager and captured Q/K preparation at exact `B=1..=8`.
pub fn qualify_qwen36_attention_qk_prepare()
-> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError> {
    qualify_target::<Qwen36AttentionQkPrepareOp>(None)
}

/// Qualifies source-backed MTP Q/K preparation at exact `B=1..=8` and prompt tiles.
pub fn qualify_mtp_bf16_qk_prepare(
    root: &Path,
) -> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError> {
    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(root)?;
    let bindings = MtpBindings::bind(&snapshot)?;
    let source = SourceNorms {
        query: bindings.query_norm.words().collect(),
        key: bindings.key_norm.words().collect(),
    };

    qualify_target::<MtpBf16QkPrepareOp>(Some(&source))
}

fn qualify_target<O>(
    source_norms: Option<&SourceNorms>,
) -> Result<AttentionQkPrepareQualification, AttentionQkPrepareQualificationError>
where
    O: QualifiedQkPrepareOp,
{
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
    let (layout, regions) = layout::<O::Target>(O::MAX_TOKENS, O::CACHE_ELEMENT_BYTES)?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture::<O::Target>(O::MAX_TOKENS, source_norms);
    load_fixture(&arena, &stream, regions, &fixture)?;
    let op = O::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = AttentionQkPrepareQualification {
        query_values: 0,
        appended_cache_values: 0,
        untouched_values: 0,
        immutable_input_values: 0,
        graph_replay_values: 0,
        weight_bytes: regions.weight_bytes(),
        cache_bytes: regions.cache_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_query_error: 0.0,
    };

    for &tokens in O::ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, tokens)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_oracle::<O::Target>(tokens, O::CACHE_FORMAT, &fixture, &eager, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, tokens))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(tokens, &eager, &replay, &mut report)?;
        verify_inputs(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "device addresses changed while qualifying tokens={tokens}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn verify_no_post_warmup_allocation<O: QualifiedQkPrepareOp>(
    context: &CudaContext,
    op: &O,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
) -> Result<(), AttentionQkPrepareQualificationError> {
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
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn layout<A: Arch>(
    max_tokens: usize,
    cache_element_bytes: usize,
) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(max_tokens * A::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let query_norm = layout.reserve(A::HEAD_DIM, ALIGNMENT)?;
    let key_norm = layout.reserve(A::HEAD_DIM, ALIGNMENT)?;
    let rope_cos = layout.reserve(max_tokens * ROTARY_PAIRS, ALIGNMENT)?;
    let rope_sin = layout.reserve(max_tokens * ROTARY_PAIRS, ALIGNMENT)?;
    let block_tables = layout.reserve(TABLE_ROWS * TABLE_STRIDE, ALIGNMENT)?;
    let decode_table_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let decode_cache_positions = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let prefill_table_rows = layout.reserve(max_tokens, ALIGNMENT)?;
    let prefill_cache_positions = layout.reserve(max_tokens, ALIGNMENT)?;
    let query = layout.reserve(max_tokens * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let plane_bytes =
        PHYSICAL_PAGES * A::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * A::HEAD_DIM * cache_element_bytes;
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
            decode_table_rows,
            decode_cache_positions,
            prefill_table_rows,
            prefill_cache_positions,
            query,
            key_pages,
            value_pages,
        },
    ))
}

fn fixture<A: Arch>(max_tokens: usize, source_norms: Option<&SourceNorms>) -> Fixture {
    let qkv = (0..max_tokens * A::ATTENTION_QKV_ROWS)
        .map(|index| {
            let token = index / A::ATTENTION_QKV_ROWS;
            let factor = 1.0 - (token & 7) as f32 / 16.0;
            f32_to_bf16(INPUT_PATTERN[(index + 3 * token) & 15] * factor)
        })
        .collect();
    let query_norm = source_norms.map_or_else(
        || {
            (0..A::HEAD_DIM)
                .map(|index| f32_to_bf16(NORM_PATTERN[(index + 3) & 7]))
                .collect()
        },
        |source| source.query.clone(),
    );
    let key_norm = source_norms.map_or_else(
        || {
            (0..A::HEAD_DIM)
                .map(|index| f32_to_bf16(NORM_PATTERN[(index + 5) & 7]))
                .collect()
        },
        |source| source.key.clone(),
    );
    let mut positions = [
        vec![0u32; max_tokens],
        vec![0u32; max_tokens],
        vec![0u32; max_tokens],
    ];
    for token in 0..max_tokens {
        positions[0][token] = token as u32;
        positions[1][token] = (3 * token + 17) as u32;
        positions[2][token] = (5 * token + 29) as u32;
    }
    let (rope_cos, rope_sin) = make_mrope_coefficients(&positions);

    Fixture {
        qkv,
        query_norm,
        key_norm,
        rope_cos,
        rope_sin,
        block_tables: (0..TABLE_ROWS)
            .flat_map(|row| {
                (0..TABLE_STRIDE).map(move |page| ((2 * row + page) % PHYSICAL_PAGES) as u32)
            })
            .collect(),
        prefill_table_rows: vec![0; max_tokens],
        prefill_cache_positions: (0..max_tokens as u32).collect(),
    }
}

fn make_mrope_coefficients(positions: &[Vec<u32>; 3]) -> (Vec<f32>, Vec<f32>) {
    let tokens = positions[0].len();
    let mut cosine = vec![0.0f32; tokens * ROTARY_PAIRS];
    let mut sine = vec![0.0f32; tokens * ROTARY_PAIRS];
    for token in 0..tokens {
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
    arena.copy_from_host(stream, regions.decode_table_rows, &DECODE_TABLE_ROWS)?;
    arena.copy_from_host(
        stream,
        regions.decode_cache_positions,
        &DECODE_CACHE_POSITIONS,
    )?;
    arena.copy_from_host(
        stream,
        regions.prefill_table_rows,
        &fixture.prefill_table_rows,
    )?;
    arena.copy_from_host(
        stream,
        regions.prefill_cache_positions,
        &fixture.prefill_cache_positions,
    )
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

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 13]> {
    Ok([
        arena.address(regions.qkv)?.addr(),
        arena.address(regions.query_norm)?.addr(),
        arena.address(regions.key_norm)?.addr(),
        arena.address(regions.rope_cos)?.addr(),
        arena.address(regions.rope_sin)?.addr(),
        arena.address(regions.block_tables)?.addr(),
        arena.address(regions.decode_table_rows)?.addr(),
        arena.address(regions.decode_cache_positions)?.addr(),
        arena.address(regions.prefill_table_rows)?.addr(),
        arena.address(regions.prefill_cache_positions)?.addr(),
        arena.address(regions.query)?.addr(),
        arena.address(regions.key_pages)?.addr(),
        arena.address(regions.value_pages)?.addr(),
    ])
}

fn launch(
    op: &impl QualifiedQkPrepareOp,
    arena: &DeviceArena,
    stream: &tuisko_gpu::CudaStream,
    regions: Regions,
    tokens: usize,
) -> GpuResult<()> {
    let (table_rows, cache_positions) = if tokens <= MAX_BATCH {
        (
            arena.address(regions.decode_table_rows)?,
            arena.address(regions.decode_cache_positions)?,
        )
    } else {
        (
            arena.address(regions.prefill_table_rows)?,
            arena.address(regions.prefill_cache_positions)?,
        )
    };
    op.launch(
        stream,
        tokens,
        arena.address(regions.qkv)?,
        arena.address(regions.query_norm)?,
        arena.address(regions.key_norm)?,
        arena.address(regions.rope_cos)?,
        arena.address(regions.rope_sin)?,
        arena.address(regions.block_tables)?,
        table_rows,
        TABLE_STRIDE,
        cache_positions,
        arena.address(regions.query)?,
        arena.address(regions.key_pages)?,
        arena.address(regions.value_pages)?,
        KEY_SCALE,
        VALUE_SCALE,
    )
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

fn verify_oracle<A: Arch>(
    tokens: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut AttentionQkPrepareQualification,
) -> Result<(), AttentionQkPrepareQualificationError> {
    let expected = oracle::<A>(tokens, cache_format, fixture)?;
    let active_query = tokens * A::ATTENTION_OUTPUT_COLUMNS;
    for (index, (&actual, &expected)) in observed.query[..active_query]
        .iter()
        .zip(&expected.query[..active_query])
        .enumerate()
    {
        let error = (actual - expected).abs();
        report.maximum_query_error = report.maximum_query_error.max(error);
        let tolerance = 0.002f32.max(expected.abs() * 0.003);
        if !actual.is_finite() || error > tolerance {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "query at tokens={tokens}, index={index}: device={actual}, oracle={expected}, tolerance={tolerance}"
            )));
        }
    }
    for (index, value) in observed.query[active_query..].iter().enumerate() {
        if value.to_bits() != F32_SENTINEL_BITS {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "tokens={tokens} modified inactive query word {}",
                active_query + index
            )));
        }
    }
    match cache_format {
        CacheFormat::E4m3 | CacheFormat::Bf16Exact => {
            compare_cache(tokens, "key", &observed.key_pages, &expected.key_pages)?;
        }
        CacheFormat::Bf16Math => compare_bf16_key_cache(
            tokens,
            &observed.key_pages,
            &expected.key_pages,
            &mut report.maximum_query_error,
        )?,
    }
    compare_cache(
        tokens,
        "value",
        &observed.value_pages,
        &expected.value_pages,
    )?;

    let appended = tokens * 2 * A::ATTENTION_KV_ROWS;
    report.query_values += active_query;
    report.appended_cache_values += appended;
    report.untouched_values +=
        observed.query.len() - active_query + observed.key_pages.len() + observed.value_pages.len()
            - appended * cache_element_bytes(cache_format);

    Ok(())
}

fn compare_bf16_key_cache(
    tokens: usize,
    actual: &[u8],
    expected: &[u8],
    maximum_error: &mut f32,
) -> Result<(), AttentionQkPrepareQualificationError> {
    for (index, (actual, expected)) in actual
        .as_chunks::<2>()
        .0
        .iter()
        .zip(expected.as_chunks::<2>().0)
        .enumerate()
    {
        let actual = u16::from_le_bytes(*actual);
        let expected = u16::from_le_bytes(*expected);
        if expected == u16::from_le_bytes([BYTE_SENTINEL; 2]) {
            if actual != expected {
                return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                    "key cache at tokens={tokens} modified inactive BF16 word {index}"
                )));
            }
            continue;
        }

        let actual_value = bf16_to_f32(actual);
        let expected_value = bf16_to_f32(expected);
        let error = (actual_value - expected_value).abs();
        *maximum_error = maximum_error.max(error);
        let tolerance = 0.002f32.max(expected_value.abs() * 0.003);
        if !actual_value.is_finite() || error > tolerance {
            return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
                "key cache at tokens={tokens}, BF16 word={index}: device={actual_value} ({actual:#06x}), oracle={expected_value} ({expected:#06x}), tolerance={tolerance}"
            )));
        }
    }
    Ok(())
}

fn compare_cache(
    tokens: usize,
    name: &str,
    actual: &[u8],
    expected: &[u8],
) -> Result<(), AttentionQkPrepareQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        let word = index & !1;
        let words = actual
            .get(word..word + 2)
            .zip(expected.get(word..word + 2))
            .map(|(actual, expected)| {
                (
                    u16::from_le_bytes([actual[0], actual[1]]),
                    u16::from_le_bytes([expected[0], expected[1]]),
                )
            });
        return Err(AttentionQkPrepareQualificationError::Mismatch(format!(
            "{name} cache at tokens={tokens}, byte={index}: device={:#04x}, oracle={:#04x}, BF16 words={words:?}",
            actual[index], expected[index],
        )));
    }
    Ok(())
}

fn oracle<A: Arch>(
    tokens: usize,
    cache_format: CacheFormat,
    fixture: &Fixture,
) -> Result<Observed, AttentionQkPrepareQualificationError> {
    let max_tokens = fixture.rope_cos.len() / ROTARY_PAIRS;
    let mut query =
        vec![f32::from_bits(F32_SENTINEL_BITS); max_tokens * A::ATTENTION_OUTPUT_COLUMNS];
    let plane_bytes = PHYSICAL_PAGES
        * A::NUM_KV_HEADS
        * ATTENTION_PAGE_SIZE
        * A::HEAD_DIM
        * cache_element_bytes(cache_format);
    let mut key_pages = vec![BYTE_SENTINEL; plane_bytes];
    let mut value_pages = vec![BYTE_SENTINEL; plane_bytes];

    let (table_rows, cache_positions): (&[u32], &[u32]) = if tokens <= MAX_BATCH {
        (&DECODE_TABLE_ROWS, &DECODE_CACHE_POSITIONS)
    } else {
        (
            &fixture.prefill_table_rows,
            &fixture.prefill_cache_positions,
        )
    };
    for token in 0..tokens {
        let token_base = token * A::ATTENTION_QKV_ROWS;
        let cosine = &fixture.rope_cos[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        let sine = &fixture.rope_sin[token * ROTARY_PAIRS..(token + 1) * ROTARY_PAIRS];
        for head in 0..A::NUM_ATTENTION_HEADS {
            let source = token_base + head * 2 * A::HEAD_DIM;
            let destination = (token * A::NUM_ATTENTION_HEADS + head) * A::HEAD_DIM;
            normalize_rotate_oracle::<A>(
                &fixture.qkv[source..source + A::HEAD_DIM],
                &fixture.query_norm,
                cosine,
                sine,
                &mut query[destination..destination + A::HEAD_DIM],
            );
        }

        let table_row = table_rows[token] as usize;
        let position = cache_positions[token] as usize;
        let physical_page = fixture.block_tables
            [table_row * TABLE_STRIDE + position / ATTENTION_PAGE_SIZE]
            as usize;
        let key_source = token_base + A::ATTENTION_QUERY_ROWS;
        let value_source = key_source + A::ATTENTION_KV_ROWS;
        for head in 0..A::NUM_KV_HEADS {
            let mut normalized = vec![0.0f32; A::HEAD_DIM];
            let source = key_source + head * A::HEAD_DIM;
            normalize_rotate_oracle::<A>(
                &fixture.qkv[source..source + A::HEAD_DIM],
                &fixture.key_norm,
                cosine,
                sine,
                &mut normalized,
            );
            for (dimension, &key_value) in normalized.iter().enumerate() {
                let destination = cache_offset::<A>(physical_page, head, position, dimension);
                match cache_format {
                    CacheFormat::E4m3 => {
                        key_pages[destination] = encode_e4m3fn(key_value / KEY_SCALE)
                            .map_err(AttentionQkPrepareQualificationError::Mismatch)?;
                        value_pages[destination] = encode_e4m3fn(
                            bf16_to_f32(fixture.qkv[value_source + head * A::HEAD_DIM + dimension])
                                / VALUE_SCALE,
                        )
                        .map_err(AttentionQkPrepareQualificationError::Mismatch)?;
                    }
                    CacheFormat::Bf16Exact | CacheFormat::Bf16Math => {
                        let byte = destination * size_of::<u16>();
                        key_pages[byte..byte + 2]
                            .copy_from_slice(&f32_to_bf16(key_value).to_le_bytes());
                        value_pages[byte..byte + 2].copy_from_slice(
                            &fixture.qkv[value_source + head * A::HEAD_DIM + dimension]
                                .to_le_bytes(),
                        );
                    }
                }
            }
        }
    }

    Ok(Observed {
        query,
        key_pages,
        value_pages,
    })
}

const fn cache_element_bytes(format: CacheFormat) -> usize {
    match format {
        CacheFormat::E4m3 => size_of::<u8>(),
        CacheFormat::Bf16Exact | CacheFormat::Bf16Math => size_of::<u16>(),
    }
}

fn normalize_rotate_oracle<A: Arch>(
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
    let inverse_rms = 1.0 / (sum / A::HEAD_DIM as f64 + f64::from(A::RMS_NORM_EPSILON)).sqrt();
    let normalized = source
        .iter()
        .zip(norm)
        .map(|(&value, &weight)| {
            f64::from(bf16_to_f32(value)) * inverse_rms * (1.0 + f64::from(bf16_to_f32(weight)))
        })
        .collect::<Vec<_>>();
    for dimension in 0..A::HEAD_DIM {
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

fn cache_offset<A: Arch>(
    physical_page: usize,
    head: usize,
    position: usize,
    dimension: usize,
) -> usize {
    A::HEAD_DIM
        * ((position & (ATTENTION_PAGE_SIZE - 1))
            + ATTENTION_PAGE_SIZE * (head + A::NUM_KV_HEADS * physical_page))
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
    check!(
        regions.decode_table_rows,
        &DECODE_TABLE_ROWS,
        "decode table rows"
    );
    check!(
        regions.decode_cache_positions,
        &DECODE_CACHE_POSITIONS,
        "decode cache positions"
    );
    check!(
        regions.prefill_table_rows,
        &fixture.prefill_table_rows,
        "prefill table rows"
    );
    check!(
        regions.prefill_cache_positions,
        &fixture.prefill_cache_positions,
        "prefill cache positions"
    );

    Ok(())
}

fn verify_replay(
    tokens: usize,
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
            "tokens={tokens} graph query word {index} differs from eager"
        )));
    }
    compare_cache(tokens, "graph key", &replay.key_pages, &eager.key_pages)?;
    compare_cache(
        tokens,
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
        MAX_BATCH, MAX_TOKENS, PHYSICAL_PAGES, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, ROUTES,
        TABLE_ROWS, TABLE_STRIDE, layout, qualify_attention_qk_prepare,
        qualify_mtp_bf16_qk_prepare, qualify_qwen35_attention_qk_prepare,
        qualify_qwen36_attention_qk_prepare,
    };
    use std::{mem::size_of, path::PathBuf};
    use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
    use tuisko_model::Arch;

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::AttentionQkPrepareQualificationError> {
        let report = qualify_attention_qk_prepare()?;
        let active_tokens = ROUTES.iter().sum::<usize>();
        let query_per_token = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
        let cache_per_token = 2 * Qwen38_27B::ATTENTION_KV_ROWS;
        let plane_bytes =
            PHYSICAL_PAGES * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
        let replay_per_route = MAX_TOKENS * query_per_token + 2 * plane_bytes;
        let total_observable = ROUTES.len() * replay_per_route;

        assert_eq!(report.query_values, active_tokens * query_per_token);
        assert_eq!(
            report.appended_cache_values,
            active_tokens * cache_per_token
        );
        assert_eq!(
            report.untouched_values,
            total_observable - active_tokens * (query_per_token + cache_per_token)
        );
        assert_eq!(report.graph_replay_values, total_observable);
        assert_eq!(TABLE_ROWS * TABLE_STRIDE, 128);
        let (layout, regions) = layout::<Qwen38_27B>(MAX_TOKENS, size_of::<u8>())?;
        assert_eq!(report.arena_bytes, layout.byte_len());
        assert_eq!(
            report.arena_bytes - report.padding_bytes,
            regions.payload_bytes()
        );
        assert_eq!(report.weight_bytes, regions.weight_bytes());
        assert_eq!(report.cache_bytes, regions.cache_bytes());
        assert_eq!(report.workspace_bytes, regions.workspace_bytes());
        assert_eq!(report.immutable_input_values, 2 * ROUTES.len() * 14_748_304);
        assert!(report.maximum_query_error <= 0.003);
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn qwen35_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::AttentionQkPrepareQualificationError> {
        let report = qualify_qwen35_attention_qk_prepare()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let query_per_token = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
        let cache_per_token = 2 * Qwen35_9B::ATTENTION_KV_ROWS;
        let plane_bytes = PHYSICAL_PAGES
            * Qwen35_9B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen35_9B::HEAD_DIM
            * size_of::<u16>();
        let replay_per_route = MAX_BATCH * query_per_token + 2 * plane_bytes;
        let immutable_per_check = MAX_BATCH * Qwen35_9B::ATTENTION_QKV_ROWS
            + 2 * Qwen35_9B::HEAD_DIM
            + 2 * MAX_BATCH * super::ROTARY_PAIRS
            + TABLE_ROWS * TABLE_STRIDE
            + 4 * MAX_BATCH;

        assert_eq!(report.query_values, active_tokens * query_per_token);
        assert_eq!(
            report.appended_cache_values,
            active_tokens * cache_per_token
        );
        assert_eq!(report.untouched_values, 33_521_664);
        assert_eq!(
            report.immutable_input_values,
            2 * MAX_BATCH * immutable_per_check
        );
        assert_eq!(report.graph_replay_values, MAX_BATCH * replay_per_route);
        let (layout, regions) = layout::<Qwen35_9B>(MAX_BATCH, size_of::<u16>())?;
        assert_eq!(report.arena_bytes, layout.byte_len());
        assert_eq!(report.arena_bytes - report.padding_bytes, 4_492_928);
        assert_eq!(regions.payload_bytes(), 4_492_928);
        assert_eq!(report.weight_bytes, regions.weight_bytes());
        assert_eq!(report.cache_bytes, regions.cache_bytes());
        assert_eq!(report.workspace_bytes, regions.workspace_bytes());
        assert!(report.maximum_query_error <= 0.003);
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA compute-capability 12.0 device"]
    fn qwen36_exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), super::AttentionQkPrepareQualificationError> {
        let report = qualify_qwen36_attention_qk_prepare()?;
        let active_tokens = (1..=MAX_BATCH).sum::<usize>();
        let query_per_token = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
        let cache_elements_per_token = 2 * Qwen36Moe35B::ATTENTION_KV_ROWS;
        let cache_bytes_per_token = cache_elements_per_token * size_of::<u16>();
        let plane_bytes = PHYSICAL_PAGES
            * Qwen36Moe35B::NUM_KV_HEADS
            * ATTENTION_PAGE_SIZE
            * Qwen36Moe35B::HEAD_DIM
            * size_of::<u16>();
        let replay_per_route = MAX_BATCH * query_per_token + 2 * plane_bytes;
        let total_observable = MAX_BATCH * replay_per_route;

        assert_eq!(report.query_values, active_tokens * query_per_token);
        assert_eq!(
            report.appended_cache_values,
            active_tokens * cache_elements_per_token
        );
        assert_eq!(
            report.untouched_values,
            total_observable - active_tokens * (query_per_token + cache_bytes_per_token)
        );
        assert_eq!(report.graph_replay_values, total_observable);
        let (layout, regions) = layout::<Qwen36Moe35B>(MAX_BATCH, size_of::<u16>())?;
        assert_eq!(report.arena_bytes, layout.byte_len());
        assert_eq!(report.arena_bytes - report.padding_bytes, 2_379_392);
        assert_eq!(regions.payload_bytes(), 2_379_392);
        assert!(report.maximum_query_error <= 0.003);
        Ok(())
    }

    #[test]
    fn mtp_bf16_qk_prepare_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout::<Qwen38_27B>(MAX_TOKENS, size_of::<u16>()).unwrap();

        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(regions.weight_bytes(), 1_024);
        assert_eq!(regions.cache_bytes(), 4_194_304);
        assert_eq!(regions.workspace_bytes(), 54_796_864);
        assert_eq!(regions.payload_bytes(), 58_992_192);
        assert_eq!(layout.byte_len(), 58_992_640);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 448);
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device and the pinned Qwen3.8 snapshot"]
    fn mtp_bf16_qk_prepare_suite_source_norms_match_every_route_and_graph_replay() {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT").expect("TUISKO_SNAPSHOT must name the snapshot"),
        );
        let report = qualify_mtp_bf16_qk_prepare(&root).expect("MTP BF16 Q/K qualification");

        assert_eq!(report.query_values, 7_888_896);
        assert_eq!(report.appended_cache_values, 2_629_632);
        assert_eq!(report.untouched_values, 112_680_960);
        assert_eq!(report.immutable_input_values, 353_959_296);
        assert_eq!(report.graph_replay_values, 125_829_120);
        assert_eq!(report.weight_bytes, 1_024);
        assert_eq!(report.cache_bytes, 4_194_304);
        assert_eq!(report.workspace_bytes, 54_796_864);
        assert_eq!(report.arena_bytes, 58_992_640);
        assert_eq!(report.padding_bytes, 448);
        assert!(report.maximum_query_error <= 0.003);
    }
}

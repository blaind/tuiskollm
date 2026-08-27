//! Source-BF16 gated attention-output projection for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_kernels_sm120_common::attention_output::attention_gate_bf16;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
const OUTPUT_ROWS: usize = Qwen38_27B::HIDDEN;
const OUTPUT_TILES: usize = OUTPUT_ROWS / 8;
const QWEN35_INPUT_COLUMNS: usize = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
const QWEN35_OUTPUT_ROWS: usize = Qwen35_9B::HIDDEN;
const QWEN35_OUTPUT_TILES: usize = QWEN35_OUTPUT_ROWS / 8;
const QWEN36_INPUT_COLUMNS: usize = Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS;
const QWEN36_OUTPUT_ROWS: usize = Qwen36Moe35B::HIDDEN;
const QWEN36_OUTPUT_TILES: usize = QWEN36_OUTPUT_ROWS / 8;
// Four warps publish 32 adjacent output rows per CTA. The 160-CTA grid is
// almost one CTA per target SM while every warp retains one exact MMA tile.
const WARPS: usize = 4;
const PROJECTION_THREADS: u32 = (WARPS * 32) as u32;
const PROJECTION_BLOCKS: u32 = (OUTPUT_TILES / WARPS) as u32;
const GATE_THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(input: *const u32, row: usize, column: usize) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete gated BF16 rows.
        unsafe { *input.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: source weights are the exact `[OUTPUT_ROWS, INPUT_COLUMNS]` BF16 matrix.
        unsafe { *weight.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn projection_body<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < INPUT_COLUMNS {
            // Native m16n8k16 BF16 MMA keeps the source matrix represented;
            // B<=8 occupies only lower rows and no padded output is published.
            let activation = unsafe {
                [
                    input_pair::<TOKENS>(input, group, column + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group + 8, column + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one output pair.
            unsafe {
                *output.add(group * (OUTPUT_ROWS / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Publishes the exact gated FP32 and represented-BF16 MTP seams.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_attention_gate<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 6,144-wide row. Its 256 threads make exactly 24
        // coalesced passes and preserve the query-head gate-column mapping.
        unsafe { attention_gate_bf16::<Qwen38_27B>(attention, qkv, activation) }
    }

    /// Projects exact gated MTP rows through the source-BF16 output matrix.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_attention_output<const TOKENS: usize>(
        activation: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers all 640 eight-row output tiles once.
        unsafe { projection_body::<TOKENS>(activation, weight, output) }
    }
}

#[cuda_module]
mod variant_kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<A: Arch, const TOKENS: usize>(
        input: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete gated BF16 rows.
        unsafe { *input.add(row * (A::ATTENTION_OUTPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair<A: Arch>(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: source weights are the exact target BF16 output matrix.
        unsafe { *weight.add(row * (A::ATTENTION_OUTPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn projection_body<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < A::ATTENTION_OUTPUT_COLUMNS {
            let activation = unsafe {
                [
                    input_pair::<A, TOKENS>(input, group, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, group + 8, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one output pair.
            unsafe {
                *output.add(group * (A::HIDDEN / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Publishes exact gated FP32/BF16 Qwen3.5 MTP seams.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_attention_gate<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 4,096-wide row, requiring exactly 16 coalesced
        // passes at 256 threads. The query-head gate mapping is unchanged.
        unsafe { attention_gate_bf16::<Qwen35_9B>(attention, qkv, activation) }
    }

    /// Projects exact gated Qwen3.5 MTP rows through the source-BF16 output matrix.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_attention_output<const TOKENS: usize>(
        activation: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Four warps publish 32 rows per CTA. Qwen3.5 therefore uses 128
        // CTAs for its 4,096 rows while preserving each MMA accumulation.
        unsafe { projection_body::<Qwen35_9B, TOKENS>(activation, weight, output) }
    }

    /// Publishes exact gated FP32/BF16 Qwen3.6 MTP seams.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_attention_gate<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 4,096-wide row and writes 16 columns per thread;
        // the gate mapping and represented BF16 rounding are identical to
        // the already-qualified target attention boundary.
        unsafe { attention_gate_bf16::<Qwen36Moe35B>(attention, qkv, activation) }
    }

    /// Projects exact gated Qwen3.6 MTP rows through the source-BF16 output matrix.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_attention_output<const TOKENS: usize>(
        activation: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Four warps publish 32 rows per CTA. Qwen3.6 uses 64 CTAs for its
        // 2,048 rows while preserving each output's MMA accumulation order.
        unsafe { projection_body::<Qwen36Moe35B, TOKENS>(activation, weight, output) }
    }
}

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared gate and projection entries for an exact batch.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entries the module does not emit.
pub trait MtpAttentionOutputRoute<A: Arch>: Sized + private::Sealed {
    /// Embedded module that owns this route's entries.
    ///
    /// Qwen3.8 compiles into the anchor module and the two variants into a
    /// second one, so the module type travels with the route.
    type Module;

    /// Prepares both entries of this route's exact batch.
    fn prepare(module: &Self::Module) -> GpuResult<Self>;

    /// Launches this route's attention gate and then its output projection.
    ///
    /// # Safety
    ///
    /// The pointers carry `MtpBf16AttentionOutputOp::launch`'s contract
    /// unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &Self::Module,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's attention-output routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 and Qwen3.6 here never widens the
/// artifact-level admission bound. Each table names only the entries its own
/// model emits, which keeps the compiled inventory fixed while the three
/// prepared owners share one wrapper.
pub trait MtpAttentionOutputEntries<A: Arch>: private::Sealed {
    /// Embedded module this table's entries live in.
    type Module;
    /// Prepared decode route for one exact batch.
    type Decode<const TOKENS: usize>: MtpAttentionOutputRoute<A, Module = Self::Module>;

    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not tile.
    fn require_geometry() -> GpuResult<()>;

    /// Loads this table's embedded module.
    fn load(context: &Arc<CudaContext>) -> GpuResult<Self::Module>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8 gate and projection entries for one exact batch.
pub struct PreparedRoute<const TOKENS: usize> {
    gate: PreparedLaunch<kernels::__mtp_bf16_attention_gate_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__mtp_bf16_attention_output_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 gate and projection entries for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    gate: PreparedLaunch<variant_kernels::__qwen35_mtp_bf16_attention_gate_CudaKernel<TOKENS>>,
    projection:
        PreparedLaunch<variant_kernels::__qwen35_mtp_bf16_attention_output_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 gate and projection entries for one exact batch.
pub struct PreparedQwen36Route<const TOKENS: usize> {
    gate: PreparedLaunch<variant_kernels::__qwen36_mtp_bf16_attention_gate_CudaKernel<TOKENS>>,
    projection:
        PreparedLaunch<variant_kernels::__qwen36_mtp_bf16_attention_output_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> private::Sealed for PreparedRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36Route<TOKENS> {}

impl<const TOKENS: usize> MtpAttentionOutputRoute<Qwen35_9B> for PreparedQwen35Route<TOKENS> {
    type Module = variant_kernels::LoadedModule;

    fn prepare(module: &Self::Module) -> GpuResult<Self> {
        let gate = module
            .prepare_qwen35_mtp_bf16_attention_gate::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                GATE_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the Qwen3.5 MTP BF16 attention gate", source)
            })?;
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 attention-output grid exceeds u32")
        })?;
        let projection = module
            .prepare_qwen35_mtp_bf16_attention_output::<TOKENS>(LaunchConfig1D::new(
                blocks,
                PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch(
                    "preparing the Qwen3.5 MTP BF16 attention-output projection",
                    source,
                )
            })?;
        Ok(Self { gate, projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &Self::Module,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_mtp_bf16_attention_gate::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 attention gate", source)
            })?;
        module
            .qwen35_mtp_bf16_attention_output::<TOKENS>(
                stream,
                &self.projection,
                activation.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.5 MTP BF16 attention-output projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> MtpAttentionOutputRoute<Qwen36Moe35B> for PreparedQwen36Route<TOKENS> {
    type Module = variant_kernels::LoadedModule;

    fn prepare(module: &Self::Module) -> GpuResult<Self> {
        let gate = module
            .prepare_qwen36_mtp_bf16_attention_gate::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                GATE_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the Qwen3.6 MTP BF16 attention gate", source)
            })?;
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 MTP BF16 attention-output grid exceeds u32")
        })?;
        let projection = module
            .prepare_qwen36_mtp_bf16_attention_output::<TOKENS>(LaunchConfig1D::new(
                blocks,
                PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch(
                    "preparing the Qwen3.6 MTP BF16 attention-output projection",
                    source,
                )
            })?;
        Ok(Self { gate, projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &Self::Module,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_mtp_bf16_attention_gate::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.6 MTP BF16 attention gate", source)
            })?;
        module
            .qwen36_mtp_bf16_attention_output::<TOKENS>(
                stream,
                &self.projection,
                activation.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.6 MTP BF16 attention-output projection",
                    source,
                )
            })
    }
}

// The Qwen3.8 entries compile that model's widths into concrete symbols, so
// this route stays bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> MtpAttentionOutputRoute<A> for PreparedRoute<TOKENS> {
    type Module = kernels::LoadedModule;

    fn prepare(module: &Self::Module) -> GpuResult<Self> {
        let gate = module
            .prepare_mtp_bf16_attention_gate::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                GATE_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 attention gate", source))?;
        let projection = module
            .prepare_mtp_bf16_attention_output::<TOKENS>(LaunchConfig1D::new(
                PROJECTION_BLOCKS,
                PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the MTP BF16 attention-output projection", source)
            })?;

        Ok(Self { gate, projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &Self::Module,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_attention_gate::<TOKENS>(stream, &self.gate, attention, qkv, activation)
            .map_err(|source| GpuError::launch("launching the MTP BF16 attention gate", source))?;
        module
            .mtp_bf16_attention_output::<TOKENS>(
                stream,
                &self.projection,
                activation.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the MTP BF16 attention-output projection", source)
            })
    }
}

/// Qwen3.8 entry table: the 160-CTA anchor-module gate and projection entries.
pub struct Qwen38MtpAttentionOutputEntries;

/// Qwen3.5 entry table: the 128-CTA variant-module gate and projection entries.
pub struct Qwen35MtpAttentionOutputEntries;

/// Qwen3.6 entry table: the 64-CTA variant-module gate and projection entries.
pub struct Qwen36MtpAttentionOutputEntries;

impl private::Sealed for Qwen38MtpAttentionOutputEntries {}
impl private::Sealed for Qwen35MtpAttentionOutputEntries {}
impl private::Sealed for Qwen36MtpAttentionOutputEntries {}

impl<A: Sm120Arch> MtpAttentionOutputEntries<A> for Qwen38MtpAttentionOutputEntries {
    type Module = kernels::LoadedModule;
    type Decode<const TOKENS: usize> = PreparedRoute<TOKENS>;

    const LABEL: &'static str = "";

    fn require_geometry() -> GpuResult<()> {
        if INPUT_COLUMNS != 6_144
            || OUTPUT_ROWS != 5_120
            || !INPUT_COLUMNS.is_multiple_of(16)
            || !OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP attention-output geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn load(context: &Arc<CudaContext>) -> GpuResult<Self::Module> {
        // SAFETY: this crate owns the embedded exact MTP artifact.
        unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading MTP BF16 attention output", source))
    }

    fn ptx_names() -> Vec<&'static str> {
        mtp_bf16_attention_output_ptx_names()
    }
}

impl MtpAttentionOutputEntries<Qwen35_9B> for Qwen35MtpAttentionOutputEntries {
    type Module = variant_kernels::LoadedModule;
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;

    const LABEL: &'static str = "Qwen3.5 ";

    fn require_geometry() -> GpuResult<()> {
        if QWEN35_INPUT_COLUMNS != 4_096
            || QWEN35_OUTPUT_ROWS != 4_096
            || !QWEN35_INPUT_COLUMNS.is_multiple_of(16)
            || !QWEN35_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP attention-output geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn load(context: &Arc<CudaContext>) -> GpuResult<Self::Module> {
        // SAFETY: this crate owns the embedded exact MTP artifact.
        unsafe { variant_kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.5 MTP BF16 attention output", source))
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_mtp_bf16_attention_output_ptx_names()
    }
}

impl MtpAttentionOutputEntries<Qwen36Moe35B> for Qwen36MtpAttentionOutputEntries {
    type Module = variant_kernels::LoadedModule;
    type Decode<const TOKENS: usize> = PreparedQwen36Route<TOKENS>;

    const LABEL: &'static str = "Qwen3.6 ";

    fn require_geometry() -> GpuResult<()> {
        if QWEN36_INPUT_COLUMNS != 4_096
            || QWEN36_OUTPUT_ROWS != 2_048
            || !QWEN36_INPUT_COLUMNS.is_multiple_of(16)
            || !QWEN36_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 MTP attention-output geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn load(context: &Arc<CudaContext>) -> GpuResult<Self::Module> {
        // SAFETY: this crate owns the embedded exact MTP artifact.
        unsafe { variant_kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 MTP BF16 attention output", source))
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_mtp_bf16_attention_output_ptx_names()
    }
}

/// Stable PTX inventory for every exact MTP attention-output stage.
pub(crate) fn mtp_bf16_attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::mtp_bf16_attention_gate_ptx_name::<1>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<2>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<3>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<4>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<5>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<6>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<7>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<8>(),
        kernels::mtp_bf16_attention_output_ptx_name::<1>(),
        kernels::mtp_bf16_attention_output_ptx_name::<2>(),
        kernels::mtp_bf16_attention_output_ptx_name::<3>(),
        kernels::mtp_bf16_attention_output_ptx_name::<4>(),
        kernels::mtp_bf16_attention_output_ptx_name::<5>(),
        kernels::mtp_bf16_attention_output_ptx_name::<6>(),
        kernels::mtp_bf16_attention_output_ptx_name::<7>(),
        kernels::mtp_bf16_attention_output_ptx_name::<8>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.5 MTP attention-output stage.
pub(crate) fn qwen35_mtp_bf16_attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<1>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<2>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<3>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<4>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<5>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<6>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<7>(),
        variant_kernels::qwen35_mtp_bf16_attention_gate_ptx_name::<8>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<1>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<2>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<3>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<4>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<5>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<6>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<7>(),
        variant_kernels::qwen35_mtp_bf16_attention_output_ptx_name::<8>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.6 MTP attention-output stage.
pub(crate) fn qwen36_mtp_bf16_attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<1>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<2>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<3>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<4>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<5>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<6>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<7>(),
        variant_kernels::qwen36_mtp_bf16_attention_gate_ptx_name::<8>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<1>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<2>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<3>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<4>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<5>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<6>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<7>(),
        variant_kernels::qwen36_mtp_bf16_attention_output_ptx_name::<8>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(E::Module),
    error(GpuError),
    dispatch(dispatch_mtp_bf16_attention_output),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct MtpBf16AttentionOutputRoutes<A: Arch, E: MtpAttentionOutputEntries<A>> {
    #[route(1)]
    b1: E::Decode<1>,
    #[route(2)]
    b2: E::Decode<2>,
    #[route(3)]
    b3: E::Decode<3>,
    #[route(4)]
    b4: E::Decode<4>,
    #[route(5)]
    b5: E::Decode<5>,
    #[route(6)]
    b6: E::Decode<6>,
    #[route(7)]
    b7: E::Decode<7>,
    #[route(8)]
    b8: E::Decode<8>,
}

/// Prepared gated source-BF16 attention-output routes for exact MTP `B=1..=8`.
pub struct MtpBf16AttentionOutputOp<
    A: Arch = Qwen38_27B,
    E: MtpAttentionOutputEntries<A> = Qwen38MtpAttentionOutputEntries,
> {
    module: E::Module,
    routes: MtpBf16AttentionOutputRoutes<A, E>,
}

/// Prepared gated attention-output routes for exact Qwen3.5 MTP batches.
pub type Qwen35MtpBf16AttentionOutputOp =
    MtpBf16AttentionOutputOp<Qwen35_9B, Qwen35MtpAttentionOutputEntries>;

/// Prepared gated attention-output routes for exact Qwen3.6 MTP batches.
pub type Qwen36MtpBf16AttentionOutputOp =
    MtpBf16AttentionOutputOp<Qwen36Moe35B, Qwen36MtpAttentionOutputEntries>;

impl<A: Arch, E: MtpAttentionOutputEntries<A>> MtpBf16AttentionOutputOp<A, E> {
    /// Loads the embedded module and prepares every exact MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        let module = E::load(context)?;

        Ok(Self {
            routes: MtpBf16AttentionOutputRoutes::prepare(&module)?,
            module,
        })
    }

    /// Gates attention into FP32/BF16 seams and applies the source-BF16 projection.
    ///
    /// # Safety
    ///
    /// `attention` covers `[batch, A::ATTENTION_OUTPUT_COLUMNS]` FP32 values
    /// and is mutable scratch; `qkv` covers `[batch, A::ATTENTION_QKV_ROWS]`
    /// BF16 values; `activation` covers `[batch, A::ATTENTION_OUTPUT_COLUMNS]`
    /// BF16 values; `weight` covers `[A::HIDDEN, A::ATTENTION_OUTPUT_COLUMNS]`
    /// unchanged source BF16 values; and `output` covers `[batch, A::HIDDEN]`
    /// BF16 values. Four-byte-loaded regions are aligned, non-overlapping,
    /// context-local, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        attention,
                        qkv,
                        activation,
                        weight,
                        output,
                    )
                }
            };
        }

        dispatch_mtp_bf16_attention_output!(&self.routes, batch, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "{}MTP BF16 attention-output batch {batch} is outside exact B=1..={MAX_BATCH}",
                E::LABEL
            ))) )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GATE_THREADS, INPUT_COLUMNS, MAX_BATCH, MtpAttentionOutputEntries, OUTPUT_ROWS,
        OUTPUT_TILES, PROJECTION_BLOCKS, PROJECTION_THREADS, QWEN35_INPUT_COLUMNS,
        QWEN35_OUTPUT_ROWS, QWEN35_OUTPUT_TILES, QWEN36_INPUT_COLUMNS, QWEN36_OUTPUT_ROWS,
        QWEN36_OUTPUT_TILES, Qwen35MtpAttentionOutputEntries, Qwen36MtpAttentionOutputEntries,
        Qwen38MtpAttentionOutputEntries, WARPS, mtp_bf16_attention_output_ptx_names,
        qwen35_mtp_bf16_attention_output_ptx_names, qwen36_mtp_bf16_attention_output_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

    #[test]
    fn exact_geometry_and_batch_inventory_are_complete() {
        assert_eq!(INPUT_COLUMNS, 6_144);
        assert_eq!(OUTPUT_ROWS, 5_120);
        assert_eq!(OUTPUT_TILES, 640);
        assert_eq!(WARPS, 4);
        assert_eq!(PROJECTION_THREADS, 128);
        assert_eq!(PROJECTION_BLOCKS, 160);
        assert_eq!(GATE_THREADS, 256);

        let names = mtp_bf16_attention_output_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }

    #[test]
    fn qwen35_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN35_INPUT_COLUMNS, 4_096);
        assert_eq!(QWEN35_OUTPUT_ROWS, 4_096);
        assert_eq!(QWEN35_OUTPUT_TILES, 512);
        assert_eq!(QWEN35_OUTPUT_TILES / WARPS, 128);
        let names = qwen35_mtp_bf16_attention_output_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 16);
    }

    #[test]
    fn qwen36_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN36_INPUT_COLUMNS, 4_096);
        assert_eq!(QWEN36_OUTPUT_ROWS, 2_048);
        assert_eq!(QWEN36_OUTPUT_TILES, 256);
        assert_eq!(QWEN36_OUTPUT_TILES / WARPS, 64);
        let names = qwen36_mtp_bf16_attention_output_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 16);
    }

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen38_27B>>::ptx_names(),
            mtp_bf16_attention_output_ptx_names()
        );
        assert_eq!(
            <Qwen35MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen35_9B>>::ptx_names(),
            qwen35_mtp_bf16_attention_output_ptx_names()
        );
        assert_eq!(
            <Qwen36MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen36Moe35B>>::ptx_names(
            ),
            qwen36_mtp_bf16_attention_output_ptx_names()
        );
    }

    /// The three tables keep their owners' geometry rejections separate.
    #[test]
    fn every_entry_table_admits_its_own_geometry() {
        assert!(
            <Qwen38MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen38_27B>>::require_geometry()
                .is_ok()
        );
        assert!(
            <Qwen35MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen35_9B>>::require_geometry()
                .is_ok()
        );
        assert!(
            <Qwen36MtpAttentionOutputEntries as MtpAttentionOutputEntries<Qwen36Moe35B>>::require_geometry()
                .is_ok()
        );
    }
}

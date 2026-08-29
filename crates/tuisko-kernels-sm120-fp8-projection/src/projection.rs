//! Source-native FP8 projections with dynamic E4M3 activation quantization.

use crate::device::fp8_projection::{
    fp8_projection, prefill_projection_mma, qkv_projection_mma_t16,
};
use crate::gdn_input_tma::{
    DenseFp8GdnInputTmaMaps, DenseFp8GdnInputTmaRoute, ptx_name as gdn_input_tma_ptx_name,
};
use crate::qkv_tma::{DenseFp8QkvTmaMaps, DenseFp8QkvTmaRoute, ptx_name as qkv_tma_ptx_name};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::{marker::PhantomData, sync::Arc};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_kernels_sm120_common::device::quantize_activation;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
// Eight warps quantize one 5,120-wide row: each thread owns exactly ten BF16 pairs.
const QUANTIZE_WARPS: usize = 8;
const QUANTIZE_THREADS: u32 = (QUANTIZE_WARPS * 32) as u32;
// Retained paired-row schedule: eight warps produce 16 rows per CTA; B=1 reached
// 90.65% measured DRAM throughput without changing the per-row accumulation order.
const DECODE_PROJECTION_WARPS: usize = 8;
const DECODE_PROJECTION_THREADS: u32 = (DECODE_PROJECTION_WARPS * 32) as u32;
// T=16 uses two warps for one 16x64 tile. T=32/64/128 use a 64-row CTA so eight
// warps share each 64-row weight tile; the retained schedule measured better than
// narrower token CTAs while preserving each m16n8k32 accumulation sequence.
const QKV_MMA_T16_TOKENS: usize = 16;
const QKV_MMA_PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const QKV_MMA_MACRO_TOKENS: usize = 1_024;
const QKV_MMA_OUTPUT_ROWS: usize = 64;
const QKV_MMA_T16_BLOCK_ROWS: usize = 16;
const QKV_MMA_PREFILL_BLOCK_ROWS: usize = 64;
const QKV_MMA_K_WORDS: usize = 32;
// The 1,024-row route halves the staged K words. It keeps the same ordered sequence
// of m16n8k32 operations while reducing dynamic shared memory from 32 KiB to 16 KiB.
const QKV_MMA_MACRO_K_WORDS: usize = 16;
const QKV_MMA_T16_THREADS: u32 = 64;
const QKV_MMA_PREFILL_THREADS: u32 = 256;
const QKV_MMA_T16_SHARED_BYTES: u32 =
    (2 * (QKV_MMA_T16_BLOCK_ROWS + QKV_MMA_OUTPUT_ROWS) * QKV_MMA_K_WORDS * size_of::<u32>())
        as u32;
const QKV_MMA_PREFILL_SHARED_BYTES: u32 =
    (2 * (QKV_MMA_PREFILL_BLOCK_ROWS + QKV_MMA_OUTPUT_ROWS) * QKV_MMA_K_WORDS * size_of::<u32>())
        as u32;
const QKV_MMA_MACRO_SHARED_BYTES: u32 = (2
    * (QKV_MMA_PREFILL_BLOCK_ROWS + QKV_MMA_OUTPUT_ROWS)
    * QKV_MMA_MACRO_K_WORDS
    * size_of::<u32>()) as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fp8Geometry {
    quantize_pairs_per_thread: usize,
    qkv_decode_blocks: usize,
    gdn_decode_blocks: usize,
    lm_head_decode_blocks: usize,
    qkv_mma_output_tiles: usize,
    qkv_mma_k_tiles: usize,
}

fn fp8_geometry<A: Arch>() -> Option<Fp8Geometry> {
    let decode_rows = 2 * DECODE_PROJECTION_WARPS;
    // The decode kernels instantiate `fp8_projection::<5_120, ..>`; only the
    // exact width is admissible regardless of divisibility.
    if A::HIDDEN != 5_120
        || !A::ATTENTION_QKV_ROWS.is_multiple_of(decode_rows)
        || !A::GDN_INPUT_ROWS.is_multiple_of(decode_rows)
        || !A::VOCAB.is_multiple_of(decode_rows)
        || !A::ATTENTION_QKV_ROWS.is_multiple_of(QKV_MMA_OUTPUT_ROWS)
        || !(A::HIDDEN / 4).is_multiple_of(QKV_MMA_K_WORDS)
    {
        return None;
    }

    Some(Fp8Geometry {
        quantize_pairs_per_thread: (A::HIDDEN / 2).div_ceil(QUANTIZE_THREADS as usize),
        qkv_decode_blocks: A::ATTENTION_QKV_ROWS / decode_rows,
        gdn_decode_blocks: A::GDN_INPUT_ROWS / decode_rows,
        lm_head_decode_blocks: A::VOCAB / decode_rows,
        qkv_mma_output_tiles: A::ATTENTION_QKV_ROWS / QKV_MMA_OUTPUT_ROWS,
        qkv_mma_k_tiles: A::HIDDEN / 4 / QKV_MMA_K_WORDS,
    })
}

fn require_fp8_geometry<A: Arch>() -> GpuResult<Fp8Geometry> {
    fp8_geometry::<A>().ok_or_else(|| {
        GpuError::invalid_launch("architecture geometry is incompatible with the FP8 schedules")
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Dynamically quantizes one BF16 row per block to E4M3 plus an FP32 scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn quantize_activation_e4m3(input: *const u32, codes: *mut u16, scales: *mut f32) {
        static mut WARP_MAXIMUM: SharedArray<f32, QUANTIZE_WARPS, 16> = SharedArray::UNINIT;
        let warp_maximum = core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>();

        // SAFETY: the exact launch assigns one complete Qwen row to each block.
        unsafe {
            quantize_activation::<Qwen38_27B>(input, codes, scales, warp_maximum);
        }
    }

    /// Projects dynamically quantized rows through the fused source-native QKV plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_qkv<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // SAFETY: the prepared exact-B grid covers every QKV row pair once.
        unsafe {
            fp8_projection::<5_120, TOKENS, DECODE_PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                A::ATTENTION_QKV_ROWS,
            );
        }
    }

    /// Projects dynamically quantized rows through the fused GDN Q/K/V/Z plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_gdn_input<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // SAFETY: the prepared exact-B grid covers every GDN input row pair once.
        unsafe {
            fp8_projection::<5_120, TOKENS, DECODE_PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                A::GDN_INPUT_ROWS,
            );
        }
    }

    /// Projects dynamically quantized rows through the full-vocabulary LM head.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_lm_head<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // SAFETY: the prepared exact-B grid covers every vocabulary row pair once.
        unsafe {
            fp8_projection::<5_120, TOKENS, DECODE_PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                A::VOCAB,
            );
        }
    }

    /// Projects exactly 16 dynamically quantized rows with the retained MMA tile.
    #[kernel]
    #[launch_bounds(64, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (64, 1, 1),
        dynamic_shared = 20480,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_qkv_mma_t16(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // SAFETY: the fixed launch covers the exact 16-row QKV plane.
        unsafe {
            qkv_projection_mma_t16::<Qwen38_27B>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
            );
        }
    }

    /// Projects one exact 32/64/128-row prefill tile through source-native QKV.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 32768,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_qkv_mma<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // SAFETY: the prepared route instantiates only T=32/64/128 with a complete
        // padded 64-row activation tile and exact 64-row output tiles.
        unsafe {
            prefill_projection_mma::<{ Qwen38_27B::HIDDEN }, TOKENS, 64, 32, 4>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
                Qwen38_27B::ATTENTION_QKV_ROWS,
            );
        }
    }

    /// Projects one exact 32/64/128-row prefill tile through source-native GDN Q/K/V/Z.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 32768,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_gdn_input_mma<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // SAFETY: the exact 64x64 CTA tile preserves every m16n8k32 accumulation
        // and exposes 256 output CTAs per token tile for the 16,384-row GDN plane.
        unsafe {
            prefill_projection_mma::<{ Qwen38_27B::HIDDEN }, TOKENS, 64, 32, 4>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
                Qwen38_27B::GDN_INPUT_ROWS,
            );
        }
    }
}

fn prepare_quantize<const TOKENS: usize>(
    module: &kernels::LoadedModule,
) -> GpuResult<PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>> {
    let blocks = u32::try_from(TOKENS).map_err(|_| {
        GpuError::invalid_launch("FP8 projection token count exceeds CUDA grid width")
    })?;
    module
        .prepare_quantize_activation_e4m3(LaunchConfig1D::new(blocks, QUANTIZE_THREADS, 0))
        .map_err(|source| GpuError::launch("preparing FP8 activation quantization", source))
}

struct PreparedQkvRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_qkv_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedQkvRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let row_pairs_per_block = 2 * DECODE_PROJECTION_WARPS;
        let projection_blocks = A::ATTENTION_QKV_ROWS / row_pairs_per_block;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("FP8 QKV rows exceed CUDA grid width"))?;
        let quantize = prepare_quantize::<TOKENS>(module)?;
        let projection = module
            .prepare_fp8_qkv::<A, TOKENS>(LaunchConfig1D::new(
                projection_blocks,
                DECODE_PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the FP8 QKV projection", source))?;

        Ok(Self {
            quantize,
            projection,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        module
            .fp8_qkv::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching the FP8 QKV projection", source))
    }
}

struct PreparedGdnInputRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_gdn_input_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedGdnInputRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let row_pairs_per_block = 2 * DECODE_PROJECTION_WARPS;
        let projection_blocks = A::GDN_INPUT_ROWS / row_pairs_per_block;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("GDN input rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_gdn_input::<A, TOKENS>(LaunchConfig1D::new(
                projection_blocks,
                DECODE_PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the FP8 GDN input projection", source))?;

        Ok(Self {
            quantize: prepare_quantize::<TOKENS>(module)?,
            projection,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        module
            .fp8_gdn_input::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching the FP8 GDN input projection", source))
    }
}

struct PreparedLmHeadRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_lm_head_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedLmHeadRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let row_pairs_per_block = 2 * DECODE_PROJECTION_WARPS;
        let projection_blocks = A::VOCAB / row_pairs_per_block;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("LM-head rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_lm_head::<A, TOKENS>(LaunchConfig1D::new(
                projection_blocks,
                DECODE_PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the FP8 LM-head projection", source))?;

        Ok(Self {
            quantize: prepare_quantize::<TOKENS>(module)?,
            projection,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        module
            .fp8_lm_head::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching the FP8 LM-head projection", source))
    }
}

struct PreparedQkvT16Route<A: Arch> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_qkv_mma_t16_CudaKernel>,
    architecture: PhantomData<A>,
}

impl<A: Arch> PreparedQkvT16Route<A> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection_blocks = A::ATTENTION_QKV_ROWS / QKV_MMA_OUTPUT_ROWS;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("FP8 QKV rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_qkv_mma_t16(LaunchConfig1D::new(
                projection_blocks,
                QKV_MMA_T16_THREADS,
                QKV_MMA_T16_SHARED_BYTES,
            ))
            .map_err(|source| GpuError::launch("preparing the FP8 QKV T=16 projection", source))?;

        Ok(Self {
            quantize: prepare_quantize::<QKV_MMA_T16_TOKENS>(module)?,
            projection,
            architecture: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        let k_tiles = A::HIDDEN / 4 / QKV_MMA_K_WORDS;
        module
            .fp8_qkv_mma_t16(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                k_tiles as u32,
            )
            .map_err(|source| GpuError::launch("launching the FP8 QKV T=16 projection", source))
    }
}

struct PreparedQkvPrefillRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_qkv_mma_CudaKernel<TOKENS>>,
    architecture: PhantomData<A>,
}

impl<A: Arch, const TOKENS: usize> PreparedQkvPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QKV_MMA_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "FP8 QKV prefill route T={TOKENS} is not admitted"
            )));
        }
        let token_tiles = TOKENS.div_ceil(QKV_MMA_PREFILL_BLOCK_ROWS);
        let projection_blocks = A::ATTENTION_QKV_ROWS / QKV_MMA_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("FP8 QKV prefill grid exceeds CUDA width"))?;
        let projection = module
            .prepare_fp8_qkv_mma::<TOKENS>(LaunchConfig1D::new(
                projection_blocks,
                QKV_MMA_PREFILL_THREADS,
                QKV_MMA_PREFILL_SHARED_BYTES,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the FP8 QKV prefill projection", source)
            })?;

        Ok(Self {
            quantize: prepare_quantize::<TOKENS>(module)?,
            projection,
            architecture: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        let k_tiles = A::HIDDEN / 4 / QKV_MMA_K_WORDS;
        module
            .fp8_qkv_mma::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                k_tiles as u32,
            )
            .map_err(|source| GpuError::launch("launching the FP8 QKV prefill projection", source))
    }
}

struct PreparedQkvT1024Route {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: DenseFp8QkvTmaRoute,
}

impl PreparedQkvT1024Route {
    fn prepare(context: &Arc<CudaContext>, module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            quantize: prepare_quantize::<QKV_MMA_MACRO_TOKENS>(module)?,
            projection: DenseFp8QkvTmaRoute::new(context)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8QkvTmaMaps,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.projection
                .launch(stream, maps, activation_scales, weight_scales, output)
        }
    }
}

struct PreparedGdnInputPrefillRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_gdn_input_mma_CudaKernel<TOKENS>>,
    architecture: PhantomData<A>,
}

impl<A: Arch, const TOKENS: usize> PreparedGdnInputPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QKV_MMA_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "FP8 GDN input prefill route T={TOKENS} is not admitted"
            )));
        }
        let token_tiles = TOKENS.div_ceil(QKV_MMA_PREFILL_BLOCK_ROWS);
        let projection_blocks = A::GDN_INPUT_ROWS / QKV_MMA_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks).map_err(|_| {
            GpuError::invalid_launch("FP8 GDN input prefill grid exceeds CUDA width")
        })?;
        let projection = module
            .prepare_fp8_gdn_input_mma::<TOKENS>(LaunchConfig1D::new(
                projection_blocks,
                QKV_MMA_PREFILL_THREADS,
                QKV_MMA_PREFILL_SHARED_BYTES,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the FP8 GDN input prefill projection", source)
            })?;

        Ok(Self {
            quantize: prepare_quantize::<TOKENS>(module)?,
            projection,
            architecture: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        let k_tiles = A::HIDDEN / 4 / QKV_MMA_K_WORDS;
        module
            .fp8_gdn_input_mma::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                k_tiles as u32,
            )
            .map_err(|source| {
                GpuError::launch("launching the FP8 GDN input prefill projection", source)
            })
    }
}

struct PreparedGdnInputT1024Route {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: DenseFp8GdnInputTmaRoute,
}

impl PreparedGdnInputT1024Route {
    fn prepare(context: &Arc<CudaContext>, module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            quantize: prepare_quantize::<QKV_MMA_MACRO_TOKENS>(module)?,
            projection: DenseFp8GdnInputTmaRoute::new(context)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8GdnInputTmaMaps,
    ) -> GpuResult<()> {
        module
            .quantize_activation_e4m3(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.projection
                .launch(stream, maps, activation_scales, weight_scales, output)
        }
    }
}

/// PTX symbols retained for activation quantization and every admitted QKV route.
pub(crate) fn fp8_qkv_ptx_names() -> [&'static str; 14] {
    [
        "quantize_activation_e4m3",
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_qkv_ptx_name::<Qwen38_27B, 8>(),
        "fp8_qkv_mma_t16",
        kernels::fp8_qkv_mma_ptx_name::<32>(),
        kernels::fp8_qkv_mma_ptx_name::<64>(),
        kernels::fp8_qkv_mma_ptx_name::<128>(),
        qkv_tma_ptx_name(),
    ]
}

/// PTX symbols retained for every exact GDN input projection route.
pub(crate) fn fp8_gdn_input_ptx_names() -> [&'static str; 12] {
    [
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 8>(),
        kernels::fp8_gdn_input_mma_ptx_name::<32>(),
        kernels::fp8_gdn_input_mma_ptx_name::<64>(),
        kernels::fp8_gdn_input_mma_ptx_name::<128>(),
        gdn_input_tma_ptx_name(),
    ]
}

/// PTX symbols retained for every exact full-vocabulary LM-head batch.
pub(crate) fn fp8_lm_head_ptx_names() -> [&'static str; 8] {
    [
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_lm_head_ptx_name::<Qwen38_27B, 8>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_fp8_qkv),
    required(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128),
    inventory(false)
)]
struct FullAttentionQkvRoutes<A: Arch> {
    #[route(1)]
    b1: PreparedQkvRoute<A, 1>,
    #[route(2)]
    b2: PreparedQkvRoute<A, 2>,
    #[route(3)]
    b3: PreparedQkvRoute<A, 3>,
    #[route(4)]
    b4: PreparedQkvRoute<A, 4>,
    #[route(5)]
    b5: PreparedQkvRoute<A, 5>,
    #[route(6)]
    b6: PreparedQkvRoute<A, 6>,
    #[route(7)]
    b7: PreparedQkvRoute<A, 7>,
    #[route(8)]
    b8: PreparedQkvRoute<A, 8>,
    #[route(16)]
    t16: PreparedQkvT16Route<A>,
    #[route(32)]
    t32: PreparedQkvPrefillRoute<A, 32>,
    #[route(64)]
    t64: PreparedQkvPrefillRoute<A, 64>,
    #[route(128)]
    t128: PreparedQkvPrefillRoute<A, 128>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_fp8_gdn_input),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct GdnInputProjectionRoutes<A: Arch> {
    #[route(1)]
    b1: PreparedGdnInputRoute<A, 1>,
    #[route(2)]
    b2: PreparedGdnInputRoute<A, 2>,
    #[route(3)]
    b3: PreparedGdnInputRoute<A, 3>,
    #[route(4)]
    b4: PreparedGdnInputRoute<A, 4>,
    #[route(5)]
    b5: PreparedGdnInputRoute<A, 5>,
    #[route(6)]
    b6: PreparedGdnInputRoute<A, 6>,
    #[route(7)]
    b7: PreparedGdnInputRoute<A, 7>,
    #[route(8)]
    b8: PreparedGdnInputRoute<A, 8>,
    #[route(32)]
    t32: PreparedGdnInputPrefillRoute<A, 32>,
    #[route(64)]
    t64: PreparedGdnInputPrefillRoute<A, 64>,
    #[route(128)]
    t128: PreparedGdnInputPrefillRoute<A, 128>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_fp8_lm_head),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct LmHeadRoutes<A: Arch> {
    #[route(1)]
    b1: PreparedLmHeadRoute<A, 1>,
    #[route(2)]
    b2: PreparedLmHeadRoute<A, 2>,
    #[route(3)]
    b3: PreparedLmHeadRoute<A, 3>,
    #[route(4)]
    b4: PreparedLmHeadRoute<A, 4>,
    #[route(5)]
    b5: PreparedLmHeadRoute<A, 5>,
    #[route(6)]
    b6: PreparedLmHeadRoute<A, 6>,
    #[route(7)]
    b7: PreparedLmHeadRoute<A, 7>,
    #[route(8)]
    b8: PreparedLmHeadRoute<A, 8>,
}

/// Prepared dynamic-quantize plus source-native QKV routes for decode, MTP, and prefill.
pub struct FullAttentionQkvOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: FullAttentionQkvRoutes<A>,
    t1024: PreparedQkvT1024Route,
}

impl<A: Sm120Arch> FullAttentionQkvOp<A> {
    /// Loads the embedded SM120 module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_fp8_geometry::<A>()?;
        let _ = fp8_qkv_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the FP8 QKV module", source))?;

        Ok(Self {
            routes: FullAttentionQkvRoutes::prepare(&module)?,
            t1024: PreparedQkvT1024Route::prepare(context, &module)?,
            module,
        })
    }

    /// Dynamically quantizes and applies the exact T=1024 TMA QKV route.
    ///
    /// # Safety
    ///
    /// The pointer contract matches [`Self::launch`] at exactly 1024 rows.
    /// `maps` was constructed for the same `activation_codes` and
    /// `weight_codes` addresses, which remain live and stable through replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_macro_prefill(
        &self,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8QkvTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 QKV tensor maps do not match the launch addresses",
            ));
        }
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024.launch(
                &self.module,
                stream,
                input,
                activation_codes,
                activation_scales,
                weight_scales,
                output,
                maps,
            )
        }
    }

    /// Dynamically quantizes and applies the T=128 tile of the TMA QKV route.
    ///
    /// # Safety
    ///
    /// The pointer contract matches [`Self::launch`] at exactly 128 rows.
    /// `maps` was constructed for the same `activation_codes` and
    /// `weight_codes` addresses, which remain live and stable through replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_t128_prefill(
        &self,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8QkvTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 QKV tensor maps do not match the launch addresses",
            ));
        }
        self.module
            .quantize_activation_e4m3(
                stream,
                &self.routes.t128.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024.projection.launch_t128(
                stream,
                maps,
                activation_scales,
                weight_scales,
                output,
            )
        }
    }

    /// Dynamically quantizes an admitted row count and projects fused Q/K/V output.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * A::HIDDEN` BF16 values. `activation_codes` covers
    /// at least 64 rows for T=32 and otherwise `rows`, so the retained padded CTA
    /// can read its complete immutable tile; `activation_scales` covers `rows` FP32 values;
    /// weights cover `[A::ATTENTION_QKV_ROWS, A::HIDDEN]` E4M3 codes and one
    /// BF16 scale per output row; and `output` covers all projected rows.
    /// Four-byte-loaded planes must be four-byte aligned. All allocations must
    /// belong to `stream`'s context, remain live through completion, and not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_fp8_qkv!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    output,
                )
            },
            else => {
                if rows == QKV_MMA_MACRO_TOKENS {
                    Err(GpuError::invalid_launch(
                        "FP8 QKV T=1024 requires launch_macro_prefill with its tensor maps",
                    ))
                } else {
                    Err(GpuError::invalid_launch(format!(
                        "FP8 QKV row count {rows} is outside the admitted routes 1..={MAX_BATCH}, 16, 32, 64, 128, and 1024"
                    )))
                }
            }
        )
    }
}

/// Prepared dynamic-quantize plus source-native GDN Q/K/V/Z decode and prefill routes.
pub struct GdnInputProjectionOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: GdnInputProjectionRoutes<A>,
    t1024: PreparedGdnInputT1024Route,
}

impl<A: Sm120Arch> GdnInputProjectionOp<A> {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_fp8_geometry::<A>()?;
        let _ = fp8_gdn_input_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the FP8 projection module", source))?;

        Ok(Self {
            routes: GdnInputProjectionRoutes::prepare(&module)?,
            t1024: PreparedGdnInputT1024Route::prepare(context, &module)?,
            module,
        })
    }

    /// Dynamically quantizes and applies the exact T=1024 TMA GDN input route.
    ///
    /// # Safety
    ///
    /// The pointer contract matches [`Self::launch`] at exactly 1024 rows.
    /// `maps` was constructed for the same `activation_codes` and
    /// `weight_codes` addresses, which remain live and stable through replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_macro_prefill(
        &self,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8GdnInputTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 GDN input tensor maps do not match the launch addresses",
            ));
        }
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024.launch(
                &self.module,
                stream,
                input,
                activation_codes,
                activation_scales,
                weight_scales,
                output,
                maps,
            )
        }
    }

    /// Dynamically quantizes and applies the T=128 tile of the TMA GDN input route.
    ///
    /// # Safety
    ///
    /// The pointer contract matches [`Self::launch`] at exactly 128 rows.
    /// `maps` was constructed for the same `activation_codes` and
    /// `weight_codes` addresses, which remain live and stable through replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_t128_prefill(
        &self,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
        maps: &DenseFp8GdnInputTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 GDN input tensor maps do not match the launch addresses",
            ));
        }
        self.module
            .quantize_activation_e4m3(
                stream,
                &self.routes.t128.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching FP8 activation quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024.projection.launch_t128(
                stream,
                maps,
                activation_scales,
                weight_scales,
                output,
            )
        }
    }

    /// Dynamically quantizes an exact row count and projects fused GDN Q/K/V/Z output.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * A::HIDDEN` BF16 values. `activation_codes` covers
    /// at least 64 rows for T=32 and otherwise `rows`, so the retained padded CTA
    /// can read its complete immutable tile; `activation_scales` covers `rows` FP32 values;
    /// weights cover `[A::GDN_INPUT_ROWS, A::HIDDEN]` E4M3 codes and one BF16
    /// scale per output row; and `output` covers all projected rows.
    /// Four-byte-loaded planes must be four-byte aligned. All allocations must
    /// belong to `stream`'s context, remain live through completion, and not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_fp8_gdn_input!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    output,
                )
            },
            else => {
                if rows == QKV_MMA_MACRO_TOKENS {
                    Err(GpuError::invalid_launch(
                        "FP8 GDN input T=1024 requires launch_macro_prefill with its tensor maps",
                    ))
                } else {
                    Err(GpuError::invalid_launch(format!(
                        "FP8 GDN input row count {rows} is outside the admitted routes 1..={MAX_BATCH}, 32, 64, 128, and 1024"
                    )))
                }
            }
        )
    }
}

/// Prepared dynamic-quantize plus full-vocabulary LM-head routes for exact `B=1..=8`.
pub struct LmHeadOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: LmHeadRoutes<A>,
}

impl<A: Sm120Arch> LmHeadOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_fp8_geometry::<A>()?;
        let _ = fp8_lm_head_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the FP8 projection module", source))?;

        Ok(Self {
            routes: LmHeadRoutes::prepare(&module)?,
            module,
        })
    }

    /// Dynamically quantizes an exact batch and projects full-vocabulary logits.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * A::HIDDEN` BF16 values; `activation_codes` covers
    /// the same number of bytes; `activation_scales` covers `batch` FP32 values;
    /// weights cover `[A::VOCAB, A::HIDDEN]` E4M3 codes and one BF16 scale per
    /// output row; and `output` covers all logits.
    /// Four-byte-loaded planes must be four-byte aligned. All allocations must
    /// belong to `stream`'s context, remain live through completion, and not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_fp8_lm_head!(
            &self.routes,
            batch,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    output,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "FP8 LM-head batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DECODE_PROJECTION_THREADS, DECODE_PROJECTION_WARPS, FullAttentionQkvRoutes,
        GdnInputProjectionRoutes, LmHeadRoutes, MAX_BATCH, QKV_MMA_K_WORDS, QKV_MMA_MACRO_K_WORDS,
        QKV_MMA_MACRO_SHARED_BYTES, QKV_MMA_MACRO_TOKENS, QKV_MMA_OUTPUT_ROWS,
        QKV_MMA_PREFILL_BLOCK_ROWS, QKV_MMA_PREFILL_SHARED_BYTES, QKV_MMA_PREFILL_THREADS,
        QKV_MMA_PREFILL_TOKENS, QKV_MMA_T16_BLOCK_ROWS, QKV_MMA_T16_SHARED_BYTES,
        QKV_MMA_T16_THREADS, QKV_MMA_T16_TOKENS, QUANTIZE_THREADS, fp8_gdn_input_ptx_names,
        fp8_geometry, fp8_lm_head_ptx_names, fp8_qkv_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_kernels_sm120_common::TestArch;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn exact_geometry_matches_the_retained_decode_schedule() {
        assert_eq!(QUANTIZE_THREADS, 256);
        assert_eq!((Qwen38_27B::HIDDEN / 2) / QUANTIZE_THREADS as usize, 10);
        assert_eq!(DECODE_PROJECTION_THREADS, 256);
        assert_eq!(DECODE_PROJECTION_WARPS, 8);
        assert_eq!(
            Qwen38_27B::ATTENTION_QKV_ROWS % (2 * DECODE_PROJECTION_WARPS),
            0
        );
        assert_eq!(
            Qwen38_27B::GDN_INPUT_ROWS % (2 * DECODE_PROJECTION_WARPS),
            0
        );
        assert_eq!(Qwen38_27B::VOCAB % (2 * DECODE_PROJECTION_WARPS), 0);
        assert_eq!(Qwen38_27B::HIDDEN % (32 * 16), 0);
        assert_eq!(MAX_BATCH, 8);
    }

    #[test]
    fn route_tables_cover_only_the_admitted_widths() {
        // T=1024 lives outside the route table as the op-level TMA field.
        assert_eq!(
            FullAttentionQkvRoutes::<Qwen38_27B>::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]
        );
        assert_eq!(
            GdnInputProjectionRoutes::<Qwen38_27B>::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]
        );
        assert_eq!(
            LmHeadRoutes::<Qwen38_27B>::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn prefill_geometry_matches_the_retained_mma_schedules() {
        assert_eq!(QKV_MMA_T16_TOKENS, 16);
        assert_eq!(QKV_MMA_PREFILL_TOKENS, [32, 64, 128]);
        assert_eq!(QKV_MMA_MACRO_TOKENS, 1_024);
        assert_eq!(QKV_MMA_OUTPUT_ROWS, 64);
        assert_eq!(QKV_MMA_T16_BLOCK_ROWS, 16);
        assert_eq!(QKV_MMA_PREFILL_BLOCK_ROWS, 64);
        assert_eq!(QKV_MMA_K_WORDS, 32);
        assert_eq!(QKV_MMA_MACRO_K_WORDS, 16);
        assert_eq!(QKV_MMA_T16_THREADS, 64);
        assert_eq!(QKV_MMA_PREFILL_THREADS, 256);
        assert_eq!(QKV_MMA_T16_SHARED_BYTES, 20_480);
        assert_eq!(QKV_MMA_PREFILL_SHARED_BYTES, 32_768);
        assert_eq!(QKV_MMA_MACRO_SHARED_BYTES, 16_384);
        assert_eq!(Qwen38_27B::ATTENTION_QKV_ROWS % QKV_MMA_OUTPUT_ROWS, 0);
        assert_eq!(Qwen38_27B::GDN_INPUT_ROWS % QKV_MMA_OUTPUT_ROWS, 0);
        assert_eq!((Qwen38_27B::HIDDEN / 4) % QKV_MMA_K_WORDS, 0);
        assert_eq!((Qwen38_27B::HIDDEN / 4) % QKV_MMA_MACRO_K_WORDS, 0);
    }

    #[test]
    fn geometry_admits_only_the_hardcoded_decode_width() {
        let qwen = fp8_geometry::<Qwen38_27B>().unwrap();

        assert_eq!(qwen.quantize_pairs_per_thread, 10);
        assert_eq!(qwen.qkv_decode_blocks, 896);
        assert_eq!(qwen.gdn_decode_blocks, 1_024);
        assert_eq!(qwen.lm_head_decode_blocks, 15_520);
        assert_eq!(qwen.qkv_mma_output_tiles, 224);
        assert_eq!(qwen.qkv_mma_k_tiles, 40);
        // HIDDEN=1024 satisfies every divisibility requirement but not the
        // 5,120 width baked into the decode kernel instantiations.
        assert!(fp8_geometry::<TestArch>().is_none());
    }

    #[test]
    fn ptx_inventory_has_quantization_and_one_entry_per_route() {
        let names = fp8_qkv_ptx_names()
            .into_iter()
            .chain(fp8_gdn_input_ptx_names())
            .chain(fp8_lm_head_ptx_names())
            .collect::<Vec<_>>();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 3 * MAX_BATCH + 10);
        assert_eq!(unique.len(), names.len());
    }
}

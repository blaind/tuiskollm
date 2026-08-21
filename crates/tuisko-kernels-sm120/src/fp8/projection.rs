//! Source-native FP8 projections with dynamic E4M3 activation quantization.

use crate::device::fp8_projection::{fp8_projection, qkv_projection_mma_t16, quantize_activation};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
// Eight warps quantize one 5,120-wide row: each thread owns exactly ten BF16 pairs.
const QUANTIZE_WARPS: usize = 8;
const QUANTIZE_THREADS: u32 = (QUANTIZE_WARPS * 32) as u32;
// Retained paired-row schedule: eight warps produce 16 rows per CTA; B=1 reached
// 90.65% measured DRAM throughput without changing the per-row accumulation order.
const DECODE_PROJECTION_WARPS: usize = 8;
const DECODE_PROJECTION_THREADS: u32 = (DECODE_PROJECTION_WARPS * 32) as u32;
const QKV_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
const GDN_INPUT_ROWS: usize = Qwen38_27B::GDN_INPUT_ROWS;
// The MTP boundary is exactly 16 rows. Two warps cover one 16x64 output tile,
// with 128 K bytes per stage split into four m16n8k32 subtiles.
const QKV_MMA_TOKENS: usize = 16;
const QKV_MMA_OUTPUT_ROWS: usize = 64;
const QKV_MMA_K_WORDS: usize = 32;
const QKV_MMA_THREADS: u32 = 64;
const QKV_MMA_SHARED_BYTES: u32 =
    (2 * (QKV_MMA_TOKENS + QKV_MMA_OUTPUT_ROWS) * QKV_MMA_K_WORDS * size_of::<u32>()) as u32;

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
            fp8_projection::<A, TOKENS, QKV_ROWS, DECODE_PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
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
            fp8_projection::<A, TOKENS, GDN_INPUT_ROWS, DECODE_PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
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

struct PreparedGdnInputRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_gdn_input_CudaKernel<Qwen38_27B, TOKENS>>,
}

impl<const TOKENS: usize> PreparedGdnInputRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let row_pairs_per_block = 2 * DECODE_PROJECTION_WARPS;
        let projection_blocks = GDN_INPUT_ROWS / row_pairs_per_block;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("GDN input rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_gdn_input::<Qwen38_27B, TOKENS>(LaunchConfig1D::new(
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
            .fp8_gdn_input::<Qwen38_27B, TOKENS>(
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

struct PreparedQkvT16Route {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_qkv_mma_t16_CudaKernel>,
}

impl PreparedQkvT16Route {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection_blocks = Qwen38_27B::ATTENTION_QKV_ROWS / QKV_MMA_OUTPUT_ROWS;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("FP8 QKV rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_qkv_mma_t16(LaunchConfig1D::new(
                projection_blocks,
                QKV_MMA_THREADS,
                QKV_MMA_SHARED_BYTES,
            ))
            .map_err(|source| GpuError::launch("preparing the FP8 QKV T=16 projection", source))?;

        Ok(Self {
            quantize: prepare_quantize::<QKV_MMA_TOKENS>(module)?,
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
        let k_tiles = Qwen38_27B::HIDDEN / 4 / QKV_MMA_K_WORDS;
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

/// PTX symbols retained for activation quantization and every admitted QKV route.
pub(crate) fn fp8_qkv_ptx_names() -> [&'static str; 10] {
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
    ]
}

/// PTX symbols retained for every exact GDN input projection batch.
pub(crate) fn fp8_gdn_input_ptx_names() -> [&'static str; 8] {
    [
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_gdn_input_ptx_name::<Qwen38_27B, 8>(),
    ]
}

/// Prepared dynamic-quantize plus source-native QKV routes for exact `B=1..=8` and `T=16`.
pub struct FullAttentionQkvOp {
    module: kernels::LoadedModule,
    b1: PreparedQkvRoute<Qwen38_27B, 1>,
    b2: PreparedQkvRoute<Qwen38_27B, 2>,
    b3: PreparedQkvRoute<Qwen38_27B, 3>,
    b4: PreparedQkvRoute<Qwen38_27B, 4>,
    b5: PreparedQkvRoute<Qwen38_27B, 5>,
    b6: PreparedQkvRoute<Qwen38_27B, 6>,
    b7: PreparedQkvRoute<Qwen38_27B, 7>,
    b8: PreparedQkvRoute<Qwen38_27B, 8>,
    t16: PreparedQkvT16Route,
}

impl FullAttentionQkvOp {
    /// Loads the embedded SM120 module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = fp8_qkv_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the FP8 QKV module", source))?;

        Ok(Self {
            b1: PreparedQkvRoute::prepare(&module)?,
            b2: PreparedQkvRoute::prepare(&module)?,
            b3: PreparedQkvRoute::prepare(&module)?,
            b4: PreparedQkvRoute::prepare(&module)?,
            b5: PreparedQkvRoute::prepare(&module)?,
            b6: PreparedQkvRoute::prepare(&module)?,
            b7: PreparedQkvRoute::prepare(&module)?,
            b8: PreparedQkvRoute::prepare(&module)?,
            t16: PreparedQkvT16Route::prepare(&module)?,
            module,
        })
    }

    /// Dynamically quantizes an admitted row count and projects fused Q/K/V output.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * 5_120` BF16 values; `activation_codes` covers
    /// `rows * 5_120` bytes; `activation_scales` covers `rows` FP32 values;
    /// `weight_codes` and `weight_scales` cover `[14_336, 5_120]` E4M3 codes and
    /// 14,336 BF16 scales; and `output` covers `rows * 14_336` BF16 values.
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
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        activation_codes,
                        activation_scales,
                        weight_codes,
                        weight_scales,
                        output,
                    )
                }
            };
        }

        match rows {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            16 => launch!(t16),
            _ => Err(GpuError::invalid_launch(format!(
                "FP8 QKV row count {rows} is outside the admitted routes 1..={MAX_BATCH} and 16"
            ))),
        }
    }
}

/// Prepared dynamic-quantize plus source-native GDN Q/K/V/Z routes for exact `B=1..=8`.
pub struct GdnInputProjectionOp {
    module: kernels::LoadedModule,
    b1: PreparedGdnInputRoute<1>,
    b2: PreparedGdnInputRoute<2>,
    b3: PreparedGdnInputRoute<3>,
    b4: PreparedGdnInputRoute<4>,
    b5: PreparedGdnInputRoute<5>,
    b6: PreparedGdnInputRoute<6>,
    b7: PreparedGdnInputRoute<7>,
    b8: PreparedGdnInputRoute<8>,
}

impl GdnInputProjectionOp {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = fp8_gdn_input_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the FP8 projection module", source))?;

        Ok(Self {
            b1: PreparedGdnInputRoute::prepare(&module)?,
            b2: PreparedGdnInputRoute::prepare(&module)?,
            b3: PreparedGdnInputRoute::prepare(&module)?,
            b4: PreparedGdnInputRoute::prepare(&module)?,
            b5: PreparedGdnInputRoute::prepare(&module)?,
            b6: PreparedGdnInputRoute::prepare(&module)?,
            b7: PreparedGdnInputRoute::prepare(&module)?,
            b8: PreparedGdnInputRoute::prepare(&module)?,
            module,
        })
    }

    /// Dynamically quantizes an exact batch and projects fused GDN Q/K/V/Z output.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 5_120` BF16 values; `activation_codes` covers
    /// `batch * 5_120` bytes; `activation_scales` covers `batch` FP32 values;
    /// `weight_codes` and `weight_scales` cover `[16_384, 5_120]` E4M3 codes and
    /// 16,384 BF16 scales; and `output` covers `batch * 16_384` BF16 values.
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
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        activation_codes,
                        activation_scales,
                        weight_codes,
                        weight_scales,
                        output,
                    )
                }
            };
        }

        match batch {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            _ => Err(GpuError::invalid_launch(format!(
                "FP8 GDN input batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DECODE_PROJECTION_THREADS, DECODE_PROJECTION_WARPS, MAX_BATCH, QKV_MMA_K_WORDS,
        QKV_MMA_OUTPUT_ROWS, QKV_MMA_SHARED_BYTES, QKV_MMA_THREADS, QKV_MMA_TOKENS,
        QUANTIZE_THREADS, fp8_gdn_input_ptx_names, fp8_qkv_ptx_names,
    };
    use std::collections::BTreeSet;
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
        assert_eq!(Qwen38_27B::HIDDEN % (32 * 16), 0);
        assert_eq!(MAX_BATCH, 8);
    }

    #[test]
    fn t16_geometry_matches_the_retained_mma_schedule() {
        assert_eq!(QKV_MMA_TOKENS, 16);
        assert_eq!(QKV_MMA_OUTPUT_ROWS, 64);
        assert_eq!(QKV_MMA_K_WORDS, 32);
        assert_eq!(QKV_MMA_THREADS, 64);
        assert_eq!(QKV_MMA_SHARED_BYTES, 20_480);
        assert_eq!(Qwen38_27B::ATTENTION_QKV_ROWS % QKV_MMA_OUTPUT_ROWS, 0);
        assert_eq!((Qwen38_27B::HIDDEN / 4) % QKV_MMA_K_WORDS, 0);
    }

    #[test]
    fn ptx_inventory_has_quantization_and_one_entry_per_route() {
        let names = fp8_qkv_ptx_names()
            .into_iter()
            .chain(fp8_gdn_input_ptx_names())
            .collect::<Vec<_>>();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 2 * MAX_BATCH + 2);
        assert_eq!(unique.len(), names.len());
    }
}

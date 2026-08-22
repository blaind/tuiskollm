//! Source-native FP8 QKV projection with dynamic E4M3 activation quantization.

use crate::Sm89Arch;
use crate::device::fp8_projection::{fp8_projection, quantize_activation};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;

// One 256-thread CTA owns one 5,120-wide token: each thread quantizes ten
// BF16 pairs, and eight warp maxima reduce through 32 bytes of shared memory.
const QUANTIZE_WARPS: usize = 8;
const QUANTIZE_THREADS: u32 = (QUANTIZE_WARPS * 32) as u32;

// One warp retains two output-row reductions and reads each 512-byte
// activation phase once for that pair. Eight warps produce 16 rows per CTA;
// 14,336 / 16 = 896 CTAs provide seven waves on 128 SMs. Pair ownership and
// lane reduction order remain fixed for every exact batch.
const PROJECTION_WARPS: usize = 8;
const PROJECTION_THREADS: u32 = (PROJECTION_WARPS * 32) as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fp8Geometry {
    quantize_pairs_per_thread: usize,
    projection_blocks: usize,
}

fn fp8_geometry<A: Arch>() -> Option<Fp8Geometry> {
    let rows_per_block = 2 * PROJECTION_WARPS;
    if A::HIDDEN == 0
        || !A::HIDDEN.is_multiple_of(512)
        || !A::ATTENTION_QKV_ROWS.is_multiple_of(rows_per_block)
    {
        return None;
    }

    Some(Fp8Geometry {
        quantize_pairs_per_thread: (A::HIDDEN / 2).div_ceil(QUANTIZE_THREADS as usize),
        projection_blocks: A::ATTENTION_QKV_ROWS / rows_per_block,
    })
}

fn require_fp8_geometry<A: Arch>() -> GpuResult<Fp8Geometry> {
    fp8_geometry::<A>().ok_or_else(|| {
        GpuError::invalid_launch("architecture geometry is incompatible with SM89 FP8 QKV")
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
        min_compute_capability = (8, 9),
    )]
    pub fn quantize_activation_e4m3(input: *const u32, codes: *mut u16, scales: *mut f32) {
        static mut WARP_MAXIMUM: SharedArray<f32, QUANTIZE_WARPS, 16> = SharedArray::UNINIT;

        unsafe {
            quantize_activation::<Qwen38_27B>(
                input,
                codes,
                scales,
                core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>(),
            );
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
        min_compute_capability = (8, 9),
    )]
    pub fn fp8_qkv<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        unsafe {
            fp8_projection::<5_120, TOKENS, PROJECTION_WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                A::ATTENTION_QKV_ROWS,
            );
        }
    }
}

fn prepare_quantize<const TOKENS: usize>(
    module: &kernels::LoadedModule,
) -> GpuResult<PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>> {
    let blocks = u32::try_from(TOKENS)
        .map_err(|_| GpuError::invalid_launch("FP8 QKV token count exceeds CUDA grid width"))?;
    module
        .prepare_quantize_activation_e4m3(LaunchConfig1D::new(blocks, QUANTIZE_THREADS, 0))
        .map_err(|source| GpuError::launch("preparing SM89 FP8 activation quantization", source))
}

struct PreparedQkvRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__quantize_activation_e4m3_CudaKernel>,
    projection: PreparedLaunch<kernels::__fp8_qkv_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedQkvRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let geometry = require_fp8_geometry::<A>()?;
        let blocks = u32::try_from(geometry.projection_blocks)
            .map_err(|_| GpuError::invalid_launch("FP8 QKV rows exceed CUDA grid width"))?;
        let projection = module
            .prepare_fp8_qkv::<A, TOKENS>(LaunchConfig1D::new(blocks, PROJECTION_THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the SM89 FP8 QKV projection", source))?;

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
            .map_err(|source| GpuError::launch("launching SM89 FP8 quantization", source))?;
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
            .map_err(|source| GpuError::launch("launching the SM89 FP8 QKV projection", source))
    }
}

/// PTX symbols retained for activation quantization and every exact SM89 QKV batch.
pub(crate) fn fp8_qkv_ptx_names() -> [&'static str; MAX_BATCH + 1] {
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
    ]
}

/// Prepared dynamic-quantize plus source-native QKV routes for exact `B=1..=8`.
pub struct FullAttentionQkvOp<A: Sm89Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedQkvRoute<A, 1>,
    b2: PreparedQkvRoute<A, 2>,
    b3: PreparedQkvRoute<A, 3>,
    b4: PreparedQkvRoute<A, 4>,
    b5: PreparedQkvRoute<A, 5>,
    b6: PreparedQkvRoute<A, 6>,
    b7: PreparedQkvRoute<A, 7>,
    b8: PreparedQkvRoute<A, 8>,
}

impl<A: Sm89Arch> FullAttentionQkvOp<A> {
    /// Loads the embedded SM89 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_fp8_geometry::<A>()?;
        let _ = fp8_qkv_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the SM89 FP8 QKV module", source))?;

        Ok(Self {
            b1: PreparedQkvRoute::prepare(&module)?,
            b2: PreparedQkvRoute::prepare(&module)?,
            b3: PreparedQkvRoute::prepare(&module)?,
            b4: PreparedQkvRoute::prepare(&module)?,
            b5: PreparedQkvRoute::prepare(&module)?,
            b6: PreparedQkvRoute::prepare(&module)?,
            b7: PreparedQkvRoute::prepare(&module)?,
            b8: PreparedQkvRoute::prepare(&module)?,
            module,
        })
    }

    /// Quantizes and projects fused Q/K/V output for exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * A::HIDDEN` BF16 values; `activation_codes`
    /// covers the same number of bytes; `activation_scales` covers `batch`
    /// FP32 values; weights cover `[A::ATTENTION_QKV_ROWS, A::HIDDEN]` E4M3
    /// codes and one BF16 scale per output row; and `output` covers the complete
    /// projected batch. Four-byte-loaded planes are four-byte aligned. All
    /// allocations share `stream`'s context, remain live, and do not overlap.
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
                "SM89 FP8 QKV batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, fp8_geometry, fp8_qkv_ptx_names};
    use std::collections::BTreeSet;
    use tuisko_model::Qwen38_27B;

    #[test]
    fn exact_geometry_matches_the_qkv_owner() {
        let geometry = fp8_geometry::<Qwen38_27B>().unwrap();

        assert_eq!(geometry.quantize_pairs_per_thread, 10);
        assert_eq!(geometry.projection_blocks, 896);
    }

    #[test]
    fn inventory_has_quantization_and_one_projection_per_batch() {
        let names = fp8_qkv_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH + 1);
        assert_eq!(unique.len(), names.len());
    }
}

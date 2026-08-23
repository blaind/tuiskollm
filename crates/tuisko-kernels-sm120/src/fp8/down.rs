//! Source-native dense-FP8 MLP down projection.

use crate::Sm120Arch;
use crate::device::fp8_down::{fp8_down_projection, quantize_down_activation};
use crate::fp8::down_tma::{
    DenseFp8DownTmaMaps, DenseFp8DownTmaRoute, TOKENS as MACRO_TOKENS, ptx_name as tma_ptx_name,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::INTERMEDIATE == 0
        || !A::INTERMEDIATE.is_multiple_of(512)
        || !A::HIDDEN.is_multiple_of(2 * WARPS)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the dense-FP8 down schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Dynamically quantizes one BF16 SwiGLU row to E4M3 plus an FP32 scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_down_quantize<A: Arch>(input: *const u32, codes: *mut u16, scales: *mut f32) {
        static mut WARP_MAXIMUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let warp_maximum = core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>();

        // SAFETY: one launch block owns one complete intermediate-width row.
        unsafe {
            quantize_down_activation::<A>(input, codes, scales, warp_maximum);
        }
    }

    /// Projects one exact batch through the source-native down matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_down<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // The 89.1 MB source plane measured 71.200 us / 1,252.8 GB/s at
        // B=1 and 73.696 us / 1,216.3 GB/s at B=8. Eight warps pair adjacent
        // output rows and reuse each weight fragment across exactly the live
        // batch; every lane keeps the same 34 phases and warp-reduction order.
        // SAFETY: the exact grid assigns every output row pair once.
        unsafe {
            fp8_down_projection::<A, TOKENS, WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
            );
        }
    }
}

fn prepare_quantize<A: Arch, const TOKENS: usize>(
    module: &kernels::LoadedModule,
) -> GpuResult<PreparedLaunch<kernels::__fp8_down_quantize_CudaKernel<A>>> {
    module
        .prepare_fp8_down_quantize::<A>(LaunchConfig1D::new(TOKENS as u32, THREADS, 0))
        .map_err(|source| GpuError::launch("preparing dense-FP8 down quantization", source))
}

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__fp8_down_quantize_CudaKernel<A>>,
    projection: PreparedLaunch<kernels::__fp8_down_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(A::HIDDEN / (2 * WARPS))
            .map_err(|_| GpuError::invalid_launch("dense-FP8 down rows exceed grid width"))?;
        let projection = module
            .prepare_fp8_down::<A, TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(|source| GpuError::launch("preparing dense-FP8 down projection", source))?;

        Ok(Self {
            quantize: prepare_quantize::<A, TOKENS>(module)?,
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
            .fp8_down_quantize::<A>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 down quantization", source))?;
        module
            .fp8_down::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 down projection", source))
    }
}

/// Prepared source-native dense-FP8 down routes for every exact decode batch.
pub struct DenseFp8DownOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedRoute<A, 1>,
    b2: PreparedRoute<A, 2>,
    b3: PreparedRoute<A, 3>,
    b4: PreparedRoute<A, 4>,
    b5: PreparedRoute<A, 5>,
    b6: PreparedRoute<A, 6>,
    b7: PreparedRoute<A, 7>,
    b8: PreparedRoute<A, 8>,
    t1024_quantize: PreparedLaunch<kernels::__fp8_down_quantize_CudaKernel<A>>,
    t1024: DenseFp8DownTmaRoute,
}

impl<A: Sm120Arch> DenseFp8DownOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = fp8_down_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the dense-FP8 down module", source))?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            t1024_quantize: prepare_quantize::<A, MACRO_TOKENS>(&module)?,
            t1024: DenseFp8DownTmaRoute::new(context)?,
            module,
        })
    }

    /// Dynamically quantizes and applies the source-native down projection.
    ///
    /// # Safety
    ///
    /// `input` and `activation_codes` cover `batch * A::INTERMEDIATE` values;
    /// `activation_scales` covers `batch`; weights cover
    /// `[A::HIDDEN, A::INTERMEDIATE]` E4M3 codes and one BF16 scale per output
    /// row; and `output` covers `[batch, A::HIDDEN]` BF16 values. Four-byte
    /// planes are aligned. Allocations belong to `stream`'s context, remain
    /// live through completion, and do not overlap.
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
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "dense-FP8 down batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-batch dispatch preserves the public pointer contract.
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
            _ => unreachable!(),
        }
    }

    /// Dynamically quantizes and applies the exact T=1024 TMA down route.
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
        maps: &DenseFp8DownTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 down tensor maps do not match the launch addresses",
            ));
        }
        self.module
            .fp8_down_quantize::<A>(
                stream,
                &self.t1024_quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 down quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024
                .launch(stream, maps, activation_scales, weight_scales, output)
        }
    }
}

/// PTX symbols retained for quantization and every exact dense-FP8 down route.
pub(crate) fn fp8_down_ptx_names() -> [&'static str; 10] {
    [
        kernels::fp8_down_quantize_ptx_name::<Qwen38_27B>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_down_ptx_name::<Qwen38_27B, 8>(),
        tma_ptx_name(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{THREADS, WARPS, admitted_batch, fp8_down_ptx_names};
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn batch_table_covers_only_the_exact_decode_routes() {
        let cases = [
            (0, false),
            (1, true),
            (4, true),
            (8, true),
            (9, false),
            (16, false),
        ];

        for (batch, expected) in cases {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
    }

    #[test]
    fn exact_geometry_matches_the_retained_decode_schedule() {
        assert_eq!(THREADS, 256);
        assert_eq!(WARPS, 8);
        assert_eq!(Qwen38_27B::INTERMEDIATE / (32 * 16), 34);
        assert_eq!(Qwen38_27B::HIDDEN % (2 * WARPS), 0);
    }

    #[test]
    fn ptx_inventory_has_quantization_and_one_entry_per_batch() {
        let names = fp8_down_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 10);
        assert_eq!(unique.len(), names.len());
    }
}

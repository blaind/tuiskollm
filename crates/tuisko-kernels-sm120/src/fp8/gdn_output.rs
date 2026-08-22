//! Source-native FP8 GDN output projection.

use crate::Sm120Arch;
use crate::device::fp8_projection::{fp8_projection, quantize_gdn_output_activation};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
// One 6,144-wide row contains 3,072 BF16 pairs, exactly 12 per thread.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::GDN_VALUE_ROWS != 6_144
        || A::HIDDEN != 5_120
        || !A::GDN_VALUE_ROWS.is_multiple_of(512)
        || !A::HIDDEN.is_multiple_of(2 * WARPS)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the FP8 GDN output schedule",
        ));
    }
    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Quantizes one 6,144-wide BF16 row to E4M3 plus an FP32 scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared = 0, min_compute_capability = (12, 0))]
    pub fn gdn_output_quantize(input: *const u32, codes: *mut u16, scales: *mut f32) {
        static mut WARP_MAXIMUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        unsafe {
            quantize_gdn_output_activation(
                input,
                codes,
                scales,
                core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>(),
            );
        }
    }

    /// Projects one exact batch through the source-native GDN output matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared = 0, min_compute_capability = (12, 0))]
    pub fn gdn_output_projection<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // Pairing 16 rows per CTA yields 320 CTAs over 170 SMs; each CTA streams
        // 96 KiB of the 31.5 MB source plane. This retains the qualified FP8
        // projection topology and each row's phase and reduction order exactly.
        unsafe {
            fp8_projection::<6_144, TOKENS, WARPS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                A::HIDDEN,
            );
        }
    }
}

struct Route<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__gdn_output_quantize_CudaKernel>,
    projection: PreparedLaunch<kernels::__gdn_output_projection_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> Route<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            quantize: module
                .prepare_gdn_output_quantize(LaunchConfig1D::new(TOKENS as u32, THREADS, 0))
                .map_err(|source| GpuError::launch("preparing GDN output quantization", source))?,
            projection: module
                .prepare_gdn_output_projection::<A, TOKENS>(LaunchConfig1D::new(
                    (A::HIDDEN / (2 * WARPS)) as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN output projection", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        codes: *mut u8,
        scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .gdn_output_quantize(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                codes.cast::<u16>(),
                scales,
            )
            .map_err(|source| GpuError::launch("launching GDN output quantization", source))?;
        module
            .gdn_output_projection::<A, TOKENS>(
                stream,
                &self.projection,
                codes.cast::<u32>(),
                scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching GDN output projection", source))
    }
}

/// Prepared source-native FP8 GDN output routes for exact `B=1..=8`.
pub struct GdnOutputProjectionOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: Route<A, 1>,
    b2: Route<A, 2>,
    b3: Route<A, 3>,
    b4: Route<A, 4>,
    b5: Route<A, 5>,
    b6: Route<A, 6>,
    b7: Route<A, 7>,
    b8: Route<A, 8>,
}

impl<A: Sm120Arch> GdnOutputProjectionOp<A> {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = gdn_output_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN output module", source))?;
        Ok(Self {
            b1: Route::prepare(&module)?,
            b2: Route::prepare(&module)?,
            b3: Route::prepare(&module)?,
            b4: Route::prepare(&module)?,
            b5: Route::prepare(&module)?,
            b6: Route::prepare(&module)?,
            b7: Route::prepare(&module)?,
            b8: Route::prepare(&module)?,
            module,
        })
    }

    /// Dynamically quantizes recurrent values and applies the output projection.
    ///
    /// # Safety
    ///
    /// Inputs and activation codes cover `batch * A::GDN_VALUE_ROWS`; scales
    /// cover `batch`; source weights cover `[A::HIDDEN, A::GDN_VALUE_ROWS]`
    /// E4M3 values plus one BF16 scale per output row; outputs cover
    /// `[batch, A::HIDDEN]`. Regions are aligned, non-overlapping, context-local,
    /// and live through completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        codes: *mut u8,
        scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "FP8 GDN output batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }
        macro_rules! call {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        codes,
                        scales,
                        weight_codes,
                        weight_scales,
                        output,
                    )
                }
            };
        }
        match batch {
            1 => call!(b1),
            2 => call!(b2),
            3 => call!(b3),
            4 => call!(b4),
            5 => call!(b5),
            6 => call!(b6),
            7 => call!(b7),
            8 => call!(b8),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn gdn_output_ptx_names() -> Vec<&'static str> {
    vec![
        "gdn_output_quantize",
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 1>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 2>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 3>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 4>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 5>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 6>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 7>(),
        kernels::gdn_output_projection_ptx_name::<Qwen38_27B, 8>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{admitted_batch, gdn_output_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn route_and_inventory_are_exact() {
        for (batch, expected) in [(0, false), (1, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected);
        }
        let names = gdn_output_ptx_names();
        assert_eq!(names.len(), 9);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 9);
    }
}

//! Source-native FP8 GDN output projection.

use crate::device::fp8_projection::{fp8_projection, quantize_gdn_output_activation};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_kernels_sm120_common::attention_output::attention_output_projection_mma;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
// One 6,144-wide row contains 3,072 BF16 pairs, exactly 12 per thread.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
const PREFILL_OUTPUT_ROWS: usize = 32;
const PREFILL_K_WORDS: usize = 32;
const PREFILL_K_SUBTILES: usize = 4;
const PREFILL_BLOCK_ROWS: usize = 32;
const PREFILL_THREADS: u32 = 64;
const PREFILL_SHARED_BYTES: u32 = 16 * 1_024;
const MACRO_BLOCK_ROWS: usize = 64;
const MACRO_THREADS: u32 = 128;
const MACRO_SHARED_BYTES: u32 = 24 * 1_024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteClass {
    Decode,
    Prefill,
    Macro,
}

fn route_class(rows: usize) -> Option<RouteClass> {
    match rows {
        1..=MAX_BATCH => Some(RouteClass::Decode),
        32 | 64 | 128 => Some(RouteClass::Prefill),
        1_024 => Some(RouteClass::Macro),
        _ => None,
    }
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::GDN_VALUE_ROWS != 6_144
        || A::HIDDEN != 5_120
        || !A::GDN_VALUE_ROWS.is_multiple_of(512)
        || !A::HIDDEN.is_multiple_of(2 * WARPS)
        || !A::HIDDEN.is_multiple_of(PREFILL_OUTPUT_ROWS)
        || !(A::GDN_VALUE_ROWS / 4).is_multiple_of(PREFILL_K_WORDS)
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

    /// Projects one exact 32/64/128-row GDN output tile with native E4M3 MMA.
    #[kernel]
    #[launch_bounds(64, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (64, 1, 1),
        dynamic_shared = 16384,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_output_projection_mma_exact<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // A 32x32 tile uses two warps for two 16-row token quadrants. Its
        // 16 KiB double buffer holds 32 activation and 32 weight rows at K=128.
        unsafe {
            attention_output_projection_mma::<
                Qwen38_27B,
                TOKENS,
                PREFILL_BLOCK_ROWS,
                PREFILL_OUTPUT_ROWS,
                PREFILL_K_WORDS,
                PREFILL_K_SUBTILES,
            >(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
            );
        }
    }

    /// Projects exactly 1,024 GDN output rows with native E4M3 MMA.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 24576,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_output_projection_mma_t1024(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // A 64x32 macro tile doubles activation reuse while four warps retain
        // exact 16x32 MMA quadrants. Its K=128 double buffer is exactly 24 KiB.
        unsafe {
            attention_output_projection_mma::<
                Qwen38_27B,
                1_024,
                MACRO_BLOCK_ROWS,
                PREFILL_OUTPUT_ROWS,
                PREFILL_K_WORDS,
                PREFILL_K_SUBTILES,
            >(
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

struct Route<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__gdn_output_quantize_CudaKernel>,
    projection: PreparedLaunch<kernels::__gdn_output_projection_CudaKernel<A, TOKENS>>,
}

struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__gdn_output_quantize_CudaKernel>,
    projection: PreparedLaunch<kernels::__gdn_output_projection_mma_exact_CudaKernel<TOKENS>>,
    _arch: core::marker::PhantomData<A>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS[..3].contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "GDN output prefill route T={TOKENS} is not admitted"
            )));
        }
        let token_tiles = TOKENS / PREFILL_BLOCK_ROWS;
        let projection_blocks = A::HIDDEN / PREFILL_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks)
            .map_err(|_| GpuError::invalid_launch("GDN output prefill grid exceeds CUDA width"))?;

        Ok(Self {
            quantize: module
                .prepare_gdn_output_quantize(LaunchConfig1D::new(TOKENS as u32, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing GDN output prefill quantization", source)
                })?,
            projection: module
                .prepare_gdn_output_projection_mma_exact::<TOKENS>(LaunchConfig1D::new(
                    projection_blocks,
                    PREFILL_THREADS,
                    PREFILL_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing GDN output prefill projection", source)
                })?,
            _arch: core::marker::PhantomData,
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
            .map_err(|source| {
                GpuError::launch("launching GDN output prefill quantization", source)
            })?;
        module
            .gdn_output_projection_mma_exact::<TOKENS>(
                stream,
                &self.projection,
                codes.cast::<u32>(),
                scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                (A::GDN_VALUE_ROWS / 4 / PREFILL_K_WORDS) as u32,
            )
            .map_err(|source| GpuError::launch("launching GDN output prefill projection", source))
    }
}

struct PreparedMacroRoute<A: Arch> {
    quantize: PreparedLaunch<kernels::__gdn_output_quantize_CudaKernel>,
    projection: PreparedLaunch<kernels::__gdn_output_projection_mma_t1024_CudaKernel>,
    _arch: core::marker::PhantomData<A>,
}

impl<A: Arch> PreparedMacroRoute<A> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = PREFILL_ROWS[3] / MACRO_BLOCK_ROWS;
        let projection_blocks = A::HIDDEN / PREFILL_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks).map_err(|_| {
            GpuError::invalid_launch("GDN output macro-prefill grid exceeds CUDA width")
        })?;

        Ok(Self {
            quantize: module
                .prepare_gdn_output_quantize(LaunchConfig1D::new(
                    PREFILL_ROWS[3] as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing GDN output macro quantization", source)
                })?,
            projection: module
                .prepare_gdn_output_projection_mma_t1024(LaunchConfig1D::new(
                    projection_blocks,
                    MACRO_THREADS,
                    MACRO_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing GDN output macro projection", source)
                })?,
            _arch: core::marker::PhantomData,
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
            .map_err(|source| {
                GpuError::launch("launching GDN output macro quantization", source)
            })?;
        module
            .gdn_output_projection_mma_t1024(
                stream,
                &self.projection,
                codes.cast::<u32>(),
                scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                (A::GDN_VALUE_ROWS / 4 / PREFILL_K_WORDS) as u32,
            )
            .map_err(|source| GpuError::launch("launching GDN output macro projection", source))
    }
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

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_gdn_output),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct GdnOutputRoutes<A: Sm120Arch> {
    #[route(1)]
    b1: Route<A, 1>,
    #[route(2)]
    b2: Route<A, 2>,
    #[route(3)]
    b3: Route<A, 3>,
    #[route(4)]
    b4: Route<A, 4>,
    #[route(5)]
    b5: Route<A, 5>,
    #[route(6)]
    b6: Route<A, 6>,
    #[route(7)]
    b7: Route<A, 7>,
    #[route(8)]
    b8: Route<A, 8>,
    #[route(32)]
    t32: PreparedPrefillRoute<A, 32>,
    #[route(64)]
    t64: PreparedPrefillRoute<A, 64>,
    #[route(128)]
    t128: PreparedPrefillRoute<A, 128>,
    #[route(1024)]
    t1024: PreparedMacroRoute<A>,
}

/// Prepared source-native FP8 GDN output routes for exact decode and prefill rows.
pub struct GdnOutputProjectionOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: GdnOutputRoutes<A>,
}

impl<A: Sm120Arch> GdnOutputProjectionOp<A> {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = gdn_output_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN output module", source))?;
        Ok(Self {
            routes: GdnOutputRoutes::prepare(&module)?,
            module,
        })
    }

    /// Dynamically quantizes recurrent values and applies the output projection.
    ///
    /// # Safety
    ///
    /// Inputs and activation codes cover `rows * A::GDN_VALUE_ROWS`; scales
    /// cover `rows`; source weights cover `[A::HIDDEN, A::GDN_VALUE_ROWS]`
    /// E4M3 values plus one BF16 scale per output row; outputs cover
    /// `[rows, A::HIDDEN]`. Regions are aligned, non-overlapping, context-local,
    /// and live through completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        codes: *mut u8,
        scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        let Some(class) = route_class(rows) else {
            return Err(GpuError::invalid_launch(format!(
                "FP8 GDN output row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024"
            )));
        };
        macro_rules! call {
            ($route:expr) => {
                unsafe {
                    $route.launch(
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
        let _ = class;
        dispatch_gdn_output!(&self.routes, rows, |route| call!(route), else => unreachable!())
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
        kernels::gdn_output_projection_mma_exact_ptx_name::<32>(),
        kernels::gdn_output_projection_mma_exact_ptx_name::<64>(),
        kernels::gdn_output_projection_mma_exact_ptx_name::<128>(),
        "gdn_output_projection_mma_t1024",
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        MACRO_SHARED_BYTES, MACRO_THREADS, PREFILL_SHARED_BYTES, PREFILL_THREADS, RouteClass,
        THREADS, gdn_output_ptx_names, route_class,
    };
    use std::collections::BTreeSet;

    #[test]
    fn route_and_inventory_are_exact() {
        for (rows, expected) in [
            (0, None),
            (1, Some(RouteClass::Decode)),
            (8, Some(RouteClass::Decode)),
            (9, None),
            (32, Some(RouteClass::Prefill)),
            (64, Some(RouteClass::Prefill)),
            (128, Some(RouteClass::Prefill)),
            (1_024, Some(RouteClass::Macro)),
            (1_025, None),
        ] {
            assert_eq!(route_class(rows), expected, "rows={rows}");
        }
        assert_eq!(THREADS, 256);
        assert_eq!((PREFILL_THREADS, PREFILL_SHARED_BYTES), (64, 16_384));
        assert_eq!((MACRO_THREADS, MACRO_SHARED_BYTES), (128, 24_576));
        let names = gdn_output_ptx_names();
        assert_eq!(names.len(), 13);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 13);
    }
}

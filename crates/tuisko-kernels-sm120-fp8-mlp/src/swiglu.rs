//! Source-native dense-FP8 gate/up projection with fused SwiGLU.

use crate::device::fp8_swiglu::{
    fp8_swiglu_decode as fp8_swiglu_decode_body, fp8_swiglu_decode_b1,
    fp8_swiglu_mma as fp8_swiglu_mma_body,
};
use crate::swiglu_tma::{
    DenseFp8SwiGluTmaMaps, DenseFp8SwiGluTmaRoute, TOKENS as MACRO_TOKENS, ptx_name as tma_ptx_name,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_kernels_sm120_common::device::quantize_activation;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const DECODE_WARPS: usize = 8;
const DECODE_THREADS: u32 = (DECODE_WARPS * 32) as u32;
const QUANTIZE_WARPS: usize = 8;
const QUANTIZE_THREADS: u32 = (QUANTIZE_WARPS * 32) as u32;
const PREFILL_OUTPUT_ROWS: usize = 64;
const PREFILL_THREADS: u32 = 256;
const PREFILL_K128_WORDS: usize = 32;
const PREFILL_K64_WORDS: usize = 16;
const PREFILL_K128_SHARED_BYTES: u32 = 48 * 1024;
const PREFILL_K64_SHARED_BYTES: u32 = 24 * 1024;

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::HIDDEN == 0
        || !A::HIDDEN.is_multiple_of(512)
        || !A::INTERMEDIATE.is_multiple_of(DECODE_WARPS)
        || !A::INTERMEDIATE.is_multiple_of(PREFILL_OUTPUT_ROWS)
        || !(A::HIDDEN / 4).is_multiple_of(PREFILL_K128_WORDS)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the dense-FP8 SwiGLU schedules",
        ));
    }

    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Dynamically quantizes one BF16 input row to E4M3 plus an FP32 scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_swiglu_quantize<A: Arch>(input: *const u32, codes: *mut u16, scales: *mut f32) {
        static mut WARP_MAXIMUM: SharedArray<f32, QUANTIZE_WARPS, 16> = SharedArray::UNINIT;
        let warp_maximum = core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>();

        // SAFETY: one exact launch block owns one complete model-width row.
        unsafe { quantize_activation::<A>(input, codes, scales, warp_maximum) };
    }

    /// Reuses each streamed gate/up row across one exact decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_swiglu_decode<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // The rejected padded-MMA route measured 212.992/219.104/225.280 us
        // at B=2/4/8 versus 133.120/133.120/139.264 us here. Eight warps let
        // each warp stream one gate/up row and reuse it across exactly the live
        // rows; B=1 instead keeps four independent accumulator chains. Every
        // lane retains its ten 512-value phases and reduction order.
        // SAFETY: the exact grid assigns one unique output row to each warp.
        unsafe {
            if TOKENS == 1 {
                fp8_swiglu_decode_b1::<A, DECODE_WARPS>(
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    output,
                );
            } else {
                fp8_swiglu_decode_body::<A, TOKENS, DECODE_WARPS>(
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    output,
                );
            }
        };
    }

    /// Applies the retained K=128 tensor-core tile at exactly 32 rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 49152,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_swiglu_mma_t32(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // K=128 reduced the retained B=32 cold route from 200.704 to
        // 173.056 us by halving barriers. The 64x64 tile preserves the same
        // m16n8k32 K order and scaled SwiGLU epilogue for every output.
        // SAFETY: the prepared grid and 48 KiB shared arena cover every tile;
        // the launch contract pads `activation_codes` to the 64-row tile.
        unsafe {
            fp8_swiglu_mma_body::<Qwen38_27B, 32, 64, 32, 4>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
            )
        };
    }

    /// Applies the retained K=128 tensor-core tile at exactly 64 rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 49152,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_swiglu_mma_t64(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // K=128 reduced the retained B=64 cold route from 202.752 to
        // 174.080 us by halving barriers. The 64x64 tile preserves the same
        // m16n8k32 K order and scaled SwiGLU epilogue for every output.
        // SAFETY: the prepared grid and 48 KiB shared arena cover every tile.
        unsafe {
            fp8_swiglu_mma_body::<Qwen38_27B, 64, 64, 32, 4>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
            )
        };
    }

    /// Applies the retained K=64 tensor-core tile at exactly 128 rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 24576,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_swiglu_mma_t128(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // Two 64-row tiles made K=128's 48 KiB footprint regress to 333.408
        // us; K=64 measured 307.168 us and the retained hybrid 286.720 us.
        // Only barrier grouping changes: m16n8k32 K order and epilogue remain.
        // SAFETY: the prepared grid and 24 KiB shared arena cover every tile.
        unsafe {
            fp8_swiglu_mma_body::<Qwen38_27B, 128, 64, 16, 2>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                k_tiles,
            )
        };
    }
}

fn prepare_quantize<A: Arch, const TOKENS: usize>(
    module: &kernels::LoadedModule,
) -> GpuResult<PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>> {
    module
        .prepare_fp8_swiglu_quantize::<A>(LaunchConfig1D::new(TOKENS as u32, QUANTIZE_THREADS, 0))
        .map_err(|source| GpuError::launch("preparing dense-FP8 activation quantization", source))
}

struct PreparedDecodeRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
    projection: PreparedLaunch<kernels::__fp8_swiglu_decode_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedDecodeRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(A::INTERMEDIATE / DECODE_WARPS)
            .map_err(|_| GpuError::invalid_launch("dense-FP8 SwiGLU rows exceed grid width"))?;
        let projection = module
            .prepare_fp8_swiglu_decode::<A, TOKENS>(LaunchConfig1D::new(blocks, DECODE_THREADS, 0))
            .map_err(|source| GpuError::launch("preparing dense-FP8 SwiGLU decode", source))?;

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
            .fp8_swiglu_quantize::<A>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 quantization", source))?;
        module
            .fp8_swiglu_decode::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 SwiGLU decode", source))
    }
}

macro_rules! define_prefill_route {
    ($name:ident, $tokens:literal, $kernel:ty, $prepare:ident, $block_multiplier:literal, $shared:expr, $launch:ident, $k_words:expr) => {
        struct $name<A: Arch> {
            quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
            projection: PreparedLaunch<$kernel>,
        }

        impl<A: Arch> $name<A> {
            fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
                let prefill_blocks =
                    u32::try_from(A::INTERMEDIATE / PREFILL_OUTPUT_ROWS).map_err(|_| {
                        GpuError::invalid_launch("dense-FP8 SwiGLU rows exceed grid width")
                    })?;

                Ok(Self {
                    quantize: prepare_quantize::<A, $tokens>(module)?,
                    projection: module
                        .$prepare(LaunchConfig1D::new(
                            $block_multiplier * prefill_blocks,
                            PREFILL_THREADS,
                            $shared,
                        ))
                        .map_err(|source| {
                            GpuError::launch(
                                concat!("preparing dense-FP8 SwiGLU T=", stringify!($tokens)),
                                source,
                            )
                        })?,
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
                    .fp8_swiglu_quantize::<A>(
                        stream,
                        &self.quantize,
                        input.cast::<u32>(),
                        activation_codes.cast::<u16>(),
                        activation_scales,
                    )
                    .map_err(|source| {
                        GpuError::launch("launching dense-FP8 quantization", source)
                    })?;
                module
                    .$launch(
                        stream,
                        &self.projection,
                        activation_codes.cast::<u32>(),
                        activation_scales,
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        output,
                        (A::HIDDEN / 4 / $k_words) as u32,
                    )
                    .map_err(|source| {
                        GpuError::launch("launching dense-FP8 SwiGLU prefill", source)
                    })
            }
        }
    };
}

define_prefill_route!(
    PreparedT32Route,
    32,
    kernels::__fp8_swiglu_mma_t32_CudaKernel,
    prepare_fp8_swiglu_mma_t32,
    1,
    PREFILL_K128_SHARED_BYTES,
    fp8_swiglu_mma_t32,
    PREFILL_K128_WORDS
);
define_prefill_route!(
    PreparedT64Route,
    64,
    kernels::__fp8_swiglu_mma_t64_CudaKernel,
    prepare_fp8_swiglu_mma_t64,
    1,
    PREFILL_K128_SHARED_BYTES,
    fp8_swiglu_mma_t64,
    PREFILL_K128_WORDS
);
define_prefill_route!(
    PreparedT128Route,
    128,
    kernels::__fp8_swiglu_mma_t128_CudaKernel,
    prepare_fp8_swiglu_mma_t128,
    2,
    PREFILL_K64_SHARED_BYTES,
    fp8_swiglu_mma_t128,
    PREFILL_K64_WORDS
);

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_fp8_swiglu),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct DenseFp8SwiGluRoutes<A: Arch> {
    #[route(1)]
    b1: PreparedDecodeRoute<A, 1>,
    #[route(2)]
    b2: PreparedDecodeRoute<A, 2>,
    #[route(3)]
    b3: PreparedDecodeRoute<A, 3>,
    #[route(4)]
    b4: PreparedDecodeRoute<A, 4>,
    #[route(5)]
    b5: PreparedDecodeRoute<A, 5>,
    #[route(6)]
    b6: PreparedDecodeRoute<A, 6>,
    #[route(7)]
    b7: PreparedDecodeRoute<A, 7>,
    #[route(8)]
    b8: PreparedDecodeRoute<A, 8>,
    #[route(32)]
    t32: PreparedT32Route<A>,
    #[route(64)]
    t64: PreparedT64Route<A>,
    #[route(128)]
    t128: PreparedT128Route<A>,
}

/// Prepared dense-FP8 gate/up plus SwiGLU routes for exact decode and prefill rows.
pub struct DenseFp8SwiGluOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: DenseFp8SwiGluRoutes<A>,
    t1024_quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
    t1024: DenseFp8SwiGluTmaRoute,
}

impl<A: Sm120Arch> DenseFp8SwiGluOp<A> {
    /// Loads the embedded SM120 module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = fp8_swiglu_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the dense-FP8 SwiGLU module", source))?;

        Ok(Self {
            routes: DenseFp8SwiGluRoutes::prepare(&module)?,
            t1024_quantize: prepare_quantize::<A, MACRO_TOKENS>(&module)?,
            t1024: DenseFp8SwiGluTmaRoute::new(context)?,
            module,
        })
    }

    /// Dynamically quantizes and applies the source-native gate/up projection.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * A::HIDDEN` values. `activation_codes` covers
    /// at least 64 rows for T=32 and otherwise `rows`, so the retained padded
    /// CTA can read its complete immutable tile; `activation_scales` covers
    /// `rows`; weights cover `[2 * A::INTERMEDIATE, A::HIDDEN]` E4M3 codes
    /// plus one BF16 scale per source row; and `output` covers
    /// `[rows, A::INTERMEDIATE]` BF16 values. Four-byte-loaded planes are
    /// four-byte aligned. All allocations belong to `stream`'s context,
    /// remain live through completion, and do not overlap.
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
        dispatch_fp8_swiglu!(
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
            else => Err(GpuError::invalid_launch(format!(
                "dense-FP8 SwiGLU row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128"
            )))
        )
    }

    /// Dynamically quantizes and applies the exact T=1024 TMA route.
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
        maps: &DenseFp8SwiGluTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 SwiGLU tensor maps do not match the launch addresses",
            ));
        }
        self.module
            .fp8_swiglu_quantize::<A>(
                stream,
                &self.t1024_quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024
                .launch(stream, maps, activation_scales, weight_scales, output)
        }
    }
    /// Dynamically quantizes and applies the T=128 tile of the TMA SwiGLU route.
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
        maps: &DenseFp8SwiGluTmaMaps,
    ) -> GpuResult<()> {
        if maps.activation_codes() != activation_codes.addr()
            || maps.weight_codes() != weight_codes.addr()
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 SwiGLU tensor maps do not match the launch addresses",
            ));
        }
        self.module
            .fp8_swiglu_quantize::<A>(
                stream,
                &self.routes.t128.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 quantization", source))?;
        // SAFETY: the public method admits every pointer and map boundary.
        unsafe {
            self.t1024
                .launch_t128(stream, maps, activation_scales, weight_scales, output)
        }
    }
}

/// PTX symbols retained for quantization and every exact dense-FP8 SwiGLU route.
pub(crate) fn fp8_swiglu_ptx_names() -> [&'static str; 13] {
    [
        kernels::fp8_swiglu_quantize_ptx_name::<Qwen38_27B>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 1>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 2>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 3>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 4>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 5>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 6>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 7>(),
        kernels::fp8_swiglu_decode_ptx_name::<Qwen38_27B, 8>(),
        "fp8_swiglu_mma_t32",
        "fp8_swiglu_mma_t64",
        "fp8_swiglu_mma_t128",
        tma_ptx_name(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        DECODE_THREADS, DECODE_WARPS, DenseFp8SwiGluRoutes, PREFILL_K64_SHARED_BYTES,
        PREFILL_K128_SHARED_BYTES, fp8_swiglu_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn route_table_covers_only_the_admitted_shapes() {
        assert_eq!(
            DenseFp8SwiGluRoutes::<Qwen38_27B>::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]
        );
    }

    #[test]
    fn exact_geometry_matches_the_retained_schedules() {
        assert_eq!(DECODE_THREADS, 256);
        assert_eq!(DECODE_WARPS, 8);
        assert_eq!(Qwen38_27B::HIDDEN / (32 * 16), 10);
        assert_eq!(Qwen38_27B::INTERMEDIATE % DECODE_WARPS, 0);
        assert_eq!(Qwen38_27B::INTERMEDIATE % 64, 0);
        assert_eq!(PREFILL_K128_SHARED_BYTES, 49_152);
        assert_eq!(PREFILL_K64_SHARED_BYTES, 24_576);
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_exact_route() {
        let names = fp8_swiglu_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 13);
        assert_eq!(unique.len(), names.len());
    }
}

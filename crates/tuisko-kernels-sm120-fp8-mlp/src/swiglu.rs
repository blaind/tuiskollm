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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteClass {
    Decode,
    PrefillK128,
    PrefillK64,
}

fn route_class(rows: usize) -> Option<RouteClass> {
    match rows {
        1..=MAX_BATCH => Some(RouteClass::Decode),
        32 | 64 => Some(RouteClass::PrefillK128),
        128 => Some(RouteClass::PrefillK64),
        _ => None,
    }
}

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

/// Prepared dense-FP8 gate/up plus SwiGLU routes for exact decode and prefill rows.
pub struct DenseFp8SwiGluOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedDecodeRoute<A, 1>,
    b2: PreparedDecodeRoute<A, 2>,
    b3: PreparedDecodeRoute<A, 3>,
    b4: PreparedDecodeRoute<A, 4>,
    b5: PreparedDecodeRoute<A, 5>,
    b6: PreparedDecodeRoute<A, 6>,
    b7: PreparedDecodeRoute<A, 7>,
    b8: PreparedDecodeRoute<A, 8>,
    t32_quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
    t32: PreparedLaunch<kernels::__fp8_swiglu_mma_t32_CudaKernel>,
    t64_quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
    t64: PreparedLaunch<kernels::__fp8_swiglu_mma_t64_CudaKernel>,
    t128_quantize: PreparedLaunch<kernels::__fp8_swiglu_quantize_CudaKernel<A>>,
    t128: PreparedLaunch<kernels::__fp8_swiglu_mma_t128_CudaKernel>,
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
        let prefill_blocks = u32::try_from(A::INTERMEDIATE / PREFILL_OUTPUT_ROWS)
            .map_err(|_| GpuError::invalid_launch("dense-FP8 SwiGLU rows exceed grid width"))?;

        Ok(Self {
            b1: PreparedDecodeRoute::prepare(&module)?,
            b2: PreparedDecodeRoute::prepare(&module)?,
            b3: PreparedDecodeRoute::prepare(&module)?,
            b4: PreparedDecodeRoute::prepare(&module)?,
            b5: PreparedDecodeRoute::prepare(&module)?,
            b6: PreparedDecodeRoute::prepare(&module)?,
            b7: PreparedDecodeRoute::prepare(&module)?,
            b8: PreparedDecodeRoute::prepare(&module)?,
            t32_quantize: prepare_quantize::<A, 32>(&module)?,
            t32: module
                .prepare_fp8_swiglu_mma_t32(LaunchConfig1D::new(
                    prefill_blocks,
                    PREFILL_THREADS,
                    PREFILL_K128_SHARED_BYTES,
                ))
                .map_err(|source| GpuError::launch("preparing dense-FP8 SwiGLU T=32", source))?,
            t64_quantize: prepare_quantize::<A, 64>(&module)?,
            t64: module
                .prepare_fp8_swiglu_mma_t64(LaunchConfig1D::new(
                    prefill_blocks,
                    PREFILL_THREADS,
                    PREFILL_K128_SHARED_BYTES,
                ))
                .map_err(|source| GpuError::launch("preparing dense-FP8 SwiGLU T=64", source))?,
            t128_quantize: prepare_quantize::<A, 128>(&module)?,
            t128: module
                .prepare_fp8_swiglu_mma_t128(LaunchConfig1D::new(
                    2 * prefill_blocks,
                    PREFILL_THREADS,
                    PREFILL_K64_SHARED_BYTES,
                ))
                .map_err(|source| GpuError::launch("preparing dense-FP8 SwiGLU T=128", source))?,
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
        macro_rules! decode {
            ($route:ident) => {
                // SAFETY: exact route dispatch preserves the public pointer contract.
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
        macro_rules! prefill {
            ($quantize:ident, $route:ident, $method:ident, $k_words:expr) => {{
                self.module
                    .fp8_swiglu_quantize::<A>(
                        stream,
                        &self.$quantize,
                        input.cast::<u32>(),
                        activation_codes.cast::<u16>(),
                        activation_scales,
                    )
                    .map_err(|source| {
                        GpuError::launch("launching dense-FP8 quantization", source)
                    })?;
                self.module
                    .$method(
                        stream,
                        &self.$route,
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
            }};
        }

        let Some(class) = route_class(rows) else {
            return Err(GpuError::invalid_launch(format!(
                "dense-FP8 SwiGLU row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128"
            )));
        };

        match class {
            RouteClass::Decode => match rows {
                1 => decode!(b1),
                2 => decode!(b2),
                3 => decode!(b3),
                4 => decode!(b4),
                5 => decode!(b5),
                6 => decode!(b6),
                7 => decode!(b7),
                8 => decode!(b8),
                _ => unreachable!(),
            },
            RouteClass::PrefillK128 => match rows {
                32 => prefill!(t32_quantize, t32, fp8_swiglu_mma_t32, PREFILL_K128_WORDS),
                64 => prefill!(t64_quantize, t64, fp8_swiglu_mma_t64, PREFILL_K128_WORDS),
                _ => unreachable!(),
            },
            RouteClass::PrefillK64 => {
                prefill!(t128_quantize, t128, fp8_swiglu_mma_t128, PREFILL_K64_WORDS)
            }
        }
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
        DECODE_THREADS, DECODE_WARPS, PREFILL_K64_SHARED_BYTES, PREFILL_K128_SHARED_BYTES,
        RouteClass, fp8_swiglu_ptx_names, route_class,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn route_table_covers_only_the_admitted_shapes() {
        let cases = [
            (0, None),
            (1, Some(RouteClass::Decode)),
            (8, Some(RouteClass::Decode)),
            (9, None),
            (16, None),
            (31, None),
            (32, Some(RouteClass::PrefillK128)),
            (64, Some(RouteClass::PrefillK128)),
            (127, None),
            (128, Some(RouteClass::PrefillK64)),
            (129, None),
        ];

        for (rows, expected) in cases {
            assert_eq!(route_class(rows), expected, "rows={rows}");
        }
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

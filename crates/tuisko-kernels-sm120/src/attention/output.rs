//! Gated activation, dynamic E4M3 quantization, and source-native output projection.

use crate::Sm120Arch;
use crate::device::attention_output::{attention_gate_quantize, attention_output_projection_mma};
use crate::device::fp8_projection::fp8_projection;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];
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
    if A::HIDDEN != 5_120
        || A::ATTENTION_OUTPUT_COLUMNS != 6_144
        || A::ATTENTION_QUERY_ROWS != 12_288
        || A::ATTENTION_QKV_ROWS != 14_336
        || !A::ATTENTION_OUTPUT_COLUMNS.is_multiple_of(512)
        || !A::HIDDEN.is_multiple_of(2 * WARPS)
        || !A::HIDDEN.is_multiple_of(PREFILL_OUTPUT_ROWS)
        || !(A::ATTENTION_OUTPUT_COLUMNS / 4).is_multiple_of(PREFILL_K_WORDS)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the FP8 attention-output schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Applies `sigmoid(gate)`, publishes the gated FP32 seam, and quantizes it.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared = 0, min_compute_capability = (12, 0))]
    pub fn attention_gate_quantize_exact<A: Arch>(
        attention: *mut f32,
        qkv: *const u16,
        codes: *mut u16,
        scales: *mut f32,
    ) {
        static mut WARP_MAXIMUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        // Eight warps cover one 6,144-wide row in 24 iterations. This is the
        // retained exact decode topology; one CTA owns one token and scale.
        unsafe {
            attention_gate_quantize::<A>(
                attention,
                qkv,
                codes,
                scales,
                core::ptr::addr_of_mut!(WARP_MAXIMUM).cast::<f32>(),
            );
        }
    }

    /// Projects one exact batch through the source-native attention output matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared = 0, min_compute_capability = (12, 0))]
    pub fn attention_output_projection<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // Pairing sixteen output rows per CTA gives 320 CTAs and preserves the
        // qualified 512-value FP8 accumulation phases of the shared projection.
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

    /// Projects one exact 32/64/128-row prefill tile with native E4M3 MMA.
    #[kernel]
    #[launch_bounds(64, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (64, 1, 1),
        dynamic_shared = 16384,
        min_compute_capability = (12, 0),
    )]
    pub fn attention_output_projection_mma_exact<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // A 32x32 tile assigns two warps to two 16-row token tiles. Its 16 KiB
        // double buffer holds exactly 32 activation and 32 weight rows at K=128.
        // SAFETY: prepared routes instantiate only exact T=32/64/128 grids.
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

    /// Projects exactly 1,024 rows with native E4M3 MMA.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 24576,
        min_compute_capability = (12, 0),
    )]
    pub fn attention_output_projection_mma_t1024(
        activation_codes: *const u32,
        activation_scales: *const f32,
        weight_codes: *const u32,
        weight_scales: *const u16,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // A 64x32 tile doubles activation reuse while its four warps still own
        // exact 16x32 MMA quadrants. The K=128 double buffer is exactly 24 KiB.
        // SAFETY: the fixed grid covers all sixteen 64-token macro tiles.
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

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__attention_gate_quantize_exact_CudaKernel<A>>,
    projection: PreparedLaunch<kernels::__attention_output_projection_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            quantize: module
                .prepare_attention_gate_quantize_exact::<A>(LaunchConfig1D::new(
                    TOKENS as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output gate quantization", source)
                })?,
            projection: module
                .prepare_attention_output_projection::<A, TOKENS>(LaunchConfig1D::new(
                    (A::HIDDEN / (2 * WARPS)) as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output projection", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .attention_gate_quantize_exact::<A>(
                stream,
                &self.quantize,
                attention,
                qkv,
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| {
                GpuError::launch("launching attention-output gate quantization", source)
            })?;
        module
            .attention_output_projection::<A, TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching attention-output projection", source))
    }
}

struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__attention_gate_quantize_exact_CudaKernel<A>>,
    projection: PreparedLaunch<kernels::__attention_output_projection_mma_exact_CudaKernel<TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS[..3].contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "attention-output prefill route T={TOKENS} is not admitted"
            )));
        }
        let token_tiles = TOKENS / PREFILL_BLOCK_ROWS;
        let projection_blocks = A::HIDDEN / PREFILL_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks).map_err(|_| {
            GpuError::invalid_launch("attention-output prefill grid exceeds CUDA width")
        })?;

        Ok(Self {
            quantize: module
                .prepare_attention_gate_quantize_exact::<A>(LaunchConfig1D::new(
                    TOKENS as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output gate quantization", source)
                })?,
            projection: module
                .prepare_attention_output_projection_mma_exact::<TOKENS>(LaunchConfig1D::new(
                    projection_blocks,
                    PREFILL_THREADS,
                    PREFILL_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output prefill projection", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .attention_gate_quantize_exact::<A>(
                stream,
                &self.quantize,
                attention,
                qkv,
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| {
                GpuError::launch("launching attention-output gate quantization", source)
            })?;
        module
            .attention_output_projection_mma_exact::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                (A::ATTENTION_OUTPUT_COLUMNS / 4 / PREFILL_K_WORDS) as u32,
            )
            .map_err(|source| {
                GpuError::launch("launching attention-output prefill projection", source)
            })
    }
}

struct PreparedMacroRoute<A: Arch> {
    quantize: PreparedLaunch<kernels::__attention_gate_quantize_exact_CudaKernel<A>>,
    projection: PreparedLaunch<kernels::__attention_output_projection_mma_t1024_CudaKernel>,
}

impl<A: Arch> PreparedMacroRoute<A> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = PREFILL_TOKENS[3] / MACRO_BLOCK_ROWS;
        let projection_blocks = A::HIDDEN / PREFILL_OUTPUT_ROWS * token_tiles;
        let projection_blocks = u32::try_from(projection_blocks).map_err(|_| {
            GpuError::invalid_launch("attention-output macro-prefill grid exceeds CUDA width")
        })?;

        Ok(Self {
            quantize: module
                .prepare_attention_gate_quantize_exact::<A>(LaunchConfig1D::new(
                    PREFILL_TOKENS[3] as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output macro gate quantization", source)
                })?,
            projection: module
                .prepare_attention_output_projection_mma_t1024(LaunchConfig1D::new(
                    projection_blocks,
                    MACRO_THREADS,
                    MACRO_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention-output macro projection", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .attention_gate_quantize_exact::<A>(
                stream,
                &self.quantize,
                attention,
                qkv,
                activation_codes.cast::<u16>(),
                activation_scales,
            )
            .map_err(|source| {
                GpuError::launch("launching attention-output macro gate quantization", source)
            })?;
        module
            .attention_output_projection_mma_t1024(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                (A::ATTENTION_OUTPUT_COLUMNS / 4 / PREFILL_K_WORDS) as u32,
            )
            .map_err(|source| {
                GpuError::launch("launching attention-output macro projection", source)
            })
    }
}

/// Prepared gated FP8 attention-output routes for exact decode and prefill rows.
pub struct AttentionOutputOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedRoute<A, 1>,
    b2: PreparedRoute<A, 2>,
    b3: PreparedRoute<A, 3>,
    b4: PreparedRoute<A, 4>,
    b5: PreparedRoute<A, 5>,
    b6: PreparedRoute<A, 6>,
    b7: PreparedRoute<A, 7>,
    b8: PreparedRoute<A, 8>,
    t32: PreparedPrefillRoute<A, 32>,
    t64: PreparedPrefillRoute<A, 64>,
    t128: PreparedPrefillRoute<A, 128>,
    t1024: PreparedMacroRoute<A>,
}

impl<A: Sm120Arch> AttentionOutputOp<A> {
    /// Loads the embedded module and prepares every exact decode and prefill route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = attention_output_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading attention output", source))?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: PreparedMacroRoute::prepare(&module)?,
            module,
        })
    }

    /// Gates the paged-attention output, dynamically quantizes, and projects it.
    ///
    /// # Safety
    ///
    /// `attention` covers `[rows, A::ATTENTION_OUTPUT_COLUMNS]` FP32 values and
    /// is mutable scratch; the gated FP32 seam is published in place. `qkv`
    /// covers `[rows, A::ATTENTION_QKV_ROWS]` BF16 values. Activation scratch
    /// covers the output columns plus one FP32 scale per token. Source weights
    /// cover `[A::HIDDEN, A::ATTENTION_OUTPUT_COLUMNS]` E4M3 values plus one
    /// BF16 scale per row. Output covers `[rows, A::HIDDEN]` BF16 values. All
    /// regions are aligned, non-overlapping, context-local, and live through
    /// completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        let Some(class) = route_class(rows) else {
            return Err(GpuError::invalid_launch(format!(
                "attention output row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024"
            )));
        };

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        attention,
                        qkv,
                        activation_codes,
                        activation_scales,
                        weight_codes,
                        weight_scales,
                        output,
                    )
                }
            };
        }

        match class {
            RouteClass::Decode => match rows {
                1 => launch!(b1),
                2 => launch!(b2),
                3 => launch!(b3),
                4 => launch!(b4),
                5 => launch!(b5),
                6 => launch!(b6),
                7 => launch!(b7),
                8 => launch!(b8),
                _ => unreachable!(),
            },
            RouteClass::Prefill => match rows {
                32 => launch!(t32),
                64 => launch!(t64),
                128 => launch!(t128),
                _ => unreachable!(),
            },
            RouteClass::Macro => launch!(t1024),
        }
    }
}

/// PTX symbols retained for gated quantization and every exact projection route.
pub(crate) fn attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::attention_gate_quantize_exact_ptx_name::<Qwen38_27B>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 1>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 2>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 3>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 4>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 5>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 6>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 7>(),
        kernels::attention_output_projection_ptx_name::<Qwen38_27B, 8>(),
        kernels::attention_output_projection_mma_exact_ptx_name::<32>(),
        kernels::attention_output_projection_mma_exact_ptx_name::<64>(),
        kernels::attention_output_projection_mma_exact_ptx_name::<128>(),
        "attention_output_projection_mma_t1024",
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        MACRO_SHARED_BYTES, MACRO_THREADS, PREFILL_SHARED_BYTES, PREFILL_THREADS, RouteClass,
        THREADS, attention_output_ptx_names, route_class,
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
        let names = attention_output_ptx_names();
        assert_eq!(names.len(), 13);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 13);
    }
}

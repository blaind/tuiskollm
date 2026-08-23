//! Gated activation, dynamic E4M3 quantization, and source-native output projection.

use crate::Sm120Arch;
use crate::device::attention_output::attention_gate_quantize;
use crate::device::fp8_projection::fp8_projection;
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
    if A::HIDDEN != 5_120
        || A::ATTENTION_OUTPUT_COLUMNS != 6_144
        || A::ATTENTION_QUERY_ROWS != 12_288
        || A::ATTENTION_QKV_ROWS != 14_336
        || !A::ATTENTION_OUTPUT_COLUMNS.is_multiple_of(512)
        || !A::HIDDEN.is_multiple_of(2 * WARPS)
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

/// Prepared gated FP8 attention-output routes for exact `B=1..=8`.
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
}

impl<A: Sm120Arch> AttentionOutputOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
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
            module,
        })
    }

    /// Gates the paged-attention output, dynamically quantizes, and projects it.
    ///
    /// # Safety
    ///
    /// `attention` covers `[batch, A::ATTENTION_OUTPUT_COLUMNS]` FP32 values and
    /// is mutable scratch; the gated FP32 seam is published in place. `qkv`
    /// covers `[batch, A::ATTENTION_QKV_ROWS]` BF16 values. Activation scratch
    /// covers the output columns plus one FP32 scale per token. Source weights
    /// cover `[A::HIDDEN, A::ATTENTION_OUTPUT_COLUMNS]` E4M3 values plus one
    /// BF16 scale per row. Output covers `[batch, A::HIDDEN]` BF16 values. All
    /// regions are aligned, non-overlapping, context-local, and live through
    /// completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut f32,
        weight_codes: *const u8,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "attention output batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }

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
    ]
}

#[cfg(test)]
mod tests {
    use super::{THREADS, admitted_batch, attention_output_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn route_and_inventory_are_exact() {
        for (batch, expected) in [(0, false), (1, true), (4, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
        assert_eq!(THREADS, 256);
        let names = attention_output_ptx_names();
        assert_eq!(names.len(), 9);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 9);
    }
}

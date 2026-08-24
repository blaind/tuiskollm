//! Source-BF16 gated attention-output projection for the Qwen3.8 MTP layer.

use crate::device::attention_output::attention_gate_bf16;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
const OUTPUT_ROWS: usize = Qwen38_27B::HIDDEN;
const OUTPUT_TILES: usize = OUTPUT_ROWS / 8;
// Four warps publish 32 adjacent output rows per CTA. The 160-CTA grid is
// almost one CTA per target SM while every warp retains one exact MMA tile.
const WARPS: usize = 4;
const PROJECTION_THREADS: u32 = (WARPS * 32) as u32;
const PROJECTION_BLOCKS: u32 = (OUTPUT_TILES / WARPS) as u32;
const GATE_THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(input: *const u32, row: usize, column: usize) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete gated BF16 rows.
        unsafe { *input.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: source weights are the exact `[OUTPUT_ROWS, INPUT_COLUMNS]` BF16 matrix.
        unsafe { *weight.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn projection_body<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < INPUT_COLUMNS {
            // Native m16n8k16 BF16 MMA keeps the source matrix represented;
            // B<=8 occupies only lower rows and no padded output is published.
            let activation = unsafe {
                [
                    input_pair::<TOKENS>(input, group, column + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group + 8, column + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                    input_pair::<TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one output pair.
            unsafe {
                *output.add(group * (OUTPUT_ROWS / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Publishes the exact gated FP32 and represented-BF16 MTP seams.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_attention_gate<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 6,144-wide row. Its 256 threads make exactly 24
        // coalesced passes and preserve the query-head gate-column mapping.
        unsafe { attention_gate_bf16::<Qwen38_27B>(attention, qkv, activation) }
    }

    /// Projects exact gated MTP rows through the source-BF16 output matrix.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_attention_output<const TOKENS: usize>(
        activation: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers all 640 eight-row output tiles once.
        unsafe { projection_body::<TOKENS>(activation, weight, output) }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    gate: PreparedLaunch<kernels::__mtp_bf16_attention_gate_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__mtp_bf16_attention_output_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let gate = module
            .prepare_mtp_bf16_attention_gate::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                GATE_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 attention gate", source))?;
        let projection = module
            .prepare_mtp_bf16_attention_output::<TOKENS>(LaunchConfig1D::new(
                PROJECTION_BLOCKS,
                PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the MTP BF16 attention-output projection", source)
            })?;

        Ok(Self { gate, projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_attention_gate::<TOKENS>(stream, &self.gate, attention, qkv, activation)
            .map_err(|source| GpuError::launch("launching the MTP BF16 attention gate", source))?;
        module
            .mtp_bf16_attention_output::<TOKENS>(
                stream,
                &self.projection,
                activation.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the MTP BF16 attention-output projection", source)
            })
    }
}

/// Stable PTX inventory for every exact MTP attention-output stage.
pub(crate) fn mtp_bf16_attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::mtp_bf16_attention_gate_ptx_name::<1>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<2>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<3>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<4>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<5>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<6>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<7>(),
        kernels::mtp_bf16_attention_gate_ptx_name::<8>(),
        kernels::mtp_bf16_attention_output_ptx_name::<1>(),
        kernels::mtp_bf16_attention_output_ptx_name::<2>(),
        kernels::mtp_bf16_attention_output_ptx_name::<3>(),
        kernels::mtp_bf16_attention_output_ptx_name::<4>(),
        kernels::mtp_bf16_attention_output_ptx_name::<5>(),
        kernels::mtp_bf16_attention_output_ptx_name::<6>(),
        kernels::mtp_bf16_attention_output_ptx_name::<7>(),
        kernels::mtp_bf16_attention_output_ptx_name::<8>(),
    ]
}

/// Prepared gated source-BF16 attention-output routes for exact MTP `B=1..=8`.
pub struct MtpBf16AttentionOutputOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
}

impl MtpBf16AttentionOutputOp {
    /// Loads the embedded module and prepares every exact MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if INPUT_COLUMNS != 6_144
            || OUTPUT_ROWS != 5_120
            || !INPUT_COLUMNS.is_multiple_of(16)
            || !OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP attention-output geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = mtp_bf16_attention_output_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading MTP BF16 attention output", source))?;

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

    /// Gates attention into FP32/BF16 seams and applies the source-BF16 projection.
    ///
    /// # Safety
    ///
    /// `attention` covers `[batch, 6144]` FP32 values and is mutable scratch;
    /// `qkv` covers `[batch, 14336]` BF16 values; `activation` covers
    /// `[batch, 6144]` BF16 values; `weight` covers `[5120, 6144]` unchanged
    /// source BF16 values; and `output` covers `[batch, 5120]` BF16 values.
    /// Four-byte-loaded regions are aligned, non-overlapping, context-local,
    /// and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        attention,
                        qkv,
                        activation,
                        weight,
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
                "MTP BF16 attention-output batch {batch} is outside exact B=1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GATE_THREADS, INPUT_COLUMNS, MAX_BATCH, OUTPUT_ROWS, OUTPUT_TILES, PROJECTION_BLOCKS,
        PROJECTION_THREADS, WARPS, mtp_bf16_attention_output_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn exact_geometry_and_batch_inventory_are_complete() {
        assert_eq!(INPUT_COLUMNS, 6_144);
        assert_eq!(OUTPUT_ROWS, 5_120);
        assert_eq!(OUTPUT_TILES, 640);
        assert_eq!(WARPS, 4);
        assert_eq!(PROJECTION_THREADS, 128);
        assert_eq!(PROJECTION_BLOCKS, 160);
        assert_eq!(GATE_THREADS, 256);

        let names = mtp_bf16_attention_output_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

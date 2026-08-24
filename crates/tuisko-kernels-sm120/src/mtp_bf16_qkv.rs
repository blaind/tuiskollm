//! Source-BF16 Q/gate/K/V projection for the Qwen3.8 MTP layer.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
const OUTPUT_TILES: usize = OUTPUT_ROWS / 8;
// Eight warps publish 64 adjacent fused rows per CTA. The resulting 224 CTAs
// expose 1.32 waves on the 170-SM target while each CTA owns only its MMA state.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const BLOCKS: u32 = (OUTPUT_TILES / WARPS) as u32;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(input: *const u32, row: usize, column: usize) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete 5,120-wide BF16 rows.
        unsafe { *input.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the gathered source-BF16 plane is `[OUTPUT_ROWS, INPUT_COLUMNS]`.
        unsafe { *weight.add(row * (INPUT_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn qkv_body<const TOKENS: usize>(
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
            // m16n8k16 is the smallest native BF16 Tensor Core tile. B<=8 uses
            // only the lower token rows, and no padded row is ever published.
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
            // SAFETY: the lower fragment maps to one active token and one fused-row pair.
            unsafe {
                *output.add(group * (OUTPUT_ROWS / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Projects exact MTP decode rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers every 8-row fused-output tile once.
        unsafe { qkv_body::<TOKENS>(input, weight, output) }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(BLOCKS, THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 QKV projection", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the MTP BF16 QKV projection", source))
    }
}

/// Stable PTX symbol inventory for every exact MTP QKV decode batch.
pub(crate) fn mtp_bf16_qkv_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::mtp_bf16_qkv_ptx_name::<1>(),
        kernels::mtp_bf16_qkv_ptx_name::<2>(),
        kernels::mtp_bf16_qkv_ptx_name::<3>(),
        kernels::mtp_bf16_qkv_ptx_name::<4>(),
        kernels::mtp_bf16_qkv_ptx_name::<5>(),
        kernels::mtp_bf16_qkv_ptx_name::<6>(),
        kernels::mtp_bf16_qkv_ptx_name::<7>(),
        kernels::mtp_bf16_qkv_ptx_name::<8>(),
    ]
}

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact MTP `B=1..=8`.
pub struct MtpBf16QkvOp {
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

impl MtpBf16QkvOp {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !INPUT_COLUMNS.is_multiple_of(16)
            || !OUTPUT_ROWS.is_multiple_of(64)
            || !OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = mtp_bf16_qkv_ptx_names();
        // SAFETY: this crate owns the embedded MTP QKV artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the MTP BF16 QKV module", source))?;

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

    /// Applies the exact gathered source-BF16 Q/gate/K/V projection.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned, context-local, live through stream completion, and
    /// non-overlapping. `input` covers `batch * 5120` BF16 values, `weight` covers the gathered
    /// `[14336, 5120]` Q/gate/K/V plane, and `output` covers `batch * 14336` BF16 values.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route
                        .launch(&self.module, stream, input, weight, output)
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
                "MTP BF16 QKV batch {batch} is outside exact B=1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCKS, INPUT_COLUMNS, OUTPUT_ROWS, OUTPUT_TILES, WARPS, mtp_bf16_qkv_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn exact_geometry_covers_the_gathered_source_plane() {
        assert_eq!(INPUT_COLUMNS, 5_120);
        assert_eq!(OUTPUT_ROWS, 14_336);
        assert_eq!(OUTPUT_TILES, 1_792);
        assert_eq!(BLOCKS, 224);
        assert_eq!(BLOCKS as usize * WARPS, OUTPUT_TILES);
    }

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
        let names = mtp_bf16_qkv_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(unique.len(), names.len());
    }
}

//! Source-BF16 normalization and fusion projection for the Qwen3.8 MTP layer.

use crate::ResidualNormOp;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const FUSION_COLUMNS: usize = 2 * HIDDEN;
const OUTPUT_TILES: usize = HIDDEN / 8;
// Eight warps give one block 64 output columns. The 80-block grid reads the
// 100 MiB matrix once per route while the small normalized inputs remain L2-resident.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const BLOCKS: u32 = (OUTPUT_TILES / WARPS) as u32;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        let words = HIDDEN / 2;
        if column < HIDDEN {
            // SAFETY: the exact route owns `TOKENS` complete normalized embedding rows.
            unsafe { *normalized_embedding.add(row * words + column / 2) }
        } else {
            // SAFETY: the second half of MTP FC input is the complete normalized hidden row.
            unsafe { *normalized_hidden.add(row * words + (column - HIDDEN) / 2) }
        }
    }

    #[inline(always)]
    unsafe fn weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the source-BF16 matrix is `[HIDDEN, 2 * HIDDEN]`; every requested
        // pair is aligned and lies within one complete output row.
        unsafe { *weight.add(row * (FUSION_COLUMNS / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn projection_body<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
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

        while column < FUSION_COLUMNS {
            // m16n8k16 is the smallest native BF16 Tensor Core tile. B<=8 occupies
            // the lower token rows; zero upper rows do not publish a padded model row.
            let activation = unsafe {
                [
                    input_pair::<TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group + 8,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group,
                        column + 8 + 2 * thread_in_group,
                    ),
                    input_pair::<TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all 32 lanes execute the same m16n8k16 instruction with the
            // documented row-major A and column-major B register fragments.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_words = HIDDEN / 2;
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower accumulator half maps bijectively to one exact active
            // token row and one adjacent BF16 output pair.
            unsafe {
                *output.add(group * output_words + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Projects exact MTP decode rows through the source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_fusion<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers every 8-column output tile exactly once.
        unsafe {
            projection_body::<TOKENS>(normalized_embedding, normalized_hidden, weight, output);
        }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_fusion_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_mtp_bf16_fusion::<TOKENS>(LaunchConfig1D::new(BLOCKS, THREADS, 0))
            .map_err(|source| {
                GpuError::launch("preparing the MTP BF16 fusion projection", source)
            })?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized_embedding: *const u16,
        normalized_hidden: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_fusion::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the MTP BF16 fusion projection", source))
    }
}

/// Stable PTX symbol inventory for every exact MTP fusion decode batch.
pub(crate) fn mtp_bf16_fusion_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::mtp_bf16_fusion_ptx_name::<1>(),
        kernels::mtp_bf16_fusion_ptx_name::<2>(),
        kernels::mtp_bf16_fusion_ptx_name::<3>(),
        kernels::mtp_bf16_fusion_ptx_name::<4>(),
        kernels::mtp_bf16_fusion_ptx_name::<5>(),
        kernels::mtp_bf16_fusion_ptx_name::<6>(),
        kernels::mtp_bf16_fusion_ptx_name::<7>(),
        kernels::mtp_bf16_fusion_ptx_name::<8>(),
    ]
}

/// Prepared source-BF16 input-fusion routes for exact MTP decode `B=1..=8`.
pub struct MtpBf16FusionOp {
    norm: ResidualNormOp,
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

impl MtpBf16FusionOp {
    /// Loads both embedded modules and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !HIDDEN.is_multiple_of(16)
            || !FUSION_COLUMNS.is_multiple_of(16)
            || !OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP fusion geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = mtp_bf16_fusion_ptx_names();
        // SAFETY: this crate owns the embedded MTP fusion artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the MTP BF16 fusion module", source))?;

        Ok(Self {
            norm: ResidualNormOp::new(context)?,
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

    /// Normalizes embedding and hidden rows, then applies the exact source-BF16 projection.
    ///
    /// The checkpoint column order is normalized embedding followed by normalized hidden.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned, context-local, and live through stream
    /// completion. Inputs, normalized workspaces, weights, and output must not overlap. Input and
    /// workspace planes cover `batch * 5120` BF16 values, norm weights cover 5120 values, the
    /// projection weight covers `[5120, 10240]`, and output covers `batch * 5120` values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        embedding: *const u16,
        hidden: *const u16,
        embedding_norm_weight: *const u16,
        hidden_norm_weight: *const u16,
        normalized_embedding: *mut u16,
        normalized_hidden: *mut u16,
        projection_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: the public pointer contract covers the two exact RMSNorm launches.
        unsafe {
            self.norm.launch_plain(
                stream,
                batch,
                embedding,
                embedding_norm_weight,
                normalized_embedding,
            )?;
            self.norm
                .launch_plain(stream, batch, hidden, hidden_norm_weight, normalized_hidden)?;
        }

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: dispatch preserves the public pointer and exact-batch contracts.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        normalized_embedding,
                        normalized_hidden,
                        projection_weight,
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
                "MTP BF16 fusion batch {batch} is outside exact B=1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCKS, FUSION_COLUMNS, HIDDEN, OUTPUT_TILES, WARPS, mtp_bf16_fusion_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn exact_geometry_covers_the_source_matrix() {
        assert_eq!(HIDDEN, 5_120);
        assert_eq!(FUSION_COLUMNS, 10_240);
        assert_eq!(OUTPUT_TILES, 640);
        assert_eq!(BLOCKS, 80);
        assert_eq!(BLOCKS as usize * WARPS, OUTPUT_TILES);
    }

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
        let names = mtp_bf16_fusion_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(unique.len(), names.len());
    }
}

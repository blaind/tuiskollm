//! Source-BF16 normalization and fusion projection for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_norm::{Qwen35ResidualNormOp, Qwen36ResidualNormOp, ResidualNormOp};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const QWEN35_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const FUSION_COLUMNS: usize = 2 * HIDDEN;
const OUTPUT_TILES: usize = HIDDEN / 8;
const QWEN35_HIDDEN: usize = Qwen35_9B::HIDDEN;
const QWEN35_FUSION_COLUMNS: usize = 2 * QWEN35_HIDDEN;
const QWEN35_OUTPUT_TILES: usize = QWEN35_HIDDEN / 8;
const QWEN36_HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const QWEN36_FUSION_COLUMNS: usize = 2 * QWEN36_HIDDEN;
const QWEN36_OUTPUT_TILES: usize = QWEN36_HIDDEN / 8;
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
    unsafe fn input_pair<A: Arch, const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        let words = A::HIDDEN / 2;
        if column < A::HIDDEN {
            // SAFETY: the exact route owns `TOKENS` complete normalized embedding rows.
            unsafe { *normalized_embedding.add(row * words + column / 2) }
        } else {
            // SAFETY: the second half of MTP FC input is the complete normalized hidden row.
            unsafe { *normalized_hidden.add(row * words + (column - A::HIDDEN) / 2) }
        }
    }

    #[inline(always)]
    unsafe fn weight_pair<A: Arch>(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the source-BF16 matrix is `[A::HIDDEN, 2 * A::HIDDEN]`; every requested
        // pair is aligned and lies within one complete output row.
        unsafe { *weight.add(row * A::HIDDEN + column / 2) }
    }

    #[inline(always)]
    unsafe fn projection_body<A: Arch, const TOKENS: usize>(
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

        while column < 2 * A::HIDDEN {
            // m16n8k16 is the smallest native BF16 Tensor Core tile. B<=8 occupies
            // the lower token rows; zero upper rows do not publish a padded model row.
            let activation = unsafe {
                [
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group + 8,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group,
                        column + 8 + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        group + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all 32 lanes execute the same m16n8k16 instruction with the
            // documented row-major A and column-major B register fragments.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_words = A::HIDDEN / 2;
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower accumulator half maps bijectively to one exact active
            // token row and one adjacent BF16 output pair.
            unsafe {
                *output.add(group * output_words + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    #[inline(always)]
    unsafe fn projection_prefill_body<A: Arch, const TOKENS: usize>(
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
        let block = thread::blockIdx_x() as usize;
        let blocks = A::HIDDEN / 8 / WARPS;
        let output_block = block % blocks;
        let token_tile = block / blocks;
        let output_tile = output_block * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let token_row = token_tile * 16 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < 2 * A::HIDDEN {
            let activation = unsafe {
                [
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        token_row,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        token_row + 8,
                        column + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        token_row,
                        column + 8 + 2 * thread_in_group,
                    ),
                    input_pair::<A, TOKENS>(
                        normalized_embedding,
                        normalized_hidden,
                        token_row + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        let output_words = A::HIDDEN / 2;
        let output_column_word = output_tile * 4 + thread_in_group;
        // Sixteen prompt rows fill both native fragment halves. One CTA still
        // owns 64 adjacent outputs, retaining the measured decode weight walk.
        unsafe {
            *output.add(token_row * output_words + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            *output.add((token_row + 8) * output_words + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[2], accumulator[3]);
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
            projection_body::<Qwen38_27B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }

    /// Projects an exact MTP prompt tile through the source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_fusion_prefill<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            projection_prefill_body::<Qwen38_27B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }

    /// Projects exact Qwen3.5 MTP decode rows through its source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_fusion<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.5 has 512 eight-column tiles. Eight warps per CTA therefore
        // produce 64 columns across 64 CTAs while reading the 64 MiB matrix
        // once. This retains each warp's m16n8k16 accumulation order.
        unsafe {
            projection_body::<Qwen35_9B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }

    /// Projects an exact Qwen3.5 MTP prompt tile through its source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_fusion_prefill<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            projection_prefill_body::<Qwen35_9B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }

    /// Projects exact Qwen3.6 MTP decode rows through its source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_fusion<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.6 has 256 eight-column tiles. Eight warps publish 64
        // columns per CTA, so four CTAs read its 16 MiB matrix once while
        // preserving every output's m16n8k16 accumulation order.
        unsafe {
            projection_body::<Qwen36Moe35B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }

    /// Projects an exact Qwen3.6 MTP prompt tile through its source-BF16 fusion matrix.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_fusion_prefill<const TOKENS: usize>(
        normalized_embedding: *const u32,
        normalized_hidden: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            projection_prefill_body::<Qwen36Moe35B, TOKENS>(
                normalized_embedding,
                normalized_hidden,
                weight,
                output,
            );
        }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_fusion_CudaKernel<TOKENS>>,
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_fusion_prefill_CudaKernel<TOKENS>>,
}

struct PreparedQwen35Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_fusion_CudaKernel<TOKENS>>,
}

struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_fusion_prefill_CudaKernel<TOKENS>>,
}

struct PreparedQwen36Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_fusion_CudaKernel<TOKENS>>,
}

struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_fusion_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 MTP BF16 fusion grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_fusion::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.6 MTP BF16 fusion projection", source)
                })?,
        })
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
            .qwen36_mtp_bf16_fusion::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.6 MTP BF16 fusion projection", source)
            })
    }
}

impl<const TOKENS: usize> PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 MTP BF16 fusion prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_fusion_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.6 MTP BF16 fusion prefill projection",
                        source,
                    )
                })?,
        })
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
            .qwen36_mtp_bf16_fusion_prefill::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.6 MTP BF16 fusion prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 fusion prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_fusion_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.5 MTP BF16 fusion prefill projection",
                        source,
                    )
                })?,
        })
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
            .qwen35_mtp_bf16_fusion_prefill::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.5 MTP BF16 fusion prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 MTP BF16 fusion grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_fusion::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.5 MTP BF16 fusion projection", source)
                })?,
        })
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
            .qwen35_mtp_bf16_fusion::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 fusion projection", source)
            })
    }
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(BLOCKS as usize * token_tiles)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 fusion prefill grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_mtp_bf16_fusion_prefill::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the MTP BF16 fusion prefill projection", source)
                })?,
        })
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
            .mtp_bf16_fusion_prefill::<TOKENS>(
                stream,
                &self.projection,
                normalized_embedding.cast::<u32>(),
                normalized_hidden.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the MTP BF16 fusion prefill projection", source)
            })
    }
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

/// Stable PTX symbol inventory for every exact MTP fusion prompt tile.
pub(crate) fn mtp_bf16_fusion_prefill_ptx_names() -> [&'static str; 4] {
    [
        kernels::mtp_bf16_fusion_prefill_ptx_name::<32>(),
        kernels::mtp_bf16_fusion_prefill_ptx_name::<64>(),
        kernels::mtp_bf16_fusion_prefill_ptx_name::<128>(),
        kernels::mtp_bf16_fusion_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX symbol inventory for every exact Qwen3.5 MTP fusion route.
pub(crate) fn qwen35_mtp_bf16_fusion_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<1>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<2>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<3>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<4>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<5>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<6>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<7>(),
        kernels::qwen35_mtp_bf16_fusion_ptx_name::<8>(),
        kernels::qwen35_mtp_bf16_fusion_prefill_ptx_name::<32>(),
        kernels::qwen35_mtp_bf16_fusion_prefill_ptx_name::<64>(),
        kernels::qwen35_mtp_bf16_fusion_prefill_ptx_name::<128>(),
    ]
}

/// Stable PTX symbol inventory for every exact Qwen3.6 MTP fusion route.
pub(crate) fn qwen36_mtp_bf16_fusion_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<1>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<2>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<3>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<4>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<5>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<6>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<7>(),
        kernels::qwen36_mtp_bf16_fusion_ptx_name::<8>(),
        kernels::qwen36_mtp_bf16_fusion_prefill_ptx_name::<32>(),
        kernels::qwen36_mtp_bf16_fusion_prefill_ptx_name::<64>(),
        kernels::qwen36_mtp_bf16_fusion_prefill_ptx_name::<128>(),
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
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
    t1024: PreparedPrefillRoute<1_024>,
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
        let _ = (
            mtp_bf16_fusion_ptx_names(),
            mtp_bf16_fusion_prefill_ptx_names(),
        );
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
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: PreparedPrefillRoute::prepare(&module)?,
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

        macro_rules! launch_prefill {
            ($route:ident) => {
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
            32 => launch_prefill!(t32),
            64 => launch_prefill!(t64),
            128 => launch_prefill!(t128),
            1_024 => launch_prefill!(t1024),
            _ => Err(GpuError::invalid_launch(format!(
                "MTP BF16 fusion rows {batch} are outside exact B=1..={MAX_BATCH} or T={PREFILL_ROUTES:?}"
            ))),
        }
    }
}

/// Prepared source-BF16 input-fusion routes for exact Qwen3.5 MTP rows.
pub struct Qwen35MtpBf16FusionOp {
    norm: Qwen35ResidualNormOp,
    module: kernels::LoadedModule,
    b1: PreparedQwen35Route<1>,
    b2: PreparedQwen35Route<2>,
    b3: PreparedQwen35Route<3>,
    b4: PreparedQwen35Route<4>,
    b5: PreparedQwen35Route<5>,
    b6: PreparedQwen35Route<6>,
    b7: PreparedQwen35Route<7>,
    b8: PreparedQwen35Route<8>,
    t32: PreparedQwen35PrefillRoute<32>,
    t64: PreparedQwen35PrefillRoute<64>,
    t128: PreparedQwen35PrefillRoute<128>,
}

impl Qwen35MtpBf16FusionOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !QWEN35_HIDDEN.is_multiple_of(16)
            || !QWEN35_FUSION_COLUMNS.is_multiple_of(16)
            || !QWEN35_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP fusion geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = qwen35_mtp_bf16_fusion_ptx_names();
        // SAFETY: this crate owns the embedded MTP fusion artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 MTP BF16 fusion module", source)
        })?;

        Ok(Self {
            norm: Qwen35ResidualNormOp::new(context)?,
            b1: PreparedQwen35Route::prepare(&module)?,
            b2: PreparedQwen35Route::prepare(&module)?,
            b3: PreparedQwen35Route::prepare(&module)?,
            b4: PreparedQwen35Route::prepare(&module)?,
            b5: PreparedQwen35Route::prepare(&module)?,
            b6: PreparedQwen35Route::prepare(&module)?,
            b7: PreparedQwen35Route::prepare(&module)?,
            b8: PreparedQwen35Route::prepare(&module)?,
            t32: PreparedQwen35PrefillRoute::prepare(&module)?,
            t64: PreparedQwen35PrefillRoute::prepare(&module)?,
            t128: PreparedQwen35PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Normalizes Qwen3.5 rows and applies its source-BF16 fusion projection.
    ///
    /// # Safety
    ///
    /// Every pointer is four-byte aligned, context-local, live through stream
    /// completion, and non-overlapping. Row planes cover `rows * 4_096` BF16
    /// values, norm weights cover 4,096 values, and `projection_weight` covers
    /// the row-major `[4_096, 8_192]` source matrix.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        embedding: *const u16,
        hidden: *const u16,
        embedding_norm_weight: *const u16,
        hidden_norm_weight: *const u16,
        normalized_embedding: *mut u16,
        normalized_hidden: *mut u16,
        projection_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        unsafe {
            self.norm.launch_plain(
                stream,
                rows,
                embedding,
                embedding_norm_weight,
                normalized_embedding,
            )?;
            self.norm
                .launch_plain(stream, rows, hidden, hidden_norm_weight, normalized_hidden)?;
        }

        macro_rules! launch {
            ($route:ident) => {
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

        match rows {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 MTP BF16 fusion rows {rows} are outside exact B=1..={MAX_BATCH} or T={QWEN35_PREFILL_ROUTES:?}"
            ))),
        }
    }
}

/// Prepared source-BF16 input-fusion routes for exact Qwen3.6 MTP rows.
pub struct Qwen36MtpBf16FusionOp {
    norm: Qwen36ResidualNormOp,
    module: kernels::LoadedModule,
    b1: PreparedQwen36Route<1>,
    b2: PreparedQwen36Route<2>,
    b3: PreparedQwen36Route<3>,
    b4: PreparedQwen36Route<4>,
    b5: PreparedQwen36Route<5>,
    b6: PreparedQwen36Route<6>,
    b7: PreparedQwen36Route<7>,
    b8: PreparedQwen36Route<8>,
    t32: PreparedQwen36PrefillRoute<32>,
    t64: PreparedQwen36PrefillRoute<64>,
    t128: PreparedQwen36PrefillRoute<128>,
}

impl Qwen36MtpBf16FusionOp {
    /// Loads the embedded module and prepares every exact Qwen3.6 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !QWEN36_HIDDEN.is_multiple_of(16)
            || !QWEN36_FUSION_COLUMNS.is_multiple_of(16)
            || !QWEN36_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 MTP fusion geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = qwen36_mtp_bf16_fusion_ptx_names();
        // SAFETY: this crate owns the embedded MTP fusion artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.6 MTP BF16 fusion module", source)
        })?;

        Ok(Self {
            norm: Qwen36ResidualNormOp::new(context)?,
            b1: PreparedQwen36Route::prepare(&module)?,
            b2: PreparedQwen36Route::prepare(&module)?,
            b3: PreparedQwen36Route::prepare(&module)?,
            b4: PreparedQwen36Route::prepare(&module)?,
            b5: PreparedQwen36Route::prepare(&module)?,
            b6: PreparedQwen36Route::prepare(&module)?,
            b7: PreparedQwen36Route::prepare(&module)?,
            b8: PreparedQwen36Route::prepare(&module)?,
            t32: PreparedQwen36PrefillRoute::prepare(&module)?,
            t64: PreparedQwen36PrefillRoute::prepare(&module)?,
            t128: PreparedQwen36PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Normalizes Qwen3.6 rows and applies its source-BF16 fusion projection.
    ///
    /// # Safety
    ///
    /// Row planes cover `rows * 2_048` BF16 values, norm weights cover 2,048
    /// values, and `projection_weight` covers `[2_048,4_096]`. All pointers
    /// are four-byte aligned, non-overlapping, context-local, and live through
    /// stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        embedding: *const u16,
        hidden: *const u16,
        embedding_norm_weight: *const u16,
        hidden_norm_weight: *const u16,
        normalized_embedding: *mut u16,
        normalized_hidden: *mut u16,
        projection_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        unsafe {
            self.norm.launch_plain(
                stream,
                rows,
                embedding,
                embedding_norm_weight,
                normalized_embedding,
            )?;
            self.norm
                .launch_plain(stream, rows, hidden, hidden_norm_weight, normalized_hidden)?;
        }

        macro_rules! launch {
            ($route:ident) => {
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

        match rows {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MTP BF16 fusion rows {rows} are outside exact B=1..={MAX_BATCH} or T={QWEN35_PREFILL_ROUTES:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCKS, FUSION_COLUMNS, HIDDEN, OUTPUT_TILES, PREFILL_ROUTES, QWEN35_FUSION_COLUMNS,
        QWEN35_HIDDEN, QWEN35_OUTPUT_TILES, QWEN35_PREFILL_ROUTES, QWEN36_FUSION_COLUMNS,
        QWEN36_HIDDEN, QWEN36_OUTPUT_TILES, WARPS, mtp_bf16_fusion_prefill_ptx_names,
        mtp_bf16_fusion_ptx_names, qwen35_mtp_bf16_fusion_ptx_names,
        qwen36_mtp_bf16_fusion_ptx_names,
    };
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
        let prefill = mtp_bf16_fusion_prefill_ptx_names();
        assert_eq!(PREFILL_ROUTES, [32, 64, 128, 1_024]);
        assert_eq!(prefill.len(), PREFILL_ROUTES.len());
        assert_eq!(prefill.iter().copied().collect::<BTreeSet<_>>().len(), 4);
    }

    #[test]
    fn qwen35_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN35_HIDDEN, 4_096);
        assert_eq!(QWEN35_FUSION_COLUMNS, 8_192);
        assert_eq!(QWEN35_OUTPUT_TILES, 512);
        assert_eq!(QWEN35_OUTPUT_TILES / WARPS, 64);
        assert_eq!(QWEN35_PREFILL_ROUTES, [32, 64, 128]);
        let names = qwen35_mtp_bf16_fusion_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }

    #[test]
    fn qwen36_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN36_HIDDEN, 2_048);
        assert_eq!(QWEN36_FUSION_COLUMNS, 4_096);
        assert_eq!(QWEN36_OUTPUT_TILES, 256);
        assert_eq!(QWEN36_OUTPUT_TILES / WARPS, 32);
        let names = qwen36_mtp_bf16_fusion_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }
}

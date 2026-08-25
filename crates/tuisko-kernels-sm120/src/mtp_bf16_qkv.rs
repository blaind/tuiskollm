//! Source-BF16 Q/gate/K/V projection for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const INPUT_COLUMNS: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
const OUTPUT_TILES: usize = OUTPUT_ROWS / 8;
const QWEN35_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const QWEN35_INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
const QWEN35_OUTPUT_ROWS: usize = Qwen35_9B::ATTENTION_QKV_ROWS;
const QWEN35_OUTPUT_TILES: usize = QWEN35_OUTPUT_ROWS / 8;
const QWEN36_INPUT_COLUMNS: usize = Qwen36Moe35B::HIDDEN;
const QWEN36_OUTPUT_ROWS: usize = Qwen36Moe35B::ATTENTION_QKV_ROWS;
const QWEN36_OUTPUT_TILES: usize = QWEN36_OUTPUT_ROWS / 8;
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
    unsafe fn input_pair<A: Arch, const TOKENS: usize>(
        input: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete 5,120-wide BF16 rows.
        unsafe { *input.add(row * (A::HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair<A: Arch>(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the gathered source-BF16 plane is `[OUTPUT_ROWS, INPUT_COLUMNS]`.
        unsafe { *weight.add(row * (A::HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn qkv_body<A: Arch, const TOKENS: usize>(
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

        macro_rules! k_step {
            ($column:expr) => {{
                let column = $column;
                // m16n8k16 is the smallest native BF16 Tensor Core tile. B<=8
                // uses only the lower token rows; no padded row is published.
                let activation = unsafe {
                    [
                        input_pair::<A, TOKENS>(input, group, column + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group + 8, column + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                    ]
                };
                let weights = unsafe {
                    [
                        weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                        weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                    ]
                };
                // SAFETY: all lanes execute the same row-major A / column-major B MMA.
                accumulator =
                    unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            }};
        }
        // The narrow exact routes fold most activation rows to constants and
        // lose load depth; four K-blocks per iteration restore the weight
        // pipeline, and the wide routes take eight (INPUT_COLUMNS = 5,120).
        if TOKENS <= 2 {
            while column < A::HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                column += 64;
            }
        } else {
            while column < A::HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                k_step!(column + 64);
                k_step!(column + 80);
                k_step!(column + 96);
                k_step!(column + 112);
                column += 128;
            }
        }

        if group < TOKENS {
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one fused-row pair.
            unsafe {
                *output.add(group * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    #[inline(always)]
    unsafe fn qkv_prefill_body<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let block = thread::blockIdx_x() as usize;
        let blocks = A::ATTENTION_QKV_ROWS / 8 / WARPS;
        let output_block = block % blocks;
        let token_tile = block / blocks;
        let output_tile = output_block * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let token_row = token_tile * 16 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < A::HIDDEN {
            let activation = unsafe {
                [
                    input_pair::<A, TOKENS>(input, token_row, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row + 8, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row, column + 8 + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row + 8, column + 8 + 2 * thread_in_group),
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

        let output_column_word = output_tile * 4 + thread_in_group;
        // A 16-row token tile publishes both native accumulator halves, while
        // 224 output CTAs retain 1.32 target-SM waves per token tile.
        unsafe {
            *output.add(token_row * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            *output.add((token_row + 8) * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[2], accumulator[3]);
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
        unsafe { qkv_body::<Qwen38_27B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact MTP prompt tile through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen38_27B, TOKENS>(input, weight, output) }
    }

    /// Projects exact Qwen3.5 MTP rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.5 has 1,280 eight-row tiles. Eight warps publish 64 rows
        // per CTA, so 160 CTAs read its 80 MiB matrix once without changing
        // the per-output m16n8k16 accumulation order.
        unsafe { qkv_body::<Qwen35_9B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact Qwen3.5 MTP prompt tile through gathered BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen35_9B, TOKENS>(input, weight, output) }
    }

    /// Projects exact Qwen3.6 MTP rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.6 has 1,152 eight-row tiles. Eight warps publish 64 rows per
        // CTA, so 144 CTAs read its 36 MiB matrix once without changing any
        // per-output m16n8k16 accumulation order.
        unsafe { qkv_body::<Qwen36Moe35B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact Qwen3.6 MTP prompt tile through gathered BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen36Moe35B, TOKENS>(input, weight, output) }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

struct PreparedQwen35Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

struct PreparedQwen36Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 MTP BF16 QKV grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.6 MTP BF16 QKV projection", source)
                })?,
        })
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
            .qwen36_mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.6 MTP BF16 QKV projection", source)
            })
    }
}

impl<const TOKENS: usize> PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 MTP BF16 QKV prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.6 MTP BF16 QKV prefill projection",
                        source,
                    )
                })?,
        })
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
            .qwen36_mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.6 MTP BF16 QKV prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 MTP BF16 QKV grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.5 MTP BF16 QKV projection", source)
                })?,
        })
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
            .qwen35_mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 QKV projection", source)
            })
    }
}

impl<const TOKENS: usize> PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 QKV prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.5 MTP BF16 QKV prefill projection",
                        source,
                    )
                })?,
        })
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
            .qwen35_mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.5 MTP BF16 QKV prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(BLOCKS as usize * token_tiles)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 QKV prefill grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the MTP BF16 QKV prefill projection", source)
                })?,
        })
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
            .mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the MTP BF16 QKV prefill projection", source)
            })
    }
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

/// Stable PTX symbol inventory for every exact MTP QKV prompt tile.
pub(crate) fn mtp_bf16_qkv_prefill_ptx_names() -> [&'static str; 4] {
    [
        kernels::mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<128>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.5 MTP QKV route.
pub(crate) fn qwen35_mtp_bf16_qkv_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<1>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<2>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<3>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<4>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<5>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<6>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<7>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<8>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<128>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.6 MTP QKV route.
pub(crate) fn qwen36_mtp_bf16_qkv_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<1>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<2>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<3>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<4>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<5>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<6>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<7>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<8>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<128>(),
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
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
    t1024: PreparedPrefillRoute<1_024>,
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
        let _ = (mtp_bf16_qkv_ptx_names(), mtp_bf16_qkv_prefill_ptx_names());
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
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: PreparedPrefillRoute::prepare(&module)?,
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

        macro_rules! launch_prefill {
            ($route:ident) => {
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
            32 => launch_prefill!(t32),
            64 => launch_prefill!(t64),
            128 => launch_prefill!(t128),
            1_024 => launch_prefill!(t1024),
            _ => Err(GpuError::invalid_launch(format!(
                "MTP BF16 QKV rows {batch} are outside exact B=1..={MAX_BATCH} or T={PREFILL_ROUTES:?}"
            ))),
        }
    }
}

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact Qwen3.5 MTP rows.
pub struct Qwen35MtpBf16QkvOp {
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

impl Qwen35MtpBf16QkvOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !QWEN35_INPUT_COLUMNS.is_multiple_of(128)
            || !QWEN35_OUTPUT_ROWS.is_multiple_of(64)
            || !QWEN35_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = qwen35_mtp_bf16_qkv_ptx_names();
        // SAFETY: this crate owns the embedded MTP QKV artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 MTP BF16 QKV module", source)
        })?;

        Ok(Self {
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

    /// Applies the exact gathered Qwen3.5 source-BF16 Q/gate/K/V projection.
    ///
    /// # Safety
    ///
    /// Pointers are four-byte aligned, context-local, live through stream
    /// completion, and non-overlapping. Input covers `rows * 4_096` BF16
    /// values, weights cover `[10_240, 4_096]`, and output covers
    /// `rows * 10_240` values.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route
                        .launch(&self.module, stream, input, weight, output)
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
                "Qwen3.5 MTP BF16 QKV rows {rows} are outside exact B=1..={MAX_BATCH} or T={QWEN35_PREFILL_ROUTES:?}"
            ))),
        }
    }
}

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact Qwen3.6 MTP rows.
pub struct Qwen36MtpBf16QkvOp {
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

impl Qwen36MtpBf16QkvOp {
    /// Loads the embedded module and prepares every exact Qwen3.6 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !QWEN36_INPUT_COLUMNS.is_multiple_of(128)
            || !QWEN36_OUTPUT_ROWS.is_multiple_of(64)
            || !QWEN36_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = qwen36_mtp_bf16_qkv_ptx_names();
        // SAFETY: this crate owns the embedded MTP QKV artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.6 MTP BF16 QKV module", source)
        })?;

        Ok(Self {
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

    /// Applies the exact gathered Qwen3.6 source-BF16 Q/gate/K/V projection.
    ///
    /// # Safety
    ///
    /// Input covers `rows * 2_048` BF16 values, weights cover
    /// `[9_216,2_048]`, and output covers `rows * 9_216` values. All pointers
    /// are four-byte aligned, non-overlapping, context-local, and live through
    /// stream completion.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route
                        .launch(&self.module, stream, input, weight, output)
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
                "Qwen3.6 MTP BF16 QKV rows {rows} are outside exact B=1..={MAX_BATCH} or T={QWEN35_PREFILL_ROUTES:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCKS, INPUT_COLUMNS, OUTPUT_ROWS, OUTPUT_TILES, PREFILL_ROUTES, QWEN35_INPUT_COLUMNS,
        QWEN35_OUTPUT_ROWS, QWEN35_OUTPUT_TILES, QWEN35_PREFILL_ROUTES, QWEN36_INPUT_COLUMNS,
        QWEN36_OUTPUT_ROWS, QWEN36_OUTPUT_TILES, WARPS, mtp_bf16_qkv_prefill_ptx_names,
        mtp_bf16_qkv_ptx_names, qwen35_mtp_bf16_qkv_ptx_names, qwen36_mtp_bf16_qkv_ptx_names,
    };
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
        let prefill = mtp_bf16_qkv_prefill_ptx_names();
        assert_eq!(PREFILL_ROUTES, [32, 64, 128, 1_024]);
        assert_eq!(prefill.len(), PREFILL_ROUTES.len());
        assert_eq!(prefill.iter().copied().collect::<BTreeSet<_>>().len(), 4);
    }

    #[test]
    fn qwen35_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN35_INPUT_COLUMNS, 4_096);
        assert_eq!(QWEN35_OUTPUT_ROWS, 10_240);
        assert_eq!(QWEN35_OUTPUT_TILES, 1_280);
        assert_eq!(QWEN35_OUTPUT_TILES / WARPS, 160);
        assert_eq!(QWEN35_PREFILL_ROUTES, [32, 64, 128]);
        let names = qwen35_mtp_bf16_qkv_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }

    #[test]
    fn qwen36_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN36_INPUT_COLUMNS, 2_048);
        assert_eq!(QWEN36_OUTPUT_ROWS, 9_216);
        assert_eq!(QWEN36_OUTPUT_TILES, 1_152);
        assert_eq!(QWEN36_OUTPUT_TILES / WARPS, 144);
        let names = qwen36_mtp_bf16_qkv_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }
}

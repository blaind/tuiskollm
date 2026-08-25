//! Source-BF16 SwiGLU and down projection for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const INTERMEDIATE: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const GATE_TILES: usize = INTERMEDIATE / 8;
// Eight warps publish 64 adjacent SwiGLU rows per CTA. The resulting 272
// CTAs span 1.6 target-SM waves while every warp reuses one input row for
// the paired source gate and up matrices.
const GATE_WARPS: usize = 8;
const GATE_THREADS: u32 = (GATE_WARPS * 32) as u32;
const GATE_BLOCKS: u32 = (GATE_TILES / GATE_WARPS) as u32;
const DOWN_TILES: usize = HIDDEN / 8;
// Four warps publish 32 down-projection rows per CTA. Exactly 160 CTAs put
// almost one complete 17,408-wide dot-product owner on every target SM.
const DOWN_WARPS: usize = 4;
const DOWN_THREADS: u32 = (DOWN_WARPS * 32) as u32;
const DOWN_BLOCKS: u32 = (DOWN_TILES / DOWN_WARPS) as u32;
const QWEN35_HIDDEN: usize = Qwen35_9B::HIDDEN;
const QWEN35_INTERMEDIATE: usize = Qwen35_9B::INTERMEDIATE;
const QWEN35_GATE_UP_ROWS: usize = 2 * QWEN35_INTERMEDIATE;
const QWEN35_GATE_TILES: usize = QWEN35_INTERMEDIATE / 8;
const QWEN35_GATE_BLOCKS: u32 = (QWEN35_GATE_TILES / GATE_WARPS) as u32;
const QWEN35_DOWN_TILES: usize = QWEN35_HIDDEN / 8;
const QWEN35_DOWN_BLOCKS: u32 = (QWEN35_DOWN_TILES / DOWN_WARPS) as u32;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{float, tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(input: *const u32, row: usize, column: usize) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete source-BF16 input rows.
        unsafe { *input.add(row * (HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn gate_up_weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: gate rows precede up rows in the unchanged adjacent source span.
        unsafe { *weight.add(row * (HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    fn silu(value: f32) -> f32 {
        value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    unsafe fn swiglu_body<const TOKENS: usize>(
        input: *const u32,
        gate_up_weight: *const u32,
        activation: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * GATE_WARPS + warp_index;
        let gate_row = output_tile * 8 + group;
        let up_row = INTERMEDIATE + gate_row;
        let mut gate = [0.0f32; 4];
        let mut up = [0.0f32; 4];
        let mut column = 0usize;

        macro_rules! k_step {
            ($column:expr) => {{
                let column = $column;
                let input = unsafe {
                    [
                        input_pair::<TOKENS>(input, group, column + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group + 8, column + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                    ]
                };
                let gate_weight = unsafe {
                    [
                        gate_up_weight_pair(gate_up_weight, gate_row, column + 2 * thread_in_group),
                        gate_up_weight_pair(
                            gate_up_weight,
                            gate_row,
                            column + 8 + 2 * thread_in_group,
                        ),
                    ]
                };
                let up_weight = unsafe {
                    [
                        gate_up_weight_pair(gate_up_weight, up_row, column + 2 * thread_in_group),
                        gate_up_weight_pair(
                            gate_up_weight,
                            up_row,
                            column + 8 + 2 * thread_in_group,
                        ),
                    ]
                };
                // SAFETY: all lanes execute identical native BF16 MMA instructions.
                gate = unsafe { wmma::mma_m16n8k16_f32_bf16(gate, input, gate_weight) };
                up = unsafe { wmma::mma_m16n8k16_f32_bf16(up, input, up_weight) };
            }};
        }
        // The narrow exact routes fold most input rows to constants and lose
        // load depth (B=1 compiled to 32 registers and ran 2.1x slower than
        // B=8 on the same weight stream); four K-blocks per iteration restore
        // the pipeline there while the wide routes take eight before register
        // pressure bites (HIDDEN = 5,120 = 320 sixteen-column blocks).
        if TOKENS <= 2 {
            while column < HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                column += 64;
            }
        } else {
            while column < HIDDEN {
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
            let output_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one
            // adjacent intermediate pair; raw gate/up products are not published.
            unsafe {
                *activation.add(group * (INTERMEDIATE / 2) + output_word) =
                    tcgen05::cvt_f32x2_bf16x2(silu(gate[0]) * up[0], silu(gate[1]) * up[1]);
            }
        }
    }

    #[inline(always)]
    unsafe fn activation_pair<const TOKENS: usize>(
        activation: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete SwiGLU rows.
        unsafe { *activation.add(row * (INTERMEDIATE / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn down_weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the source matrix is exactly `[HIDDEN, INTERMEDIATE]`.
        unsafe { *weight.add(row * (INTERMEDIATE / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn down_body<const TOKENS: usize>(
        activation: *const u32,
        down_weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * DOWN_WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < INTERMEDIATE {
            let input = unsafe {
                [
                    activation_pair::<TOKENS>(activation, group, column + 2 * thread_in_group),
                    activation_pair::<TOKENS>(activation, group + 8, column + 2 * thread_in_group),
                    activation_pair::<TOKENS>(activation, group, column + 8 + 2 * thread_in_group),
                    activation_pair::<TOKENS>(
                        activation,
                        group + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    down_weight_pair(down_weight, weight_row, column + 2 * thread_in_group),
                    down_weight_pair(down_weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, input, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and output pair.
            unsafe {
                *output.add(group * (HIDDEN / 2) + output_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Applies both unchanged BF16 projections and publishes one BF16 SwiGLU seam.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_swiglu<const TOKENS: usize>(
        input: *const u32,
        gate_up_weight: *const u32,
        activation: *mut u32,
    ) {
        // SAFETY: the grid covers every intermediate eight-row tile once.
        unsafe { swiglu_body::<TOKENS>(input, gate_up_weight, activation) }
    }

    /// Applies the unchanged source-BF16 MTP down projection.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_down<const TOKENS: usize>(
        activation: *const u32,
        down_weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the grid covers all 640 hidden-output tiles once.
        unsafe { down_body::<TOKENS>(activation, down_weight, output) }
    }
}

#[cuda_module]
mod qwen35_kernels {
    use super::*;
    use cuda_device::{float, tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<const TOKENS: usize>(input: *const u32, row: usize, column: usize) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete source-BF16 input rows.
        unsafe { *input.add(row * (QWEN35_HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn gate_up_weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: gate rows precede up rows in the unchanged adjacent source span.
        unsafe { *weight.add(row * (QWEN35_HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    fn silu(value: f32) -> f32 {
        value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    unsafe fn swiglu_body<const TOKENS: usize>(
        input: *const u32,
        gate_up_weight: *const u32,
        activation: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * GATE_WARPS + warp_index;
        let gate_row = output_tile * 8 + group;
        let up_row = QWEN35_INTERMEDIATE + gate_row;
        let mut gate = [0.0f32; 4];
        let mut up = [0.0f32; 4];
        let mut column = 0usize;

        macro_rules! k_step {
            ($column:expr) => {{
                let column = $column;
                let input = unsafe {
                    [
                        input_pair::<TOKENS>(input, group, column + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group + 8, column + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                        input_pair::<TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                    ]
                };
                let gate_weight = unsafe {
                    [
                        gate_up_weight_pair(gate_up_weight, gate_row, column + 2 * thread_in_group),
                        gate_up_weight_pair(
                            gate_up_weight,
                            gate_row,
                            column + 8 + 2 * thread_in_group,
                        ),
                    ]
                };
                let up_weight = unsafe {
                    [
                        gate_up_weight_pair(gate_up_weight, up_row, column + 2 * thread_in_group),
                        gate_up_weight_pair(
                            gate_up_weight,
                            up_row,
                            column + 8 + 2 * thread_in_group,
                        ),
                    ]
                };
                // SAFETY: all lanes execute identical native BF16 MMA instructions.
                gate = unsafe { wmma::mma_m16n8k16_f32_bf16(gate, input, gate_weight) };
                up = unsafe { wmma::mma_m16n8k16_f32_bf16(up, input, up_weight) };
            }};
        }

        // The retained Qwen3.8 schedule uses four K-blocks for B<=2 and
        // eight otherwise. Qwen3.5's 4,096 columns divide both strides,
        // preserving every 16-column MMA and its accumulation order.
        if TOKENS <= 2 {
            while column < QWEN35_HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                column += 64;
            }
        } else {
            while column < QWEN35_HIDDEN {
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
            let output_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one
            // adjacent intermediate pair; raw gate/up products are not published.
            unsafe {
                *activation.add(group * (QWEN35_INTERMEDIATE / 2) + output_word) =
                    tcgen05::cvt_f32x2_bf16x2(silu(gate[0]) * up[0], silu(gate[1]) * up[1]);
            }
        }
    }

    #[inline(always)]
    unsafe fn activation_pair<const TOKENS: usize>(
        activation: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete SwiGLU rows.
        unsafe { *activation.add(row * (QWEN35_INTERMEDIATE / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn down_weight_pair(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the source matrix is exactly `[4096, 12288]`.
        unsafe { *weight.add(row * (QWEN35_INTERMEDIATE / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn down_body<const TOKENS: usize>(
        activation: *const u32,
        down_weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * DOWN_WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < QWEN35_INTERMEDIATE {
            let input = unsafe {
                [
                    activation_pair::<TOKENS>(activation, group, column + 2 * thread_in_group),
                    activation_pair::<TOKENS>(activation, group + 8, column + 2 * thread_in_group),
                    activation_pair::<TOKENS>(activation, group, column + 8 + 2 * thread_in_group),
                    activation_pair::<TOKENS>(
                        activation,
                        group + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    down_weight_pair(down_weight, weight_row, column + 2 * thread_in_group),
                    down_weight_pair(down_weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, input, weights) };
            column += 16;
        }

        if group < TOKENS {
            let output_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and output pair.
            unsafe {
                *output.add(group * (QWEN35_HIDDEN / 2) + output_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    /// Applies both unchanged Qwen3.5 BF16 projections and publishes one SwiGLU seam.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_swiglu<const TOKENS: usize>(
        input: *const u32,
        gate_up_weight: *const u32,
        activation: *mut u32,
    ) {
        // Eight warps publish 64 rows per CTA. The exact 12,288-row plane
        // uses 192 CTAs (1.13 target-SM waves) without changing MMA order.
        unsafe { swiglu_body::<TOKENS>(input, gate_up_weight, activation) }
    }

    /// Applies the unchanged source-BF16 Qwen3.5 MTP down projection.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_down<const TOKENS: usize>(
        activation: *const u32,
        down_weight: *const u32,
        output: *mut u32,
    ) {
        // Four warps publish 32 rows per CTA. The 4,096-row output uses 128
        // CTAs while preserving every 12,288-wide accumulation.
        unsafe { down_body::<TOKENS>(activation, down_weight, output) }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    swiglu: PreparedLaunch<kernels::__mtp_bf16_swiglu_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__mtp_bf16_down_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let swiglu = module
            .prepare_mtp_bf16_swiglu::<TOKENS>(LaunchConfig1D::new(GATE_BLOCKS, GATE_THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 SwiGLU", source))?;
        let down = module
            .prepare_mtp_bf16_down::<TOKENS>(LaunchConfig1D::new(DOWN_BLOCKS, DOWN_THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 down projection", source))?;

        Ok(Self { swiglu, down })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        gate_up_weight: *const u16,
        activation: *mut u16,
        down_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_swiglu::<TOKENS>(
                stream,
                &self.swiglu,
                input.cast::<u32>(),
                gate_up_weight.cast::<u32>(),
                activation.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the MTP BF16 SwiGLU", source))?;
        module
            .mtp_bf16_down::<TOKENS>(
                stream,
                &self.down,
                activation.cast::<u32>(),
                down_weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the MTP BF16 down projection", source))
    }
}

struct PreparedQwen35Route<const TOKENS: usize> {
    swiglu: PreparedLaunch<qwen35_kernels::__qwen35_mtp_bf16_swiglu_CudaKernel<TOKENS>>,
    down: PreparedLaunch<qwen35_kernels::__qwen35_mtp_bf16_down_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &qwen35_kernels::LoadedModule) -> GpuResult<Self> {
        let swiglu = module
            .prepare_qwen35_mtp_bf16_swiglu::<TOKENS>(LaunchConfig1D::new(
                QWEN35_GATE_BLOCKS,
                GATE_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the Qwen3.5 MTP BF16 SwiGLU", source))?;
        let down = module
            .prepare_qwen35_mtp_bf16_down::<TOKENS>(LaunchConfig1D::new(
                QWEN35_DOWN_BLOCKS,
                DOWN_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the Qwen3.5 MTP BF16 down projection", source)
            })?;

        Ok(Self { swiglu, down })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &qwen35_kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        gate_up_weight: *const u16,
        activation: *mut u16,
        down_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_mtp_bf16_swiglu::<TOKENS>(
                stream,
                &self.swiglu,
                input.cast::<u32>(),
                gate_up_weight.cast::<u32>(),
                activation.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the Qwen3.5 MTP BF16 SwiGLU", source))?;
        module
            .qwen35_mtp_bf16_down::<TOKENS>(
                stream,
                &self.down,
                activation.cast::<u32>(),
                down_weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 down projection", source)
            })
    }
}

/// Stable PTX inventory for every exact MTP BF16 MLP stage.
pub(crate) fn mtp_bf16_mlp_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::mtp_bf16_swiglu_ptx_name::<1>(),
        kernels::mtp_bf16_swiglu_ptx_name::<2>(),
        kernels::mtp_bf16_swiglu_ptx_name::<3>(),
        kernels::mtp_bf16_swiglu_ptx_name::<4>(),
        kernels::mtp_bf16_swiglu_ptx_name::<5>(),
        kernels::mtp_bf16_swiglu_ptx_name::<6>(),
        kernels::mtp_bf16_swiglu_ptx_name::<7>(),
        kernels::mtp_bf16_swiglu_ptx_name::<8>(),
        kernels::mtp_bf16_down_ptx_name::<1>(),
        kernels::mtp_bf16_down_ptx_name::<2>(),
        kernels::mtp_bf16_down_ptx_name::<3>(),
        kernels::mtp_bf16_down_ptx_name::<4>(),
        kernels::mtp_bf16_down_ptx_name::<5>(),
        kernels::mtp_bf16_down_ptx_name::<6>(),
        kernels::mtp_bf16_down_ptx_name::<7>(),
        kernels::mtp_bf16_down_ptx_name::<8>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.5 MTP BF16 MLP stage.
pub(crate) fn qwen35_mtp_bf16_mlp_ptx_names() -> Vec<&'static str> {
    vec![
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<1>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<2>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<3>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<4>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<5>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<6>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<7>(),
        qwen35_kernels::qwen35_mtp_bf16_swiglu_ptx_name::<8>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<1>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<2>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<3>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<4>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<5>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<6>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<7>(),
        qwen35_kernels::qwen35_mtp_bf16_down_ptx_name::<8>(),
    ]
}

/// Prepared source-BF16 MTP SwiGLU and down-projection routes for `B=1..=8`.
pub struct MtpBf16MlpOp {
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

impl MtpBf16MlpOp {
    /// Loads the embedded module and prepares every exact MTP MLP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if HIDDEN != 5_120
            || INTERMEDIATE != 17_408
            || GATE_UP_ROWS != 34_816
            || !HIDDEN.is_multiple_of(16)
            || !INTERMEDIATE.is_multiple_of(16)
            || !GATE_TILES.is_multiple_of(GATE_WARPS)
            || !DOWN_TILES.is_multiple_of(DOWN_WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP MLP geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = mtp_bf16_mlp_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP MLP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the MTP BF16 MLP", source))?;

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

    /// Applies the exact source-BF16 gate/up SwiGLU and down projection.
    ///
    /// # Safety
    ///
    /// All pointers are four-byte aligned, context-local, non-overlapping, and
    /// live through stream completion. `input` and `output` cover
    /// `[batch,5120]`, `gate_up_weight` covers `[34816,5120]`, `activation`
    /// covers `[batch,17408]`, and `down_weight` covers `[5120,17408]` BF16 values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        gate_up_weight: *const u16,
        activation: *mut u16,
        down_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        gate_up_weight,
                        activation,
                        down_weight,
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
                "MTP BF16 MLP batch {batch} is outside exact B=1..={MAX_BATCH}"
            ))),
        }
    }
}

/// Prepared source-BF16 Qwen3.5 MTP SwiGLU and down routes for `B=1..=8`.
pub struct Qwen35MtpBf16MlpOp {
    module: qwen35_kernels::LoadedModule,
    b1: PreparedQwen35Route<1>,
    b2: PreparedQwen35Route<2>,
    b3: PreparedQwen35Route<3>,
    b4: PreparedQwen35Route<4>,
    b5: PreparedQwen35Route<5>,
    b6: PreparedQwen35Route<6>,
    b7: PreparedQwen35Route<7>,
    b8: PreparedQwen35Route<8>,
}

impl Qwen35MtpBf16MlpOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 MTP MLP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if QWEN35_HIDDEN != 4_096
            || QWEN35_INTERMEDIATE != 12_288
            || QWEN35_GATE_UP_ROWS != 24_576
            || !QWEN35_HIDDEN.is_multiple_of(16)
            || !QWEN35_INTERMEDIATE.is_multiple_of(16)
            || !QWEN35_GATE_TILES.is_multiple_of(GATE_WARPS)
            || !QWEN35_DOWN_TILES.is_multiple_of(DOWN_WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP MLP geometry does not tile exact BF16 MMA shapes",
            ));
        }
        let _ = qwen35_mtp_bf16_mlp_ptx_names();
        // SAFETY: this crate owns the embedded exact Qwen3.5 MTP MLP artifact.
        let module = unsafe { qwen35_kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.5 MTP BF16 MLP", source))?;

        Ok(Self {
            b1: PreparedQwen35Route::prepare(&module)?,
            b2: PreparedQwen35Route::prepare(&module)?,
            b3: PreparedQwen35Route::prepare(&module)?,
            b4: PreparedQwen35Route::prepare(&module)?,
            b5: PreparedQwen35Route::prepare(&module)?,
            b6: PreparedQwen35Route::prepare(&module)?,
            b7: PreparedQwen35Route::prepare(&module)?,
            b8: PreparedQwen35Route::prepare(&module)?,
            module,
        })
    }

    /// Applies the exact Qwen3.5 source-BF16 gate/up SwiGLU and down projection.
    ///
    /// # Safety
    ///
    /// All pointers are four-byte aligned, context-local, non-overlapping, and
    /// live through stream completion. `input` and `output` cover
    /// `[batch,4096]`, `gate_up_weight` covers `[24576,4096]`, `activation`
    /// covers `[batch,12288]`, and `down_weight` covers `[4096,12288]` BF16 values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        gate_up_weight: *const u16,
        activation: *mut u16,
        down_weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        gate_up_weight,
                        activation,
                        down_weight,
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
                "Qwen3.5 MTP BF16 MLP batch {batch} is outside exact B=1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DOWN_BLOCKS, DOWN_TILES, DOWN_WARPS, GATE_BLOCKS, GATE_TILES, GATE_UP_ROWS, GATE_WARPS,
        HIDDEN, INTERMEDIATE, MAX_BATCH, QWEN35_DOWN_BLOCKS, QWEN35_DOWN_TILES, QWEN35_GATE_BLOCKS,
        QWEN35_GATE_TILES, QWEN35_GATE_UP_ROWS, QWEN35_HIDDEN, QWEN35_INTERMEDIATE,
        mtp_bf16_mlp_ptx_names, qwen35_mtp_bf16_mlp_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn exact_geometry_covers_both_source_projections() {
        assert_eq!(HIDDEN, 5_120);
        assert_eq!(INTERMEDIATE, 17_408);
        assert_eq!(GATE_UP_ROWS, 34_816);
        assert_eq!(GATE_BLOCKS, 272);
        assert_eq!(GATE_BLOCKS as usize * GATE_WARPS, GATE_TILES);
        assert_eq!(DOWN_BLOCKS, 160);
        assert_eq!(DOWN_BLOCKS as usize * DOWN_WARPS, DOWN_TILES);
    }

    #[test]
    fn exact_batch_and_stage_inventory_is_complete() {
        let names = mtp_bf16_mlp_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }

    #[test]
    fn qwen35_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN35_HIDDEN, 4_096);
        assert_eq!(QWEN35_INTERMEDIATE, 12_288);
        assert_eq!(QWEN35_GATE_UP_ROWS, 24_576);
        assert_eq!(QWEN35_GATE_BLOCKS, 192);
        assert_eq!(QWEN35_GATE_BLOCKS as usize * GATE_WARPS, QWEN35_GATE_TILES);
        assert_eq!(QWEN35_DOWN_BLOCKS, 128);
        assert_eq!(QWEN35_DOWN_BLOCKS as usize * DOWN_WARPS, QWEN35_DOWN_TILES);
        let names = qwen35_mtp_bf16_mlp_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 16);
    }
}

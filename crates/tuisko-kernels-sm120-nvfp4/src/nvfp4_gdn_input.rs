//! Exact Qwen3.5 NVFP4 GDN input projections.

use crate::device::nvfp4_prefill::{
    BLOCK_N as W4_BLOCK_N, GROUP_K as W4_GROUP_K, THREADS as W4_THREADS, TILE_M as W4_TILE_M,
    project_w4a4, quantize_bf16_rows,
};
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, LaunchConfig2D, PreparedLaunch,
};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
const PROJECTED_ROWS: usize = Qwen35_9B::GDN_INPUT_ROWS;
const CONTROL_ROWS: usize = 2 * Qwen35_9B::GDN_CONTROL_ROWS;
const PADDED_CONTROL_ROWS: usize = 128;
const TOTAL_ROWS: usize = PROJECTED_ROWS + PADDED_CONTROL_ROWS;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP_K;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const PHASE_GROUPS: usize = 32;
const PHASES: usize = GROUPS_PER_ROW / PHASE_GROUPS;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();

// A separate 128-row control projection would launch only eight CTAs, leaving
// 162 of 170 SMs without a first wave. Appending that padded tile to the 12,288
// QKV/Z rows gives 776 rather than 768 CTAs in one graph node. Each warp still
// owns the same two rows and K16 groups, so every accumulation order is unchanged.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = PHASE_GROUPS * GROUP_K / 2;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

const _: () = assert!(INPUT_COLUMNS == 4_096);
const _: () = assert!(PROJECTED_ROWS == 12_288);
const _: () = assert!(CONTROL_ROWS == 64);
const _: () = assert!(PADDED_CONTROL_ROWS == 128);
const _: () = assert!(TOTAL_ROWS == 12_416);
const _: () = assert!(GROUPS_PER_ROW == 256);
const _: () = assert!(PHASES == 8);
const _: () = assert!(SHARED_U32 * size_of::<u32>() == 8_192);

#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, warp};

    #[inline(always)]
    fn weight_scale_offset(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;
        let scale_tiles_per_row = GROUPS_PER_ROW / 4;

        (persistent_tile * scale_tiles_per_row + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset(parent_row: usize, group: usize) -> usize {
        weight_scale_offset(parent_row, group >> 2) + (group & 3)
    }

    #[inline(always)]
    fn physical_row(index: usize) -> usize {
        let tile = index >> 7;
        let in_tile = index & 127;

        tile * 128 + (in_tile >> 2) + (in_tile & 3) * 32
    }

    #[inline(always)]
    fn e4m3_to_f32(code: u8) -> f32 {
        let exponent = (code >> 3) & 15;
        let fraction = code & 7;

        if exponent == 0 {
            fraction as f32 * (1.0 / 512.0)
        } else {
            f32::from_bits(((exponent as u32 + 120) << 23) | ((fraction as u32) << 20))
        }
    }

    #[inline(always)]
    fn e2m1x2_to_f32(packed: u8) -> (f32, f32) {
        let packed_f16: u32;
        let storage = packed as u16;

        unsafe {
            ptx_asm!(
                "{ .reg .b8 lo, zero; mov.b16 {lo, zero}, %1; \
                 cvt.rn.f16x2.e2m1x2 %0, lo; }",
                out("=r") packed_f16,
                in("h") storage,
                options(register_only),
            );
        }

        convert::cvt_f32x2_f16x2(packed_f16)
    }

    #[inline(always)]
    unsafe fn load_u32x2_read_only(source: *const u32) -> (u32, u32) {
        let first: u32;
        let second: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.v2.u32 {%0, %1}, [%2];",
                out("=r") first,
                out("=r") second,
                in("l") source,
                clobber("memory"),
            );
        }

        (first, second)
    }

    #[inline(always)]
    unsafe fn load_u8_read_only(source: *const u8) -> u8 {
        let value: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.u8 %0, [%1];",
                out("=r") value,
                in("l") source,
                clobber("memory"),
            );
        }

        value as u8
    }

    #[inline(always)]
    fn reduce_sum_lane0(mut value: f32) -> f32 {
        value += warp::shuffle_down_f32(value, 16);
        value += warp::shuffle_down_f32(value, 8);
        value += warp::shuffle_down_f32(value, 4);
        value += warp::shuffle_down_f32(value, 2);
        value += warp::shuffle_down_f32(value, 1);

        value
    }

    #[inline(always)]
    fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

        (rounded >> 16) as u16
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn projection_body<const TOKENS: usize>(
        input: *const u32,
        projected_weight_codes: *const u32,
        projected_weight_scales: *const u8,
        projected_weight_reciprocal: f32,
        control_weight_codes: *const u32,
        control_weight_scales: *const u8,
        control_weight_reciprocal: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
        shared: *mut u32,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let pair_index = block * (2 * WARPS) + 2 * warp_index;
        let first_row = physical_row(pair_index);
        let second_row = physical_row(pair_index + 1);
        let first_is_control = first_row >= PROJECTED_ROWS;
        let second_is_control = second_row >= PROJECTED_ROWS;
        let first_parent_row = if first_is_control {
            first_row - PROJECTED_ROWS
        } else {
            first_row
        };
        let second_parent_row = if second_is_control {
            second_row - PROJECTED_ROWS
        } else {
            second_row
        };
        let first_weight_codes = if first_is_control {
            control_weight_codes
        } else {
            projected_weight_codes
        };
        let second_weight_codes = if second_is_control {
            control_weight_codes
        } else {
            projected_weight_codes
        };
        let first_weight_scales = if first_is_control {
            control_weight_scales
        } else {
            projected_weight_scales
        };
        let second_weight_scales = if second_is_control {
            control_weight_scales
        } else {
            projected_weight_scales
        };
        let first_reciprocal = if first_is_control {
            control_weight_reciprocal
        } else {
            projected_weight_reciprocal
        };
        let second_reciprocal = if second_is_control {
            control_weight_reciprocal
        } else {
            projected_weight_reciprocal
        };
        let mut first_accumulators = [0.0f32; TOKENS];
        let mut second_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < PHASES {
            let mut task = tid;
            while task < TOKENS * PHASE_PACKED_PAIRS {
                let token = task / PHASE_PACKED_PAIRS;
                let pair = task - token * PHASE_PACKED_PAIRS;
                unsafe {
                    *shared.add(task) =
                        *input.add(token * (INPUT_COLUMNS / 2) + phase * 256 + pair);
                }
                task += THREADS as usize;
            }
            thread::sync_threads();

            let group = phase * PHASE_GROUPS + lane;
            let first_scale = unsafe {
                load_u8_read_only(
                    first_weight_scales.add(weight_group_scale_offset(first_parent_row, group)),
                )
            };
            let second_scale = unsafe {
                load_u8_read_only(
                    second_weight_scales.add(weight_group_scale_offset(second_parent_row, group)),
                )
            };
            let first_coefficient = e4m3_to_f32(first_scale) * first_reciprocal;
            let second_coefficient = e4m3_to_f32(second_scale) * second_reciprocal;
            let row_words = CODE_BYTES_PER_ROW / size_of::<u32>();
            let word_offset = phase * CODE_WORDS_PER_PHASE + lane * 2;
            let first_source =
                unsafe { first_weight_codes.add(first_parent_row * row_words + word_offset) };
            let second_source =
                unsafe { second_weight_codes.add(second_parent_row * row_words + word_offset) };
            let first_words = unsafe { load_u32x2_read_only(first_source) };
            let second_words = unsafe { load_u32x2_read_only(second_source) };

            macro_rules! accumulate_pair {
                ($pair:literal) => {{
                    let shift = ($pair & 3) * 8;
                    let first_packed = if $pair < 4 {
                        (first_words.0 >> shift) as u8
                    } else {
                        (first_words.1 >> shift) as u8
                    };
                    let second_packed = if $pair < 4 {
                        (second_words.0 >> shift) as u8
                    } else {
                        (second_words.1 >> shift) as u8
                    };
                    let (first_weight0, first_weight1) = e2m1x2_to_f32(first_packed);
                    let (second_weight0, second_weight1) = e2m1x2_to_f32(second_packed);

                    macro_rules! accumulate_token {
                        ($token:literal) => {
                            if $token < TOKENS {
                                let bits = unsafe {
                                    *shared.add($token * PHASE_PACKED_PAIRS + lane * 8 + $pair)
                                };
                                let (activation0, activation1) = convert::cvt_f32x2_bf16x2(bits);
                                first_accumulators[$token] = float::fma_rn_f32(
                                    first_weight0 * first_coefficient,
                                    activation0,
                                    first_accumulators[$token],
                                );
                                first_accumulators[$token] = float::fma_rn_f32(
                                    first_weight1 * first_coefficient,
                                    activation1,
                                    first_accumulators[$token],
                                );
                                second_accumulators[$token] = float::fma_rn_f32(
                                    second_weight0 * second_coefficient,
                                    activation0,
                                    second_accumulators[$token],
                                );
                                second_accumulators[$token] = float::fma_rn_f32(
                                    second_weight1 * second_coefficient,
                                    activation1,
                                    second_accumulators[$token],
                                );
                            }
                        };
                    }

                    accumulate_token!(0);
                    accumulate_token!(1);
                    accumulate_token!(2);
                    accumulate_token!(3);
                    accumulate_token!(4);
                    accumulate_token!(5);
                    accumulate_token!(6);
                    accumulate_token!(7);
                }};
            }

            accumulate_pair!(0);
            accumulate_pair!(1);
            accumulate_pair!(2);
            accumulate_pair!(3);
            accumulate_pair!(4);
            accumulate_pair!(5);
            accumulate_pair!(6);
            accumulate_pair!(7);
            thread::sync_threads();
            phase += 1;
        }

        macro_rules! finish_token {
            ($token:literal) => {
                if $token < TOKENS {
                    let first = reduce_sum_lane0(first_accumulators[$token]);
                    let second = reduce_sum_lane0(second_accumulators[$token]);

                    if lane == 0 {
                        unsafe {
                            if first_is_control {
                                *control_output
                                    .add($token * PADDED_CONTROL_ROWS + first_parent_row) =
                                    f32_to_bf16(first);
                                *control_output
                                    .add($token * PADDED_CONTROL_ROWS + second_parent_row) =
                                    f32_to_bf16(second);
                            } else {
                                *projected_output.add($token * PROJECTED_ROWS + first_parent_row) =
                                    f32_to_bf16(first);
                                *projected_output
                                    .add($token * PROJECTED_ROWS + second_parent_row) =
                                    f32_to_bf16(second);
                            }
                        }
                    }
                }
            };
        }

        finish_token!(0);
        finish_token!(1);
        finish_token!(2);
        finish_token!(3);
        finish_token!(4);
        finish_token!(5);
        finish_token!(6);
        finish_token!(7);
    }

    /// Projects exact Qwen3.5 BF16 rows into Q/K/V/Z and A/B controls.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_gdn_input_a16<const TOKENS: usize>(
        input: *const u32,
        projected_weight_codes: *const u32,
        projected_weight_scales: *const u8,
        projected_weight_reciprocal: f32,
        control_weight_codes: *const u32,
        control_weight_scales: *const u8,
        control_weight_reciprocal: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            projection_body::<TOKENS>(
                input,
                projected_weight_codes,
                projected_weight_scales,
                projected_weight_reciprocal,
                control_weight_codes,
                control_weight_scales,
                control_weight_reciprocal,
                projected_output,
                control_output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    /// Quantizes exact Qwen3.5 prompt rows once for both GDN projections.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_gdn_input_quantize<const TOKENS: usize>(
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        unsafe {
            quantize_bf16_rows::<INPUT_COLUMNS, TOKENS>(
                thread::index_1d().get(),
                input,
                codes,
                scales,
                input_scale_divisor,
            );
        }
    }

    /// Projects represented prompt rows into fused Q/K/V/Z.
    #[kernel]
    #[launch_bounds(384, 2)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_gdn_input_projected_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        // T=32/64/128 expose 192/384/576 independent 48x64 tiles. Every
        // m16n8k64 keeps the same K64 words and order; only independent
        // token/output tiles move out of the 776 decode CTAs.
        unsafe {
            project_w4a4::<INPUT_COLUMNS, PROJECTED_ROWS, PROJECTED_ROWS, 0, TOKENS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                alpha,
                alpha,
                alpha,
            );
        }
    }

    /// Projects the same represented prompt rows into padded A/B controls.
    #[kernel]
    #[launch_bounds(384, 2)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_gdn_input_control_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        // Two 64-row tiles retain the padded 128-row source owner; rows 64..127
        // consume its exact zero words and therefore publish exact BF16 zero.
        unsafe {
            project_w4a4::<INPUT_COLUMNS, PADDED_CONTROL_ROWS, PADDED_CONTROL_ROWS, 0, TOKENS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                alpha,
                alpha,
                alpha,
            );
        }
    }
}

fn launch_config() -> LaunchConfig1D {
    LaunchConfig1D::new((TOTAL_ROWS / (2 * WARPS)) as u32, THREADS, 0)
}

struct PreparedBatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_gdn_input_a16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen35_nvfp4_gdn_input_a16::<TOKENS>(launch_config())
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.5 SM120 NVFP4 GDN input", source)
            })?;

        Ok(Self { projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        projected_weight_codes: *const u8,
        projected_weight_scales: *const u8,
        projected_weight_reciprocal: f32,
        control_weight_codes: *const u8,
        control_weight_scales: *const u8,
        control_weight_reciprocal: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_gdn_input_a16::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                projected_weight_codes.cast::<u32>(),
                projected_weight_scales,
                projected_weight_reciprocal,
                control_weight_codes.cast::<u32>(),
                control_weight_scales,
                control_weight_reciprocal,
                projected_output,
                control_output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 SM120 NVFP4 GDN input", source))
    }
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__qwen35_nvfp4_gdn_input_quantize_CudaKernel<TOKENS>>,
    projected: PreparedLaunch<kernels::__qwen35_nvfp4_gdn_input_projected_w4a4_CudaKernel<TOKENS>>,
    control: PreparedLaunch<kernels::__qwen35_nvfp4_gdn_input_control_w4a4_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 GDN-input prefill route T={TOKENS} is not admitted"
            )));
        }
        let groups_per_row = INPUT_COLUMNS / W4_GROUP_K;
        let quantize_blocks = u32::try_from((TOKENS * groups_per_row).div_ceil(256))
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 GDN quantization grid is too wide"))?;
        let token_tiles = u32::try_from(TOKENS.div_ceil(W4_TILE_M))
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 GDN input grid is too tall"))?;
        let projected_blocks = u32::try_from(PROJECTED_ROWS / W4_BLOCK_N)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 GDN projected grid is too wide"))?;
        let control_blocks = u32::try_from(PADDED_CONTROL_ROWS / W4_BLOCK_N)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 GDN control grid is too wide"))?;

        Ok(Self {
            quantize: module
                .prepare_qwen35_nvfp4_gdn_input_quantize::<TOKENS>(LaunchConfig1D::new(
                    quantize_blocks,
                    256,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 GDN activation quantization", source)
                })?,
            projected: module
                .prepare_qwen35_nvfp4_gdn_input_projected_w4a4::<TOKENS>(LaunchConfig2D::new(
                    (projected_blocks, token_tiles),
                    (W4_THREADS, 1),
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 projected GDN W4A4", source)
                })?,
            control: module
                .prepare_qwen35_nvfp4_gdn_input_control_w4a4::<TOKENS>(LaunchConfig2D::new(
                    (control_blocks, token_tiles),
                    (W4_THREADS, 1),
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing Qwen3.5 control GDN W4A4", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut u8,
        projected_weight_codes: *const u8,
        projected_weight_scales: *const u8,
        projected_weight_scale_divisor: f32,
        control_weight_codes: *const u8,
        control_weight_scales: *const u8,
        control_weight_scale_divisor: f32,
        input_scale_divisor: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_gdn_input_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u32>(),
                activation_scales,
                input_scale_divisor,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.5 GDN activation quantization", source)
            })?;
        module
            .qwen35_nvfp4_gdn_input_projected_w4a4::<TOKENS>(
                stream,
                &self.projected,
                activation_codes.cast::<u32>(),
                activation_scales,
                projected_weight_codes.cast::<u32>(),
                projected_weight_scales,
                projected_output,
                1.0 / (input_scale_divisor * projected_weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 projected GDN W4A4", source))?;
        module
            .qwen35_nvfp4_gdn_input_control_w4a4::<TOKENS>(
                stream,
                &self.control,
                activation_codes.cast::<u32>(),
                activation_scales,
                control_weight_codes.cast::<u32>(),
                control_weight_scales,
                control_output,
                1.0 / (input_scale_divisor * control_weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 control GDN W4A4", source))
    }
}

/// PTX symbols retained for every exact Qwen3.5 GDN input batch.
pub(crate) fn qwen35_nvfp4_gdn_input_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_gdn_input_a16_ptx_name::<8>(),
        kernels::qwen35_nvfp4_gdn_input_quantize_ptx_name::<32>(),
        kernels::qwen35_nvfp4_gdn_input_quantize_ptx_name::<64>(),
        kernels::qwen35_nvfp4_gdn_input_quantize_ptx_name::<128>(),
        kernels::qwen35_nvfp4_gdn_input_projected_w4a4_ptx_name::<32>(),
        kernels::qwen35_nvfp4_gdn_input_projected_w4a4_ptx_name::<64>(),
        kernels::qwen35_nvfp4_gdn_input_projected_w4a4_ptx_name::<128>(),
        kernels::qwen35_nvfp4_gdn_input_control_w4a4_ptx_name::<32>(),
        kernels::qwen35_nvfp4_gdn_input_control_w4a4_ptx_name::<64>(),
        kernels::qwen35_nvfp4_gdn_input_control_w4a4_ptx_name::<128>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_nvfp4_gdn_input_decode),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct Qwen35Nvfp4GdnInputDecodeRoutes {
    #[route(1)]
    b1: PreparedBatchRoute<1>,
    #[route(2)]
    b2: PreparedBatchRoute<2>,
    #[route(3)]
    b3: PreparedBatchRoute<3>,
    #[route(4)]
    b4: PreparedBatchRoute<4>,
    #[route(5)]
    b5: PreparedBatchRoute<5>,
    #[route(6)]
    b6: PreparedBatchRoute<6>,
    #[route(7)]
    b7: PreparedBatchRoute<7>,
    #[route(8)]
    b8: PreparedBatchRoute<8>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_nvfp4_gdn_input_prefill),
    required(32, 64, 128),
    inventory(false)
)]
struct Qwen35Nvfp4GdnInputPrefillRoutes {
    #[route(32)]
    t32: PreparedPrefillRoute<32>,
    #[route(64)]
    t64: PreparedPrefillRoute<64>,
    #[route(128)]
    t128: PreparedPrefillRoute<128>,
}

/// Prepared exact-batch Qwen3.5 NVFP4 GDN input routes on SM120.
pub struct Qwen35Nvfp4GdnInputOp {
    module: kernels::LoadedModule,
    decode_routes: Qwen35Nvfp4GdnInputDecodeRoutes,
    prefill_routes: Qwen35Nvfp4GdnInputPrefillRoutes,
}

impl Qwen35Nvfp4GdnInputOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_nvfp4_gdn_input_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 SM120 NVFP4 GDN input module", source)
        })?;

        Ok(Self {
            decode_routes: Qwen35Nvfp4GdnInputDecodeRoutes::prepare(&module)?,
            prefill_routes: Qwen35Nvfp4GdnInputPrefillRoutes::prepare(&module)?,
            module,
        })
    }

    /// Executes represented-weight A16 GDN input projections at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 4_096` BF16 values. The projected planes cover
    /// packed E2M1 `[12_288, 4_096]` plus swizzled E4M3 `[12_288, 256]`;
    /// controls cover padded `[128, 4_096]` and `[128, 256]` planes. Outputs
    /// cover BF16 `[batch, 12_288]` and `[batch, 128]`. Four-byte-loaded planes
    /// are aligned, all divisors are finite and positive, and disjoint
    /// allocations remain live in `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        projected_weight_codes: *const u8,
        projected_weight_scales: *const u8,
        projected_weight_scale_divisor: f32,
        control_weight_codes: *const u8,
        control_weight_scales: *const u8,
        control_weight_scale_divisor: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
    ) -> GpuResult<()> {
        if [projected_weight_scale_divisor, control_weight_scale_divisor]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 NVFP4 GDN input weight scale divisors must be finite and positive",
            ));
        }
        let projected_weight_reciprocal = 1.0 / projected_weight_scale_divisor;
        let control_weight_reciprocal = 1.0 / control_weight_scale_divisor;

        macro_rules! launch {
            ($route:expr) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        input,
                        projected_weight_codes,
                        projected_weight_scales,
                        projected_weight_reciprocal,
                        control_weight_codes,
                        control_weight_scales,
                        control_weight_reciprocal,
                        projected_output,
                        control_output,
                    )
                }
            };
        }

        dispatch_qwen35_nvfp4_gdn_input_decode!(&self.decode_routes, batch, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 NVFP4 GDN input batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))) )
    }

    /// Quantizes exact prompt rows once and projects both GDN input families.
    ///
    /// # Safety
    ///
    /// `input` covers `tokens * 4_096` BF16 values. Activation scratch covers
    /// packed E2M1 `[tokens, 4_096]` and E4M3 `[tokens, 256]`. Weight planes
    /// cover projected `[12_288, 4_096]` and padded control `[128, 4_096]`
    /// represented NVFP4. Outputs cover BF16 `[tokens, 12_288]` and
    /// `[tokens, 128]`. Four-byte-loaded planes are aligned; all three divisors
    /// are finite and positive; allocations are disjoint, live, and belong to
    /// `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill(
        &self,
        stream: &CudaStream,
        tokens: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut u8,
        projected_weight_codes: *const u8,
        projected_weight_scales: *const u8,
        projected_weight_scale_divisor: f32,
        control_weight_codes: *const u8,
        control_weight_scales: *const u8,
        control_weight_scale_divisor: f32,
        input_scale_divisor: f32,
        projected_output: *mut u16,
        control_output: *mut u16,
    ) -> GpuResult<()> {
        if [
            input_scale_divisor,
            projected_weight_scale_divisor,
            control_weight_scale_divisor,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 NVFP4 GDN input prefill divisors must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:expr) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        input,
                        activation_codes,
                        activation_scales,
                        projected_weight_codes,
                        projected_weight_scales,
                        projected_weight_scale_divisor,
                        control_weight_codes,
                        control_weight_scales,
                        control_weight_scale_divisor,
                        input_scale_divisor,
                        projected_output,
                        control_output,
                    )
                }
            };
        }

        dispatch_qwen35_nvfp4_gdn_input_prefill!(&self.prefill_routes, tokens, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 NVFP4 GDN input prefill row count {tokens} is outside the exact T=32,64,128 routes"
            ))) )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_WORDS_PER_PHASE, CONTROL_ROWS, GROUPS_PER_ROW, MAX_BATCH, PADDED_CONTROL_ROWS, PHASES,
        PREFILL_TOKENS, PROJECTED_ROWS, SHARED_U32, TOTAL_ROWS, WARPS,
        qwen35_nvfp4_gdn_input_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(PROJECTED_ROWS, 12_288);
        assert_eq!(CONTROL_ROWS, 64);
        assert_eq!(PADDED_CONTROL_ROWS, 128);
        assert_eq!(TOTAL_ROWS, 12_416);
        assert_eq!(GROUPS_PER_ROW, 256);
        assert_eq!(PHASES, 8);
        assert_eq!(CODE_WORDS_PER_PHASE, 64);
        assert_eq!(TOTAL_ROWS / (2 * WARPS), 776);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);

        let names = qwen35_nvfp4_gdn_input_ptx_names();
        assert_eq!(PREFILL_TOKENS, [32, 64, 128]);
        assert_eq!(names.len(), MAX_BATCH + 3 * PREFILL_TOKENS.len());
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

//! Exact Qwen3.6 routed and shared NVFP4 expert execution.

use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 3] = [32, 64, 128];
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const SLOTS_PER_TOKEN: usize = TOP_K + 1;
const GROUP_K: usize = 16;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const GATE_UP_GROUPS: usize = HIDDEN / GROUP_K;
const DOWN_GROUPS: usize = INTERMEDIATE / GROUP_K;
const GATE_UP_CODE_BYTES_PER_EXPERT: usize = GATE_UP_ROWS * HIDDEN / 2;
const GATE_UP_SCALE_BYTES_PER_EXPERT: usize = GATE_UP_ROWS * GATE_UP_GROUPS;
const DOWN_CODE_BYTES_PER_EXPERT: usize = HIDDEN * INTERMEDIATE / 2;
const DOWN_SCALE_BYTES_PER_EXPERT: usize = HIDDEN * DOWN_GROUPS;

// One warp owns one gate/up output row and retains its reduction order. Eight
// warps share the same 4,096-byte activation row, so one CTA emits eight of the
// 512 rows. This gives 576 CTAs/token across nine expert slots instead of 4,608
// independent activation reads while leaving every K16 group on one lane.
const GATE_UP_WARPS: usize = 8;
const GATE_UP_THREADS: u32 = (GATE_UP_WARPS * 32) as u32;
const GATE_UP_SHARED_U32: usize = HIDDEN / 2;

// The 512-wide down input has exactly 32 K16 groups. A warp therefore owns one
// group per lane and two output rows; eight warps emit 16 rows while sharing one
// 1,024-byte activation row. Each output remains a single-warp fixed reduction.
const DOWN_WARPS: usize = 8;
const DOWN_ROWS_PER_WARP: usize = 2;
const DOWN_ROWS_PER_CTA: usize = DOWN_WARPS * DOWN_ROWS_PER_WARP;
const DOWN_THREADS: u32 = (DOWN_WARPS * 32) as u32;
const DOWN_SHARED_U32: usize = INTERMEDIATE / 2;

// Eight 256-thread CTAs cover each 2,048-wide output row. At B=8 this exposes
// 64 CTAs rather than only eight token CTAs; each thread retains the exact
// routed-slot 0..7 then shared accumulation order.
const COMBINE_THREADS: u32 = 256;
const COMBINE_BLOCKS_PER_TOKEN: usize = HIDDEN / COMBINE_THREADS as usize;

const _: () = assert!(HIDDEN == 2_048);
const _: () = assert!(INTERMEDIATE == 512);
const _: () = assert!(EXPERTS == 256);
const _: () = assert!(TOP_K == 8);
const _: () = assert!(HIDDEN.is_multiple_of(GROUP_K));
const _: () = assert!(INTERMEDIATE.is_multiple_of(GROUP_K));
const _: () = assert!(INTERMEDIATE.is_multiple_of(GATE_UP_WARPS));
const _: () = assert!(HIDDEN.is_multiple_of(DOWN_ROWS_PER_CTA));
const _: () = assert!(HIDDEN.is_multiple_of(COMBINE_THREADS as usize));

#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, warp};

    #[inline(always)]
    fn scale_offset(row: usize, group: usize, groups_per_row: usize) -> usize {
        let persistent_tile = row >> 7;
        let row_in_tile = row & 127;
        let scale_tile = group >> 2;

        (persistent_tile * (groups_per_row >> 2) + scale_tile) * 512
            + (row_in_tile & 31) * 16
            + (row_in_tile >> 5) * 4
            + (group & 3)
    }

    #[inline(always)]
    fn e4m3_to_f32(code: u8) -> f32 {
        let duplicated = code as u16 | ((code as u16) << 8);
        let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(duplicated);

        convert::cvt_f32_f16x2_lo(packed_f16)
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
    fn reduce_sum_lane_zero(mut value: f32) -> f32 {
        value += warp::shuffle_down_f32(value, 16);
        value += warp::shuffle_down_f32(value, 8);
        value += warp::shuffle_down_f32(value, 4);
        value += warp::shuffle_down_f32(value, 2);
        value += warp::shuffle_down_f32(value, 1);

        value
    }

    #[inline(always)]
    fn silu(value: f32) -> f32 {
        value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    fn slot_expert(token: usize, position: usize, expert_indices: *const u16) -> usize {
        if position < TOP_K {
            unsafe { *expert_indices.add(token * TOP_K + position) as usize }
        } else {
            0
        }
    }

    #[inline(always)]
    unsafe fn expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        shared_gate_weight: *const u32,
        intermediate_output: *mut u16,
        shared_gate_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, GATE_UP_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_row = thread::blockIdx_x() as usize * GATE_UP_WARPS + warp_index;
        let slot = flat_row / INTERMEDIATE;
        let row = flat_row - slot * INTERMEDIATE;
        let token = slot / SLOTS_PER_TOKEN;
        let position = slot - token * SLOTS_PER_TOKEN;
        let expert = slot_expert(token, position, expert_indices);
        let input_row = unsafe { input.add(token * (HIDDEN / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();
        let mut word = tid;

        while word < GATE_UP_SHARED_U32 {
            unsafe { *shared_input.add(word) = *input_row.add(word) };
            word += GATE_UP_THREADS as usize;
        }
        thread::sync_threads();

        let routed = position < TOP_K;
        let expert_code_offset = expert * GATE_UP_CODE_BYTES_PER_EXPERT;
        let expert_scale_offset = expert * GATE_UP_SCALE_BYTES_PER_EXPERT;
        let codes = if routed {
            unsafe { routed_codes.add(expert_code_offset) }
        } else {
            shared_codes
        };
        let scales = if routed {
            unsafe { routed_scales.add(expert_scale_offset) }
        } else {
            shared_scales
        };
        let weight_scale_2 = if routed {
            unsafe { *routed_weight_scales_2.add(expert) }
        } else {
            shared_weight_scale_2
        };
        let gate_row = row;
        let up_row = row + INTERMEDIATE;
        let row_code_bytes = HIDDEN / 2;
        let mut gate = [0.0f32; 4];
        let mut up = [0.0f32; 4];
        let mut shared_gate = [0.0f32; 4];
        let mut phase = 0usize;

        while phase < GATE_UP_GROUPS / 32 {
            let group = phase * 32 + lane;
            let gate_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(gate_row, group, GATE_UP_GROUPS)))
            };
            let up_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(up_row, group, GATE_UP_GROUPS)))
            };
            let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_2;
            let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_2;
            let gate_source =
                unsafe { codes.add(gate_row * row_code_bytes + group * (GROUP_K / 2)) };
            let up_source = unsafe { codes.add(up_row * row_code_bytes + group * (GROUP_K / 2)) };
            let gate_words = unsafe { load_u32x2_read_only(gate_source.cast::<u32>()) };
            let up_words = unsafe { load_u32x2_read_only(up_source.cast::<u32>()) };
            let activation_source = unsafe { shared_input.add(group * (GROUP_K / 2)) };
            let shared_gate_source = unsafe { shared_gate_weight.add(group * (GROUP_K / 2)) };

            macro_rules! accumulate_pair {
                ($pair:literal, $chain:literal) => {{
                    let shift = ($pair & 3) * 8;
                    let gate_packed = if $pair < 4 {
                        (gate_words.0 >> shift) as u8
                    } else {
                        (gate_words.1 >> shift) as u8
                    };
                    let up_packed = if $pair < 4 {
                        (up_words.0 >> shift) as u8
                    } else {
                        (up_words.1 >> shift) as u8
                    };
                    let (gate_low, gate_high) = e2m1x2_to_f32(gate_packed);
                    let (up_low, up_high) = e2m1x2_to_f32(up_packed);
                    let activation_bits = unsafe { *activation_source.add($pair) };
                    let (activation_low, activation_high) =
                        convert::cvt_f32x2_bf16x2(activation_bits);
                    gate[$chain] = float::fma_rn_f32(
                        gate_low * gate_coefficient,
                        activation_low,
                        gate[$chain],
                    );
                    gate[$chain] = float::fma_rn_f32(
                        gate_high * gate_coefficient,
                        activation_high,
                        gate[$chain],
                    );
                    up[$chain] =
                        float::fma_rn_f32(up_low * up_coefficient, activation_low, up[$chain]);
                    up[$chain] =
                        float::fma_rn_f32(up_high * up_coefficient, activation_high, up[$chain]);
                    if !routed && row == 0 {
                        let shared_bits = unsafe { *shared_gate_source.add($pair) };
                        let (shared_low, shared_high) = convert::cvt_f32x2_bf16x2(shared_bits);
                        shared_gate[$chain] =
                            float::fma_rn_f32(shared_low, activation_low, shared_gate[$chain]);
                        shared_gate[$chain] =
                            float::fma_rn_f32(shared_high, activation_high, shared_gate[$chain]);
                    }
                }};
            }

            accumulate_pair!(0, 0);
            accumulate_pair!(1, 1);
            accumulate_pair!(2, 2);
            accumulate_pair!(3, 3);
            accumulate_pair!(4, 0);
            accumulate_pair!(5, 1);
            accumulate_pair!(6, 2);
            accumulate_pair!(7, 3);
            phase += 1;
        }

        let gate = reduce_sum_lane_zero(gate[0] + gate[1] + gate[2] + gate[3]);
        let up = reduce_sum_lane_zero(up[0] + up[1] + up[2] + up[3]);
        if lane == 0 {
            unsafe {
                *intermediate_output.add(slot * INTERMEDIATE + row) =
                    tcgen05::f32_to_bf16_rne(silu(gate) * up);
            }
        }
        if !routed && row == 0 {
            let shared_gate = reduce_sum_lane_zero(
                shared_gate[0] + shared_gate[1] + shared_gate[2] + shared_gate[3],
            );
            if lane == 0 {
                unsafe {
                    *shared_gate_output.add(token) = tcgen05::f32_to_bf16_rne(shared_gate);
                }
            }
        }
    }

    /// Executes routed and shared gate/up for an exact decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        shared_gate_weight: *const u32,
        intermediate_output: *mut u16,
        shared_gate_output: *mut u16,
    ) {
        unsafe {
            expert_gate_up::<TOKENS>(
                input,
                expert_indices,
                routed_codes,
                routed_scales,
                routed_weight_scales_2,
                shared_codes,
                shared_scales,
                shared_weight_scale_2,
                shared_gate_weight,
                intermediate_output,
                shared_gate_output,
            )
        }
    }

    /// Executes routed and shared gate/up for an exact prompt tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_gate_up_prefill<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        shared_gate_weight: *const u32,
        intermediate_output: *mut u16,
        shared_gate_output: *mut u16,
    ) {
        unsafe {
            expert_gate_up::<TOKENS>(
                input,
                expert_indices,
                routed_codes,
                routed_scales,
                routed_weight_scales_2,
                shared_codes,
                shared_scales,
                shared_weight_scale_2,
                shared_gate_weight,
                intermediate_output,
                shared_gate_output,
            )
        }
    }

    #[inline(always)]
    unsafe fn expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        expert_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, DOWN_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_pair = thread::blockIdx_x() as usize * DOWN_ROWS_PER_CTA + 2 * warp_index;
        let slot = flat_pair / HIDDEN;
        let first_row = flat_pair - slot * HIDDEN;
        let second_row = first_row + 1;
        let token = slot / SLOTS_PER_TOKEN;
        let position = slot - token * SLOTS_PER_TOKEN;
        let expert = slot_expert(token, position, expert_indices);
        let input_row = unsafe { intermediate_input.add(slot * (INTERMEDIATE / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();

        if tid < DOWN_SHARED_U32 {
            unsafe { *shared_input.add(tid) = *input_row.add(tid) };
        }
        thread::sync_threads();

        let routed = position < TOP_K;
        let expert_code_offset = expert * DOWN_CODE_BYTES_PER_EXPERT;
        let expert_scale_offset = expert * DOWN_SCALE_BYTES_PER_EXPERT;
        let codes = if routed {
            unsafe { routed_codes.add(expert_code_offset) }
        } else {
            shared_codes
        };
        let scales = if routed {
            unsafe { routed_scales.add(expert_scale_offset) }
        } else {
            shared_scales
        };
        let weight_scale_2 = if routed {
            unsafe { *routed_weight_scales_2.add(expert) }
        } else {
            shared_weight_scale_2
        };
        let group = lane;
        let row_code_bytes = INTERMEDIATE / 2;
        let first_scale =
            unsafe { load_u8_read_only(scales.add(scale_offset(first_row, group, DOWN_GROUPS))) };
        let second_scale =
            unsafe { load_u8_read_only(scales.add(scale_offset(second_row, group, DOWN_GROUPS))) };
        let first_coefficient = e4m3_to_f32(first_scale) * weight_scale_2;
        let second_coefficient = e4m3_to_f32(second_scale) * weight_scale_2;
        let first_source = unsafe { codes.add(first_row * row_code_bytes + group * (GROUP_K / 2)) };
        let second_source =
            unsafe { codes.add(second_row * row_code_bytes + group * (GROUP_K / 2)) };
        let first_words = unsafe { load_u32x2_read_only(first_source.cast::<u32>()) };
        let second_words = unsafe { load_u32x2_read_only(second_source.cast::<u32>()) };
        let activation_source = unsafe { shared_input.add(group * (GROUP_K / 2)) };
        let mut first = [0.0f32; 4];
        let mut second = [0.0f32; 4];

        macro_rules! accumulate_pair {
            ($pair:literal, $chain:literal) => {{
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
                let (first_low, first_high) = e2m1x2_to_f32(first_packed);
                let (second_low, second_high) = e2m1x2_to_f32(second_packed);
                let activation_bits = unsafe { *activation_source.add($pair) };
                let (activation_low, activation_high) = convert::cvt_f32x2_bf16x2(activation_bits);
                first[$chain] =
                    float::fma_rn_f32(first_low * first_coefficient, activation_low, first[$chain]);
                first[$chain] = float::fma_rn_f32(
                    first_high * first_coefficient,
                    activation_high,
                    first[$chain],
                );
                second[$chain] = float::fma_rn_f32(
                    second_low * second_coefficient,
                    activation_low,
                    second[$chain],
                );
                second[$chain] = float::fma_rn_f32(
                    second_high * second_coefficient,
                    activation_high,
                    second[$chain],
                );
            }};
        }

        accumulate_pair!(0, 0);
        accumulate_pair!(1, 1);
        accumulate_pair!(2, 2);
        accumulate_pair!(3, 3);
        accumulate_pair!(4, 0);
        accumulate_pair!(5, 1);
        accumulate_pair!(6, 2);
        accumulate_pair!(7, 3);

        let first = reduce_sum_lane_zero(first[0] + first[1] + first[2] + first[3]);
        let second = reduce_sum_lane_zero(second[0] + second[1] + second[2] + second[3]);
        if lane == 0 {
            unsafe {
                *expert_output.add(slot * HIDDEN + first_row) = tcgen05::f32_to_bf16_rne(first);
                *expert_output.add(slot * HIDDEN + second_row) = tcgen05::f32_to_bf16_rne(second);
            }
        }
    }

    /// Executes routed and shared down projections for an exact decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        expert_output: *mut u16,
    ) {
        unsafe {
            expert_down::<TOKENS>(
                intermediate_input,
                expert_indices,
                routed_codes,
                routed_scales,
                routed_weight_scales_2,
                shared_codes,
                shared_scales,
                shared_weight_scale_2,
                expert_output,
            )
        }
    }

    /// Executes routed and shared down projections for an exact prompt tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_down_prefill<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        routed_codes: *const u8,
        routed_scales: *const u8,
        routed_weight_scales_2: *const f32,
        shared_codes: *const u8,
        shared_scales: *const u8,
        shared_weight_scale_2: f32,
        expert_output: *mut u16,
    ) {
        unsafe {
            expert_down::<TOKENS>(
                intermediate_input,
                expert_indices,
                routed_codes,
                routed_scales,
                routed_weight_scales_2,
                shared_codes,
                shared_scales,
                shared_weight_scale_2,
                expert_output,
            )
        }
    }

    #[inline(always)]
    unsafe fn expert_combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_gate: *const u16,
        output: *mut u16,
    ) {
        let flat = thread::blockIdx_x() as usize * COMBINE_THREADS as usize
            + thread::threadIdx_x() as usize;
        let token = flat / HIDDEN;
        let column = flat - token * HIDDEN;
        let token_slots = unsafe { expert_output.add(token * SLOTS_PER_TOKEN * HIDDEN) };
        let token_weights = unsafe { routing_weights.add(token * TOP_K) };
        let mut sum = 0.0f32;
        let mut position = 0usize;

        while position < TOP_K {
            let expert_value = f32::from_bits(
                u32::from(unsafe { *token_slots.add(position * HIDDEN + column) }) << 16,
            );
            let routing_weight =
                f32::from_bits(u32::from(unsafe { *token_weights.add(position) }) << 16);
            sum = float::fma_rn_f32(expert_value, routing_weight, sum);
            position += 1;
        }

        let shared_value =
            f32::from_bits(u32::from(unsafe { *token_slots.add(TOP_K * HIDDEN + column) }) << 16);
        let shared_logit = f32::from_bits(u32::from(unsafe { *shared_gate.add(token) }) << 16);
        sum = float::fma_rn_f32(shared_value, sigmoid(shared_logit), sum);
        unsafe { *output.add(token * HIDDEN + column) = tcgen05::f32_to_bf16_rne(sum) };
    }

    /// Combines routed and shared outputs for an exact decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_gate: *const u16,
        output: *mut u16,
    ) {
        unsafe { expert_combine::<TOKENS>(expert_output, routing_weights, shared_gate, output) }
    }

    /// Combines routed and shared outputs for an exact prompt tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_expert_combine_prefill<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_gate: *const u16,
        output: *mut u16,
    ) {
        unsafe { expert_combine::<TOKENS>(expert_output, routing_weights, shared_gate, output) }
    }
}

fn gate_up_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * SLOTS_PER_TOKEN * INTERMEDIATE / GATE_UP_WARPS) as u32,
        GATE_UP_THREADS,
        0,
    )
}

fn down_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * SLOTS_PER_TOKEN * HIDDEN / DOWN_ROWS_PER_CTA) as u32,
        DOWN_THREADS,
        0,
    )
}

fn combine_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * COMBINE_BLOCKS_PER_TOKEN) as u32,
        COMBINE_THREADS,
        0,
    )
}

struct PreparedBatchRoute<const TOKENS: usize> {
    gate_up: PreparedLaunch<kernels::__qwen36_moe_expert_gate_up_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__qwen36_moe_expert_down_CudaKernel<TOKENS>>,
    combine: PreparedLaunch<kernels::__qwen36_moe_expert_combine_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !(1..=MAX_BATCH).contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE expert decode row count {TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            gate_up: module
                .prepare_qwen36_moe_expert_gate_up::<TOKENS>(gate_up_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing Qwen3.6 MoE gate/up", source))?,
            down: module
                .prepare_qwen36_moe_expert_down::<TOKENS>(down_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing Qwen3.6 MoE down", source))?,
            combine: module
                .prepare_qwen36_moe_expert_combine::<TOKENS>(combine_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing Qwen3.6 MoE combine", source))?,
        })
    }

    fn ptx_names() -> Vec<&'static str> {
        vec![
            kernels::qwen36_moe_expert_gate_up_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_expert_down_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_expert_combine_ptx_name::<TOKENS>(),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        expert_indices: *const u16,
        routing_weights: *const u16,
        routed_gate_up_codes: *const u8,
        routed_gate_up_scales: *const u8,
        routed_gate_up_weight_scales_2: *const f32,
        routed_down_codes: *const u8,
        routed_down_scales: *const u8,
        routed_down_weight_scales_2: *const f32,
        shared_gate_up_codes: *const u8,
        shared_gate_up_scales: *const u8,
        shared_gate_up_weight_scale_2: f32,
        shared_down_codes: *const u8,
        shared_down_scales: *const u8,
        shared_down_weight_scale_2: f32,
        shared_gate_weight: *const u16,
        intermediate: *mut u16,
        expert_output: *mut u16,
        shared_gate: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_moe_expert_gate_up::<TOKENS>(
                stream,
                &self.gate_up,
                input.cast::<u32>(),
                expert_indices,
                routed_gate_up_codes,
                routed_gate_up_scales,
                routed_gate_up_weight_scales_2,
                shared_gate_up_codes,
                shared_gate_up_scales,
                shared_gate_up_weight_scale_2,
                shared_gate_weight.cast::<u32>(),
                intermediate,
                shared_gate,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE gate/up", source))?;
        module
            .qwen36_moe_expert_down::<TOKENS>(
                stream,
                &self.down,
                intermediate.cast::<u32>(),
                expert_indices,
                routed_down_codes,
                routed_down_scales,
                routed_down_weight_scales_2,
                shared_down_codes,
                shared_down_scales,
                shared_down_weight_scale_2,
                expert_output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE down", source))?;
        module
            .qwen36_moe_expert_combine::<TOKENS>(
                stream,
                &self.combine,
                expert_output,
                routing_weights,
                shared_gate,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE combine", source))
    }
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    gate_up: PreparedLaunch<kernels::__qwen36_moe_expert_gate_up_prefill_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__qwen36_moe_expert_down_prefill_CudaKernel<TOKENS>>,
    combine: PreparedLaunch<kernels::__qwen36_moe_expert_combine_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE expert prefill row count {TOKENS} is not admitted"
            )));
        }
        // T=32/64/128 measure 252/480/937 us and 1,895/1,991/2,040 GiB/s of
        // logical selected-weight traffic at locked clocks. T=128 already
        // exposes 73,728 gate/up, 147,456 down, and 1,024 combine CTAs across
        // 170 SMs, so this first prompt route retains the decode topology.
        // Specialization changes only the independent token count: one warp
        // still owns each gate/up row or down-row pair and combine retains its
        // routed-slot 0..7 then shared order, so arithmetic is unchanged.
        Ok(Self {
            gate_up: module
                .prepare_qwen36_moe_expert_gate_up_prefill::<TOKENS>(gate_up_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE prompt gate/up", source)
                })?,
            down: module
                .prepare_qwen36_moe_expert_down_prefill::<TOKENS>(down_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing Qwen3.6 MoE prompt down", source))?,
            combine: module
                .prepare_qwen36_moe_expert_combine_prefill::<TOKENS>(combine_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE prompt combine", source)
                })?,
        })
    }

    fn ptx_names() -> Vec<&'static str> {
        vec![
            kernels::qwen36_moe_expert_gate_up_prefill_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_expert_down_prefill_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_expert_combine_prefill_ptx_name::<TOKENS>(),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        expert_indices: *const u16,
        routing_weights: *const u16,
        routed_gate_up_codes: *const u8,
        routed_gate_up_scales: *const u8,
        routed_gate_up_weight_scales_2: *const f32,
        routed_down_codes: *const u8,
        routed_down_scales: *const u8,
        routed_down_weight_scales_2: *const f32,
        shared_gate_up_codes: *const u8,
        shared_gate_up_scales: *const u8,
        shared_gate_up_weight_scale_2: f32,
        shared_down_codes: *const u8,
        shared_down_scales: *const u8,
        shared_down_weight_scale_2: f32,
        shared_gate_weight: *const u16,
        intermediate: *mut u16,
        expert_output: *mut u16,
        shared_gate: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_moe_expert_gate_up_prefill::<TOKENS>(
                stream,
                &self.gate_up,
                input.cast::<u32>(),
                expert_indices,
                routed_gate_up_codes,
                routed_gate_up_scales,
                routed_gate_up_weight_scales_2,
                shared_gate_up_codes,
                shared_gate_up_scales,
                shared_gate_up_weight_scale_2,
                shared_gate_weight.cast::<u32>(),
                intermediate,
                shared_gate,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE prompt gate/up", source))?;
        module
            .qwen36_moe_expert_down_prefill::<TOKENS>(
                stream,
                &self.down,
                intermediate.cast::<u32>(),
                expert_indices,
                routed_down_codes,
                routed_down_scales,
                routed_down_weight_scales_2,
                shared_down_codes,
                shared_down_scales,
                shared_down_weight_scale_2,
                expert_output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE prompt down", source))?;
        module
            .qwen36_moe_expert_combine_prefill::<TOKENS>(
                stream,
                &self.combine,
                expert_output,
                routing_weights,
                shared_gate,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE prompt combine", source))
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_moe_experts),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128)
)]
struct MoeExpertRoutes {
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
    #[route(32)]
    t32: PreparedPrefillRoute<32>,
    #[route(64)]
    t64: PreparedPrefillRoute<64>,
    #[route(128)]
    t128: PreparedPrefillRoute<128>,
}

/// PTX symbols retained for every exact Qwen3.6 expert batch.
pub(crate) fn qwen36_moe_experts_ptx_names() -> Vec<&'static str> {
    MoeExpertRoutes::ptx_names()
}

/// Prepared exact-batch Qwen3.6 routed/shared NVFP4 expert routes on SM120.
pub struct Qwen36MoeExpertsOp {
    module: kernels::LoadedModule,
    routes: MoeExpertRoutes,
}

impl Qwen36MoeExpertsOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_moe_experts_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.6 MoE experts", source))?;

        let routes = MoeExpertRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Executes the selected routed experts, shared expert, and fixed-order combine.
    ///
    /// # Safety
    ///
    /// Every pointer covers the exact Qwen3.6 planes documented by its name.
    /// Routed planes contain 256 numeric-order experts and every selected index is
    /// below 256. Workspaces cover `rows * 9 * 512`, `rows * 9 * 2_048`,
    /// `rows`, and `rows * 2_048` values respectively. Four-byte-loaded planes
    /// are aligned, disjoint, and live in `stream`'s context through completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        expert_indices: *const u16,
        routing_weights: *const u16,
        routed_gate_up_codes: *const u8,
        routed_gate_up_scales: *const u8,
        routed_gate_up_weight_scales_2: *const f32,
        routed_down_codes: *const u8,
        routed_down_scales: *const u8,
        routed_down_weight_scales_2: *const f32,
        shared_gate_up_codes: *const u8,
        shared_gate_up_scales: *const u8,
        shared_gate_up_weight_scale_2: f32,
        shared_down_codes: *const u8,
        shared_down_scales: *const u8,
        shared_down_weight_scale_2: f32,
        shared_gate_weight: *const u16,
        intermediate: *mut u16,
        expert_output: *mut u16,
        shared_gate: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_moe_experts!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    expert_indices,
                    routing_weights,
                    routed_gate_up_codes,
                    routed_gate_up_scales,
                    routed_gate_up_weight_scales_2,
                    routed_down_codes,
                    routed_down_scales,
                    routed_down_weight_scales_2,
                    shared_gate_up_codes,
                    shared_gate_up_scales,
                    shared_gate_up_weight_scale_2,
                    shared_down_codes,
                    shared_down_scales,
                    shared_down_weight_scale_2,
                    shared_gate_weight,
                    intermediate,
                    expert_output,
                    shared_gate,
                    output,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE expert row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128"
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DOWN_CODE_BYTES_PER_EXPERT, DOWN_SCALE_BYTES_PER_EXPERT, EXPERTS,
        GATE_UP_CODE_BYTES_PER_EXPERT, GATE_UP_SCALE_BYTES_PER_EXPERT, MAX_BATCH, MoeExpertRoutes,
        PREFILL_ROWS, SLOTS_PER_TOKEN, qwen36_moe_experts_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(SLOTS_PER_TOKEN, 9);
        assert_eq!(GATE_UP_CODE_BYTES_PER_EXPERT, 1_048_576);
        assert_eq!(GATE_UP_SCALE_BYTES_PER_EXPERT, 131_072);
        assert_eq!(DOWN_CODE_BYTES_PER_EXPERT, 524_288);
        assert_eq!(DOWN_SCALE_BYTES_PER_EXPERT, 65_536);
        assert_eq!(EXPERTS, 256);

        let names = qwen36_moe_experts_ptx_names();
        assert_eq!(names.len(), 3 * (MAX_BATCH + PREFILL_ROWS.len()));
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }

    #[test]
    fn row_table_covers_only_exact_decode_and_prefill_routes() {
        for (rows, expected) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (16, false),
            (32, true),
            (33, false),
            (64, true),
            (65, false),
            (128, true),
            (129, false),
        ] {
            assert_eq!(MoeExpertRoutes::contains(rows), expected, "rows={rows}");
        }
    }
}

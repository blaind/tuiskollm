//! Exact-target NVFP4 gate/up projection with fused SwiGLU.

use cuda_device::async_copy::{
    cp_async_ca_4, cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, LaunchConfig2D, PreparedLaunch,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * OUTPUT_ROWS;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = HIDDEN / GROUP_K;
const CODE_BYTES_PER_ROW: usize = HIDDEN / 2;
const SCALE_TILES_PER_ROW: usize = GROUPS_PER_ROW / 4;

// Eight warps cover eight gate/up row pairs per CTA. The 2,176-block grid
// preserves the scale-plane M128x4 row order used by materialization.
const A16_WARPS: usize = 8;
const A16_THREADS: u32 = (A16_WARPS * 32) as u32;
const A16_T2_SHARED_U32: usize = 2_128 / 4;
const A16_T4_SHARED_U32: usize = 4_176 / 4;

// The retained W4A4 decode tile uses 48 token slots and four N warp groups.
// Twelve warps stage each 256-wide K tile while keeping two CTAs resident.
const BLOCK_M: usize = 64;
const SMALL_BLOCK_M: usize = 48;
const BLOCK_N: usize = 64;
const BLOCK_K: usize = 256;
const WARPS_N: usize = 4;
const WARP_M: usize = 16;
const WARP_N: usize = BLOCK_N / WARPS_N;
const MMA_N: usize = WARP_N / 8;
const STAGES: usize = 2;
const K64_PER_STAGE: usize = BLOCK_K / 64;
const CODE_ROW_BYTES: usize = BLOCK_K / 2;
const SEGMENTS_PER_ROW: usize = CODE_ROW_BYTES / 16;
const K_TILES: usize = HIDDEN / BLOCK_K;
const ROWS_PER_BRANCH: usize = BLOCK_N / 2;
const OUTPUT_STRIDE: usize = BLOCK_N + 8;
const W4A4_THREADS: u32 = ((SMALL_BLOCK_M / WARP_M) * WARPS_N * 32) as u32;

const A_CODE_BYTES: usize = STAGES * BLOCK_M * CODE_ROW_BYTES;
const B_CODE_BYTES: usize = STAGES * BLOCK_N * CODE_ROW_BYTES;
const A_SCALE_BYTES: usize = STAGES * BLOCK_M * K64_PER_STAGE * 4;
const B_SCALE_BYTES: usize = STAGES * BLOCK_N * K64_PER_STAGE * 4;
const A_CODE_OFFSET: usize = 0;
const B_CODE_OFFSET: usize = A_CODE_OFFSET + A_CODE_BYTES;
const A_SCALE_OFFSET: usize = B_CODE_OFFSET + B_CODE_BYTES;
const B_SCALE_OFFSET: usize = A_SCALE_OFFSET + A_SCALE_BYTES;
const SHARED_BYTES: usize = B_SCALE_OFFSET + B_SCALE_BYTES;
const SHARED_U32: usize = SHARED_BYTES / 4;

const _: () = assert!(HIDDEN == 5_120);
const _: () = assert!(OUTPUT_ROWS == 17_408);
const _: () = assert!(GATE_UP_ROWS == 34_816);
const _: () = assert!(SHARED_BYTES == 36_864);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, warp, wmma};

    #[inline(always)]
    fn swizzled_byte(row: usize, logical_byte: usize) -> usize {
        let logical_segment = logical_byte >> 4;
        let byte_in_segment = logical_byte & 15;
        let physical_segment = logical_segment ^ (row & (SEGMENTS_PER_ROW - 1));

        physical_segment * 16 + byte_in_segment
    }

    #[inline(always)]
    fn weight_row(row_begin: usize, local_row: usize) -> usize {
        row_begin
            + (local_row & (ROWS_PER_BRANCH - 1))
            + if local_row >= ROWS_PER_BRANCH {
                OUTPUT_ROWS
            } else {
                0
            }
    }

    #[inline(always)]
    fn weight_scale_offset(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;

        (persistent_tile * SCALE_TILES_PER_ROW + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset(parent_row: usize, group: usize) -> usize {
        weight_scale_offset(parent_row, group >> 2) + (group & 3)
    }

    #[inline(always)]
    fn e4m3_to_f32(scale: u8) -> f32 {
        let duplicated = scale as u16 | ((scale as u16) << 8);
        let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(duplicated);

        convert::cvt_f32_f16x2_lo(packed_f16)
    }

    #[inline(always)]
    unsafe fn load_u32x4(source: *const u32) -> (u32, u32, u32, u32) {
        let first: u32;
        let second: u32;
        let third: u32;
        let fourth: u32;

        unsafe {
            ptx_asm!(
                "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
                out("=r") first,
                out("=r") second,
                out("=r") third,
                out("=r") fourth,
                in("l") source,
                clobber("memory"),
            );
        }

        (first, second, third, fourth)
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
    unsafe fn load_u32x4_read_only(source: *const u32) -> (u32, u32, u32, u32) {
        let first: u32;
        let second: u32;
        let third: u32;
        let fourth: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];",
                out("=r") first,
                out("=r") second,
                out("=r") third,
                out("=r") fourth,
                in("l") source,
                clobber("memory"),
            );
        }

        (first, second, third, fourth)
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
    #[allow(clippy::too_many_arguments)]
    unsafe fn decode_phase<const PHASE: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        gate_row: usize,
        up_row: usize,
        lane: usize,
        gate_accumulators: &mut [f32; 4],
        up_accumulators: &mut [f32; 4],
    ) {
        let group = PHASE * 32 + lane;
        let gate_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset(gate_row, group)))
        };
        let up_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset(up_row, group)))
        };
        let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_reciprocal;
        let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_reciprocal;
        let row_words = CODE_BYTES_PER_ROW / 4;
        let phase_words = (32 * GROUP_K / 2) / 4;
        let lane_words = GROUP_K / 2 / 4;
        let gate_source = unsafe {
            weight_codes.add(gate_row * row_words + PHASE * phase_words + lane * lane_words)
        };
        let up_source = unsafe {
            weight_codes.add(up_row * row_words + PHASE * phase_words + lane * lane_words)
        };
        let gate_words = unsafe { load_u32x2_read_only(gate_source) };
        let up_words = unsafe { load_u32x2_read_only(up_source) };
        let activation_source = unsafe { input.add(PHASE * 256 + lane * 8) };

        macro_rules! accumulate_pair {
            ($pair:literal, $chain0:literal, $chain1:literal) => {{
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
                let (gate_code0, gate_code1) = e2m1x2_to_f32(gate_packed);
                let (up_code0, up_code1) = e2m1x2_to_f32(up_packed);
                let activation_bits = unsafe { *activation_source.add($pair) };
                let (activation0, activation1) = convert::cvt_f32x2_bf16x2(activation_bits);

                gate_accumulators[$chain0] = float::fma_rn_f32(
                    gate_code0 * gate_coefficient,
                    activation0,
                    gate_accumulators[$chain0],
                );
                gate_accumulators[$chain1] = float::fma_rn_f32(
                    gate_code1 * gate_coefficient,
                    activation1,
                    gate_accumulators[$chain1],
                );
                up_accumulators[$chain0] = float::fma_rn_f32(
                    up_code0 * up_coefficient,
                    activation0,
                    up_accumulators[$chain0],
                );
                up_accumulators[$chain1] = float::fma_rn_f32(
                    up_code1 * up_coefficient,
                    activation1,
                    up_accumulators[$chain1],
                );
            }};
        }

        accumulate_pair!(0, 0, 1);
        accumulate_pair!(1, 2, 3);
        accumulate_pair!(2, 0, 1);
        accumulate_pair!(3, 2, 3);
        accumulate_pair!(4, 0, 1);
        accumulate_pair!(5, 2, 3);
        accumulate_pair!(6, 0, 1);
        accumulate_pair!(7, 2, 3);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn small_t_phase<const TOKENS: usize>(
        phase: usize,
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        shared: *mut u32,
        gate_row: usize,
        up_row: usize,
        tid: usize,
        lane: usize,
        gate_accumulators: &mut [f32; TOKENS],
        up_accumulators: &mut [f32; TOKENS],
    ) {
        const PACKS_PER_TOKEN_PHASE: usize = 512 / 8;
        let mut task = tid;

        while task < TOKENS * PACKS_PER_TOKEN_PHASE {
            let token = task / PACKS_PER_TOKEN_PHASE;
            let pack = task - token * PACKS_PER_TOKEN_PHASE;
            let source = unsafe { input.add(token * (HIDDEN / 2) + phase * 256 + pack * 4) };
            let values = unsafe { load_u32x4_read_only(source) };
            let destination = unsafe { shared.add(token * 256 + pack * 4).cast::<[u32; 4]>() };

            unsafe { destination.write([values.0, values.1, values.2, values.3]) };
            task += A16_WARPS * 32;
        }
        thread::sync_threads();

        let group = phase * 32 + lane;
        let gate_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset(gate_row, group)))
        };
        let up_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset(up_row, group)))
        };
        let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_reciprocal;
        let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_reciprocal;
        let row_words = CODE_BYTES_PER_ROW / 4;
        let phase_words = (32 * GROUP_K / 2) / 4;
        let lane_words = GROUP_K / 2 / 4;
        let gate_source = unsafe {
            weight_codes.add(gate_row * row_words + phase * phase_words + lane * lane_words)
        };
        let up_source = unsafe {
            weight_codes.add(up_row * row_words + phase * phase_words + lane * lane_words)
        };
        let gate_words = unsafe { load_u32x2_read_only(gate_source) };
        let up_words = unsafe { load_u32x2_read_only(up_source) };

        macro_rules! accumulate_pair {
            ($pair:literal) => {{
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
                let (gate_weight0, gate_weight1) = e2m1x2_to_f32(gate_packed);
                let (up_weight0, up_weight1) = e2m1x2_to_f32(up_packed);

                macro_rules! accumulate_token {
                    ($token:literal) => {
                        if $token < TOKENS {
                            let activation_bits =
                                unsafe { *shared.add($token * 256 + lane * 8 + $pair) };
                            let (activation0, activation1) =
                                convert::cvt_f32x2_bf16x2(activation_bits);

                            gate_accumulators[$token] = float::fma_rn_f32(
                                gate_weight0 * gate_coefficient,
                                activation0,
                                gate_accumulators[$token],
                            );
                            gate_accumulators[$token] = float::fma_rn_f32(
                                gate_weight1 * gate_coefficient,
                                activation1,
                                gate_accumulators[$token],
                            );
                            up_accumulators[$token] = float::fma_rn_f32(
                                up_weight0 * up_coefficient,
                                activation0,
                                up_accumulators[$token],
                            );
                            up_accumulators[$token] = float::fma_rn_f32(
                                up_weight1 * up_coefficient,
                                activation1,
                                up_accumulators[$token],
                            );
                        }
                    };
                }

                accumulate_token!(0);
                accumulate_token!(1);
                accumulate_token!(2);
                accumulate_token!(3);
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
    }

    #[inline(always)]
    unsafe fn small_t_body<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
        shared: *mut u32,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let m_tile = block >> 4;
        let cta_in_tile = block & 15;
        let flat_pair = cta_in_tile * A16_WARPS + warp_index;
        let row_mod32 = flat_pair >> 2;
        let quartile = flat_pair & 3;
        let gate_row = m_tile * 128 + row_mod32 + quartile * 32;
        let up_row = gate_row + OUTPUT_ROWS;
        let mut gate_accumulators = [0.0f32; TOKENS];
        let mut up_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < 10 {
            unsafe {
                small_t_phase::<TOKENS>(
                    phase,
                    input,
                    weight_codes,
                    weight_scales,
                    weight_scale_reciprocal,
                    shared,
                    gate_row,
                    up_row,
                    tid,
                    lane,
                    &mut gate_accumulators,
                    &mut up_accumulators,
                );
            }
            phase += 1;
        }

        macro_rules! finish_token {
            ($token:literal) => {
                if $token < TOKENS {
                    let gate = reduce_sum_lane0(gate_accumulators[$token]);
                    let up = reduce_sum_lane0(up_accumulators[$token]);

                    if lane == 0 {
                        unsafe {
                            *output.add($token * OUTPUT_ROWS + gate_row) =
                                tcgen05::cvt_f32x2_bf16x2(silu(gate) * up, 0.0) as u16;
                        }
                    }
                }
            };
        }

        finish_token!(0);
        finish_token!(1);
        finish_token!(2);
        finish_token!(3);
    }

    #[inline(always)]
    fn silu(value: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E));

        value * sigmoid
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

    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_swiglu_a16_t1(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let m_tile = block >> 4;
        let cta_in_tile = block & 15;
        let flat_pair = cta_in_tile * A16_WARPS + warp_index;
        let row_mod32 = flat_pair >> 2;
        let quartile = flat_pair & 3;
        let gate_row = m_tile * 128 + row_mod32 + quartile * 32;
        let up_row = gate_row + OUTPUT_ROWS;
        let mut gate_accumulators = [0.0f32; 4];
        let mut up_accumulators = [0.0f32; 4];

        macro_rules! phase {
            ($phase:literal) => {
                unsafe {
                    decode_phase::<$phase>(
                        input,
                        weight_codes,
                        weight_scales,
                        weight_scale_reciprocal,
                        gate_row,
                        up_row,
                        lane,
                        &mut gate_accumulators,
                        &mut up_accumulators,
                    );
                }
            };
        }

        phase!(0);
        phase!(1);
        phase!(2);
        phase!(3);
        phase!(4);
        phase!(5);
        phase!(6);
        phase!(7);
        phase!(8);
        phase!(9);

        let mut gate = gate_accumulators[0]
            + gate_accumulators[1]
            + gate_accumulators[2]
            + gate_accumulators[3];
        let mut up =
            up_accumulators[0] + up_accumulators[1] + up_accumulators[2] + up_accumulators[3];
        gate = reduce_sum_lane0(gate);
        up = reduce_sum_lane0(up);

        if lane == 0 {
            unsafe {
                *output.add(gate_row) = tcgen05::cvt_f32x2_bf16x2(silu(gate) * up, 0.0) as u16;
            }
        }
    }

    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_swiglu_a16_t2(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, A16_T2_SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            small_t_body::<2>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_swiglu_a16_t3(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, A16_T4_SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            small_t_body::<3>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_swiglu_a16_t4(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, A16_T4_SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            small_t_body::<4>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    #[inline(always)]
    fn accumulate_max_abs(maximum: f32, value: f32) -> f32 {
        let mut result = maximum;

        unsafe {
            ptx_asm!(
                "{ .reg .f32 absolute; abs.f32 absolute, %1; max.f32 %0, %0, absolute; }",
                inout("+f") result,
                in("f") value,
                options(register_only),
            );
        }

        result
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pack_e2m1x16(
        v0: f32,
        v1: f32,
        v2: f32,
        v3: f32,
        v4: f32,
        v5: f32,
        v6: f32,
        v7: f32,
        v8: f32,
        v9: f32,
        v10: f32,
        v11: f32,
        v12: f32,
        v13: f32,
        v14: f32,
        v15: f32,
    ) -> (u32, u32) {
        let codes_lo: u32;
        let codes_hi: u32;

        unsafe {
            ptx_asm!(
                "{ .reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3; \
                   .reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7; \
                   cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2; \
                   cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4; \
                   cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6; \
                   cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8; \
                   cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10; \
                   cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12; \
                   cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14; \
                   cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16; \
                   mov.b32 %0, {b0, b1, b2, b3}; \
                   mov.b32 %1, {b4, b5, b6, b7}; }",
                out("=r") codes_lo,
                out("=r") codes_hi,
                in("f") v0,
                in("f") v1,
                in("f") v2,
                in("f") v3,
                in("f") v4,
                in("f") v5,
                in("f") v6,
                in("f") v7,
                in("f") v8,
                in("f") v9,
                in("f") v10,
                in("f") v11,
                in("f") v12,
                in("f") v13,
                in("f") v14,
                in("f") v15,
                options(register_only),
            );
        }

        (codes_lo, codes_hi)
    }

    #[inline(always)]
    unsafe fn quantize_body<const TOKENS: usize>(
        task: usize,
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        if task >= TOKENS * GROUPS_PER_ROW {
            return;
        }

        let token = task / GROUPS_PER_ROW;
        let group = task - token * GROUPS_PER_ROW;
        let source = unsafe { input.add(token * (HIDDEN / 2) + group * (GROUP_K / 2)) };
        let (p0, p1, p2, p3) = unsafe { load_u32x4(source) };
        let (p4, p5, p6, p7) = unsafe { load_u32x4(source.add(4)) };
        let (v0, v1) = convert::cvt_f32x2_bf16x2(p0);
        let (v2, v3) = convert::cvt_f32x2_bf16x2(p1);
        let (v4, v5) = convert::cvt_f32x2_bf16x2(p2);
        let (v6, v7) = convert::cvt_f32x2_bf16x2(p3);
        let (v8, v9) = convert::cvt_f32x2_bf16x2(p4);
        let (v10, v11) = convert::cvt_f32x2_bf16x2(p5);
        let (v12, v13) = convert::cvt_f32x2_bf16x2(p6);
        let (v14, v15) = convert::cvt_f32x2_bf16x2(p7);
        let values = [
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        ];
        let mut max_abs = 0.0f32;
        let mut index = 0usize;

        while index < GROUP_K {
            max_abs = accumulate_max_abs(max_abs, values[index]);
            index += 1;
        }

        let scale_unencoded = float::div_rn_f32(input_scale_divisor * max_abs, 6.0);
        let encoded_pair = convert::cvt_rn_satfinite_e4m3x2_f32(scale_unencoded, scale_unencoded);
        let scale = encoded_pair as u8;
        let code_destination =
            unsafe { codes.add(token * (CODE_BYTES_PER_ROW / 4) + group * (GROUP_K / 8)) };

        if scale == 0 {
            unsafe {
                *code_destination = 0;
                *code_destination.add(1) = 0;
                *scales.add(task) = 0;
            }
            return;
        }

        let decoded_scale = e4m3_to_f32(scale);
        let (codes_lo, codes_hi) = pack_e2m1x16(
            float::div_rn_f32(v0 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v1 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v2 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v3 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v4 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v5 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v6 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v7 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v8 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v9 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v10 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v11 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v12 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v13 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v14 * input_scale_divisor, decoded_scale),
            float::div_rn_f32(v15 * input_scale_divisor, decoded_scale),
        );

        unsafe {
            *code_destination = codes_lo;
            *code_destination.add(1) = codes_hi;
            *scales.add(task) = scale;
        }
    }

    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_quantize<const TOKENS: usize>(
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        unsafe {
            quantize_body::<TOKENS>(
                thread::index_1d().get(),
                input,
                codes,
                scales,
                input_scale_divisor,
            );
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn stage_tile<const TOKENS: usize>(
        shared: *mut u8,
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        stage: usize,
        k_tile: usize,
        token_begin: usize,
        row_begin: usize,
        tid: usize,
    ) {
        if tid < SMALL_BLOCK_M * SEGMENTS_PER_ROW {
            let row = tid / SEGMENTS_PER_ROW;
            let segment = tid - row * SEGMENTS_PER_ROW;
            let valid = token_begin + row < TOKENS;
            let source_token = if valid { token_begin + row } else { 0 };
            let physical = swizzled_byte(row, segment * 16);
            let destination = unsafe {
                shared
                    .add(A_CODE_OFFSET + (stage * BLOCK_M + row) * CODE_ROW_BYTES + physical)
                    .cast::<u32>()
            };
            let source = unsafe {
                activation_codes.add(
                    source_token * (CODE_BYTES_PER_ROW / 4)
                        + k_tile * (CODE_ROW_BYTES / 4)
                        + segment * 4,
                )
            };

            unsafe {
                cp_async_cg_zfill_16(destination, source.cast::<u8>(), if valid { 16 } else { 0 });
            }
        }

        if tid < SMALL_BLOCK_M {
            let valid = token_begin + tid < TOKENS;
            let source_token = if valid { token_begin + tid } else { 0 };
            let destination = unsafe {
                shared
                    .add(A_SCALE_OFFSET + (stage * BLOCK_M + tid) * K64_PER_STAGE * 4)
                    .cast::<u32>()
            };
            let source = unsafe {
                activation_scales.add(source_token * GROUPS_PER_ROW + k_tile * K64_PER_STAGE * 4)
            };

            unsafe {
                cp_async_cg_zfill_16(destination, source, if valid { 16 } else { 0 });
            }
        }

        let mut task = tid;
        while task < BLOCK_N * SEGMENTS_PER_ROW {
            let row = task / SEGMENTS_PER_ROW;
            let segment = task - row * SEGMENTS_PER_ROW;
            let parent_row = weight_row(row_begin, row);
            let physical = swizzled_byte(row, segment * 16);
            let destination = unsafe {
                shared
                    .add(B_CODE_OFFSET + (stage * BLOCK_N + row) * CODE_ROW_BYTES + physical)
                    .cast::<u32>()
            };
            let source = unsafe {
                weight_codes.add(
                    parent_row * (CODE_BYTES_PER_ROW / 4)
                        + k_tile * (CODE_ROW_BYTES / 4)
                        + segment * 4,
                )
            };

            unsafe { cp_async_cg_16(destination, source) };
            task += W4A4_THREADS as usize;
        }

        let mut scale_task = tid;
        while scale_task < BLOCK_N * K64_PER_STAGE {
            let row = scale_task / K64_PER_STAGE;
            let local_k64 = scale_task - row * K64_PER_STAGE;
            let parent_row = weight_row(row_begin, row);
            let global_k64 = k_tile * K64_PER_STAGE + local_k64;
            let destination = unsafe {
                shared
                    .add(B_SCALE_OFFSET + (stage * BLOCK_N * K64_PER_STAGE + scale_task) * 4)
                    .cast::<u32>()
            };
            let source = unsafe { weight_scales.add(weight_scale_offset(parent_row, global_k64)) };

            unsafe { cp_async_ca_4(destination, source.cast::<u32>()) };
            scale_task += W4A4_THREADS as usize;
        }
    }

    #[inline(always)]
    fn mma_nvfp4(
        accumulators: &mut [f32; 4],
        a: [u32; 4],
        b: [u32; 2],
        scale_a: u32,
        scale_b: u32,
    ) {
        let scale_block_id = 0u16;
        let scale_thread_id = 0u16;

        unsafe {
            ptx_asm!(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.\
                 m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
                 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, \
                 {%10}, {%11,%12}, {%13}, {%14,%15};",
                inout("+f") accumulators[0],
                inout("+f") accumulators[1],
                inout("+f") accumulators[2],
                inout("+f") accumulators[3],
                in("r") a[0],
                in("r") a[1],
                in("r") a[2],
                in("r") a[3],
                in("r") b[0],
                in("r") b[1],
                in("r") scale_a,
                in("h") scale_block_id,
                in("h") scale_thread_id,
                in("r") scale_b,
                in("h") scale_block_id,
                in("h") scale_thread_id,
                options(register_only),
            );
        }
    }

    #[inline(always)]
    unsafe fn w4a4_body<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(SHARED).cast::<u8>();
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        let warp_m = warp / WARPS_N;
        let warp_n = warp - warp_m * WARPS_N;
        let token_begin = thread::blockIdx_y() as usize * SMALL_BLOCK_M;
        let row_begin = thread::blockIdx_x() as usize * ROWS_PER_BRANCH;
        let mut stage = 0usize;

        while stage < STAGES {
            unsafe {
                stage_tile::<TOKENS>(
                    shared,
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    stage,
                    stage,
                    token_begin,
                    row_begin,
                    tid,
                );
                cp_async_commit_group();
            }
            stage += 1;
        }

        let mut accumulators = [[0.0f32; 4]; MMA_N];
        let a_matrix = lane >> 3;
        let a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
        let a_column_byte = (a_matrix >> 1) * 16;
        let b_row_offset = lane & 7;
        let b_column_byte = ((lane >> 3) & 1) * 16;
        let scale_a_row_offset = ((lane & 1) << 3) | (lane >> 2);
        let scale_b_row_offset = lane >> 2;
        let mut k_tile = 0usize;

        while k_tile < K_TILES {
            stage = k_tile % STAGES;
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();

            let mut local_k64 = 0usize;
            while local_k64 < K64_PER_STAGE {
                let a_row = warp_m * WARP_M + a_row_offset;
                let a_logical_byte = local_k64 * 32 + a_column_byte;
                let a_physical_byte = swizzled_byte(a_row, a_logical_byte);
                let a_address = unsafe {
                    shared
                        .add(
                            A_CODE_OFFSET
                                + (stage * BLOCK_M + a_row) * CODE_ROW_BYTES
                                + a_physical_byte,
                        )
                        .cast::<u32>()
                };
                let a_fragments = unsafe { wmma::ldmatrix_x4(a_address) };
                let scale_a_row = warp_m * WARP_M + scale_a_row_offset;
                let scale_a = unsafe {
                    *shared
                        .add(
                            A_SCALE_OFFSET
                                + (stage * BLOCK_M + scale_a_row) * K64_PER_STAGE * 4
                                + local_k64 * 4,
                        )
                        .cast::<u32>()
                };
                let mut mma_n = 0usize;

                while mma_n < MMA_N {
                    let b_row = warp_n * WARP_N + mma_n * 8 + b_row_offset;
                    let b_logical_byte = local_k64 * 32 + b_column_byte;
                    let b_physical_byte = swizzled_byte(b_row, b_logical_byte);
                    let b_address = unsafe {
                        shared
                            .add(
                                B_CODE_OFFSET
                                    + (stage * BLOCK_N + b_row) * CODE_ROW_BYTES
                                    + b_physical_byte,
                            )
                            .cast::<u32>()
                    };
                    let b_fragments = unsafe { wmma::ldmatrix_x2(b_address) };
                    let scale_b_row = warp_n * WARP_N + mma_n * 8 + scale_b_row_offset;
                    let scale_b = unsafe {
                        *shared
                            .add(
                                B_SCALE_OFFSET
                                    + (stage * BLOCK_N + scale_b_row) * K64_PER_STAGE * 4
                                    + local_k64 * 4,
                            )
                            .cast::<u32>()
                    };

                    mma_nvfp4(
                        &mut accumulators[mma_n],
                        a_fragments,
                        b_fragments,
                        scale_a,
                        scale_b,
                    );
                    mma_n += 1;
                }
                local_k64 += 1;
            }

            thread::sync_threads();
            let next_k_tile = k_tile + STAGES;
            if next_k_tile < K_TILES {
                unsafe {
                    stage_tile::<TOKENS>(
                        shared,
                        activation_codes,
                        activation_scales,
                        weight_codes,
                        weight_scales,
                        stage,
                        next_k_tile,
                        token_begin,
                        row_begin,
                        tid,
                    );
                }
            }
            unsafe { cp_async_commit_group() };
            k_tile += 1;
        }

        let accumulator_row = lane >> 2;
        let accumulator_col = 2 * (lane & 3);
        let token0 = warp_m * WARP_M + accumulator_row;
        let token1 = token0 + 8;
        let output_shared = shared.cast::<u16>();
        let mut mma_n = 0usize;

        while mma_n < MMA_N {
            let local_row0 = warp_n * WARP_N + mma_n * 8 + accumulator_col;
            let destination0 = unsafe {
                output_shared
                    .add(token0 * OUTPUT_STRIDE + local_row0)
                    .cast::<u32>()
            };
            let destination1 = unsafe {
                output_shared
                    .add(token1 * OUTPUT_STRIDE + local_row0)
                    .cast::<u32>()
            };
            let values = accumulators[mma_n];

            unsafe {
                *destination0 = tcgen05::cvt_f32x2_bf16x2(values[0] * alpha, values[1] * alpha);
                *destination1 = tcgen05::cvt_f32x2_bf16x2(values[2] * alpha, values[3] * alpha);
            }
            mma_n += 1;
        }
        thread::sync_threads();

        const VECTORS_PER_ROW: usize = ROWS_PER_BRANCH / 8;
        if tid < SMALL_BLOCK_M * VECTORS_PER_ROW {
            let local_token = tid / VECTORS_PER_ROW;
            let token = token_begin + local_token;

            if token < TOKENS {
                let row_vector = tid - local_token * VECTORS_PER_ROW;
                let gate = unsafe {
                    output_shared
                        .add(local_token * OUTPUT_STRIDE + row_vector * 8)
                        .cast::<[u32; 4]>()
                        .read()
                };
                let up = unsafe {
                    output_shared
                        .add(local_token * OUTPUT_STRIDE + ROWS_PER_BRANCH + row_vector * 8)
                        .cast::<[u32; 4]>()
                        .read()
                };
                let mut result = [0u32; 4];
                let mut pair = 0usize;

                while pair < 4 {
                    let gate_values = convert::cvt_f32x2_bf16x2(gate[pair]);
                    let up_values = convert::cvt_f32x2_bf16x2(up[pair]);
                    result[pair] = tcgen05::cvt_f32x2_bf16x2(
                        silu(gate_values.0) * up_values.0,
                        silu(gate_values.1) * up_values.1,
                    );
                    pair += 1;
                }

                let destination = unsafe {
                    output
                        .add(token * OUTPUT_ROWS + row_begin + row_vector * 8)
                        .cast::<[u32; 4]>()
                };

                unsafe { destination.write(result) };
            }
        }
    }

    #[kernel]
    #[launch_bounds(384, 2)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_swiglu_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        unsafe {
            w4a4_body::<TOKENS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                alpha,
            );
        }
    }
}

fn a16_config() -> LaunchConfig1D {
    // One eight-warp CTA emits eight gate/up row pairs, so 17,408 outputs
    // require exactly 2,176 CTAs without a tail route.
    LaunchConfig1D::new((OUTPUT_ROWS / A16_WARPS) as u32, A16_THREADS, 0)
}

struct PreparedW4a4Route<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__nvfp4_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__nvfp4_swiglu_w4a4_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedW4a4Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        // One quantizer thread owns one 16-value activation group; 256 groups
        // per CTA amortize the reduction while preserving the source grouping.
        let quantize_blocks = (TOKENS * GROUPS_PER_ROW).div_ceil(256);
        let quantize_blocks = u32::try_from(quantize_blocks)
            .map_err(|_| GpuError::invalid_launch("NVFP4 quantization grid exceeds CUDA width"))?;
        // Each projection CTA emits 32 fused gate/up rows, yielding 544 exact
        // CTAs and no output-row padding.
        let projection_blocks = u32::try_from(OUTPUT_ROWS / ROWS_PER_BRANCH)
            .map_err(|_| GpuError::invalid_launch("NVFP4 projection grid exceeds CUDA width"))?;
        let quantize = module
            .prepare_nvfp4_quantize::<TOKENS>(LaunchConfig1D::new(quantize_blocks, 256, 0))
            .map_err(|source| {
                GpuError::launch("preparing NVFP4 activation quantization", source)
            })?;
        let projection = module
            .prepare_nvfp4_swiglu_w4a4::<TOKENS>(LaunchConfig2D::new(
                (projection_blocks, 1),
                (W4A4_THREADS, 1),
                0,
            ))
            .map_err(|source| GpuError::launch("preparing NVFP4 W4A4 SwiGLU", source))?;

        Ok(Self {
            quantize,
            projection,
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
        weight_codes: *const u8,
        weight_scales: *const u8,
        input_scale_divisor: f32,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .nvfp4_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u32>(),
                activation_scales,
                input_scale_divisor,
            )
            .map_err(|source| {
                GpuError::launch("launching NVFP4 activation quantization", source)
            })?;
        module
            .nvfp4_swiglu_w4a4::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                1.0 / (input_scale_divisor * weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching NVFP4 W4A4 SwiGLU", source))
    }
}

/// PTX symbols retained for every admitted NVFP4 SwiGLU schedule.
pub(crate) fn nvfp4_swiglu_ptx_names() -> [&'static str; 14] {
    [
        "nvfp4_swiglu_a16_t1",
        "nvfp4_swiglu_a16_t2",
        "nvfp4_swiglu_a16_t3",
        "nvfp4_swiglu_a16_t4",
        kernels::nvfp4_quantize_ptx_name::<1>(),
        kernels::nvfp4_quantize_ptx_name::<5>(),
        kernels::nvfp4_quantize_ptx_name::<6>(),
        kernels::nvfp4_quantize_ptx_name::<7>(),
        kernels::nvfp4_quantize_ptx_name::<8>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<1>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<5>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<6>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<7>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<8>(),
    ]
}

/// Prepared A16 and W4A4 routes for the exact NVFP4 MLP gate/up operation.
pub struct Nvfp4SwiGluOp {
    module: kernels::LoadedModule,
    a16_t1: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t1_CudaKernel>,
    a16_t2: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t2_CudaKernel>,
    a16_t3: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t3_CudaKernel>,
    a16_t4: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t4_CudaKernel>,
    w4a4_b1: PreparedW4a4Route<1>,
    w4a4_b5: PreparedW4a4Route<5>,
    w4a4_b6: PreparedW4a4Route<6>,
    w4a4_b7: PreparedW4a4Route<7>,
    w4a4_b8: PreparedW4a4Route<8>,
}

impl Nvfp4SwiGluOp {
    /// Loads the embedded SM120 module and prepares every admitted decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = nvfp4_swiglu_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the NVFP4 SwiGLU module", source))?;

        Ok(Self {
            a16_t1: module
                .prepare_nvfp4_swiglu_a16_t1(a16_config())
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=1", source))?,
            a16_t2: module
                .prepare_nvfp4_swiglu_a16_t2(a16_config())
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=2", source))?,
            a16_t3: module
                .prepare_nvfp4_swiglu_a16_t3(a16_config())
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=3", source))?,
            a16_t4: module
                .prepare_nvfp4_swiglu_a16_t4(a16_config())
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=4", source))?,
            w4a4_b1: PreparedW4a4Route::prepare(&module)?,
            w4a4_b5: PreparedW4a4Route::prepare(&module)?,
            w4a4_b6: PreparedW4a4Route::prepare(&module)?,
            w4a4_b7: PreparedW4a4Route::prepare(&module)?,
            w4a4_b8: PreparedW4a4Route::prepare(&module)?,
            module,
        })
    }

    /// Executes the retained production route for an exact `B=1..=8`.
    ///
    /// B=1 and B=5..8 dynamically quantize the input and use W4A4 MMA; B=2..4
    /// preserve the represented BF16 activation and use the A16 schedule.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 5_120` BF16 values; activation scratch covers
    /// `batch * 2_560` code bytes and `batch * 320` scale bytes; `weight_codes`
    /// covers the fused packed `[34_816, 5_120]` plane; `weight_scales` covers
    /// its swizzled `[34_816, 320]` plane; and `output` covers
    /// `batch * 17_408` BF16 values. Four-byte-loaded planes are four-byte
    /// aligned. Divisors are finite and positive. All allocations belong to
    /// `stream`'s context, remain live through completion, and do not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut u8,
        weight_codes: *const u8,
        weight_scales: *const u8,
        input_scale_divisor: f32,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !(1..=MAX_BATCH).contains(&batch) {
            return Err(GpuError::invalid_launch(format!(
                "NVFP4 SwiGLU batch {batch} is not an exact B=1..=8 route"
            )));
        }
        if !input_scale_divisor.is_finite() || input_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(
                "NVFP4 input scale divisor must be finite and positive",
            ));
        }
        if !weight_scale_divisor.is_finite() || weight_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(
                "NVFP4 weight scale divisor must be finite and positive",
            ));
        }

        macro_rules! w4a4 {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        activation_codes,
                        activation_scales,
                        weight_codes,
                        weight_scales,
                        input_scale_divisor,
                        weight_scale_divisor,
                        output,
                    )
                }
            };
        }

        match batch {
            1 => w4a4!(w4a4_b1),
            2..=4 => unsafe {
                self.launch_a16(
                    stream,
                    batch,
                    input,
                    weight_codes,
                    weight_scales,
                    weight_scale_divisor,
                    output,
                )
            },
            5 => w4a4!(w4a4_b5),
            6 => w4a4!(w4a4_b6),
            7 => w4a4!(w4a4_b7),
            8 => w4a4!(w4a4_b8),
            _ => unreachable!("batch range was validated above"),
        }
    }

    /// Executes the retained A16 comparison route for exact `B=1..=4`.
    ///
    /// # Safety
    ///
    /// The input, weight, scale, output, alignment, lifetime, context, and
    /// overlap requirements are the corresponding subset of [`Self::launch`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_a16(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !weight_scale_divisor.is_finite() || weight_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(
                "NVFP4 weight scale divisor must be finite and positive",
            ));
        }

        let reciprocal = 1.0 / weight_scale_divisor;
        macro_rules! launch {
            ($method:ident, $prepared:ident, $label:literal) => {
                self.module
                    .$method(
                        stream,
                        &self.$prepared,
                        input.cast::<u32>(),
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        reciprocal,
                        output,
                    )
                    .map_err(|source| GpuError::launch($label, source))
            };
        }

        match batch {
            1 => launch!(nvfp4_swiglu_a16_t1, a16_t1, "launching NVFP4 A16 B=1"),
            2 => launch!(nvfp4_swiglu_a16_t2, a16_t2, "launching NVFP4 A16 B=2"),
            3 => launch!(nvfp4_swiglu_a16_t3, a16_t3, "launching NVFP4 A16 B=3"),
            4 => launch!(nvfp4_swiglu_a16_t4, a16_t4, "launching NVFP4 A16 B=4"),
            _ => Err(GpuError::invalid_launch(format!(
                "NVFP4 A16 batch {batch} is not an exact B=1..=4 route"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, nvfp4_swiglu_ptx_names};

    #[test]
    fn inventory_covers_retained_decode_routes() {
        let names = nvfp4_swiglu_ptx_names();

        assert_eq!(MAX_BATCH, 8);
        assert_eq!(names.len(), 14);
        assert_eq!(names.iter().filter(|name| name.contains("a16")).count(), 4);
        assert_eq!(
            names
                .iter()
                .filter(|name| name.contains("quantize"))
                .count(),
            5
        );
        assert_eq!(names.iter().filter(|name| name.contains("w4a4")).count(), 5);
    }
}

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
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * OUTPUT_ROWS;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = HIDDEN / GROUP_K;

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
const _: () = assert!(GROUPS_PER_ROW == 320);
const _: () = assert!(Qwen35_9B::HIDDEN == 4_096);
const _: () = assert!(Qwen35_9B::INTERMEDIATE == 12_288);
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
    fn weight_row<A: Arch>(row_begin: usize, local_row: usize) -> usize {
        row_begin
            + (local_row & (ROWS_PER_BRANCH - 1))
            + if local_row >= ROWS_PER_BRANCH {
                A::INTERMEDIATE
            } else {
                0
            }
    }

    #[inline(always)]
    fn weight_scale_offset<A: Arch>(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;
        let scale_tiles_per_row = A::HIDDEN / GROUP_K / 4;

        (persistent_tile * scale_tiles_per_row + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset<A: Arch>(parent_row: usize, group: usize) -> usize {
        weight_scale_offset::<A>(parent_row, group >> 2) + (group & 3)
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
    unsafe fn decode_phase<A: Arch, const PHASE: usize>(
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
            load_u8_read_only(weight_scales.add(weight_group_scale_offset::<A>(gate_row, group)))
        };
        let up_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset::<A>(up_row, group)))
        };
        let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_reciprocal;
        let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_reciprocal;
        let row_words = (A::HIDDEN / 2) / 4;
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
    unsafe fn small_t_phase<A: Arch, const TOKENS: usize>(
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
            let source = unsafe { input.add(token * (A::HIDDEN / 2) + phase * 256 + pack * 4) };
            let values = unsafe { load_u32x4_read_only(source) };
            let destination = unsafe { shared.add(token * 256 + pack * 4).cast::<[u32; 4]>() };

            unsafe { destination.write([values.0, values.1, values.2, values.3]) };
            task += A16_WARPS * 32;
        }
        thread::sync_threads();

        let group = phase * 32 + lane;
        let gate_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset::<A>(gate_row, group)))
        };
        let up_scale = unsafe {
            load_u8_read_only(weight_scales.add(weight_group_scale_offset::<A>(up_row, group)))
        };
        let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_reciprocal;
        let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_reciprocal;
        let row_words = (A::HIDDEN / 2) / 4;
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
    unsafe fn small_t_body<A: Arch, const TOKENS: usize>(
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
        let up_row = gate_row + A::INTERMEDIATE;
        let mut gate_accumulators = [0.0f32; TOKENS];
        let mut up_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < A::HIDDEN / 512 {
            unsafe {
                small_t_phase::<A, TOKENS>(
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
                            *output.add($token * A::INTERMEDIATE + gate_row) =
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
                    decode_phase::<Qwen38_27B, $phase>(
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
            small_t_body::<Qwen38_27B, 2>(
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
            small_t_body::<Qwen38_27B, 3>(
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
            small_t_body::<Qwen38_27B, 4>(
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
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_swiglu_a16_t1(
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
        let up_row = gate_row + Qwen35_9B::INTERMEDIATE;
        let mut gate_accumulators = [0.0f32; 4];
        let mut up_accumulators = [0.0f32; 4];

        macro_rules! phase {
            ($phase:literal) => {
                unsafe {
                    decode_phase::<Qwen35_9B, $phase>(
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
    pub fn qwen35_nvfp4_swiglu_a16_t2(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, A16_T2_SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            small_t_body::<Qwen35_9B, 2>(
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
    pub fn qwen35_nvfp4_swiglu_a16<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, A16_T4_SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            small_t_body::<Qwen35_9B, TOKENS>(
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
    unsafe fn quantize_body<A: Arch, const TOKENS: usize>(
        task: usize,
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        let groups_per_row = A::HIDDEN / GROUP_K;
        if task >= TOKENS * groups_per_row {
            return;
        }

        let token = task / groups_per_row;
        let group = task - token * groups_per_row;
        let source = unsafe { input.add(token * (A::HIDDEN / 2) + group * (GROUP_K / 2)) };
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
            unsafe { codes.add(token * (A::HIDDEN / 8) + group * (GROUP_K / 8)) };

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
            quantize_body::<Qwen38_27B, TOKENS>(
                thread::index_1d().get(),
                input,
                codes,
                scales,
                input_scale_divisor,
            );
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
    pub fn qwen35_nvfp4_quantize<const TOKENS: usize>(
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        unsafe {
            quantize_body::<Qwen35_9B, TOKENS>(
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
    unsafe fn stage_tile<A: Arch, const TOKENS: usize>(
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
                    source_token * (A::HIDDEN / 8) + k_tile * (CODE_ROW_BYTES / 4) + segment * 4,
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
                activation_scales
                    .add(source_token * (A::HIDDEN / GROUP_K) + k_tile * K64_PER_STAGE * 4)
            };

            unsafe {
                cp_async_cg_zfill_16(destination, source, if valid { 16 } else { 0 });
            }
        }

        let mut task = tid;
        while task < BLOCK_N * SEGMENTS_PER_ROW {
            let row = task / SEGMENTS_PER_ROW;
            let segment = task - row * SEGMENTS_PER_ROW;
            let parent_row = weight_row::<A>(row_begin, row);
            let physical = swizzled_byte(row, segment * 16);
            let destination = unsafe {
                shared
                    .add(B_CODE_OFFSET + (stage * BLOCK_N + row) * CODE_ROW_BYTES + physical)
                    .cast::<u32>()
            };
            let source = unsafe {
                weight_codes
                    .add(parent_row * (A::HIDDEN / 8) + k_tile * (CODE_ROW_BYTES / 4) + segment * 4)
            };

            unsafe { cp_async_cg_16(destination, source) };
            task += W4A4_THREADS as usize;
        }

        let mut scale_task = tid;
        while scale_task < BLOCK_N * K64_PER_STAGE {
            let row = scale_task / K64_PER_STAGE;
            let local_k64 = scale_task - row * K64_PER_STAGE;
            let parent_row = weight_row::<A>(row_begin, row);
            let global_k64 = k_tile * K64_PER_STAGE + local_k64;
            let destination = unsafe {
                shared
                    .add(B_SCALE_OFFSET + (stage * BLOCK_N * K64_PER_STAGE + scale_task) * 4)
                    .cast::<u32>()
            };
            let source =
                unsafe { weight_scales.add(weight_scale_offset::<A>(parent_row, global_k64)) };

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
    unsafe fn w4a4_body<A: Arch, const TOKENS: usize>(
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
                stage_tile::<A, TOKENS>(
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

        while k_tile < A::HIDDEN / BLOCK_K {
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
            if next_k_tile < A::HIDDEN / BLOCK_K {
                unsafe {
                    stage_tile::<A, TOKENS>(
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
                        .add(token * A::INTERMEDIATE + row_begin + row_vector * 8)
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
            w4a4_body::<Qwen38_27B, TOKENS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                alpha,
            );
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
    pub fn qwen35_nvfp4_swiglu_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        unsafe {
            w4a4_body::<Qwen35_9B, TOKENS>(
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

fn a16_launch_config<A: Arch>() -> LaunchConfig1D {
    // One eight-warp CTA emits eight gate/up row pairs, so Qwen3.8's 17,408
    // outputs require exactly 2,176 CTAs and Qwen3.5's 12,288 exactly 1,536,
    // neither with a tail route; only the K loop shrinks from ten phases to
    // eight.
    LaunchConfig1D::new((A::INTERMEDIATE / A16_WARPS) as u32, A16_THREADS, 0)
}

fn w4a4_launch_configs<A: Arch>(
    tokens: usize,
    label: &str,
) -> GpuResult<(LaunchConfig1D, LaunchConfig2D)> {
    // One quantizer thread owns one 16-value activation group; 256 groups per
    // CTA amortize the reduction while preserving the source grouping. Both
    // admitted rows divide exactly: 320 groups on Qwen3.8, 256 on Qwen3.5.
    let quantize_blocks =
        u32::try_from((tokens * (A::HIDDEN / GROUP_K)).div_ceil(256)).map_err(|_| {
            GpuError::invalid_launch(format!("{label}NVFP4 quantization grid exceeds CUDA width"))
        })?;
    // Each projection CTA emits 32 fused gate/up rows, yielding 544 exact
    // column CTAs on Qwen3.8 and 384 on Qwen3.5 with no output-row padding.
    // Forty-eight token slots retain two CTAs per SM, so exact
    // T=32/64/128/1024 use 1/2/3/22 tiles and extending grid Y preserves every
    // m16n8k64 accumulation.
    let projection_blocks = u32::try_from(A::INTERMEDIATE / ROWS_PER_BRANCH).map_err(|_| {
        GpuError::invalid_launch(format!("{label}NVFP4 projection grid exceeds CUDA width"))
    })?;
    let token_tiles = u32::try_from(tokens.div_ceil(SMALL_BLOCK_M)).map_err(|_| {
        GpuError::invalid_launch(format!("{label}NVFP4 token grid exceeds CUDA height"))
    })?;

    Ok((
        LaunchConfig1D::new(quantize_blocks, 256, 0),
        LaunchConfig2D::new((projection_blocks, token_tiles), (W4A4_THREADS, 1), 0),
    ))
}

mod private {
    pub trait Sealed {}
}

/// The A16 entry one exact `B=1..=4` batch selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A16Slot {
    /// `B=1`.
    B1,
    /// `B=2`.
    B2,
    /// `B=3`.
    B3,
    /// `B=4`.
    B4,
}

/// One architecture's four prepared represented-BF16 A16 entries.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entries the module does not emit. The
/// four entries stay one route because `B=1..=4` is a single retained
/// comparison schedule that is prepared and qualified together.
pub trait SwiGluA16Routes<A: Arch>: Sized + private::Sealed {
    /// Prepares this architecture's four A16 entries.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches the A16 entry the slot selects.
    ///
    /// # Safety
    ///
    /// The pointers carry `Nvfp4SwiGluOp::launch_a16`'s contract unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        slot: A16Slot,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// One architecture's prepared quantization and W4A4 entries for exact rows.
///
/// Sealed for the same reason as [`SwiGluA16Routes`]. The pair stays one route
/// because the quantizer's output grouping is the projection's input contract
/// and the two are never prepared apart.
pub trait SwiGluW4a4Route<A: Arch>: Sized + private::Sealed {
    /// Prepares both entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Quantizes the represented activations and launches the W4A4 entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `Nvfp4SwiGluOp::launch`'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's NVFP4 gate/up routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 here never widens the artifact-level
/// admission bound. Each table names only the entries its own model emits,
/// which is what keeps the compiled inventory fixed while both prepared owners
/// share one wrapper.
pub trait Nvfp4SwiGluEntries<A: Arch>: private::Sealed {
    /// Prepared A16 entries for `B=1..=4`.
    type A16: SwiGluA16Routes<A>;
    /// Prepared W4A4 route for a batch both schedules project through W4A4.
    type W4a4Decode<const TOKENS: usize>: SwiGluW4a4Route<A>;
    /// Prepared W4A4 route for `B=2..=4`, where the two schedules disagree.
    ///
    /// Qwen3.5 keeps these as measured comparison routes reachable through
    /// `launch_w4a4`; Qwen3.8's production schedule takes A16 across that
    /// whole range and prepares no W4A4 entry there.
    type W4a4Crossover<const TOKENS: usize>: SwiGluW4a4Route<A>;
    /// Prepared W4A4 route for one exact prefill row count.
    type W4a4Prefill<const TOKENS: usize>: SwiGluW4a4Route<A>;

    /// Message prefix that keeps this architecture's launch errors distinct.
    const LABEL: &'static str;
    /// Context this architecture's module-load failure reports.
    const MODULE_CONTEXT: &'static str;
    /// Whether `launch` rejects a non-finite or non-positive activation-scale
    /// divisor even when the selected schedule keeps represented BF16
    /// activations and never reads it.
    ///
    /// Transcribed, not normalized: Qwen3.8's dispatch validated both divisors
    /// before selecting a schedule, and Qwen3.5's validated inside the selected
    /// route, so the two owners' rejection sets differ at their A16 batches.
    const A16_VALIDATES_INPUT_DIVISOR: bool;

    /// The route this architecture's production dispatch selects for `rows`.
    fn route(rows: usize) -> Option<SwiGluRoute>;

    /// This architecture's rejection of a row count its schedule does not admit.
    fn unadmitted_rows(rows: usize) -> GpuError;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// The compiled route one admitted row count selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwiGluRoute {
    /// Represented-BF16 A16 entry for `B=1..=4`.
    A16(A16Slot),
    /// W4A4 decode entry for `B=1`.
    W4a4B1,
    /// W4A4 decode entry for `B=2`.
    W4a4B2,
    /// W4A4 decode entry for `B=3`.
    W4a4B3,
    /// W4A4 decode entry for `B=4`.
    W4a4B4,
    /// W4A4 decode entry for `B=5`.
    W4a4B5,
    /// W4A4 decode entry for `B=6`.
    W4a4B6,
    /// W4A4 decode entry for `B=7`.
    W4a4B7,
    /// W4A4 decode entry for `B=8`.
    W4a4B8,
    /// W4A4 prefill entry for `T=32`.
    W4a4T32,
    /// W4A4 prefill entry for `T=64`.
    W4a4T64,
    /// W4A4 prefill entry for `T=128`.
    W4a4T128,
    /// W4A4 prefill entry for `T=1024`.
    W4a4T1024,
}

// The retained comparison schedule both owners expose through `launch_a16`:
// exact B=1..=4 and nothing else.
fn a16_slot(batch: usize) -> Option<A16Slot> {
    match batch {
        1 => Some(A16Slot::B1),
        2 => Some(A16Slot::B2),
        3 => Some(A16Slot::B3),
        4 => Some(A16Slot::B4),
        _ => None,
    }
}

// The W4A4 route an exact prefill row count selects; both owners admit
// exactly `PREFILL_ROWS` here.
fn prefill_route(rows: usize) -> Option<SwiGluRoute> {
    if !PREFILL_ROWS.contains(&rows) {
        return None;
    }

    match rows {
        32 => Some(SwiGluRoute::W4a4T32),
        64 => Some(SwiGluRoute::W4a4T64),
        128 => Some(SwiGluRoute::W4a4T128),
        1_024 => Some(SwiGluRoute::W4a4T1024),
        _ => unreachable!("PREFILL_ROWS admits only the exact T routes"),
    }
}

/// Prepared Qwen3.8 A16 entries for `B=1..=4`.
pub struct PreparedA16Routes {
    b1: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t1_CudaKernel>,
    b2: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t2_CudaKernel>,
    b3: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t3_CudaKernel>,
    b4: PreparedLaunch<kernels::__nvfp4_swiglu_a16_t4_CudaKernel>,
}

/// Prepared Qwen3.5 A16 entries for `B=1..=4`.
///
/// `B=1` and `B=2` keep their concrete entries; `B=3` and `B=4` share the
/// generic entry whose shared footprint they already qualified together.
pub struct PreparedQwen35A16Routes {
    b1: PreparedLaunch<kernels::__qwen35_nvfp4_swiglu_a16_t1_CudaKernel>,
    b2: PreparedLaunch<kernels::__qwen35_nvfp4_swiglu_a16_t2_CudaKernel>,
    b3: PreparedLaunch<kernels::__qwen35_nvfp4_swiglu_a16_CudaKernel<3>>,
    b4: PreparedLaunch<kernels::__qwen35_nvfp4_swiglu_a16_CudaKernel<4>>,
}

/// Prepared Qwen3.8 quantization and W4A4 entries for one exact row count.
pub struct PreparedW4a4Route<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__nvfp4_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__nvfp4_swiglu_w4a4_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 quantization and W4A4 entries for one exact row count.
pub struct PreparedQwen35W4a4Route<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__qwen35_nvfp4_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_swiglu_w4a4_CudaKernel<TOKENS>>,
}

/// Stands in for a batch an architecture's schedule never routes through W4A4.
///
/// It prepares and launches no entry, so an unrouted batch can never reach the
/// device and never enters the emitted inventory.
pub struct UnadmittedRoute;

impl private::Sealed for PreparedA16Routes {}
impl private::Sealed for PreparedQwen35A16Routes {}
impl<const TOKENS: usize> private::Sealed for PreparedW4a4Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35W4a4Route<TOKENS> {}
impl private::Sealed for UnadmittedRoute {}

// The Qwen3.8 A16 entries compile that model's exact row width into four
// concrete entries, so they stay bound to the sealed artifact-level
// architecture.
impl<A: Sm120Arch> SwiGluA16Routes<A> for PreparedA16Routes {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = a16_launch_config::<A>();

        Ok(Self {
            b1: module
                .prepare_nvfp4_swiglu_a16_t1(launch)
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=1", source))?,
            b2: module
                .prepare_nvfp4_swiglu_a16_t2(launch)
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=2", source))?,
            b3: module
                .prepare_nvfp4_swiglu_a16_t3(launch)
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=3", source))?,
            b4: module
                .prepare_nvfp4_swiglu_a16_t4(launch)
                .map_err(|source| GpuError::launch("preparing NVFP4 A16 B=4", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        slot: A16Slot,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($method:ident, $prepared:ident, $label:literal) => {
                module
                    .$method(
                        stream,
                        &self.$prepared,
                        input.cast::<u32>(),
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        weight_scale_reciprocal,
                        output,
                    )
                    .map_err(|source| GpuError::launch($label, source))
            };
        }

        match slot {
            A16Slot::B1 => launch!(nvfp4_swiglu_a16_t1, b1, "launching NVFP4 A16 B=1"),
            A16Slot::B2 => launch!(nvfp4_swiglu_a16_t2, b2, "launching NVFP4 A16 B=2"),
            A16Slot::B3 => launch!(nvfp4_swiglu_a16_t3, b3, "launching NVFP4 A16 B=3"),
            A16Slot::B4 => launch!(nvfp4_swiglu_a16_t4, b4, "launching NVFP4 A16 B=4"),
        }
    }
}

impl SwiGluA16Routes<Qwen35_9B> for PreparedQwen35A16Routes {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = a16_launch_config::<Qwen35_9B>();

        Ok(Self {
            b1: module
                .prepare_qwen35_nvfp4_swiglu_a16_t1(launch)
                .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 A16 B=1", source))?,
            b2: module
                .prepare_qwen35_nvfp4_swiglu_a16_t2(launch)
                .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 A16 B=2", source))?,
            b3: module
                .prepare_qwen35_nvfp4_swiglu_a16::<3>(launch)
                .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 A16 B=3", source))?,
            b4: module
                .prepare_qwen35_nvfp4_swiglu_a16::<4>(launch)
                .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 A16 B=4", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        slot: A16Slot,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($method:ident $(::<$tokens:literal>)?, $prepared:ident, $label:literal) => {
                module
                    .$method$(::<$tokens>)?(
                        stream,
                        &self.$prepared,
                        input.cast::<u32>(),
                        weight_codes.cast::<u32>(),
                        weight_scales,
                        weight_scale_reciprocal,
                        output,
                    )
                    .map_err(|source| GpuError::launch($label, source))
            };
        }

        match slot {
            A16Slot::B1 => launch!(
                qwen35_nvfp4_swiglu_a16_t1,
                b1,
                "launching Qwen3.5 NVFP4 A16 B=1"
            ),
            A16Slot::B2 => launch!(
                qwen35_nvfp4_swiglu_a16_t2,
                b2,
                "launching Qwen3.5 NVFP4 A16 B=2"
            ),
            A16Slot::B3 => launch!(
                qwen35_nvfp4_swiglu_a16::<3>,
                b3,
                "launching Qwen3.5 NVFP4 A16 B=3"
            ),
            A16Slot::B4 => launch!(
                qwen35_nvfp4_swiglu_a16::<4>,
                b4,
                "launching Qwen3.5 NVFP4 A16 B=4"
            ),
        }
    }
}

// The Qwen3.8 W4A4 entries compile that model's exact extents, so they stay
// bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> SwiGluW4a4Route<A> for PreparedW4a4Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let (quantize_launch, projection_launch) = w4a4_launch_configs::<A>(TOKENS, "")?;
        let quantize = module
            .prepare_nvfp4_quantize::<TOKENS>(quantize_launch)
            .map_err(|source| {
                GpuError::launch("preparing NVFP4 activation quantization", source)
            })?;
        let projection = module
            .prepare_nvfp4_swiglu_w4a4::<TOKENS>(projection_launch)
            .map_err(|source| GpuError::launch("preparing NVFP4 W4A4 SwiGLU", source))?;

        Ok(Self {
            quantize,
            projection,
        })
    }

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

impl<const TOKENS: usize> SwiGluW4a4Route<Qwen35_9B> for PreparedQwen35W4a4Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let (quantize_launch, projection_launch) =
            w4a4_launch_configs::<Qwen35_9B>(TOKENS, "Qwen3.5 ")?;
        let quantize = module
            .prepare_qwen35_nvfp4_quantize::<TOKENS>(quantize_launch)
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.5 NVFP4 activation quantization", source)
            })?;
        let projection = module
            .prepare_qwen35_nvfp4_swiglu_w4a4::<TOKENS>(projection_launch)
            .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 W4A4 SwiGLU", source))?;

        Ok(Self {
            quantize,
            projection,
        })
    }

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
            .qwen35_nvfp4_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u32>(),
                activation_scales,
                input_scale_divisor,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.5 NVFP4 activation quantization", source)
            })?;
        module
            .qwen35_nvfp4_swiglu_w4a4::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                1.0 / (input_scale_divisor * weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 NVFP4 W4A4 SwiGLU", source))
    }
}

// `route` never selects an unadmitted batch, so this is the defensive tail of
// a route that owns no entry.
impl<A: Arch> SwiGluW4a4Route<A> for UnadmittedRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    unsafe fn launch(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _input: *const u16,
        _activation_codes: *mut u8,
        _activation_scales: *mut u8,
        _weight_codes: *const u8,
        _weight_scales: *const u8,
        _input_scale_divisor: f32,
        _weight_scale_divisor: f32,
        _output: *mut u16,
    ) -> GpuResult<()> {
        Err(GpuError::invalid_launch(
            "NVFP4 W4A4 SwiGLU route is not admitted for this architecture",
        ))
    }
}

/// Qwen3.8 entry table: four concrete A16 entries, W4A4 wherever the measured
/// schedule selects it, and this model's own prefill entries.
pub struct Qwen38Nvfp4SwiGluEntries;

/// Qwen3.5 entry table: its own A16 entry family and a complete `B=1..=8`
/// W4A4 family, so the measured crossover stays reproducible.
pub struct Qwen35Nvfp4SwiGluEntries;

impl private::Sealed for Qwen38Nvfp4SwiGluEntries {}
impl private::Sealed for Qwen35Nvfp4SwiGluEntries {}

impl<A: Sm120Arch> Nvfp4SwiGluEntries<A> for Qwen38Nvfp4SwiGluEntries {
    type A16 = PreparedA16Routes;
    type W4a4Decode<const TOKENS: usize> = PreparedW4a4Route<TOKENS>;
    type W4a4Crossover<const TOKENS: usize> = UnadmittedRoute;
    type W4a4Prefill<const TOKENS: usize> = PreparedW4a4Route<TOKENS>;

    const LABEL: &'static str = "";
    const MODULE_CONTEXT: &'static str = "loading the NVFP4 SwiGLU module";
    const A16_VALIDATES_INPUT_DIVISOR: bool = true;

    // Transcribed from `Nvfp4SwiGluOp::launch`'s dispatch: B=1 and B=5..=8
    // dynamically quantize and use W4A4 MMA; B=2..=4 preserve the represented
    // BF16 activation and use the A16 schedule; exact T=32,64,128,1024 prefill
    // rows use W4A4 dynamic quantization and MMA.
    fn route(rows: usize) -> Option<SwiGluRoute> {
        match rows {
            1 => Some(SwiGluRoute::W4a4B1),
            2 => Some(SwiGluRoute::A16(A16Slot::B2)),
            3 => Some(SwiGluRoute::A16(A16Slot::B3)),
            4 => Some(SwiGluRoute::A16(A16Slot::B4)),
            5 => Some(SwiGluRoute::W4a4B5),
            6 => Some(SwiGluRoute::W4a4B6),
            7 => Some(SwiGluRoute::W4a4B7),
            8 => Some(SwiGluRoute::W4a4B8),
            _ => prefill_route(rows),
        }
    }

    fn unadmitted_rows(rows: usize) -> GpuError {
        GpuError::invalid_launch(format!(
            "NVFP4 SwiGLU row count {rows} is outside the exact B=1..=8, T=32,64,128,1024 routes"
        ))
    }

    fn ptx_names() -> Vec<&'static str> {
        nvfp4_swiglu_ptx_names().to_vec()
    }
}

impl Nvfp4SwiGluEntries<Qwen35_9B> for Qwen35Nvfp4SwiGluEntries {
    type A16 = PreparedQwen35A16Routes;
    type W4a4Decode<const TOKENS: usize> = PreparedQwen35W4a4Route<TOKENS>;
    type W4a4Crossover<const TOKENS: usize> = PreparedQwen35W4a4Route<TOKENS>;
    type W4a4Prefill<const TOKENS: usize> = PreparedQwen35W4a4Route<TOKENS>;

    const LABEL: &'static str = "Qwen3.5 ";
    const MODULE_CONTEXT: &'static str = "loading the Qwen3.5 NVFP4 SwiGLU module";
    const A16_VALIDATES_INPUT_DIVISOR: bool = false;

    // Transcribed from `qwen35_swiglu_schedule`. At 2,197/14,001 MHz
    // SM/memory clocks, paired device-path medians for A16/W4A4 were
    // 28.988/27.032 us (B=1), 25.623/27.023 (B=2), 30.726/27.148 (B=3), and
    // 37.498/27.172 (B=4). Thus only B=2 keeps represented BF16 activations;
    // the row formula is unchanged in either schedule and both candidates pass
    // the same independent FP64 oracle. Prefill rows are not admitted here:
    // they reach the device through `launch_prefill`.
    fn route(rows: usize) -> Option<SwiGluRoute> {
        match rows {
            1 => Some(SwiGluRoute::W4a4B1),
            2 => Some(SwiGluRoute::A16(A16Slot::B2)),
            3 => Some(SwiGluRoute::W4a4B3),
            4 => Some(SwiGluRoute::W4a4B4),
            5 => Some(SwiGluRoute::W4a4B5),
            6 => Some(SwiGluRoute::W4a4B6),
            7 => Some(SwiGluRoute::W4a4B7),
            8 => Some(SwiGluRoute::W4a4B8),
            _ => None,
        }
    }

    fn unadmitted_rows(rows: usize) -> GpuError {
        GpuError::invalid_launch(format!(
            "Qwen3.5 NVFP4 SwiGLU batch {rows} is not an exact B=1..={MAX_BATCH} route"
        ))
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_nvfp4_swiglu_ptx_names().to_vec()
    }
}

/// PTX symbols retained for every admitted NVFP4 SwiGLU schedule.
pub(crate) fn nvfp4_swiglu_ptx_names() -> [&'static str; 22] {
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
        kernels::nvfp4_quantize_ptx_name::<32>(),
        kernels::nvfp4_quantize_ptx_name::<64>(),
        kernels::nvfp4_quantize_ptx_name::<128>(),
        kernels::nvfp4_quantize_ptx_name::<1_024>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<1>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<5>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<6>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<7>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<8>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<32>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<64>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<128>(),
        kernels::nvfp4_swiglu_w4a4_ptx_name::<1_024>(),
    ]
}

/// PTX symbols retained for Qwen3.5 production and crossover comparison.
pub(crate) fn qwen35_nvfp4_swiglu_ptx_names() -> [&'static str; 28] {
    [
        "qwen35_nvfp4_swiglu_a16_t1",
        "qwen35_nvfp4_swiglu_a16_t2",
        kernels::qwen35_nvfp4_swiglu_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_swiglu_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<1>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<2>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<3>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<4>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<5>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<6>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<7>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<8>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<32>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<64>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<128>(),
        kernels::qwen35_nvfp4_quantize_ptx_name::<1_024>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<1>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<2>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<3>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<4>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<5>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<6>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<7>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<8>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<32>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<64>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<128>(),
        kernels::qwen35_nvfp4_swiglu_w4a4_ptx_name::<1_024>(),
    ]
}

/// Prepared A16 and W4A4 routes for one admitted architecture's exact NVFP4
/// MLP gate/up operation.
///
/// Both schedules keep the A16 (`B=1..=4`) and W4A4 candidates their measured
/// crossover compared, even though production selects one per batch.
pub struct Nvfp4SwiGluOp<A: Arch = Qwen38_27B, E: Nvfp4SwiGluEntries<A> = Qwen38Nvfp4SwiGluEntries>
{
    module: kernels::LoadedModule,
    a16: E::A16,
    w4a4_b1: E::W4a4Decode<1>,
    w4a4_b2: E::W4a4Crossover<2>,
    w4a4_b3: E::W4a4Crossover<3>,
    w4a4_b4: E::W4a4Crossover<4>,
    w4a4_b5: E::W4a4Decode<5>,
    w4a4_b6: E::W4a4Decode<6>,
    w4a4_b7: E::W4a4Decode<7>,
    w4a4_b8: E::W4a4Decode<8>,
    w4a4_t32: E::W4a4Prefill<32>,
    w4a4_t64: E::W4a4Prefill<64>,
    w4a4_t128: E::W4a4Prefill<128>,
    w4a4_t1024: E::W4a4Prefill<1_024>,
}

/// Prepared production and comparison routes for exact Qwen3.5 NVFP4 gate/up.
pub type Qwen35Nvfp4SwiGluOp = Nvfp4SwiGluOp<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>;

impl<A: Arch, E: Nvfp4SwiGluEntries<A>> Nvfp4SwiGluOp<A, E> {
    /// Loads the embedded SM120 module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_CONTEXT, source))?;

        Ok(Self {
            a16: E::A16::prepare(&module)?,
            w4a4_b1: E::W4a4Decode::<1>::prepare(&module)?,
            w4a4_b2: E::W4a4Crossover::<2>::prepare(&module)?,
            w4a4_b3: E::W4a4Crossover::<3>::prepare(&module)?,
            w4a4_b4: E::W4a4Crossover::<4>::prepare(&module)?,
            w4a4_b5: E::W4a4Decode::<5>::prepare(&module)?,
            w4a4_b6: E::W4a4Decode::<6>::prepare(&module)?,
            w4a4_b7: E::W4a4Decode::<7>::prepare(&module)?,
            w4a4_b8: E::W4a4Decode::<8>::prepare(&module)?,
            w4a4_t32: E::W4a4Prefill::<32>::prepare(&module)?,
            w4a4_t64: E::W4a4Prefill::<64>::prepare(&module)?,
            w4a4_t128: E::W4a4Prefill::<128>::prepare(&module)?,
            w4a4_t1024: E::W4a4Prefill::<1_024>::prepare(&module)?,
            module,
        })
    }

    /// Executes the retained production route for this architecture's exact
    /// decode rows, and for its exact prefill rows where the table admits them.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * A::HIDDEN` BF16 values; activation scratch covers
    /// `rows * A::HIDDEN / 2` code bytes and `rows * A::HIDDEN / 16` scale
    /// bytes; `weight_codes` covers the fused packed
    /// `[2 * A::INTERMEDIATE, A::HIDDEN]` plane; `weight_scales` covers its
    /// swizzled `[2 * A::INTERMEDIATE, A::HIDDEN / 16]` plane; and `output`
    /// covers `rows * A::INTERMEDIATE` BF16 values. Four-byte-loaded planes are
    /// four-byte aligned. Divisors are finite and positive. All allocations
    /// belong to `stream`'s context, remain live through completion, and do not
    /// overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut u8,
        weight_codes: *const u8,
        weight_scales: *const u8,
        input_scale_divisor: f32,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        let Some(route) = E::route(rows) else {
            return Err(E::unadmitted_rows(rows));
        };
        if validates_input_divisor::<A, E>(route) {
            check_input_divisor::<A, E>(input_scale_divisor)?;
        }
        check_weight_divisor::<A, E>(weight_scale_divisor)?;

        macro_rules! w4a4 {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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

        match route {
            SwiGluRoute::A16(slot) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    self.a16.launch(
                        &self.module,
                        stream,
                        slot,
                        input,
                        weight_codes,
                        weight_scales,
                        1.0 / weight_scale_divisor,
                        output,
                    )
                }
            }
            SwiGluRoute::W4a4B1 => w4a4!(w4a4_b1),
            SwiGluRoute::W4a4B2 => w4a4!(w4a4_b2),
            SwiGluRoute::W4a4B3 => w4a4!(w4a4_b3),
            SwiGluRoute::W4a4B4 => w4a4!(w4a4_b4),
            SwiGluRoute::W4a4B5 => w4a4!(w4a4_b5),
            SwiGluRoute::W4a4B6 => w4a4!(w4a4_b6),
            SwiGluRoute::W4a4B7 => w4a4!(w4a4_b7),
            SwiGluRoute::W4a4B8 => w4a4!(w4a4_b8),
            SwiGluRoute::W4a4T32 => w4a4!(w4a4_t32),
            SwiGluRoute::W4a4T64 => w4a4!(w4a4_t64),
            SwiGluRoute::W4a4T128 => w4a4!(w4a4_t128),
            SwiGluRoute::W4a4T1024 => w4a4!(w4a4_t1024),
        }
    }

    /// Executes the retained represented-BF16 A16 route for exact `B=1..=4`.
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
        check_weight_divisor::<A, E>(weight_scale_divisor)?;

        let Some(slot) = a16_slot(batch) else {
            return Err(GpuError::invalid_launch(format!(
                "{}NVFP4 A16 batch {batch} is not an exact B=1..=4 route",
                E::LABEL
            )));
        };

        // SAFETY: the public method's pointer contract is unchanged by dispatch.
        unsafe {
            self.a16.launch(
                &self.module,
                stream,
                slot,
                input,
                weight_codes,
                weight_scales,
                1.0 / weight_scale_divisor,
                output,
            )
        }
    }
}

impl Qwen35Nvfp4SwiGluOp {
    /// Quantizes BF16 activations and executes W4A4 at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// The requirements are identical to [`Self::launch`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_w4a4(
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
        check_input_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(input_scale_divisor)?;
        check_weight_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(weight_scale_divisor)?;

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
            1 => launch!(w4a4_b1),
            2 => launch!(w4a4_b2),
            3 => launch!(w4a4_b3),
            4 => launch!(w4a4_b4),
            5 => launch!(w4a4_b5),
            6 => launch!(w4a4_b6),
            7 => launch!(w4a4_b7),
            8 => launch!(w4a4_b8),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 NVFP4 W4A4 batch {batch} is not an exact B=1..={MAX_BATCH} route"
            ))),
        }
    }

    /// Quantizes and executes W4A4 at exact `T=32,64,128,1024`.
    ///
    /// # Safety
    ///
    /// The planes satisfy [`Self::launch`] for `rows` rather than `batch`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        activation_codes: *mut u8,
        activation_scales: *mut u8,
        weight_codes: *const u8,
        weight_scales: *const u8,
        input_scale_divisor: f32,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        let Some(route) = prefill_route(rows) else {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 NVFP4 SwiGLU prefill row count {rows} is outside the exact T=32,64,128,1024 routes"
            )));
        };
        check_input_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(input_scale_divisor)?;
        check_weight_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(weight_scale_divisor)?;

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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

        match route {
            SwiGluRoute::W4a4T32 => launch!(w4a4_t32),
            SwiGluRoute::W4a4T64 => launch!(w4a4_t64),
            SwiGluRoute::W4a4T128 => launch!(w4a4_t128),
            SwiGluRoute::W4a4T1024 => launch!(w4a4_t1024),
            _ => unreachable!("prefill_route only selects the exact T routes"),
        }
    }
}

// Whether `launch` rejects an invalid activation-scale divisor before it
// dispatches `route`. Transcribed from the two dispatches this wrapper
// replaces: Qwen3.8's validated both divisors before selecting a schedule, and
// Qwen3.5's validated inside the selected route, so only the W4A4 routes and
// Qwen3.8's A16 batches read the activation divisor before dispatch.
fn validates_input_divisor<A: Arch, E: Nvfp4SwiGluEntries<A>>(route: SwiGluRoute) -> bool {
    E::A16_VALIDATES_INPUT_DIVISOR || !matches!(route, SwiGluRoute::A16(_))
}

fn check_input_divisor<A: Arch, E: Nvfp4SwiGluEntries<A>>(divisor: f32) -> GpuResult<()> {
    if !divisor.is_finite() || divisor <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{}NVFP4 input scale divisor must be finite and positive",
            E::LABEL
        )));
    }

    Ok(())
}

fn check_weight_divisor<A: Arch, E: Nvfp4SwiGluEntries<A>>(divisor: f32) -> GpuResult<()> {
    if !divisor.is_finite() || divisor <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{}NVFP4 weight scale divisor must be finite and positive",
            E::LABEL
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        A16Slot, GROUP_K, MAX_BATCH, Nvfp4SwiGluEntries, PREFILL_ROWS, Qwen35Nvfp4SwiGluEntries,
        Qwen38Nvfp4SwiGluEntries, ROWS_PER_BRANCH, SwiGluRoute, W4A4_THREADS, a16_launch_config,
        a16_slot, check_input_divisor, check_weight_divisor, nvfp4_swiglu_ptx_names, prefill_route,
        qwen35_nvfp4_swiglu_ptx_names, validates_input_divisor, w4a4_launch_configs,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_gpu::{LaunchConfig1D, LaunchConfig2D};
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    /// Qwen3.8's production schedule, transcribed from the dispatch this
    /// wrapper replaces: `Nvfp4SwiGluOp::launch` matched `1 => w4a4_b1`,
    /// `2..=4 => launch_a16(rows)`, `5..=8 => w4a4_b5..b8`, and
    /// `32|64|128|1024 => w4a4_t32..t1024`, rejecting everything else.
    const QWEN38_SCHEDULE: [(usize, SwiGluRoute); 12] = [
        (1, SwiGluRoute::W4a4B1),
        (2, SwiGluRoute::A16(A16Slot::B2)),
        (3, SwiGluRoute::A16(A16Slot::B3)),
        (4, SwiGluRoute::A16(A16Slot::B4)),
        (5, SwiGluRoute::W4a4B5),
        (6, SwiGluRoute::W4a4B6),
        (7, SwiGluRoute::W4a4B7),
        (8, SwiGluRoute::W4a4B8),
        (32, SwiGluRoute::W4a4T32),
        (64, SwiGluRoute::W4a4T64),
        (128, SwiGluRoute::W4a4T128),
        (1_024, SwiGluRoute::W4a4T1024),
    ];

    /// Qwen3.5's production schedule, transcribed from `qwen35_swiglu_schedule`
    /// and the `Qwen35Nvfp4SwiGluOp::launch` arms it drove: `Some(A16)` at
    /// `B=2` reached `launch_a16(2)`, `Some(W4a4)` at `B=1` and `B=3..=8`
    /// reached `launch_w4a4(batch)`, and `None` — including every prefill row
    /// count, which only `launch_prefill` admits — was rejected.
    const QWEN35_SCHEDULE: [(usize, SwiGluRoute); 8] = [
        (1, SwiGluRoute::W4a4B1),
        (2, SwiGluRoute::A16(A16Slot::B2)),
        (3, SwiGluRoute::W4a4B3),
        (4, SwiGluRoute::W4a4B4),
        (5, SwiGluRoute::W4a4B5),
        (6, SwiGluRoute::W4a4B6),
        (7, SwiGluRoute::W4a4B7),
        (8, SwiGluRoute::W4a4B8),
    ];

    /// Every row count an entry table admits, swept exhaustively so an
    /// unadmitted width cannot hide between the transcribed ones.
    fn admitted_schedule<A: Arch, E: Nvfp4SwiGluEntries<A>>() -> Vec<(usize, SwiGluRoute)> {
        (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|rows| E::route(rows).map(|route| (rows, route)))
            .collect()
    }

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    #[test]
    fn inventory_covers_every_exact_route() {
        let names = nvfp4_swiglu_ptx_names();

        assert_eq!(MAX_BATCH, 8);
        assert_eq!(PREFILL_ROWS, [32, 64, 128, 1_024]);
        assert_eq!(names.len(), 22);
        assert_eq!(names.iter().filter(|name| name.contains("a16")).count(), 4);
        assert_eq!(
            names
                .iter()
                .filter(|name| name.contains("quantize"))
                .count(),
            9
        );
        assert_eq!(names.iter().filter(|name| name.contains("w4a4")).count(), 9);
    }

    #[test]
    fn qwen35_inventory_is_exact() {
        let names = qwen35_nvfp4_swiglu_ptx_names();
        let unique = names.into_iter().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 28);
        assert_eq!(unique.len(), names.len());
    }

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen38_27B>>::ptx_names(),
            nvfp4_swiglu_ptx_names().to_vec()
        );
        assert_eq!(
            <Qwen35Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen35_9B>>::ptx_names(),
            qwen35_nvfp4_swiglu_ptx_names().to_vec()
        );
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this file is
    /// its per-base-name count. These are the counts the pinned SM120 device
    /// build emits; a wrapper change that instantiates one more specialization
    /// moves one of them. `UnadmittedRoute` standing in for Qwen3.8's
    /// `B=2..=4` W4A4 slots is what keeps `nvfp4_quantize` and
    /// `nvfp4_swiglu_w4a4` at nine rather than twelve.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in nvfp4_swiglu_ptx_names()
            .into_iter()
            .chain(qwen35_nvfp4_swiglu_ptx_names())
        {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("nvfp4_quantize", 9),
                ("nvfp4_swiglu_a16_t1", 1),
                ("nvfp4_swiglu_a16_t2", 1),
                ("nvfp4_swiglu_a16_t3", 1),
                ("nvfp4_swiglu_a16_t4", 1),
                ("nvfp4_swiglu_w4a4", 9),
                ("qwen35_nvfp4_quantize", 12),
                ("qwen35_nvfp4_swiglu_a16", 2),
                ("qwen35_nvfp4_swiglu_a16_t1", 1),
                ("qwen35_nvfp4_swiglu_a16_t2", 1),
                ("qwen35_nvfp4_swiglu_w4a4", 12),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 50);
    }

    /// Route parity, the locked form of the transcription rule: for every
    /// admitted row count each entry table selects exactly the arm its replaced
    /// production dispatch took, and admits nothing the replaced dispatch
    /// rejected. The sweep runs `0..=2_048` plus `usize::MAX`, so the two
    /// schedules' disagreements — Qwen3.8 taking A16 across `B=2..=4` where
    /// Qwen3.5 takes it only at `B=2`, and Qwen3.8 admitting prefill rows in
    /// `launch` where Qwen3.5 rejects them — are pinned rather than assumed.
    #[test]
    fn production_route_selection_matches_every_replaced_dispatch() {
        assert_eq!(
            admitted_schedule::<Qwen38_27B, Qwen38Nvfp4SwiGluEntries>(),
            QWEN38_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(),
            QWEN35_SCHEDULE.to_vec()
        );
    }

    /// The sole measured crossover is Qwen3.5 `B=2`; every other Qwen3.5 batch
    /// quantizes. Stated separately from the table above because this is the
    /// value an earlier plan revision asserted from memory and inverted twice.
    #[test]
    fn qwen35_keeps_represented_activations_only_at_batch_two() {
        let a16 = (1..=MAX_BATCH)
            .filter(|batch| {
                matches!(
                    <Qwen35Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen35_9B>>::route(*batch),
                    Some(SwiGluRoute::A16(_))
                )
            })
            .collect::<Vec<_>>();
        let qwen38_a16 = (1..=MAX_BATCH)
            .filter(|batch| {
                matches!(
                    <Qwen38Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen38_27B>>::route(*batch),
                    Some(SwiGluRoute::A16(_))
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(a16, vec![2]);
        assert_eq!(qwen38_a16, vec![2, 3, 4]);
    }

    /// The retained comparison schedule both owners expose is exact `B=1..=4`,
    /// and the prefill schedule both owners quantize is exact `PREFILL_ROWS`.
    #[test]
    fn comparison_and_prefill_schedules_are_exact() {
        let a16 = (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|batch| a16_slot(batch).map(|slot| (batch, slot)))
            .collect::<Vec<_>>();
        let prefill = (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|rows| prefill_route(rows).map(|route| (rows, route)))
            .collect::<Vec<_>>();

        assert_eq!(
            a16,
            vec![
                (1, A16Slot::B1),
                (2, A16Slot::B2),
                (3, A16Slot::B3),
                (4, A16Slot::B4),
            ]
        );
        assert_eq!(
            prefill,
            vec![
                (32, SwiGluRoute::W4a4T32),
                (64, SwiGluRoute::W4a4T64),
                (128, SwiGluRoute::W4a4T128),
                (1_024, SwiGluRoute::W4a4T1024),
            ]
        );
    }

    /// Validation-order parity: `Nvfp4SwiGluOp::launch` validated both divisors
    /// before selecting a schedule, so it rejected an invalid activation
    /// divisor even at its A16 batches; `Qwen35Nvfp4SwiGluOp::launch` validated
    /// inside the selected route, so its A16 batch accepted any activation
    /// divisor. Both rejection sets are preserved, swept over every admitted
    /// row count.
    #[test]
    fn divisor_validation_order_matches_every_replaced_dispatch() {
        for (rows, route) in QWEN38_SCHEDULE {
            assert!(
                validates_input_divisor::<Qwen38_27B, Qwen38Nvfp4SwiGluEntries>(route),
                "Qwen3.8 row count {rows} must still reject an invalid activation divisor"
            );
        }
        for (rows, route) in QWEN35_SCHEDULE {
            assert_eq!(
                validates_input_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(route),
                !matches!(route, SwiGluRoute::A16(_)),
                "Qwen3.5 row count {rows} changed its activation-divisor rejection"
            );
        }
    }

    /// Each owner's rejection keeps naming the architecture and the wording it
    /// reported before the merge.
    #[test]
    fn rejections_keep_their_original_wording() {
        for (error, expected) in [
            (
                <Qwen38Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen38_27B>>::unadmitted_rows(9),
                "NVFP4 SwiGLU row count 9 is outside the exact B=1..=8, T=32,64,128,1024 routes",
            ),
            (
                <Qwen35Nvfp4SwiGluEntries as Nvfp4SwiGluEntries<Qwen35_9B>>::unadmitted_rows(32),
                "Qwen3.5 NVFP4 SwiGLU batch 32 is not an exact B=1..=8 route",
            ),
            (
                check_input_divisor::<Qwen38_27B, Qwen38Nvfp4SwiGluEntries>(0.0).unwrap_err(),
                "NVFP4 input scale divisor must be finite and positive",
            ),
            (
                check_weight_divisor::<Qwen38_27B, Qwen38Nvfp4SwiGluEntries>(f32::NAN).unwrap_err(),
                "NVFP4 weight scale divisor must be finite and positive",
            ),
            (
                check_input_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(-1.0).unwrap_err(),
                "Qwen3.5 NVFP4 input scale divisor must be finite and positive",
            ),
            (
                check_weight_divisor::<Qwen35_9B, Qwen35Nvfp4SwiGluEntries>(f32::INFINITY)
                    .unwrap_err(),
                "Qwen3.5 NVFP4 weight scale divisor must be finite and positive",
            ),
        ] {
            assert!(
                error.to_string().ends_with(expected),
                "{error} does not end with {expected}"
            );
        }
    }

    /// The shared launch geometry reproduces the exact grids the two replaced
    /// owners hard-coded: 2,176 and 1,536 A16 CTAs, 544 and 384 W4A4 column
    /// CTAs, and the same 1/2/3/22 token tiles at T=32/64/128/1024.
    #[test]
    fn shared_geometry_reproduces_every_replaced_owner_grid() {
        assert_eq!(
            a16_launch_config::<Qwen38_27B>(),
            LaunchConfig1D::new(2_176, 256, 0)
        );
        assert_eq!(
            a16_launch_config::<Qwen35_9B>(),
            LaunchConfig1D::new(1_536, 256, 0)
        );
        assert_eq!(Qwen38_27B::HIDDEN / GROUP_K, 320);
        assert_eq!(Qwen35_9B::HIDDEN / GROUP_K, 256);
        assert_eq!(Qwen38_27B::INTERMEDIATE / ROWS_PER_BRANCH, 544);
        assert_eq!(Qwen35_9B::INTERMEDIATE / ROWS_PER_BRANCH, 384);
        assert_eq!(W4A4_THREADS, 384);

        for (rows, tiles) in [(1, 1), (8, 1), (32, 1), (64, 2), (128, 3), (1_024, 22)] {
            let (qwen38_quantize, qwen38_projection) =
                w4a4_launch_configs::<Qwen38_27B>(rows, "").unwrap();
            let (qwen35_quantize, qwen35_projection) =
                w4a4_launch_configs::<Qwen35_9B>(rows, "Qwen3.5 ").unwrap();

            assert_eq!(
                qwen38_quantize,
                LaunchConfig1D::new((rows as u32 * 320).div_ceil(256), 256, 0)
            );
            assert_eq!(
                qwen38_projection,
                LaunchConfig2D::new((544, tiles), (W4A4_THREADS, 1), 0)
            );
            assert_eq!(
                qwen35_quantize,
                LaunchConfig1D::new((rows as u32 * 256).div_ceil(256), 256, 0)
            );
            assert_eq!(
                qwen35_projection,
                LaunchConfig2D::new((384, tiles), (W4A4_THREADS, 1), 0)
            );
        }
    }
}

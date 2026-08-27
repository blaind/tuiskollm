//! Exact-target NVFP4 down projection.

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
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
#[cfg(test)]
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const INPUT_COLUMNS: usize = Qwen38_27B::INTERMEDIATE;
const OUTPUT_ROWS: usize = HIDDEN;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP_K;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const PHASE_GROUPS: usize = 32;
const PHASES: usize = GROUPS_PER_ROW / PHASE_GROUPS;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();

// One warp retains two complete output-row reductions. The 32 lanes cover 32
// K16 groups per phase, so 34 phases cover all 1,088 groups without changing
// either output's reduction owner or lane order. Eight warps/CTA therefore
// produce 16 rows; 5,120 / 16 = 320 CTAs provide 1.88 blocks per 170-SM RTX 5090.
// Staging each 512-value phase once removes 15 duplicate activation reads per
// CTA while preserving both dot-product accumulation orders. Two resident
// 256-thread CTAs fit the measured register and 9,216-byte shared footprints.
// B=1 pairs the adjacent four-scale words owned by each row pair, reducing 16
// scale sectors per warp/phase to eight before subgroup broadcast.
// Its shared activation rows retain one padding word after eight packed pairs.
// The resulting nine-bank lane stride removes the eight-way conflicts from all
// eight reuse reads while preserving the represented values and FMA order.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = 32 * GROUP_K / 2;
const B1_SHARED_LANE_STRIDE: usize = GROUP_K / 2 + 1;
const B1_SHARED_PHASE_U32: usize = 32 * B1_SHARED_LANE_STRIDE;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

// Twelve warps cover a native 48-token by 64-output tile. Two 256-wide K
// stages occupy 36 KiB, retaining two CTAs per SM while all 68 K tiles traverse
// the source 17,408-wide represented plane without padding or chunking.
const W4_BLOCK_M: usize = 64;
const W4_TILE_M: usize = 48;
const W4_BLOCK_N: usize = 64;
const W4_BLOCK_K: usize = 256;
const W4_WARPS_N: usize = 4;
const W4_WARP_M: usize = 16;
const W4_WARP_N: usize = W4_BLOCK_N / W4_WARPS_N;
const W4_MMA_N: usize = W4_WARP_N / 8;
const W4_STAGES: usize = 2;
const W4_K64_PER_STAGE: usize = W4_BLOCK_K / 64;
const W4_CODE_ROW_BYTES: usize = W4_BLOCK_K / 2;
const W4_SEGMENTS_PER_ROW: usize = W4_CODE_ROW_BYTES / 16;
const W4_THREADS: u32 = ((W4_TILE_M / W4_WARP_M) * W4_WARPS_N * 32) as u32;

const W4_A_CODE_BYTES: usize = W4_STAGES * W4_BLOCK_M * W4_CODE_ROW_BYTES;
const W4_B_CODE_BYTES: usize = W4_STAGES * W4_BLOCK_N * W4_CODE_ROW_BYTES;
const W4_A_SCALE_BYTES: usize = W4_STAGES * W4_BLOCK_M * W4_K64_PER_STAGE * 4;
const W4_B_SCALE_BYTES: usize = W4_STAGES * W4_BLOCK_N * W4_K64_PER_STAGE * 4;
const W4_A_CODE_OFFSET: usize = 0;
const W4_B_CODE_OFFSET: usize = W4_A_CODE_OFFSET + W4_A_CODE_BYTES;
const W4_A_SCALE_OFFSET: usize = W4_B_CODE_OFFSET + W4_B_CODE_BYTES;
const W4_B_SCALE_OFFSET: usize = W4_A_SCALE_OFFSET + W4_A_SCALE_BYTES;
const W4_SHARED_BYTES: usize = W4_B_SCALE_OFFSET + W4_B_SCALE_BYTES;
const W4_SHARED_U32: usize = W4_SHARED_BYTES / 4;

const _: () = assert!(HIDDEN == 5_120);
const _: () = assert!(INPUT_COLUMNS == 17_408);
const _: () = assert!(OUTPUT_ROWS == 5_120);
const _: () = assert!(GROUPS_PER_ROW == 1_088);
const _: () = assert!(Qwen35_9B::HIDDEN == 4_096);
const _: () = assert!(Qwen35_9B::INTERMEDIATE == 12_288);
const _: () = assert!(PHASES * CODE_WORDS_PER_PHASE == CODE_BYTES_PER_ROW / size_of::<u32>());
const _: () = assert!(SHARED_U32 * size_of::<u32>() == 8_192);
const _: () = assert!(INPUT_COLUMNS.is_multiple_of(W4_BLOCK_K));
const _: () = assert!(OUTPUT_ROWS.is_multiple_of(W4_BLOCK_N));
const _: () = assert!(W4_SHARED_BYTES == 36_864);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, warp, wmma};

    #[inline(always)]
    fn weight_scale_offset<A: Arch>(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;

        let scale_tiles_per_row = A::INTERMEDIATE / GROUP_K / 4;

        (persistent_tile * scale_tiles_per_row + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset<A: Arch>(parent_row: usize, group: usize) -> usize {
        weight_scale_offset::<A>(parent_row, group >> 2) + (group & 3)
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
    unsafe fn down_body<
        A: Arch,
        const TOKENS: usize,
        const COALESCED_SCALES: bool,
        const PADDED_ACTIVATIONS: bool,
    >(
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
        let pair_index = block * (2 * WARPS) + 2 * warp_index;
        let first_row = physical_row(pair_index);
        let second_row = physical_row(pair_index + 1);
        let mut first_accumulators = [0.0f32; TOKENS];
        let mut second_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < A::INTERMEDIATE / GROUP_K / PHASE_GROUPS {
            let mut task = tid;
            while task < TOKENS * PHASE_PACKED_PAIRS {
                let token = task / PHASE_PACKED_PAIRS;
                let pair = task - token * PHASE_PACKED_PAIRS;
                // SAFETY: every exact route supplies `TOKENS` complete input rows.
                unsafe {
                    let shared_pair = if PADDED_ACTIVATIONS {
                        (pair / 8) * B1_SHARED_LANE_STRIDE + pair % 8
                    } else {
                        pair
                    };
                    let shared_stride = if PADDED_ACTIVATIONS {
                        B1_SHARED_PHASE_U32
                    } else {
                        PHASE_PACKED_PAIRS
                    };
                    *shared.add(token * shared_stride + shared_pair) =
                        *input.add(token * (A::INTERMEDIATE / 2) + phase * 256 + pair);
                }
                task += THREADS as usize;
            }
            thread::sync_threads();

            let group = phase * PHASE_GROUPS + lane;
            let (first_scale, second_scale) = if COALESCED_SCALES {
                let scale_lane = lane & 3;
                let mut first_word = 0u32;
                let mut second_word = 0u32;
                if scale_lane == 0 {
                    let offset = weight_scale_offset::<A>(first_row, group >> 2);
                    // SAFETY: paired physical rows own adjacent aligned four-scale words.
                    (first_word, second_word) =
                        unsafe { load_u32x2_read_only(weight_scales.add(offset).cast::<u32>()) };
                }
                let source_lane = (lane - scale_lane) as u32;
                first_word = warp::shuffle(first_word, source_lane);
                second_word = warp::shuffle(second_word, source_lane);
                let shift = scale_lane * 8;

                ((first_word >> shift) as u8, (second_word >> shift) as u8)
            } else {
                // SAFETY: source validation admitted one swizzled scale per logical group.
                let first = unsafe {
                    load_u8_read_only(
                        weight_scales.add(weight_group_scale_offset::<A>(first_row, group)),
                    )
                };
                let second = unsafe {
                    load_u8_read_only(
                        weight_scales.add(weight_group_scale_offset::<A>(second_row, group)),
                    )
                };

                (first, second)
            };
            let first_coefficient = e4m3_to_f32(first_scale) * weight_scale_reciprocal;
            let second_coefficient = e4m3_to_f32(second_scale) * weight_scale_reciprocal;
            let row_words = (A::INTERMEDIATE / 2) / size_of::<u32>();
            let word_offset = phase * CODE_WORDS_PER_PHASE + lane * 2;
            let first_source = unsafe { weight_codes.add(first_row * row_words + word_offset) };
            let second_source = unsafe { weight_codes.add(second_row * row_words + word_offset) };
            // SAFETY: one logical group contains exactly two packed u32 words.
            let first_words = unsafe { load_u32x2_read_only(first_source) };
            // SAFETY: one logical group contains exactly two packed u32 words.
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
                                    let shared_stride = if PADDED_ACTIVATIONS {
                                        B1_SHARED_PHASE_U32
                                    } else {
                                        PHASE_PACKED_PAIRS
                                    };
                                    let lane_stride = if PADDED_ACTIVATIONS {
                                        B1_SHARED_LANE_STRIDE
                                    } else {
                                        8
                                    };
                                    *shared.add($token * shared_stride + lane * lane_stride + $pair)
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
                        // SAFETY: one lane writes two unique token/output-row values.
                        unsafe {
                            *output.add($token * A::HIDDEN + first_row) = f32_to_bf16(first);
                            *output.add($token * A::HIDDEN + second_row) = f32_to_bf16(second);
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
    /// Projects the singleton BF16 activation through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_down_a16_b1(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            down_body::<Qwen38_27B, 1, true, true>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    /// Projects `TOKENS` BF16 activations through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_down_a16<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;
        let _ = A::HIDDEN;

        unsafe {
            down_body::<A, TOKENS, false, false>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    /// Projects exact Qwen3.5 BF16 activations through represented NVFP4.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_down_a16<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            down_body::<Qwen35_9B, TOKENS, false, false>(
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
    fn w4_swizzled_byte(row: usize, logical_byte: usize) -> usize {
        let logical_segment = logical_byte >> 4;
        let byte_in_segment = logical_byte & 15;
        let physical_segment = logical_segment ^ (row & (W4_SEGMENTS_PER_ROW - 1));

        physical_segment * 16 + byte_in_segment
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
    unsafe fn quantize_prefill_body<A: Arch, const TOKENS: usize>(
        task: usize,
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        let groups_per_row = A::INTERMEDIATE / GROUP_K;
        if task >= TOKENS * groups_per_row {
            return;
        }

        let token = task / groups_per_row;
        let group = task - token * groups_per_row;
        let source = unsafe { input.add(token * (A::INTERMEDIATE / 2) + group * (GROUP_K / 2)) };
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
            unsafe { codes.add(token * (A::INTERMEDIATE / 8) + group * (GROUP_K / 8)) };

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
    pub fn nvfp4_down_quantize<const TOKENS: usize>(
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        unsafe {
            quantize_prefill_body::<Qwen38_27B, TOKENS>(
                thread::index_1d().get(),
                input,
                codes,
                scales,
                input_scale_divisor,
            );
        }
    }

    /// Quantizes exact Qwen3.5 BF16 prompt rows into represented NVFP4.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_down_quantize<const TOKENS: usize>(
        input: *const u32,
        codes: *mut u32,
        scales: *mut u8,
        input_scale_divisor: f32,
    ) {
        unsafe {
            quantize_prefill_body::<Qwen35_9B, TOKENS>(
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
    unsafe fn stage_w4_tile<A: Arch, const TOKENS: usize>(
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
        if tid < W4_TILE_M * W4_SEGMENTS_PER_ROW {
            let row = tid / W4_SEGMENTS_PER_ROW;
            let segment = tid - row * W4_SEGMENTS_PER_ROW;
            let valid = token_begin + row < TOKENS;
            let source_token = if valid { token_begin + row } else { 0 };
            let physical = w4_swizzled_byte(row, segment * 16);
            let destination = unsafe {
                shared
                    .add(
                        W4_A_CODE_OFFSET
                            + (stage * W4_BLOCK_M + row) * W4_CODE_ROW_BYTES
                            + physical,
                    )
                    .cast::<u32>()
            };
            let source = unsafe {
                activation_codes.add(
                    source_token * (A::INTERMEDIATE / 8)
                        + k_tile * (W4_CODE_ROW_BYTES / 4)
                        + segment * 4,
                )
            };

            unsafe {
                cp_async_cg_zfill_16(destination, source.cast::<u8>(), if valid { 16 } else { 0 });
            }
        }

        if tid < W4_TILE_M {
            let valid = token_begin + tid < TOKENS;
            let source_token = if valid { token_begin + tid } else { 0 };
            let destination = unsafe {
                shared
                    .add(W4_A_SCALE_OFFSET + (stage * W4_BLOCK_M + tid) * W4_K64_PER_STAGE * 4)
                    .cast::<u32>()
            };
            let source = unsafe {
                activation_scales
                    .add(source_token * (A::INTERMEDIATE / GROUP_K) + k_tile * W4_K64_PER_STAGE * 4)
            };

            unsafe {
                cp_async_cg_zfill_16(destination, source, if valid { 16 } else { 0 });
            }
        }

        let mut task = tid;
        while task < W4_BLOCK_N * W4_SEGMENTS_PER_ROW {
            let row = task / W4_SEGMENTS_PER_ROW;
            let segment = task - row * W4_SEGMENTS_PER_ROW;
            let parent_row = row_begin + row;
            let physical = w4_swizzled_byte(row, segment * 16);
            let destination = unsafe {
                shared
                    .add(
                        W4_B_CODE_OFFSET
                            + (stage * W4_BLOCK_N + row) * W4_CODE_ROW_BYTES
                            + physical,
                    )
                    .cast::<u32>()
            };
            let source = unsafe {
                weight_codes.add(
                    parent_row * (A::INTERMEDIATE / 8)
                        + k_tile * (W4_CODE_ROW_BYTES / 4)
                        + segment * 4,
                )
            };

            unsafe { cp_async_cg_16(destination, source) };
            task += W4_THREADS as usize;
        }

        let mut scale_task = tid;
        while scale_task < W4_BLOCK_N * W4_K64_PER_STAGE {
            let row = scale_task / W4_K64_PER_STAGE;
            let local_k64 = scale_task - row * W4_K64_PER_STAGE;
            let parent_row = row_begin + row;
            let global_k64 = k_tile * W4_K64_PER_STAGE + local_k64;
            let destination = unsafe {
                shared
                    .add(
                        W4_B_SCALE_OFFSET
                            + (stage * W4_BLOCK_N * W4_K64_PER_STAGE + scale_task) * 4,
                    )
                    .cast::<u32>()
            };
            let source =
                unsafe { weight_scales.add(weight_scale_offset::<A>(parent_row, global_k64)) };

            unsafe { cp_async_ca_4(destination, source.cast::<u32>()) };
            scale_task += W4_THREADS as usize;
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
    unsafe fn down_w4a4_body<A: Arch, const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        static mut SHARED: SharedArray<u32, W4_SHARED_U32, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(SHARED).cast::<u8>();
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        let warp_m = warp / W4_WARPS_N;
        let warp_n = warp - warp_m * W4_WARPS_N;
        let token_begin = thread::blockIdx_y() as usize * W4_TILE_M;
        let row_begin = thread::blockIdx_x() as usize * W4_BLOCK_N;
        let mut stage = 0usize;

        while stage < W4_STAGES {
            unsafe {
                stage_w4_tile::<A, TOKENS>(
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

        let mut accumulators = [[0.0f32; 4]; W4_MMA_N];
        let a_matrix = lane >> 3;
        let a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
        let a_column_byte = (a_matrix >> 1) * 16;
        let b_row_offset = lane & 7;
        let b_column_byte = ((lane >> 3) & 1) * 16;
        let scale_a_row_offset = ((lane & 1) << 3) | (lane >> 2);
        let scale_b_row_offset = lane >> 2;
        let mut k_tile = 0usize;

        while k_tile < A::INTERMEDIATE / W4_BLOCK_K {
            stage = k_tile % W4_STAGES;
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();

            let mut local_k64 = 0usize;
            while local_k64 < W4_K64_PER_STAGE {
                let a_row = warp_m * W4_WARP_M + a_row_offset;
                let a_logical_byte = local_k64 * 32 + a_column_byte;
                let a_physical_byte = w4_swizzled_byte(a_row, a_logical_byte);
                let a_address = unsafe {
                    shared
                        .add(
                            W4_A_CODE_OFFSET
                                + (stage * W4_BLOCK_M + a_row) * W4_CODE_ROW_BYTES
                                + a_physical_byte,
                        )
                        .cast::<u32>()
                };
                let a_fragments = unsafe { wmma::ldmatrix_x4(a_address) };
                let scale_a_row = warp_m * W4_WARP_M + scale_a_row_offset;
                let scale_a = unsafe {
                    *shared
                        .add(
                            W4_A_SCALE_OFFSET
                                + (stage * W4_BLOCK_M + scale_a_row) * W4_K64_PER_STAGE * 4
                                + local_k64 * 4,
                        )
                        .cast::<u32>()
                };
                let mut mma_n = 0usize;

                while mma_n < W4_MMA_N {
                    let b_row = warp_n * W4_WARP_N + mma_n * 8 + b_row_offset;
                    let b_logical_byte = local_k64 * 32 + b_column_byte;
                    let b_physical_byte = w4_swizzled_byte(b_row, b_logical_byte);
                    let b_address = unsafe {
                        shared
                            .add(
                                W4_B_CODE_OFFSET
                                    + (stage * W4_BLOCK_N + b_row) * W4_CODE_ROW_BYTES
                                    + b_physical_byte,
                            )
                            .cast::<u32>()
                    };
                    let b_fragments = unsafe { wmma::ldmatrix_x2(b_address) };
                    let scale_b_row = warp_n * W4_WARP_N + mma_n * 8 + scale_b_row_offset;
                    let scale_b = unsafe {
                        *shared
                            .add(
                                W4_B_SCALE_OFFSET
                                    + (stage * W4_BLOCK_N + scale_b_row) * W4_K64_PER_STAGE * 4
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
            let next_k_tile = k_tile + W4_STAGES;
            if next_k_tile < A::INTERMEDIATE / W4_BLOCK_K {
                unsafe {
                    stage_w4_tile::<A, TOKENS>(
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
        let token0 = warp_m * W4_WARP_M + accumulator_row;
        let token1 = token0 + 8;
        let mut mma_n = 0usize;

        while mma_n < W4_MMA_N {
            let local_row = warp_n * W4_WARP_N + mma_n * 8 + accumulator_col;
            let values = accumulators[mma_n];

            if token_begin + token0 < TOKENS {
                let destination = unsafe {
                    output
                        .add((token_begin + token0) * A::HIDDEN + row_begin + local_row)
                        .cast::<u32>()
                };
                unsafe {
                    *destination = tcgen05::cvt_f32x2_bf16x2(values[0] * alpha, values[1] * alpha);
                }
            }
            if token_begin + token1 < TOKENS {
                let destination = unsafe {
                    output
                        .add((token_begin + token1) * A::HIDDEN + row_begin + local_row)
                        .cast::<u32>()
                };
                unsafe {
                    *destination = tcgen05::cvt_f32x2_bf16x2(values[2] * alpha, values[3] * alpha);
                }
            }
            mma_n += 1;
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
    pub fn nvfp4_down_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        unsafe {
            down_w4a4_body::<Qwen38_27B, TOKENS>(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                output,
                alpha,
            );
        }
    }

    /// Projects exact Qwen3.5 prompt rows through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(384, 2)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_down_w4a4<const TOKENS: usize>(
        activation_codes: *const u32,
        activation_scales: *const u8,
        weight_codes: *const u32,
        weight_scales: *const u8,
        output: *mut u16,
        alpha: f32,
    ) {
        unsafe {
            down_w4a4_body::<Qwen35_9B, TOKENS>(
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
    // Eight warps emit two rows each, so Qwen3.8's 5,120 outputs form 320
    // exact CTAs and Qwen3.5's 4,096 form 256, both without a tail branch
    // while each CTA shares one staged input phase. Qwen3.5's row has 768
    // K16 groups, so the unchanged lane order traverses 24 phases instead of
    // Qwen3.8's 34 without changing arithmetic.
    LaunchConfig1D::new((A::HIDDEN / (2 * WARPS)) as u32, THREADS, 0)
}

/// Exact W4A4 prefill grid extents for one architecture and row count.
struct PrefillGeometry {
    quantize_blocks: usize,
    projection_blocks: usize,
    token_tiles: usize,
}

fn prefill_geometry<A: Arch>(tokens: usize) -> PrefillGeometry {
    // A thread owns one represented K16 group, so 256 threads cover every
    // architecture-specific group without a partial: T=1,024 has 3,145,728
    // groups on Qwen3.5 and 12,288 CTAs. Each twelve-warp CTA emits a native
    // 48x64 tile, giving Qwen3.8 80 output columns and Qwen3.5 64, with
    // 1/2/3/22 token tiles at T=32/64/128/1024 and no N padding in either.
    PrefillGeometry {
        quantize_blocks: (tokens * (A::INTERMEDIATE / GROUP_K)).div_ceil(256),
        projection_blocks: A::HIDDEN / W4_BLOCK_N,
        token_tiles: tokens.div_ceil(W4_TILE_M),
    }
}

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared represented-weight A16 entry for an exact batch.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait Nvfp4DownA16Route<A: Arch>: Sized + private::Sealed {
    /// Prepares this route's exact batch entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's A16 entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `Nvfp4DownOp::launch`'s contract unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// One architecture's prepared quantization and W4A4 entries for exact rows.
///
/// Sealed for the same reason as [`Nvfp4DownA16Route`]. The pair stays one
/// route because the quantizer's output grouping is the projection's input
/// contract and the two are never prepared apart.
pub trait Nvfp4DownPrefillRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares both entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Quantizes the represented activations and launches the W4A4 entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `Nvfp4DownOp::launch_prefill`'s contract unchanged.
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

/// Exact entry table of one admitted architecture's NVFP4 down routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 here never widens the artifact-level
/// admission bound. Each table names only the entries its own model emits,
/// which is what keeps the compiled inventory fixed while both prepared
/// owners share one wrapper.
pub trait Nvfp4DownEntries<A: Arch>: private::Sealed {
    /// Prepared A16 route for `B=1`, which stays separate because Qwen3.8
    /// anchors it with a concrete entry.
    type DecodeOne: Nvfp4DownA16Route<A>;
    /// Prepared A16 route for `B=2..=8`.
    type Decode<const TOKENS: usize>: Nvfp4DownA16Route<A>;
    /// Prepared W4A4 route for one exact prefill row count.
    type Prefill<const TOKENS: usize>: Nvfp4DownPrefillRoute<A>;

    /// Message prefix that keeps this architecture's launch errors distinct.
    const LABEL: &'static str;
    /// Operation name this architecture's batch rejection reports.
    const OPERATION: &'static str;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8 `B=1` A16 entry.
///
/// `B=1` keeps the concrete entry that anchors the embedded module artifact.
pub struct PreparedBatchOneRoute {
    projection: PreparedLaunch<kernels::__nvfp4_down_a16_b1_CudaKernel>,
}

/// Prepared generic A16 entry for one exact batch.
pub struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__nvfp4_down_a16_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.5 A16 entry for one exact batch.
pub struct PreparedQwen35BatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_down_a16_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8 quantization and W4A4 entries for one exact row count.
pub struct PreparedPrefillRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__nvfp4_down_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__nvfp4_down_w4a4_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 quantization and W4A4 entries for one exact row count.
pub struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__qwen35_nvfp4_down_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_down_w4a4_CudaKernel<TOKENS>>,
}

impl private::Sealed for PreparedBatchOneRoute {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedBatchRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35BatchRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedPrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35PrefillRoute<TOKENS> {}

// The B=1 anchor compiles the exact Qwen3.8 row width into a concrete entry,
// so it stays bound to the sealed artifact-level architecture.
impl<A: Sm120Arch> Nvfp4DownA16Route<A> for PreparedBatchOneRoute {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_nvfp4_down_a16_b1(a16_launch_config::<A>())
            .map_err(|source| GpuError::launch("preparing SM120 NVFP4 A16 B=1", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .nvfp4_down_a16_b1(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching SM120 NVFP4 A16 B=1", source))
    }
}

impl<A: Arch, const TOKENS: usize> Nvfp4DownA16Route<A> for PreparedBatchRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_nvfp4_down_a16::<A, TOKENS>(a16_launch_config::<A>())
            .map_err(|source| GpuError::launch("preparing SM120 NVFP4 A16", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .nvfp4_down_a16::<A, TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching SM120 NVFP4 A16", source))
    }
}

impl<const TOKENS: usize> Nvfp4DownA16Route<Qwen35_9B> for PreparedQwen35BatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen35_nvfp4_down_a16::<TOKENS>(a16_launch_config::<Qwen35_9B>())
            .map_err(|source| GpuError::launch("preparing Qwen3.5 SM120 NVFP4 A16", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_down_a16::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 SM120 NVFP4 A16", source))
    }
}

// The Qwen3.8 prefill entries compile that model's exact extents, so they stay
// bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> Nvfp4DownPrefillRoute<A> for PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let geometry = prefill_geometry::<A>(TOKENS);
        let quantize_blocks = u32::try_from(geometry.quantize_blocks).map_err(|_| {
            GpuError::invalid_launch("NVFP4 down quantization grid exceeds CUDA width")
        })?;
        let projection_blocks = u32::try_from(geometry.projection_blocks)
            .map_err(|_| GpuError::invalid_launch("NVFP4 down grid exceeds CUDA width"))?;
        let token_tiles = u32::try_from(geometry.token_tiles)
            .map_err(|_| GpuError::invalid_launch("NVFP4 down grid exceeds CUDA height"))?;
        let quantize = module
            .prepare_nvfp4_down_quantize::<TOKENS>(LaunchConfig1D::new(quantize_blocks, 256, 0))
            .map_err(|source| {
                GpuError::launch("preparing NVFP4 down activation quantization", source)
            })?;
        let projection = module
            .prepare_nvfp4_down_w4a4::<TOKENS>(LaunchConfig2D::new(
                (projection_blocks, token_tiles),
                (W4_THREADS, 1),
                0,
            ))
            .map_err(|source| GpuError::launch("preparing NVFP4 W4A4 down", source))?;

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
            .nvfp4_down_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u32>(),
                activation_scales,
                input_scale_divisor,
            )
            .map_err(|source| {
                GpuError::launch("launching NVFP4 down activation quantization", source)
            })?;
        module
            .nvfp4_down_w4a4::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                1.0 / (input_scale_divisor * weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching NVFP4 W4A4 down", source))
    }
}

impl<const TOKENS: usize> Nvfp4DownPrefillRoute<Qwen35_9B> for PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let geometry = prefill_geometry::<Qwen35_9B>(TOKENS);
        let quantize_blocks = u32::try_from(geometry.quantize_blocks)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 down quantization grid is too wide"))?;
        let projection_blocks = u32::try_from(geometry.projection_blocks)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 down grid is too wide"))?;
        let token_tiles = u32::try_from(geometry.token_tiles)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 down grid is too tall"))?;
        let quantize = module
            .prepare_qwen35_nvfp4_down_quantize::<TOKENS>(LaunchConfig1D::new(
                quantize_blocks,
                256,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.5 down activation quantization", source)
            })?;
        let projection = module
            .prepare_qwen35_nvfp4_down_w4a4::<TOKENS>(LaunchConfig2D::new(
                (projection_blocks, token_tiles),
                (W4_THREADS, 1),
                0,
            ))
            .map_err(|source| GpuError::launch("preparing Qwen3.5 W4A4 down", source))?;

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
            .qwen35_nvfp4_down_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                activation_codes.cast::<u32>(),
                activation_scales,
                input_scale_divisor,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.5 down activation quantization", source)
            })?;
        module
            .qwen35_nvfp4_down_w4a4::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                activation_scales,
                weight_codes.cast::<u32>(),
                weight_scales,
                output,
                1.0 / (input_scale_divisor * weight_scale_divisor),
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 W4A4 down", source))
    }
}

/// Qwen3.8 entry table: the concrete `B=1` artifact anchor, the generic A16
/// entries at `B=2..=8`, and this model's own prefill entries.
pub struct Qwen38Nvfp4DownEntries;

/// Qwen3.5 entry table: its own A16 entry family and prefill entries.
pub struct Qwen35Nvfp4DownEntries;

impl private::Sealed for Qwen38Nvfp4DownEntries {}
impl private::Sealed for Qwen35Nvfp4DownEntries {}

impl<A: Sm120Arch> Nvfp4DownEntries<A> for Qwen38Nvfp4DownEntries {
    type DecodeOne = PreparedBatchOneRoute;
    type Decode<const TOKENS: usize> = PreparedBatchRoute<A, TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<TOKENS>;

    const LABEL: &'static str = "";
    const OPERATION: &'static str = "NVFP4 down projection";

    fn ptx_names() -> Vec<&'static str> {
        nvfp4_down_ptx_names().to_vec()
    }
}

impl Nvfp4DownEntries<Qwen35_9B> for Qwen35Nvfp4DownEntries {
    type DecodeOne = PreparedQwen35BatchRoute<1>;
    type Decode<const TOKENS: usize> = PreparedQwen35BatchRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen35PrefillRoute<TOKENS>;

    const LABEL: &'static str = "Qwen3.5 ";
    const OPERATION: &'static str = "Qwen3.5 NVFP4 down";

    fn ptx_names() -> Vec<&'static str> {
        qwen35_nvfp4_down_ptx_names().to_vec()
    }
}

/// PTX symbols retained for every exact SM120 NVFP4 down schedule.
pub(crate) fn nvfp4_down_ptx_names() -> [&'static str; 16] {
    [
        "nvfp4_down_a16_b1",
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 2>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 3>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 4>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 5>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 6>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 7>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 8>(),
        kernels::nvfp4_down_quantize_ptx_name::<32>(),
        kernels::nvfp4_down_quantize_ptx_name::<64>(),
        kernels::nvfp4_down_quantize_ptx_name::<128>(),
        kernels::nvfp4_down_quantize_ptx_name::<1_024>(),
        kernels::nvfp4_down_w4a4_ptx_name::<32>(),
        kernels::nvfp4_down_w4a4_ptx_name::<64>(),
        kernels::nvfp4_down_w4a4_ptx_name::<128>(),
        kernels::nvfp4_down_w4a4_ptx_name::<1_024>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.5 NVFP4 down route.
pub(crate) fn qwen35_nvfp4_down_ptx_names() -> [&'static str; 16] {
    [
        kernels::qwen35_nvfp4_down_a16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<8>(),
        kernels::qwen35_nvfp4_down_quantize_ptx_name::<32>(),
        kernels::qwen35_nvfp4_down_quantize_ptx_name::<64>(),
        kernels::qwen35_nvfp4_down_quantize_ptx_name::<128>(),
        kernels::qwen35_nvfp4_down_quantize_ptx_name::<1_024>(),
        kernels::qwen35_nvfp4_down_w4a4_ptx_name::<32>(),
        kernels::qwen35_nvfp4_down_w4a4_ptx_name::<64>(),
        kernels::qwen35_nvfp4_down_w4a4_ptx_name::<128>(),
        kernels::qwen35_nvfp4_down_w4a4_ptx_name::<1_024>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_nvfp4_down_decode),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct Nvfp4DownDecodeRoutes<A: Arch, E: Nvfp4DownEntries<A>> {
    #[route(1)]
    b1: E::DecodeOne,
    #[route(2)]
    b2: E::Decode<2>,
    #[route(3)]
    b3: E::Decode<3>,
    #[route(4)]
    b4: E::Decode<4>,
    #[route(5)]
    b5: E::Decode<5>,
    #[route(6)]
    b6: E::Decode<6>,
    #[route(7)]
    b7: E::Decode<7>,
    #[route(8)]
    b8: E::Decode<8>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_nvfp4_down_prefill),
    required(32, 64, 128, 1024),
    inventory(false)
)]
struct Nvfp4DownPrefillRoutes<A: Arch, E: Nvfp4DownEntries<A>> {
    #[route(32)]
    t32: E::Prefill<32>,
    #[route(64)]
    t64: E::Prefill<64>,
    #[route(128)]
    t128: E::Prefill<128>,
    #[route(1024)]
    t1024: E::Prefill<1_024>,
}

/// Prepared exact-batch NVFP4 down routes for one admitted architecture.
pub struct Nvfp4DownOp<A: Arch = Qwen38_27B, E: Nvfp4DownEntries<A> = Qwen38Nvfp4DownEntries> {
    module: kernels::LoadedModule,
    decode_routes: Nvfp4DownDecodeRoutes<A, E>,
    prefill_routes: Nvfp4DownPrefillRoutes<A, E>,
}

/// Prepared exact-batch Qwen3.5 NVFP4 down routes on SM120.
pub type Qwen35Nvfp4DownOp = Nvfp4DownOp<Qwen35_9B, Qwen35Nvfp4DownEntries>;

impl<A: Arch, E: Nvfp4DownEntries<A>> Nvfp4DownOp<A, E> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the SM120 NVFP4 down projection module", source)
        })?;

        let decode_routes = Nvfp4DownDecodeRoutes::prepare(&module)?;
        let prefill_routes = Nvfp4DownPrefillRoutes::prepare(&module)?;

        Ok(Self {
            module,
            decode_routes,
            prefill_routes,
        })
    }

    /// Executes the represented-weight A16 route for exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * A::INTERMEDIATE` BF16 values; `weight_codes`
    /// covers the packed `[A::HIDDEN, A::INTERMEDIATE]` E2M1 plane;
    /// `weight_scales` covers its swizzled `[A::HIDDEN, A::INTERMEDIATE / 16]`
    /// E4M3 plane; and `output` covers `batch * A::HIDDEN` BF16 values.
    /// Four-byte-loaded planes are four-byte aligned. The divisor is finite
    /// and positive. Allocations belong to `stream`'s context, remain live
    /// through completion, and do not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
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
            return Err(GpuError::invalid_launch(format!(
                "{}NVFP4 weight scale divisor must be finite and positive",
                E::LABEL
            )));
        }

        let reciprocal = 1.0 / weight_scale_divisor;
        dispatch_nvfp4_down_decode!(
            &self.decode_routes,
            batch,
            |route| unsafe {
                route.launch(&self.module, stream, input, weight_codes, weight_scales, reciprocal, output)
            },
            else => Err(GpuError::invalid_launch(format!(
                "{} batch {batch} is outside the exact range 1..={MAX_BATCH}",
                E::OPERATION
            )))
        )
    }

    /// Dynamically quantizes and projects exact `T=32,64,128,1024` rows.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * A::INTERMEDIATE` BF16 values; activation scratch
    /// covers `rows * A::INTERMEDIATE / 2` code bytes and
    /// `rows * A::INTERMEDIATE / 16` scale bytes; the weight planes satisfy
    /// [`Self::launch`]; and `output` covers `rows * A::HIDDEN` BF16 values.
    /// Four-byte-loaded planes are aligned. Divisors are finite and positive.
    /// All allocations belong to `stream`'s context, remain live through
    /// completion, and do not overlap.
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
        if !input_scale_divisor.is_finite() || input_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(format!(
                "{}NVFP4 down input scale divisor must be finite and positive",
                E::LABEL
            )));
        }
        if !weight_scale_divisor.is_finite() || weight_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(format!(
                "{}NVFP4 down weight scale divisor must be finite and positive",
                E::LABEL
            )));
        }

        dispatch_nvfp4_down_prefill!(
            &self.prefill_routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module, stream, input, activation_codes, activation_scales, weight_codes,
                    weight_scales, input_scale_divisor, weight_scale_divisor, output,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "{}NVFP4 down prefill row count {rows} is outside the exact T=32,64,128,1024 routes",
                E::LABEL
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_WORDS_PER_PHASE, GROUP_K, GROUPS_PER_ROW, INPUT_COLUMNS, MAX_BATCH,
        Nvfp4DownDecodeRoutes, Nvfp4DownEntries, Nvfp4DownPrefillRoutes, OUTPUT_ROWS, PHASE_GROUPS,
        PHASES, PREFILL_ROWS, Qwen35Nvfp4DownEntries, Qwen38Nvfp4DownEntries, SHARED_U32, THREADS,
        W4_BLOCK_N, W4_TILE_M, WARPS, a16_launch_config, nvfp4_down_ptx_names, prefill_geometry,
        qwen35_nvfp4_down_ptx_names,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_gpu::LaunchConfig1D;
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    /// The decode routes every admitted architecture selects, in order.
    const DECODE_SCHEDULE: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    /// The prefill routes every admitted architecture selects, in order.
    const PREFILL_SCHEDULE: [usize; 4] = [32, 64, 128, 1_024];

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    #[test]
    fn exact_geometry_matches_the_source_owner() {
        assert_eq!(INPUT_COLUMNS, 17_408);
        assert_eq!(OUTPUT_ROWS, 5_120);
        assert_eq!(GROUPS_PER_ROW, 1_088);
        assert_eq!(PHASES, 34);
        assert_eq!(CODE_WORDS_PER_PHASE, 64);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);
    }

    #[test]
    fn qwen35_geometry_preserves_the_phase_and_row_ownership() {
        assert_eq!(Qwen35_9B::INTERMEDIATE / GROUP_K, 768);
        assert_eq!(Qwen35_9B::INTERMEDIATE / GROUP_K / PHASE_GROUPS, 24);
        assert_eq!(Qwen35_9B::HIDDEN / (2 * WARPS), 256);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);
        assert_eq!(PREFILL_ROWS, [32, 64, 128, 1_024]);
    }

    /// The shared launch geometry reproduces the exact grids the two replaced
    /// owners hard-coded: 320 and 256 A16 CTAs, 80 and 64 W4A4 output tiles,
    /// and the same 1/2/3/22 token tiles at T=32/64/128/1024.
    #[test]
    fn shared_geometry_reproduces_every_replaced_owner_grid() {
        assert_eq!(
            a16_launch_config::<Qwen38_27B>(),
            LaunchConfig1D::new(320, THREADS, 0)
        );
        assert_eq!(
            a16_launch_config::<Qwen35_9B>(),
            LaunchConfig1D::new(256, THREADS, 0)
        );

        for (rows, tiles) in [(32, 1), (64, 2), (128, 3), (1_024, 22)] {
            let qwen38 = prefill_geometry::<Qwen38_27B>(rows);
            let qwen35 = prefill_geometry::<Qwen35_9B>(rows);

            assert_eq!(qwen38.quantize_blocks, (rows * 1_088).div_ceil(256));
            assert_eq!(qwen38.projection_blocks, OUTPUT_ROWS / W4_BLOCK_N);
            assert_eq!(qwen38.projection_blocks, 80);
            assert_eq!(qwen38.token_tiles, tiles);
            assert_eq!(qwen35.quantize_blocks, (rows * 768).div_ceil(256));
            assert_eq!(qwen35.projection_blocks, 64);
            assert_eq!(qwen35.token_tiles, tiles);
            assert_eq!(qwen35.token_tiles, rows.div_ceil(W4_TILE_M));
        }
    }

    #[test]
    fn inventory_has_one_distinct_entry_per_batch() {
        let names = nvfp4_down_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH + 2 * PREFILL_ROWS.len());
        assert_eq!(unique.len(), names.len());

        let qwen35 = qwen35_nvfp4_down_ptx_names();
        let all = names.into_iter().chain(qwen35).collect::<BTreeSet<_>>();
        assert_eq!(qwen35.len(), MAX_BATCH + 2 * PREFILL_ROWS.len());
        assert_eq!(all.len(), 2 * MAX_BATCH + 4 * PREFILL_ROWS.len());
    }

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38Nvfp4DownEntries as Nvfp4DownEntries<Qwen38_27B>>::ptx_names(),
            nvfp4_down_ptx_names().to_vec()
        );
        assert_eq!(
            <Qwen35Nvfp4DownEntries as Nvfp4DownEntries<Qwen35_9B>>::ptx_names(),
            qwen35_nvfp4_down_ptx_names().to_vec()
        );
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this file is
    /// its per-base-name count. These are the counts the pinned SM120 device
    /// build emits; a wrapper change that instantiates one more specialization
    /// moves one of them.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in nvfp4_down_ptx_names()
            .into_iter()
            .chain(qwen35_nvfp4_down_ptx_names())
        {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("nvfp4_down_a16", 7),
                ("nvfp4_down_a16_b1", 1),
                ("nvfp4_down_quantize", 4),
                ("nvfp4_down_w4a4", 4),
                ("qwen35_nvfp4_down_a16", 8),
                ("qwen35_nvfp4_down_quantize", 4),
                ("qwen35_nvfp4_down_w4a4", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 32);
    }

    /// Route parity: the merged selectors reproduce, for every admitted row
    /// count, the exact arm each replaced dispatch took — `Nvfp4DownOp::launch`
    /// and `Qwen35Nvfp4DownOp::launch` both matched `B=1..=8` onto their own
    /// `b1..b8` fields, and both `launch_prefill` bodies matched
    /// `T=32,64,128,1024` onto `t32..t1024`. Neither owner admitted anything
    /// else, and the two domains never overlapped.
    #[test]
    fn row_routing_is_exact_and_disjoint() {
        assert_eq!(
            Nvfp4DownDecodeRoutes::<Qwen38_27B, Qwen38Nvfp4DownEntries>::admitted_rows(),
            DECODE_SCHEDULE
        );
        assert_eq!(
            Nvfp4DownPrefillRoutes::<Qwen38_27B, Qwen38Nvfp4DownEntries>::admitted_rows(),
            PREFILL_SCHEDULE
        );

        for rows in (0..=2_048).chain([usize::MAX]) {
            assert!(
                !Nvfp4DownDecodeRoutes::<Qwen38_27B, Qwen38Nvfp4DownEntries>::contains(rows)
                    || !Nvfp4DownPrefillRoutes::<Qwen38_27B, Qwen38Nvfp4DownEntries>::contains(
                        rows
                    ),
                "row count {rows} reaches both the decode and prefill schedules"
            );
        }
    }

    /// An unadmitted row count keeps naming the architecture that rejected it,
    /// with each owner's original wording preserved.
    #[test]
    fn unadmitted_row_counts_name_their_architecture() {
        for (operation, expected) in [
            (
                <Qwen38Nvfp4DownEntries as Nvfp4DownEntries<Qwen38_27B>>::OPERATION,
                "NVFP4 down projection",
            ),
            (
                <Qwen35Nvfp4DownEntries as Nvfp4DownEntries<Qwen35_9B>>::OPERATION,
                "Qwen3.5 NVFP4 down",
            ),
        ] {
            assert_eq!(operation, expected);
        }
        assert_eq!(
            <Qwen38Nvfp4DownEntries as Nvfp4DownEntries<Qwen38_27B>>::LABEL,
            ""
        );
        assert_eq!(
            <Qwen35Nvfp4DownEntries as Nvfp4DownEntries<Qwen35_9B>>::LABEL,
            "Qwen3.5 "
        );
    }
}

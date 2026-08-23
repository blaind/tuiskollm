use crate::device::fp8_projection::{e4m3x2_to_f32, load_u32x4_read_only, reduce_sum_lane_zero};
use cuda_device::async_copy::{
    cp_async_ca_16, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{DynamicSharedArray, convert, float, tcgen05, thread, warp};
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 16;
const VALUES_PER_PHASE: usize = 32 * VALUES_PER_LANE;
const WORDS_PER_LANE: usize = VALUES_PER_LANE / 4;
const FP8_MAX: f32 = 448.0;

#[inline(always)]
pub(crate) unsafe fn quantize_down_activation<A: Arch>(
    input: *const u32,
    codes: *mut u16,
    scale: *mut f32,
    warp_maximum: *mut f32,
) {
    let threads = thread::blockDim_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let token = thread::blockIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let pairs = A::INTERMEDIATE / 2;
    // SAFETY: one block owns one complete intermediate-width row.
    let input = unsafe { input.add(token * pairs) };
    // SAFETY: the code plane contains one packed byte pair per BF16 pair.
    let codes = unsafe { codes.add(token * pairs) };
    // SAFETY: the scale plane contains one value per launched block.
    let scale = unsafe { scale.add(token) };
    let mut maximum = 0.0f32;
    let mut pair = tid;

    while pair < pairs {
        // SAFETY: `pair` remains within this block's complete row.
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        maximum = maximum.max(low.abs()).max(high.abs());
        pair += threads;
    }

    maximum = warp::reduce_max_f32(maximum);
    if lane == 0 {
        // SAFETY: one lane writes its warp's unique shared slot.
        unsafe { *warp_maximum.add(warp_index) = maximum };
    }
    thread::sync_threads();

    if warp_index == 0 {
        maximum = if lane < threads / 32 {
            // SAFETY: the barrier published every active warp maximum.
            unsafe { *warp_maximum.add(lane) }
        } else {
            0.0
        };
        maximum = warp::reduce_max_f32(maximum);
        if lane == 0 {
            // SAFETY: lane zero owns this token's scale.
            unsafe {
                *scale = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
            }
        }
    }
    thread::sync_threads();

    // SAFETY: the second barrier makes the represented scale visible.
    let represented_scale = unsafe { *scale };
    pair = tid;
    while pair < pairs {
        // SAFETY: the read and write remain within this block's complete row.
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        unsafe {
            *codes.add(pair) = convert::cvt_rn_satfinite_e4m3x2_f32(
                float::div_rn_f32(low, represented_scale),
                float::div_rn_f32(high, represented_scale),
            );
        }
        pair += threads;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fp8_down_projection<A: Arch, const TOKENS: usize, const WARPS: usize>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let first_row = (thread::blockIdx_x() as usize * WARPS + (tid >> 5)) * 2;
    let words_per_row = A::INTERMEDIATE / 4;
    // SAFETY: the exact grid assigns this warp one complete adjacent row pair.
    let first_weight = unsafe { weight_codes.add(first_row * words_per_row) };
    // SAFETY: hidden rows are even and the paired row is admitted.
    let second_weight = unsafe { first_weight.add(words_per_row) };
    let mut first_sums = [0.0f32; TOKENS];
    let mut second_sums = [0.0f32; TOKENS];
    let mut phase = 0usize;

    while phase < A::INTERMEDIATE / VALUES_PER_PHASE {
        let lane_offset = phase * (VALUES_PER_PHASE / 4) + lane * WORDS_PER_LANE;
        // SAFETY: every phase/lane fragment is aligned and within its source row.
        let first_weight_words = unsafe { load_u32x4_read_only(first_weight.add(lane_offset)) };
        let second_weight_words = unsafe { load_u32x4_read_only(second_weight.add(lane_offset)) };
        let activation0 = unsafe { load_u32x4_read_only(activation_codes.add(lane_offset)) };
        let activation1 = if TOKENS > 1 {
            unsafe { load_u32x4_read_only(activation_codes.add(words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation2 = if TOKENS > 2 {
            unsafe { load_u32x4_read_only(activation_codes.add(2 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation3 = if TOKENS > 3 {
            unsafe { load_u32x4_read_only(activation_codes.add(3 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation4 = if TOKENS > 4 {
            unsafe { load_u32x4_read_only(activation_codes.add(4 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation5 = if TOKENS > 5 {
            unsafe { load_u32x4_read_only(activation_codes.add(5 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation6 = if TOKENS > 6 {
            unsafe { load_u32x4_read_only(activation_codes.add(6 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation7 = if TOKENS > 7 {
            unsafe { load_u32x4_read_only(activation_codes.add(7 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };

        macro_rules! packed_pair {
            ($words:ident, $pair:literal) => {{
                let word = match $pair >> 1 {
                    0 => $words.0,
                    1 => $words.1,
                    2 => $words.2,
                    _ => $words.3,
                };
                (word >> (($pair & 1) * 16)) as u16
            }};
        }
        macro_rules! accumulate_token {
            ($token:literal, $words:ident, $pair:literal, $first:ident, $second:ident) => {
                if TOKENS > $token {
                    let (low, high) = e4m3x2_to_f32(packed_pair!($words, $pair));
                    first_sums[$token] = float::fma_rn_f32($first.0, low, first_sums[$token]);
                    first_sums[$token] = float::fma_rn_f32($first.1, high, first_sums[$token]);
                    second_sums[$token] = float::fma_rn_f32($second.0, low, second_sums[$token]);
                    second_sums[$token] = float::fma_rn_f32($second.1, high, second_sums[$token]);
                }
            };
        }
        macro_rules! accumulate_pair {
            ($pair:literal) => {{
                let first = e4m3x2_to_f32(packed_pair!(first_weight_words, $pair));
                let second = e4m3x2_to_f32(packed_pair!(second_weight_words, $pair));
                accumulate_token!(0, activation0, $pair, first, second);
                accumulate_token!(1, activation1, $pair, first, second);
                accumulate_token!(2, activation2, $pair, first, second);
                accumulate_token!(3, activation3, $pair, first, second);
                accumulate_token!(4, activation4, $pair, first, second);
                accumulate_token!(5, activation5, $pair, first, second);
                accumulate_token!(6, activation6, $pair, first, second);
                accumulate_token!(7, activation7, $pair, first, second);
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
        phase += 1;
    }

    // SAFETY: the admitted source has one BF16 scale per hidden output row.
    let first_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_row) } as u32) << 16);
    let second_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_row + 1) } as u32) << 16);

    macro_rules! store_token {
        ($token:literal) => {
            if TOKENS > $token {
                let activation_scale = unsafe { *activation_scales.add($token) };
                let first_value = reduce_sum_lane_zero(first_sums[$token])
                    * activation_scale
                    * first_weight_scale;
                let second_value = reduce_sum_lane_zero(second_sums[$token])
                    * activation_scale
                    * second_weight_scale;
                if lane == 0 {
                    // SAFETY: lane zero uniquely owns this token's output-row pair.
                    unsafe {
                        *output.add($token * A::HIDDEN + first_row) =
                            tcgen05::cvt_f32x2_bf16x2(first_value, 0.0) as u16;
                        *output.add($token * A::HIDDEN + first_row + 1) =
                            tcgen05::cvt_f32x2_bf16x2(second_value, 0.0) as u16;
                    }
                }
            }
        };
    }

    store_token!(0);
    store_token!(1);
    store_token!(2);
    store_token!(3);
    store_token!(4);
    store_token!(5);
    store_token!(6);
    store_token!(7);
}

#[inline(always)]
unsafe fn store_down_prefill_tile<A: Arch, const TOKENS: usize>(
    values: [f32; 4],
    activation_scales: *const f32,
    weight_scales: *const u16,
    output: *mut u16,
    first_token: usize,
    first_output: usize,
) {
    // SAFETY: the exact tile maps both adjacent source scale words.
    let weight_scale0 = f32::from_bits((unsafe { *weight_scales.add(first_output) } as u32) << 16);
    // SAFETY: output rows are paired and the second scale is adjacent.
    let weight_scale1 =
        f32::from_bits((unsafe { *weight_scales.add(first_output + 1) } as u32) << 16);
    if first_token < TOKENS {
        // SAFETY: this lane uniquely owns the packed output pair.
        unsafe {
            let activation_scale = *activation_scales.add(first_token);
            *output
                .add(first_token * A::HIDDEN + first_output)
                .cast::<u32>() = tcgen05::cvt_f32x2_bf16x2(
                values[0] * activation_scale * weight_scale0,
                values[1] * activation_scale * weight_scale1,
            );
        }
    }

    let second_token = first_token + 8;
    if second_token < TOKENS {
        // SAFETY: the paired MMA row remains within the active-token extent.
        unsafe {
            let activation_scale = *activation_scales.add(second_token);
            *output
                .add(second_token * A::HIDDEN + first_output)
                .cast::<u32>() = tcgen05::cvt_f32x2_bf16x2(
                values[2] * activation_scale * weight_scale0,
                values[3] * activation_scale * weight_scale1,
            );
        }
    }
}

/// Projects one admitted dense-FP8 down tail with an exact 16x64 MMA tile.
#[inline(always)]
pub(crate) unsafe fn fp8_down_prefill_mma<
    A: Arch,
    const TOKENS: usize,
    const BM: usize,
    const BK_WORDS: usize,
    const K_SUBTILES: usize,
>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
    k_tiles: u32,
) {
    const OUTPUT_ROWS_PER_BLOCK: usize = 64;
    const STAGES: usize = 2;

    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let lane_group = lane >> 2;
    let thread_in_group = lane & 3;
    let words_per_row = A::INTERMEDIATE / 4;
    let output_tiles = A::HIDDEN / OUTPUT_ROWS_PER_BLOCK;
    let token_tile = block / output_tiles;
    let output_tile = block - token_tile * output_tiles;
    let first_token_tile = token_tile * BM;
    let first_output_tile = output_tile * OUTPUT_ROWS_PER_BLOCK;
    let warp_token = first_token_tile + (warp_index >> 1) * 16;
    let warp_output = first_output_tile + (warp_index & 1) * 32;
    let activation_tile = DynamicSharedArray::<u32, 16>::get();
    // SAFETY: two complete activation stages precede two weight stages.
    let weight_tile = unsafe { activation_tile.add(STAGES * BM * BK_WORDS) };
    let mut accumulator0 = [0.0f32; 4];
    let mut accumulator1 = [0.0f32; 4];
    let mut accumulator2 = [0.0f32; 4];
    let mut accumulator3 = [0.0f32; 4];

    debug_assert_eq!(thread::blockDim_x() as usize, (BM / 16) * 2 * 32);
    // SAFETY: the launch and dynamic-shared contract cover the first K stage.
    unsafe {
        let chunks_per_row = BK_WORDS / 4;
        let mut task = tid;
        while task < BM * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_ca_16(
                activation_tile.add(row * BK_WORDS + word),
                activation_codes.add((first_token_tile + row) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        task = tid;
        while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_cg_16(
                weight_tile.add(row * BK_WORDS + word),
                weight_codes.add((first_output_tile + row) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        cp_async_commit_group();
    }

    let mut stage = 0usize;
    let mut k_tile = 0usize;
    while k_tile < k_tiles as usize {
        // SAFETY: every next-stage copy remains inside both source planes.
        unsafe {
            if k_tile + 1 < k_tiles as usize {
                let next_stage = stage ^ 1;
                let next_word = (k_tile + 1) * BK_WORDS;
                let chunks_per_row = BK_WORDS / 4;
                let mut task = tid;
                while task < BM * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_ca_16(
                        activation_tile.add(next_stage * BM * BK_WORDS + row * BK_WORDS + word),
                        activation_codes
                            .add((first_token_tile + row) * words_per_row + next_word + word),
                    );
                    task += thread::blockDim_x() as usize;
                }
                task = tid;
                while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_cg_16(
                        weight_tile.add(
                            next_stage * OUTPUT_ROWS_PER_BLOCK * BK_WORDS + row * BK_WORDS + word,
                        ),
                        weight_codes
                            .add((first_output_tile + row) * words_per_row + next_word + word),
                    );
                    task += thread::blockDim_x() as usize;
                }
                cp_async_commit_group();
                cp_async_wait_group(1);
            } else {
                cp_async_wait_group(0);
            }
        }
        thread::sync_threads();

        let activation_base = stage * BM * BK_WORDS + (warp_index >> 1) * 16 * BK_WORDS;
        let first_weight_base =
            stage * OUTPUT_ROWS_PER_BLOCK * BK_WORDS + (warp_index & 1) * 32 * BK_WORDS;
        let weight0 = first_weight_base;
        let weight1 = weight0 + 8 * BK_WORDS;
        let weight2 = weight1 + 8 * BK_WORDS;
        let weight3 = weight2 + 8 * BK_WORDS;
        let mut k_subtile = 0usize;
        while k_subtile < K_SUBTILES {
            let k_offset = k_subtile * 8;
            // SAFETY: each lane group loads one complete m16n8k32 fragment.
            let activation_fragment = unsafe {
                [
                    *activation_tile
                        .add(activation_base + lane_group * BK_WORDS + k_offset + thread_in_group),
                    *activation_tile.add(
                        activation_base + (lane_group + 8) * BK_WORDS + k_offset + thread_in_group,
                    ),
                    *activation_tile.add(
                        activation_base + lane_group * BK_WORDS + k_offset + thread_in_group + 4,
                    ),
                    *activation_tile.add(
                        activation_base
                            + (lane_group + 8) * BK_WORDS
                            + k_offset
                            + thread_in_group
                            + 4,
                    ),
                ]
            };
            macro_rules! weight_fragment {
                ($base:ident) => {
                    [
                        *weight_tile
                            .add($base + lane_group * BK_WORDS + k_offset + thread_in_group),
                        *weight_tile
                            .add($base + lane_group * BK_WORDS + k_offset + thread_in_group + 4),
                    ]
                };
            }
            accumulator0 = unsafe {
                cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator0,
                    activation_fragment,
                    weight_fragment!(weight0),
                )
            };
            accumulator1 = unsafe {
                cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator1,
                    activation_fragment,
                    weight_fragment!(weight1),
                )
            };
            accumulator2 = unsafe {
                cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator2,
                    activation_fragment,
                    weight_fragment!(weight2),
                )
            };
            accumulator3 = unsafe {
                cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator3,
                    activation_fragment,
                    weight_fragment!(weight3),
                )
            };
            k_subtile += 1;
        }
        thread::sync_threads();
        stage ^= 1;
        k_tile += 1;
    }

    let first_token = warp_token + lane_group;
    let first_output = warp_output + thread_in_group * 2;
    // SAFETY: the route warps partition every active 16x64 output tile.
    unsafe {
        store_down_prefill_tile::<A, TOKENS>(
            accumulator0,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output,
        );
        store_down_prefill_tile::<A, TOKENS>(
            accumulator1,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 8,
        );
        store_down_prefill_tile::<A, TOKENS>(
            accumulator2,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 16,
        );
        store_down_prefill_tile::<A, TOKENS>(
            accumulator3,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 24,
        );
    }
}

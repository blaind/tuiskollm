use crate::device::fp8_projection::{e4m3x2_to_f32, load_u32x4_read_only, reduce_sum_lane_zero};
use cuda_device::{convert, float, tcgen05, thread, warp};
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

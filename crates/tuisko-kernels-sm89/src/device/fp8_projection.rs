use cuda_device::{convert, float, ptx_asm, thread, warp};
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 16;
const VALUES_PER_PHASE: usize = 32 * VALUES_PER_LANE;
const WORDS_PER_LANE: usize = VALUES_PER_LANE / 4;
const FP8_MAX: f32 = 448.0;

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
fn e4m3x2_to_f32(packed: u16) -> (f32, f32) {
    let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(packed);

    convert::cvt_f32x2_f16x2(packed_f16)
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
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[inline(always)]
pub(crate) unsafe fn quantize_activation<A: Arch>(
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
    let pairs = A::HIDDEN / 2;
    let input = unsafe { input.add(token * pairs) };
    let codes = unsafe { codes.add(token * pairs) };
    let scale = unsafe { scale.add(token) };
    let mut maximum = 0.0f32;
    let mut pair = tid;

    while pair < pairs {
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        maximum = maximum.max(low.abs()).max(high.abs());
        pair += threads;
    }

    maximum = warp::reduce_max_f32(maximum);
    if lane == 0 {
        unsafe { *warp_maximum.add(warp_index) = maximum };
    }
    thread::sync_threads();

    if warp_index == 0 {
        maximum = if lane < threads / 32 {
            unsafe { *warp_maximum.add(lane) }
        } else {
            0.0
        };
        maximum = warp::reduce_max_f32(maximum);
        if lane == 0 {
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

    let represented_scale = unsafe { *scale };
    pair = tid;
    while pair < pairs {
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
pub(crate) unsafe fn fp8_projection<
    const INPUT_COLUMNS: usize,
    const TOKENS: usize,
    const WARPS: usize,
>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
    output_rows: usize,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let first_row = (thread::blockIdx_x() as usize * WARPS + (tid >> 5)) * 2;
    let words_per_row = INPUT_COLUMNS / 4;
    let first_weight = unsafe { weight_codes.add(first_row * words_per_row) };
    let second_weight = unsafe { first_weight.add(words_per_row) };
    let mut first_sums = [0.0f32; TOKENS];
    let mut second_sums = [0.0f32; TOKENS];
    let mut phase = 0usize;

    while phase < INPUT_COLUMNS / VALUES_PER_PHASE {
        let lane_offset = phase * (VALUES_PER_PHASE / 4) + lane * WORDS_PER_LANE;
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
                    unsafe {
                        *output.add($token * output_rows + first_row) = f32_to_bf16(first_value);
                        *output.add($token * output_rows + first_row + 1) =
                            f32_to_bf16(second_value);
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

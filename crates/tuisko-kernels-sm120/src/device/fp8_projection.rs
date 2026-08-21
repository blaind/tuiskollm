use cuda_device::async_copy::{
    cp_async_ca_16, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{DynamicSharedArray, convert, float, ptx_asm, tcgen05, thread, warp};
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

    // SAFETY: the caller guarantees one aligned 16-byte source fragment.
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
    // SAFETY: one block owns one complete input row and its output code row.
    let input = unsafe { input.add(token * pairs) };
    // SAFETY: the code plane contains one packed byte pair per BF16 pair.
    let codes = unsafe { codes.add(token * pairs) };
    // SAFETY: the scale plane contains one value per launched block.
    let scale = unsafe { scale.add(token) };
    let mut maximum = 0.0f32;
    let mut pair = tid;

    while pair < pairs {
        // SAFETY: `pair < pairs` within this block's row.
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
            // SAFETY: lane zero owns the token's scale output.
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
        // SAFETY: the read and write are within this block's complete row.
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        // SAFETY: each thread writes disjoint packed E4M3 byte pairs.
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
pub(crate) unsafe fn fp8_projection<A: Arch, const TOKENS: usize, const WARPS: usize>(
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
    let words_per_row = A::HIDDEN / 4;
    // SAFETY: the exact grid assigns this warp one complete adjacent row pair.
    let first_weight = unsafe { weight_codes.add(first_row * words_per_row) };
    // SAFETY: output rows are even and the row pair is within the admitted plane.
    let second_weight = unsafe { first_weight.add(words_per_row) };
    let mut first_sums = [0.0f32; TOKENS];
    let mut second_sums = [0.0f32; TOKENS];
    let mut phase = 0usize;

    while phase < A::HIDDEN / VALUES_PER_PHASE {
        let lane_offset = phase * (VALUES_PER_PHASE / 4) + lane * WORDS_PER_LANE;
        // SAFETY: every phase/lane fragment is 16-byte aligned and within one row.
        let first_weight_words = unsafe { load_u32x4_read_only(first_weight.add(lane_offset)) };
        // SAFETY: every phase/lane fragment is 16-byte aligned and within one row.
        let second_weight_words = unsafe { load_u32x4_read_only(second_weight.add(lane_offset)) };
        // SAFETY: each admitted token owns one complete activation-code row.
        let activation0 = unsafe { load_u32x4_read_only(activation_codes.add(lane_offset)) };
        let activation1 = if TOKENS > 1 {
            // SAFETY: the const route admits the second complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation2 = if TOKENS > 2 {
            // SAFETY: the const route admits the third complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(2 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation3 = if TOKENS > 3 {
            // SAFETY: the const route admits the fourth complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(3 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation4 = if TOKENS > 4 {
            // SAFETY: the const route admits the fifth complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(4 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation5 = if TOKENS > 5 {
            // SAFETY: the const route admits the sixth complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(5 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation6 = if TOKENS > 6 {
            // SAFETY: the const route admits the seventh complete activation row.
            unsafe { load_u32x4_read_only(activation_codes.add(6 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation7 = if TOKENS > 7 {
            // SAFETY: the const route admits the eighth complete activation row.
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

    // SAFETY: the admitted scale plane has one BF16 scale per output row.
    let first_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_row) } as u32) << 16);
    // SAFETY: the paired row has its adjacent BF16 scale.
    let second_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_row + 1) } as u32) << 16);

    macro_rules! store_token {
        ($token:literal) => {
            if TOKENS > $token {
                // SAFETY: every admitted token has one activation scale.
                let activation_scale = unsafe { *activation_scales.add($token) };
                let first_value = reduce_sum_lane_zero(first_sums[$token])
                    * activation_scale
                    * first_weight_scale;
                let second_value = reduce_sum_lane_zero(second_sums[$token])
                    * activation_scale
                    * second_weight_scale;
                if lane == 0 {
                    // SAFETY: lane zero writes this warp's unique row pair.
                    unsafe {
                        *output.add($token * output_rows + first_row) =
                            tcgen05::cvt_f32x2_bf16x2(first_value, 0.0) as u16;
                        *output.add($token * output_rows + first_row + 1) =
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
unsafe fn store_qkv_mma_tile<A: Arch>(
    values: [f32; 4],
    activation_scales: *const f32,
    weight_scales: *const u16,
    output: *mut u16,
    first_token: usize,
    first_output: usize,
) {
    // SAFETY: the exact QKV tile maps both scale words within the source plane.
    let first_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_output) } as u32) << 16);
    // SAFETY: output columns are paired and the second scale is adjacent.
    let second_weight_scale =
        f32::from_bits((unsafe { *weight_scales.add(first_output + 1) } as u32) << 16);
    // SAFETY: the T=16 tile maps both token rows and output columns uniquely.
    unsafe {
        let activation_scale = *activation_scales.add(first_token);
        *output.add(first_token * A::ATTENTION_QKV_ROWS + first_output) =
            tcgen05::cvt_f32x2_bf16x2(values[0] * activation_scale * first_weight_scale, 0.0)
                as u16;
        *output.add(first_token * A::ATTENTION_QKV_ROWS + first_output + 1) =
            tcgen05::cvt_f32x2_bf16x2(values[1] * activation_scale * second_weight_scale, 0.0)
                as u16;

        let second_token = first_token + 8;
        let activation_scale = *activation_scales.add(second_token);
        *output.add(second_token * A::ATTENTION_QKV_ROWS + first_output) =
            tcgen05::cvt_f32x2_bf16x2(values[2] * activation_scale * first_weight_scale, 0.0)
                as u16;
        *output.add(second_token * A::ATTENTION_QKV_ROWS + first_output + 1) =
            tcgen05::cvt_f32x2_bf16x2(values[3] * activation_scale * second_weight_scale, 0.0)
                as u16;
    }
}

/// Projects exactly 16 quantized rows with the retained two-warp tensor-core tile.
#[inline(always)]
pub(crate) unsafe fn qkv_projection_mma_t16<A: Arch>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
    k_tiles: u32,
) {
    const TOKENS: usize = 16;
    const OUTPUT_ROWS_PER_BLOCK: usize = 64;
    const K_WORDS: usize = 32;
    const K_SUBTILES: usize = 4;
    const STAGES: usize = 2;

    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let lane_group = lane >> 2;
    let thread_in_group = lane & 3;
    let words_per_row = A::HIDDEN / 4;
    let output_tile = block % (A::ATTENTION_QKV_ROWS / OUTPUT_ROWS_PER_BLOCK);
    let first_output_tile = output_tile * OUTPUT_ROWS_PER_BLOCK;
    let warp_output = first_output_tile + warp_index * 32;
    let activation_tile = DynamicSharedArray::<u32, 16>::get();
    // SAFETY: two 16-row activation stages precede the weight stages.
    let weight_tile = unsafe { activation_tile.add(STAGES * TOKENS * K_WORDS) };
    let mut accumulator0 = [0.0f32; 4];
    let mut accumulator1 = [0.0f32; 4];
    let mut accumulator2 = [0.0f32; 4];
    let mut accumulator3 = [0.0f32; 4];

    debug_assert_eq!(thread::blockDim_x(), 64);
    // SAFETY: the launch and dynamic-shared contract cover the first K stage.
    unsafe {
        let chunks_per_row = K_WORDS / 4;
        let mut task = tid;
        while task < TOKENS * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_ca_16(
                activation_tile.add(row * K_WORDS + word),
                activation_codes.add(row * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        task = tid;
        while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_cg_16(
                weight_tile.add(row * K_WORDS + word),
                weight_codes.add((first_output_tile + row) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        cp_async_commit_group();
    }

    let mut stage = 0usize;
    let mut k_tile = 0usize;
    while k_tile < k_tiles as usize {
        // SAFETY: every next-stage copy remains within the admitted source planes and shared arena.
        unsafe {
            if k_tile + 1 < k_tiles as usize {
                let next_stage = stage ^ 1;
                let next_word = (k_tile + 1) * K_WORDS;
                let chunks_per_row = K_WORDS / 4;
                let mut task = tid;
                while task < TOKENS * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_ca_16(
                        activation_tile.add(next_stage * TOKENS * K_WORDS + row * K_WORDS + word),
                        activation_codes.add(row * words_per_row + next_word + word),
                    );
                    task += thread::blockDim_x() as usize;
                }
                task = tid;
                while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_cg_16(
                        weight_tile.add(
                            next_stage * OUTPUT_ROWS_PER_BLOCK * K_WORDS + row * K_WORDS + word,
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

        let activation_base = stage * TOKENS * K_WORDS;
        let first_weight_base = stage * OUTPUT_ROWS_PER_BLOCK * K_WORDS + warp_index * 32 * K_WORDS;
        let weight0 = first_weight_base;
        let weight1 = weight0 + 8 * K_WORDS;
        let weight2 = weight1 + 8 * K_WORDS;
        let weight3 = weight2 + 8 * K_WORDS;
        let mut k_subtile = 0usize;
        while k_subtile < K_SUBTILES {
            let k_offset = k_subtile * 8;
            // SAFETY: each lane group loads the fragments required by m16n8k32.
            let activation_fragment = unsafe {
                [
                    *activation_tile
                        .add(activation_base + lane_group * K_WORDS + k_offset + thread_in_group),
                    *activation_tile.add(
                        activation_base + (lane_group + 8) * K_WORDS + k_offset + thread_in_group,
                    ),
                    *activation_tile.add(
                        activation_base + lane_group * K_WORDS + k_offset + thread_in_group + 4,
                    ),
                    *activation_tile.add(
                        activation_base
                            + (lane_group + 8) * K_WORDS
                            + k_offset
                            + thread_in_group
                            + 4,
                    ),
                ]
            };
            macro_rules! weight_fragment {
                ($base:ident) => {
                    [
                        *weight_tile.add($base + lane_group * K_WORDS + k_offset + thread_in_group),
                        *weight_tile
                            .add($base + lane_group * K_WORDS + k_offset + thread_in_group + 4),
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

    let first_token = lane_group;
    let first_output = warp_output + thread_in_group * 2;
    // SAFETY: the two warps partition the complete 16x64 output tile.
    unsafe {
        store_qkv_mma_tile::<A>(
            accumulator0,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output,
        );
        store_qkv_mma_tile::<A>(
            accumulator1,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 8,
        );
        store_qkv_mma_tile::<A>(
            accumulator2,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 16,
        );
        store_qkv_mma_tile::<A>(
            accumulator3,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 24,
        );
    }
}

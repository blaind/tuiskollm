//! Plain source-BF16 matrix projection shared by the backbone and LM-head families.
//!
//! One `nn.Linear` with BF16 weights and a BF16 activation row accumulates in
//! FP32 across the whole contraction and rounds to BF16 exactly once, at the
//! store. Both bodies below implement that and nothing else: no bias, no gate,
//! no activation. A caller that needs an epilogue owns a separate entry for it,
//! which is what lets one projection serve call sites whose epilogues differ.
//!
//! The geometry travels as const parameters rather than [`Arch`] members
//! because the four widths this artifact projects at are drawn from four
//! different constants. The arithmetic is identical at every one of them: a
//! warp owns one `m16n8k16` output tile of eight adjacent weight rows and walks
//! the contraction in ascending column order.
//!
//! [`Arch`]: tuisko_model::Arch

use cuda_device::{tcgen05, thread, wmma};

/// Weight rows one warp publishes per output tile: the MMA's `N`.
pub const ROWS_PER_TILE: usize = 8;
/// Token rows one prompt tile publishes: the MMA's `M`.
pub const TOKENS_PER_TILE: usize = 16;
/// Contraction columns consumed by one MMA step: the MMA's `K`.
pub const COLUMNS_PER_STEP: usize = 16;

/// Whether a projection geometry tiles the exact BF16 MMA shapes.
///
/// Both bodies read the result as a compile-time constant, so a geometry that
/// does not tile cannot be instantiated rather than being rejected at launch.
#[must_use]
pub const fn tiles_exactly(columns: usize, output_rows: usize, warps: usize) -> bool {
    columns.is_multiple_of(128) && output_rows.is_multiple_of(ROWS_PER_TILE * warps) && warps > 0
}

#[inline(always)]
unsafe fn input_pair<const COLUMNS: usize, const TOKENS: usize>(
    input: *const u32,
    row: usize,
    column: usize,
) -> u32 {
    if row >= TOKENS {
        return 0;
    }

    // SAFETY: the exact route owns `TOKENS` complete `COLUMNS`-wide BF16 rows.
    unsafe { *input.add(row * (COLUMNS / 2) + column / 2) }
}

#[inline(always)]
unsafe fn weight_pair<const COLUMNS: usize>(weight: *const u32, row: usize, column: usize) -> u32 {
    // SAFETY: the source plane is the exact `[OUTPUT_ROWS, COLUMNS]` BF16 matrix.
    unsafe { *weight.add(row * (COLUMNS / 2) + column / 2) }
}

/// Projects up to `TOKENS` decode rows through a source-BF16 `[OUTPUT_ROWS, COLUMNS]` plane.
///
/// The grid covers every eight-row output tile once: `OUTPUT_ROWS / 8 / WARPS`
/// blocks of `WARPS * 32` threads. Rows at or above `TOKENS` contribute zeroed
/// activation fragments and publish nothing, so no padded row reaches `output`.
///
/// # Safety
///
/// `input` covers `TOKENS * COLUMNS` BF16 values, `weight` covers the
/// `[OUTPUT_ROWS, COLUMNS]` BF16 plane, and `output` covers
/// `TOKENS * OUTPUT_ROWS` BF16 values. All three are four-byte aligned and
/// non-overlapping.
#[inline(always)]
pub unsafe fn bf16_projection_decode<
    const COLUMNS: usize,
    const OUTPUT_ROWS: usize,
    const WARPS: usize,
    const TOKENS: usize,
>(
    input: *const u32,
    weight: *const u32,
    output: *mut u32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let group = lane >> 2;
    let thread_in_group = lane & 3;
    let output_tile = thread::blockIdx_x() as usize * WARPS + warp_index;
    let weight_row = output_tile * ROWS_PER_TILE + group;
    let mut accumulator = [0.0f32; 4];
    let mut column = 0usize;

    macro_rules! k_step {
        ($column:expr) => {{
            let column = $column;
            // m16n8k16 is the smallest native BF16 Tensor Core tile, and the
            // whole contraction accumulates into one FP32 register file.
            let activation = unsafe {
                [
                    input_pair::<COLUMNS, TOKENS>(input, group, column + 2 * thread_in_group),
                    input_pair::<COLUMNS, TOKENS>(input, group + 8, column + 2 * thread_in_group),
                    input_pair::<COLUMNS, TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                    input_pair::<COLUMNS, TOKENS>(
                        input,
                        group + 8,
                        column + 8 + 2 * thread_in_group,
                    ),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair::<COLUMNS>(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair::<COLUMNS>(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            // SAFETY: all lanes execute the same row-major A / column-major B MMA.
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
        }};
    }

    // The narrow exact routes fold most activation rows to constants and lose
    // load depth; four K-blocks per iteration restore the weight pipeline, and
    // the wider batches take eight. Unroll width never reorders the
    // accumulation: every step folds ascending columns into one accumulator.
    if TOKENS <= 2 {
        while column < COLUMNS {
            k_step!(column);
            k_step!(column + 16);
            k_step!(column + 32);
            k_step!(column + 48);
            column += 64;
        }
    } else {
        while column < COLUMNS {
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
        let output_column_word = output_tile * 4 + thread_in_group;
        // SAFETY: the lower fragment maps to one active token and one output pair.
        unsafe {
            *output.add(group * (OUTPUT_ROWS / 2) + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
        }
    }
}

/// Projects one prompt tile through a source-BF16 `[OUTPUT_ROWS, COLUMNS]` plane.
///
/// The grid is `OUTPUT_ROWS / 8 / WARPS` output blocks per sixteen-row token
/// tile, so `TOKENS` must be a multiple of sixteen. Both native accumulator
/// halves are published, which is what makes a prompt tile twice as dense per
/// MMA as a decode batch at the same weight traffic.
///
/// # Safety
///
/// Carries [`bf16_projection_decode`]'s contract with `TOKENS` complete rows.
#[inline(always)]
pub unsafe fn bf16_projection_prefill<
    const COLUMNS: usize,
    const OUTPUT_ROWS: usize,
    const WARPS: usize,
    const TOKENS: usize,
>(
    input: *const u32,
    weight: *const u32,
    output: *mut u32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let group = lane >> 2;
    let thread_in_group = lane & 3;
    let block = thread::blockIdx_x() as usize;
    let blocks = OUTPUT_ROWS / ROWS_PER_TILE / WARPS;
    let output_block = block % blocks;
    let token_tile = block / blocks;
    let output_tile = output_block * WARPS + warp_index;
    let weight_row = output_tile * ROWS_PER_TILE + group;
    let token_row = token_tile * TOKENS_PER_TILE + group;
    let mut accumulator = [0.0f32; 4];
    let mut column = 0usize;

    while column < COLUMNS {
        let activation = unsafe {
            [
                input_pair::<COLUMNS, TOKENS>(input, token_row, column + 2 * thread_in_group),
                input_pair::<COLUMNS, TOKENS>(input, token_row + 8, column + 2 * thread_in_group),
                input_pair::<COLUMNS, TOKENS>(input, token_row, column + 8 + 2 * thread_in_group),
                input_pair::<COLUMNS, TOKENS>(
                    input,
                    token_row + 8,
                    column + 8 + 2 * thread_in_group,
                ),
            ]
        };
        let weights = unsafe {
            [
                weight_pair::<COLUMNS>(weight, weight_row, column + 2 * thread_in_group),
                weight_pair::<COLUMNS>(weight, weight_row, column + 8 + 2 * thread_in_group),
            ]
        };
        // SAFETY: all lanes execute the same row-major A / column-major B MMA.
        accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
        column += COLUMNS_PER_STEP;
    }

    let output_column_word = output_tile * 4 + thread_in_group;
    // SAFETY: a sixteen-row token tile owns both accumulator halves outright.
    unsafe {
        *output.add(token_row * (OUTPUT_ROWS / 2) + output_column_word) =
            tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
        *output.add((token_row + 8) * (OUTPUT_ROWS / 2) + output_column_word) =
            tcgen05::cvt_f32x2_bf16x2(accumulator[2], accumulator[3]);
    }
}

//! Exact Qwen3.8-Flash-Next PLE operators.
//!
//! Layer 1 injects this n-gram delta before its attention hyper-connection:
//!
//! ```text
//! E            = dequant(codes) * table_scale             # [2560] BF16
//! key_normed   = norm_key(key_proj(E))                    # [10240], grouped
//! value        = value_proj(E)                            # [2560]
//! query_normed = norm_query(hidden)                       # [10240], grouped
//! gate_c       = sum_j key_normed[c,j] * query_normed[c,j] / sqrt(2560)
//! gate_c       = |gate_c|.max(1e-6).sqrt() * sign(gate_c) # signed sqrt
//! gated[c,j]   = sigmoid(gate_c) * value[j]               # [10240] flattened
//! gated_normed = norm_conv(gated)                         # [10240], grouped
//! out          = gated + silu(short_conv(gated_normed))   # [10240]
//! hidden      += out
//! ```
//!
//! The three grouped norms reuse the hyper-connection module's exact entry.
//! PLE owns its depthwise convolution because dilation 3 requires nine history
//! columns and taps `t-9, t-6, t-3, t`. Dequantization, projection, gate,
//! convolution, and injection remain observable BF16 boundaries.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_hyper_connection::Qwen38FlashNextHyperConnectionOp;
use tuisko_model::{Arch, Qwen38FlashNext};

/// Compact batching owns one compiled route for every `B=1..8`.
const MAX_BATCH: usize = 8;
/// Prefill tile widths this family admits, matching the other per-token owners.
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

/// Parallel residual branches carried by the widened stream.
const BRANCHES: usize = Qwen38FlashNext::HC_COUNT;
/// Width of one branch, which is also the engram embedding width.
const BRANCH: usize = Qwen38FlashNext::HIDDEN;
/// Width of the widened residual stream this module injects into.
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
/// Width of the gathered engram embedding row.
const EMBED: usize = Qwen38FlashNext::PLE_EMBED_DIM;
/// Taps in the short convolution.
const CONV_TAPS: usize = Qwen38FlashNext::PLE_CONV_KERNEL;
/// Tap spacing: the reference ties the dilation to `ngram_size`.
const CONV_DILATION: usize = Qwen38FlashNext::PLE_CONV_DILATION;
/// Columns of convolution history one sequence carries.
const CONV_STATE: usize = Qwen38FlashNext::PLE_CONV_STATE_LEN;
/// Floor the signed square root clamps the gate magnitude to.
const GATE_FLOOR: f32 = Qwen38FlashNext::PLE_GATE_FLOOR;
/// The literal `sqrt(2560)` the reference divides the gate dot product by.
///
/// It is a scalar divide of an already-BF16 tensor, never a folded weight scale
/// and never a multiply by a precomputed reciprocal.
const GATE_DIVISOR: f32 = 50.596_443_f32;

/// Packed BF16 words in one widened residual row.
const ROW_WORDS: usize = WIDTH / 2;
/// Packed BF16 words in one branch of one widened residual row.
const BRANCH_WORDS: usize = BRANCH / 2;
/// Packed BF16 words in one engram embedding row.
const EMBED_WORDS: usize = EMBED / 2;
/// Packed E4M3 code pairs in one staged engram plane row.
const CODE_WORDS: usize = EMBED / 2;

// The dequantization, the gate, and the injection all walk one row per CTA at
// 256 threads, which is the same mapping the grouped norm this family reuses
// already fixes: 1,280 packed pairs per branch, exactly five per thread.
const ROW_WARPS: usize = 8;
const ROW_THREADS: u32 = (ROW_WARPS * 32) as u32;
/// Packed pairs one thread consumes per branch of the gate reduction.
const GATE_PAIRS_PER_THREAD: usize = BRANCH_WORDS / ROW_THREADS as usize;

// The fused projection is a warp-per-output-row GEMV: one warp owns one output
// row's whole reduction, so the accumulation order is fixed by the lane stride
// and the five-step butterfly and never varies with the row count.
// Register-blocking eight tokens per CTA lets a weight row loaded once serve a
// whole decode batch or one prefill tile column.
const TOKEN_TILE: usize = MAX_BATCH;
const PROJECT_WARPS: usize = 8;
const PROJECT_THREADS: u32 = (PROJECT_WARPS * 32) as u32;
/// Output rows the fused projection owns: the key rows then the value rows.
const PROJECT_ROWS: usize = WIDTH + BRANCH;
/// CTAs that cover those rows, eight rows each.
const PROJECT_ROW_GROUPS: usize = PROJECT_ROWS / PROJECT_WARPS;
/// Row group at which the projection crosses from `key_proj` into `value_proj`.
const PROJECT_KEY_GROUPS: usize = WIDTH / PROJECT_WARPS;
/// Packed words one lane consumes per phase of the embedding reduction.
const WORDS_PER_LANE: usize = 4;
/// Packed words one warp consumes per phase of the embedding reduction.
const PROJECT_PHASE_WORDS: usize = 32 * WORDS_PER_LANE;
/// Phases one warp needs to cover a complete embedding row.
const PROJECT_PHASES: usize = EMBED_WORDS / PROJECT_PHASE_WORDS;

/// Threads per convolution CTA; the traversal is one channel per thread.
const CONV_THREADS: u32 = 256;
/// CTAs the history publication needs to cover every channel.
const CONV_HISTORY_BLOCKS: u32 = (WIDTH / CONV_THREADS as usize) as u32;

const _: () = assert!(BRANCHES == 4);
const _: () = assert!(BRANCH == 2_560);
const _: () = assert!(WIDTH == 10_240);
const _: () = assert!(EMBED == 2_560);
const _: () = assert!(WIDTH == BRANCHES * BRANCH);
const _: () = assert!(CONV_TAPS == 4);
const _: () = assert!(CONV_DILATION == 3);
const _: () = assert!(CONV_STATE == (CONV_TAPS - 1) * CONV_DILATION);
const _: () = assert!(GATE_PAIRS_PER_THREAD * ROW_THREADS as usize == BRANCH_WORDS);
const _: () = assert!(PROJECT_ROWS.is_multiple_of(PROJECT_WARPS));
const _: () = assert!(WIDTH.is_multiple_of(PROJECT_WARPS));
const _: () = assert!(EMBED_WORDS.is_multiple_of(PROJECT_PHASE_WORDS));
const _: () = assert!(WIDTH.is_multiple_of(CONV_THREADS as usize));
// Every admitted prefill tile is at least as wide as the convolution state, so
// the history publication never has to splice a partial window.
const _: () = assert!(PREFILL_ROWS[0] >= CONV_STATE);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, warp};
    use tuisko_kernels_sm120_common::device::{e4m3x2_to_f32, load_u32x4_read_only};

    #[inline(always)]
    fn bf16_to_f32(bits: u16) -> f32 {
        convert::cvt_f32x2_bf16x2(bits as u32).0
    }

    /// Rounds an FP32 intermediate at a represented BF16 boundary.
    #[inline(always)]
    fn round_bf16(value: f32) -> f32 {
        bf16_to_f32(tcgen05::f32_to_bf16_rne(value))
    }

    #[inline(always)]
    fn silu(value: f32) -> f32 {
        value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    /// Five-step butterfly that leaves the warp's whole reduction on lane zero.
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
    fn block_sum(value: f32, shared: *mut f32, lane: usize, warp_index: usize) -> f32 {
        let value = warp::reduce_sum_f32(value);
        if lane == 0 {
            // SAFETY: one lane writes its warp's unique shared slot.
            unsafe { *shared.add(warp_index) = value };
        }
        thread::sync_threads();

        if warp_index == 0 {
            let value = if lane < ROW_WARPS {
                // SAFETY: the first warp reads the initialized warp-sum slots.
                unsafe { *shared.add(lane) }
            } else {
                0.0
            };
            let value = warp::reduce_sum_f32(value);
            if lane == 0 {
                // SAFETY: lane zero publishes the block sum before the barrier.
                unsafe { *shared = value };
            }
        }
        thread::sync_threads();

        // SAFETY: the second barrier makes the published block sum visible.
        unsafe { *shared }
    }

    /// Widens one staged engram row from its E4M3 codes.
    ///
    /// `*.ple.*` is outside the ModelOpt quantization list, so `scale` is a
    /// plain positive multiplier over the codes and never a reciprocal divisor.
    /// Every E4M3 value is exact in BF16, so the single rounding site is the
    /// product.
    #[inline(always)]
    unsafe fn ple_dequant_body<const TOKENS: usize>(
        codes: *const u16,
        scale: f32,
        embedding: *mut u32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: the launch contract gives every active block one code row.
        let codes = unsafe { codes.add(token * CODE_WORDS) };
        // SAFETY: the output plane has the code plane's row coverage.
        let embedding = unsafe { embedding.add(token * EMBED_WORDS) };
        let mut word = tid;

        while word < CODE_WORDS {
            // SAFETY: `word < CODE_WORDS` stays inside this token's row.
            let (low, high) = e4m3x2_to_f32(unsafe { *codes.add(word) });
            // SAFETY: each thread writes disjoint packed BF16 pairs.
            unsafe {
                *embedding.add(word) = tcgen05::cvt_f32x2_bf16x2(low * scale, high * scale);
            }
            word += ROW_THREADS as usize;
        }
    }

    /// Projects one dequantized row through `key_proj` and `value_proj`.
    ///
    /// The two projections share an entry because they contract the same
    /// 2,560-wide row; the row group selects which plane a warp reads and which
    /// output it writes. Both are `nn.Linear` calls, so each result is rounded
    /// to BF16 once.
    #[inline(always)]
    unsafe fn ple_project_body<const TOKENS: usize>(
        embedding: *const u32,
        key_weight: *const u32,
        value_weight: *const u32,
        key: *mut u16,
        value: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let block = thread::blockIdx_x() as usize;
        let row_group = block % PROJECT_ROW_GROUPS;
        let token_base = if TOKENS <= TOKEN_TILE {
            0
        } else {
            (block / PROJECT_ROW_GROUPS) * TOKEN_TILE
        };
        let value_group = row_group >= PROJECT_KEY_GROUPS;
        let row = row_group * PROJECT_WARPS + warp_index;
        let plane_row = if value_group { row - WIDTH } else { row };
        let plane = if value_group {
            value_weight
        } else {
            key_weight
        };
        // SAFETY: `plane_row` is inside the selected plane's row count.
        let plane = unsafe { plane.add(plane_row * EMBED_WORDS) };
        let mut sums = [0.0f32; TOKEN_TILE];
        let mut phase = 0usize;

        while phase < PROJECT_PHASES {
            let offset = phase * PROJECT_PHASE_WORDS + lane * WORDS_PER_LANE;
            // SAFETY: `offset` stays inside one complete embedding row.
            let weight = unsafe { load_u32x4_read_only(plane.add(offset)) };

            macro_rules! word {
                ($words:ident, $index:literal) => {
                    match $index {
                        0 => $words.0,
                        1 => $words.1,
                        2 => $words.2,
                        _ => $words.3,
                    }
                };
            }

            macro_rules! accumulate_word {
                ($token:literal, $activation:ident, $index:literal) => {{
                    let (weight_low, weight_high) =
                        convert::cvt_f32x2_bf16x2(word!(weight, $index));
                    let (low, high) = convert::cvt_f32x2_bf16x2(word!($activation, $index));
                    sums[$token] = float::fma_rn_f32(weight_low, low, sums[$token]);
                    sums[$token] = float::fma_rn_f32(weight_high, high, sums[$token]);
                }};
            }

            macro_rules! accumulate {
                ($token:literal) => {
                    if token_base + $token < TOKENS {
                        // SAFETY: the token is inside the launched row count.
                        let activation = unsafe {
                            load_u32x4_read_only(
                                embedding.add((token_base + $token) * EMBED_WORDS + offset),
                            )
                        };
                        accumulate_word!($token, activation, 0);
                        accumulate_word!($token, activation, 1);
                        accumulate_word!($token, activation, 2);
                        accumulate_word!($token, activation, 3);
                    }
                };
            }

            accumulate!(0);
            accumulate!(1);
            accumulate!(2);
            accumulate!(3);
            accumulate!(4);
            accumulate!(5);
            accumulate!(6);
            accumulate!(7);
            phase += 1;
        }

        macro_rules! store {
            ($token:literal) => {
                if token_base + $token < TOKENS {
                    let projected = reduce_sum_lane_zero(sums[$token]);
                    if lane == 0 {
                        let token = token_base + $token;
                        let bits = tcgen05::f32_to_bf16_rne(projected);
                        if value_group {
                            // SAFETY: one lane owns this token's value column.
                            unsafe { *value.add(token * BRANCH + plane_row) = bits };
                        } else {
                            // SAFETY: one lane owns this token's key column.
                            unsafe { *key.add(token * WIDTH + plane_row) = bits };
                        }
                    }
                }
            };
        }

        store!(0);
        store!(1);
        store!(2);
        store!(3);
        store!(4);
        store!(5);
        store!(6);
        store!(7);
    }

    /// The checkpoint's signed square root.
    ///
    /// `clamp_min` guards the square root only. `sign` is exactly zero when the
    /// dot product is exactly zero, and the clamp does not restore it, so an
    /// exactly-orthogonal branch gates on `sigmoid(0) = 0.5` rather than on
    /// `sigmoid(1e-3)`. Every step is a separate BF16 tensor op in the
    /// reference and rounds once.
    #[inline(always)]
    fn signed_root(scaled: f32) -> f32 {
        let magnitude = if scaled.abs() > round_bf16(GATE_FLOOR) {
            scaled.abs()
        } else {
            round_bf16(GATE_FLOOR)
        };
        let root = round_bf16(float::sqrt_rn_f32(magnitude));

        if scaled > 0.0 {
            root
        } else if scaled < 0.0 {
            -root
        } else {
            0.0
        }
    }

    /// Reduces four branch gates and broadcasts into the widened plane.
    ///
    /// The reference materializes `key_normed * query_normed` as a BF16 tensor
    /// before `sum(-1)`, so each product is rounded before the FP32
    /// accumulation rather than accumulated with an FMA.
    #[inline(always)]
    unsafe fn ple_gate_body<const TOKENS: usize>(
        key_normed: *const u32,
        query_normed: *const u32,
        value: *const u32,
        gated: *mut u32,
        warp_sums: *mut f32,
        activations: *mut f32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        // SAFETY: the launch contract gives every active block one complete row.
        let key_normed = unsafe { key_normed.add(token * ROW_WORDS) };
        // SAFETY: the query plane has the widened stream's row coverage.
        let query_normed = unsafe { query_normed.add(token * ROW_WORDS) };
        // SAFETY: the value plane holds one branch-wide row per token.
        let value = unsafe { value.add(token * BRANCH_WORDS) };
        // SAFETY: the output plane has the widened stream's row coverage.
        let gated = unsafe { gated.add(token * ROW_WORDS) };

        macro_rules! reduce_branch {
            ($branch:literal) => {{
                let mut sum = 0.0f32;
                let mut word = tid;

                while word < BRANCH_WORDS {
                    let index = $branch * BRANCH_WORDS + word;
                    // SAFETY: `word < BRANCH_WORDS` stays inside this branch.
                    let (key_low, key_high) =
                        convert::cvt_f32x2_bf16x2(unsafe { *key_normed.add(index) });
                    // SAFETY: `word < BRANCH_WORDS` stays inside this branch.
                    let (query_low, query_high) =
                        convert::cvt_f32x2_bf16x2(unsafe { *query_normed.add(index) });
                    sum += round_bf16(key_low * query_low);
                    sum += round_bf16(key_high * query_high);
                    word += ROW_THREADS as usize;
                }

                let sum = block_sum(sum, warp_sums, lane, warp_index);
                if tid == 0 {
                    let scaled = round_bf16(float::div_rn_f32(round_bf16(sum), GATE_DIVISOR));
                    // SAFETY: thread zero owns this branch's published gate.
                    unsafe {
                        *activations.add($branch) = round_bf16(sigmoid(signed_root(scaled)));
                    }
                }

                // Every thread has read the published block sum, so the next
                // branch may reuse the same reduction slots.
                thread::sync_threads();
            }};
        }

        reduce_branch!(0);
        reduce_branch!(1);
        reduce_branch!(2);
        reduce_branch!(3);

        macro_rules! write_branch {
            ($branch:literal) => {{
                // SAFETY: the barrier above published every branch activation.
                let activation = unsafe { *activations.add($branch) };
                let mut word = tid;

                while word < BRANCH_WORDS {
                    // SAFETY: `word < BRANCH_WORDS` stays inside the value row.
                    let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *value.add(word) });
                    // SAFETY: each thread writes disjoint packed BF16 pairs.
                    unsafe {
                        *gated.add($branch * BRANCH_WORDS + word) =
                            tcgen05::cvt_f32x2_bf16x2(activation * low, activation * high);
                    }
                    word += ROW_THREADS as usize;
                }
            }};
        }

        write_branch!(0);
        write_branch!(1);
        write_branch!(2);
        write_branch!(3);
    }

    /// `gated + silu(conv1d(gated_normed))` for one tap window.
    ///
    /// `F.conv1d` returns a BF16 tensor before `silu` reads it, and the residual
    /// add is a second BF16 tensor op, so both round.
    #[inline(always)]
    fn conv_epilogue(sum: f32, residual: f32) -> u16 {
        let activated = round_bf16(silu(round_bf16(sum)));

        tcgen05::f32_to_bf16_rne(residual + activated)
    }

    /// Contracts the four dilated taps in ascending tap order.
    #[inline(always)]
    unsafe fn conv_taps(weights: *const u16, taps: [u16; CONV_TAPS]) -> f32 {
        // SAFETY: the weight plane holds `CONV_TAPS` values per channel.
        let (first, second, third, fourth) = unsafe {
            (
                bf16_to_f32(*weights),
                bf16_to_f32(*weights.add(1)),
                bf16_to_f32(*weights.add(2)),
                bf16_to_f32(*weights.add(3)),
            )
        };

        float::fma_rn_f32(
            first,
            bf16_to_f32(taps[0]),
            float::fma_rn_f32(
                second,
                bf16_to_f32(taps[1]),
                float::fma_rn_f32(third, bf16_to_f32(taps[2]), fourth * bf16_to_f32(taps[3])),
            ),
        )
    }

    /// Advances one decode step's convolution history and applies the taps.
    ///
    /// The nine-column state is the last nine columns of the reference's
    /// `cat(state, x)` window, so the taps read columns `0`, `3`, `6`, and the
    /// current value, and the new state is the window shifted left by one.
    #[inline(always)]
    unsafe fn ple_convolution_body<const TOKENS: usize>(
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) {
        let index = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if index >= TOKENS * WIDTH {
            return;
        }

        let token = index / WIDTH;
        let channel = index - token * WIDTH;
        // SAFETY: the slot plane holds one row index per launched token.
        let state_row = unsafe { *state_rows.add(token) as usize };
        // SAFETY: the state plane holds `CONV_STATE` columns per channel per slot.
        let state = unsafe { state.add((state_row * WIDTH + channel) * CONV_STATE) };
        // SAFETY: the launched token is inside the normalized plane.
        let current = unsafe { *gated_normed.add(index) };
        let mut window = [0u16; CONV_STATE];
        let mut column = 0usize;

        while column < CONV_STATE {
            // SAFETY: `column < CONV_STATE` stays inside this channel's state.
            window[column] = unsafe { *state.add(column) };
            column += 1;
        }

        column = 0;
        while column + 1 < CONV_STATE {
            // SAFETY: the whole window is live in registers, so the shift is
            // safe to publish column by column.
            unsafe { *state.add(column) = window[column + 1] };
            column += 1;
        }
        // SAFETY: the newest column always lands in the last state slot.
        unsafe { *state.add(CONV_STATE - 1) = current };

        // SAFETY: the weight plane holds `CONV_TAPS` values per channel.
        let weights = unsafe { weights.add(channel * CONV_TAPS) };
        // SAFETY: `conv_taps` reads exactly those four weights.
        let sum = unsafe {
            conv_taps(
                weights,
                [
                    window[0],
                    window[CONV_DILATION],
                    window[2 * CONV_DILATION],
                    current,
                ],
            )
        };
        // SAFETY: the residual plane has the output plane's coverage.
        let residual = bf16_to_f32(unsafe { *gated.add(index) });

        // SAFETY: one thread owns this token's channel.
        unsafe { *output.add(index) = conv_epilogue(sum, residual) };
    }

    /// Applies the dilated convolution across one exact prefill tile.
    ///
    /// Taps inside the tile read the normalized plane directly; taps before it
    /// read the carried nine-column state, which the history publication
    /// rewrites only after this grid has finished.
    #[inline(always)]
    unsafe fn ple_convolution_prefill_body<const TOKENS: usize>(
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *const u16,
        output: *mut u16,
    ) {
        let index = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if index >= TOKENS * WIDTH {
            return;
        }

        let token = index / WIDTH;
        let channel = index - token * WIDTH;
        // SAFETY: one prefill tile is one sequence, so it names one slot.
        let state_row = unsafe { *state_rows as usize };
        // SAFETY: the state plane holds `CONV_STATE` columns per channel per slot.
        let state = unsafe { state.add((state_row * WIDTH + channel) * CONV_STATE) };

        macro_rules! tap {
            ($distance:expr) => {
                if token >= $distance {
                    // SAFETY: the tap is inside this tile's normalized plane.
                    unsafe { *gated_normed.add((token - $distance) * WIDTH + channel) }
                } else {
                    // SAFETY: `CONV_STATE - $distance + token < CONV_STATE`
                    // exactly when the tap falls before the tile.
                    unsafe { *state.add(CONV_STATE - $distance + token) }
                }
            };
        }

        let taps = [
            tap!(3 * CONV_DILATION),
            tap!(2 * CONV_DILATION),
            tap!(CONV_DILATION),
            // SAFETY: the current column is always inside the tile.
            unsafe { *gated_normed.add(index) },
        ];
        // SAFETY: the weight plane holds `CONV_TAPS` values per channel.
        let weights = unsafe { weights.add(channel * CONV_TAPS) };
        // SAFETY: `conv_taps` reads exactly those four weights.
        let sum = unsafe { conv_taps(weights, taps) };
        // SAFETY: the residual plane has the output plane's coverage.
        let residual = bf16_to_f32(unsafe { *gated.add(index) });

        // SAFETY: one thread owns this token's channel.
        unsafe { *output.add(index) = conv_epilogue(sum, residual) };
    }

    /// Publishes one prefill tile's trailing nine columns as the new history.
    ///
    /// Every admitted tile is at least `CONV_STATE` wide, so the new window is
    /// always a contiguous slice of the tile and never splices carried columns.
    #[inline(always)]
    unsafe fn ple_convolution_prefill_history_body<const TOKENS: usize>(
        gated_normed: *const u16,
        state_rows: *const u32,
        state: *mut u16,
    ) {
        let channel =
            (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if channel >= WIDTH {
            return;
        }

        // SAFETY: one prefill tile is one sequence, so it names one slot.
        let state_row = unsafe { *state_rows as usize };
        // SAFETY: the state plane holds `CONV_STATE` columns per channel per slot.
        let state = unsafe { state.add((state_row * WIDTH + channel) * CONV_STATE) };
        let mut column = 0usize;

        while column < CONV_STATE {
            // SAFETY: `TOKENS >= CONV_STATE` on every admitted tile, so the
            // source token index is inside the tile.
            unsafe {
                *state.add(column) =
                    *gated_normed.add((TOKENS - CONV_STATE + column) * WIDTH + channel);
            }
            column += 1;
        }
    }

    /// Adds the engram delta into the widened residual stream.
    ///
    /// The injection happens before `attn_hyper_connection` and applies no
    /// normalization to the stream itself. There is no reduction, so the entry
    /// is bitwise reproducible and may publish in place.
    #[inline(always)]
    unsafe fn ple_inject_body<const TOKENS: usize>(
        hidden: *const u32,
        delta: *const u32,
        output: *mut u32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: the launch contract gives every active block one complete row.
        let hidden = unsafe { hidden.add(token * ROW_WORDS) };
        // SAFETY: the delta plane has the widened stream's row coverage.
        let delta = unsafe { delta.add(token * ROW_WORDS) };
        // SAFETY: the output plane has the widened stream's row coverage.
        let output = unsafe { output.add(token * ROW_WORDS) };
        let mut word = tid;

        while word < ROW_WORDS {
            // SAFETY: `word < ROW_WORDS` stays inside this token's row.
            let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *hidden.add(word) });
            // SAFETY: `word < ROW_WORDS` stays inside this token's row.
            let (delta_low, delta_high) = convert::cvt_f32x2_bf16x2(unsafe { *delta.add(word) });
            // SAFETY: each thread writes the packed pair it just read.
            unsafe {
                *output.add(word) = tcgen05::cvt_f32x2_bf16x2(low + delta_low, high + delta_high);
            }
            word += ROW_THREADS as usize;
        }
    }

    /// Widens one exact decode batch of staged engram code rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_dequant<const TOKENS: usize>(
        codes: *const u16,
        scale: f32,
        embedding: *mut u32,
    ) {
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { ple_dequant_body::<TOKENS>(codes, scale, embedding) };
    }

    /// Widens one exact prefill tile of staged engram code rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_dequant_prefill<const TOKENS: usize>(
        codes: *const u16,
        scale: f32,
        embedding: *mut u32,
    ) {
        // Prefill retains separate symbols so its resource authority cannot
        // drift with decode; the traversal is identical.
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { ple_dequant_body::<TOKENS>(codes, scale, embedding) };
    }

    /// Projects one exact decode batch onto both engram projections.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_project<const TOKENS: usize>(
        embedding: *const u32,
        key_weight: *const u32,
        value_weight: *const u32,
        key: *mut u16,
        value: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        unsafe {
            ple_project_body::<TOKENS>(embedding, key_weight, value_weight, key, value);
        }
    }

    /// Projects one exact prefill tile onto both engram projections.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_project_prefill<const TOKENS: usize>(
        embedding: *const u32,
        key_weight: *const u32,
        value_weight: *const u32,
        key: *mut u16,
        value: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        unsafe {
            ple_project_body::<TOKENS>(embedding, key_weight, value_weight, key, value);
        }
    }

    /// Gates one exact decode batch and flattens the broadcast product.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_gate<const TOKENS: usize>(
        key_normed: *const u32,
        query_normed: *const u32,
        value: *const u32,
        gated: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, ROW_WARPS, 16> = SharedArray::UNINIT;
        static mut ACTIVATIONS: SharedArray<f32, BRANCHES, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();
        let activations = core::ptr::addr_of_mut!(ACTIVATIONS).cast::<f32>();

        // SAFETY: the launch contract pins one complete row per active block.
        unsafe {
            ple_gate_body::<TOKENS>(
                key_normed,
                query_normed,
                value,
                gated,
                warp_sums,
                activations,
            );
        }
    }

    /// Gates one exact prefill tile and flattens the broadcast product.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_gate_prefill<const TOKENS: usize>(
        key_normed: *const u32,
        query_normed: *const u32,
        value: *const u32,
        gated: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, ROW_WARPS, 16> = SharedArray::UNINIT;
        static mut ACTIVATIONS: SharedArray<f32, BRANCHES, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();
        let activations = core::ptr::addr_of_mut!(ACTIVATIONS).cast::<f32>();

        // SAFETY: the launch contract pins one complete row per active block.
        unsafe {
            ple_gate_body::<TOKENS>(
                key_normed,
                query_normed,
                value,
                gated,
                warp_sums,
                activations,
            );
        }
    }

    /// Advances one exact decode batch's convolution history and applies it.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_convolution<const TOKENS: usize>(
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) {
        // SAFETY: the launch contract pins one channel per thread.
        unsafe {
            ple_convolution_body::<TOKENS>(gated, gated_normed, weights, state_rows, state, output);
        }
    }

    /// Applies the dilated convolution across one exact prefill tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_convolution_prefill<const TOKENS: usize>(
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *const u16,
        output: *mut u16,
    ) {
        // SAFETY: the launch contract pins one channel per thread.
        unsafe {
            ple_convolution_prefill_body::<TOKENS>(
                gated,
                gated_normed,
                weights,
                state_rows,
                state,
                output,
            );
        }
    }

    /// Publishes one exact prefill tile's trailing convolution history.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_convolution_prefill_history<const TOKENS: usize>(
        gated_normed: *const u16,
        state_rows: *const u32,
        state: *mut u16,
    ) {
        // The convolution grid must have retired before this runs: it rewrites
        // the columns that grid reads.
        // SAFETY: the launch contract pins one channel per thread.
        unsafe {
            ple_convolution_prefill_history_body::<TOKENS>(gated_normed, state_rows, state);
        }
    }

    /// Injects one exact decode batch's engram delta into the raw stream.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_inject<const TOKENS: usize>(
        hidden: *const u32,
        delta: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { ple_inject_body::<TOKENS>(hidden, delta, output) };
    }

    /// Injects one exact prefill tile's engram delta into the raw stream.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_inject_prefill<const TOKENS: usize>(
        hidden: *const u32,
        delta: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { ple_inject_body::<TOKENS>(hidden, delta, output) };
    }
}

/// CTAs the fused projection needs for one exact row count.
const fn project_blocks(tokens: usize) -> u32 {
    (PROJECT_ROW_GROUPS * tokens.div_ceil(TOKEN_TILE)) as u32
}

/// CTAs the convolution needs for one exact row count.
const fn convolution_blocks(tokens: usize) -> u32 {
    (tokens * WIDTH / CONV_THREADS as usize) as u32
}

/// Checkpoint planes one engram module reads.
///
/// `table_scale_bits` is the checkpoint's exact BF16 source word for the single
/// engram table multiplier; the launcher widens it the way materialization
/// does and never converts it.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextEngramSources {
    /// `key_proj` weight, `[HC_WIDTH, PLE_EMBED_DIM]` BF16.
    pub key_proj: *const u16,
    /// `value_proj` weight, `[PLE_EMBED_DIM, PLE_EMBED_DIM]` BF16.
    pub value_proj: *const u16,
    /// `norm_key` gamma, `[HC_WIDTH]` BF16.
    pub norm_key: *const u16,
    /// `norm_query` gamma, `[HC_WIDTH]` BF16.
    pub norm_query: *const u16,
    /// `norm_conv` gamma, `[HC_WIDTH]` BF16.
    pub norm_conv: *const u16,
    /// Dilated depthwise convolution weight, `[HC_WIDTH, PLE_CONV_KERNEL]` BF16.
    pub convolution: *const u16,
    /// Exact BF16 source word of the engram table multiplier.
    pub table_scale_bits: u16,
}

/// Caller-owned planes the staged engram pipeline publishes.
///
/// Every one is observable so each fused boundary can be qualified. They are an
/// implementation detail of this route, not part of the operator contract.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextEngramWorkspace {
    /// Dequantized embedding, `rows * PLE_EMBED_DIM` BF16.
    pub embedding: *mut u16,
    /// Projected key, `rows * HC_WIDTH` BF16.
    pub key: *mut u16,
    /// Normalized key, `rows * HC_WIDTH` BF16.
    pub key_normed: *mut u16,
    /// Normalized residual copy used only by the gate, `rows * HC_WIDTH` BF16.
    pub query_normed: *mut u16,
    /// Projected value, `rows * HIDDEN` BF16.
    pub value: *mut u16,
    /// Gated and flattened value, `rows * HC_WIDTH` BF16.
    pub gated: *mut u16,
    /// Normalized copy of `gated`, `rows * HC_WIDTH` BF16.
    pub gated_normed: *mut u16,
    /// The engram delta before injection, `rows * HC_WIDTH` BF16.
    pub delta: *mut u16,
}

/// Prepared decode entries for one exact batch.
///
/// Decode carries per-token convolution state: every row is its own sequence,
/// so the entry reads one slot index per token.
struct PreparedDecodeRoute<const TOKENS: usize> {
    dequant: PreparedLaunch<kernels::__qwen38_flash_next_ple_dequant_CudaKernel<TOKENS>>,
    project: PreparedLaunch<kernels::__qwen38_flash_next_ple_project_CudaKernel<TOKENS>>,
    gate: PreparedLaunch<kernels::__qwen38_flash_next_ple_gate_CudaKernel<TOKENS>>,
    convolution: PreparedLaunch<kernels::__qwen38_flash_next_ple_convolution_CudaKernel<TOKENS>>,
    inject: PreparedLaunch<kernels::__qwen38_flash_next_ple_inject_CudaKernel<TOKENS>>,
}

/// Prepared prefill entries for one exact tile width.
///
/// Prefill retains separate symbols so its resource authority cannot drift with
/// decode, and it publishes its trailing history in a second entry so the
/// convolution grid never reads a column another CTA has already rewritten.
struct PreparedPrefillRoute<const TOKENS: usize> {
    dequant: PreparedLaunch<kernels::__qwen38_flash_next_ple_dequant_prefill_CudaKernel<TOKENS>>,
    project: PreparedLaunch<kernels::__qwen38_flash_next_ple_project_prefill_CudaKernel<TOKENS>>,
    gate: PreparedLaunch<kernels::__qwen38_flash_next_ple_gate_prefill_CudaKernel<TOKENS>>,
    convolution:
        PreparedLaunch<kernels::__qwen38_flash_next_ple_convolution_prefill_CudaKernel<TOKENS>>,
    history: PreparedLaunch<
        kernels::__qwen38_flash_next_ple_convolution_prefill_history_CudaKernel<TOKENS>,
    >,
    inject: PreparedLaunch<kernels::__qwen38_flash_next_ple_inject_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedDecodeRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let rows = u32::try_from(TOKENS)
            .map_err(|_| GpuError::invalid_launch("engram batch exceeds CUDA grid width"))?;
        let dequant = module
            .prepare_qwen38_flash_next_ple_dequant::<TOKENS>(LaunchConfig1D::new(
                rows,
                ROW_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram dequantization", source))?;
        let project = module
            .prepare_qwen38_flash_next_ple_project::<TOKENS>(LaunchConfig1D::new(
                project_blocks(TOKENS),
                PROJECT_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram projection", source))?;
        let gate = module
            .prepare_qwen38_flash_next_ple_gate::<TOKENS>(LaunchConfig1D::new(rows, ROW_THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the engram gate", source))?;
        let convolution = module
            .prepare_qwen38_flash_next_ple_convolution::<TOKENS>(LaunchConfig1D::new(
                convolution_blocks(TOKENS),
                CONV_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram convolution", source))?;
        let inject = module
            .prepare_qwen38_flash_next_ple_inject::<TOKENS>(LaunchConfig1D::new(
                rows,
                ROW_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram injection", source))?;

        Ok(Self {
            dequant,
            project,
            gate,
            convolution,
            inject,
        })
    }

    unsafe fn launch_dequant(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        codes: *const u8,
        scale: f32,
        embedding: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_dequant::<TOKENS>(
                stream,
                &self.dequant,
                codes.cast::<u16>(),
                scale,
                embedding.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the engram dequantization", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_project(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        embedding: *const u16,
        key_weight: *const u16,
        value_weight: *const u16,
        key: *mut u16,
        value: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_project::<TOKENS>(
                stream,
                &self.project,
                embedding.cast::<u32>(),
                key_weight.cast::<u32>(),
                value_weight.cast::<u32>(),
                key,
                value,
            )
            .map_err(|source| GpuError::launch("launching the engram projection", source))
    }

    unsafe fn launch_gate(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        key_normed: *const u16,
        query_normed: *const u16,
        value: *const u16,
        gated: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_gate::<TOKENS>(
                stream,
                &self.gate,
                key_normed.cast::<u32>(),
                query_normed.cast::<u32>(),
                value.cast::<u32>(),
                gated.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the engram gate", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_convolution(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_convolution::<TOKENS>(
                stream,
                &self.convolution,
                gated,
                gated_normed,
                weights,
                state_rows,
                state,
                output,
            )
            .map_err(|source| GpuError::launch("launching the engram convolution", source))
    }

    unsafe fn launch_inject(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        hidden: *const u16,
        delta: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_inject::<TOKENS>(
                stream,
                &self.inject,
                hidden.cast::<u32>(),
                delta.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the engram injection", source))
    }
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let rows = u32::try_from(TOKENS)
            .map_err(|_| GpuError::invalid_launch("engram prefill exceeds CUDA grid width"))?;
        let dequant = module
            .prepare_qwen38_flash_next_ple_dequant_prefill::<TOKENS>(LaunchConfig1D::new(
                rows,
                ROW_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the engram prefill dequantization", source)
            })?;
        let project = module
            .prepare_qwen38_flash_next_ple_project_prefill::<TOKENS>(LaunchConfig1D::new(
                project_blocks(TOKENS),
                PROJECT_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the engram prefill projection", source)
            })?;
        let gate = module
            .prepare_qwen38_flash_next_ple_gate_prefill::<TOKENS>(LaunchConfig1D::new(
                rows,
                ROW_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram prefill gate", source))?;
        let convolution = module
            .prepare_qwen38_flash_next_ple_convolution_prefill::<TOKENS>(LaunchConfig1D::new(
                convolution_blocks(TOKENS),
                CONV_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the engram prefill convolution", source)
            })?;
        let history = module
            .prepare_qwen38_flash_next_ple_convolution_prefill_history::<TOKENS>(
                LaunchConfig1D::new(CONV_HISTORY_BLOCKS, CONV_THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch("preparing the engram history publication", source)
            })?;
        let inject = module
            .prepare_qwen38_flash_next_ple_inject_prefill::<TOKENS>(LaunchConfig1D::new(
                rows,
                ROW_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram prefill injection", source))?;

        Ok(Self {
            dequant,
            project,
            gate,
            convolution,
            history,
            inject,
        })
    }

    unsafe fn launch_dequant(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        codes: *const u8,
        scale: f32,
        embedding: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_dequant_prefill::<TOKENS>(
                stream,
                &self.dequant,
                codes.cast::<u16>(),
                scale,
                embedding.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the engram prefill dequantization", source)
            })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_project(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        embedding: *const u16,
        key_weight: *const u16,
        value_weight: *const u16,
        key: *mut u16,
        value: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_project_prefill::<TOKENS>(
                stream,
                &self.project,
                embedding.cast::<u32>(),
                key_weight.cast::<u32>(),
                value_weight.cast::<u32>(),
                key,
                value,
            )
            .map_err(|source| GpuError::launch("launching the engram prefill projection", source))
    }

    unsafe fn launch_gate(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        key_normed: *const u16,
        query_normed: *const u16,
        value: *const u16,
        gated: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_gate_prefill::<TOKENS>(
                stream,
                &self.gate,
                key_normed.cast::<u32>(),
                query_normed.cast::<u32>(),
                value.cast::<u32>(),
                gated.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the engram prefill gate", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_convolution(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_convolution_prefill::<TOKENS>(
                stream,
                &self.convolution,
                gated,
                gated_normed,
                weights,
                state_rows,
                state.cast_const(),
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching the engram prefill convolution", source)
            })?;
        module
            .qwen38_flash_next_ple_convolution_prefill_history::<TOKENS>(
                stream,
                &self.history,
                gated_normed,
                state_rows,
                state,
            )
            .map_err(|source| GpuError::launch("launching the engram history publication", source))
    }

    unsafe fn launch_inject(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        hidden: *const u16,
        delta: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_ple_inject_prefill::<TOKENS>(
                stream,
                &self.inject,
                hidden.cast::<u32>(),
                delta.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the engram prefill injection", source))
    }
}

/// PTX symbols retained for every admitted engram route.
pub(crate) fn engram_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<1>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<2>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<3>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<4>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<5>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<6>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<7>(),
        kernels::qwen38_flash_next_ple_dequant_ptx_name::<8>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<1>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<2>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<3>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<4>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<5>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<6>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<7>(),
        kernels::qwen38_flash_next_ple_project_ptx_name::<8>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<1>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<2>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<3>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<4>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<5>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<6>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<7>(),
        kernels::qwen38_flash_next_ple_gate_ptx_name::<8>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<1>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<2>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<3>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<4>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<5>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<6>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<7>(),
        kernels::qwen38_flash_next_ple_convolution_ptx_name::<8>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<1>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<2>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<3>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<4>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<5>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<6>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<7>(),
        kernels::qwen38_flash_next_ple_inject_ptx_name::<8>(),
        kernels::qwen38_flash_next_ple_dequant_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_dequant_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_dequant_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_dequant_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_ple_project_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_project_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_project_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_project_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_ple_gate_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_gate_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_gate_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_gate_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_history_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_history_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_history_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_convolution_prefill_history_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_ple_inject_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_ple_inject_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_ple_inject_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_ple_inject_prefill_ptx_name::<1_024>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_ple),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextPleRoutes {
    #[route(1)]
    b1: PreparedDecodeRoute<1>,
    #[route(2)]
    b2: PreparedDecodeRoute<2>,
    #[route(3)]
    b3: PreparedDecodeRoute<3>,
    #[route(4)]
    b4: PreparedDecodeRoute<4>,
    #[route(5)]
    b5: PreparedDecodeRoute<5>,
    #[route(6)]
    b6: PreparedDecodeRoute<6>,
    #[route(7)]
    b7: PreparedDecodeRoute<7>,
    #[route(8)]
    b8: PreparedDecodeRoute<8>,
    #[route(32)]
    t32: PreparedPrefillRoute<32>,
    #[route(64)]
    t64: PreparedPrefillRoute<64>,
    #[route(128)]
    t128: PreparedPrefillRoute<128>,
    #[route(1024)]
    t1024: PreparedPrefillRoute<1_024>,
}

fn unsupported_rows(operation: &str, rows: usize) -> GpuError {
    let prefill = PREFILL_ROWS.map(|rows| rows.to_string()).join(",");

    GpuError::invalid_launch(format!(
        "Qwen3.8 Flash-Next {operation} row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T={prefill}",
    ))
}

/// Widens the checkpoint's exact BF16 table-multiplier word.
///
/// This is admission, not conversion: `*.ple.*` is outside the ModelOpt
/// quantization list, so the word is a plain positive multiplier and must
/// survive the widening unchanged.
fn table_scale(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Prepared PLE routes for decode `B=1..=8` and prefill `T=32,64,128,1024`.
pub struct Qwen38FlashNextEngramOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextPleRoutes,
}

impl Qwen38FlashNextEngramOp {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = engram_ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the engram module", source))?;

        Ok(Self {
            routes: Qwen38FlashNextPleRoutes::prepare(&module)?,
            module,
        })
    }

    /// Widens one staged engram plane's E4M3 codes into BF16.
    ///
    /// # Safety
    ///
    /// `codes` must cover `rows * PLE_EMBED_DIM` bytes and be two-byte aligned;
    /// `embedding` must cover `rows * PLE_EMBED_DIM` BF16 values and be
    /// four-byte aligned. Both must belong to `stream`'s context, remain live
    /// through stream completion, and must not overlap.
    pub unsafe fn launch_dequant(
        &self,
        stream: &CudaStream,
        rows: usize,
        codes: *const u8,
        table_scale_bits: u16,
        embedding: *mut u16,
    ) -> GpuResult<()> {
        let scale = table_scale(table_scale_bits);

        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe { $route.launch_dequant(&self.module, stream, codes, scale, embedding) }
            };
        }

        dispatch_qwen38_flash_next_ple!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("engram dequantization", rows)))
    }

    /// Projects the dequantized embedding onto `key_proj` and `value_proj`.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned except `key` and `value`, which
    /// must be two-byte aligned. `embedding` must cover
    /// `rows * PLE_EMBED_DIM` BF16 values, `key_weight`
    /// `HC_WIDTH * PLE_EMBED_DIM`, `value_weight` `PLE_EMBED_DIM^2`, `key`
    /// `rows * HC_WIDTH`, and `value` `rows * HIDDEN`. Allocations must belong
    /// to `stream`'s context, remain live through stream completion, and must
    /// not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_project(
        &self,
        stream: &CudaStream,
        rows: usize,
        embedding: *const u16,
        key_weight: *const u16,
        value_weight: *const u16,
        key: *mut u16,
        value: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_project(
                        &self.module,
                        stream,
                        embedding,
                        key_weight,
                        value_weight,
                        key,
                        value,
                    )
                }
            };
        }

        dispatch_qwen38_flash_next_ple!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("engram projection", rows)))
    }

    /// Gates the value by the per-branch signed-root dot product.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned. `key_normed`, `query_normed`,
    /// and `gated` must cover `rows * HC_WIDTH` BF16 values and `value`
    /// `rows * HIDDEN`. Allocations must belong to `stream`'s context, remain
    /// live through stream completion, and must not overlap.
    pub unsafe fn launch_gate(
        &self,
        stream: &CudaStream,
        rows: usize,
        key_normed: *const u16,
        query_normed: *const u16,
        value: *const u16,
        gated: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_gate(&self.module, stream, key_normed, query_normed, value, gated)
                }
            };
        }

        dispatch_qwen38_flash_next_ple!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("engram gate", rows)))
    }

    /// Applies the dilated short convolution and its residual add.
    ///
    /// Decode advances one column of every named slot's nine-column history;
    /// prefill reads the carried history and republishes the tile's trailing
    /// nine columns in a second entry, after the convolution grid has retired.
    ///
    /// # Safety
    ///
    /// Every pointer must be two-byte aligned except `state_rows`, which must
    /// be four-byte aligned. `gated`, `gated_normed`, and `output` must cover
    /// `rows * HC_WIDTH` BF16 values, `weights` `HC_WIDTH * PLE_CONV_KERNEL`,
    /// and `state` `slots * HC_WIDTH * PLE_CONV_STATE_LEN`. `state_rows` must
    /// name one valid slot per row at decode and at least one at prefill, where
    /// only the first is read. Allocations must belong to `stream`'s context,
    /// remain live through stream completion, and must not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_convolution(
        &self,
        stream: &CudaStream,
        rows: usize,
        gated: *const u16,
        gated_normed: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_convolution(
                        &self.module,
                        stream,
                        gated,
                        gated_normed,
                        weights,
                        state_rows,
                        state,
                        output,
                    )
                }
            };
        }

        dispatch_qwen38_flash_next_ple!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("engram convolution", rows)))
    }

    /// Adds the engram delta into the widened residual stream.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned and cover `rows * HC_WIDTH`
    /// BF16 values. Allocations must belong to `stream`'s context and remain
    /// live through stream completion. The planes must not overlap except that
    /// `output` may alias `hidden` exactly, which is the production form.
    pub unsafe fn launch_inject(
        &self,
        stream: &CudaStream,
        rows: usize,
        hidden: *const u16,
        delta: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe { $route.launch_inject(&self.module, stream, hidden, delta, output) }
            };
        }

        dispatch_qwen38_flash_next_ple!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("engram injection", rows)))
    }

    /// Runs the complete PLE module and injects its delta into `hidden`.
    ///
    /// `norm` supplies the identical grouped norm used at all three boundaries.
    ///
    /// # Safety
    ///
    /// Every plane carries the contract of the per-stage method that reads or
    /// writes it, and `output` may alias `hidden` exactly.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_engram(
        &self,
        norm: &Qwen38FlashNextHyperConnectionOp,
        stream: &CudaStream,
        rows: usize,
        codes: *const u8,
        hidden: *const u16,
        sources: Qwen38FlashNextEngramSources,
        workspace: Qwen38FlashNextEngramWorkspace,
        state_rows: *const u32,
        state: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        // SAFETY: every plane carries its per-stage method's contract unchanged.
        unsafe {
            self.launch_dequant(
                stream,
                rows,
                codes,
                sources.table_scale_bits,
                workspace.embedding,
            )?;
            self.launch_project(
                stream,
                rows,
                workspace.embedding,
                sources.key_proj,
                sources.value_proj,
                workspace.key,
                workspace.value,
            )?;
            norm.launch_grouped_norm(
                stream,
                rows,
                workspace.key,
                sources.norm_key,
                workspace.key_normed,
            )?;
            norm.launch_grouped_norm(
                stream,
                rows,
                hidden,
                sources.norm_query,
                workspace.query_normed,
            )?;
            self.launch_gate(
                stream,
                rows,
                workspace.key_normed,
                workspace.query_normed,
                workspace.value,
                workspace.gated,
            )?;
            norm.launch_grouped_norm(
                stream,
                rows,
                workspace.gated,
                sources.norm_conv,
                workspace.gated_normed,
            )?;
            self.launch_convolution(
                stream,
                rows,
                workspace.gated,
                workspace.gated_normed,
                sources.convolution,
                state_rows,
                state,
                workspace.delta,
            )?;
            self.launch_inject(stream, rows, hidden, workspace.delta, output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BRANCH, BRANCHES, CODE_WORDS, CONV_DILATION, CONV_HISTORY_BLOCKS, CONV_STATE, CONV_TAPS,
        CONV_THREADS, EMBED, EMBED_WORDS, GATE_DIVISOR, GATE_FLOOR, GATE_PAIRS_PER_THREAD,
        MAX_BATCH, PREFILL_ROWS, PROJECT_KEY_GROUPS, PROJECT_PHASES, PROJECT_ROW_GROUPS,
        PROJECT_ROWS, PROJECT_THREADS, Qwen38FlashNextPleRoutes, ROW_THREADS, ROW_WARPS, ROW_WORDS,
        WIDTH, convolution_blocks, engram_ptx_names, project_blocks, table_scale, unsupported_rows,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_model::{Arch, Qwen38FlashNext};

    /// The exact schedule, decode first and prefill after.
    const ADMITTED_SCHEDULE: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    #[test]
    fn geometry_flows_from_the_admitted_architecture() {
        assert_eq!(BRANCHES, Qwen38FlashNext::HC_COUNT);
        assert_eq!(BRANCH, Qwen38FlashNext::HIDDEN);
        assert_eq!(WIDTH, Qwen38FlashNext::HC_WIDTH);
        assert_eq!(EMBED, Qwen38FlashNext::PLE_EMBED_DIM);
        assert_eq!(CONV_TAPS, Qwen38FlashNext::PLE_CONV_KERNEL);
        assert_eq!(CONV_DILATION, Qwen38FlashNext::PLE_CONV_DILATION);
        assert_eq!(CONV_STATE, Qwen38FlashNext::PLE_CONV_STATE_LEN);
        assert_eq!(GATE_FLOOR, Qwen38FlashNext::PLE_GATE_FLOOR);
        assert_eq!(ROW_WORDS, 5_120);
        assert_eq!(EMBED_WORDS, 1_280);
        assert_eq!(CODE_WORDS, EMBED_WORDS);
    }

    /// The gate's `/ sqrt(2560)` is the literal the reference divides by, not a
    /// reciprocal folded into a weight.
    #[test]
    fn the_gate_divisor_is_the_represented_square_root() {
        assert_eq!(GATE_DIVISOR, (BRANCH as f32).sqrt());
        assert_eq!(GATE_DIVISOR, (BRANCH as f64).sqrt() as f32);
        assert_ne!(GATE_DIVISOR, 1.0 / (BRANCH as f32).sqrt());
    }

    /// The checkpoint's engram multiplier is admitted, never converted.
    #[test]
    fn the_table_scale_widens_its_exact_source_word() {
        assert_eq!(table_scale(0x3951), f32::from_bits(0x3951_0000));
        assert_eq!(table_scale(0x3951), 1.993_179_3e-4);
        assert!(table_scale(0x3951) > 0.0);
    }

    /// Every admitted width maps exactly onto its CTA, which is what keeps each
    /// entry's reduction order fixed for every row count.
    #[test]
    fn exact_geometry_is_cta_aligned() {
        assert_eq!(ROW_THREADS, 256);
        assert_eq!(ROW_WARPS, 8);
        assert_eq!(GATE_PAIRS_PER_THREAD, 5);
        assert_eq!(PROJECT_THREADS, 256);
        assert_eq!(CONV_THREADS, 256);
        assert_eq!(PROJECT_ROWS, 12_800);
        assert_eq!(PROJECT_ROW_GROUPS, 1_600);
        assert_eq!(PROJECT_KEY_GROUPS, 1_280);
        assert_eq!(PROJECT_PHASES, 10);
        assert_eq!(CONV_HISTORY_BLOCKS, 40);
        assert_eq!(PROJECT_ROWS, WIDTH + BRANCH);
    }

    /// The convolution's taps are `t-9, t-6, t-3, t`, which is what makes the
    /// GDN convolution unusable here even though both are depthwise width four.
    #[test]
    fn the_convolution_is_dilated_and_its_state_is_nine_columns() {
        assert_eq!(CONV_STATE, 9);
        assert_eq!(CONV_STATE, (CONV_TAPS - 1) * CONV_DILATION);
        assert_eq!(
            [3 * CONV_DILATION, 2 * CONV_DILATION, CONV_DILATION, 0_usize],
            [9, 6, 3, 0]
        );
        assert_ne!(CONV_STATE, Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM - 1);
        // The history publication only ever copies a contiguous slice of the
        // tile, which every admitted tile is wide enough to supply.
        for rows in PREFILL_ROWS {
            assert!(rows >= CONV_STATE);
        }
    }

    #[test]
    fn grid_widths_cover_every_row_and_token_tile() {
        for rows in 1..=MAX_BATCH {
            assert_eq!(project_blocks(rows), 1_600);
            assert_eq!(convolution_blocks(rows), (rows * 40) as u32);
        }
        for rows in PREFILL_ROWS {
            let tiles = (rows / MAX_BATCH) as u32;
            assert_eq!(project_blocks(rows), 1_600 * tiles);
            assert_eq!(convolution_blocks(rows), (rows * 40) as u32);
        }
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_stage_and_route() {
        let names = engram_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        // Five decode stages and six prefill stages: prefill publishes its
        // trailing convolution history in an entry decode does not need.
        assert_eq!(names.len(), 5 * MAX_BATCH + 6 * PREFILL_ROWS.len());
        assert_eq!(names.len(), 64);
        assert_eq!(unique.len(), names.len());
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this family
    /// is its per-base-name count.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in engram_ptx_names() {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("qwen38_flash_next_ple_convolution", 8),
                ("qwen38_flash_next_ple_convolution_prefill", 4),
                ("qwen38_flash_next_ple_convolution_prefill_history", 4),
                ("qwen38_flash_next_ple_dequant", 8),
                ("qwen38_flash_next_ple_dequant_prefill", 4),
                ("qwen38_flash_next_ple_gate", 8),
                ("qwen38_flash_next_ple_gate_prefill", 4),
                ("qwen38_flash_next_ple_inject", 8),
                ("qwen38_flash_next_ple_inject_prefill", 4),
                ("qwen38_flash_next_ple_project", 8),
                ("qwen38_flash_next_ple_project_prefill", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 64);
    }

    /// All three norms reuse the hyper-connection module's entry.
    #[test]
    fn the_family_emits_no_grouped_norm_entry() {
        assert!(
            engram_ptx_names()
                .into_iter()
                .all(|name| !name.contains("norm"))
        );
    }

    /// Every admitted row count, swept exhaustively.
    #[test]
    fn row_routing_is_exact() {
        let admitted = (0..=2_048)
            .chain([usize::MAX])
            .filter(|&rows| Qwen38FlashNextPleRoutes::contains(rows))
            .collect::<Vec<_>>();

        assert_eq!(admitted, ADMITTED_SCHEDULE.to_vec());
        assert_eq!(Qwen38FlashNextPleRoutes::admitted_rows(), ADMITTED_SCHEDULE);
    }

    #[test]
    fn unadmitted_row_counts_name_their_operation() {
        for (message, error) in [
            (
                "Qwen3.8 Flash-Next engram dequantization row count 9 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("engram dequantization", 9),
            ),
            (
                "Qwen3.8 Flash-Next engram convolution row count 16 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("engram convolution", 16),
            ),
            (
                "Qwen3.8 Flash-Next engram injection row count 2048 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("engram injection", 2_048),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }
}

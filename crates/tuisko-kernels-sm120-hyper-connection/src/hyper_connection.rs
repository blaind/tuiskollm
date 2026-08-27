//! Exact Qwen3.8-Flash-Next hyper-connection (gated-residual) operators.
//!
//! Each gated residual owns the target's pre-block `hc_norm`; the collapsing
//! `hyper_connection_mixer` owns its final normalization. For one row of the
//! widened `HC_COUNT * HIDDEN` stream:
//!
//! ```text
//! hn    = hc_norm(h)                        # grouped RMSNorm, then (1 + w)
//! t     = silu(down(hn) / 4)                # [320]
//! m     = sigmoid(up(t))                    # [10240]
//! mixed = mean_c(m * hn)                    # [2560]
//! w_inj = 2 * sigmoid(inject(hn) / 4)       # [4], absent from the mixer
//! ```
//!
//! The write-back adds into the raw stream:
//!
//! ```text
//! h' = h + broadcast_c(block_output) * w_inj
//! ```
//!
//! The checkpoint stores unfolded `hc_norm` weights, so `(1 + w)` remains in
//! the norm epilogue. The projections are sequentially dependent; caller-owned
//! intermediates preserve their BF16 boundaries and avoid repeated norms.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen38FlashNext};

/// Compact batching owns one compiled route for every `B=1..8`.
const MAX_BATCH: usize = 8;
/// Prefill tile widths this family admits, matching the other per-token owners.
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

/// Parallel residual branches carried by the widened stream.
const BRANCHES: usize = Qwen38FlashNext::HC_COUNT;
/// Width of one branch, which is also the block input and output width.
const BRANCH: usize = Qwen38FlashNext::HIDDEN;
/// Width of the widened residual stream.
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
/// Rank of the read gate's low-rank projection pair.
const RANK: usize = Qwen38FlashNext::HC_LOWRANK;
/// Pinned `rms_norm_eps` used by the grouped `hc_norm`.
const EPSILON: f32 = Qwen38FlashNext::RMS_NORM_EPSILON;
/// Scalar `/ hc_count` applied before both nonlinearities.
const HC_DIVISOR: f32 = 1.0 / BRANCHES as f32;

/// Packed BF16 words in one widened residual row.
const ROW_WORDS: usize = WIDTH / 2;
/// Packed BF16 words in one branch of one widened residual row.
const BRANCH_WORDS: usize = BRANCH / 2;
/// Packed BF16 words in one low-rank row.
const RANK_WORDS: usize = RANK / 2;

// The grouped norm reduces each branch independently, so the CTA is sized to
// the branch rather than the widened row: 2,560 values are 1,280 packed pairs,
// exactly five per thread across eight warps. An exact pair-per-thread mapping
// is what fixes the reduction order for every admitted row count.
const NORM_WARPS: usize = 8;
const NORM_THREADS: u32 = (NORM_WARPS * 32) as u32;
/// Packed pairs one thread consumes per branch of the grouped norm.
const NORM_PAIRS_PER_THREAD: usize = BRANCH_WORDS / NORM_THREADS as usize;

// Both low-rank projections are warp-per-output-row GEMVs: one warp owns one
// output row's whole reduction, so the accumulation order is fixed by the lane
// stride and the five-step butterfly and never varies with the row count.
// Register-blocking eight tokens per CTA lets a weight row loaded once serve a
// whole decode batch or one prefill tile column.
const TOKEN_TILE: usize = MAX_BATCH;
const DOWN_WARPS: usize = 8;
const DOWN_THREADS: u32 = (DOWN_WARPS * 32) as u32;
/// CTAs that cover the 320 `input_mix_weight_down` rows, eight rows each.
const DOWN_ROW_GROUPS: usize = RANK / DOWN_WARPS;
/// Packed words one lane consumes per phase of the widened reduction.
const WORDS_PER_LANE: usize = 4;
/// Packed words one warp consumes per phase of the widened reduction.
const DOWN_PHASE_WORDS: usize = 32 * WORDS_PER_LANE;
/// Phases one warp needs to cover a complete widened row.
const DOWN_PHASES: usize = ROW_WORDS / DOWN_PHASE_WORDS;

const UP_WARPS: usize = 8;
const UP_THREADS: u32 = (UP_WARPS * 32) as u32;
/// CTAs that cover the 2,560 mixed columns, eight columns each. One warp owns
/// all four branch rows of its column so the four-way mean stays inside it.
const UP_COLUMN_GROUPS: usize = BRANCH / UP_WARPS;
/// Packed-word steps one lane takes across a low-rank row.
const UP_SLOTS: usize = RANK_WORDS / 32;

const _: () = assert!(BRANCHES == 4);
const _: () = assert!(BRANCH == 2_560);
const _: () = assert!(WIDTH == 10_240);
const _: () = assert!(RANK == 320);
const _: () = assert!(WIDTH == BRANCHES * BRANCH);
const _: () = assert!(NORM_PAIRS_PER_THREAD * NORM_THREADS as usize == BRANCH_WORDS);
const _: () = assert!(RANK.is_multiple_of(DOWN_WARPS));
const _: () = assert!(ROW_WORDS.is_multiple_of(DOWN_PHASE_WORDS));
const _: () = assert!(BRANCH.is_multiple_of(UP_WARPS));
const _: () = assert!(RANK_WORDS.is_multiple_of(32));

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, ptx_asm, tcgen05, warp};

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
    fn bf16_to_f32(bits: u16) -> f32 {
        convert::cvt_f32x2_bf16x2(bits as u32).0
    }

    /// Rounds an FP32 value through the BF16 grid the reference's tensors carry.
    ///
    /// The reference's low-rank projections are `nn.Linear` calls that return
    /// BF16, so the `/4`, nonlinearities, and branch product all read values
    /// already rounded once. Dropping this round-trip keeps extra FP32 precision.
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
            let value = if lane < NORM_WARPS {
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

    /// Normalizes one widened row per CTA: four independent 2,560-wide RMSNorms
    /// in FP32, flattened, then one 10,240-wide `(1 + w)`.
    #[inline(always)]
    unsafe fn hc_norm_body<const TOKENS: usize>(
        residual: *const u32,
        weight: *const u32,
        normalized: *mut u32,
        warp_sums: *mut f32,
        scales: *mut f32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        // SAFETY: the launch contract gives every active block one complete row.
        let residual = unsafe { residual.add(token * ROW_WORDS) };
        // SAFETY: the output plane has the input's row coverage.
        let normalized = unsafe { normalized.add(token * ROW_WORDS) };

        macro_rules! reduce_branch {
            ($branch:literal) => {{
                let mut sum = 0.0f32;
                let mut word = tid;

                while word < BRANCH_WORDS {
                    // SAFETY: `word < BRANCH_WORDS` stays inside this branch.
                    let (low, high) = convert::cvt_f32x2_bf16x2(unsafe {
                        *residual.add($branch * BRANCH_WORDS + word)
                    });
                    sum = float::fma_rn_f32(low, low, sum);
                    sum = float::fma_rn_f32(high, high, sum);
                    word += NORM_THREADS as usize;
                }

                let sum = block_sum(sum, warp_sums, lane, warp_index);
                if tid == 0 {
                    // SAFETY: thread zero owns this branch's published scale.
                    unsafe {
                        *scales.add($branch) =
                            float::rsqrt_approx_f32(sum / BRANCH as f32 + EPSILON);
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
                // SAFETY: the barrier above published every branch scale.
                let scale = unsafe { *scales.add($branch) };
                let mut word = tid;

                while word < BRANCH_WORDS {
                    let index = $branch * BRANCH_WORDS + word;
                    // SAFETY: `word < BRANCH_WORDS` stays inside this branch.
                    let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *residual.add(index) });
                    // SAFETY: the weight plane covers one complete widened row.
                    let (weight_low, weight_high) =
                        convert::cvt_f32x2_bf16x2(unsafe { *weight.add(index) });
                    // SAFETY: each thread writes disjoint packed BF16 pairs.
                    unsafe {
                        *normalized.add(index) = tcgen05::cvt_f32x2_bf16x2(
                            low * scale * (1.0 + weight_low),
                            high * scale * (1.0 + weight_high),
                        );
                    }
                    word += NORM_THREADS as usize;
                }
            }};
        }

        write_branch!(0);
        write_branch!(1);
        write_branch!(2);
        write_branch!(3);
    }

    /// Projects the normalized stream onto the low rank and, when the module
    /// combines, onto the four per-branch write gates.
    ///
    /// `INJECT` selects the reference's `use_combine` arm: `true` appends the
    /// four `block_inject_weight` rows and emits `2 * sigmoid(x / 4)`, `false`
    /// is the model-level mixer, which has no `block_inject_weight`.
    #[inline(always)]
    unsafe fn hc_down_body<const TOKENS: usize, const INJECT: bool>(
        normalized: *const u32,
        down: *const u32,
        inject: *const u32,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let row_groups = if INJECT {
            DOWN_ROW_GROUPS + 1
        } else {
            DOWN_ROW_GROUPS
        };
        let block = thread::blockIdx_x() as usize;
        let row_group = block % row_groups;
        let token_base = if TOKENS <= TOKEN_TILE {
            0
        } else {
            (block / row_groups) * TOKEN_TILE
        };
        let gate_group = INJECT && row_group == DOWN_ROW_GROUPS;
        if gate_group && warp_index >= BRANCHES {
            return;
        }

        let row = if gate_group {
            warp_index
        } else {
            row_group * DOWN_WARPS + warp_index
        };
        let plane = if gate_group { inject } else { down };
        // SAFETY: `row` is inside the selected plane's row count.
        let plane = unsafe { plane.add(row * ROW_WORDS) };
        let mut sums = [0.0f32; TOKEN_TILE];
        let mut phase = 0usize;

        while phase < DOWN_PHASES {
            let offset = phase * DOWN_PHASE_WORDS + lane * WORDS_PER_LANE;
            // SAFETY: `offset` stays inside one complete widened row.
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
                                normalized.add((token_base + $token) * ROW_WORDS + offset),
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
                        let scaled = round_bf16(projected) * HC_DIVISOR;
                        if INJECT && gate_group {
                            let gate = round_bf16(sigmoid(scaled));
                            // SAFETY: one lane owns this token's branch gate.
                            unsafe {
                                *write_gate.add((token_base + $token) * BRANCHES + row) =
                                    tcgen05::f32_to_bf16_rne(2.0 * gate);
                            }
                        } else {
                            // SAFETY: one lane owns this token's low-rank column.
                            unsafe {
                                *low_rank.add((token_base + $token) * RANK + row) =
                                    tcgen05::f32_to_bf16_rne(silu(scaled));
                            }
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

    /// Expands the low rank back to the widened width, gates the normalized
    /// stream with it, and folds the four branches into the block input.
    ///
    /// One warp owns one mixed column across all four branches so the FP32
    /// four-way mean happens on one lane in ascending branch order.
    #[inline(always)]
    unsafe fn hc_up_body<const TOKENS: usize>(
        normalized: *const u16,
        up: *const u32,
        low_rank: *const u32,
        mixed: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let block = thread::blockIdx_x() as usize;
        let column_group = block % UP_COLUMN_GROUPS;
        let token_base = if TOKENS <= TOKEN_TILE {
            0
        } else {
            (block / UP_COLUMN_GROUPS) * TOKEN_TILE
        };
        let column = column_group * UP_WARPS + warp_index;
        let mut first = [0.0f32; TOKEN_TILE];
        let mut second = [0.0f32; TOKEN_TILE];
        let mut third = [0.0f32; TOKEN_TILE];
        let mut fourth = [0.0f32; TOKEN_TILE];
        let mut slot = 0usize;

        while slot < UP_SLOTS {
            let offset = slot * 32 + lane;
            // SAFETY: every `up` row holds `RANK_WORDS` packed words.
            let weight_first = unsafe { *up.add(column * RANK_WORDS + offset) };
            // SAFETY: every `up` row holds `RANK_WORDS` packed words.
            let weight_second = unsafe { *up.add((BRANCH + column) * RANK_WORDS + offset) };
            // SAFETY: every `up` row holds `RANK_WORDS` packed words.
            let weight_third = unsafe { *up.add((2 * BRANCH + column) * RANK_WORDS + offset) };
            // SAFETY: every `up` row holds `RANK_WORDS` packed words.
            let weight_fourth = unsafe { *up.add((3 * BRANCH + column) * RANK_WORDS + offset) };

            macro_rules! accumulate_branch {
                ($token:literal, $weight:ident, $sums:ident, $low:ident, $high:ident) => {{
                    let (weight_low, weight_high) = convert::cvt_f32x2_bf16x2($weight);
                    $sums[$token] = float::fma_rn_f32(weight_low, $low, $sums[$token]);
                    $sums[$token] = float::fma_rn_f32(weight_high, $high, $sums[$token]);
                }};
            }

            macro_rules! accumulate {
                ($token:literal) => {
                    if token_base + $token < TOKENS {
                        // SAFETY: the token is inside the launched row count.
                        let activation =
                            unsafe { *low_rank.add((token_base + $token) * RANK_WORDS + offset) };
                        let (low, high) = convert::cvt_f32x2_bf16x2(activation);
                        accumulate_branch!($token, weight_first, first, low, high);
                        accumulate_branch!($token, weight_second, second, low, high);
                        accumulate_branch!($token, weight_third, third, low, high);
                        accumulate_branch!($token, weight_fourth, fourth, low, high);
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
            slot += 1;
        }

        macro_rules! store {
            ($token:literal) => {
                if token_base + $token < TOKENS {
                    let token = token_base + $token;
                    let gate_first = reduce_sum_lane_zero(first[$token]);
                    let gate_second = reduce_sum_lane_zero(second[$token]);
                    let gate_third = reduce_sum_lane_zero(third[$token]);
                    let gate_fourth = reduce_sum_lane_zero(fourth[$token]);
                    if lane == 0 {
                        let base = token * WIDTH + column;
                        // SAFETY: the normalized plane covers every launched row.
                        let value_first = bf16_to_f32(unsafe { *normalized.add(base) });
                        // SAFETY: the normalized plane covers every launched row.
                        let value_second = bf16_to_f32(unsafe { *normalized.add(base + BRANCH) });
                        // SAFETY: the normalized plane covers every launched row.
                        let value_third =
                            bf16_to_f32(unsafe { *normalized.add(base + 2 * BRANCH) });
                        // SAFETY: the normalized plane covers every launched row.
                        let value_fourth =
                            bf16_to_f32(unsafe { *normalized.add(base + 3 * BRANCH) });
                        // The reference rounds each BF16 branch product before
                        // the FP32 four-way mean.
                        let mut total =
                            round_bf16(round_bf16(sigmoid(round_bf16(gate_first))) * value_first);
                        total +=
                            round_bf16(round_bf16(sigmoid(round_bf16(gate_second))) * value_second);
                        total +=
                            round_bf16(round_bf16(sigmoid(round_bf16(gate_third))) * value_third);
                        total +=
                            round_bf16(round_bf16(sigmoid(round_bf16(gate_fourth))) * value_fourth);
                        // SAFETY: one lane owns this token's mixed column.
                        unsafe {
                            *mixed.add(token * BRANCH + column) =
                                tcgen05::f32_to_bf16_rne(total * HC_DIVISOR);
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

    /// Broadcasts one block output into the four branches of the **raw** stream,
    /// each scaled by its own write gate.
    ///
    /// The stream this reads is not necessarily the previous write-back's
    /// output: the engram layer first adds its PLE delta into the same widened
    /// stream, so no branch symmetry may be assumed.
    #[inline(always)]
    unsafe fn hc_write_back_body<const TOKENS: usize>(
        residual: *const u32,
        block_output: *const u32,
        write_gate: *const u16,
        output: *mut u32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: the launch contract gives every active block one complete row.
        let residual = unsafe { residual.add(token * ROW_WORDS) };
        // SAFETY: the output plane has the input's row coverage.
        let output = unsafe { output.add(token * ROW_WORDS) };
        // SAFETY: the block output holds one branch-wide row per token.
        let block_output = unsafe { block_output.add(token * BRANCH_WORDS) };

        macro_rules! inject_branch {
            ($branch:literal) => {{
                // SAFETY: the gate plane holds `BRANCHES` values per token.
                let gate = bf16_to_f32(unsafe { *write_gate.add(token * BRANCHES + $branch) });
                let mut word = tid;

                while word < BRANCH_WORDS {
                    let index = $branch * BRANCH_WORDS + word;
                    // SAFETY: `word < BRANCH_WORDS` stays inside this branch.
                    let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *residual.add(index) });
                    // SAFETY: `word < BRANCH_WORDS` stays inside the block row.
                    let (block_low, block_high) =
                        convert::cvt_f32x2_bf16x2(unsafe { *block_output.add(word) });
                    // SAFETY: each thread writes the packed pair it just read.
                    unsafe {
                        *output.add(index) = tcgen05::cvt_f32x2_bf16x2(
                            low + round_bf16(block_low * gate),
                            high + round_bf16(block_high * gate),
                        );
                    }
                    word += NORM_THREADS as usize;
                }
            }};
        }

        inject_branch!(0);
        inject_branch!(1);
        inject_branch!(2);
        inject_branch!(3);
    }

    /// Normalizes one exact decode batch of widened residual rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_norm<const TOKENS: usize>(
        residual: *const u32,
        weight: *const u32,
        normalized: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, NORM_WARPS, 16> = SharedArray::UNINIT;
        static mut SCALES: SharedArray<f32, BRANCHES, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();
        let scales = core::ptr::addr_of_mut!(SCALES).cast::<f32>();

        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { hc_norm_body::<TOKENS>(residual, weight, normalized, warp_sums, scales) };
    }

    /// Normalizes one exact prefill tile of widened residual rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_norm_prefill<const TOKENS: usize>(
        residual: *const u32,
        weight: *const u32,
        normalized: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, NORM_WARPS, 16> = SharedArray::UNINIT;
        static mut SCALES: SharedArray<f32, BRANCHES, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();
        let scales = core::ptr::addr_of_mut!(SCALES).cast::<f32>();

        // Prefill retains separate symbols so its resource authority cannot
        // drift with decode; the traversal and reduction order are identical.
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { hc_norm_body::<TOKENS>(residual, weight, normalized, warp_sums, scales) };
    }

    /// Projects one exact decode batch onto the low rank and the write gates.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_mix_down<const TOKENS: usize>(
        normalized: *const u32,
        down: *const u32,
        inject: *const u32,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        unsafe {
            hc_down_body::<TOKENS, true>(normalized, down, inject, low_rank, write_gate);
        }
    }

    /// Projects one exact prefill tile onto the low rank and the write gates.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_mix_down_prefill<const TOKENS: usize>(
        normalized: *const u32,
        down: *const u32,
        inject: *const u32,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        unsafe {
            hc_down_body::<TOKENS, true>(normalized, down, inject, low_rank, write_gate);
        }
    }

    /// Projects one exact decode batch onto the mixer's low rank.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_final_down<const TOKENS: usize>(
        normalized: *const u32,
        down: *const u32,
        low_rank: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        // `INJECT` is false, so the gate plane is never formed or read.
        unsafe {
            hc_down_body::<TOKENS, false>(
                normalized,
                down,
                core::ptr::null(),
                low_rank,
                core::ptr::null_mut(),
            );
        }
    }

    /// Projects one exact prefill tile onto the mixer's low rank.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_final_down_prefill<const TOKENS: usize>(
        normalized: *const u32,
        down: *const u32,
        low_rank: *mut u16,
    ) {
        // SAFETY: the launch contract pins the row-group and token-tile grid.
        // `INJECT` is false, so the gate plane is never formed or read.
        unsafe {
            hc_down_body::<TOKENS, false>(
                normalized,
                down,
                core::ptr::null(),
                low_rank,
                core::ptr::null_mut(),
            );
        }
    }

    /// Gates and folds one exact decode batch into its block input.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_mix_up<const TOKENS: usize>(
        normalized: *const u16,
        up: *const u32,
        low_rank: *const u32,
        mixed: *mut u16,
    ) {
        // SAFETY: the launch contract pins the column-group and token-tile grid.
        unsafe { hc_up_body::<TOKENS>(normalized, up, low_rank, mixed) };
    }

    /// Gates and folds one exact prefill tile into its block input.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_mix_up_prefill<const TOKENS: usize>(
        normalized: *const u16,
        up: *const u32,
        low_rank: *const u32,
        mixed: *mut u16,
    ) {
        // SAFETY: the launch contract pins the column-group and token-tile grid.
        unsafe { hc_up_body::<TOKENS>(normalized, up, low_rank, mixed) };
    }

    /// Injects one exact decode batch's block output into the raw stream.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_write_back<const TOKENS: usize>(
        residual: *const u32,
        block_output: *const u32,
        write_gate: *const u16,
        output: *mut u32,
    ) {
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { hc_write_back_body::<TOKENS>(residual, block_output, write_gate, output) };
    }

    /// Injects one exact prefill tile's block output into the raw stream.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_hyper_connection_write_back_prefill<const TOKENS: usize>(
        residual: *const u32,
        block_output: *const u32,
        write_gate: *const u16,
        output: *mut u32,
    ) {
        // SAFETY: the launch contract pins one complete row per active block.
        unsafe { hc_write_back_body::<TOKENS>(residual, block_output, write_gate, output) };
    }
}

mod private {
    pub trait Sealed {}
}

/// CTAs the mixing projection needs for one exact row count.
const fn down_blocks(tokens: usize, inject: bool) -> u32 {
    let groups = if inject {
        DOWN_ROW_GROUPS + 1
    } else {
        DOWN_ROW_GROUPS
    };

    (groups * tokens.div_ceil(TOKEN_TILE)) as u32
}

/// CTAs the expanding projection needs for one exact row count.
const fn up_blocks(tokens: usize) -> u32 {
    (UP_COLUMN_GROUPS * tokens.div_ceil(TOKEN_TILE)) as u32
}

/// One exact row count's prepared entries for the whole family.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entries the module does not emit.
trait HyperConnectionRoute: Sized + private::Sealed {
    /// Prepares every entry of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's grouped `hc_norm` entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextHyperConnectionOp::launch_input_mix`]'s
    /// contract unchanged.
    unsafe fn launch_norm(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        weight: *const u16,
        normalized: *mut u16,
    ) -> GpuResult<()>;

    /// Launches this route's combining low-rank projection entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextHyperConnectionOp::launch_input_mix`]'s
    /// contract unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_mix_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        inject: *const u16,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) -> GpuResult<()>;

    /// Launches this route's mixer low-rank projection entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextHyperConnectionOp::launch_final_mix`]'s
    /// contract unchanged.
    unsafe fn launch_final_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        low_rank: *mut u16,
    ) -> GpuResult<()>;

    /// Launches this route's expanding projection and branch-fold entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextHyperConnectionOp::launch_input_mix`]'s
    /// contract unchanged.
    unsafe fn launch_mix_up(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        up: *const u16,
        low_rank: *const u16,
        mixed: *mut u16,
    ) -> GpuResult<()>;

    /// Launches this route's raw-stream write-back entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextHyperConnectionOp::launch_write_back`]'s
    /// contract unchanged.
    unsafe fn launch_write_back(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        block_output: *const u16,
        write_gate: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// Prepared decode entries for one exact batch.
struct PreparedDecodeRoute<const TOKENS: usize> {
    norm: PreparedLaunch<kernels::__qwen38_flash_next_hyper_connection_norm_CudaKernel<TOKENS>>,
    mix_down:
        PreparedLaunch<kernels::__qwen38_flash_next_hyper_connection_mix_down_CudaKernel<TOKENS>>,
    final_down:
        PreparedLaunch<kernels::__qwen38_flash_next_hyper_connection_final_down_CudaKernel<TOKENS>>,
    mix_up: PreparedLaunch<kernels::__qwen38_flash_next_hyper_connection_mix_up_CudaKernel<TOKENS>>,
    write_back:
        PreparedLaunch<kernels::__qwen38_flash_next_hyper_connection_write_back_CudaKernel<TOKENS>>,
}

/// Prepared prefill entries for one exact tile width.
///
/// Prefill retains separate symbols so its resource authority cannot drift
/// with decode.
struct PreparedPrefillRoute<const TOKENS: usize> {
    norm: PreparedLaunch<
        kernels::__qwen38_flash_next_hyper_connection_norm_prefill_CudaKernel<TOKENS>,
    >,
    mix_down: PreparedLaunch<
        kernels::__qwen38_flash_next_hyper_connection_mix_down_prefill_CudaKernel<TOKENS>,
    >,
    final_down: PreparedLaunch<
        kernels::__qwen38_flash_next_hyper_connection_final_down_prefill_CudaKernel<TOKENS>,
    >,
    mix_up: PreparedLaunch<
        kernels::__qwen38_flash_next_hyper_connection_mix_up_prefill_CudaKernel<TOKENS>,
    >,
    write_back: PreparedLaunch<
        kernels::__qwen38_flash_next_hyper_connection_write_back_prefill_CudaKernel<TOKENS>,
    >,
}

impl<const TOKENS: usize> private::Sealed for PreparedDecodeRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedPrefillRoute<TOKENS> {}

impl<const TOKENS: usize> HyperConnectionRoute for PreparedDecodeRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let rows = u32::try_from(TOKENS).map_err(|_| {
            GpuError::invalid_launch("hyper-connection batch exceeds CUDA grid width")
        })?;
        let norm = module
            .prepare_qwen38_flash_next_hyper_connection_norm::<TOKENS>(LaunchConfig1D::new(
                rows,
                NORM_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the hyper-connection norm", source))?;
        let mix_down = module
            .prepare_qwen38_flash_next_hyper_connection_mix_down::<TOKENS>(LaunchConfig1D::new(
                down_blocks(TOKENS, true),
                DOWN_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the hyper-connection mix projection", source)
            })?;
        let final_down =
            module
                .prepare_qwen38_flash_next_hyper_connection_final_down::<TOKENS>(
                    LaunchConfig1D::new(down_blocks(TOKENS, false), DOWN_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing the hyper-connection mixer projection", source)
                })?;
        let mix_up = module
            .prepare_qwen38_flash_next_hyper_connection_mix_up::<TOKENS>(LaunchConfig1D::new(
                up_blocks(TOKENS),
                UP_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing the hyper-connection fold", source))?;
        let write_back =
            module
                .prepare_qwen38_flash_next_hyper_connection_write_back::<TOKENS>(
                    LaunchConfig1D::new(rows, NORM_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing the hyper-connection write-back", source)
                })?;

        Ok(Self {
            norm,
            mix_down,
            final_down,
            mix_up,
            write_back,
        })
    }

    unsafe fn launch_norm(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        weight: *const u16,
        normalized: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_norm::<TOKENS>(
                stream,
                &self.norm,
                residual.cast::<u32>(),
                weight.cast::<u32>(),
                normalized.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the hyper-connection norm", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_mix_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        inject: *const u16,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_mix_down::<TOKENS>(
                stream,
                &self.mix_down,
                normalized.cast::<u32>(),
                down.cast::<u32>(),
                inject.cast::<u32>(),
                low_rank,
                write_gate,
            )
            .map_err(|source| {
                GpuError::launch("launching the hyper-connection mix projection", source)
            })
    }

    unsafe fn launch_final_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        low_rank: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_final_down::<TOKENS>(
                stream,
                &self.final_down,
                normalized.cast::<u32>(),
                down.cast::<u32>(),
                low_rank,
            )
            .map_err(|source| {
                GpuError::launch("launching the hyper-connection mixer projection", source)
            })
    }

    unsafe fn launch_mix_up(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        up: *const u16,
        low_rank: *const u16,
        mixed: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_mix_up::<TOKENS>(
                stream,
                &self.mix_up,
                normalized,
                up.cast::<u32>(),
                low_rank.cast::<u32>(),
                mixed,
            )
            .map_err(|source| GpuError::launch("launching the hyper-connection fold", source))
    }

    unsafe fn launch_write_back(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        block_output: *const u16,
        write_gate: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_write_back::<TOKENS>(
                stream,
                &self.write_back,
                residual.cast::<u32>(),
                block_output.cast::<u32>(),
                write_gate,
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the hyper-connection write-back", source))
    }
}

impl<const TOKENS: usize> HyperConnectionRoute for PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let rows = u32::try_from(TOKENS).map_err(|_| {
            GpuError::invalid_launch("hyper-connection prefill exceeds CUDA grid width")
        })?;
        let norm =
            module
                .prepare_qwen38_flash_next_hyper_connection_norm_prefill::<TOKENS>(
                    LaunchConfig1D::new(rows, NORM_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing the hyper-connection prefill norm", source)
                })?;
        let mix_down = module
            .prepare_qwen38_flash_next_hyper_connection_mix_down_prefill::<TOKENS>(
                LaunchConfig1D::new(down_blocks(TOKENS, true), DOWN_THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch(
                    "preparing the hyper-connection prefill mix projection",
                    source,
                )
            })?;
        let final_down = module
            .prepare_qwen38_flash_next_hyper_connection_final_down_prefill::<TOKENS>(
                LaunchConfig1D::new(down_blocks(TOKENS, false), DOWN_THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch(
                    "preparing the hyper-connection prefill mixer projection",
                    source,
                )
            })?;
        let mix_up = module
            .prepare_qwen38_flash_next_hyper_connection_mix_up_prefill::<TOKENS>(
                LaunchConfig1D::new(up_blocks(TOKENS), UP_THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch("preparing the hyper-connection prefill fold", source)
            })?;
        let write_back = module
            .prepare_qwen38_flash_next_hyper_connection_write_back_prefill::<TOKENS>(
                LaunchConfig1D::new(rows, NORM_THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch("preparing the hyper-connection prefill write-back", source)
            })?;

        Ok(Self {
            norm,
            mix_down,
            final_down,
            mix_up,
            write_back,
        })
    }

    unsafe fn launch_norm(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        weight: *const u16,
        normalized: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_norm_prefill::<TOKENS>(
                stream,
                &self.norm,
                residual.cast::<u32>(),
                weight.cast::<u32>(),
                normalized.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the hyper-connection prefill norm", source)
            })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_mix_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        inject: *const u16,
        low_rank: *mut u16,
        write_gate: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_mix_down_prefill::<TOKENS>(
                stream,
                &self.mix_down,
                normalized.cast::<u32>(),
                down.cast::<u32>(),
                inject.cast::<u32>(),
                low_rank,
                write_gate,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the hyper-connection prefill mix projection",
                    source,
                )
            })
    }

    unsafe fn launch_final_down(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        down: *const u16,
        low_rank: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_final_down_prefill::<TOKENS>(
                stream,
                &self.final_down,
                normalized.cast::<u32>(),
                down.cast::<u32>(),
                low_rank,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the hyper-connection prefill mixer projection",
                    source,
                )
            })
    }

    unsafe fn launch_mix_up(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        normalized: *const u16,
        up: *const u16,
        low_rank: *const u16,
        mixed: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_mix_up_prefill::<TOKENS>(
                stream,
                &self.mix_up,
                normalized,
                up.cast::<u32>(),
                low_rank.cast::<u32>(),
                mixed,
            )
            .map_err(|source| {
                GpuError::launch("launching the hyper-connection prefill fold", source)
            })
    }

    unsafe fn launch_write_back(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual: *const u16,
        block_output: *const u16,
        write_gate: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_hyper_connection_write_back_prefill::<TOKENS>(
                stream,
                &self.write_back,
                residual.cast::<u32>(),
                block_output.cast::<u32>(),
                write_gate,
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the hyper-connection prefill write-back", source)
            })
    }
}

/// PTX symbols retained for every admitted hyper-connection route.
pub(crate) fn hyper_connection_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<1>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<2>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<3>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<4>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<5>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<6>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<7>(),
        kernels::qwen38_flash_next_hyper_connection_norm_ptx_name::<8>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<1>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<2>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<3>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<4>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<5>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<6>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<7>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_ptx_name::<8>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<1>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<2>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<3>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<4>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<5>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<6>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<7>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_ptx_name::<8>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<1>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<2>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<3>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<4>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<5>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<6>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<7>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_ptx_name::<8>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<1>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<2>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<3>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<4>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<5>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<6>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<7>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_ptx_name::<8>(),
        kernels::qwen38_flash_next_hyper_connection_norm_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_hyper_connection_norm_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_hyper_connection_norm_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_hyper_connection_norm_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_hyper_connection_mix_down_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_hyper_connection_final_down_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_hyper_connection_mix_up_prefill_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_hyper_connection_write_back_prefill_ptx_name::<1_024>(),
    ]
}

/// The compiled route one admitted row count selects.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowRoute {
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    T32,
    T64,
    T128,
    T1024,
}

// Decode `B=1..=8` and prefill `T=32,64,128,1024`: the same schedule the other
// per-token owners route, with the small-batch entries kept separate from the
// tiles so a shape-aware dispatch never reaches a prefill body at decode.
#[cfg(test)]
fn row_route(rows: usize) -> Option<RowRoute> {
    match rows {
        1 => Some(RowRoute::B1),
        2 => Some(RowRoute::B2),
        3 => Some(RowRoute::B3),
        4 => Some(RowRoute::B4),
        5 => Some(RowRoute::B5),
        6 => Some(RowRoute::B6),
        7 => Some(RowRoute::B7),
        8 => Some(RowRoute::B8),
        32 => Some(RowRoute::T32),
        64 => Some(RowRoute::T64),
        128 => Some(RowRoute::T128),
        1_024 => Some(RowRoute::T1024),
        _ => None,
    }
}

fn unsupported_rows(operation: &str, rows: usize) -> GpuError {
    let prefill = PREFILL_ROWS.map(|rows| rows.to_string()).join(",");

    GpuError::invalid_launch(format!(
        "Qwen3.8-Flash-Next {operation} row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T={prefill}",
    ))
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_hyper_connection),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextHyperConnectionRoutes {
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

/// Prepared hyper-connection routes for decode `B=1..=8` and prefill
/// `T=32,64,128,1024`.
pub struct Qwen38FlashNextHyperConnectionOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextHyperConnectionRoutes,
}

impl Qwen38FlashNextHyperConnectionOp {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = hyper_connection_ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the hyper-connection module", source))?;

        Ok(Self {
            routes: Qwen38FlashNextHyperConnectionRoutes::prepare(&module)?,
            module,
        })
    }

    /// Runs one combining gated residual: `hc_norm`, both low-rank
    /// projections, the four-way branch fold, and the per-branch write gates.
    ///
    /// `normalized` and `low_rank` are the staged intermediates this route
    /// publishes. They are observable so that every fused boundary can be
    /// qualified; they are not part of the algebra a caller must interpret.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned except `write_gate`, which must
    /// be two-byte aligned. `residual` and `normalized` must cover
    /// `rows * HC_WIDTH` BF16 values, `weight` `HC_WIDTH` values, `down` and
    /// `up` `HC_LOWRANK * HC_WIDTH` values, `inject` `HC_COUNT * HC_WIDTH`
    /// values, `low_rank` `rows * HC_LOWRANK` values, `mixed` `rows * HIDDEN`
    /// values, and `write_gate` `rows * HC_COUNT` values. Allocations must
    /// belong to `stream`'s context, remain live through stream completion, and
    /// must not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_input_mix(
        &self,
        stream: &CudaStream,
        rows: usize,
        residual: *const u16,
        weight: *const u16,
        down: *const u16,
        up: *const u16,
        inject: *const u16,
        normalized: *mut u16,
        low_rank: *mut u16,
        mixed: *mut u16,
        write_gate: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {{
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_norm(&self.module, stream, residual, weight, normalized)?;
                    $route.launch_mix_down(
                        &self.module,
                        stream,
                        normalized,
                        down,
                        inject,
                        low_rank,
                        write_gate,
                    )?;
                    $route.launch_mix_up(&self.module, stream, normalized, up, low_rank, mixed)
                }
            }};
        }

        dispatch_qwen38_flash_next_hyper_connection!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("hyper-connection input mix", rows)))
    }

    /// Runs the model-level mixer without `block_inject_weight`; this is the
    /// target's only final norm.
    ///
    /// # Safety
    ///
    /// The pointer contract is [`Self::launch_input_mix`]'s without `inject`
    /// and `write_gate`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_final_mix(
        &self,
        stream: &CudaStream,
        rows: usize,
        residual: *const u16,
        weight: *const u16,
        down: *const u16,
        up: *const u16,
        normalized: *mut u16,
        low_rank: *mut u16,
        mixed: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {{
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_norm(&self.module, stream, residual, weight, normalized)?;
                    $route.launch_final_down(&self.module, stream, normalized, down, low_rank)?;
                    $route.launch_mix_up(&self.module, stream, normalized, up, low_rank, mixed)
                }
            }};
        }

        dispatch_qwen38_flash_next_hyper_connection!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("hyper-connection final mix", rows)))
    }

    /// Injects one block output into the raw widened stream.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned except `write_gate`, which must
    /// be two-byte aligned. `residual` and `output` must cover
    /// `rows * HC_WIDTH` BF16 values, `block_output` `rows * HIDDEN` values,
    /// and `write_gate` `rows * HC_COUNT` values. Allocations must belong to
    /// `stream`'s context and remain live through stream completion. The
    /// planes must not overlap except that `output` may alias `residual`
    /// exactly, which is the in-place production form.
    pub unsafe fn launch_write_back(
        &self,
        stream: &CudaStream,
        rows: usize,
        residual: *const u16,
        block_output: *const u16,
        write_gate: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    $route.launch_write_back(
                        &self.module,
                        stream,
                        residual,
                        block_output,
                        write_gate,
                        output,
                    )
                }
            };
        }

        dispatch_qwen38_flash_next_hyper_connection!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows("hyper-connection write-back", rows)))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BRANCH, BRANCHES, DOWN_PHASES, DOWN_ROW_GROUPS, DOWN_THREADS, HC_DIVISOR, MAX_BATCH,
        NORM_PAIRS_PER_THREAD, NORM_THREADS, NORM_WARPS, PREFILL_ROWS, RANK, RANK_WORDS, ROW_WORDS,
        RowRoute, TOKEN_TILE, UP_COLUMN_GROUPS, UP_SLOTS, UP_THREADS, WIDTH, down_blocks,
        hyper_connection_ptx_names, row_route, unsupported_rows, up_blocks,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_model::{Arch, Qwen38FlashNext};

    /// The exact schedule, decode first and prefill after.
    const ADMITTED_SCHEDULE: [(usize, RowRoute); 12] = [
        (1, RowRoute::B1),
        (2, RowRoute::B2),
        (3, RowRoute::B3),
        (4, RowRoute::B4),
        (5, RowRoute::B5),
        (6, RowRoute::B6),
        (7, RowRoute::B7),
        (8, RowRoute::B8),
        (32, RowRoute::T32),
        (64, RowRoute::T64),
        (128, RowRoute::T128),
        (1_024, RowRoute::T1024),
    ];

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    #[test]
    fn geometry_flows_from_the_admitted_architecture() {
        assert_eq!(BRANCHES, Qwen38FlashNext::HC_COUNT);
        assert_eq!(BRANCH, Qwen38FlashNext::HIDDEN);
        assert_eq!(WIDTH, Qwen38FlashNext::HC_WIDTH);
        assert_eq!(RANK, Qwen38FlashNext::HC_LOWRANK);
        assert_eq!(WIDTH, BRANCHES * BRANCH);
        assert_eq!(ROW_WORDS, 5_120);
        assert_eq!(RANK_WORDS, 160);
        assert_eq!(HC_DIVISOR, 0.25);
    }

    /// Every admitted width maps exactly onto its CTA, which is what keeps each
    /// entry's reduction order fixed for every row count.
    #[test]
    fn exact_geometry_is_cta_aligned() {
        assert_eq!(NORM_THREADS, 256);
        assert_eq!(NORM_WARPS, 8);
        assert_eq!(NORM_PAIRS_PER_THREAD, 5);
        assert_eq!(DOWN_THREADS, 256);
        assert_eq!(UP_THREADS, 256);
        assert_eq!(TOKEN_TILE, MAX_BATCH);
        assert_eq!(DOWN_ROW_GROUPS, 40);
        assert_eq!(DOWN_PHASES, 40);
        assert_eq!(UP_COLUMN_GROUPS, 320);
        assert_eq!(UP_SLOTS, 5);
        assert_eq!(DOWN_ROW_GROUPS * 8, RANK);
        assert_eq!(UP_COLUMN_GROUPS * 8, BRANCH);
        assert_eq!(UP_SLOTS * 32, RANK_WORDS);
    }

    /// The mixing projection adds exactly one CTA group for the four
    /// `block_inject_weight` rows; the mixer, lacking them, does not.
    #[test]
    fn grid_widths_cover_every_row_and_token_tile() {
        for &rows in &[1usize, 2, 3, 4, 5, 6, 7, 8] {
            assert_eq!(down_blocks(rows, true), 41);
            assert_eq!(down_blocks(rows, false), 40);
            assert_eq!(up_blocks(rows), 320);
        }
        for &rows in &PREFILL_ROWS {
            let tiles = (rows / TOKEN_TILE) as u32;
            assert_eq!(down_blocks(rows, true), 41 * tiles);
            assert_eq!(down_blocks(rows, false), 40 * tiles);
            assert_eq!(up_blocks(rows), 320 * tiles);
        }
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_stage_and_route() {
        let names = hyper_connection_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 5 * (MAX_BATCH + PREFILL_ROWS.len()));
        assert_eq!(names.len(), 60);
        assert_eq!(unique.len(), names.len());
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this family
    /// is its per-base-name count.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in hyper_connection_ptx_names() {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("qwen38_flash_next_hyper_connection_final_down", 8),
                ("qwen38_flash_next_hyper_connection_final_down_prefill", 4),
                ("qwen38_flash_next_hyper_connection_mix_down", 8),
                ("qwen38_flash_next_hyper_connection_mix_down_prefill", 4),
                ("qwen38_flash_next_hyper_connection_mix_up", 8),
                ("qwen38_flash_next_hyper_connection_mix_up_prefill", 4),
                ("qwen38_flash_next_hyper_connection_norm", 8),
                ("qwen38_flash_next_hyper_connection_norm_prefill", 4),
                ("qwen38_flash_next_hyper_connection_write_back", 8),
                ("qwen38_flash_next_hyper_connection_write_back_prefill", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 60);
    }

    /// Every row count the table admits, swept exhaustively so an unadmitted
    /// width cannot hide between the transcribed ones.
    #[test]
    fn row_routing_is_exact() {
        let admitted = (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|rows| row_route(rows).map(|route| (rows, route)))
            .collect::<Vec<_>>();

        assert_eq!(admitted, ADMITTED_SCHEDULE.to_vec());
    }

    #[test]
    fn unadmitted_row_counts_name_their_operation() {
        for (message, error) in [
            (
                "Qwen3.8-Flash-Next hyper-connection input mix row count 9 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("hyper-connection input mix", 9),
            ),
            (
                "Qwen3.8-Flash-Next hyper-connection final mix row count 16 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("hyper-connection final mix", 16),
            ),
            (
                "Qwen3.8-Flash-Next hyper-connection write-back row count 2048 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows("hyper-connection write-back", 2_048),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }
}

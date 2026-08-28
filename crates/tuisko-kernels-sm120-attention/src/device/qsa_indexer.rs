//! Device bodies for the Qwen3.8-Flash-Next sparse-attention indexer.
//!
//! Four stages turn one round's hidden states into an ascending key-position list:
//!
//! ```text
//! prepare   index_qk -> q_layernorm -> MRoPE at the query position  -> query plane
//!                    -> raw token_k, neither normed nor rotated     -> raw-key ring
//! compress  4 raw keys -> fp32 mean -> k_layernorm -> MRoPE at the
//!           block's first token                                     -> block-key pages
//! score     s[b] = (1/sqrt(128)) * sum_h relu(q_h . kbar_b)         -> score plane
//! select    top-512 blocks + the unconditional tail                 -> selected positions
//! ```
//!
//! Cached stages resolve addresses through the sequence's block-table row.
//! The raw keys are the exception, and deliberately: they are consumed by the
//! compression of their own block and dead afterwards, so they live in a
//! four-slot per-sequence ring rather than a paged plane, and a prompt tile
//! keeps them round-locally. See [`ring_element`].

use cuda_device::{float, tcgen05, thread, warp};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Width of one indexer head.
pub(crate) const INDEXER_HEAD_DIM: usize = Qwen38FlashNext::INDEXER_HEAD_DIM;
/// Indexer query heads whose ReLU scores are summed into one block score.
pub(crate) const INDEXER_HEADS: usize = Qwen38FlashNext::INDEXER_HEADS;
/// Rows of the fused indexer query and key projection.
pub(crate) const INDEXER_ROWS: usize = Qwen38FlashNext::INDEXER_ROWS;
/// Tokens pooled into one micro-block.
pub(crate) const COMPRESS_RATIO: usize = Qwen38FlashNext::INDEXER_COMPRESS_RATIO;
/// Blocks one query selects once the candidates exceed the budget.
pub(crate) const BLOCK_TOPK: usize = Qwen38FlashNext::INDEXER_BUDGET / COMPRESS_RATIO;
/// Widest selected-position list one query can own.
pub(crate) const MAX_SELECTED: usize =
    Qwen38FlashNext::INDEXER_BUDGET + Qwen38FlashNext::INDEXER_COMPRESS_RATIO - 1;
/// Token positions one physical cache page holds.
pub(crate) const PAGE_SIZE: usize = 64;
/// Micro-blocks one physical cache page holds.
pub(crate) const BLOCKS_PER_PAGE: usize = PAGE_SIZE / COMPRESS_RATIO;
/// Rows one scoring or selection tile owns.
pub(crate) const SELECT_ROW_TILE: usize = 64;
/// Warps one indexer CTA launches.
pub(crate) const WARPS_PER_CTA: usize = 8;
/// Threads one indexer CTA launches.
pub(crate) const THREADS_PER_CTA: usize = WARPS_PER_CTA * 32;
/// Radix bins the block selection resolves per pass.
const RADIX_BINS: usize = 256;
/// Passes the block selection spends resolving its exact threshold key.
const RADIX_PASSES: usize = 4;
/// Widest per-row CTA split the selection schedule prepares.
///
/// Up to 64 CTAs keep the deepest admitted row from serializing on one SM.
pub(crate) const SELECT_MAX_CTAS_PER_ROW: usize = 64;
/// Partial-histogram slots the selection scratch funds.
///
/// A launch owns `rows * ctas_per_row` slots and the schedule never exceeds
/// this many, so one fixed plane serves every admitted width.
pub(crate) const SELECT_PARTIAL_SLOTS: usize = 512;
/// Words the selection scratch plane owns.
///
/// Three regions in one allocation: the double-buffered partial histograms one
/// pass publishes and the next reduces, the per-row `(threshold, remaining)`
/// chain that carries a pass's answer to its successor, and the per-CTA
/// above-the-digit counts the expansion turns into ascending output bases.
pub(crate) const SELECT_SCRATCH_WORDS: usize =
    SELECT_SCRATCH_ABOVE + SELECT_ROW_TILE * (RADIX_PASSES - 1) * SELECT_MAX_CTAS_PER_ROW;
/// Words one partial-histogram buffer owns.
const SELECT_PARTIAL_WORDS: usize = SELECT_PARTIAL_SLOTS * RADIX_BINS;
/// First scratch word of the per-row pass state.
const SELECT_SCRATCH_STATE: usize = 2 * SELECT_PARTIAL_WORDS;
/// First scratch word of the per-CTA above-the-digit counts.
const SELECT_SCRATCH_ABOVE: usize = SELECT_SCRATCH_STATE + SELECT_ROW_TILE * RADIX_PASSES * 2;
/// Shared words one selection pass CTA owns: a private histogram per warp, two
/// scan rows the chunk loop alternates between, and the scalars a pass
/// publishes.
pub(crate) const SELECT_PASS_SHARED_WORDS: usize =
    WARPS_PER_CTA * RADIX_BINS + 2 * WARPS_PER_CTA + 4;
/// Shared words one selection expansion CTA owns.
pub(crate) const SELECT_EXPAND_SHARED_WORDS: usize = 2 * WARPS_PER_CTA + 4;

// One thread owns one radix bin in every reduction, which is what lets a pass
// resolve its digit with one CTA-wide scan instead of a serial bin walk.
const _: () = assert!(THREADS_PER_CTA == RADIX_BINS);
const _: () = assert!(SELECT_MAX_CTAS_PER_ROW <= THREADS_PER_CTA);

const VALUES_PER_LANE: usize = INDEXER_HEAD_DIM / 32;
const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
// Lane L owns dimensions `[4L, 4L+3]`, so the NeoX half-split partner of a
// rotating dimension `d`, whose partner is `d +/- 32`, sits eight lanes away.
const HALF_SPLIT_LANES: u32 = (ROTARY_PAIRS / VALUES_PER_LANE) as u32;
const HEADS_PER_TOKEN: usize = INDEXER_HEADS + Qwen38FlashNext::INDEXER_KV_HEADS;
// `1 / sqrt(128)`. The reference divides after the ReLU sum; the factor is
// positive and uniform, so it fixes the published score scale and cannot
// reorder the ranking the selection reads.
const SCORE_SCALE: f32 = 0.088_388_35;

const _: () = assert!(INDEXER_HEAD_DIM == 128);
const _: () = assert!(INDEXER_HEADS == 4);
const _: () = assert!(INDEXER_ROWS == 640);
const _: () = assert!(COMPRESS_RATIO == 4);
const _: () = assert!(BLOCK_TOPK == 512);
const _: () = assert!(MAX_SELECTED == 2_051);
const _: () = assert!(VALUES_PER_LANE == 4);
const _: () = assert!(HALF_SPLIT_LANES == 8);
const _: () = assert!(HEADS_PER_TOKEN == 5);
const _: () = assert!(BLOCKS_PER_PAGE == 16);

#[inline(always)]
fn bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Rounds one FP32 value to the BF16 the reference materializes at this seam.
///
/// `cvt.rn.bf16x2.f32` is round-to-nearest-even, which is the rounding every
/// other BF16 store in this crate already uses.
#[inline(always)]
fn round_bf16(value: f32) -> f32 {
    f32::from_bits(tcgen05::cvt_f32x2_bf16x2(value, value) << 16)
}

#[inline(always)]
unsafe fn load_bf16x4(source: *const u16) -> [f32; VALUES_PER_LANE] {
    unsafe {
        [
            bf16_bits(*source),
            bf16_bits(*source.add(1)),
            bf16_bits(*source.add(2)),
            bf16_bits(*source.add(3)),
        ]
    }
}

#[inline(always)]
unsafe fn store_bf16x4(destination: *mut u16, values: [f32; VALUES_PER_LANE]) {
    let destination = destination.cast::<u32>();
    unsafe {
        *destination = tcgen05::cvt_f32x2_bf16x2(values[0], values[1]);
        *destination.add(1) = tcgen05::cvt_f32x2_bf16x2(values[2], values[3]);
    }
}

#[inline(always)]
unsafe fn copy_bf16x4(destination: *mut u16, source: *const u16) {
    let destination = destination.cast::<u32>();
    let source = source.cast::<u32>();
    unsafe {
        *destination = *source;
        *destination.add(1) = *source.add(1);
    }
}

/// Resolves one sequence position to its `(physical_page, offset)` cache slot.
///
/// The pair is plane-independent: the key, value, indexer-key and block-key
/// planes all derive their element offset from it and differ only in the width
/// they multiply it by.
#[inline(always)]
unsafe fn resolve_slot(block_table: *const u32, position: usize) -> usize {
    let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
    (position & (PAGE_SIZE - 1)) + PAGE_SIZE * physical_page
}

/// Element offset of one micro-block's key in the block-key plane.
#[inline(always)]
unsafe fn block_key_element(block_table: *const u32, block: usize, dimension: usize) -> usize {
    let slot = unsafe { resolve_slot(block_table, block * COMPRESS_RATIO) };
    INDEXER_HEAD_DIM * (slot / PAGE_SIZE * BLOCKS_PER_PAGE + (block & (BLOCKS_PER_PAGE - 1)))
        + dimension
}

/// Element offset of one raw indexer key in the per-sequence ring.
///
/// A closed micro-block's raw keys are dead: the block key exists, and the at
/// most three tail tokens of the *open* block are attended unconditionally
/// through the K/V planes rather than scored. So the raw vectors never need a
/// paged plane; a sequence carries only its open block, four slots wide.
///
/// The engine never left-pads, so position `p` always belongs to block `p / 4`
/// at member `p % 4`.
#[inline(always)]
const fn ring_element(table_row: usize, position: usize, dimension: usize) -> usize {
    INDEXER_HEAD_DIM * (table_row * COMPRESS_RATIO + position % COMPRESS_RATIO) + dimension
}

/// Applies the 128-wide `(1 + w)` RMSNorm and rounds to the reference's BF16.
#[inline(always)]
unsafe fn indexer_normalize(
    values: [f32; VALUES_PER_LANE],
    norm: *const u16,
    dimension: usize,
) -> [f32; VALUES_PER_LANE] {
    let weights = unsafe { load_bf16x4(norm.add(dimension)) };
    let sum = values.iter().map(|value| value * value).sum::<f32>();
    let inverse_rms = float::rsqrt_approx_f32(
        warp::reduce_sum_f32(sum) / INDEXER_HEAD_DIM as f32
            + <Qwen38FlashNext as Arch>::RMS_NORM_EPSILON,
    );
    let mut normalized = [0.0f32; VALUES_PER_LANE];
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        normalized[element] = round_bf16(values[element] * inverse_rms * (1.0 + weights[element]));
        element += 1;
    }

    normalized
}

/// Rotates the leading 64 of the 128 indexer dimensions at one rotary row.
///
/// The rotary width is the model's, not the indexer's: `cos`/`sin` carry the
/// attention's 32 frequency pairs, so dimensions `0..63` rotate under the NeoX
/// half-split and `64..127` pass through carrying their post-norm value.
#[inline(always)]
unsafe fn indexer_rotate(
    values: [f32; VALUES_PER_LANE],
    rope_cos: *const f32,
    rope_sin: *const f32,
    row: usize,
    dimension: usize,
) -> [f32; VALUES_PER_LANE] {
    let mut rotated = values;
    if dimension < ROTARY_DIM {
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            let value = values[element];
            let peer = warp::shuffle_xor_f32(value, HALF_SPLIT_LANES);
            let rotary_element = dimension + element;
            let pair = rotary_element & (ROTARY_PAIRS - 1);
            let cosine = unsafe { *rope_cos.add(row * ROTARY_PAIRS + pair) };
            let sine = unsafe { *rope_sin.add(row * ROTARY_PAIRS + pair) };
            rotated[element] = round_bf16(if rotary_element < ROTARY_PAIRS {
                value * cosine - peer * sine
            } else {
                peer * sine + value * cosine
            });
            element += 1;
        }
    }

    rotated
}

/// Publishes the indexer query rows and appends this round's raw indexer keys.
///
/// `RING` selects where the raw key lands, and the two destinations are the two
/// shapes a round has. A decode round advances one position per sequence, so its
/// raw keys belong in that sequence's four-slot ring and have to survive until
/// the block closes. A prompt tile carries every position of one block it will
/// close in the *same* round, so its raw keys are round-local scratch indexed by
/// the token's own row and never outlive the compression that reads them.
///
/// # Safety
///
/// Every plane must address one complete row per launched warp. With `RING`,
/// `raw_keys` covers `[rows, 4, 128]`; without it, `[TOKENS, 128]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_prepare<const TOKENS: usize, const RING: bool>(
    indexer_qk: *const u16,
    query_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    table_rows: *const u32,
    cache_positions: *const u32,
    query: *mut f32,
    raw_keys: *mut u16,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = thread::blockIdx_x() as usize * WARPS_PER_CTA + tid / 32;
    let token = warp / HEADS_PER_TOKEN;
    if token >= TOKENS {
        return;
    }

    let head = warp - token * HEADS_PER_TOKEN;
    let dimension = lane * VALUES_PER_LANE;

    if head == INDEXER_HEADS {
        // The cached vector is the raw projection row: pooling, `k_layernorm`
        // and the rotation are all recomputed later from this plane, so the
        // store branches off the untouched source and never a prepared one.
        let source = unsafe {
            indexer_qk.add(token * INDEXER_ROWS + INDEXER_HEADS * INDEXER_HEAD_DIM + dimension)
        };
        let element = if RING {
            let table_row = unsafe { *table_rows.add(token) as usize };
            let position = unsafe { *cache_positions.add(token) as usize };
            ring_element(table_row, position, dimension)
        } else {
            INDEXER_HEAD_DIM * token + dimension
        };
        unsafe { copy_bf16x4(raw_keys.add(element), source) };

        return;
    }

    let source =
        unsafe { indexer_qk.add(token * INDEXER_ROWS + head * INDEXER_HEAD_DIM + dimension) };
    let values = unsafe { load_bf16x4(source) };
    let normalized = unsafe { indexer_normalize(values, query_norm, dimension) };
    let rotated = unsafe { indexer_rotate(normalized, rope_cos, rope_sin, token, dimension) };
    let destination =
        unsafe { query.add((token * INDEXER_HEADS + head) * INDEXER_HEAD_DIM + dimension) };
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *destination.add(element) = rotated[element] };
        element += 1;
    }
}

/// Compresses this round's newly completed micro-blocks into block keys.
///
/// The raw keys come out of the round's own plane and the published key goes to
/// the paged plane, so the two use different addressing on purpose: the block
/// key is per-sequence cache that every later query scores, and the raw keys it
/// pooled are dead the moment it exists.
///
/// `BLOCKS == 1` is the decode schedule: one sequence per row and at most one
/// closing block. Every wider schedule is
/// one prompt tile of a single sequence, whose raw keys sit in the round-local
/// plane at the tile row each position occupied.
///
/// # Safety
///
/// Every plane must address one complete row per launched warp, and every block
/// the count plane names must already hold all `COMPRESS_RATIO` raw keys. A
/// prompt round's first block must be tile-aligned, so that `4 * slot + member`
/// is the member's own row in the round-local plane.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_block_compress<const ROWS: usize, const BLOCKS: usize>(
    raw_keys: *const u16,
    key_norm: *const u16,
    block_rope_cos: *const f32,
    block_rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    first_blocks: *const u32,
    block_counts: *const u32,
    block_keys: *mut u16,
) {
    // A compile-time predicate: every instantiation fixes it, so the branch
    // below folds away rather than costing a comparison per warp.
    let ring = BLOCKS == 1;

    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = thread::blockIdx_x() as usize * WARPS_PER_CTA + tid / 32;
    let row = warp / BLOCKS;
    if row >= ROWS {
        return;
    }

    let slot = warp - row * BLOCKS;
    if slot >= unsafe { *block_counts.add(row) as usize } {
        return;
    }

    let block = unsafe { *first_blocks.add(row) as usize } + slot;
    let table_row = unsafe { *table_rows.add(row) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let dimension = lane * VALUES_PER_LANE;

    // Ascending FP32 accumulation over the four members, then one rounding to
    // BF16, matching `keys.float().mean(1).to(bf16)`.
    let mut pooled = [0.0f32; VALUES_PER_LANE];
    let mut member = 0usize;
    while member < COMPRESS_RATIO {
        let element = if ring {
            ring_element(table_row, member, dimension)
        } else {
            INDEXER_HEAD_DIM * (COMPRESS_RATIO * slot + member) + dimension
        };
        let raw = unsafe { load_bf16x4(raw_keys.add(element)) };
        let mut value = 0usize;
        while value < VALUES_PER_LANE {
            pooled[value] += raw[value];
            value += 1;
        }
        member += 1;
    }
    let mut value = 0usize;
    while value < VALUES_PER_LANE {
        pooled[value] = round_bf16(pooled[value] * 0.25);
        value += 1;
    }

    let normalized = unsafe { indexer_normalize(pooled, key_norm, dimension) };
    let rotated = unsafe {
        indexer_rotate(
            normalized,
            block_rope_cos,
            block_rope_sin,
            row * BLOCKS + slot,
            dimension,
        )
    };
    let destination = unsafe { block_key_element(block_table, block, dimension) };
    unsafe { store_bf16x4(block_keys.add(destination), rotated) };
}

/// Scores every candidate block of one row tile against its indexer query.
///
/// # Safety
///
/// `scores` must cover `[ROWS, score_stride]` values, `blocks_per_row` must be a
/// multiple of the warps per CTA, and the grid must supply
/// `ROWS * blocks_per_row / WARPS_PER_CTA` blocks.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_score<const ROWS: usize>(
    query: *const f32,
    block_keys: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    block_counts: *const u32,
    scores: *mut f32,
    score_stride: u32,
    row_offset: u32,
    blocks_per_row: u32,
) {
    let ctas_per_row = blocks_per_row as usize / WARPS_PER_CTA;
    let cta = thread::blockIdx_x() as usize;
    let tile_row = cta / ctas_per_row;
    if tile_row >= ROWS {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let candidate = (cta - tile_row * ctas_per_row) * WARPS_PER_CTA + tid / 32;
    let row = tile_row + row_offset as usize;
    if candidate >= unsafe { *block_counts.add(row) as usize } {
        return;
    }

    let table_row = unsafe { *table_rows.add(row) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let dimension = lane * VALUES_PER_LANE;
    let key = unsafe {
        load_bf16x4(block_keys.add(block_key_element(block_table, candidate, dimension)))
    };

    // ReLU is applied per head and the four results summed, so a head whose dot
    // product is negative contributes exactly zero rather than cancelling.
    let mut total = 0.0f32;
    let mut head = 0usize;
    while head < INDEXER_HEADS {
        let row_query =
            unsafe { query.add((row * INDEXER_HEADS + head) * INDEXER_HEAD_DIM + dimension) };
        let mut partial = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            partial = float::fma_rn_f32(unsafe { *row_query.add(element) }, key[element], partial);
            element += 1;
        }
        let reduced = warp::reduce_sum_f32(partial);
        total += if reduced > 0.0 { reduced } else { 0.0 };
        head += 1;
    }

    if lane == 0 {
        unsafe { *scores.add(tile_row * score_stride as usize + candidate) = total * SCORE_SCALE };
    }
}

#[inline(always)]
unsafe fn shared_zero(shared: *mut u32, words: usize) {
    let mut index = thread::threadIdx_x() as usize;
    while index < words {
        unsafe { *shared.add(index) = 0 };
        index += THREADS_PER_CTA;
    }
}

/// Returns this thread's exclusive prefix of `value` and the CTA total.
///
/// Both are counts, so the result is independent of warp scheduling and two
/// runs over the same scores publish the same selected list.
///
/// `scan` names one of two rows the caller alternates between successive calls.
/// A row is only rewritten after the barrier of the call that used the other
/// one, so one `sync_threads` carries a whole scan rather than the two a single
/// reused row would need.
#[inline(always)]
unsafe fn cta_exclusive_scan(scan: *mut u32, value: u32) -> (u32, u32) {
    let tid = thread::threadIdx_x() as usize;
    let lane = (tid & 31) as u32;
    let warp_index = tid / 32;
    let mut running = value;
    let mut offset = 1u32;
    while offset < 32 {
        let peer = warp::shuffle_up(running, offset);
        if lane >= offset {
            running += peer;
        }
        offset <<= 1;
    }
    if lane == 31 {
        unsafe { *scan.add(warp_index) = running };
    }
    thread::sync_threads();

    let mut base = 0u32;
    let mut total = 0u32;
    let mut index = 0usize;
    while index < WARPS_PER_CTA {
        let warp_total = unsafe { *scan.add(index) };
        if index < warp_index {
            base += warp_total;
        }
        total += warp_total;
        index += 1;
    }

    (base + running - value, total)
}

/// Blocks one CTA of a per-row split owns.
///
/// The even split is rounded up to a whole CTA stride so every slice begins on
/// a 256-block boundary and its loads stay one coalesced sector per warp; a
/// split whose last slices come out empty publishes zero bins and costs the
/// reduction nothing.
#[inline(always)]
const fn slice_blocks(count: usize, ctas: usize) -> usize {
    count.div_ceil(ctas).div_ceil(THREADS_PER_CTA) * THREADS_PER_CTA
}

/// Adds one score's radix digit to its warp's private histogram.
///
/// The prefix test is what makes a pass exact: only keys that already agree
/// with the resolved threshold on the higher bytes can decide this byte.
#[inline(always)]
unsafe fn tally(
    warp_histogram: *mut u32,
    key: u32,
    threshold: u32,
    prefix_mask: u32,
    shift: usize,
) {
    if key & prefix_mask != threshold & prefix_mask {
        return;
    }

    let digit = (key >> shift) & (RADIX_BINS as u32 - 1);
    // One lane per distinct digit updates its warp's private bin, so the
    // histogram is exact without an atomic.
    let peers = warp::match_any_sync(warp::active_mask(), digit);
    if warp::lane_id() == peers.trailing_zeros() {
        let bin = unsafe { warp_histogram.add(digit as usize) };
        unsafe { *bin += peers.count_ones() };
    }
}

/// Histograms one CTA's slice of the score plane into its private bins.
///
/// Four scores are in flight per thread. At one CTA per row the pass was bound
/// by a single outstanding load per thread rather than by bandwidth, and the
/// slice a split leaves each CTA is short enough that the tail loop rarely runs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn histogram_slice(
    scores: *const f32,
    warp_histogram: *mut u32,
    start: usize,
    end: usize,
    threshold: u32,
    prefix_mask: u32,
    shift: usize,
) {
    let stride = THREADS_PER_CTA;
    let mut index = start + thread::threadIdx_x() as usize;
    while index + 3 * stride < end {
        let first = unsafe { *scores.add(index) }.to_bits();
        let second = unsafe { *scores.add(index + stride) }.to_bits();
        let third = unsafe { *scores.add(index + 2 * stride) }.to_bits();
        let fourth = unsafe { *scores.add(index + 3 * stride) }.to_bits();
        unsafe {
            tally(warp_histogram, first, threshold, prefix_mask, shift);
            tally(warp_histogram, second, threshold, prefix_mask, shift);
            tally(warp_histogram, third, threshold, prefix_mask, shift);
            tally(warp_histogram, fourth, threshold, prefix_mask, shift);
        }
        index += 4 * stride;
    }
    while index < end {
        let key = unsafe { *scores.add(index) }.to_bits();
        unsafe { tally(warp_histogram, key, threshold, prefix_mask, shift) };
        index += stride;
    }
}

/// Resolves one radix digit from a row's published partial histograms.
///
/// Every CTA of the row reduces the whole partial plane, so all of them derive
/// the same digit from the same integers without a second launch to broadcast
/// it. `total` is this thread's bin summed over every slice, `before` the same
/// bin summed over the slices ahead of this one, and `own` this slice's.
#[inline(always)]
unsafe fn reduce_partials(
    partials: *const u32,
    bin: usize,
    ctas: usize,
    slice: usize,
) -> (u32, u32, u32) {
    let mut total = 0u32;
    let mut before = 0u32;
    let mut own = 0u32;
    let mut cta = 0usize;
    while cta < ctas {
        let value = unsafe { *partials.add(cta * RADIX_BINS + bin) };
        total += value;
        if cta < slice {
            before += value;
        }
        if cta == slice {
            own = value;
        }
        cta += 1;
    }

    (total, before, own)
}

/// Publishes the digit whose suffix total first reaches the outstanding count.
///
/// Thread `tid` owns bin `RADIX_BINS - 1 - tid`, so a plain exclusive scan over
/// thread order is the count of blocks strictly above that bin. Exactly one bin
/// straddles `remaining`, which is what makes the answer independent of the
/// order the scan happened to run in.
#[inline(always)]
unsafe fn resolve_digit(
    scan: *mut u32,
    publish: *mut u32,
    bin: usize,
    total: u32,
    remaining: u32,
) -> (u32, u32) {
    let (above, _) = unsafe { cta_exclusive_scan(scan, total) };
    if above < remaining && above + total >= remaining {
        unsafe { *publish = bin as u32 };
        unsafe { *publish.add(1) = above };
    }
    thread::sync_threads();

    (unsafe { *publish }, unsafe { *publish.add(1) })
}

/// Runs one radix pass of the block selection over a row's split score plane.
///
/// Scores are non-negative by construction, so their IEEE-754 bit patterns
/// order exactly as unsigned integers and a byte-at-a-time radix select is
/// exact. The four passes are sequenced as four launches because the digit one
/// pass resolves is the prefix the next filters on, and a partitioned histogram
/// is only complete once every CTA of the row has published its bins.
///
/// Pass `p` reduces pass `p-1`'s partial plane, extends the threshold and the
/// outstanding count, records how many of its own blocks sat above the digit,
/// and publishes its own bins for its successor. Every quantity it derives is
/// an integer count, so no split, warp order or launch order can move it.
///
/// # Safety
///
/// The grid must be `ROWS * ctas_per_row` blocks with `ctas_per_row` no greater
/// than [`SELECT_MAX_CTAS_PER_ROW`] and `ROWS * ctas_per_row` no greater than
/// [`SELECT_PARTIAL_SLOTS`]. `scratch` must cover [`SELECT_SCRATCH_WORDS`] and
/// `scores` the `[ROWS, score_stride]` plane the scorer published.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_select_pass<const ROWS: usize>(
    scores: *const f32,
    block_counts: *const u32,
    scratch: *mut u32,
    score_stride: u32,
    row_offset: u32,
    ctas_per_row: u32,
    pass: u32,
    shared: *mut u32,
) {
    let ctas = ctas_per_row as usize;
    let cta = thread::blockIdx_x() as usize;
    let tile_row = cta / ctas;
    if tile_row >= ROWS {
        return;
    }

    let slice = cta - tile_row * ctas;
    let tid = thread::threadIdx_x() as usize;
    let row = tile_row + row_offset as usize;
    let count = unsafe { *block_counts.add(row) as usize };
    // At or below the budget the expansion emits the whole visible list, so no
    // pass has anything to resolve.
    if count <= BLOCK_TOPK {
        return;
    }

    let pass = pass as usize;
    let scores = unsafe { scores.add(tile_row * score_stride as usize) };
    let scan = unsafe { shared.add(WARPS_PER_CTA * RADIX_BINS) };
    let publish = unsafe { scan.add(2 * WARPS_PER_CTA) };
    let state = unsafe { scratch.add(SELECT_SCRATCH_STATE + tile_row * RADIX_PASSES * 2) };
    let bin = RADIX_BINS - 1 - tid;

    let mut threshold = 0u32;
    let mut remaining = BLOCK_TOPK as u32;
    if pass > 0 {
        threshold = unsafe { *state.add((pass - 1) * 2) };
        remaining = unsafe { *state.add((pass - 1) * 2 + 1) };

        let previous = unsafe {
            scratch
                .add(((pass - 1) & 1) * SELECT_PARTIAL_WORDS + tile_row * ctas * RADIX_BINS)
                .cast_const()
        };
        let (total, _, own) = unsafe { reduce_partials(previous, bin, ctas, slice) };
        let (digit, above) = unsafe { resolve_digit(scan, publish, bin, total, remaining) };
        threshold |= digit << (24 - 8 * (pass - 1));
        remaining -= above;

        // The expansion turns these per-CTA counts into the ascending base of
        // every later slice, which is what removes the serial prefix walk the
        // one-CTA selection had to run over the whole plane.
        let contribution = if bin as u32 > digit { own } else { 0 };
        let (_, slice_above) = unsafe { cta_exclusive_scan(scan.add(WARPS_PER_CTA), contribution) };
        if tid == 0 {
            let above_plane = unsafe {
                scratch.add(
                    SELECT_SCRATCH_ABOVE
                        + (tile_row * (RADIX_PASSES - 1) + pass - 1) * SELECT_MAX_CTAS_PER_ROW,
                )
            };
            unsafe { *above_plane.add(slice) = slice_above };
        }
    }

    if slice == 0 && tid == 0 {
        unsafe { *state.add(pass * 2) = threshold };
        unsafe { *state.add(pass * 2 + 1) = remaining };
    }

    unsafe { shared_zero(shared, WARPS_PER_CTA * RADIX_BINS) };
    thread::sync_threads();

    let shift = 24 - 8 * pass;
    let prefix_mask = if pass == 0 {
        0u32
    } else {
        u32::MAX << (shift + 8)
    };
    let chunk = slice_blocks(count, ctas);
    let start = slice * chunk;
    let end = if start + chunk < count {
        start + chunk
    } else {
        count
    };
    unsafe {
        histogram_slice(
            scores,
            shared.add(tid / 32 * RADIX_BINS),
            start,
            end,
            threshold,
            prefix_mask,
            shift,
        );
    }
    thread::sync_threads();

    let mut sum = 0u32;
    let mut warp_index = 0usize;
    while warp_index < WARPS_PER_CTA {
        sum += unsafe { *shared.add(warp_index * RADIX_BINS + bin) };
        warp_index += 1;
    }
    let published = unsafe {
        scratch.add((pass & 1) * SELECT_PARTIAL_WORDS + (tile_row * ctas + slice) * RADIX_BINS)
    };
    unsafe { *published.add(bin) = sum };
}

/// Expands the resolved threshold into the ascending selected-position list.
///
/// The last radix pass leaves `threshold` one digit short and `remaining` equal
/// to the blocks tied at it the budget still admits. **Lowest block index
/// wins**, and the rule stays independent of thread and launch order because
/// every rank is an ascending prefix count: the blocks ahead of a slice come
/// from the partial histograms the passes published, and the blocks ahead of a
/// position inside a slice from one packed CTA scan per 256-block chunk.
///
/// # Safety
///
/// Carries [`indexer_select_pass`]'s contract, and every pass must already have
/// run over this tile. `selected` must cover `[rows, MAX_SELECTED]` values.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_select_expand<const ROWS: usize>(
    scores: *const f32,
    visible_lengths: *const u32,
    block_counts: *const u32,
    selected: *mut u32,
    selected_counts: *mut u32,
    scratch: *const u32,
    score_stride: u32,
    row_offset: u32,
    ctas_per_row: u32,
    shared: *mut u32,
) {
    let ctas = ctas_per_row as usize;
    let cta = thread::blockIdx_x() as usize;
    let tile_row = cta / ctas;
    if tile_row >= ROWS {
        return;
    }

    let slice = cta - tile_row * ctas;
    let tid = thread::threadIdx_x() as usize;
    let row = tile_row + row_offset as usize;
    let visible = unsafe { *visible_lengths.add(row) as usize };
    let count = unsafe { *block_counts.add(row) as usize };
    let scores = unsafe { scores.add(tile_row * score_stride as usize) };
    let selected = unsafe { selected.add(row * MAX_SELECTED) };
    let scan = shared;
    let publish = unsafe { scan.add(2 * WARPS_PER_CTA) };
    let stride = ctas * THREADS_PER_CTA;

    // At or below the budget every complete block is selected, so the union
    // with the tail is the whole visible list and the route is the dense one.
    if count <= BLOCK_TOPK {
        let mut index = slice * THREADS_PER_CTA + tid;
        while index < visible {
            unsafe { *selected.add(index) = index as u32 };
            index += stride;
        }
        if slice == 0 && tid == 0 {
            unsafe { *selected_counts.add(row) = visible as u32 };
        }

        return;
    }

    let state = unsafe { scratch.add(SELECT_SCRATCH_STATE + tile_row * RADIX_PASSES * 2) };
    let mut threshold = unsafe { *state.add((RADIX_PASSES - 1) * 2) };
    let mut remaining = unsafe { *state.add((RADIX_PASSES - 1) * 2 + 1) };
    let bin = RADIX_BINS - 1 - tid;
    let previous = unsafe {
        scratch.add(((RADIX_PASSES - 1) & 1) * SELECT_PARTIAL_WORDS + tile_row * ctas * RADIX_BINS)
    };
    let (total, before, _) = unsafe { reduce_partials(previous, bin, ctas, slice) };
    let (digit, above) = unsafe { resolve_digit(scan, publish, bin, total, remaining) };
    threshold |= digit;
    remaining -= above;

    // Blocks ahead of this slice, split the way the selection law splits them:
    // strictly above the threshold, and tied at it.
    let (_, base_tie) = unsafe {
        cta_exclusive_scan(
            scan.add(WARPS_PER_CTA),
            if bin as u32 == digit { before } else { 0 },
        )
    };
    let (_, last_above) =
        unsafe { cta_exclusive_scan(scan, if bin as u32 > digit { before } else { 0 }) };
    let mut earlier = 0u32;
    if tid < slice {
        let above_plane = unsafe {
            scratch
                .add(SELECT_SCRATCH_ABOVE + tile_row * (RADIX_PASSES - 1) * SELECT_MAX_CTAS_PER_ROW)
        };
        let mut pass = 0usize;
        while pass < RADIX_PASSES - 1 {
            earlier += unsafe { *above_plane.add(pass * SELECT_MAX_CTAS_PER_ROW + tid) };
            pass += 1;
        }
    }
    let (_, base_above) = unsafe { cta_exclusive_scan(scan.add(WARPS_PER_CTA), earlier) };

    // Ties are admitted in ascending block order, so the slices ahead of this
    // one consumed exactly `min(base_tie, remaining)` of the budget's tail.
    let taken_before = if base_tie < remaining {
        base_tie
    } else {
        remaining
    };
    let room = remaining - taken_before;
    let base_selected = base_above + last_above + taken_before;

    let chunk = slice_blocks(count, ctas);
    let start = slice * chunk;
    let end = if start + chunk < count {
        start + chunk
    } else {
        count
    };
    let mut above_seen = 0u32;
    let mut tie_seen = 0u32;
    let mut row_index = 0usize;
    let mut base = start;
    while base < end {
        let index = base + tid;
        let key = if index < end {
            unsafe { *scores.add(index) }.to_bits()
        } else {
            0
        };
        let is_above = index < end && key > threshold;
        let is_tie = index < end && key == threshold;
        // One packed scan carries both counts: the tie rank decides admission
        // and the selected rank decides the ascending slot. A chunk contributes
        // at most 256 to either field, so the halves never carry into one
        // another.
        let (prefix, total) = unsafe {
            cta_exclusive_scan(
                scan.add(row_index * WARPS_PER_CTA),
                u32::from(is_above) | (u32::from(is_tie) << 16),
            )
        };
        row_index ^= 1;
        let above_prefix = prefix & 0xffff;
        let tie_prefix = prefix >> 16;
        let ties_ahead = tie_seen + tie_prefix;
        let take = is_tie && ties_ahead < room;
        if is_above || take {
            let taken = if ties_ahead < room { ties_ahead } else { room };
            let rank = base_selected + above_seen + above_prefix + taken;
            let slot = rank * COMPRESS_RATIO as u32;
            let first = (index * COMPRESS_RATIO) as u32;
            let mut member = 0u32;
            while member < COMPRESS_RATIO as u32 {
                unsafe { *selected.add((slot + member) as usize) = first + member };
                member += 1;
            }
        }

        above_seen += total & 0xffff;
        tie_seen += total >> 16;
        base += THREADS_PER_CTA;
    }

    // Above the budget the selection is exactly `BLOCK_TOPK` blocks, so the
    // unconditional tail of the incomplete block always begins at the same
    // offset however the row's scores fell.
    let written = (BLOCK_TOPK * COMPRESS_RATIO) as u32;
    let tail_start = count * COMPRESS_RATIO;
    let mut tail = slice * THREADS_PER_CTA + tid;
    while tail_start + tail < visible {
        unsafe { *selected.add(written as usize + tail) = (tail_start + tail) as u32 };
        tail += stride;
    }
    if slice == 0 && tid == 0 {
        unsafe { *selected_counts.add(row) = written + (visible - tail_start) as u32 };
    }
}

//! Device bodies for the Qwen3.8-Flash-Next sparse-attention indexer.
//!
//! Four stages produce an ascending attention-position list:
//!
//! ```text
//! prepare   index_qk -> q_layernorm -> MRoPE at the query position  -> query plane
//!                    -> raw token_k, neither normed nor rotated     -> indexer pages
//! compress  4 raw keys -> fp32 mean -> k_layernorm -> MRoPE at the
//!           block's first token                                     -> block-key pages
//! score     s[b] = (1/sqrt(128)) * sum_h relu(q_h . kbar_b)         -> score plane
//! select    top-512 blocks + the unconditional tail                 -> selected positions
//! ```
//!
//! Cache addresses always resolve through the sequence's block-table row.

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
/// Shared words one selection CTA owns: a private histogram per warp, a scan
/// row, and the two scalars each pass publishes.
pub(crate) const SELECT_SHARED_WORDS: usize = WARPS_PER_CTA * RADIX_BINS + WARPS_PER_CTA + 2;

const VALUES_PER_LANE: usize = INDEXER_HEAD_DIM / 32;
const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
// Lane L owns dimensions `[4L, 4L+3]`, so the NeoX half-split partner of a
// rotating dimension `d`, which is `d +/- 32`, sits eight lanes away.
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
/// # Safety
///
/// Every plane must address one complete row per launched warp.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_prepare<const TOKENS: usize>(
    indexer_qk: *const u16,
    query_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    cache_positions: *const u32,
    query: *mut f32,
    indexer_pages: *mut u16,
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
        let table_row = unsafe { *table_rows.add(token) as usize };
        let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
        let position = unsafe { *cache_positions.add(token) as usize };
        let slot = unsafe { resolve_slot(block_table, position) };
        unsafe {
            copy_bf16x4(
                indexer_pages.add(INDEXER_HEAD_DIM * slot + dimension),
                source,
            )
        };

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
/// # Safety
///
/// Every plane must address one complete row per launched warp, and every block
/// the count plane names must already hold all `COMPRESS_RATIO` raw keys.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_block_compress<const ROWS: usize, const BLOCKS: usize>(
    indexer_pages: *const u16,
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
    // BF16, matching the `keys.float().mean(1).to(bf16)` seam.
    let mut pooled = [0.0f32; VALUES_PER_LANE];
    let mut member = 0usize;
    while member < COMPRESS_RATIO {
        let slot = unsafe { resolve_slot(block_table, block * COMPRESS_RATIO + member) };
        let raw = unsafe { load_bf16x4(indexer_pages.add(INDEXER_HEAD_DIM * slot + dimension)) };
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
    thread::sync_threads();

    (base + running - value, total)
}

/// Selects the block-granular top-`BLOCK_TOPK` and expands it to positions.
///
/// # Safety
///
/// One CTA owns one tile row. `selected` must cover `[rows, MAX_SELECTED]`
/// values and `scores` the `[ROWS, score_stride]` plane the scorer published.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn indexer_select<const ROWS: usize>(
    scores: *const f32,
    visible_lengths: *const u32,
    block_counts: *const u32,
    selected: *mut u32,
    selected_counts: *mut u32,
    score_stride: u32,
    row_offset: u32,
    shared: *mut u32,
) {
    let tile_row = thread::blockIdx_x() as usize;
    if tile_row >= ROWS {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let row = tile_row + row_offset as usize;
    let visible = unsafe { *visible_lengths.add(row) as usize };
    let count = unsafe { *block_counts.add(row) as usize };
    let scores = unsafe { scores.add(tile_row * score_stride as usize) };
    let selected = unsafe { selected.add(row * MAX_SELECTED) };
    let scan = unsafe { shared.add(WARPS_PER_CTA * RADIX_BINS) };
    let published = unsafe { scan.add(WARPS_PER_CTA) };

    // At or below the budget every complete block is selected, so the union
    // with the tail is the whole visible list and the route is the dense one.
    if count <= BLOCK_TOPK {
        let mut index = tid;
        while index < visible {
            unsafe { *selected.add(index) = index as u32 };
            index += THREADS_PER_CTA;
        }
        if tid == 0 {
            unsafe { *selected_counts.add(row) = visible as u32 };
        }

        return;
    }

    // Scores are non-negative by construction, so their IEEE-754 bit patterns
    // order exactly as unsigned integers and a plain radix select is exact.
    let mut threshold = 0u32;
    let mut remaining = BLOCK_TOPK as u32;
    let mut pass = 0usize;
    while pass < RADIX_PASSES {
        let shift = 24 - 8 * pass;
        let prefix_mask = if pass == 0 {
            0u32
        } else {
            u32::MAX << (shift + 8)
        };
        unsafe { shared_zero(shared, WARPS_PER_CTA * RADIX_BINS) };
        thread::sync_threads();

        let warp_histogram = unsafe { shared.add(tid / 32 * RADIX_BINS) };
        let mut index = tid;
        while index < count {
            let key = unsafe { *scores.add(index) }.to_bits();
            if key & prefix_mask == threshold & prefix_mask {
                let digit = (key >> shift) & (RADIX_BINS as u32 - 1);
                // One lane per distinct digit updates its warp's private bin,
                // so the histogram is exact without an atomic.
                let peers = warp::match_any_sync(warp::active_mask(), digit);
                if warp::lane_id() == peers.trailing_zeros() {
                    let bin = unsafe { warp_histogram.add(digit as usize) };
                    unsafe { *bin += peers.count_ones() };
                }
            }
            index += THREADS_PER_CTA;
        }
        thread::sync_threads();

        // Walk the bins downward: the first whose suffix total reaches the
        // outstanding count owns this pass's threshold digit.
        if tid == 0 {
            let mut suffix = 0u32;
            let mut bin = RADIX_BINS;
            while bin > 0 {
                bin -= 1;
                let mut total = 0u32;
                let mut warp_index = 0usize;
                while warp_index < WARPS_PER_CTA {
                    total += unsafe { *shared.add(warp_index * RADIX_BINS + bin) };
                    warp_index += 1;
                }
                if suffix + total >= remaining {
                    unsafe { *published = bin as u32 };
                    unsafe { *published.add(1) = suffix };
                    break;
                }
                suffix += total;
            }
        }
        thread::sync_threads();

        threshold |= unsafe { *published } << shift;
        remaining -= unsafe { *published.add(1) };
        thread::sync_threads();
        pass += 1;
    }

    // `remaining` now counts the blocks tied at the threshold the budget still
    // admits. Lowest block index wins, and the ascending prefix count makes
    // that independent of thread and launch order.
    let mut tie_base = 0u32;
    let mut written = 0u32;
    let mut chunk = 0usize;
    while chunk < count {
        let index = chunk + tid;
        let key = if index < count {
            unsafe { *scores.add(index) }.to_bits()
        } else {
            0
        };
        let is_greater = index < count && key > threshold;
        let is_tie = index < count && key == threshold;
        let (tie_prefix, tie_total) = unsafe { cta_exclusive_scan(scan, u32::from(is_tie)) };
        let take = is_tie && tie_base + tie_prefix < remaining;
        let (write_prefix, write_total) =
            unsafe { cta_exclusive_scan(scan, u32::from(is_greater || take)) };

        if is_greater || take {
            let base = written + write_prefix * COMPRESS_RATIO as u32;
            let first = (index * COMPRESS_RATIO) as u32;
            let mut member = 0u32;
            while member < COMPRESS_RATIO as u32 {
                unsafe { *selected.add((base + member) as usize) = first + member };
                member += 1;
            }
        }

        written += write_total * COMPRESS_RATIO as u32;
        tie_base += tie_total;
        chunk += THREADS_PER_CTA;
    }

    // The trailing tokens of the incomplete block are attended unconditionally.
    let tail_start = count * COMPRESS_RATIO;
    let mut tail = tid;
    while tail_start + tail < visible {
        unsafe { *selected.add(written as usize + tail) = (tail_start + tail) as u32 };
        tail += THREADS_PER_CTA;
    }
    if tid == 0 {
        unsafe { *selected_counts.add(row) = written + (visible - tail_start) as u32 };
    }
}

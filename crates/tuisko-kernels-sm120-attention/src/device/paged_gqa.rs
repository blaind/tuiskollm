use cuda_device::async_copy::{
    cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{DynamicSharedArray, convert, f16x2, float, thread, warp, wmma};
use tuisko_kernels_sm120_common::device::e4m3x2_to_f32;
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 8;
const PAGE_SIZE: usize = 64;
const WARP_THREADS: usize = 32;
const PREFILL_QUERY_WARPS: usize = 6;
pub(crate) const PREFILL_THREADS: usize = 384;
pub(crate) const QWEN36_FP8_PREFILL_THREADS: usize = 256;
const PREFILL_KEY_TILE: usize = 64;
const PREFILL_PLANE_WORDS: usize = PREFILL_KEY_TILE * 256 / size_of::<u32>();
pub(crate) const PREFILL_SHARED_BYTES: usize = 2 * PREFILL_PLANE_WORDS * size_of::<u32>();
pub(crate) const BF16_PREFILL_THREADS: usize = 256;
pub(crate) const QWEN35_BF16_PREFILL_THREADS: usize = 128;
const BF16_PREFILL_PLANE_VALUES: usize = PREFILL_KEY_TILE * 256;
pub(crate) const BF16_PREFILL_SHARED_BYTES: usize =
    2 * BF16_PREFILL_PLANE_VALUES * size_of::<u16>();
pub(crate) const PREFILL_PARTIAL_VALUES: usize = 258;
pub(crate) const DECODE_WARPS: usize = 8;
/// Decode threads per CTA, shared with the MTP BF16 decode route.
pub const DECODE_THREADS: usize = DECODE_WARPS * WARP_THREADS;
pub(crate) const DECODE_PARTIAL_VALUES: usize = 258;
/// Decode partial-reduction slots per CTA, shared with the MTP BF16 decode route.
pub const DECODE_SHARED_VALUES: usize = DECODE_WARPS * DECODE_PARTIAL_VALUES;
const FLASH_PREFILL_MMA_ROWS: usize = 16;
const FLASH_PREFILL_QUERY_GROUPS: usize = 2;
const FLASH_PREFILL_QUERY_ROWS: usize = FLASH_PREFILL_QUERY_GROUPS * FLASH_PREFILL_MMA_ROWS;
const FLASH_PREFILL_WARPS_PER_GROUP: usize = 4;
pub(crate) const FLASH_PREFILL_THREADS: usize =
    FLASH_PREFILL_QUERY_GROUPS * FLASH_PREFILL_WARPS_PER_GROUP * WARP_THREADS;
const FLASH_PREFILL_P16_KEY_TILE: usize = 32;
const FLASH_PREFILL_Q_BYTES: usize = FLASH_PREFILL_QUERY_ROWS * 256;
const FLASH_PREFILL_Q_SCALE_BYTES: usize = FLASH_PREFILL_QUERY_ROWS * core::mem::size_of::<f32>();
const FLASH_PREFILL_STATS_VALUES: usize = 3 * FLASH_PREFILL_QUERY_ROWS;

const fn flash_prefill_shared_bytes(key_tile: usize) -> usize {
    // Single buffering keeps P16 at two CTAs/SM. A second K=32 buffer
    // consumes 76,288 bytes and measured 1.63x slower at 98K; doubled K=64
    // exceeds the SM120 per-block shared-memory limit.
    FLASH_PREFILL_Q_BYTES
        + key_tile * 256
        + key_tile * 256
        + key_tile * 256 * 2
        + FLASH_PREFILL_QUERY_ROWS * key_tile * 2
        + FLASH_PREFILL_Q_SCALE_BYTES
        + FLASH_PREFILL_STATS_VALUES * core::mem::size_of::<f32>()
}

pub(crate) const FLASH_PREFILL_P8_SHARED_BYTES: usize =
    flash_prefill_shared_bytes(PREFILL_KEY_TILE);
pub(crate) const FLASH_PREFILL_P16_SHARED_BYTES: usize =
    flash_prefill_shared_bytes(FLASH_PREFILL_P16_KEY_TILE);
const _: () = assert!(FLASH_PREFILL_P8_SHARED_BYTES == 78_336);
const _: () = assert!(FLASH_PREFILL_P16_SHARED_BYTES == 43_520);
// Eight in-flight positions give the one-warp decode scan enough lookahead
// to cover K/V load latency; K and V halves of one slot are 512 bytes each.
const DECODE_RING_DEPTH: usize = 8;
/// BF16 decode ring bytes per CTA, shared with the MTP BF16 decode route.
pub const DECODE_RING_SHARED_BYTES: usize = DECODE_RING_DEPTH * 2 * 256 * size_of::<u16>();
// E4M3 code rows halve each K/V slot while retaining the same eight-position lookahead.
pub(crate) const DECODE_RING_E4M3_SHARED_BYTES: usize = DECODE_RING_DEPTH * 2 * 256;
const _: () = assert!(DECODE_RING_DEPTH.is_power_of_two());
const _: () = assert!(DECODE_RING_SHARED_BYTES == 8_192);
const _: () = assert!(DECODE_RING_E4M3_SHARED_BYTES == 4_096);

pub(crate) const LONG_CONTEXT_PARTITION_SIZE: usize = 256;
pub(crate) const LONG_CONTEXT_MAX_TOKENS: usize = 220_000;
pub(crate) const LONG_CONTEXT_MAX_PARTITIONS: usize =
    LONG_CONTEXT_MAX_TOKENS.div_ceil(LONG_CONTEXT_PARTITION_SIZE);

#[inline(always)]
unsafe fn load_e4m3x8(source: *const u8, scale: f32) -> [f32; VALUES_PER_LANE] {
    let source = source.cast::<u16>();
    let (x0, x1) = e4m3x2_to_f32(unsafe { *source });
    let (x2, x3) = e4m3x2_to_f32(unsafe { *source.add(1) });
    let (x4, x5) = e4m3x2_to_f32(unsafe { *source.add(2) });
    let (x6, x7) = e4m3x2_to_f32(unsafe { *source.add(3) });

    [
        x0 * scale,
        x1 * scale,
        x2 * scale,
        x3 * scale,
        x4 * scale,
        x5 * scale,
        x6 * scale,
        x7 * scale,
    ]
}

#[inline(always)]
unsafe fn load_bf16x8(source: *const u16) -> [f32; VALUES_PER_LANE] {
    unsafe {
        [
            f32::from_bits((*source as u32) << 16),
            f32::from_bits((*source.add(1) as u32) << 16),
            f32::from_bits((*source.add(2) as u32) << 16),
            f32::from_bits((*source.add(3) as u32) << 16),
            f32::from_bits((*source.add(4) as u32) << 16),
            f32::from_bits((*source.add(5) as u32) << 16),
            f32::from_bits((*source.add(6) as u32) << 16),
            f32::from_bits((*source.add(7) as u32) << 16),
        ]
    }
}

#[inline(always)]
fn fast_exp(value: f32) -> f32 {
    float::ex2_approx_f32(value * core::f32::consts::LOG2_E)
}

#[inline(always)]
fn warp_max(mut value: f32) -> f32 {
    value = value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 16));
    value = value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 8));
    value = value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 4));
    value = value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 2));
    value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 1))
}

#[inline(always)]
fn quad_sum(mut value: f32) -> f32 {
    value += warp::shuffle_xor_f32_sync(0xffff_ffff, value, 2);
    value + warp::shuffle_xor_f32_sync(0xffff_ffff, value, 1)
}

#[inline(always)]
fn quad_max(mut value: f32) -> f32 {
    value = value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 2));
    value.max(warp::shuffle_xor_f32_sync(0xffff_ffff, value, 1))
}

#[inline(always)]
fn flash_swizzle(row: usize, column: usize) -> usize {
    (((column >> 3) ^ (row & 7)) << 3) | (column & 7)
}

#[inline(always)]
fn flash_p_swizzle<const KEY_TILE: usize>(row: usize, column: usize) -> usize {
    if KEY_TILE == FLASH_PREFILL_P16_KEY_TILE {
        (((column >> 3) ^ (row & 3)) << 3) | (column & 7)
    } else {
        flash_swizzle(row, column)
    }
}

#[inline(always)]
fn flash_qk_word(row: usize, logical_word: usize) -> usize {
    (((logical_word >> 2) ^ (row & 7)) << 2) | (logical_word & 3)
}

#[inline(always)]
fn scale_flash_fragment(fragment: [f32; 4], alpha0: f32, alpha1: f32) -> [f32; 4] {
    [
        fragment[0] * alpha0,
        fragment[1] * alpha0,
        fragment[2] * alpha1,
        fragment[3] * alpha1,
    ]
}

#[inline(always)]
unsafe fn store_flash_fragment(
    partials: *mut f32,
    base0: usize,
    base1: usize,
    dimension: usize,
    fragment: [f32; 4],
) {
    unsafe {
        *partials.add(base0 + 2 + dimension) = fragment[0];
        *partials.add(base0 + 2 + dimension + 1) = fragment[1];
        *partials.add(base1 + 2 + dimension) = fragment[2];
        *partials.add(base1 + 2 + dimension + 1) = fragment[3];
    }
}

#[inline(always)]
unsafe fn load_flash_v_fragment<A: Arch>(
    v_shared: *const u16,
    row: usize,
    output_tile: usize,
) -> [u32; 2] {
    let column = output_tile * 8;
    let address =
        unsafe { v_shared.add(row * A::HEAD_DIM + flash_swizzle(row, column)) }.cast::<u32>();
    unsafe { wmma::ldmatrix_x2_trans(address) }
}

#[inline(always)]
unsafe fn load_flash_k_fragment<A: Arch>(
    k_words: *const u32,
    key_base: usize,
    k_offset: usize,
    lane_group: usize,
    lane_in_group: usize,
) -> [u32; 2] {
    let row = key_base + lane_group;
    unsafe {
        [
            *k_words.add(
                row * (A::HEAD_DIM / size_of::<u32>())
                    + flash_qk_word(row, k_offset + lane_in_group),
            ),
            *k_words.add(
                row * (A::HEAD_DIM / size_of::<u32>())
                    + flash_qk_word(row, k_offset + lane_in_group + 4),
            ),
        ]
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scale_flash_scores(
    score: [f32; 4],
    key_base: usize,
    tile_position: usize,
    lane_in_group: usize,
    length0: usize,
    length1: usize,
    partition_end: usize,
    scale0: f32,
    scale1: f32,
) -> [f32; 4] {
    let key0 = tile_position + key_base + lane_in_group * 2;
    let key1 = key0 + 1;
    [
        if key0 < length0 && key0 < partition_end {
            score[0] * scale0
        } else {
            -1.0e30
        },
        if key1 < length0 && key1 < partition_end {
            score[1] * scale0
        } else {
            -1.0e30
        },
        if key0 < length1 && key0 < partition_end {
            score[2] * scale1
        } else {
            -1.0e30
        },
        if key1 < length1 && key1 < partition_end {
            score[3] * scale1
        } else {
            -1.0e30
        },
    ]
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn write_flash_probabilities<const KEY_TILE: usize>(
    p_shared: *mut u16,
    score: [f32; 4],
    key_base: usize,
    tile_position: usize,
    lane_in_group: usize,
    row0: usize,
    row1: usize,
    length0: usize,
    length1: usize,
    partition_end: usize,
    maximum0: f32,
    maximum1: f32,
) -> (f32, f32) {
    let key0 = tile_position + key_base + lane_in_group * 2;
    let key1 = key0 + 1;
    let p00 = if key0 < length0 && key0 < partition_end {
        fast_exp(score[0] - maximum0)
    } else {
        0.0
    };
    let p01 = if key1 < length0 && key1 < partition_end {
        fast_exp(score[1] - maximum0)
    } else {
        0.0
    };
    let p10 = if key0 < length1 && key0 < partition_end {
        fast_exp(score[2] - maximum1)
    } else {
        0.0
    };
    let p11 = if key1 < length1 && key1 < partition_end {
        fast_exp(score[3] - maximum1)
    } else {
        0.0
    };
    let column = key_base + lane_in_group * 2;
    unsafe {
        *p_shared
            .add(row0 * KEY_TILE + flash_p_swizzle::<KEY_TILE>(row0, column))
            .cast::<u32>() = convert::cvt_f16x2_f32(p00, p01);
        *p_shared
            .add(row1 * KEY_TILE + flash_p_swizzle::<KEY_TILE>(row1, column))
            .cast::<u32>() = convert::cvt_f16x2_f32(p10, p11);
    }
    (p00 + p01, p10 + p11)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn flash_tile_cp_async<A: Arch, const KEY_TILE: usize>(
    k_words: *mut u32,
    v_codes: *mut u8,
    key_pages: *const u8,
    value_pages: *const u8,
    block_table: *const u32,
    kv_head: usize,
    tile_position: usize,
    tile_positions: usize,
    tid: usize,
) {
    let physical_page = unsafe { *block_table.add(tile_position / PAGE_SIZE) as usize };
    let mut task = tid;
    while task < KEY_TILE * (A::HEAD_DIM / 16) {
        let key_in_tile = task / (A::HEAD_DIM / 16);
        let dimension_segment = task - key_in_tile * (A::HEAD_DIM / 16);
        let valid = key_in_tile < tile_positions;
        let position = tile_position + key_in_tile;
        let source_offset = A::HEAD_DIM
            * ((position & (PAGE_SIZE - 1))
                + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension_segment * 16;
        let physical_segment = dimension_segment ^ (key_in_tile & 7);
        unsafe {
            cp_async_cg_zfill_16(
                k_words.add(key_in_tile * (A::HEAD_DIM / 4) + physical_segment * 4),
                key_pages.add(source_offset),
                if valid { 16 } else { 0 },
            );
            cp_async_cg_zfill_16(
                v_codes
                    .add(key_in_tile * A::HEAD_DIM + dimension_segment * 16)
                    .cast::<u32>(),
                value_pages.add(source_offset),
                if valid { 16 } else { 0 },
            );
        }
        task += FLASH_PREFILL_THREADS;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn paged_gqa_prefill_flash_partitioned<A: Arch, const KEY_TILE: usize>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    partitions: u32,
    partials: *mut f32,
    key_scale: f32,
    value_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let partitions = partitions as usize;
    let partition = block % partitions;
    let group = block / partitions;
    let query_head = group % A::NUM_ATTENTION_HEADS;
    let query_block = group / A::NUM_ATTENTION_HEADS;
    let first_token = query_block * FLASH_PREFILL_QUERY_ROWS;
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let query_group = warp_index / FLASH_PREFILL_WARPS_PER_GROUP;
    let output_warp = warp_index - query_group * FLASH_PREFILL_WARPS_PER_GROUP;
    let lane = tid & (WARP_THREADS - 1);
    let lane_group = lane >> 2;
    let lane_in_group = lane & 3;
    let kv_head = query_head / PREFILL_QUERY_WARPS;
    let group_length = unsafe { *lengths.add(first_token + FLASH_PREFILL_QUERY_ROWS - 1) as usize };
    let key_tiles = group_length.div_ceil(KEY_TILE);
    let tiles_per_partition = key_tiles.div_ceil(partitions);
    let partition_tile_begin = partition * tiles_per_partition;
    let partition_tile_end = (partition_tile_begin + tiles_per_partition).min(key_tiles);
    let partition_begin = partition_tile_begin * KEY_TILE;
    let partition_end = (partition_tile_end * KEY_TILE).min(group_length);

    if partition_begin >= partition_end {
        let mut task = tid;
        while task < FLASH_PREFILL_QUERY_ROWS * PREFILL_PARTIAL_VALUES {
            let row = task / PREFILL_PARTIAL_VALUES;
            let field = task - row * PREFILL_PARTIAL_VALUES;
            let token = first_token + row;
            let base = ((token * A::NUM_ATTENTION_HEADS + query_head) * partitions + partition)
                * PREFILL_PARTIAL_VALUES;
            unsafe {
                *partials.add(base + field) = if field == 0 { -1.0e30 } else { 0.0 };
            }
            task += FLASH_PREFILL_THREADS;
        }
        return;
    }

    let shared = DynamicSharedArray::<u32, 16>::get().cast::<u8>();
    let q_shared = shared;
    let k_base = unsafe { q_shared.add(FLASH_PREFILL_Q_BYTES) };
    let v_codes_base = unsafe { k_base.add(KEY_TILE * A::HEAD_DIM) };
    let v_shared_base = unsafe { v_codes_base.add(KEY_TILE * A::HEAD_DIM) }.cast::<u16>();
    let p_shared = unsafe {
        v_shared_base
            .cast::<u8>()
            .add(KEY_TILE * A::HEAD_DIM * size_of::<u16>())
    }
    .cast::<u16>();
    let q_scales = unsafe {
        p_shared
            .cast::<u8>()
            .add(FLASH_PREFILL_QUERY_ROWS * KEY_TILE * size_of::<u16>())
    }
    .cast::<f32>();
    let stats = unsafe {
        q_scales
            .cast::<u8>()
            .add(FLASH_PREFILL_Q_SCALE_BYTES)
            .cast::<f32>()
    };

    // Eight warps quantize 32 rows once per CTA. Each lane owns eight
    // contiguous dimensions; the row scale is reused by all eight QK steps.
    let mut row = warp_index;
    while row < FLASH_PREFILL_QUERY_ROWS {
        let token = first_token + row;
        let dimension = lane * VALUES_PER_LANE;
        let source = unsafe {
            query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
        };
        let q0 = unsafe { *source };
        let q1 = unsafe { *source.add(1) };
        let q2 = unsafe { *source.add(2) };
        let q3 = unsafe { *source.add(3) };
        let q4 = unsafe { *source.add(4) };
        let q5 = unsafe { *source.add(5) };
        let q6 = unsafe { *source.add(6) };
        let q7 = unsafe { *source.add(7) };
        let mut absolute_maximum = q0
            .abs()
            .max(q1.abs())
            .max(q2.abs())
            .max(q3.abs())
            .max(q4.abs())
            .max(q5.abs())
            .max(q6.abs())
            .max(q7.abs());
        absolute_maximum = warp_max(absolute_maximum);
        let scale = if absolute_maximum > 0.0 {
            absolute_maximum / 448.0
        } else {
            1.0
        };
        if lane == 0 {
            unsafe { *q_scales.add(row) = scale };
        }
        let inverse = 1.0 / scale;
        let logical_word = lane * 2;
        let destination = unsafe {
            q_shared.add(
                row * A::HEAD_DIM + flash_qk_word(row, logical_word) * core::mem::size_of::<u32>(),
            )
        }
        .cast::<u32>();
        let packed0 = convert::cvt_rn_satfinite_e4m3x2_f32(q0 * inverse, q1 * inverse);
        let packed1 = convert::cvt_rn_satfinite_e4m3x2_f32(q2 * inverse, q3 * inverse);
        let packed2 = convert::cvt_rn_satfinite_e4m3x2_f32(q4 * inverse, q5 * inverse);
        let packed3 = convert::cvt_rn_satfinite_e4m3x2_f32(q6 * inverse, q7 * inverse);
        unsafe {
            *destination = u32::from(packed0) | (u32::from(packed1) << 16);
            *destination.add(1) = u32::from(packed2) | (u32::from(packed3) << 16);
        }
        row += FLASH_PREFILL_THREADS / WARP_THREADS;
    }
    thread::sync_threads();

    let table_row = unsafe { *table_rows.add(first_token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let q_words = q_shared.cast::<u32>();
    let value_scale_pair = convert::cvt_f16x2_f32(value_scale, value_scale);
    let mut pv0 = [0.0f32; 4];
    let mut pv1 = [0.0f32; 4];
    let mut pv2 = [0.0f32; 4];
    let mut pv3 = [0.0f32; 4];
    let mut pv4 = [0.0f32; 4];
    let mut pv5 = [0.0f32; 4];
    let mut pv6 = [0.0f32; 4];
    let mut pv7 = [0.0f32; 4];
    let mut running_maximum0 = -1.0e30f32;
    let mut running_maximum1 = -1.0e30f32;
    let mut running_denominator0 = 0.0f32;
    let mut running_denominator1 = 0.0f32;

    let a_matrix = lane >> 3;
    let a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    let a_column_offset = (a_matrix >> 1) << 3;
    let b_row_offset = lane & 7;
    let b_k_offset = ((lane >> 3) & 1) << 3;

    let mut tile_position = partition_begin;
    while tile_position < partition_end {
        unsafe {
            flash_tile_cp_async::<A, KEY_TILE>(
                k_base.cast::<u32>(),
                v_codes_base,
                key_pages,
                value_pages,
                block_table,
                kv_head,
                tile_position,
                (partition_end - tile_position).min(KEY_TILE),
                tid,
            );
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();
        let k_words = k_base.cast::<u32>();
        let v_codes = v_codes_base;
        let v_shared = v_shared_base;

        if output_warp == 0 {
            let row0 = query_group * FLASH_PREFILL_MMA_ROWS + lane_group;
            let row1 = row0 + 8;
            let length0 = unsafe { *lengths.add(first_token + row0) as usize };
            let length1 = unsafe { *lengths.add(first_token + row1) as usize };
            let score_scale0 = unsafe { *q_scales.add(row0) } * key_scale * 0.0625;
            let score_scale1 = unsafe { *q_scales.add(row1) } * key_scale * 0.0625;
            let mut score0 = [0.0f32; 4];
            let mut score1 = [0.0f32; 4];
            let mut score2 = [0.0f32; 4];
            let mut score3 = [0.0f32; 4];
            let mut score4 = [0.0f32; 4];
            let mut score5 = [0.0f32; 4];
            let mut score6 = [0.0f32; 4];
            let mut score7 = [0.0f32; 4];
            let mut k_subtile = 0usize;
            while k_subtile < A::HEAD_DIM / 32 {
                let k_offset = k_subtile * 8;
                let activation_fragment = unsafe {
                    [
                        *q_words.add(
                            row0 * (A::HEAD_DIM / 4)
                                + flash_qk_word(row0, k_offset + lane_in_group),
                        ),
                        *q_words.add(
                            row1 * (A::HEAD_DIM / 4)
                                + flash_qk_word(row1, k_offset + lane_in_group),
                        ),
                        *q_words.add(
                            row0 * (A::HEAD_DIM / 4)
                                + flash_qk_word(row0, k_offset + lane_in_group + 4),
                        ),
                        *q_words.add(
                            row1 * (A::HEAD_DIM / 4)
                                + flash_qk_word(row1, k_offset + lane_in_group + 4),
                        ),
                    ]
                };
                unsafe {
                    score0 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                        score0,
                        activation_fragment,
                        load_flash_k_fragment::<A>(k_words, 0, k_offset, lane_group, lane_in_group),
                    );
                    score1 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                        score1,
                        activation_fragment,
                        load_flash_k_fragment::<A>(k_words, 8, k_offset, lane_group, lane_in_group),
                    );
                    score2 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                        score2,
                        activation_fragment,
                        load_flash_k_fragment::<A>(
                            k_words,
                            16,
                            k_offset,
                            lane_group,
                            lane_in_group,
                        ),
                    );
                    score3 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                        score3,
                        activation_fragment,
                        load_flash_k_fragment::<A>(
                            k_words,
                            24,
                            k_offset,
                            lane_group,
                            lane_in_group,
                        ),
                    );
                    if KEY_TILE == PREFILL_KEY_TILE {
                        score4 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                            score4,
                            activation_fragment,
                            load_flash_k_fragment::<A>(
                                k_words,
                                32,
                                k_offset,
                                lane_group,
                                lane_in_group,
                            ),
                        );
                        score5 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                            score5,
                            activation_fragment,
                            load_flash_k_fragment::<A>(
                                k_words,
                                40,
                                k_offset,
                                lane_group,
                                lane_in_group,
                            ),
                        );
                        score6 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                            score6,
                            activation_fragment,
                            load_flash_k_fragment::<A>(
                                k_words,
                                48,
                                k_offset,
                                lane_group,
                                lane_in_group,
                            ),
                        );
                        score7 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                            score7,
                            activation_fragment,
                            load_flash_k_fragment::<A>(
                                k_words,
                                56,
                                k_offset,
                                lane_group,
                                lane_in_group,
                            ),
                        );
                    }
                }
                k_subtile += 1;
            }

            score0 = scale_flash_scores(
                score0,
                0,
                tile_position,
                lane_in_group,
                length0,
                length1,
                partition_end,
                score_scale0,
                score_scale1,
            );
            score1 = scale_flash_scores(
                score1,
                8,
                tile_position,
                lane_in_group,
                length0,
                length1,
                partition_end,
                score_scale0,
                score_scale1,
            );
            score2 = scale_flash_scores(
                score2,
                16,
                tile_position,
                lane_in_group,
                length0,
                length1,
                partition_end,
                score_scale0,
                score_scale1,
            );
            score3 = scale_flash_scores(
                score3,
                24,
                tile_position,
                lane_in_group,
                length0,
                length1,
                partition_end,
                score_scale0,
                score_scale1,
            );
            if KEY_TILE == PREFILL_KEY_TILE {
                score4 = scale_flash_scores(
                    score4,
                    32,
                    tile_position,
                    lane_in_group,
                    length0,
                    length1,
                    partition_end,
                    score_scale0,
                    score_scale1,
                );
                score5 = scale_flash_scores(
                    score5,
                    40,
                    tile_position,
                    lane_in_group,
                    length0,
                    length1,
                    partition_end,
                    score_scale0,
                    score_scale1,
                );
                score6 = scale_flash_scores(
                    score6,
                    48,
                    tile_position,
                    lane_in_group,
                    length0,
                    length1,
                    partition_end,
                    score_scale0,
                    score_scale1,
                );
                score7 = scale_flash_scores(
                    score7,
                    56,
                    tile_position,
                    lane_in_group,
                    length0,
                    length1,
                    partition_end,
                    score_scale0,
                    score_scale1,
                );
            }

            let mut block_maximum0 = score0[0]
                .max(score0[1])
                .max(score1[0])
                .max(score1[1])
                .max(score2[0])
                .max(score2[1])
                .max(score3[0])
                .max(score3[1]);
            let mut block_maximum1 = score0[2]
                .max(score0[3])
                .max(score1[2])
                .max(score1[3])
                .max(score2[2])
                .max(score2[3])
                .max(score3[2])
                .max(score3[3]);
            if KEY_TILE == PREFILL_KEY_TILE {
                block_maximum0 = block_maximum0
                    .max(score4[0])
                    .max(score4[1])
                    .max(score5[0])
                    .max(score5[1])
                    .max(score6[0])
                    .max(score6[1])
                    .max(score7[0])
                    .max(score7[1]);
                block_maximum1 = block_maximum1
                    .max(score4[2])
                    .max(score4[3])
                    .max(score5[2])
                    .max(score5[3])
                    .max(score6[2])
                    .max(score6[3])
                    .max(score7[2])
                    .max(score7[3]);
            }
            block_maximum0 = quad_max(block_maximum0);
            block_maximum1 = quad_max(block_maximum1);
            let next_maximum0 = running_maximum0.max(block_maximum0);
            let next_maximum1 = running_maximum1.max(block_maximum1);
            let alpha0 = if running_denominator0 > 0.0 {
                fast_exp(running_maximum0 - next_maximum0)
            } else {
                0.0
            };
            let alpha1 = if running_denominator1 > 0.0 {
                fast_exp(running_maximum1 - next_maximum1)
            } else {
                0.0
            };

            let p0 = unsafe {
                write_flash_probabilities::<KEY_TILE>(
                    p_shared,
                    score0,
                    0,
                    tile_position,
                    lane_in_group,
                    row0,
                    row1,
                    length0,
                    length1,
                    partition_end,
                    next_maximum0,
                    next_maximum1,
                )
            };
            let p1 = unsafe {
                write_flash_probabilities::<KEY_TILE>(
                    p_shared,
                    score1,
                    8,
                    tile_position,
                    lane_in_group,
                    row0,
                    row1,
                    length0,
                    length1,
                    partition_end,
                    next_maximum0,
                    next_maximum1,
                )
            };
            let p2 = unsafe {
                write_flash_probabilities::<KEY_TILE>(
                    p_shared,
                    score2,
                    16,
                    tile_position,
                    lane_in_group,
                    row0,
                    row1,
                    length0,
                    length1,
                    partition_end,
                    next_maximum0,
                    next_maximum1,
                )
            };
            let p3 = unsafe {
                write_flash_probabilities::<KEY_TILE>(
                    p_shared,
                    score3,
                    24,
                    tile_position,
                    lane_in_group,
                    row0,
                    row1,
                    length0,
                    length1,
                    partition_end,
                    next_maximum0,
                    next_maximum1,
                )
            };
            let mut block_denominator0 = p0.0 + p1.0 + p2.0 + p3.0;
            let mut block_denominator1 = p0.1 + p1.1 + p2.1 + p3.1;
            if KEY_TILE == PREFILL_KEY_TILE {
                let p4 = unsafe {
                    write_flash_probabilities::<KEY_TILE>(
                        p_shared,
                        score4,
                        32,
                        tile_position,
                        lane_in_group,
                        row0,
                        row1,
                        length0,
                        length1,
                        partition_end,
                        next_maximum0,
                        next_maximum1,
                    )
                };
                let p5 = unsafe {
                    write_flash_probabilities::<KEY_TILE>(
                        p_shared,
                        score5,
                        40,
                        tile_position,
                        lane_in_group,
                        row0,
                        row1,
                        length0,
                        length1,
                        partition_end,
                        next_maximum0,
                        next_maximum1,
                    )
                };
                let p6 = unsafe {
                    write_flash_probabilities::<KEY_TILE>(
                        p_shared,
                        score6,
                        48,
                        tile_position,
                        lane_in_group,
                        row0,
                        row1,
                        length0,
                        length1,
                        partition_end,
                        next_maximum0,
                        next_maximum1,
                    )
                };
                let p7 = unsafe {
                    write_flash_probabilities::<KEY_TILE>(
                        p_shared,
                        score7,
                        56,
                        tile_position,
                        lane_in_group,
                        row0,
                        row1,
                        length0,
                        length1,
                        partition_end,
                        next_maximum0,
                        next_maximum1,
                    )
                };
                block_denominator0 += p4.0 + p5.0 + p6.0 + p7.0;
                block_denominator1 += p4.1 + p5.1 + p6.1 + p7.1;
            }
            block_denominator0 = quad_sum(block_denominator0);
            block_denominator1 = quad_sum(block_denominator1);
            running_denominator0 = running_denominator0 * alpha0 + block_denominator0;
            running_denominator1 = running_denominator1 * alpha1 + block_denominator1;
            running_maximum0 = next_maximum0;
            running_maximum1 = next_maximum1;
            if lane_in_group == 0 {
                unsafe {
                    *stats.add(row0) = running_maximum0;
                    *stats.add(row1) = running_maximum1;
                    *stats.add(FLASH_PREFILL_QUERY_ROWS + row0) = running_denominator0;
                    *stats.add(FLASH_PREFILL_QUERY_ROWS + row1) = running_denominator1;
                    *stats.add(2 * FLASH_PREFILL_QUERY_ROWS + row0) = alpha0;
                    *stats.add(2 * FLASH_PREFILL_QUERY_ROWS + row1) = alpha1;
                }
            }
        } else {
            // One warp per query group produces QK; the other six warps
            // convert disjoint V rows while the score producer runs.
            let worker_warp = query_group * (FLASH_PREFILL_WARPS_PER_GROUP - 1) + output_warp - 1;
            let mut worker_task = worker_warp * WARP_THREADS + lane;
            let worker_threads =
                FLASH_PREFILL_QUERY_GROUPS * (FLASH_PREFILL_WARPS_PER_GROUP - 1) * WARP_THREADS;
            while worker_task < KEY_TILE * (A::HEAD_DIM / 8) {
                let key_in_tile = worker_task / (A::HEAD_DIM / 8);
                let dimension_chunk = worker_task - key_in_tile * (A::HEAD_DIM / 8);
                let dimension = dimension_chunk * 8;
                let packed = unsafe {
                    *v_codes
                        .add(key_in_tile * A::HEAD_DIM + dimension)
                        .cast::<u64>()
                };
                let destination = unsafe {
                    v_shared.add(key_in_tile * A::HEAD_DIM + flash_swizzle(key_in_tile, dimension))
                }
                .cast::<u32>();
                unsafe {
                    *destination = f16x2::mul_f16x2(
                        convert::cvt_rn_f16x2_e4m3x2(packed as u16),
                        value_scale_pair,
                    );
                    *destination.add(1) = f16x2::mul_f16x2(
                        convert::cvt_rn_f16x2_e4m3x2((packed >> 16) as u16),
                        value_scale_pair,
                    );
                    *destination.add(2) = f16x2::mul_f16x2(
                        convert::cvt_rn_f16x2_e4m3x2((packed >> 32) as u16),
                        value_scale_pair,
                    );
                    *destination.add(3) = f16x2::mul_f16x2(
                        convert::cvt_rn_f16x2_e4m3x2((packed >> 48) as u16),
                        value_scale_pair,
                    );
                }
                worker_task += worker_threads;
            }
        }
        thread::sync_threads();

        let row0 = query_group * FLASH_PREFILL_MMA_ROWS + lane_group;
        let row1 = row0 + 8;
        let alpha0 = unsafe { *stats.add(2 * FLASH_PREFILL_QUERY_ROWS + row0) };
        let alpha1 = unsafe { *stats.add(2 * FLASH_PREFILL_QUERY_ROWS + row1) };
        pv0 = scale_flash_fragment(pv0, alpha0, alpha1);
        pv1 = scale_flash_fragment(pv1, alpha0, alpha1);
        pv2 = scale_flash_fragment(pv2, alpha0, alpha1);
        pv3 = scale_flash_fragment(pv3, alpha0, alpha1);
        pv4 = scale_flash_fragment(pv4, alpha0, alpha1);
        pv5 = scale_flash_fragment(pv5, alpha0, alpha1);
        pv6 = scale_flash_fragment(pv6, alpha0, alpha1);
        pv7 = scale_flash_fragment(pv7, alpha0, alpha1);

        let mut key_subtile = 0usize;
        while key_subtile < KEY_TILE / 16 {
            let p_column = key_subtile * 16 + a_column_offset;
            let p_row = query_group * FLASH_PREFILL_MMA_ROWS + a_row_offset;
            let p_address = unsafe {
                p_shared.add(p_row * KEY_TILE + flash_p_swizzle::<KEY_TILE>(p_row, p_column))
            }
            .cast::<u32>();
            let p_fragment = unsafe { wmma::ldmatrix_x4(p_address) };
            let v_row = key_subtile * 16 + b_k_offset + b_row_offset;
            let output_base = output_warp * 8;
            pv0 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv0,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base),
                )
            };
            pv1 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv1,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 1),
                )
            };
            pv2 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv2,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 2),
                )
            };
            pv3 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv3,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 3),
                )
            };
            pv4 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv4,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 4),
                )
            };
            pv5 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv5,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 5),
                )
            };
            pv6 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv6,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 6),
                )
            };
            pv7 = unsafe {
                wmma::mma_m16n8k16_f32_f16(
                    pv7,
                    p_fragment,
                    load_flash_v_fragment::<A>(v_shared, v_row, output_base + 7),
                )
            };
            key_subtile += 1;
        }
        tile_position += KEY_TILE;
    }

    let row0 = query_group * FLASH_PREFILL_MMA_ROWS + lane_group;
    let row1 = row0 + 8;
    let token0 = first_token + row0;
    let token1 = first_token + row1;
    let base0 = ((token0 * A::NUM_ATTENTION_HEADS + query_head) * partitions + partition)
        * PREFILL_PARTIAL_VALUES;
    let base1 = ((token1 * A::NUM_ATTENTION_HEADS + query_head) * partitions + partition)
        * PREFILL_PARTIAL_VALUES;
    if output_warp == 0 && lane_in_group == 0 {
        unsafe {
            *partials.add(base0) = *stats.add(row0);
            *partials.add(base0 + 1) = *stats.add(FLASH_PREFILL_QUERY_ROWS + row0);
            *partials.add(base1) = *stats.add(row1);
            *partials.add(base1 + 1) = *stats.add(FLASH_PREFILL_QUERY_ROWS + row1);
        }
    }
    let output_base = output_warp * 8;
    unsafe {
        store_flash_fragment(
            partials,
            base0,
            base1,
            output_base * 8 + lane_in_group * 2,
            pv0,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 1) * 8 + lane_in_group * 2,
            pv1,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 2) * 8 + lane_in_group * 2,
            pv2,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 3) * 8 + lane_in_group * 2,
            pv3,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 4) * 8 + lane_in_group * 2,
            pv4,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 5) * 8 + lane_in_group * 2,
            pv5,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 6) * 8 + lane_in_group * 2,
            pv6,
        );
        store_flash_fragment(
            partials,
            base0,
            base1,
            (output_base + 7) * 8 + lane_in_group * 2,
            pv7,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn paged_gqa_prefill_shared<
    A: Arch,
    const TOKENS: usize,
    const TOKEN_GROUP: usize,
    const QUERY_WARPS: usize,
>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
    key_scale: f32,
    value_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token_group = block / A::NUM_KV_HEADS;
    let kv_head = block - token_group * A::NUM_KV_HEADS;
    let first_token = token_group * TOKEN_GROUP;
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let token_in_group = warp_index / QUERY_WARPS;
    let query_in_group = warp_index - token_in_group * QUERY_WARPS;
    let token = first_token + token_in_group;
    let query_head = kv_head * QUERY_WARPS + query_in_group;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let length0 = unsafe { *lengths.add(first_token) as usize };
    let length1 = if TOKEN_GROUP == 2 {
        unsafe { *lengths.add(first_token + 1) as usize }
    } else {
        length0
    };
    let group_length = length0.max(length1);
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let shared = DynamicSharedArray::<u32, 16>::get();
    let shared_bytes = shared.cast::<u8>();
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut tile_position = 0usize;

    while tile_position < group_length {
        let mut task = tid;
        while task < 2 * PREFILL_KEY_TILE * (A::HEAD_DIM / 16) {
            let plane = task / (PREFILL_KEY_TILE * (A::HEAD_DIM / 16));
            let within_plane = task - plane * PREFILL_KEY_TILE * (A::HEAD_DIM / 16);
            let position_in_tile = within_plane / (A::HEAD_DIM / 16);
            let dimension_segment = within_plane - position_in_tile * (A::HEAD_DIM / 16);
            let position = tile_position + position_in_tile;
            let valid = position < group_length;
            let table_row = unsafe { *table_rows.add(first_token) as usize };
            let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
            let physical_page = if valid {
                unsafe { *block_table.add(position / PAGE_SIZE) as usize }
            } else {
                0
            };
            let cache_element = A::HEAD_DIM
                * ((position & (PAGE_SIZE - 1))
                    + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
                + dimension_segment * 16;
            let source = if plane == 0 { key_pages } else { value_pages };
            let destination_word = plane * PREFILL_PLANE_WORDS
                + position_in_tile * (A::HEAD_DIM / size_of::<u32>())
                + dimension_segment * (16 / size_of::<u32>());
            unsafe {
                cp_async_cg_zfill_16(
                    shared.add(destination_word),
                    source.add(cache_element),
                    if valid { 16 } else { 0 },
                );
            }
            task += WARP_THREADS * TOKEN_GROUP * QUERY_WARPS;
        }
        unsafe {
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();

        let token_length = if token_in_group == 0 {
            length0
        } else {
            length1
        };
        let tile_end = core::cmp::min(tile_position + PREFILL_KEY_TILE, token_length);
        let mut position = tile_position;
        while position < tile_end {
            let tile_element = (position - tile_position) * A::HEAD_DIM + dimension;
            let key = unsafe { load_e4m3x8(shared_bytes.add(tile_element), key_scale) };
            let value = unsafe {
                load_e4m3x8(
                    shared_bytes.add(PREFILL_PLANE_WORDS * size_of::<u32>() + tile_element),
                    value_scale,
                )
            };
            let mut score = 0.0f32;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                score = float::fma_rn_f32(q[element], key[element], score);
                element += 1;
            }
            score = warp::reduce_sum_f32(score) * 0.0625;

            if score > maximum {
                let old_scale = fast_exp(maximum - score);
                denominator = denominator * old_scale + 1.0;
                maximum = score;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                    element += 1;
                }
            } else {
                let weight = fast_exp(score - maximum);
                denominator += weight;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(weight, value[element], accumulator[element]);
                    element += 1;
                }
            }
            position += 1;
        }
        thread::sync_threads();
        tile_position += PREFILL_KEY_TILE;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn bf16_paged_gqa_prefill_shared<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u16,
    value_pages: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_KV_HEADS;
    if token >= TOKENS {
        return;
    }
    let kv_head = block - token * A::NUM_KV_HEADS;
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let query_heads_per_kv = A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS;
    let query_head = kv_head * query_heads_per_kv + warp_index;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let length = unsafe { *lengths.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let shared = DynamicSharedArray::<u32, 16>::get();
    let shared_values = shared.cast::<u16>();
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut tile_position = 0usize;

    while tile_position < length {
        let mut task = tid;
        while task < 2 * PREFILL_KEY_TILE * (A::HEAD_DIM / 8) {
            let plane = task / (PREFILL_KEY_TILE * (A::HEAD_DIM / 8));
            let within_plane = task - plane * PREFILL_KEY_TILE * (A::HEAD_DIM / 8);
            let position_in_tile = within_plane / (A::HEAD_DIM / 8);
            let dimension_segment = within_plane - position_in_tile * (A::HEAD_DIM / 8);
            let position = tile_position + position_in_tile;
            let valid = position < length;
            let physical_page = if valid {
                unsafe { *block_table.add(position / PAGE_SIZE) as usize }
            } else {
                0
            };
            let cache_element = A::HEAD_DIM
                * ((position & (PAGE_SIZE - 1))
                    + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
                + dimension_segment * 8;
            let source = if plane == 0 { key_pages } else { value_pages };
            let destination_word = plane * (BF16_PREFILL_PLANE_VALUES / 2)
                + position_in_tile * (A::HEAD_DIM / 2)
                + dimension_segment * 4;
            unsafe {
                cp_async_cg_zfill_16(
                    shared.add(destination_word),
                    source.add(cache_element).cast::<u8>(),
                    if valid { 16 } else { 0 },
                );
            }
            task += WARP_THREADS * query_heads_per_kv;
        }
        unsafe {
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();

        let tile_end = core::cmp::min(tile_position + PREFILL_KEY_TILE, length);
        let mut position = tile_position;
        while position < tile_end {
            let tile_element = (position - tile_position) * A::HEAD_DIM + dimension;
            let key = unsafe { load_bf16x8(shared_values.add(tile_element)) };
            let value =
                unsafe { load_bf16x8(shared_values.add(BF16_PREFILL_PLANE_VALUES + tile_element)) };
            let mut score = 0.0f32;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                score = float::fma_rn_f32(q[element], key[element], score);
                element += 1;
            }
            score = warp::reduce_sum_f32(score) * 0.0625;

            if score > maximum {
                let old_scale = fast_exp(maximum - score);
                denominator = denominator * old_scale + 1.0;
                maximum = score;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                    element += 1;
                }
            } else {
                let weight = fast_exp(score - maximum);
                denominator += weight;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(weight, value[element], accumulator[element]);
                    element += 1;
                }
            }
            position += 1;
        }
        thread::sync_threads();
        tile_position += PREFILL_KEY_TILE;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn paged_gqa_prefill_partitioned_reduce<
    A: Arch,
    const TOKENS: usize,
    const PARTITIONS: usize,
>(
    partials: *const f32,
    output: *mut f32,
) {
    let head_token = thread::blockIdx_x() as usize;
    if head_token >= TOKENS * A::NUM_ATTENTION_HEADS {
        return;
    }
    let lane = thread::threadIdx_x() as usize;
    let dimension = lane * VALUES_PER_LANE;
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut partition = 0usize;
    while partition < PARTITIONS {
        let base = (head_token * PARTITIONS + partition) * PREFILL_PARTIAL_VALUES;
        let partial_denominator = unsafe { *partials.add(base + 1) };
        if partial_denominator > 0.0 {
            let partial_maximum = unsafe { *partials.add(base) };
            let next_maximum = maximum.max(partial_maximum);
            let old_scale = fast_exp(maximum - next_maximum);
            let partial_scale = fast_exp(partial_maximum - next_maximum);
            denominator = denominator * old_scale + partial_denominator * partial_scale;
            maximum = next_maximum;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                accumulator[element] = float::fma_rn_f32(
                    unsafe { *partials.add(base + 2 + dimension + element) },
                    partial_scale,
                    accumulator[element] * old_scale,
                );
                element += 1;
            }
        }
        partition += 1;
    }
    let output = unsafe { output.add(head_token * A::HEAD_DIM + dimension) };
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn paged_gqa<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
    key_scale: f32,
    value_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = block - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let lane = thread::threadIdx_x() as usize;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let length = unsafe { *lengths.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;

    // Same async ring as the BF16 scan, adapted to one-byte codes: an E4M3
    // row is 256 bytes, so lanes 0-15 copy the K row and lanes 16-31 the V
    // row, one 16-byte chunk each. Lanes consume bytes other lanes copied,
    // so a warp barrier follows each wait (visibility) and each read window
    // (before the slot is overwritten). One (possibly empty) commit per
    // iteration keeps the wait depth a compile-time literal.
    let ring = DynamicSharedArray::<u32, 16>::get();
    let slot_words = 2 * A::HEAD_DIM / 4;
    let issue = |position: usize| {
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let row_element =
            A::HEAD_DIM * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page));
        let slot = (position & (DECODE_RING_DEPTH - 1)) * slot_words;
        unsafe {
            if lane < 16 {
                cp_async_cg_16(
                    ring.add(slot + lane * 4),
                    key_pages.add(row_element + lane * 16).cast::<u32>(),
                );
            } else {
                cp_async_cg_16(
                    ring.add(slot + A::HEAD_DIM / 4 + (lane - 16) * 4),
                    value_pages
                        .add(row_element + (lane - 16) * 16)
                        .cast::<u32>(),
                );
            }
        }
    };
    let mut ahead = 0usize;
    while ahead < DECODE_RING_DEPTH {
        if ahead < length {
            issue(ahead);
        }
        // SAFETY: the preceding copies form one device-side asynchronous group.
        unsafe { cp_async_commit_group() };
        ahead += 1;
    }
    let mut position = 0usize;

    while position < length {
        // SAFETY: eight groups were committed before the first wait and one
        // replacement group is committed after every consumed position.
        unsafe { cp_async_wait_group(DECODE_RING_DEPTH as u32 - 1) };
        warp::sync_mask(u32::MAX);
        let slot_bytes = unsafe {
            ring.add((position & (DECODE_RING_DEPTH - 1)) * slot_words)
                .cast::<u8>()
        };
        let key = unsafe { load_e4m3x8(slot_bytes.add(dimension), key_scale) };
        let value = unsafe { load_e4m3x8(slot_bytes.add(A::HEAD_DIM + dimension), value_scale) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        warp::sync_mask(u32::MAX);
        if position + DECODE_RING_DEPTH < length {
            issue(position + DECODE_RING_DEPTH);
        }
        // SAFETY: this closes the replacement group, including empty tail groups.
        unsafe { cp_async_commit_group() };
        position += 1;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn paged_gqa_partitioned<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
    key_scale: f32,
    value_scale: f32,
    partials: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = block - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let length = unsafe { *lengths.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    // Each warp owns one contiguous context slice and keeps the established
    // per-position online-softmax order inside its slice. Warp zero merges
    // slice states below in ascending order.
    let slice_positions = length.div_ceil(DECODE_WARPS);
    let slice_begin = warp_index * slice_positions;
    let slice_end = core::cmp::min(slice_begin + slice_positions, length);
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut position = slice_begin;

    while position < slice_end {
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let cache_element = A::HEAD_DIM
            * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension;
        let key = unsafe { load_e4m3x8(key_pages.add(cache_element), key_scale) };
        let value = unsafe { load_e4m3x8(value_pages.add(cache_element), value_scale) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        position += 1;
    }

    let partial_base = warp_index * DECODE_PARTIAL_VALUES;
    if lane == 0 {
        unsafe {
            *partials.add(partial_base) = maximum;
            *partials.add(partial_base + 1) = denominator;
        }
    }
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *partials.add(partial_base + 2 + dimension + element) = accumulator[element] };
        element += 1;
    }
    thread::sync_threads();
    if warp_index != 0 {
        return;
    }

    let mut merged = [0.0f32; VALUES_PER_LANE];
    let mut merged_maximum = -1.0e30f32;
    let mut merged_denominator = 0.0f32;
    let mut slice = 0usize;
    while slice < DECODE_WARPS {
        let base = slice * DECODE_PARTIAL_VALUES;
        let slice_denominator = unsafe { *partials.add(base + 1) };
        if slice_denominator > 0.0 {
            let slice_maximum = unsafe { *partials.add(base) };
            let next_maximum = merged_maximum.max(slice_maximum);
            let old_scale = fast_exp(merged_maximum - next_maximum);
            let slice_scale = fast_exp(slice_maximum - next_maximum);
            merged_denominator = merged_denominator * old_scale + slice_denominator * slice_scale;
            merged_maximum = next_maximum;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                merged[element] = float::fma_rn_f32(
                    unsafe { *partials.add(base + 2 + dimension + element) },
                    slice_scale,
                    merged[element] * old_scale,
                );
                element += 1;
            }
        }
        slice += 1;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = merged[element] / merged_denominator };
        element += 1;
    }
}

/// Runs one exact BF16 decode step over the paged cache.
///
/// # Safety
///
/// Every plane must address one complete row per launched block.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16_paged_gqa<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u16,
    value_pages: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = block - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let lane = thread::threadIdx_x() as usize;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let length = unsafe { *lengths.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;

    // Each lane async-copies its own 16-byte K and V slices eight positions
    // ahead into a shared ring, so the serial per-position chain no longer
    // stalls on a dependent global load. Every iteration commits exactly one
    // (possibly empty) group, keeping the wait depth a compile-time constant;
    // lanes only ever read the bytes they copied, so no barrier is needed.
    let ring = DynamicSharedArray::<u32, 16>::get();
    let slot_words = 2 * A::HEAD_DIM / 2;
    let lane_words = dimension / 2;
    let issue = |position: usize| {
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let cache_element = A::HEAD_DIM
            * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension;
        let slot = (position & (DECODE_RING_DEPTH - 1)) * slot_words;
        unsafe {
            cp_async_cg_16(
                ring.add(slot + lane_words),
                key_pages.add(cache_element).cast::<u32>(),
            );
            cp_async_cg_16(
                ring.add(slot + A::HEAD_DIM / 2 + lane_words),
                value_pages.add(cache_element).cast::<u32>(),
            );
        }
    };
    let mut ahead = 0usize;
    while ahead < DECODE_RING_DEPTH {
        if ahead < length {
            issue(ahead);
        }
        // SAFETY: the preceding copies form one device-side asynchronous group.
        unsafe { cp_async_commit_group() };
        ahead += 1;
    }
    let mut position = 0usize;

    while position < length {
        // SAFETY: eight groups were committed before the first wait and one
        // replacement group is committed after every consumed position.
        unsafe { cp_async_wait_group(DECODE_RING_DEPTH as u32 - 1) };
        let slot = (position & (DECODE_RING_DEPTH - 1)) * slot_words;
        let key = unsafe { load_bf16x8(ring.add(slot + lane_words).cast::<u16>()) };
        let value =
            unsafe { load_bf16x8(ring.add(slot + A::HEAD_DIM / 2 + lane_words).cast::<u16>()) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        if position + DECODE_RING_DEPTH < length {
            issue(position + DECODE_RING_DEPTH);
        }
        // SAFETY: this closes the replacement group, including empty tail groups.
        unsafe { cp_async_commit_group() };
        position += 1;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

/// Runs one exact partitioned BF16 decode step over the paged cache.
///
/// # Safety
///
/// Every plane must address one complete row per launched block.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16_paged_gqa_partitioned<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u16,
    value_pages: *const u16,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
    partials: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = block - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let length = unsafe { *lengths.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    // The represented-BF16 route uses the same contiguous slice and ordered
    // merge schedule as the FP8 target route.
    let slice_positions = length.div_ceil(DECODE_WARPS);
    let slice_begin = warp_index * slice_positions;
    let slice_end = core::cmp::min(slice_begin + slice_positions, length);
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut position = slice_begin;

    while position < slice_end {
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let cache_element = A::HEAD_DIM
            * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension;
        let key = unsafe { load_bf16x8(key_pages.add(cache_element)) };
        let value = unsafe { load_bf16x8(value_pages.add(cache_element)) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        position += 1;
    }

    let partial_base = warp_index * DECODE_PARTIAL_VALUES;
    if lane == 0 {
        unsafe {
            *partials.add(partial_base) = maximum;
            *partials.add(partial_base + 1) = denominator;
        }
    }
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *partials.add(partial_base + 2 + dimension + element) = accumulator[element] };
        element += 1;
    }
    thread::sync_threads();
    if warp_index != 0 {
        return;
    }

    let mut merged = [0.0f32; VALUES_PER_LANE];
    let mut merged_maximum = -1.0e30f32;
    let mut merged_denominator = 0.0f32;
    let mut slice = 0usize;
    while slice < DECODE_WARPS {
        let base = slice * DECODE_PARTIAL_VALUES;
        let slice_denominator = unsafe { *partials.add(base + 1) };
        if slice_denominator > 0.0 {
            let slice_maximum = unsafe { *partials.add(base) };
            let next_maximum = merged_maximum.max(slice_maximum);
            let old_scale = fast_exp(merged_maximum - next_maximum);
            let slice_scale = fast_exp(slice_maximum - next_maximum);
            merged_denominator = merged_denominator * old_scale + slice_denominator * slice_scale;
            merged_maximum = next_maximum;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                merged[element] = float::fma_rn_f32(
                    unsafe { *partials.add(base + 2 + dimension + element) },
                    slice_scale,
                    merged[element] * old_scale,
                );
                element += 1;
            }
        }
        slice += 1;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = merged[element] / merged_denominator };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn long_context_paged_gqa_partial<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    partial_maximum: *mut f32,
    partial_denominator: *mut f32,
    partial_numerator: *mut f32,
    key_scale: f32,
    value_scale: f32,
    launched_partitions: u32,
) {
    let block = thread::blockIdx_x() as usize;
    let launched_partitions = launched_partitions as usize;
    let partition = block % launched_partitions;
    let head_token = block / launched_partitions;
    let token = head_token / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = head_token - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let length = unsafe { *lengths.add(token) as usize };
    let first_position = partition * LONG_CONTEXT_PARTITION_SIZE;
    if first_position >= length {
        return;
    }

    let lane = thread::threadIdx_x() as usize;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let partition_end = core::cmp::min(first_position + LONG_CONTEXT_PARTITION_SIZE, length);
    let mut position = first_position;

    while position < partition_end {
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let cache_element = A::HEAD_DIM
            * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension;
        let key = unsafe { load_e4m3x8(key_pages.add(cache_element), key_scale) };
        let value = unsafe { load_e4m3x8(value_pages.add(cache_element), value_scale) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        position += 1;
    }

    let partial = head_token * LONG_CONTEXT_MAX_PARTITIONS + partition;
    if lane == 0 {
        unsafe {
            *partial_maximum.add(partial) = maximum;
            *partial_denominator.add(partial) = denominator;
        }
    }
    let numerator = unsafe { partial_numerator.add(partial * A::HEAD_DIM + dimension) };
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *numerator.add(element) = accumulator[element] };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn long_context_paged_gqa_reduce<A: Arch, const TOKENS: usize>(
    lengths: *const u32,
    partial_maximum: *const f32,
    partial_denominator: *const f32,
    partial_numerator: *const f32,
    output: *mut f32,
) {
    let head_token = thread::blockIdx_x() as usize;
    let token = head_token / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let lane = thread::threadIdx_x() as usize;
    let dimension = lane * VALUES_PER_LANE;
    let length = unsafe { *lengths.add(token) as usize };
    let active_partitions = length.div_ceil(LONG_CONTEXT_PARTITION_SIZE);
    let partial_base = head_token * LONG_CONTEXT_MAX_PARTITIONS;
    let mut maximum = -1.0e30f32;
    let mut partition = lane;
    while partition < active_partitions {
        maximum = maximum.max(unsafe { *partial_maximum.add(partial_base + partition) });
        partition += WARP_THREADS;
    }
    maximum = warp::reduce_max_f32(maximum);

    let weights = DynamicSharedArray::<f32, 16>::get();
    let mut denominator = 0.0f32;
    partition = lane;
    while partition < active_partitions {
        let partial = partial_base + partition;
        let weight = fast_exp(unsafe { *partial_maximum.add(partial) } - maximum);
        unsafe { *weights.add(partition) = weight };
        denominator = float::fma_rn_f32(
            weight,
            unsafe { *partial_denominator.add(partial) },
            denominator,
        );
        partition += WARP_THREADS;
    }
    denominator = warp::reduce_sum_f32(denominator);
    thread::sync_threads();

    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    partition = 0;
    while partition < active_partitions {
        let weight = unsafe { *weights.add(partition) };
        let numerator =
            unsafe { partial_numerator.add((partial_base + partition) * A::HEAD_DIM + dimension) };
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            accumulator[element] = float::fma_rn_f32(
                weight,
                unsafe { *numerator.add(element) },
                accumulator[element],
            );
            element += 1;
        }
        partition += 1;
    }

    let output = unsafe { output.add(head_token * A::HEAD_DIM + dimension) };
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

/// Runs one exact decode step over a selected-position list.
///
/// The body is `paged_gqa_partitioned` with its contiguous position walk
/// replaced by an indirection through `selected`. The slice division, the
/// per-position online-softmax order and the ascending slice merge are
/// unchanged, so a list that names every visible position in ascending order
/// reproduces the dense route's bits exactly, which makes the
/// selection route admissible below the indexer's budget.
///
/// # Safety
///
/// Every plane must address one complete row per launched block, `selected`
/// must cover `[TOKENS, selected_stride]` ascending positions, and every named
/// position must be mapped by the token's block-table row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn selected_paged_gqa_partitioned<A: Arch, const TOKENS: usize>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    selected: *const u32,
    selected_counts: *const u32,
    selected_stride: u32,
    output: *mut f32,
    key_scale: f32,
    value_scale: f32,
    partials: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_ATTENTION_HEADS;
    if token >= TOKENS {
        return;
    }
    let query_head = block - token * A::NUM_ATTENTION_HEADS;
    let kv_head = query_head / (A::NUM_ATTENTION_HEADS / A::NUM_KV_HEADS);
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let selected = unsafe { selected.add(token * selected_stride as usize) };
    let length = unsafe { *selected_counts.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let slice_positions = length.div_ceil(DECODE_WARPS);
    let slice_begin = warp_index * slice_positions;
    let slice_end = core::cmp::min(slice_begin + slice_positions, length);
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut entry = slice_begin;

    while entry < slice_end {
        let position = unsafe { *selected.add(entry) as usize };
        let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
        let page_offset = position & (PAGE_SIZE - 1);
        let cache_element = A::HEAD_DIM
            * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
            + dimension;
        let key = unsafe { load_e4m3x8(key_pages.add(cache_element), key_scale) };
        let value = unsafe { load_e4m3x8(value_pages.add(cache_element), value_scale) };
        let mut score = 0.0f32;
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            score = float::fma_rn_f32(q[element], key[element], score);
            element += 1;
        }
        score = warp::reduce_sum_f32(score) * 0.0625;

        if score > maximum {
            let old_scale = fast_exp(maximum - score);
            denominator = denominator * old_scale + 1.0;
            maximum = score;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                element += 1;
            }
        } else {
            let weight = fast_exp(score - maximum);
            denominator += weight;
            element = 0;
            while element < VALUES_PER_LANE {
                accumulator[element] =
                    float::fma_rn_f32(weight, value[element], accumulator[element]);
                element += 1;
            }
        }
        entry += 1;
    }

    let partial_base = warp_index * DECODE_PARTIAL_VALUES;
    if lane == 0 {
        unsafe {
            *partials.add(partial_base) = maximum;
            *partials.add(partial_base + 1) = denominator;
        }
    }
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *partials.add(partial_base + 2 + dimension + element) = accumulator[element] };
        element += 1;
    }
    thread::sync_threads();
    if warp_index != 0 {
        return;
    }

    let mut merged = [0.0f32; VALUES_PER_LANE];
    let mut merged_maximum = -1.0e30f32;
    let mut merged_denominator = 0.0f32;
    let mut slice = 0usize;
    while slice < DECODE_WARPS {
        let base = slice * DECODE_PARTIAL_VALUES;
        let slice_denominator = unsafe { *partials.add(base + 1) };
        if slice_denominator > 0.0 {
            let slice_maximum = unsafe { *partials.add(base) };
            let next_maximum = merged_maximum.max(slice_maximum);
            let old_scale = fast_exp(merged_maximum - next_maximum);
            let slice_scale = fast_exp(slice_maximum - next_maximum);
            merged_denominator = merged_denominator * old_scale + slice_denominator * slice_scale;
            merged_maximum = next_maximum;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                merged[element] = float::fma_rn_f32(
                    unsafe { *partials.add(base + 2 + dimension + element) },
                    slice_scale,
                    merged[element] * old_scale,
                );
                element += 1;
            }
        }
        slice += 1;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = merged[element] / merged_denominator };
        element += 1;
    }
}

/// Runs one exact prompt tile over per-row selected-position lists.
///
/// The body is `paged_gqa_prefill_shared` at `TOKEN_GROUP = 1` with the tile
/// gathering `PREFILL_KEY_TILE` *selected* positions instead of a contiguous
/// span. Each row keeps one ascending online-softmax chain, so a list naming
/// every visible position reproduces the dense route's bits.
///
/// # Safety
///
/// Carries `paged_gqa_prefill_shared`'s contract, plus: `selected` covers
/// `[TOKENS, selected_stride]` ascending positions and every named position is
/// mapped by the token's block-table row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn selected_paged_gqa_prefill_shared<
    A: Arch,
    const TOKENS: usize,
    const QUERY_WARPS: usize,
>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    selected: *const u32,
    selected_counts: *const u32,
    selected_stride: u32,
    output: *mut f32,
    key_scale: f32,
    value_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / A::NUM_KV_HEADS;
    let kv_head = block - token * A::NUM_KV_HEADS;
    if token >= TOKENS {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let query_head = kv_head * QUERY_WARPS + warp_index;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(token) as usize };
    let block_table = unsafe { block_tables.add(table_row * table_stride as usize) };
    let selected = unsafe { selected.add(token * selected_stride as usize) };
    let length = unsafe { *selected_counts.add(token) as usize };
    let q = unsafe {
        [
            *query,
            *query.add(1),
            *query.add(2),
            *query.add(3),
            *query.add(4),
            *query.add(5),
            *query.add(6),
            *query.add(7),
        ]
    };
    let shared = DynamicSharedArray::<u32, 16>::get();
    let shared_bytes = shared.cast::<u8>();
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut tile_entry = 0usize;

    while tile_entry < length {
        let mut task = tid;
        while task < 2 * PREFILL_KEY_TILE * (A::HEAD_DIM / 16) {
            let plane = task / (PREFILL_KEY_TILE * (A::HEAD_DIM / 16));
            let within_plane = task - plane * PREFILL_KEY_TILE * (A::HEAD_DIM / 16);
            let entry_in_tile = within_plane / (A::HEAD_DIM / 16);
            let dimension_segment = within_plane - entry_in_tile * (A::HEAD_DIM / 16);
            let entry = tile_entry + entry_in_tile;
            let valid = entry < length;
            let position = if valid {
                unsafe { *selected.add(entry) as usize }
            } else {
                0
            };
            let physical_page = if valid {
                unsafe { *block_table.add(position / PAGE_SIZE) as usize }
            } else {
                0
            };
            let cache_element = A::HEAD_DIM
                * ((position & (PAGE_SIZE - 1))
                    + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
                + dimension_segment * 16;
            let source = if plane == 0 { key_pages } else { value_pages };
            let destination_word = plane * PREFILL_PLANE_WORDS
                + entry_in_tile * (A::HEAD_DIM / size_of::<u32>())
                + dimension_segment * (16 / size_of::<u32>());
            unsafe {
                cp_async_cg_zfill_16(
                    shared.add(destination_word),
                    source.add(cache_element),
                    if valid { 16 } else { 0 },
                );
            }
            task += WARP_THREADS * QUERY_WARPS;
        }
        unsafe {
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();

        let tile_end = core::cmp::min(tile_entry + PREFILL_KEY_TILE, length);
        let mut entry = tile_entry;
        while entry < tile_end {
            let tile_element = (entry - tile_entry) * A::HEAD_DIM + dimension;
            let key = unsafe { load_e4m3x8(shared_bytes.add(tile_element), key_scale) };
            let value = unsafe {
                load_e4m3x8(
                    shared_bytes.add(PREFILL_PLANE_WORDS * size_of::<u32>() + tile_element),
                    value_scale,
                )
            };
            let mut score = 0.0f32;
            let mut element = 0usize;
            while element < VALUES_PER_LANE {
                score = float::fma_rn_f32(q[element], key[element], score);
                element += 1;
            }
            score = warp::reduce_sum_f32(score) * 0.0625;

            if score > maximum {
                let old_scale = fast_exp(maximum - score);
                denominator = denominator * old_scale + 1.0;
                maximum = score;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(1.0, value[element], accumulator[element] * old_scale);
                    element += 1;
                }
            } else {
                let weight = fast_exp(score - maximum);
                denominator += weight;
                element = 0;
                while element < VALUES_PER_LANE {
                    accumulator[element] =
                        float::fma_rn_f32(weight, value[element], accumulator[element]);
                    element += 1;
                }
            }
            entry += 1;
        }
        thread::sync_threads();
        tile_entry += PREFILL_KEY_TILE;
    }

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

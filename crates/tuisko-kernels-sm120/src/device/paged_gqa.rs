use crate::device::fp8_projection::e4m3x2_to_f32;
use cuda_device::async_copy::{cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::{DynamicSharedArray, float, thread, warp};
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 8;
const PAGE_SIZE: usize = 64;
const WARP_THREADS: usize = 32;
const PREFILL_TOKEN_GROUP: usize = 2;
const PREFILL_QUERY_WARPS: usize = 6;
pub(crate) const PREFILL_THREADS: usize = 384;
const PREFILL_KEY_TILE: usize = 64;
const PREFILL_PLANE_WORDS: usize = PREFILL_KEY_TILE * 256 / size_of::<u32>();
pub(crate) const PREFILL_SHARED_BYTES: usize = 2 * PREFILL_PLANE_WORDS * size_of::<u32>();
pub(crate) const PREFILL_PARTIAL_VALUES: usize = 258;
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
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn paged_gqa_prefill_shared<A: Arch, const TOKENS: usize>(
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
    let token_pair = block / A::NUM_KV_HEADS;
    let kv_head = block - token_pair * A::NUM_KV_HEADS;
    let first_token = token_pair * PREFILL_TOKEN_GROUP;
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let token_in_group = warp_index / PREFILL_QUERY_WARPS;
    let query_in_group = warp_index - token_in_group * PREFILL_QUERY_WARPS;
    let token = first_token + token_in_group;
    let query_head = kv_head * PREFILL_QUERY_WARPS + query_in_group;
    let dimension = lane * VALUES_PER_LANE;
    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let output = unsafe {
        output.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let length0 = unsafe { *lengths.add(first_token) as usize };
    let length1 = unsafe { *lengths.add(first_token + 1) as usize };
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
            task += PREFILL_THREADS;
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
pub(crate) unsafe fn paged_gqa_prefill_partitioned<
    A: Arch,
    const TOKENS: usize,
    const PARTITIONS: usize,
>(
    query: *const f32,
    key_pages: *const u8,
    value_pages: *const u8,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    partials: *mut f32,
    key_scale: f32,
    value_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let partition = block % PARTITIONS;
    let group = block / PARTITIONS;
    let kv_head = group % A::NUM_KV_HEADS;
    let token_pair = group / A::NUM_KV_HEADS;
    let first_token = token_pair * PREFILL_TOKEN_GROUP;
    let tid = thread::threadIdx_x() as usize;
    let warp_index = tid / WARP_THREADS;
    let lane = tid & (WARP_THREADS - 1);
    let token_in_group = warp_index / PREFILL_QUERY_WARPS;
    let query_in_group = warp_index - token_in_group * PREFILL_QUERY_WARPS;
    let token = first_token + token_in_group;
    let query_head = kv_head * PREFILL_QUERY_WARPS + query_in_group;
    let dimension = lane * VALUES_PER_LANE;
    let length0 = unsafe { *lengths.add(first_token) as usize };
    let length1 = unsafe { *lengths.add(first_token + 1) as usize };
    let group_length = length0.max(length1);
    let positions_per_partition = group_length.div_ceil(PARTITIONS);
    let partition_begin = partition * positions_per_partition;
    let partition_end = core::cmp::min(partition_begin + positions_per_partition, group_length);
    let token_length = if token_in_group == 0 {
        length0
    } else {
        length1
    };
    let token_partition_end = core::cmp::min(partition_end, token_length);
    let partial_base = ((token * A::NUM_ATTENTION_HEADS + query_head) * PARTITIONS + partition)
        * PREFILL_PARTIAL_VALUES;

    if partition_begin >= partition_end {
        if lane == 0 {
            unsafe {
                *partials.add(partial_base) = -1.0e30;
                *partials.add(partial_base + 1) = 0.0;
            }
        }
        let mut offset = lane;
        while offset < A::HEAD_DIM {
            unsafe { *partials.add(partial_base + 2 + offset) = 0.0 };
            offset += WARP_THREADS;
        }
        return;
    }

    let query = unsafe {
        query.add((token * A::NUM_ATTENTION_HEADS + query_head) * A::HEAD_DIM + dimension)
    };
    let table_row = unsafe { *table_rows.add(first_token) as usize };
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
    let shared = DynamicSharedArray::<u32, 16>::get();
    let shared_bytes = shared.cast::<u8>();
    let mut accumulator = [0.0f32; VALUES_PER_LANE];
    let mut maximum = -1.0e30f32;
    let mut denominator = 0.0f32;
    let mut tile_position = partition_begin;

    while tile_position < partition_end {
        let tile_positions = core::cmp::min(partition_end - tile_position, PREFILL_KEY_TILE);
        let mut task = tid;
        while task < 2 * tile_positions * (A::HEAD_DIM / 16) {
            let plane = task / (tile_positions * (A::HEAD_DIM / 16));
            let within_plane = task - plane * tile_positions * (A::HEAD_DIM / 16);
            let position_in_tile = within_plane / (A::HEAD_DIM / 16);
            let dimension_segment = within_plane - position_in_tile * (A::HEAD_DIM / 16);
            let position = tile_position + position_in_tile;
            let physical_page = unsafe { *block_table.add(position / PAGE_SIZE) as usize };
            let cache_element = A::HEAD_DIM
                * ((position & (PAGE_SIZE - 1))
                    + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
                + dimension_segment * 16;
            let source = if plane == 0 { key_pages } else { value_pages };
            let destination_word = plane * PREFILL_PLANE_WORDS
                + position_in_tile * (A::HEAD_DIM / size_of::<u32>())
                + dimension_segment * (16 / size_of::<u32>());
            unsafe {
                cp_async_cg_zfill_16(shared.add(destination_word), source.add(cache_element), 16);
            }
            task += PREFILL_THREADS;
        }
        unsafe {
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();

        let mut position = tile_position;
        while position < tile_position + tile_positions {
            if position < token_partition_end {
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
                        accumulator[element] = float::fma_rn_f32(
                            1.0,
                            value[element],
                            accumulator[element] * old_scale,
                        );
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
            }
            position += 1;
        }
        thread::sync_threads();
        tile_position += tile_positions;
    }

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
    let mut position = 0usize;

    while position < length {
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

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
        element += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen35_paged_gqa_bf16<A: Arch, const TOKENS: usize>(
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
    let mut position = 0usize;

    while position < length {
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

    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        unsafe { *output.add(element) = accumulator[element] / denominator };
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

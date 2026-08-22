use crate::device::fp8_projection::e4m3x2_to_f32;
use cuda_device::{float, thread, warp};
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 8;
const PAGE_SIZE: usize = 64;

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
fn fast_exp(value: f32) -> f32 {
    float::ex2_approx_f32(value * core::f32::consts::LOG2_E)
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

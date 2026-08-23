use cuda_device::{convert, float, thread, warp};
use tuisko_model::Arch;

const WARPS_PER_CTA: usize = 8;
const VALUES_PER_LANE: usize = 8;
const ROTARY_DIM: usize = 64;
const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const PAGE_SIZE: usize = 64;

#[inline(always)]
fn bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
unsafe fn load_bf16x8(source: *const u16) -> [f32; VALUES_PER_LANE] {
    unsafe {
        [
            bf16_bits(*source),
            bf16_bits(*source.add(1)),
            bf16_bits(*source.add(2)),
            bf16_bits(*source.add(3)),
            bf16_bits(*source.add(4)),
            bf16_bits(*source.add(5)),
            bf16_bits(*source.add(6)),
            bf16_bits(*source.add(7)),
        ]
    }
}

#[inline(always)]
unsafe fn store_fp8x8(destination: *mut u8, scale: f32, values: [f32; VALUES_PER_LANE]) {
    let inverse_scale = 1.0 / scale;
    let destination = destination.cast::<u16>();
    unsafe {
        *destination = convert::cvt_rn_satfinite_e4m3x2_f32(
            values[0] * inverse_scale,
            values[1] * inverse_scale,
        );
        *destination.add(1) = convert::cvt_rn_satfinite_e4m3x2_f32(
            values[2] * inverse_scale,
            values[3] * inverse_scale,
        );
        *destination.add(2) = convert::cvt_rn_satfinite_e4m3x2_f32(
            values[4] * inverse_scale,
            values[5] * inverse_scale,
        );
        *destination.add(3) = convert::cvt_rn_satfinite_e4m3x2_f32(
            values[6] * inverse_scale,
            values[7] * inverse_scale,
        );
    }
}

#[inline(always)]
unsafe fn normalize_rotate<A: Arch>(
    source: *const u16,
    norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    token: usize,
    dimension: usize,
) -> [f32; VALUES_PER_LANE] {
    let values = unsafe { load_bf16x8(source) };
    let weights = unsafe { load_bf16x8(norm.add(dimension)) };
    let sum = values.iter().map(|value| value * value).sum::<f32>();
    let inverse_rms = float::rsqrt_approx_f32(
        warp::reduce_sum_f32(sum) / A::HEAD_DIM as f32 + A::RMS_NORM_EPSILON,
    );
    let mut normalized = [0.0f32; VALUES_PER_LANE];
    let mut element = 0usize;
    while element < VALUES_PER_LANE {
        normalized[element] = values[element] * inverse_rms * (1.0 + weights[element]);
        element += 1;
    }

    if dimension < ROTARY_DIM {
        element = 0;
        while element < VALUES_PER_LANE {
            let value = normalized[element];
            let peer = warp::shuffle_xor_f32(value, 4);
            let rotary_element = dimension + element;
            let pair = rotary_element & (ROTARY_PAIRS - 1);
            let cosine = unsafe { *rope_cos.add(token * ROTARY_PAIRS + pair) };
            let sine = unsafe { *rope_sin.add(token * ROTARY_PAIRS + pair) };
            normalized[element] = if rotary_element < ROTARY_PAIRS {
                value * cosine - peer * sine
            } else {
                peer * sine + value * cosine
            };
            element += 1;
        }
    }

    normalized
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention_qk_prepare<A: Arch, const TOKENS: usize>(
    qkv: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    cache_positions: *const u32,
    query: *mut f32,
    key_pages: *mut u8,
    value_pages: *mut u8,
    key_scale: f32,
    value_scale: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = thread::blockIdx_x() as usize * WARPS_PER_CTA + tid / 32;
    let heads_per_token = A::NUM_ATTENTION_HEADS + A::NUM_KV_HEADS;
    let token = warp / heads_per_token;
    if token >= TOKENS {
        return;
    }

    let combined_head = warp - token * heads_per_token;
    let dimension = lane * VALUES_PER_LANE;
    let (source, norm, destination, kv_head) = if combined_head < A::NUM_ATTENTION_HEADS {
        let source = unsafe {
            qkv.add(token * A::ATTENTION_QKV_ROWS + combined_head * 2 * A::HEAD_DIM + dimension)
        };
        let destination = unsafe {
            query.add((token * A::NUM_ATTENTION_HEADS + combined_head) * A::HEAD_DIM + dimension)
        };
        (source, query_norm, destination, usize::MAX)
    } else {
        let kv_head = combined_head - A::NUM_ATTENTION_HEADS;
        let source = unsafe {
            qkv.add(
                token * A::ATTENTION_QKV_ROWS
                    + A::ATTENTION_QUERY_ROWS
                    + kv_head * A::HEAD_DIM
                    + dimension,
            )
        };
        (source, key_norm, core::ptr::null_mut(), kv_head)
    };
    let prepared =
        unsafe { normalize_rotate::<A>(source, norm, rope_cos, rope_sin, token, dimension) };

    if combined_head < A::NUM_ATTENTION_HEADS {
        let mut element = 0usize;
        while element < VALUES_PER_LANE {
            unsafe { *destination.add(element) = prepared[element] };
            element += 1;
        }
        return;
    }

    let table_row = unsafe { *table_rows.add(token) as usize };
    let position = unsafe { *cache_positions.add(token) as usize };
    let physical_page = unsafe {
        *block_tables.add(table_row * table_stride as usize + position / PAGE_SIZE) as usize
    };
    let page_offset = position & (PAGE_SIZE - 1);
    let cache_element = A::HEAD_DIM
        * (page_offset + PAGE_SIZE * (kv_head + A::NUM_KV_HEADS * physical_page))
        + dimension;
    unsafe { store_fp8x8(key_pages.add(cache_element), key_scale, prepared) };

    let value_source = unsafe {
        qkv.add(
            token * A::ATTENTION_QKV_ROWS
                + A::ATTENTION_QUERY_ROWS
                + A::ATTENTION_KV_ROWS
                + kv_head * A::HEAD_DIM
                + dimension,
        )
    };
    let values = unsafe { load_bf16x8(value_source) };
    unsafe { store_fp8x8(value_pages.add(cache_element), value_scale, values) };
}

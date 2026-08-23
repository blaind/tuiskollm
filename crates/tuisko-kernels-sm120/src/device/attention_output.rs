use cuda_device::{convert, float, thread, warp};
use tuisko_model::Arch;

const WARPS: usize = 8;
const THREADS: usize = WARPS * 32;
const FP8_MAX: f32 = 448.0;

#[inline(always)]
fn bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
unsafe fn gated_value<A: Arch>(attention: *const f32, qkv: *const u16, index: usize) -> f32 {
    let token = thread::blockIdx_x() as usize;
    let head = index / A::HEAD_DIM;
    let dimension = index - head * A::HEAD_DIM;
    let gate = unsafe {
        *qkv.add(token * A::ATTENTION_QKV_ROWS + head * (2 * A::HEAD_DIM) + A::HEAD_DIM + dimension)
    };
    let gate = bf16_bits(gate);
    let sigmoid = 1.0 / (1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E));

    (unsafe { *attention.add(token * A::ATTENTION_OUTPUT_COLUMNS + index) }) * sigmoid
}

#[inline(always)]
pub(crate) unsafe fn attention_gate_quantize<A: Arch>(
    attention: *mut f32,
    qkv: *const u16,
    codes: *mut u16,
    scales: *mut f32,
    warp_maximum: *mut f32,
) {
    let token = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let mut maximum = 0.0f32;
    let mut index = tid;

    // The paged-attention output is scratch after this boundary. Publishing
    // the gated FP32 seam avoids a second sigmoid and keeps it observable.
    while index < A::ATTENTION_OUTPUT_COLUMNS {
        let gated = unsafe { gated_value::<A>(attention, qkv, index) };
        unsafe { *attention.add(token * A::ATTENTION_OUTPUT_COLUMNS + index) = gated };
        maximum = maximum.max(gated.abs());
        index += THREADS;
    }
    maximum = warp::reduce_max_f32(maximum);
    if lane == 0 {
        unsafe { *warp_maximum.add(warp_index) = maximum };
    }
    thread::sync_threads();

    if warp_index == 0 {
        maximum = if lane < WARPS {
            unsafe { *warp_maximum.add(lane) }
        } else {
            0.0
        };
        maximum = warp::reduce_max_f32(maximum);
        if lane == 0 {
            unsafe {
                *scales.add(token) = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
            }
        }
    }
    thread::sync_threads();

    let inverse_scale = 1.0 / unsafe { *scales.add(token) };
    let mut pair = tid;
    while pair < A::ATTENTION_OUTPUT_COLUMNS / 2 {
        let base = token * A::ATTENTION_OUTPUT_COLUMNS + pair * 2;
        let low = unsafe { *attention.add(base) };
        let high = unsafe { *attention.add(base + 1) };
        unsafe {
            *codes.add(token * A::ATTENTION_OUTPUT_COLUMNS / 2 + pair) =
                convert::cvt_rn_satfinite_e4m3x2_f32(low * inverse_scale, high * inverse_scale);
        }
        pair += THREADS;
    }
}

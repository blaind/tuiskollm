//! Device bodies shared by more than one SM120 kernel crate.
//!
//! Every item here is prepared again by cuda-oxide once per kernel crate that
//! calls it, so this module stays limited to leaf primitives.

use cuda_device::{convert, float, ptx_asm, thread, warp};
use tuisko_model::Arch;

/// Largest magnitude representable by E4M3.
const FP8_MAX: f32 = 448.0;

/// Loads one aligned 16-byte fragment through the read-only data path.
///
/// # Safety
///
/// `source` must address sixteen readable bytes aligned to sixteen bytes.
#[inline(always)]
pub unsafe fn load_u32x4_read_only(source: *const u32) -> (u32, u32, u32, u32) {
    let first: u32;
    let second: u32;
    let third: u32;
    let fourth: u32;

    // SAFETY: the caller guarantees one aligned 16-byte source fragment.
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

/// Widens one packed E4M3 pair to two `f32` values.
#[inline(always)]
pub fn e4m3x2_to_f32(packed: u16) -> (f32, f32) {
    let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(packed);

    convert::cvt_f32x2_f16x2(packed_f16)
}

/// Reduces one `f32` across the warp, leaving the total in lane zero.
#[inline(always)]
pub fn reduce_sum_lane_zero(mut value: f32) -> f32 {
    value += warp::shuffle_down_f32(value, 16);
    value += warp::shuffle_down_f32(value, 8);
    value += warp::shuffle_down_f32(value, 4);
    value += warp::shuffle_down_f32(value, 2);
    value += warp::shuffle_down_f32(value, 1);

    value
}

/// Quantizes one BF16 activation row to E4M3 codes plus a per-token scale.
///
/// # Safety
///
/// The planes must address one complete row per launched block and
/// `warp_maximum` must address one `f32` per warp in the block.
#[inline(always)]
pub unsafe fn quantize_activation<A: Arch>(
    input: *const u32,
    codes: *mut u16,
    scale: *mut f32,
    warp_maximum: *mut f32,
) {
    let threads = thread::blockDim_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let token = thread::blockIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let pairs = A::HIDDEN / 2;
    // SAFETY: one block owns one complete input row and its output code row.
    let input = unsafe { input.add(token * pairs) };
    // SAFETY: the code plane contains one packed byte pair per BF16 pair.
    let codes = unsafe { codes.add(token * pairs) };
    // SAFETY: the scale plane contains one value per launched block.
    let scale = unsafe { scale.add(token) };
    let mut maximum = 0.0f32;
    let mut pair = tid;

    while pair < pairs {
        // SAFETY: `pair < pairs` within this block's row.
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        maximum = maximum.max(low.abs()).max(high.abs());
        pair += threads;
    }

    maximum = warp::reduce_max_f32(maximum);
    if lane == 0 {
        // SAFETY: one lane writes its warp's unique shared slot.
        unsafe { *warp_maximum.add(warp_index) = maximum };
    }
    thread::sync_threads();

    if warp_index == 0 {
        maximum = if lane < threads / 32 {
            // SAFETY: the barrier published every active warp maximum.
            unsafe { *warp_maximum.add(lane) }
        } else {
            0.0
        };
        maximum = warp::reduce_max_f32(maximum);
        if lane == 0 {
            // SAFETY: lane zero owns the token's scale output.
            unsafe {
                *scale = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
            }
        }
    }
    thread::sync_threads();

    // SAFETY: the second barrier makes the represented scale visible.
    let represented_scale = unsafe { *scale };
    pair = tid;

    while pair < pairs {
        // SAFETY: the read and write are within this block's complete row.
        let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
        // SAFETY: each thread writes disjoint packed E4M3 byte pairs.
        unsafe {
            *codes.add(pair) = convert::cvt_rn_satfinite_e4m3x2_f32(
                float::div_rn_f32(low, represented_scale),
                float::div_rn_f32(high, represented_scale),
            );
        }
        pair += threads;
    }
}

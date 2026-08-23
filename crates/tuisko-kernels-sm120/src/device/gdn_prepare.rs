use cuda_device::{float, tcgen05, thread, warp};
use tuisko_model::Arch;

#[inline(always)]
fn bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
unsafe fn bf16(source: *const u16) -> f32 {
    bf16_bits(unsafe { *source })
}

#[inline(always)]
fn fast_exp(value: f32) -> f32 {
    float::ex2_approx_f32(value * core::f32::consts::LOG2_E)
}

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-value))
}

#[inline(always)]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        float::lg2_approx_f32(1.0 + fast_exp(value)) * core::f32::consts::LN_2
    }
}

#[inline(always)]
pub(crate) unsafe fn gdn_control<A: Arch, const TOKENS: usize>(
    input: *const u32,
    control_weights: *const u16,
    a_log: *const u16,
    dt_bias: *const u16,
    log_decay: *mut f32,
    beta: *mut f32,
    warp_sums: *mut f32,
) {
    const WARPS: usize = 16;
    const ROWS_PER_CTA: usize = WARPS / 2;

    let block = thread::blockIdx_x() as usize;
    let ctas_per_token = 2 * A::GDN_CONTROL_ROWS / ROWS_PER_CTA;
    let token = block / ctas_per_token;
    if token >= TOKENS {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let row = (block % ctas_per_token) * ROWS_PER_CTA + warp_index / 2;
    let input = unsafe { input.add(token * A::HIDDEN / 2) }.cast::<u16>();
    let weight = unsafe { control_weights.add(row * A::HIDDEN) };
    let mut sum = 0.0f32;
    let mut column = lane + (warp_index & 1) * 32;

    while column < A::HIDDEN {
        sum = float::fma_rn_f32(
            unsafe { bf16(input.add(column)) },
            unsafe { bf16(weight.add(column)) },
            sum,
        );
        column += 64;
    }

    sum = warp::reduce_sum_f32(sum);
    if lane == 0 {
        unsafe { *warp_sums.add(warp_index) = sum };
    }
    thread::sync_threads();

    if lane == 0 && warp_index & 1 == 0 {
        let sum = unsafe { *warp_sums.add(warp_index) + *warp_sums.add(warp_index + 1) };
        if row < A::GDN_CONTROL_ROWS {
            let control = sum + unsafe { bf16(dt_bias.add(row)) };
            unsafe {
                *log_decay.add(token * A::GDN_CONTROL_ROWS + row) =
                    -fast_exp(bf16(a_log.add(row))) * softplus(control);
            }
        } else {
            unsafe {
                *beta.add(token * A::GDN_CONTROL_ROWS + row - A::GDN_CONTROL_ROWS) = sigmoid(sum);
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn gdn_convolution<A: Arch, const TOKENS: usize>(
    projected: *const u16,
    weights: *const u16,
    state_rows: *const u32,
    history: *mut u16,
    output: *mut u16,
) {
    const HISTORY: usize = 3;

    let index = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if index >= TOKENS * A::GDN_QKV_ROWS {
        return;
    }

    let token = index / A::GDN_QKV_ROWS;
    let channel = index - token * A::GDN_QKV_ROWS;
    let state_row = unsafe { *state_rows.add(token) as usize };
    let history = unsafe { history.add((state_row * A::GDN_QKV_ROWS + channel) * HISTORY) };
    let current = unsafe { *projected.add(token * A::GDN_INPUT_ROWS + channel) };
    let h0 = unsafe { *history };
    let h1 = unsafe { *history.add(1) };
    let h2 = unsafe { *history.add(2) };

    unsafe {
        *history = h1;
        *history.add(1) = h2;
        *history.add(2) = current;
    }

    let weights = unsafe { weights.add(channel * A::LINEAR_CONV_KERNEL_DIM) };
    let sum = float::fma_rn_f32(
        unsafe { bf16(weights) },
        bf16_bits(h0),
        float::fma_rn_f32(
            unsafe { bf16(weights.add(1)) },
            bf16_bits(h1),
            float::fma_rn_f32(
                unsafe { bf16(weights.add(2)) },
                bf16_bits(h2),
                unsafe { bf16(weights.add(3)) } * bf16_bits(current),
            ),
        ),
    );
    let activated = sum * sigmoid(sum);

    unsafe {
        *output.add(token * A::GDN_QKV_ROWS + channel) = tcgen05::f32_to_bf16_rne(activated);
    }
}

#[inline(always)]
unsafe fn projected_channel<A: Arch>(projected: *const u16, token: usize, channel: usize) -> u16 {
    unsafe { *projected.add(token * A::GDN_INPUT_ROWS + channel) }
}

#[inline(always)]
pub(crate) unsafe fn gdn_convolution_prefill<A: Arch, const TOKENS: usize>(
    projected: *const u16,
    weights: *const u16,
    state_rows: *const u32,
    history: *const u16,
    output: *mut u16,
) {
    const HISTORY: usize = 3;

    let index = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if index >= TOKENS * A::GDN_QKV_ROWS {
        return;
    }

    let token = index / A::GDN_QKV_ROWS;
    let channel = index - token * A::GDN_QKV_ROWS;
    let state_row = unsafe { *state_rows as usize };
    let history = unsafe { history.add((state_row * A::GDN_QKV_ROWS + channel) * HISTORY) };
    let current = unsafe { projected_channel::<A>(projected, token, channel) };
    let h0 = if token >= 3 {
        unsafe { projected_channel::<A>(projected, token - 3, channel) }
    } else {
        unsafe { *history.add(token) }
    };
    let h1 = if token >= 2 {
        unsafe { projected_channel::<A>(projected, token - 2, channel) }
    } else {
        unsafe { *history.add(token + 1) }
    };
    let h2 = if token >= 1 {
        unsafe { projected_channel::<A>(projected, token - 1, channel) }
    } else {
        unsafe { *history.add(2) }
    };

    let weights = unsafe { weights.add(channel * A::LINEAR_CONV_KERNEL_DIM) };
    let sum = float::fma_rn_f32(
        unsafe { bf16(weights) },
        bf16_bits(h0),
        float::fma_rn_f32(
            unsafe { bf16(weights.add(1)) },
            bf16_bits(h1),
            float::fma_rn_f32(
                unsafe { bf16(weights.add(2)) },
                bf16_bits(h2),
                unsafe { bf16(weights.add(3)) } * bf16_bits(current),
            ),
        ),
    );
    let activated = sum * sigmoid(sum);

    unsafe {
        *output.add(token * A::GDN_QKV_ROWS + channel) = tcgen05::f32_to_bf16_rne(activated);
    }
}

#[inline(always)]
pub(crate) unsafe fn gdn_convolution_prefill_history<A: Arch, const TOKENS: usize>(
    projected: *const u16,
    state_rows: *const u32,
    history: *mut u16,
) {
    const HISTORY: usize = 3;

    let channel = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if channel >= A::GDN_QKV_ROWS {
        return;
    }

    let state_row = unsafe { *state_rows as usize };
    let history = unsafe { history.add((state_row * A::GDN_QKV_ROWS + channel) * HISTORY) };
    unsafe {
        *history = projected_channel::<A>(projected, TOKENS - 3, channel);
        *history.add(1) = projected_channel::<A>(projected, TOKENS - 2, channel);
        *history.add(2) = projected_channel::<A>(projected, TOKENS - 1, channel);
    }
}

use cuda_device::{float, tcgen05, thread, warp};
use tuisko_model::Arch;

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const WARPS: usize = 16;
const RMS_EPSILON: f32 = 1.0e-6;
const DELTA_SCALE: f32 = 0.088_388_35;

#[inline(always)]
fn load_bf16(source: *const u16) -> f32 {
    f32::from_bits((unsafe { *source } as u32) << 16)
}

#[inline(always)]
fn block_sum(value: f32, shared: *mut f32, lane: usize, warp_index: usize) -> f32 {
    let value = warp::reduce_sum_f32(value);
    if lane == 0 {
        unsafe { *shared.add(warp_index) = value };
    }
    thread::sync_threads();
    if warp_index == 0 {
        let value = if lane < WARPS {
            unsafe { *shared.add(lane) }
        } else {
            0.0
        };
        let value = warp::reduce_sum_f32(value);
        if lane == 0 {
            unsafe { *shared = value };
        }
    }
    thread::sync_threads();

    unsafe { *shared }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_recurrence_token<A: Arch>(
    qkv: *const u16,
    projected: *const u16,
    log_decay: *const f32,
    beta: *const f32,
    norm_weight: *const u16,
    state: *mut f32,
    output: *mut u16,
    token: usize,
    value_head: usize,
    state_row: usize,
    query: *mut f32,
    key: *mut f32,
    recurrent_output: *mut f32,
    reduction: *mut f32,
) {
    let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let token_qkv = unsafe { qkv.add(token * A::GDN_QKV_ROWS) };
    let query_row = unsafe { token_qkv.add(key_head * HEAD_DIM) };
    let key_row = unsafe { token_qkv.add(QK_WIDTH + key_head * HEAD_DIM) };

    if tid < HEAD_DIM {
        unsafe {
            *query.add(tid) = load_bf16(query_row.add(tid));
            *key.add(tid) = load_bf16(key_row.add(tid));
        }
    }
    thread::sync_threads();

    let query_square = if tid < HEAD_DIM {
        let value = unsafe { *query.add(tid) };
        value * value
    } else {
        0.0
    };
    let query_norm =
        float::rsqrt_approx_f32(block_sum(query_square, reduction, lane, warp_index) + RMS_EPSILON);
    if tid < HEAD_DIM {
        unsafe { *query.add(tid) *= query_norm };
    }
    thread::sync_threads();

    let key_square = if tid < HEAD_DIM {
        let value = unsafe { *key.add(tid) };
        value * value
    } else {
        0.0
    };
    let key_norm =
        float::rsqrt_approx_f32(block_sum(key_square, reduction, lane, warp_index) + RMS_EPSILON);
    if tid < HEAD_DIM {
        unsafe { *key.add(tid) *= key_norm };
    }
    thread::sync_threads();

    let control = token * VALUE_HEADS + value_head;
    let decay =
        float::ex2_approx_f32(unsafe { *log_decay.add(control) } * core::f32::consts::LOG2_E);
    let beta = unsafe { *beta.add(control) };
    let value_row = unsafe { token_qkv.add(2 * QK_WIDTH + value_head * HEAD_DIM) };
    let state = unsafe { state.add((state_row * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM) };
    let mut row = warp_index;

    while row < HEAD_DIM {
        let state_row = unsafe { state.add(row * HEAD_DIM) };
        let column0 = lane;
        let column1 = lane + 32;
        let column2 = lane + 64;
        let column3 = lane + 96;
        let old0 = unsafe { *state_row.add(column0) };
        let old1 = unsafe { *state_row.add(column1) };
        let old2 = unsafe { *state_row.add(column2) };
        let old3 = unsafe { *state_row.add(column3) };
        let k0 = unsafe { *key.add(column0) };
        let k1 = unsafe { *key.add(column1) };
        let k2 = unsafe { *key.add(column2) };
        let k3 = unsafe { *key.add(column3) };
        let state_key = warp::reduce_sum_f32(float::fma_rn_f32(
            old0,
            k0,
            float::fma_rn_f32(old1, k1, float::fma_rn_f32(old2, k2, old3 * k3)),
        ));
        let update = beta * (load_bf16(unsafe { value_row.add(row) }) - decay * state_key);
        let new0 = float::fma_rn_f32(old0, decay, k0 * update);
        let new1 = float::fma_rn_f32(old1, decay, k1 * update);
        let new2 = float::fma_rn_f32(old2, decay, k2 * update);
        let new3 = float::fma_rn_f32(old3, decay, k3 * update);
        unsafe {
            *state_row.add(column0) = new0;
            *state_row.add(column1) = new1;
            *state_row.add(column2) = new2;
            *state_row.add(column3) = new3;
        }
        let recurrent = warp::reduce_sum_f32(float::fma_rn_f32(
            new0,
            unsafe { *query.add(column0) },
            float::fma_rn_f32(
                new1,
                unsafe { *query.add(column1) },
                float::fma_rn_f32(
                    new2,
                    unsafe { *query.add(column2) },
                    new3 * unsafe { *query.add(column3) },
                ),
            ),
        ));
        if lane == 0 {
            unsafe { *recurrent_output.add(row) = recurrent * DELTA_SCALE };
        }
        row += WARPS;
    }
    thread::sync_threads();

    let square = if tid < HEAD_DIM {
        let value = unsafe { *recurrent_output.add(tid) };
        value * value
    } else {
        0.0
    };
    let inverse_rms = float::rsqrt_approx_f32(
        block_sum(square, reduction, lane, warp_index) / HEAD_DIM as f32 + RMS_EPSILON,
    );
    if tid < HEAD_DIM {
        let normalized = unsafe { *recurrent_output.add(tid) }
            * inverse_rms
            * load_bf16(unsafe { norm_weight.add(tid) });
        let gate = load_bf16(unsafe {
            projected.add(token * A::GDN_INPUT_ROWS + A::GDN_QKV_ROWS + value_head * HEAD_DIM + tid)
        });
        let silu = gate / (1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E));
        unsafe {
            *output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + tid) =
                tcgen05::f32_to_bf16_rne(normalized * silu);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gdn_recurrence<A: Arch, const TOKENS: usize>(
    qkv: *const u16,
    projected: *const u16,
    log_decay: *const f32,
    beta: *const f32,
    norm_weight: *const u16,
    state_rows: *const u32,
    state: *mut f32,
    output: *mut u16,
    query: *mut f32,
    key: *mut f32,
    recurrent_output: *mut f32,
    reduction: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / VALUE_HEADS;
    if token >= TOKENS {
        return;
    }
    let value_head = block - token * VALUE_HEADS;
    let state_row = unsafe { *state_rows.add(token) as usize };
    unsafe {
        gdn_recurrence_token::<A>(
            qkv,
            projected,
            log_decay,
            beta,
            norm_weight,
            state,
            output,
            token,
            value_head,
            state_row,
            query,
            key,
            recurrent_output,
            reduction,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gdn_recurrence_prefill<A: Arch, const TOKENS: usize>(
    qkv: *const u16,
    projected: *const u16,
    log_decay: *const f32,
    beta: *const f32,
    norm_weight: *const u16,
    state_rows: *const u32,
    state: *mut f32,
    output: *mut u16,
    query: *mut f32,
    key: *mut f32,
    recurrent_output: *mut f32,
    reduction: *mut f32,
) {
    let value_head = thread::blockIdx_x() as usize;
    if value_head >= VALUE_HEADS {
        return;
    }
    let state_row = unsafe { *state_rows as usize };
    let mut token = 0;
    while token < TOKENS {
        unsafe {
            gdn_recurrence_token::<A>(
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state,
                output,
                token,
                value_head,
                state_row,
                query,
                key,
                recurrent_output,
                reduction,
            );
        }
        token += 1;
    }
}

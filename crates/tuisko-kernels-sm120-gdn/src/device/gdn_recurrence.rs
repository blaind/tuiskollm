use cuda_device::{float, tcgen05, thread, warp};
use tuisko_model::Arch;

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const WARPS: usize = 16;
// Two CTAs advance each value head's state plane; every row's update stays
// wholly inside one CTA, so the split changes scheduling, never arithmetic.
pub(crate) const SPLIT_CTAS_PER_HEAD: usize = 2;
pub(crate) const SPLIT_ROWS: usize = HEAD_DIM / SPLIT_CTAS_PER_HEAD;
const RMS_EPSILON: f32 = 1.0e-6;
const DELTA_SCALE: f32 = 0.088_388_35;

#[inline(always)]
fn load_bf16(source: *const u16) -> f32 {
    f32::from_bits((unsafe { *source } as u32) << 16)
}

/// Applies the route's exact `sigmoid` or SiLU output gate.
///
/// Keep `gate / denominator` for SiLU to preserve its represented result.
#[inline(always)]
fn output_gate<const SIGMOID: bool>(gate: f32) -> f32 {
    let denominator = 1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E);
    if SIGMOID {
        1.0 / denominator
    } else {
        gate / denominator
    }
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

/// Reduces two independent values with one barrier pair; each value walks
/// exactly the tree `block_sum` walks, so both sums stay bit-identical to
/// two separate `block_sum` calls while paying half the synchronization.
#[inline(always)]
fn block_sum2(a: f32, b: f32, shared: *mut f32, lane: usize, warp_index: usize) -> (f32, f32) {
    let a = warp::reduce_sum_f32(a);
    let b = warp::reduce_sum_f32(b);
    if lane == 0 {
        unsafe { *shared.add(warp_index) = a };
        unsafe { *shared.add(WARPS + warp_index) = b };
    }
    thread::sync_threads();
    if warp_index == 0 {
        let a = if lane < WARPS {
            unsafe { *shared.add(lane) }
        } else {
            0.0
        };
        let b = if lane < WARPS {
            unsafe { *shared.add(WARPS + lane) }
        } else {
            0.0
        };
        let a = warp::reduce_sum_f32(a);
        let b = warp::reduce_sum_f32(b);
        if lane == 0 {
            unsafe { *shared = a };
            unsafe { *shared.add(1) = b };
        }
    }
    thread::sync_threads();

    (unsafe { *shared }, unsafe { *shared.add(1) })
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_recurrence_token<A: Arch, const PREFILL: bool, const SIGMOID_GATE: bool>(
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
    value: *mut f32,
    row_offset: usize,
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
    if PREFILL && tid < HEAD_DIM {
        // Prefetching the value row through the same barrier removes a
        // dependent global load from every row iteration; the conversion is
        // the identical load_bf16, so the consumed value is bit-exact.
        let prefetch_value_row = unsafe { token_qkv.add(2 * QK_WIDTH + value_head * HEAD_DIM) };
        unsafe { *value.add(tid) = load_bf16(prefetch_value_row.add(tid)) };
    }
    thread::sync_threads();

    // The prefill route fuses both squared-sum reductions into one barrier
    // pair and applies the norm scalars at every use site instead of writing
    // normalized rows back to shared. Each sum walks the identical reduction
    // tree and each normalized element is the identical two-operand multiply,
    // so every downstream value stays bit-exact while four of the nine
    // per-token barriers disappear. The decode route keeps its original
    // statement sequence untouched.
    let (query_norm, key_norm) = if PREFILL {
        let (query_square, key_square) = if tid < HEAD_DIM {
            let query_value = unsafe { *query.add(tid) };
            let key_value = unsafe { *key.add(tid) };
            (query_value * query_value, key_value * key_value)
        } else {
            (0.0, 0.0)
        };
        let (query_sum, key_sum) =
            block_sum2(query_square, key_square, reduction, lane, warp_index);
        (
            float::rsqrt_approx_f32(query_sum + RMS_EPSILON),
            float::rsqrt_approx_f32(key_sum + RMS_EPSILON),
        )
    } else {
        let query_square = if tid < HEAD_DIM {
            let value = unsafe { *query.add(tid) };
            value * value
        } else {
            0.0
        };
        let query_norm = float::rsqrt_approx_f32(
            block_sum(query_square, reduction, lane, warp_index) + RMS_EPSILON,
        );
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
        let key_norm = float::rsqrt_approx_f32(
            block_sum(key_square, reduction, lane, warp_index) + RMS_EPSILON,
        );
        if tid < HEAD_DIM {
            unsafe { *key.add(tid) *= key_norm };
        }
        thread::sync_threads();
        (1.0, 1.0)
    };

    let control = token * VALUE_HEADS + value_head;
    let decay =
        float::ex2_approx_f32(unsafe { *log_decay.add(control) } * core::f32::consts::LOG2_E);
    let beta = unsafe { *beta.add(control) };
    let value_row = unsafe { token_qkv.add(2 * QK_WIDTH + value_head * HEAD_DIM) };
    let state = if PREFILL {
        state
    } else {
        unsafe { state.add((state_row * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM) }
    };
    if PREFILL {
        // The prefill route hoists the loop-invariant normalized q/k lanes
        // into registers, consumes the prefetched value row from shared, and
        // walks its four state rows as two interleaved pairs so the two
        // dependent shuffle-reduce chains of a pair overlap. Every per-row
        // formula and its operand order are identical to the decode walk, so
        // all published values stay bit-exact.
        let column0 = lane;
        let column1 = lane + 32;
        let column2 = lane + 64;
        let column3 = lane + 96;
        let k0 = unsafe { *key.add(column0) } * key_norm;
        let k1 = unsafe { *key.add(column1) } * key_norm;
        let k2 = unsafe { *key.add(column2) } * key_norm;
        let k3 = unsafe { *key.add(column3) } * key_norm;
        let q0 = unsafe { *query.add(column0) } * query_norm;
        let q1 = unsafe { *query.add(column1) } * query_norm;
        let q2 = unsafe { *query.add(column2) } * query_norm;
        let q3 = unsafe { *query.add(column3) } * query_norm;
        let mut row = warp_index;
        while row < SPLIT_ROWS {
            let row_b = row + WARPS;
            let state_row_a = unsafe { state.add(row * HEAD_DIM) };
            let state_row_b = unsafe { state.add(row_b * HEAD_DIM) };
            let old0_a = unsafe { *state_row_a.add(column0) };
            let old0_b = unsafe { *state_row_b.add(column0) };
            let old1_a = unsafe { *state_row_a.add(column1) };
            let old1_b = unsafe { *state_row_b.add(column1) };
            let old2_a = unsafe { *state_row_a.add(column2) };
            let old2_b = unsafe { *state_row_b.add(column2) };
            let old3_a = unsafe { *state_row_a.add(column3) };
            let old3_b = unsafe { *state_row_b.add(column3) };
            let state_key_a = warp::reduce_sum_f32(float::fma_rn_f32(
                old0_a,
                k0,
                float::fma_rn_f32(old1_a, k1, float::fma_rn_f32(old2_a, k2, old3_a * k3)),
            ));
            let state_key_b = warp::reduce_sum_f32(float::fma_rn_f32(
                old0_b,
                k0,
                float::fma_rn_f32(old1_b, k1, float::fma_rn_f32(old2_b, k2, old3_b * k3)),
            ));
            let update_a = beta * (unsafe { *value.add(row_offset + row) } - decay * state_key_a);
            let update_b = beta * (unsafe { *value.add(row_offset + row_b) } - decay * state_key_b);
            let new0_a = float::fma_rn_f32(old0_a, decay, k0 * update_a);
            let new1_a = float::fma_rn_f32(old1_a, decay, k1 * update_a);
            let new2_a = float::fma_rn_f32(old2_a, decay, k2 * update_a);
            let new3_a = float::fma_rn_f32(old3_a, decay, k3 * update_a);
            let new0_b = float::fma_rn_f32(old0_b, decay, k0 * update_b);
            let new1_b = float::fma_rn_f32(old1_b, decay, k1 * update_b);
            let new2_b = float::fma_rn_f32(old2_b, decay, k2 * update_b);
            let new3_b = float::fma_rn_f32(old3_b, decay, k3 * update_b);
            unsafe {
                *state_row_a.add(column0) = new0_a;
                *state_row_a.add(column1) = new1_a;
                *state_row_a.add(column2) = new2_a;
                *state_row_a.add(column3) = new3_a;
                *state_row_b.add(column0) = new0_b;
                *state_row_b.add(column1) = new1_b;
                *state_row_b.add(column2) = new2_b;
                *state_row_b.add(column3) = new3_b;
            }
            let recurrent_a = warp::reduce_sum_f32(float::fma_rn_f32(
                new0_a,
                q0,
                float::fma_rn_f32(new1_a, q1, float::fma_rn_f32(new2_a, q2, new3_a * q3)),
            ));
            let recurrent_b = warp::reduce_sum_f32(float::fma_rn_f32(
                new0_b,
                q0,
                float::fma_rn_f32(new1_b, q1, float::fma_rn_f32(new2_b, q2, new3_b * q3)),
            ));
            if lane == 0 {
                // Publishing the scaled recurrent rows to the dead prefill
                // partials plane defers the RMS/gate epilogue to a fully
                // parallel kernel; the stored products are the identical
                // values the epilogue's reduction consumes.
                let plane = unsafe {
                    recurrent_output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + row_offset)
                };
                unsafe { *plane.add(row) = recurrent_a * DELTA_SCALE };
                unsafe { *plane.add(row_b) = recurrent_b * DELTA_SCALE };
            }
            row += 2 * WARPS;
        }
        // The value tile is re-filled by the next token's prefetch; one
        // barrier keeps that write ordered behind this token's row walk. The
        // RMS/gate/store epilogue leaves the serial loop entirely.
        thread::sync_threads();
        return;
    } else {
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
        unsafe {
            *output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + tid) =
                tcgen05::f32_to_bf16_rne(normalized * output_gate::<SIGMOID_GATE>(gate));
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gdn_recurrence<A: Arch, const TOKENS: usize, const SIGMOID_GATE: bool>(
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
        gdn_recurrence_token::<A, false, SIGMOID_GATE>(
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
            core::ptr::null_mut(),
            0,
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
    state_tile: *mut f32,
    value_tile: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let value_head = block / SPLIT_CTAS_PER_HEAD;
    let half = block - value_head * SPLIT_CTAS_PER_HEAD;
    if value_head >= VALUE_HEADS {
        return;
    }
    let row_offset = half * SPLIT_ROWS;
    let state_row = unsafe { *state_rows as usize };
    let resident = unsafe {
        state.add(
            (state_row * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM + row_offset * HEAD_DIM,
        )
    };
    let tid = thread::threadIdx_x() as usize;
    let mut element = tid;
    while element < SPLIT_ROWS * HEAD_DIM {
        unsafe { *state_tile.add(element) = *resident.add(element) };
        element += WARPS * 32;
    }
    thread::sync_threads();
    let mut token = 0;
    while token < TOKENS {
        unsafe {
            // Prefill returns before the gate; retain the Qwen3.8-27B monomorphization.
            gdn_recurrence_token::<A, true, false>(
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_tile,
                output,
                token,
                value_head,
                state_row,
                query,
                key,
                recurrent_output,
                reduction,
                value_tile,
                row_offset,
            );
        }
        token += 1;
    }
    thread::sync_threads();
    let mut element = tid;
    while element < SPLIT_ROWS * HEAD_DIM {
        unsafe { *resident.add(element) = *state_tile.add(element) };
        element += WARPS * 32;
    }
}

/// Applies the deferred RMS/gate/store epilogue to every published prefill
/// recurrent row in parallel. Each CTA replicates the sixteen-warp reduction
/// tree the serial loop used, so every emitted value stays bit-exact.
#[inline(always)]
pub(crate) unsafe fn gdn_recurrence_prefill_epilogue<
    A: Arch,
    const TOKENS: usize,
    const SIGMOID_GATE: bool,
>(
    projected: *const u16,
    norm_weight: *const u16,
    recurrent_plane: *const f32,
    output: *mut u16,
    reduction: *mut f32,
) {
    let block = thread::blockIdx_x() as usize;
    let token = block / VALUE_HEADS;
    if token >= TOKENS {
        return;
    }
    let value_head = block - token * VALUE_HEADS;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let row_base = unsafe { recurrent_plane.add(token * VALUE_WIDTH + value_head * HEAD_DIM) };

    let square = if tid < HEAD_DIM {
        let value = unsafe { *row_base.add(tid) };
        value * value
    } else {
        0.0
    };
    let inverse_rms = float::rsqrt_approx_f32(
        block_sum(square, reduction, lane, warp_index) / HEAD_DIM as f32 + RMS_EPSILON,
    );
    if tid < HEAD_DIM {
        let normalized = unsafe { *row_base.add(tid) }
            * inverse_rms
            * load_bf16(unsafe { norm_weight.add(tid) });
        let gate = load_bf16(unsafe {
            projected.add(token * A::GDN_INPUT_ROWS + A::GDN_QKV_ROWS + value_head * HEAD_DIM + tid)
        });
        unsafe {
            *output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + tid) =
                tcgen05::f32_to_bf16_rne(normalized * output_gate::<SIGMOID_GATE>(gate));
        }
    }
}

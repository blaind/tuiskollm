//! Exact Qwen3.5/Qwen3.6 FP32 GDN recurrence and gated normalization.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const KEY_HEADS: usize = Qwen35_9B::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = Qwen35_9B::LINEAR_VALUE_HEADS;
const HEAD_DIM: usize = Qwen35_9B::LINEAR_HEAD_DIM;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const WARPS: usize = 16;
const THREADS: u32 = (WARPS * 32) as u32;
const CAUSAL_ROWS: [usize; 3] = [2, 3, 4];
// Two CTAs advance each value head's state plane; every row's update stays
// wholly inside one CTA, so the split changes scheduling, never arithmetic.
const SPLIT_CTAS_PER_HEAD: usize = 2;
const SPLIT_ROWS: usize = HEAD_DIM / SPLIT_CTAS_PER_HEAD;
const PREFILL_ROWS: [usize; 3] = [32, 64, 128];
const RMS_EPSILON: f32 = 1.0e-6;
const QUERY_SCALE: f32 = 0.088_388_35;

const _: () = assert!(KEY_HEADS == 16);
const _: () = assert!(VALUE_HEADS == 32);
const _: () = assert!(HEAD_DIM == 128);
const _: () = assert!(VALUE_HEADS.is_multiple_of(KEY_HEADS));
const _: () = assert!(Qwen36Moe35B::LINEAR_KEY_HEADS == KEY_HEADS);
const _: () = assert!(Qwen36Moe35B::LINEAR_VALUE_HEADS == VALUE_HEADS);
const _: () = assert!(Qwen36Moe35B::LINEAR_HEAD_DIM == HEAD_DIM);
const _: () = assert!(Qwen36Moe35B::GDN_QK_ROWS == QK_WIDTH);
const _: () = assert!(Qwen36Moe35B::GDN_VALUE_ROWS == VALUE_WIDTH);

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn admitted_rows(rows: usize) -> bool {
    admitted_batch(rows) || PREFILL_ROWS.contains(&rows)
}

#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{float, tcgen05, thread, warp};

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
    unsafe fn recurrence_token<const PREFILL: bool>(
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
        let token_qkv = unsafe { qkv.add(token * Qwen35_9B::GDN_QKV_ROWS) };
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
            // dependent global load from every row iteration; the conversion
            // is the identical load_bf16, so the consumed value is bit-exact.
            let prefetch_value_row = unsafe { token_qkv.add(2 * QK_WIDTH + value_head * HEAD_DIM) };
            unsafe { *value.add(tid) = load_bf16(prefetch_value_row.add(tid)) };
        }
        thread::sync_threads();

        // The prefill route fuses both squared-sum reductions into one barrier
        // pair and applies the norm scalars at every use site instead of
        // writing normalized rows back to shared; each sum walks the identical
        // reduction tree and each normalized element is the identical
        // two-operand multiply, so every downstream value stays bit-exact.
        // The decode route keeps its original statement sequence untouched.
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
            // into registers, consumes the prefetched value row from shared,
            // and walks its eight state rows as four interleaved pairs so the
            // two dependent shuffle-reduce chains of a pair overlap. Every
            // per-row formula and its operand order are identical to the
            // decode walk, so all published values stay bit-exact.
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
                let update_a =
                    beta * (unsafe { *value.add(row_offset + row) } - decay * state_key_a);
                let update_b =
                    beta * (unsafe { *value.add(row_offset + row_b) } - decay * state_key_b);
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
                    // Publishing the scaled recurrent rows to the caller's
                    // plane defers the RMS/gate epilogue to a fully parallel
                    // kernel; the stored products are the identical values the
                    // epilogue's reduction consumes.
                    let plane = unsafe {
                        recurrent_output
                            .add(token * VALUE_WIDTH + value_head * HEAD_DIM + row_offset)
                    };
                    unsafe { *plane.add(row) = recurrent_a * QUERY_SCALE };
                    unsafe { *plane.add(row_b) = recurrent_b * QUERY_SCALE };
                }
                row += 2 * WARPS;
            }
            // The value tile is re-filled by the next token's prefetch; one
            // barrier keeps that write ordered behind this token's row walk.
            thread::sync_threads();
            return;
        }
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
                unsafe { *recurrent_output.add(row) = recurrent * QUERY_SCALE };
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
                projected.add(
                    token * Qwen35_9B::GDN_INPUT_ROWS
                        + Qwen35_9B::GDN_QKV_ROWS
                        + value_head * HEAD_DIM
                        + tid,
                )
            });
            let silu = gate / (1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E));
            unsafe {
                *output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + tid) =
                    tcgen05::f32_to_bf16_rne(normalized * silu);
            }
        }
    }

    /// Advances independent mapped states and emits gated normalized values.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_recurrence_exact<const TOKENS: usize>(
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) {
        static mut QUERY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut KEY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut RECURRENT_OUTPUT: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;

        // T=1 reads and writes 4.19 MiB of FP32 state in 32 independent
        // value-head CTAs. Sixteen warps give each lane four fixed columns and
        // eight rows; the state dots and output reductions keep their exact
        // order, so changing this width is a numerics change, not mere tiling.
        let block = thread::blockIdx_x() as usize;
        let token = block / VALUE_HEADS;
        if token >= TOKENS {
            return;
        }
        let value_head = block - token * VALUE_HEADS;
        let state_row = unsafe { *state_rows.add(token) as usize };
        unsafe {
            recurrence_token::<false>(
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
                core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                core::ptr::addr_of_mut!(RECURRENT_OUTPUT).cast::<f32>(),
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
                core::ptr::null_mut(),
                0,
            );
        }
    }

    /// Advances one mapped prompt state causally and emits every token.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(512, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_recurrence_prefill_exact<const TOKENS: usize>(
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        recurrent: *mut f32,
        output: *mut u16,
    ) {
        static mut QUERY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut KEY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, { 2 * WARPS }, 16> = SharedArray::UNINIT;
        static mut STATE_TILE: SharedArray<f32, { SPLIT_ROWS * HEAD_DIM }, 16> =
            SharedArray::UNINIT;
        static mut VALUE: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;

        let block = thread::blockIdx_x() as usize;
        let value_head = block / SPLIT_CTAS_PER_HEAD;
        let half = block - value_head * SPLIT_CTAS_PER_HEAD;
        if value_head >= VALUE_HEADS {
            return;
        }
        let row_offset = half * SPLIT_ROWS;
        let state_row = unsafe { *state_rows as usize };
        // Two CTAs per value head advance the causal token loop against
        // shared-resident 32KB halves of the state plane, loaded once and
        // written back once. Every row's dot products, update, and publish
        // stay wholly inside one CTA in decode's operand order, and the
        // paired epilogue kernel applies the RMS/gate/store phase over whole
        // rows, so outputs and final state stay bit-exact.
        let state_tile = core::ptr::addr_of_mut!(STATE_TILE).cast::<f32>();
        let resident = unsafe {
            state.add(
                (state_row * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM
                    + row_offset * HEAD_DIM,
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
                recurrence_token::<true>(
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
                    core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                    core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                    recurrent,
                    core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
                    core::ptr::addr_of_mut!(VALUE).cast::<f32>(),
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

    /// Applies the deferred prefill RMS/gate/store epilogue in parallel.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_recurrence_prefill_epilogue_exact<const TOKENS: usize>(
        projected: *const u16,
        norm_weight: *const u16,
        recurrent: *const f32,
        output: *mut u16,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let reduction = core::ptr::addr_of_mut!(REDUCTION).cast::<f32>();

        // One CTA per (token, value head) replicates the serial loop's
        // sixteen-warp RMS reduction tree over the published recurrent row,
        // so the emitted gated values are bit-exact.
        let block = thread::blockIdx_x() as usize;
        let token = block / VALUE_HEADS;
        if token >= TOKENS {
            return;
        }
        let value_head = block - token * VALUE_HEADS;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let row_base = unsafe { recurrent.add(token * VALUE_WIDTH + value_head * HEAD_DIM) };

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
                projected.add(
                    token * Qwen35_9B::GDN_INPUT_ROWS
                        + Qwen35_9B::GDN_QKV_ROWS
                        + value_head * HEAD_DIM
                        + tid,
                )
            });
            let silu = gate / (1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E));
            unsafe {
                *output.add(token * VALUE_WIDTH + value_head * HEAD_DIM + tid) =
                    tcgen05::f32_to_bf16_rne(normalized * silu);
            }
        }
    }

    /// Advances one mapped state across an exact short verification span.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_recurrence_causal_exact<const TOKENS: usize>(
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) {
        static mut QUERY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut KEY: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut RECURRENT_OUTPUT: SharedArray<f32, HEAD_DIM, 16> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;

        let value_head = thread::blockIdx_x() as usize;
        if value_head >= VALUE_HEADS {
            return;
        }
        let state_row = unsafe { *state_rows as usize };
        // K=4 replaces 128 mutually racing decode CTAs with 32 head CTAs.
        // Each CTA performs the same four recurrence_token calls in order, so
        // every state FMA, reduction, and BF16 publication remains unchanged.
        let mut token = 0;
        while token < TOKENS {
            unsafe {
                recurrence_token::<false>(
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
                    core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                    core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                    core::ptr::addr_of_mut!(RECURRENT_OUTPUT).cast::<f32>(),
                    core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
                    core::ptr::null_mut(),
                    0,
                );
            }
            token += 1;
        }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__qwen35_gdn_recurrence_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = module
            .prepare_qwen35_gdn_recurrence_exact::<TOKENS>(LaunchConfig1D::new(
                (TOKENS * VALUE_HEADS) as u32,
                THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing 2,048-wide GDN recurrence", source))?;

        Ok(Self { launch })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_gdn_recurrence_exact::<TOKENS>(
                stream,
                &self.launch,
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                output,
            )
            .map_err(|source| GpuError::launch("launching 2,048-wide GDN recurrence", source))
    }
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__qwen35_gdn_recurrence_prefill_exact_CudaKernel<TOKENS>>,
    epilogue:
        PreparedLaunch<kernels::__qwen35_gdn_recurrence_prefill_epilogue_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN recurrence prefill route T={TOKENS} is not admitted"
            )));
        }
        let launch = module
            .prepare_qwen35_gdn_recurrence_prefill_exact::<TOKENS>(LaunchConfig1D::new(
                (VALUE_HEADS * SPLIT_CTAS_PER_HEAD) as u32,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing 2,048-wide GDN prompt recurrence", source)
            })?;
        let epilogue_blocks = u32::try_from(TOKENS * VALUE_HEADS)
            .map_err(|_| GpuError::invalid_launch("GDN prompt epilogue grid exceeds u32"))?;
        let epilogue =
            module
                .prepare_qwen35_gdn_recurrence_prefill_epilogue_exact::<TOKENS>(
                    LaunchConfig1D::new(epilogue_blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing 2,048-wide GDN prompt epilogue", source)
                })?;

        Ok(Self { launch, epilogue })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        recurrent: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        // The epilogue reads the plane the serial pass just published on the
        // same stream, so ordering is inherent.
        module
            .qwen35_gdn_recurrence_prefill_exact::<TOKENS>(
                stream,
                &self.launch,
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                recurrent,
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching 2,048-wide GDN prompt recurrence", source)
            })?;
        module
            .qwen35_gdn_recurrence_prefill_epilogue_exact::<TOKENS>(
                stream,
                &self.epilogue,
                projected,
                norm_weight,
                recurrent.cast_const(),
                output,
            )
            .map_err(|source| GpuError::launch("launching 2,048-wide GDN prompt epilogue", source))
    }
}

struct PreparedCausalRoute<const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__qwen35_gdn_recurrence_causal_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedCausalRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !CAUSAL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN recurrence causal route K={TOKENS} is not admitted"
            )));
        }
        let launch = module
            .prepare_qwen35_gdn_recurrence_causal_exact::<TOKENS>(LaunchConfig1D::new(
                VALUE_HEADS as u32,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing 2,048-wide causal GDN recurrence", source)
            })?;
        Ok(Self { launch })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_gdn_recurrence_causal_exact::<TOKENS>(
                stream,
                &self.launch,
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching 2,048-wide causal GDN recurrence", source)
            })
    }
}

/// Prepared recurrent-state routes for exact Qwen3.5/Qwen3.6 row counts.
#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_gdn_recurrence_decode),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct Qwen35GdnRecurrenceDecodeRoutes {
    #[route(1)]
    b1: PreparedRoute<1>,
    #[route(2)]
    b2: PreparedRoute<2>,
    #[route(3)]
    b3: PreparedRoute<3>,
    #[route(4)]
    b4: PreparedRoute<4>,
    #[route(5)]
    b5: PreparedRoute<5>,
    #[route(6)]
    b6: PreparedRoute<6>,
    #[route(7)]
    b7: PreparedRoute<7>,
    #[route(8)]
    b8: PreparedRoute<8>,
}
#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_gdn_recurrence_prefill),
    required(32, 64, 128),
    inventory(false)
)]
struct Qwen35GdnRecurrencePrefillRoutes {
    #[route(32)]
    t32: PreparedPrefillRoute<32>,
    #[route(64)]
    t64: PreparedPrefillRoute<64>,
    #[route(128)]
    t128: PreparedPrefillRoute<128>,
}
#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_gdn_recurrence_causal),
    required(2, 3, 4),
    inventory(false)
)]
struct Qwen35GdnRecurrenceCausalRoutes {
    #[route(2)]
    c2: PreparedCausalRoute<2>,
    #[route(3)]
    c3: PreparedCausalRoute<3>,
    #[route(4)]
    c4: PreparedCausalRoute<4>,
}
/// Prepared recurrent-state routes for exact Qwen3.5/Qwen3.6 row counts.
pub struct Qwen35GdnRecurrenceOp {
    module: kernels::LoadedModule,
    decode_routes: Qwen35GdnRecurrenceDecodeRoutes,
    prefill_routes: Qwen35GdnRecurrencePrefillRoutes,
    causal_routes: Qwen35GdnRecurrenceCausalRoutes,
}

impl Qwen35GdnRecurrenceOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_gdn_recurrence_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the 2,048-wide GDN recurrence module", source)
        })?;

        Ok(Self {
            decode_routes: Qwen35GdnRecurrenceDecodeRoutes::prepare(&module)?,
            prefill_routes: Qwen35GdnRecurrencePrefillRoutes::prepare(&module)?,
            causal_routes: Qwen35GdnRecurrenceCausalRoutes::prepare(&module)?,
            module,
        })
    }

    /// Advances mapped FP32 state and emits gated BF16 values.
    ///
    /// # Safety
    ///
    /// `qkv` and `projected` cover BF16 `[rows, 8_192]` and `[rows,
    /// 12_288]`. Controls cover FP32 `[rows, 32]`; `norm_weight` covers 128
    /// BF16 values. Every state row is within `[rows, 32, 128, 128]`;
    /// `recurrent` covers FP32 `[rows, 4_096]`; and `output` covers BF16
    /// `[rows, 4_096]`. Prompt routes read the first state-row index, advance
    /// that row across the contiguous sequence, and use `recurrent` as their
    /// intermediate output plane.
    /// Allocations are aligned, disjoint, live through completion, and belong
    /// to the stream's context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        recurrent: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_rows(rows) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN recurrence row count {rows} is outside 1..={MAX_BATCH}, 32, 64, and 128"
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        qkv,
                        projected,
                        log_decay,
                        beta,
                        norm_weight,
                        state_rows,
                        state,
                        output,
                    )
                }
            };
        }

        macro_rules! launch_prefill {
            ($route:ident) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        qkv,
                        projected,
                        log_decay,
                        beta,
                        norm_weight,
                        state_rows,
                        state,
                        recurrent,
                        output,
                    )
                }
            };
        }

        if rows <= MAX_BATCH {
            dispatch_qwen35_gdn_recurrence_decode!(&self.decode_routes, rows, |route| launch!(route), else => unreachable!())
        } else {
            dispatch_qwen35_gdn_recurrence_prefill!(&self.prefill_routes, rows, |route| launch_prefill!(route), else => Err(GpuError::invalid_launch(format!("2,048-wide GDN recurrence row count {rows} is outside 1..={MAX_BATCH}, 32, 64, and 128"))) )
        }
    }

    /// Advances one mapped FP32 state causally across an exact `K=2..4` transaction.
    ///
    /// # Safety
    ///
    /// The pointer contract matches [`Self::launch`], but `state_rows[0]`
    /// selects the single state advanced by every ordered token.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_causal(
        &self,
        stream: &CudaStream,
        rows: usize,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        qkv,
                        projected,
                        log_decay,
                        beta,
                        norm_weight,
                        state_rows,
                        state,
                        output,
                    )
                }
            };
        }

        dispatch_qwen35_gdn_recurrence_causal!(&self.causal_routes, rows, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN recurrence causal row count {rows} is outside 2..=4"
            ))) )
    }
}

/// Qwen3.6 uses the exact Qwen3.5 recurrent-state binary route.
///
/// Both profiles have 16 Q/K heads, 32 value heads, width-128 state heads,
/// and the same Q/K/value/gate row mapping. Compile-time assertions above
/// keep this alias tied to that complete arithmetic contract.
pub type Qwen36GdnRecurrenceOp = Qwen35GdnRecurrenceOp;

/// PTX symbols retained for every exact Qwen3.5/Qwen3.6 recurrence route.
pub(crate) fn qwen35_gdn_recurrence_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<1>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<5>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<6>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<7>(),
        kernels::qwen35_gdn_recurrence_exact_ptx_name::<8>(),
        kernels::qwen35_gdn_recurrence_causal_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_recurrence_causal_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_recurrence_causal_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<32>(),
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<64>(),
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<128>(),
        kernels::qwen35_gdn_recurrence_prefill_epilogue_exact_ptx_name::<32>(),
        kernels::qwen35_gdn_recurrence_prefill_epilogue_exact_ptx_name::<64>(),
        kernels::qwen35_gdn_recurrence_prefill_epilogue_exact_ptx_name::<128>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        CAUSAL_ROWS, HEAD_DIM, KEY_HEADS, MAX_BATCH, PREFILL_ROWS, QK_WIDTH, THREADS, VALUE_HEADS,
        VALUE_WIDTH, admitted_batch, admitted_rows, qwen35_gdn_recurrence_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen36Moe35B};

    #[test]
    fn geometry_routing_and_inventory_are_exact() {
        assert_eq!(THREADS, 512);
        assert_eq!(VALUE_HEADS / KEY_HEADS, 2);
        assert_eq!(VALUE_HEADS * HEAD_DIM * HEAD_DIM, 524_288);
        assert_eq!(Qwen36Moe35B::LINEAR_KEY_HEADS, KEY_HEADS);
        assert_eq!(Qwen36Moe35B::LINEAR_VALUE_HEADS, VALUE_HEADS);
        assert_eq!(Qwen36Moe35B::LINEAR_HEAD_DIM, HEAD_DIM);
        assert_eq!(Qwen36Moe35B::GDN_QK_ROWS, QK_WIDTH);
        assert_eq!(Qwen36Moe35B::GDN_VALUE_ROWS, VALUE_WIDTH);
        for (batch, expected) in [(0, false), (1, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
        for (rows, expected) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (32, true),
            (64, true),
            (128, true),
            (129, false),
        ] {
            assert_eq!(admitted_rows(rows), expected, "rows={rows}");
        }

        let names = qwen35_gdn_recurrence_ptx_names();
        assert_eq!(CAUSAL_ROWS, [2, 3, 4]);
        assert_eq!(PREFILL_ROWS, [32, 64, 128]);
        assert_eq!(
            names.len(),
            MAX_BATCH + CAUSAL_ROWS.len() + 2 * PREFILL_ROWS.len()
        );
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

//! Exact Qwen3.5/Qwen3.6 FP32 GDN recurrence and gated normalization.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const KEY_HEADS: usize = Qwen35_9B::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = Qwen35_9B::LINEAR_VALUE_HEADS;
const HEAD_DIM: usize = Qwen35_9B::LINEAR_HEAD_DIM;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const WARPS: usize = 16;
const THREADS: u32 = (WARPS * 32) as u32;
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

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn recurrence_token(
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
        let token_qkv = unsafe { qkv.add(token * Qwen35_9B::GDN_QKV_ROWS) };
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

        let control = token * VALUE_HEADS + value_head;
        let decay =
            float::ex2_approx_f32(unsafe { *log_decay.add(control) } * core::f32::consts::LOG2_E);
        let beta = unsafe { *beta.add(control) };
        let value_row = unsafe { token_qkv.add(2 * QK_WIDTH + value_head * HEAD_DIM) };
        let state =
            unsafe { state.add((state_row * VALUE_HEADS + value_head) * HEAD_DIM * HEAD_DIM) };
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
            recurrence_token(
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
            );
        }
    }

    /// Advances one mapped prompt state causally and emits every token.
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
    pub fn qwen35_gdn_recurrence_prefill_exact<const TOKENS: usize>(
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
        // Flattening T=128 would launch 4,096 CTAs that race one recurrent
        // state. Thirty-two head CTAs instead advance 128 tokens serially;
        // each token retains decode's sixteen-warp row and reduction order, so
        // the only changed dimension is the required causal token schedule.
        let mut token = 0;
        while token < TOKENS {
            unsafe {
                recurrence_token(
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
                VALUE_HEADS as u32,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing 2,048-wide GDN prompt recurrence", source)
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
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching 2,048-wide GDN prompt recurrence", source)
            })
    }
}

/// Prepared recurrent-state routes for exact Qwen3.5/Qwen3.6 row counts.
pub struct Qwen35GdnRecurrenceOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
}

impl Qwen35GdnRecurrenceOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_gdn_recurrence_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the 2,048-wide GDN recurrence module", source)
        })?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Advances mapped FP32 state and emits gated BF16 values.
    ///
    /// # Safety
    ///
    /// `qkv` and `projected` cover BF16 `[rows, 8_192]` and `[rows,
    /// 12_288]`. Controls cover FP32 `[rows, 32]`; `norm_weight` covers 128
    /// BF16 values. Every state row is within `[rows, 32, 128, 128]`, and
    /// `output` covers BF16 `[rows, 4_096]`. Prompt routes read the first
    /// state-row index and advance that row across the contiguous sequence.
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
                    self.$route.launch(
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

        match rows {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            _ => unreachable!(),
        }
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
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<32>(),
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<64>(),
        kernels::qwen35_gdn_recurrence_prefill_exact_ptx_name::<128>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        HEAD_DIM, KEY_HEADS, MAX_BATCH, PREFILL_ROWS, QK_WIDTH, THREADS, VALUE_HEADS, VALUE_WIDTH,
        admitted_batch, admitted_rows, qwen35_gdn_recurrence_ptx_names,
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
        assert_eq!(PREFILL_ROWS, [32, 64, 128]);
        assert_eq!(names.len(), MAX_BATCH + PREFILL_ROWS.len());
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

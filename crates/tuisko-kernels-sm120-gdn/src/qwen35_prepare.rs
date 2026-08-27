//! Exact Qwen3.5/Qwen3.6 GDN control and causal-convolution preparation.

use crate::device::gdn_prepare::{gdn_convolution_prefill, gdn_convolution_prefill_history};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const CONTROL_ROWS: usize = Qwen35_9B::GDN_CONTROL_ROWS;
const CONTROL_STRIDE: usize = 128;
const PROJECTED_STRIDE: usize = Qwen35_9B::GDN_INPUT_ROWS;
const QKV_ROWS: usize = Qwen35_9B::GDN_QKV_ROWS;
const HISTORY_TAPS: usize = Qwen35_9B::LINEAR_CONV_KERNEL_DIM - 1;
const THREADS: u32 = 256;
const CTAS_PER_TOKEN: usize = QKV_ROWS / THREADS as usize;
const CAUSAL_ROWS: [usize; 3] = [2, 3, 4];
const PREFILL_ROWS: [usize; 3] = [32, 64, 128];

// The input projection already emits 64 A/B controls, so a standalone control
// node would add B CTAs beside 32*B convolution CTAs. Threads 0..63 of the
// first convolution CTA handle those scalars instead. The four-tap channel FMA
// order and every history update remain identical; only the extra launch ends.
const _: () = assert!(CONTROL_ROWS == 32);
const _: () = assert!(CONTROL_STRIDE == 128);
const _: () = assert!(PROJECTED_STRIDE == 12_288);
const _: () = assert!(QKV_ROWS == 8_192);
const _: () = assert!(HISTORY_TAPS == 3);
const _: () = assert!(CTAS_PER_TOKEN == 32);
const _: () = assert!(Qwen36Moe35B::GDN_CONTROL_ROWS == CONTROL_ROWS);
const _: () = assert!(Qwen36Moe35B::GDN_INPUT_ROWS == PROJECTED_STRIDE);
const _: () = assert!(Qwen36Moe35B::GDN_QKV_ROWS == QKV_ROWS);
const _: () = assert!(Qwen36Moe35B::LINEAR_CONV_KERNEL_DIM - 1 == HISTORY_TAPS);

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
    use cuda_device::{float, tcgen05, thread};

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

    /// Converts projected controls and advances the width-four causal history.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_prepare_exact<const TOKENS: usize>(
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block % CTAS_PER_TOKEN;
        let token = block / CTAS_PER_TOKEN;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        if tile == 0 && tid < 2 * CONTROL_ROWS {
            let row = tid & (CONTROL_ROWS - 1);
            let raw = unsafe { bf16(projected_controls.add(token * CONTROL_STRIDE + tid)) };
            if tid < CONTROL_ROWS {
                let control = raw + unsafe { bf16(dt_bias.add(row)) };
                unsafe {
                    *log_decay.add(token * CONTROL_ROWS + row) =
                        -fast_exp(bf16(a_log.add(row))) * softplus(control);
                }
            } else {
                unsafe {
                    *beta.add(token * CONTROL_ROWS + row) = sigmoid(raw);
                }
            }
        }

        let channel = tile * THREADS as usize + tid;
        let state_row = unsafe { *state_rows.add(token) as usize };
        let history = unsafe { history.add((state_row * QKV_ROWS + channel) * HISTORY_TAPS) };
        let current = unsafe { *projected.add(token * PROJECTED_STRIDE + channel) };
        let h0 = unsafe { *history };
        let h1 = unsafe { *history.add(1) };
        let h2 = unsafe { *history.add(2) };

        unsafe {
            *history = h1;
            *history.add(1) = h2;
            *history.add(2) = current;
        }

        let weights =
            unsafe { convolution_weights.add(channel * Qwen35_9B::LINEAR_CONV_KERNEL_DIM) };
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

        unsafe {
            *convolved.add(token * QKV_ROWS + channel) =
                tcgen05::f32_to_bf16_rne(sum * sigmoid(sum));
        }
    }

    /// Converts controls and applies one from-empty causal prompt convolution.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_prepare_prefill_exact<const TOKENS: usize>(
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block % CTAS_PER_TOKEN;
        let token = block / CTAS_PER_TOKEN;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        if tile == 0 && tid < 2 * CONTROL_ROWS {
            let row = tid & (CONTROL_ROWS - 1);
            let raw = unsafe { bf16(projected_controls.add(token * CONTROL_STRIDE + tid)) };
            if tid < CONTROL_ROWS {
                let control = raw + unsafe { bf16(dt_bias.add(row)) };
                unsafe {
                    *log_decay.add(token * CONTROL_ROWS + row) =
                        -fast_exp(bf16(a_log.add(row))) * softplus(control);
                }
            } else {
                unsafe {
                    *beta.add(token * CONTROL_ROWS + row) = sigmoid(raw);
                }
            }
        }

        // T=128 exposes 4,096 independent 256-thread channel CTAs. Each
        // thread retains decode's ordered four-tap FMA, but reads the preceding
        // represented prompt rows instead of racing 128 in-place history shifts.
        unsafe {
            gdn_convolution_prefill::<Qwen35_9B, TOKENS>(
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            );
        }
    }

    /// Publishes the last three represented prompt values to mapped history.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_prepare_prefill_history_exact<const TOKENS: usize>(
        projected: *const u16,
        state_rows: *const u32,
        history: *mut u16,
    ) {
        // Thirty-two CTAs publish one owner per 8,192-wide channel after all
        // prompt convolution reads complete; arithmetic is unchanged because
        // this node only copies the final three represented BF16 source words.
        unsafe {
            gdn_convolution_prefill_history::<Qwen35_9B, TOKENS>(projected, state_rows, history);
        }
    }

    /// Converts controls and applies an exact short causal verification span.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_prepare_causal_exact<const TOKENS: usize>(
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block % CTAS_PER_TOKEN;
        let token = block / CTAS_PER_TOKEN;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        if tile == 0 && tid < 2 * CONTROL_ROWS {
            let row = tid & (CONTROL_ROWS - 1);
            let raw = unsafe { bf16(projected_controls.add(token * CONTROL_STRIDE + tid)) };
            if tid < CONTROL_ROWS {
                let control = raw + unsafe { bf16(dt_bias.add(row)) };
                unsafe {
                    *log_decay.add(token * CONTROL_ROWS + row) =
                        -fast_exp(bf16(a_log.add(row))) * softplus(control);
                }
            } else {
                unsafe {
                    *beta.add(token * CONTROL_ROWS + row) = sigmoid(raw);
                }
            }
        }

        // K=4 uses 128 independent channel CTAs, the same count as four B=1
        // nodes, but reads the preceding represented rows without racing four
        // in-place history shifts. The ordered four-tap FMA is unchanged.
        unsafe {
            gdn_convolution_prefill::<Qwen35_9B, TOKENS>(
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            );
        }
    }

    /// Publishes a short verified span's last three represented values.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_gdn_prepare_causal_history_exact<const TOKENS: usize>(
        projected: *const u16,
        state_rows: *const u32,
        history: *mut u16,
    ) {
        unsafe {
            gdn_convolution_prefill_history::<Qwen35_9B, TOKENS>(projected, state_rows, history);
        }
    }
}

struct PreparedRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_gdn_prepare_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            prepare: module
                .prepare_qwen35_gdn_prepare_exact::<TOKENS>(LaunchConfig1D::new(
                    (TOKENS * CTAS_PER_TOKEN) as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing 2,048-wide GDN controls", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_gdn_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                projected_controls,
                a_log,
                dt_bias,
                projected,
                convolution_weights,
                state_rows,
                history,
                log_decay,
                beta,
                convolved,
            )
            .map_err(|source| GpuError::launch("launching 2,048-wide GDN controls", source))
    }
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_gdn_prepare_prefill_exact_CudaKernel<TOKENS>>,
    history: PreparedLaunch<kernels::__qwen35_gdn_prepare_prefill_history_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN prepare prefill route T={TOKENS} is not admitted"
            )));
        }

        Ok(Self {
            prepare: module
                .prepare_qwen35_gdn_prepare_prefill_exact::<TOKENS>(LaunchConfig1D::new(
                    (TOKENS * CTAS_PER_TOKEN) as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing 2,048-wide GDN prompt convolution", source)
                })?,
            history: module
                .prepare_qwen35_gdn_prepare_prefill_history_exact::<TOKENS>(LaunchConfig1D::new(
                    CTAS_PER_TOKEN as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing 2,048-wide GDN prompt history", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_gdn_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                projected_controls,
                a_log,
                dt_bias,
                projected,
                convolution_weights,
                state_rows,
                history,
                log_decay,
                beta,
                convolved,
            )
            .map_err(|source| {
                GpuError::launch("launching 2,048-wide GDN prompt convolution", source)
            })?;
        module
            .qwen35_gdn_prepare_prefill_history_exact::<TOKENS>(
                stream,
                &self.history,
                projected,
                state_rows,
                history,
            )
            .map_err(|source| GpuError::launch("launching 2,048-wide GDN prompt history", source))
    }
}

struct PreparedCausalRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_gdn_prepare_causal_exact_CudaKernel<TOKENS>>,
    history: PreparedLaunch<kernels::__qwen35_gdn_prepare_causal_history_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedCausalRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !CAUSAL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN prepare causal route K={TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            prepare: module
                .prepare_qwen35_gdn_prepare_causal_exact::<TOKENS>(LaunchConfig1D::new(
                    (TOKENS * CTAS_PER_TOKEN) as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing 2,048-wide causal GDN convolution", source)
                })?,
            history: module
                .prepare_qwen35_gdn_prepare_causal_history_exact::<TOKENS>(LaunchConfig1D::new(
                    CTAS_PER_TOKEN as u32,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing 2,048-wide causal GDN history", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_gdn_prepare_causal_exact::<TOKENS>(
                stream,
                &self.prepare,
                projected_controls,
                a_log,
                dt_bias,
                projected,
                convolution_weights,
                state_rows,
                history,
                log_decay,
                beta,
                convolved,
            )
            .map_err(|source| {
                GpuError::launch("launching 2,048-wide causal GDN convolution", source)
            })?;
        module
            .qwen35_gdn_prepare_causal_history_exact::<TOKENS>(
                stream,
                &self.history,
                projected,
                state_rows,
                history,
            )
            .map_err(|source| GpuError::launch("launching causal GDN history", source))
    }
}

/// PTX symbols retained for every exact Qwen3.5/Qwen3.6 GDN prepare route.
pub(crate) fn qwen35_gdn_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_gdn_prepare_exact_ptx_name::<1>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<5>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<6>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<7>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<8>(),
        kernels::qwen35_gdn_prepare_causal_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_prepare_causal_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_prepare_causal_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen35_gdn_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen35_gdn_prepare_prefill_exact_ptx_name::<128>(),
        kernels::qwen35_gdn_prepare_causal_history_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_prepare_causal_history_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_prepare_causal_history_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_prepare_prefill_history_exact_ptx_name::<32>(),
        kernels::qwen35_gdn_prepare_prefill_history_exact_ptx_name::<64>(),
        kernels::qwen35_gdn_prepare_prefill_history_exact_ptx_name::<128>(),
    ]
}

/// Prepared control and convolution routes for exact Qwen3.5/Qwen3.6 rows.
#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_gdn_prepare),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct Qwen35GdnPrepareRoutes {
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
    dispatch(dispatch_qwen35_gdn_prepare_causal),
    required(2, 3, 4),
    inventory(false)
)]
struct Qwen35GdnPrepareCausalRoutes {
    #[route(2)]
    c2: PreparedCausalRoute<2>,
    #[route(3)]
    c3: PreparedCausalRoute<3>,
    #[route(4)]
    c4: PreparedCausalRoute<4>,
}
/// Prepared control and convolution routes for exact Qwen3.5/Qwen3.6 rows.
pub struct Qwen35GdnPrepareOp {
    module: kernels::LoadedModule,
    routes: Qwen35GdnPrepareRoutes,
    causal_routes: Qwen35GdnPrepareCausalRoutes,
}

impl Qwen35GdnPrepareOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_gdn_prepare_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the 2,048-wide GDN prepare module", source)
        })?;

        Ok(Self {
            routes: Qwen35GdnPrepareRoutes::prepare(&module)?,
            causal_routes: Qwen35GdnPrepareCausalRoutes::prepare(&module)?,
            module,
        })
    }

    /// Converts projected A/B controls and advances mapped causal history.
    ///
    /// # Safety
    ///
    /// Controls cover padded BF16 `[rows, 128]`; `a_log` and `dt_bias` each
    /// cover 32 BF16 values. The projection covers BF16 `[rows, 12_288]`,
    /// convolution weights cover BF16 `[8_192, 4]`, and every state row is
    /// within the caller-owned `[rows, 8_192, 3]` history. Outputs cover FP32
    /// `[rows, 32]`, `[rows, 32]`, and BF16 `[rows, 8_192]`. Prompt routes read
    /// the first state-row index and advance that row across the contiguous
    /// sequence. Allocations are aligned, disjoint, live through completion,
    /// and belong to the stream's context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_rows(rows) {
            return Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN prepare row count {rows} is outside 1..={MAX_BATCH}, 32, 64, and 128"
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        projected_controls,
                        a_log,
                        dt_bias,
                        projected,
                        convolution_weights,
                        state_rows,
                        history,
                        log_decay,
                        beta,
                        convolved,
                    )
                }
            };
        }

        dispatch_qwen35_gdn_prepare!(&self.routes, rows, |route| launch!(route), else => Err(GpuError::invalid_launch(format!("2,048-wide GDN prepare row count {rows} is outside 1..={MAX_BATCH}, 32, 64, and 128"))) )
    }

    /// Advances one mapped history row causally across an exact `K=2..4` transaction.
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
        projected_controls: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        log_decay: *mut f32,
        beta: *mut f32,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    $route.launch(
                        &self.module,
                        stream,
                        projected_controls,
                        a_log,
                        dt_bias,
                        projected,
                        convolution_weights,
                        state_rows,
                        history,
                        log_decay,
                        beta,
                        convolved,
                    )
                }
            };
        }

        dispatch_qwen35_gdn_prepare_causal!(&self.causal_routes, rows, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "2,048-wide GDN prepare causal row count {rows} is outside 2..=4"
            ))) )
    }
}

/// Qwen3.6 uses the exact Qwen3.5 control/convolution binary route.
///
/// Both profiles have 32 control rows, 12,288 projected rows, 8,192 Q/K/V
/// rows, and a width-four causal history. Compile-time assertions above keep
/// this alias from silently widening to a merely similar geometry.
pub type Qwen36GdnPrepareOp = Qwen35GdnPrepareOp;

#[cfg(test)]
mod tests {
    use super::{
        CAUSAL_ROWS, CONTROL_ROWS, CONTROL_STRIDE, CTAS_PER_TOKEN, MAX_BATCH, PREFILL_ROWS,
        PROJECTED_STRIDE, QKV_ROWS, admitted_batch, admitted_rows, qwen35_gdn_prepare_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen36Moe35B};

    #[test]
    fn geometry_routing_and_inventory_are_exact() {
        assert_eq!(CONTROL_ROWS, 32);
        assert_eq!(CONTROL_STRIDE, 128);
        assert_eq!(PROJECTED_STRIDE, 12_288);
        assert_eq!(QKV_ROWS, 8_192);
        assert_eq!(CTAS_PER_TOKEN, 32);
        assert_eq!(Qwen36Moe35B::GDN_CONTROL_ROWS, CONTROL_ROWS);
        assert_eq!(Qwen36Moe35B::GDN_INPUT_ROWS, PROJECTED_STRIDE);
        assert_eq!(Qwen36Moe35B::GDN_QKV_ROWS, QKV_ROWS);
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

        let names = qwen35_gdn_prepare_ptx_names();
        assert_eq!(CAUSAL_ROWS, [2, 3, 4]);
        assert_eq!(PREFILL_ROWS, [32, 64, 128]);
        assert_eq!(
            names.len(),
            MAX_BATCH + 2 * CAUSAL_ROWS.len() + 2 * PREFILL_ROWS.len()
        );
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

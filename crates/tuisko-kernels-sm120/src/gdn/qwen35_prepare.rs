//! Exact Qwen3.5 GDN control and causal-convolution preparation.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const CONTROL_ROWS: usize = Qwen35_9B::GDN_CONTROL_ROWS;
const CONTROL_STRIDE: usize = 128;
const PROJECTED_STRIDE: usize = Qwen35_9B::GDN_INPUT_ROWS;
const QKV_ROWS: usize = Qwen35_9B::GDN_QKV_ROWS;
const HISTORY_TAPS: usize = Qwen35_9B::LINEAR_CONV_KERNEL_DIM - 1;
const THREADS: u32 = 256;
const CTAS_PER_TOKEN: usize = QKV_ROWS / THREADS as usize;

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

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
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
                .map_err(|source| GpuError::launch("preparing Qwen3.5 GDN controls", source))?,
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
            .map_err(|source| GpuError::launch("launching Qwen3.5 GDN controls", source))
    }
}

/// PTX symbols retained for every exact Qwen3.5 GDN prepare route.
pub(crate) fn qwen35_gdn_prepare_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen35_gdn_prepare_exact_ptx_name::<1>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<2>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<3>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<4>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<5>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<6>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<7>(),
        kernels::qwen35_gdn_prepare_exact_ptx_name::<8>(),
    ]
}

/// Prepared control and convolution routes for exact Qwen3.5 batches.
pub struct Qwen35GdnPrepareOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
}

impl Qwen35GdnPrepareOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_gdn_prepare_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.5 GDN prepare module", source))?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            module,
        })
    }

    /// Converts projected A/B controls and advances mapped causal history.
    ///
    /// # Safety
    ///
    /// Controls cover padded BF16 `[batch, 128]`; `a_log` and `dt_bias` each
    /// cover 32 BF16 values. The projection covers BF16 `[batch, 12_288]`,
    /// convolution weights cover BF16 `[8_192, 4]`, and every state row is
    /// within the caller-owned `[rows, 8_192, 3]` history. Outputs cover FP32
    /// `[batch, 32]`, `[batch, 32]`, and BF16 `[batch, 8_192]`. Allocations are
    /// aligned, disjoint, live through completion, and belong to the stream's
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
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
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 GDN prepare batch {batch} is outside the exact range 1..={MAX_BATCH}"
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
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

        match batch {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_ROWS, CONTROL_STRIDE, CTAS_PER_TOKEN, MAX_BATCH, PROJECTED_STRIDE, QKV_ROWS,
        admitted_batch, qwen35_gdn_prepare_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_routing_and_inventory_are_exact() {
        assert_eq!(CONTROL_ROWS, 32);
        assert_eq!(CONTROL_STRIDE, 128);
        assert_eq!(PROJECTED_STRIDE, 12_288);
        assert_eq!(QKV_ROWS, 8_192);
        assert_eq!(CTAS_PER_TOKEN, 32);
        for (batch, expected) in [(0, false), (1, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }

        let names = qwen35_gdn_prepare_ptx_names();
        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

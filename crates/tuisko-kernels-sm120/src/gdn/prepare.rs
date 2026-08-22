//! Exact-batch GDN control and causal-convolution preparation.

use crate::Sm120Arch;
use crate::device::gdn_prepare::{gdn_control, gdn_convolution};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const CONTROL_WARPS: usize = 16;
const CONTROL_THREADS: u32 = (CONTROL_WARPS * 32) as u32;
const CONTROL_ROWS_PER_CTA: usize = CONTROL_WARPS / 2;
const CONV_THREADS: u32 = 256;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::HIDDEN == 0
        || !A::HIDDEN.is_multiple_of(64)
        || A::GDN_CONTROL_ROWS == 0
        || !(2 * A::GDN_CONTROL_ROWS).is_multiple_of(CONTROL_ROWS_PER_CTA)
        || A::GDN_QKV_ROWS == 0
        || A::LINEAR_CONV_KERNEL_DIM != 4
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the GDN prepare schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Computes the two control vectors for one exact batch.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_control_exact<A: Arch, const TOKENS: usize>(
        input: *const u32,
        control_weights: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, CONTROL_WARPS, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUMS).cast::<f32>();

        // T=1 is 12 CTAs on 170 SMs and reads about 960 KiB in 9.856 us,
        // roughly 99 GB/s: it is latency-starved. Two warps retain each row's
        // `column += 64` walk and fixed pairwise combine, so the measured launch
        // shape preserves every output's accumulation order.
        unsafe {
            gdn_control::<A, TOKENS>(
                input,
                control_weights,
                a_log,
                dt_bias,
                log_decay,
                beta,
                warp_sums,
            );
        }
    }

    /// Updates causal history and applies the represented width-4 convolution.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_convolution_exact<A: Arch, const TOKENS: usize>(
        projected: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        output: *mut u16,
    ) {
        // Each CTA covers 256 adjacent channels, giving 40 CTAs per token for
        // the exact 10,240-wide plane. One thread owns one output and its three
        // history words, preserving the scalar four-tap FMA order.
        unsafe {
            gdn_convolution::<A, TOKENS>(projected, weights, state_rows, history, output);
        }
    }
}

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    control: PreparedLaunch<kernels::__gdn_control_exact_CudaKernel<A, TOKENS>>,
    convolution: PreparedLaunch<kernels::__gdn_convolution_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let control_blocks = u32::try_from(TOKENS * 2 * A::GDN_CONTROL_ROWS / CONTROL_ROWS_PER_CTA)
            .map_err(|_| GpuError::invalid_launch("GDN control grid exceeds u32"))?;
        let convolution_blocks =
            u32::try_from((TOKENS * A::GDN_QKV_ROWS).div_ceil(CONV_THREADS as usize))
                .map_err(|_| GpuError::invalid_launch("GDN convolution grid exceeds u32"))?;

        Ok(Self {
            control: module
                .prepare_gdn_control_exact::<A, TOKENS>(LaunchConfig1D::new(
                    control_blocks,
                    CONTROL_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN control", source))?,
            convolution: module
                .prepare_gdn_convolution_exact::<A, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN convolution", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        control_weights: *const u16,
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
            .gdn_control_exact::<A, TOKENS>(
                stream,
                &self.control,
                input.cast::<u32>(),
                control_weights,
                a_log,
                dt_bias,
                log_decay,
                beta,
            )
            .map_err(|source| GpuError::launch("launching GDN control", source))?;
        module
            .gdn_convolution_exact::<A, TOKENS>(
                stream,
                &self.convolution,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
            .map_err(|source| GpuError::launch("launching GDN convolution", source))
    }
}

/// Prepared GDN control and convolution routes for every exact decode batch.
pub struct GdnPrepareOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedRoute<A, 1>,
    b2: PreparedRoute<A, 2>,
    b3: PreparedRoute<A, 3>,
    b4: PreparedRoute<A, 4>,
    b5: PreparedRoute<A, 5>,
    b6: PreparedRoute<A, 6>,
    b7: PreparedRoute<A, 7>,
    b8: PreparedRoute<A, 8>,
}

impl<A: Sm120Arch> GdnPrepareOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = gdn_prepare_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN prepare module", source))?;

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

    /// Computes controls, advances mapped causal history, and emits SiLU values.
    ///
    /// # Safety
    ///
    /// Inputs cover `[batch, A::HIDDEN]` and `[batch, A::GDN_INPUT_ROWS]` BF16
    /// values. The fused control weights cover `[2 * A::GDN_CONTROL_ROWS,
    /// A::HIDDEN]`; `a_log` and `dt_bias` each cover `A::GDN_CONTROL_ROWS`;
    /// convolution weights cover `[A::GDN_QKV_ROWS, 4]`; and each state-row
    /// index is below the caller-owned history capacity. Outputs cover
    /// `[batch, A::GDN_CONTROL_ROWS]` FP32 controls and
    /// `[batch, A::GDN_QKV_ROWS]` BF16 values. Allocations are aligned,
    /// non-overlapping, live through completion, and belong to `stream`'s
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        control_weights: *const u16,
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
                "GDN prepare batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        control_weights,
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

/// PTX symbols retained for both leaves of every exact GDN prepare route.
pub(crate) fn gdn_prepare_ptx_names() -> Vec<&'static str> {
    macro_rules! names {
        ($tokens:literal) => {
            [
                kernels::gdn_control_exact_ptx_name::<Qwen38_27B, $tokens>(),
                kernels::gdn_convolution_exact_ptx_name::<Qwen38_27B, $tokens>(),
            ]
        };
    }

    names!(1)
        .into_iter()
        .chain(names!(2))
        .chain(names!(3))
        .chain(names!(4))
        .chain(names!(5))
        .chain(names!(6))
        .chain(names!(7))
        .chain(names!(8))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_ROWS_PER_CTA, CONTROL_THREADS, CONV_THREADS, admitted_batch, gdn_prepare_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn batch_table_covers_only_exact_decode_routes() {
        for (batch, expected) in [
            (0, false),
            (1, true),
            (4, true),
            (8, true),
            (9, false),
            (16, false),
        ] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
    }

    #[test]
    fn geometry_matches_the_retained_schedules() {
        assert_eq!(CONTROL_THREADS, 512);
        assert_eq!(CONV_THREADS, 256);
        assert_eq!(2 * Qwen38_27B::GDN_CONTROL_ROWS / CONTROL_ROWS_PER_CTA, 12);
        assert_eq!(Qwen38_27B::GDN_QKV_ROWS, 10_240);
        assert_eq!(Qwen38_27B::GDN_INPUT_ROWS, 16_384);
        assert_eq!(Qwen38_27B::LINEAR_CONV_KERNEL_DIM, 4);
    }

    #[test]
    fn ptx_inventory_has_two_entries_per_batch() {
        let names = gdn_prepare_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 16);
        assert_eq!(unique.len(), names.len());
    }
}

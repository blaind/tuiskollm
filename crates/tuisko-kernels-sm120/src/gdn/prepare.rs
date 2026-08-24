//! Exact-batch GDN control and causal-convolution preparation.

use crate::Sm120Arch;
use crate::device::gdn_prepare::{
    gdn_control, gdn_convolution, gdn_convolution_prefill, gdn_convolution_prefill_history,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const CONTROL_WARPS: usize = 16;
const CONTROL_THREADS: u32 = (CONTROL_WARPS * 32) as u32;
const CONTROL_ROWS_PER_CTA: usize = CONTROL_WARPS / 2;
const CONV_THREADS: u32 = 256;
const CAUSAL_ROWS: [usize; 4] = [1, 2, 3, 4];
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn admitted_rows(rows: usize) -> bool {
    admitted_batch(rows) || PREFILL_ROWS.contains(&rows)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::HIDDEN == 0
        || !A::HIDDEN.is_multiple_of(64)
        || A::GDN_CONTROL_ROWS == 0
        || !(2 * A::GDN_CONTROL_ROWS).is_multiple_of(CONTROL_ROWS_PER_CTA)
        || A::GDN_QKV_ROWS == 0
        || !A::GDN_QKV_ROWS.is_multiple_of(CONV_THREADS as usize)
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

    /// Computes the two control vectors for one exact prefill sequence.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_control_prefill_exact<A: Arch, const TOKENS: usize>(
        input: *const u32,
        control_weights: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, CONTROL_WARPS, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUMS).cast::<f32>();

        // The same two-warps-per-row reduction retains decode's represented
        // accumulation order. T tokens expose 12*T independent CTAs.
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

    /// Publishes the final three represented projected values to causal history.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_convolution_prefill_history_exact<A: Arch, const TOKENS: usize>(
        projected: *const u16,
        state_rows: *const u32,
        history: *mut u16,
    ) {
        // One thread owns all three words for one channel. Forty CTAs cover
        // 10,240 channels after the convolution grid has finished reading history.
        unsafe {
            gdn_convolution_prefill_history::<A, TOKENS>(projected, state_rows, history);
        }
    }

    /// Applies from-empty causal convolution across one exact prefill sequence.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_convolution_prefill_exact<A: Arch, const TOKENS: usize>(
        projected: *const u16,
        weights: *const u16,
        state_rows: *const u32,
        history: *const u16,
        output: *mut u16,
    ) {
        // Width four makes every token an independent represented-value read
        // from the projected sequence. Forty CTAs per token expose the complete
        // 10,240-channel plane without racing the later history publication.
        unsafe {
            gdn_convolution_prefill::<A, TOKENS>(projected, weights, state_rows, history, output);
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

struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    control: PreparedLaunch<kernels::__gdn_control_prefill_exact_CudaKernel<A, TOKENS>>,
    convolution: PreparedLaunch<kernels::__gdn_convolution_prefill_exact_CudaKernel<A, TOKENS>>,
    history: PreparedLaunch<kernels::__gdn_convolution_prefill_history_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !CAUSAL_ROWS.contains(&TOKENS) && !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "GDN prepare causal route T={TOKENS} is not admitted"
            )));
        }
        let control_blocks = u32::try_from(TOKENS * 2 * A::GDN_CONTROL_ROWS / CONTROL_ROWS_PER_CTA)
            .map_err(|_| GpuError::invalid_launch("GDN prefill control grid exceeds u32"))?;
        let convolution_blocks = u32::try_from(
            (TOKENS * A::GDN_QKV_ROWS).div_ceil(CONV_THREADS as usize),
        )
        .map_err(|_| GpuError::invalid_launch("GDN prefill convolution grid exceeds u32"))?;
        let history_blocks = u32::try_from(A::GDN_QKV_ROWS / CONV_THREADS as usize)
            .map_err(|_| GpuError::invalid_launch("GDN prefill history grid exceeds u32"))?;

        Ok(Self {
            control: module
                .prepare_gdn_control_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                    control_blocks,
                    CONTROL_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN prefill control", source))?,
            convolution: module
                .prepare_gdn_convolution_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN prefill convolution", source))?,
            history: module
                .prepare_gdn_convolution_prefill_history_exact::<A, TOKENS>(LaunchConfig1D::new(
                    history_blocks,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing GDN prefill history publication", source)
                })?,
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
            .gdn_control_prefill_exact::<A, TOKENS>(
                stream,
                &self.control,
                input.cast::<u32>(),
                control_weights,
                a_log,
                dt_bias,
                log_decay,
                beta,
            )
            .map_err(|source| GpuError::launch("launching GDN prefill control", source))?;
        module
            .gdn_convolution_prefill_exact::<A, TOKENS>(
                stream,
                &self.convolution,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
            .map_err(|source| GpuError::launch("launching GDN prefill convolution", source))?;
        module
            .gdn_convolution_prefill_history_exact::<A, TOKENS>(
                stream,
                &self.history,
                projected,
                state_rows,
                history,
            )
            .map_err(|source| GpuError::launch("launching GDN prefill history publication", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_replay(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        module
            .gdn_convolution_prefill_exact::<A, TOKENS>(
                stream,
                &self.convolution,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
            .map_err(|source| {
                GpuError::launch("launching GDN causal replay convolution", source)
            })?;
        module
            .gdn_convolution_prefill_history_exact::<A, TOKENS>(
                stream,
                &self.history,
                projected,
                state_rows,
                history,
            )
            .map_err(|source| GpuError::launch("launching GDN causal replay history", source))
    }
}

/// Prepared GDN control and convolution routes for exact decode and prefill rows.
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
    k1: PreparedPrefillRoute<A, 1>,
    k2: PreparedPrefillRoute<A, 2>,
    k3: PreparedPrefillRoute<A, 3>,
    k4: PreparedPrefillRoute<A, 4>,
    t32: PreparedPrefillRoute<A, 32>,
    t64: PreparedPrefillRoute<A, 64>,
    t128: PreparedPrefillRoute<A, 128>,
    t1024: PreparedPrefillRoute<A, 1_024>,
}

impl<A: Sm120Arch> GdnPrepareOp<A> {
    /// Loads the embedded SM120 module and prepares every exact route.
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
            k1: PreparedPrefillRoute::prepare(&module)?,
            k2: PreparedPrefillRoute::prepare(&module)?,
            k3: PreparedPrefillRoute::prepare(&module)?,
            k4: PreparedPrefillRoute::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Computes controls, advances mapped causal history, and emits SiLU values.
    ///
    /// # Safety
    ///
    /// Inputs cover `[rows, A::HIDDEN]` and `[rows, A::GDN_INPUT_ROWS]` BF16
    /// values. The fused control weights cover `[2 * A::GDN_CONTROL_ROWS,
    /// A::HIDDEN]`; `a_log` and `dt_bias` each cover `A::GDN_CONTROL_ROWS`;
    /// convolution weights cover `[A::GDN_QKV_ROWS, 4]`; and each state-row
    /// index is below the caller-owned history capacity. Prefill routes read one
    /// state-row index and advance that row across the contiguous sequence. Outputs cover
    /// `[rows, A::GDN_CONTROL_ROWS]` FP32 controls and
    /// `[rows, A::GDN_QKV_ROWS]` BF16 values. Allocations are aligned,
    /// non-overlapping, live through completion, and belong to `stream`'s
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
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
        if !admitted_rows(rows) {
            return Err(GpuError::invalid_launch(format!(
                "GDN prepare row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024"
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
            1_024 => launch!(t1024),
            _ => unreachable!(),
        }
    }

    /// Advances one state row through an exact `K=1..4` causal sequence.
    ///
    /// Unlike the decode routes, every row names the same state row and the
    /// history publication happens once after all convolution reads complete.
    ///
    /// # Safety
    ///
    /// Inputs and outputs cover the same planes documented by [`Self::launch`]
    /// for `tokens` rows. `state_rows` covers one valid row index. The caller
    /// owns that row exclusively through completion; allocations are aligned,
    /// non-overlapping, live through completion, and belong to `stream`'s
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_causal(
        &self,
        stream: &CudaStream,
        tokens: usize,
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

        match tokens {
            1 => launch!(k1),
            2 => launch!(k2),
            3 => launch!(k3),
            4 => launch!(k4),
            _ => Err(GpuError::invalid_launch(format!(
                "GDN causal prepare token count {tokens} is outside the admitted routes 1..=4"
            ))),
        }
    }

    /// Replays recorded projected values into one live causal-history row.
    ///
    /// # Safety
    ///
    /// `projected` and `convolved` cover `[tokens, A::GDN_QKV_ROWS]` BF16
    /// values, convolution weights cover `[A::GDN_QKV_ROWS, 4]`, and
    /// `state_rows` covers one valid row index. The caller owns that history
    /// row exclusively through completion; allocations are aligned,
    /// non-overlapping, live through completion, and belong to `stream`'s
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_causal_replay(
        &self,
        stream: &CudaStream,
        tokens: usize,
        projected: *const u16,
        convolution_weights: *const u16,
        state_rows: *const u32,
        history: *mut u16,
        convolved: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch_replay(
                        &self.module,
                        stream,
                        projected,
                        convolution_weights,
                        state_rows,
                        history,
                        convolved,
                    )
                }
            };
        }

        match tokens {
            1 => launch!(k1),
            2 => launch!(k2),
            3 => launch!(k3),
            4 => launch!(k4),
            _ => Err(GpuError::invalid_launch(format!(
                "GDN causal replay token count {tokens} is outside the admitted routes 1..=4"
            ))),
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

    macro_rules! prefill_names {
        ($tokens:literal) => {
            [
                kernels::gdn_control_prefill_exact_ptx_name::<Qwen38_27B, $tokens>(),
                kernels::gdn_convolution_prefill_exact_ptx_name::<Qwen38_27B, $tokens>(),
                kernels::gdn_convolution_prefill_history_exact_ptx_name::<Qwen38_27B, $tokens>(),
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
        .chain(prefill_names!(1))
        .chain(prefill_names!(2))
        .chain(prefill_names!(3))
        .chain(prefill_names!(4))
        .chain(prefill_names!(32))
        .chain(prefill_names!(64))
        .chain(prefill_names!(128))
        .chain(prefill_names!(1_024))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_ROWS_PER_CTA, CONTROL_THREADS, CONV_THREADS, admitted_batch, admitted_rows,
        gdn_prepare_ptx_names,
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
    fn row_table_covers_exact_decode_and_prefill_routes() {
        for (rows, expected) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (31, false),
            (32, true),
            (64, true),
            (128, true),
            (1_024, true),
            (1_025, false),
        ] {
            assert_eq!(admitted_rows(rows), expected, "rows={rows}");
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
    fn ptx_inventory_has_decode_pairs_and_prefill_triples() {
        let names = gdn_prepare_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 40);
        assert_eq!(unique.len(), names.len());
    }
}

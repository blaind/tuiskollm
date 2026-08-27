//! Exact-batch GDN control and causal-convolution preparation.

use crate::device::gdn_prepare::{
    gdn_control, gdn_convolution, gdn_convolution_prefill, gdn_convolution_prefill_history,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen38_27B, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const CONTROL_WARPS: usize = 16;
const CONTROL_THREADS: u32 = (CONTROL_WARPS * 32) as u32;
const CONTROL_ROWS_PER_CTA: usize = CONTROL_WARPS / 2;
const CONV_THREADS: u32 = 256;
const CAUSAL_ROWS: [usize; 4] = [1, 2, 3, 4];
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
/// Flash-Next control input width.
const QWEN38_FLASH_NEXT_HIDDEN: usize = 2_560;

// Bind reused Qwen3.8-27B convolution and history entries to exact geometry.
const _: () = assert!(Qwen38FlashNext::GDN_QKV_ROWS == Qwen38_27B::GDN_QKV_ROWS);
const _: () = assert!(Qwen38FlashNext::GDN_INPUT_ROWS == Qwen38_27B::GDN_INPUT_ROWS);
const _: () = assert!(Qwen38FlashNext::GDN_CONTROL_ROWS == Qwen38_27B::GDN_CONTROL_ROWS);
const _: () =
    assert!(Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM == Qwen38_27B::LINEAR_CONV_KERNEL_DIM);
const _: () = assert!(Qwen38FlashNext::HIDDEN == QWEN38_FLASH_NEXT_HIDDEN);
const _: () = assert!(Qwen38FlashNext::HIDDEN != Qwen38_27B::HIDDEN);

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

/// Checks the exact Flash-Next geometry used by emitted and reused entries.
fn require_qwen38_flash_next_geometry() -> GpuResult<()> {
    if Qwen38FlashNext::HIDDEN != QWEN38_FLASH_NEXT_HIDDEN
        || Qwen38FlashNext::GDN_CONTROL_ROWS != Qwen38_27B::GDN_CONTROL_ROWS
        || Qwen38FlashNext::GDN_QKV_ROWS != Qwen38_27B::GDN_QKV_ROWS
        || Qwen38FlashNext::GDN_INPUT_ROWS != Qwen38_27B::GDN_INPUT_ROWS
        || Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM != 4
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next geometry is incompatible with the GDN prepare schedule",
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

    /// Computes the two Qwen3.8-Flash-Next control vectors for one exact batch.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_gdn_control_exact<A: Arch, const TOKENS: usize>(
        input: *const u32,
        control_weights: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, CONTROL_WARPS, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUMS).cast::<f32>();

        // The 2,560-wide column walk requires a distinct entry.
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

    /// Computes the two Qwen3.8-Flash-Next control vectors for one exact prefill sequence.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_gdn_control_prefill_exact<A: Arch, const TOKENS: usize>(
        input: *const u32,
        control_weights: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        log_decay: *mut f32,
        beta: *mut f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, CONTROL_WARPS, 16> = SharedArray::UNINIT;
        let warp_sums = core::ptr::addr_of_mut!(WARP_SUMS).cast::<f32>();

        // Retain the decode entry's two-warps-per-row accumulation order.
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

mod private {
    pub trait Sealed {}
}

/// Prepared control and convolution entries for one exact decode row count.
pub trait GdnPrepareRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares both entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's control entry and then its convolution entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`GdnPrepareOp::launch`]'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// One architecture's prepared control, convolution, and history-publication
/// entries for an exact causal or prefill row count.
pub trait GdnPrefillPrepareRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares all three entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches control, convolution, and the history publication in order.
    ///
    /// # Safety
    ///
    /// The pointers carry [`GdnPrepareOp::launch`]'s contract unchanged.
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
    ) -> GpuResult<()>;

    /// Replays recorded projected values without recomputing the controls.
    ///
    /// # Safety
    ///
    /// The pointers carry [`GdnPrepareOp::launch_causal_replay`]'s contract
    /// unchanged.
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
    ) -> GpuResult<()>;
}

/// Sealed entry table for one admitted architecture's GDN prepare routes.
pub trait GdnPrepareEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: GdnPrepareRoute<A>;
    /// Prepared causal or prefill route for one exact row count.
    type Prefill<const TOKENS: usize>: GdnPrefillPrepareRoute<A>;

    /// Message prefix that keeps this architecture's launch errors distinct.
    const LABEL: &'static str;

    /// Rejects an architecture whose prepare contract is not this schedule's.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8-27B control and convolution entries for one exact batch.
pub struct PreparedRoute<A: Arch, const TOKENS: usize> {
    control: PreparedLaunch<kernels::__gdn_control_exact_CudaKernel<A, TOKENS>>,
    convolution: PreparedLaunch<kernels::__gdn_convolution_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Flash-Next control and reused convolution entries for one exact batch.
pub struct Qwen38FlashNextPreparedRoute<const TOKENS: usize> {
    control: PreparedLaunch<
        kernels::__qwen38_flash_next_gdn_control_exact_CudaKernel<Qwen38FlashNext, TOKENS>,
    >,
    convolution: PreparedLaunch<kernels::__gdn_convolution_exact_CudaKernel<Qwen38_27B, TOKENS>>,
}

/// Prepared Qwen3.8-27B causal or prefill entries for one exact row count.
pub struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    control: PreparedLaunch<kernels::__gdn_control_prefill_exact_CudaKernel<A, TOKENS>>,
    convolution: PreparedLaunch<kernels::__gdn_convolution_prefill_exact_CudaKernel<A, TOKENS>>,
    history: PreparedLaunch<kernels::__gdn_convolution_prefill_history_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Flash-Next control and reused prefill entries for one exact row count.
pub struct Qwen38FlashNextPreparedPrefillRoute<const TOKENS: usize> {
    control: PreparedLaunch<
        kernels::__qwen38_flash_next_gdn_control_prefill_exact_CudaKernel<Qwen38FlashNext, TOKENS>,
    >,
    convolution:
        PreparedLaunch<kernels::__gdn_convolution_prefill_exact_CudaKernel<Qwen38_27B, TOKENS>>,
    history: PreparedLaunch<
        kernels::__gdn_convolution_prefill_history_exact_CudaKernel<Qwen38_27B, TOKENS>,
    >,
}

impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for Qwen38FlashNextPreparedRoute<TOKENS> {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedPrefillRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for Qwen38FlashNextPreparedPrefillRoute<TOKENS> {}

fn control_blocks<A: Arch>(tokens: usize) -> GpuResult<u32> {
    u32::try_from(tokens * 2 * A::GDN_CONTROL_ROWS / CONTROL_ROWS_PER_CTA)
        .map_err(|_| GpuError::invalid_launch("GDN control grid exceeds u32"))
}

fn convolution_blocks<A: Arch>(tokens: usize) -> GpuResult<u32> {
    u32::try_from((tokens * A::GDN_QKV_ROWS).div_ceil(CONV_THREADS as usize))
        .map_err(|_| GpuError::invalid_launch("GDN convolution grid exceeds u32"))
}

fn history_blocks<A: Arch>() -> GpuResult<u32> {
    u32::try_from(A::GDN_QKV_ROWS / CONV_THREADS as usize)
        .map_err(|_| GpuError::invalid_launch("GDN prefill history grid exceeds u32"))
}

fn require_admitted_prefill(tokens: usize) -> GpuResult<()> {
    if !CAUSAL_ROWS.contains(&tokens) && !PREFILL_ROWS.contains(&tokens) {
        return Err(GpuError::invalid_launch(format!(
            "GDN prepare causal route T={tokens} is not admitted"
        )));
    }

    Ok(())
}

impl<A: Sm120Arch, const TOKENS: usize> GdnPrepareRoute<A> for PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            control: module
                .prepare_gdn_control_exact::<A, TOKENS>(LaunchConfig1D::new(
                    control_blocks::<A>(TOKENS)?,
                    CONTROL_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN control", source))?,
            convolution: module
                .prepare_gdn_convolution_exact::<A, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks::<A>(TOKENS)?,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN convolution", source))?,
        })
    }

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

impl<const TOKENS: usize> GdnPrepareRoute<Qwen38FlashNext>
    for Qwen38FlashNextPreparedRoute<TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            control: module
                .prepare_qwen38_flash_next_gdn_control_exact::<Qwen38FlashNext, TOKENS>(
                    LaunchConfig1D::new(
                        control_blocks::<Qwen38FlashNext>(TOKENS)?,
                        CONTROL_THREADS,
                        0,
                    ),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next GDN control", source)
                })?,
            convolution: module
                .prepare_gdn_convolution_exact::<Qwen38_27B, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks::<Qwen38FlashNext>(TOKENS)?,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next GDN convolution", source)
                })?,
        })
    }

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
            .qwen38_flash_next_gdn_control_exact::<Qwen38FlashNext, TOKENS>(
                stream,
                &self.control,
                input.cast::<u32>(),
                control_weights,
                a_log,
                dt_bias,
                log_decay,
                beta,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next GDN control", source)
            })?;
        module
            .gdn_convolution_exact::<Qwen38_27B, TOKENS>(
                stream,
                &self.convolution,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next GDN convolution", source)
            })
    }
}

impl<A: Sm120Arch, const TOKENS: usize> GdnPrefillPrepareRoute<A>
    for PreparedPrefillRoute<A, TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        require_admitted_prefill(TOKENS)?;

        Ok(Self {
            control: module
                .prepare_gdn_control_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                    control_blocks::<A>(TOKENS)?,
                    CONTROL_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN prefill control", source))?,
            convolution: module
                .prepare_gdn_convolution_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks::<A>(TOKENS)?,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing GDN prefill convolution", source))?,
            history: module
                .prepare_gdn_convolution_prefill_history_exact::<A, TOKENS>(LaunchConfig1D::new(
                    history_blocks::<A>()?,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing GDN prefill history publication", source)
                })?,
        })
    }

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
        // SAFETY: the public method's pointer contract is unchanged by dispatch.
        unsafe {
            self.launch_replay(
                module,
                stream,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
        }
    }

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
}

impl<const TOKENS: usize> GdnPrefillPrepareRoute<Qwen38FlashNext>
    for Qwen38FlashNextPreparedPrefillRoute<TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        require_admitted_prefill(TOKENS)?;

        Ok(Self {
            control: module
                .prepare_qwen38_flash_next_gdn_control_prefill_exact::<Qwen38FlashNext, TOKENS>(
                    LaunchConfig1D::new(
                        control_blocks::<Qwen38FlashNext>(TOKENS)?,
                        CONTROL_THREADS,
                        0,
                    ),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next GDN prefill control", source)
                })?,
            convolution: module
                .prepare_gdn_convolution_prefill_exact::<Qwen38_27B, TOKENS>(LaunchConfig1D::new(
                    convolution_blocks::<Qwen38FlashNext>(TOKENS)?,
                    CONV_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing Qwen3.8-Flash-Next GDN prefill convolution",
                        source,
                    )
                })?,
            history: module
                .prepare_gdn_convolution_prefill_history_exact::<Qwen38_27B, TOKENS>(
                    LaunchConfig1D::new(history_blocks::<Qwen38FlashNext>()?, CONV_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next GDN prefill history", source)
                })?,
        })
    }

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
            .qwen38_flash_next_gdn_control_prefill_exact::<Qwen38FlashNext, TOKENS>(
                stream,
                &self.control,
                input.cast::<u32>(),
                control_weights,
                a_log,
                dt_bias,
                log_decay,
                beta,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next GDN prefill control", source)
            })?;
        // SAFETY: the public method's pointer contract is unchanged by dispatch.
        unsafe {
            self.launch_replay(
                module,
                stream,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
        }
    }

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
            .gdn_convolution_prefill_exact::<Qwen38_27B, TOKENS>(
                stream,
                &self.convolution,
                projected,
                convolution_weights,
                state_rows,
                history,
                convolved,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching Qwen3.8-Flash-Next GDN prefill convolution",
                    source,
                )
            })?;
        module
            .gdn_convolution_prefill_history_exact::<Qwen38_27B, TOKENS>(
                stream,
                &self.history,
                projected,
                state_rows,
                history,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next GDN prefill history", source)
            })
    }
}

/// Qwen3.8-27B control, convolution, and history entry table.
pub struct Qwen38GdnPrepareEntries;

/// Flash-Next control entries with reused Qwen3.8-27B convolution entries.
pub struct Qwen38FlashNextGdnPrepareEntries;

impl private::Sealed for Qwen38GdnPrepareEntries {}
impl private::Sealed for Qwen38FlashNextGdnPrepareEntries {}

impl<A: Sm120Arch> GdnPrepareEntries<A> for Qwen38GdnPrepareEntries {
    type Decode<const TOKENS: usize> = PreparedRoute<A, TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<A, TOKENS>;

    const LABEL: &'static str = "";

    fn require_geometry() -> GpuResult<()> {
        require_geometry::<A>()
    }

    fn ptx_names() -> Vec<&'static str> {
        gdn_prepare_ptx_names()
    }
}

impl GdnPrepareEntries<Qwen38FlashNext> for Qwen38FlashNextGdnPrepareEntries {
    type Decode<const TOKENS: usize> = Qwen38FlashNextPreparedRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = Qwen38FlashNextPreparedPrefillRoute<TOKENS>;

    const LABEL: &'static str = "Qwen3.8-Flash-Next ";

    fn require_geometry() -> GpuResult<()> {
        require_qwen38_flash_next_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_gdn_prepare_ptx_names()
    }
}

/// Prepared GDN control and convolution routes for exact decode and prefill rows.
pub struct GdnPrepareOp<A: Arch = Qwen38_27B, E: GdnPrepareEntries<A> = Qwen38GdnPrepareEntries> {
    module: kernels::LoadedModule,
    b1: E::Decode<1>,
    b2: E::Decode<2>,
    b3: E::Decode<3>,
    b4: E::Decode<4>,
    b5: E::Decode<5>,
    b6: E::Decode<6>,
    b7: E::Decode<7>,
    b8: E::Decode<8>,
    k1: E::Prefill<1>,
    k2: E::Prefill<2>,
    k3: E::Prefill<3>,
    k4: E::Prefill<4>,
    t32: E::Prefill<32>,
    t64: E::Prefill<64>,
    t128: E::Prefill<128>,
    t1024: E::Prefill<1_024>,
}

/// Prepared Qwen3.8-Flash-Next GDN control and convolution routes.
pub type Qwen38FlashNextGdnPrepareOp =
    GdnPrepareOp<Qwen38FlashNext, Qwen38FlashNextGdnPrepareEntries>;

impl<A: Arch, E: GdnPrepareEntries<A>> GdnPrepareOp<A, E> {
    /// Loads the embedded SM120 module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN prepare module", source))?;

        Ok(Self {
            b1: E::Decode::<1>::prepare(&module)?,
            b2: E::Decode::<2>::prepare(&module)?,
            b3: E::Decode::<3>::prepare(&module)?,
            b4: E::Decode::<4>::prepare(&module)?,
            b5: E::Decode::<5>::prepare(&module)?,
            b6: E::Decode::<6>::prepare(&module)?,
            b7: E::Decode::<7>::prepare(&module)?,
            b8: E::Decode::<8>::prepare(&module)?,
            k1: E::Prefill::<1>::prepare(&module)?,
            k2: E::Prefill::<2>::prepare(&module)?,
            k3: E::Prefill::<3>::prepare(&module)?,
            k4: E::Prefill::<4>::prepare(&module)?,
            t32: E::Prefill::<32>::prepare(&module)?,
            t64: E::Prefill::<64>::prepare(&module)?,
            t128: E::Prefill::<128>::prepare(&module)?,
            t1024: E::Prefill::<1_024>::prepare(&module)?,
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
                "{}GDN prepare row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024",
                E::LABEL
            )));
        }

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
                "{}GDN causal prepare token count {tokens} is outside the admitted routes 1..=4",
                E::LABEL
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
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
                "{}GDN causal replay token count {tokens} is outside the admitted routes 1..=4",
                E::LABEL
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

/// Target-specific PTX symbols retained for exact Flash-Next prepare routes.
pub(crate) fn qwen38_flash_next_gdn_prepare_ptx_names() -> Vec<&'static str> {
    macro_rules! names {
        ($tokens:literal) => {
            [kernels::qwen38_flash_next_gdn_control_exact_ptx_name::<
                Qwen38FlashNext,
                $tokens,
            >()]
        };
    }

    macro_rules! prefill_names {
        ($tokens:literal) => {
            [
                kernels::qwen38_flash_next_gdn_control_prefill_exact_ptx_name::<
                    Qwen38FlashNext,
                    $tokens,
                >(),
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
        CONTROL_ROWS_PER_CTA, CONTROL_THREADS, CONV_THREADS, GdnPrepareEntries,
        Qwen38FlashNextGdnPrepareEntries, Qwen38GdnPrepareEntries, admitted_batch, admitted_rows,
        gdn_prepare_ptx_names, qwen38_flash_next_gdn_prepare_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B, Qwen38FlashNext};

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
    fn qwen38_flash_next_prepare_reuses_the_convolution_and_diverges_on_hidden() {
        assert_eq!(Qwen38FlashNext::GDN_QKV_ROWS, Qwen38_27B::GDN_QKV_ROWS);
        assert_eq!(Qwen38FlashNext::GDN_INPUT_ROWS, Qwen38_27B::GDN_INPUT_ROWS);
        assert_eq!(
            Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM,
            Qwen38_27B::LINEAR_CONV_KERNEL_DIM
        );
        assert_eq!(
            Qwen38FlashNext::GDN_CONTROL_ROWS,
            Qwen38_27B::GDN_CONTROL_ROWS
        );
        assert_ne!(Qwen38FlashNext::HIDDEN, Qwen38_27B::HIDDEN);
        assert_eq!(Qwen38FlashNext::HIDDEN, 2_560);
        assert_eq!(
            2 * Qwen38FlashNext::GDN_CONTROL_ROWS / CONTROL_ROWS_PER_CTA,
            12
        );
    }

    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38GdnPrepareEntries as GdnPrepareEntries<Qwen38_27B>>::ptx_names(),
            gdn_prepare_ptx_names()
        );
        assert_eq!(
            <Qwen38FlashNextGdnPrepareEntries as GdnPrepareEntries<Qwen38FlashNext>>::ptx_names(),
            qwen38_flash_next_gdn_prepare_ptx_names()
        );
    }

    #[test]
    fn ptx_inventory_has_decode_pairs_and_prefill_triples() {
        let names = gdn_prepare_ptx_names();
        let qwen38_flash_next = qwen38_flash_next_gdn_prepare_ptx_names();
        let unique = names
            .iter()
            .chain(&qwen38_flash_next)
            .copied()
            .collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 40);
        assert_eq!(qwen38_flash_next.len(), 16);
        assert_eq!(unique.len(), names.len() + qwen38_flash_next.len());
    }
}

//! Exact-batch FP32 GDN recurrence and gated normalization.

use crate::device::gdn_recurrence::{
    SPLIT_CTAS_PER_HEAD, SPLIT_ROWS, gdn_recurrence, gdn_recurrence_prefill,
    gdn_recurrence_prefill_epilogue,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen38_27B, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const WARPS: usize = 16;
const THREADS: u32 = (WARPS * 32) as u32;
const CAUSAL_ROWS: [usize; 4] = [1, 2, 3, 4];
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];

// Bind the reused serial entry to every target-dependent stride it reads.
const _: () = assert!(Qwen38FlashNext::GDN_QKV_ROWS == Qwen38_27B::GDN_QKV_ROWS);
const _: () = assert!(Qwen38FlashNext::GDN_INPUT_ROWS == Qwen38_27B::GDN_INPUT_ROWS);
const _: () = assert!(Qwen38FlashNext::GDN_CONTROL_ROWS == Qwen38_27B::GDN_CONTROL_ROWS);
const _: () = assert!(Qwen38FlashNext::LINEAR_KEY_HEADS == Qwen38_27B::LINEAR_KEY_HEADS);
const _: () = assert!(Qwen38FlashNext::LINEAR_VALUE_HEADS == Qwen38_27B::LINEAR_VALUE_HEADS);
const _: () = assert!(Qwen38FlashNext::LINEAR_HEAD_DIM == Qwen38_27B::LINEAR_HEAD_DIM);
const _: () = assert!(Qwen38FlashNext::RMS_NORM_EPSILON == Qwen38_27B::RMS_NORM_EPSILON);

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn admitted_rows(rows: usize) -> bool {
    admitted_batch(rows) || PREFILL_ROWS.contains(&rows)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::LINEAR_KEY_HEADS != KEY_HEADS
        || A::LINEAR_VALUE_HEADS != VALUE_HEADS
        || A::LINEAR_HEAD_DIM != HEAD_DIM
        || !VALUE_HEADS.is_multiple_of(KEY_HEADS)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the GDN recurrence schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Updates mapped FP32 state and emits the gated normalized value plane.
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
    pub fn gdn_recurrence_exact<A: Arch, const TOKENS: usize>(
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

        // T=1 moves 6.29 MB of FP32 state through 48 CTAs in about 8.416 us
        // (~748 GB/s). Sixteen warps expose 16 independent state-row reductions
        // per CTA; each keeps its four fixed columns per lane and `row += 16`,
        // so route specialization changes only the number of CTAs, never a
        // state's update or reduction order.
        unsafe {
            gdn_recurrence::<A, TOKENS, false>(
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                output,
                core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                core::ptr::addr_of_mut!(RECURRENT_OUTPUT).cast::<f32>(),
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
            );
        }
    }

    /// Advances one mapped state row through an exact causal prefill sequence.
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
    pub fn gdn_recurrence_prefill_exact<A: Arch, const TOKENS: usize>(
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

        // State dependence permits two independent CTAs per value head: every
        // row's update stays wholly inside one CTA, so ninety-six CTAs advance
        // tokens serially with half the per-warp row walk.
        // Each CTA copies its 32KB FP32 state half into shared once, advances
        // tokens serially against the shared tile with decode's
        // four-columns-per-lane state update and reduction order unchanged, and
        // writes the plane back once. The serial loop publishes its scaled
        // recurrent rows to the caller's plane; the paired epilogue kernel
        // applies the RMS/gate/store phase in parallel with the identical
        // reduction tree, so outputs and final state stay bit-exact.
        unsafe {
            gdn_recurrence_prefill::<A, TOKENS>(
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                output,
                core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                recurrent,
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
                core::ptr::addr_of_mut!(STATE_TILE).cast::<f32>(),
                core::ptr::addr_of_mut!(VALUE).cast::<f32>(),
            );
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
    pub fn gdn_recurrence_prefill_epilogue_exact<A: Arch, const TOKENS: usize>(
        projected: *const u16,
        norm_weight: *const u16,
        recurrent: *const f32,
        output: *mut u16,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;

        // One CTA per (token, value head) replicates the serial loop's
        // sixteen-warp RMS reduction tree over the published recurrent row,
        // so the emitted gated values are bit-exact.
        unsafe {
            gdn_recurrence_prefill_epilogue::<A, TOKENS, false>(
                projected,
                norm_weight,
                recurrent,
                output,
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
            );
        }
    }

    /// Updates mapped FP32 state and emits the sigmoid-gated value plane.
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
    pub fn qwen38_flash_next_gdn_recurrence_exact<A: Arch, const TOKENS: usize>(
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

        // The checkpoint requires a distinct sigmoid-gated entry.
        unsafe {
            gdn_recurrence::<A, TOKENS, true>(
                qkv,
                projected,
                log_decay,
                beta,
                norm_weight,
                state_rows,
                state,
                output,
                core::ptr::addr_of_mut!(QUERY).cast::<f32>(),
                core::ptr::addr_of_mut!(KEY).cast::<f32>(),
                core::ptr::addr_of_mut!(RECURRENT_OUTPUT).cast::<f32>(),
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
            );
        }
    }

    /// Applies the deferred prefill RMS/sigmoid-gate/store epilogue in parallel.
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
    pub fn qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact<A: Arch, const TOKENS: usize>(
        projected: *const u16,
        norm_weight: *const u16,
        recurrent: *const f32,
        output: *mut u16,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;

        // The shared serial pass returns before this target-specific gate.
        unsafe {
            gdn_recurrence_prefill_epilogue::<A, TOKENS, true>(
                projected,
                norm_weight,
                recurrent,
                output,
                core::ptr::addr_of_mut!(REDUCTION).cast::<f32>(),
            );
        }
    }
}

mod private {
    pub trait Sealed {}
}

/// Prepared decode entry for one exact row count.
pub trait GdnRecurrenceDecodeRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares this route's decode entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's decode entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`GdnRecurrenceOp::launch`]'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// One architecture's prepared serial-pass and epilogue entries for an exact
/// causal or prefill row count.
pub trait GdnRecurrencePrefillRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares both entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches the serial state advance and then its gated epilogue.
    ///
    /// # Safety
    ///
    /// The pointers carry [`GdnRecurrenceOp::launch`]'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// Sealed entry table for one admitted architecture's recurrence routes.
pub trait GdnRecurrenceEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: GdnRecurrenceDecodeRoute<A>;
    /// Prepared causal or prefill route for one exact row count.
    type Prefill<const TOKENS: usize>: GdnRecurrencePrefillRoute<A>;

    /// Message prefix that keeps this architecture's launch errors distinct.
    const LABEL: &'static str;
    /// Rejects an architecture whose recurrence contract is not this schedule's.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8-27B decode entry for one exact batch.
pub struct PreparedRoute<A: Arch, const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__gdn_recurrence_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Flash-Next sigmoid-gated decode entry for one exact batch.
pub struct Qwen38FlashNextPreparedRoute<const TOKENS: usize> {
    launch: PreparedLaunch<
        kernels::__qwen38_flash_next_gdn_recurrence_exact_CudaKernel<Qwen38FlashNext, TOKENS>,
    >,
}

/// Prepared Qwen3.8-27B serial-pass and epilogue entries for one exact row count.
pub struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__gdn_recurrence_prefill_exact_CudaKernel<A, TOKENS>>,
    epilogue:
        PreparedLaunch<kernels::__gdn_recurrence_prefill_epilogue_exact_CudaKernel<A, TOKENS>>,
}

/// Reused serial pass and Flash-Next epilogue for one exact row count.
pub struct Qwen38FlashNextPreparedPrefillRoute<const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__gdn_recurrence_prefill_exact_CudaKernel<Qwen38_27B, TOKENS>>,
    epilogue: PreparedLaunch<
        kernels::__qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_CudaKernel<
            Qwen38FlashNext,
            TOKENS,
        >,
    >,
}

impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for Qwen38FlashNextPreparedRoute<TOKENS> {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedPrefillRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for Qwen38FlashNextPreparedPrefillRoute<TOKENS> {}

fn decode_blocks(tokens: usize) -> GpuResult<u32> {
    u32::try_from(tokens * VALUE_HEADS)
        .map_err(|_| GpuError::invalid_launch("GDN recurrence grid exceeds u32"))
}

fn epilogue_blocks(tokens: usize) -> GpuResult<u32> {
    u32::try_from(tokens * VALUE_HEADS)
        .map_err(|_| GpuError::invalid_launch("GDN recurrence epilogue grid exceeds u32"))
}

fn require_admitted_prefill(tokens: usize) -> GpuResult<()> {
    if !CAUSAL_ROWS.contains(&tokens) && !PREFILL_ROWS.contains(&tokens) {
        return Err(GpuError::invalid_launch(format!(
            "GDN recurrence causal route T={tokens} is not admitted"
        )));
    }

    Ok(())
}

impl<A: Sm120Arch, const TOKENS: usize> GdnRecurrenceDecodeRoute<A> for PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = module
            .prepare_gdn_recurrence_exact::<A, TOKENS>(LaunchConfig1D::new(
                decode_blocks(TOKENS)?,
                THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing GDN recurrence", source))?;

        Ok(Self { launch })
    }

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
            .gdn_recurrence_exact::<A, TOKENS>(
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
            .map_err(|source| GpuError::launch("launching GDN recurrence", source))
    }
}

impl<const TOKENS: usize> GdnRecurrenceDecodeRoute<Qwen38FlashNext>
    for Qwen38FlashNextPreparedRoute<TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = module
            .prepare_qwen38_flash_next_gdn_recurrence_exact::<Qwen38FlashNext, TOKENS>(
                LaunchConfig1D::new(decode_blocks(TOKENS)?, THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.8-Flash-Next GDN recurrence", source)
            })?;

        Ok(Self { launch })
    }

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
            .qwen38_flash_next_gdn_recurrence_exact::<Qwen38FlashNext, TOKENS>(
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
                GpuError::launch("launching Qwen3.8-Flash-Next GDN recurrence", source)
            })
    }
}

impl<A: Sm120Arch, const TOKENS: usize> GdnRecurrencePrefillRoute<A>
    for PreparedPrefillRoute<A, TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        require_admitted_prefill(TOKENS)?;
        let launch = module
            .prepare_gdn_recurrence_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                (VALUE_HEADS * SPLIT_CTAS_PER_HEAD) as u32,
                THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing GDN recurrence prefill", source))?;
        let epilogue = module
            .prepare_gdn_recurrence_prefill_epilogue_exact::<A, TOKENS>(LaunchConfig1D::new(
                epilogue_blocks(TOKENS)?,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing GDN recurrence prefill epilogue", source)
            })?;

        Ok(Self { launch, epilogue })
    }

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
        // SAFETY: the epilogue reads the plane the serial pass just published
        // on the same stream, so ordering is inherent.
        module
            .gdn_recurrence_prefill_exact::<A, TOKENS>(
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
            .map_err(|source| GpuError::launch("launching GDN recurrence prefill", source))?;
        module
            .gdn_recurrence_prefill_epilogue_exact::<A, TOKENS>(
                stream,
                &self.epilogue,
                projected,
                norm_weight,
                recurrent.cast_const(),
                output,
            )
            .map_err(|source| GpuError::launch("launching GDN recurrence prefill epilogue", source))
    }
}

impl<const TOKENS: usize> GdnRecurrencePrefillRoute<Qwen38FlashNext>
    for Qwen38FlashNextPreparedPrefillRoute<TOKENS>
{
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        require_admitted_prefill(TOKENS)?;
        let launch = module
            .prepare_gdn_recurrence_prefill_exact::<Qwen38_27B, TOKENS>(LaunchConfig1D::new(
                (VALUE_HEADS * SPLIT_CTAS_PER_HEAD) as u32,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch(
                    "preparing Qwen3.8-Flash-Next GDN recurrence prefill",
                    source,
                )
            })?;
        let epilogue = module
            .prepare_qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact::<Qwen38FlashNext, TOKENS>(
                LaunchConfig1D::new(epilogue_blocks(TOKENS)?, THREADS, 0),
            )
            .map_err(|source| {
                GpuError::launch(
                    "preparing Qwen3.8-Flash-Next GDN recurrence prefill epilogue",
                    source,
                )
            })?;

        Ok(Self { launch, epilogue })
    }

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
        // SAFETY: the epilogue reads the plane the serial pass just published
        // on the same stream, so ordering is inherent.
        module
            .gdn_recurrence_prefill_exact::<Qwen38_27B, TOKENS>(
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
                GpuError::launch(
                    "launching Qwen3.8-Flash-Next GDN recurrence prefill",
                    source,
                )
            })?;
        module
            .qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact::<Qwen38FlashNext, TOKENS>(
                stream,
                &self.epilogue,
                projected,
                norm_weight,
                recurrent.cast_const(),
                output,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching Qwen3.8-Flash-Next GDN recurrence prefill epilogue",
                    source,
                )
            })
    }
}

/// Qwen3.8-27B SiLU-gated recurrence entry table.
pub struct Qwen38GdnRecurrenceEntries;

/// Flash-Next sigmoid entries with the reused Qwen3.8-27B serial pass.
pub struct Qwen38FlashNextGdnRecurrenceEntries;

impl private::Sealed for Qwen38GdnRecurrenceEntries {}
impl private::Sealed for Qwen38FlashNextGdnRecurrenceEntries {}

impl<A: Sm120Arch> GdnRecurrenceEntries<A> for Qwen38GdnRecurrenceEntries {
    type Decode<const TOKENS: usize> = PreparedRoute<A, TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<A, TOKENS>;

    const LABEL: &'static str = "";
    fn require_geometry() -> GpuResult<()> {
        require_geometry::<A>()
    }

    fn ptx_names() -> Vec<&'static str> {
        gdn_recurrence_ptx_names()
    }
}

impl GdnRecurrenceEntries<Qwen38FlashNext> for Qwen38FlashNextGdnRecurrenceEntries {
    type Decode<const TOKENS: usize> = Qwen38FlashNextPreparedRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = Qwen38FlashNextPreparedPrefillRoute<TOKENS>;

    const LABEL: &'static str = "Qwen3.8-Flash-Next ";
    fn require_geometry() -> GpuResult<()> {
        require_geometry::<Qwen38FlashNext>()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_gdn_recurrence_ptx_names()
    }
}

/// Prepared FP32 GDN recurrence routes for exact decode and prefill rows.
pub struct GdnRecurrenceOp<
    A: Arch = Qwen38_27B,
    E: GdnRecurrenceEntries<A> = Qwen38GdnRecurrenceEntries,
> {
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

/// Prepared Flash-Next GDN routes with sigmoid output gating.
pub type Qwen38FlashNextGdnRecurrenceOp =
    GdnRecurrenceOp<Qwen38FlashNext, Qwen38FlashNextGdnRecurrenceEntries>;

impl<A: Arch, E: GdnRecurrenceEntries<A>> GdnRecurrenceOp<A, E> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN recurrence module", source))?;

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

    /// Advances mapped FP32 state and emits gated BF16 recurrent values.
    ///
    /// # Safety
    ///
    /// `qkv` and `projected` cover `[rows, A::GDN_QKV_ROWS]` and
    /// `[rows, A::GDN_INPUT_ROWS]` BF16 values. Controls cover
    /// `[rows, A::GDN_CONTROL_ROWS]`; `norm_weight` covers one head;
    /// every state-row index is below the caller-owned `[rows,
    /// A::GDN_CONTROL_ROWS, A::LINEAR_HEAD_DIM, A::LINEAR_HEAD_DIM]` FP32
    /// state; `recurrent` covers `[rows, A::GDN_VALUE_ROWS]` FP32 values;
    /// and `output` covers `[rows, A::GDN_VALUE_ROWS]` BF16 values. Prefill
    /// routes read one state-row index, advance that row causally, and use
    /// `recurrent` as the intermediate output plane.
    /// Allocations are aligned, non-overlapping, live through completion, and
    /// belong to `stream`'s context.
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
                "{}GDN recurrence row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024",
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
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
                        recurrent,
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
            32 => launch_prefill!(t32),
            64 => launch_prefill!(t64),
            128 => launch_prefill!(t128),
            1_024 => launch_prefill!(t1024),
            _ => unreachable!(),
        }
    }

    /// Advances one state row through an exact `K=1..4` causal sequence.
    ///
    /// Forty-eight value-head CTAs each advance their owned state serially;
    /// this exposes head parallelism without changing token dependence.
    ///
    /// # Safety
    ///
    /// Inputs and outputs cover the same planes documented by [`Self::launch`]
    /// for `tokens` rows. `state_rows` covers one valid row index. The caller
    /// owns that state row exclusively through completion; allocations are
    /// aligned, non-overlapping, live through completion, and belong to
    /// `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_causal(
        &self,
        stream: &CudaStream,
        tokens: usize,
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
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
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
                        recurrent,
                        output,
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
                "{}GDN causal recurrence token count {tokens} is outside the admitted routes 1..=4",
                E::LABEL
            ))),
        }
    }
}

/// PTX symbols retained for every exact GDN recurrence route.
pub(crate) fn gdn_recurrence_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::gdn_recurrence_prefill_exact_ptx_name::<Qwen38_27B, 1_024>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::gdn_recurrence_prefill_epilogue_exact_ptx_name::<Qwen38_27B, 1_024>(),
    ]
}

/// Target-specific PTX symbols retained for exact Flash-Next recurrence routes.
pub(crate) fn qwen38_flash_next_gdn_recurrence_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 1>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 2>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 3>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 4>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 5>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 6>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 7>(),
        kernels::qwen38_flash_next_gdn_recurrence_exact_ptx_name::<Qwen38FlashNext, 8>(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            1,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            2,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            3,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            4,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            32,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            64,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            128,
        >(),
        kernels::qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_ptx_name::<
            Qwen38FlashNext,
            1_024,
        >(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        GdnRecurrenceEntries, HEAD_DIM, KEY_HEADS, MAX_BATCH, Qwen38FlashNextGdnRecurrenceEntries,
        Qwen38GdnRecurrenceEntries, THREADS, VALUE_HEADS, admitted_batch, admitted_rows,
        gdn_recurrence_ptx_names, qwen38_flash_next_gdn_recurrence_ptx_names,
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
    fn geometry_matches_the_exact_state_contract() {
        assert_eq!(THREADS, 512);
        assert_eq!(VALUE_HEADS / KEY_HEADS, 3);
        assert_eq!(Qwen38_27B::GDN_QK_ROWS, KEY_HEADS * HEAD_DIM);
        assert_eq!(Qwen38_27B::GDN_VALUE_ROWS, VALUE_HEADS * HEAD_DIM);
        assert_eq!(VALUE_HEADS * HEAD_DIM * HEAD_DIM, 786_432);
    }

    #[test]
    fn qwen38_flash_next_reuses_the_exact_qwen38_recurrence_geometry() {
        assert_eq!(
            Qwen38FlashNext::LINEAR_KEY_HEADS,
            Qwen38_27B::LINEAR_KEY_HEADS
        );
        assert_eq!(
            Qwen38FlashNext::LINEAR_VALUE_HEADS,
            Qwen38_27B::LINEAR_VALUE_HEADS
        );
        assert_eq!(
            Qwen38FlashNext::LINEAR_HEAD_DIM,
            Qwen38_27B::LINEAR_HEAD_DIM
        );
        assert_eq!(Qwen38FlashNext::GDN_QKV_ROWS, Qwen38_27B::GDN_QKV_ROWS);
        assert_eq!(Qwen38FlashNext::GDN_INPUT_ROWS, Qwen38_27B::GDN_INPUT_ROWS);
        assert_eq!(
            Qwen38FlashNext::GDN_CONTROL_ROWS,
            Qwen38_27B::GDN_CONTROL_ROWS
        );
        assert_eq!(
            Qwen38FlashNext::RMS_NORM_EPSILON,
            Qwen38_27B::RMS_NORM_EPSILON
        );
    }

    #[test]
    fn every_entry_table_publishes_its_inventory() {
        assert_eq!(Qwen38FlashNext::GDN_OUTPUT_GATE, "sigmoid");
        assert_eq!(
            <Qwen38GdnRecurrenceEntries as GdnRecurrenceEntries<Qwen38_27B>>::ptx_names(),
            gdn_recurrence_ptx_names()
        );
        assert_eq!(
            <Qwen38FlashNextGdnRecurrenceEntries as GdnRecurrenceEntries<Qwen38FlashNext>>::ptx_names(),
            qwen38_flash_next_gdn_recurrence_ptx_names()
        );
    }

    #[test]
    fn ptx_inventory_has_decode_and_prefill_entries() {
        let names = gdn_recurrence_ptx_names();
        let qwen38_flash_next = qwen38_flash_next_gdn_recurrence_ptx_names();
        let unique = names
            .iter()
            .chain(&qwen38_flash_next)
            .copied()
            .collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH + 16);
        assert_eq!(qwen38_flash_next.len(), MAX_BATCH + 8);
        assert_eq!(unique.len(), names.len() + qwen38_flash_next.len());
    }
}

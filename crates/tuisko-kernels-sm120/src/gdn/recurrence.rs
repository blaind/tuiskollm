//! Exact-batch FP32 GDN recurrence and gated normalization.

use crate::Sm120Arch;
use crate::device::gdn_recurrence::gdn_recurrence;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const WARPS: usize = 16;
const THREADS: u32 = (WARPS * 32) as u32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
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
            gdn_recurrence::<A, TOKENS>(
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
}

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    launch: PreparedLaunch<kernels::__gdn_recurrence_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * VALUE_HEADS)
            .map_err(|_| GpuError::invalid_launch("GDN recurrence grid exceeds u32"))?;
        let launch = module
            .prepare_gdn_recurrence_exact::<A, TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(|source| GpuError::launch("preparing GDN recurrence", source))?;

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

/// Prepared FP32 GDN recurrence routes for every exact decode batch.
pub struct GdnRecurrenceOp<A: Sm120Arch = Qwen38_27B> {
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

impl<A: Sm120Arch> GdnRecurrenceOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = gdn_recurrence_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN recurrence module", source))?;

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

    /// Advances mapped FP32 state and emits gated BF16 recurrent values.
    ///
    /// # Safety
    ///
    /// `qkv` and `projected` cover `[batch, A::GDN_QKV_ROWS]` and
    /// `[batch, A::GDN_INPUT_ROWS]` BF16 values. Controls cover
    /// `[batch, A::GDN_CONTROL_ROWS]`; `norm_weight` covers one head;
    /// every state-row index is below the caller-owned `[rows,
    /// A::GDN_CONTROL_ROWS, A::LINEAR_HEAD_DIM, A::LINEAR_HEAD_DIM]` FP32
    /// state; and `output` covers `[batch, A::GDN_VALUE_ROWS]` BF16 values.
    /// Allocations are aligned, non-overlapping, live through completion, and
    /// belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        qkv: *const u16,
        projected: *const u16,
        log_decay: *const f32,
        beta: *const f32,
        norm_weight: *const u16,
        state_rows: *const u32,
        state: *mut f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "GDN recurrence batch {batch} is outside the admitted range 1..={MAX_BATCH}"
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

/// PTX symbols retained for every exact GDN recurrence route.
pub(crate) fn gdn_recurrence_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::gdn_recurrence_exact_ptx_name::<Qwen38_27B, 8>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        HEAD_DIM, KEY_HEADS, MAX_BATCH, THREADS, VALUE_HEADS, admitted_batch,
        gdn_recurrence_ptx_names,
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
    fn geometry_matches_the_exact_state_contract() {
        assert_eq!(THREADS, 512);
        assert_eq!(VALUE_HEADS / KEY_HEADS, 3);
        assert_eq!(Qwen38_27B::GDN_QK_ROWS, KEY_HEADS * HEAD_DIM);
        assert_eq!(Qwen38_27B::GDN_VALUE_ROWS, VALUE_HEADS * HEAD_DIM);
        assert_eq!(VALUE_HEADS * HEAD_DIM * HEAD_DIM, 786_432);
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_batch() {
        let names = gdn_recurrence_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(unique.len(), names.len());
    }
}

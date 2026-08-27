//! Bit-preserving snapshot of one mapped GDN history and recurrent-state row.

use crate::device::gdn_state_snapshot::gdn_state_snapshot;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen38_27B, Qwen38FlashNext};

const THREADS: u32 = 256;
const VECTOR_BYTES: usize = size_of::<u128>();

const _: () = assert!(Qwen38FlashNext::GDN_QKV_ROWS == Qwen38_27B::GDN_QKV_ROWS);
const _: () = assert!(Qwen38FlashNext::GDN_CONTROL_ROWS == Qwen38_27B::GDN_CONTROL_ROWS);
const _: () = assert!(Qwen38FlashNext::LINEAR_HEAD_DIM == Qwen38_27B::LINEAR_HEAD_DIM);
const _: () =
    assert!(Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM == Qwen38_27B::LINEAR_CONV_KERNEL_DIM);

fn history_bytes<A: Arch>() -> usize {
    A::GDN_QKV_ROWS * (A::LINEAR_CONV_KERNEL_DIM - 1) * size_of::<u16>()
}

fn state_bytes<A: Arch>() -> usize {
    A::GDN_CONTROL_ROWS * A::LINEAR_HEAD_DIM * A::LINEAR_HEAD_DIM * size_of::<f32>()
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::LINEAR_CONV_KERNEL_DIM != 4
        || !history_bytes::<A>().is_multiple_of(VECTOR_BYTES)
        || !state_bytes::<A>().is_multiple_of(VECTOR_BYTES)
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the GDN snapshot schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Copies one selected persistent row into the provisional target workspace.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn gdn_state_snapshot_exact<A: Arch>(
        source_row: *const u32,
        history: *const u128,
        state: *const u128,
        scratch_history: *mut u128,
        scratch_state: *mut u128,
    ) {
        // The exact row is 200,448 aligned 16-byte words. Seven hundred
        // eighty-three CTAs fill the 170-SM device while reducing the copy to
        // one represented-value load/store pair per thread.
        unsafe {
            gdn_state_snapshot::<A>(source_row, history, state, scratch_history, scratch_state);
        }
    }
}

/// Prepared bit-preserving snapshot used before provisional target verification.
pub struct GdnStateSnapshotOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    launch: PreparedLaunch<kernels::__gdn_state_snapshot_exact_CudaKernel<A>>,
}

impl<A: Sm120Arch> GdnStateSnapshotOp<A> {
    /// Loads the embedded SM120 module and prepares the exact row-copy route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let words = (history_bytes::<A>() + state_bytes::<A>()) / VECTOR_BYTES;
        let blocks = u32::try_from(words.div_ceil(THREADS as usize))
            .map_err(|_| GpuError::invalid_launch("GDN snapshot grid exceeds u32"))?;
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the GDN snapshot module", source))?;
        let launch = module
            .prepare_gdn_state_snapshot_exact::<A>(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(|source| GpuError::launch("preparing GDN state snapshot", source))?;

        Ok(Self { module, launch })
    }

    /// Copies one selected persistent row into the provisional scratch row.
    ///
    /// # Safety
    ///
    /// `source_row` covers one valid row index. `history` and `state` cover all
    /// persistent rows; scratch destinations cover exactly one history and one
    /// state row. All pointers are 16-byte aligned, non-overlapping, live
    /// through completion, and belong to `stream`'s context.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        source_row: *const u32,
        history: *const u16,
        state: *const f32,
        scratch_history: *mut u16,
        scratch_state: *mut f32,
    ) -> GpuResult<()> {
        self.module
            .gdn_state_snapshot_exact::<A>(
                stream,
                &self.launch,
                source_row,
                history.cast::<u128>(),
                state.cast::<u128>(),
                scratch_history.cast::<u128>(),
                scratch_state.cast::<u128>(),
            )
            .map_err(|source| GpuError::launch("launching GDN state snapshot", source))
    }
}

/// Flash-Next alias for the exact Qwen3.8-27B snapshot byte extents.
pub type Qwen38FlashNextGdnStateSnapshotOp = GdnStateSnapshotOp<Qwen38_27B>;

/// PTX symbol retained for the exact GDN state snapshot route.
pub(crate) fn gdn_state_snapshot_ptx_name() -> &'static str {
    kernels::gdn_state_snapshot_exact_ptx_name::<Qwen38_27B>()
}

#[cfg(test)]
mod tests {
    use super::{THREADS, VECTOR_BYTES, history_bytes, state_bytes};
    use tuisko_model::{Qwen38_27B, Qwen38FlashNext};

    #[test]
    fn exact_snapshot_geometry_is_fully_vectorized() {
        let history = history_bytes::<Qwen38_27B>();
        let state = state_bytes::<Qwen38_27B>();

        assert_eq!(history, 61_440);
        assert_eq!(state, 3_145_728);
        assert_eq!(VECTOR_BYTES, 16);
        assert_eq!((history + state) / VECTOR_BYTES, 200_448);
        assert_eq!(
            ((history + state) / VECTOR_BYTES).div_ceil(THREADS as usize),
            783
        );
    }

    #[test]
    fn qwen38_flash_next_snapshot_extent_matches_the_aliased_route() {
        assert_eq!(
            history_bytes::<Qwen38FlashNext>(),
            history_bytes::<Qwen38_27B>()
        );
        assert_eq!(
            state_bytes::<Qwen38FlashNext>(),
            state_bytes::<Qwen38_27B>()
        );
    }
}

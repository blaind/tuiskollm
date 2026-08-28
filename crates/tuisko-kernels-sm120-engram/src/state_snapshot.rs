//! Bit-preserving snapshot and restore of one engram convolution slot.
//!
//! A provisional step mutates nine BF16 history columns per channel. Capture
//! and restore use the same 11,520 aligned 16-byte vectors.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::Qwen38FlashNext;

const THREADS: u32 = 256;
const VECTOR_BYTES: usize = size_of::<u128>();
/// BF16 columns one convolution slot carries.
const SLOT_VALUES: usize = Qwen38FlashNext::HC_WIDTH * Qwen38FlashNext::PLE_CONV_STATE_LEN;
/// Aligned 16-byte vectors in one convolution slot.
const SLOT_VECTORS: usize = SLOT_VALUES * size_of::<u16>() / VECTOR_BYTES;
/// CTAs that cover one slot with one vector per thread.
const BLOCKS: u32 = SLOT_VECTORS.div_ceil(THREADS as usize) as u32;

const _: () = assert!((SLOT_VALUES * size_of::<u16>()).is_multiple_of(VECTOR_BYTES));
const _: () = assert!(SLOT_VECTORS.is_multiple_of(THREADS as usize));

#[cuda_module]
mod kernels {
    use super::*;

    /// Copies one selected convolution slot between the persistent plane and
    /// the provisional scratch row.
    ///
    /// `RESTORE` selects the direction. Both arms move the identical vectors,
    /// so a slot put back through the restore arm is bit-identical to what the
    /// capture read. It is one kernel rather than two so the two directions
    /// can never drift apart in extent or traversal.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_ple_state_copy_exact<const RESTORE: bool>(
        slot_row: *const u32,
        state: *mut u128,
        scratch: *mut u128,
    ) {
        let vector = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if vector >= SLOT_VECTORS {
            return;
        }

        // SAFETY: the caller names one valid slot index.
        let row = unsafe { *slot_row as usize };
        // SAFETY: the persistent plane holds `SLOT_VECTORS` vectors per slot.
        let persistent = unsafe { state.add(row * SLOT_VECTORS + vector) };
        // SAFETY: the scratch row holds exactly one slot.
        let provisional = unsafe { scratch.add(vector) };

        if RESTORE {
            // SAFETY: one thread owns this vector of the selected slot.
            unsafe { *persistent = *provisional };
        } else {
            // SAFETY: one thread owns this vector of the scratch row.
            unsafe { *provisional = *persistent };
        }
    }
}

/// Prepared bit-preserving capture and restore of one engram convolution slot.
pub struct Qwen38FlashNextPleStateSnapshotOp {
    module: kernels::LoadedModule,
    snapshot: PreparedLaunch<kernels::__qwen38_flash_next_ple_state_copy_exact_CudaKernel<false>>,
    restore: PreparedLaunch<kernels::__qwen38_flash_next_ple_state_copy_exact_CudaKernel<true>>,
}

impl Qwen38FlashNextPleStateSnapshotOp {
    /// Loads the embedded SM120 module and prepares both copy directions.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the engram snapshot module", source))?;
        let snapshot = module
            .prepare_qwen38_flash_next_ple_state_copy_exact::<false>(LaunchConfig1D::new(
                BLOCKS, THREADS, 0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram state snapshot", source))?;
        let restore = module
            .prepare_qwen38_flash_next_ple_state_copy_exact::<true>(LaunchConfig1D::new(
                BLOCKS, THREADS, 0,
            ))
            .map_err(|source| GpuError::launch("preparing the engram state restore", source))?;

        Ok(Self {
            module,
            snapshot,
            restore,
        })
    }

    /// Captures the named convolution slot into the provisional scratch row.
    ///
    /// # Safety
    ///
    /// `slot_row` names one valid slot. `state` covers all persistent slots and
    /// `scratch` exactly one. All pointers are 16-byte aligned, do not overlap,
    /// live through completion, and belong to `stream`'s context.
    pub unsafe fn launch_snapshot(
        &self,
        stream: &CudaStream,
        slot_row: *const u32,
        state: *mut u16,
        scratch: *mut u16,
    ) -> GpuResult<()> {
        self.module
            .qwen38_flash_next_ple_state_copy_exact::<false>(
                stream,
                &self.snapshot,
                slot_row,
                state.cast::<u128>(),
                scratch.cast::<u128>(),
            )
            .map_err(|source| GpuError::launch("launching the engram state snapshot", source))
    }

    /// Puts a captured convolution slot back over the named persistent slot.
    ///
    /// # Safety
    ///
    /// [`Self::launch_snapshot`]'s contract, with the roles reversed.
    pub unsafe fn launch_restore(
        &self,
        stream: &CudaStream,
        slot_row: *const u32,
        state: *mut u16,
        scratch: *mut u16,
    ) -> GpuResult<()> {
        self.module
            .qwen38_flash_next_ple_state_copy_exact::<true>(
                stream,
                &self.restore,
                slot_row,
                state.cast::<u128>(),
                scratch.cast::<u128>(),
            )
            .map_err(|source| GpuError::launch("launching the engram state restore", source))
    }
}

/// PTX symbols retained for the exact engram slot capture and restore.
pub(crate) fn ple_state_snapshot_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_ple_state_copy_exact_ptx_name::<false>(),
        kernels::qwen38_flash_next_ple_state_copy_exact_ptx_name::<true>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{BLOCKS, SLOT_VALUES, SLOT_VECTORS, THREADS, VECTOR_BYTES};
    use tuisko_model::{Arch, Qwen38FlashNext};

    #[test]
    fn exact_slot_geometry_is_fully_vectorized() {
        assert_eq!(SLOT_VALUES, 92_160);
        assert_eq!(SLOT_VALUES * size_of::<u16>(), 184_320);
        assert_eq!(VECTOR_BYTES, 16);
        assert_eq!(SLOT_VECTORS, 11_520);
        assert_eq!(BLOCKS, 45);
        assert_eq!(SLOT_VECTORS % THREADS as usize, 0);
    }

    /// The slot the snapshot copies is exactly the plane the convolution
    /// advances, so the two can never disagree on extent.
    #[test]
    fn the_snapshot_extent_is_the_convolution_state_extent() {
        assert_eq!(
            SLOT_VALUES,
            Qwen38FlashNext::HC_WIDTH * Qwen38FlashNext::PLE_CONV_STATE_LEN
        );
        assert_ne!(
            SLOT_VALUES,
            Qwen38FlashNext::GDN_QKV_ROWS * (Qwen38FlashNext::LINEAR_CONV_KERNEL_DIM - 1)
        );
    }
}

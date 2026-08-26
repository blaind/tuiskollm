//! Exact-batch zero-centered RMSNorm operators.

use crate::Sm89Arch;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

// Compact batching owns one compiled route for every B=1..8.
const MAX_BATCH: usize = 8;
// The first portable route preserves the SM120 reduction order: 2,560 packed BF16
// pairs / 512 threads = five pairs per thread. Target qualification owns any later retune.
const WARPS: usize = 16;
const THREADS: u32 = (WARPS * 32) as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidualNormGeometry {
    pairs_per_row: usize,
    pairs_per_thread: usize,
}

fn residual_norm_geometry<A: Arch>() -> Option<ResidualNormGeometry> {
    if A::HIDDEN == 0 || !A::HIDDEN.is_multiple_of(2) {
        return None;
    }

    let pairs_per_row = A::HIDDEN / 2;
    Some(ResidualNormGeometry {
        pairs_per_row,
        pairs_per_thread: pairs_per_row.div_ceil(THREADS as usize),
    })
}

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, warp};
    use cuda_intrinsics::convert::cvt_f32x2_bf16x2;

    #[inline(always)]
    fn block_sum(value: f32, shared: *mut f32, lane: usize, warp_index: usize) -> f32 {
        let value = warp::reduce_sum_f32(value);
        if lane == 0 {
            // SAFETY: one lane writes its warp's unique shared slot.
            unsafe { *shared.add(warp_index) = value };
        }
        thread::sync_threads();

        if warp_index == 0 {
            let value = if lane < WARPS {
                // SAFETY: the first warp reads the initialized warp-sum slots.
                unsafe { *shared.add(lane) }
            } else {
                0.0
            };
            let value = warp::reduce_sum_f32(value);
            if lane == 0 {
                // SAFETY: lane zero publishes the block sum before the barrier.
                unsafe { *shared = value };
            }
        }
        thread::sync_threads();

        // SAFETY: the second barrier makes the published block sum visible.
        unsafe { *shared }
    }

    #[inline(always)]
    fn rms_norm_body<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
        shared: *mut f32,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let pairs = A::HIDDEN / 2;
        // SAFETY: the launch contract requires one complete row per active block.
        let input = unsafe { input.add(token * pairs) };
        // SAFETY: the output contract matches the input row coverage.
        let output = unsafe { output.add(token * pairs) };
        let mut sum = 0.0f32;
        let mut pair = tid;

        while pair < pairs {
            // SAFETY: `pair < pairs` and this block owns `token`.
            let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
            sum = float::fma_rn_f32(low, low, sum);
            sum = float::fma_rn_f32(high, high, sum);
            pair += THREADS as usize;
        }

        sum = block_sum(sum, shared, lane, warp_index);
        let inverse_rms = float::rsqrt_approx_f32(sum / A::HIDDEN as f32 + A::RMS_NORM_EPSILON);
        pair = tid;

        while pair < pairs {
            // SAFETY: both reads are within one complete hidden row.
            let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
            // SAFETY: the weight plane contains one complete hidden row.
            let (wlow, whigh) = convert::cvt_f32x2_bf16x2(unsafe { *weight.add(pair) });
            // SAFETY: each thread writes disjoint packed BF16 pairs.
            unsafe {
                *output.add(pair) = cvt_f32x2_bf16x2(
                    low * inverse_rms * (1.0 + wlow),
                    high * inverse_rms * (1.0 + whigh),
                );
            }
            pair += THREADS as usize;
        }
    }

    /// Normalizes the singleton BF16 row and pins this module's artifact.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 9),
    )]
    pub fn rms_norm_b1(input: *const u32, weight: *const u32, output: *mut u32) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        rms_norm_body::<Qwen38_27B, 1>(input, weight, output, shared);
    }

    /// Normalizes `TOKENS` BF16 rows with zero-centered BF16 weights.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 9),
    )]
    pub fn rms_norm<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        rms_norm_body::<A, TOKENS>(input, weight, output, shared);
    }

    /// Publishes a BF16 residual sum and normalizes that represented value.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 9),
    )]
    pub fn residual_rms_norm<A: Arch, const TOKENS: usize>(
        residual_input: *const u32,
        branch: *const u32,
        weight: *const u32,
        residual_output: *mut u32,
        normalized_output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let pairs = A::HIDDEN / 2;
        // SAFETY: the launch contract requires one complete row per active block.
        let residual_input = unsafe { residual_input.add(token * pairs) };
        // SAFETY: the branch plane has the same row coverage.
        let branch = unsafe { branch.add(token * pairs) };
        // SAFETY: the two output planes have the same row coverage.
        let residual_output = unsafe { residual_output.add(token * pairs) };
        // SAFETY: the two output planes have the same row coverage.
        let normalized_output = unsafe { normalized_output.add(token * pairs) };
        let mut sum = 0.0f32;
        let mut pair = tid;

        while pair < pairs {
            // SAFETY: both reads are within this block's row.
            let (xlow, xhigh) = convert::cvt_f32x2_bf16x2(unsafe { *residual_input.add(pair) });
            // SAFETY: both reads are within this block's row.
            let (blow, bhigh) = convert::cvt_f32x2_bf16x2(unsafe { *branch.add(pair) });
            let represented = cvt_f32x2_bf16x2(xlow + blow, xhigh + bhigh);
            // SAFETY: each thread publishes one disjoint packed BF16 pair.
            unsafe { *residual_output.add(pair) = represented };
            let (low, high) = convert::cvt_f32x2_bf16x2(represented);
            sum = float::fma_rn_f32(low, low, sum);
            sum = float::fma_rn_f32(high, high, sum);
            pair += THREADS as usize;
        }

        sum = block_sum(sum, shared, lane, warp_index);
        let inverse_rms = float::rsqrt_approx_f32(sum / A::HIDDEN as f32 + A::RMS_NORM_EPSILON);
        pair = tid;

        while pair < pairs {
            // SAFETY: the published residual and weights cover a complete row.
            let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *residual_output.add(pair) });
            // SAFETY: the weight plane contains one complete hidden row.
            let (wlow, whigh) = convert::cvt_f32x2_bf16x2(unsafe { *weight.add(pair) });
            // SAFETY: each thread writes disjoint packed BF16 pairs.
            unsafe {
                *normalized_output.add(pair) = cvt_f32x2_bf16x2(
                    low * inverse_rms * (1.0 + wlow),
                    high * inverse_rms * (1.0 + whigh),
                );
            }
            pair += THREADS as usize;
        }
    }
}

tuisko_kernels_simt::residual_norm_batch_routes!();

/// PTX symbols retained for both residual-norm families at every exact batch.
pub(crate) fn residual_norm_ptx_names() -> [&'static str; 16] {
    [
        "rms_norm_b1",
        kernels::rms_norm_ptx_name::<Qwen38_27B, 2>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 3>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 4>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 5>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 6>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 7>(),
        kernels::rms_norm_ptx_name::<Qwen38_27B, 8>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 1>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 2>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 3>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 4>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 5>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 6>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 7>(),
        kernels::residual_rms_norm_ptx_name::<Qwen38_27B, 8>(),
    ]
}

/// Prepared plain and fused-residual RMSNorm routes for every exact batch `1..=8`.
pub struct ResidualNormOp<A: Sm89Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedBatchOneRoute,
    b2: PreparedBatchRoute<A, 2>,
    b3: PreparedBatchRoute<A, 3>,
    b4: PreparedBatchRoute<A, 4>,
    b5: PreparedBatchRoute<A, 5>,
    b6: PreparedBatchRoute<A, 6>,
    b7: PreparedBatchRoute<A, 7>,
    b8: PreparedBatchRoute<A, 8>,
}

impl<A: Sm89Arch> ResidualNormOp<A> {
    /// Loads the embedded SM89 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        residual_norm_geometry::<A>().ok_or_else(|| {
            GpuError::invalid_launch("RMSNorm requires a positive even hidden width")
        })?;
        let _ = residual_norm_ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the residual-norm module", source))?;

        Ok(Self {
            b1: PreparedBatchOneRoute::prepare(&module)?,
            b2: PreparedBatchRoute::prepare(&module)?,
            b3: PreparedBatchRoute::prepare(&module)?,
            b4: PreparedBatchRoute::prepare(&module)?,
            b5: PreparedBatchRoute::prepare(&module)?,
            b6: PreparedBatchRoute::prepare(&module)?,
            b7: PreparedBatchRoute::prepare(&module)?,
            b8: PreparedBatchRoute::prepare(&module)?,
            module,
        })
    }

    /// Launches the plain RMSNorm route for exactly `batch` complete rows.
    ///
    /// # Safety
    ///
    /// The pointers must be four-byte aligned and cover complete `A::HIDDEN`
    /// rows. Their allocations must belong to `stream`'s context and
    /// remain live through stream completion. Input and output must not overlap.
    pub unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    self.$route
                        .launch_plain(&self.module, stream, input, weight, output)
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
            _ => Err(GpuError::invalid_launch(format!(
                "RMSNorm batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }

    /// Publishes BF16 residual sums and launches their next RMSNorm.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned. Row planes must cover
    /// `batch * A::HIDDEN` BF16 values and `weight` must cover `A::HIDDEN` values.
    /// Allocations must belong to `stream`'s context, remain live through
    /// completion, and not overlap except that the two input planes may alias.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the public method's pointer contract is unchanged by dispatch.
                unsafe {
                    self.$route.launch_residual(
                        &self.module,
                        stream,
                        residual_input,
                        branch,
                        weight,
                        residual_output,
                        normalized_output,
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
            _ => Err(GpuError::invalid_launch(format!(
                "residual RMSNorm batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, THREADS, WARPS, residual_norm_geometry, residual_norm_ptx_names};
    use crate::test_arch::TestArch;
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn exact_geometry_is_pair_and_cta_aligned() {
        assert_eq!(THREADS, 512);
        assert_eq!(WARPS, 16);
        assert_eq!(Qwen38_27B::HIDDEN % 2, 0);
        assert_eq!((Qwen38_27B::HIDDEN / 2) % THREADS as usize, 0);
        assert_eq!((Qwen38_27B::HIDDEN / 2) / THREADS as usize, 5);
        assert_eq!(MAX_BATCH, 8);
    }

    #[test]
    fn geometry_flows_from_the_architecture() {
        let qwen = residual_norm_geometry::<Qwen38_27B>().unwrap();
        let test = residual_norm_geometry::<TestArch>().unwrap();

        assert_eq!(qwen.pairs_per_row, 2_560);
        assert_eq!(qwen.pairs_per_thread, 5);
        assert_eq!(test.pairs_per_row, 512);
        assert_eq!(test.pairs_per_thread, 1);
    }

    #[test]
    fn ptx_inventory_has_two_distinct_entries_per_batch() {
        let names = residual_norm_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(unique.len(), names.len());
    }
}

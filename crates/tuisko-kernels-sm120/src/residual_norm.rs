//! Exact-batch zero-centered RMSNorm operators.

use crate::Sm120Arch;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

// Compact batching owns one compiled route for every B=1..8.
const MAX_BATCH: usize = 8;
// At 2,197 MHz the 512-thread Qwen3.5 B=1 path measures 2.283/2.572 us
// plain/fused. Its 2,048 packed pairs map to four per thread versus five for
// Qwen3.8; retaining the 16-warp reduction preserves the qualified topology.
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
    use cuda_device::{convert, float, tcgen05, warp};

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
                *output.add(pair) = tcgen05::cvt_f32x2_bf16x2(
                    low * inverse_rms * (1.0 + wlow),
                    high * inverse_rms * (1.0 + whigh),
                );
            }
            pair += THREADS as usize;
        }
    }

    #[inline(always)]
    fn residual_rms_norm_body<A: Arch, const TOKENS: usize>(
        residual_input: *const u32,
        branch: *const u32,
        weight: *const u32,
        residual_output: *mut u32,
        normalized_output: *mut u32,
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
            let represented = tcgen05::cvt_f32x2_bf16x2(xlow + blow, xhigh + bhigh);
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
                *normalized_output.add(pair) = tcgen05::cvt_f32x2_bf16x2(
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
        min_compute_capability = (12, 0),
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
        min_compute_capability = (12, 0),
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

    /// Normalizes exact Qwen3.5 BF16 rows with zero-centered weights.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_rms_norm<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        rms_norm_body::<Qwen35_9B, TOKENS>(input, weight, output, shared);
    }

    /// Publishes and normalizes exact Qwen3.5 BF16 residual rows.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_residual_rms_norm<const TOKENS: usize>(
        residual_input: *const u32,
        branch: *const u32,
        weight: *const u32,
        residual_output: *mut u32,
        normalized_output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        residual_rms_norm_body::<Qwen35_9B, TOKENS>(
            residual_input,
            branch,
            weight,
            residual_output,
            normalized_output,
            shared,
        );
    }

    /// Publishes a BF16 residual sum and normalizes one exact decode batch.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
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

        residual_rms_norm_body::<A, TOKENS>(
            residual_input,
            branch,
            weight,
            residual_output,
            normalized_output,
            shared,
        );
    }

    /// Normalizes one exact prefill width.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn rms_norm_prefill<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        // One 512-thread CTA owns each row: Qwen3.5/3.6/3.8 consume exactly
        // four/two/five packed pairs per thread. Exact T changes only CTA count.
        rms_norm_body::<A, TOKENS>(input, weight, output, shared);
    }

    /// Publishes and normalizes one exact prefill width.
    #[kernel]
    #[launch_bounds(512, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (512, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn residual_rms_norm_prefill<A: Arch, const TOKENS: usize>(
        residual_input: *const u32,
        branch: *const u32,
        weight: *const u32,
        residual_output: *mut u32,
        normalized_output: *mut u32,
    ) {
        static mut WARP_SUM: SharedArray<f32, WARPS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(WARP_SUM).cast::<f32>();

        residual_rms_norm_body::<A, TOKENS>(
            residual_input,
            branch,
            weight,
            residual_output,
            normalized_output,
            shared,
        );
    }
}

// Prepared generic entries for one exact batch.
struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__rms_norm_CudaKernel<A, TOKENS>>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<A, TOKENS>>,
}

// B=1 keeps the concrete plain entry that anchors the embedded module artifact.
struct PreparedBatchOneRoute {
    plain: PreparedLaunch<kernels::__rms_norm_b1_CudaKernel>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<Qwen38_27B, 1>>,
}

impl PreparedBatchOneRoute {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let launch = LaunchConfig1D::new(1, THREADS, 0);
        let plain = module
            .prepare_rms_norm_b1(launch)
            .map_err(|source| GpuError::launch("preparing the B=1 RMSNorm kernel", source))?;
        let residual = module
            .prepare_residual_rms_norm::<Qwen38_27B, 1>(launch)
            .map_err(|source| {
                GpuError::launch("preparing the B=1 residual RMSNorm kernel", source)
            })?;

        Ok(Self { plain, residual })
    }

    unsafe fn launch_plain(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .rms_norm_b1(
                stream,
                &self.plain,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the B=1 RMSNorm kernel", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .residual_rms_norm::<Qwen38_27B, 1>(
                stream,
                &self.residual,
                residual_input.cast::<u32>(),
                branch.cast::<u32>(),
                weight.cast::<u32>(),
                residual_output.cast::<u32>(),
                normalized_output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the B=1 residual RMSNorm kernel", source))
    }
}

impl<A: Arch, const TOKENS: usize> PreparedBatchRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS)
            .map_err(|_| GpuError::invalid_launch("RMSNorm batch exceeds CUDA grid width"))?;
        let launch = LaunchConfig1D::new(blocks, THREADS, 0);
        let plain = module
            .prepare_rms_norm::<A, TOKENS>(launch)
            .map_err(|source| GpuError::launch("preparing the RMSNorm kernel", source))?;
        let residual = module
            .prepare_residual_rms_norm::<A, TOKENS>(launch)
            .map_err(|source| GpuError::launch("preparing the residual RMSNorm kernel", source))?;

        Ok(Self { plain, residual })
    }

    unsafe fn launch_plain(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .rms_norm::<A, TOKENS>(
                stream,
                &self.plain,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the RMSNorm kernel", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .residual_rms_norm::<A, TOKENS>(
                stream,
                &self.residual,
                residual_input.cast::<u32>(),
                branch.cast::<u32>(),
                weight.cast::<u32>(),
                residual_output.cast::<u32>(),
                normalized_output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the residual RMSNorm kernel", source))
    }
}

struct PreparedQwen35BatchRoute<const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__qwen35_rms_norm_CudaKernel<TOKENS>>,
    residual: PreparedLaunch<kernels::__qwen35_residual_rms_norm_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35BatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 RMSNorm batch exceeds CUDA grid width")
        })?;
        let launch = LaunchConfig1D::new(blocks, THREADS, 0);
        let plain = module
            .prepare_qwen35_rms_norm::<TOKENS>(launch)
            .map_err(|source| GpuError::launch("preparing the Qwen3.5 RMSNorm kernel", source))?;
        let residual = module
            .prepare_qwen35_residual_rms_norm::<TOKENS>(launch)
            .map_err(|source| {
                GpuError::launch("preparing the Qwen3.5 residual RMSNorm kernel", source)
            })?;

        Ok(Self { plain, residual })
    }

    unsafe fn launch_plain(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_rms_norm::<TOKENS>(
                stream,
                &self.plain,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the Qwen3.5 RMSNorm kernel", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_residual_rms_norm::<TOKENS>(
                stream,
                &self.residual,
                residual_input.cast::<u32>(),
                branch.cast::<u32>(),
                weight.cast::<u32>(),
                residual_output.cast::<u32>(),
                normalized_output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 residual RMSNorm kernel", source)
            })
    }
}

// Prefill retains separate symbols so its resource authority cannot drift with decode.
struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__rms_norm_prefill_CudaKernel<A, TOKENS>>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_prefill_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS)
            .map_err(|_| GpuError::invalid_launch("RMSNorm prefill exceeds CUDA grid width"))?;
        let launch = LaunchConfig1D::new(blocks, THREADS, 0);
        let plain = module
            .prepare_rms_norm_prefill::<A, TOKENS>(launch)
            .map_err(|source| GpuError::launch("preparing the RMSNorm prefill kernel", source))?;
        let residual = module
            .prepare_residual_rms_norm_prefill::<A, TOKENS>(launch)
            .map_err(|source| {
                GpuError::launch("preparing the residual RMSNorm prefill kernel", source)
            })?;

        Ok(Self { plain, residual })
    }

    unsafe fn launch_plain(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .rms_norm_prefill::<A, TOKENS>(
                stream,
                &self.plain,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the RMSNorm prefill kernel", source))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        residual_input: *const u16,
        branch: *const u16,
        weight: *const u16,
        residual_output: *mut u16,
        normalized_output: *mut u16,
    ) -> GpuResult<()> {
        module
            .residual_rms_norm_prefill::<A, TOKENS>(
                stream,
                &self.residual,
                residual_input.cast::<u32>(),
                branch.cast::<u32>(),
                weight.cast::<u32>(),
                residual_output.cast::<u32>(),
                normalized_output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the residual RMSNorm prefill kernel", source)
            })
    }
}

/// PTX symbols retained for both residual-norm families at every exact batch.
pub(crate) fn residual_norm_ptx_names() -> Vec<&'static str> {
    vec![
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
        kernels::rms_norm_prefill_ptx_name::<Qwen38_27B, 32>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen38_27B, 64>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen38_27B, 128>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen38_27B, 1024>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen38_27B, 32>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen38_27B, 64>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen38_27B, 128>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen38_27B, 1024>(),
    ]
}

/// PTX symbols retained for Qwen3.5 plain and fused-residual routes.
pub(crate) fn qwen35_residual_norm_ptx_names() -> [&'static str; 22] {
    [
        kernels::qwen35_rms_norm_ptx_name::<1>(),
        kernels::qwen35_rms_norm_ptx_name::<2>(),
        kernels::qwen35_rms_norm_ptx_name::<3>(),
        kernels::qwen35_rms_norm_ptx_name::<4>(),
        kernels::qwen35_rms_norm_ptx_name::<5>(),
        kernels::qwen35_rms_norm_ptx_name::<6>(),
        kernels::qwen35_rms_norm_ptx_name::<7>(),
        kernels::qwen35_rms_norm_ptx_name::<8>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<1>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<2>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<3>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<4>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<5>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<6>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<7>(),
        kernels::qwen35_residual_rms_norm_ptx_name::<8>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen35_9B, 32>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen35_9B, 64>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen35_9B, 128>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen35_9B, 32>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen35_9B, 64>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen35_9B, 128>(),
    ]
}

/// PTX symbols retained for Qwen3.6 plain and fused-residual routes.
pub(crate) fn qwen36_residual_norm_ptx_names() -> [&'static str; 22] {
    [
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 1>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 2>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 3>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 4>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 5>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 6>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 7>(),
        kernels::rms_norm_ptx_name::<Qwen36Moe35B, 8>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 1>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 2>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 3>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 4>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 5>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 6>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 7>(),
        kernels::residual_rms_norm_ptx_name::<Qwen36Moe35B, 8>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen36Moe35B, 32>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen36Moe35B, 64>(),
        kernels::rms_norm_prefill_ptx_name::<Qwen36Moe35B, 128>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen36Moe35B, 32>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen36Moe35B, 64>(),
        kernels::residual_rms_norm_prefill_ptx_name::<Qwen36Moe35B, 128>(),
    ]
}

/// Prepared RMSNorm routes for decode `B=1..=8` and prefill `T=32,64,128,1024`.
pub struct ResidualNormOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedBatchOneRoute,
    b2: PreparedBatchRoute<A, 2>,
    b3: PreparedBatchRoute<A, 3>,
    b4: PreparedBatchRoute<A, 4>,
    b5: PreparedBatchRoute<A, 5>,
    b6: PreparedBatchRoute<A, 6>,
    b7: PreparedBatchRoute<A, 7>,
    b8: PreparedBatchRoute<A, 8>,
    t32: PreparedPrefillRoute<A, 32>,
    t64: PreparedPrefillRoute<A, 64>,
    t128: PreparedPrefillRoute<A, 128>,
    t1024: PreparedPrefillRoute<A, 1024>,
}

impl<A: Sm120Arch> ResidualNormOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
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
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Launches the plain RMSNorm route for one exact decode or prefill row count.
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
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            1024 => launch!(t1024),
            _ => Err(GpuError::invalid_launch(format!(
                "RMSNorm row count {batch} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128,1024"
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
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            1024 => launch!(t1024),
            _ => Err(GpuError::invalid_launch(format!(
                "residual RMSNorm row count {batch} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128,1024"
            ))),
        }
    }
}

/// Prepared Qwen3.5 RMSNorm routes for decode `B=1..8` and prefill `T=32,64,128`.
pub struct Qwen35ResidualNormOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen35BatchRoute<1>,
    b2: PreparedQwen35BatchRoute<2>,
    b3: PreparedQwen35BatchRoute<3>,
    b4: PreparedQwen35BatchRoute<4>,
    b5: PreparedQwen35BatchRoute<5>,
    b6: PreparedQwen35BatchRoute<6>,
    b7: PreparedQwen35BatchRoute<7>,
    b8: PreparedQwen35BatchRoute<8>,
    t32: PreparedPrefillRoute<Qwen35_9B, 32>,
    t64: PreparedPrefillRoute<Qwen35_9B, 64>,
    t128: PreparedPrefillRoute<Qwen35_9B, 128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35RowRoute {
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    T32,
    T64,
    T128,
}

fn qwen35_row_route(rows: usize) -> Option<Qwen35RowRoute> {
    match rows {
        1 => Some(Qwen35RowRoute::B1),
        2 => Some(Qwen35RowRoute::B2),
        3 => Some(Qwen35RowRoute::B3),
        4 => Some(Qwen35RowRoute::B4),
        5 => Some(Qwen35RowRoute::B5),
        6 => Some(Qwen35RowRoute::B6),
        7 => Some(Qwen35RowRoute::B7),
        8 => Some(Qwen35RowRoute::B8),
        32 => Some(Qwen35RowRoute::T32),
        64 => Some(Qwen35RowRoute::T64),
        128 => Some(Qwen35RowRoute::T128),
        _ => None,
    }
}

impl Qwen35ResidualNormOp {
    /// Loads the embedded SM120 module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let geometry = residual_norm_geometry::<Qwen35_9B>().ok_or_else(|| {
            GpuError::invalid_launch("Qwen3.5 RMSNorm requires a positive even hidden width")
        })?;
        if geometry.pairs_per_thread * THREADS as usize != geometry.pairs_per_row {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 RMSNorm requires an exact packed-pair/thread mapping",
            ));
        }
        let _ = qwen35_residual_norm_ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the residual-norm module", source))?;

        // T=128 is 128 independent 4,096-wide rows. One 128-CTA launch
        // removes 15 boundaries versus sixteen B=8 launches while each CTA
        // retains the same 512-thread traversal and reduction order.
        Ok(Self {
            b1: PreparedQwen35BatchRoute::prepare(&module)?,
            b2: PreparedQwen35BatchRoute::prepare(&module)?,
            b3: PreparedQwen35BatchRoute::prepare(&module)?,
            b4: PreparedQwen35BatchRoute::prepare(&module)?,
            b5: PreparedQwen35BatchRoute::prepare(&module)?,
            b6: PreparedQwen35BatchRoute::prepare(&module)?,
            b7: PreparedQwen35BatchRoute::prepare(&module)?,
            b8: PreparedQwen35BatchRoute::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Launches plain zero-centered RMSNorm for one admitted row count.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned and cover complete 4,096-value rows.
    /// Allocations must belong to `stream`'s context, remain live through stream
    /// completion, and input and output must not overlap.
    pub unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        rows: usize,
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

        match qwen35_row_route(rows) {
            Some(Qwen35RowRoute::B1) => launch!(b1),
            Some(Qwen35RowRoute::B2) => launch!(b2),
            Some(Qwen35RowRoute::B3) => launch!(b3),
            Some(Qwen35RowRoute::B4) => launch!(b4),
            Some(Qwen35RowRoute::B5) => launch!(b5),
            Some(Qwen35RowRoute::B6) => launch!(b6),
            Some(Qwen35RowRoute::B7) => launch!(b7),
            Some(Qwen35RowRoute::B8) => launch!(b8),
            Some(Qwen35RowRoute::T32) => launch!(t32),
            Some(Qwen35RowRoute::T64) => launch!(t64),
            Some(Qwen35RowRoute::T128) => launch!(t128),
            None => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 RMSNorm row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128"
            ))),
        }
    }

    /// Publishes BF16 residual sums and normalizes the represented rows.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned. Row planes must cover
    /// `rows * 4,096` BF16 values and `weight` must cover 4,096 values.
    /// Allocations must belong to `stream`'s context, remain live through
    /// completion, and not overlap except that the two input planes may alias.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
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

        match qwen35_row_route(rows) {
            Some(Qwen35RowRoute::B1) => launch!(b1),
            Some(Qwen35RowRoute::B2) => launch!(b2),
            Some(Qwen35RowRoute::B3) => launch!(b3),
            Some(Qwen35RowRoute::B4) => launch!(b4),
            Some(Qwen35RowRoute::B5) => launch!(b5),
            Some(Qwen35RowRoute::B6) => launch!(b6),
            Some(Qwen35RowRoute::B7) => launch!(b7),
            Some(Qwen35RowRoute::B8) => launch!(b8),
            Some(Qwen35RowRoute::T32) => launch!(t32),
            Some(Qwen35RowRoute::T64) => launch!(t64),
            Some(Qwen35RowRoute::T128) => launch!(t128),
            None => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 residual RMSNorm row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128"
            ))),
        }
    }
}

/// Prepared Qwen3.6 RMSNorm routes for decode `B=1..8` and prefill `T=32,64,128`.
pub struct Qwen36ResidualNormOp {
    module: kernels::LoadedModule,
    b1: PreparedBatchRoute<Qwen36Moe35B, 1>,
    b2: PreparedBatchRoute<Qwen36Moe35B, 2>,
    b3: PreparedBatchRoute<Qwen36Moe35B, 3>,
    b4: PreparedBatchRoute<Qwen36Moe35B, 4>,
    b5: PreparedBatchRoute<Qwen36Moe35B, 5>,
    b6: PreparedBatchRoute<Qwen36Moe35B, 6>,
    b7: PreparedBatchRoute<Qwen36Moe35B, 7>,
    b8: PreparedBatchRoute<Qwen36Moe35B, 8>,
    t32: PreparedPrefillRoute<Qwen36Moe35B, 32>,
    t64: PreparedPrefillRoute<Qwen36Moe35B, 64>,
    t128: PreparedPrefillRoute<Qwen36Moe35B, 128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen36RowRoute {
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    T32,
    T64,
    T128,
}

fn qwen36_row_route(rows: usize) -> Option<Qwen36RowRoute> {
    match rows {
        1 => Some(Qwen36RowRoute::B1),
        2 => Some(Qwen36RowRoute::B2),
        3 => Some(Qwen36RowRoute::B3),
        4 => Some(Qwen36RowRoute::B4),
        5 => Some(Qwen36RowRoute::B5),
        6 => Some(Qwen36RowRoute::B6),
        7 => Some(Qwen36RowRoute::B7),
        8 => Some(Qwen36RowRoute::B8),
        32 => Some(Qwen36RowRoute::T32),
        64 => Some(Qwen36RowRoute::T64),
        128 => Some(Qwen36RowRoute::T128),
        _ => None,
    }
}

impl Qwen36ResidualNormOp {
    /// Loads the embedded SM120 module and prepares every exact Qwen3.6 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let geometry = residual_norm_geometry::<Qwen36Moe35B>().ok_or_else(|| {
            GpuError::invalid_launch("Qwen3.6 RMSNorm requires a positive even hidden width")
        })?;
        // The exact 2,048-wide row has 1,024 packed pairs: 512 threads retain
        // the qualified 16-warp reduction and consume exactly two pairs/thread.
        // This changes only the independent row width; each lane's two-pair
        // accumulation and the fixed warp/block reduction order stay explicit.
        if geometry.pairs_per_thread * THREADS as usize != geometry.pairs_per_row {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 RMSNorm requires an exact packed-pair/thread mapping",
            ));
        }
        let _ = qwen36_residual_norm_ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the residual-norm module", source))?;

        // T=128 is 128 independent 2,048-wide rows. One 128-CTA launch
        // removes 15 boundaries versus sixteen B=8 launches while each CTA
        // retains the same 512-thread pair traversal and reduction order.
        Ok(Self {
            b1: PreparedBatchRoute::prepare(&module)?,
            b2: PreparedBatchRoute::prepare(&module)?,
            b3: PreparedBatchRoute::prepare(&module)?,
            b4: PreparedBatchRoute::prepare(&module)?,
            b5: PreparedBatchRoute::prepare(&module)?,
            b6: PreparedBatchRoute::prepare(&module)?,
            b7: PreparedBatchRoute::prepare(&module)?,
            b8: PreparedBatchRoute::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Launches plain zero-centered RMSNorm for one admitted row count.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned and cover complete 2,048-value rows.
    /// Allocations must belong to `stream`'s context, remain live through stream
    /// completion, and input and output must not overlap.
    pub unsafe fn launch_plain(
        &self,
        stream: &CudaStream,
        rows: usize,
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

        match qwen36_row_route(rows) {
            Some(Qwen36RowRoute::B1) => launch!(b1),
            Some(Qwen36RowRoute::B2) => launch!(b2),
            Some(Qwen36RowRoute::B3) => launch!(b3),
            Some(Qwen36RowRoute::B4) => launch!(b4),
            Some(Qwen36RowRoute::B5) => launch!(b5),
            Some(Qwen36RowRoute::B6) => launch!(b6),
            Some(Qwen36RowRoute::B7) => launch!(b7),
            Some(Qwen36RowRoute::B8) => launch!(b8),
            Some(Qwen36RowRoute::T32) => launch!(t32),
            Some(Qwen36RowRoute::T64) => launch!(t64),
            Some(Qwen36RowRoute::T128) => launch!(t128),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 RMSNorm row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128"
            ))),
        }
    }

    /// Publishes BF16 residual sums and normalizes the represented rows.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned. Row planes must cover
    /// `rows * 2,048` BF16 values and `weight` must cover 2,048 values.
    /// Allocations must belong to `stream`'s context, remain live through
    /// completion, and not overlap except that the two input planes may alias.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
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

        match qwen36_row_route(rows) {
            Some(Qwen36RowRoute::B1) => launch!(b1),
            Some(Qwen36RowRoute::B2) => launch!(b2),
            Some(Qwen36RowRoute::B3) => launch!(b3),
            Some(Qwen36RowRoute::B4) => launch!(b4),
            Some(Qwen36RowRoute::B5) => launch!(b5),
            Some(Qwen36RowRoute::B6) => launch!(b6),
            Some(Qwen36RowRoute::B7) => launch!(b7),
            Some(Qwen36RowRoute::B8) => launch!(b8),
            Some(Qwen36RowRoute::T32) => launch!(t32),
            Some(Qwen36RowRoute::T64) => launch!(t64),
            Some(Qwen36RowRoute::T128) => launch!(t128),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 residual RMSNorm row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T=32,64,128"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, Qwen35RowRoute, Qwen36RowRoute, THREADS, WARPS, qwen35_residual_norm_ptx_names,
        qwen35_row_route, qwen36_residual_norm_ptx_names, qwen36_row_route, residual_norm_geometry,
        residual_norm_ptx_names,
    };
    use crate::test_arch::TestArch;
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

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
        let qwen35 = residual_norm_geometry::<Qwen35_9B>().unwrap();
        let qwen36 = residual_norm_geometry::<Qwen36Moe35B>().unwrap();
        let test = residual_norm_geometry::<TestArch>().unwrap();

        assert_eq!(qwen.pairs_per_row, 2_560);
        assert_eq!(qwen.pairs_per_thread, 5);
        assert_eq!(qwen35.pairs_per_row, 2_048);
        assert_eq!(qwen35.pairs_per_thread, 4);
        assert_eq!(qwen36.pairs_per_row, 1_024);
        assert_eq!(qwen36.pairs_per_thread, 2);
        assert_eq!(test.pairs_per_row, 512);
        assert_eq!(test.pairs_per_thread, 1);
    }

    #[test]
    fn ptx_inventory_has_decode_and_prefill_entries() {
        let names = residual_norm_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 2 * MAX_BATCH + 8);
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn qwen35_ptx_inventory_has_two_distinct_entries_per_route() {
        let qwen38 = residual_norm_ptx_names();
        let qwen35 = qwen35_residual_norm_ptx_names();
        let unique = qwen38
            .iter()
            .chain(&qwen35)
            .copied()
            .collect::<BTreeSet<_>>();

        assert_eq!(qwen35.len(), 2 * (MAX_BATCH + 3));
        assert_eq!(unique.len(), qwen38.len() + qwen35.len());
    }

    #[test]
    fn qwen36_ptx_inventory_has_two_distinct_entries_per_route() {
        let qwen38 = residual_norm_ptx_names();
        let qwen35 = qwen35_residual_norm_ptx_names();
        let qwen36 = qwen36_residual_norm_ptx_names();
        let unique = qwen38
            .iter()
            .chain(&qwen35)
            .chain(&qwen36)
            .copied()
            .collect::<BTreeSet<_>>();

        assert_eq!(qwen36.len(), 2 * (MAX_BATCH + 3));
        assert_eq!(unique.len(), qwen38.len() + qwen35.len() + qwen36.len());
    }

    #[test]
    fn qwen36_row_routing_is_exact() {
        assert_eq!(qwen36_row_route(0), None);
        assert_eq!(
            [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128].map(qwen36_row_route),
            [
                Some(Qwen36RowRoute::B1),
                Some(Qwen36RowRoute::B2),
                Some(Qwen36RowRoute::B3),
                Some(Qwen36RowRoute::B4),
                Some(Qwen36RowRoute::B5),
                Some(Qwen36RowRoute::B6),
                Some(Qwen36RowRoute::B7),
                Some(Qwen36RowRoute::B8),
                Some(Qwen36RowRoute::T32),
                Some(Qwen36RowRoute::T64),
                Some(Qwen36RowRoute::T128),
            ]
        );
        for rows in [9, 31, 33, 63, 65, 127, 129, usize::MAX] {
            assert_eq!(qwen36_row_route(rows), None);
        }
    }

    #[test]
    fn qwen35_row_routing_is_exact() {
        assert_eq!(qwen35_row_route(0), None);
        assert_eq!(
            [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128].map(qwen35_row_route),
            [
                Some(Qwen35RowRoute::B1),
                Some(Qwen35RowRoute::B2),
                Some(Qwen35RowRoute::B3),
                Some(Qwen35RowRoute::B4),
                Some(Qwen35RowRoute::B5),
                Some(Qwen35RowRoute::B6),
                Some(Qwen35RowRoute::B7),
                Some(Qwen35RowRoute::B8),
                Some(Qwen35RowRoute::T32),
                Some(Qwen35RowRoute::T64),
                Some(Qwen35RowRoute::T128),
            ]
        );
        for rows in [9, 31, 33, 63, 65, 127, 129, usize::MAX] {
            assert_eq!(qwen35_row_route(rows), None);
        }
    }
}

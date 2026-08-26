//! Exact-batch zero-centered RMSNorm operators.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
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

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared plain and fused entries for an exact row count.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entries the module does not emit.
pub trait ResidualNormRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares both entries of this route's exact row count.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's plain RMSNorm entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `ResidualNormOp::launch_plain`'s contract unchanged.
    unsafe fn launch_plain(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;

    /// Launches this route's fused residual RMSNorm entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `ResidualNormOp::launch_residual`'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's RMSNorm routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 and Qwen3.6 here never widens the
/// artifact-level admission bound. Each table names only the entries its own
/// model emits, which is what keeps the compiled inventory fixed while the
/// three prepared owners share one wrapper.
pub trait ResidualNormEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for `B=1`.
    type DecodeOne: ResidualNormRoute<A>;
    /// Prepared decode route for `B=2..=8`.
    type Decode<const TOKENS: usize>: ResidualNormRoute<A>;
    /// Prepared prefill route for `T=1024`, unadmitted outside Qwen3.8.
    type Prefill1024: ResidualNormRoute<A>;

    /// Whether `T=1024` is an admitted prefill row count.
    const HAS_T1024: bool;
    /// Message prefix that keeps this architecture's launch errors distinct.
    const LABEL: &'static str;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared generic decode entries for one exact batch.
pub struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__rms_norm_CudaKernel<A, TOKENS>>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.8 `B=1` decode entries.
///
/// `B=1` keeps the concrete plain entry that anchors the embedded module
/// artifact.
pub struct PreparedBatchOneRoute {
    plain: PreparedLaunch<kernels::__rms_norm_b1_CudaKernel>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_CudaKernel<Qwen38_27B, 1>>,
}

/// Prepared Qwen3.5 decode entries for one exact batch.
pub struct PreparedQwen35BatchRoute<const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__qwen35_rms_norm_CudaKernel<TOKENS>>,
    residual: PreparedLaunch<kernels::__qwen35_residual_rms_norm_CudaKernel<TOKENS>>,
}

/// Prepared prefill entries for one exact row count.
///
/// Prefill retains separate symbols so its resource authority cannot drift
/// with decode.
pub struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    plain: PreparedLaunch<kernels::__rms_norm_prefill_CudaKernel<A, TOKENS>>,
    residual: PreparedLaunch<kernels::__residual_rms_norm_prefill_CudaKernel<A, TOKENS>>,
}

/// Stands in for a row count an architecture does not admit.
///
/// It prepares and launches no entry, so an unadmitted width can never reach
/// the device and never enters the emitted inventory.
pub struct UnadmittedRoute;

impl private::Sealed for PreparedBatchOneRoute {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedBatchRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35BatchRoute<TOKENS> {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedPrefillRoute<A, TOKENS> {}
impl private::Sealed for UnadmittedRoute {}

// The B=1 anchor compiles the exact Qwen3.8 row width into a concrete entry,
// so it stays bound to the sealed artifact-level architecture.
impl<A: Sm120Arch> ResidualNormRoute<A> for PreparedBatchOneRoute {
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

impl<A: Arch, const TOKENS: usize> ResidualNormRoute<A> for PreparedBatchRoute<A, TOKENS> {
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

impl<const TOKENS: usize> ResidualNormRoute<Qwen35_9B> for PreparedQwen35BatchRoute<TOKENS> {
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

impl<A: Arch, const TOKENS: usize> ResidualNormRoute<A> for PreparedPrefillRoute<A, TOKENS> {
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

impl<A: Arch> ResidualNormRoute<A> for UnadmittedRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    unsafe fn launch_plain(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _input: *const u16,
        _weight: *const u16,
        _output: *mut u16,
    ) -> GpuResult<()> {
        Err(unadmitted_route())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_residual(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _residual_input: *const u16,
        _branch: *const u16,
        _weight: *const u16,
        _residual_output: *mut u16,
        _normalized_output: *mut u16,
    ) -> GpuResult<()> {
        Err(unadmitted_route())
    }
}

// `row_route` rejects an unadmitted width before dispatch, so this is the
// defensive tail of a route that owns no entry.
fn unadmitted_route() -> GpuError {
    GpuError::invalid_launch("RMSNorm route is not admitted for this architecture")
}

/// Qwen3.8 entry table: the concrete `B=1` artifact anchor, the generic
/// decode entries at `B=2..=8`, and the admitted `T=1024` prefill route.
pub struct Qwen38ResidualNormEntries;

/// Qwen3.5 entry table: its own decode entry family and prefill through `T=128`.
pub struct Qwen35ResidualNormEntries;

/// Qwen3.6 entry table: the generic decode entries and prefill through `T=128`.
pub struct Qwen36ResidualNormEntries;

impl private::Sealed for Qwen38ResidualNormEntries {}
impl private::Sealed for Qwen35ResidualNormEntries {}
impl private::Sealed for Qwen36ResidualNormEntries {}

impl<A: Sm120Arch> ResidualNormEntries<A> for Qwen38ResidualNormEntries {
    type DecodeOne = PreparedBatchOneRoute;
    type Decode<const TOKENS: usize> = PreparedBatchRoute<A, TOKENS>;
    type Prefill1024 = PreparedPrefillRoute<A, 1024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "";

    fn ptx_names() -> Vec<&'static str> {
        residual_norm_ptx_names()
    }
}

impl ResidualNormEntries<Qwen35_9B> for Qwen35ResidualNormEntries {
    type DecodeOne = PreparedQwen35BatchRoute<1>;
    type Decode<const TOKENS: usize> = PreparedQwen35BatchRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.5 ";

    fn ptx_names() -> Vec<&'static str> {
        qwen35_residual_norm_ptx_names().to_vec()
    }
}

impl ResidualNormEntries<Qwen36Moe35B> for Qwen36ResidualNormEntries {
    type DecodeOne = PreparedBatchRoute<Qwen36Moe35B, 1>;
    type Decode<const TOKENS: usize> = PreparedBatchRoute<Qwen36Moe35B, TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.6 ";

    fn ptx_names() -> Vec<&'static str> {
        qwen36_residual_norm_ptx_names().to_vec()
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

/// The compiled route one admitted row count selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowRoute {
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
    T1024,
}

// The admitted row schedule, transcribed from the three prepared dispatches
// it replaces: decode B=1..=8 and prefill T=32,64,128 everywhere, and the
// T=1024 prefill only where the entry table admits it.
fn row_route<A: Arch, E: ResidualNormEntries<A>>(rows: usize) -> Option<RowRoute> {
    match rows {
        1 => Some(RowRoute::B1),
        2 => Some(RowRoute::B2),
        3 => Some(RowRoute::B3),
        4 => Some(RowRoute::B4),
        5 => Some(RowRoute::B5),
        6 => Some(RowRoute::B6),
        7 => Some(RowRoute::B7),
        8 => Some(RowRoute::B8),
        32 => Some(RowRoute::T32),
        64 => Some(RowRoute::T64),
        128 => Some(RowRoute::T128),
        1024 if E::HAS_T1024 => Some(RowRoute::T1024),
        _ => None,
    }
}

fn admitted_prefill_rows<A: Arch, E: ResidualNormEntries<A>>() -> &'static str {
    if E::HAS_T1024 {
        "32,64,128,1024"
    } else {
        "32,64,128"
    }
}

fn unsupported_rows<A: Arch, E: ResidualNormEntries<A>>(operation: &str, rows: usize) -> GpuError {
    GpuError::invalid_launch(format!(
        "{}{operation} row count {rows} is outside exact decode 1..={MAX_BATCH} and prefill T={}",
        E::LABEL,
        admitted_prefill_rows::<A, E>(),
    ))
}

/// Prepared RMSNorm routes for decode `B=1..=8` and the entry table's
/// admitted prefill widths.
pub struct ResidualNormOp<
    A: Arch = Qwen38_27B,
    E: ResidualNormEntries<A> = Qwen38ResidualNormEntries,
> {
    module: kernels::LoadedModule,
    b1: E::DecodeOne,
    b2: E::Decode<2>,
    b3: E::Decode<3>,
    b4: E::Decode<4>,
    b5: E::Decode<5>,
    b6: E::Decode<6>,
    b7: E::Decode<7>,
    b8: E::Decode<8>,
    t32: PreparedPrefillRoute<A, 32>,
    t64: PreparedPrefillRoute<A, 64>,
    t128: PreparedPrefillRoute<A, 128>,
    t1024: E::Prefill1024,
}

/// Prepared Qwen3.5 RMSNorm routes for decode `B=1..8` and prefill `T=32,64,128`.
pub type Qwen35ResidualNormOp = ResidualNormOp<Qwen35_9B, Qwen35ResidualNormEntries>;

/// Prepared Qwen3.6 RMSNorm routes for decode `B=1..8` and prefill `T=32,64,128`.
pub type Qwen36ResidualNormOp = ResidualNormOp<Qwen36Moe35B, Qwen36ResidualNormEntries>;

impl<A: Arch, E: ResidualNormEntries<A>> ResidualNormOp<A, E> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let geometry = residual_norm_geometry::<A>().ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "{}RMSNorm requires a positive even hidden width",
                E::LABEL
            ))
        })?;
        // Every admitted width divides evenly across the 512-thread CTA:
        // Qwen3.5/3.6/3.8 consume exactly four/two/five packed pairs per
        // thread, which is what keeps the qualified 16-warp reduction order.
        if geometry.pairs_per_thread * THREADS as usize != geometry.pairs_per_row {
            return Err(GpuError::invalid_launch(format!(
                "{}RMSNorm requires an exact packed-pair/thread mapping",
                E::LABEL
            )));
        }
        let _ = E::ptx_names();
        // SAFETY: this crate owns one cuda-oxide module and its embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the residual-norm module", source))?;

        // T=128 is 128 independent rows. One 128-CTA launch removes 15
        // boundaries versus sixteen B=8 launches while each CTA retains the
        // same 512-thread traversal and reduction order.
        Ok(Self {
            b1: E::DecodeOne::prepare(&module)?,
            b2: E::Decode::<2>::prepare(&module)?,
            b3: E::Decode::<3>::prepare(&module)?,
            b4: E::Decode::<4>::prepare(&module)?,
            b5: E::Decode::<5>::prepare(&module)?,
            b6: E::Decode::<6>::prepare(&module)?,
            b7: E::Decode::<7>::prepare(&module)?,
            b8: E::Decode::<8>::prepare(&module)?,
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            t1024: E::Prefill1024::prepare(&module)?,
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

        match row_route::<A, E>(rows) {
            Some(RowRoute::B1) => launch!(b1),
            Some(RowRoute::B2) => launch!(b2),
            Some(RowRoute::B3) => launch!(b3),
            Some(RowRoute::B4) => launch!(b4),
            Some(RowRoute::B5) => launch!(b5),
            Some(RowRoute::B6) => launch!(b6),
            Some(RowRoute::B7) => launch!(b7),
            Some(RowRoute::B8) => launch!(b8),
            Some(RowRoute::T32) => launch!(t32),
            Some(RowRoute::T64) => launch!(t64),
            Some(RowRoute::T128) => launch!(t128),
            Some(RowRoute::T1024) => launch!(t1024),
            None => Err(unsupported_rows::<A, E>("RMSNorm", rows)),
        }
    }

    /// Publishes BF16 residual sums and launches their next RMSNorm.
    ///
    /// # Safety
    ///
    /// Every pointer must be four-byte aligned. Row planes must cover
    /// `rows * A::HIDDEN` BF16 values and `weight` must cover `A::HIDDEN` values.
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

        match row_route::<A, E>(rows) {
            Some(RowRoute::B1) => launch!(b1),
            Some(RowRoute::B2) => launch!(b2),
            Some(RowRoute::B3) => launch!(b3),
            Some(RowRoute::B4) => launch!(b4),
            Some(RowRoute::B5) => launch!(b5),
            Some(RowRoute::B6) => launch!(b6),
            Some(RowRoute::B7) => launch!(b7),
            Some(RowRoute::B8) => launch!(b8),
            Some(RowRoute::T32) => launch!(t32),
            Some(RowRoute::T64) => launch!(t64),
            Some(RowRoute::T128) => launch!(t128),
            Some(RowRoute::T1024) => launch!(t1024),
            None => Err(unsupported_rows::<A, E>("residual RMSNorm", rows)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, Qwen35ResidualNormEntries, Qwen36ResidualNormEntries, Qwen38ResidualNormEntries,
        ResidualNormEntries, RowRoute, THREADS, WARPS, qwen35_residual_norm_ptx_names,
        qwen36_residual_norm_ptx_names, residual_norm_geometry, residual_norm_ptx_names, row_route,
        unsupported_rows,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_kernels_sm120_common::TestArch;
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

    /// The decode and prefill widths every admitted architecture routes.
    const SHARED_SCHEDULE: [(usize, RowRoute); 11] = [
        (1, RowRoute::B1),
        (2, RowRoute::B2),
        (3, RowRoute::B3),
        (4, RowRoute::B4),
        (5, RowRoute::B5),
        (6, RowRoute::B6),
        (7, RowRoute::B7),
        (8, RowRoute::B8),
        (32, RowRoute::T32),
        (64, RowRoute::T64),
        (128, RowRoute::T128),
    ];

    /// Every row count the entry table admits, swept exhaustively so an
    /// unadmitted width cannot hide between the transcribed ones.
    fn admitted_schedule<A: Arch, E: ResidualNormEntries<A>>() -> Vec<(usize, RowRoute)> {
        (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|rows| row_route::<A, E>(rows).map(|route| (rows, route)))
            .collect()
    }

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

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

    /// The shared prepare path rejects an inexact packed-pair/thread mapping;
    /// every admitted architecture satisfies it, so the merged owner keeps
    /// each model's qualified 16-warp reduction.
    #[test]
    fn every_admitted_width_maps_exactly_onto_the_cta() {
        for geometry in [
            residual_norm_geometry::<Qwen38_27B>().unwrap(),
            residual_norm_geometry::<Qwen35_9B>().unwrap(),
            residual_norm_geometry::<Qwen36Moe35B>().unwrap(),
        ] {
            assert_eq!(
                geometry.pairs_per_thread * THREADS as usize,
                geometry.pairs_per_row
            );
        }
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

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38ResidualNormEntries as ResidualNormEntries<Qwen38_27B>>::ptx_names(),
            residual_norm_ptx_names()
        );
        assert_eq!(
            <Qwen35ResidualNormEntries as ResidualNormEntries<Qwen35_9B>>::ptx_names(),
            qwen35_residual_norm_ptx_names().to_vec()
        );
        assert_eq!(
            <Qwen36ResidualNormEntries as ResidualNormEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_residual_norm_ptx_names().to_vec()
        );
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this family
    /// is its per-base-name count. These are the counts the pinned SM120
    /// device build emits; a wrapper change that instantiates one more
    /// specialization moves one of them.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in residual_norm_ptx_names()
            .into_iter()
            .chain(qwen35_residual_norm_ptx_names())
            .chain(qwen36_residual_norm_ptx_names())
        {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("qwen35_residual_rms_norm", 8),
                ("qwen35_rms_norm", 8),
                ("residual_rms_norm", 16),
                ("residual_rms_norm_prefill", 10),
                ("rms_norm", 15),
                ("rms_norm_b1", 1),
                ("rms_norm_prefill", 10),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 68);
    }

    /// The merged schedule, checked against the three dispatches it replaces:
    /// Qwen3.5 and Qwen3.6 stop at `T=128`, and only Qwen3.8 admits `T=1024`.
    #[test]
    fn row_routing_is_exact_for_every_admitted_architecture() {
        let qwen38 = SHARED_SCHEDULE
            .iter()
            .copied()
            .chain([(1_024, RowRoute::T1024)])
            .collect::<Vec<_>>();

        assert_eq!(
            admitted_schedule::<Qwen38_27B, Qwen38ResidualNormEntries>(),
            qwen38
        );
        assert_eq!(
            admitted_schedule::<Qwen35_9B, Qwen35ResidualNormEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule::<Qwen36Moe35B, Qwen36ResidualNormEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
    }

    /// An unadmitted row count keeps naming the architecture that rejected it.
    #[test]
    fn unadmitted_row_counts_name_their_architecture() {
        for (message, error) in [
            (
                "RMSNorm row count 9 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows::<Qwen38_27B, Qwen38ResidualNormEntries>("RMSNorm", 9),
            ),
            (
                "residual RMSNorm row count 9 is outside exact decode 1..=8 and prefill T=32,64,128,1024",
                unsupported_rows::<Qwen38_27B, Qwen38ResidualNormEntries>("residual RMSNorm", 9),
            ),
            (
                "Qwen3.5 RMSNorm row count 1024 is outside exact decode 1..=8 and prefill T=32,64,128",
                unsupported_rows::<Qwen35_9B, Qwen35ResidualNormEntries>("RMSNorm", 1_024),
            ),
            (
                "Qwen3.5 residual RMSNorm row count 1024 is outside exact decode 1..=8 and prefill T=32,64,128",
                unsupported_rows::<Qwen35_9B, Qwen35ResidualNormEntries>("residual RMSNorm", 1_024),
            ),
            (
                "Qwen3.6 RMSNorm row count 1024 is outside exact decode 1..=8 and prefill T=32,64,128",
                unsupported_rows::<Qwen36Moe35B, Qwen36ResidualNormEntries>("RMSNorm", 1_024),
            ),
            (
                "Qwen3.6 residual RMSNorm row count 1024 is outside exact decode 1..=8 and prefill T=32,64,128",
                unsupported_rows::<Qwen36Moe35B, Qwen36ResidualNormEntries>(
                    "residual RMSNorm",
                    1_024,
                ),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }
}

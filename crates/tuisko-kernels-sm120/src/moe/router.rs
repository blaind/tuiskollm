//! Exact Qwen3.6 BF16 router and normalized top-8 selection.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
// One warp owns one 2,048-wide dot product, so every lane reads 32 adjacent
// BF16 pairs. Eight warps produce eight experts per CTA. Splitting the same
// warps into 256 one-warp CTAs slowed B=1/B=8 by 11%/10%; 32 CTAs/token wins
// despite incomplete SM coverage and preserves each expert's reduction order.
const PROJECTION_WARPS: usize = 8;
const PROJECTION_THREADS: u32 = (PROJECTION_WARPS * 32) as u32;
// The scalar spill-free selector made the complete router 316 us at B=8.
// One warp cuts each rank scan from 256 to eight candidates/lane; the same
// value-then-expert ordering is reduced deterministically, and lane zero keeps
// the original eight-term normalization order.
const SELECT_THREADS: u32 = 32;
const EXPERT_BLOCKS: usize = EXPERTS / PROJECTION_WARPS;
const WORDS_PER_ROW: usize = HIDDEN / 2;

const _: () = assert!(HIDDEN == 2_048);
const _: () = assert!(EXPERTS == 256);
const _: () = assert!(TOP_K == 8);
const _: () = assert!(HIDDEN.is_multiple_of(64));
const _: () = assert!(EXPERTS.is_multiple_of(PROJECTION_WARPS));

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, thread, warp};

    #[inline(always)]
    fn reduce_sum_lane_zero(mut value: f32) -> f32 {
        value += warp::shuffle_down_f32(value, 16);
        value += warp::shuffle_down_f32(value, 8);
        value += warp::shuffle_down_f32(value, 4);
        value += warp::shuffle_down_f32(value, 2);
        value += warp::shuffle_down_f32(value, 1);

        value
    }

    /// Projects represented BF16 rows through the exact 256-row router.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_router_logits<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        logits: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let block = thread::blockIdx_x() as usize;
        let token = block / EXPERT_BLOCKS;
        let expert = (block % EXPERT_BLOCKS) * PROJECTION_WARPS + warp_index;
        let input_row = unsafe { input.add(token * WORDS_PER_ROW) };
        let weight_row = unsafe { weights.add(expert * WORDS_PER_ROW) };
        let mut pair = lane;
        let mut sum = 0.0f32;

        while pair < WORDS_PER_ROW {
            let (input_low, input_high) =
                convert::cvt_f32x2_bf16x2(unsafe { *input_row.add(pair) });
            let (weight_low, weight_high) =
                convert::cvt_f32x2_bf16x2(unsafe { *weight_row.add(pair) });
            sum = float::fma_rn_f32(input_low, weight_low, sum);
            sum = float::fma_rn_f32(input_high, weight_high, sum);
            pair += 32;
        }

        let sum = reduce_sum_lane_zero(sum);
        if lane == 0 {
            unsafe {
                *logits.add(token * EXPERTS + expert) = tcgen05::f32_to_bf16_rne(sum);
            }
        }
    }

    /// Selects and normalizes the exact top-eight router experts.
    #[kernel]
    #[launch_bounds(32, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_router_select<const TOKENS: usize>(
        logits: *const u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) {
        let lane = thread::threadIdx_x();
        let token = thread::blockIdx_x() as usize;
        let indices = unsafe { expert_indices.add(token * TOP_K) };
        let weights = unsafe { expert_weights.add(token * TOP_K) };
        let logits = unsafe { logits.add(token * EXPERTS) };
        let mut position = 0usize;

        while position < TOP_K {
            let mut best_value = f32::NEG_INFINITY;
            let mut best_index = u16::MAX;
            let mut expert = lane as usize;

            while expert < EXPERTS {
                let mut prior = 0usize;
                let mut selected = false;
                while prior < position {
                    if unsafe { *indices.add(prior) } as usize == expert {
                        selected = true;
                    }
                    prior += 1;
                }
                if !selected {
                    let bits = unsafe { *logits.add(expert) };
                    let value = f32::from_bits(u32::from(bits) << 16);
                    let better =
                        value > best_value || (value == best_value && expert < best_index as usize);
                    if better {
                        best_value = value;
                        best_index = expert as u16;
                    }
                }
                expert += 32;
            }

            let mut delta = 16u32;
            while delta != 0 {
                let other_value = warp::shuffle_down_f32(best_value, delta);
                let other_index = warp::shuffle_down(u32::from(best_index), delta) as u16;
                if lane + delta < 32 {
                    let better = other_value > best_value
                        || (other_value == best_value && other_index < best_index);
                    if better {
                        best_value = other_value;
                        best_index = other_index;
                    }
                }
                delta >>= 1;
            }
            if lane == 0 {
                unsafe { *indices.add(position) = best_index };
            }
            warp::sync_mask(u32::MAX);
            position += 1;
        }

        if lane != 0 {
            return;
        }

        // The common 256-way softmax denominator cancels when the selected
        // probabilities are normalized again, leaving an exact top-8 softmax.
        let maximum_index = unsafe { *indices } as usize;
        let maximum = f32::from_bits(u32::from(unsafe { *logits.add(maximum_index) }) << 16);
        let mut denominator = 0.0f32;
        position = 0;

        while position < TOP_K {
            let expert = unsafe { *indices.add(position) } as usize;
            let value = f32::from_bits(u32::from(unsafe { *logits.add(expert) }) << 16);
            denominator += float::ex2_approx_f32((value - maximum) * core::f32::consts::LOG2_E);
            position += 1;
        }

        position = 0;
        while position < TOP_K {
            let expert = unsafe { *indices.add(position) } as usize;
            let value = f32::from_bits(u32::from(unsafe { *logits.add(expert) }) << 16);
            let exponential = float::ex2_approx_f32((value - maximum) * core::f32::consts::LOG2_E);
            unsafe { *weights.add(position) = tcgen05::f32_to_bf16_rne(exponential / denominator) };
            position += 1;
        }
    }
}

fn projection_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new((TOKENS * EXPERT_BLOCKS) as u32, PROJECTION_THREADS, 0)
}

fn select_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(TOKENS as u32, SELECT_THREADS, 0)
}

struct PreparedBatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_moe_router_logits_CudaKernel<TOKENS>>,
    select: PreparedLaunch<kernels::__qwen36_moe_router_select_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            projection: module
                .prepare_qwen36_moe_router_logits::<TOKENS>(projection_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE router logits", source)
                })?,
            select: module
                .prepare_qwen36_moe_router_select::<TOKENS>(select_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE top-8 selection", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weights: *const u16,
        logits: *mut u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_moe_router_logits::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                logits,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE router logits", source))?;
        module
            .qwen36_moe_router_select::<TOKENS>(
                stream,
                &self.select,
                logits,
                expert_indices,
                expert_weights,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MoE top-8 selection", source))
    }
}

/// PTX symbols retained for every exact Qwen3.6 router batch.
pub(crate) fn qwen36_moe_router_ptx_names() -> Vec<&'static str> {
    let mut names = Vec::with_capacity(2 * MAX_BATCH);

    macro_rules! push_route {
        ($tokens:literal) => {
            names.push(kernels::qwen36_moe_router_logits_ptx_name::<$tokens>());
            names.push(kernels::qwen36_moe_router_select_ptx_name::<$tokens>());
        };
    }

    push_route!(1);
    push_route!(2);
    push_route!(3);
    push_route!(4);
    push_route!(5);
    push_route!(6);
    push_route!(7);
    push_route!(8);
    names
}

/// Prepared exact-batch Qwen3.6 BF16 router routes on SM120.
pub struct Qwen36MoeRouterOp {
    module: kernels::LoadedModule,
    b1: PreparedBatchRoute<1>,
    b2: PreparedBatchRoute<2>,
    b3: PreparedBatchRoute<3>,
    b4: PreparedBatchRoute<4>,
    b5: PreparedBatchRoute<5>,
    b6: PreparedBatchRoute<6>,
    b7: PreparedBatchRoute<7>,
    b8: PreparedBatchRoute<8>,
}

impl Qwen36MoeRouterOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_moe_router_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.6 MoE router", source))?;

        Ok(Self {
            b1: PreparedBatchRoute::prepare(&module)?,
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

    /// Projects BF16 logits, selects top eight experts, and emits BF16 routing weights.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 2_048` BF16 values, `weights` covers BF16
    /// `[256, 2_048]`, `logits` covers `batch * 256` BF16 values, and both
    /// top-8 outputs cover `batch * 8` values. Four-byte-loaded input and
    /// weight planes are aligned, disjoint, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weights: *const u16,
        logits: *mut u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        weights,
                        logits,
                        expert_indices,
                        expert_weights,
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
                "Qwen3.6 MoE router batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EXPERT_BLOCKS, MAX_BATCH, WORDS_PER_ROW, qwen36_moe_router_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(EXPERT_BLOCKS, 32);
        assert_eq!(WORDS_PER_ROW, 1_024);

        let names = qwen36_moe_router_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

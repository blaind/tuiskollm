//! Exact Qwen3.6 BF16 router and normalized top-8 selection.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 3] = [32, 64, 128];
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

    #[inline(always)]
    unsafe fn router_logits<const TOKENS: usize>(
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

    /// Projects represented BF16 decode rows through the exact 256-row router.
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
        unsafe { router_logits::<TOKENS>(input, weights, logits) }
    }

    /// Projects represented BF16 prompt rows through the exact 256-row router.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_router_logits_prefill<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        logits: *mut u16,
    ) {
        unsafe { router_logits::<TOKENS>(input, weights, logits) }
    }

    #[inline(always)]
    unsafe fn router_select<const TOKENS: usize>(
        logits: *const u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) {
        let lane = thread::threadIdx_x();
        let token = thread::blockIdx_x() as usize;
        let indices = unsafe { expert_indices.add(token * TOP_K) };
        let weights = unsafe { expert_weights.add(token * TOP_K) };
        let logits = unsafe { logits.add(token * EXPERTS) };

        // Each lane loads its eight owned logits once and the eight selection
        // passes run entirely in registers: consumption is a register mask on
        // the owner lane and the butterfly hands every lane the pass winner.
        // The per-lane maximum, the cross-lane comparison, and the
        // lowest-index tie-break are the exact predicates of the original
        // global-memory walk, so the selected experts, their order, and every
        // published weight are bit-identical.
        const OWNED: usize = EXPERTS / 32;
        let mut owned_values = [0.0f32; OWNED];
        let mut owned_taken = 0u32;
        let mut slot = 0usize;
        while slot < OWNED {
            let expert = slot * 32 + lane as usize;
            let bits = unsafe { *logits.add(expert) };
            owned_values[slot] = f32::from_bits(u32::from(bits) << 16);
            slot += 1;
        }

        let mut winner_values = [0.0f32; TOP_K];
        let mut position = 0usize;
        while position < TOP_K {
            let mut best_value = f32::NEG_INFINITY;
            let mut best_index = u16::MAX;
            let mut slot = 0usize;
            while slot < OWNED {
                if owned_taken & (1u32 << slot) == 0 {
                    let value = owned_values[slot];
                    let index = (slot * 32 + lane as usize) as u16;
                    let better = value > best_value || (value == best_value && index < best_index);
                    if better {
                        best_value = value;
                        best_index = index;
                    }
                }
                slot += 1;
            }

            let mut delta = 16u32;
            while delta != 0 {
                let other_value = warp::shuffle_xor_f32_sync(u32::MAX, best_value, delta);
                let other_index =
                    warp::shuffle_xor_sync(u32::MAX, u32::from(best_index), delta) as u16;
                let better = other_value > best_value
                    || (other_value == best_value && other_index < best_index);
                if better {
                    best_value = other_value;
                    best_index = other_index;
                }
                delta >>= 1;
            }
            if lane == 0 {
                unsafe { *indices.add(position) = best_index };
            }
            winner_values[position] = best_value;
            if (best_index as usize & 31) == lane as usize {
                owned_taken |= 1u32 << ((best_index as usize) >> 5);
            }
            position += 1;
        }

        if lane != 0 {
            return;
        }

        // The common 256-way softmax denominator cancels when the selected
        // probabilities are normalized again, leaving an exact top-8 softmax.
        // winner_values holds the identical logit values the original re-read
        // from global memory, in the identical selection order.
        let maximum = winner_values[0];
        let mut denominator = 0.0f32;
        position = 0;

        while position < TOP_K {
            let value = winner_values[position];
            denominator += float::ex2_approx_f32((value - maximum) * core::f32::consts::LOG2_E);
            position += 1;
        }

        position = 0;
        while position < TOP_K {
            let value = winner_values[position];
            let exponential = float::ex2_approx_f32((value - maximum) * core::f32::consts::LOG2_E);
            unsafe { *weights.add(position) = tcgen05::f32_to_bf16_rne(exponential / denominator) };
            position += 1;
        }
    }

    /// Selects and normalizes the exact top-eight decode experts.
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
        unsafe { router_select::<TOKENS>(logits, expert_indices, expert_weights) }
    }

    /// Selects and normalizes the exact top-eight prompt experts.
    #[kernel]
    #[launch_bounds(32, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_moe_router_select_prefill<const TOKENS: usize>(
        logits: *const u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) {
        unsafe { router_select::<TOKENS>(logits, expert_indices, expert_weights) }
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
        if !(1..=MAX_BATCH).contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE router decode row count {TOKENS} is not admitted"
            )));
        }
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

    fn ptx_names() -> Vec<&'static str> {
        vec![
            kernels::qwen36_moe_router_logits_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_router_select_ptx_name::<TOKENS>(),
        ]
    }

    #[allow(clippy::too_many_arguments)]
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

struct PreparedPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_moe_router_logits_prefill_CudaKernel<TOKENS>>,
    select: PreparedLaunch<kernels::__qwen36_moe_router_select_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE router prefill row count {TOKENS} is not admitted"
            )));
        }
        // T=128 exposes 4,096 projection CTAs and 128 selector CTAs. The
        // existing eight-warps-per-eight-experts topology already fills the
        // device; prompt specialization changes only the independent token
        // count, while every expert dot and top-eight comparison keeps its
        // exact decode order.
        Ok(Self {
            projection: module
                .prepare_qwen36_moe_router_logits_prefill::<TOKENS>(projection_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE prompt router logits", source)
                })?,
            select: module
                .prepare_qwen36_moe_router_select_prefill::<TOKENS>(select_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MoE prompt top-8 selection", source)
                })?,
        })
    }

    fn ptx_names() -> Vec<&'static str> {
        vec![
            kernels::qwen36_moe_router_logits_prefill_ptx_name::<TOKENS>(),
            kernels::qwen36_moe_router_select_prefill_ptx_name::<TOKENS>(),
        ]
    }

    #[allow(clippy::too_many_arguments)]
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
            .qwen36_moe_router_logits_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                logits,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 MoE prompt router logits", source)
            })?;
        module
            .qwen36_moe_router_select_prefill::<TOKENS>(
                stream,
                &self.select,
                logits,
                expert_indices,
                expert_weights,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 MoE prompt top-8 selection", source)
            })
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_moe_router),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128)
)]
struct MoeRouterRoutes {
    #[route(1)]
    b1: PreparedBatchRoute<1>,
    #[route(2)]
    b2: PreparedBatchRoute<2>,
    #[route(3)]
    b3: PreparedBatchRoute<3>,
    #[route(4)]
    b4: PreparedBatchRoute<4>,
    #[route(5)]
    b5: PreparedBatchRoute<5>,
    #[route(6)]
    b6: PreparedBatchRoute<6>,
    #[route(7)]
    b7: PreparedBatchRoute<7>,
    #[route(8)]
    b8: PreparedBatchRoute<8>,
    #[route(32)]
    t32: PreparedPrefillRoute<32>,
    #[route(64)]
    t64: PreparedPrefillRoute<64>,
    #[route(128)]
    t128: PreparedPrefillRoute<128>,
}

/// PTX symbols retained for every exact Qwen3.6 router batch.
pub(crate) fn qwen36_moe_router_ptx_names() -> Vec<&'static str> {
    MoeRouterRoutes::ptx_names()
}

/// Prepared exact-batch Qwen3.6 BF16 router routes on SM120.
pub struct Qwen36MoeRouterOp {
    module: kernels::LoadedModule,
    routes: MoeRouterRoutes,
}

impl Qwen36MoeRouterOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_moe_router_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.6 MoE router", source))?;

        let routes = MoeRouterRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Projects BF16 logits, selects top eight experts, and emits BF16 routing weights.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * 2_048` BF16 values, `weights` covers BF16
    /// `[256, 2_048]`, `logits` covers `rows * 256` BF16 values, and both
    /// top-8 outputs cover `rows * 8` values. Four-byte-loaded input and
    /// weight planes are aligned, disjoint, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weights: *const u16,
        logits: *mut u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) -> GpuResult<()> {
        dispatch_moe_router!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    weights,
                    logits,
                    expert_indices,
                    expert_weights,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MoE router row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128"
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EXPERT_BLOCKS, MAX_BATCH, MoeRouterRoutes, PREFILL_ROWS, WORDS_PER_ROW,
        qwen36_moe_router_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(EXPERT_BLOCKS, 32);
        assert_eq!(WORDS_PER_ROW, 1_024);

        let names = qwen36_moe_router_ptx_names();
        assert_eq!(names.len(), 2 * (MAX_BATCH + PREFILL_ROWS.len()));
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }

    #[test]
    fn row_table_covers_only_exact_decode_and_prefill_routes() {
        for (rows, expected) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (16, false),
            (32, true),
            (33, false),
            (64, true),
            (65, false),
            (128, true),
            (129, false),
        ] {
            assert_eq!(MoeRouterRoutes::contains(rows), expected, "rows={rows}");
        }
    }
}

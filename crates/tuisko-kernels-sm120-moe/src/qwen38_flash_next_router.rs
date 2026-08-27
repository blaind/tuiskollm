//! Exact Qwen3.8-Flash-Next BF16 router and normalized top-10 selection.
//!
//! The exact route performs a full 512-way FP32 softmax, selects the top ten,
//! then renormalizes those probabilities. It does not use Qwen3.6's shortcut
//! of selecting logits before softmax; denominator cancellation is inexact in
//! FP32 and would change represented values.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
const HIDDEN: usize = Qwen38FlashNext::HIDDEN;
const EXPERTS: usize = Qwen38FlashNext::NUM_EXPERTS;
const TOP_K: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
// One warp owns one 2,560-wide dot product, so every lane reads 40 adjacent
// BF16 pairs. Eight warps produce eight experts per CTA, which is the Qwen3.6
// router topology at twice the expert count and 1.25x the row width; each
// expert's reduction order is preserved inside its own warp.
const PROJECTION_WARPS: usize = 8;
const PROJECTION_THREADS: u32 = (PROJECTION_WARPS * 32) as u32;
// One warp per token holds all 512 logits in registers: sixteen per lane. The
// softmax maximum, the 512-wide denominator, and the ten selection passes all
// run over that register file, so the logits plane is read exactly once.
const SELECT_THREADS: u32 = 32;
const OWNED: usize = EXPERTS / SELECT_THREADS as usize;
const EXPERT_BLOCKS: usize = EXPERTS / PROJECTION_WARPS;
const WORDS_PER_ROW: usize = HIDDEN / 2;

const _: () = assert!(HIDDEN == 2_560);
const _: () = assert!(EXPERTS == 512);
const _: () = assert!(TOP_K == 10);
const _: () = assert!(OWNED == 16);
const _: () = assert!(HIDDEN.is_multiple_of(64));
const _: () = assert!(EXPERTS.is_multiple_of(PROJECTION_WARPS));
const _: () = assert!(EXPERTS.is_multiple_of(SELECT_THREADS as usize));

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

    /// Projects represented BF16 decode rows through the exact 512-row router.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_router_logits<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        logits: *mut u16,
    ) {
        unsafe { router_logits::<TOKENS>(input, weights, logits) }
    }

    /// Projects represented BF16 prompt rows through the exact 512-row router.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_router_logits_prefill<const TOKENS: usize>(
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

        // Every walk over the sixteen owned experts and the ten ranks is
        // expanded to constant indices. At this width `while slot < OWNED` is
        // past the unroller's threshold, and a dynamically indexed register
        // array becomes a 128-byte local depot -- which the zero-stack/local
        // gate forbids. The Qwen3.6 selector stays inside the threshold at
        // eight and eight, so it needs none of this.
        macro_rules! each_slot {
            ($action:ident) => {
                $action!(0);
                $action!(1);
                $action!(2);
                $action!(3);
                $action!(4);
                $action!(5);
                $action!(6);
                $action!(7);
                $action!(8);
                $action!(9);
                $action!(10);
                $action!(11);
                $action!(12);
                $action!(13);
                $action!(14);
                $action!(15);
            };
        }
        macro_rules! each_rank {
            ($action:ident) => {
                $action!(0);
                $action!(1);
                $action!(2);
                $action!(3);
                $action!(4);
                $action!(5);
                $action!(6);
                $action!(7);
                $action!(8);
                $action!(9);
            };
        }

        // Lane `l` owns experts `slot * 32 + l` for `slot` in `0..16`. Every
        // later pass reads this register file, so the plane is read once.
        let mut owned = [0.0f32; OWNED];
        macro_rules! load_owned {
            ($slot:literal) => {{
                let expert = $slot * 32 + lane as usize;
                let bits = unsafe { *logits.add(expert) };
                owned[$slot] = f32::from_bits(u32::from(bits) << 16);
            }};
        }
        each_slot!(load_owned);

        // The full softmax subtracts the row maximum before exponentiating.
        // Maximum is order-free, so the lane scan and butterfly carry no
        // accumulation-order obligation.
        let mut maximum = f32::NEG_INFINITY;
        macro_rules! fold_maximum {
            ($slot:literal) => {{
                maximum = maximum.max(owned[$slot]);
            }};
        }
        each_slot!(fold_maximum);
        let mut delta = 16u32;
        while delta != 0 {
            maximum = maximum.max(warp::shuffle_xor_f32_sync(u32::MAX, maximum, delta));
            delta >>= 1;
        }

        // The 512-wide denominator follows ascending expert order: each lane
        // sums its sixteen owned exponentials in ascending slot order, then the
        // five-step butterfly folds the thirty-two partial sums. `owned` is
        // rewritten in place with the exponentials and then with the
        // probabilities, so no second register array is needed.
        let mut partial = 0.0f32;
        macro_rules! exponentiate {
            ($slot:literal) => {{
                let exponential =
                    float::ex2_approx_f32((owned[$slot] - maximum) * core::f32::consts::LOG2_E);
                owned[$slot] = exponential;
                partial += exponential;
            }};
        }
        each_slot!(exponentiate);
        let mut denominator = partial;
        delta = 16;
        while delta != 0 {
            denominator += warp::shuffle_xor_f32_sync(u32::MAX, denominator, delta);
            delta >>= 1;
        }

        macro_rules! normalize {
            ($slot:literal) => {{
                owned[$slot] = float::div_rn_f32(owned[$slot], denominator);
            }};
        }
        each_slot!(normalize);

        // Ten selection passes over the probabilities. Consumption is a
        // register mask on the owning lane and the butterfly hands every lane
        // the pass winner. Ties resolve to the lowest expert index in both the
        // lane scan and the cross-lane fold. `torch.topk` offers no contractual
        // order, so this target pins the rule and the oracle asserts it.
        let mut winners = [0.0f32; TOP_K];
        let mut winner_experts = [0u16; TOP_K];
        let mut taken = 0u32;
        macro_rules! select_pass {
            ($position:literal) => {{
                let mut best_value = f32::NEG_INFINITY;
                let mut best_index = u16::MAX;
                macro_rules! scan_slot {
                    ($slot:literal) => {{
                        if taken & (1u32 << $slot) == 0 {
                            let value = owned[$slot];
                            let index = ($slot * 32 + lane as usize) as u16;
                            let better =
                                value > best_value || (value == best_value && index < best_index);
                            if better {
                                best_value = value;
                                best_index = index;
                            }
                        }
                    }};
                }
                each_slot!(scan_slot);

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
                winners[$position] = best_value;
                winner_experts[$position] = best_index;
                if (best_index as usize & 31) == lane as usize {
                    taken |= 1u32 << ((best_index as usize) >> 5);
                }
            }};
        }
        each_rank!(select_pass);
        // The tenth pass still marks its winner consumed, which nothing reads.
        // Keeping the mark uniform across all ten passes is what makes them one
        // macro rather than nine plus a special case.
        let _ = taken;

        if lane != 0 {
            return;
        }

        // `norm_topk_prob = True`: the ten selected probabilities are summed in
        // descending rank order, exactly as `top_v.sum(-1)` sees them, and the
        // quotient is demoted to BF16 last.
        let mut selected = 0.0f32;
        macro_rules! accumulate_selected {
            ($position:literal) => {{
                selected += winners[$position];
            }};
        }
        each_rank!(accumulate_selected);

        // Publish ascending expert indices, not rank order, to preserve the
        // expert sum to the reference's `index_add_`, which visits experts
        // ascending, so the combine kernel's sequential walk over these ten
        // slots *is* that order. The rank order is consumed above by the
        // renormalization sum and is not published, because nothing downstream
        // may depend on it.
        //
        // The permutation is computed by counting rather than by sorting: a
        // sort's data-dependent shifts would index a register array
        // dynamically, which forfeits the zero-stack/local gate. Every index
        // here is a compile-time constant; only the destination is dynamic, and
        // it addresses global memory.
        macro_rules! publish {
            ($source:literal) => {{
                let expert = winner_experts[$source];
                let mut destination = 0usize;
                macro_rules! count_lower {
                    ($other:literal) => {{
                        if winner_experts[$other] < expert {
                            destination += 1;
                        }
                    }};
                }
                each_rank!(count_lower);
                unsafe {
                    *indices.add(destination) = expert;
                    *weights.add(destination) =
                        tcgen05::f32_to_bf16_rne(float::div_rn_f32(winners[$source], selected));
                }
            }};
        }
        each_rank!(publish);
    }

    /// Selects and normalizes the exact top-ten decode experts.
    #[kernel]
    #[launch_bounds(32, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_router_select<const TOKENS: usize>(
        logits: *const u16,
        expert_indices: *mut u16,
        expert_weights: *mut u16,
    ) {
        unsafe { router_select::<TOKENS>(logits, expert_indices, expert_weights) }
    }

    /// Selects and normalizes the exact top-ten prompt experts.
    #[kernel]
    #[launch_bounds(32, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_router_select_prefill<const TOKENS: usize>(
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
    projection: PreparedLaunch<kernels::__qwen38_flash_next_moe_router_logits_CudaKernel<TOKENS>>,
    select: PreparedLaunch<kernels::__qwen38_flash_next_moe_router_select_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !(1..=MAX_BATCH).contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE router decode row count {TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            projection:
                module
                    .prepare_qwen38_flash_next_moe_router_logits::<TOKENS>(projection_config::<
                        TOKENS,
                    >())
                    .map_err(|source| {
                        GpuError::launch(
                            "preparing the Qwen3.8-Flash-Next MoE router logits",
                            source,
                        )
                    })?,
            select: module
                .prepare_qwen38_flash_next_moe_router_select::<TOKENS>(select_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next MoE top-10 selection",
                        source,
                    )
                })?,
        })
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
            .qwen38_flash_next_moe_router_logits::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                logits,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next MoE router logits", source)
            })?;
        module
            .qwen38_flash_next_moe_router_select::<TOKENS>(
                stream,
                &self.select,
                logits,
                expert_indices,
                expert_weights,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next MoE top-10 selection",
                    source,
                )
            })
    }
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    projection:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_router_logits_prefill_CudaKernel<TOKENS>>,
    select:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_router_select_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE router prefill row count {TOKENS} is not admitted"
            )));
        }
        // T=1024 exposes 65,536 projection CTAs and 1,024 selector CTAs. The
        // eight-warps-per-eight-experts topology already fills the device at
        // T=32; the prompt routes change only the independent token count,
        // while every expert dot and every top-ten comparison keeps its exact
        // decode order.
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_moe_router_logits_prefill::<TOKENS>(projection_config::<
                    TOKENS,
                >())
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next MoE prompt router logits",
                        source,
                    )
                })?,
            select: module
                .prepare_qwen38_flash_next_moe_router_select_prefill::<TOKENS>(select_config::<
                    TOKENS,
                >())
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next MoE prompt top-10 selection",
                        source,
                    )
                })?,
        })
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
            .qwen38_flash_next_moe_router_logits_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                logits,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next MoE prompt router logits",
                    source,
                )
            })?;
        module
            .qwen38_flash_next_moe_router_select_prefill::<TOKENS>(
                stream,
                &self.select,
                logits,
                expert_indices,
                expert_weights,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next MoE prompt top-10 selection",
                    source,
                )
            })
    }
}

/// PTX symbols retained for every exact Qwen3.8-Flash-Next router batch.
pub(crate) fn qwen38_flash_next_moe_router_ptx_names() -> Vec<&'static str> {
    let mut names = Vec::with_capacity(2 * (MAX_BATCH + PREFILL_ROWS.len()));

    macro_rules! push_decode {
        ($tokens:literal) => {
            names.push(kernels::qwen38_flash_next_moe_router_logits_ptx_name::<
                $tokens,
            >());
            names.push(kernels::qwen38_flash_next_moe_router_select_ptx_name::<
                $tokens,
            >());
        };
    }
    macro_rules! push_prefill {
        ($tokens:literal) => {
            names.push(kernels::qwen38_flash_next_moe_router_logits_prefill_ptx_name::<$tokens>());
            names.push(kernels::qwen38_flash_next_moe_router_select_prefill_ptx_name::<$tokens>());
        };
    }

    push_decode!(1);
    push_decode!(2);
    push_decode!(3);
    push_decode!(4);
    push_decode!(5);
    push_decode!(6);
    push_decode!(7);
    push_decode!(8);
    push_prefill!(32);
    push_prefill!(64);
    push_prefill!(128);
    push_prefill!(1024);
    names
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_moe_router),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextMoeRouterRoutes {
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
    #[route(1024)]
    t1024: PreparedPrefillRoute<1024>,
}

/// Prepared exact-batch Qwen3.8-Flash-Next BF16 router routes on SM120.
pub struct Qwen38FlashNextMoeRouterOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextMoeRouterRoutes,
}

impl Qwen38FlashNextMoeRouterOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen38_flash_next_moe_router_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next MoE router", source)
        })?;

        Ok(Self {
            routes: Qwen38FlashNextMoeRouterRoutes::prepare(&module)?,
            module,
        })
    }

    /// Projects BF16 logits, selects top ten experts, and emits BF16 routing weights.
    ///
    /// Both outputs are published in **ascending expert index**, paired
    /// position by position, so a consumer that walks them in order reproduces
    /// the checkpoint route's accumulation order. Rank order remains internal
    /// to renormalization.
    ///
    /// # Safety
    ///
    /// `input` covers `rows * 2_560` BF16 values, `weights` covers BF16
    /// `[512, 2_560]`, `logits` covers `rows * 512` BF16 values, and both
    /// top-10 outputs cover `rows * 10` values. Four-byte-loaded input and
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
        dispatch_qwen38_flash_next_moe_router!(
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
                "Qwen3.8-Flash-Next MoE router row count {rows} is outside the admitted routes 1..={MAX_BATCH},32,64,128,1024"
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EXPERT_BLOCKS, MAX_BATCH, OWNED, PREFILL_ROWS, Qwen38FlashNextMoeRouterRoutes,
        WORDS_PER_ROW, qwen38_flash_next_moe_router_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(EXPERT_BLOCKS, 64);
        assert_eq!(WORDS_PER_ROW, 1_280);
        assert_eq!(OWNED, 16);

        let names = qwen38_flash_next_moe_router_ptx_names();
        assert_eq!(names.len(), 2 * (MAX_BATCH + PREFILL_ROWS.len()));
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }

    #[test]
    fn row_table_covers_only_exact_decode_and_prefill_routes() {
        assert_eq!(
            Qwen38FlashNextMoeRouterRoutes::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]
        );
    }
}

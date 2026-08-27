//! Exact Qwen3.6 MTP routed and shared BF16 expert execution.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 3] = [32, 64, 128];
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const SLOTS_PER_TOKEN: usize = TOP_K + 1;
const WORDS_PER_HIDDEN_ROW: usize = HIDDEN / 2;
const WORDS_PER_INTERMEDIATE_ROW: usize = INTERMEDIATE / 2;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const ROUTED_GATE_UP_WORDS_PER_EXPERT: usize = GATE_UP_ROWS * WORDS_PER_HIDDEN_ROW;
const ROUTED_DOWN_WORDS_PER_EXPERT: usize = HIDDEN * WORDS_PER_INTERMEDIATE_ROW;

// One warp owns one gate/up row, and eight adjacent rows share a CTA. The
// resulting 576 CTAs/token expose 3.39 target-SM waves while each selected
// expert streams its BF16 source rows once. This preserves each output's four
// FMA chains and fixed warp reduction; only independent rows are grouped.
const GATE_UP_WARPS: usize = 8;
const GATE_UP_THREADS: u32 = (GATE_UP_WARPS * 32) as u32;

// The 512-wide down dot has 256 BF16 pairs. One warp owns two output rows,
// so eight warps publish 16 rows and create 1,152 CTAs/token. Each output
// keeps the same four chains and warp reduction as the gate/up projection.
const DOWN_WARPS: usize = 8;
const DOWN_ROWS_PER_WARP: usize = 2;
const DOWN_ROWS_PER_CTA: usize = DOWN_WARPS * DOWN_ROWS_PER_WARP;
const DOWN_THREADS: u32 = (DOWN_WARPS * 32) as u32;

// Eight CTAs/token cover the 2,048-wide fixed-order combine. Every thread
// owns one output, retaining routed slots 0..7 followed by the shared slot.
const COMBINE_THREADS: u32 = 256;
const COMBINE_BLOCKS_PER_TOKEN: usize = HIDDEN / COMBINE_THREADS as usize;

const _: () = assert!(HIDDEN == 2_048);
const _: () = assert!(INTERMEDIATE == 512);
const _: () = assert!(EXPERTS == 256);
const _: () = assert!(TOP_K == 8);
const _: () = assert!(INTERMEDIATE.is_multiple_of(GATE_UP_WARPS));
const _: () = assert!(HIDDEN.is_multiple_of(DOWN_ROWS_PER_CTA));
const _: () = assert!(HIDDEN.is_multiple_of(COMBINE_THREADS as usize));

#[allow(clippy::too_many_arguments)]
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
    fn silu(value: f32) -> f32 {
        value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
    }

    #[inline(always)]
    fn fma_bf16_pair(input: (f32, f32), weight_word: u32, accumulator: f32) -> f32 {
        let (weight_low, weight_high) = convert::cvt_f32x2_bf16x2(weight_word);
        let accumulator = float::fma_rn_f32(weight_low, input.0, accumulator);
        float::fma_rn_f32(weight_high, input.1, accumulator)
    }

    #[inline(always)]
    unsafe fn selected_expert(token: usize, position: usize, expert_indices: *const u16) -> usize {
        if position < TOP_K {
            unsafe { *expert_indices.add(token * TOP_K + position) as usize }
        } else {
            0
        }
    }

    #[inline(always)]
    unsafe fn gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        routed_gate_up: *const u32,
        shared_gate: *const u32,
        shared_up: *const u32,
        shared_gate_weight: *const u32,
        intermediate_output: *mut u16,
        shared_gate_output: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_row = thread::blockIdx_x() as usize * GATE_UP_WARPS + warp_index;
        let slot = flat_row / INTERMEDIATE;
        let row = flat_row - slot * INTERMEDIATE;
        let token = slot / SLOTS_PER_TOKEN;
        let position = slot - token * SLOTS_PER_TOKEN;
        let expert = unsafe { selected_expert(token, position, expert_indices) };
        let input_row = unsafe { input.add(token * WORDS_PER_HIDDEN_ROW) };
        let routed = position < TOP_K;
        let (gate_row, up_row) = if routed {
            let base = unsafe { routed_gate_up.add(expert * ROUTED_GATE_UP_WORDS_PER_EXPERT) };
            (unsafe { base.add(row * WORDS_PER_HIDDEN_ROW) }, unsafe {
                base.add((INTERMEDIATE + row) * WORDS_PER_HIDDEN_ROW)
            })
        } else {
            (
                unsafe { shared_gate.add(row * WORDS_PER_HIDDEN_ROW) },
                unsafe { shared_up.add(row * WORDS_PER_HIDDEN_ROW) },
            )
        };
        let mut gate0 = 0.0f32;
        let mut gate1 = 0.0f32;
        let mut gate2 = 0.0f32;
        let mut gate3 = 0.0f32;
        let mut up0 = 0.0f32;
        let mut up1 = 0.0f32;
        let mut up2 = 0.0f32;
        let mut up3 = 0.0f32;
        let mut shared0 = 0.0f32;
        let mut shared1 = 0.0f32;
        let mut shared2 = 0.0f32;
        let mut shared3 = 0.0f32;
        let mut word = lane;

        // Runtime-indexed four-element accumulators produced 48-byte stack
        // frames. Scalar groups retain each chain's exact word sequence
        // (`lane + chain*32 + group*128`) while keeping all chains in registers.
        while word < WORDS_PER_HIDDEN_ROW {
            macro_rules! accumulate {
                ($offset:literal, $gate:ident, $up:ident, $shared:ident) => {{
                    let index = word + $offset;
                    let input = convert::cvt_f32x2_bf16x2(unsafe { *input_row.add(index) });
                    $gate = fma_bf16_pair(input, unsafe { *gate_row.add(index) }, $gate);
                    $up = fma_bf16_pair(input, unsafe { *up_row.add(index) }, $up);
                    if !routed && row == 0 {
                        $shared = fma_bf16_pair(
                            input,
                            unsafe { *shared_gate_weight.add(index) },
                            $shared,
                        );
                    }
                }};
            }
            accumulate!(0, gate0, up0, shared0);
            accumulate!(32, gate1, up1, shared1);
            accumulate!(64, gate2, up2, shared2);
            accumulate!(96, gate3, up3, shared3);
            word += 128;
        }

        let gate = reduce_sum_lane_zero(gate0 + gate1 + gate2 + gate3);
        let up = reduce_sum_lane_zero(up0 + up1 + up2 + up3);
        if lane == 0 {
            unsafe {
                *intermediate_output.add(slot * INTERMEDIATE + row) =
                    tcgen05::f32_to_bf16_rne(silu(gate) * up);
            }
        }
        if !routed && row == 0 {
            let shared = reduce_sum_lane_zero(shared0 + shared1 + shared2 + shared3);
            if lane == 0 {
                unsafe { *shared_gate_output.add(token) = tcgen05::f32_to_bf16_rne(shared) };
            }
        }
    }

    #[inline(always)]
    unsafe fn down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        routed_down: *const u32,
        shared_down: *const u32,
        expert_output: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_pair =
            thread::blockIdx_x() as usize * DOWN_ROWS_PER_CTA + warp_index * DOWN_ROWS_PER_WARP;
        let slot = flat_pair / HIDDEN;
        let first_row = flat_pair - slot * HIDDEN;
        let second_row = first_row + 1;
        let token = slot / SLOTS_PER_TOKEN;
        let position = slot - token * SLOTS_PER_TOKEN;
        let expert = unsafe { selected_expert(token, position, expert_indices) };
        let input_row = unsafe { intermediate_input.add(slot * WORDS_PER_INTERMEDIATE_ROW) };
        let weights = if position < TOP_K {
            unsafe { routed_down.add(expert * ROUTED_DOWN_WORDS_PER_EXPERT) }
        } else {
            shared_down
        };
        let first_weight = unsafe { weights.add(first_row * WORDS_PER_INTERMEDIATE_ROW) };
        let second_weight = unsafe { weights.add(second_row * WORDS_PER_INTERMEDIATE_ROW) };
        let mut first0 = 0.0f32;
        let mut first1 = 0.0f32;
        let mut first2 = 0.0f32;
        let mut first3 = 0.0f32;
        let mut second0 = 0.0f32;
        let mut second1 = 0.0f32;
        let mut second2 = 0.0f32;
        let mut second3 = 0.0f32;
        let mut word = lane;

        // Scalarizing the same four chains removes the 32-byte stack frame;
        // each chain still visits `lane + chain*32 + group*128` in order.
        while word < WORDS_PER_INTERMEDIATE_ROW {
            macro_rules! accumulate {
                ($offset:literal, $first:ident, $second:ident) => {{
                    let index = word + $offset;
                    let input = convert::cvt_f32x2_bf16x2(unsafe { *input_row.add(index) });
                    $first = fma_bf16_pair(input, unsafe { *first_weight.add(index) }, $first);
                    $second = fma_bf16_pair(input, unsafe { *second_weight.add(index) }, $second);
                }};
            }
            accumulate!(0, first0, second0);
            accumulate!(32, first1, second1);
            accumulate!(64, first2, second2);
            accumulate!(96, first3, second3);
            word += 128;
        }

        let first = reduce_sum_lane_zero(first0 + first1 + first2 + first3);
        let second = reduce_sum_lane_zero(second0 + second1 + second2 + second3);
        if lane == 0 {
            unsafe {
                *expert_output.add(slot * HIDDEN + first_row) = tcgen05::f32_to_bf16_rne(first);
                *expert_output.add(slot * HIDDEN + second_row) = tcgen05::f32_to_bf16_rne(second);
            }
        }
    }

    #[inline(always)]
    unsafe fn combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_gate: *const u16,
        output: *mut u16,
    ) {
        let flat = thread::blockIdx_x() as usize * COMBINE_THREADS as usize
            + thread::threadIdx_x() as usize;
        let token = flat / HIDDEN;
        let column = flat - token * HIDDEN;
        let token_slots = unsafe { expert_output.add(token * SLOTS_PER_TOKEN * HIDDEN) };
        let token_weights = unsafe { routing_weights.add(token * TOP_K) };
        let mut sum = 0.0f32;
        let mut position = 0usize;
        while position < TOP_K {
            let value = f32::from_bits(
                u32::from(unsafe { *token_slots.add(position * HIDDEN + column) }) << 16,
            );
            let weight = f32::from_bits(u32::from(unsafe { *token_weights.add(position) }) << 16);
            sum = float::fma_rn_f32(value, weight, sum);
            position += 1;
        }
        let shared =
            f32::from_bits(u32::from(unsafe { *token_slots.add(TOP_K * HIDDEN + column) }) << 16);
        let gate = f32::from_bits(u32::from(unsafe { *shared_gate.add(token) }) << 16);
        sum = float::fma_rn_f32(shared, sigmoid(gate), sum);
        unsafe { *output.add(token * HIDDEN + column) = tcgen05::f32_to_bf16_rne(sum) };
    }

    /// Executes routed and shared BF16 gate/up projections.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        routed_gate_up: *const u32,
        shared_gate: *const u32,
        shared_up: *const u32,
        shared_gate_weight: *const u32,
        intermediate_output: *mut u16,
        shared_gate_output: *mut u16,
    ) {
        unsafe {
            gate_up::<TOKENS>(
                input,
                expert_indices,
                routed_gate_up,
                shared_gate,
                shared_up,
                shared_gate_weight,
                intermediate_output,
                shared_gate_output,
            )
        }
    }

    /// Executes routed and shared BF16 down projections.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        routed_down: *const u32,
        shared_down: *const u32,
        expert_output: *mut u16,
    ) {
        unsafe {
            down::<TOKENS>(
                intermediate_input,
                expert_indices,
                routed_down,
                shared_down,
                expert_output,
            )
        }
    }

    /// Combines eight routed outputs and the gated shared output in fixed order.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_expert_combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_gate: *const u16,
        output: *mut u16,
    ) {
        unsafe { combine::<TOKENS>(expert_output, routing_weights, shared_gate, output) }
    }
}

fn gate_up_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * SLOTS_PER_TOKEN * INTERMEDIATE / GATE_UP_WARPS) as u32,
        GATE_UP_THREADS,
        0,
    )
}

fn down_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * SLOTS_PER_TOKEN * HIDDEN / DOWN_ROWS_PER_CTA) as u32,
        DOWN_THREADS,
        0,
    )
}

fn combine_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * COMBINE_BLOCKS_PER_TOKEN) as u32,
        COMBINE_THREADS,
        0,
    )
}

struct PreparedRoute<const TOKENS: usize> {
    gate_up: PreparedLaunch<kernels::__qwen36_mtp_bf16_expert_gate_up_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__qwen36_mtp_bf16_expert_down_CudaKernel<TOKENS>>,
    combine: PreparedLaunch<kernels::__qwen36_mtp_bf16_expert_combine_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            gate_up: module
                .prepare_qwen36_mtp_bf16_expert_gate_up::<TOKENS>(gate_up_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MTP BF16 expert gate/up", source)
                })?,
            down: module
                .prepare_qwen36_mtp_bf16_expert_down::<TOKENS>(down_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MTP BF16 expert down", source)
                })?,
            combine: module
                .prepare_qwen36_mtp_bf16_expert_combine::<TOKENS>(combine_config::<TOKENS>())
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 MTP BF16 expert combine", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        expert_indices: *const u16,
        routing_weights: *const u16,
        routed_gate_up: *const u16,
        routed_down: *const u16,
        shared_gate: *const u16,
        shared_up: *const u16,
        shared_down: *const u16,
        shared_gate_weight: *const u16,
        intermediate: *mut u16,
        expert_output: *mut u16,
        shared_gate_output: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_mtp_bf16_expert_gate_up::<TOKENS>(
                stream,
                &self.gate_up,
                input.cast::<u32>(),
                expert_indices,
                routed_gate_up.cast::<u32>(),
                shared_gate.cast::<u32>(),
                shared_up.cast::<u32>(),
                shared_gate_weight.cast::<u32>(),
                intermediate,
                shared_gate_output,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 MTP BF16 expert gate/up", source)
            })?;
        module
            .qwen36_mtp_bf16_expert_down::<TOKENS>(
                stream,
                &self.down,
                intermediate.cast::<u32>(),
                expert_indices,
                routed_down.cast::<u32>(),
                shared_down.cast::<u32>(),
                expert_output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MTP BF16 expert down", source))?;
        module
            .qwen36_mtp_bf16_expert_combine::<TOKENS>(
                stream,
                &self.combine,
                expert_output,
                routing_weights,
                shared_gate_output,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 MTP BF16 expert combine", source))
    }
}

/// PTX symbols retained for every exact Qwen3.6 MTP BF16 expert route.
pub(crate) fn qwen36_mtp_bf16_moe_ptx_names() -> Vec<&'static str> {
    let mut names = Vec::with_capacity(3 * (MAX_BATCH + PREFILL_ROWS.len()));
    macro_rules! push_route {
        ($tokens:literal) => {
            names.push(kernels::qwen36_mtp_bf16_expert_gate_up_ptx_name::<$tokens>());
            names.push(kernels::qwen36_mtp_bf16_expert_down_ptx_name::<$tokens>());
            names.push(kernels::qwen36_mtp_bf16_expert_combine_ptx_name::<$tokens>());
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
    push_route!(32);
    push_route!(64);
    push_route!(128);
    names
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen36_mtp_bf16_moe),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct Qwen36MtpBf16MoeRoutes {
    #[route(1)]
    b1: PreparedRoute<1>,
    #[route(2)]
    b2: PreparedRoute<2>,
    #[route(3)]
    b3: PreparedRoute<3>,
    #[route(4)]
    b4: PreparedRoute<4>,
    #[route(5)]
    b5: PreparedRoute<5>,
    #[route(6)]
    b6: PreparedRoute<6>,
    #[route(7)]
    b7: PreparedRoute<7>,
    #[route(8)]
    b8: PreparedRoute<8>,
    #[route(32)]
    t32: PreparedRoute<32>,
    #[route(64)]
    t64: PreparedRoute<64>,
    #[route(128)]
    t128: PreparedRoute<128>,
}

/// Prepared exact-route Qwen3.6 routed/shared BF16 MTP experts.
pub struct Qwen36MtpBf16MoeOp {
    module: kernels::LoadedModule,
    routes: Qwen36MtpBf16MoeRoutes,
}

impl Qwen36MtpBf16MoeOp {
    /// Loads the module and prepares every exact decode and prompt route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_mtp_bf16_moe_ptx_names();
        // SAFETY: this crate owns the embedded exact-target artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 MTP BF16 experts", source))?;
        let routes = Qwen36MtpBf16MoeRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Executes selected routed experts, the shared expert, and fixed-order reduction.
    ///
    /// # Safety
    ///
    /// Every pointer covers its complete source-BF16 Qwen3.6 MTP plane.
    /// Selected expert indices are below 256. Workspaces cover
    /// `[rows,9,512]`, `[rows,9,2048]`, `[rows]`, and `[rows,2048]`.
    /// Four-byte-loaded planes are aligned, disjoint, context-local, and live
    /// through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        expert_indices: *const u16,
        routing_weights: *const u16,
        routed_gate_up: *const u16,
        routed_down: *const u16,
        shared_gate: *const u16,
        shared_up: *const u16,
        shared_down: *const u16,
        shared_gate_weight: *const u16,
        intermediate: *mut u16,
        expert_output: *mut u16,
        shared_gate_output: *mut u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_qwen36_mtp_bf16_moe!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    input,
                    expert_indices,
                    routing_weights,
                    routed_gate_up,
                    routed_down,
                    shared_gate,
                    shared_up,
                    shared_down,
                    shared_gate_weight,
                    intermediate,
                    expert_output,
                    shared_gate_output,
                    output,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 MTP BF16 expert rows {rows} are outside 1..={MAX_BATCH},32,64,128"
            )))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn geometry_route_table_and_inventory_are_exact() {
        assert_eq!(ROUTED_GATE_UP_WORDS_PER_EXPERT * 4, 4_194_304);
        assert_eq!(ROUTED_DOWN_WORDS_PER_EXPERT * 4, 2_097_152);
        for (rows, admitted) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (32, true),
            (64, true),
            (128, true),
            (129, false),
        ] {
            assert_eq!(
                Qwen36MtpBf16MoeRoutes::contains(rows),
                admitted,
                "rows={rows}"
            );
        }
        assert_eq!(
            Qwen36MtpBf16MoeRoutes::admitted_rows(),
            [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]
        );
        let names = qwen36_mtp_bf16_moe_ptx_names();
        assert_eq!(names.len(), 33);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 33);
    }
}

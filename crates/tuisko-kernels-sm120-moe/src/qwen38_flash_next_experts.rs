//! Exact Qwen3.8-Flash-Next expert dispatch: slot-indirected NVFP4 routed
//! experts and the resident BF16 shared expert.
//!
//! The routed expert pool is streaming-resident, so these kernels never address
//! an expert directly. They read a device-visible indirection table, expert id ->
//! slot index, and dereference an address-stable slot in a sealed pool. Cache
//! state must never change produced bits, which is why the slot index appears
//! only in an address computation and never in a value: permuting the slot
//! assignment moves the bytes and leaves the arithmetic identical.
//!
//! Kernels never handle absence. By the streaming publication law a replay only
//! observes a table whose every referenced slot is uploaded - `require` stalls
//! rather than skipping - so an `ABSENT_SLOT` reaching a kernel is a host-side
//! contract violation. [`Qwen38FlashNextSlotPlane`] proves the published table
//! well-formed before replay.

use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const PREFILL_ROWS: [usize; 4] = [32, 64, 128, 1_024];
const HIDDEN: usize = Qwen38FlashNext::HIDDEN;
const INTERMEDIATE: usize = Qwen38FlashNext::INTERMEDIATE;
const EXPERTS: usize = Qwen38FlashNext::NUM_EXPERTS;
const TOP_K: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
const SHARED_INTERMEDIATE: usize = Qwen38FlashNext::SHARED_EXPERT_INTERMEDIATE;
const GROUP_K: usize = 16;

/// Rows of the fused gate||up plane: gate occupies `0..640`, up `640..1280`.
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const GATE_UP_GROUPS: usize = HIDDEN / GROUP_K;
const DOWN_GROUPS: usize = INTERMEDIATE / GROUP_K;

/// Sentinel the indirection table carries for an expert with no resident slot.
pub const QWEN38_FLASH_NEXT_ABSENT_SLOT: u32 = u32::MAX;

// One expert's slot extent, and the four planes inside it. The packed E2M1
// planes keep the checkpoint adapter's `down || gate || up` order, so staging
// one expert is a single contiguous read. Scales follow as gate||up then down.
const DOWN_CODE_OFFSET: usize = 0;
const DOWN_CODE_BYTES: usize = HIDDEN * INTERMEDIATE / 2;
const GATE_UP_CODE_OFFSET: usize = DOWN_CODE_OFFSET + DOWN_CODE_BYTES;
const GATE_UP_CODE_BYTES: usize = GATE_UP_ROWS * HIDDEN / 2;
const GATE_UP_SCALE_OFFSET: usize = GATE_UP_CODE_OFFSET + GATE_UP_CODE_BYTES;
const GATE_UP_SCALE_BYTES: usize = GATE_UP_ROWS * GATE_UP_GROUPS;
const DOWN_SCALE_OFFSET: usize = GATE_UP_SCALE_OFFSET + GATE_UP_SCALE_BYTES;
const DOWN_SCALE_BYTES: usize = HIDDEN * DOWN_GROUPS;

/// Byte stride between two slots in the sealed pool.
pub const QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES: usize = DOWN_SCALE_OFFSET + DOWN_SCALE_BYTES;

/// Per-expert F32 globals: gate, up, and down `weight_scale_2`, in that order.
const GLOBAL_SCALES_PER_EXPERT: usize = 3;

// The routed gate/up warp owns one intermediate row and walks the 2,560-wide
// activation from shared memory; eight warps per CTA is the Qwen3.6 expert
// topology at 1.25x the row width. 160 scale groups divide evenly by the warp,
// so every lane runs the same five phases.
const GATE_UP_WARPS: usize = 8;
const GATE_UP_THREADS: u32 = (GATE_UP_WARPS * 32) as u32;
const GATE_UP_SHARED_U32: usize = HIDDEN / 2;
const GATE_UP_PHASES: usize = GATE_UP_GROUPS / 32;

// The down warp owns two output rows so one shared activation load serves two
// dot products. 40 scale groups do not divide by the warp, so lanes 0..7 run a
// second group; the per-lane order stays ascending and the fold is unchanged.
const DOWN_WARPS: usize = 8;
const DOWN_ROWS_PER_WARP: usize = 2;
const DOWN_ROWS_PER_CTA: usize = DOWN_WARPS * DOWN_ROWS_PER_WARP;
const DOWN_THREADS: u32 = (DOWN_WARPS * 32) as u32;
const DOWN_SHARED_U32: usize = INTERMEDIATE / 2;

const COMBINE_THREADS: u32 = 256;

const _: () = assert!(HIDDEN == 2_560);
const _: () = assert!(INTERMEDIATE == 640);
const _: () = assert!(EXPERTS == 512);
const _: () = assert!(TOP_K == 10);
const _: () = assert!(SHARED_INTERMEDIATE == INTERMEDIATE);
const _: () = assert!(GATE_UP_GROUPS == 160);
const _: () = assert!(DOWN_GROUPS == 40);
const _: () = assert!(GATE_UP_PHASES == 5);
// Recompute the adapter's exact extents from admitted geometry.
const _: () = assert!(DOWN_CODE_BYTES == 819_200);
const _: () = assert!(GATE_UP_CODE_BYTES == 1_638_400);
const _: () = assert!(GATE_UP_CODE_OFFSET + GATE_UP_CODE_BYTES == 2_457_600);
const _: () = assert!(GATE_UP_SCALE_BYTES == 204_800);
const _: () = assert!(DOWN_SCALE_BYTES == 102_400);
const _: () = assert!(QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES == 2_764_800);
const _: () = assert!(HIDDEN.is_multiple_of(GROUP_K));
const _: () = assert!(INTERMEDIATE.is_multiple_of(GROUP_K));
const _: () = assert!(INTERMEDIATE.is_multiple_of(GATE_UP_WARPS));
const _: () = assert!(HIDDEN.is_multiple_of(DOWN_ROWS_PER_CTA));
// Both staged rows are loaded by a strided walk, so neither has to divide the
// block; this records that the one-shot form the Qwen3.6 route uses would be
// wrong here, rather than leaving it to be rediscovered.
const _: () = assert!(DOWN_SHARED_U32 > DOWN_THREADS as usize);
const _: () = assert!(GATE_UP_SHARED_U32 > GATE_UP_THREADS as usize);
const _: () = assert!(HIDDEN.is_multiple_of(COMBINE_THREADS as usize));

fn admitted_rows(rows: usize) -> bool {
    (1..=MAX_BATCH).contains(&rows) || PREFILL_ROWS.contains(&rows)
}

#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, warp};

    /// `BlockScaleK16M128x4`: 128-row persistent tiles of four groups, each a
    /// 512-byte block addressed by `(row & 31) * 16 + (row >> 5) * 4 + group`.
    #[inline(always)]
    fn scale_offset(row: usize, group: usize, groups_per_row: usize) -> usize {
        let persistent_tile = row >> 7;
        let row_in_tile = row & 127;
        let scale_tile = group >> 2;

        (persistent_tile * (groups_per_row >> 2) + scale_tile) * 512
            + (row_in_tile & 31) * 16
            + (row_in_tile >> 5) * 4
            + (group & 3)
    }

    #[inline(always)]
    fn e4m3_to_f32(code: u8) -> f32 {
        let duplicated = code as u16 | ((code as u16) << 8);
        let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(duplicated);

        convert::cvt_f32_f16x2_lo(packed_f16)
    }

    #[inline(always)]
    fn e2m1x2_to_f32(packed: u8) -> (f32, f32) {
        let packed_f16: u32;
        let storage = packed as u16;

        unsafe {
            ptx_asm!(
                "{ .reg .b8 lo, zero; mov.b16 {lo, zero}, %1; \
                 cvt.rn.f16x2.e2m1x2 %0, lo; }",
                out("=r") packed_f16,
                in("h") storage,
                options(register_only),
            );
        }

        convert::cvt_f32x2_f16x2(packed_f16)
    }

    #[inline(always)]
    unsafe fn load_u8_read_only(source: *const u8) -> u8 {
        let value: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.u8 %0, [%1];",
                out("=r") value,
                in("l") source,
                clobber("memory"),
            );
        }

        value as u8
    }

    #[inline(always)]
    unsafe fn load_u32x2_read_only(source: *const u32) -> (u32, u32) {
        let first: u32;
        let second: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.v2.u32 {%0, %1}, [%2];",
                out("=r") first,
                out("=r") second,
                in("l") source,
                clobber("memory"),
            );
        }

        (first, second)
    }

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

    /// Dereferences one expert's slot. The table is trusted: by the streaming
    /// publication law every entry a replay reads has been uploaded, so no
    /// `ABSENT_SLOT` branch exists here - see [`Qwen38FlashNextSlotPlane`].
    #[inline(always)]
    unsafe fn slot_base(expert: usize, slot_table: *const u32, slot_pool: *const u8) -> *const u8 {
        let slot = unsafe { *slot_table.add(expert) } as usize;

        unsafe { slot_pool.add(slot * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES) }
    }

    #[inline(always)]
    unsafe fn expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        intermediate_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, GATE_UP_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_row = thread::blockIdx_x() as usize * GATE_UP_WARPS + warp_index;
        let slot = flat_row / INTERMEDIATE;
        let row = flat_row - slot * INTERMEDIATE;
        let token = slot / TOP_K;
        let position = slot - token * TOP_K;
        let expert = unsafe { *expert_indices.add(token * TOP_K + position) } as usize;
        let input_row = unsafe { input.add(token * (HIDDEN / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();
        let mut word = tid;

        while word < GATE_UP_SHARED_U32 {
            unsafe { *shared_input.add(word) = *input_row.add(word) };
            word += GATE_UP_THREADS as usize;
        }
        thread::sync_threads();

        let base = unsafe { slot_base(expert, slot_table, slot_pool) };
        let codes = unsafe { base.add(GATE_UP_CODE_OFFSET) };
        let scales = unsafe { base.add(GATE_UP_SCALE_OFFSET) };
        let gate_scale_2 = unsafe { *weight_scales_2.add(expert * GLOBAL_SCALES_PER_EXPERT) };
        let up_scale_2 = unsafe { *weight_scales_2.add(expert * GLOBAL_SCALES_PER_EXPERT + 1) };
        let gate_row = row;
        let up_row = row + INTERMEDIATE;
        let row_code_bytes = HIDDEN / 2;
        let mut gate = [0.0f32; 4];
        let mut up = [0.0f32; 4];
        let mut phase = 0usize;

        while phase < GATE_UP_PHASES {
            let group = phase * 32 + lane;
            let gate_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(gate_row, group, GATE_UP_GROUPS)))
            };
            let up_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(up_row, group, GATE_UP_GROUPS)))
            };
            let gate_coefficient = e4m3_to_f32(gate_scale) * gate_scale_2;
            let up_coefficient = e4m3_to_f32(up_scale) * up_scale_2;
            let gate_source =
                unsafe { codes.add(gate_row * row_code_bytes + group * (GROUP_K / 2)) };
            let up_source = unsafe { codes.add(up_row * row_code_bytes + group * (GROUP_K / 2)) };
            let gate_words = unsafe { load_u32x2_read_only(gate_source.cast::<u32>()) };
            let up_words = unsafe { load_u32x2_read_only(up_source.cast::<u32>()) };
            let activation_source = unsafe { shared_input.add(group * (GROUP_K / 2)) };

            macro_rules! accumulate_pair {
                ($pair:literal, $chain:literal) => {{
                    let shift = ($pair & 3) * 8;
                    let gate_packed = if $pair < 4 {
                        (gate_words.0 >> shift) as u8
                    } else {
                        (gate_words.1 >> shift) as u8
                    };
                    let up_packed = if $pair < 4 {
                        (up_words.0 >> shift) as u8
                    } else {
                        (up_words.1 >> shift) as u8
                    };
                    let (gate_low, gate_high) = e2m1x2_to_f32(gate_packed);
                    let (up_low, up_high) = e2m1x2_to_f32(up_packed);
                    let activation_bits = unsafe { *activation_source.add($pair) };
                    let (activation_low, activation_high) =
                        convert::cvt_f32x2_bf16x2(activation_bits);
                    gate[$chain] = float::fma_rn_f32(
                        gate_low * gate_coefficient,
                        activation_low,
                        gate[$chain],
                    );
                    gate[$chain] = float::fma_rn_f32(
                        gate_high * gate_coefficient,
                        activation_high,
                        gate[$chain],
                    );
                    up[$chain] =
                        float::fma_rn_f32(up_low * up_coefficient, activation_low, up[$chain]);
                    up[$chain] =
                        float::fma_rn_f32(up_high * up_coefficient, activation_high, up[$chain]);
                }};
            }

            accumulate_pair!(0, 0);
            accumulate_pair!(1, 1);
            accumulate_pair!(2, 2);
            accumulate_pair!(3, 3);
            accumulate_pair!(4, 0);
            accumulate_pair!(5, 1);
            accumulate_pair!(6, 2);
            accumulate_pair!(7, 3);
            phase += 1;
        }

        let gate = reduce_sum_lane_zero(gate[0] + gate[1] + gate[2] + gate[3]);
        let up = reduce_sum_lane_zero(up[0] + up[1] + up[2] + up[3]);
        if lane == 0 {
            unsafe {
                *intermediate_output.add(slot * INTERMEDIATE + row) =
                    tcgen05::f32_to_bf16_rne(silu(gate) * up);
            }
        }
    }

    /// Executes one decode batch's routed gate/up through the slot pool.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        intermediate_output: *mut u16,
    ) {
        unsafe {
            expert_gate_up::<TOKENS>(
                input,
                expert_indices,
                slot_table,
                slot_pool,
                weight_scales_2,
                intermediate_output,
            )
        }
    }

    /// Executes one prompt tile's routed gate/up through the slot pool.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_gate_up_prefill<const TOKENS: usize>(
        input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        intermediate_output: *mut u16,
    ) {
        unsafe {
            expert_gate_up::<TOKENS>(
                input,
                expert_indices,
                slot_table,
                slot_pool,
                weight_scales_2,
                intermediate_output,
            )
        }
    }

    #[inline(always)]
    unsafe fn expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        expert_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, DOWN_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_pair =
            thread::blockIdx_x() as usize * DOWN_ROWS_PER_CTA + DOWN_ROWS_PER_WARP * warp_index;
        let slot = flat_pair / HIDDEN;
        let first_row = flat_pair - slot * HIDDEN;
        let second_row = first_row + 1;
        let token = slot / TOP_K;
        let position = slot - token * TOP_K;
        let expert = unsafe { *expert_indices.add(token * TOP_K + position) } as usize;
        let input_row = unsafe { intermediate_input.add(slot * (INTERMEDIATE / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();

        // Stage all 320 words with 256 threads. A one-shot load would leave the
        // final 128 activations unwritten.
        let mut word = tid;
        while word < DOWN_SHARED_U32 {
            unsafe { *shared_input.add(word) = *input_row.add(word) };
            word += DOWN_THREADS as usize;
        }
        thread::sync_threads();

        let base = unsafe { slot_base(expert, slot_table, slot_pool) };
        let codes = unsafe { base.add(DOWN_CODE_OFFSET) };
        let scales = unsafe { base.add(DOWN_SCALE_OFFSET) };
        let weight_scale_2 = unsafe { *weight_scales_2.add(expert * GLOBAL_SCALES_PER_EXPERT + 2) };
        let row_code_bytes = INTERMEDIATE / 2;
        let mut first = [0.0f32; 4];
        let mut second = [0.0f32; 4];
        let mut group = lane;

        while group < DOWN_GROUPS {
            let first_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(first_row, group, DOWN_GROUPS)))
            };
            let second_scale = unsafe {
                load_u8_read_only(scales.add(scale_offset(second_row, group, DOWN_GROUPS)))
            };
            let first_coefficient = e4m3_to_f32(first_scale) * weight_scale_2;
            let second_coefficient = e4m3_to_f32(second_scale) * weight_scale_2;
            let first_source =
                unsafe { codes.add(first_row * row_code_bytes + group * (GROUP_K / 2)) };
            let second_source =
                unsafe { codes.add(second_row * row_code_bytes + group * (GROUP_K / 2)) };
            let first_words = unsafe { load_u32x2_read_only(first_source.cast::<u32>()) };
            let second_words = unsafe { load_u32x2_read_only(second_source.cast::<u32>()) };
            let activation_source = unsafe { shared_input.add(group * (GROUP_K / 2)) };

            macro_rules! accumulate_pair {
                ($pair:literal, $chain:literal) => {{
                    let shift = ($pair & 3) * 8;
                    let first_packed = if $pair < 4 {
                        (first_words.0 >> shift) as u8
                    } else {
                        (first_words.1 >> shift) as u8
                    };
                    let second_packed = if $pair < 4 {
                        (second_words.0 >> shift) as u8
                    } else {
                        (second_words.1 >> shift) as u8
                    };
                    let (first_low, first_high) = e2m1x2_to_f32(first_packed);
                    let (second_low, second_high) = e2m1x2_to_f32(second_packed);
                    let activation_bits = unsafe { *activation_source.add($pair) };
                    let (activation_low, activation_high) =
                        convert::cvt_f32x2_bf16x2(activation_bits);
                    first[$chain] = float::fma_rn_f32(
                        first_low * first_coefficient,
                        activation_low,
                        first[$chain],
                    );
                    first[$chain] = float::fma_rn_f32(
                        first_high * first_coefficient,
                        activation_high,
                        first[$chain],
                    );
                    second[$chain] = float::fma_rn_f32(
                        second_low * second_coefficient,
                        activation_low,
                        second[$chain],
                    );
                    second[$chain] = float::fma_rn_f32(
                        second_high * second_coefficient,
                        activation_high,
                        second[$chain],
                    );
                }};
            }

            accumulate_pair!(0, 0);
            accumulate_pair!(1, 1);
            accumulate_pair!(2, 2);
            accumulate_pair!(3, 3);
            accumulate_pair!(4, 0);
            accumulate_pair!(5, 1);
            accumulate_pair!(6, 2);
            accumulate_pair!(7, 3);
            group += 32;
        }

        let first = reduce_sum_lane_zero(first[0] + first[1] + first[2] + first[3]);
        let second = reduce_sum_lane_zero(second[0] + second[1] + second[2] + second[3]);
        if lane == 0 {
            unsafe {
                *expert_output.add(slot * HIDDEN + first_row) = tcgen05::f32_to_bf16_rne(first);
                *expert_output.add(slot * HIDDEN + second_row) = tcgen05::f32_to_bf16_rne(second);
            }
        }
    }

    /// Executes one decode batch's routed down projection through the slot pool.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        expert_output: *mut u16,
    ) {
        unsafe {
            expert_down::<TOKENS>(
                intermediate_input,
                expert_indices,
                slot_table,
                slot_pool,
                weight_scales_2,
                expert_output,
            )
        }
    }

    /// Executes one prompt tile's routed down projection through the slot pool.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_down_prefill<const TOKENS: usize>(
        intermediate_input: *const u32,
        expert_indices: *const u16,
        slot_table: *const u32,
        slot_pool: *const u8,
        weight_scales_2: *const f32,
        expert_output: *mut u16,
    ) {
        unsafe {
            expert_down::<TOKENS>(
                intermediate_input,
                expert_indices,
                slot_table,
                slot_pool,
                weight_scales_2,
                expert_output,
            )
        }
    }

    /// The always-active shared expert's gate/up, and its scalar gate logit.
    ///
    /// `*.mlp.shared_expert.*` and `*.mlp.shared_expert_gate*` are both in the
    /// checkpoint's `exclude_modules`, so this plane is BF16 and device-resident
    /// - no slot, no codes, no block scales. Keeping it in its own entries
    /// rather than folding it into the routed ones as a `position == TOP_K`
    /// branch is what lets the resource gate assert that a routed entry
    /// converts E2M1 and that this one never does.
    #[inline(always)]
    unsafe fn shared_expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        gate_weight: *const u32,
        up_weight: *const u32,
        gate_logit_weight: *const u32,
        intermediate_output: *mut u16,
        gate_logit_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, GATE_UP_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_row = thread::blockIdx_x() as usize * GATE_UP_WARPS + warp_index;
        let token = flat_row / INTERMEDIATE;
        let row = flat_row - token * INTERMEDIATE;
        let input_row = unsafe { input.add(token * (HIDDEN / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();
        let mut word = tid;

        while word < GATE_UP_SHARED_U32 {
            unsafe { *shared_input.add(word) = *input_row.add(word) };
            word += GATE_UP_THREADS as usize;
        }
        thread::sync_threads();

        let row_words = HIDDEN / 2;
        let gate_row = unsafe { gate_weight.add(row * row_words) };
        let up_row = unsafe { up_weight.add(row * row_words) };
        let mut gate = 0.0f32;
        let mut up = 0.0f32;
        let mut logit = 0.0f32;
        let mut word = lane;

        while word < row_words {
            let activation_bits = unsafe { *shared_input.add(word) };
            let (activation_low, activation_high) = convert::cvt_f32x2_bf16x2(activation_bits);
            let (gate_low, gate_high) = convert::cvt_f32x2_bf16x2(unsafe { *gate_row.add(word) });
            let (up_low, up_high) = convert::cvt_f32x2_bf16x2(unsafe { *up_row.add(word) });
            gate = float::fma_rn_f32(gate_low, activation_low, gate);
            gate = float::fma_rn_f32(gate_high, activation_high, gate);
            up = float::fma_rn_f32(up_low, activation_low, up);
            up = float::fma_rn_f32(up_high, activation_high, up);
            if row == 0 {
                let (logit_low, logit_high) =
                    convert::cvt_f32x2_bf16x2(unsafe { *gate_logit_weight.add(word) });
                logit = float::fma_rn_f32(logit_low, activation_low, logit);
                logit = float::fma_rn_f32(logit_high, activation_high, logit);
            }
            word += 32;
        }

        let gate = reduce_sum_lane_zero(gate);
        let up = reduce_sum_lane_zero(up);
        if lane == 0 {
            unsafe {
                *intermediate_output.add(token * INTERMEDIATE + row) =
                    tcgen05::f32_to_bf16_rne(silu(gate) * up);
            }
        }
        if row == 0 {
            let logit = reduce_sum_lane_zero(logit);
            if lane == 0 {
                unsafe { *gate_logit_output.add(token) = tcgen05::f32_to_bf16_rne(logit) };
            }
        }
    }

    /// Executes one decode batch's BF16 shared expert gate/up.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_shared_expert_gate_up<const TOKENS: usize>(
        input: *const u32,
        gate_weight: *const u32,
        up_weight: *const u32,
        gate_logit_weight: *const u32,
        intermediate_output: *mut u16,
        gate_logit_output: *mut u16,
    ) {
        unsafe {
            shared_expert_gate_up::<TOKENS>(
                input,
                gate_weight,
                up_weight,
                gate_logit_weight,
                intermediate_output,
                gate_logit_output,
            )
        }
    }

    /// Executes one prompt tile's BF16 shared expert gate/up.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_shared_expert_gate_up_prefill<const TOKENS: usize>(
        input: *const u32,
        gate_weight: *const u32,
        up_weight: *const u32,
        gate_logit_weight: *const u32,
        intermediate_output: *mut u16,
        gate_logit_output: *mut u16,
    ) {
        unsafe {
            shared_expert_gate_up::<TOKENS>(
                input,
                gate_weight,
                up_weight,
                gate_logit_weight,
                intermediate_output,
                gate_logit_output,
            )
        }
    }

    #[inline(always)]
    unsafe fn shared_expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        down_weight: *const u32,
        shared_output: *mut u16,
    ) {
        static mut SHARED_INPUT: SharedArray<u32, DOWN_SHARED_U32, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let flat_pair =
            thread::blockIdx_x() as usize * DOWN_ROWS_PER_CTA + DOWN_ROWS_PER_WARP * warp_index;
        let token = flat_pair / HIDDEN;
        let first_row = flat_pair - token * HIDDEN;
        let second_row = first_row + 1;
        let input_row = unsafe { intermediate_input.add(token * (INTERMEDIATE / 2)) };
        let shared_input = core::ptr::addr_of_mut!(SHARED_INPUT).cast::<u32>();

        // Strided, not a single `if tid < DOWN_SHARED_U32`: at
        // INTERMEDIATE = 640 the staged row is 320 words against 256 threads,
        // so a one-shot load would silently leave words 256..320 -- the last
        // 128 activations of every down input -- unwritten. Qwen3.6's 512-wide
        // intermediate made that exactly one word per thread, which is why the
        // idiom it inherited did not survive the widening.
        let mut word = tid;
        while word < DOWN_SHARED_U32 {
            unsafe { *shared_input.add(word) = *input_row.add(word) };
            word += DOWN_THREADS as usize;
        }
        thread::sync_threads();

        let row_words = INTERMEDIATE / 2;
        let first_source = unsafe { down_weight.add(first_row * row_words) };
        let second_source = unsafe { down_weight.add(second_row * row_words) };
        let mut first = 0.0f32;
        let mut second = 0.0f32;
        let mut word = lane;

        while word < row_words {
            let activation_bits = unsafe { *shared_input.add(word) };
            let (activation_low, activation_high) = convert::cvt_f32x2_bf16x2(activation_bits);
            let (first_low, first_high) =
                convert::cvt_f32x2_bf16x2(unsafe { *first_source.add(word) });
            let (second_low, second_high) =
                convert::cvt_f32x2_bf16x2(unsafe { *second_source.add(word) });
            first = float::fma_rn_f32(first_low, activation_low, first);
            first = float::fma_rn_f32(first_high, activation_high, first);
            second = float::fma_rn_f32(second_low, activation_low, second);
            second = float::fma_rn_f32(second_high, activation_high, second);
            word += 32;
        }

        let first = reduce_sum_lane_zero(first);
        let second = reduce_sum_lane_zero(second);
        if lane == 0 {
            unsafe {
                *shared_output.add(token * HIDDEN + first_row) = tcgen05::f32_to_bf16_rne(first);
                *shared_output.add(token * HIDDEN + second_row) = tcgen05::f32_to_bf16_rne(second);
            }
        }
    }

    /// Executes one decode batch's BF16 shared expert down projection.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_shared_expert_down<const TOKENS: usize>(
        intermediate_input: *const u32,
        down_weight: *const u32,
        shared_output: *mut u16,
    ) {
        unsafe { shared_expert_down::<TOKENS>(intermediate_input, down_weight, shared_output) }
    }

    /// Executes one prompt tile's BF16 shared expert down projection.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_shared_expert_down_prefill<const TOKENS: usize>(
        intermediate_input: *const u32,
        down_weight: *const u32,
        shared_output: *mut u16,
    ) {
        unsafe { shared_expert_down::<TOKENS>(intermediate_input, down_weight, shared_output) }
    }

    /// Weighted sum of the ten routed experts, then the gated shared expert.
    ///
    /// The routed walk is ascending expert index, because the router publishes
    /// its two output planes in that order. The shared expert is added last.
    #[inline(always)]
    unsafe fn expert_combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_output: *const u16,
        shared_gate_logit: *const u16,
        output: *mut u16,
    ) {
        let flat = thread::blockIdx_x() as usize * COMBINE_THREADS as usize
            + thread::threadIdx_x() as usize;
        let token = flat / HIDDEN;
        let column = flat - token * HIDDEN;
        let token_slots = unsafe { expert_output.add(token * TOP_K * HIDDEN) };
        let token_weights = unsafe { routing_weights.add(token * TOP_K) };
        let mut sum = 0.0f32;
        let mut position = 0usize;

        while position < TOP_K {
            let expert_value = f32::from_bits(
                u32::from(unsafe { *token_slots.add(position * HIDDEN + column) }) << 16,
            );
            let routing_weight =
                f32::from_bits(u32::from(unsafe { *token_weights.add(position) }) << 16);
            sum = float::fma_rn_f32(expert_value, routing_weight, sum);
            position += 1;
        }

        let shared_value =
            f32::from_bits(u32::from(unsafe { *shared_output.add(token * HIDDEN + column) }) << 16);
        let logit = f32::from_bits(u32::from(unsafe { *shared_gate_logit.add(token) }) << 16);
        sum = float::fma_rn_f32(shared_value, sigmoid(logit), sum);
        unsafe { *output.add(token * HIDDEN + column) = tcgen05::f32_to_bf16_rne(sum) };
    }

    /// Combines routed and shared outputs for an exact decode batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_combine<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_output: *const u16,
        shared_gate_logit: *const u16,
        output: *mut u16,
    ) {
        unsafe {
            expert_combine::<TOKENS>(
                expert_output,
                routing_weights,
                shared_output,
                shared_gate_logit,
                output,
            )
        }
    }

    /// Combines routed and shared outputs for an exact prompt tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_moe_expert_combine_prefill<const TOKENS: usize>(
        expert_output: *const u16,
        routing_weights: *const u16,
        shared_output: *const u16,
        shared_gate_logit: *const u16,
        output: *mut u16,
    ) {
        unsafe {
            expert_combine::<TOKENS>(
                expert_output,
                routing_weights,
                shared_output,
                shared_gate_logit,
                output,
            )
        }
    }
}

/// Geometry of one sealed Qwen3.8-Flash-Next expert slot pool, and its host gate.
///
/// The kernels trust the indirection table because the streaming publication
/// law guarantees it; this type is where that guarantee is checked. The engine
/// calls it host-side before a round is allowed to replay, and the oracle
/// exercises it directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextSlotPlane {
    slot_count: usize,
}

/// Describes a sealed pool of `slot_count` address-stable expert slots.
pub fn qwen38_flash_next_expert_slot_plane(slot_count: usize) -> Qwen38FlashNextSlotPlane {
    Qwen38FlashNextSlotPlane { slot_count }
}

impl Qwen38FlashNextSlotPlane {
    /// Slots this pool owns.
    pub fn slot_count(self) -> usize {
        self.slot_count
    }

    /// Bytes the sealed slot arena occupies, excluding the table.
    pub fn slot_bytes(self) -> usize {
        self.slot_count * QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES
    }

    /// Checks a published table is structurally well-formed.
    ///
    /// Every entry is `QWEN38_FLASH_NEXT_ABSENT_SLOT` or a slot this pool owns, and no
    /// two experts name the same slot - a shared slot would make one expert
    /// read another's bytes, which no eviction order can excuse.
    pub fn validate_published_table(self, table: &[u32]) -> GpuResult<()> {
        if table.len() != EXPERTS {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next expert indirection table has {} entries, not {EXPERTS}",
                table.len()
            )));
        }

        let mut owner = vec![usize::MAX; self.slot_count];
        for (expert, &slot) in table.iter().enumerate() {
            if slot == QWEN38_FLASH_NEXT_ABSENT_SLOT {
                continue;
            }
            let slot = slot as usize;
            if slot >= self.slot_count {
                return Err(GpuError::invalid_launch(format!(
                    "Qwen3.8-Flash-Next expert {expert} names slot {slot} outside the {} this pool owns",
                    self.slot_count
                )));
            }
            if owner[slot] != usize::MAX {
                return Err(GpuError::invalid_launch(format!(
                    "Qwen3.8-Flash-Next slot {slot} is named by both expert {} and expert {expert}",
                    owner[slot]
                )));
            }
            owner[slot] = expert;
        }

        Ok(())
    }

    /// Checks every expert this round routed to is resident.
    ///
    /// This is the host-side half of "require = stall": a miss the pool could
    /// not satisfy must have stalled before publication, so an `ABSENT_SLOT`
    /// under a routed expert means the round was published early.
    pub fn validate_routed_presence(self, table: &[u32], routed: &[u16]) -> GpuResult<()> {
        self.validate_published_table(table)?;
        for &expert in routed {
            let expert = expert as usize;
            if expert >= EXPERTS {
                return Err(GpuError::invalid_launch(format!(
                    "Qwen3.8-Flash-Next route names expert {expert}, outside 0..{EXPERTS}"
                )));
            }
            if table[expert] == QWEN38_FLASH_NEXT_ABSENT_SLOT {
                return Err(GpuError::invalid_launch(format!(
                    "Qwen3.8-Flash-Next routed expert {expert} has no resident slot; the round was \
                     published before `require` completed"
                )));
            }
        }

        Ok(())
    }
}

fn gate_up_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * TOP_K * INTERMEDIATE / GATE_UP_WARPS) as u32,
        GATE_UP_THREADS,
        0,
    )
}

fn down_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * TOP_K * HIDDEN / DOWN_ROWS_PER_CTA) as u32,
        DOWN_THREADS,
        0,
    )
}

fn shared_gate_up_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * INTERMEDIATE / GATE_UP_WARPS) as u32,
        GATE_UP_THREADS,
        0,
    )
}

fn shared_down_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * HIDDEN / DOWN_ROWS_PER_CTA) as u32,
        DOWN_THREADS,
        0,
    )
}

fn combine_config<const TOKENS: usize>() -> LaunchConfig1D {
    LaunchConfig1D::new(
        (TOKENS * HIDDEN / COMBINE_THREADS as usize) as u32,
        COMBINE_THREADS,
        0,
    )
}

/// Every device pointer one dispatch reads or writes.
///
/// # Safety
///
/// `input` covers `rows * 2_560` BF16 values and `output` the same.
/// `expert_indices` and `routing_weights` cover `rows * 10` values each, paired
/// and ascending by expert index, as [`Qwen38FlashNextMoeRouterOp`] publishes them.
/// `slot_table` covers 512 `u32` entries whose routed experts are all resident
/// as checked by [`Qwen38FlashNextSlotPlane::validate_routed_presence`]. `slot_pool` is the
/// sealed arena's base and stays address-stable for the capture's lifetime.
/// `weight_scales_2` covers `512 * 3` F32 values, gate, up, then down per
/// expert. The scratch planes cover `rows * 10 * 640`, `rows * 10 * 2_560`,
/// `rows * 640`, `rows * 2_560`, and `rows` values. Every plane is aligned for
/// its four-byte loads, mutually disjoint, and lives through stream completion.
///
/// [`Qwen38FlashNextMoeRouterOp`]: crate::Qwen38FlashNextMoeRouterOp
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextExpertDispatch {
    /// BF16 `[rows, 2_560]` residual-stream rows entering the block.
    pub input: *const u16,
    /// Ascending routed expert ids, BF16-paired with `routing_weights`.
    pub expert_indices: *const u16,
    /// Renormalized BF16 routing weights, ascending by expert index.
    pub routing_weights: *const u16,
    /// Device-visible expert id -> slot index table, 512 `u32` entries.
    pub slot_table: *const u32,
    /// Base of the sealed, address-stable slot arena.
    pub slot_pool: *const u8,
    /// Per-expert F32 gate, up, and down `weight_scale_2`.
    pub weight_scales_2: *const f32,
    /// Resident BF16 shared `gate_proj` `[640, 2_560]`.
    pub shared_gate_weight: *const u16,
    /// Resident BF16 shared `up_proj` `[640, 2_560]`.
    pub shared_up_weight: *const u16,
    /// Resident BF16 shared `down_proj` `[2_560, 640]`.
    pub shared_down_weight: *const u16,
    /// Resident BF16 `shared_expert_gate` `[1, 2_560]`.
    pub shared_gate_logit_weight: *const u16,
    /// Scratch BF16 `[rows, 10, 640]` routed SwiGLU intermediates.
    pub routed_intermediate: *mut u16,
    /// Scratch BF16 `[rows, 10, 2_560]` per-expert outputs.
    pub routed_output: *mut u16,
    /// Scratch BF16 `[rows, 640]` shared SwiGLU intermediate.
    pub shared_intermediate: *mut u16,
    /// Scratch BF16 `[rows, 2_560]` shared expert output.
    pub shared_output: *mut u16,
    /// Scratch BF16 `[rows]` shared gate logits.
    pub shared_gate_logit: *mut u16,
    /// BF16 `[rows, 2_560]` block output.
    pub output: *mut u16,
}

struct PreparedBatchRoute<const TOKENS: usize> {
    gate_up: PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_gate_up_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_down_CudaKernel<TOKENS>>,
    shared_gate_up:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_shared_expert_gate_up_CudaKernel<TOKENS>>,
    shared_down:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_shared_expert_down_CudaKernel<TOKENS>>,
    combine: PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_combine_CudaKernel<TOKENS>>,
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    gate_up:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_gate_up_prefill_CudaKernel<TOKENS>>,
    down: PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_down_prefill_CudaKernel<TOKENS>>,
    shared_gate_up: PreparedLaunch<
        kernels::__qwen38_flash_next_moe_shared_expert_gate_up_prefill_CudaKernel<TOKENS>,
    >,
    shared_down: PreparedLaunch<
        kernels::__qwen38_flash_next_moe_shared_expert_down_prefill_CudaKernel<TOKENS>,
    >,
    combine:
        PreparedLaunch<kernels::__qwen38_flash_next_moe_expert_combine_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !(1..=MAX_BATCH).contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE expert decode row count {TOKENS} is not admitted"
            )));
        }
        Ok(Self {
            gate_up: module
                .prepare_qwen38_flash_next_moe_expert_gate_up::<TOKENS>(gate_up_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing the routed gate/up", source))?,
            down: module
                .prepare_qwen38_flash_next_moe_expert_down::<TOKENS>(down_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing the routed down", source))?,
            shared_gate_up: module
                .prepare_qwen38_flash_next_moe_shared_expert_gate_up::<TOKENS>(
                    shared_gate_up_config::<TOKENS>(),
                )
                .map_err(|source| GpuError::launch("preparing the shared gate/up", source))?,
            shared_down: module
                .prepare_qwen38_flash_next_moe_shared_expert_down::<TOKENS>(shared_down_config::<
                    TOKENS,
                >())
                .map_err(|source| GpuError::launch("preparing the shared down", source))?,
            combine: module
                .prepare_qwen38_flash_next_moe_expert_combine::<TOKENS>(combine_config::<TOKENS>())
                .map_err(|source| GpuError::launch("preparing the expert combine", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        dispatch: &Qwen38FlashNextExpertDispatch,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_moe_expert_gate_up::<TOKENS>(
                stream,
                &self.gate_up,
                dispatch.input.cast::<u32>(),
                dispatch.expert_indices,
                dispatch.slot_table,
                dispatch.slot_pool,
                dispatch.weight_scales_2,
                dispatch.routed_intermediate,
            )
            .map_err(|source| GpuError::launch("launching the routed gate/up", source))?;
        module
            .qwen38_flash_next_moe_shared_expert_gate_up::<TOKENS>(
                stream,
                &self.shared_gate_up,
                dispatch.input.cast::<u32>(),
                dispatch.shared_gate_weight.cast::<u32>(),
                dispatch.shared_up_weight.cast::<u32>(),
                dispatch.shared_gate_logit_weight.cast::<u32>(),
                dispatch.shared_intermediate,
                dispatch.shared_gate_logit,
            )
            .map_err(|source| GpuError::launch("launching the shared gate/up", source))?;
        module
            .qwen38_flash_next_moe_expert_down::<TOKENS>(
                stream,
                &self.down,
                dispatch.routed_intermediate.cast_const().cast::<u32>(),
                dispatch.expert_indices,
                dispatch.slot_table,
                dispatch.slot_pool,
                dispatch.weight_scales_2,
                dispatch.routed_output,
            )
            .map_err(|source| GpuError::launch("launching the routed down", source))?;
        module
            .qwen38_flash_next_moe_shared_expert_down::<TOKENS>(
                stream,
                &self.shared_down,
                dispatch.shared_intermediate.cast_const().cast::<u32>(),
                dispatch.shared_down_weight.cast::<u32>(),
                dispatch.shared_output,
            )
            .map_err(|source| GpuError::launch("launching the shared down", source))?;
        module
            .qwen38_flash_next_moe_expert_combine::<TOKENS>(
                stream,
                &self.combine,
                dispatch.routed_output.cast_const(),
                dispatch.routing_weights,
                dispatch.shared_output.cast_const(),
                dispatch.shared_gate_logit.cast_const(),
                dispatch.output,
            )
            .map_err(|source| GpuError::launch("launching the expert combine", source))
    }
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROWS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE expert prefill row count {TOKENS} is not admitted"
            )));
        }
        // T=1024 exposes 819,200 routed gate/up CTAs and 1,638,400 routed down
        // CTAs. Prompt tiles change only the independent token count; every
        // expert dot keeps its decode reduction order, so the tile width can
        // never move a bit.
        Ok(Self {
            gate_up: module
                .prepare_qwen38_flash_next_moe_expert_gate_up_prefill::<TOKENS>(gate_up_config::<
                    TOKENS,
                >())
                .map_err(|source| {
                    GpuError::launch("preparing the prompt routed gate/up", source)
                })?,
            down:
                module
                    .prepare_qwen38_flash_next_moe_expert_down_prefill::<TOKENS>(down_config::<
                        TOKENS,
                    >(
                    ))
                    .map_err(|source| {
                        GpuError::launch("preparing the prompt routed down", source)
                    })?,
            shared_gate_up: module
                .prepare_qwen38_flash_next_moe_shared_expert_gate_up_prefill::<TOKENS>(
                    shared_gate_up_config::<TOKENS>(),
                )
                .map_err(|source| {
                    GpuError::launch("preparing the prompt shared gate/up", source)
                })?,
            shared_down: module
                .prepare_qwen38_flash_next_moe_shared_expert_down_prefill::<TOKENS>(
                    shared_down_config::<TOKENS>(),
                )
                .map_err(|source| GpuError::launch("preparing the prompt shared down", source))?,
            combine: module
                .prepare_qwen38_flash_next_moe_expert_combine_prefill::<TOKENS>(combine_config::<
                    TOKENS,
                >())
                .map_err(|source| {
                    GpuError::launch("preparing the prompt expert combine", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        dispatch: &Qwen38FlashNextExpertDispatch,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_moe_expert_gate_up_prefill::<TOKENS>(
                stream,
                &self.gate_up,
                dispatch.input.cast::<u32>(),
                dispatch.expert_indices,
                dispatch.slot_table,
                dispatch.slot_pool,
                dispatch.weight_scales_2,
                dispatch.routed_intermediate,
            )
            .map_err(|source| GpuError::launch("launching the prompt routed gate/up", source))?;
        module
            .qwen38_flash_next_moe_shared_expert_gate_up_prefill::<TOKENS>(
                stream,
                &self.shared_gate_up,
                dispatch.input.cast::<u32>(),
                dispatch.shared_gate_weight.cast::<u32>(),
                dispatch.shared_up_weight.cast::<u32>(),
                dispatch.shared_gate_logit_weight.cast::<u32>(),
                dispatch.shared_intermediate,
                dispatch.shared_gate_logit,
            )
            .map_err(|source| GpuError::launch("launching the prompt shared gate/up", source))?;
        module
            .qwen38_flash_next_moe_expert_down_prefill::<TOKENS>(
                stream,
                &self.down,
                dispatch.routed_intermediate.cast_const().cast::<u32>(),
                dispatch.expert_indices,
                dispatch.slot_table,
                dispatch.slot_pool,
                dispatch.weight_scales_2,
                dispatch.routed_output,
            )
            .map_err(|source| GpuError::launch("launching the prompt routed down", source))?;
        module
            .qwen38_flash_next_moe_shared_expert_down_prefill::<TOKENS>(
                stream,
                &self.shared_down,
                dispatch.shared_intermediate.cast_const().cast::<u32>(),
                dispatch.shared_down_weight.cast::<u32>(),
                dispatch.shared_output,
            )
            .map_err(|source| GpuError::launch("launching the prompt shared down", source))?;
        module
            .qwen38_flash_next_moe_expert_combine_prefill::<TOKENS>(
                stream,
                &self.combine,
                dispatch.routed_output.cast_const(),
                dispatch.routing_weights,
                dispatch.shared_output.cast_const(),
                dispatch.shared_gate_logit.cast_const(),
                dispatch.output,
            )
            .map_err(|source| GpuError::launch("launching the prompt expert combine", source))
    }
}

/// PTX symbols retained for every exact Qwen3.8-Flash-Next expert route.
pub(crate) fn qwen38_flash_next_moe_experts_ptx_names() -> Vec<&'static str> {
    let mut names = Vec::with_capacity(5 * (MAX_BATCH + PREFILL_ROWS.len()));

    macro_rules! push_decode {
        ($tokens:literal) => {
            names.push(kernels::qwen38_flash_next_moe_expert_gate_up_ptx_name::<
                $tokens,
            >());
            names.push(kernels::qwen38_flash_next_moe_expert_down_ptx_name::<$tokens>());
            names.push(kernels::qwen38_flash_next_moe_shared_expert_gate_up_ptx_name::<$tokens>());
            names.push(kernels::qwen38_flash_next_moe_shared_expert_down_ptx_name::<$tokens>());
            names.push(kernels::qwen38_flash_next_moe_expert_combine_ptx_name::<
                $tokens,
            >());
        };
    }
    macro_rules! push_prefill {
        ($tokens:literal) => {
            names.push(kernels::qwen38_flash_next_moe_expert_gate_up_prefill_ptx_name::<$tokens>());
            names.push(kernels::qwen38_flash_next_moe_expert_down_prefill_ptx_name::<$tokens>());
            names.push(
                kernels::qwen38_flash_next_moe_shared_expert_gate_up_prefill_ptx_name::<$tokens>(),
            );
            names.push(
                kernels::qwen38_flash_next_moe_shared_expert_down_prefill_ptx_name::<$tokens>(),
            );
            names.push(kernels::qwen38_flash_next_moe_expert_combine_prefill_ptx_name::<$tokens>());
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

/// Prepared exact-batch Qwen3.8-Flash-Next slot-indirected expert routes on SM120.
pub struct Qwen38FlashNextMoeExpertsOp {
    module: kernels::LoadedModule,
    b1: PreparedBatchRoute<1>,
    b2: PreparedBatchRoute<2>,
    b3: PreparedBatchRoute<3>,
    b4: PreparedBatchRoute<4>,
    b5: PreparedBatchRoute<5>,
    b6: PreparedBatchRoute<6>,
    b7: PreparedBatchRoute<7>,
    b8: PreparedBatchRoute<8>,
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
    t1024: PreparedPrefillRoute<1024>,
}

impl Qwen38FlashNextMoeExpertsOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen38_flash_next_moe_experts_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next MoE experts", source)
        })?;

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
            t1024: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Runs the routed and shared experts and combines them.
    ///
    /// # Safety
    ///
    /// Every pointer in `dispatch` satisfies the contract on
    /// [`Qwen38FlashNextExpertDispatch`], and the caller has completed the streaming
    /// pool's `require` for every routed expert so no routed expert's table
    /// entry is `QWEN38_FLASH_NEXT_ABSENT_SLOT`.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        dispatch: &Qwen38FlashNextExpertDispatch,
    ) -> GpuResult<()> {
        if !admitted_rows(rows) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE expert row count {rows} is outside the admitted routes \
                 1..={MAX_BATCH},32,64,128,1024"
            )));
        }
        macro_rules! launch {
            ($route:ident) => {
                unsafe { self.$route.launch(&self.module, stream, dispatch) }
            };
        }

        match rows {
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
            1_024 => launch!(t1024),
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next MoE expert row count {rows} is outside the admitted routes \
                 1..={MAX_BATCH},32,64,128,1024"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DOWN_CODE_OFFSET, DOWN_SCALE_BYTES, DOWN_SCALE_OFFSET, EXPERTS, GATE_UP_CODE_OFFSET,
        GATE_UP_SCALE_BYTES, GATE_UP_SCALE_OFFSET, MAX_BATCH, PREFILL_ROWS,
        QWEN38_FLASH_NEXT_ABSENT_SLOT, QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, admitted_rows,
        qwen38_flash_next_expert_slot_plane, qwen38_flash_next_moe_experts_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn slot_extent_matches_the_admitted_source_binding() {
        assert_eq!(DOWN_CODE_OFFSET, 0);
        assert_eq!(GATE_UP_CODE_OFFSET, 819_200);
        assert_eq!(GATE_UP_SCALE_OFFSET, 2_457_600);
        assert_eq!(DOWN_SCALE_OFFSET, 2_662_400);
        assert_eq!(GATE_UP_SCALE_BYTES + DOWN_SCALE_BYTES, 307_200);
        assert_eq!(QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES, 2_764_800);
    }

    #[test]
    fn geometry_and_inventory_are_exact() {
        let names = qwen38_flash_next_moe_experts_ptx_names();
        assert_eq!(names.len(), 5 * (MAX_BATCH + PREFILL_ROWS.len()));
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
            (32, true),
            (33, false),
            (64, true),
            (128, true),
            (129, false),
            (512, false),
            (1_024, true),
            (1_025, false),
        ] {
            assert_eq!(admitted_rows(rows), expected, "rows={rows}");
        }
    }

    #[test]
    fn host_gate_accepts_a_sparse_published_table() {
        let plane = qwen38_flash_next_expert_slot_plane(8);
        let mut table = vec![QWEN38_FLASH_NEXT_ABSENT_SLOT; EXPERTS];
        for (slot, expert) in [3usize, 17, 200, 511].into_iter().enumerate() {
            table[expert] = slot as u32;
        }

        plane.validate_published_table(&table).unwrap();
        plane
            .validate_routed_presence(&table, &[3, 17, 200, 511])
            .unwrap();
    }

    #[test]
    fn host_gate_refuses_absence_under_a_routed_expert() {
        let plane = qwen38_flash_next_expert_slot_plane(8);
        let mut table = vec![QWEN38_FLASH_NEXT_ABSENT_SLOT; EXPERTS];
        table[3] = 0;

        let error = plane
            .validate_routed_presence(&table, &[3, 9])
            .expect_err("expert 9 has no slot");
        assert!(
            format!("{error}").contains("routed expert 9 has no resident slot"),
            "{error}"
        );
    }

    #[test]
    fn host_gate_refuses_a_slot_two_experts_share() {
        let plane = qwen38_flash_next_expert_slot_plane(4);
        let mut table = vec![QWEN38_FLASH_NEXT_ABSENT_SLOT; EXPERTS];
        table[1] = 2;
        table[6] = 2;

        let error = plane
            .validate_published_table(&table)
            .expect_err("slot 2 is claimed twice");
        assert!(
            format!("{error}").contains("named by both expert 1"),
            "{error}"
        );
    }

    #[test]
    fn host_gate_refuses_a_slot_outside_the_pool() {
        let plane = qwen38_flash_next_expert_slot_plane(4);
        let mut table = vec![QWEN38_FLASH_NEXT_ABSENT_SLOT; EXPERTS];
        table[5] = 4;

        let error = plane
            .validate_published_table(&table)
            .expect_err("slot 4 is outside a four-slot pool");
        assert!(format!("{error}").contains("outside the 4"), "{error}");
    }

    #[test]
    fn host_gate_refuses_a_table_of_the_wrong_width() {
        let plane = qwen38_flash_next_expert_slot_plane(4);
        let error = plane
            .validate_published_table(&[0; 8])
            .expect_err("the table must name every expert");
        assert!(format!("{error}").contains("not 512"), "{error}");
    }
}

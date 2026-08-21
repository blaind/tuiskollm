use crate::device::fp8_projection::{e4m3x2_to_f32, load_u32x4_read_only, reduce_sum_lane_zero};
use cuda_device::async_copy::{
    cp_async_ca_16, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{DynamicSharedArray, float, tcgen05, thread};
use tuisko_model::Arch;

const VALUES_PER_LANE: usize = 16;
const VALUES_PER_PHASE: usize = 32 * VALUES_PER_LANE;
const WORDS_PER_LANE: usize = VALUES_PER_LANE / 4;

#[inline(always)]
fn silu(value: f32) -> f32 {
    value / (1.0 + float::ex2_approx_f32(-value * core::f32::consts::LOG2_E))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_b1_phase<A: Arch, const PHASE: usize>(
    activation_codes: *const u32,
    weight_codes: *const u32,
    gate_row: usize,
    up_row: usize,
    lane: usize,
    gate: &mut [f32; 4],
    up: &mut [f32; 4],
) {
    let words_per_row = A::HIDDEN / 4;
    let lane_offset = PHASE * (VALUES_PER_PHASE / 4) + lane * WORDS_PER_LANE;
    // SAFETY: the exact row and phase geometry covers all three 16-byte fragments.
    let activation = unsafe { load_u32x4_read_only(activation_codes.add(lane_offset)) };
    let gate_words =
        unsafe { load_u32x4_read_only(weight_codes.add(gate_row * words_per_row + lane_offset)) };
    let up_words =
        unsafe { load_u32x4_read_only(weight_codes.add(up_row * words_per_row + lane_offset)) };
    let activation_words = [activation.0, activation.1, activation.2, activation.3];
    let gate_words = [gate_words.0, gate_words.1, gate_words.2, gate_words.3];
    let up_words = [up_words.0, up_words.1, up_words.2, up_words.3];

    macro_rules! accumulate_pair {
        ($pair:literal, $chain0:literal, $chain1:literal) => {{
            let shift = ($pair & 1) * 16;
            let activation_packed = (activation_words[$pair >> 1] >> shift) as u16;
            let gate_packed = (gate_words[$pair >> 1] >> shift) as u16;
            let up_packed = (up_words[$pair >> 1] >> shift) as u16;
            let (activation0, activation1) = e4m3x2_to_f32(activation_packed);
            let (gate0, gate1) = e4m3x2_to_f32(gate_packed);
            let (up0, up1) = e4m3x2_to_f32(up_packed);
            gate[$chain0] = float::fma_rn_f32(gate0, activation0, gate[$chain0]);
            gate[$chain1] = float::fma_rn_f32(gate1, activation1, gate[$chain1]);
            up[$chain0] = float::fma_rn_f32(up0, activation0, up[$chain0]);
            up[$chain1] = float::fma_rn_f32(up1, activation1, up[$chain1]);
        }};
    }

    accumulate_pair!(0, 0, 1);
    accumulate_pair!(1, 2, 3);
    accumulate_pair!(2, 0, 1);
    accumulate_pair!(3, 2, 3);
    accumulate_pair!(4, 0, 1);
    accumulate_pair!(5, 2, 3);
    accumulate_pair!(6, 0, 1);
    accumulate_pair!(7, 2, 3);
}

#[inline(always)]
pub(crate) unsafe fn fp8_swiglu_decode_b1<A: Arch, const WARPS: usize>(
    activation_codes: *const u32,
    activation_scale: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
) {
    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let m_tile = block >> 4;
    let cta_in_tile = block & 15;
    let flat_pair = cta_in_tile * WARPS + warp_index;
    let row_mod32 = flat_pair >> 2;
    let quartile = flat_pair & 3;
    let gate_row = m_tile * 128 + row_mod32 + quartile * 32;
    let up_row = gate_row + A::INTERMEDIATE;
    let mut gate = [0.0f32; 4];
    let mut up = [0.0f32; 4];

    // SAFETY: ten phases cover the exact 5,120-wide target row once.
    unsafe {
        accumulate_b1_phase::<A, 0>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 1>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 2>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 3>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 4>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 5>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 6>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 7>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 8>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
        accumulate_b1_phase::<A, 9>(
            activation_codes,
            weight_codes,
            gate_row,
            up_row,
            lane,
            &mut gate,
            &mut up,
        );
    }

    let gate_sum = gate[0] + gate[1] + gate[2] + gate[3];
    let up_sum = up[0] + up[1] + up[2] + up[3];
    // SAFETY: both source rows and the sole activation scale are admitted.
    let activation_scale = unsafe { *activation_scale };
    let gate_scale = f32::from_bits((unsafe { *weight_scales.add(gate_row) } as u32) << 16);
    let up_scale = f32::from_bits((unsafe { *weight_scales.add(up_row) } as u32) << 16);
    let gate_value = reduce_sum_lane_zero(gate_sum) * activation_scale * gate_scale;
    let up_value = reduce_sum_lane_zero(up_sum) * activation_scale * up_scale;
    if lane == 0 {
        // SAFETY: lane zero uniquely owns this swizzled output row.
        unsafe {
            *output.add(gate_row) =
                tcgen05::cvt_f32x2_bf16x2(silu(gate_value) * up_value, 0.0) as u16;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fp8_swiglu_decode<A: Arch, const TOKENS: usize, const WARPS: usize>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * WARPS + (tid >> 5);
    let words_per_row = A::HIDDEN / 4;
    let output_rows = A::INTERMEDIATE;
    // SAFETY: the exact grid assigns this warp one gate/up source-row pair.
    let gate_weight = unsafe { weight_codes.add(row * words_per_row) };
    // SAFETY: the source-native fused plane stores all up rows after all gate rows.
    let up_weight = unsafe { weight_codes.add((row + output_rows) * words_per_row) };
    let mut gate = [0.0f32; TOKENS];
    let mut up = [0.0f32; TOKENS];
    let mut phase = 0usize;

    while phase < A::HIDDEN / VALUES_PER_PHASE {
        let lane_offset = phase * (VALUES_PER_PHASE / 4) + lane * WORDS_PER_LANE;
        // SAFETY: each phase/lane fragment is aligned and contained in its source row.
        let gate_words = unsafe { load_u32x4_read_only(gate_weight.add(lane_offset)) };
        // SAFETY: each phase/lane fragment is aligned and contained in its source row.
        let up_words = unsafe { load_u32x4_read_only(up_weight.add(lane_offset)) };
        // SAFETY: every const route admits the referenced complete activation row.
        let activation0 = unsafe { load_u32x4_read_only(activation_codes.add(lane_offset)) };
        let activation1 = if TOKENS > 1 {
            unsafe { load_u32x4_read_only(activation_codes.add(words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation2 = if TOKENS > 2 {
            unsafe { load_u32x4_read_only(activation_codes.add(2 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation3 = if TOKENS > 3 {
            unsafe { load_u32x4_read_only(activation_codes.add(3 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation4 = if TOKENS > 4 {
            unsafe { load_u32x4_read_only(activation_codes.add(4 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation5 = if TOKENS > 5 {
            unsafe { load_u32x4_read_only(activation_codes.add(5 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation6 = if TOKENS > 6 {
            unsafe { load_u32x4_read_only(activation_codes.add(6 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };
        let activation7 = if TOKENS > 7 {
            unsafe { load_u32x4_read_only(activation_codes.add(7 * words_per_row + lane_offset)) }
        } else {
            (0, 0, 0, 0)
        };

        macro_rules! packed_pair {
            ($words:ident, $pair:literal) => {{
                let word = match $pair >> 1 {
                    0 => $words.0,
                    1 => $words.1,
                    2 => $words.2,
                    _ => $words.3,
                };
                (word >> (($pair & 1) * 16)) as u16
            }};
        }

        macro_rules! accumulate_token {
            ($token:literal, $words:ident, $pair:literal, $gate_values:ident, $up_values:ident) => {
                if TOKENS > $token {
                    let (low, high) = e4m3x2_to_f32(packed_pair!($words, $pair));
                    gate[$token] = float::fma_rn_f32($gate_values.0, low, gate[$token]);
                    gate[$token] = float::fma_rn_f32($gate_values.1, high, gate[$token]);
                    up[$token] = float::fma_rn_f32($up_values.0, low, up[$token]);
                    up[$token] = float::fma_rn_f32($up_values.1, high, up[$token]);
                }
            };
        }

        macro_rules! accumulate_pair {
            ($pair:literal) => {{
                let gate_values = e4m3x2_to_f32(packed_pair!(gate_words, $pair));
                let up_values = e4m3x2_to_f32(packed_pair!(up_words, $pair));
                accumulate_token!(0, activation0, $pair, gate_values, up_values);
                accumulate_token!(1, activation1, $pair, gate_values, up_values);
                accumulate_token!(2, activation2, $pair, gate_values, up_values);
                accumulate_token!(3, activation3, $pair, gate_values, up_values);
                accumulate_token!(4, activation4, $pair, gate_values, up_values);
                accumulate_token!(5, activation5, $pair, gate_values, up_values);
                accumulate_token!(6, activation6, $pair, gate_values, up_values);
                accumulate_token!(7, activation7, $pair, gate_values, up_values);
            }};
        }

        accumulate_pair!(0);
        accumulate_pair!(1);
        accumulate_pair!(2);
        accumulate_pair!(3);
        accumulate_pair!(4);
        accumulate_pair!(5);
        accumulate_pair!(6);
        accumulate_pair!(7);
        phase += 1;
    }

    // SAFETY: both scale rows exist in the admitted fused gate/up scale plane.
    let gate_scale = f32::from_bits((unsafe { *weight_scales.add(row) } as u32) << 16);
    let up_scale = f32::from_bits((unsafe { *weight_scales.add(row + output_rows) } as u32) << 16);

    macro_rules! store_token {
        ($token:literal) => {
            if TOKENS > $token {
                // SAFETY: the exact route owns this activation scale.
                let activation_scale = unsafe { *activation_scales.add($token) };
                let gate_value = reduce_sum_lane_zero(gate[$token]) * activation_scale * gate_scale;
                let up_value = reduce_sum_lane_zero(up[$token]) * activation_scale * up_scale;
                if lane == 0 {
                    // SAFETY: lane zero writes this warp's unique output row.
                    unsafe {
                        *output.add($token * output_rows + row) =
                            tcgen05::cvt_f32x2_bf16x2(silu(gate_value) * up_value, 0.0) as u16;
                    }
                }
            }
        };
    }

    store_token!(0);
    store_token!(1);
    store_token!(2);
    store_token!(3);
    store_token!(4);
    store_token!(5);
    store_token!(6);
    store_token!(7);
}

#[inline(always)]
unsafe fn store_mma_tile<A: Arch, const TOKENS: usize>(
    gate: [f32; 4],
    up: [f32; 4],
    activation_scales: *const f32,
    weight_scales: *const u16,
    output: *mut u16,
    first_token: usize,
    first_output: usize,
) {
    if first_token >= TOKENS || first_output + 1 >= A::INTERMEDIATE {
        return;
    }
    // SAFETY: the tile maps two adjacent gate/up rows and two token rows.
    unsafe {
        let gate_scale0 = f32::from_bits((*weight_scales.add(first_output) as u32) << 16);
        let gate_scale1 = f32::from_bits((*weight_scales.add(first_output + 1) as u32) << 16);
        let up_scale0 =
            f32::from_bits((*weight_scales.add(first_output + A::INTERMEDIATE) as u32) << 16);
        let up_scale1 =
            f32::from_bits((*weight_scales.add(first_output + A::INTERMEDIATE + 1) as u32) << 16);
        let activation_scale = *activation_scales.add(first_token);
        *output.add(first_token * A::INTERMEDIATE + first_output) = tcgen05::cvt_f32x2_bf16x2(
            silu(gate[0] * activation_scale * gate_scale0) * (up[0] * activation_scale * up_scale0),
            0.0,
        ) as u16;
        *output.add(first_token * A::INTERMEDIATE + first_output + 1) = tcgen05::cvt_f32x2_bf16x2(
            silu(gate[1] * activation_scale * gate_scale1) * (up[1] * activation_scale * up_scale1),
            0.0,
        ) as u16;

        let second_token = first_token + 8;
        if second_token < TOKENS {
            let activation_scale = *activation_scales.add(second_token);
            *output.add(second_token * A::INTERMEDIATE + first_output) = tcgen05::cvt_f32x2_bf16x2(
                silu(gate[2] * activation_scale * gate_scale0)
                    * (up[2] * activation_scale * up_scale0),
                0.0,
            ) as u16;
            *output.add(second_token * A::INTERMEDIATE + first_output + 1) =
                tcgen05::cvt_f32x2_bf16x2(
                    silu(gate[3] * activation_scale * gate_scale1)
                        * (up[3] * activation_scale * up_scale1),
                    0.0,
                ) as u16;
        }
    }
}

/// Applies the retained 64-output FP8 MMA tile to an exact prefill row count.
#[inline(always)]
pub(crate) unsafe fn fp8_swiglu_mma<
    A: Arch,
    const TOKENS: usize,
    const BLOCK_ROWS: usize,
    const K_WORDS: usize,
    const K_SUBTILES: usize,
>(
    activation_codes: *const u32,
    activation_scales: *const f32,
    weight_codes: *const u32,
    weight_scales: *const u16,
    output: *mut u16,
    k_tiles: u32,
) {
    const OUTPUT_ROWS_PER_BLOCK: usize = 64;
    const STAGES: usize = 2;

    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let lane_group = lane >> 2;
    let thread_in_group = lane & 3;
    let words_per_row = A::HIDDEN / 4;
    let output_tiles = A::INTERMEDIATE / OUTPUT_ROWS_PER_BLOCK;
    let token_tile = block / output_tiles;
    let output_tile = block % output_tiles;
    let first_token_tile = token_tile * BLOCK_ROWS;
    let first_output_tile = output_tile * OUTPUT_ROWS_PER_BLOCK;
    let warp_token = first_token_tile + (warp_index >> 1) * 16;
    let warp_output = first_output_tile + (warp_index & 1) * 32;
    let activation_tile = DynamicSharedArray::<u32, 16>::get();
    // SAFETY: the launch reserves both activation stages before the gate stages.
    let gate_tile = unsafe { activation_tile.add(STAGES * BLOCK_ROWS * K_WORDS) };
    // SAFETY: the launch reserves both gate stages before the up stages.
    let up_tile = unsafe { gate_tile.add(STAGES * OUTPUT_ROWS_PER_BLOCK * K_WORDS) };
    let mut gate0 = [0.0f32; 4];
    let mut gate1 = [0.0f32; 4];
    let mut gate2 = [0.0f32; 4];
    let mut gate3 = [0.0f32; 4];
    let mut up0 = [0.0f32; 4];
    let mut up1 = [0.0f32; 4];
    let mut up2 = [0.0f32; 4];
    let mut up3 = [0.0f32; 4];

    debug_assert_eq!(thread::blockDim_x() as usize, (BLOCK_ROWS / 16) * 2 * 32);

    // SAFETY: the fixed grid and shared-memory contract cover the first K stage.
    unsafe {
        let chunks_per_row = K_WORDS / 4;
        let mut task = tid;
        while task < BLOCK_ROWS * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_ca_16(
                activation_tile.add(row * K_WORDS + word),
                activation_codes.add((first_token_tile + row) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        task = tid;
        while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_cg_16(
                gate_tile.add(row * K_WORDS + word),
                weight_codes.add((first_output_tile + row) * words_per_row + word),
            );
            cp_async_cg_16(
                up_tile.add(row * K_WORDS + word),
                weight_codes
                    .add((first_output_tile + row + A::INTERMEDIATE) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        cp_async_commit_group();
    }

    let mut stage = 0usize;
    let mut k_tile = 0usize;
    while k_tile < k_tiles as usize {
        // SAFETY: each next-stage transaction remains in the admitted planes and shared arena.
        unsafe {
            if k_tile + 1 < k_tiles as usize {
                let next_stage = stage ^ 1;
                let next_word = (k_tile + 1) * K_WORDS;
                let chunks_per_row = K_WORDS / 4;
                let mut task = tid;
                while task < BLOCK_ROWS * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_ca_16(
                        activation_tile
                            .add(next_stage * BLOCK_ROWS * K_WORDS + row * K_WORDS + word),
                        activation_codes
                            .add((first_token_tile + row) * words_per_row + next_word + word),
                    );
                    task += thread::blockDim_x() as usize;
                }
                task = tid;
                while task < OUTPUT_ROWS_PER_BLOCK * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_cg_16(
                        gate_tile.add(
                            next_stage * OUTPUT_ROWS_PER_BLOCK * K_WORDS + row * K_WORDS + word,
                        ),
                        weight_codes
                            .add((first_output_tile + row) * words_per_row + next_word + word),
                    );
                    cp_async_cg_16(
                        up_tile.add(
                            next_stage * OUTPUT_ROWS_PER_BLOCK * K_WORDS + row * K_WORDS + word,
                        ),
                        weight_codes.add(
                            (first_output_tile + row + A::INTERMEDIATE) * words_per_row
                                + next_word
                                + word,
                        ),
                    );
                    task += thread::blockDim_x() as usize;
                }
                cp_async_commit_group();
                cp_async_wait_group(1);
            } else {
                cp_async_wait_group(0);
            }
        }
        thread::sync_threads();

        let activation_base = stage * BLOCK_ROWS * K_WORDS + (warp_index >> 1) * 16 * K_WORDS;
        let first_weight_base =
            stage * OUTPUT_ROWS_PER_BLOCK * K_WORDS + (warp_index & 1) * 32 * K_WORDS;
        let weight0 = first_weight_base;
        let weight1 = weight0 + 8 * K_WORDS;
        let weight2 = weight1 + 8 * K_WORDS;
        let weight3 = weight2 + 8 * K_WORDS;
        let mut k_subtile = 0usize;

        while k_subtile < K_SUBTILES {
            let k_offset = k_subtile * 8;
            // SAFETY: every lane group loads the fragments required by m16n8k32.
            let activation_fragment = unsafe {
                [
                    *activation_tile
                        .add(activation_base + lane_group * K_WORDS + k_offset + thread_in_group),
                    *activation_tile.add(
                        activation_base + (lane_group + 8) * K_WORDS + k_offset + thread_in_group,
                    ),
                    *activation_tile.add(
                        activation_base + lane_group * K_WORDS + k_offset + thread_in_group + 4,
                    ),
                    *activation_tile.add(
                        activation_base
                            + (lane_group + 8) * K_WORDS
                            + k_offset
                            + thread_in_group
                            + 4,
                    ),
                ]
            };
            macro_rules! fragment {
                ($tile:ident, $base:ident) => {
                    [
                        *$tile.add($base + lane_group * K_WORDS + k_offset + thread_in_group),
                        *$tile.add($base + lane_group * K_WORDS + k_offset + thread_in_group + 4),
                    ]
                };
            }
            // SAFETY: each fragment is a complete E4M3 m16n8k32 operand tile.
            unsafe {
                gate0 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    gate0,
                    activation_fragment,
                    fragment!(gate_tile, weight0),
                );
                gate1 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    gate1,
                    activation_fragment,
                    fragment!(gate_tile, weight1),
                );
                gate2 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    gate2,
                    activation_fragment,
                    fragment!(gate_tile, weight2),
                );
                gate3 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    gate3,
                    activation_fragment,
                    fragment!(gate_tile, weight3),
                );
                up0 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    up0,
                    activation_fragment,
                    fragment!(up_tile, weight0),
                );
                up1 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    up1,
                    activation_fragment,
                    fragment!(up_tile, weight1),
                );
                up2 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    up2,
                    activation_fragment,
                    fragment!(up_tile, weight2),
                );
                up3 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    up3,
                    activation_fragment,
                    fragment!(up_tile, weight3),
                );
            }
            k_subtile += 1;
        }
        thread::sync_threads();
        stage ^= 1;
        k_tile += 1;
    }

    let first_token = warp_token + lane_group;
    let first_output = warp_output + thread_in_group * 2;
    // SAFETY: the warp partition gives every output pair a unique writer.
    unsafe {
        store_mma_tile::<A, TOKENS>(
            gate0,
            up0,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output,
        );
        store_mma_tile::<A, TOKENS>(
            gate1,
            up1,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 8,
        );
        store_mma_tile::<A, TOKENS>(
            gate2,
            up2,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 16,
        );
        store_mma_tile::<A, TOKENS>(
            gate3,
            up3,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 24,
        );
    }
}

use cuda_device::async_copy::{
    cp_async_ca_16, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{DynamicSharedArray, convert, float, tcgen05, thread, warp};
use tuisko_model::Arch;

const WARPS: usize = 8;
const THREADS: usize = WARPS * 32;
const FP8_MAX: f32 = 448.0;

#[inline(always)]
fn bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
unsafe fn gated_value<A: Arch>(attention: *const f32, qkv: *const u16, index: usize) -> f32 {
    let token = thread::blockIdx_x() as usize;
    let head = index / A::HEAD_DIM;
    let dimension = index - head * A::HEAD_DIM;
    let gate = unsafe {
        *qkv.add(token * A::ATTENTION_QKV_ROWS + head * (2 * A::HEAD_DIM) + A::HEAD_DIM + dimension)
    };
    let gate = bf16_bits(gate);
    let sigmoid = 1.0 / (1.0 + float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E));

    (unsafe { *attention.add(token * A::ATTENTION_OUTPUT_COLUMNS + index) }) * sigmoid
}

#[inline(always)]
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[inline(always)]
pub(crate) unsafe fn attention_gate_bf16<A: Arch>(
    attention: *mut f32,
    qkv: *const u16,
    activation: *mut u16,
) {
    let token = thread::blockIdx_x() as usize;
    let mut index = thread::threadIdx_x() as usize;

    while index < A::ATTENTION_OUTPUT_COLUMNS {
        let gated = unsafe { gated_value::<A>(attention, qkv, index) };
        let offset = token * A::ATTENTION_OUTPUT_COLUMNS + index;
        unsafe {
            *attention.add(offset) = gated;
            *activation.add(offset) = f32_to_bf16(gated);
        }
        index += THREADS;
    }
}

#[inline(always)]
pub(crate) unsafe fn attention_gate_quantize<A: Arch>(
    attention: *mut f32,
    qkv: *const u16,
    codes: *mut u16,
    scales: *mut f32,
    warp_maximum: *mut f32,
) {
    let token = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let mut maximum = 0.0f32;
    let mut index = tid;

    // The paged-attention output is scratch after this boundary. Publishing
    // the gated FP32 seam avoids a second sigmoid and keeps it observable.
    while index < A::ATTENTION_OUTPUT_COLUMNS {
        let gated = unsafe { gated_value::<A>(attention, qkv, index) };
        unsafe { *attention.add(token * A::ATTENTION_OUTPUT_COLUMNS + index) = gated };
        maximum = maximum.max(gated.abs());
        index += THREADS;
    }
    maximum = warp::reduce_max_f32(maximum);
    if lane == 0 {
        unsafe { *warp_maximum.add(warp_index) = maximum };
    }
    thread::sync_threads();

    if warp_index == 0 {
        maximum = if lane < WARPS {
            unsafe { *warp_maximum.add(lane) }
        } else {
            0.0
        };
        maximum = warp::reduce_max_f32(maximum);
        if lane == 0 {
            unsafe {
                *scales.add(token) = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
            }
        }
    }
    thread::sync_threads();

    let inverse_scale = 1.0 / unsafe { *scales.add(token) };
    let mut pair = tid;
    while pair < A::ATTENTION_OUTPUT_COLUMNS / 2 {
        let base = token * A::ATTENTION_OUTPUT_COLUMNS + pair * 2;
        let low = unsafe { *attention.add(base) };
        let high = unsafe { *attention.add(base + 1) };
        unsafe {
            *codes.add(token * A::ATTENTION_OUTPUT_COLUMNS / 2 + pair) =
                convert::cvt_rn_satfinite_e4m3x2_f32(low * inverse_scale, high * inverse_scale);
        }
        pair += THREADS;
    }
}

#[inline(always)]
unsafe fn store_attention_output_mma_tile<A: Arch, const TOKENS: usize>(
    values: [f32; 4],
    activation_scales: *const f32,
    weight_scales: *const u16,
    output: *mut u16,
    first_token: usize,
    first_output: usize,
) {
    if first_token >= TOKENS || first_output + 1 >= A::HIDDEN {
        return;
    }

    // SAFETY: the exact tile maps two adjacent source rows and one active token row.
    unsafe {
        let first_weight_scale = f32::from_bits((*weight_scales.add(first_output) as u32) << 16);
        let second_weight_scale =
            f32::from_bits((*weight_scales.add(first_output + 1) as u32) << 16);
        let activation_scale = *activation_scales.add(first_token);
        *output.add(first_token * A::HIDDEN + first_output) =
            tcgen05::cvt_f32x2_bf16x2(values[0] * activation_scale * first_weight_scale, 0.0)
                as u16;
        *output.add(first_token * A::HIDDEN + first_output + 1) =
            tcgen05::cvt_f32x2_bf16x2(values[1] * activation_scale * second_weight_scale, 0.0)
                as u16;

        let second_token = first_token + 8;
        if second_token < TOKENS {
            let activation_scale = *activation_scales.add(second_token);
            *output.add(second_token * A::HIDDEN + first_output) =
                tcgen05::cvt_f32x2_bf16x2(values[2] * activation_scale * first_weight_scale, 0.0)
                    as u16;
            *output.add(second_token * A::HIDDEN + first_output + 1) =
                tcgen05::cvt_f32x2_bf16x2(values[3] * activation_scale * second_weight_scale, 0.0)
                    as u16;
        }
    }
}

/// Projects one exact prefill width through the source-native attention-output matrix.
#[inline(always)]
pub(crate) unsafe fn attention_output_projection_mma<
    A: Arch,
    const TOKENS: usize,
    const BLOCK_ROWS: usize,
    const OUTPUT_ROWS: usize,
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
    const STAGES: usize = 2;

    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let lane_group = lane >> 2;
    let thread_in_group = lane & 3;
    let words_per_row = A::ATTENTION_OUTPUT_COLUMNS / 4;
    let output_tiles = A::HIDDEN / OUTPUT_ROWS;
    let token_tile = block / output_tiles;
    let output_tile = block - token_tile * output_tiles;
    let first_token_tile = token_tile * BLOCK_ROWS;
    let first_output_tile = output_tile * OUTPUT_ROWS;
    let output_warps = OUTPUT_ROWS / 32;
    let warp_token = first_token_tile + (warp_index / output_warps) * 16;
    let warp_output = first_output_tile + (warp_index % output_warps) * 32;
    let activation_tile = DynamicSharedArray::<u32, 16>::get();
    // SAFETY: the launch reserves both activation stages before both weight stages.
    let weight_tile = unsafe { activation_tile.add(STAGES * BLOCK_ROWS * K_WORDS) };
    let mut accumulator0 = [0.0f32; 4];
    let mut accumulator1 = [0.0f32; 4];
    let mut accumulator2 = [0.0f32; 4];
    let mut accumulator3 = [0.0f32; 4];

    debug_assert_eq!(
        thread::blockDim_x() as usize,
        (BLOCK_ROWS / 16) * output_warps * 32
    );

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
        while task < OUTPUT_ROWS * chunks_per_row {
            let row = task / chunks_per_row;
            let word = (task - row * chunks_per_row) * 4;
            cp_async_cg_16(
                weight_tile.add(row * K_WORDS + word),
                weight_codes.add((first_output_tile + row) * words_per_row + word),
            );
            task += thread::blockDim_x() as usize;
        }
        cp_async_commit_group();
    }

    let mut stage = 0usize;
    let mut k_tile = 0usize;
    while k_tile < k_tiles as usize {
        // SAFETY: every next-stage transaction remains inside the admitted planes and arena.
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
                while task < OUTPUT_ROWS * chunks_per_row {
                    let row = task / chunks_per_row;
                    let word = (task - row * chunks_per_row) * 4;
                    cp_async_cg_16(
                        weight_tile.add(next_stage * OUTPUT_ROWS * K_WORDS + row * K_WORDS + word),
                        weight_codes
                            .add((first_output_tile + row) * words_per_row + next_word + word),
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

        let activation_base =
            stage * BLOCK_ROWS * K_WORDS + (warp_index / output_warps) * 16 * K_WORDS;
        let first_weight_base =
            stage * OUTPUT_ROWS * K_WORDS + (warp_index % output_warps) * 32 * K_WORDS;
        let weight0 = first_weight_base;
        let weight1 = weight0 + 8 * K_WORDS;
        let weight2 = weight1 + 8 * K_WORDS;
        let weight3 = weight2 + 8 * K_WORDS;
        let mut k_subtile = 0usize;

        while k_subtile < K_SUBTILES {
            let k_offset = k_subtile * 8;
            // SAFETY: each lane group loads the complete fragments required by m16n8k32.
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
            macro_rules! weight_fragment {
                ($base:ident) => {
                    [
                        *weight_tile.add($base + lane_group * K_WORDS + k_offset + thread_in_group),
                        *weight_tile
                            .add($base + lane_group * K_WORDS + k_offset + thread_in_group + 4),
                    ]
                };
            }
            // SAFETY: each fragment is one native E4M3 m16n8k32 operand tile.
            unsafe {
                accumulator0 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator0,
                    activation_fragment,
                    weight_fragment!(weight0),
                );
                accumulator1 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator1,
                    activation_fragment,
                    weight_fragment!(weight1),
                );
                accumulator2 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator2,
                    activation_fragment,
                    weight_fragment!(weight2),
                );
                accumulator3 = cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    accumulator3,
                    activation_fragment,
                    weight_fragment!(weight3),
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
    // SAFETY: the warp partition gives each active output pair one unique writer.
    unsafe {
        store_attention_output_mma_tile::<A, TOKENS>(
            accumulator0,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output,
        );
        store_attention_output_mma_tile::<A, TOKENS>(
            accumulator1,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 8,
        );
        store_attention_output_mma_tile::<A, TOKENS>(
            accumulator2,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 16,
        );
        store_attention_output_mma_tile::<A, TOKENS>(
            accumulator3,
            activation_scales,
            weight_scales,
            output,
            first_token,
            first_output + 24,
        );
    }
}

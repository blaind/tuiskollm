use cuda_device::async_copy::{
    cp_async_ca_4, cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::{SharedArray, convert, float, ptx_asm, tcgen05, thread, wmma};

pub(crate) const GROUP_K: usize = 16;
pub(crate) const BLOCK_N: usize = 64;
pub(crate) const TILE_M: usize = 48;
pub(crate) const THREADS: u32 = 384;

const BLOCK_M: usize = 64;
const BLOCK_K: usize = 256;
const WARPS_N: usize = 4;
const WARP_M: usize = 16;
const WARP_N: usize = BLOCK_N / WARPS_N;
const MMA_N: usize = WARP_N / 8;
const STAGES: usize = 2;
const K64_PER_STAGE: usize = BLOCK_K / 64;
const CODE_ROW_BYTES: usize = BLOCK_K / 2;
const SEGMENTS_PER_ROW: usize = CODE_ROW_BYTES / 16;
const A_CODE_BYTES: usize = STAGES * BLOCK_M * CODE_ROW_BYTES;
const B_CODE_BYTES: usize = STAGES * BLOCK_N * CODE_ROW_BYTES;
const A_SCALE_BYTES: usize = STAGES * BLOCK_M * K64_PER_STAGE * 4;
const B_SCALE_BYTES: usize = STAGES * BLOCK_N * K64_PER_STAGE * 4;
const A_CODE_OFFSET: usize = 0;
const B_CODE_OFFSET: usize = A_CODE_OFFSET + A_CODE_BYTES;
const A_SCALE_OFFSET: usize = B_CODE_OFFSET + B_CODE_BYTES;
const B_SCALE_OFFSET: usize = A_SCALE_OFFSET + A_SCALE_BYTES;
pub(crate) const SHARED_BYTES: usize = B_SCALE_OFFSET + B_SCALE_BYTES;
const SHARED_U32: usize = SHARED_BYTES / 4;

const _: () = assert!(THREADS as usize == (TILE_M / WARP_M) * WARPS_N * 32);
const _: () = assert!(SHARED_BYTES == 36_864);

#[inline(always)]
unsafe fn load_u32x4(source: *const u32) -> (u32, u32, u32, u32) {
    let first: u32;
    let second: u32;
    let third: u32;
    let fourth: u32;

    unsafe {
        ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") first,
            out("=r") second,
            out("=r") third,
            out("=r") fourth,
            in("l") source,
            clobber("memory"),
        );
    }

    (first, second, third, fourth)
}

#[inline(always)]
fn e4m3_to_f32(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;

    if exponent == 0 {
        fraction as f32 * (1.0 / 512.0)
    } else {
        f32::from_bits(((exponent as u32 + 120) << 23) | ((fraction as u32) << 20))
    }
}

#[inline(always)]
fn accumulate_max_abs(maximum: f32, value: f32) -> f32 {
    let mut result = maximum;

    unsafe {
        ptx_asm!(
            "{ .reg .f32 absolute; abs.f32 absolute, %1; max.f32 %0, %0, absolute; }",
            inout("+f") result,
            in("f") value,
            options(register_only),
        );
    }

    result
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pack_e2m1x16(
    v0: f32,
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
    v5: f32,
    v6: f32,
    v7: f32,
    v8: f32,
    v9: f32,
    v10: f32,
    v11: f32,
    v12: f32,
    v13: f32,
    v14: f32,
    v15: f32,
) -> (u32, u32) {
    let codes_lo: u32;
    let codes_hi: u32;

    unsafe {
        ptx_asm!(
            "{ .reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3; \
               .reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7; \
               cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2; \
               cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4; \
               cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6; \
               cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8; \
               cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10; \
               cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12; \
               cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14; \
               cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16; \
               mov.b32 %0, {b0, b1, b2, b3}; \
               mov.b32 %1, {b4, b5, b6, b7}; }",
            out("=r") codes_lo,
            out("=r") codes_hi,
            in("f") v0,
            in("f") v1,
            in("f") v2,
            in("f") v3,
            in("f") v4,
            in("f") v5,
            in("f") v6,
            in("f") v7,
            in("f") v8,
            in("f") v9,
            in("f") v10,
            in("f") v11,
            in("f") v12,
            in("f") v13,
            in("f") v14,
            in("f") v15,
            options(register_only),
        );
    }

    (codes_lo, codes_hi)
}

#[inline(always)]
pub(crate) unsafe fn quantize_bf16_rows<const INPUT_COLUMNS: usize, const TOKENS: usize>(
    task: usize,
    input: *const u32,
    codes: *mut u32,
    scales: *mut u8,
    input_scale_divisor: f32,
) {
    let groups_per_row = INPUT_COLUMNS / GROUP_K;
    if task >= TOKENS * groups_per_row {
        return;
    }

    let token = task / groups_per_row;
    let group = task - token * groups_per_row;
    let source = unsafe { input.add(token * (INPUT_COLUMNS / 2) + group * (GROUP_K / 2)) };
    let (p0, p1, p2, p3) = unsafe { load_u32x4(source) };
    let (p4, p5, p6, p7) = unsafe { load_u32x4(source.add(4)) };
    let (v0, v1) = convert::cvt_f32x2_bf16x2(p0);
    let (v2, v3) = convert::cvt_f32x2_bf16x2(p1);
    let (v4, v5) = convert::cvt_f32x2_bf16x2(p2);
    let (v6, v7) = convert::cvt_f32x2_bf16x2(p3);
    let (v8, v9) = convert::cvt_f32x2_bf16x2(p4);
    let (v10, v11) = convert::cvt_f32x2_bf16x2(p5);
    let (v12, v13) = convert::cvt_f32x2_bf16x2(p6);
    let (v14, v15) = convert::cvt_f32x2_bf16x2(p7);
    let values = [
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ];
    let mut max_abs = 0.0f32;
    let mut index = 0usize;
    while index < GROUP_K {
        max_abs = accumulate_max_abs(max_abs, values[index]);
        index += 1;
    }

    let scale_unencoded = float::div_rn_f32(input_scale_divisor * max_abs, 6.0);
    let encoded_pair = convert::cvt_rn_satfinite_e4m3x2_f32(scale_unencoded, scale_unencoded);
    let scale = encoded_pair as u8;
    let code_destination =
        unsafe { codes.add(token * (INPUT_COLUMNS / 8) + group * (GROUP_K / 8)) };
    if scale == 0 {
        unsafe {
            *code_destination = 0;
            *code_destination.add(1) = 0;
            *scales.add(task) = 0;
        }
        return;
    }

    let decoded_scale = e4m3_to_f32(scale);
    let (codes_lo, codes_hi) = pack_e2m1x16(
        float::div_rn_f32(v0 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v1 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v2 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v3 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v4 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v5 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v6 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v7 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v8 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v9 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v10 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v11 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v12 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v13 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v14 * input_scale_divisor, decoded_scale),
        float::div_rn_f32(v15 * input_scale_divisor, decoded_scale),
    );
    unsafe {
        *code_destination = codes_lo;
        *code_destination.add(1) = codes_hi;
        *scales.add(task) = scale;
    }
}

#[inline(always)]
fn swizzled_byte(row: usize, logical_byte: usize) -> usize {
    let logical_segment = logical_byte >> 4;
    let byte_in_segment = logical_byte & 15;
    let physical_segment = logical_segment ^ (row & (SEGMENTS_PER_ROW - 1));
    physical_segment * 16 + byte_in_segment
}

#[inline(always)]
fn weight_scale_offset<const INPUT_COLUMNS: usize>(parent_row: usize, scale_tile: usize) -> usize {
    let persistent_tile = parent_row / 128;
    let row_in_tile = parent_row & 127;
    let row_mod32 = row_in_tile & 31;
    let row_quartile = row_in_tile >> 5;
    let scale_tiles_per_row = INPUT_COLUMNS / GROUP_K / 4;

    (persistent_tile * scale_tiles_per_row + scale_tile) * 512 + row_mod32 * 16 + row_quartile * 4
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_tile<const INPUT_COLUMNS: usize, const TOKENS: usize>(
    shared: *mut u8,
    activation_codes: *const u32,
    activation_scales: *const u8,
    weight_codes: *const u32,
    weight_scales: *const u8,
    stage: usize,
    k_tile: usize,
    token_begin: usize,
    row_begin: usize,
    tid: usize,
) {
    if tid < TILE_M * SEGMENTS_PER_ROW {
        let row = tid / SEGMENTS_PER_ROW;
        let segment = tid - row * SEGMENTS_PER_ROW;
        let valid = token_begin + row < TOKENS;
        let source_token = if valid { token_begin + row } else { 0 };
        let physical = swizzled_byte(row, segment * 16);
        let destination = unsafe {
            shared
                .add(A_CODE_OFFSET + (stage * BLOCK_M + row) * CODE_ROW_BYTES + physical)
                .cast::<u32>()
        };
        let source = unsafe {
            activation_codes.add(
                source_token * (INPUT_COLUMNS / 8) + k_tile * (CODE_ROW_BYTES / 4) + segment * 4,
            )
        };
        unsafe {
            cp_async_cg_zfill_16(destination, source.cast::<u8>(), if valid { 16 } else { 0 });
        }
    }

    if tid < TILE_M {
        let valid = token_begin + tid < TOKENS;
        let source_token = if valid { token_begin + tid } else { 0 };
        let destination = unsafe {
            shared
                .add(A_SCALE_OFFSET + (stage * BLOCK_M + tid) * K64_PER_STAGE * 4)
                .cast::<u32>()
        };
        let source = unsafe {
            activation_scales
                .add(source_token * (INPUT_COLUMNS / GROUP_K) + k_tile * K64_PER_STAGE * 4)
        };
        unsafe { cp_async_cg_zfill_16(destination, source, if valid { 16 } else { 0 }) };
    }

    let mut task = tid;
    while task < BLOCK_N * SEGMENTS_PER_ROW {
        let row = task / SEGMENTS_PER_ROW;
        let segment = task - row * SEGMENTS_PER_ROW;
        let parent_row = row_begin + row;
        let physical = swizzled_byte(row, segment * 16);
        let destination = unsafe {
            shared
                .add(B_CODE_OFFSET + (stage * BLOCK_N + row) * CODE_ROW_BYTES + physical)
                .cast::<u32>()
        };
        let source = unsafe {
            weight_codes
                .add(parent_row * (INPUT_COLUMNS / 8) + k_tile * (CODE_ROW_BYTES / 4) + segment * 4)
        };
        unsafe { cp_async_cg_16(destination, source) };
        task += THREADS as usize;
    }

    let mut scale_task = tid;
    while scale_task < BLOCK_N * K64_PER_STAGE {
        let row = scale_task / K64_PER_STAGE;
        let local_k64 = scale_task - row * K64_PER_STAGE;
        let parent_row = row_begin + row;
        let global_k64 = k_tile * K64_PER_STAGE + local_k64;
        let destination = unsafe {
            shared
                .add(B_SCALE_OFFSET + (stage * BLOCK_N * K64_PER_STAGE + scale_task) * 4)
                .cast::<u32>()
        };
        let source = unsafe {
            weight_scales.add(weight_scale_offset::<INPUT_COLUMNS>(parent_row, global_k64))
        };
        unsafe { cp_async_ca_4(destination, source.cast::<u32>()) };
        scale_task += THREADS as usize;
    }
}

#[inline(always)]
fn mma_nvfp4(accumulators: &mut [f32; 4], a: [u32; 4], b: [u32; 2], scale_a: u32, scale_b: u32) {
    let scale_block_id = 0u16;
    let scale_thread_id = 0u16;
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.\
             m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, \
             {%10}, {%11,%12}, {%13}, {%14,%15};",
            inout("+f") accumulators[0],
            inout("+f") accumulators[1],
            inout("+f") accumulators[2],
            inout("+f") accumulators[3],
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b[0],
            in("r") b[1],
            in("r") scale_a,
            in("h") scale_block_id,
            in("h") scale_thread_id,
            in("r") scale_b,
            in("h") scale_block_id,
            in("h") scale_thread_id,
            options(register_only),
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn project_w4a4<
    const INPUT_COLUMNS: usize,
    const OUTPUT_ROWS: usize,
    const FIRST_ROWS: usize,
    const SECOND_ROWS: usize,
    const TOKENS: usize,
>(
    activation_codes: *const u32,
    activation_scales: *const u8,
    weight_codes: *const u32,
    weight_scales: *const u8,
    output: *mut u16,
    alpha0: f32,
    alpha1: f32,
    alpha2: f32,
) {
    static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;
    let shared = core::ptr::addr_of_mut!(SHARED).cast::<u8>();
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    let warp_m = warp / WARPS_N;
    let warp_n = warp - warp_m * WARPS_N;
    let token_begin = thread::blockIdx_y() as usize * TILE_M;
    let row_begin = thread::blockIdx_x() as usize * BLOCK_N;
    let alpha = if row_begin < FIRST_ROWS {
        alpha0
    } else if row_begin < FIRST_ROWS + SECOND_ROWS {
        alpha1
    } else {
        alpha2
    };
    let mut stage = 0usize;
    while stage < STAGES {
        unsafe {
            stage_tile::<INPUT_COLUMNS, TOKENS>(
                shared,
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                stage,
                stage,
                token_begin,
                row_begin,
                tid,
            );
            cp_async_commit_group();
        }
        stage += 1;
    }

    let mut accumulators = [[0.0f32; 4]; MMA_N];
    let a_matrix = lane >> 3;
    let a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    let a_column_byte = (a_matrix >> 1) * 16;
    let b_row_offset = lane & 7;
    let b_column_byte = ((lane >> 3) & 1) * 16;
    let scale_a_row_offset = ((lane & 1) << 3) | (lane >> 2);
    let scale_b_row_offset = lane >> 2;
    let mut k_tile = 0usize;
    while k_tile < INPUT_COLUMNS / BLOCK_K {
        stage = k_tile % STAGES;
        unsafe { cp_async_wait_group(1) };
        thread::sync_threads();

        let mut local_k64 = 0usize;
        while local_k64 < K64_PER_STAGE {
            let a_row = warp_m * WARP_M + a_row_offset;
            let a_logical_byte = local_k64 * 32 + a_column_byte;
            let a_physical_byte = swizzled_byte(a_row, a_logical_byte);
            let a_address = unsafe {
                shared
                    .add(
                        A_CODE_OFFSET
                            + (stage * BLOCK_M + a_row) * CODE_ROW_BYTES
                            + a_physical_byte,
                    )
                    .cast::<u32>()
            };
            let a_fragments = unsafe { wmma::ldmatrix_x4(a_address) };
            let scale_a_row = warp_m * WARP_M + scale_a_row_offset;
            let scale_a = unsafe {
                *shared
                    .add(
                        A_SCALE_OFFSET
                            + (stage * BLOCK_M + scale_a_row) * K64_PER_STAGE * 4
                            + local_k64 * 4,
                    )
                    .cast::<u32>()
            };
            let mut mma_n = 0usize;
            while mma_n < MMA_N {
                let b_row = warp_n * WARP_N + mma_n * 8 + b_row_offset;
                let b_logical_byte = local_k64 * 32 + b_column_byte;
                let b_physical_byte = swizzled_byte(b_row, b_logical_byte);
                let b_address = unsafe {
                    shared
                        .add(
                            B_CODE_OFFSET
                                + (stage * BLOCK_N + b_row) * CODE_ROW_BYTES
                                + b_physical_byte,
                        )
                        .cast::<u32>()
                };
                let b_fragments = unsafe { wmma::ldmatrix_x2(b_address) };
                let scale_b_row = warp_n * WARP_N + mma_n * 8 + scale_b_row_offset;
                let scale_b = unsafe {
                    *shared
                        .add(
                            B_SCALE_OFFSET
                                + (stage * BLOCK_N + scale_b_row) * K64_PER_STAGE * 4
                                + local_k64 * 4,
                        )
                        .cast::<u32>()
                };
                mma_nvfp4(
                    &mut accumulators[mma_n],
                    a_fragments,
                    b_fragments,
                    scale_a,
                    scale_b,
                );
                mma_n += 1;
            }
            local_k64 += 1;
        }

        thread::sync_threads();
        let next_k_tile = k_tile + STAGES;
        if next_k_tile < INPUT_COLUMNS / BLOCK_K {
            unsafe {
                stage_tile::<INPUT_COLUMNS, TOKENS>(
                    shared,
                    activation_codes,
                    activation_scales,
                    weight_codes,
                    weight_scales,
                    stage,
                    next_k_tile,
                    token_begin,
                    row_begin,
                    tid,
                );
            }
        }
        unsafe { cp_async_commit_group() };
        k_tile += 1;
    }

    let accumulator_row = lane >> 2;
    let accumulator_col = 2 * (lane & 3);
    let token0 = warp_m * WARP_M + accumulator_row;
    let token1 = token0 + 8;
    let mut mma_n = 0usize;
    while mma_n < MMA_N {
        let local_row = warp_n * WARP_N + mma_n * 8 + accumulator_col;
        let values = accumulators[mma_n];
        if token_begin + token0 < TOKENS {
            let destination = unsafe {
                output
                    .add((token_begin + token0) * OUTPUT_ROWS + row_begin + local_row)
                    .cast::<u32>()
            };
            unsafe {
                *destination = tcgen05::cvt_f32x2_bf16x2(values[0] * alpha, values[1] * alpha);
            }
        }
        if token_begin + token1 < TOKENS {
            let destination = unsafe {
                output
                    .add((token_begin + token1) * OUTPUT_ROWS + row_begin + local_row)
                    .cast::<u32>()
            };
            unsafe {
                *destination = tcgen05::cvt_f32x2_bf16x2(values[2] * alpha, values[3] * alpha);
            }
        }
        mma_n += 1;
    }
}

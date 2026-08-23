//! Private T=1024 TMA route for the dense-FP8 down projection.

use cuda_core::sys::{
    self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8,
    CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B, cuTensorMapEncodeTiled,
};
use cuda_device::{
    DynamicSharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaStream, DeviceBuffer, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch,
};
use tuisko_model::{Arch, Qwen38_27B};

pub(super) const TOKENS: usize = 1_024;
const BLOCK_M: usize = 128;
const BLOCK_N: usize = 64;
const BLOCK_K: usize = 64;
const STAGES: usize = 3;
const PRODUCER_THREADS: usize = 32;
const CONSUMER_WARPS: usize = 8;
const CONSUMER_THREADS: usize = CONSUMER_WARPS * 32;
const THREADS: usize = PRODUCER_THREADS + CONSUMER_THREADS;
const WARPS_M: usize = 4;
const WARPS_N: usize = 2;
const WARP_M: usize = BLOCK_M / WARPS_M;
const WARP_N: usize = BLOCK_N / WARPS_N;
const MMA_M: usize = WARP_M / 16;
const MMA_N: usize = WARP_N / 8;
const K32_PER_STAGE: usize = BLOCK_K / 32;
const K_TILES: usize = Qwen38_27B::INTERMEDIATE / BLOCK_K;
const CODE_ROW_BYTES: usize = BLOCK_K;
const OUTPUT_STRIDE: usize = BLOCK_N + 8;

const A_CODE_OFFSET: usize = 0;
const A_CODE_BYTES: usize = STAGES * BLOCK_M * CODE_ROW_BYTES;
const WEIGHT_CODE_OFFSET: usize = A_CODE_OFFSET + A_CODE_BYTES;
const WEIGHT_CODE_BYTES: usize = STAGES * BLOCK_N * CODE_ROW_BYTES;
const FULL_BARRIER_OFFSET: usize = WEIGHT_CODE_OFFSET + WEIGHT_CODE_BYTES;
const EMPTY_BARRIER_OFFSET: usize = FULL_BARRIER_OFFSET + STAGES * 8;
const SHARED_BYTES: usize = EMPTY_BARRIER_OFFSET + STAGES * 8;
const TRANSACTION_BYTES: u32 = (BLOCK_M * CODE_ROW_BYTES + BLOCK_N * CODE_ROW_BYTES) as u32;
const _: () = assert!(SHARED_BYTES == 36_912);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
        mbarrier_init, mbarrier_try_wait_parity,
    };
    use cuda_device::tma::TmaDescriptor;
    use cuda_device::{tcgen05, wmma};

    #[inline(always)]
    fn swizzled_byte(row: usize, logical_byte: usize) -> usize {
        (((logical_byte >> 4) ^ ((row >> 1) & 3)) << 4) + (logical_byte & 15)
    }

    #[inline(always)]
    unsafe fn cp_async_bulk_tensor_2d_g2s_cta(
        destination: *mut u8,
        tensor_map: *const TmaDescriptor,
        coordinate0: i32,
        coordinate1: i32,
        barrier: *mut Barrier,
    ) {
        // This exact route has no multi-CTA cluster. The CTA-shared form avoids
        // the generic cluster-rank conversion and its out-of-line stack frame.
        unsafe {
            ptx_asm!(
                "{\n\
                    .reg .u64 destination_shared;\n\
                    .reg .u64 barrier_shared;\n\
                    cvta.to.shared.u64 destination_shared, %0;\n\
                    cvta.to.shared.u64 barrier_shared, %4;\n\
                    cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes \
                        [destination_shared], [%1, {%2, %3}], [barrier_shared];\n\
                }",
                in("l") destination,
                in("l") tensor_map,
                in("r") coordinate0,
                in("r") coordinate1,
                in("l") barrier,
                clobber("memory"),
            )
        }
    }

    /// Applies the retained three-stage down route at exactly T=1024.
    #[kernel]
    #[launch_bounds(288, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (288, 1, 1),
        dynamic_shared = 36912,
        dynamic_shared_alignment = 128,
        min_compute_capability = (12, 0),
    )]
    pub fn fp8_down_tma_t1024(
        activation_code_map: *const TmaDescriptor,
        weight_code_map: *const TmaDescriptor,
        activation_scales: *const f32,
        weight_scales: *const u16,
        output: *mut u16,
    ) {
        // One producer warp overlaps 64-wide K stages with eight consumer
        // warps. A 128x64 tile exposes 640 CTAs and retains two-CTA residency
        // in 36,912 shared bytes for this exact 17,408-wide projection.
        let shared = DynamicSharedArray::<u8, 128>::get();
        // SAFETY: the launch reserves both barrier arrays after the code stages.
        let full = unsafe { shared.add(FULL_BARRIER_OFFSET).cast::<Barrier>() };
        // SAFETY: the launch reserves both barrier arrays after the code stages.
        let empty = unsafe { shared.add(EMPTY_BARRIER_OFFSET).cast::<Barrier>() };
        let tid = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let output_tiles = Qwen38_27B::HIDDEN / BLOCK_N;
        let token_tile = block / output_tiles;
        let output_tile = block - token_tile * output_tiles;
        let token_begin = token_tile * BLOCK_M;
        let row_begin = output_tile * BLOCK_N;

        if tid == 0 {
            let mut stage = 0usize;
            while stage < STAGES {
                // SAFETY: lane zero initializes each disjoint barrier slot once.
                unsafe {
                    mbarrier_init(full.add(stage), 1);
                    mbarrier_init(empty.add(stage), CONSUMER_WARPS as u32);
                }
                stage += 1;
            }
            // SAFETY: initialization must be visible to the async proxy.
            unsafe { fence_proxy_async_shared_cta() };
        }
        thread::sync_threads();

        if tid < PRODUCER_THREADS {
            if tid == 0 {
                let mut k_tile = 0usize;
                while k_tile < K_TILES {
                    let stage = k_tile % STAGES;
                    let empty_phase = 1 ^ ((k_tile / STAGES) & 1);
                    // SAFETY: descriptors, phases, and exact tiles are admitted.
                    unsafe {
                        while !mbarrier_try_wait_parity(empty.add(stage), empty_phase as u32) {}
                        let barrier = full.add(stage);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            shared.add(A_CODE_OFFSET + stage * BLOCK_M * CODE_ROW_BYTES),
                            activation_code_map,
                            (k_tile * CODE_ROW_BYTES) as i32,
                            token_begin as i32,
                            barrier,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            shared.add(WEIGHT_CODE_OFFSET + stage * BLOCK_N * CODE_ROW_BYTES),
                            weight_code_map,
                            (k_tile * CODE_ROW_BYTES) as i32,
                            row_begin as i32,
                            barrier,
                        );
                        mbarrier_arrive_expect_tx(barrier, 1, TRANSACTION_BYTES);
                    }
                    k_tile += 1;
                }
            }
            return;
        }

        let consumer_thread = tid - PRODUCER_THREADS;
        let lane = consumer_thread & 31;
        let warp = consumer_thread >> 5;
        let warp_m = warp / WARPS_N;
        let warp_n = warp - warp_m * WARPS_N;
        let a_matrix = lane >> 3;
        let a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
        let a_column_byte = (a_matrix >> 1) * 16;
        let b_row_offset = lane & 7;
        let b_column_byte = ((lane >> 3) & 1) * 16;
        let mut accumulators = [[[0.0f32; 4]; MMA_N]; MMA_M];

        let mut k_tile = 0usize;
        while k_tile < K_TILES {
            let stage = k_tile % STAGES;
            let full_phase = (k_tile / STAGES) & 1;
            // SAFETY: every consumer waits for the current stage before reading.
            unsafe { while !mbarrier_try_wait_parity(full.add(stage), full_phase as u32) {} }
            let mut local_k32 = 0usize;
            #[unroll]
            while local_k32 < K32_PER_STAGE {
                let mut a_fragments = [[0u32; 4]; MMA_M];
                let mut mma_m = 0usize;
                #[unroll]
                while mma_m < MMA_M {
                    let row = warp_m * WARP_M + mma_m * 16 + a_row_offset;
                    let logical_byte = local_k32 * 32 + a_column_byte;
                    // SAFETY: the swizzled address names one m16k32 fragment.
                    a_fragments[mma_m] = unsafe {
                        wmma::ldmatrix_x4(
                            shared
                                .add(
                                    A_CODE_OFFSET
                                        + stage * BLOCK_M * CODE_ROW_BYTES
                                        + row * CODE_ROW_BYTES
                                        + swizzled_byte(row, logical_byte),
                                )
                                .cast::<u32>(),
                        )
                    };
                    mma_m += 1;
                }

                let mut b_fragments = [[0u32; 2]; MMA_N];
                let mut mma_n = 0usize;
                #[unroll]
                while mma_n < MMA_N {
                    let row = warp_n * WARP_N + mma_n * 8 + b_row_offset;
                    let logical_byte = local_k32 * 32 + b_column_byte;
                    // SAFETY: the swizzled address names one n8k32 fragment.
                    b_fragments[mma_n] = unsafe {
                        wmma::ldmatrix_x2(
                            shared
                                .add(
                                    WEIGHT_CODE_OFFSET
                                        + stage * BLOCK_N * CODE_ROW_BYTES
                                        + row * CODE_ROW_BYTES
                                        + swizzled_byte(row, logical_byte),
                                )
                                .cast::<u32>(),
                        )
                    };
                    mma_n += 1;
                }

                mma_m = 0;
                #[unroll]
                while mma_m < MMA_M {
                    mma_n = 0;
                    #[unroll]
                    while mma_n < MMA_N {
                        // SAFETY: fragments preserve the m16n8k32 E4M3 contract.
                        accumulators[mma_m][mma_n] = unsafe {
                            cuda_intrinsics::matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                                accumulators[mma_m][mma_n],
                                a_fragments[mma_m],
                                b_fragments[mma_n],
                            )
                        };
                        mma_n += 1;
                    }
                    mma_m += 1;
                }
                local_k32 += 1;
            }
            if lane == 0 {
                // SAFETY: one lane per consumer warp releases the stage once.
                unsafe {
                    let _ = mbarrier_arrive(empty.add(stage));
                }
            }
            k_tile += 1;
        }

        // SAFETY: all consumers participate before shared-memory output reuse.
        unsafe { ptx_asm!("bar.sync 1, 256;", clobber("memory")) };
        let output_shared = shared.cast::<u16>();
        let accumulator_row = lane >> 2;
        let accumulator_column = 2 * (lane & 3);
        let mut mma_m = 0usize;
        #[unroll]
        while mma_m < MMA_M {
            let token0 = warp_m * WARP_M + mma_m * 16 + accumulator_row;
            let token1 = token0 + 8;
            // SAFETY: the exact token tile owns both activation scales.
            let activation_scale0 = unsafe { *activation_scales.add(token_begin + token0) };
            // SAFETY: the exact token tile owns both activation scales.
            let activation_scale1 = unsafe { *activation_scales.add(token_begin + token1) };
            let mut mma_n = 0usize;
            #[unroll]
            while mma_n < MMA_N {
                let row = warp_n * WARP_N + mma_n * 8 + accumulator_column;
                // SAFETY: the scale plane covers both adjacent output rows.
                let weight_scale0 =
                    f32::from_bits((unsafe { *weight_scales.add(row_begin + row) } as u32) << 16);
                let weight_scale1 = f32::from_bits(
                    (unsafe { *weight_scales.add(row_begin + row + 1) } as u32) << 16,
                );
                let values = accumulators[mma_m][mma_n];
                let packed0 = tcgen05::cvt_f32x2_bf16x2(
                    values[0] * activation_scale0 * weight_scale0,
                    values[1] * activation_scale0 * weight_scale1,
                );
                let packed1 = tcgen05::cvt_f32x2_bf16x2(
                    values[2] * activation_scale1 * weight_scale0,
                    values[3] * activation_scale1 * weight_scale1,
                );
                // SAFETY: each lane owns disjoint packed output pairs.
                unsafe {
                    *output_shared
                        .add(token0 * OUTPUT_STRIDE + row)
                        .cast::<u32>() = packed0;
                    *output_shared
                        .add(token1 * OUTPUT_STRIDE + row)
                        .cast::<u32>() = packed1;
                }
                mma_n += 1;
            }
            mma_m += 1;
        }
        // SAFETY: shared output publication completes before the global copies.
        unsafe { ptx_asm!("bar.sync 1, 256;", clobber("memory")) };

        let vectors_per_row = BLOCK_N / 8;
        let output_vectors = BLOCK_M * vectors_per_row;
        let mut task = consumer_thread;
        while task < output_vectors {
            let token = task / vectors_per_row;
            let row_vector = task - token * vectors_per_row;
            // SAFETY: vector tasks partition shared and global output planes.
            unsafe {
                let values = *output_shared
                    .add(token * OUTPUT_STRIDE + row_vector * 8)
                    .cast::<[u32; 4]>();
                *output
                    .add((token_begin + token) * Qwen38_27B::HIDDEN + row_begin + row_vector * 8)
                    .cast::<[u32; 4]>() = values;
            }
            task += CONSUMER_THREADS;
        }
    }
}

/// Address-bound tensor maps for the exact dense-FP8 down macro route.
pub struct DenseFp8DownTmaMaps {
    activation_code_map: DeviceBuffer<u64>,
    weight_code_map: DeviceBuffer<u64>,
    activation_codes: usize,
    weight_codes: usize,
}

impl DenseFp8DownTmaMaps {
    /// Exact bytes in the two opaque CUDA tensor maps.
    pub const BYTE_LEN: usize = 2 * size_of::<CUtensorMap>();

    /// Encodes and uploads the two exact 64-byte-swizzled tensor maps.
    ///
    /// # Safety
    ///
    /// `activation_codes` covers `[1024, 17408]` E4M3 bytes and
    /// `weight_codes` covers source-native `[5120, 17408]` E4M3 bytes. Both
    /// allocations belong to `stream`'s context, remain address-stable for
    /// this owner's life, and are at least 16-byte aligned.
    pub unsafe fn new(
        stream: &CudaStream,
        activation_codes: *const u8,
        weight_codes: *const u8,
    ) -> GpuResult<Self> {
        let activation = create_map(
            activation_codes.cast_mut().cast::<c_void>(),
            Qwen38_27B::INTERMEDIATE as u64,
            TOKENS as u64,
            Qwen38_27B::INTERMEDIATE as u64,
            BLOCK_K as u32,
            BLOCK_M as u32,
        )?;
        let weight = create_map(
            weight_codes.cast_mut().cast::<c_void>(),
            Qwen38_27B::INTERMEDIATE as u64,
            Qwen38_27B::HIDDEN as u64,
            Qwen38_27B::INTERMEDIATE as u64,
            BLOCK_K as u32,
            BLOCK_N as u32,
        )?;
        let activation_code_map =
            DeviceBuffer::from_host(stream, &activation.opaque).map_err(GpuError::from)?;
        let weight_code_map =
            DeviceBuffer::from_host(stream, &weight.opaque).map_err(GpuError::from)?;

        Ok(Self {
            activation_code_map,
            weight_code_map,
            activation_codes: activation_codes.addr(),
            weight_codes: weight_codes.addr(),
        })
    }

    /// Exact bytes owned by the two uploaded tensor maps.
    pub fn byte_len(&self) -> usize {
        Self::BYTE_LEN
    }

    /// Stable addresses of the two device-resident descriptors.
    pub fn device_addresses(&self) -> [usize; 2] {
        [
            self.activation_code_map.cu_deviceptr() as usize,
            self.weight_code_map.cu_deviceptr() as usize,
        ]
    }

    /// Source addresses encoded by the two maps.
    pub const fn source_addresses(&self) -> [usize; 2] {
        [self.activation_codes, self.weight_codes]
    }

    /// Copies both opaque descriptors for qualification of immutable ownership.
    pub fn copy_to_host(&self, stream: &CudaStream) -> GpuResult<[Vec<u64>; 2]> {
        if !Arc::ptr_eq(stream.context(), self.activation_code_map.context())
            || !Arc::ptr_eq(stream.context(), self.weight_code_map.context())
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 down tensor maps belong to another CUDA context",
            ));
        }
        Ok([
            self.activation_code_map
                .to_host_vec(stream)
                .map_err(GpuError::from)?,
            self.weight_code_map
                .to_host_vec(stream)
                .map_err(GpuError::from)?,
        ])
    }

    pub(super) fn activation_codes(&self) -> usize {
        self.activation_codes
    }

    pub(super) fn weight_codes(&self) -> usize {
        self.weight_codes
    }
}

pub(super) struct DenseFp8DownTmaRoute {
    module: kernels::LoadedModule,
    prepared: PreparedLaunch<kernels::__fp8_down_tma_t1024_CudaKernel>,
}

impl DenseFp8DownTmaRoute {
    pub(super) fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        // SAFETY: this route owns one cuda-oxide module and embedded artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading dense-FP8 down TMA module", source))?;
        let blocks = ((TOKENS / BLOCK_M) * (Qwen38_27B::HIDDEN / BLOCK_N)) as u32;
        let prepared = module
            .prepare_fp8_down_tma_t1024(LaunchConfig1D::new(
                blocks,
                THREADS as u32,
                SHARED_BYTES as u32,
            ))
            .map_err(|source| GpuError::launch("preparing dense-FP8 down T=1024", source))?;

        Ok(Self { module, prepared })
    }

    pub(super) unsafe fn launch(
        &self,
        stream: &CudaStream,
        maps: &DenseFp8DownTmaMaps,
        activation_scales: *const f32,
        weight_scales: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !Arc::ptr_eq(stream.context(), maps.activation_code_map.context())
            || !Arc::ptr_eq(stream.context(), maps.weight_code_map.context())
        {
            return Err(GpuError::invalid_launch(
                "dense-FP8 down tensor maps belong to another CUDA context",
            ));
        }
        self.module
            .fp8_down_tma_t1024(
                stream,
                &self.prepared,
                maps.activation_code_map.cu_deviceptr() as *const cuda_device::tma::TmaDescriptor,
                maps.weight_code_map.cu_deviceptr() as *const cuda_device::tma::TmaDescriptor,
                activation_scales,
                weight_scales,
                output,
            )
            .map_err(|source| GpuError::launch("launching dense-FP8 down T=1024", source))
    }
}

pub(super) fn ptx_name() -> &'static str {
    "fp8_down_tma_t1024"
}

fn create_map(
    address: *mut c_void,
    columns: u64,
    rows: u64,
    row_stride: u64,
    box_columns: u32,
    box_rows: u32,
) -> GpuResult<CUtensorMap> {
    let mut descriptor = MaybeUninit::<CUtensorMap>::uninit();
    let dimensions = [columns, rows];
    let strides = [row_stride];
    let box_dimensions = [box_columns, box_rows];
    let element_strides = [1u32, 1];
    // SAFETY: metadata has exact fixed sizes and the caller supplies the live
    // address-stable allocation encoded by the descriptor.
    let result = unsafe {
        cuTensorMapEncodeTiled(
            descriptor.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            address,
            dimensions.as_ptr(),
            strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if result != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        return Err(GpuError::invalid_launch(format!(
            "encoding dense-FP8 down tensor map failed: {result:?}"
        )));
    }

    // SAFETY: CUDA_SUCCESS initializes the full opaque descriptor.
    Ok(unsafe { descriptor.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_K, BLOCK_M, BLOCK_N, CONSUMER_WARPS, PRODUCER_THREADS, SHARED_BYTES, THREADS, TOKENS,
    };

    #[test]
    fn exact_tma_geometry_preserves_the_down_route() {
        assert_eq!(TOKENS, 1_024);
        assert_eq!((BLOCK_M, BLOCK_N, BLOCK_K), (128, 64, 64));
        assert_eq!(PRODUCER_THREADS + CONSUMER_WARPS * 32, THREADS);
        assert_eq!(THREADS, 288);
        assert_eq!(SHARED_BYTES, 36_912);
    }
}

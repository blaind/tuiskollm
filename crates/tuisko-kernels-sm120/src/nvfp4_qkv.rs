//! Exact Qwen3.5 NVFP4 full-attention QKV projection.

use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen35_9B::ATTENTION_QKV_ROWS;
const QUERY_ROWS: usize = Qwen35_9B::ATTENTION_QUERY_ROWS;
const KEY_ROWS: usize = Qwen35_9B::ATTENTION_KV_ROWS;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP_K;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const PHASE_GROUPS: usize = 32;
const PHASES: usize = GROUPS_PER_ROW / PHASE_GROUPS;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();

// The spill-free Qwen3.5 down route uses eight warps, two rows per warp,
// and 8,192 shared bytes. QKV shortens K from 12,288 to 4,096 and raises M
// from 4,096 to 10,240; retaining that ownership yields 640 CTAs = 3.76
// CTAs/SM on the 170-SM target. Each lane still owns one K16 group and each
// warp still reduces its two rows in the same phase/lane order.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = PHASE_GROUPS * GROUP_K / 2;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

const _: () = assert!(INPUT_COLUMNS == 4_096);
const _: () = assert!(OUTPUT_ROWS == 10_240);
const _: () = assert!(QUERY_ROWS == 8_192);
const _: () = assert!(KEY_ROWS == 1_024);
const _: () = assert!(GROUPS_PER_ROW == 256);
const _: () = assert!(PHASES == 8);
const _: () = assert!(SHARED_U32 * size_of::<u32>() == 8_192);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, warp};

    #[inline(always)]
    fn weight_scale_offset(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;
        let scale_tiles_per_row = GROUPS_PER_ROW / 4;

        (persistent_tile * scale_tiles_per_row + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset(parent_row: usize, group: usize) -> usize {
        weight_scale_offset(parent_row, group >> 2) + (group & 3)
    }

    #[inline(always)]
    fn physical_row(index: usize) -> usize {
        let tile = index >> 7;
        let in_tile = index & 127;

        tile * 128 + (in_tile >> 2) + (in_tile & 3) * 32
    }

    #[inline(always)]
    fn row_weight_reciprocal(
        row: usize,
        query_reciprocal: f32,
        key_reciprocal: f32,
        value_reciprocal: f32,
    ) -> f32 {
        if row < QUERY_ROWS {
            query_reciprocal
        } else if row < QUERY_ROWS + KEY_ROWS {
            key_reciprocal
        } else {
            value_reciprocal
        }
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
    fn reduce_sum_lane0(mut value: f32) -> f32 {
        value += warp::shuffle_down_f32(value, 16);
        value += warp::shuffle_down_f32(value, 8);
        value += warp::shuffle_down_f32(value, 4);
        value += warp::shuffle_down_f32(value, 2);
        value += warp::shuffle_down_f32(value, 1);

        value
    }

    #[inline(always)]
    fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

        (rounded >> 16) as u16
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn qkv_body<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        query_weight_reciprocal: f32,
        key_weight_reciprocal: f32,
        value_weight_reciprocal: f32,
        output: *mut u16,
        shared: *mut u32,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let pair_index = block * (2 * WARPS) + 2 * warp_index;
        let first_row = physical_row(pair_index);
        let second_row = physical_row(pair_index + 1);
        let first_reciprocal = row_weight_reciprocal(
            first_row,
            query_weight_reciprocal,
            key_weight_reciprocal,
            value_weight_reciprocal,
        );
        let second_reciprocal = row_weight_reciprocal(
            second_row,
            query_weight_reciprocal,
            key_weight_reciprocal,
            value_weight_reciprocal,
        );
        let mut first_accumulators = [0.0f32; TOKENS];
        let mut second_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < PHASES {
            let mut task = tid;
            while task < TOKENS * PHASE_PACKED_PAIRS {
                let token = task / PHASE_PACKED_PAIRS;
                let pair = task - token * PHASE_PACKED_PAIRS;
                unsafe {
                    *shared.add(task) =
                        *input.add(token * (INPUT_COLUMNS / 2) + phase * 256 + pair);
                }
                task += THREADS as usize;
            }
            thread::sync_threads();

            let group = phase * PHASE_GROUPS + lane;
            let first_scale = unsafe {
                load_u8_read_only(weight_scales.add(weight_group_scale_offset(first_row, group)))
            };
            let second_scale = unsafe {
                load_u8_read_only(weight_scales.add(weight_group_scale_offset(second_row, group)))
            };
            let first_coefficient = e4m3_to_f32(first_scale) * first_reciprocal;
            let second_coefficient = e4m3_to_f32(second_scale) * second_reciprocal;
            let row_words = CODE_BYTES_PER_ROW / size_of::<u32>();
            let word_offset = phase * CODE_WORDS_PER_PHASE + lane * 2;
            let first_source = unsafe { weight_codes.add(first_row * row_words + word_offset) };
            let second_source = unsafe { weight_codes.add(second_row * row_words + word_offset) };
            let first_words = unsafe { load_u32x2_read_only(first_source) };
            let second_words = unsafe { load_u32x2_read_only(second_source) };

            macro_rules! accumulate_pair {
                ($pair:literal) => {{
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
                    let (first_weight0, first_weight1) = e2m1x2_to_f32(first_packed);
                    let (second_weight0, second_weight1) = e2m1x2_to_f32(second_packed);

                    macro_rules! accumulate_token {
                        ($token:literal) => {
                            if $token < TOKENS {
                                let bits = unsafe {
                                    *shared.add($token * PHASE_PACKED_PAIRS + lane * 8 + $pair)
                                };
                                let (activation0, activation1) = convert::cvt_f32x2_bf16x2(bits);
                                first_accumulators[$token] = float::fma_rn_f32(
                                    first_weight0 * first_coefficient,
                                    activation0,
                                    first_accumulators[$token],
                                );
                                first_accumulators[$token] = float::fma_rn_f32(
                                    first_weight1 * first_coefficient,
                                    activation1,
                                    first_accumulators[$token],
                                );
                                second_accumulators[$token] = float::fma_rn_f32(
                                    second_weight0 * second_coefficient,
                                    activation0,
                                    second_accumulators[$token],
                                );
                                second_accumulators[$token] = float::fma_rn_f32(
                                    second_weight1 * second_coefficient,
                                    activation1,
                                    second_accumulators[$token],
                                );
                            }
                        };
                    }

                    accumulate_token!(0);
                    accumulate_token!(1);
                    accumulate_token!(2);
                    accumulate_token!(3);
                    accumulate_token!(4);
                    accumulate_token!(5);
                    accumulate_token!(6);
                    accumulate_token!(7);
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
            thread::sync_threads();
            phase += 1;
        }

        macro_rules! finish_token {
            ($token:literal) => {
                if $token < TOKENS {
                    let first = reduce_sum_lane0(first_accumulators[$token]);
                    let second = reduce_sum_lane0(second_accumulators[$token]);

                    if lane == 0 {
                        unsafe {
                            *output.add($token * OUTPUT_ROWS + first_row) = f32_to_bf16(first);
                            *output.add($token * OUTPUT_ROWS + second_row) = f32_to_bf16(second);
                        }
                    }
                }
            };
        }

        finish_token!(0);
        finish_token!(1);
        finish_token!(2);
        finish_token!(3);
        finish_token!(4);
        finish_token!(5);
        finish_token!(6);
        finish_token!(7);
    }

    /// Projects exact Qwen3.5 BF16 rows into Q/gate, K, and V.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_qkv_a16<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        query_weight_reciprocal: f32,
        key_weight_reciprocal: f32,
        value_weight_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            qkv_body::<TOKENS>(
                input,
                weight_codes,
                weight_scales,
                query_weight_reciprocal,
                key_weight_reciprocal,
                value_weight_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }
}

fn launch_config() -> LaunchConfig1D {
    LaunchConfig1D::new((OUTPUT_ROWS / (2 * WARPS)) as u32, THREADS, 0)
}

struct PreparedBatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_qkv_a16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen35_nvfp4_qkv_a16::<TOKENS>(launch_config())
            .map_err(|source| GpuError::launch("preparing Qwen3.5 SM120 NVFP4 QKV", source))?;

        Ok(Self { projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_reciprocals: [f32; 3],
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_qkv_a16::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocals[0],
                weight_scale_reciprocals[1],
                weight_scale_reciprocals[2],
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 SM120 NVFP4 QKV", source))
    }
}

/// PTX symbols retained for every exact Qwen3.5 QKV batch.
pub(crate) fn qwen35_nvfp4_qkv_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_qkv_a16_ptx_name::<8>(),
    ]
}

/// Prepared exact-batch Qwen3.5 NVFP4 QKV routes on SM120.
pub struct Qwen35Nvfp4QkvOp {
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

impl Qwen35Nvfp4QkvOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_nvfp4_qkv_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 SM120 NVFP4 QKV module", source)
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
            module,
        })
    }

    /// Executes represented-weight A16 QKV at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 4_096` BF16 values; `weight_codes` covers
    /// packed E2M1 `[10_240, 4_096]`; `weight_scales` covers swizzled E4M3
    /// `[10_240, 256]`; and `output` covers `batch * 10_240` BF16 values.
    /// Four-byte-loaded planes are aligned, all divisors are finite and
    /// positive, and disjoint allocations remain live in `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_divisors: [f32; 3],
        output: *mut u16,
    ) -> GpuResult<()> {
        if weight_scale_divisors
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 NVFP4 QKV weight scale divisors must be finite and positive",
            ));
        }
        let reciprocals = weight_scale_divisors.map(|value| 1.0 / value);

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        weight_codes,
                        weight_scales,
                        reciprocals,
                        output,
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
                "Qwen3.5 NVFP4 QKV batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_WORDS_PER_PHASE, GROUPS_PER_ROW, KEY_ROWS, MAX_BATCH, OUTPUT_ROWS, PHASES, QUERY_ROWS,
        SHARED_U32, WARPS, qwen35_nvfp4_qkv_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(OUTPUT_ROWS, 10_240);
        assert_eq!(QUERY_ROWS, 8_192);
        assert_eq!(KEY_ROWS, 1_024);
        assert_eq!(GROUPS_PER_ROW, 256);
        assert_eq!(PHASES, 8);
        assert_eq!(CODE_WORDS_PER_PHASE, 64);
        assert_eq!(OUTPUT_ROWS / (2 * WARPS), 640);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);

        let names = qwen35_nvfp4_qkv_ptx_names();
        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

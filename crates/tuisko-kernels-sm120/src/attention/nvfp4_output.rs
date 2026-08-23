//! Exact Qwen3.5 gated NVFP4 attention-output projection.

use crate::device::attention_output::attention_gate_bf16;
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen35_9B::ATTENTION_OUTPUT_COLUMNS;
const OUTPUT_ROWS: usize = Qwen35_9B::HIDDEN;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP_K;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const PHASE_GROUPS: usize = 32;
const PHASES: usize = GROUPS_PER_ROW / PHASE_GROUPS;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();

// Eight warps own two output rows each, giving 256 CTAs = 1.51 CTAs/SM on
// the 170-SM target. Eight K phases cover all 256 K16 groups; each lane
// retains one group and the same phase/lane accumulation order as the
// qualified Qwen3.5 QKV A16 route.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = PHASE_GROUPS * GROUP_K / 2;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

const _: () = assert!(INPUT_COLUMNS == 4_096);
const _: () = assert!(OUTPUT_ROWS == 4_096);
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
    unsafe fn projection_body<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_reciprocal: f32,
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
            let first_coefficient = e4m3_to_f32(first_scale) * weight_reciprocal;
            let second_coefficient = e4m3_to_f32(second_scale) * weight_reciprocal;
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

    /// Applies the Qwen3.5 attention gate and publishes BF16 projection input.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_attention_output_gate_bf16<const TOKENS: usize>(
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
    ) {
        // One CTA owns one 4,096-wide row; 256 threads make exactly 16
        // coalesced passes and preserve the established gate-column mapping.
        unsafe {
            attention_gate_bf16::<Qwen35_9B>(attention, qkv, activation);
        }
    }

    /// Projects gated BF16 attention rows through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_attention_output_a16<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            projection_body::<TOKENS>(
                input,
                weight_codes,
                weight_scales,
                weight_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }
}

fn projection_launch_config() -> LaunchConfig1D {
    LaunchConfig1D::new((OUTPUT_ROWS / (2 * WARPS)) as u32, THREADS, 0)
}

struct PreparedRoute<const TOKENS: usize> {
    gate: PreparedLaunch<kernels::__qwen35_nvfp4_attention_output_gate_bf16_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_attention_output_a16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let gate = module
            .prepare_qwen35_nvfp4_attention_output_gate_bf16::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing Qwen3.5 NVFP4 attention gate", source))?;
        let projection = module
            .prepare_qwen35_nvfp4_attention_output_a16::<TOKENS>(projection_launch_config())
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.5 NVFP4 attention output", source)
            })?;

        Ok(Self { gate, projection })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_attention_output_gate_bf16::<TOKENS>(
                stream, &self.gate, attention, qkv, activation,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 NVFP4 attention gate", source))?;
        module
            .qwen35_nvfp4_attention_output_a16::<TOKENS>(
                stream,
                &self.projection,
                activation.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 NVFP4 attention output", source))
    }
}

/// Prepared gated NVFP4 attention-output routes for exact `B=1..=8`.
pub struct Qwen35Nvfp4AttentionOutputOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
}

impl Qwen35Nvfp4AttentionOutputOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_nvfp4_attention_output_ptx_names();
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module(
                "loading the Qwen3.5 SM120 NVFP4 attention-output module",
                source,
            )
        })?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            module,
        })
    }

    /// Gates paged-attention output, converts it to BF16, and projects it.
    ///
    /// # Safety
    ///
    /// `attention` covers `batch * 4_096` FP32 values and is mutable scratch;
    /// `qkv` covers `batch * 10_240` BF16 values; `activation` covers
    /// `batch * 4_096` BF16 values; weights cover packed E2M1 `[4_096,4_096]`
    /// plus swizzled E4M3 `[4_096,256]`; and `output` covers
    /// `batch * 4_096` BF16 values. Four-byte-loaded planes are aligned,
    /// disjoint, context-local, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        attention: *mut f32,
        qkv: *const u16,
        activation: *mut u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !weight_scale_divisor.is_finite() || weight_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 NVFP4 attention-output weight scale divisor must be finite and positive",
            ));
        }
        let reciprocal = 1.0 / weight_scale_divisor;

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        attention,
                        qkv,
                        activation,
                        weight_codes,
                        weight_scales,
                        reciprocal,
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
                "Qwen3.5 NVFP4 attention-output batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

/// PTX symbols retained for every exact Qwen3.5 attention-output stage.
pub(crate) fn qwen35_nvfp4_attention_output_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_attention_output_gate_bf16_ptx_name::<8>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_attention_output_a16_ptx_name::<8>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        GROUPS_PER_ROW, MAX_BATCH, OUTPUT_ROWS, PHASES, SHARED_U32, WARPS,
        qwen35_nvfp4_attention_output_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(OUTPUT_ROWS, 4_096);
        assert_eq!(GROUPS_PER_ROW, 256);
        assert_eq!(PHASES, 8);
        assert_eq!(OUTPUT_ROWS / (2 * WARPS), 256);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);

        let names = qwen35_nvfp4_attention_output_ptx_names();
        assert_eq!(names.len(), 2 * MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

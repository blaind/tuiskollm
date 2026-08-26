//! Exact-target NVFP4 gate/up projection with fused SwiGLU.

use crate::Sm86Arch;
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_simt::{e2m1x2_to_f32, e4m3_to_f32, f32_to_bf16};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::INTERMEDIATE;
const GATE_UP_ROWS: usize = 2 * OUTPUT_ROWS;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = HIDDEN / GROUP_K;
const CODE_BYTES_PER_ROW: usize = HIDDEN / 2;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();
const SCALE_TILES_PER_ROW: usize = GROUPS_PER_ROW / 4;

// One warp retains one complete gate/up row-pair reduction. The 32 lanes cover
// 32 groups per phase, so ten phases cover all 320 groups without changing the
// per-output reduction owner. Eight warps/CTA therefore produce eight row pairs,
// and 17,408 / 8 = 2,176 CTAs provide more than 26 waves on an 82-SM RTX 3090.
// The B=2..8 entries stage each 512-value phase once for all eight row pairs,
// removing eight identical activation reads while preserving every lane's group
// order and the warp reduction order. This is the first exact A16 feasibility
// schedule; target measurements, not this topology, decide any later retune.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = 32 * GROUP_K / 2;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

const _: () = assert!(HIDDEN == 5_120);
const _: () = assert!(OUTPUT_ROWS == 17_408);
const _: () = assert!(GATE_UP_ROWS == 34_816);
const _: () = assert!(GROUPS_PER_ROW == 320);
const _: () = assert!(10 * CODE_WORDS_PER_PHASE == CODE_BYTES_PER_ROW / size_of::<u32>());
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

        (persistent_tile * SCALE_TILES_PER_ROW + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset(parent_row: usize, group: usize) -> usize {
        weight_scale_offset(parent_row, group >> 2) + (group & 3)
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
    fn silu(value: f32) -> f32 {
        let exponent = -value.abs() * core::f32::consts::LOG2_E;
        let exp_negative_absolute = if exponent < -126.0 {
            0.0
        } else {
            float::ex2_approx_f32(exponent)
        };

        if value >= 0.0 {
            value / (1.0 + exp_negative_absolute)
        } else {
            value * exp_negative_absolute / (1.0 + exp_negative_absolute)
        }
    }

    #[inline(always)]
    unsafe fn swiglu_body<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
        shared: *mut u32,
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
        let up_row = gate_row + OUTPUT_ROWS;
        let mut gate_accumulators = [0.0f32; TOKENS];
        let mut up_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < 10 {
            let mut task = tid;
            while task < TOKENS * PHASE_PACKED_PAIRS {
                let token = task / PHASE_PACKED_PAIRS;
                let pair = task - token * PHASE_PACKED_PAIRS;
                // SAFETY: every exact route supplies `TOKENS` complete input rows.
                unsafe {
                    *shared.add(task) = *input.add(token * (HIDDEN / 2) + phase * 256 + pair);
                }
                task += THREADS as usize;
            }
            thread::sync_threads();

            let group = phase * 32 + lane;
            // SAFETY: source validation admitted one swizzled scale per logical group.
            let gate_scale = unsafe {
                load_u8_read_only(weight_scales.add(weight_group_scale_offset(gate_row, group)))
            };
            // SAFETY: the up plane follows the gate plane in the fused source owner.
            let up_scale = unsafe {
                load_u8_read_only(weight_scales.add(weight_group_scale_offset(up_row, group)))
            };
            let gate_coefficient = e4m3_to_f32(gate_scale) * weight_scale_reciprocal;
            let up_coefficient = e4m3_to_f32(up_scale) * weight_scale_reciprocal;
            let row_words = CODE_BYTES_PER_ROW / 4;
            let gate_source = unsafe {
                weight_codes.add(gate_row * row_words + phase * CODE_WORDS_PER_PHASE + lane * 2)
            };
            let up_source = unsafe {
                weight_codes.add(up_row * row_words + phase * CODE_WORDS_PER_PHASE + lane * 2)
            };
            // SAFETY: one logical group contains exactly two packed u32 words.
            let gate_words = unsafe { load_u32x2_read_only(gate_source) };
            // SAFETY: one logical group contains exactly two packed u32 words.
            let up_words = unsafe { load_u32x2_read_only(up_source) };

            macro_rules! accumulate_pair {
                ($pair:literal) => {{
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
                    let (gate_weight0, gate_weight1) = e2m1x2_to_f32(gate_packed);
                    let (up_weight0, up_weight1) = e2m1x2_to_f32(up_packed);

                    macro_rules! accumulate_token {
                        ($token:literal) => {
                            if $token < TOKENS {
                                let bits = unsafe {
                                    *shared.add($token * PHASE_PACKED_PAIRS + lane * 8 + $pair)
                                };
                                let (activation0, activation1) = convert::cvt_f32x2_bf16x2(bits);
                                gate_accumulators[$token] = float::fma_rn_f32(
                                    gate_weight0 * gate_coefficient,
                                    activation0,
                                    gate_accumulators[$token],
                                );
                                gate_accumulators[$token] = float::fma_rn_f32(
                                    gate_weight1 * gate_coefficient,
                                    activation1,
                                    gate_accumulators[$token],
                                );
                                up_accumulators[$token] = float::fma_rn_f32(
                                    up_weight0 * up_coefficient,
                                    activation0,
                                    up_accumulators[$token],
                                );
                                up_accumulators[$token] = float::fma_rn_f32(
                                    up_weight1 * up_coefficient,
                                    activation1,
                                    up_accumulators[$token],
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
                    let gate = reduce_sum_lane0(gate_accumulators[$token]);
                    let up = reduce_sum_lane0(up_accumulators[$token]);

                    if lane == 0 {
                        // SAFETY: one lane writes one unique token/output row value.
                        unsafe {
                            *output.add($token * OUTPUT_ROWS + gate_row) =
                                f32_to_bf16(silu(gate) * up);
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

    /// Projects the singleton BF16 activation through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 6),
    )]
    pub fn nvfp4_swiglu_a16_b1(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            swiglu_body::<1>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    /// Projects `TOKENS` BF16 activations through represented NVFP4 weights.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 6),
    )]
    pub fn nvfp4_swiglu_a16<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;
        let _ = A::HIDDEN;

        unsafe {
            swiglu_body::<TOKENS>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }
}

fn launch_config() -> LaunchConfig1D {
    LaunchConfig1D::new((OUTPUT_ROWS / WARPS) as u32, THREADS, 0)
}

tuisko_kernels_simt::nvfp4_a16_batch_routes! {
    label = "SM86 NVFP4 A16",
    b1 = { __nvfp4_swiglu_a16_b1_CudaKernel, prepare_nvfp4_swiglu_a16_b1, nvfp4_swiglu_a16_b1 },
    batched = { __nvfp4_swiglu_a16_CudaKernel, prepare_nvfp4_swiglu_a16, nvfp4_swiglu_a16 },
}

/// PTX symbols retained for every exact SM86 NVFP4 SwiGLU batch.
pub(crate) fn nvfp4_swiglu_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        "nvfp4_swiglu_a16_b1",
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 2>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 3>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 4>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 5>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 6>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 7>(),
        kernels::nvfp4_swiglu_a16_ptx_name::<Qwen38_27B, 8>(),
    ]
}

/// Prepared A16 routes consuming the exact source NVFP4 gate/up owner on SM86.
pub struct Nvfp4SwiGluOp<A: Sm86Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedBatchOneRoute,
    b2: PreparedBatchRoute<A, 2>,
    b3: PreparedBatchRoute<A, 3>,
    b4: PreparedBatchRoute<A, 4>,
    b5: PreparedBatchRoute<A, 5>,
    b6: PreparedBatchRoute<A, 6>,
    b7: PreparedBatchRoute<A, 7>,
    b8: PreparedBatchRoute<A, 8>,
}

impl<A: Sm86Arch> Nvfp4SwiGluOp<A> {
    /// Loads the embedded SM86 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = nvfp4_swiglu_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the SM86 NVFP4 SwiGLU module", source))?;

        Ok(Self {
            b1: PreparedBatchOneRoute::prepare(&module)?,
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

    /// Executes the represented-weight A16 route for exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 5_120` BF16 values; `weight_codes` covers the
    /// fused packed `[34_816, 5_120]` E2M1 plane; `weight_scales` covers its
    /// swizzled `[34_816, 320]` E4M3 plane; and `output` covers
    /// `batch * 17_408` BF16 values. Four-byte-loaded planes are four-byte
    /// aligned. The divisor is finite and positive. Allocations belong to
    /// `stream`'s context, remain live through completion, and do not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weight_codes: *const u8,
        weight_scales: *const u8,
        weight_scale_divisor: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if !weight_scale_divisor.is_finite() || weight_scale_divisor <= 0.0 {
            return Err(GpuError::invalid_launch(
                "NVFP4 weight scale divisor must be finite and positive",
            ));
        }

        let reciprocal = 1.0 / weight_scale_divisor;
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
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
                "NVFP4 SwiGLU batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_WORDS_PER_PHASE, GATE_UP_ROWS, GROUPS_PER_ROW, MAX_BATCH, OUTPUT_ROWS, SHARED_U32,
        nvfp4_swiglu_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn exact_geometry_matches_the_source_owner() {
        assert_eq!(OUTPUT_ROWS, 17_408);
        assert_eq!(GATE_UP_ROWS, 34_816);
        assert_eq!(GROUPS_PER_ROW, 320);
        assert_eq!(CODE_WORDS_PER_PHASE, 64);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);
    }

    #[test]
    fn inventory_has_one_distinct_entry_per_batch() {
        let names = nvfp4_swiglu_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(unique.len(), names.len());
    }
}

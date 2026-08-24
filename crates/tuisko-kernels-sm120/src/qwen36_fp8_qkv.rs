//! Exact Qwen3.6 static-FP8 full-attention QKV projection.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen36Moe35B::HIDDEN;
const QUERY_ROWS: usize = Qwen36Moe35B::ATTENTION_QUERY_ROWS;
const KV_ROWS: usize = Qwen36Moe35B::ATTENTION_KV_ROWS;
const PROJECTED_ROWS: usize = Qwen36Moe35B::ATTENTION_QKV_ROWS;
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;

const _: () = assert!(INPUT_COLUMNS == 2_048);
const _: () = assert!(QUERY_ROWS == 8_192);
const _: () = assert!(KV_ROWS == 512);
const _: () = assert!(PROJECTED_ROWS == 9_216);
const _: () = assert!(QUERY_ROWS.is_multiple_of(2 * WARPS));
const _: () = assert!(KV_ROWS.is_multiple_of(2 * WARPS));
const _: () = assert!(PROJECTED_ROWS.is_multiple_of(2 * WARPS));

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, thread, warp};

    #[inline(always)]
    fn e4m3x2_to_f32(packed: u16) -> (f32, f32) {
        let packed_f16 = convert::cvt_rn_f16x2_e4m3x2(packed);

        convert::cvt_f32x2_f16x2(packed_f16)
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

    /// Quantizes exact BF16 rows with the checkpoint's shared static FP8 scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_attention_fp8_quantize<const TOKENS: usize>(
        input: *const u32,
        input_scale: f32,
        codes: *mut u16,
    ) {
        let token = thread::blockIdx_x() as usize;
        if token >= TOKENS {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let pairs = INPUT_COLUMNS / 2;
        let input = unsafe { input.add(token * pairs) };
        let codes = unsafe { codes.add(token * pairs) };
        let mut pair = tid;

        while pair < pairs {
            let (low, high) = convert::cvt_f32x2_bf16x2(unsafe { *input.add(pair) });
            unsafe {
                *codes.add(pair) = convert::cvt_rn_satfinite_e4m3x2_f32(
                    float::div_rn_f32(low, input_scale),
                    float::div_rn_f32(high, input_scale),
                );
            }
            pair += THREADS as usize;
        }
    }

    /// Projects static E4M3 activations through fused Q/gate, K, and V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_fp8_qkv<const TOKENS: usize>(
        activation_codes: *const u16,
        input_scale: f32,
        weight_codes: *const u16,
        query_weight_scale: f32,
        key_weight_scale: f32,
        value_weight_scale: f32,
        output: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let first_row = (thread::blockIdx_x() as usize * WARPS + (tid >> 5)) * 2;
        let pairs = INPUT_COLUMNS / 2;
        let first_weight = unsafe { weight_codes.add(first_row * pairs) };
        let second_weight = unsafe { first_weight.add(pairs) };
        let weight_scale = if first_row < QUERY_ROWS {
            query_weight_scale
        } else if first_row < QUERY_ROWS + KV_ROWS {
            key_weight_scale
        } else {
            value_weight_scale
        };
        let mut first_sums = [0.0f32; TOKENS];
        let mut second_sums = [0.0f32; TOKENS];
        let mut pair = lane;

        while pair < pairs {
            let first = e4m3x2_to_f32(unsafe { *first_weight.add(pair) });
            let second = e4m3x2_to_f32(unsafe { *second_weight.add(pair) });

            macro_rules! accumulate {
                ($token:literal) => {
                    if TOKENS > $token {
                        let activation =
                            e4m3x2_to_f32(unsafe { *activation_codes.add($token * pairs + pair) });
                        first_sums[$token] =
                            float::fma_rn_f32(first.0, activation.0, first_sums[$token]);
                        first_sums[$token] =
                            float::fma_rn_f32(first.1, activation.1, first_sums[$token]);
                        second_sums[$token] =
                            float::fma_rn_f32(second.0, activation.0, second_sums[$token]);
                        second_sums[$token] =
                            float::fma_rn_f32(second.1, activation.1, second_sums[$token]);
                    }
                };
            }

            accumulate!(0);
            accumulate!(1);
            accumulate!(2);
            accumulate!(3);
            accumulate!(4);
            accumulate!(5);
            accumulate!(6);
            accumulate!(7);
            pair += 32;
        }

        macro_rules! store {
            ($token:literal) => {
                if TOKENS > $token {
                    let first =
                        reduce_sum_lane_zero(first_sums[$token]) * input_scale * weight_scale;
                    let second =
                        reduce_sum_lane_zero(second_sums[$token]) * input_scale * weight_scale;
                    if lane == 0 {
                        unsafe {
                            *output.add($token * PROJECTED_ROWS + first_row) =
                                tcgen05::cvt_f32x2_bf16x2(first, 0.0) as u16;
                            *output.add($token * PROJECTED_ROWS + first_row + 1) =
                                tcgen05::cvt_f32x2_bf16x2(second, 0.0) as u16;
                        }
                    }
                }
            };
        }

        store!(0);
        store!(1);
        store!(2);
        store!(3);
        store!(4);
        store!(5);
        store!(6);
        store!(7);
    }
}

struct PreparedBatchRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__qwen36_attention_fp8_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen36_fp8_qkv_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        // One CTA owns one 2,048-wide token. Its 256 threads each retain four
        // ordered BF16 pairs, so exact-B changes only the number of CTAs.
        let quantize = module
            .prepare_qwen36_attention_fp8_quantize::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.6 attention FP8 quantization", source)
            })?;
        // The 18.9 MB Q/K/V plane is latency-bound at decode. Spreading 9,216
        // rows over 576 CTAs exposes more than three CTAs per 170-SM RTX 5090;
        // each warp still owns the same adjacent row pair, 32 ordered E4M3
        // pairs per lane, and the same warp reduction, so arithmetic is unchanged.
        let projection = module
            .prepare_qwen36_fp8_qkv::<TOKENS>(LaunchConfig1D::new(
                (PROJECTED_ROWS / (2 * WARPS)) as u32,
                THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing Qwen3.6 FP8 QKV", source))?;

        Ok(Self {
            quantize,
            projection,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        activation_codes: *mut u8,
        input_scale: f32,
        weight_codes: *const u8,
        query_weight_scale: f32,
        key_weight_scale: f32,
        value_weight_scale: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_attention_fp8_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                input_scale,
                activation_codes.cast::<u16>(),
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 attention FP8 quantization", source)
            })?;
        module
            .qwen36_fp8_qkv::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u16>(),
                input_scale,
                weight_codes.cast::<u16>(),
                query_weight_scale,
                key_weight_scale,
                value_weight_scale,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 QKV", source))
    }
}

/// PTX symbols retained for every exact Qwen3.6 full-attention QKV route.
pub(crate) fn qwen36_fp8_qkv_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_attention_fp8_quantize_ptx_name::<1>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<2>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<3>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<4>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<5>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<6>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<7>(),
        kernels::qwen36_attention_fp8_quantize_ptx_name::<8>(),
        kernels::qwen36_fp8_qkv_ptx_name::<1>(),
        kernels::qwen36_fp8_qkv_ptx_name::<2>(),
        kernels::qwen36_fp8_qkv_ptx_name::<3>(),
        kernels::qwen36_fp8_qkv_ptx_name::<4>(),
        kernels::qwen36_fp8_qkv_ptx_name::<5>(),
        kernels::qwen36_fp8_qkv_ptx_name::<6>(),
        kernels::qwen36_fp8_qkv_ptx_name::<7>(),
        kernels::qwen36_fp8_qkv_ptx_name::<8>(),
    ]
}

/// Prepared exact-batch Qwen3.6 full-attention QKV routes on SM120.
pub struct Qwen36Fp8QkvOp {
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

impl Qwen36Fp8QkvOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_fp8_qkv_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 FP8 QKV kernels", source))?;

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

    /// Runs the shared-static-FP8 Q/gate, K, and V projection at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// The input covers BF16 `[batch,2048]`; activation workspace covers E4M3
    /// `[batch,2048]`; weights cover E4M3 `[9216,2048]`; and output covers BF16
    /// `[batch,9216]`. All planes are aligned, disjoint, context-local, and live
    /// until `stream` completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        activation_codes: *mut u8,
        input_scale: f32,
        weight_codes: *const u8,
        query_weight_scale: f32,
        key_weight_scale: f32,
        value_weight_scale: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if [
            input_scale,
            query_weight_scale,
            key_weight_scale,
            value_weight_scale,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 attention FP8 scales must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        input,
                        activation_codes,
                        input_scale,
                        weight_codes,
                        query_weight_scale,
                        key_weight_scale,
                        value_weight_scale,
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
                "Qwen3.6 attention QKV batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(INPUT_COLUMNS, 2_048);
        assert_eq!(QUERY_ROWS, 8_192);
        assert_eq!(KV_ROWS, 512);
        assert_eq!(PROJECTED_ROWS, 9_216);

        let names = qwen36_fp8_qkv_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(names.len(), 16);
        assert_eq!(unique.len(), names.len());
    }
}

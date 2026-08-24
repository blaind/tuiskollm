//! Exact Qwen3.6 static-FP8 GDN output projection.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen36Moe35B::GDN_VALUE_ROWS;
const OUTPUT_ROWS: usize = Qwen36Moe35B::HIDDEN;
const QUANTIZE_THREADS: u32 = 256;
const PROJECTION_WARPS: usize = 4;
const PROJECTION_THREADS: u32 = (PROJECTION_WARPS * 32) as u32;
const ROWS_PER_CTA: usize = 2 * PROJECTION_WARPS;

const _: () = assert!(INPUT_COLUMNS == 4_096);
const _: () = assert!(OUTPUT_ROWS == 2_048);
const _: () = assert!(INPUT_COLUMNS.is_multiple_of(2 * QUANTIZE_THREADS as usize));
const _: () = assert!(OUTPUT_ROWS.is_multiple_of(ROWS_PER_CTA));

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, tcgen05, thread, warp};

    #[inline(always)]
    fn e4m3x2_to_f32(packed: u16) -> (f32, f32) {
        convert::cvt_f32x2_f16x2(convert::cvt_rn_f16x2_e4m3x2(packed))
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

    /// Quantizes exact BF16 recurrence rows with the checkpoint's static scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_gdn_output_static_quantize<const TOKENS: usize>(
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
            pair += QUANTIZE_THREADS as usize;
        }
    }

    /// Projects static E4M3 recurrence rows to the residual width.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_gdn_output_projection<const TOKENS: usize>(
        activation_codes: *const u16,
        input_scale: f32,
        weight_codes: *const u16,
        weight_scale: f32,
        output: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let first_row = (thread::blockIdx_x() as usize * PROJECTION_WARPS + (tid >> 5)) * 2;
        let pairs = INPUT_COLUMNS / 2;
        let first_weight = unsafe { weight_codes.add(first_row * pairs) };
        let second_weight = unsafe { first_weight.add(pairs) };
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
                            *output.add($token * OUTPUT_ROWS + first_row) =
                                tcgen05::cvt_f32x2_bf16x2(first, 0.0) as u16;
                            *output.add($token * OUTPUT_ROWS + first_row + 1) =
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

struct PreparedRoute<const TOKENS: usize> {
    quantize: PreparedLaunch<kernels::__qwen36_gdn_output_static_quantize_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen36_gdn_output_projection_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        // Each 4,096-wide token gives 256 threads exactly eight packed BF16
        // pairs; exact-B changes only the number of independent CTAs.
        let quantize = module
            .prepare_qwen36_gdn_output_static_quantize::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                QUANTIZE_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.6 GDN output quantization", source)
            })?;
        // The 8 MiB weight plane would expose only 128 CTAs with the existing
        // eight-warp/two-row topology. Four warps produce 256 CTAs over 170
        // SMs while every warp keeps the same adjacent row pair, 64 ordered
        // pairs per lane, and reduction order; arithmetic is unchanged.
        let projection = module
            .prepare_qwen36_gdn_output_projection::<TOKENS>(LaunchConfig1D::new(
                (OUTPUT_ROWS / ROWS_PER_CTA) as u32,
                PROJECTION_THREADS,
                0,
            ))
            .map_err(|source| GpuError::launch("preparing Qwen3.6 GDN output", source))?;

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
        weight_scale: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_gdn_output_static_quantize::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                input_scale,
                activation_codes.cast::<u16>(),
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 GDN output quantization", source)
            })?;
        module
            .qwen36_gdn_output_projection::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u16>(),
                input_scale,
                weight_codes.cast::<u16>(),
                weight_scale,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 GDN output", source))
    }
}

/// PTX symbols retained for every exact Qwen3.6 GDN output route.
pub(crate) fn qwen36_gdn_output_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<1>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<2>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<3>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<4>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<5>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<6>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<7>(),
        kernels::qwen36_gdn_output_static_quantize_ptx_name::<8>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<1>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<2>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<3>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<4>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<5>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<6>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<7>(),
        kernels::qwen36_gdn_output_projection_ptx_name::<8>(),
    ]
}

/// Prepared exact-batch Qwen3.6 GDN output routes on SM120.
pub struct Qwen36GdnOutputOp {
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

impl Qwen36GdnOutputOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen36_gdn_output_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 GDN output kernels", source))?;

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

    /// Runs static-FP8 recurrent output projection at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// The input covers BF16 `[batch,4096]`, the code workspace covers E4M3
    /// `[batch,4096]`, weights cover E4M3 `[2048,4096]`, and output covers
    /// BF16 `[batch,2048]`. All planes are aligned, disjoint, context-local,
    /// and live until `stream` completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        activation_codes: *mut u8,
        input_scale: f32,
        weight_codes: *const u8,
        weight_scale: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        if [input_scale, weight_scale]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 GDN output FP8 scales must be finite and positive",
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
                        weight_scale,
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
                "Qwen3.6 GDN output batch {batch} is outside the exact range 1..={MAX_BATCH}"
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
        assert_eq!(INPUT_COLUMNS, 4_096);
        assert_eq!(OUTPUT_ROWS, 2_048);
        assert_eq!(PROJECTION_THREADS, 128);
        assert_eq!(OUTPUT_ROWS / ROWS_PER_CTA, 256);

        let names = qwen36_gdn_output_ptx_names();
        assert_eq!(names.len(), 16);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

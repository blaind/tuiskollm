//! Exact Qwen3.6 static-FP8 GDN output projection.

use crate::device::fp8_projection::prefill_projection_mma_static_scales;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen36Moe35B::GDN_VALUE_ROWS;
const OUTPUT_ROWS: usize = Qwen36Moe35B::HIDDEN;
const QUANTIZE_THREADS: u32 = 256;
// One output row per warp exposes 512 CTAs for the 2,048-row decode projection,
// enough to cover 170 SMs while preserving each row's stride-32 FMA order.
const PROJECTION_WARPS: usize = 4;
const PROJECTION_THREADS: u32 = (PROJECTION_WARPS * 32) as u32;
const ROWS_PER_CTA: usize = PROJECTION_WARPS;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const PREFILL_BLOCK_ROWS: usize = 64;
const PREFILL_OUTPUT_ROWS: usize = 64;
const PREFILL_K_WORDS: usize = 32;
const PREFILL_K_SUBTILES: usize = 4;
const PREFILL_THREADS: u32 = 256;
const PREFILL_SHARED_BYTES: u32 =
    (2 * (PREFILL_BLOCK_ROWS + PREFILL_OUTPUT_ROWS) * PREFILL_K_WORDS * size_of::<u32>()) as u32;

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

    #[inline(always)]
    unsafe fn static_fp8_quantize<const TOKENS: usize>(
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
        unsafe { static_fp8_quantize::<TOKENS>(input, input_scale, codes) }
    }

    /// Quantizes exact BF16 prompt rows with the checkpoint's static scale.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_gdn_output_static_quantize_prefill<const TOKENS: usize>(
        input: *const u32,
        input_scale: f32,
        codes: *mut u16,
    ) {
        unsafe { static_fp8_quantize::<TOKENS>(input, input_scale, codes) }
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
        let row = thread::blockIdx_x() as usize * PROJECTION_WARPS + (tid >> 5);
        let pairs = INPUT_COLUMNS / 2;
        let row_weight = unsafe { weight_codes.add(row * pairs) };
        let mut sums = [0.0f32; TOKENS];
        let mut pair = lane;

        while pair < pairs {
            let weight = e4m3x2_to_f32(unsafe { *row_weight.add(pair) });

            macro_rules! accumulate {
                ($token:literal) => {
                    if TOKENS > $token {
                        let activation =
                            e4m3x2_to_f32(unsafe { *activation_codes.add($token * pairs + pair) });
                        sums[$token] = float::fma_rn_f32(weight.0, activation.0, sums[$token]);
                        sums[$token] = float::fma_rn_f32(weight.1, activation.1, sums[$token]);
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
                    let value = reduce_sum_lane_zero(sums[$token]) * input_scale * weight_scale;
                    if lane == 0 {
                        unsafe {
                            *output.add($token * OUTPUT_ROWS + row) =
                                tcgen05::cvt_f32x2_bf16x2(value, 0.0) as u16;
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

    /// Projects one exact prompt tile through source-static E4M3 output weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 32768,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_gdn_output_projection_prefill<const TOKENS: usize>(
        activation_codes: *const u32,
        input_scale: f32,
        weight_codes: *const u32,
        weight_scale: f32,
        output: *mut u16,
        k_tiles: u32,
    ) {
        // SAFETY: the exact 64x64 CTA inventory covers all active prompt/output tiles.
        unsafe {
            prefill_projection_mma_static_scales::<
                INPUT_COLUMNS,
                TOKENS,
                PREFILL_BLOCK_ROWS,
                PREFILL_K_WORDS,
                PREFILL_K_SUBTILES,
            >(
                activation_codes,
                input_scale,
                weight_codes,
                OUTPUT_ROWS,
                weight_scale,
                weight_scale,
                output,
                k_tiles,
                OUTPUT_ROWS,
            );
        }
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
        // Four one-row warps produce 512 CTAs over 170 SMs. Each warp retains
        // its row's 64 ordered pairs per lane and the original reduction order.
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

struct PreparedPrefillRoute<const TOKENS: usize> {
    quantize:
        PreparedLaunch<kernels::__qwen36_gdn_output_static_quantize_prefill_CudaKernel<TOKENS>>,
    projection: PreparedLaunch<kernels::__qwen36_gdn_output_projection_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_ROUTES.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 GDN output prefill route T={TOKENS} is not admitted"
            )));
        }
        let quantize = module
            .prepare_qwen36_gdn_output_static_quantize_prefill::<TOKENS>(LaunchConfig1D::new(
                TOKENS as u32,
                QUANTIZE_THREADS,
                0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing Qwen3.6 GDN output prefill quantization", source)
            })?;
        let token_tiles = TOKENS.div_ceil(PREFILL_BLOCK_ROWS);
        let projection_blocks = OUTPUT_ROWS / PREFILL_OUTPUT_ROWS * token_tiles;
        // At T=128 the decode topology would scan the 8 MiB weight plane 16
        // times. Two 64-token MMA tiles scan it twice instead, while every
        // output retains the ordered m16n8k32 K sequence and exact scalar scales.
        let projection = module
            .prepare_qwen36_gdn_output_projection_prefill::<TOKENS>(LaunchConfig1D::new(
                projection_blocks as u32,
                PREFILL_THREADS,
                PREFILL_SHARED_BYTES,
            ))
            .map_err(|source| GpuError::launch("preparing Qwen3.6 GDN output prefill", source))?;

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
            .qwen36_gdn_output_static_quantize_prefill::<TOKENS>(
                stream,
                &self.quantize,
                input.cast::<u32>(),
                input_scale,
                activation_codes.cast::<u16>(),
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 GDN output prefill quantization", source)
            })?;
        module
            .qwen36_gdn_output_projection_prefill::<TOKENS>(
                stream,
                &self.projection,
                activation_codes.cast::<u32>(),
                input_scale,
                weight_codes.cast::<u32>(),
                weight_scale,
                output,
                (INPUT_COLUMNS / 4 / PREFILL_K_WORDS) as u32,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 GDN output prefill", source))
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
        kernels::qwen36_gdn_output_static_quantize_prefill_ptx_name::<32>(),
        kernels::qwen36_gdn_output_static_quantize_prefill_ptx_name::<64>(),
        kernels::qwen36_gdn_output_static_quantize_prefill_ptx_name::<128>(),
        kernels::qwen36_gdn_output_projection_prefill_ptx_name::<32>(),
        kernels::qwen36_gdn_output_projection_prefill_ptx_name::<64>(),
        kernels::qwen36_gdn_output_projection_prefill_ptx_name::<128>(),
    ]
}

/// Prepared exact-row Qwen3.6 GDN output routes on SM120.
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
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
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
            t32: PreparedPrefillRoute::prepare(&module)?,
            t64: PreparedPrefillRoute::prepare(&module)?,
            t128: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Runs static-FP8 recurrent output projection at one exact row count.
    ///
    /// # Safety
    ///
    /// The input covers BF16 `[rows,4096]`, the code workspace covers E4M3
    /// `[rows,4096]`, weights cover E4M3 `[2048,4096]`, and output covers
    /// BF16 `[rows,2048]`. All planes are aligned, disjoint, context-local,
    /// and live until `stream` completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
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
            _ => Err(GpuError::invalid_launch(format!(
                "Qwen3.6 GDN output row count {rows} is outside 1..={MAX_BATCH}, 32, 64, and 128"
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
        assert_eq!(OUTPUT_ROWS / ROWS_PER_CTA, 512);
        assert_eq!(PREFILL_ROUTES, [32, 64, 128]);
        assert_eq!(PREFILL_SHARED_BYTES, 32_768);

        let names = qwen36_gdn_output_ptx_names();
        assert_eq!(names.len(), 22);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

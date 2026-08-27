//! Exact Qwen3.5 BF16 language-model head.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen35_9B};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen35_9B::VOCAB;
// The full head streams 2,034,237,440 BF16 weight bytes per invocation. The
// retained LM-head topology emits 16 rows per CTA: 15,520 CTAs, or 91.3 per
// SM on the 170-SM target. Each lane accumulates the same eight adjacent K
// values per phase, so changing this width changes the numerical reduction.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const VALUES_PER_LANE: usize = 8;
const VALUES_PER_PHASE: usize = 32 * VALUES_PER_LANE;
const WORDS_PER_LANE: usize = VALUES_PER_LANE / 2;
const PHASES: usize = INPUT_COLUMNS / VALUES_PER_PHASE;

const _: () = assert!(INPUT_COLUMNS == 4_096);
const _: () = assert!(OUTPUT_ROWS == 248_320);
const _: () = assert!(OUTPUT_ROWS.is_multiple_of(2 * WARPS));
const _: () = assert!(INPUT_COLUMNS.is_multiple_of(VALUES_PER_PHASE));

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, ptx_asm, tcgen05, thread, warp};

    #[inline(always)]
    unsafe fn load_u32x4_read_only(source: *const u32) -> (u32, u32, u32, u32) {
        let first: u32;
        let second: u32;
        let third: u32;
        let fourth: u32;

        unsafe {
            ptx_asm!(
                "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];",
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
    fn reduce_sum_lane_zero(mut value: f32) -> f32 {
        value += warp::shuffle_down_f32(value, 16);
        value += warp::shuffle_down_f32(value, 8);
        value += warp::shuffle_down_f32(value, 4);
        value += warp::shuffle_down_f32(value, 2);
        value += warp::shuffle_down_f32(value, 1);

        value
    }

    #[inline(always)]
    unsafe fn bf16_lm_head_body<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        output: *mut u16,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let first_row = (thread::blockIdx_x() as usize * WARPS + warp_index) * 2;
        let words_per_row = INPUT_COLUMNS / 2;
        let first_weight = unsafe { weights.add(first_row * words_per_row) };
        let second_weight = unsafe { first_weight.add(words_per_row) };
        let mut first_sums = [0.0f32; TOKENS];
        let mut second_sums = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < PHASES {
            let lane_offset = phase * (VALUES_PER_PHASE / 2) + lane * WORDS_PER_LANE;
            let first_words = unsafe { load_u32x4_read_only(first_weight.add(lane_offset)) };
            let second_words = unsafe { load_u32x4_read_only(second_weight.add(lane_offset)) };
            let activation0 = unsafe { load_u32x4_read_only(input.add(lane_offset)) };
            let activation1 = if TOKENS > 1 {
                unsafe { load_u32x4_read_only(input.add(words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation2 = if TOKENS > 2 {
                unsafe { load_u32x4_read_only(input.add(2 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation3 = if TOKENS > 3 {
                unsafe { load_u32x4_read_only(input.add(3 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation4 = if TOKENS > 4 {
                unsafe { load_u32x4_read_only(input.add(4 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation5 = if TOKENS > 5 {
                unsafe { load_u32x4_read_only(input.add(5 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation6 = if TOKENS > 6 {
                unsafe { load_u32x4_read_only(input.add(6 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };
            let activation7 = if TOKENS > 7 {
                unsafe { load_u32x4_read_only(input.add(7 * words_per_row + lane_offset)) }
            } else {
                (0, 0, 0, 0)
            };

            macro_rules! word {
                ($words:ident, $index:literal) => {
                    match $index {
                        0 => $words.0,
                        1 => $words.1,
                        2 => $words.2,
                        _ => $words.3,
                    }
                };
            }

            macro_rules! accumulate_token {
                ($token:literal, $activation:ident, $index:literal, $first:ident, $second:ident) => {
                    if TOKENS > $token {
                        let (low, high) = convert::cvt_f32x2_bf16x2(word!($activation, $index));
                        first_sums[$token] = float::fma_rn_f32($first.0, low, first_sums[$token]);
                        first_sums[$token] = float::fma_rn_f32($first.1, high, first_sums[$token]);
                        second_sums[$token] =
                            float::fma_rn_f32($second.0, low, second_sums[$token]);
                        second_sums[$token] =
                            float::fma_rn_f32($second.1, high, second_sums[$token]);
                    }
                };
            }

            macro_rules! accumulate_word {
                ($index:literal) => {{
                    let first = convert::cvt_f32x2_bf16x2(word!(first_words, $index));
                    let second = convert::cvt_f32x2_bf16x2(word!(second_words, $index));
                    accumulate_token!(0, activation0, $index, first, second);
                    accumulate_token!(1, activation1, $index, first, second);
                    accumulate_token!(2, activation2, $index, first, second);
                    accumulate_token!(3, activation3, $index, first, second);
                    accumulate_token!(4, activation4, $index, first, second);
                    accumulate_token!(5, activation5, $index, first, second);
                    accumulate_token!(6, activation6, $index, first, second);
                    accumulate_token!(7, activation7, $index, first, second);
                }};
            }

            accumulate_word!(0);
            accumulate_word!(1);
            accumulate_word!(2);
            accumulate_word!(3);
            phase += 1;
        }

        macro_rules! store_token {
            ($token:literal) => {
                if TOKENS > $token {
                    let first = reduce_sum_lane_zero(first_sums[$token]);
                    let second = reduce_sum_lane_zero(second_sums[$token]);
                    if lane == 0 {
                        unsafe {
                            *output.add($token * OUTPUT_ROWS + first_row) =
                                tcgen05::f32_to_bf16_rne(first);
                            *output.add($token * OUTPUT_ROWS + first_row + 1) =
                                tcgen05::f32_to_bf16_rne(second);
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

    /// Projects exact Qwen3.5 BF16 rows through the untied BF16 LM head.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_bf16_lm_head<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        output: *mut u16,
    ) {
        unsafe { bf16_lm_head_body::<TOKENS>(input, weights, output) };
    }
}

fn launch_config() -> LaunchConfig1D {
    LaunchConfig1D::new((OUTPUT_ROWS / (2 * WARPS)) as u32, THREADS, 0)
}

struct PreparedBatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_bf16_lm_head_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen35_bf16_lm_head::<TOKENS>(launch_config())
            .map_err(|source| GpuError::launch("preparing Qwen3.5 BF16 LM head", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weights: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_bf16_lm_head::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 BF16 LM head", source))
    }
}

/// PTX symbols retained for every exact Qwen3.5 BF16 LM-head batch.
pub(crate) fn qwen35_bf16_lm_head_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen35_bf16_lm_head_ptx_name::<1>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<2>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<3>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<4>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<5>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<6>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<7>(),
        kernels::qwen35_bf16_lm_head_ptx_name::<8>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen35_bf16_lm_head),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct Qwen35Bf16LmHeadRoutes {
    #[route(1)]
    b1: PreparedBatchRoute<1>,
    #[route(2)]
    b2: PreparedBatchRoute<2>,
    #[route(3)]
    b3: PreparedBatchRoute<3>,
    #[route(4)]
    b4: PreparedBatchRoute<4>,
    #[route(5)]
    b5: PreparedBatchRoute<5>,
    #[route(6)]
    b6: PreparedBatchRoute<6>,
    #[route(7)]
    b7: PreparedBatchRoute<7>,
    #[route(8)]
    b8: PreparedBatchRoute<8>,
}

/// Prepared exact-batch Qwen3.5 BF16 LM-head routes on SM120.
pub struct Qwen35Bf16LmHeadOp {
    module: kernels::LoadedModule,
    routes: Qwen35Bf16LmHeadRoutes,
}

impl Qwen35Bf16LmHeadOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_bf16_lm_head_ptx_names();
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the Qwen3.5 BF16 LM head", source))?;

        Ok(Self {
            routes: Qwen35Bf16LmHeadRoutes::prepare(&module)?,
            module,
        })
    }

    /// Projects represented BF16 activations through represented BF16 weights.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 4_096` BF16 values, `weights` covers BF16
    /// `[248_320, 4_096]`, and `output` covers `batch * 248_320` BF16 values.
    /// Four-byte-loaded planes are aligned, disjoint, and remain live in the
    /// stream's context until completion.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
        weights: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                unsafe { $route.launch(&self.module, stream, input, weights, output) }
            };
        }

        dispatch_qwen35_bf16_lm_head!(&self.routes, batch, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
                "Qwen3.5 BF16 LM-head batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))) )
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, OUTPUT_ROWS, PHASES, WARPS, qwen35_bf16_lm_head_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn geometry_and_inventory_are_exact() {
        assert_eq!(PHASES, 16);
        assert_eq!(OUTPUT_ROWS / (2 * WARPS), 15_520);

        let names = qwen35_bf16_lm_head_ptx_names();
        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            names.len()
        );
    }
}

//! Exact-target NVFP4 down projection.

use crate::Sm120Arch;
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const INPUT_COLUMNS: usize = Qwen38_27B::INTERMEDIATE;
const OUTPUT_ROWS: usize = HIDDEN;
const GROUP_K: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP_K;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const PHASE_GROUPS: usize = 32;
const PHASES: usize = GROUPS_PER_ROW / PHASE_GROUPS;
const CODE_WORDS_PER_PHASE: usize = 32 * (GROUP_K / 2) / size_of::<u32>();

// One warp retains two complete output-row reductions. The 32 lanes cover 32
// K16 groups per phase, so 34 phases cover all 1,088 groups without changing
// either output's reduction owner or lane order. Eight warps/CTA therefore
// produce 16 rows; 5,120 / 16 = 320 CTAs provide 1.88 blocks per 170-SM RTX 5090.
// Staging each 512-value phase once removes 15 duplicate activation reads per
// CTA while preserving both dot-product accumulation orders. Two resident
// 256-thread CTAs fit the measured register and 9,216-byte shared footprints.
// B=1 pairs the adjacent four-scale words owned by each row pair, reducing 16
// scale sectors per warp/phase to eight before subgroup broadcast.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const PHASE_PACKED_PAIRS: usize = 32 * GROUP_K / 2;
const SHARED_U32: usize = MAX_BATCH * PHASE_PACKED_PAIRS;

const _: () = assert!(HIDDEN == 5_120);
const _: () = assert!(INPUT_COLUMNS == 17_408);
const _: () = assert!(OUTPUT_ROWS == 5_120);
const _: () = assert!(GROUPS_PER_ROW == 1_088);
const _: () = assert!(Qwen35_9B::HIDDEN == 4_096);
const _: () = assert!(Qwen35_9B::INTERMEDIATE == 12_288);
const _: () = assert!(PHASES * CODE_WORDS_PER_PHASE == CODE_BYTES_PER_ROW / size_of::<u32>());
const _: () = assert!(SHARED_U32 * size_of::<u32>() == 8_192);

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{convert, float, warp};

    #[inline(always)]
    fn weight_scale_offset<A: Arch>(parent_row: usize, scale_tile: usize) -> usize {
        let persistent_tile = parent_row / 128;
        let row_in_tile = parent_row & 127;
        let row_mod32 = row_in_tile & 31;
        let row_quartile = row_in_tile >> 5;

        let scale_tiles_per_row = A::INTERMEDIATE / GROUP_K / 4;

        (persistent_tile * scale_tiles_per_row + scale_tile) * 512
            + row_mod32 * 16
            + row_quartile * 4
    }

    #[inline(always)]
    fn weight_group_scale_offset<A: Arch>(parent_row: usize, group: usize) -> usize {
        weight_scale_offset::<A>(parent_row, group >> 2) + (group & 3)
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
    unsafe fn down_body<A: Arch, const TOKENS: usize, const COALESCED_SCALES: bool>(
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
        let pair_index = block * (2 * WARPS) + 2 * warp_index;
        let first_row = physical_row(pair_index);
        let second_row = physical_row(pair_index + 1);
        let mut first_accumulators = [0.0f32; TOKENS];
        let mut second_accumulators = [0.0f32; TOKENS];
        let mut phase = 0usize;

        while phase < A::INTERMEDIATE / GROUP_K / PHASE_GROUPS {
            let mut task = tid;
            while task < TOKENS * PHASE_PACKED_PAIRS {
                let token = task / PHASE_PACKED_PAIRS;
                let pair = task - token * PHASE_PACKED_PAIRS;
                // SAFETY: every exact route supplies `TOKENS` complete input rows.
                unsafe {
                    *shared.add(task) =
                        *input.add(token * (A::INTERMEDIATE / 2) + phase * 256 + pair);
                }
                task += THREADS as usize;
            }
            thread::sync_threads();

            let group = phase * PHASE_GROUPS + lane;
            let (first_scale, second_scale) = if COALESCED_SCALES {
                let scale_lane = lane & 3;
                let mut first_word = 0u32;
                let mut second_word = 0u32;
                if scale_lane == 0 {
                    let offset = weight_scale_offset::<A>(first_row, group >> 2);
                    // SAFETY: paired physical rows own adjacent aligned four-scale words.
                    (first_word, second_word) =
                        unsafe { load_u32x2_read_only(weight_scales.add(offset).cast::<u32>()) };
                }
                let source_lane = (lane - scale_lane) as u32;
                first_word = warp::shuffle(first_word, source_lane);
                second_word = warp::shuffle(second_word, source_lane);
                let shift = scale_lane * 8;

                ((first_word >> shift) as u8, (second_word >> shift) as u8)
            } else {
                // SAFETY: source validation admitted one swizzled scale per logical group.
                let first = unsafe {
                    load_u8_read_only(
                        weight_scales.add(weight_group_scale_offset::<A>(first_row, group)),
                    )
                };
                let second = unsafe {
                    load_u8_read_only(
                        weight_scales.add(weight_group_scale_offset::<A>(second_row, group)),
                    )
                };

                (first, second)
            };
            let first_coefficient = e4m3_to_f32(first_scale) * weight_scale_reciprocal;
            let second_coefficient = e4m3_to_f32(second_scale) * weight_scale_reciprocal;
            let row_words = (A::INTERMEDIATE / 2) / size_of::<u32>();
            let word_offset = phase * CODE_WORDS_PER_PHASE + lane * 2;
            let first_source = unsafe { weight_codes.add(first_row * row_words + word_offset) };
            let second_source = unsafe { weight_codes.add(second_row * row_words + word_offset) };
            // SAFETY: one logical group contains exactly two packed u32 words.
            let first_words = unsafe { load_u32x2_read_only(first_source) };
            // SAFETY: one logical group contains exactly two packed u32 words.
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
                        // SAFETY: one lane writes two unique token/output-row values.
                        unsafe {
                            *output.add($token * A::HIDDEN + first_row) = f32_to_bf16(first);
                            *output.add($token * A::HIDDEN + second_row) = f32_to_bf16(second);
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
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_down_a16_b1(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            down_body::<Qwen38_27B, 1, true>(
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
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn nvfp4_down_a16<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;
        let _ = A::HIDDEN;

        unsafe {
            down_body::<A, TOKENS, false>(
                input,
                weight_codes,
                weight_scales,
                weight_scale_reciprocal,
                output,
                core::ptr::addr_of_mut!(SHARED).cast::<u32>(),
            );
        }
    }

    /// Projects exact Qwen3.5 BF16 activations through represented NVFP4.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_nvfp4_down_a16<const TOKENS: usize>(
        input: *const u32,
        weight_codes: *const u32,
        weight_scales: *const u8,
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) {
        static mut SHARED: SharedArray<u32, SHARED_U32, 16> = SharedArray::UNINIT;

        unsafe {
            down_body::<Qwen35_9B, TOKENS, false>(
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
    // Eight warps emit two rows each, so 320 exact CTAs cover all 5,120
    // outputs without a tail branch while sharing each staged input phase.
    LaunchConfig1D::new((OUTPUT_ROWS / (2 * WARPS)) as u32, THREADS, 0)
}

fn qwen35_launch_config() -> LaunchConfig1D {
    // The same eight warps retain two output rows each; 4,096 / 16 gives 256
    // exact CTAs. Each row has 768 K16 groups, so the unchanged lane order
    // traverses 24 phases instead of Qwen3.8's 34 without changing arithmetic.
    LaunchConfig1D::new((Qwen35_9B::HIDDEN / (2 * WARPS)) as u32, THREADS, 0)
}

struct PreparedBatchOneRoute {
    projection: PreparedLaunch<kernels::__nvfp4_down_a16_b1_CudaKernel>,
}

struct PreparedBatchRoute<A: Arch, const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__nvfp4_down_a16_CudaKernel<A, TOKENS>>,
}

impl PreparedBatchOneRoute {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_nvfp4_down_a16_b1(launch_config())
            .map_err(|source| GpuError::launch("preparing SM120 NVFP4 A16 B=1", source))?;

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
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .nvfp4_down_a16_b1(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching SM120 NVFP4 A16 B=1", source))
    }
}

impl<A: Arch, const TOKENS: usize> PreparedBatchRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_nvfp4_down_a16::<A, TOKENS>(launch_config())
            .map_err(|source| GpuError::launch("preparing SM120 NVFP4 A16", source))?;

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
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .nvfp4_down_a16::<A, TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching SM120 NVFP4 A16", source))
    }
}

struct PreparedQwen35BatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_nvfp4_down_a16_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35BatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen35_nvfp4_down_a16::<TOKENS>(qwen35_launch_config())
            .map_err(|source| GpuError::launch("preparing Qwen3.5 SM120 NVFP4 A16", source))?;

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
        weight_scale_reciprocal: f32,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_nvfp4_down_a16::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight_codes.cast::<u32>(),
                weight_scales,
                weight_scale_reciprocal,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 SM120 NVFP4 A16", source))
    }
}

/// PTX symbols retained for every exact SM120 NVFP4 down projection batch.
pub(crate) fn nvfp4_down_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        "nvfp4_down_a16_b1",
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 2>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 3>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 4>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 5>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 6>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 7>(),
        kernels::nvfp4_down_a16_ptx_name::<Qwen38_27B, 8>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.5 NVFP4 down batch.
pub(crate) fn qwen35_nvfp4_down_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen35_nvfp4_down_a16_ptx_name::<1>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<2>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<3>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<4>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<5>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<6>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<7>(),
        kernels::qwen35_nvfp4_down_a16_ptx_name::<8>(),
    ]
}

/// Prepared A16 routes consuming the exact source NVFP4 down owner on SM120.
pub struct Nvfp4DownOp<A: Sm120Arch = Qwen38_27B> {
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

impl<A: Sm120Arch> Nvfp4DownOp<A> {
    /// Loads the embedded SM120 module and prepares every exact-batch route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = nvfp4_down_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the SM120 NVFP4 down projection module", source)
        })?;

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
    /// `input` covers `batch * 17_408` BF16 values; `weight_codes` covers the
    /// packed `[5_120, 17_408]` E2M1 plane; `weight_scales` covers its
    /// swizzled `[5_120, 1_088]` E4M3 plane; and `output` covers
    /// `batch * 5_120` BF16 values. Four-byte-loaded planes are four-byte
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
                "NVFP4 down projection batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

/// Prepared exact-batch Qwen3.5 NVFP4 down routes on SM120.
pub struct Qwen35Nvfp4DownOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen35BatchRoute<1>,
    b2: PreparedQwen35BatchRoute<2>,
    b3: PreparedQwen35BatchRoute<3>,
    b4: PreparedQwen35BatchRoute<4>,
    b5: PreparedQwen35BatchRoute<5>,
    b6: PreparedQwen35BatchRoute<6>,
    b7: PreparedQwen35BatchRoute<7>,
    b8: PreparedQwen35BatchRoute<8>,
}

impl Qwen35Nvfp4DownOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = qwen35_nvfp4_down_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 SM120 NVFP4 down module", source)
        })?;

        Ok(Self {
            b1: PreparedQwen35BatchRoute::prepare(&module)?,
            b2: PreparedQwen35BatchRoute::prepare(&module)?,
            b3: PreparedQwen35BatchRoute::prepare(&module)?,
            b4: PreparedQwen35BatchRoute::prepare(&module)?,
            b5: PreparedQwen35BatchRoute::prepare(&module)?,
            b6: PreparedQwen35BatchRoute::prepare(&module)?,
            b7: PreparedQwen35BatchRoute::prepare(&module)?,
            b8: PreparedQwen35BatchRoute::prepare(&module)?,
            module,
        })
    }

    /// Executes represented-weight A16 at exact `B=1..=8`.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 12_288` BF16 values; `weight_codes` covers
    /// packed E2M1 `[4_096, 12_288]`; `weight_scales` covers swizzled E4M3
    /// `[4_096, 768]`; and `output` covers `batch * 4_096` BF16 values.
    /// Four-byte-loaded planes are aligned, the divisor is finite and
    /// positive, and disjoint allocations remain live in `stream`'s context.
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
                "Qwen3.5 NVFP4 weight scale divisor must be finite and positive",
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
                "Qwen3.5 NVFP4 down batch {batch} is outside the exact range 1..={MAX_BATCH}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CODE_WORDS_PER_PHASE, GROUP_K, GROUPS_PER_ROW, INPUT_COLUMNS, MAX_BATCH, OUTPUT_ROWS,
        PHASE_GROUPS, PHASES, SHARED_U32, WARPS, nvfp4_down_ptx_names, qwen35_nvfp4_down_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen35_9B};

    #[test]
    fn exact_geometry_matches_the_source_owner() {
        assert_eq!(INPUT_COLUMNS, 17_408);
        assert_eq!(OUTPUT_ROWS, 5_120);
        assert_eq!(GROUPS_PER_ROW, 1_088);
        assert_eq!(PHASES, 34);
        assert_eq!(CODE_WORDS_PER_PHASE, 64);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);
    }

    #[test]
    fn qwen35_geometry_preserves_the_phase_and_row_ownership() {
        assert_eq!(Qwen35_9B::INTERMEDIATE / GROUP_K, 768);
        assert_eq!(Qwen35_9B::INTERMEDIATE / GROUP_K / PHASE_GROUPS, 24);
        assert_eq!(Qwen35_9B::HIDDEN / (2 * WARPS), 256);
        assert_eq!(SHARED_U32 * size_of::<u32>(), 8_192);
    }

    #[test]
    fn inventory_has_one_distinct_entry_per_batch() {
        let names = nvfp4_down_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(unique.len(), names.len());

        let qwen35 = qwen35_nvfp4_down_ptx_names();
        let all = names.into_iter().chain(qwen35).collect::<BTreeSet<_>>();
        assert_eq!(qwen35.len(), MAX_BATCH);
        assert_eq!(all.len(), 2 * MAX_BATCH);
    }
}

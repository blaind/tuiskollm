//! Source-BF16 Q/gate/K/V projection for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const INPUT_COLUMNS: usize = Qwen38_27B::HIDDEN;
const OUTPUT_ROWS: usize = Qwen38_27B::ATTENTION_QKV_ROWS;
const OUTPUT_TILES: usize = OUTPUT_ROWS / 8;
const QWEN35_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const QWEN35_INPUT_COLUMNS: usize = Qwen35_9B::HIDDEN;
const QWEN35_OUTPUT_ROWS: usize = Qwen35_9B::ATTENTION_QKV_ROWS;
const QWEN35_OUTPUT_TILES: usize = QWEN35_OUTPUT_ROWS / 8;
const QWEN36_INPUT_COLUMNS: usize = Qwen36Moe35B::HIDDEN;
const QWEN36_OUTPUT_ROWS: usize = Qwen36Moe35B::ATTENTION_QKV_ROWS;
const QWEN36_OUTPUT_TILES: usize = QWEN36_OUTPUT_ROWS / 8;
// Eight warps publish 64 adjacent fused rows per CTA. The resulting 224 CTAs
// expose 1.32 waves on the 170-SM target while each CTA owns only its MMA state.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const BLOCKS: u32 = (OUTPUT_TILES / WARPS) as u32;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{tcgen05, thread, wmma};

    #[inline(always)]
    unsafe fn input_pair<A: Arch, const TOKENS: usize>(
        input: *const u32,
        row: usize,
        column: usize,
    ) -> u32 {
        if row >= TOKENS {
            return 0;
        }

        // SAFETY: the exact route owns `TOKENS` complete 5,120-wide BF16 rows.
        unsafe { *input.add(row * (A::HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn weight_pair<A: Arch>(weight: *const u32, row: usize, column: usize) -> u32 {
        // SAFETY: the gathered source-BF16 plane is `[OUTPUT_ROWS, INPUT_COLUMNS]`.
        unsafe { *weight.add(row * (A::HIDDEN / 2) + column / 2) }
    }

    #[inline(always)]
    unsafe fn qkv_body<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let output_tile = thread::blockIdx_x() as usize * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        macro_rules! k_step {
            ($column:expr) => {{
                let column = $column;
                // m16n8k16 is the smallest native BF16 Tensor Core tile. B<=8
                // uses only the lower token rows; no padded row is published.
                let activation = unsafe {
                    [
                        input_pair::<A, TOKENS>(input, group, column + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group + 8, column + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group, column + 8 + 2 * thread_in_group),
                        input_pair::<A, TOKENS>(input, group + 8, column + 8 + 2 * thread_in_group),
                    ]
                };
                let weights = unsafe {
                    [
                        weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                        weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                    ]
                };
                // SAFETY: all lanes execute the same row-major A / column-major B MMA.
                accumulator =
                    unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            }};
        }
        // The narrow exact routes fold most activation rows to constants and
        // lose load depth; four K-blocks per iteration restore the weight
        // pipeline, and the wide routes take eight (INPUT_COLUMNS = 5,120).
        if TOKENS <= 2 {
            while column < A::HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                column += 64;
            }
        } else {
            while column < A::HIDDEN {
                k_step!(column);
                k_step!(column + 16);
                k_step!(column + 32);
                k_step!(column + 48);
                k_step!(column + 64);
                k_step!(column + 80);
                k_step!(column + 96);
                k_step!(column + 112);
                column += 128;
            }
        }

        if group < TOKENS {
            let output_column_word = output_tile * 4 + thread_in_group;
            // SAFETY: the lower fragment maps to one active token and one fused-row pair.
            unsafe {
                *output.add(group * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                    tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            }
        }
    }

    #[inline(always)]
    unsafe fn qkv_prefill_body<A: Arch, const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_index = tid >> 5;
        let group = lane >> 2;
        let thread_in_group = lane & 3;
        let block = thread::blockIdx_x() as usize;
        let blocks = A::ATTENTION_QKV_ROWS / 8 / WARPS;
        let output_block = block % blocks;
        let token_tile = block / blocks;
        let output_tile = output_block * WARPS + warp_index;
        let weight_row = output_tile * 8 + group;
        let token_row = token_tile * 16 + group;
        let mut accumulator = [0.0f32; 4];
        let mut column = 0usize;

        while column < A::HIDDEN {
            let activation = unsafe {
                [
                    input_pair::<A, TOKENS>(input, token_row, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row + 8, column + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row, column + 8 + 2 * thread_in_group),
                    input_pair::<A, TOKENS>(input, token_row + 8, column + 8 + 2 * thread_in_group),
                ]
            };
            let weights = unsafe {
                [
                    weight_pair::<A>(weight, weight_row, column + 2 * thread_in_group),
                    weight_pair::<A>(weight, weight_row, column + 8 + 2 * thread_in_group),
                ]
            };
            accumulator = unsafe { wmma::mma_m16n8k16_f32_bf16(accumulator, activation, weights) };
            column += 16;
        }

        let output_column_word = output_tile * 4 + thread_in_group;
        // A 16-row token tile publishes both native accumulator halves, while
        // 224 output CTAs retain 1.32 target-SM waves per token tile.
        unsafe {
            *output.add(token_row * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[0], accumulator[1]);
            *output.add((token_row + 8) * (A::ATTENTION_QKV_ROWS / 2) + output_column_word) =
                tcgen05::cvt_f32x2_bf16x2(accumulator[2], accumulator[3]);
        }
    }

    /// Projects exact MTP decode rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers every 8-row fused-output tile once.
        unsafe { qkv_body::<Qwen38_27B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact MTP prompt tile through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen38_27B, TOKENS>(input, weight, output) }
    }

    /// Projects exact Qwen3.5 MTP rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.5 has 1,280 eight-row tiles. Eight warps publish 64 rows
        // per CTA, so 160 CTAs read its 80 MiB matrix once without changing
        // the per-output m16n8k16 accumulation order.
        unsafe { qkv_body::<Qwen35_9B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact Qwen3.5 MTP prompt tile through gathered BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen35_9B, TOKENS>(input, weight, output) }
    }

    /// Projects exact Qwen3.6 MTP rows through gathered source-BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_qkv<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // Qwen3.6 has 1,152 eight-row tiles. Eight warps publish 64 rows per
        // CTA, so 144 CTAs read its 36 MiB matrix once without changing any
        // per-output m16n8k16 accumulation order.
        unsafe { qkv_body::<Qwen36Moe35B, TOKENS>(input, weight, output) }
    }

    /// Projects an exact Qwen3.6 MTP prompt tile through gathered BF16 Q/gate/K/V weights.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_mtp_bf16_qkv_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { qkv_prefill_body::<Qwen36Moe35B, TOKENS>(input, weight, output) }
    }
}

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared Q/gate/K/V entry for an exact row count.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait MtpQkvRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares this route's exact-width entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's projection entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `MtpBf16QkvOp::launch`'s contract unchanged.
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's MTP QKV routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 and Qwen3.6 here never widens the
/// artifact-level admission bound. Each table names only the entries its own
/// model emits, which keeps the compiled inventory fixed while the three
/// prepared owners share one wrapper.
pub trait MtpQkvEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: MtpQkvRoute<A>;
    /// Prepared prefill route for `T=32,64,128`.
    type Prefill<const TOKENS: usize>: MtpQkvRoute<A>;
    /// Prepared `T=1024` prefill route, unadmitted outside Qwen3.8.
    type Prefill1024: MtpQkvRoute<A>;

    /// Whether `T=1024` is an admitted prefill row count.
    const HAS_T1024: bool;
    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not tile.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8 decode entry for one exact batch.
pub struct PreparedRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8 prefill entry for one exact prompt tile.
pub struct PreparedPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 decode entry for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 prefill entry for one exact prompt tile.
pub struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen35_mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 decode entry for one exact batch.
pub struct PreparedQwen36Route<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_qkv_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 prefill entry for one exact prompt tile.
pub struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen36_mtp_bf16_qkv_prefill_CudaKernel<TOKENS>>,
}

/// Stands in for a prefill width an architecture does not admit.
///
/// It prepares and launches no entry, so an unadmitted width can never reach
/// the device and never enters the emitted inventory.
pub struct UnadmittedQkvRoute;

impl<const TOKENS: usize> private::Sealed for PreparedRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedPrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35PrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36PrefillRoute<TOKENS> {}
impl private::Sealed for UnadmittedQkvRoute {}

impl<const TOKENS: usize> MtpQkvRoute<Qwen36Moe35B> for PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 MTP BF16 QKV grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.6 MTP BF16 QKV projection", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.6 MTP BF16 QKV projection", source)
            })
    }
}

impl<const TOKENS: usize> MtpQkvRoute<Qwen36Moe35B> for PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN36_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 MTP BF16 QKV prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen36_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.6 MTP BF16 QKV prefill projection",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.6 MTP BF16 QKV prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> MtpQkvRoute<Qwen35_9B> for PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 MTP BF16 QKV grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.5 MTP BF16 QKV projection", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 QKV projection", source)
            })
    }
}

impl<const TOKENS: usize> MtpQkvRoute<Qwen35_9B> for PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(QWEN35_OUTPUT_TILES / WARPS * token_tiles).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 QKV prefill grid exceeds u32")
        })?;
        Ok(Self {
            projection: module
                .prepare_qwen35_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.5 MTP BF16 QKV prefill projection",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.5 MTP BF16 QKV prefill projection",
                    source,
                )
            })
    }
}

// The Qwen3.8 entries compile that model's widths into concrete symbols, so
// these routes stay bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> MtpQkvRoute<A> for PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let token_tiles = TOKENS / 16;
        let blocks = u32::try_from(BLOCKS as usize * token_tiles)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 QKV prefill grid exceeds u32"))?;
        Ok(Self {
            projection: module
                .prepare_mtp_bf16_qkv_prefill::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| {
                    GpuError::launch("preparing the MTP BF16 QKV prefill projection", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_qkv_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the MTP BF16 QKV prefill projection", source)
            })
    }
}

impl<A: Sm120Arch, const TOKENS: usize> MtpQkvRoute<A> for PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_mtp_bf16_qkv::<TOKENS>(LaunchConfig1D::new(BLOCKS, THREADS, 0))
            .map_err(|source| GpuError::launch("preparing the MTP BF16 QKV projection", source))?;

        Ok(Self { projection })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_qkv::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| GpuError::launch("launching the MTP BF16 QKV projection", source))
    }
}

impl<A: Arch> MtpQkvRoute<A> for UnadmittedQkvRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    unsafe fn launch(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _input: *const u16,
        _weight: *const u16,
        _output: *mut u16,
    ) -> GpuResult<()> {
        Err(unadmitted_route())
    }
}

// The derived table rejects an unadmitted width before dispatch, so this is
// the defensive tail of a route that owns no entry.
fn unadmitted_route() -> GpuError {
    GpuError::invalid_launch("MTP BF16 QKV route is not admitted for this architecture")
}

/// Qwen3.8 entry table: the 224-CTA decode entries and prefill through `T=1024`.
pub struct Qwen38MtpQkvEntries;

/// Qwen3.5 entry table: the 160-CTA decode entries and prefill through `T=128`.
pub struct Qwen35MtpQkvEntries;

/// Qwen3.6 entry table: the 144-CTA decode entries and prefill through `T=128`.
pub struct Qwen36MtpQkvEntries;

impl private::Sealed for Qwen38MtpQkvEntries {}
impl private::Sealed for Qwen35MtpQkvEntries {}
impl private::Sealed for Qwen36MtpQkvEntries {}

impl<A: Sm120Arch> MtpQkvEntries<A> for Qwen38MtpQkvEntries {
    type Decode<const TOKENS: usize> = PreparedRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<TOKENS>;
    type Prefill1024 = PreparedPrefillRoute<1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "";
    const MODULE_OPERATION: &'static str = "loading the MTP BF16 QKV module";

    fn require_geometry() -> GpuResult<()> {
        if !INPUT_COLUMNS.is_multiple_of(16)
            || !OUTPUT_ROWS.is_multiple_of(64)
            || !OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        mtp_bf16_qkv_ptx_names()
            .into_iter()
            .chain(mtp_bf16_qkv_prefill_ptx_names())
            .collect()
    }
}

impl MtpQkvEntries<Qwen35_9B> for Qwen35MtpQkvEntries {
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen35PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedQkvRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.5 ";
    const MODULE_OPERATION: &'static str = "loading the Qwen3.5 MTP BF16 QKV module";

    fn require_geometry() -> GpuResult<()> {
        if !QWEN35_INPUT_COLUMNS.is_multiple_of(128)
            || !QWEN35_OUTPUT_ROWS.is_multiple_of(64)
            || !QWEN35_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_mtp_bf16_qkv_ptx_names().to_vec()
    }
}

impl MtpQkvEntries<Qwen36Moe35B> for Qwen36MtpQkvEntries {
    type Decode<const TOKENS: usize> = PreparedQwen36Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen36PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedQkvRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.6 ";
    const MODULE_OPERATION: &'static str = "loading the Qwen3.6 MTP BF16 QKV module";

    fn require_geometry() -> GpuResult<()> {
        if !QWEN36_INPUT_COLUMNS.is_multiple_of(128)
            || !QWEN36_OUTPUT_ROWS.is_multiple_of(64)
            || !QWEN36_OUTPUT_TILES.is_multiple_of(WARPS)
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 MTP QKV geometry does not tile exact BF16 MMA shapes",
            ));
        }
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_mtp_bf16_qkv_ptx_names().to_vec()
    }
}

/// Stable PTX symbol inventory for every exact MTP QKV decode batch.
pub(crate) fn mtp_bf16_qkv_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::mtp_bf16_qkv_ptx_name::<1>(),
        kernels::mtp_bf16_qkv_ptx_name::<2>(),
        kernels::mtp_bf16_qkv_ptx_name::<3>(),
        kernels::mtp_bf16_qkv_ptx_name::<4>(),
        kernels::mtp_bf16_qkv_ptx_name::<5>(),
        kernels::mtp_bf16_qkv_ptx_name::<6>(),
        kernels::mtp_bf16_qkv_ptx_name::<7>(),
        kernels::mtp_bf16_qkv_ptx_name::<8>(),
    ]
}

/// Stable PTX symbol inventory for every exact MTP QKV prompt tile.
pub(crate) fn mtp_bf16_qkv_prefill_ptx_names() -> [&'static str; 4] {
    [
        kernels::mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<128>(),
        kernels::mtp_bf16_qkv_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX symbol inventory for every exact Qwen3.5 MTP QKV route.
pub(crate) fn qwen35_mtp_bf16_qkv_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<1>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<2>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<3>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<4>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<5>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<6>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<7>(),
        kernels::qwen35_mtp_bf16_qkv_ptx_name::<8>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::qwen35_mtp_bf16_qkv_prefill_ptx_name::<128>(),
    ]
}

/// Stable PTX symbol inventory for every exact Qwen3.6 MTP QKV route.
pub(crate) fn qwen36_mtp_bf16_qkv_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<1>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<2>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<3>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<4>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<5>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<6>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<7>(),
        kernels::qwen36_mtp_bf16_qkv_ptx_name::<8>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<32>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<64>(),
        kernels::qwen36_mtp_bf16_qkv_prefill_ptx_name::<128>(),
    ]
}

fn unsupported_rows<A: Arch, E: MtpQkvEntries<A>>(rows: usize) -> GpuError {
    let admitted = if E::HAS_T1024 {
        format!("{PREFILL_ROUTES:?}")
    } else {
        format!("{QWEN35_PREFILL_ROUTES:?}")
    };
    GpuError::invalid_launch(format!(
        "{}MTP BF16 QKV rows {rows} are outside exact B=1..={MAX_BATCH} or T={admitted}",
        E::LABEL
    ))
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_mtp_bf16_qkv),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct MtpBf16QkvRoutes<A: Arch, E: MtpQkvEntries<A>> {
    #[route(1)]
    b1: E::Decode<1>,
    #[route(2)]
    b2: E::Decode<2>,
    #[route(3)]
    b3: E::Decode<3>,
    #[route(4)]
    b4: E::Decode<4>,
    #[route(5)]
    b5: E::Decode<5>,
    #[route(6)]
    b6: E::Decode<6>,
    #[route(7)]
    b7: E::Decode<7>,
    #[route(8)]
    b8: E::Decode<8>,
    #[route(32)]
    t32: E::Prefill<32>,
    #[route(64)]
    t64: E::Prefill<64>,
    #[route(128)]
    t128: E::Prefill<128>,
    #[route(1024, admitted(E::HAS_T1024))]
    t1024: E::Prefill1024,
}

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact MTP decode
/// `B=1..=8` and the entry table's admitted prefill widths.
pub struct MtpBf16QkvOp<A: Arch = Qwen38_27B, E: MtpQkvEntries<A> = Qwen38MtpQkvEntries> {
    module: kernels::LoadedModule,
    routes: MtpBf16QkvRoutes<A, E>,
}

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact Qwen3.5 MTP rows.
pub type Qwen35MtpBf16QkvOp = MtpBf16QkvOp<Qwen35_9B, Qwen35MtpQkvEntries>;

/// Prepared gathered source-BF16 Q/gate/K/V routes for exact Qwen3.6 MTP rows.
pub type Qwen36MtpBf16QkvOp = MtpBf16QkvOp<Qwen36Moe35B, Qwen36MtpQkvEntries>;

impl<A: Arch, E: MtpQkvEntries<A>> MtpBf16QkvOp<A, E> {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded MTP QKV artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_OPERATION, source))?;

        let routes = MtpBf16QkvRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Applies the exact gathered source-BF16 Q/gate/K/V projection.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned, context-local, live through stream
    /// completion, and non-overlapping. `input` covers `rows * A::HIDDEN` BF16
    /// values, `weight` covers the gathered
    /// `[A::ATTENTION_QKV_ROWS, A::HIDDEN]` Q/gate/K/V plane, and `output`
    /// covers `rows * A::ATTENTION_QKV_ROWS` BF16 values.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        dispatch_mtp_bf16_qkv!(
            &self.routes,
            rows,
            |route| unsafe { route.launch(&self.module, stream, input, weight, output) },
            else => Err(unsupported_rows::<A, E>(rows))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCKS, INPUT_COLUMNS, MtpBf16QkvRoutes, MtpQkvEntries, OUTPUT_ROWS, OUTPUT_TILES,
        PREFILL_ROUTES, QWEN35_INPUT_COLUMNS, QWEN35_OUTPUT_ROWS, QWEN35_OUTPUT_TILES,
        QWEN35_PREFILL_ROUTES, QWEN36_INPUT_COLUMNS, QWEN36_OUTPUT_ROWS, QWEN36_OUTPUT_TILES,
        Qwen35MtpQkvEntries, Qwen36MtpQkvEntries, Qwen38MtpQkvEntries, WARPS,
        mtp_bf16_qkv_prefill_ptx_names, mtp_bf16_qkv_ptx_names, qwen35_mtp_bf16_qkv_ptx_names,
        qwen36_mtp_bf16_qkv_ptx_names, unsupported_rows,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

    /// The decode and prefill widths every admitted architecture routes.
    const SHARED_SCHEDULE: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];

    /// The derive's ordered admission inventory for one entry table.
    fn admitted_schedule<A: Arch, E: MtpQkvEntries<A>>() -> Vec<usize> {
        MtpBf16QkvRoutes::<A, E>::admitted_rows()
    }

    #[test]
    fn exact_geometry_covers_the_gathered_source_plane() {
        assert_eq!(INPUT_COLUMNS, 5_120);
        assert_eq!(OUTPUT_ROWS, 14_336);
        assert_eq!(OUTPUT_TILES, 1_792);
        assert_eq!(BLOCKS, 224);
        assert_eq!(BLOCKS as usize * WARPS, OUTPUT_TILES);
    }

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
        let names = mtp_bf16_qkv_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(unique.len(), names.len());
        let prefill = mtp_bf16_qkv_prefill_ptx_names();
        assert_eq!(PREFILL_ROUTES, [32, 64, 128, 1_024]);
        assert_eq!(prefill.len(), PREFILL_ROUTES.len());
        assert_eq!(prefill.iter().copied().collect::<BTreeSet<_>>().len(), 4);
    }

    #[test]
    fn qwen35_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN35_INPUT_COLUMNS, 4_096);
        assert_eq!(QWEN35_OUTPUT_ROWS, 10_240);
        assert_eq!(QWEN35_OUTPUT_TILES, 1_280);
        assert_eq!(QWEN35_OUTPUT_TILES / WARPS, 160);
        assert_eq!(QWEN35_PREFILL_ROUTES, [32, 64, 128]);
        let names = qwen35_mtp_bf16_qkv_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }

    #[test]
    fn qwen36_geometry_and_inventory_are_exact() {
        assert_eq!(QWEN36_INPUT_COLUMNS, 2_048);
        assert_eq!(QWEN36_OUTPUT_ROWS, 9_216);
        assert_eq!(QWEN36_OUTPUT_TILES, 1_152);
        assert_eq!(QWEN36_OUTPUT_TILES / WARPS, 144);
        let names = qwen36_mtp_bf16_qkv_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38MtpQkvEntries as MtpQkvEntries<Qwen38_27B>>::ptx_names(),
            mtp_bf16_qkv_ptx_names()
                .into_iter()
                .chain(mtp_bf16_qkv_prefill_ptx_names())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            <Qwen35MtpQkvEntries as MtpQkvEntries<Qwen35_9B>>::ptx_names(),
            qwen35_mtp_bf16_qkv_ptx_names().to_vec()
        );
        assert_eq!(
            <Qwen36MtpQkvEntries as MtpQkvEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_mtp_bf16_qkv_ptx_names().to_vec()
        );
    }

    /// The merged schedule, checked against the three dispatches it replaces:
    /// Qwen3.5 and Qwen3.6 stop at `T=128`, and only Qwen3.8 admits `T=1024`.
    #[test]
    fn row_routing_is_exact_for_every_admitted_architecture() {
        let qwen38 = SHARED_SCHEDULE
            .iter()
            .copied()
            .chain([1_024])
            .collect::<Vec<_>>();

        assert_eq!(
            admitted_schedule::<Qwen38_27B, Qwen38MtpQkvEntries>(),
            qwen38
        );
        assert_eq!(
            admitted_schedule::<Qwen35_9B, Qwen35MtpQkvEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule::<Qwen36Moe35B, Qwen36MtpQkvEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
    }

    /// An unadmitted row count keeps its owner's rejection wording.
    #[test]
    fn unadmitted_row_counts_name_their_architecture() {
        for (message, error) in [
            (
                "MTP BF16 QKV rows 9 are outside exact B=1..=8 or T=[32, 64, 128, 1024]",
                unsupported_rows::<Qwen38_27B, Qwen38MtpQkvEntries>(9),
            ),
            (
                "Qwen3.5 MTP BF16 QKV rows 1024 are outside exact B=1..=8 or T=[32, 64, 128]",
                unsupported_rows::<Qwen35_9B, Qwen35MtpQkvEntries>(1_024),
            ),
            (
                "Qwen3.6 MTP BF16 QKV rows 1024 are outside exact B=1..=8 or T=[32, 64, 128]",
                unsupported_rows::<Qwen36Moe35B, Qwen36MtpQkvEntries>(1_024),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }
}

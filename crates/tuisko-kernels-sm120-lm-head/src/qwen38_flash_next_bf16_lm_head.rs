//! Exact Qwen3.8-Flash-Next BF16 language-model head.
//!
//! `lm_head: Linear(2560 -> 248320, bias=False)`, untied, reading the collapsed
//! stream the hyper-connection mixer publishes. This is the only projection not preceded by
//! a norm: there is no final RMSNorm, and the mixer's four-way mean is the whole
//! epilogue.
//!
//! The head is BF16 in the pinned checkpoint even though the layer weights are
//! NVFP4, so it is the same plain projection the backbone shapes use and shares
//! their device body.
//!
//! Decode only. Logits are 496,640 B per row, and the reference reads them for
//! the rows it samples, never for a whole prompt tile; a `T=1024` route would
//! materialize 508 MiB to discard all but the last row.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::projection::{
    ROWS_PER_TILE, bf16_projection_decode, tiles_exactly,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const INPUT_COLUMNS: usize = <Qwen38FlashNext as Arch>::HIDDEN;
const OUTPUT_ROWS: usize = <Qwen38FlashNext as Arch>::VOCAB;
// Eight warps publish 64 adjacent vocabulary rows per CTA. The resulting 3,880
// CTAs stream 1,271,398,400 BF16 weight bytes once per invocation at 22.8 CTAs
// per SM on the 170-SM target, and every CTA owns only its MMA state.
const WARPS: usize = 8;
const THREADS: u32 = (WARPS * 32) as u32;
const BLOCKS: u32 = (OUTPUT_ROWS / ROWS_PER_TILE / WARPS) as u32;

const _: () = assert!(INPUT_COLUMNS == 2_560);
const _: () = assert!(OUTPUT_ROWS == 248_320);
const _: () = assert!(tiles_exactly(INPUT_COLUMNS, OUTPUT_ROWS, WARPS));

#[cuda_module]
mod kernels {
    use super::*;

    /// Projects exact Qwen3.8-Flash-Next decode rows through the untied BF16 LM head.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_bf16_lm_head<const TOKENS: usize>(
        input: *const u32,
        weights: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers every eight-row vocabulary tile once.
        unsafe {
            bf16_projection_decode::<INPUT_COLUMNS, OUTPUT_ROWS, WARPS, TOKENS>(
                input, weights, output,
            )
        }
    }
}

struct PreparedBatchRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen38_flash_next_bf16_lm_head_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedBatchRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let projection = module
            .prepare_qwen38_flash_next_bf16_lm_head::<TOKENS>(LaunchConfig1D::new(
                BLOCKS, THREADS, 0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing the Qwen3.8-Flash-Next BF16 LM head", source)
            })?;

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
            .qwen38_flash_next_bf16_lm_head::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weights.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next BF16 LM head", source)
            })
    }
}

/// PTX symbols retained for every exact Qwen3.8-Flash-Next BF16 LM-head batch.
pub(crate) fn qwen38_flash_next_bf16_lm_head_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<1>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<2>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<3>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<4>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<5>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<6>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<7>(),
        kernels::qwen38_flash_next_bf16_lm_head_ptx_name::<8>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_bf16_lm_head),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct Qwen38FlashNextBf16LmHeadRoutes {
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

/// Prepared exact-batch Qwen3.8-Flash-Next BF16 LM-head routes on SM120.
pub struct Qwen38FlashNextBf16LmHeadOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextBf16LmHeadRoutes,
}

impl Qwen38FlashNextBf16LmHeadOp {
    /// Loads the embedded module and prepares every exact batch.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry()?;
        let _ = qwen38_flash_next_bf16_lm_head_ptx_names();
        // SAFETY: this crate owns the embedded LM-head artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next BF16 LM head", source)
        })?;

        let routes = Qwen38FlashNextBf16LmHeadRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Projects represented BF16 activations through represented BF16 weights.
    ///
    /// # Safety
    ///
    /// `input` covers `batch * 2_560` BF16 values, `weights` covers BF16
    /// `[248_320, 2_560]`, and `output` covers `batch * 248_320` BF16 values.
    /// Four-byte-loaded planes are aligned, disjoint, context-local, and remain
    /// live until the stream completes.
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
                // SAFETY: exact-B dispatch preserves the public pointer contract.
                unsafe { $route.launch(&self.module, stream, input, weights, output) }
            };
        }

        dispatch_qwen38_flash_next_bf16_lm_head!(&self.routes, batch, |route| launch!(route), else => Err(GpuError::invalid_launch(format!(
            "Qwen3.8-Flash-Next BF16 LM-head batch {batch} is outside exact B=1..={MAX_BATCH}"
        ))))
    }
}

/// Rejects a vocabulary or stream width the emitted entries do not tile.
fn require_geometry() -> GpuResult<()> {
    if INPUT_COLUMNS != 2_560
        || OUTPUT_ROWS != 248_320
        || !tiles_exactly(INPUT_COLUMNS, OUTPUT_ROWS, WARPS)
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next LM-head geometry does not tile exact BF16 MMA shapes",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCKS, INPUT_COLUMNS, MAX_BATCH, OUTPUT_ROWS, Qwen38FlashNextBf16LmHeadRoutes, THREADS,
        WARPS, qwen38_flash_next_bf16_lm_head_ptx_names, require_geometry,
    };
    use std::collections::BTreeSet;
    use tuisko_kernels_sm120_common::projection::ROWS_PER_TILE;
    use tuisko_model::{Arch, Qwen35_9B, Qwen38FlashNext};

    #[test]
    fn exact_geometry_covers_the_untied_vocabulary_plane() {
        assert_eq!(INPUT_COLUMNS, 2_560);
        assert_eq!(OUTPUT_ROWS, 248_320);
        assert_eq!(THREADS, 256);
        assert_eq!(BLOCKS, 3_880);
        assert_eq!(BLOCKS as usize * WARPS * ROWS_PER_TILE, OUTPUT_ROWS);
        assert!(require_geometry().is_ok());
    }

    /// The two BF16 heads share a vocabulary and nothing else, which is why
    /// this target needs its own entries rather than the Qwen3.5 ones.
    #[test]
    fn the_stream_width_is_what_separates_this_head_from_the_other_bf16_head() {
        assert_eq!(<Qwen38FlashNext as Arch>::VOCAB, <Qwen35_9B as Arch>::VOCAB);
        assert_ne!(
            <Qwen38FlashNext as Arch>::HIDDEN,
            <Qwen35_9B as Arch>::HIDDEN
        );
    }

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
        let names = qwen38_flash_next_bf16_lm_head_ptx_names();

        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(
            names.iter().copied().collect::<BTreeSet<_>>().len(),
            MAX_BATCH
        );
        assert_eq!(
            Qwen38FlashNextBf16LmHeadRoutes::admitted_rows(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
}

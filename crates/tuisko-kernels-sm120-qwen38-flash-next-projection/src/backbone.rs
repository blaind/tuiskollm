//! Source-BF16 backbone projections for the Qwen3.8-Flash-Next decoder layers.
//!
//! Three shapes move activations between the layer's widths. Each is one
//! `nn.Linear` over the fused source plane the model lane materializes, with
//! FP32 accumulation over the whole contraction and one BF16 rounding at the
//! store.
//!
//! ```text
//! gdn_input     [rows, 2560] x [16384, 2560]^T -> [rows, 16384]   in_proj_qkv + z
//! qsa_qkv       [rows, 2560] x [13312, 2560]^T -> [rows, 13312]   q + k + v
//! block_output  [rows, 6144] x [ 2560, 6144]^T -> [rows,  2560]   out_proj and o_proj
//! ```
//!
//! The third shape is one entry family serving two call sites. The GDN layer's
//! `out_proj` and the sparse-attention layer's `o_proj` have identical geometry
//! and identical numerics; they differ only in which weight plane the caller
//! passes. Neither carries an epilogue: the sparse-attention sigmoid gate is a
//! separate entry that publishes this projection's input, and the GDN layer
//! applies no gate here at all.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::projection::{
    ROWS_PER_TILE, TOKENS_PER_TILE, bf16_projection_decode, bf16_projection_prefill, tiles_exactly,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Largest decode batch any Qwen3.8-Flash-Next route admits.
const MAX_BATCH: usize = 8;
/// Prompt tiles the Qwen3.8-Flash-Next capture schedule admits.
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
/// Residual-stream width every widening projection contracts over.
const HIDDEN: usize = <Qwen38FlashNext as Arch>::HIDDEN;
/// Fused QKV-then-Z rows the gated DeltaNet mixer reads.
const GDN_INPUT_ROWS: usize = <Qwen38FlashNext as Arch>::GDN_INPUT_ROWS;
/// Fused query/gate, key, and value rows the sparse-attention prepare reads.
const QSA_QKV_ROWS: usize = <Qwen38FlashNext as Arch>::ATTENTION_QKV_ROWS;
/// Attention value width the sparse-attention output projection contracts over.
const ATTENTION_OUTPUT_COLUMNS: usize = <Qwen38FlashNext as Arch>::ATTENTION_OUTPUT_COLUMNS;
/// Recurrent value width the gated DeltaNet output projection contracts over.
const GDN_VALUE_ROWS: usize = <Qwen38FlashNext as Arch>::GDN_VALUE_ROWS;
/// Shared contraction width of the two output projection call sites.
const BLOCK_COLUMNS: usize = ATTENTION_OUTPUT_COLUMNS;

// Eight warps publish 64 adjacent output rows per CTA for the two widening
// shapes: 256 and 208 CTAs against the 170-SM target, each CTA owning only its
// MMA state. The narrow reduction publishes 320 rows from 80 CTAs at four
// warps, which is the widest tile its output admits without idle warps.
const WIDE_WARPS: usize = 8;
const WIDE_THREADS: u32 = (WIDE_WARPS * 32) as u32;
const NARROW_WARPS: usize = 4;
const NARROW_THREADS: u32 = (NARROW_WARPS * 32) as u32;
const GDN_INPUT_BLOCKS: u32 = (GDN_INPUT_ROWS / ROWS_PER_TILE / WIDE_WARPS) as u32;
const QSA_QKV_BLOCKS: u32 = (QSA_QKV_ROWS / ROWS_PER_TILE / WIDE_WARPS) as u32;
const BLOCK_OUTPUT_BLOCKS: u32 = (HIDDEN / ROWS_PER_TILE / NARROW_WARPS) as u32;
const MTP_FUSION_DECODE_ROWS: usize = Qwen38FlashNext::HC_COUNT;
const MTP_FUSION_BLOCKS: u32 = (HIDDEN / ROWS_PER_TILE / WIDE_WARPS) as u32;

const _: () = assert!(HIDDEN == 2_560);
const _: () = assert!(GDN_INPUT_ROWS == 16_384);
const _: () = assert!(QSA_QKV_ROWS == 13_312);
// The two output call sites share one entry family only because these two
// independently derived widths agree. Divergence in either must break the build
// rather than silently route one call site through the other's geometry.
const _: () = assert!(ATTENTION_OUTPUT_COLUMNS == 6_144);
const _: () = assert!(GDN_VALUE_ROWS == ATTENTION_OUTPUT_COLUMNS);
const _: () = assert!(tiles_exactly(HIDDEN, GDN_INPUT_ROWS, WIDE_WARPS));
const _: () = assert!(tiles_exactly(HIDDEN, QSA_QKV_ROWS, WIDE_WARPS));
const _: () = assert!(tiles_exactly(BLOCK_COLUMNS, HIDDEN, NARROW_WARPS));
const _: () = assert!(MTP_FUSION_DECODE_ROWS == 4);
const _: () = assert!(tiles_exactly(HIDDEN, HIDDEN, WIDE_WARPS));
const _: () = assert!(MTP_FUSION_BLOCKS == 40);

#[cuda_module]
mod kernels {
    use super::*;

    /// Projects exact Qwen3.8-Flash-Next decode rows into the fused GDN QKV/Z plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_gdn_input_projection<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // SAFETY: the prepared grid covers every eight-row output tile once.
        unsafe {
            bf16_projection_decode::<HIDDEN, GDN_INPUT_ROWS, WIDE_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Projects one exact Qwen3.8-Flash-Next prompt tile into the fused GDN QKV/Z plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_gdn_input_projection_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            bf16_projection_prefill::<HIDDEN, GDN_INPUT_ROWS, WIDE_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Projects exact Qwen3.8-Flash-Next decode rows into the fused sparse-attention QKV plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_qsa_qkv_projection<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // The per-head packed `[query|gate]` split is a property of the source
        // plane's row order, so this entry publishes it by projecting the rows
        // in place and never reorders them.
        unsafe {
            bf16_projection_decode::<HIDDEN, QSA_QKV_ROWS, WIDE_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Projects one exact Qwen3.8-Flash-Next prompt tile into the fused sparse-attention QKV plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_qsa_qkv_projection_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            bf16_projection_prefill::<HIDDEN, QSA_QKV_ROWS, WIDE_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Reduces exact Qwen3.8-Flash-Next decode rows from the block width to the stream width.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_block_output_projection<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        // No epilogue: the sparse-attention gate is its own entry ahead of this
        // one, and the gated DeltaNet call site has no gate at this seam.
        unsafe {
            bf16_projection_decode::<BLOCK_COLUMNS, HIDDEN, NARROW_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Reduces one exact Qwen3.8-Flash-Next prompt tile from the block width to the stream width.
    #[kernel]
    #[launch_bounds(128, 4)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_block_output_projection_prefill<const TOKENS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            bf16_projection_prefill::<BLOCK_COLUMNS, HIDDEN, NARROW_WARPS, TOKENS>(
                input, weight, output,
            )
        }
    }

    /// Projects the MTP branch rows through one square BF16 plane.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_mtp_fusion_projection<const ROWS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe { bf16_projection_decode::<HIDDEN, HIDDEN, WIDE_WARPS, ROWS>(input, weight, output) }
    }

    /// Projects one MTP prompt tile's branch rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_mtp_fusion_projection_prefill<const ROWS: usize>(
        input: *const u32,
        weight: *const u32,
        output: *mut u32,
    ) {
        unsafe {
            bf16_projection_prefill::<HIDDEN, HIDDEN, WIDE_WARPS, ROWS>(input, weight, output)
        }
    }
}

mod private {
    pub trait Sealed {}
}

/// One prepared backbone projection entry for an exact row count.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait ProjectionRoute: Sized + private::Sealed {
    /// Prepares this route's exact-width entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's projection entry.
    ///
    /// # Safety
    ///
    /// The pointers carry [`Qwen38FlashNextProjectionOp::launch`]'s contract unchanged.
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()>;
}

/// Exact entry table of one backbone projection shape's twelve routes.
///
/// Each table names only the entries its own shape emits, so the compiled
/// inventory stays fixed while the three prepared owners share one wrapper.
pub trait ProjectionEntries: private::Sealed {
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: ProjectionRoute;
    /// Prepared prefill route for `T=32,64,128,1024`.
    type Prefill<const TOKENS: usize>: ProjectionRoute;

    /// Contraction width this shape reads from its activation plane.
    const INPUT_COLUMNS: usize;
    /// Output rows this shape publishes per represented row.
    const OUTPUT_ROWS: usize;
    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects a geometry the emitted entries do not tile.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared GDN input projection entry for one exact decode batch.
pub struct PreparedGdnInputRoute<const TOKENS: usize> {
    projection:
        PreparedLaunch<kernels::__qwen38_flash_next_gdn_input_projection_CudaKernel<TOKENS>>,
}

/// Prepared GDN input projection entry for one exact prompt tile.
pub struct PreparedGdnInputPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<
        kernels::__qwen38_flash_next_gdn_input_projection_prefill_CudaKernel<TOKENS>,
    >,
}

/// Prepared sparse-attention QKV projection entry for one exact decode batch.
pub struct PreparedQsaQkvRoute<const TOKENS: usize> {
    projection: PreparedLaunch<kernels::__qwen38_flash_next_qsa_qkv_projection_CudaKernel<TOKENS>>,
}

/// Prepared sparse-attention QKV projection entry for one exact prompt tile.
pub struct PreparedQsaQkvPrefillRoute<const TOKENS: usize> {
    projection:
        PreparedLaunch<kernels::__qwen38_flash_next_qsa_qkv_projection_prefill_CudaKernel<TOKENS>>,
}

/// Prepared block output projection entry for one exact decode batch.
pub struct PreparedBlockOutputRoute<const TOKENS: usize> {
    projection:
        PreparedLaunch<kernels::__qwen38_flash_next_block_output_projection_CudaKernel<TOKENS>>,
}

/// Prepared block output projection entry for one exact prompt tile.
pub struct PreparedBlockOutputPrefillRoute<const TOKENS: usize> {
    projection: PreparedLaunch<
        kernels::__qwen38_flash_next_block_output_projection_prefill_CudaKernel<TOKENS>,
    >,
}

impl<const TOKENS: usize> private::Sealed for PreparedGdnInputRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedGdnInputPrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQsaQkvRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQsaQkvPrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedBlockOutputRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedBlockOutputPrefillRoute<TOKENS> {}

impl<const TOKENS: usize> ProjectionRoute for PreparedGdnInputRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_gdn_input_projection::<TOKENS>(LaunchConfig1D::new(
                    GDN_INPUT_BLOCKS,
                    WIDE_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next GDN input projection",
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
            .qwen38_flash_next_gdn_input_projection::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next GDN input projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> ProjectionRoute for PreparedGdnInputPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = prefill_blocks(
            GDN_INPUT_BLOCKS,
            TOKENS,
            "Qwen3.8-Flash-Next GDN input projection",
        )?;
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_gdn_input_projection_prefill::<TOKENS>(
                    LaunchConfig1D::new(blocks, WIDE_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next GDN input prefill projection",
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
            .qwen38_flash_next_gdn_input_projection_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next GDN input prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> ProjectionRoute for PreparedQsaQkvRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_qsa_qkv_projection::<TOKENS>(LaunchConfig1D::new(
                    QSA_QKV_BLOCKS,
                    WIDE_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next sparse-attention QKV projection",
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
            .qwen38_flash_next_qsa_qkv_projection::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next sparse-attention QKV projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> ProjectionRoute for PreparedQsaQkvPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = prefill_blocks(
            QSA_QKV_BLOCKS,
            TOKENS,
            "Qwen3.8-Flash-Next sparse-attention QKV projection",
        )?;
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_qsa_qkv_projection_prefill::<TOKENS>(
                    LaunchConfig1D::new(blocks, WIDE_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next sparse-attention QKV prefill projection",
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
            .qwen38_flash_next_qsa_qkv_projection_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next sparse-attention QKV prefill projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> ProjectionRoute for PreparedBlockOutputRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_block_output_projection::<TOKENS>(LaunchConfig1D::new(
                    BLOCK_OUTPUT_BLOCKS,
                    NARROW_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next block output projection",
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
            .qwen38_flash_next_block_output_projection::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next block output projection",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> ProjectionRoute for PreparedBlockOutputPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = prefill_blocks(
            BLOCK_OUTPUT_BLOCKS,
            TOKENS,
            "Qwen3.8-Flash-Next block output projection",
        )?;
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_block_output_projection_prefill::<TOKENS>(
                    LaunchConfig1D::new(blocks, NARROW_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next block output prefill projection",
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
            .qwen38_flash_next_block_output_projection_prefill::<TOKENS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next block output prefill projection",
                    source,
                )
            })
    }
}

struct PreparedMtpFusionDecodeRoute<const ROWS: usize> {
    projection: PreparedLaunch<kernels::__qwen38_flash_next_mtp_fusion_projection_CudaKernel<ROWS>>,
}

struct PreparedMtpFusionPrefillRoute<const ROWS: usize> {
    projection:
        PreparedLaunch<kernels::__qwen38_flash_next_mtp_fusion_projection_prefill_CudaKernel<ROWS>>,
}

impl<const ROWS: usize> PreparedMtpFusionDecodeRoute<ROWS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_mtp_fusion_projection::<ROWS>(LaunchConfig1D::new(
                    MTP_FUSION_BLOCKS,
                    WIDE_THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.8 Flash-Next MTP fusion", source)
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
            .qwen38_flash_next_mtp_fusion_projection::<ROWS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8 Flash-Next MTP fusion", source)
            })
    }
}

impl<const ROWS: usize> PreparedMtpFusionPrefillRoute<ROWS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = prefill_blocks(MTP_FUSION_BLOCKS, ROWS, "Qwen3.8 Flash-Next MTP fusion")?;
        Ok(Self {
            projection: module
                .prepare_qwen38_flash_next_mtp_fusion_projection_prefill::<ROWS>(
                    LaunchConfig1D::new(blocks, WIDE_THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.8 Flash-Next MTP prompt fusion", source)
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
            .qwen38_flash_next_mtp_fusion_projection_prefill::<ROWS>(
                stream,
                &self.projection,
                input.cast::<u32>(),
                weight.cast::<u32>(),
                output.cast::<u32>(),
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8 Flash-Next MTP prompt fusion", source)
            })
    }
}

/// Grid of one prompt tile: the shape's output blocks once per sixteen tokens.
fn prefill_blocks(output_blocks: u32, tokens: usize, label: &str) -> GpuResult<u32> {
    let token_tiles = tokens / TOKENS_PER_TILE;
    u32::try_from(output_blocks as usize * token_tiles)
        .map_err(|_| GpuError::invalid_launch(format!("{label} prefill grid exceeds u32")))
}

/// Entry table of the fused GDN QKV/Z input projection.
pub struct Qwen38FlashNextGdnInputEntries;

/// Entry table of the fused sparse-attention query/key/value projection.
pub struct Qwen38FlashNextQsaQkvEntries;

/// Entry table of the block output projection shared by both layer kinds.
pub struct Qwen38FlashNextBlockOutputEntries;

impl private::Sealed for Qwen38FlashNextGdnInputEntries {}
impl private::Sealed for Qwen38FlashNextQsaQkvEntries {}
impl private::Sealed for Qwen38FlashNextBlockOutputEntries {}

impl ProjectionEntries for Qwen38FlashNextGdnInputEntries {
    type Decode<const TOKENS: usize> = PreparedGdnInputRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedGdnInputPrefillRoute<TOKENS>;

    const INPUT_COLUMNS: usize = HIDDEN;
    const OUTPUT_ROWS: usize = GDN_INPUT_ROWS;
    const LABEL: &'static str = "Qwen3.8-Flash-Next GDN input";
    const MODULE_OPERATION: &'static str =
        "loading the Qwen3.8-Flash-Next backbone projection module";

    fn require_geometry() -> GpuResult<()> {
        require_shape::<Self>(2_560, 16_384, WIDE_WARPS)
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_gdn_input_projection_ptx_names()
    }
}

impl ProjectionEntries for Qwen38FlashNextQsaQkvEntries {
    type Decode<const TOKENS: usize> = PreparedQsaQkvRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQsaQkvPrefillRoute<TOKENS>;

    const INPUT_COLUMNS: usize = HIDDEN;
    const OUTPUT_ROWS: usize = QSA_QKV_ROWS;
    const LABEL: &'static str = "Qwen3.8-Flash-Next sparse-attention QKV";
    const MODULE_OPERATION: &'static str =
        "loading the Qwen3.8-Flash-Next backbone projection module";

    fn require_geometry() -> GpuResult<()> {
        require_shape::<Self>(2_560, 13_312, WIDE_WARPS)
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_qsa_qkv_projection_ptx_names()
    }
}

impl ProjectionEntries for Qwen38FlashNextBlockOutputEntries {
    type Decode<const TOKENS: usize> = PreparedBlockOutputRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedBlockOutputPrefillRoute<TOKENS>;

    const INPUT_COLUMNS: usize = BLOCK_COLUMNS;
    const OUTPUT_ROWS: usize = HIDDEN;
    const LABEL: &'static str = "Qwen3.8-Flash-Next block output";
    const MODULE_OPERATION: &'static str =
        "loading the Qwen3.8-Flash-Next backbone projection module";

    fn require_geometry() -> GpuResult<()> {
        // Both call sites derive this width independently; the entry family is
        // only shareable while they agree.
        if GDN_VALUE_ROWS != ATTENTION_OUTPUT_COLUMNS {
            return Err(GpuError::invalid_launch(
                "Qwen3.8-Flash-Next recurrent and attention block widths differ; the output projection \
                 can no longer serve both call sites",
            ));
        }
        require_shape::<Self>(6_144, 2_560, NARROW_WARPS)
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_block_output_projection_ptx_names()
    }
}

/// Rejects a shape whose widths moved off the emitted entries' geometry.
fn require_shape<E: ProjectionEntries>(
    columns: usize,
    output_rows: usize,
    warps: usize,
) -> GpuResult<()> {
    if E::INPUT_COLUMNS != columns
        || E::OUTPUT_ROWS != output_rows
        || !tiles_exactly(E::INPUT_COLUMNS, E::OUTPUT_ROWS, warps)
    {
        return Err(GpuError::invalid_launch(format!(
            "{} projection geometry does not tile exact BF16 MMA shapes",
            E::LABEL
        )));
    }

    Ok(())
}

/// Stable PTX symbol inventory of every fused GDN input projection route.
pub(crate) fn qwen38_flash_next_gdn_input_projection_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<1>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<2>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<3>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<4>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<5>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<6>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<7>(),
        kernels::qwen38_flash_next_gdn_input_projection_ptx_name::<8>(),
        kernels::qwen38_flash_next_gdn_input_projection_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_gdn_input_projection_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_gdn_input_projection_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_gdn_input_projection_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX symbol inventory of every fused sparse-attention QKV route.
pub(crate) fn qwen38_flash_next_qsa_qkv_projection_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<1>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<2>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<3>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<4>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<5>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<6>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<7>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_ptx_name::<8>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_qsa_qkv_projection_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX symbol inventory of every block output projection route.
pub(crate) fn qwen38_flash_next_block_output_projection_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<1>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<2>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<3>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<4>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<5>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<6>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<7>(),
        kernels::qwen38_flash_next_block_output_projection_ptx_name::<8>(),
        kernels::qwen38_flash_next_block_output_projection_prefill_ptx_name::<32>(),
        kernels::qwen38_flash_next_block_output_projection_prefill_ptx_name::<64>(),
        kernels::qwen38_flash_next_block_output_projection_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_block_output_projection_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX symbol inventory of every MTP input-fusion route.
pub(crate) fn qwen38_flash_next_mtp_fusion_projection_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_mtp_fusion_projection_ptx_name::<MTP_FUSION_DECODE_ROWS>(),
        kernels::qwen38_flash_next_mtp_fusion_projection_prefill_ptx_name::<128>(),
        kernels::qwen38_flash_next_mtp_fusion_projection_prefill_ptx_name::<256>(),
        kernels::qwen38_flash_next_mtp_fusion_projection_prefill_ptx_name::<512>(),
        kernels::qwen38_flash_next_mtp_fusion_projection_prefill_ptx_name::<4_096>(),
    ]
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_mtp_fusion),
    required(1, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextMtpFusionRoutes {
    #[route(1)]
    decode: PreparedMtpFusionDecodeRoute<MTP_FUSION_DECODE_ROWS>,
    #[route(32)]
    t32: PreparedMtpFusionPrefillRoute<128>,
    #[route(64)]
    t64: PreparedMtpFusionPrefillRoute<256>,
    #[route(128)]
    t128: PreparedMtpFusionPrefillRoute<512>,
    #[route(1024)]
    t1024: PreparedMtpFusionPrefillRoute<4_096>,
}

/// Prepared square projections for the MTP input fusion.
pub struct Qwen38FlashNextMtpFusionProjectionOp {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextMtpFusionRoutes,
}

impl Qwen38FlashNextMtpFusionProjectionOp {
    /// Loads the module and prepares every admitted MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if !tiles_exactly(HIDDEN, HIDDEN, WIDE_WARPS) {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 Flash-Next MTP fusion does not tile the exact BF16 MMA shape",
            ));
        }
        let _ = qwen38_flash_next_mtp_fusion_projection_ptx_names();
        // SAFETY: this crate owns the embedded projection artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.8 Flash-Next MTP fusion", source))?;
        let routes = Qwen38FlashNextMtpFusionRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Projects each draft row's four branch rows.
    ///
    /// # Safety
    ///
    /// `input` and `output` cover `rows * HC_WIDTH` BF16 values. `weight`
    /// covers one `[HIDDEN, HIDDEN]` BF16 plane. Pointers are aligned,
    /// non-overlapping, context-local, and live through stream completion.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                unsafe { $route.launch(&self.module, stream, input, weight, output) }
            };
        }

        dispatch_qwen38_flash_next_mtp_fusion!(&self.routes, rows, |route| launch!(route), else => Err(
            GpuError::invalid_launch(format!(
                "Qwen3.8 Flash-Next MTP fusion rows {rows} are outside 1 or T={PREFILL_ROUTES:?}"
            ))
        ))
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_qwen38_flash_next_projection),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct Qwen38FlashNextProjectionRoutes<E: ProjectionEntries> {
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
    #[route(1024)]
    t1024: E::Prefill<1_024>,
}

fn unsupported_rows<E: ProjectionEntries>(rows: usize) -> GpuError {
    GpuError::invalid_launch(format!(
        "{} projection rows {rows} are outside exact B=1..={MAX_BATCH} or T={PREFILL_ROUTES:?}",
        E::LABEL
    ))
}

/// Prepared source-BF16 backbone projection routes for one shape.
pub struct Qwen38FlashNextProjectionOp<E: ProjectionEntries> {
    module: kernels::LoadedModule,
    routes: Qwen38FlashNextProjectionRoutes<E>,
}

/// Prepared routes writing the fused GDN QKV/Z plane the mixer reads.
pub type Qwen38FlashNextGdnInputProjectionOp =
    Qwen38FlashNextProjectionOp<Qwen38FlashNextGdnInputEntries>;

/// Prepared routes writing the fused sparse-attention QKV plane.
pub type Qwen38FlashNextQsaQkvProjectionOp =
    Qwen38FlashNextProjectionOp<Qwen38FlashNextQsaQkvEntries>;

/// Prepared routes reducing a block's output to the residual-stream width.
///
/// One owner serves the gated DeltaNet `out_proj` and the sparse-attention
/// `o_proj`: the geometry and the numerics are identical, and the call sites
/// differ only in the weight plane they pass to [`Qwen38FlashNextProjectionOp::launch`].
pub type Qwen38FlashNextBlockOutputProjectionOp =
    Qwen38FlashNextProjectionOp<Qwen38FlashNextBlockOutputEntries>;

impl<E: ProjectionEntries> Qwen38FlashNextProjectionOp<E> {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded backbone projection artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_OPERATION, source))?;

        let routes = Qwen38FlashNextProjectionRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Applies the exact source-BF16 projection for one admitted row count.
    ///
    /// # Safety
    ///
    /// Pointers must be four-byte aligned, context-local, live through stream
    /// completion, and non-overlapping. `input` covers
    /// `rows * E::INPUT_COLUMNS` BF16 values, `weight` covers the materialized
    /// `[E::OUTPUT_ROWS, E::INPUT_COLUMNS]` BF16 plane, and `output` covers
    /// `rows * E::OUTPUT_ROWS` BF16 values.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
        weight: *const u16,
        output: *mut u16,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:expr) => {
                // SAFETY: exact-width dispatch preserves the public pointer contract.
                unsafe { $route.launch(&self.module, stream, input, weight, output) }
            };
        }

        dispatch_qwen38_flash_next_projection!(&self.routes, rows, |route| launch!(route), else => Err(unsupported_rows::<E>(rows)))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_COLUMNS, BLOCK_OUTPUT_BLOCKS, GDN_INPUT_BLOCKS, GDN_INPUT_ROWS, HIDDEN, MAX_BATCH,
        MTP_FUSION_BLOCKS, MTP_FUSION_DECODE_ROWS, NARROW_THREADS, NARROW_WARPS, PREFILL_ROUTES,
        ProjectionEntries, QSA_QKV_BLOCKS, QSA_QKV_ROWS, Qwen38FlashNextBlockOutputEntries,
        Qwen38FlashNextGdnInputEntries, Qwen38FlashNextMtpFusionRoutes,
        Qwen38FlashNextProjectionRoutes, Qwen38FlashNextQsaQkvEntries, WIDE_THREADS, WIDE_WARPS,
        prefill_blocks, qwen38_flash_next_block_output_projection_ptx_names,
        qwen38_flash_next_gdn_input_projection_ptx_names,
        qwen38_flash_next_mtp_fusion_projection_ptx_names,
        qwen38_flash_next_qsa_qkv_projection_ptx_names, unsupported_rows,
    };
    use std::collections::BTreeSet;
    use tuisko_kernels_sm120_common::projection::ROWS_PER_TILE;

    /// Every row count the schedule admits, swept exhaustively so an
    /// unadmitted width cannot hide between the transcribed ones.
    fn admitted_schedule<E: ProjectionEntries>() -> Vec<usize> {
        (0..=2_048)
            .chain([usize::MAX])
            .filter(|rows| Qwen38FlashNextProjectionRoutes::<E>::contains(*rows))
            .collect()
    }

    /// The three shapes, read off the architecture rather than transcribed.
    #[test]
    fn every_shape_covers_its_materialized_source_plane() {
        assert_eq!(HIDDEN, 2_560);
        assert_eq!(GDN_INPUT_ROWS, 16_384);
        assert_eq!(QSA_QKV_ROWS, 13_312);
        assert_eq!(BLOCK_COLUMNS, 6_144);

        assert_eq!(GDN_INPUT_BLOCKS, 256);
        assert_eq!(QSA_QKV_BLOCKS, 208);
        assert_eq!(BLOCK_OUTPUT_BLOCKS, 80);
        assert_eq!(MTP_FUSION_DECODE_ROWS, 4);
        assert_eq!(MTP_FUSION_BLOCKS, 40);
        assert_eq!(WIDE_THREADS, 256);
        assert_eq!(NARROW_THREADS, 128);
        assert_eq!(
            GDN_INPUT_BLOCKS as usize * WIDE_WARPS * ROWS_PER_TILE,
            GDN_INPUT_ROWS
        );
        assert_eq!(
            QSA_QKV_BLOCKS as usize * WIDE_WARPS * ROWS_PER_TILE,
            QSA_QKV_ROWS
        );
        assert_eq!(
            BLOCK_OUTPUT_BLOCKS as usize * NARROW_WARPS * ROWS_PER_TILE,
            HIDDEN
        );
    }

    /// Every prompt tile's grid is its shape's output grid once per sixteen rows.
    #[test]
    fn prompt_grids_scale_with_the_token_tile_count() {
        for (blocks, label) in [
            (GDN_INPUT_BLOCKS, "gdn"),
            (QSA_QKV_BLOCKS, "qsa"),
            (BLOCK_OUTPUT_BLOCKS, "output"),
        ] {
            for tokens in PREFILL_ROUTES {
                assert_eq!(
                    prefill_blocks(blocks, tokens, label).unwrap(),
                    blocks * (tokens / 16) as u32
                );
            }
        }
        for rows in PREFILL_ROUTES.map(|rows| rows * MTP_FUSION_DECODE_ROWS) {
            assert_eq!(
                prefill_blocks(MTP_FUSION_BLOCKS, rows, "mtp fusion").unwrap(),
                MTP_FUSION_BLOCKS * (rows / 16) as u32
            );
        }
    }

    /// Each table publishes exactly the twelve names retaining its own
    /// specializations, so sharing one wrapper cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_twelve_routes() {
        for (declared, expected) in [
            (
                <Qwen38FlashNextGdnInputEntries as ProjectionEntries>::ptx_names(),
                qwen38_flash_next_gdn_input_projection_ptx_names(),
            ),
            (
                <Qwen38FlashNextQsaQkvEntries as ProjectionEntries>::ptx_names(),
                qwen38_flash_next_qsa_qkv_projection_ptx_names(),
            ),
            (
                <Qwen38FlashNextBlockOutputEntries as ProjectionEntries>::ptx_names(),
                qwen38_flash_next_block_output_projection_ptx_names(),
            ),
        ] {
            assert_eq!(declared, expected);
            assert_eq!(declared.len(), MAX_BATCH + PREFILL_ROUTES.len());
            assert_eq!(
                declared.iter().copied().collect::<BTreeSet<_>>().len(),
                declared.len()
            );
        }
    }

    /// The four inventories are disjoint: no shape names another's entry.
    #[test]
    fn the_four_shapes_share_no_entry() {
        let names = qwen38_flash_next_gdn_input_projection_ptx_names()
            .into_iter()
            .chain(qwen38_flash_next_qsa_qkv_projection_ptx_names())
            .chain(qwen38_flash_next_block_output_projection_ptx_names())
            .chain(qwen38_flash_next_mtp_fusion_projection_ptx_names())
            .collect::<Vec<_>>();

        assert_eq!(names.len(), 41);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 41);
    }

    #[test]
    fn mtp_fusion_admits_only_its_five_draft_routes() {
        assert_eq!(
            Qwen38FlashNextMtpFusionRoutes::admitted_rows(),
            [1, 32, 64, 128, 1_024]
        );
        for rows in 0..=1_024 {
            assert_eq!(
                Qwen38FlashNextMtpFusionRoutes::contains(rows),
                [1, 32, 64, 128, 1_024].contains(&rows)
            );
        }
    }

    #[test]
    fn row_routing_is_exact_for_every_admitted_width() {
        assert_eq!(
            admitted_schedule::<Qwen38FlashNextGdnInputEntries>(),
            [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]
        );
        assert_eq!(
            Qwen38FlashNextProjectionRoutes::<Qwen38FlashNextGdnInputEntries>::admitted_rows(),
            [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]
        );
    }

    /// An unadmitted row count keeps its shape's rejection wording.
    #[test]
    fn unadmitted_row_counts_name_their_shape() {
        for (message, error) in [
            (
                "Qwen3.8-Flash-Next GDN input projection rows 9 are outside exact B=1..=8 or T=[32, 64, 128, 1024]",
                unsupported_rows::<Qwen38FlashNextGdnInputEntries>(9),
            ),
            (
                "Qwen3.8-Flash-Next sparse-attention QKV projection rows 2051 are outside exact B=1..=8 or T=[32, 64, 128, 1024]",
                unsupported_rows::<Qwen38FlashNextQsaQkvEntries>(2_051),
            ),
            (
                "Qwen3.8-Flash-Next block output projection rows 0 are outside exact B=1..=8 or T=[32, 64, 128, 1024]",
                unsupported_rows::<Qwen38FlashNextBlockOutputEntries>(0),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }

    /// Every shape admits its own geometry and no other's.
    #[test]
    fn every_entry_table_admits_its_own_geometry() {
        assert!(<Qwen38FlashNextGdnInputEntries as ProjectionEntries>::require_geometry().is_ok());
        assert!(<Qwen38FlashNextQsaQkvEntries as ProjectionEntries>::require_geometry().is_ok());
        assert!(
            <Qwen38FlashNextBlockOutputEntries as ProjectionEntries>::require_geometry().is_ok()
        );

        assert_eq!(
            <Qwen38FlashNextGdnInputEntries as ProjectionEntries>::INPUT_COLUMNS,
            <Qwen38FlashNextQsaQkvEntries as ProjectionEntries>::INPUT_COLUMNS
        );
        assert_eq!(
            <Qwen38FlashNextBlockOutputEntries as ProjectionEntries>::OUTPUT_ROWS,
            <Qwen38FlashNextGdnInputEntries as ProjectionEntries>::INPUT_COLUMNS
        );
    }
}

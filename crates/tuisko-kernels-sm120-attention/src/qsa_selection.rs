//! Qwen3.8-Flash-Next sparse-attention selection: indexer, block scoring, and
//! gather attention that reads the positions the indexer named.
//!
//! The dense route in `paged_gqa` remains authoritative while a query's visible
//! count stays at or below `MAX_SELECTED`; this family serves every longer
//! query. The two agree bit for bit inside the dense band by construction: the
//! selection list is then the whole visible list in ascending order, and the
//! gather bodies are their dense twins with the position walk indirected.

use crate::device::paged_gqa::{
    DECODE_SHARED_VALUES, DECODE_THREADS, PREFILL_SHARED_BYTES, PREFILL_THREADS,
    selected_paged_gqa_partitioned, selected_paged_gqa_prefill_shared,
};
use crate::device::qsa_indexer::{
    COMPRESS_RATIO, INDEXER_HEADS, MAX_SELECTED, SELECT_ROW_TILE, SELECT_SHARED_WORDS,
    THREADS_PER_CTA, WARPS_PER_CTA, indexer_block_compress, indexer_prepare, indexer_score,
    indexer_select,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_model::{Arch, Qwen38FlashNext};

/// Largest decode batch the selection routes admit.
pub const SELECTION_MAX_BATCH: usize = 8;
/// Prompt widths the selection capture schedule admits.
pub const SELECTION_PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];
/// Widest selected-position list one query owns.
pub const SELECTION_MAX_SELECTED: usize = MAX_SELECTED;
/// Rows one scoring and selection tile owns.
pub const SELECTION_ROW_TILE: usize = SELECT_ROW_TILE;
/// Micro-blocks the block-key plane holds per physical cache page.
pub const SELECTION_BLOCKS_PER_PAGE: usize = 64 / COMPRESS_RATIO;
/// Candidate-block grids the scorer prepares, in ascending order.
///
/// Each is a multiple of the warps per CTA, so a bucket divides into whole CTAs
/// and the last bucket covers the architecture's own position ceiling.
pub const SELECTION_BLOCK_BUCKETS: [usize; 5] = [64, 512, 4_096, 32_768, 65_536];
/// Candidate blocks the widest prepared grid scores.
pub const SELECTION_MAX_BLOCKS: usize = Qwen38FlashNext::MAX_POSITION_EMBEDDINGS / COMPRESS_RATIO;

const _: () = assert!(SELECTION_MAX_BLOCKS == 65_536);
const _: () = assert!(SELECTION_BLOCK_BUCKETS[4] == SELECTION_MAX_BLOCKS);
const _: () = assert!(SELECTION_MAX_SELECTED == 2_051);
const _: () = assert!(SELECTION_ROW_TILE == 64);

const THREADS: u32 = THREADS_PER_CTA as u32;
const PREFILL_QUERY_WARPS: usize =
    <Qwen38FlashNext as Arch>::NUM_ATTENTION_HEADS / <Qwen38FlashNext as Arch>::NUM_KV_HEADS;

const _: () = assert!(PREFILL_QUERY_WARPS == 12);
const _: () = assert!(PREFILL_THREADS == PREFILL_QUERY_WARPS * 32);

/// Micro-blocks one round can complete for one sequence at a given width.
///
/// A decode row advances one position, so at most one block closes. A prompt
/// tile spanning `tokens` positions closes `tokens / 4`, plus one for a tile
/// whose first position is not itself block-aligned.
pub const fn selection_round_blocks(tokens: usize) -> usize {
    if tokens <= SELECTION_MAX_BATCH {
        1
    } else {
        tokens / COMPRESS_RATIO + 1
    }
}

/// Sequences one round's block compression owns at a given width.
pub const fn selection_round_rows(tokens: usize) -> usize {
    if tokens <= SELECTION_MAX_BATCH {
        tokens
    } else {
        1
    }
}

fn require_geometry() -> GpuResult<()> {
    if Qwen38FlashNext::INDEXER_HEADS != 4
        || Qwen38FlashNext::INDEXER_KV_HEADS != 1
        || Qwen38FlashNext::INDEXER_HEAD_DIM != 128
        || Qwen38FlashNext::INDEXER_ROWS != 640
        || Qwen38FlashNext::INDEXER_BUDGET != 2_048
        || Qwen38FlashNext::INDEXER_COMPRESS_RATIO != 4
        || <Qwen38FlashNext as Arch>::NUM_ATTENTION_HEADS != 24
        || <Qwen38FlashNext as Arch>::NUM_KV_HEADS != 2
        || <Qwen38FlashNext as Arch>::HEAD_DIM != 256
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next geometry is incompatible with the admitted QSA selection schedule",
        ));
    }

    Ok(())
}

fn warp_blocks(warps: usize, overflow: &'static str) -> GpuResult<u32> {
    u32::try_from(warps.div_ceil(WARPS_PER_CTA)).map_err(|_| GpuError::invalid_launch(overflow))
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Prepares indexer queries and appends raw indexer keys for one batch.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_indexer_prepare_exact<const TOKENS: usize>(
        indexer_qk: *const u16,
        query_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        indexer_pages: *mut u16,
    ) {
        // Five head-warps per token: four indexer query heads and the single
        // shared key head, one warp per complete 128-wide vector so the RMS
        // reduction and the half-split lane exchange stay inside one warp.
        unsafe {
            indexer_prepare::<TOKENS>(
                indexer_qk,
                query_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                indexer_pages,
            );
        }
    }

    /// Prepares indexer queries and appends raw indexer keys for one prompt tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_indexer_prepare_prefill_exact<const TOKENS: usize>(
        indexer_qk: *const u16,
        query_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        indexer_pages: *mut u16,
    ) {
        unsafe {
            indexer_prepare::<TOKENS>(
                indexer_qk,
                query_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                indexer_pages,
            );
        }
    }

    /// Compresses one round's newly completed micro-blocks into block keys.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_indexer_block_compress_exact<
        const ROWS: usize,
        const BLOCKS: usize,
    >(
        indexer_pages: *const u16,
        key_norm: *const u16,
        block_rope_cos: *const f32,
        block_rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        first_blocks: *const u32,
        block_counts: *const u32,
        block_keys: *mut u16,
    ) {
        // One warp owns one completing block: it gathers four raw keys, pools
        // them in FP32, norms and rotates, and publishes one BF16 vector.
        unsafe {
            indexer_block_compress::<ROWS, BLOCKS>(
                indexer_pages,
                key_norm,
                block_rope_cos,
                block_rope_sin,
                block_tables,
                table_rows,
                table_stride,
                first_blocks,
                block_counts,
                block_keys,
            );
        }
    }

    /// Scores every candidate block of one row tile.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_indexer_score_exact<const ROWS: usize>(
        query: *const f32,
        block_keys: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        block_counts: *const u32,
        scores: *mut f32,
        score_stride: u32,
        row_offset: u32,
        blocks_per_row: u32,
    ) {
        // Eight warps per CTA score eight adjacent candidates against the same
        // four query heads, so the prepared grid is the bucket that covers the
        // round's candidate count divided into whole CTAs.
        unsafe {
            indexer_score::<ROWS>(
                query,
                block_keys,
                block_tables,
                table_rows,
                table_stride,
                block_counts,
                scores,
                score_stride,
                row_offset,
                blocks_per_row,
            );
        }
    }

    /// Selects the top-512 blocks of one row tile and expands them to positions.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_indexer_select_exact<const ROWS: usize>(
        scores: *const f32,
        visible_lengths: *const u32,
        block_counts: *const u32,
        selected: *mut u32,
        selected_counts: *mut u32,
        score_stride: u32,
        row_offset: u32,
    ) {
        // One CTA owns one row. The eight private histograms and the scan row
        // are static shared: the radix select's four passes reuse them, and the
        // entry carries no dynamic shared allocation.
        static mut SELECT_SCRATCH: SharedArray<u32, SELECT_SHARED_WORDS, 16> = SharedArray::UNINIT;
        let shared = core::ptr::addr_of_mut!(SELECT_SCRATCH).cast::<u32>();

        unsafe {
            indexer_select::<ROWS>(
                scores,
                visible_lengths,
                block_counts,
                selected,
                selected_counts,
                score_stride,
                row_offset,
                shared,
            );
        }
    }

    /// Runs one exact decode step over the selected positions.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_paged_gqa_selected_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        selected: *const u32,
        selected_counts: *const u32,
        selected_stride: u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // The dense decode entry's slice division and merge order, unchanged:
        // only the position each step reads is indirected.
        static mut SELECTED_DECODE_PARTIALS: SharedArray<f32, DECODE_SHARED_VALUES, 16> =
            SharedArray::UNINIT;
        let partials = core::ptr::addr_of_mut!(SELECTED_DECODE_PARTIALS).cast::<f32>();

        unsafe {
            selected_paged_gqa_partitioned::<Qwen38FlashNext, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                selected,
                selected_counts,
                selected_stride,
                output,
                key_scale,
                value_scale,
                partials,
            );
        }
    }

    /// Runs one exact prompt tile over the per-row selected positions.
    #[kernel]
    #[launch_bounds(384, 1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 32768,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen38_flash_next_paged_gqa_prefill_selected_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        selected: *const u32,
        selected_counts: *const u32,
        selected_stride: u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // Twelve warps are the twelve query heads one KV head serves, so one
        // gathered 64-position tile feeds the whole group exactly as the dense
        // prompt entry's contiguous tile does.
        unsafe {
            selected_paged_gqa_prefill_shared::<Qwen38FlashNext, TOKENS, PREFILL_QUERY_WARPS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                selected,
                selected_counts,
                selected_stride,
                output,
                key_scale,
                value_scale,
            );
        }
    }
}

const _: () = assert!(PREFILL_SHARED_BYTES == 32_768);

/// Addresses one indexer prepare launch reads and writes.
#[derive(Clone, Copy)]
pub struct IndexerPrepareArgs {
    /// Fused indexer projection rows `[tokens, 640]`.
    pub indexer_qk: *const u16,
    /// Indexer query RMSNorm weights `[128]`.
    pub query_norm: *const u16,
    /// Rotary cosines `[tokens, 32]` at the launched positions.
    pub rope_cos: *const f32,
    /// Rotary sines `[tokens, 32]` at the launched positions.
    pub rope_sin: *const f32,
    /// Page table shared with the K/V planes.
    pub block_tables: *const u32,
    /// Table row each token's sequence owns.
    pub table_rows: *const u32,
    /// Entries between two table rows.
    pub table_stride: u32,
    /// Absolute cache position of each token.
    pub cache_positions: *const u32,
    /// Prepared indexer queries `[tokens, 4, 128]`.
    pub query: *mut f32,
    /// Raw indexer key plane, one 128-wide BF16 vector per cached token.
    pub indexer_pages: *mut u16,
}

/// Addresses one block-compression launch reads and writes.
#[derive(Clone, Copy)]
pub struct IndexerCompressArgs {
    /// Raw indexer key plane.
    pub indexer_pages: *const u16,
    /// Indexer key RMSNorm weights `[128]`.
    pub key_norm: *const u16,
    /// Rotary cosines `[rows, blocks, 32]` at each block's first token.
    pub block_rope_cos: *const f32,
    /// Rotary sines `[rows, blocks, 32]` at each block's first token.
    pub block_rope_sin: *const f32,
    /// Page table shared with the K/V planes.
    pub block_tables: *const u32,
    /// Table row each sequence owns.
    pub table_rows: *const u32,
    /// Entries between two table rows.
    pub table_stride: u32,
    /// First micro-block each sequence closes this round.
    pub first_blocks: *const u32,
    /// Micro-blocks each sequence closes this round.
    pub block_counts: *const u32,
    /// Block-key plane, one 128-wide BF16 vector per completed micro-block.
    pub block_keys: *mut u16,
}

/// Addresses one scoring and selection pair reads and writes.
#[derive(Clone, Copy)]
pub struct IndexerSelectionArgs {
    /// Prepared indexer queries `[tokens, 4, 128]`.
    pub query: *const f32,
    /// Block-key plane.
    pub block_keys: *const u16,
    /// Page table shared with the K/V planes.
    pub block_tables: *const u32,
    /// Table row each token's sequence owns.
    pub table_rows: *const u32,
    /// Entries between two table rows.
    pub table_stride: u32,
    /// Visible key count of each token.
    pub visible_lengths: *const u32,
    /// Complete candidate blocks each token sees.
    pub block_counts: *const u32,
    /// Score scratch `[SELECTION_ROW_TILE, score_stride]`.
    pub scores: *mut f32,
    /// Values between two score rows.
    pub score_stride: u32,
    /// Selected positions `[tokens, 2051]`, ascending.
    pub selected: *mut u32,
    /// Selected position count of each token.
    pub selected_counts: *mut u32,
}

/// Addresses one gather attention launch reads and writes.
#[derive(Clone, Copy)]
pub struct SelectedAttentionArgs {
    /// Prepared attention queries `[tokens, 24, 256]`.
    pub query: *const f32,
    /// Key page plane.
    pub key_pages: *const u8,
    /// Value page plane.
    pub value_pages: *const u8,
    /// Page table shared with the indexer planes.
    pub block_tables: *const u32,
    /// Table row each token's sequence owns.
    pub table_rows: *const u32,
    /// Entries between two table rows.
    pub table_stride: u32,
    /// Selected positions `[tokens, 2051]`, ascending.
    pub selected: *const u32,
    /// Selected position count of each token.
    pub selected_counts: *const u32,
    /// Attention output `[tokens, 24, 256]`.
    pub output: *mut f32,
    /// Represented key-plane scale.
    pub key_scale: f32,
    /// Represented value-plane scale.
    pub value_scale: f32,
}

struct PreparedPrepare<const TOKENS: usize> {
    decode: PreparedLaunch<kernels::__qwen38_flash_next_indexer_prepare_exact_CudaKernel<TOKENS>>,
}

struct PreparedPrefillPrepare<const TOKENS: usize> {
    prefill: PreparedLaunch<
        kernels::__qwen38_flash_next_indexer_prepare_prefill_exact_CudaKernel<TOKENS>,
    >,
}

struct PreparedCompress<const ROWS: usize, const BLOCKS: usize> {
    compress: PreparedLaunch<
        kernels::__qwen38_flash_next_indexer_block_compress_exact_CudaKernel<ROWS, BLOCKS>,
    >,
}

struct PreparedSelection<const ROWS: usize> {
    score: [PreparedLaunch<kernels::__qwen38_flash_next_indexer_score_exact_CudaKernel<ROWS>>;
        SELECTION_BLOCK_BUCKETS.len()],
    select: PreparedLaunch<kernels::__qwen38_flash_next_indexer_select_exact_CudaKernel<ROWS>>,
}

struct PreparedSelectedDecode<const TOKENS: usize> {
    attention:
        PreparedLaunch<kernels::__qwen38_flash_next_paged_gqa_selected_exact_CudaKernel<TOKENS>>,
}

struct PreparedSelectedPrefill<const TOKENS: usize> {
    attention: PreparedLaunch<
        kernels::__qwen38_flash_next_paged_gqa_prefill_selected_exact_CudaKernel<TOKENS>,
    >,
}

impl<const TOKENS: usize> PreparedPrepare<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = warp_blocks(
            TOKENS * (INDEXER_HEADS + 1),
            "Qwen3.8-Flash-Next indexer prepare grid exceeds u32",
        )?;

        Ok(Self {
            decode: module
                .prepare_qwen38_flash_next_indexer_prepare_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next indexer prepare route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: IndexerPrepareArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_indexer_prepare_exact::<TOKENS>(
                stream,
                &self.decode,
                args.indexer_qk,
                args.query_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.indexer_pages,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next indexer prepare", source)
            })
    }
}

impl<const TOKENS: usize> PreparedPrefillPrepare<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = warp_blocks(
            TOKENS * (INDEXER_HEADS + 1),
            "Qwen3.8-Flash-Next indexer prompt prepare grid exceeds u32",
        )?;

        Ok(Self {
            prefill: module
                .prepare_qwen38_flash_next_indexer_prepare_prefill_exact::<TOKENS>(
                    LaunchConfig1D::new(blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next indexer prompt prepare route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: IndexerPrepareArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_indexer_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prefill,
                args.indexer_qk,
                args.query_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.indexer_pages,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next indexer prepare", source)
            })
    }
}

impl<const ROWS: usize, const BLOCKS: usize> PreparedCompress<ROWS, BLOCKS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = warp_blocks(
            ROWS * BLOCKS,
            "Qwen3.8-Flash-Next indexer block compression grid exceeds u32",
        )?;

        Ok(Self {
            compress: module
                .prepare_qwen38_flash_next_indexer_block_compress_exact::<ROWS, BLOCKS>(
                    LaunchConfig1D::new(blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next indexer block compression route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: IndexerCompressArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_indexer_block_compress_exact::<ROWS, BLOCKS>(
                stream,
                &self.compress,
                args.indexer_pages,
                args.key_norm,
                args.block_rope_cos,
                args.block_rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.first_blocks,
                args.block_counts,
                args.block_keys,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next indexer block compression",
                    source,
                )
            })
    }
}

impl<const ROWS: usize> PreparedSelection<ROWS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let mut score = Vec::with_capacity(SELECTION_BLOCK_BUCKETS.len());
        for bucket in SELECTION_BLOCK_BUCKETS {
            let blocks = u32::try_from(ROWS * (bucket / WARPS_PER_CTA)).map_err(|_| {
                GpuError::invalid_launch("Qwen3.8-Flash-Next indexer scoring grid exceeds u32")
            })?;
            score.push(
                module
                    .prepare_qwen38_flash_next_indexer_score_exact::<ROWS>(LaunchConfig1D::new(
                        blocks, THREADS, 0,
                    ))
                    .map_err(|source| {
                        GpuError::launch(
                            "preparing the Qwen3.8-Flash-Next indexer scoring route",
                            source,
                        )
                    })?,
            );
        }

        Ok(Self {
            score: score
                .try_into()
                .map_err(|_| GpuError::invalid_launch("indexer scoring bucket count changed"))?,
            select: module
                .prepare_qwen38_flash_next_indexer_select_exact::<ROWS>(LaunchConfig1D::new(
                    u32::try_from(ROWS).map_err(|_| {
                        GpuError::invalid_launch(
                            "Qwen3.8-Flash-Next indexer selection grid exceeds u32",
                        )
                    })?,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next indexer selection route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch_score(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        index: usize,
        row_offset: u32,
        blocks_per_row: u32,
        args: IndexerSelectionArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_indexer_score_exact::<ROWS>(
                stream,
                &self.score[index],
                args.query,
                args.block_keys,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.block_counts,
                args.scores,
                args.score_stride,
                row_offset,
                blocks_per_row,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next indexer scoring", source)
            })
    }

    unsafe fn launch_select(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        row_offset: u32,
        args: IndexerSelectionArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_indexer_select_exact::<ROWS>(
                stream,
                &self.select,
                args.scores.cast_const(),
                args.visible_lengths,
                args.block_counts,
                args.selected,
                args.selected_counts,
                args.score_stride,
                row_offset,
            )
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.8-Flash-Next indexer selection", source)
            })
    }
}

impl<const TOKENS: usize> PreparedSelectedDecode<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * <Qwen38FlashNext as Arch>::NUM_ATTENTION_HEADS)
            .map_err(|_| {
                GpuError::invalid_launch("Qwen3.8-Flash-Next selected decode grid exceeds u32")
            })?;

        Ok(Self {
            attention: module
                .prepare_qwen38_flash_next_paged_gqa_selected_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    DECODE_THREADS as u32,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next selected decode route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: SelectedAttentionArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_paged_gqa_selected_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.selected,
                args.selected_counts,
                args.selected_stride(),
                args.output,
                args.key_scale,
                args.value_scale,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next selected attention",
                    source,
                )
            })
    }
}

impl<const TOKENS: usize> PreparedSelectedPrefill<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks =
            u32::try_from(TOKENS * <Qwen38FlashNext as Arch>::NUM_KV_HEADS).map_err(|_| {
                GpuError::invalid_launch("Qwen3.8-Flash-Next selected prompt grid exceeds u32")
            })?;

        Ok(Self {
            attention: module
                .prepare_qwen38_flash_next_paged_gqa_prefill_selected_exact::<TOKENS>(
                    LaunchConfig1D::new(
                        blocks,
                        PREFILL_THREADS as u32,
                        PREFILL_SHARED_BYTES as u32,
                    ),
                )
                .map_err(|source| {
                    GpuError::launch(
                        "preparing the Qwen3.8-Flash-Next selected prompt route",
                        source,
                    )
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: SelectedAttentionArgs,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_paged_gqa_prefill_selected_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.selected,
                args.selected_counts,
                args.selected_stride(),
                args.output,
                args.key_scale,
                args.value_scale,
            )
            .map_err(|source| {
                GpuError::launch(
                    "launching the Qwen3.8-Flash-Next selected attention",
                    source,
                )
            })
    }
}

/// Selects the prepared candidate-block grid covering a round's block count.
pub fn selection_block_bucket(blocks: usize) -> Option<usize> {
    SELECTION_BLOCK_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| bucket >= blocks)
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_indexer_prepare),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct IndexerPrepareRoutes {
    #[route(1)]
    b1: PreparedPrepare<1>,
    #[route(2)]
    b2: PreparedPrepare<2>,
    #[route(3)]
    b3: PreparedPrepare<3>,
    #[route(4)]
    b4: PreparedPrepare<4>,
    #[route(5)]
    b5: PreparedPrepare<5>,
    #[route(6)]
    b6: PreparedPrepare<6>,
    #[route(7)]
    b7: PreparedPrepare<7>,
    #[route(8)]
    b8: PreparedPrepare<8>,
    #[route(32)]
    t32: PreparedPrefillPrepare<32>,
    #[route(64)]
    t64: PreparedPrefillPrepare<64>,
    #[route(128)]
    t128: PreparedPrefillPrepare<128>,
    #[route(1024)]
    t1024: PreparedPrefillPrepare<1_024>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_indexer_compress),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct IndexerCompressRoutes {
    #[route(1)]
    b1: PreparedCompress<1, 1>,
    #[route(2)]
    b2: PreparedCompress<2, 1>,
    #[route(3)]
    b3: PreparedCompress<3, 1>,
    #[route(4)]
    b4: PreparedCompress<4, 1>,
    #[route(5)]
    b5: PreparedCompress<5, 1>,
    #[route(6)]
    b6: PreparedCompress<6, 1>,
    #[route(7)]
    b7: PreparedCompress<7, 1>,
    #[route(8)]
    b8: PreparedCompress<8, 1>,
    #[route(32)]
    t32: PreparedCompress<1, 9>,
    #[route(64)]
    t64: PreparedCompress<1, 17>,
    #[route(128)]
    t128: PreparedCompress<1, 33>,
    #[route(1024)]
    t1024: PreparedCompress<1, 257>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_indexer_selection),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64),
    inventory(false)
)]
struct IndexerSelectionRoutes {
    #[route(1)]
    b1: PreparedSelection<1>,
    #[route(2)]
    b2: PreparedSelection<2>,
    #[route(3)]
    b3: PreparedSelection<3>,
    #[route(4)]
    b4: PreparedSelection<4>,
    #[route(5)]
    b5: PreparedSelection<5>,
    #[route(6)]
    b6: PreparedSelection<6>,
    #[route(7)]
    b7: PreparedSelection<7>,
    #[route(8)]
    b8: PreparedSelection<8>,
    #[route(32)]
    t32: PreparedSelection<32>,
    #[route(64)]
    t64: PreparedSelection<64>,
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_selected_paged_gqa),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1024),
    inventory(false)
)]
struct SelectedPagedGqaRoutes {
    #[route(1)]
    b1: PreparedSelectedDecode<1>,
    #[route(2)]
    b2: PreparedSelectedDecode<2>,
    #[route(3)]
    b3: PreparedSelectedDecode<3>,
    #[route(4)]
    b4: PreparedSelectedDecode<4>,
    #[route(5)]
    b5: PreparedSelectedDecode<5>,
    #[route(6)]
    b6: PreparedSelectedDecode<6>,
    #[route(7)]
    b7: PreparedSelectedDecode<7>,
    #[route(8)]
    b8: PreparedSelectedDecode<8>,
    #[route(32)]
    t32: PreparedSelectedPrefill<32>,
    #[route(64)]
    t64: PreparedSelectedPrefill<64>,
    #[route(128)]
    t128: PreparedSelectedPrefill<128>,
    #[route(1024)]
    t1024: PreparedSelectedPrefill<1_024>,
}

/// Prepared Qwen3.8-Flash-Next indexer prepare routes for every admitted width.
pub struct Qwen38FlashNextIndexerPrepareOp {
    module: kernels::LoadedModule,
    prepare_routes: IndexerPrepareRoutes,
    compress_routes: IndexerCompressRoutes,
}

impl Qwen38FlashNextIndexerPrepareOp {
    /// Loads the embedded module and prepares every admitted indexer route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry()?;
        let _ = qwen38_flash_next_indexer_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next QSA selection", source)
        })?;

        Ok(Self {
            prepare_routes: IndexerPrepareRoutes::prepare(&module)?,
            compress_routes: IndexerCompressRoutes::prepare(&module)?,
            module,
        })
    }

    /// Publishes the indexer queries and appends this round's raw indexer keys.
    ///
    /// # Safety
    ///
    /// `indexer_qk` covers `[tokens, 640]` BF16 rows and `query` `[tokens, 4,
    /// 128]` FP32 values. `indexer_pages` covers `[pages, 64, 128]` BF16
    /// values addressed through the same block table the K/V planes use. Every
    /// cache position is mapped by its token's table row. Allocations are
    /// aligned, disjoint, live through completion, and belong to `stream`'s
    /// context.
    pub unsafe fn launch_prepare(
        &self,
        stream: &CudaStream,
        tokens: usize,
        args: IndexerPrepareArgs,
    ) -> GpuResult<()> {
        dispatch_indexer_prepare!(
            &self.prepare_routes,
            tokens,
            |route| unsafe { route.launch(&self.module, stream, args) },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next indexer prepare tokens {tokens} must be one of \
                 1..={SELECTION_MAX_BATCH},32,64,128,1024"
            )))
        )
    }

    /// Compresses this round's newly completed micro-blocks into block keys.
    ///
    /// # Safety
    ///
    /// `block_keys` covers `[pages, 16, 128]` BF16 values on the same block
    /// table as `indexer_pages`, and every block the count plane names already
    /// holds all four of its raw keys. The rotary planes cover `[rows, blocks,
    /// 32]` FP32 values at the block-start positions. Allocations are aligned,
    /// disjoint, live through completion, and belong to `stream`'s context.
    pub unsafe fn launch_compress(
        &self,
        stream: &CudaStream,
        tokens: usize,
        args: IndexerCompressArgs,
    ) -> GpuResult<()> {
        dispatch_indexer_compress!(
            &self.compress_routes,
            tokens,
            |route| unsafe { route.launch(&self.module, stream, args) },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next indexer block compression tokens {tokens} must be one of \
                 1..={SELECTION_MAX_BATCH},32,64,128,1024"
            )))
        )
    }
}

/// Prepared Qwen3.8-Flash-Next block scoring and selection routes.
pub struct Qwen38FlashNextIndexerSelectionOp {
    module: kernels::LoadedModule,
    routes: IndexerSelectionRoutes,
}

impl Qwen38FlashNextIndexerSelectionOp {
    /// Loads the embedded module and prepares every admitted tile width.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry()?;
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next QSA selection", source)
        })?;

        Ok(Self {
            routes: IndexerSelectionRoutes::prepare(&module)?,
            module,
        })
    }

    /// Scores one row tile's candidate blocks and selects its top-512.
    ///
    /// # Safety
    ///
    /// `scores` covers `[rows, score_stride]` FP32 values with `score_stride`
    /// at least the round's candidate count, `selected` covers `[tokens,
    /// 2051]` and `selected_counts` `[tokens]`. Every named block is resident
    /// in the block-key plane. Allocations are aligned, disjoint, live through
    /// completion, and belong to `stream`'s context.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
        row_offset: usize,
        maximum_blocks: usize,
        args: IndexerSelectionArgs,
    ) -> GpuResult<()> {
        // SAFETY: the caller's plane contract reaches both stages unchanged.
        unsafe {
            self.launch_score(stream, rows, row_offset, maximum_blocks, args)?;
            self.launch_select(stream, rows, row_offset, args)
        }
    }

    /// Scores one row tile's candidate blocks into the score plane.
    ///
    /// Published beside [`Self::launch`] so a benchmark can separate the stage
    /// that grows with the context from the one that does not; a composed route
    /// should call `launch`, which runs the pair in the only admitted order.
    ///
    /// # Safety
    ///
    /// Carries [`Self::launch`]'s contract for the planes this stage reads.
    pub unsafe fn launch_score(
        &self,
        stream: &CudaStream,
        rows: usize,
        row_offset: usize,
        maximum_blocks: usize,
        args: IndexerSelectionArgs,
    ) -> GpuResult<()> {
        let bucket = selection_block_bucket(maximum_blocks).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next indexer candidate blocks {maximum_blocks} exceed \
                 {SELECTION_MAX_BLOCKS}"
            ))
        })?;
        let index = SELECTION_BLOCK_BUCKETS
            .iter()
            .position(|&candidate| candidate == bucket)
            .expect("the bucket came from the table");
        let row_offset = u32::try_from(row_offset)
            .map_err(|_| GpuError::invalid_launch("selection row offset exceeds u32"))?;
        let blocks_per_row = u32::try_from(bucket)
            .map_err(|_| GpuError::invalid_launch("selection block bucket exceeds u32"))?;

        dispatch_indexer_selection!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch_score(
                    &self.module,
                    stream,
                    index,
                    row_offset,
                    blocks_per_row,
                    args,
                )
            },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next indexer selection rows {rows} must be one of \
                 1..={SELECTION_MAX_BATCH},32,64"
            )))
        )
    }

    /// Selects one row tile's top-512 blocks from the published score plane.
    ///
    /// # Safety
    ///
    /// Carries [`Self::launch`]'s contract, and the score plane must already
    /// hold this tile's scores.
    pub unsafe fn launch_select(
        &self,
        stream: &CudaStream,
        rows: usize,
        row_offset: usize,
        args: IndexerSelectionArgs,
    ) -> GpuResult<()> {
        let row_offset = u32::try_from(row_offset)
            .map_err(|_| GpuError::invalid_launch("selection row offset exceeds u32"))?;

        dispatch_indexer_selection!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch_select(&self.module, stream, row_offset, args)
            },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next indexer selection rows {rows} must be one of \
                 1..={SELECTION_MAX_BATCH},32,64"
            )))
        )
    }
}

/// Prepared Qwen3.8-Flash-Next gather attention over selected positions.
pub struct Qwen38FlashNextSelectedPagedGqaOp {
    module: kernels::LoadedModule,
    routes: SelectedPagedGqaRoutes,
}

impl Qwen38FlashNextSelectedPagedGqaOp {
    /// Loads the embedded module and prepares every admitted width.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry()?;
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.8-Flash-Next QSA selection", source)
        })?;

        Ok(Self {
            routes: SelectedPagedGqaRoutes::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over the positions the indexer selected.
    ///
    /// # Safety
    ///
    /// Carries the dense route's plane contract, and additionally: `selected`
    /// covers `[tokens, 2051]` ascending positions, every count is nonzero and
    /// no greater than 2051, and every named position is mapped by its token's
    /// block-table row.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        args: SelectedAttentionArgs,
    ) -> GpuResult<()> {
        if !args.key_scale.is_finite()
            || args.key_scale <= 0.0
            || !args.value_scale.is_finite()
            || args.value_scale <= 0.0
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8-Flash-Next selected attention cache scales must be finite and positive",
            ));
        }
        if args.table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.8-Flash-Next selected attention table stride must be nonzero",
            ));
        }

        dispatch_selected_paged_gqa!(
            &self.routes,
            tokens,
            |route| unsafe { route.launch(&self.module, stream, args) },
            else => Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next selected attention tokens {tokens} must be one of \
                 1..={SELECTION_MAX_BATCH},32,64,128,1024"
            )))
        )
    }
}

impl SelectedAttentionArgs {
    fn selected_stride(&self) -> u32 {
        SELECTION_MAX_SELECTED as u32
    }
}

/// PTX symbols retained for every admitted selection route.
pub(crate) fn qwen38_flash_next_indexer_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_indexer_prepare_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_indexer_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_indexer_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_indexer_prepare_prefill_exact_ptx_name::<128>(),
        kernels::qwen38_flash_next_indexer_prepare_prefill_exact_ptx_name::<1_024>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<1, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<2, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<3, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<4, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<5, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<6, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<7, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<8, 1>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<1, 9>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<1, 17>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<1, 33>(),
        kernels::qwen38_flash_next_indexer_block_compress_exact_ptx_name::<1, 257>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_indexer_score_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_indexer_select_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_paged_gqa_selected_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_selected_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_selected_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_selected_exact_ptx_name::<128>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_selected_exact_ptx_name::<1_024>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        IndexerCompressRoutes, IndexerPrepareRoutes, IndexerSelectionRoutes,
        SELECTION_BLOCK_BUCKETS, SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_BLOCKS,
        SELECTION_MAX_SELECTED, SELECTION_ROW_TILE, SelectedPagedGqaRoutes,
        qwen38_flash_next_indexer_ptx_names, selection_block_bucket, selection_round_blocks,
        selection_round_rows,
    };
    use crate::device::qsa_indexer::{SELECT_SHARED_WORDS, WARPS_PER_CTA};
    use std::collections::BTreeSet;

    #[test]
    fn the_selection_widths_are_the_reference_s_own_arithmetic() {
        assert_eq!(SELECTION_MAX_SELECTED, 2_051);
        assert_eq!(SELECTION_MAX_BLOCKS, 65_536);
        assert_eq!(SELECTION_BLOCKS_PER_PAGE, 16);
        assert_eq!(SELECTION_ROW_TILE, 64);
        // Eight private histograms of 256 bins, the warp scan row, and the two
        // scalars a radix pass publishes.
        assert_eq!(SELECT_SHARED_WORDS * size_of::<u32>(), 8_232);

        for bucket in SELECTION_BLOCK_BUCKETS {
            assert_eq!(bucket % WARPS_PER_CTA, 0, "bucket {bucket} splits unevenly");
        }
        assert_eq!(selection_block_bucket(0), Some(64));
        assert_eq!(selection_block_bucket(64), Some(64));
        assert_eq!(selection_block_bucket(65), Some(512));
        assert_eq!(selection_block_bucket(32_768), Some(32_768));
        assert_eq!(selection_block_bucket(32_769), Some(65_536));
        assert_eq!(selection_block_bucket(65_536), Some(65_536));
        assert_eq!(selection_block_bucket(65_537), None);
    }

    #[test]
    fn one_round_closes_the_blocks_its_positions_complete() {
        for batch in 1..=8usize {
            assert_eq!(selection_round_rows(batch), batch);
            assert_eq!(selection_round_blocks(batch), 1);
        }
        for (tokens, blocks) in [(32, 9), (64, 17), (128, 33), (1_024, 257)] {
            assert_eq!(selection_round_rows(tokens), 1);
            assert_eq!(selection_round_blocks(tokens), blocks);
        }
    }

    #[test]
    fn route_tables_cover_only_the_admitted_widths() {
        let full = vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];

        assert_eq!(IndexerPrepareRoutes::admitted_rows(), full);
        assert_eq!(IndexerCompressRoutes::admitted_rows(), full);
        assert_eq!(
            IndexerSelectionRoutes::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 32, 64]
        );
        assert_eq!(SelectedPagedGqaRoutes::admitted_rows(), full);
    }

    #[test]
    fn ptx_inventory_covers_every_admitted_selection_route() {
        let names = qwen38_flash_next_indexer_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 56);
        assert_eq!(unique.len(), names.len());
    }
}

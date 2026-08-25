//! Exact decode and paged-prefill grouped-query attention.

use crate::Sm120Arch;
use crate::device::paged_gqa::{
    BF16_PREFILL_SHARED_BYTES, BF16_PREFILL_THREADS, DECODE_SHARED_VALUES, DECODE_THREADS,
    FLASH_PREFILL_P8_SHARED_BYTES, FLASH_PREFILL_P16_SHARED_BYTES, FLASH_PREFILL_THREADS,
    PREFILL_PARTIAL_VALUES, PREFILL_SHARED_BYTES, PREFILL_THREADS, QWEN35_BF16_PREFILL_THREADS,
    QWEN36_FP8_PREFILL_THREADS, bf16_paged_gqa, bf16_paged_gqa_prefill_shared, paged_gqa,
    paged_gqa_partitioned, paged_gqa_prefill_flash_partitioned,
    paged_gqa_prefill_partitioned_reduce, paged_gqa_prefill_shared,
};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const THREADS: u32 = 32;
const DECODE_THREADS_U32: u32 = DECODE_THREADS as u32;
const PREFILL_SHARED_BYTES_U32: u32 = PREFILL_SHARED_BYTES as u32;
const BF16_PREFILL_SHARED_BYTES_U32: u32 = BF16_PREFILL_SHARED_BYTES as u32;
const FLASH_PREFILL_P8_SHARED_BYTES_U32: u32 = FLASH_PREFILL_P8_SHARED_BYTES as u32;
const FLASH_PREFILL_P16_SHARED_BYTES_U32: u32 = FLASH_PREFILL_P16_SHARED_BYTES as u32;
const QWEN35_PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const QWEN36_PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
/// First context length routed to the sixteen-partition T=128 schedule.
pub const PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT: usize = 32_769;
/// Largest admitted T=128 prefill context.
pub const PAGED_GQA_PREFILL_MAX_CONTEXT: usize = 220_000;
/// Maximum resident FP32 partial workspace for partitioned T=128 prefill.
pub const PAGED_GQA_PREFILL_PARTIAL_BYTES: usize =
    128 * 24 * 16 * PREFILL_PARTIAL_VALUES * size_of::<f32>();
/// Exact token width of one macro-prefill attention route.
pub const PAGED_GQA_PREFILL_MACRO_TOKENS: usize = 1_024;
/// Largest admitted macro-prefill partition count.
pub const PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS: usize = 16;
/// Maximum resident FP32 partial workspace for macro prefill.
pub const PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES: usize = PAGED_GQA_PREFILL_MACRO_TOKENS
    * 24
    * PAGED_GQA_PREFILL_MACRO_MAX_PARTITIONS
    * PREFILL_PARTIAL_VALUES
    * size_of::<f32>();

/// Returns the exact partition count for one admitted deep T=128 context.
pub fn paged_gqa_prefill_partitions(context_tokens: usize) -> GpuResult<usize> {
    if !(129..=PAGED_GQA_PREFILL_MAX_CONTEXT).contains(&context_tokens) {
        return Err(GpuError::invalid_launch(format!(
            "partitioned paged GQA prefill context {context_tokens} is outside 129..={PAGED_GQA_PREFILL_MAX_CONTEXT}"
        )));
    }

    Ok(
        if context_tokens < PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT {
            8
        } else {
            16
        },
    )
}

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn admitted_macro_partitions(partitions: usize) -> bool {
    matches!(partitions, 1 | 2 | 4 | 8 | 16)
}

fn require_qwen38_geometry<A: Arch>() -> GpuResult<()> {
    if A::NUM_ATTENTION_HEADS != 24
        || A::NUM_KV_HEADS != 4
        || A::HEAD_DIM != 256
        || A::ATTENTION_OUTPUT_COLUMNS != 6_144
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the admitted paged GQA schedule",
        ));
    }

    Ok(())
}

fn require_qwen35_geometry() -> GpuResult<()> {
    if Qwen35_9B::NUM_ATTENTION_HEADS != 16
        || Qwen35_9B::NUM_KV_HEADS != 4
        || Qwen35_9B::HEAD_DIM != 256
        || Qwen35_9B::ATTENTION_OUTPUT_COLUMNS != 4_096
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.5 geometry is incompatible with its admitted paged GQA schedule",
        ));
    }

    Ok(())
}

fn require_qwen36_geometry() -> GpuResult<()> {
    if Qwen36Moe35B::NUM_ATTENTION_HEADS != 16
        || Qwen36Moe35B::NUM_KV_HEADS != 2
        || Qwen36Moe35B::HEAD_DIM != 256
        || Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS != 4_096
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.6 geometry is incompatible with its admitted paged GQA schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Applies paged FP8 GQA for one exact decode batch.
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
    pub fn paged_gqa_exact<A: Arch, const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        static mut DECODE_PARTIALS: SharedArray<f32, DECODE_SHARED_VALUES, 16> =
            SharedArray::UNINIT;
        let partials = core::ptr::addr_of_mut!(DECODE_PARTIALS).cast::<f32>();

        // Eight warps split the context into contiguous slices and merge their
        // online-softmax states in ascending slice order.
        unsafe {
            paged_gqa_partitioned::<A, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
                partials,
            );
        }
    }

    /// Applies shared-cache paged FP8 GQA for one exact prefill tile.
    #[kernel]
    #[launch_bounds(384, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (384, 1, 1),
        dynamic_shared = 32768,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_shared_exact<A: Arch, const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // A CTA owns two adjacent tokens and one KV head. Its twelve warps
        // cover two tokens by six query heads while one 64-position, 32-KiB
        // K/V tile serves every consumer; T=32 still exposes 64 CTAs.
        unsafe {
            paged_gqa_prefill_shared::<A, TOKENS, 2, 6>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            );
        }
    }

    /// Applies Qwen3.5 paged BF16 GQA for one exact decode batch.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_paged_gqa_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        // One warp covers the 256 columns as eight values/lane and scans
        // 130 * 256 * 2 * 2 = 133,120 BF16 cache bytes/head at the benchmark
        // context. The 16/128 B=1/B=8 CTAs preserve the established per-head
        // online-softmax order; grouping heads would be a new arithmetic route.
        unsafe {
            bf16_paged_gqa::<Qwen35_9B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            );
        }
    }

    /// Applies shared-cache Qwen3.5 BF16 GQA for one exact prompt width.
    #[kernel]
    #[launch_bounds(128, 1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
        dynamic_shared = 65536,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen35_paged_gqa_prefill_shared_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        // At T=128 the decode topology launches 2,048 one-warp CTAs and
        // rereads each 64-position BF16 K/V tile four times. This route uses
        // 512 one-token/KV-head CTAs: the four query-head warps share one
        // 65,536-byte tile. Each warp retains its head, key order, FP32
        // reduction, and online-softmax recurrence, so arithmetic is unchanged.
        unsafe {
            bf16_paged_gqa_prefill_shared::<Qwen35_9B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            );
        }
    }

    /// Applies Qwen3.6 paged BF16 GQA for one exact decode batch.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_paged_gqa_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        // One warp preserves the qualified 256-column online-softmax order.
        // Qwen3.6's 16 query heads still expose 16/128 B=1/B=8 CTAs, while
        // its 8:1 query/KV grouping changes only the selected cache head.
        unsafe {
            bf16_paged_gqa::<Qwen36Moe35B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            );
        }
    }

    /// Applies shared-cache Qwen3.6 BF16 GQA for one exact prompt width.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 65536,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_paged_gqa_prefill_shared_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        // At T=128 the decode topology launches 2,048 one-warp CTAs and
        // rereads each 64-position BF16 K/V tile eight times. This route uses
        // 256 one-token/KV-head CTAs: eight warps share one 65,536-byte tile.
        // Every warp retains its original head, key order, FP32 reduction, and
        // online-softmax recurrence, so the arithmetic is unchanged.
        unsafe {
            bf16_paged_gqa_prefill_shared::<Qwen36Moe35B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            );
        }
    }

    /// Applies paged E4M3 GQA for one exact Qwen3.6 decode batch.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_fp8_paged_gqa_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // One warp preserves the qualified 256-column online-softmax order.
        // Qwen3.6 exposes 16/128 B=1/B=8 CTAs; the 8:1 query/KV grouping
        // changes only the selected represented E4M3 cache head.
        unsafe {
            paged_gqa::<Qwen36Moe35B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            );
        }
    }

    /// Applies shared-cache E4M3 GQA for one exact Qwen3.6 prompt width.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 32768,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn qwen36_fp8_paged_gqa_prefill_shared_exact<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // At T=128, 256 one-token/KV-head CTAs replace 2,048 one-warp CTAs.
        // Eight query-head warps share one 32,768-byte E4M3 K/V tile while
        // retaining each head's key order, reduction, and softmax recurrence.
        unsafe {
            paged_gqa_prefill_shared::<Qwen36Moe35B, TOKENS, 1, 8>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            );
        }
    }

    /// Produces FP32 online-softmax states for the P8/K64 T=128 flash route.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 78336,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_flash_p8_exact<A: Arch>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partitions: u32,
        partials: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // K=64 halves the tile loop below 32,769 positions. Its 78,336-byte
        // single buffer permits one CTA/SM; P8 exposes 768 independent CTAs.
        unsafe {
            paged_gqa_prefill_flash_partitioned::<A, 64>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partitions,
                partials,
                key_scale,
                value_scale,
            );
        }
    }

    /// Produces FP32 online-softmax states for the P16/K32 T=128 flash route.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 43520,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_flash_p16_exact<A: Arch>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partitions: u32,
        partials: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // K=32 keeps two CTAs resident at deep contexts. P16 exposes 1,536
        // independent CTAs without the occupancy loss of a second buffer.
        unsafe {
            paged_gqa_prefill_flash_partitioned::<A, 32>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partitions,
                partials,
                key_scale,
                value_scale,
            );
        }
    }

    /// Merges exact FP32 T=128 partition states into the public output seam.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_partitioned_reduce_exact<A: Arch, const PARTITIONS: usize>(
        partials: *const f32,
        output: *mut f32,
    ) {
        unsafe {
            paged_gqa_prefill_partitioned_reduce::<A, 128, PARTITIONS>(partials, output);
        }
    }

    /// Produces FP32 online-softmax states for the K32 T=1024 macro route.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 43520,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_flash_macro_exact<A: Arch>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partitions: u32,
        partials: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // K=32 preserves two CTAs/SM, while even P1 exposes 768 independent
        // 32-row/query-head CTAs. More partitions only shorten each key scan.
        unsafe {
            paged_gqa_prefill_flash_partitioned::<A, 32>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partitions,
                partials,
                key_scale,
                value_scale,
            );
        }
    }

    /// Merges one exact T=1024 FP32 partition inventory into the output seam.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_macro_reduce_exact<A: Arch, const PARTITIONS: usize>(
        partials: *const f32,
        output: *mut f32,
    ) {
        unsafe {
            paged_gqa_prefill_partitioned_reduce::<A, PAGED_GQA_PREFILL_MACRO_TOKENS, PARTITIONS>(
                partials, output,
            );
        }
    }
}

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__paged_gqa_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_paged_gqa_exact::<A, TOKENS>(LaunchConfig1D::new(
                    blocks,
                    DECODE_THREADS_U32,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing paged GQA route", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .paged_gqa_exact::<A, TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching paged GQA", source))
    }
}

struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__paged_gqa_prefill_shared_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS / 2 * A::NUM_KV_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA prefill grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_paged_gqa_prefill_shared_exact::<A, TOKENS>(LaunchConfig1D::new(
                    blocks,
                    PREFILL_THREADS as u32,
                    PREFILL_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing shared paged GQA prefill route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .paged_gqa_prefill_shared_exact::<A, TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching shared paged GQA prefill", source))
    }
}

struct PreparedQwen35Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_paged_gqa_exact_CudaKernel<TOKENS>>,
}

struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN35_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = u32::try_from(TOKENS * Qwen35_9B::NUM_KV_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 paged GQA prefill grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_qwen35_paged_gqa_prefill_shared_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    QWEN35_BF16_PREFILL_THREADS as u32,
                    BF16_PREFILL_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 shared paged GQA prefill", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .qwen35_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * Qwen35_9B::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 paged GQA grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_qwen35_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| GpuError::launch("preparing Qwen3.5 paged GQA", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .qwen35_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 paged GQA", source))
    }
}

struct PreparedQwen36Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_paged_gqa_exact_CudaKernel<TOKENS>>,
}

struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = u32::try_from(TOKENS * Qwen36Moe35B::NUM_KV_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 paged GQA prefill grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_qwen36_paged_gqa_prefill_shared_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    BF16_PREFILL_THREADS as u32,
                    BF16_PREFILL_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 shared paged GQA prefill", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .qwen36_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * Qwen36Moe35B::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 paged GQA grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_qwen36_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| GpuError::launch("preparing Qwen3.6 paged GQA", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .qwen36_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 paged GQA", source))
    }
}

struct PreparedQwen36Fp8Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_fp8_paged_gqa_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Fp8Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * Qwen36Moe35B::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 FP8 paged GQA grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_qwen36_fp8_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| GpuError::launch("preparing Qwen3.6 FP8 paged GQA", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 paged GQA", source))
    }
}

struct PreparedQwen36Fp8PrefillRoute<const TOKENS: usize> {
    attention:
        PreparedLaunch<kernels::__qwen36_fp8_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Fp8PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 FP8 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = u32::try_from(TOKENS * Qwen36Moe35B::NUM_KV_HEADS).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 FP8 paged GQA prefill grid exceeds u32")
        })?;

        Ok(Self {
            attention: module
                .prepare_qwen36_fp8_paged_gqa_prefill_shared_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    QWEN36_FP8_PREFILL_THREADS as u32,
                    PREFILL_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 FP8 paged GQA prefill", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 paged GQA prefill", source))
    }
}

struct PreparedPartitionedPrefillP8<A: Arch> {
    partial: PreparedLaunch<kernels::__paged_gqa_prefill_flash_p8_exact_CudaKernel<A>>,
    reduce: PreparedLaunch<kernels::__paged_gqa_prefill_partitioned_reduce_exact_CudaKernel<A, 8>>,
}

impl<A: Arch> PreparedPartitionedPrefillP8<A> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let partial_blocks = u32::try_from(128 / 32 * A::NUM_ATTENTION_HEADS * 8)
            .map_err(|_| GpuError::invalid_launch("P8 flash paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(128 * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_flash_p8_exact::<A>(LaunchConfig1D::new(
                    partial_blocks,
                    FLASH_PREFILL_THREADS as u32,
                    FLASH_PREFILL_P8_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing P8 flash paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_partitioned_reduce_exact::<A, 8>(LaunchConfig1D::new(
                    reduce_blocks,
                    THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing P8 paged GQA reduction", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partials: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .paged_gqa_prefill_flash_p8_exact::<A>(
                stream,
                &self.partial,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                8,
                partials,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching P8 flash paged GQA prefill", source))?;
        module
            .paged_gqa_prefill_partitioned_reduce_exact::<A, 8>(
                stream,
                &self.reduce,
                partials,
                output,
            )
            .map_err(|source| GpuError::launch("launching P8 paged GQA reduction", source))
    }
}

struct PreparedPartitionedPrefillP16<A: Arch> {
    partial: PreparedLaunch<kernels::__paged_gqa_prefill_flash_p16_exact_CudaKernel<A>>,
    reduce: PreparedLaunch<kernels::__paged_gqa_prefill_partitioned_reduce_exact_CudaKernel<A, 16>>,
}

impl<A: Arch> PreparedPartitionedPrefillP16<A> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let partial_blocks = u32::try_from(128 / 32 * A::NUM_ATTENTION_HEADS * 16)
            .map_err(|_| GpuError::invalid_launch("P16 flash paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(128 * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_flash_p16_exact::<A>(LaunchConfig1D::new(
                    partial_blocks,
                    FLASH_PREFILL_THREADS as u32,
                    FLASH_PREFILL_P16_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing P16 flash paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_partitioned_reduce_exact::<A, 16>(LaunchConfig1D::new(
                    reduce_blocks,
                    THREADS,
                    0,
                ))
                .map_err(|source| GpuError::launch("preparing P16 paged GQA reduction", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partials: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .paged_gqa_prefill_flash_p16_exact::<A>(
                stream,
                &self.partial,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                16,
                partials,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching P16 flash paged GQA prefill", source))?;
        module
            .paged_gqa_prefill_partitioned_reduce_exact::<A, 16>(
                stream,
                &self.reduce,
                partials,
                output,
            )
            .map_err(|source| GpuError::launch("launching P16 paged GQA reduction", source))
    }
}

struct PreparedMacroPrefill<A: Arch, const PARTITIONS: usize> {
    partial: PreparedLaunch<kernels::__paged_gqa_prefill_flash_macro_exact_CudaKernel<A>>,
    reduce:
        PreparedLaunch<kernels::__paged_gqa_prefill_macro_reduce_exact_CudaKernel<A, PARTITIONS>>,
}

impl<A: Arch, const PARTITIONS: usize> PreparedMacroPrefill<A, PARTITIONS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let partial_blocks = u32::try_from(
            PAGED_GQA_PREFILL_MACRO_TOKENS / 32 * A::NUM_ATTENTION_HEADS * PARTITIONS,
        )
        .map_err(|_| GpuError::invalid_launch("macro flash paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(PAGED_GQA_PREFILL_MACRO_TOKENS * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("macro paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_flash_macro_exact::<A>(LaunchConfig1D::new(
                    partial_blocks,
                    FLASH_PREFILL_THREADS as u32,
                    FLASH_PREFILL_P16_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing macro flash paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_macro_reduce_exact::<A, PARTITIONS>(LaunchConfig1D::new(
                    reduce_blocks,
                    THREADS,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing macro paged GQA reduction", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partials: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .paged_gqa_prefill_flash_macro_exact::<A>(
                stream,
                &self.partial,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                PARTITIONS as u32,
                partials,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching macro flash paged GQA", source))?;
        module
            .paged_gqa_prefill_macro_reduce_exact::<A, PARTITIONS>(
                stream,
                &self.reduce,
                partials,
                output,
            )
            .map_err(|source| GpuError::launch("launching macro paged GQA reduction", source))
    }
}

/// Prepared paged GQA routes for exact `B=1..8` decode and early prefill tails.
pub struct PagedGqaOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedRoute<A, 1>,
    b2: PreparedRoute<A, 2>,
    b3: PreparedRoute<A, 3>,
    b4: PreparedRoute<A, 4>,
    b5: PreparedRoute<A, 5>,
    b6: PreparedRoute<A, 6>,
    b7: PreparedRoute<A, 7>,
    b8: PreparedRoute<A, 8>,
    t32: PreparedPrefillRoute<A, 32>,
    t64: PreparedPrefillRoute<A, 64>,
    t128: PreparedPrefillRoute<A, 128>,
    p8: PreparedPartitionedPrefillP8<A>,
    p16: PreparedPartitionedPrefillP16<A>,
    macro_p1: PreparedMacroPrefill<A, 1>,
    macro_p2: PreparedMacroPrefill<A, 2>,
    macro_p4: PreparedMacroPrefill<A, 4>,
    macro_p8: PreparedMacroPrefill<A, 8>,
    macro_p16: PreparedMacroPrefill<A, 16>,
}

impl<A: Sm120Arch> PagedGqaOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen38_geometry::<A>()?;
        let _ = paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading paged GQA", source))?;

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
            p8: PreparedPartitionedPrefillP8::prepare(&module)?,
            p16: PreparedPartitionedPrefillP16::prepare(&module)?,
            macro_p1: PreparedMacroPrefill::prepare(&module)?,
            macro_p2: PreparedMacroPrefill::prepare(&module)?,
            macro_p4: PreparedMacroPrefill::prepare(&module)?,
            macro_p8: PreparedMacroPrefill::prepare(&module)?,
            macro_p16: PreparedMacroPrefill::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented E4M3 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, 24, 256]` FP32 values. Cache planes
    /// use `[physical_page, 4, 64, 256]` E4M3 bytes. Metadata covers `batch`;
    /// each length is nonzero, its selected table row covers that length
    /// rounded up to 64, and every physical page ID is resident. Allocations are aligned,
    /// non-overlapping, live through completion, and belong to `stream`'s
    /// context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "paged GQA batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }
        let table_stride = u32::try_from(table_stride)
            .map_err(|_| GpuError::invalid_launch("paged GQA table stride exceeds u32"))?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "paged GQA table stride must be nonzero",
            ));
        }
        if !key_scale.is_finite() || key_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "paged GQA key scale must be finite and positive",
            ));
        }
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "paged GQA value scale must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        output,
                        key_scale,
                        value_scale,
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
            _ => unreachable!(),
        }
    }

    /// Applies shared-cache paged GQA to one exact `T=32/64/128` prefill tile.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens, 24, 256]` FP32 values. Cache planes
    /// use `[physical_page, 4, 64, 256]` represented E4M3 bytes. Metadata
    /// covers `tokens`; adjacent token pairs select the same table row, each
    /// causal length is nonzero and covered by that row, and every selected
    /// physical page is resident. Allocations are aligned, non-overlapping,
    /// live through completion, and belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill_shared(
        &self,
        stream: &CudaStream,
        tokens: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        let table_stride = validate_launch(table_stride, key_scale, value_scale)?;

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        output,
                        key_scale,
                        value_scale,
                    )
                }
            };
        }

        match tokens {
            32 => launch!(t32),
            64 => launch!(t64),
            128 => launch!(t128),
            _ => Err(GpuError::invalid_launch(format!(
                "paged GQA shared prefill tokens {tokens} are outside the admitted set 32, 64, 128"
            ))),
        }
    }

    /// Applies exact partitioned paged GQA to a deep `T=128` prefill tail.
    ///
    /// # Safety
    ///
    /// Query/output and cache metadata follow `launch_prefill_shared`. Every
    /// token selects the same resident table row, its causal length is at most
    /// `context_tokens`, and `partials` covers
    /// [`PAGED_GQA_PREFILL_PARTIAL_BYTES`] writable bytes. All allocations are
    /// aligned, non-overlapping, live through completion, and belong to the
    /// stream context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill_partitioned(
        &self,
        stream: &CudaStream,
        context_tokens: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        partials: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        let partitions = paged_gqa_prefill_partitions(context_tokens)?;
        let table_stride = validate_launch(table_stride, key_scale, value_scale)?;

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        partials,
                        output,
                        key_scale,
                        value_scale,
                    )
                }
            };
        }

        match partitions {
            8 => launch!(p8),
            16 => launch!(p16),
            _ => unreachable!(),
        }
    }

    /// Applies exact K32 flash GQA to one T=1024 macro-prefill tile.
    ///
    /// The admitted `partitions` inventory is `P=1,2,4,8,16`. Production
    /// scheduling selects P4; the other exact routes remain qualified tuning
    /// sentinels without introducing a generic fallback.
    ///
    /// # Safety
    ///
    /// Query/output cover `[1024, 24, 256]` FP32 values. Cache metadata covers
    /// every token, all tokens select the same live table row, and `partials`
    /// covers [`PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES`] writable bytes. The
    /// remaining allocation and stream requirements match
    /// `launch_prefill_partitioned`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill_macro(
        &self,
        stream: &CudaStream,
        partitions: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        partials: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !admitted_macro_partitions(partitions) {
            return Err(GpuError::invalid_launch(format!(
                "macro paged GQA partitions {partitions} are outside the admitted set 1, 2, 4, 8, 16"
            )));
        }
        let table_stride = validate_launch(table_stride, key_scale, value_scale)?;

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        partials,
                        output,
                        key_scale,
                        value_scale,
                    )
                }
            };
        }

        match partitions {
            1 => launch!(macro_p1),
            2 => launch!(macro_p2),
            4 => launch!(macro_p4),
            8 => launch!(macro_p8),
            16 => launch!(macro_p16),
            _ => unreachable!(),
        }
    }
}

fn validate_launch(table_stride: usize, key_scale: f32, value_scale: f32) -> GpuResult<u32> {
    let table_stride = u32::try_from(table_stride)
        .map_err(|_| GpuError::invalid_launch("paged GQA table stride exceeds u32"))?;
    if table_stride == 0 {
        return Err(GpuError::invalid_launch(
            "paged GQA table stride must be nonzero",
        ));
    }
    if !key_scale.is_finite() || key_scale <= 0.0 {
        return Err(GpuError::invalid_launch(
            "paged GQA key scale must be finite and positive",
        ));
    }
    if !value_scale.is_finite() || value_scale <= 0.0 {
        return Err(GpuError::invalid_launch(
            "paged GQA value scale must be finite and positive",
        ));
    }

    Ok(table_stride)
}

/// Prepared Qwen3.5 BF16 paged GQA routes for exact `B=1..8` and `T=32,64,128`.
pub struct Qwen35PagedGqaOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen35Route<1>,
    b2: PreparedQwen35Route<2>,
    b3: PreparedQwen35Route<3>,
    b4: PreparedQwen35Route<4>,
    b5: PreparedQwen35Route<5>,
    b6: PreparedQwen35Route<6>,
    b7: PreparedQwen35Route<7>,
    b8: PreparedQwen35Route<8>,
    t32: PreparedQwen35PrefillRoute<32>,
    t64: PreparedQwen35PrefillRoute<64>,
    t128: PreparedQwen35PrefillRoute<128>,
}

impl Qwen35PagedGqaOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen35_geometry()?;
        let _ = qwen35_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.5 paged GQA", source))?;

        Ok(Self {
            b1: PreparedQwen35Route::prepare(&module)?,
            b2: PreparedQwen35Route::prepare(&module)?,
            b3: PreparedQwen35Route::prepare(&module)?,
            b4: PreparedQwen35Route::prepare(&module)?,
            b5: PreparedQwen35Route::prepare(&module)?,
            b6: PreparedQwen35Route::prepare(&module)?,
            b7: PreparedQwen35Route::prepare(&module)?,
            b8: PreparedQwen35Route::prepare(&module)?,
            t32: PreparedQwen35PrefillRoute::prepare(&module)?,
            t64: PreparedQwen35PrefillRoute::prepare(&module)?,
            t128: PreparedQwen35PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens, 16, 256]` FP32 values. Cache planes
    /// use `[physical_page, 4, 64, 256]` BF16 values. Metadata covers `tokens`;
    /// each length is nonzero, its table row covers that length rounded up to
    /// 64, and every physical page is resident. Allocations are aligned,
    /// disjoint, live through completion, and belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        if !admitted_batch(tokens) && !QWEN35_PREFILL_TOKENS.contains(&tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 paged GQA tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128"
            )));
        }
        let table_stride = u32::try_from(table_stride)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 paged GQA table stride exceeds u32"))?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 paged GQA table stride must be nonzero",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        output,
                    )
                }
            };
        }

        match tokens {
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
            _ => unreachable!(),
        }
    }
}

/// Prepared Qwen3.6 BF16 paged GQA routes for exact `B=1..8` and `T=32,64,128`.
pub struct Qwen36PagedGqaOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen36Route<1>,
    b2: PreparedQwen36Route<2>,
    b3: PreparedQwen36Route<3>,
    b4: PreparedQwen36Route<4>,
    b5: PreparedQwen36Route<5>,
    b6: PreparedQwen36Route<6>,
    b7: PreparedQwen36Route<7>,
    b8: PreparedQwen36Route<8>,
    t32: PreparedQwen36PrefillRoute<32>,
    t64: PreparedQwen36PrefillRoute<64>,
    t128: PreparedQwen36PrefillRoute<128>,
}

impl Qwen36PagedGqaOp {
    /// Loads the embedded module and prepares every exact Qwen3.6 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen36_geometry()?;
        let _ = qwen36_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 paged GQA", source))?;

        Ok(Self {
            b1: PreparedQwen36Route::prepare(&module)?,
            b2: PreparedQwen36Route::prepare(&module)?,
            b3: PreparedQwen36Route::prepare(&module)?,
            b4: PreparedQwen36Route::prepare(&module)?,
            b5: PreparedQwen36Route::prepare(&module)?,
            b6: PreparedQwen36Route::prepare(&module)?,
            b7: PreparedQwen36Route::prepare(&module)?,
            b8: PreparedQwen36Route::prepare(&module)?,
            t32: PreparedQwen36PrefillRoute::prepare(&module)?,
            t64: PreparedQwen36PrefillRoute::prepare(&module)?,
            t128: PreparedQwen36PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens,16,256]` FP32 values. Cache planes use
    /// `[physical_page,2,64,256]` BF16 values. Metadata covers `tokens`; each
    /// length is nonzero, its table row covers that length rounded up to 64,
    /// and every physical page is resident. Allocations are aligned,
    /// disjoint, live through completion, and context-local.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        if !admitted_batch(tokens) && !QWEN36_PREFILL_TOKENS.contains(&tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 paged GQA tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128"
            )));
        }
        let table_stride = u32::try_from(table_stride)
            .map_err(|_| GpuError::invalid_launch("Qwen3.6 paged GQA table stride exceeds u32"))?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 paged GQA table stride must be nonzero",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        output,
                    )
                }
            };
        }

        match tokens {
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
            _ => unreachable!(),
        }
    }
}

/// Prepared Qwen3.6 E4M3 paged GQA routes for exact decode and prompt widths.
pub struct Qwen36Fp8PagedGqaOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen36Fp8Route<1>,
    b2: PreparedQwen36Fp8Route<2>,
    b3: PreparedQwen36Fp8Route<3>,
    b4: PreparedQwen36Fp8Route<4>,
    b5: PreparedQwen36Fp8Route<5>,
    b6: PreparedQwen36Fp8Route<6>,
    b7: PreparedQwen36Fp8Route<7>,
    b8: PreparedQwen36Fp8Route<8>,
    t32: PreparedQwen36Fp8PrefillRoute<32>,
    t64: PreparedQwen36Fp8PrefillRoute<64>,
    t128: PreparedQwen36Fp8PrefillRoute<128>,
}

impl Qwen36Fp8PagedGqaOp {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen36_geometry()?;
        let _ = qwen36_fp8_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 FP8 paged GQA", source))?;

        Ok(Self {
            b1: PreparedQwen36Fp8Route::prepare(&module)?,
            b2: PreparedQwen36Fp8Route::prepare(&module)?,
            b3: PreparedQwen36Fp8Route::prepare(&module)?,
            b4: PreparedQwen36Fp8Route::prepare(&module)?,
            b5: PreparedQwen36Fp8Route::prepare(&module)?,
            b6: PreparedQwen36Fp8Route::prepare(&module)?,
            b7: PreparedQwen36Fp8Route::prepare(&module)?,
            b8: PreparedQwen36Fp8Route::prepare(&module)?,
            t32: PreparedQwen36Fp8PrefillRoute::prepare(&module)?,
            t64: PreparedQwen36Fp8PrefillRoute::prepare(&module)?,
            t128: PreparedQwen36Fp8PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented E4M3 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens,16,256]` FP32 values. Cache planes use
    /// `[physical_page,2,64,256]` E4M3 bytes. Metadata covers `tokens`; each
    /// length is nonzero, its table row covers that length rounded up to 64,
    /// and every selected physical page is resident. Allocations are aligned,
    /// disjoint, live through completion, and context-local.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !admitted_batch(tokens) && !QWEN36_PREFILL_TOKENS.contains(&tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 FP8 paged GQA tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128"
            )));
        }
        let table_stride = validate_launch(table_stride, key_scale, value_scale)?;

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        query,
                        key_pages,
                        value_pages,
                        block_tables,
                        table_rows,
                        table_stride,
                        lengths,
                        output,
                        key_scale,
                        value_scale,
                    )
                }
            };
        }

        match tokens {
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
            _ => unreachable!(),
        }
    }
}

/// PTX symbols retained for every exact paged GQA route.
pub(crate) fn paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::paged_gqa_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::paged_gqa_prefill_shared_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::paged_gqa_prefill_shared_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::paged_gqa_prefill_shared_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::paged_gqa_prefill_flash_p8_exact_ptx_name::<Qwen38_27B>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::paged_gqa_prefill_flash_p16_exact_ptx_name::<Qwen38_27B>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 16>(),
        kernels::paged_gqa_prefill_flash_macro_exact_ptx_name::<Qwen38_27B>(),
        kernels::paged_gqa_prefill_macro_reduce_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::paged_gqa_prefill_macro_reduce_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::paged_gqa_prefill_macro_reduce_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::paged_gqa_prefill_macro_reduce_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::paged_gqa_prefill_macro_reduce_exact_ptx_name::<Qwen38_27B, 16>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.5 BF16 paged GQA route.
pub(crate) fn qwen35_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_paged_gqa_exact_ptx_name::<1>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<2>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<3>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<4>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<5>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<6>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<7>(),
        kernels::qwen35_paged_gqa_exact_ptx_name::<8>(),
        kernels::qwen35_paged_gqa_prefill_shared_exact_ptx_name::<32>(),
        kernels::qwen35_paged_gqa_prefill_shared_exact_ptx_name::<64>(),
        kernels::qwen35_paged_gqa_prefill_shared_exact_ptx_name::<128>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.6 BF16 paged GQA route.
pub(crate) fn qwen36_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_paged_gqa_exact_ptx_name::<1>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<2>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<3>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<4>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<5>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<6>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<7>(),
        kernels::qwen36_paged_gqa_exact_ptx_name::<8>(),
        kernels::qwen36_paged_gqa_prefill_shared_exact_ptx_name::<32>(),
        kernels::qwen36_paged_gqa_prefill_shared_exact_ptx_name::<64>(),
        kernels::qwen36_paged_gqa_prefill_shared_exact_ptx_name::<128>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.6 E4M3 paged GQA route.
pub(crate) fn qwen36_fp8_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<1>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<2>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<3>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<4>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<5>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<6>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<7>(),
        kernels::qwen36_fp8_paged_gqa_exact_ptx_name::<8>(),
        kernels::qwen36_fp8_paged_gqa_prefill_shared_exact_ptx_name::<32>(),
        kernels::qwen36_fp8_paged_gqa_prefill_shared_exact_ptx_name::<64>(),
        kernels::qwen36_fp8_paged_gqa_prefill_shared_exact_ptx_name::<128>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        BF16_PREFILL_SHARED_BYTES, BF16_PREFILL_THREADS, FLASH_PREFILL_P8_SHARED_BYTES,
        FLASH_PREFILL_P16_SHARED_BYTES, FLASH_PREFILL_THREADS,
        PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES,
        PAGED_GQA_PREFILL_MACRO_TOKENS, PAGED_GQA_PREFILL_PARTIAL_BYTES, PREFILL_SHARED_BYTES,
        PREFILL_THREADS, QWEN35_BF16_PREFILL_THREADS, QWEN35_PREFILL_TOKENS,
        QWEN36_FP8_PREFILL_THREADS, QWEN36_PREFILL_TOKENS, THREADS, admitted_batch,
        admitted_macro_partitions, paged_gqa_prefill_partitions, paged_gqa_ptx_names,
        qwen35_paged_gqa_ptx_names, qwen36_fp8_paged_gqa_ptx_names, qwen36_paged_gqa_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn batch_table_covers_only_exact_decode_routes() {
        for (batch, expected) in [(0, false), (1, true), (4, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
        assert_eq!(THREADS, 32);
    }

    #[test]
    fn ptx_inventory_has_every_decode_and_prefill_route() {
        let names = paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 21);
        assert_eq!(unique.len(), names.len());
        assert_eq!(PREFILL_THREADS, 384);
        assert_eq!(PREFILL_SHARED_BYTES, 32_768);
        assert_eq!(FLASH_PREFILL_THREADS, 256);
        assert_eq!(FLASH_PREFILL_P8_SHARED_BYTES, 78_336);
        assert_eq!(FLASH_PREFILL_P16_SHARED_BYTES, 43_520);
        assert_eq!(PAGED_GQA_PREFILL_PARTIAL_BYTES, 50_724_864);
        assert_eq!(PAGED_GQA_PREFILL_MACRO_TOKENS, 1_024);
        assert_eq!(PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, 405_798_912);

        let qwen35 = qwen35_paged_gqa_ptx_names();
        let qwen35_unique = qwen35.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen35.len(), 11);
        assert_eq!(qwen35_unique.len(), qwen35.len());
        assert!(names.iter().all(|name| !qwen35_unique.contains(name)));

        let qwen36 = qwen36_paged_gqa_ptx_names();
        let qwen36_unique = qwen36.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen36.len(), 11);
        let qwen36_fp8 = qwen36_fp8_paged_gqa_ptx_names();
        let qwen36_fp8_unique = qwen36_fp8.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen36_fp8.len(), 11);
        assert_eq!(qwen36_fp8_unique.len(), qwen36_fp8.len());
        assert!(qwen36_fp8_unique.is_disjoint(&qwen36_unique));
        assert_eq!(QWEN36_FP8_PREFILL_THREADS, 256);
        assert_eq!(qwen36_unique.len(), qwen36.len());
        assert!(names.iter().all(|name| !qwen36_unique.contains(name)));
        assert!(qwen35_unique.is_disjoint(&qwen36_unique));
        assert_eq!(QWEN35_PREFILL_TOKENS, [32, 64, 128]);
        assert_eq!(QWEN35_BF16_PREFILL_THREADS, 128);
        assert_eq!(QWEN36_PREFILL_TOKENS, [32, 64, 128]);
        assert_eq!(BF16_PREFILL_THREADS, 256);
        assert_eq!(BF16_PREFILL_SHARED_BYTES, 65_536);
    }

    #[test]
    fn partition_inventory_selects_only_the_admitted_context_bands() {
        assert!(paged_gqa_prefill_partitions(128).is_err());
        assert_eq!(paged_gqa_prefill_partitions(129).unwrap(), 8);
        assert_eq!(
            paged_gqa_prefill_partitions(PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT - 1).unwrap(),
            8
        );
        assert_eq!(
            paged_gqa_prefill_partitions(PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT).unwrap(),
            16
        );
        assert!(paged_gqa_prefill_partitions(220_001).is_err());
    }

    #[test]
    fn macro_partition_inventory_is_exact() {
        for partitions in 0..=17 {
            assert_eq!(
                admitted_macro_partitions(partitions),
                matches!(partitions, 1 | 2 | 4 | 8 | 16),
                "partitions={partitions}"
            );
        }
    }
}

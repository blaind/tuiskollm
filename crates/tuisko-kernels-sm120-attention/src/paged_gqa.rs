//! Exact decode and paged-prefill grouped-query attention.

use crate::device::paged_gqa::{
    BF16_PREFILL_SHARED_BYTES, BF16_PREFILL_THREADS, DECODE_RING_E4M3_SHARED_BYTES,
    DECODE_RING_SHARED_BYTES, DECODE_SHARED_VALUES, DECODE_THREADS, FLASH_PREFILL_P8_SHARED_BYTES,
    FLASH_PREFILL_P16_SHARED_BYTES, FLASH_PREFILL_THREADS, PREFILL_PARTIAL_VALUES,
    PREFILL_SHARED_BYTES, PREFILL_THREADS, QWEN35_BF16_PREFILL_THREADS, QWEN36_FP8_PREFILL_THREADS,
    bf16_paged_gqa, bf16_paged_gqa_prefill_shared, paged_gqa, paged_gqa_partitioned,
    paged_gqa_prefill_flash_partitioned, paged_gqa_prefill_partitioned_reduce,
    paged_gqa_prefill_shared,
};
use crate::qk_prepare::{Bf16Cache, CacheFormat, CacheScales, Fp8Cache};
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

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

// Each KV head serves 12 query heads, matching the shared-tile CTA warp count.
fn require_qwen38_flash_next_geometry() -> GpuResult<()> {
    if Qwen38FlashNext::NUM_ATTENTION_HEADS != 24
        || Qwen38FlashNext::NUM_KV_HEADS != 2
        || Qwen38FlashNext::HEAD_DIM != 256
        || Qwen38FlashNext::ATTENTION_OUTPUT_COLUMNS != 6_144
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next geometry is incompatible with its admitted QSA paged GQA schedule",
        ));
    }

    Ok(())
}

/// Query heads one Qwen3.8-Flash-Next QSA KV head serves, and the CTA's warp count.
const QWEN38_FLASH_NEXT_QUERY_WARPS: usize = 12;
/// Qwen3.8-Flash-Next QSA shared-tile prefill threads: twelve query-head warps.
const QWEN38_FLASH_NEXT_PREFILL_THREADS: usize = QWEN38_FLASH_NEXT_QUERY_WARPS * 32;
/// Prefill widths the Qwen3.8-Flash-Next QSA shared-tile schedule admits.
const QWEN38_FLASH_NEXT_PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];

const _: () = assert!(
    QWEN38_FLASH_NEXT_QUERY_WARPS
        == Qwen38FlashNext::NUM_ATTENTION_HEADS / Qwen38FlashNext::NUM_KV_HEADS
);
const _: () = assert!(QWEN38_FLASH_NEXT_PREFILL_THREADS == 384);

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
        dynamic_shared = 8192,
        dynamic_shared_alignment = 16,
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
        dynamic_shared = 8192,
        dynamic_shared_alignment = 16,
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
        dynamic_shared = 4096,
        dynamic_shared_alignment = 16,
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

    /// Applies dense causal paged E4M3 GQA for one Qwen3.8-Flash-Next QSA batch.
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
    pub fn qwen38_flash_next_paged_gqa_exact<const TOKENS: usize>(
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
        // Eight warps split the context into contiguous slices and merge their
        // online-softmax states in ascending slice order, so the sum order is
        // the same one the prefill route uses. Qwen3.8-Flash-Next exposes 24/192 CTAs
        // at B=1/B=8; the 12:1 query/KV grouping changes only which E4M3
        // cache head a query head reads.
        static mut QWEN38_FLASH_NEXT_DECODE_PARTIALS: SharedArray<f32, DECODE_SHARED_VALUES, 16> =
            SharedArray::UNINIT;
        let partials = core::ptr::addr_of_mut!(QWEN38_FLASH_NEXT_DECODE_PARTIALS).cast::<f32>();

        unsafe {
            paged_gqa_partitioned::<Qwen38FlashNext, TOKENS>(
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

    /// Applies dense causal shared-tile paged E4M3 GQA for one QSA prompt.
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
    pub fn qwen38_flash_next_paged_gqa_prefill_shared_exact<const TOKENS: usize>(
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
        // One CTA owns one token and one KV head. Its twelve warps are exactly
        // the twelve query heads that KV head serves, so one 64-position,
        // 32-KiB E4M3 K/V tile feeds every consumer and each head keeps its
        // own key order, reduction, and softmax recurrence. T=32 still exposes
        // 64 CTAs; T=1024 exposes 2,048.
        unsafe {
            paged_gqa_prefill_shared::<Qwen38FlashNext, TOKENS, 1, QWEN38_FLASH_NEXT_QUERY_WARPS>(
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
    pub fn paged_gqa_prefill_flash_p8_exact<A: Arch, const TOKENS: usize>(
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
    pub fn paged_gqa_prefill_flash_p16_exact<A: Arch, const TOKENS: usize>(
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

    /// Merges exact FP32 partition states into the public output seam.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (12, 0),
    )]
    pub fn paged_gqa_prefill_partitioned_reduce_exact<
        A: Arch,
        const TOKENS: usize,
        const PARTITIONS: usize,
    >(
        partials: *const f32,
        output: *mut f32,
    ) {
        unsafe {
            paged_gqa_prefill_partitioned_reduce::<A, TOKENS, PARTITIONS>(partials, output);
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

mod private {
    pub trait Sealed {}
}

/// One launch's paged GQA operands in the entries' parameter order.
///
/// Every admitted width passes the same operands to its entry, so bundling
/// them keeps the merged dispatch's argument order identical to the launchers
/// it replaces.
pub struct PagedGqaArgs<C: CacheFormat> {
    query: *const f32,
    key_pages: *const C::Element,
    value_pages: *const C::Element,
    block_tables: *const u32,
    table_rows: *const u32,
    table_stride: u32,
    lengths: *const u32,
    output: *mut f32,
    scales: C::Scales,
}

/// One architecture's prepared entry for an exact token width.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait PagedGqaRoute<C: CacheFormat>: Sized + private::Sealed {
    /// Prepares this route's entry at its qualified grid, block, and shared
    /// footprint.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's entry.
    ///
    /// # Safety
    ///
    /// `args` carries the owner's pointer contract unchanged.
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<C>,
    ) -> GpuResult<()>;
}

/// Decode CTA count: one CTA per token and query head.
fn decode_blocks<A: Arch>(tokens: usize, label: &str) -> GpuResult<u32> {
    u32::try_from(tokens * A::NUM_ATTENTION_HEADS)
        .map_err(|_| GpuError::invalid_launch(format!("{label}paged GQA grid exceeds u32")))
}

/// Shared-cache prefill CTA count: one CTA per token group and KV head.
fn prefill_blocks<A: Arch>(tokens: usize, tokens_per_cta: usize, label: &str) -> GpuResult<u32> {
    u32::try_from(tokens / tokens_per_cta * A::NUM_KV_HEADS)
        .map_err(|_| GpuError::invalid_launch(format!("{label}paged GQA prefill grid exceeds u32")))
}

/// Prepared Qwen3.8 E4M3 decode entry for one exact batch.
pub struct PreparedRoute<A: Arch, const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__paged_gqa_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.8 shared-cache prefill entry for one exact prompt width.
pub struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__paged_gqa_prefill_shared_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.5 BF16 decode entry for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_paged_gqa_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 shared-cache prefill entry for one exact prompt width.
pub struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 BF16 decode entry for one exact batch.
pub struct PreparedQwen36Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_paged_gqa_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 shared-cache prefill entry for one exact prompt width.
pub struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 E4M3 decode entry for one exact batch.
pub struct PreparedQwen36Fp8Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen36_fp8_paged_gqa_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 E4M3 shared-cache prefill entry for one exact prompt width.
pub struct PreparedQwen36Fp8PrefillRoute<const TOKENS: usize> {
    attention:
        PreparedLaunch<kernels::__qwen36_fp8_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8-Flash-Next QSA E4M3 decode entry for one exact batch.
pub struct PreparedQwen38FlashNextRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen38_flash_next_paged_gqa_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8-Flash-Next QSA shared-tile prefill entry for one prompt width.
pub struct PreparedQwen38FlashNextPrefillRoute<const TOKENS: usize> {
    attention: PreparedLaunch<
        kernels::__qwen38_flash_next_paged_gqa_prefill_shared_exact_CudaKernel<TOKENS>,
    >,
}

/// Stands in for a prefill width an entry table does not admit.
///
/// It prepares and launches no entry, so an unadmitted width can never reach
/// the device and never enters the emitted inventory.
pub struct UnadmittedRoute;

impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedRoute<A, TOKENS> {}
impl<A: Arch, const TOKENS: usize> private::Sealed for PreparedPrefillRoute<A, TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35PrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36PrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36Fp8Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen36Fp8PrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen38FlashNextRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen38FlashNextPrefillRoute<TOKENS> {}
impl private::Sealed for UnadmittedRoute {}

impl<A: Arch, const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = decode_blocks::<A>(TOKENS, "")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .paged_gqa_exact::<A, TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching paged GQA", source))
    }
}

impl<A: Arch, const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = prefill_blocks::<A>(TOKENS, 2, "")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .paged_gqa_prefill_shared_exact::<A, TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching shared paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Bf16Cache> for PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = decode_blocks::<Qwen35_9B>(TOKENS, "Qwen3.5 ")?;

        Ok(Self {
            attention: module
                .prepare_qwen35_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    THREADS,
                    DECODE_RING_SHARED_BYTES as u32,
                ))
                .map_err(|source| GpuError::launch("preparing Qwen3.5 paged GQA", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen35_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 paged GQA", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Bf16Cache> for PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN35_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = prefill_blocks::<Qwen35_9B>(TOKENS, 1, "Qwen3.5 ")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen35_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Bf16Cache> for PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = decode_blocks::<Qwen36Moe35B>(TOKENS, "Qwen3.6 ")?;

        Ok(Self {
            attention: module
                .prepare_qwen36_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    THREADS,
                    DECODE_RING_SHARED_BYTES as u32,
                ))
                .map_err(|source| GpuError::launch("preparing Qwen3.6 paged GQA", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 paged GQA", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Bf16Cache> for PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = prefill_blocks::<Qwen36Moe35B>(TOKENS, 1, "Qwen3.6 ")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedQwen36Fp8Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = decode_blocks::<Qwen36Moe35B>(TOKENS, "Qwen3.6 FP8 ")?;

        Ok(Self {
            attention: module
                .prepare_qwen36_fp8_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    THREADS,
                    DECODE_RING_E4M3_SHARED_BYTES as u32,
                ))
                .map_err(|source| GpuError::launch("preparing Qwen3.6 FP8 paged GQA", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 paged GQA", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedQwen36Fp8PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 FP8 paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = prefill_blocks::<Qwen36Moe35B>(TOKENS, 1, "Qwen3.6 FP8 ")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 paged GQA prefill", source))
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedQwen38FlashNextRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = decode_blocks::<Qwen38FlashNext>(TOKENS, "Qwen3.8-Flash-Next QSA ")?;

        Ok(Self {
            attention: module
                .prepare_qwen38_flash_next_paged_gqa_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    DECODE_THREADS_U32,
                    0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next QSA paged GQA", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_paged_gqa_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next QSA paged GQA", source)
            })
    }
}

impl<const TOKENS: usize> PagedGqaRoute<Fp8Cache> for PreparedQwen38FlashNextPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN38_FLASH_NEXT_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.8-Flash-Next QSA paged GQA prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = prefill_blocks::<Qwen38FlashNext>(TOKENS, 1, "Qwen3.8-Flash-Next QSA ")?;

        Ok(Self {
            attention: module
                .prepare_qwen38_flash_next_paged_gqa_prefill_shared_exact::<TOKENS>(
                    LaunchConfig1D::new(
                        blocks,
                        QWEN38_FLASH_NEXT_PREFILL_THREADS as u32,
                        PREFILL_SHARED_BYTES_U32,
                    ),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next QSA paged GQA prefill", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &PagedGqaArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_paged_gqa_prefill_shared_exact::<TOKENS>(
                stream,
                &self.attention,
                args.query,
                args.key_pages,
                args.value_pages,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.lengths,
                args.output,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next QSA paged GQA prefill", source)
            })
    }
}

// `width_route` rejects an unadmitted width before dispatch, so this is the
// defensive tail of a route that owns no entry.
impl<C: CacheFormat> PagedGqaRoute<C> for UnadmittedRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    unsafe fn launch(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _args: &PagedGqaArgs<C>,
    ) -> GpuResult<()> {
        Err(GpuError::invalid_launch(
            "paged GQA prefill route is not admitted for this architecture",
        ))
    }
}

/// The compiled entry one admitted token width selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WidthRoute {
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    T32,
    T64,
    T128,
    T1024,
}

impl WidthRoute {
    const fn rows(self) -> usize {
        match self {
            Self::B1 => 1,
            Self::B2 => 2,
            Self::B3 => 3,
            Self::B4 => 4,
            Self::B5 => 5,
            Self::B6 => 6,
            Self::B7 => 7,
            Self::B8 => 8,
            Self::T32 => 32,
            Self::T64 => 64,
            Self::T128 => 128,
            Self::T1024 => 1_024,
        }
    }
}

// Transcribed from the four prepared dispatches this module replaces: every
// admitted table decodes `B=1..=8` and prefills `T=32,64,128`. Qwen3.8 splits
// the two halves across two public launchers; the other three admit their
// union through one. Qwen3.8-Flash-Next adds `T=1024` on the same shared-tile
// schedule, which the other four tables leave unadmitted so their emitted
// inventory is unchanged.
fn decode_route(batch: usize) -> Option<WidthRoute> {
    match batch {
        1 => Some(WidthRoute::B1),
        2 => Some(WidthRoute::B2),
        3 => Some(WidthRoute::B3),
        4 => Some(WidthRoute::B4),
        5 => Some(WidthRoute::B5),
        6 => Some(WidthRoute::B6),
        7 => Some(WidthRoute::B7),
        8 => Some(WidthRoute::B8),
        _ => None,
    }
}

fn prefill_route<A: Arch, E: PagedGqaEntries<A>>(tokens: usize) -> Option<WidthRoute> {
    match tokens {
        32 => Some(WidthRoute::T32),
        64 => Some(WidthRoute::T64),
        128 => Some(WidthRoute::T128),
        1_024 if E::HAS_T1024 => Some(WidthRoute::T1024),
        _ => None,
    }
}

fn width_route<A: Arch, E: PagedGqaEntries<A>>(tokens: usize) -> Option<WidthRoute> {
    decode_route(tokens).or_else(|| prefill_route::<A, E>(tokens))
}

fn admitted_prefill_widths<A: Arch, E: PagedGqaEntries<A>>() -> &'static str {
    if E::HAS_T1024 {
        "32,64,128,1024"
    } else {
        "32,64,128"
    }
}

fn checked_table_stride(table_stride: usize, label: &str) -> GpuResult<u32> {
    let table_stride = u32::try_from(table_stride).map_err(|_| {
        GpuError::invalid_launch(format!("{label}paged GQA table stride exceeds u32"))
    })?;
    if table_stride == 0 {
        return Err(GpuError::invalid_launch(format!(
            "{label}paged GQA table stride must be nonzero"
        )));
    }

    Ok(table_stride)
}

fn checked_cache_scales(key_scale: f32, value_scale: f32, label: &str) -> GpuResult<CacheScales> {
    if !key_scale.is_finite() || key_scale <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{label}paged GQA key scale must be finite and positive"
        )));
    }
    if !value_scale.is_finite() || value_scale <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{label}paged GQA value scale must be finite and positive"
        )));
    }

    Ok(CacheScales {
        key: key_scale,
        value: value_scale,
    })
}

/// Exact entry table of one admitted architecture and cache format.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 and Qwen3.6 here never widens the
/// artifact-level admission bound. Each table names only the entries its own
/// model emits, which is what keeps the compiled inventory fixed while the
/// four owners share one width dispatch.
pub trait PagedGqaEntries<A: Arch>: private::Sealed {
    /// Represented storage format this table's cache planes hold.
    type Cache: CacheFormat;
    /// Prepared decode entry for `B=1..=8`.
    type Decode<const TOKENS: usize>: PagedGqaRoute<Self::Cache>;
    /// Prepared shared-cache prefill entry for `T=32,64,128`.
    type Prefill<const TOKENS: usize>: PagedGqaRoute<Self::Cache>;
    /// Prepared `T=1024` prefill entry, unadmitted outside Qwen3.8-Flash-Next QSA.
    type Prefill1024: PagedGqaRoute<Self::Cache>;

    /// Whether `T=1024` is an admitted shared-tile prefill width.
    const HAS_T1024: bool;
    /// Message prefix that keeps this table's width rejections distinct.
    const LABEL: &'static str;
    /// Message prefix this table's stride and cache-scale rejections carry.
    ///
    /// Qwen3.8 and the Qwen3.6 E4M3 owner both reject through the unprefixed
    /// wording their launchers already published, so this is not always
    /// [`Self::LABEL`].
    const VALIDATION_LABEL: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not cover.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Qwen3.8 entry table: E4M3 cache, decode `B=1..=8`, prefill `T=32,64,128`.
pub struct Qwen38PagedGqaEntries;

/// Qwen3.5 entry table: BF16 cache, decode `B=1..=8`, prefill `T=32,64,128`.
pub struct Qwen35PagedGqaEntries;

/// Qwen3.6 entry table: BF16 cache, decode `B=1..=8`, prefill `T=32,64,128`.
pub struct Qwen36PagedGqaEntries;

/// Qwen3.6 entry table: E4M3 cache, decode `B=1..=8`, prefill `T=32,64,128`.
pub struct Qwen36Fp8PagedGqaEntries;

/// Qwen3.8-Flash-Next QSA table: E4M3 cache, decode `B=1..=8`, prefill to `T=1024`.
pub struct Qwen38FlashNextPagedGqaEntries;

impl private::Sealed for Qwen38PagedGqaEntries {}
impl private::Sealed for Qwen35PagedGqaEntries {}
impl private::Sealed for Qwen36PagedGqaEntries {}
impl private::Sealed for Qwen36Fp8PagedGqaEntries {}
impl private::Sealed for Qwen38FlashNextPagedGqaEntries {}

// The Qwen3.8 entries stay bound to the sealed artifact-level architecture:
// they are the only routes whose kernels are instantiated over `A`.
impl<A: Sm120Arch> PagedGqaEntries<A> for Qwen38PagedGqaEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedRoute<A, TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<A, TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;

    const LABEL: &'static str = "";
    const VALIDATION_LABEL: &'static str = "";
    const MODULE_OPERATION: &'static str = "loading paged GQA";

    fn require_geometry() -> GpuResult<()> {
        require_qwen38_geometry::<A>()
    }

    fn ptx_names() -> Vec<&'static str> {
        paged_gqa_ptx_names()
    }
}

impl PagedGqaEntries<Qwen35_9B> for Qwen35PagedGqaEntries {
    type Cache = Bf16Cache;
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen35PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;

    const LABEL: &'static str = "Qwen3.5 ";
    const VALIDATION_LABEL: &'static str = "Qwen3.5 ";
    const MODULE_OPERATION: &'static str = "loading Qwen3.5 paged GQA";

    fn require_geometry() -> GpuResult<()> {
        require_qwen35_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_paged_gqa_ptx_names()
    }
}

impl PagedGqaEntries<Qwen36Moe35B> for Qwen36PagedGqaEntries {
    type Cache = Bf16Cache;
    type Decode<const TOKENS: usize> = PreparedQwen36Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen36PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;

    const LABEL: &'static str = "Qwen3.6 ";
    const VALIDATION_LABEL: &'static str = "Qwen3.6 ";
    const MODULE_OPERATION: &'static str = "loading Qwen3.6 paged GQA";

    fn require_geometry() -> GpuResult<()> {
        require_qwen36_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_paged_gqa_ptx_names()
    }
}

impl PagedGqaEntries<Qwen36Moe35B> for Qwen36Fp8PagedGqaEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedQwen36Fp8Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen36Fp8PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;

    const LABEL: &'static str = "Qwen3.6 FP8 ";
    const VALIDATION_LABEL: &'static str = "";
    const MODULE_OPERATION: &'static str = "loading Qwen3.6 FP8 paged GQA";

    fn require_geometry() -> GpuResult<()> {
        require_qwen36_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_fp8_paged_gqa_ptx_names()
    }
}

impl PagedGqaEntries<Qwen38FlashNext> for Qwen38FlashNextPagedGqaEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedQwen38FlashNextRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen38FlashNextPrefillRoute<TOKENS>;
    type Prefill1024 = PreparedQwen38FlashNextPrefillRoute<1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "Qwen3.8-Flash-Next QSA ";
    const VALIDATION_LABEL: &'static str = "Qwen3.8-Flash-Next QSA ";
    const MODULE_OPERATION: &'static str = "loading Qwen3.8-Flash-Next QSA paged GQA";

    fn require_geometry() -> GpuResult<()> {
        require_qwen38_flash_next_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_paged_gqa_ptx_names()
    }
}

/// PTX symbols retained for every exact Qwen3.8-Flash-Next QSA paged GQA route.
pub(crate) fn qwen38_flash_next_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_paged_gqa_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_shared_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_shared_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_shared_exact_ptx_name::<128>(),
        kernels::qwen38_flash_next_paged_gqa_prefill_shared_exact_ptx_name::<1_024>(),
    ]
}

/// One entry table's prepared decode and shared-prefill entries.
///
/// Every table prepares the shared eleven widths; an entry table can also
/// admit its own `T=1024` route. Qwen3.8's partitioned prefill tails have no
/// counterpart and stay on [`PagedGqaOp`].
#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_paged_gqa_exact_width),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct ExactWidthRoutes<A: Arch, E: PagedGqaEntries<A>> {
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

impl<A: Arch, E: PagedGqaEntries<A>> ExactWidthRoutes<A, E> {
    /// Dispatches one prepared width.
    ///
    /// # Safety
    ///
    /// `args` carries the owner's pointer contract unchanged.
    unsafe fn dispatch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        route: WidthRoute,
        args: &PagedGqaArgs<E::Cache>,
    ) -> GpuResult<()> {
        dispatch_paged_gqa_exact_width!(
            self,
            route.rows(),
            |entry| unsafe { entry.launch(module, stream, args) },
            else => unreachable!("WidthRoute always names one exact route")
        )
    }
}

struct PreparedPartitionedPrefillP8<A: Arch, const TOKENS: usize> {
    partial: PreparedLaunch<kernels::__paged_gqa_prefill_flash_p8_exact_CudaKernel<A, TOKENS>>,
    reduce: PreparedLaunch<
        kernels::__paged_gqa_prefill_partitioned_reduce_exact_CudaKernel<A, TOKENS, 8>,
    >,
}

impl<A: Arch, const TOKENS: usize> PreparedPartitionedPrefillP8<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !matches!(TOKENS, 32 | 64 | 128) {
            return Err(GpuError::invalid_launch(format!(
                "P8 flash paged GQA tokens {TOKENS} are outside the admitted set 32, 64, 128"
            )));
        }
        let partial_blocks = u32::try_from(TOKENS / 32 * A::NUM_ATTENTION_HEADS * 8)
            .map_err(|_| GpuError::invalid_launch("P8 flash paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_flash_p8_exact::<A, TOKENS>(LaunchConfig1D::new(
                    partial_blocks,
                    FLASH_PREFILL_THREADS as u32,
                    FLASH_PREFILL_P8_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing P8 flash paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_partitioned_reduce_exact::<A, TOKENS, 8>(
                    LaunchConfig1D::new(reduce_blocks, THREADS, 0),
                )
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
            .paged_gqa_prefill_flash_p8_exact::<A, TOKENS>(
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
            .paged_gqa_prefill_partitioned_reduce_exact::<A, TOKENS, 8>(
                stream,
                &self.reduce,
                partials,
                output,
            )
            .map_err(|source| GpuError::launch("launching P8 paged GQA reduction", source))
    }
}

struct PreparedPartitionedPrefillP16<A: Arch, const TOKENS: usize> {
    partial: PreparedLaunch<kernels::__paged_gqa_prefill_flash_p16_exact_CudaKernel<A, TOKENS>>,
    reduce: PreparedLaunch<
        kernels::__paged_gqa_prefill_partitioned_reduce_exact_CudaKernel<A, TOKENS, 16>,
    >,
}

impl<A: Arch, const TOKENS: usize> PreparedPartitionedPrefillP16<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !matches!(TOKENS, 32 | 64 | 128) {
            return Err(GpuError::invalid_launch(format!(
                "P16 flash paged GQA tokens {TOKENS} are outside the admitted set 32, 64, 128"
            )));
        }
        let partial_blocks = u32::try_from(TOKENS / 32 * A::NUM_ATTENTION_HEADS * 16)
            .map_err(|_| GpuError::invalid_launch("P16 flash paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_flash_p16_exact::<A, TOKENS>(LaunchConfig1D::new(
                    partial_blocks,
                    FLASH_PREFILL_THREADS as u32,
                    FLASH_PREFILL_P16_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing P16 flash paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_partitioned_reduce_exact::<A, TOKENS, 16>(
                    LaunchConfig1D::new(reduce_blocks, THREADS, 0),
                )
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
            .paged_gqa_prefill_flash_p16_exact::<A, TOKENS>(
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
            .paged_gqa_prefill_partitioned_reduce_exact::<A, TOKENS, 16>(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PartitionedPrefillRoute {
    T32P8,
    T32P16,
    T64P8,
    T64P16,
    T128P8,
    T128P16,
}

fn partitioned_prefill_route(
    tokens: usize,
    partitions: usize,
) -> GpuResult<PartitionedPrefillRoute> {
    match (tokens, partitions) {
        (32, 8) => Ok(PartitionedPrefillRoute::T32P8),
        (32, 16) => Ok(PartitionedPrefillRoute::T32P16),
        (64, 8) => Ok(PartitionedPrefillRoute::T64P8),
        (64, 16) => Ok(PartitionedPrefillRoute::T64P16),
        (128, 8) => Ok(PartitionedPrefillRoute::T128P8),
        (128, 16) => Ok(PartitionedPrefillRoute::T128P16),
        _ => Err(GpuError::invalid_launch(format!(
            "partitioned paged GQA prefill T={tokens}/P={partitions} is outside the admitted T=32/64/128, P=8/16 matrix"
        ))),
    }
}

/// Prepared paged GQA routes for exact `B=1..8` decode and early prefill tails.
///
/// Qwen3.8 keeps its own owner: beyond the eleven widths every table shares it
/// admits P8/P16 partitioned `T=32/64/128` tails and the five `T=1024` macro
/// routes, whose two-stage partial/reduce launches take a partials workspace
/// the shared width dispatch does not carry.
pub struct PagedGqaOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    widths: ExactWidthRoutes<A, Qwen38PagedGqaEntries>,
    t32_p8: PreparedPartitionedPrefillP8<A, 32>,
    t32_p16: PreparedPartitionedPrefillP16<A, 32>,
    t64_p8: PreparedPartitionedPrefillP8<A, 64>,
    t64_p16: PreparedPartitionedPrefillP16<A, 64>,
    t128_p8: PreparedPartitionedPrefillP8<A, 128>,
    t128_p16: PreparedPartitionedPrefillP16<A, 128>,
    macro_p1: PreparedMacroPrefill<A, 1>,
    macro_p2: PreparedMacroPrefill<A, 2>,
    macro_p4: PreparedMacroPrefill<A, 4>,
    macro_p8: PreparedMacroPrefill<A, 8>,
    macro_p16: PreparedMacroPrefill<A, 16>,
}

impl<A: Sm120Arch> PagedGqaOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        <Qwen38PagedGqaEntries as PagedGqaEntries<A>>::require_geometry()?;
        let _ = <Qwen38PagedGqaEntries as PagedGqaEntries<A>>::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module(
                <Qwen38PagedGqaEntries as PagedGqaEntries<A>>::MODULE_OPERATION,
                source,
            )
        })?;

        Ok(Self {
            widths: ExactWidthRoutes::prepare(&module)?,
            t32_p8: PreparedPartitionedPrefillP8::prepare(&module)?,
            t32_p16: PreparedPartitionedPrefillP16::prepare(&module)?,
            t64_p8: PreparedPartitionedPrefillP8::prepare(&module)?,
            t64_p16: PreparedPartitionedPrefillP16::prepare(&module)?,
            t128_p8: PreparedPartitionedPrefillP8::prepare(&module)?,
            t128_p16: PreparedPartitionedPrefillP16::prepare(&module)?,
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
        let route = decode_route(batch).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "paged GQA batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            ))
        })?;
        let args = self.exact_width_args(
            table_stride,
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            output,
            key_scale,
            value_scale,
        )?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe { self.widths.dispatch(&self.module, stream, route, &args) }
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
        // The replaced launcher validated the stride and scales before it
        // selected a width, so an unadmitted width keeps reporting second.
        let args = self.exact_width_args(
            table_stride,
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            lengths,
            output,
            key_scale,
            value_scale,
        )?;
        let route = prefill_route::<A, Qwen38PagedGqaEntries>(tokens).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "paged GQA shared prefill tokens {tokens} are outside the admitted set 32, 64, 128"
            ))
        })?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe { self.widths.dispatch(&self.module, stream, route, &args) }
    }

    /// Validates the operands both width launchers share and bundles them.
    #[allow(clippy::too_many_arguments)]
    fn exact_width_args(
        &self,
        table_stride: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        lengths: *const u32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<PagedGqaArgs<Fp8Cache>> {
        let label = <Qwen38PagedGqaEntries as PagedGqaEntries<A>>::VALIDATION_LABEL;

        Ok(PagedGqaArgs {
            query,
            key_pages,
            value_pages,
            block_tables,
            table_rows,
            table_stride: checked_table_stride(table_stride, label)?,
            lengths,
            output,
            scales: checked_cache_scales(key_scale, value_scale, label)?,
        })
    }

    /// Applies exact partitioned paged GQA to a deep `T=32/64/128` prefill tail.
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
        tokens: usize,
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

        match partitioned_prefill_route(tokens, partitions)? {
            PartitionedPrefillRoute::T32P8 => launch!(t32_p8),
            PartitionedPrefillRoute::T32P16 => launch!(t32_p16),
            PartitionedPrefillRoute::T64P8 => launch!(t64_p8),
            PartitionedPrefillRoute::T64P16 => launch!(t64_p16),
            PartitionedPrefillRoute::T128P8 => launch!(t128_p8),
            PartitionedPrefillRoute::T128P16 => launch!(t128_p16),
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
    let table_stride = checked_table_stride(table_stride, "")?;
    let _ = checked_cache_scales(key_scale, value_scale, "")?;

    Ok(table_stride)
}

/// Prepared paged GQA routes whose single launcher admits every exact width.
///
/// Qwen3.5, Qwen3.6, and the Qwen3.6 E4M3 cache route `B=1..8` decode and
/// `T=32,64,128` prefill through one launcher over one entry table. `C`
/// restates `E::Cache` as a parameter so the E4M3 and BF16 launch signatures
/// live in disjoint inherent impls: an E4M3 launch carries the two represented
/// cache scales a BF16 launch does not have.
pub struct ExactWidthPagedGqaOp<A: Arch, C: CacheFormat, E: PagedGqaEntries<A, Cache = C>> {
    module: kernels::LoadedModule,
    widths: ExactWidthRoutes<A, E>,
    cache: PhantomData<C>,
}

/// Prepared Qwen3.5 BF16 paged GQA routes for exact `B=1..8` and `T=32,64,128`.
pub type Qwen35PagedGqaOp = ExactWidthPagedGqaOp<Qwen35_9B, Bf16Cache, Qwen35PagedGqaEntries>;

/// Prepared Qwen3.6 BF16 paged GQA routes for exact `B=1..8` and `T=32,64,128`.
pub type Qwen36PagedGqaOp = ExactWidthPagedGqaOp<Qwen36Moe35B, Bf16Cache, Qwen36PagedGqaEntries>;

/// Prepared Qwen3.6 E4M3 paged GQA routes for exact decode and prompt widths.
pub type Qwen36Fp8PagedGqaOp =
    ExactWidthPagedGqaOp<Qwen36Moe35B, Fp8Cache, Qwen36Fp8PagedGqaEntries>;

/// Prepared Qwen3.8-Flash-Next QSA E4M3 dense causal paged GQA routes.
///
/// The composed route must keep total visible length at or below 2,051, where
/// the QSA selection mask is the identity.
pub type Qwen38FlashNextPagedGqaOp =
    ExactWidthPagedGqaOp<Qwen38FlashNext, Fp8Cache, Qwen38FlashNextPagedGqaEntries>;

impl<A: Arch, C: CacheFormat, E: PagedGqaEntries<A, Cache = C>> ExactWidthPagedGqaOp<A, C, E> {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_OPERATION, source))?;

        Ok(Self {
            widths: ExactWidthRoutes::prepare(&module)?,
            cache: PhantomData,
            module,
        })
    }

    /// Selects the prepared width, rejecting one this table does not admit.
    fn admitted_route(tokens: usize) -> GpuResult<WidthRoute> {
        width_route::<A, E>(tokens).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "{}paged GQA tokens {tokens} must be one of 1..={MAX_BATCH},{}",
                E::LABEL,
                admitted_prefill_widths::<A, E>(),
            ))
        })
    }
}

impl<A: Arch, E: PagedGqaEntries<A, Cache = Bf16Cache>> ExactWidthPagedGqaOp<A, Bf16Cache, E> {
    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens, A::NUM_ATTENTION_HEADS, 256]` FP32
    /// values. Cache planes use `[physical_page, A::NUM_KV_HEADS, 64, 256]`
    /// BF16 values. Metadata covers `tokens`; each length is nonzero, its
    /// table row covers that length rounded up to 64, and every physical page
    /// is resident. Allocations are aligned, disjoint, live through
    /// completion, and belong to `stream`'s context.
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
        let route = Self::admitted_route(tokens)?;
        let table_stride = checked_table_stride(table_stride, E::VALIDATION_LABEL)?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe {
            self.widths.dispatch(
                &self.module,
                stream,
                route,
                &PagedGqaArgs {
                    query,
                    key_pages,
                    value_pages,
                    block_tables,
                    table_rows,
                    table_stride,
                    lengths,
                    output,
                    scales: (),
                },
            )
        }
    }
}

impl<A: Arch, E: PagedGqaEntries<A, Cache = Fp8Cache>> ExactWidthPagedGqaOp<A, Fp8Cache, E> {
    /// Applies online-softmax GQA over page-major represented E4M3 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens, A::NUM_ATTENTION_HEADS, 256]` FP32
    /// values. Cache planes use `[physical_page, A::NUM_KV_HEADS, 64, 256]`
    /// E4M3 bytes. Metadata covers `tokens`; each length is nonzero, its table
    /// row covers that length rounded up to 64, and every selected physical
    /// page is resident. Allocations are aligned, disjoint, live through
    /// completion, and belong to `stream`'s context.
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
        let route = Self::admitted_route(tokens)?;
        let table_stride = checked_table_stride(table_stride, E::VALIDATION_LABEL)?;
        let scales = checked_cache_scales(key_scale, value_scale, E::VALIDATION_LABEL)?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe {
            self.widths.dispatch(
                &self.module,
                stream,
                route,
                &PagedGqaArgs {
                    query,
                    key_pages,
                    value_pages,
                    block_tables,
                    table_rows,
                    table_stride,
                    lengths,
                    output,
                    scales,
                },
            )
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
        kernels::paged_gqa_prefill_flash_p8_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 32, 8>(),
        kernels::paged_gqa_prefill_flash_p16_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 32, 16>(),
        kernels::paged_gqa_prefill_flash_p8_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 64, 8>(),
        kernels::paged_gqa_prefill_flash_p16_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 64, 16>(),
        kernels::paged_gqa_prefill_flash_p8_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 128, 8>(),
        kernels::paged_gqa_prefill_flash_p16_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 128, 16>(),
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
        BF16_PREFILL_SHARED_BYTES, BF16_PREFILL_THREADS, Bf16Cache, CacheFormat,
        DECODE_RING_E4M3_SHARED_BYTES, DECODE_RING_SHARED_BYTES, DECODE_THREADS,
        FLASH_PREFILL_P8_SHARED_BYTES, FLASH_PREFILL_P16_SHARED_BYTES, FLASH_PREFILL_THREADS,
        Fp8Cache, PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT,
        PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, PAGED_GQA_PREFILL_MACRO_TOKENS,
        PAGED_GQA_PREFILL_PARTIAL_BYTES, PREFILL_SHARED_BYTES, PREFILL_THREADS, PagedGqaEntries,
        QWEN35_BF16_PREFILL_THREADS, QWEN35_PREFILL_TOKENS, QWEN36_FP8_PREFILL_THREADS,
        QWEN36_PREFILL_TOKENS, Qwen35PagedGqaEntries, Qwen36Fp8PagedGqaEntries,
        Qwen36PagedGqaEntries, Qwen38PagedGqaEntries, THREADS, WidthRoute,
        admitted_macro_partitions, checked_cache_scales, checked_table_stride, decode_blocks,
        decode_route, paged_gqa_prefill_partitions, paged_gqa_ptx_names, partitioned_prefill_route,
        prefill_blocks, prefill_route, qwen35_paged_gqa_ptx_names, qwen36_fp8_paged_gqa_ptx_names,
        qwen36_paged_gqa_ptx_names, width_route,
    };
    use super::{
        QWEN38_FLASH_NEXT_PREFILL_THREADS, QWEN38_FLASH_NEXT_PREFILL_TOKENS,
        QWEN38_FLASH_NEXT_QUERY_WARPS, Qwen38FlashNextPagedGqaEntries, admitted_prefill_widths,
        qwen38_flash_next_paged_gqa_ptx_names,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

    /// The eight decode widths every admitted entry table routes.
    const DECODE_SCHEDULE: [(usize, WidthRoute); 8] = [
        (1, WidthRoute::B1),
        (2, WidthRoute::B2),
        (3, WidthRoute::B3),
        (4, WidthRoute::B4),
        (5, WidthRoute::B5),
        (6, WidthRoute::B6),
        (7, WidthRoute::B7),
        (8, WidthRoute::B8),
    ];

    /// The three shared-cache prefill widths every admitted entry table routes.
    const PREFILL_SCHEDULE: [(usize, WidthRoute); 3] = [
        (32, WidthRoute::T32),
        (64, WidthRoute::T64),
        (128, WidthRoute::T128),
    ];

    /// The four prefill widths only the Qwen3.8-Flash-Next QSA table routes.
    const QWEN38_FLASH_NEXT_PREFILL_SCHEDULE: [(usize, WidthRoute); 4] = [
        (32, WidthRoute::T32),
        (64, WidthRoute::T64),
        (128, WidthRoute::T128),
        (1_024, WidthRoute::T1024),
    ];

    /// Sweeps every width exhaustively so an unadmitted one cannot hide
    /// between the transcribed ones.
    fn admitted_schedule(route: fn(usize) -> Option<WidthRoute>) -> Vec<(usize, WidthRoute)> {
        (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|tokens| route(tokens).map(|selected| (tokens, selected)))
            .collect()
    }

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    fn cache_element_bytes<A: Arch, E: PagedGqaEntries<A>>() -> usize {
        size_of::<<E::Cache as CacheFormat>::Element>()
    }

    /// The merged width schedule, checked against the four dispatches it
    /// replaces: Qwen3.8 keeps decode and shared prefill on separate public
    /// launchers, and the other three admit their union through one.
    #[test]
    fn width_table_covers_only_exact_decode_and_prefill_widths() {
        assert_eq!(admitted_schedule(decode_route), DECODE_SCHEDULE.to_vec());
        assert_eq!(
            admitted_schedule(prefill_route::<Qwen38_27B, Qwen38PagedGqaEntries>),
            PREFILL_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule(width_route::<Qwen38_27B, Qwen38PagedGqaEntries>),
            DECODE_SCHEDULE
                .iter()
                .copied()
                .chain(PREFILL_SCHEDULE.iter().copied())
                .collect::<Vec<_>>()
        );

        // `T=1024` reaches the shared-tile schedule only through the table
        // that admits it, so the other four tables' emitted inventory is
        // unchanged by its addition.
        for (label, admitted) in [
            (
                "Qwen3.5",
                admitted_schedule(width_route::<Qwen35_9B, Qwen35PagedGqaEntries>),
            ),
            (
                "Qwen3.6",
                admitted_schedule(width_route::<Qwen36Moe35B, Qwen36PagedGqaEntries>),
            ),
            (
                "Qwen3.6 FP8",
                admitted_schedule(width_route::<Qwen36Moe35B, Qwen36Fp8PagedGqaEntries>),
            ),
        ] {
            assert!(
                !admitted.iter().any(|(tokens, _)| *tokens == 1_024),
                "{label} must not admit T=1024"
            );
        }
        assert_eq!(
            admitted_schedule(prefill_route::<Qwen38FlashNext, Qwen38FlashNextPagedGqaEntries>),
            QWEN38_FLASH_NEXT_PREFILL_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_prefill_widths::<Qwen38FlashNext, Qwen38FlashNextPagedGqaEntries>(),
            "32,64,128,1024"
        );
        assert_eq!(
            admitted_prefill_widths::<Qwen38_27B, Qwen38PagedGqaEntries>(),
            "32,64,128"
        );

        // One shared K/V tile feeds the CTA's 12 query-head warps.
        assert_eq!(
            QWEN38_FLASH_NEXT_QUERY_WARPS,
            Qwen38FlashNext::NUM_ATTENTION_HEADS / Qwen38FlashNext::NUM_KV_HEADS
        );
        assert_eq!(QWEN38_FLASH_NEXT_PREFILL_THREADS, 384);
        assert_eq!(QWEN38_FLASH_NEXT_PREFILL_TOKENS, [32, 64, 128, 1_024]);

        assert_eq!(THREADS, 32);
        assert_eq!(DECODE_THREADS, 256);
        assert_eq!(QWEN35_PREFILL_TOKENS, [32, 64, 128]);
        assert_eq!(QWEN36_PREFILL_TOKENS, [32, 64, 128]);
    }

    /// One CTA covers one token and query head at decode, and one token group
    /// and KV head at prefill. The merged grids must reproduce the CTA counts
    /// each replaced owner launched, including Qwen3.8's two-token prefill CTA
    /// and the 4:1 versus 2:1 KV-head split of the two BF16 targets.
    #[test]
    fn width_grids_match_each_admitted_route() {
        for (tokens, qwen38, qwen35, qwen36) in [(1, 24, 16, 16), (8, 192, 128, 128)] {
            assert_eq!(decode_blocks::<Qwen38_27B>(tokens, "").unwrap(), qwen38);
            assert_eq!(decode_blocks::<Qwen35_9B>(tokens, "").unwrap(), qwen35);
            assert_eq!(decode_blocks::<Qwen36Moe35B>(tokens, "").unwrap(), qwen36);
        }
        for (tokens, qwen38, qwen35, qwen36) in
            [(32, 64, 128, 64), (64, 128, 256, 128), (128, 256, 512, 256)]
        {
            assert_eq!(prefill_blocks::<Qwen38_27B>(tokens, 2, "").unwrap(), qwen38);
            assert_eq!(prefill_blocks::<Qwen35_9B>(tokens, 1, "").unwrap(), qwen35);
            assert_eq!(
                prefill_blocks::<Qwen36Moe35B>(tokens, 1, "").unwrap(),
                qwen36
            );
        }
    }

    /// Every route's block and shared-memory footprint is blessed measured
    /// behaviour, so the merged owners must keep the four schedules distinct.
    #[test]
    fn every_route_keeps_its_measured_launch_shape() {
        assert_eq!(PREFILL_THREADS, 384);
        assert_eq!(PREFILL_SHARED_BYTES, 32_768);
        assert_eq!(QWEN35_BF16_PREFILL_THREADS, 128);
        assert_eq!(BF16_PREFILL_THREADS, 256);
        assert_eq!(BF16_PREFILL_SHARED_BYTES, 65_536);
        assert_eq!(QWEN36_FP8_PREFILL_THREADS, 256);
        assert_eq!(DECODE_RING_SHARED_BYTES, 8_192);
        assert_eq!(DECODE_RING_E4M3_SHARED_BYTES, 4_096);
        assert_eq!(FLASH_PREFILL_THREADS, 256);
        assert_eq!(FLASH_PREFILL_P8_SHARED_BYTES, 78_336);
        assert_eq!(FLASH_PREFILL_P16_SHARED_BYTES, 43_520);
        assert_eq!(PAGED_GQA_PREFILL_PARTIAL_BYTES, 50_724_864);
        assert_eq!(PAGED_GQA_PREFILL_MACRO_TOKENS, 1_024);
        assert_eq!(PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES, 405_798_912);
    }

    /// Each entry table keeps its qualified cache format, so merging the
    /// owners cannot read E4M3 codes out of a BF16 plane or the reverse.
    #[test]
    fn every_entry_table_keeps_its_qualified_cache_format() {
        assert_eq!(
            cache_element_bytes::<Qwen38_27B, Qwen38PagedGqaEntries>(),
            size_of::<<Fp8Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen36Moe35B, Qwen36Fp8PagedGqaEntries>(),
            size_of::<<Fp8Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen35_9B, Qwen35PagedGqaEntries>(),
            size_of::<<Bf16Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen36Moe35B, Qwen36PagedGqaEntries>(),
            size_of::<<Bf16Cache as CacheFormat>::Element>()
        );
    }

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38PagedGqaEntries as PagedGqaEntries<Qwen38_27B>>::ptx_names(),
            paged_gqa_ptx_names()
        );
        assert_eq!(
            <Qwen35PagedGqaEntries as PagedGqaEntries<Qwen35_9B>>::ptx_names(),
            qwen35_paged_gqa_ptx_names()
        );
        assert_eq!(
            <Qwen36PagedGqaEntries as PagedGqaEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_paged_gqa_ptx_names()
        );
        assert_eq!(
            <Qwen36Fp8PagedGqaEntries as PagedGqaEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_fp8_paged_gqa_ptx_names()
        );
        assert_eq!(
            <Qwen38FlashNextPagedGqaEntries as PagedGqaEntries<Qwen38FlashNext>>::ptx_names(),
            qwen38_flash_next_paged_gqa_ptx_names()
        );
    }

    #[test]
    fn ptx_inventory_has_every_decode_and_prefill_route() {
        let names = paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 29);
        assert_eq!(unique.len(), names.len());

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
        assert_eq!(qwen36_unique.len(), qwen36.len());
        assert!(names.iter().all(|name| !qwen36_unique.contains(name)));
        assert!(qwen35_unique.is_disjoint(&qwen36_unique));

        let qwen38_flash_next = qwen38_flash_next_paged_gqa_ptx_names();
        let qwen38_flash_next_unique = qwen38_flash_next.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen38_flash_next.len(), 12);
        assert_eq!(qwen38_flash_next_unique.len(), qwen38_flash_next.len());
        assert!(
            names
                .iter()
                .all(|name| !qwen38_flash_next_unique.contains(name))
        );
        assert!(qwen38_flash_next_unique.is_disjoint(&qwen35_unique));
        assert!(qwen38_flash_next_unique.is_disjoint(&qwen36_unique));
        assert!(qwen38_flash_next_unique.is_disjoint(&qwen36_fp8_unique));
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this file is
    /// its per-base-name count. These are the counts the pinned SM120 device
    /// build emits; a wrapper change that instantiates one more specialization
    /// moves one of them.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in paged_gqa_ptx_names()
            .into_iter()
            .chain(qwen35_paged_gqa_ptx_names())
            .chain(qwen36_paged_gqa_ptx_names())
            .chain(qwen36_fp8_paged_gqa_ptx_names())
            .chain(qwen38_flash_next_paged_gqa_ptx_names())
        {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("paged_gqa_exact", 8),
                ("paged_gqa_prefill_flash_macro_exact", 1),
                ("paged_gqa_prefill_flash_p16_exact", 3),
                ("paged_gqa_prefill_flash_p8_exact", 3),
                ("paged_gqa_prefill_macro_reduce_exact", 5),
                ("paged_gqa_prefill_partitioned_reduce_exact", 6),
                ("paged_gqa_prefill_shared_exact", 3),
                ("qwen35_paged_gqa_exact", 8),
                ("qwen35_paged_gqa_prefill_shared_exact", 3),
                ("qwen36_fp8_paged_gqa_exact", 8),
                ("qwen36_fp8_paged_gqa_prefill_shared_exact", 3),
                ("qwen36_paged_gqa_exact", 8),
                ("qwen36_paged_gqa_prefill_shared_exact", 3),
                ("qwen38_flash_next_paged_gqa_exact", 8),
                ("qwen38_flash_next_paged_gqa_prefill_shared_exact", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 74);
    }

    /// Each owner keeps the rejection wording its launcher published, which is
    /// not uniform: the Qwen3.6 E4M3 owner names itself for an unadmitted
    /// width but rejects strides and scales unprefixed, as it always did.
    #[test]
    fn unadmitted_launches_keep_their_owner_wording() {
        for (message, error) in [
            (
                "paged GQA tokens 9 must be one of 1..=8,32,64,128",
                width_rejection::<Qwen38_27B, Qwen38PagedGqaEntries>(9),
            ),
            (
                "Qwen3.5 paged GQA tokens 9 must be one of 1..=8,32,64,128",
                width_rejection::<Qwen35_9B, Qwen35PagedGqaEntries>(9),
            ),
            (
                "Qwen3.6 paged GQA tokens 256 must be one of 1..=8,32,64,128",
                width_rejection::<Qwen36Moe35B, Qwen36PagedGqaEntries>(256),
            ),
            (
                "Qwen3.6 FP8 paged GQA tokens 256 must be one of 1..=8,32,64,128",
                width_rejection::<Qwen36Moe35B, Qwen36Fp8PagedGqaEntries>(256),
            ),
            (
                "paged GQA table stride must be nonzero",
                checked_table_stride(
                    0,
                    <Qwen38PagedGqaEntries as PagedGqaEntries<Qwen38_27B>>::VALIDATION_LABEL,
                )
                .unwrap_err(),
            ),
            (
                "Qwen3.5 paged GQA table stride must be nonzero",
                checked_table_stride(
                    0,
                    <Qwen35PagedGqaEntries as PagedGqaEntries<Qwen35_9B>>::VALIDATION_LABEL,
                )
                .unwrap_err(),
            ),
            (
                "Qwen3.6 paged GQA table stride must be nonzero",
                checked_table_stride(
                    0,
                    <Qwen36PagedGqaEntries as PagedGqaEntries<Qwen36Moe35B>>::VALIDATION_LABEL,
                )
                .unwrap_err(),
            ),
            (
                "paged GQA table stride must be nonzero",
                checked_table_stride(
                    0,
                    <Qwen36Fp8PagedGqaEntries as PagedGqaEntries<Qwen36Moe35B>>::VALIDATION_LABEL,
                )
                .unwrap_err(),
            ),
            (
                "paged GQA key scale must be finite and positive",
                checked_cache_scales(
                    0.0,
                    1.0,
                    <Qwen36Fp8PagedGqaEntries as PagedGqaEntries<Qwen36Moe35B>>::VALIDATION_LABEL,
                )
                .err()
                .unwrap(),
            ),
            (
                "paged GQA value scale must be finite and positive",
                checked_cache_scales(
                    1.0,
                    f32::NAN,
                    <Qwen38PagedGqaEntries as PagedGqaEntries<Qwen38_27B>>::VALIDATION_LABEL,
                )
                .err()
                .unwrap(),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "unexpected rejection: {error}"
            );
        }
    }

    fn width_rejection<A: Arch, E: PagedGqaEntries<A>>(tokens: usize) -> tuisko_gpu::GpuError {
        assert!(width_route::<A, E>(tokens).is_none());

        tuisko_gpu::GpuError::invalid_launch(format!(
            "{}paged GQA tokens {tokens} must be one of 1..=8,{}",
            E::LABEL,
            admitted_prefill_widths::<A, E>(),
        ))
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
    fn partitioned_prefill_width_inventory_is_the_exact_six_route_matrix() {
        for tokens in [31, 32, 33, 63, 64, 65, 127, 128, 129] {
            for partitions in [7, 8, 9, 15, 16, 17] {
                assert_eq!(
                    partitioned_prefill_route(tokens, partitions).is_ok(),
                    matches!(tokens, 32 | 64 | 128) && matches!(partitions, 8 | 16),
                    "T={tokens}/P={partitions}"
                );
            }
        }
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

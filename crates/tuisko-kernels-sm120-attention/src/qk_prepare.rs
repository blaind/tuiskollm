//! Exact-batch Q/K normalization, MRoPE, and KV-cache append.

use crate::device::attention_qk_prepare::{attention_qk_prepare, bf16_attention_qk_prepare};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

/// Number of token positions held by one physical KV-cache page.
pub const ATTENTION_PAGE_SIZE: usize = 64;

const MAX_BATCH: usize = 8;
const PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];
const QWEN36_PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const WARPS_PER_CTA: usize = 8;
const THREADS: u32 = (WARPS_PER_CTA * 32) as u32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

fn admitted_tokens(tokens: usize) -> bool {
    admitted_batch(tokens) || PREFILL_TOKENS.contains(&tokens)
}

fn admitted_qwen36_tokens(tokens: usize) -> bool {
    admitted_batch(tokens) || QWEN36_PREFILL_TOKENS.contains(&tokens)
}

fn require_qwen38_geometry<A: Arch>() -> GpuResult<()> {
    if A::NUM_ATTENTION_HEADS != 24
        || A::NUM_KV_HEADS != 4
        || A::HEAD_DIM != 256
        || A::ATTENTION_QUERY_ROWS != 12_288
        || A::ATTENTION_KV_ROWS != 1_024
        || A::ATTENTION_QKV_ROWS != 14_336
        || A::RMS_NORM_EPSILON != 1.0e-6
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the admitted attention Q/K prepare schedule",
        ));
    }

    Ok(())
}

fn require_qwen35_geometry() -> GpuResult<()> {
    if Qwen35_9B::NUM_ATTENTION_HEADS != 16
        || Qwen35_9B::NUM_KV_HEADS != 4
        || Qwen35_9B::HEAD_DIM != 256
        || Qwen35_9B::ATTENTION_QUERY_ROWS != 8_192
        || Qwen35_9B::ATTENTION_KV_ROWS != 1_024
        || Qwen35_9B::ATTENTION_QKV_ROWS != 10_240
        || Qwen35_9B::RMS_NORM_EPSILON != 1.0e-6
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.5 geometry is incompatible with its admitted attention Q/K prepare schedule",
        ));
    }

    Ok(())
}

fn require_qwen36_geometry() -> GpuResult<()> {
    if Qwen36Moe35B::NUM_ATTENTION_HEADS != 16
        || Qwen36Moe35B::NUM_KV_HEADS != 2
        || Qwen36Moe35B::HEAD_DIM != 256
        || Qwen36Moe35B::ATTENTION_QUERY_ROWS != 8_192
        || Qwen36Moe35B::ATTENTION_KV_ROWS != 512
        || Qwen36Moe35B::ATTENTION_QKV_ROWS != 9_216
        || Qwen36Moe35B::RMS_NORM_EPSILON != 1.0e-6
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.6 geometry is incompatible with its admitted attention Q/K prepare schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Prepares Q/K and appends K/V for one exact decode batch.
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
    pub fn attention_qk_prepare_exact<A: Arch, const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) {
        // One warp owns one complete 256-wide head. Eight heads per CTA gives
        // 28 Qwen3.8, 20 Qwen3.5, or 18 Qwen3.6 CTAs at B=8. Keeping the head
        // warp-local preserves the exact reduction and 64-wide MRoPE exchange
        // order; the target performance gate decides whether another topology
        // is owed.
        unsafe {
            attention_qk_prepare::<A, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            );
        }
    }

    /// Prepares Q/K and appends K/V for one exact prefill width.
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
    pub fn attention_qk_prepare_prefill_exact<A: Arch, const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) {
        // Eight complete heads per CTA retains the warp-local 256-value
        // normalization/MRoPE seam. T=32 supplies 112 Qwen3.8 or 72 Qwen3.6
        // CTAs; T=1024 supplies 3,584 Qwen3.8 CTAs, so a wider cooperative tile
        // would only add exchange.
        unsafe {
            attention_qk_prepare::<A, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            );
        }
    }

    /// Prepares Qwen3.5 Q/K and appends K/V for one exact decode batch.
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
    pub fn qwen35_attention_qk_prepare_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) {
        // Qwen3.5 has 20 head-warps per B=1..8 batch: eight warps per
        // CTA make 20 CTAs at B=8. This first qualified route preserves the
        // existing one-warp/head reduction and MRoPE order exactly; its paired
        // performance slice decides whether a different topology is owed.
        unsafe {
            bf16_attention_qk_prepare::<Qwen35_9B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            );
        }
    }

    /// Prepares Qwen3.5 Q/K and appends K/V for one exact prompt width.
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
    pub fn qwen35_attention_qk_prepare_prefill_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) {
        // T=32 and T=1,024 contain 640 and 20,480 independent head-warps,
        // giving 80 and 2,560 CTAs. Each warp retains the decode route's
        // 256-value reduction, MRoPE exchange, and BF16 store order.
        unsafe {
            bf16_attention_qk_prepare::<Qwen35_9B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            );
        }
    }

    /// Prepares Qwen3.6 Q/K and appends K/V for one exact decode batch.
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
    pub fn qwen36_attention_qk_prepare_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) {
        // Qwen3.6 has 18 head-warps per token. Eight warps per CTA produce
        // 18 CTAs at B=8 while preserving the same one-warp/head reduction,
        // 64-wide MRoPE exchange, and BF16 store order as Qwen3.5.
        unsafe {
            bf16_attention_qk_prepare::<Qwen36Moe35B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            );
        }
    }

    /// Prepares Qwen3.6 Q/K and appends K/V for one exact prompt width.
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
    pub fn qwen36_attention_qk_prepare_prefill_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) {
        // T=128 contains 2,304 independent head-warps. One 288-CTA launch
        // replaces 16 B=8 launches without regrouping any head: every warp
        // retains the same 256-value reduction, MRoPE exchange, and BF16 stores.
        unsafe {
            bf16_attention_qk_prepare::<Qwen36Moe35B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            );
        }
    }

    /// Prepares Qwen3.6 Q/K and appends E4M3 K/V for one exact decode batch.
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
    pub fn qwen36_fp8_attention_qk_prepare_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) {
        // Qwen3.6 has 18 head-warps per token. Eight warps per CTA produce
        // 18 CTAs at B=8 while preserving the established one-warp/head
        // reduction, 64-wide MRoPE exchange, and represented E4M3 stores.
        unsafe {
            attention_qk_prepare::<Qwen36Moe35B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            );
        }
    }

    /// Prepares Qwen3.6 Q/K and appends E4M3 K/V for one exact prompt width.
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
    pub fn qwen36_fp8_attention_qk_prepare_prefill_exact<const TOKENS: usize>(
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) {
        // T=32/64/128 supplies 72/144/288 CTAs. Each warp retains the decode
        // route's 256-value reduction, MRoPE exchange, and E4M3 store order.
        unsafe {
            attention_qk_prepare::<Qwen36Moe35B, TOKENS>(
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            );
        }
    }
}

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__attention_qk_prepare_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (A::NUM_ATTENTION_HEADS + A::NUM_KV_HEADS)).div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("attention Q/K prepare grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_attention_qk_prepare_exact::<A, TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| GpuError::launch("preparing attention Q/K route", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .attention_qk_prepare_exact::<A, TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching attention Q/K prepare", source))
    }
}

struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__attention_qk_prepare_prefill_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (A::NUM_ATTENTION_HEADS + A::NUM_KV_HEADS)).div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("attention Q/K prefill grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_attention_qk_prepare_prefill_exact::<A, TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing attention Q/K prefill route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .attention_qk_prepare_prefill_exact::<A, TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching attention Q/K prefill", source))
    }
}

struct PreparedQwen35Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("Qwen3.5 attention Q/K grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_qwen35_attention_qk_prepare_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 attention Q/K route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 attention Q/K prepare", source))
    }
}

struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen35_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 attention Q/K prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = u32::try_from(
            (TOKENS * (Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("Qwen3.5 attention Q/K prefill grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_qwen35_attention_qk_prepare_prefill_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 attention Q/K prefill route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen35_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 attention Q/K prefill", source))
    }
}

struct PreparedQwen36Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen36_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("Qwen3.6 attention Q/K grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_qwen36_attention_qk_prepare_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 attention Q/K route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 attention Q/K prepare", source))
    }
}

struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen36_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 attention Q/K prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = u32::try_from(
            (TOKENS * (Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("Qwen3.6 attention Q/K prefill grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_qwen36_attention_qk_prepare_prefill_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 attention Q/K prefill route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        module
            .qwen36_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 attention Q/K prefill", source))
    }
}

struct PreparedQwen36Fp8Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen36_fp8_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Fp8Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| GpuError::invalid_launch("Qwen3.6 FP8 attention Q/K grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_qwen36_fp8_attention_qk_prepare_exact::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 FP8 attention Q/K route", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 attention Q/K", source))
    }
}

struct PreparedQwen36Fp8PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen36_fp8_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen36Fp8PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(
            (TOKENS * (Qwen36Moe35B::NUM_ATTENTION_HEADS + Qwen36Moe35B::NUM_KV_HEADS))
                .div_ceil(WARPS_PER_CTA),
        )
        .map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 FP8 attention Q/K prefill grid exceeds u32")
        })?;

        Ok(Self {
            prepare: module
                .prepare_qwen36_fp8_attention_qk_prepare_prefill_exact::<TOKENS>(
                    LaunchConfig1D::new(blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.6 FP8 attention Q/K prefill", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                qkv,
                query_norm,
                key_norm,
                rope_cos,
                rope_sin,
                block_tables,
                table_rows,
                table_stride,
                cache_positions,
                query,
                key_pages,
                value_pages,
                key_scale,
                value_scale,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 FP8 attention Q/K prefill", source)
            })
    }
}

/// Prepared Q/K normalization, MRoPE, and KV-cache append routes for exact
/// `B=1..8` decode and `T=32,64,128,1024` prefill widths.
pub struct AttentionQkPrepareOp<A: Sm120Arch = Qwen38_27B> {
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
    t1024: PreparedPrefillRoute<A, 1_024>,
}

impl<A: Sm120Arch> AttentionQkPrepareOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen38_geometry::<A>()?;
        let _ = attention_qk_prepare_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading attention Q/K prepare", source))?;

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
            t1024: PreparedPrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Normalizes and rotates Q/K, then appends represented E4M3 K/V codes.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[tokens, A::ATTENTION_QKV_ROWS]` BF16 values in the fused
    /// query/gate, key, value order. Norms cover `A::HEAD_DIM`; rotary planes
    /// cover `[tokens, 32]`; metadata covers `tokens`; and the block-table row
    /// selected for each token covers its cache position. Query covers
    /// `[batch, A::NUM_ATTENTION_HEADS, A::HEAD_DIM]` FP32 values. Cache
    /// planes use page-major `[physical_page, A::NUM_KV_HEADS, 64,
    /// A::HEAD_DIM]` bytes and cover every selected page.
    /// Allocations are aligned, non-overlapping, live through completion, and
    /// belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !admitted_tokens(tokens) {
            return Err(GpuError::invalid_launch(format!(
                "attention Q/K prepare tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128,1024"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("attention Q/K block-table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "attention Q/K block-table stride must be nonzero",
            ));
        }
        if !key_scale.is_finite() || key_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "attention key-cache scale must be finite and positive",
            ));
        }
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "attention value-cache scale must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        qkv,
                        query_norm,
                        key_norm,
                        rope_cos,
                        rope_sin,
                        block_tables,
                        table_rows,
                        table_stride,
                        cache_positions,
                        query,
                        key_pages,
                        value_pages,
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
            1_024 => launch!(t1024),
            _ => unreachable!(),
        }
    }
}

/// Prepared Qwen3.5 Q/K normalization, MRoPE, and KV-cache routes.
pub struct Qwen35AttentionQkPrepareOp {
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
    t1024: PreparedQwen35PrefillRoute<1_024>,
}

impl Qwen35AttentionQkPrepareOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen35_geometry()?;
        let _ = qwen35_attention_qk_prepare_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.5 attention Q/K prepare", source))?;

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
            t1024: PreparedQwen35PrefillRoute::prepare(&module)?,
            module,
        })
    }

    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[batch, 10_240]` BF16 values in query/gate, key, value
    /// order. Norms cover 256 values; rotary planes cover `[batch, 32]`;
    /// metadata covers `batch`; and each selected block-table row covers its
    /// cache position. Query covers `[batch, 16, 256]` FP32 values. Cache
    /// planes use page-major `[physical_page, 4, 64, 256]` BF16 values.
    /// Allocations are aligned, non-overlapping, live through completion, and
    /// belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_tokens(tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 attention Q/K prepare tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128,1024"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 attention Q/K block-table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 attention Q/K block-table stride must be nonzero",
            ));
        }
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        qkv,
                        query_norm,
                        key_norm,
                        rope_cos,
                        rope_sin,
                        block_tables,
                        table_rows,
                        table_stride,
                        cache_positions,
                        query,
                        key_pages,
                        value_pages,
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
            1_024 => launch!(t1024),
            _ => unreachable!(),
        }
    }
}

/// Prepared Qwen3.6 Q/K normalization, MRoPE, and BF16 KV-cache decode and prompt routes.
pub struct Qwen36AttentionQkPrepareOp {
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

impl Qwen36AttentionQkPrepareOp {
    /// Loads the embedded module and prepares every exact Qwen3.6 decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen36_geometry()?;
        let _ = qwen36_attention_qk_prepare_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.6 attention Q/K prepare", source))?;

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

    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[tokens,9216]` BF16 values in query/gate, key, value order.
    /// Norms cover 256 values; rotary planes cover `[tokens,32]`; metadata
    /// covers `tokens`; and each selected block-table row covers its cache
    /// position. Query covers `[tokens,16,256]` FP32 values. Cache planes use
    /// page-major `[physical_page,2,64,256]` BF16 values. Allocations are
    /// aligned, non-overlapping, live through completion, and context-local.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u16,
        value_pages: *mut u16,
    ) -> GpuResult<()> {
        if !admitted_qwen36_tokens(tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 attention Q/K prepare tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 attention Q/K block-table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 attention Q/K block-table stride must be nonzero",
            ));
        }
        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        qkv,
                        query_norm,
                        key_norm,
                        rope_cos,
                        rope_sin,
                        block_tables,
                        table_rows,
                        table_stride,
                        cache_positions,
                        query,
                        key_pages,
                        value_pages,
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

/// Prepared Qwen3.6 Q/K normalization, MRoPE, and E4M3 KV-cache routes.
pub struct Qwen36Fp8AttentionQkPrepareOp {
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

impl Qwen36Fp8AttentionQkPrepareOp {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_qwen36_geometry()?;
        let _ = qwen36_fp8_attention_qk_prepare_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading Qwen3.6 FP8 attention Q/K prepare", source)
        })?;

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

    /// Normalizes and rotates Q/K, then appends represented E4M3 K/V codes.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[tokens,9216]` BF16 values in query/gate, key, value order.
    /// Norms cover 256 values; rotary planes cover `[tokens,32]`; metadata
    /// covers `tokens`; and each selected block-table row covers its cache
    /// position. Query covers `[tokens,16,256]` FP32 values. Cache planes use
    /// page-major `[physical_page,2,64,256]` E4M3 bytes. Allocations are
    /// aligned, non-overlapping, live through completion, and context-local.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        qkv: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        rope_cos: *const f32,
        rope_sin: *const f32,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        cache_positions: *const u32,
        query: *mut f32,
        key_pages: *mut u8,
        value_pages: *mut u8,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !admitted_qwen36_tokens(tokens) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 FP8 attention Q/K prepare tokens {tokens} must be one of 1..={MAX_BATCH},32,64,128"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("Qwen3.6 FP8 attention Q/K block-table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 FP8 attention Q/K block-table stride must be nonzero",
            ));
        }
        if !key_scale.is_finite() || key_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 key-cache scale must be finite and positive",
            ));
        }
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.6 value-cache scale must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
                        &self.module,
                        stream,
                        qkv,
                        query_norm,
                        key_norm,
                        rope_cos,
                        rope_sin,
                        block_tables,
                        table_rows,
                        table_stride,
                        cache_positions,
                        query,
                        key_pages,
                        value_pages,
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

/// PTX symbols retained for every exact attention Q/K prepare route.
pub(crate) fn attention_qk_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::attention_qk_prepare_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::attention_qk_prepare_prefill_exact_ptx_name::<Qwen38_27B, 32>(),
        kernels::attention_qk_prepare_prefill_exact_ptx_name::<Qwen38_27B, 64>(),
        kernels::attention_qk_prepare_prefill_exact_ptx_name::<Qwen38_27B, 128>(),
        kernels::attention_qk_prepare_prefill_exact_ptx_name::<Qwen38_27B, 1_024>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.5 attention Q/K prepare route.
pub(crate) fn qwen35_attention_qk_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<1>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<2>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<3>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<4>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<5>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<6>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<7>(),
        kernels::qwen35_attention_qk_prepare_exact_ptx_name::<8>(),
        kernels::qwen35_attention_qk_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen35_attention_qk_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen35_attention_qk_prepare_prefill_exact_ptx_name::<128>(),
        kernels::qwen35_attention_qk_prepare_prefill_exact_ptx_name::<1_024>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.6 attention Q/K prepare route.
pub(crate) fn qwen36_attention_qk_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<1>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<2>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<3>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<4>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<5>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<6>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<7>(),
        kernels::qwen36_attention_qk_prepare_exact_ptx_name::<8>(),
        kernels::qwen36_attention_qk_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen36_attention_qk_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen36_attention_qk_prepare_prefill_exact_ptx_name::<128>(),
    ]
}

/// PTX symbols retained for every exact Qwen3.6 E4M3-cache preparation route.
pub(crate) fn qwen36_fp8_attention_qk_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<1>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<2>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<3>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<4>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<5>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<6>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<7>(),
        kernels::qwen36_fp8_attention_qk_prepare_exact_ptx_name::<8>(),
        kernels::qwen36_fp8_attention_qk_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen36_fp8_attention_qk_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen36_fp8_attention_qk_prepare_prefill_exact_ptx_name::<128>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN36_PREFILL_TOKENS, THREADS, admitted_qwen36_tokens, admitted_tokens,
        attention_qk_prepare_ptx_names, qwen35_attention_qk_prepare_ptx_names,
        qwen36_attention_qk_prepare_ptx_names, qwen36_fp8_attention_qk_prepare_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn route_table_covers_only_exact_decode_and_prefill_widths() {
        for (tokens, expected) in [
            (0, false),
            (1, true),
            (4, true),
            (8, true),
            (9, false),
            (16, false),
            (32, true),
            (64, true),
            (128, true),
            (1_024, true),
            (1_025, false),
        ] {
            assert_eq!(admitted_tokens(tokens), expected, "tokens={tokens}");
        }
        assert_eq!(THREADS, 256);

        for (tokens, expected) in [
            (0, false),
            (1, true),
            (8, true),
            (9, false),
            (32, true),
            (64, true),
            (128, true),
            (1_024, false),
        ] {
            assert_eq!(admitted_qwen36_tokens(tokens), expected, "tokens={tokens}");
        }
        assert_eq!(QWEN36_PREFILL_TOKENS, [32, 64, 128]);
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_exact_route() {
        let names = attention_qk_prepare_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 12);
        assert_eq!(unique.len(), names.len());

        let qwen35 = qwen35_attention_qk_prepare_ptx_names();
        let qwen35_unique = qwen35.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen35.len(), 12);
        assert_eq!(qwen35_unique.len(), qwen35.len());
        assert!(names.iter().all(|name| !qwen35_unique.contains(name)));

        let qwen36 = qwen36_attention_qk_prepare_ptx_names();
        let qwen36_unique = qwen36.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen36.len(), 11);
        assert_eq!(qwen36_unique.len(), qwen36.len());
        assert!(names.iter().all(|name| !qwen36_unique.contains(name)));
        assert!(qwen35_unique.is_disjoint(&qwen36_unique));

        let qwen36_fp8 = qwen36_fp8_attention_qk_prepare_ptx_names();
        let qwen36_fp8_unique = qwen36_fp8.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(qwen36_fp8.len(), 11);
        assert_eq!(qwen36_fp8_unique.len(), qwen36_fp8.len());
        assert!(qwen36_fp8_unique.is_disjoint(&qwen36_unique));
    }
}

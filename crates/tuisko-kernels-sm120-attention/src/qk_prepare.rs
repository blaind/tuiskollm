//! Exact-batch Q/K normalization, MRoPE, and KV-cache append.

use crate::device::attention_qk_prepare::{attention_qk_prepare, bf16_attention_qk_prepare};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

/// Number of token positions held by one physical KV-cache page.
pub const ATTENTION_PAGE_SIZE: usize = 64;

const MAX_BATCH: usize = 8;
const PREFILL_TOKENS: [usize; 4] = [32, 64, 128, 1_024];
const QWEN36_PREFILL_TOKENS: [usize; 3] = [32, 64, 128];
const WARPS_PER_CTA: usize = 8;
const THREADS: u32 = (WARPS_PER_CTA * 32) as u32;

// One warp owns one complete head on every admitted route, so all four
// owners' grids are the same head-warp division: 28/20/18 CTAs at B=8 for
// Qwen3.8/3.5/3.6. Merging the owners keeps that division in one place.
fn head_warp_blocks<A: Arch>(tokens: usize, overflow: &'static str) -> GpuResult<u32> {
    u32::try_from((tokens * (A::NUM_ATTENTION_HEADS + A::NUM_KV_HEADS)).div_ceil(WARPS_PER_CTA))
        .map_err(|_| GpuError::invalid_launch(overflow))
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

// The 24/2 head split, packed query rows, and cache stride are target-specific.
fn require_qwen38_flash_next_geometry() -> GpuResult<()> {
    if Qwen38FlashNext::NUM_ATTENTION_HEADS != 24
        || Qwen38FlashNext::NUM_KV_HEADS != 2
        || Qwen38FlashNext::HEAD_DIM != 256
        || Qwen38FlashNext::ATTENTION_QUERY_ROWS != 12_288
        || Qwen38FlashNext::ATTENTION_KV_ROWS != 512
        || Qwen38FlashNext::ATTENTION_QKV_ROWS != 13_312
        || Qwen38FlashNext::RMS_NORM_EPSILON != 1.0e-6
    {
        return Err(GpuError::invalid_launch(
            "Qwen3.8-Flash-Next geometry is incompatible with its admitted QSA Q/K prepare schedule",
        ));
    }

    Ok(())
}

// Keep generated address arithmetic bound to the admitted geometry.
const _: () = assert!(Qwen38FlashNext::NUM_ATTENTION_HEADS == 24);
const _: () = assert!(Qwen38FlashNext::NUM_KV_HEADS == 2);
const _: () = assert!(Qwen38FlashNext::HEAD_DIM == 256);
const _: () = assert!(Qwen38FlashNext::ATTENTION_QUERY_ROWS == 12_288);
const _: () = assert!(Qwen38FlashNext::ATTENTION_KV_ROWS == 512);
const _: () = assert!(Qwen38FlashNext::ATTENTION_QKV_ROWS == 13_312);

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

    /// Prepares Qwen3.8-Flash-Next QSA Q/K and appends E4M3 K/V for one exact batch.
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
    pub fn qwen38_flash_next_attention_qk_prepare_exact<const TOKENS: usize>(
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
        // Qwen3.8-Flash-Next has 26 head-warps per token (24 query + 2 KV). Eight
        // warps per CTA gives 26 CTAs at B=8 while keeping one warp per
        // complete 256-wide head, which is what preserves the exact RMS
        // reduction order and the 64-wide partial-MRoPE lane exchange.
        unsafe {
            attention_qk_prepare::<Qwen38FlashNext, TOKENS>(
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

    /// Prepares Qwen3.8-Flash-Next QSA Q/K and appends E4M3 K/V for one prompt width.
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
    pub fn qwen38_flash_next_attention_qk_prepare_prefill_exact<const TOKENS: usize>(
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
        // T=32/64/128/1024 supplies 104/208/416/3,328 CTAs. Each warp keeps
        // the decode route's 256-value reduction, MRoPE exchange, and E4M3
        // store order, so a prompt and a decode step append bit-identical
        // cache rows for the same token.
        unsafe {
            attention_qk_prepare::<Qwen38FlashNext, TOKENS>(
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

mod private {
    pub trait Sealed {}
}

/// Represented storage format of one admitted route's K/V cache planes.
///
/// Sealed: these two formats are the only ones this family emits entries for,
/// so an entry table can never name a third.
pub trait CacheFormat: private::Sealed {
    /// Storage element of the key and value page planes.
    type Element: Copy;
    /// Per-plane scales the append stage applies, if the format carries any.
    type Scales: Copy;
}

/// Page planes holding represented E4M3 codes with per-plane scales.
pub struct Fp8Cache;

/// Page planes holding represented BF16 values, which carry no scale.
pub struct Bf16Cache;

/// Represented key and value cache scales one E4M3 append applies.
#[derive(Clone, Copy)]
pub struct CacheScales {
    pub(crate) key: f32,
    pub(crate) value: f32,
}

impl private::Sealed for Fp8Cache {}
impl private::Sealed for Bf16Cache {}

impl CacheFormat for Fp8Cache {
    type Element = u8;
    type Scales = CacheScales;
}

impl CacheFormat for Bf16Cache {
    type Element = u16;
    type Scales = ();
}

/// One launch's Q/K prepare operands in the entries' parameter order.
///
/// The four merged owners pass the same operands to every route, so bundling
/// them keeps the dispatch's argument order identical to the launchers it
/// replaces.
pub struct QkPrepareArgs<C: CacheFormat> {
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
    key_pages: *mut C::Element,
    value_pages: *mut C::Element,
    scales: C::Scales,
}

/// One architecture's prepared entry for an exact token count.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait QkPrepareRoute<C: CacheFormat>: Sized + private::Sealed {
    /// Prepares this route's entry at its qualified head-warp grid.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's entry.
    ///
    /// # Safety
    ///
    /// `args` carries `AttentionQkPrepareOp::launch`'s pointer contract
    /// unchanged.
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<C>,
    ) -> GpuResult<()>;
}

/// Prepared Qwen3.8 decode entry for one exact batch.
pub struct PreparedRoute<A: Arch, const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__attention_qk_prepare_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.8 prefill entry for one exact prompt width.
pub struct PreparedPrefillRoute<A: Arch, const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__attention_qk_prepare_prefill_exact_CudaKernel<A, TOKENS>>,
}

/// Prepared Qwen3.5 decode entry for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 prefill entry for one exact prompt width.
pub struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen35_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 BF16-cache decode entry for one exact batch.
pub struct PreparedQwen36Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen36_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 BF16-cache prefill entry for one exact prompt width.
pub struct PreparedQwen36PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen36_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 E4M3-cache decode entry for one exact batch.
pub struct PreparedQwen36Fp8Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen36_fp8_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.6 E4M3-cache prefill entry for one exact prompt width.
pub struct PreparedQwen36Fp8PrefillRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen36_fp8_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8-Flash-Next QSA E4M3-cache decode entry for one exact batch.
pub struct PreparedQwen38FlashNextRoute<const TOKENS: usize> {
    prepare:
        PreparedLaunch<kernels::__qwen38_flash_next_attention_qk_prepare_exact_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8-Flash-Next QSA E4M3-cache prefill entry for one prompt width.
pub struct PreparedQwen38FlashNextPrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<
        kernels::__qwen38_flash_next_attention_qk_prepare_prefill_exact_CudaKernel<TOKENS>,
    >,
}

/// Stands in for a prompt width an architecture does not admit.
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

impl<A: Arch, const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = head_warp_blocks::<A>(TOKENS, "attention Q/K prepare grid exceeds u32")?;

        Ok(Self {
            prepare: module
                .prepare_attention_qk_prepare_exact::<A, TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| GpuError::launch("preparing attention Q/K route", source))?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .attention_qk_prepare_exact::<A, TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching attention Q/K prepare", source))
    }
}

impl<A: Arch, const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedPrefillRoute<A, TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = head_warp_blocks::<A>(TOKENS, "attention Q/K prefill grid exceeds u32")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .attention_qk_prepare_prefill_exact::<A, TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching attention Q/K prefill", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Bf16Cache> for PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks =
            head_warp_blocks::<Qwen35_9B>(TOKENS, "Qwen3.5 attention Q/K grid exceeds u32")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen35_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 attention Q/K prepare", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Bf16Cache> for PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 attention Q/K prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = head_warp_blocks::<Qwen35_9B>(
            TOKENS,
            "Qwen3.5 attention Q/K prefill grid exceeds u32",
        )?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen35_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 attention Q/K prefill", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Bf16Cache> for PreparedQwen36Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks =
            head_warp_blocks::<Qwen36Moe35B>(TOKENS, "Qwen3.6 attention Q/K grid exceeds u32")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 attention Q/K prepare", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Bf16Cache> for PreparedQwen36PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        if !QWEN36_PREFILL_TOKENS.contains(&TOKENS) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.6 attention Q/K prefill route T={TOKENS} is not admitted"
            )));
        }
        let blocks = head_warp_blocks::<Qwen36Moe35B>(
            TOKENS,
            "Qwen3.6 attention Q/K prefill grid exceeds u32",
        )?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Bf16Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 attention Q/K prefill", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedQwen36Fp8Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks =
            head_warp_blocks::<Qwen36Moe35B>(TOKENS, "Qwen3.6 FP8 attention Q/K grid exceeds u32")?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.6 FP8 attention Q/K", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedQwen36Fp8PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = head_warp_blocks::<Qwen36Moe35B>(
            TOKENS,
            "Qwen3.6 FP8 attention Q/K prefill grid exceeds u32",
        )?;

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

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen36_fp8_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.6 FP8 attention Q/K prefill", source)
            })
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedQwen38FlashNextRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = head_warp_blocks::<Qwen38FlashNext>(
            TOKENS,
            "Qwen3.8-Flash-Next QSA Q/K grid exceeds u32",
        )?;

        Ok(Self {
            prepare: module
                .prepare_qwen38_flash_next_attention_qk_prepare_exact::<TOKENS>(
                    LaunchConfig1D::new(blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next QSA Q/K route", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_attention_qk_prepare_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| GpuError::launch("launching Qwen3.8-Flash-Next QSA Q/K", source))
    }
}

impl<const TOKENS: usize> QkPrepareRoute<Fp8Cache> for PreparedQwen38FlashNextPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = head_warp_blocks::<Qwen38FlashNext>(
            TOKENS,
            "Qwen3.8-Flash-Next QSA Q/K prefill grid exceeds u32",
        )?;

        Ok(Self {
            prepare: module
                .prepare_qwen38_flash_next_attention_qk_prepare_prefill_exact::<TOKENS>(
                    LaunchConfig1D::new(blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.8-Flash-Next QSA Q/K prefill", source)
                })?,
        })
    }

    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        args: &QkPrepareArgs<Fp8Cache>,
    ) -> GpuResult<()> {
        module
            .qwen38_flash_next_attention_qk_prepare_prefill_exact::<TOKENS>(
                stream,
                &self.prepare,
                args.qkv,
                args.query_norm,
                args.key_norm,
                args.rope_cos,
                args.rope_sin,
                args.block_tables,
                args.table_rows,
                args.table_stride,
                args.cache_positions,
                args.query,
                args.key_pages,
                args.value_pages,
                args.scales.key,
                args.scales.value,
            )
            .map_err(|source| {
                GpuError::launch("launching Qwen3.8-Flash-Next QSA Q/K prefill", source)
            })
    }
}

// `token_route` rejects an unadmitted width before dispatch, so this is the
// defensive tail of a route that owns no entry.
impl<C: CacheFormat> QkPrepareRoute<C> for UnadmittedRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    unsafe fn launch(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _args: &QkPrepareArgs<C>,
    ) -> GpuResult<()> {
        Err(GpuError::invalid_launch(
            "attention Q/K prepare route is not admitted for this architecture",
        ))
    }
}

/// Exact entry table of one admitted architecture and cache format.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 and Qwen3.6 here never widens the
/// artifact-level admission bound. Each table names only the entries its own
/// model emits, which is what keeps the compiled inventory fixed while the
/// four prepared owners share one wrapper.
pub trait QkPrepareEntries<A: Arch>: private::Sealed {
    /// Represented cache format this table's entries append.
    type Cache: CacheFormat;
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: QkPrepareRoute<Self::Cache>;
    /// Prepared prefill route for `T=32,64,128`.
    type Prefill<const TOKENS: usize>: QkPrepareRoute<Self::Cache>;
    /// Prepared `T=1024` prefill route, unadmitted outside Qwen3.8 and Qwen3.5.
    type Prefill1024: QkPrepareRoute<Self::Cache>;

    /// Whether `T=1024` is an admitted prefill width.
    const HAS_T1024: bool;
    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;
    /// Owner named by this table's E4M3 cache-scale rejections.
    const SCALE_LABEL: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not cover.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Qwen3.8 entry table: E4M3 cache, decode `B=1..=8`, prefill through `T=1024`.
pub struct Qwen38QkPrepareEntries;

/// Qwen3.5 entry table: BF16 cache, decode `B=1..=8`, prefill through `T=1024`.
pub struct Qwen35QkPrepareEntries;

/// Qwen3.6 entry table: BF16 cache, decode `B=1..=8`, prefill through `T=128`.
pub struct Qwen36QkPrepareEntries;

/// Qwen3.6 entry table: E4M3 cache, decode `B=1..=8`, prefill through `T=128`.
pub struct Qwen36Fp8QkPrepareEntries;

/// Qwen3.8-Flash-Next QSA entry table: E4M3 cache, decode `B=1..=8`, prefill to `T=1024`.
pub struct Qwen38FlashNextQkPrepareEntries;

impl private::Sealed for Qwen38QkPrepareEntries {}
impl private::Sealed for Qwen35QkPrepareEntries {}
impl private::Sealed for Qwen36QkPrepareEntries {}
impl private::Sealed for Qwen36Fp8QkPrepareEntries {}
impl private::Sealed for Qwen38FlashNextQkPrepareEntries {}

// The Qwen3.8 entries stay bound to the sealed artifact-level architecture:
// they are the only routes whose kernels are instantiated over `A`.
impl<A: Sm120Arch> QkPrepareEntries<A> for Qwen38QkPrepareEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedRoute<A, TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<A, TOKENS>;
    type Prefill1024 = PreparedPrefillRoute<A, 1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "";
    const SCALE_LABEL: &'static str = "attention";
    const MODULE_OPERATION: &'static str = "loading attention Q/K prepare";

    fn require_geometry() -> GpuResult<()> {
        require_qwen38_geometry::<A>()
    }

    fn ptx_names() -> Vec<&'static str> {
        attention_qk_prepare_ptx_names()
    }
}

impl QkPrepareEntries<Qwen35_9B> for Qwen35QkPrepareEntries {
    type Cache = Bf16Cache;
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen35PrefillRoute<TOKENS>;
    type Prefill1024 = PreparedQwen35PrefillRoute<1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "Qwen3.5 ";
    const SCALE_LABEL: &'static str = "Qwen3.5";
    const MODULE_OPERATION: &'static str = "loading Qwen3.5 attention Q/K prepare";

    fn require_geometry() -> GpuResult<()> {
        require_qwen35_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_attention_qk_prepare_ptx_names()
    }
}

impl QkPrepareEntries<Qwen36Moe35B> for Qwen36QkPrepareEntries {
    type Cache = Bf16Cache;
    type Decode<const TOKENS: usize> = PreparedQwen36Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen36PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.6 ";
    const SCALE_LABEL: &'static str = "Qwen3.6";
    const MODULE_OPERATION: &'static str = "loading Qwen3.6 attention Q/K prepare";

    fn require_geometry() -> GpuResult<()> {
        require_qwen36_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_attention_qk_prepare_ptx_names()
    }
}

impl QkPrepareEntries<Qwen36Moe35B> for Qwen36Fp8QkPrepareEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedQwen36Fp8Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen36Fp8PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.6 FP8 ";
    const SCALE_LABEL: &'static str = "Qwen3.6";
    const MODULE_OPERATION: &'static str = "loading Qwen3.6 FP8 attention Q/K prepare";

    fn require_geometry() -> GpuResult<()> {
        require_qwen36_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen36_fp8_attention_qk_prepare_ptx_names()
    }
}

impl QkPrepareEntries<Qwen38FlashNext> for Qwen38FlashNextQkPrepareEntries {
    type Cache = Fp8Cache;
    type Decode<const TOKENS: usize> = PreparedQwen38FlashNextRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen38FlashNextPrefillRoute<TOKENS>;
    type Prefill1024 = PreparedQwen38FlashNextPrefillRoute<1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "Qwen3.8-Flash-Next QSA ";
    const SCALE_LABEL: &'static str = "Qwen3.8-Flash-Next QSA";
    const MODULE_OPERATION: &'static str = "loading Qwen3.8-Flash-Next QSA Q/K prepare";

    fn require_geometry() -> GpuResult<()> {
        require_qwen38_flash_next_geometry()
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen38_flash_next_attention_qk_prepare_ptx_names()
    }
}

/// PTX symbols retained for every exact Qwen3.8-Flash-Next QSA Q/K prepare route.
pub(crate) fn qwen38_flash_next_attention_qk_prepare_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<1>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<2>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<3>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<4>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<5>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<6>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<7>(),
        kernels::qwen38_flash_next_attention_qk_prepare_exact_ptx_name::<8>(),
        kernels::qwen38_flash_next_attention_qk_prepare_prefill_exact_ptx_name::<32>(),
        kernels::qwen38_flash_next_attention_qk_prepare_prefill_exact_ptx_name::<64>(),
        kernels::qwen38_flash_next_attention_qk_prepare_prefill_exact_ptx_name::<128>(),
        kernels::qwen38_flash_next_attention_qk_prepare_prefill_exact_ptx_name::<1_024>(),
    ]
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

/// The compiled route one admitted token count selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenRoute {
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

// The admitted token schedule, transcribed from the four prepared dispatches
// it replaces: decode B=1..=8 and prefill T=32,64,128 everywhere, and the
// T=1024 prefill only where the entry table admits it.
fn token_route<A: Arch, E: QkPrepareEntries<A>>(tokens: usize) -> Option<TokenRoute> {
    match tokens {
        1 => Some(TokenRoute::B1),
        2 => Some(TokenRoute::B2),
        3 => Some(TokenRoute::B3),
        4 => Some(TokenRoute::B4),
        5 => Some(TokenRoute::B5),
        6 => Some(TokenRoute::B6),
        7 => Some(TokenRoute::B7),
        8 => Some(TokenRoute::B8),
        32 => Some(TokenRoute::T32),
        64 => Some(TokenRoute::T64),
        128 => Some(TokenRoute::T128),
        1_024 if E::HAS_T1024 => Some(TokenRoute::T1024),
        _ => None,
    }
}

fn admitted_prefill_widths<A: Arch, E: QkPrepareEntries<A>>() -> &'static str {
    if E::HAS_T1024 {
        "32,64,128,1024"
    } else {
        "32,64,128"
    }
}

fn unsupported_tokens<A: Arch, E: QkPrepareEntries<A>>(tokens: usize) -> GpuError {
    GpuError::invalid_launch(format!(
        "{}attention Q/K prepare tokens {tokens} must be one of 1..={MAX_BATCH},{}",
        E::LABEL,
        admitted_prefill_widths::<A, E>(),
    ))
}

fn checked_table_stride<A: Arch, E: QkPrepareEntries<A>>(table_stride: usize) -> GpuResult<u32> {
    let table_stride = u32::try_from(table_stride).map_err(|_| {
        GpuError::invalid_launch(format!(
            "{}attention Q/K block-table stride exceeds u32",
            E::LABEL
        ))
    })?;
    if table_stride == 0 {
        return Err(GpuError::invalid_launch(format!(
            "{}attention Q/K block-table stride must be nonzero",
            E::LABEL
        )));
    }

    Ok(table_stride)
}

fn checked_cache_scales<A: Arch, E: QkPrepareEntries<A>>(
    key_scale: f32,
    value_scale: f32,
) -> GpuResult<CacheScales> {
    if !key_scale.is_finite() || key_scale <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{} key-cache scale must be finite and positive",
            E::SCALE_LABEL
        )));
    }
    if !value_scale.is_finite() || value_scale <= 0.0 {
        return Err(GpuError::invalid_launch(format!(
            "{} value-cache scale must be finite and positive",
            E::SCALE_LABEL
        )));
    }

    Ok(CacheScales {
        key: key_scale,
        value: value_scale,
    })
}

/// Prepared Q/K normalization, MRoPE, and KV-cache append routes for exact
/// `B=1..8` decode and the entry table's admitted prefill widths.
///
/// `C` restates `E::Cache` as a parameter so the E4M3 and BF16 launch
/// signatures live in disjoint inherent impls: an E4M3 append carries the two
/// represented cache scales a BF16 append does not have.
pub struct AttentionQkPrepareOp<
    A: Arch = Qwen38_27B,
    C: CacheFormat = Fp8Cache,
    E: QkPrepareEntries<A, Cache = C> = Qwen38QkPrepareEntries,
> {
    module: kernels::LoadedModule,
    b1: E::Decode<1>,
    b2: E::Decode<2>,
    b3: E::Decode<3>,
    b4: E::Decode<4>,
    b5: E::Decode<5>,
    b6: E::Decode<6>,
    b7: E::Decode<7>,
    b8: E::Decode<8>,
    t32: E::Prefill<32>,
    t64: E::Prefill<64>,
    t128: E::Prefill<128>,
    t1024: E::Prefill1024,
    cache: PhantomData<C>,
}

/// Prepared Qwen3.5 BF16-cache Q/K normalization, MRoPE, and KV-cache routes.
pub type Qwen35AttentionQkPrepareOp =
    AttentionQkPrepareOp<Qwen35_9B, Bf16Cache, Qwen35QkPrepareEntries>;

/// Prepared Qwen3.6 BF16-cache Q/K normalization, MRoPE, and KV-cache routes.
pub type Qwen36AttentionQkPrepareOp =
    AttentionQkPrepareOp<Qwen36Moe35B, Bf16Cache, Qwen36QkPrepareEntries>;

/// Prepared Qwen3.6 E4M3-cache Q/K normalization, MRoPE, and KV-cache routes.
pub type Qwen36Fp8AttentionQkPrepareOp =
    AttentionQkPrepareOp<Qwen36Moe35B, Fp8Cache, Qwen36Fp8QkPrepareEntries>;

/// Prepared Qwen3.8-Flash-Next QSA Q/K normalization, partial MRoPE, and KV-cache routes.
///
/// Appends normalized, rotated keys and raw values.
pub type Qwen38FlashNextAttentionQkPrepareOp =
    AttentionQkPrepareOp<Qwen38FlashNext, Fp8Cache, Qwen38FlashNextQkPrepareEntries>;

impl<A: Arch, C: CacheFormat, E: QkPrepareEntries<A, Cache = C>> AttentionQkPrepareOp<A, C, E> {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_OPERATION, source))?;

        Ok(Self {
            b1: E::Decode::<1>::prepare(&module)?,
            b2: E::Decode::<2>::prepare(&module)?,
            b3: E::Decode::<3>::prepare(&module)?,
            b4: E::Decode::<4>::prepare(&module)?,
            b5: E::Decode::<5>::prepare(&module)?,
            b6: E::Decode::<6>::prepare(&module)?,
            b7: E::Decode::<7>::prepare(&module)?,
            b8: E::Decode::<8>::prepare(&module)?,
            t32: E::Prefill::<32>::prepare(&module)?,
            t64: E::Prefill::<64>::prepare(&module)?,
            t128: E::Prefill::<128>::prepare(&module)?,
            t1024: E::Prefill1024::prepare(&module)?,
            cache: PhantomData,
            module,
        })
    }

    /// Dispatches one prepared route.
    ///
    /// # Safety
    ///
    /// `args` carries the caller's pointer contract unchanged.
    unsafe fn dispatch(
        &self,
        stream: &CudaStream,
        route: TokenRoute,
        args: &QkPrepareArgs<C>,
    ) -> GpuResult<()> {
        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: the caller's contract reaches the entry unchanged.
                unsafe { self.$route.launch(&self.module, stream, args) }
            };
        }

        match route {
            TokenRoute::B1 => launch!(b1),
            TokenRoute::B2 => launch!(b2),
            TokenRoute::B3 => launch!(b3),
            TokenRoute::B4 => launch!(b4),
            TokenRoute::B5 => launch!(b5),
            TokenRoute::B6 => launch!(b6),
            TokenRoute::B7 => launch!(b7),
            TokenRoute::B8 => launch!(b8),
            TokenRoute::T32 => launch!(t32),
            TokenRoute::T64 => launch!(t64),
            TokenRoute::T128 => launch!(t128),
            TokenRoute::T1024 => launch!(t1024),
        }
    }
}

impl<A: Arch, E: QkPrepareEntries<A, Cache = Fp8Cache>> AttentionQkPrepareOp<A, Fp8Cache, E> {
    /// Normalizes and rotates Q/K, then appends represented E4M3 K/V codes.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[tokens, A::ATTENTION_QKV_ROWS]` BF16 values in the fused
    /// query/gate, key, value order. Norms cover `A::HEAD_DIM`; rotary planes
    /// cover `[tokens, 32]`; metadata covers `tokens`; and the block-table row
    /// selected for each token covers its cache position. Query covers
    /// `[tokens, A::NUM_ATTENTION_HEADS, A::HEAD_DIM]` FP32 values. Cache
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
        let route =
            token_route::<A, E>(tokens).ok_or_else(|| unsupported_tokens::<A, E>(tokens))?;
        let table_stride = checked_table_stride::<A, E>(table_stride)?;
        let scales = checked_cache_scales::<A, E>(key_scale, value_scale)?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe {
            self.dispatch(
                stream,
                route,
                &QkPrepareArgs {
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
                    scales,
                },
            )
        }
    }
}

impl<A: Arch, E: QkPrepareEntries<A, Cache = Bf16Cache>> AttentionQkPrepareOp<A, Bf16Cache, E> {
    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[tokens, A::ATTENTION_QKV_ROWS]` BF16 values in the fused
    /// query/gate, key, value order. Norms cover `A::HEAD_DIM`; rotary planes
    /// cover `[tokens, 32]`; metadata covers `tokens`; and the block-table row
    /// selected for each token covers its cache position. Query covers
    /// `[tokens, A::NUM_ATTENTION_HEADS, A::HEAD_DIM]` FP32 values. Cache
    /// planes use page-major `[physical_page, A::NUM_KV_HEADS, 64,
    /// A::HEAD_DIM]` BF16 values and cover every selected page.
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
        let route =
            token_route::<A, E>(tokens).ok_or_else(|| unsupported_tokens::<A, E>(tokens))?;
        let table_stride = checked_table_stride::<A, E>(table_stride)?;

        // SAFETY: the caller's pointer contract reaches the entry unchanged.
        unsafe {
            self.dispatch(
                stream,
                route,
                &QkPrepareArgs {
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
                    scales: (),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bf16Cache, CacheFormat, Fp8Cache, MAX_BATCH, PREFILL_TOKENS, QWEN36_PREFILL_TOKENS,
        QkPrepareEntries, Qwen35QkPrepareEntries, Qwen36Fp8QkPrepareEntries,
        Qwen36QkPrepareEntries, Qwen38QkPrepareEntries, THREADS, TokenRoute, WARPS_PER_CTA,
        attention_qk_prepare_ptx_names, head_warp_blocks, qwen35_attention_qk_prepare_ptx_names,
        qwen36_attention_qk_prepare_ptx_names, qwen36_fp8_attention_qk_prepare_ptx_names,
        token_route, unsupported_tokens,
    };
    use super::{
        Qwen38FlashNextQkPrepareEntries, qwen38_flash_next_attention_qk_prepare_ptx_names,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use tuisko_model::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext};

    /// The decode and prefill widths every admitted entry table routes.
    const SHARED_SCHEDULE: [(usize, TokenRoute); 11] = [
        (1, TokenRoute::B1),
        (2, TokenRoute::B2),
        (3, TokenRoute::B3),
        (4, TokenRoute::B4),
        (5, TokenRoute::B5),
        (6, TokenRoute::B6),
        (7, TokenRoute::B7),
        (8, TokenRoute::B8),
        (32, TokenRoute::T32),
        (64, TokenRoute::T64),
        (128, TokenRoute::T128),
    ];

    /// Every token count the entry table admits, swept exhaustively so an
    /// unadmitted width cannot hide between the transcribed ones.
    fn admitted_schedule<A: Arch, E: QkPrepareEntries<A>>() -> Vec<(usize, TokenRoute)> {
        (0..=2_048)
            .chain([usize::MAX])
            .filter_map(|tokens| token_route::<A, E>(tokens).map(|route| (tokens, route)))
            .collect()
    }

    fn cache_element_bytes<A: Arch, E: QkPrepareEntries<A>>() -> usize {
        size_of::<<E::Cache as CacheFormat>::Element>()
    }

    fn base_name(name: &str) -> &str {
        name.split_once("_TID_").map_or(name, |(base, _)| base)
    }

    /// The merged schedule, checked against the four dispatches it replaces:
    /// Qwen3.6 stops at `T=128` in both cache formats, and Qwen3.8 and
    /// Qwen3.5 admit `T=1024`.
    #[test]
    fn route_table_covers_only_exact_decode_and_prefill_widths() {
        let with_t1024 = SHARED_SCHEDULE
            .iter()
            .copied()
            .chain([(1_024, TokenRoute::T1024)])
            .collect::<Vec<_>>();

        assert_eq!(
            admitted_schedule::<Qwen38_27B, Qwen38QkPrepareEntries>(),
            with_t1024
        );
        assert_eq!(
            admitted_schedule::<Qwen35_9B, Qwen35QkPrepareEntries>(),
            with_t1024
        );
        assert_eq!(
            admitted_schedule::<Qwen36Moe35B, Qwen36QkPrepareEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule::<Qwen36Moe35B, Qwen36Fp8QkPrepareEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
        assert_eq!(
            admitted_schedule::<Qwen38FlashNext, Qwen38FlashNextQkPrepareEntries>(),
            with_t1024
        );

        assert_eq!(THREADS, 256);
        assert_eq!(WARPS_PER_CTA, 8);
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(PREFILL_TOKENS, [32, 64, 128, 1_024]);
        assert_eq!(QWEN36_PREFILL_TOKENS, [32, 64, 128]);
    }

    /// Each entry table keeps its qualified cache format, so merging the four
    /// owners cannot append E4M3 codes into a BF16 plane or the reverse.
    #[test]
    fn every_entry_table_keeps_its_qualified_cache_format() {
        assert_eq!(
            cache_element_bytes::<Qwen38_27B, Qwen38QkPrepareEntries>(),
            size_of::<<Fp8Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen36Moe35B, Qwen36Fp8QkPrepareEntries>(),
            size_of::<<Fp8Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen35_9B, Qwen35QkPrepareEntries>(),
            size_of::<<Bf16Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen36Moe35B, Qwen36QkPrepareEntries>(),
            size_of::<<Bf16Cache as CacheFormat>::Element>()
        );
        assert_eq!(
            cache_element_bytes::<Qwen38FlashNext, Qwen38FlashNextQkPrepareEntries>(),
            size_of::<<Fp8Cache as CacheFormat>::Element>()
        );
    }

    /// One warp owns one complete head, so the merged grid must reproduce the
    /// CTA counts each replaced owner launched: 28/20/18 at `B=8` and
    /// 3,584/2,560/288 at each target's widest prompt.
    #[test]
    fn head_warp_grids_match_each_admitted_route() {
        for (tokens, blocks) in [
            (1, 4),
            (8, 28),
            (32, 112),
            (64, 224),
            (128, 448),
            (1_024, 3_584),
        ] {
            assert_eq!(head_warp_blocks::<Qwen38_27B>(tokens, "").unwrap(), blocks);
        }
        for (tokens, blocks) in [
            (1, 3),
            (8, 20),
            (32, 80),
            (64, 160),
            (128, 320),
            (1_024, 2_560),
        ] {
            assert_eq!(head_warp_blocks::<Qwen35_9B>(tokens, "").unwrap(), blocks);
        }
        for (tokens, blocks) in [(1, 3), (8, 18), (32, 72), (64, 144), (128, 288)] {
            assert_eq!(
                head_warp_blocks::<Qwen36Moe35B>(tokens, "").unwrap(),
                blocks
            );
        }
        // Qwen3.8-Flash-Next QSA is 26 head-warps per token: 24 query plus 2 KV.
        for (tokens, blocks) in [
            (1, 4),
            (8, 26),
            (32, 104),
            (64, 208),
            (128, 416),
            (1_024, 3_328),
        ] {
            assert_eq!(
                head_warp_blocks::<Qwen38FlashNext>(tokens, "").unwrap(),
                blocks
            );
        }
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

        let qwen38_flash_next = qwen38_flash_next_attention_qk_prepare_ptx_names();
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

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38QkPrepareEntries as QkPrepareEntries<Qwen38_27B>>::ptx_names(),
            attention_qk_prepare_ptx_names()
        );
        assert_eq!(
            <Qwen35QkPrepareEntries as QkPrepareEntries<Qwen35_9B>>::ptx_names(),
            qwen35_attention_qk_prepare_ptx_names()
        );
        assert_eq!(
            <Qwen36QkPrepareEntries as QkPrepareEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_attention_qk_prepare_ptx_names()
        );
        assert_eq!(
            <Qwen36Fp8QkPrepareEntries as QkPrepareEntries<Qwen36Moe35B>>::ptx_names(),
            qwen36_fp8_attention_qk_prepare_ptx_names()
        );
        assert_eq!(
            <Qwen38FlashNextQkPrepareEntries as QkPrepareEntries<Qwen38FlashNext>>::ptx_names(),
            qwen38_flash_next_attention_qk_prepare_ptx_names()
        );
    }

    /// A generic specialization's `_TID_` hash is only reproducible inside the
    /// compilation that emitted it, so the stable statement about this family
    /// is its per-base-name count. These are the counts the pinned SM120
    /// device build emits; a wrapper change that instantiates one more
    /// specialization moves one of them.
    #[test]
    fn semantic_entry_inventory_is_pinned_per_base_name() {
        let mut counts = BTreeMap::new();
        for name in attention_qk_prepare_ptx_names()
            .into_iter()
            .chain(qwen35_attention_qk_prepare_ptx_names())
            .chain(qwen36_attention_qk_prepare_ptx_names())
            .chain(qwen36_fp8_attention_qk_prepare_ptx_names())
            .chain(qwen38_flash_next_attention_qk_prepare_ptx_names())
        {
            *counts.entry(base_name(name)).or_insert(0_usize) += 1;
        }

        assert_eq!(
            counts
                .iter()
                .map(|(name, count)| (*name, *count))
                .collect::<Vec<_>>(),
            vec![
                ("attention_qk_prepare_exact", 8),
                ("attention_qk_prepare_prefill_exact", 4),
                ("qwen35_attention_qk_prepare_exact", 8),
                ("qwen35_attention_qk_prepare_prefill_exact", 4),
                ("qwen36_attention_qk_prepare_exact", 8),
                ("qwen36_attention_qk_prepare_prefill_exact", 3),
                ("qwen36_fp8_attention_qk_prepare_exact", 8),
                ("qwen36_fp8_attention_qk_prepare_prefill_exact", 3),
                ("qwen38_flash_next_attention_qk_prepare_exact", 8),
                ("qwen38_flash_next_attention_qk_prepare_prefill_exact", 4),
            ]
        );
        assert_eq!(counts.values().sum::<usize>(), 58);
    }

    /// An unadmitted token count keeps naming the owner that rejected it.
    #[test]
    fn unadmitted_token_counts_name_their_owner() {
        for (message, error) in [
            (
                "attention Q/K prepare tokens 9 must be one of 1..=8,32,64,128,1024",
                unsupported_tokens::<Qwen38_27B, Qwen38QkPrepareEntries>(9),
            ),
            (
                "Qwen3.5 attention Q/K prepare tokens 9 must be one of 1..=8,32,64,128,1024",
                unsupported_tokens::<Qwen35_9B, Qwen35QkPrepareEntries>(9),
            ),
            (
                "Qwen3.6 attention Q/K prepare tokens 1024 must be one of 1..=8,32,64,128",
                unsupported_tokens::<Qwen36Moe35B, Qwen36QkPrepareEntries>(1_024),
            ),
            (
                "Qwen3.6 FP8 attention Q/K prepare tokens 1024 must be one of 1..=8,32,64,128",
                unsupported_tokens::<Qwen36Moe35B, Qwen36Fp8QkPrepareEntries>(1_024),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "unexpected rejection: {error}"
            );
        }
    }
}

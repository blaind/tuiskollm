//! Source-BF16 Q/K preparation and cache append for admitted MTP layers.

use crate::device::attention_qk_prepare::bf16_attention_qk_prepare;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const QWEN35_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const WARPS_PER_CTA: usize = 8;
const THREADS: u32 = (WARPS_PER_CTA * 32) as u32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Prepares Q/K and appends represented BF16 K/V for one exact MTP batch.
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
    pub fn mtp_bf16_qk_prepare<const TOKENS: usize>(
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
        // One warp owns one 256-wide head. Eight warps retain warp-local RMSNorm
        // and MRoPE while B=8 exposes 28 CTAs on the 170-SM target.
        unsafe {
            bf16_attention_qk_prepare::<Qwen38_27B, TOKENS>(
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

    /// Prepares Q/K and appends represented BF16 K/V for one exact MTP prompt tile.
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
    pub fn mtp_bf16_qk_prepare_prefill<const TOKENS: usize>(
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
        // One warp still owns one 256-wide head. The smallest prompt route
        // exposes 112 CTAs and T=1024 exposes 3,584, without changing the
        // warp-local reduction or MRoPE arithmetic qualified for decode.
        unsafe {
            bf16_attention_qk_prepare::<Qwen38_27B, TOKENS>(
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

    /// Prepares Q/K and appends represented BF16 K/V for one exact Qwen3.5 MTP batch.
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
    pub fn qwen35_mtp_bf16_qk_prepare<const TOKENS: usize>(
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
        // One warp owns one 256-wide head. Qwen3.5 B=8 exposes 20 CTAs
        // (16 query + 4 KV heads per row); the schedule preserves warp-local
        // RMSNorm and MRoPE arithmetic while changing only the head count.
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

    /// Prepares Q/K and appends represented BF16 K/V for one exact Qwen3.5 prompt tile.
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
    pub fn qwen35_mtp_bf16_qk_prepare_prefill<const TOKENS: usize>(
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
}

struct PreparedRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__mtp_bf16_qk_prepare_CudaKernel<TOKENS>>,
}

struct PreparedPrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__mtp_bf16_qk_prepare_prefill_CudaKernel<TOKENS>>,
}

struct PreparedQwen35Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_mtp_bf16_qk_prepare_CudaKernel<TOKENS>>,
}

struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_mtp_bf16_qk_prepare_prefill_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let head_warps = TOKENS * (Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS);
        let blocks = u32::try_from(head_warps.div_ceil(WARPS_PER_CTA))
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 MTP BF16 Q/K grid exceeds u32"))?;
        Ok(Self {
            prepare: module
                .prepare_qwen35_mtp_bf16_qk_prepare::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.5 MTP BF16 Q/K route", source)
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
            .qwen35_mtp_bf16_qk_prepare::<TOKENS>(
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
            .map_err(|source| GpuError::launch("launching the Qwen3.5 MTP BF16 Q/K route", source))
    }
}

impl<const TOKENS: usize> PreparedQwen35PrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let head_warps = TOKENS * (Qwen35_9B::NUM_ATTENTION_HEADS + Qwen35_9B::NUM_KV_HEADS);
        let blocks = u32::try_from(head_warps.div_ceil(WARPS_PER_CTA)).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 Q/K prefill grid exceeds u32")
        })?;
        Ok(Self {
            prepare: module
                .prepare_qwen35_mtp_bf16_qk_prepare_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing the Qwen3.5 MTP BF16 Q/K prefill route", source)
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
            .qwen35_mtp_bf16_qk_prepare_prefill::<TOKENS>(
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
            .map_err(|source| {
                GpuError::launch("launching the Qwen3.5 MTP BF16 Q/K prefill route", source)
            })
    }
}

impl<const TOKENS: usize> PreparedPrefillRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let head_warps = TOKENS * (Qwen38_27B::NUM_ATTENTION_HEADS + Qwen38_27B::NUM_KV_HEADS);
        let blocks = u32::try_from(head_warps.div_ceil(WARPS_PER_CTA))
            .map_err(|_| GpuError::invalid_launch("MTP BF16 Q/K prefill grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_mtp_bf16_qk_prepare_prefill::<TOKENS>(LaunchConfig1D::new(
                    blocks, THREADS, 0,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing the MTP BF16 Q/K prefill route", source)
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
            .mtp_bf16_qk_prepare_prefill::<TOKENS>(
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
            .map_err(|source| GpuError::launch("launching the MTP BF16 Q/K prefill route", source))
    }
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let head_warps = TOKENS * (Qwen38_27B::NUM_ATTENTION_HEADS + Qwen38_27B::NUM_KV_HEADS);
        let blocks = u32::try_from(head_warps.div_ceil(WARPS_PER_CTA))
            .map_err(|_| GpuError::invalid_launch("MTP BF16 Q/K grid exceeds u32"))?;

        Ok(Self {
            prepare: module
                .prepare_mtp_bf16_qk_prepare::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| GpuError::launch("preparing the MTP BF16 Q/K route", source))?,
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
            .mtp_bf16_qk_prepare::<TOKENS>(
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
            .map_err(|source| GpuError::launch("launching the MTP BF16 Q/K route", source))
    }
}

/// Stable PTX symbol inventory for every exact MTP Q/K preparation batch.
pub(crate) fn mtp_bf16_qk_prepare_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::mtp_bf16_qk_prepare_ptx_name::<1>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<2>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<3>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<4>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<5>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<6>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<7>(),
        kernels::mtp_bf16_qk_prepare_ptx_name::<8>(),
    ]
}

/// Stable PTX symbol inventory for every exact MTP Q/K prompt tile.
pub(crate) fn mtp_bf16_qk_prepare_prefill_ptx_names() -> [&'static str; 4] {
    [
        kernels::mtp_bf16_qk_prepare_prefill_ptx_name::<32>(),
        kernels::mtp_bf16_qk_prepare_prefill_ptx_name::<64>(),
        kernels::mtp_bf16_qk_prepare_prefill_ptx_name::<128>(),
        kernels::mtp_bf16_qk_prepare_prefill_ptx_name::<1_024>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.5 MTP Q/K route.
pub(crate) fn qwen35_mtp_bf16_qk_prepare_ptx_names() -> [&'static str; 11] {
    [
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<1>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<2>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<3>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<4>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<5>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<6>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<7>(),
        kernels::qwen35_mtp_bf16_qk_prepare_ptx_name::<8>(),
        kernels::qwen35_mtp_bf16_qk_prepare_prefill_ptx_name::<32>(),
        kernels::qwen35_mtp_bf16_qk_prepare_prefill_ptx_name::<64>(),
        kernels::qwen35_mtp_bf16_qk_prepare_prefill_ptx_name::<128>(),
    ]
}

/// Prepared Qwen3.8 MTP Q/K normalization, MRoPE, and BF16 cache routes.
pub struct MtpBf16QkPrepareOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
    t32: PreparedPrefillRoute<32>,
    t64: PreparedPrefillRoute<64>,
    t128: PreparedPrefillRoute<128>,
    t1024: PreparedPrefillRoute<1_024>,
}

impl MtpBf16QkPrepareOp {
    /// Loads the embedded module and prepares every exact MTP decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if Qwen38_27B::NUM_ATTENTION_HEADS != 24
            || Qwen38_27B::NUM_KV_HEADS != 4
            || Qwen38_27B::HEAD_DIM != 256
            || Qwen38_27B::ATTENTION_QUERY_ROWS != 12_288
            || Qwen38_27B::ATTENTION_KV_ROWS != 1_024
            || Qwen38_27B::ATTENTION_QKV_ROWS != 14_336
            || Qwen38_27B::RMS_NORM_EPSILON != 1.0e-6
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 geometry is incompatible with the MTP BF16 Q/K schedule",
            ));
        }
        let _ = (
            mtp_bf16_qk_prepare_ptx_names(),
            mtp_bf16_qk_prepare_prefill_ptx_names(),
        );
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading the MTP BF16 Q/K module", source))?;

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

    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[batch, 14336]` BF16 values in query/gate, key, value
    /// order. Norms cover 256 values; rotary planes cover `[batch, 32]`;
    /// metadata covers `batch`; and each selected block-table row covers its
    /// cache position. Query covers `[batch, 24, 256]` FP32 values. Cache
    /// planes use page-major `[physical_page, 4, 64, 256]` BF16 values.
    /// Allocations are aligned, non-overlapping, context-local, and live
    /// through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
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
        if !admitted_batch(batch) && !PREFILL_ROUTES.contains(&batch) {
            return Err(GpuError::invalid_launch(format!(
                "MTP BF16 Q/K rows {batch} are outside exact B=1..={MAX_BATCH} or T={PREFILL_ROUTES:?}"
            )));
        }
        let table_stride = u32::try_from(table_stride)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 Q/K table stride exceeds u32"))?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "MTP BF16 Q/K table stride must be nonzero",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
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

        macro_rules! launch_prefill {
            ($route:ident) => {
                // SAFETY: exact-T dispatch preserves the public pointer contract.
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

        match batch {
            1 => launch!(b1),
            2 => launch!(b2),
            3 => launch!(b3),
            4 => launch!(b4),
            5 => launch!(b5),
            6 => launch!(b6),
            7 => launch!(b7),
            8 => launch!(b8),
            32 => launch_prefill!(t32),
            64 => launch_prefill!(t64),
            128 => launch_prefill!(t128),
            1_024 => launch_prefill!(t1024),
            _ => unreachable!(),
        }
    }
}

/// Prepared Qwen3.5 MTP Q/K normalization, MRoPE, and BF16 cache routes.
pub struct Qwen35MtpBf16QkPrepareOp {
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

impl Qwen35MtpBf16QkPrepareOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if Qwen35_9B::NUM_ATTENTION_HEADS != 16
            || Qwen35_9B::NUM_KV_HEADS != 4
            || Qwen35_9B::HEAD_DIM != 256
            || Qwen35_9B::ATTENTION_QUERY_ROWS != 8_192
            || Qwen35_9B::ATTENTION_KV_ROWS != 1_024
            || Qwen35_9B::ATTENTION_QKV_ROWS != 10_240
            || Qwen35_9B::RMS_NORM_EPSILON != 1.0e-6
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 geometry is incompatible with the MTP BF16 Q/K schedule",
            ));
        }
        let _ = qwen35_mtp_bf16_qk_prepare_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }.map_err(|source| {
            GpuError::module("loading the Qwen3.5 MTP BF16 Q/K module", source)
        })?;

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

    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[rows, 10240]` BF16 values in query/gate, key, value
    /// order. Norms cover 256 values; rotary planes cover `[rows, 32]`;
    /// metadata covers every row and selected block-table entry. Query covers
    /// `[rows, 16, 256]` FP32 values. Cache planes use page-major
    /// `[physical_page, 4, 64, 256]` BF16 values. Allocations are aligned,
    /// non-overlapping, context-local, and live through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        rows: usize,
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
        if !admitted_batch(rows) && !QWEN35_PREFILL_ROUTES.contains(&rows) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 MTP BF16 Q/K rows {rows} are outside exact B=1..={MAX_BATCH} or T={QWEN35_PREFILL_ROUTES:?}"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 Q/K table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP BF16 Q/K table stride must be nonzero",
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

        match rows {
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

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, PREFILL_ROUTES, QWEN35_PREFILL_ROUTES, THREADS, admitted_batch,
        mtp_bf16_qk_prepare_prefill_ptx_names, mtp_bf16_qk_prepare_ptx_names,
        qwen35_mtp_bf16_qk_prepare_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
        for (batch, admitted) in [(0, false), (1, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), admitted);
        }
        let names = mtp_bf16_qk_prepare_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), MAX_BATCH);
        assert_eq!(unique.len(), names.len());
        let prefill = mtp_bf16_qk_prepare_prefill_ptx_names();
        assert_eq!(PREFILL_ROUTES, [32, 64, 128, 1_024]);
        assert_eq!(prefill.len(), PREFILL_ROUTES.len());
        assert_eq!(prefill.iter().copied().collect::<BTreeSet<_>>().len(), 4);
        assert_eq!(THREADS, 256);
    }

    #[test]
    fn qwen35_inventory_is_complete_and_unique() {
        assert_eq!(QWEN35_PREFILL_ROUTES, [32, 64, 128]);
        let names = qwen35_mtp_bf16_qk_prepare_ptx_names();
        assert_eq!(names.len(), 11);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 11);
    }
}

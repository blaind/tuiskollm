//! Source-BF16 Q/K preparation and cache append for admitted MTP layers.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_attention::shared_device::bf16_attention_qk_prepare;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const QWEN35_PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const WARPS_PER_CTA: usize = 8;
const THREADS: u32 = (WARPS_PER_CTA * 32) as u32;

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

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared Q/K entry for an exact row count.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait MtpQkPrepareRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares this route's exact-width entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's normalization, MRoPE, and cache-append entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `MtpBf16QkPrepareOp::launch`'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's MTP Q/K routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 here never widens the artifact-level
/// admission bound. Each table names only the entries its own model emits,
/// which keeps the compiled inventory fixed while both prepared owners share
/// one wrapper.
pub trait MtpQkPrepareEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for `B=1..=8`.
    type Decode<const TOKENS: usize>: MtpQkPrepareRoute<A>;
    /// Prepared prefill route for `T=32,64,128`.
    type Prefill<const TOKENS: usize>: MtpQkPrepareRoute<A>;
    /// Prepared `T=1024` prefill route, unadmitted outside Qwen3.8.
    type Prefill1024: MtpQkPrepareRoute<A>;

    /// Whether `T=1024` is an admitted prefill row count.
    const HAS_T1024: bool;
    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not cover.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8 decode entry for one exact batch.
pub struct PreparedRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__mtp_bf16_qk_prepare_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.8 prefill entry for one exact prompt tile.
pub struct PreparedPrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__mtp_bf16_qk_prepare_prefill_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 decode entry for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_mtp_bf16_qk_prepare_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 prefill entry for one exact prompt tile.
pub struct PreparedQwen35PrefillRoute<const TOKENS: usize> {
    prepare: PreparedLaunch<kernels::__qwen35_mtp_bf16_qk_prepare_prefill_CudaKernel<TOKENS>>,
}

/// Stands in for a prefill width an architecture does not admit.
///
/// It prepares and launches no entry, so an unadmitted width can never reach
/// the device and never enters the emitted inventory.
pub struct UnadmittedQkPrepareRoute;

impl<const TOKENS: usize> private::Sealed for PreparedRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedPrefillRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35Route<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35PrefillRoute<TOKENS> {}
impl private::Sealed for UnadmittedQkPrepareRoute {}

impl<const TOKENS: usize> MtpQkPrepareRoute<Qwen35_9B> for PreparedQwen35Route<TOKENS> {
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

impl<const TOKENS: usize> MtpQkPrepareRoute<Qwen35_9B> for PreparedQwen35PrefillRoute<TOKENS> {
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

// The Qwen3.8 entries compile that model's head counts into concrete symbols,
// so these routes stay bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> MtpQkPrepareRoute<A> for PreparedPrefillRoute<TOKENS> {
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

impl<A: Sm120Arch, const TOKENS: usize> MtpQkPrepareRoute<A> for PreparedRoute<TOKENS> {
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

impl<A: Arch> MtpQkPrepareRoute<A> for UnadmittedQkPrepareRoute {
    fn prepare(_module: &kernels::LoadedModule) -> GpuResult<Self> {
        Ok(Self)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        _module: &kernels::LoadedModule,
        _stream: &CudaStream,
        _qkv: *const u16,
        _query_norm: *const u16,
        _key_norm: *const u16,
        _rope_cos: *const f32,
        _rope_sin: *const f32,
        _block_tables: *const u32,
        _table_rows: *const u32,
        _table_stride: u32,
        _cache_positions: *const u32,
        _query: *mut f32,
        _key_pages: *mut u16,
        _value_pages: *mut u16,
    ) -> GpuResult<()> {
        Err(unadmitted_route())
    }
}

// The derived table rejects an unadmitted width before dispatch, so this is
// the defensive tail of a route that owns no entry.
fn unadmitted_route() -> GpuError {
    GpuError::invalid_launch("MTP BF16 Q/K route is not admitted for this architecture")
}

/// Qwen3.8 entry table: decode `B=1..=8` and prefill through `T=1024`.
pub struct Qwen38MtpQkPrepareEntries;

/// Qwen3.5 entry table: decode `B=1..=8` and prefill through `T=128`.
pub struct Qwen35MtpQkPrepareEntries;

impl private::Sealed for Qwen38MtpQkPrepareEntries {}
impl private::Sealed for Qwen35MtpQkPrepareEntries {}

impl<A: Sm120Arch> MtpQkPrepareEntries<A> for Qwen38MtpQkPrepareEntries {
    type Decode<const TOKENS: usize> = PreparedRoute<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedPrefillRoute<TOKENS>;
    type Prefill1024 = PreparedPrefillRoute<1_024>;

    const HAS_T1024: bool = true;
    const LABEL: &'static str = "";
    const MODULE_OPERATION: &'static str = "loading the MTP BF16 Q/K module";

    fn require_geometry() -> GpuResult<()> {
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
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        mtp_bf16_qk_prepare_ptx_names()
            .into_iter()
            .chain(mtp_bf16_qk_prepare_prefill_ptx_names())
            .collect()
    }
}

impl MtpQkPrepareEntries<Qwen35_9B> for Qwen35MtpQkPrepareEntries {
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;
    type Prefill<const TOKENS: usize> = PreparedQwen35PrefillRoute<TOKENS>;
    type Prefill1024 = UnadmittedQkPrepareRoute;

    const HAS_T1024: bool = false;
    const LABEL: &'static str = "Qwen3.5 ";
    const MODULE_OPERATION: &'static str = "loading the Qwen3.5 MTP BF16 Q/K module";

    fn require_geometry() -> GpuResult<()> {
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
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_mtp_bf16_qk_prepare_ptx_names().to_vec()
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

fn unsupported_rows<A: Arch, E: MtpQkPrepareEntries<A>>(rows: usize) -> GpuError {
    let admitted = if E::HAS_T1024 {
        format!("{PREFILL_ROUTES:?}")
    } else {
        format!("{QWEN35_PREFILL_ROUTES:?}")
    };
    GpuError::invalid_launch(format!(
        "{}MTP BF16 Q/K rows {rows} are outside exact B=1..={MAX_BATCH} or T={admitted}",
        E::LABEL
    ))
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_mtp_bf16_qk_prepare),
    required(1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128),
    inventory(false)
)]
struct MtpBf16QkPrepareRoutes<A: Arch, E: MtpQkPrepareEntries<A>> {
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

/// Prepared MTP Q/K normalization, MRoPE, and BF16 cache routes for exact
/// decode `B=1..=8` and the entry table's admitted prefill widths.
pub struct MtpBf16QkPrepareOp<
    A: Arch = Qwen38_27B,
    E: MtpQkPrepareEntries<A> = Qwen38MtpQkPrepareEntries,
> {
    module: kernels::LoadedModule,
    routes: MtpBf16QkPrepareRoutes<A, E>,
}

/// Prepared Qwen3.5 MTP Q/K normalization, MRoPE, and BF16 cache routes.
pub type Qwen35MtpBf16QkPrepareOp = MtpBf16QkPrepareOp<Qwen35_9B, Qwen35MtpQkPrepareEntries>;

impl<A: Arch, E: MtpQkPrepareEntries<A>> MtpBf16QkPrepareOp<A, E> {
    /// Loads the embedded module and prepares every admitted route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module(E::MODULE_OPERATION, source))?;

        let routes = MtpBf16QkPrepareRoutes::prepare(&module)?;

        Ok(Self { module, routes })
    }

    /// Normalizes and rotates Q/K, then appends represented BF16 K/V values.
    ///
    /// # Safety
    ///
    /// `qkv` covers `[rows, A::ATTENTION_QKV_ROWS]` BF16 values in query/gate,
    /// key, value order. Norms cover 256 values; rotary planes cover
    /// `[rows, 32]`; metadata covers `rows`; and each selected block-table row
    /// covers its cache position. Query covers
    /// `[rows, A::NUM_ATTENTION_HEADS, 256]` FP32 values. Cache planes use
    /// page-major `[physical_page, 4, 64, 256]` BF16 values. Allocations are
    /// aligned, non-overlapping, context-local, and live through stream
    /// completion.
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
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch(format!("{}MTP BF16 Q/K table stride exceeds u32", E::LABEL))
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(format!(
                "{}MTP BF16 Q/K table stride must be nonzero",
                E::LABEL
            )));
        }

        dispatch_mtp_bf16_qk_prepare!(
            &self.routes,
            rows,
            |route| unsafe {
                route.launch(
                    &self.module, stream, qkv, query_norm, key_norm, rope_cos, rope_sin,
                    block_tables, table_rows, table_stride, cache_positions, query, key_pages,
                    value_pages,
                )
            },
            else => Err(unsupported_rows::<A, E>(rows))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BATCH, MtpBf16QkPrepareRoutes, MtpQkPrepareEntries, PREFILL_ROUTES,
        QWEN35_PREFILL_ROUTES, Qwen35MtpQkPrepareEntries, Qwen38MtpQkPrepareEntries, THREADS,
        mtp_bf16_qk_prepare_prefill_ptx_names, mtp_bf16_qk_prepare_ptx_names,
        qwen35_mtp_bf16_qk_prepare_ptx_names, unsupported_rows,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    /// The decode and prefill widths both admitted architectures route.
    const SHARED_SCHEDULE: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];

    /// The derive's ordered admission inventory for one entry table.
    fn admitted_schedule<A: Arch, E: MtpQkPrepareEntries<A>>() -> Vec<usize> {
        MtpBf16QkPrepareRoutes::<A, E>::admitted_rows()
    }

    #[test]
    fn exact_batch_inventory_is_complete_and_unique() {
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

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38MtpQkPrepareEntries as MtpQkPrepareEntries<Qwen38_27B>>::ptx_names(),
            mtp_bf16_qk_prepare_ptx_names()
                .into_iter()
                .chain(mtp_bf16_qk_prepare_prefill_ptx_names())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            <Qwen35MtpQkPrepareEntries as MtpQkPrepareEntries<Qwen35_9B>>::ptx_names(),
            qwen35_mtp_bf16_qk_prepare_ptx_names().to_vec()
        );
    }

    /// The merged schedule, checked against the two dispatches it replaces:
    /// Qwen3.5 stops at `T=128` and only Qwen3.8 admits `T=1024`.
    #[test]
    fn row_routing_is_exact_for_every_admitted_architecture() {
        let qwen38 = SHARED_SCHEDULE
            .iter()
            .copied()
            .chain([1_024])
            .collect::<Vec<_>>();

        assert_eq!(
            admitted_schedule::<Qwen38_27B, Qwen38MtpQkPrepareEntries>(),
            qwen38
        );
        assert_eq!(
            admitted_schedule::<Qwen35_9B, Qwen35MtpQkPrepareEntries>(),
            SHARED_SCHEDULE.to_vec()
        );
    }

    /// An unadmitted row count keeps its owner's rejection wording.
    #[test]
    fn unadmitted_row_counts_name_their_architecture() {
        for (message, error) in [
            (
                "MTP BF16 Q/K rows 9 are outside exact B=1..=8 or T=[32, 64, 128, 1024]",
                unsupported_rows::<Qwen38_27B, Qwen38MtpQkPrepareEntries>(9),
            ),
            (
                "Qwen3.5 MTP BF16 Q/K rows 1024 are outside exact B=1..=8 or T=[32, 64, 128]",
                unsupported_rows::<Qwen35_9B, Qwen35MtpQkPrepareEntries>(1_024),
            ),
        ] {
            assert!(
                error.to_string().ends_with(message),
                "{error} does not end with {message}"
            );
        }
    }
}

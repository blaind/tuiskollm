//! Exact decode and early-context prefill paged grouped-query attention.

use crate::Sm120Arch;
use crate::device::paged_gqa::{
    PREFILL_PARTIAL_VALUES, PREFILL_SHARED_BYTES, PREFILL_THREADS, paged_gqa,
    paged_gqa_prefill_partitioned, paged_gqa_prefill_partitioned_reduce, paged_gqa_prefill_shared,
};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const THREADS: u32 = 32;
const PREFILL_SHARED_BYTES_U32: u32 = PREFILL_SHARED_BYTES as u32;
/// First context length routed to the sixteen-partition T=128 schedule.
pub const PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT: usize = 32_769;
/// Largest admitted T=128 prefill context.
pub const PAGED_GQA_PREFILL_MAX_CONTEXT: usize = 220_000;
/// Maximum resident FP32 partial workspace for partitioned T=128 prefill.
pub const PAGED_GQA_PREFILL_PARTIAL_BYTES: usize =
    128 * 24 * 16 * PREFILL_PARTIAL_VALUES * size_of::<f32>();

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

fn require_geometry<A: Arch>() -> GpuResult<()> {
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

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Applies paged FP8 GQA for one exact decode batch.
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
        // One lane owns eight of the 256 dimensions, so one warp owns one
        // query head without cross-CTA partials. Keeping one warp per CTA gives
        // B=1 twenty-four independent CTAs instead of three eight-warp CTAs;
        // sixteen-CTA launch bounds retain the short-context occupancy target.
        unsafe {
            paged_gqa::<A, TOKENS>(
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
            paged_gqa_prefill_shared::<A, TOKENS>(
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

    /// Produces FP32 online-softmax states for one exact T=128 partition route.
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
    pub fn paged_gqa_prefill_partitioned_exact<A: Arch, const PARTITIONS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partials: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) {
        // Eight partitions expose 2,048 CTAs below 32,769 positions; sixteen
        // expose 4,096 above it. Each CTA retains the two-token/six-head GQA
        // reuse topology and a 32-KiB K/V tile while bounding serial prefix work.
        unsafe {
            paged_gqa_prefill_partitioned::<A, 128, PARTITIONS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
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
                .prepare_paged_gqa_exact::<A, TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
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

struct PreparedPartitionedPrefillRoute<A: Arch, const PARTITIONS: usize> {
    partial:
        PreparedLaunch<kernels::__paged_gqa_prefill_partitioned_exact_CudaKernel<A, PARTITIONS>>,
    reduce: PreparedLaunch<
        kernels::__paged_gqa_prefill_partitioned_reduce_exact_CudaKernel<A, PARTITIONS>,
    >,
}

impl<A: Arch, const PARTITIONS: usize> PreparedPartitionedPrefillRoute<A, PARTITIONS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let partial_blocks = u32::try_from(128 / 2 * A::NUM_KV_HEADS * PARTITIONS)
            .map_err(|_| GpuError::invalid_launch("partitioned paged GQA grid exceeds u32"))?;
        let reduce_blocks = u32::try_from(128 * A::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("paged GQA reduction grid exceeds u32"))?;

        Ok(Self {
            partial: module
                .prepare_paged_gqa_prefill_partitioned_exact::<A, PARTITIONS>(LaunchConfig1D::new(
                    partial_blocks,
                    PREFILL_THREADS as u32,
                    PREFILL_SHARED_BYTES_U32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing partitioned paged GQA prefill", source)
                })?,
            reduce: module
                .prepare_paged_gqa_prefill_partitioned_reduce_exact::<A, PARTITIONS>(
                    LaunchConfig1D::new(reduce_blocks, THREADS, 0),
                )
                .map_err(|source| {
                    GpuError::launch("preparing partitioned paged GQA reduction", source)
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
            .paged_gqa_prefill_partitioned_exact::<A, PARTITIONS>(
                stream,
                &self.partial,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partials,
                key_scale,
                value_scale,
            )
            .map_err(|source| {
                GpuError::launch("launching partitioned paged GQA prefill", source)
            })?;
        module
            .paged_gqa_prefill_partitioned_reduce_exact::<A, PARTITIONS>(
                stream,
                &self.reduce,
                partials,
                output,
            )
            .map_err(|source| GpuError::launch("launching partitioned paged GQA reduction", source))
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
    p8: PreparedPartitionedPrefillRoute<A, 8>,
    p16: PreparedPartitionedPrefillRoute<A, 16>,
}

impl<A: Sm120Arch> PagedGqaOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
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
            p8: PreparedPartitionedPrefillRoute::prepare(&module)?,
            p16: PreparedPartitionedPrefillRoute::prepare(&module)?,
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
        kernels::paged_gqa_prefill_partitioned_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::paged_gqa_prefill_partitioned_exact_ptx_name::<Qwen38_27B, 16>(),
        kernels::paged_gqa_prefill_partitioned_reduce_exact_ptx_name::<Qwen38_27B, 16>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PAGED_GQA_PREFILL_PARTIAL_BYTES,
        PREFILL_SHARED_BYTES, PREFILL_THREADS, THREADS, admitted_batch,
        paged_gqa_prefill_partitions, paged_gqa_ptx_names,
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

        assert_eq!(names.len(), 15);
        assert_eq!(unique.len(), names.len());
        assert_eq!(PREFILL_THREADS, 384);
        assert_eq!(PREFILL_SHARED_BYTES, 32_768);
        assert_eq!(PAGED_GQA_PREFILL_PARTIAL_BYTES, 50_724_864);
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
}

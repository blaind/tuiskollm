//! Exact-K long-context MTP attention with represented-cache tile reuse.

use crate::device::paged_gqa::{
    LONG_CONTEXT_MAX_PARTITIONS, LONG_CONTEXT_MAX_TOKENS, LONG_CONTEXT_MTP_SHARED_BYTES,
    LONG_CONTEXT_MTP_THREADS, LONG_CONTEXT_PARTITION_SIZE, long_context_mtp_paged_gqa_partial,
    long_context_paged_gqa_reduce,
};
use crate::long_context_paged_gqa::LONG_CONTEXT_GQA_PARTITION_BUCKETS;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_common::Sm120Arch;
use tuisko_model::{Arch, Qwen38_27B};

const THREADS: u32 = LONG_CONTEXT_MTP_THREADS as u32;
const SHARED_BYTES: u32 = LONG_CONTEXT_MTP_SHARED_BYTES as u32;
const REDUCTION_THREADS: u32 = 32;
const REDUCTION_SHARED_BYTES: u32 = (LONG_CONTEXT_MAX_PARTITIONS * size_of::<f32>()) as u32;

fn partition_bucket(maximum_length: usize) -> Option<usize> {
    if !(1..=LONG_CONTEXT_MAX_TOKENS).contains(&maximum_length) {
        return None;
    }
    let required = maximum_length.div_ceil(LONG_CONTEXT_PARTITION_SIZE);

    LONG_CONTEXT_GQA_PARTITION_BUCKETS
        .iter()
        .copied()
        .find(|&partitions| partitions >= required)
}

fn require_geometry<A: Arch>() -> GpuResult<()> {
    if A::NUM_ATTENTION_HEADS != 24
        || A::NUM_KV_HEADS != 4
        || A::HEAD_DIM != 256
        || A::ATTENTION_OUTPUT_COLUMNS != 6_144
    {
        return Err(GpuError::invalid_launch(
            "architecture geometry is incompatible with the admitted long-context MTP GQA schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Uses native FP8 QK while reusing each represented K/V tile across provisional rows.
    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        dynamic_shared = 47232,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn long_context_mtp_paged_gqa_partial_exact<A: Arch, const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        key_scale: f32,
        value_scale: f32,
        launched_partitions: u32,
    ) {
        // One CTA owns a KV head and partition. Eight warps cover up to
        // K=4 x six GQA rows while the CTA loads represented cache bytes once.
        unsafe {
            long_context_mtp_paged_gqa_partial::<A, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                key_scale,
                value_scale,
                launched_partitions,
            );
        }
    }

    /// Reduces the unchanged FP32 partition statistics for every provisional row.
    #[kernel]
    #[launch_bounds(32, 16)]
    #[allow(clippy::too_many_arguments)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 3440,
        dynamic_shared_alignment = 16,
        min_compute_capability = (12, 0),
    )]
    pub fn long_context_mtp_paged_gqa_reduce_exact<A: Arch, const TOKENS: usize>(
        lengths: *const u32,
        partial_maximum: *const f32,
        partial_denominator: *const f32,
        partial_numerator: *const f32,
        output: *mut f32,
    ) {
        unsafe {
            long_context_paged_gqa_reduce::<A, TOKENS>(
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                output,
            );
        }
    }
}

type PartialKernel<A, const TOKENS: usize> =
    kernels::__long_context_mtp_paged_gqa_partial_exact_CudaKernel<A, TOKENS>;

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    p4: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p16: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p64: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p256: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p512: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p860: PreparedLaunch<PartialKernel<A, TOKENS>>,
    reduction:
        PreparedLaunch<kernels::__long_context_mtp_paged_gqa_reduce_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare_partial(
        module: &kernels::LoadedModule,
        partitions: usize,
    ) -> GpuResult<PreparedLaunch<PartialKernel<A, TOKENS>>> {
        let blocks = u32::try_from(A::NUM_KV_HEADS * partitions)
            .map_err(|_| GpuError::invalid_launch("long-context MTP paged GQA grid exceeds u32"))?;

        module
            .prepare_long_context_mtp_paged_gqa_partial_exact::<A, TOKENS>(LaunchConfig1D::new(
                blocks,
                THREADS,
                SHARED_BYTES,
            ))
            .map_err(|source| {
                GpuError::launch("preparing long-context MTP paged GQA partial route", source)
            })
    }

    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let reduction_blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS).map_err(|_| {
            GpuError::invalid_launch("long-context MTP paged GQA reduction grid exceeds u32")
        })?;

        Ok(Self {
            p4: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[0])?,
            p16: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[1])?,
            p64: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[2])?,
            p256: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[3])?,
            p512: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[4])?,
            p860: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[5])?,
            reduction: module
                .prepare_long_context_mtp_paged_gqa_reduce_exact::<A, TOKENS>(LaunchConfig1D::new(
                    reduction_blocks,
                    REDUCTION_THREADS,
                    REDUCTION_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch(
                        "preparing long-context MTP paged GQA reduction route",
                        source,
                    )
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        partitions: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        let partial = match partitions {
            4 => &self.p4,
            16 => &self.p16,
            64 => &self.p64,
            256 => &self.p256,
            512 => &self.p512,
            LONG_CONTEXT_MAX_PARTITIONS => &self.p860,
            _ => unreachable!(),
        };
        let partitions = u32::try_from(partitions).expect("partition bucket fits u32");

        module
            .long_context_mtp_paged_gqa_partial_exact::<A, TOKENS>(
                stream,
                partial,
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                key_scale,
                value_scale,
                partitions,
            )
            .map_err(|source| {
                GpuError::launch("launching long-context MTP paged GQA partials", source)
            })?;
        module
            .long_context_mtp_paged_gqa_reduce_exact::<A, TOKENS>(
                stream,
                &self.reduction,
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching long-context MTP paged GQA reduction", source)
            })
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_long_context_mtp_paged_gqa),
    required(2, 3, 4),
    inventory(false)
)]
struct LongContextMtpPagedGqaRoutes<A: Arch> {
    #[route(2)]
    k2: PreparedRoute<A, 2>,
    #[route(3)]
    k3: PreparedRoute<A, 3>,
    #[route(4)]
    k4: PreparedRoute<A, 4>,
}

/// Prepared long-context target-MTP GQA routes for exact `K=2..4`.
pub struct LongContextMtpPagedGqaOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    routes: LongContextMtpPagedGqaRoutes<A>,
}

impl<A: Sm120Arch> LongContextMtpPagedGqaOp<A> {
    /// Loads the embedded module and prepares every exact target-MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = long_context_mtp_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading long-context MTP paged GQA", source))?;

        Ok(Self {
            routes: LongContextMtpPagedGqaRoutes::prepare(&module)?,
            module,
        })
    }

    /// Applies partitioned GQA while sharing each represented K/V tile across K rows.
    ///
    /// # Safety
    ///
    /// Query and output cover `[tokens, 24, 256]` FP32 values for exact
    /// `tokens=2..4`. Cache planes use `[physical_page, 4, 64, 256]` E4M3
    /// bytes. Every table-row entry selects the same resident table row and
    /// lengths are consecutive, ascending positions for one slot. Lengths are
    /// nonzero and no greater than `maximum_length`; the common table row covers
    /// `maximum_length` rounded up to 64. Partial maximum and denominator cover
    /// `[tokens, 24, 860]` FP32 values and partial numerator covers
    /// `[tokens, 24, 860, 256]`. All allocations are aligned, non-overlapping,
    /// live through completion, and belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        tokens: usize,
        maximum_length: usize,
        query: *const f32,
        key_pages: *const u8,
        value_pages: *const u8,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        output: *mut f32,
        key_scale: f32,
        value_scale: f32,
    ) -> GpuResult<()> {
        if !LongContextMtpPagedGqaRoutes::<A>::contains(tokens) {
            return Err(GpuError::invalid_launch(format!(
                "long-context MTP paged GQA K={tokens} is outside the admitted range 2..=4"
            )));
        }
        let partitions = partition_bucket(maximum_length).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "long-context MTP paged GQA maximum length {maximum_length} is outside 1..={LONG_CONTEXT_MAX_TOKENS}"
            ))
        })?;
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("long-context MTP paged GQA table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "long-context MTP paged GQA table stride must be nonzero",
            ));
        }
        if !key_scale.is_finite() || key_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "long-context MTP paged GQA key scale must be finite and positive",
            ));
        }
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "long-context MTP paged GQA value scale must be finite and positive",
            ));
        }

        dispatch_long_context_mtp_paged_gqa!(
            &self.routes,
            tokens,
            |route| unsafe {
                route.launch(
                    &self.module,
                    stream,
                    partitions,
                    query,
                    key_pages,
                    value_pages,
                    block_tables,
                    table_rows,
                    table_stride,
                    lengths,
                    partial_maximum,
                    partial_denominator,
                    partial_numerator,
                    output,
                    key_scale,
                    value_scale,
                )
            },
            else => unreachable!()
        )
    }
}

/// PTX symbols retained for both stages of every exact target-MTP route.
pub(crate) fn long_context_mtp_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::long_context_mtp_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::long_context_mtp_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::long_context_mtp_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::long_context_mtp_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::long_context_mtp_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::long_context_mtp_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 4>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        LONG_CONTEXT_MTP_SHARED_BYTES, LONG_CONTEXT_MTP_THREADS, LongContextMtpPagedGqaRoutes,
        long_context_mtp_paged_gqa_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn target_mtp_routes_and_tile_resources_are_exact() {
        assert_eq!(
            LongContextMtpPagedGqaRoutes::<tuisko_model::Qwen38_27B>::admitted_rows(),
            vec![2, 3, 4]
        );
        assert_eq!(LONG_CONTEXT_MTP_THREADS, 256);
        assert_eq!(LONG_CONTEXT_MTP_SHARED_BYTES, 47_232);
    }

    #[test]
    fn ptx_inventory_has_two_stages_per_route() {
        let names = long_context_mtp_paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 6);
        assert_eq!(unique.len(), names.len());
    }
}

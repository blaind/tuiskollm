//! Cross-CTA represented-BF16 paged GQA for long-context MTP decode.

use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_macros::ExactRoutes;
use tuisko_kernels_sm120_attention::shared_device::{
    long_context_bf16_paged_gqa_partial, long_context_paged_gqa_reduce,
};
use tuisko_kernels_sm120_attention::{
    LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
    LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const THREADS: u32 = 32;
const REDUCTION_SHARED_BYTES: u32 = (LONG_CONTEXT_GQA_MAX_PARTITIONS * size_of::<f32>()) as u32;

/// FP32 scratch bytes retained per exact token row.
pub const MTP_BF16_SPLIT_KV_WORKSPACE_BYTES_PER_TOKEN: usize = Qwen38_27B::NUM_ATTENTION_HEADS
    * LONG_CONTEXT_GQA_MAX_PARTITIONS
    * (2 + Qwen38_27B::HEAD_DIM)
    * size_of::<f32>();

fn partition_bucket(maximum_length: usize) -> Option<usize> {
    if !(1..=LONG_CONTEXT_GQA_MAX_TOKENS).contains(&maximum_length) {
        return None;
    }
    let required = maximum_length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);

    LONG_CONTEXT_GQA_PARTITION_BUCKETS
        .iter()
        .copied()
        .find(|&partitions| partitions >= required)
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Produces one stable BF16 attention partial per 256-position partition.
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
    pub fn mtp_bf16_split_kv_paged_gqa_partial<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        launched_partitions: u32,
    ) {
        unsafe {
            long_context_bf16_paged_gqa_partial::<Qwen38_27B, TOKENS>(
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
                launched_partitions,
            );
        }
    }

    /// Reduces partition statistics into one exact output per query head.
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
    pub fn mtp_bf16_split_kv_paged_gqa_reduce<const TOKENS: usize>(
        lengths: *const u32,
        partial_maximum: *const f32,
        partial_denominator: *const f32,
        partial_numerator: *const f32,
        output: *mut f32,
    ) {
        unsafe {
            long_context_paged_gqa_reduce::<Qwen38_27B, TOKENS>(
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                output,
            );
        }
    }
}

type PartialKernel<const TOKENS: usize> =
    kernels::__mtp_bf16_split_kv_paged_gqa_partial_CudaKernel<TOKENS>;

struct PreparedRoute<const TOKENS: usize> {
    p4: PreparedLaunch<PartialKernel<TOKENS>>,
    p16: PreparedLaunch<PartialKernel<TOKENS>>,
    p64: PreparedLaunch<PartialKernel<TOKENS>>,
    p256: PreparedLaunch<PartialKernel<TOKENS>>,
    p512: PreparedLaunch<PartialKernel<TOKENS>>,
    p860: PreparedLaunch<PartialKernel<TOKENS>>,
    reduction: PreparedLaunch<kernels::__mtp_bf16_split_kv_paged_gqa_reduce_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare_partial(
        module: &kernels::LoadedModule,
        partitions: usize,
    ) -> GpuResult<PreparedLaunch<PartialKernel<TOKENS>>> {
        let blocks = u32::try_from(TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS * partitions)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 split-KV partial grid exceeds u32"))?;

        module
            .prepare_mtp_bf16_split_kv_paged_gqa_partial::<TOKENS>(LaunchConfig1D::new(
                blocks, THREADS, 0,
            ))
            .map_err(|source| GpuError::launch("preparing MTP BF16 split-KV partial route", source))
    }

    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let reduction_blocks =
            u32::try_from(TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS).map_err(|_| {
                GpuError::invalid_launch("MTP BF16 split-KV reduction grid exceeds u32")
            })?;

        Ok(Self {
            p4: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[0])?,
            p16: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[1])?,
            p64: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[2])?,
            p256: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[3])?,
            p512: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[4])?,
            p860: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[5])?,
            reduction: module
                .prepare_mtp_bf16_split_kv_paged_gqa_reduce::<TOKENS>(LaunchConfig1D::new(
                    reduction_blocks,
                    THREADS,
                    REDUCTION_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing MTP BF16 split-KV reduction route", source)
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
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        output: *mut f32,
    ) -> GpuResult<()> {
        let partial = match partitions {
            4 => &self.p4,
            16 => &self.p16,
            64 => &self.p64,
            256 => &self.p256,
            512 => &self.p512,
            LONG_CONTEXT_GQA_MAX_PARTITIONS => &self.p860,
            _ => unreachable!(),
        };
        let partitions = u32::try_from(partitions).expect("partition bucket fits u32");

        module
            .mtp_bf16_split_kv_paged_gqa_partial::<TOKENS>(
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
                partitions,
            )
            .map_err(|source| GpuError::launch("launching MTP BF16 split-KV partials", source))?;
        module
            .mtp_bf16_split_kv_paged_gqa_reduce::<TOKENS>(
                stream,
                &self.reduction,
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                output,
            )
            .map_err(|source| GpuError::launch("launching MTP BF16 split-KV reduction", source))
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(kernels::LoadedModule),
    error(GpuError),
    dispatch(dispatch_mtp_bf16_split_kv_paged_gqa),
    required(1, 2, 3, 4, 5, 6, 7, 8),
    inventory(false)
)]
struct MtpBf16SplitKvPagedGqaRoutes {
    #[route(1)]
    b1: PreparedRoute<1>,
    #[route(2)]
    b2: PreparedRoute<2>,
    #[route(3)]
    b3: PreparedRoute<3>,
    #[route(4)]
    b4: PreparedRoute<4>,
    #[route(5)]
    b5: PreparedRoute<5>,
    #[route(6)]
    b6: PreparedRoute<6>,
    #[route(7)]
    b7: PreparedRoute<7>,
    #[route(8)]
    b8: PreparedRoute<8>,
}

/// Prepared production split-KV route for exact MTP decode `B=1..=8`.
pub struct MtpBf16SplitKvPagedGqaOp {
    module: kernels::LoadedModule,
    routes: MtpBf16SplitKvPagedGqaRoutes,
}

impl MtpBf16SplitKvPagedGqaOp {
    /// Loads the embedded module and prepares every exact route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        let _ = mtp_bf16_split_kv_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading MTP BF16 split-KV paged GQA", source))?;

        Ok(Self {
            routes: MtpBf16SplitKvPagedGqaRoutes::prepare(&module)?,
            module,
        })
    }

    /// Applies cross-CTA online-softmax GQA over represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, 24, 256]` FP32 values. Cache planes use
    /// `[physical_page, 4, 64, 256]` BF16 values. Metadata covers `batch`; every
    /// length is nonzero and no greater than `maximum_length`, its table row
    /// covers that length rounded up to 64, and every physical page is resident.
    /// Maximum and denominator scratch each cover `[batch, 24, 860]`; numerator
    /// scratch covers `[batch, 24, 860, 256]`. Allocations are aligned, disjoint,
    /// live through completion, and belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        maximum_length: usize,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        partial_maximum: *mut f32,
        partial_denominator: *mut f32,
        partial_numerator: *mut f32,
        output: *mut f32,
    ) -> GpuResult<()> {
        if !MtpBf16SplitKvPagedGqaRoutes::contains(batch) {
            return Err(GpuError::invalid_launch(format!(
                "MTP BF16 split-KV paged GQA batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }
        let partitions = partition_bucket(maximum_length).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "MTP BF16 split-KV paged GQA maximum length {maximum_length} is outside 1..={LONG_CONTEXT_GQA_MAX_TOKENS}"
            ))
        })?;
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("MTP BF16 split-KV paged GQA table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "MTP BF16 split-KV paged GQA table stride must be nonzero",
            ));
        }

        dispatch_mtp_bf16_split_kv_paged_gqa!(
            &self.routes,
            batch,
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
                )
            },
            else => unreachable!()
        )
    }
}

pub(crate) fn mtp_bf16_split_kv_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<1>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<1>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<2>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<2>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<3>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<3>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<4>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<4>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<5>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<5>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<6>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<6>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<7>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<7>(),
        kernels::mtp_bf16_split_kv_paged_gqa_partial_ptx_name::<8>(),
        kernels::mtp_bf16_split_kv_paged_gqa_reduce_ptx_name::<8>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        MTP_BF16_SPLIT_KV_WORKSPACE_BYTES_PER_TOKEN, MtpBf16SplitKvPagedGqaRoutes,
        REDUCTION_SHARED_BYTES, mtp_bf16_split_kv_paged_gqa_ptx_names, partition_bucket,
    };
    use std::collections::BTreeSet;

    #[test]
    fn exact_routes_and_partition_buckets_are_pinned() {
        assert_eq!(
            MtpBf16SplitKvPagedGqaRoutes::admitted_rows(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(partition_bucket(0), None);
        assert_eq!(partition_bucket(1), Some(4));
        assert_eq!(partition_bucket(1_024), Some(4));
        assert_eq!(partition_bucket(1_025), Some(16));
        assert_eq!(partition_bucket(65_536), Some(256));
        assert_eq!(partition_bucket(65_537), Some(512));
        assert_eq!(partition_bucket(131_072), Some(512));
        assert_eq!(partition_bucket(131_073), Some(860));
        assert_eq!(partition_bucket(220_000), Some(860));
        assert_eq!(partition_bucket(220_001), None);
        assert_eq!(REDUCTION_SHARED_BYTES, 3_440);
        assert_eq!(MTP_BF16_SPLIT_KV_WORKSPACE_BYTES_PER_TOKEN, 21_300_480);
    }

    #[test]
    fn inventory_has_two_stages_per_exact_batch() {
        let names = mtp_bf16_split_kv_paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 16);
        assert_eq!(unique.len(), names.len());
    }
}

//! Exact-batch partitioned long-context paged grouped-query attention.

use crate::Sm120Arch;
use crate::device::paged_gqa::{
    LONG_CONTEXT_MAX_PARTITIONS, LONG_CONTEXT_MAX_TOKENS, LONG_CONTEXT_PARTITION_SIZE,
    long_context_paged_gqa_partial, long_context_paged_gqa_reduce,
};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
// One lane owns eight of the 256 dimensions; sixteen one-warp CTAs per SM
// bound register growth and preserve 25% warp occupancy when the grid supplies it.
const THREADS: u32 = 32;
const REDUCTION_SHARED_BYTES: u32 = (LONG_CONTEXT_MAX_PARTITIONS * size_of::<f32>()) as u32;
// These six captured grids cap excess partial CTAs at 4x while covering
// 1K, 4K, 16K, 65K, 131K, and the exact 220K position ceiling.
/// Prepared partial-grid widths spanning the complete context ceiling.
pub const LONG_CONTEXT_GQA_PARTITION_BUCKETS: [usize; 6] =
    [4, 16, 64, 256, 512, LONG_CONTEXT_MAX_PARTITIONS];

/// Maximum context admitted by the partitioned decode route.
pub const LONG_CONTEXT_GQA_MAX_TOKENS: usize = LONG_CONTEXT_MAX_TOKENS;
/// Positions scanned by one partial-attention CTA.
pub const LONG_CONTEXT_GQA_PARTITION_SIZE: usize = LONG_CONTEXT_PARTITION_SIZE;
/// Maximum partials retained per token and query head.
pub const LONG_CONTEXT_GQA_MAX_PARTITIONS: usize = LONG_CONTEXT_MAX_PARTITIONS;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

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
            "architecture geometry is incompatible with the admitted long-context paged GQA schedule",
        ));
    }

    Ok(())
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Produces stable online-softmax statistics for one 256-position partition.
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
    pub fn long_context_paged_gqa_partial_exact<A: Arch, const TOKENS: usize>(
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
        // A warp retains all 256 output dimensions as eight values per lane.
        // A 256-position partition exposes 860 CTAs at 220K/B=1, or 5.1
        // CTAs per SM, while capping the B=8 FP32 workspace at 162.5 MiB.
        unsafe {
            long_context_paged_gqa_partial::<A, TOKENS>(
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
    pub fn long_context_paged_gqa_reduce_exact<A: Arch, const TOKENS: usize>(
        lengths: *const u32,
        partial_maximum: *const f32,
        partial_denominator: *const f32,
        partial_numerator: *const f32,
        output: *mut f32,
    ) {
        // One warp maps the 256 output dimensions eight-per-lane. The 860
        // partition weights occupy 3,440 shared bytes and avoid 256-fold exp.
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
    kernels::__long_context_paged_gqa_partial_exact_CudaKernel<A, TOKENS>;

struct PreparedRoute<A: Arch, const TOKENS: usize> {
    p4: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p16: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p64: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p256: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p512: PreparedLaunch<PartialKernel<A, TOKENS>>,
    p860: PreparedLaunch<PartialKernel<A, TOKENS>>,
    reduction: PreparedLaunch<kernels::__long_context_paged_gqa_reduce_exact_CudaKernel<A, TOKENS>>,
}

impl<A: Arch, const TOKENS: usize> PreparedRoute<A, TOKENS> {
    fn prepare_partial(
        module: &kernels::LoadedModule,
        partitions: usize,
    ) -> GpuResult<PreparedLaunch<PartialKernel<A, TOKENS>>> {
        let blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS * partitions)
            .map_err(|_| GpuError::invalid_launch("long-context paged GQA grid exceeds u32"))?;

        module
            .prepare_long_context_paged_gqa_partial_exact::<A, TOKENS>(LaunchConfig1D::new(
                blocks, THREADS, 0,
            ))
            .map_err(|source| {
                GpuError::launch("preparing long-context paged GQA partial route", source)
            })
    }

    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let reduction_blocks = u32::try_from(TOKENS * A::NUM_ATTENTION_HEADS).map_err(|_| {
            GpuError::invalid_launch("long-context paged GQA reduction grid exceeds u32")
        })?;

        Ok(Self {
            p4: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[0])?,
            p16: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[1])?,
            p64: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[2])?,
            p256: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[3])?,
            p512: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[4])?,
            p860: Self::prepare_partial(module, LONG_CONTEXT_GQA_PARTITION_BUCKETS[5])?,
            reduction: module
                .prepare_long_context_paged_gqa_reduce_exact::<A, TOKENS>(LaunchConfig1D::new(
                    reduction_blocks,
                    THREADS,
                    REDUCTION_SHARED_BYTES,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing long-context paged GQA reduction route", source)
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
            .long_context_paged_gqa_partial_exact::<A, TOKENS>(
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
                GpuError::launch("launching long-context paged GQA partials", source)
            })?;
        module
            .long_context_paged_gqa_reduce_exact::<A, TOKENS>(
                stream,
                &self.reduction,
                lengths,
                partial_maximum,
                partial_denominator,
                partial_numerator,
                output,
            )
            .map_err(|source| {
                GpuError::launch("launching long-context paged GQA reduction", source)
            })
    }
}

/// Prepared partitioned paged GQA decode routes for exact `B=1..8`.
pub struct LongContextPagedGqaOp<A: Sm120Arch = Qwen38_27B> {
    module: kernels::LoadedModule,
    b1: PreparedRoute<A, 1>,
    b2: PreparedRoute<A, 2>,
    b3: PreparedRoute<A, 3>,
    b4: PreparedRoute<A, 4>,
    b5: PreparedRoute<A, 5>,
    b6: PreparedRoute<A, 6>,
    b7: PreparedRoute<A, 7>,
    b8: PreparedRoute<A, 8>,
}

impl<A: Sm120Arch> LongContextPagedGqaOp<A> {
    /// Loads the embedded module and prepares every exact decode route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        require_geometry::<A>()?;
        let _ = long_context_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded cuda-oxide module artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading long-context paged GQA", source))?;

        Ok(Self {
            b1: PreparedRoute::prepare(&module)?,
            b2: PreparedRoute::prepare(&module)?,
            b3: PreparedRoute::prepare(&module)?,
            b4: PreparedRoute::prepare(&module)?,
            b5: PreparedRoute::prepare(&module)?,
            b6: PreparedRoute::prepare(&module)?,
            b7: PreparedRoute::prepare(&module)?,
            b8: PreparedRoute::prepare(&module)?,
            module,
        })
    }

    /// Applies partitioned online-softmax GQA over page-major represented E4M3 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, 24, 256]` FP32 values. Cache planes use
    /// `[physical_page, 4, 64, 256]` E4M3 bytes. Metadata covers `batch`; every
    /// length is nonzero and no greater than `maximum_length`, its selected table
    /// row covers that length rounded up to 64, and every physical page ID is
    /// resident. Partial maximum and denominator each cover `[batch, 24, 860]`
    /// FP32 values; partial numerator covers `[batch, 24, 860, 256]`. Allocations
    /// are aligned, non-overlapping, live through completion, and belong to
    /// `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
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
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "long-context paged GQA batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }
        let partitions = partition_bucket(maximum_length).ok_or_else(|| {
            GpuError::invalid_launch(format!(
                "long-context paged GQA maximum length {maximum_length} is outside 1..={LONG_CONTEXT_MAX_TOKENS}"
            ))
        })?;
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("long-context paged GQA table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "long-context paged GQA table stride must be nonzero",
            ));
        }
        if !key_scale.is_finite() || key_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "long-context paged GQA key scale must be finite and positive",
            ));
        }
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(GpuError::invalid_launch(
                "long-context paged GQA value scale must be finite and positive",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                unsafe {
                    self.$route.launch(
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
}

/// PTX symbols retained for both stages of every exact long-context route.
pub(crate) fn long_context_paged_gqa_ptx_names() -> Vec<&'static str> {
    vec![
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 1>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 2>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 3>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 4>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 5>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 6>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 7>(),
        kernels::long_context_paged_gqa_partial_exact_ptx_name::<Qwen38_27B, 8>(),
        kernels::long_context_paged_gqa_reduce_exact_ptx_name::<Qwen38_27B, 8>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_MAX_TOKENS,
        LONG_CONTEXT_GQA_PARTITION_BUCKETS, LONG_CONTEXT_GQA_PARTITION_SIZE,
        REDUCTION_SHARED_BYTES, admitted_batch, long_context_paged_gqa_ptx_names, partition_bucket,
    };
    use std::collections::BTreeSet;

    #[test]
    fn batch_and_partition_routes_are_exact() {
        for (batch, expected) in [(0, false), (1, true), (4, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
        assert_eq!(LONG_CONTEXT_GQA_PARTITION_SIZE, 256);
        assert_eq!(LONG_CONTEXT_GQA_MAX_TOKENS, 220_000);
        assert_eq!(LONG_CONTEXT_GQA_MAX_PARTITIONS, 860);
        assert_eq!(REDUCTION_SHARED_BYTES, 3_440);
        assert_eq!(
            LONG_CONTEXT_GQA_PARTITION_BUCKETS,
            [4, 16, 64, 256, 512, 860]
        );
        assert_eq!(partition_bucket(0), None);
        assert_eq!(partition_bucket(1), Some(4));
        assert_eq!(partition_bucket(1_024), Some(4));
        assert_eq!(partition_bucket(1_025), Some(16));
        assert_eq!(partition_bucket(4_096), Some(16));
        assert_eq!(partition_bucket(4_097), Some(64));
        assert_eq!(partition_bucket(16_384), Some(64));
        assert_eq!(partition_bucket(16_385), Some(256));
        assert_eq!(partition_bucket(65_536), Some(256));
        assert_eq!(partition_bucket(65_537), Some(512));
        assert_eq!(partition_bucket(131_072), Some(512));
        assert_eq!(partition_bucket(131_073), Some(860));
        assert_eq!(partition_bucket(220_000), Some(860));
        assert_eq!(partition_bucket(220_001), None);
    }

    #[test]
    fn ptx_inventory_has_two_stages_per_batch() {
        let names = long_context_paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 16);
        assert_eq!(unique.len(), names.len());
    }
}

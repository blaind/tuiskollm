//! Exact-batch short-context paged grouped-query attention.

use crate::Sm120Arch;
use crate::device::paged_gqa::paged_gqa;
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const THREADS: u32 = 32;

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

/// Prepared short-context paged GQA decode routes for exact `B=1..8`.
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
    ]
}

#[cfg(test)]
mod tests {
    use super::{THREADS, admitted_batch, paged_gqa_ptx_names};
    use std::collections::BTreeSet;

    #[test]
    fn batch_table_covers_only_exact_decode_routes() {
        for (batch, expected) in [(0, false), (1, true), (4, true), (8, true), (9, false)] {
            assert_eq!(admitted_batch(batch), expected, "batch={batch}");
        }
        assert_eq!(THREADS, 32);
    }

    #[test]
    fn ptx_inventory_has_one_entry_per_batch() {
        let names = paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(unique.len(), names.len());
    }
}

//! Source-BF16 paged grouped-query attention for admitted MTP layers.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_attention::shared_device::{
    DECODE_RING_SHARED_BYTES, DECODE_SHARED_VALUES, DECODE_THREADS, bf16_paged_gqa,
    bf16_paged_gqa_partitioned,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const THREADS: u32 = DECODE_THREADS as u32;
const QWEN35_THREADS: u32 = 32;

fn admitted_batch(batch: usize) -> bool {
    (1..=MAX_BATCH).contains(&batch)
}

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;

    /// Applies represented-BF16 paged GQA for one exact MTP decode batch.
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
    pub fn mtp_bf16_paged_gqa<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        static mut DECODE_PARTIALS: SharedArray<f32, DECODE_SHARED_VALUES, 16> =
            SharedArray::UNINIT;
        let partials = core::ptr::addr_of_mut!(DECODE_PARTIALS).cast::<f32>();

        // Eight warps use the same represented-BF16 slice schedule as the
        // target decode route.
        unsafe {
            bf16_paged_gqa_partitioned::<Qwen38_27B, TOKENS>(
                query,
                key_pages,
                value_pages,
                block_tables,
                table_rows,
                table_stride,
                lengths,
                output,
                partials,
            );
        }
    }

    /// Applies represented-BF16 paged GQA for one exact Qwen3.5 MTP batch.
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
    pub fn qwen35_mtp_bf16_paged_gqa<const TOKENS: usize>(
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) {
        // One warp retains one 256-wide head and its online-softmax order.
        // Qwen3.5 B=8 exposes 128 CTAs for 16 query heads; combining heads
        // would change the represented reduction rather than just its tiling.
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
}

struct PreparedRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__mtp_bf16_paged_gqa_CudaKernel<TOKENS>>,
}

struct PreparedQwen35Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_mtp_bf16_paged_gqa_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> PreparedQwen35Route<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * Qwen35_9B::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("Qwen3.5 MTP BF16 paged GQA grid exceeds u32"))?;
        Ok(Self {
            attention: module
                .prepare_qwen35_mtp_bf16_paged_gqa::<TOKENS>(LaunchConfig1D::new(
                    blocks,
                    QWEN35_THREADS,
                    DECODE_RING_SHARED_BYTES as u32,
                ))
                .map_err(|source| {
                    GpuError::launch("preparing Qwen3.5 MTP BF16 paged GQA", source)
                })?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .qwen35_mtp_bf16_paged_gqa::<TOKENS>(
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
            )
            .map_err(|source| GpuError::launch("launching Qwen3.5 MTP BF16 paged GQA", source))
    }
}

impl<const TOKENS: usize> PreparedRoute<TOKENS> {
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self> {
        let blocks = u32::try_from(TOKENS * Qwen38_27B::NUM_ATTENTION_HEADS)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 paged GQA grid exceeds u32"))?;

        Ok(Self {
            attention: module
                .prepare_mtp_bf16_paged_gqa::<TOKENS>(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(|source| GpuError::launch("preparing MTP BF16 paged GQA", source))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: u32,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        module
            .mtp_bf16_paged_gqa::<TOKENS>(
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
            )
            .map_err(|source| GpuError::launch("launching MTP BF16 paged GQA", source))
    }
}

/// Stable PTX symbol inventory for every exact MTP BF16 paged-GQA batch.
pub(crate) fn mtp_bf16_paged_gqa_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::mtp_bf16_paged_gqa_ptx_name::<1>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<2>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<3>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<4>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<5>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<6>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<7>(),
        kernels::mtp_bf16_paged_gqa_ptx_name::<8>(),
    ]
}

/// Stable PTX inventory for every exact Qwen3.5 MTP BF16 paged-GQA batch.
pub(crate) fn qwen35_mtp_bf16_paged_gqa_ptx_names() -> [&'static str; MAX_BATCH] {
    [
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<1>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<2>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<3>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<4>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<5>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<6>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<7>(),
        kernels::qwen35_mtp_bf16_paged_gqa_ptx_name::<8>(),
    ]
}

/// Prepared Qwen3.8 MTP represented-BF16 paged-GQA routes.
pub struct MtpBf16PagedGqaOp {
    module: kernels::LoadedModule,
    b1: PreparedRoute<1>,
    b2: PreparedRoute<2>,
    b3: PreparedRoute<3>,
    b4: PreparedRoute<4>,
    b5: PreparedRoute<5>,
    b6: PreparedRoute<6>,
    b7: PreparedRoute<7>,
    b8: PreparedRoute<8>,
}

impl MtpBf16PagedGqaOp {
    /// Loads the embedded module and prepares every exact MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if Qwen38_27B::NUM_ATTENTION_HEADS != 24
            || Qwen38_27B::NUM_KV_HEADS != 4
            || Qwen38_27B::HEAD_DIM != 256
            || Qwen38_27B::ATTENTION_OUTPUT_COLUMNS != 6_144
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 geometry is incompatible with the MTP BF16 paged-GQA schedule",
            ));
        }
        let _ = mtp_bf16_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading MTP BF16 paged GQA", source))?;

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

    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, 24, 256]` FP32 values. Cache planes
    /// use `[physical_page, 4, 64, 256]` BF16 values. Metadata covers `batch`;
    /// each length is nonzero, its table row covers that length rounded up to
    /// 64, and every physical page is resident. Allocations are aligned,
    /// disjoint, live through completion, and belong to `stream`'s context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "MTP BF16 paged GQA batch {batch} is outside the admitted range 1..={MAX_BATCH}"
            )));
        }
        let table_stride = u32::try_from(table_stride)
            .map_err(|_| GpuError::invalid_launch("MTP BF16 paged GQA table stride exceeds u32"))?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "MTP BF16 paged GQA table stride must be nonzero",
            ));
        }

        macro_rules! launch {
            ($route:ident) => {
                // SAFETY: exact-B dispatch preserves the public pointer contract.
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

/// Prepared Qwen3.5 MTP represented-BF16 paged-GQA routes.
pub struct Qwen35MtpBf16PagedGqaOp {
    module: kernels::LoadedModule,
    b1: PreparedQwen35Route<1>,
    b2: PreparedQwen35Route<2>,
    b3: PreparedQwen35Route<3>,
    b4: PreparedQwen35Route<4>,
    b5: PreparedQwen35Route<5>,
    b6: PreparedQwen35Route<6>,
    b7: PreparedQwen35Route<7>,
    b8: PreparedQwen35Route<8>,
}

impl Qwen35MtpBf16PagedGqaOp {
    /// Loads the embedded module and prepares every exact Qwen3.5 route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        if Qwen35_9B::NUM_ATTENTION_HEADS != 16
            || Qwen35_9B::NUM_KV_HEADS != 4
            || Qwen35_9B::HEAD_DIM != 256
            || Qwen35_9B::ATTENTION_OUTPUT_COLUMNS != 4_096
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 geometry is incompatible with the MTP BF16 paged-GQA schedule",
            ));
        }
        let _ = qwen35_mtp_bf16_paged_gqa_ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
        let module = unsafe { kernels::load(context) }
            .map_err(|source| GpuError::module("loading Qwen3.5 MTP BF16 paged GQA", source))?;

        Ok(Self {
            b1: PreparedQwen35Route::prepare(&module)?,
            b2: PreparedQwen35Route::prepare(&module)?,
            b3: PreparedQwen35Route::prepare(&module)?,
            b4: PreparedQwen35Route::prepare(&module)?,
            b5: PreparedQwen35Route::prepare(&module)?,
            b6: PreparedQwen35Route::prepare(&module)?,
            b7: PreparedQwen35Route::prepare(&module)?,
            b8: PreparedQwen35Route::prepare(&module)?,
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, 16, 256]` FP32 values. Cache planes
    /// use `[physical_page, 4, 64, 256]` BF16 values. Metadata covers `batch`;
    /// every nonzero length has enough resident table entries. Allocations are
    /// aligned, disjoint, live through completion, and context-local.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        batch: usize,
        query: *const f32,
        key_pages: *const u16,
        value_pages: *const u16,
        block_tables: *const u32,
        table_rows: *const u32,
        table_stride: usize,
        lengths: *const u32,
        output: *mut f32,
    ) -> GpuResult<()> {
        if !admitted_batch(batch) {
            return Err(GpuError::invalid_launch(format!(
                "Qwen3.5 MTP BF16 paged GQA batch {batch} is outside 1..={MAX_BATCH}"
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch("Qwen3.5 MTP BF16 paged GQA table stride exceeds u32")
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 MTP BF16 paged GQA table stride must be nonzero",
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

#[cfg(test)]
mod tests {
    use super::{
        QWEN35_THREADS, THREADS, admitted_batch, mtp_bf16_paged_gqa_ptx_names,
        qwen35_mtp_bf16_paged_gqa_ptx_names,
    };
    use std::collections::BTreeSet;

    #[test]
    fn inventory_has_every_exact_batch_once() {
        let names = mtp_bf16_paged_gqa_ptx_names();
        let unique = names.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(unique.len(), names.len());
        assert_eq!(THREADS, 256);
        for batch in 0..=9 {
            assert_eq!(admitted_batch(batch), (1..=8).contains(&batch));
        }
    }

    #[test]
    fn qwen35_inventory_has_every_exact_batch_once() {
        let names = qwen35_mtp_bf16_paged_gqa_ptx_names();
        assert_eq!(names.len(), 8);
        assert_eq!(names.iter().copied().collect::<BTreeSet<_>>().len(), 8);
        assert_eq!(QWEN35_THREADS, 32);
    }
}

//! Source-BF16 paged grouped-query attention for admitted MTP layers.

use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaStream, GpuError, GpuResult, LaunchConfig1D, PreparedLaunch};
use tuisko_kernels_sm120_attention::shared_device::{
    DECODE_RING_SHARED_BYTES, DECODE_SHARED_VALUES, DECODE_THREADS, bf16_paged_gqa,
    bf16_paged_gqa_partitioned,
};
use tuisko_kernels_sm120_common::Sm120Arch;
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

mod private {
    pub trait Sealed {}
}

/// One architecture's prepared paged-GQA entry for an exact decode batch.
///
/// Sealed: the implementors are this module's prepared routes, so an entry
/// table can never name a route whose entry the module does not emit.
pub trait MtpPagedGqaRoute<A: Arch>: Sized + private::Sealed {
    /// Prepares this route's exact-batch entry.
    fn prepare(module: &kernels::LoadedModule) -> GpuResult<Self>;

    /// Launches this route's paged-GQA entry.
    ///
    /// # Safety
    ///
    /// The pointers carry `MtpBf16PagedGqaOp::launch`'s contract unchanged.
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
    ) -> GpuResult<()>;
}

/// Exact entry table of one admitted architecture's MTP paged-GQA routes.
///
/// The table is parameterized by the architecture instead of bounding
/// [`Sm120Arch`], so admitting Qwen3.5 here never widens the artifact-level
/// admission bound. Each table names only the entries its own model emits,
/// which keeps the compiled inventory fixed while both prepared owners share
/// one wrapper.
pub trait MtpPagedGqaEntries<A: Arch>: private::Sealed {
    /// Prepared decode route for one exact batch.
    type Decode<const TOKENS: usize>: MtpPagedGqaRoute<A>;

    /// Message prefix that keeps this table's launch rejections distinct.
    const LABEL: &'static str;
    /// Phrase this table uses for an unadmitted batch, retained per owner.
    const RANGE_PHRASE: &'static str;
    /// Operation named when loading the embedded module fails.
    const MODULE_OPERATION: &'static str;

    /// Rejects an architecture whose geometry the emitted entries do not cover.
    fn require_geometry() -> GpuResult<()>;

    /// Retained PTX entry names of every route this table admits.
    fn ptx_names() -> Vec<&'static str>;
}

/// Prepared Qwen3.8 partitioned decode entry for one exact batch.
pub struct PreparedRoute<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__mtp_bf16_paged_gqa_CudaKernel<TOKENS>>,
}

/// Prepared Qwen3.5 one-warp decode entry for one exact batch.
pub struct PreparedQwen35Route<const TOKENS: usize> {
    attention: PreparedLaunch<kernels::__qwen35_mtp_bf16_paged_gqa_CudaKernel<TOKENS>>,
}

impl<const TOKENS: usize> private::Sealed for PreparedRoute<TOKENS> {}
impl<const TOKENS: usize> private::Sealed for PreparedQwen35Route<TOKENS> {}

impl<const TOKENS: usize> MtpPagedGqaRoute<Qwen35_9B> for PreparedQwen35Route<TOKENS> {
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

// The Qwen3.8 entry compiles that model's head count into a concrete symbol,
// so its route stays bound to the sealed artifact-level architecture.
impl<A: Sm120Arch, const TOKENS: usize> MtpPagedGqaRoute<A> for PreparedRoute<TOKENS> {
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

/// Qwen3.8 entry table: the eight-warp partitioned decode entries.
pub struct Qwen38MtpPagedGqaEntries;

/// Qwen3.5 entry table: the one-warp ring decode entries.
pub struct Qwen35MtpPagedGqaEntries;

impl private::Sealed for Qwen38MtpPagedGqaEntries {}
impl private::Sealed for Qwen35MtpPagedGqaEntries {}

impl<A: Sm120Arch> MtpPagedGqaEntries<A> for Qwen38MtpPagedGqaEntries {
    type Decode<const TOKENS: usize> = PreparedRoute<TOKENS>;

    const LABEL: &'static str = "";
    const RANGE_PHRASE: &'static str = "the admitted range ";
    const MODULE_OPERATION: &'static str = "loading MTP BF16 paged GQA";

    fn require_geometry() -> GpuResult<()> {
        if Qwen38_27B::NUM_ATTENTION_HEADS != 24
            || Qwen38_27B::NUM_KV_HEADS != 4
            || Qwen38_27B::HEAD_DIM != 256
            || Qwen38_27B::ATTENTION_OUTPUT_COLUMNS != 6_144
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.8 geometry is incompatible with the MTP BF16 paged-GQA schedule",
            ));
        }
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        mtp_bf16_paged_gqa_ptx_names().to_vec()
    }
}

impl MtpPagedGqaEntries<Qwen35_9B> for Qwen35MtpPagedGqaEntries {
    type Decode<const TOKENS: usize> = PreparedQwen35Route<TOKENS>;

    const LABEL: &'static str = "Qwen3.5 ";
    const RANGE_PHRASE: &'static str = "";
    const MODULE_OPERATION: &'static str = "loading Qwen3.5 MTP BF16 paged GQA";

    fn require_geometry() -> GpuResult<()> {
        if Qwen35_9B::NUM_ATTENTION_HEADS != 16
            || Qwen35_9B::NUM_KV_HEADS != 4
            || Qwen35_9B::HEAD_DIM != 256
            || Qwen35_9B::ATTENTION_OUTPUT_COLUMNS != 4_096
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 geometry is incompatible with the MTP BF16 paged-GQA schedule",
            ));
        }
        Ok(())
    }

    fn ptx_names() -> Vec<&'static str> {
        qwen35_mtp_bf16_paged_gqa_ptx_names().to_vec()
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

/// Prepared represented-BF16 paged-GQA routes for exact MTP decode `B=1..=8`.
pub struct MtpBf16PagedGqaOp<
    A: Arch = Qwen38_27B,
    E: MtpPagedGqaEntries<A> = Qwen38MtpPagedGqaEntries,
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
}

/// Prepared Qwen3.5 represented-BF16 paged-GQA routes.
pub type Qwen35MtpBf16PagedGqaOp = MtpBf16PagedGqaOp<Qwen35_9B, Qwen35MtpPagedGqaEntries>;

impl<A: Arch, E: MtpPagedGqaEntries<A>> MtpBf16PagedGqaOp<A, E> {
    /// Loads the embedded module and prepares every exact MTP route.
    pub fn new(context: &Arc<CudaContext>) -> GpuResult<Self> {
        E::require_geometry()?;
        let _ = E::ptx_names();
        // SAFETY: this crate owns the embedded exact MTP artifact.
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
            module,
        })
    }

    /// Applies online-softmax GQA over page-major represented BF16 K/V.
    ///
    /// # Safety
    ///
    /// Query and output cover `[batch, A::NUM_ATTENTION_HEADS, 256]` FP32
    /// values. Cache planes use `[physical_page, 4, 64, 256]` BF16 values.
    /// Metadata covers `batch`; each length is nonzero, its table row covers
    /// that length rounded up to 64, and every physical page is resident.
    /// Allocations are aligned, disjoint, live through completion, and belong
    /// to `stream`'s context.
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
                "{}MTP BF16 paged GQA batch {batch} is outside {}1..={MAX_BATCH}",
                E::LABEL,
                E::RANGE_PHRASE,
            )));
        }
        let table_stride = u32::try_from(table_stride).map_err(|_| {
            GpuError::invalid_launch(format!(
                "{}MTP BF16 paged GQA table stride exceeds u32",
                E::LABEL
            ))
        })?;
        if table_stride == 0 {
            return Err(GpuError::invalid_launch(format!(
                "{}MTP BF16 paged GQA table stride must be nonzero",
                E::LABEL
            )));
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

#[cfg(test)]
mod tests {
    use super::{
        MtpPagedGqaEntries, QWEN35_THREADS, Qwen35MtpPagedGqaEntries, Qwen38MtpPagedGqaEntries,
        THREADS, admitted_batch, mtp_bf16_paged_gqa_ptx_names, qwen35_mtp_bf16_paged_gqa_ptx_names,
    };
    use std::collections::BTreeSet;
    use tuisko_model::{Qwen35_9B, Qwen38_27B};

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

    /// Each entry table publishes exactly the list that retains its own
    /// specializations, so merging the owners cannot merge the inventories.
    #[test]
    fn every_entry_table_publishes_its_own_inventory() {
        assert_eq!(
            <Qwen38MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen38_27B>>::ptx_names(),
            mtp_bf16_paged_gqa_ptx_names().to_vec()
        );
        assert_eq!(
            <Qwen35MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen35_9B>>::ptx_names(),
            qwen35_mtp_bf16_paged_gqa_ptx_names().to_vec()
        );
    }

    /// Both owners keep the rejection wording they had before the merge.
    #[test]
    fn unadmitted_batches_keep_their_owner_wording() {
        assert_eq!(
            format!(
                "{}MTP BF16 paged GQA batch 9 is outside {}1..=8",
                <Qwen38MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen38_27B>>::LABEL,
                <Qwen38MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen38_27B>>::RANGE_PHRASE,
            ),
            "MTP BF16 paged GQA batch 9 is outside the admitted range 1..=8"
        );
        assert_eq!(
            format!(
                "{}MTP BF16 paged GQA batch 9 is outside {}1..=8",
                <Qwen35MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen35_9B>>::LABEL,
                <Qwen35MtpPagedGqaEntries as MtpPagedGqaEntries<Qwen35_9B>>::RANGE_PHRASE,
            ),
            "Qwen3.5 MTP BF16 paged GQA batch 9 is outside 1..=8"
        );
    }
}

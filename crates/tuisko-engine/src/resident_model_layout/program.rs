//! Resident exact-target execution owner for the complete text model.

use super::{
    AttentionWeights, EndpointWeights, GdnPersistent, GdnWeights, MixerWeights, MlpWeights,
    ResidentLoadProgress, ResidentModelLayout, ResidentUploadArena, ResidentUploadPlan,
    ResidentUploadPreparation,
};
#[cfg(feature = "qualification")]
use crate::PagedKvSlotState;
use crate::long_context_kv_layout::LayerKvRegions;
use crate::resident_mtp::ResidentMtpArenaReservation;
use crate::{
    EngineError, EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, PagedKvSlotPool,
    PagedKvTableUpdate, full_attention_layer_layout::CONTEXT_CAPACITY as SHORT_CONTEXT_CAPACITY,
    long_context_kv_layout::MAX_CONTEXT_TOKENS,
};
#[cfg(feature = "qualification")]
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use tuisko_gpu::{
    ArenaRegion, CudaContext, CudaGraph, CudaGraphDefinition, CudaGraphVariants, CudaStream,
    DeviceArena, DeviceCopy, GpuError, GpuResult, LoadingDeviceArena, PinnedHostBuffer,
};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_kernels_sm120::{
    AttentionOutputOp, AttentionQkPrepareOp, DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp,
    DenseFp8SwiGluTmaMaps, FullAttentionQkvOp, GdnInputProjectionOp, GdnOutputProjectionOp,
    GdnPrepareOp, GdnRecurrenceOp, GdnStateSnapshotOp, LONG_CONTEXT_GQA_PARTITION_BUCKETS,
    LONG_CONTEXT_GQA_PARTITION_SIZE, LmHeadOp, LongContextPagedGqaOp, Nvfp4DownOp, Nvfp4SwiGluOp,
    PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT, PagedGqaOp, ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, DenseFp8DownBindings, DenseFp8GateUpBindings,
    FullAttentionPostBindings, FullAttentionQkvBindings, GdnBindings, Nvfp4DownBindings,
    Nvfp4GateUpBindings, Qwen38_27B, TextEndpointBindings, nvfp4_scale_materialization_workers,
};

const ROTARY_PAIRS: usize = 32;
#[cfg(feature = "qualification")]
const SHORT_CONTEXT_PAGES_PER_SLOT: usize = SHORT_CONTEXT_CAPACITY / ATTENTION_PAGE_SIZE;
#[cfg(feature = "qualification")]
const SHORT_CONTEXT_PHYSICAL_PAGES: usize = MAX_BATCH * SHORT_CONTEXT_PAGES_PER_SLOT;
const LONG_CONTEXT_ROUTE_COUNT: usize = LONG_CONTEXT_GQA_PARTITION_BUCKETS.len();
const PREFILL_GRAPH_ROUTE_COUNT: usize = 6;
const TARGET_VERIFY_ROUTE_COUNT: usize = 4;
const TARGET_SEGMENTED_BATCH_ROUTES: usize = MAX_BATCH - 1;
const TARGET_VERIFY_ROWS: usize = MAX_BATCH * TARGET_VERIFY_ROUTE_COUNT;
const GDN_LAYER_COUNT: usize = 48;

/// Exact resident graph selected by one checked decode-state upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the decode route must be replayed with the state that selected it"]
pub struct ResidentDecodeRoute {
    batch: usize,
    maximum_length: usize,
    attention: AttentionRoute,
}

/// Exact from-empty prompt route selected by one checked state upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the prefill route must be replayed with the state that selected it"]
pub struct ResidentPrefillRoute {
    tokens: usize,
    first_position: usize,
    context_tokens: usize,
    attention: PrefillAttentionRoute,
}

/// Exact provisional target-verification graph selected by one checked upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the target-verification route must be replayed with the state that selected it"]
pub struct ResidentMtpVerifyRoute {
    tokens: usize,
    slot: usize,
    maximum_length: usize,
    attention: AttentionRoute,
}

/// Exact lane-major provisional target-verification graph selected by one checked upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the segmented target-verification route must be replayed with its uploaded state"]
pub struct ResidentMtpSegmentedVerifyRoute {
    tokens: usize,
    batch: usize,
    maximum_length: usize,
    attention: AttentionRoute,
}

impl ResidentMtpSegmentedVerifyRoute {
    /// Number of causal target input rows owned by each lane.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Number of distinct resident slots in the compact transaction.
    pub const fn batch(self) -> usize {
        self.batch
    }

    /// Total lane-major target rows produced by this route.
    pub const fn rows(self) -> usize {
        self.tokens * self.batch
    }

    /// Largest causal context length after the provisional rows are evaluated.
    pub const fn maximum_length(self) -> usize {
        self.maximum_length
    }

    /// Whether this route uses partitioned long-context attention.
    pub const fn is_long_context(self) -> bool {
        matches!(self.attention, AttentionRoute::Long { .. })
    }

    /// Captured partition capacity, or `None` for the short-context graph.
    pub const fn partition_capacity(self) -> Option<usize> {
        match self.attention {
            AttentionRoute::Short => None,
            AttentionRoute::Long { partitions, .. } => Some(partitions),
        }
    }
}

impl ResidentMtpVerifyRoute {
    /// Number of causal target input rows in this exact route.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Stable resident slot whose state is snapshotted provisionally.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// Causal context length after all provisional rows are evaluated.
    pub const fn maximum_length(self) -> usize {
        self.maximum_length
    }

    /// Whether this route uses partitioned long-context attention.
    pub const fn is_long_context(self) -> bool {
        matches!(self.attention, AttentionRoute::Long { .. })
    }

    /// Captured partition capacity, or `None` for the short-context graph.
    pub const fn partition_capacity(self) -> Option<usize> {
        match self.attention {
            AttentionRoute::Short => None,
            AttentionRoute::Long { partitions, .. } => Some(partitions),
        }
    }
}

impl ResidentPrefillRoute {
    /// Number of contiguous prompt tokens captured by this exact route.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// First absolute cache position written by this tile.
    pub const fn first_position(self) -> usize {
        self.first_position
    }

    /// Causal context length after the complete tile is processed.
    pub const fn context_tokens(self) -> usize {
        self.context_tokens
    }

    /// Exact partition count used by a partitioned or macro attention route.
    pub const fn partition_capacity(self) -> Option<usize> {
        match self.attention {
            PrefillAttentionRoute::Shared => None,
            PrefillAttentionRoute::Partitioned { partitions }
            | PrefillAttentionRoute::Macro { partitions } => Some(partitions),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefillAttentionRoute {
    Shared,
    Partitioned { partitions: usize },
    Macro { partitions: usize },
}

impl ResidentDecodeRoute {
    /// Number of compact rows captured by this exact route.
    pub const fn batch(self) -> usize {
        self.batch
    }

    /// Largest uploaded cache length across the active rows.
    pub const fn maximum_length(self) -> usize {
        self.maximum_length
    }

    /// Whether this route uses partitioned long-context attention.
    pub const fn is_long_context(self) -> bool {
        matches!(self.attention, AttentionRoute::Long { .. })
    }

    /// Captured partition capacity, or `None` for the short-context graph.
    pub const fn partition_capacity(self) -> Option<usize> {
        match self.attention {
            AttentionRoute::Short => None,
            AttentionRoute::Long { partitions, .. } => Some(partitions),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionRoute {
    Short,
    Long { index: usize, partitions: usize },
}

/// Resident checkpoint loading implementation selected before graph construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentLoadMode {
    /// Existing pageable, synchronized upload path retained as A/B authority.
    Legacy,
    /// Selective initialization with queued direct uploads over a sealed arena.
    Selective,
}

const PRODUCTION_LOAD_MODE: ResidentLoadMode = ResidentLoadMode::Selective;

impl ResidentLoadMode {
    /// Stable spelling used in startup reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Selective => "selective",
        }
    }
}

/// Exact initialization work performed before resident graph capture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentLoadStats {
    mode: ResidentLoadMode,
    upload_bytes: usize,
    upload_submissions: usize,
    zeroed_bytes: usize,
    pinned_stager_bytes: usize,
    layout_plan_ns: u64,
    arena_allocation_ns: u64,
    operator_setup_ns: u64,
    source_prefault_ns: u64,
    weight_prepare_ns: u64,
    source_binding_ns: u64,
    qkv_gather_ns: u64,
    nvfp4_materialize_ns: u64,
    preparation_other_ns: u64,
    weight_copy_ns: u64,
    weight_load_ns: u64,
    nonweight_init_ns: u64,
    graph_capture_ns: u64,
    prefault_bytes: usize,
    borrowed_source_bytes: usize,
    gathered_source_bytes: usize,
    swizzled_source_bytes: usize,
    nvfp4_materialize_workers: usize,
}

impl ResidentLoadStats {
    /// Selected loading implementation.
    pub const fn mode(self) -> ResidentLoadMode {
        self.mode
    }

    /// Source-backed weight and host-derived metadata bytes uploaded.
    pub const fn upload_bytes(self) -> usize {
        self.upload_bytes
    }

    /// Host-to-device copy submissions used for those bytes.
    pub const fn upload_submissions(self) -> usize {
        self.upload_submissions
    }

    /// Device bytes initialized through memset operations.
    pub const fn zeroed_bytes(self) -> usize {
        self.zeroed_bytes
    }

    /// Fixed page-locked host bytes retained during loading, if any.
    pub const fn pinned_stager_bytes(self) -> usize {
        self.pinned_stager_bytes
    }

    /// Host nanoseconds spent deriving the layout and exact upload plan.
    pub const fn layout_plan_ns(self) -> u64 {
        self.layout_plan_ns
    }

    /// Host nanoseconds spent creating and completing the device allocations.
    pub const fn arena_allocation_ns(self) -> u64 {
        self.arena_allocation_ns
    }

    /// Host nanoseconds spent loading operators and allocating their fixed host state.
    pub const fn operator_setup_ns(self) -> u64 {
        self.operator_setup_ns
    }

    /// Host nanoseconds spent populating source-mapping page tables before CUDA access.
    pub const fn source_prefault_ns(self) -> u64 {
        self.source_prefault_ns
    }

    /// Host nanoseconds spent binding and materializing source values outside CUDA copy calls.
    pub const fn weight_prepare_ns(self) -> u64 {
        self.weight_prepare_ns
    }

    /// Host nanoseconds spent binding and validating exact source families.
    pub const fn source_binding_ns(self) -> u64 {
        self.source_binding_ns
    }

    /// Host nanoseconds spent gathering separate Q/K/V planes into resident order.
    pub const fn qkv_gather_ns(self) -> u64 {
        self.qkv_gather_ns
    }

    /// Host nanoseconds spent losslessly swizzling NVFP4 scale planes.
    pub const fn nvfp4_materialize_ns(self) -> u64 {
        self.nvfp4_materialize_ns
    }

    /// Remaining host preparation not attributed to binding, QKV gathering, or NVFP4 swizzling.
    pub const fn preparation_other_ns(self) -> u64 {
        self.preparation_other_ns
    }

    /// Host nanoseconds spent inside CUDA host-to-device copy calls and their required waits.
    pub const fn weight_copy_ns(self) -> u64 {
        self.weight_copy_ns
    }

    /// Host nanoseconds spent materializing and uploading source-backed weights.
    pub const fn weight_load_ns(self) -> u64 {
        self.weight_load_ns
    }

    /// Host nanoseconds spent initializing metadata, runtime state, cache, and padding.
    pub const fn nonweight_init_ns(self) -> u64 {
        self.nonweight_init_ns
    }

    /// Host nanoseconds spent binding stable pointers and capturing the graph inventory.
    pub const fn graph_capture_ns(self) -> u64 {
        self.graph_capture_ns
    }

    /// Immutable source-mapping bytes submitted for page-table population.
    pub const fn prefault_bytes(self) -> usize {
        self.prefault_bytes
    }

    /// Weight bytes borrowed directly from admitted mmap-backed source planes.
    pub const fn borrowed_source_bytes(self) -> usize {
        self.borrowed_source_bytes
    }

    /// Weight bytes gathered from multiple admitted source planes.
    pub const fn gathered_source_bytes(self) -> usize {
        self.gathered_source_bytes
    }

    /// Weight bytes losslessly swizzled into kernel scale order.
    pub const fn swizzled_source_bytes(self) -> usize {
        self.swizzled_source_bytes
    }

    /// Maximum worker count used to materialize target-size NVFP4 scale planes.
    pub const fn nvfp4_materialize_workers(self) -> usize {
        self.nvfp4_materialize_workers
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResidentPreparationStats {
    source_binding_ns: u64,
    qkv_gather_ns: u64,
    nvfp4_materialize_ns: u64,
}

impl ResidentPreparationStats {
    fn total_ns(self) -> EngineResult<u64> {
        self.source_binding_ns
            .checked_add(self.qkv_gather_ns)
            .and_then(|total| total.checked_add(self.nvfp4_materialize_ns))
            .ok_or_else(|| EngineError::layout("resident preparation timing overflows"))
    }
}

enum ArenaLoading {
    Legacy {
        arena: DeviceArena,
        kv_arena: DeviceArena,
    },
    Selective {
        arena: LoadingDeviceArena,
        kv_arena: LoadingDeviceArena,
    },
}

struct ResidentGraphs {
    short: [CudaGraph; MAX_BATCH],
    long: [[CudaGraph; MAX_BATCH]; LONG_CONTEXT_ROUTE_COUNT],
    prefill: [CudaGraph; PREFILL_GRAPH_ROUTE_COUNT],
    target_verify_short: [CudaGraph; TARGET_VERIFY_ROUTE_COUNT],
    target_verify_long: [[CudaGraph; TARGET_VERIFY_ROUTE_COUNT]; LONG_CONTEXT_ROUTE_COUNT],
    target_segmented_verify_short:
        [[CudaGraph; TARGET_SEGMENTED_BATCH_ROUTES]; TARGET_VERIFY_ROUTE_COUNT],
    target_segmented_verify_long: [[CudaGraphVariants<LONG_CONTEXT_ROUTE_COUNT>;
        TARGET_SEGMENTED_BATCH_ROUTES];
        TARGET_VERIFY_ROUTE_COUNT],
    target_commit: [CudaGraph; TARGET_VERIFY_ROUTE_COUNT],
}

impl ResidentGraphs {
    fn select(&self, route: ResidentDecodeRoute) -> &CudaGraph {
        match route.attention {
            AttentionRoute::Short => &self.short[route.batch - 1],
            AttentionRoute::Long { index, .. } => &self.long[index][route.batch - 1],
        }
    }

    fn select_prefill(&self, route: ResidentPrefillRoute) -> EngineResult<&CudaGraph> {
        let index = prefill_graph_index(route)?;
        Ok(&self.prefill[index])
    }

    fn select_target_verify(&self, route: ResidentMtpVerifyRoute) -> &CudaGraph {
        match route.attention {
            AttentionRoute::Short => &self.target_verify_short[route.tokens - 1],
            AttentionRoute::Long { index, .. } => &self.target_verify_long[index][route.tokens - 1],
        }
    }

    fn select_direct_target_segmented_verify(
        &self,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> Option<&CudaGraph> {
        if route.batch == 1 {
            return Some(match route.attention {
                AttentionRoute::Short => &self.target_verify_short[route.tokens - 1],
                AttentionRoute::Long { index, .. } => {
                    &self.target_verify_long[index][route.tokens - 1]
                }
            });
        }
        match route.attention {
            AttentionRoute::Short => {
                Some(&self.target_segmented_verify_short[route.tokens - 1][route.batch - 2])
            }
            AttentionRoute::Long { .. } => None,
        }
    }

    /// Enqueues the direct graph or updated shared variant for one segmented route.
    ///
    /// # Safety
    ///
    /// This inventory does not own the allocations its recordings captured. The
    /// caller must keep every one of them alive and unmoved until `stream`
    /// completes the replay — the `ResidentModelProgram` holding this inventory
    /// alongside those allocations is what provides that guarantee.
    unsafe fn launch_target_segmented_verify(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> GpuResult<()> {
        if let Some(graph) = self.select_direct_target_segmented_verify(route) {
            // SAFETY: the caller keeps every allocation this graph captured alive
            // until the replay completes.
            return unsafe { graph.launch(stream) };
        }
        match route.attention {
            // SAFETY: the caller keeps every allocation this variant's definitions
            // captured alive until the replay completes.
            AttentionRoute::Long { index, .. } => unsafe {
                self.target_segmented_verify_long[route.tokens - 1][route.batch - 2]
                    .launch(stream, index)
            },
            AttentionRoute::Short => unreachable!("short segmented routes are direct graphs"),
        }
    }

    const fn target_mtp_executable_count(&self) -> usize {
        TARGET_VERIFY_ROUTE_COUNT
            * (LONG_CONTEXT_ROUTE_COUNT + 2 + 2 * TARGET_SEGMENTED_BATCH_ROUTES)
    }

    const fn executable_count(&self) -> usize {
        MAX_BATCH
            + PREFILL_GRAPH_ROUTE_COUNT
            + self.target_mtp_executable_count()
            + LONG_CONTEXT_ROUTE_COUNT * MAX_BATCH
    }

    const fn route_count(&self) -> usize {
        MAX_BATCH
            + PREFILL_GRAPH_ROUTE_COUNT
            + TARGET_VERIFY_ROUTE_COUNT
                * (LONG_CONTEXT_ROUTE_COUNT
                    + 2
                    + TARGET_SEGMENTED_BATCH_ROUTES * (LONG_CONTEXT_ROUTE_COUNT + 1))
            + LONG_CONTEXT_ROUTE_COUNT * MAX_BATCH
    }

    fn select_target_commit(&self, tokens: usize) -> EngineResult<&CudaGraph> {
        self.target_commit
            .get(tokens.wrapping_sub(1))
            .ok_or_else(|| {
                EngineError::route("target MTP commit must contain exactly 1..=4 input rows")
            })
    }
}

struct DenseMlpMaps {
    gate_up: DenseFp8SwiGluTmaMaps,
    down: DenseFp8DownTmaMaps,
}

/// Resident and shared-KV arenas plus immutable `B=1..=8` graphs for all 64 text layers.
pub struct ResidentModelProgram {
    // Graphs retain both arena addresses and module handles, so they drop first.
    graphs: ResidentGraphs,
    dense_mlp_maps: Vec<DenseMlpMaps>,
    arena: DeviceArena,
    kv_arena: DeviceArena,
    kv_slots: PagedKvSlotPool,
    _norm: ResidualNormOp<Qwen38_27B>,
    _gdn_input: GdnInputProjectionOp<Qwen38_27B>,
    _gdn_prepare: GdnPrepareOp<Qwen38_27B>,
    _gdn_recurrence: GdnRecurrenceOp<Qwen38_27B>,
    _gdn_state_snapshot: GdnStateSnapshotOp<Qwen38_27B>,
    _gdn_output: GdnOutputProjectionOp<Qwen38_27B>,
    _attention_qkv: FullAttentionQkvOp<Qwen38_27B>,
    _attention_qk_prepare: AttentionQkPrepareOp<Qwen38_27B>,
    _paged_gqa: PagedGqaOp<Qwen38_27B>,
    _long_context_paged_gqa: LongContextPagedGqaOp<Qwen38_27B>,
    _attention_output: AttentionOutputOp<Qwen38_27B>,
    _dense_swiglu: DenseFp8SwiGluOp<Qwen38_27B>,
    _dense_down: DenseFp8DownOp<Qwen38_27B>,
    _nvfp4_swiglu: Nvfp4SwiGluOp,
    _nvfp4_down: Nvfp4DownOp<Qwen38_27B>,
    _lm_head: LmHeadOp<Qwen38_27B>,
    embedding_stager: PinnedHostBuffer<u16>,
    snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    context: Arc<CudaContext>,
    layout: ResidentModelLayout,
    _pointers: ProgramPointers,
    base_address: u64,
    kv_base_address: u64,
    load_stats: ResidentLoadStats,
}

#[cfg(feature = "qualification")]
/// Captured embedding upload borrowing its page-locked source for every replay.
pub struct ResidentEmbeddingStageGraph<'a> {
    graph: CudaGraph,
    source: PhantomData<&'a PinnedHostBuffer<u16>>,
}

#[cfg(feature = "qualification")]
impl ResidentEmbeddingStageGraph<'_> {
    /// Immutable graph whose replay restores represented embedding rows.
    pub const fn graph(&self) -> &CudaGraph {
        &self.graph
    }
}

#[cfg(feature = "qualification")]
/// Captured production embedding and prompt-metadata uploads for one prefill route.
pub struct ResidentPrefillStageGraph<'a> {
    graph: CudaGraph,
    _positions: PinnedHostBuffer<u32>,
    _lengths: PinnedHostBuffer<u32>,
    _rows: PinnedHostBuffer<u32>,
    _rope_cos: PinnedHostBuffer<f32>,
    _rope_sin: PinnedHostBuffer<f32>,
    source: PhantomData<&'a PinnedHostBuffer<u16>>,
}

#[cfg(feature = "qualification")]
impl ResidentPrefillStageGraph<'_> {
    /// Immutable graph restoring represented embeddings and exact prompt metadata.
    pub const fn graph(&self) -> &CudaGraph {
        &self.graph
    }
}

#[cfg(feature = "qualification")]
/// Captured segmented target embedding and metadata uploads with retained pinned sources.
pub struct ResidentMtpSegmentedStageGraph<'a> {
    graph: CudaGraph,
    _positions: PinnedHostBuffer<u32>,
    _lengths: PinnedHostBuffer<u32>,
    _rows: PinnedHostBuffer<u32>,
    _rope_cos: PinnedHostBuffer<f32>,
    _rope_sin: PinnedHostBuffer<f32>,
    source: PhantomData<&'a PinnedHostBuffer<u16>>,
}

#[cfg(feature = "qualification")]
impl ResidentMtpSegmentedStageGraph<'_> {
    /// Immutable graph restoring exact lane-major target embeddings and metadata.
    pub const fn graph(&self) -> &CudaGraph {
        &self.graph
    }
}

impl ResidentModelProgram {
    /// Loads every exact source plane through the qualified prefaulted route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_mode(context, snapshot, PRODUCTION_LOAD_MODE, None, 0, false)
            .map(|(program, _)| program)
    }

    pub(crate) fn from_snapshot_reserving_mtp(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<(Self, ResidentMtpArenaReservation)> {
        let (program, reservation) =
            Self::from_snapshot_with_mode(context, snapshot, PRODUCTION_LOAD_MODE, None, 0, true)?;
        Ok((
            program,
            reservation.ok_or_else(|| {
                EngineError::layout("resident target did not return its requested MTP reservation")
            })?,
        ))
    }

    pub(crate) fn from_snapshot_reserving_mtp_with_progress(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        progress: &ResidentLoadProgress,
        mtp_upload_bytes: usize,
    ) -> EngineResult<(Self, ResidentMtpArenaReservation)> {
        let (program, reservation) = Self::from_snapshot_with_mode(
            context,
            snapshot,
            PRODUCTION_LOAD_MODE,
            Some(progress),
            mtp_upload_bytes,
            true,
        )?;
        Ok((
            program,
            reservation.ok_or_else(|| {
                EngineError::layout("resident target did not return its requested MTP reservation")
            })?,
        ))
    }

    pub(crate) fn from_snapshot_with_progress(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        progress: &ResidentLoadProgress,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_mode(
            context,
            snapshot,
            PRODUCTION_LOAD_MODE,
            Some(progress),
            0,
            false,
        )
        .map(|(program, _)| program)
    }

    /// Loads through selective initialization for focused qualification.
    #[cfg(feature = "qualification")]
    pub fn from_snapshot_selective(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_mode(
            context,
            snapshot,
            ResidentLoadMode::Selective,
            None,
            0,
            false,
        )
        .map(|(program, _)| program)
    }

    /// Loads through the retained eager-zeroing A/B authority.
    #[cfg(feature = "qualification")]
    pub fn from_snapshot_legacy(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_mode(context, snapshot, ResidentLoadMode::Legacy, None, 0, false)
            .map(|(program, _)| program)
    }

    fn from_snapshot_with_mode(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
        mode: ResidentLoadMode,
        progress: Option<&ResidentLoadProgress>,
        progress_tail_upload_bytes: usize,
        reserve_mtp: bool,
    ) -> EngineResult<(Self, Option<ResidentMtpArenaReservation>)> {
        let layout_start = Instant::now();
        let layout = ResidentModelLayout::build()?;
        let upload_plan = ResidentUploadPlan::build(&layout)?;
        let borrowed_source_bytes =
            upload_plan.weight_bytes_for(ResidentUploadPreparation::BorrowedSource);
        let gathered_source_bytes =
            upload_plan.weight_bytes_for(ResidentUploadPreparation::GatheredSource);
        let swizzled_source_bytes =
            upload_plan.weight_bytes_for(ResidentUploadPreparation::SwizzledSource);
        let layout_plan_ns = elapsed_ns("resident layout and upload plan", layout_start)?;

        let allocation_start = Instant::now();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arenas = match mode {
            ResidentLoadMode::Legacy => ArenaLoading::Legacy {
                arena: DeviceArena::zeroed(&stream, &layout.builder)?,
                kv_arena: DeviceArena::zeroed(&stream, layout.kv_layout.builder())?,
            },
            ResidentLoadMode::Selective => ArenaLoading::Selective {
                arena: LoadingDeviceArena::allocate(&stream, &layout.builder)?,
                kv_arena: LoadingDeviceArena::allocate(&stream, layout.kv_layout.builder())?,
            },
        };
        stream.synchronize().map_err(GpuError::from)?;
        let arena_allocation_ns = elapsed_ns("resident arena allocation", allocation_start)?;
        let mtp_reservation = reserve_mtp
            .then(|| ResidentMtpArenaReservation::allocate(&stream))
            .transpose()?;
        stream.synchronize().map_err(GpuError::from)?;

        let operator_start = Instant::now();
        let kv_slots = PagedKvSlotPool::new(LONG_CONTEXT_PHYSICAL_PAGES)?;
        let norm = ResidualNormOp::new(context)?;
        let gdn_input = GdnInputProjectionOp::new(context)?;
        let gdn_prepare = GdnPrepareOp::new(context)?;
        let gdn_recurrence = GdnRecurrenceOp::new(context)?;
        let gdn_state_snapshot = GdnStateSnapshotOp::new(context)?;
        let gdn_output = GdnOutputProjectionOp::new(context)?;
        let attention_qkv = FullAttentionQkvOp::new(context)?;
        let attention_qk_prepare = AttentionQkPrepareOp::new(context)?;
        let paged_gqa = PagedGqaOp::new(context)?;
        let long_context_paged_gqa = LongContextPagedGqaOp::new(context)?;
        let attention_output = AttentionOutputOp::new(context)?;
        let dense_swiglu = DenseFp8SwiGluOp::new(context)?;
        let dense_down = DenseFp8DownOp::new(context)?;
        let nvfp4_swiglu = Nvfp4SwiGluOp::new(context)?;
        let nvfp4_down = Nvfp4DownOp::new(context)?;
        let lm_head = LmHeadOp::new(context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(
            context,
            product(
                "resident embedding stager elements",
                super::MAX_ROWS,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let operator_setup_ns = elapsed_ns("resident operator setup", operator_start)?;

        let (arena, kv_arena, scalars, mut load_stats) = match arenas {
            ArenaLoading::Legacy { arena, kv_arena } => {
                let weight_start = Instant::now();
                let mut sink = LegacyWeightSink {
                    arena: &arena,
                    copy_ns: 0,
                };
                let mut preparation = ResidentPreparationStats::default();
                let scalars = load_source_weights(
                    &mut sink,
                    &stream,
                    &layout,
                    snapshot.as_ref(),
                    &mut preparation,
                )?;
                let weight_load_ns = elapsed_ns("resident legacy weight load", weight_start)?;
                let weight_copy_ns = sink.copy_ns;
                let weight_prepare_ns =
                    weight_load_ns.checked_sub(weight_copy_ns).ok_or_else(|| {
                        EngineError::layout(
                            "legacy weight-copy time exceeds total weight-load time",
                        )
                    })?;
                let preparation_other_ns = weight_prepare_ns
                    .checked_sub(preparation.total_ns()?)
                    .ok_or_else(|| {
                    EngineError::layout("classified legacy preparation exceeds total preparation")
                })?;

                let nonweight_start = Instant::now();
                initialize_metadata(&arena, &kv_arena, &stream, &layout)?;
                let nonweight_init_ns =
                    elapsed_ns("resident legacy non-weight initialization", nonweight_start)?;
                let upload_submissions = upload_plan
                    .entries()
                    .iter()
                    .filter(|entry| entry.preparation() != ResidentUploadPreparation::Zero)
                    .count();
                let load_stats = ResidentLoadStats {
                    mode,
                    upload_bytes: upload_plan.weight_bytes() + upload_plan.host_derived_bytes(),
                    upload_submissions,
                    zeroed_bytes: layout.arena_bytes(),
                    pinned_stager_bytes: 0,
                    layout_plan_ns,
                    arena_allocation_ns,
                    operator_setup_ns,
                    source_prefault_ns: 0,
                    weight_prepare_ns,
                    source_binding_ns: preparation.source_binding_ns,
                    qkv_gather_ns: preparation.qkv_gather_ns,
                    nvfp4_materialize_ns: preparation.nvfp4_materialize_ns,
                    preparation_other_ns,
                    weight_copy_ns,
                    weight_load_ns,
                    nonweight_init_ns,
                    graph_capture_ns: 0,
                    prefault_bytes: 0,
                    borrowed_source_bytes,
                    gathered_source_bytes,
                    swizzled_source_bytes,
                    nvfp4_materialize_workers: nvfp4_scale_materialization_workers(),
                };
                (arena, kv_arena, scalars, load_stats)
            }
            ArenaLoading::Selective {
                mut arena,
                mut kv_arena,
            } => {
                let weight_start = Instant::now();
                let expected_upload_bytes = upload_plan
                    .weight_bytes()
                    .checked_add(upload_plan.host_derived_bytes())
                    .ok_or_else(|| EngineError::layout("resident upload byte total overflows"))?;
                if let Some(progress) = progress {
                    let total = expected_upload_bytes
                        .checked_add(progress_tail_upload_bytes)
                        .ok_or_else(|| {
                            EngineError::layout("resident progress byte total overflows")
                        })?;
                    progress.begin_upload(total);
                }
                #[cfg(target_os = "linux")]
                let (prefault_bytes, source_prefault_ns) = {
                    let prefault_start = Instant::now();
                    let bytes = snapshot.prefault_model_shard()?;
                    (
                        bytes,
                        elapsed_ns("resident source prefault", prefault_start)?,
                    )
                };
                #[cfg(not(target_os = "linux"))]
                let (prefault_bytes, source_prefault_ns) = (0, 0);
                let (scalars, upload_bytes, upload_submissions, weight_copy_ns, preparation) = {
                    let mut sink = SelectiveWeightSink {
                        arena: &mut arena,
                        plan: &upload_plan,
                        bytes: 0,
                        submissions: 0,
                        copy_ns: 0,
                        progress,
                    };
                    let mut preparation = ResidentPreparationStats::default();
                    let scalars = load_source_weights(
                        &mut sink,
                        &stream,
                        &layout,
                        snapshot.as_ref(),
                        &mut preparation,
                    )?;
                    (
                        scalars,
                        sink.bytes,
                        sink.submissions,
                        sink.copy_ns,
                        preparation,
                    )
                };
                let weight_load_ns = elapsed_ns("resident selective weight load", weight_start)?;
                let weight_prepare_ns =
                    weight_load_ns.checked_sub(weight_copy_ns).ok_or_else(|| {
                        EngineError::layout(
                            "selective weight-copy time exceeds total weight-load time",
                        )
                    })?;
                let preparation_other_ns = weight_prepare_ns
                    .checked_sub(preparation.total_ns()?)
                    .and_then(|nanoseconds| nanoseconds.checked_sub(source_prefault_ns))
                    .ok_or_else(|| {
                        EngineError::layout(
                            "classified selective preparation exceeds total preparation",
                        )
                    })?;

                let nonweight_start = Instant::now();
                let metadata = initialize_selective_nonweights(
                    &mut arena,
                    &mut kv_arena,
                    &upload_plan,
                    &layout,
                    &stream,
                )?;
                if let Some(progress) = progress {
                    progress.submit(metadata.bytes)?;
                }
                let uploaded_bytes = upload_bytes
                    .checked_add(metadata.bytes)
                    .ok_or_else(|| EngineError::layout("resident upload byte total overflows"))?;
                let upload_submissions = upload_submissions
                    .checked_add(metadata.submissions)
                    .ok_or_else(|| {
                        EngineError::layout("resident upload submission total overflows")
                    })?;
                if uploaded_bytes != expected_upload_bytes {
                    return Err(EngineError::layout(format!(
                        "selective loader uploaded {uploaded_bytes} bytes, expected {expected_upload_bytes}",
                    )));
                }
                if let Some(progress) = progress
                    && progress_tail_upload_bytes == 0
                {
                    progress.finish_upload()?;
                }
                let arena = arena.seal(&stream)?;
                let kv_arena = kv_arena.seal(&stream)?;
                let nonweight_init_ns = elapsed_ns(
                    "resident selective non-weight initialization",
                    nonweight_start,
                )?;
                let load_stats = ResidentLoadStats {
                    mode,
                    upload_bytes: uploaded_bytes,
                    upload_submissions,
                    zeroed_bytes: upload_plan.zeroed_owner_bytes() + upload_plan.padding_bytes(),
                    pinned_stager_bytes: 0,
                    layout_plan_ns,
                    arena_allocation_ns,
                    operator_setup_ns,
                    source_prefault_ns,
                    weight_prepare_ns,
                    source_binding_ns: preparation.source_binding_ns,
                    qkv_gather_ns: preparation.qkv_gather_ns,
                    nvfp4_materialize_ns: preparation.nvfp4_materialize_ns,
                    preparation_other_ns,
                    weight_copy_ns,
                    weight_load_ns,
                    nonweight_init_ns,
                    graph_capture_ns: 0,
                    prefault_bytes,
                    borrowed_source_bytes,
                    gathered_source_bytes,
                    swizzled_source_bytes,
                    nvfp4_materialize_workers: nvfp4_scale_materialization_workers(),
                };
                (arena, kv_arena, scalars, load_stats)
            }
        };
        let graph_start = Instant::now();
        let pointers = ProgramPointers::bind(&arena, &kv_arena, &layout, &scalars)?;
        let dense_mlp_maps = DenseMlpMaps::bind_all(&stream, &pointers)?;
        let base_address = arena.base_address();
        let kv_base_address = kv_arena.base_address();
        let ops = Ops {
            norm: &norm,
            gdn_input: &gdn_input,
            gdn_prepare: &gdn_prepare,
            gdn_recurrence: &gdn_recurrence,
            gdn_state_snapshot: &gdn_state_snapshot,
            gdn_output: &gdn_output,
            attention_qkv: &attention_qkv,
            attention_qk_prepare: &attention_qk_prepare,
            paged_gqa: &paged_gqa,
            long_context_paged_gqa: &long_context_paged_gqa,
            attention_output: &attention_output,
            dense_swiglu: &dense_swiglu,
            dense_down: &dense_down,
            nvfp4_swiglu: &nvfp4_swiglu,
            nvfp4_down: &nvfp4_down,
            lm_head: &lm_head,
        };
        let graphs = capture_routes(&stream, ops, &pointers, &dense_mlp_maps)?;
        load_stats.graph_capture_ns = elapsed_ns("resident graph capture", graph_start)?;
        if let Some(progress) = progress
            && progress_tail_upload_bytes == 0
        {
            progress.finish();
        }

        Ok((
            Self {
                graphs,
                dense_mlp_maps,
                arena,
                kv_arena,
                kv_slots,
                _norm: norm,
                _gdn_input: gdn_input,
                _gdn_prepare: gdn_prepare,
                _gdn_recurrence: gdn_recurrence,
                _gdn_state_snapshot: gdn_state_snapshot,
                _gdn_output: gdn_output,
                _attention_qkv: attention_qkv,
                _attention_qk_prepare: attention_qk_prepare,
                _paged_gqa: paged_gqa,
                _long_context_paged_gqa: long_context_paged_gqa,
                _attention_output: attention_output,
                _dense_swiglu: dense_swiglu,
                _dense_down: dense_down,
                _nvfp4_swiglu: nvfp4_swiglu,
                _nvfp4_down: nvfp4_down,
                _lm_head: lm_head,
                embedding_stager,
                snapshot,
                context: context.clone(),
                layout,
                _pointers: pointers,
                base_address,
                kv_base_address,
                load_stats,
            },
            mtp_reservation,
        ))
    }

    /// Copies exact mmap-backed BF16 embedding rows into the first residual plane.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        require_rows(token_ids.len())?;
        self.stage_embedding_rows(stream, token_ids)
    }

    /// Copies exact BF16 embedding rows for one lane-major target transaction.
    pub fn stage_target_mtp_segmented_embeddings(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
    ) -> EngineResult<()> {
        if !(1..=TARGET_VERIFY_ROWS).contains(&token_ids.len()) {
            return Err(EngineError::route(format!(
                "segmented target MTP embedding rows {} are outside 1..={TARGET_VERIFY_ROWS}",
                token_ids.len()
            )));
        }
        self.stage_embedding_rows(stream, token_ids)
    }

    fn stage_embedding_rows(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        let embedding = TextEndpointBindings::bind_embedding(self.snapshot.as_ref())?;
        let active = product(
            "resident active embedding elements",
            token_ids.len(),
            Qwen38_27B::HIDDEN,
        )?;

        for (row, &token) in token_ids.iter().enumerate() {
            let token = usize::try_from(token)
                .map_err(|_| EngineError::route("token identifier exceeds host width"))?;
            if token >= Qwen38_27B::VOCAB {
                return Err(EngineError::route(format!(
                    "token identifier {token} is outside vocabulary 0..{}",
                    Qwen38_27B::VOCAB
                )));
            }
            copy_embedding_row(
                embedding.bytes(),
                token,
                &mut self.embedding_stager
                    [row * Qwen38_27B::HIDDEN..(row + 1) * Qwen38_27B::HIDDEN],
            )?;
        }
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.workspace.residual_a,
            &self.embedding_stager[..active],
        )?;

        Ok(())
    }

    /// Uploads represented BF16 residual rows directly for source-backed qualification.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_rows(rows)?;
        let expected = product("resident residual elements", rows, Qwen38_27B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "resident residual input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.workspace.residual_a, values)?;

        Ok(())
    }

    /// Updates shared positions, lengths, and 32 MRoPE pairs for every attention layer.
    pub fn load_decode_state(
        &self,
        stream: &CudaStream,
        batch: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentDecodeRoute> {
        require_batch(batch)?;
        if positions.len() != batch {
            return Err(EngineError::layout(format!(
                "resident positions have {} values, expected {batch}",
                positions.len()
            )));
        }
        let rotary_values = product("resident rotary values", batch, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "resident rotary planes must each have {rotary_values} values for B={batch}"
            )));
        }
        let lengths = decode_lengths(positions, self.layout.context_capacity())?;
        let workspace = self.layout.workspace;
        self.arena
            .copy_prefix_from_host(stream, workspace.cache_positions, positions)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.lengths, &lengths[..batch])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_sin, rope_sin)?;

        select_decode_route(batch, &lengths[..batch])
    }

    /// Loads one exact provisional target-verification window for a stable slot.
    pub fn load_target_mtp_verify_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpVerifyRoute> {
        require_target_verify_tokens(tokens)?;
        require_slot(slot)?;
        let context_tokens = first_position
            .checked_add(tokens)
            .ok_or_else(|| EngineError::route("target MTP verification context overflows"))?;
        if context_tokens > self.context_capacity() {
            return Err(EngineError::route(format!(
                "target MTP verification requires {context_tokens} positions, current resident capacity is {}",
                self.context_capacity()
            )));
        }
        let reserved_tokens = self
            .kv_slots
            .page_count(slot)?
            .checked_mul(ATTENTION_PAGE_SIZE)
            .ok_or_else(|| EngineError::layout("target MTP reserved capacity overflows"))?;
        if reserved_tokens < context_tokens {
            return Err(EngineError::route(format!(
                "target MTP slot {slot} owns {reserved_tokens} cache positions, expected at least {context_tokens}"
            )));
        }
        let rotary_values = product("target MTP rotary values", tokens, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "target MTP rotary planes must each have {rotary_values} values for K={tokens}"
            )));
        }

        let slot_row =
            u32::try_from(slot).map_err(|_| EngineError::layout("target MTP slot exceeds u32"))?;
        let mut positions = [0u32; TARGET_VERIFY_ROUTE_COUNT];
        let mut lengths = [0u32; TARGET_VERIFY_ROUTE_COUNT];
        let rows = [slot_row; TARGET_VERIFY_ROUTE_COUNT];
        for token in 0..tokens {
            let position = first_position
                .checked_add(token)
                .and_then(|position| u32::try_from(position).ok())
                .ok_or_else(|| EngineError::route("target MTP position exceeds u32"))?;
            positions[token] = position;
            lengths[token] = position + 1;
        }
        let decode = select_decode_route(tokens, &lengths[..tokens])?;
        let workspace = self.layout.workspace;
        self.arena
            .copy_prefix_from_host(stream, workspace.state_rows, &rows[..tokens])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.table_rows, &rows[..tokens])?;
        self.arena.copy_prefix_from_host(
            stream,
            workspace.cache_positions,
            &positions[..tokens],
        )?;
        self.arena
            .copy_prefix_from_host(stream, workspace.lengths, &lengths[..tokens])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_sin, rope_sin)?;

        Ok(ResidentMtpVerifyRoute {
            tokens,
            slot,
            maximum_length: decode.maximum_length,
            attention: decode.attention,
        })
    }

    /// Loads one exact lane-major provisional target-verification transaction.
    pub fn load_target_mtp_segmented_verify_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slots: &[usize],
        first_positions: &[usize],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpSegmentedVerifyRoute> {
        require_target_verify_tokens(tokens)?;
        let slot_ids = slot_rows(slots)?;
        if first_positions.len() != slots.len() {
            return Err(EngineError::layout(format!(
                "segmented target MTP has {} first positions, expected B={}",
                first_positions.len(),
                slots.len()
            )));
        }
        let rows = product("segmented target MTP rows", slots.len(), tokens)?;
        let rotary_values = product("segmented target MTP rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "segmented target MTP rotary planes must each have {rotary_values} values for B={} K={tokens}",
                slots.len()
            )));
        }

        let mut state_rows = [0u32; TARGET_VERIFY_ROWS];
        let mut table_rows = [0u32; TARGET_VERIFY_ROWS];
        let mut positions = [0u32; TARGET_VERIFY_ROWS];
        let mut lengths = [0u32; TARGET_VERIFY_ROWS];
        let mut lane_lengths = [0u32; MAX_BATCH];
        for lane in 0..slots.len() {
            let context_tokens = first_positions[lane].checked_add(tokens).ok_or_else(|| {
                EngineError::route("segmented target MTP verification context overflows")
            })?;
            if context_tokens > self.context_capacity() {
                return Err(EngineError::route(format!(
                    "segmented target MTP lane {lane} requires {context_tokens} positions, current resident capacity is {}",
                    self.context_capacity()
                )));
            }
            let reserved_tokens = self
                .kv_slots
                .page_count(slots[lane])?
                .checked_mul(ATTENTION_PAGE_SIZE)
                .ok_or_else(|| EngineError::layout("segmented target MTP capacity overflows"))?;
            if reserved_tokens < context_tokens {
                return Err(EngineError::route(format!(
                    "segmented target MTP slot {} owns {reserved_tokens} cache positions, expected at least {context_tokens}",
                    slots[lane]
                )));
            }
            lane_lengths[lane] = u32::try_from(context_tokens)
                .map_err(|_| EngineError::route("segmented target MTP length exceeds u32"))?;
            for token in 0..tokens {
                let row = lane * tokens + token;
                let position = first_positions[lane]
                    .checked_add(token)
                    .and_then(|position| u32::try_from(position).ok())
                    .ok_or_else(|| {
                        EngineError::route("segmented target MTP position exceeds u32")
                    })?;
                state_rows[row] = slot_ids[lane];
                table_rows[row] = slot_ids[lane];
                positions[row] = position;
                lengths[row] = position + 1;
            }
        }
        let route =
            select_segmented_target_route(tokens, slots.len(), &lane_lengths[..slots.len()])?;
        let workspace = self.layout.workspace;
        self.arena
            .copy_prefix_from_host(stream, workspace.state_rows, &state_rows[..rows])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.table_rows, &table_rows[..rows])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.cache_positions, &positions[..rows])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.lengths, &lengths[..rows])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_sin, rope_sin)?;

        Ok(route)
    }

    /// Loads one exact from-empty prompt tile into one active slot.
    pub fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentPrefillRoute> {
        self.load_prefill_tile_state(stream, tokens, slot, 0, rope_cos, rope_sin)
    }

    /// Loads one exact contiguous prompt tile after an already processed prefix.
    pub fn load_prefill_tile_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentPrefillRoute> {
        let route = select_prefill_route(tokens, first_position, self.context_capacity())?;
        require_slot(slot)?;
        let reserved_tokens = self
            .kv_slots
            .page_count(slot)?
            .checked_mul(ATTENTION_PAGE_SIZE)
            .ok_or_else(|| EngineError::layout("resident prefill page capacity overflows"))?;
        if reserved_tokens < route.context_tokens {
            return Err(EngineError::route(format!(
                "resident prefill slot {slot} owns {reserved_tokens} cache positions, expected at least {}",
                route.context_tokens
            )));
        }
        let rotary_values = product("resident prefill rotary values", tokens, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "resident prefill rotary planes must each have {rotary_values} values for T={tokens}"
            )));
        }

        let slot = u32::try_from(slot)
            .map_err(|_| EngineError::layout("resident prefill slot exceeds u32"))?;
        let mut positions = [0u32; super::MAX_ROWS];
        let mut lengths = [0u32; super::MAX_ROWS];
        let mut rows = [0u32; super::MAX_ROWS];
        for token in 0..tokens {
            let position = first_position
                .checked_add(token)
                .and_then(|position| u32::try_from(position).ok())
                .ok_or_else(|| EngineError::route("resident prefill position exceeds u32"))?;
            positions[token] = position;
            lengths[token] = position + 1;
            rows[token] = slot;
        }
        let workspace = self.layout.workspace;
        self.arena
            .copy_prefix_from_host(stream, workspace.state_rows, &rows[..tokens])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.table_rows, &rows[..tokens])?;
        self.arena.copy_prefix_from_host(
            stream,
            workspace.cache_positions,
            &positions[..tokens],
        )?;
        self.arena
            .copy_prefix_from_host(stream, workspace.lengths, &lengths[..tokens])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_sin, rope_sin)?;

        Ok(route)
    }

    /// Selects distinct physical persistent slots for the compact active rows.
    pub fn load_slot_routes(&self, stream: &CudaStream, slots: &[usize]) -> EngineResult<()> {
        let rows = slot_rows(slots)?;
        let workspace = self.layout.workspace;
        self.arena
            .copy_prefix_from_host(stream, workspace.state_rows, &rows[..slots.len()])?;
        self.arena
            .copy_prefix_from_host(stream, workspace.table_rows, &rows[..slots.len()])?;

        Ok(())
    }

    /// Activates one vacant or retained stable page-table row.
    pub fn activate_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.kv_slots.activate(slot)
    }

    /// Extends one active row and synchronizes only its stable device table row.
    pub fn reserve_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        let update = self.reserve_kv_slot_tokens_unpublished(stream, slot, token_count)?;
        self.publish_kv_slot_update(stream, update)?;
        Ok(update)
    }

    pub(crate) fn reserve_kv_slot_tokens_unpublished(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        let update = self.kv_slots.reserve_tokens(slot, token_count)?;
        for logical_page in update.first_entry()..update.first_entry() + update.entry_count() {
            let position = product(
                "resident newly assigned cache position",
                logical_page,
                ATTENTION_PAGE_SIZE,
            )?;
            let physical_page = usize::try_from(
                self.kv_slots.route(slot, position)?.physical_page(),
            )
            .map_err(|_| EngineError::layout("resident physical page exceeds host width"))?;
            clear_physical_cache_page(&self.kv_arena, stream, &self.layout, physical_page)?;
        }
        Ok(update)
    }

    pub(crate) fn publish_kv_slot_update(
        &self,
        stream: &CudaStream,
        update: PagedKvTableUpdate,
    ) -> EngineResult<()> {
        if !update.is_empty() {
            self.sync_kv_table_row(stream, update.slot())?;
        }
        Ok(())
    }

    /// Releases trailing pages while retaining an exact processed-token boundary.
    pub fn truncate_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<usize> {
        let released = self.kv_slots.truncate_tokens(slot, token_count)?;
        if released != 0 {
            self.sync_kv_table_row(stream, slot)?;
        }
        Ok(released)
    }

    /// Marks one active page-table row as an exact reusable prefix.
    pub fn retain_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.kv_slots.retain(slot)
    }

    /// Releases every page owned by one active or retained row.
    pub fn recycle_kv_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        let released = self.kv_slots.recycle(slot)?;
        if released != 0 {
            self.sync_kv_table_row(stream, slot)?;
        }
        Ok(released)
    }

    /// Clears all recurrent owners and the complete shared physical cache pool.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        for layer in &self.layout.layers {
            if let super::PersistentState::Gdn(state) = layer.persistent {
                self.arena.fill(stream, state.history, 0)?;
                self.arena.fill(stream, state.state, 0)?;
            }
        }
        for cache in self.layout.kv_layout.layers() {
            self.kv_arena.fill(stream, cache.key.data, 0)?;
            self.kv_arena.fill(stream, cache.value.data, 0)?;
        }

        Ok(())
    }

    /// Clears one physical slot without changing any other persistent owner bytes.
    pub fn reset_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        let mut attention_layer = 0;
        for layer in &self.layout.layers {
            match layer.persistent {
                super::PersistentState::Gdn(state) => {
                    fill_slot(&self.arena, stream, state.history, slot)?;
                    fill_slot(&self.arena, stream, state.state, slot)?;
                }
                super::PersistentState::Attention => attention_layer += 1,
            }
        }
        if attention_layer != self.layout.kv_layout.layers().len() {
            return Err(EngineError::layout(
                "resident slot reset did not visit every shared KV layer",
            ));
        }
        self.clear_slot_cache(stream, slot)?;

        Ok(())
    }

    /// Captures one slot's exact GDN history and recurrent state into its stable host row.
    pub fn capture_gdn_slot(
        &self,
        stream: &CudaStream,
        slot: usize,
        history: &mut PinnedHostBuffer<u16>,
        state: &mut PinnedHostBuffer<f32>,
    ) -> EngineResult<()> {
        require_slot(slot)?;
        require_gdn_snapshot_buffers(self, history, state)?;
        let mut history_offset = product(
            "resident GDN snapshot history slot offset",
            slot,
            self.gdn_slot_history_values(),
        )?;
        let mut state_offset = product(
            "resident GDN snapshot state slot offset",
            slot,
            self.gdn_slot_state_values(),
        )?;
        for layer in &self.layout.layers {
            let super::PersistentState::Gdn(persistent) = layer.persistent else {
                continue;
            };
            let history_values = persistent.history.len() / MAX_BATCH;
            let state_values = persistent.state.len() / MAX_BATCH;
            // SAFETY: the exact slot slices and pinned destination rows remain stable until the
            // single synchronization after all 48 layer copies.
            unsafe {
                self.arena.copy_slice_to_pinned_host_async(
                    stream,
                    persistent.history,
                    slot * history_values,
                    history,
                    history_offset,
                    history_values,
                )?;
                self.arena.copy_slice_to_pinned_host_async(
                    stream,
                    persistent.state,
                    slot * state_values,
                    state,
                    state_offset,
                    state_values,
                )?;
            }
            history_offset += history_values;
            state_offset += state_values;
        }
        debug_assert_eq!(history_offset, (slot + 1) * self.gdn_slot_history_values());
        debug_assert_eq!(state_offset, (slot + 1) * self.gdn_slot_state_values());
        stream.synchronize().map_err(GpuError::from)?;
        Ok(())
    }

    /// Restores one slot's exact GDN history and recurrent state from its stable host row.
    pub fn restore_gdn_slot(
        &self,
        stream: &CudaStream,
        slot: usize,
        history: &PinnedHostBuffer<u16>,
        state: &PinnedHostBuffer<f32>,
    ) -> EngineResult<()> {
        require_slot(slot)?;
        require_gdn_snapshot_buffers(self, history, state)?;
        let mut history_offset = product(
            "resident GDN restore history slot offset",
            slot,
            self.gdn_slot_history_values(),
        )?;
        let mut state_offset = product(
            "resident GDN restore state slot offset",
            slot,
            self.gdn_slot_state_values(),
        )?;
        for layer in &self.layout.layers {
            let super::PersistentState::Gdn(persistent) = layer.persistent else {
                continue;
            };
            let history_values = persistent.history.len() / MAX_BATCH;
            let state_values = persistent.state.len() / MAX_BATCH;
            // SAFETY: the pinned source rows remain immutable and address-stable. Subsequent
            // work consumes the restored state on the same ordered stream.
            unsafe {
                self.arena.copy_slice_from_pinned_host_async(
                    stream,
                    persistent.history,
                    slot * history_values,
                    history,
                    history_offset,
                    history_values,
                )?;
                self.arena.copy_slice_from_pinned_host_async(
                    stream,
                    persistent.state,
                    slot * state_values,
                    state,
                    state_offset,
                    state_values,
                )?;
            }
            history_offset += history_values;
            state_offset += state_values;
        }
        debug_assert_eq!(history_offset, (slot + 1) * self.gdn_slot_history_values());
        debug_assert_eq!(state_offset, (slot + 1) * self.gdn_slot_state_values());
        Ok(())
    }

    fn clear_slot_cache(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        let pages = self.kv_slots.page_count(slot)?;
        for logical_page in 0..pages {
            let position = product(
                "resident assigned cache position",
                logical_page,
                ATTENTION_PAGE_SIZE,
            )?;
            let physical_page = usize::try_from(
                self.kv_slots.route(slot, position)?.physical_page(),
            )
            .map_err(|_| EngineError::layout("resident physical page exceeds host width"))?;
            clear_physical_cache_page(&self.kv_arena, stream, &self.layout, physical_page)?;
        }
        Ok(())
    }

    fn sync_kv_table_row(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        let start = product(
            "resident block-table row offset",
            slot,
            LONG_CONTEXT_PHYSICAL_PAGES,
        )?;
        self.kv_arena.copy_slice_from_host(
            stream,
            self.layout.kv_layout.block_tables(),
            start,
            self.kv_slots.page_table(slot)?,
        )?;
        Ok(())
    }

    pub(crate) fn validate_mtp_prompt_cache(
        &self,
        slot: usize,
        first_position: usize,
        rows: usize,
    ) -> EngineResult<()> {
        require_slot(slot)?;
        let context_tokens = first_position
            .checked_add(rows)
            .ok_or_else(|| EngineError::route("MTP prompt cache position overflows"))?;
        if context_tokens > self.context_capacity() {
            return Err(EngineError::route(format!(
                "MTP prompt cache requires {context_tokens} positions, current resident capacity is {}",
                self.context_capacity()
            )));
        }
        let reserved_tokens = self
            .kv_slots
            .page_count(slot)?
            .checked_mul(ATTENTION_PAGE_SIZE)
            .ok_or_else(|| EngineError::layout("MTP prompt reserved capacity overflows"))?;
        if reserved_tokens < context_tokens {
            return Err(EngineError::route(format!(
                "MTP prompt slot {slot} owns {reserved_tokens} cache positions, expected at least {context_tokens}"
            )));
        }

        Ok(())
    }

    /// Replays the immutable graph selected by the matching decode-state upload.
    pub fn replay(&self, stream: &CudaStream, route: ResidentDecodeRoute) -> EngineResult<()> {
        // SAFETY: this ResidentModelProgram owns every captured allocation
        // (resident and KV arenas, TMA maps, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.select(route).launch(stream) }?;

        Ok(())
    }

    /// Replays one immutable from-empty prompt graph.
    pub fn replay_prefill(
        &self,
        stream: &CudaStream,
        route: ResidentPrefillRoute,
    ) -> EngineResult<()> {
        // SAFETY: this ResidentModelProgram owns every captured allocation
        // (resident and KV arenas, TMA maps, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.select_prefill(route)?.launch(stream) }?;
        Ok(())
    }

    /// Replays one immutable provisional target-verification graph.
    pub fn replay_target_mtp_verify(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
    ) -> EngineResult<()> {
        // SAFETY: this ResidentModelProgram owns every captured allocation
        // (resident and KV arenas, TMA maps, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.select_target_verify(route).launch(stream) }?;
        Ok(())
    }

    /// Replays one immutable exact lane-major target-verification graph.
    pub fn replay_target_mtp_segmented_verify(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<()> {
        // SAFETY: this ResidentModelProgram owns every captured allocation
        // (resident and KV arenas, TMA maps, op modules) for its whole life and
        // drops the graph inventory first.
        unsafe { self.graphs.launch_target_segmented_verify(stream, route) }?;
        Ok(())
    }

    /// Commits an accepted prefix from the matching provisional verification.
    pub fn replay_target_mtp_commit(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
        accepted_tokens: usize,
    ) -> EngineResult<()> {
        if accepted_tokens > route.tokens {
            return Err(EngineError::route(format!(
                "target MTP commit accepts {accepted_tokens} rows from a K={} verification",
                route.tokens
            )));
        }
        let graph = self.graphs.select_target_commit(accepted_tokens)?;
        // SAFETY: this ResidentModelProgram owns every captured allocation
        // (resident and KV arenas, TMA maps, op modules) for its whole life and
        // drops the graphs first.
        unsafe { graph.launch(stream) }?;
        Ok(())
    }

    /// Commits one accepted target-input prefix per lane from a segmented verification.
    pub fn commit_target_mtp_segmented(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
        accepted_tokens: &[usize],
    ) -> EngineResult<()> {
        require_segmented_commit(route, accepted_tokens)?;
        launch_target_mtp_segmented_commit(
            stream,
            route,
            accepted_tokens,
            self.ops(),
            &self._pointers,
        )?;
        Ok(())
    }

    /// Reads active BF16 vocabulary logits.
    pub fn read_logits(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = product("resident logit elements", batch, Qwen38_27B::VOCAB)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.workspace.logits, values)?)
    }

    /// Reads every lane-major target logit row from one segmented verification.
    pub fn read_target_mtp_segmented_logits(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<Vec<u16>> {
        let values = product(
            "segmented target MTP logit elements",
            route.rows(),
            Qwen38_27B::VOCAB,
        )?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.workspace.logits, values)?)
    }

    /// Reads every lane-major target logit row into reusable host storage.
    pub fn read_target_mtp_segmented_logits_into(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        let expected = product(
            "segmented target MTP logit elements",
            route.rows(),
            Qwen38_27B::VOCAB,
        )?;
        if destination.len() != expected {
            return Err(EngineError::layout(format!(
                "segmented target MTP logit destination has {} values, expected {expected} for B={} K={}",
                destination.len(),
                route.batch(),
                route.tokens()
            )));
        }
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.workspace.logits, destination)?;
        Ok(())
    }

    /// Preserves every lane-major final target residual before per-lane MTP realignment.
    pub fn backup_target_mtp_segmented_residuals(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<()> {
        let values = product(
            "segmented target MTP residual backup values",
            route.rows(),
            Qwen38_27B::HIDDEN,
        )?;
        let workspace = self.layout.workspace;
        // SAFETY: residual A and B are disjoint, address-stable maximum-row planes owned by this
        // program. Stream order keeps the backup ahead of every later lane selection.
        unsafe {
            self.arena.copy_prefix_from_arena_async(
                stream,
                workspace.residual_b,
                &self.arena,
                workspace.residual_a,
                values,
            )?;
        }
        Ok(())
    }

    /// Selects one backed-up lane prefix for the existing exact-K MTP realignment graph.
    pub fn select_target_mtp_segmented_residual_lane(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
        lane: usize,
        rows: usize,
    ) -> EngineResult<()> {
        if lane >= route.batch {
            return Err(EngineError::route(format!(
                "segmented target MTP residual lane {lane} is outside B={}",
                route.batch
            )));
        }
        if !(1..=route.tokens).contains(&rows) {
            return Err(EngineError::route(format!(
                "segmented target MTP residual lane selects {rows} rows from K={}",
                route.tokens
            )));
        }
        let source_row = product(
            "segmented target MTP residual source row",
            lane,
            route.tokens,
        )?;
        let source = product(
            "segmented target MTP residual source values",
            source_row,
            Qwen38_27B::HIDDEN,
        )?;
        let values = product(
            "segmented target MTP selected residual values",
            rows,
            Qwen38_27B::HIDDEN,
        )?;
        let workspace = self.layout.workspace;
        // SAFETY: the complete lane-major plane was copied to disjoint residual B first. The
        // destination prefix is consumed before another selection and cannot corrupt the backup.
        unsafe {
            self.arena.copy_slice_from_arena_async(
                stream,
                workspace.residual_a,
                0,
                &self.arena,
                workspace.residual_b,
                source,
                values,
            )?;
        }
        Ok(())
    }

    /// Reads the final-token BF16 vocabulary logits from one prefill graph.
    pub fn read_prefill_logits(&self, stream: &CudaStream) -> EngineResult<Vec<u16>> {
        Ok(self.arena.copy_prefix_to_host(
            stream,
            self.layout.workspace.logits,
            Qwen38_27B::VOCAB,
        )?)
    }

    #[cfg(feature = "qualification")]
    /// Reads every raw target residual row produced by one exact prefill route.
    pub fn qualification_prefill_residual(
        &self,
        stream: &CudaStream,
        route: ResidentPrefillRoute,
    ) -> EngineResult<Vec<u16>> {
        let values = product(
            "resident prefill residual elements",
            route.tokens,
            Qwen38_27B::HIDDEN,
        )?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.workspace.residual_a, values)?)
    }

    /// Reads active BF16 logits into one reusable host allocation.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let expected = product("resident logit elements", batch, Qwen38_27B::VOCAB)?;
        if destination.len() != expected {
            return Err(EngineError::layout(format!(
                "resident logit destination has {} values, expected {expected} for B={batch}",
                destination.len()
            )));
        }
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.workspace.logits, destination)?;

        Ok(())
    }

    /// Reads the final active BF16 residual rows before final normalization.
    pub fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = product("resident output elements", batch, Qwen38_27B::HIDDEN)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.workspace.residual_a, values)?)
    }

    /// Reads one exact final target residual row into reusable host storage.
    pub fn read_residual_row_into(
        &self,
        stream: &CudaStream,
        row: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        if row >= TARGET_VERIFY_ROWS {
            return Err(EngineError::route(format!(
                "resident target residual row {row} is outside 0..{TARGET_VERIFY_ROWS}"
            )));
        }
        if destination.len() != Qwen38_27B::HIDDEN {
            return Err(EngineError::layout(format!(
                "resident target residual-row destination has {} values, expected {}",
                destination.len(),
                Qwen38_27B::HIDDEN
            )));
        }
        let start = product(
            "resident target residual-row offset",
            row,
            Qwen38_27B::HIDDEN,
        )?;
        self.arena.copy_slice_to_host_slice(
            stream,
            self.layout.workspace.residual_a,
            start,
            destination,
        )?;
        Ok(())
    }

    /// Enqueues the raw target residual and current page table into an MTP prompt owner.
    ///
    /// # Safety
    ///
    /// The destination arena must remain live until the stream reaches both copies. If these
    /// copies are captured in a graph, this resident owner and the destination arena must keep
    /// their addresses stable through the final replay.
    pub(crate) unsafe fn enqueue_mtp_prompt_handoff(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &DeviceArena,
        target_hidden: ArenaRegion<u16>,
        block_tables: ArenaRegion<u32>,
    ) -> GpuResult<()> {
        if !(1..=super::MAX_ROWS).contains(&rows) {
            return Err(GpuError::invalid_launch(format!(
                "MTP prompt handoff rows {rows} are outside 1..={}",
                super::MAX_ROWS
            )));
        }
        let hidden_values = rows.checked_mul(Qwen38_27B::HIDDEN).ok_or_else(|| {
            GpuError::invalid_launch("MTP prompt target-hidden element count overflows")
        })?;
        // SAFETY: the checked resident and destination regions remain owned by the two programs;
        // the caller supplies the graph/stream lifetime promised by this method.
        unsafe {
            destination.copy_prefix_from_arena_async(
                stream,
                target_hidden,
                &self.arena,
                self.layout.workspace.residual_a,
                hidden_values,
            )?;
            self.enqueue_mtp_block_table_handoff(stream, destination, block_tables)?;
        }

        Ok(())
    }

    /// Enqueues the current page table without changing an MTP hidden-input plane.
    ///
    /// # Safety
    ///
    /// The destination arena must remain live until the stream reaches the copy. Captured callers
    /// must retain both owners at stable addresses through the final replay.
    pub(crate) unsafe fn enqueue_mtp_block_table_handoff(
        &self,
        stream: &CudaStream,
        destination: &DeviceArena,
        block_tables: ArenaRegion<u32>,
    ) -> GpuResult<()> {
        unsafe {
            destination.copy_prefix_from_arena_async(
                stream,
                block_tables,
                &self.kv_arena,
                self.layout.kv_layout.block_tables(),
                self.layout.kv_layout.block_tables().len(),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "qualification")]
    pub(crate) fn qualification_mtp_prompt_source_addresses(&self) -> GpuResult<[usize; 2]> {
        Ok([
            self.arena.address(self.layout.workspace.residual_a)?.addr(),
            self.kv_arena
                .address(self.layout.kv_layout.block_tables())?
                .addr(),
        ])
    }

    /// CUDA context shared by the arena, graphs, and prepared operators.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen38_27B>> {
        &self.snapshot
    }

    pub(crate) const fn mtp_lm_head_op(&self) -> &LmHeadOp<Qwen38_27B> {
        &self._lm_head
    }

    pub(crate) const fn mtp_lm_head_weights(&self) -> (*const u8, *const u16) {
        (
            self._pointers.endpoint.lm_head_codes,
            self._pointers.endpoint.lm_head_scales,
        )
    }

    pub(crate) fn mtp_kv_page_count(&self, slot: usize) -> EngineResult<usize> {
        self.kv_slots.page_count(slot)
    }

    pub(crate) fn mtp_kv_token_count(&self, slot: usize) -> EngineResult<usize> {
        self.kv_slots.token_count(slot)
    }

    pub(crate) fn mtp_kv_physical_page(
        &self,
        slot: usize,
        logical_page: usize,
    ) -> EngineResult<usize> {
        let position = product(
            "resident MTP logical-page position",
            logical_page,
            ATTENTION_PAGE_SIZE,
        )?;
        usize::try_from(self.kv_slots.route(slot, position)?.physical_page())
            .map_err(|_| EngineError::layout("resident MTP physical page exceeds host width"))
    }

    /// Stable base address captured by all exact-batch graphs.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable base address of the shared page-table and KV arena.
    pub const fn kv_base_address(&self) -> u64 {
        self.kv_base_address
    }

    /// Exact allocation and transfer work used to make this program resident.
    pub const fn load_stats(&self) -> ResidentLoadStats {
        self.load_stats
    }

    /// Exact source-backed device weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact causal-history bytes across the 48 GDN layers.
    pub const fn history_bytes(&self) -> usize {
        self.layout.history_bytes()
    }

    /// Exact recurrent-state bytes across the 48 GDN layers.
    pub const fn state_bytes(&self) -> usize {
        self.layout.state_bytes()
    }

    /// Exact represented KV-cache bytes across the 16 attention layers.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Stable long-context page-table bytes across all eight slot rows.
    pub const fn kv_table_bytes(&self) -> usize {
        self.layout.kv_table_bytes()
    }

    /// Fixed host bytes owning all stable table rows and physical-page tags.
    pub const fn kv_route_host_bytes(&self) -> usize {
        self.kv_slots.host_allocation_bytes()
    }

    /// Fixed GDN history and recurrent-state bytes owned by one slot.
    ///
    /// KV pages are drawn from the shared pool and therefore have no fixed
    /// per-slot ownership byte count.
    pub const fn persistent_slot_bytes(&self) -> usize {
        (self.layout.history_bytes() + self.layout.state_bytes()) / MAX_BATCH
    }

    /// BF16 history values in one exact GDN slot snapshot.
    pub const fn gdn_slot_history_values(&self) -> usize {
        self.layout.history_bytes() / MAX_BATCH / std::mem::size_of::<u16>()
    }

    /// FP32 recurrent values in one exact GDN slot snapshot.
    pub const fn gdn_slot_state_values(&self) -> usize {
        self.layout.state_bytes() / MAX_BATCH / std::mem::size_of::<f32>()
    }

    /// Exact address-stable workspace bytes shared by every layer and endpoint.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Exact address-bound tensor-map bytes across the eight dense MLP layers.
    pub fn descriptor_bytes(&self) -> usize {
        self.dense_mlp_maps.iter().map(DenseMlpMaps::byte_len).sum()
    }

    /// Complete device arena bytes, including exact alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Weight, GDN-state, and shared-workspace allocation bytes.
    pub const fn resident_arena_bytes(&self) -> usize {
        self.layout.resident_arena_bytes()
    }

    /// Shared page-table and represented KV allocation bytes.
    pub const fn kv_arena_bytes(&self) -> usize {
        self.layout.kv_arena_bytes()
    }

    /// Exact alignment padding inside the complete device arena.
    pub const fn padding_bytes(&self) -> usize {
        self.layout.padding_bytes()
    }

    /// Page-locked host bytes used to stage selected embedding rows.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
    }

    /// Largest admitted compact decode batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Largest admitted exact prompt tile.
    pub const fn row_capacity(&self) -> usize {
        super::MAX_ROWS
    }

    /// Immutable singleton/segmented provisional-verify and accepted-prefix graph entries.
    pub const fn target_mtp_graph_count(&self) -> usize {
        TARGET_VERIFY_ROUTE_COUNT
            * (LONG_CONTEXT_ROUTE_COUNT
                + 2
                + TARGET_SEGMENTED_BATCH_ROUTES * (LONG_CONTEXT_ROUTE_COUNT + 1))
    }

    /// Executable instances retained for the complete target MTP route inventory.
    pub const fn target_mtp_graph_executable_count(&self) -> usize {
        self.graphs.target_mtp_executable_count()
    }

    /// Complete resident route-definition inventory.
    pub const fn graph_route_count(&self) -> usize {
        self.graphs.route_count()
    }

    /// Complete resident executable-graph ownership after compatible variant sharing.
    pub const fn graph_executable_count(&self) -> usize {
        self.graphs.executable_count()
    }

    /// Fixed per-slot capacity of the current exact attention caches.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Checked exact-target resident layout.
    pub const fn layout(&self) -> &ResidentModelLayout {
        &self.layout
    }

    #[cfg(feature = "qualification")]
    /// Launches the complete production schedule eagerly for graph-agreement checks.
    pub fn launch_eager(
        &self,
        stream: &CudaStream,
        route: ResidentDecodeRoute,
    ) -> EngineResult<()> {
        launch_route(
            stream,
            route,
            self.ops(),
            &self._pointers,
            &self.dense_mlp_maps,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete prefill schedule eagerly for graph agreement.
    pub fn launch_prefill_eager(
        &self,
        stream: &CudaStream,
        route: ResidentPrefillRoute,
    ) -> EngineResult<()> {
        launch_prefill_route(
            stream,
            route,
            self.ops(),
            &self._pointers,
            &self.dense_mlp_maps,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one provisional target-verification schedule eagerly.
    pub fn launch_target_mtp_verify_eager(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
    ) -> EngineResult<()> {
        launch_target_mtp_verify(
            stream,
            route,
            self.ops(),
            &self._pointers,
            &self.dense_mlp_maps,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one exact lane-major target-verification schedule eagerly.
    pub fn launch_target_mtp_segmented_verify_eager(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<()> {
        launch_target_mtp_segmented_verify(
            stream,
            route,
            self.ops(),
            &self._pointers,
            &self.dense_mlp_maps,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one accepted-prefix state commit eagerly.
    pub fn launch_target_mtp_commit_eager(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
        accepted_tokens: usize,
    ) -> EngineResult<()> {
        if !(1..=route.tokens).contains(&accepted_tokens) {
            return Err(EngineError::route(format!(
                "target MTP eager commit accepts {accepted_tokens} rows from a K={} verification",
                route.tokens
            )));
        }
        launch_target_mtp_commit(stream, accepted_tokens, self.ops(), &self._pointers)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns the captured complete-model graph for one checked state route.
    pub fn qualification_graph(&self, route: ResidentDecodeRoute) -> &CudaGraph {
        self.graphs.select(route)
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured complete-model prefill graph.
    pub fn qualification_prefill_graph(
        &self,
        route: ResidentPrefillRoute,
    ) -> EngineResult<&CudaGraph> {
        self.graphs.select_prefill(route)
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured provisional target-verification graph.
    pub fn qualification_target_mtp_verify_graph(
        &self,
        route: ResidentMtpVerifyRoute,
    ) -> &CudaGraph {
        self.graphs.select_target_verify(route)
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured exact lane-major target-verification graph.
    pub fn qualification_target_mtp_segmented_verify_graph(
        &self,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<&CudaGraph> {
        self.graphs
            .select_direct_target_segmented_verify(route)
            .ok_or_else(|| {
                EngineError::route(
                    "long-context segmented target verification uses an updated graph variant",
                )
            })
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured accepted-prefix commit graph.
    pub fn qualification_target_mtp_commit_graph(
        &self,
        accepted_tokens: usize,
    ) -> EngineResult<&CudaGraph> {
        self.graphs.select_target_commit(accepted_tokens)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete-model schedules for direct intrinsic timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        route: ResidentDecodeRoute,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        if operations == 0 {
            return Err(EngineError::route(
                "repeated resident-model graph requires at least one operation",
            ));
        }
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, route, ops, &self._pointers, &self.dense_mlp_maps)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete-model prefill schedules for direct timing.
    pub fn qualification_repeated_prefill_graph(
        &self,
        stream: &CudaStream,
        route: ResidentPrefillRoute,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        if operations == 0 {
            return Err(EngineError::route(
                "repeated resident-model prefill graph requires at least one operation",
            ));
        }
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_prefill_route(stream, route, ops, &self._pointers, &self.dense_mlp_maps)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated production target verify and optional commit schedules.
    pub fn qualification_repeated_target_mtp_graph(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
        committed_tokens: Option<usize>,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        if operations == 0 {
            return Err(EngineError::route(
                "repeated target MTP graph requires at least one operation",
            ));
        }
        if committed_tokens.is_some_and(|tokens| !(1..=route.tokens).contains(&tokens)) {
            return Err(EngineError::route(
                "repeated target MTP commit exceeds its verification window",
            ));
        }
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_target_mtp_verify(
                    stream,
                    route,
                    ops,
                    &self._pointers,
                    &self.dense_mlp_maps,
                )?;
                if let Some(tokens) = committed_tokens {
                    launch_target_mtp_commit(stream, tokens, ops, &self._pointers)?;
                }
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated segmented verification and optional per-lane commit work.
    pub fn qualification_repeated_target_mtp_segmented_graph(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
        accepted_tokens: Option<&[usize]>,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        if operations == 0 {
            return Err(EngineError::route(
                "repeated segmented target MTP graph requires at least one operation",
            ));
        }
        if let Some(accepted) = accepted_tokens {
            require_segmented_commit(route, accepted)?;
        }
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_target_mtp_segmented_verify(
                    stream,
                    route,
                    ops,
                    &self._pointers,
                    &self.dense_mlp_maps,
                )?;
                if let Some(accepted) = accepted_tokens {
                    launch_target_mtp_segmented_commit(
                        stream,
                        route,
                        accepted,
                        ops,
                        &self._pointers,
                    )?;
                }
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures production embedding staging separately from model-graph timing.
    pub fn qualification_embedding_stage_graph(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<ResidentEmbeddingStageGraph<'_>> {
        require_rows(rows)?;
        let active = product(
            "resident staged embedding elements",
            rows,
            Qwen38_27B::HIDDEN,
        )?;
        let graph = CudaGraph::capture(stream, || {
            // SAFETY: the returned graph borrows `self`, so the page-locked source cannot be
            // mutated or dropped before this graph and all of its replays are finished.
            unsafe {
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    self.layout.workspace.residual_a,
                    &self.embedding_stager,
                    active,
                )
            }
        })?;
        Ok(ResidentEmbeddingStageGraph {
            graph,
            source: PhantomData,
        })
    }

    #[cfg(feature = "qualification")]
    /// Captures the production uploads that prepare one exact prefill graph replay.
    pub fn qualification_prefill_stage_graph(
        &self,
        stream: &CudaStream,
        route: ResidentPrefillRoute,
        slot: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentPrefillStageGraph<'_>> {
        require_slot(slot)?;
        prefill_graph_index(route)?;
        let rotary_values = product(
            "resident prefill stage rotary values",
            route.tokens,
            ROTARY_PAIRS,
        )?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "resident prefill stage rotary planes must each have {rotary_values} values"
            )));
        }

        let mut positions =
            PinnedHostBuffer::zeroed(&self.context, route.tokens).map_err(GpuError::from)?;
        let mut lengths =
            PinnedHostBuffer::zeroed(&self.context, route.tokens).map_err(GpuError::from)?;
        let mut rows =
            PinnedHostBuffer::zeroed(&self.context, route.tokens).map_err(GpuError::from)?;
        for token in 0..route.tokens {
            let position = u32::try_from(route.first_position + token)
                .map_err(|_| EngineError::route("resident prefill stage position exceeds u32"))?;
            positions.as_mut_slice()[token] = position;
            lengths.as_mut_slice()[token] = position + 1;
            rows.as_mut_slice()[token] = slot as u32;
        }
        let mut pinned_cos =
            PinnedHostBuffer::zeroed(&self.context, rotary_values).map_err(GpuError::from)?;
        pinned_cos.as_mut_slice().copy_from_slice(rope_cos);
        let mut pinned_sin =
            PinnedHostBuffer::zeroed(&self.context, rotary_values).map_err(GpuError::from)?;
        pinned_sin.as_mut_slice().copy_from_slice(rope_sin);

        let active_embeddings = product(
            "resident prefill stage embedding elements",
            route.tokens,
            Qwen38_27B::HIDDEN,
        )?;
        let workspace = self.layout.workspace;
        let graph = CudaGraph::capture(stream, || {
            // SAFETY: the returned owner retains every page-locked source through all replays.
            unsafe {
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.residual_a,
                    &self.embedding_stager,
                    active_embeddings,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.state_rows,
                    &rows,
                    route.tokens,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.table_rows,
                    &rows,
                    route.tokens,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.cache_positions,
                    &positions,
                    route.tokens,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.lengths,
                    &lengths,
                    route.tokens,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.rope_cos,
                    &pinned_cos,
                    rotary_values,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.rope_sin,
                    &pinned_sin,
                    rotary_values,
                )
            }
        })?;
        Ok(ResidentPrefillStageGraph {
            graph,
            _positions: positions,
            _lengths: lengths,
            _rows: rows,
            _rope_cos: pinned_cos,
            _rope_sin: pinned_sin,
            source: PhantomData,
        })
    }

    #[cfg(feature = "qualification")]
    /// Captures the production uploads for one exact segmented target graph replay.
    pub fn qualification_target_mtp_segmented_stage_graph(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
        slots: &[usize],
        first_positions: &[usize],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpSegmentedStageGraph<'_>> {
        let slot_ids = slot_rows(slots)?;
        if slots.len() != route.batch || first_positions.len() != route.batch {
            return Err(EngineError::layout(
                "segmented target stage slot/position inventory differs from its route",
            ));
        }
        let active_rows = route.rows();
        let rotary_values = product(
            "segmented target stage rotary values",
            active_rows,
            ROTARY_PAIRS,
        )?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "segmented target stage rotary planes must each have {rotary_values} values"
            )));
        }

        let mut positions =
            PinnedHostBuffer::zeroed(&self.context, active_rows).map_err(GpuError::from)?;
        let mut lengths =
            PinnedHostBuffer::zeroed(&self.context, active_rows).map_err(GpuError::from)?;
        let mut rows =
            PinnedHostBuffer::zeroed(&self.context, active_rows).map_err(GpuError::from)?;
        let mut lane_lengths = [0u32; MAX_BATCH];
        for lane in 0..route.batch {
            for token in 0..route.tokens {
                let row = lane * route.tokens + token;
                let position = first_positions[lane]
                    .checked_add(token)
                    .and_then(|position| u32::try_from(position).ok())
                    .ok_or_else(|| {
                        EngineError::route("segmented target stage position exceeds u32")
                    })?;
                positions.as_mut_slice()[row] = position;
                lengths.as_mut_slice()[row] = position + 1;
                rows.as_mut_slice()[row] = slot_ids[lane];
            }
            lane_lengths[lane] = first_positions[lane]
                .checked_add(route.tokens)
                .and_then(|length| u32::try_from(length).ok())
                .ok_or_else(|| EngineError::route("segmented target stage length exceeds u32"))?;
        }
        if select_segmented_target_route(route.tokens, route.batch, &lane_lengths[..route.batch])?
            != route
        {
            return Err(EngineError::route(
                "segmented target stage metadata selects a different graph route",
            ));
        }
        let mut pinned_cos =
            PinnedHostBuffer::zeroed(&self.context, rotary_values).map_err(GpuError::from)?;
        pinned_cos.as_mut_slice().copy_from_slice(rope_cos);
        let mut pinned_sin =
            PinnedHostBuffer::zeroed(&self.context, rotary_values).map_err(GpuError::from)?;
        pinned_sin.as_mut_slice().copy_from_slice(rope_sin);
        let active_embeddings = product(
            "segmented target stage embedding elements",
            active_rows,
            Qwen38_27B::HIDDEN,
        )?;
        let workspace = self.layout.workspace;
        let graph = CudaGraph::capture(stream, || {
            // SAFETY: this returned owner retains every pinned source and borrows
            // the resident embedding stager through the final graph replay.
            unsafe {
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.residual_a,
                    &self.embedding_stager,
                    active_embeddings,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.state_rows,
                    &rows,
                    active_rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.table_rows,
                    &rows,
                    active_rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.cache_positions,
                    &positions,
                    active_rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.lengths,
                    &lengths,
                    active_rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.rope_cos,
                    &pinned_cos,
                    rotary_values,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    workspace.rope_sin,
                    &pinned_sin,
                    rotary_values,
                )
            }
        })?;
        Ok(ResidentMtpSegmentedStageGraph {
            graph,
            _positions: positions,
            _lengths: lengths,
            _rows: rows,
            _rope_cos: pinned_cos,
            _rope_sin: pinned_sin,
            source: PhantomData,
        })
    }

    #[cfg(feature = "qualification")]
    /// Returns every immutable and mutable address captured by the owner.
    pub fn qualification_addresses(&self) -> Vec<usize> {
        let mut addresses = self._pointers.addresses();
        for maps in &self.dense_mlp_maps {
            maps.push_addresses(&mut addresses);
        }
        addresses
    }

    #[cfg(feature = "qualification")]
    /// Reads all eight stable long-context page-table rows.
    pub fn qualification_block_tables(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        Ok(self
            .kv_arena
            .copy_to_host(stream, self.layout.kv_layout.block_tables())?)
    }

    #[cfg(feature = "qualification")]
    /// Stable host addresses owned by the allocation-free page router.
    pub fn qualification_kv_route_addresses(&self) -> [usize; 2] {
        self.kv_slots.qualification_addresses()
    }

    #[cfg(feature = "qualification")]
    /// Current lifecycle state for one stable page-table row.
    pub fn qualification_kv_slot_state(&self, slot: usize) -> EngineResult<PagedKvSlotState> {
        self.kv_slots.state(slot)
    }

    #[cfg(feature = "qualification")]
    /// Current logical-page count for one stable page-table row.
    pub fn qualification_kv_page_count(&self, slot: usize) -> EngineResult<usize> {
        self.kv_slots.page_count(slot)
    }

    #[cfg(feature = "qualification")]
    /// Physical page selected for one already-owned slot position.
    pub fn qualification_kv_physical_page(
        &self,
        slot: usize,
        position: usize,
    ) -> EngineResult<u32> {
        Ok(self.kv_slots.route(slot, position)?.physical_page())
    }

    #[cfg(feature = "qualification")]
    /// Reads one physical K/V page from every full-attention layer.
    pub fn qualification_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<u8>, Vec<u8>)> {
        if physical_page >= LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::route(format!(
                "qualification cache page {physical_page} exceeds 0..{LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        let page_values = cache_values(1)?;
        let start = cache_values(physical_page)?;
        let layers = self.layout.kv_layout.layers().len();
        let mut key = Vec::with_capacity(page_values * layers);
        let mut value = Vec::with_capacity(page_values * layers);
        for cache in self.layout.kv_layout.layers() {
            key.extend(self.kv_arena.copy_slice_to_host(
                stream,
                cache.key.data,
                start,
                page_values,
            )?);
            value.extend(self.kv_arena.copy_slice_to_host(
                stream,
                cache.value.data,
                start,
                page_values,
            )?);
        }
        Ok((key, value))
    }

    #[cfg(feature = "qualification")]
    /// Loads one selected slot's complete patterned GDN history and state rows.
    pub fn qualification_load_target_mtp_gdn_slot(
        &self,
        stream: &CudaStream,
        slot: usize,
        history: &[u16],
        state: &[f32],
    ) -> EngineResult<()> {
        require_slot(slot)?;
        let expected_history = self.layout.history_bytes() / MAX_BATCH / size_of::<u16>();
        let expected_state = self.layout.state_bytes() / MAX_BATCH / size_of::<f32>();
        if history.len() != expected_history || state.len() != expected_state {
            return Err(EngineError::layout(format!(
                "target MTP slot fixture has {}/{} history/state values, expected {expected_history}/{expected_state}",
                history.len(),
                state.len()
            )));
        }
        let mut history_offset = 0;
        let mut state_offset = 0;
        for layer in &self.layout.layers {
            let super::PersistentState::Gdn(persistent) = layer.persistent else {
                continue;
            };
            let history_values = persistent.history.len() / MAX_BATCH;
            let state_values = persistent.state.len() / MAX_BATCH;
            self.arena.copy_slice_from_host(
                stream,
                persistent.history,
                slot * history_values,
                &history[history_offset..history_offset + history_values],
            )?;
            self.arena.copy_slice_from_host(
                stream,
                persistent.state,
                slot * state_values,
                &state[state_offset..state_offset + state_values],
            )?;
            history_offset += history_values;
            state_offset += state_values;
        }
        debug_assert_eq!(history_offset, expected_history);
        debug_assert_eq!(state_offset, expected_state);
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Compares one live GDN slot with its exact stable host snapshot.
    pub fn qualification_gdn_slot_matches_snapshot(
        &self,
        stream: &CudaStream,
        slot: usize,
        history: &PinnedHostBuffer<u16>,
        state: &PinnedHostBuffer<f32>,
    ) -> EngineResult<bool> {
        require_slot(slot)?;
        require_gdn_snapshot_buffers(self, history, state)?;
        let mut history_offset = slot * self.gdn_slot_history_values();
        let mut state_offset = slot * self.gdn_slot_state_values();
        for layer in &self.layout.layers {
            let super::PersistentState::Gdn(persistent) = layer.persistent else {
                continue;
            };
            let history_values = persistent.history.len() / MAX_BATCH;
            let state_values = persistent.state.len() / MAX_BATCH;
            let live_history = self.arena.copy_slice_to_host(
                stream,
                persistent.history,
                slot * history_values,
                history_values,
            )?;
            let live_state = self.arena.copy_slice_to_host(
                stream,
                persistent.state,
                slot * state_values,
                state_values,
            )?;
            if live_history != history[history_offset..history_offset + history_values]
                || live_state != state[state_offset..state_offset + state_values]
            {
                return Ok(false);
            }
            history_offset += history_values;
            state_offset += state_values;
        }
        Ok(true)
    }

    #[cfg(feature = "qualification")]
    /// Fills every mutable workspace seam with one byte sentinel.
    pub fn qualification_reset_workspace(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let workspace = self.layout.workspace;
        for region in [
            workspace.residual_a,
            workspace.residual_b,
            workspace.mixer_residual,
            workspace.mixer_normalized,
            workspace.mlp_normalized,
            workspace.mixer_branch,
            workspace.mlp_branch,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            workspace.activation_codes,
            workspace.nvfp4_activation_codes,
            workspace.nvfp4_activation_scales,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            workspace.projected,
            workspace.convolved,
            workspace.recurrent_output,
            workspace.swiglu,
            workspace.logits,
            workspace.provisional_history,
            workspace.recorded_projected,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            workspace.activation_scales,
            workspace.log_decay,
            workspace.beta,
            workspace.query,
            workspace.partial_maximum,
            workspace.partial_denominator,
            workspace.partial_numerator,
            workspace.prefill_partials,
            workspace.attention,
            workspace.provisional_state,
            workspace.recorded_log_decay,
            workspace.recorded_beta,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every target-verification seam plus the selected live GDN rows.
    pub fn qualification_target_mtp_observables(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
    ) -> EngineResult<ResidentMtpVerifyObservables> {
        let workspace = self.layout.workspace;
        let hidden_values = product(
            "target MTP observable hidden values",
            route.tokens,
            Qwen38_27B::HIDDEN,
        )?;
        let logit_values = product(
            "target MTP observable logits",
            route.tokens,
            Qwen38_27B::VOCAB,
        )?;
        let mut live_history =
            Vec::with_capacity(self.layout.history_bytes() / MAX_BATCH / size_of::<u16>());
        let mut live_state =
            Vec::with_capacity(self.layout.state_bytes() / MAX_BATCH / size_of::<f32>());
        for layer in &self.layout.layers {
            let super::PersistentState::Gdn(persistent) = layer.persistent else {
                continue;
            };
            let history_values = persistent.history.len() / MAX_BATCH;
            let state_values = persistent.state.len() / MAX_BATCH;
            live_history.extend(self.arena.copy_slice_to_host(
                stream,
                persistent.history,
                route.slot * history_values,
                history_values,
            )?);
            live_state.extend(self.arena.copy_slice_to_host(
                stream,
                persistent.state,
                route.slot * state_values,
                state_values,
            )?);
        }
        let provisional_history_values =
            Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
        let provisional_state_values = Qwen38_27B::GDN_CONTROL_ROWS
            * Qwen38_27B::LINEAR_HEAD_DIM
            * Qwen38_27B::LINEAR_HEAD_DIM;
        let mut recorded_projected = Vec::new();
        let mut recorded_log_decay = Vec::new();
        let mut recorded_beta = Vec::new();
        for layer in 0..GDN_LAYER_COUNT {
            recorded_projected.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_projected,
                layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_INPUT_ROWS,
                TARGET_VERIFY_ROUTE_COUNT * Qwen38_27B::GDN_INPUT_ROWS,
            )?);
            let control_offset = layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_CONTROL_ROWS;
            let control_values = TARGET_VERIFY_ROUTE_COUNT * Qwen38_27B::GDN_CONTROL_ROWS;
            recorded_log_decay.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_log_decay,
                control_offset,
                control_values,
            )?);
            recorded_beta.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_beta,
                control_offset,
                control_values,
            )?);
        }

        Ok(ResidentMtpVerifyObservables {
            residual_a: self.arena.copy_prefix_to_host(
                stream,
                workspace.residual_a,
                hidden_values,
            )?,
            final_normalized: self.arena.copy_prefix_to_host(
                stream,
                workspace.mixer_normalized,
                hidden_values,
            )?,
            logits: self
                .arena
                .copy_prefix_to_host(stream, workspace.logits, logit_values)?,
            provisional_history: self.arena.copy_prefix_to_host(
                stream,
                workspace.provisional_history,
                provisional_history_values,
            )?,
            provisional_state: self.arena.copy_prefix_to_host(
                stream,
                workspace.provisional_state,
                provisional_state_values,
            )?,
            recorded_projected,
            recorded_log_decay,
            recorded_beta,
            live_history,
            live_state,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads all lane-major segmented target seams and selected live GDN rows.
    pub fn qualification_target_mtp_segmented_observables(
        &self,
        stream: &CudaStream,
        route: ResidentMtpSegmentedVerifyRoute,
    ) -> EngineResult<ResidentMtpVerifyObservables> {
        let workspace = self.layout.workspace;
        let hidden_values = product(
            "segmented target MTP observable hidden values",
            route.rows(),
            Qwen38_27B::HIDDEN,
        )?;
        let logit_values = product(
            "segmented target MTP observable logits",
            route.rows(),
            Qwen38_27B::VOCAB,
        )?;
        let state_rows =
            self.arena
                .copy_prefix_to_host(stream, workspace.state_rows, route.rows())?;
        let history_values =
            self.layout.history_bytes() / MAX_BATCH / GDN_LAYER_COUNT / size_of::<u16>();
        let state_values =
            self.layout.state_bytes() / MAX_BATCH / GDN_LAYER_COUNT / size_of::<f32>();
        let mut live_history = Vec::with_capacity(route.batch * GDN_LAYER_COUNT * history_values);
        let mut live_state = Vec::with_capacity(route.batch * GDN_LAYER_COUNT * state_values);
        for lane in 0..route.batch {
            let slot = usize::try_from(state_rows[lane * route.tokens])
                .map_err(|_| EngineError::layout("segmented target MTP slot exceeds usize"))?;
            for layer in &self.layout.layers {
                let super::PersistentState::Gdn(persistent) = layer.persistent else {
                    continue;
                };
                live_history.extend(self.arena.copy_slice_to_host(
                    stream,
                    persistent.history,
                    slot * history_values,
                    history_values,
                )?);
                live_state.extend(self.arena.copy_slice_to_host(
                    stream,
                    persistent.state,
                    slot * state_values,
                    state_values,
                )?);
            }
        }

        let provisional_history_values =
            Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
        let provisional_state_values = Qwen38_27B::GDN_CONTROL_ROWS
            * Qwen38_27B::LINEAR_HEAD_DIM
            * Qwen38_27B::LINEAR_HEAD_DIM;
        let record_rows = route.batch * TARGET_VERIFY_ROUTE_COUNT;
        let mut recorded_projected = Vec::new();
        let mut recorded_log_decay = Vec::new();
        let mut recorded_beta = Vec::new();
        for layer in 0..GDN_LAYER_COUNT {
            recorded_projected.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_projected,
                layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_INPUT_ROWS,
                record_rows * Qwen38_27B::GDN_INPUT_ROWS,
            )?);
            let control_offset = layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_CONTROL_ROWS;
            let control_values = record_rows * Qwen38_27B::GDN_CONTROL_ROWS;
            recorded_log_decay.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_log_decay,
                control_offset,
                control_values,
            )?);
            recorded_beta.extend(self.arena.copy_slice_to_host(
                stream,
                workspace.recorded_beta,
                control_offset,
                control_values,
            )?);
        }

        Ok(ResidentMtpVerifyObservables {
            residual_a: self.arena.copy_prefix_to_host(
                stream,
                workspace.residual_a,
                hidden_values,
            )?,
            final_normalized: self.arena.copy_prefix_to_host(
                stream,
                workspace.mixer_normalized,
                hidden_values,
            )?,
            logits: self
                .arena
                .copy_prefix_to_host(stream, workspace.logits, logit_values)?,
            provisional_history: self.arena.copy_prefix_to_host(
                stream,
                workspace.provisional_history,
                route.batch * provisional_history_values,
            )?,
            provisional_state: self.arena.copy_prefix_to_host(
                stream,
                workspace.provisional_state,
                route.batch * provisional_state_values,
            )?,
            recorded_projected,
            recorded_log_decay,
            recorded_beta,
            live_history,
            live_state,
        })
    }

    #[cfg(feature = "qualification")]
    /// Launches and observes the first GDN mixer through target or decode ownership.
    pub fn qualification_first_gdn_mtp_seams(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
        target: bool,
    ) -> EngineResult<ResidentMtpGdnObservables> {
        if route.tokens != 1 {
            return Err(EngineError::route(
                "first-GDN target/decode seam comparison requires K=1",
            ));
        }
        let first = self
            ._pointers
            .layers
            .first()
            .ok_or_else(|| EngineError::layout("resident layer inventory is empty"))?;
        let MixerPointers::Gdn(_) = first.mixer else {
            return Err(EngineError::layout("resident first layer is not GDN"));
        };
        let workspace = self._pointers.workspace;
        // SAFETY: the checked K=1 route and resident owner cover every seam.
        unsafe {
            self._norm.launch_plain(
                stream,
                1,
                workspace.residual_a,
                first.mixer.input_norm(),
                workspace.mixer_normalized,
            )?;
        }
        if target {
            let mut gdn_layer = 0;
            launch_target_mtp_mixer(
                stream,
                route.tokens,
                0,
                route.maximum_length,
                route.attention,
                self.ops(),
                workspace,
                first.mixer,
                &mut gdn_layer,
            )?;
            debug_assert_eq!(gdn_layer, 1);
        } else {
            launch_mixer(
                stream,
                ResidentDecodeRoute {
                    batch: 1,
                    maximum_length: route.maximum_length,
                    attention: AttentionRoute::Short,
                },
                self.ops(),
                workspace,
                first.mixer,
            )?;
        }
        let workspace_regions = self.layout.workspace;
        let layout_layer = self
            .layout
            .layers
            .first()
            .ok_or_else(|| EngineError::layout("resident layout layer inventory is empty"))?;
        let super::PersistentState::Gdn(persistent_regions) = layout_layer.persistent else {
            return Err(EngineError::layout(
                "resident first layout layer is not GDN",
            ));
        };
        let projected = if target {
            workspace_regions.recorded_projected
        } else {
            workspace_regions.projected
        };
        let log_decay = if target {
            workspace_regions.recorded_log_decay
        } else {
            workspace_regions.log_decay
        };
        let beta = if target {
            workspace_regions.recorded_beta
        } else {
            workspace_regions.beta
        };
        let (history, state) = if target {
            (
                workspace_regions.provisional_history,
                workspace_regions.provisional_state,
            )
        } else {
            (persistent_regions.history, persistent_regions.state)
        };
        let history_values = persistent_regions.history.len() / MAX_BATCH;
        let state_values = persistent_regions.state.len() / MAX_BATCH;
        let history_offset = if target {
            0
        } else {
            route.slot * history_values
        };
        let state_offset = if target { 0 } else { route.slot * state_values };
        Ok(ResidentMtpGdnObservables {
            normalized: self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mixer_normalized,
                Qwen38_27B::HIDDEN,
            )?,
            projected: self.arena.copy_slice_to_host(
                stream,
                projected,
                0,
                Qwen38_27B::GDN_INPUT_ROWS,
            )?,
            log_decay: self.arena.copy_slice_to_host(
                stream,
                log_decay,
                0,
                Qwen38_27B::GDN_CONTROL_ROWS,
            )?,
            beta: self
                .arena
                .copy_slice_to_host(stream, beta, 0, Qwen38_27B::GDN_CONTROL_ROWS)?,
            convolved: self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.convolved,
                Qwen38_27B::GDN_QKV_ROWS,
            )?,
            recurrent: self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.recurrent_output,
                Qwen38_27B::GDN_VALUE_ROWS,
            )?,
            branch: self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mixer_branch,
                Qwen38_27B::HIDDEN,
            )?,
            history: self.arena.copy_slice_to_host(
                stream,
                history,
                history_offset,
                history_values,
            )?,
            state: self
                .arena
                .copy_slice_to_host(stream, state, state_offset, state_values)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Launches K=1 through every layer and observes each residual seam.
    pub fn qualification_mtp_k1_layer_seams(
        &self,
        stream: &CudaStream,
        route: ResidentMtpVerifyRoute,
        target: bool,
    ) -> EngineResult<Vec<ResidentMtpLayerObservables>> {
        if route.tokens != 1 {
            return Err(EngineError::route(
                "target/decode layer seam comparison requires K=1",
            ));
        }
        let first = self
            ._pointers
            .layers
            .first()
            .ok_or_else(|| EngineError::layout("resident layer inventory is empty"))?;
        let workspace = self._pointers.workspace;
        let workspace_regions = self.layout.workspace;
        // SAFETY: the checked K=1 route and resident owner cover every seam.
        unsafe {
            self._norm.launch_plain(
                stream,
                1,
                workspace.residual_a,
                first.mixer.input_norm(),
                workspace.mixer_normalized,
            )?;
        }

        let decode = ResidentDecodeRoute {
            batch: 1,
            maximum_length: route.maximum_length,
            attention: route.attention,
        };
        let mut residual_input = workspace.residual_a;
        let mut gdn_layer = 0;
        let mut observed = Vec::with_capacity(self._pointers.layers.len());
        for (index, layer) in self._pointers.layers.iter().enumerate() {
            if target {
                launch_target_mtp_mixer(
                    stream,
                    route.tokens,
                    0,
                    route.maximum_length,
                    route.attention,
                    self.ops(),
                    workspace,
                    layer.mixer,
                    &mut gdn_layer,
                )?;
            } else {
                launch_mixer(stream, decode, self.ops(), workspace, layer.mixer)?;
            }
            let mixer_branch = self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mixer_branch,
                Qwen38_27B::HIDDEN,
            )?;
            // SAFETY: branch and residual planes are disjoint maximum-row regions.
            unsafe {
                self._norm.launch_residual(
                    stream,
                    1,
                    residual_input,
                    workspace.mixer_branch,
                    layer.mixer.post_attention_norm(),
                    workspace.mixer_residual,
                    workspace.mlp_normalized,
                )?;
            }
            let mixer_residual = self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mixer_residual,
                Qwen38_27B::HIDDEN,
            )?;
            let mlp_normalized = self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mlp_normalized,
                Qwen38_27B::HIDDEN,
            )?;
            launch_mlp(
                stream,
                1,
                self.ops(),
                workspace,
                layer.mlp,
                &self.dense_mlp_maps,
            )?;
            let mlp_branch = self.arena.copy_prefix_to_host(
                stream,
                workspace_regions.mlp_branch,
                Qwen38_27B::HIDDEN,
            )?;

            let residual_output = if index.is_multiple_of(2) {
                workspace.residual_b
            } else {
                workspace.residual_a
            };
            let residual_region = if index.is_multiple_of(2) {
                workspace_regions.residual_b
            } else {
                workspace_regions.residual_a
            };
            let next_norm = self
                ._pointers
                .layers
                .get(index + 1)
                .map_or(self._pointers.endpoint.final_norm, |next| {
                    next.mixer.input_norm()
                });
            // SAFETY: residual ping-pong and both branch planes do not alias.
            unsafe {
                self._norm.launch_residual(
                    stream,
                    1,
                    workspace.mixer_residual,
                    workspace.mlp_branch,
                    next_norm,
                    residual_output,
                    workspace.mixer_normalized,
                )?;
            }
            observed.push(ResidentMtpLayerObservables {
                mixer_branch,
                mixer_residual,
                mlp_normalized,
                mlp_branch,
                residual: self.arena.copy_prefix_to_host(
                    stream,
                    residual_region,
                    Qwen38_27B::HIDDEN,
                )?,
                next_normalized: self.arena.copy_prefix_to_host(
                    stream,
                    workspace_regions.mixer_normalized,
                    Qwen38_27B::HIDDEN,
                )?,
            });
            residual_input = residual_output;
        }
        if target && gdn_layer != GDN_LAYER_COUNT {
            return Err(EngineError::layout(format!(
                "target layer seam comparison visited {gdn_layer} GDN layers, expected {GDN_LAYER_COUNT}"
            )));
        }
        Ok(observed)
    }

    #[cfg(feature = "qualification")]
    /// Returns all source cache scales in ascending attention-layer order.
    pub fn qualification_cache_scales(&self) -> Vec<[f32; 2]> {
        self._pointers
            .layers
            .iter()
            .filter_map(|layer| match layer.mixer {
                MixerPointers::Attention(pointers) => Some([
                    pointers.scalars.key_cache_scale,
                    pointers.scalars.value_cache_scale,
                ]),
                MixerPointers::Gdn(_) => None,
            })
            .collect()
    }

    #[cfg(feature = "qualification")]
    /// Returns all source NVFP4 divisors in ascending early-layer order.
    pub fn qualification_nvfp4_divisors(&self) -> Vec<[f32; 4]> {
        self._pointers
            .layers
            .iter()
            .filter_map(|layer| match layer.mlp {
                MlpPointers::Nvfp4(pointers) => Some([
                    pointers.scalars.gate_up_input,
                    pointers.scalars.gate_up_weight,
                    pointers.scalars.down_input,
                    pointers.scalars.down_weight,
                ]),
                MlpPointers::DenseFp8(_) => None,
            })
            .collect()
    }

    #[cfg(feature = "qualification")]
    /// Reads final residual, final-normalized rows, logits, and all persistent state.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<ResidentModelObservables> {
        let mut history =
            Vec::with_capacity(self.layout.history_bytes() / std::mem::size_of::<u16>());
        let mut state = Vec::with_capacity(self.layout.state_bytes() / std::mem::size_of::<f32>());
        let active_cache_values = cache_values(SHORT_CONTEXT_PHYSICAL_PAGES)?;
        let guard_values = cache_values(1)?;
        let observed_plane_values = active_cache_values + guard_values;
        let attention_layers = self.layout.kv_layout.layers().len();
        let mut key_pages = Vec::with_capacity(active_cache_values * attention_layers);
        let mut value_pages = Vec::with_capacity(active_cache_values * attention_layers);
        let mut key_guard_pages = Vec::with_capacity(guard_values * attention_layers);
        let mut value_guard_pages = Vec::with_capacity(guard_values * attention_layers);
        for layer in &self.layout.layers {
            match layer.persistent {
                super::PersistentState::Gdn(regions) => {
                    history.extend(self.arena.copy_to_host(stream, regions.history)?);
                    state.extend(self.arena.copy_to_host(stream, regions.state)?);
                }
                super::PersistentState::Attention => {}
            }
        }
        for cache in self.layout.kv_layout.layers() {
            let key =
                self.kv_arena
                    .copy_prefix_to_host(stream, cache.key.data, observed_plane_values)?;
            let value = self.kv_arena.copy_prefix_to_host(
                stream,
                cache.value.data,
                observed_plane_values,
            )?;
            key_pages.extend_from_slice(&key[..active_cache_values]);
            value_pages.extend_from_slice(&value[..active_cache_values]);
            key_guard_pages.extend_from_slice(&key[active_cache_values..]);
            value_guard_pages.extend_from_slice(&value[active_cache_values..]);
        }
        let workspace = self.layout.workspace;
        Ok(ResidentModelObservables {
            residual_a: self.arena.copy_to_host(stream, workspace.residual_a)?,
            residual_b: self.arena.copy_to_host(stream, workspace.residual_b)?,
            mixer_residual: self.arena.copy_to_host(stream, workspace.mixer_residual)?,
            mixer_normalized: self
                .arena
                .copy_to_host(stream, workspace.mixer_normalized)?,
            mlp_normalized: self.arena.copy_to_host(stream, workspace.mlp_normalized)?,
            activation_codes: self
                .arena
                .copy_to_host(stream, workspace.activation_codes)?,
            activation_scales: self
                .arena
                .copy_to_host(stream, workspace.activation_scales)?,
            nvfp4_activation_codes: self
                .arena
                .copy_to_host(stream, workspace.nvfp4_activation_codes)?,
            nvfp4_activation_scales: self
                .arena
                .copy_to_host(stream, workspace.nvfp4_activation_scales)?,
            projected: self.arena.copy_to_host(stream, workspace.projected)?,
            log_decay: self.arena.copy_to_host(stream, workspace.log_decay)?,
            beta: self.arena.copy_to_host(stream, workspace.beta)?,
            convolved: self.arena.copy_to_host(stream, workspace.convolved)?,
            recurrent_output: self
                .arena
                .copy_to_host(stream, workspace.recurrent_output)?,
            query: self.arena.copy_to_host(stream, workspace.query)?,
            partial_maximum: self.arena.copy_to_host(stream, workspace.partial_maximum)?,
            partial_denominator: self
                .arena
                .copy_to_host(stream, workspace.partial_denominator)?,
            partial_numerator: self
                .arena
                .copy_to_host(stream, workspace.partial_numerator)?,
            prefill_partials: self
                .arena
                .copy_to_host(stream, workspace.prefill_partials)?,
            attention: self.arena.copy_to_host(stream, workspace.attention)?,
            mixer_branch: self.arena.copy_to_host(stream, workspace.mixer_branch)?,
            swiglu: self.arena.copy_to_host(stream, workspace.swiglu)?,
            mlp_branch: self.arena.copy_to_host(stream, workspace.mlp_branch)?,
            logits: self.arena.copy_to_host(stream, workspace.logits)?,
            history,
            state,
            key_pages,
            value_pages,
            key_guard_pages,
            value_guard_pages,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads the shared attention scratch and downstream seams changed by a long route.
    pub fn qualification_long_context_observables(
        &self,
        stream: &CudaStream,
        route: ResidentDecodeRoute,
    ) -> EngineResult<ResidentLongContextObservables> {
        let workspace = self.layout.workspace;
        let attention_values = product(
            "resident long qualification attention values",
            route.batch,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let projected_values = product(
            "resident long qualification QKV values",
            route.batch,
            Qwen38_27B::ATTENTION_QKV_ROWS,
        )?;
        let partial_values = product(
            "resident long qualification partial values",
            product(
                "resident long qualification head rows",
                route.batch,
                Qwen38_27B::NUM_ATTENTION_HEADS,
            )?,
            tuisko_kernels_sm120::LONG_CONTEXT_GQA_MAX_PARTITIONS,
        )?;
        let partial_numerator_values = product(
            "resident long qualification numerator values",
            partial_values,
            Qwen38_27B::HEAD_DIM,
        )?;
        let hidden_values = product(
            "resident long qualification hidden values",
            route.batch,
            Qwen38_27B::HIDDEN,
        )?;
        let logit_values = product(
            "resident long qualification logit values",
            route.batch,
            Qwen38_27B::VOCAB,
        )?;
        Ok(ResidentLongContextObservables {
            projected: self.arena.copy_prefix_to_host(
                stream,
                workspace.projected,
                projected_values,
            )?,
            query: self
                .arena
                .copy_prefix_to_host(stream, workspace.query, attention_values)?,
            partial_maximum: self.arena.copy_prefix_to_host(
                stream,
                workspace.partial_maximum,
                partial_values,
            )?,
            partial_denominator: self.arena.copy_prefix_to_host(
                stream,
                workspace.partial_denominator,
                partial_values,
            )?,
            partial_numerator: self.arena.copy_prefix_to_host(
                stream,
                workspace.partial_numerator,
                partial_numerator_values,
            )?,
            attention: self.arena.copy_prefix_to_host(
                stream,
                workspace.attention,
                attention_values,
            )?,
            mixer_branch: self.arena.copy_prefix_to_host(
                stream,
                workspace.mixer_branch,
                hidden_values,
            )?,
            residual_a: self.arena.copy_prefix_to_host(
                stream,
                workspace.residual_a,
                hidden_values,
            )?,
            logits: self
                .arena
                .copy_prefix_to_host(stream, workspace.logits, logit_values)?,
        })
    }

    fn ops(&self) -> Ops<'_> {
        Ops {
            norm: &self._norm,
            gdn_input: &self._gdn_input,
            gdn_prepare: &self._gdn_prepare,
            gdn_recurrence: &self._gdn_recurrence,
            gdn_state_snapshot: &self._gdn_state_snapshot,
            gdn_output: &self._gdn_output,
            attention_qkv: &self._attention_qkv,
            attention_qk_prepare: &self._attention_qk_prepare,
            paged_gqa: &self._paged_gqa,
            long_context_paged_gqa: &self._long_context_paged_gqa,
            attention_output: &self._attention_output,
            dense_swiglu: &self._dense_swiglu,
            dense_down: &self._dense_down,
            nvfp4_swiglu: &self._nvfp4_swiglu,
            nvfp4_down: &self._nvfp4_down,
            lm_head: &self._lm_head,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete externally observable outputs and persistent planes.
pub struct ResidentModelObservables {
    /// Final BF16 residual rows before endpoint normalization.
    pub residual_a: Vec<u16>,
    /// Penultimate-layer BF16 residual rows retained by ping-pong ownership.
    pub residual_b: Vec<u16>,
    /// Final-layer residual rows before the MLP branch is added.
    pub mixer_residual: Vec<u16>,
    /// Final-normalized BF16 rows consumed by the LM head.
    pub mixer_normalized: Vec<u16>,
    /// Final-layer normalized MLP inputs.
    pub mlp_normalized: Vec<u16>,
    /// Shared dynamic E4M3 codes: endpoint prefix plus the last wider writer's tail.
    pub activation_codes: Vec<u8>,
    /// Last dynamic FP32 scales, owned by the endpoint after replay.
    pub activation_scales: Vec<f32>,
    /// Last early-layer dynamic packed E2M1 activation codes.
    pub nvfp4_activation_codes: Vec<u8>,
    /// Last early-layer dynamic E4M3 activation scales.
    pub nvfp4_activation_scales: Vec<u8>,
    /// Last fused mixer projection rows.
    pub projected: Vec<u16>,
    /// Last GDN log-decay rows.
    pub log_decay: Vec<f32>,
    /// Last GDN beta rows.
    pub beta: Vec<f32>,
    /// Last causal-convolved GDN rows.
    pub convolved: Vec<u16>,
    /// Last gated recurrent GDN output rows.
    pub recurrent_output: Vec<u16>,
    /// Last prepared attention query rows.
    pub query: Vec<f32>,
    /// Partition maxima retained by the long-context attention route.
    pub partial_maximum: Vec<f32>,
    /// Partition softmax denominators retained by the long-context attention route.
    pub partial_denominator: Vec<f32>,
    /// Partition attention numerators retained by the long-context attention route.
    pub partial_numerator: Vec<f32>,
    /// Macro-prefill producer and reduction scratch.
    pub prefill_partials: Vec<f32>,
    /// Last gated attention-output rows.
    pub attention: Vec<f32>,
    /// Final mixer projection branch rows.
    pub mixer_branch: Vec<u16>,
    /// Final MLP SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Final MLP down-projection branch rows.
    pub mlp_branch: Vec<u16>,
    /// Full BF16 vocabulary logits.
    pub logits: Vec<u16>,
    /// Concatenated causal histories in ascending GDN-layer order.
    pub history: Vec<u16>,
    /// Concatenated recurrent states in ascending GDN-layer order.
    pub state: Vec<f32>,
    /// Concatenated represented key caches in ascending attention-layer order.
    pub key_pages: Vec<u8>,
    /// Concatenated represented value caches in ascending attention-layer order.
    pub value_pages: Vec<u8>,
    /// First unassigned key page from every attention layer.
    pub key_guard_pages: Vec<u8>,
    /// First unassigned value page from every attention layer.
    pub value_guard_pages: Vec<u8>,
}

#[cfg(feature = "qualification")]
/// First-layer GDN seams observed through target-causal or ordinary decode ownership.
pub struct ResidentMtpGdnObservables {
    /// Input-normalized hidden row.
    pub normalized: Vec<u16>,
    /// Complete GDN input projection.
    pub projected: Vec<u16>,
    /// Decay controls.
    pub log_decay: Vec<f32>,
    /// Beta controls.
    pub beta: Vec<f32>,
    /// Activated causal convolution rows.
    pub convolved: Vec<u16>,
    /// Gated recurrent output before projection.
    pub recurrent: Vec<u16>,
    /// Projected mixer branch.
    pub branch: Vec<u16>,
    /// Updated selected history row.
    pub history: Vec<u16>,
    /// Updated selected recurrent-state row.
    pub state: Vec<f32>,
}

#[cfg(feature = "qualification")]
/// Complete K=1 residual seams for one target or ordinary decode layer.
pub struct ResidentMtpLayerObservables {
    /// Mixer projection branch before the first residual seam.
    pub mixer_branch: Vec<u16>,
    /// Residual after adding the mixer branch.
    pub mixer_residual: Vec<u16>,
    /// Normalized input to the MLP.
    pub mlp_normalized: Vec<u16>,
    /// MLP projection branch before the second residual seam.
    pub mlp_branch: Vec<u16>,
    /// Layer output residual.
    pub residual: Vec<u16>,
    /// Normalized input to the following layer or endpoint.
    pub next_normalized: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Provisional target outputs, replay records, and selected live GDN rows.
pub struct ResidentMtpVerifyObservables {
    /// Active target residual rows before endpoint normalization.
    pub residual_a: Vec<u16>,
    /// Final-normalized target rows that feed the shared LM head.
    pub final_normalized: Vec<u16>,
    /// Active lane-major BF16 target vocabulary logits.
    pub logits: Vec<u16>,
    /// Final provisional causal-history rows from the last GDN layer.
    pub provisional_history: Vec<u16>,
    /// Final provisional recurrent-state rows from the last GDN layer.
    pub provisional_state: Vec<f32>,
    /// Per-GDN-layer projected values retained for accepted-prefix replay.
    pub recorded_projected: Vec<u16>,
    /// Per-GDN-layer log-decay controls retained for replay.
    pub recorded_log_decay: Vec<f32>,
    /// Per-GDN-layer beta controls retained for replay.
    pub recorded_beta: Vec<f32>,
    /// Selected live history rows concatenated lane-major, then by GDN layer.
    pub live_history: Vec<u16>,
    /// Selected live recurrent rows concatenated lane-major, then by GDN layer.
    pub live_state: Vec<f32>,
}

#[cfg(feature = "qualification")]
/// Long-attention scratch plus the downstream complete-model seams it changes.
pub struct ResidentLongContextObservables {
    /// Last attention layer's QKV projection, including output-gate rows.
    pub projected: Vec<u16>,
    /// Last attention layer's prepared query rows.
    pub query: Vec<f32>,
    /// Last attention layer's partition maxima.
    pub partial_maximum: Vec<f32>,
    /// Last attention layer's partition denominators.
    pub partial_denominator: Vec<f32>,
    /// Last attention layer's partition numerators.
    pub partial_numerator: Vec<f32>,
    /// Last attention layer's gated GQA output.
    pub attention: Vec<f32>,
    /// Final attention projection branch.
    pub mixer_branch: Vec<u16>,
    /// Final residual rows before endpoint normalization.
    pub residual_a: Vec<u16>,
    /// Full BF16 vocabulary logits.
    pub logits: Vec<u16>,
}

#[derive(Clone, Copy)]
struct AttentionScalars {
    key_cache_scale: f32,
    value_cache_scale: f32,
}

#[derive(Clone, Copy)]
struct Nvfp4Scalars {
    gate_up_input: f32,
    gate_up_weight: f32,
    down_input: f32,
    down_weight: f32,
}

#[derive(Clone, Copy)]
enum MixerScalars {
    Gdn,
    Attention(AttentionScalars),
}

#[derive(Clone, Copy)]
enum MlpScalars {
    Nvfp4(Nvfp4Scalars),
    DenseFp8,
}

#[derive(Clone, Copy)]
struct LayerScalars {
    mixer: MixerScalars,
    mlp: MlpScalars,
}

#[derive(Clone, Copy)]
struct GdnPointers {
    input_norm: *const u16,
    input_weight_codes: *const u8,
    input_weight_scales: *const u16,
    control_weights: *const u16,
    a_log: *const u16,
    dt_bias: *const u16,
    convolution_weights: *const u16,
    recurrent_norm: *const u16,
    output_weight_codes: *const u8,
    output_weight_scales: *const u16,
    post_attention_norm: *const u16,
    history: *mut u16,
    state: *mut f32,
}

#[derive(Clone, Copy)]
struct AttentionPointers {
    input_norm: *const u16,
    qkv_weight_codes: *const u8,
    qkv_weight_scales: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,
    output_weight_codes: *const u8,
    output_weight_scales: *const u16,
    post_attention_norm: *const u16,
    key_pages: *mut u8,
    value_pages: *mut u8,
    scalars: AttentionScalars,
}

#[derive(Clone, Copy)]
enum MixerPointers {
    Gdn(GdnPointers),
    Attention(AttentionPointers),
}

impl MixerPointers {
    const fn input_norm(self) -> *const u16 {
        match self {
            Self::Gdn(pointers) => pointers.input_norm,
            Self::Attention(pointers) => pointers.input_norm,
        }
    }

    const fn post_attention_norm(self) -> *const u16 {
        match self {
            Self::Gdn(pointers) => pointers.post_attention_norm,
            Self::Attention(pointers) => pointers.post_attention_norm,
        }
    }

    #[cfg(feature = "qualification")]
    fn push_addresses(self, addresses: &mut Vec<usize>) {
        match self {
            Self::Gdn(p) => addresses.extend([
                p.input_norm.addr(),
                p.input_weight_codes.addr(),
                p.input_weight_scales.addr(),
                p.control_weights.addr(),
                p.a_log.addr(),
                p.dt_bias.addr(),
                p.convolution_weights.addr(),
                p.recurrent_norm.addr(),
                p.output_weight_codes.addr(),
                p.output_weight_scales.addr(),
                p.post_attention_norm.addr(),
                p.history.addr(),
                p.state.addr(),
            ]),
            Self::Attention(p) => addresses.extend([
                p.input_norm.addr(),
                p.qkv_weight_codes.addr(),
                p.qkv_weight_scales.addr(),
                p.query_norm.addr(),
                p.key_norm.addr(),
                p.output_weight_codes.addr(),
                p.output_weight_scales.addr(),
                p.post_attention_norm.addr(),
                p.key_pages.addr(),
                p.value_pages.addr(),
            ]),
        }
    }
}

#[derive(Clone, Copy)]
struct Nvfp4Pointers {
    gate_weight_codes: *const u8,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    up_weight_codes: *const u8,
    gate_up_weight_scales: *const u8,
    down_weight_codes: *const u8,
    down_weight_scales: *const u8,
    scalars: Nvfp4Scalars,
}

#[derive(Clone, Copy)]
struct DenseFp8Pointers {
    gate_up_weight_codes: *const u8,
    gate_up_weight_scales: *const u16,
    down_weight_codes: *const u8,
    down_weight_scales: *const u16,
    map_index: usize,
}

#[derive(Clone, Copy)]
enum MlpPointers {
    Nvfp4(Nvfp4Pointers),
    DenseFp8(DenseFp8Pointers),
}

impl MlpPointers {
    #[cfg(feature = "qualification")]
    fn push_addresses(self, addresses: &mut Vec<usize>) {
        match self {
            Self::Nvfp4(p) => addresses.extend([
                p.gate_weight_codes.addr(),
                p.up_weight_codes.addr(),
                p.gate_up_weight_scales.addr(),
                p.down_weight_codes.addr(),
                p.down_weight_scales.addr(),
            ]),
            Self::DenseFp8(p) => addresses.extend([
                p.gate_up_weight_codes.addr(),
                p.gate_up_weight_scales.addr(),
                p.down_weight_codes.addr(),
                p.down_weight_scales.addr(),
            ]),
        }
    }
}

#[derive(Clone, Copy)]
struct LayerPointers {
    mixer: MixerPointers,
    mlp: MlpPointers,
}

#[derive(Clone, Copy)]
struct WorkspacePointers {
    residual_a: *mut u16,
    residual_b: *mut u16,
    mixer_residual: *mut u16,
    mixer_normalized: *mut u16,
    mlp_normalized: *mut u16,
    activation_codes: *mut u8,
    activation_scales: *mut f32,
    nvfp4_activation_codes: *mut u8,
    nvfp4_activation_scales: *mut u8,
    projected: *mut u16,
    state_rows: *const u32,
    log_decay: *mut f32,
    beta: *mut f32,
    convolved: *mut u16,
    recurrent_output: *mut u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    query: *mut f32,
    partial_maximum: *mut f32,
    partial_denominator: *mut f32,
    partial_numerator: *mut f32,
    prefill_partials: *mut f32,
    attention: *mut f32,
    mixer_branch: *mut u16,
    swiglu: *mut u16,
    mlp_branch: *mut u16,
    logits: *mut u16,
    provisional_history: *mut u16,
    provisional_state: *mut f32,
    provisional_state_row: *const u32,
    recorded_projected: *mut u16,
    recorded_log_decay: *mut f32,
    recorded_beta: *mut f32,
}

impl WorkspacePointers {
    fn recorded_projected_for(self, gdn_layer: usize) -> *mut u16 {
        self.recorded_projected
            .wrapping_add(gdn_layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_INPUT_ROWS)
    }

    fn recorded_log_decay_for(self, gdn_layer: usize) -> *mut f32 {
        self.recorded_log_decay
            .wrapping_add(gdn_layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_CONTROL_ROWS)
    }

    fn recorded_beta_for(self, gdn_layer: usize) -> *mut f32 {
        self.recorded_beta
            .wrapping_add(gdn_layer * TARGET_VERIFY_ROWS * Qwen38_27B::GDN_CONTROL_ROWS)
    }

    fn target_record_offset(lane: usize) -> usize {
        lane * TARGET_VERIFY_ROUTE_COUNT
    }

    fn row_offset(self, rows: usize) -> Self {
        Self {
            residual_a: self.residual_a.wrapping_add(rows * Qwen38_27B::HIDDEN),
            residual_b: self.residual_b.wrapping_add(rows * Qwen38_27B::HIDDEN),
            mixer_residual: self.mixer_residual.wrapping_add(rows * Qwen38_27B::HIDDEN),
            mixer_normalized: self
                .mixer_normalized
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            mlp_normalized: self.mlp_normalized.wrapping_add(rows * Qwen38_27B::HIDDEN),
            activation_codes: self
                .activation_codes
                .wrapping_add(rows * Qwen38_27B::INTERMEDIATE),
            activation_scales: self.activation_scales.wrapping_add(rows),
            nvfp4_activation_codes: self
                .nvfp4_activation_codes
                .wrapping_add(rows * Qwen38_27B::INTERMEDIATE / 2),
            nvfp4_activation_scales: self
                .nvfp4_activation_scales
                .wrapping_add(rows * Qwen38_27B::INTERMEDIATE / 16),
            projected: self
                .projected
                .wrapping_add(rows * Qwen38_27B::GDN_INPUT_ROWS),
            state_rows: self.state_rows.wrapping_add(rows),
            log_decay: self
                .log_decay
                .wrapping_add(rows * Qwen38_27B::GDN_CONTROL_ROWS),
            beta: self.beta.wrapping_add(rows * Qwen38_27B::GDN_CONTROL_ROWS),
            convolved: self.convolved.wrapping_add(rows * Qwen38_27B::GDN_QKV_ROWS),
            recurrent_output: self
                .recurrent_output
                .wrapping_add(rows * Qwen38_27B::GDN_VALUE_ROWS),
            rope_cos: self.rope_cos.wrapping_add(rows * ROTARY_PAIRS),
            rope_sin: self.rope_sin.wrapping_add(rows * ROTARY_PAIRS),
            block_tables: self.block_tables,
            table_rows: self.table_rows.wrapping_add(rows),
            cache_positions: self.cache_positions.wrapping_add(rows),
            lengths: self.lengths.wrapping_add(rows),
            query: self
                .query
                .wrapping_add(rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS),
            partial_maximum: self.partial_maximum,
            partial_denominator: self.partial_denominator,
            partial_numerator: self.partial_numerator,
            prefill_partials: self.prefill_partials,
            attention: self
                .attention
                .wrapping_add(rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS),
            mixer_branch: self.mixer_branch.wrapping_add(rows * Qwen38_27B::HIDDEN),
            swiglu: self.swiglu.wrapping_add(rows * Qwen38_27B::INTERMEDIATE),
            mlp_branch: self.mlp_branch.wrapping_add(rows * Qwen38_27B::HIDDEN),
            logits: self.logits.wrapping_add(rows * Qwen38_27B::VOCAB),
            provisional_history: self.provisional_history,
            provisional_state: self.provisional_state,
            provisional_state_row: self.provisional_state_row,
            recorded_projected: self.recorded_projected,
            recorded_log_decay: self.recorded_log_decay,
            recorded_beta: self.recorded_beta,
        }
    }
}

#[derive(Clone)]
struct ProgramPointers {
    layers: Vec<LayerPointers>,
    endpoint: EndpointPointers,
    workspace: WorkspacePointers,
}

impl ProgramPointers {
    fn bind(
        arena: &DeviceArena,
        kv_arena: &DeviceArena,
        layout: &ResidentModelLayout,
        scalars: &[LayerScalars],
    ) -> EngineResult<Self> {
        if scalars.len() != layout.layers.len() {
            return Err(EngineError::layout(
                "resident scalar inventory does not match layer inventory",
            ));
        }
        let mut layers = Vec::with_capacity(layout.layers.len());
        let mut attention_layer = 0;
        let mut dense_mlp = 0;
        for (layer, scalars) in layout.layers.iter().zip(scalars) {
            let cache = match layer.persistent {
                super::PersistentState::Gdn(_) => None,
                super::PersistentState::Attention => {
                    let cache = layout
                        .kv_layout
                        .layers()
                        .get(attention_layer)
                        .copied()
                        .ok_or_else(|| {
                            EngineError::layout(
                                "resident attention layer exceeds shared KV plane inventory",
                            )
                        })?;
                    attention_layer += 1;
                    Some(cache)
                }
            };
            layers.push(LayerPointers {
                mixer: bind_mixer(
                    arena,
                    kv_arena,
                    layer.mixer,
                    layer.persistent,
                    cache,
                    scalars.mixer,
                )?,
                mlp: bind_mlp(arena, layer.mlp, scalars.mlp, &mut dense_mlp)?,
            });
        }
        if attention_layer != layout.kv_layout.layers().len() {
            return Err(EngineError::layout(
                "resident attention layer inventory does not consume every shared KV plane",
            ));
        }
        let endpoint = EndpointPointers::bind(arena, layout.endpoint)?;
        let workspace = WorkspacePointers::bind(arena, kv_arena, layout)?;

        Ok(Self {
            layers,
            endpoint,
            workspace,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(&self) -> Vec<usize> {
        let mut addresses = Vec::new();
        for layer in &self.layers {
            layer.mixer.push_addresses(&mut addresses);
            layer.mlp.push_addresses(&mut addresses);
        }
        self.endpoint.push_addresses(&mut addresses);
        self.workspace.push_addresses(&mut addresses);
        addresses
    }
}

impl DenseMlpMaps {
    fn bind_all(stream: &CudaStream, pointers: &ProgramPointers) -> EngineResult<Vec<Self>> {
        let mut maps = Vec::new();
        for layer in &pointers.layers {
            let MlpPointers::DenseFp8(weights) = layer.mlp else {
                continue;
            };
            if weights.map_index != maps.len() {
                return Err(EngineError::layout(
                    "resident dense MLP tensor-map inventory is not contiguous",
                ));
            }
            // SAFETY: resident arenas keep the shared scratch and this layer's
            // source-native weight addresses stable for every captured graph.
            let gate_up = unsafe {
                DenseFp8SwiGluTmaMaps::new(
                    stream,
                    pointers.workspace.activation_codes.cast_const(),
                    weights.gate_up_weight_codes,
                )?
            };
            // SAFETY: the same stable scratch covers the larger down-input row.
            let down = unsafe {
                DenseFp8DownTmaMaps::new(
                    stream,
                    pointers.workspace.activation_codes.cast_const(),
                    weights.down_weight_codes,
                )?
            };
            maps.push(Self { gate_up, down });
        }
        if maps.len() != 8 {
            return Err(EngineError::layout(format!(
                "resident dense MLP tensor-map inventory has {} layers, expected 8",
                maps.len()
            )));
        }
        Ok(maps)
    }

    fn byte_len(&self) -> usize {
        self.gate_up.byte_len() + self.down.byte_len()
    }

    #[cfg(feature = "qualification")]
    fn push_addresses(&self, addresses: &mut Vec<usize>) {
        addresses.extend(self.gate_up.device_addresses());
        addresses.extend(self.down.device_addresses());
    }
}

#[derive(Clone, Copy)]
struct EndpointPointers {
    final_norm: *const u16,
    lm_head_codes: *const u8,
    lm_head_scales: *const u16,
}

impl EndpointPointers {
    fn bind(arena: &DeviceArena, regions: EndpointWeights) -> GpuResult<Self> {
        Ok(Self {
            final_norm: arena.address(regions.final_norm)?.cast_const(),
            lm_head_codes: arena.address(regions.lm_head_codes)?.cast_const(),
            lm_head_scales: arena.address(regions.lm_head_scales)?.cast_const(),
        })
    }

    #[cfg(feature = "qualification")]
    fn push_addresses(self, addresses: &mut Vec<usize>) {
        addresses.extend([
            self.final_norm.addr(),
            self.lm_head_codes.addr(),
            self.lm_head_scales.addr(),
        ]);
    }
}

impl WorkspacePointers {
    fn bind(
        arena: &DeviceArena,
        kv_arena: &DeviceArena,
        layout: &ResidentModelLayout,
    ) -> GpuResult<Self> {
        let regions = layout.workspace;
        Ok(Self {
            residual_a: arena.address(regions.residual_a)?,
            residual_b: arena.address(regions.residual_b)?,
            mixer_residual: arena.address(regions.mixer_residual)?,
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            mlp_normalized: arena.address(regions.mlp_normalized)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            nvfp4_activation_codes: arena.address(regions.nvfp4_activation_codes)?,
            nvfp4_activation_scales: arena.address(regions.nvfp4_activation_scales)?,
            projected: arena.address(regions.projected)?,
            state_rows: arena.address(regions.state_rows)?.cast_const(),
            log_decay: arena.address(regions.log_decay)?,
            beta: arena.address(regions.beta)?,
            convolved: arena.address(regions.convolved)?,
            recurrent_output: arena.address(regions.recurrent_output)?,
            rope_cos: arena.address(regions.rope_cos)?.cast_const(),
            rope_sin: arena.address(regions.rope_sin)?.cast_const(),
            block_tables: kv_arena
                .address(layout.kv_layout.block_tables())?
                .cast_const(),
            table_rows: arena.address(regions.table_rows)?.cast_const(),
            cache_positions: arena.address(regions.cache_positions)?.cast_const(),
            lengths: arena.address(regions.lengths)?.cast_const(),
            query: arena.address(regions.query)?,
            partial_maximum: arena.address(regions.partial_maximum)?,
            partial_denominator: arena.address(regions.partial_denominator)?,
            partial_numerator: arena.address(regions.partial_numerator)?,
            prefill_partials: arena.address(regions.prefill_partials)?,
            attention: arena.address(regions.attention)?,
            mixer_branch: arena.address(regions.mixer_branch)?,
            swiglu: arena.address(regions.swiglu)?,
            mlp_branch: arena.address(regions.mlp_branch)?,
            logits: arena.address(regions.logits)?,
            provisional_history: arena.address(regions.provisional_history)?,
            provisional_state: arena.address(regions.provisional_state)?,
            provisional_state_row: arena.address(regions.provisional_state_row)?.cast_const(),
            recorded_projected: arena.address(regions.recorded_projected)?,
            recorded_log_decay: arena.address(regions.recorded_log_decay)?,
            recorded_beta: arena.address(regions.recorded_beta)?,
        })
    }

    #[cfg(feature = "qualification")]
    fn push_addresses(self, addresses: &mut Vec<usize>) {
        addresses.extend([
            self.residual_a.addr(),
            self.residual_b.addr(),
            self.mixer_residual.addr(),
            self.mixer_normalized.addr(),
            self.mlp_normalized.addr(),
            self.activation_codes.addr(),
            self.activation_scales.addr(),
            self.nvfp4_activation_codes.addr(),
            self.nvfp4_activation_scales.addr(),
            self.projected.addr(),
            self.state_rows.addr(),
            self.log_decay.addr(),
            self.beta.addr(),
            self.convolved.addr(),
            self.recurrent_output.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.block_tables.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.lengths.addr(),
            self.query.addr(),
            self.partial_maximum.addr(),
            self.partial_denominator.addr(),
            self.partial_numerator.addr(),
            self.prefill_partials.addr(),
            self.attention.addr(),
            self.mixer_branch.addr(),
            self.swiglu.addr(),
            self.mlp_branch.addr(),
            self.logits.addr(),
            self.provisional_history.addr(),
            self.provisional_state.addr(),
            self.provisional_state_row.addr(),
            self.recorded_projected.addr(),
            self.recorded_log_decay.addr(),
            self.recorded_beta.addr(),
        ]);
    }
}

fn bind_mixer(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    weights: MixerWeights,
    persistent: super::PersistentState,
    cache: Option<LayerKvRegions>,
    scalars: MixerScalars,
) -> EngineResult<MixerPointers> {
    match (weights, persistent, cache, scalars) {
        (
            MixerWeights::Gdn(weights),
            super::PersistentState::Gdn(persistent),
            None,
            MixerScalars::Gdn,
        ) => Ok(MixerPointers::Gdn(bind_gdn(arena, weights, persistent)?)),
        (
            MixerWeights::Attention(weights),
            super::PersistentState::Attention,
            Some(cache),
            MixerScalars::Attention(scalars),
        ) => Ok(MixerPointers::Attention(bind_attention(
            arena, kv_arena, weights, cache, scalars,
        )?)),
        _ => Err(EngineError::layout(
            "resident mixer scalar route does not match its source route",
        )),
    }
}

fn bind_gdn(
    arena: &DeviceArena,
    weights: GdnWeights,
    persistent: GdnPersistent,
) -> GpuResult<GdnPointers> {
    Ok(GdnPointers {
        input_norm: arena.address(weights.input_norm)?.cast_const(),
        input_weight_codes: arena.address(weights.input_weight_codes)?.cast_const(),
        input_weight_scales: arena.address(weights.input_weight_scales)?.cast_const(),
        control_weights: arena.address(weights.control_weights)?.cast_const(),
        a_log: arena.address(weights.a_log)?.cast_const(),
        dt_bias: arena.address(weights.dt_bias)?.cast_const(),
        convolution_weights: arena.address(weights.convolution_weights)?.cast_const(),
        recurrent_norm: arena.address(weights.recurrent_norm)?.cast_const(),
        output_weight_codes: arena.address(weights.output_weight_codes)?.cast_const(),
        output_weight_scales: arena.address(weights.output_weight_scales)?.cast_const(),
        post_attention_norm: arena.address(weights.post_attention_norm)?.cast_const(),
        history: arena.address(persistent.history)?,
        state: arena.address(persistent.state)?,
    })
}

fn bind_attention(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    weights: AttentionWeights,
    cache: LayerKvRegions,
    scalars: AttentionScalars,
) -> GpuResult<AttentionPointers> {
    Ok(AttentionPointers {
        input_norm: arena.address(weights.input_norm)?.cast_const(),
        qkv_weight_codes: arena.address(weights.qkv_weight_codes)?.cast_const(),
        qkv_weight_scales: arena.address(weights.qkv_weight_scales)?.cast_const(),
        query_norm: arena.address(weights.query_norm)?.cast_const(),
        key_norm: arena.address(weights.key_norm)?.cast_const(),
        output_weight_codes: arena.address(weights.output_weight_codes)?.cast_const(),
        output_weight_scales: arena.address(weights.output_weight_scales)?.cast_const(),
        post_attention_norm: arena.address(weights.post_attention_norm)?.cast_const(),
        key_pages: kv_arena.address(cache.key.data)?,
        value_pages: kv_arena.address(cache.value.data)?,
        scalars,
    })
}

fn bind_mlp(
    arena: &DeviceArena,
    weights: MlpWeights,
    scalars: MlpScalars,
    dense_mlp: &mut usize,
) -> EngineResult<MlpPointers> {
    match (weights, scalars) {
        (MlpWeights::Nvfp4(weights), MlpScalars::Nvfp4(scalars)) => {
            let gate = arena.address(weights.gate_weight_codes)?;
            let up = arena.address(weights.up_weight_codes)?;
            if up.addr() != gate.addr() + weights.gate_weight_codes.byte_len() {
                return Err(GpuError::invalid_launch(
                    "resident NVFP4 gate/up code planes are not adjacent",
                )
                .into());
            }
            Ok(MlpPointers::Nvfp4(Nvfp4Pointers {
                gate_weight_codes: gate.cast_const(),
                up_weight_codes: up.cast_const(),
                gate_up_weight_scales: arena.address(weights.gate_up_weight_scales)?.cast_const(),
                down_weight_codes: arena.address(weights.down_weight_codes)?.cast_const(),
                down_weight_scales: arena.address(weights.down_weight_scales)?.cast_const(),
                scalars,
            }))
        }
        (MlpWeights::DenseFp8(weights), MlpScalars::DenseFp8) => {
            let map_index = *dense_mlp;
            *dense_mlp = dense_mlp
                .checked_add(1)
                .ok_or_else(|| EngineError::layout("resident dense MLP inventory overflows"))?;
            Ok(MlpPointers::DenseFp8(DenseFp8Pointers {
                gate_up_weight_codes: arena.address(weights.gate_up_weight_codes)?.cast_const(),
                gate_up_weight_scales: arena.address(weights.gate_up_weight_scales)?.cast_const(),
                down_weight_codes: arena.address(weights.down_weight_codes)?.cast_const(),
                down_weight_scales: arena.address(weights.down_weight_scales)?.cast_const(),
                map_index,
            }))
        }
        _ => Err(EngineError::layout(
            "resident MLP scalar route does not match its source route",
        )),
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    norm: &'a ResidualNormOp<Qwen38_27B>,
    gdn_input: &'a GdnInputProjectionOp<Qwen38_27B>,
    gdn_prepare: &'a GdnPrepareOp<Qwen38_27B>,
    gdn_recurrence: &'a GdnRecurrenceOp<Qwen38_27B>,
    gdn_state_snapshot: &'a GdnStateSnapshotOp<Qwen38_27B>,
    gdn_output: &'a GdnOutputProjectionOp<Qwen38_27B>,
    attention_qkv: &'a FullAttentionQkvOp<Qwen38_27B>,
    attention_qk_prepare: &'a AttentionQkPrepareOp<Qwen38_27B>,
    paged_gqa: &'a PagedGqaOp<Qwen38_27B>,
    long_context_paged_gqa: &'a LongContextPagedGqaOp<Qwen38_27B>,
    attention_output: &'a AttentionOutputOp<Qwen38_27B>,
    dense_swiglu: &'a DenseFp8SwiGluOp<Qwen38_27B>,
    dense_down: &'a DenseFp8DownOp<Qwen38_27B>,
    nvfp4_swiglu: &'a Nvfp4SwiGluOp,
    nvfp4_down: &'a Nvfp4DownOp<Qwen38_27B>,
    lm_head: &'a LmHeadOp<Qwen38_27B>,
}

fn capture_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> EngineResult<ResidentGraphs> {
    let mut short = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        let route = ResidentDecodeRoute {
            batch,
            maximum_length: SHORT_CONTEXT_CAPACITY,
            attention: AttentionRoute::Short,
        };
        short.push(CudaGraph::capture(stream, || {
            launch_route(stream, route, ops, pointers, dense_mlp_maps)
        })?);
    }
    let short = short
        .try_into()
        .map_err(|_| EngineError::layout("resident short graph inventory has wrong cardinality"))?;

    let mut long = Vec::with_capacity(LONG_CONTEXT_ROUTE_COUNT);
    for (index, &partitions) in LONG_CONTEXT_GQA_PARTITION_BUCKETS.iter().enumerate() {
        let maximum_length = (partitions * LONG_CONTEXT_GQA_PARTITION_SIZE).min(MAX_CONTEXT_TOKENS);
        let mut graphs = Vec::with_capacity(MAX_BATCH);
        for batch in 1..=MAX_BATCH {
            let route = ResidentDecodeRoute {
                batch,
                maximum_length,
                attention: AttentionRoute::Long { index, partitions },
            };
            graphs.push(CudaGraph::capture(stream, || {
                launch_route(stream, route, ops, pointers, dense_mlp_maps)
            })?);
        }
        long.push(graphs.try_into().map_err(|_| {
            EngineError::layout("resident long graph batch inventory has wrong cardinality")
        })?);
    }
    let long = long.try_into().map_err(|_| {
        EngineError::layout("resident long graph partition inventory has wrong cardinality")
    })?;

    let mut prefill = Vec::with_capacity(PREFILL_GRAPH_ROUTE_COUNT);
    for route in prefill_graph_routes() {
        prefill.push(CudaGraph::capture(stream, || {
            launch_prefill_route(stream, route, ops, pointers, dense_mlp_maps)
        })?);
    }
    let prefill = prefill.try_into().map_err(|_| {
        EngineError::layout("resident prefill graph inventory has wrong cardinality")
    })?;

    let mut target_verify_short = Vec::with_capacity(TARGET_VERIFY_ROUTE_COUNT);
    for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
        let route = ResidentMtpVerifyRoute {
            tokens,
            slot: 0,
            maximum_length: SHORT_CONTEXT_CAPACITY,
            attention: AttentionRoute::Short,
        };
        target_verify_short.push(CudaGraph::capture(stream, || {
            launch_target_mtp_verify(stream, route, ops, pointers, dense_mlp_maps)
        })?);
    }
    let target_verify_short = target_verify_short.try_into().map_err(|_| {
        EngineError::layout("target MTP short graph inventory has wrong cardinality")
    })?;

    let mut target_verify_long = Vec::with_capacity(LONG_CONTEXT_ROUTE_COUNT);
    for (index, &partitions) in LONG_CONTEXT_GQA_PARTITION_BUCKETS.iter().enumerate() {
        let maximum_length = (partitions * LONG_CONTEXT_GQA_PARTITION_SIZE).min(MAX_CONTEXT_TOKENS);
        let mut graphs = Vec::with_capacity(TARGET_VERIFY_ROUTE_COUNT);
        for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
            let route = ResidentMtpVerifyRoute {
                tokens,
                slot: 0,
                maximum_length,
                attention: AttentionRoute::Long { index, partitions },
            };
            graphs.push(CudaGraph::capture(stream, || {
                launch_target_mtp_verify(stream, route, ops, pointers, dense_mlp_maps)
            })?);
        }
        target_verify_long.push(graphs.try_into().map_err(|_| {
            EngineError::layout("target MTP long graph token inventory has wrong cardinality")
        })?);
    }
    let target_verify_long = target_verify_long.try_into().map_err(|_| {
        EngineError::layout("target MTP long graph partition inventory has wrong cardinality")
    })?;

    let mut target_segmented_verify_short = Vec::with_capacity(TARGET_VERIFY_ROUTE_COUNT);
    for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
        let mut graphs = Vec::with_capacity(TARGET_SEGMENTED_BATCH_ROUTES);
        for batch in 2..=MAX_BATCH {
            let route = ResidentMtpSegmentedVerifyRoute {
                tokens,
                batch,
                maximum_length: SHORT_CONTEXT_CAPACITY,
                attention: AttentionRoute::Short,
            };
            graphs.push(CudaGraph::capture(stream, || {
                launch_target_mtp_segmented_verify(stream, route, ops, pointers, dense_mlp_maps)
            })?);
        }
        target_segmented_verify_short.push(graphs.try_into().map_err(|_| {
            EngineError::layout("segmented target MTP short batch inventory differs")
        })?);
    }
    let target_segmented_verify_short = target_segmented_verify_short
        .try_into()
        .map_err(|_| EngineError::layout("segmented target MTP short token inventory differs"))?;

    let mut target_segmented_verify_long = Vec::with_capacity(TARGET_VERIFY_ROUTE_COUNT);
    for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
        let mut batch_graphs = Vec::with_capacity(TARGET_SEGMENTED_BATCH_ROUTES);
        for batch in 2..=MAX_BATCH {
            let mut definitions = Vec::with_capacity(LONG_CONTEXT_ROUTE_COUNT);
            for (index, &partitions) in LONG_CONTEXT_GQA_PARTITION_BUCKETS.iter().enumerate() {
                let maximum_length =
                    (partitions * LONG_CONTEXT_GQA_PARTITION_SIZE).min(MAX_CONTEXT_TOKENS);
                let route = ResidentMtpSegmentedVerifyRoute {
                    tokens,
                    batch,
                    maximum_length,
                    attention: AttentionRoute::Long { index, partitions },
                };
                definitions.push(CudaGraphDefinition::capture(stream, || {
                    launch_target_mtp_segmented_verify(stream, route, ops, pointers, dense_mlp_maps)
                })?);
            }
            batch_graphs.push(CudaGraphVariants::new(definitions.try_into().map_err(
                |_| EngineError::layout("segmented target MTP long partition inventory differs"),
            )?)?);
        }
        target_segmented_verify_long.push(batch_graphs.try_into().map_err(|_| {
            EngineError::layout("segmented target MTP long batch inventory differs")
        })?);
    }
    let target_segmented_verify_long = target_segmented_verify_long
        .try_into()
        .map_err(|_| EngineError::layout("segmented target MTP long token inventory differs"))?;

    let mut target_commit = Vec::with_capacity(TARGET_VERIFY_ROUTE_COUNT);
    for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
        target_commit.push(CudaGraph::capture(stream, || {
            launch_target_mtp_commit(stream, tokens, ops, pointers)
        })?);
    }
    let target_commit = target_commit.try_into().map_err(|_| {
        EngineError::layout("target MTP commit graph inventory has wrong cardinality")
    })?;

    Ok(ResidentGraphs {
        short,
        long,
        prefill,
        target_verify_short,
        target_verify_long,
        target_segmented_verify_short,
        target_segmented_verify_long,
        target_commit,
    })
}

fn launch_route(
    stream: &CudaStream,
    route: ResidentDecodeRoute,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    let batch = route.batch;
    let workspace = pointers.workspace;
    let first = pointers
        .layers
        .first()
        .ok_or_else(|| GpuError::invalid_launch("resident layer inventory is empty"))?;
    // SAFETY: all captured pointers name checked maximum-batch regions in one
    // live arena. The exact route bounds every leaf launch to active rows.
    unsafe {
        ops.norm.launch_plain(
            stream,
            batch,
            workspace.residual_a,
            first.mixer.input_norm(),
            workspace.mixer_normalized,
        )?;
    }

    let mut residual_input = workspace.residual_a;
    for (index, layer) in pointers.layers.iter().enumerate() {
        launch_mixer(stream, route, ops, workspace, layer.mixer)?;
        // SAFETY: mixer output and both residual seams are disjoint maximum-B planes.
        unsafe {
            ops.norm.launch_residual(
                stream,
                batch,
                residual_input,
                workspace.mixer_branch,
                layer.mixer.post_attention_norm(),
                workspace.mixer_residual,
                workspace.mlp_normalized,
            )?;
        }
        launch_mlp(stream, batch, ops, workspace, layer.mlp, dense_mlp_maps)?;

        let residual_output = if index.is_multiple_of(2) {
            workspace.residual_b
        } else {
            workspace.residual_a
        };
        let next_norm = pointers
            .layers
            .get(index + 1)
            .map_or(pointers.endpoint.final_norm, |next| next.mixer.input_norm());
        // SAFETY: the residual ping-pong planes never alias the branch planes.
        unsafe {
            ops.norm.launch_residual(
                stream,
                batch,
                workspace.mixer_residual,
                workspace.mlp_branch,
                next_norm,
                residual_output,
                workspace.mixer_normalized,
            )?;
        }
        residual_input = residual_output;
    }

    if residual_input != workspace.residual_a {
        return Err(GpuError::invalid_launch(
            "resident even-layer schedule did not return to residual A",
        ));
    }
    // SAFETY: layer 63 prepared final-normalized rows in `mixer_normalized`.
    unsafe {
        ops.lm_head.launch(
            stream,
            batch,
            workspace.mixer_normalized,
            workspace.activation_codes,
            workspace.activation_scales,
            pointers.endpoint.lm_head_codes,
            pointers.endpoint.lm_head_scales,
            workspace.logits,
        )
    }
}

fn launch_target_mtp_verify(
    stream: &CudaStream,
    route: ResidentMtpVerifyRoute,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    launch_target_mtp_verify_lane(
        stream,
        route.tokens,
        0,
        route.maximum_length,
        route.attention,
        ops,
        pointers,
        dense_mlp_maps,
    )
}

fn launch_target_mtp_segmented_verify(
    stream: &CudaStream,
    route: ResidentMtpSegmentedVerifyRoute,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    for lane in 0..route.batch {
        launch_target_mtp_verify_lane(
            stream,
            route.tokens,
            lane,
            route.maximum_length,
            route.attention,
            ops,
            pointers,
            dense_mlp_maps,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_target_mtp_verify_lane(
    stream: &CudaStream,
    tokens: usize,
    lane: usize,
    maximum_length: usize,
    attention: AttentionRoute,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    let row_offset = lane * tokens;
    let workspace = pointers.workspace.row_offset(row_offset);
    let first = pointers
        .layers
        .first()
        .ok_or_else(|| GpuError::invalid_launch("resident layer inventory is empty"))?;
    // Each lane retains the exact K=1..4 leaf topology. Independent row offsets,
    // provisional state, replay records, and page-table metadata prevent aliasing.
    unsafe {
        ops.norm.launch_plain(
            stream,
            tokens,
            workspace.residual_a,
            first.mixer.input_norm(),
            workspace.mixer_normalized,
        )?;
    }

    let mut residual_input = workspace.residual_a;
    let mut gdn_layer = 0;
    for (index, layer) in pointers.layers.iter().enumerate() {
        launch_target_mtp_mixer(
            stream,
            tokens,
            lane,
            maximum_length,
            attention,
            ops,
            workspace,
            layer.mixer,
            &mut gdn_layer,
        )?;
        // SAFETY: branch and residual planes are disjoint maximum-row regions.
        unsafe {
            ops.norm.launch_residual(
                stream,
                tokens,
                residual_input,
                workspace.mixer_branch,
                layer.mixer.post_attention_norm(),
                workspace.mixer_residual,
                workspace.mlp_normalized,
            )?;
        }
        launch_mlp(stream, tokens, ops, workspace, layer.mlp, dense_mlp_maps)?;

        let residual_output = if index.is_multiple_of(2) {
            workspace.residual_b
        } else {
            workspace.residual_a
        };
        let next_norm = pointers
            .layers
            .get(index + 1)
            .map_or(pointers.endpoint.final_norm, |next| next.mixer.input_norm());
        // SAFETY: residual ping-pong and both branch planes do not alias.
        unsafe {
            ops.norm.launch_residual(
                stream,
                tokens,
                workspace.mixer_residual,
                workspace.mlp_branch,
                next_norm,
                residual_output,
                workspace.mixer_normalized,
            )?;
        }
        residual_input = residual_output;
    }

    if gdn_layer != GDN_LAYER_COUNT {
        return Err(GpuError::invalid_launch(format!(
            "target MTP verification visited {gdn_layer} GDN layers, expected {GDN_LAYER_COUNT}"
        )));
    }
    if residual_input != workspace.residual_a {
        return Err(GpuError::invalid_launch(
            "target MTP even-layer schedule did not return to residual A",
        ));
    }
    // Every row judges the following draft token; K=4 row three is the bonus
    // distribution, so verification owns a complete K-wide logits plane.
    unsafe {
        ops.lm_head.launch(
            stream,
            tokens,
            workspace.mixer_normalized,
            workspace.activation_codes,
            workspace.activation_scales,
            pointers.endpoint.lm_head_codes,
            pointers.endpoint.lm_head_scales,
            workspace.logits,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_target_mtp_mixer(
    stream: &CudaStream,
    tokens: usize,
    lane: usize,
    maximum_length: usize,
    attention: AttentionRoute,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mixer: MixerPointers,
    gdn_layer: &mut usize,
) -> GpuResult<()> {
    // SAFETY: K=1..4 scratch and replay planes are address-stable and consumed
    // before reuse. Attention rows append distinct positions to one table row.
    unsafe {
        match mixer {
            MixerPointers::Gdn(p) => {
                let layer = *gdn_layer;
                *gdn_layer = gdn_layer
                    .checked_add(1)
                    .ok_or_else(|| GpuError::invalid_launch("target GDN index overflows"))?;
                if layer >= GDN_LAYER_COUNT {
                    return Err(GpuError::invalid_launch(
                        "target GDN layer exceeds replay inventory",
                    ));
                }
                let record = WorkspacePointers::target_record_offset(lane);
                let projected = workspace
                    .recorded_projected_for(layer)
                    .wrapping_add(record * Qwen38_27B::GDN_INPUT_ROWS);
                let log_decay = workspace
                    .recorded_log_decay_for(layer)
                    .wrapping_add(record * Qwen38_27B::GDN_CONTROL_ROWS);
                let beta = workspace
                    .recorded_beta_for(layer)
                    .wrapping_add(record * Qwen38_27B::GDN_CONTROL_ROWS);
                let history_values =
                    Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
                let state_values = Qwen38_27B::GDN_CONTROL_ROWS
                    * Qwen38_27B::LINEAR_HEAD_DIM
                    * Qwen38_27B::LINEAR_HEAD_DIM;
                let provisional_history = workspace
                    .provisional_history
                    .wrapping_add(lane * history_values);
                let provisional_state = workspace
                    .provisional_state
                    .wrapping_add(lane * state_values);
                ops.gdn_state_snapshot.launch(
                    stream,
                    workspace.state_rows,
                    p.history,
                    p.state,
                    provisional_history,
                    provisional_state,
                )?;
                ops.gdn_input.launch(
                    stream,
                    tokens,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.input_weight_codes,
                    p.input_weight_scales,
                    projected,
                )?;
                ops.gdn_prepare.launch_causal(
                    stream,
                    tokens,
                    workspace.mixer_normalized,
                    p.control_weights,
                    p.a_log,
                    p.dt_bias,
                    projected,
                    p.convolution_weights,
                    workspace.provisional_state_row,
                    provisional_history,
                    log_decay,
                    beta,
                    workspace.convolved,
                )?;
                ops.gdn_recurrence.launch_causal(
                    stream,
                    tokens,
                    workspace.convolved,
                    projected,
                    log_decay,
                    beta,
                    p.recurrent_norm,
                    workspace.provisional_state_row,
                    provisional_state,
                    workspace.recurrent_output,
                )?;
                ops.gdn_output.launch(
                    stream,
                    tokens,
                    workspace.recurrent_output,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
            MixerPointers::Attention(p) => {
                ops.attention_qkv.launch(
                    stream,
                    tokens,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.qkv_weight_codes,
                    p.qkv_weight_scales,
                    workspace.projected,
                )?;
                ops.attention_qk_prepare.launch(
                    stream,
                    tokens,
                    workspace.projected,
                    p.query_norm,
                    p.key_norm,
                    workspace.rope_cos,
                    workspace.rope_sin,
                    workspace.block_tables,
                    workspace.table_rows,
                    LONG_CONTEXT_PHYSICAL_PAGES,
                    workspace.cache_positions,
                    workspace.query,
                    p.key_pages,
                    p.value_pages,
                    p.scalars.key_cache_scale,
                    p.scalars.value_cache_scale,
                )?;
                match attention {
                    AttentionRoute::Short => ops.paged_gqa.launch(
                        stream,
                        tokens,
                        workspace.query,
                        p.key_pages,
                        p.value_pages,
                        workspace.block_tables,
                        workspace.table_rows,
                        LONG_CONTEXT_PHYSICAL_PAGES,
                        workspace.lengths,
                        workspace.attention,
                        p.scalars.key_cache_scale,
                        p.scalars.value_cache_scale,
                    )?,
                    AttentionRoute::Long { .. } => ops.long_context_paged_gqa.launch(
                        stream,
                        tokens,
                        maximum_length,
                        workspace.query,
                        p.key_pages,
                        p.value_pages,
                        workspace.block_tables,
                        workspace.table_rows,
                        LONG_CONTEXT_PHYSICAL_PAGES,
                        workspace.lengths,
                        workspace.partial_maximum,
                        workspace.partial_denominator,
                        workspace.partial_numerator,
                        workspace.attention,
                        p.scalars.key_cache_scale,
                        p.scalars.value_cache_scale,
                    )?,
                }
                ops.attention_output.launch(
                    stream,
                    tokens,
                    workspace.attention,
                    workspace.projected,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
        }
    }
}

fn launch_target_mtp_commit(
    stream: &CudaStream,
    tokens: usize,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
) -> GpuResult<()> {
    let workspace = pointers.workspace;
    let mut gdn_layer = 0;
    for layer in &pointers.layers {
        let MixerPointers::Gdn(p) = layer.mixer else {
            continue;
        };
        if gdn_layer >= GDN_LAYER_COUNT {
            return Err(GpuError::invalid_launch(
                "target commit GDN layer exceeds replay inventory",
            ));
        }
        let projected = workspace.recorded_projected_for(gdn_layer);
        let log_decay = workspace.recorded_log_decay_for(gdn_layer);
        let beta = workspace.recorded_beta_for(gdn_layer);
        // The represented projected/control planes came from the provisional
        // target run; only causal convolution and recurrence mutate live state.
        unsafe {
            ops.gdn_prepare.launch_causal_replay(
                stream,
                tokens,
                projected,
                p.convolution_weights,
                workspace.state_rows,
                p.history,
                workspace.convolved,
            )?;
            ops.gdn_recurrence.launch_causal(
                stream,
                tokens,
                workspace.convolved,
                projected,
                log_decay,
                beta,
                p.recurrent_norm,
                workspace.state_rows,
                p.state,
                workspace.recurrent_output,
            )?;
        }
        gdn_layer += 1;
    }
    if gdn_layer != GDN_LAYER_COUNT {
        return Err(GpuError::invalid_launch(format!(
            "target MTP commit visited {gdn_layer} GDN layers, expected {GDN_LAYER_COUNT}"
        )));
    }

    Ok(())
}

fn launch_target_mtp_segmented_commit(
    stream: &CudaStream,
    route: ResidentMtpSegmentedVerifyRoute,
    accepted_tokens: &[usize],
    ops: Ops<'_>,
    pointers: &ProgramPointers,
) -> GpuResult<()> {
    let mut gdn_layer = 0;
    for layer in &pointers.layers {
        let MixerPointers::Gdn(p) = layer.mixer else {
            continue;
        };
        if gdn_layer >= GDN_LAYER_COUNT {
            return Err(GpuError::invalid_launch(
                "segmented target commit GDN layer exceeds replay inventory",
            ));
        }
        for (lane, &tokens) in accepted_tokens.iter().enumerate() {
            let row = lane * route.tokens;
            let workspace = pointers.workspace.row_offset(row);
            let record = WorkspacePointers::target_record_offset(lane);
            let projected = pointers
                .workspace
                .recorded_projected_for(gdn_layer)
                .wrapping_add(record * Qwen38_27B::GDN_INPUT_ROWS);
            let log_decay = pointers
                .workspace
                .recorded_log_decay_for(gdn_layer)
                .wrapping_add(record * Qwen38_27B::GDN_CONTROL_ROWS);
            let beta = pointers
                .workspace
                .recorded_beta_for(gdn_layer)
                .wrapping_add(record * Qwen38_27B::GDN_CONTROL_ROWS);
            // Per-lane replay retains the exact K=1..4 convolution and recurrence
            // arithmetic while records and live state rows remain disjoint.
            unsafe {
                ops.gdn_prepare.launch_causal_replay(
                    stream,
                    tokens,
                    projected,
                    p.convolution_weights,
                    workspace.state_rows,
                    p.history,
                    workspace.convolved,
                )?;
                ops.gdn_recurrence.launch_causal(
                    stream,
                    tokens,
                    workspace.convolved,
                    projected,
                    log_decay,
                    beta,
                    p.recurrent_norm,
                    workspace.state_rows,
                    p.state,
                    workspace.recurrent_output,
                )?;
            }
        }
        gdn_layer += 1;
    }
    if gdn_layer != GDN_LAYER_COUNT {
        return Err(GpuError::invalid_launch(format!(
            "segmented target MTP commit visited {gdn_layer} GDN layers, expected {GDN_LAYER_COUNT}"
        )));
    }
    Ok(())
}

fn launch_prefill_route(
    stream: &CudaStream,
    route: ResidentPrefillRoute,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    let rows = route.tokens;
    let workspace = pointers.workspace;
    let first = pointers
        .layers
        .first()
        .ok_or_else(|| GpuError::invalid_launch("resident layer inventory is empty"))?;
    // SAFETY: exact T routes bound all shared MAX_ROWS planes, and mapped
    // metadata selects one persistent slot and one live page-table row.
    unsafe {
        ops.norm.launch_plain(
            stream,
            rows,
            workspace.residual_a,
            first.mixer.input_norm(),
            workspace.mixer_normalized,
        )?;
    }

    let mut residual_input = workspace.residual_a;
    for (index, layer) in pointers.layers.iter().enumerate() {
        launch_prefill_mixer(stream, route, ops, workspace, layer.mixer)?;
        // SAFETY: branch and residual planes are disjoint MAX_ROWS regions.
        unsafe {
            ops.norm.launch_residual(
                stream,
                rows,
                residual_input,
                workspace.mixer_branch,
                layer.mixer.post_attention_norm(),
                workspace.mixer_residual,
                workspace.mlp_normalized,
            )?;
        }
        launch_mlp(stream, rows, ops, workspace, layer.mlp, dense_mlp_maps)?;

        let residual_output = if index.is_multiple_of(2) {
            workspace.residual_b
        } else {
            workspace.residual_a
        };
        let next_norm = pointers
            .layers
            .get(index + 1)
            .map_or(pointers.endpoint.final_norm, |next| next.mixer.input_norm());
        // SAFETY: residual ping-pong and both branch planes do not alias.
        unsafe {
            ops.norm.launch_residual(
                stream,
                rows,
                workspace.mixer_residual,
                workspace.mlp_branch,
                next_norm,
                residual_output,
                workspace.mixer_normalized,
            )?;
        }
        residual_input = residual_output;
    }

    if residual_input != workspace.residual_a {
        return Err(GpuError::invalid_launch(
            "resident even-layer prefill schedule did not return to residual A",
        ));
    }
    // Only the final prompt row feeds sampling; this avoids a 508,559,360-byte
    // T-wide logits plane at T=1024 without changing the model boundary.
    unsafe {
        let final_row = workspace
            .mixer_normalized
            .add((rows - 1) * Qwen38_27B::HIDDEN);
        ops.lm_head.launch(
            stream,
            1,
            final_row,
            workspace.activation_codes,
            workspace.activation_scales,
            pointers.endpoint.lm_head_codes,
            pointers.endpoint.lm_head_scales,
            workspace.logits,
        )
    }
}

fn launch_prefill_mixer(
    stream: &CudaStream,
    route: ResidentPrefillRoute,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mixer: MixerPointers,
) -> GpuResult<()> {
    let rows = route.tokens;
    // SAFETY: shared scratch is consumed before reuse. All prefill rows map to
    // one persistent slot; GDN kernels advance it causally in token order.
    unsafe {
        match mixer {
            MixerPointers::Gdn(p) => {
                ops.gdn_input.launch(
                    stream,
                    rows,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.input_weight_codes,
                    p.input_weight_scales,
                    workspace.projected,
                )?;
                ops.gdn_prepare.launch(
                    stream,
                    rows,
                    workspace.mixer_normalized,
                    p.control_weights,
                    p.a_log,
                    p.dt_bias,
                    workspace.projected,
                    p.convolution_weights,
                    workspace.state_rows,
                    p.history,
                    workspace.log_decay,
                    workspace.beta,
                    workspace.convolved,
                )?;
                ops.gdn_recurrence.launch(
                    stream,
                    rows,
                    workspace.convolved,
                    workspace.projected,
                    workspace.log_decay,
                    workspace.beta,
                    p.recurrent_norm,
                    workspace.state_rows,
                    p.state,
                    workspace.recurrent_output,
                )?;
                ops.gdn_output.launch(
                    stream,
                    rows,
                    workspace.recurrent_output,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
            MixerPointers::Attention(p) => {
                ops.attention_qkv.launch(
                    stream,
                    rows,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.qkv_weight_codes,
                    p.qkv_weight_scales,
                    workspace.projected,
                )?;
                ops.attention_qk_prepare.launch(
                    stream,
                    rows,
                    workspace.projected,
                    p.query_norm,
                    p.key_norm,
                    workspace.rope_cos,
                    workspace.rope_sin,
                    workspace.block_tables,
                    workspace.table_rows,
                    LONG_CONTEXT_PHYSICAL_PAGES,
                    workspace.cache_positions,
                    workspace.query,
                    p.key_pages,
                    p.value_pages,
                    p.scalars.key_cache_scale,
                    p.scalars.value_cache_scale,
                )?;
                match route.attention {
                    PrefillAttentionRoute::Shared => ops.paged_gqa.launch_prefill_shared(
                        stream,
                        rows,
                        workspace.query,
                        p.key_pages,
                        p.value_pages,
                        workspace.block_tables,
                        workspace.table_rows,
                        LONG_CONTEXT_PHYSICAL_PAGES,
                        workspace.lengths,
                        workspace.attention,
                        p.scalars.key_cache_scale,
                        p.scalars.value_cache_scale,
                    )?,
                    PrefillAttentionRoute::Partitioned { partitions } => {
                        debug_assert_eq!(partitions, route.partition_capacity().unwrap());
                        ops.paged_gqa.launch_prefill_partitioned(
                            stream,
                            route.context_tokens,
                            workspace.query,
                            p.key_pages,
                            p.value_pages,
                            workspace.block_tables,
                            workspace.table_rows,
                            LONG_CONTEXT_PHYSICAL_PAGES,
                            workspace.lengths,
                            workspace.prefill_partials,
                            workspace.attention,
                            p.scalars.key_cache_scale,
                            p.scalars.value_cache_scale,
                        )?;
                    }
                    PrefillAttentionRoute::Macro { partitions } => {
                        ops.paged_gqa.launch_prefill_macro(
                            stream,
                            partitions,
                            workspace.query,
                            p.key_pages,
                            p.value_pages,
                            workspace.block_tables,
                            workspace.table_rows,
                            LONG_CONTEXT_PHYSICAL_PAGES,
                            workspace.lengths,
                            workspace.prefill_partials,
                            workspace.attention,
                            p.scalars.key_cache_scale,
                            p.scalars.value_cache_scale,
                        )?
                    }
                }
                ops.attention_output.launch(
                    stream,
                    rows,
                    workspace.attention,
                    workspace.projected,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
        }
    }
}

fn launch_mixer(
    stream: &CudaStream,
    route: ResidentDecodeRoute,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mixer: MixerPointers,
) -> GpuResult<()> {
    let batch = route.batch;
    // SAFETY: the shared scratch planes are reused only after the preceding
    // launch has consumed them, and each persistent plane belongs to this layer.
    unsafe {
        match mixer {
            MixerPointers::Gdn(p) => {
                ops.gdn_input.launch(
                    stream,
                    batch,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.input_weight_codes,
                    p.input_weight_scales,
                    workspace.projected,
                )?;
                ops.gdn_prepare.launch(
                    stream,
                    batch,
                    workspace.mixer_normalized,
                    p.control_weights,
                    p.a_log,
                    p.dt_bias,
                    workspace.projected,
                    p.convolution_weights,
                    workspace.state_rows,
                    p.history,
                    workspace.log_decay,
                    workspace.beta,
                    workspace.convolved,
                )?;
                ops.gdn_recurrence.launch(
                    stream,
                    batch,
                    workspace.convolved,
                    workspace.projected,
                    workspace.log_decay,
                    workspace.beta,
                    p.recurrent_norm,
                    workspace.state_rows,
                    p.state,
                    workspace.recurrent_output,
                )?;
                ops.gdn_output.launch(
                    stream,
                    batch,
                    workspace.recurrent_output,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
            MixerPointers::Attention(p) => {
                ops.attention_qkv.launch(
                    stream,
                    batch,
                    workspace.mixer_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.qkv_weight_codes,
                    p.qkv_weight_scales,
                    workspace.projected,
                )?;
                ops.attention_qk_prepare.launch(
                    stream,
                    batch,
                    workspace.projected,
                    p.query_norm,
                    p.key_norm,
                    workspace.rope_cos,
                    workspace.rope_sin,
                    workspace.block_tables,
                    workspace.table_rows,
                    LONG_CONTEXT_PHYSICAL_PAGES,
                    workspace.cache_positions,
                    workspace.query,
                    p.key_pages,
                    p.value_pages,
                    p.scalars.key_cache_scale,
                    p.scalars.value_cache_scale,
                )?;
                match route.attention {
                    AttentionRoute::Short => ops.paged_gqa.launch(
                        stream,
                        batch,
                        workspace.query,
                        p.key_pages,
                        p.value_pages,
                        workspace.block_tables,
                        workspace.table_rows,
                        LONG_CONTEXT_PHYSICAL_PAGES,
                        workspace.lengths,
                        workspace.attention,
                        p.scalars.key_cache_scale,
                        p.scalars.value_cache_scale,
                    )?,
                    AttentionRoute::Long { .. } => ops.long_context_paged_gqa.launch(
                        stream,
                        batch,
                        route.maximum_length,
                        workspace.query,
                        p.key_pages,
                        p.value_pages,
                        workspace.block_tables,
                        workspace.table_rows,
                        LONG_CONTEXT_PHYSICAL_PAGES,
                        workspace.lengths,
                        workspace.partial_maximum,
                        workspace.partial_denominator,
                        workspace.partial_numerator,
                        workspace.attention,
                        p.scalars.key_cache_scale,
                        p.scalars.value_cache_scale,
                    )?,
                }
                ops.attention_output.launch(
                    stream,
                    batch,
                    workspace.attention,
                    workspace.projected,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.output_weight_codes,
                    p.output_weight_scales,
                    workspace.mixer_branch,
                )
            }
        }
    }
}

fn launch_mlp(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mlp: MlpPointers,
    dense_mlp_maps: &[DenseMlpMaps],
) -> GpuResult<()> {
    // SAFETY: all weights are source-route matched and shared scratch covers MAX_ROWS.
    unsafe {
        match mlp {
            MlpPointers::Nvfp4(p) => {
                ops.nvfp4_swiglu.launch(
                    stream,
                    rows,
                    workspace.mlp_normalized,
                    workspace.nvfp4_activation_codes,
                    workspace.nvfp4_activation_scales,
                    p.gate_weight_codes,
                    p.gate_up_weight_scales,
                    p.scalars.gate_up_input,
                    p.scalars.gate_up_weight,
                    workspace.swiglu,
                )?;
                if rows <= MAX_BATCH {
                    // Decode preserves the BF16 down input.
                    ops.nvfp4_down.launch(
                        stream,
                        rows,
                        workspace.swiglu,
                        p.down_weight_codes,
                        p.down_weight_scales,
                        p.scalars.down_weight,
                        workspace.mlp_branch,
                    )
                } else {
                    ops.nvfp4_down.launch_prefill(
                        stream,
                        rows,
                        workspace.swiglu,
                        workspace.nvfp4_activation_codes,
                        workspace.nvfp4_activation_scales,
                        p.down_weight_codes,
                        p.down_weight_scales,
                        p.scalars.down_input,
                        p.scalars.down_weight,
                        workspace.mlp_branch,
                    )
                }
            }
            MlpPointers::DenseFp8(p) => {
                if rows == super::MAX_ROWS {
                    let maps = dense_mlp_maps.get(p.map_index).ok_or_else(|| {
                        GpuError::invalid_launch(
                            "resident dense MLP tensor-map index is outside its owner inventory",
                        )
                    })?;
                    // T=1024 amortizes address-bound TMA setup across the macro tiles.
                    ops.dense_swiglu.launch_macro_prefill(
                        stream,
                        workspace.mlp_normalized,
                        workspace.activation_codes,
                        workspace.activation_scales,
                        p.gate_up_weight_codes,
                        p.gate_up_weight_scales,
                        workspace.swiglu,
                        &maps.gate_up,
                    )?;
                    ops.dense_down.launch_macro_prefill(
                        stream,
                        workspace.swiglu,
                        workspace.activation_codes,
                        workspace.activation_scales,
                        p.down_weight_codes,
                        p.down_weight_scales,
                        workspace.mlp_branch,
                        &maps.down,
                    )
                } else {
                    ops.dense_swiglu.launch(
                        stream,
                        rows,
                        workspace.mlp_normalized,
                        workspace.activation_codes,
                        workspace.activation_scales,
                        p.gate_up_weight_codes,
                        p.gate_up_weight_scales,
                        workspace.swiglu,
                    )?;
                    if rows <= MAX_BATCH {
                        ops.dense_down.launch(
                            stream,
                            rows,
                            workspace.swiglu,
                            workspace.activation_codes,
                            workspace.activation_scales,
                            p.down_weight_codes,
                            p.down_weight_scales,
                            workspace.mlp_branch,
                        )
                    } else {
                        ops.dense_down.launch_tail_prefill(
                            stream,
                            rows,
                            workspace.swiglu,
                            workspace.activation_codes,
                            workspace.activation_scales,
                            p.down_weight_codes,
                            p.down_weight_scales,
                            workspace.mlp_branch,
                        )
                    }
                }
            }
        }
    }
}

trait ResidentWeightSink {
    fn copy_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[T],
    ) -> EngineResult<()>;

    fn copy_bytes_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[u8],
    ) -> EngineResult<()>;
}

struct LegacyWeightSink<'a> {
    arena: &'a DeviceArena,
    copy_ns: u64,
}

impl ResidentWeightSink for LegacyWeightSink<'_> {
    fn copy_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[T],
    ) -> EngineResult<()> {
        let started = Instant::now();
        self.arena.copy_from_host(stream, region, source)?;
        self.copy_ns = self
            .copy_ns
            .checked_add(elapsed_ns("legacy weight copy", started)?)
            .ok_or_else(|| EngineError::layout("legacy weight-copy time overflows"))?;
        Ok(())
    }

    fn copy_bytes_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[u8],
    ) -> EngineResult<()> {
        let started = Instant::now();
        self.arena
            .copy_region_bytes_from_host(stream, region, source)?;
        self.copy_ns = self
            .copy_ns
            .checked_add(elapsed_ns("legacy weight copy", started)?)
            .ok_or_else(|| EngineError::layout("legacy weight-copy time overflows"))?;
        Ok(())
    }
}

struct SelectiveWeightSink<'a> {
    arena: &'a mut LoadingDeviceArena,
    plan: &'a ResidentUploadPlan,
    bytes: usize,
    submissions: usize,
    copy_ns: u64,
    progress: Option<&'a ResidentLoadProgress>,
}

impl SelectiveWeightSink<'_> {
    fn copy_source<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        offset: usize,
        bytes: usize,
        source: &[T],
    ) -> EngineResult<()> {
        let started = Instant::now();
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| EngineError::layout("selective weight destination overflows"))?;
        match self
            .plan
            .preparation_for(ResidentUploadArena::Resident, offset, bytes)?
        {
            ResidentUploadPreparation::BorrowedSource => {
                // SAFETY: borrowed upload-plan entries point into the admitted snapshot mmaps;
                // `ResidentModelProgram` retains that snapshot beyond the final arena seal.
                unsafe {
                    self.arena
                        .copy_from_host_async(stream, offset..end, source)?;
                }
            }
            ResidentUploadPreparation::GatheredSource
            | ResidentUploadPreparation::SwizzledSource => {
                // SAFETY: synchronization below completes the copy before its temporary source
                // can be released by the caller.
                unsafe {
                    self.arena
                        .copy_from_host_async(stream, offset..end, source)?;
                }
                stream.synchronize().map_err(GpuError::from)?;
            }
            preparation => {
                return Err(EngineError::layout(format!(
                    "weight destination unexpectedly requires {preparation:?} preparation"
                )));
            }
        }
        self.copy_ns = self
            .copy_ns
            .checked_add(elapsed_ns("selective weight copy", started)?)
            .ok_or_else(|| EngineError::layout("selective weight-copy time overflows"))?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| EngineError::layout("resident upload bytes overflow"))?;
        self.submissions = self
            .submissions
            .checked_add(1)
            .ok_or_else(|| EngineError::layout("resident upload submissions overflow"))?;
        if let Some(progress) = self.progress {
            progress.submit(bytes)?;
        }
        Ok(())
    }
}

impl ResidentWeightSink for SelectiveWeightSink<'_> {
    fn copy_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[T],
    ) -> EngineResult<()> {
        self.copy_source(stream, region.offset_bytes(), region.byte_len(), source)
    }

    fn copy_bytes_from_host<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        region: ArenaRegion<T>,
        source: &[u8],
    ) -> EngineResult<()> {
        self.copy_source(stream, region.offset_bytes(), region.byte_len(), source)
    }
}

fn load_source_weights<S: ResidentWeightSink>(
    arena: &mut S,
    stream: &CudaStream,
    layout: &ResidentModelLayout,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
    preparation: &mut ResidentPreparationStats,
) -> EngineResult<Vec<LayerScalars>> {
    let mut scalars = Vec::with_capacity(layout.layers.len());
    for (layer_index, layer) in layout.layers.iter().enumerate() {
        let mixer = load_mixer(
            arena,
            stream,
            layer_index,
            layer.mixer,
            snapshot,
            preparation,
        )?;
        let mlp = load_mlp(arena, stream, layer_index, layer.mlp, snapshot, preparation)?;
        scalars.push(LayerScalars { mixer, mlp });
    }
    let endpoint = measure_preparation(
        &mut preparation.source_binding_ns,
        "resident endpoint source binding",
        || Ok(TextEndpointBindings::bind(snapshot)?),
    )?;
    arena.copy_bytes_from_host(
        stream,
        layout.endpoint.final_norm,
        endpoint.final_norm.bytes(),
    )?;
    arena.copy_from_host(
        stream,
        layout.endpoint.lm_head_codes,
        endpoint.lm_head.codes(),
    )?;
    arena.copy_bytes_from_host(
        stream,
        layout.endpoint.lm_head_scales,
        endpoint.lm_head_scale.bytes(),
    )?;

    Ok(scalars)
}

fn load_mixer<S: ResidentWeightSink>(
    arena: &mut S,
    stream: &CudaStream,
    layer: usize,
    weights: MixerWeights,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
    preparation: &mut ResidentPreparationStats,
) -> EngineResult<MixerScalars> {
    match weights {
        MixerWeights::Gdn(weights) => {
            let source = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident GDN source binding",
                || Ok(GdnBindings::bind(snapshot, layer)?),
            )?;
            arena.copy_bytes_from_host(stream, weights.input_norm, source.input_norm.bytes())?;
            arena.copy_from_host(stream, weights.input_weight_codes, source.input_weight_e4m3)?;
            arena.copy_bytes_from_host(
                stream,
                weights.input_weight_scales,
                source.input_scale_bf16,
            )?;
            let mut control = source.a_control_weight.words().collect::<Vec<_>>();
            control.extend(source.b_control_weight.words());
            arena.copy_from_host(stream, weights.control_weights, &control)?;
            arena.copy_bytes_from_host(stream, weights.a_log, source.a_log.bytes())?;
            arena.copy_bytes_from_host(stream, weights.dt_bias, source.dt_bias.bytes())?;
            arena.copy_bytes_from_host(
                stream,
                weights.convolution_weights,
                source.convolution_weight.bytes(),
            )?;
            arena.copy_bytes_from_host(stream, weights.recurrent_norm, source.norm.bytes())?;
            arena.copy_from_host(
                stream,
                weights.output_weight_codes,
                source.output_weight.codes(),
            )?;
            arena.copy_bytes_from_host(
                stream,
                weights.output_weight_scales,
                source.output_scale.bytes(),
            )?;
            arena.copy_bytes_from_host(
                stream,
                weights.post_attention_norm,
                source.post_attention_norm.bytes(),
            )?;
            Ok(MixerScalars::Gdn)
        }
        MixerWeights::Attention(weights) => {
            let qkv = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident attention QKV source binding",
                || Ok(FullAttentionQkvBindings::bind(snapshot, layer)?),
            )?;
            let qkv = measure_preparation(
                &mut preparation.qkv_gather_ns,
                "resident attention QKV gather",
                || Ok(qkv.materialize()?),
            )?;
            let source = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident attention post source binding",
                || Ok(FullAttentionPostBindings::bind(snapshot, layer)?),
            )?;
            arena.copy_bytes_from_host(stream, weights.input_norm, source.input_norm.bytes())?;
            arena.copy_from_host(stream, weights.qkv_weight_codes, &qkv.weight_e4m3)?;
            arena.copy_bytes_from_host(stream, weights.qkv_weight_scales, &qkv.scale_bf16)?;
            arena.copy_bytes_from_host(stream, weights.query_norm, source.query_norm.bytes())?;
            arena.copy_bytes_from_host(stream, weights.key_norm, source.key_norm.bytes())?;
            arena.copy_from_host(
                stream,
                weights.output_weight_codes,
                source.output_weight.codes(),
            )?;
            arena.copy_bytes_from_host(
                stream,
                weights.output_weight_scales,
                source.output_scale.bytes(),
            )?;
            arena.copy_bytes_from_host(
                stream,
                weights.post_attention_norm,
                source.post_attention_norm.bytes(),
            )?;
            Ok(MixerScalars::Attention(AttentionScalars {
                key_cache_scale: bf16_to_f32(source.key_cache_scale_bf16),
                value_cache_scale: bf16_to_f32(source.value_cache_scale_bf16),
            }))
        }
    }
}

fn load_mlp<S: ResidentWeightSink>(
    arena: &mut S,
    stream: &CudaStream,
    layer: usize,
    weights: MlpWeights,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
    preparation: &mut ResidentPreparationStats,
) -> EngineResult<MlpScalars> {
    match weights {
        MlpWeights::Nvfp4(weights) => {
            let gate_up = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident NVFP4 gate/up source binding",
                || Ok(Nvfp4GateUpBindings::bind(snapshot, layer)?),
            )?;
            let gate_up = measure_preparation(
                &mut preparation.nvfp4_materialize_ns,
                "resident NVFP4 gate/up materialization",
                || Ok(gate_up.materialize()?),
            )?;
            let down = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident NVFP4 down source binding",
                || Ok(Nvfp4DownBindings::bind(snapshot, layer)?),
            )?;
            let down = measure_preparation(
                &mut preparation.nvfp4_materialize_ns,
                "resident NVFP4 down materialization",
                || Ok(down.materialize()?),
            )?;
            arena.copy_from_host(stream, weights.gate_weight_codes, gate_up.gate_weight_e2m1)?;
            arena.copy_from_host(stream, weights.up_weight_codes, gate_up.up_weight_e2m1)?;
            arena.copy_from_host(
                stream,
                weights.gate_up_weight_scales,
                &gate_up.scale_e4m3_swizzled,
            )?;
            arena.copy_from_host(stream, weights.down_weight_codes, down.weight_e2m1)?;
            arena.copy_from_host(
                stream,
                weights.down_weight_scales,
                &down.scale_e4m3_swizzled,
            )?;
            Ok(MlpScalars::Nvfp4(Nvfp4Scalars {
                gate_up_input: gate_up.input_scale_divisor,
                gate_up_weight: gate_up.weight_scale_divisor,
                down_input: down.input_scale_divisor,
                down_weight: down.weight_scale_divisor,
            }))
        }
        MlpWeights::DenseFp8(weights) => {
            let gate_up = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident dense-FP8 gate/up source binding",
                || Ok(DenseFp8GateUpBindings::bind(snapshot, layer)?),
            )?;
            let down = measure_preparation(
                &mut preparation.source_binding_ns,
                "resident dense-FP8 down source binding",
                || Ok(DenseFp8DownBindings::bind(snapshot, layer)?),
            )?;
            arena.copy_from_host(stream, weights.gate_up_weight_codes, gate_up.weight_e4m3)?;
            arena.copy_bytes_from_host(
                stream,
                weights.gate_up_weight_scales,
                gate_up.scale_bf16,
            )?;
            arena.copy_from_host(stream, weights.down_weight_codes, down.weight.codes())?;
            arena.copy_bytes_from_host(stream, weights.down_weight_scales, down.scale.bytes())?;
            Ok(MlpScalars::DenseFp8)
        }
    }
}

fn initialize_metadata(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    stream: &CudaStream,
    layout: &ResidentModelLayout,
) -> EngineResult<()> {
    let workspace = layout.workspace;
    let mut state_rows = vec![0u32; super::MAX_ROWS];
    for (row, state_row) in state_rows.iter_mut().take(MAX_BATCH).enumerate() {
        *state_row = row as u32;
    }
    arena.copy_from_host(stream, workspace.state_rows, &state_rows)?;
    let block_tables = vec![u32::MAX; MAX_BATCH * LONG_CONTEXT_PHYSICAL_PAGES];
    kv_arena.copy_from_host(stream, layout.kv_layout.block_tables(), &block_tables)?;
    let mut table_rows = vec![0u32; super::MAX_ROWS];
    for (row, table_row) in table_rows.iter_mut().take(MAX_BATCH).enumerate() {
        *table_row = row as u32;
    }
    arena.copy_from_host(stream, workspace.table_rows, &table_rows)?;
    Ok(())
}

fn initialize_selective_nonweights(
    arena: &mut LoadingDeviceArena,
    kv_arena: &mut LoadingDeviceArena,
    plan: &ResidentUploadPlan,
    layout: &ResidentModelLayout,
    stream: &CudaStream,
) -> EngineResult<MetadataUploadStats> {
    for entry in plan
        .entries()
        .iter()
        .filter(|entry| entry.preparation() == ResidentUploadPreparation::Zero)
    {
        let end = entry
            .offset_bytes()
            .checked_add(entry.byte_len())
            .ok_or_else(|| EngineError::layout("resident zero destination overflows"))?;
        match entry.arena() {
            ResidentUploadArena::Resident => {
                arena.fill_async(stream, entry.offset_bytes()..end, 0)?;
            }
            ResidentUploadArena::Kv => {
                kv_arena.fill_async(stream, entry.offset_bytes()..end, 0)?;
            }
        }
    }

    let mut state_rows = vec![0u32; super::MAX_ROWS];
    for (row, state_row) in state_rows.iter_mut().take(MAX_BATCH).enumerate() {
        *state_row = row as u32;
    }
    upload_region(stream, arena, layout.workspace.state_rows, &state_rows)?;
    let mut table_rows = vec![0u32; super::MAX_ROWS];
    for (row, table_row) in table_rows.iter_mut().take(MAX_BATCH).enumerate() {
        *table_row = row as u32;
    }
    upload_region(stream, arena, layout.workspace.table_rows, &table_rows)?;
    let block_tables = vec![u32::MAX; MAX_BATCH * LONG_CONTEXT_PHYSICAL_PAGES];
    upload_region(
        stream,
        kv_arena,
        layout.kv_layout.block_tables(),
        &block_tables,
    )?;
    stream.synchronize().map_err(GpuError::from)?;
    Ok(MetadataUploadStats {
        bytes: plan.host_derived_bytes(),
        submissions: 3,
    })
}

struct MetadataUploadStats {
    bytes: usize,
    submissions: usize,
}

fn upload_region<T: DeviceCopy>(
    stream: &CudaStream,
    arena: &mut LoadingDeviceArena,
    region: ArenaRegion<T>,
    source: &[T],
) -> EngineResult<()> {
    let end = region
        .offset_bytes()
        .checked_add(region.byte_len())
        .ok_or_else(|| EngineError::layout("resident metadata destination overflows"))?;
    // SAFETY: `initialize_selective_nonweights` synchronizes before its local sources can be
    // released, and the loading arena checks the exact destination range.
    unsafe { arena.copy_from_host_async(stream, region.offset_bytes()..end, source)? };
    Ok(())
}

fn decode_lengths(positions: &[u32], capacity: usize) -> EngineResult<[u32; MAX_BATCH]> {
    let mut lengths = [0u32; MAX_BATCH];
    for (target, &position) in lengths.iter_mut().zip(positions) {
        if position as usize >= capacity {
            return Err(EngineError::route(format!(
                "resident cache position {position} exceeds the {capacity}-token slot capacity"
            )));
        }
        *target = position
            .checked_add(1)
            .ok_or_else(|| EngineError::route("resident cache length overflows"))?;
    }
    Ok(lengths)
}

fn select_decode_route(batch: usize, lengths: &[u32]) -> EngineResult<ResidentDecodeRoute> {
    require_batch(batch)?;
    if lengths.len() != batch {
        return Err(EngineError::route(format!(
            "resident route has {} lengths, expected {batch}",
            lengths.len()
        )));
    }
    let maximum_length =
        lengths.iter().copied().max().ok_or_else(|| {
            EngineError::route("resident route requires at least one cache length")
        })? as usize;
    if maximum_length == 0 || maximum_length > MAX_CONTEXT_TOKENS {
        return Err(EngineError::route(format!(
            "resident maximum cache length {maximum_length} is outside 1..={MAX_CONTEXT_TOKENS}"
        )));
    }
    let attention = if maximum_length <= SHORT_CONTEXT_CAPACITY {
        AttentionRoute::Short
    } else {
        let required = maximum_length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
        let (index, partitions) = LONG_CONTEXT_GQA_PARTITION_BUCKETS
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, partitions)| partitions >= required)
            .ok_or_else(|| {
                EngineError::route(format!(
                    "resident maximum cache length {maximum_length} has no partition graph"
                ))
            })?;
        AttentionRoute::Long { index, partitions }
    };

    Ok(ResidentDecodeRoute {
        batch,
        maximum_length,
        attention,
    })
}

const fn prefill_graph_routes() -> [ResidentPrefillRoute; PREFILL_GRAPH_ROUTE_COUNT] {
    [
        ResidentPrefillRoute {
            tokens: 32,
            first_position: 0,
            context_tokens: 32,
            attention: PrefillAttentionRoute::Shared,
        },
        ResidentPrefillRoute {
            tokens: 64,
            first_position: 0,
            context_tokens: 64,
            attention: PrefillAttentionRoute::Shared,
        },
        ResidentPrefillRoute {
            tokens: 128,
            first_position: 0,
            context_tokens: 128,
            attention: PrefillAttentionRoute::Shared,
        },
        ResidentPrefillRoute {
            tokens: 128,
            first_position: 1,
            context_tokens: 129,
            attention: PrefillAttentionRoute::Partitioned { partitions: 8 },
        },
        ResidentPrefillRoute {
            tokens: 128,
            first_position: PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT - 128,
            context_tokens: PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT,
            attention: PrefillAttentionRoute::Partitioned { partitions: 16 },
        },
        ResidentPrefillRoute {
            tokens: super::MAX_ROWS,
            first_position: 0,
            context_tokens: super::MAX_ROWS,
            attention: PrefillAttentionRoute::Macro { partitions: 4 },
        },
    ]
}

fn select_prefill_route(
    tokens: usize,
    first_position: usize,
    capacity: usize,
) -> EngineResult<ResidentPrefillRoute> {
    if prefill_index(tokens).is_none() {
        return Err(EngineError::route(format!(
            "resident prefill tokens {tokens} are outside 32,64,128,1024"
        )));
    }
    let context_tokens = first_position
        .checked_add(tokens)
        .ok_or_else(|| EngineError::route("resident prefill context overflows"))?;
    if context_tokens > capacity {
        return Err(EngineError::route(format!(
            "resident prefill context {context_tokens} exceeds the {capacity}-token slot capacity"
        )));
    }

    let attention = match tokens {
        32 | 64 => PrefillAttentionRoute::Shared,
        128 if context_tokens == 128 => PrefillAttentionRoute::Shared,
        128 if context_tokens < PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT => {
            // Eight K64 partitions expose 768 producer CTAs without the P16
            // route's doubled grid below the measured 32,769-token crossover.
            PrefillAttentionRoute::Partitioned { partitions: 8 }
        }
        128 => {
            // Sixteen K32 partitions keep two CTAs resident from the admitted
            // 32,769-token crossover through the 220,000-token ceiling.
            PrefillAttentionRoute::Partitioned { partitions: 16 }
        }
        super::MAX_ROWS => {
            // P4 exposes 3,072 producer CTAs and is the directly benchmarked
            // macro route at both 32,768 and 98,304 context positions.
            PrefillAttentionRoute::Macro { partitions: 4 }
        }
        _ => unreachable!(),
    };
    Ok(ResidentPrefillRoute {
        tokens,
        first_position,
        context_tokens,
        attention,
    })
}

fn prefill_graph_index(route: ResidentPrefillRoute) -> EngineResult<usize> {
    match (route.tokens, route.attention) {
        (32, PrefillAttentionRoute::Shared) => Ok(0),
        (64, PrefillAttentionRoute::Shared) => Ok(1),
        (128, PrefillAttentionRoute::Shared) => Ok(2),
        (128, PrefillAttentionRoute::Partitioned { partitions: 8 }) => Ok(3),
        (128, PrefillAttentionRoute::Partitioned { partitions: 16 }) => Ok(4),
        (super::MAX_ROWS, PrefillAttentionRoute::Macro { partitions: 4 }) => Ok(5),
        _ => Err(EngineError::route(format!(
            "resident prefill T={} context {} has no exact attention graph",
            route.tokens, route.context_tokens
        ))),
    }
}

const fn prefill_index(rows: usize) -> Option<usize> {
    match rows {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        super::MAX_ROWS => Some(3),
        _ => None,
    }
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if rows == 0 || (rows > MAX_BATCH && prefill_index(rows).is_none()) {
        return Err(EngineError::route(format!(
            "resident row count {rows} is not an admitted B=1..{MAX_BATCH} or T=32/64/128/1024 route"
        )));
    }
    Ok(())
}

fn require_target_verify_tokens(tokens: usize) -> EngineResult<()> {
    if !(1..=TARGET_VERIFY_ROUTE_COUNT).contains(&tokens) {
        return Err(EngineError::route(format!(
            "target MTP verification token count {tokens} is outside 1..={TARGET_VERIFY_ROUTE_COUNT}"
        )));
    }
    Ok(())
}

fn require_segmented_commit(
    route: ResidentMtpSegmentedVerifyRoute,
    accepted_tokens: &[usize],
) -> EngineResult<()> {
    if accepted_tokens.len() != route.batch {
        return Err(EngineError::route(format!(
            "segmented target MTP commit has {} lane counts, expected B={}",
            accepted_tokens.len(),
            route.batch
        )));
    }
    if let Some((lane, &tokens)) = accepted_tokens
        .iter()
        .enumerate()
        .find(|(_, tokens)| !(1..=route.tokens).contains(tokens))
    {
        return Err(EngineError::route(format!(
            "segmented target MTP lane {lane} commits {tokens} rows from a K={} verification",
            route.tokens
        )));
    }
    Ok(())
}

fn select_segmented_target_route(
    tokens: usize,
    batch: usize,
    maximum_lengths: &[u32],
) -> EngineResult<ResidentMtpSegmentedVerifyRoute> {
    require_target_verify_tokens(tokens)?;
    let decode = select_decode_route(batch, maximum_lengths)?;
    Ok(ResidentMtpSegmentedVerifyRoute {
        tokens,
        batch,
        maximum_length: decode.maximum_length,
        attention: decode.attention,
    })
}

fn slot_rows(slots: &[usize]) -> EngineResult<[u32; MAX_BATCH]> {
    require_batch(slots.len())?;
    let mut seen = [false; MAX_BATCH];
    let mut rows = [0u32; MAX_BATCH];
    for (row, &slot) in rows.iter_mut().zip(slots) {
        require_slot(slot)?;
        if std::mem::replace(&mut seen[slot], true) {
            return Err(EngineError::route(format!(
                "resident physical slot {slot} appears more than once"
            )));
        }
        *row = slot as u32;
    }
    Ok(rows)
}

fn require_slot(slot: usize) -> EngineResult<()> {
    if slot >= MAX_BATCH {
        return Err(EngineError::route(format!(
            "resident physical slot {slot} is outside 0..{MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_gdn_snapshot_buffers(
    program: &ResidentModelProgram,
    history: &PinnedHostBuffer<u16>,
    state: &PinnedHostBuffer<f32>,
) -> EngineResult<()> {
    let expected_history = product(
        "resident GDN snapshot history values",
        MAX_BATCH,
        program.gdn_slot_history_values(),
    )?;
    let expected_state = product(
        "resident GDN snapshot state values",
        MAX_BATCH,
        program.gdn_slot_state_values(),
    )?;
    if history.len() != expected_history || state.len() != expected_state {
        return Err(EngineError::layout(format!(
            "resident GDN snapshot buffers have {}/{} history/state values, expected {expected_history}/{expected_state}",
            history.len(),
            state.len()
        )));
    }
    Ok(())
}

fn fill_slot<T: DeviceCopy>(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: ArenaRegion<T>,
    slot: usize,
) -> EngineResult<()> {
    if !region.len().is_multiple_of(MAX_BATCH) {
        return Err(EngineError::layout(format!(
            "resident persistent region of {} values is not divisible by {MAX_BATCH} slots",
            region.len()
        )));
    }
    let width = region.len() / MAX_BATCH;
    let start = product("resident persistent slot offset", slot, width)?;
    arena.fill_slice(stream, region, start, width, 0)?;
    Ok(())
}

fn clear_physical_cache_page(
    arena: &DeviceArena,
    stream: &CudaStream,
    layout: &ResidentModelLayout,
    physical_page: usize,
) -> EngineResult<()> {
    if physical_page >= LONG_CONTEXT_PHYSICAL_PAGES {
        return Err(EngineError::layout(format!(
            "resident physical cache page {physical_page} exceeds 0..{LONG_CONTEXT_PHYSICAL_PAGES}"
        )));
    }
    let start = cache_values(physical_page)?;
    let len = cache_values(1)?;
    for cache in layout.kv_layout.layers() {
        arena.fill_slice(stream, cache.key.data, start, len, 0)?;
        arena.fill_slice(stream, cache.value.data, start, len, 0)?;
    }
    Ok(())
}

fn cache_values(pages: usize) -> EngineResult<usize> {
    product(
        "resident represented cache values",
        pages,
        product(
            "resident represented cache page values",
            product(
                "resident represented cache head tokens",
                Qwen38_27B::NUM_KV_HEADS,
                ATTENTION_PAGE_SIZE,
            )?,
            Qwen38_27B::HEAD_DIM,
        )?,
    )
}

fn copy_embedding_row(source: &[u8], token: usize, destination: &mut [u16]) -> EngineResult<()> {
    if destination.len() != Qwen38_27B::HIDDEN {
        return Err(EngineError::layout(format!(
            "embedding destination has {} words, expected {}",
            destination.len(),
            Qwen38_27B::HIDDEN
        )));
    }
    let word_begin = product("resident embedding row offset", token, Qwen38_27B::HIDDEN)?;
    let byte_begin = product("resident embedding byte offset", word_begin, 2)?;
    let byte_len = product("resident embedding row bytes", Qwen38_27B::HIDDEN, 2)?;
    let byte_end = byte_begin
        .checked_add(byte_len)
        .ok_or_else(|| EngineError::layout("resident embedding byte range overflows"))?;
    let row = source.get(byte_begin..byte_end).ok_or_else(|| {
        EngineError::layout(format!("embedding row {token} is outside its source view"))
    })?;
    for (target, bytes) in destination.iter_mut().zip(row.as_chunks::<2>().0) {
        *target = u16::from_le_bytes(*bytes);
    }
    Ok(())
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "resident-model batch {batch} is outside the exact range 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn elapsed_ns(phase: &str, started: Instant) -> EngineResult<u64> {
    u64::try_from(started.elapsed().as_nanos())
        .map_err(|_| EngineError::layout(format!("{phase} duration exceeds u64 nanoseconds")))
}

fn measure_preparation<T>(
    counter: &mut u64,
    phase: &str,
    operation: impl FnOnce() -> EngineResult<T>,
) -> EngineResult<T> {
    let started = Instant::now();
    let result = operation();
    *counter = counter
        .checked_add(elapsed_ns(phase, started)?)
        .ok_or_else(|| EngineError::layout(format!("{phase} timing overflows")))?;
    result
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{
        LONG_CONTEXT_ROUTE_COUNT, PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT,
        PREFILL_GRAPH_ROUTE_COUNT, PRODUCTION_LOAD_MODE, ResidentLoadMode,
        TARGET_VERIFY_ROUTE_COUNT, bf16_to_f32, decode_lengths, prefill_graph_index,
        prefill_graph_routes, prefill_index, require_batch, require_rows, require_segmented_commit,
        require_target_verify_tokens, select_decode_route, select_prefill_route,
        select_segmented_target_route, slot_rows,
    };
    use crate::EngineErrorCode;

    #[test]
    fn production_loader_is_the_prefaulted_selective_route() {
        assert_eq!(PRODUCTION_LOAD_MODE, ResidentLoadMode::Selective);
    }

    #[test]
    fn exact_batch_inventory_rejects_every_neighbor() {
        for batch in 1..=8 {
            require_batch(batch).unwrap();
        }
        for batch in [0, 9, 16, usize::MAX] {
            let error = require_batch(batch).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn target_mtp_inventory_covers_every_k_and_attention_band() {
        assert_eq!(
            TARGET_VERIFY_ROUTE_COUNT
                * (LONG_CONTEXT_ROUTE_COUNT
                    + 2
                    + super::TARGET_SEGMENTED_BATCH_ROUTES * (LONG_CONTEXT_ROUTE_COUNT + 1)),
            228
        );
        assert_eq!(4 * (LONG_CONTEXT_ROUTE_COUNT + 2 + 2 * 7), 88);
        for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
            require_target_verify_tokens(tokens).unwrap();
            for maximum_length in [
                1u32, 192, 193, 1_024, 1_025, 4_096, 4_097, 16_384, 16_385, 65_536, 65_537,
                131_072, 131_073, 220_000,
            ] {
                let mut lengths = vec![1; tokens];
                lengths[tokens - 1] = maximum_length;
                let route = select_decode_route(tokens, &lengths).unwrap();
                assert_eq!(route.batch(), tokens);
                assert_eq!(route.maximum_length(), maximum_length as usize);
            }
        }
        for tokens in [0, TARGET_VERIFY_ROUTE_COUNT + 1, usize::MAX] {
            assert_eq!(
                require_target_verify_tokens(tokens).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
    }

    #[test]
    fn segmented_target_inventory_covers_every_batch_k_and_commit_prefix() {
        for batch in 1..=8 {
            for tokens in 1..=TARGET_VERIFY_ROUTE_COUNT {
                let mut lengths = vec![1; batch];
                lengths[batch - 1] = 220_000;
                let route = select_segmented_target_route(tokens, batch, &lengths).unwrap();
                assert_eq!(route.batch(), batch);
                assert_eq!(route.tokens(), tokens);
                assert_eq!(route.rows(), batch * tokens);
                assert_eq!(route.maximum_length(), 220_000);
                for accepted in 1..=tokens {
                    require_segmented_commit(route, &vec![accepted; batch]).unwrap();
                }
                assert!(require_segmented_commit(route, &vec![1; batch - 1]).is_err());
                assert!(require_segmented_commit(route, &vec![tokens + 1; batch]).is_err());
            }
        }
        for batch in [0, 9] {
            assert!(select_segmented_target_route(4, batch, &vec![1; batch]).is_err());
        }
        for tokens in [0, 5] {
            assert!(select_segmented_target_route(tokens, 1, &[1]).is_err());
        }
    }

    #[test]
    fn exact_row_inventory_adds_only_the_four_prefill_tiles() {
        for rows in 1..=8 {
            require_rows(rows).unwrap();
        }
        for (index, rows) in [32, 64, 128, 1_024].into_iter().enumerate() {
            require_rows(rows).unwrap();
            assert_eq!(prefill_index(rows), Some(index));
        }
        for rows in [0, 9, 31, 33, 63, 65, 127, 129, 1_023, 1_025, usize::MAX] {
            assert_eq!(
                require_rows(rows).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
            assert_eq!(prefill_index(rows), None);
        }
    }

    #[test]
    fn prefill_graph_inventory_covers_every_admitted_context_band() {
        let captured = prefill_graph_routes();
        for (index, route) in captured.into_iter().enumerate() {
            assert_eq!(prefill_graph_index(route).unwrap(), index);
        }
        assert_eq!(captured.len(), PREFILL_GRAPH_ROUTE_COUNT);

        for (tokens, first_position, partitions) in [
            (32, 0, None),
            (32, 219_968, None),
            (64, 0, None),
            (64, 219_936, None),
            (128, 0, None),
            (128, 1, Some(8)),
            (
                128,
                PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT - 129,
                Some(8),
            ),
            (
                128,
                PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT - 128,
                Some(16),
            ),
            (128, 219_872, Some(16)),
            (1_024, 0, Some(4)),
            (1_024, 218_976, Some(4)),
        ] {
            let route = select_prefill_route(tokens, first_position, 220_000).unwrap();
            assert_eq!(route.tokens(), tokens);
            assert_eq!(route.first_position(), first_position);
            assert_eq!(route.context_tokens(), first_position + tokens);
            assert_eq!(route.partition_capacity(), partitions);
            prefill_graph_index(route).unwrap();
        }

        for (tokens, first_position) in [
            (31, 0),
            (33, 0),
            (127, 0),
            (129, 0),
            (1_023, 0),
            (1_025, 0),
            (32, 219_969),
            (1_024, 218_977),
            (32, usize::MAX),
        ] {
            assert_eq!(
                select_prefill_route(tokens, first_position, 220_000)
                    .unwrap_err()
                    .code(),
                Some(EngineErrorCode::Route)
            );
        }
    }

    #[test]
    fn decode_lengths_enforce_the_current_cache_capacity() {
        assert_eq!(
            &decode_lengths(&[0, 63, 219_999], 220_000).unwrap()[..3],
            [1, 64, 220_000]
        );
        assert_eq!(
            decode_lengths(&[220_000], 220_000).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
    }

    #[test]
    fn decode_graph_inventory_covers_every_batch_and_partition_boundary() {
        let cases = [
            (1u32, None),
            (192, None),
            (193, Some(4)),
            (1_024, Some(4)),
            (1_025, Some(16)),
            (4_096, Some(16)),
            (4_097, Some(64)),
            (16_384, Some(64)),
            (16_385, Some(256)),
            (65_536, Some(256)),
            (65_537, Some(512)),
            (131_072, Some(512)),
            (131_073, Some(860)),
            (220_000, Some(860)),
        ];
        for batch in 1..=8 {
            for (maximum_length, partitions) in cases {
                let mut lengths = vec![1; batch];
                lengths[batch - 1] = maximum_length;
                let route = select_decode_route(batch, &lengths).unwrap();
                assert_eq!(route.batch(), batch);
                assert_eq!(route.maximum_length(), maximum_length as usize);
                assert_eq!(route.partition_capacity(), partitions);
                assert_eq!(route.is_long_context(), partitions.is_some());
            }
        }
        for (batch, lengths) in [(1, vec![]), (2, vec![1]), (1, vec![220_001])] {
            assert_eq!(
                select_decode_route(batch, &lengths).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
    }

    #[test]
    fn slot_routes_cover_every_exact_batch_and_reject_aliasing() {
        let inventory = [7, 0, 6, 1, 5, 2, 4, 3];
        for batch in 1..=8 {
            let rows = slot_rows(&inventory[..batch]).unwrap();
            assert!(
                rows[..batch]
                    .iter()
                    .zip(&inventory[..batch])
                    .all(|(&row, &slot)| row as usize == slot)
            );
        }
        for slots in [
            &[][..],
            &[0, 0][..],
            &[8][..],
            &[0, 1, 2, 3, 4, 5, 6, 7, 0][..],
        ] {
            assert_eq!(
                slot_rows(slots).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
    }

    #[test]
    fn bf16_word_widens_exactly() {
        assert_eq!(bf16_to_f32(0x3f80).to_bits(), 1.0f32.to_bits());
    }
}

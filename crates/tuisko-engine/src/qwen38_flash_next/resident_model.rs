//! Qwen3.8 Flash-Next resident program over one streaming expert cache.
//!
//! Router-dependent expert uploads force 49 graph segments. Each segment retains all 12 exact
//! routes, and each layer reads its 512-entry view of the global streaming slot table.

use crate::common::math::product;
use crate::common::progress::ResidentLoadProgress;
use crate::common::streaming::{
    StreamingMappedPrimary, StreamingPrimarySource, StreamingRound, StreamingWeightPool,
};
use crate::qwen38_flash_next::engram_stager::gather_qwen38_flash_next_engram_window;
use crate::qwen38_flash_next::layer_route::{
    QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, QWEN38_FLASH_NEXT_PREFILL_ROWS,
    Qwen38FlashNextRowRoute, qwen38_flash_next_row_route,
    require_qwen38_flash_next_dense_qsa_visible,
};
use crate::qwen38_flash_next::layer_upload::{
    bf16_words, expert_slot_image, upload_hyper_connection,
};
use crate::qwen38_flash_next::resident_model_layout::{
    QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT, QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES,
    QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES, QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS,
    QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, Qwen38FlashNextBlockWeightRegions,
    Qwen38FlashNextResidentEndpoint, Qwen38FlashNextResidentLayout,
};
use crate::qwen38_flash_next::slot_lifecycle::{
    QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS, QWEN38_FLASH_NEXT_UNMAPPED_PAGE, Qwen38FlashNextSlotChange,
    Qwen38FlashNextSlotPool, Qwen38FlashNextSlotRelease, Qwen38FlashNextSlotState,
};
use crate::qwen38_flash_next::text_generation::Qwen38FlashNextGenerationTelemetry;
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{
    Qwen38FlashNextAttentionGateOp, Qwen38FlashNextAttentionQkPrepareOp,
    Qwen38FlashNextBf16LmHeadOp, Qwen38FlashNextBlockOutputProjectionOp, Qwen38FlashNextEngramOp,
    Qwen38FlashNextEngramSources, Qwen38FlashNextEngramWorkspace, Qwen38FlashNextExpertDispatch,
    Qwen38FlashNextGdnInputProjectionOp, Qwen38FlashNextGdnPrepareOp,
    Qwen38FlashNextGdnRecurrenceOp, Qwen38FlashNextHyperConnectionOp, Qwen38FlashNextMoeExpertsOp,
    Qwen38FlashNextMoeRouterOp, Qwen38FlashNextQsaQkvProjectionOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextEngramBindings,
    Qwen38FlashNextEngramCarry, Qwen38FlashNextGdnBindings, Qwen38FlashNextLayerHyperConnections,
    Qwen38FlashNextMoeBindings, Qwen38FlashNextSparseAttentionBindings,
    Qwen38FlashNextTextEndpointBindings,
};

type A = Qwen38FlashNext;

/// Represented E4M3 key-plane scale this target's cache is qualified at.
const KEY_CACHE_SCALE: f32 = 0.031_25;

/// Represented E4M3 value-plane scale this target's cache is qualified at.
const VALUE_CACHE_SCALE: f32 = 0.062_5;

/// Rotary elements one token carries.
const ROTARY_ELEMENTS: usize = 32;

/// Admitted routes per segment: decode `B=1..8` then prefill `T=32/64/128/1024`.
const ROUTES_PER_SEGMENT: usize = MAX_BATCH + QWEN38_FLASH_NEXT_PREFILL_ROWS.len();

const _: () = assert!(ROUTES_PER_SEGMENT == 12);

/// Flat index with decode at `0..8` and prefill at `8..12`.
const fn segment_route_index(route: Qwen38FlashNextRowRoute) -> usize {
    match route {
        Qwen38FlashNextRowRoute::Decode(rows) => rows - 1,
        Qwen38FlashNextRowRoute::Prefill(_) => MAX_BATCH + route.graph_index(),
    }
}

/// Borrowed checkpoint extents for the mapped-primary posture.
struct Qwen38FlashNextExpertSource {
    extents: Vec<(*const u8, usize)>,
    // Field order is drop order: the extents point into this mapping.
    _snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
}

// SAFETY: `_snapshot` keeps every read-only extent alive and no method writes through it.
unsafe impl Send for Qwen38FlashNextExpertSource {}
// SAFETY: as above; `primary_extent` takes `&self` and only reads.
unsafe impl Sync for Qwen38FlashNextExpertSource {}

impl StreamingMappedPrimary for Qwen38FlashNextExpertSource {
    fn primary_extent(&self, item: usize) -> EngineResult<&[u8]> {
        let &(pointer, len) = self.extents.get(item).ok_or_else(|| {
            EngineError::layout(format!(
                "Qwen3.8 Flash-Next expert item {item} is outside 0..{QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT}"
            ))
        })?;

        // SAFETY: `pointer` and `len` were validated as one contiguous run inside the retained
        // mapping when this source was built, and the mapping outlives `&self`.
        Ok(unsafe { std::slice::from_raw_parts(pointer, len) })
    }
}

/// Per-layer streaming evidence one forward step produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextLayerStreamTelemetry {
    layer: usize,
    requests: usize,
    hits: usize,
    misses: usize,
    uploaded_bytes: usize,
    stalled: bool,
}

impl Qwen38FlashNextLayerStreamTelemetry {
    /// Decoder layer this round served.
    pub const fn layer(self) -> usize {
        self.layer
    }

    /// Expert selections the round named, duplicates included.
    pub const fn requests(self) -> usize {
        self.requests
    }

    /// Distinct items the round found already resident.
    pub const fn hits(self) -> usize {
        self.hits
    }

    /// Distinct items the round had to upload.
    pub const fn misses(self) -> usize {
        self.misses
    }

    /// Host-to-device bytes this round enqueued.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }

    /// Whether the round took the stalling `require` route.
    pub const fn stalled(self) -> bool {
        self.stalled
    }

    /// Fraction of this round's distinct items that were already resident.
    pub fn hit_rate(self) -> f64 {
        let distinct = self.hits + self.misses;
        if distinct == 0 {
            return 0.0;
        }

        self.hits as f64 / distinct as f64
    }
}

/// Counted telemetry for one whole-model step.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextStepTelemetry {
    rows: usize,
    layers: Vec<Qwen38FlashNextLayerStreamTelemetry>,
    embedding_h2d_bytes: usize,
    engram_h2d_bytes: usize,
    engram_rows: usize,
    kv_append_bytes: usize,
    forward: Duration,
    segment_replays: usize,
    expert_readbacks: usize,
}

impl Qwen38FlashNextStepTelemetry {
    /// Rows this step carried: `B` for a decode step, `T` for a prefill tile.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// One entry per decoder layer, in stack order.
    pub fn layers(&self) -> &[Qwen38FlashNextLayerStreamTelemetry] {
        &self.layers
    }

    /// Expert selections across the whole stack: 48 layers of top-10 per row.
    pub fn expert_requests(&self) -> usize {
        self.layers.iter().map(|layer| layer.requests).sum()
    }

    /// Host-to-device expert bytes this step streamed.
    pub fn expert_h2d_bytes(&self) -> usize {
        self.layers.iter().map(|layer| layer.uploaded_bytes).sum()
    }

    /// Bytes every routed selection would read out of VRAM, hit or miss.
    pub fn expert_bytes_routed(&self) -> usize {
        self.expert_requests() * tuisko_kernels_sm120::QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES
    }

    /// Whole-stack expert hit rate over distinct per-round items.
    pub fn expert_hit_rate(&self) -> f64 {
        let hits = self.layers.iter().map(|layer| layer.hits).sum::<usize>();
        let misses = self.layers.iter().map(|layer| layer.misses).sum::<usize>();
        if hits + misses == 0 {
            return 0.0;
        }

        hits as f64 / (hits + misses) as f64
    }

    /// Streaming rounds this step issued: one per MoE layer.
    pub fn streaming_rounds(&self) -> usize {
        self.layers.len()
    }

    /// Token-embedding bytes the host stager uploaded.
    pub const fn embedding_h2d_bytes(&self) -> usize {
        self.embedding_h2d_bytes
    }

    /// Engram FP8 bytes the host gather uploaded.
    pub const fn engram_h2d_bytes(&self) -> usize {
        self.engram_h2d_bytes
    }

    /// Engram rows the host hash addressed.
    pub const fn engram_rows(&self) -> usize {
        self.engram_rows
    }

    /// Bytes this step appended to the paged K/V planes.
    pub const fn kv_append_bytes(&self) -> usize {
        self.kv_append_bytes
    }

    /// Wall time from the first staged byte to the readable logits.
    pub const fn forward(&self) -> Duration {
        self.forward
    }

    /// Captured segments this step replayed.
    pub const fn segment_replays(&self) -> usize {
        self.segment_replays
    }

    /// Synchronous device-to-host reads this step performed, one per round.
    pub const fn expert_readbacks(&self) -> usize {
        self.expert_readbacks
    }

    /// Milliseconds per token at this step's row count.
    pub fn forward_ms_per_token(&self) -> f64 {
        self.forward.as_secs_f64() * 1_000.0 / self.rows as f64
    }

    /// Tokens per second this step sustained.
    pub fn tokens_per_second(&self) -> f64 {
        self.rows as f64 / self.forward.as_secs_f64()
    }
}

/// A complete restore point for one slot's sequence state.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextSlotSnapshot {
    owner: Arc<()>,
    slot: usize,
    sequence: u64,
    tokens: usize,
    carry: Qwen38FlashNextEngramCarry,
    gdn_history: Vec<u16>,
    gdn_state: Vec<f32>,
    ple_conv_state: Vec<u16>,
}

impl Qwen38FlashNextSlotSnapshot {
    /// Slot this snapshot belongs to.
    pub const fn slot(&self) -> usize {
        self.slot
    }

    /// Tokens covered by the snapshot.
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Recurrent payload bytes held by the snapshot.
    pub fn byte_len(&self) -> usize {
        size_of::<Qwen38FlashNextEngramCarry>()
            + self.gdn_history.len() * size_of::<u16>()
            + self.gdn_state.len() * size_of::<f32>()
            + self.ple_conv_state.len() * size_of::<u16>()
    }
}

/// Resident-program construction measurements.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextResidentLoadStats {
    weight_upload: Duration,
    expert_stage: Duration,
    graph_capture: Duration,
    host_pin: Duration,
    executables: usize,
    definitions: usize,
    staged_items: usize,
    staged_bytes: usize,
}

impl Qwen38FlashNextResidentLoadStats {
    /// Wall time for resident uploads and expert staging together.
    pub const fn weight_upload(self) -> Duration {
        self.weight_upload
    }

    /// Wall time staging every expert's pinned extents took.
    pub const fn expert_stage(self) -> Duration {
        self.expert_stage
    }

    /// Wall time for all graph captures.
    pub const fn graph_capture(self) -> Duration {
        self.graph_capture
    }

    /// Wall time page-locking the streaming pool's host classes took.
    pub const fn host_pin(self) -> Duration {
        self.host_pin
    }

    /// Captured executables the program retains: segments times routes.
    pub const fn executables(self) -> usize {
        self.executables
    }

    /// Distinct graph definitions captured, which is the same number here.
    pub const fn definitions(self) -> usize {
        self.definitions
    }

    /// Items staged into the pinned pool.
    pub const fn staged_items(self) -> usize {
        self.staged_items
    }

    /// Host bytes `stage_item` copied, posture-dependent.
    pub const fn staged_bytes(self) -> usize {
        self.staged_bytes
    }

    /// Mean capture cost per executable.
    pub fn capture_per_executable(self) -> Duration {
        self.graph_capture / self.executables.max(1) as u32
    }
}

/// Every device address one layer's launches read or write.
#[derive(Clone, Copy)]
struct LayerPointers {
    attention_hc: BracketPointers,
    mlp_hc: BracketPointers,

    block: BlockPointers,

    router_weight: *const u16,
    expert_weight_scales_2: *const f32,
    shared_gate_weight: *const u16,
    shared_up_weight: *const u16,
    shared_down_weight: *const u16,
    shared_gate_logit_weight: *const u16,

    /// This layer's 512-entry view into the global indirection table.
    slot_table: *const u32,

    ple: Option<PlePointers>,
    gdn_history: *mut u16,
    gdn_state: *mut f32,
}

#[derive(Clone, Copy)]
struct BracketPointers {
    norm: *const u16,
    down: *const u16,
    up: *const u16,
    inject: *const u16,
}

#[derive(Clone, Copy)]
enum BlockPointers {
    Gdn {
        input_weight: *const u16,
        control_weight: *const u16,
        convolution_weight: *const u16,
        a_log: *const u16,
        dt_bias: *const u16,
        norm: *const u16,
        output_weight: *const u16,
    },
    Qsa {
        qkv_weight: *const u16,
        output_weight: *const u16,
        query_norm: *const u16,
        key_norm: *const u16,
        key_pages: *mut u8,
        value_pages: *mut u8,
    },
}

#[derive(Clone, Copy)]
struct PlePointers {
    key_proj: *const u16,
    value_proj: *const u16,
    norm_key: *const u16,
    norm_query: *const u16,
    norm_conv: *const u16,
    convolution: *const u16,
    conv_state: *mut u16,
}

/// The shared activation planes every layer addresses in turn.
#[derive(Clone, Copy)]
struct WorkspacePointers {
    residual_a: *mut u16,
    residual_b: *mut u16,

    hc_normalized: *mut u16,
    hc_low_rank: *mut u16,
    hc_mixed: *mut u16,
    hc_write_gate: *mut u16,

    gdn_projected: *mut u16,
    gdn_convolved: *mut u16,
    gdn_log_decay: *mut f32,
    gdn_beta: *mut f32,
    gdn_recurrent_plane: *mut f32,
    gdn_recurrent_output: *mut u16,

    qkv: *mut u16,
    query: *mut f32,
    attention: *mut f32,
    attention_gated: *mut u16,

    router_logits: *mut u16,
    expert_indices: *mut u16,
    routing_weights: *mut u16,
    routed_intermediate: *mut u16,
    routed_output: *mut u16,
    shared_intermediate: *mut u16,
    shared_output: *mut u16,
    shared_gate_logit: *mut u16,

    block_output: *mut u16,

    state_rows: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    rope_cos: *const f32,
    rope_sin: *const f32,

    block_tables: *const u32,

    ple_codes: *const u8,
    ple_injected: *mut u16,
    ple_embedding: *mut u16,
    ple_key: *mut u16,
    ple_key_normed: *mut u16,
    ple_query_normed: *mut u16,
    ple_value: *mut u16,
    ple_gated: *mut u16,
    ple_gated_normed: *mut u16,
    ple_delta: *mut u16,

    /// The whole sealed slot pool, addressed only through a layer's table view.
    slot_pool: *const u8,
}

/// The endpoint's weights and the two planes the tail segment publishes.
#[derive(Clone, Copy)]
struct EndpointPointers {
    mixer_norm: *const u16,
    mixer_down: *const u16,
    mixer_up: *const u16,
    lm_head: *const u16,
    mixer_normalized: *mut u16,
    mixer_low_rank: *mut u16,
    mixer_mixed: *mut u16,
    logits: *mut u16,
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    hyper: &'a Qwen38FlashNextHyperConnectionOp,
    gdn_input: &'a Qwen38FlashNextGdnInputProjectionOp,
    gdn_prepare: &'a Qwen38FlashNextGdnPrepareOp,
    gdn_recurrence: &'a Qwen38FlashNextGdnRecurrenceOp,
    qsa_qkv: &'a Qwen38FlashNextQsaQkvProjectionOp,
    qsa_prepare: &'a Qwen38FlashNextAttentionQkPrepareOp,
    qsa_attention: &'a tuisko_kernels_sm120::Qwen38FlashNextPagedGqaOp,
    qsa_gate: &'a Qwen38FlashNextAttentionGateOp,
    block_output: &'a Qwen38FlashNextBlockOutputProjectionOp,
    router: &'a Qwen38FlashNextMoeRouterOp,
    experts: &'a Qwen38FlashNextMoeExpertsOp,
    engram: &'a Qwen38FlashNextEngramOp,
    lm_head: &'a Qwen38FlashNextBf16LmHeadOp,
}

/// The whole Qwen3.8 Flash-Next model, resident over one shared expert cache.
pub struct Qwen38FlashNextResidentModel {
    // Drop graphs before the arenas, the pool, and the loaded modules they retain.
    segments: Vec<CudaGraph>,
    arena: DeviceArena,
    kv_arena: DeviceArena,
    pool: StreamingWeightPool,

    _hyper: Qwen38FlashNextHyperConnectionOp,
    _gdn_input: Qwen38FlashNextGdnInputProjectionOp,
    _gdn_prepare: Qwen38FlashNextGdnPrepareOp,
    _gdn_recurrence: Qwen38FlashNextGdnRecurrenceOp,
    _qsa_qkv: Qwen38FlashNextQsaQkvProjectionOp,
    _qsa_prepare: Qwen38FlashNextAttentionQkPrepareOp,
    _qsa_attention: tuisko_kernels_sm120::Qwen38FlashNextPagedGqaOp,
    _qsa_gate: Qwen38FlashNextAttentionGateOp,
    _block_output: Qwen38FlashNextBlockOutputProjectionOp,
    _router: Qwen38FlashNextMoeRouterOp,
    _experts: Qwen38FlashNextMoeExpertsOp,
    _engram: Qwen38FlashNextEngramOp,
    _lm_head: Qwen38FlashNextBf16LmHeadOp,

    embedding_stager: PinnedHostBuffer<u16>,
    engram_stager: PinnedHostBuffer<u8>,
    logit_bank: PinnedHostBuffer<u16>,
    expert_readback: Vec<u16>,
    engram_rows: Vec<i64>,
    carries: [Qwen38FlashNextEngramCarry; MAX_BATCH],
    round_items: Vec<u32>,
    slots: Qwen38FlashNextSlotPool,
    snapshot_owner: Arc<()>,
    generation: Qwen38FlashNextGenerationTelemetry,

    layers: Vec<LayerPointers>,
    workspace: WorkspacePointers,
    endpoint: EndpointPointers,

    snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    context: Arc<CudaContext>,
    layout: Qwen38FlashNextResidentLayout,
    base_address: u64,
    kv_base_address: u64,
    table_scale_bits: u16,
    load_stats: Qwen38FlashNextResidentLoadStats,
}

impl Qwen38FlashNextResidentModel {
    /// Loads every layer, stages every expert, and captures all 588 segment executables.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_progress(context, snapshot, None)
    }

    /// Loads the resident program while reporting source-backed upload progress.
    pub fn from_snapshot_with_progress(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let layout = Qwen38FlashNextResidentLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;

        let arena = DeviceArena::zeroed(&stream, layout.resident_builder())?;
        let kv_arena = DeviceArena::zeroed(&stream, layout.kv_builder())?;
        stream.synchronize().map_err(GpuError::from)?;

        // The pool owns its own arena and its own transfer stream; the mapped-primary posture
        // binds the checkpoint's borrowed extents once, at construction, and never again.
        let pool = match layout.streaming().primary_source() {
            StreamingPrimarySource::Mapped => {
                let source = Qwen38FlashNextExpertSource::bind(&snapshot)?;
                StreamingWeightPool::new_with_mapped_primary(
                    context,
                    *layout.streaming(),
                    Box::new(source),
                )?
            }
            StreamingPrimarySource::Pinned => {
                StreamingWeightPool::new(context, *layout.streaming())?
            }
        };

        let hyper = Qwen38FlashNextHyperConnectionOp::new(context)?;
        let gdn_input = Qwen38FlashNextGdnInputProjectionOp::new(context)?;
        let gdn_prepare = Qwen38FlashNextGdnPrepareOp::new(context)?;
        let gdn_recurrence = Qwen38FlashNextGdnRecurrenceOp::new(context)?;
        let qsa_qkv = Qwen38FlashNextQsaQkvProjectionOp::new(context)?;
        let qsa_prepare = Qwen38FlashNextAttentionQkPrepareOp::new(context)?;
        let qsa_attention = tuisko_kernels_sm120::Qwen38FlashNextPagedGqaOp::new(context)?;
        let qsa_gate = Qwen38FlashNextAttentionGateOp::new(context)?;
        let block_output = Qwen38FlashNextBlockOutputProjectionOp::new(context)?;
        let router = Qwen38FlashNextMoeRouterOp::new(context)?;
        let experts = Qwen38FlashNextMoeExpertsOp::new(context)?;
        let engram = Qwen38FlashNextEngramOp::new(context)?;
        let lm_head = Qwen38FlashNextBf16LmHeadOp::new(context)?;

        let mut model = Self::assemble(
            context,
            snapshot,
            layout,
            arena,
            kv_arena,
            pool,
            hyper,
            gdn_input,
            gdn_prepare,
            gdn_recurrence,
            qsa_qkv,
            qsa_prepare,
            qsa_attention,
            qsa_gate,
            block_output,
            router,
            experts,
            engram,
            lm_head,
        )?;
        model.upload(&stream, progress)?;
        model.capture(&stream)?;

        Ok(model)
    }

    /// Cumulative construction measurements.
    pub const fn load_stats(&self) -> Qwen38FlashNextResidentLoadStats {
        self.load_stats
    }

    /// The four-arena plan this program was built against.
    pub const fn layout(&self) -> &Qwen38FlashNextResidentLayout {
        &self.layout
    }

    /// The shared expert cache.
    pub const fn pool(&self) -> &StreamingWeightPool {
        &self.pool
    }

    /// CUDA context every arena, stream and module belongs to.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of the resident arena.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable base address of the paged cache arena.
    pub const fn kv_base_address(&self) -> u64 {
        self.kv_base_address
    }

    /// Context depth one decode slot reaches, as the KV solver funded it.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_tokens_per_slot()
    }

    /// Longest sequence admitted by both the funded cache and dense QSA.
    pub const fn generation_capacity(&self) -> usize {
        let funded = self.layout.context_tokens_per_slot();
        if funded < QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING {
            funded
        } else {
            QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
        }
    }

    /// Whole-request streaming and timing evidence accumulated since the last reset.
    pub const fn generation_telemetry(&self) -> Qwen38FlashNextGenerationTelemetry {
        self.generation
    }

    /// Starts a fresh request's accounting.
    pub fn reset_generation_telemetry(&mut self) {
        self.generation = Qwen38FlashNextGenerationTelemetry::default();
    }

    pub(crate) fn observe_prime_round(&mut self, step: &Qwen38FlashNextStepTelemetry, tile: bool) {
        self.generation.observe_prime(step, tile);
    }

    pub(crate) fn observe_decode_round(&mut self, step: &Qwen38FlashNextStepTelemetry) {
        self.generation.observe_decode(step);
    }

    /// Captured executables this program retains.
    pub fn executables(&self) -> usize {
        self.segments.len()
    }

    /// Returns the unrecoverable streaming-pool failure, if any.
    pub fn poisoned(&self) -> Option<&str> {
        self.pool.poisoned()
    }
}

impl Qwen38FlashNextExpertSource {
    /// Resolves each contiguous `down/gate/up` run, refusing a changed source layout.
    fn bind(snapshot: &Arc<CheckpointSnapshot<Qwen38FlashNext>>) -> EngineResult<Self> {
        let mut extents = Vec::with_capacity(QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT);
        for layer in 0..<A as Arch>::LAYERS {
            let moe = Qwen38FlashNextMoeBindings::bind(snapshot, layer)?.materialize()?;
            for expert in &moe.experts.experts {
                let down = expert.down_weight_e2m1;
                let gate = expert.gate_weight_e2m1;
                let up = expert.up_weight_e2m1;
                let base = down.as_ptr();
                let contiguous = std::ptr::eq(
                    // SAFETY: pointer arithmetic for a comparison only; never dereferenced.
                    unsafe { base.add(down.len()) },
                    gate.as_ptr(),
                ) && std::ptr::eq(
                    // SAFETY: as above.
                    unsafe { gate.as_ptr().add(gate.len()) },
                    up.as_ptr(),
                );
                let len = down.len() + gate.len() + up.len();
                if !contiguous || len != QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES {
                    return Err(EngineError::layout(format!(
                        "Qwen3.8 Flash-Next layer {layer} expert {} is not one contiguous \
                         {QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES} B down-gate-up run in the \
                         mapping, so the mapped-primary posture cannot borrow it",
                        expert.expert
                    )));
                }
                extents.push((base, len));
            }
        }

        Ok(Self {
            extents,
            _snapshot: Arc::clone(snapshot),
        })
    }
}

/// Rotary cosines and sines for one absolute position, the target's own table.
pub fn qwen38_flash_next_rope(position: u32) -> ([f32; ROTARY_ELEMENTS], [f32; ROTARY_ELEMENTS]) {
    let mut cos = [0.0f32; ROTARY_ELEMENTS];
    let mut sin = [0.0f32; ROTARY_ELEMENTS];
    for element in 0..ROTARY_ELEMENTS {
        let inverse =
            (A::ROPE_THETA as f64).powf(-((2 * element) as f64) / (2 * ROTARY_ELEMENTS) as f64);
        let angle = f64::from(position) * inverse;
        cos[element] = angle.cos() as f32;
        sin[element] = angle.sin() as f32;
    }

    (cos, sin)
}

/// Launches one segment, from the preceding layer's experts through this layer's router.
#[allow(clippy::too_many_arguments)]
fn launch_segment(
    stream: &CudaStream,
    arena: &DeviceArena,
    segment: usize,
    rows: usize,
    ops: Ops<'_>,
    layers: &[LayerPointers],
    workspace: WorkspacePointers,
    endpoint: EndpointPointers,
    embedding_rows: tuisko_gpu::ArenaRegion<u16>,
    residual_a: tuisko_gpu::ArenaRegion<u16>,
    table_scale_bits: u16,
) -> GpuResult<()> {
    // SAFETY: one sealed resident arena, one sealed KV arena and one sealed slot pool own every
    // plane below for the program's whole life, and the graphs are dropped before any of them.
    // Every leaf in the composition selects the same exact row count.
    unsafe {
        if segment == 0 {
            widen_embedding(stream, arena, rows, embedding_rows, residual_a)?;
        } else {
            // --- the tail of layer `segment - 1`: its experts, its combine, its write-back ---
            let previous = layers[segment - 1];
            launch_experts(stream, rows, ops, previous, workspace)?;
            ops.hyper.launch_write_back(
                stream,
                rows,
                workspace.residual_b.cast_const(),
                workspace.block_output.cast_const(),
                workspace.hc_write_gate.cast_const(),
                workspace.residual_a,
            )?;
        }

        if segment == layers.len() {
            // --- the tail segment: the collapsing mixer, then the head ---
            ops.hyper.launch_final_mix(
                stream,
                rows,
                workspace.residual_a.cast_const(),
                endpoint.mixer_norm,
                endpoint.mixer_down,
                endpoint.mixer_up,
                endpoint.mixer_normalized,
                endpoint.mixer_low_rank,
                endpoint.mixer_mixed,
            )?;
            // Decode publishes every sequence; prefill publishes only its final row. Selecting
            // `rows - 1` avoids silently sampling an earlier, valid token position.
            let head_rows = if rows > MAX_BATCH { 1 } else { rows };
            let first = rows - head_rows;
            return ops.lm_head.launch(
                stream,
                head_rows,
                endpoint
                    .mixer_mixed
                    .cast_const()
                    .add(first * <A as Arch>::HIDDEN),
                endpoint.lm_head,
                endpoint.logits,
            );
        }

        // --- the head of layer `segment`, up to and including its router ---
        let layer = layers[segment];
        if let Some(ple) = layer.ple {
            ops.engram.launch_engram(
                ops.hyper,
                stream,
                rows,
                workspace.ple_codes,
                workspace.residual_a.cast_const(),
                Qwen38FlashNextEngramSources {
                    key_proj: ple.key_proj,
                    value_proj: ple.value_proj,
                    norm_key: ple.norm_key,
                    norm_query: ple.norm_query,
                    norm_conv: ple.norm_conv,
                    convolution: ple.convolution,
                    table_scale_bits,
                },
                Qwen38FlashNextEngramWorkspace {
                    embedding: workspace.ple_embedding,
                    key: workspace.ple_key,
                    key_normed: workspace.ple_key_normed,
                    query_normed: workspace.ple_query_normed,
                    value: workspace.ple_value,
                    gated: workspace.ple_gated,
                    gated_normed: workspace.ple_gated_normed,
                    delta: workspace.ple_delta,
                },
                workspace.state_rows,
                ple.conv_state,
                workspace.ple_injected,
            )?;
        }
        let stream_in = match layer.ple {
            Some(_) => workspace.ple_injected.cast_const(),
            None => workspace.residual_a.cast_const(),
        };

        // --- attention bracket ---
        ops.hyper.launch_input_mix(
            stream,
            rows,
            stream_in,
            layer.attention_hc.norm,
            layer.attention_hc.down,
            layer.attention_hc.up,
            layer.attention_hc.inject,
            workspace.hc_normalized,
            workspace.hc_low_rank,
            workspace.hc_mixed,
            workspace.hc_write_gate,
        )?;
        match layer.block {
            BlockPointers::Gdn {
                input_weight,
                control_weight,
                convolution_weight,
                a_log,
                dt_bias,
                norm,
                output_weight,
            } => {
                ops.gdn_input.launch(
                    stream,
                    rows,
                    workspace.hc_mixed.cast_const(),
                    input_weight,
                    workspace.gdn_projected,
                )?;
                ops.gdn_prepare.launch(
                    stream,
                    rows,
                    workspace.hc_mixed.cast_const(),
                    control_weight,
                    a_log,
                    dt_bias,
                    workspace.gdn_projected.cast_const(),
                    convolution_weight,
                    workspace.state_rows,
                    layer.gdn_history,
                    workspace.gdn_log_decay,
                    workspace.gdn_beta,
                    workspace.gdn_convolved,
                )?;
                ops.gdn_recurrence.launch(
                    stream,
                    rows,
                    workspace.gdn_convolved.cast_const(),
                    workspace.gdn_projected.cast_const(),
                    workspace.gdn_log_decay.cast_const(),
                    workspace.gdn_beta.cast_const(),
                    norm,
                    workspace.state_rows,
                    layer.gdn_state,
                    workspace.gdn_recurrent_plane,
                    workspace.gdn_recurrent_output,
                )?;
                ops.block_output.launch(
                    stream,
                    rows,
                    workspace.gdn_recurrent_output.cast_const(),
                    output_weight,
                    workspace.block_output,
                )?;
            }
            BlockPointers::Qsa {
                qkv_weight,
                output_weight,
                query_norm,
                key_norm,
                key_pages,
                value_pages,
            } => {
                ops.qsa_qkv.launch(
                    stream,
                    rows,
                    workspace.hc_mixed.cast_const(),
                    qkv_weight,
                    workspace.qkv,
                )?;
                ops.qsa_prepare.launch(
                    stream,
                    rows,
                    workspace.qkv.cast_const(),
                    query_norm,
                    key_norm,
                    workspace.rope_cos,
                    workspace.rope_sin,
                    workspace.block_tables,
                    workspace.table_rows,
                    crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
                    workspace.cache_positions,
                    workspace.query,
                    key_pages,
                    value_pages,
                    KEY_CACHE_SCALE,
                    VALUE_CACHE_SCALE,
                )?;
                ops.qsa_attention.launch(
                    stream,
                    rows,
                    workspace.query.cast_const(),
                    key_pages.cast_const(),
                    value_pages.cast_const(),
                    workspace.block_tables,
                    workspace.table_rows,
                    crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
                    workspace.lengths,
                    workspace.attention,
                    KEY_CACHE_SCALE,
                    VALUE_CACHE_SCALE,
                )?;
                ops.qsa_gate.launch(
                    stream,
                    rows,
                    workspace.attention,
                    workspace.qkv.cast_const(),
                    workspace.attention_gated,
                )?;
                ops.block_output.launch(
                    stream,
                    rows,
                    workspace.attention_gated.cast_const(),
                    output_weight,
                    workspace.block_output,
                )?;
            }
        }
        ops.hyper.launch_write_back(
            stream,
            rows,
            stream_in,
            workspace.block_output.cast_const(),
            workspace.hc_write_gate.cast_const(),
            workspace.residual_b,
        )?;

        // --- MLP bracket, up to the router: the segment ends where the round begins ---
        ops.hyper.launch_input_mix(
            stream,
            rows,
            workspace.residual_b.cast_const(),
            layer.mlp_hc.norm,
            layer.mlp_hc.down,
            layer.mlp_hc.up,
            layer.mlp_hc.inject,
            workspace.hc_normalized,
            workspace.hc_low_rank,
            workspace.hc_mixed,
            workspace.hc_write_gate,
        )?;
        ops.router.launch(
            stream,
            rows,
            workspace.hc_mixed.cast_const(),
            layer.router_weight,
            workspace.router_logits,
            workspace.expert_indices,
            workspace.routing_weights,
        )?;
    }

    Ok(())
}

/// Widens embeddings into the row-major hyper-connection stream.
///
/// # Safety
///
/// Both regions belong to `arena`, which outlives every replay of the capturing graph.
unsafe fn widen_embedding(
    stream: &CudaStream,
    arena: &DeviceArena,
    rows: usize,
    embedding_rows: tuisko_gpu::ArenaRegion<u16>,
    residual_a: tuisko_gpu::ArenaRegion<u16>,
) -> GpuResult<()> {
    for row in 0..rows {
        for branch in 0..A::HC_COUNT {
            // SAFETY: the caller's contract; source and destination never overlap because the
            // embedding plane and the stream plane are distinct arena regions.
            unsafe {
                arena.copy_slice_from_arena_async(
                    stream,
                    residual_a,
                    row * A::HC_WIDTH + branch * <A as Arch>::HIDDEN,
                    arena,
                    embedding_rows,
                    row * <A as Arch>::HIDDEN,
                    <A as Arch>::HIDDEN,
                )?;
            }
        }
    }

    Ok(())
}

/// One layer's routed experts, shared expert, and combine.
///
/// # Safety
///
/// Every pointer belongs to a sealed arena or the sealed slot pool, and `slot_table` is this
/// layer's own 512-entry view into the global indirection table.
unsafe fn launch_experts(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    layer: LayerPointers,
    workspace: WorkspacePointers,
) -> GpuResult<()> {
    // SAFETY: the caller's contract.
    unsafe {
        ops.experts.launch(
            stream,
            rows,
            &Qwen38FlashNextExpertDispatch {
                input: workspace.hc_mixed.cast_const(),
                expert_indices: workspace.expert_indices.cast_const(),
                routing_weights: workspace.routing_weights.cast_const(),
                slot_table: layer.slot_table,
                slot_pool: workspace.slot_pool,
                weight_scales_2: layer.expert_weight_scales_2,
                shared_gate_weight: layer.shared_gate_weight,
                shared_up_weight: layer.shared_up_weight,
                shared_down_weight: layer.shared_down_weight,
                shared_gate_logit_weight: layer.shared_gate_logit_weight,
                routed_intermediate: workspace.routed_intermediate,
                routed_output: workspace.routed_output,
                shared_intermediate: workspace.shared_intermediate,
                shared_output: workspace.shared_output,
                shared_gate_logit: workspace.shared_gate_logit,
                output: workspace.block_output,
            },
        )
    }
}

impl Qwen38FlashNextResidentModel {
    /// Binds every pointer the captured segments will hold, before a byte is uploaded.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        layout: Qwen38FlashNextResidentLayout,
        arena: DeviceArena,
        kv_arena: DeviceArena,
        pool: StreamingWeightPool,
        hyper: Qwen38FlashNextHyperConnectionOp,
        gdn_input: Qwen38FlashNextGdnInputProjectionOp,
        gdn_prepare: Qwen38FlashNextGdnPrepareOp,
        gdn_recurrence: Qwen38FlashNextGdnRecurrenceOp,
        qsa_qkv: Qwen38FlashNextQsaQkvProjectionOp,
        qsa_prepare: Qwen38FlashNextAttentionQkPrepareOp,
        qsa_attention: tuisko_kernels_sm120::Qwen38FlashNextPagedGqaOp,
        qsa_gate: Qwen38FlashNextAttentionGateOp,
        block_output: Qwen38FlashNextBlockOutputProjectionOp,
        router: Qwen38FlashNextMoeRouterOp,
        experts: Qwen38FlashNextMoeExpertsOp,
        engram: Qwen38FlashNextEngramOp,
        lm_head: Qwen38FlashNextBf16LmHeadOp,
    ) -> EngineResult<Self> {
        let workspace = bind_workspace(&arena, &kv_arena, &pool, &layout)?;
        let endpoint = bind_endpoint(&arena, layout.endpoint())?;
        let layers = bind_layers(&arena, &kv_arena, &pool, &layout)?;

        let embedding_stager = PinnedHostBuffer::zeroed(
            context,
            product(
                "Qwen3.8 Flash-Next embedding stager",
                QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS,
                <A as Arch>::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let engram_stager = PinnedHostBuffer::zeroed(
            context,
            product(
                "Qwen3.8 Flash-Next engram stager",
                QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS,
                product(
                    "Qwen3.8 Flash-Next engram token",
                    A::NGRAM_HEADS,
                    A::NGRAM_HEAD_DIM,
                )?,
            )?,
        )
        .map_err(GpuError::from)?;
        let logit_bank = PinnedHostBuffer::zeroed(
            context,
            product(
                "Qwen3.8 Flash-Next logit bank",
                2 * MAX_BATCH,
                <A as Arch>::VOCAB,
            )?,
        )
        .map_err(GpuError::from)?;

        let table_scale_bits =
            Qwen38FlashNextEngramBindings::bind(snapshot.as_ref(), A::PLE_LAYER)?
                .materialize()?
                .table_scale_bits;
        let base_address = arena.base_address();
        let kv_base_address = kv_arena.base_address();
        let host_pin = pool.host_pin_duration() + pool.bounce_pin_duration();

        Ok(Self {
            segments: Vec::new(),
            arena,
            kv_arena,
            pool,
            _hyper: hyper,
            _gdn_input: gdn_input,
            _gdn_prepare: gdn_prepare,
            _gdn_recurrence: gdn_recurrence,
            _qsa_qkv: qsa_qkv,
            _qsa_prepare: qsa_prepare,
            _qsa_attention: qsa_attention,
            _qsa_gate: qsa_gate,
            _block_output: block_output,
            _router: router,
            _experts: experts,
            _engram: engram,
            _lm_head: lm_head,
            embedding_stager,
            engram_stager,
            logit_bank,
            expert_readback: vec![
                0u16;
                QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS * A::NUM_EXPERTS_PER_TOKEN
            ],
            engram_rows: vec![0i64; QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS * A::NGRAM_HEADS],
            carries: [Qwen38FlashNextEngramCarry::start(); MAX_BATCH],
            round_items: Vec::with_capacity(
                QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS * A::NUM_EXPERTS_PER_TOKEN,
            ),
            slots: Qwen38FlashNextSlotPool::new(layout.physical_pages())?,
            snapshot_owner: Arc::new(()),
            generation: Qwen38FlashNextGenerationTelemetry::default(),
            layers,
            workspace,
            endpoint,
            snapshot,
            context: Arc::clone(context),
            layout,
            base_address,
            kv_base_address,
            table_scale_bits,
            load_stats: Qwen38FlashNextResidentLoadStats {
                weight_upload: Duration::ZERO,
                expert_stage: Duration::ZERO,
                graph_capture: Duration::ZERO,
                host_pin,
                executables: 0,
                definitions: 0,
                staged_items: 0,
                staged_bytes: 0,
            },
        })
    }

    /// Sweeps every backbone weight into the resident arena, then stages every expert.
    fn upload(
        &mut self,
        stream: &CudaStream,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<()> {
        let started = Instant::now();
        let snapshot = Arc::clone(&self.snapshot);
        let regions = self.layout.layers().clone();
        let mut staged_bytes = 0usize;
        let mut staged = Duration::ZERO;
        if let Some(progress) = progress {
            progress.begin_upload(self.layout.resident_weight_bytes());
        }

        for (layer, plan) in regions.iter().enumerate() {
            let hyper = Qwen38FlashNextLayerHyperConnections::bind(snapshot.as_ref(), layer)?;
            upload_hyper_connection(&self.arena, stream, plan.attention_hc, hyper.attention)?;
            upload_hyper_connection(&self.arena, stream, plan.mlp_hc, hyper.mlp)?;

            let moe = Qwen38FlashNextMoeBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
            crate::qwen38_flash_next::layer_upload::upload_moe(
                &self.arena,
                stream,
                plan.moe,
                &moe,
            )?;
            // Reuse this materialization because it swizzles 157 MB of scales per layer.
            let staging = Instant::now();
            staged_bytes += self.stage_layer_experts(layer, &moe)?;
            staged += staging.elapsed();

            match plan.block {
                Qwen38FlashNextBlockWeightRegions::Gdn(regions) => {
                    let gdn = Qwen38FlashNextGdnBindings::bind(snapshot.as_ref(), layer)?
                        .materialize()?;
                    self.arena.copy_from_host(
                        stream,
                        regions.input_weight,
                        &bf16_words(&gdn.input_weight_bf16)?,
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.control_weight,
                        &bf16_words(&gdn.control_weight_bf16)?,
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.convolution_weight,
                        &gdn.convolution_weight.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.a_log,
                        &gdn.a_log.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.dt_bias,
                        &gdn.dt_bias.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.norm,
                        &gdn.norm.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.output_weight,
                        &gdn.output_weight.words().collect::<Vec<_>>(),
                    )?;
                }
                Qwen38FlashNextBlockWeightRegions::Qsa(regions) => {
                    let qsa =
                        Qwen38FlashNextSparseAttentionBindings::bind(snapshot.as_ref(), layer)?
                            .materialize()?;
                    self.arena.copy_from_host(
                        stream,
                        regions.qkv_weight,
                        &bf16_words(&qsa.qkv_weight_bf16)?,
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.output_weight,
                        &qsa.output_weight.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.query_norm,
                        &qsa.query_norm.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.key_norm,
                        &qsa.key_norm.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.indexer_qk_weight,
                        &qsa.indexer.qk_weight.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.indexer_query_norm,
                        &qsa.indexer.query_norm.words().collect::<Vec<_>>(),
                    )?;
                    self.arena.copy_from_host(
                        stream,
                        regions.indexer_key_norm,
                        &qsa.indexer.key_norm.words().collect::<Vec<_>>(),
                    )?;
                }
            }

            if let Some(ple) = plan.ple {
                let engram =
                    Qwen38FlashNextEngramBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
                self.arena.copy_from_host(
                    stream,
                    ple.key_proj,
                    &engram.key_proj_weight.words().collect::<Vec<_>>(),
                )?;
                self.arena.copy_from_host(
                    stream,
                    ple.value_proj,
                    &engram.value_proj_weight.words().collect::<Vec<_>>(),
                )?;
                self.arena.copy_from_host(
                    stream,
                    ple.norm_key,
                    &engram.norm_key.words().collect::<Vec<_>>(),
                )?;
                self.arena.copy_from_host(
                    stream,
                    ple.norm_query,
                    &engram.norm_query.words().collect::<Vec<_>>(),
                )?;
                self.arena.copy_from_host(
                    stream,
                    ple.norm_conv,
                    &engram.norm_conv.words().collect::<Vec<_>>(),
                )?;
                self.arena.copy_from_host(
                    stream,
                    ple.convolution,
                    &engram.convolution_weight.words().collect::<Vec<_>>(),
                )?;
            }
            stream.synchronize().map_err(GpuError::from)?;
            if let Some(progress) = progress {
                progress.submit(self.layout.layer_weight_bytes(layer)?)?;
            }
        }

        let endpoint = Qwen38FlashNextTextEndpointBindings::bind(snapshot.as_ref())?;
        let regions = self.layout.endpoint();
        self.arena.copy_from_host(
            stream,
            regions.mixer_norm,
            &endpoint.mixer.hc_norm.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.mixer_down,
            &endpoint.mixer.input_mix_down.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.mixer_up,
            &endpoint.mixer.input_mix_up.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.lm_head,
            &endpoint.lm_head.words().collect::<Vec<_>>(),
        )?;
        if let Some(progress) = progress {
            progress.submit(self.layout.endpoint_weight_bytes()?)?;
            progress.finish_upload()?;
        }

        // Reservations populate the initially unmapped block table.
        let table = vec![
            QWEN38_FLASH_NEXT_UNMAPPED_PAGE;
            MAX_BATCH * crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
        ];
        self.kv_arena
            .copy_from_host(stream, self.layout.kv_regions().block_tables, &table)?;
        stream.synchronize().map_err(GpuError::from)?;
        if !self.pool.is_fully_staged() {
            return Err(EngineError::layout(
                "Qwen3.8 Flash-Next expert pool is not fully staged after admission",
            ));
        }
        self.load_stats.weight_upload = started.elapsed();
        self.load_stats.expert_stage = staged;
        self.load_stats.staged_items = QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT;
        self.load_stats.staged_bytes = staged_bytes;

        Ok(())
    }

    /// Stages one layer's experts; mapped primary extents remain borrowed.
    fn stage_layer_experts(
        &mut self,
        layer: usize,
        moe: &tuisko_model::MaterializedQwen38FlashNextMoe<'_>,
    ) -> EngineResult<usize> {
        let pinned = self.layout.streaming().primary_source().is_pinned();
        let scales = &moe.experts.scale_e4m3_swizzled;
        let mut staged = 0usize;

        for expert in &moe.experts.experts {
            let item = layer * A::NUM_EXPERTS + expert.expert;
            if pinned {
                let image = expert_slot_image(expert, scales)?;
                let (primary, secondary) =
                    image.split_at(QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES);
                self.pool.stage_item(item, primary, secondary)?;
                staged += image.len();
            } else {
                let mut secondary =
                    Vec::with_capacity(QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES);
                for extent in [expert.gate_up_scale, expert.down_scale] {
                    secondary.extend_from_slice(
                        scales
                            .get(extent.offset..extent.offset + extent.bytes)
                            .ok_or_else(|| {
                                EngineError::layout(format!(
                                    "Qwen3.8 Flash-Next layer {layer} expert {} names a scale extent \
                                     outside its pool",
                                    expert.expert
                                ))
                            })?,
                    );
                }
                self.pool.stage_item(item, &[], &secondary)?;
                staged += secondary.len();
            }
        }

        Ok(staged)
    }

    /// Captures every admitted route in segment-major order.
    fn capture(&mut self, stream: &CudaStream) -> EngineResult<()> {
        let started = Instant::now();
        let ops = self.ops();
        let mut segments =
            Vec::with_capacity(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS * ROUTES_PER_SEGMENT);

        for segment in 0..QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS {
            for rows in (1..=MAX_BATCH).chain(QWEN38_FLASH_NEXT_PREFILL_ROWS) {
                segments.push(CudaGraph::capture(stream, || {
                    launch_segment(
                        stream,
                        &self.arena,
                        segment,
                        rows,
                        ops,
                        &self.layers,
                        self.workspace,
                        self.endpoint,
                        self.layout.endpoint().embedding_rows,
                        self.layout.workspace().residual_a,
                        self.table_scale_bits,
                    )
                })?);
            }
        }

        self.load_stats.graph_capture = started.elapsed();
        self.load_stats.executables = segments.len();
        self.load_stats.definitions = segments.len();
        self.segments = segments;

        Ok(())
    }

    const fn ops(&self) -> Ops<'_> {
        Ops {
            hyper: &self._hyper,
            gdn_input: &self._gdn_input,
            gdn_prepare: &self._gdn_prepare,
            gdn_recurrence: &self._gdn_recurrence,
            qsa_qkv: &self._qsa_qkv,
            qsa_prepare: &self._qsa_prepare,
            qsa_attention: &self._qsa_attention,
            qsa_gate: &self._qsa_gate,
            block_output: &self._block_output,
            router: &self._router,
            experts: &self._experts,
            engram: &self._engram,
            lm_head: &self._lm_head,
        }
    }
}

fn bind_workspace(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    pool: &StreamingWeightPool,
    layout: &Qwen38FlashNextResidentLayout,
) -> EngineResult<WorkspacePointers> {
    let workspace = layout.workspace();

    Ok(WorkspacePointers {
        residual_a: arena.address(workspace.residual_a)?,
        residual_b: arena.address(workspace.residual_b)?,
        hc_normalized: arena.address(workspace.hc_normalized)?,
        hc_low_rank: arena.address(workspace.hc_low_rank)?,
        hc_mixed: arena.address(workspace.hc_mixed)?,
        hc_write_gate: arena.address(workspace.hc_write_gate)?,
        gdn_projected: arena.address(workspace.gdn_projected)?,
        gdn_convolved: arena.address(workspace.gdn_convolved)?,
        gdn_log_decay: arena.address(workspace.gdn_log_decay)?,
        gdn_beta: arena.address(workspace.gdn_beta)?,
        gdn_recurrent_plane: arena.address(workspace.gdn_recurrent_plane)?,
        gdn_recurrent_output: arena.address(workspace.gdn_recurrent_output)?,
        qkv: arena.address(workspace.qkv)?,
        query: arena.address(workspace.query)?,
        attention: arena.address(workspace.attention)?,
        attention_gated: arena.address(workspace.attention_gated)?,
        router_logits: arena.address(workspace.router_logits)?,
        expert_indices: arena.address(workspace.expert_indices)?,
        routing_weights: arena.address(workspace.routing_weights)?,
        routed_intermediate: arena.address(workspace.routed_intermediate)?,
        routed_output: arena.address(workspace.routed_output)?,
        shared_intermediate: arena.address(workspace.shared_intermediate)?,
        shared_output: arena.address(workspace.shared_output)?,
        shared_gate_logit: arena.address(workspace.shared_gate_logit)?,
        block_output: arena.address(workspace.block_output)?,
        state_rows: arena.address(workspace.state_rows)?.cast_const(),
        table_rows: arena.address(workspace.table_rows)?.cast_const(),
        cache_positions: arena.address(workspace.cache_positions)?.cast_const(),
        lengths: arena.address(workspace.lengths)?.cast_const(),
        rope_cos: arena.address(workspace.rope_cos)?.cast_const(),
        rope_sin: arena.address(workspace.rope_sin)?.cast_const(),
        block_tables: kv_arena
            .address(layout.kv_regions().block_tables)?
            .cast_const(),
        ple_codes: arena.address(workspace.ple_codes)?.cast_const(),
        ple_injected: arena.address(workspace.ple_injected)?,
        ple_embedding: arena.address(workspace.ple_embedding)?,
        ple_key: arena.address(workspace.ple_key)?,
        ple_key_normed: arena.address(workspace.ple_key_normed)?,
        ple_query_normed: arena.address(workspace.ple_query_normed)?,
        ple_value: arena.address(workspace.ple_value)?,
        ple_gated: arena.address(workspace.ple_gated)?,
        ple_gated_normed: arena.address(workspace.ple_gated_normed)?,
        ple_delta: arena.address(workspace.ple_delta)?,
        slot_pool: pool.slot_address(0)? as *const u8,
    })
}

fn bind_endpoint(
    arena: &DeviceArena,
    endpoint: Qwen38FlashNextResidentEndpoint,
) -> EngineResult<EndpointPointers> {
    Ok(EndpointPointers {
        mixer_norm: arena.address(endpoint.mixer_norm)?.cast_const(),
        mixer_down: arena.address(endpoint.mixer_down)?.cast_const(),
        mixer_up: arena.address(endpoint.mixer_up)?.cast_const(),
        lm_head: arena.address(endpoint.lm_head)?.cast_const(),
        mixer_normalized: arena.address(endpoint.mixer_normalized)?,
        mixer_low_rank: arena.address(endpoint.mixer_low_rank)?,
        mixer_mixed: arena.address(endpoint.mixer_mixed)?,
        logits: arena.address(endpoint.logits)?,
    })
}

fn bind_layers(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    pool: &StreamingWeightPool,
    layout: &Qwen38FlashNextResidentLayout,
) -> EngineResult<Vec<LayerPointers>> {
    let table_base = pool.table_address()?;
    let kv = layout.kv_regions();
    let mut bound = Vec::with_capacity(<A as Arch>::LAYERS);

    for (layer, plan) in layout.layers().iter().enumerate() {
        let bracket = |hc: crate::qwen38_flash_next::layer_upload::HyperConnectionRegions| {
            Ok::<_, EngineError>(BracketPointers {
                norm: arena.address(hc.norm)?.cast_const(),
                down: arena.address(hc.down)?.cast_const(),
                up: arena.address(hc.up)?.cast_const(),
                inject: arena.address(hc.inject)?.cast_const(),
            })
        };
        let block = match plan.block {
            Qwen38FlashNextBlockWeightRegions::Gdn(gdn) => BlockPointers::Gdn {
                input_weight: arena.address(gdn.input_weight)?.cast_const(),
                control_weight: arena.address(gdn.control_weight)?.cast_const(),
                convolution_weight: arena.address(gdn.convolution_weight)?.cast_const(),
                a_log: arena.address(gdn.a_log)?.cast_const(),
                dt_bias: arena.address(gdn.dt_bias)?.cast_const(),
                norm: arena.address(gdn.norm)?.cast_const(),
                output_weight: arena.address(gdn.output_weight)?.cast_const(),
            },
            Qwen38FlashNextBlockWeightRegions::Qsa(qsa) => {
                // Layer `4k + 3` is the `k`-th attention layer, so its planes are `kv.layers[k]`.
                let planes = kv
                    .layers
                    .get(layer / A::FULL_ATTENTION_INTERVAL)
                    .ok_or_else(|| {
                        EngineError::layout(format!(
                            "Qwen3.8 Flash-Next layer {layer} has no paged cache planes"
                        ))
                    })?;
                BlockPointers::Qsa {
                    qkv_weight: arena.address(qsa.qkv_weight)?.cast_const(),
                    output_weight: arena.address(qsa.output_weight)?.cast_const(),
                    query_norm: arena.address(qsa.query_norm)?.cast_const(),
                    key_norm: arena.address(qsa.key_norm)?.cast_const(),
                    key_pages: kv_arena.address(planes.key_pages)?,
                    value_pages: kv_arena.address(planes.value_pages)?,
                }
            }
        };
        let gdn = plan.persistent.gdn();

        bound.push(LayerPointers {
            attention_hc: bracket(plan.attention_hc)?,
            mlp_hc: bracket(plan.mlp_hc)?,
            block,
            router_weight: arena.address(plan.moe.router_weight)?.cast_const(),
            expert_weight_scales_2: arena.address(plan.moe.expert_weight_scales_2)?.cast_const(),
            shared_gate_weight: arena.address(plan.moe.shared_gate_weight)?.cast_const(),
            shared_up_weight: arena.address(plan.moe.shared_up_weight)?.cast_const(),
            shared_down_weight: arena.address(plan.moe.shared_down_weight)?.cast_const(),
            shared_gate_logit_weight: arena
                .address(plan.moe.shared_gate_logit_weight)?
                .cast_const(),
            // Layer `L`'s view of the global indirection table. The kernels index it by a
            // layer-local expert id, so the view *is* the mapping from `expert` to `item`.
            slot_table: (table_base as usize + layer * A::NUM_EXPERTS * size_of::<u32>())
                as *const u32,
            ple: plan
                .ple
                .map(|ple| {
                    Ok::<_, EngineError>(PlePointers {
                        key_proj: arena.address(ple.key_proj)?.cast_const(),
                        value_proj: arena.address(ple.value_proj)?.cast_const(),
                        norm_key: arena.address(ple.norm_key)?.cast_const(),
                        norm_query: arena.address(ple.norm_query)?.cast_const(),
                        norm_conv: arena.address(ple.norm_conv)?.cast_const(),
                        convolution: arena.address(ple.convolution)?.cast_const(),
                        conv_state: arena.address(
                            plan.persistent
                                .ple()
                                .ok_or_else(|| {
                                    EngineError::layout(
                                        "the Qwen3.8 Flash-Next PLE layer reserves no conv state",
                                    )
                                })?
                                .conv_state,
                        )?,
                    })
                })
                .transpose()?,
            gdn_history: match gdn {
                Some(gdn) => arena.address(gdn.history)?,
                None => std::ptr::null_mut(),
            },
            gdn_state: match gdn {
                Some(gdn) => arena.address(gdn.state)?,
                None => std::ptr::null_mut(),
            },
        });
    }

    Ok(bound)
}

impl Qwen38FlashNextResidentModel {
    /// Resolves each router round before replaying its consuming segment.
    fn forward(
        &mut self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Vec<Qwen38FlashNextLayerStreamTelemetry>> {
        if let Some(reason) = self.pool.poisoned() {
            return Err(EngineError::layout(format!(
                "Qwen3.8 Flash-Next expert cache is poisoned: {reason}"
            )));
        }
        let route = qwen38_flash_next_row_route(rows)?;
        let mut telemetry = Vec::with_capacity(<A as Arch>::LAYERS);

        self.replay_segment(stream, 0, route)?;
        for layer in 0..<A as Arch>::LAYERS {
            // (1) the round's identity, read off the plane the router just published.
            let requests = self.read_expert_round(stream, rows, layer)?;
            // (2) the correctness route: a miss stalls, never skips.
            let round = self.pool.require(&self.round_items)?;
            // (3) the explicit consumer wait.
            self.pool.fence_replay(stream)?;
            // (4) the only replay that reads this round.
            self.replay_segment(stream, layer + 1, route)?;
            // (5) the reclaim fence the next round resolves against.
            self.pool.record_replay_release(stream)?;

            telemetry.push(layer_telemetry(layer, requests, round));
        }
        stream.synchronize().map_err(GpuError::from)?;

        Ok(telemetry)
    }

    fn replay_segment(
        &self,
        stream: &CudaStream,
        segment: usize,
        route: Qwen38FlashNextRowRoute,
    ) -> EngineResult<()> {
        let index = segment * ROUTES_PER_SEGMENT + segment_route_index(route);
        let graph = self.segments.get(index).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.8 Flash-Next segment {segment} has no captured graph for {route:?}"
            ))
        })?;

        // SAFETY: this program owns every captured allocation - the resident arena, the KV
        // arena, the slot pool and every op module - for its whole life, and drops the graphs
        // first.
        unsafe { graph.launch(stream) }?;

        Ok(())
    }

    /// Reads one layer's `u16` selections and maps them to global item ids.
    fn read_expert_round(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        layer: usize,
    ) -> EngineResult<usize> {
        let selections = product(
            "Qwen3.8 Flash-Next round selections",
            rows,
            A::NUM_EXPERTS_PER_TOKEN,
        )?;
        self.arena.copy_prefix_to_host_slice(
            stream,
            self.layout.workspace().expert_indices,
            &mut self.expert_readback[..selections],
        )?;

        self.round_items.clear();
        let base = layer * A::NUM_EXPERTS;
        for &expert in &self.expert_readback[..selections] {
            let expert = expert as usize;
            if expert >= A::NUM_EXPERTS {
                return Err(EngineError::route(format!(
                    "Qwen3.8 Flash-Next layer {layer} router published expert {expert}, outside \
                     0..{}",
                    A::NUM_EXPERTS
                )));
            }
            self.round_items.push((base + expert) as u32);
        }

        Ok(selections)
    }

    /// Gathers one row per token out of the borrowed embedding matrix and uploads it once.
    ///
    /// `T * HIDDEN` BF16 crosses PCIe, never `T * HC_WIDTH`: the four identical branch copies
    /// are made on the device inside `S_0`.
    fn stage_embeddings(&mut self, stream: &CudaStream, tokens: &[u32]) -> EngineResult<usize> {
        let embedding =
            Qwen38FlashNextTextEndpointBindings::bind_embedding(self.snapshot.as_ref())?;
        let source = embedding.bytes();
        let width = <A as Arch>::HIDDEN;
        for (row, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            if token >= <A as Arch>::VOCAB {
                return Err(EngineError::route(format!(
                    "Qwen3.8 Flash-Next token {token} is outside 0..{}",
                    <A as Arch>::VOCAB
                )));
            }
            let begin = token * width * size_of::<u16>();
            let row_bytes = source
                .get(begin..begin + width * size_of::<u16>())
                .ok_or_else(|| {
                    EngineError::layout(
                        "Qwen3.8 Flash-Next embedding row falls outside the mapping",
                    )
                })?;
            let destination = &mut self.embedding_stager[row * width..(row + 1) * width];
            for (word, bytes) in destination.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *word = u16::from_le_bytes([bytes[0], bytes[1]]);
            }
        }

        let values = product("Qwen3.8 Flash-Next staged embedding", tokens.len(), width)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.endpoint().embedding_rows,
            &self.embedding_stager[..values],
        )?;

        Ok(values * size_of::<u16>())
    }

    /// Hashes and gathers engram rows with one carry per sequence slot.
    fn stage_engram(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        slots: &[usize],
        single_sequence: bool,
    ) -> EngineResult<(usize, usize)> {
        let bindings = Qwen38FlashNextEngramBindings::bind(self.snapshot.as_ref(), A::PLE_LAYER)?
            .materialize()?;
        let table = bindings.table()?;
        let token_bytes = table.token_bytes();
        let rows_per_token = A::NGRAM_HEADS;

        if single_sequence {
            let slot = *slots.first().unwrap_or(&0);
            let carry = &mut self.carries[slot];
            let rows = &mut self.engram_rows[..tokens.len() * rows_per_token];
            let destination = &mut self.engram_stager[..tokens.len() * token_bytes];
            gather_qwen38_flash_next_engram_window(table, carry, tokens, rows, destination)?;
        } else {
            for (row, (&token, &slot)) in tokens.iter().zip(slots).enumerate() {
                let carry = &mut self.carries[slot];
                let rows = &mut self.engram_rows[row * rows_per_token..(row + 1) * rows_per_token];
                let destination =
                    &mut self.engram_stager[row * token_bytes..(row + 1) * token_bytes];
                gather_qwen38_flash_next_engram_window(
                    bindings.table()?,
                    carry,
                    std::slice::from_ref(&token),
                    rows,
                    destination,
                )?;
            }
        }

        let bytes = product(
            "Qwen3.8 Flash-Next staged engram",
            tokens.len(),
            token_bytes,
        )?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.workspace().ple_codes,
            &self.engram_stager[..bytes],
        )?;

        Ok((bytes, tokens.len() * rows_per_token))
    }

    /// Uploads per-round inputs; `state_rows` is per sequence rather than per token.
    fn stage_round_inputs(
        &self,
        stream: &CudaStream,
        rows: usize,
        slots: &[usize],
        carry_slots: &[usize],
        positions: &[u32],
    ) -> EngineResult<()> {
        let workspace = self.layout.workspace();
        let state_rows = carry_slots
            .iter()
            .map(|&slot| slot as u32)
            .collect::<Vec<_>>();
        if state_rows.len() > MAX_BATCH {
            return Err(EngineError::route(format!(
                "a Qwen3.8 Flash-Next round names {} carry slots, more than the {MAX_BATCH} the plane holds",
                state_rows.len()
            )));
        }
        let table_rows = slots.iter().map(|&slot| slot as u32).collect::<Vec<_>>();
        let lengths = positions
            .iter()
            .map(|&position| position + 1)
            .collect::<Vec<_>>();
        let mut cosines = Vec::with_capacity(rows * ROTARY_ELEMENTS);
        let mut sines = Vec::with_capacity(rows * ROTARY_ELEMENTS);
        for &position in positions {
            let (cos, sin) = qwen38_flash_next_rope(position);
            cosines.extend_from_slice(&cos);
            sines.extend_from_slice(&sin);
        }

        self.arena
            .copy_prefix_from_host(stream, workspace.state_rows, &state_rows)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.table_rows, &table_rows)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.cache_positions, positions)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.lengths, &lengths)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_cos, &cosines)?;
        self.arena
            .copy_prefix_from_host(stream, workspace.rope_sin, &sines)?;

        Ok(())
    }

    /// Runs one decode step over `B` single-token sequences in the admitted dense band.
    pub fn decode_step(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        positions: &[u32],
        slots: &[usize],
    ) -> EngineResult<Qwen38FlashNextStepTelemetry> {
        let rows = tokens.len();
        if rows == 0 || rows > MAX_BATCH {
            return Err(EngineError::route(format!(
                "Qwen3.8 Flash-Next decode batch {rows} is outside 1..={MAX_BATCH}"
            )));
        }
        if positions.len() != rows || slots.len() != rows {
            return Err(EngineError::route(
                "a Qwen3.8 Flash-Next decode step needs one position and one slot per token",
            ));
        }
        require_distinct_decode_slots(slots)?;
        for &position in positions {
            require_qwen38_flash_next_dense_qsa_visible(position as usize + 1)?;
        }
        for (&slot, &position) in slots.iter().zip(positions) {
            self.admit_round(slot, position as usize, position as usize + 1)?;
        }

        let started = Instant::now();
        self.flush_block_tables(stream)?;
        let embedding_h2d_bytes = self.stage_embeddings(stream, tokens)?;
        let (engram_h2d_bytes, engram_rows) = self.stage_engram(stream, tokens, slots, false)?;
        self.stage_round_inputs(stream, rows, slots, slots, positions)?;
        let layers = self.forward(stream, rows)?;
        let forward = started.elapsed();
        for (&slot, &position) in slots.iter().zip(positions) {
            self.slots.commit(slot, position as usize + 1)?;
        }

        Ok(Qwen38FlashNextStepTelemetry {
            rows,
            layers,
            embedding_h2d_bytes,
            engram_h2d_bytes,
            engram_rows,
            kv_append_bytes: kv_append_bytes(rows),
            forward,
            segment_replays: QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS,
            expert_readbacks: <A as Arch>::LAYERS,
        })
    }

    /// One prefill tile of `T` tokens for a single sequence.
    pub fn prefill_tile(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        first_position: u32,
        slot: usize,
    ) -> EngineResult<Qwen38FlashNextStepTelemetry> {
        let rows = tokens.len();
        if !QWEN38_FLASH_NEXT_PREFILL_ROWS.contains(&rows) {
            return Err(EngineError::route(format!(
                "Qwen3.8 Flash-Next prefill tile {rows} is not an admitted T=32/64/128/1024 route"
            )));
        }
        let last = first_position as usize + rows;
        require_qwen38_flash_next_dense_qsa_visible(last)?;
        self.admit_round(slot, first_position as usize, last)?;

        let positions = (0..rows as u32)
            .map(|offset| first_position + offset)
            .collect::<Vec<_>>();
        let slots = vec![slot; rows];

        let started = Instant::now();
        self.flush_block_tables(stream)?;
        let embedding_h2d_bytes = self.stage_embeddings(stream, tokens)?;
        let (engram_h2d_bytes, engram_rows) = self.stage_engram(stream, tokens, &slots, true)?;
        // One carry slot for the whole causal tile, not one per token.
        self.stage_round_inputs(
            stream,
            rows,
            &slots,
            std::slice::from_ref(&slot),
            &positions,
        )?;
        let layers = self.forward(stream, rows)?;
        let forward = started.elapsed();
        self.slots.commit(slot, last)?;

        Ok(Qwen38FlashNextStepTelemetry {
            rows,
            layers,
            embedding_h2d_bytes,
            engram_h2d_bytes,
            engram_rows,
            kv_append_bytes: kv_append_bytes(rows),
            forward,
            segment_replays: QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS,
            expert_readbacks: <A as Arch>::LAYERS,
        })
    }

    /// Reads this step's logits back through the pinned bank.
    pub fn read_logits(&mut self, stream: &CudaStream, batch: usize) -> EngineResult<&[u16]> {
        let values = product(
            "Qwen3.8 Flash-Next logit readback",
            batch,
            <A as Arch>::VOCAB,
        )?;
        let bank = &mut self.logit_bank[..values];
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.endpoint().logits, bank)?;

        Ok(&self.logit_bank[..values])
    }

    /// Reads this step's logits into a caller-owned row bank.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        let values = product("Flash-Next logit readback", batch, <A as Arch>::VOCAB)?;
        let offered = destination.len();
        let destination = destination.get_mut(..values).ok_or_else(|| {
            EngineError::route(format!(
                "a Flash-Next logit readback of {batch} rows needs {values} values, the caller \
                 offered {offered}"
            ))
        })?;
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.endpoint().logits, destination)?;

        Ok(())
    }

    /// Page-locked host bytes this program owns for staging and readback.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
            + self.engram_stager.num_bytes()
            + self.logit_bank.num_bytes()
    }

    /// The shared page pool behind the eight block-table rows.
    pub const fn slots(&self) -> &Qwen38FlashNextSlotPool {
        &self.slots
    }

    /// Grows one slot's mapping to cover `tokens`, drawing from the shared pool.
    pub fn reserve_slot(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        tokens: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        let change = self.slots.reserve(slot, tokens)?;
        self.flush_block_tables(stream)?;

        Ok(change)
    }

    /// Truncates paged cache for lifecycle qualification only.
    #[cfg(feature = "qualification")]
    pub fn qualification_truncate_slot(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        tokens: usize,
    ) -> EngineResult<Qwen38FlashNextSlotChange> {
        let change = self.slots.truncate(slot, tokens)?;
        self.flush_block_tables(stream)?;

        Ok(change)
    }

    /// Holds one slot's committed prefix for reuse after its request finished.
    pub fn retain_slot(&mut self, slot: usize) -> EngineResult<usize> {
        self.slots.retain(slot)
    }

    /// Returns a slot's pages and clears its sequence carries.
    pub fn recycle_slot(
        &mut self,
        stream: &CudaStream,
        slot: usize,
    ) -> EngineResult<Qwen38FlashNextSlotRelease> {
        let release = self.slots.recycle(slot)?;
        self.reset_slot(stream, slot)?;
        self.flush_block_tables(stream)?;

        Ok(release)
    }

    /// Lifecycle position of one slot.
    pub fn slot_state(&self, slot: usize) -> EngineResult<Qwen38FlashNextSlotState> {
        self.slots.state(slot)
    }

    /// Tokens one slot's cache currently covers.
    pub fn slot_tokens(&self, slot: usize) -> EngineResult<usize> {
        self.slots.tokens(slot)
    }

    /// Host engram carry for qualification.
    #[cfg(feature = "qualification")]
    pub fn qualification_engram_carry(
        &self,
        slot: usize,
    ) -> EngineResult<[u32; tuisko_model::QWEN38_FLASH_NEXT_ENGRAM_CONTEXT_LEN]> {
        self.carries
            .get(slot)
            .map(Qwen38FlashNextEngramCarry::previous)
            .ok_or_else(|| {
                EngineError::route(format!(
                    "Qwen3.8 Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
                ))
            })
    }

    /// Snapshots overwritten recurrent state. Append-only paged K/V remains in place.
    pub fn snapshot_slot(
        &self,
        stream: &CudaStream,
        slot: usize,
    ) -> EngineResult<Qwen38FlashNextSlotSnapshot> {
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if self.slots.state(slot)? != Qwen38FlashNextSlotState::Active {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} must be active before it can be snapshotted"
            )));
        }
        let mut gdn_history = Vec::new();
        let mut gdn_state = Vec::new();
        let mut ple_conv_state = Vec::new();

        for plan in self.layout.layers() {
            if let Some(gdn) = plan.persistent.gdn() {
                let (history, state) = gdn.slot_widths();
                let base = gdn_history.len();
                gdn_history.resize(base + history, 0u16);
                self.arena.copy_slice_to_host_slice(
                    stream,
                    gdn.history,
                    slot * history,
                    &mut gdn_history[base..],
                )?;
                let base = gdn_state.len();
                gdn_state.resize(base + state, 0.0f32);
                self.arena.copy_slice_to_host_slice(
                    stream,
                    gdn.state,
                    slot * state,
                    &mut gdn_state[base..],
                )?;
            }
            if let Some(ple) = plan.persistent.ple() {
                let width = ple.slot_width();
                ple_conv_state.resize(width, 0u16);
                self.arena.copy_slice_to_host_slice(
                    stream,
                    ple.conv_state,
                    slot * width,
                    &mut ple_conv_state,
                )?;
            }
        }
        stream.synchronize().map_err(GpuError::from)?;

        Ok(Qwen38FlashNextSlotSnapshot {
            owner: Arc::clone(&self.snapshot_owner),
            slot,
            sequence: self.slots.sequence(slot)?,
            tokens: self.slots.tokens(slot)?,
            carry: self.carries[slot],
            gdn_history,
            gdn_state,
            ple_conv_state,
        })
    }

    /// Restores recurrent state and cache length for the snapshot's slot.
    pub fn restore_slot(
        &mut self,
        stream: &CudaStream,
        snapshot: &Qwen38FlashNextSlotSnapshot,
    ) -> EngineResult<()> {
        let slot = snapshot.slot;
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if !Arc::ptr_eq(&snapshot.owner, &self.snapshot_owner) {
            return Err(EngineError::route(
                "a Flash-Next slot snapshot belongs to another resident model",
            ));
        }
        if self.slots.sequence(slot)? != snapshot.sequence {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} has restarted since this snapshot was taken"
            )));
        }
        let current = self.slots.tokens(slot)?;
        if current < snapshot.tokens {
            return Err(EngineError::route(format!(
                "Flash-Next slot {slot} covers {current} tokens, short of the snapshot's {}; \
                 released K/V pages cannot be restored from recurrent state",
                snapshot.tokens
            )));
        }
        let mut history_cursor = 0usize;
        let mut state_cursor = 0usize;

        for plan in self.layout.layers().clone() {
            if let Some(gdn) = plan.persistent.gdn() {
                let (history, state) = gdn.slot_widths();
                let source = snapshot
                    .gdn_history
                    .get(history_cursor..history_cursor + history)
                    .ok_or_else(|| {
                        EngineError::layout(
                            "a Flash-Next slot snapshot is short of one layer's GDN history",
                        )
                    })?;
                self.arena
                    .copy_slice_from_host(stream, gdn.history, slot * history, source)?;
                history_cursor += history;

                let source = snapshot
                    .gdn_state
                    .get(state_cursor..state_cursor + state)
                    .ok_or_else(|| {
                        EngineError::layout(
                            "a Flash-Next slot snapshot is short of one layer's GDN state",
                        )
                    })?;
                self.arena
                    .copy_slice_from_host(stream, gdn.state, slot * state, source)?;
                state_cursor += state;
            }
            if let Some(ple) = plan.persistent.ple() {
                let width = ple.slot_width();
                if snapshot.ple_conv_state.len() != width {
                    return Err(EngineError::layout(
                        "a Flash-Next slot snapshot carries the wrong PLE conv-state width",
                    ));
                }
                self.arena.copy_slice_from_host(
                    stream,
                    ple.conv_state,
                    slot * width,
                    &snapshot.ple_conv_state,
                )?;
            }
        }
        self.carries[slot] = snapshot.carry;
        if current > snapshot.tokens {
            self.slots.rollback(slot, snapshot.tokens)?;
        }
        self.slots.reserve(slot, snapshot.tokens)?;
        self.flush_block_tables(stream)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    /// Refuses a round that is not an active, exactly append-only write.
    fn admit_round(&self, slot: usize, first: usize, visible: usize) -> EngineResult<()> {
        let state = self.slots.state(slot)?;
        let committed = self.slots.tokens(slot)?;
        require_append_position(slot, state, committed, first)?;
        let mapped = self.slots.pages(slot)?.len() * QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS;
        if visible > mapped {
            return Err(EngineError::route(format!(
                "a Flash-Next round on slot {slot} reaches visible length {visible} against \
                 {mapped} mapped tokens; reserve the slot before the round rather than letting \
                 it attend through an unmapped page"
            )));
        }

        Ok(())
    }

    /// Uploads block-table entries changed since the last flush.
    fn flush_block_tables(&mut self, stream: &CudaStream) -> EngineResult<()> {
        if !self.slots.has_dirty() {
            return Ok(());
        }
        let stride = crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES;
        let region = self.layout.kv_regions().block_tables;
        for slot in 0..MAX_BATCH {
            let Some(entries) = self.slots.dirty_range(slot)? else {
                continue;
            };
            let row = self.slots.table_row(slot)?;
            let source = row.get(entries.clone()).ok_or_else(|| {
                EngineError::layout("a Flash-Next block-table flush named entries outside its row")
            })?;
            self.kv_arena.copy_slice_from_host(
                stream,
                region,
                slot * stride + entries.start,
                source,
            )?;
            self.slots.clear_dirty(slot)?;
        }

        Ok(())
    }

    /// Clears every slot-owned carry and the whole expert cache.
    pub fn reset_state(&mut self, stream: &CudaStream) -> EngineResult<()> {
        for plan in self.layout.layers() {
            if let Some(gdn) = plan.persistent.gdn() {
                self.arena.fill(stream, gdn.history, 0)?;
                self.arena.fill(stream, gdn.state, 0)?;
            }
            if let Some(ple) = plan.persistent.ple() {
                self.arena.fill(stream, ple.conv_state, 0)?;
            }
        }
        for planes in &self.layout.kv_regions().layers {
            self.arena_kv_fill(stream, planes.key_pages)?;
            self.arena_kv_fill(stream, planes.value_pages)?;
            self.arena_kv_fill(stream, planes.indexer_pages)?;
        }
        self.carries = [Qwen38FlashNextEngramCarry::start(); MAX_BATCH];
        self.pool.reset()?;
        self.slots.reset()?;
        self.flush_block_tables(stream)?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    /// Restarts one slot without releasing its pages or touching another sequence.
    pub fn reset_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Qwen3.8 Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        for plan in self.layout.layers() {
            if let Some(gdn) = plan.persistent.gdn() {
                let (history, state) = gdn.slot_widths();
                self.arena
                    .fill_slice(stream, gdn.history, slot * history, history, 0)?;
                self.arena
                    .fill_slice(stream, gdn.state, slot * state, state, 0)?;
            }
            if let Some(ple) = plan.persistent.ple() {
                let width = ple.slot_width();
                self.arena
                    .fill_slice(stream, ple.conv_state, slot * width, width, 0)?;
            }
        }
        self.carries[slot] = Qwen38FlashNextEngramCarry::start();
        stream.synchronize().map_err(GpuError::from)?;
        self.slots.restart(slot)?;

        Ok(())
    }

    fn arena_kv_fill(
        &self,
        stream: &CudaStream,
        region: tuisko_gpu::ArenaRegion<u8>,
    ) -> EngineResult<()> {
        self.kv_arena.fill(stream, region, 0)?;

        Ok(())
    }
}

fn require_append_position(
    slot: usize,
    state: Qwen38FlashNextSlotState,
    committed: usize,
    first: usize,
) -> EngineResult<()> {
    if state != Qwen38FlashNextSlotState::Active {
        return Err(EngineError::route(format!(
            "a Flash-Next round cannot append to slot {slot} while it is {state:?}"
        )));
    }
    if first != committed {
        return Err(EngineError::route(format!(
            "a Flash-Next round on slot {slot} starts at {first}, but its append position is \
             {committed}"
        )));
    }

    Ok(())
}

fn require_distinct_decode_slots(slots: &[usize]) -> EngineResult<()> {
    let mut seen = [false; MAX_BATCH];
    for &slot in slots {
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Qwen3.8 Flash-Next slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if std::mem::replace(&mut seen[slot], true) {
            return Err(EngineError::route(format!(
                "Qwen3.8 Flash-Next slot {slot} appears more than once in one decode batch"
            )));
        }
    }

    Ok(())
}

fn layer_telemetry(
    layer: usize,
    requests: usize,
    round: StreamingRound,
) -> Qwen38FlashNextLayerStreamTelemetry {
    Qwen38FlashNextLayerStreamTelemetry {
        layer,
        requests,
        hits: round.hits(),
        misses: round.misses(),
        uploaded_bytes: round.uploaded_bytes(),
        stalled: round.stalled(),
    }
}

/// Bytes appended to the K/V planes; the reserved indexer plane is not written.
fn kv_append_bytes(rows: usize) -> usize {
    rows * crate::QWEN38_FLASH_NEXT_ATTENTION_LAYERS
        * 2
        * <A as Arch>::NUM_KV_HEADS
        * <A as Arch>::HEAD_DIM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decode_batch_requires_distinct_in_range_slots() {
        assert!(require_distinct_decode_slots(&[0, 3, 7]).is_ok());
        assert!(require_distinct_decode_slots(&[2, 2]).is_err());
        assert!(require_distinct_decode_slots(&[MAX_BATCH]).is_err());
    }

    #[test]
    fn a_round_is_active_and_exactly_append_only() {
        assert!(require_append_position(0, Qwen38FlashNextSlotState::Active, 17, 17).is_ok());
        assert!(require_append_position(0, Qwen38FlashNextSlotState::Free, 0, 0).is_err());
        assert!(require_append_position(0, Qwen38FlashNextSlotState::Retained, 17, 17).is_err());
        assert!(require_append_position(0, Qwen38FlashNextSlotState::Active, 17, 18).is_err());
        assert!(require_append_position(0, Qwen38FlashNextSlotState::Active, 17, 16).is_err());
    }

    #[test]
    fn the_segment_inventory_is_forty_nine_by_twelve() {
        assert_eq!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, 49);
        assert_eq!(ROUTES_PER_SEGMENT, 12);
        assert_eq!(
            QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS * ROUTES_PER_SEGMENT,
            588
        );
    }

    #[test]
    fn every_admitted_route_indexes_a_distinct_executable() {
        let mut seen = std::collections::BTreeSet::new();
        for segment in 0..QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS {
            for rows in (1..=MAX_BATCH).chain(QWEN38_FLASH_NEXT_PREFILL_ROWS) {
                let route = qwen38_flash_next_row_route(rows).unwrap();
                assert!(seen.insert(segment * ROUTES_PER_SEGMENT + segment_route_index(route)));
            }
        }

        assert_eq!(seen.len(), 588);
        assert_eq!(seen.last().copied(), Some(587));
    }

    #[test]
    fn the_two_route_families_are_laid_end_to_end_not_overlaid() {
        // `graph_index` numbers decode and prefill separately, so B=1 and T=32 are both zero.
        // A flat per-segment inventory has to separate them, and this is the assertion that
        // says so: without it a prefill replay launches a decode graph and the failure looks
        // like a numerical bug rather than a routing one.
        assert_eq!(
            qwen38_flash_next_row_route(1).unwrap().graph_index(),
            qwen38_flash_next_row_route(32).unwrap().graph_index()
        );
        assert_eq!(
            segment_route_index(qwen38_flash_next_row_route(1).unwrap()),
            0
        );
        assert_eq!(
            segment_route_index(qwen38_flash_next_row_route(8).unwrap()),
            7
        );
        assert_eq!(
            segment_route_index(qwen38_flash_next_row_route(32).unwrap()),
            8
        );
        assert_eq!(
            segment_route_index(qwen38_flash_next_row_route(1_024).unwrap()),
            11
        );
    }

    #[test]
    fn the_kv_append_figure_counts_only_written_planes() {
        // The reserved indexer plane is not written by the dense QSA route.
        assert_eq!(kv_append_bytes(1), 12_288);
        assert_eq!(kv_append_bytes(8), 98_304);
    }

    #[test]
    fn a_layers_table_view_addresses_its_own_five_hundred_and_twelve_items() {
        // The composition fact that lets 48 layers share one cache: the kernels index the table
        // by a layer-local expert id, so the view offset *is* the item mapping.
        for layer in 0..<A as Arch>::LAYERS {
            let offset = layer * A::NUM_EXPERTS * size_of::<u32>();
            assert_eq!(offset, layer * 2_048);
            assert_eq!(
                offset / size_of::<u32>() + A::NUM_EXPERTS,
                (layer + 1) * 512
            );
        }
        assert_eq!(<A as Arch>::LAYERS * A::NUM_EXPERTS, 24_576);
    }
}

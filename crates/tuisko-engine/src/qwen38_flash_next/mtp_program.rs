//! Resident Qwen3.8 Flash-Next MTP draft program.

use crate::common::math::product;
use crate::common::mtp::VERIFY_ROWS;
use crate::common::progress::ResidentLoadProgress;
use crate::common::streaming::{
    StreamingMappedPrimary, StreamingPrimarySource, StreamingRound, StreamingWeightPool,
};
use crate::qwen38_flash_next::layer_route::{
    QWEN38_FLASH_NEXT_PREFILL_ROWS, require_qwen38_flash_next_dense_qsa_visible,
};
use crate::qwen38_flash_next::layer_upload::{bf16_words, upload_hyper_connection};
use crate::qwen38_flash_next::mtp_layout::{
    QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT, QWEN38_FLASH_NEXT_MTP_EXPERT_RESIDENT_SLOTS,
    QWEN38_FLASH_NEXT_MTP_MAX_ROWS, Qwen38FlashNextMtpFusionRegions, Qwen38FlashNextMtpKvPlanes,
    Qwen38FlashNextMtpLayerRegions, Qwen38FlashNextMtpLayout, Qwen38FlashNextMtpResidency,
    Qwen38FlashNextMtpWorkspace,
};
use crate::qwen38_flash_next::resident_model::{
    Ops, Qwen38FlashNextLayerStreamTelemetry, Qwen38FlashNextResidentModel, layer_telemetry,
    qwen38_flash_next_rope,
};
use crate::{EngineError, EngineResult, MAX_BATCH};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{
    Qwen38FlashNextMtpExpertDispatch, Qwen38FlashNextMtpFusionProjectionOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextMtpBindings,
    Qwen38FlashNextTextEndpointBindings,
};

type A = Qwen38FlashNext;

/// Qualified E4M3 key-cache scale.
const KEY_CACHE_SCALE: f32 = 0.031_25;

/// Qualified E4M3 value-cache scale.
const VALUE_CACHE_SCALE: f32 = 0.062_5;

const ROTARY_ELEMENTS: usize = 32;

/// Fusion row inventory: one proposal row plus the prompt ladder.
pub const QWEN38_FLASH_NEXT_MTP_ROUTES: [usize; 5] = [1, 32, 64, 128, 1_024];

/// Routed segments, excluding the LM head.
pub const QWEN38_FLASH_NEXT_MTP_SEGMENTS: usize = 2;

/// Rows carried by one proposal.
pub const QWEN38_FLASH_NEXT_PROPOSAL_ROWS: usize = 1;

/// Maximum routed rows supported by the draft expert pool.
pub const QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS: usize =
    QWEN38_FLASH_NEXT_MTP_EXPERT_RESIDENT_SLOTS / A::NUM_EXPERTS_PER_TOKEN;

const _: () = assert!(QWEN38_FLASH_NEXT_MTP_ROUTES[0] == 1);
const _: () =
    assert!(QWEN38_FLASH_NEXT_MTP_ROUTES.len() == 1 + QWEN38_FLASH_NEXT_PREFILL_ROWS.len());
const _: () = assert!(QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS == 12);
const _: () = assert!(QWEN38_FLASH_NEXT_PROPOSAL_ROWS <= QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS);
const _: () = assert!(QWEN38_FLASH_NEXT_MTP_ROUTES[1] > QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS);

fn draft_route_index(rows: usize) -> EngineResult<usize> {
    QWEN38_FLASH_NEXT_MTP_ROUTES
        .iter()
        .position(|&admitted| admitted == rows)
        .ok_or_else(|| {
            EngineError::route(format!(
                "Flash-Next draft row count {rows} is outside the draft's own schedule \
                 {QWEN38_FLASH_NEXT_MTP_ROUTES:?}; a wider realignment replays the one-row route"
            ))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen38FlashNextMtpStage {
    Prime,
    Head,
    Tail,
}

/// Source of the target stream at `position - 1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38FlashNextMtpStream {
    /// Retained final row from the previous target round.
    Carry,
    /// A verified row in the target's published stream.
    TargetRow(usize),
    /// The draft's previous output.
    Draft,
    /// Carry followed by the target tile shifted one row.
    ShiftedTile,
}

/// Streaming and timing evidence one draft launch produced.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextMtpStepTelemetry {
    rows: usize,
    layer: Qwen38FlashNextLayerStreamTelemetry,
    embedding_h2d_bytes: usize,
    kv_append_bytes: usize,
    forward: Duration,
    proposed: bool,
}

impl Qwen38FlashNextMtpStepTelemetry {
    /// Rows this launch carried.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The draft pool's own round evidence for this launch.
    pub const fn layer(&self) -> Qwen38FlashNextLayerStreamTelemetry {
        self.layer
    }

    /// Token-embedding bytes this launch uploaded.
    pub const fn embedding_h2d_bytes(&self) -> usize {
        self.embedding_h2d_bytes
    }

    /// Bytes this launch appended to the draft's own cache planes.
    pub const fn kv_append_bytes(&self) -> usize {
        self.kv_append_bytes
    }

    /// Host-observed wall time from the first staged byte to the readable output.
    pub const fn forward(&self) -> Duration {
        self.forward
    }

    /// Whether the head ran, which is what separates a proposal from an extend.
    pub const fn proposed(&self) -> bool {
        self.proposed
    }
}

/// Construction evidence the draft block adds beside the target's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen38FlashNextMtpLoadStats {
    upload: Duration,
    staged_bytes: usize,
    capture: Duration,
    executables: usize,
}

impl Qwen38FlashNextMtpLoadStats {
    /// Wall time the draft's resident weights took to reach the device.
    pub const fn upload(self) -> Duration {
        self.upload
    }

    /// Host bytes the draft pool staged, which is its `down` plane alone under a mapped primary.
    pub const fn staged_bytes(self) -> usize {
        self.staged_bytes
    }

    /// Wall time capturing every draft executable took.
    pub const fn capture(self) -> Duration {
        self.capture
    }

    /// Captured executables the draft program retains.
    pub const fn executables(self) -> usize {
        self.executables
    }
}

/// Mapped `gate_up` expert extents.
struct Qwen38FlashNextMtpExpertSource {
    extents: Vec<(*const u8, usize)>,
    // Field order is drop order: the extents point into this mapping.
    _snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
}

// SAFETY: every extent points into the read-only checkpoint mapping the retained `Arc` keeps
// alive for at least as long as this source, and nothing here ever writes through one.
unsafe impl Send for Qwen38FlashNextMtpExpertSource {}
// SAFETY: as above; `primary_extent` takes `&self` and only reads.
unsafe impl Sync for Qwen38FlashNextMtpExpertSource {}

impl StreamingMappedPrimary for Qwen38FlashNextMtpExpertSource {
    fn primary_extent(&self, item: usize) -> EngineResult<&[u8]> {
        let &(pointer, len) = self.extents.get(item).ok_or_else(|| {
            EngineError::layout(format!(
                "Flash-Next draft expert item {item} is outside \
                 0..{QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT}"
            ))
        })?;

        // SAFETY: `pointer` and `len` were validated as one contiguous run inside the retained
        // mapping when this source was built, and the mapping outlives `&self`.
        Ok(unsafe { std::slice::from_raw_parts(pointer, len) })
    }
}

/// Every resident weight address one draft segment reads.
#[derive(Clone, Copy)]
struct DraftWeights {
    norm_embedding: *const u16,
    norm_hidden: *const u16,
    fc_embedding: *const u16,
    fc_hidden: *const u16,

    attention_hc_norm: *const u16,
    attention_hc_down: *const u16,
    attention_hc_up: *const u16,
    attention_hc_inject: *const u16,
    mlp_hc_norm: *const u16,
    mlp_hc_down: *const u16,
    mlp_hc_up: *const u16,
    mlp_hc_inject: *const u16,

    qkv_weight: *const u16,
    output_weight: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,
    router_weight: *const u16,
    shared_gate_weight: *const u16,
    shared_up_weight: *const u16,
    shared_down_weight: *const u16,
    shared_gate_logit_weight: *const u16,

    mixer_norm: *const u16,
    mixer_down: *const u16,
    mixer_up: *const u16,

    slot_table: *const u32,
    slot_pool: *const u8,
}

/// Every activation address one draft segment writes.
#[derive(Clone, Copy)]
struct DraftPointers {
    residual_a: *mut u16,
    residual_b: *mut u16,

    fusion_hidden: *mut u16,
    fusion_projected: *mut u16,

    hc_normalized: *mut u16,
    hc_low_rank: *mut u16,
    hc_mixed: *mut u16,
    hc_write_gate: *mut u16,

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

    mixer_normalized: *mut u16,
    mixer_low_rank: *mut u16,
    mixer_mixed: *mut u16,
    logits: *mut u16,

    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,

    key_pages: *mut u8,
    value_pages: *mut u8,
}

/// The Flash-Next draft block, resident beside the target it proposes for.
pub struct Qwen38FlashNextMtpProgram {
    // Drop graphs before the arenas, the pool and the target's modules they retain.
    head: Vec<CudaGraph>,
    tail_segments: Vec<CudaGraph>,
    segments: Vec<CudaGraph>,
    primes: Vec<CudaGraph>,
    arena: DeviceArena,
    // Retained for its lifetime alone: every draft graph holds addresses inside it, and the
    // draft reaches its planes through the pointers bound before capture.
    _kv_arena: DeviceArena,
    pool: StreamingWeightPool,
    _fusion: Qwen38FlashNextMtpFusionProjectionOp,

    embedding_stager: PinnedHostBuffer<u16>,
    logit_bank: PinnedHostBuffer<u16>,
    expert_readback: Vec<u16>,
    round_items: Vec<u32>,

    weights: DraftWeights,
    pointers: DraftPointers,

    target: Qwen38FlashNextResidentModel,
    layout: Qwen38FlashNextMtpLayout,
    base_address: u64,
    kv_base_address: u64,
    load_stats: Qwen38FlashNextMtpLoadStats,
}

impl Qwen38FlashNextMtpProgram {
    /// Loads the target and the draft against one joint residency solve.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_with_progress(context, snapshot, None)
    }

    /// Loads the joint target and draft plan with progress reporting.
    pub fn from_snapshot_with_progress(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<Self> {
        let residency = Qwen38FlashNextMtpResidency::build()?;
        let layout = residency.draft().clone();
        let target = Qwen38FlashNextResidentModel::from_plan_with_progress(
            context,
            Arc::clone(&snapshot),
            residency.target().clone(),
            progress,
            layout.resident_weight_bytes(),
        )?;
        let stream = context.new_stream().map_err(GpuError::from)?;

        let arena = DeviceArena::zeroed(&stream, layout.resident_builder())?;
        let kv_arena = DeviceArena::zeroed(&stream, layout.kv_builder())?;
        stream.synchronize().map_err(GpuError::from)?;

        let pool = match layout.streaming().primary_source() {
            StreamingPrimarySource::Mapped => {
                let source = bind_expert_source(&snapshot)?;
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

        let fusion = Qwen38FlashNextMtpFusionProjectionOp::new(context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(
            context,
            product(
                "Flash-Next draft embedding stager",
                QWEN38_FLASH_NEXT_MTP_MAX_ROWS,
                <A as Arch>::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let logit_bank = PinnedHostBuffer::zeroed(
            context,
            product(
                "Flash-Next draft logit bank",
                VERIFY_ROWS,
                <A as Arch>::VOCAB,
            )?,
        )
        .map_err(GpuError::from)?;

        let base_address = arena.base_address();
        let kv_base_address = kv_arena.base_address();
        let weights = bind_weights(&arena, &pool, &layout)?;
        let pointers = bind_pointers(&arena, &kv_arena, &target, &layout)?;

        let mut program = Self {
            head: Vec::new(),
            tail_segments: Vec::new(),
            segments: Vec::new(),
            primes: Vec::new(),
            arena,
            _kv_arena: kv_arena,
            pool,
            _fusion: fusion,
            embedding_stager,
            logit_bank,
            expert_readback: vec![0; MAX_BATCH * A::NUM_EXPERTS_PER_TOKEN],
            round_items: Vec::with_capacity(
                QWEN38_FLASH_NEXT_MTP_MAX_ROWS * A::NUM_EXPERTS_PER_TOKEN,
            ),
            weights,
            pointers,
            target,
            layout,
            base_address,
            kv_base_address,
            load_stats: Qwen38FlashNextMtpLoadStats::default(),
        };
        program.upload(&stream, &snapshot, progress)?;
        program.capture(&stream)?;
        if let Some(progress) = progress {
            progress.finish();
        }

        Ok(program)
    }

    /// The target this draft proposes for.
    pub const fn target(&self) -> &Qwen38FlashNextResidentModel {
        &self.target
    }

    /// The target, mutably: every round the loop drives goes through it.
    pub const fn target_mut(&mut self) -> &mut Qwen38FlashNextResidentModel {
        &mut self.target
    }

    /// The draft's own three-arena plan.
    pub const fn layout(&self) -> &Qwen38FlashNextMtpLayout {
        &self.layout
    }

    /// The draft's own expert cache.
    pub const fn pool(&self) -> &StreamingWeightPool {
        &self.pool
    }

    /// Construction evidence the draft added beside the target's.
    pub const fn load_stats(&self) -> Qwen38FlashNextMtpLoadStats {
        self.load_stats
    }

    /// Captured executables the draft program retains.
    pub fn executables(&self) -> usize {
        self.segments.len() + self.primes.len() + self.tail_segments.len() + self.head.len()
    }

    /// Stable base address of the draft's resident arena.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable base address of the draft's own cache mirror.
    pub const fn kv_base_address(&self) -> u64 {
        self.kv_base_address
    }

    /// Page-locked staging bytes both programs hold.
    pub fn host_stager_bytes(&self) -> usize {
        self.target.host_stager_bytes()
            + self.embedding_stager.num_bytes()
            + self.logit_bank.num_bytes()
    }
}

fn bind_expert_source(
    snapshot: &Arc<CheckpointSnapshot<Qwen38FlashNext>>,
) -> EngineResult<Qwen38FlashNextMtpExpertSource> {
    let block = Qwen38FlashNextMtpBindings::bind(snapshot.as_ref())?.materialize()?;
    let layer = block.layers.first().ok_or_else(|| {
        EngineError::layout("the Flash-Next draft block bound no layer to stream experts from")
    })?;
    let pool = &layer.mlp.experts;
    let mut extents = Vec::with_capacity(pool.expert_count);
    for expert in 0..pool.expert_count {
        let begin = product(
            "Flash-Next draft expert offset",
            expert,
            pool.gate_up_stride_bytes,
        )?;
        let extent = pool
            .gate_up
            .get(begin..begin + pool.gate_up_stride_bytes)
            .ok_or_else(|| {
                EngineError::layout(format!(
                    "Flash-Next draft expert {expert} names a gate-up extent outside its plane"
                ))
            })?;
        extents.push((extent.as_ptr(), extent.len()));
    }

    Ok(Qwen38FlashNextMtpExpertSource {
        extents,
        _snapshot: Arc::clone(snapshot),
    })
}

fn bind_weights(
    arena: &DeviceArena,
    pool: &StreamingWeightPool,
    layout: &Qwen38FlashNextMtpLayout,
) -> EngineResult<DraftWeights> {
    let fusion: Qwen38FlashNextMtpFusionRegions = layout.fusion();
    let layer: Qwen38FlashNextMtpLayerRegions = layout.layer();
    let mixer = layout.mixer();

    Ok(DraftWeights {
        norm_embedding: arena.address(fusion.norm_embedding)?.cast_const(),
        norm_hidden: arena.address(fusion.norm_hidden)?.cast_const(),
        fc_embedding: arena.address(fusion.fc_embedding)?.cast_const(),
        fc_hidden: arena.address(fusion.fc_hidden)?.cast_const(),

        attention_hc_norm: arena.address(layer.attention_hc.norm)?.cast_const(),
        attention_hc_down: arena.address(layer.attention_hc.down)?.cast_const(),
        attention_hc_up: arena.address(layer.attention_hc.up)?.cast_const(),
        attention_hc_inject: arena.address(layer.attention_hc.inject)?.cast_const(),
        mlp_hc_norm: arena.address(layer.mlp_hc.norm)?.cast_const(),
        mlp_hc_down: arena.address(layer.mlp_hc.down)?.cast_const(),
        mlp_hc_up: arena.address(layer.mlp_hc.up)?.cast_const(),
        mlp_hc_inject: arena.address(layer.mlp_hc.inject)?.cast_const(),

        qkv_weight: arena.address(layer.attention.qkv_weight)?.cast_const(),
        output_weight: arena.address(layer.attention.output_weight)?.cast_const(),
        query_norm: arena.address(layer.attention.query_norm)?.cast_const(),
        key_norm: arena.address(layer.attention.key_norm)?.cast_const(),

        router_weight: arena.address(layer.moe.router_weight)?.cast_const(),
        shared_gate_weight: arena.address(layer.moe.shared_gate_weight)?.cast_const(),
        shared_up_weight: arena.address(layer.moe.shared_up_weight)?.cast_const(),
        shared_down_weight: arena.address(layer.moe.shared_down_weight)?.cast_const(),
        shared_gate_logit_weight: arena
            .address(layer.moe.shared_gate_logit_weight)?
            .cast_const(),

        mixer_norm: arena.address(mixer.norm)?.cast_const(),
        mixer_down: arena.address(mixer.down)?.cast_const(),
        mixer_up: arena.address(mixer.up)?.cast_const(),

        // One layer, so the item id is the expert id and the table needs no view.
        slot_table: pool.table_address()? as *const u32,
        slot_pool: pool.slot_address(0)? as *const u8,
    })
}

fn bind_pointers(
    arena: &DeviceArena,
    kv_arena: &DeviceArena,
    target: &Qwen38FlashNextResidentModel,
    layout: &Qwen38FlashNextMtpLayout,
) -> EngineResult<DraftPointers> {
    let workspace: Qwen38FlashNextMtpWorkspace = layout.workspace();
    let kv: Qwen38FlashNextMtpKvPlanes = layout.kv_planes();
    let (target_kv_arena, block_tables) = target.block_tables();

    Ok(DraftPointers {
        residual_a: arena.address(workspace.residual_a)?,
        residual_b: arena.address(workspace.residual_b)?,

        fusion_hidden: arena.address(workspace.fusion_hidden)?,
        fusion_projected: arena.address(workspace.fusion_projected)?,

        hc_normalized: arena.address(workspace.hc_normalized)?,
        hc_low_rank: arena.address(workspace.hc_low_rank)?,
        hc_mixed: arena.address(workspace.hc_mixed)?,
        hc_write_gate: arena.address(workspace.hc_write_gate)?,

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

        mixer_normalized: arena.address(workspace.mixer_normalized)?,
        mixer_low_rank: arena.address(workspace.mixer_low_rank)?,
        mixer_mixed: arena.address(workspace.mixer_mixed)?,
        logits: arena.address(workspace.logits)?,

        table_rows: arena.address(workspace.table_rows)?.cast_const(),
        cache_positions: arena.address(workspace.cache_positions)?.cast_const(),
        lengths: arena.address(workspace.lengths)?.cast_const(),
        rope_cos: arena.address(workspace.rope_cos)?.cast_const(),
        rope_sin: arena.address(workspace.rope_sin)?.cast_const(),
        // Draft K/V uses the target's physical-page mapping.
        block_tables: target_kv_arena.address(block_tables)?.cast_const(),

        key_pages: kv_arena.address(kv.key_pages)?,
        value_pages: kv_arena.address(kv.value_pages)?,
    })
}

/// Runs input fusion over sealed, non-overlapping arena planes.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fusion(
    stream: &CudaStream,
    arena: &DeviceArena,
    rows: usize,
    ops: Ops<'_>,
    fusion: &Qwen38FlashNextMtpFusionProjectionOp,
    weights: DraftWeights,
    pointers: DraftPointers,
    embedding_rows: tuisko_gpu::ArenaRegion<u16>,
    residual_a: tuisko_gpu::ArenaRegion<u16>,
) -> GpuResult<()> {
    // SAFETY: the caller's contract.
    unsafe {
        // Normalize the target stream before residual_a is reused.
        ops.hyper.launch_grouped_norm(
            stream,
            rows,
            pointers.residual_b.cast_const(),
            weights.norm_hidden,
            pointers.fusion_hidden,
        )?;
        // Normalize four copies of each embedding row in place.
        widen_embedding(stream, arena, rows, embedding_rows, residual_a)?;
        ops.hyper.launch_grouped_norm(
            stream,
            rows,
            pointers.residual_a.cast_const(),
            weights.norm_embedding,
            pointers.residual_a,
        )?;
        fusion.launch(
            stream,
            rows,
            pointers.residual_a.cast_const(),
            weights.fc_embedding,
            pointers.fusion_projected,
        )?;
        fusion.launch(
            stream,
            rows,
            pointers.fusion_hidden.cast_const(),
            weights.fc_hidden,
            pointers.residual_a,
        )?;
        ops.engram.launch_inject(
            stream,
            rows,
            pointers.residual_a.cast_const(),
            pointers.fusion_projected.cast_const(),
            pointers.residual_a,
        )
    }
}

/// Copies each embedding row into all hyper-connection branches.
unsafe fn widen_embedding(
    stream: &CudaStream,
    arena: &DeviceArena,
    rows: usize,
    embedding_rows: tuisko_gpu::ArenaRegion<u16>,
    residual_a: tuisko_gpu::ArenaRegion<u16>,
) -> GpuResult<()> {
    for row in 0..rows {
        for branch in 0..A::HC_COUNT {
            // SAFETY: the caller provides distinct sealed regions.
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

/// Launches one captured draft stage over sealed program storage.
#[allow(clippy::too_many_arguments)]
fn launch_draft_segment(
    stream: &CudaStream,
    arena: &DeviceArena,
    stage: Qwen38FlashNextMtpStage,
    rows: usize,
    ops: Ops<'_>,
    fusion: &Qwen38FlashNextMtpFusionProjectionOp,
    weights: DraftWeights,
    pointers: DraftPointers,
    embedding_rows: tuisko_gpu::ArenaRegion<u16>,
    residual_a: tuisko_gpu::ArenaRegion<u16>,
) -> GpuResult<()> {
    // SAFETY: the caller's contract.
    unsafe {
        if stage == Qwen38FlashNextMtpStage::Tail {
            launch_draft_experts(stream, rows, ops, weights, pointers)?;
            ops.hyper.launch_write_back(
                stream,
                rows,
                pointers.residual_b.cast_const(),
                pointers.block_output.cast_const(),
                pointers.hc_write_gate.cast_const(),
                pointers.residual_a,
            )?;
            // Publish the collapsed hidden state consumed by the shared head.
            return ops.hyper.launch_final_mix(
                stream,
                rows,
                pointers.residual_a.cast_const(),
                weights.mixer_norm,
                weights.mixer_down,
                weights.mixer_up,
                pointers.mixer_normalized,
                pointers.mixer_low_rank,
                pointers.mixer_mixed,
            );
        }

        launch_fusion(
            stream,
            arena,
            rows,
            ops,
            fusion,
            weights,
            pointers,
            embedding_rows,
            residual_a,
        )?;

        ops.hyper.launch_input_mix(
            stream,
            rows,
            pointers.residual_a.cast_const(),
            weights.attention_hc_norm,
            weights.attention_hc_down,
            weights.attention_hc_up,
            weights.attention_hc_inject,
            pointers.hc_normalized,
            pointers.hc_low_rank,
            pointers.hc_mixed,
            pointers.hc_write_gate,
        )?;

        ops.qsa_qkv.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            weights.qkv_weight,
            pointers.qkv,
        )?;
        ops.qsa_prepare.launch(
            stream,
            rows,
            pointers.qkv.cast_const(),
            weights.query_norm,
            weights.key_norm,
            pointers.rope_cos,
            pointers.rope_sin,
            pointers.block_tables,
            pointers.table_rows,
            crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            KEY_CACHE_SCALE,
            VALUE_CACHE_SCALE,
        )?;
        if stage == Qwen38FlashNextMtpStage::Prime {
            return Ok(());
        }
        ops.qsa_attention.launch(
            stream,
            rows,
            pointers.query.cast_const(),
            pointers.key_pages.cast_const(),
            pointers.value_pages.cast_const(),
            pointers.block_tables,
            pointers.table_rows,
            crate::QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
            pointers.lengths,
            pointers.attention,
            KEY_CACHE_SCALE,
            VALUE_CACHE_SCALE,
        )?;
        ops.qsa_gate.launch(
            stream,
            rows,
            pointers.attention,
            pointers.qkv.cast_const(),
            pointers.attention_gated,
        )?;
        ops.block_output.launch(
            stream,
            rows,
            pointers.attention_gated.cast_const(),
            weights.output_weight,
            pointers.block_output,
        )?;
        ops.hyper.launch_write_back(
            stream,
            rows,
            pointers.residual_a.cast_const(),
            pointers.block_output.cast_const(),
            pointers.hc_write_gate.cast_const(),
            pointers.residual_b,
        )?;

        ops.hyper.launch_input_mix(
            stream,
            rows,
            pointers.residual_b.cast_const(),
            weights.mlp_hc_norm,
            weights.mlp_hc_down,
            weights.mlp_hc_up,
            weights.mlp_hc_inject,
            pointers.hc_normalized,
            pointers.hc_low_rank,
            pointers.hc_mixed,
            pointers.hc_write_gate,
        )?;
        ops.router.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            weights.router_weight,
            pointers.router_logits,
            pointers.expert_indices,
            pointers.routing_weights,
        )
    }
}

/// Runs experts after the caller has admitted the full routed round.
unsafe fn launch_draft_experts(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    weights: DraftWeights,
    pointers: DraftPointers,
) -> GpuResult<()> {
    // SAFETY: the caller's contract.
    unsafe {
        ops.experts.launch_draft(
            stream,
            rows,
            &Qwen38FlashNextMtpExpertDispatch {
                input: pointers.hc_mixed.cast_const(),
                expert_indices: pointers.expert_indices.cast_const(),
                routing_weights: pointers.routing_weights.cast_const(),
                slot_table: weights.slot_table,
                slot_pool: weights.slot_pool,
                shared_gate_weight: weights.shared_gate_weight,
                shared_up_weight: weights.shared_up_weight,
                shared_down_weight: weights.shared_down_weight,
                shared_gate_logit_weight: weights.shared_gate_logit_weight,
                routed_intermediate: pointers.routed_intermediate,
                routed_output: pointers.routed_output,
                shared_intermediate: pointers.shared_intermediate,
                shared_output: pointers.shared_output,
                shared_gate_logit: pointers.shared_gate_logit,
                output: pointers.block_output,
            },
        )
    }
}

/// Runs the shared LM head over the draft's collapsed state.
fn launch_draft_head(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    lm_head: *const u16,
    pointers: DraftPointers,
) -> GpuResult<()> {
    // SAFETY: the caller's contract.
    unsafe {
        ops.lm_head.launch(
            stream,
            rows,
            pointers.mixer_mixed.cast_const(),
            lm_head,
            pointers.logits,
        )
    }
}

fn kv_append_bytes(rows: usize) -> usize {
    let cache_row = <A as Arch>::NUM_KV_HEADS * <A as Arch>::HEAD_DIM;

    rows * 2 * cache_row
}

impl Qwen38FlashNextMtpProgram {
    /// Uploads resident draft weights and stages the expert pool.
    fn upload(
        &mut self,
        stream: &CudaStream,
        snapshot: &Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        progress: Option<&ResidentLoadProgress>,
    ) -> EngineResult<()> {
        let started = Instant::now();
        let block = Qwen38FlashNextMtpBindings::bind(snapshot.as_ref())?.materialize()?;
        let fusion = self.layout.fusion();
        let regions = self.layout.layer();
        let mixer = self.layout.mixer();

        let embedding_gain = block.pre_fc_norm_embedding.words().collect::<Vec<_>>();
        let mut replicated = Vec::with_capacity(A::HC_WIDTH);
        for _ in 0..A::HC_COUNT {
            replicated.extend_from_slice(&embedding_gain);
        }
        self.arena
            .copy_from_host(stream, fusion.norm_embedding, &replicated)?;
        self.arena.copy_from_host(
            stream,
            fusion.norm_hidden,
            &block.pre_fc_norm_hidden.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            fusion.fc_embedding,
            &block.fc_embedding.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            fusion.fc_hidden,
            &block.fc_hidden.words().collect::<Vec<_>>(),
        )?;

        let layer = block.layers.into_iter().next().ok_or_else(|| {
            EngineError::layout("the Flash-Next draft block bound no layer to upload")
        })?;
        upload_hyper_connection(
            &self.arena,
            stream,
            regions.attention_hc,
            layer.attention_hyper_connection,
        )?;
        upload_hyper_connection(
            &self.arena,
            stream,
            regions.mlp_hc,
            layer.mlp_hyper_connection,
        )?;

        let attention = layer.attention;
        self.arena.copy_from_host(
            stream,
            regions.attention.qkv_weight,
            &bf16_words(&attention.qkv_weight_bf16)?,
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.output_weight,
            &attention.output_weight.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.query_norm,
            &attention.query_norm.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.key_norm,
            &attention.key_norm.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.indexer_qk_weight,
            &attention.indexer.qk_weight.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.indexer_query_norm,
            &attention.indexer.query_norm.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.attention.indexer_key_norm,
            &attention.indexer.key_norm.words().collect::<Vec<_>>(),
        )?;

        // BF16 draft experts have no ModelOpt weight_scale_2 plane.
        let moe = layer.mlp;
        self.arena.copy_from_host(
            stream,
            regions.moe.router_weight,
            &moe.router_weight.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.moe.shared_gate_weight,
            &moe.shared_expert
                .gate_proj_weight
                .words()
                .collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.moe.shared_up_weight,
            &moe.shared_expert.up_proj_weight.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.moe.shared_down_weight,
            &moe.shared_expert
                .down_proj_weight
                .words()
                .collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            regions.moe.shared_gate_logit_weight,
            &moe.shared_expert.gate_weight.words().collect::<Vec<_>>(),
        )?;

        self.arena.copy_from_host(
            stream,
            mixer.norm,
            &block.mixer.hc_norm.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            mixer.down,
            &block.mixer.input_mix_down.words().collect::<Vec<_>>(),
        )?;
        self.arena.copy_from_host(
            stream,
            mixer.up,
            &block.mixer.input_mix_up.words().collect::<Vec<_>>(),
        )?;

        // Pool item IDs are expert IDs; require publishes the device table.
        let pool = &moe.experts;
        let mut staged_bytes = 0usize;
        for expert in 0..pool.expert_count {
            let begin = product(
                "Flash-Next draft down offset",
                expert,
                pool.down_stride_bytes,
            )?;
            let secondary = pool
                .down
                .get(begin..begin + pool.down_stride_bytes)
                .ok_or_else(|| {
                    EngineError::layout(format!(
                        "Flash-Next draft expert {expert} names a down extent outside its plane"
                    ))
                })?;
            self.pool.stage_item(expert, &[], secondary)?;
            staged_bytes += secondary.len();
        }
        stream.synchronize().map_err(GpuError::from)?;
        if let Some(progress) = progress {
            progress.submit(self.layout.resident_weight_bytes())?;
            progress.finish_upload()?;
        }

        self.load_stats.upload = started.elapsed();
        self.load_stats.staged_bytes = staged_bytes;

        Ok(())
    }

    /// Captures the prime ladder, one routed pair, and the proposal head.
    fn capture(&mut self, stream: &CudaStream) -> EngineResult<()> {
        let started = Instant::now();
        let arena = &self.arena;
        let ops = self.target.ops();
        let fusion = &self._fusion;
        let weights = self.weights;
        let pointers = self.pointers;
        let workspace = self.layout.workspace();

        let capture = |stage, rows| {
            CudaGraph::capture(stream, || {
                launch_draft_segment(
                    stream,
                    arena,
                    stage,
                    rows,
                    ops,
                    fusion,
                    weights,
                    pointers,
                    workspace.embedding_rows,
                    workspace.residual_a,
                )
            })
        };

        let mut primes = Vec::with_capacity(QWEN38_FLASH_NEXT_MTP_ROUTES.len());
        for rows in QWEN38_FLASH_NEXT_MTP_ROUTES {
            primes.push(capture(Qwen38FlashNextMtpStage::Prime, rows)?);
        }
        let segments = vec![capture(
            Qwen38FlashNextMtpStage::Head,
            QWEN38_FLASH_NEXT_PROPOSAL_ROWS,
        )?];
        let tails = vec![capture(
            Qwen38FlashNextMtpStage::Tail,
            QWEN38_FLASH_NEXT_PROPOSAL_ROWS,
        )?];

        let lm_head = self
            .target
            .resident_arena()
            .address(self.target.layout().endpoint().lm_head)?
            .cast_const();
        let head = vec![CudaGraph::capture(stream, || {
            launch_draft_head(
                stream,
                QWEN38_FLASH_NEXT_PROPOSAL_ROWS,
                ops,
                lm_head,
                pointers,
            )
        })?];

        self.primes = primes;
        self.segments = segments;
        self.tail_segments = tails;
        self.head = head;
        self.load_stats.capture = started.elapsed();
        self.load_stats.executables = self.executables();

        Ok(())
    }
}

impl Qwen38FlashNextMtpProgram {
    /// Stages the stream row preceding each drafted token.
    fn stage_stream(
        &self,
        stream: &CudaStream,
        rows: usize,
        source: Qwen38FlashNextMtpStream,
    ) -> EngineResult<()> {
        let workspace = self.layout.workspace();
        let (target_arena, published) = self.target.published_stream();
        let width = A::HC_WIDTH;

        // SAFETY: copies use distinct sealed regions and checked extents.
        unsafe {
            match source {
                Qwen38FlashNextMtpStream::Carry => self.arena.copy_slice_from_arena_async(
                    stream,
                    workspace.residual_b,
                    0,
                    &self.arena,
                    workspace.stream_carry,
                    0,
                    width,
                )?,
                Qwen38FlashNextMtpStream::TargetRow(row) => {
                    self.arena.copy_slice_from_arena_async(
                        stream,
                        workspace.residual_b,
                        0,
                        target_arena,
                        published,
                        product("Flash-Next draft stream row", row, width)?,
                        product("Flash-Next draft stream extent", rows, width)?,
                    )?
                }
                Qwen38FlashNextMtpStream::Draft => self.arena.copy_slice_from_arena_async(
                    stream,
                    workspace.residual_b,
                    0,
                    &self.arena,
                    workspace.residual_a,
                    0,
                    product("Flash-Next draft stream extent", rows, width)?,
                )?,
                Qwen38FlashNextMtpStream::ShiftedTile => {
                    self.arena.copy_slice_from_arena_async(
                        stream,
                        workspace.residual_b,
                        0,
                        &self.arena,
                        workspace.stream_carry,
                        0,
                        width,
                    )?;
                    if rows > 1 {
                        self.arena.copy_slice_from_arena_async(
                            stream,
                            workspace.residual_b,
                            width,
                            target_arena,
                            published,
                            0,
                            product("Flash-Next draft shifted extent", rows - 1, width)?,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Retains a target stream row for the next draft round.
    pub fn carry_target_row(&self, stream: &CudaStream, row: usize) -> EngineResult<()> {
        let workspace = self.layout.workspace();
        let (target_arena, published) = self.target.published_stream();

        // SAFETY: two sealed arenas, distinct regions, one ordered stream.
        unsafe {
            self.arena.copy_slice_from_arena_async(
                stream,
                workspace.stream_carry,
                0,
                target_arena,
                published,
                product("Flash-Next draft carry row", row, A::HC_WIDTH)?,
                A::HC_WIDTH,
            )?;
        }

        Ok(())
    }

    /// Gathers one embedding row per token out of the shared table into the draft's own plane.
    fn stage_embeddings(&mut self, stream: &CudaStream, tokens: &[u32]) -> EngineResult<usize> {
        let snapshot = Arc::clone(self.target.snapshot());
        let embedding = Qwen38FlashNextTextEndpointBindings::bind_embedding(snapshot.as_ref())?;
        let source = embedding.bytes();
        let width = <A as Arch>::HIDDEN;
        for (row, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            if token >= <A as Arch>::VOCAB {
                return Err(EngineError::route(format!(
                    "Flash-Next draft token {token} is outside 0..{}",
                    <A as Arch>::VOCAB
                )));
            }
            let begin = token * width * size_of::<u16>();
            let row_bytes = source
                .get(begin..begin + width * size_of::<u16>())
                .ok_or_else(|| {
                    EngineError::layout("Flash-Next embedding row falls outside the mapping")
                })?;
            let destination = &mut self.embedding_stager[row * width..(row + 1) * width];
            for (word, bytes) in destination.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *word = u16::from_le_bytes([bytes[0], bytes[1]]);
            }
        }

        let values = product("Flash-Next draft staged embedding", tokens.len(), width)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.workspace().embedding_rows,
            &self.embedding_stager[..values],
        )?;

        Ok(values * size_of::<u16>())
    }

    /// Stages dense-attention inputs for one draft round.
    fn stage_round_inputs(
        &self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        positions: &[u32],
    ) -> EngineResult<()> {
        let workspace = self.layout.workspace();
        let table_rows = vec![slot as u32; rows];
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

    /// Runs the routed head, expert round, tail, and optional LM head.
    fn forward(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        propose: bool,
    ) -> EngineResult<Qwen38FlashNextLayerStreamTelemetry> {
        if let Some(reason) = self.pool.poisoned() {
            return Err(EngineError::layout(format!(
                "the Flash-Next draft expert cache is poisoned and refuses every round: {reason}"
            )));
        }
        if rows != QWEN38_FLASH_NEXT_PROPOSAL_ROWS {
            return Err(EngineError::route(format!(
                "the Flash-Next draft routed half admits {QWEN38_FLASH_NEXT_PROPOSAL_ROWS} row, \
                 not {rows}; its pool ceiling is {QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS} rows"
            )));
        }
        let head = self.segments.first().ok_or_else(|| {
            EngineError::route("the Flash-Next draft block captured no routed head")
        })?;
        // SAFETY: the program owns all captured storage until graph drop.
        unsafe { head.launch(stream) }?;

        let requests = self.read_expert_round(stream, rows)?;
        let round = self.pool.require(&self.round_items)?;
        self.pool.fence_replay(stream)?;
        let tail = self.tail_segments.first().ok_or_else(|| {
            EngineError::route("the Flash-Next draft block captured no routed tail")
        })?;
        // SAFETY: as above.
        unsafe { tail.launch(stream) }?;
        self.pool.record_replay_release(stream)?;

        if propose {
            let head = self.head.first().ok_or_else(|| {
                EngineError::route("the Flash-Next draft block captured no proposal head")
            })?;
            // SAFETY: as above.
            unsafe { head.launch(stream) }?;
        }
        stream.synchronize().map_err(GpuError::from)?;

        Ok(layer_telemetry(0, requests, round))
    }

    fn forward_prime(
        &mut self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Qwen38FlashNextLayerStreamTelemetry> {
        let index = draft_route_index(rows)?;
        let prime = self.primes.get(index).ok_or_else(|| {
            EngineError::route(format!(
                "the Flash-Next draft block has no captured prime for {rows} rows"
            ))
        })?;
        // SAFETY: the program owns all captured storage until its graphs are dropped.
        unsafe { prime.launch(stream) }?;
        stream.synchronize().map_err(GpuError::from)?;

        Ok(layer_telemetry(0, 0, StreamingRound::default()))
    }

    /// Converts router output to unique expert-pool item IDs.
    fn read_expert_round(&mut self, stream: &CudaStream, rows: usize) -> EngineResult<usize> {
        let selections = product(
            "Flash-Next draft round selections",
            rows,
            A::NUM_EXPERTS_PER_TOKEN,
        )?;
        if self.expert_readback.len() < selections {
            self.expert_readback.resize(selections, 0);
        }
        self.arena.copy_prefix_to_host_slice(
            stream,
            self.layout.workspace().expert_indices,
            &mut self.expert_readback[..selections],
        )?;

        self.round_items.clear();
        for &expert in &self.expert_readback[..selections] {
            let expert = expert as usize;
            if expert >= A::NUM_EXPERTS {
                return Err(EngineError::layout(format!(
                    "the Flash-Next draft router published expert {expert}, outside \
                     0..{}",
                    A::NUM_EXPERTS
                )));
            }
            let item = expert as u32;
            if !self.round_items.contains(&item) {
                self.round_items.push(item);
            }
        }

        Ok(selections)
    }

    /// Requires the shared page mapping to cover the speculative window.
    fn admit_draft_round(&self, slot: usize, visible: usize) -> EngineResult<()> {
        let mapped = self.target.mapped_tokens(slot)?;
        if visible > mapped {
            return Err(EngineError::route(format!(
                "a Flash-Next draft row on slot {slot} reaches visible length {visible} against \
                 {mapped} mapped tokens; a speculative round must reserve the draft window ahead \
                 of the tokens it proposes rather than attend through an unmapped page"
            )));
        }

        Ok(())
    }

    /// Advances one draft token and publishes its next-token proposal.
    pub fn draft_step(
        &mut self,
        stream: &CudaStream,
        token: u32,
        position: u32,
        slot: usize,
        source: Qwen38FlashNextMtpStream,
    ) -> EngineResult<Qwen38FlashNextMtpStepTelemetry> {
        self.run_rows(stream, &[token], position, slot, source, true)
    }

    /// One drafted position with no proposal: the mirror advances, the head does not run.
    pub fn draft_extend(
        &mut self,
        stream: &CudaStream,
        token: u32,
        position: u32,
        slot: usize,
        source: Qwen38FlashNextMtpStream,
    ) -> EngineResult<Qwen38FlashNextMtpStepTelemetry> {
        self.run_rows(stream, &[token], position, slot, source, false)
    }

    /// One prompt tile of the draft's mirror, beside the target's tile of the same positions.
    pub fn prime_tile(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        first_position: u32,
        slot: usize,
    ) -> EngineResult<Qwen38FlashNextMtpStepTelemetry> {
        if !QWEN38_FLASH_NEXT_PREFILL_ROWS.contains(&tokens.len()) {
            return Err(EngineError::route(format!(
                "Flash-Next draft prime tile {} is not an admitted T=32/64/128/1024 route",
                tokens.len()
            )));
        }
        self.run_rows(
            stream,
            tokens,
            first_position,
            slot,
            Qwen38FlashNextMtpStream::ShiftedTile,
            false,
        )
    }

    fn run_rows(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        first_position: u32,
        slot: usize,
        source: Qwen38FlashNextMtpStream,
        propose: bool,
    ) -> EngineResult<Qwen38FlashNextMtpStepTelemetry> {
        let rows = tokens.len();
        draft_route_index(rows)?;
        let last = first_position as usize + rows;
        require_qwen38_flash_next_dense_qsa_visible(last)?;
        self.admit_draft_round(slot, last)?;

        let positions = (0..rows as u32)
            .map(|offset| first_position + offset)
            .collect::<Vec<_>>();

        let started = Instant::now();
        let embedding_h2d_bytes = self.stage_embeddings(stream, tokens)?;
        self.stage_stream(stream, rows, source)?;
        self.stage_round_inputs(stream, rows, slot, &positions)?;
        let layer = if propose {
            self.forward(stream, rows, true)?
        } else {
            self.forward_prime(stream, rows)?
        };
        let forward = started.elapsed();

        Ok(Qwen38FlashNextMtpStepTelemetry {
            rows,
            layer,
            embedding_h2d_bytes,
            kv_append_bytes: kv_append_bytes(rows),
            forward,
            proposed: propose,
        })
    }

    /// The proposal row the last `draft_step` published, in the head's own BF16 bits.
    pub fn read_proposal(&mut self, stream: &CudaStream) -> EngineResult<&[u16]> {
        let vocab = <A as Arch>::VOCAB;
        self.arena.copy_prefix_to_host_slice(
            stream,
            self.layout.workspace().logits,
            &mut self.logit_bank[..vocab],
        )?;

        Ok(&self.logit_bank[..vocab])
    }

    /// Clears the carry inherited by a new request.
    pub fn reset_carry(&self, stream: &CudaStream) -> EngineResult<()> {
        self.arena
            .fill(stream, self.layout.workspace().stream_carry, 0)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS, QWEN38_FLASH_NEXT_MTP_ROUTES,
        QWEN38_FLASH_NEXT_MTP_SEGMENTS, QWEN38_FLASH_NEXT_PROPOSAL_ROWS, draft_route_index,
        kv_append_bytes,
    };
    use crate::qwen38_flash_next::layer_route::{
        QWEN38_FLASH_NEXT_PREFILL_ROWS, qwen38_flash_next_row_route,
    };
    use crate::{EngineErrorCode, MAX_BATCH};

    #[test]
    fn the_draft_captures_the_fusions_schedule_and_not_the_backbones() {
        assert_eq!(QWEN38_FLASH_NEXT_MTP_ROUTES, [1, 32, 64, 128, 1_024]);
        for (index, rows) in QWEN38_FLASH_NEXT_MTP_ROUTES.iter().enumerate() {
            assert_eq!(draft_route_index(*rows).unwrap(), index);
            qwen38_flash_next_row_route(*rows).unwrap();
        }

        for rows in 2..=MAX_BATCH {
            let error = draft_route_index(rows).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
            assert!(error.to_string().contains("replays the one-row route"));
        }
        assert_eq!(
            QWEN38_FLASH_NEXT_MTP_ROUTES.len(),
            1 + QWEN38_FLASH_NEXT_PREFILL_ROWS.len()
        );
    }

    #[test]
    fn the_draft_inventory_is_the_prime_ladder_plus_one_routed_pair() {
        let executables = QWEN38_FLASH_NEXT_MTP_ROUTES.len() + QWEN38_FLASH_NEXT_MTP_SEGMENTS + 1;

        assert_eq!(QWEN38_FLASH_NEXT_MTP_SEGMENTS, 2);
        assert_eq!(executables, 8);
    }

    #[test]
    fn the_routed_width_follows_the_pool_capacity() {
        assert_eq!(QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS, 12);
        assert_eq!(QWEN38_FLASH_NEXT_PROPOSAL_ROWS, 1);
        const {
            assert!(QWEN38_FLASH_NEXT_PROPOSAL_ROWS <= QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS);
        }
        assert_eq!(
            QWEN38_FLASH_NEXT_MTP_ROUTES
                .iter()
                .filter(|rows| **rows <= QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS)
                .count(),
            1
        );
    }

    #[test]
    fn the_kv_append_figure_counts_dense_cache_writes() {
        let cache_row = 2 * 2 * 256;

        assert_eq!(kv_append_bytes(1), cache_row);
        assert_eq!(kv_append_bytes(32), 32 * cache_row);
    }
}

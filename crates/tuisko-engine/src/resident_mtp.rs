//! Resident long-context ownership for the exact Qwen3.8 MTP layer.

use crate::resident_mtp_layout::{MTP_PROMPT_ROWS, ResidentMtpCacheRegions, ResidentMtpRegions};
use crate::{
    EngineError, EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, PagedKvTableUpdate,
    ResidentModelProgram, ResidentMtpLayout,
};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, LmHeadOp, MtpBf16AttentionOutputOp, MtpBf16FusionOp, MtpBf16MlpOp,
    MtpBf16PagedGqaOp, MtpBf16QkPrepareOp, MtpBf16QkvOp, ResidualNormOp,
};
use tuisko_model::{Arch, CheckpointSnapshot, MtpBindings, Qwen38_27B, TextEndpointBindings};

const ROTARY_PAIRS: usize = 32;
const PROMPT_ROUTES: [usize; 5] = [1, 32, 64, 128, MTP_PROMPT_ROWS];
const REALIGN_ROUTES: usize = 4;

/// Exact resident prompt-prime graph selected by a checked staging call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the resident MTP prompt route must be replayed with its staged inputs"]
pub struct ResidentMtpPromptRoute {
    rows: usize,
    slot: usize,
    first_position: usize,
}

impl ResidentMtpPromptRoute {
    /// Exact aligned prompt rows appended to the MTP cache.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Shared target/MTP page-table row.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// Absolute cache position of the first row.
    pub const fn first_position(self) -> usize {
        self.first_position
    }
}

/// Exact compact draft graph selected by a checked staging call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the resident MTP draft route must be replayed with its staged inputs"]
pub struct ResidentMtpDraftRoute {
    batch: usize,
}

impl ResidentMtpDraftRoute {
    /// Number of distinct active slots in compact row order.
    pub const fn batch(self) -> usize {
        self.batch
    }

    #[cfg(feature = "qualification")]
    /// Constructs one exact route for qualification graph lookup.
    pub fn qualified(batch: usize) -> EngineResult<Self> {
        require_batch(batch)?;
        Ok(Self { batch })
    }
}

/// Exact causal MTP realignment graph selected by a checked staging call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the resident MTP realignment route must be replayed with its staged inputs"]
pub struct ResidentMtpRealignRoute {
    tokens: usize,
    slot: usize,
    first_position: usize,
}

impl ResidentMtpRealignRoute {
    /// Number of contiguous target-conditioned rows.
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Shared target/MTP page-table row.
    pub const fn slot(self) -> usize {
        self.slot
    }

    /// Absolute cache position of the first row.
    pub const fn first_position(self) -> usize {
        self.first_position
    }
}

struct Graphs {
    prompt: [CudaGraph; PROMPT_ROUTES.len()],
    draft: [CudaGraph; MAX_BATCH],
    continue_draft: [CudaGraph; MAX_BATCH],
    staged_continue_draft: [CudaGraph; MAX_BATCH],
    prime: [CudaGraph; REALIGN_ROUTES],
    realign: [CudaGraph; REALIGN_ROUTES],
}

/// Target plus one source-native MTP weight set and one shared-lifecycle cache mirror.
pub struct ResidentMtpProgram {
    // Graphs retain every local/target device address, pinned source, and module handle.
    graphs: Graphs,
    arena: DeviceArena,
    cache_arena: DeviceArena,
    _fusion: MtpBf16FusionOp,
    _norm: ResidualNormOp<Qwen38_27B>,
    _qkv: MtpBf16QkvOp,
    _qk_prepare: MtpBf16QkPrepareOp,
    _paged_gqa: MtpBf16PagedGqaOp,
    _attention_output: MtpBf16AttentionOutputOp,
    _mlp: MtpBf16MlpOp,
    embedding_stager: PinnedHostBuffer<u16>,
    table_rows_stager: PinnedHostBuffer<u32>,
    positions_stager: PinnedHostBuffer<u32>,
    lengths_stager: PinnedHostBuffer<u32>,
    rope_cos_stager: PinnedHostBuffer<f32>,
    rope_sin_stager: PinnedHostBuffer<f32>,
    continuation_hidden_stager: PinnedHostBuffer<u16>,
    target: ResidentModelProgram,
    context: Arc<CudaContext>,
    layout: ResidentMtpLayout,
    base_address: u64,
    cache_base_address: u64,
}

#[derive(Clone, Copy)]
struct Pointers {
    embedding: *const u16,
    target_hidden: *const u16,
    embedding_norm: *const u16,
    hidden_norm: *const u16,
    normalized_embedding: *mut u16,
    normalized_hidden: *mut u16,
    input_projection: *const u16,
    residual: *mut u16,
    input_norm: *const u16,
    attention_normalized: *mut u16,
    qkv_weight: *const u16,
    qkv: *mut u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    query: *mut f32,
    key_pages: *mut u16,
    value_pages: *mut u16,
    attention: *mut f32,
    attention_activation: *mut u16,
    attention_output_weight: *const u16,
    attention_branch: *mut u16,
    post_attention_norm: *const u16,
    post_attention_residual: *mut u16,
    mlp_normalized: *mut u16,
    gate_up_weight: *const u16,
    swiglu: *mut u16,
    down_weight: *const u16,
    mlp_branch: *mut u16,
    final_norm: *const u16,
    residual_output: *mut u16,
    final_normalized: *mut u16,
    lm_head_activation_codes: *mut u8,
    lm_head_activation_scales: *mut f32,
    lm_head_codes: *const u8,
    lm_head_scales: *const u16,
    logits: *mut u16,
}

impl Pointers {
    fn bind(
        arena: &DeviceArena,
        cache_arena: &DeviceArena,
        regions: ResidentMtpRegions,
        cache: ResidentMtpCacheRegions,
        target: &ResidentModelProgram,
    ) -> GpuResult<Self> {
        let (lm_head_codes, lm_head_scales) = target.mtp_lm_head_weights();
        Ok(Self {
            embedding: arena.address(regions.embedding)?.cast_const(),
            target_hidden: arena.address(regions.target_hidden)?.cast_const(),
            embedding_norm: arena.address(regions.embedding_norm)?.cast_const(),
            hidden_norm: arena.address(regions.hidden_norm)?.cast_const(),
            normalized_embedding: arena.address(regions.normalized_embedding)?,
            normalized_hidden: arena.address(regions.normalized_hidden)?,
            input_projection: arena.address(regions.input_projection)?.cast_const(),
            residual: arena.address(regions.residual)?,
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            attention_normalized: arena.address(regions.attention_normalized)?,
            qkv_weight: arena.address(regions.qkv_weight)?.cast_const(),
            qkv: arena.address(regions.qkv)?,
            query_norm: arena.address(regions.query_norm)?.cast_const(),
            key_norm: arena.address(regions.key_norm)?.cast_const(),
            rope_cos: arena.address(regions.rope_cos)?.cast_const(),
            rope_sin: arena.address(regions.rope_sin)?.cast_const(),
            block_tables: arena.address(regions.block_tables)?.cast_const(),
            table_rows: arena.address(regions.table_rows)?.cast_const(),
            cache_positions: arena.address(regions.cache_positions)?.cast_const(),
            lengths: arena.address(regions.lengths)?.cast_const(),
            query: arena.address(regions.query)?,
            key_pages: cache_arena.address(cache.key_pages)?,
            value_pages: cache_arena.address(cache.value_pages)?,
            attention: arena.address(regions.attention)?,
            attention_activation: arena.address(regions.attention_activation)?,
            attention_output_weight: arena.address(regions.attention_output_weight)?.cast_const(),
            attention_branch: arena.address(regions.attention_branch)?,
            post_attention_norm: arena.address(regions.post_attention_norm)?.cast_const(),
            post_attention_residual: arena.address(regions.post_attention_residual)?,
            mlp_normalized: arena.address(regions.mlp_normalized)?,
            gate_up_weight: arena.address(regions.gate_up_weight)?.cast_const(),
            swiglu: arena.address(regions.swiglu)?,
            down_weight: arena.address(regions.down_weight)?.cast_const(),
            mlp_branch: arena.address(regions.mlp_branch)?,
            final_norm: arena.address(regions.final_norm)?.cast_const(),
            residual_output: arena.address(regions.residual_output)?,
            final_normalized: arena.address(regions.final_normalized)?,
            lm_head_activation_codes: arena.address(regions.lm_head_activation_codes)?,
            lm_head_activation_scales: arena.address(regions.lm_head_activation_scales)?,
            lm_head_codes,
            lm_head_scales,
            logits: arena.address(regions.logits)?,
        })
    }

    fn offset_rows(self, rows: usize) -> Self {
        Self {
            embedding: self.embedding.wrapping_add(rows * Qwen38_27B::HIDDEN),
            target_hidden: self.target_hidden.wrapping_add(rows * Qwen38_27B::HIDDEN),
            normalized_embedding: self
                .normalized_embedding
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            normalized_hidden: self
                .normalized_hidden
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            residual: self.residual.wrapping_add(rows * Qwen38_27B::HIDDEN),
            attention_normalized: self
                .attention_normalized
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            qkv: self.qkv.wrapping_add(rows * Qwen38_27B::ATTENTION_QKV_ROWS),
            rope_cos: self.rope_cos.wrapping_add(rows * ROTARY_PAIRS),
            rope_sin: self.rope_sin.wrapping_add(rows * ROTARY_PAIRS),
            table_rows: self.table_rows.wrapping_add(rows),
            cache_positions: self.cache_positions.wrapping_add(rows),
            lengths: self.lengths.wrapping_add(rows),
            query: self
                .query
                .wrapping_add(rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS),
            attention: self
                .attention
                .wrapping_add(rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS),
            attention_activation: self
                .attention_activation
                .wrapping_add(rows * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS),
            attention_branch: self
                .attention_branch
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            post_attention_residual: self
                .post_attention_residual
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            mlp_normalized: self.mlp_normalized.wrapping_add(rows * Qwen38_27B::HIDDEN),
            swiglu: self.swiglu.wrapping_add(rows * Qwen38_27B::INTERMEDIATE),
            mlp_branch: self.mlp_branch.wrapping_add(rows * Qwen38_27B::HIDDEN),
            residual_output: self.residual_output.wrapping_add(rows * Qwen38_27B::HIDDEN),
            final_normalized: self
                .final_normalized
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            lm_head_activation_codes: self
                .lm_head_activation_codes
                .wrapping_add(rows * Qwen38_27B::HIDDEN),
            lm_head_activation_scales: self.lm_head_activation_scales.wrapping_add(rows),
            logits: self.logits.wrapping_add(rows * Qwen38_27B::VOCAB),
            ..self
        }
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> Vec<usize> {
        vec![
            self.embedding.addr(),
            self.target_hidden.addr(),
            self.embedding_norm.addr(),
            self.hidden_norm.addr(),
            self.normalized_embedding.addr(),
            self.normalized_hidden.addr(),
            self.input_projection.addr(),
            self.residual.addr(),
            self.input_norm.addr(),
            self.attention_normalized.addr(),
            self.qkv_weight.addr(),
            self.qkv.addr(),
            self.query_norm.addr(),
            self.key_norm.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.block_tables.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.lengths.addr(),
            self.query.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
            self.attention.addr(),
            self.attention_activation.addr(),
            self.attention_output_weight.addr(),
            self.attention_branch.addr(),
            self.post_attention_norm.addr(),
            self.post_attention_residual.addr(),
            self.mlp_normalized.addr(),
            self.gate_up_weight.addr(),
            self.swiglu.addr(),
            self.down_weight.addr(),
            self.mlp_branch.addr(),
            self.final_norm.addr(),
            self.residual_output.addr(),
            self.final_normalized.addr(),
            self.lm_head_activation_codes.addr(),
            self.lm_head_activation_scales.addr(),
            self.lm_head_codes.addr(),
            self.lm_head_scales.addr(),
            self.logits.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    fusion: &'a MtpBf16FusionOp,
    norm: &'a ResidualNormOp<Qwen38_27B>,
    qkv: &'a MtpBf16QkvOp,
    qk_prepare: &'a MtpBf16QkPrepareOp,
    paged_gqa: &'a MtpBf16PagedGqaOp,
    attention_output: &'a MtpBf16AttentionOutputOp,
    mlp: &'a MtpBf16MlpOp,
    lm_head: &'a LmHeadOp<Qwen38_27B>,
}

#[derive(Clone, Copy)]
struct Stagers<'a> {
    embedding: &'a PinnedHostBuffer<u16>,
    table_rows: &'a PinnedHostBuffer<u32>,
    positions: &'a PinnedHostBuffer<u32>,
    lengths: &'a PinnedHostBuffer<u32>,
    rope_cos: &'a PinnedHostBuffer<f32>,
    rope_sin: &'a PinnedHostBuffer<f32>,
    continuation_hidden: &'a PinnedHostBuffer<u16>,
}

pub(crate) struct ResidentMtpArenaReservation {
    layout: ResidentMtpLayout,
    arena: DeviceArena,
    cache_arena: DeviceArena,
}

impl ResidentMtpArenaReservation {
    pub(crate) fn allocate(stream: &CudaStream) -> EngineResult<Self> {
        let layout = ResidentMtpLayout::build()?;
        let arena = DeviceArena::zeroed(stream, layout.arena())?;
        let cache_arena = DeviceArena::zeroed(stream, layout.cache_arena())?;
        Ok(Self {
            layout,
            arena,
            cache_arena,
        })
    }
}

impl ResidentMtpProgram {
    /// Loads the target and MTP arenas before either owner instantiates its resident graphs.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let (target, reservation) =
            ResidentModelProgram::from_snapshot_reserving_mtp(context, snapshot)?;
        Self::from_target_reservation(target, reservation)
    }

    /// Adds one exact resident MTP owner around an already-loaded target program.
    pub fn from_target(target: ResidentModelProgram) -> EngineResult<Self> {
        let context = target.context().clone();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let reservation = ResidentMtpArenaReservation::allocate(&stream)?;
        stream.synchronize().map_err(GpuError::from)?;
        Self::from_target_reservation(target, reservation)
    }

    fn from_target_reservation(
        target: ResidentModelProgram,
        reservation: ResidentMtpArenaReservation,
    ) -> EngineResult<Self> {
        let context = target.context().clone();
        let mtp = MtpBindings::bind(target.snapshot().as_ref())?;
        let qkv = mtp.materialize_qkv()?;
        let ResidentMtpArenaReservation {
            layout,
            arena,
            cache_arena,
        } = reservation;
        let regions = layout.regions();
        let cache_regions = layout.cache_regions();
        let stream = context.new_stream().map_err(GpuError::from)?;

        arena.copy_region_bytes_from_host(
            &stream,
            regions.embedding_norm,
            mtp.embedding_norm.bytes(),
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.hidden_norm, mtp.hidden_norm.bytes())?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.input_projection,
            mtp.input_projection.bytes(),
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.input_norm, mtp.input_norm.bytes())?;
        arena.copy_region_bytes_from_host(&stream, regions.qkv_weight, &qkv.weight_bf16)?;
        arena.copy_region_bytes_from_host(&stream, regions.query_norm, mtp.query_norm.bytes())?;
        arena.copy_region_bytes_from_host(&stream, regions.key_norm, mtp.key_norm.bytes())?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.attention_output_weight,
            mtp.attention_output_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.post_attention_norm,
            mtp.post_attention_norm.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.gate_up_weight,
            mtp.gate_up_weight_bf16,
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.down_weight, mtp.down_weight.bytes())?;
        arena.copy_region_bytes_from_host(&stream, regions.final_norm, mtp.final_norm.bytes())?;

        let fusion = MtpBf16FusionOp::new(&context)?;
        let norm = ResidualNormOp::new(&context)?;
        let qkv_op = MtpBf16QkvOp::new(&context)?;
        let qk_prepare = MtpBf16QkPrepareOp::new(&context)?;
        let paged_gqa = MtpBf16PagedGqaOp::new(&context)?;
        let attention_output = MtpBf16AttentionOutputOp::new(&context)?;
        let mlp = MtpBf16MlpOp::new(&context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(
            &context,
            product(
                "resident MTP embedding stager",
                MTP_PROMPT_ROWS,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let table_rows_stager =
            PinnedHostBuffer::zeroed(&context, MTP_PROMPT_ROWS).map_err(GpuError::from)?;
        let positions_stager =
            PinnedHostBuffer::zeroed(&context, MTP_PROMPT_ROWS).map_err(GpuError::from)?;
        let lengths_stager =
            PinnedHostBuffer::zeroed(&context, MTP_PROMPT_ROWS).map_err(GpuError::from)?;
        let rotary_values = product("resident MTP rotary stager", MTP_PROMPT_ROWS, ROTARY_PAIRS)?;
        let rope_cos_stager =
            PinnedHostBuffer::zeroed(&context, rotary_values).map_err(GpuError::from)?;
        let rope_sin_stager =
            PinnedHostBuffer::zeroed(&context, rotary_values).map_err(GpuError::from)?;
        let continuation_hidden_stager = PinnedHostBuffer::zeroed(
            &context,
            product(
                "resident MTP compact continuation hidden stager",
                MAX_BATCH,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;
        let pointers = Pointers::bind(&arena, &cache_arena, regions, cache_regions, &target)?;
        let ops = Ops {
            fusion: &fusion,
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            mlp: &mlp,
            lm_head: target.mtp_lm_head_op(),
        };
        let stagers = Stagers {
            embedding: &embedding_stager,
            table_rows: &table_rows_stager,
            positions: &positions_stager,
            lengths: &lengths_stager,
            rope_cos: &rope_cos_stager,
            rope_sin: &rope_sin_stager,
            continuation_hidden: &continuation_hidden_stager,
        };
        let graphs = capture_graphs(&stream, &target, &arena, regions, pointers, ops, stagers)?;
        stream.synchronize().map_err(GpuError::from)?;
        let base_address = arena.base_address();
        let cache_base_address = cache_arena.base_address();

        Ok(Self {
            graphs,
            arena,
            cache_arena,
            _fusion: fusion,
            _norm: norm,
            _qkv: qkv_op,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
            _mlp: mlp,
            embedding_stager,
            table_rows_stager,
            positions_stager,
            lengths_stager,
            rope_cos_stager,
            rope_sin_stager,
            continuation_hidden_stager,
            target,
            context,
            layout,
            base_address,
            cache_base_address,
        })
    }

    /// Stages one exact prompt-prime tile or scalar tail row.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_prompt(
        &mut self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        first_position: usize,
        next_token_ids: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpPromptRoute> {
        require_prompt(rows)?;
        let mut slots = [0usize; MTP_PROMPT_ROWS];
        let mut positions = [0u32; MTP_PROMPT_ROWS];
        fill_contiguous_metadata(
            &mut slots[..rows],
            &mut positions[..rows],
            slot,
            first_position,
        )?;
        self.stage_rows(
            stream,
            next_token_ids,
            &slots[..rows],
            &positions[..rows],
            rope_cos,
            rope_sin,
        )?;
        Ok(ResidentMtpPromptRoute {
            rows,
            slot,
            first_position,
        })
    }

    /// Stages distinct resident slots in compact draft row order.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_draft(
        &mut self,
        stream: &CudaStream,
        slots: &[usize],
        positions: &[u32],
        next_token_ids: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpDraftRoute> {
        require_batch(slots.len())?;
        for (index, &slot) in slots.iter().enumerate() {
            if slots[..index].contains(&slot) {
                return Err(EngineError::route(format!(
                    "resident MTP draft slot {slot} appears more than once"
                )));
            }
        }
        self.stage_rows(stream, next_token_ids, slots, positions, rope_cos, rope_sin)?;
        Ok(ResidentMtpDraftRoute { batch: slots.len() })
    }

    /// Stages compact draft rows with explicit prior target-conditioned hidden values.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_continuation_draft(
        &mut self,
        stream: &CudaStream,
        slots: &[usize],
        positions: &[u32],
        next_token_ids: &[u32],
        target_hidden: &[u16],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpDraftRoute> {
        require_batch(slots.len())?;
        let values = product(
            "resident MTP compact continuation hidden values",
            slots.len(),
            Qwen38_27B::HIDDEN,
        )?;
        if target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "resident MTP compact continuation hidden plane has {} values, expected {values}",
                target_hidden.len()
            )));
        }
        let route =
            self.stage_draft(stream, slots, positions, next_token_ids, rope_cos, rope_sin)?;
        self.continuation_hidden_stager[..values].copy_from_slice(target_hidden);
        Ok(route)
    }

    /// Stages one causal target-conditioned realignment sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_realign(
        &mut self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        first_position: usize,
        next_token_ids: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<ResidentMtpRealignRoute> {
        require_realign(tokens)?;
        let mut slots = [0usize; REALIGN_ROUTES];
        let mut positions = [0u32; REALIGN_ROUTES];
        fill_contiguous_metadata(
            &mut slots[..tokens],
            &mut positions[..tokens],
            slot,
            first_position,
        )?;
        self.stage_rows(
            stream,
            next_token_ids,
            &slots[..tokens],
            &positions[..tokens],
            rope_cos,
            rope_sin,
        )?;
        Ok(ResidentMtpRealignRoute {
            tokens,
            slot,
            first_position,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_rows(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
        slots: &[usize],
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        let rows = slots.len();
        if rows == 0 || rows > MTP_PROMPT_ROWS {
            return Err(EngineError::route(format!(
                "resident MTP rows {rows} are outside 1..={MTP_PROMPT_ROWS}"
            )));
        }
        if token_ids.len() != rows || positions.len() != rows {
            return Err(EngineError::layout(format!(
                "resident MTP tokens/positions have {}/{} rows, expected {rows}",
                token_ids.len(),
                positions.len()
            )));
        }
        let rotary_values = product("resident MTP rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "resident MTP rotary planes have {}/{} values, expected {rotary_values}",
                rope_cos.len(),
                rope_sin.len()
            )));
        }
        stream.synchronize().map_err(GpuError::from)?;

        let embedding = TextEndpointBindings::bind_embedding(self.target.snapshot().as_ref())?;
        for row in 0..rows {
            let slot = slots[row];
            if slot >= MAX_BATCH {
                return Err(EngineError::route(format!(
                    "resident MTP slot {slot} is outside 0..{MAX_BATCH}"
                )));
            }
            let position = usize::try_from(positions[row])
                .map_err(|_| EngineError::route("resident MTP position exceeds host width"))?;
            let required = position
                .checked_add(1)
                .ok_or_else(|| EngineError::route("resident MTP cache length overflows"))?;
            let reserved = self.target.mtp_kv_token_count(slot)?;
            if required > reserved {
                return Err(EngineError::route(format!(
                    "resident MTP slot {slot} owns {reserved} positions, expected at least {required}"
                )));
            }
            let token = usize::try_from(token_ids[row])
                .map_err(|_| EngineError::route("resident MTP token exceeds host width"))?;
            if token >= Qwen38_27B::VOCAB {
                return Err(EngineError::route(format!(
                    "resident MTP token {token} is outside vocabulary 0..{}",
                    Qwen38_27B::VOCAB
                )));
            }
            copy_embedding_row(
                embedding.bytes(),
                token,
                &mut self.embedding_stager
                    [row * Qwen38_27B::HIDDEN..(row + 1) * Qwen38_27B::HIDDEN],
            )?;
            self.table_rows_stager[row] = u32::try_from(slot)
                .map_err(|_| EngineError::layout("resident MTP slot exceeds u32"))?;
            self.positions_stager[row] = positions[row];
            self.lengths_stager[row] = positions[row]
                .checked_add(1)
                .ok_or_else(|| EngineError::route("resident MTP cache length exceeds u32"))?;
        }
        self.rope_cos_stager[..rotary_values].copy_from_slice(rope_cos);
        self.rope_sin_stager[..rotary_values].copy_from_slice(rope_sin);
        Ok(())
    }

    /// Replays one prompt-prime graph without producing logits.
    pub fn replay_prompt(
        &self,
        stream: &CudaStream,
        route: ResidentMtpPromptRoute,
    ) -> EngineResult<()> {
        let graph = &self.graphs.prompt[prompt_index(route.rows).ok_or_else(|| {
            EngineError::route(format!(
                "resident MTP prompt route {} is not admitted",
                route.rows
            ))
        })?];
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { graph.launch(stream) }?;
        Ok(())
    }

    /// Replays one full compact draft graph for exact `B=1..8`.
    pub fn replay_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.draft[route.batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays an exact compact continuation whose hidden rows are prior residuals in lane order.
    pub fn replay_continue_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.continue_draft[route.batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays an exact compact continuation from explicitly staged hidden rows.
    pub fn replay_staged_continue_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.staged_continue_draft[route.batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays one prime-only realignment graph for exact `K=1..4`.
    pub fn replay_prime(
        &self,
        stream: &CudaStream,
        route: ResidentMtpRealignRoute,
    ) -> EngineResult<()> {
        require_realign(route.tokens)?;
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.prime[route.tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays a prime-only prefix and one final full-logit realignment row.
    pub fn replay_realign(
        &self,
        stream: &CudaStream,
        route: ResidentMtpRealignRoute,
    ) -> EngineResult<()> {
        require_realign(route.tokens)?;
        // SAFETY: this ResidentMtpProgram owns every captured allocation (local
        // and target arenas, pinned stagers, op modules) for its whole life and
        // drops the graphs first.
        unsafe { self.graphs.realign[route.tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Reads active BF16 full-vocabulary draft logits.
    pub fn read_logits(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_batch(rows)?;
        let values = product("resident MTP logit values", rows, Qwen38_27B::VOCAB)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().logits, values)?)
    }

    /// Reads active BF16 draft logits into one reusable host allocation.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        require_batch(rows)?;
        let expected = product("resident MTP logit values", rows, Qwen38_27B::VOCAB)?;
        if destination.len() != expected {
            return Err(EngineError::layout(format!(
                "resident MTP logit destination has {} values, expected {expected} for B={rows}",
                destination.len()
            )));
        }
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.regions().logits, destination)?;
        Ok(())
    }

    /// Reads one exact BF16 draft-logit row into reusable host storage.
    pub fn read_logit_row_into(
        &self,
        stream: &CudaStream,
        row: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        if row >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "resident MTP logit row {row} is outside 0..{MAX_BATCH}"
            )));
        }
        if destination.len() != Qwen38_27B::VOCAB {
            return Err(EngineError::layout(format!(
                "resident MTP logit-row destination has {} values, expected {}",
                destination.len(),
                Qwen38_27B::VOCAB
            )));
        }
        let start = product("resident MTP logit-row offset", row, Qwen38_27B::VOCAB)?;
        self.arena.copy_slice_to_host_slice(
            stream,
            self.layout.regions().logits,
            start,
            destination,
        )?;
        Ok(())
    }

    /// Reads one exact final-residual row into reusable host storage.
    pub fn read_residual_row_into(
        &self,
        stream: &CudaStream,
        row: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        if row >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "resident MTP residual row {row} is outside 0..{MAX_BATCH}"
            )));
        }
        if destination.len() != Qwen38_27B::HIDDEN {
            return Err(EngineError::layout(format!(
                "resident MTP residual-row destination has {} values, expected {}",
                destination.len(),
                Qwen38_27B::HIDDEN
            )));
        }
        let start = product("resident MTP residual-row offset", row, Qwen38_27B::HIDDEN)?;
        self.arena.copy_slice_to_host_slice(
            stream,
            self.layout.regions().residual_output,
            start,
            destination,
        )?;
        Ok(())
    }

    /// Activates the one shared target/MTP page-table row.
    pub fn activate_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.target.activate_kv_slot(slot)
    }

    /// Extends shared ownership and clears both target and MTP pages before returning the route.
    pub fn reserve_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<PagedKvTableUpdate> {
        let update = self
            .target
            .reserve_kv_slot_tokens_unpublished(stream, slot, token_count)?;
        for logical_page in update.first_entry()..update.first_entry() + update.entry_count() {
            let physical_page = self.target.mtp_kv_physical_page(slot, logical_page)?;
            self.clear_mtp_physical_page(stream, physical_page)?;
        }
        self.target.publish_kv_slot_update(stream, update)?;
        Ok(update)
    }

    /// Releases trailing pages and updates the one shared logical length.
    pub fn truncate_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<usize> {
        let current_tokens = self.target.mtp_kv_token_count(slot)?;
        if token_count > current_tokens {
            return Err(EngineError::generation(format!(
                "resident MTP slot {slot} cannot truncate forwards from {current_tokens} to {token_count} tokens"
            )));
        }
        let retained_pages = token_count.div_ceil(ATTENTION_PAGE_SIZE);
        let existing_pages = self.target.mtp_kv_page_count(slot)?;
        for logical_page in retained_pages..existing_pages {
            let physical_page = self.target.mtp_kv_physical_page(slot, logical_page)?;
            self.clear_mtp_physical_page(stream, physical_page)?;
        }
        self.target
            .truncate_kv_slot_tokens(stream, slot, token_count)
    }

    /// Marks the one shared row as a reusable prefix.
    pub fn retain_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.target.retain_kv_slot(slot)
    }

    /// Clears MTP pages before the shared route is returned to the free inventory.
    pub fn recycle_kv_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        self.clear_mtp_slot_cache(stream, slot)?;
        self.target.recycle_kv_slot(stream, slot)
    }

    /// Clears target recurrent/cache state and the complete MTP cache mirror.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        self.target.reset_state(stream)?;
        let cache = self.layout.cache_regions();
        self.cache_arena.fill(stream, cache.key_pages, 0)?;
        self.cache_arena.fill(stream, cache.value_pages, 0)?;
        Ok(())
    }

    /// Clears one assigned slot in both target and MTP persistent owners.
    pub fn reset_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        self.target.reset_slot(stream, slot)?;
        self.clear_mtp_slot_cache(stream, slot)
    }

    fn clear_mtp_slot_cache(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        let pages = self.target.mtp_kv_page_count(slot)?;
        for logical_page in 0..pages {
            let physical_page = self.target.mtp_kv_physical_page(slot, logical_page)?;
            self.clear_mtp_physical_page(stream, physical_page)?;
        }
        Ok(())
    }

    fn clear_mtp_physical_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<()> {
        if physical_page >= LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::layout(format!(
                "resident MTP physical page {physical_page} is outside 0..{LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        let page_values = product(
            "resident MTP physical-page values",
            product(
                "resident MTP physical-page heads",
                Qwen38_27B::NUM_KV_HEADS,
                ATTENTION_PAGE_SIZE,
            )?,
            Qwen38_27B::HEAD_DIM,
        )?;
        let start = product(
            "resident MTP physical-page offset",
            physical_page,
            page_values,
        )?;
        let cache = self.layout.cache_regions();
        self.cache_arena
            .fill_slice(stream, cache.key_pages, start, page_values, 0)?;
        self.cache_arena
            .fill_slice(stream, cache.value_pages, start, page_values, 0)?;
        Ok(())
    }

    /// Exact target owner whose endpoint weights and page lifecycle are shared.
    pub const fn target(&self) -> &ResidentModelProgram {
        &self.target
    }

    pub(crate) const fn target_mut(&mut self) -> &mut ResidentModelProgram {
        &mut self.target
    }

    /// CUDA context shared by target, MTP arenas, operators, and graphs.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable MTP weight/workspace base address.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable MTP long-context cache base address.
    pub const fn cache_base_address(&self) -> u64 {
        self.cache_base_address
    }

    /// Exact unchanged source-BF16 MTP weights, excluding shared target endpoint weights.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact represented long-context MTP cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Address-stable typed MTP route workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete incremental MTP device bytes including padding.
    pub const fn owner_bytes(&self) -> usize {
        self.layout.owner_bytes()
    }

    /// Exact incremental alignment padding.
    pub const fn padding_bytes(&self) -> usize {
        self.layout.padding_bytes()
    }

    /// Page-locked bytes retained by all MTP graph upload nodes.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
            + self.table_rows_stager.num_bytes()
            + self.positions_stager.num_bytes()
            + self.lengths_stager.num_bytes()
            + self.rope_cos_stager.num_bytes()
            + self.rope_sin_stager.num_bytes()
            + self.continuation_hidden_stager.num_bytes()
    }

    /// Exact prompt, seeded draft, two continuation, prime, and realignment inventories.
    pub const fn graph_count(&self) -> usize {
        PROMPT_ROUTES.len() + 3 * MAX_BATCH + 2 * REALIGN_ROUTES
    }

    /// Checked resident MTP layout.
    pub const fn layout(&self) -> &ResidentMtpLayout {
        &self.layout
    }

    #[cfg(feature = "qualification")]
    /// Launches one prompt route eagerly with the production uploads and target handoff.
    pub fn qualification_launch_eager_prompt(
        &self,
        stream: &CudaStream,
        route: ResidentMtpPromptRoute,
    ) -> EngineResult<()> {
        require_prompt(route.rows)?;
        launch_upload(
            stream,
            route.rows,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_prime(stream, route.rows, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one full compact draft route eagerly.
    pub fn qualification_launch_eager_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        launch_upload(
            stream,
            route.batch,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_full(stream, route.batch, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one exact compact continuation eagerly.
    pub fn qualification_launch_eager_continue_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        launch_continue_upload(
            stream,
            route.batch,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_full(stream, route.batch, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one exact explicitly staged compact continuation eagerly.
    pub fn qualification_launch_eager_staged_continue_draft(
        &self,
        stream: &CudaStream,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<()> {
        require_batch(route.batch)?;
        launch_staged_continue_upload(
            stream,
            route.batch,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_full(stream, route.batch, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one prime-only realignment route eagerly.
    pub fn qualification_launch_eager_prime(
        &self,
        stream: &CudaStream,
        route: ResidentMtpRealignRoute,
    ) -> EngineResult<()> {
        require_realign(route.tokens)?;
        launch_upload(
            stream,
            route.tokens,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_prime(stream, route.tokens, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one full causal realignment route eagerly.
    pub fn qualification_launch_eager_realign(
        &self,
        stream: &CudaStream,
        route: ResidentMtpRealignRoute,
    ) -> EngineResult<()> {
        require_realign(route.tokens)?;
        launch_upload(
            stream,
            route.tokens,
            &self.target,
            &self.arena,
            self.layout.regions(),
            self.stagers(),
        )?;
        launch_realign(stream, route.tokens, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated direct production draft work for intrinsic timing.
    pub fn qualification_repeated_draft_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated resident MTP graph requires at least one operation",
            ));
        }
        let pointers = self.pointers()?;
        let ops = self.ops();
        let stagers = self.stagers();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_upload(
                    stream,
                    batch,
                    &self.target,
                    &self.arena,
                    self.layout.regions(),
                    stagers,
                )?;
                launch_full(stream, batch, ops, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated compact continuation work for intrinsic timing.
    pub fn qualification_repeated_continue_draft_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated resident MTP continuation graph requires at least one operation",
            ));
        }
        let pointers = self.pointers()?;
        let ops = self.ops();
        let stagers = self.stagers();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_continue_upload(
                    stream,
                    batch,
                    &self.target,
                    &self.arena,
                    self.layout.regions(),
                    stagers,
                )?;
                launch_full(stream, batch, ops, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production prompt graph.
    pub fn qualification_prompt_graph(
        &self,
        route: ResidentMtpPromptRoute,
    ) -> EngineResult<&CudaGraph> {
        Ok(&self.graphs.prompt[prompt_index(route.rows).ok_or_else(|| {
            EngineError::route(format!(
                "resident MTP prompt route {} is not admitted",
                route.rows
            ))
        })?])
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production draft graph.
    pub fn qualification_draft_graph(
        &self,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<&CudaGraph> {
        require_batch(route.batch)?;
        Ok(&self.graphs.draft[route.batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production continuation graph.
    pub fn qualification_continue_draft_graph(
        &self,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<&CudaGraph> {
        require_batch(route.batch)?;
        Ok(&self.graphs.continue_draft[route.batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production explicitly staged continuation graph.
    pub fn qualification_staged_continue_draft_graph(
        &self,
        route: ResidentMtpDraftRoute,
    ) -> EngineResult<&CudaGraph> {
        require_batch(route.batch)?;
        Ok(&self.graphs.staged_continue_draft[route.batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Returns every local and borrowed device address captured by MTP graphs.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = self.pointers()?.addresses();
        addresses.extend(self.target.qualification_mtp_prompt_source_addresses()?);
        Ok(addresses)
    }

    #[cfg(feature = "qualification")]
    /// Returns every page-locked source address retained by local MTP graphs.
    pub fn qualification_host_stager_addresses(&self) -> [usize; 7] {
        [
            self.embedding_stager.as_ptr().addr(),
            self.table_rows_stager.as_ptr().addr(),
            self.positions_stager.as_ptr().addr(),
            self.lengths_stager.as_ptr().addr(),
            self.rope_cos_stager.as_ptr().addr(),
            self.rope_sin_stager.as_ptr().addr(),
            self.continuation_hidden_stager.as_ptr().addr(),
        ]
    }

    #[cfg(feature = "qualification")]
    /// Fills every mutable MTP workspace plane with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena
            .fill(stream, regions.normalized_embedding, byte)?;
        self.arena.fill(stream, regions.normalized_hidden, byte)?;
        self.arena.fill(stream, regions.residual, byte)?;
        self.arena
            .fill(stream, regions.attention_normalized, byte)?;
        self.arena.fill(stream, regions.qkv, byte)?;
        self.arena.fill(stream, regions.query, byte)?;
        self.arena.fill(stream, regions.attention, byte)?;
        self.arena
            .fill(stream, regions.attention_activation, byte)?;
        self.arena.fill(stream, regions.attention_branch, byte)?;
        self.arena
            .fill(stream, regions.post_attention_residual, byte)?;
        self.arena.fill(stream, regions.mlp_normalized, byte)?;
        self.arena.fill(stream, regions.swiglu, byte)?;
        self.arena.fill(stream, regions.mlp_branch, byte)?;
        self.arena.fill(stream, regions.residual_output, byte)?;
        self.arena.fill(stream, regions.final_normalized, byte)?;
        self.arena
            .fill(stream, regions.lm_head_activation_codes, byte)?;
        self.arena
            .fill(stream, regions.lm_head_activation_scales, byte)?;
        self.arena.fill(stream, regions.logits, byte)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads active upstream seams and, for full routes, every downstream seam.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
        rows: usize,
        full: bool,
    ) -> EngineResult<ResidentMtpObservables> {
        if rows == 0 || rows > MTP_PROMPT_ROWS || (full && rows > MAX_BATCH) {
            return Err(EngineError::route(format!(
                "resident MTP observable rows {rows} are invalid for full={full}"
            )));
        }
        let regions = self.layout.regions();
        let hidden = product("resident MTP observed hidden", rows, Qwen38_27B::HIDDEN)?;
        let qkv = product(
            "resident MTP observed QKV",
            rows,
            Qwen38_27B::ATTENTION_QKV_ROWS,
        )?;
        let rotary = product("resident MTP observed rotary", rows, ROTARY_PAIRS)?;
        let attention = product(
            "resident MTP observed attention",
            rows,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let intermediate = product(
            "resident MTP observed intermediate",
            rows,
            Qwen38_27B::INTERMEDIATE,
        )?;
        let logits = product("resident MTP observed logits", rows, Qwen38_27B::VOCAB)?;
        Ok(ResidentMtpObservables {
            embedding: self
                .arena
                .copy_prefix_to_host(stream, regions.embedding, hidden)?,
            target_hidden: self
                .arena
                .copy_prefix_to_host(stream, regions.target_hidden, hidden)?,
            normalized_embedding: self.arena.copy_prefix_to_host(
                stream,
                regions.normalized_embedding,
                hidden,
            )?,
            normalized_hidden: self.arena.copy_prefix_to_host(
                stream,
                regions.normalized_hidden,
                hidden,
            )?,
            residual: self
                .arena
                .copy_prefix_to_host(stream, regions.residual, hidden)?,
            attention_normalized: self.arena.copy_prefix_to_host(
                stream,
                regions.attention_normalized,
                hidden,
            )?,
            qkv: self.arena.copy_prefix_to_host(stream, regions.qkv, qkv)?,
            rope_cos: self
                .arena
                .copy_prefix_to_host(stream, regions.rope_cos, rotary)?,
            rope_sin: self
                .arena
                .copy_prefix_to_host(stream, regions.rope_sin, rotary)?,
            block_tables: self.arena.copy_to_host(stream, regions.block_tables)?,
            table_rows: self
                .arena
                .copy_prefix_to_host(stream, regions.table_rows, rows)?,
            cache_positions: self.arena.copy_prefix_to_host(
                stream,
                regions.cache_positions,
                rows,
            )?,
            lengths: self
                .arena
                .copy_prefix_to_host(stream, regions.lengths, rows)?,
            query: self
                .arena
                .copy_prefix_to_host(stream, regions.query, attention)?,
            attention: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.attention, attention)
                })
                .transpose()?,
            attention_activation: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.attention_activation, attention)
                })
                .transpose()?,
            attention_branch: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.attention_branch, hidden)
                })
                .transpose()?,
            post_attention_residual: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.post_attention_residual, hidden)
                })
                .transpose()?,
            mlp_normalized: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.mlp_normalized, hidden)
                })
                .transpose()?,
            swiglu: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.swiglu, intermediate)
                })
                .transpose()?,
            mlp_branch: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.mlp_branch, hidden)
                })
                .transpose()?,
            residual_output: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.residual_output, hidden)
                })
                .transpose()?,
            final_normalized: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.final_normalized, hidden)
                })
                .transpose()?,
            lm_head_activation_codes: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.lm_head_activation_codes, hidden)
                })
                .transpose()?,
            lm_head_activation_scales: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.lm_head_activation_scales, rows)
                })
                .transpose()?,
            logits: full
                .then(|| {
                    self.arena
                        .copy_prefix_to_host(stream, regions.logits, logits)
                })
                .transpose()?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads one complete physical MTP K/V page.
    pub fn qualification_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        if physical_page >= LONG_CONTEXT_PHYSICAL_PAGES {
            return Err(EngineError::route(format!(
                "resident MTP cache page {physical_page} is outside 0..{LONG_CONTEXT_PHYSICAL_PAGES}"
            )));
        }
        let values = product(
            "resident MTP cache-page values",
            product(
                "resident MTP cache-page heads",
                Qwen38_27B::NUM_KV_HEADS,
                ATTENTION_PAGE_SIZE,
            )?,
            Qwen38_27B::HEAD_DIM,
        )?;
        let start = product("resident MTP cache-page offset", physical_page, values)?;
        let cache = self.layout.cache_regions();
        Ok((
            self.cache_arena
                .copy_slice_to_host(stream, cache.key_pages, start, values)?,
            self.cache_arena
                .copy_slice_to_host(stream, cache.value_pages, start, values)?,
        ))
    }

    #[cfg(feature = "qualification")]
    fn pointers(&self) -> EngineResult<Pointers> {
        Ok(Pointers::bind(
            &self.arena,
            &self.cache_arena,
            self.layout.regions(),
            self.layout.cache_regions(),
            &self.target,
        )?)
    }

    #[cfg(feature = "qualification")]
    fn ops(&self) -> Ops<'_> {
        Ops {
            fusion: &self._fusion,
            norm: &self._norm,
            qkv: &self._qkv,
            qk_prepare: &self._qk_prepare,
            paged_gqa: &self._paged_gqa,
            attention_output: &self._attention_output,
            mlp: &self._mlp,
            lm_head: self.target.mtp_lm_head_op(),
        }
    }

    #[cfg(feature = "qualification")]
    fn stagers(&self) -> Stagers<'_> {
        Stagers {
            embedding: &self.embedding_stager,
            table_rows: &self.table_rows_stager,
            positions: &self.positions_stager,
            lengths: &self.lengths_stager,
            rope_cos: &self.rope_cos_stager,
            rope_sin: &self.rope_sin_stager,
            continuation_hidden: &self.continuation_hidden_stager,
        }
    }
}

#[cfg(feature = "qualification")]
/// Every active resident MTP seam; downstream fields exist only for full routes.
#[derive(Debug, PartialEq)]
pub struct ResidentMtpObservables {
    /// Source embedding rows.
    pub embedding: Vec<u16>,
    /// Raw target residual handoff rows.
    pub target_hidden: Vec<u16>,
    /// Pre-fusion normalized embeddings.
    pub normalized_embedding: Vec<u16>,
    /// Pre-fusion normalized target rows.
    pub normalized_hidden: Vec<u16>,
    /// Fusion-projection residual rows.
    pub residual: Vec<u16>,
    /// Attention-input normalized rows.
    pub attention_normalized: Vec<u16>,
    /// Gathered source-BF16 QKV projection rows.
    pub qkv: Vec<u16>,
    /// Staged MRoPE cosine values.
    pub rope_cos: Vec<f32>,
    /// Staged MRoPE sine values.
    pub rope_sin: Vec<f32>,
    /// Complete shared logical-to-physical page table.
    pub block_tables: Vec<u32>,
    /// Active page-table row selectors.
    pub table_rows: Vec<u32>,
    /// Active absolute cache positions.
    pub cache_positions: Vec<u32>,
    /// Active causal lengths.
    pub lengths: Vec<u32>,
    /// Prepared FP32 query rows.
    pub query: Vec<f32>,
    /// Full-route FP32 paged-GQA output.
    pub attention: Option<Vec<f32>>,
    /// Full-route represented gated attention activation.
    pub attention_activation: Option<Vec<u16>>,
    /// Full-route attention projection branch.
    pub attention_branch: Option<Vec<u16>>,
    /// Full-route post-attention residual.
    pub post_attention_residual: Option<Vec<u16>>,
    /// Full-route normalized MLP input.
    pub mlp_normalized: Option<Vec<u16>>,
    /// Full-route represented SwiGLU activation.
    pub swiglu: Option<Vec<u16>>,
    /// Full-route MLP projection branch.
    pub mlp_branch: Option<Vec<u16>>,
    /// Full-route final residual.
    pub residual_output: Option<Vec<u16>>,
    /// Full-route final-normalized rows.
    pub final_normalized: Option<Vec<u16>>,
    /// Full-route dynamic E4M3 activation codes.
    pub lm_head_activation_codes: Option<Vec<u8>>,
    /// Full-route dynamic FP32 activation scales.
    pub lm_head_activation_scales: Option<Vec<f32>>,
    /// Full-route BF16 vocabulary logits.
    pub logits: Option<Vec<u16>>,
}

fn capture_graphs(
    stream: &CudaStream,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: ResidentMtpRegions,
    pointers: Pointers,
    ops: Ops<'_>,
    stagers: Stagers<'_>,
) -> EngineResult<Graphs> {
    let mut prompt = Vec::with_capacity(PROMPT_ROUTES.len());
    for rows in PROMPT_ROUTES {
        prompt.push(CudaGraph::capture(stream, || {
            launch_upload(stream, rows, target, arena, regions, stagers)?;
            launch_prime(stream, rows, ops, pointers)
        })?);
    }
    let mut draft = Vec::with_capacity(MAX_BATCH);
    for rows in 1..=MAX_BATCH {
        draft.push(CudaGraph::capture(stream, || {
            launch_upload(stream, rows, target, arena, regions, stagers)?;
            launch_full(stream, rows, ops, pointers)
        })?);
    }
    let mut continue_draft = Vec::with_capacity(MAX_BATCH);
    for rows in 1..=MAX_BATCH {
        continue_draft.push(CudaGraph::capture(stream, || {
            launch_continue_upload(stream, rows, target, arena, regions, stagers)?;
            launch_full(stream, rows, ops, pointers)
        })?);
    }
    let mut staged_continue_draft = Vec::with_capacity(MAX_BATCH);
    for rows in 1..=MAX_BATCH {
        staged_continue_draft.push(CudaGraph::capture(stream, || {
            launch_staged_continue_upload(stream, rows, target, arena, regions, stagers)?;
            launch_full(stream, rows, ops, pointers)
        })?);
    }
    let mut prime = Vec::with_capacity(REALIGN_ROUTES);
    for rows in 1..=REALIGN_ROUTES {
        prime.push(CudaGraph::capture(stream, || {
            launch_upload(stream, rows, target, arena, regions, stagers)?;
            launch_prime(stream, rows, ops, pointers)
        })?);
    }
    let mut realign = Vec::with_capacity(REALIGN_ROUTES);
    for rows in 1..=REALIGN_ROUTES {
        realign.push(CudaGraph::capture(stream, || {
            launch_upload(stream, rows, target, arena, regions, stagers)?;
            launch_realign(stream, rows, ops, pointers)
        })?);
    }
    Ok(Graphs {
        prompt: prompt
            .try_into()
            .map_err(|_| EngineError::layout("resident MTP prompt graph inventory differs"))?,
        draft: draft
            .try_into()
            .map_err(|_| EngineError::layout("resident MTP draft graph inventory differs"))?,
        continue_draft: continue_draft.try_into().map_err(|_| {
            EngineError::layout("resident MTP continuation graph inventory differs")
        })?,
        staged_continue_draft: staged_continue_draft.try_into().map_err(|_| {
            EngineError::layout("resident MTP staged-continuation graph inventory differs")
        })?,
        prime: prime
            .try_into()
            .map_err(|_| EngineError::layout("resident MTP prime graph inventory differs"))?,
        realign: realign
            .try_into()
            .map_err(|_| EngineError::layout("resident MTP realign graph inventory differs"))?,
    })
}

fn launch_upload(
    stream: &CudaStream,
    rows: usize,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: ResidentMtpRegions,
    stagers: Stagers<'_>,
) -> GpuResult<()> {
    let hidden_values = rows
        .checked_mul(Qwen38_27B::HIDDEN)
        .ok_or_else(|| GpuError::invalid_launch("resident MTP upload hidden count overflows"))?;
    let rotary_values = rows
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| GpuError::invalid_launch("resident MTP upload rotary count overflows"))?;
    // SAFETY: the owner retains all pinned sources and both device owners at fixed addresses.
    unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.embedding,
            stagers.embedding,
            hidden_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.table_rows,
            stagers.table_rows,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.cache_positions,
            stagers.positions,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(stream, regions.lengths, stagers.lengths, rows)?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_cos,
            stagers.rope_cos,
            rotary_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_sin,
            stagers.rope_sin,
            rotary_values,
        )?;
        target.enqueue_mtp_prompt_handoff(
            stream,
            rows,
            arena,
            regions.target_hidden,
            regions.block_tables,
        )?;
    }
    Ok(())
}

fn launch_continue_upload(
    stream: &CudaStream,
    rows: usize,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: ResidentMtpRegions,
    stagers: Stagers<'_>,
) -> GpuResult<()> {
    let hidden_values = rows.checked_mul(Qwen38_27B::HIDDEN).ok_or_else(|| {
        GpuError::invalid_launch("resident MTP continuation hidden count overflows")
    })?;
    let rotary_values = rows.checked_mul(ROTARY_PAIRS).ok_or_else(|| {
        GpuError::invalid_launch("resident MTP continuation rotary count overflows")
    })?;
    // Every compact row consumes the same row of the prior MTP residual before
    // the full B route overwrites it. B=1..8 retains the qualified leaf tilings.
    unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.embedding,
            stagers.embedding,
            hidden_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.table_rows,
            stagers.table_rows,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.cache_positions,
            stagers.positions,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(stream, regions.lengths, stagers.lengths, rows)?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_cos,
            stagers.rope_cos,
            rotary_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_sin,
            stagers.rope_sin,
            rotary_values,
        )?;
        arena.copy_prefix_from_arena_async(
            stream,
            regions.target_hidden,
            arena,
            regions.residual_output,
            hidden_values,
        )?;
        target.enqueue_mtp_block_table_handoff(stream, arena, regions.block_tables)?;
    }
    Ok(())
}

fn launch_staged_continue_upload(
    stream: &CudaStream,
    rows: usize,
    target: &ResidentModelProgram,
    arena: &DeviceArena,
    regions: ResidentMtpRegions,
    stagers: Stagers<'_>,
) -> GpuResult<()> {
    let hidden_values = rows.checked_mul(Qwen38_27B::HIDDEN).ok_or_else(|| {
        GpuError::invalid_launch("resident MTP staged continuation hidden count overflows")
    })?;
    let rotary_values = rows.checked_mul(ROTARY_PAIRS).ok_or_else(|| {
        GpuError::invalid_launch("resident MTP staged continuation rotary count overflows")
    })?;
    // B=1..8 uploads one exact 10,240-byte BF16 hidden row per lane; the kernels retain their
    // independently qualified exact-B launch shapes.
    unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.embedding,
            stagers.embedding,
            hidden_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.target_hidden,
            stagers.continuation_hidden,
            hidden_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.table_rows,
            stagers.table_rows,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.cache_positions,
            stagers.positions,
            rows,
        )?;
        arena.copy_prefix_from_pinned_host_async(stream, regions.lengths, stagers.lengths, rows)?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_cos,
            stagers.rope_cos,
            rotary_values,
        )?;
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.rope_sin,
            stagers.rope_sin,
            rotary_values,
        )?;
        target.enqueue_mtp_block_table_handoff(stream, arena, regions.block_tables)?;
    }
    Ok(())
}

fn launch_prime(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: exact route dispatch bounds every typed plane and the shared page table owns each
    // selected append position before the graph becomes observable to its caller.
    unsafe {
        ops.fusion.launch(
            stream,
            rows,
            pointers.embedding,
            pointers.target_hidden,
            pointers.embedding_norm,
            pointers.hidden_norm,
            pointers.normalized_embedding,
            pointers.normalized_hidden,
            pointers.input_projection,
            pointers.residual,
        )?;
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual,
            pointers.input_norm,
            pointers.attention_normalized,
        )?;
        ops.qkv.launch(
            stream,
            rows,
            pointers.attention_normalized,
            pointers.qkv_weight,
            pointers.qkv,
        )?;
        ops.qk_prepare.launch(
            stream,
            rows,
            pointers.qkv,
            pointers.query_norm,
            pointers.key_norm,
            pointers.rope_cos,
            pointers.rope_sin,
            pointers.block_tables,
            pointers.table_rows,
            LONG_CONTEXT_PHYSICAL_PAGES,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
        )?;
    }
    Ok(())
}

fn launch_full(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
) -> GpuResult<()> {
    launch_prime(stream, rows, ops, pointers)?;
    // SAFETY: downstream storage covers exact B=1..8 and target endpoint weights remain owned by
    // the outer resident program until these graphs are dropped.
    unsafe {
        ops.paged_gqa.launch(
            stream,
            rows,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            pointers.block_tables,
            pointers.table_rows,
            LONG_CONTEXT_PHYSICAL_PAGES,
            pointers.lengths,
            pointers.attention,
        )?;
        ops.attention_output.launch(
            stream,
            rows,
            pointers.attention,
            pointers.qkv,
            pointers.attention_activation,
            pointers.attention_output_weight,
            pointers.attention_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            rows,
            pointers.residual,
            pointers.attention_branch,
            pointers.post_attention_norm,
            pointers.post_attention_residual,
            pointers.mlp_normalized,
        )?;
        ops.mlp.launch(
            stream,
            rows,
            pointers.mlp_normalized,
            pointers.gate_up_weight,
            pointers.swiglu,
            pointers.down_weight,
            pointers.mlp_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            rows,
            pointers.post_attention_residual,
            pointers.mlp_branch,
            pointers.final_norm,
            pointers.residual_output,
            pointers.final_normalized,
        )?;
        ops.lm_head.launch(
            stream,
            rows,
            pointers.final_normalized,
            pointers.lm_head_activation_codes,
            pointers.lm_head_activation_scales,
            pointers.lm_head_codes,
            pointers.lm_head_scales,
            pointers.logits,
        )?;
    }
    Ok(())
}

fn launch_realign(
    stream: &CudaStream,
    tokens: usize,
    ops: Ops<'_>,
    pointers: Pointers,
) -> GpuResult<()> {
    if tokens > 1 {
        launch_prime(stream, tokens - 1, ops, pointers)?;
    }
    launch_full(stream, 1, ops, pointers.offset_rows(tokens - 1))
}

fn prompt_index(rows: usize) -> Option<usize> {
    PROMPT_ROUTES.iter().position(|&route| route == rows)
}

fn require_prompt(rows: usize) -> EngineResult<()> {
    if prompt_index(rows).is_none() {
        return Err(EngineError::route(format!(
            "resident MTP prompt rows {rows} are outside exact K=1 or T=32,64,128,1024"
        )));
    }
    Ok(())
}

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "resident MTP draft batch {batch} is outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_realign(tokens: usize) -> EngineResult<()> {
    if !(1..=REALIGN_ROUTES).contains(&tokens) {
        return Err(EngineError::route(format!(
            "resident MTP realignment length {tokens} is outside 1..={REALIGN_ROUTES}"
        )));
    }
    Ok(())
}

fn fill_contiguous_metadata(
    slots: &mut [usize],
    positions: &mut [u32],
    slot: usize,
    first_position: usize,
) -> EngineResult<()> {
    if slots.len() != positions.len() {
        return Err(EngineError::layout(
            "resident MTP contiguous metadata planes differ in length",
        ));
    }
    for (row, (slot_value, position)) in slots.iter_mut().zip(positions).enumerate() {
        *slot_value = slot;
        *position = u32::try_from(
            first_position
                .checked_add(row)
                .ok_or_else(|| EngineError::route("resident MTP position overflows"))?,
        )
        .map_err(|_| EngineError::route("resident MTP position exceeds u32"))?;
    }
    Ok(())
}

fn copy_embedding_row(source: &[u8], token: usize, destination: &mut [u16]) -> EngineResult<()> {
    let word_begin = product(
        "resident MTP embedding row offset",
        token,
        Qwen38_27B::HIDDEN,
    )?;
    let byte_begin = product("resident MTP embedding byte offset", word_begin, 2)?;
    let byte_len = product("resident MTP embedding row bytes", Qwen38_27B::HIDDEN, 2)?;
    let byte_end = byte_begin
        .checked_add(byte_len)
        .ok_or_else(|| EngineError::layout("resident MTP embedding byte range overflows"))?;
    let row = source.get(byte_begin..byte_end).ok_or_else(|| {
        EngineError::layout(format!(
            "resident MTP embedding row {token} is outside source"
        ))
    })?;
    for (target, bytes) in destination.iter_mut().zip(row.as_chunks::<2>().0) {
        *target = u16::from_le_bytes(*bytes);
    }
    Ok(())
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{PROMPT_ROUTES, REALIGN_ROUTES, prompt_index, require_batch, require_realign};

    #[test]
    fn resident_mtp_route_inventory_is_exact() {
        assert_eq!(PROMPT_ROUTES, [1, 32, 64, 128, 1_024]);
        for (index, rows) in PROMPT_ROUTES.into_iter().enumerate() {
            assert_eq!(prompt_index(rows), Some(index));
        }
        for batch in 1..=8 {
            assert!(require_batch(batch).is_ok());
        }
        for tokens in 1..=REALIGN_ROUTES {
            assert!(require_realign(tokens).is_ok());
        }
        assert!(require_batch(0).is_err());
        assert!(require_batch(9).is_err());
        assert!(require_realign(0).is_err());
        assert!(require_realign(5).is_err());
        assert_eq!(prompt_index(31), None);
    }
}

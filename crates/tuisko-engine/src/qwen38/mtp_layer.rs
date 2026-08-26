//! Resident source-backed Qwen3.8 MTP draft layer.

use crate::qwen38::mtp_layer_layout::{
    CONTEXT_CAPACITY, MtpLayerRegions, PHYSICAL_PAGES, TABLE_STRIDE,
};
use crate::{EngineError, EngineResult, MAX_BATCH, MtpLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    LmHeadOp, MtpBf16AttentionOutputOp, MtpBf16FusionOp, MtpBf16MlpOp, MtpBf16PagedGqaOp,
    MtpBf16QkPrepareOp, MtpBf16QkvOp, ResidualNormOp,
};
use tuisko_model::{Arch, CheckpointSnapshot, MtpBindings, Qwen38_27B, TextEndpointBindings};

const ROTARY_PAIRS: usize = 32;
const REALIGN_ROUTES: usize = 4;

/// One exact source-backed MTP owner with immutable draft and realignment graphs.
pub struct MtpLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    draft_graphs: [CudaGraph; MAX_BATCH],
    prime_graphs: [CudaGraph; REALIGN_ROUTES],
    realign_graphs: [CudaGraph; REALIGN_ROUTES],
    arena: DeviceArena,
    _fusion: MtpBf16FusionOp,
    _norm: ResidualNormOp<Qwen38_27B>,
    _qkv: MtpBf16QkvOp,
    _qk_prepare: MtpBf16QkPrepareOp,
    _paged_gqa: MtpBf16PagedGqaOp,
    _attention_output: MtpBf16AttentionOutputOp,
    _mlp: MtpBf16MlpOp,
    _lm_head: LmHeadOp<Qwen38_27B>,
    context: Arc<CudaContext>,
    layout: MtpLayerLayout,
    base_address: u64,
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
    fn bind(arena: &DeviceArena, regions: MtpLayerRegions) -> GpuResult<Self> {
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
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,
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
            lm_head_codes: arena.address(regions.lm_head_codes)?.cast_const(),
            lm_head_scales: arena.address(regions.lm_head_scales)?.cast_const(),
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

impl MtpLayerProgram {
    /// Loads the complete MTP source family and one shared source-native LM head.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: &CheckpointSnapshot<Qwen38_27B>,
    ) -> EngineResult<Self> {
        let mtp = MtpBindings::bind(snapshot)?;
        let qkv = mtp.materialize_qkv()?;
        let endpoint = TextEndpointBindings::bind(snapshot)?;
        let layout = MtpLayerLayout::build::<Qwen38_27B>()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;

        arena.copy_from_host(
            &stream,
            regions.embedding_norm,
            &mtp.embedding_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.hidden_norm,
            &mtp.hidden_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.input_projection,
            &mtp.input_projection.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &mtp.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.qkv_weight,
            &little_endian_words(&qkv.weight_bf16)?,
        )?;
        arena.copy_from_host(
            &stream,
            regions.query_norm,
            &mtp.query_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.key_norm,
            &mtp.key_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.attention_output_weight,
            &mtp.attention_output_weight.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.post_attention_norm,
            &mtp.post_attention_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.gate_up_weight,
            &little_endian_words(mtp.gate_up_weight_bf16)?,
        )?;
        arena.copy_from_host(
            &stream,
            regions.down_weight,
            &mtp.down_weight.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.final_norm,
            &mtp.final_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.lm_head_codes, endpoint.lm_head.codes())?;
        arena.copy_from_host(
            &stream,
            regions.lm_head_scales,
            &endpoint.lm_head_scale.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.block_tables,
            &(0..PHYSICAL_PAGES as u32).collect::<Vec<_>>(),
        )?;

        let fusion = MtpBf16FusionOp::new(context)?;
        let norm = ResidualNormOp::new(context)?;
        let qkv_op = MtpBf16QkvOp::new(context)?;
        let qk_prepare = MtpBf16QkPrepareOp::new(context)?;
        let paged_gqa = MtpBf16PagedGqaOp::new(context)?;
        let attention_output = MtpBf16AttentionOutputOp::new(context)?;
        let mlp = MtpBf16MlpOp::new(context)?;
        let lm_head = LmHeadOp::new(context)?;
        let pointers = Pointers::bind(&arena, regions)?;
        let ops = Ops {
            fusion: &fusion,
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            mlp: &mlp,
            lm_head: &lm_head,
        };
        let draft_graphs = capture_draft_routes(&stream, ops, pointers)?;
        let prime_graphs = capture_prime_routes(&stream, ops, pointers)?;
        let realign_graphs = capture_realign_routes(&stream, ops, pointers)?;
        let base_address = arena.base_address();

        Ok(Self {
            draft_graphs,
            prime_graphs,
            realign_graphs,
            arena,
            _fusion: fusion,
            _norm: norm,
            _qkv: qkv_op,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
            _mlp: mlp,
            _lm_head: lm_head,
            context: context.clone(),
            layout,
            base_address,
        })
    }

    /// Uploads aligned embedding and target-hidden inputs for one exact route.
    pub fn load_inputs(
        &self,
        stream: &CudaStream,
        rows: usize,
        embedding: &[u16],
        target_hidden: &[u16],
    ) -> EngineResult<()> {
        require_rows(rows)?;
        let values = product("MTP input values", rows, Qwen38_27B::HIDDEN)?;
        if embedding.len() != values || target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "MTP inputs have {}/{} values, expected {values} for rows={rows}",
                embedding.len(),
                target_hidden.len()
            )));
        }
        let regions = self.layout.regions();
        self.arena
            .copy_prefix_from_host(stream, regions.embedding, embedding)?;
        self.arena
            .copy_prefix_from_host(stream, regions.target_hidden, target_hidden)?;
        Ok(())
    }

    /// Selects one independent cache slot per active draft lane.
    pub fn load_draft_state(
        &self,
        stream: &CudaStream,
        batch: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        self.load_state(
            stream,
            batch,
            positions,
            &(0..batch as u32).collect::<Vec<_>>(),
            rope_cos,
            rope_sin,
        )
    }

    /// Selects one cache slot for a contiguous causal realignment sequence.
    pub fn load_realign_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_realign(tokens)?;
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "MTP realignment slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if positions
            .windows(2)
            .any(|pair| pair[1] != pair[0].saturating_add(1))
        {
            return Err(EngineError::route(
                "MTP realignment positions must form one contiguous sequence",
            ));
        }
        self.load_state(
            stream,
            tokens,
            positions,
            &vec![slot as u32; tokens],
            rope_cos,
            rope_sin,
        )
    }

    fn load_state(
        &self,
        stream: &CudaStream,
        rows: usize,
        positions: &[u32],
        table_rows: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        if positions.len() != rows || table_rows.len() != rows {
            return Err(EngineError::layout(format!(
                "MTP route metadata has {}/{} rows, expected {rows}",
                positions.len(),
                table_rows.len()
            )));
        }
        let rotary_values = product("MTP rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "MTP rotary planes must each have {rotary_values} values for rows={rows}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= CONTEXT_CAPACITY {
                    return Err(EngineError::route(format!(
                        "MTP cache position {position} exceeds the {CONTEXT_CAPACITY}-token isolated capacity"
                    )));
                }
                position
                    .checked_add(1)
                    .ok_or_else(|| EngineError::route("MTP cache length overflows"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let regions = self.layout.regions();
        self.arena
            .copy_prefix_from_host(stream, regions.table_rows, table_rows)?;
        self.arena
            .copy_prefix_from_host(stream, regions.cache_positions, positions)?;
        self.arena
            .copy_prefix_from_host(stream, regions.lengths, &lengths)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_sin, rope_sin)?;
        Ok(())
    }

    /// Replaces both complete represented BF16 cache planes.
    pub fn load_cache(
        &self,
        stream: &CudaStream,
        key_pages: &[u16],
        value_pages: &[u16],
    ) -> EngineResult<()> {
        let regions = self.layout.regions();
        if key_pages.len() != regions.key_pages.len()
            || value_pages.len() != regions.value_pages.len()
        {
            return Err(EngineError::layout(format!(
                "MTP cache planes must each have {} BF16 values",
                regions.key_pages.len()
            )));
        }
        self.arena
            .copy_from_host(stream, regions.key_pages, key_pages)?;
        self.arena
            .copy_from_host(stream, regions.value_pages, value_pages)?;
        Ok(())
    }

    /// Clears every represented BF16 MTP cache page.
    pub fn reset_cache(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.key_pages, 0)?;
        self.arena.fill(stream, regions.value_pages, 0)?;
        Ok(())
    }

    /// Replays one complete draft route for exact `B=1..=8`.
    pub fn replay_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this MtpLayerProgram owns every captured allocation (arena,
        // op modules) for its whole life and drops the graphs first.
        unsafe { self.draft_graphs[batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays one prime-only cache append for exact `K=1..=4`.
    pub fn replay_prime(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        // SAFETY: this MtpLayerProgram owns every captured allocation (arena,
        // op modules) for its whole life and drops the graphs first.
        unsafe { self.prime_graphs[tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays a causal realignment with prime-only prefix and final full row.
    pub fn replay_realign(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        // SAFETY: this MtpLayerProgram owns every captured allocation (arena,
        // op modules) for its whole life and drops the graphs first.
        unsafe { self.realign_graphs[tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Reads active full-vocabulary BF16 logits.
    pub fn read_logits(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("MTP logit values", rows, Qwen38_27B::VOCAB)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().logits, values)?)
    }

    /// CUDA context shared by all owner allocations and graphs.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every route.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Exact unchanged BF16 MTP weight bytes.
    pub const fn mtp_weight_bytes(&self) -> usize {
        self.layout.mtp_weight_bytes()
    }

    /// One source-native FP8 LM head used by both target and draft composition.
    pub const fn shared_endpoint_weight_bytes(&self) -> usize {
        self.layout.shared_endpoint_weight_bytes()
    }

    /// MTP plus shared endpoint resident weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact represented BF16 short-cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete single allocation, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Weights, cache, and workspace without padding.
    pub const fn owner_bytes(&self) -> usize {
        self.layout.owner_bytes()
    }

    /// Short per-slot cache capacity of this isolated composition owner.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Number of immutable exact graph entries.
    pub const fn graph_count(&self) -> usize {
        MAX_BATCH + 2 * REALIGN_ROUTES
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete draft route eagerly.
    pub fn launch_eager_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_full(
            stream,
            batch,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete B=1 route at an existing maximum-B row offset.
    pub fn launch_eager_draft_row(&self, stream: &CudaStream, row: usize) -> EngineResult<()> {
        if row >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "MTP qualification row {row} is outside 0..{MAX_BATCH}"
            )));
        }
        launch_full(
            stream,
            1,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?.offset_rows(row),
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one prime-only route eagerly.
    pub fn launch_eager_prime(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        launch_prime(
            stream,
            tokens,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one causal realignment route eagerly.
    pub fn launch_eager_realign(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        launch_realign(
            stream,
            tokens,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one production draft graph for direct benchmark registration.
    pub fn qualification_draft_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.draft_graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete draft routes for intrinsic-path timing.
    pub fn qualification_repeated_draft_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated MTP layer graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_full(stream, batch, ops, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every arena address in checked layout order.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        Ok(Pointers::bind(&self.arena, self.layout.regions())?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Fills every mutable non-cache output plane with one byte sentinel.
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
    /// Reads every externally observable mutable seam and route metadata plane.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<MtpLayerObservables> {
        let regions = self.layout.regions();
        Ok(MtpLayerObservables {
            embedding: self.arena.copy_to_host(stream, regions.embedding)?,
            target_hidden: self.arena.copy_to_host(stream, regions.target_hidden)?,
            normalized_embedding: self
                .arena
                .copy_to_host(stream, regions.normalized_embedding)?,
            normalized_hidden: self.arena.copy_to_host(stream, regions.normalized_hidden)?,
            residual: self.arena.copy_to_host(stream, regions.residual)?,
            attention_normalized: self
                .arena
                .copy_to_host(stream, regions.attention_normalized)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            rope_cos: self.arena.copy_to_host(stream, regions.rope_cos)?,
            rope_sin: self.arena.copy_to_host(stream, regions.rope_sin)?,
            block_tables: self.arena.copy_to_host(stream, regions.block_tables)?,
            table_rows: self.arena.copy_to_host(stream, regions.table_rows)?,
            cache_positions: self.arena.copy_to_host(stream, regions.cache_positions)?,
            lengths: self.arena.copy_to_host(stream, regions.lengths)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
            key_pages: self.arena.copy_to_host(stream, regions.key_pages)?,
            value_pages: self.arena.copy_to_host(stream, regions.value_pages)?,
            attention: self.arena.copy_to_host(stream, regions.attention)?,
            attention_activation: self
                .arena
                .copy_to_host(stream, regions.attention_activation)?,
            attention_branch: self.arena.copy_to_host(stream, regions.attention_branch)?,
            post_attention_residual: self
                .arena
                .copy_to_host(stream, regions.post_attention_residual)?,
            mlp_normalized: self.arena.copy_to_host(stream, regions.mlp_normalized)?,
            swiglu: self.arena.copy_to_host(stream, regions.swiglu)?,
            mlp_branch: self.arena.copy_to_host(stream, regions.mlp_branch)?,
            residual_output: self.arena.copy_to_host(stream, regions.residual_output)?,
            final_normalized: self.arena.copy_to_host(stream, regions.final_normalized)?,
            lm_head_activation_codes: self
                .arena
                .copy_to_host(stream, regions.lm_head_activation_codes)?,
            lm_head_activation_scales: self
                .arena
                .copy_to_host(stream, regions.lm_head_activation_scales)?,
            logits: self.arena.copy_to_host(stream, regions.logits)?,
        })
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
            lm_head: &self._lm_head,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable MTP owner state exposed only to qualification.
pub struct MtpLayerObservables {
    /// Input embedding rows.
    pub embedding: Vec<u16>,
    /// Aligned target hidden rows.
    pub target_hidden: Vec<u16>,
    /// Pre-fusion normalized embedding rows.
    pub normalized_embedding: Vec<u16>,
    /// Pre-fusion normalized target hidden rows.
    pub normalized_hidden: Vec<u16>,
    /// BF16 fusion-projection residual rows.
    pub residual: Vec<u16>,
    /// Pre-attention normalized rows.
    pub attention_normalized: Vec<u16>,
    /// Gathered query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Loaded rotary cosine values.
    pub rope_cos: Vec<f32>,
    /// Loaded rotary sine values.
    pub rope_sin: Vec<f32>,
    /// Complete physical-page mapping.
    pub block_tables: Vec<u32>,
    /// Per-row selected page-table slot.
    pub table_rows: Vec<u32>,
    /// Per-row cache append positions.
    pub cache_positions: Vec<u32>,
    /// Per-row causal attention lengths.
    pub lengths: Vec<u32>,
    /// Prepared FP32 query values.
    pub query: Vec<f32>,
    /// Complete represented BF16 key cache.
    pub key_pages: Vec<u16>,
    /// Complete represented BF16 value cache.
    pub value_pages: Vec<u16>,
    /// FP32 paged-GQA values, gated in place by attention output.
    pub attention: Vec<f32>,
    /// Represented BF16 gated attention activation.
    pub attention_activation: Vec<u16>,
    /// Source-BF16 attention output-projection branch.
    pub attention_branch: Vec<u16>,
    /// Residual after the attention branch.
    pub post_attention_residual: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub mlp_normalized: Vec<u16>,
    /// Represented BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Source-BF16 MLP down-projection branch.
    pub mlp_branch: Vec<u16>,
    /// Final draft residual rows.
    pub residual_output: Vec<u16>,
    /// Final-normalized rows consumed by the shared LM head.
    pub final_normalized: Vec<u16>,
    /// Shared LM-head dynamic E4M3 activation codes.
    pub lm_head_activation_codes: Vec<u8>,
    /// Shared LM-head dynamic FP32 activation scales.
    pub lm_head_activation_scales: Vec<f32>,
    /// Full-vocabulary BF16 logits.
    pub logits: Vec<u16>,
}

fn capture_draft_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_full(stream, batch, ops, pointers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("MTP draft graph inventory has wrong cardinality"))
}

fn capture_prime_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; REALIGN_ROUTES]> {
    let mut graphs = Vec::with_capacity(REALIGN_ROUTES);
    for tokens in 1..=REALIGN_ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            launch_prime(stream, tokens, ops, pointers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("MTP prime graph inventory has wrong cardinality"))
}

fn capture_realign_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; REALIGN_ROUTES]> {
    let mut graphs = Vec::with_capacity(REALIGN_ROUTES);
    for tokens in 1..=REALIGN_ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            launch_realign(stream, tokens, ops, pointers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("MTP realignment graph inventory has wrong cardinality"))
}

fn launch_prime(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: the one owner arena provides complete aligned maximum-B planes;
    // exact dispatch restricts writes to the active prime-only prefix.
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
            TABLE_STRIDE,
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
    // SAFETY: every pointer names one aligned, disjoint maximum-B owner plane.
    // The page table covers each published causal length and exact dispatch
    // restricts all downstream writes to `rows`.
    unsafe {
        ops.paged_gqa.launch(
            stream,
            rows,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            pointers.block_tables,
            pointers.table_rows,
            TABLE_STRIDE,
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

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "MTP draft batch {batch} is outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_realign(tokens: usize) -> EngineResult<()> {
    if !(1..=REALIGN_ROUTES).contains(&tokens) {
        return Err(EngineError::route(format!(
            "MTP realignment length {tokens} is outside 1..={REALIGN_ROUTES}"
        )));
    }
    Ok(())
}

fn require_rows(rows: usize) -> EngineResult<()> {
    require_batch(rows)
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(EngineError::layout(
            "BF16 source plane has an odd byte length",
        ));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, REALIGN_ROUTES, require_batch, require_realign};

    #[test]
    fn exact_draft_and_realign_route_inventory_is_complete() {
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(REALIGN_ROUTES, 4);
        for batch in 1..=MAX_BATCH {
            assert!(require_batch(batch).is_ok());
        }
        for tokens in 1..=REALIGN_ROUTES {
            assert!(require_realign(tokens).is_ok());
        }
        assert!(require_batch(0).is_err());
        assert!(require_batch(9).is_err());
        assert!(require_realign(0).is_err());
        assert!(require_realign(5).is_err());
    }
}

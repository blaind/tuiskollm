//! Resident source-backed Qwen3.6 MTP transformer layer.

use crate::qwen36::mtp_layer_layout::{
    QWEN36_MTP_PHYSICAL_PAGES, QWEN36_MTP_PROMPT_ROWS, QWEN36_MTP_TABLE_STRIDE,
    Qwen36MtpLayerRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen36MtpLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen36Fp8AttentionQkPrepareOp, Qwen36Fp8PagedGqaOp, Qwen36MoeRouterOp,
    Qwen36MtpBf16AttentionOutputOp, Qwen36MtpBf16FusionOp, Qwen36MtpBf16MoeOp, Qwen36MtpBf16QkvOp,
    Qwen36ResidualNormOp,
};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B, Qwen36MtpBindings};

const ROTARY_PAIRS: usize = 32;
const REALIGN_ROUTES: usize = 4;
const PROMPT_ROUTES: [usize; 3] = [32, 64, QWEN36_MTP_PROMPT_ROWS];

/// One exact source-backed Qwen3.6 MTP layer without a duplicate text endpoint.
pub struct Qwen36MtpLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    draft_graphs: [CudaGraph; MAX_BATCH],
    prime_graphs: [CudaGraph; REALIGN_ROUTES],
    realign_graphs: [CudaGraph; REALIGN_ROUTES],
    prompt_graphs: [CudaGraph; PROMPT_ROUTES.len()],
    arena: DeviceArena,
    _fusion: Qwen36MtpBf16FusionOp,
    _norm: Qwen36ResidualNormOp,
    _qkv: Qwen36MtpBf16QkvOp,
    _qk_prepare: Qwen36Fp8AttentionQkPrepareOp,
    _paged_gqa: Qwen36Fp8PagedGqaOp,
    _attention_output: Qwen36MtpBf16AttentionOutputOp,
    _router: Qwen36MoeRouterOp,
    _experts: Qwen36MtpBf16MoeOp,
    context: Arc<CudaContext>,
    layout: Qwen36MtpLayerLayout,
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
    key_pages: *mut u8,
    value_pages: *mut u8,
    attention: *mut f32,
    attention_activation: *mut u16,
    attention_output_weight: *const u16,
    attention_branch: *mut u16,
    post_attention_norm: *const u16,
    post_attention_residual: *mut u16,
    moe_normalized: *mut u16,
    router_weight: *const u16,
    router_logits: *mut u16,
    expert_indices: *mut u16,
    routing_weights: *mut u16,
    routed_gate_up_weight: *const u16,
    routed_down_weight: *const u16,
    shared_gate_weight: *const u16,
    shared_up_weight: *const u16,
    shared_down_weight: *const u16,
    shared_expert_gate_weight: *const u16,
    expert_intermediate: *mut u16,
    expert_output: *mut u16,
    shared_gate_output: *mut u16,
    moe_branch: *mut u16,
    final_norm: *const u16,
    residual_output: *mut u16,
    final_normalized: *mut u16,
    table_stride: usize,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: Qwen36MtpLayerRegions) -> GpuResult<Self> {
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
            moe_normalized: arena.address(regions.moe_normalized)?,
            router_weight: arena.address(regions.router_weight)?.cast_const(),
            router_logits: arena.address(regions.router_logits)?,
            expert_indices: arena.address(regions.expert_indices)?,
            routing_weights: arena.address(regions.routing_weights)?,
            routed_gate_up_weight: arena.address(regions.routed_gate_up_weight)?.cast_const(),
            routed_down_weight: arena.address(regions.routed_down_weight)?.cast_const(),
            shared_gate_weight: arena.address(regions.shared_gate_weight)?.cast_const(),
            shared_up_weight: arena.address(regions.shared_up_weight)?.cast_const(),
            shared_down_weight: arena.address(regions.shared_down_weight)?.cast_const(),
            shared_expert_gate_weight: arena
                .address(regions.shared_expert_gate_weight)?
                .cast_const(),
            expert_intermediate: arena.address(regions.expert_intermediate)?,
            expert_output: arena.address(regions.expert_output)?,
            shared_gate_output: arena.address(regions.shared_gate_output)?,
            moe_branch: arena.address(regions.moe_branch)?,
            final_norm: arena.address(regions.final_norm)?.cast_const(),
            residual_output: arena.address(regions.residual_output)?,
            final_normalized: arena.address(regions.final_normalized)?,
            table_stride: QWEN36_MTP_TABLE_STRIDE,
        })
    }

    fn offset_rows(self, rows: usize) -> Self {
        Self {
            embedding: self.embedding.wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            target_hidden: self.target_hidden.wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            normalized_embedding: self
                .normalized_embedding
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            normalized_hidden: self
                .normalized_hidden
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            residual: self.residual.wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            attention_normalized: self
                .attention_normalized
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            qkv: self
                .qkv
                .wrapping_add(rows * Qwen36Moe35B::ATTENTION_QKV_ROWS),
            rope_cos: self.rope_cos.wrapping_add(rows * ROTARY_PAIRS),
            rope_sin: self.rope_sin.wrapping_add(rows * ROTARY_PAIRS),
            table_rows: self.table_rows.wrapping_add(rows),
            cache_positions: self.cache_positions.wrapping_add(rows),
            lengths: self.lengths.wrapping_add(rows),
            query: self
                .query
                .wrapping_add(rows * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS),
            attention: self
                .attention
                .wrapping_add(rows * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS),
            attention_activation: self
                .attention_activation
                .wrapping_add(rows * Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS),
            attention_branch: self
                .attention_branch
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            post_attention_residual: self
                .post_attention_residual
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            moe_normalized: self
                .moe_normalized
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            router_logits: self
                .router_logits
                .wrapping_add(rows * Qwen36Moe35B::NUM_EXPERTS),
            expert_indices: self
                .expert_indices
                .wrapping_add(rows * Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN),
            routing_weights: self
                .routing_weights
                .wrapping_add(rows * Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN),
            expert_intermediate: self.expert_intermediate.wrapping_add(
                rows * (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::INTERMEDIATE,
            ),
            expert_output: self.expert_output.wrapping_add(
                rows * (Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1) * Qwen36Moe35B::HIDDEN,
            ),
            shared_gate_output: self.shared_gate_output.wrapping_add(rows),
            moe_branch: self.moe_branch.wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            residual_output: self
                .residual_output
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
            final_normalized: self
                .final_normalized
                .wrapping_add(rows * Qwen36Moe35B::HIDDEN),
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
            self.moe_normalized.addr(),
            self.router_weight.addr(),
            self.router_logits.addr(),
            self.expert_indices.addr(),
            self.routing_weights.addr(),
            self.routed_gate_up_weight.addr(),
            self.routed_down_weight.addr(),
            self.shared_gate_weight.addr(),
            self.shared_up_weight.addr(),
            self.shared_down_weight.addr(),
            self.shared_expert_gate_weight.addr(),
            self.expert_intermediate.addr(),
            self.expert_output.addr(),
            self.shared_gate_output.addr(),
            self.moe_branch.addr(),
            self.final_norm.addr(),
            self.residual_output.addr(),
            self.final_normalized.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    fusion: &'a Qwen36MtpBf16FusionOp,
    norm: &'a Qwen36ResidualNormOp,
    qkv: &'a Qwen36MtpBf16QkvOp,
    qk_prepare: &'a Qwen36Fp8AttentionQkPrepareOp,
    paged_gqa: &'a Qwen36Fp8PagedGqaOp,
    attention_output: &'a Qwen36MtpBf16AttentionOutputOp,
    router: &'a Qwen36MoeRouterOp,
    experts: &'a Qwen36MtpBf16MoeOp,
}

impl Qwen36MtpLayerProgram {
    /// Loads the complete Qwen3.6 MTP source family and captures exact routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: &CheckpointSnapshot<Qwen36Moe35B>,
    ) -> EngineResult<Self> {
        let mtp = Qwen36MtpBindings::bind(snapshot)?;
        let qkv = mtp.materialize_qkv()?;
        let layout = Qwen36MtpLayerLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;

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
            regions.router_weight,
            mtp.router_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.routed_gate_up_weight,
            mtp.routed_gate_up_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.routed_down_weight,
            mtp.routed_down_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.shared_gate_weight,
            mtp.shared_gate_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.shared_up_weight,
            mtp.shared_up_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.shared_down_weight,
            mtp.shared_down_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            regions.shared_expert_gate_weight,
            mtp.shared_expert_gate_weight.bytes(),
        )?;
        arena.copy_region_bytes_from_host(&stream, regions.final_norm, mtp.final_norm.bytes())?;
        arena.copy_from_host(
            &stream,
            regions.block_tables,
            &(0..QWEN36_MTP_PHYSICAL_PAGES as u32).collect::<Vec<_>>(),
        )?;

        let fusion = Qwen36MtpBf16FusionOp::new(context)?;
        let norm = Qwen36ResidualNormOp::new(context)?;
        let qkv_op = Qwen36MtpBf16QkvOp::new(context)?;
        let qk_prepare = Qwen36Fp8AttentionQkPrepareOp::new(context)?;
        let paged_gqa = Qwen36Fp8PagedGqaOp::new(context)?;
        let attention_output = Qwen36MtpBf16AttentionOutputOp::new(context)?;
        let router = Qwen36MoeRouterOp::new(context)?;
        let experts = Qwen36MtpBf16MoeOp::new(context)?;
        let pointers = Pointers::bind(&arena, regions)?;
        let ops = Ops {
            fusion: &fusion,
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            router: &router,
            experts: &experts,
        };
        let draft_graphs = capture_draft_routes(&stream, ops, pointers)?;
        let prime_graphs = capture_prime_routes(&stream, ops, pointers)?;
        let realign_graphs = capture_realign_routes(&stream, ops, pointers)?;
        let prompt_graphs = capture_prompt_routes(&stream, ops, pointers)?;
        let base_address = arena.base_address();

        Ok(Self {
            draft_graphs,
            prime_graphs,
            realign_graphs,
            prompt_graphs,
            arena,
            _fusion: fusion,
            _norm: norm,
            _qkv: qkv_op,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
            _router: router,
            _experts: experts,
            context: Arc::clone(context),
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
        require_rows_or_prompt(rows)?;
        let values = product("Qwen3.6 MTP input values", rows, Qwen36Moe35B::HIDDEN)?;
        if embedding.len() != values || target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.6 MTP inputs have {}/{} values, expected {values} for rows={rows}",
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

    /// Uploads next-token embedding rows for a resident target handoff.
    pub fn load_embeddings(
        &self,
        stream: &CudaStream,
        rows: usize,
        embedding: &[u16],
    ) -> EngineResult<()> {
        require_rows_or_prompt(rows)?;
        let values = product("Qwen3.6 MTP embedding values", rows, Qwen36Moe35B::HIDDEN)?;
        if embedding.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.6 MTP embedding input has {} values, expected {values} for rows={rows}",
                embedding.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().embedding, embedding)?;

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

    /// Selects explicit stable slots for compact resident draft rows.
    pub fn load_compact_draft_state(
        &self,
        stream: &CudaStream,
        positions: &[u32],
        slots: &[usize],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_batch(positions.len())?;
        if slots.len() != positions.len() {
            return Err(EngineError::layout(format!(
                "Qwen3.6 MTP compact route has {} positions and {} slots",
                positions.len(),
                slots.len()
            )));
        }
        let mut seen = [false; MAX_BATCH];
        let table_rows = slots
            .iter()
            .map(|&slot| {
                if slot >= MAX_BATCH || seen[slot] {
                    return Err(EngineError::route(format!(
                        "Qwen3.6 MTP compact slot {slot} is repeated or outside 0..{MAX_BATCH}"
                    )));
                }
                seen[slot] = true;
                u32::try_from(slot).map_err(|_| EngineError::layout("Qwen3.6 MTP slot exceeds u32"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        self.load_state(
            stream,
            positions.len(),
            positions,
            &table_rows,
            rope_cos,
            rope_sin,
        )
    }

    /// Stages one exact prompt-prime tile into a stable MTP cache row.
    pub fn load_prompt_state(
        &self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_prompt(rows)?;
        if slot >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Qwen3.6 MTP prompt slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        let end = first_position
            .checked_add(rows)
            .ok_or_else(|| EngineError::route("Qwen3.6 MTP prompt positions overflow"))?;
        let positions = (first_position..end)
            .map(|position| {
                u32::try_from(position)
                    .map_err(|_| EngineError::route("Qwen3.6 MTP prompt position exceeds u32"))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        self.load_state(
            stream,
            rows,
            &positions,
            &vec![slot as u32; rows],
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
                "Qwen3.6 MTP realignment slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if positions
            .windows(2)
            .any(|pair| pair[1] != pair[0].saturating_add(1))
        {
            return Err(EngineError::route(
                "Qwen3.6 MTP realignment positions must form one contiguous sequence",
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
                "Qwen3.6 MTP route metadata has {}/{} rows, expected {rows}",
                positions.len(),
                table_rows.len()
            )));
        }
        let rotary_values = product("Qwen3.6 MTP rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.6 MTP rotary planes must each have {rotary_values} values for rows={rows}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= self.context_capacity() {
                    return Err(EngineError::route(format!(
                        "Qwen3.6 MTP cache position {position} exceeds the {}-token capacity",
                        self.context_capacity()
                    )));
                }
                position
                    .checked_add(1)
                    .ok_or_else(|| EngineError::route("Qwen3.6 MTP cache length overflows"))
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

    /// Replaces both complete represented E4M3 cache planes.
    pub fn load_cache(
        &self,
        stream: &CudaStream,
        key_pages: &[u8],
        value_pages: &[u8],
    ) -> EngineResult<()> {
        let regions = self.layout.regions();
        if key_pages.len() != regions.key_pages.len()
            || value_pages.len() != regions.value_pages.len()
        {
            return Err(EngineError::layout(format!(
                "Qwen3.6 MTP cache planes must each have {} E4M3 values",
                regions.key_pages.len()
            )));
        }
        self.arena
            .copy_from_host(stream, regions.key_pages, key_pages)?;
        self.arena
            .copy_from_host(stream, regions.value_pages, value_pages)?;
        Ok(())
    }

    /// Clears every represented E4M3 MTP cache page.
    pub fn reset_cache(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.key_pages, 0)?;
        self.arena.fill(stream, regions.value_pages, 0)?;
        Ok(())
    }

    /// Replays one complete transformer-layer route for exact `B=1..=8`.
    pub fn replay_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this owner retains the arena and modules captured by every graph.
        unsafe { self.draft_graphs[batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays one prime-only cache append for exact `K=1..=4`.
    pub fn replay_prime(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        // SAFETY: this owner retains the arena and modules captured by every graph.
        unsafe { self.prime_graphs[tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays a causal realignment with prime-only prefix and final full row.
    pub fn replay_realign(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        // SAFETY: this owner retains the arena and modules captured by every graph.
        unsafe { self.realign_graphs[tokens - 1].launch(stream) }?;
        Ok(())
    }

    /// Replays one exact `T=32,64,128` prompt cache-prime graph.
    pub fn replay_prompt_prime(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        let route = prompt_route(rows)?;
        // SAFETY: this owner retains the arena and modules captured by every graph.
        unsafe { self.prompt_graphs[route].launch(stream) }?;

        Ok(())
    }

    /// Reads active final-normalized BF16 rows for shared endpoint projection.
    pub fn read_final_normalized(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("Qwen3.6 MTP normalized values", rows, Qwen36Moe35B::HIDDEN)?;
        Ok(self.arena.copy_prefix_to_host(
            stream,
            self.layout.regions().final_normalized,
            values,
        )?)
    }

    /// Reads active final draft-residual BF16 rows.
    pub fn read_residual_output(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("Qwen3.6 MTP residual values", rows, Qwen36Moe35B::HIDDEN)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().residual_output, values)?)
    }

    /// CUDA context shared by all owner allocations and graphs.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every route.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Exact unchanged source-BF16 MTP weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact represented E4M3 short-cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete allocation, including alignment padding.
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
        MAX_BATCH + 2 * REALIGN_ROUTES + PROMPT_ROUTES.len()
    }

    #[cfg(feature = "qualification")]
    fn pointers(&self) -> GpuResult<Pointers> {
        Pointers::bind(&self.arena, self.layout.regions())
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete transformer-layer route eagerly.
    pub fn launch_eager_draft(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_full(stream, batch, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete B=1 route at an existing maximum-B row offset.
    pub fn launch_eager_draft_row(&self, stream: &CudaStream, row: usize) -> EngineResult<()> {
        if row >= MAX_BATCH {
            return Err(EngineError::route(format!(
                "Qwen3.6 MTP qualification row {row} is outside 0..{MAX_BATCH}"
            )));
        }
        launch_full(stream, 1, self.ops(), self.pointers()?.offset_rows(row))?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one prime-only route eagerly.
    pub fn launch_eager_prime(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        launch_prime(stream, tokens, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one exact prompt-prime route eagerly.
    pub fn launch_eager_prompt_prime(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        require_prompt(rows)?;
        launch_prime(stream, rows, self.ops(), self.pointers()?)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches one causal realignment route eagerly.
    pub fn launch_eager_realign(&self, stream: &CudaStream, tokens: usize) -> EngineResult<()> {
        require_realign(tokens)?;
        launch_realign(stream, tokens, self.ops(), self.pointers()?)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one production graph for direct benchmark registration.
    pub fn qualification_draft_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.draft_graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete routes for intrinsic timing.
    pub fn qualification_repeated_draft_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.6 MTP layer graph requires at least one operation",
            ));
        }
        let pointers = self.pointers()?;
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
        Ok(self.pointers()?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Samples the beginning, middle, and end of every immutable weight plane.
    pub fn qualification_immutable_samples(&self, stream: &CudaStream) -> EngineResult<Vec<u16>> {
        let regions = self.layout.regions();
        let mut samples = Vec::with_capacity(17 * 24);
        for region in [
            regions.embedding_norm,
            regions.hidden_norm,
            regions.input_projection,
            regions.input_norm,
            regions.qkv_weight,
            regions.query_norm,
            regions.key_norm,
            regions.attention_output_weight,
            regions.post_attention_norm,
            regions.router_weight,
            regions.routed_gate_up_weight,
            regions.routed_down_weight,
            regions.shared_gate_weight,
            regions.shared_up_weight,
            regions.shared_down_weight,
            regions.shared_expert_gate_weight,
            regions.final_norm,
        ] {
            samples.extend(sample_region(&self.arena, stream, region)?);
        }

        Ok(samples)
    }

    #[cfg(feature = "qualification")]
    /// Fills every mutable non-cache output plane with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.normalized_embedding,
            regions.normalized_hidden,
            regions.residual,
            regions.attention_normalized,
            regions.qkv,
            regions.attention_activation,
            regions.attention_branch,
            regions.post_attention_residual,
            regions.moe_normalized,
            regions.router_logits,
            regions.expert_indices,
            regions.routing_weights,
            regions.expert_intermediate,
            regions.expert_output,
            regions.shared_gate_output,
            regions.moe_branch,
            regions.residual_output,
            regions.final_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        self.arena.fill(stream, regions.query, byte)?;
        self.arena.fill(stream, regions.attention, byte)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every externally observable mutable seam and route metadata plane.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36MtpLayerObservables> {
        let regions = self.layout.regions();
        Ok(Qwen36MtpLayerObservables {
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
            moe_normalized: self.arena.copy_to_host(stream, regions.moe_normalized)?,
            router_logits: self.arena.copy_to_host(stream, regions.router_logits)?,
            expert_indices: self.arena.copy_to_host(stream, regions.expert_indices)?,
            routing_weights: self.arena.copy_to_host(stream, regions.routing_weights)?,
            expert_intermediate: self
                .arena
                .copy_to_host(stream, regions.expert_intermediate)?,
            expert_output: self.arena.copy_to_host(stream, regions.expert_output)?,
            shared_gate_output: self
                .arena
                .copy_to_host(stream, regions.shared_gate_output)?,
            moe_branch: self.arena.copy_to_host(stream, regions.moe_branch)?,
            residual_output: self.arena.copy_to_host(stream, regions.residual_output)?,
            final_normalized: self.arena.copy_to_host(stream, regions.final_normalized)?,
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
            router: &self._router,
            experts: &self._experts,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable Qwen3.6 MTP layer state exposed only to qualification.
pub struct Qwen36MtpLayerObservables {
    /// Input embedding rows.
    pub embedding: Vec<u16>,
    /// Aligned target-hidden rows.
    pub target_hidden: Vec<u16>,
    /// Pre-fusion normalized embedding rows.
    pub normalized_embedding: Vec<u16>,
    /// Pre-fusion normalized target-hidden rows.
    pub normalized_hidden: Vec<u16>,
    /// BF16 fusion-projection residual rows.
    pub residual: Vec<u16>,
    /// Pre-attention normalized rows.
    pub attention_normalized: Vec<u16>,
    /// Gathered query/gate, key, and value rows.
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
    /// Complete represented E4M3 key cache.
    pub key_pages: Vec<u8>,
    /// Complete represented E4M3 value cache.
    pub value_pages: Vec<u8>,
    /// FP32 paged-GQA output, gated in place.
    pub attention: Vec<f32>,
    /// Represented BF16 gated attention activation.
    pub attention_activation: Vec<u16>,
    /// Source-BF16 attention output branch.
    pub attention_branch: Vec<u16>,
    /// Residual after the attention branch.
    pub post_attention_residual: Vec<u16>,
    /// Pre-MoE normalized rows.
    pub moe_normalized: Vec<u16>,
    /// Router logits for every expert.
    pub router_logits: Vec<u16>,
    /// Selected top-eight expert indices.
    pub expert_indices: Vec<u16>,
    /// Renormalized top-eight routing weights.
    pub routing_weights: Vec<u16>,
    /// Routed and shared expert SwiGLU values.
    pub expert_intermediate: Vec<u16>,
    /// Routed and shared expert output values.
    pub expert_output: Vec<u16>,
    /// Per-row shared-expert gate values.
    pub shared_gate_output: Vec<u16>,
    /// Combined routed and shared MoE branch.
    pub moe_branch: Vec<u16>,
    /// Final draft residual rows.
    pub residual_output: Vec<u16>,
    /// Final-normalized rows for the shared endpoint.
    pub final_normalized: Vec<u16>,
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
        .map_err(|_| EngineError::layout("Qwen3.6 MTP draft graph inventory has wrong cardinality"))
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
        .map_err(|_| EngineError::layout("Qwen3.6 MTP prime graph inventory has wrong cardinality"))
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
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.6 MTP realignment graph inventory has wrong cardinality")
    })
}

fn capture_prompt_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; PROMPT_ROUTES.len()]> {
    let mut graphs = Vec::with_capacity(PROMPT_ROUTES.len());
    for rows in PROMPT_ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            launch_prime(stream, rows, ops, pointers)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.6 MTP prompt-prime graph inventory has wrong cardinality")
    })
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
            pointers.table_stride,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            Qwen36Moe35B::FP8_CACHE_SCALE,
            Qwen36Moe35B::FP8_CACHE_SCALE,
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
    // The page table covers every published causal length, and exact dispatch
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
            pointers.table_stride,
            pointers.lengths,
            pointers.attention,
            Qwen36Moe35B::FP8_CACHE_SCALE,
            Qwen36Moe35B::FP8_CACHE_SCALE,
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
            pointers.moe_normalized,
        )?;
        ops.router.launch(
            stream,
            rows,
            pointers.moe_normalized,
            pointers.router_weight,
            pointers.router_logits,
            pointers.expert_indices,
            pointers.routing_weights,
        )?;
        ops.experts.launch(
            stream,
            rows,
            pointers.moe_normalized,
            pointers.expert_indices,
            pointers.routing_weights,
            pointers.routed_gate_up_weight,
            pointers.routed_down_weight,
            pointers.shared_gate_weight,
            pointers.shared_up_weight,
            pointers.shared_down_weight,
            pointers.shared_expert_gate_weight,
            pointers.expert_intermediate,
            pointers.expert_output,
            pointers.shared_gate_output,
            pointers.moe_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            rows,
            pointers.post_attention_residual,
            pointers.moe_branch,
            pointers.final_norm,
            pointers.residual_output,
            pointers.final_normalized,
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
            "Qwen3.6 MTP draft batch {batch} is outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_realign(tokens: usize) -> EngineResult<()> {
    if !(1..=REALIGN_ROUTES).contains(&tokens) {
        return Err(EngineError::route(format!(
            "Qwen3.6 MTP realignment width {tokens} is outside 1..={REALIGN_ROUTES}"
        )));
    }
    Ok(())
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&rows) {
        return Err(EngineError::route(format!(
            "Qwen3.6 MTP rows {rows} are outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_prompt(rows: usize) -> EngineResult<()> {
    if !PROMPT_ROUTES.contains(&rows) {
        return Err(EngineError::route(format!(
            "Qwen3.6 MTP prompt rows {rows} are outside 32,64,128"
        )));
    }
    Ok(())
}

fn prompt_route(rows: usize) -> EngineResult<usize> {
    PROMPT_ROUTES
        .iter()
        .position(|&route| route == rows)
        .ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.6 MTP prompt rows {rows} are outside 32,64,128"
            ))
        })
}

fn require_rows_or_prompt(rows: usize) -> EngineResult<()> {
    if (1..=MAX_BATCH).contains(&rows) || PROMPT_ROUTES.contains(&rows) {
        Ok(())
    } else {
        Err(EngineError::route(format!(
            "Qwen3.6 MTP rows {rows} are outside 1..={MAX_BATCH},32,64,128"
        )))
    }
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(feature = "qualification")]
fn sample_region(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: tuisko_gpu::ArenaRegion<u16>,
) -> EngineResult<Vec<u16>> {
    let mut samples = Vec::with_capacity(24);
    for start in [0, region.len() / 2 - 4, region.len() - 8] {
        samples.extend(arena.copy_slice_to_host(stream, region, start, 8)?);
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::{
        PROMPT_ROUTES, REALIGN_ROUTES, prompt_route, require_batch, require_prompt,
        require_realign, require_rows,
    };
    use crate::{EngineErrorCode, MAX_BATCH};

    #[test]
    fn qwen36_mtp_route_tables_are_exact() {
        for route in 1..=MAX_BATCH {
            require_batch(route).unwrap();
            require_rows(route).unwrap();
        }
        for route in 1..=REALIGN_ROUTES {
            require_realign(route).unwrap();
        }
        for route in [0, 9, 16, usize::MAX] {
            for error in [require_batch(route), require_rows(route)] {
                assert_eq!(error.unwrap_err().code(), Some(EngineErrorCode::Route));
            }
        }
        for route in [0, 5, 8, usize::MAX] {
            assert_eq!(
                require_realign(route).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
        for (index, route) in PROMPT_ROUTES.into_iter().enumerate() {
            require_prompt(route).unwrap();
            assert_eq!(prompt_route(route).unwrap(), index);
        }
        for route in [0, 8, 16, 31, 33, 63, 65, 127, 129, usize::MAX] {
            assert_eq!(
                require_prompt(route).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
            assert_eq!(
                prompt_route(route).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
    }
}

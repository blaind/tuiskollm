//! Resident source-backed Qwen3.5 MTP transformer layer.

use crate::common::graph::capture_batch_graphs;
use crate::common::math::product;
use crate::qwen35::mtp_layer_layout::{
    QWEN35_MTP_PHYSICAL_PAGES, QWEN35_MTP_PROMPT_ROWS, QWEN35_MTP_TABLE_STRIDE,
    Qwen35MtpLayerRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen35MtpLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen35MtpBf16AttentionOutputOp, Qwen35MtpBf16FusionOp, Qwen35MtpBf16MlpOp,
    Qwen35MtpBf16PagedGqaOp, Qwen35MtpBf16QkPrepareOp, Qwen35MtpBf16QkvOp, Qwen35ResidualNormOp,
};
use tuisko_model::{Arch, CheckpointSnapshot, MtpBindings, Qwen35_9B};

const ROTARY_PAIRS: usize = 32;
const REALIGN_ROUTES: usize = 4;
const PROMPT_ROUTES: [usize; 3] = [32, 64, QWEN35_MTP_PROMPT_ROWS];

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35MtpKvBinding {
    pub(crate) block_tables: u64,
    pub(crate) key_pages: u64,
    pub(crate) value_pages: u64,
    pub(crate) table_stride: usize,
    pub(crate) context_capacity: usize,
}

/// One exact source-backed Qwen3.5 MTP layer without a duplicate text endpoint.
pub struct Qwen35MtpLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    draft_graphs: [CudaGraph; MAX_BATCH],
    prime_graphs: [CudaGraph; REALIGN_ROUTES],
    realign_graphs: [CudaGraph; REALIGN_ROUTES],
    arena: DeviceArena,
    fusion: Qwen35MtpBf16FusionOp,
    norm: Qwen35ResidualNormOp,
    qkv: Qwen35MtpBf16QkvOp,
    qk_prepare: Qwen35MtpBf16QkPrepareOp,
    paged_gqa: Qwen35MtpBf16PagedGqaOp,
    attention_output: Qwen35MtpBf16AttentionOutputOp,
    mlp: Qwen35MtpBf16MlpOp,
    context: Arc<CudaContext>,
    layout: Qwen35MtpLayerLayout,
    base_address: u64,
    kv_binding: Option<Qwen35MtpKvBinding>,
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
    table_stride: usize,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: Qwen35MtpLayerRegions) -> GpuResult<Self> {
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
            table_stride: QWEN35_MTP_TABLE_STRIDE,
        })
    }

    fn bind_with_kv(
        arena: &DeviceArena,
        regions: Qwen35MtpLayerRegions,
        binding: Option<Qwen35MtpKvBinding>,
    ) -> GpuResult<Self> {
        let mut pointers = Self::bind(arena, regions)?;
        if let Some(binding) = binding {
            pointers.block_tables = binding.block_tables as *const u32;
            pointers.key_pages = binding.key_pages as *mut u16;
            pointers.value_pages = binding.value_pages as *mut u16;
            pointers.table_stride = binding.table_stride;
        }

        Ok(pointers)
    }

    fn offset_rows(self, rows: usize) -> Self {
        Self {
            embedding: self.embedding.wrapping_add(rows * Qwen35_9B::HIDDEN),
            target_hidden: self.target_hidden.wrapping_add(rows * Qwen35_9B::HIDDEN),
            normalized_embedding: self
                .normalized_embedding
                .wrapping_add(rows * Qwen35_9B::HIDDEN),
            normalized_hidden: self
                .normalized_hidden
                .wrapping_add(rows * Qwen35_9B::HIDDEN),
            residual: self.residual.wrapping_add(rows * Qwen35_9B::HIDDEN),
            attention_normalized: self
                .attention_normalized
                .wrapping_add(rows * Qwen35_9B::HIDDEN),
            qkv: self.qkv.wrapping_add(rows * Qwen35_9B::ATTENTION_QKV_ROWS),
            rope_cos: self.rope_cos.wrapping_add(rows * ROTARY_PAIRS),
            rope_sin: self.rope_sin.wrapping_add(rows * ROTARY_PAIRS),
            table_rows: self.table_rows.wrapping_add(rows),
            cache_positions: self.cache_positions.wrapping_add(rows),
            lengths: self.lengths.wrapping_add(rows),
            query: self
                .query
                .wrapping_add(rows * Qwen35_9B::ATTENTION_OUTPUT_COLUMNS),
            attention: self
                .attention
                .wrapping_add(rows * Qwen35_9B::ATTENTION_OUTPUT_COLUMNS),
            attention_activation: self
                .attention_activation
                .wrapping_add(rows * Qwen35_9B::ATTENTION_OUTPUT_COLUMNS),
            attention_branch: self.attention_branch.wrapping_add(rows * Qwen35_9B::HIDDEN),
            post_attention_residual: self
                .post_attention_residual
                .wrapping_add(rows * Qwen35_9B::HIDDEN),
            mlp_normalized: self.mlp_normalized.wrapping_add(rows * Qwen35_9B::HIDDEN),
            swiglu: self.swiglu.wrapping_add(rows * Qwen35_9B::INTERMEDIATE),
            mlp_branch: self.mlp_branch.wrapping_add(rows * Qwen35_9B::HIDDEN),
            residual_output: self.residual_output.wrapping_add(rows * Qwen35_9B::HIDDEN),
            final_normalized: self.final_normalized.wrapping_add(rows * Qwen35_9B::HIDDEN),
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
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    fusion: &'a Qwen35MtpBf16FusionOp,
    norm: &'a Qwen35ResidualNormOp,
    qkv: &'a Qwen35MtpBf16QkvOp,
    qk_prepare: &'a Qwen35MtpBf16QkPrepareOp,
    paged_gqa: &'a Qwen35MtpBf16PagedGqaOp,
    attention_output: &'a Qwen35MtpBf16AttentionOutputOp,
    mlp: &'a Qwen35MtpBf16MlpOp,
}

impl Qwen35MtpLayerProgram {
    /// Loads the complete Qwen3.5 MTP source family and captures exact routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: &CheckpointSnapshot<Qwen35_9B>,
    ) -> EngineResult<Self> {
        Self::from_snapshot_inner(context, snapshot, None)
    }

    /// Loads the layer while borrowing one stable, externally owned MTP cache.
    ///
    /// # Safety
    /// The binding owner must outlive this program and every replay of its graphs.
    pub(crate) unsafe fn from_snapshot_with_kv(
        context: &Arc<CudaContext>,
        snapshot: &CheckpointSnapshot<Qwen35_9B>,
        binding: Qwen35MtpKvBinding,
    ) -> EngineResult<Self> {
        if binding.table_stride == 0 || binding.context_capacity == 0 {
            return Err(EngineError::layout(
                "external Qwen3.5 MTP cache requires nonzero table stride and capacity",
            ));
        }
        Self::from_snapshot_inner(context, snapshot, Some(binding))
    }

    fn from_snapshot_inner(
        context: &Arc<CudaContext>,
        snapshot: &CheckpointSnapshot<Qwen35_9B>,
        kv_binding: Option<Qwen35MtpKvBinding>,
    ) -> EngineResult<Self> {
        let mtp = MtpBindings::bind(snapshot)?;
        let qkv = mtp.materialize_qkv()?;
        let layout = if kv_binding.is_some() {
            Qwen35MtpLayerLayout::build_for_external_cache()?
        } else {
            Qwen35MtpLayerLayout::build()?
        };
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
        if kv_binding.is_none() {
            arena.copy_from_host(
                &stream,
                regions.block_tables,
                &(0..QWEN35_MTP_PHYSICAL_PAGES as u32).collect::<Vec<_>>(),
            )?;
        }

        let fusion = Qwen35MtpBf16FusionOp::new(context)?;
        let norm = Qwen35ResidualNormOp::new(context)?;
        let qkv_op = Qwen35MtpBf16QkvOp::new(context)?;
        let qk_prepare = Qwen35MtpBf16QkPrepareOp::new(context)?;
        let paged_gqa = Qwen35MtpBf16PagedGqaOp::new(context)?;
        let attention_output = Qwen35MtpBf16AttentionOutputOp::new(context)?;
        let mlp = Qwen35MtpBf16MlpOp::new(context)?;
        let pointers = Pointers::bind_with_kv(&arena, regions, kv_binding)?;
        let ops = Ops {
            fusion: &fusion,
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            mlp: &mlp,
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
            fusion,
            norm,
            qkv: qkv_op,
            qk_prepare,
            paged_gqa,
            attention_output,
            mlp,
            context: Arc::clone(context),
            layout,
            base_address,
            kv_binding,
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
        let values = product("Qwen3.5 MTP input values", rows, Qwen35_9B::HIDDEN)?;
        if embedding.len() != values || target_hidden.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP inputs have {}/{} values, expected {values} for rows={rows}",
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
        let values = product("Qwen3.5 MTP embedding values", rows, Qwen35_9B::HIDDEN)?;
        if embedding.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP embedding input has {} values, expected {values} for rows={rows}",
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
                "Qwen3.5 MTP compact route has {} positions and {} slots",
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
                        "Qwen3.5 MTP compact slot {slot} is repeated or outside 0..{MAX_BATCH}"
                    )));
                }
                seen[slot] = true;
                u32::try_from(slot).map_err(|_| EngineError::layout("Qwen3.5 MTP slot exceeds u32"))
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
                "Qwen3.5 MTP prompt slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        let end = first_position
            .checked_add(rows)
            .ok_or_else(|| EngineError::route("Qwen3.5 MTP prompt positions overflow"))?;
        let positions = (first_position..end)
            .map(|position| {
                u32::try_from(position)
                    .map_err(|_| EngineError::route("Qwen3.5 MTP prompt position exceeds u32"))
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
                "Qwen3.5 MTP realignment slot {slot} is outside 0..{MAX_BATCH}"
            )));
        }
        if positions
            .windows(2)
            .any(|pair| pair[1] != pair[0].saturating_add(1))
        {
            return Err(EngineError::route(
                "Qwen3.5 MTP realignment positions must form one contiguous sequence",
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
                "Qwen3.5 MTP route metadata has {}/{} rows, expected {rows}",
                positions.len(),
                table_rows.len()
            )));
        }
        let rotary_values = product("Qwen3.5 MTP rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP rotary planes must each have {rotary_values} values for rows={rows}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= self.context_capacity() {
                    return Err(EngineError::route(format!(
                        "Qwen3.5 MTP cache position {position} exceeds the {}-token capacity",
                        self.context_capacity()
                    )));
                }
                position
                    .checked_add(1)
                    .ok_or_else(|| EngineError::route("Qwen3.5 MTP cache length overflows"))
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
                "Qwen3.5 MTP cache planes must each have {} BF16 values",
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

    /// Reads active final-normalized BF16 rows for shared endpoint projection.
    pub fn read_final_normalized(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("Qwen3.5 MTP normalized values", rows, Qwen35_9B::HIDDEN)?;
        Ok(self.arena.copy_prefix_to_host(
            stream,
            self.layout.regions().final_normalized,
            values,
        )?)
    }

    /// Reads active final draft-residual BF16 rows.
    pub fn read_residual_output(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("Qwen3.5 MTP residual values", rows, Qwen35_9B::HIDDEN)?;
        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().residual_output, values)?)
    }

    pub(crate) fn read_residual_output_into(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        require_rows(rows)?;
        let values = product("Qwen3.5 MTP residual values", rows, Qwen35_9B::HIDDEN)?;
        if destination.len() != values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 MTP residual destination has {} values, expected {values}",
                destination.len()
            )));
        }
        self.arena.copy_prefix_to_host_slice(
            stream,
            self.layout.regions().residual_output,
            destination,
        )?;

        Ok(())
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

    /// Exact represented BF16 short-cache bytes.
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
        match self.kv_binding {
            Some(binding) => binding.context_capacity,
            None => self.layout.context_capacity(),
        }
    }

    /// Number of immutable exact graph entries.
    pub const fn graph_count(&self) -> usize {
        MAX_BATCH + 2 * REALIGN_ROUTES
    }

    fn pointers(&self) -> GpuResult<Pointers> {
        Pointers::bind_with_kv(&self.arena, self.layout.regions(), self.kv_binding)
    }

    pub(crate) fn final_normalized_address(&self) -> GpuResult<*const u16> {
        Ok(self.pointers()?.final_normalized.cast_const())
    }

    pub(crate) fn target_hidden_address(&self) -> GpuResult<*const u16> {
        Ok(self.pointers()?.target_hidden)
    }

    pub(crate) fn residual_output_address(&self) -> GpuResult<*const u16> {
        Ok(self.pointers()?.residual_output.cast_const())
    }

    /// Launches a draft route from one external stable target-hidden plane.
    ///
    /// # Safety
    /// `target_hidden` must cover `batch * 4096` BF16 values until completion.
    pub(crate) unsafe fn launch_draft_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        target_hidden: *const u16,
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let mut pointers = self.pointers()?;
        pointers.target_hidden = target_hidden;
        launch_full(stream, batch, self.ops(), pointers)?;

        Ok(())
    }

    /// Launches a prompt-prime tile from the target's stable residual plane.
    ///
    /// # Safety
    /// `target_hidden` must cover `rows * 4096` BF16 values until completion.
    pub(crate) unsafe fn launch_prompt_prime_from(
        &self,
        stream: &CudaStream,
        rows: usize,
        target_hidden: *const u16,
    ) -> EngineResult<()> {
        require_prompt(rows)?;
        let mut pointers = self.pointers()?;
        pointers.target_hidden = target_hidden;
        launch_prime(stream, rows, self.ops(), pointers)?;

        Ok(())
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
                "Qwen3.5 MTP qualification row {row} is outside 0..{MAX_BATCH}"
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
                "repeated Qwen3.5 MTP layer graph requires at least one operation",
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
            regions.mlp_normalized,
            regions.swiglu,
            regions.mlp_branch,
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
    ) -> EngineResult<Qwen35MtpLayerObservables> {
        let regions = self.layout.regions();
        Ok(Qwen35MtpLayerObservables {
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
        })
    }

    fn ops(&self) -> Ops<'_> {
        Ops {
            fusion: &self.fusion,
            norm: &self.norm,
            qkv: &self.qkv,
            qk_prepare: &self.qk_prepare,
            paged_gqa: &self.paged_gqa,
            attention_output: &self.attention_output,
            mlp: &self.mlp,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable Qwen3.5 MTP layer state exposed only to qualification.
pub struct Qwen35MtpLayerObservables {
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
    /// Complete represented BF16 key cache.
    pub key_pages: Vec<u16>,
    /// Complete represented BF16 value cache.
    pub value_pages: Vec<u16>,
    /// FP32 paged-GQA output, gated in place.
    pub attention: Vec<f32>,
    /// Represented BF16 gated attention activation.
    pub attention_activation: Vec<u16>,
    /// Source-BF16 attention output branch.
    pub attention_branch: Vec<u16>,
    /// Residual after the attention branch.
    pub post_attention_residual: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub mlp_normalized: Vec<u16>,
    /// Represented BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Source-BF16 MLP down branch.
    pub mlp_branch: Vec<u16>,
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
    capture_batch_graphs(
        stream,
        "Qwen3.5 MTP draft graph inventory has wrong cardinality",
        |batch| launch_full(stream, batch, ops, pointers),
    )
}

fn capture_prime_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; REALIGN_ROUTES]> {
    capture_batch_graphs(
        stream,
        "Qwen3.5 MTP prime graph inventory has wrong cardinality",
        |tokens| launch_prime(stream, tokens, ops, pointers),
    )
}

fn capture_realign_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; REALIGN_ROUTES]> {
    capture_batch_graphs(
        stream,
        "Qwen3.5 MTP realignment graph inventory has wrong cardinality",
        |tokens| launch_realign(stream, tokens, ops, pointers),
    )
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
            "Qwen3.5 MTP draft batch {batch} is outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_realign(tokens: usize) -> EngineResult<()> {
    if !(1..=REALIGN_ROUTES).contains(&tokens) {
        return Err(EngineError::route(format!(
            "Qwen3.5 MTP realignment width {tokens} is outside 1..={REALIGN_ROUTES}"
        )));
    }
    Ok(())
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&rows) {
        return Err(EngineError::route(format!(
            "Qwen3.5 MTP rows {rows} are outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_prompt(rows: usize) -> EngineResult<()> {
    if !PROMPT_ROUTES.contains(&rows) {
        return Err(EngineError::route(format!(
            "Qwen3.5 MTP prompt rows {rows} are outside 32,64,128"
        )));
    }
    Ok(())
}

fn require_rows_or_prompt(rows: usize) -> EngineResult<()> {
    if (1..=MAX_BATCH).contains(&rows) || PROMPT_ROUTES.contains(&rows) {
        Ok(())
    } else {
        Err(EngineError::route(format!(
            "Qwen3.5 MTP rows {rows} are outside 1..={MAX_BATCH},32,64,128"
        )))
    }
}

fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(EngineError::layout(
            "Qwen3.5 MTP BF16 source plane has odd byte length",
        ));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{REALIGN_ROUTES, require_batch, require_realign, require_rows};
    use crate::{EngineErrorCode, MAX_BATCH};

    #[test]
    fn qwen35_mtp_route_tables_are_exact() {
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
    }
}

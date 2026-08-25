//! Resident composition of every Qwen3.5 text layer and endpoint.

use crate::qwen35_long_context_kv::Qwen35AttentionKvBinding;
use crate::{
    EngineError, EngineResult, MAX_BATCH, Qwen35FullAttentionLayerLayout,
    Qwen35FullAttentionLayerProgram, Qwen35GdnLayerLayout, Qwen35GdnLayerProgram,
    Qwen35LongContextKvLayout, Qwen35LongContextKvProgram, Qwen35TextEndpointLayout,
    Qwen35TextEndpointProgram,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const ROTARY_PAIRS: usize = 32;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const MAX_PREFILL_ROWS: usize = 128;

/// Exact from-empty prompt graph selected by matching state uploads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the Qwen3.5 prefill route must be replayed with the state that selected it"]
pub struct Qwen35ResidentPrefillRoute {
    tokens: usize,
}

impl Qwen35ResidentPrefillRoute {
    /// Number of contiguous prompt tokens represented by this route.
    pub const fn tokens(self) -> usize {
        self.tokens
    }
}

/// Exact source route owned by one Qwen3.5 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35ResidentLayerKind {
    /// Gated-delta mixer followed by the ModelOpt NVFP4 MLP.
    Gdn,
    /// Full attention followed by the ModelOpt NVFP4 MLP.
    FullAttention,
}

/// Exact byte accounting for the initial address-stable resident composition.
#[derive(Clone, Debug)]
pub struct Qwen35ResidentModelLayout {
    layers: [Qwen35ResidentLayerKind; Qwen35_9B::LAYERS],
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
    arena_bytes: usize,
}

impl Qwen35ResidentModelLayout {
    /// Accounts all 32 layer owners and the source-backed BF16 endpoint.
    pub fn build() -> EngineResult<Self> {
        require_geometry()?;
        let gdn = Qwen35GdnLayerLayout::build()?;
        let attention = Qwen35FullAttentionLayerLayout::build()?;
        let long_context = Qwen35LongContextKvLayout::build()?;
        let endpoint = Qwen35TextEndpointLayout::build()?;
        let layers = core::array::from_fn(layer_kind);
        let gdn_layers = layers
            .iter()
            .filter(|&&kind| kind == Qwen35ResidentLayerKind::Gdn)
            .count();
        let attention_layers = Qwen35_9B::LAYERS - gdn_layers;
        let resident_weight_bytes = sum_products(
            "Qwen3.5 resident weight bytes",
            &[
                (gdn_layers, gdn.resident_weight_bytes()),
                (attention_layers, attention.resident_weight_bytes()),
                (1, endpoint.resident_weight_bytes()),
            ],
        )?;
        let cache_bytes = checked_sum(
            "Qwen3.5 resident BF16 cache bytes",
            product(
                "Qwen3.5 resident short BF16 cache bytes",
                attention_layers,
                attention.cache_bytes(),
            )?,
            long_context.cache_bytes(),
        )?;
        let workspace_bytes = sum_products(
            "Qwen3.5 resident workspace bytes",
            &[
                (gdn_layers, gdn.workspace_bytes()),
                (attention_layers, attention.workspace_bytes()),
                (1, endpoint.workspace_bytes()),
                (1, long_context.block_table_bytes()),
            ],
        )?;
        let arena_bytes = sum_products(
            "Qwen3.5 resident arena bytes",
            &[
                (gdn_layers, gdn.arena_bytes()),
                (attention_layers, attention.arena_bytes()),
                (1, endpoint.arena_bytes()),
                (1, long_context.arena_bytes()),
            ],
        )?;

        Ok(Self {
            layers,
            resident_weight_bytes,
            cache_bytes,
            workspace_bytes,
            arena_bytes,
        })
    }

    /// Number of exact decoder layers.
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Source route for one layer, or `None` outside `0..32`.
    pub fn layer_kind(&self, layer: usize) -> Option<Qwen35ResidentLayerKind> {
        self.layers.get(layer).copied()
    }

    /// Source-backed decoder, norm, and BF16 LM-head bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// BF16 K/V bytes in the shared pool and retained layer-local fallback planes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable per-layer state and working bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Complete device allocation across the 32 layer arenas and endpoint arena.
    pub const fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }

    /// All represented bytes, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Aggregate alignment padding across the 34 stable arenas.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes - self.owner_bytes()
    }

    /// Number of independently allocated, address-stable device arenas.
    pub const fn arena_count(&self) -> usize {
        Qwen35_9B::LAYERS + 2
    }

    /// mmap-backed BF16 embeddings intentionally excluded from device residency.
    pub fn source_mapped_embedding_bytes(&self) -> EngineResult<usize> {
        product(
            "Qwen3.5 source-mapped embedding bytes",
            product(
                "Qwen3.5 source-mapped embedding elements",
                Qwen35_9B::VOCAB,
                Qwen35_9B::HIDDEN,
            )?,
            size_of::<u16>(),
        )
    }

    /// Maximum context admitted by the pinned Qwen3.5 config.
    pub const fn context_capacity(&self) -> usize {
        crate::QWEN35_MAX_CONTEXT_TOKENS
    }
}

enum ResidentLayer {
    Gdn(Box<Qwen35GdnLayerProgram>),
    FullAttention(Box<Qwen35FullAttentionLayerProgram>),
}

impl ResidentLayer {
    fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
        layer: usize,
        kv_binding: Option<Qwen35AttentionKvBinding>,
    ) -> EngineResult<Self> {
        match layer_kind(layer) {
            Qwen35ResidentLayerKind::Gdn => Ok(Self::Gdn(Box::new(
                Qwen35GdnLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
            Qwen35ResidentLayerKind::FullAttention => {
                let binding = kv_binding.ok_or_else(|| {
                    EngineError::layout(format!(
                        "Qwen3.5 full-attention layer {layer} has no shared KV binding"
                    ))
                })?;
                // SAFETY: the resident program declares its graph-bearing layers
                // before the shared KV owner, so every captured address outlives them.
                let program = unsafe {
                    Qwen35FullAttentionLayerProgram::from_snapshot_with_kv(
                        context, snapshot, layer, binding,
                    )?
                };
                Ok(Self::FullAttention(Box::new(program)))
            }
        }
    }

    fn reset(&self, stream: &CudaStream) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.reset_state(stream),
            Self::FullAttention(_) => Ok(()),
        }
    }

    fn reset_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.reset_slot(stream, slot),
            Self::FullAttention(_) => Ok(()),
        }
    }

    fn gdn(&self) -> Option<&Qwen35GdnLayerProgram> {
        match self {
            Self::Gdn(program) => Some(program),
            Self::FullAttention(_) => None,
        }
    }

    fn load_slot_routes(&self, stream: &CudaStream, slots: &[usize]) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.load_slot_routes(stream, slots),
            Self::FullAttention(program) => program.load_slot_routes(stream, slots),
        }
    }

    fn load_decode_state(
        &self,
        stream: &CudaStream,
        batch: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        match self {
            Self::Gdn(_) => Ok(()),
            Self::FullAttention(program) => {
                program.load_decode_state(stream, batch, positions, rope_cos, rope_sin)
            }
        }
    }

    fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.load_prefill_slot(stream, slot),
            Self::FullAttention(program) => program.load_prefill_slot_state_at(
                stream,
                tokens,
                slot,
                first_position,
                rope_cos,
                rope_sin,
            ),
        }
    }

    fn load_verify_state(
        &self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.load_verify_slot(stream, slot),
            Self::FullAttention(program) => {
                program.load_verify_state(stream, rows, slot, first_position, rope_cos, rope_sin)
            }
        }
    }

    fn load_residual(&self, stream: &CudaStream, rows: usize, values: &[u16]) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.load_residual(stream, rows, values),
            Self::FullAttention(program) => program.load_residual(stream, rows, values),
        }
    }

    fn input_address(&self) -> GpuResult<*const u16> {
        match self {
            Self::Gdn(program) => program.input_address(),
            Self::FullAttention(program) => program.input_address(),
        }
    }

    fn output_address(&self) -> GpuResult<*const u16> {
        match self {
            Self::Gdn(program) => program.output_address(),
            Self::FullAttention(program) => program.output_address(),
        }
    }

    /// # Safety
    /// `input` names the active BF16 residual rows in the shared CUDA context.
    unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        match self {
            Self::Gdn(program) => unsafe { program.launch_from(stream, rows, input) },
            Self::FullAttention(program) => unsafe { program.launch_from(stream, rows, input) },
        }
    }

    /// # Safety
    /// `input` names `rows * 4,096` retained BF16 values in this CUDA context.
    unsafe fn launch_verify_from(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        match self {
            Self::Gdn(program) => unsafe { program.launch_verify_from(stream, rows, input) },
            Self::FullAttention(program) => unsafe { program.launch_from(stream, rows, input) },
        }
    }

    fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        match self {
            Self::Gdn(program) => program.read_residual(stream, rows),
            Self::FullAttention(program) => program.read_residual(stream, rows),
        }
    }

    fn read_residual_into(
        &self,
        stream: &CudaStream,
        rows: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.read_residual_into(stream, rows, destination),
            Self::FullAttention(program) => program.read_residual_into(stream, rows, destination),
        }
    }

    fn base_address(&self) -> u64 {
        match self {
            Self::Gdn(program) => program.base_address(),
            Self::FullAttention(program) => program.base_address(),
        }
    }

    fn arena_bytes(&self) -> usize {
        match self {
            Self::Gdn(program) => program.arena_bytes(),
            Self::FullAttention(program) => program.arena_bytes(),
        }
    }
}

/// Every Qwen3.5 text layer and the BF16 endpoint held resident at stable addresses.
pub struct Qwen35ResidentModelProgram {
    // Drop whole-model graphs and layers before the shared KV arena they retain.
    graphs: [CudaGraph; MAX_BATCH],
    verify_graphs: [CudaGraph; 4],
    prefill_graphs: [CudaGraph; 3],
    layers: Vec<ResidentLayer>,
    long_context_kv: Qwen35LongContextKvProgram,
    endpoint: Qwen35TextEndpointProgram,
    prefill_embedding_stager: PinnedHostBuffer<u16>,
    context: Arc<CudaContext>,
    layout: Qwen35ResidentModelLayout,
}

impl Qwen35ResidentModelProgram {
    /// Loads all source weights and captures exact decode and native prefill graphs.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let layout = Qwen35ResidentModelLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let long_context_kv = Qwen35LongContextKvProgram::new(context)?;
        let mut layers = Vec::with_capacity(Qwen35_9B::LAYERS);
        let mut attention_layer = 0;
        for layer in 0..Qwen35_9B::LAYERS {
            let kv_binding = if layer_kind(layer) == Qwen35ResidentLayerKind::FullAttention {
                let binding = long_context_kv.layer_binding(attention_layer)?;
                attention_layer += 1;
                Some(binding)
            } else {
                None
            };
            layers.push(ResidentLayer::from_snapshot(
                context,
                Arc::clone(&snapshot),
                layer,
                kv_binding,
            )?);
        }
        if attention_layer != long_context_kv.layout().layers().len() {
            return Err(EngineError::layout(format!(
                "Qwen3.5 resident model bound {attention_layer} attention layers, shared KV owner has {}",
                long_context_kv.layout().layers().len()
            )));
        }
        let endpoint = Qwen35TextEndpointProgram::from_snapshot(context, snapshot)?;
        let prefill_embedding_stager =
            PinnedHostBuffer::zeroed(context, MAX_PREFILL_ROWS * Qwen35_9B::HIDDEN)
                .map_err(GpuError::from)?;
        for layer in &layers {
            layer.reset(&stream)?;
        }
        let graphs = capture_decode_routes(&stream, &layers, &endpoint)?;
        let verify_graphs = capture_verify_routes(&stream, &layers, &endpoint)?;
        let prefill_graphs = capture_prefill_routes(&stream, &layers, &endpoint)?;
        let program = Self {
            graphs,
            verify_graphs,
            prefill_graphs,
            layers,
            long_context_kv,
            endpoint,
            prefill_embedding_stager,
            context: Arc::clone(context),
            layout,
        };
        program.require_accounting()?;

        Ok(program)
    }

    /// Copies exact mmap-backed embedding rows into the stable endpoint input plane.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        self.endpoint.stage_embeddings(stream, token_ids)
    }

    /// Gathers one exact prompt tile into layer zero's stable input plane.
    pub fn stage_prefill_embeddings(
        &mut self,
        stream: &CudaStream,
        token_ids: &[u32],
    ) -> EngineResult<()> {
        require_prefill(token_ids.len())?;
        let active = product(
            "Qwen3.5 resident prefill embedding values",
            token_ids.len(),
            Qwen35_9B::HIDDEN,
        )?;
        self.endpoint
            .gather_embedding_rows(token_ids, &mut self.prefill_embedding_stager[..active])?;
        self.layers
            .first()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .load_residual(
                stream,
                token_ids.len(),
                &self.prefill_embedding_stager[..active],
            )
    }

    /// Updates decode positions and MRoPE values in every full-attention layer.
    pub fn load_decode_state(
        &self,
        stream: &CudaStream,
        batch: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        for layer in &self.layers {
            layer.load_decode_state(stream, batch, positions, rope_cos, rope_sin)?;
        }

        Ok(())
    }

    /// Maps compact decode rows to distinct physical persistent slots.
    pub fn load_slot_routes(&self, stream: &CudaStream, slots: &[usize]) -> EngineResult<()> {
        slot_rows(slots)?;
        for layer in &self.layers {
            layer.load_slot_routes(stream, slots)?;
        }

        Ok(())
    }

    /// Stages one contiguous `K=1..4` causal target span in a physical slot.
    pub(crate) fn load_verify_state(
        &self,
        stream: &CudaStream,
        rows: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_verify_rows(rows)?;
        require_slot(slot)?;
        let end = first_position
            .checked_add(rows)
            .ok_or_else(|| EngineError::route("Qwen3.5 verification range overflows"))?;
        let reserved_tokens = self.long_context_kv.slot_token_count(slot)?;
        if end > reserved_tokens {
            return Err(EngineError::route(format!(
                "Qwen3.5 verification positions {first_position}..{end} exceed slot {slot}'s {reserved_tokens} reserved tokens"
            )));
        }
        let rotary_values = product("Qwen3.5 verification rotary values", rows, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 verification rotary planes must each have {rotary_values} values for K={rows}"
            )));
        }
        for layer in &self.layers {
            layer.load_verify_state(stream, rows, slot, first_position, rope_cos, rope_sin)?;
        }

        Ok(())
    }

    /// Updates every attention layer for one contiguous from-empty prompt tile.
    pub fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen35ResidentPrefillRoute> {
        self.load_prefill_slot_state(stream, tokens, 0, rope_cos, rope_sin)
    }

    /// Updates every persistent layer for one from-empty prompt in `slot`.
    pub fn load_prefill_slot_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen35ResidentPrefillRoute> {
        self.load_prefill_slot_state_at(stream, tokens, slot, 0, rope_cos, rope_sin)
    }

    /// Updates every persistent layer for one exact prompt tile at an existing offset.
    pub fn load_prefill_slot_state_at(
        &self,
        stream: &CudaStream,
        tokens: usize,
        slot: usize,
        first_position: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen35ResidentPrefillRoute> {
        require_prefill(tokens)?;
        require_slot(slot)?;
        let context_tokens = first_position.checked_add(tokens).ok_or_else(|| {
            EngineError::route("Qwen3.5 resident prefill context length overflows")
        })?;
        let reserved_tokens = self.long_context_kv.slot_token_count(slot)?;
        if context_tokens > reserved_tokens {
            return Err(EngineError::route(format!(
                "Qwen3.5 resident prefill positions {first_position}..{context_tokens} exceed slot {slot}'s {reserved_tokens} reserved tokens"
            )));
        }
        let rotary_values = product(
            "Qwen3.5 resident prefill rotary values",
            tokens,
            ROTARY_PAIRS,
        )?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 resident prefill rotary planes must each have {rotary_values} values for T={tokens}"
            )));
        }
        for layer in &self.layers {
            layer.load_prefill_state(stream, tokens, slot, first_position, rope_cos, rope_sin)?;
        }

        Ok(Qwen35ResidentPrefillRoute { tokens })
    }

    /// Clears every GDN state and recycles all shared attention page-table rows.
    pub fn reset_state(&mut self, stream: &CudaStream) -> EngineResult<()> {
        for layer in &self.layers {
            layer.reset(stream)?;
        }
        self.long_context_kv.reset_ownership(stream)?;

        Ok(())
    }

    /// Clears one GDN slot and releases its shared attention pages.
    pub fn reset_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        for layer in &self.layers {
            layer.reset_slot(stream, slot)?;
        }
        self.long_context_kv.recycle_slot(stream, slot)?;

        Ok(())
    }

    /// Captures one physical slot's exact GDN state into device-resident scratch.
    pub fn capture_gdn_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        for layer in self.layers.iter().filter_map(ResidentLayer::gdn) {
            layer.capture_slot(stream, slot)?;
        }

        Ok(())
    }

    /// Restores one physical slot from its device-resident GDN snapshot.
    pub fn restore_gdn_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        for layer in self.layers.iter().filter_map(ResidentLayer::gdn) {
            layer.restore_slot(stream, slot)?;
        }

        Ok(())
    }

    /// Marks one stable page-table row active for a new or retained request.
    pub fn activate_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.long_context_kv.activate_slot(slot)
    }

    /// Reserves the pages required by one request's admitted position capacity.
    pub fn reserve_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<()> {
        self.long_context_kv
            .reserve_slot_tokens(stream, slot, token_count)?;
        Ok(())
    }

    /// Releases trailing pages while preserving one exact cached prefix.
    pub fn truncate_kv_slot_tokens(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        token_count: usize,
    ) -> EngineResult<usize> {
        self.long_context_kv
            .truncate_slot_tokens(stream, slot, token_count)
    }

    /// Retains an active slot's exact cached prefix for later reuse.
    pub fn retain_kv_slot(&mut self, slot: usize) -> EngineResult<()> {
        self.long_context_kv.retain_slot(slot)
    }

    /// Releases all pages held by one active or retained slot.
    pub fn recycle_kv_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        self.long_context_kv.recycle_slot(stream, slot)
    }

    /// Replays the immutable whole-model graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen35ResidentModelProgram owns every captured allocation
        // (layer owners and endpoint) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Replays one exact causal target-verification graph.
    pub(crate) fn replay_verify(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        require_verify_rows(rows)?;
        // SAFETY: this owner retains all addresses captured by the K-route.
        unsafe { self.verify_graphs[rows - 1].launch(stream) }?;

        Ok(())
    }

    /// Replays one immutable from-empty prompt graph.
    pub fn replay_prefill(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<()> {
        // SAFETY: this owner retains every allocation captured by the prompt graph.
        unsafe { self.prefill_graph(route)?.launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 full-vocabulary logits.
    pub fn read_logits(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        self.endpoint.read_logits(stream, batch)
    }

    /// Reads the final-token BF16 vocabulary logits from one prompt graph.
    pub fn read_prefill_logits(&self, stream: &CudaStream) -> EngineResult<Vec<u16>> {
        self.endpoint.read_logits(stream, 1)
    }

    /// Reads active BF16 logits into one reusable host allocation.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        self.endpoint.read_logits_into(stream, batch, destination)
    }

    /// Reads the final decoder residual before endpoint normalization.
    pub fn read_final_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        self.layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .read_residual(stream, batch)
    }

    /// Reads the final decoder residual into one reusable host row bank.
    pub fn read_final_residual_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        self.layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .read_residual_into(stream, batch, destination)
    }

    /// Reads every final-layer residual row emitted by one exact prompt tile.
    pub fn read_prefill_final_residual(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<Vec<u16>> {
        self.prefill_graph(route)?;
        self.layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .read_residual(stream, route.tokens)
    }

    pub(crate) fn gather_embedding_rows(
        &self,
        token_ids: &[u32],
        destination: &mut [u16],
    ) -> EngineResult<()> {
        self.endpoint.gather_embedding_rows(token_ids, destination)
    }

    pub(crate) fn final_residual_address(&self) -> GpuResult<*const u16> {
        self.layers
            .last()
            .ok_or_else(|| GpuError::invalid_launch("Qwen3.5 resident layer inventory is empty"))?
            .output_address()
    }

    /// Projects externally final-normalized rows with the shared BF16 LM head.
    ///
    /// # Safety
    /// `normalized` must cover `batch * 4096` BF16 values until completion.
    pub(crate) unsafe fn launch_lm_head_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        normalized: *const u16,
    ) -> GpuResult<()> {
        unsafe { self.endpoint.launch_lm_head_from(stream, batch, normalized) }
    }

    pub(crate) fn kv_slot_state(&self, slot: usize) -> EngineResult<crate::PagedKvSlotState> {
        self.long_context_kv.slot_state(slot)
    }

    pub(crate) fn kv_slot_token_count(&self, slot: usize) -> EngineResult<usize> {
        self.long_context_kv.slot_token_count(slot)
    }

    pub(crate) fn kv_route(
        &self,
        slot: usize,
        position: usize,
    ) -> EngineResult<crate::PagedKvRoute> {
        self.long_context_kv.route(slot, position)
    }

    /// CUDA context shared by every resident owner.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of every layer arena followed by the endpoint arena.
    pub fn base_addresses(&self) -> Vec<u64> {
        self.layers
            .iter()
            .map(ResidentLayer::base_address)
            .chain(core::iter::once(self.long_context_kv.base_address()))
            .chain(core::iter::once(self.endpoint.base_address()))
            .collect()
    }

    /// Exact aggregate owner layout.
    pub const fn layout(&self) -> &Qwen35ResidentModelLayout {
        &self.layout
    }

    /// Largest exact batch compiled into the graph inventory.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Fixed short-context capacity of each full-attention slot.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Page-locked bytes used to gather mmap-backed embedding rows.
    pub fn host_stager_bytes(&self) -> usize {
        self.endpoint.host_stager_bytes() + self.prefill_embedding_stager.num_bytes()
    }

    /// Fixed host page-table and physical-owner inventory bytes.
    pub const fn kv_host_owner_bytes(&self) -> usize {
        self.long_context_kv.host_allocation_bytes()
    }

    fn prefill_graph(&self, route: Qwen35ResidentPrefillRoute) -> EngineResult<&CudaGraph> {
        let index = prefill_index(route.tokens).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.5 resident prefill token count {} is outside 32,64,128",
                route.tokens
            ))
        })?;

        Ok(&self.prefill_graphs[index])
    }

    #[cfg(feature = "qualification")]
    /// Whether one live GDN slot equals its device-resident snapshot.
    pub fn qualification_gdn_slot_matches_snapshot(
        &self,
        stream: &CudaStream,
        slot: usize,
    ) -> EngineResult<bool> {
        require_slot(slot)?;
        for layer in self.layers.iter().filter_map(ResidentLayer::gdn) {
            if !layer.slot_matches_snapshot(stream, slot)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    #[cfg(feature = "qualification")]
    /// Launches the same whole-model route eagerly.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        Ok(launch_route(stream, batch, &self.layers, &self.endpoint)?)
    }

    #[cfg(feature = "qualification")]
    /// Launches one causal target-verification route eagerly.
    pub fn qualification_launch_verify(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<()> {
        require_verify_rows(rows)?;
        Ok(launch_verify_route(
            stream,
            rows,
            &self.layers,
            &self.endpoint,
        )?)
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production graph.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Physical page selected for one already-reserved logical cache position.
    pub fn qualification_kv_route(
        &self,
        slot: usize,
        position: usize,
    ) -> EngineResult<crate::PagedKvRoute> {
        self.long_context_kv.route(slot, position)
    }

    #[cfg(feature = "qualification")]
    /// Reads one complete physical K/V page from every attention layer.
    pub fn qualification_cache_page(
        &self,
        stream: &CudaStream,
        physical_page: usize,
    ) -> EngineResult<(Vec<u16>, Vec<u16>)> {
        self.long_context_kv
            .qualification_cache_page(stream, physical_page)
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete native prompt route eagerly.
    pub fn launch_prefill_eager(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<()> {
        self.prefill_graph(route)?;
        launch_prefill_route(stream, route, &self.layers, &self.endpoint)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured complete-model prompt graph.
    pub fn qualification_prefill_graph(
        &self,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<&CudaGraph> {
        self.prefill_graph(route)
    }

    #[cfg(feature = "qualification")]
    /// Fills the endpoint output planes before a whole-model route.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        self.endpoint.qualification_reset_outputs(stream, byte)
    }

    #[cfg(feature = "qualification")]
    /// Reads the final residual, normalized row, and logits for every batch slot.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen35ResidentModelObservables> {
        self.qualification_observables_for(stream, MAX_BATCH)
    }

    #[cfg(feature = "qualification")]
    /// Reads all final residual rows plus the final-token endpoint planes.
    pub fn qualification_prefill_observables(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
    ) -> EngineResult<Qwen35ResidentModelObservables> {
        self.prefill_graph(route)?;
        self.qualification_observables_for(stream, route.tokens)
    }

    #[cfg(feature = "qualification")]
    fn qualification_observables_for(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Qwen35ResidentModelObservables> {
        let endpoint = self.endpoint.qualification_observables(stream)?;
        let final_residual = self
            .layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .read_residual(stream, rows)?;

        Ok(Qwen35ResidentModelObservables {
            final_residual,
            normalized: endpoint.normalized,
            logits: endpoint.logits,
        })
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated whole-model paths for high-resolution timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.5 model graph requires at least one operation",
            ));
        }
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, batch, &self.layers, &self.endpoint)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete-model prompt paths for direct timing.
    pub fn qualification_repeated_prefill_graph(
        &self,
        stream: &CudaStream,
        route: Qwen35ResidentPrefillRoute,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        self.prefill_graph(route)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.5 model prefill graph requires at least one operation",
            ));
        }
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_prefill_route(stream, route, &self.layers, &self.endpoint)?;
            }
            Ok(())
        })?)
    }

    fn require_accounting(&self) -> EngineResult<()> {
        let arena_bytes = self.layers.iter().try_fold(0usize, |total, layer| {
            checked_sum(
                "Qwen3.5 resident program arena bytes",
                total,
                layer.arena_bytes(),
            )
        })?;
        let arena_bytes = checked_sum(
            "Qwen3.5 resident program shared KV bytes",
            arena_bytes,
            self.long_context_kv.arena_bytes(),
        )?;
        let arena_bytes = checked_sum(
            "Qwen3.5 resident program endpoint bytes",
            arena_bytes,
            self.endpoint.arena_bytes(),
        )?;
        if arena_bytes != self.layout.arena_bytes() {
            return Err(EngineError::layout(format!(
                "Qwen3.5 resident program owns {arena_bytes} bytes, layout owns {}",
                self.layout.arena_bytes()
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "qualification")]
/// Complete final planes exposed to the qualification crate.
pub struct Qwen35ResidentModelObservables {
    /// BF16 residual emitted by decoder layer 31.
    pub final_residual: Vec<u16>,
    /// BF16 endpoint-normalized rows.
    pub normalized: Vec<u16>,
    /// BF16 full-vocabulary logits.
    pub logits: Vec<u16>,
}

fn capture_decode_routes(
    stream: &CudaStream,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, layers, endpoint)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("Qwen3.5 whole-model graph inventory is incomplete"))
}

fn capture_verify_routes(
    stream: &CudaStream,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> EngineResult<[CudaGraph; 4]> {
    // Four serial B=1 passes cost 24.996 ms inside the measured 32.825-ms K=4
    // transaction. The K routes reuse each decode matrix pass across rows;
    // GDN history/state retains token order, while attention publishes the
    // complete represented K/V span before the causal row-length reads.
    let mut graphs = Vec::with_capacity(4);
    for rows in 1..=4 {
        graphs.push(CudaGraph::capture(stream, || {
            launch_verify_route(stream, rows, layers, endpoint)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 target-verification graph inventory is incomplete")
    })
}

fn capture_prefill_routes(
    stream: &CudaStream,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> EngineResult<[CudaGraph; 3]> {
    // T=128 otherwise crosses 32 layer graphs plus the endpoint and exposes
    // 32 host-visible boundaries. This graph composes the same qualified
    // per-layer routes without changing any leaf accumulation order.
    let mut graphs = Vec::with_capacity(PREFILL_ROUTES.len());
    for tokens in PREFILL_ROUTES {
        let route = Qwen35ResidentPrefillRoute { tokens };
        graphs.push(CudaGraph::capture(stream, || {
            launch_prefill_route(stream, route, layers, endpoint)
        })?);
    }

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 whole-model prefill graph inventory is incomplete")
    })
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> GpuResult<()> {
    let mut residual = endpoint.input_address()?;
    for layer in layers {
        // Each owner and pointer belongs to the same retained CUDA context; every layer returns
        // its own stable, batch-sized BF16 output plane for the next layer.
        residual = unsafe { layer.launch_from(stream, batch, residual)? };
    }
    // The final layer's stable output remains live for the endpoint launch and graph lifetime.
    unsafe { endpoint.launch_from(stream, batch, residual) }
}

fn launch_verify_route(
    stream: &CudaStream,
    rows: usize,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> GpuResult<()> {
    // K=1 has no shared-state race; reusing B=1 preserves the production
    // transition bit-for-bit instead of compiling a second equivalent route.
    if verify_uses_decode(rows) {
        return launch_route(stream, rows, layers, endpoint);
    }

    let mut residual = endpoint.input_address()?;
    for layer in layers {
        residual = unsafe { layer.launch_verify_from(stream, rows, residual)? };
    }
    unsafe { endpoint.launch_from(stream, rows, residual) }
}

fn launch_prefill_route(
    stream: &CudaStream,
    route: Qwen35ResidentPrefillRoute,
    layers: &[ResidentLayer],
    endpoint: &Qwen35TextEndpointProgram,
) -> GpuResult<()> {
    let first = layers
        .first()
        .ok_or_else(|| GpuError::invalid_launch("Qwen3.5 resident layer inventory is empty"))?;
    let mut residual = first.input_address()?;
    for layer in layers {
        // All layer owners retain 128-row publication planes in one context;
        // the next layer consumes them directly without a staging boundary.
        residual = unsafe { layer.launch_from(stream, route.tokens, residual)? };
    }
    // Only the final prompt row feeds sampling. At T=128, retaining every
    // 248,320-wide logit row would add 63,569,920 BF16 values.
    let final_row = unsafe { residual.add((route.tokens - 1) * Qwen35_9B::HIDDEN) };
    unsafe { endpoint.launch_from(stream, 1, final_row) }
}

const fn layer_kind(layer: usize) -> Qwen35ResidentLayerKind {
    if (layer + 1).is_multiple_of(Qwen35_9B::FULL_ATTENTION_INTERVAL) {
        Qwen35ResidentLayerKind::FullAttention
    } else {
        Qwen35ResidentLayerKind::Gdn
    }
}

fn require_geometry() -> EngineResult<()> {
    if Qwen35_9B::LAYERS != 32 || Qwen35_9B::FULL_ATTENTION_INTERVAL != 4 {
        return Err(EngineError::layout(
            "resident model geometry does not match the admitted Qwen3.5 layer routes",
        ));
    }
    Ok(())
}

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "batch {batch} is outside the exact range 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_verify_rows(rows: usize) -> EngineResult<()> {
    if (1..=4).contains(&rows) {
        return Ok(());
    }
    Err(EngineError::route(format!(
        "Qwen3.5 target-verification row count {rows} is outside 1..=4"
    )))
}

const fn verify_uses_decode(rows: usize) -> bool {
    rows == 1
}

fn slot_rows(slots: &[usize]) -> EngineResult<[u32; MAX_BATCH]> {
    require_batch(slots.len())?;
    let mut seen = [false; MAX_BATCH];
    let mut rows = [0u32; MAX_BATCH];
    for (row, &slot) in rows.iter_mut().zip(slots) {
        require_slot(slot)?;
        if std::mem::replace(&mut seen[slot], true) {
            return Err(EngineError::route(format!(
                "Qwen3.5 physical slot {slot} appears more than once"
            )));
        }
        *row = slot as u32;
    }
    Ok(rows)
}

fn require_slot(slot: usize) -> EngineResult<()> {
    if slot >= MAX_BATCH {
        return Err(EngineError::route(format!(
            "Qwen3.5 physical slot {slot} is outside 0..{MAX_BATCH}"
        )));
    }
    Ok(())
}

const fn prefill_index(tokens: usize) -> Option<usize> {
    match tokens {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        _ => None,
    }
}

fn require_prefill(tokens: usize) -> EngineResult<()> {
    if prefill_index(tokens).is_some() {
        return Ok(());
    }

    Err(EngineError::route(format!(
        "Qwen3.5 resident prefill token count {tokens} is outside 32,64,128"
    )))
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn checked_sum(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn sum_products(name: &str, terms: &[(usize, usize)]) -> EngineResult<usize> {
    terms.iter().try_fold(0usize, |total, &(count, bytes)| {
        checked_sum(name, total, product(name, count, bytes)?)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen35ResidentLayerKind, Qwen35ResidentModelLayout, prefill_index, require_prefill,
        require_verify_rows, slot_rows, verify_uses_decode,
    };
    use crate::EngineErrorCode;

    #[test]
    fn exact_layer_route_inventory_is_complete() {
        let layout = Qwen35ResidentModelLayout::build().unwrap();
        let mut counts = [0usize; 2];
        for layer in 0..layout.layer_count() {
            let kind = layout.layer_kind(layer).unwrap();
            counts[usize::from(kind == Qwen35ResidentLayerKind::FullAttention)] += 1;
            assert_eq!(
                kind == Qwen35ResidentLayerKind::FullAttention,
                (layer + 1).is_multiple_of(4),
                "layer {layer}"
            );
        }

        assert_eq!(layout.layer_count(), 32);
        assert_eq!(counts, [24, 8]);
        assert_eq!(layout.layer_kind(32), None);
    }

    #[test]
    fn resident_byte_accounting_is_exact() {
        let layout = Qwen35ResidentModelLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 5_931_820_032);
        assert_eq!(layout.cache_bytes(), 8_640_266_240);
        assert_eq!(layout.workspace_bytes(), 1_159_672_064);
        assert_eq!(layout.owner_bytes(), 15_731_758_336);
        assert_eq!(layout.padding_bytes(), 21_248);
        assert_eq!(layout.arena_bytes(), 15_731_779_584);
        assert_eq!(layout.arena_count(), 34);
        assert_eq!(
            layout.source_mapped_embedding_bytes().unwrap(),
            2_034_237_440
        );
        assert_eq!(layout.context_capacity(), 262_144);
    }

    #[test]
    fn compact_slot_table_rejects_aliases_and_boundaries() {
        assert_eq!(slot_rows(&[7, 0, 5, 2]).unwrap()[..4], [7, 0, 5, 2]);
        for slots in [&[][..], &[0, 0], &[0, 8], &[0, 1, 2, 3, 4, 5, 6, 7, 0]] {
            let error = slot_rows(slots).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn exact_prefill_inventory_rejects_every_neighbor() {
        for (index, tokens) in [32, 64, 128].into_iter().enumerate() {
            assert_eq!(prefill_index(tokens), Some(index));
            require_prefill(tokens).unwrap();
        }
        for tokens in [0, 1, 8, 16, 31, 33, 63, 65, 127, 129, usize::MAX] {
            assert_eq!(prefill_index(tokens), None);
            assert_eq!(
                require_prefill(tokens).unwrap_err().code(),
                Some(EngineErrorCode::Route)
            );
        }
    }

    #[test]
    fn exact_verification_inventory_covers_only_k1_through_k4() {
        for rows in 1..=4 {
            require_verify_rows(rows).unwrap();
        }
        for rows in [0, 5, 8, 16, 32, usize::MAX] {
            assert_eq!(
                require_verify_rows(rows).unwrap_err().code(),
                Some(EngineErrorCode::Route),
                "rows={rows}"
            );
        }
        for (rows, expected) in [
            (0, false),
            (1, true),
            (2, false),
            (3, false),
            (4, false),
            (5, false),
            (usize::MAX, false),
        ] {
            assert_eq!(verify_uses_decode(rows), expected, "rows={rows}");
        }
    }
}

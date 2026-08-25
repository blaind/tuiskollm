//! Resident composition of every Qwen3.6 text layer and endpoint.

use crate::qwen36_long_context_kv::Qwen36AttentionKvBinding;
use crate::{
    EngineError, EngineResult, MAX_BATCH, Qwen36FullAttentionLayerLayout,
    Qwen36FullAttentionLayerProgram, Qwen36GdnMoeLayerLayout, Qwen36GdnMoeLayerProgram,
    Qwen36LongContextKvLayout, Qwen36LongContextKvProgram, Qwen36TextEndpointLayout,
    Qwen36TextEndpointProgram,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult, PinnedHostBuffer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

const ROTARY_PAIRS: usize = 32;
const PREFILL_ROUTES: [usize; 3] = [32, 64, 128];
const MAX_PREFILL_ROWS: usize = 128;

/// Exact from-empty prompt graph selected by matching state uploads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the Qwen3.6 prefill route must be replayed with the state that selected it"]
pub struct Qwen36ResidentPrefillRoute {
    tokens: usize,
}

impl Qwen36ResidentPrefillRoute {
    /// Number of contiguous prompt tokens represented by this route.
    pub const fn tokens(self) -> usize {
        self.tokens
    }
}

/// Exact source route owned by one Qwen3.6 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36ResidentLayerKind {
    /// Gated-delta mixer followed by the routed and shared experts.
    GdnMoe,
    /// Full attention followed by the routed and shared experts.
    FullAttentionMoe,
}

/// Exact byte accounting for the initial address-stable resident composition.
#[derive(Clone, Debug)]
pub struct Qwen36ResidentModelLayout {
    layers: [Qwen36ResidentLayerKind; Qwen36Moe35B::LAYERS],
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
    arena_bytes: usize,
}

impl Qwen36ResidentModelLayout {
    /// Accounts all 40 decoder owners and the source-backed NVFP4 endpoint.
    pub fn build() -> EngineResult<Self> {
        require_geometry()?;
        let gdn = Qwen36GdnMoeLayerLayout::build()?;
        let attention = Qwen36FullAttentionLayerLayout::build()?;
        let long_context = Qwen36LongContextKvLayout::build()?;
        let endpoint = Qwen36TextEndpointLayout::build()?;
        let layers = core::array::from_fn(layer_kind);
        let gdn_layers = layers
            .iter()
            .filter(|&&kind| kind == Qwen36ResidentLayerKind::GdnMoe)
            .count();
        let attention_layers = Qwen36Moe35B::LAYERS - gdn_layers;
        let resident_weight_bytes = sum_products(
            "Qwen3.6 resident device weight bytes",
            &[
                (gdn_layers, gdn.resident_weight_bytes()),
                (attention_layers, attention.resident_weight_bytes()),
                (1, endpoint.resident_weight_bytes()),
            ],
        )?;
        let cache_bytes = checked_sum(
            "Qwen3.6 resident E4M3 cache bytes",
            product(
                "Qwen3.6 resident short E4M3 cache bytes",
                attention_layers,
                attention.cache_bytes(),
            )?,
            long_context.cache_bytes(),
        )?;
        let workspace_bytes = sum_products(
            "Qwen3.6 resident workspace bytes",
            &[
                (gdn_layers, gdn.workspace_bytes()),
                (attention_layers, attention.workspace_bytes()),
                (1, endpoint.workspace_bytes()),
                (1, long_context.block_table_bytes()),
            ],
        )?;
        let arena_bytes = sum_products(
            "Qwen3.6 resident arena bytes",
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

    /// Source route for one layer, or `None` outside `0..40`.
    pub fn layer_kind(&self, layer: usize) -> Option<Qwen36ResidentLayerKind> {
        self.layers.get(layer).copied()
    }

    /// Source-backed decoder, final-norm, and NVFP4 LM-head device bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// E4M3 K/V bytes in the shared pool and retained layer-local fallback planes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable per-layer state and working bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Complete device allocation across layers, shared KV, and endpoint arenas.
    pub const fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }

    /// All represented device bytes, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Aggregate alignment padding across the 42 stable arenas.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes - self.owner_bytes()
    }

    /// Number of independently allocated, address-stable device arenas.
    pub const fn arena_count(&self) -> usize {
        Qwen36Moe35B::LAYERS + 2
    }

    /// mmap-backed BF16 embeddings intentionally excluded from device residency.
    pub fn source_mapped_embedding_bytes(&self) -> EngineResult<usize> {
        product(
            "Qwen3.6 source-mapped embedding bytes",
            product(
                "Qwen3.6 source-mapped embedding elements",
                Qwen36Moe35B::VOCAB,
                Qwen36Moe35B::HIDDEN,
            )?,
            size_of::<u16>(),
        )
    }

    /// Maximum context admitted by the pinned Qwen3.6 config.
    pub const fn context_capacity(&self) -> usize {
        crate::QWEN36_MAX_CONTEXT_TOKENS
    }
}

enum ResidentLayer {
    GdnMoe(Box<Qwen36GdnMoeLayerProgram>),
    FullAttentionMoe(Box<Qwen36FullAttentionLayerProgram>),
}

impl ResidentLayer {
    fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
        layer: usize,
        kv_binding: Option<Qwen36AttentionKvBinding>,
    ) -> EngineResult<Self> {
        match layer_kind(layer) {
            Qwen36ResidentLayerKind::GdnMoe => Ok(Self::GdnMoe(Box::new(
                Qwen36GdnMoeLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
            Qwen36ResidentLayerKind::FullAttentionMoe => {
                let binding = kv_binding.ok_or_else(|| {
                    EngineError::layout(format!(
                        "Qwen3.6 full-attention layer {layer} has no shared KV binding"
                    ))
                })?;
                // SAFETY: the resident program declares graph-bearing layers before
                // the shared KV owner, so every captured address outlives them.
                let program = unsafe {
                    Qwen36FullAttentionLayerProgram::from_snapshot_with_kv(
                        context, snapshot, layer, binding,
                    )?
                };
                Ok(Self::FullAttentionMoe(Box::new(program)))
            }
        }
    }

    fn reset(&self, stream: &CudaStream) -> EngineResult<()> {
        match self {
            Self::GdnMoe(program) => program.reset_state(stream),
            Self::FullAttentionMoe(_) => Ok(()),
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
            Self::GdnMoe(_) => Ok(()),
            Self::FullAttentionMoe(program) => {
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
            Self::GdnMoe(_) => Ok(()),
            Self::FullAttentionMoe(program) => program.load_prefill_slot_state_at(
                stream,
                tokens,
                slot,
                first_position,
                rope_cos,
                rope_sin,
            ),
        }
    }

    fn load_residual(&self, stream: &CudaStream, rows: usize, values: &[u16]) -> EngineResult<()> {
        match self {
            Self::GdnMoe(program) => program.load_residual(stream, rows, values),
            Self::FullAttentionMoe(program) => program.load_residual(stream, rows, values),
        }
    }

    fn input_address(&self) -> GpuResult<*const u16> {
        match self {
            Self::GdnMoe(program) => program.input_address(),
            Self::FullAttentionMoe(program) => program.input_address(),
        }
    }

    /// # Safety
    /// `input` names the active BF16 residual rows in the shared CUDA context.
    unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        match self {
            Self::GdnMoe(program) => unsafe { program.launch_from(stream, batch, input) },
            Self::FullAttentionMoe(program) => unsafe { program.launch_from(stream, batch, input) },
        }
    }

    fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        match self {
            Self::GdnMoe(program) => program.read_residual(stream, batch),
            Self::FullAttentionMoe(program) => program.read_residual(stream, batch),
        }
    }

    fn base_address(&self) -> u64 {
        match self {
            Self::GdnMoe(program) => program.base_address(),
            Self::FullAttentionMoe(program) => program.base_address(),
        }
    }

    fn arena_bytes(&self) -> usize {
        match self {
            Self::GdnMoe(program) => program.arena_bytes(),
            Self::FullAttentionMoe(program) => program.arena_bytes(),
        }
    }
}

/// Every Qwen3.6 text layer and the NVFP4 endpoint held at stable addresses.
pub struct Qwen36ResidentModelProgram {
    // Drop graphs and layers before the shared KV arena whose addresses they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 3],
    layers: Vec<ResidentLayer>,
    long_context_kv: Qwen36LongContextKvProgram,
    endpoint: Qwen36TextEndpointProgram,
    prefill_embedding_stager: PinnedHostBuffer<u16>,
    context: Arc<CudaContext>,
    layout: Qwen36ResidentModelLayout,
}

impl Qwen36ResidentModelProgram {
    /// Loads all source weights and captures exact decode and native prefill graphs.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    ) -> EngineResult<Self> {
        let layout = Qwen36ResidentModelLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let long_context_kv = Qwen36LongContextKvProgram::new(context)?;
        let mut layers = Vec::with_capacity(Qwen36Moe35B::LAYERS);
        let mut attention_layer = 0;
        for layer in 0..Qwen36Moe35B::LAYERS {
            let kv_binding = if layer_kind(layer) == Qwen36ResidentLayerKind::FullAttentionMoe {
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
                "Qwen3.6 resident model bound {attention_layer} attention layers, shared KV owner has {}",
                long_context_kv.layout().layers().len()
            )));
        }
        let endpoint = Qwen36TextEndpointProgram::from_snapshot(context, snapshot)?;
        let prefill_embedding_stager =
            PinnedHostBuffer::zeroed(context, MAX_PREFILL_ROWS * Qwen36Moe35B::HIDDEN)
                .map_err(GpuError::from)?;
        for layer in &layers {
            layer.reset(&stream)?;
        }
        let graphs = capture_decode_routes(&stream, &layers, &endpoint)?;
        let prefill_graphs = capture_prefill_routes(&stream, &layers, &endpoint)?;
        let program = Self {
            graphs,
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
            "Qwen3.6 resident prefill embedding values",
            token_ids.len(),
            Qwen36Moe35B::HIDDEN,
        )?;
        self.endpoint
            .gather_embedding_rows(token_ids, &mut self.prefill_embedding_stager[..active])?;
        self.layers
            .first()
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
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

    /// Updates every attention layer for one contiguous from-empty prompt tile.
    pub fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<Qwen36ResidentPrefillRoute> {
        require_prefill(tokens)?;
        let reserved_tokens = self.long_context_kv.slot_token_count(0)?;
        if tokens > reserved_tokens {
            return Err(EngineError::route(format!(
                "Qwen3.6 resident prefill T={tokens} exceeds slot 0's {reserved_tokens} reserved tokens"
            )));
        }
        let rotary_values = product(
            "Qwen3.6 resident prefill rotary values",
            tokens,
            ROTARY_PAIRS,
        )?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.6 resident prefill rotary planes must each have {rotary_values} values for T={tokens}"
            )));
        }
        for layer in &self.layers {
            layer.load_prefill_state(stream, tokens, 0, 0, rope_cos, rope_sin)?;
        }

        Ok(Qwen36ResidentPrefillRoute { tokens })
    }

    /// Clears every GDN state and recycles all shared attention page-table rows.
    pub fn reset_state(&mut self, stream: &CudaStream) -> EngineResult<()> {
        for layer in &self.layers {
            layer.reset(stream)?;
        }
        self.long_context_kv.reset_ownership(stream)?;

        Ok(())
    }

    /// Marks one stable page-table row active for a new request.
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

    /// Releases all pages held by one active or retained slot.
    pub fn recycle_kv_slot(&mut self, stream: &CudaStream, slot: usize) -> EngineResult<usize> {
        self.long_context_kv.recycle_slot(stream, slot)
    }

    /// Replays the immutable whole-model graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen36ResidentModelProgram owns every captured allocation
        // (layer owners and endpoint) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Replays one immutable from-empty prompt graph.
    pub fn replay_prefill(
        &self,
        stream: &CudaStream,
        route: Qwen36ResidentPrefillRoute,
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
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
            .read_residual(stream, batch)
    }

    /// Reads every final-layer residual row emitted by one exact prompt tile.
    pub fn read_prefill_final_residual(
        &self,
        stream: &CudaStream,
        route: Qwen36ResidentPrefillRoute,
    ) -> EngineResult<Vec<u16>> {
        self.prefill_graph(route)?;
        self.layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
            .read_residual(stream, route.tokens)
    }

    /// CUDA context shared by every resident owner.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of every layer, shared KV, and endpoint arena.
    pub fn base_addresses(&self) -> Vec<u64> {
        self.layers
            .iter()
            .map(ResidentLayer::base_address)
            .chain(core::iter::once(self.long_context_kv.base_address()))
            .chain(core::iter::once(self.endpoint.base_address()))
            .collect()
    }

    /// Exact aggregate owner layout.
    pub const fn layout(&self) -> &Qwen36ResidentModelLayout {
        &self.layout
    }

    /// Largest exact batch compiled into the graph inventory.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Maximum context admitted by the pinned checkpoint.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Page-locked bytes used to gather mmap-backed embedding rows.
    pub fn host_stager_bytes(&self) -> usize {
        self.endpoint.host_stager_bytes() + self.prefill_embedding_stager.num_bytes()
    }

    /// Fixed host page-table and owner-map bytes for the shared cache.
    pub const fn kv_host_owner_bytes(&self) -> usize {
        self.long_context_kv.host_allocation_bytes()
    }

    fn prefill_graph(&self, route: Qwen36ResidentPrefillRoute) -> EngineResult<&CudaGraph> {
        let index = prefill_index(route.tokens).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.6 resident prefill token count {} is outside 32,64,128",
                route.tokens
            ))
        })?;

        Ok(&self.prefill_graphs[index])
    }

    #[cfg(feature = "qualification")]
    /// Launches the same whole-model route eagerly.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        Ok(launch_route(stream, batch, &self.layers, &self.endpoint)?)
    }

    #[cfg(feature = "qualification")]
    /// Returns one immutable production graph.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Launches one complete native prompt route eagerly.
    pub fn launch_prefill_eager(
        &self,
        stream: &CudaStream,
        route: Qwen36ResidentPrefillRoute,
    ) -> EngineResult<()> {
        self.prefill_graph(route)?;
        launch_prefill_route(stream, route, &self.layers, &self.endpoint)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured complete-model prompt graph.
    pub fn qualification_prefill_graph(
        &self,
        route: Qwen36ResidentPrefillRoute,
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
    ) -> EngineResult<Qwen36ResidentModelObservables> {
        self.qualification_observables_for(stream, MAX_BATCH)
    }

    #[cfg(feature = "qualification")]
    /// Reads all final residual rows plus the final-token endpoint planes.
    pub fn qualification_prefill_observables(
        &self,
        stream: &CudaStream,
        route: Qwen36ResidentPrefillRoute,
    ) -> EngineResult<Qwen36ResidentModelObservables> {
        self.prefill_graph(route)?;
        self.qualification_observables_for(stream, route.tokens)
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
                "repeated Qwen3.6 model graph requires at least one operation",
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
        route: Qwen36ResidentPrefillRoute,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        self.prefill_graph(route)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.6 model prefill graph requires at least one operation",
            ));
        }
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_prefill_route(stream, route, &self.layers, &self.endpoint)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    fn qualification_observables_for(
        &self,
        stream: &CudaStream,
        rows: usize,
    ) -> EngineResult<Qwen36ResidentModelObservables> {
        let endpoint = self.endpoint.qualification_observables(stream)?;
        let final_residual = self
            .layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
            .read_residual(stream, rows)?;

        Ok(Qwen36ResidentModelObservables {
            final_residual,
            normalized: endpoint.normalized,
            logits: endpoint.logits,
        })
    }

    fn require_accounting(&self) -> EngineResult<()> {
        let arena_bytes = self.layers.iter().try_fold(0usize, |total, layer| {
            checked_sum(
                "Qwen3.6 resident program arena bytes",
                total,
                layer.arena_bytes(),
            )
        })?;
        let arena_bytes = checked_sum(
            "Qwen3.6 resident program shared KV bytes",
            arena_bytes,
            self.long_context_kv.arena_bytes(),
        )?;
        let arena_bytes = checked_sum(
            "Qwen3.6 resident program endpoint bytes",
            arena_bytes,
            self.endpoint.arena_bytes(),
        )?;
        if arena_bytes != self.layout.arena_bytes() {
            return Err(EngineError::layout(format!(
                "Qwen3.6 resident program owns {arena_bytes} bytes, layout owns {}",
                self.layout.arena_bytes()
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "qualification")]
/// Complete final planes exposed to the qualification crate.
pub struct Qwen36ResidentModelObservables {
    /// BF16 residual emitted by decoder layer 39.
    pub final_residual: Vec<u16>,
    /// BF16 endpoint-normalized rows.
    pub normalized: Vec<u16>,
    /// BF16 full-vocabulary logits.
    pub logits: Vec<u16>,
}

fn capture_decode_routes(
    stream: &CudaStream,
    layers: &[ResidentLayer],
    endpoint: &Qwen36TextEndpointProgram,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, layers, endpoint)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("Qwen3.6 whole-model graph inventory is incomplete"))
}

fn capture_prefill_routes(
    stream: &CudaStream,
    layers: &[ResidentLayer],
    endpoint: &Qwen36TextEndpointProgram,
) -> EngineResult<[CudaGraph; 3]> {
    // T=128 otherwise crosses 40 layer graphs plus the endpoint and exposes
    // 40 host-visible boundaries. This graph composes the same qualified
    // per-layer T=128 routes without changing any leaf accumulation order.
    let mut graphs = Vec::with_capacity(PREFILL_ROUTES.len());
    for tokens in PREFILL_ROUTES {
        let route = Qwen36ResidentPrefillRoute { tokens };
        graphs.push(CudaGraph::capture(stream, || {
            launch_prefill_route(stream, route, layers, endpoint)
        })?);
    }

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.6 whole-model prefill graph inventory is incomplete")
    })
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    layers: &[ResidentLayer],
    endpoint: &Qwen36TextEndpointProgram,
) -> GpuResult<()> {
    let mut residual = endpoint.input_address()?;
    for layer in layers {
        // Every retained layer owns an address-stable BF16 publication plane in the same CUDA
        // context, so the following layer consumes it directly without a staging copy.
        residual = unsafe { layer.launch_from(stream, batch, residual)? };
    }
    // The final layer's publication remains live for the endpoint and graph lifetime.
    unsafe { endpoint.launch_from(stream, batch, residual) }
}

fn launch_prefill_route(
    stream: &CudaStream,
    route: Qwen36ResidentPrefillRoute,
    layers: &[ResidentLayer],
    endpoint: &Qwen36TextEndpointProgram,
) -> GpuResult<()> {
    let first = layers
        .first()
        .ok_or_else(|| GpuError::invalid_launch("Qwen3.6 resident layer inventory is empty"))?;
    let mut residual = first.input_address()?;
    for layer in layers {
        // All layer owners retain 128-row publication planes in one context;
        // the next layer consumes them directly without a staging boundary.
        residual = unsafe { layer.launch_from(stream, route.tokens, residual)? };
    }
    // Only the final prompt row feeds sampling. At T=128, retaining all
    // 248,320-wide logits would require 63,569,920 extra BF16 values.
    let final_row = unsafe { residual.add((route.tokens - 1) * Qwen36Moe35B::HIDDEN) };
    unsafe { endpoint.launch_from(stream, 1, final_row) }
}

const fn layer_kind(layer: usize) -> Qwen36ResidentLayerKind {
    if (layer + 1).is_multiple_of(Qwen36Moe35B::FULL_ATTENTION_INTERVAL) {
        Qwen36ResidentLayerKind::FullAttentionMoe
    } else {
        Qwen36ResidentLayerKind::GdnMoe
    }
}

fn require_geometry() -> EngineResult<()> {
    if Qwen36Moe35B::LAYERS != 40 || Qwen36Moe35B::FULL_ATTENTION_INTERVAL != 4 {
        return Err(EngineError::layout(
            "resident model geometry does not match the admitted Qwen3.6 layer routes",
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
        "Qwen3.6 resident prefill token count {tokens} is outside 32,64,128"
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
        Qwen36ResidentLayerKind, Qwen36ResidentModelLayout, prefill_index, require_prefill,
    };
    use crate::EngineErrorCode;

    #[test]
    fn exact_layer_route_inventory_is_complete() {
        let layout = Qwen36ResidentModelLayout::build().unwrap();
        let mut counts = [0usize; 2];
        for layer in 0..layout.layer_count() {
            let kind = layout.layer_kind(layer).unwrap();
            counts[usize::from(kind == Qwen36ResidentLayerKind::FullAttentionMoe)] += 1;
            assert_eq!(
                kind == Qwen36ResidentLayerKind::FullAttentionMoe,
                (layer + 1).is_multiple_of(4),
                "layer {layer}"
            );
        }

        assert_eq!(layout.layer_count(), 40);
        assert_eq!(counts, [30, 10]);
        assert_eq!(layout.layer_kind(40), None);
    }

    #[test]
    fn resident_byte_accounting_is_exact() {
        let layout = Qwen36ResidentModelLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 19_808_036_096);
        assert_eq!(layout.cache_bytes(), 2_700_083_200);
        assert_eq!(layout.workspace_bytes(), 1_223_843_648);
        assert_eq!(layout.owner_bytes(), 23_731_962_944);
        assert_eq!(layout.padding_bytes(), 26_560);
        assert_eq!(layout.arena_bytes(), 23_731_989_504);
        assert_eq!(layout.arena_count(), 42);
        assert_eq!(
            layout.source_mapped_embedding_bytes().unwrap(),
            1_017_118_720
        );
        assert_eq!(layout.context_capacity(), 262_144);
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
}

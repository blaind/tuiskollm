//! Resident exact-target execution owner for the complete text model.

use super::{
    AttentionWeights, EndpointWeights, GdnPersistent, GdnWeights, MixerWeights, MlpWeights,
    ResidentModelLayout,
};
use crate::long_context_kv_layout::LayerKvRegions;
use crate::{EngineError, EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH};
#[cfg(feature = "qualification")]
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, DeviceCopy, GpuError, GpuResult,
    PinnedHostBuffer,
};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_kernels_sm120::{
    AttentionOutputOp, AttentionQkPrepareOp, DenseFp8DownOp, DenseFp8SwiGluOp, FullAttentionQkvOp,
    GdnInputProjectionOp, GdnOutputProjectionOp, GdnPrepareOp, GdnRecurrenceOp, LmHeadOp,
    Nvfp4DownOp, Nvfp4SwiGluOp, PagedGqaOp, ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, DenseFp8DownBindings, DenseFp8GateUpBindings,
    FullAttentionPostBindings, FullAttentionQkvBindings, GdnBindings, Nvfp4DownBindings,
    Nvfp4GateUpBindings, Qwen38_27B, TextEndpointBindings,
};

const ROTARY_PAIRS: usize = 32;
const SHORT_CONTEXT_PAGES_PER_SLOT: usize = super::CONTEXT_CAPACITY / ATTENTION_PAGE_SIZE;
const SHORT_CONTEXT_PHYSICAL_PAGES: usize = MAX_BATCH * SHORT_CONTEXT_PAGES_PER_SLOT;

/// Resident and shared-KV arenas plus immutable `B=1..=8` graphs for all 64 text layers.
pub struct ResidentModelProgram {
    // Graphs retain both arena addresses and module handles, so they drop first.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    kv_arena: DeviceArena,
    _norm: ResidualNormOp<Qwen38_27B>,
    _gdn_input: GdnInputProjectionOp<Qwen38_27B>,
    _gdn_prepare: GdnPrepareOp<Qwen38_27B>,
    _gdn_recurrence: GdnRecurrenceOp<Qwen38_27B>,
    _gdn_output: GdnOutputProjectionOp<Qwen38_27B>,
    _attention_qkv: FullAttentionQkvOp<Qwen38_27B>,
    _attention_qk_prepare: AttentionQkPrepareOp<Qwen38_27B>,
    _paged_gqa: PagedGqaOp<Qwen38_27B>,
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

impl ResidentModelProgram {
    /// Loads every exact source plane and captures one complete graph for each `B=1..=8` route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38_27B>>,
    ) -> EngineResult<Self> {
        let layout = ResidentModelLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, &layout.builder)?;
        let kv_arena = DeviceArena::zeroed(&stream, layout.kv_layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let gdn_input = GdnInputProjectionOp::new(context)?;
        let gdn_prepare = GdnPrepareOp::new(context)?;
        let gdn_recurrence = GdnRecurrenceOp::new(context)?;
        let gdn_output = GdnOutputProjectionOp::new(context)?;
        let attention_qkv = FullAttentionQkvOp::new(context)?;
        let attention_qk_prepare = AttentionQkPrepareOp::new(context)?;
        let paged_gqa = PagedGqaOp::new(context)?;
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
                MAX_BATCH,
                Qwen38_27B::HIDDEN,
            )?,
        )
        .map_err(GpuError::from)?;

        let scalars = load_source_weights(&arena, &stream, &layout, snapshot.as_ref())?;
        initialize_metadata(&arena, &kv_arena, &stream, &layout)?;
        let pointers = ProgramPointers::bind(&arena, &kv_arena, &layout, &scalars)?;
        let base_address = arena.base_address();
        let kv_base_address = kv_arena.base_address();
        let ops = Ops {
            norm: &norm,
            gdn_input: &gdn_input,
            gdn_prepare: &gdn_prepare,
            gdn_recurrence: &gdn_recurrence,
            gdn_output: &gdn_output,
            attention_qkv: &attention_qkv,
            attention_qk_prepare: &attention_qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            dense_swiglu: &dense_swiglu,
            dense_down: &dense_down,
            nvfp4_swiglu: &nvfp4_swiglu,
            nvfp4_down: &nvfp4_down,
            lm_head: &lm_head,
        };
        let graphs = capture_routes(&stream, ops, &pointers)?;

        Ok(Self {
            graphs,
            arena,
            kv_arena,
            _norm: norm,
            _gdn_input: gdn_input,
            _gdn_prepare: gdn_prepare,
            _gdn_recurrence: gdn_recurrence,
            _gdn_output: gdn_output,
            _attention_qkv: attention_qkv,
            _attention_qk_prepare: attention_qk_prepare,
            _paged_gqa: paged_gqa,
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
        })
    }

    /// Copies exact mmap-backed BF16 embedding rows into the first residual plane.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        require_batch(token_ids.len())?;
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
        batch: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let expected = product("resident residual elements", batch, Qwen38_27B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "resident residual input has {} values, expected {expected} for B={batch}",
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
    ) -> EngineResult<()> {
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

        Ok(())
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

    /// Clears all recurrent owners and the 24 pages assigned to the current decode route.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        let mut attention_layer = 0;
        for layer in &self.layout.layers {
            match layer.persistent {
                super::PersistentState::Gdn(state) => {
                    self.arena.fill(stream, state.history, 0)?;
                    self.arena.fill(stream, state.state, 0)?;
                }
                super::PersistentState::Attention => {
                    let cache = self
                        .layout
                        .kv_layout
                        .layers()
                        .get(attention_layer)
                        .copied()
                        .ok_or_else(|| {
                            EngineError::layout(
                                "resident reset visited more attention layers than the shared KV inventory",
                            )
                        })?;
                    fill_cache_prefix(&self.kv_arena, stream, cache.key.data)?;
                    fill_cache_prefix(&self.kv_arena, stream, cache.value.data)?;
                    attention_layer += 1;
                }
            }
        }
        if attention_layer != self.layout.kv_layout.layers().len() {
            return Err(EngineError::layout(
                "resident reset did not visit every shared KV layer",
            ));
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
                super::PersistentState::Attention => {
                    let cache = self
                        .layout
                        .kv_layout
                        .layers()
                        .get(attention_layer)
                        .copied()
                        .ok_or_else(|| {
                            EngineError::layout(
                                "resident slot reset visited more attention layers than the shared KV inventory",
                            )
                        })?;
                    fill_cache_slot(&self.kv_arena, stream, cache.key.data, slot)?;
                    fill_cache_slot(&self.kv_arena, stream, cache.value.data, slot)?;
                    attention_layer += 1;
                }
            }
        }
        if attention_layer != self.layout.kv_layout.layers().len() {
            return Err(EngineError::layout(
                "resident slot reset did not visit every shared KV layer",
            ));
        }

        Ok(())
    }

    /// Replays the immutable complete-model graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        self.graphs[batch - 1].launch(stream)?;

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

    /// CUDA context shared by the arena, graphs, and prepared operators.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by all exact-batch graphs.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable base address of the shared page-table and KV arena.
    pub const fn kv_base_address(&self) -> u64 {
        self.kv_base_address
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

    /// GDN state plus the three cache pages assigned to one current decode slot.
    pub const fn persistent_slot_bytes(&self) -> usize {
        (self.layout.history_bytes()
            + self.layout.state_bytes()
            + SHORT_CONTEXT_PHYSICAL_PAGES
                * Qwen38_27B::NUM_KV_HEADS
                * ATTENTION_PAGE_SIZE
                * Qwen38_27B::HEAD_DIM
                * 2
                * (Qwen38_27B::LAYERS / Qwen38_27B::FULL_ATTENTION_INTERVAL))
            / MAX_BATCH
    }

    /// Exact address-stable workspace bytes shared by every layer and endpoint.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
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
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_route(stream, batch, self.ops(), &self._pointers)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured complete-model graph.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated complete-model schedules for direct intrinsic timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated resident-model graph requires at least one operation",
            ));
        }
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, batch, ops, &self._pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures production embedding staging separately from model-graph timing.
    pub fn qualification_embedding_stage_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
    ) -> EngineResult<ResidentEmbeddingStageGraph<'_>> {
        require_batch(batch)?;
        let active = product(
            "resident staged embedding elements",
            batch,
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
    /// Returns every immutable and mutable address captured by the owner.
    pub fn qualification_addresses(&self) -> Vec<usize> {
        self._pointers.addresses()
    }

    #[cfg(feature = "qualification")]
    /// Reads all eight stable long-context page-table rows.
    pub fn qualification_block_tables(&self, stream: &CudaStream) -> EngineResult<Vec<u32>> {
        Ok(self
            .kv_arena
            .copy_to_host(stream, self.layout.kv_layout.block_tables())?)
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
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            workspace.activation_scales,
            workspace.log_decay,
            workspace.beta,
            workspace.query,
            workspace.attention,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        Ok(())
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
    fn ops(&self) -> Ops<'_> {
        Ops {
            norm: &self._norm,
            gdn_input: &self._gdn_input,
            gdn_prepare: &self._gdn_prepare,
            gdn_recurrence: &self._gdn_recurrence,
            gdn_output: &self._gdn_output,
            attention_qkv: &self._attention_qkv,
            attention_qk_prepare: &self._attention_qk_prepare,
            paged_gqa: &self._paged_gqa,
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
    attention: *mut f32,
    mixer_branch: *mut u16,
    swiglu: *mut u16,
    mlp_branch: *mut u16,
    logits: *mut u16,
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
                mlp: bind_mlp(arena, layer.mlp, scalars.mlp)?,
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
            attention: arena.address(regions.attention)?,
            mixer_branch: arena.address(regions.mixer_branch)?,
            swiglu: arena.address(regions.swiglu)?,
            mlp_branch: arena.address(regions.mlp_branch)?,
            logits: arena.address(regions.logits)?,
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
            self.attention.addr(),
            self.mixer_branch.addr(),
            self.swiglu.addr(),
            self.mlp_branch.addr(),
            self.logits.addr(),
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
            Ok(MlpPointers::DenseFp8(DenseFp8Pointers {
                gate_up_weight_codes: arena.address(weights.gate_up_weight_codes)?.cast_const(),
                gate_up_weight_scales: arena.address(weights.gate_up_weight_scales)?.cast_const(),
                down_weight_codes: arena.address(weights.down_weight_codes)?.cast_const(),
                down_weight_scales: arena.address(weights.down_weight_scales)?.cast_const(),
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
    gdn_output: &'a GdnOutputProjectionOp<Qwen38_27B>,
    attention_qkv: &'a FullAttentionQkvOp<Qwen38_27B>,
    attention_qk_prepare: &'a AttentionQkPrepareOp<Qwen38_27B>,
    paged_gqa: &'a PagedGqaOp<Qwen38_27B>,
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
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, ops, pointers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("resident graph inventory has wrong cardinality"))
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    ops: Ops<'_>,
    pointers: &ProgramPointers,
) -> GpuResult<()> {
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
        launch_mixer(stream, batch, ops, workspace, layer.mixer)?;
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
        launch_mlp(stream, batch, ops, workspace, layer.mlp)?;

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

fn launch_mixer(
    stream: &CudaStream,
    batch: usize,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mixer: MixerPointers,
) -> GpuResult<()> {
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
                ops.paged_gqa.launch(
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
                )?;
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
    batch: usize,
    ops: Ops<'_>,
    workspace: WorkspacePointers,
    mlp: MlpPointers,
) -> GpuResult<()> {
    // SAFETY: all weights are source-route matched and shared scratch covers MAX_BATCH.
    unsafe {
        match mlp {
            MlpPointers::Nvfp4(p) => {
                ops.nvfp4_swiglu.launch(
                    stream,
                    batch,
                    workspace.mlp_normalized,
                    workspace.nvfp4_activation_codes,
                    workspace.nvfp4_activation_scales,
                    p.gate_weight_codes,
                    p.gate_up_weight_scales,
                    p.scalars.gate_up_input,
                    p.scalars.gate_up_weight,
                    workspace.swiglu,
                )?;
                // The admitted A16 down route consumes BF16 directly; its
                // source input divisor remains observable but is not applied.
                let _ = p.scalars.down_input;
                ops.nvfp4_down.launch(
                    stream,
                    batch,
                    workspace.swiglu,
                    p.down_weight_codes,
                    p.down_weight_scales,
                    p.scalars.down_weight,
                    workspace.mlp_branch,
                )
            }
            MlpPointers::DenseFp8(p) => {
                ops.dense_swiglu.launch(
                    stream,
                    batch,
                    workspace.mlp_normalized,
                    workspace.activation_codes,
                    workspace.activation_scales,
                    p.gate_up_weight_codes,
                    p.gate_up_weight_scales,
                    workspace.swiglu,
                )?;
                ops.dense_down.launch(
                    stream,
                    batch,
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

fn load_source_weights(
    arena: &DeviceArena,
    stream: &CudaStream,
    layout: &ResidentModelLayout,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> EngineResult<Vec<LayerScalars>> {
    let mut scalars = Vec::with_capacity(layout.layers.len());
    for (layer_index, layer) in layout.layers.iter().enumerate() {
        let mixer = load_mixer(arena, stream, layer_index, layer.mixer, snapshot)?;
        let mlp = load_mlp(arena, stream, layer_index, layer.mlp, snapshot)?;
        scalars.push(LayerScalars { mixer, mlp });
    }
    let endpoint = TextEndpointBindings::bind(snapshot)?;
    arena.copy_from_host(
        stream,
        layout.endpoint.final_norm,
        &endpoint.final_norm.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        layout.endpoint.lm_head_codes,
        endpoint.lm_head.codes(),
    )?;
    arena.copy_from_host(
        stream,
        layout.endpoint.lm_head_scales,
        &endpoint.lm_head_scale.words().collect::<Vec<_>>(),
    )?;

    Ok(scalars)
}

fn load_mixer(
    arena: &DeviceArena,
    stream: &CudaStream,
    layer: usize,
    weights: MixerWeights,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> EngineResult<MixerScalars> {
    match weights {
        MixerWeights::Gdn(weights) => {
            let source = GdnBindings::bind(snapshot, layer)?;
            arena.copy_from_host(
                stream,
                weights.input_norm,
                &source.input_norm.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(stream, weights.input_weight_codes, source.input_weight_e4m3)?;
            arena.copy_from_host(
                stream,
                weights.input_weight_scales,
                &little_endian_words(source.input_scale_bf16)?,
            )?;
            let mut control = source.a_control_weight.words().collect::<Vec<_>>();
            control.extend(source.b_control_weight.words());
            arena.copy_from_host(stream, weights.control_weights, &control)?;
            arena.copy_from_host(
                stream,
                weights.a_log,
                &source.a_log.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.dt_bias,
                &source.dt_bias.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.convolution_weights,
                &source.convolution_weight.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.recurrent_norm,
                &source.norm.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.output_weight_codes,
                source.output_weight.codes(),
            )?;
            arena.copy_from_host(
                stream,
                weights.output_weight_scales,
                &source.output_scale.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.post_attention_norm,
                &source.post_attention_norm.words().collect::<Vec<_>>(),
            )?;
            Ok(MixerScalars::Gdn)
        }
        MixerWeights::Attention(weights) => {
            let qkv = FullAttentionQkvBindings::bind(snapshot, layer)?.materialize()?;
            let source = FullAttentionPostBindings::bind(snapshot, layer)?;
            arena.copy_from_host(
                stream,
                weights.input_norm,
                &source.input_norm.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(stream, weights.qkv_weight_codes, &qkv.weight_e4m3)?;
            arena.copy_from_host(
                stream,
                weights.qkv_weight_scales,
                &little_endian_words(&qkv.scale_bf16)?,
            )?;
            arena.copy_from_host(
                stream,
                weights.query_norm,
                &source.query_norm.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.key_norm,
                &source.key_norm.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.output_weight_codes,
                source.output_weight.codes(),
            )?;
            arena.copy_from_host(
                stream,
                weights.output_weight_scales,
                &source.output_scale.words().collect::<Vec<_>>(),
            )?;
            arena.copy_from_host(
                stream,
                weights.post_attention_norm,
                &source.post_attention_norm.words().collect::<Vec<_>>(),
            )?;
            Ok(MixerScalars::Attention(AttentionScalars {
                key_cache_scale: bf16_to_f32(source.key_cache_scale_bf16),
                value_cache_scale: bf16_to_f32(source.value_cache_scale_bf16),
            }))
        }
    }
}

fn load_mlp(
    arena: &DeviceArena,
    stream: &CudaStream,
    layer: usize,
    weights: MlpWeights,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> EngineResult<MlpScalars> {
    match weights {
        MlpWeights::Nvfp4(weights) => {
            let gate_up = Nvfp4GateUpBindings::bind(snapshot, layer)?.materialize()?;
            let down = Nvfp4DownBindings::bind(snapshot, layer)?.materialize()?;
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
            let gate_up = DenseFp8GateUpBindings::bind(snapshot, layer)?;
            let down = DenseFp8DownBindings::bind(snapshot, layer)?;
            arena.copy_from_host(stream, weights.gate_up_weight_codes, gate_up.weight_e4m3)?;
            arena.copy_from_host(
                stream,
                weights.gate_up_weight_scales,
                &little_endian_words(gate_up.scale_bf16)?,
            )?;
            arena.copy_from_host(stream, weights.down_weight_codes, down.weight.codes())?;
            arena.copy_from_host(
                stream,
                weights.down_weight_scales,
                &down.scale.words().collect::<Vec<_>>(),
            )?;
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
    arena.copy_from_host(
        stream,
        workspace.state_rows,
        &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
    )?;
    let mut block_tables = vec![u32::MAX; MAX_BATCH * LONG_CONTEXT_PHYSICAL_PAGES];
    for slot in 0..MAX_BATCH {
        for logical_page in 0..SHORT_CONTEXT_PAGES_PER_SLOT {
            block_tables[slot * LONG_CONTEXT_PHYSICAL_PAGES + logical_page] =
                u32::try_from(slot * SHORT_CONTEXT_PAGES_PER_SLOT + logical_page)
                    .map_err(|_| EngineError::layout("resident physical page exceeds u32"))?;
        }
    }
    kv_arena.copy_from_host(stream, layout.kv_layout.block_tables(), &block_tables)?;
    arena.copy_from_host(
        stream,
        workspace.table_rows,
        &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
    )?;
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

fn fill_cache_prefix(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: ArenaRegion<u8>,
) -> EngineResult<()> {
    arena.fill_slice(
        stream,
        region,
        0,
        cache_values(SHORT_CONTEXT_PHYSICAL_PAGES)?,
        0,
    )?;
    Ok(())
}

fn fill_cache_slot(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: ArenaRegion<u8>,
    slot: usize,
) -> EngineResult<()> {
    let first_page = product(
        "resident slot first cache page",
        slot,
        SHORT_CONTEXT_PAGES_PER_SLOT,
    )?;
    arena.fill_slice(
        stream,
        region,
        cache_values(first_page)?,
        cache_values(SHORT_CONTEXT_PAGES_PER_SLOT)?,
        0,
    )?;
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

fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(EngineError::layout(
            "resident BF16 source plane has an odd byte length",
        ));
    }
    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
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

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{bf16_to_f32, decode_lengths, little_endian_words, require_batch, slot_rows};
    use crate::EngineErrorCode;

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
    fn decode_lengths_enforce_the_current_cache_capacity() {
        assert_eq!(
            &decode_lengths(&[0, 63, 191], 192).unwrap()[..3],
            [1, 64, 192]
        );
        assert_eq!(
            decode_lengths(&[192], 192).unwrap_err().code(),
            Some(EngineErrorCode::Route)
        );
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
    fn represented_source_words_are_not_numerically_reencoded() {
        assert_eq!(
            little_endian_words(&[0x80, 0x3f, 0x00, 0xbf]).unwrap(),
            [0x3f80, 0xbf00]
        );
        assert_eq!(bf16_to_f32(0x3f80).to_bits(), 1.0f32.to_bits());
        assert!(little_endian_words(&[0]).is_err());
    }
}

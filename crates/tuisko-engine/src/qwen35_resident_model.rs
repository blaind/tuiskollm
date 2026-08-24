//! Resident composition of every Qwen3.5 text layer and endpoint.

use crate::{
    EngineError, EngineResult, MAX_BATCH, Qwen35FullAttentionLayerLayout,
    Qwen35FullAttentionLayerProgram, Qwen35GdnLayerLayout, Qwen35GdnLayerProgram,
    Qwen35TextEndpointLayout, Qwen35TextEndpointProgram,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

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
        let cache_bytes = product(
            "Qwen3.5 resident BF16 cache bytes",
            attention_layers,
            attention.cache_bytes(),
        )?;
        let workspace_bytes = sum_products(
            "Qwen3.5 resident workspace bytes",
            &[
                (gdn_layers, gdn.workspace_bytes()),
                (attention_layers, attention.workspace_bytes()),
                (1, endpoint.workspace_bytes()),
            ],
        )?;
        let arena_bytes = sum_products(
            "Qwen3.5 resident arena bytes",
            &[
                (gdn_layers, gdn.arena_bytes()),
                (attention_layers, attention.arena_bytes()),
                (1, endpoint.arena_bytes()),
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

    /// BF16 K/V bytes across all eight full-attention layers.
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

    /// Aggregate alignment padding across the 33 stable arenas.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes - self.owner_bytes()
    }

    /// Number of independently allocated, address-stable device arenas.
    pub const fn arena_count(&self) -> usize {
        Qwen35_9B::LAYERS + 1
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

    /// Fixed short-context capacity of each full-attention slot.
    pub const fn context_capacity(&self) -> usize {
        192
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
    ) -> EngineResult<Self> {
        match layer_kind(layer) {
            Qwen35ResidentLayerKind::Gdn => Ok(Self::Gdn(Box::new(
                Qwen35GdnLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
            Qwen35ResidentLayerKind::FullAttention => Ok(Self::FullAttention(Box::new(
                Qwen35FullAttentionLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
        }
    }

    fn reset(&self, stream: &CudaStream) -> EngineResult<()> {
        match self {
            Self::Gdn(program) => program.reset_state(stream),
            Self::FullAttention(program) => program.reset_cache(stream),
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

    /// # Safety
    /// `input` must name the active BF16 residual rows in the shared CUDA context.
    unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        match self {
            Self::Gdn(program) => unsafe { program.launch_from(stream, batch, input) },
            Self::FullAttention(program) => unsafe { program.launch_from(stream, batch, input) },
        }
    }

    fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        match self {
            Self::Gdn(program) => program.read_residual(stream, batch),
            Self::FullAttention(program) => program.read_residual(stream, batch),
        }
    }

    fn base_address(&self) -> u64 {
        match self {
            Self::Gdn(program) => program.base_address(),
            Self::FullAttention(program) => program.base_address(),
        }
    }
}

/// Every Qwen3.5 text layer and the BF16 endpoint held resident at stable addresses.
pub struct Qwen35ResidentModelProgram {
    // Drop whole-model graphs before the layer arenas and loaded modules they retain.
    graphs: [CudaGraph; MAX_BATCH],
    layers: Vec<ResidentLayer>,
    endpoint: Qwen35TextEndpointProgram,
    context: Arc<CudaContext>,
    layout: Qwen35ResidentModelLayout,
}

impl Qwen35ResidentModelProgram {
    /// Loads all source weights and captures one immutable graph per exact batch.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let layout = Qwen35ResidentModelLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut layers = Vec::with_capacity(Qwen35_9B::LAYERS);
        for layer in 0..Qwen35_9B::LAYERS {
            layers.push(ResidentLayer::from_snapshot(
                context,
                Arc::clone(&snapshot),
                layer,
            )?);
        }
        let endpoint = Qwen35TextEndpointProgram::from_snapshot(context, snapshot)?;
        for layer in &layers {
            layer.reset(&stream)?;
        }
        let graphs = capture_routes(&stream, &layers, &endpoint)?;
        let program = Self {
            graphs,
            layers,
            endpoint,
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

    /// Clears every GDN history/state and full-attention BF16 cache plane.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        for layer in &self.layers {
            layer.reset(stream)?;
        }

        Ok(())
    }

    /// Replays the immutable whole-model graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen35ResidentModelProgram owns every captured allocation
        // (layer owners and endpoint) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 full-vocabulary logits.
    pub fn read_logits(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        self.endpoint.read_logits(stream, batch)
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

    /// CUDA context shared by every resident owner.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address of every layer arena followed by the endpoint arena.
    pub fn base_addresses(&self) -> Vec<u64> {
        self.layers
            .iter()
            .map(ResidentLayer::base_address)
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
        self.endpoint.host_stager_bytes()
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
        let endpoint = self.endpoint.qualification_observables(stream)?;
        let final_residual = self
            .layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.5 resident layer inventory is empty"))?
            .read_residual(stream, MAX_BATCH)?;

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

    fn require_accounting(&self) -> EngineResult<()> {
        let arena_bytes = self.layers.iter().try_fold(0usize, |total, layer| {
            let bytes = match layer {
                ResidentLayer::Gdn(program) => program.arena_bytes(),
                ResidentLayer::FullAttention(program) => program.arena_bytes(),
            };
            checked_sum("Qwen3.5 resident program arena bytes", total, bytes)
        })?;
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

fn capture_routes(
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
    use super::{Qwen35ResidentLayerKind, Qwen35ResidentModelLayout};

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
        assert_eq!(layout.cache_bytes(), 50_331_648);
        assert_eq!(layout.workspace_bytes(), 453_339_392);
        assert_eq!(layout.owner_bytes(), 6_435_491_072);
        assert_eq!(layout.padding_bytes(), 21_248);
        assert_eq!(layout.arena_bytes(), 6_435_512_320);
        assert_eq!(layout.arena_count(), 33);
        assert_eq!(
            layout.source_mapped_embedding_bytes().unwrap(),
            2_034_237_440
        );
        assert_eq!(layout.context_capacity(), 192);
    }
}

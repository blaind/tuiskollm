//! Resident composition of every Qwen3.6 text layer and endpoint.

use crate::{
    EngineError, EngineResult, MAX_BATCH, Qwen36FullAttentionLayerLayout,
    Qwen36FullAttentionLayerProgram, Qwen36GdnMoeLayerLayout, Qwen36GdnMoeLayerProgram,
    Qwen36TextEndpointLayout, Qwen36TextEndpointProgram,
};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuResult};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen36Moe35B};

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
        let cache_bytes = product(
            "Qwen3.6 resident BF16 cache bytes",
            attention_layers,
            attention.cache_bytes(),
        )?;
        let workspace_bytes = sum_products(
            "Qwen3.6 resident workspace bytes",
            &[
                (gdn_layers, gdn.workspace_bytes()),
                (attention_layers, attention.workspace_bytes()),
                (1, endpoint.workspace_bytes()),
            ],
        )?;
        let arena_bytes = sum_products(
            "Qwen3.6 resident arena bytes",
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

    /// Source route for one layer, or `None` outside `0..40`.
    pub fn layer_kind(&self, layer: usize) -> Option<Qwen36ResidentLayerKind> {
        self.layers.get(layer).copied()
    }

    /// Source-backed decoder, final-norm, and NVFP4 LM-head device bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// BF16 K/V bytes across all ten full-attention layers.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable per-layer state and working bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Complete device allocation across the 40 layer arenas and endpoint arena.
    pub const fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }

    /// All represented device bytes, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Aggregate alignment padding across the 41 stable arenas.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes - self.owner_bytes()
    }

    /// Number of independently allocated, address-stable device arenas.
    pub const fn arena_count(&self) -> usize {
        Qwen36Moe35B::LAYERS + 1
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

    /// Fixed short-context capacity of each full-attention slot.
    pub const fn context_capacity(&self) -> usize {
        192
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
    ) -> EngineResult<Self> {
        match layer_kind(layer) {
            Qwen36ResidentLayerKind::GdnMoe => Ok(Self::GdnMoe(Box::new(
                Qwen36GdnMoeLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
            Qwen36ResidentLayerKind::FullAttentionMoe => Ok(Self::FullAttentionMoe(Box::new(
                Qwen36FullAttentionLayerProgram::from_snapshot(context, snapshot, layer)?,
            ))),
        }
    }

    fn reset(&self, stream: &CudaStream) -> EngineResult<()> {
        match self {
            Self::GdnMoe(program) => program.reset_state(stream),
            Self::FullAttentionMoe(program) => program.reset_cache(stream),
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
    // Drop whole-model graphs before the layer arenas and loaded modules they retain.
    graphs: [CudaGraph; MAX_BATCH],
    layers: Vec<ResidentLayer>,
    endpoint: Qwen36TextEndpointProgram,
    context: Arc<CudaContext>,
    layout: Qwen36ResidentModelLayout,
}

impl Qwen36ResidentModelProgram {
    /// Loads all source weights and captures one immutable graph per exact batch.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    ) -> EngineResult<Self> {
        let layout = Qwen36ResidentModelLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut layers = Vec::with_capacity(Qwen36Moe35B::LAYERS);
        for layer in 0..Qwen36Moe35B::LAYERS {
            layers.push(ResidentLayer::from_snapshot(
                context,
                Arc::clone(&snapshot),
                layer,
            )?);
        }
        let endpoint = Qwen36TextEndpointProgram::from_snapshot(context, snapshot)?;
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
        // SAFETY: this Qwen36ResidentModelProgram owns every captured allocation
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
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
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
    pub const fn layout(&self) -> &Qwen36ResidentModelLayout {
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
    ) -> EngineResult<Qwen36ResidentModelObservables> {
        let endpoint = self.endpoint.qualification_observables(stream)?;
        let final_residual = self
            .layers
            .last()
            .ok_or_else(|| EngineError::layout("Qwen3.6 resident layer inventory is empty"))?
            .read_residual(stream, MAX_BATCH)?;

        Ok(Qwen36ResidentModelObservables {
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

    fn require_accounting(&self) -> EngineResult<()> {
        let arena_bytes = self.layers.iter().try_fold(0usize, |total, layer| {
            checked_sum(
                "Qwen3.6 resident program arena bytes",
                total,
                layer.arena_bytes(),
            )
        })?;
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

fn capture_routes(
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
    use super::{Qwen36ResidentLayerKind, Qwen36ResidentModelLayout};

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
        assert_eq!(layout.cache_bytes(), 31_457_280);
        assert_eq!(layout.workspace_bytes(), 737_446_176);
        assert_eq!(layout.owner_bytes(), 20_576_939_552);
        assert_eq!(layout.padding_bytes(), 41_440);
        assert_eq!(layout.arena_bytes(), 20_576_980_992);
        assert_eq!(layout.arena_count(), 41);
        assert_eq!(
            layout.source_mapped_embedding_bytes().unwrap(),
            1_017_118_720
        );
        assert_eq!(layout.context_capacity(), 192);
    }
}

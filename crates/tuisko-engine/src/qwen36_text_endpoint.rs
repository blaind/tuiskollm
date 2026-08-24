//! Resident Qwen3.6 final normalization and NVFP4 LM head.

use crate::{EngineError, EngineResult, MAX_BATCH, Qwen36TextEndpointLayout};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{Qwen36Nvfp4LmHeadOp, Qwen36ResidualNormOp};
use tuisko_model::{Arch, Bf16View, CheckpointSnapshot, Qwen36Moe35B, Qwen36TextEndpointBindings};

/// Source-backed Qwen3.6 endpoint with immutable exact-batch graphs.
pub struct Qwen36TextEndpointProgram {
    // Graphs must drop before the arena and loaded modules they reference.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: Qwen36ResidualNormOp,
    _lm_head: Qwen36Nvfp4LmHeadOp,
    embedding_stager: PinnedHostBuffer<u16>,
    snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    context: Arc<CudaContext>,
    layout: Qwen36TextEndpointLayout,
    base_address: u64,
    _lm_head_weight_scale_2: f32,
}

#[derive(Clone, Copy)]
struct EndpointPointers {
    input: *const u16,
    final_norm_weight: *const u16,
    normalized: *mut u16,
    lm_head_weight_codes: *const u8,
    lm_head_weight_scales: *const u8,
    logits: *mut u16,
}

impl EndpointPointers {
    fn bind(arena: &DeviceArena, layout: &Qwen36TextEndpointLayout) -> GpuResult<Self> {
        Ok(Self {
            input: arena.address(layout.input())?.cast_const(),
            final_norm_weight: arena.address(layout.final_norm_weight())?.cast_const(),
            normalized: arena.address(layout.normalized())?,
            lm_head_weight_codes: arena.address(layout.lm_head_weight_codes())?.cast_const(),
            lm_head_weight_scales: arena.address(layout.lm_head_weight_scales())?.cast_const(),
            logits: arena.address(layout.logits())?,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 6] {
        [
            self.input.addr(),
            self.final_norm_weight.addr(),
            self.normalized.addr(),
            self.lm_head_weight_codes.addr(),
            self.lm_head_weight_scales.addr(),
            self.logits.addr(),
        ]
    }
}

impl Qwen36TextEndpointProgram {
    /// Loads source endpoint weights and captures one graph per exact batch.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    ) -> EngineResult<Self> {
        let materialized = Qwen36TextEndpointBindings::bind(snapshot.as_ref())?.materialize()?;
        let layout = Qwen36TextEndpointLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen36ResidualNormOp::new(context)?;
        let lm_head = Qwen36Nvfp4LmHeadOp::new(context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(context, MAX_BATCH * Qwen36Moe35B::HIDDEN)
            .map_err(GpuError::from)?;

        arena.copy_region_bytes_from_host(
            &stream,
            layout.final_norm_weight(),
            materialized.final_norm.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            layout.lm_head_weight_codes(),
            materialized.lm_head_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            layout.lm_head_weight_scales(),
            &materialized.lm_head_scale_e4m3_swizzled,
        )?;

        let pointers = EndpointPointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let lm_head_weight_scale_2 = materialized.lm_head_weight_scale_2;
        let graphs = capture_routes(&stream, &norm, &lm_head, pointers, lm_head_weight_scale_2)?;

        Ok(Self {
            graphs,
            arena,
            _norm: norm,
            _lm_head: lm_head,
            embedding_stager,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            _lm_head_weight_scale_2: lm_head_weight_scale_2,
        })
    }

    /// Copies exact mmap-backed embedding rows through the pinned stager.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        require_batch(token_ids.len())?;
        let embedding = Qwen36TextEndpointBindings::bind_embedding(self.snapshot.as_ref())?;
        for (row, &token) in token_ids.iter().enumerate() {
            let token = usize::try_from(token)
                .map_err(|_| EngineError::route("token identifier exceeds host width"))?;
            if token >= Qwen36Moe35B::VOCAB {
                return Err(EngineError::route(format!(
                    "token identifier {token} is outside vocabulary 0..{}",
                    Qwen36Moe35B::VOCAB
                )));
            }
            copy_embedding_row(
                embedding,
                token,
                &mut self.embedding_stager
                    [row * Qwen36Moe35B::HIDDEN..(row + 1) * Qwen36Moe35B::HIDDEN],
            )?;
        }
        let active = token_ids.len() * Qwen36Moe35B::HIDDEN;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.input(),
            &self.embedding_stager[..active],
        )?;

        Ok(())
    }

    /// Replays the immutable graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen36TextEndpointProgram owns every captured allocation
        // (arena, pinned stager, op modules) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;
        Ok(())
    }

    /// Reads active BF16 logits for one exact batch.
    pub fn read_logits(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        Ok(self.arena.copy_prefix_to_host(
            stream,
            self.layout.logits(),
            batch * Qwen36Moe35B::VOCAB,
        )?)
    }

    /// Reads active BF16 logits into one reusable host allocation.
    pub fn read_logits_into(
        &self,
        stream: &CudaStream,
        batch: usize,
        destination: &mut [u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let expected = batch
            .checked_mul(Qwen36Moe35B::VOCAB)
            .ok_or_else(|| EngineError::layout("Qwen3.6 endpoint logit count overflows"))?;
        if destination.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.6 endpoint logit destination has {} values, expected {expected} for B={batch}",
                destination.len()
            )));
        }
        self.arena
            .copy_prefix_to_host_slice(stream, self.layout.logits(), destination)?;
        Ok(())
    }

    /// CUDA context shared by the program and all resources.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every graph.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Source-backed device weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Address-stable device workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete arena allocation, including padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Page-locked host bytes used to gather embeddings.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Checked endpoint layout.
    pub const fn layout(&self) -> &Qwen36TextEndpointLayout {
        &self.layout
    }

    #[allow(dead_code)]
    pub(crate) fn input_address(&self) -> GpuResult<*const u16> {
        Ok(EndpointPointers::bind(&self.arena, &self.layout)?.input)
    }

    /// Launches the endpoint from another resident owner's BF16 residual plane.
    ///
    /// # Safety
    /// `input` covers `batch * 2_048` BF16 values in this CUDA context.
    #[allow(dead_code)]
    pub(crate) unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
    ) -> GpuResult<()> {
        let mut pointers = EndpointPointers::bind(&self.arena, &self.layout)?;
        pointers.input = input;
        launch_route(
            stream,
            batch,
            &self._norm,
            &self._lm_head,
            pointers,
            self._lm_head_weight_scale_2,
        )
    }

    #[cfg(feature = "qualification")]
    /// Launches the same route eagerly for graph-agreement checks.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_route(
            stream,
            batch,
            &self._norm,
            &self._lm_head,
            EndpointPointers::bind(&self.arena, &self.layout)?,
            self._lm_head_weight_scale_2,
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns the production graph for benchmark registration.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;
        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated eager routes for intrinsic endpoint timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.6 endpoint graph requires at least one operation",
            ));
        }
        let pointers = EndpointPointers::bind(&self.arena, &self.layout)?;
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(
                    stream,
                    batch,
                    &self._norm,
                    &self._lm_head,
                    pointers,
                    self._lm_head_weight_scale_2,
                )?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every checked arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 6]> {
        Ok(EndpointPointers::bind(&self.arena, &self.layout)?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Fills mutable output planes before an exact route.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        self.arena.fill(stream, self.layout.normalized(), byte)?;
        self.arena.fill(stream, self.layout.logits(), byte)?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads complete mutable planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36EndpointObservables> {
        Ok(Qwen36EndpointObservables {
            input: self.arena.copy_to_host(stream, self.layout.input())?,
            normalized: self.arena.copy_to_host(stream, self.layout.normalized())?,
            logits: self.arena.copy_to_host(stream, self.layout.logits())?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads every immutable resident endpoint plane.
    pub fn qualification_immutable(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36EndpointImmutable> {
        Ok(Qwen36EndpointImmutable {
            final_norm: self
                .arena
                .copy_to_host(stream, self.layout.final_norm_weight())?,
            lm_head_weight_codes: self
                .arena
                .copy_to_host(stream, self.layout.lm_head_weight_codes())?,
            lm_head_weight_scales: self
                .arena
                .copy_to_host(stream, self.layout.lm_head_weight_scales())?,
            lm_head_weight_scale_2: self._lm_head_weight_scale_2,
        })
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen36EndpointObservables {
    /// Staged BF16 embedding rows.
    pub input: Vec<u16>,
    /// Final-normalized BF16 rows.
    pub normalized: Vec<u16>,
    /// Full-vocabulary BF16 logits.
    pub logits: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Immutable resident planes exposed to the qualification crate.
pub struct Qwen36EndpointImmutable {
    /// Final RMSNorm weights.
    pub final_norm: Vec<u16>,
    /// Packed E2M1 LM-head words.
    pub lm_head_weight_codes: Vec<u8>,
    /// Swizzled E4M3 LM-head scales.
    pub lm_head_weight_scales: Vec<u8>,
    /// Source second-stage LM-head weight scale.
    pub lm_head_weight_scale_2: f32,
}

fn capture_routes(
    stream: &CudaStream,
    norm: &Qwen36ResidualNormOp,
    lm_head: &Qwen36Nvfp4LmHeadOp,
    pointers: EndpointPointers,
    weight_scale_2: f32,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, norm, lm_head, pointers, weight_scale_2)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("Qwen3.6 endpoint graph inventory is incomplete"))
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    norm: &Qwen36ResidualNormOp,
    lm_head: &Qwen36Nvfp4LmHeadOp,
    pointers: EndpointPointers,
    weight_scale_2: f32,
) -> GpuResult<()> {
    unsafe {
        norm.launch_plain(
            stream,
            batch,
            pointers.input,
            pointers.final_norm_weight,
            pointers.normalized,
        )?;
        lm_head.launch(
            stream,
            batch,
            pointers.normalized,
            pointers.lm_head_weight_codes,
            pointers.lm_head_weight_scales,
            weight_scale_2,
            pointers.logits,
        )
    }
}

fn copy_embedding_row(
    embedding: Bf16View<'_, 2>,
    token: usize,
    destination: &mut [u16],
) -> EngineResult<()> {
    let byte_begin = token
        .checked_mul(Qwen36Moe35B::HIDDEN)
        .and_then(|words| words.checked_mul(2))
        .ok_or_else(|| EngineError::layout("Qwen3.6 embedding offset overflows"))?;
    let byte_end = byte_begin
        .checked_add(Qwen36Moe35B::HIDDEN * 2)
        .ok_or_else(|| EngineError::layout("Qwen3.6 embedding range overflows"))?;
    let source = embedding
        .bytes()
        .get(byte_begin..byte_end)
        .ok_or_else(|| EngineError::layout(format!("embedding row {token} is outside its view")))?;
    for (target, bytes) in destination.iter_mut().zip(source.as_chunks::<2>().0) {
        *target = u16::from_le_bytes(*bytes);
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

#[cfg(test)]
mod tests {
    use super::require_batch;
    use crate::EngineErrorCode;

    #[test]
    fn only_exact_decode_batches_are_admitted() {
        for batch in 1..=8 {
            require_batch(batch).unwrap();
        }
        for batch in [0, 9, 16, usize::MAX] {
            let error = require_batch(batch).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
            assert!(error.to_string().contains(&format!("batch {batch}")));
        }
    }
}

//! Resident Qwen3.5 final normalization and BF16 LM head.

use crate::{EngineError, EngineResult, MAX_BATCH, Qwen35TextEndpointLayout};
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{Qwen35Bf16LmHeadOp, Qwen35ResidualNormOp};
use tuisko_model::{Arch, Bf16TextEndpointBindings, Bf16View, CheckpointSnapshot, Qwen35_9B};

/// Source-backed Qwen3.5 endpoint with immutable exact-batch graphs.
pub struct Qwen35TextEndpointProgram {
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    norm: Qwen35ResidualNormOp,
    lm_head: Qwen35Bf16LmHeadOp,
    embedding_stager: PinnedHostBuffer<u16>,
    snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    context: Arc<CudaContext>,
    layout: Qwen35TextEndpointLayout,
    base_address: u64,
}

#[derive(Clone, Copy)]
struct EndpointPointers {
    input: *const u16,
    final_norm_weight: *const u16,
    normalized: *mut u16,
    lm_head_weight: *const u16,
    logits: *mut u16,
}

impl EndpointPointers {
    fn bind(arena: &DeviceArena, layout: &Qwen35TextEndpointLayout) -> GpuResult<Self> {
        Ok(Self {
            input: arena.address(layout.input())?.cast_const(),
            final_norm_weight: arena.address(layout.final_norm_weight())?.cast_const(),
            normalized: arena.address(layout.normalized())?,
            lm_head_weight: arena.address(layout.lm_head_weight())?.cast_const(),
            logits: arena.address(layout.logits())?,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 5] {
        [
            self.input.addr(),
            self.final_norm_weight.addr(),
            self.normalized.addr(),
            self.lm_head_weight.addr(),
            self.logits.addr(),
        ]
    }
}

impl Qwen35TextEndpointProgram {
    /// Loads the source endpoint and captures one graph per exact batch.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    ) -> EngineResult<Self> {
        let bindings = Bf16TextEndpointBindings::bind(snapshot.as_ref())?;
        let layout = Qwen35TextEndpointLayout::build()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen35ResidualNormOp::new(context)?;
        let lm_head = Qwen35Bf16LmHeadOp::new(context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(context, MAX_BATCH * Qwen35_9B::HIDDEN)
            .map_err(GpuError::from)?;

        arena.copy_region_bytes_from_host(
            &stream,
            layout.final_norm_weight(),
            bindings.final_norm.bytes(),
        )?;
        arena.copy_region_bytes_from_host(
            &stream,
            layout.lm_head_weight(),
            bindings.lm_head.bytes(),
        )?;

        let pointers = EndpointPointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let graphs = capture_routes(&stream, &norm, &lm_head, pointers)?;

        Ok(Self {
            graphs,
            arena,
            norm,
            lm_head,
            embedding_stager,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
        })
    }

    /// Copies exact mmap-backed embedding rows through the pinned stager.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        require_batch(token_ids.len())?;
        let embedding = Bf16TextEndpointBindings::bind(self.snapshot.as_ref())?.embedding;
        for (row, &token) in token_ids.iter().enumerate() {
            let token = usize::try_from(token)
                .map_err(|_| EngineError::route("token identifier exceeds host width"))?;
            if token >= Qwen35_9B::VOCAB {
                return Err(EngineError::route(format!(
                    "token identifier {token} is outside vocabulary 0..{}",
                    Qwen35_9B::VOCAB
                )));
            }
            copy_embedding_row(
                embedding,
                token,
                &mut self.embedding_stager[row * Qwen35_9B::HIDDEN..(row + 1) * Qwen35_9B::HIDDEN],
            )?;
        }
        let active = token_ids.len() * Qwen35_9B::HIDDEN;
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
        // SAFETY: this Qwen35TextEndpointProgram owns every captured allocation
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
            batch * Qwen35_9B::VOCAB,
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
            .checked_mul(Qwen35_9B::VOCAB)
            .ok_or_else(|| EngineError::layout("Qwen3.5 endpoint logit count overflows"))?;
        if destination.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.5 endpoint logit destination has {} values, expected {expected} for B={batch}",
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
    pub const fn layout(&self) -> &Qwen35TextEndpointLayout {
        &self.layout
    }

    pub(crate) fn input_address(&self) -> GpuResult<*const u16> {
        Ok(EndpointPointers::bind(&self.arena, &self.layout)?.input)
    }

    /// Launches the endpoint from another resident owner's BF16 residual plane.
    ///
    /// # Safety
    /// `input` must address at least `batch * Qwen35_9B::HIDDEN` BF16 values in this context.
    pub(crate) unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
    ) -> GpuResult<()> {
        let mut pointers = EndpointPointers::bind(&self.arena, &self.layout)?;
        pointers.input = input;
        launch_route(stream, batch, &self.norm, &self.lm_head, pointers)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Launches the same route eagerly for graph-agreement checks.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_route(
            stream,
            batch,
            &self.norm,
            &self.lm_head,
            EndpointPointers::bind(&self.arena, &self.layout)?,
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
                "repeated Qwen3.5 endpoint graph requires at least one operation",
            ));
        }
        let pointers = EndpointPointers::bind(&self.arena, &self.layout)?;

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, batch, &self.norm, &self.lm_head, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every checked arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 5]> {
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
    /// Reads complete input and output planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen35EndpointObservables> {
        Ok(Qwen35EndpointObservables {
            input: self.arena.copy_to_host(stream, self.layout.input())?,
            normalized: self.arena.copy_to_host(stream, self.layout.normalized())?,
            logits: self.arena.copy_to_host(stream, self.layout.logits())?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads selected immutable BF16 LM-head rows.
    pub fn qualification_lm_head_row(
        &self,
        stream: &CudaStream,
        row: usize,
    ) -> EngineResult<Vec<u16>> {
        if row >= Qwen35_9B::VOCAB {
            return Err(EngineError::route(format!(
                "LM-head row {row} is outside vocabulary"
            )));
        }
        Ok(self.arena.copy_slice_to_host(
            stream,
            self.layout.lm_head_weight(),
            row * Qwen35_9B::HIDDEN,
            Qwen35_9B::HIDDEN,
        )?)
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen35EndpointObservables {
    /// Staged BF16 embedding rows.
    pub input: Vec<u16>,
    /// Final-normalized BF16 rows.
    pub normalized: Vec<u16>,
    /// Full-vocabulary BF16 logits.
    pub logits: Vec<u16>,
}

fn capture_routes(
    stream: &CudaStream,
    norm: &Qwen35ResidualNormOp,
    lm_head: &Qwen35Bf16LmHeadOp,
    pointers: EndpointPointers,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, norm, lm_head, pointers)
        })?);
    }
    graphs
        .try_into()
        .map_err(|_| EngineError::layout("Qwen3.5 endpoint graph inventory is incomplete"))
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    norm: &Qwen35ResidualNormOp,
    lm_head: &Qwen35Bf16LmHeadOp,
    pointers: EndpointPointers,
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
            pointers.lm_head_weight,
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
        .checked_mul(Qwen35_9B::HIDDEN)
        .and_then(|words| words.checked_mul(2))
        .ok_or_else(|| EngineError::layout("Qwen3.5 embedding offset overflows"))?;
    let byte_end = byte_begin
        .checked_add(Qwen35_9B::HIDDEN * 2)
        .ok_or_else(|| EngineError::layout("Qwen3.5 embedding range overflows"))?;
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

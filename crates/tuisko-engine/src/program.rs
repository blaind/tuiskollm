//! Resident final-normalization and LM-head program.

use crate::{EndpointLayout, EngineError, EngineResult, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, PinnedHostBuffer,
};
use tuisko_kernels_sm120::{LmHeadOp, ResidualNormOp, Sm120Arch};
use tuisko_model::{Arch, Bf16View, CheckpointSnapshot, Qwen38_27B, TextEndpointBindings};

/// Source-backed endpoint program with immutable exact-batch CUDA Graphs.
pub struct TextEndpointProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: ResidualNormOp<A>,
    _lm_head: LmHeadOp<A>,
    embedding_stager: PinnedHostBuffer<u16>,
    snapshot: Arc<CheckpointSnapshot<A>>,
    context: Arc<CudaContext>,
    layout: EndpointLayout,
    base_address: u64,
    arch: PhantomData<A>,
}

#[derive(Clone, Copy)]
struct EndpointPointers {
    input: *const u16,
    final_norm_weight: *const u16,
    normalized: *mut u16,
    activation_codes: *mut u8,
    activation_scales: *mut f32,
    weight_codes: *const u8,
    weight_scales: *const u16,
    logits: *mut u16,
}

impl EndpointPointers {
    fn bind(arena: &DeviceArena, layout: &EndpointLayout) -> GpuResult<Self> {
        Ok(Self {
            input: arena.address(layout.input())?.cast_const(),
            final_norm_weight: arena.address(layout.final_norm_weight())?.cast_const(),
            normalized: arena.address(layout.normalized())?,
            activation_codes: arena.address(layout.activation_codes())?,
            activation_scales: arena.address(layout.activation_scales())?,
            weight_codes: arena.address(layout.weight_codes())?.cast_const(),
            weight_scales: arena.address(layout.weight_scales())?.cast_const(),
            logits: arena.address(layout.logits())?,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 8] {
        [
            self.input.addr(),
            self.final_norm_weight.addr(),
            self.normalized.addr(),
            self.activation_codes.addr(),
            self.activation_scales.addr(),
            self.weight_codes.addr(),
            self.weight_scales.addr(),
            self.logits.addr(),
        ]
    }
}

impl<A: Sm120Arch> TextEndpointProgram<A> {
    /// Loads source weights, allocates one arena, and captures `B=1..=8`.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<A>>,
    ) -> EngineResult<Self> {
        let bindings = TextEndpointBindings::bind(snapshot.as_ref())?;
        let layout = EndpointLayout::build::<A>()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let lm_head = LmHeadOp::new(context)?;
        let embedding_stager = PinnedHostBuffer::zeroed(
            context,
            checked_product("embedding stager elements", MAX_BATCH, A::HIDDEN)?,
        )
        .map_err(GpuError::from)?;

        arena.copy_from_host(
            &stream,
            layout.final_norm_weight(),
            &bindings.final_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, layout.weight_codes(), bindings.lm_head.codes())?;
        arena.copy_from_host(
            &stream,
            layout.weight_scales(),
            &bindings.lm_head_scale.words().collect::<Vec<_>>(),
        )?;

        let pointers = EndpointPointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let graphs = capture_routes(&stream, &norm, &lm_head, pointers)?;

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
            arch: PhantomData,
        })
    }

    /// Copies exact source embedding rows through the pinned host stager.
    pub fn stage_embeddings(&mut self, stream: &CudaStream, token_ids: &[u32]) -> EngineResult<()> {
        require_batch(token_ids.len())?;
        for &token in token_ids {
            let token = usize::try_from(token)
                .map_err(|_| EngineError::route("token identifier exceeds host width"))?;
            if token >= A::VOCAB {
                return Err(EngineError::route(format!(
                    "token identifier {token} is outside vocabulary 0..{}",
                    A::VOCAB
                )));
            }
        }
        let embedding = TextEndpointBindings::bind_embedding(self.snapshot.as_ref())?;
        let active = checked_product("active embedding elements", token_ids.len(), A::HIDDEN)?;

        for (row, &token) in token_ids.iter().enumerate() {
            copy_embedding_row::<A>(
                embedding,
                token as usize,
                &mut self.embedding_stager[row * A::HIDDEN..(row + 1) * A::HIDDEN],
            )?;
        }
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
        // SAFETY: this TextEndpointProgram owns every captured allocation (arena,
        // pinned embedding stager, op modules) for its whole life and drops the
        // graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 logits for one exact batch.
    pub fn read_logits(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = checked_product("active logit elements", batch, A::VOCAB)?;

        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.logits(), values)?)
    }

    /// CUDA context shared by the program and all of its resources.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every exact-batch graph.
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

    /// Complete arena allocation, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Page-locked host bytes used to gather embedding rows.
    pub fn host_stager_bytes(&self) -> usize {
        self.embedding_stager.num_bytes()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Checked resident layout.
    pub const fn layout(&self) -> &EndpointLayout {
        &self.layout
    }

    #[cfg(feature = "qualification")]
    /// Launches the same route eagerly for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        let pointers = EndpointPointers::bind(&self.arena, &self.layout)?;
        launch_route(stream, batch, &self._norm, &self._lm_head, pointers)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns the production graph for benchmark registration.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;

        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated eager routes for intrinsic path timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated endpoint graph requires at least one operation",
            ));
        }
        let pointers = EndpointPointers::bind(&self.arena, &self.layout)?;

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, batch, &self._norm, &self._lm_head, pointers)?;
            }

            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every checked arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 8]> {
        Ok(EndpointPointers::bind(&self.arena, &self.layout)?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Fills output planes with a byte sentinel before an exact route.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        self.arena.fill(stream, self.layout.normalized(), byte)?;
        self.arena
            .fill(stream, self.layout.activation_codes(), byte)?;
        self.arena
            .fill(stream, self.layout.activation_scales(), byte)?;
        self.arena.fill(stream, self.layout.logits(), byte)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every working plane, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<EndpointObservables> {
        Ok(EndpointObservables {
            input: self.arena.copy_to_host(stream, self.layout.input())?,
            normalized: self.arena.copy_to_host(stream, self.layout.normalized())?,
            activation_codes: self
                .arena
                .copy_to_host(stream, self.layout.activation_codes())?,
            activation_scales: self
                .arena
                .copy_to_host(stream, self.layout.activation_scales())?,
            logits: self.arena.copy_to_host(stream, self.layout.logits())?,
        })
    }
}

#[cfg(feature = "qualification")]
/// Working planes exposed only to the external qualification crate.
pub struct EndpointObservables {
    /// Staged BF16 embedding rows.
    pub input: Vec<u16>,
    /// Final-normalized BF16 rows.
    pub normalized: Vec<u16>,
    /// Dynamically quantized E4M3 activation codes.
    pub activation_codes: Vec<u8>,
    /// Dynamic FP32 activation scales.
    pub activation_scales: Vec<f32>,
    /// BF16 full-vocabulary logits.
    pub logits: Vec<u16>,
}

fn capture_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    lm_head: &LmHeadOp<A>,
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
        .map_err(|_| EngineError::layout("exact-batch graph inventory has the wrong cardinality"))
}

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    batch: usize,
    norm: &ResidualNormOp<A>,
    lm_head: &LmHeadOp<A>,
    pointers: EndpointPointers,
) -> GpuResult<()> {
    // SAFETY: every pointer names an aligned, non-overlapping region sized for
    // MAX_BATCH, and exact dispatch restricts each launch to `batch` rows.
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
            pointers.activation_codes,
            pointers.activation_scales,
            pointers.weight_codes,
            pointers.weight_scales,
            pointers.logits,
        )
    }
}

fn copy_embedding_row<A: Arch>(
    embedding: Bf16View<'_, 2>,
    token: usize,
    destination: &mut [u16],
) -> EngineResult<()> {
    if destination.len() != A::HIDDEN {
        return Err(EngineError::layout(format!(
            "embedding destination has {} words, expected {}",
            destination.len(),
            A::HIDDEN
        )));
    }
    let word_begin = checked_product("embedding row offset", token, A::HIDDEN)?;
    let byte_begin = checked_product("embedding byte offset", word_begin, 2)?;
    let byte_len = checked_product("embedding row bytes", A::HIDDEN, 2)?;
    let byte_end = byte_begin
        .checked_add(byte_len)
        .ok_or_else(|| EngineError::layout("embedding byte range overflows"))?;
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

fn checked_product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
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

//! Resident source-backed dense-FP8 MLP program.

use crate::{DenseFp8MlpLayout, EngineError, EngineResult, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{DenseFp8DownOp, DenseFp8SwiGluOp, ResidualNormOp, Sm120Arch};
use tuisko_model::{CheckpointSnapshot, DenseFp8MlpBindings, Qwen38_27B};

/// One late-layer dense-FP8 MLP with immutable exact-batch graph routes.
pub struct DenseFp8MlpProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: ResidualNormOp<A>,
    _swiglu: DenseFp8SwiGluOp<A>,
    _down: DenseFp8DownOp<A>,
    snapshot: Arc<CheckpointSnapshot<A>>,
    context: Arc<CudaContext>,
    layout: DenseFp8MlpLayout,
    base_address: u64,
    layer: usize,
    arch: PhantomData<A>,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    normalized: *mut u16,
    gate_up_activation_codes: *mut u8,
    gate_up_activation_scales: *mut f32,
    gate_up_weight_codes: *const u8,
    gate_up_weight_scales: *const u16,
    swiglu: *mut u16,
    down_activation_codes: *mut u8,
    down_activation_scales: *mut f32,
    down_weight_codes: *const u8,
    down_weight_scales: *const u16,
    branch: *mut u16,
    next_norm: *const u16,
    residual_output: *mut u16,
    next_normalized: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, layout: &DenseFp8MlpLayout) -> GpuResult<Self> {
        Ok(Self {
            residual_input: arena.address(layout.residual_input())?.cast_const(),
            input_norm: arena.address(layout.input_norm())?.cast_const(),
            normalized: arena.address(layout.normalized())?,
            gate_up_activation_codes: arena.address(layout.gate_up_activation_codes())?,
            gate_up_activation_scales: arena.address(layout.gate_up_activation_scales())?,
            gate_up_weight_codes: arena.address(layout.gate_up_weight_codes())?.cast_const(),
            gate_up_weight_scales: arena.address(layout.gate_up_weight_scales())?.cast_const(),
            swiglu: arena.address(layout.swiglu())?,
            down_activation_codes: arena.address(layout.down_activation_codes())?,
            down_activation_scales: arena.address(layout.down_activation_scales())?,
            down_weight_codes: arena.address(layout.down_weight_codes())?.cast_const(),
            down_weight_scales: arena.address(layout.down_weight_scales())?.cast_const(),
            branch: arena.address(layout.branch())?,
            next_norm: arena.address(layout.next_norm())?.cast_const(),
            residual_output: arena.address(layout.residual_output())?,
            next_normalized: arena.address(layout.next_normalized())?,
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 16] {
        [
            self.residual_input.addr(),
            self.input_norm.addr(),
            self.normalized.addr(),
            self.gate_up_activation_codes.addr(),
            self.gate_up_activation_scales.addr(),
            self.gate_up_weight_codes.addr(),
            self.gate_up_weight_scales.addr(),
            self.swiglu.addr(),
            self.down_activation_codes.addr(),
            self.down_activation_scales.addr(),
            self.down_weight_codes.addr(),
            self.down_weight_scales.addr(),
            self.branch.addr(),
            self.next_norm.addr(),
            self.residual_output.addr(),
            self.next_normalized.addr(),
        ]
    }
}

impl<A: Sm120Arch> DenseFp8MlpProgram<A> {
    /// Loads one admitted layer, allocates one arena, and captures B=1..8.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<A>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let bindings = DenseFp8MlpBindings::bind(snapshot.as_ref(), layer)?;
        let layout = DenseFp8MlpLayout::build::<A>()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let swiglu = DenseFp8SwiGluOp::new(context)?;
        let down = DenseFp8DownOp::new(context)?;

        arena.copy_from_host(
            &stream,
            layout.input_norm(),
            &bindings.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            layout.gate_up_weight_codes(),
            bindings.gate_up.weight_e4m3,
        )?;
        arena.copy_from_host(
            &stream,
            layout.gate_up_weight_scales(),
            &little_endian_words(bindings.gate_up.scale_bf16)?,
        )?;
        arena.copy_from_host(
            &stream,
            layout.down_weight_codes(),
            bindings.down.weight.codes(),
        )?;
        arena.copy_from_host(
            &stream,
            layout.down_weight_scales(),
            &bindings.down.scale.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            layout.next_norm(),
            &bindings.next_norm.words().collect::<Vec<_>>(),
        )?;

        let pointers = Pointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let graphs = capture_routes(&stream, &norm, &swiglu, &down, pointers)?;

        Ok(Self {
            graphs,
            arena,
            _norm: norm,
            _swiglu: swiglu,
            _down: down,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            layer,
            arch: PhantomData,
        })
    }

    /// Uploads exactly `batch` BF16 residual rows into stable input storage.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let expected = checked_product("dense-FP8 MLP input elements", batch, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "dense-FP8 MLP input has {} values, expected {expected} for B={batch}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.residual_input(), values)?;

        Ok(())
    }

    /// Replays the immutable graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        self.graphs[batch - 1].launch(stream)?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = checked_product("dense-FP8 MLP output elements", batch, A::HIDDEN)?;

        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.residual_output(), values)?)
    }

    /// Decoder layer owned by this program.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// CUDA context shared by the arena, graphs, and prepared operators.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable base address captured by every graph.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Exact source-backed device weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact address-stable workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete single allocation, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &DenseFp8MlpLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<A>> {
        &self.snapshot
    }

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        let pointers = Pointers::bind(&self.arena, &self.layout)?;
        launch_route(
            stream,
            batch,
            &self._norm,
            &self._swiglu,
            &self._down,
            pointers,
        )?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured production graph.
    pub fn qualification_graph(&self, batch: usize) -> EngineResult<&CudaGraph> {
        require_batch(batch)?;

        Ok(&self.graphs[batch - 1])
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated production paths for high-resolution device timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        batch: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_batch(batch)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated dense-FP8 MLP graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, &self.layout)?;

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(
                    stream,
                    batch,
                    &self._norm,
                    &self._swiglu,
                    &self._down,
                    pointers,
                )?;
            }

            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 16]> {
        Ok(Pointers::bind(&self.arena, &self.layout)?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Fills every observable output plane with a byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        self.arena.fill(stream, self.layout.normalized(), byte)?;
        self.arena
            .fill(stream, self.layout.gate_up_activation_codes(), byte)?;
        self.arena
            .fill(stream, self.layout.gate_up_activation_scales(), byte)?;
        self.arena.fill(stream, self.layout.swiglu(), byte)?;
        self.arena
            .fill(stream, self.layout.down_activation_codes(), byte)?;
        self.arena
            .fill(stream, self.layout.down_activation_scales(), byte)?;
        self.arena.fill(stream, self.layout.branch(), byte)?;
        self.arena
            .fill(stream, self.layout.residual_output(), byte)?;
        self.arena
            .fill(stream, self.layout.next_normalized(), byte)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all working planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<DenseFp8MlpObservables> {
        Ok(DenseFp8MlpObservables {
            residual_input: self
                .arena
                .copy_to_host(stream, self.layout.residual_input())?,
            normalized: self.arena.copy_to_host(stream, self.layout.normalized())?,
            gate_up_activation_codes: self
                .arena
                .copy_to_host(stream, self.layout.gate_up_activation_codes())?,
            gate_up_activation_scales: self
                .arena
                .copy_to_host(stream, self.layout.gate_up_activation_scales())?,
            swiglu: self.arena.copy_to_host(stream, self.layout.swiglu())?,
            down_activation_codes: self
                .arena
                .copy_to_host(stream, self.layout.down_activation_codes())?,
            down_activation_scales: self
                .arena
                .copy_to_host(stream, self.layout.down_activation_scales())?,
            branch: self.arena.copy_to_host(stream, self.layout.branch())?,
            residual_output: self
                .arena
                .copy_to_host(stream, self.layout.residual_output())?,
            next_normalized: self
                .arena
                .copy_to_host(stream, self.layout.next_normalized())?,
        })
    }
}

#[cfg(feature = "qualification")]
/// Complete working-plane snapshot exposed to the qualification crate.
pub struct DenseFp8MlpObservables {
    /// Input residual rows.
    pub residual_input: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub normalized: Vec<u16>,
    /// Gate/up dynamic E4M3 codes.
    pub gate_up_activation_codes: Vec<u8>,
    /// Gate/up dynamic FP32 scales.
    pub gate_up_activation_scales: Vec<f32>,
    /// Fused BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Down dynamic E4M3 codes.
    pub down_activation_codes: Vec<u8>,
    /// Down dynamic FP32 scales.
    pub down_activation_scales: Vec<f32>,
    /// BF16 down-projection branch rows.
    pub branch: Vec<u16>,
    /// Published BF16 residual rows.
    pub residual_output: Vec<u16>,
    /// Next-boundary normalized BF16 rows.
    pub next_normalized: Vec<u16>,
}

fn capture_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    swiglu: &DenseFp8SwiGluOp<A>,
    down: &DenseFp8DownOp<A>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, norm, swiglu, down, pointers)
        })?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout("dense-FP8 MLP graph inventory has wrong cardinality"))
}

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    batch: usize,
    norm: &ResidualNormOp<A>,
    swiglu: &DenseFp8SwiGluOp<A>,
    down: &DenseFp8DownOp<A>,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: all pointers name aligned non-overlapping regions sized for
    // MAX_BATCH, and exact dispatch restricts each launch to `batch` rows.
    unsafe {
        norm.launch_plain(
            stream,
            batch,
            pointers.residual_input,
            pointers.input_norm,
            pointers.normalized,
        )?;
        swiglu.launch(
            stream,
            batch,
            pointers.normalized,
            pointers.gate_up_activation_codes,
            pointers.gate_up_activation_scales,
            pointers.gate_up_weight_codes,
            pointers.gate_up_weight_scales,
            pointers.swiglu,
        )?;
        down.launch(
            stream,
            batch,
            pointers.swiglu,
            pointers.down_activation_codes,
            pointers.down_activation_scales,
            pointers.down_weight_codes,
            pointers.down_weight_scales,
            pointers.branch,
        )?;
        norm.launch_residual(
            stream,
            batch,
            pointers.residual_input,
            pointers.branch,
            pointers.next_norm,
            pointers.residual_output,
            pointers.next_normalized,
        )
    }
}

fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(EngineError::layout(
            "dense-FP8 BF16 source plane has an odd byte length",
        ));
    }

    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "dense-FP8 MLP batch {batch} is outside the exact range 1..={MAX_BATCH}"
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
    use super::{little_endian_words, require_batch};
    use crate::EngineErrorCode;

    #[test]
    fn exact_batch_table_rejects_every_boundary_neighbor() {
        for batch in 1..=8 {
            require_batch(batch).unwrap();
        }
        for batch in [0, 9, 16, usize::MAX] {
            let error = require_batch(batch).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn bf16_source_bytes_are_decoded_without_value_conversion() {
        assert_eq!(
            little_endian_words(&[0x80, 0x3f, 0x00, 0xbf]).unwrap(),
            [0x3f80, 0xbf00]
        );
        assert!(little_endian_words(&[0x80]).is_err());
    }
}

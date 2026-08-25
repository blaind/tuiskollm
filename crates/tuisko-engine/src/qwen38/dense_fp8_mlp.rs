//! Resident source-backed dense-FP8 MLP program.

use crate::common::graph::{capture_batch_graphs, capture_route_graphs};
use crate::common::math::checked_product;
use crate::qwen38::dense_fp8_mlp_layout::MAX_ROWS;
use crate::{DenseFp8MlpLayout, EngineError, EngineResult, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp, DenseFp8SwiGluTmaMaps, ResidualNormOp,
    Sm120Arch,
};
use tuisko_model::{CheckpointSnapshot, DenseFp8MlpBindings, Qwen38_27B};

/// One late-layer dense-FP8 MLP with immutable exact decode and prefill graphs.
pub struct DenseFp8MlpProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 4],
    // Captured TMA launches retain these address-bound descriptor allocations.
    #[allow(dead_code)]
    gate_up_maps: DenseFp8SwiGluTmaMaps,
    #[allow(dead_code)]
    down_maps: DenseFp8DownTmaMaps,
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
    /// Loads one admitted layer, allocates one arena, and captures all exact routes.
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
        // SAFETY: the owner arena keeps all encoded source addresses stable.
        let gate_up_maps = unsafe {
            DenseFp8SwiGluTmaMaps::new(
                &stream,
                pointers.gate_up_activation_codes.cast_const(),
                pointers.gate_up_weight_codes,
            )?
        };
        // SAFETY: the owner arena keeps all encoded source addresses stable.
        let down_maps = unsafe {
            DenseFp8DownTmaMaps::new(
                &stream,
                pointers.down_activation_codes.cast_const(),
                pointers.down_weight_codes,
            )?
        };
        let base_address = arena.base_address();
        let graphs = capture_decode_routes(
            &stream,
            &norm,
            &swiglu,
            &down,
            &gate_up_maps,
            &down_maps,
            pointers,
        )?;
        let prefill_graphs = capture_prefill_routes(
            &stream,
            &norm,
            &swiglu,
            &down,
            &gate_up_maps,
            &down_maps,
            pointers,
        )?;

        Ok(Self {
            graphs,
            prefill_graphs,
            gate_up_maps,
            down_maps,
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

    /// Uploads one exact decode or prefill width into stable input storage.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_rows(rows)?;
        let expected = checked_product("dense-FP8 MLP input elements", rows, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "dense-FP8 MLP input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.residual_input(), values)?;

        Ok(())
    }

    /// Replays the immutable graph for one exact decode or prefill width.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        // SAFETY: this DenseFp8MlpProgram owns every captured allocation (arena,
        // TMA maps, op modules) for its whole life and drops the graphs first.
        unsafe { self.graph(rows)?.launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = checked_product("dense-FP8 MLP output elements", rows, A::HIDDEN)?;

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

    /// Resident weights plus working planes, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.layout.owner_bytes()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Largest admitted exact prefill width.
    pub const fn row_capacity(&self) -> usize {
        MAX_ROWS
    }

    /// Exact bytes in all four address-bound tensor-map descriptors.
    pub const fn descriptor_bytes(&self) -> usize {
        DenseFp8SwiGluTmaMaps::BYTE_LEN + DenseFp8DownTmaMaps::BYTE_LEN
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &DenseFp8MlpLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<A>> {
        &self.snapshot
    }

    fn graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        if (1..=MAX_BATCH).contains(&rows) {
            return Ok(&self.graphs[rows - 1]);
        }
        let index = prefill_index(rows).ok_or_else(|| {
            EngineError::route(format!(
                "dense-FP8 MLP row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
            ))
        })?;

        Ok(&self.prefill_graphs[index])
    }

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        require_rows(rows)?;
        let pointers = Pointers::bind(&self.arena, &self.layout)?;
        launch_route(
            stream,
            rows,
            &self._norm,
            &self._swiglu,
            &self._down,
            &self.gate_up_maps,
            &self.down_maps,
            pointers,
        )?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured production graph.
    pub fn qualification_graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        self.graph(rows)
    }

    #[cfg(feature = "qualification")]
    /// Captures repeated production paths for high-resolution device timing.
    pub fn qualification_repeated_graph(
        &self,
        stream: &CudaStream,
        rows: usize,
        operations: u64,
    ) -> EngineResult<CudaGraph> {
        require_rows(rows)?;
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
                    rows,
                    &self._norm,
                    &self._swiglu,
                    &self._down,
                    &self.gate_up_maps,
                    &self.down_maps,
                    pointers,
                )?;
            }

            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 20]> {
        let arena = Pointers::bind(&self.arena, &self.layout)?.addresses();
        let gate_up = self.gate_up_maps.device_addresses();
        let down = self.down_maps.device_addresses();

        Ok([
            arena[0], arena[1], arena[2], arena[3], arena[4], arena[5], arena[6], arena[7],
            arena[8], arena[9], arena[10], arena[11], arena[12], arena[13], arena[14], arena[15],
            gate_up[0], gate_up[1], down[0], down[1],
        ])
    }

    #[cfg(feature = "qualification")]
    /// Copies all four opaque tensor maps for immutable-owner qualification.
    pub fn qualification_descriptors(&self, stream: &CudaStream) -> EngineResult<[Vec<u64>; 4]> {
        let gate_up = self.gate_up_maps.copy_to_host(stream)?;
        let down = self.down_maps.copy_to_host(stream)?;

        Ok([
            gate_up[0].clone(),
            gate_up[1].clone(),
            down[0].clone(),
            down[1].clone(),
        ])
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

fn capture_decode_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    swiglu: &DenseFp8SwiGluOp<A>,
    down: &DenseFp8DownOp<A>,
    gate_up_maps: &DenseFp8SwiGluTmaMaps,
    down_maps: &DenseFp8DownTmaMaps,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    capture_batch_graphs(
        stream,
        "dense-FP8 MLP graph inventory has wrong cardinality",
        |batch| {
            launch_route(
                stream,
                batch,
                norm,
                swiglu,
                down,
                gate_up_maps,
                down_maps,
                pointers,
            )
        },
    )
}

fn capture_prefill_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    swiglu: &DenseFp8SwiGluOp<A>,
    down: &DenseFp8DownOp<A>,
    gate_up_maps: &DenseFp8SwiGluTmaMaps,
    down_maps: &DenseFp8DownTmaMaps,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; 4]> {
    capture_route_graphs(
        stream,
        [32, 64, 128, MAX_ROWS],
        "dense-FP8 MLP prefill graph inventory has wrong cardinality",
        |rows| {
            launch_route(
                stream,
                rows,
                norm,
                swiglu,
                down,
                gate_up_maps,
                down_maps,
                pointers,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    rows: usize,
    norm: &ResidualNormOp<A>,
    swiglu: &DenseFp8SwiGluOp<A>,
    down: &DenseFp8DownOp<A>,
    gate_up_maps: &DenseFp8SwiGluTmaMaps,
    down_maps: &DenseFp8DownTmaMaps,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: all pointers name aligned non-overlapping regions sized for
    // MAX_ROWS, and exact dispatch restricts each launch to `rows` rows.
    unsafe {
        norm.launch_plain(
            stream,
            rows,
            pointers.residual_input,
            pointers.input_norm,
            pointers.normalized,
        )?;
        if rows == MAX_ROWS {
            // T=1024 amortizes address-bound TMA setup across the complete macro tile.
            swiglu.launch_macro_prefill(
                stream,
                pointers.normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_up_weight_codes,
                pointers.gate_up_weight_scales,
                pointers.swiglu,
                gate_up_maps,
            )?;
            down.launch_macro_prefill(
                stream,
                pointers.swiglu,
                pointers.down_activation_codes,
                pointers.down_activation_scales,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                pointers.branch,
                down_maps,
            )?;
        } else {
            swiglu.launch(
                stream,
                rows,
                pointers.normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_up_weight_codes,
                pointers.gate_up_weight_scales,
                pointers.swiglu,
            )?;
            if rows <= MAX_BATCH {
                down.launch(
                    stream,
                    rows,
                    pointers.swiglu,
                    pointers.down_activation_codes,
                    pointers.down_activation_scales,
                    pointers.down_weight_codes,
                    pointers.down_weight_scales,
                    pointers.branch,
                )?;
            } else {
                down.launch_tail_prefill(
                    stream,
                    rows,
                    pointers.swiglu,
                    pointers.down_activation_codes,
                    pointers.down_activation_scales,
                    pointers.down_weight_codes,
                    pointers.down_weight_scales,
                    pointers.branch,
                )?;
            }
        }
        norm.launch_residual(
            stream,
            rows,
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

fn prefill_index(rows: usize) -> Option<usize> {
    match rows {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        MAX_ROWS => Some(3),
        _ => None,
    }
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&rows) && prefill_index(rows).is_none() {
        return Err(EngineError::route(format!(
            "dense-FP8 MLP row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{little_endian_words, require_rows};
    use crate::EngineErrorCode;

    #[test]
    fn exact_route_table_rejects_every_boundary_neighbor() {
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024] {
            require_rows(rows).unwrap();
        }
        for rows in [0, 9, 31, 33, 63, 65, 127, 129, 1_023, 1_025, usize::MAX] {
            let error = require_rows(rows).unwrap_err();
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

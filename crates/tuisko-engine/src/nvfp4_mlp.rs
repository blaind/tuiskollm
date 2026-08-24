//! Resident source-backed NVFP4 MLP program.

use crate::nvfp4_mlp_layout::MAX_ROWS;
use crate::{EngineError, EngineResult, MAX_BATCH, Nvfp4MlpLayout};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{Nvfp4DownOp, Nvfp4SwiGluOp, ResidualNormOp, Sm120Arch};
use tuisko_model::{CheckpointSnapshot, Nvfp4MlpBindings, Qwen38_27B};

/// One early-layer NVFP4 MLP with immutable exact decode and prefill graphs.
pub struct Nvfp4MlpProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 4],
    arena: DeviceArena,
    _norm: ResidualNormOp<A>,
    _swiglu: Nvfp4SwiGluOp,
    _down: Nvfp4DownOp<A>,
    snapshot: Arc<CheckpointSnapshot<A>>,
    context: Arc<CudaContext>,
    layout: Nvfp4MlpLayout,
    base_address: u64,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    gate_up_input_scale_divisor: f32,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    gate_up_weight_scale_divisor: f32,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    down_input_scale_divisor: f32,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    down_weight_scale_divisor: f32,
    layer: usize,
    arch: PhantomData<A>,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    normalized: *mut u16,
    gate_up_activation_codes: *mut u8,
    gate_up_activation_scales: *mut u8,
    gate_weight_codes: *const u8,
    up_weight_codes: *const u8,
    gate_up_weight_scales: *const u8,
    swiglu: *mut u16,
    down_activation_codes: *mut u8,
    down_activation_scales: *mut u8,
    down_weight_codes: *const u8,
    down_weight_scales: *const u8,
    branch: *mut u16,
    next_norm: *const u16,
    residual_output: *mut u16,
    next_normalized: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, layout: &Nvfp4MlpLayout) -> GpuResult<Self> {
        let pointers = Self {
            residual_input: arena.address(layout.residual_input())?.cast_const(),
            input_norm: arena.address(layout.input_norm())?.cast_const(),
            normalized: arena.address(layout.normalized())?,
            gate_up_activation_codes: arena.address(layout.gate_up_activation_codes())?,
            gate_up_activation_scales: arena.address(layout.gate_up_activation_scales())?,
            gate_weight_codes: arena.address(layout.gate_weight_codes())?.cast_const(),
            up_weight_codes: arena.address(layout.up_weight_codes())?.cast_const(),
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
        };
        if pointers.up_weight_codes.addr()
            != pointers.gate_weight_codes.addr() + layout.gate_weight_codes().byte_len()
        {
            return Err(GpuError::invalid_launch(
                "NVFP4 gate/up device code planes are not adjacent",
            ));
        }

        Ok(pointers)
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 17] {
        [
            self.residual_input.addr(),
            self.input_norm.addr(),
            self.normalized.addr(),
            self.gate_up_activation_codes.addr(),
            self.gate_up_activation_scales.addr(),
            self.gate_weight_codes.addr(),
            self.up_weight_codes.addr(),
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

#[derive(Clone, Copy)]
struct Divisors {
    gate_up_input: f32,
    gate_up_weight: f32,
    down_input: f32,
    down_weight: f32,
}

impl<A: Sm120Arch> Nvfp4MlpProgram<A> {
    /// Loads one admitted layer, allocates one arena, and captures all exact routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<A>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let bindings = Nvfp4MlpBindings::bind(snapshot.as_ref(), layer)?;
        let input_norm = bindings.input_norm.words().collect::<Vec<_>>();
        let next_norm = bindings.next_norm.words().collect::<Vec<_>>();
        let gate_up = bindings.gate_up.materialize()?;
        let down = bindings.down.materialize()?;
        let layout = Nvfp4MlpLayout::build_prefill::<A>()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let swiglu = Nvfp4SwiGluOp::new(context)?;
        let down_op = Nvfp4DownOp::new(context)?;

        arena.copy_from_host(&stream, layout.input_norm(), &input_norm)?;
        arena.copy_from_host(
            &stream,
            layout.gate_weight_codes(),
            gate_up.gate_weight_e2m1,
        )?;
        arena.copy_from_host(&stream, layout.up_weight_codes(), gate_up.up_weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            layout.gate_up_weight_scales(),
            &gate_up.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(&stream, layout.down_weight_codes(), down.weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            layout.down_weight_scales(),
            &down.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(&stream, layout.next_norm(), &next_norm)?;

        let pointers = Pointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let divisors = Divisors {
            gate_up_input: gate_up.input_scale_divisor,
            gate_up_weight: gate_up.weight_scale_divisor,
            down_input: down.input_scale_divisor,
            down_weight: down.weight_scale_divisor,
        };
        let graphs = capture_decode_routes(&stream, &norm, &swiglu, &down_op, pointers, divisors)?;
        let prefill_graphs =
            capture_prefill_routes(&stream, &norm, &swiglu, &down_op, pointers, divisors)?;

        Ok(Self {
            graphs,
            prefill_graphs,
            arena,
            _norm: norm,
            _swiglu: swiglu,
            _down: down_op,
            snapshot: snapshot.clone(),
            context: context.clone(),
            layout,
            base_address,
            gate_up_input_scale_divisor: gate_up.input_scale_divisor,
            gate_up_weight_scale_divisor: gate_up.weight_scale_divisor,
            down_input_scale_divisor: down.input_scale_divisor,
            down_weight_scale_divisor: down.weight_scale_divisor,
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
        let expected = checked_product("NVFP4 MLP input elements", rows, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "NVFP4 MLP input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.residual_input(), values)?;

        Ok(())
    }

    /// Replays the immutable graph for one exact decode or prefill width.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        self.graph(rows)?.launch(stream)?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = checked_product("NVFP4 MLP output elements", rows, A::HIDDEN)?;

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

    /// Largest admitted exact prefill width.
    pub const fn row_capacity(&self) -> usize {
        MAX_ROWS
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &Nvfp4MlpLayout {
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
                "NVFP4 MLP row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
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
            pointers,
            self.divisors(),
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
                "repeated NVFP4 MLP graph requires at least one operation",
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
                    pointers,
                    self.divisors(),
                )?;
            }

            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 17]> {
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
    ) -> EngineResult<Nvfp4MlpObservables> {
        Ok(Nvfp4MlpObservables {
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

    #[cfg(feature = "qualification")]
    /// Reads every immutable device plane in source/materialized order.
    pub fn qualification_immutable(&self, stream: &CudaStream) -> EngineResult<Nvfp4MlpImmutable> {
        Ok(Nvfp4MlpImmutable {
            input_norm: self.arena.copy_to_host(stream, self.layout.input_norm())?,
            gate_weight_codes: self
                .arena
                .copy_to_host(stream, self.layout.gate_weight_codes())?,
            up_weight_codes: self
                .arena
                .copy_to_host(stream, self.layout.up_weight_codes())?,
            gate_up_weight_scales: self
                .arena
                .copy_to_host(stream, self.layout.gate_up_weight_scales())?,
            down_weight_codes: self
                .arena
                .copy_to_host(stream, self.layout.down_weight_codes())?,
            down_weight_scales: self
                .arena
                .copy_to_host(stream, self.layout.down_weight_scales())?,
            next_norm: self.arena.copy_to_host(stream, self.layout.next_norm())?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Returns all admitted checkpoint divisors without conversion.
    pub const fn qualification_divisors(&self) -> [f32; 4] {
        [
            self.gate_up_input_scale_divisor,
            self.gate_up_weight_scale_divisor,
            self.down_input_scale_divisor,
            self.down_weight_scale_divisor,
        ]
    }

    #[cfg(feature = "qualification")]
    const fn divisors(&self) -> Divisors {
        Divisors {
            gate_up_input: self.gate_up_input_scale_divisor,
            gate_up_weight: self.gate_up_weight_scale_divisor,
            down_input: self.down_input_scale_divisor,
            down_weight: self.down_weight_scale_divisor,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete working-plane snapshot exposed to the qualification crate.
pub struct Nvfp4MlpObservables {
    /// Input residual rows.
    pub residual_input: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub normalized: Vec<u16>,
    /// Dynamic packed E2M1 activation codes for W4A4 routes.
    pub gate_up_activation_codes: Vec<u8>,
    /// Dynamic E4M3 block scales for W4A4 routes.
    pub gate_up_activation_scales: Vec<u8>,
    /// Fused BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Dynamic packed E2M1 down-input codes for prefill routes.
    pub down_activation_codes: Vec<u8>,
    /// Dynamic E4M3 down-input block scales for prefill routes.
    pub down_activation_scales: Vec<u8>,
    /// BF16 down-projection branch rows.
    pub branch: Vec<u16>,
    /// Published BF16 residual rows.
    pub residual_output: Vec<u16>,
    /// Next-boundary normalized BF16 rows.
    pub next_normalized: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Immutable device planes exposed for source-preservation qualification.
pub struct Nvfp4MlpImmutable {
    /// Pre-MLP norm weights.
    pub input_norm: Vec<u16>,
    /// Packed gate source codes.
    pub gate_weight_codes: Vec<u8>,
    /// Packed up source codes.
    pub up_weight_codes: Vec<u8>,
    /// Losslessly swizzled fused gate/up scales.
    pub gate_up_weight_scales: Vec<u8>,
    /// Packed down source codes.
    pub down_weight_codes: Vec<u8>,
    /// Losslessly swizzled down scales.
    pub down_weight_scales: Vec<u8>,
    /// Next-boundary norm weights.
    pub next_norm: Vec<u16>,
}

fn capture_decode_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    swiglu: &Nvfp4SwiGluOp,
    down: &Nvfp4DownOp<A>,
    pointers: Pointers,
    divisors: Divisors,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, norm, swiglu, down, pointers, divisors)
        })?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout("NVFP4 MLP graph inventory has wrong cardinality"))
}

fn capture_prefill_routes<A: Sm120Arch>(
    stream: &CudaStream,
    norm: &ResidualNormOp<A>,
    swiglu: &Nvfp4SwiGluOp,
    down: &Nvfp4DownOp<A>,
    pointers: Pointers,
    divisors: Divisors,
) -> EngineResult<[CudaGraph; 4]> {
    let mut graphs = Vec::with_capacity(4);
    for rows in [32, 64, 128, MAX_ROWS] {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, rows, norm, swiglu, down, pointers, divisors)
        })?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout("NVFP4 MLP prefill graph inventory has wrong cardinality"))
}

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    rows: usize,
    norm: &ResidualNormOp<A>,
    swiglu: &Nvfp4SwiGluOp,
    down: &Nvfp4DownOp<A>,
    pointers: Pointers,
    divisors: Divisors,
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
        swiglu.launch(
            stream,
            rows,
            pointers.normalized,
            pointers.gate_up_activation_codes,
            pointers.gate_up_activation_scales,
            pointers.gate_weight_codes,
            pointers.gate_up_weight_scales,
            divisors.gate_up_input,
            divisors.gate_up_weight,
            pointers.swiglu,
        )?;
        if rows <= MAX_BATCH {
            // Decode preserves its BF16 input, so only the represented weight
            // divisor participates in the retained A16 schedule.
            down.launch(
                stream,
                rows,
                pointers.swiglu,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                divisors.down_weight,
                pointers.branch,
            )?;
        } else {
            down.launch_prefill(
                stream,
                rows,
                pointers.swiglu,
                pointers.down_activation_codes,
                pointers.down_activation_scales,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                divisors.down_input,
                divisors.down_weight,
                pointers.branch,
            )?;
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

fn require_rows(rows: usize) -> EngineResult<()> {
    if (1..=MAX_BATCH).contains(&rows) || prefill_index(rows).is_some() {
        return Ok(());
    }

    Err(EngineError::route(format!(
        "NVFP4 MLP row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
    )))
}

const fn prefill_index(rows: usize) -> Option<usize> {
    match rows {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        MAX_ROWS => Some(3),
        _ => None,
    }
}

fn checked_product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::require_rows;
    use crate::EngineErrorCode;

    #[test]
    fn exact_route_table_rejects_every_boundary_neighbor() {
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024] {
            require_rows(rows).unwrap();
        }
        for rows in [0, 9, 16, 31, 33, 63, 65, 127, 129, 1_023, 1_025, usize::MAX] {
            let error = require_rows(rows).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }
}

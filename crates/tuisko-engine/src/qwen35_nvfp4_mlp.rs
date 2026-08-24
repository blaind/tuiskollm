//! Resident source-backed Qwen3.5 NVFP4 MLP program.

#[cfg(feature = "qualification")]
use crate::nvfp4_mlp::{Nvfp4MlpImmutable, Nvfp4MlpObservables};
use crate::{EngineError, EngineResult, MAX_BATCH, Nvfp4MlpLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{Qwen35Nvfp4DownOp, Qwen35Nvfp4SwiGluOp, Qwen35ResidualNormOp};
use tuisko_model::{Arch, CheckpointSnapshot, ModelOptNvfp4MlpBindings, Qwen35_9B};

/// One Qwen3.5 decoder MLP with immutable exact-batch graph routes.
pub struct Qwen35Nvfp4MlpProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: Qwen35ResidualNormOp,
    _swiglu: Qwen35Nvfp4SwiGluOp,
    _down: Qwen35Nvfp4DownOp,
    snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
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
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    source_scales: [f32; 4],
    layer: usize,
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
                "Qwen3.5 NVFP4 gate/up device code planes are not adjacent",
            ));
        }

        Ok(pointers)
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> [usize; 15] {
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
    down_weight: f32,
}

impl Qwen35Nvfp4MlpProgram {
    /// Loads one admitted layer, allocates one arena, and captures `B=1..=8`.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let materialized =
            ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let input_norm = materialized.input_norm.words().collect::<Vec<_>>();
        let next_norm = materialized.next_norm.words().collect::<Vec<_>>();
        let layout = Nvfp4MlpLayout::build::<Qwen35_9B>()?;
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen35ResidualNormOp::new(context)?;
        let swiglu = Qwen35Nvfp4SwiGluOp::new(context)?;
        let down = Qwen35Nvfp4DownOp::new(context)?;

        arena.copy_from_host(&stream, layout.input_norm(), &input_norm)?;
        arena.copy_from_host(
            &stream,
            layout.gate_weight_codes(),
            materialized.gate_up.gate_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            layout.up_weight_codes(),
            materialized.gate_up.up_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            layout.gate_up_weight_scales(),
            &materialized.gate_up.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            layout.down_weight_codes(),
            materialized.down.weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            layout.down_weight_scales(),
            &materialized.down.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(&stream, layout.next_norm(), &next_norm)?;

        let pointers = Pointers::bind(&arena, &layout)?;
        let base_address = arena.base_address();
        let divisors = Divisors {
            gate_up_input: materialized.gate_up.input_scale_divisor,
            gate_up_weight: materialized.gate_up.weight_scale_divisor,
            down_weight: materialized.down.weight_scale_divisor,
        };
        let graphs = capture_routes(&stream, &norm, &swiglu, &down, pointers, divisors)?;

        Ok(Self {
            graphs,
            arena,
            _norm: norm,
            _swiglu: swiglu,
            _down: down,
            snapshot: snapshot.clone(),
            context: context.clone(),
            layout,
            base_address,
            gate_up_input_scale_divisor: materialized.gate_up.input_scale_divisor,
            gate_up_weight_scale_divisor: materialized.gate_up.weight_scale_divisor,
            down_input_scale_divisor: materialized.down.input_scale_divisor,
            down_weight_scale_divisor: materialized.down.weight_scale_divisor,
            source_scales: [
                materialized.gate_up_input_scale,
                materialized.gate_up_weight_scale_2,
                materialized.down_input_scale,
                materialized.down_weight_scale_2,
            ],
            layer,
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
        let expected =
            checked_product("Qwen3.5 NVFP4 MLP input elements", batch, Qwen35_9B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.5 NVFP4 MLP input has {} values, expected {expected} for B={batch}",
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
        let values = checked_product(
            "Qwen3.5 NVFP4 MLP output elements",
            batch,
            Qwen35_9B::HIDDEN,
        )?;

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
    pub const fn layout(&self) -> &Nvfp4MlpLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen35_9B>> {
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
            self.divisors(),
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
                "repeated Qwen3.5 NVFP4 MLP graph requires at least one operation",
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
                    self.divisors(),
                )?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<[usize; 15]> {
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
        self.arena.fill(stream, self.layout.branch(), byte)?;
        self.arena
            .fill(stream, self.layout.residual_output(), byte)?;
        self.arena
            .fill(stream, self.layout.next_normalized(), byte)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every working plane, including inactive rows.
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
            down_activation_codes: Vec::new(),
            down_activation_scales: Vec::new(),
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
    /// Returns the runtime divisors after the ModelOpt convention change.
    pub const fn qualification_divisors(&self) -> [f32; 4] {
        [
            self.gate_up_input_scale_divisor,
            self.gate_up_weight_scale_divisor,
            self.down_input_scale_divisor,
            self.down_weight_scale_divisor,
        ]
    }

    #[cfg(feature = "qualification")]
    /// Returns the exact four ModelOpt F32 source scalars.
    pub const fn qualification_source_scales(&self) -> [f32; 4] {
        self.source_scales
    }

    #[cfg(feature = "qualification")]
    const fn divisors(&self) -> Divisors {
        Divisors {
            gate_up_input: self.gate_up_input_scale_divisor,
            gate_up_weight: self.gate_up_weight_scale_divisor,
            down_weight: self.down_weight_scale_divisor,
        }
    }
}

fn capture_routes(
    stream: &CudaStream,
    norm: &Qwen35ResidualNormOp,
    swiglu: &Qwen35Nvfp4SwiGluOp,
    down: &Qwen35Nvfp4DownOp,
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
        .map_err(|_| EngineError::layout("Qwen3.5 NVFP4 MLP graph inventory has wrong cardinality"))
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    norm: &Qwen35ResidualNormOp,
    swiglu: &Qwen35Nvfp4SwiGluOp,
    down: &Qwen35Nvfp4DownOp,
    pointers: Pointers,
    divisors: Divisors,
) -> GpuResult<()> {
    // SAFETY: every pointer names an aligned non-overlapping maximum-batch
    // region and exact dispatch limits all four launches to `batch` rows.
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
            pointers.gate_weight_codes,
            pointers.gate_up_weight_scales,
            divisors.gate_up_input,
            divisors.gate_up_weight,
            pointers.swiglu,
        )?;
        down.launch(
            stream,
            batch,
            pointers.swiglu,
            pointers.down_weight_codes,
            pointers.down_weight_scales,
            divisors.down_weight,
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

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "Qwen3.5 NVFP4 MLP batch {batch} is outside the exact range 1..={MAX_BATCH}"
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
    fn exact_batch_table_rejects_every_boundary_neighbor() {
        for batch in 1..=8 {
            require_batch(batch).unwrap();
        }
        for batch in [0, 9, 16, usize::MAX] {
            let error = require_batch(batch).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }
}

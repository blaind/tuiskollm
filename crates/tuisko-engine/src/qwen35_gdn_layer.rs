//! Resident source-backed Qwen3.5 GDN decoder layer.

use crate::qwen35_gdn_layer_layout::{QWEN35_GDN_MAX_ROWS, Qwen35GdnLayerRegions};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen35GdnLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen35GdnPrepareOp, Qwen35GdnRecurrenceOp, Qwen35Nvfp4DownOp, Qwen35Nvfp4GdnInputOp,
    Qwen35Nvfp4GdnOutputOp, Qwen35Nvfp4SwiGluOp, Qwen35ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, ModelOptNvfp4GdnBindings, ModelOptNvfp4MlpBindings, Qwen35_9B,
};

/// One Qwen3.5 GDN layer with immutable exact decode and prefill graphs.
pub struct Qwen35GdnLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 3],
    arena: DeviceArena,
    _norm: Qwen35ResidualNormOp,
    _input: Qwen35Nvfp4GdnInputOp,
    _prepare: Qwen35GdnPrepareOp,
    _recurrence: Qwen35GdnRecurrenceOp,
    _output: Qwen35Nvfp4GdnOutputOp,
    _swiglu: Qwen35Nvfp4SwiGluOp,
    _down: Qwen35Nvfp4DownOp,
    snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    context: Arc<CudaContext>,
    layout: Qwen35GdnLayerLayout,
    base_address: u64,
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    scale_divisors: [f32; 10],
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    source_scales: [f32; 10],
    layer: usize,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    mixer_normalized: *mut u16,
    input_activation_codes: *mut u8,
    input_activation_scales: *mut u8,
    input_weight_codes: *const u8,
    input_weight_scales: *const u8,
    control_weight_codes: *const u8,
    control_weight_scales: *const u8,
    projected: *mut u16,
    projected_controls: *mut u16,
    a_log: *const u16,
    dt_bias: *const u16,
    convolution_weights: *const u16,
    state_rows: *const u32,
    history: *mut u16,
    log_decay: *mut f32,
    beta: *mut f32,
    convolved: *mut u16,
    recurrent_norm: *const u16,
    state: *mut f32,
    recurrent_output: *mut u16,
    output_activation_codes: *mut u8,
    output_activation_scales: *mut u8,
    output_weight_codes: *const u8,
    output_weight_scales: *const u8,
    mixer_branch: *mut u16,
    post_attention_norm: *const u16,
    mixer_residual: *mut u16,
    mlp_normalized: *mut u16,
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
    mlp_branch: *mut u16,
    next_norm: *const u16,
    residual_output: *mut u16,
    next_normalized: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: Qwen35GdnLayerRegions) -> GpuResult<Self> {
        let pointers = Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            input_activation_codes: arena.address(regions.input_activation_codes)?,
            input_activation_scales: arena.address(regions.input_activation_scales)?,
            input_weight_codes: arena.address(regions.input_weight_codes)?.cast_const(),
            input_weight_scales: arena.address(regions.input_weight_scales)?.cast_const(),
            control_weight_codes: arena.address(regions.control_weight_codes)?.cast_const(),
            control_weight_scales: arena.address(regions.control_weight_scales)?.cast_const(),
            projected: arena.address(regions.projected)?,
            projected_controls: arena.address(regions.projected_controls)?,
            a_log: arena.address(regions.a_log)?.cast_const(),
            dt_bias: arena.address(regions.dt_bias)?.cast_const(),
            convolution_weights: arena.address(regions.convolution_weights)?.cast_const(),
            state_rows: arena.address(regions.state_rows)?.cast_const(),
            history: arena.address(regions.history)?,
            log_decay: arena.address(regions.log_decay)?,
            beta: arena.address(regions.beta)?,
            convolved: arena.address(regions.convolved)?,
            recurrent_norm: arena.address(regions.recurrent_norm)?.cast_const(),
            state: arena.address(regions.state)?,
            recurrent_output: arena.address(regions.recurrent_output)?,
            output_activation_codes: arena.address(regions.output_activation_codes)?,
            output_activation_scales: arena.address(regions.output_activation_scales)?,
            output_weight_codes: arena.address(regions.output_weight_codes)?.cast_const(),
            output_weight_scales: arena.address(regions.output_weight_scales)?.cast_const(),
            mixer_branch: arena.address(regions.mixer_branch)?,
            post_attention_norm: arena.address(regions.post_attention_norm)?.cast_const(),
            mixer_residual: arena.address(regions.mixer_residual)?,
            mlp_normalized: arena.address(regions.mlp_normalized)?,
            gate_up_activation_codes: arena.address(regions.gate_up_activation_codes)?,
            gate_up_activation_scales: arena.address(regions.gate_up_activation_scales)?,
            gate_weight_codes: arena.address(regions.gate_weight_codes)?.cast_const(),
            up_weight_codes: arena.address(regions.up_weight_codes)?.cast_const(),
            gate_up_weight_scales: arena.address(regions.gate_up_weight_scales)?.cast_const(),
            swiglu: arena.address(regions.swiglu)?,
            down_activation_codes: arena.address(regions.down_activation_codes)?,
            down_activation_scales: arena.address(regions.down_activation_scales)?,
            down_weight_codes: arena.address(regions.down_weight_codes)?.cast_const(),
            down_weight_scales: arena.address(regions.down_weight_scales)?.cast_const(),
            mlp_branch: arena.address(regions.mlp_branch)?,
            next_norm: arena.address(regions.next_norm)?.cast_const(),
            residual_output: arena.address(regions.residual_output)?,
            next_normalized: arena.address(regions.next_normalized)?,
        };
        if pointers.up_weight_codes.addr()
            != pointers.gate_weight_codes.addr() + regions.gate_weight_codes.byte_len()
        {
            return Err(GpuError::invalid_launch(
                "Qwen3.5 GDN gate/up device code planes are not adjacent",
            ));
        }

        Ok(pointers)
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> Vec<usize> {
        vec![
            self.residual_input.addr(),
            self.input_norm.addr(),
            self.mixer_normalized.addr(),
            self.input_activation_codes.addr(),
            self.input_activation_scales.addr(),
            self.input_weight_codes.addr(),
            self.input_weight_scales.addr(),
            self.control_weight_codes.addr(),
            self.control_weight_scales.addr(),
            self.projected.addr(),
            self.projected_controls.addr(),
            self.a_log.addr(),
            self.dt_bias.addr(),
            self.convolution_weights.addr(),
            self.state_rows.addr(),
            self.history.addr(),
            self.log_decay.addr(),
            self.beta.addr(),
            self.convolved.addr(),
            self.recurrent_norm.addr(),
            self.state.addr(),
            self.recurrent_output.addr(),
            self.output_activation_codes.addr(),
            self.output_activation_scales.addr(),
            self.output_weight_codes.addr(),
            self.output_weight_scales.addr(),
            self.mixer_branch.addr(),
            self.post_attention_norm.addr(),
            self.mixer_residual.addr(),
            self.mlp_normalized.addr(),
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
            self.mlp_branch.addr(),
            self.next_norm.addr(),
            self.residual_output.addr(),
            self.next_normalized.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Divisors {
    input_activation: f32,
    input_weight: f32,
    control_weight: f32,
    output_input: f32,
    output_weight: f32,
    gate_up_input: f32,
    gate_up_weight: f32,
    down_input: f32,
    down_weight: f32,
}

impl Qwen35GdnLayerProgram {
    /// Loads one source layer and captures exact `B=1..8` and `T=32,64,128` routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let gdn = ModelOptNvfp4GdnBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let mlp = ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let layout = Qwen35GdnLayerLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen35ResidualNormOp::new(context)?;
        let input = Qwen35Nvfp4GdnInputOp::new(context)?;
        let prepare = Qwen35GdnPrepareOp::new(context)?;
        let recurrence = Qwen35GdnRecurrenceOp::new(context)?;
        let output = Qwen35Nvfp4GdnOutputOp::new(context)?;
        let swiglu = Qwen35Nvfp4SwiGluOp::new(context)?;
        let down = Qwen35Nvfp4DownOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &gdn.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.input_weight_codes, &gdn.input_weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            regions.input_weight_scales,
            &gdn.input_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_codes,
            &gdn.control_weight_e2m1_padded,
        )?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_scales,
            &gdn.control_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.a_log,
            &gdn.a_log.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.dt_bias,
            &gdn.dt_bias.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.convolution_weights,
            &gdn.convolution_weight.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.recurrent_norm,
            &gdn.norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.output_weight_codes, gdn.output.weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_scales,
            &gdn.output.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.post_attention_norm,
            &gdn.post_attention_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.gate_weight_codes,
            mlp.gate_up.gate_weight_e2m1,
        )?;
        arena.copy_from_host(&stream, regions.up_weight_codes, mlp.gate_up.up_weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            regions.gate_up_weight_scales,
            &mlp.gate_up.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(&stream, regions.down_weight_codes, mlp.down.weight_e2m1)?;
        arena.copy_from_host(
            &stream,
            regions.down_weight_scales,
            &mlp.down.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.next_norm,
            &mlp.next_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.state_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let scale_divisors = [
            gdn.input_scale_divisor,
            gdn.input_weight_scale_divisor,
            gdn.control_input_scale_divisor,
            gdn.control_weight_scale_divisor,
            gdn.output.input_scale_divisor,
            gdn.output.weight_scale_divisor,
            mlp.gate_up.input_scale_divisor,
            mlp.gate_up.weight_scale_divisor,
            mlp.down.input_scale_divisor,
            mlp.down.weight_scale_divisor,
        ];
        let source_scales = [
            gdn.input_scale,
            gdn.input_weight_scale_2,
            gdn.control_input_scale,
            gdn.control_weight_scale_2,
            gdn.output.input_scale,
            gdn.output.weight_scale_2,
            mlp.gate_up_input_scale,
            mlp.gate_up_weight_scale_2,
            mlp.down_input_scale,
            mlp.down_weight_scale_2,
        ];
        let pointers = Pointers::bind(&arena, regions)?;
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            input: &input,
            prepare: &prepare,
            recurrence: &recurrence,
            output: &output,
            swiglu: &swiglu,
            down: &down,
        };
        let divisors = launch_divisors(scale_divisors);
        let graphs = capture_decode_routes(&stream, ops, pointers, divisors)?;
        let prefill_graphs = capture_prefill_routes(&stream, ops, pointers, divisors)?;

        Ok(Self {
            graphs,
            prefill_graphs,
            arena,
            _norm: norm,
            _input: input,
            _prepare: prepare,
            _recurrence: recurrence,
            _output: output,
            _swiglu: swiglu,
            _down: down,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            scale_divisors,
            source_scales,
            layer,
        })
    }

    /// Uploads one exact route's BF16 residual rows into stable input storage.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_rows(rows)?;
        let expected = product("Qwen3.5 GDN input elements", rows, Qwen35_9B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.5 GDN input has {} values, expected {expected} for {}",
                values.len(),
                route_name(rows),
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;

        Ok(())
    }

    /// Clears all slot-owned causal history and recurrent state.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.history, 0)?;
        self.arena.fill(stream, regions.state, 0)?;

        Ok(())
    }

    /// Maps compact rows to distinct physical recurrent-state slots.
    pub fn load_slot_routes(&self, stream: &CudaStream, slots: &[usize]) -> EngineResult<()> {
        let rows = slot_rows(slots)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.regions().state_rows,
            &rows[..slots.len()],
        )?;

        Ok(())
    }

    /// Selects one physical recurrent-state slot for a causal prompt route.
    pub fn load_prefill_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.regions().state_rows,
            &[slot as u32],
        )?;

        Ok(())
    }

    /// Selects one physical state row for an exact causal verification route.
    pub(crate) fn load_verify_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        self.load_prefill_slot(stream, slot)
    }

    /// Clears one physical slot's causal history and recurrent state.
    pub fn reset_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        let regions = self.layout.regions();
        fill_slot(&self.arena, stream, regions.history, slot)?;
        fill_slot(&self.arena, stream, regions.state, slot)?;

        Ok(())
    }

    /// Replays the immutable graph for one exact row route.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        // SAFETY: this Qwen35GdnLayerProgram owns every captured allocation
        // (arena, op modules) for its whole life and drops the graphs first.
        unsafe { self.graph(rows)?.launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("Qwen3.5 GDN output elements", rows, Qwen35_9B::HIDDEN)?;

        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().residual_output, values)?)
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

    /// Exact address-stable workspace and recurrent-state bytes.
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

    /// Largest exact row route backed by the workspace.
    pub const fn row_capacity(&self) -> usize {
        QWEN35_GDN_MAX_ROWS
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &Qwen35GdnLayerLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen35_9B>> {
        &self.snapshot
    }

    pub(crate) fn input_address(&self) -> GpuResult<*const u16> {
        Ok(Pointers::bind(&self.arena, self.layout.regions())?.residual_input)
    }

    pub(crate) fn output_address(&self) -> GpuResult<*const u16> {
        Ok(Pointers::bind(&self.arena, self.layout.regions())?
            .residual_output
            .cast_const())
    }

    fn graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        if (1..=MAX_BATCH).contains(&rows) {
            return Ok(&self.graphs[rows - 1]);
        }
        let index = prefill_index(rows).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.5 GDN row count {rows} is outside 1..={MAX_BATCH},32,64,128"
            ))
        })?;

        Ok(&self.prefill_graphs[index])
    }

    /// Launches this layer from another resident owner's BF16 residual plane.
    ///
    /// # Safety
    /// `input` covers `rows * 4,096` BF16 values in this CUDA context.
    pub(crate) unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        let mut pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        pointers.residual_input = input;
        require_rows(rows).map_err(|error| GpuError::invalid_launch(error.to_string()))?;
        launch_route(
            stream,
            rows,
            self.ops(),
            pointers,
            launch_divisors(self.scale_divisors),
        )?;

        Ok(pointers.residual_output.cast_const())
    }

    /// Launches one causal target-verification route from another retained residual plane.
    ///
    /// # Safety
    /// `input` covers `rows * 4,096` BF16 values in this CUDA context.
    pub(crate) unsafe fn launch_verify_from(
        &self,
        stream: &CudaStream,
        rows: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        let mut pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        pointers.residual_input = input;
        launch_verify_route(
            stream,
            rows,
            self.ops(),
            pointers,
            launch_divisors(self.scale_divisors),
        )?;

        Ok(pointers.residual_output.cast_const())
    }

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        require_rows(rows)?;
        launch_route(
            stream,
            rows,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
            launch_divisors(self.scale_divisors),
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
                "repeated Qwen3.5 GDN graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        let divisors = launch_divisors(self.scale_divisors);

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, rows, ops, pointers, divisors)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        Ok(Pointers::bind(&self.arena, self.layout.regions())?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Returns source-to-kernel scale divisors in mixer then MLP order.
    pub const fn qualification_divisors(&self) -> [f32; 10] {
        self.scale_divisors
    }

    #[cfg(feature = "qualification")]
    /// Returns the corresponding exact ModelOpt F32 source scales.
    pub const fn qualification_source_scales(&self) -> [f32; 10] {
        self.source_scales
    }

    #[cfg(feature = "qualification")]
    /// Fills every non-state mutable seam with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.mixer_normalized,
            regions.projected,
            regions.projected_controls,
            regions.convolved,
            regions.recurrent_output,
            regions.mixer_branch,
            regions.mixer_residual,
            regions.mlp_normalized,
            regions.swiglu,
            regions.mlp_branch,
            regions.residual_output,
            regions.next_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.input_activation_codes,
            regions.input_activation_scales,
            regions.output_activation_codes,
            regions.output_activation_scales,
            regions.gate_up_activation_codes,
            regions.gate_up_activation_scales,
            regions.down_activation_codes,
            regions.down_activation_scales,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [regions.log_decay, regions.beta] {
            self.arena.fill(stream, region, byte)?;
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all working and persistent planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen35GdnLayerObservables> {
        let regions = self.layout.regions();

        Ok(Qwen35GdnLayerObservables {
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            input_activation_codes: self
                .arena
                .copy_to_host(stream, regions.input_activation_codes)?,
            input_activation_scales: self
                .arena
                .copy_to_host(stream, regions.input_activation_scales)?,
            projected: self.arena.copy_to_host(stream, regions.projected)?,
            projected_controls: self
                .arena
                .copy_to_host(stream, regions.projected_controls)?,
            log_decay: self.arena.copy_to_host(stream, regions.log_decay)?,
            beta: self.arena.copy_to_host(stream, regions.beta)?,
            convolved: self.arena.copy_to_host(stream, regions.convolved)?,
            state_rows: self.arena.copy_to_host(stream, regions.state_rows)?,
            history: self.arena.copy_to_host(stream, regions.history)?,
            state: self.arena.copy_to_host(stream, regions.state)?,
            recurrent_output: self.arena.copy_to_host(stream, regions.recurrent_output)?,
            output_activation_codes: self
                .arena
                .copy_to_host(stream, regions.output_activation_codes)?,
            output_activation_scales: self
                .arena
                .copy_to_host(stream, regions.output_activation_scales)?,
            mixer_branch: self.arena.copy_to_host(stream, regions.mixer_branch)?,
            mixer_residual: self.arena.copy_to_host(stream, regions.mixer_residual)?,
            mlp_normalized: self.arena.copy_to_host(stream, regions.mlp_normalized)?,
            gate_up_activation_codes: self
                .arena
                .copy_to_host(stream, regions.gate_up_activation_codes)?,
            gate_up_activation_scales: self
                .arena
                .copy_to_host(stream, regions.gate_up_activation_scales)?,
            swiglu: self.arena.copy_to_host(stream, regions.swiglu)?,
            down_activation_codes: self
                .arena
                .copy_to_host(stream, regions.down_activation_codes)?,
            down_activation_scales: self
                .arena
                .copy_to_host(stream, regions.down_activation_scales)?,
            mlp_branch: self.arena.copy_to_host(stream, regions.mlp_branch)?,
            residual_output: self.arena.copy_to_host(stream, regions.residual_output)?,
            next_normalized: self.arena.copy_to_host(stream, regions.next_normalized)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads every immutable device plane in source/materialized order.
    pub fn qualification_immutable(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen35GdnLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen35GdnLayerImmutable {
            input_norm: self.arena.copy_to_host(stream, regions.input_norm)?,
            input_weight_codes: self
                .arena
                .copy_to_host(stream, regions.input_weight_codes)?,
            input_weight_scales: self
                .arena
                .copy_to_host(stream, regions.input_weight_scales)?,
            control_weight_codes: self
                .arena
                .copy_to_host(stream, regions.control_weight_codes)?,
            control_weight_scales: self
                .arena
                .copy_to_host(stream, regions.control_weight_scales)?,
            a_log: self.arena.copy_to_host(stream, regions.a_log)?,
            dt_bias: self.arena.copy_to_host(stream, regions.dt_bias)?,
            convolution_weights: self
                .arena
                .copy_to_host(stream, regions.convolution_weights)?,
            recurrent_norm: self.arena.copy_to_host(stream, regions.recurrent_norm)?,
            output_weight_codes: self
                .arena
                .copy_to_host(stream, regions.output_weight_codes)?,
            output_weight_scales: self
                .arena
                .copy_to_host(stream, regions.output_weight_scales)?,
            post_attention_norm: self
                .arena
                .copy_to_host(stream, regions.post_attention_norm)?,
            gate_weight_codes: self.arena.copy_to_host(stream, regions.gate_weight_codes)?,
            up_weight_codes: self.arena.copy_to_host(stream, regions.up_weight_codes)?,
            gate_up_weight_scales: self
                .arena
                .copy_to_host(stream, regions.gate_up_weight_scales)?,
            down_weight_codes: self.arena.copy_to_host(stream, regions.down_weight_codes)?,
            down_weight_scales: self
                .arena
                .copy_to_host(stream, regions.down_weight_scales)?,
            next_norm: self.arena.copy_to_host(stream, regions.next_norm)?,
        })
    }

    fn ops(&self) -> Ops<'_> {
        Ops {
            norm: &self._norm,
            input: &self._input,
            prepare: &self._prepare,
            recurrence: &self._recurrence,
            output: &self._output,
            swiglu: &self._swiglu,
            down: &self._down,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen35GdnLayerObservables {
    /// Pre-mixer normalized rows.
    pub mixer_normalized: Vec<u16>,
    /// Packed E2M1 mixer-input activation codes.
    pub input_activation_codes: Vec<u8>,
    /// E4M3 mixer-input activation scales.
    pub input_activation_scales: Vec<u8>,
    /// Fused Q/K/V/Z projection rows.
    pub projected: Vec<u16>,
    /// Padded A/B control projection rows.
    pub projected_controls: Vec<u16>,
    /// Per-value-head log decays.
    pub log_decay: Vec<f32>,
    /// Per-value-head update gates.
    pub beta: Vec<f32>,
    /// Causal-convolved Q/K/V rows.
    pub convolved: Vec<u16>,
    /// Physical recurrent-state row selected by each compact token row.
    pub state_rows: Vec<u32>,
    /// Slot-owned causal history.
    pub history: Vec<u16>,
    /// Slot-owned FP32 recurrent state.
    pub state: Vec<f32>,
    /// Gated normalized recurrent values.
    pub recurrent_output: Vec<u16>,
    /// Packed E2M1 recurrent-output activation codes.
    pub output_activation_codes: Vec<u8>,
    /// E4M3 recurrent-output activation scales.
    pub output_activation_scales: Vec<u8>,
    /// GDN output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after the mixer.
    pub mixer_residual: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub mlp_normalized: Vec<u16>,
    /// Packed E2M1 gate/up activation codes.
    pub gate_up_activation_codes: Vec<u8>,
    /// E4M3 gate/up activation scales.
    pub gate_up_activation_scales: Vec<u8>,
    /// Fused BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Packed E2M1 down-projection activation codes.
    pub down_activation_codes: Vec<u8>,
    /// E4M3 down-projection activation scales.
    pub down_activation_scales: Vec<u8>,
    /// NVFP4 down-projection branch.
    pub mlp_branch: Vec<u16>,
    /// Published layer residual rows.
    pub residual_output: Vec<u16>,
    /// Next-boundary normalized rows.
    pub next_normalized: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Immutable source-backed planes exposed to the qualification crate.
pub struct Qwen35GdnLayerImmutable {
    /// Input RMSNorm weights.
    pub input_norm: Vec<u16>,
    /// Fused packed Q/K/V/Z weight codes.
    pub input_weight_codes: Vec<u8>,
    /// Fused swizzled Q/K/V/Z block scales.
    pub input_weight_scales: Vec<u8>,
    /// Padded packed A/B control weight codes.
    pub control_weight_codes: Vec<u8>,
    /// Padded swizzled A/B control block scales.
    pub control_weight_scales: Vec<u8>,
    /// Log-space recurrence decay parameters.
    pub a_log: Vec<u16>,
    /// Recurrence time-step bias.
    pub dt_bias: Vec<u16>,
    /// Width-four causal-convolution weights.
    pub convolution_weights: Vec<u16>,
    /// Per-head recurrent RMSNorm weights.
    pub recurrent_norm: Vec<u16>,
    /// Packed recurrent-output projection codes.
    pub output_weight_codes: Vec<u8>,
    /// Swizzled recurrent-output projection scales.
    pub output_weight_scales: Vec<u8>,
    /// Post-mixer RMSNorm weights.
    pub post_attention_norm: Vec<u16>,
    /// Packed MLP gate weight codes.
    pub gate_weight_codes: Vec<u8>,
    /// Packed MLP up weight codes.
    pub up_weight_codes: Vec<u8>,
    /// Fused swizzled gate/up block scales.
    pub gate_up_weight_scales: Vec<u8>,
    /// Packed MLP down weight codes.
    pub down_weight_codes: Vec<u8>,
    /// Swizzled MLP down block scales.
    pub down_weight_scales: Vec<u8>,
    /// Next-boundary RMSNorm weights.
    pub next_norm: Vec<u16>,
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    norm: &'a Qwen35ResidualNormOp,
    input: &'a Qwen35Nvfp4GdnInputOp,
    prepare: &'a Qwen35GdnPrepareOp,
    recurrence: &'a Qwen35GdnRecurrenceOp,
    output: &'a Qwen35Nvfp4GdnOutputOp,
    swiglu: &'a Qwen35Nvfp4SwiGluOp,
    down: &'a Qwen35Nvfp4DownOp,
}

fn launch_divisors(values: [f32; 10]) -> Divisors {
    Divisors {
        input_activation: values[0],
        input_weight: values[1],
        control_weight: values[3],
        output_input: values[4],
        output_weight: values[5],
        gate_up_input: values[6],
        gate_up_weight: values[7],
        down_input: values[8],
        down_weight: values[9],
    }
}

fn capture_decode_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, ops, pointers, divisors)
        })?);
    }

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 GDN decode graph inventory has wrong cardinality")
    })
}

fn capture_prefill_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> EngineResult<[CudaGraph; 3]> {
    // T=128 would otherwise require sixteen B=8 layer graphs and 15 extra
    // boundaries. One graph composes the same eight qualified T=128 leaves;
    // every leaf retains its accumulation and rounding order.
    let mut graphs = Vec::with_capacity(3);
    for rows in [32, 64, 128] {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, rows, ops, pointers, divisors)
        })?);
    }

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 GDN prefill graph inventory has wrong cardinality")
    })
}

fn launch_route(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> GpuResult<()> {
    launch_route_kind(
        stream,
        rows,
        ops,
        pointers,
        divisors,
        if rows <= MAX_BATCH {
            RouteKind::Decode
        } else {
            RouteKind::Prefill
        },
    )
}

fn launch_verify_route(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> GpuResult<()> {
    if !(1..=4).contains(&rows) {
        return Err(GpuError::invalid_launch(format!(
            "Qwen3.5 GDN verification row count {rows} is outside 1..=4"
        )));
    }
    launch_route_kind(stream, rows, ops, pointers, divisors, RouteKind::Verify)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RouteKind {
    Decode,
    Verify,
    Prefill,
}

fn launch_route_kind(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
    kind: RouteKind,
) -> GpuResult<()> {
    // SAFETY: one arena owns aligned, disjoint 128-row working planes and
    // eight persistent state slots. Prompt leaves advance state row zero
    // causally; every leaf selects the same exact row count.
    unsafe {
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        if kind != RouteKind::Prefill {
            ops.input.launch(
                stream,
                rows,
                pointers.mixer_normalized,
                pointers.input_weight_codes,
                pointers.input_weight_scales,
                divisors.input_weight,
                pointers.control_weight_codes,
                pointers.control_weight_scales,
                divisors.control_weight,
                pointers.projected,
                pointers.projected_controls,
            )?;
        } else {
            ops.input.launch_prefill(
                stream,
                rows,
                pointers.mixer_normalized,
                pointers.input_activation_codes,
                pointers.input_activation_scales,
                pointers.input_weight_codes,
                pointers.input_weight_scales,
                divisors.input_weight,
                pointers.control_weight_codes,
                pointers.control_weight_scales,
                divisors.control_weight,
                divisors.input_activation,
                pointers.projected,
                pointers.projected_controls,
            )?;
        }
        if kind == RouteKind::Verify {
            ops.prepare.launch_causal(
                stream,
                rows,
                pointers.projected_controls,
                pointers.a_log,
                pointers.dt_bias,
                pointers.projected,
                pointers.convolution_weights,
                pointers.state_rows,
                pointers.history,
                pointers.log_decay,
                pointers.beta,
                pointers.convolved,
            )?;
            ops.recurrence.launch_causal(
                stream,
                rows,
                pointers.convolved,
                pointers.projected,
                pointers.log_decay,
                pointers.beta,
                pointers.recurrent_norm,
                pointers.state_rows,
                pointers.state,
                pointers.recurrent_output,
            )?;
        } else {
            ops.prepare.launch(
                stream,
                rows,
                pointers.projected_controls,
                pointers.a_log,
                pointers.dt_bias,
                pointers.projected,
                pointers.convolution_weights,
                pointers.state_rows,
                pointers.history,
                pointers.log_decay,
                pointers.beta,
                pointers.convolved,
            )?;
            ops.recurrence.launch(
                stream,
                rows,
                pointers.convolved,
                pointers.projected,
                pointers.log_decay,
                pointers.beta,
                pointers.recurrent_norm,
                pointers.state_rows,
                pointers.state,
                pointers.recurrent_output,
            )?;
        }
        if kind != RouteKind::Prefill {
            ops.output.launch(
                stream,
                rows,
                pointers.recurrent_output,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
                divisors.output_weight,
                pointers.mixer_branch,
            )?;
        } else {
            ops.output.launch_prefill(
                stream,
                rows,
                pointers.recurrent_output,
                pointers.output_activation_codes,
                pointers.output_activation_scales,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
                divisors.output_input,
                divisors.output_weight,
                pointers.mixer_branch,
            )?;
        }
        ops.norm.launch_residual(
            stream,
            rows,
            pointers.residual_input,
            pointers.mixer_branch,
            pointers.post_attention_norm,
            pointers.mixer_residual,
            pointers.mlp_normalized,
        )?;
        if kind != RouteKind::Prefill {
            ops.swiglu.launch(
                stream,
                rows,
                pointers.mlp_normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_weight_codes,
                pointers.gate_up_weight_scales,
                divisors.gate_up_input,
                divisors.gate_up_weight,
                pointers.swiglu,
            )?;
            ops.down.launch(
                stream,
                rows,
                pointers.swiglu,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                divisors.down_weight,
                pointers.mlp_branch,
            )?;
        } else {
            ops.swiglu.launch_prefill(
                stream,
                rows,
                pointers.mlp_normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_weight_codes,
                pointers.gate_up_weight_scales,
                divisors.gate_up_input,
                divisors.gate_up_weight,
                pointers.swiglu,
            )?;
            ops.down.launch_prefill(
                stream,
                rows,
                pointers.swiglu,
                pointers.down_activation_codes,
                pointers.down_activation_scales,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                divisors.down_input,
                divisors.down_weight,
                pointers.mlp_branch,
            )?;
        }
        ops.norm.launch_residual(
            stream,
            rows,
            pointers.mixer_residual,
            pointers.mlp_branch,
            pointers.next_norm,
            pointers.residual_output,
            pointers.next_normalized,
        )?;
    }

    Ok(())
}

fn prefill_index(rows: usize) -> Option<usize> {
    match rows {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        _ => None,
    }
}

fn slot_rows(slots: &[usize]) -> EngineResult<[u32; MAX_BATCH]> {
    require_batch(slots.len())?;
    let mut seen = [false; MAX_BATCH];
    let mut rows = [0u32; MAX_BATCH];
    for (row, &slot) in rows.iter_mut().zip(slots) {
        require_slot(slot)?;
        if std::mem::replace(&mut seen[slot], true) {
            return Err(EngineError::route(format!(
                "Qwen3.5 physical slot {slot} appears more than once"
            )));
        }
        *row = slot as u32;
    }
    Ok(rows)
}

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "Qwen3.5 compact batch {batch} is outside 1..={MAX_BATCH}"
        )));
    }
    Ok(())
}

fn require_slot(slot: usize) -> EngineResult<()> {
    if slot >= MAX_BATCH {
        return Err(EngineError::route(format!(
            "Qwen3.5 physical slot {slot} is outside 0..{MAX_BATCH}"
        )));
    }
    Ok(())
}

fn fill_slot<T: tuisko_gpu::DeviceCopy>(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: tuisko_gpu::ArenaRegion<T>,
    slot: usize,
) -> EngineResult<()> {
    if !region.len().is_multiple_of(MAX_BATCH) {
        return Err(EngineError::layout(format!(
            "Qwen3.5 persistent region of {} values is not divisible by {MAX_BATCH} slots",
            region.len()
        )));
    }
    let width = region.len() / MAX_BATCH;
    let start = product("Qwen3.5 persistent slot offset", slot, width)?;
    arena.fill_slice(stream, region, start, width, 0)?;
    Ok(())
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if (1..=MAX_BATCH).contains(&rows) || prefill_index(rows).is_some() {
        return Ok(());
    }

    Err(EngineError::route(format!(
        "Qwen3.5 GDN row count {rows} is outside 1..={MAX_BATCH},32,64,128"
    )))
}

fn route_name(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, prefill_index, require_rows, slot_rows};
    use crate::EngineErrorCode;

    #[test]
    fn exact_batch_table_rejects_every_boundary_neighbor() {
        for batch in 1..=8 {
            require_rows(batch).unwrap();
        }
        for rows in [32, 64, 128] {
            assert!(prefill_index(rows).is_some());
            require_rows(rows).unwrap();
        }
        for rows in [0, 9, 16, 31, 33, 63, 65, 127, 129, usize::MAX] {
            let error = require_rows(rows).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn compact_state_slot_table_is_bijective() {
        assert_eq!(slot_rows(&[4, 0, 7]).unwrap()[..3], [4, 0, 7]);
        for slots in [&[][..], &[3, 3], &[MAX_BATCH], &[0, 1, 2, 3, 4, 5, 6, 7, 0]] {
            assert!(slot_rows(slots).is_err(), "slots={slots:?}");
        }
    }
}

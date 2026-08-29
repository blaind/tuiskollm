//! Resident source-backed dense-FP8 GDN decoder layer.

use crate::common::graph::{capture_batch_graphs, capture_route_graphs};
use crate::common::math::{little_endian_words, product};
use crate::qwen38::dense_fp8_gdn_layer_layout::{GdnLayerRegions, MAX_ROWS};
use crate::{DenseFp8GdnLayerLayout, EngineError, EngineResult, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8GdnInputTmaMaps, DenseFp8GdnOutputTmaMaps,
    DenseFp8SwiGluOp, DenseFp8SwiGluTmaMaps, GdnInputProjectionOp, GdnOutputProjectionOp,
    GdnPrepareOp, GdnRecurrenceOp, ResidualNormOp, Sm120Arch,
};
use tuisko_model::{CheckpointSnapshot, DenseFp8MlpBindings, GdnBindings, Qwen38_27B};

/// One late dense-FP8 GDN layer with immutable exact decode and prefill graphs.
pub struct DenseFp8GdnLayerProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 4],
    // Captured TMA launches retain these address-bound descriptor allocations.
    #[allow(dead_code)]
    gate_up_maps: DenseFp8SwiGluTmaMaps,
    #[allow(dead_code)]
    down_maps: DenseFp8DownTmaMaps,
    #[allow(dead_code)]
    input_maps: DenseFp8GdnInputTmaMaps,
    #[allow(dead_code)]
    output_maps: DenseFp8GdnOutputTmaMaps,
    arena: DeviceArena,
    _norm: ResidualNormOp<A>,
    _input: GdnInputProjectionOp<A>,
    _prepare: GdnPrepareOp<A>,
    _recurrence: GdnRecurrenceOp<A>,
    _output: GdnOutputProjectionOp<A>,
    _swiglu: DenseFp8SwiGluOp<A>,
    _down: DenseFp8DownOp<A>,
    snapshot: Arc<CheckpointSnapshot<A>>,
    context: Arc<CudaContext>,
    layout: DenseFp8GdnLayerLayout,
    base_address: u64,
    layer: usize,
    arch: PhantomData<A>,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    mixer_normalized: *mut u16,
    input_activation_codes: *mut u8,
    input_activation_scales: *mut f32,
    input_weight_codes: *const u8,
    input_weight_scales: *const u16,
    projected: *mut u16,
    control_weights: *const u16,
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
    recurrent_plane: *mut f32,
    recurrent_output: *mut u16,
    output_activation_codes: *mut u8,
    output_activation_scales: *mut f32,
    output_weight_codes: *const u8,
    output_weight_scales: *const u16,
    mixer_branch: *mut u16,
    post_attention_norm: *const u16,
    mixer_residual: *mut u16,
    mlp_normalized: *mut u16,
    gate_up_activation_codes: *mut u8,
    gate_up_activation_scales: *mut f32,
    gate_up_weight_codes: *const u8,
    gate_up_weight_scales: *const u16,
    swiglu: *mut u16,
    down_activation_codes: *mut u8,
    down_activation_scales: *mut f32,
    down_weight_codes: *const u8,
    down_weight_scales: *const u16,
    mlp_branch: *mut u16,
    next_norm: *const u16,
    residual_output: *mut u16,
    next_normalized: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: GdnLayerRegions) -> GpuResult<Self> {
        Ok(Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            input_activation_codes: arena.address(regions.input_activation_codes)?,
            input_activation_scales: arena.address(regions.input_activation_scales)?,
            input_weight_codes: arena.address(regions.input_weight_codes)?.cast_const(),
            input_weight_scales: arena.address(regions.input_weight_scales)?.cast_const(),
            projected: arena.address(regions.projected)?,
            control_weights: arena.address(regions.control_weights)?.cast_const(),
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
            recurrent_plane: arena.address(regions.recurrent_plane)?,
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
            gate_up_weight_codes: arena.address(regions.gate_up_weight_codes)?.cast_const(),
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
        })
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
            self.projected.addr(),
            self.control_weights.addr(),
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
            self.gate_up_weight_codes.addr(),
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

impl<A: Sm120Arch> DenseFp8GdnLayerProgram<A> {
    /// Loads one admitted layer into one arena and captures exact `B=1..=8`.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<A>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let gdn = GdnBindings::bind(snapshot.as_ref(), layer)?;
        let mlp = DenseFp8MlpBindings::bind(snapshot.as_ref(), layer)?;
        let layout = DenseFp8GdnLayerLayout::build::<A>()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let input = GdnInputProjectionOp::new(context)?;
        let prepare = GdnPrepareOp::new(context)?;
        let recurrence = GdnRecurrenceOp::new(context)?;
        let output = GdnOutputProjectionOp::new(context)?;
        let swiglu = DenseFp8SwiGluOp::new(context)?;
        let down = DenseFp8DownOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &gdn.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.input_weight_codes, gdn.input_weight_e4m3)?;
        arena.copy_from_host(
            &stream,
            regions.input_weight_scales,
            &little_endian_words(gdn.input_scale_bf16)?,
        )?;
        let mut control_weights = gdn.a_control_weight.words().collect::<Vec<_>>();
        control_weights.extend(gdn.b_control_weight.words());
        arena.copy_from_host(&stream, regions.control_weights, &control_weights)?;
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
        arena.copy_from_host(
            &stream,
            regions.output_weight_codes,
            gdn.output_weight.codes(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_scales,
            &gdn.output_scale.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.post_attention_norm,
            &gdn.post_attention_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.gate_up_weight_codes,
            mlp.gate_up.weight_e4m3,
        )?;
        arena.copy_from_host(
            &stream,
            regions.gate_up_weight_scales,
            &little_endian_words(mlp.gate_up.scale_bf16)?,
        )?;
        arena.copy_from_host(&stream, regions.down_weight_codes, mlp.down.weight.codes())?;
        arena.copy_from_host(
            &stream,
            regions.down_weight_scales,
            &mlp.down.scale.words().collect::<Vec<_>>(),
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

        let pointers = Pointers::bind(&arena, regions)?;
        // SAFETY: the one owner arena keeps all encoded source addresses stable.
        let gate_up_maps = unsafe {
            DenseFp8SwiGluTmaMaps::new(
                &stream,
                pointers.gate_up_activation_codes.cast_const(),
                pointers.gate_up_weight_codes,
            )?
        };
        // SAFETY: the one owner arena keeps all encoded source addresses stable.
        let down_maps = unsafe {
            DenseFp8DownTmaMaps::new(
                &stream,
                pointers.down_activation_codes.cast_const(),
                pointers.down_weight_codes,
            )?
        };
        // SAFETY: the one owner arena keeps all encoded source addresses stable.
        let input_maps = unsafe {
            DenseFp8GdnInputTmaMaps::new(
                &stream,
                pointers.input_activation_codes.cast_const(),
                pointers.input_weight_codes,
            )?
        };
        // SAFETY: the one owner arena keeps all encoded source addresses stable.
        let output_maps = unsafe {
            DenseFp8GdnOutputTmaMaps::new(
                &stream,
                pointers.output_activation_codes.cast_const(),
                pointers.output_weight_codes,
            )?
        };
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            input: &input,
            prepare: &prepare,
            recurrence: &recurrence,
            output: &output,
            swiglu: &swiglu,
            down: &down,
            gate_up_maps: &gate_up_maps,
            down_maps: &down_maps,
            input_maps: &input_maps,
            output_maps: &output_maps,
        };
        let graphs = capture_decode_routes(&stream, ops, pointers)?;
        let prefill_graphs = capture_prefill_routes(&stream, ops, pointers)?;

        Ok(Self {
            graphs,
            prefill_graphs,
            gate_up_maps,
            down_maps,
            input_maps,
            output_maps,
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
        let expected = product("dense-FP8 GDN input elements", rows, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "dense-FP8 GDN input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;
        Ok(())
    }

    /// Clears all slot-owned causal-convolution history and recurrent state.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.history, 0)?;
        self.arena.fill(stream, regions.state, 0)?;
        Ok(())
    }

    /// Replays the immutable graph for one exact decode or prefill width.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        // SAFETY: this DenseFp8GdnLayerProgram owns every captured allocation
        // (arena, TMA maps, op modules) for its whole life and drops the graphs first.
        unsafe { self.graph(rows)?.launch(stream) }?;
        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("dense-FP8 GDN output elements", rows, A::HIDDEN)?;
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

    /// Resident weights plus workspace, excluding alignment padding.
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

    /// Exact bytes in all eight address-bound projection tensor-map descriptors.
    pub const fn descriptor_bytes(&self) -> usize {
        DenseFp8SwiGluTmaMaps::BYTE_LEN
            + DenseFp8DownTmaMaps::BYTE_LEN
            + DenseFp8GdnInputTmaMaps::BYTE_LEN
            + DenseFp8GdnOutputTmaMaps::BYTE_LEN
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &DenseFp8GdnLayerLayout {
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
                "dense-FP8 GDN layer row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
            ))
        })?;

        Ok(&self.prefill_graphs[index])
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
        )?;
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Returns one captured production graph.
    pub fn qualification_graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        self.graph(rows)
    }

    #[cfg(feature = "qualification")]
    /// Captures the production owner's exact history and state reset.
    pub fn qualification_state_reset_graph(&self, stream: &CudaStream) -> EngineResult<CudaGraph> {
        let regions = self.layout.regions();
        Ok(CudaGraph::capture(stream, || {
            self.arena.fill(stream, regions.history, 0)?;
            self.arena.fill(stream, regions.state, 0)
        })?)
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
                "repeated dense-FP8 GDN graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, rows, ops, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        let mut addresses = Pointers::bind(&self.arena, self.layout.regions())?.addresses();
        addresses.extend(self.gate_up_maps.device_addresses());
        addresses.extend(self.down_maps.device_addresses());
        addresses.extend(self.input_maps.device_addresses());
        addresses.extend(self.output_maps.device_addresses());

        Ok(addresses)
    }

    #[cfg(feature = "qualification")]
    /// Copies all eight opaque address-bound tensor maps.
    pub fn qualification_descriptors(&self, stream: &CudaStream) -> EngineResult<[Vec<u64>; 8]> {
        let gate_up = self.gate_up_maps.copy_to_host(stream)?;
        let down = self.down_maps.copy_to_host(stream)?;
        let input = self.input_maps.copy_to_host(stream)?;
        let output = self.output_maps.copy_to_host(stream)?;

        Ok([
            gate_up[0].clone(),
            gate_up[1].clone(),
            down[0].clone(),
            down[1].clone(),
            input[0].clone(),
            input[1].clone(),
            output[0].clone(),
            output[1].clone(),
        ])
    }

    #[cfg(feature = "qualification")]
    /// Fills every non-state output plane with a byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.mixer_normalized,
            regions.mixer_branch,
            regions.mixer_residual,
            regions.mlp_normalized,
            regions.mlp_branch,
            regions.residual_output,
            regions.next_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.projected,
            regions.convolved,
            regions.recurrent_output,
            regions.swiglu,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.input_activation_codes,
            regions.output_activation_codes,
            regions.gate_up_activation_codes,
            regions.down_activation_codes,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.input_activation_scales,
            regions.log_decay,
            regions.beta,
            regions.output_activation_scales,
            regions.gate_up_activation_scales,
            regions.down_activation_scales,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all working and persistent planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<DenseFp8GdnLayerObservables> {
        let regions = self.layout.regions();
        Ok(DenseFp8GdnLayerObservables {
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            input_activation_codes: self
                .arena
                .copy_to_host(stream, regions.input_activation_codes)?,
            input_activation_scales: self
                .arena
                .copy_to_host(stream, regions.input_activation_scales)?,
            projected: self.arena.copy_to_host(stream, regions.projected)?,
            log_decay: self.arena.copy_to_host(stream, regions.log_decay)?,
            beta: self.arena.copy_to_host(stream, regions.beta)?,
            convolved: self.arena.copy_to_host(stream, regions.convolved)?,
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
    fn ops(&self) -> Ops<'_, A> {
        Ops {
            norm: &self._norm,
            input: &self._input,
            prepare: &self._prepare,
            recurrence: &self._recurrence,
            output: &self._output,
            swiglu: &self._swiglu,
            down: &self._down,
            gate_up_maps: &self.gate_up_maps,
            down_maps: &self.down_maps,
            input_maps: &self.input_maps,
            output_maps: &self.output_maps,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete working and persistent planes exposed to the qualification crate.
pub struct DenseFp8GdnLayerObservables {
    /// Pre-mixer normalized rows.
    pub mixer_normalized: Vec<u16>,
    /// GDN-input dynamic E4M3 codes.
    pub input_activation_codes: Vec<u8>,
    /// GDN-input dynamic FP32 scales.
    pub input_activation_scales: Vec<f32>,
    /// Fused Q/K/V/Z projection rows.
    pub projected: Vec<u16>,
    /// Per-value-head log decays.
    pub log_decay: Vec<f32>,
    /// Per-value-head update gates.
    pub beta: Vec<f32>,
    /// Causal-convolved Q/K/V rows.
    pub convolved: Vec<u16>,
    /// Slot-owned causal history.
    pub history: Vec<u16>,
    /// Slot-owned FP32 recurrent state.
    pub state: Vec<f32>,
    /// Gated normalized recurrent values.
    pub recurrent_output: Vec<u16>,
    /// GDN-output dynamic E4M3 codes.
    pub output_activation_codes: Vec<u8>,
    /// GDN-output dynamic FP32 scales.
    pub output_activation_scales: Vec<f32>,
    /// GDN output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after the mixer.
    pub mixer_residual: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub mlp_normalized: Vec<u16>,
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
    /// Dense-FP8 down-projection branch.
    pub mlp_branch: Vec<u16>,
    /// Published layer residual rows.
    pub residual_output: Vec<u16>,
    /// Next-boundary normalized rows.
    pub next_normalized: Vec<u16>,
}

#[derive(Clone, Copy)]
struct Ops<'a, A: Sm120Arch> {
    norm: &'a ResidualNormOp<A>,
    input: &'a GdnInputProjectionOp<A>,
    prepare: &'a GdnPrepareOp<A>,
    recurrence: &'a GdnRecurrenceOp<A>,
    output: &'a GdnOutputProjectionOp<A>,
    swiglu: &'a DenseFp8SwiGluOp<A>,
    down: &'a DenseFp8DownOp<A>,
    gate_up_maps: &'a DenseFp8SwiGluTmaMaps,
    down_maps: &'a DenseFp8DownTmaMaps,
    input_maps: &'a DenseFp8GdnInputTmaMaps,
    output_maps: &'a DenseFp8GdnOutputTmaMaps,
}

fn capture_decode_routes<A: Sm120Arch>(
    stream: &CudaStream,
    ops: Ops<'_, A>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    capture_batch_graphs(
        stream,
        "dense-FP8 GDN layer graph inventory has wrong cardinality",
        |batch| launch_route(stream, batch, ops, pointers),
    )
}

fn capture_prefill_routes<A: Sm120Arch>(
    stream: &CudaStream,
    ops: Ops<'_, A>,
    pointers: Pointers,
) -> EngineResult<[CudaGraph; 4]> {
    capture_route_graphs(
        stream,
        [32, 64, 128, MAX_ROWS],
        "dense-FP8 GDN layer prefill graph inventory has wrong cardinality",
        |rows| launch_route(stream, rows, ops, pointers),
    )
}

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_, A>,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: the single arena provides aligned, disjoint MAX_ROWS regions,
    // and exact dispatch restricts every launch to `rows` rows. Persistent
    // history and state keep eight decode rows; prefill uses mapped row zero.
    unsafe {
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        if rows == MAX_ROWS {
            // T=1024 amortizes address-bound TMA setup across the macro tile.
            ops.input.launch_macro_prefill(
                stream,
                pointers.mixer_normalized,
                pointers.input_activation_codes,
                pointers.input_activation_scales,
                pointers.input_weight_codes,
                pointers.input_weight_scales,
                pointers.projected,
                ops.input_maps,
            )?;
        } else {
            ops.input.launch(
                stream,
                rows,
                pointers.mixer_normalized,
                pointers.input_activation_codes,
                pointers.input_activation_scales,
                pointers.input_weight_codes,
                pointers.input_weight_scales,
                pointers.projected,
            )?;
        }
        ops.prepare.launch(
            stream,
            rows,
            pointers.mixer_normalized,
            pointers.control_weights,
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
            pointers.recurrent_plane,
            pointers.recurrent_output,
        )?;
        if rows == MAX_ROWS {
            // T=1024 amortizes address-bound TMA setup across the macro tile.
            ops.output.launch_macro_prefill(
                stream,
                pointers.recurrent_output,
                pointers.output_activation_codes,
                pointers.output_activation_scales,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
                pointers.mixer_branch,
                ops.output_maps,
            )?;
        } else {
            ops.output.launch(
                stream,
                rows,
                pointers.recurrent_output,
                pointers.output_activation_codes,
                pointers.output_activation_scales,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
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
        if rows == MAX_ROWS {
            // T=1024 amortizes address-bound TMA setup across the complete macro tile.
            ops.swiglu.launch_macro_prefill(
                stream,
                pointers.mlp_normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_up_weight_codes,
                pointers.gate_up_weight_scales,
                pointers.swiglu,
                ops.gate_up_maps,
            )?;
            ops.down.launch_macro_prefill(
                stream,
                pointers.swiglu,
                pointers.down_activation_codes,
                pointers.down_activation_scales,
                pointers.down_weight_codes,
                pointers.down_weight_scales,
                pointers.mlp_branch,
                ops.down_maps,
            )?;
        } else {
            ops.swiglu.launch(
                stream,
                rows,
                pointers.mlp_normalized,
                pointers.gate_up_activation_codes,
                pointers.gate_up_activation_scales,
                pointers.gate_up_weight_codes,
                pointers.gate_up_weight_scales,
                pointers.swiglu,
            )?;
            if rows <= MAX_BATCH {
                ops.down.launch(
                    stream,
                    rows,
                    pointers.swiglu,
                    pointers.down_activation_codes,
                    pointers.down_activation_scales,
                    pointers.down_weight_codes,
                    pointers.down_weight_scales,
                    pointers.mlp_branch,
                )?;
            } else {
                ops.down.launch_tail_prefill(
                    stream,
                    rows,
                    pointers.swiglu,
                    pointers.down_activation_codes,
                    pointers.down_activation_scales,
                    pointers.down_weight_codes,
                    pointers.down_weight_scales,
                    pointers.mlp_branch,
                )?;
            }
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
        MAX_ROWS => Some(3),
        _ => None,
    }
}

fn require_rows(rows: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&rows) && prefill_index(rows).is_none() {
        return Err(EngineError::route(format!(
            "dense-FP8 GDN layer row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, require_rows};

    #[test]
    fn exact_route_table_rejects_every_uncompiled_route() {
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024] {
            require_rows(rows).unwrap();
        }
        for rows in [0, 9, 31, 33, 63, 65, 127, 129, 1_023, 1_025, usize::MAX] {
            assert!(require_rows(rows).is_err(), "rows={rows}");
        }
        assert_eq!(MAX_BATCH, 8);
    }
}

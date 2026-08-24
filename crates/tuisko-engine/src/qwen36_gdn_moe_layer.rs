//! Resident source-backed Qwen3.6 GDN plus MoE decoder layer.

use crate::qwen36_gdn_moe_layer_layout::Qwen36GdnMoeLayerRegions;
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen36GdnMoeLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen36GdnInputOp, Qwen36GdnOutputOp, Qwen36GdnPrepareOp, Qwen36GdnRecurrenceOp,
    Qwen36MoeExpertsOp, Qwen36MoeRouterOp, Qwen36ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen36GdnBindings, Qwen36Moe35B, Qwen36MoeLayerBindings,
};

/// One Qwen3.6 linear-attention layer with immutable exact-batch graph routes.
pub struct Qwen36GdnMoeLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: Qwen36ResidualNormOp,
    _input: Qwen36GdnInputOp,
    _prepare: Qwen36GdnPrepareOp,
    _recurrence: Qwen36GdnRecurrenceOp,
    _output: Qwen36GdnOutputOp,
    _router: Qwen36MoeRouterOp,
    _experts: Qwen36MoeExpertsOp,
    snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    context: Arc<CudaContext>,
    layout: Qwen36GdnMoeLayerLayout,
    base_address: u64,
    source_scales: SourceScales,
    layer: usize,
}

#[derive(Clone, Copy)]
struct SourceScales {
    input: f32,
    qkv_weight: f32,
    z_weight: f32,
    output_input: f32,
    output_weight: f32,
    shared_gate_up_weight: f32,
    shared_down_weight: f32,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    mixer_normalized: *mut u16,
    input_activation_codes: *mut u8,
    input_weight_codes: *const u8,
    control_weight_bf16: *const u16,
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
    output_weight_codes: *const u8,
    mixer_branch: *mut u16,
    post_attention_norm: *const u16,
    mixer_residual: *mut u16,
    moe_normalized: *mut u16,
    router_weight: *const u16,
    router_logits: *mut u16,
    expert_indices: *mut u16,
    routing_weights: *mut u16,
    routed_gate_up_codes: *const u8,
    routed_gate_up_scales: *const u8,
    routed_gate_up_weight_scales_2: *const f32,
    routed_down_codes: *const u8,
    routed_down_scales: *const u8,
    routed_down_weight_scales_2: *const f32,
    shared_gate_up_codes: *const u8,
    shared_gate_up_scales: *const u8,
    shared_down_codes: *const u8,
    shared_down_scales: *const u8,
    shared_gate_weight: *const u16,
    expert_intermediate: *mut u16,
    expert_output: *mut u16,
    shared_gate: *mut u16,
    moe_branch: *mut u16,
    next_norm: *const u16,
    residual_output: *mut u16,
    next_normalized: *mut u16,
}

impl Pointers {
    fn bind(arena: &DeviceArena, regions: Qwen36GdnMoeLayerRegions) -> GpuResult<Self> {
        Ok(Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            input_activation_codes: arena.address(regions.input_activation_codes)?,
            input_weight_codes: arena.address(regions.input_weight_codes)?.cast_const(),
            control_weight_bf16: arena.address(regions.control_weight_bf16)?.cast_const(),
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
            output_weight_codes: arena.address(regions.output_weight_codes)?.cast_const(),
            mixer_branch: arena.address(regions.mixer_branch)?,
            post_attention_norm: arena.address(regions.post_attention_norm)?.cast_const(),
            mixer_residual: arena.address(regions.mixer_residual)?,
            moe_normalized: arena.address(regions.moe_normalized)?,
            router_weight: arena.address(regions.router_weight)?.cast_const(),
            router_logits: arena.address(regions.router_logits)?,
            expert_indices: arena.address(regions.expert_indices)?,
            routing_weights: arena.address(regions.routing_weights)?,
            routed_gate_up_codes: arena.address(regions.routed_gate_up_codes)?.cast_const(),
            routed_gate_up_scales: arena.address(regions.routed_gate_up_scales)?.cast_const(),
            routed_gate_up_weight_scales_2: arena
                .address(regions.routed_gate_up_weight_scales_2)?
                .cast_const(),
            routed_down_codes: arena.address(regions.routed_down_codes)?.cast_const(),
            routed_down_scales: arena.address(regions.routed_down_scales)?.cast_const(),
            routed_down_weight_scales_2: arena
                .address(regions.routed_down_weight_scales_2)?
                .cast_const(),
            shared_gate_up_codes: arena.address(regions.shared_gate_up_codes)?.cast_const(),
            shared_gate_up_scales: arena.address(regions.shared_gate_up_scales)?.cast_const(),
            shared_down_codes: arena.address(regions.shared_down_codes)?.cast_const(),
            shared_down_scales: arena.address(regions.shared_down_scales)?.cast_const(),
            shared_gate_weight: arena.address(regions.shared_gate_weight)?.cast_const(),
            expert_intermediate: arena.address(regions.expert_intermediate)?,
            expert_output: arena.address(regions.expert_output)?,
            shared_gate: arena.address(regions.shared_gate)?,
            moe_branch: arena.address(regions.moe_branch)?,
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
            self.input_weight_codes.addr(),
            self.control_weight_bf16.addr(),
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
            self.output_weight_codes.addr(),
            self.mixer_branch.addr(),
            self.post_attention_norm.addr(),
            self.mixer_residual.addr(),
            self.moe_normalized.addr(),
            self.router_weight.addr(),
            self.router_logits.addr(),
            self.expert_indices.addr(),
            self.routing_weights.addr(),
            self.routed_gate_up_codes.addr(),
            self.routed_gate_up_scales.addr(),
            self.routed_gate_up_weight_scales_2.addr(),
            self.routed_down_codes.addr(),
            self.routed_down_scales.addr(),
            self.routed_down_weight_scales_2.addr(),
            self.shared_gate_up_codes.addr(),
            self.shared_gate_up_scales.addr(),
            self.shared_down_codes.addr(),
            self.shared_down_scales.addr(),
            self.shared_gate_weight.addr(),
            self.expert_intermediate.addr(),
            self.expert_output.addr(),
            self.shared_gate.addr(),
            self.moe_branch.addr(),
            self.next_norm.addr(),
            self.residual_output.addr(),
            self.next_normalized.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    norm: &'a Qwen36ResidualNormOp,
    input: &'a Qwen36GdnInputOp,
    prepare: &'a Qwen36GdnPrepareOp,
    recurrence: &'a Qwen36GdnRecurrenceOp,
    output: &'a Qwen36GdnOutputOp,
    router: &'a Qwen36MoeRouterOp,
    experts: &'a Qwen36MoeExpertsOp,
}

impl Qwen36GdnMoeLayerProgram {
    /// Loads one admitted source layer and captures exact `B=1..=8` decode routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let gdn = Qwen36GdnBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let moe = Qwen36MoeLayerBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let post_attention_norm = gdn.post_attention_norm.words().collect::<Vec<_>>();
        if post_attention_norm != moe.input_norm.words().collect::<Vec<_>>() {
            return Err(EngineError::layout(format!(
                "Qwen3.6 layer {layer} GDN/MoE boundary norms differ"
            )));
        }

        let layout = Qwen36GdnMoeLayerLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen36ResidualNormOp::new(context)?;
        let input = Qwen36GdnInputOp::new(context)?;
        let prepare = Qwen36GdnPrepareOp::new(context)?;
        let recurrence = Qwen36GdnRecurrenceOp::new(context)?;
        let output = Qwen36GdnOutputOp::new(context)?;
        let router = Qwen36MoeRouterOp::new(context)?;
        let experts = Qwen36MoeExpertsOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &gdn.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.input_weight_codes, &gdn.input_weight_e4m3)?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_bf16,
            &bf16_words(&gdn.control_weight_bf16)?,
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
        arena.copy_from_host(&stream, regions.output_weight_codes, gdn.output.weight_e4m3)?;
        arena.copy_from_host(&stream, regions.post_attention_norm, &post_attention_norm)?;
        arena.copy_from_host(
            &stream,
            regions.router_weight,
            &moe.router_weight.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_gate_up_codes,
            &moe.experts.gate_up_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_gate_up_scales,
            &moe.experts.gate_up_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_gate_up_weight_scales_2,
            &moe.experts.gate_up_weight_scales_2,
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_down_codes,
            &moe.experts.down_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_down_scales,
            &moe.experts.down_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.routed_down_weight_scales_2,
            &moe.experts.down_weight_scales_2,
        )?;
        arena.copy_from_host(
            &stream,
            regions.shared_gate_up_codes,
            &moe.shared_expert.gate_up_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.shared_gate_up_scales,
            &moe.shared_expert.gate_up_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.shared_down_codes,
            &moe.shared_expert.down_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.shared_down_scales,
            &moe.shared_expert.down_scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.shared_gate_weight,
            &moe.shared_expert_gate_weight.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.next_norm,
            &moe.next_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.state_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let source_scales = SourceScales {
            input: gdn.input_scale,
            qkv_weight: gdn.input_weight_scales[0],
            z_weight: gdn.input_weight_scales[1],
            output_input: gdn.output.input_scale,
            output_weight: gdn.output.weight_scale,
            shared_gate_up_weight: moe.shared_expert.gate_up_weight_scales_2[0],
            shared_down_weight: moe.shared_expert.down_weight_scales_2[0],
        };
        let pointers = Pointers::bind(&arena, regions)?;
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            input: &input,
            prepare: &prepare,
            recurrence: &recurrence,
            output: &output,
            router: &router,
            experts: &experts,
        };
        let graphs = capture_routes(&stream, ops, pointers, source_scales)?;

        Ok(Self {
            graphs,
            arena,
            _norm: norm,
            _input: input,
            _prepare: prepare,
            _recurrence: recurrence,
            _output: output,
            _router: router,
            _experts: experts,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            source_scales,
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
        let expected = product("Qwen3.6 layer input", batch, Qwen36Moe35B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.6 layer input has {} values, expected {expected} for B={batch}",
                values.len()
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

    /// Replays the immutable graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen36GdnMoeLayerProgram owns every captured allocation
        // (arena, op modules) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = product("Qwen3.6 layer output", batch, Qwen36Moe35B::HIDDEN)?;

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

    /// Checked owner layout.
    pub const fn layout(&self) -> &Qwen36GdnMoeLayerLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen36Moe35B>> {
        &self.snapshot
    }

    /// Launches this layer from another resident owner's BF16 residual plane.
    ///
    /// # Safety
    /// `input` covers `batch * 2,048` BF16 values in this CUDA context.
    #[allow(dead_code)]
    pub(crate) unsafe fn launch_from(
        &self,
        stream: &CudaStream,
        batch: usize,
        input: *const u16,
    ) -> GpuResult<*const u16> {
        let mut pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        pointers.residual_input = input;
        launch_route(stream, batch, self.ops(), pointers, self.source_scales)?;

        Ok(pointers.residual_output.cast_const())
    }

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        launch_route(
            stream,
            batch,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
            self.source_scales,
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
                "repeated Qwen3.6 GDN/MoE graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        let scales = self.source_scales;

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, batch, ops, pointers, scales)?;
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
    /// Returns the exact static scales passed to the projection and shared-expert routes.
    pub const fn qualification_source_scales(&self) -> [f32; 7] {
        [
            self.source_scales.input,
            self.source_scales.qkv_weight,
            self.source_scales.z_weight,
            self.source_scales.output_input,
            self.source_scales.output_weight,
            self.source_scales.shared_gate_up_weight,
            self.source_scales.shared_down_weight,
        ]
    }

    #[cfg(feature = "qualification")]
    /// Fills all non-state mutable seams with one byte sentinel.
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
            regions.moe_normalized,
            regions.router_logits,
            regions.expert_indices,
            regions.routing_weights,
            regions.expert_intermediate,
            regions.expert_output,
            regions.shared_gate,
            regions.moe_branch,
            regions.residual_output,
            regions.next_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.input_activation_codes,
            regions.output_activation_codes,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [regions.log_decay, regions.beta] {
            self.arena.fill(stream, region, byte)?;
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all mutable planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36GdnMoeLayerObservables> {
        let regions = self.layout.regions();

        Ok(Qwen36GdnMoeLayerObservables {
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            input_activation_codes: self
                .arena
                .copy_to_host(stream, regions.input_activation_codes)?,
            projected: self.arena.copy_to_host(stream, regions.projected)?,
            projected_controls: self
                .arena
                .copy_to_host(stream, regions.projected_controls)?,
            log_decay: self.arena.copy_to_host(stream, regions.log_decay)?,
            beta: self.arena.copy_to_host(stream, regions.beta)?,
            convolved: self.arena.copy_to_host(stream, regions.convolved)?,
            history: self.arena.copy_to_host(stream, regions.history)?,
            state: self.arena.copy_to_host(stream, regions.state)?,
            recurrent_output: self.arena.copy_to_host(stream, regions.recurrent_output)?,
            output_activation_codes: self
                .arena
                .copy_to_host(stream, regions.output_activation_codes)?,
            mixer_branch: self.arena.copy_to_host(stream, regions.mixer_branch)?,
            mixer_residual: self.arena.copy_to_host(stream, regions.mixer_residual)?,
            moe_normalized: self.arena.copy_to_host(stream, regions.moe_normalized)?,
            router_logits: self.arena.copy_to_host(stream, regions.router_logits)?,
            expert_indices: self.arena.copy_to_host(stream, regions.expert_indices)?,
            routing_weights: self.arena.copy_to_host(stream, regions.routing_weights)?,
            expert_intermediate: self
                .arena
                .copy_to_host(stream, regions.expert_intermediate)?,
            expert_output: self.arena.copy_to_host(stream, regions.expert_output)?,
            shared_gate: self.arena.copy_to_host(stream, regions.shared_gate)?,
            moe_branch: self.arena.copy_to_host(stream, regions.moe_branch)?,
            residual_output: self.arena.copy_to_host(stream, regions.residual_output)?,
            next_normalized: self.arena.copy_to_host(stream, regions.next_normalized)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads every immutable device plane in source/materialized order.
    pub fn qualification_immutable(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36GdnMoeLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen36GdnMoeLayerImmutable {
            input_norm: self.arena.copy_to_host(stream, regions.input_norm)?,
            input_weight_codes: self
                .arena
                .copy_to_host(stream, regions.input_weight_codes)?,
            control_weight_bf16: self
                .arena
                .copy_to_host(stream, regions.control_weight_bf16)?,
            a_log: self.arena.copy_to_host(stream, regions.a_log)?,
            dt_bias: self.arena.copy_to_host(stream, regions.dt_bias)?,
            convolution_weights: self
                .arena
                .copy_to_host(stream, regions.convolution_weights)?,
            recurrent_norm: self.arena.copy_to_host(stream, regions.recurrent_norm)?,
            output_weight_codes: self
                .arena
                .copy_to_host(stream, regions.output_weight_codes)?,
            post_attention_norm: self
                .arena
                .copy_to_host(stream, regions.post_attention_norm)?,
            router_weight: self.arena.copy_to_host(stream, regions.router_weight)?,
            routed_gate_up_codes: self
                .arena
                .copy_to_host(stream, regions.routed_gate_up_codes)?,
            routed_gate_up_scales: self
                .arena
                .copy_to_host(stream, regions.routed_gate_up_scales)?,
            routed_gate_up_weight_scales_2: self
                .arena
                .copy_to_host(stream, regions.routed_gate_up_weight_scales_2)?,
            routed_down_codes: self.arena.copy_to_host(stream, regions.routed_down_codes)?,
            routed_down_scales: self
                .arena
                .copy_to_host(stream, regions.routed_down_scales)?,
            routed_down_weight_scales_2: self
                .arena
                .copy_to_host(stream, regions.routed_down_weight_scales_2)?,
            shared_gate_up_codes: self
                .arena
                .copy_to_host(stream, regions.shared_gate_up_codes)?,
            shared_gate_up_scales: self
                .arena
                .copy_to_host(stream, regions.shared_gate_up_scales)?,
            shared_down_codes: self.arena.copy_to_host(stream, regions.shared_down_codes)?,
            shared_down_scales: self
                .arena
                .copy_to_host(stream, regions.shared_down_scales)?,
            shared_gate_weight: self
                .arena
                .copy_to_host(stream, regions.shared_gate_weight)?,
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
            router: &self._router,
            experts: &self._experts,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen36GdnMoeLayerObservables {
    /// Pre-mixer normalized rows.
    pub mixer_normalized: Vec<u16>,
    /// Static E4M3 mixer-input activation codes.
    pub input_activation_codes: Vec<u8>,
    /// Fused Q/K/V/Z projection rows.
    pub projected: Vec<u16>,
    /// BF16 A/B control projection rows.
    pub projected_controls: Vec<u16>,
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
    /// Static E4M3 output-projection activation codes.
    pub output_activation_codes: Vec<u8>,
    /// GDN output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after the GDN mixer.
    pub mixer_residual: Vec<u16>,
    /// Pre-MoE normalized rows.
    pub moe_normalized: Vec<u16>,
    /// BF16 logits for all 256 routed experts.
    pub router_logits: Vec<u16>,
    /// Selected top-eight expert indices.
    pub expert_indices: Vec<u16>,
    /// Renormalized top-eight BF16 routing weights.
    pub routing_weights: Vec<u16>,
    /// Routed and shared expert SwiGLU values.
    pub expert_intermediate: Vec<u16>,
    /// Routed and shared expert down-projection values.
    pub expert_output: Vec<u16>,
    /// Shared-expert gate logits.
    pub shared_gate: Vec<u16>,
    /// Combined routed plus shared MoE branch.
    pub moe_branch: Vec<u16>,
    /// Published layer residual rows.
    pub residual_output: Vec<u16>,
    /// Next-boundary normalized rows.
    pub next_normalized: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Immutable source-backed planes exposed to the qualification crate.
pub struct Qwen36GdnMoeLayerImmutable {
    /// Input RMSNorm weights.
    pub input_norm: Vec<u16>,
    /// Fused source E4M3 Q/K/V/Z weights.
    pub input_weight_codes: Vec<u8>,
    /// BF16 A/B control weights.
    pub control_weight_bf16: Vec<u16>,
    /// Log-space recurrence decay parameters.
    pub a_log: Vec<u16>,
    /// Recurrence time-step bias.
    pub dt_bias: Vec<u16>,
    /// Width-four convolution weights.
    pub convolution_weights: Vec<u16>,
    /// Per-head recurrent RMSNorm weights.
    pub recurrent_norm: Vec<u16>,
    /// Source E4M3 recurrent-output projection weights.
    pub output_weight_codes: Vec<u8>,
    /// Post-mixer RMSNorm weights shared with the MoE boundary.
    pub post_attention_norm: Vec<u16>,
    /// BF16 router weights.
    pub router_weight: Vec<u16>,
    /// Numeric-order routed gate/up E2M1 codes.
    pub routed_gate_up_codes: Vec<u8>,
    /// Swizzled routed gate/up E4M3 scales.
    pub routed_gate_up_scales: Vec<u8>,
    /// Routed gate/up second-stage scales.
    pub routed_gate_up_weight_scales_2: Vec<f32>,
    /// Numeric-order routed down E2M1 codes.
    pub routed_down_codes: Vec<u8>,
    /// Swizzled routed down E4M3 scales.
    pub routed_down_scales: Vec<u8>,
    /// Routed down second-stage scales.
    pub routed_down_weight_scales_2: Vec<f32>,
    /// Shared-expert gate/up E2M1 codes.
    pub shared_gate_up_codes: Vec<u8>,
    /// Shared-expert gate/up E4M3 scales.
    pub shared_gate_up_scales: Vec<u8>,
    /// Shared-expert down E2M1 codes.
    pub shared_down_codes: Vec<u8>,
    /// Shared-expert down E4M3 scales.
    pub shared_down_scales: Vec<u8>,
    /// BF16 shared-expert gate weights.
    pub shared_gate_weight: Vec<u16>,
    /// Next-boundary RMSNorm weights.
    pub next_norm: Vec<u16>,
}

fn capture_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
    scales: SourceScales,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, ops, pointers, scales)
        })?);
    }

    graphs
        .try_into()
        .map_err(|_| EngineError::layout("Qwen3.6 GDN/MoE graph inventory has wrong cardinality"))
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    scales: SourceScales,
) -> GpuResult<()> {
    // SAFETY: one arena owns aligned, disjoint maximum-batch planes; each
    // exact-B launcher bounds work to active rows and mapped slot state.
    unsafe {
        ops.norm.launch_plain(
            stream,
            batch,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        ops.input.launch(
            stream,
            batch,
            pointers.mixer_normalized,
            pointers.input_activation_codes,
            scales.input,
            pointers.input_weight_codes,
            scales.qkv_weight,
            scales.z_weight,
            pointers.control_weight_bf16,
            pointers.projected,
            pointers.projected_controls,
        )?;
        ops.prepare.launch(
            stream,
            batch,
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
            batch,
            pointers.convolved,
            pointers.projected,
            pointers.log_decay,
            pointers.beta,
            pointers.recurrent_norm,
            pointers.state_rows,
            pointers.state,
            pointers.recurrent_output,
        )?;
        ops.output.launch(
            stream,
            batch,
            pointers.recurrent_output,
            pointers.output_activation_codes,
            scales.output_input,
            pointers.output_weight_codes,
            scales.output_weight,
            pointers.mixer_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            batch,
            pointers.residual_input,
            pointers.mixer_branch,
            pointers.post_attention_norm,
            pointers.mixer_residual,
            pointers.moe_normalized,
        )?;
        ops.router.launch(
            stream,
            batch,
            pointers.moe_normalized,
            pointers.router_weight,
            pointers.router_logits,
            pointers.expert_indices,
            pointers.routing_weights,
        )?;
        ops.experts.launch(
            stream,
            batch,
            pointers.moe_normalized,
            pointers.expert_indices,
            pointers.routing_weights,
            pointers.routed_gate_up_codes,
            pointers.routed_gate_up_scales,
            pointers.routed_gate_up_weight_scales_2,
            pointers.routed_down_codes,
            pointers.routed_down_scales,
            pointers.routed_down_weight_scales_2,
            pointers.shared_gate_up_codes,
            pointers.shared_gate_up_scales,
            scales.shared_gate_up_weight,
            pointers.shared_down_codes,
            pointers.shared_down_scales,
            scales.shared_down_weight,
            pointers.shared_gate_weight,
            pointers.expert_intermediate,
            pointers.expert_output,
            pointers.shared_gate,
            pointers.moe_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            batch,
            pointers.mixer_residual,
            pointers.moe_branch,
            pointers.next_norm,
            pointers.residual_output,
            pointers.next_normalized,
        )?;
    }

    Ok(())
}

fn bf16_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(EngineError::layout(
            "Qwen3.6 BF16 source plane has an odd byte length",
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
            "Qwen3.6 GDN/MoE batch {batch} is outside the exact range 1..={MAX_BATCH}"
        )));
    }

    Ok(())
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

#[cfg(test)]
mod tests {
    use super::{bf16_words, require_batch};
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
    fn bf16_source_words_are_little_endian_and_even() {
        assert_eq!(
            bf16_words(&[0x80, 0x3f, 0x00, 0xbf]).unwrap(),
            [0x3f80, 0xbf00]
        );
        assert_eq!(
            bf16_words(&[0]).unwrap_err().code(),
            Some(EngineErrorCode::Layout)
        );
    }
}

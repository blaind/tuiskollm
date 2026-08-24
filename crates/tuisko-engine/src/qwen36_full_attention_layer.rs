//! Resident source-backed Qwen3.6 full-attention plus MoE decoder layer.

use crate::qwen36_full_attention_layer_layout::{
    QWEN36_CONTEXT_CAPACITY, QWEN36_TABLE_STRIDE, Qwen36FullAttentionLayerRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen36FullAttentionLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen36AttentionOutputOp, Qwen36AttentionQkPrepareOp, Qwen36Fp8QkvOp, Qwen36MoeExpertsOp,
    Qwen36MoeRouterOp, Qwen36PagedGqaOp, Qwen36ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen36FullAttentionBindings, Qwen36Moe35B, Qwen36MoeLayerBindings,
};

const ROTARY_PAIRS: usize = 32;

/// One Qwen3.6 full-attention layer with immutable exact-batch graph routes.
pub struct Qwen36FullAttentionLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    arena: DeviceArena,
    _norm: Qwen36ResidualNormOp,
    _qkv: Qwen36Fp8QkvOp,
    _qk_prepare: Qwen36AttentionQkPrepareOp,
    _paged_gqa: Qwen36PagedGqaOp,
    _attention_output: Qwen36AttentionOutputOp,
    _router: Qwen36MoeRouterOp,
    _experts: Qwen36MoeExpertsOp,
    snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
    context: Arc<CudaContext>,
    layout: Qwen36FullAttentionLayerLayout,
    base_address: u64,
    scales: SourceScales,
    layer: usize,
}

#[derive(Clone, Copy)]
struct SourceScales {
    qkv_input: f32,
    qkv_weight: [f32; 3],
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
    qkv_activation_codes: *mut u8,
    qkv_weight_codes: *const u8,
    qkv: *mut u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    query: *mut f32,
    key_pages: *mut u16,
    value_pages: *mut u16,
    attention: *mut f32,
    output_activation: *mut u16,
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
    fn bind(arena: &DeviceArena, regions: Qwen36FullAttentionLayerRegions) -> GpuResult<Self> {
        Ok(Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            qkv_activation_codes: arena.address(regions.qkv_activation_codes)?,
            qkv_weight_codes: arena.address(regions.qkv_weight_codes)?.cast_const(),
            qkv: arena.address(regions.qkv)?,
            query_norm: arena.address(regions.query_norm)?.cast_const(),
            key_norm: arena.address(regions.key_norm)?.cast_const(),
            rope_cos: arena.address(regions.rope_cos)?.cast_const(),
            rope_sin: arena.address(regions.rope_sin)?.cast_const(),
            block_tables: arena.address(regions.block_tables)?.cast_const(),
            table_rows: arena.address(regions.table_rows)?.cast_const(),
            cache_positions: arena.address(regions.cache_positions)?.cast_const(),
            lengths: arena.address(regions.lengths)?.cast_const(),
            query: arena.address(regions.query)?,
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,
            attention: arena.address(regions.attention)?,
            output_activation: arena.address(regions.output_activation)?,
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
            self.qkv_activation_codes.addr(),
            self.qkv_weight_codes.addr(),
            self.qkv.addr(),
            self.query_norm.addr(),
            self.key_norm.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.block_tables.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.lengths.addr(),
            self.query.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
            self.attention.addr(),
            self.output_activation.addr(),
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
    qkv: &'a Qwen36Fp8QkvOp,
    qk_prepare: &'a Qwen36AttentionQkPrepareOp,
    paged_gqa: &'a Qwen36PagedGqaOp,
    attention_output: &'a Qwen36AttentionOutputOp,
    router: &'a Qwen36MoeRouterOp,
    experts: &'a Qwen36MoeExpertsOp,
}

impl Qwen36FullAttentionLayerProgram {
    /// Loads one admitted source layer and captures exact `B=1..=8` decode routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen36Moe35B>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let attention =
            Qwen36FullAttentionBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let moe = Qwen36MoeLayerBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let post_attention_norm = attention.post_attention_norm.words().collect::<Vec<_>>();
        if post_attention_norm != moe.input_norm.words().collect::<Vec<_>>() {
            return Err(EngineError::layout(format!(
                "Qwen3.6 layer {layer} attention/MoE boundary norms differ"
            )));
        }

        let layout = Qwen36FullAttentionLayerLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen36ResidualNormOp::new(context)?;
        let qkv = Qwen36Fp8QkvOp::new(context)?;
        let qk_prepare = Qwen36AttentionQkPrepareOp::new(context)?;
        let paged_gqa = Qwen36PagedGqaOp::new(context)?;
        let attention_output = Qwen36AttentionOutputOp::new(context)?;
        let router = Qwen36MoeRouterOp::new(context)?;
        let experts = Qwen36MoeExpertsOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &attention.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.qkv_weight_codes,
            &attention.qkv_weight_e4m3,
        )?;
        arena.copy_from_host(
            &stream,
            regions.query_norm,
            &attention.query_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.key_norm,
            &attention.key_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_codes,
            attention.output.weight_e4m3,
        )?;
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
            regions.block_tables,
            &(0..(MAX_BATCH * QWEN36_TABLE_STRIDE) as u32).collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.table_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let scales = SourceScales {
            qkv_input: attention.qkv_input_scale,
            qkv_weight: attention.qkv_weight_scales,
            output_input: attention.output.input_scale,
            output_weight: attention.output.weight_scale,
            shared_gate_up_weight: moe.shared_expert.gate_up_weight_scales_2[0],
            shared_down_weight: moe.shared_expert.down_weight_scales_2[0],
        };
        let pointers = Pointers::bind(&arena, regions)?;
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            qkv: &qkv,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            router: &router,
            experts: &experts,
        };
        let graphs = capture_routes(&stream, ops, pointers, scales)?;

        Ok(Self {
            graphs,
            arena,
            _norm: norm,
            _qkv: qkv,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
            _router: router,
            _experts: experts,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            scales,
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
        let expected = product("Qwen3.6 attention input", batch, Qwen36Moe35B::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.6 attention input has {} values, expected {expected} for B={batch}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;

        Ok(())
    }

    /// Updates active positions, causal lengths, and 32 MRoPE pairs per token.
    pub fn load_decode_state(
        &self,
        stream: &CudaStream,
        batch: usize,
        positions: &[u32],
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        if positions.len() != batch {
            return Err(EngineError::layout(format!(
                "Qwen3.6 attention positions have {} values, expected {batch}",
                positions.len()
            )));
        }
        let rotary_values = product("Qwen3.6 attention rotary values", batch, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.6 attention rotary planes must each have {rotary_values} values for B={batch}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= QWEN36_CONTEXT_CAPACITY {
                    return Err(EngineError::route(format!(
                        "Qwen3.6 attention cache position {position} exceeds the {QWEN36_CONTEXT_CAPACITY}-token slot capacity"
                    )));
                }
                position.checked_add(1).ok_or_else(|| {
                    EngineError::route("Qwen3.6 attention cache length overflows")
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let regions = self.layout.regions();
        self.arena
            .copy_prefix_from_host(stream, regions.cache_positions, positions)?;
        self.arena
            .copy_prefix_from_host(stream, regions.lengths, &lengths)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_sin, rope_sin)?;

        Ok(())
    }

    /// Replaces both complete represented BF16 cache planes.
    pub fn load_cache(
        &self,
        stream: &CudaStream,
        key_pages: &[u16],
        value_pages: &[u16],
    ) -> EngineResult<()> {
        let regions = self.layout.regions();
        if key_pages.len() != regions.key_pages.len()
            || value_pages.len() != regions.value_pages.len()
        {
            return Err(EngineError::layout(format!(
                "Qwen3.6 attention cache planes must each have {} BF16 values",
                regions.key_pages.len()
            )));
        }
        self.arena
            .copy_from_host(stream, regions.key_pages, key_pages)?;
        self.arena
            .copy_from_host(stream, regions.value_pages, value_pages)?;

        Ok(())
    }

    /// Clears all slot-owned represented key/value cache pages.
    pub fn reset_cache(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        self.arena.fill(stream, regions.key_pages, 0)?;
        self.arena.fill(stream, regions.value_pages, 0)?;

        Ok(())
    }

    /// Replays the immutable graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        // SAFETY: this Qwen36FullAttentionLayerProgram owns every captured
        // allocation (arena, op modules) for its whole life and drops the graphs first.
        unsafe { self.graphs[batch - 1].launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = product("Qwen3.6 attention output", batch, Qwen36Moe35B::HIDDEN)?;

        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().residual_output, values)?)
    }

    /// Decoder layer owned by this program.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// CUDA context shared by the arena, graphs, and operators.
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

    /// Exact represented BF16 key/value cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Exact address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete single allocation, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Fixed short-context capacity of each decode slot.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &Qwen36FullAttentionLayerLayout {
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
        launch_route(stream, batch, self.ops(), pointers, self.scales)?;

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
            self.scales,
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
                "repeated Qwen3.6 full-attention graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        let scales = self.scales;

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
    /// Returns exact scalar FP8 and shared-expert scales in launch order.
    pub const fn qualification_source_scales(&self) -> [f32; 9] {
        [
            self.scales.qkv_input,
            self.scales.qkv_weight[0],
            self.scales.qkv_weight[1],
            self.scales.qkv_weight[2],
            self.scales.output_input,
            self.scales.output_weight,
            self.scales.shared_gate_up_weight,
            self.scales.shared_down_weight,
            Qwen36Moe35B::RMS_NORM_EPSILON,
        ]
    }

    #[cfg(feature = "qualification")]
    /// Fills every non-cache mutable seam with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.mixer_normalized,
            regions.qkv,
            regions.output_activation,
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
            regions.qkv_activation_codes,
            regions.output_activation_codes,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [regions.query, regions.attention] {
            self.arena.fill(stream, region, byte)?;
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every mutable seam, including complete persistent cache planes.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen36FullAttentionLayerObservables> {
        let regions = self.layout.regions();

        Ok(Qwen36FullAttentionLayerObservables {
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            qkv_activation_codes: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_codes)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
            key_pages: self.arena.copy_to_host(stream, regions.key_pages)?,
            value_pages: self.arena.copy_to_host(stream, regions.value_pages)?,
            attention: self.arena.copy_to_host(stream, regions.attention)?,
            output_activation: self.arena.copy_to_host(stream, regions.output_activation)?,
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
    ) -> EngineResult<Qwen36FullAttentionLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen36FullAttentionLayerImmutable {
            input_norm: self.arena.copy_to_host(stream, regions.input_norm)?,
            qkv_weight_codes: self.arena.copy_to_host(stream, regions.qkv_weight_codes)?,
            query_norm: self.arena.copy_to_host(stream, regions.query_norm)?,
            key_norm: self.arena.copy_to_host(stream, regions.key_norm)?,
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
            qkv: &self._qkv,
            qk_prepare: &self._qk_prepare,
            paged_gqa: &self._paged_gqa,
            attention_output: &self._attention_output,
            router: &self._router,
            experts: &self._experts,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen36FullAttentionLayerObservables {
    /// Pre-attention normalized rows.
    pub mixer_normalized: Vec<u16>,
    /// Static E4M3 QKV activation codes.
    pub qkv_activation_codes: Vec<u8>,
    /// Fused query/gate, key, and value rows.
    pub qkv: Vec<u16>,
    /// Prepared FP32 query heads.
    pub query: Vec<f32>,
    /// Complete represented BF16 key cache.
    pub key_pages: Vec<u16>,
    /// Complete represented BF16 value cache.
    pub value_pages: Vec<u16>,
    /// FP32 GQA output, gated in place by attention output.
    pub attention: Vec<f32>,
    /// Gated BF16 attention values.
    pub output_activation: Vec<u16>,
    /// Static E4M3 attention-output activation codes.
    pub output_activation_codes: Vec<u8>,
    /// Attention output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after attention.
    pub mixer_residual: Vec<u16>,
    /// Pre-MoE normalized rows.
    pub moe_normalized: Vec<u16>,
    /// Router logits for all experts.
    pub router_logits: Vec<u16>,
    /// Selected top-eight expert indices.
    pub expert_indices: Vec<u16>,
    /// Renormalized top-eight routing weights.
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
pub struct Qwen36FullAttentionLayerImmutable {
    /// Input RMSNorm weights.
    pub input_norm: Vec<u16>,
    /// Fused source E4M3 QKV weights.
    pub qkv_weight_codes: Vec<u8>,
    /// Per-head query RMSNorm weights.
    pub query_norm: Vec<u16>,
    /// Per-head key RMSNorm weights.
    pub key_norm: Vec<u16>,
    /// Source E4M3 attention-output weights.
    pub output_weight_codes: Vec<u8>,
    /// Post-attention RMSNorm weights.
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

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.6 full-attention graph inventory has wrong cardinality")
    })
}

fn launch_route(
    stream: &CudaStream,
    batch: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    scales: SourceScales,
) -> GpuResult<()> {
    // SAFETY: one arena owns aligned, disjoint maximum-batch planes. Fixed
    // three-page slot tables cover every admitted decode position, and every
    // launcher selects the separately compiled exact-B route.
    unsafe {
        ops.norm.launch_plain(
            stream,
            batch,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        ops.qkv.launch(
            stream,
            batch,
            pointers.mixer_normalized,
            pointers.qkv_activation_codes,
            scales.qkv_input,
            pointers.qkv_weight_codes,
            scales.qkv_weight[0],
            scales.qkv_weight[1],
            scales.qkv_weight[2],
            pointers.qkv,
        )?;
        ops.qk_prepare.launch(
            stream,
            batch,
            pointers.qkv,
            pointers.query_norm,
            pointers.key_norm,
            pointers.rope_cos,
            pointers.rope_sin,
            pointers.block_tables,
            pointers.table_rows,
            QWEN36_TABLE_STRIDE,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
        )?;
        ops.paged_gqa.launch(
            stream,
            batch,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            pointers.block_tables,
            pointers.table_rows,
            QWEN36_TABLE_STRIDE,
            pointers.lengths,
            pointers.attention,
        )?;
        ops.attention_output.launch(
            stream,
            batch,
            pointers.attention,
            pointers.qkv,
            pointers.output_activation,
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

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "Qwen3.6 full-attention batch {batch} is outside 1..={MAX_BATCH}"
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
    use super::{MAX_BATCH, ROTARY_PAIRS, require_batch};
    use crate::EngineErrorCode;

    #[test]
    fn exact_batch_table_rejects_every_boundary_neighbor() {
        for batch in 1..=MAX_BATCH {
            require_batch(batch).unwrap();
        }
        for batch in [0, 9, 16, usize::MAX] {
            let error = require_batch(batch).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
        assert_eq!(ROTARY_PAIRS, 32);
    }
}

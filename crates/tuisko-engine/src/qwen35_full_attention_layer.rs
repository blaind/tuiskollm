//! Resident source-backed Qwen3.5 full-attention layer.

use crate::qwen35_full_attention_layer_layout::{
    QWEN35_ATTENTION_MAX_ROWS, QWEN35_CONTEXT_CAPACITY, QWEN35_PREFILL_CONTEXT_CAPACITY,
    QWEN35_PREFILL_TABLE_STRIDE, QWEN35_TABLE_STRIDE, Qwen35FullAttentionLayerRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen35FullAttentionLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen35AttentionQkPrepareOp, Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4DownOp, Qwen35Nvfp4QkvOp,
    Qwen35Nvfp4SwiGluOp, Qwen35PagedGqaOp, Qwen35ResidualNormOp,
};
use tuisko_model::{
    Arch, CheckpointSnapshot, ModelOptNvfp4AttentionBindings, ModelOptNvfp4MlpBindings, Qwen35_9B,
};

const ROTARY_PAIRS: usize = 32;

/// One Qwen3.5 full-attention layer with immutable exact decode and prefill graphs.
pub struct Qwen35FullAttentionLayerProgram {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; 3],
    arena: DeviceArena,
    _norm: Qwen35ResidualNormOp,
    _qkv: Qwen35Nvfp4QkvOp,
    _qk_prepare: Qwen35AttentionQkPrepareOp,
    _paged_gqa: Qwen35PagedGqaOp,
    _attention_output: Qwen35Nvfp4AttentionOutputOp,
    _swiglu: Qwen35Nvfp4SwiGluOp,
    _down: Qwen35Nvfp4DownOp,
    snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
    context: Arc<CudaContext>,
    layout: Qwen35FullAttentionLayerLayout,
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
    qkv_activation_codes: *mut u8,
    qkv_activation_scales: *mut u8,
    qkv_weight_codes: *const u8,
    qkv_weight_scales: *const u8,
    qkv: *mut u16,
    query_norm: *const u16,
    key_norm: *const u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    block_tables: *const u32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    prefill_rope_cos: *const f32,
    prefill_rope_sin: *const f32,
    prefill_table_rows: *const u32,
    prefill_cache_positions: *const u32,
    prefill_lengths: *const u32,
    query: *mut f32,
    key_pages: *mut u16,
    value_pages: *mut u16,
    attention: *mut f32,
    output_activation: *mut u16,
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
    fn bind(arena: &DeviceArena, regions: Qwen35FullAttentionLayerRegions) -> GpuResult<Self> {
        let pointers = Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            input_norm: arena.address(regions.input_norm)?.cast_const(),
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            qkv_activation_codes: arena.address(regions.qkv_activation_codes)?,
            qkv_activation_scales: arena.address(regions.qkv_activation_scales)?,
            qkv_weight_codes: arena.address(regions.qkv_weight_codes)?.cast_const(),
            qkv_weight_scales: arena.address(regions.qkv_weight_scales)?.cast_const(),
            qkv: arena.address(regions.qkv)?,
            query_norm: arena.address(regions.query_norm)?.cast_const(),
            key_norm: arena.address(regions.key_norm)?.cast_const(),
            rope_cos: arena.address(regions.rope_cos)?.cast_const(),
            rope_sin: arena.address(regions.rope_sin)?.cast_const(),
            block_tables: arena.address(regions.block_tables)?.cast_const(),
            table_rows: arena.address(regions.table_rows)?.cast_const(),
            cache_positions: arena.address(regions.cache_positions)?.cast_const(),
            lengths: arena.address(regions.lengths)?.cast_const(),
            prefill_rope_cos: arena.address(regions.prefill_rope_cos)?.cast_const(),
            prefill_rope_sin: arena.address(regions.prefill_rope_sin)?.cast_const(),
            prefill_table_rows: arena.address(regions.prefill_table_rows)?.cast_const(),
            prefill_cache_positions: arena.address(regions.prefill_cache_positions)?.cast_const(),
            prefill_lengths: arena.address(regions.prefill_lengths)?.cast_const(),
            query: arena.address(regions.query)?,
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,
            attention: arena.address(regions.attention)?,
            output_activation: arena.address(regions.output_activation)?,
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
                "Qwen3.5 full-attention gate/up code planes are not adjacent",
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
            self.qkv_activation_codes.addr(),
            self.qkv_activation_scales.addr(),
            self.qkv_weight_codes.addr(),
            self.qkv_weight_scales.addr(),
            self.qkv.addr(),
            self.query_norm.addr(),
            self.key_norm.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.block_tables.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.lengths.addr(),
            self.prefill_rope_cos.addr(),
            self.prefill_rope_sin.addr(),
            self.prefill_table_rows.addr(),
            self.prefill_cache_positions.addr(),
            self.prefill_lengths.addr(),
            self.query.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
            self.attention.addr(),
            self.output_activation.addr(),
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
    qkv_input: f32,
    qkv_weight: [f32; 3],
    attention_output_input: f32,
    attention_output_weight: f32,
    gate_up_input: f32,
    gate_up_weight: f32,
    down_input: f32,
    down_weight: f32,
}

impl Qwen35FullAttentionLayerProgram {
    /// Loads one source layer and captures exact `B=1..8` and `T=32,64,128` routes.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen35_9B>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let attention =
            ModelOptNvfp4AttentionBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let mlp = ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let layout = Qwen35FullAttentionLayerLayout::build()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = Qwen35ResidualNormOp::new(context)?;
        let qkv = Qwen35Nvfp4QkvOp::new(context)?;
        let qk_prepare = Qwen35AttentionQkPrepareOp::new(context)?;
        let paged_gqa = Qwen35PagedGqaOp::new(context)?;
        let attention_output = Qwen35Nvfp4AttentionOutputOp::new(context)?;
        let swiglu = Qwen35Nvfp4SwiGluOp::new(context)?;
        let down = Qwen35Nvfp4DownOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &attention.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.qkv_weight_codes,
            &attention.qkv_weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.qkv_weight_scales,
            &attention.qkv_scale_e4m3_swizzled,
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
            attention.output.weight_e2m1,
        )?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_scales,
            &attention.output.scale_e4m3_swizzled,
        )?;
        arena.copy_from_host(
            &stream,
            regions.post_attention_norm,
            &attention.post_attention_norm.words().collect::<Vec<_>>(),
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
            regions.block_tables,
            &(0..(MAX_BATCH * QWEN35_TABLE_STRIDE) as u32).collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.table_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let scale_divisors = [
            attention.qkv_input_scale_divisor,
            attention.qkv_weight_scale_divisors[0],
            attention.qkv_weight_scale_divisors[1],
            attention.qkv_weight_scale_divisors[2],
            attention.output.input_scale_divisor,
            attention.output.weight_scale_divisor,
            mlp.gate_up.input_scale_divisor,
            mlp.gate_up.weight_scale_divisor,
            mlp.down.input_scale_divisor,
            mlp.down.weight_scale_divisor,
        ];
        let source_scales = [
            attention.qkv_input_scale,
            attention.qkv_weight_scales_2[0],
            attention.qkv_weight_scales_2[1],
            attention.qkv_weight_scales_2[2],
            attention.output.input_scale,
            attention.output.weight_scale_2,
            mlp.gate_up_input_scale,
            mlp.gate_up_weight_scale_2,
            mlp.down_input_scale,
            mlp.down_weight_scale_2,
        ];
        let divisors = launch_divisors(scale_divisors);
        let pointers = Pointers::bind(&arena, regions)?;
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            qkv: &qkv,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            swiglu: &swiglu,
            down: &down,
        };
        let graphs = capture_decode_routes(&stream, ops, pointers, divisors)?;
        let prefill_graphs = capture_prefill_routes(&stream, ops, pointers, divisors)?;

        Ok(Self {
            graphs,
            prefill_graphs,
            arena,
            _norm: norm,
            _qkv: qkv,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
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
        let expected = product(
            "Qwen3.5 full-attention input elements",
            rows,
            Qwen35_9B::HIDDEN,
        )?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.5 full-attention input has {} values, expected {expected} for {}",
                values.len(),
                route_name(rows),
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;

        Ok(())
    }

    /// Loads a contiguous from-empty causal prompt tile and its 32 MRoPE pairs.
    pub fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        if prefill_index(tokens).is_none() {
            return Err(EngineError::route(format!(
                "Qwen3.5 full-attention prefill tokens {tokens} are outside 32,64,128"
            )));
        }
        let rotary_values = product(
            "Qwen3.5 attention prefill rotary values",
            tokens,
            ROTARY_PAIRS,
        )?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 attention prefill rotary planes must each have {rotary_values} values for T={tokens}"
            )));
        }
        if tokens > QWEN35_PREFILL_CONTEXT_CAPACITY {
            return Err(EngineError::route(format!(
                "Qwen3.5 attention prefill T={tokens} exceeds the {QWEN35_PREFILL_CONTEXT_CAPACITY}-token shared cache"
            )));
        }

        let positions = (0..tokens as u32).collect::<Vec<_>>();
        let lengths = (1..=tokens as u32).collect::<Vec<_>>();
        let table_rows = vec![0u32; tokens];
        let regions = self.layout.regions();
        self.arena
            .copy_prefix_from_host(stream, regions.prefill_table_rows, &table_rows)?;
        self.arena
            .copy_prefix_from_host(stream, regions.prefill_cache_positions, &positions)?;
        self.arena
            .copy_prefix_from_host(stream, regions.prefill_lengths, &lengths)?;
        self.arena
            .copy_prefix_from_host(stream, regions.prefill_rope_cos, rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, regions.prefill_rope_sin, rope_sin)?;

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
                "Qwen3.5 full-attention positions have {} values, expected {batch}",
                positions.len()
            )));
        }
        let rotary_values = product("Qwen3.5 full-attention rotary values", batch, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "Qwen3.5 full-attention rotary planes must each have {rotary_values} values for B={batch}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= QWEN35_CONTEXT_CAPACITY {
                    return Err(EngineError::route(format!(
                        "Qwen3.5 full-attention cache position {position} exceeds the {QWEN35_CONTEXT_CAPACITY}-token slot capacity"
                    )));
                }
                position.checked_add(1).ok_or_else(|| {
                    EngineError::route("Qwen3.5 full-attention cache length overflows")
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
                "Qwen3.5 full-attention cache planes must each have {} BF16 values",
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

    /// Replays the immutable graph for one exact row route.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        // SAFETY: this Qwen35FullAttentionLayerProgram owns every captured
        // allocation (arena, op modules) for its whole life and drops the
        // graphs first.
        unsafe { self.graph(rows)?.launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product(
            "Qwen3.5 full-attention output elements",
            rows,
            Qwen35_9B::HIDDEN,
        )?;

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

    /// Fixed short-context capacity of each initial slot.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// From-empty prompt capacity of the shared physical-page row.
    pub const fn prefill_context_capacity(&self) -> usize {
        self.layout.prefill_context_capacity()
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Largest admitted exact row count.
    pub const fn row_capacity(&self) -> usize {
        QWEN35_ATTENTION_MAX_ROWS
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &Qwen35FullAttentionLayerLayout {
        &self.layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen35_9B>> {
        &self.snapshot
    }

    pub(crate) fn input_address(&self) -> GpuResult<*const u16> {
        Ok(Pointers::bind(&self.arena, self.layout.regions())?.residual_input)
    }

    fn graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        if (1..=MAX_BATCH).contains(&rows) {
            return Ok(&self.graphs[rows - 1]);
        }
        let index = prefill_index(rows).ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.5 full-attention row count {rows} is outside 1..={MAX_BATCH},32,64,128"
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
                "repeated Qwen3.5 full-attention graph requires at least one operation",
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
    /// Returns source-to-kernel scale divisors in QKV, output, gate/up, and down order.
    pub const fn qualification_divisors(&self) -> [f32; 10] {
        self.scale_divisors
    }

    #[cfg(feature = "qualification")]
    /// Returns the corresponding exact ModelOpt F32 source scales.
    pub const fn qualification_source_scales(&self) -> [f32; 10] {
        self.source_scales
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
            regions.mlp_normalized,
            regions.swiglu,
            regions.mlp_branch,
            regions.residual_output,
            regions.next_normalized,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [
            regions.qkv_activation_codes,
            regions.qkv_activation_scales,
            regions.output_activation_codes,
            regions.output_activation_scales,
            regions.gate_up_activation_codes,
            regions.gate_up_activation_scales,
            regions.down_activation_codes,
            regions.down_activation_scales,
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
    ) -> EngineResult<Qwen35FullAttentionLayerObservables> {
        let regions = self.layout.regions();

        Ok(Qwen35FullAttentionLayerObservables {
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            qkv_activation_codes: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_codes)?,
            qkv_activation_scales: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_scales)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            prefill_rope_cos: self.arena.copy_to_host(stream, regions.prefill_rope_cos)?,
            prefill_rope_sin: self.arena.copy_to_host(stream, regions.prefill_rope_sin)?,
            prefill_table_rows: self
                .arena
                .copy_to_host(stream, regions.prefill_table_rows)?,
            prefill_cache_positions: self
                .arena
                .copy_to_host(stream, regions.prefill_cache_positions)?,
            prefill_lengths: self.arena.copy_to_host(stream, regions.prefill_lengths)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
            key_pages: self.arena.copy_to_host(stream, regions.key_pages)?,
            value_pages: self.arena.copy_to_host(stream, regions.value_pages)?,
            attention: self.arena.copy_to_host(stream, regions.attention)?,
            output_activation: self.arena.copy_to_host(stream, regions.output_activation)?,
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
    ) -> EngineResult<Qwen35FullAttentionLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen35FullAttentionLayerImmutable {
            input_norm: self.arena.copy_to_host(stream, regions.input_norm)?,
            qkv_weight_codes: self.arena.copy_to_host(stream, regions.qkv_weight_codes)?,
            qkv_weight_scales: self.arena.copy_to_host(stream, regions.qkv_weight_scales)?,
            query_norm: self.arena.copy_to_host(stream, regions.query_norm)?,
            key_norm: self.arena.copy_to_host(stream, regions.key_norm)?,
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
            qkv: &self._qkv,
            qk_prepare: &self._qk_prepare,
            paged_gqa: &self._paged_gqa,
            attention_output: &self._attention_output,
            swiglu: &self._swiglu,
            down: &self._down,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen35FullAttentionLayerObservables {
    /// Pre-attention normalized residual rows.
    pub mixer_normalized: Vec<u16>,
    /// Packed E2M1 QKV-input activation codes.
    pub qkv_activation_codes: Vec<u8>,
    /// E4M3 QKV-input activation scales.
    pub qkv_activation_scales: Vec<u8>,
    /// Fused query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Prompt MRoPE cosine values.
    pub prefill_rope_cos: Vec<f32>,
    /// Prompt MRoPE sine values.
    pub prefill_rope_sin: Vec<f32>,
    /// Prompt block-table row indices.
    pub prefill_table_rows: Vec<u32>,
    /// Prompt cache append positions.
    pub prefill_cache_positions: Vec<u32>,
    /// Prompt causal attention lengths.
    pub prefill_lengths: Vec<u32>,
    /// Prepared FP32 query heads.
    pub query: Vec<f32>,
    /// Complete represented BF16 key cache.
    pub key_pages: Vec<u16>,
    /// Complete represented BF16 value cache.
    pub value_pages: Vec<u16>,
    /// FP32 paged-GQA output, gated in place by attention output.
    pub attention: Vec<f32>,
    /// BF16 gated attention values consumed by the output projection.
    pub output_activation: Vec<u16>,
    /// Packed E2M1 attention-output activation codes.
    pub output_activation_codes: Vec<u8>,
    /// E4M3 attention-output activation scales.
    pub output_activation_scales: Vec<u8>,
    /// BF16 attention output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after attention.
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
pub struct Qwen35FullAttentionLayerImmutable {
    /// Input RMSNorm weights.
    pub input_norm: Vec<u16>,
    /// Fused packed QKV weight codes.
    pub qkv_weight_codes: Vec<u8>,
    /// Fused swizzled QKV block scales.
    pub qkv_weight_scales: Vec<u8>,
    /// Query RMSNorm weights.
    pub query_norm: Vec<u16>,
    /// Key RMSNorm weights.
    pub key_norm: Vec<u16>,
    /// Packed attention-output weight codes.
    pub output_weight_codes: Vec<u8>,
    /// Swizzled attention-output block scales.
    pub output_weight_scales: Vec<u8>,
    /// Post-attention RMSNorm weights.
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
    qkv: &'a Qwen35Nvfp4QkvOp,
    qk_prepare: &'a Qwen35AttentionQkPrepareOp,
    paged_gqa: &'a Qwen35PagedGqaOp,
    attention_output: &'a Qwen35Nvfp4AttentionOutputOp,
    swiglu: &'a Qwen35Nvfp4SwiGluOp,
    down: &'a Qwen35Nvfp4DownOp,
}

fn launch_divisors(values: [f32; 10]) -> Divisors {
    Divisors {
        qkv_input: values[0],
        qkv_weight: [values[1], values[2], values[3]],
        attention_output_input: values[4],
        attention_output_weight: values[5],
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
        EngineError::layout("Qwen3.5 full-attention decode graph inventory has wrong cardinality")
    })
}

fn capture_prefill_routes(
    stream: &CudaStream,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> EngineResult<[CudaGraph; 3]> {
    // T=128 would otherwise require sixteen B=8 layer graphs and 15 extra
    // boundaries. One graph composes the same nine qualified T=128 leaves;
    // every leaf retains its accumulation and rounding order.
    let mut graphs = Vec::with_capacity(3);
    for rows in [32, 64, 128] {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, rows, ops, pointers, divisors)
        })?);
    }

    graphs.try_into().map_err(|_| {
        EngineError::layout("Qwen3.5 full-attention prefill graph inventory has wrong cardinality")
    })
}

fn launch_route(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
    divisors: Divisors,
) -> GpuResult<()> {
    let (rope_cos, rope_sin, table_rows, cache_positions, lengths, table_stride) =
        if rows <= MAX_BATCH {
            (
                pointers.rope_cos,
                pointers.rope_sin,
                pointers.table_rows,
                pointers.cache_positions,
                pointers.lengths,
                QWEN35_TABLE_STRIDE,
            )
        } else {
            (
                pointers.prefill_rope_cos,
                pointers.prefill_rope_sin,
                pointers.prefill_table_rows,
                pointers.prefill_cache_positions,
                pointers.prefill_lengths,
                QWEN35_PREFILL_TABLE_STRIDE,
            )
        };
    // SAFETY: one arena owns aligned, disjoint 128-row working planes. Decode
    // uses three-page slot rows; prefill uses one shared 24-page row. Every
    // leaf selects the same exact row count and retains its qualified arithmetic.
    unsafe {
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        if rows <= MAX_BATCH {
            ops.qkv.launch(
                stream,
                rows,
                pointers.mixer_normalized,
                pointers.qkv_weight_codes,
                pointers.qkv_weight_scales,
                divisors.qkv_weight,
                pointers.qkv,
            )?;
        } else {
            ops.qkv.launch_prefill(
                stream,
                rows,
                pointers.mixer_normalized,
                pointers.qkv_activation_codes,
                pointers.qkv_activation_scales,
                pointers.qkv_weight_codes,
                pointers.qkv_weight_scales,
                divisors.qkv_input,
                divisors.qkv_weight,
                pointers.qkv,
            )?;
        }
        ops.qk_prepare.launch(
            stream,
            rows,
            pointers.qkv,
            pointers.query_norm,
            pointers.key_norm,
            rope_cos,
            rope_sin,
            pointers.block_tables,
            table_rows,
            table_stride,
            cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
        )?;
        ops.paged_gqa.launch(
            stream,
            rows,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            pointers.block_tables,
            table_rows,
            table_stride,
            lengths,
            pointers.attention,
        )?;
        if rows <= MAX_BATCH {
            ops.attention_output.launch(
                stream,
                rows,
                pointers.attention,
                pointers.qkv,
                pointers.output_activation,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
                divisors.attention_output_weight,
                pointers.mixer_branch,
            )?;
        } else {
            ops.attention_output.launch_prefill(
                stream,
                rows,
                pointers.attention,
                pointers.qkv,
                pointers.output_activation,
                pointers.output_activation_codes,
                pointers.output_activation_scales,
                pointers.output_weight_codes,
                pointers.output_weight_scales,
                divisors.attention_output_input,
                divisors.attention_output_weight,
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
        if rows <= MAX_BATCH {
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

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "Qwen3.5 full-attention batch {batch} is outside 1..={MAX_BATCH}"
        )));
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

fn require_rows(rows: usize) -> EngineResult<()> {
    if (1..=MAX_BATCH).contains(&rows) || prefill_index(rows).is_some() {
        return Ok(());
    }

    Err(EngineError::route(format!(
        "Qwen3.5 full-attention row count {rows} is outside 1..={MAX_BATCH},32,64,128"
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
    use super::{MAX_BATCH, ROTARY_PAIRS, prefill_index, require_batch, require_rows};

    #[test]
    fn exact_batch_table_rejects_every_uncompiled_route() {
        for (batch, admitted) in [(0, false), (1, true), (8, true), (9, false), (16, false)] {
            assert_eq!(require_batch(batch).is_ok(), admitted, "batch={batch}");
        }
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(ROTARY_PAIRS, 32);
    }

    #[test]
    fn exact_row_table_covers_decode_and_prefill_only() {
        for rows in 1..=MAX_BATCH {
            require_rows(rows).unwrap();
        }
        assert_eq!(prefill_index(32), Some(0));
        assert_eq!(prefill_index(64), Some(1));
        assert_eq!(prefill_index(128), Some(2));
        for rows in [0, 9, 16, 31, 33, 63, 65, 127, 129, usize::MAX] {
            assert_eq!(prefill_index(rows), None);
            assert!(require_rows(rows).is_err(), "rows={rows}");
        }
    }
}

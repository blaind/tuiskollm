//! Resident source-backed dense-FP8 full-attention decoder layer.

use crate::full_attention_layer_layout::{
    CONTEXT_CAPACITY, FullAttentionLayerRegions, MAX_ROWS, PREFILL_CONTEXT_CAPACITY,
    PREFILL_TABLE_STRIDE, TABLE_STRIDE,
};
use crate::{EngineError, EngineResult, FullAttentionLayerLayout, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    AttentionOutputOp, AttentionQkPrepareOp, DenseFp8DownOp, DenseFp8DownTmaMaps, DenseFp8SwiGluOp,
    DenseFp8SwiGluTmaMaps, FullAttentionQkvOp, PagedGqaOp, ResidualNormOp, Sm120Arch,
};
use tuisko_model::{
    CheckpointSnapshot, DenseFp8MlpBindings, FullAttentionPostBindings, FullAttentionQkvBindings,
    Qwen38_27B,
};

const ROTARY_PAIRS: usize = 32;

/// One late full-attention layer with immutable exact decode and prefill graphs.
pub struct FullAttentionLayerProgram<A: Sm120Arch = Qwen38_27B> {
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
    _qkv: FullAttentionQkvOp<A>,
    _qk_prepare: AttentionQkPrepareOp<A>,
    _paged_gqa: PagedGqaOp<A>,
    _attention_output: AttentionOutputOp<A>,
    _swiglu: DenseFp8SwiGluOp<A>,
    _down: DenseFp8DownOp<A>,
    snapshot: Arc<CheckpointSnapshot<A>>,
    context: Arc<CudaContext>,
    layout: FullAttentionLayerLayout,
    base_address: u64,
    key_cache_scale: f32,
    value_cache_scale: f32,
    layer: usize,
    arch: PhantomData<A>,
}

#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    input_norm: *const u16,
    mixer_normalized: *mut u16,
    qkv_activation_codes: *mut u8,
    qkv_activation_scales: *mut f32,
    qkv_weight_codes: *const u8,
    qkv_weight_scales: *const u16,
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
    key_pages: *mut u8,
    value_pages: *mut u8,
    attention: *mut f32,
    macro_partials: *mut f32,
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
    fn bind(arena: &DeviceArena, regions: FullAttentionLayerRegions) -> GpuResult<Self> {
        Ok(Self {
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
            macro_partials: arena.address(regions.macro_partials)?,
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
            self.macro_partials.addr(),
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

impl<A: Sm120Arch> FullAttentionLayerProgram<A> {
    /// Loads one admitted source layer and captures every exact graph route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<A>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let qkv = FullAttentionQkvBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let post = FullAttentionPostBindings::bind(snapshot.as_ref(), layer)?;
        let mlp = DenseFp8MlpBindings::bind(snapshot.as_ref(), layer)?;
        let layout = FullAttentionLayerLayout::build::<A>()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let norm = ResidualNormOp::new(context)?;
        let qkv_op = FullAttentionQkvOp::new(context)?;
        let qk_prepare = AttentionQkPrepareOp::new(context)?;
        let paged_gqa = PagedGqaOp::new(context)?;
        let attention_output = AttentionOutputOp::new(context)?;
        let swiglu = DenseFp8SwiGluOp::new(context)?;
        let down = DenseFp8DownOp::new(context)?;

        arena.copy_from_host(
            &stream,
            regions.input_norm,
            &post.input_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(&stream, regions.qkv_weight_codes, &qkv.weight_e4m3)?;
        arena.copy_from_host(
            &stream,
            regions.qkv_weight_scales,
            &little_endian_words(&qkv.scale_bf16)?,
        )?;
        arena.copy_from_host(
            &stream,
            regions.query_norm,
            &post.query_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.key_norm,
            &post.key_norm.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_codes,
            post.output_weight.codes(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.output_weight_scales,
            &post.output_scale.words().collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.post_attention_norm,
            &post.post_attention_norm.words().collect::<Vec<_>>(),
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
            regions.block_tables,
            &(0..(MAX_BATCH * TABLE_STRIDE) as u32).collect::<Vec<_>>(),
        )?;
        arena.copy_from_host(
            &stream,
            regions.table_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let key_cache_scale = bf16_to_f32(post.key_cache_scale_bf16);
        let value_cache_scale = bf16_to_f32(post.value_cache_scale_bf16);
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
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            swiglu: &swiglu,
            down: &down,
            gate_up_maps: &gate_up_maps,
            down_maps: &down_maps,
        };
        let scales = CacheScales {
            key: key_cache_scale,
            value: value_cache_scale,
        };
        let graphs = capture_decode_routes(&stream, ops, pointers, scales)?;
        let prefill_graphs = capture_prefill_routes(&stream, ops, pointers, scales)?;

        Ok(Self {
            graphs,
            prefill_graphs,
            gate_up_maps,
            down_maps,
            arena,
            _norm: norm,
            _qkv: qkv_op,
            _qk_prepare: qk_prepare,
            _paged_gqa: paged_gqa,
            _attention_output: attention_output,
            _swiglu: swiglu,
            _down: down,
            snapshot,
            context: context.clone(),
            layout,
            base_address,
            key_cache_scale,
            value_cache_scale,
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
        let expected = product("full-attention input elements", rows, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "full-attention input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;
        Ok(())
    }

    /// Loads a contiguous from-empty causal prefill tile and its 32 MRoPE pairs.
    pub fn load_prefill_state(
        &self,
        stream: &CudaStream,
        tokens: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> EngineResult<()> {
        if prefill_index(tokens).is_none() {
            return Err(EngineError::route(format!(
                "full-attention prefill tokens {tokens} are outside 32,64,128,1024"
            )));
        }
        let rotary_values = product("full-attention prefill rotary values", tokens, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "full-attention prefill rotary planes must each have {rotary_values} values for T={tokens}"
            )));
        }
        if tokens > PREFILL_CONTEXT_CAPACITY {
            return Err(EngineError::route(format!(
                "full-attention prefill T={tokens} exceeds the {PREFILL_CONTEXT_CAPACITY}-token shared cache"
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

    /// Updates the active positions, causal lengths, and 32 MRoPE pairs per token.
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
                "full-attention positions have {} values, expected {batch}",
                positions.len()
            )));
        }
        let rotary_values = product("full-attention rotary values", batch, ROTARY_PAIRS)?;
        if rope_cos.len() != rotary_values || rope_sin.len() != rotary_values {
            return Err(EngineError::layout(format!(
                "full-attention rotary planes must each have {rotary_values} values for B={batch}"
            )));
        }
        let lengths = positions
            .iter()
            .map(|&position| {
                if position as usize >= CONTEXT_CAPACITY {
                    return Err(EngineError::route(format!(
                        "full-attention cache position {position} exceeds the {}-token slot capacity",
                        CONTEXT_CAPACITY
                    )));
                }
                position
                    .checked_add(1)
                    .ok_or_else(|| EngineError::route("full-attention cache length overflows"))
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

    /// Replaces both complete represented E4M3 cache planes.
    pub fn load_cache(
        &self,
        stream: &CudaStream,
        key_pages: &[u8],
        value_pages: &[u8],
    ) -> EngineResult<()> {
        let regions = self.layout.regions();
        if key_pages.len() != regions.key_pages.len()
            || value_pages.len() != regions.value_pages.len()
        {
            return Err(EngineError::layout(format!(
                "full-attention cache planes must each have {} bytes",
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

    /// Replays the immutable graph for one exact decode or prefill width.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        // SAFETY: this FullAttentionLayerProgram owns every captured allocation
        // (arena, TMA maps, op modules) for its whole life and drops the graphs first.
        unsafe { self.graph(rows)?.launch(stream) }?;
        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        require_rows(rows)?;
        let values = product("full-attention output elements", rows, A::HIDDEN)?;
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

    /// Exact represented key/value cache bytes.
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

    /// Weights, cache, and working planes without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.layout.owner_bytes()
    }

    /// Fixed short-context capacity of each initial slot.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Shared-cache capacity available to exact from-empty prefill routes.
    pub const fn prefill_context_capacity(&self) -> usize {
        self.layout.prefill_context_capacity()
    }

    /// Exact source BF16 key-cache scale promoted to FP32 for kernel arguments.
    pub const fn key_cache_scale(&self) -> f32 {
        self.key_cache_scale
    }

    /// Exact source BF16 value-cache scale promoted to FP32 for kernel arguments.
    pub const fn value_cache_scale(&self) -> f32 {
        self.value_cache_scale
    }

    /// Largest admitted exact batch.
    pub const fn batch_capacity(&self) -> usize {
        MAX_BATCH
    }

    /// Largest exact row route owned by the layer.
    pub const fn row_capacity(&self) -> usize {
        MAX_ROWS
    }

    /// Exact bytes in the four address-bound MLP tensor maps.
    pub const fn descriptor_bytes(&self) -> usize {
        DenseFp8SwiGluTmaMaps::BYTE_LEN + DenseFp8DownTmaMaps::BYTE_LEN
    }

    /// Checked owner layout.
    pub const fn layout(&self) -> &FullAttentionLayerLayout {
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
                "full-attention layer row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
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
            self.scales(),
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
                "repeated full-attention graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        let scales = self.scales();
        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, rows, ops, pointers, scales)?;
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

        Ok(addresses)
    }

    #[cfg(feature = "qualification")]
    /// Copies every opaque address-bound tensor map.
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
    /// Fills every non-cache mutable seam with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.mixer_normalized,
            regions.qkv,
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
            regions.output_activation_codes,
            regions.gate_up_activation_codes,
            regions.down_activation_codes,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [regions.query, regions.attention] {
            self.arena.fill(stream, region, byte)?;
        }
        self.arena.fill(stream, regions.macro_partials, byte)?;
        for region in [
            regions.qkv_activation_scales,
            regions.output_activation_scales,
            regions.gate_up_activation_scales,
            regions.down_activation_scales,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every mutable seam, including complete persistent cache planes.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<FullAttentionLayerObservables> {
        let regions = self.layout.regions();
        Ok(FullAttentionLayerObservables {
            residual_input: self.arena.copy_to_host(stream, regions.residual_input)?,
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            qkv_activation_codes: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_codes)?,
            qkv_activation_scales: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_scales)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            rope_cos: self.arena.copy_to_host(stream, regions.rope_cos)?,
            rope_sin: self.arena.copy_to_host(stream, regions.rope_sin)?,
            block_tables: self.arena.copy_to_host(stream, regions.block_tables)?,
            table_rows: self.arena.copy_to_host(stream, regions.table_rows)?,
            cache_positions: self.arena.copy_to_host(stream, regions.cache_positions)?,
            lengths: self.arena.copy_to_host(stream, regions.lengths)?,
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
            macro_partials: self.arena.copy_to_host(stream, regions.macro_partials)?,
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
            qkv: &self._qkv,
            qk_prepare: &self._qk_prepare,
            paged_gqa: &self._paged_gqa,
            attention_output: &self._attention_output,
            swiglu: &self._swiglu,
            down: &self._down,
            gate_up_maps: &self.gate_up_maps,
            down_maps: &self.down_maps,
        }
    }

    #[cfg(feature = "qualification")]
    const fn scales(&self) -> CacheScales {
        CacheScales {
            key: self.key_cache_scale,
            value: self.value_cache_scale,
        }
    }
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct FullAttentionLayerObservables {
    /// Input residual rows loaded through the public owner boundary.
    pub residual_input: Vec<u16>,
    /// Pre-attention normalized residual rows.
    pub mixer_normalized: Vec<u16>,
    /// QKV dynamic E4M3 activation codes.
    pub qkv_activation_codes: Vec<u8>,
    /// QKV dynamic FP32 activation scales.
    pub qkv_activation_scales: Vec<f32>,
    /// Fused query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Loaded MRoPE cosine values.
    pub rope_cos: Vec<f32>,
    /// Loaded MRoPE sine values.
    pub rope_sin: Vec<f32>,
    /// Complete shared physical-page inventory.
    pub block_tables: Vec<u32>,
    /// Per-row page-table selections.
    pub table_rows: Vec<u32>,
    /// Per-row cache append positions.
    pub cache_positions: Vec<u32>,
    /// Per-row causal attention lengths.
    pub lengths: Vec<u32>,
    /// Loaded prefill MRoPE cosine values.
    pub prefill_rope_cos: Vec<f32>,
    /// Loaded prefill MRoPE sine values.
    pub prefill_rope_sin: Vec<f32>,
    /// Shared page-table row selected by every prefill token.
    pub prefill_table_rows: Vec<u32>,
    /// Contiguous from-empty prefill append positions.
    pub prefill_cache_positions: Vec<u32>,
    /// Contiguous from-empty prefill causal lengths.
    pub prefill_lengths: Vec<u32>,
    /// Prepared FP32 query heads.
    pub query: Vec<f32>,
    /// Complete represented E4M3 key cache.
    pub key_pages: Vec<u8>,
    /// Complete represented E4M3 value cache.
    pub value_pages: Vec<u8>,
    /// FP32 paged-GQA output, gated in place by attention output.
    pub attention: Vec<f32>,
    /// Maximum FP32 macro-prefill partial workspace.
    pub macro_partials: Vec<f32>,
    /// Attention-output dynamic E4M3 activation codes.
    pub output_activation_codes: Vec<u8>,
    /// Attention-output dynamic FP32 activation scales.
    pub output_activation_scales: Vec<f32>,
    /// BF16 attention output-projection branch.
    pub mixer_branch: Vec<u16>,
    /// Residual after attention.
    pub mixer_residual: Vec<u16>,
    /// Pre-MLP normalized rows.
    pub mlp_normalized: Vec<u16>,
    /// Gate/up dynamic E4M3 activation codes.
    pub gate_up_activation_codes: Vec<u8>,
    /// Gate/up dynamic FP32 activation scales.
    pub gate_up_activation_scales: Vec<f32>,
    /// Fused BF16 SwiGLU rows.
    pub swiglu: Vec<u16>,
    /// Down dynamic E4M3 activation codes.
    pub down_activation_codes: Vec<u8>,
    /// Down dynamic FP32 activation scales.
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
    qkv: &'a FullAttentionQkvOp<A>,
    qk_prepare: &'a AttentionQkPrepareOp<A>,
    paged_gqa: &'a PagedGqaOp<A>,
    attention_output: &'a AttentionOutputOp<A>,
    swiglu: &'a DenseFp8SwiGluOp<A>,
    down: &'a DenseFp8DownOp<A>,
    gate_up_maps: &'a DenseFp8SwiGluTmaMaps,
    down_maps: &'a DenseFp8DownTmaMaps,
}

#[derive(Clone, Copy)]
struct CacheScales {
    key: f32,
    value: f32,
}

fn capture_decode_routes<A: Sm120Arch>(
    stream: &CudaStream,
    ops: Ops<'_, A>,
    pointers: Pointers,
    scales: CacheScales,
) -> EngineResult<[CudaGraph; MAX_BATCH]> {
    let mut graphs = Vec::with_capacity(MAX_BATCH);
    for batch in 1..=MAX_BATCH {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, batch, ops, pointers, scales)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("full-attention layer graph inventory has wrong cardinality")
    })
}

fn capture_prefill_routes<A: Sm120Arch>(
    stream: &CudaStream,
    ops: Ops<'_, A>,
    pointers: Pointers,
    scales: CacheScales,
) -> EngineResult<[CudaGraph; 4]> {
    let mut graphs = Vec::with_capacity(4);
    for rows in [32, 64, 128, MAX_ROWS] {
        graphs.push(CudaGraph::capture(stream, || {
            launch_route(stream, rows, ops, pointers, scales)
        })?);
    }
    graphs.try_into().map_err(|_| {
        EngineError::layout("full-attention layer prefill graph inventory has wrong cardinality")
    })
}

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_, A>,
    pointers: Pointers,
    scales: CacheScales,
) -> GpuResult<()> {
    let (rope_cos, rope_sin, table_rows, cache_positions, lengths, table_stride) =
        if rows <= MAX_BATCH {
            (
                pointers.rope_cos,
                pointers.rope_sin,
                pointers.table_rows,
                pointers.cache_positions,
                pointers.lengths,
                TABLE_STRIDE,
            )
        } else {
            (
                pointers.prefill_rope_cos,
                pointers.prefill_rope_sin,
                pointers.prefill_table_rows,
                pointers.prefill_cache_positions,
                pointers.prefill_lengths,
                PREFILL_TABLE_STRIDE,
            )
        };
    // SAFETY: the one arena owns aligned, disjoint MAX_ROWS planes. Decode
    // uses three-page slot rows; prefill uses the complete shared page row.
    // Exact dispatch bounds every kernel to `rows` active values.
    unsafe {
        ops.norm.launch_plain(
            stream,
            rows,
            pointers.residual_input,
            pointers.input_norm,
            pointers.mixer_normalized,
        )?;
        ops.qkv.launch(
            stream,
            rows,
            pointers.mixer_normalized,
            pointers.qkv_activation_codes,
            pointers.qkv_activation_scales,
            pointers.qkv_weight_codes,
            pointers.qkv_weight_scales,
            pointers.qkv,
        )?;
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
            scales.key,
            scales.value,
        )?;
        if rows <= MAX_BATCH {
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
                scales.key,
                scales.value,
            )?;
        } else if rows < MAX_ROWS {
            ops.paged_gqa.launch_prefill_shared(
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
                scales.key,
                scales.value,
            )?;
        } else {
            // P4 bounds a T=1024 causal scan to at most 256 keys per partition
            // while exposing 3,072 independent producer CTAs before reduction.
            ops.paged_gqa.launch_prefill_macro(
                stream,
                4,
                pointers.query,
                pointers.key_pages,
                pointers.value_pages,
                pointers.block_tables,
                table_rows,
                table_stride,
                lengths,
                pointers.macro_partials,
                pointers.attention,
                scales.key,
                scales.value,
            )?;
        }
        ops.attention_output.launch(
            stream,
            rows,
            pointers.attention,
            pointers.qkv,
            pointers.output_activation_codes,
            pointers.output_activation_scales,
            pointers.output_weight_codes,
            pointers.output_weight_scales,
            pointers.mixer_branch,
        )?;
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
            // T=1024 amortizes both stable tensor-map descriptors across the macro tile.
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

fn require_batch(batch: usize) -> EngineResult<()> {
    if !(1..=MAX_BATCH).contains(&batch) {
        return Err(EngineError::route(format!(
            "full-attention layer batch {batch} is outside 1..={MAX_BATCH}"
        )));
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
            "full-attention layer row count {rows} is outside 1..={MAX_BATCH},32,64,128,1024"
        )));
    }

    Ok(())
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(EngineError::layout(
            "BF16 source plane has an odd byte length",
        ));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, ROTARY_PAIRS, require_batch, require_rows};

    #[test]
    fn exact_batch_table_rejects_every_uncompiled_route() {
        for (batch, admitted) in [(0, false), (1, true), (8, true), (9, false), (16, false)] {
            assert_eq!(require_batch(batch).is_ok(), admitted, "batch={batch}");
        }
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(ROTARY_PAIRS, 32);
    }

    #[test]
    fn exact_row_table_rejects_every_boundary_neighbor() {
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024] {
            require_rows(rows).unwrap();
        }
        for rows in [0, 9, 31, 33, 63, 65, 127, 129, 1_023, 1_025] {
            assert!(require_rows(rows).is_err(), "rows={rows}");
        }
    }
}

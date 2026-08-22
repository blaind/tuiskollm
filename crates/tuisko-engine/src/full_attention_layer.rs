//! Resident source-backed dense-FP8 full-attention decoder layer.

use crate::full_attention_layer_layout::{
    CONTEXT_CAPACITY, FullAttentionLayerRegions, TABLE_STRIDE,
};
use crate::{EngineError, EngineResult, FullAttentionLayerLayout, MAX_BATCH};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    AttentionOutputOp, AttentionQkPrepareOp, DenseFp8DownOp, DenseFp8SwiGluOp, FullAttentionQkvOp,
    PagedGqaOp, ResidualNormOp, Sm120Arch,
};
use tuisko_model::{
    CheckpointSnapshot, DenseFp8MlpBindings, FullAttentionPostBindings, FullAttentionQkvBindings,
    Qwen38_27B,
};

const ROTARY_PAIRS: usize = 32;

/// One late full-attention layer with immutable exact-batch graph routes.
pub struct FullAttentionLayerProgram<A: Sm120Arch = Qwen38_27B> {
    // Drop graphs before the arena and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
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
    query: *mut f32,
    key_pages: *mut u8,
    value_pages: *mut u8,
    attention: *mut f32,
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
            query: arena.address(regions.query)?,
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,
            attention: arena.address(regions.attention)?,
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
            self.query.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
            self.attention.addr(),
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
    /// Loads one admitted source layer and captures exact `B=1..=8` decode routes.
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
        let base_address = arena.base_address();
        let ops = Ops {
            norm: &norm,
            qkv: &qkv_op,
            qk_prepare: &qk_prepare,
            paged_gqa: &paged_gqa,
            attention_output: &attention_output,
            swiglu: &swiglu,
            down: &down,
        };
        let scales = CacheScales {
            key: key_cache_scale,
            value: value_cache_scale,
        };
        let graphs = capture_routes(&stream, ops, pointers, scales)?;

        Ok(Self {
            graphs,
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

    /// Uploads exactly `batch` BF16 residual rows into stable input storage.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        batch: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        require_batch(batch)?;
        let expected = product("full-attention input elements", batch, A::HIDDEN)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "full-attention input has {} values, expected {expected} for B={batch}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;
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

    /// Replays the immutable graph for one exact batch.
    pub fn replay(&self, stream: &CudaStream, batch: usize) -> EngineResult<()> {
        require_batch(batch)?;
        self.graphs[batch - 1].launch(stream)?;
        Ok(())
    }

    /// Reads active BF16 residual output rows.
    pub fn read_residual(&self, stream: &CudaStream, batch: usize) -> EngineResult<Vec<u16>> {
        require_batch(batch)?;
        let values = product("full-attention output elements", batch, A::HIDDEN)?;
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

    /// Fixed short-context capacity of each initial slot.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
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

    /// Checked owner layout.
    pub const fn layout(&self) -> &FullAttentionLayerLayout {
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
        launch_route(
            stream,
            batch,
            self.ops(),
            Pointers::bind(&self.arena, self.layout.regions())?,
            self.scales(),
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
                "repeated full-attention graph requires at least one operation",
            ));
        }
        let pointers = Pointers::bind(&self.arena, self.layout.regions())?;
        let ops = self.ops();
        let scales = self.scales();
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
            mixer_normalized: self.arena.copy_to_host(stream, regions.mixer_normalized)?,
            qkv_activation_codes: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_codes)?,
            qkv_activation_scales: self
                .arena
                .copy_to_host(stream, regions.qkv_activation_scales)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
            key_pages: self.arena.copy_to_host(stream, regions.key_pages)?,
            value_pages: self.arena.copy_to_host(stream, regions.value_pages)?,
            attention: self.arena.copy_to_host(stream, regions.attention)?,
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
    /// Pre-attention normalized residual rows.
    pub mixer_normalized: Vec<u16>,
    /// QKV dynamic E4M3 activation codes.
    pub qkv_activation_codes: Vec<u8>,
    /// QKV dynamic FP32 activation scales.
    pub qkv_activation_scales: Vec<f32>,
    /// Fused query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Prepared FP32 query heads.
    pub query: Vec<f32>,
    /// Complete represented E4M3 key cache.
    pub key_pages: Vec<u8>,
    /// Complete represented E4M3 value cache.
    pub value_pages: Vec<u8>,
    /// FP32 paged-GQA output, gated in place by attention output.
    pub attention: Vec<f32>,
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
}

#[derive(Clone, Copy)]
struct CacheScales {
    key: f32,
    value: f32,
}

fn capture_routes<A: Sm120Arch>(
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

fn launch_route<A: Sm120Arch>(
    stream: &CudaStream,
    batch: usize,
    ops: Ops<'_, A>,
    pointers: Pointers,
    scales: CacheScales,
) -> GpuResult<()> {
    // SAFETY: the one arena owns aligned, disjoint maximum-batch planes. Fixed
    // three-page slot tables cover every admitted position, while exact-B
    // dispatch bounds each kernel to active rows.
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
            pointers.qkv_activation_scales,
            pointers.qkv_weight_codes,
            pointers.qkv_weight_scales,
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
            TABLE_STRIDE,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            scales.key,
            scales.value,
        )?;
        ops.paged_gqa.launch(
            stream,
            batch,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            pointers.block_tables,
            pointers.table_rows,
            TABLE_STRIDE,
            pointers.lengths,
            pointers.attention,
            scales.key,
            scales.value,
        )?;
        ops.attention_output.launch(
            stream,
            batch,
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
            batch,
            pointers.residual_input,
            pointers.mixer_branch,
            pointers.post_attention_norm,
            pointers.mixer_residual,
            pointers.mlp_normalized,
        )?;
        ops.swiglu.launch(
            stream,
            batch,
            pointers.mlp_normalized,
            pointers.gate_up_activation_codes,
            pointers.gate_up_activation_scales,
            pointers.gate_up_weight_codes,
            pointers.gate_up_weight_scales,
            pointers.swiglu,
        )?;
        ops.down.launch(
            stream,
            batch,
            pointers.swiglu,
            pointers.down_activation_codes,
            pointers.down_activation_scales,
            pointers.down_weight_codes,
            pointers.down_weight_scales,
            pointers.mlp_branch,
        )?;
        ops.norm.launch_residual(
            stream,
            batch,
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
    use super::{MAX_BATCH, ROTARY_PAIRS, require_batch};

    #[test]
    fn exact_batch_table_rejects_every_uncompiled_route() {
        for (batch, admitted) in [(0, false), (1, true), (8, true), (9, false), (16, false)] {
            assert_eq!(require_batch(batch).is_ok(), admitted, "batch={batch}");
        }
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(ROTARY_PAIRS, 32);
    }
}

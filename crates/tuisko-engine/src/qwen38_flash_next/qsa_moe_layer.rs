//! Qwen3.8-Flash-Next QSA/MoE decoder layer.
//!
//! Dense GQA is exact through 2,051 visible keys and refused above that ceiling. The owner keeps
//! indexer weights resident for exact checkpoint accounting and captures the same twelve routes
//! as the GDN/MoE layer.

use crate::common::graph::{capture_batch_graphs, capture_route_graphs};
use crate::common::math::product;
use crate::qwen38_flash_next::expert_pool_layout::{
    Qwen38FlashNextExpertPoolLayout, Qwen38FlashNextExpertPoolRegions,
};
use crate::qwen38_flash_next::gdn_moe_layer_layout::QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;
use crate::qwen38_flash_next::layer_route::{
    QWEN38_FLASH_NEXT_PREFILL_ROWS, Qwen38FlashNextRowRoute, qwen38_flash_next_row_route,
};
use crate::qwen38_flash_next::layer_upload::{
    HyperConnectionRegions, MoeRegions, bf16_words, upload_expert_pool, upload_hyper_connection,
    upload_moe,
};
use crate::qwen38_flash_next::qsa_moe_layer_layout::{
    QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES, QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE,
    Qwen38FlashNextQsaMoeLayerRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen38FlashNextQsaMoeLayerLayout};
use std::sync::Arc;
#[cfg(feature = "qualification")]
use tuisko_gpu::PinnedHostBuffer;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult};
use tuisko_kernels_sm120::{
    Qwen38FlashNextAttentionGateOp, Qwen38FlashNextAttentionQkPrepareOp,
    Qwen38FlashNextBlockOutputProjectionOp, Qwen38FlashNextExpertDispatch,
    Qwen38FlashNextHyperConnectionOp, Qwen38FlashNextMoeExpertsOp, Qwen38FlashNextMoeRouterOp,
    Qwen38FlashNextPagedGqaOp, Qwen38FlashNextQsaQkvProjectionOp,
    qwen38_flash_next_expert_slot_plane,
};
use tuisko_model::{
    CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextLayerHyperConnections,
    Qwen38FlashNextMoeBindings, Qwen38FlashNextSparseAttentionBindings,
};

type A = Qwen38FlashNext;

/// Represented E4M3 key-plane scale this target's cache is qualified at.
const KEY_CACHE_SCALE: f32 = 0.031_25;

/// Represented E4M3 value-plane scale this target's cache is qualified at.
const VALUE_CACHE_SCALE: f32 = 0.062_5;

/// One Qwen3.8-Flash-Next QSA/MoE decoder layer with immutable exact decode and prefill graphs.
pub struct Qwen38FlashNextQsaMoeLayerProgram {
    // Drop graphs before the arenas and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; QWEN38_FLASH_NEXT_PREFILL_ROWS.len()],
    arena: DeviceArena,
    pool_arena: DeviceArena,
    _hyper: Qwen38FlashNextHyperConnectionOp,
    _qkv: Qwen38FlashNextQsaQkvProjectionOp,
    _prepare: Qwen38FlashNextAttentionQkPrepareOp,
    _attention: Qwen38FlashNextPagedGqaOp,
    _gate: Qwen38FlashNextAttentionGateOp,
    _block_output: Qwen38FlashNextBlockOutputProjectionOp,
    _router: Qwen38FlashNextMoeRouterOp,
    _experts: Qwen38FlashNextMoeExpertsOp,
    snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    context: Arc<CudaContext>,
    layout: Qwen38FlashNextQsaMoeLayerLayout,
    pool_layout: Qwen38FlashNextExpertPoolLayout,
    base_address: u64,
    pool_base_address: u64,
    layer: usize,
}

/// Every device address one launch of this layer reads or writes.
#[derive(Clone, Copy)]
struct Pointers {
    residual_input: *const u16,
    attention_residual: *mut u16,
    residual_output: *mut u16,

    attention_hc_norm: *const u16,
    attention_hc_down: *const u16,
    attention_hc_up: *const u16,
    attention_hc_inject: *const u16,
    mlp_hc_norm: *const u16,
    mlp_hc_down: *const u16,
    mlp_hc_up: *const u16,
    mlp_hc_inject: *const u16,

    hc_normalized: *mut u16,
    hc_low_rank: *mut u16,
    hc_mixed: *mut u16,
    hc_write_gate: *mut u16,

    qkv_weight: *const u16,
    output_weight: *const u16,
    query_norm: *const u16,
    key_norm: *const u16,

    qkv: *mut u16,
    query: *mut f32,
    attention: *mut f32,
    attention_gated: *mut u16,
    rope_cos: *const f32,
    rope_sin: *const f32,
    table_rows: *const u32,
    cache_positions: *const u32,
    lengths: *const u32,
    block_tables: *const u32,
    key_pages: *mut u8,
    value_pages: *mut u8,

    router_weight: *const u16,
    expert_weight_scales_2: *const f32,
    shared_gate_weight: *const u16,
    shared_up_weight: *const u16,
    shared_down_weight: *const u16,
    shared_gate_logit_weight: *const u16,

    router_logits: *mut u16,
    expert_indices: *mut u16,
    routing_weights: *mut u16,
    routed_intermediate: *mut u16,
    routed_output: *mut u16,
    shared_intermediate: *mut u16,
    shared_output: *mut u16,
    shared_gate_logit: *mut u16,

    block_output: *mut u16,

    slot_table: *const u32,
    slot_pool: *const u8,
}

impl Pointers {
    fn bind(
        arena: &DeviceArena,
        pool: &DeviceArena,
        regions: Qwen38FlashNextQsaMoeLayerRegions,
        pool_regions: Qwen38FlashNextExpertPoolRegions,
    ) -> GpuResult<Self> {
        Ok(Self {
            residual_input: arena.address(regions.residual_input)?.cast_const(),
            attention_residual: arena.address(regions.attention_residual)?,
            residual_output: arena.address(regions.residual_output)?,

            attention_hc_norm: arena.address(regions.attention_hc_norm)?.cast_const(),
            attention_hc_down: arena.address(regions.attention_hc_down)?.cast_const(),
            attention_hc_up: arena.address(regions.attention_hc_up)?.cast_const(),
            attention_hc_inject: arena.address(regions.attention_hc_inject)?.cast_const(),
            mlp_hc_norm: arena.address(regions.mlp_hc_norm)?.cast_const(),
            mlp_hc_down: arena.address(regions.mlp_hc_down)?.cast_const(),
            mlp_hc_up: arena.address(regions.mlp_hc_up)?.cast_const(),
            mlp_hc_inject: arena.address(regions.mlp_hc_inject)?.cast_const(),

            hc_normalized: arena.address(regions.hc_normalized)?,
            hc_low_rank: arena.address(regions.hc_low_rank)?,
            hc_mixed: arena.address(regions.hc_mixed)?,
            hc_write_gate: arena.address(regions.hc_write_gate)?,

            qkv_weight: arena.address(regions.qkv_weight)?.cast_const(),
            output_weight: arena.address(regions.output_weight)?.cast_const(),
            query_norm: arena.address(regions.query_norm)?.cast_const(),
            key_norm: arena.address(regions.key_norm)?.cast_const(),

            qkv: arena.address(regions.qkv)?,
            query: arena.address(regions.query)?,
            attention: arena.address(regions.attention)?,
            attention_gated: arena.address(regions.attention_gated)?,
            rope_cos: arena.address(regions.rope_cos)?.cast_const(),
            rope_sin: arena.address(regions.rope_sin)?.cast_const(),
            table_rows: arena.address(regions.table_rows)?.cast_const(),
            cache_positions: arena.address(regions.cache_positions)?.cast_const(),
            lengths: arena.address(regions.lengths)?.cast_const(),
            block_tables: arena.address(regions.block_tables)?.cast_const(),
            key_pages: arena.address(regions.key_pages)?,
            value_pages: arena.address(regions.value_pages)?,

            router_weight: arena.address(regions.router_weight)?.cast_const(),
            expert_weight_scales_2: arena.address(regions.expert_weight_scales_2)?.cast_const(),
            shared_gate_weight: arena.address(regions.shared_gate_weight)?.cast_const(),
            shared_up_weight: arena.address(regions.shared_up_weight)?.cast_const(),
            shared_down_weight: arena.address(regions.shared_down_weight)?.cast_const(),
            shared_gate_logit_weight: arena
                .address(regions.shared_gate_logit_weight)?
                .cast_const(),

            router_logits: arena.address(regions.router_logits)?,
            expert_indices: arena.address(regions.expert_indices)?,
            routing_weights: arena.address(regions.routing_weights)?,
            routed_intermediate: arena.address(regions.routed_intermediate)?,
            routed_output: arena.address(regions.routed_output)?,
            shared_intermediate: arena.address(regions.shared_intermediate)?,
            shared_output: arena.address(regions.shared_output)?,
            shared_gate_logit: arena.address(regions.shared_gate_logit)?,

            block_output: arena.address(regions.block_output)?,

            slot_table: pool.address(pool_regions.slot_table)?.cast_const(),
            slot_pool: pool.address(pool_regions.slot_pool)?.cast_const(),
        })
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> Vec<usize> {
        vec![
            self.residual_input.addr(),
            self.attention_residual.addr(),
            self.residual_output.addr(),
            self.attention_hc_norm.addr(),
            self.attention_hc_down.addr(),
            self.attention_hc_up.addr(),
            self.attention_hc_inject.addr(),
            self.mlp_hc_norm.addr(),
            self.mlp_hc_down.addr(),
            self.mlp_hc_up.addr(),
            self.mlp_hc_inject.addr(),
            self.hc_normalized.addr(),
            self.hc_low_rank.addr(),
            self.hc_mixed.addr(),
            self.hc_write_gate.addr(),
            self.qkv_weight.addr(),
            self.output_weight.addr(),
            self.query_norm.addr(),
            self.key_norm.addr(),
            self.qkv.addr(),
            self.query.addr(),
            self.attention.addr(),
            self.attention_gated.addr(),
            self.rope_cos.addr(),
            self.rope_sin.addr(),
            self.table_rows.addr(),
            self.cache_positions.addr(),
            self.lengths.addr(),
            self.block_tables.addr(),
            self.key_pages.addr(),
            self.value_pages.addr(),
            self.router_weight.addr(),
            self.expert_weight_scales_2.addr(),
            self.shared_gate_weight.addr(),
            self.shared_up_weight.addr(),
            self.shared_down_weight.addr(),
            self.shared_gate_logit_weight.addr(),
            self.router_logits.addr(),
            self.expert_indices.addr(),
            self.routing_weights.addr(),
            self.routed_intermediate.addr(),
            self.routed_output.addr(),
            self.shared_intermediate.addr(),
            self.shared_output.addr(),
            self.shared_gate_logit.addr(),
            self.block_output.addr(),
            self.slot_table.addr(),
            self.slot_pool.addr(),
        ]
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    hyper: &'a Qwen38FlashNextHyperConnectionOp,
    qkv: &'a Qwen38FlashNextQsaQkvProjectionOp,
    prepare: &'a Qwen38FlashNextAttentionQkPrepareOp,
    attention: &'a Qwen38FlashNextPagedGqaOp,
    gate: &'a Qwen38FlashNextAttentionGateOp,
    block_output: &'a Qwen38FlashNextBlockOutputProjectionOp,
    router: &'a Qwen38FlashNextMoeRouterOp,
    experts: &'a Qwen38FlashNextMoeExpertsOp,
}

impl Qwen38FlashNextQsaMoeLayerProgram {
    /// Loads one source layer and captures every admitted decode and prefill route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let qsa = Qwen38FlashNextSparseAttentionBindings::bind(snapshot.as_ref(), layer)?
            .materialize()?;
        let moe = Qwen38FlashNextMoeBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let hc =
            Qwen38FlashNextLayerHyperConnections::bind(snapshot.as_ref(), layer)?.materialize()?;

        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(layer)?;
        let pool_layout = Qwen38FlashNextExpertPoolLayout::resident()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let pool_arena = DeviceArena::zeroed(&stream, pool_layout.builder())?;

        let hyper = Qwen38FlashNextHyperConnectionOp::new(context)?;
        let qkv = Qwen38FlashNextQsaQkvProjectionOp::new(context)?;
        let prepare = Qwen38FlashNextAttentionQkPrepareOp::new(context)?;
        let attention = Qwen38FlashNextPagedGqaOp::new(context)?;
        let gate = Qwen38FlashNextAttentionGateOp::new(context)?;
        let block_output = Qwen38FlashNextBlockOutputProjectionOp::new(context)?;
        let router = Qwen38FlashNextMoeRouterOp::new(context)?;
        let experts = Qwen38FlashNextMoeExpertsOp::new(context)?;

        for (bindings, hc_regions) in [
            (
                hc.attention,
                HyperConnectionRegions {
                    norm: regions.attention_hc_norm,
                    down: regions.attention_hc_down,
                    up: regions.attention_hc_up,
                    inject: regions.attention_hc_inject,
                },
            ),
            (
                hc.mlp,
                HyperConnectionRegions {
                    norm: regions.mlp_hc_norm,
                    down: regions.mlp_hc_down,
                    up: regions.mlp_hc_up,
                    inject: regions.mlp_hc_inject,
                },
            ),
        ] {
            upload_hyper_connection(&arena, &stream, hc_regions, bindings)?;
        }
        arena.copy_from_host(
            &stream,
            regions.qkv_weight,
            &bf16_words(&qsa.qkv_weight_bf16)?,
        )?;
        for (region, view) in [
            (regions.output_weight, qsa.output_weight),
            (regions.indexer_qk_weight, qsa.indexer.qk_weight),
        ] {
            arena.copy_from_host(&stream, region, &view.words().collect::<Vec<_>>())?;
        }
        for (region, view) in [
            (regions.query_norm, qsa.query_norm),
            (regions.key_norm, qsa.key_norm),
            (regions.indexer_query_norm, qsa.indexer.query_norm),
            (regions.indexer_key_norm, qsa.indexer.key_norm),
        ] {
            arena.copy_from_host(&stream, region, &view.words().collect::<Vec<_>>())?;
        }
        upload_moe(
            &arena,
            &stream,
            MoeRegions {
                router_weight: regions.router_weight,
                shared_gate_weight: regions.shared_gate_weight,
                shared_up_weight: regions.shared_up_weight,
                shared_down_weight: regions.shared_down_weight,
                shared_gate_logit_weight: regions.shared_gate_logit_weight,
                expert_weight_scales_2: regions.expert_weight_scales_2,
            },
            &moe,
        )?;
        upload_expert_pool(&pool_arena, &stream, pool_layout.regions(), &moe.experts)?;

        // One page-table row per slot, ascending, so slot `s` owns pages
        // `[s * TABLE_STRIDE, (s + 1) * TABLE_STRIDE)`.
        arena.copy_from_host(
            &stream,
            regions.block_tables,
            &(0..QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES as u32).collect::<Vec<_>>(),
        )?;
        stream.synchronize().map_err(GpuError::from)?;

        let pointers = Pointers::bind(&arena, &pool_arena, regions, pool_layout.regions())?;
        let base_address = arena.base_address();
        let pool_base_address = pool_arena.base_address();
        let ops = Ops {
            hyper: &hyper,
            qkv: &qkv,
            prepare: &prepare,
            attention: &attention,
            gate: &gate,
            block_output: &block_output,
            router: &router,
            experts: &experts,
        };
        let graphs = capture_batch_graphs(
            &stream,
            "Qwen3.8-Flash-Next QSA/MoE decode graph inventory has wrong cardinality",
            |rows| launch_route(&stream, rows, ops, pointers),
        )?;
        let prefill_graphs = capture_route_graphs(
            &stream,
            QWEN38_FLASH_NEXT_PREFILL_ROWS,
            "Qwen3.8-Flash-Next QSA/MoE prefill graph inventory has wrong cardinality",
            |rows| launch_route(&stream, rows, ops, pointers),
        )?;

        Ok(Self {
            graphs,
            prefill_graphs,
            arena,
            pool_arena,
            _hyper: hyper,
            _qkv: qkv,
            _prepare: prepare,
            _attention: attention,
            _gate: gate,
            _block_output: block_output,
            _router: router,
            _experts: experts,
            snapshot,
            context: context.clone(),
            layout,
            pool_layout,
            base_address,
            pool_base_address,
            layer,
        })
    }

    /// Uploads one exact decode or prefill width into stable input storage.
    pub fn load_residual(
        &self,
        stream: &CudaStream,
        rows: usize,
        values: &[u16],
    ) -> EngineResult<()> {
        qwen38_flash_next_row_route(rows)?;
        let expected = product("Qwen3.8-Flash-Next layer input", rows, A::HC_WIDTH)?;
        if values.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next layer input has {} values, expected {expected} for rows={rows}",
                values.len()
            )));
        }
        self.arena
            .copy_prefix_from_host(stream, self.layout.regions().residual_input, values)?;

        Ok(())
    }

    /// Uploads one round's rotary tables and page metadata, refusing an inadmissible round.
    ///
    /// Every visible length is checked against the proven dense-equivalence ceiling and against
    /// this owner's own page capacity before any of the round reaches the device.
    pub fn load_round(
        &self,
        stream: &CudaStream,
        rows: usize,
        round: Qwen38FlashNextQsaRound<'_>,
    ) -> EngineResult<()> {
        self.require_round(rows, round)?;

        let regions = self.layout.regions();
        self.arena
            .copy_prefix_from_host(stream, regions.table_rows, round.table_rows)?;
        self.arena
            .copy_prefix_from_host(stream, regions.cache_positions, round.cache_positions)?;
        self.arena
            .copy_prefix_from_host(stream, regions.lengths, round.lengths)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_cos, round.rope_cos)?;
        self.arena
            .copy_prefix_from_host(stream, regions.rope_sin, round.rope_sin)?;

        Ok(())
    }

    fn require_round(&self, rows: usize, round: Qwen38FlashNextQsaRound<'_>) -> EngineResult<()> {
        qwen38_flash_next_row_route(rows)?;
        for (role, len) in [
            ("table rows", round.table_rows.len()),
            ("cache positions", round.cache_positions.len()),
            ("lengths", round.lengths.len()),
        ] {
            if len != rows {
                return Err(EngineError::layout(format!(
                    "Qwen3.8-Flash-Next QSA {role} has {len} entries, expected {rows}"
                )));
            }
        }
        let rotary = product("Qwen3.8-Flash-Next QSA rotary", rows, ROTARY_ELEMENTS)?;
        if round.rope_cos.len() != rotary || round.rope_sin.len() != rotary {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next QSA rotary tables must each cover {rotary} values"
            )));
        }
        for &length in round.lengths {
            crate::require_qwen38_flash_next_dense_qsa_visible(length as usize)?;
            if length as usize > self.layout.context_capacity() {
                return Err(EngineError::route(format!(
                    "Qwen3.8-Flash-Next QSA visible length {length} exceeds this owner's {} page capacity",
                    self.layout.context_capacity()
                )));
            }
        }
        for &row in round.table_rows {
            if row as usize >= MAX_BATCH {
                return Err(EngineError::route(format!(
                    "Qwen3.8-Flash-Next QSA page-table row {row} is outside 0..{MAX_BATCH}"
                )));
            }
        }

        Ok(())
    }

    /// Publishes one expert indirection table, refusing a structurally invalid one.
    pub fn load_slot_table(&self, stream: &CudaStream, table: &[u32]) -> EngineResult<()> {
        qwen38_flash_next_expert_slot_plane(self.pool_layout.slot_count())
            .validate_published_table(table)?;
        self.pool_arena
            .copy_from_host(stream, self.pool_layout.regions().slot_table, table)?;

        Ok(())
    }

    /// Clears every paged cache plane, including the reserved indexer keys.
    pub fn reset_cache(&self, stream: &CudaStream) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.key_pages,
            regions.value_pages,
            regions.indexer_pages,
        ] {
            self.arena.fill(stream, region, 0)?;
        }

        Ok(())
    }

    /// Replays the immutable graph for one admitted row count.
    pub fn replay(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        let graph = self.graph(rows)?;
        // SAFETY: this program owns every captured allocation (both arenas and the op
        // modules) for its whole life and drops the graphs first.
        unsafe { graph.launch(stream) }?;

        Ok(())
    }

    /// Reads active BF16 stream output rows.
    pub fn read_residual(&self, stream: &CudaStream, rows: usize) -> EngineResult<Vec<u16>> {
        qwen38_flash_next_row_route(rows)?;
        let values = product("Qwen3.8-Flash-Next layer output", rows, A::HC_WIDTH)?;

        Ok(self
            .arena
            .copy_prefix_to_host(stream, self.layout.regions().residual_output, values)?)
    }

    /// Decoder layer owned by this program.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// CUDA context shared by both arenas, the graphs, and the prepared operators.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Stable layer-arena base address captured by every graph.
    pub const fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Stable slot-pool base address captured by every graph.
    pub const fn pool_base_address(&self) -> u64 {
        self.pool_base_address
    }

    /// Exact source-backed device weight bytes, excluding the routed pool.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.layout.resident_weight_bytes()
    }

    /// Exact represented E4M3 cache bytes plus the reserved indexer key plane.
    pub const fn cache_bytes(&self) -> usize {
        self.layout.cache_bytes()
    }

    /// Exact address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.layout.workspace_bytes()
    }

    /// Complete layer allocation, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.layout.arena_bytes()
    }

    /// Complete routed-expert pool allocation, including its table.
    pub const fn pool_arena_bytes(&self) -> usize {
        self.pool_layout.arena_bytes()
    }

    /// Context depth one decode slot reaches.
    pub const fn context_capacity(&self) -> usize {
        self.layout.context_capacity()
    }

    /// Largest admitted exact row count.
    pub const fn row_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_LAYER_MAX_ROWS
    }

    /// Checked layer layout.
    pub const fn layout(&self) -> &Qwen38FlashNextQsaMoeLayerLayout {
        &self.layout
    }

    /// Checked routed-expert pool layout.
    pub const fn pool_layout(&self) -> &Qwen38FlashNextExpertPoolLayout {
        &self.pool_layout
    }

    /// Keeps the admitted mmap-backed snapshot alive with the resident owner.
    pub const fn snapshot(&self) -> &Arc<CheckpointSnapshot<Qwen38FlashNext>> {
        &self.snapshot
    }

    fn graph(&self, rows: usize) -> EngineResult<&CudaGraph> {
        let route = qwen38_flash_next_row_route(rows)?;

        Ok(match route {
            Qwen38FlashNextRowRoute::Decode(_) => &self.graphs[route.graph_index()],
            Qwen38FlashNextRowRoute::Prefill(_) => &self.prefill_graphs[route.graph_index()],
        })
    }

    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    fn pointers(&self) -> GpuResult<Pointers> {
        Pointers::bind(
            &self.arena,
            &self.pool_arena,
            self.layout.regions(),
            self.pool_layout.regions(),
        )
    }

    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    fn ops(&self) -> Ops<'_> {
        Ops {
            hyper: &self._hyper,
            qkv: &self._qkv,
            prepare: &self._prepare,
            attention: &self._attention,
            gate: &self._gate,
            block_output: &self._block_output,
            router: &self._router,
            experts: &self._experts,
        }
    }

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly, for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        qwen38_flash_next_row_route(rows)?;
        launch_route(stream, rows, self.ops(), self.pointers()?)?;

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
        qwen38_flash_next_row_route(rows)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.8-Flash-Next QSA/MoE graph requires at least one operation",
            ));
        }
        let pointers = self.pointers()?;
        let ops = self.ops();

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, rows, ops, pointers)?;
            }
            Ok(())
        })?)
    }

    #[cfg(feature = "qualification")]
    /// Captures one admitted round's metadata upload for benchmark replay.
    pub fn qualification_round_stage_graph(
        &self,
        stream: &CudaStream,
        rows: usize,
        round: Qwen38FlashNextQsaRound<'_>,
    ) -> EngineResult<Qwen38FlashNextQsaRoundStageGraph> {
        self.require_round(rows, round)?;
        let table_rows = PinnedHostBuffer::from_slice(&self.context, round.table_rows)
            .map_err(GpuError::from)?;
        let cache_positions = PinnedHostBuffer::from_slice(&self.context, round.cache_positions)
            .map_err(GpuError::from)?;
        let lengths =
            PinnedHostBuffer::from_slice(&self.context, round.lengths).map_err(GpuError::from)?;
        let rope_cos =
            PinnedHostBuffer::from_slice(&self.context, round.rope_cos).map_err(GpuError::from)?;
        let rope_sin =
            PinnedHostBuffer::from_slice(&self.context, round.rope_sin).map_err(GpuError::from)?;
        let rotary = product("Qwen3.8-Flash-Next QSA rotary", rows, ROTARY_ELEMENTS)?;
        let regions = self.layout.regions();
        let graph = CudaGraph::capture(stream, || {
            // SAFETY: the returned owner retains every pinned source through all replays.
            unsafe {
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    regions.table_rows,
                    &table_rows,
                    rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    regions.cache_positions,
                    &cache_positions,
                    rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    regions.lengths,
                    &lengths,
                    rows,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    regions.rope_cos,
                    &rope_cos,
                    rotary,
                )?;
                self.arena.copy_prefix_from_pinned_host_async(
                    stream,
                    regions.rope_sin,
                    &rope_sin,
                    rotary,
                )
            }
        })?;

        Ok(Qwen38FlashNextQsaRoundStageGraph {
            graph,
            _table_rows: table_rows,
            _cache_positions: cache_positions,
            _lengths: lengths,
            _rope_cos: rope_cos,
            _rope_sin: rope_sin,
        })
    }

    #[cfg(feature = "qualification")]
    /// Returns every stable arena address in layout order, both arenas included.
    pub fn qualification_addresses(&self) -> EngineResult<Vec<usize>> {
        Ok(self.pointers()?.addresses())
    }

    #[cfg(feature = "qualification")]
    /// Reads every graph input whose contents may vary between launches.
    pub fn qualification_runtime_inputs(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen38FlashNextQsaMoeLayerInputs> {
        let regions = self.layout.regions();

        Ok(Qwen38FlashNextQsaMoeLayerInputs {
            residual_input: self.arena.copy_to_host(stream, regions.residual_input)?,
            table_rows: self.arena.copy_to_host(stream, regions.table_rows)?,
            cache_positions: self.arena.copy_to_host(stream, regions.cache_positions)?,
            lengths: self.arena.copy_to_host(stream, regions.lengths)?,
            block_tables: self.arena.copy_to_host(stream, regions.block_tables)?,
            rope_cos: self.arena.copy_to_host(stream, regions.rope_cos)?,
            rope_sin: self.arena.copy_to_host(stream, regions.rope_sin)?,
            slot_table: self
                .pool_arena
                .copy_to_host(stream, self.pool_layout.regions().slot_table)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Fills every non-cache mutable seam with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.attention_residual,
            regions.residual_output,
            regions.hc_normalized,
            regions.hc_low_rank,
            regions.hc_mixed,
            regions.hc_write_gate,
            regions.qkv,
            regions.attention_gated,
            regions.router_logits,
            regions.expert_indices,
            regions.routing_weights,
            regions.routed_intermediate,
            regions.routed_output,
            regions.shared_intermediate,
            regions.shared_output,
            regions.shared_gate_logit,
            regions.block_output,
        ] {
            self.arena.fill(stream, region, byte)?;
        }
        for region in [regions.query, regions.attention] {
            self.arena.fill(stream, region, byte)?;
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all mutable planes, including inactive rows and the whole cache.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen38FlashNextQsaMoeLayerObservables> {
        let regions = self.layout.regions();

        Ok(Qwen38FlashNextQsaMoeLayerObservables {
            hc_normalized: self.arena.copy_to_host(stream, regions.hc_normalized)?,
            hc_low_rank: self.arena.copy_to_host(stream, regions.hc_low_rank)?,
            hc_mixed: self.arena.copy_to_host(stream, regions.hc_mixed)?,
            hc_write_gate: self.arena.copy_to_host(stream, regions.hc_write_gate)?,
            qkv: self.arena.copy_to_host(stream, regions.qkv)?,
            query: self.arena.copy_to_host(stream, regions.query)?,
            attention: self.arena.copy_to_host(stream, regions.attention)?,
            attention_gated: self.arena.copy_to_host(stream, regions.attention_gated)?,
            key_pages: self.arena.copy_to_host(stream, regions.key_pages)?,
            value_pages: self.arena.copy_to_host(stream, regions.value_pages)?,
            indexer_pages: self.arena.copy_to_host(stream, regions.indexer_pages)?,
            attention_residual: self
                .arena
                .copy_to_host(stream, regions.attention_residual)?,
            router_logits: self.arena.copy_to_host(stream, regions.router_logits)?,
            expert_indices: self.arena.copy_to_host(stream, regions.expert_indices)?,
            routing_weights: self.arena.copy_to_host(stream, regions.routing_weights)?,
            routed_intermediate: self
                .arena
                .copy_to_host(stream, regions.routed_intermediate)?,
            routed_output: self.arena.copy_to_host(stream, regions.routed_output)?,
            shared_intermediate: self
                .arena
                .copy_to_host(stream, regions.shared_intermediate)?,
            shared_output: self.arena.copy_to_host(stream, regions.shared_output)?,
            shared_gate_logit: self.arena.copy_to_host(stream, regions.shared_gate_logit)?,
            block_output: self.arena.copy_to_host(stream, regions.block_output)?,
            residual_output: self.arena.copy_to_host(stream, regions.residual_output)?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Reads every immutable device plane in source order.
    pub fn qualification_immutable(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen38FlashNextQsaMoeLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen38FlashNextQsaMoeLayerImmutable {
            attention_hc_norm: self.arena.copy_to_host(stream, regions.attention_hc_norm)?,
            attention_hc_down: self.arena.copy_to_host(stream, regions.attention_hc_down)?,
            attention_hc_up: self.arena.copy_to_host(stream, regions.attention_hc_up)?,
            attention_hc_inject: self
                .arena
                .copy_to_host(stream, regions.attention_hc_inject)?,
            mlp_hc_norm: self.arena.copy_to_host(stream, regions.mlp_hc_norm)?,
            mlp_hc_down: self.arena.copy_to_host(stream, regions.mlp_hc_down)?,
            mlp_hc_up: self.arena.copy_to_host(stream, regions.mlp_hc_up)?,
            mlp_hc_inject: self.arena.copy_to_host(stream, regions.mlp_hc_inject)?,
            qkv_weight: self.arena.copy_to_host(stream, regions.qkv_weight)?,
            output_weight: self.arena.copy_to_host(stream, regions.output_weight)?,
            query_norm: self.arena.copy_to_host(stream, regions.query_norm)?,
            key_norm: self.arena.copy_to_host(stream, regions.key_norm)?,
            indexer_qk_weight: self.arena.copy_to_host(stream, regions.indexer_qk_weight)?,
            indexer_query_norm: self
                .arena
                .copy_to_host(stream, regions.indexer_query_norm)?,
            indexer_key_norm: self.arena.copy_to_host(stream, regions.indexer_key_norm)?,
            router_weight: self.arena.copy_to_host(stream, regions.router_weight)?,
            expert_weight_scales_2: self
                .arena
                .copy_to_host(stream, regions.expert_weight_scales_2)?,
            shared_gate_weight: self
                .arena
                .copy_to_host(stream, regions.shared_gate_weight)?,
            shared_up_weight: self.arena.copy_to_host(stream, regions.shared_up_weight)?,
            shared_down_weight: self
                .arena
                .copy_to_host(stream, regions.shared_down_weight)?,
            shared_gate_logit_weight: self
                .arena
                .copy_to_host(stream, regions.shared_gate_logit_weight)?,
            slot_pool: self
                .pool_arena
                .copy_to_host(stream, self.pool_layout.regions().slot_pool)?,
        })
    }
}

/// Rotary elements one token carries, `rotary_dim / 2` at `partial_rotary_factor = 0.25`.
const ROTARY_ELEMENTS: usize = 32;

/// One round's per-row page metadata and rotary tables.
#[derive(Clone, Copy)]
pub struct Qwen38FlashNextQsaRound<'a> {
    /// Page-table row each row of the round attends through.
    pub table_rows: &'a [u32],
    /// Absolute cache position each row appends at.
    pub cache_positions: &'a [u32],
    /// Visible key count each row attends over, checked against the dense ceiling.
    pub lengths: &'a [u32],
    /// Rotary cosines, `rows * 32` values.
    pub rope_cos: &'a [f32],
    /// Rotary sines, `rows * 32` values.
    pub rope_sin: &'a [f32],
}

#[cfg(feature = "qualification")]
/// Captured round upload retaining its pinned sources.
pub struct Qwen38FlashNextQsaRoundStageGraph {
    graph: CudaGraph,
    _table_rows: PinnedHostBuffer<u32>,
    _cache_positions: PinnedHostBuffer<u32>,
    _lengths: PinnedHostBuffer<u32>,
    _rope_cos: PinnedHostBuffer<f32>,
    _rope_sin: PinnedHostBuffer<f32>,
}

#[cfg(feature = "qualification")]
impl Qwen38FlashNextQsaRoundStageGraph {
    /// Immutable graph restoring one exact round's runtime metadata.
    pub const fn graph(&self) -> &CudaGraph {
        &self.graph
    }
}

#[cfg(feature = "qualification")]
/// Runtime-owned planes that must stay immutable across one layer launch.
pub struct Qwen38FlashNextQsaMoeLayerInputs {
    /// The widened BF16 stream entering the layer.
    pub residual_input: Vec<u16>,
    /// Page-table row selected by each decode row or prompt sequence.
    pub table_rows: Vec<u32>,
    /// Absolute cache position each row appends at.
    pub cache_positions: Vec<u32>,
    /// Visible key count each row attends over.
    pub lengths: Vec<u32>,
    /// Physical page mapping shared by the key, value, and indexer planes.
    pub block_tables: Vec<u32>,
    /// Rotary cosines for this round.
    pub rope_cos: Vec<f32>,
    /// Rotary sines for this round.
    pub rope_sin: Vec<f32>,
    /// Published expert id to slot assignment.
    pub slot_table: Vec<u32>,
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen38FlashNextQsaMoeLayerObservables {
    /// Grouped-normalized stream from the most recent bracket.
    pub hc_normalized: Vec<u16>,
    /// Low-rank read-gate activations from the most recent bracket.
    pub hc_low_rank: Vec<u16>,
    /// Four-way folded block input from the most recent bracket.
    pub hc_mixed: Vec<u16>,
    /// Per-branch scalar write gates from the most recent bracket.
    pub hc_write_gate: Vec<u16>,
    /// Fused query/gate, key, and value projection rows.
    pub qkv: Vec<u16>,
    /// Normalized and rotated FP32 query rows.
    pub query: Vec<f32>,
    /// FP32 attention output before the gate.
    pub attention: Vec<f32>,
    /// Sigmoid-gated BF16 attention output.
    pub attention_gated: Vec<u16>,
    /// Represented E4M3 key pages.
    pub key_pages: Vec<u8>,
    /// Represented E4M3 value pages.
    pub value_pages: Vec<u8>,
    /// Reserved indexer key pages, unwritten while the dense route is the admitted one.
    pub indexer_pages: Vec<u8>,
    /// The stream after the attention write-back.
    pub attention_residual: Vec<u16>,
    /// BF16 logits for all 512 routed experts.
    pub router_logits: Vec<u16>,
    /// Selected top-ten expert indices, ascending.
    pub expert_indices: Vec<u16>,
    /// Renormalized top-ten BF16 routing weights.
    pub routing_weights: Vec<u16>,
    /// Per-rank routed SwiGLU intermediates.
    pub routed_intermediate: Vec<u16>,
    /// Per-rank routed down-projection outputs.
    pub routed_output: Vec<u16>,
    /// Shared-expert SwiGLU intermediate.
    pub shared_intermediate: Vec<u16>,
    /// Shared-expert down-projection output.
    pub shared_output: Vec<u16>,
    /// Shared-expert gate logits.
    pub shared_gate_logit: Vec<u16>,
    /// The 2,560-wide sublayer output, holding the MoE combine after a full launch.
    pub block_output: Vec<u16>,
    /// Published layer stream rows.
    pub residual_output: Vec<u16>,
}

#[cfg(feature = "qualification")]
/// Immutable source-backed device planes exposed to the qualification crate.
pub struct Qwen38FlashNextQsaMoeLayerImmutable {
    /// Attention-bracket grouped RMSNorm weights.
    pub attention_hc_norm: Vec<u16>,
    /// Attention-bracket read-gate down projection.
    pub attention_hc_down: Vec<u16>,
    /// Attention-bracket read-gate up projection.
    pub attention_hc_up: Vec<u16>,
    /// Attention-bracket per-branch write gate.
    pub attention_hc_inject: Vec<u16>,
    /// MLP-bracket grouped RMSNorm weights.
    pub mlp_hc_norm: Vec<u16>,
    /// MLP-bracket read-gate down projection.
    pub mlp_hc_down: Vec<u16>,
    /// MLP-bracket read-gate up projection.
    pub mlp_hc_up: Vec<u16>,
    /// MLP-bracket per-branch write gate.
    pub mlp_hc_inject: Vec<u16>,
    /// Fused BF16 query/gate, key, and value projection.
    pub qkv_weight: Vec<u16>,
    /// Attention output projection.
    pub output_weight: Vec<u16>,
    /// Per-head query RMSNorm weights.
    pub query_norm: Vec<u16>,
    /// Per-head key RMSNorm weights.
    pub key_norm: Vec<u16>,
    /// Indexer projection, resident and unread on the dense route.
    pub indexer_qk_weight: Vec<u16>,
    /// Indexer query RMSNorm weights.
    pub indexer_query_norm: Vec<u16>,
    /// Indexer key RMSNorm weights.
    pub indexer_key_norm: Vec<u16>,
    /// Full 512-way router weights.
    pub router_weight: Vec<u16>,
    /// Per-expert gate, up, and down `weight_scale_2`.
    pub expert_weight_scales_2: Vec<f32>,
    /// Shared-expert gate projection.
    pub shared_gate_weight: Vec<u16>,
    /// Shared-expert up projection.
    pub shared_up_weight: Vec<u16>,
    /// Shared-expert down projection.
    pub shared_down_weight: Vec<u16>,
    /// Shared-expert scalar routing gate.
    pub shared_gate_logit_weight: Vec<u16>,
    /// The sealed routed-expert slot arena.
    pub slot_pool: Vec<u8>,
}

fn launch_route(
    stream: &CudaStream,
    rows: usize,
    ops: Ops<'_>,
    pointers: Pointers,
) -> GpuResult<()> {
    // SAFETY: two arenas own aligned, disjoint 1,024-row working planes, a 264-page cache
    // addressed off one block table, and one sealed slot pool. Every leaf in the composition
    // selects the same exact row count.
    unsafe {
        // --- attention bracket ---
        ops.hyper.launch_input_mix(
            stream,
            rows,
            pointers.residual_input,
            pointers.attention_hc_norm,
            pointers.attention_hc_down,
            pointers.attention_hc_up,
            pointers.attention_hc_inject,
            pointers.hc_normalized,
            pointers.hc_low_rank,
            pointers.hc_mixed,
            pointers.hc_write_gate,
        )?;
        ops.qkv.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            pointers.qkv_weight,
            pointers.qkv,
        )?;
        ops.prepare.launch(
            stream,
            rows,
            pointers.qkv.cast_const(),
            pointers.query_norm,
            pointers.key_norm,
            pointers.rope_cos,
            pointers.rope_sin,
            pointers.block_tables,
            pointers.table_rows,
            QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE,
            pointers.cache_positions,
            pointers.query,
            pointers.key_pages,
            pointers.value_pages,
            KEY_CACHE_SCALE,
            VALUE_CACHE_SCALE,
        )?;
        ops.attention.launch(
            stream,
            rows,
            pointers.query.cast_const(),
            pointers.key_pages.cast_const(),
            pointers.value_pages.cast_const(),
            pointers.block_tables,
            pointers.table_rows,
            QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE,
            pointers.lengths,
            pointers.attention,
            KEY_CACHE_SCALE,
            VALUE_CACHE_SCALE,
        )?;
        ops.gate.launch(
            stream,
            rows,
            pointers.attention,
            pointers.qkv.cast_const(),
            pointers.attention_gated,
        )?;
        ops.block_output.launch(
            stream,
            rows,
            pointers.attention_gated.cast_const(),
            pointers.output_weight,
            pointers.block_output,
        )?;
        ops.hyper.launch_write_back(
            stream,
            rows,
            pointers.residual_input,
            pointers.block_output.cast_const(),
            pointers.hc_write_gate.cast_const(),
            pointers.attention_residual,
        )?;

        // --- MLP bracket ---
        ops.hyper.launch_input_mix(
            stream,
            rows,
            pointers.attention_residual.cast_const(),
            pointers.mlp_hc_norm,
            pointers.mlp_hc_down,
            pointers.mlp_hc_up,
            pointers.mlp_hc_inject,
            pointers.hc_normalized,
            pointers.hc_low_rank,
            pointers.hc_mixed,
            pointers.hc_write_gate,
        )?;
        ops.router.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            pointers.router_weight,
            pointers.router_logits,
            pointers.expert_indices,
            pointers.routing_weights,
        )?;
        ops.experts.launch(
            stream,
            rows,
            &Qwen38FlashNextExpertDispatch {
                input: pointers.hc_mixed.cast_const(),
                expert_indices: pointers.expert_indices.cast_const(),
                routing_weights: pointers.routing_weights.cast_const(),
                slot_table: pointers.slot_table,
                slot_pool: pointers.slot_pool,
                weight_scales_2: pointers.expert_weight_scales_2,
                shared_gate_weight: pointers.shared_gate_weight,
                shared_up_weight: pointers.shared_up_weight,
                shared_down_weight: pointers.shared_down_weight,
                shared_gate_logit_weight: pointers.shared_gate_logit_weight,
                routed_intermediate: pointers.routed_intermediate,
                routed_output: pointers.routed_output,
                shared_intermediate: pointers.shared_intermediate,
                shared_output: pointers.shared_output,
                shared_gate_logit: pointers.shared_gate_logit,
                output: pointers.block_output,
            },
        )?;
        ops.hyper.launch_write_back(
            stream,
            rows,
            pointers.attention_residual.cast_const(),
            pointers.block_output.cast_const(),
            pointers.hc_write_gate.cast_const(),
            pointers.residual_output,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{KEY_CACHE_SCALE, ROTARY_ELEMENTS, VALUE_CACHE_SCALE};
    use crate::{QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, Qwen38FlashNextQsaMoeLayerLayout};

    #[test]
    fn the_cache_scales_are_exactly_representable_powers_of_two() {
        // A cache scale that is not a power of two would add a rounding step the E4M3 codec's
        // qualification never covered.
        assert_eq!(KEY_CACHE_SCALE, 2.0_f32.powi(-5));
        assert_eq!(VALUE_CACHE_SCALE, 2.0_f32.powi(-4));
        assert_eq!(ROTARY_ELEMENTS, 32);
    }

    #[test]
    fn the_owner_covers_the_whole_admitted_dense_band() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();

        assert!(layout.context_capacity() >= QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING);
    }
}

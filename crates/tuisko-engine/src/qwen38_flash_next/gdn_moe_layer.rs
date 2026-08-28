//! Qwen3.8-Flash-Next GDN/MoE decoder layer.
//!
//! The layer arena owns backbone weights, workspace, and recurrent state. Routed experts remain
//! in a separate address-stable slot pool. The owner captures `B=1..8` and
//! `T=32/64/128/1024` exactly.

use crate::common::graph::{capture_batch_graphs, capture_route_graphs};
use crate::common::math::product;
use crate::qwen38_flash_next::expert_pool_layout::{
    Qwen38FlashNextExpertPoolLayout, Qwen38FlashNextExpertPoolRegions,
};
use crate::qwen38_flash_next::gdn_moe_layer_layout::{
    QWEN38_FLASH_NEXT_LAYER_MAX_ROWS, Qwen38FlashNextGdnMoeLayerRegions, Qwen38FlashNextPleRegions,
};
use crate::qwen38_flash_next::layer_route::{
    QWEN38_FLASH_NEXT_PREFILL_ROWS, Qwen38FlashNextRowRoute, qwen38_flash_next_row_route,
};
use crate::qwen38_flash_next::layer_upload::{
    HyperConnectionRegions, MoeRegions, bf16_words, upload_expert_pool, upload_hyper_connection,
    upload_moe,
};
use crate::qwen38_flash_next::persistent_state::{
    Qwen38FlashNextGdnPersistent, Qwen38FlashNextPlePersistent,
};
use crate::{EngineError, EngineResult, MAX_BATCH, Qwen38FlashNextGdnMoeLayerLayout};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
};
use tuisko_kernels_sm120::{
    Qwen38FlashNextBlockOutputProjectionOp, Qwen38FlashNextEngramOp, Qwen38FlashNextEngramSources,
    Qwen38FlashNextEngramWorkspace, Qwen38FlashNextExpertDispatch,
    Qwen38FlashNextGdnInputProjectionOp, Qwen38FlashNextGdnPrepareOp,
    Qwen38FlashNextGdnRecurrenceOp, Qwen38FlashNextHyperConnectionOp, Qwen38FlashNextMoeExpertsOp,
    Qwen38FlashNextMoeRouterOp, qwen38_flash_next_expert_slot_plane,
};
use tuisko_model::{
    CheckpointSnapshot, MaterializedQwen38FlashNextEngram, Qwen38FlashNext,
    Qwen38FlashNextEngramBindings, Qwen38FlashNextGdnBindings,
    Qwen38FlashNextLayerHyperConnections, Qwen38FlashNextMoeBindings,
};

type A = Qwen38FlashNext;

/// One Qwen3.8-Flash-Next GDN/MoE decoder layer with immutable exact decode and prefill graphs.
pub struct Qwen38FlashNextGdnMoeLayerProgram {
    // Drop graphs before the arenas and loaded modules whose handles they retain.
    graphs: [CudaGraph; MAX_BATCH],
    prefill_graphs: [CudaGraph; QWEN38_FLASH_NEXT_PREFILL_ROWS.len()],
    arena: DeviceArena,
    pool_arena: DeviceArena,
    _hyper: Qwen38FlashNextHyperConnectionOp,
    _gdn_input: Qwen38FlashNextGdnInputProjectionOp,
    _prepare: Qwen38FlashNextGdnPrepareOp,
    _recurrence: Qwen38FlashNextGdnRecurrenceOp,
    _block_output: Qwen38FlashNextBlockOutputProjectionOp,
    _router: Qwen38FlashNextMoeRouterOp,
    _experts: Qwen38FlashNextMoeExpertsOp,
    _engram: Option<Qwen38FlashNextEngramOp>,
    snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
    context: Arc<CudaContext>,
    layout: Qwen38FlashNextGdnMoeLayerLayout,
    pool_layout: Qwen38FlashNextExpertPoolLayout,
    base_address: u64,
    pool_base_address: u64,
    // Read by the qualification surface, which is the only caller that re-launches a route
    // outside its captured graph.
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    table_scale_bits: u16,
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

    gdn_input_weight: *const u16,
    gdn_control_weight: *const u16,
    gdn_convolution_weight: *const u16,
    gdn_a_log: *const u16,
    gdn_dt_bias: *const u16,
    gdn_norm: *const u16,
    gdn_output_weight: *const u16,

    gdn_projected: *mut u16,
    gdn_convolved: *mut u16,
    gdn_log_decay: *mut f32,
    gdn_beta: *mut f32,
    gdn_recurrent_plane: *mut f32,
    gdn_recurrent_output: *mut u16,
    state_rows: *const u32,
    history: *mut u16,
    state: *mut f32,

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

    ple: Option<PlePointers>,
}

/// The engram module's addresses, present only on the layer that runs one.
#[derive(Clone, Copy)]
struct PlePointers {
    key_proj: *const u16,
    value_proj: *const u16,
    norm_key: *const u16,
    norm_query: *const u16,
    norm_conv: *const u16,
    convolution: *const u16,
    codes: *const u8,
    injected: *mut u16,
    conv_state: *mut u16,
    embedding: *mut u16,
    key: *mut u16,
    key_normed: *mut u16,
    query_normed: *mut u16,
    value: *mut u16,
    gated: *mut u16,
    gated_normed: *mut u16,
    delta: *mut u16,
}

impl Pointers {
    fn bind(
        arena: &DeviceArena,
        pool: &DeviceArena,
        regions: Qwen38FlashNextGdnMoeLayerRegions,
        pool_regions: Qwen38FlashNextExpertPoolRegions,
        gdn: Qwen38FlashNextGdnPersistent,
        ple: Option<(Qwen38FlashNextPleRegions, Qwen38FlashNextPlePersistent)>,
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

            gdn_input_weight: arena.address(regions.gdn_input_weight)?.cast_const(),
            gdn_control_weight: arena.address(regions.gdn_control_weight)?.cast_const(),
            gdn_convolution_weight: arena.address(regions.gdn_convolution_weight)?.cast_const(),
            gdn_a_log: arena.address(regions.gdn_a_log)?.cast_const(),
            gdn_dt_bias: arena.address(regions.gdn_dt_bias)?.cast_const(),
            gdn_norm: arena.address(regions.gdn_norm)?.cast_const(),
            gdn_output_weight: arena.address(regions.gdn_output_weight)?.cast_const(),

            gdn_projected: arena.address(regions.gdn_projected)?,
            gdn_convolved: arena.address(regions.gdn_convolved)?,
            gdn_log_decay: arena.address(regions.gdn_log_decay)?,
            gdn_beta: arena.address(regions.gdn_beta)?,
            gdn_recurrent_plane: arena.address(regions.gdn_recurrent_plane)?,
            gdn_recurrent_output: arena.address(regions.gdn_recurrent_output)?,
            state_rows: arena.address(regions.state_rows)?.cast_const(),
            history: arena.address(gdn.history)?,
            state: arena.address(gdn.state)?,

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

            ple: ple
                .map(|(regions, carry)| {
                    Ok::<_, GpuError>(PlePointers {
                        key_proj: arena.address(regions.key_proj)?.cast_const(),
                        value_proj: arena.address(regions.value_proj)?.cast_const(),
                        norm_key: arena.address(regions.norm_key)?.cast_const(),
                        norm_query: arena.address(regions.norm_query)?.cast_const(),
                        norm_conv: arena.address(regions.norm_conv)?.cast_const(),
                        convolution: arena.address(regions.convolution)?.cast_const(),
                        codes: arena.address(regions.codes)?.cast_const(),
                        injected: arena.address(regions.injected)?,
                        conv_state: arena.address(carry.conv_state)?,
                        embedding: arena.address(regions.embedding)?,
                        key: arena.address(regions.key)?,
                        key_normed: arena.address(regions.key_normed)?,
                        query_normed: arena.address(regions.query_normed)?,
                        value: arena.address(regions.value)?,
                        gated: arena.address(regions.gated)?,
                        gated_normed: arena.address(regions.gated_normed)?,
                        delta: arena.address(regions.delta)?,
                    })
                })
                .transpose()?,
        })
    }

    /// The stream the attention bracket reads: the engram output where there is one.
    const fn block_stream(self) -> *const u16 {
        match self.ple {
            Some(ple) => ple.injected.cast_const(),
            None => self.residual_input,
        }
    }

    #[cfg(feature = "qualification")]
    fn addresses(self) -> Vec<usize> {
        let mut addresses = vec![
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
            self.gdn_input_weight.addr(),
            self.gdn_control_weight.addr(),
            self.gdn_convolution_weight.addr(),
            self.gdn_a_log.addr(),
            self.gdn_dt_bias.addr(),
            self.gdn_norm.addr(),
            self.gdn_output_weight.addr(),
            self.gdn_projected.addr(),
            self.gdn_convolved.addr(),
            self.gdn_log_decay.addr(),
            self.gdn_beta.addr(),
            self.gdn_recurrent_output.addr(),
            self.state_rows.addr(),
            self.history.addr(),
            self.state.addr(),
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
        ];
        if let Some(ple) = self.ple {
            addresses.extend([
                ple.key_proj.addr(),
                ple.value_proj.addr(),
                ple.norm_key.addr(),
                ple.norm_query.addr(),
                ple.norm_conv.addr(),
                ple.convolution.addr(),
                ple.codes.addr(),
                ple.injected.addr(),
                ple.conv_state.addr(),
                ple.embedding.addr(),
                ple.key.addr(),
                ple.key_normed.addr(),
                ple.query_normed.addr(),
                ple.value.addr(),
                ple.gated.addr(),
                ple.gated_normed.addr(),
                ple.delta.addr(),
            ]);
        }

        addresses
    }
}

#[derive(Clone, Copy)]
struct Ops<'a> {
    hyper: &'a Qwen38FlashNextHyperConnectionOp,
    gdn_input: &'a Qwen38FlashNextGdnInputProjectionOp,
    prepare: &'a Qwen38FlashNextGdnPrepareOp,
    recurrence: &'a Qwen38FlashNextGdnRecurrenceOp,
    block_output: &'a Qwen38FlashNextBlockOutputProjectionOp,
    router: &'a Qwen38FlashNextMoeRouterOp,
    experts: &'a Qwen38FlashNextMoeExpertsOp,
    engram: Option<&'a Qwen38FlashNextEngramOp>,
}

impl Qwen38FlashNextGdnMoeLayerProgram {
    /// Loads one source layer and captures every admitted decode and prefill route.
    pub fn from_snapshot(
        context: &Arc<CudaContext>,
        snapshot: Arc<CheckpointSnapshot<Qwen38FlashNext>>,
        layer: usize,
    ) -> EngineResult<Self> {
        let gdn = Qwen38FlashNextGdnBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let moe = Qwen38FlashNextMoeBindings::bind(snapshot.as_ref(), layer)?.materialize()?;
        let hc =
            Qwen38FlashNextLayerHyperConnections::bind(snapshot.as_ref(), layer)?.materialize()?;
        let engram_source = (layer == A::PLE_LAYER)
            .then(|| Qwen38FlashNextEngramBindings::bind(snapshot.as_ref(), layer)?.materialize())
            .transpose()?;

        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(layer)?;
        let pool_layout = Qwen38FlashNextExpertPoolLayout::resident()?;
        let regions = layout.regions();
        let stream = context.new_stream().map_err(GpuError::from)?;
        let arena = DeviceArena::zeroed(&stream, layout.builder())?;
        let pool_arena = DeviceArena::zeroed(&stream, pool_layout.builder())?;

        let hyper = Qwen38FlashNextHyperConnectionOp::new(context)?;
        let gdn_input = Qwen38FlashNextGdnInputProjectionOp::new(context)?;
        let prepare = Qwen38FlashNextGdnPrepareOp::new(context)?;
        let recurrence = Qwen38FlashNextGdnRecurrenceOp::new(context)?;
        let block_output = Qwen38FlashNextBlockOutputProjectionOp::new(context)?;
        let router = Qwen38FlashNextMoeRouterOp::new(context)?;
        let experts = Qwen38FlashNextMoeExpertsOp::new(context)?;
        let engram = engram_source
            .is_some()
            .then(|| Qwen38FlashNextEngramOp::new(context))
            .transpose()?;

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
        upload_gdn(&arena, &stream, regions, &gdn)?;
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
        arena.copy_from_host(
            &stream,
            regions.state_rows,
            &(0..MAX_BATCH as u32).collect::<Vec<_>>(),
        )?;

        let table_scale_bits = match (&engram_source, layout.ple()) {
            (Some(source), Some(ple)) => {
                upload_ple(&arena, &stream, ple, source)?;
                source.table_scale_bits
            }
            (None, None) => 0,
            _ => {
                return Err(EngineError::layout(format!(
                    "Qwen3.8-Flash-Next layer {layer} disagrees with its own engram reservation"
                )));
            }
        };
        stream.synchronize().map_err(GpuError::from)?;

        let pointers = Pointers::bind(
            &arena,
            &pool_arena,
            regions,
            pool_layout.regions(),
            layout.persistent().gdn().ok_or_else(|| {
                EngineError::layout("Qwen3.8-Flash-Next GDN layer has no recurrent carry")
            })?,
            layout.ple().zip(layout.persistent().ple()),
        )?;
        let base_address = arena.base_address();
        let pool_base_address = pool_arena.base_address();
        let ops = Ops {
            hyper: &hyper,
            gdn_input: &gdn_input,
            prepare: &prepare,
            recurrence: &recurrence,
            block_output: &block_output,
            router: &router,
            experts: &experts,
            engram: engram.as_ref(),
        };
        let graphs = capture_batch_graphs(
            &stream,
            "Qwen3.8-Flash-Next GDN/MoE decode graph inventory has wrong cardinality",
            |rows| launch_route(&stream, rows, ops, pointers, table_scale_bits),
        )?;
        let prefill_graphs = capture_route_graphs(
            &stream,
            QWEN38_FLASH_NEXT_PREFILL_ROWS,
            "Qwen3.8-Flash-Next GDN/MoE prefill graph inventory has wrong cardinality",
            |rows| launch_route(&stream, rows, ops, pointers, table_scale_bits),
        )?;

        Ok(Self {
            graphs,
            prefill_graphs,
            arena,
            pool_arena,
            _hyper: hyper,
            _gdn_input: gdn_input,
            _prepare: prepare,
            _recurrence: recurrence,
            _block_output: block_output,
            _router: router,
            _experts: experts,
            _engram: engram,
            snapshot,
            context: context.clone(),
            layout,
            pool_layout,
            base_address,
            pool_base_address,
            table_scale_bits,
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

    /// Uploads one round of staged engram codes, on the layer that consumes them.
    pub fn load_engram_codes(
        &self,
        stream: &CudaStream,
        rows: usize,
        codes: &[u8],
    ) -> EngineResult<()> {
        qwen38_flash_next_row_route(rows)?;
        let ple = self.layout.ple().ok_or_else(|| {
            EngineError::route(format!(
                "Qwen3.8-Flash-Next layer {} runs no engram module",
                self.layer
            ))
        })?;
        let expected = product(
            "Qwen3.8-Flash-Next engram round",
            rows,
            product(
                "Qwen3.8-Flash-Next engram token bytes",
                A::NGRAM_HEADS,
                A::NGRAM_HEAD_DIM,
            )?,
        )?;
        if codes.len() != expected {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next engram round has {} bytes, expected {expected} for rows={rows}",
                codes.len()
            )));
        }
        self.arena.copy_prefix_from_host(stream, ple.codes, codes)?;

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

    /// Clears all slot-owned recurrent, convolution, and engram carries.
    pub fn reset_state(&self, stream: &CudaStream) -> EngineResult<()> {
        let persistent = self.layout.persistent();
        if let Some(gdn) = persistent.gdn() {
            self.arena.fill(stream, gdn.history, 0)?;
            self.arena.fill(stream, gdn.state, 0)?;
        }
        if let Some(ple) = persistent.ple() {
            self.arena.fill(stream, ple.conv_state, 0)?;
        }

        Ok(())
    }

    /// Maps compact rows to distinct physical carry slots.
    pub fn load_slot_routes(&self, stream: &CudaStream, slots: &[usize]) -> EngineResult<()> {
        let rows = slot_rows(slots)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.regions().state_rows,
            &rows[..slots.len()],
        )?;

        Ok(())
    }

    /// Selects one physical carry slot for a causal prompt route.
    pub fn load_prefill_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        self.arena.copy_prefix_from_host(
            stream,
            self.layout.regions().state_rows,
            &[slot as u32],
        )?;

        Ok(())
    }

    /// Clears one physical slot's carries, leaving every other slot untouched.
    pub fn reset_slot(&self, stream: &CudaStream, slot: usize) -> EngineResult<()> {
        require_slot(slot)?;
        let persistent = self.layout.persistent();
        if let Some(gdn) = persistent.gdn() {
            fill_slot(&self.arena, stream, gdn.history, slot)?;
            fill_slot(&self.arena, stream, gdn.state, slot)?;
        }
        if let Some(ple) = persistent.ple() {
            fill_slot(&self.arena, stream, ple.conv_state, slot)?;
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

    /// Exact address-stable workspace and carry bytes.
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

    /// Largest admitted exact row count.
    pub const fn row_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_LAYER_MAX_ROWS
    }

    /// Checked layer layout.
    pub const fn layout(&self) -> &Qwen38FlashNextGdnMoeLayerLayout {
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

    #[cfg(feature = "qualification")]
    /// Launches the production route eagerly, for graph-agreement qualification.
    pub fn launch_eager(&self, stream: &CudaStream, rows: usize) -> EngineResult<()> {
        qwen38_flash_next_row_route(rows)?;
        launch_route(
            stream,
            rows,
            self.ops(),
            self.pointers()?,
            self.table_scale_bits,
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
        qwen38_flash_next_row_route(rows)?;
        if operations == 0 {
            return Err(EngineError::route(
                "repeated Qwen3.8-Flash-Next GDN/MoE graph requires at least one operation",
            ));
        }
        let pointers = self.pointers()?;
        let ops = self.ops();
        let scale = self.table_scale_bits;

        Ok(CudaGraph::capture(stream, || {
            for _ in 0..operations {
                launch_route(stream, rows, ops, pointers, scale)?;
            }
            Ok(())
        })?)
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
    ) -> EngineResult<Qwen38FlashNextGdnMoeLayerInputs> {
        let regions = self.layout.regions();

        Ok(Qwen38FlashNextGdnMoeLayerInputs {
            residual_input: self.arena.copy_to_host(stream, regions.residual_input)?,
            state_rows: self.arena.copy_to_host(stream, regions.state_rows)?,
            slot_table: self
                .pool_arena
                .copy_to_host(stream, self.pool_layout.regions().slot_table)?,
            engram_codes: self
                .layout
                .ple()
                .map(|ple| self.arena.copy_to_host(stream, ple.codes))
                .transpose()?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Fills every non-carry mutable seam with one byte sentinel.
    pub fn qualification_reset_outputs(&self, stream: &CudaStream, byte: u8) -> EngineResult<()> {
        let regions = self.layout.regions();
        for region in [
            regions.attention_residual,
            regions.residual_output,
            regions.hc_normalized,
            regions.hc_low_rank,
            regions.hc_mixed,
            regions.hc_write_gate,
            regions.gdn_projected,
            regions.gdn_convolved,
            regions.gdn_recurrent_output,
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
        for region in [regions.gdn_log_decay, regions.gdn_beta] {
            self.arena.fill(stream, region, byte)?;
        }
        if let Some(ple) = self.layout.ple() {
            for region in [
                ple.injected,
                ple.embedding,
                ple.key,
                ple.key_normed,
                ple.query_normed,
                ple.value,
                ple.gated,
                ple.gated_normed,
                ple.delta,
            ] {
                self.arena.fill(stream, region, byte)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads all mutable planes, including inactive rows.
    pub fn qualification_observables(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen38FlashNextGdnMoeLayerObservables> {
        let regions = self.layout.regions();
        let persistent = self.layout.persistent();
        let gdn = persistent
            .gdn()
            .ok_or_else(|| EngineError::layout("Qwen3.8-Flash-Next GDN layer has no carry"))?;

        Ok(Qwen38FlashNextGdnMoeLayerObservables {
            hc_normalized: self.arena.copy_to_host(stream, regions.hc_normalized)?,
            hc_low_rank: self.arena.copy_to_host(stream, regions.hc_low_rank)?,
            hc_mixed: self.arena.copy_to_host(stream, regions.hc_mixed)?,
            hc_write_gate: self.arena.copy_to_host(stream, regions.hc_write_gate)?,
            gdn_projected: self.arena.copy_to_host(stream, regions.gdn_projected)?,
            gdn_convolved: self.arena.copy_to_host(stream, regions.gdn_convolved)?,
            gdn_log_decay: self.arena.copy_to_host(stream, regions.gdn_log_decay)?,
            gdn_beta: self.arena.copy_to_host(stream, regions.gdn_beta)?,
            gdn_recurrent_output: self
                .arena
                .copy_to_host(stream, regions.gdn_recurrent_output)?,
            history: self.arena.copy_to_host(stream, gdn.history)?,
            state: self.arena.copy_to_host(stream, gdn.state)?,
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
            ple_injected: self
                .layout
                .ple()
                .map(|ple| self.arena.copy_to_host(stream, ple.injected))
                .transpose()?,
            ple_gated: self
                .layout
                .ple()
                .map(|ple| self.arena.copy_to_host(stream, ple.gated))
                .transpose()?,
            ple_delta: self
                .layout
                .ple()
                .map(|ple| self.arena.copy_to_host(stream, ple.delta))
                .transpose()?,
            ple_conv_state: persistent
                .ple()
                .map(|ple| self.arena.copy_to_host(stream, ple.conv_state))
                .transpose()?,
        })
    }

    #[cfg(feature = "qualification")]
    /// Overwrites the sealed slot arena, so a permuted assignment can be published over it.
    ///
    /// Permuting the table alone would point every expert at another expert's weights. The
    /// invariance the streaming law actually claims is that `(pool, table)` pairs which resolve
    /// to the same bytes produce the same output, so a permutation test must move both.
    pub fn qualification_load_slot_pool(
        &self,
        stream: &CudaStream,
        bytes: &[u8],
    ) -> EngineResult<()> {
        self.pool_arena
            .copy_from_host(stream, self.pool_layout.regions().slot_pool, bytes)?;

        Ok(())
    }

    #[cfg(feature = "qualification")]
    /// Reads every immutable device plane in source order.
    pub fn qualification_immutable(
        &self,
        stream: &CudaStream,
    ) -> EngineResult<Qwen38FlashNextGdnMoeLayerImmutable> {
        let regions = self.layout.regions();

        Ok(Qwen38FlashNextGdnMoeLayerImmutable {
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
            gdn_input_weight: self.arena.copy_to_host(stream, regions.gdn_input_weight)?,
            gdn_control_weight: self
                .arena
                .copy_to_host(stream, regions.gdn_control_weight)?,
            gdn_convolution_weight: self
                .arena
                .copy_to_host(stream, regions.gdn_convolution_weight)?,
            gdn_a_log: self.arena.copy_to_host(stream, regions.gdn_a_log)?,
            gdn_dt_bias: self.arena.copy_to_host(stream, regions.gdn_dt_bias)?,
            gdn_norm: self.arena.copy_to_host(stream, regions.gdn_norm)?,
            gdn_output_weight: self.arena.copy_to_host(stream, regions.gdn_output_weight)?,
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

    // Read by the qualification surface, which is the only caller that re-launches a route
    // outside its captured graph.
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    fn pointers(&self) -> GpuResult<Pointers> {
        Pointers::bind(
            &self.arena,
            &self.pool_arena,
            self.layout.regions(),
            self.pool_layout.regions(),
            self.layout.persistent().gdn().ok_or_else(|| {
                GpuError::invalid_launch("Qwen3.8-Flash-Next GDN layer has no carry")
            })?,
            self.layout.ple().zip(self.layout.persistent().ple()),
        )
    }

    // Read by the qualification surface, which is the only caller that re-launches a route
    // outside its captured graph.
    #[cfg_attr(not(feature = "qualification"), allow(dead_code))]
    fn ops(&self) -> Ops<'_> {
        Ops {
            hyper: &self._hyper,
            gdn_input: &self._gdn_input,
            prepare: &self._prepare,
            recurrence: &self._recurrence,
            block_output: &self._block_output,
            router: &self._router,
            experts: &self._experts,
            engram: self._engram.as_ref(),
        }
    }
}

#[cfg(feature = "qualification")]
/// Runtime-owned planes that must stay immutable across one layer launch.
pub struct Qwen38FlashNextGdnMoeLayerInputs {
    /// The widened BF16 stream entering the layer.
    pub residual_input: Vec<u16>,
    /// Physical carry row selected by each decode slot or prompt sequence.
    pub state_rows: Vec<u32>,
    /// Published expert id to slot assignment.
    pub slot_table: Vec<u32>,
    /// Staged FP8 engram codes, on the layer that consumes them.
    pub engram_codes: Option<Vec<u8>>,
}

#[cfg(feature = "qualification")]
/// Complete mutable planes exposed to the qualification crate.
pub struct Qwen38FlashNextGdnMoeLayerObservables {
    /// Grouped-normalized stream from the most recent bracket.
    pub hc_normalized: Vec<u16>,
    /// Low-rank read-gate activations from the most recent bracket.
    pub hc_low_rank: Vec<u16>,
    /// Four-way folded block input from the most recent bracket.
    pub hc_mixed: Vec<u16>,
    /// Per-branch scalar write gates from the most recent bracket.
    pub hc_write_gate: Vec<u16>,
    /// Fused Q/K/V then Z projection rows.
    pub gdn_projected: Vec<u16>,
    /// Causal-convolved Q/K/V rows.
    pub gdn_convolved: Vec<u16>,
    /// Per-value-head log decays.
    pub gdn_log_decay: Vec<f32>,
    /// Per-value-head update gates.
    pub gdn_beta: Vec<f32>,
    /// Sigmoid-gated normalized recurrent values.
    pub gdn_recurrent_output: Vec<u16>,
    /// Slot-owned causal convolution history.
    pub history: Vec<u16>,
    /// Slot-owned FP32 recurrent state.
    pub state: Vec<f32>,
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
    /// The stream after engram injection, on the layer that has one.
    pub ple_injected: Option<Vec<u16>>,
    /// Gated engram value before its grouped norm.
    pub ple_gated: Option<Vec<u16>>,
    /// The engram delta injected into the stream.
    pub ple_delta: Option<Vec<u16>>,
    /// Slot-owned dilated convolution state.
    pub ple_conv_state: Option<Vec<u16>>,
}

#[cfg(feature = "qualification")]
/// Immutable source-backed device planes exposed to the qualification crate.
pub struct Qwen38FlashNextGdnMoeLayerImmutable {
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
    /// Fused BF16 Q/K/V then Z input projection.
    pub gdn_input_weight: Vec<u16>,
    /// Fused BF16 A then B control projection.
    pub gdn_control_weight: Vec<u16>,
    /// Width-four depthwise convolution weights.
    pub gdn_convolution_weight: Vec<u16>,
    /// Log-space recurrence decay parameters.
    pub gdn_a_log: Vec<u16>,
    /// Recurrence time-step bias.
    pub gdn_dt_bias: Vec<u16>,
    /// Per-head gated RMSNorm weights.
    pub gdn_norm: Vec<u16>,
    /// Recurrent output projection.
    pub gdn_output_weight: Vec<u16>,
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
    table_scale_bits: u16,
) -> GpuResult<()> {
    // SAFETY: two arenas own aligned, disjoint 1,024-row working planes, eight persistent carry
    // slots, and one sealed slot pool. Prompt routes advance a single carry row causally; every
    // leaf in the composition selects the same exact row count.
    unsafe {
        if let (Some(engram), Some(ple)) = (ops.engram, pointers.ple) {
            engram.launch_engram(
                ops.hyper,
                stream,
                rows,
                ple.codes,
                pointers.residual_input,
                Qwen38FlashNextEngramSources {
                    key_proj: ple.key_proj,
                    value_proj: ple.value_proj,
                    norm_key: ple.norm_key,
                    norm_query: ple.norm_query,
                    norm_conv: ple.norm_conv,
                    convolution: ple.convolution,
                    table_scale_bits,
                },
                Qwen38FlashNextEngramWorkspace {
                    embedding: ple.embedding,
                    key: ple.key,
                    key_normed: ple.key_normed,
                    query_normed: ple.query_normed,
                    value: ple.value,
                    gated: ple.gated,
                    gated_normed: ple.gated_normed,
                    delta: ple.delta,
                },
                pointers.state_rows,
                ple.conv_state,
                ple.injected,
            )?;
        }
        let stream_in = pointers.block_stream();

        // --- attention bracket ---
        ops.hyper.launch_input_mix(
            stream,
            rows,
            stream_in,
            pointers.attention_hc_norm,
            pointers.attention_hc_down,
            pointers.attention_hc_up,
            pointers.attention_hc_inject,
            pointers.hc_normalized,
            pointers.hc_low_rank,
            pointers.hc_mixed,
            pointers.hc_write_gate,
        )?;
        ops.gdn_input.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            pointers.gdn_input_weight,
            pointers.gdn_projected,
        )?;
        ops.prepare.launch(
            stream,
            rows,
            pointers.hc_mixed.cast_const(),
            pointers.gdn_control_weight,
            pointers.gdn_a_log,
            pointers.gdn_dt_bias,
            pointers.gdn_projected.cast_const(),
            pointers.gdn_convolution_weight,
            pointers.state_rows,
            pointers.history,
            pointers.gdn_log_decay,
            pointers.gdn_beta,
            pointers.gdn_convolved,
        )?;
        ops.recurrence.launch(
            stream,
            rows,
            pointers.gdn_convolved.cast_const(),
            pointers.gdn_projected.cast_const(),
            pointers.gdn_log_decay.cast_const(),
            pointers.gdn_beta.cast_const(),
            pointers.gdn_norm,
            pointers.state_rows,
            pointers.state,
            pointers.gdn_recurrent_plane,
            pointers.gdn_recurrent_output,
        )?;
        ops.block_output.launch(
            stream,
            rows,
            pointers.gdn_recurrent_output.cast_const(),
            pointers.gdn_output_weight,
            pointers.block_output,
        )?;
        ops.hyper.launch_write_back(
            stream,
            rows,
            stream_in,
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

fn upload_gdn(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Qwen38FlashNextGdnMoeLayerRegions,
    gdn: &tuisko_model::MaterializedQwen38FlashNextGdn<'_>,
) -> EngineResult<()> {
    arena.copy_from_host(
        stream,
        regions.gdn_input_weight,
        &bf16_words(&gdn.input_weight_bf16)?,
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_control_weight,
        &bf16_words(&gdn.control_weight_bf16)?,
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_convolution_weight,
        &gdn.convolution_weight.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_a_log,
        &gdn.a_log.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_dt_bias,
        &gdn.dt_bias.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_norm,
        &gdn.norm.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.gdn_output_weight,
        &gdn.output_weight.words().collect::<Vec<_>>(),
    )?;

    Ok(())
}

fn upload_ple(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Qwen38FlashNextPleRegions,
    engram: &MaterializedQwen38FlashNextEngram<'_>,
) -> EngineResult<()> {
    arena.copy_from_host(
        stream,
        regions.key_proj,
        &engram.key_proj_weight.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.value_proj,
        &engram.value_proj_weight.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.norm_key,
        &engram.norm_key.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.norm_query,
        &engram.norm_query.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.norm_conv,
        &engram.norm_conv.words().collect::<Vec<_>>(),
    )?;
    arena.copy_from_host(
        stream,
        regions.convolution,
        &engram.convolution_weight.words().collect::<Vec<_>>(),
    )?;

    Ok(())
}

fn slot_rows(slots: &[usize]) -> EngineResult<[u32; MAX_BATCH]> {
    if !(1..=MAX_BATCH).contains(&slots.len()) {
        return Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next compact batch {} is outside 1..={MAX_BATCH}",
            slots.len()
        )));
    }
    let mut seen = [false; MAX_BATCH];
    let mut rows = [0u32; MAX_BATCH];
    for (row, &slot) in rows.iter_mut().zip(slots) {
        require_slot(slot)?;
        if std::mem::replace(&mut seen[slot], true) {
            return Err(EngineError::route(format!(
                "Qwen3.8-Flash-Next physical slot {slot} appears more than once"
            )));
        }
        *row = slot as u32;
    }

    Ok(rows)
}

fn require_slot(slot: usize) -> EngineResult<()> {
    if slot >= MAX_BATCH {
        return Err(EngineError::route(format!(
            "Qwen3.8-Flash-Next physical slot {slot} is outside 0..{MAX_BATCH}"
        )));
    }

    Ok(())
}

fn fill_slot<T: tuisko_gpu::DeviceCopy>(
    arena: &DeviceArena,
    stream: &CudaStream,
    region: ArenaRegion<T>,
    slot: usize,
) -> EngineResult<()> {
    if !region.len().is_multiple_of(MAX_BATCH) {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next carry of {} values is not divisible by {MAX_BATCH} slots",
            region.len()
        )));
    }
    let width = region.len() / MAX_BATCH;
    let start = product("Qwen3.8-Flash-Next carry slot offset", slot, width)?;
    arena.fill_slice(stream, region, start, width, 0)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{require_slot, slot_rows};
    use crate::{EngineErrorCode, MAX_BATCH, qwen38_flash_next_row_route};

    #[test]
    fn exact_row_table_covers_decode_and_prefill_only() {
        for rows in [1, 4, 8, 32, 64, 128, 1_024] {
            qwen38_flash_next_row_route(rows).unwrap();
        }
        for rows in [0, 9, 31, 129, 1_023, 2_048] {
            let error = qwen38_flash_next_row_route(rows).unwrap_err();
            assert_eq!(error.code(), Some(EngineErrorCode::Route));
        }
    }

    #[test]
    fn compact_state_slot_table_is_bijective() {
        assert_eq!(slot_rows(&[7, 2]).unwrap()[..2], [7, 2]);
        assert_eq!(slot_rows(&[0]).unwrap()[..1], [0]);

        // A repeated slot would let two rows share one carry.
        assert!(slot_rows(&[1, 1]).is_err());
        assert!(slot_rows(&[]).is_err());
        assert!(slot_rows(&[MAX_BATCH]).is_err());
        assert!(require_slot(MAX_BATCH).is_err());
        assert!(require_slot(MAX_BATCH - 1).is_ok());
    }
}

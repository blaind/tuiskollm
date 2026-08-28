//! Arena layout for one Qwen3.8-Flash-Next GDN/MoE layer.
//!
//! The layer has no classic layer norms; both normalizations belong to its hyper-connection
//! brackets. Routed expert slots are owned separately. The two sequential brackets reuse their
//! staging planes, so inactive tails begin after the second bracket's widest writer.

use crate::common::math::{product, sum};
use crate::qwen38_flash_next::persistent_state::{ALIGNMENT, Qwen38FlashNextPersistentState};
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Widest exact row count this owner captures, the `T=1024` prefill tile.
pub(crate) const QWEN38_FLASH_NEXT_LAYER_MAX_ROWS: usize = 1_024;

/// Scratch slots one token's routed experts occupy, one per selected rank.
const ROUTED_SLOTS: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;

/// Per-expert `weight_scale_2` scalars the routed kernels read: gate, up, then down.
const EXPERT_WEIGHT_SCALES: usize = 3;

/// Every region one Qwen3.8-Flash-Next GDN/MoE layer owns, in launch order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextGdnMoeLayerRegions {
    // --- the four-branch stream ---
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) attention_residual: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,

    // --- hyper-connection weights, one set per bracket ---
    pub(crate) attention_hc_norm: ArenaRegion<u16>,
    pub(crate) attention_hc_down: ArenaRegion<u16>,
    pub(crate) attention_hc_up: ArenaRegion<u16>,
    pub(crate) attention_hc_inject: ArenaRegion<u16>,
    pub(crate) mlp_hc_norm: ArenaRegion<u16>,
    pub(crate) mlp_hc_down: ArenaRegion<u16>,
    pub(crate) mlp_hc_up: ArenaRegion<u16>,
    pub(crate) mlp_hc_inject: ArenaRegion<u16>,

    // --- hyper-connection staging, reserved once and used by both brackets ---
    pub(crate) hc_normalized: ArenaRegion<u16>,
    pub(crate) hc_low_rank: ArenaRegion<u16>,
    pub(crate) hc_mixed: ArenaRegion<u16>,
    pub(crate) hc_write_gate: ArenaRegion<u16>,

    // --- GDN weights ---
    pub(crate) gdn_input_weight: ArenaRegion<u16>,
    pub(crate) gdn_control_weight: ArenaRegion<u16>,
    pub(crate) gdn_convolution_weight: ArenaRegion<u16>,
    pub(crate) gdn_a_log: ArenaRegion<u16>,
    pub(crate) gdn_dt_bias: ArenaRegion<u16>,
    pub(crate) gdn_norm: ArenaRegion<u16>,
    pub(crate) gdn_output_weight: ArenaRegion<u16>,

    // --- GDN activations ---
    pub(crate) gdn_projected: ArenaRegion<u16>,
    pub(crate) gdn_convolved: ArenaRegion<u16>,
    pub(crate) gdn_log_decay: ArenaRegion<f32>,
    pub(crate) gdn_beta: ArenaRegion<f32>,
    pub(crate) gdn_recurrent_plane: ArenaRegion<f32>,
    pub(crate) gdn_recurrent_output: ArenaRegion<u16>,
    pub(crate) state_rows: ArenaRegion<u32>,

    // --- MoE weights that stay resident ---
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) expert_weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) shared_up_weight: ArenaRegion<u16>,
    pub(crate) shared_down_weight: ArenaRegion<u16>,
    pub(crate) shared_gate_logit_weight: ArenaRegion<u16>,

    // --- MoE activations ---
    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_intermediate: ArenaRegion<u16>,
    pub(crate) routed_output: ArenaRegion<u16>,
    pub(crate) shared_intermediate: ArenaRegion<u16>,
    pub(crate) shared_output: ArenaRegion<u16>,
    pub(crate) shared_gate_logit: ArenaRegion<u16>,

    /// The 2,560-wide sublayer output both write-backs inject, written twice per layer.
    pub(crate) block_output: ArenaRegion<u16>,
}

/// Weights and staging one decoder layer's engram module owns, reserved only where one runs.
///
/// The module is self-contained: `launch_engram` dequantizes the staged FP8 rows, projects key
/// and value, takes all three grouped norms from the hyper-connection op, gates, convolves
/// through the dilated state, and injects. Every plane below is one of its documented
/// caller-owned intermediates, so they are observable and each fused boundary stays qualifiable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextPleRegions {
    // --- weights ---
    pub(crate) key_proj: ArenaRegion<u16>,
    pub(crate) value_proj: ArenaRegion<u16>,
    pub(crate) norm_key: ArenaRegion<u16>,
    pub(crate) norm_query: ArenaRegion<u16>,
    pub(crate) norm_conv: ArenaRegion<u16>,
    pub(crate) convolution: ArenaRegion<u16>,

    /// Staged FP8 engram rows for this round, one `NGRAM_HEADS * NGRAM_HEAD_DIM` block per token.
    pub(crate) codes: ArenaRegion<u8>,
    /// The stream after injection, which the attention bracket then reads.
    pub(crate) injected: ArenaRegion<u16>,

    // --- staged intermediates, in pipeline order ---
    pub(crate) embedding: ArenaRegion<u16>,
    pub(crate) key: ArenaRegion<u16>,
    pub(crate) key_normed: ArenaRegion<u16>,
    pub(crate) query_normed: ArenaRegion<u16>,
    pub(crate) value: ArenaRegion<u16>,
    pub(crate) gated: ArenaRegion<u16>,
    pub(crate) gated_normed: ArenaRegion<u16>,
    pub(crate) delta: ArenaRegion<u16>,
}

impl Qwen38FlashNextPleRegions {
    fn reserve(builder: &mut ArenaLayout, rows: usize) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        let row_stream = product("Qwen3.8-Flash-Next PLE stream rows", rows, A::HC_WIDTH)?;
        let row_embed = product("Qwen3.8-Flash-Next PLE embed rows", rows, A::PLE_EMBED_DIM)?;
        let key_proj = product(
            "Qwen3.8-Flash-Next PLE key projection",
            A::HC_WIDTH,
            A::PLE_EMBED_DIM,
        )?;
        let value_proj = product(
            "Qwen3.8-Flash-Next PLE value projection",
            A::PLE_EMBED_DIM,
            A::PLE_EMBED_DIM,
        )?;
        let convolution = product(
            "Qwen3.8-Flash-Next PLE convolution",
            A::HC_WIDTH,
            A::PLE_CONV_KERNEL,
        )?;
        let codes = product(
            "Qwen3.8-Flash-Next PLE staged codes",
            rows,
            product(
                "Qwen3.8-Flash-Next PLE token bytes",
                A::NGRAM_HEADS,
                A::NGRAM_HEAD_DIM,
            )?,
        )?;

        Ok(Self {
            key_proj: builder.reserve(key_proj, ALIGNMENT)?,
            value_proj: builder.reserve(value_proj, ALIGNMENT)?,
            norm_key: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            norm_query: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            norm_conv: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            convolution: builder.reserve(convolution, ALIGNMENT)?,
            codes: builder.reserve(codes, ALIGNMENT)?,
            injected: builder.reserve(row_stream, ALIGNMENT)?,
            embedding: builder.reserve(row_embed, ALIGNMENT)?,
            key: builder.reserve(row_stream, ALIGNMENT)?,
            key_normed: builder.reserve(row_stream, ALIGNMENT)?,
            query_normed: builder.reserve(row_stream, ALIGNMENT)?,
            value: builder.reserve(row_embed, ALIGNMENT)?,
            gated: builder.reserve(row_stream, ALIGNMENT)?,
            gated_normed: builder.reserve(row_stream, ALIGNMENT)?,
            delta: builder.reserve(row_stream, ALIGNMENT)?,
        })
    }

    fn weight_bytes(self) -> EngineResult<usize> {
        sum(
            "Qwen3.8-Flash-Next PLE weights",
            &[
                self.key_proj.byte_len(),
                self.value_proj.byte_len(),
                self.norm_key.byte_len(),
                self.norm_query.byte_len(),
                self.norm_conv.byte_len(),
                self.convolution.byte_len(),
            ],
        )
    }

    fn workspace_bytes(self) -> EngineResult<usize> {
        sum(
            "Qwen3.8-Flash-Next PLE workspace",
            &[
                self.codes.byte_len(),
                self.injected.byte_len(),
                self.embedding.byte_len(),
                self.key.byte_len(),
                self.key_normed.byte_len(),
                self.query_normed.byte_len(),
                self.value.byte_len(),
                self.gated.byte_len(),
                self.gated_normed.byte_len(),
                self.delta.byte_len(),
            ],
        )
    }
}

/// Checked weights, workspace, and slot-owned state for one Qwen3.8-Flash-Next GDN/MoE layer.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextGdnMoeLayerLayout {
    builder: ArenaLayout,
    regions: Qwen38FlashNextGdnMoeLayerRegions,
    ple: Option<Qwen38FlashNextPleRegions>,
    persistent: Qwen38FlashNextPersistentState,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
    layer: usize,
}

impl Qwen38FlashNextGdnMoeLayerLayout {
    /// Reserves one GDN/MoE decoder layer, including the PLE carry when `layer` has one.
    pub fn build(layer: usize) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        require_geometry()?;
        require_gdn_layer(layer)?;

        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;
        let row_stream = product("Qwen3.8-Flash-Next layer stream rows", rows, A::HC_WIDTH)?;
        let row_hidden = product(
            "Qwen3.8-Flash-Next layer hidden rows",
            rows,
            <A as Arch>::HIDDEN,
        )?;
        let row_low_rank = product(
            "Qwen3.8-Flash-Next layer low-rank rows",
            rows,
            A::HC_LOWRANK,
        )?;
        let row_write_gate = product("Qwen3.8-Flash-Next layer write gates", rows, A::HC_COUNT)?;
        let hc_projection = product(
            "Qwen3.8-Flash-Next hyper-connection projection",
            A::HC_LOWRANK,
            A::HC_WIDTH,
        )?;
        let hc_inject = product(
            "Qwen3.8-Flash-Next hyper-connection inject",
            A::HC_COUNT,
            A::HC_WIDTH,
        )?;

        let gdn_input_weight = product(
            "Qwen3.8-Flash-Next GDN input weight",
            A::GDN_INPUT_ROWS,
            <A as Arch>::HIDDEN,
        )?;
        let gdn_control_weight = product(
            "Qwen3.8-Flash-Next GDN control weight",
            product(
                "Qwen3.8-Flash-Next GDN control rows",
                2,
                A::GDN_CONTROL_ROWS,
            )?,
            <A as Arch>::HIDDEN,
        )?;
        let gdn_convolution_weight = product(
            "Qwen3.8-Flash-Next GDN convolution weight",
            A::GDN_QKV_ROWS,
            A::LINEAR_CONV_KERNEL_DIM,
        )?;
        let gdn_output_weight = product(
            "Qwen3.8-Flash-Next GDN output weight",
            <A as Arch>::HIDDEN,
            A::GDN_VALUE_ROWS,
        )?;
        let row_projected = product(
            "Qwen3.8-Flash-Next GDN projected rows",
            rows,
            A::GDN_INPUT_ROWS,
        )?;
        let row_qkv = product(
            "Qwen3.8-Flash-Next GDN convolved rows",
            rows,
            A::GDN_QKV_ROWS,
        )?;
        let row_control = product(
            "Qwen3.8-Flash-Next GDN control values",
            rows,
            A::GDN_CONTROL_ROWS,
        )?;
        let row_value = product("Qwen3.8-Flash-Next GDN value rows", rows, A::GDN_VALUE_ROWS)?;

        let router_weight = product(
            "Qwen3.8-Flash-Next router weight",
            A::NUM_EXPERTS,
            <A as Arch>::HIDDEN,
        )?;
        let expert_weight_scales_2 = product(
            "Qwen3.8-Flash-Next routed weight scales",
            A::NUM_EXPERTS,
            EXPERT_WEIGHT_SCALES,
        )?;
        let shared_gate_up = product(
            "Qwen3.8-Flash-Next shared expert projection",
            A::SHARED_EXPERT_INTERMEDIATE,
            <A as Arch>::HIDDEN,
        )?;
        let row_router_logits = product("Qwen3.8-Flash-Next router logits", rows, A::NUM_EXPERTS)?;
        let row_routed = product("Qwen3.8-Flash-Next routed ranks", rows, ROUTED_SLOTS)?;
        let row_routed_intermediate = product(
            "Qwen3.8-Flash-Next routed intermediate",
            row_routed,
            <A as Arch>::INTERMEDIATE,
        )?;
        let row_routed_output = product(
            "Qwen3.8-Flash-Next routed output",
            row_routed,
            <A as Arch>::HIDDEN,
        )?;
        let row_shared_intermediate = product(
            "Qwen3.8-Flash-Next shared intermediate",
            rows,
            A::SHARED_EXPERT_INTERMEDIATE,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen38FlashNextGdnMoeLayerRegions {
            residual_input: builder.reserve(row_stream, ALIGNMENT)?,
            attention_residual: builder.reserve(row_stream, ALIGNMENT)?,
            residual_output: builder.reserve(row_stream, ALIGNMENT)?,

            attention_hc_norm: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            attention_hc_down: builder.reserve(hc_projection, ALIGNMENT)?,
            attention_hc_up: builder.reserve(hc_projection, ALIGNMENT)?,
            attention_hc_inject: builder.reserve(hc_inject, ALIGNMENT)?,
            mlp_hc_norm: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            mlp_hc_down: builder.reserve(hc_projection, ALIGNMENT)?,
            mlp_hc_up: builder.reserve(hc_projection, ALIGNMENT)?,
            mlp_hc_inject: builder.reserve(hc_inject, ALIGNMENT)?,

            hc_normalized: builder.reserve(row_stream, ALIGNMENT)?,
            hc_low_rank: builder.reserve(row_low_rank, ALIGNMENT)?,
            hc_mixed: builder.reserve(row_hidden, ALIGNMENT)?,
            hc_write_gate: builder.reserve(row_write_gate, ALIGNMENT)?,

            gdn_input_weight: builder.reserve(gdn_input_weight, ALIGNMENT)?,
            gdn_control_weight: builder.reserve(gdn_control_weight, ALIGNMENT)?,
            gdn_convolution_weight: builder.reserve(gdn_convolution_weight, ALIGNMENT)?,
            gdn_a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            gdn_dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            gdn_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            gdn_output_weight: builder.reserve(gdn_output_weight, ALIGNMENT)?,

            gdn_projected: builder.reserve(row_projected, ALIGNMENT)?,
            gdn_convolved: builder.reserve(row_qkv, ALIGNMENT)?,
            gdn_log_decay: builder.reserve(row_control, ALIGNMENT)?,
            gdn_beta: builder.reserve(row_control, ALIGNMENT)?,
            gdn_recurrent_plane: builder.reserve(row_value, ALIGNMENT)?,
            gdn_recurrent_output: builder.reserve(row_value, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,

            router_weight: builder.reserve(router_weight, ALIGNMENT)?,
            expert_weight_scales_2: builder.reserve(expert_weight_scales_2, ALIGNMENT)?,
            shared_gate_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
            shared_up_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
            shared_down_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
            shared_gate_logit_weight: builder.reserve(<A as Arch>::HIDDEN, ALIGNMENT)?,

            router_logits: builder.reserve(row_router_logits, ALIGNMENT)?,
            expert_indices: builder.reserve(row_routed, ALIGNMENT)?,
            routing_weights: builder.reserve(row_routed, ALIGNMENT)?,
            routed_intermediate: builder.reserve(row_routed_intermediate, ALIGNMENT)?,
            routed_output: builder.reserve(row_routed_output, ALIGNMENT)?,
            shared_intermediate: builder.reserve(row_shared_intermediate, ALIGNMENT)?,
            shared_output: builder.reserve(row_hidden, ALIGNMENT)?,
            shared_gate_logit: builder.reserve(rows, ALIGNMENT)?,

            block_output: builder.reserve(row_hidden, ALIGNMENT)?,
        };
        // The engram module is reserved beside the layer that runs it, in the same arena, so a
        // layer-1 program binds one allocation and the PLE planes cannot drift from the stream
        // they inject into.
        let ple = (layer == A::PLE_LAYER)
            .then(|| Qwen38FlashNextPleRegions::reserve(&mut builder, rows))
            .transpose()?;
        let persistent = Qwen38FlashNextPersistentState::reserve(&mut builder, layer)?;

        let resident_weight_bytes = sum(
            "Qwen3.8-Flash-Next GDN/MoE resident weights",
            &[
                regions.attention_hc_norm.byte_len(),
                regions.attention_hc_down.byte_len(),
                regions.attention_hc_up.byte_len(),
                regions.attention_hc_inject.byte_len(),
                regions.mlp_hc_norm.byte_len(),
                regions.mlp_hc_down.byte_len(),
                regions.mlp_hc_up.byte_len(),
                regions.mlp_hc_inject.byte_len(),
                regions.gdn_input_weight.byte_len(),
                regions.gdn_control_weight.byte_len(),
                regions.gdn_convolution_weight.byte_len(),
                regions.gdn_a_log.byte_len(),
                regions.gdn_dt_bias.byte_len(),
                regions.gdn_norm.byte_len(),
                regions.gdn_output_weight.byte_len(),
                regions.router_weight.byte_len(),
                regions.expert_weight_scales_2.byte_len(),
                regions.shared_gate_weight.byte_len(),
                regions.shared_up_weight.byte_len(),
                regions.shared_down_weight.byte_len(),
                regions.shared_gate_logit_weight.byte_len(),
                ple.map(Qwen38FlashNextPleRegions::weight_bytes)
                    .transpose()?
                    .unwrap_or(0),
            ],
        )?;
        let workspace_bytes = sum(
            "Qwen3.8-Flash-Next GDN/MoE workspace",
            &[
                ple.map(Qwen38FlashNextPleRegions::workspace_bytes)
                    .transpose()?
                    .unwrap_or(0),
                regions.residual_input.byte_len(),
                regions.attention_residual.byte_len(),
                regions.residual_output.byte_len(),
                regions.hc_normalized.byte_len(),
                regions.hc_low_rank.byte_len(),
                regions.hc_mixed.byte_len(),
                regions.hc_write_gate.byte_len(),
                regions.gdn_projected.byte_len(),
                regions.gdn_convolved.byte_len(),
                regions.gdn_log_decay.byte_len(),
                regions.gdn_beta.byte_len(),
                regions.gdn_recurrent_plane.byte_len(),
                regions.gdn_recurrent_output.byte_len(),
                regions.state_rows.byte_len(),
                regions.router_logits.byte_len(),
                regions.expert_indices.byte_len(),
                regions.routing_weights.byte_len(),
                regions.routed_intermediate.byte_len(),
                regions.routed_output.byte_len(),
                regions.shared_intermediate.byte_len(),
                regions.shared_output.byte_len(),
                regions.shared_gate_logit.byte_len(),
                regions.block_output.byte_len(),
                persistent.byte_len()?,
            ],
        )?;

        Ok(Self {
            builder,
            regions,
            ple,
            persistent,
            resident_weight_bytes,
            workspace_bytes,
            layer,
        })
    }

    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn regions(&self) -> Qwen38FlashNextGdnMoeLayerRegions {
        self.regions
    }

    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn persistent(&self) -> Qwen38FlashNextPersistentState {
        self.persistent
    }

    // Consumed by the composed layer program, which awaits the Qwen3.8-Flash-Next BF16 backbone
    // projection entries (see this module's gap-marker tests). Kept beside the layout it
    // describes rather than added later, so the program lands as a caller and not as a
    // second source of truth for the region set.
    #[allow(dead_code)]
    pub(crate) const fn ple(&self) -> Option<Qwen38FlashNextPleRegions> {
        self.ple
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed device weight bytes, excluding the streamed routed experts.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact address-stable working and slot-owned state bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights plus workspace, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.workspace_bytes
    }

    /// Decoder layer this layout was built for.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// Whether this layer also carries the PLE dilated convolution state.
    pub const fn carries_ple_state(&self) -> bool {
        self.persistent.ple().is_some()
    }

    /// Largest exact row route this owner captures.
    pub const fn row_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_LAYER_MAX_ROWS
    }
}

impl LayerMemoryLayout for Qwen38FlashNextGdnMoeLayerLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes()
    }

    // This owner holds no paged key/value cache.
    fn cache_bytes(&self) -> usize {
        0
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes()
    }
}

fn require_geometry() -> EngineResult<()> {
    type A = Qwen38FlashNext;
    if A::HC_WIDTH != A::HC_COUNT * <A as Arch>::HIDDEN
        || A::GDN_INPUT_ROWS != A::GDN_QKV_ROWS + A::GDN_VALUE_ROWS
        || A::GDN_CONTROL_ROWS != <A as Arch>::LINEAR_VALUE_HEADS
        || A::GDN_VALUE_ROWS != <A as Arch>::LINEAR_VALUE_HEADS * <A as Arch>::LINEAR_HEAD_DIM
        || A::SHARED_EXPERT_INTERMEDIATE != <A as Arch>::INTERMEDIATE
        || A::NUM_EXPERTS_PER_TOKEN != 10
        || <A as Arch>::LINEAR_CONV_KERNEL_DIM != 4
        || A::GDN_OUTPUT_GATE != "sigmoid"
    {
        return Err(EngineError::layout(
            "Qwen3.8-Flash-Next GDN/MoE geometry differs from the qualified layer contract",
        ));
    }

    Ok(())
}

fn require_gdn_layer(layer: usize) -> EngineResult<()> {
    type A = Qwen38FlashNext;
    if layer >= A::LAYERS {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next layer {layer} is outside 0..{}",
            A::LAYERS
        )));
    }
    if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next layer {layer} is a sparse-attention layer, not a GDN layer"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, QWEN38_FLASH_NEXT_LAYER_MAX_ROWS, Qwen38FlashNextGdnMoeLayerLayout};
    use crate::LayerMemoryLayout;
    use tuisko_gpu::ArenaRegion;
    use tuisko_model::{Arch, Qwen38FlashNext};

    type A = Qwen38FlashNext;

    #[test]
    fn byte_accounting_is_exact() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();

        assert_eq!(layout.resident_weight_bytes(), 154_799_552);
        assert_eq!(layout.workspace_bytes(), 286_541_856);
        assert_eq!(layout.owner_bytes(), 441_341_408);
        assert_eq!(layout.arena_bytes(), 441_341_952);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 544);
        assert_eq!(layout.row_capacity(), QWEN38_FLASH_NEXT_LAYER_MAX_ROWS);
        assert_eq!(LayerMemoryLayout::cache_bytes(&layout), 0);
    }

    #[test]
    fn the_backbone_weights_reproduce_the_resident_plan_s_per_layer_table() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let regions = layout.regions();

        let hyper_connections = regions.attention_hc_norm.byte_len()
            + regions.attention_hc_down.byte_len()
            + regions.attention_hc_up.byte_len()
            + regions.attention_hc_inject.byte_len()
            + regions.mlp_hc_norm.byte_len()
            + regions.mlp_hc_down.byte_len()
            + regions.mlp_hc_up.byte_len()
            + regions.mlp_hc_inject.byte_len();
        assert_eq!(hyper_connections, 26_419_200);
        assert_eq!(regions.gdn_input_weight.byte_len(), 83_886_080);
        assert_eq!(regions.gdn_control_weight.byte_len(), 491_520);
        assert_eq!(regions.gdn_convolution_weight.byte_len(), 81_920);
        assert_eq!(
            regions.gdn_a_log.byte_len()
                + regions.gdn_dt_bias.byte_len()
                + regions.gdn_norm.byte_len(),
            448
        );
        assert_eq!(regions.gdn_output_weight.byte_len(), 31_457_280);
        assert_eq!(regions.router_weight.byte_len(), 2_621_440);
        assert_eq!(
            regions.shared_gate_weight.byte_len()
                + regions.shared_up_weight.byte_len()
                + regions.shared_down_weight.byte_len(),
            9_830_400
        );
        assert_eq!(regions.shared_gate_logit_weight.byte_len(), 5_120);
    }

    #[test]
    fn the_resident_expert_scalar_plane_is_three_scalars_per_expert_not_six() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();

        // Input scales are folded at materialization; kernels read three f32 values per expert.
        assert_eq!(layout.regions().expert_weight_scales_2.byte_len(), 6_144);
        assert_eq!(
            layout.regions().expert_weight_scales_2.len(),
            A::NUM_EXPERTS * 3
        );
    }

    #[test]
    fn the_routed_expert_pool_is_absent_from_the_layer_arena() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();

        // One layer's routed pool is 512 * 2,764,800 B. If it were reserved here the arena
        // would be over 1.4 GiB; it is streamed through the slot cache instead.
        let routed_pool = A::NUM_EXPERTS * 2_764_800;
        assert!(routed_pool > 1_400_000_000);
        assert!(layout.arena_bytes() < routed_pool);
    }

    #[test]
    fn only_the_ple_layer_reserves_the_dilated_conv_state() {
        let plain = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let ple = Qwen38FlashNextGdnMoeLayerLayout::build(A::PLE_LAYER).unwrap();

        assert!(!plain.carries_ple_state());
        assert!(ple.carries_ple_state());
        assert!(plain.ple().is_none());
        assert!(ple.ple().is_some());

        // Layer 1 pays for the whole engram module and nothing else does: its six weight
        // planes, its staged pipeline, and the nine-column dilated conv state.
        assert_eq!(
            ple.resident_weight_bytes() - plain.resident_weight_bytes(),
            65_679_360
        );
        let workspace = ple.workspace_bytes() - plain.workspace_bytes();
        assert_eq!(workspace, 161_382_400);
        assert_eq!(
            workspace - 1_474_560,
            159_907_840,
            "staging without the carry"
        );
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(A::PLE_LAYER).unwrap();
        let regions = layout.regions();
        let persistent = layout.persistent();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.attention_residual),
            span(regions.residual_output),
            span(regions.attention_hc_norm),
            span(regions.attention_hc_down),
            span(regions.attention_hc_up),
            span(regions.attention_hc_inject),
            span(regions.mlp_hc_norm),
            span(regions.mlp_hc_down),
            span(regions.mlp_hc_up),
            span(regions.mlp_hc_inject),
            span(regions.hc_normalized),
            span(regions.hc_low_rank),
            span(regions.hc_mixed),
            span(regions.hc_write_gate),
            span(regions.gdn_input_weight),
            span(regions.gdn_control_weight),
            span(regions.gdn_convolution_weight),
            span(regions.gdn_a_log),
            span(regions.gdn_dt_bias),
            span(regions.gdn_norm),
            span(regions.gdn_output_weight),
            span(regions.gdn_projected),
            span(regions.gdn_convolved),
            span(regions.gdn_log_decay),
            span(regions.gdn_beta),
            span(regions.gdn_recurrent_plane),
            span(regions.gdn_recurrent_output),
            span(regions.state_rows),
            span(regions.router_weight),
            span(regions.expert_weight_scales_2),
            span(regions.shared_gate_weight),
            span(regions.shared_up_weight),
            span(regions.shared_down_weight),
            span(regions.shared_gate_logit_weight),
            span(regions.router_logits),
            span(regions.expert_indices),
            span(regions.routing_weights),
            span(regions.routed_intermediate),
            span(regions.routed_output),
            span(regions.shared_intermediate),
            span(regions.shared_output),
            span(regions.shared_gate_logit),
            span(regions.block_output),
        ];
        let gdn = persistent.gdn().unwrap();
        spans.push(span(gdn.history));
        spans.push(span(gdn.state));
        spans.push(span(persistent.ple().unwrap().conv_state));

        let engram = layout.ple().unwrap();
        spans.push(span(engram.key_proj));
        spans.push(span(engram.value_proj));
        spans.push(span(engram.norm_key));
        spans.push(span(engram.norm_query));
        spans.push(span(engram.norm_conv));
        spans.push(span(engram.convolution));
        spans.push(span(engram.codes));
        spans.push(span(engram.injected));
        spans.push(span(engram.embedding));
        spans.push(span(engram.key));
        spans.push(span(engram.key_normed));
        spans.push(span(engram.query_normed));
        spans.push(span(engram.value));
        spans.push(span(engram.gated));
        spans.push(span(engram.gated_normed));
        spans.push(span(engram.delta));

        spans.sort_unstable_by_key(|(offset, _)| *offset);
        for &(offset, bytes) in &spans {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in spans.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    #[test]
    fn the_staging_planes_match_the_hyper_connection_op_s_extents() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let regions = layout.regions();
        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;

        // launch_input_mix's caller-owned outputs, exactly as its safety contract states.
        assert_eq!(regions.hc_normalized.len(), rows * A::HC_WIDTH);
        assert_eq!(regions.hc_low_rank.len(), rows * A::HC_LOWRANK);
        assert_eq!(regions.hc_mixed.len(), rows * <A as Arch>::HIDDEN);
        assert_eq!(regions.hc_write_gate.len(), rows * A::HC_COUNT);
    }

    #[test]
    fn the_moe_planes_match_the_expert_dispatch_s_extents() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let regions = layout.regions();
        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;
        let top_k = A::NUM_EXPERTS_PER_TOKEN;

        assert_eq!(regions.router_logits.len(), rows * A::NUM_EXPERTS);
        assert_eq!(regions.expert_indices.len(), rows * top_k);
        assert_eq!(regions.routing_weights.len(), rows * top_k);
        assert_eq!(
            regions.routed_intermediate.len(),
            rows * top_k * <A as Arch>::INTERMEDIATE
        );
        assert_eq!(
            regions.routed_output.len(),
            rows * top_k * <A as Arch>::HIDDEN
        );
        assert_eq!(
            regions.shared_intermediate.len(),
            rows * A::SHARED_EXPERT_INTERMEDIATE
        );
        assert_eq!(regions.shared_output.len(), rows * <A as Arch>::HIDDEN);
        assert_eq!(regions.shared_gate_logit.len(), rows);
    }

    #[test]
    fn a_sparse_attention_layer_is_refused_by_this_owner() {
        for layer in [3, 7, 47] {
            let error = Qwen38FlashNextGdnMoeLayerLayout::build(layer).unwrap_err();
            assert!(error.to_string().contains("sparse-attention layer"));
        }
        assert!(Qwen38FlashNextGdnMoeLayerLayout::build(A::LAYERS).is_err());
    }

    #[test]
    fn every_gdn_layer_index_builds() {
        let built = (0..A::LAYERS)
            .filter(|layer| Qwen38FlashNextGdnMoeLayerLayout::build(*layer).is_ok())
            .count();

        assert_eq!(built, 36);
    }

    #[test]
    fn backbone_projection_planes_have_exact_routes() {
        let layout = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let regions = layout.regions();
        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;

        assert_eq!(regions.gdn_projected.len(), rows * A::GDN_INPUT_ROWS);
        assert_eq!(regions.block_output.len(), rows * <A as Arch>::HIDDEN);

        let entries = tuisko_kernels_sm120::kernel_ptx_names();
        let projections = entries
            .iter()
            .filter(|name| {
                name.starts_with("qwen38_flash_next_")
                    && (name.contains("in_proj")
                        || name.contains("out_proj")
                        || name.contains("_projection"))
            })
            .count();
        // Three decoder families, indexer QK, and the MTP fusion schedule.
        assert_eq!(projections, 36 + 12 + 5);

        for base in [
            "qwen38_flash_next_gdn_input_projection",
            "qwen38_flash_next_block_output_projection",
        ] {
            assert_eq!(
                entries.iter().filter(|name| name.starts_with(base)).count(),
                12,
                "{base} does not cover every admitted route"
            );
        }
    }

    #[test]
    fn builder_covers_each_composed_gdn_arena() {
        for layer in [0, A::PLE_LAYER] {
            let layout = Qwen38FlashNextGdnMoeLayerLayout::build(layer).unwrap();
            assert_eq!(layout.builder().byte_len(), layout.arena_bytes());
            assert!(layout.arena_bytes() >= layout.owner_bytes());
        }
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

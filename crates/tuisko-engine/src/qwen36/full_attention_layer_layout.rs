//! Single-allocation layout for one Qwen3.6 attention plus MoE decoder layer.

use crate::common::math::{product, sum};
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen36Moe35B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
const SLOTS_PER_TOKEN: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1;
pub(crate) const QWEN36_PHYSICAL_PAGES: usize = 24;
pub(crate) const QWEN36_TABLE_STRIDE: usize = 3;
pub(crate) const QWEN36_CONTEXT_CAPACITY: usize = QWEN36_TABLE_STRIDE * ATTENTION_PAGE_SIZE;
pub(crate) const QWEN36_PREFILL_TABLE_STRIDE: usize = QWEN36_PHYSICAL_PAGES;
pub(crate) const QWEN36_PREFILL_CONTEXT_CAPACITY: usize =
    QWEN36_PREFILL_TABLE_STRIDE * ATTENTION_PAGE_SIZE;
pub(crate) const QWEN36_MAX_ROWS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36FullAttentionLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) qkv_activation_codes: ArenaRegion<u8>,
    pub(crate) qkv_weight_codes: ArenaRegion<u8>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) query_norm: ArenaRegion<u16>,
    pub(crate) key_norm: ArenaRegion<u16>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,
    pub(crate) block_tables: ArenaRegion<u32>,
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) prefill_rope_cos: ArenaRegion<f32>,
    pub(crate) prefill_rope_sin: ArenaRegion<f32>,
    pub(crate) prefill_table_rows: ArenaRegion<u32>,
    pub(crate) prefill_cache_positions: ArenaRegion<u32>,
    pub(crate) prefill_lengths: ArenaRegion<u32>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) key_pages: ArenaRegion<u8>,
    pub(crate) value_pages: ArenaRegion<u8>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) output_activation: ArenaRegion<u16>,
    pub(crate) output_activation_codes: ArenaRegion<u8>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) moe_normalized: ArenaRegion<u16>,
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_gate_up_codes: ArenaRegion<u8>,
    pub(crate) routed_gate_up_scales: ArenaRegion<u8>,
    pub(crate) routed_gate_up_weight_scales_2: ArenaRegion<f32>,
    pub(crate) routed_down_codes: ArenaRegion<u8>,
    pub(crate) routed_down_scales: ArenaRegion<u8>,
    pub(crate) routed_down_weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate_up_codes: ArenaRegion<u8>,
    pub(crate) shared_gate_up_scales: ArenaRegion<u8>,
    pub(crate) shared_down_codes: ArenaRegion<u8>,
    pub(crate) shared_down_scales: ArenaRegion<u8>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) expert_intermediate: ArenaRegion<u16>,
    pub(crate) expert_output: ArenaRegion<u16>,
    pub(crate) shared_gate: ArenaRegion<u16>,
    pub(crate) moe_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked weights, E4M3 KV cache, and workspace for one Qwen3.6 attention layer.
#[derive(Clone, Debug)]
pub struct Qwen36FullAttentionLayerLayout {
    builder: ArenaLayout,
    regions: Qwen36FullAttentionLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen36FullAttentionLayerLayout {
    /// Reserves every source plane and exact decode/prefill seam through T=128.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen36Moe35B;
        require_geometry()?;

        let batch_hidden = product("Qwen3.6 attention row-hidden", QWEN36_MAX_ROWS, A::HIDDEN)?;
        let batch_qkv = product(
            "Qwen3.6 attention fused QKV",
            QWEN36_MAX_ROWS,
            A::ATTENTION_QKV_ROWS,
        )?;
        let batch_attention = product(
            "Qwen3.6 attention output",
            QWEN36_MAX_ROWS,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let cache_plane = product(
            "Qwen3.6 attention E4M3 cache plane",
            product(
                "Qwen3.6 attention cache page heads",
                QWEN36_PHYSICAL_PAGES,
                A::NUM_KV_HEADS,
            )?,
            product(
                "Qwen3.6 attention cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;
        let routed_gate_up_codes = product(
            "Qwen3.6 routed gate/up codes",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed gate/up rows",
                2 * A::INTERMEDIATE,
                A::HIDDEN / 2,
            )?,
        )?;
        let routed_gate_up_scales = product(
            "Qwen3.6 routed gate/up scales",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed gate/up scale rows",
                2 * A::INTERMEDIATE,
                A::HIDDEN / NVFP4_GROUP,
            )?,
        )?;
        let routed_down_codes = product(
            "Qwen3.6 routed down codes",
            A::NUM_EXPERTS,
            product("Qwen3.6 routed down rows", A::HIDDEN, A::INTERMEDIATE / 2)?,
        )?;
        let routed_down_scales = product(
            "Qwen3.6 routed down scales",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed down scale rows",
                A::HIDDEN,
                A::INTERMEDIATE / NVFP4_GROUP,
            )?,
        )?;
        let shared_gate_up_codes = product(
            "Qwen3.6 shared gate/up codes",
            2 * A::SHARED_EXPERT_INTERMEDIATE,
            A::HIDDEN / 2,
        )?;
        let shared_gate_up_scales = product(
            "Qwen3.6 shared gate/up scales",
            2 * A::SHARED_EXPERT_INTERMEDIATE,
            A::HIDDEN / NVFP4_GROUP,
        )?;
        let shared_down_codes = product(
            "Qwen3.6 shared down codes",
            A::HIDDEN,
            A::SHARED_EXPERT_INTERMEDIATE / 2,
        )?;
        let shared_down_scales = product(
            "Qwen3.6 shared down scales",
            A::HIDDEN,
            A::SHARED_EXPERT_INTERMEDIATE / NVFP4_GROUP,
        )?;
        let expert_intermediate = product(
            "Qwen3.6 expert intermediate",
            product("Qwen3.6 expert slots", QWEN36_MAX_ROWS, SLOTS_PER_TOKEN)?,
            A::INTERMEDIATE,
        )?;
        let expert_output = product(
            "Qwen3.6 expert output",
            product(
                "Qwen3.6 expert output slots",
                QWEN36_MAX_ROWS,
                SLOTS_PER_TOKEN,
            )?,
            A::HIDDEN,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen36FullAttentionLayerRegions {
            residual_input: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            qkv_activation_codes: builder.reserve(batch_hidden, ALIGNMENT)?,
            qkv_weight_codes: builder.reserve(A::ATTENTION_QKV_ROWS * A::HIDDEN, ALIGNMENT)?,
            qkv: builder.reserve(batch_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            block_tables: builder.reserve(MAX_BATCH * QWEN36_TABLE_STRIDE, ALIGNMENT)?,
            table_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            cache_positions: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            lengths: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            prefill_rope_cos: builder.reserve(QWEN36_MAX_ROWS * 32, ALIGNMENT)?,
            prefill_rope_sin: builder.reserve(QWEN36_MAX_ROWS * 32, ALIGNMENT)?,
            prefill_table_rows: builder.reserve(QWEN36_MAX_ROWS, ALIGNMENT)?,
            prefill_cache_positions: builder.reserve(QWEN36_MAX_ROWS, ALIGNMENT)?,
            prefill_lengths: builder.reserve(QWEN36_MAX_ROWS, ALIGNMENT)?,
            query: builder.reserve(batch_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(batch_attention, ALIGNMENT)?,
            output_activation: builder.reserve(batch_attention, ALIGNMENT)?,
            output_activation_codes: builder.reserve(batch_attention, ALIGNMENT)?,
            output_weight_codes: builder
                .reserve(A::HIDDEN * A::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?,
            mixer_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(batch_hidden, ALIGNMENT)?,
            moe_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            router_weight: builder.reserve(A::NUM_EXPERTS * A::HIDDEN, ALIGNMENT)?,
            router_logits: builder.reserve(QWEN36_MAX_ROWS * A::NUM_EXPERTS, ALIGNMENT)?,
            expert_indices: builder
                .reserve(QWEN36_MAX_ROWS * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routing_weights: builder
                .reserve(QWEN36_MAX_ROWS * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routed_gate_up_codes: builder.reserve(routed_gate_up_codes, ALIGNMENT)?,
            routed_gate_up_scales: builder.reserve(routed_gate_up_scales, ALIGNMENT)?,
            routed_gate_up_weight_scales_2: builder.reserve(A::NUM_EXPERTS, ALIGNMENT)?,
            routed_down_codes: builder.reserve(routed_down_codes, ALIGNMENT)?,
            routed_down_scales: builder.reserve(routed_down_scales, ALIGNMENT)?,
            routed_down_weight_scales_2: builder.reserve(A::NUM_EXPERTS, ALIGNMENT)?,
            shared_gate_up_codes: builder.reserve(shared_gate_up_codes, ALIGNMENT)?,
            shared_gate_up_scales: builder.reserve(shared_gate_up_scales, ALIGNMENT)?,
            shared_down_codes: builder.reserve(shared_down_codes, ALIGNMENT)?,
            shared_down_scales: builder.reserve(shared_down_scales, ALIGNMENT)?,
            shared_gate_weight: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            expert_intermediate: builder.reserve(expert_intermediate, ALIGNMENT)?,
            expert_output: builder.reserve(expert_output, ALIGNMENT)?,
            shared_gate: builder.reserve(QWEN36_MAX_ROWS, ALIGNMENT)?,
            moe_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.6 full-attention resident weights",
            &[
                regions.input_norm.byte_len(),
                regions.qkv_weight_codes.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.router_weight.byte_len(),
                regions.routed_gate_up_codes.byte_len(),
                regions.routed_gate_up_scales.byte_len(),
                regions.routed_gate_up_weight_scales_2.byte_len(),
                regions.routed_down_codes.byte_len(),
                regions.routed_down_scales.byte_len(),
                regions.routed_down_weight_scales_2.byte_len(),
                regions.shared_gate_up_codes.byte_len(),
                regions.shared_gate_up_scales.byte_len(),
                regions.shared_down_codes.byte_len(),
                regions.shared_down_scales.byte_len(),
                regions.shared_gate_weight.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "Qwen3.6 full-attention E4M3 cache",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "Qwen3.6 full-attention workspace",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.qkv_activation_codes.byte_len(),
                regions.qkv.byte_len(),
                regions.rope_cos.byte_len(),
                regions.rope_sin.byte_len(),
                regions.block_tables.byte_len(),
                regions.table_rows.byte_len(),
                regions.cache_positions.byte_len(),
                regions.lengths.byte_len(),
                regions.prefill_rope_cos.byte_len(),
                regions.prefill_rope_sin.byte_len(),
                regions.prefill_table_rows.byte_len(),
                regions.prefill_cache_positions.byte_len(),
                regions.prefill_lengths.byte_len(),
                regions.query.byte_len(),
                regions.attention.byte_len(),
                regions.output_activation.byte_len(),
                regions.output_activation_codes.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.moe_normalized.byte_len(),
                regions.router_logits.byte_len(),
                regions.expert_indices.byte_len(),
                regions.routing_weights.byte_len(),
                regions.expert_intermediate.byte_len(),
                regions.expert_output.byte_len(),
                regions.shared_gate.byte_len(),
                regions.moe_branch.byte_len(),
                regions.residual_output.byte_len(),
                regions.next_normalized.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            regions,
            resident_weight_bytes,
            cache_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn regions(&self) -> Qwen36FullAttentionLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed device weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact represented E4M3 key/value cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Exact address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights, cache, and workspace without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Fixed per-slot short-context capacity of the decode owner.
    pub const fn context_capacity(&self) -> usize {
        QWEN36_CONTEXT_CAPACITY
    }

    /// Largest exact row route owned by the layer.
    pub const fn row_capacity(&self) -> usize {
        QWEN36_MAX_ROWS
    }

    /// Shared from-empty prompt cache capacity.
    pub const fn prefill_context_capacity(&self) -> usize {
        QWEN36_PREFILL_CONTEXT_CAPACITY
    }
}

impl LayerMemoryLayout for Qwen36FullAttentionLayerLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes()
    }

    fn cache_bytes(&self) -> usize {
        self.cache_bytes()
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes()
    }
}

fn require_geometry() -> EngineResult<()> {
    type A = Qwen36Moe35B;
    if A::ATTENTION_QKV_ROWS != 9_216
        || A::ATTENTION_OUTPUT_COLUMNS != 4_096
        || A::NUM_ATTENTION_HEADS != 16
        || A::NUM_KV_HEADS != 2
        || A::HEAD_DIM != 256
        || !A::HIDDEN.is_multiple_of(NVFP4_GROUP)
        || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP)
        || A::SHARED_EXPERT_INTERMEDIATE != A::INTERMEDIATE
        || A::NUM_EXPERTS_PER_TOKEN != 8
    {
        return Err(EngineError::layout(
            "Qwen3.6 attention/MoE geometry differs from the qualified layer contract",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_accounting_and_geometry_are_exact() {
        let layout = Qwen36FullAttentionLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 483_085_312);
        assert_eq!(layout.cache_bytes(), 1_572_864);
        assert_eq!(layout.workspace_bytes(), 18_587_584);
        assert_eq!(layout.owner_bytes(), 503_245_760);
        assert_eq!(layout.arena_bytes(), 503_246_592);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 832);
        assert_eq!(layout.context_capacity(), 192);
        assert_eq!(layout.row_capacity(), 128);
        assert_eq!(layout.prefill_context_capacity(), 1_536);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen36FullAttentionLayerLayout::build().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
            span(regions.qkv_activation_codes),
            span(regions.qkv_weight_codes),
            span(regions.qkv),
            span(regions.query_norm),
            span(regions.key_norm),
            span(regions.rope_cos),
            span(regions.rope_sin),
            span(regions.block_tables),
            span(regions.table_rows),
            span(regions.cache_positions),
            span(regions.lengths),
            span(regions.prefill_rope_cos),
            span(regions.prefill_rope_sin),
            span(regions.prefill_table_rows),
            span(regions.prefill_cache_positions),
            span(regions.prefill_lengths),
            span(regions.query),
            span(regions.key_pages),
            span(regions.value_pages),
            span(regions.attention),
            span(regions.output_activation),
            span(regions.output_activation_codes),
            span(regions.output_weight_codes),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.moe_normalized),
            span(regions.router_weight),
            span(regions.router_logits),
            span(regions.expert_indices),
            span(regions.routing_weights),
            span(regions.routed_gate_up_codes),
            span(regions.routed_gate_up_scales),
            span(regions.routed_gate_up_weight_scales_2),
            span(regions.routed_down_codes),
            span(regions.routed_down_scales),
            span(regions.routed_down_weight_scales_2),
            span(regions.shared_gate_up_codes),
            span(regions.shared_gate_up_scales),
            span(regions.shared_down_codes),
            span(regions.shared_down_scales),
            span(regions.shared_gate_weight),
            span(regions.expert_intermediate),
            span(regions.expert_output),
            span(regions.shared_gate),
            span(regions.moe_branch),
            span(regions.next_norm),
            span(regions.residual_output),
            span(regions.next_normalized),
        ];
        spans.sort_unstable_by_key(|(offset, _)| *offset);
        for &(offset, bytes) in &spans {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in spans.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

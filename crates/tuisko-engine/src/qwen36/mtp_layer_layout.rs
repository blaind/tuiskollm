//! Single-allocation layout for the source-BF16 Qwen3.6 MTP layer.

use crate::common::math::{product, sum};
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen36Moe35B};

const ALIGNMENT: usize = 256;
const SLOTS_PER_TOKEN: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1;
pub(crate) const QWEN36_MTP_PROMPT_ROWS: usize = 128;
pub(crate) const QWEN36_MTP_PHYSICAL_PAGES: usize = 24;
pub(crate) const QWEN36_MTP_TABLE_STRIDE: usize = QWEN36_MTP_PHYSICAL_PAGES / MAX_BATCH;
pub(crate) const QWEN36_MTP_CONTEXT_CAPACITY: usize = QWEN36_MTP_TABLE_STRIDE * ATTENTION_PAGE_SIZE;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36MtpLayerRegions {
    pub(crate) embedding: ArenaRegion<u16>,
    pub(crate) target_hidden: ArenaRegion<u16>,
    pub(crate) embedding_norm: ArenaRegion<u16>,
    pub(crate) hidden_norm: ArenaRegion<u16>,
    pub(crate) normalized_embedding: ArenaRegion<u16>,
    pub(crate) normalized_hidden: ArenaRegion<u16>,
    pub(crate) input_projection: ArenaRegion<u16>,
    pub(crate) residual: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) attention_normalized: ArenaRegion<u16>,
    pub(crate) qkv_weight: ArenaRegion<u16>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) query_norm: ArenaRegion<u16>,
    pub(crate) key_norm: ArenaRegion<u16>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,
    pub(crate) block_tables: ArenaRegion<u32>,
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) key_pages: ArenaRegion<u8>,
    pub(crate) value_pages: ArenaRegion<u8>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_activation: ArenaRegion<u16>,
    pub(crate) attention_output_weight: ArenaRegion<u16>,
    pub(crate) attention_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) post_attention_residual: ArenaRegion<u16>,
    pub(crate) moe_normalized: ArenaRegion<u16>,
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_gate_up_weight: ArenaRegion<u16>,
    pub(crate) routed_down_weight: ArenaRegion<u16>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) shared_up_weight: ArenaRegion<u16>,
    pub(crate) shared_down_weight: ArenaRegion<u16>,
    pub(crate) shared_expert_gate_weight: ArenaRegion<u16>,
    pub(crate) expert_intermediate: ArenaRegion<u16>,
    pub(crate) expert_output: ArenaRegion<u16>,
    pub(crate) shared_gate_output: ArenaRegion<u16>,
    pub(crate) moe_branch: ArenaRegion<u16>,
    pub(crate) final_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) final_normalized: ArenaRegion<u16>,
}

/// Checked Qwen3.6 MTP weights, short cache, and graph workspace.
#[derive(Clone, Debug)]
pub struct Qwen36MtpLayerLayout {
    builder: ArenaLayout,
    regions: Qwen36MtpLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen36MtpLayerLayout {
    /// Reserves exact decode, causal realignment, and `T=32,64,128` prime workspace.
    pub fn build() -> EngineResult<Self> {
        Self::build_with_cache(QWEN36_MTP_PHYSICAL_PAGES)
    }

    #[cfg(test)]
    fn build_for_external_cache() -> EngineResult<Self> {
        Self::build_with_cache(0)
    }

    fn build_with_cache(cache_pages: usize) -> EngineResult<Self> {
        type A = Qwen36Moe35B;
        require_geometry()?;

        let prompt_hidden = product(
            "Qwen3.6 MTP prompt hidden",
            QWEN36_MTP_PROMPT_ROWS,
            A::HIDDEN,
        )?;
        let prompt_qkv = product(
            "Qwen3.6 MTP prompt QKV",
            QWEN36_MTP_PROMPT_ROWS,
            A::ATTENTION_QKV_ROWS,
        )?;
        let prompt_attention = product(
            "Qwen3.6 MTP prompt attention",
            QWEN36_MTP_PROMPT_ROWS,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_attention = product(
            "Qwen3.6 MTP batch attention",
            MAX_BATCH,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let input_projection = product(
            "Qwen3.6 MTP input projection",
            A::HIDDEN,
            product("Qwen3.6 MTP input projection columns", 2, A::HIDDEN)?,
        )?;
        let qkv_weight = product("Qwen3.6 MTP QKV weights", A::ATTENTION_QKV_ROWS, A::HIDDEN)?;
        let attention_output_weight = product(
            "Qwen3.6 MTP attention output weights",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let router_weight = product("Qwen3.6 MTP router weights", A::NUM_EXPERTS, A::HIDDEN)?;
        let routed_gate_up_weight = product(
            "Qwen3.6 MTP routed gate/up weights",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 MTP routed gate/up expert weights",
                2 * A::INTERMEDIATE,
                A::HIDDEN,
            )?,
        )?;
        let routed_down_weight = product(
            "Qwen3.6 MTP routed down weights",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 MTP routed down expert weights",
                A::HIDDEN,
                A::INTERMEDIATE,
            )?,
        )?;
        let shared_projection = product(
            "Qwen3.6 MTP shared expert weights",
            A::INTERMEDIATE,
            A::HIDDEN,
        )?;
        let expert_intermediate = product(
            "Qwen3.6 MTP expert intermediate",
            product("Qwen3.6 MTP expert slots", MAX_BATCH, SLOTS_PER_TOKEN)?,
            A::INTERMEDIATE,
        )?;
        let expert_output = product(
            "Qwen3.6 MTP expert output",
            product(
                "Qwen3.6 MTP expert output slots",
                MAX_BATCH,
                SLOTS_PER_TOKEN,
            )?,
            A::HIDDEN,
        )?;
        let cache_plane = product(
            "Qwen3.6 MTP E4M3 cache plane",
            product("Qwen3.6 MTP cache page heads", cache_pages, A::NUM_KV_HEADS)?,
            product(
                "Qwen3.6 MTP cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen36MtpLayerRegions {
            embedding: builder.reserve(prompt_hidden, ALIGNMENT)?,
            target_hidden: builder.reserve(prompt_hidden, ALIGNMENT)?,
            embedding_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            hidden_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            normalized_embedding: builder.reserve(prompt_hidden, ALIGNMENT)?,
            normalized_hidden: builder.reserve(prompt_hidden, ALIGNMENT)?,
            input_projection: builder.reserve(input_projection, ALIGNMENT)?,
            residual: builder.reserve(prompt_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            attention_normalized: builder.reserve(prompt_hidden, ALIGNMENT)?,
            qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
            qkv: builder.reserve(prompt_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(QWEN36_MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(QWEN36_MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            block_tables: builder.reserve(QWEN36_MTP_PHYSICAL_PAGES, ALIGNMENT)?,
            table_rows: builder.reserve(QWEN36_MTP_PROMPT_ROWS, ALIGNMENT)?,
            cache_positions: builder.reserve(QWEN36_MTP_PROMPT_ROWS, ALIGNMENT)?,
            lengths: builder.reserve(QWEN36_MTP_PROMPT_ROWS, ALIGNMENT)?,
            query: builder.reserve(prompt_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(batch_attention, ALIGNMENT)?,
            attention_activation: builder.reserve(batch_attention, ALIGNMENT)?,
            attention_output_weight: builder.reserve(attention_output_weight, ALIGNMENT)?,
            attention_branch: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            post_attention_residual: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
            moe_normalized: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
            router_weight: builder.reserve(router_weight, ALIGNMENT)?,
            router_logits: builder.reserve(MAX_BATCH * A::NUM_EXPERTS, ALIGNMENT)?,
            expert_indices: builder.reserve(MAX_BATCH * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routing_weights: builder.reserve(MAX_BATCH * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routed_gate_up_weight: builder.reserve(routed_gate_up_weight, ALIGNMENT)?,
            routed_down_weight: builder.reserve(routed_down_weight, ALIGNMENT)?,
            shared_gate_weight: builder.reserve(shared_projection, ALIGNMENT)?,
            shared_up_weight: builder.reserve(shared_projection, ALIGNMENT)?,
            shared_down_weight: builder.reserve(shared_projection, ALIGNMENT)?,
            shared_expert_gate_weight: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            expert_intermediate: builder.reserve(expert_intermediate, ALIGNMENT)?,
            expert_output: builder.reserve(expert_output, ALIGNMENT)?,
            shared_gate_output: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            moe_branch: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
            final_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
            final_normalized: builder.reserve(MAX_BATCH * A::HIDDEN, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.6 MTP represented weight bytes",
            &[
                regions.embedding_norm.byte_len(),
                regions.hidden_norm.byte_len(),
                regions.input_projection.byte_len(),
                regions.input_norm.byte_len(),
                regions.qkv_weight.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
                regions.attention_output_weight.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.router_weight.byte_len(),
                regions.routed_gate_up_weight.byte_len(),
                regions.routed_down_weight.byte_len(),
                regions.shared_gate_weight.byte_len(),
                regions.shared_up_weight.byte_len(),
                regions.shared_down_weight.byte_len(),
                regions.shared_expert_gate_weight.byte_len(),
                regions.final_norm.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "Qwen3.6 MTP represented cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "Qwen3.6 MTP address-stable workspace bytes",
            &[
                regions.embedding.byte_len(),
                regions.target_hidden.byte_len(),
                regions.normalized_embedding.byte_len(),
                regions.normalized_hidden.byte_len(),
                regions.residual.byte_len(),
                regions.attention_normalized.byte_len(),
                regions.qkv.byte_len(),
                regions.rope_cos.byte_len(),
                regions.rope_sin.byte_len(),
                regions.block_tables.byte_len(),
                regions.table_rows.byte_len(),
                regions.cache_positions.byte_len(),
                regions.lengths.byte_len(),
                regions.query.byte_len(),
                regions.attention.byte_len(),
                regions.attention_activation.byte_len(),
                regions.attention_branch.byte_len(),
                regions.post_attention_residual.byte_len(),
                regions.moe_normalized.byte_len(),
                regions.router_logits.byte_len(),
                regions.expert_indices.byte_len(),
                regions.routing_weights.byte_len(),
                regions.expert_intermediate.byte_len(),
                regions.expert_output.byte_len(),
                regions.shared_gate_output.byte_len(),
                regions.moe_branch.byte_len(),
                regions.residual_output.byte_len(),
                regions.final_normalized.byte_len(),
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

    pub(crate) const fn regions(&self) -> Qwen36MtpLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact unchanged source-BF16 MTP weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact represented E4M3 key/value cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Weights, cache, and workspace without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Per-slot capacity of the isolated short-context owner.
    pub const fn context_capacity(&self) -> usize {
        QWEN36_MTP_CONTEXT_CAPACITY
    }
}

impl LayerMemoryLayout for Qwen36MtpLayerLayout {
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
    if A::HIDDEN != 2_048
        || A::INTERMEDIATE != 512
        || A::NUM_EXPERTS != 256
        || A::NUM_EXPERTS_PER_TOKEN != 8
        || A::ATTENTION_QKV_ROWS != 9_216
        || A::ATTENTION_OUTPUT_COLUMNS != 4_096
        || A::NUM_KV_HEADS != 2
        || A::HEAD_DIM != 256
    {
        return Err(EngineError::layout(
            "Qwen3.6 MTP geometry differs from the qualified layer contract",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_accounting_and_geometry_are_exact() {
        let layout = Qwen36MtpLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 1_689_281_536);
        assert_eq!(layout.cache_bytes(), 1_572_864);
        assert_eq!(layout.workspace_bytes(), 8_402_800);
        assert_eq!(layout.owner_bytes(), 1_699_257_200);
        assert_eq!(layout.arena_bytes(), 1_699_257_856);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 656);
        assert_eq!(layout.context_capacity(), 192);
        assert_eq!(QWEN36_MTP_PROMPT_ROWS, 128);
    }

    #[test]
    fn external_cache_layout_owns_no_duplicate_cache_plane() {
        let layout = Qwen36MtpLayerLayout::build_for_external_cache().unwrap();

        assert_eq!(layout.cache_bytes(), 0);
        assert_eq!(layout.resident_weight_bytes(), 1_689_281_536);
        assert_eq!(layout.owner_bytes(), 1_697_684_336);
        assert_eq!(layout.arena_bytes(), 1_697_684_992);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_complete() {
        let layout = Qwen36MtpLayerLayout::build().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.embedding),
            span(regions.target_hidden),
            span(regions.embedding_norm),
            span(regions.hidden_norm),
            span(regions.normalized_embedding),
            span(regions.normalized_hidden),
            span(regions.input_projection),
            span(regions.residual),
            span(regions.input_norm),
            span(regions.attention_normalized),
            span(regions.qkv_weight),
            span(regions.qkv),
            span(regions.query_norm),
            span(regions.key_norm),
            span(regions.rope_cos),
            span(regions.rope_sin),
            span(regions.block_tables),
            span(regions.table_rows),
            span(regions.cache_positions),
            span(regions.lengths),
            span(regions.query),
            span(regions.key_pages),
            span(regions.value_pages),
            span(regions.attention),
            span(regions.attention_activation),
            span(regions.attention_output_weight),
            span(regions.attention_branch),
            span(regions.post_attention_norm),
            span(regions.post_attention_residual),
            span(regions.moe_normalized),
            span(regions.router_weight),
            span(regions.router_logits),
            span(regions.expert_indices),
            span(regions.routing_weights),
            span(regions.routed_gate_up_weight),
            span(regions.routed_down_weight),
            span(regions.shared_gate_weight),
            span(regions.shared_up_weight),
            span(regions.shared_down_weight),
            span(regions.shared_expert_gate_weight),
            span(regions.expert_intermediate),
            span(regions.expert_output),
            span(regions.shared_gate_output),
            span(regions.moe_branch),
            span(regions.final_norm),
            span(regions.residual_output),
            span(regions.final_normalized),
        ];
        spans.sort_unstable_by_key(|(offset, _)| *offset);
        for &(offset, bytes) in &spans {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in spans.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
        let (offset, bytes) = spans.last().copied().unwrap();
        assert_eq!(offset + bytes, layout.arena_bytes());
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

//! Single-allocation layout for the source-BF16 Qwen3.8 MTP layer.

use crate::common::math::{product, sum};
use crate::{EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::Arch;

const ALIGNMENT: usize = 256;
pub(crate) const PHYSICAL_PAGES: usize = 24;
pub(crate) const TABLE_STRIDE: usize = PHYSICAL_PAGES / MAX_BATCH;
pub(crate) const CONTEXT_CAPACITY: usize = TABLE_STRIDE * ATTENTION_PAGE_SIZE;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpLayerRegions {
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
    pub(crate) key_pages: ArenaRegion<u16>,
    pub(crate) value_pages: ArenaRegion<u16>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_activation: ArenaRegion<u16>,
    pub(crate) attention_output_weight: ArenaRegion<u16>,
    pub(crate) attention_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) post_attention_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) gate_up_weight: ArenaRegion<u16>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) down_weight: ArenaRegion<u16>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) final_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) final_normalized: ArenaRegion<u16>,
    pub(crate) lm_head_activation_codes: ArenaRegion<u8>,
    pub(crate) lm_head_activation_scales: ArenaRegion<f32>,
    pub(crate) lm_head_codes: ArenaRegion<u8>,
    pub(crate) lm_head_scales: ArenaRegion<u16>,
    pub(crate) logits: ArenaRegion<u16>,
}

/// Checked source weights, short qualification cache, and stable graph workspace.
#[derive(Clone, Debug)]
pub struct MtpLayerLayout {
    builder: ArenaLayout,
    regions: MtpLayerRegions,
    mtp_weight_bytes: usize,
    shared_endpoint_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl MtpLayerLayout {
    /// Reserves the complete exact `B=1..=8` and causal `K=1..=4` owner.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        let row_hidden = product("MTP row-hidden elements", MAX_BATCH, A::HIDDEN)?;
        let row_qkv = product("MTP row-QKV elements", MAX_BATCH, A::ATTENTION_QKV_ROWS)?;
        let row_attention = product(
            "MTP row-attention elements",
            MAX_BATCH,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let row_intermediate =
            product("MTP row-intermediate elements", MAX_BATCH, A::INTERMEDIATE)?;
        let input_projection = product(
            "MTP input projection elements",
            A::HIDDEN,
            product("MTP input projection columns", 2, A::HIDDEN)?,
        )?;
        let qkv_weight = product("MTP QKV weight elements", A::ATTENTION_QKV_ROWS, A::HIDDEN)?;
        let attention_output_weight = product(
            "MTP attention-output weight elements",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let gate_up_weight = product(
            "MTP gate/up weight elements",
            product("MTP gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weight = product("MTP down weight elements", A::HIDDEN, A::INTERMEDIATE)?;
        let lm_head_weight = product("shared LM-head code elements", A::VOCAB, A::HIDDEN)?;
        let cache_plane = product(
            "MTP cache plane elements",
            product("MTP cache page heads", PHYSICAL_PAGES, A::NUM_KV_HEADS)?,
            product("MTP cache page values", ATTENTION_PAGE_SIZE, A::HEAD_DIM)?,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = MtpLayerRegions {
            embedding: builder.reserve(row_hidden, ALIGNMENT)?,
            target_hidden: builder.reserve(row_hidden, ALIGNMENT)?,
            embedding_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            hidden_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            normalized_embedding: builder.reserve(row_hidden, ALIGNMENT)?,
            normalized_hidden: builder.reserve(row_hidden, ALIGNMENT)?,
            input_projection: builder.reserve(input_projection, ALIGNMENT)?,
            residual: builder.reserve(row_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            attention_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
            qkv: builder.reserve(row_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            block_tables: builder.reserve(PHYSICAL_PAGES, ALIGNMENT)?,
            table_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            cache_positions: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            lengths: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            query: builder.reserve(row_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(row_attention, ALIGNMENT)?,
            attention_activation: builder.reserve(row_attention, ALIGNMENT)?,
            attention_output_weight: builder.reserve(attention_output_weight, ALIGNMENT)?,
            attention_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            post_attention_residual: builder.reserve(row_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            gate_up_weight: builder.reserve(gate_up_weight, ALIGNMENT)?,
            swiglu: builder.reserve(row_intermediate, ALIGNMENT)?,
            down_weight: builder.reserve(down_weight, ALIGNMENT)?,
            mlp_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            final_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(row_hidden, ALIGNMENT)?,
            final_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            lm_head_activation_codes: builder.reserve(row_hidden, ALIGNMENT)?,
            lm_head_activation_scales: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            lm_head_codes: builder.reserve(lm_head_weight, ALIGNMENT)?,
            lm_head_scales: builder.reserve(A::VOCAB, ALIGNMENT)?,
            logits: builder.reserve(
                product("MTP logit elements", MAX_BATCH, A::VOCAB)?,
                ALIGNMENT,
            )?,
        };
        let mtp_weight_bytes = sum(
            "MTP represented weight bytes",
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
                regions.gate_up_weight.byte_len(),
                regions.down_weight.byte_len(),
                regions.final_norm.byte_len(),
            ],
        )?;
        let shared_endpoint_weight_bytes = sum(
            "MTP shared endpoint weight bytes",
            &[
                regions.lm_head_codes.byte_len(),
                regions.lm_head_scales.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "MTP represented cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "MTP address-stable workspace bytes",
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
                regions.mlp_normalized.byte_len(),
                regions.swiglu.byte_len(),
                regions.mlp_branch.byte_len(),
                regions.residual_output.byte_len(),
                regions.final_normalized.byte_len(),
                regions.lm_head_activation_codes.byte_len(),
                regions.lm_head_activation_scales.byte_len(),
                regions.logits.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            regions,
            mtp_weight_bytes,
            shared_endpoint_weight_bytes,
            cache_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn regions(&self) -> MtpLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact unchanged BF16 MTP source bytes.
    pub const fn mtp_weight_bytes(&self) -> usize {
        self.mtp_weight_bytes
    }

    /// One source-native FP8 LM head, shared with target endpoint composition.
    pub const fn shared_endpoint_weight_bytes(&self) -> usize {
        self.shared_endpoint_weight_bytes
    }

    /// MTP and shared endpoint weights resident in this isolated owner.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.mtp_weight_bytes + self.shared_endpoint_weight_bytes
    }

    /// Exact represented BF16 key/value cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable non-cache workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Weights, cache, and workspace without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes() + self.cache_bytes + self.workspace_bytes
    }

    /// Per-slot short-context capacity of the isolated qualification owner.
    pub const fn context_capacity(&self) -> usize {
        CONTEXT_CAPACITY
    }
}

impl LayerMemoryLayout for MtpLayerLayout {
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

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, CONTEXT_CAPACITY, MtpLayerLayout};
    use tuisko_model::Qwen38_27B;

    #[test]
    fn qwen_mtp_layer_byte_accounting_is_exact() {
        let layout = MtpLayerLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.mtp_weight_bytes(), 849_398_784);
        assert_eq!(layout.shared_endpoint_weight_bytes(), 1_271_895_040);
        assert_eq!(layout.resident_weight_bytes(), 2_121_293_824);
        assert_eq!(layout.cache_bytes(), 6_291_456);
        assert_eq!(layout.workspace_bytes(), 5_998_816);
        assert_eq!(layout.owner_bytes(), 2_133_584_096);
        assert_eq!(layout.arena_bytes(), 2_133_585_152);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 1_056);
        assert_eq!(layout.context_capacity(), CONTEXT_CAPACITY);
        assert_eq!(layout.context_capacity(), 192);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = MtpLayerLayout::build::<Qwen38_27B>().unwrap();
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
            span(regions.mlp_normalized),
            span(regions.gate_up_weight),
            span(regions.swiglu),
            span(regions.down_weight),
            span(regions.mlp_branch),
            span(regions.final_norm),
            span(regions.residual_output),
            span(regions.final_normalized),
            span(regions.lm_head_activation_codes),
            span(regions.lm_head_activation_scales),
            span(regions.lm_head_codes),
            span(regions.lm_head_scales),
            span(regions.logits),
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

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

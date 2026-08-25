//! Single-allocation layout for the source-BF16 Qwen3.5 MTP layer.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;
pub(crate) const QWEN35_MTP_PROMPT_ROWS: usize = 128;
pub(crate) const QWEN35_MTP_PHYSICAL_PAGES: usize = 24;
pub(crate) const QWEN35_MTP_TABLE_STRIDE: usize = QWEN35_MTP_PHYSICAL_PAGES / MAX_BATCH;
pub(crate) const QWEN35_MTP_CONTEXT_CAPACITY: usize = QWEN35_MTP_TABLE_STRIDE * ATTENTION_PAGE_SIZE;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35MtpLayerRegions {
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
}

/// Checked Qwen3.5 MTP weights, short cache, and graph workspace.
#[derive(Clone, Debug)]
pub struct Qwen35MtpLayerLayout {
    builder: ArenaLayout,
    regions: Qwen35MtpLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen35MtpLayerLayout {
    /// Reserves exact decode, causal realignment, and `T=32,64,128` prime workspace.
    pub fn build() -> EngineResult<Self> {
        Self::build_with_cache(QWEN35_MTP_PHYSICAL_PAGES)
    }

    pub(crate) fn build_for_external_cache() -> EngineResult<Self> {
        Self::build_with_cache(0)
    }

    fn build_with_cache(cache_pages: usize) -> EngineResult<Self> {
        let prompt_hidden = product(
            "Qwen3.5 MTP prompt-hidden elements",
            QWEN35_MTP_PROMPT_ROWS,
            Qwen35_9B::HIDDEN,
        )?;
        let prompt_qkv = product(
            "Qwen3.5 MTP prompt-QKV elements",
            QWEN35_MTP_PROMPT_ROWS,
            Qwen35_9B::ATTENTION_QKV_ROWS,
        )?;
        let prompt_attention = product(
            "Qwen3.5 MTP prompt-attention elements",
            QWEN35_MTP_PROMPT_ROWS,
            Qwen35_9B::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_attention = product(
            "Qwen3.5 MTP batch-attention elements",
            MAX_BATCH,
            Qwen35_9B::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let row_intermediate = product(
            "Qwen3.5 MTP row-intermediate elements",
            MAX_BATCH,
            Qwen35_9B::INTERMEDIATE,
        )?;
        let input_projection = product(
            "Qwen3.5 MTP input projection elements",
            Qwen35_9B::HIDDEN,
            product("Qwen3.5 MTP input projection columns", 2, Qwen35_9B::HIDDEN)?,
        )?;
        let qkv_weight = product(
            "Qwen3.5 MTP QKV weight elements",
            Qwen35_9B::ATTENTION_QKV_ROWS,
            Qwen35_9B::HIDDEN,
        )?;
        let attention_output_weight = product(
            "Qwen3.5 MTP attention-output weight elements",
            Qwen35_9B::HIDDEN,
            Qwen35_9B::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let gate_up_weight = product(
            "Qwen3.5 MTP gate/up weight elements",
            product("Qwen3.5 MTP gate/up rows", 2, Qwen35_9B::INTERMEDIATE)?,
            Qwen35_9B::HIDDEN,
        )?;
        let down_weight = product(
            "Qwen3.5 MTP down weight elements",
            Qwen35_9B::HIDDEN,
            Qwen35_9B::INTERMEDIATE,
        )?;
        let cache_plane = product(
            "Qwen3.5 MTP cache plane elements",
            product(
                "Qwen3.5 MTP cache page heads",
                cache_pages,
                Qwen35_9B::NUM_KV_HEADS,
            )?,
            product(
                "Qwen3.5 MTP cache page values",
                ATTENTION_PAGE_SIZE,
                Qwen35_9B::HEAD_DIM,
            )?,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen35MtpLayerRegions {
            embedding: builder.reserve(prompt_hidden, ALIGNMENT)?,
            target_hidden: builder.reserve(prompt_hidden, ALIGNMENT)?,
            embedding_norm: builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?,
            hidden_norm: builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?,
            normalized_embedding: builder.reserve(prompt_hidden, ALIGNMENT)?,
            normalized_hidden: builder.reserve(prompt_hidden, ALIGNMENT)?,
            input_projection: builder.reserve(input_projection, ALIGNMENT)?,
            residual: builder.reserve(prompt_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?,
            attention_normalized: builder.reserve(prompt_hidden, ALIGNMENT)?,
            qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
            qkv: builder.reserve(prompt_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(Qwen35_9B::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(Qwen35_9B::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(QWEN35_MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(QWEN35_MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            block_tables: builder.reserve(QWEN35_MTP_PHYSICAL_PAGES, ALIGNMENT)?,
            table_rows: builder.reserve(QWEN35_MTP_PROMPT_ROWS, ALIGNMENT)?,
            cache_positions: builder.reserve(QWEN35_MTP_PROMPT_ROWS, ALIGNMENT)?,
            lengths: builder.reserve(QWEN35_MTP_PROMPT_ROWS, ALIGNMENT)?,
            query: builder.reserve(prompt_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(batch_attention, ALIGNMENT)?,
            attention_activation: builder.reserve(batch_attention, ALIGNMENT)?,
            attention_output_weight: builder.reserve(attention_output_weight, ALIGNMENT)?,
            attention_branch: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
            post_attention_norm: builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?,
            post_attention_residual: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
            mlp_normalized: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
            gate_up_weight: builder.reserve(gate_up_weight, ALIGNMENT)?,
            swiglu: builder.reserve(row_intermediate, ALIGNMENT)?,
            down_weight: builder.reserve(down_weight, ALIGNMENT)?,
            mlp_branch: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
            final_norm: builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
            final_normalized: builder.reserve(MAX_BATCH * Qwen35_9B::HIDDEN, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.5 MTP represented weight bytes",
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
        let cache_bytes = sum(
            "Qwen3.5 MTP represented cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "Qwen3.5 MTP address-stable workspace bytes",
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

    pub(crate) const fn regions(&self) -> Qwen35MtpLayerRegions {
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
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Per-slot short-context capacity of this isolated qualification owner.
    pub const fn context_capacity(&self) -> usize {
        QWEN35_MTP_CONTEXT_CAPACITY
    }
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn sum(name: &str, values: &[usize]) -> EngineResult<usize> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ALIGNMENT, QWEN35_MTP_CONTEXT_CAPACITY, QWEN35_MTP_PROMPT_ROWS, Qwen35MtpLayerLayout,
    };

    #[test]
    fn qwen35_mtp_layer_byte_accounting_is_exact() {
        let layout = Qwen35MtpLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 486_581_248);
        assert_eq!(layout.cache_bytes(), 6_291_456);
        assert_eq!(layout.workspace_bytes(), 11_830_880);
        assert_eq!(layout.owner_bytes(), 504_703_584);
        assert_eq!(layout.arena_bytes(), 504_703_744);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 160);
        assert_eq!(layout.context_capacity(), QWEN35_MTP_CONTEXT_CAPACITY);
        assert_eq!(layout.context_capacity(), 192);
        assert_eq!(QWEN35_MTP_PROMPT_ROWS, 128);
    }

    #[test]
    fn external_cache_layout_owns_no_duplicate_cache_plane() {
        let layout = Qwen35MtpLayerLayout::build_for_external_cache().unwrap();

        assert_eq!(layout.cache_bytes(), 0);
        assert_eq!(layout.owner_bytes(), 498_412_128);
        assert_eq!(layout.arena_bytes(), 498_412_288);
    }

    #[test]
    fn qwen35_mtp_regions_are_aligned_disjoint_and_complete() {
        let layout = Qwen35MtpLayerLayout::build().unwrap();
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

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

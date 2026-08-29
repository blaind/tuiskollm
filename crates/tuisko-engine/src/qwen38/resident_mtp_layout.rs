//! Long-context resident layout for the exact Qwen3.8 MTP layer.

use crate::common::math::{product, sum};
use crate::{EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, LONG_CONTEXT_GQA_MAX_PARTITIONS};
use tuisko_model::{Arch, Qwen38_27B};

const ALIGNMENT: usize = 256;
pub(crate) const MTP_PROMPT_ROWS: usize = 1_024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentMtpRegions {
    pub(crate) embedding_norm: ArenaRegion<u16>,
    pub(crate) hidden_norm: ArenaRegion<u16>,
    pub(crate) input_projection: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) qkv_weight: ArenaRegion<u16>,
    pub(crate) query_norm: ArenaRegion<u16>,
    pub(crate) key_norm: ArenaRegion<u16>,
    pub(crate) attention_output_weight: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) gate_up_weight: ArenaRegion<u16>,
    pub(crate) down_weight: ArenaRegion<u16>,
    pub(crate) final_norm: ArenaRegion<u16>,
    pub(crate) embedding: ArenaRegion<u16>,
    pub(crate) target_hidden: ArenaRegion<u16>,
    pub(crate) normalized_embedding: ArenaRegion<u16>,
    pub(crate) normalized_hidden: ArenaRegion<u16>,
    pub(crate) residual: ArenaRegion<u16>,
    pub(crate) attention_normalized: ArenaRegion<u16>,
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,
    pub(crate) block_tables: ArenaRegion<u32>,
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_partial_maximum: ArenaRegion<f32>,
    pub(crate) attention_partial_denominator: ArenaRegion<f32>,
    pub(crate) attention_partial_numerator: ArenaRegion<f32>,
    pub(crate) attention_activation: ArenaRegion<u16>,
    pub(crate) attention_branch: ArenaRegion<u16>,
    pub(crate) post_attention_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) final_normalized: ArenaRegion<u16>,
    pub(crate) lm_head_activation_codes: ArenaRegion<u8>,
    pub(crate) lm_head_activation_scales: ArenaRegion<f32>,
    pub(crate) logits: ArenaRegion<u16>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentMtpCacheRegions {
    pub(crate) key_pages: ArenaRegion<u16>,
    pub(crate) value_pages: ArenaRegion<u16>,
}

/// Exact MTP weights, one long-context cache mirror, and address-stable route workspace.
#[derive(Clone, Debug)]
pub struct ResidentMtpLayout {
    arena: ArenaLayout,
    cache_arena: ArenaLayout,
    regions: ResidentMtpRegions,
    cache_regions: ResidentMtpCacheRegions,
    weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl ResidentMtpLayout {
    /// Reserves prompt `T=1,32,64,128,1024`, draft `B=1..8`, and realign `K=1..4`.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen38_27B;
        let prompt_hidden = product("resident MTP prompt hidden", MTP_PROMPT_ROWS, A::HIDDEN)?;
        let prompt_qkv = product(
            "resident MTP prompt QKV",
            MTP_PROMPT_ROWS,
            A::ATTENTION_QKV_ROWS,
        )?;
        let prompt_attention = product(
            "resident MTP prompt attention",
            MTP_PROMPT_ROWS,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_hidden = product("resident MTP batch hidden", MAX_BATCH, A::HIDDEN)?;
        let batch_attention = product(
            "resident MTP batch attention",
            MAX_BATCH,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_attention_partials = product(
            "resident MTP batch attention partials",
            product(
                "resident MTP batch attention heads",
                MAX_BATCH,
                A::NUM_ATTENTION_HEADS,
            )?,
            LONG_CONTEXT_GQA_MAX_PARTITIONS,
        )?;
        let batch_attention_numerator = product(
            "resident MTP batch attention numerator",
            batch_attention_partials,
            A::HEAD_DIM,
        )?;
        let batch_intermediate = product(
            "resident MTP batch intermediate",
            MAX_BATCH,
            A::INTERMEDIATE,
        )?;
        let input_projection = product(
            "resident MTP input projection",
            A::HIDDEN,
            product("resident MTP fused columns", 2, A::HIDDEN)?,
        )?;
        let qkv_weight = product("resident MTP QKV weights", A::ATTENTION_QKV_ROWS, A::HIDDEN)?;
        let attention_output_weight = product(
            "resident MTP attention-output weights",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let gate_up_weight = product(
            "resident MTP gate/up weights",
            product("resident MTP gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weight = product("resident MTP down weights", A::HIDDEN, A::INTERMEDIATE)?;
        let block_table_values = product(
            "resident MTP block tables",
            MAX_BATCH,
            LONG_CONTEXT_PHYSICAL_PAGES,
        )?;

        let mut arena = ArenaLayout::new();
        let regions = ResidentMtpRegions {
            embedding_norm: arena.reserve(A::HIDDEN, ALIGNMENT)?,
            hidden_norm: arena.reserve(A::HIDDEN, ALIGNMENT)?,
            input_projection: arena.reserve(input_projection, ALIGNMENT)?,
            input_norm: arena.reserve(A::HIDDEN, ALIGNMENT)?,
            qkv_weight: arena.reserve(qkv_weight, ALIGNMENT)?,
            query_norm: arena.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: arena.reserve(A::HEAD_DIM, ALIGNMENT)?,
            attention_output_weight: arena.reserve(attention_output_weight, ALIGNMENT)?,
            post_attention_norm: arena.reserve(A::HIDDEN, ALIGNMENT)?,
            gate_up_weight: arena.reserve(gate_up_weight, ALIGNMENT)?,
            down_weight: arena.reserve(down_weight, ALIGNMENT)?,
            final_norm: arena.reserve(A::HIDDEN, ALIGNMENT)?,
            embedding: arena.reserve(prompt_hidden, ALIGNMENT)?,
            target_hidden: arena.reserve(prompt_hidden, ALIGNMENT)?,
            normalized_embedding: arena.reserve(prompt_hidden, ALIGNMENT)?,
            normalized_hidden: arena.reserve(prompt_hidden, ALIGNMENT)?,
            residual: arena.reserve(prompt_hidden, ALIGNMENT)?,
            attention_normalized: arena.reserve(prompt_hidden, ALIGNMENT)?,
            qkv: arena.reserve(prompt_qkv, ALIGNMENT)?,
            rope_cos: arena.reserve(MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            rope_sin: arena.reserve(MTP_PROMPT_ROWS * 32, ALIGNMENT)?,
            block_tables: arena.reserve(block_table_values, ALIGNMENT)?,
            table_rows: arena.reserve(MTP_PROMPT_ROWS, ALIGNMENT)?,
            cache_positions: arena.reserve(MTP_PROMPT_ROWS, ALIGNMENT)?,
            lengths: arena.reserve(MTP_PROMPT_ROWS, ALIGNMENT)?,
            query: arena.reserve(prompt_attention, ALIGNMENT)?,
            attention: arena.reserve(batch_attention, ALIGNMENT)?,
            attention_partial_maximum: arena.reserve(batch_attention_partials, ALIGNMENT)?,
            attention_partial_denominator: arena.reserve(batch_attention_partials, ALIGNMENT)?,
            attention_partial_numerator: arena.reserve(batch_attention_numerator, ALIGNMENT)?,
            attention_activation: arena.reserve(batch_attention, ALIGNMENT)?,
            attention_branch: arena.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_residual: arena.reserve(batch_hidden, ALIGNMENT)?,
            mlp_normalized: arena.reserve(batch_hidden, ALIGNMENT)?,
            swiglu: arena.reserve(batch_intermediate, ALIGNMENT)?,
            mlp_branch: arena.reserve(batch_hidden, ALIGNMENT)?,
            residual_output: arena.reserve(batch_hidden, ALIGNMENT)?,
            final_normalized: arena.reserve(batch_hidden, ALIGNMENT)?,
            lm_head_activation_codes: arena.reserve(batch_hidden, ALIGNMENT)?,
            lm_head_activation_scales: arena.reserve(MAX_BATCH, ALIGNMENT)?,
            logits: arena.reserve(
                product("resident MTP logits", MAX_BATCH, A::VOCAB)?,
                ALIGNMENT,
            )?,
        };

        let cache_plane = product(
            "resident MTP cache plane",
            product(
                "resident MTP cache page heads",
                LONG_CONTEXT_PHYSICAL_PAGES,
                A::NUM_KV_HEADS,
            )?,
            product(
                "resident MTP cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;
        let mut cache_arena = ArenaLayout::new();
        let cache_regions = ResidentMtpCacheRegions {
            key_pages: cache_arena.reserve(cache_plane, ALIGNMENT)?,
            value_pages: cache_arena.reserve(cache_plane, ALIGNMENT)?,
        };

        let weight_bytes = sum(
            "resident MTP represented weights",
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
            "resident MTP represented cache",
            &[
                cache_regions.key_pages.byte_len(),
                cache_regions.value_pages.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "resident MTP address-stable workspace",
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
                regions.attention_partial_maximum.byte_len(),
                regions.attention_partial_denominator.byte_len(),
                regions.attention_partial_numerator.byte_len(),
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
            arena,
            cache_arena,
            regions,
            cache_regions,
            weight_bytes,
            cache_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn arena(&self) -> &ArenaLayout {
        &self.arena
    }

    pub(crate) const fn cache_arena(&self) -> &ArenaLayout {
        &self.cache_arena
    }

    pub(crate) const fn regions(&self) -> ResidentMtpRegions {
        self.regions
    }

    pub(crate) const fn cache_regions(&self) -> ResidentMtpCacheRegions {
        self.cache_regions
    }

    /// Exact unchanged source-BF16 MTP weights; target endpoint weights are shared.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Exact represented BF16 long-context K/V cache bytes.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable typed route workspace bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Main weights/workspace allocation bytes.
    pub const fn arena_bytes(&self) -> usize {
        self.arena.byte_len()
    }

    /// Dedicated MTP cache allocation bytes.
    pub const fn cache_arena_bytes(&self) -> usize {
        self.cache_arena.byte_len()
    }

    /// Complete incremental device bytes owned by resident MTP.
    pub const fn owner_bytes(&self) -> usize {
        self.arena_bytes() + self.cache_arena_bytes()
    }

    /// Alignment bytes outside represented weights, cache, and typed workspace regions.
    pub const fn padding_bytes(&self) -> usize {
        self.owner_bytes() - self.weight_bytes - self.cache_bytes - self.workspace_bytes
    }
}

impl LayerMemoryLayout for ResidentMtpLayout {
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
    use super::{ALIGNMENT, ResidentMtpLayout};

    #[test]
    fn resident_mtp_byte_accounting_is_exact() {
        let layout = ResidentMtpLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 849_398_784);
        assert_eq!(layout.cache_bytes(), 901_251_072);
        assert_eq!(layout.workspace_bytes(), 293_307_872);
        assert_eq!(layout.arena_bytes(), 1_142_706_944);
        assert_eq!(layout.cache_arena_bytes(), 901_251_072);
        assert_eq!(layout.owner_bytes(), 2_043_958_016);
        assert_eq!(layout.padding_bytes(), 288);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_their_arenas() {
        let layout = ResidentMtpLayout::build().unwrap();
        let regions = layout.regions();
        let mut main = vec![
            span(regions.embedding_norm),
            span(regions.hidden_norm),
            span(regions.input_projection),
            span(regions.input_norm),
            span(regions.qkv_weight),
            span(regions.query_norm),
            span(regions.key_norm),
            span(regions.attention_output_weight),
            span(regions.post_attention_norm),
            span(regions.gate_up_weight),
            span(regions.down_weight),
            span(regions.final_norm),
            span(regions.embedding),
            span(regions.target_hidden),
            span(regions.normalized_embedding),
            span(regions.normalized_hidden),
            span(regions.residual),
            span(regions.attention_normalized),
            span(regions.qkv),
            span(regions.rope_cos),
            span(regions.rope_sin),
            span(regions.block_tables),
            span(regions.table_rows),
            span(regions.cache_positions),
            span(regions.lengths),
            span(regions.query),
            span(regions.attention),
            span(regions.attention_partial_maximum),
            span(regions.attention_partial_denominator),
            span(regions.attention_partial_numerator),
            span(regions.attention_activation),
            span(regions.attention_branch),
            span(regions.post_attention_residual),
            span(regions.mlp_normalized),
            span(regions.swiglu),
            span(regions.mlp_branch),
            span(regions.residual_output),
            span(regions.final_normalized),
            span(regions.lm_head_activation_codes),
            span(regions.lm_head_activation_scales),
            span(regions.logits),
        ];
        assert_layout(&mut main, layout.arena_bytes());

        let cache = layout.cache_regions();
        let mut cache_spans = vec![span(cache.key_pages), span(cache.value_pages)];
        assert_layout(&mut cache_spans, layout.cache_arena_bytes());
    }

    fn assert_layout(spans: &mut [(usize, usize)], arena_bytes: usize) {
        spans.sort_unstable_by_key(|(offset, _)| *offset);
        for &(offset, bytes) in spans.iter() {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= arena_bytes);
        }
        for adjacent in spans.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }

    use tuisko_gpu::ArenaRegion;
}

//! Address-stable prompt-priming layout for the exact Qwen3.8 MTP layer.

use crate::common::math::{product, sum};
use crate::{EngineResult, LONG_CONTEXT_PHYSICAL_PAGES, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen38_27B};

const ALIGNMENT: usize = 256;
pub(crate) const MTP_PROMPT_TILE_CAPACITY: usize = 1_024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpPromptPrimeRegions {
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
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) key_pages: ArenaRegion<u16>,
    pub(crate) value_pages: ArenaRegion<u16>,
}

/// Exact prompt-prime weights, shared-page cache mirror, and maximum-tile workspace.
#[derive(Clone, Debug)]
pub struct MtpPromptPrimeLayout {
    builder: ArenaLayout,
    regions: MtpPromptPrimeRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl MtpPromptPrimeLayout {
    /// Reserves exact `T=1,32,64,128,1024` priming routes and the resident cache inventory.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen38_27B;
        let row_hidden = product(
            "MTP prompt row-hidden elements",
            MTP_PROMPT_TILE_CAPACITY,
            A::HIDDEN,
        )?;
        let row_qkv = product(
            "MTP prompt row-QKV elements",
            MTP_PROMPT_TILE_CAPACITY,
            A::ATTENTION_QKV_ROWS,
        )?;
        let row_attention = product(
            "MTP prompt row-attention elements",
            MTP_PROMPT_TILE_CAPACITY,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let input_projection = product(
            "MTP prompt fusion weights",
            A::HIDDEN,
            product("MTP prompt fusion columns", 2, A::HIDDEN)?,
        )?;
        let qkv_weight = product("MTP prompt QKV weights", A::ATTENTION_QKV_ROWS, A::HIDDEN)?;
        let cache_plane = product(
            "MTP prompt cache plane elements",
            product(
                "MTP prompt cache page heads",
                LONG_CONTEXT_PHYSICAL_PAGES,
                A::NUM_KV_HEADS,
            )?,
            product(
                "MTP prompt cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;
        let block_table_values = product(
            "MTP prompt block-table values",
            MAX_BATCH,
            LONG_CONTEXT_PHYSICAL_PAGES,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = MtpPromptPrimeRegions {
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
            rope_cos: builder.reserve(MTP_PROMPT_TILE_CAPACITY * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MTP_PROMPT_TILE_CAPACITY * 32, ALIGNMENT)?,
            block_tables: builder.reserve(block_table_values, ALIGNMENT)?,
            table_rows: builder.reserve(MTP_PROMPT_TILE_CAPACITY, ALIGNMENT)?,
            cache_positions: builder.reserve(MTP_PROMPT_TILE_CAPACITY, ALIGNMENT)?,
            query: builder.reserve(row_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "MTP prompt represented weight bytes",
            &[
                regions.embedding_norm.byte_len(),
                regions.hidden_norm.byte_len(),
                regions.input_projection.byte_len(),
                regions.input_norm.byte_len(),
                regions.qkv_weight.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "MTP prompt represented cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "MTP prompt workspace bytes",
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
                regions.query.byte_len(),
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

    pub(crate) const fn regions(&self) -> MtpPromptPrimeRegions {
        self.regions
    }

    /// Exact unchanged BF16 prompt-prime source weights.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Complete represented BF16 MTP K/V cache inventory.
    pub const fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// Address-stable maximum-tile and metadata workspace.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Weights, cache, and workspace without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Complete device allocation including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Alignment bytes not attributed to a represented owner plane.
    pub const fn padding_bytes(&self) -> usize {
        self.arena_bytes() - self.owner_bytes()
    }
}

impl LayerMemoryLayout for MtpPromptPrimeLayout {
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
    use super::{ALIGNMENT, MtpPromptPrimeLayout};

    #[test]
    fn qwen_mtp_prompt_prime_byte_accounting_is_exact() {
        let layout = MtpPromptPrimeLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 251_689_984);
        assert_eq!(layout.cache_bytes(), 901_251_072);
        assert_eq!(layout.workspace_bytes(), 117_820_864);
        assert_eq!(layout.owner_bytes(), 1_270_761_920);
        assert_eq!(layout.arena_bytes(), 1_270_761_984);
        assert_eq!(layout.padding_bytes(), 64);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = MtpPromptPrimeLayout::build().unwrap();
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
            span(regions.query),
            span(regions.key_pages),
            span(regions.value_pages),
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

//! Single-allocation layout for one late dense-FP8 full-attention decoder layer.

use crate::common::math::{product, sum};
use crate::{EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::{ATTENTION_PAGE_SIZE, PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES};
use tuisko_model::Arch;

const ALIGNMENT: usize = 256;
pub(crate) const PHYSICAL_PAGES: usize = 24;
pub(crate) const TABLE_STRIDE: usize = PHYSICAL_PAGES / MAX_BATCH;
pub(crate) const CONTEXT_CAPACITY: usize = TABLE_STRIDE * ATTENTION_PAGE_SIZE;
pub(crate) const PREFILL_TABLE_STRIDE: usize = PHYSICAL_PAGES;
pub(crate) const PREFILL_CONTEXT_CAPACITY: usize = PREFILL_TABLE_STRIDE * ATTENTION_PAGE_SIZE;
pub(crate) const MAX_ROWS: usize = 1_024;

const _: () = assert!(MAX_BATCH * TABLE_STRIDE == PHYSICAL_PAGES);
const _: () = assert!(MAX_ROWS <= PREFILL_CONTEXT_CAPACITY);

#[derive(Clone, Copy, Debug)]
pub(crate) struct FullAttentionLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) qkv_activation_codes: ArenaRegion<u8>,
    pub(crate) qkv_activation_scales: ArenaRegion<f32>,
    pub(crate) qkv_weight_codes: ArenaRegion<u8>,
    pub(crate) qkv_weight_scales: ArenaRegion<u16>,
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
    pub(crate) macro_partials: ArenaRegion<f32>,
    pub(crate) output_activation_codes: ArenaRegion<u8>,
    pub(crate) output_activation_scales: ArenaRegion<f32>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) output_weight_scales: ArenaRegion<u16>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) gate_up_activation_codes: ArenaRegion<u8>,
    pub(crate) gate_up_activation_scales: ArenaRegion<f32>,
    pub(crate) gate_up_weight_codes: ArenaRegion<u8>,
    pub(crate) gate_up_weight_scales: ArenaRegion<u16>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) down_activation_codes: ArenaRegion<u8>,
    pub(crate) down_activation_scales: ArenaRegion<f32>,
    pub(crate) down_weight_codes: ArenaRegion<u8>,
    pub(crate) down_weight_scales: ArenaRegion<u16>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked weights, KV cache, metadata, and workspace for one exact layer owner.
#[derive(Clone, Debug)]
pub struct FullAttentionLayerLayout {
    builder: ArenaLayout,
    regions: FullAttentionLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl FullAttentionLayerLayout {
    /// Reserves exact decode and prefill seams plus the shared 24-page cache pool.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        let batch_hidden = product("attention row-hidden elements", MAX_ROWS, A::HIDDEN)?;
        let batch_qkv = product(
            "attention fused QKV row elements",
            MAX_ROWS,
            A::ATTENTION_QKV_ROWS,
        )?;
        let batch_attention = product(
            "attention row-output elements",
            MAX_ROWS,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_intermediate = product("attention MLP elements", MAX_ROWS, A::INTERMEDIATE)?;
        let qkv_weights = product("attention QKV weights", A::ATTENTION_QKV_ROWS, A::HIDDEN)?;
        let output_weights = product(
            "attention output weights",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let gate_up_weights = product(
            "attention dense-FP8 gate/up weights",
            product("attention gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weights = product(
            "attention dense-FP8 down weights",
            A::HIDDEN,
            A::INTERMEDIATE,
        )?;
        let cache_plane = product(
            "attention cache plane",
            product(
                "attention cache page heads",
                PHYSICAL_PAGES,
                A::NUM_KV_HEADS,
            )?,
            product(
                "attention cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = FullAttentionLayerRegions {
            residual_input: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            qkv_activation_codes: builder.reserve(batch_hidden, ALIGNMENT)?,
            qkv_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            qkv_weight_codes: builder.reserve(qkv_weights, ALIGNMENT)?,
            qkv_weight_scales: builder.reserve(A::ATTENTION_QKV_ROWS, ALIGNMENT)?,
            qkv: builder.reserve(batch_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            block_tables: builder.reserve(MAX_BATCH * TABLE_STRIDE, ALIGNMENT)?,
            table_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            cache_positions: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            lengths: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            prefill_rope_cos: builder.reserve(MAX_ROWS * 32, ALIGNMENT)?,
            prefill_rope_sin: builder.reserve(MAX_ROWS * 32, ALIGNMENT)?,
            prefill_table_rows: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            prefill_cache_positions: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            prefill_lengths: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            query: builder.reserve(batch_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(batch_attention, ALIGNMENT)?,
            macro_partials: builder.reserve(
                PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES / size_of::<f32>(),
                ALIGNMENT,
            )?,
            output_activation_codes: builder.reserve(batch_attention, ALIGNMENT)?,
            output_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weights, ALIGNMENT)?,
            output_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(batch_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            gate_up_activation_codes: builder.reserve(batch_hidden, ALIGNMENT)?,
            gate_up_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            gate_up_weight_codes: builder.reserve(gate_up_weights, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(2 * A::INTERMEDIATE, ALIGNMENT)?,
            swiglu: builder.reserve(batch_intermediate, ALIGNMENT)?,
            down_activation_codes: builder.reserve(batch_intermediate, ALIGNMENT)?,
            down_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weights, ALIGNMENT)?,
            down_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mlp_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "full-attention resident weight bytes",
            &[
                regions.input_norm.byte_len(),
                regions.qkv_weight_codes.byte_len(),
                regions.qkv_weight_scales.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.output_weight_scales.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.gate_up_weight_codes.byte_len(),
                regions.gate_up_weight_scales.byte_len(),
                regions.down_weight_codes.byte_len(),
                regions.down_weight_scales.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "full-attention KV cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "full-attention workspace bytes",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.qkv_activation_codes.byte_len(),
                regions.qkv_activation_scales.byte_len(),
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
                regions.macro_partials.byte_len(),
                regions.output_activation_codes.byte_len(),
                regions.output_activation_scales.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.mlp_normalized.byte_len(),
                regions.gate_up_activation_codes.byte_len(),
                regions.gate_up_activation_scales.byte_len(),
                regions.swiglu.byte_len(),
                regions.down_activation_codes.byte_len(),
                regions.down_activation_scales.byte_len(),
                regions.mlp_branch.byte_len(),
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

    pub(crate) const fn regions(&self) -> FullAttentionLayerRegions {
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

    /// Resident weights, KV cache, and workspace without alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.cache_bytes + self.workspace_bytes
    }

    /// Fixed per-slot short-context capacity of this initial decode owner.
    pub const fn context_capacity(&self) -> usize {
        CONTEXT_CAPACITY
    }

    /// Complete shared-page capacity used by exact from-empty prefill.
    pub const fn prefill_context_capacity(&self) -> usize {
        PREFILL_CONTEXT_CAPACITY
    }
}

impl LayerMemoryLayout for FullAttentionLayerLayout {
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
    use super::{
        ALIGNMENT, CONTEXT_CAPACITY, FullAttentionLayerLayout, MAX_ROWS, PREFILL_CONTEXT_CAPACITY,
    };
    use tuisko_model::Qwen38_27B;

    #[test]
    fn qwen_full_attention_layer_byte_accounting_is_exact() {
        let layout = FullAttentionLayerLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 372_395_008);
        assert_eq!(layout.cache_bytes(), 3_145_728);
        assert_eq!(layout.workspace_bytes(), 639_924_416);
        assert_eq!(layout.owner_bytes(), 1_015_465_152);
        assert_eq!(layout.arena_bytes(), 1_015_465_984);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 832);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = FullAttentionLayerLayout::build::<Qwen38_27B>().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
            span(regions.qkv_activation_codes),
            span(regions.qkv_activation_scales),
            span(regions.qkv_weight_codes),
            span(regions.qkv_weight_scales),
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
            span(regions.macro_partials),
            span(regions.output_activation_codes),
            span(regions.output_activation_scales),
            span(regions.output_weight_codes),
            span(regions.output_weight_scales),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.mlp_normalized),
            span(regions.gate_up_activation_codes),
            span(regions.gate_up_activation_scales),
            span(regions.gate_up_weight_codes),
            span(regions.gate_up_weight_scales),
            span(regions.swiglu),
            span(regions.down_activation_codes),
            span(regions.down_activation_scales),
            span(regions.down_weight_codes),
            span(regions.down_weight_scales),
            span(regions.mlp_branch),
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

    #[test]
    fn cache_geometry_is_fixed_per_slot() {
        let layout = FullAttentionLayerLayout::build::<Qwen38_27B>().unwrap();
        assert_eq!(layout.context_capacity(), CONTEXT_CAPACITY);
        assert_eq!(layout.context_capacity(), 192);
        assert_eq!(layout.prefill_context_capacity(), PREFILL_CONTEXT_CAPACITY);
        assert_eq!(layout.prefill_context_capacity(), 1_536);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(layout.cache_bytes(), 3_145_728);
    }

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

//! Single-allocation layout for one Qwen3.5 full-attention decoder layer.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
pub(crate) const QWEN35_PHYSICAL_PAGES: usize = 24;
pub(crate) const QWEN35_TABLE_STRIDE: usize = 3;
pub(crate) const QWEN35_CONTEXT_CAPACITY: usize = QWEN35_TABLE_STRIDE * ATTENTION_PAGE_SIZE;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35FullAttentionLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) qkv_weight_codes: ArenaRegion<u8>,
    pub(crate) qkv_weight_scales: ArenaRegion<u8>,
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
    pub(crate) output_activation: ArenaRegion<u16>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) output_weight_scales: ArenaRegion<u8>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) gate_up_activation_codes: ArenaRegion<u8>,
    pub(crate) gate_up_activation_scales: ArenaRegion<u8>,
    pub(crate) gate_weight_codes: ArenaRegion<u8>,
    pub(crate) up_weight_codes: ArenaRegion<u8>,
    pub(crate) gate_up_weight_scales: ArenaRegion<u8>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) down_weight_codes: ArenaRegion<u8>,
    pub(crate) down_weight_scales: ArenaRegion<u8>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked weights, BF16 KV cache, metadata, and workspace for one Qwen3.5 layer.
#[derive(Clone, Debug)]
pub struct Qwen35FullAttentionLayerLayout {
    builder: ArenaLayout,
    regions: Qwen35FullAttentionLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen35FullAttentionLayerLayout {
    /// Reserves every source and seam for exact decode `B=1..=8` and 192-token slots.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen35_9B;
        require_geometry::<A>()?;

        let batch_hidden = product(
            "Qwen3.5 attention batch-hidden elements",
            MAX_BATCH,
            A::HIDDEN,
        )?;
        let batch_qkv = product(
            "Qwen3.5 attention fused QKV elements",
            MAX_BATCH,
            A::ATTENTION_QKV_ROWS,
        )?;
        let batch_attention = product(
            "Qwen3.5 attention output elements",
            MAX_BATCH,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let batch_intermediate =
            product("Qwen3.5 attention MLP elements", MAX_BATCH, A::INTERMEDIATE)?;
        let qkv_weight_codes = packed_codes(
            "Qwen3.5 attention QKV weight codes",
            A::ATTENTION_QKV_ROWS,
            A::HIDDEN,
        )?;
        let qkv_weight_scales = scales(
            "Qwen3.5 attention QKV weight scales",
            A::ATTENTION_QKV_ROWS,
            A::HIDDEN,
        )?;
        let output_weight_codes = packed_codes(
            "Qwen3.5 attention output weight codes",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let output_weight_scales = scales(
            "Qwen3.5 attention output weight scales",
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let gate_codes = packed_codes(
            "Qwen3.5 attention gate weight codes",
            A::INTERMEDIATE,
            A::HIDDEN,
        )?;
        let gate_up_scales = scales(
            "Qwen3.5 attention gate/up weight scales",
            product("Qwen3.5 attention gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weight_codes = packed_codes(
            "Qwen3.5 attention down weight codes",
            A::HIDDEN,
            A::INTERMEDIATE,
        )?;
        let down_weight_scales = scales(
            "Qwen3.5 attention down weight scales",
            A::HIDDEN,
            A::INTERMEDIATE,
        )?;
        let cache_plane = product(
            "Qwen3.5 attention BF16 cache plane",
            product(
                "Qwen3.5 attention cache page heads",
                QWEN35_PHYSICAL_PAGES,
                A::NUM_KV_HEADS,
            )?,
            product(
                "Qwen3.5 attention cache page values",
                ATTENTION_PAGE_SIZE,
                A::HEAD_DIM,
            )?,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen35FullAttentionLayerRegions {
            residual_input: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            qkv_weight_codes: builder.reserve(qkv_weight_codes, ALIGNMENT)?,
            qkv_weight_scales: builder.reserve(qkv_weight_scales, ALIGNMENT)?,
            qkv: builder.reserve(batch_qkv, ALIGNMENT)?,
            query_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(A::HEAD_DIM, ALIGNMENT)?,
            rope_cos: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            rope_sin: builder.reserve(MAX_BATCH * 32, ALIGNMENT)?,
            block_tables: builder.reserve(MAX_BATCH * QWEN35_TABLE_STRIDE, ALIGNMENT)?,
            table_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            cache_positions: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            lengths: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            query: builder.reserve(batch_attention, ALIGNMENT)?,
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            attention: builder.reserve(batch_attention, ALIGNMENT)?,
            output_activation: builder.reserve(batch_attention, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weight_codes, ALIGNMENT)?,
            output_weight_scales: builder.reserve(output_weight_scales, ALIGNMENT)?,
            mixer_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(batch_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            gate_up_activation_codes: builder.reserve(batch_hidden / 2, ALIGNMENT)?,
            gate_up_activation_scales: builder.reserve(batch_hidden / NVFP4_GROUP, ALIGNMENT)?,
            gate_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            up_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(gate_up_scales, ALIGNMENT)?,
            swiglu: builder.reserve(batch_intermediate, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weight_codes, ALIGNMENT)?,
            down_weight_scales: builder.reserve(down_weight_scales, ALIGNMENT)?,
            mlp_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.5 full-attention resident weight bytes",
            &[
                regions.input_norm.byte_len(),
                regions.qkv_weight_codes.byte_len(),
                regions.qkv_weight_scales.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.output_weight_scales.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.gate_weight_codes.byte_len(),
                regions.up_weight_codes.byte_len(),
                regions.gate_up_weight_scales.byte_len(),
                regions.down_weight_codes.byte_len(),
                regions.down_weight_scales.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let cache_bytes = sum(
            "Qwen3.5 full-attention BF16 KV cache bytes",
            &[regions.key_pages.byte_len(), regions.value_pages.byte_len()],
        )?;
        let workspace_bytes = sum(
            "Qwen3.5 full-attention workspace bytes",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.qkv.byte_len(),
                regions.rope_cos.byte_len(),
                regions.rope_sin.byte_len(),
                regions.block_tables.byte_len(),
                regions.table_rows.byte_len(),
                regions.cache_positions.byte_len(),
                regions.lengths.byte_len(),
                regions.query.byte_len(),
                regions.attention.byte_len(),
                regions.output_activation.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.mlp_normalized.byte_len(),
                regions.gate_up_activation_codes.byte_len(),
                regions.gate_up_activation_scales.byte_len(),
                regions.swiglu.byte_len(),
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

    pub(crate) const fn regions(&self) -> Qwen35FullAttentionLayerRegions {
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

    /// Exact represented BF16 key/value cache bytes.
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
        QWEN35_CONTEXT_CAPACITY
    }
}

fn require_geometry<A: Arch>() -> EngineResult<()> {
    if !A::HIDDEN.is_multiple_of(NVFP4_GROUP)
        || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP)
        || !A::ATTENTION_QKV_ROWS.is_multiple_of(128)
        || !A::HIDDEN.is_multiple_of(128)
    {
        return Err(EngineError::layout(
            "Qwen3.5 full-attention geometry must satisfy K16 and M128 NVFP4 tiling",
        ));
    }

    Ok(())
}

fn packed_codes(name: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    product(name, rows, columns / 2)
}

fn scales(name: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    product(name, rows, columns / NVFP4_GROUP)
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
    use super::{ALIGNMENT, QWEN35_CONTEXT_CAPACITY, Qwen35FullAttentionLayerLayout};

    #[test]
    fn byte_accounting_is_exact() {
        let layout = Qwen35FullAttentionLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 117_990_400);
        assert_eq!(layout.cache_bytes(), 6_291_456);
        assert_eq!(layout.workspace_bytes(), 1_233_088);
        assert_eq!(layout.owner_bytes(), 125_514_944);
        assert_eq!(layout.arena_bytes(), 125_515_776);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 832);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen35FullAttentionLayerLayout::build().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
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
            span(regions.query),
            span(regions.key_pages),
            span(regions.value_pages),
            span(regions.attention),
            span(regions.output_activation),
            span(regions.output_weight_codes),
            span(regions.output_weight_scales),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.mlp_normalized),
            span(regions.gate_up_activation_codes),
            span(regions.gate_up_activation_scales),
            span(regions.gate_weight_codes),
            span(regions.up_weight_codes),
            span(regions.gate_up_weight_scales),
            span(regions.swiglu),
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
    fn bf16_cache_geometry_is_fixed_per_slot() {
        let layout = Qwen35FullAttentionLayerLayout::build().unwrap();

        assert_eq!(layout.context_capacity(), QWEN35_CONTEXT_CAPACITY);
        assert_eq!(layout.context_capacity(), 192);
        assert_eq!(layout.cache_bytes(), 6_291_456);
    }

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

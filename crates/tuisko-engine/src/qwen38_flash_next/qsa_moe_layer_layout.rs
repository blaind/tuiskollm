//! Single-allocation layout for one Qwen3.8-Flash-Next sparse-attention plus MoE decoder layer.
//!
//! Twelve decoder layers use this shape: two hyper-connection brackets around QSA and MoE.
//!
//! ```text
//! residual_input  [rows, 10240]
//!   HC attn Mix   -> hc_mixed [rows, 2560]
//!   QSA block     -> qkv [rows, 13312] -> qk-prepare (+KV append) -> query [rows, 6144] f32
//!                 -> paged dense GQA   -> attention [rows, 6144] f32
//!                 -> sigmoid gate      -> attention_gated [rows, 6144]
//!                 -> o_proj            -> block_output [rows, 2560]
//!   HC write-back -> attention_residual
//!   HC mlp Mix / MoE / HC write-back   -> residual_output
//! ```
//!
//! Dense attention is exact through 2,051 visible keys; longer rounds use selection. Compressed
//! block keys share K/V page mappings. Raw keys use a four-token ring and one round-local plane.
//! A single selection scratch plane is safe because every captured tile runs serially on one
//! stream; a multi-stream capture would require separate scratch ownership.

use crate::common::math::{product, sum};
use crate::qwen38_flash_next::gdn_moe_layer_layout::QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;
use crate::qwen38_flash_next::persistent_state::{ALIGNMENT, Qwen38FlashNextPersistentState};
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_kernels_sm120::{
    SELECTION_BLOCKS_PER_PAGE, SELECTION_MAX_SELECTED, SELECTION_RING_SLOTS, SELECTION_ROW_TILE,
    SELECTION_SCRATCH_WORDS, selection_block_bucket, selection_round_blocks,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Tokens one physical page holds, the house paged-attention page size.
pub(crate) const QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE: usize = 64;

/// Pages reserved so every slot can drive the selected route to 4,096 visible keys.
pub(crate) const QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES: usize = 512;

/// Pages one decode slot owns.
pub(crate) const QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE: usize =
    QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES / MAX_BATCH;

/// Context depth one decode slot reaches.
pub(crate) const QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY: usize =
    QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE * QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;

/// Candidate blocks the widest round this owner admits can present.
pub(crate) const QWEN38_FLASH_NEXT_QSA_CANDIDATE_BLOCKS: usize =
    QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY / Qwen38FlashNext::INDEXER_COMPRESS_RATIO;

/// Values between two rows of this owner's score scratch.
///
/// The prepared scoring grid is a bucket rather than an exact count, and the scorer indexes its
/// plane by candidate, so the stride is the bucket the capture picked and never the round's own
/// block count.
pub(crate) const QWEN38_FLASH_NEXT_QSA_SCORE_STRIDE: usize =
    match selection_block_bucket(QWEN38_FLASH_NEXT_QSA_CANDIDATE_BLOCKS) {
        Some(bucket) => bucket,
        None => panic!("this owner's context is inside the widest prepared scoring grid"),
    };

/// Micro-blocks one round of the widest admitted width can close.
pub(crate) const QWEN38_FLASH_NEXT_QSA_ROUND_BLOCKS: usize =
    selection_round_blocks(QWEN38_FLASH_NEXT_LAYER_MAX_ROWS);

/// Rotary elements one token carries: `rotary_dim / 2` with `partial_rotary_factor = 0.25`.
const ROTARY_ELEMENTS: usize = 32;

const ROUTED_SLOTS: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
const EXPERT_WEIGHT_SCALES: usize = 3;

const _: () =
    assert!(MAX_BATCH * QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE == QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES);
const _: () = assert!(QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY == 4_096);
const _: () = assert!(QWEN38_FLASH_NEXT_QSA_SCORE_STRIDE == 4_096);
const _: () = assert!(QWEN38_FLASH_NEXT_QSA_ROUND_BLOCKS == 257);

/// Every region one Qwen3.8-Flash-Next QSA/MoE layer owns, in launch order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextQsaMoeLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) attention_residual: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,

    pub(crate) attention_hc_norm: ArenaRegion<u16>,
    pub(crate) attention_hc_down: ArenaRegion<u16>,
    pub(crate) attention_hc_up: ArenaRegion<u16>,
    pub(crate) attention_hc_inject: ArenaRegion<u16>,
    pub(crate) mlp_hc_norm: ArenaRegion<u16>,
    pub(crate) mlp_hc_down: ArenaRegion<u16>,
    pub(crate) mlp_hc_up: ArenaRegion<u16>,
    pub(crate) mlp_hc_inject: ArenaRegion<u16>,

    pub(crate) hc_normalized: ArenaRegion<u16>,
    pub(crate) hc_low_rank: ArenaRegion<u16>,
    pub(crate) hc_mixed: ArenaRegion<u16>,
    pub(crate) hc_write_gate: ArenaRegion<u16>,

    // --- attention weights ---
    pub(crate) qkv_weight: ArenaRegion<u16>,
    pub(crate) output_weight: ArenaRegion<u16>,
    pub(crate) query_norm: ArenaRegion<u16>,
    pub(crate) key_norm: ArenaRegion<u16>,
    pub(crate) indexer_qk_weight: ArenaRegion<u16>,
    pub(crate) indexer_query_norm: ArenaRegion<u16>,
    pub(crate) indexer_key_norm: ArenaRegion<u16>,

    // --- attention activations and metadata ---
    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_gated: ArenaRegion<u16>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) block_tables: ArenaRegion<u32>,

    // --- selection stage planes ---
    /// Fused indexer query and key projection rows `[rows, 640]`.
    pub(crate) indexer_qk: ArenaRegion<u16>,
    /// Prepared indexer queries `[rows, 4, 128]`, normed and rotated at each query's position.
    pub(crate) indexer_query: ArenaRegion<f32>,
    /// Per-sequence raw-key ring, four slots wide: one open micro-block.
    pub(crate) indexer_ring: ArenaRegion<u16>,
    /// Round-local raw keys, one row per position a prompt tile carries.
    pub(crate) indexer_raw_round: ArenaRegion<u16>,
    /// Rotary cosines at each closing block's first token.
    pub(crate) block_rope_cos: ArenaRegion<f32>,
    /// Rotary sines at each closing block's first token.
    pub(crate) block_rope_sin: ArenaRegion<f32>,
    /// First micro-block each compression row closes this round.
    pub(crate) closing_first_blocks: ArenaRegion<u32>,
    /// Micro-blocks each compression row closes this round.
    pub(crate) closing_block_counts: ArenaRegion<u32>,
    /// Complete candidate blocks each token of the round may score.
    pub(crate) candidate_blocks: ArenaRegion<u32>,
    /// Score scratch, one row tile at a time.
    pub(crate) scores: ArenaRegion<f32>,
    /// Selection scratch for one row tile; serialized graph launches share this plane.
    pub(crate) select_scratch: ArenaRegion<u32>,
    /// Selected positions `[rows, 2051]`, ascending.
    pub(crate) selected: ArenaRegion<u32>,
    /// Selected position count of each token.
    pub(crate) selected_counts: ArenaRegion<u32>,

    // --- paged cache planes ---
    pub(crate) key_pages: ArenaRegion<u8>,
    pub(crate) value_pages: ArenaRegion<u8>,
    /// Compressed 128-wide block keys, one per four cached tokens, on this layer's page mapping.
    pub(crate) block_keys: ArenaRegion<u16>,

    // --- MoE, identical to the GDN layer's half ---
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) expert_weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) shared_up_weight: ArenaRegion<u16>,
    pub(crate) shared_down_weight: ArenaRegion<u16>,
    pub(crate) shared_gate_logit_weight: ArenaRegion<u16>,
    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_intermediate: ArenaRegion<u16>,
    pub(crate) routed_output: ArenaRegion<u16>,
    pub(crate) shared_intermediate: ArenaRegion<u16>,
    pub(crate) shared_output: ArenaRegion<u16>,
    pub(crate) shared_gate_logit: ArenaRegion<u16>,

    pub(crate) block_output: ArenaRegion<u16>,
}

/// Checked weights, paged cache, and workspace for one Qwen3.8-Flash-Next QSA/MoE layer.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextQsaMoeLayerLayout {
    builder: ArenaLayout,
    regions: Qwen38FlashNextQsaMoeLayerRegions,
    resident_weight_bytes: usize,
    cache_bytes: usize,
    workspace_bytes: usize,
    layer: usize,
}

impl Qwen38FlashNextQsaMoeLayerLayout {
    /// Reserves one sparse-attention decoder layer's complete allocation.
    pub fn build(layer: usize) -> EngineResult<Self> {
        type A = Qwen38FlashNext;
        require_geometry()?;
        require_qsa_layer(layer)?;

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

        let qkv_weight = product(
            "Qwen3.8-Flash-Next QSA fused projection",
            A::ATTENTION_QKV_ROWS,
            <A as Arch>::HIDDEN,
        )?;
        let output_weight = product(
            "Qwen3.8-Flash-Next QSA output projection",
            <A as Arch>::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let indexer_qk_weight = product(
            "Qwen3.8-Flash-Next indexer projection",
            A::INDEXER_ROWS,
            <A as Arch>::HIDDEN,
        )?;
        let row_qkv = product(
            "Qwen3.8-Flash-Next QSA projection rows",
            rows,
            A::ATTENTION_QKV_ROWS,
        )?;
        let row_attention = product(
            "Qwen3.8-Flash-Next QSA attention rows",
            rows,
            A::ATTENTION_OUTPUT_COLUMNS,
        )?;
        let row_rotary = product("Qwen3.8-Flash-Next QSA rotary rows", rows, ROTARY_ELEMENTS)?;

        let cache_plane = product(
            "Qwen3.8-Flash-Next QSA cache plane",
            product(
                "Qwen3.8-Flash-Next QSA cache page heads",
                QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
                <A as Arch>::NUM_KV_HEADS,
            )?,
            product(
                "Qwen3.8-Flash-Next QSA cache page values",
                QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
                <A as Arch>::HEAD_DIM,
            )?,
        )?;
        let block_key_plane = product(
            "Qwen3.8-Flash-Next block-key cache plane",
            product(
                "Qwen3.8-Flash-Next block-key cache page blocks",
                QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
                SELECTION_BLOCKS_PER_PAGE,
            )?,
            A::INDEXER_HEAD_DIM,
        )?;
        let indexer_qk = product(
            "Qwen3.8-Flash-Next indexer projection rows",
            rows,
            A::INDEXER_ROWS,
        )?;
        let indexer_query = product(
            "Qwen3.8-Flash-Next indexer query rows",
            product(
                "Qwen3.8-Flash-Next indexer query heads",
                rows,
                A::INDEXER_HEADS,
            )?,
            A::INDEXER_HEAD_DIM,
        )?;
        let indexer_ring = product(
            "Qwen3.8-Flash-Next indexer ring",
            product(
                "Qwen3.8-Flash-Next indexer ring slots",
                MAX_BATCH,
                SELECTION_RING_SLOTS,
            )?,
            A::INDEXER_HEAD_DIM,
        )?;
        let indexer_raw_round = product(
            "Qwen3.8-Flash-Next indexer round keys",
            rows,
            A::INDEXER_HEAD_DIM,
        )?;
        let block_rotary = product(
            "Qwen3.8-Flash-Next block rotary rows",
            QWEN38_FLASH_NEXT_QSA_ROUND_BLOCKS,
            ROTARY_ELEMENTS,
        )?;
        let score_plane = product(
            "Qwen3.8-Flash-Next score scratch",
            SELECTION_ROW_TILE,
            QWEN38_FLASH_NEXT_QSA_SCORE_STRIDE,
        )?;
        let selected_plane = product(
            "Qwen3.8-Flash-Next selected positions",
            rows,
            SELECTION_MAX_SELECTED,
        )?;

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
        let regions = Qwen38FlashNextQsaMoeLayerRegions {
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

            qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
            output_weight: builder.reserve(output_weight, ALIGNMENT)?,
            query_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
            key_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
            indexer_qk_weight: builder.reserve(indexer_qk_weight, ALIGNMENT)?,
            indexer_query_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,
            indexer_key_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,

            qkv: builder.reserve(row_qkv, ALIGNMENT)?,
            query: builder.reserve(row_attention, ALIGNMENT)?,
            attention: builder.reserve(row_attention, ALIGNMENT)?,
            attention_gated: builder.reserve(row_attention, ALIGNMENT)?,
            rope_cos: builder.reserve(row_rotary, ALIGNMENT)?,
            rope_sin: builder.reserve(row_rotary, ALIGNMENT)?,
            table_rows: builder.reserve(rows, ALIGNMENT)?,
            cache_positions: builder.reserve(rows, ALIGNMENT)?,
            lengths: builder.reserve(rows, ALIGNMENT)?,
            block_tables: builder.reserve(QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES, ALIGNMENT)?,

            indexer_qk: builder.reserve(indexer_qk, ALIGNMENT)?,
            indexer_query: builder.reserve(indexer_query, ALIGNMENT)?,
            indexer_ring: builder.reserve(indexer_ring, ALIGNMENT)?,
            indexer_raw_round: builder.reserve(indexer_raw_round, ALIGNMENT)?,
            block_rope_cos: builder.reserve(block_rotary, ALIGNMENT)?,
            block_rope_sin: builder.reserve(block_rotary, ALIGNMENT)?,
            closing_first_blocks: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            closing_block_counts: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            candidate_blocks: builder.reserve(rows, ALIGNMENT)?,
            scores: builder.reserve(score_plane, ALIGNMENT)?,
            select_scratch: builder.reserve(SELECTION_SCRATCH_WORDS, ALIGNMENT)?,
            selected: builder.reserve(selected_plane, ALIGNMENT)?,
            selected_counts: builder.reserve(rows, ALIGNMENT)?,

            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            block_keys: builder.reserve(block_key_plane, ALIGNMENT)?,

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
        // A sparse-attention layer's per-sequence carry is its paged cache, never a
        // recurrent plane. Reserving nothing here is the point, and it is checked.
        let persistent = Qwen38FlashNextPersistentState::reserve(&mut builder, layer)?;
        if !matches!(persistent, Qwen38FlashNextPersistentState::Qsa) {
            return Err(EngineError::layout(format!(
                "Qwen3.8-Flash-Next layer {layer} reserved a recurrent carry inside a QSA owner"
            )));
        }

        let resident_weight_bytes = sum(
            "Qwen3.8-Flash-Next QSA/MoE resident weights",
            &[
                regions.attention_hc_norm.byte_len(),
                regions.attention_hc_down.byte_len(),
                regions.attention_hc_up.byte_len(),
                regions.attention_hc_inject.byte_len(),
                regions.mlp_hc_norm.byte_len(),
                regions.mlp_hc_down.byte_len(),
                regions.mlp_hc_up.byte_len(),
                regions.mlp_hc_inject.byte_len(),
                regions.qkv_weight.byte_len(),
                regions.output_weight.byte_len(),
                regions.query_norm.byte_len(),
                regions.key_norm.byte_len(),
                regions.indexer_qk_weight.byte_len(),
                regions.indexer_query_norm.byte_len(),
                regions.indexer_key_norm.byte_len(),
                regions.router_weight.byte_len(),
                regions.expert_weight_scales_2.byte_len(),
                regions.shared_gate_weight.byte_len(),
                regions.shared_up_weight.byte_len(),
                regions.shared_down_weight.byte_len(),
                regions.shared_gate_logit_weight.byte_len(),
            ],
        )?;
        // The ring counts as cache and not workspace: it carries the open micro-block's raw keys
        // between rounds, so a slot recycle has to clear it like any other per-sequence plane.
        // It is not a *page* class, which is exactly why the page cost dropped.
        let cache_bytes = sum(
            "Qwen3.8-Flash-Next QSA cache",
            &[
                regions.key_pages.byte_len(),
                regions.value_pages.byte_len(),
                regions.block_keys.byte_len(),
                regions.indexer_ring.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "Qwen3.8-Flash-Next QSA/MoE workspace",
            &[
                regions.residual_input.byte_len(),
                regions.attention_residual.byte_len(),
                regions.residual_output.byte_len(),
                regions.hc_normalized.byte_len(),
                regions.hc_low_rank.byte_len(),
                regions.hc_mixed.byte_len(),
                regions.hc_write_gate.byte_len(),
                regions.qkv.byte_len(),
                regions.query.byte_len(),
                regions.attention.byte_len(),
                regions.attention_gated.byte_len(),
                regions.rope_cos.byte_len(),
                regions.rope_sin.byte_len(),
                regions.table_rows.byte_len(),
                regions.cache_positions.byte_len(),
                regions.lengths.byte_len(),
                regions.block_tables.byte_len(),
                regions.indexer_qk.byte_len(),
                regions.indexer_query.byte_len(),
                regions.indexer_raw_round.byte_len(),
                regions.block_rope_cos.byte_len(),
                regions.block_rope_sin.byte_len(),
                regions.closing_first_blocks.byte_len(),
                regions.closing_block_counts.byte_len(),
                regions.candidate_blocks.byte_len(),
                regions.scores.byte_len(),
                regions.select_scratch.byte_len(),
                regions.selected.byte_len(),
                regions.selected_counts.byte_len(),
                regions.router_logits.byte_len(),
                regions.expert_indices.byte_len(),
                regions.routing_weights.byte_len(),
                regions.routed_intermediate.byte_len(),
                regions.routed_output.byte_len(),
                regions.shared_intermediate.byte_len(),
                regions.shared_output.byte_len(),
                regions.shared_gate_logit.byte_len(),
                regions.block_output.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            regions,
            resident_weight_bytes,
            cache_bytes,
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
    pub(crate) const fn regions(&self) -> Qwen38FlashNextQsaMoeLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed device weight bytes, excluding the streamed routed experts.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact represented E4M3 key/value bytes, the block-key plane, and the raw-key ring.
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

    /// Decoder layer this layout was built for.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// Context depth one decode slot reaches in this owner.
    pub const fn context_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY
    }

    /// Largest exact row route this owner captures.
    pub const fn row_capacity(&self) -> usize {
        QWEN38_FLASH_NEXT_LAYER_MAX_ROWS
    }
}

impl LayerMemoryLayout for Qwen38FlashNextQsaMoeLayerLayout {
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
    type A = Qwen38FlashNext;
    if A::HC_WIDTH != A::HC_COUNT * <A as Arch>::HIDDEN
        || A::ATTENTION_QKV_ROWS != A::ATTENTION_QUERY_ROWS + 2 * A::ATTENTION_KV_ROWS
        || A::ATTENTION_OUTPUT_COLUMNS != <A as Arch>::NUM_ATTENTION_HEADS * <A as Arch>::HEAD_DIM
        || A::INDEXER_ROWS != (A::INDEXER_HEADS + A::INDEXER_KV_HEADS) * A::INDEXER_HEAD_DIM
        || A::SHARED_EXPERT_INTERMEDIATE != <A as Arch>::INTERMEDIATE
        || A::NUM_EXPERTS_PER_TOKEN != 10
    {
        return Err(EngineError::layout(
            "Qwen3.8-Flash-Next QSA/MoE geometry differs from the qualified layer contract",
        ));
    }

    Ok(())
}

fn require_qsa_layer(layer: usize) -> EngineResult<()> {
    type A = Qwen38FlashNext;
    if layer >= A::LAYERS {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next layer {layer} is outside 0..{}",
            A::LAYERS
        )));
    }
    if !(layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
        return Err(EngineError::layout(format!(
            "Qwen3.8-Flash-Next layer {layer} is a gated-DeltaNet layer, not a sparse-attention layer"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ALIGNMENT, QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE, QWEN38_FLASH_NEXT_LAYER_MAX_ROWS,
        QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY, QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
        QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE, Qwen38FlashNextQsaMoeLayerLayout,
    };
    use crate::{LayerMemoryLayout, MAX_BATCH, QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING};
    use tuisko_gpu::ArenaRegion;
    use tuisko_model::{Arch, Qwen38FlashNext};

    type A = Qwen38FlashNext;

    #[test]
    fn byte_accounting_is_exact() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();

        assert_eq!(layout.resident_weight_bytes(), 141_775_360);
        assert_eq!(layout.cache_bytes(), 35_659_776);
        // 271,864,128 of activations and metadata plus the one selection scratch plane this
        // owner funds for the split selection: 274,944 u32 words = 1,099,776 B, a multiple of
        // the 256-byte alignment, so it costs its own size and no padding.
        assert_eq!(layout.workspace_bytes(), 272_963_904);
        assert_eq!(layout.workspace_bytes(), 271_864_128 + 1_099_776);
        assert_eq!(layout.regions().select_scratch.byte_len(), 1_099_776);
        assert_eq!(layout.owner_bytes(), 450_399_040);
        assert_eq!(layout.arena_bytes(), 450_399_744);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 704);
        assert_eq!(layout.row_capacity(), QWEN38_FLASH_NEXT_LAYER_MAX_ROWS);

        // The inspection trait reports the same three classes the inherent accessors do, and
        // never folds the cache into the weights.
        assert_eq!(
            LayerMemoryLayout::resident_weight_bytes(&layout),
            layout.resident_weight_bytes()
        );
        assert_eq!(
            LayerMemoryLayout::cache_bytes(&layout),
            layout.cache_bytes()
        );
        assert_eq!(
            LayerMemoryLayout::workspace_bytes(&layout),
            layout.workspace_bytes()
        );
        assert_eq!(
            LayerMemoryLayout::arena_bytes(&layout),
            layout.arena_bytes()
        );
    }

    #[test]
    fn the_cache_covers_both_routes_at_every_slot() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();

        assert_eq!(QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE, 64);
        assert_eq!(QWEN38_FLASH_NEXT_QSA_CONTEXT_CAPACITY, 4_096);
        assert!(layout.context_capacity() > QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING);
        // Twice the dense floor, and the doubling is what this owner's second route needs: the
        // selection only runs once a query sees more than 2,051 keys, so a pool sized to
        // `ceil(2051 / 64)` per slot would leave it unreachable and unqualifiable.
        assert_eq!(
            QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE,
            2 * QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
                .div_ceil(QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE)
                - 2
        );
        assert_eq!(
            MAX_BATCH * QWEN38_FLASH_NEXT_QSA_TABLE_STRIDE,
            QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES
        );
    }

    #[test]
    fn the_indexer_pays_for_block_keys_per_page_and_raw_keys_per_sequence() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();
        let regions = layout.regions();

        // 32,768 B per page for K, the same for V, and 4,096 B for compressed block keys:
        // one BF16 vector per four cached tokens rather than one per token.
        assert_eq!(
            regions.key_pages.byte_len() / QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
            32_768
        );
        assert_eq!(
            regions.value_pages.byte_len() / QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
            32_768
        );
        assert_eq!(
            regions.block_keys.byte_len() / QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES,
            4_096
        );

        // The raw keys are not a page class at all: the ring is `MAX_BATCH * 4` vectors however
        // deep the pool runs, which is what took 16,384 B off every page.
        assert_eq!(regions.indexer_ring.byte_len(), 8_192);
        assert_eq!(
            regions.indexer_ring.byte_len(),
            MAX_BATCH * 4 * A::INDEXER_HEAD_DIM * 2
        );

        let per_page = (layout.cache_bytes() - regions.indexer_ring.byte_len())
            / QWEN38_FLASH_NEXT_QSA_PHYSICAL_PAGES;
        assert_eq!(per_page, 69_632);
        // 12 QSA layers x 1,088 B/token, of which 64 is the block-key plane's share.
        assert_eq!(
            per_page / QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE * 12,
            13_056
        );
    }

    #[test]
    fn the_backbone_weights_reproduce_the_resident_plan_s_per_layer_table() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();
        let regions = layout.regions();

        assert_eq!(regions.qkv_weight.byte_len(), 68_157_440);
        assert_eq!(regions.output_weight.byte_len(), 31_457_280);
        assert_eq!(
            regions.query_norm.byte_len() + regions.key_norm.byte_len(),
            1_024
        );
        assert_eq!(regions.indexer_qk_weight.byte_len(), 3_276_800);
        assert_eq!(
            regions.indexer_query_norm.byte_len() + regions.indexer_key_norm.byte_len(),
            512
        );
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();
        let regions = layout.regions();
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
            span(regions.qkv_weight),
            span(regions.output_weight),
            span(regions.query_norm),
            span(regions.key_norm),
            span(regions.indexer_qk_weight),
            span(regions.indexer_query_norm),
            span(regions.indexer_key_norm),
            span(regions.qkv),
            span(regions.query),
            span(regions.attention),
            span(regions.attention_gated),
            span(regions.rope_cos),
            span(regions.rope_sin),
            span(regions.table_rows),
            span(regions.cache_positions),
            span(regions.lengths),
            span(regions.block_tables),
            span(regions.indexer_qk),
            span(regions.indexer_query),
            span(regions.indexer_ring),
            span(regions.indexer_raw_round),
            span(regions.block_rope_cos),
            span(regions.block_rope_sin),
            span(regions.closing_first_blocks),
            span(regions.closing_block_counts),
            span(regions.candidate_blocks),
            span(regions.scores),
            span(regions.select_scratch),
            span(regions.selected),
            span(regions.selected_counts),
            span(regions.key_pages),
            span(regions.value_pages),
            span(regions.block_keys),
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
    fn the_attention_planes_match_the_qsa_op_contracts() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();
        let regions = layout.regions();
        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;

        // qk-prepare reads qkv [tokens, 13312] and writes query [tokens, 24, 256] f32;
        // the gate reads the same qkv and writes activation [tokens, 6144].
        assert_eq!(regions.qkv.len(), rows * A::ATTENTION_QKV_ROWS);
        assert_eq!(
            regions.query.len(),
            rows * <A as Arch>::NUM_ATTENTION_HEADS * <A as Arch>::HEAD_DIM
        );
        assert_eq!(regions.attention.len(), rows * A::ATTENTION_OUTPUT_COLUMNS);
        assert_eq!(
            regions.attention_gated.len(),
            rows * A::ATTENTION_OUTPUT_COLUMNS
        );
        assert_eq!(regions.rope_cos.len(), regions.rope_sin.len());
    }

    #[test]
    fn a_gated_deltanet_layer_is_refused_by_this_owner() {
        for layer in [0, 1, 2, 46] {
            let error = Qwen38FlashNextQsaMoeLayerLayout::build(layer).unwrap_err();
            assert!(error.to_string().contains("gated-DeltaNet layer"));
        }
        assert!(Qwen38FlashNextQsaMoeLayerLayout::build(A::LAYERS).is_err());
    }

    #[test]
    fn every_sparse_attention_layer_index_builds() {
        let built = (0..A::LAYERS)
            .filter(|layer| Qwen38FlashNextQsaMoeLayerLayout::build(*layer).is_ok())
            .collect::<Vec<_>>();

        assert_eq!(built, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
        assert_eq!(built.len(), 12);
    }

    #[test]
    fn both_layer_kinds_carry_the_same_mlp_half() {
        use crate::Qwen38FlashNextGdnMoeLayerLayout;

        let gdn = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        let qsa = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();

        // Spec 1.3: MoE is present in both layer kinds with identical configuration.
        assert_eq!(
            qsa.regions().router_weight.byte_len(),
            gdn.regions().router_weight.byte_len()
        );
        assert_eq!(
            qsa.regions().routed_output.len(),
            gdn.regions().routed_output.len()
        );
        assert_eq!(
            qsa.regions().shared_gate_logit_weight.byte_len(),
            gdn.regions().shared_gate_logit_weight.byte_len()
        );
    }

    /// The same backbone projections the GDN owner records, in this layer's two shapes.
    ///
    /// `Qwen38FlashNextAttentionQkPrepareOp::launch` and `Qwen38FlashNextAttentionGateOp::launch` both read
    /// `qkv` covering `[tokens, 13312]`, and the gate publishes `activation` covering
    /// `[tokens, 6144]`. So this layer needs
    ///
    /// * `q || k || v`: `[rows, 2560] x [13312, 2560]^T -> [rows, 13312]`, and
    /// * `o_proj`: `[rows, 6144] x [ 2560, 6144]^T -> [rows,  2560]`.
    ///
    /// `o_proj` and the GDN layer's `out_proj` are the **same shape**, so one entry family
    /// serves both layer kinds and this owner names the same reduction entry the GDN owner
    /// does, with its own weight plane.
    #[test]
    fn the_backbone_projections_have_reserved_planes_and_admitted_producers() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();
        let regions = layout.regions();
        let rows = QWEN38_FLASH_NEXT_LAYER_MAX_ROWS;

        assert_eq!(regions.qkv.len(), rows * A::ATTENTION_QKV_ROWS);
        assert_eq!(regions.block_output.len(), rows * <A as Arch>::HIDDEN);

        // The output projection this layer needs is byte-for-byte the shape the GDN layer's
        // out_proj needs, which is why the family is three shapes and not four.
        use crate::Qwen38FlashNextGdnMoeLayerLayout;
        let gdn = Qwen38FlashNextGdnMoeLayerLayout::build(0).unwrap();
        assert_eq!(
            regions.output_weight.byte_len(),
            gdn.regions().gdn_output_weight.byte_len()
        );
        assert_eq!(
            regions.attention_gated.len(),
            gdn.regions().gdn_recurrent_output.len()
        );

        let entries = tuisko_kernels_sm120::kernel_ptx_names();
        for base in [
            "qwen38_flash_next_qsa_qkv_projection",
            "qwen38_flash_next_block_output_projection",
        ] {
            let routes = entries.iter().filter(|name| name.starts_with(base)).count();
            assert_eq!(
                routes, 12,
                "{base} does not cover the twelve admitted routes"
            );
        }
    }

    /// The arena this layout describes is the one the composed program allocates.
    ///
    /// The program that consumes it is qualified against real weights in
    /// `qual/src/qwen38_flash_next_qsa_moe_layer.rs`, which supersedes the placeholder this test used
    /// to be.
    #[test]
    fn the_builder_describes_the_whole_arena_the_program_allocates() {
        let layout = Qwen38FlashNextQsaMoeLayerLayout::build(3).unwrap();

        assert_eq!(layout.builder().byte_len(), layout.arena_bytes());
        assert!(layout.arena_bytes() >= layout.owner_bytes());
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

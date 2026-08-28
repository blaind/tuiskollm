//! Memory plan for the Qwen3.8 Flash-Next MTP block.
//!
//! Non-expert weights stay resident. The distinct BF16 expert pool uses 128 streaming slots;
//! its 9,830,400-byte stride cannot share the target's NVFP4 pool. Draft cache and workspace
//! remain separate so rejected drafts cannot corrupt provisional target state.

use crate::common::math::{product, sum};
use crate::common::mtp::VERIFY_ROWS;
use crate::common::streaming::{StreamingPrimarySource, StreamingWeightLayout};
use crate::qwen38_flash_next::layer_upload::{HyperConnectionRegions, MoeRegions};
use crate::qwen38_flash_next::persistent_state::ALIGNMENT;
use crate::qwen38_flash_next::qsa_moe_layer_layout::QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;
use crate::qwen38_flash_next::resident_model_layout::{
    QWEN38_FLASH_NEXT_PRIMARY_SOURCE, QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS,
    Qwen38FlashNextQsaWeightRegions,
};
use crate::{EngineError, EngineResult, MAX_BATCH, StreamingResidencyAccounting};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen38FlashNext};

type A = Qwen38FlashNext;

/// Routed experts in the draft block's pool.
pub const QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT: usize = A::NUM_EXPERTS;

/// Contiguous BF16 bytes one draft expert occupies.
pub const QWEN38_FLASH_NEXT_MTP_EXPERT_EXTENT_BYTES: usize =
    MTP_GATE_UP_EXTENT_BYTES + MTP_DOWN_EXTENT_BYTES;

const MTP_GATE_UP_EXTENT_BYTES: usize =
    2 * A::INTERMEDIATE * <A as Arch>::HIDDEN * size_of::<u16>();
const MTP_DOWN_EXTENT_BYTES: usize = <A as Arch>::HIDDEN * A::INTERMEDIATE * size_of::<u16>();

/// Device slots the draft pool funds: the same 25 % posture the main pool runs at.
pub const QWEN38_FLASH_NEXT_MTP_EXPERT_RESIDENT_SLOTS: usize = 128;

/// One mapped-primary bounce extent per decode row.
const QWEN38_FLASH_NEXT_MTP_BOUNCE_RING_SLOTS: usize = MAX_BATCH;

/// Widest exact row count captured by the draft block.
pub const QWEN38_FLASH_NEXT_MTP_MAX_ROWS: usize = QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS;

/// Cache rows one MTP round appends across all slots.
pub const QWEN38_FLASH_NEXT_MTP_ROUND_ROWS: usize = MAX_BATCH * VERIFY_ROWS;

const _: () = assert!(QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT == 512);
const _: () = assert!(QWEN38_FLASH_NEXT_MTP_EXPERT_EXTENT_BYTES == 9_830_400);
const _: () = assert!(QWEN38_FLASH_NEXT_MTP_ROUND_ROWS == 32);

/// Input-fusion weights over embeddings and the pre-mixer target stream.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct Qwen38FlashNextMtpFusionRegions {
    /// RMSNorm gain over the embedding term `[HIDDEN]`.
    pub(crate) norm_embedding: ArenaRegion<u16>,
    /// Grouped RMSNorm gain over the stream term `[HC_WIDTH]`.
    pub(crate) norm_hidden: ArenaRegion<u16>,
    /// Embedding-term projection `[HIDDEN, HIDDEN]`, shared by all four branches.
    pub(crate) fc_embedding: ArenaRegion<u16>,
    /// Stream-term projection `[HIDDEN, HIDDEN]`, applied to each branch in turn.
    pub(crate) fc_hidden: ArenaRegion<u16>,
}

/// Resident weights for the draft block's sparse-attention/MoE layer.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct Qwen38FlashNextMtpLayerRegions {
    pub(crate) attention_hc: HyperConnectionRegions,
    pub(crate) mlp_hc: HyperConnectionRegions,
    pub(crate) attention: Qwen38FlashNextQsaWeightRegions,
    pub(crate) moe: MoeRegions,
}

/// Draft-only activations that preserve provisional target state.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct Qwen38FlashNextMtpWorkspace {
    /// The draft's own four-branch stream, published by the input fusion.
    pub(crate) residual_a: ArenaRegion<u16>,
    /// The stream the attention write-back publishes.
    pub(crate) residual_b: ArenaRegion<u16>,

    /// Host-gathered embedding rows of the tokens being drafted.
    pub(crate) embedding_rows: ArenaRegion<u16>,
    /// The normalized embedding term, before `fc_embedding`.
    pub(crate) fusion_embedding: ArenaRegion<u16>,
    /// The grouped-normalized stream term, before `fc_hidden`.
    pub(crate) fusion_hidden: ArenaRegion<u16>,
    /// `fc_embedding(enorm(e))`, one 2,560-wide row broadcast to every branch.
    pub(crate) fusion_projected: ArenaRegion<u16>,

    pub(crate) hc_normalized: ArenaRegion<u16>,
    pub(crate) hc_low_rank: ArenaRegion<u16>,
    pub(crate) hc_mixed: ArenaRegion<u16>,
    pub(crate) hc_write_gate: ArenaRegion<u16>,

    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_gated: ArenaRegion<u16>,

    pub(crate) indexer_qk: ArenaRegion<u16>,
    pub(crate) indexer_query: ArenaRegion<f32>,
    pub(crate) indexer_raw_round: ArenaRegion<u16>,

    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_intermediate: ArenaRegion<u16>,
    pub(crate) routed_output: ArenaRegion<u16>,
    pub(crate) shared_intermediate: ArenaRegion<u16>,
    pub(crate) shared_output: ArenaRegion<u16>,
    pub(crate) shared_gate_logit: ArenaRegion<u16>,

    pub(crate) block_output: ArenaRegion<u16>,

    /// The collapsing mixer's three staging planes.
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) mixer_low_rank: ArenaRegion<u16>,
    pub(crate) mixer_mixed: ArenaRegion<u16>,

    // Runtime inputs staged per round.
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,
}

/// Draft-only K/V, block-key, and indexer planes.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct Qwen38FlashNextMtpKvPlanes {
    pub(crate) key_pages: ArenaRegion<u8>,
    pub(crate) value_pages: ArenaRegion<u8>,
    pub(crate) block_keys: ArenaRegion<u16>,
    pub(crate) indexer_ring: ArenaRegion<u16>,
}

/// Checked three-arena plan of the Flash-Next draft block.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMtpLayout {
    resident: ArenaLayout,
    kv: ArenaLayout,
    streaming: StreamingWeightLayout,
    #[allow(dead_code)]
    fusion: Qwen38FlashNextMtpFusionRegions,
    #[allow(dead_code)]
    layer: Qwen38FlashNextMtpLayerRegions,
    #[allow(dead_code)]
    mixer: HyperConnectionRegions,
    #[allow(dead_code)]
    workspace: Qwen38FlashNextMtpWorkspace,
    #[allow(dead_code)]
    kv_planes: Qwen38FlashNextMtpKvPlanes,
    resident_weight_bytes: usize,
}

impl Qwen38FlashNextMtpLayout {
    /// Plans the draft block at the adopted posture.
    pub fn build(physical_pages: usize) -> EngineResult<Self> {
        Self::plan(QWEN38_FLASH_NEXT_MTP_EXPERT_RESIDENT_SLOTS, physical_pages)
    }

    /// Plans the draft block at an explicit slot and page budget.
    pub fn plan(slot_count: usize, physical_pages: usize) -> EngineResult<Self> {
        require_geometry()?;

        // BF16 experts have no secondary scale extent.
        let streaming = match QWEN38_FLASH_NEXT_PRIMARY_SOURCE {
            StreamingPrimarySource::Pinned => StreamingWeightLayout::build(
                QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT,
                QWEN38_FLASH_NEXT_MTP_EXPERT_EXTENT_BYTES,
                None,
                slot_count,
            )?,
            StreamingPrimarySource::Mapped => StreamingWeightLayout::build_mapped_primary(
                QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT,
                MTP_GATE_UP_EXTENT_BYTES,
                Some(MTP_DOWN_EXTENT_BYTES),
                slot_count,
                QWEN38_FLASH_NEXT_MTP_BOUNCE_RING_SLOTS,
            )?,
        };

        let mut resident = ArenaLayout::new();
        let fusion = reserve_fusion(&mut resident)?;
        let layer = reserve_layer(&mut resident)?;
        let mixer = reserve_mixer(&mut resident)?;
        let resident_weight_bytes = weight_bytes(fusion, layer, mixer)?;
        let workspace = reserve_workspace(&mut resident)?;

        let mut kv = ArenaLayout::new();
        let kv_planes = reserve_kv(&mut kv, physical_pages)?;

        Ok(Self {
            resident,
            kv,
            streaming,
            fusion,
            layer,
            mixer,
            workspace,
            kv_planes,
            resident_weight_bytes,
        })
    }

    /// Device bytes the draft block's resident arena occupies.
    pub const fn resident_arena_bytes(&self) -> usize {
        self.resident.byte_len()
    }

    /// Device bytes the draft block's own cache mirror occupies.
    pub const fn kv_arena_bytes(&self) -> usize {
        self.kv.byte_len()
    }

    /// The draft pool's plan.
    pub const fn streaming(&self) -> &StreamingWeightLayout {
        &self.streaming
    }

    /// Resident weight bytes, excluding workspace and cache.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Every device byte the draft block adds, its streaming slots included.
    pub fn total_device_bytes(&self) -> EngineResult<usize> {
        sum(
            "Flash-Next MTP device bytes",
            &[
                self.resident_arena_bytes(),
                self.kv_arena_bytes(),
                self.streaming.device_resident_bytes(),
            ],
        )
    }

    #[allow(dead_code)]
    pub(crate) const fn fusion(&self) -> Qwen38FlashNextMtpFusionRegions {
        self.fusion
    }

    #[allow(dead_code)]
    pub(crate) const fn layer(&self) -> Qwen38FlashNextMtpLayerRegions {
        self.layer
    }

    #[allow(dead_code)]
    pub(crate) const fn mixer(&self) -> HyperConnectionRegions {
        self.mixer
    }

    #[allow(dead_code)]
    pub(crate) const fn workspace(&self) -> Qwen38FlashNextMtpWorkspace {
        self.workspace
    }

    #[allow(dead_code)]
    pub(crate) const fn kv_planes(&self) -> Qwen38FlashNextMtpKvPlanes {
        self.kv_planes
    }
}

fn reserve_fusion(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextMtpFusionRegions> {
    let square = product(
        "Flash-Next MTP fusion projection",
        <A as Arch>::HIDDEN,
        <A as Arch>::HIDDEN,
    )?;

    Ok(Qwen38FlashNextMtpFusionRegions {
        norm_embedding: builder.reserve(<A as Arch>::HIDDEN, ALIGNMENT)?,
        norm_hidden: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        fc_embedding: builder.reserve(square, ALIGNMENT)?,
        fc_hidden: builder.reserve(square, ALIGNMENT)?,
    })
}

fn reserve_hyper_connection(
    builder: &mut ArenaLayout,
    combines: bool,
) -> EngineResult<HyperConnectionRegions> {
    let projection = product(
        "Flash-Next MTP hyper-connection projection",
        A::HC_LOWRANK,
        A::HC_WIDTH,
    )?;
    let inject = product(
        "Flash-Next MTP hyper-connection inject",
        A::HC_COUNT,
        A::HC_WIDTH,
    )?;

    Ok(HyperConnectionRegions {
        norm: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        down: builder.reserve(projection, ALIGNMENT)?,
        up: builder.reserve(projection, ALIGNMENT)?,
        // A collapsing mixer has no write-back gate.
        inject: builder.reserve(if combines { inject } else { 0 }, ALIGNMENT)?,
    })
}

fn reserve_layer(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextMtpLayerRegions> {
    let attention_hc = reserve_hyper_connection(builder, true)?;
    let mlp_hc = reserve_hyper_connection(builder, true)?;
    let qkv_weight = product(
        "Flash-Next MTP fused projection",
        A::ATTENTION_QKV_ROWS,
        <A as Arch>::HIDDEN,
    )?;
    let output_weight = product(
        "Flash-Next MTP attention output projection",
        <A as Arch>::HIDDEN,
        A::ATTENTION_OUTPUT_COLUMNS,
    )?;
    let indexer_qk_weight = product(
        "Flash-Next MTP indexer projection",
        A::INDEXER_ROWS,
        <A as Arch>::HIDDEN,
    )?;
    let attention = Qwen38FlashNextQsaWeightRegions {
        qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
        output_weight: builder.reserve(output_weight, ALIGNMENT)?,
        query_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
        key_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
        indexer_qk_weight: builder.reserve(indexer_qk_weight, ALIGNMENT)?,
        indexer_query_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,
        indexer_key_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,
    };

    let router_weight = product(
        "Flash-Next MTP router weight",
        A::NUM_EXPERTS,
        <A as Arch>::HIDDEN,
    )?;
    let shared_gate_up = product(
        "Flash-Next MTP shared expert projection",
        A::SHARED_EXPERT_INTERMEDIATE,
        <A as Arch>::HIDDEN,
    )?;
    // BF16 experts have no ModelOpt second-stage scale.
    let moe = MoeRegions {
        router_weight: builder.reserve(router_weight, ALIGNMENT)?,
        shared_gate_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_up_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_down_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_gate_logit_weight: builder.reserve(<A as Arch>::HIDDEN, ALIGNMENT)?,
        expert_weight_scales_2: builder.reserve(0, ALIGNMENT)?,
    };

    Ok(Qwen38FlashNextMtpLayerRegions {
        attention_hc,
        mlp_hc,
        attention,
        moe,
    })
}

fn reserve_mixer(builder: &mut ArenaLayout) -> EngineResult<HyperConnectionRegions> {
    reserve_hyper_connection(builder, false)
}

fn reserve_workspace(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextMtpWorkspace> {
    let rows = QWEN38_FLASH_NEXT_MTP_MAX_ROWS;
    let row_stream = product("Flash-Next MTP stream rows", rows, A::HC_WIDTH)?;
    let row_hidden = product("Flash-Next MTP hidden rows", rows, <A as Arch>::HIDDEN)?;
    let row_low_rank = product("Flash-Next MTP low-rank rows", rows, A::HC_LOWRANK)?;
    let row_branches = product("Flash-Next MTP branch rows", rows, A::HC_COUNT)?;
    let qkv = product("Flash-Next MTP QKV rows", rows, A::ATTENTION_QKV_ROWS)?;
    let query = product(
        "Flash-Next MTP query rows",
        rows,
        product(
            "Flash-Next MTP query width",
            <A as Arch>::NUM_ATTENTION_HEADS,
            <A as Arch>::HEAD_DIM,
        )?,
    )?;
    let attention_out = product(
        "Flash-Next MTP attention rows",
        rows,
        A::ATTENTION_OUTPUT_COLUMNS,
    )?;
    let indexer_qk = product("Flash-Next MTP indexer rows", rows, A::INDEXER_ROWS)?;
    let indexer_query = product(
        "Flash-Next MTP indexer query rows",
        rows,
        product(
            "Flash-Next MTP indexer query width",
            A::INDEXER_HEADS,
            A::INDEXER_HEAD_DIM,
        )?,
    )?;
    let indexer_raw = product("Flash-Next MTP indexer raw rows", rows, A::INDEXER_HEAD_DIM)?;
    let routed = product(
        "Flash-Next MTP routed rows",
        rows,
        product(
            "Flash-Next MTP routed width",
            A::NUM_EXPERTS_PER_TOKEN,
            A::INTERMEDIATE,
        )?,
    )?;
    let shared = product(
        "Flash-Next MTP shared rows",
        rows,
        A::SHARED_EXPERT_INTERMEDIATE,
    )?;
    let experts_selected = product(
        "Flash-Next MTP selected experts",
        rows,
        A::NUM_EXPERTS_PER_TOKEN,
    )?;
    let router_logits = product("Flash-Next MTP router logits", rows, A::NUM_EXPERTS)?;

    Ok(Qwen38FlashNextMtpWorkspace {
        residual_a: builder.reserve(row_stream, ALIGNMENT)?,
        residual_b: builder.reserve(row_stream, ALIGNMENT)?,

        embedding_rows: builder.reserve(row_hidden, ALIGNMENT)?,
        fusion_embedding: builder.reserve(row_hidden, ALIGNMENT)?,
        fusion_hidden: builder.reserve(row_stream, ALIGNMENT)?,
        fusion_projected: builder.reserve(row_hidden, ALIGNMENT)?,

        hc_normalized: builder.reserve(row_stream, ALIGNMENT)?,
        hc_low_rank: builder.reserve(row_low_rank, ALIGNMENT)?,
        hc_mixed: builder.reserve(row_hidden, ALIGNMENT)?,
        hc_write_gate: builder.reserve(row_branches, ALIGNMENT)?,

        qkv: builder.reserve(qkv, ALIGNMENT)?,
        query: builder.reserve(query, ALIGNMENT)?,
        attention: builder.reserve(attention_out, ALIGNMENT)?,
        attention_gated: builder.reserve(attention_out, ALIGNMENT)?,

        indexer_qk: builder.reserve(indexer_qk, ALIGNMENT)?,
        indexer_query: builder.reserve(indexer_query, ALIGNMENT)?,
        indexer_raw_round: builder.reserve(indexer_raw, ALIGNMENT)?,

        router_logits: builder.reserve(router_logits, ALIGNMENT)?,
        expert_indices: builder.reserve(experts_selected, ALIGNMENT)?,
        routing_weights: builder.reserve(experts_selected, ALIGNMENT)?,
        routed_intermediate: builder.reserve(routed, ALIGNMENT)?,
        routed_output: builder.reserve(row_hidden, ALIGNMENT)?,
        shared_intermediate: builder.reserve(shared, ALIGNMENT)?,
        shared_output: builder.reserve(row_hidden, ALIGNMENT)?,
        shared_gate_logit: builder.reserve(rows, ALIGNMENT)?,

        block_output: builder.reserve(row_hidden, ALIGNMENT)?,

        mixer_normalized: builder.reserve(row_stream, ALIGNMENT)?,
        mixer_low_rank: builder.reserve(row_low_rank, ALIGNMENT)?,
        mixer_mixed: builder.reserve(row_hidden, ALIGNMENT)?,

        table_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
        cache_positions: builder.reserve(rows, ALIGNMENT)?,
        lengths: builder.reserve(MAX_BATCH, ALIGNMENT)?,
        rope_cos: builder.reserve(
            product(
                "Flash-Next MTP rotary rows",
                rows,
                <A as Arch>::HEAD_DIM / 4,
            )?,
            ALIGNMENT,
        )?,
        rope_sin: builder.reserve(
            product(
                "Flash-Next MTP rotary rows",
                rows,
                <A as Arch>::HEAD_DIM / 4,
            )?,
            ALIGNMENT,
        )?,
    })
}

fn reserve_kv(builder: &mut ArenaLayout, pages: usize) -> EngineResult<Qwen38FlashNextMtpKvPlanes> {
    let tokens = product(
        "Flash-Next MTP cache tokens",
        pages,
        QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
    )?;
    let plane = product(
        "Flash-Next MTP cache plane",
        tokens,
        product(
            "Flash-Next MTP cache width",
            <A as Arch>::NUM_KV_HEADS,
            <A as Arch>::HEAD_DIM,
        )?,
    )?;
    let block_keys = product(
        "Flash-Next MTP block keys",
        tokens / A::INDEXER_COMPRESS_RATIO,
        A::INDEXER_HEAD_DIM,
    )?;
    let ring = product(
        "Flash-Next MTP indexer ring",
        product(
            "Flash-Next MTP ring rows",
            MAX_BATCH,
            A::INDEXER_COMPRESS_RATIO,
        )?,
        A::INDEXER_HEAD_DIM,
    )?;

    Ok(Qwen38FlashNextMtpKvPlanes {
        key_pages: builder.reserve(plane, ALIGNMENT)?,
        value_pages: builder.reserve(plane, ALIGNMENT)?,
        block_keys: builder.reserve(block_keys, ALIGNMENT)?,
        indexer_ring: builder.reserve(ring, ALIGNMENT)?,
    })
}

fn hyper_connection_bytes(regions: HyperConnectionRegions) -> EngineResult<usize> {
    sum(
        "Flash-Next MTP hyper-connection weight bytes",
        &[
            regions.norm.byte_len(),
            regions.down.byte_len(),
            regions.up.byte_len(),
            regions.inject.byte_len(),
        ],
    )
}

fn weight_bytes(
    #[allow(dead_code)] fusion: Qwen38FlashNextMtpFusionRegions,
    #[allow(dead_code)] layer: Qwen38FlashNextMtpLayerRegions,
    #[allow(dead_code)] mixer: HyperConnectionRegions,
) -> EngineResult<usize> {
    sum(
        "Flash-Next MTP resident weight bytes",
        &[
            fusion.norm_embedding.byte_len(),
            fusion.norm_hidden.byte_len(),
            fusion.fc_embedding.byte_len(),
            fusion.fc_hidden.byte_len(),
            hyper_connection_bytes(layer.attention_hc)?,
            hyper_connection_bytes(layer.mlp_hc)?,
            layer.attention.qkv_weight.byte_len(),
            layer.attention.output_weight.byte_len(),
            layer.attention.query_norm.byte_len(),
            layer.attention.key_norm.byte_len(),
            layer.attention.indexer_qk_weight.byte_len(),
            layer.attention.indexer_query_norm.byte_len(),
            layer.attention.indexer_key_norm.byte_len(),
            layer.moe.router_weight.byte_len(),
            layer.moe.shared_gate_weight.byte_len(),
            layer.moe.shared_up_weight.byte_len(),
            layer.moe.shared_down_weight.byte_len(),
            layer.moe.shared_gate_logit_weight.byte_len(),
            hyper_connection_bytes(mixer)?,
        ],
    )
}

fn require_geometry() -> EngineResult<()> {
    if A::MTP_LAYERS != 1 {
        return Err(EngineError::layout(
            "the Flash-Next draft block is one hybrid layer; no other depth is planned",
        ));
    }
    if A::MTP_USES_DEDICATED_EMBEDDINGS {
        return Err(EngineError::layout(
            "the Flash-Next draft block shares the target's embedding table and LM head; a \
             dedicated-embedding checkpoint would need endpoints this layout does not reserve",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draft_pool_streams_because_holding_it_costs_the_target_cache() {
        let layout = Qwen38FlashNextMtpLayout::build(1).unwrap();
        let pool = layout.streaming();

        assert_eq!(pool.item_count(), 512);
        assert_eq!(pool.primary_extent_bytes(), 6_553_600);
        assert_eq!(pool.secondary_extent_bytes(), 3_276_800);
        assert_eq!(pool.slot_count(), 128);
        assert_eq!(pool.host_pool_bytes(), 1_677_721_600);
        assert_eq!(pool.host_mapped_bytes(), 3_355_443_200);
        assert_eq!(pool.bounce_ring_bytes(), 52_428_800);

        assert_eq!(
            pool.stride_bytes(),
            QWEN38_FLASH_NEXT_MTP_EXPERT_EXTENT_BYTES
        );
        assert_eq!(pool.slot_count() * pool.stride_bytes(), 1_258_291_200);

        let resident_pool = pool.item_count() * pool.stride_bytes();

        assert_eq!(resident_pool, 5_033_164_800);
        assert!(
            resident_pool - pool.slot_count() * pool.stride_bytes() > 3_000_000_000,
            "holding the draft pool must cost more than 3 GB of target cache"
        );
    }

    #[test]
    fn the_draft_blocks_resident_weights_close_at_the_planned_total() {
        let layout = Qwen38FlashNextMtpLayout::build(1).unwrap();

        assert_eq!(layout.resident_weight_bytes(), 181_136_896);

        assert_eq!(
            layout.resident_weight_bytes() + layout.streaming().slot_count() * 9_830_400,
            181_136_896 + 1_258_291_200
        );
    }

    #[test]
    fn the_draft_cache_scales_with_pages_and_not_with_the_round() {
        assert_eq!(QWEN38_FLASH_NEXT_MTP_ROUND_ROWS, MAX_BATCH * VERIFY_ROWS);
        assert_eq!(QWEN38_FLASH_NEXT_MTP_ROUND_ROWS, 32);

        let narrow = Qwen38FlashNextMtpLayout::build(64).unwrap();
        let wide = Qwen38FlashNextMtpLayout::build(512).unwrap();

        assert!(wide.kv_arena_bytes() > 7 * narrow.kv_arena_bytes());
        assert!(wide.kv_arena_bytes() < 8 * narrow.kv_arena_bytes());

        let tokens = 64 * QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;
        assert!(
            tokens > QWEN38_FLASH_NEXT_MTP_ROUND_ROWS * 8,
            "the draft cache must hold a prefix, not one round"
        );

        assert_eq!(wide.resident_arena_bytes(), narrow.resident_arena_bytes());
    }

    #[test]
    fn the_draft_block_reserves_nothing_it_has_no_tensors_for() {
        let layout = Qwen38FlashNextMtpLayout::build(1).unwrap();

        assert_eq!(layout.mixer().inject.byte_len(), 0);
        assert!(layout.layer().attention_hc.inject.byte_len() > 0);
        assert!(layout.layer().mlp_hc.inject.byte_len() > 0);

        assert_eq!(layout.layer().moe.expert_weight_scales_2.byte_len(), 0);
    }
}

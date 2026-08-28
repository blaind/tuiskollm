//! Whole-model memory plan for Qwen3.8 Flash-Next.
//!
//! All 48 layers share one workspace and alternate two residual planes. Routed experts live in
//! the streaming pool; each layer views its 512-entry window in the global slot table.

use crate::common::math::{checked_sum, product, sum};
use crate::common::streaming::{StreamingPrimarySource, StreamingWeightLayout};
use crate::qwen38_flash_next::engram_stager_layout::Qwen38FlashNextEngramStagerLayout;
use crate::qwen38_flash_next::layer_upload::{HyperConnectionRegions, MoeRegions};
use crate::qwen38_flash_next::persistent_state::{ALIGNMENT, Qwen38FlashNextPersistentState};
use crate::qwen38_flash_next::qsa_moe_layer_layout::QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;
use crate::{
    EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH, StreamingResidencyAccounting,
};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen38FlashNext};

type A = Qwen38FlashNext;

/// Widest exact row count the resident program captures, the `T=1024` prefill tile.
pub const QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS: usize = 1_024;

/// One in-flight primary extent per admitted decode row.
const QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS: usize = MAX_BATCH;

/// Rotary elements one token carries: `rotary_dim / 2` at `partial_rotary_factor = 0.25`.
const ROTARY_ELEMENTS: usize = 32;

/// Scratch slots one token's routed experts occupy, one per selected rank.
const ROUTED_SLOTS: usize = A::NUM_EXPERTS_PER_TOKEN;

/// Per-expert `weight_scale_2` scalars the routed kernels read: gate, up, then down.
const EXPERT_WEIGHT_SCALES: usize = 3;

/// One item per `(layer, expert)` pair.
pub const QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT: usize = A::LAYERS * A::NUM_EXPERTS;

/// Device slots funded by the admitted 25% expert-residency posture.
pub const QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS: usize = 6_144;

/// Contiguous packed E2M1 `down`, `gate`, and `up` bytes per expert.
pub const QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES: usize = 2_457_600;

/// Swizzled E4M3 block scales per expert.
pub const QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES: usize = 307_200;

/// Usable device bytes on the admitted product target, an RTX 5090 at 29.5 GiB.
pub const QWEN38_FLASH_NEXT_DEVICE_BUDGET_BYTES: usize = 31_675_383_808;

/// Device bytes the house holds back from every capacity plan.
pub const QWEN38_FLASH_NEXT_REQUIRED_HEADROOM_BYTES: usize = 1_000_000_000;

/// Physical pages required by the full 262,144-token context.
pub const QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES: usize =
    262_144 / QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE;

/// Decoder layers that run dense gated GQA, at `(layer + 1) % 4 == 0`.
pub const QWEN38_FLASH_NEXT_ATTENTION_LAYERS: usize = A::LAYERS / A::FULL_ATTENTION_INTERVAL;

/// One segment per layer router plus the endpoint tail.
pub const QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS: usize = A::LAYERS + 1;

/// Host posture admitted for the packed primary extent on the product machine.
pub const QWEN38_FLASH_NEXT_PRIMARY_SOURCE: StreamingPrimarySource = StreamingPrimarySource::Mapped;

/// The engram module's own I64 hash constants, gate-checked rather than reserved as planes.
const _: () = assert!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS == 49);
const _: () = assert!(QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT == 24_576);
const _: () = assert!(QWEN38_FLASH_NEXT_ATTENTION_LAYERS == 12);
const _: () = assert!(QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES == 4_096);

/// The seven device planes one GDN block's weights occupy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextGdnWeightRegions {
    pub(crate) input_weight: ArenaRegion<u16>,
    pub(crate) control_weight: ArenaRegion<u16>,
    pub(crate) convolution_weight: ArenaRegion<u16>,
    pub(crate) a_log: ArenaRegion<u16>,
    pub(crate) dt_bias: ArenaRegion<u16>,
    pub(crate) norm: ArenaRegion<u16>,
    pub(crate) output_weight: ArenaRegion<u16>,
}

/// The seven device planes one QSA block's weights occupy, indexer included.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextQsaWeightRegions {
    pub(crate) qkv_weight: ArenaRegion<u16>,
    pub(crate) output_weight: ArenaRegion<u16>,
    pub(crate) query_norm: ArenaRegion<u16>,
    pub(crate) key_norm: ArenaRegion<u16>,
    /// Reserved for the indexer route; the dense route never reads it.
    pub(crate) indexer_qk_weight: ArenaRegion<u16>,
    pub(crate) indexer_query_norm: ArenaRegion<u16>,
    pub(crate) indexer_key_norm: ArenaRegion<u16>,
}

/// The six device planes the engram module's weights occupy, on the one layer that runs one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextPleWeightRegions {
    pub(crate) key_proj: ArenaRegion<u16>,
    pub(crate) value_proj: ArenaRegion<u16>,
    pub(crate) norm_key: ArenaRegion<u16>,
    pub(crate) norm_query: ArenaRegion<u16>,
    pub(crate) norm_conv: ArenaRegion<u16>,
    pub(crate) convolution: ArenaRegion<u16>,
}

/// The middle block one decoder layer runs, by kind.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Qwen38FlashNextBlockWeightRegions {
    /// The 36 gated-DeltaNet layers.
    Gdn(Qwen38FlashNextGdnWeightRegions),
    /// The 12 dense sparse-attention layers.
    Qsa(Qwen38FlashNextQsaWeightRegions),
}

/// Every resident weight plane one decoder layer owns.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextResidentLayerRegions {
    pub(crate) attention_hc: HyperConnectionRegions,
    pub(crate) mlp_hc: HyperConnectionRegions,
    pub(crate) moe: MoeRegions,
    pub(crate) block: Qwen38FlashNextBlockWeightRegions,
    pub(crate) ple: Option<Qwen38FlashNextPleWeightRegions>,
    pub(crate) persistent: Qwen38FlashNextPersistentState,
}

/// The one activation set every decoder layer addresses in turn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextResidentWorkspace {
    /// The four-branch stream a layer reads and its MLP write-back republishes.
    pub(crate) residual_a: ArenaRegion<u16>,
    /// The four-branch stream a layer's attention write-back publishes.
    pub(crate) residual_b: ArenaRegion<u16>,

    pub(crate) hc_normalized: ArenaRegion<u16>,
    pub(crate) hc_low_rank: ArenaRegion<u16>,
    pub(crate) hc_mixed: ArenaRegion<u16>,
    pub(crate) hc_write_gate: ArenaRegion<u16>,

    pub(crate) gdn_projected: ArenaRegion<u16>,
    pub(crate) gdn_convolved: ArenaRegion<u16>,
    pub(crate) gdn_log_decay: ArenaRegion<f32>,
    pub(crate) gdn_beta: ArenaRegion<f32>,
    pub(crate) gdn_recurrent_plane: ArenaRegion<f32>,
    pub(crate) gdn_recurrent_output: ArenaRegion<u16>,

    pub(crate) qkv: ArenaRegion<u16>,
    pub(crate) query: ArenaRegion<f32>,
    pub(crate) attention: ArenaRegion<f32>,
    pub(crate) attention_gated: ArenaRegion<u16>,

    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_intermediate: ArenaRegion<u16>,
    pub(crate) routed_output: ArenaRegion<u16>,
    pub(crate) shared_intermediate: ArenaRegion<u16>,
    pub(crate) shared_output: ArenaRegion<u16>,
    pub(crate) shared_gate_logit: ArenaRegion<u16>,

    /// The 2,560-wide sublayer output both write-backs inject, written twice per layer.
    pub(crate) block_output: ArenaRegion<u16>,

    // --- runtime inputs, staged per round rather than resident ---
    pub(crate) state_rows: ArenaRegion<u32>,
    pub(crate) table_rows: ArenaRegion<u32>,
    pub(crate) cache_positions: ArenaRegion<u32>,
    pub(crate) lengths: ArenaRegion<u32>,
    pub(crate) rope_cos: ArenaRegion<f32>,
    pub(crate) rope_sin: ArenaRegion<f32>,

    // --- engram staging, reserved once for the one layer that runs the module ---
    pub(crate) ple_codes: ArenaRegion<u8>,
    pub(crate) ple_injected: ArenaRegion<u16>,
    pub(crate) ple_embedding: ArenaRegion<u16>,
    pub(crate) ple_key: ArenaRegion<u16>,
    pub(crate) ple_key_normed: ArenaRegion<u16>,
    pub(crate) ple_query_normed: ArenaRegion<u16>,
    pub(crate) ple_value: ArenaRegion<u16>,
    pub(crate) ple_gated: ArenaRegion<u16>,
    pub(crate) ple_gated_normed: ArenaRegion<u16>,
    pub(crate) ple_delta: ArenaRegion<u16>,
}

/// The endpoint's weights and the planes the collapsing mixer and the head publish.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextResidentEndpoint {
    // --- weights ---
    /// The model-level collapsing mixer, this target's only final normalization.
    pub(crate) mixer_norm: ArenaRegion<u16>,
    pub(crate) mixer_down: ArenaRegion<u16>,
    pub(crate) mixer_up: ArenaRegion<u16>,
    pub(crate) lm_head: ArenaRegion<u16>,

    // --- staging ---
    /// Host-gathered token embeddings, uploaded `T * HIDDEN` once and widened on device.
    pub(crate) embedding_rows: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) mixer_low_rank: ArenaRegion<u16>,
    pub(crate) mixer_mixed: ArenaRegion<u16>,
    /// `[MAX_BATCH, VOCAB]` BF16 logits, the one plane a decode step reads back.
    pub(crate) logits: ArenaRegion<u16>,
}

/// One QSA layer's three paged cache planes, all off the same block table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextKvPlanes {
    pub(crate) key_pages: ArenaRegion<u8>,
    pub(crate) value_pages: ArenaRegion<u8>,
    /// Raw 128-wide indexer keys sharing this layer's page mapping.
    pub(crate) indexer_pages: ArenaRegion<u8>,
}

/// The shared paged cache: one block table, twelve layers of planes.
#[derive(Clone, Debug)]
pub(crate) struct Qwen38FlashNextResidentKv {
    pub(crate) block_tables: ArenaRegion<u32>,
    pub(crate) layers: Vec<Qwen38FlashNextKvPlanes>,
}

/// Checked four-arena plan of the whole Qwen3.8 Flash-Next resident program.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextResidentLayout {
    resident: ArenaLayout,
    kv: ArenaLayout,
    streaming: StreamingWeightLayout,
    engram: Qwen38FlashNextEngramStagerLayout,
    layers: Vec<Qwen38FlashNextResidentLayerRegions>,
    workspace: Qwen38FlashNextResidentWorkspace,
    endpoint: Qwen38FlashNextResidentEndpoint,
    kv_regions: Qwen38FlashNextResidentKv,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
    persistent_bytes: usize,
    physical_pages: usize,
}

impl Qwen38FlashNextResidentLayout {
    /// Plans the whole model at the adopted 25% residency and the solved KV capacity.
    pub fn build() -> EngineResult<Self> {
        Self::plan(
            QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS,
            QWEN38_FLASH_NEXT_PRIMARY_SOURCE,
            None,
        )
    }

    /// Plans the model at an explicit slot budget, posture, and page count.
    ///
    /// The accounting tests use this to pin both host postures and both cache fractions from
    /// one code path, so neither is a number this module could transcribe differently.
    pub fn plan(
        slot_count: usize,
        primary_source: StreamingPrimarySource,
        physical_pages: Option<usize>,
    ) -> EngineResult<Self> {
        require_geometry()?;

        let streaming = match primary_source {
            StreamingPrimarySource::Pinned => StreamingWeightLayout::build(
                QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT,
                QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES,
                Some(QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES),
                slot_count,
            )?,
            StreamingPrimarySource::Mapped => StreamingWeightLayout::build_mapped_primary(
                QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT,
                QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES,
                Some(QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES),
                slot_count,
                QWEN38_FLASH_NEXT_BOUNCE_RING_SLOTS,
            )?,
        };
        let engram = Qwen38FlashNextEngramStagerLayout::build()?;

        let mut resident = ArenaLayout::new();
        let mut layers = Vec::with_capacity(A::LAYERS);
        for layer in 0..A::LAYERS {
            layers.push(reserve_layer(&mut resident, layer)?);
        }
        let workspace = reserve_workspace(&mut resident)?;
        let endpoint = reserve_endpoint(&mut resident)?;

        let resident_weight_bytes = layer_weight_bytes(&layers)?
            .checked_add(endpoint_weight_bytes(endpoint)?)
            .ok_or_else(|| {
                EngineError::layout("Qwen3.8 Flash-Next resident weight total overflows")
            })?;
        let persistent_bytes = layers.iter().try_fold(0usize, |total, layer| {
            checked_sum(
                "Qwen3.8 Flash-Next persistent state",
                total,
                layer.persistent.byte_len()?,
            )
        })?;
        let workspace_bytes = workspace_byte_len(workspace, endpoint)?;

        // The KV pool is whatever the remainder funds, exactly as the house solver does it:
        // every spare byte becomes a page rather than idle headroom.
        let fixed_non_kv = sum(
            "Qwen3.8 Flash-Next fixed resident bytes",
            &[
                resident.byte_len(),
                streaming.device_resident_bytes(),
                engram.device_resident_bytes(),
            ],
        )?;
        let physical_pages = match physical_pages {
            Some(pages) => pages,
            None => solve_physical_pages(fixed_non_kv)?,
        };

        let mut kv = ArenaLayout::new();
        let kv_regions = reserve_kv(&mut kv, physical_pages)?;

        Ok(Self {
            resident,
            kv,
            streaming,
            engram,
            layers,
            workspace,
            endpoint,
            kv_regions,
            resident_weight_bytes,
            workspace_bytes,
            persistent_bytes,
            physical_pages,
        })
    }

    /// Weight, carry, workspace, and endpoint bytes, including alignment padding.
    pub const fn resident_arena_bytes(&self) -> usize {
        self.resident.byte_len()
    }

    /// Paged cache bytes, including the block table and alignment padding.
    pub const fn kv_arena_bytes(&self) -> usize {
        self.kv.byte_len()
    }

    /// The expert slot cache's own plan.
    pub const fn streaming(&self) -> &StreamingWeightLayout {
        &self.streaming
    }

    /// The engram tier's own plan.
    pub const fn engram(&self) -> &Qwen38FlashNextEngramStagerLayout {
        &self.engram
    }

    /// Physical pages the capacity solver funded.
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    /// Pages one decode slot owns at `MAX_BATCH`.
    pub const fn table_stride(&self) -> usize {
        self.physical_pages / MAX_BATCH
    }

    /// Context depth one decode slot reaches at `MAX_BATCH`.
    pub const fn context_tokens_per_slot(&self) -> usize {
        self.table_stride() * QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE
    }

    /// Device bytes every arena of this target holds at once.
    pub fn total_device_bytes(&self) -> EngineResult<usize> {
        sum(
            "Qwen3.8 Flash-Next total device bytes",
            &[
                self.resident.byte_len(),
                self.kv.byte_len(),
                self.streaming.device_resident_bytes(),
                self.engram.device_resident_bytes(),
            ],
        )
    }

    /// Slot-owned recurrent and convolution carries across the whole stack.
    pub const fn persistent_state_bytes(&self) -> usize {
        self.persistent_bytes
    }

    pub(crate) const fn layers(&self) -> &Vec<Qwen38FlashNextResidentLayerRegions> {
        &self.layers
    }

    pub(crate) const fn workspace(&self) -> Qwen38FlashNextResidentWorkspace {
        self.workspace
    }

    pub(crate) const fn endpoint(&self) -> Qwen38FlashNextResidentEndpoint {
        self.endpoint
    }

    pub(crate) const fn kv_regions(&self) -> &Qwen38FlashNextResidentKv {
        &self.kv_regions
    }

    pub(crate) const fn resident_builder(&self) -> &ArenaLayout {
        &self.resident
    }

    pub(crate) const fn kv_builder(&self) -> &ArenaLayout {
        &self.kv
    }

    /// Source-backed bytes uploaded for one layer.
    pub(crate) fn layer_weight_bytes(&self, layer: usize) -> EngineResult<usize> {
        let regions = self.layers.get(layer..=layer).ok_or_else(|| {
            EngineError::layout(format!(
                "Flash-Next layer {layer} is outside the planned 0..{}",
                self.layers.len()
            ))
        })?;

        layer_weight_bytes(regions)
    }

    /// Source-backed weight bytes the endpoint's own upload writes.
    pub(crate) fn endpoint_weight_bytes(&self) -> EngineResult<usize> {
        endpoint_weight_bytes(self.endpoint)
    }
}

impl LayerMemoryLayout for Qwen38FlashNextResidentLayout {
    fn arena_bytes(&self) -> usize {
        self.resident.byte_len() + self.kv.byte_len()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    fn cache_bytes(&self) -> usize {
        self.kv.byte_len()
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

impl StreamingResidencyAccounting for Qwen38FlashNextResidentLayout {
    fn device_resident_bytes(&self) -> usize {
        self.total_device_bytes().unwrap_or(usize::MAX)
    }

    fn host_pinned_bytes(&self) -> usize {
        // The streaming pool's pinned class plus the two stagers and the logit bank this
        // owner page-locks itself. Never summed with the mapped class.
        self.streaming.host_pinned_bytes()
            + self.engram.host_pinned_bytes()
            + self.embedding_stager_bytes()
            + self.logit_bank_bytes()
    }

    fn host_mapped_bytes(&self) -> usize {
        // The borrowed primary extents, the whole FP8 engram table, and the embedding matrix
        // the token stager gathers rows from. Three owners, counted as the file's own bytes.
        self.streaming.host_mapped_bytes() + self.engram.host_mapped_bytes() + embedding_bytes()
    }
}

impl Qwen38FlashNextResidentLayout {
    /// Page-locked bytes the token-embedding stager holds, one widest round.
    pub const fn embedding_stager_bytes(&self) -> usize {
        QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS * <A as Arch>::HIDDEN * size_of::<u16>()
    }

    /// Page-locked bytes the logit readback bank holds, two generations of one widest batch.
    ///
    /// Two, not one: a decode step reads the previous batch's logits back while the next
    /// step's replay is already publishing into the device plane, and a single pinned
    /// destination would be the same reuse hazard rule 5 fences everywhere else.
    pub const fn logit_bank_bytes(&self) -> usize {
        2 * MAX_BATCH * <A as Arch>::VOCAB * size_of::<u16>()
    }
}

/// Bytes the borrowed BF16 embedding matrix occupies in the checkpoint mapping.
const fn embedding_bytes() -> usize {
    <A as Arch>::VOCAB * <A as Arch>::HIDDEN * size_of::<u16>()
}

/// Largest page count the remaining device budget funds, capped at full context.
fn solve_physical_pages(fixed_non_kv: usize) -> EngineResult<usize> {
    let block_table_bytes = product(
        "Qwen3.8 Flash-Next block table",
        product("Qwen3.8 Flash-Next block table rows", MAX_BATCH, {
            QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES
        })?,
        size_of::<u32>(),
    )?;
    let spendable = QWEN38_FLASH_NEXT_DEVICE_BUDGET_BYTES
        .checked_sub(QWEN38_FLASH_NEXT_REQUIRED_HEADROOM_BYTES)
        .and_then(|budget| budget.checked_sub(fixed_non_kv))
        .and_then(|budget| budget.checked_sub(block_table_bytes))
        .ok_or_else(|| {
            EngineError::layout(format!(
                "Qwen3.8 Flash-Next fixed resident bytes {fixed_non_kv} leave no room for a KV pool \
                 inside a {QWEN38_FLASH_NEXT_DEVICE_BUDGET_BYTES} B budget with \
                 {QWEN38_FLASH_NEXT_REQUIRED_HEADROOM_BYTES} B of headroom"
            ))
        })?;
    let pages = (spendable / cache_bytes_per_physical_page()?)
        .min(QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES);
    if pages < MAX_BATCH {
        return Err(EngineError::layout(format!(
            "Qwen3.8 Flash-Next KV capacity solves to {pages} pages, fewer than the {MAX_BATCH} slots"
        )));
    }

    Ok(pages)
}

/// K, V, and indexer bytes one physical page holds across all twelve QSA layers.
fn cache_bytes_per_physical_page() -> EngineResult<usize> {
    let kv = product(
        "Qwen3.8 Flash-Next KV page",
        product(
            "Qwen3.8 Flash-Next KV page heads",
            <A as Arch>::NUM_KV_HEADS,
            QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
        )?,
        <A as Arch>::HEAD_DIM,
    )?;
    let indexer = product(
        "Qwen3.8 Flash-Next indexer page",
        QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
        product(
            "Qwen3.8 Flash-Next indexer key bytes",
            A::INDEXER_HEAD_DIM,
            2,
        )?,
    )?;

    product(
        "Qwen3.8 Flash-Next cache page",
        checked_sum("Qwen3.8 Flash-Next cache page planes", 2 * kv, indexer)?,
        QWEN38_FLASH_NEXT_ATTENTION_LAYERS,
    )
}

fn reserve_layer(
    builder: &mut ArenaLayout,
    layer: usize,
) -> EngineResult<Qwen38FlashNextResidentLayerRegions> {
    let hc_projection = product(
        "Qwen3.8 Flash-Next hyper-connection projection",
        A::HC_LOWRANK,
        A::HC_WIDTH,
    )?;
    let hc_inject = product(
        "Qwen3.8 Flash-Next hyper-connection inject",
        A::HC_COUNT,
        A::HC_WIDTH,
    )?;
    let hyper = |builder: &mut ArenaLayout| -> EngineResult<HyperConnectionRegions> {
        Ok(HyperConnectionRegions {
            norm: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
            down: builder.reserve(hc_projection, ALIGNMENT)?,
            up: builder.reserve(hc_projection, ALIGNMENT)?,
            inject: builder.reserve(hc_inject, ALIGNMENT)?,
        })
    };
    let attention_hc = hyper(builder)?;
    let mlp_hc = hyper(builder)?;

    let block = if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
        Qwen38FlashNextBlockWeightRegions::Qsa(reserve_qsa_weights(builder)?)
    } else {
        Qwen38FlashNextBlockWeightRegions::Gdn(reserve_gdn_weights(builder)?)
    };

    let moe = reserve_moe_weights(builder)?;
    let ple = (layer == A::PLE_LAYER)
        .then(|| reserve_ple_weights(builder))
        .transpose()?;
    let persistent = Qwen38FlashNextPersistentState::reserve(builder, layer)?;

    Ok(Qwen38FlashNextResidentLayerRegions {
        attention_hc,
        mlp_hc,
        moe,
        block,
        ple,
        persistent,
    })
}

fn reserve_gdn_weights(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextGdnWeightRegions> {
    let input_weight = product(
        "Qwen3.8 Flash-Next GDN input weight",
        A::GDN_INPUT_ROWS,
        <A as Arch>::HIDDEN,
    )?;
    let control_weight = product(
        "Qwen3.8 Flash-Next GDN control weight",
        product(
            "Qwen3.8 Flash-Next GDN control rows",
            2,
            A::GDN_CONTROL_ROWS,
        )?,
        <A as Arch>::HIDDEN,
    )?;
    let convolution_weight = product(
        "Qwen3.8 Flash-Next GDN convolution weight",
        A::GDN_QKV_ROWS,
        A::LINEAR_CONV_KERNEL_DIM,
    )?;
    let output_weight = product(
        "Qwen3.8 Flash-Next GDN output weight",
        <A as Arch>::HIDDEN,
        A::GDN_VALUE_ROWS,
    )?;

    Ok(Qwen38FlashNextGdnWeightRegions {
        input_weight: builder.reserve(input_weight, ALIGNMENT)?,
        control_weight: builder.reserve(control_weight, ALIGNMENT)?,
        convolution_weight: builder.reserve(convolution_weight, ALIGNMENT)?,
        a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
        dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
        norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
        output_weight: builder.reserve(output_weight, ALIGNMENT)?,
    })
}

fn reserve_qsa_weights(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextQsaWeightRegions> {
    let qkv_weight = product(
        "Qwen3.8 Flash-Next QSA fused projection",
        A::ATTENTION_QKV_ROWS,
        <A as Arch>::HIDDEN,
    )?;
    let output_weight = product(
        "Qwen3.8 Flash-Next QSA output projection",
        <A as Arch>::HIDDEN,
        A::ATTENTION_OUTPUT_COLUMNS,
    )?;
    let indexer_qk_weight = product(
        "Qwen3.8 Flash-Next indexer projection",
        A::INDEXER_ROWS,
        <A as Arch>::HIDDEN,
    )?;

    Ok(Qwen38FlashNextQsaWeightRegions {
        qkv_weight: builder.reserve(qkv_weight, ALIGNMENT)?,
        output_weight: builder.reserve(output_weight, ALIGNMENT)?,
        query_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
        key_norm: builder.reserve(<A as Arch>::HEAD_DIM, ALIGNMENT)?,
        indexer_qk_weight: builder.reserve(indexer_qk_weight, ALIGNMENT)?,
        indexer_query_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,
        indexer_key_norm: builder.reserve(A::INDEXER_HEAD_DIM, ALIGNMENT)?,
    })
}

fn reserve_moe_weights(builder: &mut ArenaLayout) -> EngineResult<MoeRegions> {
    let router_weight = product(
        "Qwen3.8 Flash-Next router weight",
        A::NUM_EXPERTS,
        <A as Arch>::HIDDEN,
    )?;
    let expert_weight_scales_2 = product(
        "Qwen3.8 Flash-Next routed weight scales",
        A::NUM_EXPERTS,
        EXPERT_WEIGHT_SCALES,
    )?;
    let shared_gate_up = product(
        "Qwen3.8 Flash-Next shared expert projection",
        A::SHARED_EXPERT_INTERMEDIATE,
        <A as Arch>::HIDDEN,
    )?;

    Ok(MoeRegions {
        router_weight: builder.reserve(router_weight, ALIGNMENT)?,
        shared_gate_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_up_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_down_weight: builder.reserve(shared_gate_up, ALIGNMENT)?,
        shared_gate_logit_weight: builder.reserve(<A as Arch>::HIDDEN, ALIGNMENT)?,
        expert_weight_scales_2: builder.reserve(expert_weight_scales_2, ALIGNMENT)?,
    })
}

fn reserve_ple_weights(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextPleWeightRegions> {
    let key_proj = product(
        "Qwen3.8 Flash-Next PLE key projection",
        A::HC_WIDTH,
        A::PLE_EMBED_DIM,
    )?;
    let value_proj = product(
        "Qwen3.8 Flash-Next PLE value projection",
        A::PLE_EMBED_DIM,
        A::PLE_EMBED_DIM,
    )?;
    let convolution = product(
        "Qwen3.8 Flash-Next PLE convolution",
        A::HC_WIDTH,
        A::PLE_CONV_KERNEL,
    )?;

    Ok(Qwen38FlashNextPleWeightRegions {
        key_proj: builder.reserve(key_proj, ALIGNMENT)?,
        value_proj: builder.reserve(value_proj, ALIGNMENT)?,
        norm_key: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        norm_query: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        norm_conv: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        convolution: builder.reserve(convolution, ALIGNMENT)?,
    })
}

fn reserve_workspace(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextResidentWorkspace> {
    let rows = QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS;
    let row_stream = product("Qwen3.8 Flash-Next stream rows", rows, A::HC_WIDTH)?;
    let row_hidden = product("Qwen3.8 Flash-Next hidden rows", rows, <A as Arch>::HIDDEN)?;
    let row_low_rank = product("Qwen3.8 Flash-Next low-rank rows", rows, A::HC_LOWRANK)?;
    let row_write_gate = product("Qwen3.8 Flash-Next write gates", rows, A::HC_COUNT)?;
    let row_projected = product(
        "Qwen3.8 Flash-Next GDN projected rows",
        rows,
        A::GDN_INPUT_ROWS,
    )?;
    let row_gdn_qkv = product(
        "Qwen3.8 Flash-Next GDN convolved rows",
        rows,
        A::GDN_QKV_ROWS,
    )?;
    let row_control = product(
        "Qwen3.8 Flash-Next GDN control values",
        rows,
        A::GDN_CONTROL_ROWS,
    )?;
    let row_value = product("Qwen3.8 Flash-Next GDN value rows", rows, A::GDN_VALUE_ROWS)?;
    let row_qkv = product(
        "Qwen3.8 Flash-Next QSA projection rows",
        rows,
        A::ATTENTION_QKV_ROWS,
    )?;
    let row_attention = product(
        "Qwen3.8 Flash-Next QSA attention rows",
        rows,
        A::ATTENTION_OUTPUT_COLUMNS,
    )?;
    let row_rotary = product("Qwen3.8 Flash-Next rotary rows", rows, ROTARY_ELEMENTS)?;
    let row_router_logits = product("Qwen3.8 Flash-Next router logits", rows, A::NUM_EXPERTS)?;
    let row_routed = product("Qwen3.8 Flash-Next routed ranks", rows, ROUTED_SLOTS)?;
    let row_routed_intermediate = product(
        "Qwen3.8 Flash-Next routed intermediate",
        row_routed,
        <A as Arch>::INTERMEDIATE,
    )?;
    let row_routed_output = product(
        "Qwen3.8 Flash-Next routed output",
        row_routed,
        <A as Arch>::HIDDEN,
    )?;
    let row_shared_intermediate = product(
        "Qwen3.8 Flash-Next shared intermediate",
        rows,
        A::SHARED_EXPERT_INTERMEDIATE,
    )?;
    let row_ple_embed = product("Qwen3.8 Flash-Next PLE embed rows", rows, A::PLE_EMBED_DIM)?;
    let ple_codes = product(
        "Qwen3.8 Flash-Next PLE staged codes",
        rows,
        product(
            "Qwen3.8 Flash-Next PLE token bytes",
            A::NGRAM_HEADS,
            A::NGRAM_HEAD_DIM,
        )?,
    )?;

    Ok(Qwen38FlashNextResidentWorkspace {
        residual_a: builder.reserve(row_stream, ALIGNMENT)?,
        residual_b: builder.reserve(row_stream, ALIGNMENT)?,

        hc_normalized: builder.reserve(row_stream, ALIGNMENT)?,
        hc_low_rank: builder.reserve(row_low_rank, ALIGNMENT)?,
        hc_mixed: builder.reserve(row_hidden, ALIGNMENT)?,
        hc_write_gate: builder.reserve(row_write_gate, ALIGNMENT)?,

        gdn_projected: builder.reserve(row_projected, ALIGNMENT)?,
        gdn_convolved: builder.reserve(row_gdn_qkv, ALIGNMENT)?,
        gdn_log_decay: builder.reserve(row_control, ALIGNMENT)?,
        gdn_beta: builder.reserve(row_control, ALIGNMENT)?,
        gdn_recurrent_plane: builder.reserve(row_value, ALIGNMENT)?,
        gdn_recurrent_output: builder.reserve(row_value, ALIGNMENT)?,

        qkv: builder.reserve(row_qkv, ALIGNMENT)?,
        query: builder.reserve(row_attention, ALIGNMENT)?,
        attention: builder.reserve(row_attention, ALIGNMENT)?,
        attention_gated: builder.reserve(row_attention, ALIGNMENT)?,

        router_logits: builder.reserve(row_router_logits, ALIGNMENT)?,
        expert_indices: builder.reserve(row_routed, ALIGNMENT)?,
        routing_weights: builder.reserve(row_routed, ALIGNMENT)?,
        routed_intermediate: builder.reserve(row_routed_intermediate, ALIGNMENT)?,
        routed_output: builder.reserve(row_routed_output, ALIGNMENT)?,
        shared_intermediate: builder.reserve(row_shared_intermediate, ALIGNMENT)?,
        shared_output: builder.reserve(row_hidden, ALIGNMENT)?,
        shared_gate_logit: builder.reserve(rows, ALIGNMENT)?,

        block_output: builder.reserve(row_hidden, ALIGNMENT)?,

        state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
        table_rows: builder.reserve(rows, ALIGNMENT)?,
        cache_positions: builder.reserve(rows, ALIGNMENT)?,
        lengths: builder.reserve(rows, ALIGNMENT)?,
        rope_cos: builder.reserve(row_rotary, ALIGNMENT)?,
        rope_sin: builder.reserve(row_rotary, ALIGNMENT)?,

        ple_codes: builder.reserve(ple_codes, ALIGNMENT)?,
        ple_injected: builder.reserve(row_stream, ALIGNMENT)?,
        ple_embedding: builder.reserve(row_ple_embed, ALIGNMENT)?,
        ple_key: builder.reserve(row_stream, ALIGNMENT)?,
        ple_key_normed: builder.reserve(row_stream, ALIGNMENT)?,
        ple_query_normed: builder.reserve(row_stream, ALIGNMENT)?,
        ple_value: builder.reserve(row_ple_embed, ALIGNMENT)?,
        ple_gated: builder.reserve(row_stream, ALIGNMENT)?,
        ple_gated_normed: builder.reserve(row_stream, ALIGNMENT)?,
        ple_delta: builder.reserve(row_stream, ALIGNMENT)?,
    })
}

fn reserve_endpoint(builder: &mut ArenaLayout) -> EngineResult<Qwen38FlashNextResidentEndpoint> {
    let rows = QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS;
    let hc_projection = product(
        "Qwen3.8 Flash-Next mixer projection",
        A::HC_LOWRANK,
        A::HC_WIDTH,
    )?;
    let lm_head = product(
        "Qwen3.8 Flash-Next LM head",
        <A as Arch>::VOCAB,
        <A as Arch>::HIDDEN,
    )?;
    let logits = product(
        "Qwen3.8 Flash-Next logit plane",
        MAX_BATCH,
        <A as Arch>::VOCAB,
    )?;

    Ok(Qwen38FlashNextResidentEndpoint {
        mixer_norm: builder.reserve(A::HC_WIDTH, ALIGNMENT)?,
        mixer_down: builder.reserve(hc_projection, ALIGNMENT)?,
        mixer_up: builder.reserve(hc_projection, ALIGNMENT)?,
        lm_head: builder.reserve(lm_head, ALIGNMENT)?,

        embedding_rows: builder.reserve(
            product(
                "Qwen3.8 Flash-Next embedding rows",
                rows,
                <A as Arch>::HIDDEN,
            )?,
            ALIGNMENT,
        )?,
        mixer_normalized: builder.reserve(
            product("Qwen3.8 Flash-Next mixer normalized", rows, A::HC_WIDTH)?,
            ALIGNMENT,
        )?,
        mixer_low_rank: builder.reserve(
            product("Qwen3.8 Flash-Next mixer low rank", rows, A::HC_LOWRANK)?,
            ALIGNMENT,
        )?,
        mixer_mixed: builder.reserve(
            product("Qwen3.8 Flash-Next mixer mixed", rows, <A as Arch>::HIDDEN)?,
            ALIGNMENT,
        )?,
        logits: builder.reserve(logits, ALIGNMENT)?,
    })
}

fn reserve_kv(builder: &mut ArenaLayout, pages: usize) -> EngineResult<Qwen38FlashNextResidentKv> {
    // Full-context rows keep table stride independent of the funded page count.
    let block_tables = product(
        "Qwen3.8 Flash-Next block table",
        MAX_BATCH,
        QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES,
    )?;
    let cache_plane = product(
        "Qwen3.8 Flash-Next KV cache plane",
        product(
            "Qwen3.8 Flash-Next KV cache heads",
            pages,
            <A as Arch>::NUM_KV_HEADS,
        )?,
        product(
            "Qwen3.8 Flash-Next KV cache values",
            QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
            <A as Arch>::HEAD_DIM,
        )?,
    )?;
    let indexer_plane = product(
        "Qwen3.8 Flash-Next indexer cache plane",
        product(
            "Qwen3.8 Flash-Next indexer cache tokens",
            pages,
            QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE,
        )?,
        product(
            "Qwen3.8 Flash-Next indexer key bytes",
            A::INDEXER_HEAD_DIM,
            2,
        )?,
    )?;

    let block_tables = builder.reserve(block_tables, ALIGNMENT)?;
    let mut layers = Vec::with_capacity(QWEN38_FLASH_NEXT_ATTENTION_LAYERS);
    for _ in 0..QWEN38_FLASH_NEXT_ATTENTION_LAYERS {
        layers.push(Qwen38FlashNextKvPlanes {
            key_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            value_pages: builder.reserve(cache_plane, ALIGNMENT)?,
            indexer_pages: builder.reserve(indexer_plane, ALIGNMENT)?,
        });
    }

    Ok(Qwen38FlashNextResidentKv {
        block_tables,
        layers,
    })
}

fn layer_weight_bytes(layers: &[Qwen38FlashNextResidentLayerRegions]) -> EngineResult<usize> {
    layers.iter().try_fold(0usize, |total, layer| {
        let hyper = |hc: HyperConnectionRegions| {
            hc.norm.byte_len() + hc.down.byte_len() + hc.up.byte_len() + hc.inject.byte_len()
        };
        let block = match layer.block {
            Qwen38FlashNextBlockWeightRegions::Gdn(gdn) => {
                gdn.input_weight.byte_len()
                    + gdn.control_weight.byte_len()
                    + gdn.convolution_weight.byte_len()
                    + gdn.a_log.byte_len()
                    + gdn.dt_bias.byte_len()
                    + gdn.norm.byte_len()
                    + gdn.output_weight.byte_len()
            }
            Qwen38FlashNextBlockWeightRegions::Qsa(qsa) => {
                qsa.qkv_weight.byte_len()
                    + qsa.output_weight.byte_len()
                    + qsa.query_norm.byte_len()
                    + qsa.key_norm.byte_len()
                    + qsa.indexer_qk_weight.byte_len()
                    + qsa.indexer_query_norm.byte_len()
                    + qsa.indexer_key_norm.byte_len()
            }
        };
        let moe = layer.moe.router_weight.byte_len()
            + layer.moe.shared_gate_weight.byte_len()
            + layer.moe.shared_up_weight.byte_len()
            + layer.moe.shared_down_weight.byte_len()
            + layer.moe.shared_gate_logit_weight.byte_len()
            + layer.moe.expert_weight_scales_2.byte_len();
        let ple = layer.ple.map_or(0, |ple| {
            ple.key_proj.byte_len()
                + ple.value_proj.byte_len()
                + ple.norm_key.byte_len()
                + ple.norm_query.byte_len()
                + ple.norm_conv.byte_len()
                + ple.convolution.byte_len()
        });

        sum(
            "Qwen3.8 Flash-Next layer weights",
            &[
                total,
                hyper(layer.attention_hc),
                hyper(layer.mlp_hc),
                block,
                moe,
                ple,
            ],
        )
    })
}

fn endpoint_weight_bytes(endpoint: Qwen38FlashNextResidentEndpoint) -> EngineResult<usize> {
    sum(
        "Qwen3.8 Flash-Next endpoint weights",
        &[
            endpoint.mixer_norm.byte_len(),
            endpoint.mixer_down.byte_len(),
            endpoint.mixer_up.byte_len(),
            endpoint.lm_head.byte_len(),
        ],
    )
}

fn workspace_byte_len(
    workspace: Qwen38FlashNextResidentWorkspace,
    endpoint: Qwen38FlashNextResidentEndpoint,
) -> EngineResult<usize> {
    sum(
        "Qwen3.8 Flash-Next shared workspace",
        &[
            workspace.residual_a.byte_len(),
            workspace.residual_b.byte_len(),
            workspace.hc_normalized.byte_len(),
            workspace.hc_low_rank.byte_len(),
            workspace.hc_mixed.byte_len(),
            workspace.hc_write_gate.byte_len(),
            workspace.gdn_projected.byte_len(),
            workspace.gdn_convolved.byte_len(),
            workspace.gdn_log_decay.byte_len(),
            workspace.gdn_beta.byte_len(),
            workspace.gdn_recurrent_plane.byte_len(),
            workspace.gdn_recurrent_output.byte_len(),
            workspace.qkv.byte_len(),
            workspace.query.byte_len(),
            workspace.attention.byte_len(),
            workspace.attention_gated.byte_len(),
            workspace.router_logits.byte_len(),
            workspace.expert_indices.byte_len(),
            workspace.routing_weights.byte_len(),
            workspace.routed_intermediate.byte_len(),
            workspace.routed_output.byte_len(),
            workspace.shared_intermediate.byte_len(),
            workspace.shared_output.byte_len(),
            workspace.shared_gate_logit.byte_len(),
            workspace.block_output.byte_len(),
            workspace.state_rows.byte_len(),
            workspace.table_rows.byte_len(),
            workspace.cache_positions.byte_len(),
            workspace.lengths.byte_len(),
            workspace.rope_cos.byte_len(),
            workspace.rope_sin.byte_len(),
            workspace.ple_codes.byte_len(),
            workspace.ple_injected.byte_len(),
            workspace.ple_embedding.byte_len(),
            workspace.ple_key.byte_len(),
            workspace.ple_key_normed.byte_len(),
            workspace.ple_query_normed.byte_len(),
            workspace.ple_value.byte_len(),
            workspace.ple_gated.byte_len(),
            workspace.ple_gated_normed.byte_len(),
            workspace.ple_delta.byte_len(),
            endpoint.embedding_rows.byte_len(),
            endpoint.mixer_normalized.byte_len(),
            endpoint.mixer_low_rank.byte_len(),
            endpoint.mixer_mixed.byte_len(),
            endpoint.logits.byte_len(),
        ],
    )
}

/// Refuses a geometry the resident program's arithmetic does not describe.
fn require_geometry() -> EngineResult<()> {
    if <A as Arch>::LAYERS != 48
        || <A as Arch>::HIDDEN != 2_560
        || <A as Arch>::VOCAB != 248_320
        || A::HC_WIDTH != 10_240
        || A::NUM_EXPERTS != 512
        || A::NUM_EXPERTS_PER_TOKEN != 10
        || A::PLE_LAYER != 1
    {
        return Err(EngineError::layout(
            "Qwen3.8 Flash-Next resident layout requires the pinned geometry",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resident weights exclude the corrected scalar plane and host/launch constants.
    const RESIDENT_WEIGHT_BYTES: usize = 8_624_293_632;

    #[test]
    fn the_backbone_weight_total_is_the_layout_plans_prediction_minus_what_is_not_a_plane() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();

        assert_eq!(plan.resident_weight_bytes(), RESIDENT_WEIGHT_BYTES);
        assert_eq!(RESIDENT_WEIGHT_BYTES, 8_624_588_826 - 294_912 - 280 - 2);
    }

    #[test]
    fn the_per_layer_weight_totals_reproduce_the_composed_layer_figures() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();
        let mut gdn = Vec::new();
        let mut qsa = Vec::new();
        for (layer, regions) in plan.layers().iter().enumerate() {
            let bytes = layer_weight_bytes(std::slice::from_ref(regions)).unwrap();
            match regions.block {
                Qwen38FlashNextBlockWeightRegions::Gdn(_) if layer != A::PLE_LAYER => {
                    gdn.push(bytes)
                }
                Qwen38FlashNextBlockWeightRegions::Gdn(_) => {}
                Qwen38FlashNextBlockWeightRegions::Qsa(_) => qsa.push(bytes),
            }
        }

        assert_eq!(gdn.len(), 35);
        assert_eq!(qsa.len(), QWEN38_FLASH_NEXT_ATTENTION_LAYERS);
        assert!(gdn.iter().all(|&bytes| bytes == 154_799_552));
        assert!(qsa.iter().all(|&bytes| bytes == 141_775_360));
    }

    #[test]
    fn the_whole_stack_persistent_state_is_the_plans_prediction_exactly() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();

        assert_eq!(plan.persistent_state_bytes(), 925_138_944);
    }

    #[test]
    fn exactly_one_layer_reserves_the_engram_modules_weights() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();
        let with_ple = plan
            .layers()
            .iter()
            .enumerate()
            .filter(|(_, regions)| regions.ple.is_some())
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();

        assert_eq!(with_ple, vec![A::PLE_LAYER]);
    }

    #[test]
    fn the_layer_kinds_partition_the_stack_thirty_six_to_twelve() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();
        let qsa = plan
            .layers()
            .iter()
            .filter(|regions| matches!(regions.block, Qwen38FlashNextBlockWeightRegions::Qsa(_)))
            .count();

        assert_eq!(qsa, QWEN38_FLASH_NEXT_ATTENTION_LAYERS);
        assert_eq!(plan.layers().len() - qsa, 36);
    }

    #[test]
    fn both_host_postures_reproduce_the_layout_plans_predictions_exactly() {
        let mapped =
            Qwen38FlashNextResidentLayout::plan(6_144, StreamingPrimarySource::Mapped, None)
                .unwrap();
        let pinned =
            Qwen38FlashNextResidentLayout::plan(6_144, StreamingPrimarySource::Pinned, None)
                .unwrap();

        // Mapped posture: scales, table/bounce rings, and staging buffers are pinned.
        assert_eq!(mapped.host_pinned_bytes(), 7_585_611_776);
        assert_eq!(mapped.host_mapped_bytes(), 112_869_621_760);

        // Fully pinned posture on a host with sufficient RAM.
        assert_eq!(pinned.host_pinned_bytes(), 67_963_928_576);
        assert_eq!(pinned.host_mapped_bytes(), 52_471_644_160);

        // Nothing device-side moves between postures: the mapped posture's two uploads write
        // the same contiguous extent at the same stable address the pinned posture's one would.
        assert_eq!(
            mapped.total_device_bytes().unwrap(),
            pinned.total_device_bytes().unwrap()
        );

        // The two host columns conserve the inventory: what leaves the pinned class is exactly
        // what the mapping then holds.
        assert_eq!(
            mapped.streaming().host_pool_bytes() + mapped.streaming().host_mapped_bytes(),
            pinned.streaming().host_pool_bytes()
        );
    }

    #[test]
    fn the_adopted_posture_is_twenty_five_percent_mapped_primary() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();

        assert_eq!(plan.streaming().slot_count(), 6_144);
        assert!((plan.streaming().resident_fraction() - 0.25).abs() < f64::EPSILON);
        assert_eq!(
            plan.streaming().primary_source(),
            StreamingPrimarySource::Mapped
        );
        assert_eq!(plan.streaming().stride_bytes(), 2_764_800);
        assert_eq!(plan.streaming().extent_padding_bytes(), 0);
        assert_eq!(plan.streaming().device_resident_bytes(), 16_987_029_504);
        assert_eq!(
            plan.streaming().item_count(),
            QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT
        );
    }

    #[test]
    fn the_geometry_constants_a_test_pins_independently_are_this_targets_own() {
        assert_eq!(QWEN38_FLASH_NEXT_ATTENTION_LAYERS, 12);
        assert_eq!(QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(
            <A as Arch>::NUM_KV_HEADS
                * QWEN38_FLASH_NEXT_ATTENTION_PAGE_SIZE
                * <A as Arch>::HEAD_DIM,
            32_768
        );
        assert_eq!(cache_bytes_per_physical_page().unwrap(), 983_040);
        // 786,432 K/V plus 196,608 indexer, across all twelve QSA layers.
        assert_eq!(cache_bytes_per_physical_page().unwrap() / 12 * 12, 983_040);
        assert_eq!(cache_bytes_per_physical_page().unwrap() / 64, 15_360);
        assert_eq!(
            QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT * 2_764_800,
            67_947_724_800
        );
        assert_eq!(QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, 49);
    }

    #[test]
    fn the_kv_solver_spends_every_spare_byte_and_clears_the_dense_band() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();

        // The 25% posture funds 3,672 pages against the dense band's 264-page floor.
        assert!(plan.physical_pages() >= 264);
        assert!(
            plan.context_tokens_per_slot() >= crate::QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
        );
        assert_eq!(plan.physical_pages(), 3_672);
        assert_eq!(plan.context_tokens_per_slot(), 29_376);

        // Every arena together stays inside the admitted budget with the house headroom held
        // back, and the remainder is smaller than one more page.
        let total = plan.total_device_bytes().unwrap();
        let spendable =
            QWEN38_FLASH_NEXT_DEVICE_BUDGET_BYTES - QWEN38_FLASH_NEXT_REQUIRED_HEADROOM_BYTES;
        assert!(total <= spendable);
        assert!(spendable - total < cache_bytes_per_physical_page().unwrap());
    }

    #[test]
    fn a_thirty_percent_cache_no_longer_funds_the_dense_band_on_this_workspace() {
        // The wider workspace leaves only 215 pages at 30% expert residency, below the floor.
        let thirty =
            Qwen38FlashNextResidentLayout::plan(7_373, StreamingPrimarySource::Mapped, None)
                .unwrap();

        assert_eq!(thirty.physical_pages(), 215);
        assert!(thirty.physical_pages() < 264);
        assert!(
            thirty.context_tokens_per_slot() < crate::QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING
        );
    }

    #[test]
    fn the_shared_workspace_deviation_from_the_plan_is_accounted_plane_by_plane() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();
        let workspace = plan.workspace();
        let endpoint = plan.endpoint();

        assert_eq!(plan.workspace_bytes(), 526_358_560);

        // Ten caller-owned engram staging planes.
        let ple = workspace.ple_codes.byte_len()
            + workspace.ple_injected.byte_len()
            + workspace.ple_embedding.byte_len()
            + workspace.ple_key.byte_len()
            + workspace.ple_key_normed.byte_len()
            + workspace.ple_query_normed.byte_len()
            + workspace.ple_value.byte_len()
            + workspace.ple_gated.byte_len()
            + workspace.ple_gated_normed.byte_len()
            + workspace.ple_delta.byte_len();
        assert_eq!(ple, 159_907_840);

        // Three collapsing-mixer staging planes.
        let mixer = endpoint.mixer_normalized.byte_len()
            + endpoint.mixer_low_rank.byte_len()
            + endpoint.mixer_mixed.byte_len();
        assert_eq!(mixer, 26_869_760);

        // Fused HC/MoE intermediates are not materialized; indices are `u16`.
        assert_eq!(workspace.expert_indices.byte_len(), 20_480);
        assert_eq!(workspace.routed_output.byte_len(), 52_428_800);
    }

    #[test]
    fn the_resident_arena_represents_every_byte_it_accounts_for() {
        let plan = Qwen38FlashNextResidentLayout::build().unwrap();
        let represented =
            plan.resident_weight_bytes() + plan.persistent_state_bytes() + plan.workspace_bytes();

        // Alignment padding is the only difference, and it is small: 256 B per region
        // boundary at worst, against a 10 GiB arena.
        assert!(plan.resident_arena_bytes() >= represented);
        assert_eq!(plan.resident_arena_bytes() - represented, 11_744);
        assert!(plan.resident_arena_bytes() - represented < 32 * 1_024);
    }

    #[test]
    fn a_slot_budget_wider_than_the_inventory_is_refused() {
        assert!(
            Qwen38FlashNextResidentLayout::plan(
                QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT + 1,
                StreamingPrimarySource::Mapped,
                None,
            )
            .is_err()
        );
        assert!(
            Qwen38FlashNextResidentLayout::plan(0, StreamingPrimarySource::Mapped, None).is_err()
        );
    }

    #[test]
    fn a_cache_that_leaves_no_room_for_a_kv_pool_is_refused_rather_than_solved_to_zero() {
        // At the full inventory there is no device left for a single page, and the solver
        // says so instead of handing back a pool no request could use.
        assert!(
            Qwen38FlashNextResidentLayout::plan(
                QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT,
                StreamingPrimarySource::Mapped,
                None,
            )
            .is_err()
        );
    }
}

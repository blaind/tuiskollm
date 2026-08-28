//! Resident text inference ownership.

mod common;
mod error;
mod generation;
mod layout;
mod paged_kv_slots;
mod program;
mod qwen35;
mod qwen36;
mod qwen38;
mod qwen38_flash_next;
mod sampling;

#[cfg(feature = "qualification")]
pub use common::mtp::qualification_decide_sampled_tokens;
pub use common::mtp::{
    ResidentMtpGenerationStats, ResidentMtpGreedyStats, ResidentMtpSampledRound,
};
pub use common::progress::{ResidentLoadPhase, ResidentLoadProgress};
pub use common::streaming::{
    STREAMING_ABSENT_ITEM, STREAMING_ABSENT_SLOT, StreamingMappedPrimary, StreamingPrimarySource,
    StreamingRound, StreamingSlotAssignment, StreamingSlotCache, StreamingWeightLayout,
    StreamingWeightPool,
};
#[cfg(feature = "qualification")]
pub use common::text_generator::QualificationGeneration;
pub use common::text_generator::{
    ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents, ResidentCancellation,
    ResidentRequestId,
};
pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use generation::{
    CancelledText, ChatGenerationRequest, FinishReason, GeneratedText, GenerationSession,
    GenerationStep,
};
pub use layout::{EndpointLayout, LayerMemoryLayout, MAX_BATCH, StreamingResidencyAccounting};
pub use paged_kv_slots::{PagedKvRoute, PagedKvSlotPool, PagedKvSlotState, PagedKvTableUpdate};
pub use program::TextEndpointProgram;
pub use qwen35::batch_generation::Qwen35ResidentBatchGenerator;
pub use qwen35::full_attention_layer::Qwen35FullAttentionLayerProgram;
pub use qwen35::full_attention_layer_layout::Qwen35FullAttentionLayerLayout;
pub use qwen35::gdn_layer::Qwen35GdnLayerProgram;
pub use qwen35::gdn_layer_layout::Qwen35GdnLayerLayout;
pub use qwen35::long_context_kv::Qwen35LongContextKvProgram;
pub use qwen35::long_context_kv_layout::{
    QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS, Qwen35LongContextKvLayout,
};
pub use qwen35::mtp_batch_generation::{
    Qwen35ResidentMtpBatchEvent, Qwen35ResidentMtpBatchEvents, Qwen35ResidentMtpBatchGenerator,
};
pub use qwen35::mtp_generation::{
    Qwen35ResidentMtpGenerationSession, Qwen35ResidentMtpTextGenerator,
};
pub use qwen35::mtp_layer::Qwen35MtpLayerProgram;
pub use qwen35::mtp_layer_layout::Qwen35MtpLayerLayout;
pub use qwen35::nvfp4_mlp::Qwen35Nvfp4MlpProgram;
pub use qwen35::resident_model::{
    Qwen35ResidentLayerKind, Qwen35ResidentModelLayout, Qwen35ResidentModelProgram,
    Qwen35ResidentPrefillRoute,
};
pub use qwen35::resident_mtp::{Qwen35MtpPromptRoute, Qwen35ResidentMtpProgram};
pub use qwen35::resident_mtp_layout::Qwen35ResidentMtpLayout;
pub use qwen35::text_endpoint::Qwen35TextEndpointProgram;
pub use qwen35::text_endpoint_layout::Qwen35TextEndpointLayout;
pub use qwen36::batch_generation::Qwen36ResidentBatchGenerator;
pub use qwen36::full_attention_layer::Qwen36FullAttentionLayerProgram;
pub use qwen36::full_attention_layer_layout::Qwen36FullAttentionLayerLayout;
pub use qwen36::gdn_moe_layer::Qwen36GdnMoeLayerProgram;
pub use qwen36::gdn_moe_layer_layout::Qwen36GdnMoeLayerLayout;
pub use qwen36::long_context_kv::Qwen36LongContextKvProgram;
pub use qwen36::long_context_kv_layout::{
    QWEN36_LONG_CONTEXT_PHYSICAL_PAGES, QWEN36_MAX_CONTEXT_TOKENS, Qwen36LongContextKvLayout,
};
pub use qwen36::mtp_layer::Qwen36MtpLayerProgram;
pub use qwen36::mtp_layer_layout::Qwen36MtpLayerLayout;
pub use qwen36::resident_model::{
    Qwen36ResidentLayerKind, Qwen36ResidentModelLayout, Qwen36ResidentModelProgram,
    Qwen36ResidentPrefillRoute,
};
pub use qwen36::text_endpoint::Qwen36TextEndpointProgram;
pub use qwen36::text_endpoint_layout::Qwen36TextEndpointLayout;
pub use qwen38::batch_generation::ResidentBatchGenerator;
pub use qwen38::dense_fp8_gdn_layer::DenseFp8GdnLayerProgram;
pub use qwen38::dense_fp8_gdn_layer_layout::DenseFp8GdnLayerLayout;
pub use qwen38::dense_fp8_mlp::DenseFp8MlpProgram;
pub use qwen38::dense_fp8_mlp_layout::DenseFp8MlpLayout;
pub use qwen38::full_attention_layer::FullAttentionLayerProgram;
pub use qwen38::full_attention_layer_layout::FullAttentionLayerLayout;
pub use qwen38::long_context_kv_layout::{
    KvCacheCodec, KvCacheCodecDescriptor, LONG_CONTEXT_PHYSICAL_PAGES, MAX_CONTEXT_TOKENS,
    ResidentKvCapacityPlan, SharedPagedKvLayout, plan_resident_kv_capacity,
};
pub use qwen38::mtp_layer::MtpLayerProgram;
pub use qwen38::mtp_layer_layout::MtpLayerLayout;
pub use qwen38::mtp_prompt_prime::{MtpPromptPrimeProgram, MtpPromptPrimeRoute};
pub use qwen38::mtp_prompt_prime_layout::MtpPromptPrimeLayout;
pub use qwen38::nvfp4_mlp::Nvfp4MlpProgram;
pub use qwen38::nvfp4_mlp_layout::Nvfp4MlpLayout;
pub use qwen38::resident_model::{
    ResidentDecodeRoute, ResidentLoadMode, ResidentLoadStats, ResidentModelProgram,
    ResidentMtpSegmentedVerifyRoute, ResidentMtpVerifyRoute, ResidentPrefillRoute,
};
pub use qwen38::resident_model_layout::{ResidentLayerKind, ResidentModelLayout};
pub use qwen38::resident_mtp::{
    ResidentMtpDraftRoute, ResidentMtpLoadStats, ResidentMtpProgram, ResidentMtpPromptRoute,
    ResidentMtpRealignRoute,
};
pub use qwen38::resident_mtp_batch_generation::{
    ResidentMtpBatchEvent, ResidentMtpBatchEvents, ResidentMtpBatchGenerator,
};
pub use qwen38::resident_mtp_generation::{ResidentMtpGenerationSession, ResidentMtpTextGenerator};
pub use qwen38::resident_mtp_layout::ResidentMtpLayout;
pub use qwen38::upload_plan::{
    ResidentUploadArena, ResidentUploadEntry, ResidentUploadPlan, ResidentUploadPreparation,
};
pub use qwen38_flash_next::batch_generation::{
    Qwen38FlashNextBatchTelemetry, Qwen38FlashNextBatchWidthTelemetry,
    Qwen38FlashNextResidentBatchGenerator,
};
pub use qwen38_flash_next::compact_route::{
    Qwen38FlashNextCompactRound, qwen38_flash_next_admission_slot, qwen38_flash_next_compact_round,
    qwen38_flash_next_compact_survivors,
};
pub use qwen38_flash_next::engram_stager::{
    Qwen38FlashNextEngramStager, gather_qwen38_flash_next_engram_window,
};
pub use qwen38_flash_next::engram_stager_layout::{
    QWEN38_FLASH_NEXT_ENGRAM_WIDTHS, Qwen38FlashNextEngramStagerLayout,
    require_qwen38_flash_next_engram_width,
};
pub use qwen38_flash_next::expert_pool_layout::Qwen38FlashNextExpertPoolLayout;
pub use qwen38_flash_next::gdn_moe_layer::Qwen38FlashNextGdnMoeLayerProgram;
#[cfg(feature = "qualification")]
pub use qwen38_flash_next::gdn_moe_layer::{
    Qwen38FlashNextGdnMoeLayerImmutable, Qwen38FlashNextGdnMoeLayerInputs,
    Qwen38FlashNextGdnMoeLayerObservables,
};
pub use qwen38_flash_next::gdn_moe_layer_layout::Qwen38FlashNextGdnMoeLayerLayout;
pub use qwen38_flash_next::layer_route::{
    QWEN38_FLASH_NEXT_DENSE_QSA_VISIBLE_CEILING, QWEN38_FLASH_NEXT_MAX_ROWS,
    QWEN38_FLASH_NEXT_PREFILL_ROWS, Qwen38FlashNextRowRoute, qwen38_flash_next_row_route,
    require_qwen38_flash_next_dense_qsa_round, require_qwen38_flash_next_dense_qsa_visible,
};
pub use qwen38_flash_next::mtp_layout::{
    QWEN38_FLASH_NEXT_MTP_EXPERT_EXTENT_BYTES, QWEN38_FLASH_NEXT_MTP_EXPERT_ITEM_COUNT,
    QWEN38_FLASH_NEXT_MTP_EXPERT_RESIDENT_SLOTS, QWEN38_FLASH_NEXT_MTP_MAX_ROWS,
    QWEN38_FLASH_NEXT_MTP_ROUND_ROWS, QWEN38_FLASH_NEXT_MTP_TARGET_RESIDENT_SLOTS,
    Qwen38FlashNextMtpLayout, Qwen38FlashNextMtpResidency,
};
pub use qwen38_flash_next::mtp_program::{
    QWEN38_FLASH_NEXT_MTP_ROUTED_ROWS, QWEN38_FLASH_NEXT_MTP_ROUTES,
    QWEN38_FLASH_NEXT_MTP_SEGMENTS, QWEN38_FLASH_NEXT_PROPOSAL_ROWS, Qwen38FlashNextMtpLoadStats,
    Qwen38FlashNextMtpProgram, Qwen38FlashNextMtpStepTelemetry, Qwen38FlashNextMtpStream,
};
#[cfg(feature = "qualification")]
pub use qwen38_flash_next::qsa_moe_layer::{
    Qwen38FlashNextQsaMoeLayerImmutable, Qwen38FlashNextQsaMoeLayerInputs,
    Qwen38FlashNextQsaMoeLayerObservables,
};
pub use qwen38_flash_next::qsa_moe_layer::{
    Qwen38FlashNextQsaMoeLayerProgram, Qwen38FlashNextQsaRound,
};
pub use qwen38_flash_next::qsa_moe_layer_layout::Qwen38FlashNextQsaMoeLayerLayout;
pub use qwen38_flash_next::resident_model::{
    Qwen38FlashNextLayerStreamTelemetry, Qwen38FlashNextResidentLoadStats,
    Qwen38FlashNextResidentModel, Qwen38FlashNextSlotSnapshot, Qwen38FlashNextStepTelemetry,
    qwen38_flash_next_rope,
};
pub use qwen38_flash_next::resident_model_layout::{
    QWEN38_FLASH_NEXT_ATTENTION_LAYERS, QWEN38_FLASH_NEXT_DEVICE_BUDGET_BYTES,
    QWEN38_FLASH_NEXT_EXPERT_ITEM_COUNT, QWEN38_FLASH_NEXT_EXPERT_PRIMARY_EXTENT_BYTES,
    QWEN38_FLASH_NEXT_EXPERT_RESIDENT_SLOTS, QWEN38_FLASH_NEXT_EXPERT_SECONDARY_EXTENT_BYTES,
    QWEN38_FLASH_NEXT_LONG_CONTEXT_PHYSICAL_PAGES, QWEN38_FLASH_NEXT_PRIMARY_SOURCE,
    QWEN38_FLASH_NEXT_REQUIRED_HEADROOM_BYTES, QWEN38_FLASH_NEXT_RESIDENT_MAX_ROWS,
    QWEN38_FLASH_NEXT_RESIDENT_SEGMENTS, Qwen38FlashNextResidentLayout,
};
pub use qwen38_flash_next::slot_lifecycle::{
    QWEN38_FLASH_NEXT_SLOT_PAGE_TOKENS, QWEN38_FLASH_NEXT_UNMAPPED_PAGE, Qwen38FlashNextSlotChange,
    Qwen38FlashNextSlotPool, Qwen38FlashNextSlotRelease, Qwen38FlashNextSlotState,
};
/// Single-slot Qwen3.5 streaming request over the resident text program.
pub type Qwen35ResidentGenerationSession<'a> =
    common::text_generator::SingleSlotGenerationSession<'a, Qwen35ResidentModelProgram>;
/// Single-slot Qwen3.5 frontend, resident program, stream, and host-logit owner.
pub type Qwen35ResidentTextGenerator =
    common::text_generator::SingleSlotTextGenerator<Qwen35ResidentModelProgram>;
/// Single-slot Qwen3.6 streaming request over the resident text program.
pub type Qwen36ResidentGenerationSession<'a> =
    common::text_generator::SingleSlotGenerationSession<'a, Qwen36ResidentModelProgram>;
/// Single-slot Qwen3.6 frontend, resident program, stream, and host-logit owner.
pub type Qwen36ResidentTextGenerator =
    common::text_generator::SingleSlotTextGenerator<Qwen36ResidentModelProgram>;
pub use qwen38_flash_next::text_generation::Qwen38FlashNextGenerationTelemetry;
/// Single-slot Qwen3.8 Flash-Next streaming request.
pub type Qwen38FlashNextGenerationSession<'a> =
    common::text_generator::SingleSlotGenerationSession<'a, Qwen38FlashNextResidentModel>;
/// Single-slot Qwen3.8 Flash-Next generator.
pub type Qwen38FlashNextTextGenerator =
    common::text_generator::SingleSlotTextGenerator<Qwen38FlashNextResidentModel>;
/// Single-slot Qwen3.8 streaming request over the resident text program.
pub type ResidentGenerationSession<'a> =
    common::text_generator::SingleSlotGenerationSession<'a, ResidentModelProgram>;
/// Single-slot Qwen3.8 frontend, resident program, stream, and host-logit owner.
pub type ResidentTextGenerator =
    common::text_generator::SingleSlotTextGenerator<ResidentModelProgram>;

pub use sampling::{
    SampleDecision, Sampler, SamplingDistribution, SamplingOptions, SamplingPenalties,
    SpeculativeDecision, speculative_accept_probability, speculative_decision,
    speculative_residual,
};

#[cfg(feature = "qualification")]
pub use program::EndpointObservables;
#[cfg(feature = "qualification")]
pub use qwen35::full_attention_layer::{
    Qwen35FullAttentionLayerImmutable, Qwen35FullAttentionLayerObservables,
};
#[cfg(feature = "qualification")]
pub use qwen35::gdn_layer::{Qwen35GdnLayerImmutable, Qwen35GdnLayerObservables};
#[cfg(feature = "qualification")]
pub use qwen35::mtp_layer::Qwen35MtpLayerObservables;
#[cfg(feature = "qualification")]
pub use qwen35::resident_model::Qwen35ResidentModelObservables;
#[cfg(feature = "qualification")]
pub use qwen35::resident_mtp::Qwen35ResidentMtpObservables;
#[cfg(feature = "qualification")]
pub use qwen35::text_endpoint::Qwen35EndpointObservables;
#[cfg(feature = "qualification")]
pub use qwen36::full_attention_layer::{
    Qwen36FullAttentionLayerImmutable, Qwen36FullAttentionLayerInputs,
    Qwen36FullAttentionLayerObservables,
};
#[cfg(feature = "qualification")]
pub use qwen36::gdn_moe_layer::{
    Qwen36GdnMoeLayerImmutable, Qwen36GdnMoeLayerInputs, Qwen36GdnMoeLayerObservables,
};
#[cfg(feature = "qualification")]
pub use qwen36::mtp_layer::Qwen36MtpLayerObservables;
#[cfg(feature = "qualification")]
pub use qwen36::resident_model::Qwen36ResidentModelObservables;
#[cfg(feature = "qualification")]
pub use qwen36::text_endpoint::{Qwen36EndpointImmutable, Qwen36EndpointObservables};
#[cfg(feature = "qualification")]
pub use qwen38::dense_fp8_gdn_layer::DenseFp8GdnLayerObservables;
#[cfg(feature = "qualification")]
pub use qwen38::dense_fp8_mlp::DenseFp8MlpObservables;
#[cfg(feature = "qualification")]
pub use qwen38::full_attention_layer::FullAttentionLayerObservables;
#[cfg(feature = "qualification")]
pub use qwen38::mtp_layer::MtpLayerObservables;
#[cfg(feature = "qualification")]
pub use qwen38::mtp_prompt_prime::MtpPromptPrimeObservables;
#[cfg(feature = "qualification")]
pub use qwen38::nvfp4_mlp::{Nvfp4MlpImmutable, Nvfp4MlpObservables};
#[cfg(feature = "qualification")]
pub use qwen38::resident_model::{
    ResidentEmbeddingStageGraph, ResidentLongContextObservables, ResidentModelObservables,
    ResidentMtpGdnObservables, ResidentMtpLayerObservables, ResidentMtpSegmentedStageGraph,
    ResidentMtpVerifyObservables, ResidentPrefillStageGraph,
};
#[cfg(feature = "qualification")]
pub use qwen38::resident_mtp::ResidentMtpObservables;

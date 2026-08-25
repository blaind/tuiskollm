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
mod resident_generation;
mod sampling;

#[cfg(feature = "qualification")]
pub use common::mtp::qualification_decide_sampled_tokens;
pub use common::mtp::{
    ResidentMtpGenerationStats, ResidentMtpGreedyStats, ResidentMtpSampledRound,
};
pub use common::progress::{ResidentLoadPhase, ResidentLoadProgress};
pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use generation::{
    CancelledText, ChatGenerationRequest, FinishReason, GeneratedText, GenerationSession,
    GenerationStep,
};
pub use layout::{EndpointLayout, LayerMemoryLayout, MAX_BATCH};
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
pub use resident_generation::{
    ResidentBatchAdmission, ResidentBatchEvent, ResidentBatchEvents, ResidentBatchGenerator,
    ResidentCancellation, ResidentRequestId,
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

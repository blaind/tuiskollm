//! Resident text inference ownership.

mod dense_fp8_gdn_layer;
mod dense_fp8_gdn_layer_layout;
mod dense_fp8_mlp;
mod dense_fp8_mlp_layout;
mod error;
mod full_attention_layer;
mod full_attention_layer_layout;
mod generation;
mod layout;
mod long_context_kv_layout;
mod mtp_layer;
mod mtp_layer_layout;
mod mtp_prompt_prime;
mod mtp_prompt_prime_layout;
mod nvfp4_mlp;
mod nvfp4_mlp_layout;
mod paged_kv_slots;
mod program;
mod qwen35_full_attention_layer;
mod qwen35_full_attention_layer_layout;
mod qwen35_gdn_layer;
mod qwen35_gdn_layer_layout;
mod qwen35_nvfp4_mlp;
mod qwen35_resident_model;
mod qwen35_text_endpoint;
mod qwen35_text_endpoint_layout;
mod resident_generation;
mod resident_model_layout;
mod resident_mtp;
mod resident_mtp_batch_generation;
mod resident_mtp_generation;
mod resident_mtp_layout;
mod sampling;

pub use dense_fp8_gdn_layer::DenseFp8GdnLayerProgram;
pub use dense_fp8_gdn_layer_layout::DenseFp8GdnLayerLayout;
pub use dense_fp8_mlp::DenseFp8MlpProgram;
pub use dense_fp8_mlp_layout::DenseFp8MlpLayout;
pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use full_attention_layer::FullAttentionLayerProgram;
pub use full_attention_layer_layout::FullAttentionLayerLayout;
pub use generation::{
    CancelledText, ChatGenerationRequest, FinishReason, GeneratedText, GenerationSession,
    GenerationStep,
};
pub use layout::{EndpointLayout, MAX_BATCH};
pub use long_context_kv_layout::{
    KvCacheCodec, KvCacheCodecDescriptor, LONG_CONTEXT_PHYSICAL_PAGES, MAX_CONTEXT_TOKENS,
    ResidentKvCapacityPlan, SharedPagedKvLayout, plan_resident_kv_capacity,
};
pub use mtp_layer::MtpLayerProgram;
pub use mtp_layer_layout::MtpLayerLayout;
pub use mtp_prompt_prime::{MtpPromptPrimeProgram, MtpPromptPrimeRoute};
pub use mtp_prompt_prime_layout::MtpPromptPrimeLayout;
pub use nvfp4_mlp::Nvfp4MlpProgram;
pub use nvfp4_mlp_layout::Nvfp4MlpLayout;
pub use paged_kv_slots::{PagedKvRoute, PagedKvSlotPool, PagedKvSlotState, PagedKvTableUpdate};
pub use program::TextEndpointProgram;
pub use qwen35_full_attention_layer::Qwen35FullAttentionLayerProgram;
pub use qwen35_full_attention_layer_layout::Qwen35FullAttentionLayerLayout;
pub use qwen35_gdn_layer::Qwen35GdnLayerProgram;
pub use qwen35_gdn_layer_layout::Qwen35GdnLayerLayout;
pub use qwen35_nvfp4_mlp::Qwen35Nvfp4MlpProgram;
pub use qwen35_resident_model::{
    Qwen35ResidentLayerKind, Qwen35ResidentModelLayout, Qwen35ResidentModelProgram,
};
pub use qwen35_text_endpoint::Qwen35TextEndpointProgram;
pub use qwen35_text_endpoint_layout::Qwen35TextEndpointLayout;
pub use resident_generation::{
    Qwen35ResidentGenerationSession, Qwen35ResidentTextGenerator, ResidentBatchAdmission,
    ResidentBatchEvent, ResidentBatchEvents, ResidentBatchGenerator, ResidentCancellation,
    ResidentGenerationSession, ResidentRequestId, ResidentTextGenerator,
};
pub use resident_model_layout::{
    ResidentDecodeRoute, ResidentLayerKind, ResidentLoadMode, ResidentLoadPhase,
    ResidentLoadProgress, ResidentLoadStats, ResidentModelLayout, ResidentModelProgram,
    ResidentMtpSegmentedVerifyRoute, ResidentMtpVerifyRoute, ResidentPrefillRoute,
    ResidentUploadArena, ResidentUploadEntry, ResidentUploadPlan, ResidentUploadPreparation,
};
pub use resident_mtp::{
    ResidentMtpDraftRoute, ResidentMtpLoadStats, ResidentMtpProgram, ResidentMtpPromptRoute,
    ResidentMtpRealignRoute,
};
pub use resident_mtp_batch_generation::{
    ResidentMtpBatchEvent, ResidentMtpBatchEvents, ResidentMtpBatchGenerator,
};
#[cfg(feature = "qualification")]
pub use resident_mtp_generation::qualification_decide_sampled_tokens;
pub use resident_mtp_generation::{
    ResidentMtpGenerationSession, ResidentMtpGenerationStats, ResidentMtpGreedyStats,
    ResidentMtpSampledRound, ResidentMtpTextGenerator,
};
pub use resident_mtp_layout::ResidentMtpLayout;
pub use sampling::{
    SampleDecision, Sampler, SamplingDistribution, SamplingOptions, SamplingPenalties,
    SpeculativeDecision, speculative_accept_probability, speculative_decision,
    speculative_residual,
};

#[cfg(feature = "qualification")]
pub use dense_fp8_gdn_layer::DenseFp8GdnLayerObservables;
#[cfg(feature = "qualification")]
pub use dense_fp8_mlp::DenseFp8MlpObservables;
#[cfg(feature = "qualification")]
pub use full_attention_layer::FullAttentionLayerObservables;
#[cfg(feature = "qualification")]
pub use mtp_layer::MtpLayerObservables;
#[cfg(feature = "qualification")]
pub use mtp_prompt_prime::MtpPromptPrimeObservables;
#[cfg(feature = "qualification")]
pub use nvfp4_mlp::{Nvfp4MlpImmutable, Nvfp4MlpObservables};
#[cfg(feature = "qualification")]
pub use program::EndpointObservables;
#[cfg(feature = "qualification")]
pub use qwen35_full_attention_layer::{
    Qwen35FullAttentionLayerImmutable, Qwen35FullAttentionLayerObservables,
};
#[cfg(feature = "qualification")]
pub use qwen35_gdn_layer::{Qwen35GdnLayerImmutable, Qwen35GdnLayerObservables};
#[cfg(feature = "qualification")]
pub use qwen35_resident_model::Qwen35ResidentModelObservables;
#[cfg(feature = "qualification")]
pub use qwen35_text_endpoint::Qwen35EndpointObservables;
#[cfg(feature = "qualification")]
pub use resident_model_layout::{
    ResidentEmbeddingStageGraph, ResidentLongContextObservables, ResidentModelObservables,
    ResidentMtpGdnObservables, ResidentMtpLayerObservables, ResidentMtpSegmentedStageGraph,
    ResidentMtpVerifyObservables, ResidentPrefillStageGraph,
};
#[cfg(feature = "qualification")]
pub use resident_mtp::ResidentMtpObservables;

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
mod program;
mod sampling;

pub use dense_fp8_gdn_layer::DenseFp8GdnLayerProgram;
pub use dense_fp8_gdn_layer_layout::DenseFp8GdnLayerLayout;
pub use dense_fp8_mlp::DenseFp8MlpProgram;
pub use dense_fp8_mlp_layout::DenseFp8MlpLayout;
pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use full_attention_layer::FullAttentionLayerProgram;
pub use full_attention_layer_layout::FullAttentionLayerLayout;
pub use generation::{
    ChatGenerationRequest, FinishReason, GeneratedText, GenerationSession, GenerationStep,
};
pub use layout::{EndpointLayout, MAX_BATCH};
pub use program::TextEndpointProgram;
pub use sampling::{SampleDecision, Sampler, SamplingOptions};

#[cfg(feature = "qualification")]
pub use dense_fp8_gdn_layer::DenseFp8GdnLayerObservables;
#[cfg(feature = "qualification")]
pub use dense_fp8_mlp::DenseFp8MlpObservables;
#[cfg(feature = "qualification")]
pub use full_attention_layer::FullAttentionLayerObservables;
#[cfg(feature = "qualification")]
pub use program::EndpointObservables;

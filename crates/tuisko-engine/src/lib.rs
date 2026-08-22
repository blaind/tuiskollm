//! Resident text inference ownership.

mod dense_fp8_mlp;
mod dense_fp8_mlp_layout;
mod error;
mod generation;
mod layout;
mod program;
mod sampling;

pub use dense_fp8_mlp::DenseFp8MlpProgram;
pub use dense_fp8_mlp_layout::DenseFp8MlpLayout;
pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use generation::{
    ChatGenerationRequest, FinishReason, GeneratedText, GenerationSession, GenerationStep,
};
pub use layout::{EndpointLayout, MAX_BATCH};
pub use program::TextEndpointProgram;
pub use sampling::{SampleDecision, Sampler, SamplingOptions};

#[cfg(feature = "qualification")]
pub use dense_fp8_mlp::DenseFp8MlpObservables;
#[cfg(feature = "qualification")]
pub use program::EndpointObservables;

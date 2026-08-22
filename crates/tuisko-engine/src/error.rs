//! Engine error categories.

use std::fmt::{self, Display, Formatter};
use tuisko_frontend::FrontendError;
use tuisko_gpu::GpuError;
use tuisko_model::CheckpointError;

/// Stable category for an engine contract failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum EngineErrorCode {
    /// A batch or token identifier has no admitted route.
    Route,
    /// A resident layout or byte count is invalid.
    Layout,
    /// Sampling configuration or logits violate the text contract.
    Sampling,
    /// A generation session entered an invalid state.
    Generation,
}

impl EngineErrorCode {
    /// Stable external spelling of this category.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Route => "engine.route",
            Self::Layout => "engine.layout",
            Self::Sampling => "engine.sampling",
            Self::Generation => "engine.generation",
        }
    }
}

impl Display for EngineErrorCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Failure returned by engine ownership and routing operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A checked engine contract was violated.
    #[error("[{code}] {message}")]
    Contract {
        /// Stable external error category.
        code: EngineErrorCode,
        /// Contextual failure detail.
        message: String,
    },

    /// Checkpoint admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Frontend admission or text processing failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),

    /// GPU ownership or execution failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

impl EngineError {
    /// Stable engine category, when this is an engine contract failure.
    pub const fn code(&self) -> Option<EngineErrorCode> {
        match self {
            Self::Contract { code, .. } => Some(*code),
            Self::Checkpoint(_) | Self::Frontend(_) | Self::Gpu(_) => None,
        }
    }

    pub(crate) fn route(message: impl Into<String>) -> Self {
        Self::Contract {
            code: EngineErrorCode::Route,
            message: message.into(),
        }
    }

    pub(crate) fn layout(message: impl Into<String>) -> Self {
        Self::Contract {
            code: EngineErrorCode::Layout,
            message: message.into(),
        }
    }

    pub(crate) fn sampling(message: impl Into<String>) -> Self {
        Self::Contract {
            code: EngineErrorCode::Sampling,
            message: message.into(),
        }
    }

    pub(crate) fn generation(message: impl Into<String>) -> Self {
        Self::Contract {
            code: EngineErrorCode::Generation,
            message: message.into(),
        }
    }
}

/// Result type for engine operations.
pub type EngineResult<T> = Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::{EngineError, EngineErrorCode};

    #[test]
    fn external_error_codes_are_unique_and_stable() {
        let codes = [
            EngineErrorCode::Route,
            EngineErrorCode::Layout,
            EngineErrorCode::Sampling,
            EngineErrorCode::Generation,
        ];
        let unique = codes
            .iter()
            .copied()
            .map(EngineErrorCode::as_str)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(unique.len(), codes.len());
        assert_eq!(EngineErrorCode::Route.as_str(), "engine.route");
        assert_eq!(EngineErrorCode::Layout.as_str(), "engine.layout");
        assert_eq!(EngineErrorCode::Sampling.as_str(), "engine.sampling");
        assert_eq!(EngineErrorCode::Generation.as_str(), "engine.generation");
    }

    #[test]
    fn contract_error_preserves_category_and_context() {
        let error = EngineError::route("batch 9 has no admitted route");

        assert_eq!(error.code(), Some(EngineErrorCode::Route));
        assert!(error.to_string().contains("batch 9"));
    }
}

//! Stable checkpoint error categories with contextual failure details.

use std::fmt::{self, Display, Formatter};
use std::path::{Path, PathBuf};

/// Stable category for a checkpoint error crossing a user-facing boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CheckpointErrorCode {
    /// Filesystem, metadata, or memory-mapping failure.
    Io,
    /// JSON syntax or schema failure.
    Json,
    /// Model configuration contract mismatch.
    Config,
    /// Snapshot revision mismatch.
    Revision,
    /// File inventory or index mismatch.
    Inventory,
    /// Safetensors framing or descriptor failure.
    Safetensors,
    /// Missing tensor or typed tensor-view mismatch.
    Tensor,
    /// Operator source-plane binding mismatch.
    SourceBinding,
}

impl CheckpointErrorCode {
    /// Stable external spelling of this category.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Io => "checkpoint.io",
            Self::Json => "checkpoint.json",
            Self::Config => "checkpoint.config",
            Self::Revision => "checkpoint.revision",
            Self::Inventory => "checkpoint.inventory",
            Self::Safetensors => "checkpoint.safetensors",
            Self::Tensor => "checkpoint.tensor",
            Self::SourceBinding => "checkpoint.source_binding",
        }
    }
}

impl Display for CheckpointErrorCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Checkpoint failure with a stable external category and detailed context.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// Filesystem or memory-mapping operation failed.
    #[error("[checkpoint.io] {action} {path}: {source}", path = .path.display())]
    Io {
        /// Operation that failed.
        action: &'static str,
        /// Path involved in the operation.
        path: PathBuf,
        /// Underlying I/O failure.
        source: std::io::Error,
    },

    /// JSON syntax or schema validation failed.
    #[error("[checkpoint.json] parsing JSON in {path}: {source}", path = .path.display())]
    Json {
        /// JSON file being parsed.
        path: PathBuf,
        /// Underlying JSON failure.
        source: serde_json::Error,
    },

    /// An admitted checkpoint contract was violated.
    #[error("[{code}] {message}")]
    Contract {
        /// Stable external error category.
        code: CheckpointErrorCode,
        /// Contextual failure detail.
        message: String,
    },
}

impl CheckpointError {
    /// Stable category suitable for logs and transport error payloads.
    pub const fn code(&self) -> CheckpointErrorCode {
        match self {
            Self::Io { .. } => CheckpointErrorCode::Io,
            Self::Json { .. } => CheckpointErrorCode::Json,
            Self::Contract { code, .. } => *code,
        }
    }

    pub(crate) fn io(action: &'static str, path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            action,
            path: path.to_owned(),
            source,
        }
    }

    pub(crate) fn json(path: &Path, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.to_owned(),
            source,
        }
    }

    pub(crate) fn config(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::Config, message)
    }

    pub(crate) fn revision(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::Revision, message)
    }

    pub(crate) fn inventory(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::Inventory, message)
    }

    pub(crate) fn safetensors(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::Safetensors, message)
    }

    pub(crate) fn tensor(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::Tensor, message)
    }

    pub(crate) fn source_binding(message: impl Into<String>) -> Self {
        Self::contract(CheckpointErrorCode::SourceBinding, message)
    }

    fn contract(code: CheckpointErrorCode, message: impl Into<String>) -> Self {
        Self::Contract {
            code,
            message: message.into(),
        }
    }
}

/// Result returned by checkpoint admission and source-layout operations.
pub type CheckpointResult<T> = Result<T, CheckpointError>;

#[cfg(test)]
mod tests {
    use super::{CheckpointError, CheckpointErrorCode};
    use std::error::Error;
    use std::path::{Path, PathBuf};

    #[test]
    fn io_error_preserves_context_and_source() {
        let error = CheckpointError::io(
            "opening",
            Path::new("model.safetensors"),
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        );

        assert_eq!(error.code(), CheckpointErrorCode::Io);
        assert_eq!(
            error.to_string(),
            "[checkpoint.io] opening model.safetensors: missing"
        );
        assert!(error.source().is_some());
    }

    #[test]
    fn json_error_preserves_context_and_source() {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let error = CheckpointError::json(&PathBuf::from("config.json"), source);

        assert_eq!(error.code(), CheckpointErrorCode::Json);
        assert!(
            error
                .to_string()
                .starts_with("[checkpoint.json] parsing JSON in config.json:")
        );
        assert!(error.source().is_some());
    }

    #[test]
    fn contract_error_preserves_code_without_a_source() {
        let error = CheckpointError::tensor("invalid tensor");

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert_eq!(error.to_string(), "[checkpoint.tensor] invalid tensor");
        assert!(error.source().is_none());
    }

    #[test]
    fn external_error_codes_are_unique_and_stable() {
        let codes = [
            (CheckpointErrorCode::Io, "checkpoint.io"),
            (CheckpointErrorCode::Json, "checkpoint.json"),
            (CheckpointErrorCode::Config, "checkpoint.config"),
            (CheckpointErrorCode::Revision, "checkpoint.revision"),
            (CheckpointErrorCode::Inventory, "checkpoint.inventory"),
            (CheckpointErrorCode::Safetensors, "checkpoint.safetensors"),
            (CheckpointErrorCode::Tensor, "checkpoint.tensor"),
            (
                CheckpointErrorCode::SourceBinding,
                "checkpoint.source_binding",
            ),
        ];

        for (index, (code, expected)) in codes.iter().enumerate() {
            assert_eq!(code.as_str(), *expected);
            assert!(
                codes[..index]
                    .iter()
                    .all(|(prior, _)| prior.as_str() != code.as_str())
            );
        }
    }
}

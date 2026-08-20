use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("{action} {path}: {source}", path = .path.display())]
    Io {
        action: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("parsing JSON in {path}: {source}", path = .path.display())]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("{0}")]
    Invalid(String),
}

impl CheckpointError {
    pub(crate) fn io(action: &'static str, path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            action,
            path: path.to_owned(),
            source,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

pub type CheckpointResult<T> = Result<T, CheckpointError>;

#[cfg(test)]
mod tests {
    use super::CheckpointError;
    use std::error::Error;
    use std::path::{Path, PathBuf};

    #[test]
    fn io_error_preserves_context_and_source() {
        let error = CheckpointError::io(
            "opening",
            Path::new("model.safetensors"),
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        );

        assert_eq!(error.to_string(), "opening model.safetensors: missing");
        assert!(error.source().is_some());
    }

    #[test]
    fn json_error_preserves_context_and_source() {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let error = CheckpointError::Json {
            path: PathBuf::from("config.json"),
            source,
        };

        assert!(
            error
                .to_string()
                .starts_with("parsing JSON in config.json:")
        );
        assert!(error.source().is_some());
    }

    #[test]
    fn invalid_error_has_no_source() {
        let error = CheckpointError::invalid("invalid tensor");

        assert_eq!(error.to_string(), "invalid tensor");
        assert!(error.source().is_none());
    }
}

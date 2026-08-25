use std::fmt;
use std::io;
use std::path::PathBuf;

/// Stable category for a frontend failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendErrorCode {
    /// A required snapshot file could not be read.
    Io,
    /// A snapshot JSON file could not be decoded.
    Json,
    /// The tokenizer could not load, encode, or decode text.
    Tokenizer,
    /// The chat template could not render the request.
    Template,
    /// Frontend metadata differs from the pinned product contract.
    Contract,
}

impl FrontendErrorCode {
    /// Returns the stable external spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Io => "frontend.io",
            Self::Json => "frontend.json",
            Self::Tokenizer => "frontend.tokenizer",
            Self::Template => "frontend.template",
            Self::Contract => "frontend.contract",
        }
    }
}

impl fmt::Display for FrontendErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Frontend admission or text-processing failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FrontendError {
    /// A required snapshot file could not be read.
    #[error("{}: could not read {}: {source}", FrontendErrorCode::Io, path.display())]
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: io::Error,
    },
    /// A JSON file could not be decoded.
    #[error("{}: could not decode {}: {source}", FrontendErrorCode::Json, path.display())]
    Json {
        /// File that could not be decoded.
        path: PathBuf,
        /// Underlying JSON failure.
        #[source]
        source: serde_json::Error,
    },
    /// The tokenizer rejected an operation.
    #[error("{}: {}", FrontendErrorCode::Tokenizer, .0)]
    Tokenizer(String),
    /// The chat template could not render the request.
    #[error("{}: {}", FrontendErrorCode::Template, .0)]
    Template(#[from] minijinja::Error),
    /// Snapshot metadata differs from the pinned contract.
    #[error("{}: {}", FrontendErrorCode::Contract, .0)]
    Contract(String),
}

impl FrontendError {
    /// Returns the stable category for this failure.
    pub const fn code(&self) -> FrontendErrorCode {
        match self {
            Self::Io { .. } => FrontendErrorCode::Io,
            Self::Json { .. } => FrontendErrorCode::Json,
            Self::Tokenizer(_) => FrontendErrorCode::Tokenizer,
            Self::Template(_) => FrontendErrorCode::Template,
            Self::Contract(_) => FrontendErrorCode::Contract,
        }
    }
}

/// Result returned by frontend operations.
pub type FrontendResult<T> = Result<T, FrontendError>;

#[cfg(test)]
mod tests {
    use super::{FrontendError, FrontendErrorCode};

    #[test]
    fn external_codes_are_unique_and_stable() {
        let codes = [
            FrontendErrorCode::Io,
            FrontendErrorCode::Json,
            FrontendErrorCode::Tokenizer,
            FrontendErrorCode::Template,
            FrontendErrorCode::Contract,
        ];

        let spellings = codes.map(FrontendErrorCode::as_str);
        assert_eq!(
            spellings,
            [
                "frontend.io",
                "frontend.json",
                "frontend.tokenizer",
                "frontend.template",
                "frontend.contract",
            ]
        );

        for (index, spelling) in spellings.iter().enumerate() {
            assert!(!spellings[..index].contains(spelling));
        }

        let error = FrontendError::Contract("bad fixture".into());
        assert_eq!(error.code(), FrontendErrorCode::Contract);
        assert_eq!(error.to_string(), "frontend.contract: bad fixture");
    }

    #[test]
    fn every_variant_message_starts_with_its_code() {
        let errors = [
            FrontendError::Io {
                path: "snapshot.json".into(),
                source: std::io::Error::other("bad fixture"),
            },
            FrontendError::Json {
                path: "snapshot.json".into(),
                source: serde_json::from_str::<serde_json::Value>("{").unwrap_err(),
            },
            FrontendError::Tokenizer("bad fixture".into()),
            FrontendError::Template(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "bad fixture",
            )),
            FrontendError::Contract("bad fixture".into()),
        ];

        for error in errors {
            let prefix = format!("{}: ", error.code());
            assert!(error.to_string().starts_with(&prefix), "{error}");
        }
    }
}

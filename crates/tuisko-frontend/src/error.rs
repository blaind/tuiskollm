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
    #[error("frontend.io: could not read {}: {source}", path.display())]
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: io::Error,
    },
    /// A JSON file could not be decoded.
    #[error("frontend.json: could not decode {}: {source}", path.display())]
    Json {
        /// File that could not be decoded.
        path: PathBuf,
        /// Underlying JSON failure.
        #[source]
        source: serde_json::Error,
    },
    /// The tokenizer rejected an operation.
    #[error("frontend.tokenizer: {0}")]
    Tokenizer(String),
    /// The chat template could not render the request.
    #[error("frontend.template: {0}")]
    Template(#[from] minijinja::Error),
    /// Snapshot metadata differs from the pinned contract.
    #[error("frontend.contract: {0}")]
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
}

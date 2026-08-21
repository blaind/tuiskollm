//! Resident text inference ownership.

mod error;
mod layout;
mod program;

pub use error::{EngineError, EngineErrorCode, EngineResult};
pub use layout::{EndpointLayout, MAX_BATCH};
pub use program::TextEndpointProgram;

#[cfg(feature = "qualification")]
pub use program::EndpointObservables;

//! Stable GPU error categories with contextual failure details.

use cuda_core::{DriverError, LaunchContractError};
use cuda_host::EmbeddedModuleError;
use std::fmt::{self, Display, Formatter};

/// Stable category for a GPU ownership or driver failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GpuErrorCode {
    /// CUDA driver operation failed.
    Driver,
    /// The device does not currently have enough free memory for an exact allocation.
    Memory,
    /// Arena layout or address validation failed.
    Arena,
    /// CUDA Graph capture, instantiation, or replay failed.
    Graph,
    /// Resources from different CUDA contexts were combined.
    Context,
    /// A kernel launch contract was rejected.
    Launch,
    /// An embedded device module could not be loaded.
    Module,
}

impl GpuErrorCode {
    /// Stable external spelling of this category.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Driver => "gpu.driver",
            Self::Memory => "gpu.memory",
            Self::Arena => "gpu.arena",
            Self::Graph => "gpu.graph",
            Self::Context => "gpu.context",
            Self::Launch => "gpu.launch",
            Self::Module => "gpu.module",
        }
    }
}

impl Display for GpuErrorCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Failure returned by GPU ownership primitives.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GpuError {
    /// CUDA driver operation failed.
    #[error("[gpu.driver] {operation}: {source}")]
    Driver {
        /// Operation that failed.
        operation: &'static str,
        /// Underlying CUDA driver failure.
        source: DriverError,
    },

    /// An exact device allocation exceeds the memory currently available.
    #[error(
        "[gpu.memory] {operation}: not enough GPU memory: requires {required:.2} GiB, but {free:.2} GiB is free of {total:.2} GiB (short by {shortfall:.2} GiB); free GPU memory and retry",
        required = gibibytes(*required_bytes),
        free = gibibytes(*free_bytes),
        total = gibibytes(*total_bytes),
        shortfall = gibibytes(required_bytes.saturating_sub(*free_bytes)),
    )]
    OutOfMemory {
        /// Allocation operation that could not proceed.
        operation: &'static str,
        /// Complete physical byte count still required by the operation.
        required_bytes: usize,
        /// Device bytes free when the allocation was checked.
        free_bytes: usize,
        /// Total physical device bytes visible to the CUDA context.
        total_bytes: usize,
    },

    /// A prepared kernel launch violated its static or live-device contract.
    #[error("[gpu.launch] {operation}: {source}")]
    Launch {
        /// Operation that failed.
        operation: &'static str,
        /// Rejected launch contract.
        source: LaunchContractError,
    },

    /// An embedded device module could not be discovered, built, or loaded.
    #[error("[gpu.module] {operation}: {source}")]
    Module {
        /// Operation that failed.
        operation: &'static str,
        /// Underlying embedded-module failure.
        source: EmbeddedModuleError,
    },

    /// A checked GPU ownership contract was violated.
    #[error("[{code}] {message}")]
    Contract {
        /// Stable external error category.
        code: GpuErrorCode,
        /// Contextual failure detail.
        message: String,
    },
}

impl GpuError {
    /// Stable category suitable for logs and transport error payloads.
    pub const fn code(&self) -> GpuErrorCode {
        match self {
            Self::Driver { .. } => GpuErrorCode::Driver,
            Self::OutOfMemory { .. } => GpuErrorCode::Memory,
            Self::Launch { .. } => GpuErrorCode::Launch,
            Self::Module { .. } => GpuErrorCode::Module,
            Self::Contract { code, .. } => *code,
        }
    }

    pub(crate) fn driver(operation: &'static str, source: DriverError) -> Self {
        Self::Driver { operation, source }
    }

    pub(crate) fn out_of_memory(
        operation: &'static str,
        required_bytes: usize,
        free_bytes: usize,
        total_bytes: usize,
    ) -> Self {
        Self::OutOfMemory {
            operation,
            required_bytes,
            free_bytes,
            total_bytes,
        }
    }

    pub(crate) fn arena(message: impl Into<String>) -> Self {
        Self::contract(GpuErrorCode::Arena, message)
    }

    pub(crate) fn graph(message: impl Into<String>) -> Self {
        Self::contract(GpuErrorCode::Graph, message)
    }

    pub(crate) fn context(message: impl Into<String>) -> Self {
        Self::contract(GpuErrorCode::Context, message)
    }

    /// Wraps a rejected kernel launch contract with operation context.
    pub fn launch(operation: &'static str, source: LaunchContractError) -> Self {
        Self::Launch { operation, source }
    }

    /// Creates a launch-contract failure without a driver source.
    pub fn invalid_launch(message: impl Into<String>) -> Self {
        Self::contract(GpuErrorCode::Launch, message)
    }

    /// Wraps an embedded-module failure with operation context.
    pub fn module(operation: &'static str, source: EmbeddedModuleError) -> Self {
        Self::Module { operation, source }
    }

    fn contract(code: GpuErrorCode, message: impl Into<String>) -> Self {
        Self::Contract {
            code,
            message: message.into(),
        }
    }
}

impl From<DriverError> for GpuError {
    fn from(source: DriverError) -> Self {
        Self::driver("CUDA driver operation failed", source)
    }
}

/// Result returned by GPU ownership operations.
pub type GpuResult<T> = Result<T, GpuError>;

fn gibibytes(bytes: usize) -> f64 {
    bytes as f64 / (1_u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::{GpuError, GpuErrorCode};

    #[test]
    fn external_error_codes_are_unique_and_stable() {
        let codes = [
            (GpuErrorCode::Driver, "gpu.driver"),
            (GpuErrorCode::Memory, "gpu.memory"),
            (GpuErrorCode::Arena, "gpu.arena"),
            (GpuErrorCode::Graph, "gpu.graph"),
            (GpuErrorCode::Context, "gpu.context"),
            (GpuErrorCode::Launch, "gpu.launch"),
            (GpuErrorCode::Module, "gpu.module"),
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

    #[test]
    fn contract_error_preserves_category_and_context() {
        let error = GpuError::arena("region is outside its allocation");

        assert_eq!(error.code(), GpuErrorCode::Arena);
        assert_eq!(
            error.to_string(),
            "[gpu.arena] region is outside its allocation"
        );
    }

    #[test]
    fn out_of_memory_reports_required_free_total_and_shortfall() {
        let gib = 1usize << 30;
        let error =
            GpuError::out_of_memory("allocating VMM arena backing", 24 * gib, 20 * gib, 32 * gib);

        assert_eq!(error.code(), GpuErrorCode::Memory);
        assert_eq!(
            error.to_string(),
            "[gpu.memory] allocating VMM arena backing: not enough GPU memory: requires 24.00 GiB, but 20.00 GiB is free of 32.00 GiB (short by 4.00 GiB); free GPU memory and retry"
        );
    }
}

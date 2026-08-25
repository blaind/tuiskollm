//! Host staging footprint reported by every materialized weight layout.

pub mod sealed {
    /// Restricts `MaterializedMemory` to this crate's admitted materialized layouts.
    pub trait Sealed {}
}

/// Host RAM owned by one materialized weight layout.
///
/// Sealed through `sealed::Sealed`, whose module is unreachable outside this crate: only the
/// admitted `Materialized*` layouts implement it, and no downstream crate can add a layout.
/// Inspection only — implementations never allocate, convert, or reorder a source word.
pub trait MaterializedMemory: sealed::Sealed {
    /// Bytes this layout allocates on the host heap.
    ///
    /// Excludes mmap-backed source views and borrowed source slices: those stay mapped from
    /// the checkpoint and are never staged into pinned host memory.
    fn host_bytes(&self) -> usize;
}

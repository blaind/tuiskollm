//! MTP draft-layer host layouts shared by the admitted MTP targets.

use crate::common::materialized::{MaterializedMemory, sealed};

/// Runtime-native fused BF16 MTP QKV plane in query/gate, key, value row order.
#[derive(Debug)]
pub struct MaterializedMtpQkv {
    /// Losslessly gathered little-endian BF16 weights `[rows, columns]`.
    pub weight_bf16: Vec<u8>,
    /// Fused query/gate, key, and value row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
}

impl sealed::Sealed for MaterializedMtpQkv {}

impl MaterializedMemory for MaterializedMtpQkv {
    fn host_bytes(&self) -> usize {
        self.weight_bf16.len()
    }
}

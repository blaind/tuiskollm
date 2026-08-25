//! MTP draft-layer host layouts shared by the admitted MTP targets.

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

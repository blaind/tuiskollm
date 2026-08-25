//! Independent residual-seam reference oracles.

use crate::oracles::codecs::{bf16_to_f32, f32_to_bf16};

/// Publishes the BF16 residual seam: widen both operands, add in `f32`, round back to BF16.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn residual_oracle(input: &[u16], branch: &[u16]) -> Vec<u16> {
    input
        .iter()
        .zip(branch)
        .map(|(&input, &branch)| f32_to_bf16(bf16_to_f32(input) + bf16_to_f32(branch)))
        .collect()
}

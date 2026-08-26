//! Checked layout arithmetic and source-word helpers shared by every target.

use crate::{EngineError, EngineResult};

pub(crate) fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

pub(crate) fn sum(name: &str, values: &[usize]) -> EngineResult<usize> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
    })
}

pub(crate) fn checked_product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

pub(crate) fn checked_sum(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

pub(crate) fn sum_products(name: &str, terms: &[(usize, usize)]) -> EngineResult<usize> {
    terms.iter().try_fold(0usize, |total, &(count, bytes)| {
        checked_sum(name, total, product(name, count, bytes)?)
    })
}

pub(crate) fn little_endian_words(bytes: &[u8]) -> EngineResult<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(EngineError::layout(
            "BF16 source plane has an odd byte length",
        ));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

pub(crate) const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

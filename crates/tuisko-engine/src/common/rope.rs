//! Exact text RoPE geometry shared by every resident target.

use crate::common::mtp::MAX_NATIVE_PREFILL_TOKENS;
use crate::{EngineError, EngineResult};

const ROTARY_DIM: usize = 64;
pub(crate) const ROTARY_PAIRS: usize = ROTARY_DIM / 2;
const ROPE_THETA: f64 = 10_000_000.0;

pub(crate) fn fill_contiguous_rope(
    first_position: usize,
    rows: usize,
    cosine: &mut [f32],
    sine: &mut [f32],
) -> EngineResult<usize> {
    if rows == 0 || rows > MAX_NATIVE_PREFILL_TOKENS {
        return Err(EngineError::route(format!(
            "resident MTP rotary rows {rows} are outside 1..={MAX_NATIVE_PREFILL_TOKENS}"
        )));
    }
    let values = rows
        .checked_mul(ROTARY_PAIRS)
        .ok_or_else(|| EngineError::generation("resident MTP rotary values overflow"))?;
    if cosine.len() < values || sine.len() < values {
        return Err(EngineError::layout(format!(
            "resident MTP rotary destinations have {}/{} values, expected at least {values}",
            cosine.len(),
            sine.len()
        )));
    }
    for row in 0..rows {
        let position = first_position
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| EngineError::generation("resident MTP position exceeds u32"))?;
        let (row_cosine, row_sine) = text_rope(position);
        let begin = row * ROTARY_PAIRS;
        cosine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_cosine);
        sine[begin..begin + ROTARY_PAIRS].copy_from_slice(&row_sine);
    }
    Ok(values)
}

pub(crate) fn text_rope(position: u32) -> ([f32; ROTARY_PAIRS], [f32; ROTARY_PAIRS]) {
    let mut cosine = [0.0f32; ROTARY_PAIRS];
    let mut sine = [0.0f32; ROTARY_PAIRS];
    for pair in 0..ROTARY_PAIRS {
        let frequency = ROPE_THETA.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
        let angle = f64::from(position) * frequency;
        cosine[pair] = angle.cos() as f32;
        sine[pair] = angle.sin() as f32;
    }
    (cosine, sine)
}

#[cfg(test)]
mod tests {
    use super::{ROTARY_PAIRS, text_rope};

    #[test]
    fn text_rope_uses_the_checkpoint_theta_and_64_wide_pairing() {
        let (zero_cos, zero_sin) = text_rope(0);
        assert_eq!(zero_cos, [1.0; ROTARY_PAIRS]);
        assert_eq!(zero_sin, [0.0; ROTARY_PAIRS]);

        let (cosine, sine) = text_rope(130);
        let frequency = 10_000_000.0f64.powf(-62.0 / 64.0);
        let angle = 130.0 * frequency;
        assert_eq!(cosine[0].to_bits(), (130.0f64.cos() as f32).to_bits());
        assert_eq!(sine[0].to_bits(), (130.0f64.sin() as f32).to_bits());
        assert_eq!(cosine[31].to_bits(), (angle.cos() as f32).to_bits());
        assert_eq!(sine[31].to_bits(), (angle.sin() as f32).to_bits());
    }
}

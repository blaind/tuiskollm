//! Independent rotary-embedding reference oracles.
//!
//! `pairs`, `rotary_dim`, and `theta` stay caller-supplied: they are per-suite shape contracts,
//! not harness defaults.

/// Rotary cosine/sine tables for `positions`, evaluated as
/// `theta^(-2 * pair / rotary_dim)` in `f64` and narrowed per entry.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn rope_tables(
    positions: &[u32],
    pairs: usize,
    rotary_dim: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; positions.len() * pairs];
    let mut sine = vec![0.0; positions.len() * pairs];
    for (row, &position) in positions.iter().enumerate() {
        for pair in 0..pairs {
            let frequency = theta.powf(-((2 * pair) as f64) / rotary_dim as f64);
            let (sin, cos) = (f64::from(position) * frequency).sin_cos();
            cosine[row * pairs + pair] = cos as f32;
            sine[row * pairs + pair] = sin as f32;
        }
    }

    (cosine, sine)
}

/// Rotary tables for `tokens` consecutive positions beginning at `first_position`.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn prefill_rope_tables(
    first_position: usize,
    tokens: usize,
    pairs: usize,
    rotary_dim: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; tokens * pairs];
    let mut sine = vec![0.0; tokens * pairs];
    for token in 0..tokens {
        for pair in 0..pairs {
            let frequency = theta.powf(-((2 * pair) as f64) / rotary_dim as f64);
            let angle = (first_position + token) as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[token * pairs + pair] = cos as f32;
            sine[token * pairs + pair] = sin as f32;
        }
    }

    (cosine, sine)
}

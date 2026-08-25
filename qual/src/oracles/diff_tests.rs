//! Value-identity proofs for every call site absorbed into [`crate::oracles`].
//!
//! `legacy` holds each distinct per-suite body exactly as it stood before extraction, with only
//! the site's error *type* replaced by `String` (the numeric path is transcribed character for
//! character). Every `#[test]` below sweeps one absorbed body against its centralized
//! replacement and requires bit-equal results, so deleting the local copies stays provable.

use super::attention::{prefill_rope_tables, rope_tables};
use super::codecs::{
    bf16_to_f32, decode_e2m1, decode_e4m3fn, decode_e4m3fn_f64, encode_e2m1, encode_e4m3fn,
    encode_e4m3fn_scale, f16_to_f32, f32_to_bf16, f32_to_f16,
};
use super::norm::residual_oracle;

/// Bit patterns that exercise every BF16 rounding boundary plus the exponent extremes.
fn bf16_rounding_sweep() -> impl Iterator<Item = f32> {
    (0u32..=0xffff).flat_map(|high| {
        [
            0x0000u32, 0x0001, 0x4000, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
        ]
        .into_iter()
        .map(move |low| f32::from_bits((high << 16) | low))
    })
}

/// A dense signed magnitude sweep spanning the E4M3FN and E2M1 representable ranges.
fn magnitude_sweep() -> impl Iterator<Item = f32> {
    (0u32..4096).flat_map(|step| {
        let magnitude = f32::from(step as u16) * 0.25 - 512.0;
        [magnitude, magnitude * 0.001_953_125, magnitude * 64.0]
    })
}

mod legacy {
    //! Verbatim pre-extraction bodies. Do not simplify: these are the authority the
    //! centralized oracles are diffed against.

    /// `dense_fp8_mlp_benchmark.rs`, `fp8_down_benchmark.rs`, `fp8_qkv_benchmark.rs`,
    /// `fp8_swiglu_benchmark.rs`, `mtp_bf16_mlp.rs`, `mtp_bf16_mlp_benchmark.rs`,
    /// `nvfp4_down_benchmark.rs`, `nvfp4_down_benchmark_sm120.rs`, `nvfp4_mlp_benchmark.rs`,
    /// `nvfp4_swiglu_benchmark.rs`, `qwen35_full_attention_layer_benchmark.rs`,
    /// `qwen35_nvfp4_mlp_benchmark.rs`, `residual_norm_benchmark.rs`,
    /// `fp8_projection_oracle.rs`, `residual_norm.rs`.
    pub(super) fn f32_to_bf16_rounded(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

        (rounded >> 16) as u16
    }

    /// `dense_fp8_gdn_layer_benchmark.rs`, `full_attention_layer_benchmark.rs`,
    /// `mtp_layer_benchmark.rs`, `qwen35_gdn_layer_benchmark.rs`,
    /// `qwen35_mtp_layer_benchmark.rs`, `qwen36_full_attention_layer_benchmark.rs`,
    /// `qwen36_gdn_moe_layer_benchmark.rs`, `qwen36_mtp_layer_benchmark.rs`,
    /// `resident_model.rs`, `resident_mtp_benchmark.rs`.
    pub(super) fn f32_to_bf16_inline(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    /// `nvfp4_down.rs`, `nvfp4_swiglu.rs`, `nvfp4_swiglu_benchmark_sm120.rs`,
    /// `nvfp4_down_sm120.rs`, `nvfp4_swiglu_sm120.rs`.
    pub(super) fn f32_to_bf16_tie(value: f32) -> u16 {
        let mut bits = value.to_bits();
        let tie = (bits >> 16) & 1;
        bits = bits.wrapping_add(0x7fff + tie);

        (bits >> 16) as u16
    }

    /// All seven `bf16_to_f32` sites.
    pub(super) fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// `fp8_projection_oracle.rs`.
    pub(super) fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = bits & 0x8000_0000;
        let exponent = bits & 0x7f80_0000;
        let mantissa = bits & 0x007f_ffff;
        if exponent == 0x7f80_0000 {
            let nan = if mantissa == 0 { 0 } else { 0x0200 };
            return ((sign >> 16) | 0x7c00 | nan | (mantissa >> 13)) as u16;
        }

        let half_sign = sign >> 16;
        let half_exponent = ((exponent >> 23) as i32) - 127 + 15;
        if half_exponent >= 0x1f {
            return (half_sign | 0x7c00) as u16;
        }
        if half_exponent <= 0 {
            if 14 - half_exponent > 24 {
                return half_sign as u16;
            }
            let mantissa = mantissa | 0x0080_0000;
            let mut half_mantissa = mantissa >> (14 - half_exponent);
            let round_bit = 1 << (13 - half_exponent);
            if mantissa & round_bit != 0 && mantissa & (3 * round_bit - 1) != 0 {
                half_mantissa += 1;
            }
            return (half_sign | half_mantissa) as u16;
        }

        let half_exponent = (half_exponent as u32) << 10;
        let half_mantissa = mantissa >> 13;
        let round_bit = 0x1000;
        if mantissa & round_bit != 0 && mantissa & (3 * round_bit - 1) != 0 {
            (half_sign | half_exponent | half_mantissa) as u16 + 1
        } else {
            (half_sign | half_exponent | half_mantissa) as u16
        }
    }

    /// `fp8_projection_oracle.rs`.
    pub(super) fn f16_to_f32(bits: u16) -> f32 {
        if bits & 0x7fff == 0 {
            return f32::from_bits(u32::from(bits) << 16);
        }
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = u32::from(bits & 0x7c00);
        let mantissa = u32::from(bits & 0x03ff);
        if exponent == 0x7c00 {
            return if mantissa == 0 {
                f32::from_bits(sign | 0x7f80_0000)
            } else {
                f32::from_bits(sign | 0x7fc0_0000 | (mantissa << 13))
            };
        }
        if exponent == 0 {
            let adjustment = mantissa.leading_zeros() - 22;
            let exponent = (127 - 15 - adjustment) << 23;
            let mantissa = (mantissa << (14 + adjustment)) & 0x007f_ffff;
            return f32::from_bits(sign | exponent | mantissa);
        }

        let exponent = (((exponent >> 10) as i32 - 15 + 127) as u32) << 23;
        f32::from_bits(sign | exponent | (mantissa << 13))
    }

    /// All six `decode_e2m1` sites.
    pub(super) fn decode_e2m1(code: u8) -> f32 {
        const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let magnitude = MAGNITUDES[(code & 7) as usize];

        if code & 8 == 0 { magnitude } else { -magnitude }
    }

    /// All eight `encode_e2m1` sites.
    pub(super) fn encode_e2m1(value: f32) -> u8 {
        let mut best = 0u8;
        let mut best_distance = f32::INFINITY;
        let candidates = if value.is_sign_negative() {
            8u8..16
        } else {
            0u8..8
        };

        for code in candidates {
            let distance = (value - decode_e2m1(code)).abs();
            if distance < best_distance || (distance == best_distance && code & 1 == 0) {
                best = code;
                best_distance = distance;
            }
        }

        best
    }

    /// `fp8_projection_oracle.rs`, `nvfp4_swiglu_sm120.rs` (hex masks).
    pub(super) fn decode_e4m3fn_hex(code: u8) -> Result<f32, String> {
        let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (code >> 3) & 0x0f;
        let fraction = code & 0x07;
        let magnitude = match (exponent, fraction) {
            (0, 0) => 0.0,
            (0, fraction) => f32::from(fraction) * 2.0f32.powi(-9),
            (15, 7) => {
                return Err("oracle encountered an E4M3FN NaN code".to_string());
            }
            (exponent, fraction) => {
                (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
            }
        };

        Ok(sign * magnitude)
    }

    /// `nvfp4_down.rs`, `nvfp4_down_sm120.rs`, `nvfp4_mlp.rs`, `nvfp4_swiglu.rs`
    /// (decimal masks).
    pub(super) fn decode_e4m3fn_decimal(word: u8) -> Result<f32, String> {
        let sign = if word & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (word >> 3) & 15;
        let fraction = word & 7;
        let magnitude = match (exponent, fraction) {
            (0, 0) => 0.0,
            (0, fraction) => f32::from(fraction) * 2.0f32.powi(-9),
            (15, 7) => {
                return Err("oracle encountered an E4M3FN NaN".to_string());
            }
            (exponent, fraction) => {
                (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
            }
        };

        Ok(sign * magnitude)
    }

    /// `long_context_paged_gqa.rs`, `paged_gqa.rs`, `paged_gqa_macro_prefill.rs`,
    /// `paged_gqa_partitioned_prefill.rs`, `paged_gqa_prefill.rs`.
    pub(super) fn decode_e4m3_f64(code: u8) -> f64 {
        let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (code >> 3) & 0x0f;
        let fraction = code & 0x07;
        let magnitude = match (exponent, fraction) {
            (0, 0) => 0.0,
            (0, fraction) => f64::from(fraction) * 2.0f64.powi(-9),
            (15, 7) => f64::NAN,
            (exponent, fraction) => {
                (1.0 + f64::from(fraction) / 8.0) * 2.0f64.powi(i32::from(exponent) - 7)
            }
        };
        sign * magnitude
    }

    const FP8_MAX: f32 = 448.0;

    /// `fp8_projection_oracle.rs` — the signed, clamped encoder.
    pub(super) fn encode_e4m3fn(value: f32) -> Result<u8, String> {
        if !value.is_finite() {
            return Err("oracle E4M3 input is not finite".to_string());
        }
        if value == 0.0 {
            return Ok(if value.is_sign_negative() { 0x80 } else { 0x00 });
        }

        let value = value.clamp(-FP8_MAX, FP8_MAX);
        let mut best = 0u8;
        let mut best_distance = f32::INFINITY;
        for code in 0u8..=u8::MAX {
            if matches!(code, 0x7f | 0xff) {
                continue;
            }
            let represented = decode_e4m3fn_hex(code).expect("NaN codes were excluded");
            if represented == 0.0 && represented.is_sign_negative() != value.is_sign_negative() {
                continue;
            }
            let distance = (value - represented).abs();
            if distance < best_distance || (distance == best_distance && code & 1 == 0) {
                best = code;
                best_distance = distance;
            }
        }

        Ok(best)
    }

    /// `nvfp4_down_sm120.rs`, `nvfp4_mlp.rs`, `nvfp4_swiglu_sm120.rs`,
    /// `qwen35_nvfp4_attention_output.rs`, `qwen35_nvfp4_down.rs`,
    /// `qwen35_nvfp4_gdn_input.rs`, `qwen35_nvfp4_qkv.rs` — the non-negative scale encoder.
    pub(super) fn encode_e4m3fn_scale(value: f32) -> Result<u8, String> {
        if !value.is_finite() || value < 0.0 {
            return Err("oracle E4M3 scale is not finite and non-negative".to_string());
        }

        let mut best = 0u8;
        let mut best_distance = f32::INFINITY;
        for code in 0u8..=0x7e {
            let represented = decode_e4m3fn_decimal(code)?;
            let distance = (value - represented).abs();
            if distance < best_distance || (distance == best_distance && code & 1 == 0) {
                best = code;
                best_distance = distance;
            }
        }

        Ok(best)
    }

    /// `dense_fp8_gdn_layer.rs`, `full_attention_layer.rs`, `qwen35_full_attention_layer.rs`,
    /// `qwen35_gdn_layer.rs`, `qwen36_full_attention_layer.rs`, `qwen36_gdn_moe_layer.rs`.
    pub(super) fn residual_oracle(input: &[u16], branch: &[u16]) -> Vec<u16> {
        input
            .iter()
            .zip(branch)
            .map(|(&input, &branch)| f32_to_bf16_rounded(bf16_to_f32(input) + bf16_to_f32(branch)))
            .collect()
    }

    /// `mtp_layer.rs`, `qwen35_mtp_layer.rs`, `qwen36_mtp_layer.rs`.
    pub(super) fn residual_oracle_mtp(left: &[u16], right: &[u16]) -> Vec<u16> {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| f32_to_bf16_rounded(bf16_to_f32(left) + bf16_to_f32(right)))
            .collect()
    }

    /// Every absorbed rotary site pins `ROTARY_PAIRS = 32` and a rotary dimension of `64`.
    pub(super) const ROTARY_PAIRS: usize = 32;
    pub(super) const ROTARY_DIM: usize = 64;

    /// `mtp_layer.rs`, `qwen35_mtp_layer.rs`, `qwen36_mtp_layer.rs`,
    /// `mtp_prompt_prime.rs`, `target_mtp_verify.rs`.
    pub(super) fn rope_named_dim(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let mut cosine = vec![0.0; positions.len() * ROTARY_PAIRS];
        let mut sine = vec![0.0; positions.len() * ROTARY_PAIRS];
        for (row, &position) in positions.iter().enumerate() {
            for pair in 0..ROTARY_PAIRS {
                let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / ROTARY_DIM as f64);
                let angle = f64::from(position) * frequency;
                let (sin, cos) = angle.sin_cos();
                cosine[row * ROTARY_PAIRS + pair] = cos as f32;
                sine[row * ROTARY_PAIRS + pair] = sin as f32;
            }
        }
        (cosine, sine)
    }

    /// `mtp_prompt_prime_benchmark.rs`, `resident_mtp.rs`, `target_mtp_verify_benchmark.rs`.
    pub(super) fn rope_literal_dim(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let mut cosine = vec![0.0; positions.len() * ROTARY_PAIRS];
        let mut sine = vec![0.0; positions.len() * ROTARY_PAIRS];
        for (row, &position) in positions.iter().enumerate() {
            for pair in 0..ROTARY_PAIRS {
                let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
                let (sin, cos) = (f64::from(position) * frequency).sin_cos();
                cosine[row * ROTARY_PAIRS + pair] = cos as f32;
                sine[row * ROTARY_PAIRS + pair] = sin as f32;
            }
        }
        (cosine, sine)
    }

    /// `full_attention_layer_benchmark.rs`, `qwen35_full_attention_layer_benchmark.rs`,
    /// `qwen35_resident_model_benchmark.rs`, `qwen36_full_attention_layer_benchmark.rs`,
    /// `qwen36_resident_model_benchmark.rs`, `full_attention_layer.rs`.
    pub(super) fn prefill_rope(tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cosine = vec![0.0; tokens * ROTARY_PAIRS];
        let mut sine = vec![0.0; tokens * ROTARY_PAIRS];
        for token in 0..tokens {
            for pair in 0..ROTARY_PAIRS {
                let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
                let angle = token as f64 * frequency;
                let (sin, cos) = angle.sin_cos();
                cosine[token * ROTARY_PAIRS + pair] = cos as f32;
                sine[token * ROTARY_PAIRS + pair] = sin as f32;
            }
        }
        (cosine, sine)
    }

    /// `resident_model.rs`.
    pub(super) fn prefill_rope_at(first_position: usize, tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cosine = vec![0.0; tokens * ROTARY_PAIRS];
        let mut sine = vec![0.0; tokens * ROTARY_PAIRS];
        for token in 0..tokens {
            for pair in 0..ROTARY_PAIRS {
                let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
                let angle = (first_position + token) as f64 * frequency;
                let (sin, cos) = angle.sin_cos();
                cosine[token * ROTARY_PAIRS + pair] = cos as f32;
                sine[token * ROTARY_PAIRS + pair] = sin as f32;
            }
        }
        (cosine, sine)
    }
}

#[test]
fn f32_to_bf16_matches_all_thirty_absorbed_sites() {
    for value in bf16_rounding_sweep() {
        let central = f32_to_bf16(value);
        assert_eq!(central, legacy::f32_to_bf16_rounded(value));
        assert_eq!(central, legacy::f32_to_bf16_inline(value));
        assert_eq!(central, legacy::f32_to_bf16_tie(value));
    }
}

#[test]
fn bf16_to_f32_matches_all_seven_absorbed_sites() {
    for bits in 0u16..=u16::MAX {
        assert_eq!(
            bf16_to_f32(bits).to_bits(),
            legacy::bf16_to_f32(bits).to_bits()
        );
    }
}

#[test]
fn half_codecs_match_the_moved_fp8_projection_oracle_bodies() {
    for bits in 0u16..=u16::MAX {
        assert_eq!(
            f16_to_f32(bits).to_bits(),
            legacy::f16_to_f32(bits).to_bits()
        );
    }
    for value in bf16_rounding_sweep() {
        assert_eq!(f32_to_f16(value), legacy::f32_to_f16(value));
    }
}

#[test]
fn e2m1_codecs_match_all_absorbed_sites() {
    for code in 0u8..=u8::MAX {
        assert_eq!(
            decode_e2m1(code).to_bits(),
            legacy::decode_e2m1(code).to_bits()
        );
    }
    for value in magnitude_sweep() {
        assert_eq!(encode_e2m1(value), legacy::encode_e2m1(value));
    }
    for value in [0.0f32, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE] {
        assert_eq!(encode_e2m1(value), legacy::encode_e2m1(value));
    }
}

#[test]
fn e4m3fn_decode_matches_all_absorbed_sites() {
    for code in 0u8..=u8::MAX {
        let central = decode_e4m3fn(code);
        let hex = legacy::decode_e4m3fn_hex(code);
        let decimal = legacy::decode_e4m3fn_decimal(code);
        match central {
            Some(value) => {
                assert_eq!(value.to_bits(), hex.unwrap().to_bits());
                assert_eq!(value.to_bits(), decimal.unwrap().to_bits());
            }
            None => {
                assert!(hex.is_err() && decimal.is_err());
            }
        }

        let f64_central = decode_e4m3fn_f64(code);
        let f64_legacy = legacy::decode_e4m3_f64(code);
        assert_eq!(f64_central.is_nan(), f64_legacy.is_nan());
        if !f64_central.is_nan() {
            assert_eq!(f64_central.to_bits(), f64_legacy.to_bits());
        }
    }
}

#[test]
fn e4m3fn_encoders_match_all_absorbed_sites() {
    for value in magnitude_sweep() {
        assert_eq!(encode_e4m3fn(value), legacy::encode_e4m3fn(value).ok());
        assert_eq!(
            encode_e4m3fn_scale(value),
            legacy::encode_e4m3fn_scale(value).ok()
        );
    }
    for value in [
        0.0f32,
        -0.0,
        448.0,
        449.0,
        1.0e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        assert_eq!(encode_e4m3fn(value), legacy::encode_e4m3fn(value).ok());
        assert_eq!(
            encode_e4m3fn_scale(value),
            legacy::encode_e4m3fn_scale(value).ok()
        );
    }
}

#[test]
fn residual_seam_matches_all_nine_absorbed_sites() {
    let input = (0u16..=u16::MAX).step_by(7).collect::<Vec<_>>();
    let branch = input.iter().rev().copied().collect::<Vec<_>>();
    let central = residual_oracle(&input, &branch);
    assert_eq!(central, legacy::residual_oracle(&input, &branch));
    assert_eq!(central, legacy::residual_oracle_mtp(&input, &branch));
}

#[test]
fn rotary_tables_match_all_absorbed_sites() {
    let positions = (0u32..320).step_by(7).collect::<Vec<_>>();
    let (cosine, sine) = rope_tables(
        &positions,
        legacy::ROTARY_PAIRS,
        legacy::ROTARY_DIM,
        10_000_000.0,
    );
    for legacy_tables in [
        legacy::rope_named_dim(&positions),
        legacy::rope_literal_dim(&positions),
    ] {
        assert_eq!(bits32(&cosine), bits32(&legacy_tables.0));
        assert_eq!(bits32(&sine), bits32(&legacy_tables.1));
    }

    for tokens in [0usize, 1, 8, 65, 130] {
        let (cosine, sine) = prefill_rope_tables(
            0,
            tokens,
            legacy::ROTARY_PAIRS,
            legacy::ROTARY_DIM,
            10_000_000.0,
        );
        let expected = legacy::prefill_rope(tokens);
        assert_eq!(bits32(&cosine), bits32(&expected.0));
        assert_eq!(bits32(&sine), bits32(&expected.1));

        for first in [0usize, 1, 63, 4096] {
            let (cosine, sine) = prefill_rope_tables(
                first,
                tokens,
                legacy::ROTARY_PAIRS,
                legacy::ROTARY_DIM,
                10_000_000.0,
            );
            let expected = legacy::prefill_rope_at(first, tokens);
            assert_eq!(bits32(&cosine), bits32(&expected.0));
            assert_eq!(bits32(&sine), bits32(&expected.1));
        }
    }
}

fn bits32(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

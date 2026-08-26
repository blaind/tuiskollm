//! Independent host codecs for the represented BF16, FP16, E4M3FN, and E2M1 word formats.

/// Largest finite E4M3FN magnitude.
pub(crate) const FP8_MAX: f32 = 448.0;

/// Rounds one `f32` to its BF16 word with round-to-nearest-even on the discarded half.
pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

/// Widens one BF16 word to `f32`.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Rounds one `f32` to its FP16 word, preserving infinities, NaN payloads, and subnormals.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn f32_to_f16(value: f32) -> u16 {
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

/// Widens one FP16 word to `f32`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
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

/// Decodes one E2M1 nibble to its represented value.
pub(crate) fn decode_e2m1(code: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(code & 7) as usize];

    if code & 8 == 0 { magnitude } else { -magnitude }
}

/// Selects the E2M1 nibble nearest `value`, breaking ties toward the even code.
///
/// The sign nibble is chosen from the sign bit alone, so `-0.0` encodes to `0x8`.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn encode_e2m1(value: f32) -> u8 {
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

/// Decodes one E4M3FN word; `None` is the NaN encoding (`0x7f` / `0xff`).
pub(crate) fn decode_e4m3fn(code: u8) -> Option<f32> {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let fraction = code & 0x07;
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0.0,
        (0, fraction) => f32::from(fraction) * 2.0f32.powi(-9),
        (15, 7) => return None,
        (exponent, fraction) => {
            (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
        }
    };

    Some(sign * magnitude)
}

/// Decodes one E4M3FN word in `f64`; the NaN encoding yields `f64::NAN`.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn decode_e4m3fn_f64(code: u8) -> f64 {
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

/// Selects the E4M3FN word nearest `value` over the signed finite range, ties toward the even
/// code; `None` reports a non-finite input. The input is clamped to `±FP8_MAX` and signed zero
/// is preserved.
#[cfg_attr(not(any(feature = "device", feature = "sm89")), allow(dead_code))]
pub(crate) fn encode_e4m3fn(value: f32) -> Option<u8> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(if value.is_sign_negative() { 0x80 } else { 0x00 });
    }

    let value = value.clamp(-FP8_MAX, FP8_MAX);
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=u8::MAX {
        if matches!(code, 0x7f | 0xff) {
            continue;
        }
        let represented = decode_e4m3fn(code).expect("NaN codes were excluded");
        if represented == 0.0 && represented.is_sign_negative() != value.is_sign_negative() {
            continue;
        }
        let distance = (value - represented).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    Some(best)
}

/// Selects the E4M3FN scale word nearest `value` over the non-negative codes `0x00..=0x7e`,
/// ties toward the even code; `None` reports a non-finite or negative input. Unlike
/// [`encode_e4m3fn`] the search is unclamped, so inputs above `FP8_MAX` saturate to `0x7e`.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn encode_e4m3fn_scale(value: f32) -> Option<u8> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=0x7e {
        let represented = decode_e4m3fn(code).expect("codes 0x00..=0x7e are finite");
        let distance = (value - represented).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    Some(best)
}

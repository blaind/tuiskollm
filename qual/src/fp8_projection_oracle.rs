//! Independent host codec and represented-value fixtures for FP8 projections.

pub(crate) const BYTE_SENTINEL: u8 = 0xa5;
pub(crate) const BF16_SENTINEL: u16 = 0xa5a5;
pub(crate) const F32_SENTINEL_BITS: u32 = 0xa5a5a5a5;
pub(crate) const WEIGHT_CODES: [u8; 4] = [0x38, 0xb0, 0x30, 0x28];
pub(crate) const WEIGHT_VALUES: [f32; 4] = [1.0, -0.5, 0.5, 0.25];
pub(crate) const SCALE_VALUES: [f32; 4] = [1.0, 0.5, 0.25, 2.0];
const FP8_MAX: f32 = 448.0;

pub(crate) struct TokenOracle {
    pub(crate) codes: Vec<u8>,
    pub(crate) scale: f32,
    pub(crate) represented_sum: f64,
}

pub(crate) struct Observed {
    pub(crate) codes: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) output: Vec<u16>,
}

pub(crate) fn quantize_oracle(input: &[u16]) -> Result<TokenOracle, String> {
    let represented = input
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let maximum = represented
        .iter()
        .fold(0.0f32, |current, value| current.max(value.abs()));
    let scale = if maximum == 0.0 {
        1.0
    } else {
        maximum / FP8_MAX
    };
    let codes = represented
        .iter()
        .map(|&value| encode_e4m3fn(value / scale))
        .collect::<Result<Vec<_>, _>>()?;
    let represented_sum = codes
        .iter()
        .map(|&code| f64::from(decode_e4m3fn(code).expect("encoder emitted finite E4M3")))
        .sum();

    Ok(TokenOracle {
        codes,
        scale,
        represented_sum,
    })
}

pub(crate) fn encode_e4m3fn(value: f32) -> Result<u8, String> {
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

    Ok(best)
}

pub(crate) fn decode_e4m3fn(code: u8) -> Result<f32, String> {
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

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

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

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
pub(crate) fn verify_host_codecs() -> Result<(), String> {
    if f32_to_bf16(1.0 + 0.00390625) != 0x3f80
        || f32_to_bf16(bf16_to_f32(0x3f81) + 0.00390625) != 0x3f82
        || f32_to_f16(1.0 + 2.0f32.powi(-11)) != 0x3c00
        || f16_to_f32(0x3c01) != 1.0 + 2.0f32.powi(-10)
        || encode_e4m3fn(1.0)? != 0x38
        || encode_e4m3fn(-0.5)? != 0xb0
        || decode_e4m3fn(0x7e)? != 448.0
        || decode_e4m3fn(0x7f).is_ok()
    {
        return Err("host BF16/E4M3 codec contract changed".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verify_host_codecs;

    #[test]
    fn host_codecs_pin_bf16_and_e4m3_rounding() {
        verify_host_codecs().unwrap();
    }
}

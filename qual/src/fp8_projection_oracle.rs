//! Independent host codec and represented-value fixtures for FP8 projections.

use crate::oracles::codecs;
pub(crate) use crate::oracles::codecs::{bf16_to_f32, f32_to_bf16};
#[cfg(feature = "device")]
pub(crate) use crate::oracles::codecs::{f16_to_f32, f32_to_f16};

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
    codecs::encode_e4m3fn(value).ok_or_else(|| "oracle E4M3 input is not finite".to_string())
}

pub(crate) fn decode_e4m3fn(code: u8) -> Result<f32, String> {
    codecs::decode_e4m3fn(code).ok_or_else(|| "oracle encountered an E4M3FN NaN code".to_string())
}

#[cfg(test)]
pub(crate) fn verify_host_codecs() -> Result<(), String> {
    if f32_to_bf16(1.0 + 0.00390625) != 0x3f80
        || f32_to_bf16(bf16_to_f32(0x3f81) + 0.00390625) != 0x3f82
        || codecs::f32_to_f16(1.0 + 2.0f32.powi(-11)) != 0x3c00
        || codecs::f16_to_f32(0x3c01) != 1.0 + 2.0f32.powi(-10)
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

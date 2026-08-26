//! Leaf SIMT math shared as source by the pre-Blackwell fallback kernel crates.
//!
//! Each consuming `tuisko-kernels-sm*` crate compiles these helpers into its own
//! per-architecture PTX; nothing here is a shared compiled artifact.

#![no_std]

/// Decodes one E4M3 group-scale code, including its subnormal range.
#[inline(always)]
pub fn e4m3_to_f32(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;

    if exponent == 0 {
        fraction as f32 * (1.0 / 512.0)
    } else {
        f32::from_bits(((exponent as u32 + 120) << 23) | ((fraction as u32) << 20))
    }
}

#[cfg(test)]
mod tests {
    use super::e4m3_to_f32;

    #[test]
    fn decodes_the_subnormal_and_normal_scale_ranges() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0);
        assert_eq!(e4m3_to_f32(0x07), 7.0 / 512.0);
        assert_eq!(e4m3_to_f32(0x08), 0.015_625);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x78), 256.0);

        // Carried verbatim from the three call sites: the sign bit is not part
        // of a group scale, and `0x7f` decodes as a finite 480.0 rather than the
        // OCP E4M3 NaN encoding.
        assert_eq!(e4m3_to_f32(0x7f), 480.0);
        assert_eq!(e4m3_to_f32(0xb8), e4m3_to_f32(0x38));
    }
}

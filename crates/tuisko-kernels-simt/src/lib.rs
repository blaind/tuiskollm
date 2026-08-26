//! Leaf SIMT math and host-launcher scaffolding shared as source by the
//! pre-Blackwell fallback kernel crates.
//!
//! Each consuming `tuisko-kernels-sm*` crate compiles these helpers into its own
//! per-architecture PTX; nothing here is a shared compiled artifact. The
//! `routes` macros expand into host code only and never reach device code.

#![no_std]

mod routes;

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

/// Decodes one signed E2M1 weight code to its represented value.
#[inline(always)]
pub fn e2m1_to_f32(code: u8) -> f32 {
    let magnitude = match code & 7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };

    if code & 8 == 0 { magnitude } else { -magnitude }
}

/// Decodes one packed E2M1 byte as its low then high represented values.
#[inline(always)]
pub fn e2m1x2_to_f32(packed: u8) -> (f32, f32) {
    (e2m1_to_f32(packed & 15), e2m1_to_f32(packed >> 4))
}

/// Rounds one FP32 value to BF16 bits, half-to-even on the retained mantissa.
///
/// The fallback operators only ever publish finite accumulator sums, so this
/// carries no NaN quieting: `0x7fff + ((bits >> 16) & 1)` is the exact
/// round-to-nearest-even carry the three call sites shared verbatim.
#[inline(always)]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{e2m1_to_f32, e2m1x2_to_f32, e4m3_to_f32, f32_to_bf16};

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

    #[test]
    fn decodes_every_signed_e2m1_code() {
        let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, magnitude) in magnitudes.into_iter().enumerate() {
            let code = code as u8;
            assert_eq!(e2m1_to_f32(code), magnitude);
            assert_eq!(e2m1_to_f32(code | 8), -magnitude);
        }

        // Only the low nibble selects the code: the three call sites feed both
        // nibbles of a packed byte through the same decoder.
        assert_eq!(e2m1_to_f32(0xf0), 0.0);
    }

    #[test]
    fn unpacks_the_low_then_high_nibble() {
        assert_eq!(e2m1x2_to_f32(0x00), (0.0, 0.0));
        assert_eq!(e2m1x2_to_f32(0x71), (0.5, 6.0));
        assert_eq!(e2m1x2_to_f32(0x9f), (-6.0, -0.5));
    }

    #[test]
    fn rounds_bf16_half_to_even_on_the_retained_mantissa() {
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-1.0), 0xbf80);

        // Exact tie on the retained mantissa rounds to the even neighbour in
        // both directions; a hair above the tie always rounds up.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    }
}

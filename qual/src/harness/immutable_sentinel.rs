//! Sentinel patterns and the scans that prove inactive tails and read-only sources unchanged.
//!
//! Every scan reports the index of the first violation and formats nothing: the sentinel byte and
//! the failure message remain per-suite contracts (Part V §3.F).

/// One device fill byte and the words that fill produces.
///
/// `DeviceArena::fill` writes a byte, so a `u16` observable reads that byte twice and an `f32`
/// observable reads it four times. The byte stays caller-supplied: `0xa5` and `0x7d` are both
/// live per-suite patterns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SentinelPattern {
    byte: u8,
}

impl SentinelPattern {
    /// Describes the pattern a device byte fill of `byte` leaves behind.
    pub(crate) const fn new(byte: u8) -> Self {
        Self { byte }
    }

    /// The filled byte.
    pub(crate) const fn byte(self) -> u8 {
        self.byte
    }

    /// The 16-bit word the fill produces.
    pub(crate) const fn half(self) -> u16 {
        u16::from_ne_bytes([self.byte, self.byte])
    }

    /// The 32-bit pattern the fill produces.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    pub(crate) const fn word_bits(self) -> u32 {
        u32::from_ne_bytes([self.byte, self.byte, self.byte, self.byte])
    }
}

/// Index of the first value that is not `sentinel`.
pub(crate) fn first_non_sentinel<T: Copy + PartialEq>(values: &[T], sentinel: T) -> Option<usize> {
    values.iter().position(|&value| value != sentinel)
}

/// Index of the first `f32` whose bits are not `sentinel_bits`.
///
/// Sentinel words are not numbers: they are compared by bit pattern, never by `==`.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn first_non_sentinel_f32(values: &[f32], sentinel_bits: u32) -> Option<usize> {
    values
        .iter()
        .position(|value| value.to_bits() != sentinel_bits)
}

/// Index of the first observed value differing from its source.
///
/// The scan stops at the shorter slice, preserving the per-suite `zip` idiom it replaces.
pub(crate) fn first_difference<T: Copy + PartialEq>(observed: &[T], source: &[T]) -> Option<usize> {
    observed
        .iter()
        .zip(source)
        .position(|(observed, source)| observed != source)
}

/// Index of the first `f32` whose bits differ from the source's, comparing NaN payloads.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn first_bit_difference_f32(observed: &[f32], source: &[f32]) -> Option<usize> {
    observed
        .iter()
        .zip(source)
        .position(|(observed, source)| observed.to_bits() != source.to_bits())
}

#[cfg(test)]
mod tests {
    use super::{
        SentinelPattern, first_bit_difference_f32, first_difference, first_non_sentinel,
        first_non_sentinel_f32,
    };

    #[test]
    fn a_byte_fill_widens_to_the_suite_sentinel_words() {
        let inactive = SentinelPattern::new(0xa5);
        assert_eq!(inactive.byte(), 0xa5);
        assert_eq!(inactive.half(), 0xa5a5);
        assert_eq!(inactive.word_bits(), 0xa5a5_a5a5);

        let resident = SentinelPattern::new(0x7d);
        assert_eq!(resident.byte(), 0x7d);
        assert_eq!(resident.half(), 0x7d7d);
        assert_eq!(resident.word_bits(), 0x7d7d_7d7d);
    }

    #[test]
    fn sentinel_scans_report_the_first_written_index() {
        let sentinel = SentinelPattern::new(0xa5);
        let untouched = [sentinel.half(); 4];
        assert_eq!(first_non_sentinel(&untouched, sentinel.half()), None);

        let written = [sentinel.half(), sentinel.half(), 0x3f80, sentinel.half()];
        assert_eq!(first_non_sentinel(&written, sentinel.half()), Some(2));
        assert_eq!(first_non_sentinel::<u16>(&[], sentinel.half()), None);
    }

    #[test]
    fn f32_sentinel_scans_compare_bits_not_values() {
        let bits = SentinelPattern::new(0xa5).word_bits();
        let sentinel = f32::from_bits(bits);
        assert_eq!(first_non_sentinel_f32(&[sentinel; 3], bits), None);
        assert_eq!(
            first_non_sentinel_f32(&[sentinel, 0.0, sentinel], bits),
            Some(1)
        );

        // Two NaN payloads compare equal under `==` only through their bits.
        let quiet = f32::from_bits(0x7fc0_0000);
        let signalling = f32::from_bits(0x7fc0_0001);
        assert_eq!(
            first_non_sentinel_f32(&[quiet], signalling.to_bits()),
            Some(0)
        );
    }

    #[test]
    fn immutability_scans_stop_at_the_shorter_slice() {
        assert_eq!(first_difference(&[1_u8, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_difference(&[1_u8, 2, 3], &[1, 7, 3]), Some(1));
        assert_eq!(first_difference(&[1_u8, 2, 3, 4], &[1, 2]), None);
        assert_eq!(first_difference(&[1_u8, 2], &[1, 2, 3, 4]), None);
    }

    #[test]
    fn f32_immutability_scans_reject_equal_values_with_different_bits() {
        assert_eq!(first_bit_difference_f32(&[1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(first_bit_difference_f32(&[0.0], &[-0.0]), Some(0));
        assert_eq!(first_bit_difference_f32(&[f32::NAN], &[f32::NAN]), None);
    }
}

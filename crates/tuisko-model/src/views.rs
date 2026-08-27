//! Shape- and dtype-checked zero-copy views over admitted tensor bytes.

use crate::{CheckpointError, CheckpointResult, DType, TensorView};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Source element admitted by a typed view.
///
/// Sealed: the five marker types below are the complete admitted element set, and no other
/// crate can add a sixth. Admitting a new source element is a checkpoint-contract change,
/// never a downstream extension.
pub trait TensorElement: sealed::Sealed + Copy + 'static {
    /// Exact source dtype every view of this element validates against.
    const DTYPE: DType;
}

/// Little-endian BF16 source words.
#[derive(Clone, Copy, Debug)]
pub struct Bf16;

/// Little-endian F32 source values.
#[derive(Clone, Copy, Debug)]
pub struct F32;

/// FP8 E4M3 source codes.
#[derive(Clone, Copy, Debug)]
pub struct Fp8E4M3;

/// Little-endian signed 64-bit source words.
#[derive(Clone, Copy, Debug)]
pub struct I64;

/// Raw U8 source elements.
#[derive(Clone, Copy, Debug)]
pub struct U8;

impl sealed::Sealed for Bf16 {}
impl sealed::Sealed for F32 {}
impl sealed::Sealed for Fp8E4M3 {}
impl sealed::Sealed for I64 {}
impl sealed::Sealed for U8 {}

impl TensorElement for Bf16 {
    const DTYPE: DType = DType::Bf16;
}

impl TensorElement for F32 {
    const DTYPE: DType = DType::F32;
}

impl TensorElement for Fp8E4M3 {
    const DTYPE: DType = DType::Fp8E4M3;
}

impl TensorElement for I64 {
    const DTYPE: DType = DType::I64;
}

impl TensorElement for U8 {
    const DTYPE: DType = DType::U8;
}

/// Validated zero-copy view of admitted source bytes for one element type.
#[derive(Clone, Copy, Debug)]
pub struct TypedView<'a, E: TensorElement, const RANK: usize> {
    name: &'a str,
    shape: [u64; RANK],
    bytes: &'a [u8],
    element: PhantomData<E>,
}

/// Validated zero-copy view of little-endian BF16 source words.
pub type Bf16View<'a, const RANK: usize> = TypedView<'a, Bf16, RANK>;

/// Validated zero-copy view of little-endian F32 source values.
pub type F32View<'a, const RANK: usize> = TypedView<'a, F32, RANK>;

/// Validated zero-copy view of FP8 E4M3 source codes.
pub type Fp8E4M3View<'a, const RANK: usize> = TypedView<'a, Fp8E4M3, RANK>;

/// Validated zero-copy view of little-endian signed 64-bit source words.
pub type I64View<'a, const RANK: usize> = TypedView<'a, I64, RANK>;

/// Validated zero-copy view of raw U8 source elements.
pub type U8View<'a, const RANK: usize> = TypedView<'a, U8, RANK>;

impl<'a, E: TensorElement, const RANK: usize> TypedView<'a, E, RANK> {
    /// Returns the source tensor name.
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// Returns the validated tensor shape.
    pub fn shape(&self) -> &[u64; RANK] {
        &self.shape
    }

    /// Returns the number of represented source elements.
    pub fn len(&self) -> usize {
        self.bytes.len() / E::DTYPE.byte_width() as usize
    }

    /// Returns whether the view contains no source elements.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Validates one public tensor descriptor as an exact view of this element.
    ///
    /// Per-element `bind` constructors keep their own visibility; this is the shared body.
    fn bind_validated(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        let expected_dtype = E::DTYPE;

        if tensor.dtype != expected_dtype {
            return Err(CheckpointError::tensor(format!(
                "tensor `{}` has dtype `{}`, expected `{expected_dtype}`",
                tensor.name, tensor.dtype
            )));
        }

        if tensor.shape != expected_shape {
            return Err(CheckpointError::tensor(format!(
                "tensor `{}` has shape {:?}, expected {expected_shape:?}",
                tensor.name, tensor.shape
            )));
        }

        let elements = expected_shape.iter().try_fold(1u64, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                CheckpointError::tensor(format!("tensor `{}` shape overflows", tensor.name))
            })
        })?;
        let expected_bytes = elements
            .checked_mul(expected_dtype.byte_width())
            .ok_or_else(|| {
                CheckpointError::tensor(format!("tensor `{}` byte length overflows", tensor.name))
            })?;
        let actual_bytes = u64::try_from(tensor.bytes.len()).map_err(|_| {
            CheckpointError::tensor(format!(
                "tensor `{}` is too large for this host",
                tensor.name
            ))
        })?;

        if actual_bytes != expected_bytes {
            return Err(CheckpointError::tensor(format!(
                "tensor `{}` has {actual_bytes} bytes, expected {expected_bytes}",
                tensor.name
            )));
        }

        Ok(Self {
            name: tensor.name,
            shape: expected_shape,
            bytes: tensor.bytes,
            element: PhantomData,
        })
    }
}

impl<'a, const RANK: usize> Bf16View<'a, RANK> {
    /// Validates one public tensor descriptor as an exact BF16 view.
    pub fn bind(tensor: TensorView<'a>, expected_shape: [u64; RANK]) -> CheckpointResult<Self> {
        Self::bind_validated(tensor, expected_shape)
    }

    /// Returns the unmodified little-endian BF16 bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the source BF16 word at `index`.
    pub fn word(&self, index: usize) -> Option<u16> {
        let begin = index.checked_mul(2)?;
        let end = begin.checked_add(2)?;
        let bytes: [u8; 2] = self.bytes.get(begin..end)?.try_into().ok()?;

        Some(u16::from_le_bytes(bytes))
    }

    /// Iterates over the source BF16 words.
    pub fn words(&self) -> impl DoubleEndedIterator<Item = u16> + ExactSizeIterator + '_ {
        self.bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
    }
}

impl<'a, const RANK: usize> F32View<'a, RANK> {
    pub(crate) fn bind(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        Self::bind_validated(tensor, expected_shape)
    }

    /// Returns the unmodified little-endian F32 bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the source F32 bits at `index`.
    pub fn bits(&self, index: usize) -> Option<u32> {
        let begin = index.checked_mul(4)?;
        let end = begin.checked_add(4)?;
        let bytes: [u8; 4] = self.bytes.get(begin..end)?.try_into().ok()?;

        Some(u32::from_le_bytes(bytes))
    }

    /// Returns the F32 value at `index`.
    pub fn value(&self, index: usize) -> Option<f32> {
        self.bits(index).map(f32::from_bits)
    }
}

impl<'a, const RANK: usize> Fp8E4M3View<'a, RANK> {
    /// Binds a validated view from an admitted tensor descriptor.
    pub fn bind(tensor: TensorView<'a>, expected_shape: [u64; RANK]) -> CheckpointResult<Self> {
        Self::bind_validated(tensor, expected_shape)
    }

    /// Returns the unmodified FP8 E4M3 source codes.
    pub fn codes(&self) -> &'a [u8] {
        self.bytes
    }
}

impl<'a, const RANK: usize> I64View<'a, RANK> {
    /// Validates one public tensor descriptor as an exact I64 view.
    pub fn bind(tensor: TensorView<'a>, expected_shape: [u64; RANK]) -> CheckpointResult<Self> {
        Self::bind_validated(tensor, expected_shape)
    }

    /// Returns the unmodified little-endian I64 bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the source I64 value at `index`.
    pub fn value(&self, index: usize) -> Option<i64> {
        let begin = index.checked_mul(8)?;
        let end = begin.checked_add(8)?;
        let bytes: [u8; 8] = self.bytes.get(begin..end)?.try_into().ok()?;

        Some(i64::from_le_bytes(bytes))
    }

    /// Iterates over the source I64 values.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = i64> + ExactSizeIterator + '_ {
        self.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .copied()
            .map(i64::from_le_bytes)
    }
}

impl<'a, const RANK: usize> U8View<'a, RANK> {
    /// Binds a validated view from an admitted tensor descriptor.
    pub fn bind(tensor: TensorView<'a>, expected_shape: [u64; RANK]) -> CheckpointResult<Self> {
        Self::bind_validated(tensor, expected_shape)
    }

    /// Returns the unmodified U8 source elements.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bf16, Bf16View, F32, F32View, Fp8E4M3, Fp8E4M3View, I64, I64View, TensorElement, U8, U8View,
    };
    use crate::{CheckpointErrorCode, DType, TensorView};

    #[test]
    fn typed_view_elements_pin_their_source_dtype_and_element_width() {
        const PAYLOAD: [u8; 32] = [0; 32];

        let descriptor = |dtype: DType| TensorView {
            name: "typed",
            dtype,
            shape: &[4],
            bytes: &PAYLOAD[..4 * dtype.byte_width() as usize],
            data_range: 0..0,
        };

        assert_eq!(Bf16::DTYPE, DType::Bf16);
        assert_eq!(F32::DTYPE, DType::F32);
        assert_eq!(Fp8E4M3::DTYPE, DType::Fp8E4M3);
        assert_eq!(I64::DTYPE, DType::I64);
        assert_eq!(U8::DTYPE, DType::U8);

        let bf16 = Bf16View::bind(descriptor(DType::Bf16), [4]).unwrap();
        let f32 = F32View::bind(descriptor(DType::F32), [4]).unwrap();
        let fp8 = Fp8E4M3View::bind(descriptor(DType::Fp8E4M3), [4]).unwrap();
        let i64 = I64View::bind(descriptor(DType::I64), [4]).unwrap();
        let u8 = U8View::bind(descriptor(DType::U8), [4]).unwrap();

        assert_eq!(
            [bf16.len(), f32.len(), fp8.len(), i64.len(), u8.len()],
            [4, 4, 4, 4, 4],
            "element widths must divide the source byte length"
        );
        assert_eq!(
            [
                bf16.bytes().len(),
                f32.bytes().len(),
                fp8.codes().len(),
                i64.bytes().len(),
                u8.bytes().len()
            ],
            [8, 16, 4, 32, 4]
        );
    }

    #[test]
    fn i64_values_preserve_little_endian_source_words() {
        // Large values expose truncation and sign-extension errors.
        const MULTIPLIERS: [i64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

        let payload = MULTIPLIERS
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = I64View::bind(
            TensorView {
                name: "layer_multipliers",
                dtype: DType::I64,
                shape: &[3],
                bytes: &payload,
                data_range: 0..24,
            },
            [3],
        )
        .unwrap();

        assert_eq!(view.value(0), Some(23_703_573_157_769));
        assert_eq!(view.value(2), Some(8_052_911_324_071));
        assert_eq!(view.value(3), None);
        assert_eq!(view.values().collect::<Vec<_>>(), MULTIPLIERS);
        assert_eq!(view.bytes(), payload.as_slice());
    }

    #[test]
    fn i64_view_rejects_a_source_that_is_not_i64() {
        let payload = [0u8; 24];
        let error = I64View::bind(
            TensorView {
                name: "layer_multipliers",
                dtype: DType::F32,
                shape: &[3],
                bytes: &payload[..12],
                data_range: 0..12,
            },
            [3],
        )
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("dtype `F32`, expected `I64`"));
    }

    #[test]
    fn bf16_words_preserve_little_endian_source_bits() {
        let tensor = TensorView {
            name: "bf16",
            dtype: DType::Bf16,
            shape: &[2],
            bytes: &[0x80, 0x3f, 0x00, 0x40],
            data_range: 0..4,
        };

        let view = Bf16View::bind(tensor, [2]).unwrap();

        assert_eq!(view.word(0), Some(0x3f80));
        assert_eq!(view.word(1), Some(0x4000));
        assert_eq!(view.word(2), None);
        assert_eq!(view.words().collect::<Vec<_>>(), [0x3f80, 0x4000]);
    }

    #[test]
    fn typed_views_reject_dtype_and_shape_mismatches() {
        let wrong_dtype = TensorView {
            name: "codes",
            dtype: DType::U8,
            shape: &[2],
            bytes: &[1, 2],
            data_range: 0..2,
        };
        let wrong_shape = TensorView {
            name: "codes",
            dtype: DType::Fp8E4M3,
            shape: &[2],
            bytes: &[1, 2],
            data_range: 0..2,
        };

        let dtype_error = Fp8E4M3View::bind(wrong_dtype, [2]).err().unwrap();
        let shape_error = Fp8E4M3View::bind(wrong_shape, [1, 2]).err().unwrap();

        assert_eq!(dtype_error.code(), CheckpointErrorCode::Tensor);
        assert_eq!(shape_error.code(), CheckpointErrorCode::Tensor);
        assert!(
            dtype_error
                .to_string()
                .contains("dtype `U8`, expected `F8_E4M3`")
        );
        assert!(
            shape_error
                .to_string()
                .contains("shape [2], expected [1, 2]")
        );
    }

    #[test]
    fn typed_views_reject_byte_length_mismatch() {
        let tensor = TensorView {
            name: "bf16",
            dtype: DType::Bf16,
            shape: &[2],
            bytes: &[0x80, 0x3f],
            data_range: 0..2,
        };

        let error = Bf16View::bind(tensor, [2]).err().unwrap().to_string();

        assert!(error.contains("has 2 bytes, expected 4"));
    }

    #[test]
    fn f32_values_preserve_little_endian_source_bits() {
        let tensor = TensorView {
            name: "f32",
            dtype: DType::F32,
            shape: &[2],
            bytes: &[0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x00, 0xbf],
            data_range: 0..8,
        };

        let view = F32View::bind(tensor, [2]).unwrap();

        assert_eq!(view.bits(0), Some(3.0f32.to_bits()));
        assert_eq!(view.value(1), Some(-0.5));
        assert_eq!(view.value(2), None);
    }

    #[test]
    fn u8_view_preserves_source_bytes() {
        let tensor = TensorView {
            name: "packed",
            dtype: DType::U8,
            shape: &[2, 2],
            bytes: &[0x10, 0x32, 0x54, 0x76],
            data_range: 0..4,
        };

        let view = U8View::bind(tensor, [2, 2]).unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.bytes(), &[0x10, 0x32, 0x54, 0x76]);
    }
}

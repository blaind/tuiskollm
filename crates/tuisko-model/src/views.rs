use crate::{CheckpointError, CheckpointResult, DType, TensorView};

#[derive(Clone, Copy, Debug)]
struct ValidatedView<'a, const RANK: usize> {
    name: &'a str,
    shape: [u64; RANK],
    bytes: &'a [u8],
}

/// Validated zero-copy view of little-endian BF16 source words.
#[derive(Clone, Copy, Debug)]
pub struct Bf16View<'a, const RANK: usize> {
    view: ValidatedView<'a, RANK>,
}

impl<'a, const RANK: usize> Bf16View<'a, RANK> {
    pub(crate) fn bind(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        Ok(Self {
            view: validate(tensor, DType::Bf16, expected_shape)?,
        })
    }

    pub fn name(&self) -> &'a str {
        self.view.name
    }

    pub fn shape(&self) -> &[u64; RANK] {
        &self.view.shape
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.view.bytes
    }

    pub fn len(&self) -> usize {
        self.view.bytes.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.view.bytes.is_empty()
    }

    pub fn word(&self, index: usize) -> Option<u16> {
        let begin = index.checked_mul(2)?;
        let end = begin.checked_add(2)?;
        let bytes: [u8; 2] = self.view.bytes.get(begin..end)?.try_into().ok()?;

        Some(u16::from_le_bytes(bytes))
    }

    pub fn words(&self) -> impl DoubleEndedIterator<Item = u16> + ExactSizeIterator + '_ {
        self.view
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
    }
}

/// Validated zero-copy view of little-endian F32 source values.
#[derive(Clone, Copy, Debug)]
pub struct F32View<'a, const RANK: usize> {
    view: ValidatedView<'a, RANK>,
}

impl<'a, const RANK: usize> F32View<'a, RANK> {
    pub(crate) fn bind(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        Ok(Self {
            view: validate(tensor, DType::F32, expected_shape)?,
        })
    }

    pub fn name(&self) -> &'a str {
        self.view.name
    }

    pub fn shape(&self) -> &[u64; RANK] {
        &self.view.shape
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.view.bytes
    }

    pub fn len(&self) -> usize {
        self.view.bytes.len() / 4
    }

    pub fn is_empty(&self) -> bool {
        self.view.bytes.is_empty()
    }

    pub fn bits(&self, index: usize) -> Option<u32> {
        let begin = index.checked_mul(4)?;
        let end = begin.checked_add(4)?;
        let bytes: [u8; 4] = self.view.bytes.get(begin..end)?.try_into().ok()?;

        Some(u32::from_le_bytes(bytes))
    }

    pub fn value(&self, index: usize) -> Option<f32> {
        self.bits(index).map(f32::from_bits)
    }
}

/// Validated zero-copy view of FP8 E4M3 source codes.
#[derive(Clone, Copy, Debug)]
pub struct Fp8E4M3View<'a, const RANK: usize> {
    view: ValidatedView<'a, RANK>,
}

impl<'a, const RANK: usize> Fp8E4M3View<'a, RANK> {
    pub(crate) fn bind(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        Ok(Self {
            view: validate(tensor, DType::Fp8E4M3, expected_shape)?,
        })
    }

    pub fn name(&self) -> &'a str {
        self.view.name
    }

    pub fn shape(&self) -> &[u64; RANK] {
        &self.view.shape
    }

    pub fn codes(&self) -> &'a [u8] {
        self.view.bytes
    }

    pub fn len(&self) -> usize {
        self.view.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.view.bytes.is_empty()
    }
}

/// Validated zero-copy view of raw U8 source elements.
#[derive(Clone, Copy, Debug)]
pub struct U8View<'a, const RANK: usize> {
    view: ValidatedView<'a, RANK>,
}

impl<'a, const RANK: usize> U8View<'a, RANK> {
    pub(crate) fn bind(
        tensor: TensorView<'a>,
        expected_shape: [u64; RANK],
    ) -> CheckpointResult<Self> {
        Ok(Self {
            view: validate(tensor, DType::U8, expected_shape)?,
        })
    }

    pub fn name(&self) -> &'a str {
        self.view.name
    }

    pub fn shape(&self) -> &[u64; RANK] {
        &self.view.shape
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.view.bytes
    }

    pub fn len(&self) -> usize {
        self.view.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.view.bytes.is_empty()
    }
}

fn validate<'a, const RANK: usize>(
    tensor: TensorView<'a>,
    expected_dtype: DType,
    expected_shape: [u64; RANK],
) -> CheckpointResult<ValidatedView<'a, RANK>> {
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

    Ok(ValidatedView {
        name: tensor.name,
        shape: expected_shape,
        bytes: tensor.bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::{Bf16View, F32View, Fp8E4M3View, U8View};
    use crate::{CheckpointErrorCode, DType, TensorView};

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

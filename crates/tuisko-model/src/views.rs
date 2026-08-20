use crate::{CheckpointError, CheckpointResult, TensorView};

#[derive(Clone, Copy, Debug)]
struct ValidatedView<'a, const RANK: usize> {
    name: &'a str,
    shape: [u64; RANK],
    bytes: &'a [u8],
}

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
            view: validate(tensor, "BF16", expected_shape, 2)?,
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
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
    }
}

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
            view: validate(tensor, "F8_E4M3", expected_shape, 1)?,
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

fn validate<'a, const RANK: usize>(
    tensor: TensorView<'a>,
    expected_dtype: &str,
    expected_shape: [u64; RANK],
    element_bytes: u64,
) -> CheckpointResult<ValidatedView<'a, RANK>> {
    if tensor.dtype != expected_dtype {
        return Err(CheckpointError::invalid(format!(
            "tensor `{}` has dtype `{}`, expected `{expected_dtype}`",
            tensor.name, tensor.dtype
        )));
    }

    if tensor.shape != expected_shape {
        return Err(CheckpointError::invalid(format!(
            "tensor `{}` has shape {:?}, expected {expected_shape:?}",
            tensor.name, tensor.shape
        )));
    }

    let elements = expected_shape.iter().try_fold(1u64, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(|| {
            CheckpointError::invalid(format!("tensor `{}` shape overflows", tensor.name))
        })
    })?;
    let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
        CheckpointError::invalid(format!("tensor `{}` byte length overflows", tensor.name))
    })?;
    let actual_bytes = u64::try_from(tensor.bytes.len()).map_err(|_| {
        CheckpointError::invalid(format!(
            "tensor `{}` is too large for this host",
            tensor.name
        ))
    })?;

    if actual_bytes != expected_bytes {
        return Err(CheckpointError::invalid(format!(
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
    use super::{Bf16View, Fp8E4M3View};
    use crate::TensorView;

    #[test]
    fn bf16_words_preserve_little_endian_source_bits() {
        let tensor = TensorView {
            name: "bf16",
            dtype: "BF16",
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
            dtype: "U8",
            shape: &[2],
            bytes: &[1, 2],
            data_range: 0..2,
        };
        let wrong_shape = TensorView {
            name: "codes",
            dtype: "F8_E4M3",
            shape: &[2],
            bytes: &[1, 2],
            data_range: 0..2,
        };

        let dtype_error = Fp8E4M3View::bind(wrong_dtype, [2])
            .err()
            .unwrap()
            .to_string();
        let shape_error = Fp8E4M3View::bind(wrong_shape, [1, 2])
            .err()
            .unwrap()
            .to_string();

        assert!(dtype_error.contains("dtype `U8`, expected `F8_E4M3`"));
        assert!(shape_error.contains("shape [2], expected [1, 2]"));
    }

    #[test]
    fn typed_views_reject_byte_length_mismatch() {
        let tensor = TensorView {
            name: "bf16",
            dtype: "BF16",
            shape: &[2],
            bytes: &[0x80, 0x3f],
            data_range: 0..2,
        };

        let error = Bf16View::bind(tensor, [2]).err().unwrap().to_string();

        assert!(error.contains("has 2 bytes, expected 4"));
    }
}

//! Validated mmap access to safetensors headers and tensor byte ranges.

use crate::{CheckpointError, CheckpointResult, DType};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct TensorDescriptor {
    dtype: DType,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl TensorDescriptor {
    fn range(&self) -> Range<u64> {
        self.data_offsets[0]..self.data_offsets[1]
    }
}

/// Immutable, validated mmap of one safetensors file.
pub struct SafeTensorFile {
    path: PathBuf,
    mmap: Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorDescriptor>,
}

impl SafeTensorFile {
    /// Opens and validates one immutable safetensors file.
    pub fn open(path: &Path) -> CheckpointResult<Self> {
        let file = File::open(path).map_err(|error| CheckpointError::io("opening", path, error))?;

        let file_len = file
            .metadata()
            .map_err(|error| CheckpointError::io("reading metadata for", path, error))?
            .len();

        if file_len < 8 {
            return Err(CheckpointError::safetensors(format!(
                "{} is too small to be a safetensors file",
                path.display()
            )));
        }

        // SAFETY: admitted snapshots are trusted, immutable local inputs. The
        // mapping is read-only and owns no reference to a temporary buffer.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| CheckpointError::io("memory-mapping", path, error))?;

        let header_len = u64::from_le_bytes(mmap[..8].try_into().expect("eight bytes checked"));

        if header_len == 0 || header_len % 8 != 0 {
            return Err(CheckpointError::safetensors(format!(
                "{} has invalid safetensors header length {header_len}",
                path.display()
            )));
        }

        let data_start_u64 = 8u64.checked_add(header_len).ok_or_else(|| {
            CheckpointError::safetensors(format!("{} header length overflows", path.display()))
        })?;

        if data_start_u64 > file_len {
            return Err(CheckpointError::safetensors(format!(
                "{} declares a {header_len}-byte header beyond its {file_len}-byte file",
                path.display()
            )));
        }

        let data_start = usize::try_from(data_start_u64).map_err(|_| {
            CheckpointError::safetensors(format!("{} is too large for this host", path.display()))
        })?;

        let header: Value = serde_json::from_slice(&mmap[8..data_start])
            .map_err(|source| CheckpointError::json(path, source))?;

        let object = header.as_object().ok_or_else(|| {
            CheckpointError::safetensors(format!(
                "{} safetensors header is not a JSON object",
                path.display()
            ))
        })?;

        let payload_len = file_len - data_start_u64;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();

        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }

            let descriptor: TensorDescriptor = serde_json::from_value(value.clone())
                .map_err(|source| CheckpointError::json(path, source))?;

            validate_descriptor(path, name, &descriptor, payload_len)?;
            ranges.push((descriptor.data_offsets[0], descriptor.data_offsets[1], name));
            tensors.insert(name.clone(), descriptor);
        }

        ranges.sort_unstable_by_key(|range| range.0);

        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(CheckpointError::safetensors(format!(
                    "{} tensors `{}` and `{}` overlap",
                    path.display(),
                    pair[0].2,
                    pair[1].2
                )));
            }
        }

        Ok(Self {
            path: path.to_owned(),
            mmap,
            data_start,
            tensors,
        })
    }

    /// Returns the mapped file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of validated tensor descriptors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn header_bytes(&self) -> usize {
        self.data_start - 8
    }

    /// Returns a validated source view for `name`.
    pub fn tensor(&self, name: &str) -> CheckpointResult<TensorView<'_>> {
        let (stored_name, descriptor) = self.tensors.get_key_value(name).ok_or_else(|| {
            CheckpointError::tensor(format!(
                "{} is missing tensor `{name}`",
                self.path.display()
            ))
        })?;

        let bytes = self.bytes(descriptor.range())?;

        Ok(TensorView {
            name: stored_name,
            dtype: descriptor.dtype,
            shape: &descriptor.shape,
            bytes,
            data_range: descriptor.range(),
        })
    }

    pub(crate) fn adjacent_tensor_bytes(
        &self,
        first_name: &str,
        second_name: &str,
        role: &str,
    ) -> CheckpointResult<&[u8]> {
        let first = self.tensor(first_name)?;
        let second = self.tensor(second_name)?;

        if first.data_range.end != second.data_range.start {
            return Err(CheckpointError::source_binding(format!(
                "{role} are not source-adjacent"
            )));
        }

        self.bytes(first.data_range.start..second.data_range.end)
    }

    fn bytes(&self, range: Range<u64>) -> CheckpointResult<&[u8]> {
        let begin = usize::try_from(range.start)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .ok_or_else(|| {
                CheckpointError::safetensors("safetensors byte offset overflows host")
            })?;

        let end = usize::try_from(range.end)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .ok_or_else(|| {
                CheckpointError::safetensors("safetensors byte offset overflows host")
            })?;

        self.mmap.get(begin..end).ok_or_else(|| {
            CheckpointError::safetensors(format!(
                "{} tensor byte range {}..{} is outside the mapping",
                self.path.display(),
                range.start,
                range.end
            ))
        })
    }
}

/// Borrowed source tensor whose descriptor and byte range have been validated.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    /// Exact tensor name from the safetensors header.
    pub name: &'a str,
    /// Stored element representation.
    pub dtype: DType,
    /// Row-major dimensions from the safetensors header.
    pub shape: &'a [u64],
    /// Unmodified source bytes.
    pub bytes: &'a [u8],
    /// Byte offsets relative to the shard's data section.
    pub data_range: Range<u64>,
}

fn validate_descriptor(
    path: &Path,
    name: &str,
    descriptor: &TensorDescriptor,
    payload_len: u64,
) -> CheckpointResult<()> {
    let [begin, end] = descriptor.data_offsets;

    if begin > end || end > payload_len {
        return Err(CheckpointError::safetensors(format!(
            "{} tensor `{name}` has invalid byte range {begin}..{end} for a {payload_len}-byte payload",
            path.display()
        )));
    }

    let width = descriptor.dtype.byte_width();

    let elements = descriptor
        .shape
        .iter()
        .try_fold(1u64, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                CheckpointError::safetensors(format!(
                    "{} tensor `{name}` shape overflows",
                    path.display()
                ))
            })
        })?;

    let expected_bytes = elements.checked_mul(width).ok_or_else(|| {
        CheckpointError::safetensors(format!(
            "{} tensor `{name}` byte length overflows",
            path.display()
        ))
    })?;

    if end - begin != expected_bytes {
        return Err(CheckpointError::safetensors(format!(
            "{} tensor `{name}` dtype {} shape {:?} requires {expected_bytes} bytes, not {}",
            path.display(),
            descriptor.dtype,
            descriptor.shape,
            end - begin
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SafeTensorFile;
    use crate::{CheckpointErrorCode, DType};
    use serde_json::{Value, json};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-model-{label}-{}-{}.safetensors",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_safetensors(path: &Path, header: Value, payload: &[u8]) {
        let mut header = serde_json::to_vec(&header).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = File::create(path).unwrap();

        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
    }

    #[test]
    fn mmap_preserves_exact_represented_bytes() {
        let path = fixture_path("exact-bytes");
        write_safetensors(
            &path,
            json!({
                "codes": {"dtype":"F8_E4M3", "shape":[2,2], "data_offsets":[0,4]},
                "scales": {"dtype":"BF16", "shape":[2,1], "data_offsets":[4,8]}
            }),
            &[0x38, 0xb8, 0x01, 0x7e, 0x80, 0x3f, 0x00, 0x40],
        );

        let file = SafeTensorFile::open(&path).unwrap();
        let codes = file.tensor("codes").unwrap();

        assert_eq!(codes.dtype, DType::Fp8E4M3);
        assert_eq!(codes.shape, &[2, 2]);
        assert_eq!(codes.bytes, &[0x38, 0xb8, 0x01, 0x7e]);

        let scales = file.tensor("scales").unwrap();

        assert_eq!(scales.bytes, &[0x80, 0x3f, 0x00, 0x40]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_shape_byte_mismatch() {
        let path = fixture_path("bad-shape");
        write_safetensors(
            &path,
            json!({"x": {"dtype":"BF16", "shape":[3], "data_offsets":[0,4]}}),
            &[0; 4],
        );

        let error = SafeTensorFile::open(&path).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Safetensors);
        assert!(error.to_string().contains("requires 6 bytes, not 4"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let path = fixture_path("overlap");
        write_safetensors(
            &path,
            json!({
                "a": {"dtype":"U8", "shape":[4], "data_offsets":[0,4]},
                "b": {"dtype":"U8", "shape":[4], "data_offsets":[2,6]}
            }),
            &[0; 6],
        );

        let error = SafeTensorFile::open(&path).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Safetensors);
        assert!(error.to_string().contains("overlap"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_nonadjacent_tensor_span() {
        let path = fixture_path("nonadjacent-span");
        write_safetensors(
            &path,
            json!({
                "first": {"dtype":"U8", "shape":[1], "data_offsets":[0,1]},
                "middle": {"dtype":"U8", "shape":[1], "data_offsets":[1,2]},
                "second": {"dtype":"U8", "shape":[1], "data_offsets":[2,3]}
            }),
            &[0x10, 0x20, 0x30],
        );
        let file = SafeTensorFile::open(&path).unwrap();

        let error = file
            .adjacent_tensor_bytes("first", "second", "test planes")
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("test planes are not source-adjacent")
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_dtype_outside_checkpoint_contract() {
        let path = fixture_path("unsupported-dtype");
        write_safetensors(
            &path,
            json!({"x": {"dtype":"F16", "shape":[1], "data_offsets":[0,2]}}),
            &[0; 2],
        );

        let error = SafeTensorFile::open(&path).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Json);
        assert!(
            error
                .to_string()
                .contains("unsupported checkpoint dtype `F16`")
        );

        fs::remove_file(path).unwrap();
    }
}

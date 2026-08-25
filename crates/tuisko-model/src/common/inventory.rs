//! Shared snapshot inventory admission and mmap-backed tensor lookup.

use crate::qwen36::inventory::QWEN36_INVENTORY;
use crate::qwen38::inventory::QWEN38_INVENTORY;
use crate::{
    Arch, CheckpointContract, CheckpointError, CheckpointResult, DType, SafeTensorFile, TensorView,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

pub(crate) const CONFIG_FILE: &str = "config.json";
pub(crate) const INDEX_FILE: &str = "model.safetensors.index.json";
pub(crate) const MODEL_FILE: &str = "model.safetensors";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Index {
    pub(crate) metadata: IndexMetadata,
    pub(crate) weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IndexMetadata {
    #[serde(default)]
    pub(crate) total_parameters: Option<u64>,
    pub(crate) total_size: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shard(pub(crate) usize);

/// Exact-inventory, mmap-backed view of an admitted checkpoint snapshot.
pub struct CheckpointSnapshot<A: Arch> {
    pub(crate) root: PathBuf,
    pub(crate) inventory_path: PathBuf,
    pub(crate) tensors: BTreeMap<String, Shard>,
    pub(crate) shards: Vec<SafeTensorFile>,
    pub(crate) arch: PhantomData<A>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExpectedTensor {
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<u64>,
}

pub(crate) fn add_qwen35_vision<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<()> {
    let hidden = dimension(A::VISION_HIDDEN, "Vision hidden width")?;
    let intermediate = dimension(A::VISION_INTERMEDIATE, "Vision intermediate width")?;
    let qkv_rows = hidden
        .checked_mul(3)
        .ok_or_else(|| CheckpointError::inventory("Vision QKV rows overflow"))?;

    for block in 0..A::VISION_DEPTH {
        let prefix = format!("model.visual.blocks.{block}");

        for norm in ["norm1", "norm2"] {
            for plane in ["weight", "bias"] {
                add_expected(
                    expected,
                    format!("{prefix}.{norm}.{plane}"),
                    DType::Bf16,
                    vec![hidden],
                )?;
            }
        }
        for (name, shape) in [
            ("attn.qkv.weight", vec![qkv_rows, hidden]),
            ("attn.qkv.bias", vec![qkv_rows]),
            ("attn.proj.weight", vec![hidden, hidden]),
            ("attn.proj.bias", vec![hidden]),
            ("mlp.linear_fc1.weight", vec![intermediate, hidden]),
            ("mlp.linear_fc1.bias", vec![intermediate]),
            ("mlp.linear_fc2.weight", vec![hidden, intermediate]),
            ("mlp.linear_fc2.bias", vec![hidden]),
        ] {
            add_expected(expected, format!("{prefix}.{name}"), DType::Bf16, shape)?;
        }
    }

    let merge_width = A::VISION_HIDDEN
        .checked_mul(A::VISION_SPATIAL_MERGE_SIZE)
        .and_then(|width| width.checked_mul(A::VISION_SPATIAL_MERGE_SIZE))
        .ok_or_else(|| CheckpointError::inventory("Vision merger width overflows"))?;
    let merge_width = dimension(merge_width, "Vision merger width")?;
    let output_hidden = dimension(A::VISION_OUTPUT_HIDDEN, "Vision output width")?;

    for plane in ["weight", "bias"] {
        add_expected(
            expected,
            format!("model.visual.merger.norm.{plane}"),
            DType::Bf16,
            vec![hidden],
        )?;
    }
    for (name, shape) in [
        ("linear_fc1.weight", vec![merge_width, merge_width]),
        ("linear_fc1.bias", vec![merge_width]),
        ("linear_fc2.weight", vec![output_hidden, merge_width]),
        ("linear_fc2.bias", vec![output_hidden]),
    ] {
        add_expected(
            expected,
            format!("model.visual.merger.{name}"),
            DType::Bf16,
            shape,
        )?;
    }
    add_expected(
        expected,
        "model.visual.patch_embed.proj.weight",
        DType::Bf16,
        vec![
            hidden,
            dimension(A::VISION_INPUT_CHANNELS, "Vision input channels")?,
            dimension(A::VISION_TEMPORAL_PATCH_SIZE, "Vision temporal patch size")?,
            dimension(A::VISION_PATCH_SIZE, "Vision patch size")?,
            dimension(A::VISION_PATCH_SIZE, "Vision patch size")?,
        ],
    )?;
    add_expected(
        expected,
        "model.visual.patch_embed.proj.bias",
        DType::Bf16,
        vec![hidden],
    )?;
    add_expected(
        expected,
        "model.visual.pos_embed.weight",
        DType::Bf16,
        vec![dimension(A::VISION_POSITIONS, "Vision positions")?, hidden],
    )?;

    Ok(())
}

pub(crate) fn add_modelopt_linear(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
    rows: u64,
    columns: u64,
) -> CheckpointResult<()> {
    let packed_columns = divided_dimension(columns, 2, prefix)?;
    let scale_columns = divided_dimension(columns, 16, prefix)?;

    add_expected(
        expected,
        format!("{prefix}.input_scale"),
        DType::F32,
        vec![],
    )?;
    add_expected(
        expected,
        format!("{prefix}.weight"),
        DType::U8,
        vec![rows, packed_columns],
    )?;
    add_expected(
        expected,
        format!("{prefix}.weight_scale"),
        DType::Fp8E4M3,
        vec![rows, scale_columns],
    )?;
    add_expected(
        expected,
        format!("{prefix}.weight_scale_2"),
        DType::F32,
        vec![],
    )
}

pub(crate) fn add_expected(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    name: impl Into<String>,
    dtype: DType,
    shape: Vec<u64>,
) -> CheckpointResult<()> {
    let name = name.into();

    if expected
        .insert(name.clone(), ExpectedTensor { dtype, shape })
        .is_some()
    {
        return Err(CheckpointError::inventory(format!(
            "checkpoint inventory generates duplicate tensor `{name}`"
        )));
    }

    Ok(())
}

pub(crate) fn validate_exact_tensors(
    file: &SafeTensorFile,
    expected: &BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<()> {
    for (name, descriptor) in expected {
        validate_expected_tensor(file, name, descriptor)?;
    }

    Ok(())
}

pub(crate) fn validate_expected_tensor(
    file: &SafeTensorFile,
    name: &str,
    descriptor: &ExpectedTensor,
) -> CheckpointResult<()> {
    let tensor = file.tensor(name)?;

    if tensor.dtype != descriptor.dtype || tensor.shape != descriptor.shape {
        return Err(CheckpointError::tensor(format!(
            "{} tensor `{name}` is {} {:?}, expected {} {:?}",
            file.path().display(),
            tensor.dtype,
            tensor.shape,
            descriptor.dtype,
            descriptor.shape
        )));
    }

    Ok(())
}

pub(crate) fn dimension(value: usize, field: &str) -> CheckpointResult<u64> {
    u64::try_from(value).map_err(|_| CheckpointError::inventory(format!("{field} exceeds u64")))
}

fn divided_dimension(value: u64, divisor: u64, field: &str) -> CheckpointResult<u64> {
    if !value.is_multiple_of(divisor) {
        return Err(CheckpointError::inventory(format!(
            "{field} width {value} is not divisible by {divisor}"
        )));
    }

    Ok(value / divisor)
}

pub(crate) fn validate_revision<A: Arch>(root: &Path) -> CheckpointResult<()> {
    let actual = root.file_name().and_then(|name| name.to_str());

    if actual != Some(A::REVISION) {
        return Err(CheckpointError::revision(format!(
            "{} is revision {actual:?}, expected {:?}",
            root.display(),
            A::REVISION
        )));
    }

    Ok(())
}

pub(crate) fn validate_file_length(path: &Path, expected: u64) -> CheckpointResult<()> {
    let actual = fs::metadata(path)
        .map_err(|source| CheckpointError::io("reading metadata for", path, source))?
        .len();

    if actual != expected {
        return Err(CheckpointError::inventory(format!(
            "{} has {actual} bytes, expected {expected}",
            path.display()
        )));
    }

    Ok(())
}

pub(crate) fn read_index(path: &Path) -> CheckpointResult<Index> {
    let bytes = fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;

    serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))
}

pub(crate) fn require_count<T>(
    path: &Path,
    field: &str,
    actual: T,
    expected: T,
) -> CheckpointResult<()>
where
    T: Copy + std::fmt::Display + PartialEq,
{
    if actual != expected {
        return Err(CheckpointError::inventory(format!(
            "{} {field} is {actual}, expected {expected}",
            path.display()
        )));
    }

    Ok(())
}

impl<A: Arch> CheckpointSnapshot<A> {
    /// Opens and validates the pinned snapshot rooted at `root`.
    pub fn open(root: &Path) -> CheckpointResult<Self> {
        match A::CHECKPOINT_CONTRACT {
            CheckpointContract::CompressedTensors => {
                Self::open_split_with_spec(root, QWEN38_INVENTORY)
            }
            CheckpointContract::ModelOptNvfp4 => Self::open_modelopt(root),
            CheckpointContract::ModelOptNvfp4Moe => Self::open_indexed(root, QWEN36_INVENTORY),
        }
    }

    /// Returns the admitted snapshot root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the number of indexed tensors across both shards.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Populates page tables for the immutable base-model shard without copying its bytes.
    #[cfg(target_os = "linux")]
    pub fn prefault_model_shard(&self) -> CheckpointResult<usize> {
        self.shards[0].prefault_except(crate::common::naming::EMBEDDING)
    }

    /// Returns a validated source view for an indexed tensor.
    pub fn tensor(&self, name: &str) -> CheckpointResult<TensorView<'_>> {
        self.shard_file(name)?.tensor(name)
    }

    pub(crate) fn adjacent_tensor_bytes(
        &self,
        first_name: &str,
        second_name: &str,
        role: &str,
    ) -> CheckpointResult<&[u8]> {
        let first = self.shard(first_name)?;
        let second = self.shard(second_name)?;

        if first != second {
            return Err(CheckpointError::source_binding(format!(
                "{role} are split across checkpoint shards"
            )));
        }

        self.shards[first.0].adjacent_tensor_bytes(first_name, second_name, role)
    }

    fn shard(&self, name: &str) -> CheckpointResult<Shard> {
        self.tensors.get(name).copied().ok_or_else(|| {
            CheckpointError::tensor(format!(
                "{} inventory is missing tensor `{name}`",
                self.inventory_path.display()
            ))
        })
    }

    fn shard_file(&self, name: &str) -> CheckpointResult<&SafeTensorFile> {
        let shard = self.shard(name)?;
        self.shards.get(shard.0).ok_or_else(|| {
            CheckpointError::inventory(format!(
                "{} tensor `{name}` refers to missing shard {}",
                self.inventory_path.display(),
                shard.0
            ))
        })
    }
}

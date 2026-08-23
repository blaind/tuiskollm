//! Exact snapshot inventory admission and mmap-backed tensor lookup.

use crate::{
    Arch, CheckpointContract, CheckpointError, CheckpointResult, DType, SafeTensorFile, TensorView,
    validate_config,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const MODEL_FILE: &str = "model.safetensors";
const MTP_FILE: &str = "model_mtp.safetensors";
const QWEN36_SHARD_FILES: [&str; 3] = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
];

const QWEN38_INVENTORY: SplitInventorySpec = SplitInventorySpec {
    model_bytes: 22_568_192_096,
    model_tensors: 1_953,
    mtp_bytes: 849_400_392,
    mtp_tensors: 15,
    index_bytes: 164_371,
    index_entries: 1_968,
};

const QWEN35_MODEL_BYTES: u64 = 9_361_048_680;
const QWEN35_HEADER_BYTES: usize = 188_096;
const QWEN35_TENSORS: usize = 1_519;

const QWEN36_INVENTORY: IndexedInventorySpec = IndexedInventorySpec {
    index_bytes: 13_726_227,
    index_entries: 124_468,
    total_parameters: 18_683_860_336,
    payload_bytes: 23_407_580_856,
    shards: [
        IndexedShardSpec {
            file: QWEN36_SHARD_FILES[0],
            file_bytes: 10_006_877_608,
            header_bytes: 6_937_600,
            tensors: 51_662,
        },
        IndexedShardSpec {
            file: QWEN36_SHARD_FILES[1],
            file_bytes: 10_003_595_752,
            header_bytes: 8_566_488,
            tensors: 63_484,
        },
        IndexedShardSpec {
            file: QWEN36_SHARD_FILES[2],
            file_bytes: 3_413_864_960,
            header_bytes: 1_253_352,
            tensors: 9_322,
        },
    ],
};

#[derive(Clone, Copy)]
struct SplitInventorySpec {
    model_bytes: u64,
    model_tensors: usize,
    mtp_bytes: u64,
    mtp_tensors: usize,
    index_bytes: u64,
    index_entries: usize,
}

#[derive(Clone, Copy)]
struct IndexedInventorySpec {
    index_bytes: u64,
    index_entries: usize,
    total_parameters: u64,
    payload_bytes: u64,
    shards: [IndexedShardSpec; 3],
}

#[derive(Clone, Copy)]
struct IndexedShardSpec {
    file: &'static str,
    file_bytes: u64,
    header_bytes: usize,
    tensors: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    #[serde(default)]
    total_parameters: Option<u64>,
    total_size: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Shard(usize);

/// Exact-inventory, mmap-backed view of an admitted checkpoint snapshot.
pub struct CheckpointSnapshot<A: Arch> {
    root: PathBuf,
    inventory_path: PathBuf,
    tensors: BTreeMap<String, Shard>,
    shards: Vec<SafeTensorFile>,
    arch: PhantomData<A>,
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
        self.shards[0].prefault_except(crate::bindings::EMBEDDING)
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

    fn open_split_with_spec(root: &Path, spec: SplitInventorySpec) -> CheckpointResult<Self> {
        validate_revision::<A>(root)?;
        validate_config::<A>(&root.join(CONFIG_FILE))?;

        let index_path = root.join(INDEX_FILE);
        let model_path = root.join(MODEL_FILE);
        let mtp_path = root.join(MTP_FILE);

        validate_file_length(&index_path, spec.index_bytes)?;
        validate_file_length(&model_path, spec.model_bytes)?;
        validate_file_length(&mtp_path, spec.mtp_bytes)?;

        let index = read_index(&index_path)?;
        let total_bytes = spec
            .model_bytes
            .checked_add(spec.mtp_bytes)
            .ok_or_else(|| CheckpointError::inventory("checkpoint shard lengths overflow"))?;

        require_count(
            &index_path,
            "entries",
            index.weight_map.len(),
            spec.index_entries,
        )?;
        require_count(
            &index_path,
            "metadata.total_size",
            index.metadata.total_size,
            total_bytes,
        )?;

        let model = SafeTensorFile::open(&model_path)?;
        let mtp = SafeTensorFile::open(&mtp_path)?;

        require_count(
            &model_path,
            "tensors",
            model.tensor_count(),
            spec.model_tensors,
        )?;
        require_count(&mtp_path, "tensors", mtp.tensor_count(), spec.mtp_tensors)?;

        let tensors = validate_weight_map(&index_path, index.weight_map, &model, &mtp, spec)?;

        Ok(Self {
            root: root.to_owned(),
            inventory_path: index_path,
            tensors,
            shards: vec![model, mtp],
            arch: PhantomData,
        })
    }

    fn open_modelopt(root: &Path) -> CheckpointResult<Self> {
        validate_revision::<A>(root)?;
        validate_config::<A>(&root.join(CONFIG_FILE))?;

        let model_path = root.join(MODEL_FILE);
        validate_file_length(&model_path, QWEN35_MODEL_BYTES)?;

        let model = SafeTensorFile::open(&model_path)?;
        require_count(
            &model_path,
            "safetensors header bytes",
            model.header_bytes(),
            QWEN35_HEADER_BYTES,
        )?;

        let expected = qwen35_expected_tensors::<A>()?;
        require_count(&model_path, "tensors", model.tensor_count(), QWEN35_TENSORS)?;
        require_count(
            &model_path,
            "expected tensors",
            expected.len(),
            QWEN35_TENSORS,
        )?;
        validate_exact_tensors(&model, &expected)?;

        let tensors = expected.into_keys().map(|name| (name, Shard(0))).collect();

        Ok(Self {
            root: root.to_owned(),
            inventory_path: model_path,
            tensors,
            shards: vec![model],
            arch: PhantomData,
        })
    }

    fn open_indexed(root: &Path, spec: IndexedInventorySpec) -> CheckpointResult<Self> {
        validate_revision::<A>(root)?;
        validate_config::<A>(&root.join(CONFIG_FILE))?;

        let index_path = root.join(INDEX_FILE);
        validate_file_length(&index_path, spec.index_bytes)?;
        let index = read_index(&index_path)?;
        require_count(
            &index_path,
            "entries",
            index.weight_map.len(),
            spec.index_entries,
        )?;
        require_count(
            &index_path,
            "metadata.total_size",
            index.metadata.total_size,
            spec.payload_bytes,
        )?;
        let total_parameters = index.metadata.total_parameters.ok_or_else(|| {
            CheckpointError::inventory(format!(
                "{} is missing metadata.total_parameters",
                index_path.display()
            ))
        })?;
        require_count(
            &index_path,
            "metadata.total_parameters",
            total_parameters,
            spec.total_parameters,
        )?;

        let mut shards = Vec::with_capacity(spec.shards.len());
        for shard in spec.shards {
            let path = root.join(shard.file);
            validate_file_length(&path, shard.file_bytes)?;
            let file = SafeTensorFile::open(&path)?;
            require_count(
                &path,
                "safetensors header bytes",
                file.header_bytes(),
                shard.header_bytes,
            )?;
            require_count(&path, "tensors", file.tensor_count(), shard.tensors)?;
            shards.push(file);
        }

        let expected = qwen36_expected_tensors::<A>()?;
        require_count(
            &index_path,
            "expected tensors",
            expected.len(),
            spec.index_entries,
        )?;
        let tensors =
            validate_indexed_weight_map(&index_path, index.weight_map, &shards, &expected, spec)?;

        Ok(Self {
            root: root.to_owned(),
            inventory_path: index_path,
            tensors,
            shards,
            arch: PhantomData,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedTensor {
    dtype: DType,
    shape: Vec<u64>,
}

fn qwen36_expected_tensors<A: Arch>() -> CheckpointResult<BTreeMap<String, ExpectedTensor>> {
    const EXPERTS: usize = 256;

    if A::FULL_ATTENTION_INTERVAL == 0 {
        return Err(CheckpointError::inventory(
            "Qwen3.6 full-attention interval must be nonzero",
        ));
    }

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let expert_intermediate = dimension(A::INTERMEDIATE, "expert intermediate width")?;
    let attention_query_rows = dimension(A::ATTENTION_QUERY_ROWS, "attention query rows")?;
    let attention_kv_rows = dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?;
    let attention_output_columns =
        dimension(A::ATTENTION_OUTPUT_COLUMNS, "attention output columns")?;
    let gdn_control_rows = dimension(A::GDN_CONTROL_ROWS, "GDN control rows")?;
    let gdn_input_rows = dimension(A::GDN_INPUT_ROWS, "GDN input rows")?;
    let gdn_value_rows = dimension(A::GDN_VALUE_ROWS, "GDN value rows")?;
    let gdn_qkv_rows = dimension(A::GDN_QKV_ROWS, "GDN QKV rows")?;
    let linear_head_dim = dimension(A::LINEAR_HEAD_DIM, "linear-attention head width")?;
    let convolution_width = dimension(A::LINEAR_CONV_KERNEL_DIM, "convolution width")?;
    let mut expected = BTreeMap::new();

    add_expected(
        &mut expected,
        "model.language_model.embed_tokens.weight",
        DType::Bf16,
        vec![dimension(A::VOCAB, "vocabulary size")?, hidden],
    )?;
    add_expected(
        &mut expected,
        "model.language_model.norm.weight",
        DType::Bf16,
        vec![hidden],
    )?;
    add_modelopt_linear(
        &mut expected,
        "lm_head",
        dimension(A::VOCAB, "vocabulary size")?,
        hidden,
    )?;

    for layer in 0..A::LAYERS {
        let layer_prefix = format!("model.language_model.layers.{layer}");

        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            add_expected(
                &mut expected,
                format!("{layer_prefix}.{name}"),
                DType::Bf16,
                vec![hidden],
            )?;
        }

        let mlp_prefix = format!("{layer_prefix}.mlp");
        add_expected(
            &mut expected,
            format!("{mlp_prefix}.gate.weight"),
            DType::Bf16,
            vec![dimension(EXPERTS, "expert count")?, hidden],
        )?;
        add_expected(
            &mut expected,
            format!("{mlp_prefix}.shared_expert_gate.weight"),
            DType::Bf16,
            vec![1, hidden],
        )?;
        add_qwen36_nvfp4_mlp(
            &mut expected,
            &format!("{mlp_prefix}.shared_expert"),
            hidden,
            expert_intermediate,
        )?;

        for expert in 0..EXPERTS {
            add_qwen36_nvfp4_mlp(
                &mut expected,
                &format!("{mlp_prefix}.experts.{expert}"),
                hidden,
                expert_intermediate,
            )?;
        }

        if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            for (projection, rows, columns) in [
                ("q_proj", attention_query_rows, hidden),
                ("k_proj", attention_kv_rows, hidden),
                ("v_proj", attention_kv_rows, hidden),
                ("o_proj", hidden, attention_output_columns),
            ] {
                add_qwen36_fp8_linear(
                    &mut expected,
                    &format!("{layer_prefix}.self_attn.{projection}"),
                    rows,
                    columns,
                )?;
            }

            for name in ["q_norm.weight", "k_norm.weight"] {
                add_expected(
                    &mut expected,
                    format!("{layer_prefix}.self_attn.{name}"),
                    DType::Bf16,
                    vec![dimension(A::HEAD_DIM, "attention head width")?],
                )?;
            }
        } else {
            for (projection, rows, columns) in [
                ("in_proj_qkv", gdn_qkv_rows, hidden),
                ("in_proj_z", gdn_value_rows, hidden),
                ("out_proj", hidden, gdn_value_rows),
            ] {
                add_qwen36_fp8_linear(
                    &mut expected,
                    &format!("{layer_prefix}.linear_attn.{projection}"),
                    rows,
                    columns,
                )?;
            }

            for projection in ["in_proj_a", "in_proj_b"] {
                add_expected(
                    &mut expected,
                    format!("{layer_prefix}.linear_attn.{projection}.weight"),
                    DType::Bf16,
                    vec![gdn_control_rows, hidden],
                )?;
            }
            for name in ["A_log", "dt_bias"] {
                add_expected(
                    &mut expected,
                    format!("{layer_prefix}.linear_attn.{name}"),
                    DType::Bf16,
                    vec![gdn_control_rows],
                )?;
            }
            add_expected(
                &mut expected,
                format!("{layer_prefix}.linear_attn.conv1d.weight"),
                DType::Bf16,
                vec![gdn_input_rows - gdn_value_rows, 1, convolution_width],
            )?;
            add_expected(
                &mut expected,
                format!("{layer_prefix}.linear_attn.norm.weight"),
                DType::Bf16,
                vec![linear_head_dim],
            )?;
        }
    }

    add_qwen36_mtp::<A>(&mut expected)?;
    add_qwen35_vision::<A>(&mut expected)?;

    Ok(expected)
}

fn add_qwen36_nvfp4_mlp(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
    hidden: u64,
    intermediate: u64,
) -> CheckpointResult<()> {
    for projection in ["gate_proj", "up_proj"] {
        add_modelopt_linear(
            expected,
            &format!("{prefix}.{projection}"),
            intermediate,
            hidden,
        )?;
    }
    add_modelopt_linear(
        expected,
        &format!("{prefix}.down_proj"),
        hidden,
        intermediate,
    )
}

fn add_qwen36_fp8_linear(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    prefix: &str,
    rows: u64,
    columns: u64,
) -> CheckpointResult<()> {
    add_expected(
        expected,
        format!("{prefix}.input_scale"),
        DType::F32,
        vec![],
    )?;
    add_expected(
        expected,
        format!("{prefix}.weight"),
        DType::Fp8E4M3,
        vec![rows, columns],
    )?;
    add_expected(
        expected,
        format!("{prefix}.weight_scale"),
        DType::F32,
        vec![],
    )
}

fn add_qwen36_mtp<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<()> {
    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let intermediate = dimension(A::INTERMEDIATE, "expert intermediate width")?;
    let gate_up_rows = intermediate
        .checked_mul(2)
        .ok_or_else(|| CheckpointError::inventory("MTP expert gate/up rows overflow"))?;
    let experts = 256u64;

    add_expected(
        expected,
        "mtp.fc.weight",
        DType::Bf16,
        vec![
            hidden,
            hidden
                .checked_mul(2)
                .ok_or_else(|| CheckpointError::inventory("MTP input width overflows"))?,
        ],
    )?;
    for name in [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        add_expected(expected, name, DType::Bf16, vec![hidden])?;
    }

    let prefix = "mtp.layers.0";
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        add_expected(
            expected,
            format!("{prefix}.{name}"),
            DType::Bf16,
            vec![hidden],
        )?;
    }
    for (projection, rows, columns) in [
        (
            "q_proj",
            dimension(A::ATTENTION_QUERY_ROWS, "attention query rows")?,
            hidden,
        ),
        (
            "k_proj",
            dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?,
            hidden,
        ),
        (
            "v_proj",
            dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?,
            hidden,
        ),
        (
            "o_proj",
            hidden,
            dimension(A::ATTENTION_OUTPUT_COLUMNS, "attention output columns")?,
        ),
    ] {
        add_expected(
            expected,
            format!("{prefix}.self_attn.{projection}.weight"),
            DType::Bf16,
            vec![rows, columns],
        )?;
    }
    for name in ["q_norm.weight", "k_norm.weight"] {
        add_expected(
            expected,
            format!("{prefix}.self_attn.{name}"),
            DType::Bf16,
            vec![dimension(A::HEAD_DIM, "attention head width")?],
        )?;
    }
    add_expected(
        expected,
        format!("{prefix}.mlp.gate.weight"),
        DType::Bf16,
        vec![experts, hidden],
    )?;
    add_expected(
        expected,
        format!("{prefix}.mlp.experts.gate_up_proj"),
        DType::Bf16,
        vec![experts, gate_up_rows, hidden],
    )?;
    add_expected(
        expected,
        format!("{prefix}.mlp.experts.down_proj"),
        DType::Bf16,
        vec![experts, hidden, intermediate],
    )?;
    add_expected(
        expected,
        format!("{prefix}.mlp.shared_expert_gate.weight"),
        DType::Bf16,
        vec![1, hidden],
    )?;
    for (projection, rows, columns) in [
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ] {
        add_expected(
            expected,
            format!("{prefix}.mlp.shared_expert.{projection}.weight"),
            DType::Bf16,
            vec![rows, columns],
        )?;
    }

    Ok(())
}

fn qwen35_expected_tensors<A: Arch>() -> CheckpointResult<BTreeMap<String, ExpectedTensor>> {
    if A::FULL_ATTENTION_INTERVAL == 0 {
        return Err(CheckpointError::inventory(
            "Qwen3.5 full-attention interval must be nonzero",
        ));
    }

    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let intermediate = dimension(A::INTERMEDIATE, "intermediate width")?;
    let attention_query_rows = dimension(A::ATTENTION_QUERY_ROWS, "attention query rows")?;
    let attention_kv_rows = dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?;
    let attention_output_columns =
        dimension(A::ATTENTION_OUTPUT_COLUMNS, "attention output columns")?;
    let gdn_control_rows = dimension(A::GDN_CONTROL_ROWS, "GDN control rows")?;
    let gdn_input_rows = dimension(A::GDN_INPUT_ROWS, "GDN input rows")?;
    let gdn_value_rows = dimension(A::GDN_VALUE_ROWS, "GDN value rows")?;
    let linear_head_dim = dimension(A::LINEAR_HEAD_DIM, "linear-attention head width")?;
    let convolution_width = dimension(A::LINEAR_CONV_KERNEL_DIM, "convolution width")?;
    let mut expected = BTreeMap::new();

    add_expected(
        &mut expected,
        "model.language_model.embed_tokens.weight",
        DType::Bf16,
        vec![dimension(A::VOCAB, "vocabulary size")?, hidden],
    )?;
    add_expected(
        &mut expected,
        "model.language_model.norm.weight",
        DType::Bf16,
        vec![hidden],
    )?;
    add_expected(
        &mut expected,
        "lm_head.weight",
        DType::Bf16,
        vec![dimension(A::VOCAB, "vocabulary size")?, hidden],
    )?;

    for layer in 0..A::LAYERS {
        let layer_prefix = format!("model.language_model.layers.{layer}");

        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            add_expected(
                &mut expected,
                format!("{layer_prefix}.{name}"),
                DType::Bf16,
                vec![hidden],
            )?;
        }

        for projection in ["gate_proj", "up_proj"] {
            add_modelopt_linear(
                &mut expected,
                &format!("{layer_prefix}.mlp.{projection}"),
                intermediate,
                hidden,
            )?;
        }
        add_modelopt_linear(
            &mut expected,
            &format!("{layer_prefix}.mlp.down_proj"),
            hidden,
            intermediate,
        )?;

        if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            for (projection, rows, columns) in [
                ("q_proj", attention_query_rows, hidden),
                ("k_proj", attention_kv_rows, hidden),
                ("v_proj", attention_kv_rows, hidden),
                ("o_proj", hidden, attention_output_columns),
            ] {
                add_modelopt_linear(
                    &mut expected,
                    &format!("{layer_prefix}.self_attn.{projection}"),
                    rows,
                    columns,
                )?;
            }

            for name in ["q_norm.weight", "k_norm.weight"] {
                add_expected(
                    &mut expected,
                    format!("{layer_prefix}.self_attn.{name}"),
                    DType::Bf16,
                    vec![dimension(A::HEAD_DIM, "attention head width")?],
                )?;
            }
        } else {
            for (projection, rows, columns) in [
                ("in_proj_a", gdn_control_rows, hidden),
                ("in_proj_b", gdn_control_rows, hidden),
                ("in_proj_qkv", gdn_input_rows - gdn_value_rows, hidden),
                ("in_proj_z", gdn_value_rows, hidden),
                ("out_proj", hidden, gdn_value_rows),
            ] {
                add_modelopt_linear(
                    &mut expected,
                    &format!("{layer_prefix}.linear_attn.{projection}"),
                    rows,
                    columns,
                )?;
            }

            for name in ["A_log", "dt_bias"] {
                add_expected(
                    &mut expected,
                    format!("{layer_prefix}.linear_attn.{name}"),
                    DType::Bf16,
                    vec![gdn_control_rows],
                )?;
            }
            add_expected(
                &mut expected,
                format!("{layer_prefix}.linear_attn.conv1d.weight"),
                DType::Bf16,
                vec![gdn_input_rows - gdn_value_rows, 1, convolution_width],
            )?;
            add_expected(
                &mut expected,
                format!("{layer_prefix}.linear_attn.norm.weight"),
                DType::Bf16,
                vec![linear_head_dim],
            )?;
        }
    }

    add_qwen35_mtp::<A>(&mut expected)?;
    add_qwen35_vision::<A>(&mut expected)?;

    Ok(expected)
}

fn add_qwen35_mtp<A: Arch>(
    expected: &mut BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<()> {
    let hidden = dimension(A::HIDDEN, "hidden width")?;
    let intermediate = dimension(A::INTERMEDIATE, "intermediate width")?;
    let attention_query_rows = dimension(A::ATTENTION_QUERY_ROWS, "attention query rows")?;
    let attention_kv_rows = dimension(A::ATTENTION_KV_ROWS, "attention KV rows")?;
    let attention_output_columns =
        dimension(A::ATTENTION_OUTPUT_COLUMNS, "attention output columns")?;

    add_expected(
        expected,
        "mtp.fc.weight",
        DType::Bf16,
        vec![
            hidden,
            hidden
                .checked_mul(2)
                .ok_or_else(|| CheckpointError::inventory("MTP input width overflows"))?,
        ],
    )?;
    for name in [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        add_expected(expected, name, DType::Bf16, vec![hidden])?;
    }

    for layer in 0..A::MTP_LAYERS {
        let prefix = format!("mtp.layers.{layer}");

        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            add_expected(
                expected,
                format!("{prefix}.{name}"),
                DType::Bf16,
                vec![hidden],
            )?;
        }
        for (projection, rows, columns) in [
            ("q_proj", attention_query_rows, hidden),
            ("k_proj", attention_kv_rows, hidden),
            ("v_proj", attention_kv_rows, hidden),
            ("o_proj", hidden, attention_output_columns),
        ] {
            add_expected(
                expected,
                format!("{prefix}.self_attn.{projection}.weight"),
                DType::Bf16,
                vec![rows, columns],
            )?;
        }
        for name in ["q_norm.weight", "k_norm.weight"] {
            add_expected(
                expected,
                format!("{prefix}.self_attn.{name}"),
                DType::Bf16,
                vec![dimension(A::HEAD_DIM, "attention head width")?],
            )?;
        }
        for (projection, rows, columns) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            add_expected(
                expected,
                format!("{prefix}.mlp.{projection}.weight"),
                DType::Bf16,
                vec![rows, columns],
            )?;
        }
    }

    Ok(())
}

fn add_qwen35_vision<A: Arch>(
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

fn add_modelopt_linear(
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

fn add_expected(
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

fn validate_exact_tensors(
    file: &SafeTensorFile,
    expected: &BTreeMap<String, ExpectedTensor>,
) -> CheckpointResult<()> {
    for (name, descriptor) in expected {
        validate_expected_tensor(file, name, descriptor)?;
    }

    Ok(())
}

fn validate_expected_tensor(
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

fn dimension(value: usize, field: &str) -> CheckpointResult<u64> {
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

fn validate_revision<A: Arch>(root: &Path) -> CheckpointResult<()> {
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

fn validate_file_length(path: &Path, expected: u64) -> CheckpointResult<()> {
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

fn read_index(path: &Path) -> CheckpointResult<Index> {
    let bytes = fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;

    serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))
}

fn validate_weight_map(
    index_path: &Path,
    weight_map: BTreeMap<String, String>,
    model: &SafeTensorFile,
    mtp: &SafeTensorFile,
    spec: SplitInventorySpec,
) -> CheckpointResult<BTreeMap<String, Shard>> {
    let mut tensors = BTreeMap::new();
    let mut model_entries = 0;
    let mut mtp_entries = 0;

    for (name, file) in weight_map {
        let shard = match file.as_str() {
            MODEL_FILE => {
                model.tensor(&name)?;
                model_entries += 1;
                Shard(0)
            }
            MTP_FILE => {
                mtp.tensor(&name)?;
                mtp_entries += 1;
                Shard(1)
            }
            _ => {
                return Err(CheckpointError::inventory(format!(
                    "{} maps tensor `{name}` to unsupported shard `{file}`",
                    index_path.display()
                )));
            }
        };

        tensors.insert(name, shard);
    }

    require_count(index_path, MODEL_FILE, model_entries, spec.model_tensors)?;
    require_count(index_path, MTP_FILE, mtp_entries, spec.mtp_tensors)?;

    Ok(tensors)
}

fn validate_indexed_weight_map(
    index_path: &Path,
    weight_map: BTreeMap<String, String>,
    shards: &[SafeTensorFile],
    expected: &BTreeMap<String, ExpectedTensor>,
    spec: IndexedInventorySpec,
) -> CheckpointResult<BTreeMap<String, Shard>> {
    let mut tensors = BTreeMap::new();
    let mut shard_entries = vec![0usize; spec.shards.len()];

    for (name, file) in weight_map {
        let shard_index = spec
            .shards
            .iter()
            .position(|shard| shard.file == file)
            .ok_or_else(|| {
                CheckpointError::inventory(format!(
                    "{} maps tensor `{name}` to unsupported shard `{file}`",
                    index_path.display()
                ))
            })?;
        let descriptor = expected.get(&name).ok_or_else(|| {
            CheckpointError::inventory(format!(
                "{} contains unexpected tensor `{name}`",
                index_path.display()
            ))
        })?;

        validate_expected_tensor(&shards[shard_index], &name, descriptor)?;
        shard_entries[shard_index] += 1;
        tensors.insert(name, Shard(shard_index));
    }

    for (index, shard) in spec.shards.iter().enumerate() {
        require_count(index_path, shard.file, shard_entries[index], shard.tensors)?;
    }
    for name in expected.keys() {
        if !tensors.contains_key(name) {
            return Err(CheckpointError::inventory(format!(
                "{} is missing tensor `{name}`",
                index_path.display()
            )));
        }
    }

    Ok(tensors)
}

fn require_count<T>(path: &Path, field: &str, actual: T, expected: T) -> CheckpointResult<()>
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

#[cfg(test)]
mod tests {
    use super::{
        CheckpointSnapshot, ExpectedTensor, INDEX_FILE, MODEL_FILE, MTP_FILE, QWEN35_TENSORS,
        QWEN36_INVENTORY, SplitInventorySpec, qwen35_expected_tensors, qwen36_expected_tensors,
        validate_exact_tensors,
    };
    use crate::config::test_quantization_config;
    use crate::{Arch, CheckpointErrorCode, DType, Qwen35_9B, Qwen36Moe35B, SafeTensorFile};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 1;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 1;
        const VOCAB: usize = 1;
        const LAYERS: usize = 64;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 1;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 1;
        const VISION_HIDDEN: usize = 1;
        const VISION_INTERMEDIATE: usize = 1;
        const VISION_NUM_HEADS: usize = 1;
        const VISION_POSITIONS: usize = 1;
        const VISION_OUTPUT_HIDDEN: usize = 1;
        const VISION_INPUT_CHANNELS: usize = 1;
        const VISION_PATCH_SIZE: usize = 1;
        const VISION_SPATIAL_MERGE_SIZE: usize = 1;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 1;
    }

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        weight_map: BTreeMap<String, String>,
        spec: SplitInventorySpec,
    }

    impl Fixture {
        fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "tuisko-inventory-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            let root = base.join(TestArch::REVISION);
            fs::create_dir_all(&root).unwrap();
            write_config(&root);

            let model_path = root.join(MODEL_FILE);
            let mtp_path = root.join(MTP_FILE);
            write_safetensors(&model_path, &["main.a", "main.b"]);
            write_safetensors(&mtp_path, &["mtp.a"]);

            let model_bytes = fs::metadata(&model_path).unwrap().len();
            let mtp_bytes = fs::metadata(&mtp_path).unwrap().len();
            let total_bytes = model_bytes + mtp_bytes;
            let weight_map = BTreeMap::from([
                ("main.a".to_owned(), MODEL_FILE.to_owned()),
                ("main.b".to_owned(), MODEL_FILE.to_owned()),
                ("mtp.a".to_owned(), MTP_FILE.to_owned()),
            ]);
            let index_bytes = write_index(&root, total_bytes, &weight_map);

            Self {
                base,
                root,
                weight_map,
                spec: SplitInventorySpec {
                    model_bytes,
                    model_tensors: 2,
                    mtp_bytes,
                    mtp_tensors: 1,
                    index_bytes,
                    index_entries: 3,
                },
            }
        }

        fn rewrite_index(&mut self, total_bytes: u64) {
            self.spec.index_bytes = write_index(&self.root, total_bytes, &self.weight_map);
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn write_config(root: &Path) {
        let layer_types = (0usize..64)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let config = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "bfloat16",
            "head_dim": 1,
            "image_token_id": 248056,
            "language_model_only": false,
            "model_type": "qwen3_5",
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "quantization_config": test_quantization_config(),
            "text_config": {
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 1,
                "hidden_size": 1,
                "intermediate_size": 1,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 1,
                "linear_key_head_dim": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_value_head_dim": 1,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 1,
                "num_hidden_layers": 64,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "vocab_size": 1
            },
            "video_token_id": 248057,
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 1,
                "dtype": "bfloat16",
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1,
                "in_channels": 1,
                "initializer_range": 0.02,
                "intermediate_size": 1,
                "model_type": "qwen3_5_vision",
                "num_heads": 1,
                "num_position_embeddings": 1,
                "out_hidden_size": 1,
                "patch_size": 1,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        });

        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, names: &[&str]) {
        let mut header = serde_json::Map::new();

        for (offset, name) in names.iter().enumerate() {
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [offset, offset + 1]
                }),
            );
        }

        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = File::create(path).unwrap();

        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&vec![0; names.len()]).unwrap();
    }

    fn write_index(root: &Path, total_bytes: u64, weight_map: &BTreeMap<String, String>) -> u64 {
        let index = json!({
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map
        });
        let bytes = serde_json::to_vec(&index).unwrap();
        fs::write(root.join(INDEX_FILE), &bytes).unwrap();
        bytes.len() as u64
    }

    #[test]
    fn qwen35_inventory_is_bijective_and_byte_exact() {
        let expected = qwen35_expected_tensors::<Qwen35_9B>().unwrap();
        let mut dtype_counts = BTreeMap::new();
        let mut payload_bytes = 0u64;

        for descriptor in expected.values() {
            *dtype_counts
                .entry(descriptor.dtype.as_str())
                .or_insert(0usize) += 1;
            let elements = descriptor.shape.iter().copied().product::<u64>();
            payload_bytes += elements * descriptor.dtype.byte_width();
        }

        assert_eq!(expected.len(), QWEN35_TENSORS);
        assert_eq!(
            dtype_counts,
            BTreeMap::from([("BF16", 527), ("F32", 496), ("F8_E4M3", 248), ("U8", 248),])
        );
        assert_eq!(payload_bytes, 9_360_860_576);
        assert_eq!(
            expected["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"],
            ExpectedTensor {
                dtype: DType::U8,
                shape: vec![8_192, 2_048],
            }
        );
        assert_eq!(
            expected["model.language_model.layers.3.self_attn.q_proj.weight_scale"],
            ExpectedTensor {
                dtype: DType::Fp8E4M3,
                shape: vec![8_192, 256],
            }
        );
        assert_eq!(
            expected["model.visual.patch_embed.proj.weight"],
            ExpectedTensor {
                dtype: DType::Bf16,
                shape: vec![1_152, 3, 2, 16, 16],
            }
        );
        assert_eq!(
            expected["mtp.fc.weight"],
            ExpectedTensor {
                dtype: DType::Bf16,
                shape: vec![4_096, 8_192],
            }
        );
        assert!(!expected.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(
            !expected.contains_key("model.language_model.layers.3.linear_attn.in_proj_qkv.weight")
        );
    }

    #[test]
    fn qwen36_inventory_is_bijective_and_byte_exact() {
        let expected = qwen36_expected_tensors::<Qwen36Moe35B>().unwrap();
        let mut dtype_counts = BTreeMap::new();
        let mut payload_bytes = 0u64;

        for descriptor in expected.values() {
            *dtype_counts
                .entry(descriptor.dtype.as_str())
                .or_insert(0usize) += 1;
            let elements = descriptor.shape.iter().copied().product::<u64>();
            payload_bytes += elements * descriptor.dtype.byte_width();
        }

        assert_eq!(expected.len(), QWEN36_INVENTORY.index_entries);
        assert_eq!(
            dtype_counts,
            BTreeMap::from([
                ("BF16", 714),
                ("F32", 61_942),
                ("F8_E4M3", 30_971),
                ("U8", 30_841),
            ])
        );
        assert_eq!(payload_bytes, QWEN36_INVENTORY.payload_bytes);
        assert_eq!(
            expected["model.language_model.layers.0.mlp.experts.255.gate_proj.weight"],
            ExpectedTensor {
                dtype: DType::U8,
                shape: vec![512, 1_024],
            }
        );
        assert_eq!(
            expected["model.language_model.layers.3.self_attn.q_proj.weight"],
            ExpectedTensor {
                dtype: DType::Fp8E4M3,
                shape: vec![8_192, 2_048],
            }
        );
        assert_eq!(
            expected["mtp.layers.0.mlp.experts.gate_up_proj"],
            ExpectedTensor {
                dtype: DType::Bf16,
                shape: vec![256, 1_024, 2_048],
            }
        );
        assert_eq!(
            expected["lm_head.weight_scale"],
            ExpectedTensor {
                dtype: DType::Fp8E4M3,
                shape: vec![248_320, 128],
            }
        );
        assert!(!expected.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(
            !expected.contains_key("model.language_model.layers.3.linear_attn.in_proj_qkv.weight")
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_snapshot_inventory_is_byte_exact() {
        let root = std::env::var("TUISKO_QWEN36_SNAPSHOT").unwrap();
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(Path::new(&root)).unwrap();

        assert_eq!(snapshot.tensor_count(), QWEN36_INVENTORY.index_entries);
        assert_eq!(
            snapshot
                .tensor("model.language_model.layers.0.mlp.experts.255.gate_proj.weight")
                .unwrap()
                .shape,
            [512, 1_024]
        );
        assert_eq!(
            snapshot.tensor("lm_head.weight_scale").unwrap().dtype,
            DType::Fp8E4M3
        );
    }

    #[test]
    fn exact_inventory_rejects_descriptor_drift() {
        let fixture = Fixture::new();
        let model = SafeTensorFile::open(&fixture.root.join(MODEL_FILE)).unwrap();
        let expected = BTreeMap::from([(
            String::from("main.a"),
            ExpectedTensor {
                dtype: DType::Bf16,
                shape: vec![1],
            },
        )]);

        let error = validate_exact_tensors(&model, &expected).unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("expected BF16 [1]"));
    }

    #[test]
    fn admits_complete_inventory_and_routes_tensors() {
        let fixture = Fixture::new();

        let snapshot =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .unwrap();

        assert_eq!(snapshot.root(), fixture.root);
        assert_eq!(snapshot.tensor_count(), 3);
        assert_eq!(snapshot.tensor("main.b").unwrap().bytes, &[0]);
        assert_eq!(snapshot.tensor("mtp.a").unwrap().bytes, &[0]);
        assert_eq!(
            snapshot
                .adjacent_tensor_bytes("main.a", "main.b", "main pair")
                .unwrap(),
            &[0, 0]
        );
    }

    #[test]
    fn rejects_tensor_span_across_shards() {
        let fixture = Fixture::new();
        let snapshot =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .unwrap();

        let error = snapshot
            .adjacent_tensor_bytes("main.b", "mtp.a", "cross-shard pair")
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("split across checkpoint shards"));
    }

    #[test]
    fn rejects_file_length_mismatch() {
        let mut fixture = Fixture::new();
        fixture.spec.model_bytes += 1;

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Inventory);
        assert!(error.to_string().contains("model.safetensors has"));
    }

    #[test]
    fn rejects_inventory_count_mismatch() {
        let fixture = Fixture::new();
        let mut index_spec = fixture.spec;
        index_spec.index_entries += 1;
        let mut model_spec = fixture.spec;
        model_spec.model_tensors += 1;
        let mut mtp_spec = fixture.spec;
        mtp_spec.mtp_tensors += 1;

        for (spec, field) in [
            (index_spec, "entries"),
            (model_spec, "model.safetensors tensors"),
            (mtp_spec, "model_mtp.safetensors tensors"),
        ] {
            let error = CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, spec)
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Inventory);
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_index_metadata_mismatch() {
        let mut fixture = Fixture::new();
        fixture.rewrite_index(fixture.spec.model_bytes + fixture.spec.mtp_bytes + 1);

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap()
                .to_string();

        assert!(error.contains("metadata.total_size"));
    }

    #[test]
    fn rejects_wrong_shard_assignment() {
        let mut fixture = Fixture::new();
        fixture
            .weight_map
            .insert("mtp.a".to_owned(), MODEL_FILE.to_owned());
        fixture.rewrite_index(fixture.spec.model_bytes + fixture.spec.mtp_bytes);

        let error =
            CheckpointSnapshot::<TestArch>::open_split_with_spec(&fixture.root, fixture.spec)
                .err()
                .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(
            error
                .to_string()
                .contains("model.safetensors is missing tensor `mtp.a`")
        );
    }

    #[test]
    fn rejects_unpinned_revision_path() {
        let fixture = Fixture::new();
        let root = fixture.base.join("other-revision");

        let error = CheckpointSnapshot::<TestArch>::open_split_with_spec(&root, fixture.spec)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Revision);
        assert!(error.to_string().contains("expected \"test-revision\""));
    }
}

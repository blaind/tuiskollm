//! Qwen3.6-35B-A3B MoE indexed three-shard snapshot inventory admission.

use crate::CheckpointSnapshot;
use crate::common::config_util::validate_config;
use crate::common::inventory::{
    CONFIG_FILE, ExpectedTensor, INDEX_FILE, Shard, add_expected, add_modelopt_linear,
    add_qwen35_vision, dimension, read_index, require_count, validate_expected_tensor,
    validate_file_length, validate_revision,
};
use crate::{Arch, CheckpointError, CheckpointResult, DType, SafeTensorFile};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::Path;

const QWEN36_SHARD_FILES: [&str; 3] = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
];

pub(crate) const QWEN36_INVENTORY: IndexedInventorySpec = IndexedInventorySpec {
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
pub(crate) struct IndexedInventorySpec {
    index_bytes: u64,
    index_entries: usize,
    total_parameters: u64,
    payload_bytes: u64,
    shards: [IndexedShardSpec; 3],
}

#[derive(Clone, Copy)]
pub(crate) struct IndexedShardSpec {
    file: &'static str,
    file_bytes: u64,
    header_bytes: usize,
    tensors: usize,
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

impl<A: Arch> CheckpointSnapshot<A> {
    pub(crate) fn open_indexed(root: &Path, spec: IndexedInventorySpec) -> CheckpointResult<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::inventory::{CheckpointSnapshot, ExpectedTensor};
    use crate::{DType, Qwen36Moe35B};
    use std::collections::BTreeMap;
    use std::path::Path;

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
}

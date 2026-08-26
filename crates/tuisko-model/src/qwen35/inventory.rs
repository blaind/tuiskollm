//! Qwen3.5-9B single-shard snapshot inventory admission.

use crate::CheckpointSnapshot;
use crate::common::config_util::validate_config;
use crate::common::inventory::{
    CONFIG_FILE, ExpectedTensor, MODEL_FILE, Shard, add_expected, add_modelopt_linear,
    add_qwen35_vision, dimension, require_count, validate_exact_tensors, validate_file_length,
    validate_revision,
};
use crate::{Arch, CheckpointError, CheckpointResult, DType, SafeTensorFile};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::Path;

const QWEN35_MODEL_BYTES: u64 = 9_361_048_680;
const QWEN35_HEADER_BYTES: usize = 188_096;
const QWEN35_TENSORS: usize = 1_519;

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

impl<A: Arch> CheckpointSnapshot<A> {
    pub(crate) fn open_modelopt(root: &Path) -> CheckpointResult<Self> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::inventory::ExpectedTensor;
    use crate::{DType, Qwen35_9B};
    use std::collections::BTreeMap;

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
}

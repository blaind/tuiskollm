//! Qwen3.5-9B ModelOpt config admission.

use crate::common::config_util::{
    ARCHITECTURE, DTYPE, FLOAT_TYPE, IMAGE_TOKEN_ID, MODEL_TYPE, MODELOPT_PRODUCER,
    MODELOPT_QUANT_METHOD, ModelOptProducer, TextConfig, VIDEO_TOKEN_ID, VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID, VisionConfig, require, validate_text, validate_vision,
};
use crate::{Arch, CheckpointResult};
use serde::Deserialize;
use std::path::Path;

const MODELOPT_QUANT_ALGO: &str = "NVFP4";

const MODELOPT_PRODUCER_VERSION: &str = "0.0.1.dev1+g82f1d216d";

#[derive(Debug, Deserialize)]
pub(crate) struct ModelOptConfig {
    architectures: Vec<String>,
    dtype: String,
    image_token_id: usize,
    model_type: String,
    quantization_config: ModelOptQuantizationConfig,
    text_config: TextConfig,
    tie_word_embeddings: bool,
    video_token_id: usize,
    vision_config: VisionConfig,
    vision_end_token_id: usize,
    vision_start_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct ModelOptQuantizationConfig {
    config_groups: ModelOptConfigGroups,
    ignore: Vec<String>,
    producer: ModelOptProducer,
    quant_algo: String,
    quant_method: String,
}

#[derive(Debug, Deserialize)]
struct ModelOptConfigGroups {
    group_0: ModelOptQuantizationGroup,
}

#[derive(Debug, Deserialize)]
struct ModelOptQuantizationGroup {
    input_activations: ModelOptQuantizationScheme,
    targets: Vec<String>,
    weights: ModelOptQuantizationScheme,
}

#[derive(Debug, Deserialize)]
struct ModelOptQuantizationScheme {
    dynamic: bool,
    group_size: usize,
    num_bits: usize,
    #[serde(rename = "type")]
    kind: String,
}

pub(crate) fn validate_modelopt<A: Arch>(
    path: &Path,
    config: &ModelOptConfig,
) -> CheckpointResult<()> {
    require(path, "architectures length", config.architectures.len(), 1)?;
    require(
        path,
        "architectures[0]",
        config.architectures[0].as_str(),
        ARCHITECTURE,
    )?;
    require(path, "dtype", config.dtype.as_str(), DTYPE)?;
    require(
        path,
        "image_token_id",
        config.image_token_id,
        IMAGE_TOKEN_ID,
    )?;
    require(path, "model_type", config.model_type.as_str(), MODEL_TYPE)?;
    require(
        path,
        "tie_word_embeddings",
        config.tie_word_embeddings,
        false,
    )?;
    require(
        path,
        "video_token_id",
        config.video_token_id,
        VIDEO_TOKEN_ID,
    )?;
    require(
        path,
        "vision_end_token_id",
        config.vision_end_token_id,
        VISION_END_TOKEN_ID,
    )?;
    require(
        path,
        "vision_start_token_id",
        config.vision_start_token_id,
        VISION_START_TOKEN_ID,
    )?;

    validate_text::<A>(path, &config.text_config)?;
    require(
        path,
        "text_config.tie_word_embeddings",
        config.text_config.tie_word_embeddings,
        Some(false),
    )?;
    validate_vision::<A>(path, &config.vision_config, MODEL_TYPE)?;
    validate_modelopt_quantization::<A>(path, &config.quantization_config)
}

fn validate_modelopt_quantization<A: Arch>(
    path: &Path,
    config: &ModelOptQuantizationConfig,
) -> CheckpointResult<()> {
    require(
        path,
        "quantization_config.quant_algo",
        config.quant_algo.as_str(),
        MODELOPT_QUANT_ALGO,
    )?;
    require(
        path,
        "quantization_config.quant_method",
        config.quant_method.as_str(),
        MODELOPT_QUANT_METHOD,
    )?;
    require(
        path,
        "quantization_config.producer.name",
        config.producer.name.as_str(),
        MODELOPT_PRODUCER,
    )?;
    require(
        path,
        "quantization_config.producer.version",
        config.producer.version.as_str(),
        MODELOPT_PRODUCER_VERSION,
    )?;
    require(
        path,
        "quantization_config.ignore",
        config.ignore.as_slice(),
        modelopt_ignored_targets::<A>().as_slice(),
    )?;

    let group = &config.config_groups.group_0;
    require(
        path,
        "quantization_config.config_groups.group_0.targets",
        group.targets.as_slice(),
        &[String::from("Linear")],
    )?;
    validate_modelopt_scheme(
        path,
        "quantization_config.config_groups.group_0.input_activations",
        &group.input_activations,
    )?;
    validate_modelopt_scheme(
        path,
        "quantization_config.config_groups.group_0.weights",
        &group.weights,
    )
}

fn validate_modelopt_scheme(
    path: &Path,
    field: &str,
    scheme: &ModelOptQuantizationScheme,
) -> CheckpointResult<()> {
    require(path, &format!("{field}.dynamic"), scheme.dynamic, false)?;
    require(path, &format!("{field}.group_size"), scheme.group_size, 16)?;
    require(path, &format!("{field}.num_bits"), scheme.num_bits, 4)?;
    require(
        path,
        &format!("{field}.type"),
        scheme.kind.as_str(),
        FLOAT_TYPE,
    )
}

fn modelopt_ignored_targets<A: Arch>() -> Vec<String> {
    let mut ignored = vec![String::from("lm_head")];

    for layer in 0..A::LAYERS {
        if !(layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            ignored.push(format!(
                "model.language_model.layers.{layer}.linear_attn.conv1d"
            ));
        }
    }

    ignored.extend([String::from("model.visual*"), String::from("mtp.layers.0*")]);
    ignored.sort();
    ignored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config_util::validate_config;
    use crate::common::test_support::configs::write_config;
    use crate::{CheckpointErrorCode, Qwen35_9B};
    use serde_json::{Value, json};
    use std::fs;

    fn valid_modelopt_config() -> Value {
        let layer_types = (0usize..32)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();

        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "bfloat16",
            "image_token_id": 248056,
            "model_type": "qwen3_5",
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": false,
                            "group_size": 16,
                            "num_bits": 4,
                            "type": "float"
                        },
                        "targets": ["Linear"],
                        "weights": {
                            "dynamic": false,
                            "group_size": 16,
                            "num_bits": 4,
                            "type": "float"
                        }
                    }
                },
                "ignore": modelopt_ignored_targets::<Qwen35_9B>(),
                "producer": {
                    "name": "modelopt",
                    "version": "0.0.1.dev1+g82f1d216d"
                },
                "quant_algo": "NVFP4",
                "quant_method": "modelopt"
            },
            "text_config": {
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 16,
                "num_hidden_layers": 32,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "vocab_size": 248320
            },
            "tie_word_embeddings": false,
            "video_token_id": 248057,
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 27,
                "dtype": "bfloat16",
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "model_type": "qwen3_5",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        })
    }

    #[test]
    fn admits_exact_qwen35_modelopt_contract() {
        let path = write_config("valid-qwen35-modelopt-config", &valid_modelopt_config());

        validate_config::<Qwen35_9B>(&path).unwrap();

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_qwen35_modelopt_geometry_and_route_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen35-hidden",
                "/text_config/hidden_size",
                json!(5_120),
                "text_config.hidden_size",
            ),
            (
                "qwen35-layer-route",
                "/text_config/layer_types/3",
                json!("linear_attention"),
                "text_config.layer_types[3]",
            ),
            (
                "qwen35-vision-output",
                "/vision_config/out_hidden_size",
                json!(5_120),
                "vision_config.out_hidden_size",
            ),
        ] {
            let mut config = valid_modelopt_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let path = write_config(label, &config);

            let error = validate_config::<Qwen35_9B>(&path).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_qwen35_modelopt_quantization_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen35-quant-algo",
                "/quantization_config/quant_algo",
                json!("other"),
                "quantization_config.quant_algo",
            ),
            (
                "qwen35-producer-version",
                "/quantization_config/producer/version",
                json!("other"),
                "quantization_config.producer.version",
            ),
            (
                "qwen35-target",
                "/quantization_config/config_groups/group_0/targets/0",
                json!("Conv1D"),
                "group_0.targets",
            ),
            (
                "qwen35-weight-group",
                "/quantization_config/config_groups/group_0/weights/group_size",
                json!(32),
                "group_0.weights.group_size",
            ),
            (
                "qwen35-activation-dynamic",
                "/quantization_config/config_groups/group_0/input_activations/dynamic",
                json!(true),
                "group_0.input_activations.dynamic",
            ),
            (
                "qwen35-ignore",
                "/quantization_config/ignore/0",
                json!("other"),
                "quantization_config.ignore",
            ),
        ] {
            let mut config = valid_modelopt_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let path = write_config(label, &config);

            let error = validate_config::<Qwen35_9B>(&path).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");

            fs::remove_file(path).unwrap();
        }
    }
}

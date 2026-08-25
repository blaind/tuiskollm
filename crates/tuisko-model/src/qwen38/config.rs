//! Qwen3.8-27B compressed-tensors config admission.

use crate::common::config_util::{
    ARCHITECTURE, DTYPE, FLOAT_TYPE, IMAGE_TOKEN_ID, MODEL_TYPE, TextConfig, VIDEO_TOKEN_ID,
    VISION_END_TOKEN_ID, VISION_START_TOKEN_ID, VisionConfig, fp8_targets, require, validate_text,
    validate_vision,
};
use crate::{Arch, CheckpointResult};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

const VISION_MODEL_TYPE: &str = "qwen3_5_vision";

const MIXED_PRECISION_FORMAT: &str = "mixed-precision";
const QUANT_METHOD: &str = "compressed-tensors";
const QUANTIZATION_STATUS: &str = "compressed";
const FP8_FORMAT: &str = "float-quantized";
const NVFP4_FORMAT: &str = "nvfp4-pack-quantized";

const E4M3FN_DTYPE: &str = "torch.float8_e4m3fn";

#[derive(Debug, Deserialize)]
pub(crate) struct ModelConfig {
    architectures: Vec<String>,
    dtype: String,
    head_dim: usize,
    image_token_id: usize,
    language_model_only: bool,
    model_type: String,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    quantization_config: QuantizationConfig,
    text_config: TextConfig,
    video_token_id: usize,
    vision_config: VisionConfig,
    vision_end_token_id: usize,
    vision_start_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    config_groups: ConfigGroups,
    format: String,
    quant_method: String,
    quantization_status: String,
}

#[derive(Debug, Deserialize)]
struct ConfigGroups {
    group_0: QuantizationGroup,
    group_1: QuantizationGroup,
}

#[derive(Debug, Deserialize)]
struct QuantizationGroup {
    format: String,
    input_activations: QuantizationScheme,
    targets: Vec<String>,
    weights: QuantizationScheme,
}

#[derive(Debug, Deserialize)]
struct QuantizationScheme {
    dynamic: Value,
    group_size: Option<usize>,
    num_bits: usize,
    scale_dtype: Option<String>,
    strategy: String,
    symmetric: bool,
    #[serde(rename = "type")]
    kind: String,
}

struct SchemeContract<'a> {
    dynamic: Value,
    group_size: Option<usize>,
    num_bits: usize,
    scale_dtype: Option<&'a str>,
    strategy: &'a str,
}

pub(crate) fn validate_compressed<A: Arch>(
    path: &Path,
    config: &ModelConfig,
) -> CheckpointResult<()> {
    require(path, "architectures length", config.architectures.len(), 1)?;

    require(
        path,
        "architectures[0]",
        config.architectures[0].as_str(),
        ARCHITECTURE,
    )?;
    require(path, "dtype", config.dtype.as_str(), DTYPE)?;
    require(path, "head_dim", config.head_dim, A::HEAD_DIM)?;
    require(
        path,
        "image_token_id",
        config.image_token_id,
        IMAGE_TOKEN_ID,
    )?;
    require(
        path,
        "language_model_only",
        config.language_model_only,
        false,
    )?;
    require(path, "model_type", config.model_type.as_str(), MODEL_TYPE)?;
    require(
        path,
        "num_attention_heads",
        config.num_attention_heads,
        A::NUM_ATTENTION_HEADS,
    )?;
    require(
        path,
        "num_key_value_heads",
        config.num_key_value_heads,
        A::NUM_KV_HEADS,
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
    validate_vision::<A>(path, &config.vision_config, VISION_MODEL_TYPE)?;
    validate_compressed_quantization::<A>(path, &config.quantization_config)
}

fn validate_compressed_quantization<A: Arch>(
    path: &Path,
    config: &QuantizationConfig,
) -> CheckpointResult<()> {
    require(
        path,
        "quantization_config.format",
        config.format.as_str(),
        MIXED_PRECISION_FORMAT,
    )?;
    require(
        path,
        "quantization_config.quant_method",
        config.quant_method.as_str(),
        QUANT_METHOD,
    )?;
    require(
        path,
        "quantization_config.quantization_status",
        config.quantization_status.as_str(),
        QUANTIZATION_STATUS,
    )?;

    let fp8 = &config.config_groups.group_0;
    require(
        path,
        "quantization_config.config_groups.group_0.format",
        fp8.format.as_str(),
        FP8_FORMAT,
    )?;
    let fp8_targets = fp8_targets::<A>();
    require(
        path,
        "quantization_config.config_groups.group_0.targets",
        fp8.targets.as_slice(),
        fp8_targets.as_slice(),
    )?;
    validate_scheme(
        path,
        "quantization_config.config_groups.group_0.input_activations",
        &fp8.input_activations,
        SchemeContract {
            dynamic: Value::Bool(true),
            group_size: None,
            num_bits: 8,
            scale_dtype: None,
            strategy: "token",
        },
    )?;
    validate_scheme(
        path,
        "quantization_config.config_groups.group_0.weights",
        &fp8.weights,
        SchemeContract {
            dynamic: Value::Bool(false),
            group_size: None,
            num_bits: 8,
            scale_dtype: None,
            strategy: "channel",
        },
    )?;

    let nvfp4 = &config.config_groups.group_1;
    require(
        path,
        "quantization_config.config_groups.group_1.format",
        nvfp4.format.as_str(),
        NVFP4_FORMAT,
    )?;
    let nvfp4_targets = vec![String::from(r"re:.*mlp\.(gate|up|down)_proj$")];
    require(
        path,
        "quantization_config.config_groups.group_1.targets",
        nvfp4.targets.as_slice(),
        nvfp4_targets.as_slice(),
    )?;
    validate_scheme(
        path,
        "quantization_config.config_groups.group_1.input_activations",
        &nvfp4.input_activations,
        SchemeContract {
            dynamic: Value::String(String::from("local")),
            group_size: Some(16),
            num_bits: 4,
            scale_dtype: Some(E4M3FN_DTYPE),
            strategy: "tensor_group",
        },
    )?;
    validate_scheme(
        path,
        "quantization_config.config_groups.group_1.weights",
        &nvfp4.weights,
        SchemeContract {
            dynamic: Value::Bool(false),
            group_size: Some(16),
            num_bits: 4,
            scale_dtype: Some(E4M3FN_DTYPE),
            strategy: "tensor_group",
        },
    )
}

fn validate_scheme(
    path: &Path,
    field: &str,
    actual: &QuantizationScheme,
    expected: SchemeContract<'_>,
) -> CheckpointResult<()> {
    require(
        path,
        &format!("{field}.dynamic"),
        &actual.dynamic,
        &expected.dynamic,
    )?;
    require(
        path,
        &format!("{field}.group_size"),
        actual.group_size,
        expected.group_size,
    )?;
    require(
        path,
        &format!("{field}.num_bits"),
        actual.num_bits,
        expected.num_bits,
    )?;
    require(
        path,
        &format!("{field}.scale_dtype"),
        actual.scale_dtype.as_deref(),
        expected.scale_dtype,
    )?;
    require(
        path,
        &format!("{field}.strategy"),
        actual.strategy.as_str(),
        expected.strategy,
    )?;
    require(path, &format!("{field}.symmetric"), actual.symmetric, true)?;
    require(
        path,
        &format!("{field}.type"),
        actual.kind.as_str(),
        FLOAT_TYPE,
    )
}

#[cfg(test)]
mod tests {

    use crate::common::config_util::{test_quantization_config, validate_config};
    use crate::common::test_support::configs::write_config;
    use crate::{CheckpointErrorCode, Qwen38_27B};
    use serde_json::{Value, json};
    use std::fs;

    fn valid_config() -> Value {
        let layer_types = (0usize..64)
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
            "head_dim": 256,
            "image_token_id": 248056,
            "language_model_only": false,
            "model_type": "qwen3_5",
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "quantization_config": test_quantization_config(),
            "text_config": {
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "vocab_size": 248320
            },
            "video_token_id": 248057,
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 27,
                "dtype": "bfloat16",
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4304,
                "model_type": "qwen3_5_vision",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        })
    }

    #[test]
    fn admits_exact_target_geometry() {
        let path = write_config("valid-config", &valid_config());

        validate_config::<Qwen38_27B>(&path).unwrap();

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_target_config_identity() {
        let mut cases = Vec::new();

        let mut config = valid_config();
        config["architectures"][0] = json!("OtherArchitecture");
        cases.push(("architecture", config, "architectures[0]"));

        let mut config = valid_config();
        config["dtype"] = json!("float16");
        cases.push(("dtype", config, "field `dtype`"));

        let mut config = valid_config();
        config["language_model_only"] = json!(true);
        cases.push(("language-model-only", config, "language_model_only"));

        let mut config = valid_config();
        config["model_type"] = json!("other");
        cases.push(("model-type", config, "field `model_type`"));

        let mut config = valid_config();
        config["text_config"]["dtype"] = json!("float16");
        cases.push(("text-dtype", config, "text_config.dtype"));

        let mut config = valid_config();
        config["text_config"]["model_type"] = json!("other");
        cases.push(("text-model-type", config, "text_config.model_type"));

        let mut config = valid_config();
        config["text_config"]["tie_word_embeddings"] = json!(true);
        cases.push((
            "text-tied-embeddings",
            config,
            "text_config.tie_word_embeddings",
        ));

        let mut config = valid_config();
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        cases.push((
            "text-tied-embeddings-missing",
            config,
            "text_config.tie_word_embeddings",
        ));

        let mut config = valid_config();
        config["vision_config"]["dtype"] = json!("float16");
        cases.push(("vision-dtype", config, "vision_config.dtype"));

        let mut config = valid_config();
        config["vision_config"]["hidden_act"] = json!("other");
        cases.push(("vision-hidden-act", config, "vision_config.hidden_act"));

        let mut config = valid_config();
        config["vision_config"]["model_type"] = json!("other");
        cases.push(("vision-model-type", config, "vision_config.model_type"));

        let mut config = valid_config();
        config["vision_config"]["deepstack_visual_indexes"] = json!([3, 7, 15]);
        cases.push((
            "vision-deepstack",
            config,
            "vision_config.deepstack_visual_indexes",
        ));

        for (label, config, field) in cases {
            let path = write_config(label, &config);

            let error = validate_config::<Qwen38_27B>(&path).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_architecture_cardinality_with_field_context() {
        for (label, architectures, count) in [
            ("no-architectures", json!([]), 0),
            (
                "multiple-architectures",
                json!(["Qwen3_5ForConditionalGeneration", "OtherArchitecture"]),
                2,
            ),
        ] {
            let mut config = valid_config();
            config["architectures"] = architectures;
            let path = write_config(label, &config);

            let error = validate_config::<Qwen38_27B>(&path).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(
                error.to_string().contains(&format!(
                    "config field `architectures length` is {count}, expected 1"
                )),
                "{error}"
            );

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_every_geometry_mismatch() {
        for field in ["head_dim", "num_attention_heads", "num_key_value_heads"] {
            let mut config = valid_config();
            config[field] = json!(1);
            let path = write_config(field, &config);

            let error = validate_config::<Qwen38_27B>(&path)
                .err()
                .unwrap()
                .to_string();

            assert!(error.contains(&format!("field `{field}`")), "{error}");

            fs::remove_file(path).unwrap();
        }

        for field in [
            "full_attention_interval",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "vocab_size",
        ] {
            let mut config = valid_config();
            config["text_config"][field] = json!(1);
            let path = write_config(field, &config);

            let error = validate_config::<Qwen38_27B>(&path)
                .err()
                .unwrap()
                .to_string();

            assert!(error.contains(&format!("text_config.{field}")), "{error}");

            fs::remove_file(path).unwrap();
        }

        let mut config = valid_config();
        config["text_config"]["mtp_num_hidden_layers"] = json!(2);
        let path = write_config("mtp_num_hidden_layers", &config);
        let error = validate_config::<Qwen38_27B>(&path)
            .err()
            .unwrap()
            .to_string();

        assert!(
            error.contains("text_config.mtp_num_hidden_layers"),
            "{error}"
        );

        fs::remove_file(path).unwrap();

        let mut config = valid_config();
        config["text_config"]["mtp_use_dedicated_embeddings"] = json!(true);
        let path = write_config("mtp_use_dedicated_embeddings", &config);
        let error = validate_config::<Qwen38_27B>(&path)
            .err()
            .unwrap()
            .to_string();

        assert!(
            error.contains("text_config.mtp_use_dedicated_embeddings"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_vision_token_and_geometry_mismatches() {
        for field in [
            "image_token_id",
            "video_token_id",
            "vision_end_token_id",
            "vision_start_token_id",
        ] {
            let mut config = valid_config();
            config[field] = json!(1);
            let path = write_config(field, &config);

            let error = validate_config::<Qwen38_27B>(&path)
                .err()
                .unwrap()
                .to_string();

            assert!(error.contains(&format!("field `{field}`")), "{error}");

            fs::remove_file(path).unwrap();
        }

        for field in [
            "depth",
            "hidden_size",
            "in_channels",
            "intermediate_size",
            "num_heads",
            "num_position_embeddings",
            "out_hidden_size",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        ] {
            let mut config = valid_config();
            config["vision_config"][field] = json!(1);
            let path = write_config(field, &config);

            let error = validate_config::<Qwen38_27B>(&path)
                .err()
                .unwrap()
                .to_string();

            assert!(error.contains(&format!("vision_config.{field}")), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_layer_route_mismatch() {
        for (layer, replacement) in [
            (0, "full_attention"),
            (3, "linear_attention"),
            (63, "linear_attention"),
        ] {
            let mut config = valid_config();
            config["text_config"]["layer_types"][layer] = json!(replacement);
            let path = write_config(&format!("bad-layer-route-{layer}"), &config);

            let error = validate_config::<Qwen38_27B>(&path)
                .err()
                .unwrap()
                .to_string();

            assert!(
                error.contains(&format!("text_config.layer_types[{layer}]")),
                "{error}"
            );

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_layer_route_count_mismatch() {
        let mut config = valid_config();
        config["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .pop();
        let path = write_config("bad-layer-route-count", &config);

        let error = validate_config::<Qwen38_27B>(&path)
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("text_config.layer_types length"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_quantization_route_and_codec_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "quant-format",
                "/quantization_config/format",
                json!("other"),
                "quantization_config.format",
            ),
            (
                "fp8-layer-route",
                "/quantization_config/config_groups/group_0/targets/3",
                json!(r"re:.*layers\.(55|56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$"),
                "group_0.targets",
            ),
            (
                "nvfp4-route",
                "/quantization_config/config_groups/group_1/targets/0",
                json!(r"re:.*layers\.0\.mlp\.(gate|up|down)_proj$"),
                "group_1.targets",
            ),
            (
                "fp8-weight-bits",
                "/quantization_config/config_groups/group_0/weights/num_bits",
                json!(4),
                "group_0.weights.num_bits",
            ),
            (
                "nvfp4-group-size",
                "/quantization_config/config_groups/group_1/weights/group_size",
                json!(32),
                "group_1.weights.group_size",
            ),
            (
                "nvfp4-scale-dtype",
                "/quantization_config/config_groups/group_1/input_activations/scale_dtype",
                json!("torch.float16"),
                "group_1.input_activations.scale_dtype",
            ),
        ] {
            let mut config = valid_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let path = write_config(label, &config);

            let error = validate_config::<Qwen38_27B>(&path).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn admits_observer_only_quantization_metadata_change() {
        let mut config = valid_config();
        config["quantization_config"]["config_groups"]["group_1"]["weights"]["observer"] =
            json!("imatrix_mse");
        let path = write_config("observer-metadata", &config);

        validate_config::<Qwen38_27B>(&path).unwrap();

        fs::remove_file(path).unwrap();
    }
}

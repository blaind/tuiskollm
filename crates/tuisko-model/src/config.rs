use crate::bindings::NVFP4_MLP_LAYER_END;
use crate::{Arch, CheckpointError, CheckpointResult};
use serde::Deserialize;
use serde_json::Value;
use std::fmt::Debug;
use std::fs;
use std::path::Path;

const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const MODEL_TYPE: &str = "qwen3_5";
const TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const DTYPE: &str = "bfloat16";
const MIXED_PRECISION_FORMAT: &str = "mixed-precision";
const QUANT_METHOD: &str = "compressed-tensors";
const QUANTIZATION_STATUS: &str = "compressed";
const FP8_FORMAT: &str = "float-quantized";
const NVFP4_FORMAT: &str = "nvfp4-pack-quantized";
const FLOAT_TYPE: &str = "float";
const E4M3FN_DTYPE: &str = "torch.float8_e4m3fn";

#[derive(Debug, Deserialize)]
struct ModelConfig {
    architectures: Vec<String>,
    dtype: String,
    head_dim: usize,
    language_model_only: bool,
    model_type: String,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    quantization_config: QuantizationConfig,
    text_config: TextConfig,
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

#[derive(Debug, Deserialize)]
struct TextConfig {
    dtype: String,
    full_attention_interval: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
}

/// Validates a checkpoint config against the selected architecture.
pub fn validate_config<A: Arch>(path: &Path) -> CheckpointResult<()> {
    let bytes = fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;

    let config: ModelConfig =
        serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))?;

    validate::<A>(path, &config)
}

fn validate<A: Arch>(path: &Path, config: &ModelConfig) -> CheckpointResult<()> {
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

    let text = &config.text_config;

    require(path, "text_config.dtype", text.dtype.as_str(), DTYPE)?;
    require(
        path,
        "text_config.full_attention_interval",
        text.full_attention_interval,
        A::FULL_ATTENTION_INTERVAL,
    )?;
    require(path, "text_config.head_dim", text.head_dim, A::HEAD_DIM)?;
    require(path, "text_config.hidden_size", text.hidden_size, A::HIDDEN)?;
    require(
        path,
        "text_config.intermediate_size",
        text.intermediate_size,
        A::INTERMEDIATE,
    )?;
    require(
        path,
        "text_config.linear_conv_kernel_dim",
        text.linear_conv_kernel_dim,
        A::LINEAR_CONV_KERNEL_DIM,
    )?;
    require(
        path,
        "text_config.linear_key_head_dim",
        text.linear_key_head_dim,
        A::LINEAR_HEAD_DIM,
    )?;
    require(
        path,
        "text_config.linear_num_key_heads",
        text.linear_num_key_heads,
        A::LINEAR_KEY_HEADS,
    )?;
    require(
        path,
        "text_config.linear_num_value_heads",
        text.linear_num_value_heads,
        A::LINEAR_VALUE_HEADS,
    )?;
    require(
        path,
        "text_config.linear_value_head_dim",
        text.linear_value_head_dim,
        A::LINEAR_HEAD_DIM,
    )?;
    require(
        path,
        "text_config.model_type",
        text.model_type.as_str(),
        TEXT_MODEL_TYPE,
    )?;
    require(
        path,
        "text_config.num_attention_heads",
        text.num_attention_heads,
        A::NUM_ATTENTION_HEADS,
    )?;
    require(
        path,
        "text_config.num_hidden_layers",
        text.num_hidden_layers,
        A::LAYERS,
    )?;
    require(
        path,
        "text_config.num_key_value_heads",
        text.num_key_value_heads,
        A::NUM_KV_HEADS,
    )?;
    require(path, "text_config.vocab_size", text.vocab_size, A::VOCAB)?;

    validate_layer_types::<A>(path, &text.layer_types)?;
    validate_quantization::<A>(path, &config.quantization_config)
}

fn validate_layer_types<A: Arch>(path: &Path, layer_types: &[String]) -> CheckpointResult<()> {
    require(
        path,
        "text_config.layer_types length",
        layer_types.len(),
        A::LAYERS,
    )?;

    if A::FULL_ATTENTION_INTERVAL == 0 {
        return Err(invalid_field(
            path,
            "text_config.full_attention_interval",
            A::FULL_ATTENTION_INTERVAL,
            "a nonzero interval",
        ));
    }

    for (layer, actual) in layer_types.iter().enumerate() {
        let expected = if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            "full_attention"
        } else {
            "linear_attention"
        };

        if actual != expected {
            return Err(invalid_field(
                path,
                &format!("text_config.layer_types[{layer}]"),
                actual,
                expected,
            ));
        }
    }

    Ok(())
}

fn validate_quantization<A: Arch>(
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

fn fp8_targets<A: Arch>() -> Vec<String> {
    let late_layers = (NVFP4_MLP_LAYER_END..A::LAYERS)
        .map(|layer| layer.to_string())
        .collect::<Vec<_>>()
        .join("|");

    vec![
        String::from(r"re:.*self_attn\.(q|k|v|o)_proj$"),
        String::from(r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$"),
        String::from(r"re:.*lm_head"),
        format!(r"re:.*layers\.({late_layers})\.mlp\.(gate|up|down)_proj$"),
    ]
}

#[cfg(test)]
pub(crate) fn test_quantization_config() -> Value {
    serde_json::from_str(include_str!("../fixtures/quantization-config.json")).unwrap()
}

fn require<T>(path: &Path, field: &str, actual: T, expected: T) -> CheckpointResult<()>
where
    T: Debug + PartialEq,
{
    if actual != expected {
        return Err(invalid_field(path, field, actual, expected));
    }

    Ok(())
}

fn invalid_field(
    path: &Path,
    field: &str,
    actual: impl Debug,
    expected: impl Debug,
) -> CheckpointError {
    CheckpointError::config(format!(
        "{} config field `{field}` is {actual:?}, expected {expected:?}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::{test_quantization_config, validate_config};
    use crate::{CheckpointErrorCode, Qwen38_27B};
    use serde_json::{Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-model-{label}-{}-{}.json",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

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
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "vocab_size": 248320
            }
        })
    }

    fn write_config(label: &str, config: &Value) -> PathBuf {
        let path = fixture_path(label);
        fs::write(&path, serde_json::to_vec(config).unwrap()).unwrap();
        path
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

//! Exact admission contract for model geometry and quantization routes.

use crate::bindings::NVFP4_MLP_LAYER_END;
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use serde::Deserialize;
use serde_json::Value;
use std::fmt::Debug;
use std::fs;
use std::path::Path;

const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const MODEL_TYPE: &str = "qwen3_5";
const TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const VISION_MODEL_TYPE: &str = "qwen3_5_vision";
const VISION_HIDDEN_ACT: &str = "gelu_pytorch_tanh";
const DTYPE: &str = "bfloat16";
const IMAGE_TOKEN_ID: usize = 248_056;
const VIDEO_TOKEN_ID: usize = 248_057;
const VISION_START_TOKEN_ID: usize = 248_053;
const VISION_END_TOKEN_ID: usize = 248_054;
const MIXED_PRECISION_FORMAT: &str = "mixed-precision";
const QUANT_METHOD: &str = "compressed-tensors";
const QUANTIZATION_STATUS: &str = "compressed";
const FP8_FORMAT: &str = "float-quantized";
const NVFP4_FORMAT: &str = "nvfp4-pack-quantized";
const FLOAT_TYPE: &str = "float";
const E4M3FN_DTYPE: &str = "torch.float8_e4m3fn";
const MODELOPT_QUANT_METHOD: &str = "modelopt";
const MODELOPT_QUANT_ALGO: &str = "NVFP4";
const MODELOPT_PRODUCER: &str = "modelopt";
const MODELOPT_PRODUCER_VERSION: &str = "0.0.1.dev1+g82f1d216d";

#[derive(Debug, Deserialize)]
struct ModelConfig {
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
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: Option<bool>,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {
    deepstack_visual_indexes: Vec<usize>,
    depth: usize,
    dtype: String,
    hidden_act: String,
    hidden_size: usize,
    in_channels: usize,
    intermediate_size: usize,
    model_type: String,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(Debug, Deserialize)]
struct ModelOptConfig {
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

#[derive(Debug, Deserialize)]
struct ModelOptProducer {
    name: String,
    version: String,
}

/// Validates a checkpoint config against the selected architecture.
pub fn validate_config<A: Arch>(path: &Path) -> CheckpointResult<()> {
    let bytes = fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;

    match A::CHECKPOINT_CONTRACT {
        CheckpointContract::CompressedTensors => {
            let config: ModelConfig = serde_json::from_slice(&bytes)
                .map_err(|source| CheckpointError::json(path, source))?;
            validate_compressed::<A>(path, &config)
        }
        CheckpointContract::ModelOptNvfp4 => {
            let config: ModelOptConfig = serde_json::from_slice(&bytes)
                .map_err(|source| CheckpointError::json(path, source))?;
            validate_modelopt::<A>(path, &config)
        }
    }
}

fn validate_compressed<A: Arch>(path: &Path, config: &ModelConfig) -> CheckpointResult<()> {
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

fn validate_modelopt<A: Arch>(path: &Path, config: &ModelOptConfig) -> CheckpointResult<()> {
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

fn validate_text<A: Arch>(path: &Path, text: &TextConfig) -> CheckpointResult<()> {
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
        "text_config.mtp_num_hidden_layers",
        text.mtp_num_hidden_layers,
        A::MTP_LAYERS,
    )?;
    require(
        path,
        "text_config.mtp_use_dedicated_embeddings",
        text.mtp_use_dedicated_embeddings,
        A::MTP_USES_DEDICATED_EMBEDDINGS,
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
    require(
        path,
        "text_config.rms_norm_eps",
        text.rms_norm_eps,
        A::RMS_NORM_EPSILON,
    )?;
    require(path, "text_config.vocab_size", text.vocab_size, A::VOCAB)?;

    validate_layer_types::<A>(path, &text.layer_types)
}

fn validate_vision<A: Arch>(
    path: &Path,
    vision: &VisionConfig,
    model_type: &str,
) -> CheckpointResult<()> {
    require(
        path,
        "vision_config.deepstack_visual_indexes",
        vision.deepstack_visual_indexes.as_slice(),
        &[],
    )?;
    require(path, "vision_config.depth", vision.depth, A::VISION_DEPTH)?;
    require(path, "vision_config.dtype", vision.dtype.as_str(), DTYPE)?;
    require(
        path,
        "vision_config.hidden_act",
        vision.hidden_act.as_str(),
        VISION_HIDDEN_ACT,
    )?;
    require(
        path,
        "vision_config.hidden_size",
        vision.hidden_size,
        A::VISION_HIDDEN,
    )?;
    require(
        path,
        "vision_config.in_channels",
        vision.in_channels,
        A::VISION_INPUT_CHANNELS,
    )?;
    require(
        path,
        "vision_config.intermediate_size",
        vision.intermediate_size,
        A::VISION_INTERMEDIATE,
    )?;
    require(
        path,
        "vision_config.model_type",
        vision.model_type.as_str(),
        model_type,
    )?;
    require(
        path,
        "vision_config.num_heads",
        vision.num_heads,
        A::VISION_NUM_HEADS,
    )?;
    require(
        path,
        "vision_config.num_position_embeddings",
        vision.num_position_embeddings,
        A::VISION_POSITIONS,
    )?;
    require(
        path,
        "vision_config.out_hidden_size",
        vision.out_hidden_size,
        A::VISION_OUTPUT_HIDDEN,
    )?;
    require(
        path,
        "vision_config.patch_size",
        vision.patch_size,
        A::VISION_PATCH_SIZE,
    )?;
    require(
        path,
        "vision_config.spatial_merge_size",
        vision.spatial_merge_size,
        A::VISION_SPATIAL_MERGE_SIZE,
    )?;
    require(
        path,
        "vision_config.temporal_patch_size",
        vision.temporal_patch_size,
        A::VISION_TEMPORAL_PATCH_SIZE,
    )
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
    use super::{modelopt_ignored_targets, test_quantization_config, validate_config};
    use crate::{CheckpointErrorCode, Qwen35_9B, Qwen38_27B};
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

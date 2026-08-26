//! Shared config admission: contract dispatch plus text and vision schema validation.

use crate::common::routes::NVFP4_MLP_LAYER_END;
use crate::qwen35::config::{ModelOptConfig, validate_modelopt};
use crate::qwen36::config::{Qwen36Config, validate_qwen36, validate_qwen36_hf_quantization};
use crate::qwen38::config::{ModelConfig, validate_compressed};
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use serde::Deserialize;
#[cfg(test)]
use serde_json::Value;
use std::fmt::Debug;
use std::fs;
use std::path::Path;

pub(crate) const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
pub(crate) const MODEL_TYPE: &str = "qwen3_5";
const TEXT_MODEL_TYPE: &str = "qwen3_5_text";

const VISION_HIDDEN_ACT: &str = "gelu_pytorch_tanh";
pub(crate) const DTYPE: &str = "bfloat16";
pub(crate) const IMAGE_TOKEN_ID: usize = 248_056;
pub(crate) const VIDEO_TOKEN_ID: usize = 248_057;
pub(crate) const VISION_START_TOKEN_ID: usize = 248_053;
pub(crate) const VISION_END_TOKEN_ID: usize = 248_054;

pub(crate) const FLOAT_TYPE: &str = "float";

pub(crate) const MODELOPT_QUANT_METHOD: &str = "modelopt";

pub(crate) const MODELOPT_PRODUCER: &str = "modelopt";

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
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
    pub(crate) tie_word_embeddings: Option<bool>,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VisionConfig {
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
pub(crate) struct ModelOptProducer {
    pub(crate) name: String,
    pub(crate) version: String,
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
        CheckpointContract::ModelOptNvfp4Moe => {
            let config: Qwen36Config = serde_json::from_slice(&bytes)
                .map_err(|source| CheckpointError::json(path, source))?;
            validate_qwen36::<A>(path, &config)?;
            validate_qwen36_hf_quantization::<A>(path)
        }
    }
}

pub(crate) fn validate_text<A: Arch>(path: &Path, text: &TextConfig) -> CheckpointResult<()> {
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

pub(crate) fn validate_vision<A: Arch>(
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

pub(crate) fn validate_layer_types<A: Arch>(
    path: &Path,
    layer_types: &[String],
) -> CheckpointResult<()> {
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

pub(crate) fn fp8_targets<A: Arch>() -> Vec<String> {
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
    serde_json::from_str(include_str!("../../fixtures/quantization-config.json")).unwrap()
}

pub(crate) fn require<T>(path: &Path, field: &str, actual: T, expected: T) -> CheckpointResult<()>
where
    T: Debug + PartialEq,
{
    if actual != expected {
        return Err(invalid_field(path, field, actual, expected));
    }

    Ok(())
}

pub(crate) fn invalid_field(
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

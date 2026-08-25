//! Qwen3.6-35B-A3B MoE mixed-precision config admission.

use crate::common::config_util::{
    DTYPE, FLOAT_TYPE, IMAGE_TOKEN_ID, MODELOPT_PRODUCER, MODELOPT_QUANT_METHOD, ModelOptProducer,
    VIDEO_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID, VisionConfig, invalid_field,
    require, validate_layer_types, validate_vision,
};
use crate::{Arch, CheckpointError, CheckpointResult};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const QWEN36_ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";
const QWEN36_MODEL_TYPE: &str = "qwen3_5_moe";
const QWEN36_TEXT_MODEL_TYPE: &str = "qwen3_5_moe_text";
const QWEN36_VISION_MODEL_TYPE: &str = "qwen3_5_moe_vision";
const QWEN36_CONFIG_PRODUCER_VERSION: &str = "0.37.0";
const QWEN36_HF_PRODUCER_VERSION: &str = "0.44.0";
const QWEN36_MIXED_PRECISION: &str = "MIXED_PRECISION";
const QWEN36_NVFP4_ALGORITHM: &str = "W4A16_NVFP4";
const QWEN36_FP8_ALGORITHM: &str = "FP8";
const QWEN36_HF_QUANT_CONFIG_FILE: &str = "hf_quant_config.json";

#[derive(Debug, Deserialize)]
pub(crate) struct Qwen36Config {
    architectures: Vec<String>,
    dtype: String,
    image_token_id: usize,
    model_type: String,
    quantization_config: Qwen36QuantizationConfig,
    text_config: Qwen36TextConfig,
    tie_word_embeddings: bool,
    video_token_id: usize,
    vision_config: VisionConfig,
    vision_end_token_id: usize,
    vision_start_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen36TextConfig {
    attention_bias: bool,
    attention_dropout: f32,
    attn_output_gate: bool,
    bos_token_id: usize,
    dtype: String,
    eos_token_id: usize,
    full_attention_interval: usize,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    mamba_ssm_dtype: String,
    max_position_embeddings: usize,
    model_type: String,
    moe_intermediate_size: usize,
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    num_attention_heads: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    output_router_logits: bool,
    pad_token_id: Option<usize>,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    rope_parameters: Qwen36RopeParameters,
    shared_expert_intermediate_size: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen36RopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    partial_rotary_factor: f32,
    rope_theta: u64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct Qwen36QuantizationConfig {
    config_groups: Qwen36ConfigGroups,
    ignore: Vec<String>,
    producer: ModelOptProducer,
    quant_algo: String,
    quant_method: String,
    quantized_layers: BTreeMap<String, Qwen36QuantizedLayer>,
}

#[derive(Debug, Deserialize)]
struct Qwen36ConfigGroups {
    group_0: Qwen36QuantizationGroup,
    group_1: Qwen36QuantizationGroup,
}

#[derive(Debug, Deserialize)]
struct Qwen36QuantizationGroup {
    input_activations: Qwen36QuantizationScheme,
    targets: Vec<String>,
    weights: Qwen36QuantizationScheme,
}

#[derive(Debug, Deserialize)]
struct Qwen36QuantizationScheme {
    dynamic: bool,
    group_size: Option<usize>,
    num_bits: usize,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Qwen36QuantizedLayer {
    quant_algo: String,
    #[serde(default)]
    group_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Qwen36HfQuantConfig {
    producer: ModelOptProducer,
    quantization: Qwen36HfQuantization,
}

#[derive(Debug, Deserialize)]
struct Qwen36HfQuantization {
    kv_cache_quant_algo: String,
    quant_algo: String,
    quantized_layers: BTreeMap<String, Qwen36QuantizedLayer>,
}

pub(crate) fn validate_qwen36<A: Arch>(path: &Path, config: &Qwen36Config) -> CheckpointResult<()> {
    require(path, "architectures length", config.architectures.len(), 1)?;
    require(
        path,
        "architectures[0]",
        config.architectures[0].as_str(),
        QWEN36_ARCHITECTURE,
    )?;
    require(path, "dtype", config.dtype.as_str(), DTYPE)?;
    require(
        path,
        "image_token_id",
        config.image_token_id,
        IMAGE_TOKEN_ID,
    )?;
    require(
        path,
        "model_type",
        config.model_type.as_str(),
        QWEN36_MODEL_TYPE,
    )?;
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

    validate_qwen36_text::<A>(path, &config.text_config)?;
    validate_vision::<A>(path, &config.vision_config, QWEN36_VISION_MODEL_TYPE)?;
    validate_qwen36_quantization::<A>(path, &config.quantization_config)
}

fn validate_qwen36_text<A: Arch>(path: &Path, text: &Qwen36TextConfig) -> CheckpointResult<()> {
    require(
        path,
        "text_config.attention_bias",
        text.attention_bias,
        false,
    )?;
    require(
        path,
        "text_config.attention_dropout",
        text.attention_dropout,
        0.0,
    )?;
    require(
        path,
        "text_config.attn_output_gate",
        text.attn_output_gate,
        true,
    )?;
    require(path, "text_config.bos_token_id", text.bos_token_id, 248_044)?;
    require(path, "text_config.dtype", text.dtype.as_str(), DTYPE)?;
    require(path, "text_config.eos_token_id", text.eos_token_id, 248_044)?;
    require(
        path,
        "text_config.full_attention_interval",
        text.full_attention_interval,
        A::FULL_ATTENTION_INTERVAL,
    )?;
    require(path, "text_config.head_dim", text.head_dim, A::HEAD_DIM)?;
    require(
        path,
        "text_config.hidden_act",
        text.hidden_act.as_str(),
        "silu",
    )?;
    require(path, "text_config.hidden_size", text.hidden_size, A::HIDDEN)?;
    validate_layer_types::<A>(path, &text.layer_types)?;
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
        "text_config.mamba_ssm_dtype",
        text.mamba_ssm_dtype.as_str(),
        "float32",
    )?;
    require(
        path,
        "text_config.max_position_embeddings",
        text.max_position_embeddings,
        262_144,
    )?;
    require(
        path,
        "text_config.model_type",
        text.model_type.as_str(),
        QWEN36_TEXT_MODEL_TYPE,
    )?;
    require(
        path,
        "text_config.moe_intermediate_size",
        text.moe_intermediate_size,
        A::INTERMEDIATE,
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
    require(path, "text_config.num_experts", text.num_experts, 256)?;
    require(
        path,
        "text_config.num_experts_per_tok",
        text.num_experts_per_tok,
        8,
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
        "text_config.output_router_logits",
        text.output_router_logits,
        false,
    )?;
    require(path, "text_config.pad_token_id", text.pad_token_id, None)?;
    require(
        path,
        "text_config.partial_rotary_factor",
        text.partial_rotary_factor,
        0.25,
    )?;
    require(
        path,
        "text_config.rms_norm_eps",
        text.rms_norm_eps,
        A::RMS_NORM_EPSILON,
    )?;
    require(
        path,
        "text_config.shared_expert_intermediate_size",
        text.shared_expert_intermediate_size,
        512,
    )?;
    require(
        path,
        "text_config.tie_word_embeddings",
        text.tie_word_embeddings,
        false,
    )?;
    require(path, "text_config.use_cache", text.use_cache, true)?;
    require(path, "text_config.vocab_size", text.vocab_size, A::VOCAB)?;

    validate_qwen36_rope(path, &text.rope_parameters)
}

fn validate_qwen36_rope(path: &Path, rope: &Qwen36RopeParameters) -> CheckpointResult<()> {
    require(
        path,
        "text_config.rope_parameters.mrope_interleaved",
        rope.mrope_interleaved,
        true,
    )?;
    require(
        path,
        "text_config.rope_parameters.mrope_section",
        rope.mrope_section.as_slice(),
        [11, 11, 10].as_slice(),
    )?;
    require(
        path,
        "text_config.rope_parameters.partial_rotary_factor",
        rope.partial_rotary_factor,
        0.25,
    )?;
    require(
        path,
        "text_config.rope_parameters.rope_theta",
        rope.rope_theta,
        10_000_000,
    )?;
    require(
        path,
        "text_config.rope_parameters.rope_type",
        rope.rope_type.as_str(),
        "default",
    )
}

fn validate_qwen36_quantization<A: Arch>(
    path: &Path,
    config: &Qwen36QuantizationConfig,
) -> CheckpointResult<()> {
    require(
        path,
        "quantization_config.quant_algo",
        config.quant_algo.as_str(),
        QWEN36_MIXED_PRECISION,
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
        QWEN36_CONFIG_PRODUCER_VERSION,
    )?;
    let ignored = [String::from("mtp.layers.0*"), String::from("mtp*")];
    require(
        path,
        "quantization_config.ignore",
        config.ignore.as_slice(),
        ignored.as_slice(),
    )?;

    let fp8_targets = qwen36_fp8_targets::<A>();
    require(
        path,
        "quantization_config.config_groups.group_0.targets",
        config.config_groups.group_0.targets.as_slice(),
        fp8_targets.as_slice(),
    )?;
    validate_qwen36_scheme(
        path,
        "quantization_config.config_groups.group_0.input_activations",
        &config.config_groups.group_0.input_activations,
        8,
        None,
    )?;
    validate_qwen36_scheme(
        path,
        "quantization_config.config_groups.group_0.weights",
        &config.config_groups.group_0.weights,
        8,
        None,
    )?;

    let nvfp4_targets = qwen36_nvfp4_targets::<A>();
    require(
        path,
        "quantization_config.config_groups.group_1.targets",
        config.config_groups.group_1.targets.as_slice(),
        nvfp4_targets.as_slice(),
    )?;
    validate_qwen36_scheme(
        path,
        "quantization_config.config_groups.group_1.input_activations",
        &config.config_groups.group_1.input_activations,
        4,
        Some(16),
    )?;
    validate_qwen36_scheme(
        path,
        "quantization_config.config_groups.group_1.weights",
        &config.config_groups.group_1.weights,
        4,
        Some(16),
    )?;

    validate_qwen36_quantized_layers::<A>(
        path,
        "quantization_config.quantized_layers",
        &config.quantized_layers,
    )
}

fn validate_qwen36_scheme(
    path: &Path,
    field: &str,
    scheme: &Qwen36QuantizationScheme,
    bits: usize,
    group_size: Option<usize>,
) -> CheckpointResult<()> {
    require(path, &format!("{field}.dynamic"), scheme.dynamic, false)?;
    require(
        path,
        &format!("{field}.group_size"),
        scheme.group_size,
        group_size,
    )?;
    require(path, &format!("{field}.num_bits"), scheme.num_bits, bits)?;
    require(
        path,
        &format!("{field}.type"),
        scheme.kind.as_str(),
        FLOAT_TYPE,
    )
}

pub(crate) fn validate_qwen36_hf_quantization<A: Arch>(config_path: &Path) -> CheckpointResult<()> {
    let path = config_path.with_file_name(QWEN36_HF_QUANT_CONFIG_FILE);
    let bytes = fs::read(&path).map_err(|source| CheckpointError::io("reading", &path, source))?;
    let config: Qwen36HfQuantConfig =
        serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(&path, source))?;

    require(
        &path,
        "producer.name",
        config.producer.name.as_str(),
        MODELOPT_PRODUCER,
    )?;
    require(
        &path,
        "producer.version",
        config.producer.version.as_str(),
        QWEN36_HF_PRODUCER_VERSION,
    )?;
    require(
        &path,
        "quantization.quant_algo",
        config.quantization.quant_algo.as_str(),
        QWEN36_MIXED_PRECISION,
    )?;
    require(
        &path,
        "quantization.kv_cache_quant_algo",
        config.quantization.kv_cache_quant_algo.as_str(),
        QWEN36_FP8_ALGORITHM,
    )?;
    validate_qwen36_quantized_layers::<A>(
        &path,
        "quantization.quantized_layers",
        &config.quantization.quantized_layers,
    )
}

fn validate_qwen36_quantized_layers<A: Arch>(
    path: &Path,
    field: &str,
    actual: &BTreeMap<String, Qwen36QuantizedLayer>,
) -> CheckpointResult<()> {
    let expected = qwen36_quantized_layers::<A>();
    require(
        path,
        &format!("{field} length"),
        actual.len(),
        expected.len(),
    )?;

    for (name, expected) in expected {
        let actual = actual
            .get(&name)
            .ok_or_else(|| invalid_field(path, &format!("{field}.{name}"), "missing", &expected))?;
        require(
            path,
            &format!("{field}.{name}.quant_algo"),
            actual.quant_algo.as_str(),
            expected.quant_algo.as_str(),
        )?;
        require(
            path,
            &format!("{field}.{name}.group_size"),
            actual.group_size,
            expected.group_size,
        )?;
    }

    Ok(())
}

fn qwen36_fp8_targets<A: Arch>() -> Vec<String> {
    let mut targets = Vec::new();
    for layer in 0..A::LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        if (layer + 1).is_multiple_of(A::FULL_ATTENTION_INTERVAL) {
            for projection in ["k_proj", "o_proj", "q_proj", "v_proj"] {
                targets.push(format!("{prefix}.self_attn.{projection}"));
            }
        } else {
            for projection in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                targets.push(format!("{prefix}.linear_attn.{projection}"));
            }
        }
    }
    targets.sort();
    targets
}

fn qwen36_nvfp4_targets<A: Arch>() -> Vec<String> {
    let mut targets = vec![String::from("lm_head")];
    for layer in 0..A::LAYERS {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        targets.push(format!("{prefix}.experts"));
        for projection in ["down_proj", "gate_proj", "up_proj"] {
            targets.push(format!("{prefix}.shared_expert.{projection}"));
        }
    }
    targets.sort();
    targets
}

fn qwen36_quantized_layers<A: Arch>() -> BTreeMap<String, Qwen36QuantizedLayer> {
    qwen36_fp8_targets::<A>()
        .into_iter()
        .map(|name| {
            (
                name,
                Qwen36QuantizedLayer {
                    quant_algo: String::from(QWEN36_FP8_ALGORITHM),
                    group_size: None,
                },
            )
        })
        .chain(qwen36_nvfp4_targets::<A>().into_iter().map(|name| {
            (
                name,
                Qwen36QuantizedLayer {
                    quant_algo: String::from(QWEN36_NVFP4_ALGORITHM),
                    group_size: Some(16),
                },
            )
        }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config_util::validate_config;
    use crate::common::test_support::configs::NEXT_FIXTURE;
    use crate::{CheckpointErrorCode, Qwen36Moe35B};
    use serde_json::{Map, Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    struct Qwen36Fixture {
        root: PathBuf,
    }

    impl Qwen36Fixture {
        fn new(label: &str, config: &Value, hf_quantization: &Value) -> Self {
            let root = std::env::temp_dir().join(format!(
                "tuisko-model-{label}-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            fs::write(
                root.join("config.json"),
                serde_json::to_vec(config).unwrap(),
            )
            .unwrap();
            fs::write(
                root.join("hf_quant_config.json"),
                serde_json::to_vec(hf_quantization).unwrap(),
            )
            .unwrap();
            Self { root }
        }

        fn config(&self) -> PathBuf {
            self.root.join("config.json")
        }
    }

    impl Drop for Qwen36Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn qwen36_quantized_layers_json() -> Value {
        let layers = qwen36_quantized_layers::<Qwen36Moe35B>()
            .into_iter()
            .map(|(name, layer)| {
                let mut value = json!({"quant_algo": layer.quant_algo});
                if let Some(group_size) = layer.group_size {
                    value["group_size"] = json!(group_size);
                }
                (name, value)
            })
            .collect::<Map<_, _>>();
        Value::Object(layers)
    }

    fn valid_qwen36_config() -> Value {
        let layer_types = (0usize..40)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let quantization_config = json!({
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "targets": qwen36_fp8_targets::<Qwen36Moe35B>(),
                    "weights": {"dynamic": false, "num_bits": 8, "type": "float"}
                },
                "group_1": {
                    "input_activations": {
                        "dynamic": false,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float"
                    },
                    "targets": qwen36_nvfp4_targets::<Qwen36Moe35B>(),
                    "weights": {
                        "dynamic": false,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float"
                    }
                }
            },
            "ignore": ["mtp.layers.0*", "mtp*"],
            "producer": {"name": "modelopt", "version": "0.37.0"},
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "quantized_layers": qwen36_quantized_layers_json()
        });
        let rope_parameters = json!({
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        });
        let text_config = json!({
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_output_gate": true,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "output_router_logits": false,
            "pad_token_id": null,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": rope_parameters,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": 248320
        });
        let vision_config = json!({
            "deepstack_visual_indexes": [],
            "depth": 27,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe_vision",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        });

        json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "dtype": "bfloat16",
            "image_token_id": 248056,
            "model_type": "qwen3_5_moe",
            "quantization_config": quantization_config,
            "text_config": text_config,
            "tie_word_embeddings": false,
            "video_token_id": 248057,
            "vision_config": vision_config,
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053
        })
    }

    fn valid_qwen36_hf_quantization() -> Value {
        json!({
            "producer": {"name": "modelopt", "version": "0.44.0"},
            "quantization": {
                "kv_cache_quant_algo": "FP8",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": qwen36_quantized_layers_json()
            }
        })
    }

    #[test]
    fn admits_exact_qwen36_contract_and_quantization_inventory() {
        assert_eq!(qwen36_fp8_targets::<Qwen36Moe35B>().len(), 130);
        assert_eq!(qwen36_nvfp4_targets::<Qwen36Moe35B>().len(), 161);
        assert_eq!(qwen36_quantized_layers::<Qwen36Moe35B>().len(), 291);

        let fixture = Qwen36Fixture::new(
            "valid-qwen36",
            &valid_qwen36_config(),
            &valid_qwen36_hf_quantization(),
        );

        validate_config::<Qwen36Moe35B>(&fixture.config()).unwrap();
    }

    #[test]
    fn rejects_qwen36_geometry_and_route_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen36-architecture",
                "/architectures/0",
                json!("OtherArchitecture"),
                "architectures[0]",
            ),
            (
                "qwen36-hidden",
                "/text_config/hidden_size",
                json!(4_096),
                "text_config.hidden_size",
            ),
            (
                "qwen36-layer-route",
                "/text_config/layer_types/3",
                json!("linear_attention"),
                "text_config.layer_types[3]",
            ),
            (
                "qwen36-experts",
                "/text_config/num_experts",
                json!(128),
                "text_config.num_experts",
            ),
            (
                "qwen36-rope-section",
                "/text_config/rope_parameters/mrope_section/2",
                json!(11),
                "text_config.rope_parameters.mrope_section",
            ),
            (
                "qwen36-vision-output",
                "/vision_config/out_hidden_size",
                json!(4_096),
                "vision_config.out_hidden_size",
            ),
        ] {
            let mut config = valid_qwen36_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let fixture = Qwen36Fixture::new(label, &config, &valid_qwen36_hf_quantization());

            let error = validate_config::<Qwen36Moe35B>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_qwen36_config_quantization_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen36-producer",
                "/quantization_config/producer/version",
                json!("other"),
                "quantization_config.producer.version",
            ),
            (
                "qwen36-fp8-target",
                "/quantization_config/config_groups/group_0/targets/0",
                json!("other"),
                "config_groups.group_0.targets",
            ),
            (
                "qwen36-nvfp4-group",
                "/quantization_config/config_groups/group_1/weights/group_size",
                json!(32),
                "config_groups.group_1.weights.group_size",
            ),
        ] {
            let mut config = valid_qwen36_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let fixture = Qwen36Fixture::new(label, &config, &valid_qwen36_hf_quantization());

            let error = validate_config::<Qwen36Moe35B>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
        }

        let name = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let mut config = valid_qwen36_config();
        config["quantization_config"]["quantized_layers"][name]["quant_algo"] = json!("other");
        let fixture = Qwen36Fixture::new(
            "qwen36-layer-algorithm",
            &config,
            &valid_qwen36_hf_quantization(),
        );

        let error = validate_config::<Qwen36Moe35B>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Config);
        assert!(
            error.to_string().contains(&format!(
                "quantization_config.quantized_layers.{name}.quant_algo"
            )),
            "{error}"
        );
    }

    #[test]
    fn rejects_qwen36_external_quantization_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen36-hf-producer",
                "/producer/version",
                json!("other"),
                "producer.version",
            ),
            (
                "qwen36-hf-kv",
                "/quantization/kv_cache_quant_algo",
                json!("BF16"),
                "quantization.kv_cache_quant_algo",
            ),
        ] {
            let mut hf_quantization = valid_qwen36_hf_quantization();
            *hf_quantization.pointer_mut(pointer).unwrap() = replacement;
            let fixture = Qwen36Fixture::new(label, &valid_qwen36_config(), &hf_quantization);

            let error = validate_config::<Qwen36Moe35B>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
        }

        let name = "lm_head";
        let mut hf_quantization = valid_qwen36_hf_quantization();
        hf_quantization["quantization"]["quantized_layers"][name]["group_size"] = json!(32);
        let fixture = Qwen36Fixture::new(
            "qwen36-hf-layer-group",
            &valid_qwen36_config(),
            &hf_quantization,
        );

        let error = validate_config::<Qwen36Moe35B>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Config);
        assert!(
            error
                .to_string()
                .contains("quantization.quantized_layers.lm_head.group_size"),
            "{error}"
        );
    }

    #[test]
    fn requires_qwen36_external_quantization_file() {
        let fixture = Qwen36Fixture::new(
            "qwen36-missing-hf-quantization",
            &valid_qwen36_config(),
            &valid_qwen36_hf_quantization(),
        );
        fs::remove_file(fixture.root.join("hf_quant_config.json")).unwrap();

        let error = validate_config::<Qwen36Moe35B>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Io);
        assert!(
            error.to_string().contains("hf_quant_config.json"),
            "{error}"
        );
    }
}

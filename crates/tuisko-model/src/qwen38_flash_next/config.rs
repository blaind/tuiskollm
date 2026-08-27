//! Qwen3.8-Flash-Next ModelOpt NVFP4 config admission.
//!
//! Its vision config has a distinct schema and is validated locally.

use crate::common::config_util::{
    DTYPE, FLOAT_TYPE, IMAGE_TOKEN_ID, MODELOPT_PRODUCER, MODELOPT_QUANT_METHOD, ModelOptProducer,
    VIDEO_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID, require, validate_layer_types,
};
use crate::{Arch, CheckpointError, CheckpointResult, Qwen38FlashNext};
use serde::Deserialize;
use std::fs;
use std::path::Path;

const QWEN38_FLASH_NEXT_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const QWEN38_FLASH_NEXT_MODEL_TYPE: &str = "qwen4_exp";
const QWEN38_FLASH_NEXT_TEXT_MODEL_TYPE: &str = "qwen4_exp_text";
const QWEN38_FLASH_NEXT_TRANSFORMERS_VERSION: &str = "5.8.0.dev0";
const QWEN38_FLASH_NEXT_QUANT_ALGO: &str = "NVFP4";
const QWEN38_FLASH_NEXT_PRODUCER_VERSION: &str = "0.46.0";
const QWEN38_FLASH_NEXT_HIDDEN_ACT: &str = "silu";
const QWEN38_FLASH_NEXT_VISION_HIDDEN_ACT: &str = "gelu_pytorch_tanh";
const QWEN38_FLASH_NEXT_MAMBA_STATE_DTYPE: &str = "float32";
const QWEN38_FLASH_NEXT_ROPE_TYPE: &str = "default";
const QWEN38_FLASH_NEXT_TOKEN_ID: usize = Qwen38FlashNext::EOS_TOKEN_ID as usize;
const QWEN38_FLASH_NEXT_INITIALIZER_RANGE: f32 = 0.02;
const QWEN38_FLASH_NEXT_ROUTER_AUX_LOSS_COEFFICIENT: f32 = 0.001;
const QWEN38_FLASH_NEXT_NVFP4_GROUP_SIZE: usize = 16;
const QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE: &str = "hf_quant_config.json";

/// Exact ordered partition of unquantized modules.
const QWEN38_FLASH_NEXT_QUANTIZATION_IGNORE: [&str; 13] = [
    "model.embed_tokens",
    "mtp.*",
    "model.mtp.*",
    "*.self_attn.*",
    "*.linear_attn.*",
    "*.mlp.gate*",
    "*.mlp.shared_expert.*",
    "*.mlp.shared_expert_gate*",
    "*hyper_connection*",
    "*.ple.*",
    "model.visual.*",
    "model.language_model.embed_tokens",
    "lm_head",
];

#[derive(Debug, Deserialize)]
pub(crate) struct Qwen38FlashNextConfig {
    architectures: Vec<String>,
    image_token_id: usize,
    language_model_only: bool,
    model_type: String,
    quantization_config: Qwen38FlashNextQuantizationConfig,
    text_config: Qwen38FlashNextTextConfig,
    tie_word_embeddings: bool,
    transformers_version: String,
    video_token_id: usize,
    vision_config: Qwen38FlashNextVisionConfig,
    vision_end_token_id: usize,
    vision_start_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextTextConfig {
    attention_bias: bool,
    attention_dropout: f32,
    bos_token_id: usize,
    dtype: String,
    eos_token_id: usize,
    full_attention_interval: usize,
    hc_count: usize,
    hc_lowrank: usize,
    head_dim: usize,
    heads_per_ngram: usize,
    hidden_act: String,
    hidden_size: usize,
    indexer_budget: usize,
    indexer_compress_ratio: usize,
    indexer_head_dim: usize,
    indexer_kv_heads: usize,
    indexer_n_heads: usize,
    initializer_range: f32,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    make_ngram_vocab_size_divisible_by: usize,
    mamba_ssm_dtype: String,
    max_position_embeddings: usize,
    model_type: String,
    moe_intermediate_size: usize,
    mtp: Qwen38FlashNextMtpConfig,
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    ngram_size: usize,
    ngram_vocab_size_base: usize,
    num_attention_heads: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    output_gate_type: String,
    output_router_logits: bool,
    pad_token_id: Option<usize>,
    partial_rotary_factor: f32,
    ple_conv_kernel_size: usize,
    ple_embed_dim: usize,
    ple_embedding_dtype: String,
    ple_layer_ids: Vec<usize>,
    rms_norm_eps: f32,
    rope_parameters: Qwen38FlashNextRopeParameters,
    router_aux_loss_coef: f32,
    shared_expert_intermediate_size: usize,
    split_ngram_parts: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextMtpConfig {
    hybrid: bool,
    layer_types: Vec<String>,
    mtp_use_hidden_state_from_layer: Option<usize>,
    num_hidden_layers: usize,
    rope_theta: u64,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextRopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    partial_rotary_factor: f32,
    rope_theta: u64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextVisionConfig {
    deepstack_visual_indexes: Vec<usize>,
    depth: usize,
    hidden_act: String,
    hidden_size: usize,
    in_channels: usize,
    initializer_range: f32,
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
struct Qwen38FlashNextQuantizationConfig {
    config_groups: Qwen38FlashNextConfigGroups,
    ignore: Vec<String>,
    producer: ModelOptProducer,
    quant_algo: String,
    quant_method: String,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextConfigGroups {
    group_0: Qwen38FlashNextQuantizationGroup,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextQuantizationGroup {
    input_activations: Qwen38FlashNextQuantizationScheme,
    targets: Vec<String>,
    weights: Qwen38FlashNextQuantizationScheme,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextQuantizationScheme {
    dynamic: bool,
    group_size: usize,
    num_bits: usize,
    #[serde(rename = "type")]
    kind: String,
}

/// Target-specific `hf_quant_config.json` schema.
#[derive(Debug, Deserialize)]
struct Qwen38FlashNextHfQuantConfig {
    producer: ModelOptProducer,
    quantization: Qwen38FlashNextHfQuantization,
}

#[derive(Debug, Deserialize)]
struct Qwen38FlashNextHfQuantization {
    exclude_modules: Vec<String>,
    group_size: usize,
    quant_algo: String,
}

pub(crate) fn validate_qwen38_flash_next<A: Arch>(
    path: &Path,
    config: &Qwen38FlashNextConfig,
) -> CheckpointResult<()> {
    require(path, "architectures length", config.architectures.len(), 1)?;
    require(
        path,
        "architectures[0]",
        config.architectures[0].as_str(),
        QWEN38_FLASH_NEXT_ARCHITECTURE,
    )?;
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
    require(
        path,
        "model_type",
        config.model_type.as_str(),
        QWEN38_FLASH_NEXT_MODEL_TYPE,
    )?;
    require(
        path,
        "tie_word_embeddings",
        config.tie_word_embeddings,
        false,
    )?;
    require(
        path,
        "transformers_version",
        config.transformers_version.as_str(),
        QWEN38_FLASH_NEXT_TRANSFORMERS_VERSION,
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

    validate_qwen38_flash_next_text::<A>(path, &config.text_config)?;
    validate_qwen38_flash_next_vision::<A>(path, &config.vision_config)?;
    validate_qwen38_flash_next_quantization(path, &config.quantization_config)
}

fn validate_qwen38_flash_next_text<A: Arch>(
    path: &Path,
    text: &Qwen38FlashNextTextConfig,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

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
        "text_config.bos_token_id",
        text.bos_token_id,
        QWEN38_FLASH_NEXT_TOKEN_ID,
    )?;
    require(path, "text_config.dtype", text.dtype.as_str(), DTYPE)?;
    require(
        path,
        "text_config.eos_token_id",
        text.eos_token_id,
        QWEN38_FLASH_NEXT_TOKEN_ID,
    )?;
    require(
        path,
        "text_config.full_attention_interval",
        text.full_attention_interval,
        A::FULL_ATTENTION_INTERVAL,
    )?;
    require(path, "text_config.hc_count", text.hc_count, F::HC_COUNT)?;
    require(
        path,
        "text_config.hc_lowrank",
        text.hc_lowrank,
        F::HC_LOWRANK,
    )?;
    require(path, "text_config.head_dim", text.head_dim, A::HEAD_DIM)?;
    require(
        path,
        "text_config.heads_per_ngram",
        text.heads_per_ngram,
        F::HEADS_PER_NGRAM,
    )?;
    require(
        path,
        "text_config.hidden_act",
        text.hidden_act.as_str(),
        QWEN38_FLASH_NEXT_HIDDEN_ACT,
    )?;
    require(path, "text_config.hidden_size", text.hidden_size, A::HIDDEN)?;
    require(
        path,
        "text_config.indexer_budget",
        text.indexer_budget,
        F::INDEXER_BUDGET,
    )?;
    require(
        path,
        "text_config.indexer_compress_ratio",
        text.indexer_compress_ratio,
        F::INDEXER_COMPRESS_RATIO,
    )?;
    require(
        path,
        "text_config.indexer_head_dim",
        text.indexer_head_dim,
        F::INDEXER_HEAD_DIM,
    )?;
    require(
        path,
        "text_config.indexer_kv_heads",
        text.indexer_kv_heads,
        F::INDEXER_KV_HEADS,
    )?;
    require(
        path,
        "text_config.indexer_n_heads",
        text.indexer_n_heads,
        F::INDEXER_HEADS,
    )?;
    require(
        path,
        "text_config.initializer_range",
        text.initializer_range,
        QWEN38_FLASH_NEXT_INITIALIZER_RANGE,
    )?;
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
        "text_config.make_ngram_vocab_size_divisible_by",
        text.make_ngram_vocab_size_divisible_by,
        F::NGRAM_VOCAB_DIVISOR,
    )?;
    require(
        path,
        "text_config.mamba_ssm_dtype",
        text.mamba_ssm_dtype.as_str(),
        QWEN38_FLASH_NEXT_MAMBA_STATE_DTYPE,
    )?;
    require(
        path,
        "text_config.max_position_embeddings",
        text.max_position_embeddings,
        F::MAX_POSITION_EMBEDDINGS,
    )?;
    require(
        path,
        "text_config.model_type",
        text.model_type.as_str(),
        QWEN38_FLASH_NEXT_TEXT_MODEL_TYPE,
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
        "text_config.ngram_size",
        text.ngram_size,
        F::NGRAM_SIZE,
    )?;
    require(
        path,
        "text_config.ngram_vocab_size_base",
        text.ngram_vocab_size_base,
        F::NGRAM_VOCAB_BASE,
    )?;
    require(
        path,
        "text_config.num_attention_heads",
        text.num_attention_heads,
        A::NUM_ATTENTION_HEADS,
    )?;
    require(
        path,
        "text_config.num_experts",
        text.num_experts,
        F::NUM_EXPERTS,
    )?;
    require(
        path,
        "text_config.num_experts_per_tok",
        text.num_experts_per_tok,
        F::NUM_EXPERTS_PER_TOKEN,
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
        "text_config.output_gate_type",
        text.output_gate_type.as_str(),
        F::GDN_OUTPUT_GATE,
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
        F::PARTIAL_ROTARY_FACTOR,
    )?;
    require(
        path,
        "text_config.ple_conv_kernel_size",
        text.ple_conv_kernel_size,
        F::PLE_CONV_KERNEL,
    )?;
    require(
        path,
        "text_config.ple_embed_dim",
        text.ple_embed_dim,
        F::PLE_EMBED_DIM,
    )?;
    require(
        path,
        "text_config.ple_embedding_dtype",
        text.ple_embedding_dtype.as_str(),
        F::PLE_EMBEDDING_DTYPE,
    )?;
    require(
        path,
        "text_config.ple_layer_ids",
        text.ple_layer_ids.as_slice(),
        [F::PLE_LAYER + 1].as_slice(),
    )?;
    require(
        path,
        "text_config.rms_norm_eps",
        text.rms_norm_eps,
        A::RMS_NORM_EPSILON,
    )?;
    require(
        path,
        "text_config.router_aux_loss_coef",
        text.router_aux_loss_coef,
        QWEN38_FLASH_NEXT_ROUTER_AUX_LOSS_COEFFICIENT,
    )?;
    require(
        path,
        "text_config.shared_expert_intermediate_size",
        text.shared_expert_intermediate_size,
        F::SHARED_EXPERT_INTERMEDIATE,
    )?;
    require(
        path,
        "text_config.split_ngram_parts",
        text.split_ngram_parts,
        F::NGRAM_SHARDS,
    )?;
    require(
        path,
        "text_config.tie_word_embeddings",
        text.tie_word_embeddings,
        false,
    )?;
    require(path, "text_config.use_cache", text.use_cache, true)?;
    require(path, "text_config.vocab_size", text.vocab_size, A::VOCAB)?;

    validate_qwen38_flash_next_mtp::<A>(path, &text.mtp)?;
    validate_qwen38_flash_next_rope(path, &text.rope_parameters)
}

/// Validates the single sparse-attention MTP layer and shared rotary base.
fn validate_qwen38_flash_next_mtp<A: Arch>(
    path: &Path,
    mtp: &Qwen38FlashNextMtpConfig,
) -> CheckpointResult<()> {
    require(path, "text_config.mtp.hybrid", mtp.hybrid, true)?;
    require(
        path,
        "text_config.mtp.layer_types",
        mtp.layer_types.as_slice(),
        [String::from("full_attention")].as_slice(),
    )?;
    require(
        path,
        "text_config.mtp.mtp_use_hidden_state_from_layer",
        mtp.mtp_use_hidden_state_from_layer,
        None,
    )?;
    require(
        path,
        "text_config.mtp.num_hidden_layers",
        mtp.num_hidden_layers,
        A::MTP_LAYERS,
    )?;
    require(
        path,
        "text_config.mtp.rope_theta",
        mtp.rope_theta,
        Qwen38FlashNext::ROPE_THETA,
    )
}

fn validate_qwen38_flash_next_rope(
    path: &Path,
    rope: &Qwen38FlashNextRopeParameters,
) -> CheckpointResult<()> {
    type F = Qwen38FlashNext;

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
        F::MROPE_SECTION.as_slice(),
    )?;
    require(
        path,
        "text_config.rope_parameters.partial_rotary_factor",
        rope.partial_rotary_factor,
        F::PARTIAL_ROTARY_FACTOR,
    )?;
    require(
        path,
        "text_config.rope_parameters.rope_theta",
        rope.rope_theta,
        F::ROPE_THETA,
    )?;
    require(
        path,
        "text_config.rope_parameters.rope_type",
        rope.rope_type.as_str(),
        QWEN38_FLASH_NEXT_ROPE_TYPE,
    )
}

fn validate_qwen38_flash_next_vision<A: Arch>(
    path: &Path,
    vision: &Qwen38FlashNextVisionConfig,
) -> CheckpointResult<()> {
    require(
        path,
        "vision_config.deepstack_visual_indexes",
        vision.deepstack_visual_indexes.as_slice(),
        &[],
    )?;
    require(path, "vision_config.depth", vision.depth, A::VISION_DEPTH)?;
    require(
        path,
        "vision_config.hidden_act",
        vision.hidden_act.as_str(),
        QWEN38_FLASH_NEXT_VISION_HIDDEN_ACT,
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
        "vision_config.initializer_range",
        vision.initializer_range,
        QWEN38_FLASH_NEXT_INITIALIZER_RANGE,
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
        QWEN38_FLASH_NEXT_MODEL_TYPE,
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

fn validate_qwen38_flash_next_quantization(
    path: &Path,
    config: &Qwen38FlashNextQuantizationConfig,
) -> CheckpointResult<()> {
    require(
        path,
        "quantization_config.quant_algo",
        config.quant_algo.as_str(),
        QWEN38_FLASH_NEXT_QUANT_ALGO,
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
        QWEN38_FLASH_NEXT_PRODUCER_VERSION,
    )?;
    require(
        path,
        "quantization_config.ignore length",
        config.ignore.len(),
        QWEN38_FLASH_NEXT_QUANTIZATION_IGNORE.len(),
    )?;
    for (index, expected) in QWEN38_FLASH_NEXT_QUANTIZATION_IGNORE.iter().enumerate() {
        require(
            path,
            &format!("quantization_config.ignore[{index}]"),
            config.ignore[index].as_str(),
            expected,
        )?;
    }

    let group = &config.config_groups.group_0;
    require(
        path,
        "quantization_config.config_groups.group_0.targets",
        group.targets.as_slice(),
        [String::from("Linear")].as_slice(),
    )?;
    validate_qwen38_flash_next_scheme(
        path,
        "quantization_config.config_groups.group_0.input_activations",
        &group.input_activations,
    )?;
    validate_qwen38_flash_next_scheme(
        path,
        "quantization_config.config_groups.group_0.weights",
        &group.weights,
    )
}

/// Validates the `hf_quant_config.json` sidecar beside `config_path`.
pub(crate) fn validate_qwen38_flash_next_hf_quantization(
    config_path: &Path,
) -> CheckpointResult<()> {
    let path = config_path.with_file_name(QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE);
    let bytes = fs::read(&path).map_err(|source| CheckpointError::io("reading", &path, source))?;
    let config: Qwen38FlashNextHfQuantConfig =
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
        QWEN38_FLASH_NEXT_PRODUCER_VERSION,
    )?;
    require(
        &path,
        "quantization.quant_algo",
        config.quantization.quant_algo.as_str(),
        QWEN38_FLASH_NEXT_QUANT_ALGO,
    )?;
    require(
        &path,
        "quantization.group_size",
        config.quantization.group_size,
        QWEN38_FLASH_NEXT_NVFP4_GROUP_SIZE,
    )?;
    require(
        &path,
        "quantization.exclude_modules length",
        config.quantization.exclude_modules.len(),
        QWEN38_FLASH_NEXT_QUANTIZATION_IGNORE.len(),
    )?;

    // Both files must name the same ordered quantization partition.
    for (index, expected) in QWEN38_FLASH_NEXT_QUANTIZATION_IGNORE.iter().enumerate() {
        require(
            &path,
            &format!("quantization.exclude_modules[{index}]"),
            config.quantization.exclude_modules[index].as_str(),
            expected,
        )?;
    }

    Ok(())
}

fn validate_qwen38_flash_next_scheme(
    path: &Path,
    field: &str,
    scheme: &Qwen38FlashNextQuantizationScheme,
) -> CheckpointResult<()> {
    require(path, &format!("{field}.dynamic"), scheme.dynamic, false)?;
    require(
        path,
        &format!("{field}.group_size"),
        scheme.group_size,
        QWEN38_FLASH_NEXT_NVFP4_GROUP_SIZE,
    )?;
    require(path, &format!("{field}.num_bits"), scheme.num_bits, 4)?;
    require(
        path,
        &format!("{field}.type"),
        scheme.kind.as_str(),
        FLOAT_TYPE,
    )
}

#[cfg(test)]
pub(crate) fn pinned_qwen38_flash_next_config() -> serde_json::Value {
    serde_json::from_str(include_str!("../../fixtures/qwen38-flash-next-config.json")).unwrap()
}

#[cfg(test)]
pub(crate) fn pinned_qwen38_flash_next_hf_quant_config() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../fixtures/qwen38-flash-next-hf-quant-config.json"
    ))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointErrorCode;
    use crate::common::schema::validate_config;
    use crate::common::test_support::configs::NEXT_FIXTURE;
    use serde_json::{Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    /// A snapshot root holding `config.json` and its `hf_quant_config.json` sidecar.
    ///
    /// Admission reads both, so the fixture is a directory rather than a single file.
    struct Qwen38FlashNextFixture {
        root: PathBuf,
    }

    impl Qwen38FlashNextFixture {
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
                root.join(QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE),
                serde_json::to_vec(hf_quantization).unwrap(),
            )
            .unwrap();
            Self { root }
        }

        fn pinned(label: &str, config: &Value) -> Self {
            Self::new(label, config, &pinned_qwen38_flash_next_hf_quant_config())
        }

        fn config(&self) -> PathBuf {
            self.root.join("config.json")
        }
    }

    impl Drop for Qwen38FlashNextFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn admits_the_pinned_radixark_config() {
        let fixture = Qwen38FlashNextFixture::pinned(
            "valid-qwen38-flash-next",
            &pinned_qwen38_flash_next_config(),
        );

        validate_config::<Qwen38FlashNext>(&fixture.config()).unwrap();
    }

    #[test]
    fn rejects_qwen38_flash_next_geometry_and_route_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen38-flash-next-architecture",
                "/architectures/0",
                json!("Qwen3_5ForConditionalGeneration"),
                "architectures[0]",
            ),
            (
                "qwen38-flash-next-model-type",
                "/model_type",
                json!("qwen3_5"),
                "model_type",
            ),
            (
                "qwen38-flash-next-hidden",
                "/text_config/hidden_size",
                json!(5_120),
                "text_config.hidden_size",
            ),
            (
                "qwen38-flash-next-layers",
                "/text_config/num_hidden_layers",
                json!(64),
                "text_config.num_hidden_layers",
            ),
            (
                "qwen38-flash-next-layer-route",
                "/text_config/layer_types/3",
                json!("linear_attention"),
                "text_config.layer_types[3]",
            ),
            (
                "qwen38-flash-next-experts",
                "/text_config/num_experts",
                json!(256),
                "text_config.num_experts",
            ),
            (
                "qwen38-flash-next-experts-per-token",
                "/text_config/num_experts_per_tok",
                json!(8),
                "text_config.num_experts_per_tok",
            ),
            (
                "qwen38-flash-next-hc-lowrank",
                "/text_config/hc_lowrank",
                json!(256),
                "text_config.hc_lowrank",
            ),
            (
                "qwen38-flash-next-indexer-budget",
                "/text_config/indexer_budget",
                json!(4_096),
                "text_config.indexer_budget",
            ),
            (
                "qwen38-flash-next-gdn-output-gate",
                "/text_config/output_gate_type",
                json!("silu"),
                "text_config.output_gate_type",
            ),
            (
                "qwen38-flash-next-ple-layer",
                "/text_config/ple_layer_ids/0",
                json!(3),
                "text_config.ple_layer_ids",
            ),
            (
                "qwen38-flash-next-ngram-shards",
                "/text_config/split_ngram_parts",
                json!(64),
                "text_config.split_ngram_parts",
            ),
            (
                "qwen38-flash-next-ple-embedding-dtype",
                "/text_config/ple_embedding_dtype",
                json!("bfloat16"),
                "text_config.ple_embedding_dtype",
            ),
            (
                "qwen38-flash-next-mtp-route",
                "/text_config/mtp/layer_types/0",
                json!("linear_attention"),
                "text_config.mtp.layer_types",
            ),
            (
                "qwen38-flash-next-rope-section",
                "/text_config/rope_parameters/mrope_section/2",
                json!(11),
                "text_config.rope_parameters.mrope_section",
            ),
            (
                "qwen38-flash-next-vision-output",
                "/vision_config/out_hidden_size",
                json!(5_120),
                "vision_config.out_hidden_size",
            ),
            (
                "qwen38-flash-next-vision-model-type",
                "/vision_config/model_type",
                json!("qwen4_exp_vision"),
                "vision_config.model_type",
            ),
        ] {
            let mut config = pinned_qwen38_flash_next_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let fixture = Qwen38FlashNextFixture::pinned(label, &config);

            let error = validate_config::<Qwen38FlashNext>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_qwen38_flash_next_quantization_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen38-flash-next-quant-algo",
                "/quantization_config/quant_algo",
                json!("MIXED_PRECISION"),
                "quantization_config.quant_algo",
            ),
            (
                "qwen38-flash-next-quant-method",
                "/quantization_config/quant_method",
                json!("compressed-tensors"),
                "quantization_config.quant_method",
            ),
            (
                "qwen38-flash-next-producer-version",
                "/quantization_config/producer/version",
                json!("0.37.0"),
                "quantization_config.producer.version",
            ),
            (
                "qwen38-flash-next-target",
                "/quantization_config/config_groups/group_0/targets/0",
                json!("Conv1D"),
                "group_0.targets",
            ),
            (
                "qwen38-flash-next-weight-group",
                "/quantization_config/config_groups/group_0/weights/group_size",
                json!(32),
                "group_0.weights.group_size",
            ),
            (
                "qwen38-flash-next-activation-bits",
                "/quantization_config/config_groups/group_0/input_activations/num_bits",
                json!(8),
                "group_0.input_activations.num_bits",
            ),
            (
                "qwen38-flash-next-activation-dynamic",
                "/quantization_config/config_groups/group_0/input_activations/dynamic",
                json!(true),
                "group_0.input_activations.dynamic",
            ),
            (
                "qwen38-flash-next-ignore-entry",
                "/quantization_config/ignore/3",
                json!("*.self_attn.q_proj"),
                "quantization_config.ignore[3]",
            ),
        ] {
            let mut config = pinned_qwen38_flash_next_config();
            *config.pointer_mut(pointer).unwrap() = replacement;
            let fixture = Qwen38FlashNextFixture::pinned(label, &config);

            let error = validate_config::<Qwen38FlashNext>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    /// A shortened ignore list keeps every surviving entry equal, so only the length check
    /// separates it from the pinned partition.
    #[test]
    fn rejects_a_truncated_quantization_ignore_list() {
        let mut config = pinned_qwen38_flash_next_config();
        config["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap()
            .pop();
        let fixture = Qwen38FlashNextFixture::pinned("qwen38-flash-next-ignore-truncated", &config);

        let error = validate_config::<Qwen38FlashNext>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Config);
        assert!(
            error
                .to_string()
                .contains("quantization_config.ignore length"),
            "{error}"
        );
    }

    #[test]
    fn refuses_a_config_the_target_contract_does_not_admit() {
        let mut config = pinned_qwen38_flash_next_config();
        config
            .as_object_mut()
            .unwrap()
            .remove("quantization_config")
            .unwrap();
        let fixture =
            Qwen38FlashNextFixture::pinned("qwen38-flash-next-missing-quantization", &config);

        let error = validate_config::<Qwen38FlashNext>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Json);
    }

    #[test]
    fn admitted_sidecar_restates_the_config_ignore_list_verbatim() {
        let config = pinned_qwen38_flash_next_config();
        let sidecar = pinned_qwen38_flash_next_hf_quant_config();

        assert_eq!(
            sidecar["quantization"]["exclude_modules"], config["quantization_config"]["ignore"],
            "the two statements of the partition must be identical"
        );
        assert_eq!(sidecar["quantization"]["quant_algo"], json!("NVFP4"));
        assert_eq!(sidecar["quantization"]["group_size"], json!(16));
        assert_eq!(sidecar["producer"]["version"], json!("0.46.0"));
        assert!(
            sidecar["quantization"].get("kv_cache_quant_algo").is_none(),
            "this sidecar names no KV cache algorithm, unlike Qwen3.6's"
        );
        assert!(
            sidecar["quantization"].get("quantized_layers").is_none(),
            "this sidecar states the partition negatively, unlike Qwen3.6's"
        );
    }

    #[test]
    fn rejects_hf_quant_config_mismatches() {
        for (label, pointer, replacement, field) in [
            (
                "qwen38-flash-next-sidecar-producer",
                "/producer/version",
                json!("0.44.0"),
                "producer.version",
            ),
            (
                "qwen38-flash-next-sidecar-producer-name",
                "/producer/name",
                json!("compressed-tensors"),
                "producer.name",
            ),
            (
                "qwen38-flash-next-sidecar-algo",
                "/quantization/quant_algo",
                json!("W4A16_NVFP4"),
                "quantization.quant_algo",
            ),
            (
                "qwen38-flash-next-sidecar-group",
                "/quantization/group_size",
                json!(32),
                "quantization.group_size",
            ),
            (
                "qwen38-flash-next-sidecar-exclude-entry",
                "/quantization/exclude_modules/9",
                json!("*.ple.ple_embedding.*"),
                "quantization.exclude_modules[9]",
            ),
        ] {
            let mut sidecar = pinned_qwen38_flash_next_hf_quant_config();
            *sidecar.pointer_mut(pointer).unwrap() = replacement;
            let fixture =
                Qwen38FlashNextFixture::new(label, &pinned_qwen38_flash_next_config(), &sidecar);

            let error = validate_config::<Qwen38FlashNext>(&fixture.config())
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(error.to_string().contains(field), "{error}");
            assert!(
                error
                    .to_string()
                    .contains(QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE),
                "{error}"
            );
        }
    }

    #[test]
    fn rejects_a_truncated_sidecar_exclude_list() {
        let mut sidecar = pinned_qwen38_flash_next_hf_quant_config();
        sidecar["quantization"]["exclude_modules"]
            .as_array_mut()
            .unwrap()
            .pop();
        let fixture = Qwen38FlashNextFixture::new(
            "qwen38-flash-next-sidecar-truncated",
            &pinned_qwen38_flash_next_config(),
            &sidecar,
        );

        let error = validate_config::<Qwen38FlashNext>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Config);
        assert!(
            error
                .to_string()
                .contains("quantization.exclude_modules length"),
            "{error}"
        );
    }

    #[test]
    fn refuses_a_snapshot_with_no_hf_quant_config_sidecar() {
        let fixture = Qwen38FlashNextFixture::pinned(
            "qwen38-flash-next-sidecar-missing",
            &pinned_qwen38_flash_next_config(),
        );
        fs::remove_file(fixture.root.join(QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE)).unwrap();

        let error = validate_config::<Qwen38FlashNext>(&fixture.config())
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Io);
        assert!(
            error
                .to_string()
                .contains(QWEN38_FLASH_NEXT_HF_QUANT_CONFIG_FILE),
            "{error}"
        );
    }
}

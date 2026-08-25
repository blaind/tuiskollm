//! Qwen3.5-9B ModelOpt NVFP4 source bindings.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::modelopt_codec::ModelOptNvfp4LinearBindings;
use crate::common::naming::{EMBEDDING, FINAL_NORM, LM_HEAD};
use crate::common::routes::{
    require_full_attention_layer, require_gdn_layer, require_same_rank_zero_f32,
};
use crate::{Arch, Bf16View, CheckpointContract, CheckpointError, CheckpointResult, TensorView};

/// Complete ModelOpt NVFP4 source family for one Qwen3.5 MLP boundary.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptNvfp4MlpBindings<'a> {
    /// Gate projection packed weights and exact source scales.
    pub gate: ModelOptNvfp4LinearBindings<'a>,
    /// Up projection packed weights and exact source scales.
    pub up: ModelOptNvfp4LinearBindings<'a>,
    /// Down projection packed weights and exact source scales.
    pub down: ModelOptNvfp4LinearBindings<'a>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary `[hidden]`.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this MLP boundary.
    pub layer: usize,
    pub(crate) layer_count: usize,
}

impl<'a> ModelOptNvfp4MlpBindings<'a> {
    /// Binds one exact Qwen3.5 ModelOpt NVFP4 MLP source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(layer, |name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_modelopt_nvfp4_mlp_layer::<A>(layer)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let gate = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{mlp_prefix}.gate_proj"),
            A::INTERMEDIATE,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let up = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{mlp_prefix}.up_proj"),
            A::INTERMEDIATE,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let down = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{mlp_prefix}.down_proj"),
            A::HIDDEN,
            A::INTERMEDIATE,
            layer,
            |name| tensor(name),
        )?;

        require_same_rank_zero_f32(
            layer,
            "gate/up input_scale",
            &gate.input_scale,
            &up.input_scale,
        )?;
        require_same_rank_zero_f32(
            layer,
            "gate/up weight_scale_2",
            &gate.weight_scale_2,
            &up.weight_scale_2,
        )?;

        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [A::HIDDEN as u64],
        )?;
        let next_norm = Bf16View::bind(
            tensor(&modelopt_nvfp4_next_norm_name::<A>(layer)?)?,
            [A::HIDDEN as u64],
        )?;

        Ok(Self {
            gate,
            up,
            down,
            input_norm,
            next_norm,
            layer,
            layer_count: A::LAYERS,
        })
    }
}

fn modelopt_nvfp4_next_norm_name<A: Arch>(layer: usize) -> CheckpointResult<String> {
    require_modelopt_nvfp4_mlp_layer::<A>(layer)?;
    let next_layer = layer
        .checked_add(1)
        .ok_or_else(|| CheckpointError::source_binding("ModelOpt NVFP4 MLP layer overflows"))?;

    Ok(if next_layer == A::LAYERS {
        FINAL_NORM.to_string()
    } else {
        format!("model.language_model.layers.{next_layer}.input_layernorm.weight")
    })
}

/// Complete ModelOpt NVFP4 source planes for one Qwen3.5 full-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptNvfp4AttentionBindings<'a> {
    /// Query-plus-gate projection.
    pub query_gate: ModelOptNvfp4LinearBindings<'a>,
    /// Key projection.
    pub key: ModelOptNvfp4LinearBindings<'a>,
    /// Value projection.
    pub value: ModelOptNvfp4LinearBindings<'a>,
    /// Gated attention-output projection.
    pub output: ModelOptNvfp4LinearBindings<'a>,
    /// Per-head query RMSNorm weights `[head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights `[head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before attention `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
    pub(crate) layer_count: usize,
    pub(crate) full_attention_interval: usize,
}

impl<'a> ModelOptNvfp4AttentionBindings<'a> {
    /// Binds one exact Qwen3.5 ModelOpt NVFP4 full-attention source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(layer, |name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_modelopt_contract::<A>("full attention")?;
        require_full_attention_layer(layer, A::LAYERS, A::FULL_ATTENTION_INTERVAL)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let query_gate = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.q_proj"),
            A::ATTENTION_QUERY_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let key = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.k_proj"),
            A::ATTENTION_KV_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let value = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.v_proj"),
            A::ATTENTION_KV_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let output = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.o_proj"),
            A::HIDDEN,
            A::ATTENTION_OUTPUT_COLUMNS,
            layer,
            |name| tensor(name),
        )?;

        require_same_rank_zero_f32(
            layer,
            "query/key input_scale",
            &query_gate.input_scale,
            &key.input_scale,
        )?;
        require_same_rank_zero_f32(
            layer,
            "query/value input_scale",
            &query_gate.input_scale,
            &value.input_scale,
        )?;

        Ok(Self {
            query_gate,
            key,
            value,
            output,
            query_norm: Bf16View::bind(
                tensor(&format!("{prefix}.q_norm.weight"))?,
                [A::HEAD_DIM as u64],
            )?,
            key_norm: Bf16View::bind(
                tensor(&format!("{prefix}.k_norm.weight"))?,
                [A::HEAD_DIM as u64],
            )?,
            input_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
                [A::HIDDEN as u64],
            )?,
            post_attention_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
                [A::HIDDEN as u64],
            )?,
            layer,
            layer_count: A::LAYERS,
            full_attention_interval: A::FULL_ATTENTION_INTERVAL,
        })
    }
}

/// Complete ModelOpt NVFP4 source planes for one Qwen3.5 GDN mixer layer.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptNvfp4GdnBindings<'a> {
    /// Fused query, key, and value projection.
    pub qkv: ModelOptNvfp4LinearBindings<'a>,
    /// Z gate projection.
    pub z: ModelOptNvfp4LinearBindings<'a>,
    /// Per-value-head A-control projection.
    pub a_control: ModelOptNvfp4LinearBindings<'a>,
    /// Per-value-head B-control projection.
    pub b_control: ModelOptNvfp4LinearBindings<'a>,
    /// Recurrent-state output projection.
    pub output: ModelOptNvfp4LinearBindings<'a>,
    /// Width-four causal-convolution weights `[gdn_qkv_rows, 1, kernel]`.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters `[gdn_control_rows]`.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias `[gdn_control_rows]`.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights `[linear_head_dim]`.
    pub norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the mixer `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
    pub(crate) layer_count: usize,
    pub(crate) full_attention_interval: usize,
}

impl<'a> ModelOptNvfp4GdnBindings<'a> {
    /// Binds one exact Qwen3.5 ModelOpt NVFP4 GDN source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(layer, |name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_modelopt_contract::<A>("GDN")?;
        require_gdn_layer::<A>(layer)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let qkv = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.in_proj_qkv"),
            A::GDN_QKV_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let z = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.in_proj_z"),
            A::GDN_VALUE_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let a_control = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.in_proj_a"),
            A::GDN_CONTROL_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let b_control = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.in_proj_b"),
            A::GDN_CONTROL_ROWS,
            A::HIDDEN,
            layer,
            |name| tensor(name),
        )?;
        let output = ModelOptNvfp4LinearBindings::bind_from(
            &format!("{prefix}.out_proj"),
            A::HIDDEN,
            A::GDN_VALUE_ROWS,
            layer,
            |name| tensor(name),
        )?;

        for (role, scale) in [
            ("Z", &z.input_scale),
            ("A-control", &a_control.input_scale),
            ("B-control", &b_control.input_scale),
        ] {
            require_same_rank_zero_f32(
                layer,
                &format!("QKV/{role} input_scale"),
                &qkv.input_scale,
                scale,
            )?;
        }

        Ok(Self {
            qkv,
            z,
            a_control,
            b_control,
            output,
            convolution_weight: Bf16View::bind(
                tensor(&format!("{prefix}.conv1d.weight"))?,
                [A::GDN_QKV_ROWS as u64, 1, A::LINEAR_CONV_KERNEL_DIM as u64],
            )?,
            a_log: Bf16View::bind(
                tensor(&format!("{prefix}.A_log"))?,
                [A::GDN_CONTROL_ROWS as u64],
            )?,
            dt_bias: Bf16View::bind(
                tensor(&format!("{prefix}.dt_bias"))?,
                [A::GDN_CONTROL_ROWS as u64],
            )?,
            norm: Bf16View::bind(
                tensor(&format!("{prefix}.norm.weight"))?,
                [A::LINEAR_HEAD_DIM as u64],
            )?,
            input_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
                [A::HIDDEN as u64],
            )?,
            post_attention_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
                [A::HIDDEN as u64],
            )?,
            layer,
            layer_count: A::LAYERS,
            full_attention_interval: A::FULL_ATTENTION_INTERVAL,
        })
    }
}

/// BF16 text endpoint sources used by the exact Qwen3.5 checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct Bf16TextEndpointBindings<'a> {
    /// BF16 token embedding matrix `[vocab, hidden]`.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 final RMSNorm weights `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
    /// Untied BF16 language-model head `[vocab, hidden]`.
    pub lm_head: Bf16View<'a, 2>,
}

impl<'a> Bf16TextEndpointBindings<'a> {
    /// Binds the exact Qwen3.5 embedding, final norm, and BF16 LM head.
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(|name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_modelopt_contract::<A>("text endpoints")?;

        let vocab = A::VOCAB as u64;
        let hidden = A::HIDDEN as u64;

        Ok(Self {
            embedding: Bf16View::bind(tensor(EMBEDDING)?, [vocab, hidden])?,
            final_norm: Bf16View::bind(tensor(FINAL_NORM)?, [hidden])?,
            lm_head: Bf16View::bind(tensor(LM_HEAD)?, [vocab, hidden])?,
        })
    }
}

fn require_modelopt_nvfp4_mlp_layer<A: Arch>(layer: usize) -> CheckpointResult<()> {
    require_modelopt_contract::<A>("MLP")?;

    if layer >= A::LAYERS {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted ModelOpt NVFP4 MLP source contract"
        )));
    }

    Ok(())
}

fn require_modelopt_contract<A: Arch>(role: &str) -> CheckpointResult<()> {
    if A::CHECKPOINT_CONTRACT != CheckpointContract::ModelOptNvfp4 {
        return Err(CheckpointError::source_binding(format!(
            "{role} does not use the admitted ModelOpt NVFP4 source contract"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::routes::{E2M1_VALUES_PER_BYTE, NVFP4_GROUP_SIZE};
    use crate::common::test_support::sources::{
        Nvfp4Arch, append_bf16_tensor, fixture_path, write_safetensors_payload,
    };
    use crate::{Arch, CheckpointContract, CheckpointErrorCode, SafeTensorFile};
    use serde_json::{Value, json};
    use std::fs;

    #[derive(Clone, Copy)]
    struct ModelOptArch;

    impl Arch for ModelOptArch {
        const MODEL_ID: &'static str = "test/modelopt";
        const REVISION: &'static str = "test-revision";
        const CHECKPOINT_CONTRACT: CheckpointContract = CheckpointContract::ModelOptNvfp4;
        const HIDDEN: usize = 32;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 16;
        const VOCAB: usize = 3;
        const LAYERS: usize = 2;
        const FULL_ATTENTION_INTERVAL: usize = 2;
        const NUM_ATTENTION_HEADS: usize = 16;
        const NUM_KV_HEADS: usize = 16;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 16;
        const LINEAR_VALUE_HEADS: usize = 16;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 4;
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

    fn modelopt_endpoint_fixture() -> (Value, Vec<u8>) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for (name, shape) in [
            (
                "model.language_model.embed_tokens.weight",
                vec![ModelOptArch::VOCAB, ModelOptArch::HIDDEN],
            ),
            (
                "model.language_model.norm.weight",
                vec![ModelOptArch::HIDDEN],
            ),
            (
                "lm_head.weight",
                vec![ModelOptArch::VOCAB, ModelOptArch::HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    fn modelopt_nvfp4_mlp_fixture(layer: usize) -> (Value, Vec<u8>) {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let next_layer_prefix = format!("model.language_model.layers.{}", layer + 1);
        let mut payload = vec![0x38; 1_016];

        payload[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        payload[4..8].copy_from_slice(&0.125f32.to_le_bytes());
        payload[8..12].copy_from_slice(&0.25f32.to_le_bytes());
        payload[12..16].copy_from_slice(&0.125f32.to_le_bytes());
        payload[16..20].copy_from_slice(&0.5f32.to_le_bytes());
        payload[20..24].copy_from_slice(&0.0625f32.to_le_bytes());
        payload[888..952].fill(0x70);
        payload[952..1_016].fill(0x80);

        (
            json!({
                format!("{prefix}.gate_proj.input_scale"): {
                    "dtype":"F32", "shape":[], "data_offsets":[0,4]
                },
                format!("{prefix}.gate_proj.weight_scale_2"): {
                    "dtype":"F32", "shape":[], "data_offsets":[4,8]
                },
                format!("{prefix}.up_proj.input_scale"): {
                    "dtype":"F32", "shape":[], "data_offsets":[8,12]
                },
                format!("{prefix}.up_proj.weight_scale_2"): {
                    "dtype":"F32", "shape":[], "data_offsets":[12,16]
                },
                format!("{prefix}.down_proj.input_scale"): {
                    "dtype":"F32", "shape":[], "data_offsets":[16,20]
                },
                format!("{prefix}.down_proj.weight_scale_2"): {
                    "dtype":"F32", "shape":[], "data_offsets":[20,24]
                },
                format!("{prefix}.gate_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[24,56]
                },
                format!("{prefix}.up_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[56,88]
                },
                format!("{prefix}.down_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[32,1], "data_offsets":[88,120]
                },
                format!("{prefix}.gate_proj.weight"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[120,376]
                },
                format!("{prefix}.up_proj.weight"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[376,632]
                },
                format!("{prefix}.down_proj.weight"): {
                    "dtype":"U8", "shape":[32,8], "data_offsets":[632,888]
                },
                format!("{layer_prefix}.post_attention_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[888,952]
                },
                format!("{next_layer_prefix}.input_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[952,1016]
                }
            }),
            payload,
        )
    }

    fn append_modelopt_linear(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        prefix: &str,
        geometry: [usize; 2],
        scales: [f32; 2],
        weight_code: u8,
    ) {
        let [rows, columns] = geometry;
        let [input_scale, weight_scale_2] = scales;

        for (suffix, value) in [
            ("input_scale", input_scale),
            ("weight_scale_2", weight_scale_2),
        ] {
            let begin = payload.len();
            payload.extend_from_slice(&value.to_le_bytes());
            header.insert(
                format!("{prefix}.{suffix}"),
                json!({
                    "dtype": "F32",
                    "shape": [],
                    "data_offsets": [begin, payload.len()]
                }),
            );
        }

        let scale_begin = payload.len();
        payload.resize(scale_begin + rows * (columns / NVFP4_GROUP_SIZE), 0x38);
        header.insert(
            format!("{prefix}.weight_scale"),
            json!({
                "dtype": "F8_E4M3",
                "shape": [rows, columns / NVFP4_GROUP_SIZE],
                "data_offsets": [scale_begin, payload.len()]
            }),
        );

        let weight_begin = payload.len();
        payload.resize(
            weight_begin + rows * (columns / E2M1_VALUES_PER_BYTE),
            weight_code,
        );
        header.insert(
            format!("{prefix}.weight"),
            json!({
                "dtype": "U8",
                "shape": [rows, columns / E2M1_VALUES_PER_BYTE],
                "data_offsets": [weight_begin, payload.len()]
            }),
        );
    }

    fn modelopt_gdn_fixture(layer: usize) -> (Value, Vec<u8>) {
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for (projection, rows, input_scale, weight_scale, code) in [
            ("in_proj_qkv", ModelOptArch::GDN_QKV_ROWS, 0.25, 0.125, 0x10),
            ("in_proj_z", ModelOptArch::GDN_VALUE_ROWS, 0.25, 0.25, 0x20),
            ("in_proj_a", ModelOptArch::GDN_CONTROL_ROWS, 0.25, 0.5, 0x30),
            ("in_proj_b", ModelOptArch::GDN_CONTROL_ROWS, 0.25, 1.0, 0x40),
        ] {
            append_modelopt_linear(
                &mut header,
                &mut payload,
                &format!("{prefix}.{projection}"),
                [rows, ModelOptArch::HIDDEN],
                [input_scale, weight_scale],
                code,
            );
        }
        append_modelopt_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.out_proj"),
            [ModelOptArch::HIDDEN, ModelOptArch::GDN_VALUE_ROWS],
            [0.5, 0.0625],
            0x50,
        );

        for (name, shape) in [
            (
                format!("{prefix}.conv1d.weight"),
                vec![
                    ModelOptArch::GDN_QKV_ROWS,
                    1,
                    ModelOptArch::LINEAR_CONV_KERNEL_DIM,
                ],
            ),
            (
                format!("{prefix}.A_log"),
                vec![ModelOptArch::GDN_CONTROL_ROWS],
            ),
            (
                format!("{prefix}.dt_bias"),
                vec![ModelOptArch::GDN_CONTROL_ROWS],
            ),
            (
                format!("{prefix}.norm.weight"),
                vec![ModelOptArch::LINEAR_HEAD_DIM],
            ),
            (
                format!("{layer_prefix}.input_layernorm.weight"),
                vec![ModelOptArch::HIDDEN],
            ),
            (
                format!("{layer_prefix}.post_attention_layernorm.weight"),
                vec![ModelOptArch::HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    fn modelopt_attention_fixture(layer: usize) -> (Value, Vec<u8>) {
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for (projection, rows, input_scale, weight_scale, code) in [
            (
                "q_proj",
                ModelOptArch::ATTENTION_QUERY_ROWS,
                0.25,
                0.125,
                0x10,
            ),
            ("k_proj", ModelOptArch::ATTENTION_KV_ROWS, 0.25, 0.25, 0x20),
            ("v_proj", ModelOptArch::ATTENTION_KV_ROWS, 0.25, 0.5, 0x30),
        ] {
            append_modelopt_linear(
                &mut header,
                &mut payload,
                &format!("{prefix}.{projection}"),
                [rows, ModelOptArch::HIDDEN],
                [input_scale, weight_scale],
                code,
            );
        }
        append_modelopt_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.o_proj"),
            [ModelOptArch::HIDDEN, ModelOptArch::ATTENTION_OUTPUT_COLUMNS],
            [0.5, 0.0625],
            0x40,
        );

        for (name, shape) in [
            (
                format!("{prefix}.q_norm.weight"),
                vec![ModelOptArch::HEAD_DIM],
            ),
            (
                format!("{prefix}.k_norm.weight"),
                vec![ModelOptArch::HEAD_DIM],
            ),
            (
                format!("{layer_prefix}.input_layernorm.weight"),
                vec![ModelOptArch::HIDDEN],
            ),
            (
                format!("{layer_prefix}.post_attention_layernorm.weight"),
                vec![ModelOptArch::HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    #[test]
    fn binds_exact_bf16_text_endpoint_contract() {
        let path = fixture_path("modelopt-endpoints");
        let (header, payload) = modelopt_endpoint_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Bf16TextEndpointBindings::bind_from::<ModelOptArch>(|name| file.tensor(name)).unwrap();

        assert_eq!(bindings.embedding.shape(), &[3, 32]);
        assert_eq!(bindings.final_norm.shape(), &[32]);
        assert_eq!(bindings.lm_head.shape(), &[3, 32]);
        assert_ne!(bindings.embedding.word(0), bindings.lm_head.word(0));

        let error = Bf16TextEndpointBindings::bind_from::<Nvfp4Arch>(|_| {
            panic!("the contract check must reject before tensor lookup")
        })
        .unwrap_err();
        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("text endpoints"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_exact_modelopt_nvfp4_mlp_source_contract() {
        let path = fixture_path("modelopt-nvfp4-mlp");
        let (header, payload) = modelopt_nvfp4_mlp_fixture(0);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            ModelOptNvfp4MlpBindings::bind_from::<ModelOptArch>(0, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.gate.weight.shape(), &[16, 16]);
        assert_eq!(bindings.gate.block_scale.shape(), &[16, 2]);
        assert_eq!(bindings.gate.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.gate.weight_scale_2.value(0), Some(0.125));
        assert_eq!(bindings.up.weight.shape(), &[16, 16]);
        assert_eq!(bindings.down.weight.shape(), &[32, 8]);
        assert_eq!(bindings.down.block_scale.shape(), &[32, 1]);
        assert_eq!(bindings.down.input_scale.value(0), Some(0.5));
        assert_eq!(bindings.down.weight_scale_2.value(0), Some(0.0625));
        assert_eq!((bindings.gate.rows, bindings.gate.columns), (16, 32));
        assert_eq!((bindings.down.rows, bindings.down.columns), (32, 16));
        assert_eq!(bindings.input_norm.bytes()[0], 0x70);
        assert_eq!(bindings.next_norm.bytes()[0], 0x80);
        assert_eq!(bindings.layer, 0);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_modelopt_nvfp4_scale_and_route_drift() {
        let (header, payload) = modelopt_nvfp4_mlp_fixture(0);

        for (label, offset, bytes, expected) in [
            (
                "modelopt-nonpositive-scale",
                0,
                f32::NAN.to_le_bytes(),
                "finite positive F32 scale",
            ),
            (
                "modelopt-gate-up-scale-mismatch",
                8,
                0.5f32.to_le_bytes(),
                "gate/up input_scale values differ",
            ),
        ] {
            let path = fixture_path(label);
            let mut changed = payload.clone();
            changed[offset..offset + 4].copy_from_slice(&bytes);
            write_safetensors_payload(&path, header.clone(), &changed);
            let file = SafeTensorFile::open(&path).unwrap();

            let error =
                ModelOptNvfp4MlpBindings::bind_from::<ModelOptArch>(0, |name| file.tensor(name))
                    .unwrap_err();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains(expected), "{error}");
            fs::remove_file(path).unwrap();
        }

        let path = fixture_path("modelopt-invalid-block-scale");
        let mut changed = payload;
        changed[24] = 0x7f;
        write_safetensors_payload(&path, header, &changed);
        let file = SafeTensorFile::open(&path).unwrap();
        let error =
            ModelOptNvfp4MlpBindings::bind_from::<ModelOptArch>(0, |name| file.tensor(name))
                .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("scale plane"));
        fs::remove_file(path).unwrap();

        let error = ModelOptNvfp4MlpBindings::bind_from::<Nvfp4Arch>(0, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("ModelOpt NVFP4 source contract"));
    }

    #[test]
    fn binds_exact_modelopt_nvfp4_attention_source_contract() {
        let path = fixture_path("modelopt-attention");
        let (header, payload) = modelopt_attention_fixture(1);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            ModelOptNvfp4AttentionBindings::bind_from::<ModelOptArch>(1, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.query_gate.weight.shape(), &[32, 16]);
        assert_eq!(bindings.query_gate.block_scale.shape(), &[32, 2]);
        assert_eq!(bindings.key.weight.shape(), &[16, 16]);
        assert_eq!(bindings.value.weight.shape(), &[16, 16]);
        assert_eq!(bindings.output.weight.shape(), &[32, 8]);
        assert_eq!(bindings.output.block_scale.shape(), &[32, 1]);
        assert_eq!(bindings.query_gate.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.output.input_scale.value(0), Some(0.5));
        assert_eq!(bindings.query_norm.shape(), &[1]);
        assert_eq!(bindings.key_norm.shape(), &[1]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 1);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_modelopt_attention_route_and_shared_input_scale_drift() {
        let path = fixture_path("modelopt-attention-scale-drift");
        let (header, mut payload) = modelopt_attention_fixture(1);
        let offset =
            header["model.language_model.layers.1.self_attn.k_proj.input_scale"]["data_offsets"][0]
                .as_u64()
                .unwrap() as usize;
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error =
            ModelOptNvfp4AttentionBindings::bind_from::<ModelOptArch>(1, |name| file.tensor(name))
                .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("query/key input_scale values differ")
        );
        fs::remove_file(path).unwrap();

        let error = ModelOptNvfp4AttentionBindings::bind_from::<ModelOptArch>(0, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("full-attention source contract"));
    }

    #[test]
    fn binds_exact_modelopt_nvfp4_gdn_source_contract() {
        let path = fixture_path("modelopt-gdn");
        let (header, payload) = modelopt_gdn_fixture(0);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            ModelOptNvfp4GdnBindings::bind_from::<ModelOptArch>(0, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.qkv.weight.shape(), &[48, 16]);
        assert_eq!(bindings.qkv.block_scale.shape(), &[48, 2]);
        assert_eq!(bindings.z.weight.shape(), &[16, 16]);
        assert_eq!(bindings.a_control.weight.shape(), &[16, 16]);
        assert_eq!(bindings.b_control.weight.shape(), &[16, 16]);
        assert_eq!(bindings.output.weight.shape(), &[32, 8]);
        assert_eq!(bindings.output.block_scale.shape(), &[32, 1]);
        assert_eq!(bindings.qkv.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.output.input_scale.value(0), Some(0.5));
        assert_eq!(bindings.convolution_weight.shape(), &[48, 1, 4]);
        assert_eq!(bindings.a_log.shape(), &[16]);
        assert_eq!(bindings.dt_bias.shape(), &[16]);
        assert_eq!(bindings.norm.shape(), &[1]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 0);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_modelopt_gdn_route_and_shared_input_scale_drift() {
        let path = fixture_path("modelopt-gdn-scale-drift");
        let (header, mut payload) = modelopt_gdn_fixture(0);
        let offset = header["model.language_model.layers.0.linear_attn.in_proj_z.input_scale"]
            ["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error =
            ModelOptNvfp4GdnBindings::bind_from::<ModelOptArch>(0, |name| file.tensor(name))
                .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("QKV/Z input_scale values differ")
        );
        fs::remove_file(path).unwrap();

        let error = ModelOptNvfp4GdnBindings::bind_from::<ModelOptArch>(1, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("GDN source contract"));
    }
}

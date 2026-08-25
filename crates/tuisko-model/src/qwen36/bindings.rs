//! Qwen3.6-35B-A3B MoE source bindings.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::modelopt_codec::ModelOptNvfp4LinearBindings;
use crate::common::naming::{EMBEDDING, FINAL_NORM};
use crate::common::routes::{
    positive_rank_zero_f32, require_full_attention_layer, require_gdn_layer_route,
    require_same_rank_zero_f32,
};
use crate::{
    Arch, Bf16View, CheckpointError, CheckpointResult, F32View, Fp8E4M3View, Qwen36Moe35B,
    TensorView,
};

/// Complete BF16 source planes for the admitted Qwen3.6 MoE MTP layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36MtpBindings<'a> {
    /// Projection combining draft hidden and base embedding inputs `[hidden, 2 * hidden]`.
    pub input_projection: Bf16View<'a, 2>,
    /// Normalization for the base embedding input `[hidden]`.
    pub embedding_norm: Bf16View<'a, 1>,
    /// Normalization for the draft hidden input `[hidden]`.
    pub hidden_norm: Bf16View<'a, 1>,
    /// Query rows followed by gate rows `[attention_query_rows, hidden]`.
    pub query_gate_weight: Bf16View<'a, 2>,
    /// Key projection `[attention_kv_rows, hidden]`.
    pub key_weight: Bf16View<'a, 2>,
    /// Value projection `[attention_kv_rows, hidden]`.
    pub value_weight: Bf16View<'a, 2>,
    /// Attention output projection `[hidden, attention_output_columns]`.
    pub attention_output_weight: Bf16View<'a, 2>,
    /// Per-head query RMSNorm weights `[head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights `[head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before attention `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Top-eight router weights `[experts, hidden]`.
    pub router_weight: Bf16View<'a, 2>,
    /// Routed expert gate/up planes `[experts, 2 * intermediate, hidden]`.
    pub routed_gate_up_weight: Bf16View<'a, 3>,
    /// Routed expert down planes `[experts, hidden, intermediate]`.
    pub routed_down_weight: Bf16View<'a, 3>,
    /// Shared-expert gate weights `[1, hidden]`.
    pub shared_expert_gate_weight: Bf16View<'a, 2>,
    /// Shared-expert gate projection `[intermediate, hidden]`.
    pub shared_gate_weight: Bf16View<'a, 2>,
    /// Shared-expert up projection `[intermediate, hidden]`.
    pub shared_up_weight: Bf16View<'a, 2>,
    /// Shared-expert down projection `[hidden, intermediate]`.
    pub shared_down_weight: Bf16View<'a, 2>,
    /// Zero-centered RMSNorm weights before the MoE boundary `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Final draft hidden-state normalization `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
}

impl<'a> Qwen36MtpBindings<'a> {
    /// Binds the exact admitted Qwen3.6 MTP source family.
    pub fn bind(snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>) -> CheckpointResult<Self> {
        Self::bind_from(
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::INTERMEDIATE,
            Qwen36Moe35B::NUM_EXPERTS,
            Qwen36Moe35B::ATTENTION_QUERY_ROWS,
            Qwen36Moe35B::ATTENTION_KV_ROWS,
            Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS,
            Qwen36Moe35B::HEAD_DIM,
            |name| snapshot.tensor(name),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_from(
        hidden: usize,
        intermediate: usize,
        experts: usize,
        query_rows: usize,
        kv_rows: usize,
        output_columns: usize,
        head_dim: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let input_columns = hidden.checked_mul(2).ok_or_else(|| {
            CheckpointError::source_binding("Qwen3.6 MTP input projection width overflows")
        })?;
        let gate_up_rows = intermediate.checked_mul(2).ok_or_else(|| {
            CheckpointError::source_binding("Qwen3.6 MTP gate/up row count overflows")
        })?;
        let prefix = "mtp.layers.0";
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");

        Ok(Self {
            input_projection: Bf16View::bind(
                tensor("mtp.fc.weight")?,
                [hidden as u64, input_columns as u64],
            )?,
            embedding_norm: Bf16View::bind(
                tensor("mtp.pre_fc_norm_embedding.weight")?,
                [hidden as u64],
            )?,
            hidden_norm: Bf16View::bind(tensor("mtp.pre_fc_norm_hidden.weight")?, [hidden as u64])?,
            query_gate_weight: Bf16View::bind(
                tensor(&format!("{attention}.q_proj.weight"))?,
                [query_rows as u64, hidden as u64],
            )?,
            key_weight: Bf16View::bind(
                tensor(&format!("{attention}.k_proj.weight"))?,
                [kv_rows as u64, hidden as u64],
            )?,
            value_weight: Bf16View::bind(
                tensor(&format!("{attention}.v_proj.weight"))?,
                [kv_rows as u64, hidden as u64],
            )?,
            attention_output_weight: Bf16View::bind(
                tensor(&format!("{attention}.o_proj.weight"))?,
                [hidden as u64, output_columns as u64],
            )?,
            query_norm: Bf16View::bind(
                tensor(&format!("{attention}.q_norm.weight"))?,
                [head_dim as u64],
            )?,
            key_norm: Bf16View::bind(
                tensor(&format!("{attention}.k_norm.weight"))?,
                [head_dim as u64],
            )?,
            input_norm: Bf16View::bind(
                tensor(&format!("{prefix}.input_layernorm.weight"))?,
                [hidden as u64],
            )?,
            router_weight: Bf16View::bind(
                tensor(&format!("{mlp}.gate.weight"))?,
                [experts as u64, hidden as u64],
            )?,
            routed_gate_up_weight: Bf16View::bind(
                tensor(&format!("{mlp}.experts.gate_up_proj"))?,
                [experts as u64, gate_up_rows as u64, hidden as u64],
            )?,
            routed_down_weight: Bf16View::bind(
                tensor(&format!("{mlp}.experts.down_proj"))?,
                [experts as u64, hidden as u64, intermediate as u64],
            )?,
            shared_expert_gate_weight: Bf16View::bind(
                tensor(&format!("{mlp}.shared_expert_gate.weight"))?,
                [1, hidden as u64],
            )?,
            shared_gate_weight: Bf16View::bind(
                tensor(&format!("{mlp}.shared_expert.gate_proj.weight"))?,
                [intermediate as u64, hidden as u64],
            )?,
            shared_up_weight: Bf16View::bind(
                tensor(&format!("{mlp}.shared_expert.up_proj.weight"))?,
                [intermediate as u64, hidden as u64],
            )?,
            shared_down_weight: Bf16View::bind(
                tensor(&format!("{mlp}.shared_expert.down_proj.weight"))?,
                [hidden as u64, intermediate as u64],
            )?,
            post_attention_norm: Bf16View::bind(
                tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
                [hidden as u64],
            )?,
            final_norm: Bf16View::bind(tensor("mtp.norm.weight")?, [hidden as u64])?,
        })
    }
}

/// Exact NVFP4 planes for one Qwen3.6 routed or shared expert.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36MoeExpertBindings<'a> {
    /// Gate projection.
    pub gate: ModelOptNvfp4LinearBindings<'a>,
    /// Up projection.
    pub up: ModelOptNvfp4LinearBindings<'a>,
    /// Down projection.
    pub down: ModelOptNvfp4LinearBindings<'a>,
}

/// Complete Qwen3.6 MoE source family for one decoder layer.
#[derive(Clone, Debug)]
pub struct Qwen36MoeLayerBindings<'a> {
    /// Router weights `[experts, hidden]`.
    pub router_weight: Bf16View<'a, 2>,
    /// Shared-expert gate weights `[1, hidden]`.
    pub shared_expert_gate_weight: Bf16View<'a, 2>,
    /// Routed experts in numeric expert order.
    pub experts: Vec<Qwen36MoeExpertBindings<'a>>,
    /// Always-active shared expert.
    pub shared_expert: Qwen36MoeExpertBindings<'a>,
    /// Zero-centered RMSNorm weights before the MoE boundary `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary `[hidden]`.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen36MoeLayerBindings<'a> {
    /// Binds one exact Qwen3.6 MoE source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(
            layer,
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::INTERMEDIATE,
            Qwen36Moe35B::SHARED_EXPERT_INTERMEDIATE,
            Qwen36Moe35B::NUM_EXPERTS,
            |name| snapshot.tensor(name),
        )
    }

    fn bind_from(
        layer: usize,
        layer_count: usize,
        hidden: usize,
        expert_intermediate: usize,
        shared_intermediate: usize,
        expert_count: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_qwen36_moe_layer(layer, layer_count)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let router_weight = Bf16View::bind(
            tensor(&format!("{mlp_prefix}.gate.weight"))?,
            [expert_count as u64, hidden as u64],
        )?;
        let shared_expert_gate_weight = Bf16View::bind(
            tensor(&format!("{mlp_prefix}.shared_expert_gate.weight"))?,
            [1, hidden as u64],
        )?;
        let mut experts = Vec::with_capacity(expert_count);

        for expert in 0..expert_count {
            experts.push(bind_qwen36_expert(
                &format!("{mlp_prefix}.experts.{expert}"),
                hidden,
                expert_intermediate,
                layer,
                &format!("expert-{expert}"),
                |name| tensor(name),
            )?);
        }

        let shared_expert = bind_qwen36_expert(
            &format!("{mlp_prefix}.shared_expert"),
            hidden,
            shared_intermediate,
            layer,
            "shared expert",
            |name| tensor(name),
        )?;
        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [hidden as u64],
        )?;
        let next_norm = Bf16View::bind(
            tensor(&qwen36_next_norm_name(layer, layer_count)?)?,
            [hidden as u64],
        )?;

        Ok(Self {
            router_weight,
            shared_expert_gate_weight,
            experts,
            shared_expert,
            input_norm,
            next_norm,
            layer,
        })
    }
}

fn bind_qwen36_expert<'a>(
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    layer: usize,
    role: &str,
    mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
) -> CheckpointResult<Qwen36MoeExpertBindings<'a>> {
    let gate = ModelOptNvfp4LinearBindings::bind_from(
        &format!("{prefix}.gate_proj"),
        intermediate,
        hidden,
        layer,
        |name| tensor(name),
    )?;
    let up = ModelOptNvfp4LinearBindings::bind_from(
        &format!("{prefix}.up_proj"),
        intermediate,
        hidden,
        layer,
        |name| tensor(name),
    )?;
    let down = ModelOptNvfp4LinearBindings::bind_from(
        &format!("{prefix}.down_proj"),
        hidden,
        intermediate,
        layer,
        |name| tensor(name),
    )?;

    require_same_rank_zero_f32(
        layer,
        &format!("{role} gate/up input_scale"),
        &gate.input_scale,
        &up.input_scale,
    )?;
    require_same_rank_zero_f32(
        layer,
        &format!("{role} gate/up weight_scale_2"),
        &gate.weight_scale_2,
        &up.weight_scale_2,
    )?;

    Ok(Qwen36MoeExpertBindings { gate, up, down })
}

fn qwen36_next_norm_name(layer: usize, layer_count: usize) -> CheckpointResult<String> {
    require_qwen36_moe_layer(layer, layer_count)?;
    let next_layer = layer
        .checked_add(1)
        .ok_or_else(|| CheckpointError::source_binding("Qwen3.6 MoE layer overflows"))?;

    Ok(if next_layer == layer_count {
        FINAL_NORM.to_string()
    } else {
        format!("model.language_model.layers.{next_layer}.input_layernorm.weight")
    })
}

fn require_qwen36_moe_layer(layer: usize, layer_count: usize) -> CheckpointResult<()> {
    if layer >= layer_count {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted Qwen3.6 MoE source contract"
        )));
    }

    Ok(())
}

/// Exact scalar-scaled FP8 source plane used by Qwen3.6 projections.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36Fp8LinearBindings<'a> {
    /// Source E4M3 weights `[rows, columns]`.
    pub weight: Fp8E4M3View<'a, 2>,
    /// Positive source activation scale stored as one rank-zero F32 value.
    pub input_scale: F32View<'a, 0>,
    /// Positive source weight scale stored as one rank-zero F32 value.
    pub weight_scale: F32View<'a, 0>,
    /// Logical output row count.
    pub rows: usize,
    /// Logical input column count.
    pub columns: usize,
}

impl<'a> Qwen36Fp8LinearBindings<'a> {
    fn bind_from(
        prefix: &str,
        rows: usize,
        columns: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        Ok(Self {
            input_scale: positive_rank_zero_f32(tensor(&format!("{prefix}.input_scale"))?)?,
            weight: Fp8E4M3View::bind(
                tensor(&format!("{prefix}.weight"))?,
                [rows as u64, columns as u64],
            )?,
            weight_scale: positive_rank_zero_f32(tensor(&format!("{prefix}.weight_scale"))?)?,
            rows,
            columns,
        })
    }
}

/// Complete mixed-FP8/BF16 source family for one Qwen3.6 GDN layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36GdnBindings<'a> {
    /// Fused query, key, and value FP8 projection.
    pub qkv: Qwen36Fp8LinearBindings<'a>,
    /// Z-gate FP8 projection.
    pub z: Qwen36Fp8LinearBindings<'a>,
    /// Per-value-head BF16 A-control projection `[control_rows, hidden]`.
    pub a_control: Bf16View<'a, 2>,
    /// Per-value-head BF16 B-control projection `[control_rows, hidden]`.
    pub b_control: Bf16View<'a, 2>,
    /// Recurrent-state FP8 output projection.
    pub output: Qwen36Fp8LinearBindings<'a>,
    /// Width-four causal-convolution weights `[qkv_rows, 1, kernel]`.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters `[control_rows]`.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias `[control_rows]`.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights `[head_dim]`.
    pub norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the mixer `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MoE boundary `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen36GdnBindings<'a> {
    /// Binds one exact Qwen3.6 mixed-FP8/BF16 GDN source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(
            layer,
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::FULL_ATTENTION_INTERVAL,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::GDN_QKV_ROWS,
            Qwen36Moe35B::GDN_VALUE_ROWS,
            Qwen36Moe35B::GDN_CONTROL_ROWS,
            Qwen36Moe35B::LINEAR_CONV_KERNEL_DIM,
            Qwen36Moe35B::LINEAR_HEAD_DIM,
            |name| snapshot.tensor(name),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_from(
        layer: usize,
        layer_count: usize,
        full_attention_interval: usize,
        hidden: usize,
        qkv_rows: usize,
        value_rows: usize,
        control_rows: usize,
        convolution_width: usize,
        head_dim: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_gdn_layer_route(layer, layer_count, full_attention_interval)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let qkv = Qwen36Fp8LinearBindings::bind_from(
            &format!("{prefix}.in_proj_qkv"),
            qkv_rows,
            hidden,
            |name| tensor(name),
        )?;
        let z = Qwen36Fp8LinearBindings::bind_from(
            &format!("{prefix}.in_proj_z"),
            value_rows,
            hidden,
            |name| tensor(name),
        )?;

        require_same_rank_zero_f32(layer, "QKV/Z input_scale", &qkv.input_scale, &z.input_scale)?;

        Ok(Self {
            qkv,
            z,
            a_control: Bf16View::bind(
                tensor(&format!("{prefix}.in_proj_a.weight"))?,
                [control_rows as u64, hidden as u64],
            )?,
            b_control: Bf16View::bind(
                tensor(&format!("{prefix}.in_proj_b.weight"))?,
                [control_rows as u64, hidden as u64],
            )?,
            output: Qwen36Fp8LinearBindings::bind_from(
                &format!("{prefix}.out_proj"),
                hidden,
                value_rows,
                |name| tensor(name),
            )?,
            convolution_weight: Bf16View::bind(
                tensor(&format!("{prefix}.conv1d.weight"))?,
                [qkv_rows as u64, 1, convolution_width as u64],
            )?,
            a_log: Bf16View::bind(tensor(&format!("{prefix}.A_log"))?, [control_rows as u64])?,
            dt_bias: Bf16View::bind(tensor(&format!("{prefix}.dt_bias"))?, [control_rows as u64])?,
            norm: Bf16View::bind(tensor(&format!("{prefix}.norm.weight"))?, [head_dim as u64])?,
            input_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
                [hidden as u64],
            )?,
            post_attention_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
                [hidden as u64],
            )?,
            layer,
        })
    }
}

/// Complete scalar-scaled FP8/BF16 source family for one Qwen3.6 full-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36FullAttentionBindings<'a> {
    /// Query-plus-gate FP8 projection `[attention_query_rows, hidden]`.
    pub query_gate: Qwen36Fp8LinearBindings<'a>,
    /// Key FP8 projection `[attention_kv_rows, hidden]`.
    pub key: Qwen36Fp8LinearBindings<'a>,
    /// Value FP8 projection `[attention_kv_rows, hidden]`.
    pub value: Qwen36Fp8LinearBindings<'a>,
    /// Gated attention-output FP8 projection `[hidden, attention_output_columns]`.
    pub output: Qwen36Fp8LinearBindings<'a>,
    /// Per-head query RMSNorm weights `[head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights `[head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before attention `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MoE boundary `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen36FullAttentionBindings<'a> {
    /// Binds one exact Qwen3.6 scalar-scaled FP8 full-attention source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(
            layer,
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::FULL_ATTENTION_INTERVAL,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::ATTENTION_QUERY_ROWS,
            Qwen36Moe35B::ATTENTION_KV_ROWS,
            Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS,
            Qwen36Moe35B::HEAD_DIM,
            |name| snapshot.tensor(name),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_from(
        layer: usize,
        layer_count: usize,
        full_attention_interval: usize,
        hidden: usize,
        query_rows: usize,
        kv_rows: usize,
        output_columns: usize,
        head_dim: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_full_attention_layer(layer, layer_count, full_attention_interval)?;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let query_gate = Qwen36Fp8LinearBindings::bind_from(
            &format!("{prefix}.q_proj"),
            query_rows,
            hidden,
            |name| tensor(name),
        )?;
        let key = Qwen36Fp8LinearBindings::bind_from(
            &format!("{prefix}.k_proj"),
            kv_rows,
            hidden,
            |name| tensor(name),
        )?;
        let value = Qwen36Fp8LinearBindings::bind_from(
            &format!("{prefix}.v_proj"),
            kv_rows,
            hidden,
            |name| tensor(name),
        )?;

        require_same_rank_zero_f32(
            layer,
            "Q/K input_scale",
            &query_gate.input_scale,
            &key.input_scale,
        )?;
        require_same_rank_zero_f32(
            layer,
            "Q/V input_scale",
            &query_gate.input_scale,
            &value.input_scale,
        )?;

        Ok(Self {
            query_gate,
            key,
            value,
            output: Qwen36Fp8LinearBindings::bind_from(
                &format!("{prefix}.o_proj"),
                hidden,
                output_columns,
                |name| tensor(name),
            )?,
            query_norm: Bf16View::bind(
                tensor(&format!("{prefix}.q_norm.weight"))?,
                [head_dim as u64],
            )?,
            key_norm: Bf16View::bind(
                tensor(&format!("{prefix}.k_norm.weight"))?,
                [head_dim as u64],
            )?,
            input_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
                [hidden as u64],
            )?,
            post_attention_norm: Bf16View::bind(
                tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
                [hidden as u64],
            )?,
            layer,
        })
    }
}

/// Exact source planes for the Qwen3.6 text endpoints.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36TextEndpointBindings<'a> {
    /// BF16 token embedding matrix `[vocab, hidden]`.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 final RMSNorm weights `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
    /// ModelOpt NVFP4 language-model head `[vocab, hidden]`.
    pub lm_head: ModelOptNvfp4LinearBindings<'a>,
}

impl<'a> Qwen36TextEndpointBindings<'a> {
    /// Binds the exact Qwen3.6 embedding, final norm, and NVFP4 LM head.
    pub fn bind(snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>) -> CheckpointResult<Self> {
        Self::bind_from(Qwen36Moe35B::VOCAB, Qwen36Moe35B::HIDDEN, |name| {
            snapshot.tensor(name)
        })
    }

    /// Binds only the mmap-backed embedding used during token staging.
    pub fn bind_embedding(
        snapshot: &'a CheckpointSnapshot<Qwen36Moe35B>,
    ) -> CheckpointResult<Bf16View<'a, 2>> {
        Bf16View::bind(
            snapshot.tensor(EMBEDDING)?,
            [Qwen36Moe35B::VOCAB as u64, Qwen36Moe35B::HIDDEN as u64],
        )
    }

    fn bind_from(
        vocab: usize,
        hidden: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        Ok(Self {
            embedding: Bf16View::bind(tensor(EMBEDDING)?, [vocab as u64, hidden as u64])?,
            final_norm: Bf16View::bind(tensor(FINAL_NORM)?, [hidden as u64])?,
            lm_head: ModelOptNvfp4LinearBindings::bind_from(
                "lm_head",
                vocab,
                hidden,
                Qwen36Moe35B::LAYERS,
                |name| tensor(name),
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::inventory::CheckpointSnapshot;
    use crate::common::naming::{EMBEDDING, FINAL_NORM, LM_HEAD, LM_HEAD_SCALE};
    use crate::common::routes::{E2M1_VALUES_PER_BYTE, NVFP4_GROUP_SIZE};
    use crate::common::test_support::sources::{
        append_bf16_tensor, fixture_path, write_safetensors_payload,
    };
    use crate::{Arch, CheckpointErrorCode, Qwen36Moe35B, SafeTensorFile};
    use serde_json::{Value, json};
    use std::fs;
    use std::path::Path;

    fn qwen36_endpoint_fixture() -> (Value, Vec<u8>) {
        const VOCAB: usize = 128;
        const HIDDEN: usize = 64;

        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        append_bf16_tensor(&mut header, &mut payload, EMBEDDING, vec![VOCAB, HIDDEN]);
        append_bf16_tensor(&mut header, &mut payload, FINAL_NORM, vec![HIDDEN]);
        append_rank_zero_f32(&mut header, &mut payload, "lm_head.input_scale", 0.25);
        append_raw_tensor(
            &mut header,
            &mut payload,
            LM_HEAD,
            "U8",
            vec![VOCAB, HIDDEN / E2M1_VALUES_PER_BYTE],
            0x21,
        );
        append_raw_tensor(
            &mut header,
            &mut payload,
            LM_HEAD_SCALE,
            "F8_E4M3",
            vec![VOCAB, HIDDEN / NVFP4_GROUP_SIZE],
            0x38,
        );
        append_rank_zero_f32(&mut header, &mut payload, "lm_head.weight_scale_2", 0.125);

        (Value::Object(header), payload)
    }

    fn qwen36_moe_fixture(
        layer: usize,
        hidden: usize,
        intermediate: usize,
        experts: usize,
    ) -> (Value, Vec<u8>) {
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!("{mlp_prefix}.gate.weight"),
            vec![experts, hidden],
        );
        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!("{mlp_prefix}.shared_expert_gate.weight"),
            vec![1, hidden],
        );

        for expert in 0..experts {
            append_qwen36_expert(
                &mut header,
                &mut payload,
                &format!("{mlp_prefix}.experts.{expert}"),
                hidden,
                intermediate,
                u8::try_from(expert + 1).unwrap(),
            );
        }
        append_qwen36_expert(
            &mut header,
            &mut payload,
            &format!("{mlp_prefix}.shared_expert"),
            hidden,
            intermediate,
            0x40,
        );
        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!("{layer_prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            ),
            vec![hidden],
        );

        (Value::Object(header), payload)
    }

    fn qwen36_gdn_fixture(layer: usize) -> (Value, Vec<u8>) {
        const HIDDEN: usize = 32;
        const QKV_ROWS: usize = 16;
        const VALUE_ROWS: usize = 8;
        const CONTROL_ROWS: usize = 4;
        const HEAD_DIM: usize = 4;
        const CONVOLUTION_WIDTH: usize = 4;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        append_qwen36_fp8_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.in_proj_qkv"),
            QKV_ROWS,
            HIDDEN,
            0.25,
            0.125,
            0x10,
        );
        append_qwen36_fp8_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.in_proj_z"),
            VALUE_ROWS,
            HIDDEN,
            0.25,
            0.0625,
            0x20,
        );
        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!("{prefix}.in_proj_a.weight"),
            vec![CONTROL_ROWS, HIDDEN],
        );
        append_bf16_tensor(
            &mut header,
            &mut payload,
            format!("{prefix}.in_proj_b.weight"),
            vec![CONTROL_ROWS, HIDDEN],
        );
        append_qwen36_fp8_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.out_proj"),
            HIDDEN,
            VALUE_ROWS,
            0.5,
            0.03125,
            0x30,
        );
        for (name, shape) in [
            (
                format!("{prefix}.conv1d.weight"),
                vec![QKV_ROWS, 1, CONVOLUTION_WIDTH],
            ),
            (format!("{prefix}.A_log"), vec![CONTROL_ROWS]),
            (format!("{prefix}.dt_bias"), vec![CONTROL_ROWS]),
            (format!("{prefix}.norm.weight"), vec![HEAD_DIM]),
            (
                format!("{layer_prefix}.input_layernorm.weight"),
                vec![HIDDEN],
            ),
            (
                format!("{layer_prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    fn qwen36_attention_fixture(layer: usize) -> (Value, Vec<u8>) {
        const HIDDEN: usize = 32;
        const QUERY_ROWS: usize = 16;
        const KV_ROWS: usize = 4;
        const OUTPUT_COLUMNS: usize = 8;
        const HEAD_DIM: usize = 4;

        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for (projection, rows, columns, weight_scale, marker) in [
            ("q_proj", QUERY_ROWS, HIDDEN, 0.125, 0x10),
            ("k_proj", KV_ROWS, HIDDEN, 0.0625, 0x20),
            ("v_proj", KV_ROWS, HIDDEN, 0.03125, 0x30),
        ] {
            append_qwen36_fp8_linear(
                &mut header,
                &mut payload,
                &format!("{prefix}.{projection}"),
                rows,
                columns,
                0.25,
                weight_scale,
                marker,
            );
        }
        append_qwen36_fp8_linear(
            &mut header,
            &mut payload,
            &format!("{prefix}.o_proj"),
            HIDDEN,
            OUTPUT_COLUMNS,
            0.5,
            0.015625,
            0x40,
        );
        for (name, shape) in [
            (format!("{prefix}.q_norm.weight"), vec![HEAD_DIM]),
            (format!("{prefix}.k_norm.weight"), vec![HEAD_DIM]),
            (
                format!("{layer_prefix}.input_layernorm.weight"),
                vec![HIDDEN],
            ),
            (
                format!("{layer_prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    #[allow(clippy::too_many_arguments)]
    fn append_qwen36_fp8_linear(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        prefix: &str,
        rows: usize,
        columns: usize,
        input_scale: f32,
        weight_scale: f32,
        marker: u8,
    ) {
        append_rank_zero_f32(
            header,
            payload,
            format!("{prefix}.input_scale"),
            input_scale,
        );
        append_raw_tensor(
            header,
            payload,
            format!("{prefix}.weight"),
            "F8_E4M3",
            vec![rows, columns],
            marker,
        );
        append_rank_zero_f32(
            header,
            payload,
            format!("{prefix}.weight_scale"),
            weight_scale,
        );
    }

    fn append_qwen36_expert(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
        marker: u8,
    ) {
        for (projection, rows, columns, input_scale, weight_scale_2) in [
            ("gate_proj", intermediate, hidden, 0.25, 0.125),
            ("up_proj", intermediate, hidden, 0.25, 0.125),
            ("down_proj", hidden, intermediate, 0.5, 0.0625),
        ] {
            let projection = format!("{prefix}.{projection}");

            append_rank_zero_f32(
                header,
                payload,
                format!("{projection}.input_scale"),
                input_scale,
            );
            append_raw_tensor(
                header,
                payload,
                format!("{projection}.weight"),
                "U8",
                vec![rows, columns / E2M1_VALUES_PER_BYTE],
                marker,
            );
            append_raw_tensor(
                header,
                payload,
                format!("{projection}.weight_scale"),
                "F8_E4M3",
                vec![rows, columns / NVFP4_GROUP_SIZE],
                0x38,
            );
            append_rank_zero_f32(
                header,
                payload,
                format!("{projection}.weight_scale_2"),
                weight_scale_2,
            );
        }
    }

    fn append_rank_zero_f32(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        name: impl Into<String>,
        value: f32,
    ) {
        let begin = payload.len();
        payload.extend_from_slice(&value.to_le_bytes());
        header.insert(
            name.into(),
            json!({
                "dtype": "F32",
                "shape": [],
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    fn append_raw_tensor(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        name: impl Into<String>,
        dtype: &str,
        shape: Vec<usize>,
        value: u8,
    ) {
        let begin = payload.len();
        payload.resize(begin + shape.iter().product::<usize>(), value);
        header.insert(
            name.into(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    fn tensor_offset(header: &Value, name: &str) -> usize {
        header[name]["data_offsets"][0].as_u64().unwrap() as usize
    }

    fn qwen36_mtp_fixture() -> (Value, Vec<u8>) {
        const HIDDEN: usize = 8;
        const INTERMEDIATE: usize = 4;
        const EXPERTS: usize = 3;
        const QUERY_ROWS: usize = 6;
        const KV_ROWS: usize = 2;
        const OUTPUT_COLUMNS: usize = 3;
        const HEAD_DIM: usize = 1;

        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for (name, shape) in [
            ("mtp.fc.weight", vec![HIDDEN, 2 * HIDDEN]),
            ("mtp.norm.weight", vec![HIDDEN]),
            ("mtp.pre_fc_norm_embedding.weight", vec![HIDDEN]),
            ("mtp.pre_fc_norm_hidden.weight", vec![HIDDEN]),
            ("mtp.layers.0.input_layernorm.weight", vec![HIDDEN]),
            (
                "mtp.layers.0.mlp.experts.down_proj",
                vec![EXPERTS, HIDDEN, INTERMEDIATE],
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                vec![EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            ),
            ("mtp.layers.0.mlp.gate.weight", vec![EXPERTS, HIDDEN]),
            (
                "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
            ),
            (
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                vec![INTERMEDIATE, HIDDEN],
            ),
            (
                "mtp.layers.0.mlp.shared_expert.up_proj.weight",
                vec![INTERMEDIATE, HIDDEN],
            ),
            (
                "mtp.layers.0.mlp.shared_expert_gate.weight",
                vec![1, HIDDEN],
            ),
            ("mtp.layers.0.post_attention_layernorm.weight", vec![HIDDEN]),
            ("mtp.layers.0.self_attn.k_norm.weight", vec![HEAD_DIM]),
            (
                "mtp.layers.0.self_attn.k_proj.weight",
                vec![KV_ROWS, HIDDEN],
            ),
            (
                "mtp.layers.0.self_attn.o_proj.weight",
                vec![HIDDEN, OUTPUT_COLUMNS],
            ),
            ("mtp.layers.0.self_attn.q_norm.weight", vec![HEAD_DIM]),
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                vec![QUERY_ROWS, HIDDEN],
            ),
            (
                "mtp.layers.0.self_attn.v_proj.weight",
                vec![KV_ROWS, HIDDEN],
            ),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    #[test]
    fn binds_exact_qwen36_text_endpoint_contract() {
        const VOCAB: usize = 128;
        const HIDDEN: usize = 64;

        let path = fixture_path("qwen36-endpoints");
        let (header, payload) = qwen36_endpoint_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen36TextEndpointBindings::bind_from(VOCAB, HIDDEN, |name| file.tensor(name)).unwrap();

        assert_eq!(bindings.embedding.shape(), &[VOCAB as u64, HIDDEN as u64]);
        assert_eq!(bindings.final_norm.shape(), &[HIDDEN as u64]);
        assert_eq!(bindings.lm_head.weight.shape(), &[VOCAB as u64, 32]);
        assert_eq!(bindings.lm_head.block_scale.shape(), &[VOCAB as u64, 4]);
        assert_eq!(bindings.lm_head.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.lm_head.weight_scale_2.value(0), Some(0.125));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_qwen36_moe_experts_in_numeric_order() {
        let path = fixture_path("qwen36-moe");
        let (header, payload) = qwen36_moe_fixture(0, 32, 16, 3);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen36MoeLayerBindings::bind_from(0, 2, 32, 16, 16, 3, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.router_weight.shape(), &[3, 32]);
        assert_eq!(bindings.shared_expert_gate_weight.shape(), &[1, 32]);
        assert_eq!(bindings.experts.len(), 3);
        assert_eq!(
            bindings.experts[2].gate.weight.name(),
            "model.language_model.layers.0.mlp.experts.2.gate_proj.weight"
        );
        assert_eq!(bindings.experts[2].gate.weight.bytes()[0], 3);
        assert_eq!(bindings.experts[2].up.weight.bytes()[0], 3);
        assert_eq!(bindings.experts[2].down.weight.bytes()[0], 3);
        assert_eq!(bindings.shared_expert.gate.weight.bytes()[0], 0x40);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.next_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 0);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn qwen36_next_norm_routes_cover_the_layer_boundary() {
        for (layer, layers, expected) in [
            (
                0,
                2,
                Ok("model.language_model.layers.1.input_layernorm.weight"),
            ),
            (1, 2, Ok("model.language_model.norm.weight")),
            (2, 2, Err("Qwen3.6 MoE source contract")),
        ] {
            let result = qwen36_next_norm_name(layer, layers);

            match expected {
                Ok(name) => assert_eq!(result.unwrap(), name),
                Err(message) => assert!(result.unwrap_err().to_string().contains(message)),
            }
        }
    }

    #[test]
    fn binds_exact_qwen36_mixed_gdn_source_family() {
        let path = fixture_path("qwen36-gdn");
        let (header, payload) = qwen36_gdn_fixture(0);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen36GdnBindings::bind_from(0, 2, 2, 32, 16, 8, 4, 4, 4, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.qkv.weight.shape(), &[16, 32]);
        assert_eq!(bindings.qkv.weight.codes()[0], 0x10);
        assert_eq!(bindings.qkv.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.qkv.weight_scale.value(0), Some(0.125));
        assert_eq!(bindings.z.weight.shape(), &[8, 32]);
        assert_eq!(bindings.z.weight.codes()[0], 0x20);
        assert_eq!(bindings.a_control.shape(), &[4, 32]);
        assert_eq!(bindings.b_control.shape(), &[4, 32]);
        assert_eq!(bindings.output.weight.shape(), &[32, 8]);
        assert_eq!(bindings.output.weight.codes()[0], 0x30);
        assert_eq!(bindings.convolution_weight.shape(), &[16, 1, 4]);
        assert_eq!(bindings.a_log.shape(), &[4]);
        assert_eq!(bindings.dt_bias.shape(), &[4]);
        assert_eq!(bindings.norm.shape(), &[4]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 0);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_qwen36_gdn_route_shape_and_scale_drift() {
        let error = Qwen36GdnBindings::bind_from(1, 2, 2, 32, 16, 8, 4, 4, 4, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("GDN source contract"));

        let path = fixture_path("qwen36-gdn-shape");
        let (mut header, payload) = qwen36_gdn_fixture(0);
        header["model.language_model.layers.0.linear_attn.in_proj_a.weight"]["shape"] =
            json!([2, 64]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error =
            Qwen36GdnBindings::bind_from(0, 2, 2, 32, 16, 8, 4, 4, 4, |name| file.tensor(name))
                .unwrap_err();

        assert!(error.to_string().contains("in_proj_a.weight"));
        fs::remove_file(path).unwrap();

        let path = fixture_path("qwen36-gdn-scale");
        let (header, mut payload) = qwen36_gdn_fixture(0);
        let name = "model.language_model.layers.0.linear_attn.in_proj_z.input_scale";
        let offset = tensor_offset(&header, name);
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error =
            Qwen36GdnBindings::bind_from(0, 2, 2, 32, 16, 8, 4, 4, 4, |name| file.tensor(name))
                .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("QKV/Z input_scale values differ")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the exact pinned snapshot"]
    fn binds_all_real_qwen36_gdn_layers() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT must name the exact pinned snapshot");
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(Path::new(&root)).unwrap();
        let mut bound = 0;

        for layer in 0..Qwen36Moe35B::LAYERS {
            if (layer + 1).is_multiple_of(Qwen36Moe35B::FULL_ATTENTION_INTERVAL) {
                continue;
            }

            let bindings = Qwen36GdnBindings::bind(&snapshot, layer).unwrap();

            assert_eq!(bindings.qkv.weight.shape(), &[8_192, 2_048]);
            assert_eq!(bindings.z.weight.shape(), &[4_096, 2_048]);
            assert_eq!(bindings.a_control.shape(), &[32, 2_048]);
            assert_eq!(bindings.b_control.shape(), &[32, 2_048]);
            assert_eq!(bindings.output.weight.shape(), &[2_048, 4_096]);
            assert_eq!(bindings.convolution_weight.shape(), &[8_192, 1, 4]);
            assert_eq!(bindings.norm.shape(), &[128]);
            assert_eq!(bindings.input_norm.shape(), &[2_048]);
            assert_eq!(bindings.post_attention_norm.shape(), &[2_048]);
            assert_eq!(bindings.layer, layer);
            bound += 1;
        }

        assert_eq!(bound, 30);
    }

    #[test]
    fn binds_exact_qwen36_full_attention_source_family() {
        let path = fixture_path("qwen36-attention");
        let (header, payload) = qwen36_attention_fixture(1);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = Qwen36FullAttentionBindings::bind_from(1, 2, 2, 32, 16, 4, 8, 4, |name| {
            file.tensor(name)
        })
        .unwrap();

        assert_eq!(bindings.query_gate.weight.shape(), &[16, 32]);
        assert_eq!(bindings.query_gate.weight.codes()[0], 0x10);
        assert_eq!(bindings.query_gate.input_scale.value(0), Some(0.25));
        assert_eq!(bindings.query_gate.weight_scale.value(0), Some(0.125));
        assert_eq!(bindings.key.weight.shape(), &[4, 32]);
        assert_eq!(bindings.key.weight.codes()[0], 0x20);
        assert_eq!(bindings.value.weight.shape(), &[4, 32]);
        assert_eq!(bindings.value.weight.codes()[0], 0x30);
        assert_eq!(bindings.output.weight.shape(), &[32, 8]);
        assert_eq!(bindings.output.weight.codes()[0], 0x40);
        assert_eq!(bindings.output.input_scale.value(0), Some(0.5));
        assert_eq!(bindings.query_norm.shape(), &[4]);
        assert_eq!(bindings.key_norm.shape(), &[4]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 1);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_qwen36_full_attention_route_shape_and_scale_drift() {
        let error = Qwen36FullAttentionBindings::bind_from(0, 2, 2, 32, 16, 4, 8, 4, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("full-attention source contract"));

        let path = fixture_path("qwen36-attention-shape");
        let (mut header, payload) = qwen36_attention_fixture(1);
        header["model.language_model.layers.1.self_attn.q_norm.weight"]["shape"] = json!([2, 2]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = Qwen36FullAttentionBindings::bind_from(1, 2, 2, 32, 16, 4, 8, 4, |name| {
            file.tensor(name)
        })
        .unwrap_err();

        assert!(error.to_string().contains("q_norm.weight"));
        fs::remove_file(path).unwrap();

        let path = fixture_path("qwen36-attention-scale");
        let (header, mut payload) = qwen36_attention_fixture(1);
        let name = "model.language_model.layers.1.self_attn.k_proj.input_scale";
        let offset = tensor_offset(&header, name);
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = Qwen36FullAttentionBindings::bind_from(1, 2, 2, 32, 16, 4, 8, 4, |name| {
            file.tensor(name)
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("Q/K input_scale values differ"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the exact pinned snapshot"]
    fn binds_all_real_qwen36_full_attention_layers() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT must name the exact pinned snapshot");
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(Path::new(&root)).unwrap();
        let mut bound = 0;

        for layer in 0..Qwen36Moe35B::LAYERS {
            if !(layer + 1).is_multiple_of(Qwen36Moe35B::FULL_ATTENTION_INTERVAL) {
                continue;
            }

            let bindings = Qwen36FullAttentionBindings::bind(&snapshot, layer).unwrap();

            assert_eq!(bindings.query_gate.weight.shape(), &[8_192, 2_048]);
            assert_eq!(bindings.key.weight.shape(), &[512, 2_048]);
            assert_eq!(bindings.value.weight.shape(), &[512, 2_048]);
            assert_eq!(bindings.output.weight.shape(), &[2_048, 4_096]);
            assert_eq!(bindings.query_norm.shape(), &[256]);
            assert_eq!(bindings.key_norm.shape(), &[256]);
            assert_eq!(bindings.input_norm.shape(), &[2_048]);
            assert_eq!(bindings.post_attention_norm.shape(), &[2_048]);
            assert_eq!(bindings.layer, layer);
            bound += 1;
        }

        assert_eq!(bound, 10);
    }

    #[test]
    fn rejects_qwen36_moe_route_shape_and_scale_drift() {
        let error = Qwen36MoeLayerBindings::bind_from(2, 2, 32, 16, 16, 3, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("Qwen3.6 MoE source contract"));

        let path = fixture_path("qwen36-moe-shape");
        let (mut header, payload) = qwen36_moe_fixture(0, 32, 16, 3);
        header["model.language_model.layers.0.mlp.experts.1.gate_proj.weight"]["shape"] =
            json!([8, 32]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error =
            Qwen36MoeLayerBindings::bind_from(0, 2, 32, 16, 16, 3, |name| file.tensor(name))
                .unwrap_err();

        assert!(error.to_string().contains("experts.1.gate_proj.weight"));
        fs::remove_file(path).unwrap();

        let path = fixture_path("qwen36-moe-scale");
        let (header, mut payload) = qwen36_moe_fixture(0, 32, 16, 3);
        let name = "model.language_model.layers.0.mlp.experts.1.up_proj.input_scale";
        let offset = tensor_offset(&header, name);
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error =
            Qwen36MoeLayerBindings::bind_from(0, 2, 32, 16, 16, 3, |name| file.tensor(name))
                .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("expert-1 gate/up input_scale values differ")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the exact pinned snapshot"]
    fn binds_real_qwen36_moe_layer() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT must name the exact pinned snapshot");
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(Path::new(&root)).unwrap();
        let bindings = Qwen36MoeLayerBindings::bind(&snapshot, 0).unwrap();

        assert_eq!(bindings.router_weight.shape(), &[256, 2_048]);
        assert_eq!(bindings.shared_expert_gate_weight.shape(), &[1, 2_048]);
        assert_eq!(bindings.experts.len(), 256);
        assert_eq!(bindings.experts[0].gate.weight.shape(), &[512, 1_024]);
        assert_eq!(bindings.experts[255].down.weight.shape(), &[2_048, 256]);
        assert_eq!(
            bindings.experts[2].gate.weight.name(),
            "model.language_model.layers.0.mlp.experts.2.gate_proj.weight"
        );
        assert_eq!(
            bindings.experts[10].gate.weight.name(),
            "model.language_model.layers.0.mlp.experts.10.gate_proj.weight"
        );
        assert_eq!(bindings.shared_expert.gate.weight.shape(), &[512, 1_024]);
    }

    #[test]
    fn binds_exact_qwen36_mtp_moe_source_contract() {
        let path = fixture_path("qwen36-mtp");
        let (header, payload) = qwen36_mtp_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen36MtpBindings::bind_from(8, 4, 3, 6, 2, 3, 1, |name| file.tensor(name)).unwrap();

        assert_eq!(bindings.input_projection.shape(), &[8, 16]);
        assert_eq!(bindings.embedding_norm.shape(), &[8]);
        assert_eq!(bindings.hidden_norm.shape(), &[8]);
        assert_eq!(bindings.query_gate_weight.shape(), &[6, 8]);
        assert_eq!(bindings.key_weight.shape(), &[2, 8]);
        assert_eq!(bindings.value_weight.shape(), &[2, 8]);
        assert_eq!(bindings.attention_output_weight.shape(), &[8, 3]);
        assert_eq!(bindings.query_norm.shape(), &[1]);
        assert_eq!(bindings.key_norm.shape(), &[1]);
        assert_eq!(bindings.input_norm.shape(), &[8]);
        assert_eq!(bindings.router_weight.shape(), &[3, 8]);
        assert_eq!(bindings.routed_gate_up_weight.shape(), &[3, 8, 8]);
        assert_eq!(bindings.routed_down_weight.shape(), &[3, 8, 4]);
        assert_eq!(bindings.shared_expert_gate_weight.shape(), &[1, 8]);
        assert_eq!(bindings.shared_gate_weight.shape(), &[4, 8]);
        assert_eq!(bindings.shared_up_weight.shape(), &[4, 8]);
        assert_eq!(bindings.shared_down_weight.shape(), &[8, 4]);
        assert_eq!(bindings.post_attention_norm.shape(), &[8]);
        assert_eq!(bindings.final_norm.shape(), &[8]);
        assert_eq!(
            bindings.routed_down_weight.name(),
            "mtp.layers.0.mlp.experts.down_proj"
        );
        assert_eq!(
            bindings.routed_gate_up_weight.name(),
            "mtp.layers.0.mlp.experts.gate_up_proj"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_qwen36_mtp_routed_expert_shape_mismatch() {
        let path = fixture_path("qwen36-mtp-shape");
        let (mut header, payload) = qwen36_mtp_fixture();
        header["mtp.layers.0.mlp.experts.gate_up_proj"]["shape"] = json!([3, 16, 4]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = Qwen36MtpBindings::bind_from(8, 4, 3, 6, 2, 3, 1, |name| file.tensor(name))
            .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(
            error
                .to_string()
                .contains("mtp.layers.0.mlp.experts.gate_up_proj")
        );
        assert!(error.to_string().contains("expected [3, 8, 8]"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the exact pinned snapshot"]
    fn binds_real_qwen36_mtp_source_contract() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT must name the exact pinned snapshot");
        let snapshot = CheckpointSnapshot::<Qwen36Moe35B>::open(Path::new(&root)).unwrap();
        let bindings = Qwen36MtpBindings::bind(&snapshot).unwrap();

        assert_eq!(bindings.input_projection.shape(), &[2_048, 4_096]);
        assert_eq!(bindings.query_gate_weight.shape(), &[8_192, 2_048]);
        assert_eq!(bindings.key_weight.shape(), &[512, 2_048]);
        assert_eq!(bindings.value_weight.shape(), &[512, 2_048]);
        assert_eq!(bindings.attention_output_weight.shape(), &[2_048, 4_096]);
        assert_eq!(bindings.router_weight.shape(), &[256, 2_048]);
        assert_eq!(bindings.routed_gate_up_weight.shape(), &[256, 1_024, 2_048]);
        assert_eq!(bindings.routed_down_weight.shape(), &[256, 2_048, 512]);
        assert_eq!(bindings.shared_expert_gate_weight.shape(), &[1, 2_048]);
        assert_eq!(bindings.shared_gate_weight.shape(), &[512, 2_048]);
        assert_eq!(bindings.shared_up_weight.shape(), &[512, 2_048]);
        assert_eq!(bindings.shared_down_weight.shape(), &[2_048, 512]);
        assert_eq!(bindings.query_norm.shape(), &[256]);
        assert_eq!(bindings.key_norm.shape(), &[256]);

        let source_bytes = [
            bindings.input_projection.bytes().len(),
            bindings.embedding_norm.bytes().len(),
            bindings.hidden_norm.bytes().len(),
            bindings.query_gate_weight.bytes().len(),
            bindings.key_weight.bytes().len(),
            bindings.value_weight.bytes().len(),
            bindings.attention_output_weight.bytes().len(),
            bindings.query_norm.bytes().len(),
            bindings.key_norm.bytes().len(),
            bindings.input_norm.bytes().len(),
            bindings.router_weight.bytes().len(),
            bindings.routed_gate_up_weight.bytes().len(),
            bindings.routed_down_weight.bytes().len(),
            bindings.shared_expert_gate_weight.bytes().len(),
            bindings.shared_gate_weight.bytes().len(),
            bindings.shared_up_weight.bytes().len(),
            bindings.shared_down_weight.bytes().len(),
            bindings.post_attention_norm.bytes().len(),
            bindings.final_norm.bytes().len(),
        ]
        .into_iter()
        .sum::<usize>();

        assert_eq!(source_bytes, 1_689_281_536);
    }
}

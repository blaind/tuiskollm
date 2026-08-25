//! Qwen3.6-35B-A3B MoE lossless materialization into runtime-native host layouts.

use crate::common::modelopt_codec::{
    MaterializedModelOptNvfp4Linear, materialize_modelopt_linear, require_same_modelopt_scale,
};
use crate::common::mtp::MaterializedMtpQkv;
use crate::common::routes::{require_full_attention_layer, require_gdn_layer_route};
use crate::common::scale_swizzle::{
    gather_source_planes, host_shape, materialization_pool, materialization_workers,
};
use crate::qwen36::bindings::{
    Qwen36Fp8LinearBindings, Qwen36FullAttentionBindings, Qwen36GdnBindings,
    Qwen36MoeExpertBindings, Qwen36MoeLayerBindings, Qwen36MtpBindings, Qwen36TextEndpointBindings,
};
use crate::{Arch, Bf16View, CheckpointError, CheckpointResult, F32View, Qwen36Moe35B};
use rayon::prelude::*;
use std::mem::size_of;

impl Qwen36MtpBindings<'_> {
    /// Gathers the non-contiguous draft QKV planes without changing BF16 words.
    pub fn materialize_qkv(&self) -> CheckpointResult<MaterializedMtpQkv> {
        let [query_rows, columns] = host_shape(
            self.query_gate_weight.shape(),
            "Qwen3.6 MTP query/gate weights",
        )?;
        let [key_rows, key_columns] =
            host_shape(self.key_weight.shape(), "Qwen3.6 MTP key weights")?;
        let [value_rows, value_columns] =
            host_shape(self.value_weight.shape(), "Qwen3.6 MTP value weights")?;

        if key_rows != value_rows || columns != key_columns || columns != value_columns {
            return Err(CheckpointError::source_binding(
                "Qwen3.6 MTP QKV source planes have incompatible shapes",
            ));
        }

        let rows = query_rows
            .checked_add(key_rows)
            .and_then(|rows| rows.checked_add(value_rows))
            .ok_or_else(|| {
                CheckpointError::source_binding("Qwen3.6 MTP QKV row count overflows")
            })?;
        let weight_bf16 = gather_source_planes(
            [
                self.query_gate_weight.bytes(),
                self.key_weight.bytes(),
                self.value_weight.bytes(),
            ],
            "Qwen3.6 MTP QKV weights",
        )?;

        Ok(MaterializedMtpQkv {
            weight_bf16,
            rows,
            columns,
        })
    }
}

/// Runtime-native numeric-order Qwen3.6 expert planes.
#[derive(Debug)]
pub struct MaterializedQwen36MoeExperts {
    /// Gate then up packed E2M1 words for each expert.
    pub gate_up_weight_e2m1: Vec<u8>,
    /// Swizzled gate/up E4M3 scales in the same expert order.
    pub gate_up_scale_e4m3_swizzled: Vec<u8>,
    /// Down-projection packed E2M1 words for each expert.
    pub down_weight_e2m1: Vec<u8>,
    /// Swizzled down-projection E4M3 scales in the same expert order.
    pub down_scale_e4m3_swizzled: Vec<u8>,
    /// Exact gate/up activation scales by expert.
    pub gate_up_input_scales: Vec<f32>,
    /// Exact gate/up second-stage weight scales by expert.
    pub gate_up_weight_scales_2: Vec<f32>,
    /// Exact down-projection activation scales by expert.
    pub down_input_scales: Vec<f32>,
    /// Exact down-projection second-stage weight scales by expert.
    pub down_weight_scales_2: Vec<f32>,
    /// Reciprocal gate/up activation scales consumed by the kernels.
    pub gate_up_input_scale_divisors: Vec<f32>,
    /// Reciprocal gate/up weight scales consumed by the kernels.
    pub gate_up_weight_scale_divisors: Vec<f32>,
    /// Reciprocal down-projection activation scales consumed by the kernels.
    pub down_input_scale_divisors: Vec<f32>,
    /// Reciprocal down-projection weight scales consumed by the kernels.
    pub down_weight_scale_divisors: Vec<f32>,
    /// Number of experts in these planes.
    pub expert_count: usize,
    /// Intermediate width of each expert.
    pub intermediate: usize,
    /// Residual-stream width.
    pub hidden: usize,
    /// Decoder layer owning these planes.
    pub layer: usize,
}

impl MaterializedQwen36MoeExperts {
    /// Host bytes owned by this expert layout.
    pub fn owned_bytes(&self) -> usize {
        [
            self.gate_up_weight_e2m1.len(),
            self.gate_up_scale_e4m3_swizzled.len(),
            self.down_weight_e2m1.len(),
            self.down_scale_e4m3_swizzled.len(),
            self.gate_up_input_scales.len() * size_of::<f32>(),
            self.gate_up_weight_scales_2.len() * size_of::<f32>(),
            self.down_input_scales.len() * size_of::<f32>(),
            self.down_weight_scales_2.len() * size_of::<f32>(),
            self.gate_up_input_scale_divisors.len() * size_of::<f32>(),
            self.gate_up_weight_scale_divisors.len() * size_of::<f32>(),
            self.down_input_scale_divisors.len() * size_of::<f32>(),
            self.down_weight_scale_divisors.len() * size_of::<f32>(),
        ]
        .into_iter()
        .sum()
    }
}

/// Runtime-native Qwen3.6 MoE planes for one decoder layer.
#[derive(Debug)]
pub struct MaterializedQwen36MoeLayer<'a> {
    /// Router weights retained zero-copy from the source mapping.
    pub router_weight: Bf16View<'a, 2>,
    /// Shared-expert gate retained zero-copy from the source mapping.
    pub shared_expert_gate_weight: Bf16View<'a, 2>,
    /// Routed expert planes in numeric expert order.
    pub experts: MaterializedQwen36MoeExperts,
    /// Always-active shared-expert planes.
    pub shared_expert: MaterializedQwen36MoeExperts,
    /// Zero-centered RMSNorm weights before the MoE boundary.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl MaterializedQwen36MoeLayer<'_> {
    /// Host bytes owned by the two expert layouts.
    pub fn owned_bytes(&self) -> usize {
        self.experts.owned_bytes() + self.shared_expert.owned_bytes()
    }
}

/// One runtime-native scalar-scaled Qwen3.6 FP8 projection.
#[derive(Debug)]
pub struct MaterializedQwen36Fp8Linear<'a> {
    /// Source E4M3 weight codes retained without conversion.
    pub weight_e4m3: &'a [u8],
    /// Exact positive source activation scale.
    pub input_scale: f32,
    /// Exact positive source weight scale.
    pub weight_scale: f32,
    /// Output row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
    /// Decoder layer owning this projection.
    pub layer: usize,
}

/// Runtime-native mixed-FP8/BF16 planes for one Qwen3.6 GDN layer.
#[derive(Debug)]
pub struct MaterializedQwen36Gdn<'a> {
    /// Fused Q/K/V then Z E4M3 codes in projection-row order.
    pub input_weight_e4m3: Vec<u8>,
    /// Exact source activation scale shared by Q/K/V and Z.
    pub input_scale: f32,
    /// Exact source weight scales in Q/K/V then Z order.
    pub input_weight_scales: [f32; 2],
    /// Q/K/V rows preceding the Z rows.
    pub qkv_rows: usize,
    /// Fused Q/K/V/Z output row count.
    pub input_rows: usize,
    /// Logical residual-stream input width.
    pub input_columns: usize,
    /// BF16 A then B control weights in projection-row order.
    pub control_weight_bf16: Vec<u8>,
    /// Rows in one A or B control projection.
    pub control_rows_per_projection: usize,
    /// Logical residual-stream control input width.
    pub control_columns: usize,
    /// Recurrent-state output projection.
    pub output: MaterializedQwen36Fp8Linear<'a>,
    /// Width-four causal-convolution weights.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights.
    pub norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the mixer.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MoE boundary.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl MaterializedQwen36Gdn<'_> {
    /// Host bytes owned by the two fused source planes.
    pub fn owned_bytes(&self) -> usize {
        self.input_weight_e4m3.len() + self.control_weight_bf16.len()
    }
}

impl<'a> Qwen36GdnBindings<'a> {
    /// Fuses source rows without changing any represented FP8, BF16, or F32 value.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen36Gdn<'a>> {
        self.materialize_with_contract(
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::FULL_ATTENTION_INTERVAL,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::GDN_QKV_ROWS,
            Qwen36Moe35B::GDN_VALUE_ROWS,
            Qwen36Moe35B::GDN_CONTROL_ROWS,
            Qwen36Moe35B::LINEAR_CONV_KERNEL_DIM,
            Qwen36Moe35B::LINEAR_HEAD_DIM,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_with_contract(
        self,
        layer_count: usize,
        full_attention_interval: usize,
        hidden: usize,
        qkv_rows: usize,
        value_rows: usize,
        control_rows: usize,
        convolution_width: usize,
        head_dim: usize,
    ) -> CheckpointResult<MaterializedQwen36Gdn<'a>> {
        require_gdn_layer_route(self.layer, layer_count, full_attention_interval)?;

        let qkv = materialize_qwen36_fp8_linear(self.qkv, self.layer, "QKV")?;
        let z = materialize_qwen36_fp8_linear(self.z, self.layer, "Z")?;
        let output = materialize_qwen36_fp8_linear(self.output, self.layer, "output")?;
        let a_shape = host_shape(self.a_control.shape(), "Qwen3.6 A-control weights")?;
        let b_shape = host_shape(self.b_control.shape(), "Qwen3.6 B-control weights")?;

        if qkv.rows != qkv_rows
            || qkv.columns != hidden
            || z.rows != value_rows
            || z.columns != hidden
            || a_shape != [control_rows, hidden]
            || b_shape != [control_rows, hidden]
            || output.rows != hidden
            || output.columns != value_rows
            || self.convolution_weight.shape() != &[qkv_rows as u64, 1, convolution_width as u64]
            || self.a_log.shape() != &[control_rows as u64]
            || self.dt_bias.shape() != &[control_rows as u64]
            || self.norm.shape() != &[head_dim as u64]
            || self.input_norm.shape() != &[hidden as u64]
            || self.post_attention_norm.shape() != &[hidden as u64]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 GDN source geometry differs from its contract",
                self.layer
            )));
        }
        if self.qkv.input_scale.bits(0) != self.z.input_scale.bits(0) {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 QKV/Z input_scale values differ",
                self.layer
            )));
        }

        let input_rows = qkv_rows.checked_add(value_rows).ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 GDN input row count overflows",
                self.layer
            ))
        })?;
        let input_weight_e4m3 = gather_source_planes(
            [qkv.weight_e4m3, z.weight_e4m3],
            &format!("layer-{} Qwen3.6 QKV/Z weights", self.layer),
        )?;
        let control_weight_bf16 = gather_source_planes(
            [self.a_control.bytes(), self.b_control.bytes()],
            &format!("layer-{} Qwen3.6 A/B control weights", self.layer),
        )?;

        Ok(MaterializedQwen36Gdn {
            input_weight_e4m3,
            input_scale: qkv.input_scale,
            input_weight_scales: [qkv.weight_scale, z.weight_scale],
            qkv_rows,
            input_rows,
            input_columns: hidden,
            control_weight_bf16,
            control_rows_per_projection: control_rows,
            control_columns: hidden,
            output,
            convolution_weight: self.convolution_weight,
            a_log: self.a_log,
            dt_bias: self.dt_bias,
            norm: self.norm,
            input_norm: self.input_norm,
            post_attention_norm: self.post_attention_norm,
            layer: self.layer,
        })
    }
}

/// Runtime-native scalar-scaled FP8/BF16 planes for one Qwen3.6 full-attention layer.
#[derive(Debug)]
pub struct MaterializedQwen36FullAttention<'a> {
    /// Fused Q/gate, K, and V E4M3 codes in projection-row order.
    pub qkv_weight_e4m3: Vec<u8>,
    /// Exact source activation scale shared by Q, K, and V.
    pub qkv_input_scale: f32,
    /// Exact source weight scales in Q, K, and V order.
    pub qkv_weight_scales: [f32; 3],
    /// Query-plus-gate rows preceding the K and V rows.
    pub query_rows: usize,
    /// Rows in each K or V projection.
    pub kv_rows: usize,
    /// Fused Q/gate, K, and V output row count.
    pub qkv_rows: usize,
    /// Logical Q/K/V input width.
    pub qkv_columns: usize,
    /// Gated attention-output projection.
    pub output: MaterializedQwen36Fp8Linear<'a>,
    /// Per-head query RMSNorm weights.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights.
    pub key_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before attention.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MoE boundary.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl MaterializedQwen36FullAttention<'_> {
    /// Host bytes owned by the fused Q/K/V plane.
    pub fn owned_bytes(&self) -> usize {
        self.qkv_weight_e4m3.len()
    }
}

impl<'a> Qwen36FullAttentionBindings<'a> {
    /// Gathers Q/K/V rows without changing any represented FP8, BF16, or F32 value.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen36FullAttention<'a>> {
        self.materialize_with_contract(
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::FULL_ATTENTION_INTERVAL,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::ATTENTION_QUERY_ROWS,
            Qwen36Moe35B::ATTENTION_KV_ROWS,
            Qwen36Moe35B::ATTENTION_OUTPUT_COLUMNS,
            Qwen36Moe35B::HEAD_DIM,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_with_contract(
        self,
        layer_count: usize,
        full_attention_interval: usize,
        hidden: usize,
        query_rows: usize,
        kv_rows: usize,
        output_columns: usize,
        head_dim: usize,
    ) -> CheckpointResult<MaterializedQwen36FullAttention<'a>> {
        require_full_attention_layer(self.layer, layer_count, full_attention_interval)?;

        let query_gate = materialize_qwen36_fp8_linear(self.query_gate, self.layer, "query/gate")?;
        let key = materialize_qwen36_fp8_linear(self.key, self.layer, "key")?;
        let value = materialize_qwen36_fp8_linear(self.value, self.layer, "value")?;
        let output = materialize_qwen36_fp8_linear(self.output, self.layer, "output")?;

        if query_gate.rows != query_rows
            || query_gate.columns != hidden
            || key.rows != kv_rows
            || key.columns != hidden
            || value.rows != kv_rows
            || value.columns != hidden
            || output.rows != hidden
            || output.columns != output_columns
            || self.query_norm.shape() != &[head_dim as u64]
            || self.key_norm.shape() != &[head_dim as u64]
            || self.input_norm.shape() != &[hidden as u64]
            || self.post_attention_norm.shape() != &[hidden as u64]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 full-attention source geometry differs from its contract",
                self.layer
            )));
        }
        if self.query_gate.input_scale.bits(0) != self.key.input_scale.bits(0)
            || self.query_gate.input_scale.bits(0) != self.value.input_scale.bits(0)
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 Q/K/V input_scale values differ",
                self.layer
            )));
        }

        let qkv_rows = query_rows
            .checked_add(kv_rows)
            .and_then(|rows| rows.checked_add(kv_rows))
            .ok_or_else(|| {
                CheckpointError::source_binding(format!(
                    "layer-{} Qwen3.6 Q/K/V row count overflows",
                    self.layer
                ))
            })?;
        let qkv_weight_e4m3 = gather_source_planes(
            [query_gate.weight_e4m3, key.weight_e4m3, value.weight_e4m3],
            &format!("layer-{} Qwen3.6 Q/K/V weights", self.layer),
        )?;

        Ok(MaterializedQwen36FullAttention {
            qkv_weight_e4m3,
            qkv_input_scale: query_gate.input_scale,
            qkv_weight_scales: [
                query_gate.weight_scale,
                key.weight_scale,
                value.weight_scale,
            ],
            query_rows,
            kv_rows,
            qkv_rows,
            qkv_columns: hidden,
            output,
            query_norm: self.query_norm,
            key_norm: self.key_norm,
            input_norm: self.input_norm,
            post_attention_norm: self.post_attention_norm,
            layer: self.layer,
        })
    }
}

fn materialize_qwen36_fp8_linear<'a>(
    binding: Qwen36Fp8LinearBindings<'a>,
    layer: usize,
    role: &str,
) -> CheckpointResult<MaterializedQwen36Fp8Linear<'a>> {
    let [rows, columns] = host_shape(binding.weight.shape(), role)?;
    if rows != binding.rows || columns != binding.columns {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 {role} source geometry differs"
        )));
    }

    Ok(MaterializedQwen36Fp8Linear {
        weight_e4m3: binding.weight.codes(),
        input_scale: qwen36_fp8_scale(layer, role, "input", &binding.input_scale)?,
        weight_scale: qwen36_fp8_scale(layer, role, "weight", &binding.weight_scale)?,
        rows,
        columns,
        layer,
    })
}

fn qwen36_fp8_scale(
    layer: usize,
    role: &str,
    scale_role: &str,
    scale: &F32View<'_, 0>,
) -> CheckpointResult<f32> {
    let value = scale.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 {role} {scale_role} scale must be finite and positive, observed {value}"
        )));
    }

    Ok(value)
}

/// Runtime-native Qwen3.6 text endpoint sources.
#[derive(Debug)]
pub struct MaterializedQwen36TextEndpoint<'a> {
    /// Mmap-backed BF16 token embeddings.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 final RMSNorm weights.
    pub final_norm: Bf16View<'a, 1>,
    /// Packed E2M1 LM-head source words.
    pub lm_head_weight_e2m1: &'a [u8],
    /// Losslessly swizzled LM-head E4M3 block scales.
    pub lm_head_scale_e4m3_swizzled: Vec<u8>,
    /// Exact source activation scale retained for the endpoint contract.
    pub lm_head_input_scale: f32,
    /// Exact source second-stage weight scale consumed by the A16 route.
    pub lm_head_weight_scale_2: f32,
}

impl<'a> Qwen36TextEndpointBindings<'a> {
    /// Swizzles the LM-head scale plane without changing represented values.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen36TextEndpoint<'a>> {
        self.materialize_with_contract(Qwen36Moe35B::VOCAB, Qwen36Moe35B::HIDDEN)
    }

    fn materialize_with_contract(
        self,
        vocab: usize,
        hidden: usize,
    ) -> CheckpointResult<MaterializedQwen36TextEndpoint<'a>> {
        let lm_head = materialize_modelopt_linear(self.lm_head, Qwen36Moe35B::LAYERS, "LM head")?;
        if self.embedding.shape() != &[vocab as u64, hidden as u64]
            || self.final_norm.shape() != &[hidden as u64]
            || lm_head.rows != vocab
            || lm_head.columns != hidden
        {
            return Err(CheckpointError::source_binding(
                "Qwen3.6 text endpoint source geometry differs from its contract",
            ));
        }

        Ok(MaterializedQwen36TextEndpoint {
            embedding: self.embedding,
            final_norm: self.final_norm,
            lm_head_weight_e2m1: lm_head.weight_e2m1,
            lm_head_scale_e4m3_swizzled: lm_head.scale_e4m3_swizzled,
            lm_head_input_scale: lm_head.input_scale,
            lm_head_weight_scale_2: lm_head.weight_scale_2,
        })
    }
}

impl<'a> Qwen36MoeLayerBindings<'a> {
    /// Reorders experts numerically and converts their scale layouts without requantization.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen36MoeLayer<'a>> {
        self.materialize_with_contract(
            Qwen36Moe35B::LAYERS,
            Qwen36Moe35B::HIDDEN,
            Qwen36Moe35B::INTERMEDIATE,
            Qwen36Moe35B::SHARED_EXPERT_INTERMEDIATE,
            Qwen36Moe35B::NUM_EXPERTS,
        )
    }

    fn materialize_with_contract(
        self,
        layer_count: usize,
        hidden: usize,
        expert_intermediate: usize,
        shared_intermediate: usize,
        expert_count: usize,
    ) -> CheckpointResult<MaterializedQwen36MoeLayer<'a>> {
        if self.layer >= layer_count {
            return Err(CheckpointError::source_binding(format!(
                "layer {} does not use the admitted Qwen3.6 MoE source contract",
                self.layer
            )));
        }
        if self.experts.len() != expert_count {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 MoE source has {} routed experts, expected {expert_count}",
                self.layer,
                self.experts.len()
            )));
        }
        if self.router_weight.shape() != &[expert_count as u64, hidden as u64]
            || self.shared_expert_gate_weight.shape() != &[1, hidden as u64]
            || self.input_norm.shape() != &[hidden as u64]
            || self.next_norm.shape() != &[hidden as u64]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Qwen3.6 MoE BF16 source geometry differs from its contract",
                self.layer
            )));
        }

        let experts = materialize_qwen36_experts(
            self.experts,
            self.layer,
            hidden,
            expert_intermediate,
            "routed experts",
        )?;
        let shared_expert = materialize_qwen36_experts(
            vec![self.shared_expert],
            self.layer,
            hidden,
            shared_intermediate,
            "shared expert",
        )?;

        Ok(MaterializedQwen36MoeLayer {
            router_weight: self.router_weight,
            shared_expert_gate_weight: self.shared_expert_gate_weight,
            experts,
            shared_expert,
            input_norm: self.input_norm,
            next_norm: self.next_norm,
            layer: self.layer,
        })
    }
}

struct PreparedQwen36Expert<'a> {
    gate: MaterializedModelOptNvfp4Linear<'a>,
    up: MaterializedModelOptNvfp4Linear<'a>,
    down: MaterializedModelOptNvfp4Linear<'a>,
}

fn materialize_qwen36_experts<'a>(
    bindings: Vec<Qwen36MoeExpertBindings<'a>>,
    layer: usize,
    hidden: usize,
    intermediate: usize,
    role: &str,
) -> CheckpointResult<MaterializedQwen36MoeExperts> {
    if bindings.is_empty() {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 MoE {role} source is empty"
        )));
    }

    let expert_count = bindings.len();
    let prepare = |(expert, binding)| {
        prepare_qwen36_expert(binding, layer, &format!("{role} expert-{expert}"))
    };
    let prepared = if expert_count > 1 && materialization_workers() > 1 {
        materialization_pool(&format!("layer-{layer} Qwen3.6 MoE {role}"))?.install(|| {
            bindings
                .into_par_iter()
                .enumerate()
                .map(prepare)
                .collect::<CheckpointResult<Vec<_>>>()
        })
    } else {
        bindings
            .into_iter()
            .enumerate()
            .map(prepare)
            .collect::<CheckpointResult<Vec<_>>>()
    }?;

    flatten_qwen36_experts(prepared, layer, hidden, intermediate, role)
}

fn prepare_qwen36_expert<'a>(
    binding: Qwen36MoeExpertBindings<'a>,
    layer: usize,
    role: &str,
) -> CheckpointResult<PreparedQwen36Expert<'a>> {
    require_same_modelopt_scale(
        layer,
        &format!("{role} gate/up input_scale"),
        &binding.gate.input_scale,
        &binding.up.input_scale,
    )?;
    require_same_modelopt_scale(
        layer,
        &format!("{role} gate/up weight_scale_2"),
        &binding.gate.weight_scale_2,
        &binding.up.weight_scale_2,
    )?;

    Ok(PreparedQwen36Expert {
        gate: materialize_modelopt_linear(binding.gate, layer, &format!("{role} gate"))?,
        up: materialize_modelopt_linear(binding.up, layer, &format!("{role} up"))?,
        down: materialize_modelopt_linear(binding.down, layer, &format!("{role} down"))?,
    })
}

fn flatten_qwen36_experts(
    prepared: Vec<PreparedQwen36Expert<'_>>,
    layer: usize,
    hidden: usize,
    intermediate: usize,
    role: &str,
) -> CheckpointResult<MaterializedQwen36MoeExperts> {
    let expert_count = prepared.len();

    for (expert, planes) in prepared.iter().enumerate() {
        if planes.gate.rows != intermediate
            || planes.up.rows != intermediate
            || planes.gate.columns != hidden
            || planes.up.columns != hidden
            || planes.down.rows != hidden
            || planes.down.columns != intermediate
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} Qwen3.6 MoE {role} expert-{expert} geometry differs"
            )));
        }
    }

    let first = prepared
        .first()
        .expect("nonempty expert source was checked before materialization");
    let gate_up_weight_bytes = repeated_length(
        combined_length(
            first.gate.weight_e2m1.len(),
            first.up.weight_e2m1.len(),
            layer,
            &format!("{role} gate/up weights"),
        )?,
        expert_count,
        layer,
        &format!("{role} gate/up weights"),
    )?;
    let gate_up_scale_bytes = repeated_length(
        combined_length(
            first.gate.scale_e4m3_swizzled.len(),
            first.up.scale_e4m3_swizzled.len(),
            layer,
            &format!("{role} gate/up scales"),
        )?,
        expert_count,
        layer,
        &format!("{role} gate/up scales"),
    )?;
    let down_weight_bytes = repeated_length(
        first.down.weight_e2m1.len(),
        expert_count,
        layer,
        &format!("{role} down weights"),
    )?;
    let down_scale_bytes = repeated_length(
        first.down.scale_e4m3_swizzled.len(),
        expert_count,
        layer,
        &format!("{role} down scales"),
    )?;
    let mut gate_up_weight_e2m1 = reserved_bytes(
        gate_up_weight_bytes,
        layer,
        &format!("{role} gate/up weights"),
    )?;
    let mut gate_up_scale_e4m3_swizzled = reserved_bytes(
        gate_up_scale_bytes,
        layer,
        &format!("{role} gate/up scales"),
    )?;
    let mut down_weight_e2m1 =
        reserved_bytes(down_weight_bytes, layer, &format!("{role} down weights"))?;
    let mut down_scale_e4m3_swizzled =
        reserved_bytes(down_scale_bytes, layer, &format!("{role} down scales"))?;
    let mut gate_up_input_scales = reserved_scalars(expert_count, layer, role)?;
    let mut gate_up_weight_scales_2 = reserved_scalars(expert_count, layer, role)?;
    let mut down_input_scales = reserved_scalars(expert_count, layer, role)?;
    let mut down_weight_scales_2 = reserved_scalars(expert_count, layer, role)?;
    let mut gate_up_input_scale_divisors = reserved_scalars(expert_count, layer, role)?;
    let mut gate_up_weight_scale_divisors = reserved_scalars(expert_count, layer, role)?;
    let mut down_input_scale_divisors = reserved_scalars(expert_count, layer, role)?;
    let mut down_weight_scale_divisors = reserved_scalars(expert_count, layer, role)?;

    for planes in prepared {
        gate_up_weight_e2m1.extend_from_slice(planes.gate.weight_e2m1);
        gate_up_weight_e2m1.extend_from_slice(planes.up.weight_e2m1);
        gate_up_scale_e4m3_swizzled.extend_from_slice(&planes.gate.scale_e4m3_swizzled);
        gate_up_scale_e4m3_swizzled.extend_from_slice(&planes.up.scale_e4m3_swizzled);
        down_weight_e2m1.extend_from_slice(planes.down.weight_e2m1);
        down_scale_e4m3_swizzled.extend_from_slice(&planes.down.scale_e4m3_swizzled);
        gate_up_input_scales.push(planes.gate.input_scale);
        gate_up_weight_scales_2.push(planes.gate.weight_scale_2);
        down_input_scales.push(planes.down.input_scale);
        down_weight_scales_2.push(planes.down.weight_scale_2);
        gate_up_input_scale_divisors.push(planes.gate.input_scale_divisor);
        gate_up_weight_scale_divisors.push(planes.gate.weight_scale_divisor);
        down_input_scale_divisors.push(planes.down.input_scale_divisor);
        down_weight_scale_divisors.push(planes.down.weight_scale_divisor);
    }

    Ok(MaterializedQwen36MoeExperts {
        gate_up_weight_e2m1,
        gate_up_scale_e4m3_swizzled,
        down_weight_e2m1,
        down_scale_e4m3_swizzled,
        gate_up_input_scales,
        gate_up_weight_scales_2,
        down_input_scales,
        down_weight_scales_2,
        gate_up_input_scale_divisors,
        gate_up_weight_scale_divisors,
        down_input_scale_divisors,
        down_weight_scale_divisors,
        expert_count,
        intermediate,
        hidden,
        layer,
    })
}

fn repeated_length(
    bytes: usize,
    count: usize,
    layer: usize,
    role: &str,
) -> CheckpointResult<usize> {
    bytes.checked_mul(count).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 MoE {role} length overflows"
        ))
    })
}

fn combined_length(
    first: usize,
    second: usize,
    layer: usize,
    role: &str,
) -> CheckpointResult<usize> {
    first.checked_add(second).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 MoE {role} length overflows"
        ))
    })
}

fn reserved_bytes(bytes: usize, layer: usize, role: &str) -> CheckpointResult<Vec<u8>> {
    let mut values = Vec::new();
    values.try_reserve_exact(bytes).map_err(|_| {
        CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 MoE {role} cannot reserve {bytes} host bytes"
        ))
    })?;

    Ok(values)
}

fn reserved_scalars(count: usize, layer: usize, role: &str) -> CheckpointResult<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        CheckpointError::source_binding(format!(
            "layer-{layer} Qwen3.6 MoE {role} cannot reserve {count} scale values"
        ))
    })?;

    Ok(values)
}

#[cfg(test)]
mod tests {
    use crate::common::inventory::CheckpointSnapshot;
    use crate::common::modelopt_codec::ModelOptNvfp4LinearBindings;
    use crate::common::test_support::sources::{
        COLUMNS, GROUPS, PACKED_COLUMNS, ROWS, bf16_bytes, bf16_vector, bf16_view, bf16_volume,
        block_scale_oracle, f32_scalar_view, fp8_view, scale_codes, u8_view,
    };
    use crate::qwen36::bindings::{
        Qwen36Fp8LinearBindings, Qwen36FullAttentionBindings, Qwen36GdnBindings,
        Qwen36MoeExpertBindings, Qwen36MoeLayerBindings, Qwen36MtpBindings,
        Qwen36TextEndpointBindings,
    };
    use crate::{Arch, CheckpointErrorCode, Qwen36Moe35B};

    const QWEN36_WEIGHT_SHAPE: [u64; 2] = [ROWS as u64, PACKED_COLUMNS as u64];
    const QWEN36_SCALE_SHAPE: [u64; 2] = [ROWS as u64, GROUPS as u64];

    struct Qwen36ExpertFixture {
        gate_weight: Vec<u8>,
        gate_scale: Vec<u8>,
        gate_input_scale: [u8; 4],
        gate_weight_scale_2: [u8; 4],
        up_weight: Vec<u8>,
        up_scale: Vec<u8>,
        up_input_scale: [u8; 4],
        up_weight_scale_2: [u8; 4],
        down_weight: Vec<u8>,
        down_scale: Vec<u8>,
        down_input_scale: [u8; 4],
        down_weight_scale_2: [u8; 4],
    }

    impl Qwen36ExpertFixture {
        fn new(marker: u8) -> Self {
            Self {
                gate_weight: vec![marker; ROWS * PACKED_COLUMNS],
                gate_scale: scale_codes(usize::from(marker)),
                gate_input_scale: 0.25f32.to_le_bytes(),
                gate_weight_scale_2: 0.125f32.to_le_bytes(),
                up_weight: vec![marker.wrapping_add(0x10); ROWS * PACKED_COLUMNS],
                up_scale: scale_codes(usize::from(marker) + 11),
                up_input_scale: 0.25f32.to_le_bytes(),
                up_weight_scale_2: 0.125f32.to_le_bytes(),
                down_weight: vec![marker.wrapping_add(0x20); ROWS * PACKED_COLUMNS],
                down_scale: scale_codes(usize::from(marker) + 23),
                down_input_scale: 0.5f32.to_le_bytes(),
                down_weight_scale_2: 0.0625f32.to_le_bytes(),
            }
        }

        fn bindings(&self) -> Qwen36MoeExpertBindings<'_> {
            Qwen36MoeExpertBindings {
                gate: ModelOptNvfp4LinearBindings {
                    weight: u8_view("gate-weight", &QWEN36_WEIGHT_SHAPE, &self.gate_weight),
                    block_scale: fp8_view("gate-scale", &QWEN36_SCALE_SHAPE, &self.gate_scale),
                    input_scale: f32_scalar_view("gate-input-scale", &self.gate_input_scale),
                    weight_scale_2: f32_scalar_view(
                        "gate-weight-scale-2",
                        &self.gate_weight_scale_2,
                    ),
                    rows: ROWS,
                    columns: COLUMNS,
                },
                up: ModelOptNvfp4LinearBindings {
                    weight: u8_view("up-weight", &QWEN36_WEIGHT_SHAPE, &self.up_weight),
                    block_scale: fp8_view("up-scale", &QWEN36_SCALE_SHAPE, &self.up_scale),
                    input_scale: f32_scalar_view("up-input-scale", &self.up_input_scale),
                    weight_scale_2: f32_scalar_view("up-weight-scale-2", &self.up_weight_scale_2),
                    rows: ROWS,
                    columns: COLUMNS,
                },
                down: ModelOptNvfp4LinearBindings {
                    weight: u8_view("down-weight", &QWEN36_WEIGHT_SHAPE, &self.down_weight),
                    block_scale: fp8_view("down-scale", &QWEN36_SCALE_SHAPE, &self.down_scale),
                    input_scale: f32_scalar_view("down-input-scale", &self.down_input_scale),
                    weight_scale_2: f32_scalar_view(
                        "down-weight-scale-2",
                        &self.down_weight_scale_2,
                    ),
                    rows: ROWS,
                    columns: COLUMNS,
                },
            }
        }
    }

    #[test]
    fn qwen36_moe_materialization_preserves_numeric_expert_order() {
        let experts = [Qwen36ExpertFixture::new(1), Qwen36ExpertFixture::new(2)];
        let shared = Qwen36ExpertFixture::new(0x40);
        let router = bf16_bytes(&vec![0x3f80; experts.len() * COLUMNS]);
        let shared_gate = bf16_bytes(&vec![0x3f00; COLUMNS]);
        let input_norm = bf16_bytes(&vec![0x4000; COLUMNS]);
        let next_norm = bf16_bytes(&vec![0x4040; COLUMNS]);
        let router_shape = [experts.len() as u64, COLUMNS as u64];
        let bindings = Qwen36MoeLayerBindings {
            router_weight: bf16_view("router", &router_shape, &router),
            shared_expert_gate_weight: bf16_view("shared-gate", &[1, COLUMNS as u64], &shared_gate),
            experts: experts.iter().map(Qwen36ExpertFixture::bindings).collect(),
            shared_expert: shared.bindings(),
            input_norm: bf16_vector("input-norm", &[COLUMNS as u64], &input_norm),
            next_norm: bf16_vector("next-norm", &[COLUMNS as u64], &next_norm),
            layer: 0,
        };
        let materialized = bindings
            .materialize_with_contract(2, COLUMNS, ROWS, ROWS, experts.len())
            .unwrap();
        let mut expected_gate_up_weight = Vec::new();
        let mut expected_gate_up_scale = Vec::new();
        let mut expected_down_weight = Vec::new();
        let mut expected_down_scale = Vec::new();

        for expert in &experts {
            expected_gate_up_weight.extend_from_slice(&expert.gate_weight);
            expected_gate_up_weight.extend_from_slice(&expert.up_weight);
            expected_gate_up_scale.extend(block_scale_oracle(
                &[expert.gate_scale.as_slice(), expert.up_scale.as_slice()].concat(),
                2 * ROWS,
                GROUPS,
            ));
            expected_down_weight.extend_from_slice(&expert.down_weight);
            expected_down_scale.extend(block_scale_oracle(&expert.down_scale, ROWS, GROUPS));
        }

        assert_eq!(materialized.experts.expert_count, 2);
        assert_eq!(materialized.experts.intermediate, ROWS);
        assert_eq!(materialized.experts.hidden, COLUMNS);
        assert_eq!(
            materialized.experts.gate_up_weight_e2m1,
            expected_gate_up_weight
        );
        assert_eq!(
            materialized.experts.gate_up_scale_e4m3_swizzled,
            expected_gate_up_scale
        );
        assert_eq!(materialized.experts.down_weight_e2m1, expected_down_weight);
        assert_eq!(
            materialized.experts.down_scale_e4m3_swizzled,
            expected_down_scale
        );
        assert_eq!(materialized.experts.gate_up_input_scales, vec![0.25; 2]);
        assert_eq!(materialized.experts.gate_up_weight_scales_2, vec![0.125; 2]);
        assert_eq!(materialized.experts.down_input_scales, vec![0.5; 2]);
        assert_eq!(materialized.experts.down_weight_scales_2, vec![0.0625; 2]);
        assert_eq!(
            materialized.experts.gate_up_input_scale_divisors,
            vec![4.0; 2]
        );
        assert_eq!(materialized.experts.down_input_scale_divisors, vec![2.0; 2]);
        assert_eq!(materialized.shared_expert.expert_count, 1);
        assert_eq!(materialized.router_weight.bytes().as_ptr(), router.as_ptr());
        assert_eq!(materialized.experts.owned_bytes(), 55_360);
        assert_eq!(materialized.shared_expert.owned_bytes(), 27_680);
        assert_eq!(materialized.owned_bytes(), 83_040);
    }

    #[test]
    fn qwen36_moe_materialization_revalidates_route_count_and_scales() {
        let mut first = Qwen36ExpertFixture::new(1);
        let second = Qwen36ExpertFixture::new(2);
        let shared = Qwen36ExpertFixture::new(0x40);
        let router = bf16_bytes(&vec![0x3f80; 2 * COLUMNS]);
        let shared_gate = bf16_bytes(&vec![0x3f00; COLUMNS]);
        let norm = bf16_bytes(&vec![0x4000; COLUMNS]);

        first.up_input_scale = 0.5f32.to_le_bytes();
        let bindings = Qwen36MoeLayerBindings {
            router_weight: bf16_view("router", &[2, COLUMNS as u64], &router),
            shared_expert_gate_weight: bf16_view("shared-gate", &[1, COLUMNS as u64], &shared_gate),
            experts: vec![first.bindings(), second.bindings()],
            shared_expert: shared.bindings(),
            input_norm: bf16_vector("input-norm", &[COLUMNS as u64], &norm),
            next_norm: bf16_vector("next-norm", &[COLUMNS as u64], &norm),
            layer: 0,
        };
        let error = bindings
            .materialize_with_contract(2, COLUMNS, ROWS, ROWS, 2)
            .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("gate/up input_scale values differ")
        );

        let bindings = Qwen36MoeLayerBindings {
            router_weight: bf16_view("router", &[2, COLUMNS as u64], &router),
            shared_expert_gate_weight: bf16_view("shared-gate", &[1, COLUMNS as u64], &shared_gate),
            experts: vec![second.bindings()],
            shared_expert: shared.bindings(),
            input_norm: bf16_vector("input-norm", &[COLUMNS as u64], &norm),
            next_norm: bf16_vector("next-norm", &[COLUMNS as u64], &norm),
            layer: 0,
        };
        let error = bindings
            .materialize_with_contract(2, COLUMNS, ROWS, ROWS, 2)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("has 1 routed experts, expected 2")
        );
    }

    fn assert_qwen36_expert_is_lossless(
        source: Qwen36MoeExpertBindings<'_>,
        materialized: &super::MaterializedQwen36MoeExperts,
        expert: usize,
    ) {
        const GATE_UP_WEIGHT_BYTES: usize = 2 * 512 * 1_024;
        const GATE_UP_SCALE_BYTES: usize = 2 * 512 * 128;
        const DOWN_WEIGHT_BYTES: usize = 2_048 * 256;
        const DOWN_SCALE_BYTES: usize = 2_048 * 32;

        let gate_up_weight_begin = expert * GATE_UP_WEIGHT_BYTES;
        let gate_up_weight_end = gate_up_weight_begin + GATE_UP_WEIGHT_BYTES;
        let gate_up_scale_begin = expert * GATE_UP_SCALE_BYTES;
        let gate_up_scale_end = gate_up_scale_begin + GATE_UP_SCALE_BYTES;
        let down_weight_begin = expert * DOWN_WEIGHT_BYTES;
        let down_weight_end = down_weight_begin + DOWN_WEIGHT_BYTES;
        let down_scale_begin = expert * DOWN_SCALE_BYTES;
        let down_scale_end = down_scale_begin + DOWN_SCALE_BYTES;

        assert_eq!(
            &materialized.gate_up_weight_e2m1[gate_up_weight_begin..gate_up_weight_end],
            [source.gate.weight.bytes(), source.up.weight.bytes()].concat()
        );
        assert_eq!(
            &materialized.gate_up_scale_e4m3_swizzled[gate_up_scale_begin..gate_up_scale_end],
            block_scale_oracle(
                &[
                    source.gate.block_scale.codes(),
                    source.up.block_scale.codes(),
                ]
                .concat(),
                2 * Qwen36Moe35B::INTERMEDIATE,
                Qwen36Moe35B::HIDDEN / 16,
            )
        );
        assert_eq!(
            &materialized.down_weight_e2m1[down_weight_begin..down_weight_end],
            source.down.weight.bytes()
        );
        assert_eq!(
            &materialized.down_scale_e4m3_swizzled[down_scale_begin..down_scale_end],
            block_scale_oracle(
                source.down.block_scale.codes(),
                Qwen36Moe35B::HIDDEN,
                Qwen36Moe35B::INTERMEDIATE / 16,
            )
        );
        assert_eq!(
            materialized.gate_up_input_scales[expert].to_bits(),
            source.gate.input_scale.value(0).unwrap().to_bits()
        );
        assert_eq!(
            materialized.gate_up_weight_scales_2[expert].to_bits(),
            source.gate.weight_scale_2.value(0).unwrap().to_bits()
        );
        assert_eq!(
            materialized.down_input_scales[expert].to_bits(),
            source.down.input_scale.value(0).unwrap().to_bits()
        );
        assert_eq!(
            materialized.down_weight_scales_2[expert].to_bits(),
            source.down.weight_scale_2.value(0).unwrap().to_bits()
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_source_moe_layer0_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = Qwen36MoeLayerBindings::bind(&snapshot, 0).unwrap();
        let source = bindings.clone();
        let router = bindings.router_weight.bytes();
        let materialized = bindings.materialize().unwrap();

        assert_eq!(materialized.experts.expert_count, 256);
        assert_eq!(materialized.experts.hidden, 2_048);
        assert_eq!(materialized.experts.intermediate, 512);
        assert_eq!(materialized.experts.owned_bytes(), 452_993_024);
        assert_eq!(materialized.shared_expert.owned_bytes(), 1_769_504);
        assert_eq!(materialized.owned_bytes(), 454_762_528);
        assert_eq!(materialized.router_weight.bytes().as_ptr(), router.as_ptr());

        for expert in 0..source.experts.len() {
            assert_qwen36_expert_is_lossless(source.experts[expert], &materialized.experts, expert);
        }
        assert_qwen36_expert_is_lossless(source.shared_expert, &materialized.shared_expert, 0);
    }

    #[test]
    fn qwen36_endpoint_materialization_preserves_source_words_and_scales() {
        const VOCAB: usize = 128;
        const HIDDEN: usize = 64;
        const GROUPS: usize = HIDDEN / 16;

        let embedding_shape = [VOCAB as u64, HIDDEN as u64];
        let norm_shape = [HIDDEN as u64];
        let weight_shape = [VOCAB as u64, (HIDDEN / 2) as u64];
        let scale_shape = [VOCAB as u64, GROUPS as u64];
        let embedding = vec![0x20; VOCAB * HIDDEN * 2];
        let norm = vec![0x30; HIDDEN * 2];
        let weight = (0..VOCAB * HIDDEN / 2)
            .map(|index| index as u8)
            .collect::<Vec<_>>();
        let scales = (0..VOCAB * GROUPS)
            .map(|index| ((index * 37 + 17) % 0x7f) as u8)
            .collect::<Vec<_>>();
        let input_scale = 0.25f32.to_le_bytes();
        let weight_scale_2 = 0.125f32.to_le_bytes();
        let bindings = Qwen36TextEndpointBindings {
            embedding: bf16_view("embedding", &embedding_shape, &embedding),
            final_norm: bf16_vector("final-norm", &norm_shape, &norm),
            lm_head: ModelOptNvfp4LinearBindings {
                weight: u8_view("lm-head", &weight_shape, &weight),
                block_scale: fp8_view("lm-head-scale", &scale_shape, &scales),
                input_scale: f32_scalar_view("input-scale", &input_scale),
                weight_scale_2: f32_scalar_view("weight-scale", &weight_scale_2),
                rows: VOCAB,
                columns: HIDDEN,
            },
        };

        let error = Qwen36TextEndpointBindings {
            lm_head: ModelOptNvfp4LinearBindings {
                rows: VOCAB + 128,
                ..bindings.lm_head
            },
            ..bindings
        }
        .materialize_with_contract(VOCAB, HIDDEN)
        .unwrap_err();
        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("source geometry differs"));

        let materialized = bindings.materialize_with_contract(VOCAB, HIDDEN).unwrap();

        assert_eq!(materialized.lm_head_weight_e2m1, weight);
        assert_eq!(materialized.lm_head_weight_e2m1.as_ptr(), weight.as_ptr());
        assert_eq!(
            materialized.lm_head_scale_e4m3_swizzled,
            block_scale_oracle(&scales, VOCAB, GROUPS)
        );
        assert_eq!(
            materialized.lm_head_input_scale.to_bits(),
            0.25f32.to_bits()
        );
        assert_eq!(
            materialized.lm_head_weight_scale_2.to_bits(),
            0.125f32.to_bits()
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_source_text_endpoint_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = Qwen36TextEndpointBindings::bind(&snapshot).unwrap();
        let source_codes = bindings.lm_head.weight.bytes();
        let source_scales = bindings.lm_head.block_scale.codes();
        let source_input_scale = bindings.lm_head.input_scale.value(0).unwrap();
        let source_weight_scale = bindings.lm_head.weight_scale_2.value(0).unwrap();
        let materialized = bindings.materialize().unwrap();

        assert_eq!(materialized.embedding.shape(), &[248_320, 2_048]);
        assert_eq!(materialized.final_norm.shape(), &[2_048]);
        assert_eq!(
            materialized.lm_head_weight_e2m1.as_ptr(),
            source_codes.as_ptr()
        );
        assert_eq!(materialized.lm_head_weight_e2m1.len(), 254_279_680);
        assert_eq!(
            materialized.lm_head_scale_e4m3_swizzled,
            block_scale_oracle(source_scales, 248_320, 128)
        );
        assert_eq!(
            materialized.lm_head_input_scale.to_bits(),
            source_input_scale.to_bits()
        );
        assert_eq!(
            materialized.lm_head_weight_scale_2.to_bits(),
            source_weight_scale.to_bits()
        );
    }

    #[test]
    fn qwen36_gdn_materialization_preserves_mixed_source_planes() {
        let qkv_weight = vec![0x10; ROWS * COLUMNS];
        let z_weight = vec![0x20; ROWS * COLUMNS];
        let a_weight = bf16_bytes(&vec![0x3f80; 32 * COLUMNS]);
        let b_weight = bf16_bytes(&vec![0x4000; 32 * COLUMNS]);
        let output_weight = vec![0x30; ROWS * COLUMNS];
        let convolution = bf16_bytes(&vec![0x4040; ROWS * 4]);
        let a_log = bf16_bytes(&[0x4080; 32]);
        let dt_bias = bf16_bytes(&[0x40a0; 32]);
        let norm = bf16_bytes(&vec![0x40c0; COLUMNS]);
        let input_norm = bf16_bytes(&vec![0x40e0; COLUMNS]);
        let post_attention_norm = bf16_bytes(&vec![0x4100; COLUMNS]);
        let input_scale = 0.25f32.to_le_bytes();
        let qkv_weight_scale = 0.125f32.to_le_bytes();
        let z_weight_scale = 0.0625f32.to_le_bytes();
        let output_input_scale = 0.5f32.to_le_bytes();
        let output_weight_scale = 0.03125f32.to_le_bytes();
        let projection_shape = [ROWS as u64, COLUMNS as u64];
        let control_shape = [32, COLUMNS as u64];
        let bindings = Qwen36GdnBindings {
            qkv: Qwen36Fp8LinearBindings {
                weight: fp8_view("qkv-weight", &projection_shape, &qkv_weight),
                input_scale: f32_scalar_view("qkv-input-scale", &input_scale),
                weight_scale: f32_scalar_view("qkv-weight-scale", &qkv_weight_scale),
                rows: ROWS,
                columns: COLUMNS,
            },
            z: Qwen36Fp8LinearBindings {
                weight: fp8_view("z-weight", &projection_shape, &z_weight),
                input_scale: f32_scalar_view("z-input-scale", &input_scale),
                weight_scale: f32_scalar_view("z-weight-scale", &z_weight_scale),
                rows: ROWS,
                columns: COLUMNS,
            },
            a_control: bf16_view("a-control", &control_shape, &a_weight),
            b_control: bf16_view("b-control", &control_shape, &b_weight),
            output: Qwen36Fp8LinearBindings {
                weight: fp8_view("output-weight", &projection_shape, &output_weight),
                input_scale: f32_scalar_view("output-input-scale", &output_input_scale),
                weight_scale: f32_scalar_view("output-weight-scale", &output_weight_scale),
                rows: ROWS,
                columns: COLUMNS,
            },
            convolution_weight: bf16_volume("convolution", &[ROWS as u64, 1, 4], &convolution),
            a_log: bf16_vector("a-log", &[32], &a_log),
            dt_bias: bf16_vector("dt-bias", &[32], &dt_bias),
            norm: bf16_vector("norm", &[COLUMNS as u64], &norm),
            input_norm: bf16_vector("input-norm", &[COLUMNS as u64], &input_norm),
            post_attention_norm: bf16_vector(
                "post-attention-norm",
                &[COLUMNS as u64],
                &post_attention_norm,
            ),
            layer: 0,
        };
        let different_input_scale = 0.5f32.to_le_bytes();
        let scale_error = Qwen36GdnBindings {
            z: Qwen36Fp8LinearBindings {
                input_scale: f32_scalar_view("z-input-scale", &different_input_scale),
                ..bindings.z
            },
            ..bindings
        }
        .materialize_with_contract(2, 2, COLUMNS, ROWS, ROWS, 32, 4, COLUMNS)
        .unwrap_err();

        assert_eq!(scale_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            scale_error
                .to_string()
                .contains("QKV/Z input_scale values differ")
        );

        let materialized = bindings
            .materialize_with_contract(2, 2, COLUMNS, ROWS, ROWS, 32, 4, COLUMNS)
            .unwrap();

        assert_eq!(
            materialized.input_weight_e4m3,
            [qkv_weight.as_slice(), z_weight.as_slice()].concat()
        );
        assert_eq!(
            materialized.control_weight_bf16,
            [a_weight.as_slice(), b_weight.as_slice()].concat()
        );
        assert_eq!(materialized.input_scale.to_bits(), 0.25f32.to_bits());
        assert_eq!(
            materialized
                .input_weight_scales
                .map(|scale| scale.to_bits()),
            [0.125f32.to_bits(), 0.0625f32.to_bits()]
        );
        assert_eq!(materialized.qkv_rows, ROWS);
        assert_eq!(materialized.input_rows, 2 * ROWS);
        assert_eq!(materialized.input_columns, COLUMNS);
        assert_eq!(materialized.control_rows_per_projection, 32);
        assert_eq!(materialized.control_columns, COLUMNS);
        assert_eq!(
            materialized.output.weight_e4m3.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.output.input_scale, 0.5);
        assert_eq!(materialized.output.weight_scale, 0.03125);
        assert_eq!(materialized.convolution_weight.word(0), Some(0x4040));
        assert_eq!(materialized.a_log.word(0), Some(0x4080));
        assert_eq!(materialized.dt_bias.word(0), Some(0x40a0));
        assert_eq!(materialized.norm.word(0), Some(0x40c0));
        assert_eq!(materialized.input_norm.word(0), Some(0x40e0));
        assert_eq!(materialized.post_attention_norm.word(0), Some(0x4100));
        assert_eq!(
            materialized.owned_bytes(),
            2 * ROWS * COLUMNS + 4 * 32 * COLUMNS
        );
        assert_eq!(materialized.layer, 0);
    }

    #[test]
    fn qwen36_attention_materialization_preserves_source_planes() {
        const QUERY_ROWS: usize = 128;
        const KV_ROWS: usize = 32;
        const OUTPUT_COLUMNS: usize = 64;
        const HEAD_DIM: usize = 16;

        let query_weight = vec![0x10; QUERY_ROWS * COLUMNS];
        let key_weight = vec![0x20; KV_ROWS * COLUMNS];
        let value_weight = vec![0x30; KV_ROWS * COLUMNS];
        let output_weight = vec![0x40; COLUMNS * OUTPUT_COLUMNS];
        let query_norm = bf16_bytes(&[0x3f80; HEAD_DIM]);
        let key_norm = bf16_bytes(&[0x4000; HEAD_DIM]);
        let input_norm = bf16_bytes(&[0x4040; COLUMNS]);
        let post_attention_norm = bf16_bytes(&[0x4080; COLUMNS]);
        let input_scale = 0.25f32.to_le_bytes();
        let query_weight_scale = 0.125f32.to_le_bytes();
        let key_weight_scale = 0.0625f32.to_le_bytes();
        let value_weight_scale = 0.03125f32.to_le_bytes();
        let output_input_scale = 0.5f32.to_le_bytes();
        let output_weight_scale = 0.015625f32.to_le_bytes();
        let query_shape = [QUERY_ROWS as u64, COLUMNS as u64];
        let kv_shape = [KV_ROWS as u64, COLUMNS as u64];
        let output_shape = [COLUMNS as u64, OUTPUT_COLUMNS as u64];
        let bindings = Qwen36FullAttentionBindings {
            query_gate: Qwen36Fp8LinearBindings {
                weight: fp8_view("query-weight", &query_shape, &query_weight),
                input_scale: f32_scalar_view("query-input-scale", &input_scale),
                weight_scale: f32_scalar_view("query-weight-scale", &query_weight_scale),
                rows: QUERY_ROWS,
                columns: COLUMNS,
            },
            key: Qwen36Fp8LinearBindings {
                weight: fp8_view("key-weight", &kv_shape, &key_weight),
                input_scale: f32_scalar_view("key-input-scale", &input_scale),
                weight_scale: f32_scalar_view("key-weight-scale", &key_weight_scale),
                rows: KV_ROWS,
                columns: COLUMNS,
            },
            value: Qwen36Fp8LinearBindings {
                weight: fp8_view("value-weight", &kv_shape, &value_weight),
                input_scale: f32_scalar_view("value-input-scale", &input_scale),
                weight_scale: f32_scalar_view("value-weight-scale", &value_weight_scale),
                rows: KV_ROWS,
                columns: COLUMNS,
            },
            output: Qwen36Fp8LinearBindings {
                weight: fp8_view("output-weight", &output_shape, &output_weight),
                input_scale: f32_scalar_view("output-input-scale", &output_input_scale),
                weight_scale: f32_scalar_view("output-weight-scale", &output_weight_scale),
                rows: COLUMNS,
                columns: OUTPUT_COLUMNS,
            },
            query_norm: bf16_vector("query-norm", &[HEAD_DIM as u64], &query_norm),
            key_norm: bf16_vector("key-norm", &[HEAD_DIM as u64], &key_norm),
            input_norm: bf16_vector("input-norm", &[COLUMNS as u64], &input_norm),
            post_attention_norm: bf16_vector(
                "post-attention-norm",
                &[COLUMNS as u64],
                &post_attention_norm,
            ),
            layer: 1,
        };
        let different_input_scale = 0.5f32.to_le_bytes();
        let scale_error = Qwen36FullAttentionBindings {
            key: Qwen36Fp8LinearBindings {
                input_scale: f32_scalar_view("key-input-scale", &different_input_scale),
                ..bindings.key
            },
            ..bindings
        }
        .materialize_with_contract(2, 2, COLUMNS, QUERY_ROWS, KV_ROWS, OUTPUT_COLUMNS, HEAD_DIM)
        .unwrap_err();

        assert_eq!(scale_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            scale_error
                .to_string()
                .contains("Q/K/V input_scale values differ")
        );

        let materialized = bindings
            .materialize_with_contract(2, 2, COLUMNS, QUERY_ROWS, KV_ROWS, OUTPUT_COLUMNS, HEAD_DIM)
            .unwrap();

        assert_eq!(
            materialized.qkv_weight_e4m3,
            [
                query_weight.as_slice(),
                key_weight.as_slice(),
                value_weight.as_slice(),
            ]
            .concat()
        );
        assert_eq!(materialized.qkv_input_scale.to_bits(), 0.25f32.to_bits());
        assert_eq!(
            materialized.qkv_weight_scales.map(|scale| scale.to_bits()),
            [
                0.125f32.to_bits(),
                0.0625f32.to_bits(),
                0.03125f32.to_bits(),
            ]
        );
        assert_eq!(materialized.query_rows, QUERY_ROWS);
        assert_eq!(materialized.kv_rows, KV_ROWS);
        assert_eq!(materialized.qkv_rows, QUERY_ROWS + 2 * KV_ROWS);
        assert_eq!(materialized.qkv_columns, COLUMNS);
        assert_eq!(
            materialized.output.weight_e4m3.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.output.input_scale, 0.5);
        assert_eq!(materialized.output.weight_scale, 0.015625);
        assert_eq!(materialized.query_norm.word(0), Some(0x3f80));
        assert_eq!(materialized.key_norm.word(0), Some(0x4000));
        assert_eq!(materialized.input_norm.word(0), Some(0x4040));
        assert_eq!(materialized.post_attention_norm.word(0), Some(0x4080));
        assert_eq!(
            materialized.owned_bytes(),
            (QUERY_ROWS + 2 * KV_ROWS) * COLUMNS
        );
        assert_eq!(materialized.layer, 1);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_source_attention_layer3_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = Qwen36FullAttentionBindings::bind(&snapshot, 3).unwrap();
        let qkv_source = [
            bindings.query_gate.weight.codes(),
            bindings.key.weight.codes(),
            bindings.value.weight.codes(),
        ]
        .concat();
        let output_weight = bindings.output.weight.codes();
        let input_scale_bits = bindings.query_gate.input_scale.bits(0).unwrap();
        let weight_scale_bits = [
            bindings.query_gate.weight_scale.bits(0).unwrap(),
            bindings.key.weight_scale.bits(0).unwrap(),
            bindings.value.weight_scale.bits(0).unwrap(),
        ];
        let materialized = bindings.materialize().unwrap();

        assert_eq!(materialized.qkv_weight_e4m3, qkv_source);
        assert_eq!(materialized.qkv_input_scale.to_bits(), input_scale_bits);
        assert_eq!(
            materialized.qkv_weight_scales.map(|scale| scale.to_bits()),
            weight_scale_bits
        );
        assert_eq!(materialized.query_rows, Qwen36Moe35B::ATTENTION_QUERY_ROWS);
        assert_eq!(materialized.kv_rows, Qwen36Moe35B::ATTENTION_KV_ROWS);
        assert_eq!(materialized.qkv_rows, Qwen36Moe35B::ATTENTION_QKV_ROWS);
        assert_eq!(materialized.qkv_columns, Qwen36Moe35B::HIDDEN);
        assert_eq!(materialized.qkv_weight_e4m3.len(), 18_874_368);
        assert_eq!(
            materialized.output.weight_e4m3.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.output.weight_e4m3.len(), 8_388_608);
        assert_eq!(materialized.query_norm.shape(), &[256]);
        assert_eq!(materialized.key_norm.shape(), &[256]);
        assert_eq!(materialized.input_norm.shape(), &[2_048]);
        assert_eq!(materialized.post_attention_norm.shape(), &[2_048]);
        assert_eq!(materialized.owned_bytes(), 18_874_368);
        assert_eq!(materialized.layer, 3);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_mtp_qkv_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = Qwen36MtpBindings::bind(&snapshot).unwrap();
        let source = [
            bindings.query_gate_weight.bytes(),
            bindings.key_weight.bytes(),
            bindings.value_weight.bytes(),
        ]
        .concat();
        let materialized = bindings.materialize_qkv().unwrap();

        assert_eq!(materialized.weight_bf16, source);
        assert_eq!(materialized.rows, Qwen36Moe35B::ATTENTION_QKV_ROWS);
        assert_eq!(materialized.columns, Qwen36Moe35B::HIDDEN);
        assert_eq!(materialized.weight_bf16.len(), 37_748_736);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn qwen36_source_gdn_layer0_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = Qwen36GdnBindings::bind(&snapshot, 0).unwrap();
        let input_source = [bindings.qkv.weight.codes(), bindings.z.weight.codes()].concat();
        let control_source = [bindings.a_control.bytes(), bindings.b_control.bytes()].concat();
        let output_weight = bindings.output.weight.codes();
        let input_scale_bits = bindings.qkv.input_scale.bits(0).unwrap();
        let input_weight_scale_bits = [
            bindings.qkv.weight_scale.bits(0).unwrap(),
            bindings.z.weight_scale.bits(0).unwrap(),
        ];
        let materialized = bindings.materialize().unwrap();

        assert_eq!(materialized.input_weight_e4m3, input_source);
        assert_eq!(materialized.control_weight_bf16, control_source);
        assert_eq!(materialized.input_scale.to_bits(), input_scale_bits);
        assert_eq!(
            materialized
                .input_weight_scales
                .map(|scale| scale.to_bits()),
            input_weight_scale_bits
        );
        assert_eq!(materialized.qkv_rows, Qwen36Moe35B::GDN_QKV_ROWS);
        assert_eq!(materialized.input_rows, Qwen36Moe35B::GDN_INPUT_ROWS);
        assert_eq!(materialized.input_columns, Qwen36Moe35B::HIDDEN);
        assert_eq!(
            materialized.control_rows_per_projection,
            Qwen36Moe35B::GDN_CONTROL_ROWS
        );
        assert_eq!(materialized.control_columns, Qwen36Moe35B::HIDDEN);
        assert_eq!(
            materialized.output.weight_e4m3.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.owned_bytes(), 25_427_968);
        assert_eq!(materialized.layer, 0);
    }
}

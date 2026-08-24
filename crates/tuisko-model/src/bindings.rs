//! Shape- and dtype-checked bindings from checkpoint tensors to model roles.

use crate::{
    Arch, Bf16View, CheckpointContract, CheckpointError, CheckpointResult, CheckpointSnapshot,
    F32View, Fp8E4M3View, TensorView, U8View,
};

pub(crate) const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LM_HEAD_SCALE: &str = "lm_head.weight_scale";
const MTP_LAYER: usize = 0;

// These are source-codec facts, not architecture geometry.
/// Exclusive decoder-layer boundary for the checkpoint's packed NVFP4 MLP planes.
pub const NVFP4_MLP_LAYER_END: usize = 56;
const DENSE_FP8_MLP_LAYER_START: usize = NVFP4_MLP_LAYER_END;
const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;

/// Source-native fused gate/up planes for one dense-FP8 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct DenseFp8GateUpBindings<'a> {
    /// Gate rows followed by up rows as one E4M3 source span `[2 * intermediate, hidden]`.
    pub weight_e4m3: &'a [u8],
    /// Gate rows followed by up rows as one little-endian BF16 source span.
    pub scale_bf16: &'a [u8],
    /// Fused gate/up row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
    /// Decoder layer owning these planes.
    pub layer: usize,
}

impl<'a> DenseFp8GateUpBindings<'a> {
    /// Binds one admitted dense-FP8 gate/up source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(
            layer,
            |name| snapshot.tensor(name),
            |first, second, role| snapshot.adjacent_tensor_bytes(first, second, role),
        )
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
        mut adjacent: impl FnMut(&str, &str, &str) -> CheckpointResult<&'a [u8]>,
    ) -> CheckpointResult<Self> {
        require_dense_fp8_mlp_layer::<A>(layer)?;

        let intermediate = A::INTERMEDIATE as u64;
        let hidden = A::HIDDEN as u64;
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let gate_weight_name = format!("{prefix}.gate_proj.weight");
        let up_weight_name = format!("{prefix}.up_proj.weight");
        let gate_scale_name = format!("{prefix}.gate_proj.weight_scale");
        let up_scale_name = format!("{prefix}.up_proj.weight_scale");

        Fp8E4M3View::bind(tensor(&gate_weight_name)?, [intermediate, hidden])?;
        Fp8E4M3View::bind(tensor(&up_weight_name)?, [intermediate, hidden])?;
        let gate_scale = Bf16View::bind(tensor(&gate_scale_name)?, [intermediate, 1])?;
        let up_scale = Bf16View::bind(tensor(&up_scale_name)?, [intermediate, 1])?;

        validate_positive_bf16_scales(&gate_scale)?;
        validate_positive_bf16_scales(&up_scale)?;

        let weight_e4m3 = adjacent(
            &gate_weight_name,
            &up_weight_name,
            &format!("layer-{layer} FP8 gate/up weights"),
        )?;
        let scale_bf16 = adjacent(
            &gate_scale_name,
            &up_scale_name,
            &format!("layer-{layer} FP8 gate/up scales"),
        )?;
        let rows = A::INTERMEDIATE.checked_mul(2).ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{layer} dense-FP8 gate/up row count overflows"
            ))
        })?;

        Ok(Self {
            weight_e4m3,
            scale_bf16,
            rows,
            columns: A::HIDDEN,
            layer,
        })
    }
}

/// Source-native down-projection planes for one dense-FP8 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct DenseFp8DownBindings<'a> {
    /// E4M3 weights `[hidden, intermediate]`.
    pub weight: Fp8E4M3View<'a, 2>,
    /// One little-endian BF16 scale per output row `[hidden, 1]`.
    pub scale: Bf16View<'a, 2>,
    /// Decoder layer owning these planes.
    pub layer: usize,
}

impl<'a> DenseFp8DownBindings<'a> {
    /// Binds one admitted dense-FP8 down-projection source family.
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
        require_dense_fp8_mlp_layer::<A>(layer)?;

        let hidden = A::HIDDEN as u64;
        let intermediate = A::INTERMEDIATE as u64;
        let prefix = format!("model.language_model.layers.{layer}.mlp.down_proj");
        let weight =
            Fp8E4M3View::bind(tensor(&format!("{prefix}.weight"))?, [hidden, intermediate])?;
        let scale = Bf16View::bind(tensor(&format!("{prefix}.weight_scale"))?, [hidden, 1])?;

        validate_positive_bf16_scales(&scale)?;

        Ok(Self {
            weight,
            scale,
            layer,
        })
    }
}

/// Complete source planes for one late-layer dense-FP8 MLP boundary.
#[derive(Clone, Copy, Debug)]
pub struct DenseFp8MlpBindings<'a> {
    /// Source-adjacent gate/up weights and scales.
    pub gate_up: DenseFp8GateUpBindings<'a>,
    /// Source-native down-projection weights and scales.
    pub down: DenseFp8DownBindings<'a>,
    /// Zero-centered RMSNorm weights before the MLP.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary, or final norm at layer 63.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this MLP boundary.
    pub layer: usize,
}

impl<'a> DenseFp8MlpBindings<'a> {
    /// Binds one complete admitted dense-FP8 MLP source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(
            layer,
            |name| snapshot.tensor(name),
            |first, second, role| snapshot.adjacent_tensor_bytes(first, second, role),
        )
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
        mut adjacent: impl FnMut(&str, &str, &str) -> CheckpointResult<&'a [u8]>,
    ) -> CheckpointResult<Self> {
        let gate_up = DenseFp8GateUpBindings::bind_from::<A>(
            layer,
            |name| tensor(name),
            |first, second, role| adjacent(first, second, role),
        )?;
        let down = DenseFp8DownBindings::bind_from::<A>(layer, |name| tensor(name))?;
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [A::HIDDEN as u64],
        )?;
        let next_norm_name = dense_fp8_next_norm_name::<A>(layer)?;
        let next_norm = Bf16View::bind(tensor(&next_norm_name)?, [A::HIDDEN as u64])?;

        Ok(Self {
            gate_up,
            down,
            input_norm,
            next_norm,
            layer,
        })
    }
}

fn dense_fp8_next_norm_name<A: Arch>(layer: usize) -> CheckpointResult<String> {
    require_dense_fp8_mlp_layer::<A>(layer)?;
    let next_layer = layer
        .checked_add(1)
        .ok_or_else(|| CheckpointError::source_binding("dense-FP8 MLP layer overflows"))?;

    Ok(if next_layer == A::LAYERS {
        FINAL_NORM.to_string()
    } else {
        format!("model.language_model.layers.{next_layer}.input_layernorm.weight")
    })
}

/// Exact FP8 query/gate, key, and value source planes for one full-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct FullAttentionQkvBindings<'a> {
    /// Query rows followed by gate rows `[attention_query_rows, hidden]`.
    pub query_gate_weight: Fp8E4M3View<'a, 2>,
    /// Key projection `[attention_kv_rows, hidden]`.
    pub key_weight: Fp8E4M3View<'a, 2>,
    /// Value projection `[attention_kv_rows, hidden]`.
    pub value_weight: Fp8E4M3View<'a, 2>,
    /// One little-endian BF16 scale per query/gate row.
    pub query_gate_scale: Bf16View<'a, 2>,
    /// One little-endian BF16 scale per key row.
    pub key_scale: Bf16View<'a, 2>,
    /// One little-endian BF16 scale per value row.
    pub value_scale: Bf16View<'a, 2>,
    /// Decoder layer owning these planes.
    pub layer: usize,
    pub(crate) layer_count: usize,
    pub(crate) full_attention_interval: usize,
}

impl<'a> FullAttentionQkvBindings<'a> {
    /// Binds one admitted full-attention QKV source family.
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
        require_full_attention_layer(layer, A::LAYERS, A::FULL_ATTENTION_INTERVAL)?;

        let query_rows = A::ATTENTION_QUERY_ROWS as u64;
        let kv_rows = A::ATTENTION_KV_ROWS as u64;
        let hidden = A::HIDDEN as u64;
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let query_gate_weight = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.q_proj.weight"))?,
            [query_rows, hidden],
        )?;
        let key_weight = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.k_proj.weight"))?,
            [kv_rows, hidden],
        )?;
        let value_weight = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.v_proj.weight"))?,
            [kv_rows, hidden],
        )?;
        let query_gate_scale = Bf16View::bind(
            tensor(&format!("{prefix}.q_proj.weight_scale"))?,
            [query_rows, 1],
        )?;
        let key_scale = Bf16View::bind(
            tensor(&format!("{prefix}.k_proj.weight_scale"))?,
            [kv_rows, 1],
        )?;
        let value_scale = Bf16View::bind(
            tensor(&format!("{prefix}.v_proj.weight_scale"))?,
            [kv_rows, 1],
        )?;

        Self::from_views::<A>(
            layer,
            [query_gate_weight, key_weight, value_weight],
            [query_gate_scale, key_scale, value_scale],
        )
    }

    /// Constructs one exact QKV family from validated source views.
    pub fn from_views<A: Arch>(
        layer: usize,
        weights: [Fp8E4M3View<'a, 2>; 3],
        scales: [Bf16View<'a, 2>; 3],
    ) -> CheckpointResult<Self> {
        require_full_attention_layer(layer, A::LAYERS, A::FULL_ATTENTION_INTERVAL)?;

        let query_rows = A::ATTENTION_QUERY_ROWS as u64;
        let kv_rows = A::ATTENTION_KV_ROWS as u64;
        let hidden = A::HIDDEN as u64;
        let [query_gate_weight, key_weight, value_weight] = weights;
        let [query_gate_scale, key_scale, value_scale] = scales;

        let expected_weight_shapes = [[query_rows, hidden], [kv_rows, hidden], [kv_rows, hidden]];
        let observed_weight_shapes = [
            *query_gate_weight.shape(),
            *key_weight.shape(),
            *value_weight.shape(),
        ];
        let expected_scale_shapes = [[query_rows, 1], [kv_rows, 1], [kv_rows, 1]];
        let observed_scale_shapes = [
            *query_gate_scale.shape(),
            *key_scale.shape(),
            *value_scale.shape(),
        ];

        if observed_weight_shapes != expected_weight_shapes
            || observed_scale_shapes != expected_scale_shapes
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} full-attention QKV views do not match the target shapes"
            )));
        }

        validate_positive_bf16_scales(&query_gate_scale)?;
        validate_positive_bf16_scales(&key_scale)?;
        validate_positive_bf16_scales(&value_scale)?;

        Ok(Self {
            query_gate_weight,
            key_weight,
            value_weight,
            query_gate_scale,
            key_scale,
            value_scale,
            layer,
            layer_count: A::LAYERS,
            full_attention_interval: A::FULL_ATTENTION_INTERVAL,
        })
    }
}

/// Output projection, normalization, and cache-scale sources for one full-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct FullAttentionPostBindings<'a> {
    /// Source-native output projection `[hidden, attention_output_columns]`.
    pub output_weight: Fp8E4M3View<'a, 2>,
    /// One little-endian BF16 scale per output row `[hidden, 1]`.
    pub output_scale: Bf16View<'a, 2>,
    /// Zero-centered RMSNorm weights before attention `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Per-head query RMSNorm weights `[head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights `[head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
    /// Exact finite positive BF16 key-cache scale word.
    pub key_cache_scale_bf16: u16,
    /// Exact finite positive BF16 value-cache scale word.
    pub value_cache_scale_bf16: u16,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> FullAttentionPostBindings<'a> {
    /// Binds one admitted full-attention post-projection source family.
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
        require_full_attention_layer(layer, A::LAYERS, A::FULL_ATTENTION_INTERVAL)?;

        let hidden = A::HIDDEN as u64;
        let output_columns = A::ATTENTION_OUTPUT_COLUMNS as u64;
        let head_dim = A::HEAD_DIM as u64;
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let output_weight = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.o_proj.weight"))?,
            [hidden, output_columns],
        )?;
        let output_scale = Bf16View::bind(
            tensor(&format!("{prefix}.o_proj.weight_scale"))?,
            [hidden, 1],
        )?;

        validate_positive_bf16_scales(&output_scale)?;

        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
            [hidden],
        )?;
        let post_attention_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [hidden],
        )?;
        let query_norm = Bf16View::bind(tensor(&format!("{prefix}.q_norm.weight"))?, [head_dim])?;
        let key_norm = Bf16View::bind(tensor(&format!("{prefix}.k_norm.weight"))?, [head_dim])?;
        let key_cache_scale_bf16 = positive_bf16(tensor(&format!("{prefix}.k_scale"))?)?;
        let value_cache_scale_bf16 = positive_bf16(tensor(&format!("{prefix}.v_scale"))?)?;

        Ok(Self {
            output_weight,
            output_scale,
            input_norm,
            post_attention_norm,
            query_norm,
            key_norm,
            key_cache_scale_bf16,
            value_cache_scale_bf16,
            layer,
        })
    }
}

/// Complete source-native planes for one GDN mixer layer.
#[derive(Clone, Copy, Debug)]
pub struct GdnBindings<'a> {
    /// QKV rows followed by Z rows as one E4M3 source span `[gdn_input_rows, hidden]`.
    pub input_weight_e4m3: &'a [u8],
    /// QKV row scales followed by Z row scales as one little-endian BF16 source span.
    pub input_scale_bf16: &'a [u8],
    /// Fused QKV/Z row count.
    pub input_rows: usize,
    /// Logical input width.
    pub input_columns: usize,
    /// Per-value-head A-control projection `[gdn_control_rows, hidden]`.
    pub a_control_weight: Bf16View<'a, 2>,
    /// Per-value-head B-control projection `[gdn_control_rows, hidden]`.
    pub b_control_weight: Bf16View<'a, 2>,
    /// Width-four causal-convolution weights `[gdn_qkv_rows, 1, kernel]`.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters `[gdn_control_rows]`.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias `[gdn_control_rows]`.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights `[linear_head_dim]`.
    pub norm: Bf16View<'a, 1>,
    /// Source-native output projection `[hidden, gdn_value_rows]`.
    pub output_weight: Fp8E4M3View<'a, 2>,
    /// One positive little-endian BF16 scale per output row `[hidden, 1]`.
    pub output_scale: Bf16View<'a, 2>,
    /// Zero-centered RMSNorm weights before the mixer `[hidden]`.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> GdnBindings<'a> {
    /// Binds one admitted GDN mixer source family.
    pub fn bind<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from::<A>(
            layer,
            |name| snapshot.tensor(name),
            |first, second, role| snapshot.adjacent_tensor_bytes(first, second, role),
        )
    }

    fn bind_from<A: Arch>(
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
        mut adjacent: impl FnMut(&str, &str, &str) -> CheckpointResult<&'a [u8]>,
    ) -> CheckpointResult<Self> {
        require_gdn_layer::<A>(layer)?;

        let hidden = A::HIDDEN as u64;
        let qkv_rows = A::GDN_QKV_ROWS as u64;
        let value_rows = A::GDN_VALUE_ROWS as u64;
        let control_rows = A::GDN_CONTROL_ROWS as u64;
        let head_dim = A::LINEAR_HEAD_DIM as u64;
        let convolution = A::LINEAR_CONV_KERNEL_DIM as u64;
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let qkv_weight_name = format!("{prefix}.in_proj_qkv.weight");
        let z_weight_name = format!("{prefix}.in_proj_z.weight");
        let qkv_scale_name = format!("{prefix}.in_proj_qkv.weight_scale");
        let z_scale_name = format!("{prefix}.in_proj_z.weight_scale");

        Fp8E4M3View::bind(tensor(&qkv_weight_name)?, [qkv_rows, hidden])?;
        Fp8E4M3View::bind(tensor(&z_weight_name)?, [value_rows, hidden])?;

        let input_weight_e4m3 = adjacent(
            &qkv_weight_name,
            &z_weight_name,
            &format!("layer-{layer} GDN QKV/Z weights"),
        )?;
        let qkv_scale = Bf16View::bind(tensor(&qkv_scale_name)?, [qkv_rows, 1])?;
        let z_scale = Bf16View::bind(tensor(&z_scale_name)?, [value_rows, 1])?;

        validate_positive_bf16_scales(&qkv_scale)?;
        validate_positive_bf16_scales(&z_scale)?;

        let input_scale_bf16 = adjacent(
            &qkv_scale_name,
            &z_scale_name,
            &format!("layer-{layer} GDN QKV/Z scales"),
        )?;
        let a_control_weight = Bf16View::bind(
            tensor(&format!("{prefix}.in_proj_a.weight"))?,
            [control_rows, hidden],
        )?;
        let b_control_weight = Bf16View::bind(
            tensor(&format!("{prefix}.in_proj_b.weight"))?,
            [control_rows, hidden],
        )?;
        let convolution_weight = Bf16View::bind(
            tensor(&format!("{prefix}.conv1d.weight"))?,
            [qkv_rows, 1, convolution],
        )?;
        let a_log = Bf16View::bind(tensor(&format!("{prefix}.A_log"))?, [control_rows])?;
        let dt_bias = Bf16View::bind(tensor(&format!("{prefix}.dt_bias"))?, [control_rows])?;
        let norm = Bf16View::bind(tensor(&format!("{prefix}.norm.weight"))?, [head_dim])?;
        let output_weight = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.out_proj.weight"))?,
            [hidden, value_rows],
        )?;
        let output_scale = Bf16View::bind(
            tensor(&format!("{prefix}.out_proj.weight_scale"))?,
            [hidden, 1],
        )?;

        validate_positive_bf16_scales(&output_scale)?;

        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.input_layernorm.weight"))?,
            [hidden],
        )?;
        let post_attention_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [hidden],
        )?;

        Ok(Self {
            input_weight_e4m3,
            input_scale_bf16,
            input_rows: A::GDN_INPUT_ROWS,
            input_columns: A::HIDDEN,
            a_control_weight,
            b_control_weight,
            convolution_weight,
            a_log,
            dt_bias,
            norm,
            output_weight,
            output_scale,
            input_norm,
            post_attention_norm,
            layer,
        })
    }
}

/// Complete BF16 source planes for the single admitted MTP draft layer.
#[derive(Clone, Copy, Debug)]
pub struct MtpBindings<'a> {
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
    /// Gate rows followed by up rows as one BF16 source span `[2 * intermediate, hidden]`.
    pub gate_up_weight_bf16: &'a [u8],
    /// Fused gate/up row count.
    pub gate_up_rows: usize,
    /// Logical MLP input width.
    pub gate_up_columns: usize,
    /// MLP down projection `[hidden, intermediate]`.
    pub down_weight: Bf16View<'a, 2>,
    /// Zero-centered RMSNorm weights before the MLP `[hidden]`.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Final draft hidden-state normalization `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
}

impl<'a> MtpBindings<'a> {
    /// Binds the complete admitted MTP source family.
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(
            |name| snapshot.tensor(name),
            |first, second, role| snapshot.adjacent_tensor_bytes(first, second, role),
        )
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
        mut adjacent: impl FnMut(&str, &str, &str) -> CheckpointResult<&'a [u8]>,
    ) -> CheckpointResult<Self> {
        require_mtp_contract(A::MTP_LAYERS, A::MTP_USES_DEDICATED_EMBEDDINGS)?;

        let hidden = A::HIDDEN as u64;
        let intermediate = A::INTERMEDIATE as u64;
        let query_rows = A::ATTENTION_QUERY_ROWS as u64;
        let kv_rows = A::ATTENTION_KV_ROWS as u64;
        let output_columns = A::ATTENTION_OUTPUT_COLUMNS as u64;
        let head_dim = A::HEAD_DIM as u64;
        let input_columns = hidden.checked_mul(2).ok_or_else(|| {
            CheckpointError::source_binding("MTP input projection width overflows")
        })?;
        let prefix = format!("mtp.layers.{MTP_LAYER}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let gate_weight_name = format!("{mlp}.gate_proj.weight");
        let up_weight_name = format!("{mlp}.up_proj.weight");

        let input_projection = Bf16View::bind(tensor("mtp.fc.weight")?, [hidden, input_columns])?;
        let embedding_norm = Bf16View::bind(tensor("mtp.pre_fc_norm_embedding.weight")?, [hidden])?;
        let hidden_norm = Bf16View::bind(tensor("mtp.pre_fc_norm_hidden.weight")?, [hidden])?;
        let query_gate_weight = Bf16View::bind(
            tensor(&format!("{attention}.q_proj.weight"))?,
            [query_rows, hidden],
        )?;
        let key_weight = Bf16View::bind(
            tensor(&format!("{attention}.k_proj.weight"))?,
            [kv_rows, hidden],
        )?;
        let value_weight = Bf16View::bind(
            tensor(&format!("{attention}.v_proj.weight"))?,
            [kv_rows, hidden],
        )?;
        let attention_output_weight = Bf16View::bind(
            tensor(&format!("{attention}.o_proj.weight"))?,
            [hidden, output_columns],
        )?;
        let query_norm =
            Bf16View::bind(tensor(&format!("{attention}.q_norm.weight"))?, [head_dim])?;
        let key_norm = Bf16View::bind(tensor(&format!("{attention}.k_norm.weight"))?, [head_dim])?;
        let input_norm = Bf16View::bind(
            tensor(&format!("{prefix}.input_layernorm.weight"))?,
            [hidden],
        )?;

        Bf16View::bind(tensor(&gate_weight_name)?, [intermediate, hidden])?;
        Bf16View::bind(tensor(&up_weight_name)?, [intermediate, hidden])?;

        let gate_up_weight_bf16 =
            adjacent(&gate_weight_name, &up_weight_name, "MTP gate/up weights")?;
        let gate_up_rows = A::INTERMEDIATE
            .checked_mul(2)
            .ok_or_else(|| CheckpointError::source_binding("MTP gate/up row count overflows"))?;
        let down_weight = Bf16View::bind(
            tensor(&format!("{mlp}.down_proj.weight"))?,
            [hidden, intermediate],
        )?;
        let post_attention_norm = Bf16View::bind(
            tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            [hidden],
        )?;
        let final_norm = Bf16View::bind(tensor("mtp.norm.weight")?, [hidden])?;

        Ok(Self {
            input_projection,
            embedding_norm,
            hidden_norm,
            query_gate_weight,
            key_weight,
            value_weight,
            attention_output_weight,
            query_norm,
            key_norm,
            input_norm,
            gate_up_weight_bf16,
            gate_up_rows,
            gate_up_columns: A::HIDDEN,
            down_weight,
            post_attention_norm,
            final_norm,
        })
    }
}

/// BF16 source planes for one Vision transformer block.
#[derive(Clone, Copy, Debug)]
pub struct VisionBlockBindings<'a> {
    /// Attention output projection bias `[vision_hidden]`.
    pub attention_projection_bias: Bf16View<'a, 1>,
    /// Attention output projection weights `[vision_hidden, vision_hidden]`.
    pub attention_projection_weight: Bf16View<'a, 2>,
    /// Fused query, key, and value bias `[3 * vision_hidden]`.
    pub qkv_bias: Bf16View<'a, 1>,
    /// Fused query, key, and value weights `[3 * vision_hidden, vision_hidden]`.
    pub qkv_weight: Bf16View<'a, 2>,
    /// First MLP projection bias `[vision_intermediate]`.
    pub mlp_fc1_bias: Bf16View<'a, 1>,
    /// First MLP projection weights `[vision_intermediate, vision_hidden]`.
    pub mlp_fc1_weight: Bf16View<'a, 2>,
    /// Second MLP projection bias `[vision_hidden]`.
    pub mlp_fc2_bias: Bf16View<'a, 1>,
    /// Second MLP projection weights `[vision_hidden, vision_intermediate]`.
    pub mlp_fc2_weight: Bf16View<'a, 2>,
    /// First pre-normalization bias `[vision_hidden]`.
    pub norm1_bias: Bf16View<'a, 1>,
    /// First pre-normalization weights `[vision_hidden]`.
    pub norm1_weight: Bf16View<'a, 1>,
    /// Second pre-normalization bias `[vision_hidden]`.
    pub norm2_bias: Bf16View<'a, 1>,
    /// Second pre-normalization weights `[vision_hidden]`.
    pub norm2_weight: Bf16View<'a, 1>,
    /// Zero-based Vision block index.
    pub block: usize,
}

impl<'a> VisionBlockBindings<'a> {
    fn bind_from<A: Arch>(
        block: usize,
        tensor: &mut impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let hidden = A::VISION_HIDDEN as u64;
        let intermediate = A::VISION_INTERMEDIATE as u64;
        let qkv_rows = hidden.checked_mul(3).ok_or_else(|| {
            CheckpointError::source_binding(format!("Vision block-{block} QKV row count overflows"))
        })?;
        let prefix = format!("model.visual.blocks.{block}");

        Ok(Self {
            attention_projection_bias: Bf16View::bind(
                tensor(&format!("{prefix}.attn.proj.bias"))?,
                [hidden],
            )?,
            attention_projection_weight: Bf16View::bind(
                tensor(&format!("{prefix}.attn.proj.weight"))?,
                [hidden, hidden],
            )?,
            qkv_bias: Bf16View::bind(tensor(&format!("{prefix}.attn.qkv.bias"))?, [qkv_rows])?,
            qkv_weight: Bf16View::bind(
                tensor(&format!("{prefix}.attn.qkv.weight"))?,
                [qkv_rows, hidden],
            )?,
            mlp_fc1_bias: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc1.bias"))?,
                [intermediate],
            )?,
            mlp_fc1_weight: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc1.weight"))?,
                [intermediate, hidden],
            )?,
            mlp_fc2_bias: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc2.bias"))?,
                [hidden],
            )?,
            mlp_fc2_weight: Bf16View::bind(
                tensor(&format!("{prefix}.mlp.linear_fc2.weight"))?,
                [hidden, intermediate],
            )?,
            norm1_bias: Bf16View::bind(tensor(&format!("{prefix}.norm1.bias"))?, [hidden])?,
            norm1_weight: Bf16View::bind(tensor(&format!("{prefix}.norm1.weight"))?, [hidden])?,
            norm2_bias: Bf16View::bind(tensor(&format!("{prefix}.norm2.bias"))?, [hidden])?,
            norm2_weight: Bf16View::bind(tensor(&format!("{prefix}.norm2.weight"))?, [hidden])?,
            block,
        })
    }
}

/// Complete BF16 source family for the admitted Vision encoder.
#[derive(Debug)]
pub struct VisionBindings<'a> {
    /// Transformer blocks in execution order.
    pub blocks: Vec<VisionBlockBindings<'a>>,
    /// Patch projection bias `[vision_hidden]`.
    pub patch_embedding_bias: Bf16View<'a, 1>,
    /// Spatiotemporal patch projection weights.
    pub patch_embedding_weight: Bf16View<'a, 5>,
    /// Learned position embeddings `[vision_positions, vision_hidden]`.
    pub position_embedding: Bf16View<'a, 2>,
    /// Patch-merger normalization bias `[vision_hidden]`.
    pub merger_norm_bias: Bf16View<'a, 1>,
    /// Patch-merger normalization weights `[vision_hidden]`.
    pub merger_norm_weight: Bf16View<'a, 1>,
    /// First patch-merger projection bias `[merged_patch_width]`.
    pub merger_fc1_bias: Bf16View<'a, 1>,
    /// First patch-merger projection weights `[merged_patch_width, merged_patch_width]`.
    pub merger_fc1_weight: Bf16View<'a, 2>,
    /// Text-width patch-merger output bias `[vision_output_hidden]`.
    pub merger_fc2_bias: Bf16View<'a, 1>,
    /// Text-width patch-merger output weights `[vision_output_hidden, merged_patch_width]`.
    pub merger_fc2_weight: Bf16View<'a, 2>,
}

impl<'a> VisionBindings<'a> {
    /// Binds the complete admitted Vision source family.
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(|name| snapshot.tensor(name))
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let hidden = A::VISION_HIDDEN as u64;
        let output_hidden = A::VISION_OUTPUT_HIDDEN as u64;
        let merge_area = A::VISION_SPATIAL_MERGE_SIZE
            .checked_mul(A::VISION_SPATIAL_MERGE_SIZE)
            .ok_or_else(|| CheckpointError::source_binding("Vision merge area overflows"))?;
        let merged_width = A::VISION_HIDDEN
            .checked_mul(merge_area)
            .ok_or_else(|| CheckpointError::source_binding("Vision merged width overflows"))?
            as u64;
        let mut blocks = Vec::with_capacity(A::VISION_DEPTH);

        for block in 0..A::VISION_DEPTH {
            blocks.push(VisionBlockBindings::bind_from::<A>(block, &mut tensor)?);
        }

        Ok(Self {
            blocks,
            patch_embedding_bias: Bf16View::bind(
                tensor("model.visual.patch_embed.proj.bias")?,
                [hidden],
            )?,
            patch_embedding_weight: Bf16View::bind(
                tensor("model.visual.patch_embed.proj.weight")?,
                [
                    hidden,
                    A::VISION_INPUT_CHANNELS as u64,
                    A::VISION_TEMPORAL_PATCH_SIZE as u64,
                    A::VISION_PATCH_SIZE as u64,
                    A::VISION_PATCH_SIZE as u64,
                ],
            )?,
            position_embedding: Bf16View::bind(
                tensor("model.visual.pos_embed.weight")?,
                [A::VISION_POSITIONS as u64, hidden],
            )?,
            merger_norm_bias: Bf16View::bind(tensor("model.visual.merger.norm.bias")?, [hidden])?,
            merger_norm_weight: Bf16View::bind(
                tensor("model.visual.merger.norm.weight")?,
                [hidden],
            )?,
            merger_fc1_bias: Bf16View::bind(
                tensor("model.visual.merger.linear_fc1.bias")?,
                [merged_width],
            )?,
            merger_fc1_weight: Bf16View::bind(
                tensor("model.visual.merger.linear_fc1.weight")?,
                [merged_width, merged_width],
            )?,
            merger_fc2_bias: Bf16View::bind(
                tensor("model.visual.merger.linear_fc2.bias")?,
                [output_hidden],
            )?,
            merger_fc2_weight: Bf16View::bind(
                tensor("model.visual.merger.linear_fc2.weight")?,
                [output_hidden, merged_width],
            )?,
        })
    }
}

/// Exact ModelOpt NVFP4 planes for one linear projection.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptNvfp4LinearBindings<'a> {
    /// Packed E2M1 weights `[rows, columns / 2]`.
    pub weight: U8View<'a, 2>,
    /// E4M3 block scales `[rows, columns / 16]`.
    pub block_scale: Fp8E4M3View<'a, 2>,
    /// Positive source activation scale stored as one rank-zero F32 value.
    pub input_scale: F32View<'a, 0>,
    /// Positive second-stage weight scale stored as one rank-zero F32 value.
    pub weight_scale_2: F32View<'a, 0>,
    /// Logical output row count.
    pub rows: usize,
    /// Logical input column count before E2M1 packing.
    pub columns: usize,
}

impl<'a> ModelOptNvfp4LinearBindings<'a> {
    fn bind_from(
        prefix: &str,
        rows: usize,
        columns: usize,
        layer: usize,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let packed_columns = codec_columns(columns, E2M1_VALUES_PER_BYTE, "packed E2M1")?;
        let scale_columns = codec_columns(columns, NVFP4_GROUP_SIZE, "E4M3 block-scale")?;
        let weight = U8View::bind(
            tensor(&format!("{prefix}.weight"))?,
            [rows as u64, packed_columns],
        )?;
        let block_scale = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.weight_scale"))?,
            [rows as u64, scale_columns],
        )?;

        validate_nvfp4_scales(layer, prefix, block_scale.codes())?;

        Ok(Self {
            weight,
            block_scale,
            input_scale: positive_rank_zero_f32(tensor(&format!("{prefix}.input_scale"))?)?,
            weight_scale_2: positive_rank_zero_f32(tensor(&format!("{prefix}.weight_scale_2"))?)?,
            rows,
            columns,
        })
    }
}

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

/// Exact packed gate/up source planes for one NVFP4 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4GateUpBindings<'a> {
    /// Packed E2M1 gate weights `[intermediate, hidden / 2]`.
    pub gate_weight: U8View<'a, 2>,
    /// Packed E2M1 up weights `[intermediate, hidden / 2]`.
    pub up_weight: U8View<'a, 2>,
    /// E4M3 gate block scales `[intermediate, hidden / 16]`.
    pub gate_scale: Fp8E4M3View<'a, 2>,
    /// E4M3 up block scales `[intermediate, hidden / 16]`.
    pub up_scale: Fp8E4M3View<'a, 2>,
    /// Shared finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Shared finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Decoder layer owning these planes.
    pub layer: usize,
    /// Total decoder layer count of the bound architecture.
    pub layer_count: usize,
}

impl<'a> Nvfp4GateUpBindings<'a> {
    /// Binds one admitted NVFP4 gate/up source family.
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
        require_nvfp4_mlp_layer(layer, A::LAYERS)?;

        let intermediate = A::INTERMEDIATE as u64;
        let packed_columns = codec_columns(A::HIDDEN, E2M1_VALUES_PER_BYTE, "packed E2M1")?;
        let scale_columns = codec_columns(A::HIDDEN, NVFP4_GROUP_SIZE, "E4M3 block-scale")?;
        let prefix = format!("model.language_model.layers.{layer}.mlp");

        let gate_weight = tensor(&format!("{prefix}.gate_proj.weight_packed"))?;
        let up_weight = tensor(&format!("{prefix}.up_proj.weight_packed"))?;
        let gate_scale = tensor(&format!("{prefix}.gate_proj.weight_scale"))?;
        let up_scale = tensor(&format!("{prefix}.up_proj.weight_scale"))?;

        require_adjacent(layer, "packed gate/up weights", &gate_weight, &up_weight)?;
        require_adjacent(layer, "gate/up scale planes", &gate_scale, &up_scale)?;

        let gate_weight = U8View::bind(gate_weight, [intermediate, packed_columns])?;
        let up_weight = U8View::bind(up_weight, [intermediate, packed_columns])?;
        let gate_scale = Fp8E4M3View::bind(gate_scale, [intermediate, scale_columns])?;
        let up_scale = Fp8E4M3View::bind(up_scale, [intermediate, scale_columns])?;

        validate_nvfp4_scales(layer, "gate", gate_scale.codes())?;
        validate_nvfp4_scales(layer, "up", up_scale.codes())?;

        let gate_input_divisor =
            positive_f32(tensor(&format!("{prefix}.gate_proj.input_global_scale"))?)?;
        let up_input_divisor =
            positive_f32(tensor(&format!("{prefix}.up_proj.input_global_scale"))?)?;
        let gate_weight_divisor =
            positive_f32(tensor(&format!("{prefix}.gate_proj.weight_global_scale"))?)?;
        let up_weight_divisor =
            positive_f32(tensor(&format!("{prefix}.up_proj.weight_global_scale"))?)?;

        require_same_divisor(
            layer,
            "input_global_scale",
            gate_input_divisor,
            up_input_divisor,
        )?;
        require_same_divisor(
            layer,
            "weight_global_scale",
            gate_weight_divisor,
            up_weight_divisor,
        )?;

        Ok(Self {
            gate_weight,
            up_weight,
            gate_scale,
            up_scale,
            input_scale_divisor: gate_input_divisor,
            weight_scale_divisor: gate_weight_divisor,
            layer,
            layer_count: A::LAYERS,
        })
    }
}

/// Exact packed down-projection source planes for one NVFP4 MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4DownBindings<'a> {
    /// Packed E2M1 weights `[hidden, intermediate / 2]`.
    pub weight: U8View<'a, 2>,
    /// E4M3 block scales `[hidden, intermediate / 16]`.
    pub scale: Fp8E4M3View<'a, 2>,
    /// Finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Decoder layer owning these planes.
    pub layer: usize,
    /// Total decoder layer count of the bound architecture.
    pub layer_count: usize,
}

/// Complete source planes for one early-layer NVFP4 MLP boundary.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4MlpBindings<'a> {
    /// Packed gate/up weights, block scales, and divisors.
    pub gate_up: Nvfp4GateUpBindings<'a>,
    /// Packed down-projection weights, block scales, and divisors.
    pub down: Nvfp4DownBindings<'a>,
    /// Zero-centered RMSNorm weights before the MLP.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this MLP boundary.
    pub layer: usize,
}

impl<'a> Nvfp4MlpBindings<'a> {
    /// Binds one complete admitted NVFP4 MLP source family.
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
        let gate_up = Nvfp4GateUpBindings::bind_from::<A>(layer, |name| tensor(name))?;
        let down = Nvfp4DownBindings::bind_from::<A>(layer, |name| tensor(name))?;
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let input_norm = Bf16View::bind(
            tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            [A::HIDDEN as u64],
        )?;
        let next_norm = Bf16View::bind(
            tensor(&nvfp4_next_norm_name::<A>(layer)?)?,
            [A::HIDDEN as u64],
        )?;

        Ok(Self {
            gate_up,
            down,
            input_norm,
            next_norm,
            layer,
        })
    }
}

fn nvfp4_next_norm_name<A: Arch>(layer: usize) -> CheckpointResult<String> {
    require_nvfp4_mlp_layer(layer, A::LAYERS)?;
    let next_layer = layer
        .checked_add(1)
        .ok_or_else(|| CheckpointError::source_binding("NVFP4 MLP layer overflows"))?;

    Ok(if next_layer == A::LAYERS {
        FINAL_NORM.to_string()
    } else {
        format!("model.language_model.layers.{next_layer}.input_layernorm.weight")
    })
}

impl<'a> Nvfp4DownBindings<'a> {
    /// Binds one admitted NVFP4 down-projection source family.
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
        require_nvfp4_mlp_layer(layer, A::LAYERS)?;

        let hidden = A::HIDDEN as u64;
        let packed_columns = codec_columns(A::INTERMEDIATE, E2M1_VALUES_PER_BYTE, "packed E2M1")?;
        let scale_columns = codec_columns(A::INTERMEDIATE, NVFP4_GROUP_SIZE, "E4M3 block-scale")?;
        let prefix = format!("model.language_model.layers.{layer}.mlp.down_proj");

        let weight = U8View::bind(
            tensor(&format!("{prefix}.weight_packed"))?,
            [hidden, packed_columns],
        )?;
        let scale = Fp8E4M3View::bind(
            tensor(&format!("{prefix}.weight_scale"))?,
            [hidden, scale_columns],
        )?;

        validate_nvfp4_scales(layer, "down", scale.codes())?;

        let input_scale_divisor = positive_f32(tensor(&format!("{prefix}.input_global_scale"))?)?;
        let weight_scale_divisor = positive_f32(tensor(&format!("{prefix}.weight_global_scale"))?)?;

        Ok(Self {
            weight,
            scale,
            input_scale_divisor,
            weight_scale_divisor,
            layer,
            layer_count: A::LAYERS,
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

/// Shape- and dtype-checked source views for the text input and output endpoints.
#[derive(Clone, Copy, Debug)]
pub struct TextEndpointBindings<'a> {
    /// BF16 token embedding matrix `[vocab, hidden]`.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 final RMSNorm weights `[hidden]`.
    pub final_norm: Bf16View<'a, 1>,
    /// FP8 E4M3 language-model head `[vocab, hidden]`.
    pub lm_head: Fp8E4M3View<'a, 2>,
    /// Per-vocabulary-row BF16 language-model head scales `[vocab, 1]`.
    pub lm_head_scale: Bf16View<'a, 2>,
}

impl<'a> TextEndpointBindings<'a> {
    /// Binds the admitted embedding, final-norm, and LM-head sources.
    pub fn bind<A: Arch>(snapshot: &'a CheckpointSnapshot<A>) -> CheckpointResult<Self> {
        Self::bind_from::<A>(|name| snapshot.tensor(name))
    }

    /// Binds only the mmap-backed embedding used during token staging.
    pub fn bind_embedding<A: Arch>(
        snapshot: &'a CheckpointSnapshot<A>,
    ) -> CheckpointResult<Bf16View<'a, 2>> {
        Bf16View::bind(
            snapshot.tensor(EMBEDDING)?,
            [A::VOCAB as u64, A::HIDDEN as u64],
        )
    }

    fn bind_from<A: Arch>(
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let vocab = A::VOCAB as u64;
        let hidden = A::HIDDEN as u64;

        let embedding = Bf16View::bind(tensor(EMBEDDING)?, [vocab, hidden])?;
        let final_norm = Bf16View::bind(tensor(FINAL_NORM)?, [hidden])?;
        let lm_head = Fp8E4M3View::bind(tensor(LM_HEAD)?, [vocab, hidden])?;
        let lm_head_scale = Bf16View::bind(tensor(LM_HEAD_SCALE)?, [vocab, 1])?;

        validate_positive_bf16_scales(&lm_head_scale)?;

        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            lm_head_scale,
        })
    }
}

pub(crate) fn require_nvfp4_mlp_layer(layer: usize, layer_count: usize) -> CheckpointResult<()> {
    if layer >= layer_count || layer >= NVFP4_MLP_LAYER_END {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted NVFP4 MLP source contract"
        )));
    }

    Ok(())
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

fn require_dense_fp8_mlp_layer<A: Arch>(layer: usize) -> CheckpointResult<()> {
    if !(DENSE_FP8_MLP_LAYER_START..A::LAYERS).contains(&layer) {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted dense-FP8 MLP source contract"
        )));
    }

    Ok(())
}

fn require_gdn_layer<A: Arch>(layer: usize) -> CheckpointResult<()> {
    require_gdn_layer_route(layer, A::LAYERS, A::FULL_ATTENTION_INTERVAL)
}

pub(crate) fn require_gdn_layer_route(
    layer: usize,
    layer_count: usize,
    interval: usize,
) -> CheckpointResult<()> {
    if interval == 0 || layer >= layer_count || layer % interval == interval - 1 {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted GDN source contract"
        )));
    }

    Ok(())
}

fn require_mtp_contract(layers: usize, dedicated_embeddings: bool) -> CheckpointResult<()> {
    if layers != 1 || dedicated_embeddings {
        return Err(CheckpointError::source_binding(format!(
            "MTP source contract requires one layer without dedicated embeddings, observed {layers} layers and dedicated_embeddings={dedicated_embeddings}"
        )));
    }

    Ok(())
}

pub(crate) fn require_full_attention_layer(
    layer: usize,
    layer_count: usize,
    interval: usize,
) -> CheckpointResult<()> {
    if interval == 0 || layer >= layer_count || layer % interval != interval - 1 {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted full-attention source contract"
        )));
    }

    Ok(())
}

fn codec_columns(width: usize, divisor: usize, role: &str) -> CheckpointResult<u64> {
    if !width.is_multiple_of(divisor) {
        return Err(CheckpointError::source_binding(format!(
            "architecture width {width} is not divisible by the {role} divisor {divisor}"
        )));
    }

    Ok((width / divisor) as u64)
}

fn require_adjacent(
    layer: usize,
    role: &str,
    first: &TensorView<'_>,
    second: &TensorView<'_>,
) -> CheckpointResult<()> {
    if first.data_range.end != second.data_range.start {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} {role} are not source-adjacent"
        )));
    }

    Ok(())
}

pub(crate) fn validate_nvfp4_scales(
    layer: usize,
    role: &str,
    scales: &[u8],
) -> CheckpointResult<()> {
    if scales
        .iter()
        .any(|&scale| scale & 0x80 != 0 || scale == 0x7f)
    {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale plane contains a negative or NaN E4M3FN code"
        )));
    }

    Ok(())
}

fn positive_f32(tensor: TensorView<'_>) -> CheckpointResult<f32> {
    let view = F32View::bind(tensor, [1])?;
    let value = view.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "tensor `{}` must contain a finite positive divisor, observed {value}",
            view.name()
        )));
    }

    Ok(value)
}

fn positive_rank_zero_f32(tensor: TensorView<'_>) -> CheckpointResult<F32View<'_, 0>> {
    let view = F32View::bind(tensor, [])?;
    let value = view.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "tensor `{}` must contain a finite positive F32 scale, observed {value}",
            view.name()
        )));
    }

    Ok(view)
}

fn require_same_rank_zero_f32(
    layer: usize,
    role: &str,
    first: &F32View<'_, 0>,
    second: &F32View<'_, 0>,
) -> CheckpointResult<()> {
    if first.bits(0) != second.bits(0) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} {role} values differ"
        )));
    }

    Ok(())
}

fn positive_bf16(tensor: TensorView<'_>) -> CheckpointResult<u16> {
    let view = Bf16View::bind(tensor, [1])?;
    validate_positive_bf16_scales(&view)?;

    Ok(view.word(0).expect("validated scalar has one value"))
}

fn validate_positive_bf16_scales<const RANK: usize>(
    scales: &Bf16View<'_, RANK>,
) -> CheckpointResult<()> {
    for (index, bits) in scales.words().enumerate() {
        let value = f32::from_bits(u32::from(bits) << 16);

        if !value.is_finite() || value <= 0.0 {
            return Err(CheckpointError::source_binding(format!(
                "tensor `{}` must contain a finite positive BF16 scale, observed {value} at index {index}",
                scales.name()
            )));
        }
    }

    Ok(())
}

fn require_same_divisor(layer: usize, role: &str, gate: f32, up: f32) -> CheckpointResult<()> {
    if gate.to_bits() != up.to_bits() {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} gate/up {role} words differ and cannot share one fused operator"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Bf16TextEndpointBindings, DenseFp8DownBindings, DenseFp8GateUpBindings,
        DenseFp8MlpBindings, E2M1_VALUES_PER_BYTE, FullAttentionPostBindings,
        FullAttentionQkvBindings, GdnBindings, ModelOptNvfp4AttentionBindings,
        ModelOptNvfp4GdnBindings, ModelOptNvfp4MlpBindings, MtpBindings, NVFP4_GROUP_SIZE,
        Nvfp4DownBindings, Nvfp4GateUpBindings, Nvfp4MlpBindings, TextEndpointBindings,
        VisionBindings, dense_fp8_next_norm_name, positive_bf16, require_adjacent,
        require_dense_fp8_mlp_layer, require_full_attention_layer, require_gdn_layer,
        require_mtp_contract, require_nvfp4_mlp_layer, validate_nvfp4_scales,
    };
    use crate::{
        Arch, CheckpointContract, CheckpointErrorCode, CheckpointResult, DType, SafeTensorFile,
        TensorView,
    };
    use serde_json::{Value, json};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 4;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 8;
        const VOCAB: usize = 3;
        const LAYERS: usize = 1;
        const FULL_ATTENTION_INTERVAL: usize = 1;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
        const LINEAR_HEAD_DIM: usize = 1;
        const LINEAR_CONV_KERNEL_DIM: usize = 1;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 2;
        const VISION_HIDDEN: usize = 4;
        const VISION_INTERMEDIATE: usize = 6;
        const VISION_NUM_HEADS: usize = 2;
        const VISION_POSITIONS: usize = 8;
        const VISION_OUTPUT_HIDDEN: usize = 4;
        const VISION_INPUT_CHANNELS: usize = 3;
        const VISION_PATCH_SIZE: usize = 2;
        const VISION_SPATIAL_MERGE_SIZE: usize = 2;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
    }

    #[derive(Clone, Copy)]
    struct Nvfp4Arch;

    impl Arch for Nvfp4Arch {
        const MODEL_ID: &'static str = "test/model";
        const REVISION: &'static str = "test-revision";
        const HIDDEN: usize = 32;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 16;
        const VOCAB: usize = 3;
        const LAYERS: usize = 64;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        const NUM_ATTENTION_HEADS: usize = 1;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 1;
        const LINEAR_KEY_HEADS: usize = 1;
        const LINEAR_VALUE_HEADS: usize = 1;
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

    fn fixture_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tuisko-bindings-{label}-{}-{}.safetensors",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn endpoint_header() -> Value {
        json!({
            "model.language_model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [3, 4],
                "data_offsets": [0, 24]
            },
            "model.language_model.norm.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [24, 32]
            },
            "lm_head.weight": {
                "dtype": "F8_E4M3",
                "shape": [3, 4],
                "data_offsets": [32, 44]
            },
            "lm_head.weight_scale": {
                "dtype": "BF16",
                "shape": [3, 1],
                "data_offsets": [44, 50]
            }
        })
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

    fn write_safetensors(path: &Path, header: Value) {
        let payload = (0u8..50).collect::<Vec<_>>();

        write_safetensors_payload(path, header, &payload);
    }

    fn write_safetensors_payload(path: &Path, header: Value, payload: &[u8]) {
        let mut header = serde_json::to_vec(&header).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = File::create(path).unwrap();

        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
    }

    fn nvfp4_mlp_fixture(layer: usize) -> (Value, Vec<u8>) {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let down = format!("{prefix}.down_proj");
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let next_layer_prefix = format!("model.language_model.layers.{}", layer + 1);
        let mut payload = vec![0x38; 1_016];

        payload[0..4].copy_from_slice(&3.0f32.to_le_bytes());
        payload[4..8].copy_from_slice(&0.125f32.to_le_bytes());
        payload[8..12].copy_from_slice(&3.0f32.to_le_bytes());
        payload[12..16].copy_from_slice(&0.125f32.to_le_bytes());
        payload[592..596].copy_from_slice(&19.0f32.to_le_bytes());
        payload[596..600].copy_from_slice(&3_376.0f32.to_le_bytes());
        payload[888..952].fill(0x70);
        payload[952..1_016].fill(0x80);

        (
            json!({
                format!("{prefix}.gate_proj.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[0,4]
                },
                format!("{prefix}.gate_proj.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[4,8]
                },
                format!("{prefix}.up_proj.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[8,12]
                },
                format!("{prefix}.up_proj.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[12,16]
                },
                format!("{prefix}.gate_proj.weight_packed"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[16,272]
                },
                format!("{prefix}.up_proj.weight_packed"): {
                    "dtype":"U8", "shape":[16,16], "data_offsets":[272,528]
                },
                format!("{prefix}.gate_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[528,560]
                },
                format!("{prefix}.up_proj.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[16,2], "data_offsets":[560,592]
                },
                format!("{down}.input_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[592,596]
                },
                format!("{down}.weight_global_scale"): {
                    "dtype":"F32", "shape":[1], "data_offsets":[596,600]
                },
                format!("{down}.weight_packed"): {
                    "dtype":"U8", "shape":[32,8], "data_offsets":[600,856]
                },
                format!("{down}.weight_scale"): {
                    "dtype":"F8_E4M3", "shape":[32,1], "data_offsets":[856,888]
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

    fn dense_fp8_mlp_fixture(layer: usize) -> (Value, Vec<u8>) {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let down = format!("{prefix}.down_proj");
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let next_layer_prefix = format!("model.language_model.layers.{}", layer + 1);
        let mut payload = vec![0; 1_792];

        payload[0..512].fill(0x10);
        payload[512..1_024].fill(0x20);
        payload[1_024..1_056].fill(0x30);
        payload[1_056..1_088].fill(0x40);
        payload[1_088..1_600].fill(0x50);
        payload[1_600..1_664].fill(0x60);
        payload[1_664..1_728].fill(0x70);
        payload[1_728..1_792].fill(0x80);

        (
            json!({
                format!("{prefix}.gate_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[16,32], "data_offsets":[0,512]
                },
                format!("{prefix}.up_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[16,32], "data_offsets":[512,1024]
                },
                format!("{prefix}.gate_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[16,1], "data_offsets":[1024,1056]
                },
                format!("{prefix}.up_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[16,1], "data_offsets":[1056,1088]
                },
                format!("{down}.weight"): {
                    "dtype":"F8_E4M3", "shape":[32,16], "data_offsets":[1088,1600]
                },
                format!("{down}.weight_scale"): {
                    "dtype":"BF16", "shape":[32,1], "data_offsets":[1600,1664]
                },
                format!("{layer_prefix}.post_attention_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[1664,1728]
                },
                format!("{next_layer_prefix}.input_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[1728,1792]
                }
            }),
            payload,
        )
    }

    fn full_attention_fixture(layer: usize) -> (Value, Vec<u8>) {
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.self_attn");
        let mut payload = vec![0x80; 368];

        payload[0..64].fill(0x10);
        payload[64..96].fill(0x20);
        payload[96..128].fill(0x30);
        payload[128..132].fill(0x40);
        payload[132..134].fill(0x50);
        payload[134..136].fill(0x60);
        payload[136..168].fill(0x70);
        payload[168..232]
            .as_chunks_mut::<2>()
            .0
            .fill(0x3f80u16.to_le_bytes());
        payload[364..366].copy_from_slice(&0x3f80u16.to_le_bytes());
        payload[366..368].copy_from_slice(&0x3f00u16.to_le_bytes());

        (
            json!({
                format!("{prefix}.q_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[2,32], "data_offsets":[0,64]
                },
                format!("{prefix}.k_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[1,32], "data_offsets":[64,96]
                },
                format!("{prefix}.v_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[1,32], "data_offsets":[96,128]
                },
                format!("{prefix}.q_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[2,1], "data_offsets":[128,132]
                },
                format!("{prefix}.k_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[1,1], "data_offsets":[132,134]
                },
                format!("{prefix}.v_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[1,1], "data_offsets":[134,136]
                },
                format!("{prefix}.o_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[32,1], "data_offsets":[136,168]
                },
                format!("{prefix}.o_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[32,1], "data_offsets":[168,232]
                },
                format!("{layer_prefix}.input_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[232,296]
                },
                format!("{layer_prefix}.post_attention_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[296,360]
                },
                format!("{prefix}.q_norm.weight"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[360,362]
                },
                format!("{prefix}.k_norm.weight"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[362,364]
                },
                format!("{prefix}.k_scale"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[364,366]
                },
                format!("{prefix}.v_scale"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[366,368]
                }
            }),
            payload,
        )
    }

    fn gdn_fixture(layer: usize) -> (Value, Vec<u8>) {
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let prefix = format!("{layer_prefix}.linear_attn");
        let mut payload = vec![0x80; 518];

        payload[0..96].fill(0x10);
        payload[96..128].fill(0x20);
        payload[128..160].fill(0x30);
        payload[160..168]
            .as_chunks_mut::<2>()
            .0
            .fill(0x3f80u16.to_le_bytes());
        payload[168..232]
            .as_chunks_mut::<2>()
            .0
            .fill(0x3f00u16.to_le_bytes());

        (
            json!({
                format!("{prefix}.in_proj_qkv.weight"): {
                    "dtype":"F8_E4M3", "shape":[3,32], "data_offsets":[0,96]
                },
                format!("{prefix}.in_proj_z.weight"): {
                    "dtype":"F8_E4M3", "shape":[1,32], "data_offsets":[96,128]
                },
                format!("{prefix}.out_proj.weight"): {
                    "dtype":"F8_E4M3", "shape":[32,1], "data_offsets":[128,160]
                },
                format!("{prefix}.in_proj_qkv.weight_scale"): {
                    "dtype":"BF16", "shape":[3,1], "data_offsets":[160,166]
                },
                format!("{prefix}.in_proj_z.weight_scale"): {
                    "dtype":"BF16", "shape":[1,1], "data_offsets":[166,168]
                },
                format!("{prefix}.out_proj.weight_scale"): {
                    "dtype":"BF16", "shape":[32,1], "data_offsets":[168,232]
                },
                format!("{prefix}.in_proj_a.weight"): {
                    "dtype":"BF16", "shape":[1,32], "data_offsets":[232,296]
                },
                format!("{prefix}.in_proj_b.weight"): {
                    "dtype":"BF16", "shape":[1,32], "data_offsets":[296,360]
                },
                format!("{prefix}.conv1d.weight"): {
                    "dtype":"BF16", "shape":[3,1,4], "data_offsets":[360,384]
                },
                format!("{prefix}.A_log"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[384,386]
                },
                format!("{prefix}.dt_bias"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[386,388]
                },
                format!("{prefix}.norm.weight"): {
                    "dtype":"BF16", "shape":[1], "data_offsets":[388,390]
                },
                format!("{layer_prefix}.input_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[390,454]
                },
                format!("{layer_prefix}.post_attention_layernorm.weight"): {
                    "dtype":"BF16", "shape":[32], "data_offsets":[454,518]
                }
            }),
            payload,
        )
    }

    fn mtp_fixture() -> (Value, Vec<u8>) {
        let mut payload = vec![0x80; 7_812];

        payload[0..4_096].fill(0x10);
        payload[5_184..6_208].fill(0x20);
        payload[6_208..7_232].fill(0x30);
        payload[7_298..7_362].fill(0x40);
        payload[7_362..7_426].fill(0x50);
        payload[7_428..7_556].fill(0x60);
        payload[7_556..7_620].fill(0x70);

        (
            json!({
                "mtp.fc.weight": {
                    "dtype":"BF16", "shape":[32,64], "data_offsets":[0,4096]
                },
                "mtp.layers.0.input_layernorm.weight": {
                    "dtype":"BF16", "shape":[32], "data_offsets":[4096,4160]
                },
                "mtp.layers.0.mlp.down_proj.weight": {
                    "dtype":"BF16", "shape":[32,16], "data_offsets":[4160,5184]
                },
                "mtp.layers.0.mlp.gate_proj.weight": {
                    "dtype":"BF16", "shape":[16,32], "data_offsets":[5184,6208]
                },
                "mtp.layers.0.mlp.up_proj.weight": {
                    "dtype":"BF16", "shape":[16,32], "data_offsets":[6208,7232]
                },
                "mtp.layers.0.post_attention_layernorm.weight": {
                    "dtype":"BF16", "shape":[32], "data_offsets":[7232,7296]
                },
                "mtp.layers.0.self_attn.k_norm.weight": {
                    "dtype":"BF16", "shape":[1], "data_offsets":[7296,7298]
                },
                "mtp.layers.0.self_attn.k_proj.weight": {
                    "dtype":"BF16", "shape":[1,32], "data_offsets":[7298,7362]
                },
                "mtp.layers.0.self_attn.o_proj.weight": {
                    "dtype":"BF16", "shape":[32,1], "data_offsets":[7362,7426]
                },
                "mtp.layers.0.self_attn.q_norm.weight": {
                    "dtype":"BF16", "shape":[1], "data_offsets":[7426,7428]
                },
                "mtp.layers.0.self_attn.q_proj.weight": {
                    "dtype":"BF16", "shape":[2,32], "data_offsets":[7428,7556]
                },
                "mtp.layers.0.self_attn.v_proj.weight": {
                    "dtype":"BF16", "shape":[1,32], "data_offsets":[7556,7620]
                },
                "mtp.norm.weight": {
                    "dtype":"BF16", "shape":[32], "data_offsets":[7620,7684]
                },
                "mtp.pre_fc_norm_embedding.weight": {
                    "dtype":"BF16", "shape":[32], "data_offsets":[7684,7748]
                },
                "mtp.pre_fc_norm_hidden.weight": {
                    "dtype":"BF16", "shape":[32], "data_offsets":[7748,7812]
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

    fn append_bf16_tensor(
        header: &mut serde_json::Map<String, Value>,
        payload: &mut Vec<u8>,
        name: impl Into<String>,
        shape: Vec<usize>,
    ) {
        let begin = payload.len();
        let elements = shape.iter().product::<usize>();
        let word = u16::try_from(header.len() + 1).unwrap().to_le_bytes();

        for _ in 0..elements {
            payload.extend_from_slice(&word);
        }

        header.insert(
            name.into(),
            json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    fn vision_fixture() -> (Value, Vec<u8>) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();

        for block in 0..TestArch::VISION_DEPTH {
            let prefix = format!("model.visual.blocks.{block}");

            for (suffix, shape) in [
                ("attn.proj.bias", vec![4]),
                ("attn.proj.weight", vec![4, 4]),
                ("attn.qkv.bias", vec![12]),
                ("attn.qkv.weight", vec![12, 4]),
                ("mlp.linear_fc1.bias", vec![6]),
                ("mlp.linear_fc1.weight", vec![6, 4]),
                ("mlp.linear_fc2.bias", vec![4]),
                ("mlp.linear_fc2.weight", vec![4, 6]),
                ("norm1.bias", vec![4]),
                ("norm1.weight", vec![4]),
                ("norm2.bias", vec![4]),
                ("norm2.weight", vec![4]),
            ] {
                append_bf16_tensor(
                    &mut header,
                    &mut payload,
                    format!("{prefix}.{suffix}"),
                    shape,
                );
            }
        }

        for (name, shape) in [
            ("model.visual.merger.linear_fc1.bias", vec![16]),
            ("model.visual.merger.linear_fc1.weight", vec![16, 16]),
            ("model.visual.merger.linear_fc2.bias", vec![4]),
            ("model.visual.merger.linear_fc2.weight", vec![4, 16]),
            ("model.visual.merger.norm.bias", vec![4]),
            ("model.visual.merger.norm.weight", vec![4]),
            ("model.visual.patch_embed.proj.bias", vec![4]),
            ("model.visual.patch_embed.proj.weight", vec![4, 3, 2, 2, 2]),
            ("model.visual.pos_embed.weight", vec![8, 4]),
        ] {
            append_bf16_tensor(&mut header, &mut payload, name, shape);
        }

        (Value::Object(header), payload)
    }

    fn assert_rejects_bf16_scale(
        label: &str,
        header: Value,
        mut payload: Vec<u8>,
        offset: usize,
        tensor_name: &str,
        bind: impl FnOnce(&SafeTensorFile) -> CheckpointResult<()>,
    ) {
        payload[offset..offset + 2].copy_from_slice(&0x7fc0u16.to_le_bytes());

        let path = fixture_path(label);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = bind(&file).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains(tensor_name), "{error}");
        assert!(
            error
                .to_string()
                .contains("must contain a finite positive BF16 scale"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_exact_text_endpoint_contract() {
        let path = fixture_path("valid");
        write_safetensors(&path, endpoint_header());
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name)).unwrap();

        assert_eq!(bindings.embedding.shape(), &[3, 4]);
        assert_eq!(bindings.embedding.word(0), Some(0x0100));
        assert_eq!(bindings.final_norm.shape(), &[4]);
        assert_eq!(bindings.lm_head.shape(), &[3, 4]);
        assert_eq!(bindings.lm_head.codes(), &(32u8..44).collect::<Vec<_>>());
        assert_eq!(bindings.lm_head_scale.shape(), &[3, 1]);

        fs::remove_file(path).unwrap();
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
    fn rejects_endpoint_dtype_mismatch() {
        let path = fixture_path("dtype");
        let mut header = endpoint_header();
        header["lm_head.weight"]["dtype"] = json!("U8");
        write_safetensors(&path, header);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name))
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("lm_head.weight"));
        assert!(error.to_string().contains("dtype `U8`, expected `F8_E4M3`"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_endpoint_shape_mismatch() {
        let path = fixture_path("shape");
        let mut header = endpoint_header();
        header["lm_head.weight_scale"]["shape"] = json!([3]);
        write_safetensors(&path, header);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name))
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("lm_head.weight_scale"));
        assert!(error.contains("shape [3], expected [3, 1]"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_exact_nvfp4_mlp_source_contract() {
        let path = fixture_path("nvfp4-mlp");
        let (header, payload) = nvfp4_mlp_fixture(55);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let gate_up =
            Nvfp4GateUpBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name)).unwrap();
        let down = Nvfp4DownBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name)).unwrap();
        let complete =
            Nvfp4MlpBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name)).unwrap();

        assert_eq!(gate_up.gate_weight.shape(), &[16, 16]);
        assert_eq!(gate_up.up_weight.shape(), &[16, 16]);
        assert_eq!(gate_up.gate_scale.shape(), &[16, 2]);
        assert_eq!(gate_up.up_scale.shape(), &[16, 2]);
        assert_eq!(gate_up.input_scale_divisor.to_bits(), 3.0f32.to_bits());
        assert_eq!(gate_up.weight_scale_divisor.to_bits(), 0.125f32.to_bits());
        assert_eq!(down.weight.shape(), &[32, 8]);
        assert_eq!(down.scale.shape(), &[32, 1]);
        assert_eq!(down.input_scale_divisor.to_bits(), 19.0f32.to_bits());
        assert_eq!(down.weight_scale_divisor.to_bits(), 3_376.0f32.to_bits());
        assert_eq!((gate_up.layer, down.layer), (55, 55));
        assert_eq!(complete.input_norm.shape(), &[32]);
        assert_eq!(complete.input_norm.bytes()[0], 0x70);
        assert_eq!(complete.next_norm.shape(), &[32]);
        assert_eq!(complete.next_norm.bytes()[0], 0x80);
        assert_eq!(complete.layer, 55);

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
    fn binds_exact_dense_fp8_mlp_source_contract() {
        let path = fixture_path("dense-fp8-mlp");
        let (header, payload) = dense_fp8_mlp_fixture(56);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let gate_up = DenseFp8GateUpBindings::bind_from::<Nvfp4Arch>(
            56,
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .unwrap();
        let down =
            DenseFp8DownBindings::bind_from::<Nvfp4Arch>(56, |name| file.tensor(name)).unwrap();
        let gate = file
            .tensor("model.language_model.layers.56.mlp.gate_proj.weight")
            .unwrap();

        assert_eq!(gate_up.weight_e4m3.len(), 1_024);
        assert_eq!(gate_up.weight_e4m3[0], 0x10);
        assert_eq!(gate_up.weight_e4m3[512], 0x20);
        assert_eq!(gate_up.weight_e4m3.as_ptr(), gate.bytes.as_ptr());
        assert_eq!(gate_up.scale_bf16.len(), 64);
        assert_eq!(gate_up.scale_bf16[0], 0x30);
        assert_eq!(gate_up.scale_bf16[32], 0x40);
        assert_eq!((gate_up.rows, gate_up.columns, gate_up.layer), (32, 32, 56));
        assert_eq!(down.weight.shape(), &[32, 16]);
        assert_eq!(down.weight.codes()[0], 0x50);
        assert_eq!(down.scale.shape(), &[32, 1]);
        assert_eq!(down.scale.bytes()[0], 0x60);
        assert_eq!(down.layer, 56);

        let complete = DenseFp8MlpBindings::bind_from::<Nvfp4Arch>(
            56,
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .unwrap();
        assert_eq!(complete.input_norm.shape(), &[32]);
        assert_eq!(complete.input_norm.bytes()[0], 0x70);
        assert_eq!(complete.next_norm.shape(), &[32]);
        assert_eq!(complete.next_norm.bytes()[0], 0x80);
        assert_eq!(complete.layer, 56);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn dense_fp8_next_norm_route_is_exact() {
        let cases = [
            (
                56,
                Ok("model.language_model.layers.57.input_layernorm.weight"),
            ),
            (
                62,
                Ok("model.language_model.layers.63.input_layernorm.weight"),
            ),
            (63, Ok("model.language_model.norm.weight")),
            (55, Err(())),
            (64, Err(())),
        ];

        for (layer, expected) in cases {
            let actual = dense_fp8_next_norm_name::<Nvfp4Arch>(layer);
            match expected {
                Ok(name) => assert_eq!(actual.unwrap(), name),
                Err(()) => assert!(actual.is_err(), "layer={layer}"),
            }
        }
    }

    #[test]
    fn binds_exact_full_attention_source_contract() {
        let path = fixture_path("full-attention");
        let (header, payload) = full_attention_fixture(63);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let qkv =
            FullAttentionQkvBindings::bind_from::<Nvfp4Arch>(63, |name| file.tensor(name)).unwrap();
        let post = FullAttentionPostBindings::bind_from::<Nvfp4Arch>(63, |name| file.tensor(name))
            .unwrap();

        assert_eq!(qkv.query_gate_weight.shape(), &[2, 32]);
        assert_eq!(qkv.query_gate_weight.codes()[0], 0x10);
        assert_eq!(qkv.key_weight.shape(), &[1, 32]);
        assert_eq!(qkv.key_weight.codes()[0], 0x20);
        assert_eq!(qkv.value_weight.shape(), &[1, 32]);
        assert_eq!(qkv.value_weight.codes()[0], 0x30);
        assert_eq!(qkv.query_gate_scale.shape(), &[2, 1]);
        assert_eq!(qkv.query_gate_scale.bytes()[0], 0x40);
        assert_eq!(qkv.key_scale.bytes()[0], 0x50);
        assert_eq!(qkv.value_scale.bytes()[0], 0x60);
        assert_eq!(qkv.layer, 63);
        assert_eq!(post.output_weight.shape(), &[32, 1]);
        assert_eq!(post.output_weight.codes()[0], 0x70);
        assert_eq!(post.output_scale.shape(), &[32, 1]);
        assert_eq!(post.input_norm.shape(), &[32]);
        assert_eq!(post.post_attention_norm.shape(), &[32]);
        assert_eq!(post.query_norm.shape(), &[1]);
        assert_eq!(post.key_norm.shape(), &[1]);
        assert_eq!(post.key_cache_scale_bf16, 0x3f80);
        assert_eq!(post.value_cache_scale_bf16, 0x3f00);
        assert_eq!(post.layer, 63);

        fs::remove_file(path).unwrap();
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
    fn binds_exact_gdn_source_contract() {
        let path = fixture_path("gdn");
        let (header, payload) = gdn_fixture(62);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = GdnBindings::bind_from::<Nvfp4Arch>(
            62,
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .unwrap();

        assert_eq!(bindings.input_weight_e4m3.len(), 128);
        assert_eq!(bindings.input_weight_e4m3[0], 0x10);
        assert_eq!(bindings.input_weight_e4m3[96], 0x20);
        assert_eq!(bindings.input_scale_bf16.len(), 8);
        assert_eq!(bindings.input_scale_bf16[0], 0x80);
        assert_eq!(bindings.input_scale_bf16[6], 0x80);
        assert_eq!((bindings.input_rows, bindings.input_columns), (4, 32));
        assert_eq!(bindings.a_control_weight.shape(), &[1, 32]);
        assert_eq!(bindings.b_control_weight.shape(), &[1, 32]);
        assert_eq!(bindings.convolution_weight.shape(), &[3, 1, 4]);
        assert_eq!(bindings.a_log.shape(), &[1]);
        assert_eq!(bindings.dt_bias.shape(), &[1]);
        assert_eq!(bindings.norm.shape(), &[1]);
        assert_eq!(bindings.output_weight.shape(), &[32, 1]);
        assert_eq!(bindings.output_weight.codes()[0], 0x30);
        assert_eq!(bindings.output_scale.shape(), &[32, 1]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.layer, 62);

        fs::remove_file(path).unwrap();
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

    #[test]
    fn rejects_gdn_convolution_shape_mismatch() {
        let path = fixture_path("gdn-convolution-shape");
        let (mut header, payload) = gdn_fixture(62);
        header["model.language_model.layers.62.linear_attn.conv1d.weight"]["shape"] = json!([3, 4]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = GdnBindings::bind_from::<Nvfp4Arch>(
            62,
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("conv1d.weight"));
        assert!(error.to_string().contains("expected [3, 1, 4]"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_and_materializes_exact_mtp_source_contract() {
        let path = fixture_path("mtp");
        let (header, payload) = mtp_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = MtpBindings::bind_from::<Nvfp4Arch>(
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .unwrap();

        assert_eq!(bindings.input_projection.shape(), &[32, 64]);
        assert_eq!(bindings.embedding_norm.shape(), &[32]);
        assert_eq!(bindings.hidden_norm.shape(), &[32]);
        assert_eq!(bindings.query_gate_weight.shape(), &[2, 32]);
        assert_eq!(bindings.key_weight.shape(), &[1, 32]);
        assert_eq!(bindings.value_weight.shape(), &[1, 32]);
        assert_eq!(bindings.attention_output_weight.shape(), &[32, 1]);
        assert_eq!(bindings.query_norm.shape(), &[1]);
        assert_eq!(bindings.key_norm.shape(), &[1]);
        assert_eq!(bindings.input_norm.shape(), &[32]);
        assert_eq!(bindings.gate_up_weight_bf16.len(), 2_048);
        assert_eq!(bindings.gate_up_weight_bf16[0], 0x20);
        assert_eq!(bindings.gate_up_weight_bf16[1_024], 0x30);
        assert_eq!((bindings.gate_up_rows, bindings.gate_up_columns), (32, 32));
        assert_eq!(bindings.down_weight.shape(), &[32, 16]);
        assert_eq!(bindings.post_attention_norm.shape(), &[32]);
        assert_eq!(bindings.final_norm.shape(), &[32]);

        let materialized = bindings.materialize_qkv().unwrap();
        let query_bytes = bindings.query_gate_weight.bytes();
        let key_bytes = bindings.key_weight.bytes();
        let value_bytes = bindings.value_weight.bytes();
        let query_end = query_bytes.len();
        let key_end = query_end + key_bytes.len();

        assert_eq!(&materialized.weight_bf16[..query_end], query_bytes);
        assert_eq!(&materialized.weight_bf16[query_end..key_end], key_bytes);
        assert_eq!(&materialized.weight_bf16[key_end..], value_bytes);
        assert_eq!((materialized.rows, materialized.columns), (4, 32));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_mtp_input_projection_shape_mismatch() {
        let path = fixture_path("mtp-input-shape");
        let (mut header, payload) = mtp_fixture();
        header["mtp.fc.weight"]["shape"] = json!([64, 32]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let error = MtpBindings::bind_from::<Nvfp4Arch>(
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("mtp.fc.weight"));
        assert!(error.to_string().contains("expected [32, 64]"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_complete_vision_source_contract() {
        let path = fixture_path("vision");
        let (header, payload) = vision_fixture();
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = VisionBindings::bind_from::<TestArch>(|name| file.tensor(name)).unwrap();

        assert_eq!(bindings.blocks.len(), 2);

        for (block_index, block) in bindings.blocks.iter().enumerate() {
            assert_eq!(block.block, block_index);
            assert_eq!(block.attention_projection_bias.shape(), &[4]);
            assert_eq!(block.attention_projection_weight.shape(), &[4, 4]);
            assert_eq!(block.qkv_bias.shape(), &[12]);
            assert_eq!(block.qkv_weight.shape(), &[12, 4]);
            assert_eq!(block.mlp_fc1_bias.shape(), &[6]);
            assert_eq!(block.mlp_fc1_weight.shape(), &[6, 4]);
            assert_eq!(block.mlp_fc2_bias.shape(), &[4]);
            assert_eq!(block.mlp_fc2_weight.shape(), &[4, 6]);
            assert_eq!(block.norm1_bias.shape(), &[4]);
            assert_eq!(block.norm1_weight.shape(), &[4]);
            assert_eq!(block.norm2_bias.shape(), &[4]);
            assert_eq!(block.norm2_weight.shape(), &[4]);
        }

        assert_eq!(
            bindings.blocks[0].attention_projection_bias.word(0),
            Some(1)
        );
        assert_eq!(bindings.blocks[1].norm2_weight.word(0), Some(24));
        assert_eq!(bindings.patch_embedding_bias.shape(), &[4]);
        assert_eq!(bindings.patch_embedding_bias.word(0), Some(31));
        assert_eq!(bindings.patch_embedding_weight.shape(), &[4, 3, 2, 2, 2]);
        assert_eq!(bindings.patch_embedding_weight.word(0), Some(32));
        assert_eq!(bindings.position_embedding.shape(), &[8, 4]);
        assert_eq!(bindings.position_embedding.word(0), Some(33));
        assert_eq!(bindings.merger_norm_bias.shape(), &[4]);
        assert_eq!(bindings.merger_norm_weight.shape(), &[4]);
        assert_eq!(bindings.merger_fc1_bias.shape(), &[16]);
        assert_eq!(bindings.merger_fc1_weight.shape(), &[16, 16]);
        assert_eq!(bindings.merger_fc2_bias.shape(), &[4]);
        assert_eq!(bindings.merger_fc2_weight.shape(), &[4, 16]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_vision_tensor_dtype_and_shape_mismatches() {
        for (label, tensor_name, expected, dtype_mismatch) in [
            (
                "vision-dtype",
                "model.visual.pos_embed.weight",
                "dtype `F32`, expected `BF16`",
                true,
            ),
            (
                "vision-shape",
                "model.visual.blocks.1.attn.qkv.weight",
                "shape [6, 8], expected [12, 4]",
                false,
            ),
        ] {
            let path = fixture_path(label);
            let (mut header, payload) = vision_fixture();

            if dtype_mismatch {
                header[tensor_name]["dtype"] = json!("F32");
                header[tensor_name]["shape"] = json!([4, 4]);
            } else {
                header[tensor_name]["shape"] = json!([6, 8]);
            }

            write_safetensors_payload(&path, header, &payload);
            let file = SafeTensorFile::open(&path).unwrap();
            let error = VisionBindings::bind_from::<TestArch>(|name| file.tensor(name))
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Tensor);
            assert!(error.to_string().contains(tensor_name), "{error}");
            assert!(error.to_string().contains(expected), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_fp8_bf16_scale_planes() {
        let endpoint_payload = (0u8..50).collect::<Vec<_>>();
        assert_rejects_bf16_scale(
            "endpoint-scale",
            endpoint_header(),
            endpoint_payload,
            44,
            "lm_head.weight_scale",
            |file| {
                TextEndpointBindings::bind_from::<TestArch>(|name| file.tensor(name)).map(|_| ())
            },
        );

        let (header, payload) = dense_fp8_mlp_fixture(56);
        assert_rejects_bf16_scale(
            "dense-gate-scale",
            header,
            payload,
            1_024,
            "gate_proj.weight_scale",
            |file| {
                DenseFp8GateUpBindings::bind_from::<Nvfp4Arch>(
                    56,
                    |name| file.tensor(name),
                    |first, second, role| file.adjacent_tensor_bytes(first, second, role),
                )
                .map(|_| ())
            },
        );

        let (header, payload) = dense_fp8_mlp_fixture(56);
        assert_rejects_bf16_scale(
            "dense-down-scale",
            header,
            payload,
            1_600,
            "down_proj.weight_scale",
            |file| {
                DenseFp8DownBindings::bind_from::<Nvfp4Arch>(56, |name| file.tensor(name))
                    .map(|_| ())
            },
        );

        let (header, payload) = full_attention_fixture(63);
        assert_rejects_bf16_scale(
            "attention-qkv-scale",
            header,
            payload,
            128,
            "q_proj.weight_scale",
            |file| {
                FullAttentionQkvBindings::bind_from::<Nvfp4Arch>(63, |name| file.tensor(name))
                    .map(|_| ())
            },
        );

        let (header, payload) = full_attention_fixture(63);
        assert_rejects_bf16_scale(
            "attention-output-scale",
            header,
            payload,
            168,
            "o_proj.weight_scale",
            |file| {
                FullAttentionPostBindings::bind_from::<Nvfp4Arch>(63, |name| file.tensor(name))
                    .map(|_| ())
            },
        );

        let (header, payload) = gdn_fixture(62);
        assert_rejects_bf16_scale(
            "gdn-input-scale",
            header,
            payload,
            160,
            "in_proj_qkv.weight_scale",
            |file| {
                GdnBindings::bind_from::<Nvfp4Arch>(
                    62,
                    |name| file.tensor(name),
                    |first, second, role| file.adjacent_tensor_bytes(first, second, role),
                )
                .map(|_| ())
            },
        );

        let (header, payload) = gdn_fixture(62);
        assert_rejects_bf16_scale(
            "gdn-output-scale",
            header,
            payload,
            168,
            "out_proj.weight_scale",
            |file| {
                GdnBindings::bind_from::<Nvfp4Arch>(
                    62,
                    |name| file.tensor(name),
                    |first, second, role| file.adjacent_tensor_bytes(first, second, role),
                )
                .map(|_| ())
            },
        );
    }

    #[test]
    fn nvfp4_layer_route_is_exact() {
        for (layer, admitted) in [(0, true), (55, true), (56, false), (63, false), (64, false)] {
            assert_eq!(
                require_nvfp4_mlp_layer(layer, Nvfp4Arch::LAYERS).is_ok(),
                admitted,
                "layer {layer}"
            );
        }
    }

    #[test]
    fn dense_fp8_layer_route_is_exact() {
        for (layer, admitted) in [(0, false), (55, false), (56, true), (63, true), (64, false)] {
            assert_eq!(
                require_dense_fp8_mlp_layer::<Nvfp4Arch>(layer).is_ok(),
                admitted,
                "layer {layer}"
            );
        }
    }

    #[test]
    fn full_attention_layer_route_is_exact() {
        for (layer, admitted) in [
            (0, false),
            (2, false),
            (3, true),
            (4, false),
            (59, true),
            (63, true),
            (64, false),
        ] {
            assert_eq!(
                require_full_attention_layer(
                    layer,
                    Nvfp4Arch::LAYERS,
                    Nvfp4Arch::FULL_ATTENTION_INTERVAL,
                )
                .is_ok(),
                admitted,
                "layer {layer}"
            );
        }
    }

    #[test]
    fn gdn_layer_route_is_exact() {
        for (layer, admitted) in [
            (0, true),
            (2, true),
            (3, false),
            (4, true),
            (59, false),
            (62, true),
            (63, false),
            (64, false),
        ] {
            assert_eq!(
                require_gdn_layer::<Nvfp4Arch>(layer).is_ok(),
                admitted,
                "layer {layer}"
            );
        }
    }

    #[test]
    fn mtp_source_route_is_exact() {
        for (layers, dedicated_embeddings, admitted) in [
            (0, false, false),
            (1, false, true),
            (2, false, false),
            (1, true, false),
        ] {
            assert_eq!(
                require_mtp_contract(layers, dedicated_embeddings).is_ok(),
                admitted,
                "layers={layers}, dedicated_embeddings={dedicated_embeddings}"
            );
        }
    }

    #[test]
    fn rejects_full_attention_projection_shape_mismatch() {
        let path = fixture_path("full-attention-shape");
        let (mut header, payload) = full_attention_fixture(63);
        header["model.language_model.layers.63.self_attn.q_proj.weight"]["shape"] = json!([1, 64]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = FullAttentionQkvBindings::bind_from::<Nvfp4Arch>(63, |name| file.tensor(name))
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(
            error
                .to_string()
                .contains("shape [1, 64], expected [2, 32]")
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_bf16_cache_scales() {
        for bits in [0x0000u16, 0x8000, 0x7f80, 0x7fc0] {
            let bytes = bits.to_le_bytes();
            let error = positive_bf16(TensorView {
                name: "cache_scale",
                dtype: DType::Bf16,
                shape: &[1],
                bytes: &bytes,
                data_range: 0..2,
            })
            .err()
            .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(
                error
                    .to_string()
                    .contains("must contain a finite positive BF16 scale"),
                "{error}"
            );
        }
    }

    #[test]
    fn rejects_dense_fp8_weight_dtype_mismatch() {
        let path = fixture_path("dense-fp8-dtype");
        let (mut header, payload) = dense_fp8_mlp_fixture(56);
        header["model.language_model.layers.56.mlp.gate_proj.weight"]["dtype"] = json!("U8");
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = DenseFp8GateUpBindings::bind_from::<Nvfp4Arch>(
            56,
            |name| file.tensor(name),
            |first, second, role| file.adjacent_tensor_bytes(first, second, role),
        )
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(error.to_string().contains("dtype `U8`, expected `F8_E4M3`"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_nvfp4_gate_up_with_different_divisors() {
        let path = fixture_path("nvfp4-divisor");
        let (header, mut payload) = nvfp4_mlp_fixture(55);
        payload[8..12].copy_from_slice(&4.0f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = Nvfp4GateUpBindings::bind_from::<Nvfp4Arch>(55, |name| file.tensor(name))
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("input_global_scale words differ")
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_nvfp4_scale_codes() {
        assert!(validate_nvfp4_scales(55, "gate", &[0x7e]).is_ok());

        for code in [0x7f, 0x80, 0xfe, 0xff] {
            let error = validate_nvfp4_scales(55, "gate", &[code]).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(
                error.to_string().contains("negative or NaN E4M3FN code"),
                "code {code:#04x}: {error}"
            );
        }
    }

    #[test]
    fn rejects_nonadjacent_fused_source_planes() {
        let first = TensorView {
            name: "gate",
            dtype: DType::U8,
            shape: &[1],
            bytes: &[0],
            data_range: 0..1,
        };
        let second = TensorView {
            name: "up",
            dtype: DType::U8,
            shape: &[1],
            bytes: &[0],
            data_range: 2..3,
        };

        let error = require_adjacent(55, "packed gate/up weights", &first, &second)
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("not source-adjacent"));
    }
}

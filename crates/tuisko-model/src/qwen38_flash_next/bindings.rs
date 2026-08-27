//! Exact non-MTP source bindings for Qwen3.8-Flash-Next.
//!
//! Routed experts bind split NVFP4 planes. The engram keeps its FP8 table and plain BF16
//! multiplier outside the ModelOpt reciprocal-divisor convention.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::modelopt_codec::ModelOptNvfp4LinearBindings;
use crate::common::naming::{EMBEDDING, LM_HEAD, layer_prefix};
use crate::common::routes::{
    require_full_attention_layer, require_gdn_layer_route, require_same_rank_zero_f32,
};
use crate::qwen38_flash_next::engram::{
    Qwen38FlashNextEngramConstantBindings, engram_table_prefix,
};
use crate::{
    Arch, Bf16View, CheckpointError, CheckpointResult, Fp8E4M3View, Qwen38FlashNext, TensorView,
};

type F = Qwen38FlashNext;

/// Root of the model-level hyper-connection mixer, this target's only final normalization.
pub(crate) const HYPER_CONNECTION_MIXER: &str = "model.language_model.hyper_connection_mixer";

/// Named geometry used by production bindings and shrunken fixtures.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextGeometry {
    pub(crate) layers: usize,
    pub(crate) full_attention_interval: usize,
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    pub(crate) head_dim: usize,
    pub(crate) attention_query_rows: usize,
    pub(crate) attention_kv_rows: usize,
    pub(crate) attention_output_columns: usize,
    pub(crate) indexer_rows: usize,
    pub(crate) indexer_head_dim: usize,
    pub(crate) gdn_qkv_rows: usize,
    pub(crate) gdn_value_rows: usize,
    pub(crate) gdn_control_rows: usize,
    pub(crate) gdn_conv_kernel: usize,
    pub(crate) gdn_head_dim: usize,
    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,
    pub(crate) hc_width: usize,
    pub(crate) expert_count: usize,
    pub(crate) expert_intermediate: usize,
    pub(crate) shared_intermediate: usize,
    pub(crate) ple_layer: usize,
    pub(crate) ple_embed_dim: usize,
    pub(crate) ple_conv_kernel: usize,
    pub(crate) ngram_shards: usize,
    pub(crate) ngram_shard_rows: usize,
    pub(crate) ngram_head_dim: usize,
}

impl Qwen38FlashNextGeometry {
    /// The pinned target's own geometry.
    pub(crate) const fn target() -> Self {
        Self {
            layers: F::LAYERS,
            full_attention_interval: F::FULL_ATTENTION_INTERVAL,
            hidden: F::HIDDEN,
            vocab: F::VOCAB,
            head_dim: F::HEAD_DIM,
            attention_query_rows: F::ATTENTION_QUERY_ROWS,
            attention_kv_rows: F::ATTENTION_KV_ROWS,
            attention_output_columns: F::ATTENTION_OUTPUT_COLUMNS,
            indexer_rows: F::INDEXER_ROWS,
            indexer_head_dim: F::INDEXER_HEAD_DIM,
            gdn_qkv_rows: F::GDN_QKV_ROWS,
            gdn_value_rows: F::GDN_VALUE_ROWS,
            gdn_control_rows: F::GDN_CONTROL_ROWS,
            gdn_conv_kernel: F::LINEAR_CONV_KERNEL_DIM,
            gdn_head_dim: F::LINEAR_HEAD_DIM,
            hc_count: F::HC_COUNT,
            hc_lowrank: F::HC_LOWRANK,
            hc_width: F::HC_WIDTH,
            expert_count: F::NUM_EXPERTS,
            expert_intermediate: F::INTERMEDIATE,
            shared_intermediate: F::SHARED_EXPERT_INTERMEDIATE,
            ple_layer: F::PLE_LAYER,
            ple_embed_dim: F::PLE_EMBED_DIM,
            ple_conv_kernel: F::PLE_CONV_KERNEL,
            ngram_shards: F::NGRAM_SHARDS,
            ngram_shard_rows: F::NGRAM_SHARD_ROWS,
            ngram_head_dim: F::NGRAM_HEAD_DIM,
        }
    }
}

/// Refuses a layer outside the target's decoder stack.
pub(crate) fn require_qwen38_flash_next_layer(
    layer: usize,
    layers: usize,
    role: &str,
) -> CheckpointResult<()> {
    if layer >= layers {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted Qwen3.8-Flash-Next {role} source contract"
        )));
    }

    Ok(())
}

/// One gated-residual source family; collapsing mixers omit `block_inject_weight`.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextHyperConnectionBindings<'a> {
    /// Grouped RMSNorm weights over the widened stream `[hc_width]`.
    pub hc_norm: Bf16View<'a, 1>,
    /// Read-gate down projection `[hc_lowrank, hc_width]`.
    pub input_mix_down: Bf16View<'a, 2>,
    /// Read-gate up projection `[hc_width, hc_lowrank]`.
    pub input_mix_up: Bf16View<'a, 2>,
    /// Per-branch scalar write gate `[hc_count, hc_width]`, absent on a collapsing mixer.
    pub block_inject: Option<Bf16View<'a, 2>>,
}

impl<'a> Qwen38FlashNextHyperConnectionBindings<'a> {
    fn bind_from(
        prefix: &str,
        geometry: &Qwen38FlashNextGeometry,
        combines: bool,
        tensor: &mut impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let width = geometry.hc_width as u64;
        let lowrank = geometry.hc_lowrank as u64;
        let hc_norm = Bf16View::bind(tensor(&format!("{prefix}.hc_norm.weight"))?, [width])?;
        let input_mix_down = Bf16View::bind(
            tensor(&format!("{prefix}.input_mix_weight_down.weight"))?,
            [lowrank, width],
        )?;
        let input_mix_up = Bf16View::bind(
            tensor(&format!("{prefix}.input_mix_weight_up.weight"))?,
            [width, lowrank],
        )?;
        let block_inject = if combines {
            Some(Bf16View::bind(
                tensor(&format!("{prefix}.block_inject_weight.weight"))?,
                [geometry.hc_count as u64, width],
            )?)
        } else {
            None
        };

        Ok(Self {
            hc_norm,
            input_mix_down,
            input_mix_up,
            block_inject,
        })
    }
}

/// The two writing hyper-connections owned by one decoder layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextLayerHyperConnections<'a> {
    /// Gated residual read before the attention or GDN mixer.
    pub attention: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Gated residual read before the MoE block.
    pub mlp: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen38FlashNextLayerHyperConnections<'a> {
    /// Binds one decoder layer's hyper-connection pair.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(layer, &Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    pub(crate) fn bind_from(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_qwen38_flash_next_layer(layer, geometry.layers, "hyper-connection")?;

        let prefix = layer_prefix(layer);

        Ok(Self {
            attention: Qwen38FlashNextHyperConnectionBindings::bind_from(
                &format!("{prefix}.attn_hyper_connection"),
                geometry,
                true,
                &mut tensor,
            )?,
            mlp: Qwen38FlashNextHyperConnectionBindings::bind_from(
                &format!("{prefix}.mlp_hyper_connection"),
                geometry,
                true,
                &mut tensor,
            )?,
            layer,
        })
    }
}

/// Complete BF16 source family for one gated DeltaNet layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextGdnBindings<'a> {
    /// Fused query, key, and value projection `[gdn_qkv_rows, hidden]`.
    pub qkv_weight: Bf16View<'a, 2>,
    /// Output-gate projection `[gdn_value_rows, hidden]`.
    pub z_weight: Bf16View<'a, 2>,
    /// Per-value-head A-control projection `[gdn_control_rows, hidden]`.
    pub a_control_weight: Bf16View<'a, 2>,
    /// Per-value-head B-control projection `[gdn_control_rows, hidden]`.
    pub b_control_weight: Bf16View<'a, 2>,
    /// Recurrent-state output projection `[hidden, gdn_value_rows]`.
    pub output_weight: Bf16View<'a, 2>,
    /// Depthwise causal convolution over the QKV channels only `[gdn_qkv_rows, 1, kernel]`.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters `[gdn_control_rows]`.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias `[gdn_control_rows]`.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights `[gdn_head_dim]`.
    pub norm: Bf16View<'a, 1>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen38FlashNextGdnBindings<'a> {
    /// Binds one exact Flash-Next GDN source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(layer, &Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    pub(crate) fn bind_from(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_gdn_layer_route(layer, geometry.layers, geometry.full_attention_interval)?;

        let prefix = format!("{}.linear_attn", layer_prefix(layer));
        let hidden = geometry.hidden as u64;
        let control_rows = geometry.gdn_control_rows as u64;
        let qkv_rows = geometry.gdn_qkv_rows as u64;
        let value_rows = geometry.gdn_value_rows as u64;
        let mut projection = |name: &str, rows: u64, columns: u64| {
            Bf16View::bind(tensor(&format!("{prefix}.{name}.weight"))?, [rows, columns])
        };

        Ok(Self {
            qkv_weight: projection("in_proj_qkv", qkv_rows, hidden)?,
            z_weight: projection("in_proj_z", value_rows, hidden)?,
            a_control_weight: projection("in_proj_a", control_rows, hidden)?,
            b_control_weight: projection("in_proj_b", control_rows, hidden)?,
            output_weight: projection("out_proj", hidden, value_rows)?,
            convolution_weight: Bf16View::bind(
                tensor(&format!("{prefix}.conv1d.weight"))?,
                [qkv_rows, 1, geometry.gdn_conv_kernel as u64],
            )?,
            a_log: Bf16View::bind(tensor(&format!("{prefix}.A_log"))?, [control_rows])?,
            dt_bias: Bf16View::bind(tensor(&format!("{prefix}.dt_bias"))?, [control_rows])?,
            norm: Bf16View::bind(
                tensor(&format!("{prefix}.norm.weight"))?,
                [geometry.gdn_head_dim as u64],
            )?,
            layer,
        })
    }
}

/// The block-selection indexer owned by one sparse-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextIndexerBindings<'a> {
    /// Fused indexer query and key projection `[indexer_rows, hidden]`.
    pub qk_weight: Bf16View<'a, 2>,
    /// Indexer query RMSNorm weights `[indexer_head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Indexer key RMSNorm weights `[indexer_head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
}

/// Complete BF16 source family for one `qwen_sparse_attention` layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextSparseAttentionBindings<'a> {
    /// Query rows followed by output-gate rows `[attention_query_rows, hidden]`.
    pub query_gate_weight: Bf16View<'a, 2>,
    /// Key projection `[attention_kv_rows, hidden]`.
    pub key_weight: Bf16View<'a, 2>,
    /// Value projection `[attention_kv_rows, hidden]`.
    pub value_weight: Bf16View<'a, 2>,
    /// Attention output projection `[hidden, attention_output_columns]`.
    pub output_weight: Bf16View<'a, 2>,
    /// Per-head query RMSNorm weights `[head_dim]`.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights `[head_dim]`.
    pub key_norm: Bf16View<'a, 1>,
    /// Block-selection indexer sharing this layer's mixed block input.
    pub indexer: Qwen38FlashNextIndexerBindings<'a>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen38FlashNextSparseAttentionBindings<'a> {
    /// Binds one exact Flash-Next sparse-attention source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(layer, &Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    pub(crate) fn bind_from(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_full_attention_layer(layer, geometry.layers, geometry.full_attention_interval)?;

        let prefix = format!("{}.self_attn", layer_prefix(layer));
        let hidden = geometry.hidden as u64;
        let head_dim = geometry.head_dim as u64;
        let kv_rows = geometry.attention_kv_rows as u64;
        let indexer_head_dim = geometry.indexer_head_dim as u64;

        Ok(Self {
            query_gate_weight: Bf16View::bind(
                tensor(&format!("{prefix}.q_proj.weight"))?,
                [geometry.attention_query_rows as u64, hidden],
            )?,
            key_weight: Bf16View::bind(
                tensor(&format!("{prefix}.k_proj.weight"))?,
                [kv_rows, hidden],
            )?,
            value_weight: Bf16View::bind(
                tensor(&format!("{prefix}.v_proj.weight"))?,
                [kv_rows, hidden],
            )?,
            output_weight: Bf16View::bind(
                tensor(&format!("{prefix}.o_proj.weight"))?,
                [hidden, geometry.attention_output_columns as u64],
            )?,
            query_norm: Bf16View::bind(tensor(&format!("{prefix}.q_norm.weight"))?, [head_dim])?,
            key_norm: Bf16View::bind(tensor(&format!("{prefix}.k_norm.weight"))?, [head_dim])?,
            indexer: Qwen38FlashNextIndexerBindings {
                qk_weight: Bf16View::bind(
                    tensor(&format!("{prefix}.indexer.index_qk_proj.weight"))?,
                    [geometry.indexer_rows as u64, hidden],
                )?,
                query_norm: Bf16View::bind(
                    tensor(&format!("{prefix}.indexer.q_layernorm.weight"))?,
                    [indexer_head_dim],
                )?,
                key_norm: Bf16View::bind(
                    tensor(&format!("{prefix}.indexer.k_layernorm.weight"))?,
                    [indexer_head_dim],
                )?,
            },
            layer,
        })
    }
}

/// Exact ModelOpt NVFP4 planes for one split routed expert.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextExpertBindings<'a> {
    /// Gate projection `[expert_intermediate, hidden]`.
    pub gate: ModelOptNvfp4LinearBindings<'a>,
    /// Up projection `[expert_intermediate, hidden]`.
    pub up: ModelOptNvfp4LinearBindings<'a>,
    /// Down projection `[hidden, expert_intermediate]`.
    pub down: ModelOptNvfp4LinearBindings<'a>,
    /// Numeric expert index within its layer's pool.
    pub expert: usize,
}

/// The always-active BF16 shared expert and its scalar gate.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextSharedExpertBindings<'a> {
    /// Scalar routing gate `[1, hidden]`.
    pub gate_weight: Bf16View<'a, 2>,
    /// Gate projection `[shared_intermediate, hidden]`.
    pub gate_proj_weight: Bf16View<'a, 2>,
    /// Up projection `[shared_intermediate, hidden]`.
    pub up_proj_weight: Bf16View<'a, 2>,
    /// Down projection `[hidden, shared_intermediate]`.
    pub down_proj_weight: Bf16View<'a, 2>,
}

/// Complete MoE source family with BF16 shared and NVFP4 routed experts.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextMoeBindings<'a> {
    /// Full 512-way router weights `[expert_count, hidden]`.
    pub router_weight: Bf16View<'a, 2>,
    /// Shared expert and its gate.
    pub shared_expert: Qwen38FlashNextSharedExpertBindings<'a>,
    /// Routed experts in numeric expert order.
    pub experts: Vec<Qwen38FlashNextExpertBindings<'a>>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen38FlashNextMoeBindings<'a> {
    /// Binds one exact Flash-Next MoE source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(layer, &Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    pub(crate) fn bind_from(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        require_qwen38_flash_next_layer(layer, geometry.layers, "MoE")?;

        let prefix = format!("{}.mlp", layer_prefix(layer));
        let hidden = geometry.hidden as u64;
        let shared = geometry.shared_intermediate as u64;
        let router_weight = Bf16View::bind(
            tensor(&format!("{prefix}.gate.weight"))?,
            [geometry.expert_count as u64, hidden],
        )?;
        let shared_expert = Qwen38FlashNextSharedExpertBindings {
            gate_weight: Bf16View::bind(
                tensor(&format!("{prefix}.shared_expert_gate.weight"))?,
                [1, hidden],
            )?,
            gate_proj_weight: Bf16View::bind(
                tensor(&format!("{prefix}.shared_expert.gate_proj.weight"))?,
                [shared, hidden],
            )?,
            up_proj_weight: Bf16View::bind(
                tensor(&format!("{prefix}.shared_expert.up_proj.weight"))?,
                [shared, hidden],
            )?,
            down_proj_weight: Bf16View::bind(
                tensor(&format!("{prefix}.shared_expert.down_proj.weight"))?,
                [hidden, shared],
            )?,
        };
        let mut experts = Vec::new();

        experts
            .try_reserve_exact(geometry.expert_count)
            .map_err(|_| {
                CheckpointError::source_binding(format!(
                    "layer-{layer} cannot reserve {} Flash-Next routed expert bindings",
                    geometry.expert_count
                ))
            })?;

        for expert in 0..geometry.expert_count {
            experts.push(bind_qwen38_flash_next_expert(
                &format!("{prefix}.experts.{expert}"),
                expert,
                layer,
                geometry,
                |name| tensor(name),
            )?);
        }

        Ok(Self {
            router_weight,
            shared_expert,
            experts,
            layer,
        })
    }
}

fn bind_qwen38_flash_next_expert<'a>(
    prefix: &str,
    expert: usize,
    layer: usize,
    geometry: &Qwen38FlashNextGeometry,
    mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
) -> CheckpointResult<Qwen38FlashNextExpertBindings<'a>> {
    let hidden = geometry.hidden;
    let intermediate = geometry.expert_intermediate;
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

    // Gate and up share one runtime plane and must carry identical scalars.
    require_same_rank_zero_f32(
        layer,
        &format!("expert-{expert} gate/up input_scale"),
        &gate.input_scale,
        &up.input_scale,
    )?;
    require_same_rank_zero_f32(
        layer,
        &format!("expert-{expert} gate/up weight_scale_2"),
        &gate.weight_scale_2,
        &up.weight_scale_2,
    )?;

    Ok(Qwen38FlashNextExpertBindings {
        gate,
        up,
        down,
        expert,
    })
}

/// Complete engram (PLE) source family for the target's single PLE layer.
#[derive(Clone, Debug)]
pub struct Qwen38FlashNextEngramBindings<'a> {
    /// Key projection from the embedding into the widened stream `[hc_width, ple_embed_dim]`.
    pub key_proj_weight: Bf16View<'a, 2>,
    /// Value projection `[ple_embed_dim, ple_embed_dim]`.
    pub value_proj_weight: Bf16View<'a, 2>,
    /// Grouped RMSNorm weights over the projected key `[hc_width]`.
    pub norm_key: Bf16View<'a, 1>,
    /// Grouped RMSNorm weights over the residual copy used for the gate `[hc_width]`.
    pub norm_query: Bf16View<'a, 1>,
    /// Grouped RMSNorm weights before the short convolution `[hc_width]`.
    pub norm_conv: Bf16View<'a, 1>,
    /// Dilated depthwise short convolution `[hc_width, 1, ple_conv_kernel]`.
    pub convolution_weight: Bf16View<'a, 3>,
    /// The 128 FP8 table shards in shard-index order, each `[ngram_shard_rows, head_dim]`.
    pub table_shards: Vec<Fp8E4M3View<'a, 2>>,
    /// The single BF16 multiplier shared by every table row `[1]`.
    ///
    /// Not a ModelOpt scale: `*.ple.*` is in the quantization ignore list, so this must not
    /// travel through the reciprocal-divisor codec the routed experts use.
    pub table_scale: Bf16View<'a, 1>,
    /// The three I64 hash buffers, gated against the recomputed law.
    pub constants: Qwen38FlashNextEngramConstantBindings<'a>,
    /// Decoder layer owning these sources.
    pub layer: usize,
}

impl<'a> Qwen38FlashNextEngramBindings<'a> {
    /// Binds the exact Flash-Next engram source family.
    pub fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Self::bind_from(layer, &Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    pub(crate) fn bind_from(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let table = engram_table_prefix(layer, geometry.ple_layer)?;
        let prefix = format!("{}.ple", layer_prefix(layer));
        let width = geometry.hc_width as u64;
        let embed = geometry.ple_embed_dim as u64;
        let key_proj_weight = Bf16View::bind(
            tensor(&format!("{prefix}.key_proj.weight"))?,
            [width, embed],
        )?;
        let value_proj_weight = Bf16View::bind(
            tensor(&format!("{prefix}.value_proj.weight"))?,
            [embed, embed],
        )?;
        let mut norm =
            |name: &str| Bf16View::bind(tensor(&format!("{prefix}.{name}.weight"))?, [width]);
        let norm_key = norm("norm_key")?;
        let norm_query = norm("norm_query")?;
        let norm_conv = norm("norm_conv")?;
        let convolution_weight = Bf16View::bind(
            tensor(&format!("{prefix}.conv1d.weight"))?,
            [width, 1, geometry.ple_conv_kernel as u64],
        )?;
        let constants =
            Qwen38FlashNextEngramConstantBindings::bind_from(&table, |name| tensor(name))?;
        let mut table_shards = Vec::new();

        table_shards
            .try_reserve_exact(geometry.ngram_shards)
            .map_err(|_| {
                CheckpointError::source_binding(format!(
                    "layer-{layer} cannot reserve {} Flash-Next engram shard views",
                    geometry.ngram_shards
                ))
            })?;

        for shard in 0..geometry.ngram_shards {
            table_shards.push(Fp8E4M3View::bind(
                tensor(&format!("{table}.ngram_embedding.shard_{shard}.weight"))?,
                [
                    geometry.ngram_shard_rows as u64,
                    geometry.ngram_head_dim as u64,
                ],
            )?);
        }

        Ok(Self {
            key_proj_weight,
            value_proj_weight,
            norm_key,
            norm_query,
            norm_conv,
            convolution_weight,
            table_shards,
            table_scale: Bf16View::bind(
                tensor(&format!("{table}.ngram_embedding.weight_scale"))?,
                [1],
            )?,
            constants,
            layer,
        })
    }
}

/// BF16 text endpoints and the final collapsing mixer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38FlashNextTextEndpointBindings<'a> {
    /// BF16 token embedding matrix `[vocab, hidden]`.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 language-model head `[vocab, hidden]`.
    pub lm_head: Bf16View<'a, 2>,
    /// Model-level collapsing mixer, this target's only final normalization.
    pub mixer: Qwen38FlashNextHyperConnectionBindings<'a>,
}

impl<'a> Qwen38FlashNextTextEndpointBindings<'a> {
    /// Binds the exact Flash-Next embedding, LM head, and collapsing mixer.
    pub fn bind(snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>) -> CheckpointResult<Self> {
        Self::bind_from(&Qwen38FlashNextGeometry::target(), |name| {
            snapshot.tensor(name)
        })
    }

    /// Binds only the mmap-backed embedding used during token staging.
    pub fn bind_embedding(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
    ) -> CheckpointResult<Bf16View<'a, 2>> {
        Bf16View::bind(
            snapshot.tensor(EMBEDDING)?,
            [F::VOCAB as u64, F::HIDDEN as u64],
        )
    }

    pub(crate) fn bind_from(
        geometry: &Qwen38FlashNextGeometry,
        mut tensor: impl FnMut(&str) -> CheckpointResult<TensorView<'a>>,
    ) -> CheckpointResult<Self> {
        let shape = [geometry.vocab as u64, geometry.hidden as u64];

        Ok(Self {
            embedding: Bf16View::bind(tensor(EMBEDDING)?, shape)?,
            lm_head: Bf16View::bind(tensor(LM_HEAD)?, shape)?,
            mixer: Qwen38FlashNextHyperConnectionBindings::bind_from(
                HYPER_CONNECTION_MIXER,
                geometry,
                false,
                &mut tensor,
            )?,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::common::routes::{E2M1_VALUES_PER_BYTE, NVFP4_GROUP_SIZE};
    use crate::common::test_builder::SafeTensorTestBuilder;
    use crate::common::test_support::sources::{fixture_path, write_safetensors_payload};
    use crate::qwen38_flash_next::engram::tests::{
        MULTIPLIERS, OFFSETS, VOCAB_SIZES, engram_constant_fixture,
    };
    use crate::{CheckpointErrorCode, DType, SafeTensorFile};
    use serde_json::json;
    use std::fs;

    /// A Flash-Next geometry small enough to write to a fixture file, with every structural
    /// ratio the real target's admission depends on preserved: the four-branch widened stream,
    /// the interval-4 layer route, split experts, and a multi-shard engram table.
    pub(crate) fn test_geometry() -> Qwen38FlashNextGeometry {
        Qwen38FlashNextGeometry {
            layers: 4,
            full_attention_interval: 4,
            hidden: 128,
            vocab: 12,
            head_dim: 8,
            attention_query_rows: 16,
            attention_kv_rows: 8,
            attention_output_columns: 8,
            indexer_rows: 12,
            indexer_head_dim: 4,
            gdn_qkv_rows: 20,
            gdn_value_rows: 12,
            gdn_control_rows: 3,
            gdn_conv_kernel: 4,
            gdn_head_dim: 4,
            hc_count: 4,
            hc_lowrank: 5,
            hc_width: 512,
            expert_count: 3,
            expert_intermediate: 64,
            shared_intermediate: 64,
            ple_layer: 1,
            ple_embed_dim: 32,
            ple_conv_kernel: 4,
            ngram_shards: 4,
            ngram_shard_rows: 6,
            ngram_head_dim: 2,
        }
    }

    pub(crate) fn append_hyper_connection(
        fixture: &mut SafeTensorTestBuilder,
        prefix: &str,
        geometry: &Qwen38FlashNextGeometry,
        combines: bool,
    ) {
        fixture
            .add_bf16_ordinal(format!("{prefix}.hc_norm.weight"), &[geometry.hc_width])
            .add_bf16_ordinal(
                format!("{prefix}.input_mix_weight_down.weight"),
                &[geometry.hc_lowrank, geometry.hc_width],
            )
            .add_bf16_ordinal(
                format!("{prefix}.input_mix_weight_up.weight"),
                &[geometry.hc_width, geometry.hc_lowrank],
            );

        if combines {
            fixture.add_bf16_ordinal(
                format!("{prefix}.block_inject_weight.weight"),
                &[geometry.hc_count, geometry.hc_width],
            );
        }
    }

    pub(crate) fn hyper_connection_fixture(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
    ) -> SafeTensorTestBuilder {
        let prefix = layer_prefix(layer);
        let mut fixture = SafeTensorTestBuilder::new();

        for module in ["attn_hyper_connection", "mlp_hyper_connection"] {
            append_hyper_connection(&mut fixture, &format!("{prefix}.{module}"), geometry, true);
        }

        fixture
    }

    pub(crate) fn gdn_fixture(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
    ) -> SafeTensorTestBuilder {
        let prefix = format!("{}.linear_attn", layer_prefix(layer));
        let mut fixture = SafeTensorTestBuilder::new();

        for (projection, rows, columns) in [
            ("in_proj_qkv", geometry.gdn_qkv_rows, geometry.hidden),
            ("in_proj_z", geometry.gdn_value_rows, geometry.hidden),
            ("in_proj_a", geometry.gdn_control_rows, geometry.hidden),
            ("in_proj_b", geometry.gdn_control_rows, geometry.hidden),
            ("out_proj", geometry.hidden, geometry.gdn_value_rows),
        ] {
            fixture.add_bf16_ordinal(format!("{prefix}.{projection}.weight"), &[rows, columns]);
        }

        fixture
            .add_bf16_ordinal(
                format!("{prefix}.conv1d.weight"),
                &[geometry.gdn_qkv_rows, 1, geometry.gdn_conv_kernel],
            )
            .add_bf16_ordinal(format!("{prefix}.A_log"), &[geometry.gdn_control_rows])
            .add_bf16_ordinal(format!("{prefix}.dt_bias"), &[geometry.gdn_control_rows])
            .add_bf16_ordinal(format!("{prefix}.norm.weight"), &[geometry.gdn_head_dim]);

        fixture
    }

    pub(crate) fn sparse_attention_fixture(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
    ) -> SafeTensorTestBuilder {
        let prefix = format!("{}.self_attn", layer_prefix(layer));
        let mut fixture = SafeTensorTestBuilder::new();

        for (projection, rows, columns) in [
            ("q_proj", geometry.attention_query_rows, geometry.hidden),
            ("k_proj", geometry.attention_kv_rows, geometry.hidden),
            ("v_proj", geometry.attention_kv_rows, geometry.hidden),
            ("o_proj", geometry.hidden, geometry.attention_output_columns),
        ] {
            fixture.add_bf16_ordinal(format!("{prefix}.{projection}.weight"), &[rows, columns]);
        }

        fixture
            .add_bf16_ordinal(format!("{prefix}.q_norm.weight"), &[geometry.head_dim])
            .add_bf16_ordinal(format!("{prefix}.k_norm.weight"), &[geometry.head_dim])
            .add_bf16_ordinal(
                format!("{prefix}.indexer.index_qk_proj.weight"),
                &[geometry.indexer_rows, geometry.hidden],
            )
            .add_bf16_ordinal(
                format!("{prefix}.indexer.q_layernorm.weight"),
                &[geometry.indexer_head_dim],
            )
            .add_bf16_ordinal(
                format!("{prefix}.indexer.k_layernorm.weight"),
                &[geometry.indexer_head_dim],
            );

        fixture
    }

    /// One split routed expert with three projections and four ModelOpt planes each.
    pub(crate) fn append_expert(
        fixture: &mut SafeTensorTestBuilder,
        prefix: &str,
        geometry: &Qwen38FlashNextGeometry,
        marker: u8,
        input_scale: f32,
        weight_scale_2: f32,
    ) {
        for (projection, rows, columns) in [
            ("gate_proj", geometry.expert_intermediate, geometry.hidden),
            ("up_proj", geometry.expert_intermediate, geometry.hidden),
            ("down_proj", geometry.hidden, geometry.expert_intermediate),
        ] {
            let projection = format!("{prefix}.{projection}");

            fixture
                .add_rank0_f32(format!("{projection}.input_scale"), input_scale)
                .add_raw(
                    format!("{projection}.weight"),
                    DType::U8,
                    &[rows, columns / E2M1_VALUES_PER_BYTE],
                    marker,
                )
                .add_with(
                    format!("{projection}.weight_scale"),
                    DType::Fp8E4M3,
                    &[rows, columns / NVFP4_GROUP_SIZE],
                    |index| ((index * 37 + marker as usize) % 0x7f) as u8,
                )
                .add_rank0_f32(format!("{projection}.weight_scale_2"), weight_scale_2);
        }
    }

    pub(crate) fn moe_fixture(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
    ) -> SafeTensorTestBuilder {
        let prefix = format!("{}.mlp", layer_prefix(layer));
        let mut fixture = SafeTensorTestBuilder::new();

        fixture
            .add_bf16_ordinal(
                format!("{prefix}.gate.weight"),
                &[geometry.expert_count, geometry.hidden],
            )
            .add_bf16_ordinal(
                format!("{prefix}.shared_expert_gate.weight"),
                &[1, geometry.hidden],
            )
            .add_bf16_ordinal(
                format!("{prefix}.shared_expert.gate_proj.weight"),
                &[geometry.shared_intermediate, geometry.hidden],
            )
            .add_bf16_ordinal(
                format!("{prefix}.shared_expert.up_proj.weight"),
                &[geometry.shared_intermediate, geometry.hidden],
            )
            .add_bf16_ordinal(
                format!("{prefix}.shared_expert.down_proj.weight"),
                &[geometry.hidden, geometry.shared_intermediate],
            );

        for expert in 0..geometry.expert_count {
            append_expert(
                &mut fixture,
                &format!("{prefix}.experts.{expert}"),
                geometry,
                u8::try_from(expert + 1).unwrap(),
                0.25,
                0.125,
            );
        }

        fixture
    }

    pub(crate) fn engram_fixture(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
    ) -> SafeTensorTestBuilder {
        // One constant byte per shard: enough to tell shards apart, and no more.
        engram_fixture_with(layer, geometry, |shard, _| {
            u8::try_from(0x20 + shard).unwrap()
        })
    }

    /// The engram family with caller-chosen table bytes, for tests that address exact rows.
    pub(crate) fn engram_fixture_with(
        layer: usize,
        geometry: &Qwen38FlashNextGeometry,
        mut shard_byte: impl FnMut(usize, usize) -> u8,
    ) -> SafeTensorTestBuilder {
        let prefix = format!("{}.ple", layer_prefix(layer));
        let table = format!("{prefix}.ple_embedding");
        // The I64 buffers never shrink: they are the checkpoint's hash ground truth.
        let mut fixture = engram_constant_fixture(MULTIPLIERS, VOCAB_SIZES, OFFSETS);

        fixture
            .add_bf16_ordinal(
                format!("{prefix}.key_proj.weight"),
                &[geometry.hc_width, geometry.ple_embed_dim],
            )
            .add_bf16_ordinal(
                format!("{prefix}.value_proj.weight"),
                &[geometry.ple_embed_dim, geometry.ple_embed_dim],
            );

        for name in ["norm_key", "norm_query", "norm_conv"] {
            fixture.add_bf16_ordinal(format!("{prefix}.{name}.weight"), &[geometry.hc_width]);
        }

        fixture.add_bf16_ordinal(
            format!("{prefix}.conv1d.weight"),
            &[geometry.hc_width, 1, geometry.ple_conv_kernel],
        );

        for shard in 0..geometry.ngram_shards {
            fixture.add_with(
                format!("{table}.ngram_embedding.shard_{shard}.weight"),
                DType::Fp8E4M3,
                &[geometry.ngram_shard_rows, geometry.ngram_head_dim],
                |byte| shard_byte(shard, byte),
            );
        }

        fixture.add_bf16(
            format!("{table}.ngram_embedding.weight_scale"),
            &[1],
            0x3951,
        );

        fixture
    }

    pub(crate) fn endpoint_fixture(geometry: &Qwen38FlashNextGeometry) -> SafeTensorTestBuilder {
        let mut fixture = SafeTensorTestBuilder::new();

        fixture
            .add_bf16_ordinal(EMBEDDING, &[geometry.vocab, geometry.hidden])
            .add_bf16_ordinal(LM_HEAD, &[geometry.vocab, geometry.hidden]);
        append_hyper_connection(&mut fixture, HYPER_CONNECTION_MIXER, geometry, false);

        fixture
    }

    #[test]
    fn binds_the_exact_qwen38_flash_next_hyper_connection_pair() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-hyper-connection");
        hyper_connection_fixture(0, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextLayerHyperConnections::bind_from(0, &geometry, |name| file.tensor(name))
                .unwrap();

        for gated in [bindings.attention, bindings.mlp] {
            assert_eq!(gated.hc_norm.shape(), &[512]);
            assert_eq!(gated.input_mix_down.shape(), &[5, 512]);
            assert_eq!(gated.input_mix_up.shape(), &[512, 5]);
            assert_eq!(gated.block_inject.unwrap().shape(), &[4, 512]);
        }
        assert_eq!(bindings.layer, 0);
        assert_eq!(
            bindings.attention.hc_norm.name(),
            "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_the_exact_qwen38_flash_next_gdn_source_family() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-gdn");
        gdn_fixture(0, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextGdnBindings::bind_from(0, &geometry, |name| file.tensor(name)).unwrap();

        assert_eq!(bindings.qkv_weight.shape(), &[20, 128]);
        assert_eq!(bindings.z_weight.shape(), &[12, 128]);
        assert_eq!(bindings.a_control_weight.shape(), &[3, 128]);
        assert_eq!(bindings.b_control_weight.shape(), &[3, 128]);
        assert_eq!(bindings.output_weight.shape(), &[128, 12]);
        assert_eq!(bindings.convolution_weight.shape(), &[20, 1, 4]);
        assert_eq!(bindings.a_log.shape(), &[3]);
        assert_eq!(bindings.dt_bias.shape(), &[3]);
        assert_eq!(bindings.norm.shape(), &[4]);
        assert_eq!(bindings.layer, 0);

        // The interval-4 route: layer 3 is sparse attention, not GDN.
        let error = Qwen38FlashNextGdnBindings::bind_from(3, &geometry, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("GDN source contract"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_the_exact_qwen38_flash_next_sparse_attention_source_family() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-sparse_attention");
        sparse_attention_fixture(3, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings = Qwen38FlashNextSparseAttentionBindings::bind_from(3, &geometry, |name| {
            file.tensor(name)
        })
        .unwrap();

        assert_eq!(bindings.query_gate_weight.shape(), &[16, 128]);
        assert_eq!(bindings.key_weight.shape(), &[8, 128]);
        assert_eq!(bindings.value_weight.shape(), &[8, 128]);
        assert_eq!(bindings.output_weight.shape(), &[128, 8]);
        assert_eq!(bindings.query_norm.shape(), &[8]);
        assert_eq!(bindings.key_norm.shape(), &[8]);
        assert_eq!(bindings.indexer.qk_weight.shape(), &[12, 128]);
        assert_eq!(bindings.indexer.query_norm.shape(), &[4]);
        assert_eq!(bindings.indexer.key_norm.shape(), &[4]);
        assert_eq!(bindings.layer, 3);

        let error = Qwen38FlashNextSparseAttentionBindings::bind_from(0, &geometry, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("full-attention source contract"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_qwen38_flash_next_split_experts_in_numeric_order() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-moe");
        moe_fixture(2, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextMoeBindings::bind_from(2, &geometry, |name| file.tensor(name)).unwrap();

        assert_eq!(bindings.router_weight.shape(), &[3, 128]);
        assert_eq!(bindings.shared_expert.gate_weight.shape(), &[1, 128]);
        assert_eq!(bindings.shared_expert.gate_proj_weight.shape(), &[64, 128]);
        assert_eq!(bindings.shared_expert.down_proj_weight.shape(), &[128, 64]);
        assert_eq!(bindings.experts.len(), 3);

        for (expert, planes) in bindings.experts.iter().enumerate() {
            assert_eq!(planes.expert, expert);
            assert_eq!(planes.gate.weight.shape(), &[64, 64]);
            assert_eq!(planes.gate.block_scale.shape(), &[64, 8]);
            assert_eq!(planes.down.weight.shape(), &[128, 32]);
            assert_eq!(planes.down.block_scale.shape(), &[128, 4]);
            // Every plane carries this expert's own marker, so numeric order is observable.
            assert_eq!(planes.gate.weight.bytes()[0], expert as u8 + 1);
            assert_eq!(planes.up.weight.bytes()[0], expert as u8 + 1);
            assert_eq!(planes.down.weight.bytes()[0], expert as u8 + 1);
        }

        assert_eq!(
            bindings.experts[2].gate.weight.name(),
            "model.language_model.layers.2.mlp.experts.2.gate_proj.weight"
        );
        assert_eq!(bindings.layer, 2);

        // MoE is on every layer, so only the stack bound rejects.
        Qwen38FlashNextMoeBindings::bind_from(3, &geometry, |name| file.tensor(name)).unwrap_err();
        let error = Qwen38FlashNextMoeBindings::bind_from(4, &geometry, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert!(error.to_string().contains("Flash-Next MoE source contract"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_a_split_expert_whose_gate_and_up_scalars_disagree() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-moe-scale");
        let (header, mut payload) = moe_fixture(2, &geometry).into_parts();
        let name = "model.language_model.layers.2.mlp.experts.1.up_proj.input_scale";
        let offset = header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        payload[offset..offset + 4].copy_from_slice(&0.5f32.to_le_bytes());
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error = Qwen38FlashNextMoeBindings::bind_from(2, &geometry, |name| file.tensor(name))
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("expert-1 gate/up input_scale values differ"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binds_the_exact_qwen38_flash_next_engram_source_family() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-engram");
        engram_fixture(geometry.ple_layer, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextEngramBindings::bind_from(geometry.ple_layer, &geometry, |name| {
                file.tensor(name)
            })
            .unwrap();

        assert_eq!(bindings.key_proj_weight.shape(), &[512, 32]);
        assert_eq!(bindings.value_proj_weight.shape(), &[32, 32]);
        assert_eq!(bindings.norm_key.shape(), &[512]);
        assert_eq!(bindings.norm_query.shape(), &[512]);
        assert_eq!(bindings.norm_conv.shape(), &[512]);
        assert_eq!(bindings.convolution_weight.shape(), &[512, 1, 4]);
        assert_eq!(bindings.table_shards.len(), 4);
        assert_eq!(bindings.table_scale.word(0), Some(0x3951));

        for (shard, view) in bindings.table_shards.iter().enumerate() {
            assert_eq!(view.shape(), &[6, 2]);
            assert_eq!(view.codes()[0], 0x20 + shard as u8, "shard {shard} order");
        }

        assert_eq!(
            bindings.constants.layer_multipliers.value(0),
            Some(23_703_573_157_769)
        );
        assert_eq!(bindings.layer, geometry.ple_layer);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_an_engram_on_any_layer_but_the_ple_layer() {
        let geometry = test_geometry();
        let error = Qwen38FlashNextEngramBindings::bind_from(0, &geometry, |_| {
            panic!("the route check must reject before tensor lookup")
        })
        .unwrap_err();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("carries no engram"), "{error}");
    }

    #[test]
    fn binds_the_exact_qwen38_flash_next_text_endpoints_without_a_final_norm() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-endpoints");
        endpoint_fixture(&geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let bindings =
            Qwen38FlashNextTextEndpointBindings::bind_from(&geometry, |name| file.tensor(name))
                .unwrap();

        assert_eq!(bindings.embedding.shape(), &[12, 128]);
        assert_eq!(bindings.lm_head.shape(), &[12, 128]);
        assert_eq!(bindings.mixer.hc_norm.shape(), &[512]);
        assert_eq!(bindings.mixer.input_mix_down.shape(), &[5, 512]);
        assert_eq!(bindings.mixer.input_mix_up.shape(), &[512, 5]);
        assert!(
            bindings.mixer.block_inject.is_none(),
            "the collapsing mixer writes back nothing and carries no block_inject_weight"
        );
        assert!(
            file.tensor("model.language_model.norm.weight").is_err(),
            "this architecture has no final RMSNorm"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_qwen38_flash_next_source_shape_and_dtype_drift() {
        let geometry = test_geometry();

        for (label, tensor_name, replacement, expected) in [
            (
                "qwen38_flash_next-gdn-shape",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                json!([6, 64]),
                "shape [6, 64], expected [3, 128]",
            ),
            (
                "qwen38_flash_next-gdn-conv",
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                json!([20, 4]),
                "shape [20, 4], expected [20, 1, 4]",
            ),
        ] {
            let path = fixture_path(label);
            let (mut header, payload) = gdn_fixture(0, &geometry).into_parts();
            header[tensor_name]["shape"] = replacement;
            write_safetensors_payload(&path, header, &payload);
            let file = SafeTensorFile::open(&path).unwrap();

            let error =
                Qwen38FlashNextGdnBindings::bind_from(0, &geometry, |name| file.tensor(name))
                    .err()
                    .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::Tensor);
            assert!(error.to_string().contains(tensor_name), "{error}");
            assert!(error.to_string().contains(expected), "{error}");

            fs::remove_file(path).unwrap();
        }
    }

    /// The BF16 release stores this table as BF16; the NVFP4 release stores it as FP8 with a
    /// separate scale. Binding the wrong one would silently halve every row's stride.
    #[test]
    fn refuses_an_engram_table_shard_that_is_not_fp8() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-engram-dtype");
        let (mut header, payload) = engram_fixture(geometry.ple_layer, &geometry).into_parts();
        let name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight";
        header[name]["dtype"] = json!("BF16");
        header[name]["shape"] = json!([3, 2]);
        write_safetensors_payload(&path, header, &payload);
        let file = SafeTensorFile::open(&path).unwrap();

        let error =
            Qwen38FlashNextEngramBindings::bind_from(geometry.ple_layer, &geometry, |name| {
                file.tensor(name)
            })
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::Tensor);
        assert!(
            error
                .to_string()
                .contains("dtype `BF16`, expected `F8_E4M3`"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    /// Pins the target extents consumed by the shared vision binder.
    #[test]
    fn qwen38_flash_next_vision_extents_match_what_the_shared_encoder_binder_demands() {
        let merged = F::VISION_HIDDEN * F::VISION_SPATIAL_MERGE_SIZE * F::VISION_SPATIAL_MERGE_SIZE;

        assert_eq!(merged, 4_608);
        assert_eq!(3 * F::VISION_HIDDEN, 3_456);
        assert_eq!(F::VISION_OUTPUT_HIDDEN, F::HIDDEN);
        assert_eq!(
            [F::VISION_POSITIONS, F::VISION_HIDDEN],
            [2_304, 1_152],
            "the shared binder reads pos_embed at exactly this shape"
        );
    }
}

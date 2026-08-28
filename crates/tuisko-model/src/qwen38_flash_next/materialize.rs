//! Lossless Qwen3.8-Flash-Next host layouts.
//!
//! Packed expert weights and FP8 engram rows stay borrowed. Materialization stages only scale
//! permutations and fused BF16 projection planes, with experts restored to numeric order.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::materialized::{MaterializedMemory, sealed};
use crate::common::modelopt_codec::{
    MaterializedModelOptNvfp4Linear, ModelOptScaleCodec, logical_columns,
    materialize_modelopt_linear,
};
use crate::common::routes::validate_nvfp4_scales;
use crate::common::scale_swizzle::{
    PlaneGatherer, host_shape, materialization_pool, materialization_workers,
};
use crate::common::source_binding::{SourceLayerBinding, sealed as binding_sealed};
use crate::qwen38_flash_next::bindings::{
    Qwen38FlashNextEngramBindings, Qwen38FlashNextExpertBindings, Qwen38FlashNextGdnBindings,
    Qwen38FlashNextGeometry, Qwen38FlashNextHyperConnectionBindings,
    Qwen38FlashNextIndexerBindings, Qwen38FlashNextLayerHyperConnections,
    Qwen38FlashNextMoeBindings, Qwen38FlashNextMtpBindings, Qwen38FlashNextSharedExpertBindings,
    Qwen38FlashNextSparseAttentionBindings, Qwen38FlashNextTextEndpointBindings,
};
use crate::qwen38_flash_next::engram::Qwen38FlashNextEngramHashConstants;
use crate::qwen38_flash_next::engram_hash::Qwen38FlashNextEngramTable;
use crate::{Bf16View, CheckpointError, CheckpointResult, Qwen38FlashNext};
use rayon::prelude::*;

/// Byte extent of one plane inside a materialized pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextPlaneExtent {
    /// Byte offset from the start of the owning pool.
    pub offset: usize,
    /// Byte length of the plane.
    pub bytes: usize,
}

/// One routed expert with borrowed weights and staged scale extents.
#[derive(Clone, Copy, Debug)]
pub struct MaterializedQwen38FlashNextExpert<'a> {
    /// Borrowed packed E2M1 gate words `[intermediate, hidden / 2]`.
    pub gate_weight_e2m1: &'a [u8],
    /// Borrowed packed E2M1 up words `[intermediate, hidden / 2]`.
    pub up_weight_e2m1: &'a [u8],
    /// Borrowed packed E2M1 down words `[hidden, intermediate / 2]`.
    pub down_weight_e2m1: &'a [u8],
    /// Fused gate-then-up swizzled scale extent in the layer pool.
    pub gate_up_scale: Qwen38FlashNextPlaneExtent,
    /// Down-projection swizzled scale extent in the layer pool.
    pub down_scale: Qwen38FlashNextPlaneExtent,
    /// Exact source activation scale shared by gate and up.
    pub gate_up_input_scale: f32,
    /// Exact source second-stage weight scale shared by gate and up.
    pub gate_up_weight_scale_2: f32,
    /// Exact source down-projection activation scale.
    pub down_input_scale: f32,
    /// Exact source down-projection second-stage weight scale.
    pub down_weight_scale_2: f32,
    /// Reciprocal gate/up activation scale consumed by the kernels.
    pub gate_up_input_scale_divisor: f32,
    /// Reciprocal gate/up weight scale consumed by the kernels.
    pub gate_up_weight_scale_divisor: f32,
    /// Reciprocal down-projection activation scale consumed by the kernels.
    pub down_input_scale_divisor: f32,
    /// Reciprocal down-projection weight scale consumed by the kernels.
    pub down_weight_scale_divisor: f32,
    /// Numeric expert index within the layer's pool.
    pub expert: usize,
}

/// Runtime-native routed-expert pool for one decoder layer, expert-major in numeric order.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextExpertPool<'a> {
    /// Every expert's swizzled E4M3 scales, expert-major, fused gate/up then down.
    pub scale_e4m3_swizzled: Vec<u8>,
    /// Experts in numeric order; every extent indexes `scale_e4m3_swizzled`.
    pub experts: Vec<MaterializedQwen38FlashNextExpert<'a>>,
    /// Swizzled scale bytes owned by one expert.
    pub expert_scale_bytes: usize,
    /// Borrowed packed E2M1 bytes owned by one expert across gate, up, and down.
    pub expert_weight_bytes: usize,
    /// Routed experts in this pool.
    pub expert_count: usize,
    /// Intermediate width of each routed expert.
    pub intermediate: usize,
    /// Residual-stream width.
    pub hidden: usize,
    /// Decoder layer owning this pool.
    pub layer: usize,
}

impl MaterializedQwen38FlashNextExpertPool<'_> {
    /// Total borrowed packed E2M1 bytes this pool addresses, never staged on the host heap.
    pub fn borrowed_weight_bytes(&self) -> usize {
        self.expert_weight_bytes * self.expert_count
    }
}

impl sealed::Sealed for MaterializedQwen38FlashNextExpertPool<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextExpertPool<'_> {
    fn host_bytes(&self) -> usize {
        self.scale_e4m3_swizzled.len() + size_of_val(self.experts.as_slice())
    }
}

/// Runtime-native MoE planes for one decoder layer.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextMoe<'a> {
    /// Full 512-way router weights retained zero-copy.
    pub router_weight: Bf16View<'a, 2>,
    /// Always-active shared expert, entirely BF16 and entirely borrowed.
    pub shared_expert: Qwen38FlashNextSharedExpertBindings<'a>,
    /// Routed expert pool in numeric expert order.
    pub experts: MaterializedQwen38FlashNextExpertPool<'a>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl sealed::Sealed for MaterializedQwen38FlashNextMoe<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextMoe<'_> {
    fn host_bytes(&self) -> usize {
        self.experts.host_bytes()
    }
}

impl binding_sealed::Sealed for Qwen38FlashNextMoeBindings<'_> {}

impl<'a> SourceLayerBinding<'a, Qwen38FlashNext> for Qwen38FlashNextMoeBindings<'a> {
    type Materialized = MaterializedQwen38FlashNextMoe<'a>;

    fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Qwen38FlashNextMoeBindings::bind(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Qwen38FlashNextMoeBindings::materialize(self)
    }
}

impl<'a> Qwen38FlashNextMoeBindings<'a> {
    /// Reorders routed experts numerically and swizzles their scales without requantizing.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextMoe<'a>> {
        self.materialize_with(&Qwen38FlashNextGeometry::target())
    }

    pub(crate) fn materialize_with(
        self,
        geometry: &Qwen38FlashNextGeometry,
    ) -> CheckpointResult<MaterializedQwen38FlashNextMoe<'a>> {
        let layer = self.layer;

        if self.experts.len() != geometry.expert_count {
            return Err(moe_error(
                layer,
                format!(
                    "source has {} routed experts, expected {}",
                    self.experts.len(),
                    geometry.expert_count
                ),
            ));
        }

        let experts = materialize_qwen38_flash_next_experts(
            self.experts,
            layer,
            geometry.hidden,
            geometry.expert_intermediate,
        )?;

        Ok(MaterializedQwen38FlashNextMoe {
            router_weight: self.router_weight,
            shared_expert: self.shared_expert,
            experts,
            layer,
        })
    }
}

/// One routed expert after scale conversion, before it is placed in the layer pool.
struct PreparedQwen38FlashNextExpert<'a> {
    gate_weight_e2m1: &'a [u8],
    up_weight_e2m1: &'a [u8],
    down: MaterializedModelOptNvfp4Linear<'a>,
    gate_up_scale_e4m3_swizzled: Vec<u8>,
    gate_up_input_scale: f32,
    gate_up_weight_scale_2: f32,
    gate_up_input_scale_divisor: f32,
    gate_up_weight_scale_divisor: f32,
    expert: usize,
}

fn materialize_qwen38_flash_next_experts<'a>(
    bindings: Vec<Qwen38FlashNextExpertBindings<'a>>,
    layer: usize,
    hidden: usize,
    intermediate: usize,
) -> CheckpointResult<MaterializedQwen38FlashNextExpertPool<'a>> {
    if bindings.is_empty() {
        return Err(moe_error(layer, "routed expert source is empty"));
    }

    let expert_count = bindings.len();
    let prepare = |binding: Qwen38FlashNextExpertBindings<'a>| {
        prepare_expert(binding, layer, hidden, intermediate)
    };
    let prepared = if expert_count > 1 && materialization_workers() > 1 {
        materialization_pool(&format!("layer-{layer} Flash-Next routed experts"))?.install(|| {
            bindings
                .into_par_iter()
                .map(prepare)
                .collect::<CheckpointResult<Vec<_>>>()
        })
    } else {
        bindings
            .into_iter()
            .map(prepare)
            .collect::<CheckpointResult<Vec<_>>>()
    }?;

    flatten_expert_pool(prepared, layer, hidden, intermediate)
}

/// Converts scale conventions and swizzles fused gate/up scales.
fn prepare_expert<'a>(
    binding: Qwen38FlashNextExpertBindings<'a>,
    layer: usize,
    hidden: usize,
    intermediate: usize,
) -> CheckpointResult<PreparedQwen38FlashNextExpert<'a>> {
    let expert = binding.expert;
    let role = format!("expert-{expert}");
    let [gate_rows, packed_columns] =
        host_shape(binding.gate.weight.shape(), &format!("{role} gate weights"))?;
    let up_weight_shape = host_shape(binding.up.weight.shape(), &format!("{role} up weights"))?;
    let [gate_scale_rows, groups] = host_shape(
        binding.gate.block_scale.shape(),
        &format!("{role} gate scales"),
    )?;
    let up_scale_shape = host_shape(binding.up.block_scale.shape(), &format!("{role} up scales"))?;

    if [gate_rows, packed_columns] != up_weight_shape
        || [gate_scale_rows, groups] != up_scale_shape
        || gate_rows != gate_scale_rows
    {
        return Err(moe_error(
            layer,
            format!("{role} gate/up source planes have incompatible shapes"),
        ));
    }

    let columns = logical_columns(packed_columns, groups, layer, &format!("{role} gate/up"))?;

    if gate_rows != intermediate || columns != hidden {
        return Err(moe_error(
            layer,
            format!("{role} gate/up is {gate_rows}x{columns}, expected {intermediate}x{hidden}"),
        ));
    }

    validate_nvfp4_scales(
        layer,
        &format!("{role} gate"),
        binding.gate.block_scale.codes(),
    )?;
    validate_nvfp4_scales(layer, &format!("{role} up"), binding.up.block_scale.codes())?;
    ModelOptScaleCodec::require_same_source_scale(
        layer,
        &format!("{role} gate/up input_scale"),
        &binding.gate.input_scale,
        &binding.up.input_scale,
    )?;
    ModelOptScaleCodec::require_same_source_scale(
        layer,
        &format!("{role} gate/up weight_scale_2"),
        &binding.gate.weight_scale_2,
        &binding.up.weight_scale_2,
    )?;

    let gate_up_input_scale = ModelOptScaleCodec::source_scale(
        layer,
        &format!("{role} gate/up input_scale"),
        &binding.gate.input_scale,
    )?;
    let gate_up_weight_scale_2 = ModelOptScaleCodec::source_scale(
        layer,
        &format!("{role} gate/up weight_scale_2"),
        &binding.gate.weight_scale_2,
    )?;
    let gate_up_scale_e4m3_swizzled = PlaneGatherer::swizzle_scales(
        &[
            binding.gate.block_scale.codes(),
            binding.up.block_scale.codes(),
        ],
        gate_rows,
        groups,
        layer,
        &format!("{role} gate/up"),
    )?;
    let down = materialize_modelopt_linear(binding.down, layer, &format!("{role} down"))?;

    if down.rows != hidden || down.columns != intermediate {
        return Err(moe_error(
            layer,
            format!(
                "{role} down is {}x{}, expected {hidden}x{intermediate}",
                down.rows, down.columns
            ),
        ));
    }

    Ok(PreparedQwen38FlashNextExpert {
        gate_weight_e2m1: binding.gate.weight.bytes(),
        up_weight_e2m1: binding.up.weight.bytes(),
        down,
        gate_up_scale_e4m3_swizzled,
        gate_up_input_scale,
        gate_up_weight_scale_2,
        gate_up_input_scale_divisor: ModelOptScaleCodec::to_reciprocal_divisor(
            layer,
            &format!("{role} gate/up input"),
            gate_up_input_scale,
        )?,
        gate_up_weight_scale_divisor: ModelOptScaleCodec::to_reciprocal_divisor(
            layer,
            &format!("{role} gate/up weight"),
            gate_up_weight_scale_2,
        )?,
        expert,
    })
}

/// Places every prepared expert in the layer's expert-major pool, in numeric order.
fn flatten_expert_pool<'a>(
    prepared: Vec<PreparedQwen38FlashNextExpert<'a>>,
    layer: usize,
    hidden: usize,
    intermediate: usize,
) -> CheckpointResult<MaterializedQwen38FlashNextExpertPool<'a>> {
    let expert_count = prepared.len();
    let first = prepared
        .first()
        .expect("nonempty expert source was checked before materialization");
    let gate_up_scale_bytes = first.gate_up_scale_e4m3_swizzled.len();
    let down_scale_bytes = first.down.scale_e4m3_swizzled.len();
    let expert_scale_bytes = checked(
        gate_up_scale_bytes.checked_add(down_scale_bytes),
        layer,
        "expert scale extent",
    )?;
    let expert_weight_bytes = checked(
        first
            .gate_weight_e2m1
            .len()
            .checked_add(first.up_weight_e2m1.len())
            .and_then(|bytes| bytes.checked_add(first.down.weight_e2m1.len())),
        layer,
        "expert weight extent",
    )?;
    let pool_bytes = checked(
        expert_scale_bytes.checked_mul(expert_count),
        layer,
        "expert scale pool",
    )?;

    let mut scale_e4m3_swizzled = Vec::new();
    let mut experts = Vec::new();

    scale_e4m3_swizzled
        .try_reserve_exact(pool_bytes)
        .map_err(|_| {
            moe_error(
                layer,
                format!("cannot reserve {pool_bytes} host bytes for the expert scale pool"),
            )
        })?;
    experts.try_reserve_exact(expert_count).map_err(|_| {
        moe_error(
            layer,
            format!("cannot reserve {expert_count} expert descriptors"),
        )
    })?;

    for (position, planes) in prepared.into_iter().enumerate() {
        // Pool offsets are indexed by numeric expert id.
        if planes.expert != position {
            return Err(moe_error(
                layer,
                format!(
                    "routed expert {} is at pool position {position}",
                    planes.expert
                ),
            ));
        }
        // Every expert must match the pool's uniform stride.
        let observed_weight_bytes = checked(
            planes
                .gate_weight_e2m1
                .len()
                .checked_add(planes.up_weight_e2m1.len())
                .and_then(|bytes| bytes.checked_add(planes.down.weight_e2m1.len())),
            layer,
            "expert weight extent",
        )?;

        if planes.gate_up_scale_e4m3_swizzled.len() != gate_up_scale_bytes
            || planes.down.scale_e4m3_swizzled.len() != down_scale_bytes
            || observed_weight_bytes != expert_weight_bytes
        {
            return Err(moe_error(
                layer,
                format!(
                    "expert-{} plane extents differ from expert-0",
                    planes.expert
                ),
            ));
        }

        let base = expert_scale_bytes * position;

        scale_e4m3_swizzled.extend_from_slice(&planes.gate_up_scale_e4m3_swizzled);
        scale_e4m3_swizzled.extend_from_slice(&planes.down.scale_e4m3_swizzled);
        experts.push(MaterializedQwen38FlashNextExpert {
            gate_weight_e2m1: planes.gate_weight_e2m1,
            up_weight_e2m1: planes.up_weight_e2m1,
            down_weight_e2m1: planes.down.weight_e2m1,
            gate_up_scale: Qwen38FlashNextPlaneExtent {
                offset: base,
                bytes: gate_up_scale_bytes,
            },
            down_scale: Qwen38FlashNextPlaneExtent {
                offset: base + gate_up_scale_bytes,
                bytes: down_scale_bytes,
            },
            gate_up_input_scale: planes.gate_up_input_scale,
            gate_up_weight_scale_2: planes.gate_up_weight_scale_2,
            down_input_scale: planes.down.input_scale,
            down_weight_scale_2: planes.down.weight_scale_2,
            gate_up_input_scale_divisor: planes.gate_up_input_scale_divisor,
            gate_up_weight_scale_divisor: planes.gate_up_weight_scale_divisor,
            down_input_scale_divisor: planes.down.input_scale_divisor,
            down_weight_scale_divisor: planes.down.weight_scale_divisor,
            expert: planes.expert,
        });
    }

    Ok(MaterializedQwen38FlashNextExpertPool {
        scale_e4m3_swizzled,
        experts,
        expert_scale_bytes,
        expert_weight_bytes,
        expert_count,
        intermediate,
        hidden,
        layer,
    })
}

/// Runtime-native planes for one gated DeltaNet layer.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextGdn<'a> {
    /// Fused QKV then Z BF16 words `[gdn_input_rows, hidden]`.
    pub input_weight_bf16: Vec<u8>,
    /// Fused A then B control BF16 words `[2 * gdn_control_rows, hidden]`.
    pub control_weight_bf16: Vec<u8>,
    /// QKV rows preceding the Z rows in the fused input plane.
    pub qkv_rows: usize,
    /// Fused QKV/Z output row count.
    pub input_rows: usize,
    /// Logical residual-stream input width.
    pub input_columns: usize,
    /// Rows in one A or B control projection.
    pub control_rows_per_projection: usize,
    /// Recurrent-state output projection retained zero-copy.
    pub output_weight: Bf16View<'a, 2>,
    /// Depthwise causal convolution retained zero-copy.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Log-space recurrence decay parameters retained zero-copy.
    pub a_log: Bf16View<'a, 1>,
    /// Recurrence time-step bias retained zero-copy.
    pub dt_bias: Bf16View<'a, 1>,
    /// Per-head gated RMSNorm weights retained zero-copy.
    pub norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl sealed::Sealed for MaterializedQwen38FlashNextGdn<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextGdn<'_> {
    fn host_bytes(&self) -> usize {
        self.input_weight_bf16.len() + self.control_weight_bf16.len()
    }
}

impl binding_sealed::Sealed for Qwen38FlashNextGdnBindings<'_> {}

impl<'a> SourceLayerBinding<'a, Qwen38FlashNext> for Qwen38FlashNextGdnBindings<'a> {
    type Materialized = MaterializedQwen38FlashNextGdn<'a>;

    fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Qwen38FlashNextGdnBindings::bind(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Qwen38FlashNextGdnBindings::materialize(self)
    }
}

impl<'a> Qwen38FlashNextGdnBindings<'a> {
    /// Fuses the four separate input projections into the two planes the mixer reads.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextGdn<'a>> {
        self.materialize_with(&Qwen38FlashNextGeometry::target())
    }

    pub(crate) fn materialize_with(
        self,
        geometry: &Qwen38FlashNextGeometry,
    ) -> CheckpointResult<MaterializedQwen38FlashNextGdn<'a>> {
        let layer = self.layer;
        let hidden = geometry.hidden as u64;
        let qkv_rows = geometry.gdn_qkv_rows;
        let value_rows = geometry.gdn_value_rows;
        let control_rows = geometry.gdn_control_rows;

        if self.qkv_weight.shape() != &[qkv_rows as u64, hidden]
            || self.z_weight.shape() != &[value_rows as u64, hidden]
            || self.a_control_weight.shape() != &[control_rows as u64, hidden]
            || self.b_control_weight.shape() != &[control_rows as u64, hidden]
            || self.output_weight.shape() != &[hidden, value_rows as u64]
            || self.convolution_weight.shape()
                != &[qkv_rows as u64, 1, geometry.gdn_conv_kernel as u64]
            || self.a_log.shape() != &[control_rows as u64]
            || self.dt_bias.shape() != &[control_rows as u64]
            || self.norm.shape() != &[geometry.gdn_head_dim as u64]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} Flash-Next GDN source geometry differs from its contract"
            )));
        }

        let input_rows = checked(qkv_rows.checked_add(value_rows), layer, "GDN input rows")?;
        let input_weight_bf16 = PlaneGatherer::gather(
            [self.qkv_weight.bytes(), self.z_weight.bytes()],
            &format!("layer-{layer} Flash-Next GDN QKV/Z weights"),
        )?;
        let control_weight_bf16 = PlaneGatherer::gather(
            [self.a_control_weight.bytes(), self.b_control_weight.bytes()],
            &format!("layer-{layer} Flash-Next GDN A/B control weights"),
        )?;

        Ok(MaterializedQwen38FlashNextGdn {
            input_weight_bf16,
            control_weight_bf16,
            qkv_rows,
            input_rows,
            input_columns: geometry.hidden,
            control_rows_per_projection: control_rows,
            output_weight: self.output_weight,
            convolution_weight: self.convolution_weight,
            a_log: self.a_log,
            dt_bias: self.dt_bias,
            norm: self.norm,
            layer,
        })
    }
}

/// Runtime-native planes for one sparse-attention layer.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextSparseAttention<'a> {
    /// Fused query/gate, key, and value BF16 words `[attention_qkv_rows, hidden]`.
    pub qkv_weight_bf16: Vec<u8>,
    /// Query-plus-gate rows preceding the key and value rows.
    pub query_rows: usize,
    /// Rows in each key or value projection.
    pub kv_rows: usize,
    /// Fused query/gate, key, and value row count.
    pub qkv_rows: usize,
    /// Logical query/key/value input width.
    pub qkv_columns: usize,
    /// Attention output projection retained zero-copy.
    pub output_weight: Bf16View<'a, 2>,
    /// Per-head query RMSNorm weights retained zero-copy.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights retained zero-copy.
    pub key_norm: Bf16View<'a, 1>,
    /// Indexer planes, all retained zero-copy.
    pub indexer: Qwen38FlashNextIndexerBindings<'a>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl sealed::Sealed for MaterializedQwen38FlashNextSparseAttention<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextSparseAttention<'_> {
    fn host_bytes(&self) -> usize {
        self.qkv_weight_bf16.len()
    }
}

impl binding_sealed::Sealed for Qwen38FlashNextSparseAttentionBindings<'_> {}

impl<'a> SourceLayerBinding<'a, Qwen38FlashNext> for Qwen38FlashNextSparseAttentionBindings<'a> {
    type Materialized = MaterializedQwen38FlashNextSparseAttention<'a>;

    fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Qwen38FlashNextSparseAttentionBindings::bind(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Qwen38FlashNextSparseAttentionBindings::materialize(self)
    }
}

impl<'a> Qwen38FlashNextSparseAttentionBindings<'a> {
    /// Fuses query/gate, key, and value while retaining the indexer separately.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextSparseAttention<'a>> {
        self.materialize_with(&Qwen38FlashNextGeometry::target())
    }

    pub(crate) fn materialize_with(
        self,
        geometry: &Qwen38FlashNextGeometry,
    ) -> CheckpointResult<MaterializedQwen38FlashNextSparseAttention<'a>> {
        let layer = self.layer;
        let hidden = geometry.hidden as u64;
        let query_rows = geometry.attention_query_rows;
        let kv_rows = geometry.attention_kv_rows;

        if self.query_gate_weight.shape() != &[query_rows as u64, hidden]
            || self.key_weight.shape() != &[kv_rows as u64, hidden]
            || self.value_weight.shape() != &[kv_rows as u64, hidden]
            || self.output_weight.shape() != &[hidden, geometry.attention_output_columns as u64]
            || self.query_norm.shape() != &[geometry.head_dim as u64]
            || self.key_norm.shape() != &[geometry.head_dim as u64]
            || self.indexer.qk_weight.shape() != &[geometry.indexer_rows as u64, hidden]
            || self.indexer.query_norm.shape() != &[geometry.indexer_head_dim as u64]
            || self.indexer.key_norm.shape() != &[geometry.indexer_head_dim as u64]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} Flash-Next sparse-attention source geometry differs from its contract"
            )));
        }

        let qkv_rows = checked(
            kv_rows
                .checked_mul(2)
                .and_then(|rows| rows.checked_add(query_rows)),
            layer,
            "sparse-attention QKV rows",
        )?;
        let qkv_weight_bf16 = PlaneGatherer::gather(
            [
                self.query_gate_weight.bytes(),
                self.key_weight.bytes(),
                self.value_weight.bytes(),
            ],
            &format!("layer-{layer} Flash-Next sparse-attention QKV weights"),
        )?;

        Ok(MaterializedQwen38FlashNextSparseAttention {
            qkv_weight_bf16,
            query_rows,
            kv_rows,
            qkv_rows,
            qkv_columns: geometry.hidden,
            output_weight: self.output_weight,
            query_norm: self.query_norm,
            key_norm: self.key_norm,
            indexer: self.indexer,
            layer,
        })
    }
}

/// Borrowed engram planes and admitted hash constants.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextEngram<'a> {
    /// The FP8 table shards in shard-index order, borrowed from the mapping.
    pub table_shards: Vec<&'a [u8]>,
    /// Rows in one shard: runtime row `r` is row `r % shard_rows` of shard `r / shard_rows`.
    pub shard_rows: usize,
    /// Width contributed by one engram head, and therefore one table row.
    pub head_dim: usize,
    /// Total borrowed table bytes, never staged on the host heap.
    pub table_bytes: usize,
    /// Exact source BF16 bits of the single table multiplier.
    pub table_scale_bits: u16,
    /// The table multiplier widened from its exact BF16 source word.
    pub table_scale: f32,
    /// The admitted hash law, equal by construction to the checkpoint's own buffers.
    pub constants: Qwen38FlashNextEngramHashConstants,
    /// Key projection retained zero-copy.
    pub key_proj_weight: Bf16View<'a, 2>,
    /// Value projection retained zero-copy.
    pub value_proj_weight: Bf16View<'a, 2>,
    /// Key-path grouped RMSNorm weights retained zero-copy.
    pub norm_key: Bf16View<'a, 1>,
    /// Query-path grouped RMSNorm weights retained zero-copy.
    pub norm_query: Bf16View<'a, 1>,
    /// Convolution-path grouped RMSNorm weights retained zero-copy.
    pub norm_conv: Bf16View<'a, 1>,
    /// Dilated depthwise short convolution retained zero-copy.
    pub convolution_weight: Bf16View<'a, 3>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl sealed::Sealed for MaterializedQwen38FlashNextEngram<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextEngram<'_> {
    fn host_bytes(&self) -> usize {
        size_of_val(self.table_shards.as_slice())
    }
}

impl MaterializedQwen38FlashNextEngram<'_> {
    /// Returns the borrowed table as a row-addressed view.
    pub fn table(&self) -> CheckpointResult<Qwen38FlashNextEngramTable<'_>> {
        Qwen38FlashNextEngramTable::new(
            &self.table_shards,
            self.shard_rows,
            self.head_dim,
            self.constants,
        )
    }
}

impl binding_sealed::Sealed for Qwen38FlashNextEngramBindings<'_> {}

impl<'a> SourceLayerBinding<'a, Qwen38FlashNext> for Qwen38FlashNextEngramBindings<'a> {
    type Materialized = MaterializedQwen38FlashNextEngram<'a>;

    fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Qwen38FlashNextEngramBindings::bind(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Qwen38FlashNextEngramBindings::materialize(self)
    }
}

impl<'a> Qwen38FlashNextEngramBindings<'a> {
    /// Admits the engram table's shard order and its single multiplier without copying a row.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextEngram<'a>> {
        self.materialize_with(&Qwen38FlashNextGeometry::target())
    }

    pub(crate) fn materialize_with(
        self,
        geometry: &Qwen38FlashNextGeometry,
    ) -> CheckpointResult<MaterializedQwen38FlashNextEngram<'a>> {
        let layer = self.layer;

        if self.table_shards.len() != geometry.ngram_shards {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} Flash-Next engram has {} table shards, expected {}",
                self.table_shards.len(),
                geometry.ngram_shards
            )));
        }

        let shard_bytes = checked(
            geometry
                .ngram_shard_rows
                .checked_mul(geometry.ngram_head_dim),
            layer,
            "engram shard bytes",
        )?;
        let table_bytes = checked(
            shard_bytes.checked_mul(geometry.ngram_shards),
            layer,
            "engram table bytes",
        )?;
        let mut table_shards = Vec::new();

        table_shards
            .try_reserve_exact(geometry.ngram_shards)
            .map_err(|_| {
                CheckpointError::source_binding(format!(
                    "layer-{layer} cannot reserve {} Flash-Next engram shard slices",
                    geometry.ngram_shards
                ))
            })?;

        for (shard, view) in self.table_shards.iter().enumerate() {
            if view.codes().len() != shard_bytes {
                return Err(CheckpointError::source_binding(format!(
                    "layer-{layer} Flash-Next engram shard {shard} holds {} bytes, expected {shard_bytes}",
                    view.codes().len()
                )));
            }

            table_shards.push(view.codes());
        }

        let table_scale_bits = self
            .table_scale
            .word(0)
            .expect("validated shape carries one word");
        let table_scale = f32::from_bits(u32::from(table_scale_bits) << 16);

        // This plain BF16 multiplier is outside the ModelOpt reciprocal convention.
        if !table_scale.is_finite() || table_scale <= 0.0 {
            return Err(CheckpointError::source_binding(format!(
                "layer-{layer} Flash-Next engram table scale must be finite and positive, observed {table_scale}"
            )));
        }

        Ok(MaterializedQwen38FlashNextEngram {
            table_shards,
            shard_rows: geometry.ngram_shard_rows,
            head_dim: geometry.ngram_head_dim,
            table_bytes,
            table_scale_bits,
            table_scale,
            constants: self.constants.constants,
            key_proj_weight: self.key_proj_weight,
            value_proj_weight: self.value_proj_weight,
            norm_key: self.norm_key,
            norm_query: self.norm_query,
            norm_conv: self.norm_conv,
            convolution_weight: self.convolution_weight,
            layer,
        })
    }
}

/// Borrowed hyper-connection planes for one decoder layer.
#[derive(Clone, Copy, Debug)]
pub struct MaterializedQwen38FlashNextHyperConnections<'a> {
    /// Gated residual read before the attention or GDN mixer.
    pub attention: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Gated residual read before the MoE block.
    pub mlp: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl sealed::Sealed for MaterializedQwen38FlashNextHyperConnections<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextHyperConnections<'_> {
    fn host_bytes(&self) -> usize {
        0
    }
}

impl binding_sealed::Sealed for Qwen38FlashNextLayerHyperConnections<'_> {}

impl<'a> SourceLayerBinding<'a, Qwen38FlashNext> for Qwen38FlashNextLayerHyperConnections<'a> {
    type Materialized = MaterializedQwen38FlashNextHyperConnections<'a>;

    fn bind(
        snapshot: &'a CheckpointSnapshot<Qwen38FlashNext>,
        layer: usize,
    ) -> CheckpointResult<Self> {
        Qwen38FlashNextLayerHyperConnections::bind(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Qwen38FlashNextLayerHyperConnections::materialize(self)
    }
}

impl<'a> Qwen38FlashNextLayerHyperConnections<'a> {
    /// Admits both gated residuals; nothing is reordered and nothing is staged.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextHyperConnections<'a>> {
        if self.attention.block_inject.is_none() || self.mlp.block_inject.is_none() {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} Flash-Next hyper-connections must both write back into the stream",
                self.layer
            )));
        }

        Ok(MaterializedQwen38FlashNextHyperConnections {
            attention: self.attention,
            mlp: self.mlp,
            layer: self.layer,
        })
    }
}

/// Runtime-native Flash-Next text endpoints, entirely borrowed.
#[derive(Clone, Copy, Debug)]
pub struct MaterializedQwen38FlashNextTextEndpoint<'a> {
    /// BF16 token embedding matrix retained zero-copy.
    pub embedding: Bf16View<'a, 2>,
    /// BF16 language-model head retained zero-copy.
    pub lm_head: Bf16View<'a, 2>,
    /// Model-level collapsing mixer, this target's only final normalization.
    pub mixer: Qwen38FlashNextHyperConnectionBindings<'a>,
}

impl sealed::Sealed for MaterializedQwen38FlashNextTextEndpoint<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextTextEndpoint<'_> {
    fn host_bytes(&self) -> usize {
        0
    }
}

impl<'a> Qwen38FlashNextTextEndpointBindings<'a> {
    /// Admits the endpoints; both matrices stay BF16 and stay mapped.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextTextEndpoint<'a>> {
        if self.mixer.block_inject.is_some() {
            return Err(CheckpointError::source_binding(
                "the Flash-Next model-level mixer collapses the stream and must not write back",
            ));
        }

        Ok(MaterializedQwen38FlashNextTextEndpoint {
            embedding: self.embedding,
            lm_head: self.lm_head,
            mixer: self.mixer,
        })
    }
}

/// Borrowed expert-major BF16 planes for the MTP routed pool.
#[derive(Clone, Copy, Debug)]
pub struct MaterializedQwen38FlashNextFusedExpertPool<'a> {
    /// Gate-over-up rows in expert order.
    pub gate_up: &'a [u8],
    /// Down-projection rows in expert order.
    pub down: &'a [u8],
    /// Routed experts in the pool.
    pub expert_count: usize,
    /// Bytes one expert occupies in `gate_up`.
    pub gate_up_stride_bytes: usize,
    /// Bytes one expert occupies in `down`.
    pub down_stride_bytes: usize,
}

impl<'a> MaterializedQwen38FlashNextFusedExpertPool<'a> {
    /// Bytes one expert occupies across both planes.
    pub fn expert_stride_bytes(&self) -> usize {
        self.gate_up_stride_bytes + self.down_stride_bytes
    }

    /// Bytes the borrowed pool occupies.
    pub fn borrowed_weight_bytes(&self) -> usize {
        self.gate_up.len() + self.down.len()
    }

    /// Returns both borrowed planes for one expert.
    pub fn expert(&self, expert: usize) -> Option<(&'a [u8], &'a [u8])> {
        if expert >= self.expert_count {
            return None;
        }

        let gate_up = self
            .gate_up
            .get(expert * self.gate_up_stride_bytes..(expert + 1) * self.gate_up_stride_bytes)?;
        let down = self
            .down
            .get(expert * self.down_stride_bytes..(expert + 1) * self.down_stride_bytes)?;

        Some((gate_up, down))
    }
}

/// Runtime-native MoE planes for one draft layer.
#[derive(Clone, Copy, Debug)]
pub struct MaterializedQwen38FlashNextMtpMoe<'a> {
    /// Full 512-way router weights retained zero-copy.
    pub router_weight: Bf16View<'a, 2>,
    /// Shared expert and its gate, all retained zero-copy.
    pub shared_expert: Qwen38FlashNextSharedExpertBindings<'a>,
    /// The fused BF16 routed pool.
    pub experts: MaterializedQwen38FlashNextFusedExpertPool<'a>,
}

/// Runtime-native planes for one draft decoder layer.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextMtpLayer<'a> {
    /// Gated residual read before the attention block.
    pub attention_hyper_connection: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Gated residual read before the MoE block.
    pub mlp_hyper_connection: Qwen38FlashNextHyperConnectionBindings<'a>,
    /// Sparse attention, fused exactly as a target layer's is.
    pub attention: MaterializedQwen38FlashNextSparseAttention<'a>,
    /// This layer's own MoE.
    pub mlp: MaterializedQwen38FlashNextMtpMoe<'a>,
}

/// Runtime layout for the MTP draft block.
///
/// Attention QKV is fused; the BF16 expert pool remains borrowed.
#[derive(Debug)]
pub struct MaterializedQwen38FlashNextMtp<'a> {
    /// RMSNorm weights over the fusion's embedding term, borrowed.
    pub pre_fc_norm_embedding: Bf16View<'a, 1>,
    /// Grouped RMSNorm weights over the target's pre-mixer stream, borrowed.
    pub pre_fc_norm_hidden: Bf16View<'a, 1>,
    /// Embedding-term input projection, borrowed.
    pub fc_embedding: Bf16View<'a, 2>,
    /// Stream-term input projection, borrowed.
    pub fc_hidden: Bf16View<'a, 2>,
    /// The draft's decoder layers, in stack order.
    pub layers: Vec<MaterializedQwen38FlashNextMtpLayer<'a>>,
    /// Collapse from the draft's own stream to the shared LM head, borrowed.
    pub mixer: Qwen38FlashNextHyperConnectionBindings<'a>,
}

impl sealed::Sealed for MaterializedQwen38FlashNextMtp<'_> {}

impl MaterializedMemory for MaterializedQwen38FlashNextMtp<'_> {
    fn host_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.attention.host_bytes())
            .sum()
    }
}

impl<'a> Qwen38FlashNextMtpBindings<'a> {
    /// Admits the draft block: fuses each layer's attention QKV, borrows everything else.
    pub fn materialize(self) -> CheckpointResult<MaterializedQwen38FlashNextMtp<'a>> {
        self.materialize_with(&Qwen38FlashNextGeometry::target())
    }

    pub(crate) fn materialize_with(
        self,
        geometry: &Qwen38FlashNextGeometry,
    ) -> CheckpointResult<MaterializedQwen38FlashNextMtp<'a>> {
        if self.pre_fc_norm_embedding.shape() != &[geometry.hidden as u64]
            || self.pre_fc_norm_hidden.shape() != &[geometry.hc_width as u64]
        {
            return Err(CheckpointError::source_binding(
                "the Flash-Next draft block's input-fusion norms differ from their contract",
            ));
        }

        if self.mixer.block_inject.is_some() {
            return Err(CheckpointError::source_binding(
                "the Flash-Next draft block's mixer collapses the stream and must not write back",
            ));
        }

        let mut layers = Vec::new();

        layers.try_reserve_exact(self.layers.len()).map_err(|_| {
            CheckpointError::source_binding(format!(
                "cannot reserve {} Flash-Next MTP layer layouts",
                self.layers.len()
            ))
        })?;

        for (layer, sources) in self.layers.into_iter().enumerate() {
            if sources.attention_hyper_connection.block_inject.is_none()
                || sources.mlp_hyper_connection.block_inject.is_none()
            {
                return Err(CheckpointError::source_binding(format!(
                    "mtp-layer-{layer} Flash-Next hyper-connections must both write back into \
                     the stream"
                )));
            }

            let experts = sources.mlp.experts;
            let expert_count = geometry.expert_count;
            let hidden = geometry.hidden as u64;
            let intermediate = geometry.expert_intermediate as u64;

            if experts.gate_up.shape() != &[expert_count as u64, 2 * intermediate, hidden]
                || experts.down.shape() != &[expert_count as u64, hidden, intermediate]
            {
                return Err(CheckpointError::source_binding(format!(
                    "mtp-layer-{layer} Flash-Next fused expert pool differs from its contract"
                )));
            }

            let gate_up_bytes = experts.gate_up.bytes();
            let down_bytes = experts.down.bytes();

            // Expert-major addressing requires exact plane divisibility.
            if expert_count == 0
                || gate_up_bytes.len() % expert_count != 0
                || down_bytes.len() % expert_count != 0
            {
                return Err(CheckpointError::source_binding(format!(
                    "mtp-layer-{layer} Flash-Next fused expert pool does not divide into \
                     {expert_count} experts"
                )));
            }

            layers.push(MaterializedQwen38FlashNextMtpLayer {
                attention_hyper_connection: sources.attention_hyper_connection,
                mlp_hyper_connection: sources.mlp_hyper_connection,
                attention: sources.attention.materialize_with(geometry)?,
                mlp: MaterializedQwen38FlashNextMtpMoe {
                    router_weight: sources.mlp.router_weight,
                    shared_expert: sources.mlp.shared_expert,
                    experts: MaterializedQwen38FlashNextFusedExpertPool {
                        gate_up: gate_up_bytes,
                        down: down_bytes,
                        expert_count,
                        gate_up_stride_bytes: gate_up_bytes.len() / expert_count,
                        down_stride_bytes: down_bytes.len() / expert_count,
                    },
                },
            });
        }

        Ok(MaterializedQwen38FlashNextMtp {
            pre_fc_norm_embedding: self.pre_fc_norm_embedding,
            pre_fc_norm_hidden: self.pre_fc_norm_hidden,
            fc_embedding: self.fc_embedding,
            fc_hidden: self.fc_hidden,
            layers,
            mixer: self.mixer,
        })
    }
}

fn checked(value: Option<usize>, layer: usize, role: &str) -> CheckpointResult<usize> {
    value.ok_or_else(|| {
        CheckpointError::source_binding(format!("layer-{layer} Flash-Next {role} overflows"))
    })
}

fn moe_error(layer: usize, message: impl Into<String>) -> CheckpointError {
    CheckpointError::source_binding(format!("layer-{layer} Flash-Next MoE {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::test_support::sources::{block_scale_oracle, fixture_path};
    use crate::qwen38_flash_next::bindings::tests::{
        endpoint_fixture, engram_fixture, engram_fixture_with, gdn_fixture,
        hyper_connection_fixture, moe_fixture, mtp_fixture, sparse_attention_fixture,
        test_geometry,
    };
    use crate::{CheckpointErrorCode, MaterializedMemory, SafeTensorFile};
    use std::fs;
    use std::path::Path;

    /// The swizzled scale bytes one test-geometry expert owns: a fused gate/up plane of
    /// `2 * 64` rows by 8 groups, then a down plane of 128 rows by 4 groups.
    const EXPERT_SCALE_BYTES: usize = 128 * 8 + 128 * 4;
    /// The packed E2M1 bytes one test-geometry expert owns across gate, up, and down.
    const EXPERT_WEIGHT_BYTES: usize = 64 * 64 + 64 * 64 + 128 * 32;

    #[test]
    fn expert_pool_borrows_every_packed_word_and_addresses_scales_expert_major() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-moe-materialize");
        moe_fixture(2, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextMoeBindings::bind_from(2, &geometry, |name| file.tensor(name)).unwrap();
        let sources = bindings
            .experts
            .iter()
            .map(|expert| {
                (
                    expert.gate.weight.bytes().as_ptr(),
                    expert.up.weight.bytes().as_ptr(),
                    expert.down.weight.bytes().as_ptr(),
                    expert.gate.block_scale.codes().to_vec(),
                    expert.up.block_scale.codes().to_vec(),
                    expert.down.block_scale.codes().to_vec(),
                )
            })
            .collect::<Vec<_>>();

        let materialized = bindings.materialize_with(&geometry).unwrap();
        let pool = &materialized.experts;

        assert_eq!(pool.expert_count, 3);
        assert_eq!(pool.expert_scale_bytes, EXPERT_SCALE_BYTES);
        assert_eq!(pool.expert_weight_bytes, EXPERT_WEIGHT_BYTES);
        assert_eq!(pool.scale_e4m3_swizzled.len(), 3 * EXPERT_SCALE_BYTES);
        assert_eq!(pool.borrowed_weight_bytes(), 3 * EXPERT_WEIGHT_BYTES);

        for (expert, source) in sources.iter().enumerate() {
            let planes = &pool.experts[expert];

            // Zero-copy: the materialized packed planes are the mapped source bytes, not a
            // staged copy. This is the whole reason a 56.25 GiB expert pool is admissible.
            assert_eq!(planes.expert, expert);
            assert_eq!(planes.gate_weight_e2m1.as_ptr(), source.0);
            assert_eq!(planes.up_weight_e2m1.as_ptr(), source.1);
            assert_eq!(planes.down_weight_e2m1.as_ptr(), source.2);

            // Expert-major addressing in numeric order, with the fused gate/up plane first.
            let base = expert * EXPERT_SCALE_BYTES;
            assert_eq!(
                planes.gate_up_scale,
                Qwen38FlashNextPlaneExtent {
                    offset: base,
                    bytes: 128 * 8
                }
            );
            assert_eq!(
                planes.down_scale,
                Qwen38FlashNextPlaneExtent {
                    offset: base + 128 * 8,
                    bytes: 128 * 4
                }
            );

            // The staged scales are a byte permutation of the source planes and nothing else.
            let fused = [source.3.as_slice(), source.4.as_slice()].concat();
            let gate_up = &pool.scale_e4m3_swizzled[planes.gate_up_scale.offset
                ..planes.gate_up_scale.offset + planes.gate_up_scale.bytes];
            let down = &pool.scale_e4m3_swizzled
                [planes.down_scale.offset..planes.down_scale.offset + planes.down_scale.bytes];

            assert_eq!(gate_up, block_scale_oracle(&fused, 128, 8));
            assert_eq!(down, block_scale_oracle(&source.5, 128, 4));
        }

        // Accounting: the staged pool and its descriptors, never the borrowed weights.
        let descriptors = size_of::<MaterializedQwen38FlashNextExpert<'_>>() * 3;
        assert_eq!(
            materialized.host_bytes(),
            3 * EXPERT_SCALE_BYTES + descriptors
        );
        assert!(
            materialized.host_bytes() < pool.borrowed_weight_bytes(),
            "the borrowed expert pool must never be counted as staged host memory"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_pool_converts_only_the_modelopt_scalar_convention() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-moe-scalars");
        moe_fixture(0, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();

        let materialized =
            Qwen38FlashNextMoeBindings::bind_from(0, &geometry, |name| file.tensor(name))
                .unwrap()
                .materialize_with(&geometry)
                .unwrap();

        for planes in &materialized.experts.experts {
            // The fixture stores 0.25 and 0.125; the kernels consume their reciprocals.
            assert_eq!(planes.gate_up_input_scale.to_bits(), 0.25f32.to_bits());
            assert_eq!(planes.gate_up_weight_scale_2.to_bits(), 0.125f32.to_bits());
            assert_eq!(
                planes.gate_up_input_scale_divisor.to_bits(),
                4.0f32.to_bits()
            );
            assert_eq!(
                planes.gate_up_weight_scale_divisor.to_bits(),
                8.0f32.to_bits()
            );
            assert_eq!(planes.down_input_scale.to_bits(), 0.25f32.to_bits());
            assert_eq!(planes.down_input_scale_divisor.to_bits(), 4.0f32.to_bits());
        }

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_pool_refuses_a_short_or_reordered_expert_source() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-moe-short");
        moe_fixture(0, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextMoeBindings::bind_from(0, &geometry, |name| file.tensor(name)).unwrap();

        let mut short = bindings.clone();
        short.experts.pop();
        let error = short.materialize_with(&geometry).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error.to_string().contains("2 routed experts, expected 3"),
            "{error}"
        );

        // Numeric order is what the pool's extents mean, so a swapped pair must be refused
        // rather than silently naming the wrong expert's scales.
        let mut swapped = bindings;
        swapped.experts.swap(0, 2);
        let error = swapped.materialize_with(&geometry).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("routed expert 2 is at pool position 0"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gdn_materialization_gathers_exact_source_words() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-gdn-materialize");
        gdn_fixture(0, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextGdnBindings::bind_from(0, &geometry, |name| file.tensor(name)).unwrap();
        let qkv = bindings.qkv_weight.bytes().to_vec();
        let z = bindings.z_weight.bytes().to_vec();
        let a = bindings.a_control_weight.bytes().to_vec();
        let b = bindings.b_control_weight.bytes().to_vec();
        let output = bindings.output_weight.bytes().as_ptr();

        let materialized = bindings.materialize_with(&geometry).unwrap();

        assert_eq!(&materialized.input_weight_bf16[..qkv.len()], qkv);
        assert_eq!(&materialized.input_weight_bf16[qkv.len()..], z);
        assert_eq!(&materialized.control_weight_bf16[..a.len()], a);
        assert_eq!(&materialized.control_weight_bf16[a.len()..], b);
        assert_eq!((materialized.qkv_rows, materialized.input_rows), (20, 32));
        assert_eq!(materialized.input_columns, 128);
        assert_eq!(materialized.control_rows_per_projection, 3);
        assert_eq!(materialized.output_weight.bytes().as_ptr(), output);
        assert_eq!(
            materialized.host_bytes(),
            qkv.len() + z.len() + a.len() + b.len()
        );
        assert_eq!(materialized.layer, 0);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sparse_attention_materialization_gathers_exact_source_words_and_keeps_the_indexer_separate()
    {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-sparse_attention-materialize");
        sparse_attention_fixture(3, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings = Qwen38FlashNextSparseAttentionBindings::bind_from(3, &geometry, |name| {
            file.tensor(name)
        })
        .unwrap();
        let query = bindings.query_gate_weight.bytes().to_vec();
        let key = bindings.key_weight.bytes().to_vec();
        let value = bindings.value_weight.bytes().to_vec();
        let indexer = bindings.indexer.qk_weight.bytes().as_ptr();

        let materialized = bindings.materialize_with(&geometry).unwrap();
        let key_end = query.len() + key.len();

        assert_eq!(&materialized.qkv_weight_bf16[..query.len()], query);
        assert_eq!(&materialized.qkv_weight_bf16[query.len()..key_end], key);
        assert_eq!(&materialized.qkv_weight_bf16[key_end..], value);
        assert_eq!(
            (
                materialized.query_rows,
                materialized.kv_rows,
                materialized.qkv_rows
            ),
            (16, 8, 32)
        );
        assert_eq!(materialized.qkv_columns, 128);
        assert_eq!(
            materialized.indexer.qk_weight.bytes().as_ptr(),
            indexer,
            "the indexer projection stays separate and stays borrowed"
        );
        assert_eq!(
            materialized.host_bytes(),
            query.len() + key.len() + value.len()
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn engram_materialization_borrows_the_whole_table() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-engram-materialize");
        engram_fixture(geometry.ple_layer, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextEngramBindings::bind_from(geometry.ple_layer, &geometry, |name| {
                file.tensor(name)
            })
            .unwrap();
        let shard_pointers = bindings
            .table_shards
            .iter()
            .map(|view| view.codes().as_ptr())
            .collect::<Vec<_>>();

        let materialized = bindings.materialize_with(&geometry).unwrap();

        assert_eq!(materialized.table_shards.len(), 4);
        assert_eq!(materialized.shard_rows, 6);
        assert_eq!(materialized.head_dim, 2);
        assert_eq!(materialized.table_bytes, 4 * 6 * 2);

        for (shard, pointer) in shard_pointers.iter().enumerate() {
            assert_eq!(
                materialized.table_shards[shard].as_ptr(),
                *pointer,
                "shard {shard} must stay mapped, in shard-index order"
            );
        }

        // The single BF16 multiplier survives as its exact source word; it is not a ModelOpt
        // scale and is never converted to a reciprocal divisor.
        assert_eq!(materialized.table_scale_bits, 0x3951);
        assert_eq!(
            materialized.table_scale.to_bits(),
            0x3951_0000,
            "the multiplier is the BF16 source word widened, nothing else"
        );
        assert_eq!(
            materialized.constants,
            Qwen38FlashNextEngramHashConstants::compute().unwrap()
        );

        // Only the 128 shard slices are staged; the table itself never is.
        assert_eq!(
            materialized.host_bytes(),
            size_of::<&[u8]>() * 4,
            "the engram stages shard descriptors only"
        );

        fs::remove_file(path).unwrap();
    }

    /// Sentinel byte the gather fixture writes at byte `byte` of shard `shard`.
    ///
    /// Distinct for every (shard, row, byte) triple in the test geometry, so a gathered byte
    /// names the exact table address it came from.
    fn engram_sentinel(shard: usize, byte: usize) -> u8 {
        u8::try_from(shard * 0x40 + byte).unwrap()
    }

    #[test]
    fn engram_gather_reads_exact_rows_head_major_from_the_borrowed_shards() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-engram-gather");
        engram_fixture_with(geometry.ple_layer, &geometry, engram_sentinel).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let materialized =
            Qwen38FlashNextEngramBindings::bind_from(geometry.ple_layer, &geometry, |name| {
                file.tensor(name)
            })
            .unwrap()
            .materialize_with(&geometry)
            .unwrap();

        let table = materialized.table().unwrap();

        assert_eq!(table.heads(), 16);
        assert_eq!(table.table_rows(), 24);
        assert_eq!(table.token_bytes(), 32);

        // Sixteen rows in head order, deliberately spread across all four shards and revisiting
        // one row, exactly as a hashed token's rows may.
        let rows: [i64; 16] = [0, 5, 6, 11, 12, 17, 18, 23, 1, 7, 13, 19, 2, 2, 22, 3];
        let mut staged = vec![0u8; rows.len() * geometry.ngram_head_dim];

        table.gather_rows(&rows, &mut staged).unwrap();

        for (head, row) in rows.iter().copied().enumerate() {
            let shard = row as usize / geometry.ngram_shard_rows;
            let shard_row = row as usize % geometry.ngram_shard_rows;

            for byte in 0..geometry.ngram_head_dim {
                assert_eq!(
                    staged[head * geometry.ngram_head_dim + byte],
                    engram_sentinel(shard, shard_row * geometry.ngram_head_dim + byte),
                    "head {head} row {row} byte {byte}"
                );
            }
        }

        // Every row's codes are the mapping's own bytes, not a copy the gather owns.
        assert_eq!(
            table.row_codes(7).unwrap().as_ptr(),
            materialized.table_shards[1][2..].as_ptr()
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_an_engram_row_or_destination_the_table_cannot_address() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38_flash_next-engram-gather-range");
        engram_fixture(geometry.ple_layer, &geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let materialized =
            Qwen38FlashNextEngramBindings::bind_from(geometry.ple_layer, &geometry, |name| {
                file.tensor(name)
            })
            .unwrap()
            .materialize_with(&geometry)
            .unwrap();

        let table = materialized.table().unwrap();

        for row in [-1, 24, i64::MAX, i64::MIN] {
            let error = table.row_codes(row).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("is outside 0..24"), "{error}");
        }

        let error = table.gather_rows(&[0, 1], &mut [0u8; 3]).err().unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error.to_string().contains("expected 4 for 2 rows"),
            "{error}"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn hyper_connections_and_endpoints_stage_nothing() {
        let geometry = test_geometry();
        let hyper_path = fixture_path("qwen38_flash_next-hc-materialize");
        hyper_connection_fixture(1, &geometry).write(&hyper_path);
        let hyper_file = SafeTensorFile::open(&hyper_path).unwrap();
        let hyper = Qwen38FlashNextLayerHyperConnections::bind_from(1, &geometry, |name| {
            hyper_file.tensor(name)
        })
        .unwrap()
        .materialize()
        .unwrap();

        assert_eq!(hyper.host_bytes(), 0);
        assert!(hyper.attention.block_inject.is_some());
        assert!(hyper.mlp.block_inject.is_some());
        assert_eq!(hyper.layer, 1);

        let endpoint_path = fixture_path("qwen38_flash_next-endpoints-materialize");
        endpoint_fixture(&geometry).write(&endpoint_path);
        let endpoint_file = SafeTensorFile::open(&endpoint_path).unwrap();
        let bindings = Qwen38FlashNextTextEndpointBindings::bind_from(&geometry, |name| {
            endpoint_file.tensor(name)
        })
        .unwrap();
        let embedding = bindings.embedding.bytes().as_ptr();
        let endpoints = bindings.materialize().unwrap();

        assert_eq!(endpoints.host_bytes(), 0);
        assert_eq!(endpoints.embedding.bytes().as_ptr(), embedding);
        assert!(
            endpoints.mixer.block_inject.is_none(),
            "the collapsing mixer must not write back into the stream"
        );

        fs::remove_file(hyper_path).unwrap();
        fs::remove_file(endpoint_path).unwrap();
    }

    #[test]
    fn every_admitted_qwen38_flash_next_layer_binding_shares_the_two_phase_lifecycle() {
        fn admits<'a, B: SourceLayerBinding<'a, Qwen38FlashNext>>() {}

        admits::<Qwen38FlashNextGdnBindings<'_>>();
        admits::<Qwen38FlashNextSparseAttentionBindings<'_>>();
        admits::<Qwen38FlashNextMoeBindings<'_>>();
        admits::<Qwen38FlashNextEngramBindings<'_>>();
        admits::<Qwen38FlashNextLayerHyperConnections<'_>>();
    }

    /// Pins whole-model borrowed and staged byte accounting.
    #[test]
    fn whole_model_staging_and_borrowed_totals_close() {
        let geometry = Qwen38FlashNextGeometry::target();
        let gdn_layers = geometry.layers - geometry.layers / geometry.full_attention_interval;
        let sparse_attention_layers = geometry.layers / geometry.full_attention_interval;
        let expert_scale_bytes = 2 * geometry.expert_intermediate * (geometry.hidden / 16)
            + geometry.hidden * (geometry.expert_intermediate / 16);
        let expert_weight_bytes = 2 * geometry.expert_intermediate * (geometry.hidden / 2)
            + geometry.hidden * (geometry.expert_intermediate / 2);
        let layer_scale_pool = expert_scale_bytes * geometry.expert_count;
        let gdn_staged = (geometry.gdn_qkv_rows + geometry.gdn_value_rows) * geometry.hidden * 2
            + 2 * geometry.gdn_control_rows * geometry.hidden * 2;
        let sparse_attention_staged =
            (geometry.attention_query_rows + 2 * geometry.attention_kv_rows) * geometry.hidden * 2;

        assert_eq!((gdn_layers, sparse_attention_layers), (36, 12));
        assert_eq!(expert_scale_bytes, 307_200);
        assert_eq!(expert_weight_bytes, 2_457_600);
        assert_eq!(layer_scale_pool, 157_286_400);
        assert_eq!(gdn_staged, 84_377_600);
        assert_eq!(sparse_attention_staged, 68_157_440);

        // Staged if every layer is materialized at once: 10.62 GiB.
        let staged = layer_scale_pool * geometry.layers
            + gdn_staged * gdn_layers
            + sparse_attention_staged * sparse_attention_layers;
        assert_eq!(staged, 11_405_230_080);

        // Borrowed and never staged: the routed expert pool and the engram table.
        let expert_pool = expert_weight_bytes * geometry.expert_count * geometry.layers;
        let engram_table =
            geometry.ngram_shards * geometry.ngram_shard_rows * geometry.ngram_head_dim;

        assert_eq!(expert_pool, 60_397_977_600);
        assert_eq!(engram_table, 51_200_245_760);
        assert!(
            staged * 9 < expert_pool + engram_table,
            "borrowing must dominate: a host that staged either structure would not fit"
        );
    }

    #[test]
    fn qwen38_flash_next_mtp_block_binds_its_fused_bf16_expert_pool() {
        let geometry = test_geometry();
        let path = fixture_path("qwen38-flash-next-mtp-materialize");
        mtp_fixture(&geometry).write(&path);
        let file = SafeTensorFile::open(&path).unwrap();
        let bindings =
            Qwen38FlashNextMtpBindings::bind_from(1, &geometry, |name| file.tensor(name)).unwrap();
        let pool_sources = (
            bindings.layers[0].mlp.experts.gate_up.bytes().as_ptr(),
            bindings.layers[0].mlp.experts.down.bytes().as_ptr(),
        );
        let mtp = bindings.materialize_with(&geometry).unwrap();

        assert_eq!(mtp.layers.len(), 1);

        let pool = mtp.layers[0].mlp.experts;
        let gate_up_stride = 2 * geometry.expert_intermediate * geometry.hidden * 2;
        let down_stride = geometry.hidden * geometry.expert_intermediate * 2;

        assert_eq!(pool.expert_count, geometry.expert_count);
        assert_eq!(pool.gate_up_stride_bytes, gate_up_stride);
        assert_eq!(pool.down_stride_bytes, down_stride);
        assert_eq!(pool.expert_stride_bytes(), gate_up_stride + down_stride);
        assert_eq!(
            pool.borrowed_weight_bytes(),
            geometry.expert_count * (gate_up_stride + down_stride)
        );

        for expert in 0..geometry.expert_count {
            let (gate_up, down) = pool.expert(expert).unwrap();

            assert_eq!(gate_up.len(), gate_up_stride);
            assert_eq!(down.len(), down_stride);
        }

        assert!(pool.expert(geometry.expert_count).is_none());

        assert_eq!(
            (pool.gate_up.as_ptr(), pool.down.as_ptr()),
            pool_sources,
            "the fused pool must be borrowed, never staged"
        );

        assert_eq!(
            mtp.host_bytes(),
            (geometry.attention_query_rows + 2 * geometry.attention_kv_rows) * geometry.hidden * 2
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn qwen38_flash_next_mtp_pool_strides_match_the_target_budget() {
        let geometry = Qwen38FlashNextGeometry::target();
        let gate_up_stride = 2 * geometry.expert_intermediate * geometry.hidden * 2;
        let down_stride = geometry.hidden * geometry.expert_intermediate * 2;

        assert_eq!((gate_up_stride, down_stride), (6_553_600, 3_276_800));
        assert_eq!(gate_up_stride + down_stride, 9_830_400);
        assert_eq!(
            geometry.expert_count * (gate_up_stride + down_stride),
            5_033_164_800
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete Flash-Next checkpoint"]
    fn qwen38_flash_next_snapshot_layers_materialize_losslessly() {
        let root = std::env::var("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").unwrap();
        let snapshot = CheckpointSnapshot::<Qwen38FlashNext>::open(Path::new(&root)).unwrap();
        let geometry = Qwen38FlashNextGeometry::target();

        let endpoints = Qwen38FlashNextTextEndpointBindings::bind(&snapshot)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(endpoints.embedding.shape(), &[248_320, 2_560]);
        assert_eq!(endpoints.lm_head.shape(), &[248_320, 2_560]);
        assert_eq!(endpoints.host_bytes(), 0);

        let engram = Qwen38FlashNextEngramBindings::bind(&snapshot, geometry.ple_layer)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(engram.table_shards.len(), 128);
        assert_eq!(engram.table_bytes, 51_200_245_760);
        assert_eq!(engram.constants.padded_rows(), 320_001_536);
        assert_eq!(engram.host_bytes(), size_of::<&[u8]>() * 128);

        let hyper = Qwen38FlashNextLayerHyperConnections::bind(&snapshot, 0)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(hyper.host_bytes(), 0);

        // One GDN, one sparse-attention, and one MoE layer: a full pool is 150 MiB staged and
        // 1.17 GiB borrowed, so the gate materializes one layer rather than the stack.
        let gdn = Qwen38FlashNextGdnBindings::bind(&snapshot, 0)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!((gdn.qkv_rows, gdn.input_rows), (10_240, 16_384));
        assert_eq!(gdn.host_bytes(), 84_377_600);

        let sparse_attention = Qwen38FlashNextSparseAttentionBindings::bind(&snapshot, 3)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(sparse_attention.qkv_rows, 13_312);
        assert_eq!(sparse_attention.host_bytes(), 68_157_440);

        let moe = Qwen38FlashNextMoeBindings::bind(&snapshot, 0)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(moe.experts.expert_count, 512);
        assert_eq!(moe.experts.expert_scale_bytes, 307_200);
        assert_eq!(moe.experts.expert_weight_bytes, 2_457_600);
        assert_eq!(moe.experts.borrowed_weight_bytes(), 1_258_291_200);
        assert_eq!(
            moe.host_bytes(),
            157_286_400 + 512 * size_of::<MaterializedQwen38FlashNextExpert<'_>>()
        );

        for (expert, planes) in moe.experts.experts.iter().enumerate() {
            assert_eq!(planes.expert, expert);
            assert_eq!(planes.gate_up_scale.offset, expert * 307_200);
            assert_eq!(planes.gate_weight_e2m1.len(), 819_200);
            assert_eq!(planes.down_weight_e2m1.len(), 819_200);
        }

        let mtp = Qwen38FlashNextMtpBindings::bind(&snapshot)
            .unwrap()
            .materialize()
            .unwrap();

        assert_eq!(mtp.layers.len(), 1);
        assert_eq!(mtp.host_bytes(), 68_157_440);
        assert_eq!(mtp.layers[0].mlp.experts.expert_count, 512);
        assert_eq!(
            mtp.layers[0].mlp.experts.borrowed_weight_bytes(),
            5_033_164_800
        );

        // The vision tower is bind-only: admitted, never executed.
        let vision = crate::VisionBindings::bind::<Qwen38FlashNext>(&snapshot).unwrap();

        assert_eq!(vision.blocks.len(), 27);
    }
}

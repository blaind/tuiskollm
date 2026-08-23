//! Lossless conversion from source bindings to runtime-native host layouts.

use crate::bindings::{
    NVFP4_MLP_LAYER_END, require_full_attention_layer, require_gdn_layer_route,
    require_nvfp4_mlp_layer, validate_nvfp4_scales,
};
use crate::{
    Bf16View, CheckpointError, CheckpointResult, F32View, FullAttentionQkvBindings,
    ModelOptNvfp4AttentionBindings, ModelOptNvfp4GdnBindings, ModelOptNvfp4LinearBindings,
    ModelOptNvfp4MlpBindings, MtpBindings, Nvfp4DownBindings, Nvfp4GateUpBindings,
};
use rayon::prelude::*;
use std::sync::OnceLock;

const SCALE_TILE_ROWS: usize = 128;
const SCALE_TILE_GROUPS: usize = 4;
const SCALE_TILE_BYTES: usize = SCALE_TILE_ROWS * SCALE_TILE_GROUPS;
const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;
const PARALLEL_SWIZZLE_MIN_BYTES: usize = 1 << 20;
const PARALLEL_GATHER_MIN_BYTES: usize = 1 << 20;
const MAX_MATERIALIZATION_WORKERS: usize = 16;

static MATERIALIZATION_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

/// Worker bound used by target-size NVFP4 scale materialization.
pub fn nvfp4_scale_materialization_workers() -> usize {
    materialization_workers()
}

fn materialization_workers() -> usize {
    std::thread::available_parallelism()
        .map(|workers| workers.get())
        .unwrap_or(1)
        .min(MAX_MATERIALIZATION_WORKERS)
}

/// Runtime-native fused QKV planes in query/gate, key, value row order.
#[derive(Debug)]
pub struct MaterializedFullAttentionQkv {
    /// Losslessly gathered E4M3 weights `[rows, columns]`.
    pub weight_e4m3: Vec<u8>,
    /// Losslessly gathered little-endian BF16 row scales `[rows, 1]`.
    pub scale_bf16: Vec<u8>,
    /// Fused query/gate, key, and value row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl FullAttentionQkvBindings<'_> {
    /// Gathers the non-contiguous source planes without requantizing represented values.
    pub fn materialize(self) -> CheckpointResult<MaterializedFullAttentionQkv> {
        require_full_attention_layer(self.layer, self.layer_count, self.full_attention_interval)?;

        let [query_rows, columns] = host_shape(
            self.query_gate_weight.shape(),
            "full-attention query/gate weights",
        )?;
        let [key_rows, key_columns] =
            host_shape(self.key_weight.shape(), "full-attention key weights")?;
        let [value_rows, value_columns] =
            host_shape(self.value_weight.shape(), "full-attention value weights")?;

        let query_scale_shape = host_shape(
            self.query_gate_scale.shape(),
            "full-attention query/gate scales",
        )?;
        let key_scale_shape = host_shape(self.key_scale.shape(), "full-attention key scales")?;
        let value_scale_shape =
            host_shape(self.value_scale.shape(), "full-attention value scales")?;

        if key_rows != value_rows
            || columns != key_columns
            || columns != value_columns
            || query_scale_shape != [query_rows, 1]
            || key_scale_shape != [key_rows, 1]
            || value_scale_shape != [value_rows, 1]
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} full-attention QKV source planes have incompatible shapes",
                self.layer
            )));
        }

        let rows = query_rows
            .checked_add(key_rows)
            .and_then(|rows| rows.checked_add(value_rows))
            .ok_or_else(|| {
                CheckpointError::source_binding(format!(
                    "layer-{} full-attention QKV row count overflows",
                    self.layer
                ))
            })?;

        let weight_e4m3 = gather_source_planes(
            [
                self.query_gate_weight.codes(),
                self.key_weight.codes(),
                self.value_weight.codes(),
            ],
            &format!("layer-{} full-attention QKV weights", self.layer),
        )?;
        let scale_bf16 = gather_source_planes(
            [
                self.query_gate_scale.bytes(),
                self.key_scale.bytes(),
                self.value_scale.bytes(),
            ],
            &format!("layer-{} full-attention QKV scales", self.layer),
        )?;

        Ok(MaterializedFullAttentionQkv {
            weight_e4m3,
            scale_bf16,
            rows,
            columns,
            layer: self.layer,
        })
    }
}

/// Runtime-native fused BF16 MTP QKV plane in query/gate, key, value row order.
#[derive(Debug)]
pub struct MaterializedMtpQkv {
    /// Losslessly gathered little-endian BF16 weights `[rows, columns]`.
    pub weight_bf16: Vec<u8>,
    /// Fused query/gate, key, and value row count.
    pub rows: usize,
    /// Logical input width.
    pub columns: usize,
}

impl MtpBindings<'_> {
    /// Gathers the non-contiguous draft QKV planes without changing BF16 words.
    pub fn materialize_qkv(&self) -> CheckpointResult<MaterializedMtpQkv> {
        let [query_rows, columns] =
            host_shape(self.query_gate_weight.shape(), "MTP query/gate weights")?;
        let [key_rows, key_columns] = host_shape(self.key_weight.shape(), "MTP key weights")?;
        let [value_rows, value_columns] =
            host_shape(self.value_weight.shape(), "MTP value weights")?;

        if key_rows != value_rows || columns != key_columns || columns != value_columns {
            return Err(CheckpointError::source_binding(
                "MTP QKV source planes have incompatible shapes",
            ));
        }

        let rows = query_rows
            .checked_add(key_rows)
            .and_then(|rows| rows.checked_add(value_rows))
            .ok_or_else(|| CheckpointError::source_binding("MTP QKV row count overflows"))?;

        let weight_bf16 = gather_source_planes(
            [
                self.query_gate_weight.bytes(),
                self.key_weight.bytes(),
                self.value_weight.bytes(),
            ],
            "MTP QKV weights",
        )?;

        Ok(MaterializedMtpQkv {
            weight_bf16,
            rows,
            columns,
        })
    }
}

/// Runtime-native NVFP4 gate/up layout with source packed weights retained zero-copy.
#[derive(Debug)]
pub struct MaterializedNvfp4GateUp<'a> {
    /// Packed gate E2M1 source words.
    pub gate_weight_e2m1: &'a [u8],
    /// Packed up E2M1 source words.
    pub up_weight_e2m1: &'a [u8],
    /// Losslessly permuted `BlockScaleK16M128x4` scale plane.
    pub scale_e4m3_swizzled: Vec<u8>,
    /// Shared finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Shared finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Fused gate/up row count.
    pub rows: usize,
    /// Logical input width before E2M1 packing.
    pub columns: usize,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl<'a> Nvfp4GateUpBindings<'a> {
    /// Materializes the fused gate/up scale layout without requantizing source values.
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4GateUp<'a>> {
        require_nvfp4_mlp_layer(self.layer, NVFP4_MLP_LAYER_END)?;

        let [gate_rows, packed_columns] =
            host_shape(self.gate_weight.shape(), "NVFP4 gate weights")?;
        let up_shape = host_shape(self.up_weight.shape(), "NVFP4 up weights")?;
        let [gate_scale_rows, groups] = host_shape(self.gate_scale.shape(), "NVFP4 gate scales")?;
        let up_scale_shape = host_shape(self.up_scale.shape(), "NVFP4 up scales")?;

        if [gate_rows, packed_columns] != up_shape
            || [gate_scale_rows, groups] != up_scale_shape
            || gate_rows != gate_scale_rows
        {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} NVFP4 gate/up source planes have incompatible shapes",
                self.layer
            )));
        }

        let rows = gate_rows.checked_mul(2).ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{} NVFP4 gate/up row count overflows",
                self.layer
            ))
        })?;
        let columns = logical_columns(packed_columns, groups, self.layer, "gate/up")?;

        validate_nvfp4_scales(self.layer, "gate", self.gate_scale.codes())?;
        validate_nvfp4_scales(self.layer, "up", self.up_scale.codes())?;
        validate_divisor(self.layer, "gate/up input", self.input_scale_divisor)?;
        validate_divisor(self.layer, "gate/up weight", self.weight_scale_divisor)?;

        let scale_e4m3_swizzled = swizzle_scale_planes(
            &[self.gate_scale.codes(), self.up_scale.codes()],
            gate_rows,
            groups,
            self.layer,
            "gate/up",
        )?;

        Ok(MaterializedNvfp4GateUp {
            gate_weight_e2m1: self.gate_weight.bytes(),
            up_weight_e2m1: self.up_weight.bytes(),
            scale_e4m3_swizzled,
            input_scale_divisor: self.input_scale_divisor,
            weight_scale_divisor: self.weight_scale_divisor,
            rows,
            columns,
            layer: self.layer,
        })
    }
}

/// Runtime-native NVFP4 down layout with source packed weights retained zero-copy.
#[derive(Debug)]
pub struct MaterializedNvfp4Down<'a> {
    /// Packed E2M1 source words.
    pub weight_e2m1: &'a [u8],
    /// Losslessly permuted `BlockScaleK16M128x4` scale plane.
    pub scale_e4m3_swizzled: Vec<u8>,
    /// Finite positive activation-scale divisor.
    pub input_scale_divisor: f32,
    /// Finite positive weight-scale divisor.
    pub weight_scale_divisor: f32,
    /// Output row count.
    pub rows: usize,
    /// Logical input width before E2M1 packing.
    pub columns: usize,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

/// Runtime-native Qwen3.5 MLP planes derived losslessly from ModelOpt NVFP4 sources.
#[derive(Debug)]
pub struct MaterializedModelOptNvfp4Mlp<'a> {
    /// Fused gate/up runtime layout consumed by the qualified NVFP4 SwiGLU route.
    pub gate_up: MaterializedNvfp4GateUp<'a>,
    /// Down-projection runtime layout consumed by the qualified NVFP4 down route.
    pub down: MaterializedNvfp4Down<'a>,
    /// Exact source activation scale shared by gate and up.
    pub gate_up_input_scale: f32,
    /// Exact source second-stage weight scale shared by gate and up.
    pub gate_up_weight_scale_2: f32,
    /// Exact source down-projection activation scale.
    pub down_input_scale: f32,
    /// Exact source down-projection second-stage weight scale.
    pub down_weight_scale_2: f32,
    /// Zero-centered RMSNorm weights before the MLP.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights for the next decoder boundary.
    pub next_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

/// One runtime-native ModelOpt NVFP4 projection.
#[derive(Debug)]
pub struct MaterializedModelOptNvfp4Linear<'a> {
    /// Packed E2M1 source words.
    pub weight_e2m1: &'a [u8],
    /// Losslessly permuted `BlockScaleK16M128x4` scale plane.
    pub scale_e4m3_swizzled: Vec<u8>,
    /// Exact source activation scale.
    pub input_scale: f32,
    /// Exact source second-stage weight scale.
    pub weight_scale_2: f32,
    /// Reciprocal activation-scale convention consumed by the kernels.
    pub input_scale_divisor: f32,
    /// Reciprocal weight-scale convention consumed by the kernels.
    pub weight_scale_divisor: f32,
    /// Output row count.
    pub rows: usize,
    /// Logical input width before E2M1 packing.
    pub columns: usize,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

/// Runtime-native Qwen3.5 full-attention planes.
#[derive(Debug)]
pub struct MaterializedModelOptNvfp4Attention<'a> {
    /// Fused packed Q/gate, K, and V words in projection-row order.
    pub qkv_weight_e2m1: Vec<u8>,
    /// Fused swizzled Q/gate, K, and V block scales in the same row order.
    pub qkv_scale_e4m3_swizzled: Vec<u8>,
    /// Exact source activation scale shared by Q, K, and V.
    pub qkv_input_scale: f32,
    /// Exact per-projection second-stage weight scales in Q, K, V order.
    pub qkv_weight_scales_2: [f32; 3],
    /// Reciprocal activation-scale convention consumed by the kernels.
    pub qkv_input_scale_divisor: f32,
    /// Reciprocal weight-scale conventions in Q, K, V order.
    pub qkv_weight_scale_divisors: [f32; 3],
    /// Fused Q/gate, K, and V row count.
    pub qkv_rows: usize,
    /// Logical Q/K/V input width.
    pub qkv_columns: usize,
    /// Gated attention-output projection.
    pub output: MaterializedModelOptNvfp4Linear<'a>,
    /// Per-head query RMSNorm weights.
    pub query_norm: Bf16View<'a, 1>,
    /// Per-head key RMSNorm weights.
    pub key_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before attention.
    pub input_norm: Bf16View<'a, 1>,
    /// Zero-centered RMSNorm weights before the MLP.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

/// Runtime-native Qwen3.5 GDN source planes.
#[derive(Debug)]
pub struct MaterializedModelOptNvfp4Gdn<'a> {
    /// Fused packed Q/K/V/Z words in projection-row order.
    pub input_weight_e2m1: Vec<u8>,
    /// Fused swizzled Q/K/V/Z block scales in the same row order.
    pub input_scale_e4m3_swizzled: Vec<u8>,
    /// Exact source activation scale shared by Q/K/V/Z.
    pub input_scale: f32,
    /// Exact source second-stage weight scale shared by Q/K/V/Z.
    pub input_weight_scale_2: f32,
    /// Reciprocal activation-scale convention consumed by the kernels.
    pub input_scale_divisor: f32,
    /// Reciprocal weight-scale convention consumed by the kernels.
    pub input_weight_scale_divisor: f32,
    /// Fused Q/K/V/Z output row count.
    pub input_rows: usize,
    /// Logical hidden width of the Q/K/V/Z projection.
    pub input_columns: usize,
    /// Packed A/B control words followed by zero padding to one scale tile.
    pub control_weight_e2m1_padded: Vec<u8>,
    /// Swizzled A/B control scales with zero codes in padded rows.
    pub control_scale_e4m3_swizzled: Vec<u8>,
    /// Exact source activation scale shared by A and B controls.
    pub control_input_scale: f32,
    /// Exact source second-stage weight scale shared by A and B controls.
    pub control_weight_scale_2: f32,
    /// Reciprocal control activation-scale convention consumed by the kernels.
    pub control_input_scale_divisor: f32,
    /// Reciprocal control weight-scale convention consumed by the kernels.
    pub control_weight_scale_divisor: f32,
    /// Number of represented A/B control rows before padding.
    pub control_rows: usize,
    /// Number of rows in the runtime control planes.
    pub control_padded_rows: usize,
    /// Logical hidden width of the A/B projections.
    pub control_columns: usize,
    /// Recurrent-state output projection.
    pub output: MaterializedModelOptNvfp4Linear<'a>,
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
    /// Zero-centered RMSNorm weights before the MLP.
    pub post_attention_norm: Bf16View<'a, 1>,
    /// Decoder layer owning this layout.
    pub layer: usize,
}

impl<'a> ModelOptNvfp4GdnBindings<'a> {
    /// Fuses large projections and pads the 64 control rows without changing source values.
    pub fn materialize(self) -> CheckpointResult<MaterializedModelOptNvfp4Gdn<'a>> {
        require_gdn_layer_route(self.layer, self.layer_count, self.full_attention_interval)?;
        for (role, scale) in [
            ("QKV/Z input_scale", &self.z.input_scale),
            ("QKV/A-control input_scale", &self.a_control.input_scale),
            ("QKV/B-control input_scale", &self.b_control.input_scale),
        ] {
            require_same_modelopt_scale(self.layer, role, &self.qkv.input_scale, scale)?;
        }
        require_same_modelopt_scale(
            self.layer,
            "QKV/Z weight_scale_2",
            &self.qkv.weight_scale_2,
            &self.z.weight_scale_2,
        )?;
        require_same_modelopt_scale(
            self.layer,
            "A/B control input_scale",
            &self.a_control.input_scale,
            &self.b_control.input_scale,
        )?;
        require_same_modelopt_scale(
            self.layer,
            "A/B control weight_scale_2",
            &self.a_control.weight_scale_2,
            &self.b_control.weight_scale_2,
        )?;

        let qkv = materialize_modelopt_linear(self.qkv, self.layer, "QKV")?;
        let z = materialize_modelopt_linear(self.z, self.layer, "Z")?;
        if qkv.columns != z.columns {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} ModelOpt NVFP4 QKV/Z input widths differ",
                self.layer
            )));
        }

        let input_rows = qkv.rows.checked_add(z.rows).ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{} ModelOpt NVFP4 QKV/Z row count overflows",
                self.layer
            ))
        })?;
        let input_weight_e2m1 = gather_source_planes(
            [qkv.weight_e2m1, z.weight_e2m1],
            &format!("layer-{} ModelOpt NVFP4 QKV/Z weights", self.layer),
        )?;
        // Both exact row families end on 128-row scale-tile boundaries, so
        // concatenating their complete swizzled tiles preserves fused row order.
        let input_scale_e4m3_swizzled = gather_source_planes(
            [
                qkv.scale_e4m3_swizzled.as_slice(),
                z.scale_e4m3_swizzled.as_slice(),
            ],
            &format!("layer-{} ModelOpt NVFP4 QKV/Z scales", self.layer),
        )?;
        let controls = materialize_modelopt_controls(self.a_control, self.b_control, self.layer)?;
        let output = materialize_modelopt_linear(self.output, self.layer, "output")?;

        Ok(MaterializedModelOptNvfp4Gdn {
            input_weight_e2m1,
            input_scale_e4m3_swizzled,
            input_scale: qkv.input_scale,
            input_weight_scale_2: qkv.weight_scale_2,
            input_scale_divisor: qkv.input_scale_divisor,
            input_weight_scale_divisor: qkv.weight_scale_divisor,
            input_rows,
            input_columns: qkv.columns,
            control_weight_e2m1_padded: controls.weight_e2m1_padded,
            control_scale_e4m3_swizzled: controls.scale_e4m3_swizzled,
            control_input_scale: controls.input_scale,
            control_weight_scale_2: controls.weight_scale_2,
            control_input_scale_divisor: controls.input_scale_divisor,
            control_weight_scale_divisor: controls.weight_scale_divisor,
            control_rows: controls.rows,
            control_padded_rows: controls.padded_rows,
            control_columns: controls.columns,
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

struct MaterializedModelOptControls {
    weight_e2m1_padded: Vec<u8>,
    scale_e4m3_swizzled: Vec<u8>,
    input_scale: f32,
    weight_scale_2: f32,
    input_scale_divisor: f32,
    weight_scale_divisor: f32,
    rows: usize,
    padded_rows: usize,
    columns: usize,
}

fn materialize_modelopt_controls(
    a: ModelOptNvfp4LinearBindings<'_>,
    b: ModelOptNvfp4LinearBindings<'_>,
    layer: usize,
) -> CheckpointResult<MaterializedModelOptControls> {
    let [a_rows, packed_columns] = host_shape(a.weight.shape(), "ModelOpt NVFP4 A controls")?;
    let [b_rows, b_packed_columns] = host_shape(b.weight.shape(), "ModelOpt NVFP4 B controls")?;
    let [a_scale_rows, groups] =
        host_shape(a.block_scale.shape(), "ModelOpt NVFP4 A-control scales")?;
    let [b_scale_rows, b_groups] =
        host_shape(b.block_scale.shape(), "ModelOpt NVFP4 B-control scales")?;
    let columns = logical_columns(packed_columns, groups, layer, "A/B controls")?;
    if a_rows != a_scale_rows
        || b_rows != b_scale_rows
        || packed_columns != b_packed_columns
        || groups != b_groups
        || a_rows != a.rows
        || b_rows != b.rows
        || columns != a.columns
        || columns != b.columns
    {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 A/B control source geometry differs"
        )));
    }

    validate_nvfp4_scales(layer, "A control", a.block_scale.codes())?;
    validate_nvfp4_scales(layer, "B control", b.block_scale.codes())?;

    let rows = a_rows.checked_add(b_rows).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 A/B control row count overflows"
        ))
    })?;
    let padded_rows = rows
        .checked_add(SCALE_TILE_ROWS - 1)
        .map(|rows| rows / SCALE_TILE_ROWS * SCALE_TILE_ROWS)
        .ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{layer} ModelOpt NVFP4 A/B padded row count overflows"
            ))
        })?;

    let padded_weight_bytes = padded_rows.checked_mul(packed_columns).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 A/B padded weight length overflows"
        ))
    })?;
    let padded_scale_bytes = padded_rows.checked_mul(groups).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 A/B padded scale length overflows"
        ))
    })?;

    // The source has 64 represented control rows while the Blackwell scale
    // layout owns 128-row tiles. Rows 64..128 are never dispatched.
    let mut weight_e2m1_padded = gather_source_planes(
        [a.weight.bytes(), b.weight.bytes()],
        &format!("layer-{layer} ModelOpt NVFP4 A/B control weights"),
    )?;
    weight_e2m1_padded.resize(padded_weight_bytes, 0);
    let mut row_major_scales = gather_source_planes(
        [a.block_scale.codes(), b.block_scale.codes()],
        &format!("layer-{layer} ModelOpt NVFP4 A/B control scales"),
    )?;
    row_major_scales.resize(padded_scale_bytes, 0);
    let scale_e4m3_swizzled = swizzle_scale_planes(
        &[&row_major_scales],
        padded_rows,
        groups,
        layer,
        "A/B controls",
    )?;

    let input_scale = modelopt_scale(layer, "A/B control input_scale", &a.input_scale)?;
    let weight_scale_2 = modelopt_scale(layer, "A/B control weight_scale_2", &a.weight_scale_2)?;

    Ok(MaterializedModelOptControls {
        weight_e2m1_padded,
        scale_e4m3_swizzled,
        input_scale,
        weight_scale_2,
        input_scale_divisor: reciprocal_scale(layer, "A/B control input", input_scale)?,
        weight_scale_divisor: reciprocal_scale(layer, "A/B control weight", weight_scale_2)?,
        rows,
        padded_rows,
        columns,
    })
}

impl<'a> ModelOptNvfp4AttentionBindings<'a> {
    /// Gathers Q/K/V and converts ModelOpt scales without changing represented values.
    pub fn materialize(self) -> CheckpointResult<MaterializedModelOptNvfp4Attention<'a>> {
        require_full_attention_layer(self.layer, self.layer_count, self.full_attention_interval)?;
        require_same_modelopt_scale(
            self.layer,
            "query/key input_scale",
            &self.query_gate.input_scale,
            &self.key.input_scale,
        )?;
        require_same_modelopt_scale(
            self.layer,
            "query/value input_scale",
            &self.query_gate.input_scale,
            &self.value.input_scale,
        )?;

        let query_gate = materialize_modelopt_linear(self.query_gate, self.layer, "query/gate")?;
        let key = materialize_modelopt_linear(self.key, self.layer, "key")?;
        let value = materialize_modelopt_linear(self.value, self.layer, "value")?;
        let output = materialize_modelopt_linear(self.output, self.layer, "output")?;
        if query_gate.columns != key.columns || query_gate.columns != value.columns {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} ModelOpt NVFP4 QKV input widths differ",
                self.layer
            )));
        }

        let qkv_rows = query_gate
            .rows
            .checked_add(key.rows)
            .and_then(|rows| rows.checked_add(value.rows))
            .ok_or_else(|| {
                CheckpointError::source_binding(format!(
                    "layer-{} ModelOpt NVFP4 QKV row count overflows",
                    self.layer
                ))
            })?;
        let qkv_weight_e2m1 = gather_source_planes(
            [query_gate.weight_e2m1, key.weight_e2m1, value.weight_e2m1],
            &format!("layer-{} ModelOpt NVFP4 QKV weights", self.layer),
        )?;
        // Each admitted Q/K/V row family is independently tiled by 128 rows,
        // so concatenating complete swizzled tiles preserves fused row order.
        let qkv_scale_e4m3_swizzled = gather_source_planes(
            [
                query_gate.scale_e4m3_swizzled.as_slice(),
                key.scale_e4m3_swizzled.as_slice(),
                value.scale_e4m3_swizzled.as_slice(),
            ],
            &format!("layer-{} ModelOpt NVFP4 QKV scales", self.layer),
        )?;

        Ok(MaterializedModelOptNvfp4Attention {
            qkv_weight_e2m1,
            qkv_scale_e4m3_swizzled,
            qkv_input_scale: query_gate.input_scale,
            qkv_weight_scales_2: [
                query_gate.weight_scale_2,
                key.weight_scale_2,
                value.weight_scale_2,
            ],
            qkv_input_scale_divisor: query_gate.input_scale_divisor,
            qkv_weight_scale_divisors: [
                query_gate.weight_scale_divisor,
                key.weight_scale_divisor,
                value.weight_scale_divisor,
            ],
            qkv_rows,
            qkv_columns: query_gate.columns,
            output,
            query_norm: self.query_norm,
            key_norm: self.key_norm,
            input_norm: self.input_norm,
            post_attention_norm: self.post_attention_norm,
            layer: self.layer,
        })
    }
}

fn materialize_modelopt_linear<'a>(
    binding: ModelOptNvfp4LinearBindings<'a>,
    layer: usize,
    role: &str,
) -> CheckpointResult<MaterializedModelOptNvfp4Linear<'a>> {
    let [rows, packed_columns] = host_shape(
        binding.weight.shape(),
        &format!("ModelOpt NVFP4 {role} weights"),
    )?;
    let [scale_rows, groups] = host_shape(
        binding.block_scale.shape(),
        &format!("ModelOpt NVFP4 {role} scales"),
    )?;
    if rows != scale_rows {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 {role} source planes have incompatible row counts"
        )));
    }
    let columns = logical_columns(packed_columns, groups, layer, role)?;
    if rows != binding.rows || columns != binding.columns {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 {role} source geometry differs from its binding"
        )));
    }

    validate_nvfp4_scales(layer, role, binding.block_scale.codes())?;
    let input_scale = modelopt_scale(layer, &format!("{role} input_scale"), &binding.input_scale)?;
    let weight_scale_2 = modelopt_scale(
        layer,
        &format!("{role} weight_scale_2"),
        &binding.weight_scale_2,
    )?;
    let input_scale_divisor = reciprocal_scale(layer, &format!("{role} input"), input_scale)?;
    let weight_scale_divisor = reciprocal_scale(layer, &format!("{role} weight"), weight_scale_2)?;
    let scale_e4m3_swizzled =
        swizzle_scale_planes(&[binding.block_scale.codes()], rows, groups, layer, role)?;

    Ok(MaterializedModelOptNvfp4Linear {
        weight_e2m1: binding.weight.bytes(),
        scale_e4m3_swizzled,
        input_scale,
        weight_scale_2,
        input_scale_divisor,
        weight_scale_divisor,
        rows,
        columns,
        layer,
    })
}

impl<'a> ModelOptNvfp4MlpBindings<'a> {
    /// Converts ModelOpt scalar conventions and swizzles block scales for the SM120 kernels.
    pub fn materialize(self) -> CheckpointResult<MaterializedModelOptNvfp4Mlp<'a>> {
        if self.layer >= self.layer_count {
            return Err(CheckpointError::source_binding(format!(
                "layer {} does not use the admitted ModelOpt NVFP4 MLP source contract",
                self.layer
            )));
        }

        require_same_modelopt_scale(
            self.layer,
            "gate/up input_scale",
            &self.gate.input_scale,
            &self.up.input_scale,
        )?;
        require_same_modelopt_scale(
            self.layer,
            "gate/up weight_scale_2",
            &self.gate.weight_scale_2,
            &self.up.weight_scale_2,
        )?;

        let gate_up_input_scale =
            modelopt_scale(self.layer, "gate/up input_scale", &self.gate.input_scale)?;
        let gate_up_weight_scale_2 = modelopt_scale(
            self.layer,
            "gate/up weight_scale_2",
            &self.gate.weight_scale_2,
        )?;
        let down_input_scale =
            modelopt_scale(self.layer, "down input_scale", &self.down.input_scale)?;
        let down_weight_scale_2 =
            modelopt_scale(self.layer, "down weight_scale_2", &self.down.weight_scale_2)?;

        // ModelOpt exports amax / (E2M1_MAX * E4M3_MAX). The kernels take the
        // reciprocal global divisor, so this changes convention, not represented values.
        let gate_up = Nvfp4GateUpBindings {
            gate_weight: self.gate.weight,
            up_weight: self.up.weight,
            gate_scale: self.gate.block_scale,
            up_scale: self.up.block_scale,
            input_scale_divisor: reciprocal_scale(
                self.layer,
                "gate/up input",
                gate_up_input_scale,
            )?,
            weight_scale_divisor: reciprocal_scale(
                self.layer,
                "gate/up weight",
                gate_up_weight_scale_2,
            )?,
            layer: self.layer,
        }
        .materialize()?;
        let down = Nvfp4DownBindings {
            weight: self.down.weight,
            scale: self.down.block_scale,
            input_scale_divisor: reciprocal_scale(self.layer, "down input", down_input_scale)?,
            weight_scale_divisor: reciprocal_scale(self.layer, "down weight", down_weight_scale_2)?,
            layer: self.layer,
        }
        .materialize()?;

        Ok(MaterializedModelOptNvfp4Mlp {
            gate_up,
            down,
            gate_up_input_scale,
            gate_up_weight_scale_2,
            down_input_scale,
            down_weight_scale_2,
            input_norm: self.input_norm,
            next_norm: self.next_norm,
            layer: self.layer,
        })
    }
}

impl<'a> Nvfp4DownBindings<'a> {
    /// Materializes the down-projection scale layout without requantizing source values.
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4Down<'a>> {
        require_nvfp4_mlp_layer(self.layer, NVFP4_MLP_LAYER_END)?;

        let [rows, packed_columns] = host_shape(self.weight.shape(), "NVFP4 down weights")?;
        let [scale_rows, groups] = host_shape(self.scale.shape(), "NVFP4 down scales")?;

        if rows != scale_rows {
            return Err(CheckpointError::source_binding(format!(
                "layer-{} NVFP4 down source planes have incompatible row counts",
                self.layer
            )));
        }

        let columns = logical_columns(packed_columns, groups, self.layer, "down")?;

        validate_nvfp4_scales(self.layer, "down", self.scale.codes())?;
        validate_divisor(self.layer, "down input", self.input_scale_divisor)?;
        validate_divisor(self.layer, "down weight", self.weight_scale_divisor)?;

        let scale_e4m3_swizzled =
            swizzle_scale_planes(&[self.scale.codes()], rows, groups, self.layer, "down")?;

        Ok(MaterializedNvfp4Down {
            weight_e2m1: self.weight.bytes(),
            scale_e4m3_swizzled,
            input_scale_divisor: self.input_scale_divisor,
            weight_scale_divisor: self.weight_scale_divisor,
            rows,
            columns,
            layer: self.layer,
        })
    }
}

fn modelopt_scale(layer: usize, role: &str, scale: &F32View<'_, 0>) -> CheckpointResult<f32> {
    let value = scale.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 {role} must be finite and positive, observed {value}"
        )));
    }

    Ok(value)
}

fn require_same_modelopt_scale(
    layer: usize,
    role: &str,
    first: &F32View<'_, 0>,
    second: &F32View<'_, 0>,
) -> CheckpointResult<()> {
    if first.bits(0) != second.bits(0) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 {role} values differ"
        )));
    }

    Ok(())
}

fn reciprocal_scale(layer: usize, role: &str, scale: f32) -> CheckpointResult<f32> {
    let divisor = 1.0 / scale;
    validate_divisor(layer, role, divisor)?;

    Ok(divisor)
}

fn host_shape(shape: &[u64; 2], role: &str) -> CheckpointResult<[usize; 2]> {
    let rows = usize::try_from(shape[0]).map_err(|_| {
        CheckpointError::source_binding(format!("{role} row count exceeds this host"))
    })?;
    let columns = usize::try_from(shape[1]).map_err(|_| {
        CheckpointError::source_binding(format!("{role} column count exceeds this host"))
    })?;

    Ok([rows, columns])
}

fn gather_source_planes<const N: usize>(
    planes: [&[u8]; N],
    role: &str,
) -> CheckpointResult<Vec<u8>> {
    let bytes = planes.iter().try_fold(0usize, |bytes, plane| {
        bytes
            .checked_add(plane.len())
            .ok_or_else(|| CheckpointError::source_binding(format!("{role} length overflows")))
    })?;

    let mut gathered = Vec::new();

    gathered.try_reserve_exact(bytes).map_err(|_| {
        CheckpointError::source_binding(format!("{role} cannot reserve {bytes} host bytes"))
    })?;

    if bytes >= PARALLEL_GATHER_MIN_BYTES && materialization_workers() > 1 {
        match planes.as_slice() {
            [first, second] => {
                materialization_pool(role)?.install(|| {
                    first
                        .par_iter()
                        .copied()
                        .chain(second.par_iter().copied())
                        .collect_into_vec(&mut gathered);
                });
                return Ok(gathered);
            }
            [first, second, third] => {
                materialization_pool(role)?.install(|| {
                    first
                        .par_iter()
                        .copied()
                        .chain(second.par_iter().copied())
                        .chain(third.par_iter().copied())
                        .collect_into_vec(&mut gathered);
                });
                return Ok(gathered);
            }
            _ => {}
        }
    }

    for plane in planes {
        gathered.extend_from_slice(plane);
    }

    Ok(gathered)
}

fn logical_columns(
    packed_columns: usize,
    groups: usize,
    layer: usize,
    role: &str,
) -> CheckpointResult<usize> {
    let weight_columns = packed_columns
        .checked_mul(E2M1_VALUES_PER_BYTE)
        .ok_or_else(|| {
            CheckpointError::source_binding(format!(
                "layer-{layer} NVFP4 {role} logical weight width overflows"
            ))
        })?;
    let scale_columns = groups.checked_mul(NVFP4_GROUP_SIZE).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} logical scale width overflows"
        ))
    })?;

    if weight_columns != scale_columns {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} weights cover {weight_columns} values per row but scales cover {scale_columns}"
        )));
    }

    Ok(weight_columns)
}

fn validate_divisor(layer: usize, role: &str, value: f32) -> CheckpointResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale divisor must be finite and positive, observed {value}"
        )));
    }

    Ok(())
}

fn swizzle_scale_planes(
    planes: &[&[u8]],
    rows_per_plane: usize,
    groups_per_row: usize,
    layer: usize,
    role: &str,
) -> CheckpointResult<Vec<u8>> {
    let rows = rows_per_plane.checked_mul(planes.len()).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} fused scale row count overflows"
        ))
    })?;

    if rows == 0 || !rows.is_multiple_of(SCALE_TILE_ROWS) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale rows {rows} are not tiled by {SCALE_TILE_ROWS}"
        )));
    }

    if groups_per_row == 0 || !groups_per_row.is_multiple_of(SCALE_TILE_GROUPS) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale groups {groups_per_row} are not tiled by {SCALE_TILE_GROUPS}"
        )));
    }

    let plane_len = rows_per_plane.checked_mul(groups_per_row).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} source scale length overflows"
        ))
    })?;
    let output_len = rows.checked_mul(groups_per_row).ok_or_else(|| {
        CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} materialized scale length overflows"
        ))
    })?;

    if planes.iter().any(|plane| plane.len() != plane_len) {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} source scale plane length does not match its shape"
        )));
    }

    let mut swizzled = vec![0; output_len];
    let scale_tiles_per_row = groups_per_row / SCALE_TILE_GROUPS;
    let swizzle_tile = |(tile_index, destination): (usize, &mut [u8])| {
        swizzle_scale_tile(
            destination,
            tile_index,
            scale_tiles_per_row,
            planes,
            rows_per_plane,
            groups_per_row,
        );
    };

    if output_len >= PARALLEL_SWIZZLE_MIN_BYTES {
        materialization_pool(&format!("layer-{layer} NVFP4 {role} scales"))?.install(|| {
            swizzled
                .par_chunks_mut(SCALE_TILE_BYTES)
                .enumerate()
                .for_each(swizzle_tile);
        });
    } else {
        swizzled
            .chunks_mut(SCALE_TILE_BYTES)
            .enumerate()
            .for_each(swizzle_tile);
    }

    Ok(swizzled)
}

fn swizzle_scale_tile(
    destination: &mut [u8],
    tile_index: usize,
    scale_tiles_per_row: usize,
    planes: &[&[u8]],
    rows_per_plane: usize,
    groups_per_row: usize,
) {
    let persistent_tile = tile_index / scale_tiles_per_row;
    let scale_tile = tile_index % scale_tiles_per_row;
    let source_group = scale_tile * SCALE_TILE_GROUPS;

    // Each 512-byte destination tile is independent. Writing by its 32 contiguous
    // 16-byte rows avoids the old per-byte division and scattered store while preserving
    // the exact BlockScaleK16M128x4 address mapping.
    for row_mod32 in 0..32 {
        let destination_row = &mut destination[row_mod32 * 16..(row_mod32 + 1) * 16];
        for row_quartile in 0..4 {
            let row = persistent_tile * SCALE_TILE_ROWS + row_quartile * 32 + row_mod32;
            let source_plane = row / rows_per_plane;
            let source_row = row % rows_per_plane;
            let source = source_row * groups_per_row + source_group;
            destination_row[row_quartile * 4..(row_quartile + 1) * 4]
                .copy_from_slice(&planes[source_plane][source..source + 4]);
        }
    }
}

fn materialization_pool(role: &str) -> CheckpointResult<&'static rayon::ThreadPool> {
    let pool = MATERIALIZATION_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(materialization_workers())
            .thread_name(|index| format!("tuisko-materialize-{index}"))
            .build()
            .map_err(|error| error.to_string())
    });
    pool.as_ref().map_err(|error| {
        CheckpointError::source_binding(format!(
            "{role} cannot start bounded materialization workers: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        SCALE_TILE_GROUPS, SCALE_TILE_ROWS, gather_source_planes, swizzle_scale_planes,
        validate_divisor,
    };
    use crate::{
        Arch, Bf16View, CheckpointErrorCode, CheckpointSnapshot, DType, F32View, Fp8E4M3View,
        FullAttentionQkvBindings, ModelOptNvfp4AttentionBindings, ModelOptNvfp4GdnBindings,
        ModelOptNvfp4LinearBindings, ModelOptNvfp4MlpBindings, Nvfp4DownBindings,
        Nvfp4GateUpBindings, Qwen35_9B, TensorView, U8View,
    };

    const ROWS: usize = 128;
    const GROUPS: usize = 8;
    const COLUMNS: usize = GROUPS * 16;
    const PACKED_COLUMNS: usize = COLUMNS / 2;

    #[test]
    fn parallel_gather_preserves_three_plane_order_exactly() {
        let first = (0..1 << 20).map(|index| index as u8).collect::<Vec<_>>();
        let second = (0..4_097)
            .map(|index| (index as u8).wrapping_mul(3))
            .collect::<Vec<_>>();
        let third = (0..8_191)
            .map(|index| (index as u8).wrapping_mul(5))
            .collect::<Vec<_>>();

        let gathered = gather_source_planes([&first, &second, &third], "test QKV").unwrap();
        let second_end = first.len() + second.len();

        assert_eq!(&gathered[..first.len()], first.as_slice());
        assert_eq!(&gathered[first.len()..second_end], second.as_slice());
        assert_eq!(&gathered[second_end..], third.as_slice());
    }

    #[test]
    fn parallel_gather_preserves_two_plane_order_exactly() {
        let first = (0..1 << 20).map(|index| index as u8).collect::<Vec<_>>();
        let second = (0..8_191)
            .map(|index| (index as u8).wrapping_mul(5))
            .collect::<Vec<_>>();

        let gathered = gather_source_planes([&first, &second], "test fused planes").unwrap();

        assert_eq!(&gathered[..first.len()], first.as_slice());
        assert_eq!(&gathered[first.len()..], second.as_slice());
    }

    fn u8_view<'a>(name: &'a str, shape: &'a [u64; 2], bytes: &'a [u8]) -> U8View<'a, 2> {
        U8View::bind(
            TensorView {
                name,
                dtype: DType::U8,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    fn fp8_view<'a>(name: &'a str, shape: &'a [u64; 2], bytes: &'a [u8]) -> Fp8E4M3View<'a, 2> {
        Fp8E4M3View::bind(
            TensorView {
                name,
                dtype: DType::Fp8E4M3,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    fn bf16_view<'a>(name: &'a str, shape: &'a [u64; 2], bytes: &'a [u8]) -> Bf16View<'a, 2> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    fn bf16_vector<'a>(name: &'a str, shape: &'a [u64; 1], bytes: &'a [u8]) -> Bf16View<'a, 1> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    fn bf16_volume<'a>(name: &'a str, shape: &'a [u64; 3], bytes: &'a [u8]) -> Bf16View<'a, 3> {
        Bf16View::bind(
            TensorView {
                name,
                dtype: DType::Bf16,
                shape,
                bytes,
                data_range: 0..bytes.len() as u64,
            },
            *shape,
        )
        .unwrap()
    }

    fn f32_scalar_view<'a>(name: &'a str, bytes: &'a [u8; 4]) -> F32View<'a, 0> {
        F32View::bind(
            TensorView {
                name,
                dtype: DType::F32,
                shape: &[],
                bytes,
                data_range: 0..4,
            },
            [],
        )
        .unwrap()
    }

    fn bf16_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn scale_codes(seed: usize) -> Vec<u8> {
        scale_codes_for(ROWS, seed)
    }

    fn scale_codes_for(rows: usize, seed: usize) -> Vec<u8> {
        (0..rows * GROUPS)
            .map(|index| ((index * 37 + seed) % 0x7f) as u8)
            .collect()
    }

    fn block_scale_oracle(source: &[u8], rows: usize, groups: usize) -> Vec<u8> {
        let mut expected = Vec::with_capacity(source.len());

        for row_tile in 0..rows / SCALE_TILE_ROWS {
            for group_tile in 0..groups / SCALE_TILE_GROUPS {
                for row_mod32 in 0..32 {
                    for row_quartile in 0..4 {
                        for scale_lane in 0..SCALE_TILE_GROUPS {
                            let row = row_tile * SCALE_TILE_ROWS + row_quartile * 32 + row_mod32;
                            let group = group_tile * SCALE_TILE_GROUPS + scale_lane;
                            expected.push(source[row * groups + group]);
                        }
                    }
                }
            }
        }

        expected
    }

    #[test]
    fn full_attention_qkv_materialization_gathers_exact_source_words() {
        let query_shape = [4, 3];
        let kv_shape = [1, 3];
        let query_scale_shape = [4, 1];
        let kv_scale_shape = [1, 1];
        let query_weight = (0x10..0x1c).collect::<Vec<_>>();
        let key_weight = (0x30..0x33).collect::<Vec<_>>();
        let value_weight = (0x50..0x53).collect::<Vec<_>>();
        let query_scale = bf16_bytes(&[0x3f80, 0x4000, 0x4040, 0x4080]);
        let key_scale = bf16_bytes(&[0x40a0]);
        let value_scale = bf16_bytes(&[0x40c0]);
        let bindings = FullAttentionQkvBindings {
            query_gate_weight: fp8_view("query", &query_shape, &query_weight),
            key_weight: fp8_view("key", &kv_shape, &key_weight),
            value_weight: fp8_view("value", &kv_shape, &value_weight),
            query_gate_scale: bf16_view("query-scale", &query_scale_shape, &query_scale),
            key_scale: bf16_view("key-scale", &kv_scale_shape, &key_scale),
            value_scale: bf16_view("value-scale", &kv_scale_shape, &value_scale),
            layer: 3,
            layer_count: 8,
            full_attention_interval: 4,
        };

        let error = FullAttentionQkvBindings {
            layer: 4,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("does not use the admitted full-attention")
        );

        let materialized = bindings.materialize().unwrap();
        let query_end = query_weight.len();
        let key_end = query_end + key_weight.len();
        let query_scale_end = query_scale.len();
        let key_scale_end = query_scale_end + key_scale.len();

        assert_eq!(&materialized.weight_e4m3[..query_end], query_weight);
        assert_eq!(&materialized.weight_e4m3[query_end..key_end], key_weight);
        assert_eq!(&materialized.weight_e4m3[key_end..], value_weight);
        assert_eq!(&materialized.scale_bf16[..query_scale_end], query_scale);
        assert_eq!(
            &materialized.scale_bf16[query_scale_end..key_scale_end],
            key_scale
        );
        assert_eq!(&materialized.scale_bf16[key_scale_end..], value_scale);
        assert_eq!((materialized.rows, materialized.columns), (6, 3));
        assert_eq!(materialized.layer, 3);
    }

    #[test]
    fn full_attention_qkv_materialization_rejects_incompatible_shapes() {
        let query_shape = [4, 3];
        let key_shape = [1, 2];
        let value_shape = [1, 3];
        let query_scale_shape = [4, 1];
        let kv_scale_shape = [1, 1];
        let query_weight = vec![0x10; 12];
        let key_weight = vec![0x20; 2];
        let value_weight = vec![0x30; 3];
        let query_scale = bf16_bytes(&[0x3f80; 4]);
        let key_scale = bf16_bytes(&[0x3f80]);
        let value_scale = bf16_bytes(&[0x3f80]);
        let error = FullAttentionQkvBindings {
            query_gate_weight: fp8_view("query", &query_shape, &query_weight),
            key_weight: fp8_view("key", &key_shape, &key_weight),
            value_weight: fp8_view("value", &value_shape, &value_weight),
            query_gate_scale: bf16_view("query-scale", &query_scale_shape, &query_scale),
            key_scale: bf16_view("key-scale", &kv_scale_shape, &key_scale),
            value_scale: bf16_view("value-scale", &kv_scale_shape, &value_scale),
            layer: 3,
            layer_count: 8,
            full_attention_interval: 4,
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("incompatible shapes"));
    }

    #[test]
    fn gate_up_materialization_only_permutes_scale_codes() {
        let weight_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let scale_shape = [ROWS as u64, GROUPS as u64];
        let gate_weight = vec![0x10; ROWS * PACKED_COLUMNS];
        let up_weight = vec![0x32; ROWS * PACKED_COLUMNS];
        let gate_scale = scale_codes(0);
        let up_scale = scale_codes(11);
        let bindings = Nvfp4GateUpBindings {
            gate_weight: u8_view("gate", &weight_shape, &gate_weight),
            up_weight: u8_view("up", &weight_shape, &up_weight),
            gate_scale: fp8_view("gate-scale", &scale_shape, &gate_scale),
            up_scale: fp8_view("up-scale", &scale_shape, &up_scale),
            input_scale_divisor: 3.0,
            weight_scale_divisor: 0.125,
            layer: 55,
        };

        let error = Nvfp4GateUpBindings {
            layer: 56,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("does not use the admitted NVFP4")
        );

        let materialized = bindings.materialize().unwrap();
        let source = [gate_scale.as_slice(), up_scale.as_slice()].concat();
        let expected = block_scale_oracle(&source, 2 * ROWS, GROUPS);

        assert_eq!(materialized.scale_e4m3_swizzled, expected);
        assert_eq!(materialized.gate_weight_e2m1, gate_weight);
        assert_eq!(materialized.up_weight_e2m1, up_weight);
        assert_eq!(materialized.gate_weight_e2m1.as_ptr(), gate_weight.as_ptr());
        assert_eq!(materialized.up_weight_e2m1.as_ptr(), up_weight.as_ptr());
        assert_eq!((materialized.rows, materialized.columns), (256, 128));
        assert_eq!(materialized.layer, 55);
        assert_eq!(materialized.input_scale_divisor.to_bits(), 3.0f32.to_bits());
        assert_eq!(
            materialized.weight_scale_divisor.to_bits(),
            0.125f32.to_bits()
        );
    }

    #[test]
    fn down_materialization_only_permutes_scale_codes() {
        let weight_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let scale_shape = [ROWS as u64, GROUPS as u64];
        let weight = vec![0x54; ROWS * PACKED_COLUMNS];
        let scale = scale_codes(23);
        let bindings = Nvfp4DownBindings {
            weight: u8_view("down", &weight_shape, &weight),
            scale: fp8_view("down-scale", &scale_shape, &scale),
            input_scale_divisor: 19.0,
            weight_scale_divisor: 3_376.0,
            layer: 55,
        };

        let error = Nvfp4DownBindings {
            layer: 56,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("does not use the admitted NVFP4")
        );

        let materialized = bindings.materialize().unwrap();
        let expected = block_scale_oracle(&scale, ROWS, GROUPS);

        assert_eq!(materialized.scale_e4m3_swizzled, expected);
        assert_eq!(materialized.weight_e2m1, weight);
        assert_eq!(materialized.weight_e2m1.as_ptr(), weight.as_ptr());
        assert_eq!((materialized.rows, materialized.columns), (128, 128));
        assert_eq!(materialized.layer, 55);
    }

    #[test]
    fn modelopt_materialization_preserves_words_and_converts_scale_convention() {
        let weight_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let scale_shape = [ROWS as u64, GROUPS as u64];
        let norm_shape = [ROWS as u64];
        let gate_weight = vec![0x10; ROWS * PACKED_COLUMNS];
        let up_weight = vec![0x32; ROWS * PACKED_COLUMNS];
        let down_weight = vec![0x54; ROWS * PACKED_COLUMNS];
        let gate_scale = scale_codes(0);
        let up_scale = scale_codes(11);
        let down_scale = scale_codes(23);
        let gate_up_input = 0.25f32.to_le_bytes();
        let gate_up_weight = 0.125f32.to_le_bytes();
        let down_input = 0.5f32.to_le_bytes();
        let down_weight_scale = 0.0625f32.to_le_bytes();
        let input_norm = bf16_bytes(&[0x3f80; ROWS]);
        let next_norm = bf16_bytes(&[0x4000; ROWS]);
        let gate = ModelOptNvfp4LinearBindings {
            weight: u8_view("gate", &weight_shape, &gate_weight),
            block_scale: fp8_view("gate-scale", &scale_shape, &gate_scale),
            input_scale: f32_scalar_view("gate-input", &gate_up_input),
            weight_scale_2: f32_scalar_view("gate-weight", &gate_up_weight),
            rows: ROWS,
            columns: COLUMNS,
        };
        let up = ModelOptNvfp4LinearBindings {
            weight: u8_view("up", &weight_shape, &up_weight),
            block_scale: fp8_view("up-scale", &scale_shape, &up_scale),
            input_scale: f32_scalar_view("up-input", &gate_up_input),
            weight_scale_2: f32_scalar_view("up-weight", &gate_up_weight),
            rows: ROWS,
            columns: COLUMNS,
        };
        let down = ModelOptNvfp4LinearBindings {
            weight: u8_view("down", &weight_shape, &down_weight),
            block_scale: fp8_view("down-scale", &scale_shape, &down_scale),
            input_scale: f32_scalar_view("down-input", &down_input),
            weight_scale_2: f32_scalar_view("down-weight", &down_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let bindings = ModelOptNvfp4MlpBindings {
            gate,
            up,
            down,
            input_norm: bf16_vector("input-norm", &norm_shape, &input_norm),
            next_norm: bf16_vector("next-norm", &norm_shape, &next_norm),
            layer: 3,
            layer_count: 32,
        };

        let route_error = ModelOptNvfp4MlpBindings {
            layer_count: 3,
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(route_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            route_error
                .to_string()
                .contains("does not use the admitted")
        );

        let mismatched_input = 0.75f32.to_le_bytes();
        let scale_error = ModelOptNvfp4MlpBindings {
            up: ModelOptNvfp4LinearBindings {
                input_scale: f32_scalar_view("up-input", &mismatched_input),
                ..up
            },
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(scale_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            scale_error
                .to_string()
                .contains("input_scale values differ")
        );

        let materialized = bindings.materialize().unwrap();
        let gate_up_source = [gate_scale.as_slice(), up_scale.as_slice()].concat();

        assert_eq!(
            materialized.gate_up.scale_e4m3_swizzled,
            block_scale_oracle(&gate_up_source, 2 * ROWS, GROUPS)
        );
        assert_eq!(
            materialized.down.scale_e4m3_swizzled,
            block_scale_oracle(&down_scale, ROWS, GROUPS)
        );
        assert_eq!(
            materialized.gate_up.gate_weight_e2m1.as_ptr(),
            gate_weight.as_ptr()
        );
        assert_eq!(
            materialized.gate_up.up_weight_e2m1.as_ptr(),
            up_weight.as_ptr()
        );
        assert_eq!(materialized.down.weight_e2m1.as_ptr(), down_weight.as_ptr());
        assert_eq!(
            materialized.gate_up_input_scale.to_bits(),
            0.25f32.to_bits()
        );
        assert_eq!(
            materialized.gate_up_weight_scale_2.to_bits(),
            0.125f32.to_bits()
        );
        assert_eq!(materialized.down_input_scale.to_bits(), 0.5f32.to_bits());
        assert_eq!(
            materialized.down_weight_scale_2.to_bits(),
            0.0625f32.to_bits()
        );
        assert_eq!(materialized.gate_up.input_scale_divisor, 4.0);
        assert_eq!(materialized.gate_up.weight_scale_divisor, 8.0);
        assert_eq!(materialized.down.input_scale_divisor, 2.0);
        assert_eq!(materialized.down.weight_scale_divisor, 16.0);
        assert_eq!(materialized.input_norm.word(0), Some(0x3f80));
        assert_eq!(materialized.next_norm.word(0), Some(0x4000));
        assert_eq!(materialized.layer, 3);
    }

    #[test]
    fn modelopt_attention_materialization_preserves_words_and_per_projection_scales() {
        let query_rows = 2 * ROWS;
        let query_shape = [query_rows as u64, PACKED_COLUMNS as u64];
        let kv_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let query_scale_shape = [query_rows as u64, GROUPS as u64];
        let kv_scale_shape = [ROWS as u64, GROUPS as u64];
        let norm_shape = [ROWS as u64];
        let head_norm_shape = [1];
        let query_weight = vec![0x10; query_rows * PACKED_COLUMNS];
        let key_weight = vec![0x32; ROWS * PACKED_COLUMNS];
        let value_weight = vec![0x54; ROWS * PACKED_COLUMNS];
        let output_weight = vec![0x76; ROWS * PACKED_COLUMNS];
        let query_scale = scale_codes_for(query_rows, 0);
        let key_scale = scale_codes_for(ROWS, 11);
        let value_scale = scale_codes_for(ROWS, 23);
        let output_scale = scale_codes_for(ROWS, 31);
        let qkv_input = 0.25f32.to_le_bytes();
        let query_weight_scale = 0.125f32.to_le_bytes();
        let key_weight_scale = 0.25f32.to_le_bytes();
        let value_weight_scale = 0.5f32.to_le_bytes();
        let output_input = 0.5f32.to_le_bytes();
        let output_weight_scale = 0.0625f32.to_le_bytes();
        let norm = bf16_bytes(&[0x3f80; ROWS]);
        let head_norm = bf16_bytes(&[0x4000]);
        let query_gate = ModelOptNvfp4LinearBindings {
            weight: u8_view("query", &query_shape, &query_weight),
            block_scale: fp8_view("query-scale", &query_scale_shape, &query_scale),
            input_scale: f32_scalar_view("query-input", &qkv_input),
            weight_scale_2: f32_scalar_view("query-weight", &query_weight_scale),
            rows: query_rows,
            columns: COLUMNS,
        };
        let key = ModelOptNvfp4LinearBindings {
            weight: u8_view("key", &kv_shape, &key_weight),
            block_scale: fp8_view("key-scale", &kv_scale_shape, &key_scale),
            input_scale: f32_scalar_view("key-input", &qkv_input),
            weight_scale_2: f32_scalar_view("key-weight", &key_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let value = ModelOptNvfp4LinearBindings {
            weight: u8_view("value", &kv_shape, &value_weight),
            block_scale: fp8_view("value-scale", &kv_scale_shape, &value_scale),
            input_scale: f32_scalar_view("value-input", &qkv_input),
            weight_scale_2: f32_scalar_view("value-weight", &value_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let output = ModelOptNvfp4LinearBindings {
            weight: u8_view("output", &kv_shape, &output_weight),
            block_scale: fp8_view("output-scale", &kv_scale_shape, &output_scale),
            input_scale: f32_scalar_view("output-input", &output_input),
            weight_scale_2: f32_scalar_view("output-weight", &output_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let bindings = ModelOptNvfp4AttentionBindings {
            query_gate,
            key,
            value,
            output,
            query_norm: bf16_vector("query-norm", &head_norm_shape, &head_norm),
            key_norm: bf16_vector("key-norm", &head_norm_shape, &head_norm),
            input_norm: bf16_vector("input-norm", &norm_shape, &norm),
            post_attention_norm: bf16_vector("post-attention-norm", &norm_shape, &norm),
            layer: 3,
            layer_count: 8,
            full_attention_interval: 4,
        };

        let route_error = ModelOptNvfp4AttentionBindings {
            layer: 4,
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(route_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            route_error
                .to_string()
                .contains("does not use the admitted full-attention")
        );

        let mismatched_input = 0.75f32.to_le_bytes();
        let error = ModelOptNvfp4AttentionBindings {
            key: ModelOptNvfp4LinearBindings {
                input_scale: f32_scalar_view("key-input", &mismatched_input),
                ..key
            },
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            error
                .to_string()
                .contains("query/key input_scale values differ")
        );

        let materialized = bindings.materialize().unwrap();
        let qkv_weight_source = [
            query_weight.as_slice(),
            key_weight.as_slice(),
            value_weight.as_slice(),
        ]
        .concat();
        let qkv_scale_source = [
            query_scale.as_slice(),
            key_scale.as_slice(),
            value_scale.as_slice(),
        ]
        .concat();

        assert_eq!(materialized.qkv_weight_e2m1, qkv_weight_source);
        assert_eq!(
            materialized.qkv_scale_e4m3_swizzled,
            block_scale_oracle(&qkv_scale_source, 4 * ROWS, GROUPS)
        );
        assert_eq!(
            materialized.output.scale_e4m3_swizzled,
            block_scale_oracle(&output_scale, ROWS, GROUPS)
        );
        assert_eq!(
            materialized.output.weight_e2m1.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(
            (materialized.qkv_rows, materialized.qkv_columns),
            (512, 128)
        );
        assert_eq!(materialized.qkv_input_scale.to_bits(), 0.25f32.to_bits());
        assert_eq!(materialized.qkv_input_scale_divisor, 4.0);
        assert_eq!(materialized.qkv_weight_scales_2, [0.125, 0.25, 0.5]);
        assert_eq!(materialized.qkv_weight_scale_divisors, [8.0, 4.0, 2.0]);
        assert_eq!(materialized.output.input_scale_divisor, 2.0);
        assert_eq!(materialized.output.weight_scale_divisor, 16.0);
        assert_eq!(materialized.query_norm.word(0), Some(0x4000));
        assert_eq!(materialized.input_norm.word(0), Some(0x3f80));
        assert_eq!(materialized.layer, 3);
    }

    #[test]
    fn modelopt_gdn_materialization_preserves_source_words_and_pads_controls() {
        const CONTROL_ROWS: usize = 32;

        let projection_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let projection_scale_shape = [ROWS as u64, GROUPS as u64];
        let control_shape = [CONTROL_ROWS as u64, PACKED_COLUMNS as u64];
        let control_scale_shape = [CONTROL_ROWS as u64, GROUPS as u64];
        let convolution_shape = [ROWS as u64, 1, 4];
        let control_vector_shape = [CONTROL_ROWS as u64];
        let norm_shape = [COLUMNS as u64];
        let head_norm_shape = [1];
        let qkv_weight = vec![0x10; ROWS * PACKED_COLUMNS];
        let z_weight = vec![0x20; ROWS * PACKED_COLUMNS];
        let a_weight = vec![0x30; CONTROL_ROWS * PACKED_COLUMNS];
        let b_weight = vec![0x40; CONTROL_ROWS * PACKED_COLUMNS];
        let output_weight = vec![0x50; ROWS * PACKED_COLUMNS];
        let qkv_scale = scale_codes_for(ROWS, 0);
        let z_scale = scale_codes_for(ROWS, 11);
        let a_scale = scale_codes_for(CONTROL_ROWS, 23);
        let b_scale = scale_codes_for(CONTROL_ROWS, 31);
        let output_scale = scale_codes_for(ROWS, 43);
        let input_scale = 0.25f32.to_le_bytes();
        let input_weight_scale = 0.125f32.to_le_bytes();
        let control_weight_scale = 0.5f32.to_le_bytes();
        let output_input_scale = 0.5f32.to_le_bytes();
        let output_weight_scale = 0.0625f32.to_le_bytes();
        let convolution = bf16_bytes(&vec![0x3f80; ROWS * 4]);
        let a_log = bf16_bytes(&[0x4000; CONTROL_ROWS]);
        let dt_bias = bf16_bytes(&[0x4040; CONTROL_ROWS]);
        let head_norm = bf16_bytes(&[0x4080]);
        let input_norm = bf16_bytes(&vec![0x40a0; COLUMNS]);
        let post_attention_norm = bf16_bytes(&vec![0x40c0; COLUMNS]);
        let qkv = ModelOptNvfp4LinearBindings {
            weight: u8_view("qkv", &projection_shape, &qkv_weight),
            block_scale: fp8_view("qkv-scale", &projection_scale_shape, &qkv_scale),
            input_scale: f32_scalar_view("qkv-input", &input_scale),
            weight_scale_2: f32_scalar_view("qkv-weight", &input_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let z = ModelOptNvfp4LinearBindings {
            weight: u8_view("z", &projection_shape, &z_weight),
            block_scale: fp8_view("z-scale", &projection_scale_shape, &z_scale),
            input_scale: f32_scalar_view("z-input", &input_scale),
            weight_scale_2: f32_scalar_view("z-weight", &input_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let a_control = ModelOptNvfp4LinearBindings {
            weight: u8_view("a", &control_shape, &a_weight),
            block_scale: fp8_view("a-scale", &control_scale_shape, &a_scale),
            input_scale: f32_scalar_view("a-input", &input_scale),
            weight_scale_2: f32_scalar_view("a-weight", &control_weight_scale),
            rows: CONTROL_ROWS,
            columns: COLUMNS,
        };
        let b_control = ModelOptNvfp4LinearBindings {
            weight: u8_view("b", &control_shape, &b_weight),
            block_scale: fp8_view("b-scale", &control_scale_shape, &b_scale),
            input_scale: f32_scalar_view("b-input", &input_scale),
            weight_scale_2: f32_scalar_view("b-weight", &control_weight_scale),
            rows: CONTROL_ROWS,
            columns: COLUMNS,
        };
        let output = ModelOptNvfp4LinearBindings {
            weight: u8_view("output", &projection_shape, &output_weight),
            block_scale: fp8_view("output-scale", &projection_scale_shape, &output_scale),
            input_scale: f32_scalar_view("output-input", &output_input_scale),
            weight_scale_2: f32_scalar_view("output-weight", &output_weight_scale),
            rows: ROWS,
            columns: COLUMNS,
        };
        let bindings = ModelOptNvfp4GdnBindings {
            qkv,
            z,
            a_control,
            b_control,
            output,
            convolution_weight: bf16_volume("convolution", &convolution_shape, &convolution),
            a_log: bf16_vector("a-log", &control_vector_shape, &a_log),
            dt_bias: bf16_vector("dt-bias", &control_vector_shape, &dt_bias),
            norm: bf16_vector("norm", &head_norm_shape, &head_norm),
            input_norm: bf16_vector("input-norm", &norm_shape, &input_norm),
            post_attention_norm: bf16_vector(
                "post-attention-norm",
                &norm_shape,
                &post_attention_norm,
            ),
            layer: 0,
            layer_count: 4,
            full_attention_interval: 4,
        };

        let route_error = ModelOptNvfp4GdnBindings {
            layer: 3,
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(route_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(route_error.to_string().contains("GDN source contract"));

        let mismatched_weight_scale = 0.25f32.to_le_bytes();
        let scale_error = ModelOptNvfp4GdnBindings {
            z: ModelOptNvfp4LinearBindings {
                weight_scale_2: f32_scalar_view("z-weight", &mismatched_weight_scale),
                ..z
            },
            ..bindings
        }
        .materialize()
        .unwrap_err();
        assert_eq!(scale_error.code(), CheckpointErrorCode::SourceBinding);
        assert!(
            scale_error
                .to_string()
                .contains("QKV/Z weight_scale_2 values differ")
        );

        let materialized = bindings.materialize().unwrap();
        let input_weight_source = [qkv_weight.as_slice(), z_weight.as_slice()].concat();
        let input_scale_source = [qkv_scale.as_slice(), z_scale.as_slice()].concat();
        let control_weight_source = [a_weight.as_slice(), b_weight.as_slice()].concat();
        let mut control_scale_source = [a_scale.as_slice(), b_scale.as_slice()].concat();
        control_scale_source.resize(ROWS * GROUPS, 0);

        assert_eq!(materialized.input_weight_e2m1, input_weight_source);
        assert_eq!(
            materialized.input_scale_e4m3_swizzled,
            block_scale_oracle(&input_scale_source, 2 * ROWS, GROUPS)
        );
        assert_eq!(
            &materialized.control_weight_e2m1_padded[..control_weight_source.len()],
            control_weight_source
        );
        assert!(
            materialized.control_weight_e2m1_padded[control_weight_source.len()..]
                .iter()
                .all(|code| *code == 0)
        );
        assert_eq!(
            materialized.control_scale_e4m3_swizzled,
            block_scale_oracle(&control_scale_source, ROWS, GROUPS)
        );
        assert_eq!(
            materialized.output.weight_e2m1.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(
            materialized.output.scale_e4m3_swizzled,
            block_scale_oracle(&output_scale, ROWS, GROUPS)
        );
        assert_eq!(
            (materialized.input_rows, materialized.input_columns),
            (256, 128)
        );
        assert_eq!(
            (
                materialized.control_rows,
                materialized.control_padded_rows,
                materialized.control_columns,
            ),
            (64, 128, 128)
        );
        assert_eq!(materialized.input_scale.to_bits(), 0.25f32.to_bits());
        assert_eq!(
            materialized.input_weight_scale_2.to_bits(),
            0.125f32.to_bits()
        );
        assert_eq!(materialized.input_scale_divisor, 4.0);
        assert_eq!(materialized.input_weight_scale_divisor, 8.0);
        assert_eq!(materialized.control_input_scale_divisor, 4.0);
        assert_eq!(materialized.control_weight_scale_divisor, 2.0);
        assert_eq!(materialized.convolution_weight.word(0), Some(0x3f80));
        assert_eq!(materialized.a_log.word(0), Some(0x4000));
        assert_eq!(materialized.dt_bias.word(0), Some(0x4040));
        assert_eq!(materialized.norm.word(0), Some(0x4080));
        assert_eq!(materialized.input_norm.word(0), Some(0x40a0));
        assert_eq!(materialized.post_attention_norm.word(0), Some(0x40c0));
        assert_eq!(materialized.layer, 0);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT with the pinned complete Qwen3.5 checkpoint"]
    fn qwen35_source_gdn_layer0_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT")
            .expect("TUISKO_QWEN35_SNAPSHOT is required for the source-backed gate");
        let snapshot = CheckpointSnapshot::<Qwen35_9B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = ModelOptNvfp4GdnBindings::bind(&snapshot, 0).unwrap();
        let input_weight_source = [bindings.qkv.weight.bytes(), bindings.z.weight.bytes()].concat();
        let input_scale_source = [
            bindings.qkv.block_scale.codes(),
            bindings.z.block_scale.codes(),
        ]
        .concat();
        let control_weight_source = [
            bindings.a_control.weight.bytes(),
            bindings.b_control.weight.bytes(),
        ]
        .concat();
        let mut control_scale_source = [
            bindings.a_control.block_scale.codes(),
            bindings.b_control.block_scale.codes(),
        ]
        .concat();
        let output_weight = bindings.output.weight.bytes();
        let materialized = bindings.materialize().unwrap();
        control_scale_source.resize(128 * (Qwen35_9B::HIDDEN / 16), 0);

        assert_eq!(materialized.input_rows, Qwen35_9B::GDN_INPUT_ROWS);
        assert_eq!(materialized.input_columns, Qwen35_9B::HIDDEN);
        assert_eq!(materialized.input_weight_e2m1, input_weight_source);
        assert_eq!(
            materialized.input_scale_e4m3_swizzled,
            block_scale_oracle(
                &input_scale_source,
                Qwen35_9B::GDN_INPUT_ROWS,
                Qwen35_9B::HIDDEN / 16,
            )
        );
        assert_eq!(materialized.control_rows, 2 * Qwen35_9B::GDN_CONTROL_ROWS);
        assert_eq!(materialized.control_padded_rows, 128);
        assert_eq!(
            &materialized.control_weight_e2m1_padded[..control_weight_source.len()],
            control_weight_source
        );
        assert!(
            materialized.control_weight_e2m1_padded[control_weight_source.len()..]
                .iter()
                .all(|code| *code == 0)
        );
        assert_eq!(
            materialized.control_scale_e4m3_swizzled,
            block_scale_oracle(&control_scale_source, 128, Qwen35_9B::HIDDEN / 16,)
        );
        assert_eq!(
            materialized.output.weight_e2m1.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.layer, 0);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT with the pinned complete Qwen3.5 checkpoint"]
    fn qwen35_source_attention_layer3_materializes_losslessly() {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT")
            .expect("TUISKO_QWEN35_SNAPSHOT is required for the source-backed gate");
        let snapshot = CheckpointSnapshot::<Qwen35_9B>::open(std::path::Path::new(&root)).unwrap();
        let bindings = ModelOptNvfp4AttentionBindings::bind(&snapshot, 3).unwrap();
        let query_weight = bindings.query_gate.weight.bytes();
        let key_weight = bindings.key.weight.bytes();
        let value_weight = bindings.value.weight.bytes();
        let qkv_scale_source = [
            bindings.query_gate.block_scale.codes(),
            bindings.key.block_scale.codes(),
            bindings.value.block_scale.codes(),
        ]
        .concat();
        let output_weight = bindings.output.weight.bytes();
        let output_scale = bindings.output.block_scale.codes();
        let materialized = bindings.materialize().unwrap();

        assert_eq!(materialized.qkv_rows, Qwen35_9B::ATTENTION_QKV_ROWS);
        assert_eq!(materialized.qkv_columns, Qwen35_9B::HIDDEN);
        assert_eq!(materialized.qkv_weight_e2m1.len(), 20_971_520);
        assert_eq!(materialized.qkv_scale_e4m3_swizzled.len(), 2_621_440);
        assert_eq!(
            materialized.qkv_weight_e2m1,
            [query_weight, key_weight, value_weight].concat()
        );
        assert_eq!(
            materialized.qkv_scale_e4m3_swizzled,
            block_scale_oracle(
                &qkv_scale_source,
                Qwen35_9B::ATTENTION_QKV_ROWS,
                Qwen35_9B::HIDDEN / 16,
            )
        );
        assert_eq!(
            materialized.output.weight_e2m1.as_ptr(),
            output_weight.as_ptr()
        );
        assert_eq!(materialized.output.weight_e2m1.len(), 8_388_608);
        assert_eq!(materialized.output.scale_e4m3_swizzled.len(), 1_048_576);
        assert_eq!(
            materialized.output.scale_e4m3_swizzled,
            block_scale_oracle(
                output_scale,
                Qwen35_9B::HIDDEN,
                Qwen35_9B::ATTENTION_OUTPUT_COLUMNS / 16,
            )
        );
    }

    #[test]
    fn scale_layout_rejects_incompatible_geometry() {
        for (rows, groups, message) in [
            (127, 8, "scale rows 127 are not tiled by 128"),
            (128, 6, "scale groups 6 are not tiled by 4"),
        ] {
            let error = swizzle_scale_planes(&[&[]], rows, groups, 55, "test")
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains(message), "{error}");
        }

        let error = swizzle_scale_planes(&[&[]], ROWS, GROUPS, 55, "test")
            .err()
            .unwrap();

        assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
        assert!(error.to_string().contains("plane length does not match"));
    }

    #[test]
    fn materialization_rejects_invalid_divisors() {
        for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let error = validate_divisor(55, "test", value).err().unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(
                error.to_string().contains("must be finite and positive"),
                "{error}"
            );
        }
    }
}

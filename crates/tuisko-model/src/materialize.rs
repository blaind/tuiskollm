//! Lossless conversion from source bindings to runtime-native host layouts.

use crate::bindings::{
    NVFP4_MLP_LAYER_END, require_full_attention_layer, require_nvfp4_mlp_layer,
    validate_nvfp4_scales,
};
use crate::{
    Bf16View, CheckpointError, CheckpointResult, F32View, FullAttentionQkvBindings,
    ModelOptNvfp4MlpBindings, MtpBindings, Nvfp4DownBindings, Nvfp4GateUpBindings,
};

const SCALE_TILE_ROWS: usize = 128;
const SCALE_TILE_GROUPS: usize = 4;
const SCALE_TILE_BYTES: usize = SCALE_TILE_ROWS * SCALE_TILE_GROUPS;
const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;

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
            &[
                self.query_gate_weight.codes(),
                self.key_weight.codes(),
                self.value_weight.codes(),
            ],
            &format!("layer-{} full-attention QKV weights", self.layer),
        )?;
        let scale_bf16 = gather_source_planes(
            &[
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
            &[
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

fn gather_source_planes(planes: &[&[u8]], role: &str) -> CheckpointResult<Vec<u8>> {
    let bytes = planes.iter().try_fold(0usize, |bytes, plane| {
        bytes
            .checked_add(plane.len())
            .ok_or_else(|| CheckpointError::source_binding(format!("{role} length overflows")))
    })?;

    let mut gathered = Vec::new();

    gathered.try_reserve_exact(bytes).map_err(|_| {
        CheckpointError::source_binding(format!("{role} cannot reserve {bytes} host bytes"))
    })?;

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

    for row in 0..rows {
        let source_plane = row / rows_per_plane;
        let source_row = row % rows_per_plane;

        for group in 0..groups_per_row {
            let source = planes[source_plane][source_row * groups_per_row + group];
            swizzled[nvfp4_scale_offset(row, group, groups_per_row)] = source;
        }
    }

    Ok(swizzled)
}

fn nvfp4_scale_offset(row: usize, group: usize, groups_per_row: usize) -> usize {
    let persistent_tile = row / SCALE_TILE_ROWS;
    let row_in_tile = row % SCALE_TILE_ROWS;
    let row_mod32 = row_in_tile % 32;
    let row_quartile = row_in_tile / 32;
    let scale_tile = group / SCALE_TILE_GROUPS;
    let scale_lane = group % SCALE_TILE_GROUPS;

    (persistent_tile * (groups_per_row / SCALE_TILE_GROUPS) + scale_tile) * SCALE_TILE_BYTES
        + row_mod32 * 16
        + row_quartile * 4
        + scale_lane
}

#[cfg(test)]
mod tests {
    use super::{SCALE_TILE_GROUPS, SCALE_TILE_ROWS, swizzle_scale_planes, validate_divisor};
    use crate::{
        Bf16View, CheckpointErrorCode, DType, F32View, Fp8E4M3View, FullAttentionQkvBindings,
        ModelOptNvfp4LinearBindings, ModelOptNvfp4MlpBindings, Nvfp4DownBindings,
        Nvfp4GateUpBindings, TensorView, U8View,
    };

    const ROWS: usize = 128;
    const GROUPS: usize = 8;
    const COLUMNS: usize = GROUPS * 16;
    const PACKED_COLUMNS: usize = COLUMNS / 2;

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
        (0..ROWS * GROUPS)
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

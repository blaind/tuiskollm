use crate::bindings::validate_nvfp4_scales;
use crate::{CheckpointError, CheckpointResult, Nvfp4DownBindings, Nvfp4GateUpBindings};

const SCALE_TILE_ROWS: usize = 128;
const SCALE_TILE_GROUPS: usize = 4;
const SCALE_TILE_BYTES: usize = SCALE_TILE_ROWS * SCALE_TILE_GROUPS;
const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;

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
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4GateUp<'a>> {
        let [gate_rows, packed_columns] = host_shape(self.gate_weight.shape(), "gate weights")?;
        let up_shape = host_shape(self.up_weight.shape(), "up weights")?;
        let [gate_scale_rows, groups] = host_shape(self.gate_scale.shape(), "gate scales")?;
        let up_scale_shape = host_shape(self.up_scale.shape(), "up scales")?;

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

impl<'a> Nvfp4DownBindings<'a> {
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4Down<'a>> {
        let [rows, packed_columns] = host_shape(self.weight.shape(), "down weights")?;
        let [scale_rows, groups] = host_shape(self.scale.shape(), "down scales")?;

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

fn host_shape(shape: &[u64; 2], role: &str) -> CheckpointResult<[usize; 2]> {
    let rows = usize::try_from(shape[0]).map_err(|_| {
        CheckpointError::source_binding(format!("NVFP4 {role} row count exceeds this host"))
    })?;
    let columns = usize::try_from(shape[1]).map_err(|_| {
        CheckpointError::source_binding(format!("NVFP4 {role} column count exceeds this host"))
    })?;

    Ok([rows, columns])
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
        CheckpointErrorCode, DType, Fp8E4M3View, Nvfp4DownBindings, Nvfp4GateUpBindings,
        TensorView, U8View,
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

        let materialized = bindings.materialize().unwrap();
        let expected = block_scale_oracle(&scale, ROWS, GROUPS);

        assert_eq!(materialized.scale_e4m3_swizzled, expected);
        assert_eq!(materialized.weight_e2m1, weight);
        assert_eq!(materialized.weight_e2m1.as_ptr(), weight.as_ptr());
        assert_eq!((materialized.rows, materialized.columns), (128, 128));
        assert_eq!(materialized.layer, 55);
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

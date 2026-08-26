//! NVFP4 gate/up and down source carriers shared by the compressed-tensors and ModelOpt MLP routes.

use crate::Arch;
use crate::common::inventory::CheckpointSnapshot;
use crate::common::materialized::{MaterializedMemory, sealed};
use crate::common::modelopt_codec::{logical_columns, validate_divisor};
use crate::common::routes::{require_nvfp4_mlp_layer, validate_nvfp4_scales};
use crate::common::scale_swizzle::{PlaneGatherer, host_shape};
use crate::common::source_binding::{SourceLayerBinding, sealed as binding_sealed};
use crate::{CheckpointError, CheckpointResult, Fp8E4M3View, U8View};

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

impl binding_sealed::Sealed for Nvfp4GateUpBindings<'_> {}

impl<'a, A: Arch> SourceLayerBinding<'a, A> for Nvfp4GateUpBindings<'a> {
    type Materialized = MaterializedNvfp4GateUp<'a>;

    fn bind(snapshot: &'a CheckpointSnapshot<A>, layer: usize) -> CheckpointResult<Self> {
        Self::bind::<A>(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Self::materialize(self)
    }
}

impl sealed::Sealed for MaterializedNvfp4GateUp<'_> {}

impl MaterializedMemory for MaterializedNvfp4GateUp<'_> {
    fn host_bytes(&self) -> usize {
        self.scale_e4m3_swizzled.len()
    }
}

impl<'a> Nvfp4GateUpBindings<'a> {
    /// Materializes the fused gate/up scale layout without requantizing source values.
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4GateUp<'a>> {
        require_nvfp4_mlp_layer(self.layer, self.layer_count)?;

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

        let scale_e4m3_swizzled = PlaneGatherer::swizzle_scales(
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

impl binding_sealed::Sealed for Nvfp4DownBindings<'_> {}

impl<'a, A: Arch> SourceLayerBinding<'a, A> for Nvfp4DownBindings<'a> {
    type Materialized = MaterializedNvfp4Down<'a>;

    fn bind(snapshot: &'a CheckpointSnapshot<A>, layer: usize) -> CheckpointResult<Self> {
        Self::bind::<A>(snapshot, layer)
    }

    fn materialize(self) -> CheckpointResult<Self::Materialized> {
        Self::materialize(self)
    }
}

impl sealed::Sealed for MaterializedNvfp4Down<'_> {}

impl MaterializedMemory for MaterializedNvfp4Down<'_> {
    fn host_bytes(&self) -> usize {
        self.scale_e4m3_swizzled.len()
    }
}

impl<'a> Nvfp4DownBindings<'a> {
    /// Materializes the down-projection scale layout without requantizing source values.
    pub fn materialize(self) -> CheckpointResult<MaterializedNvfp4Down<'a>> {
        require_nvfp4_mlp_layer(self.layer, self.layer_count)?;

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
            PlaneGatherer::swizzle_scales(&[self.scale.codes()], rows, groups, self.layer, "down")?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointErrorCode;
    use crate::MaterializedMemory;
    use crate::common::test_support::sources::{
        GROUPS, PACKED_COLUMNS, ROWS, block_scale_oracle, fp8_view, scale_codes, u8_view,
    };

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
            layer_count: 64,
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

        let count_error = Nvfp4GateUpBindings {
            layer: 40,
            layer_count: 40,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(count_error.code(), CheckpointErrorCode::SourceBinding);

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
        assert_eq!(materialized.host_bytes(), 2_048);
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
            layer_count: 64,
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

        let count_error = Nvfp4DownBindings {
            layer: 40,
            layer_count: 40,
            ..bindings
        }
        .materialize()
        .err()
        .unwrap();

        assert_eq!(count_error.code(), CheckpointErrorCode::SourceBinding);

        let materialized = bindings.materialize().unwrap();
        let expected = block_scale_oracle(&scale, ROWS, GROUPS);

        assert_eq!(materialized.scale_e4m3_swizzled, expected);
        assert_eq!(materialized.weight_e2m1, weight);
        assert_eq!(materialized.weight_e2m1.as_ptr(), weight.as_ptr());
        assert_eq!(materialized.host_bytes(), 1_024);
        assert_eq!((materialized.rows, materialized.columns), (128, 128));
        assert_eq!(materialized.layer, 55);
    }
}

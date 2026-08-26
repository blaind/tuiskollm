//! ModelOpt scale validation and reciprocal divisor conversion shared by the ModelOpt targets.

use crate::common::routes::{codec_columns, positive_rank_zero_f32, validate_nvfp4_scales};
use crate::common::scale_swizzle::{host_shape, swizzle_scale_planes};
use crate::{CheckpointError, CheckpointResult, F32View, Fp8E4M3View, TensorView, U8View};

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
    pub(crate) fn bind_from(
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

const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES_PER_BYTE: usize = 2;

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

pub(crate) fn materialize_modelopt_linear<'a>(
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

pub(crate) fn modelopt_scale(
    layer: usize,
    role: &str,
    scale: &F32View<'_, 0>,
) -> CheckpointResult<f32> {
    let value = scale.value(0).expect("validated scalar has one value");

    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} ModelOpt NVFP4 {role} must be finite and positive, observed {value}"
        )));
    }

    Ok(value)
}

pub(crate) fn require_same_modelopt_scale(
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

pub(crate) fn reciprocal_scale(layer: usize, role: &str, scale: f32) -> CheckpointResult<f32> {
    let divisor = 1.0 / scale;
    validate_divisor(layer, role, divisor)?;

    Ok(divisor)
}

pub(crate) fn logical_columns(
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

pub(crate) fn validate_divisor(layer: usize, role: &str, value: f32) -> CheckpointResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CheckpointError::source_binding(format!(
            "layer-{layer} NVFP4 {role} scale divisor must be finite and positive, observed {value}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointErrorCode;

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

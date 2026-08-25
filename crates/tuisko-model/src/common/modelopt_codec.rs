//! ModelOpt scale validation and reciprocal divisor conversion shared by the ModelOpt targets.

use crate::common::materialized::{MaterializedMemory, sealed};
use crate::common::routes::{
    E2M1_VALUES_PER_BYTE, NVFP4_GROUP_SIZE, codec_columns, positive_rank_zero_f32,
    validate_nvfp4_scales,
};
use crate::common::scale_swizzle::{PlaneGatherer, host_shape};
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

impl sealed::Sealed for MaterializedModelOptNvfp4Linear<'_> {}

impl MaterializedMemory for MaterializedModelOptNvfp4Linear<'_> {
    fn host_bytes(&self) -> usize {
        self.scale_e4m3_swizzled.len()
    }
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
    let input_scale = ModelOptScaleCodec::source_scale(
        layer,
        &format!("{role} input_scale"),
        &binding.input_scale,
    )?;
    let weight_scale_2 = ModelOptScaleCodec::source_scale(
        layer,
        &format!("{role} weight_scale_2"),
        &binding.weight_scale_2,
    )?;
    let input_scale_divisor =
        ModelOptScaleCodec::to_reciprocal_divisor(layer, &format!("{role} input"), input_scale)?;
    let weight_scale_divisor = ModelOptScaleCodec::to_reciprocal_divisor(
        layer,
        &format!("{role} weight"),
        weight_scale_2,
    )?;
    let scale_e4m3_swizzled =
        PlaneGatherer::swizzle_scales(&[binding.block_scale.codes()], rows, groups, layer, role)?;

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

/// ModelOpt scalar-convention codec at the checkpoint-adapter boundary.
///
/// ModelOpt stores one positive rank-zero F32 per plane family holding
/// `amax / (E2M1_MAX * E4M3_MAX)`, while the qualified NVFP4 kernels consume the reciprocal
/// global divisor `1 / scale`. Every ModelOpt materialization site converts here, and the
/// conversion changes scalar convention only: it never decodes or requantizes a represented
/// E2M1, E4M3, or BF16 source word, and it never permutes a plane.
///
/// The compressed-tensors target never reaches this codec. Its `input_global_scale` and
/// `weight_global_scale` source words are already stored in the divisor convention, so that
/// adapter binds them directly and both adapters converge on `validate_divisor`.
pub(crate) struct ModelOptScaleCodec;

impl ModelOptScaleCodec {
    /// Reads one ModelOpt source scale, rejecting non-finite and non-positive words.
    pub(crate) fn source_scale(
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

    /// Requires two ModelOpt source scales to carry identical F32 bits.
    pub(crate) fn require_same_source_scale(
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

    /// Converts one admitted ModelOpt source scale into the kernel divisor convention.
    pub(crate) fn to_reciprocal_divisor(
        layer: usize,
        role: &str,
        scale: f32,
    ) -> CheckpointResult<f32> {
        let divisor = 1.0 / scale;
        validate_divisor(layer, role, divisor)?;

        Ok(divisor)
    }
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

/// Admits one kernel-convention global divisor, whichever adapter produced it.
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
    use crate::common::test_support::sources::{
        COLUMNS, GROUPS, PACKED_COLUMNS, ROWS, block_scale_oracle, f32_scalar_view, fp8_view,
        scale_codes, u8_view,
    };

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

    #[test]
    fn reciprocal_divisor_pins_exact_scalar_bits() {
        // Independently computed single-precision reciprocals of the ModelOpt convention
        // `amax / (E2M1_MAX * E4M3_MAX)`. The 1/448 row pins that the conversion is one f32
        // reciprocal, not an exact algebraic inverse.
        for (scale_bits, divisor_bits) in [
            (0x3f00_0000u32, 0x4000_0000u32),
            (0x4000_0000, 0x3f00_0000),
            (0x4040_0000, 0x3eaa_aaab),
            (0x3e00_0000, 0x4100_0000),
            (0x3c03_0c31, 0x42fa_0be8),
            (0x3b12_4925, 0x43df_ffff),
        ] {
            let scale = f32::from_bits(scale_bits);
            let divisor = ModelOptScaleCodec::to_reciprocal_divisor(55, "weight", scale).unwrap();

            assert_eq!(divisor.to_bits(), divisor_bits, "scale {scale_bits:#010x}");
        }
    }

    #[test]
    fn source_scale_reads_exact_source_words() {
        for bits in [0x3c03_0c31u32, 0x3b12_4925, 0x3f80_0000] {
            let bytes = bits.to_le_bytes();
            let view = f32_scalar_view("input_scale", &bytes);
            let value = ModelOptScaleCodec::source_scale(55, "input_scale", &view).unwrap();

            assert_eq!(value.to_bits(), bits);
        }

        for bits in [0f32.to_bits(), (-1f32).to_bits(), f32::NAN.to_bits()] {
            let bytes = bits.to_le_bytes();
            let view = f32_scalar_view("input_scale", &bytes);
            let error = ModelOptScaleCodec::source_scale(55, "input_scale", &view)
                .err()
                .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(
                error.to_string().contains("must be finite and positive"),
                "{error}"
            );
        }
    }

    #[test]
    fn require_same_source_scale_compares_source_bits() {
        let first = 1f32.to_bits().to_le_bytes();
        let same = 1f32.to_bits().to_le_bytes();
        let nearest = (1f32.to_bits() + 1).to_le_bytes();
        let negative_zero = (-0f32).to_bits().to_le_bytes();
        let positive_zero = 0f32.to_bits().to_le_bytes();

        ModelOptScaleCodec::require_same_source_scale(
            55,
            "gate/up input_scale",
            &f32_scalar_view("gate", &first),
            &f32_scalar_view("up", &same),
        )
        .unwrap();

        for (left, right) in [(&first, &nearest), (&positive_zero, &negative_zero)] {
            let error = ModelOptScaleCodec::require_same_source_scale(
                55,
                "gate/up input_scale",
                &f32_scalar_view("gate", left),
                &f32_scalar_view("up", right),
            )
            .err()
            .unwrap();

            assert_eq!(error.code(), CheckpointErrorCode::SourceBinding);
            assert!(error.to_string().contains("values differ"), "{error}");
        }
    }

    #[test]
    fn modelopt_linear_materialization_only_converts_scalar_convention() {
        let weight_shape = [ROWS as u64, PACKED_COLUMNS as u64];
        let scale_shape = [ROWS as u64, GROUPS as u64];
        let weight = (0..ROWS * PACKED_COLUMNS)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let block_scale = scale_codes(7);
        let input_scale = 0x3c03_0c31u32.to_le_bytes();
        let weight_scale_2 = 0x3b12_4925u32.to_le_bytes();

        let materialized = materialize_modelopt_linear(
            ModelOptNvfp4LinearBindings {
                weight: u8_view("weight", &weight_shape, &weight),
                block_scale: fp8_view("weight_scale", &scale_shape, &block_scale),
                input_scale: f32_scalar_view("input_scale", &input_scale),
                weight_scale_2: f32_scalar_view("weight_scale_2", &weight_scale_2),
                rows: ROWS,
                columns: COLUMNS,
            },
            55,
            "test",
        )
        .unwrap();

        assert_eq!(materialized.weight_e2m1, weight);
        assert_eq!(materialized.weight_e2m1.as_ptr(), weight.as_ptr());
        assert_eq!(
            materialized.scale_e4m3_swizzled,
            block_scale_oracle(&block_scale, ROWS, GROUPS)
        );
        assert_eq!(materialized.input_scale.to_bits(), 0x3c03_0c31);
        assert_eq!(materialized.weight_scale_2.to_bits(), 0x3b12_4925);
        assert_eq!(materialized.input_scale_divisor.to_bits(), 0x42fa_0be8);
        assert_eq!(materialized.weight_scale_divisor.to_bits(), 0x43df_ffff);
        assert_eq!((materialized.rows, materialized.columns), (ROWS, COLUMNS));
        assert_eq!(materialized.layer, 55);
    }
}

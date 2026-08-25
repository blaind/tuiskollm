//! Layer-route and source-codec predicates shared by more than one admitted target.

use crate::{Arch, CheckpointError, CheckpointResult, F32View, TensorView};

// These are source-codec facts, not architecture geometry.
/// Exclusive decoder-layer boundary for the checkpoint's packed NVFP4 MLP planes.
pub const NVFP4_MLP_LAYER_END: usize = 56;

pub(crate) const NVFP4_GROUP_SIZE: usize = 16;
pub(crate) const E2M1_VALUES_PER_BYTE: usize = 2;

pub(crate) fn require_nvfp4_mlp_layer(layer: usize, layer_count: usize) -> CheckpointResult<()> {
    if layer >= layer_count || layer >= NVFP4_MLP_LAYER_END {
        return Err(CheckpointError::source_binding(format!(
            "layer {layer} does not use the admitted NVFP4 MLP source contract"
        )));
    }

    Ok(())
}

pub(crate) fn require_gdn_layer<A: Arch>(layer: usize) -> CheckpointResult<()> {
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

pub(crate) fn codec_columns(width: usize, divisor: usize, role: &str) -> CheckpointResult<u64> {
    if !width.is_multiple_of(divisor) {
        return Err(CheckpointError::source_binding(format!(
            "architecture width {width} is not divisible by the {role} divisor {divisor}"
        )));
    }

    Ok((width / divisor) as u64)
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

pub(crate) fn positive_rank_zero_f32(tensor: TensorView<'_>) -> CheckpointResult<F32View<'_, 0>> {
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

pub(crate) fn require_same_rank_zero_f32(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointErrorCode;
    use crate::common::test_support::sources::Nvfp4Arch;

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
}

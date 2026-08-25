//! Single-allocation layout for one dense-FP8 GDN decoder layer.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::Arch;

const ALIGNMENT: usize = 256;
pub(crate) const MAX_ROWS: usize = 1_024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) input_activation_codes: ArenaRegion<u8>,
    pub(crate) input_activation_scales: ArenaRegion<f32>,
    pub(crate) input_weight_codes: ArenaRegion<u8>,
    pub(crate) input_weight_scales: ArenaRegion<u16>,
    pub(crate) projected: ArenaRegion<u16>,
    pub(crate) control_weights: ArenaRegion<u16>,
    pub(crate) a_log: ArenaRegion<u16>,
    pub(crate) dt_bias: ArenaRegion<u16>,
    pub(crate) convolution_weights: ArenaRegion<u16>,
    pub(crate) state_rows: ArenaRegion<u32>,
    pub(crate) history: ArenaRegion<u16>,
    pub(crate) log_decay: ArenaRegion<f32>,
    pub(crate) beta: ArenaRegion<f32>,
    pub(crate) convolved: ArenaRegion<u16>,
    pub(crate) recurrent_norm: ArenaRegion<u16>,
    pub(crate) state: ArenaRegion<f32>,
    pub(crate) recurrent_plane: ArenaRegion<f32>,
    pub(crate) recurrent_output: ArenaRegion<u16>,
    pub(crate) output_activation_codes: ArenaRegion<u8>,
    pub(crate) output_activation_scales: ArenaRegion<f32>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) output_weight_scales: ArenaRegion<u16>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) gate_up_activation_codes: ArenaRegion<u8>,
    pub(crate) gate_up_activation_scales: ArenaRegion<f32>,
    pub(crate) gate_up_weight_codes: ArenaRegion<u8>,
    pub(crate) gate_up_weight_scales: ArenaRegion<u16>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) down_activation_codes: ArenaRegion<u8>,
    pub(crate) down_activation_scales: ArenaRegion<f32>,
    pub(crate) down_weight_codes: ArenaRegion<u8>,
    pub(crate) down_weight_scales: ArenaRegion<u16>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked source-weight, recurrent-state, and workspace regions for one layer.
#[derive(Clone, Debug)]
pub struct DenseFp8GdnLayerLayout {
    builder: ArenaLayout,
    regions: GdnLayerRegions,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl DenseFp8GdnLayerLayout {
    /// Reserves every plane for the exact decode and prefill routes.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        let row_hidden = product("GDN row-hidden elements", MAX_ROWS, A::HIDDEN)?;
        let row_input = product("GDN projected elements", MAX_ROWS, A::GDN_INPUT_ROWS)?;
        let row_qkv = product("GDN convolved elements", MAX_ROWS, A::GDN_QKV_ROWS)?;
        let row_value = product("GDN recurrent output elements", MAX_ROWS, A::GDN_VALUE_ROWS)?;
        let row_control = product("GDN control elements", MAX_ROWS, A::GDN_CONTROL_ROWS)?;
        let row_intermediate = product("GDN MLP elements", MAX_ROWS, A::INTERMEDIATE)?;
        let input_weights = product("GDN input weights", A::GDN_INPUT_ROWS, A::HIDDEN)?;
        let control_weights = product(
            "GDN A/B control weights",
            product("GDN A/B control rows", 2, A::GDN_CONTROL_ROWS)?,
            A::HIDDEN,
        )?;
        let convolution_weights = product(
            "GDN convolution weights",
            A::GDN_QKV_ROWS,
            A::LINEAR_CONV_KERNEL_DIM,
        )?;
        let history = product(
            "GDN causal history",
            product("GDN history rows", MAX_BATCH, A::GDN_QKV_ROWS)?,
            A::LINEAR_CONV_KERNEL_DIM
                .checked_sub(1)
                .ok_or_else(|| EngineError::layout("GDN convolution width is zero"))?,
        )?;
        let state = product(
            "GDN recurrent state",
            product("GDN state heads", MAX_BATCH, A::GDN_CONTROL_ROWS)?,
            product(
                "GDN state head matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;
        let output_weights = product("GDN output weights", A::HIDDEN, A::GDN_VALUE_ROWS)?;
        let gate_up_weights = product(
            "dense-FP8 gate/up weights",
            product("dense-FP8 gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weights = product("dense-FP8 down weights", A::HIDDEN, A::INTERMEDIATE)?;

        let mut builder = ArenaLayout::new();
        let regions = GdnLayerRegions {
            residual_input: builder.reserve(row_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            input_activation_codes: builder.reserve(row_hidden, ALIGNMENT)?,
            input_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            input_weight_codes: builder.reserve(input_weights, ALIGNMENT)?,
            input_weight_scales: builder.reserve(A::GDN_INPUT_ROWS, ALIGNMENT)?,
            projected: builder.reserve(row_input, ALIGNMENT)?,
            control_weights: builder.reserve(control_weights, ALIGNMENT)?,
            a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            convolution_weights: builder.reserve(convolution_weights, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            history: builder.reserve(history, ALIGNMENT)?,
            log_decay: builder.reserve(row_control, ALIGNMENT)?,
            beta: builder.reserve(row_control, ALIGNMENT)?,
            convolved: builder.reserve(row_qkv, ALIGNMENT)?,
            recurrent_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
            recurrent_plane: builder.reserve(row_value, ALIGNMENT)?,
            recurrent_output: builder.reserve(row_value, ALIGNMENT)?,
            output_activation_codes: builder.reserve(row_value, ALIGNMENT)?,
            output_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weights, ALIGNMENT)?,
            output_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(row_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            gate_up_activation_codes: builder.reserve(row_hidden, ALIGNMENT)?,
            gate_up_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            gate_up_weight_codes: builder.reserve(gate_up_weights, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(2 * A::INTERMEDIATE, ALIGNMENT)?,
            swiglu: builder.reserve(row_intermediate, ALIGNMENT)?,
            down_activation_codes: builder.reserve(row_intermediate, ALIGNMENT)?,
            down_activation_scales: builder.reserve(MAX_ROWS, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weights, ALIGNMENT)?,
            down_weight_scales: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mlp_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(row_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "dense-FP8 GDN resident weight bytes",
            &[
                regions.input_norm.byte_len(),
                regions.input_weight_codes.byte_len(),
                regions.input_weight_scales.byte_len(),
                regions.control_weights.byte_len(),
                regions.a_log.byte_len(),
                regions.dt_bias.byte_len(),
                regions.convolution_weights.byte_len(),
                regions.recurrent_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.output_weight_scales.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.gate_up_weight_codes.byte_len(),
                regions.gate_up_weight_scales.byte_len(),
                regions.down_weight_codes.byte_len(),
                regions.down_weight_scales.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "dense-FP8 GDN workspace bytes",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.input_activation_codes.byte_len(),
                regions.input_activation_scales.byte_len(),
                regions.projected.byte_len(),
                regions.state_rows.byte_len(),
                regions.history.byte_len(),
                regions.log_decay.byte_len(),
                regions.beta.byte_len(),
                regions.convolved.byte_len(),
                regions.state.byte_len(),
                regions.recurrent_plane.byte_len(),
                regions.recurrent_output.byte_len(),
                regions.output_activation_codes.byte_len(),
                regions.output_activation_scales.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.mlp_normalized.byte_len(),
                regions.gate_up_activation_codes.byte_len(),
                regions.gate_up_activation_scales.byte_len(),
                regions.swiglu.byte_len(),
                regions.down_activation_codes.byte_len(),
                regions.down_activation_scales.byte_len(),
                regions.mlp_branch.byte_len(),
                regions.residual_output.byte_len(),
                regions.next_normalized.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            regions,
            resident_weight_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    pub(crate) const fn regions(&self) -> GdnLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed norm, mixer, and MLP weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact address-stable working and recurrent-state bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights plus workspace, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.workspace_bytes
    }
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn sum(name: &str, values: &[usize]) -> EngineResult<usize> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
    })
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, DenseFp8GdnLayerLayout};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn qwen_dense_fp8_gdn_layer_byte_accounting_is_exact() {
        let layout = DenseFp8GdnLayerLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 383_949_248);
        assert_eq!(layout.workspace_bytes(), 272_482_336);
        assert_eq!(layout.owner_bytes(), 656_431_584);
        assert_eq!(layout.arena_bytes(), 656_432_128);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 544);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = DenseFp8GdnLayerLayout::build::<Qwen38_27B>().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
            span(regions.input_activation_codes),
            span(regions.input_activation_scales),
            span(regions.input_weight_codes),
            span(regions.input_weight_scales),
            span(regions.projected),
            span(regions.control_weights),
            span(regions.a_log),
            span(regions.dt_bias),
            span(regions.convolution_weights),
            span(regions.state_rows),
            span(regions.history),
            span(regions.log_decay),
            span(regions.beta),
            span(regions.convolved),
            span(regions.recurrent_norm),
            span(regions.state),
            span(regions.recurrent_plane),
            span(regions.recurrent_output),
            span(regions.output_activation_codes),
            span(regions.output_activation_scales),
            span(regions.output_weight_codes),
            span(regions.output_weight_scales),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.mlp_normalized),
            span(regions.gate_up_activation_codes),
            span(regions.gate_up_activation_scales),
            span(regions.gate_up_weight_codes),
            span(regions.gate_up_weight_scales),
            span(regions.swiglu),
            span(regions.down_activation_codes),
            span(regions.down_activation_scales),
            span(regions.down_weight_codes),
            span(regions.down_weight_scales),
            span(regions.mlp_branch),
            span(regions.next_norm),
            span(regions.residual_output),
            span(regions.next_normalized),
        ];
        spans.sort_unstable_by_key(|(offset, _)| *offset);
        for &(offset, bytes) in &spans {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in spans.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    #[test]
    fn persistent_state_follows_exact_geometry() {
        let layout = DenseFp8GdnLayerLayout::build::<Qwen38_27B>().unwrap();
        let regions = layout.regions();

        assert_eq!(
            regions.history.len(),
            8 * Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1)
        );
        assert_eq!(
            regions.state.len(),
            8 * Qwen38_27B::GDN_CONTROL_ROWS
                * Qwen38_27B::LINEAR_HEAD_DIM
                * Qwen38_27B::LINEAR_HEAD_DIM
        );
    }

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

//! Single-allocation layout for one Qwen3.5 GDN decoder layer.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
const PADDED_CONTROL_ROWS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35GdnLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) input_weight_codes: ArenaRegion<u8>,
    pub(crate) input_weight_scales: ArenaRegion<u8>,
    pub(crate) control_weight_codes: ArenaRegion<u8>,
    pub(crate) control_weight_scales: ArenaRegion<u8>,
    pub(crate) projected: ArenaRegion<u16>,
    pub(crate) projected_controls: ArenaRegion<u16>,
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
    pub(crate) recurrent_output: ArenaRegion<u16>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) output_weight_scales: ArenaRegion<u8>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) mlp_normalized: ArenaRegion<u16>,
    pub(crate) gate_up_activation_codes: ArenaRegion<u8>,
    pub(crate) gate_up_activation_scales: ArenaRegion<u8>,
    pub(crate) gate_weight_codes: ArenaRegion<u8>,
    pub(crate) up_weight_codes: ArenaRegion<u8>,
    pub(crate) gate_up_weight_scales: ArenaRegion<u8>,
    pub(crate) swiglu: ArenaRegion<u16>,
    pub(crate) down_weight_codes: ArenaRegion<u8>,
    pub(crate) down_weight_scales: ArenaRegion<u8>,
    pub(crate) mlp_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked source weights, recurrent state, and workspace for one Qwen3.5 GDN layer.
#[derive(Clone, Debug)]
pub struct Qwen35GdnLayerLayout {
    builder: ArenaLayout,
    regions: Qwen35GdnLayerRegions,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen35GdnLayerLayout {
    /// Reserves every source plane and exact decode seam for `B=1..=8`.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen35_9B;
        require_geometry::<A>()?;

        let batch_hidden = product("Qwen3.5 GDN batch-hidden elements", MAX_BATCH, A::HIDDEN)?;
        let batch_input = product(
            "Qwen3.5 GDN projected elements",
            MAX_BATCH,
            A::GDN_INPUT_ROWS,
        )?;
        let batch_qkv = product("Qwen3.5 GDN convolved elements", MAX_BATCH, A::GDN_QKV_ROWS)?;
        let batch_value = product(
            "Qwen3.5 GDN recurrent output elements",
            MAX_BATCH,
            A::GDN_VALUE_ROWS,
        )?;
        let batch_control = product(
            "Qwen3.5 GDN control elements",
            MAX_BATCH,
            A::GDN_CONTROL_ROWS,
        )?;
        let batch_padded_control = product(
            "Qwen3.5 GDN padded control elements",
            MAX_BATCH,
            PADDED_CONTROL_ROWS,
        )?;
        let batch_intermediate = product("Qwen3.5 GDN MLP elements", MAX_BATCH, A::INTERMEDIATE)?;
        let input_weight_codes = packed_codes(
            "Qwen3.5 GDN input weight codes",
            A::GDN_INPUT_ROWS,
            A::HIDDEN,
        )?;
        let input_weight_scales = scales(
            "Qwen3.5 GDN input weight scales",
            A::GDN_INPUT_ROWS,
            A::HIDDEN,
        )?;
        let control_weight_codes = packed_codes(
            "Qwen3.5 GDN control weight codes",
            PADDED_CONTROL_ROWS,
            A::HIDDEN,
        )?;
        let control_weight_scales = scales(
            "Qwen3.5 GDN control weight scales",
            PADDED_CONTROL_ROWS,
            A::HIDDEN,
        )?;
        let convolution_weights = product(
            "Qwen3.5 GDN convolution weights",
            A::GDN_QKV_ROWS,
            A::LINEAR_CONV_KERNEL_DIM,
        )?;
        let history = product(
            "Qwen3.5 GDN causal history",
            product("Qwen3.5 GDN history rows", MAX_BATCH, A::GDN_QKV_ROWS)?,
            A::LINEAR_CONV_KERNEL_DIM
                .checked_sub(1)
                .ok_or_else(|| EngineError::layout("Qwen3.5 GDN convolution width is zero"))?,
        )?;
        let state = product(
            "Qwen3.5 GDN recurrent state",
            product("Qwen3.5 GDN state heads", MAX_BATCH, A::GDN_CONTROL_ROWS)?,
            product(
                "Qwen3.5 GDN state head matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;
        let output_weight_codes = packed_codes(
            "Qwen3.5 GDN output weight codes",
            A::HIDDEN,
            A::GDN_VALUE_ROWS,
        )?;
        let output_weight_scales = scales(
            "Qwen3.5 GDN output weight scales",
            A::HIDDEN,
            A::GDN_VALUE_ROWS,
        )?;
        let gate_codes = packed_codes("Qwen3.5 GDN gate weight codes", A::INTERMEDIATE, A::HIDDEN)?;
        let gate_up_scales = scales(
            "Qwen3.5 GDN gate/up weight scales",
            product("Qwen3.5 GDN gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weight_codes =
            packed_codes("Qwen3.5 GDN down weight codes", A::HIDDEN, A::INTERMEDIATE)?;
        let down_weight_scales =
            scales("Qwen3.5 GDN down weight scales", A::HIDDEN, A::INTERMEDIATE)?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen35GdnLayerRegions {
            residual_input: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_weight_codes: builder.reserve(input_weight_codes, ALIGNMENT)?,
            input_weight_scales: builder.reserve(input_weight_scales, ALIGNMENT)?,
            control_weight_codes: builder.reserve(control_weight_codes, ALIGNMENT)?,
            control_weight_scales: builder.reserve(control_weight_scales, ALIGNMENT)?,
            projected: builder.reserve(batch_input, ALIGNMENT)?,
            projected_controls: builder.reserve(batch_padded_control, ALIGNMENT)?,
            a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            convolution_weights: builder.reserve(convolution_weights, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            history: builder.reserve(history, ALIGNMENT)?,
            log_decay: builder.reserve(batch_control, ALIGNMENT)?,
            beta: builder.reserve(batch_control, ALIGNMENT)?,
            convolved: builder.reserve(batch_qkv, ALIGNMENT)?,
            recurrent_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
            recurrent_output: builder.reserve(batch_value, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weight_codes, ALIGNMENT)?,
            output_weight_scales: builder.reserve(output_weight_scales, ALIGNMENT)?,
            mixer_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(batch_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            gate_up_activation_codes: builder.reserve(batch_hidden / 2, ALIGNMENT)?,
            gate_up_activation_scales: builder.reserve(batch_hidden / NVFP4_GROUP, ALIGNMENT)?,
            gate_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            up_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(gate_up_scales, ALIGNMENT)?,
            swiglu: builder.reserve(batch_intermediate, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weight_codes, ALIGNMENT)?,
            down_weight_scales: builder.reserve(down_weight_scales, ALIGNMENT)?,
            mlp_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.5 GDN resident weight bytes",
            &[
                regions.input_norm.byte_len(),
                regions.input_weight_codes.byte_len(),
                regions.input_weight_scales.byte_len(),
                regions.control_weight_codes.byte_len(),
                regions.control_weight_scales.byte_len(),
                regions.a_log.byte_len(),
                regions.dt_bias.byte_len(),
                regions.convolution_weights.byte_len(),
                regions.recurrent_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.output_weight_scales.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.gate_weight_codes.byte_len(),
                regions.up_weight_codes.byte_len(),
                regions.gate_up_weight_scales.byte_len(),
                regions.down_weight_codes.byte_len(),
                regions.down_weight_scales.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "Qwen3.5 GDN workspace bytes",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.projected.byte_len(),
                regions.projected_controls.byte_len(),
                regions.state_rows.byte_len(),
                regions.history.byte_len(),
                regions.log_decay.byte_len(),
                regions.beta.byte_len(),
                regions.convolved.byte_len(),
                regions.state.byte_len(),
                regions.recurrent_output.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.mlp_normalized.byte_len(),
                regions.gate_up_activation_codes.byte_len(),
                regions.gate_up_activation_scales.byte_len(),
                regions.swiglu.byte_len(),
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

    pub(crate) const fn regions(&self) -> Qwen35GdnLayerRegions {
        self.regions
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed device weight bytes.
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

fn require_geometry<A: Arch>() -> EngineResult<()> {
    if !A::HIDDEN.is_multiple_of(NVFP4_GROUP)
        || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP)
        || !A::GDN_INPUT_ROWS.is_multiple_of(128)
        || !A::GDN_VALUE_ROWS.is_multiple_of(128)
        || 2 * A::GDN_CONTROL_ROWS > PADDED_CONTROL_ROWS
    {
        return Err(EngineError::layout(
            "Qwen3.5 GDN geometry must satisfy K16, M128, and padded-control tiling",
        ));
    }

    Ok(())
}

fn packed_codes(name: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    product(name, rows, columns / 2)
}

fn scales(name: &str, rows: usize, columns: usize) -> EngineResult<usize> {
    product(name, rows, columns / NVFP4_GROUP)
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
    use super::{ALIGNMENT, Qwen35GdnLayerLayout};
    use tuisko_model::{Arch, Qwen35_9B};

    #[test]
    fn byte_accounting_is_exact() {
        let layout = Qwen35GdnLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 123_068_800);
        assert_eq!(layout.workspace_bytes(), 18_307_104);
        assert_eq!(layout.owner_bytes(), 141_375_904);
        assert_eq!(layout.arena_bytes(), 141_376_512);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 608);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen35GdnLayerLayout::build().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
            span(regions.input_weight_codes),
            span(regions.input_weight_scales),
            span(regions.control_weight_codes),
            span(regions.control_weight_scales),
            span(regions.projected),
            span(regions.projected_controls),
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
            span(regions.recurrent_output),
            span(regions.output_weight_codes),
            span(regions.output_weight_scales),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.mlp_normalized),
            span(regions.gate_up_activation_codes),
            span(regions.gate_up_activation_scales),
            span(regions.gate_weight_codes),
            span(regions.up_weight_codes),
            span(regions.gate_up_weight_scales),
            span(regions.swiglu),
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
    fn state_and_nvfp4_planes_follow_exact_geometry() {
        let layout = Qwen35GdnLayerLayout::build().unwrap();
        let regions = layout.regions();

        assert_eq!(
            regions.history.len(),
            8 * Qwen35_9B::GDN_QKV_ROWS * (Qwen35_9B::LINEAR_CONV_KERNEL_DIM - 1)
        );
        assert_eq!(
            regions.state.len(),
            8 * Qwen35_9B::GDN_CONTROL_ROWS
                * Qwen35_9B::LINEAR_HEAD_DIM
                * Qwen35_9B::LINEAR_HEAD_DIM
        );
        assert_eq!(
            regions.input_weight_scales.len(),
            Qwen35_9B::GDN_INPUT_ROWS * Qwen35_9B::HIDDEN / 16
        );
        assert_eq!(
            regions.projected_controls.len(),
            8 * super::PADDED_CONTROL_ROWS
        );
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }

    use tuisko_gpu::ArenaRegion;
}

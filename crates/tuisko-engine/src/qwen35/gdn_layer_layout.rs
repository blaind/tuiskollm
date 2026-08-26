//! Single-allocation layout for one Qwen3.5 GDN layer.

use crate::common::math::{product, sum};
use crate::{EngineError, EngineResult, LayerMemoryLayout, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
const PADDED_CONTROL_ROWS: usize = 128;
pub(crate) const QWEN35_GDN_MAX_ROWS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35GdnLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) input_activation_codes: ArenaRegion<u8>,
    pub(crate) input_activation_scales: ArenaRegion<u8>,
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
    pub(crate) snapshot_history: ArenaRegion<u16>,
    pub(crate) log_decay: ArenaRegion<f32>,
    pub(crate) beta: ArenaRegion<f32>,
    pub(crate) convolved: ArenaRegion<u16>,
    pub(crate) recurrent_norm: ArenaRegion<u16>,
    pub(crate) state: ArenaRegion<f32>,
    pub(crate) snapshot_state: ArenaRegion<f32>,
    pub(crate) recurrent_plane: ArenaRegion<f32>,
    pub(crate) recurrent_output: ArenaRegion<u16>,
    pub(crate) output_activation_codes: ArenaRegion<u8>,
    pub(crate) output_activation_scales: ArenaRegion<u8>,
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
    pub(crate) down_activation_codes: ArenaRegion<u8>,
    pub(crate) down_activation_scales: ArenaRegion<u8>,
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
    /// Reserves every source plane and exact seam through `T=128`.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen35_9B;
        require_geometry::<A>()?;

        let row_hidden = product(
            "Qwen3.5 GDN row-hidden elements",
            QWEN35_GDN_MAX_ROWS,
            A::HIDDEN,
        )?;
        let row_input = product(
            "Qwen3.5 GDN projected elements",
            QWEN35_GDN_MAX_ROWS,
            A::GDN_INPUT_ROWS,
        )?;
        let row_qkv = product(
            "Qwen3.5 GDN convolved elements",
            QWEN35_GDN_MAX_ROWS,
            A::GDN_QKV_ROWS,
        )?;
        let row_value = product(
            "Qwen3.5 GDN recurrent output elements",
            QWEN35_GDN_MAX_ROWS,
            A::GDN_VALUE_ROWS,
        )?;
        let row_control = product(
            "Qwen3.5 GDN control elements",
            QWEN35_GDN_MAX_ROWS,
            A::GDN_CONTROL_ROWS,
        )?;
        let row_padded_control = product(
            "Qwen3.5 GDN padded control elements",
            QWEN35_GDN_MAX_ROWS,
            PADDED_CONTROL_ROWS,
        )?;
        let row_intermediate = product(
            "Qwen3.5 GDN MLP elements",
            QWEN35_GDN_MAX_ROWS,
            A::INTERMEDIATE,
        )?;
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
            residual_input: builder.reserve(row_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            input_activation_codes: builder.reserve(row_hidden / 2, ALIGNMENT)?,
            input_activation_scales: builder.reserve(row_hidden / NVFP4_GROUP, ALIGNMENT)?,
            input_weight_codes: builder.reserve(input_weight_codes, ALIGNMENT)?,
            input_weight_scales: builder.reserve(input_weight_scales, ALIGNMENT)?,
            control_weight_codes: builder.reserve(control_weight_codes, ALIGNMENT)?,
            control_weight_scales: builder.reserve(control_weight_scales, ALIGNMENT)?,
            projected: builder.reserve(row_input, ALIGNMENT)?,
            projected_controls: builder.reserve(row_padded_control, ALIGNMENT)?,
            a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            convolution_weights: builder.reserve(convolution_weights, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            history: builder.reserve(history, ALIGNMENT)?,
            snapshot_history: builder.reserve(history / MAX_BATCH, ALIGNMENT)?,
            log_decay: builder.reserve(row_control, ALIGNMENT)?,
            beta: builder.reserve(row_control, ALIGNMENT)?,
            convolved: builder.reserve(row_qkv, ALIGNMENT)?,
            recurrent_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
            snapshot_state: builder.reserve(state / MAX_BATCH, ALIGNMENT)?,
            recurrent_plane: builder.reserve(row_value, ALIGNMENT)?,
            recurrent_output: builder.reserve(row_value, ALIGNMENT)?,
            output_activation_codes: builder.reserve(row_hidden / 2, ALIGNMENT)?,
            output_activation_scales: builder.reserve(row_hidden / NVFP4_GROUP, ALIGNMENT)?,
            output_weight_codes: builder.reserve(output_weight_codes, ALIGNMENT)?,
            output_weight_scales: builder.reserve(output_weight_scales, ALIGNMENT)?,
            mixer_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(row_hidden, ALIGNMENT)?,
            mlp_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
            gate_up_activation_codes: builder.reserve(row_hidden / 2, ALIGNMENT)?,
            gate_up_activation_scales: builder.reserve(row_hidden / NVFP4_GROUP, ALIGNMENT)?,
            gate_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            up_weight_codes: builder.reserve(gate_codes, ALIGNMENT)?,
            gate_up_weight_scales: builder.reserve(gate_up_scales, ALIGNMENT)?,
            swiglu: builder.reserve(row_intermediate, ALIGNMENT)?,
            down_activation_codes: builder.reserve(row_intermediate / 2, ALIGNMENT)?,
            down_activation_scales: builder.reserve(row_intermediate / NVFP4_GROUP, ALIGNMENT)?,
            down_weight_codes: builder.reserve(down_weight_codes, ALIGNMENT)?,
            down_weight_scales: builder.reserve(down_weight_scales, ALIGNMENT)?,
            mlp_branch: builder.reserve(row_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(row_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(row_hidden, ALIGNMENT)?,
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
                regions.input_activation_codes.byte_len(),
                regions.input_activation_scales.byte_len(),
                regions.projected.byte_len(),
                regions.projected_controls.byte_len(),
                regions.state_rows.byte_len(),
                regions.history.byte_len(),
                regions.snapshot_history.byte_len(),
                regions.log_decay.byte_len(),
                regions.beta.byte_len(),
                regions.convolved.byte_len(),
                regions.state.byte_len(),
                regions.snapshot_state.byte_len(),
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

    /// Largest exact row route backed by the workspace.
    pub const fn row_capacity(&self) -> usize {
        QWEN35_GDN_MAX_ROWS
    }

    /// Resident weights plus workspace, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.workspace_bytes
    }
}

impl LayerMemoryLayout for Qwen35GdnLayerLayout {
    fn arena_bytes(&self) -> usize {
        self.arena_bytes()
    }

    fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes()
    }

    // This owner holds no paged key/value cache.
    fn cache_bytes(&self) -> usize {
        0
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes()
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

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, QWEN35_GDN_MAX_ROWS, Qwen35GdnLayerLayout};
    use tuisko_model::{Arch, Qwen35_9B};

    #[test]
    fn byte_accounting_is_exact() {
        let layout = Qwen35GdnLayerLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 123_068_800);
        assert_eq!(layout.workspace_bytes(), 41_074_720);
        assert_eq!(layout.owner_bytes(), 164_143_520);
        assert_eq!(layout.arena_bytes(), 164_144_128);
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
            span(regions.input_activation_codes),
            span(regions.input_activation_scales),
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
            span(regions.snapshot_history),
            span(regions.log_decay),
            span(regions.beta),
            span(regions.convolved),
            span(regions.recurrent_norm),
            span(regions.state),
            span(regions.snapshot_state),
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
            span(regions.gate_weight_codes),
            span(regions.up_weight_codes),
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
        assert_eq!(regions.snapshot_history.len(), regions.history.len() / 8);
        assert_eq!(regions.snapshot_state.len(), regions.state.len() / 8);
        assert_eq!(
            regions.input_weight_scales.len(),
            Qwen35_9B::GDN_INPUT_ROWS * Qwen35_9B::HIDDEN / 16
        );
        assert_eq!(
            regions.projected_controls.len(),
            QWEN35_GDN_MAX_ROWS * super::PADDED_CONTROL_ROWS
        );
        assert_eq!(layout.row_capacity(), 128);
    }

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }

    use tuisko_gpu::ArenaRegion;
}

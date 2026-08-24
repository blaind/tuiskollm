//! Single-allocation layout for one Qwen3.6 GDN plus MoE decoder layer.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen36Moe35B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;
const SLOTS_PER_TOKEN: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN + 1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36GdnMoeLayerRegions {
    pub(crate) residual_input: ArenaRegion<u16>,
    pub(crate) input_norm: ArenaRegion<u16>,
    pub(crate) mixer_normalized: ArenaRegion<u16>,
    pub(crate) input_activation_codes: ArenaRegion<u8>,
    pub(crate) input_weight_codes: ArenaRegion<u8>,
    pub(crate) control_weight_bf16: ArenaRegion<u16>,
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
    pub(crate) output_activation_codes: ArenaRegion<u8>,
    pub(crate) output_weight_codes: ArenaRegion<u8>,
    pub(crate) mixer_branch: ArenaRegion<u16>,
    pub(crate) post_attention_norm: ArenaRegion<u16>,
    pub(crate) mixer_residual: ArenaRegion<u16>,
    pub(crate) moe_normalized: ArenaRegion<u16>,
    pub(crate) router_weight: ArenaRegion<u16>,
    pub(crate) router_logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_gate_up_codes: ArenaRegion<u8>,
    pub(crate) routed_gate_up_scales: ArenaRegion<u8>,
    pub(crate) routed_gate_up_weight_scales_2: ArenaRegion<f32>,
    pub(crate) routed_down_codes: ArenaRegion<u8>,
    pub(crate) routed_down_scales: ArenaRegion<u8>,
    pub(crate) routed_down_weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate_up_codes: ArenaRegion<u8>,
    pub(crate) shared_gate_up_scales: ArenaRegion<u8>,
    pub(crate) shared_down_codes: ArenaRegion<u8>,
    pub(crate) shared_down_scales: ArenaRegion<u8>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) expert_intermediate: ArenaRegion<u16>,
    pub(crate) expert_output: ArenaRegion<u16>,
    pub(crate) shared_gate: ArenaRegion<u16>,
    pub(crate) moe_branch: ArenaRegion<u16>,
    pub(crate) next_norm: ArenaRegion<u16>,
    pub(crate) residual_output: ArenaRegion<u16>,
    pub(crate) next_normalized: ArenaRegion<u16>,
}

/// Checked weights, recurrent state, and workspace for one Qwen3.6 GDN/MoE layer.
#[derive(Clone, Debug)]
pub struct Qwen36GdnMoeLayerLayout {
    builder: ArenaLayout,
    regions: Qwen36GdnMoeLayerRegions,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen36GdnMoeLayerLayout {
    /// Reserves every source plane and exact decode seam for `B=1..=8`.
    pub fn build() -> EngineResult<Self> {
        type A = Qwen36Moe35B;
        require_geometry()?;

        let batch_hidden = product("Qwen3.6 layer batch-hidden", MAX_BATCH, A::HIDDEN)?;
        let batch_projected = product("Qwen3.6 layer projected", MAX_BATCH, A::GDN_INPUT_ROWS)?;
        let batch_qkv = product("Qwen3.6 layer convolved", MAX_BATCH, A::GDN_QKV_ROWS)?;
        let batch_value = product(
            "Qwen3.6 layer recurrent output",
            MAX_BATCH,
            A::GDN_VALUE_ROWS,
        )?;
        let batch_control = product("Qwen3.6 layer controls", MAX_BATCH, A::GDN_CONTROL_ROWS)?;
        let history = product(
            "Qwen3.6 layer history",
            product("Qwen3.6 layer history rows", MAX_BATCH, A::GDN_QKV_ROWS)?,
            A::LINEAR_CONV_KERNEL_DIM - 1,
        )?;
        let state = product(
            "Qwen3.6 layer state",
            product("Qwen3.6 layer state heads", MAX_BATCH, A::GDN_CONTROL_ROWS)?,
            product(
                "Qwen3.6 layer state matrix",
                A::LINEAR_HEAD_DIM,
                A::LINEAR_HEAD_DIM,
            )?,
        )?;
        let routed_gate_up_codes = product(
            "Qwen3.6 routed gate/up codes",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed gate/up rows",
                2 * A::INTERMEDIATE,
                A::HIDDEN / 2,
            )?,
        )?;
        let routed_gate_up_scales = product(
            "Qwen3.6 routed gate/up scales",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed gate/up scale rows",
                2 * A::INTERMEDIATE,
                A::HIDDEN / NVFP4_GROUP,
            )?,
        )?;
        let routed_down_codes = product(
            "Qwen3.6 routed down codes",
            A::NUM_EXPERTS,
            product("Qwen3.6 routed down rows", A::HIDDEN, A::INTERMEDIATE / 2)?,
        )?;
        let routed_down_scales = product(
            "Qwen3.6 routed down scales",
            A::NUM_EXPERTS,
            product(
                "Qwen3.6 routed down scale rows",
                A::HIDDEN,
                A::INTERMEDIATE / NVFP4_GROUP,
            )?,
        )?;
        let shared_gate_up_codes = product(
            "Qwen3.6 shared gate/up codes",
            2 * A::SHARED_EXPERT_INTERMEDIATE,
            A::HIDDEN / 2,
        )?;
        let shared_gate_up_scales = product(
            "Qwen3.6 shared gate/up scales",
            2 * A::SHARED_EXPERT_INTERMEDIATE,
            A::HIDDEN / NVFP4_GROUP,
        )?;
        let shared_down_codes = product(
            "Qwen3.6 shared down codes",
            A::HIDDEN,
            A::SHARED_EXPERT_INTERMEDIATE / 2,
        )?;
        let shared_down_scales = product(
            "Qwen3.6 shared down scales",
            A::HIDDEN,
            A::SHARED_EXPERT_INTERMEDIATE / NVFP4_GROUP,
        )?;
        let expert_intermediate = product(
            "Qwen3.6 expert intermediate",
            product("Qwen3.6 expert slots", MAX_BATCH, SLOTS_PER_TOKEN)?,
            A::INTERMEDIATE,
        )?;
        let expert_output = product(
            "Qwen3.6 expert output",
            product("Qwen3.6 expert output slots", MAX_BATCH, SLOTS_PER_TOKEN)?,
            A::HIDDEN,
        )?;

        let mut builder = ArenaLayout::new();
        let regions = Qwen36GdnMoeLayerRegions {
            residual_input: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_activation_codes: builder.reserve(batch_hidden, ALIGNMENT)?,
            input_weight_codes: builder.reserve(A::GDN_INPUT_ROWS * A::HIDDEN, ALIGNMENT)?,
            control_weight_bf16: builder.reserve(2 * A::GDN_CONTROL_ROWS * A::HIDDEN, ALIGNMENT)?,
            projected: builder.reserve(batch_projected, ALIGNMENT)?,
            projected_controls: builder.reserve(2 * batch_control, ALIGNMENT)?,
            a_log: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            dt_bias: builder.reserve(A::GDN_CONTROL_ROWS, ALIGNMENT)?,
            convolution_weights: builder
                .reserve(A::GDN_QKV_ROWS * A::LINEAR_CONV_KERNEL_DIM, ALIGNMENT)?,
            state_rows: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            history: builder.reserve(history, ALIGNMENT)?,
            log_decay: builder.reserve(batch_control, ALIGNMENT)?,
            beta: builder.reserve(batch_control, ALIGNMENT)?,
            convolved: builder.reserve(batch_qkv, ALIGNMENT)?,
            recurrent_norm: builder.reserve(A::LINEAR_HEAD_DIM, ALIGNMENT)?,
            state: builder.reserve(state, ALIGNMENT)?,
            recurrent_output: builder.reserve(batch_value, ALIGNMENT)?,
            output_activation_codes: builder.reserve(batch_value, ALIGNMENT)?,
            output_weight_codes: builder.reserve(A::HIDDEN * A::GDN_VALUE_ROWS, ALIGNMENT)?,
            mixer_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            post_attention_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            mixer_residual: builder.reserve(batch_hidden, ALIGNMENT)?,
            moe_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
            router_weight: builder.reserve(A::NUM_EXPERTS * A::HIDDEN, ALIGNMENT)?,
            router_logits: builder.reserve(MAX_BATCH * A::NUM_EXPERTS, ALIGNMENT)?,
            expert_indices: builder.reserve(MAX_BATCH * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routing_weights: builder.reserve(MAX_BATCH * A::NUM_EXPERTS_PER_TOKEN, ALIGNMENT)?,
            routed_gate_up_codes: builder.reserve(routed_gate_up_codes, ALIGNMENT)?,
            routed_gate_up_scales: builder.reserve(routed_gate_up_scales, ALIGNMENT)?,
            routed_gate_up_weight_scales_2: builder.reserve(A::NUM_EXPERTS, ALIGNMENT)?,
            routed_down_codes: builder.reserve(routed_down_codes, ALIGNMENT)?,
            routed_down_scales: builder.reserve(routed_down_scales, ALIGNMENT)?,
            routed_down_weight_scales_2: builder.reserve(A::NUM_EXPERTS, ALIGNMENT)?,
            shared_gate_up_codes: builder.reserve(shared_gate_up_codes, ALIGNMENT)?,
            shared_gate_up_scales: builder.reserve(shared_gate_up_scales, ALIGNMENT)?,
            shared_down_codes: builder.reserve(shared_down_codes, ALIGNMENT)?,
            shared_down_scales: builder.reserve(shared_down_scales, ALIGNMENT)?,
            shared_gate_weight: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            expert_intermediate: builder.reserve(expert_intermediate, ALIGNMENT)?,
            expert_output: builder.reserve(expert_output, ALIGNMENT)?,
            shared_gate: builder.reserve(MAX_BATCH, ALIGNMENT)?,
            moe_branch: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_norm: builder.reserve(A::HIDDEN, ALIGNMENT)?,
            residual_output: builder.reserve(batch_hidden, ALIGNMENT)?,
            next_normalized: builder.reserve(batch_hidden, ALIGNMENT)?,
        };
        let resident_weight_bytes = sum(
            "Qwen3.6 layer resident weights",
            &[
                regions.input_norm.byte_len(),
                regions.input_weight_codes.byte_len(),
                regions.control_weight_bf16.byte_len(),
                regions.a_log.byte_len(),
                regions.dt_bias.byte_len(),
                regions.convolution_weights.byte_len(),
                regions.recurrent_norm.byte_len(),
                regions.output_weight_codes.byte_len(),
                regions.post_attention_norm.byte_len(),
                regions.router_weight.byte_len(),
                regions.routed_gate_up_codes.byte_len(),
                regions.routed_gate_up_scales.byte_len(),
                regions.routed_gate_up_weight_scales_2.byte_len(),
                regions.routed_down_codes.byte_len(),
                regions.routed_down_scales.byte_len(),
                regions.routed_down_weight_scales_2.byte_len(),
                regions.shared_gate_up_codes.byte_len(),
                regions.shared_gate_up_scales.byte_len(),
                regions.shared_down_codes.byte_len(),
                regions.shared_down_scales.byte_len(),
                regions.shared_gate_weight.byte_len(),
                regions.next_norm.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "Qwen3.6 layer workspace",
            &[
                regions.residual_input.byte_len(),
                regions.mixer_normalized.byte_len(),
                regions.input_activation_codes.byte_len(),
                regions.projected.byte_len(),
                regions.projected_controls.byte_len(),
                regions.state_rows.byte_len(),
                regions.history.byte_len(),
                regions.log_decay.byte_len(),
                regions.beta.byte_len(),
                regions.convolved.byte_len(),
                regions.state.byte_len(),
                regions.recurrent_output.byte_len(),
                regions.output_activation_codes.byte_len(),
                regions.mixer_branch.byte_len(),
                regions.mixer_residual.byte_len(),
                regions.moe_normalized.byte_len(),
                regions.router_logits.byte_len(),
                regions.expert_indices.byte_len(),
                regions.routing_weights.byte_len(),
                regions.expert_intermediate.byte_len(),
                regions.expert_output.byte_len(),
                regions.shared_gate.byte_len(),
                regions.moe_branch.byte_len(),
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

    pub(crate) const fn regions(&self) -> Qwen36GdnMoeLayerRegions {
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

fn require_geometry() -> EngineResult<()> {
    type A = Qwen36Moe35B;
    if !A::HIDDEN.is_multiple_of(NVFP4_GROUP)
        || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP)
        || A::SHARED_EXPERT_INTERMEDIATE != A::INTERMEDIATE
        || A::NUM_EXPERTS_PER_TOKEN != 8
        || A::GDN_INPUT_ROWS != A::GDN_QKV_ROWS + A::GDN_VALUE_ROWS
        || A::GDN_CONTROL_ROWS != 32
        || A::LINEAR_CONV_KERNEL_DIM != 4
    {
        return Err(EngineError::layout(
            "Qwen3.6 GDN/MoE geometry differs from the qualified layer contract",
        ));
    }

    Ok(())
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
    use super::*;
    use tuisko_gpu::ArenaRegion;

    #[test]
    fn byte_accounting_and_geometry_are_exact() {
        let layout = Qwen36GdnMoeLayerLayout::build().unwrap();

        assert_eq!(Qwen36Moe35B::GDN_INPUT_ROWS, 12_288);
        assert_eq!(Qwen36Moe35B::NUM_EXPERTS, 256);
        assert_eq!(layout.resident_weight_bytes(), 489_703_808);
        assert_eq!(layout.workspace_bytes(), 18_251_056);
        assert_eq!(layout.owner_bytes(), 507_954_864);
        assert_eq!(layout.arena_bytes(), 507_955_968);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 1_104);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Qwen36GdnMoeLayerLayout::build().unwrap();
        let regions = layout.regions();
        let mut spans = vec![
            span(regions.residual_input),
            span(regions.input_norm),
            span(regions.mixer_normalized),
            span(regions.input_activation_codes),
            span(regions.input_weight_codes),
            span(regions.control_weight_bf16),
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
            span(regions.output_activation_codes),
            span(regions.output_weight_codes),
            span(regions.mixer_branch),
            span(regions.post_attention_norm),
            span(regions.mixer_residual),
            span(regions.moe_normalized),
            span(regions.router_weight),
            span(regions.router_logits),
            span(regions.expert_indices),
            span(regions.routing_weights),
            span(regions.routed_gate_up_codes),
            span(regions.routed_gate_up_scales),
            span(regions.routed_gate_up_weight_scales_2),
            span(regions.routed_down_codes),
            span(regions.routed_down_scales),
            span(regions.routed_down_weight_scales_2),
            span(regions.shared_gate_up_codes),
            span(regions.shared_gate_up_scales),
            span(regions.shared_down_codes),
            span(regions.shared_down_scales),
            span(regions.shared_gate_weight),
            span(regions.expert_intermediate),
            span(regions.expert_output),
            span(regions.shared_gate),
            span(regions.moe_branch),
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

    fn span<T: Copy>(region: ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }
}

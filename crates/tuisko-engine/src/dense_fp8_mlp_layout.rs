//! Single-allocation layout for one resident dense-FP8 MLP boundary.

use crate::{EngineError, EngineResult};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::Arch;

const ALIGNMENT: usize = 256;
pub(crate) const MAX_ROWS: usize = 1_024;

/// Checked source-weight and workspace regions for one late-layer MLP.
#[derive(Clone, Debug)]
pub struct DenseFp8MlpLayout {
    builder: ArenaLayout,
    residual_input: ArenaRegion<u16>,
    input_norm: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    gate_up_activation_codes: ArenaRegion<u8>,
    gate_up_activation_scales: ArenaRegion<f32>,
    gate_up_weight_codes: ArenaRegion<u8>,
    gate_up_weight_scales: ArenaRegion<u16>,
    swiglu: ArenaRegion<u16>,
    down_activation_codes: ArenaRegion<u8>,
    down_activation_scales: ArenaRegion<f32>,
    down_weight_codes: ArenaRegion<u8>,
    down_weight_scales: ArenaRegion<u16>,
    branch: ArenaRegion<u16>,
    next_norm: ArenaRegion<u16>,
    residual_output: ArenaRegion<u16>,
    next_normalized: ArenaRegion<u16>,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl DenseFp8MlpLayout {
    /// Reserves every plane for decode and exact prefill routes through T=1024.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        let batch_hidden = product("dense-FP8 MLP row-hidden elements", MAX_ROWS, A::HIDDEN)?;
        let batch_intermediate = product(
            "dense-FP8 MLP row-intermediate elements",
            MAX_ROWS,
            A::INTERMEDIATE,
        )?;
        let gate_up_weights = product(
            "dense-FP8 gate/up weight elements",
            product("dense-FP8 gate/up rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN,
        )?;
        let down_weights = product("dense-FP8 down weight elements", A::HIDDEN, A::INTERMEDIATE)?;
        let mut builder = ArenaLayout::new();
        let residual_input = builder.reserve(batch_hidden, ALIGNMENT)?;
        let input_norm = builder.reserve(A::HIDDEN, ALIGNMENT)?;
        let normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let gate_up_activation_codes = builder.reserve(batch_hidden, ALIGNMENT)?;
        let gate_up_activation_scales = builder.reserve(MAX_ROWS, ALIGNMENT)?;
        let gate_up_weight_codes = builder.reserve(gate_up_weights, ALIGNMENT)?;
        let gate_up_weight_scales = builder.reserve(2 * A::INTERMEDIATE, ALIGNMENT)?;
        let swiglu = builder.reserve(batch_intermediate, ALIGNMENT)?;
        let down_activation_codes = builder.reserve(batch_intermediate, ALIGNMENT)?;
        let down_activation_scales = builder.reserve(MAX_ROWS, ALIGNMENT)?;
        let down_weight_codes = builder.reserve(down_weights, ALIGNMENT)?;
        let down_weight_scales = builder.reserve(A::HIDDEN, ALIGNMENT)?;
        let branch = builder.reserve(batch_hidden, ALIGNMENT)?;
        let next_norm = builder.reserve(A::HIDDEN, ALIGNMENT)?;
        let residual_output = builder.reserve(batch_hidden, ALIGNMENT)?;
        let next_normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let resident_weight_bytes = sum(
            "dense-FP8 MLP resident weight bytes",
            &[
                input_norm.byte_len(),
                gate_up_weight_codes.byte_len(),
                gate_up_weight_scales.byte_len(),
                down_weight_codes.byte_len(),
                down_weight_scales.byte_len(),
                next_norm.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "dense-FP8 MLP workspace bytes",
            &[
                residual_input.byte_len(),
                normalized.byte_len(),
                gate_up_activation_codes.byte_len(),
                gate_up_activation_scales.byte_len(),
                swiglu.byte_len(),
                down_activation_codes.byte_len(),
                down_activation_scales.byte_len(),
                branch.byte_len(),
                residual_output.byte_len(),
                next_normalized.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            residual_input,
            input_norm,
            normalized,
            gate_up_activation_codes,
            gate_up_activation_scales,
            gate_up_weight_codes,
            gate_up_weight_scales,
            swiglu,
            down_activation_codes,
            down_activation_scales,
            down_weight_codes,
            down_weight_scales,
            branch,
            next_norm,
            residual_output,
            next_normalized,
            resident_weight_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    /// Complete allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Exact source-backed norm, gate/up, and down weight bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Exact address-stable working-plane bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights plus workspace, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.workspace_bytes
    }

    pub(crate) const fn residual_input(&self) -> ArenaRegion<u16> {
        self.residual_input
    }

    pub(crate) const fn input_norm(&self) -> ArenaRegion<u16> {
        self.input_norm
    }

    pub(crate) const fn normalized(&self) -> ArenaRegion<u16> {
        self.normalized
    }

    pub(crate) const fn gate_up_activation_codes(&self) -> ArenaRegion<u8> {
        self.gate_up_activation_codes
    }

    pub(crate) const fn gate_up_activation_scales(&self) -> ArenaRegion<f32> {
        self.gate_up_activation_scales
    }

    pub(crate) const fn gate_up_weight_codes(&self) -> ArenaRegion<u8> {
        self.gate_up_weight_codes
    }

    pub(crate) const fn gate_up_weight_scales(&self) -> ArenaRegion<u16> {
        self.gate_up_weight_scales
    }

    pub(crate) const fn swiglu(&self) -> ArenaRegion<u16> {
        self.swiglu
    }

    pub(crate) const fn down_activation_codes(&self) -> ArenaRegion<u8> {
        self.down_activation_codes
    }

    pub(crate) const fn down_activation_scales(&self) -> ArenaRegion<f32> {
        self.down_activation_scales
    }

    pub(crate) const fn down_weight_codes(&self) -> ArenaRegion<u8> {
        self.down_weight_codes
    }

    pub(crate) const fn down_weight_scales(&self) -> ArenaRegion<u16> {
        self.down_weight_scales
    }

    pub(crate) const fn branch(&self) -> ArenaRegion<u16> {
        self.branch
    }

    pub(crate) const fn next_norm(&self) -> ArenaRegion<u16> {
        self.next_norm
    }

    pub(crate) const fn residual_output(&self) -> ArenaRegion<u16> {
        self.residual_output
    }

    pub(crate) const fn next_normalized(&self) -> ArenaRegion<u16> {
        self.next_normalized
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
    use super::{ALIGNMENT, DenseFp8MlpLayout};
    use tuisko_model::{Arch, Qwen38_27B};

    fn spans(layout: &DenseFp8MlpLayout) -> Vec<(usize, usize)> {
        vec![
            (
                layout.residual_input().offset_bytes(),
                layout.residual_input().byte_len(),
            ),
            (
                layout.input_norm().offset_bytes(),
                layout.input_norm().byte_len(),
            ),
            (
                layout.normalized().offset_bytes(),
                layout.normalized().byte_len(),
            ),
            (
                layout.gate_up_activation_codes().offset_bytes(),
                layout.gate_up_activation_codes().byte_len(),
            ),
            (
                layout.gate_up_activation_scales().offset_bytes(),
                layout.gate_up_activation_scales().byte_len(),
            ),
            (
                layout.gate_up_weight_codes().offset_bytes(),
                layout.gate_up_weight_codes().byte_len(),
            ),
            (
                layout.gate_up_weight_scales().offset_bytes(),
                layout.gate_up_weight_scales().byte_len(),
            ),
            (layout.swiglu().offset_bytes(), layout.swiglu().byte_len()),
            (
                layout.down_activation_codes().offset_bytes(),
                layout.down_activation_codes().byte_len(),
            ),
            (
                layout.down_activation_scales().offset_bytes(),
                layout.down_activation_scales().byte_len(),
            ),
            (
                layout.down_weight_codes().offset_bytes(),
                layout.down_weight_codes().byte_len(),
            ),
            (
                layout.down_weight_scales().offset_bytes(),
                layout.down_weight_scales().byte_len(),
            ),
            (layout.branch().offset_bytes(), layout.branch().byte_len()),
            (
                layout.next_norm().offset_bytes(),
                layout.next_norm().byte_len(),
            ),
            (
                layout.residual_output().offset_bytes(),
                layout.residual_output().byte_len(),
            ),
            (
                layout.next_normalized().offset_bytes(),
                layout.next_normalized().byte_len(),
            ),
        ]
    }

    #[test]
    fn qwen_dense_fp8_mlp_byte_accounting_is_exact() {
        let layout = DenseFp8MlpLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 267_487_232);
        assert_eq!(layout.workspace_bytes(), 111_157_248);
        assert_eq!(layout.owner_bytes(), 378_644_480);
        assert_eq!(layout.arena_bytes(), 378_644_480);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 0);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = DenseFp8MlpLayout::build::<Qwen38_27B>().unwrap();
        let mut regions = spans(&layout);
        regions.sort_unstable_by_key(|(offset, _)| *offset);

        for &(offset, bytes) in &regions {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in regions.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    #[test]
    fn working_planes_follow_architecture_geometry() {
        let layout = DenseFp8MlpLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(
            layout.gate_up_weight_codes().byte_len(),
            2 * Qwen38_27B::INTERMEDIATE * Qwen38_27B::HIDDEN
        );
        assert_eq!(
            layout.down_weight_codes().byte_len(),
            Qwen38_27B::HIDDEN * Qwen38_27B::INTERMEDIATE
        );
    }
}

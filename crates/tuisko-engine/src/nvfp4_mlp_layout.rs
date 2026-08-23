//! Single-allocation layout for one resident NVFP4 MLP boundary.

use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::Arch;

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;

/// Checked source-weight and workspace regions for one early-layer MLP.
#[derive(Clone, Debug)]
pub struct Nvfp4MlpLayout {
    builder: ArenaLayout,
    residual_input: ArenaRegion<u16>,
    input_norm: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    gate_up_activation_codes: ArenaRegion<u8>,
    gate_up_activation_scales: ArenaRegion<u8>,
    gate_weight_codes: ArenaRegion<u8>,
    up_weight_codes: ArenaRegion<u8>,
    gate_up_weight_scales: ArenaRegion<u8>,
    swiglu: ArenaRegion<u16>,
    down_weight_codes: ArenaRegion<u8>,
    down_weight_scales: ArenaRegion<u8>,
    branch: ArenaRegion<u16>,
    next_norm: ArenaRegion<u16>,
    residual_output: ArenaRegion<u16>,
    next_normalized: ArenaRegion<u16>,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl Nvfp4MlpLayout {
    /// Reserves every plane for the architecture's exact `B=1..=8` routes.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        require_geometry::<A>()?;
        let batch_hidden = product("NVFP4 MLP batch-hidden elements", MAX_BATCH, A::HIDDEN)?;
        let batch_intermediate = product(
            "NVFP4 MLP batch-intermediate elements",
            MAX_BATCH,
            A::INTERMEDIATE,
        )?;
        let branch_codes = product(
            "NVFP4 gate/up branch code bytes",
            A::INTERMEDIATE,
            A::HIDDEN / 2,
        )?;
        let gate_up_scales = product(
            "NVFP4 gate/up scale bytes",
            product("NVFP4 gate/up scale rows", 2, A::INTERMEDIATE)?,
            A::HIDDEN / NVFP4_GROUP,
        )?;
        let down_codes = product("NVFP4 down code values", A::HIDDEN, A::INTERMEDIATE / 2)?;
        let down_scales = product(
            "NVFP4 down scale bytes",
            A::HIDDEN,
            A::INTERMEDIATE / NVFP4_GROUP,
        )?;
        let mut builder = ArenaLayout::new();
        let residual_input = builder.reserve(batch_hidden, ALIGNMENT)?;
        let input_norm = builder.reserve(A::HIDDEN, ALIGNMENT)?;
        let normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let gate_up_activation_codes = builder.reserve(batch_hidden / 2, ALIGNMENT)?;
        let gate_up_activation_scales = builder.reserve(batch_hidden / NVFP4_GROUP, ALIGNMENT)?;
        let gate_weight_codes = builder.reserve(branch_codes, ALIGNMENT)?;
        let up_weight_codes = builder.reserve(branch_codes, ALIGNMENT)?;
        let gate_up_weight_scales = builder.reserve(gate_up_scales, ALIGNMENT)?;
        let swiglu = builder.reserve(batch_intermediate, ALIGNMENT)?;
        let down_weight_codes = builder.reserve(down_codes, ALIGNMENT)?;
        let down_weight_scales = builder.reserve(down_scales, ALIGNMENT)?;
        let branch = builder.reserve(batch_hidden, ALIGNMENT)?;
        let next_norm = builder.reserve(A::HIDDEN, ALIGNMENT)?;
        let residual_output = builder.reserve(batch_hidden, ALIGNMENT)?;
        let next_normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let resident_weight_bytes = sum(
            "NVFP4 MLP resident weight bytes",
            &[
                input_norm.byte_len(),
                gate_weight_codes.byte_len(),
                up_weight_codes.byte_len(),
                gate_up_weight_scales.byte_len(),
                down_weight_codes.byte_len(),
                down_weight_scales.byte_len(),
                next_norm.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "NVFP4 MLP workspace bytes",
            &[
                residual_input.byte_len(),
                normalized.byte_len(),
                gate_up_activation_codes.byte_len(),
                gate_up_activation_scales.byte_len(),
                swiglu.byte_len(),
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
            gate_weight_codes,
            up_weight_codes,
            gate_up_weight_scales,
            swiglu,
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

    pub(crate) const fn gate_up_activation_scales(&self) -> ArenaRegion<u8> {
        self.gate_up_activation_scales
    }

    pub(crate) const fn gate_weight_codes(&self) -> ArenaRegion<u8> {
        self.gate_weight_codes
    }

    pub(crate) const fn up_weight_codes(&self) -> ArenaRegion<u8> {
        self.up_weight_codes
    }

    pub(crate) const fn gate_up_weight_scales(&self) -> ArenaRegion<u8> {
        self.gate_up_weight_scales
    }

    pub(crate) const fn swiglu(&self) -> ArenaRegion<u16> {
        self.swiglu
    }

    pub(crate) const fn down_weight_codes(&self) -> ArenaRegion<u8> {
        self.down_weight_codes
    }

    pub(crate) const fn down_weight_scales(&self) -> ArenaRegion<u8> {
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

fn require_geometry<A: Arch>() -> EngineResult<()> {
    if !A::HIDDEN.is_multiple_of(NVFP4_GROUP) || !A::INTERMEDIATE.is_multiple_of(NVFP4_GROUP) {
        return Err(EngineError::layout(
            "NVFP4 MLP geometry must be divisible by the K16 source group",
        ));
    }

    Ok(())
}

fn product(name: &str, left: usize, right: usize) -> EngineResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
}

fn sum(name: &str, values: &[usize]) -> EngineResult<usize> {
    values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| EngineError::layout(format!("{name} overflows")))
    })
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, Nvfp4MlpLayout};
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    fn spans(layout: &Nvfp4MlpLayout) -> Vec<(usize, usize)> {
        vec![
            span(layout.residual_input()),
            span(layout.input_norm()),
            span(layout.normalized()),
            span(layout.gate_up_activation_codes()),
            span(layout.gate_up_activation_scales()),
            span(layout.gate_weight_codes()),
            span(layout.up_weight_codes()),
            span(layout.gate_up_weight_scales()),
            span(layout.swiglu()),
            span(layout.down_weight_codes()),
            span(layout.down_weight_scales()),
            span(layout.branch()),
            span(layout.next_norm()),
            span(layout.residual_output()),
            span(layout.next_normalized()),
        ]
    }

    fn span<T: Copy>(region: tuisko_gpu::ArenaRegion<T>) -> (usize, usize) {
        (region.offset_bytes(), region.byte_len())
    }

    #[test]
    fn qwen_nvfp4_mlp_byte_accounting_is_exact() {
        let layout = Nvfp4MlpLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 150_425_600);
        assert_eq!(layout.workspace_bytes(), 711_168);
        assert_eq!(layout.owner_bytes(), 151_136_768);
        assert_eq!(layout.arena_bytes(), 151_136_768);
    }

    #[test]
    fn qwen35_nvfp4_mlp_byte_accounting_is_exact() {
        let layout = Nvfp4MlpLayout::build::<Qwen35_9B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 84_951_040);
        assert_eq!(layout.workspace_bytes(), 542_720);
        assert_eq!(layout.owner_bytes(), 85_493_760);
        assert_eq!(layout.arena_bytes(), 85_493_760);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = Nvfp4MlpLayout::build::<Qwen38_27B>().unwrap();
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
    fn packed_planes_follow_exact_source_geometry() {
        let layout = Nvfp4MlpLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(
            layout.gate_weight_codes().byte_len(),
            Qwen38_27B::INTERMEDIATE * Qwen38_27B::HIDDEN / 2,
        );
        assert_eq!(
            layout.up_weight_codes().offset_bytes(),
            layout.gate_weight_codes().offset_bytes() + layout.gate_weight_codes().byte_len(),
        );
        assert_eq!(
            layout.down_weight_codes().byte_len(),
            Qwen38_27B::HIDDEN * Qwen38_27B::INTERMEDIATE / 2,
        );
    }
}

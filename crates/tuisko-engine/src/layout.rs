//! Address-stable text-endpoint layout.

use crate::EngineResult;
use crate::common::math::{product, sum};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::Arch;

/// Byte accounting shared by every resident layout.
///
/// Inspection only: it reports totals for auditing and metrics and never constructs a layout,
/// selects a route, or exposes device state.
pub trait LayerMemoryLayout {
    /// Complete allocation size in bytes including alignment padding.
    fn arena_bytes(&self) -> usize;

    /// Source-backed resident device weight bytes.
    fn resident_weight_bytes(&self) -> usize;

    /// Quantized key/value cache bytes.
    fn cache_bytes(&self) -> usize;

    /// Address-stable activation and transient workspace bytes.
    fn workspace_bytes(&self) -> usize;
}

/// Largest admitted compact decode batch.
pub const MAX_BATCH: usize = 8;

// All current kernels use at most four-byte vector loads. A 256-byte plane
// boundary also preserves alignment when later schedules widen transactions.
const ALIGNMENT: usize = 256;

/// Checked regions owned by the resident text endpoint.
#[derive(Clone, Debug)]
pub struct EndpointLayout {
    builder: ArenaLayout,
    input: ArenaRegion<u16>,
    final_norm_weight: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    logits: ArenaRegion<u16>,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl EndpointLayout {
    /// Reserves every endpoint plane for architecture `A`.
    pub fn build<A: Arch>() -> EngineResult<Self> {
        let batch_hidden = product("batch-hidden element count", MAX_BATCH, A::HIDDEN)?;
        let weight_codes_len = product("LM-head element count", A::VOCAB, A::HIDDEN)?;
        let batch_logits = product("batch-logit element count", MAX_BATCH, A::VOCAB)?;
        let mut builder = ArenaLayout::new();
        let input = builder.reserve::<u16>(batch_hidden, ALIGNMENT)?;
        let final_norm_weight = builder.reserve::<u16>(A::HIDDEN, ALIGNMENT)?;
        let normalized = builder.reserve::<u16>(batch_hidden, ALIGNMENT)?;
        let activation_codes = builder.reserve::<u8>(batch_hidden, ALIGNMENT)?;
        let activation_scales = builder.reserve::<f32>(MAX_BATCH, ALIGNMENT)?;
        let weight_codes = builder.reserve::<u8>(weight_codes_len, ALIGNMENT)?;
        let weight_scales = builder.reserve::<u16>(A::VOCAB, ALIGNMENT)?;
        let logits = builder.reserve::<u16>(batch_logits, ALIGNMENT)?;
        let resident_weight_bytes = sum(
            "resident endpoint weight bytes",
            &[
                final_norm_weight.byte_len(),
                weight_codes.byte_len(),
                weight_scales.byte_len(),
            ],
        )?;
        let workspace_bytes = sum(
            "endpoint workspace bytes",
            &[
                input.byte_len(),
                normalized.byte_len(),
                activation_codes.byte_len(),
                activation_scales.byte_len(),
                logits.byte_len(),
            ],
        )?;

        Ok(Self {
            builder,
            input,
            final_norm_weight,
            normalized,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            logits,
            resident_weight_bytes,
            workspace_bytes,
        })
    }

    pub(crate) const fn builder(&self) -> &ArenaLayout {
        &self.builder
    }

    /// Allocation bytes, including alignment padding.
    pub const fn arena_bytes(&self) -> usize {
        self.builder.byte_len()
    }

    /// Source-backed final-norm and LM-head bytes resident on the device.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Address-stable input, intermediate, and logit bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights plus workspace, excluding alignment padding.
    pub const fn owner_bytes(&self) -> usize {
        self.resident_weight_bytes + self.workspace_bytes
    }

    pub(crate) const fn input(&self) -> ArenaRegion<u16> {
        self.input
    }

    pub(crate) const fn final_norm_weight(&self) -> ArenaRegion<u16> {
        self.final_norm_weight
    }

    pub(crate) const fn normalized(&self) -> ArenaRegion<u16> {
        self.normalized
    }

    pub(crate) const fn activation_codes(&self) -> ArenaRegion<u8> {
        self.activation_codes
    }

    pub(crate) const fn activation_scales(&self) -> ArenaRegion<f32> {
        self.activation_scales
    }

    pub(crate) const fn weight_codes(&self) -> ArenaRegion<u8> {
        self.weight_codes
    }

    pub(crate) const fn weight_scales(&self) -> ArenaRegion<u16> {
        self.weight_scales
    }

    pub(crate) const fn logits(&self) -> ArenaRegion<u16> {
        self.logits
    }
}

impl LayerMemoryLayout for EndpointLayout {
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

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, EndpointLayout, MAX_BATCH};
    use tuisko_model::{Arch, Qwen38_27B};

    #[derive(Clone, Copy, Debug)]
    struct TestArch;

    impl Arch for TestArch {
        const MODEL_ID: &'static str = "test/arch";
        const REVISION: &'static str = "0";
        const HIDDEN: usize = 128;
        const RMS_NORM_EPSILON: f32 = 1.0e-6;
        const INTERMEDIATE: usize = 256;
        const VOCAB: usize = 512;
        const LAYERS: usize = 4;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        const NUM_ATTENTION_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const LINEAR_KEY_HEADS: usize = 2;
        const LINEAR_VALUE_HEADS: usize = 4;
        const LINEAR_HEAD_DIM: usize = 32;
        const LINEAR_CONV_KERNEL_DIM: usize = 4;
        const MTP_LAYERS: usize = 1;
        const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
        const VISION_DEPTH: usize = 2;
        const VISION_HIDDEN: usize = 64;
        const VISION_INTERMEDIATE: usize = 128;
        const VISION_NUM_HEADS: usize = 4;
        const VISION_POSITIONS: usize = 16;
        const VISION_OUTPUT_HIDDEN: usize = 128;
        const VISION_INPUT_CHANNELS: usize = 3;
        const VISION_PATCH_SIZE: usize = 8;
        const VISION_SPATIAL_MERGE_SIZE: usize = 2;
        const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
    }

    fn spans(layout: &EndpointLayout) -> [(usize, usize); 8] {
        [
            (layout.input().offset_bytes(), layout.input().byte_len()),
            (
                layout.final_norm_weight().offset_bytes(),
                layout.final_norm_weight().byte_len(),
            ),
            (
                layout.normalized().offset_bytes(),
                layout.normalized().byte_len(),
            ),
            (
                layout.activation_codes().offset_bytes(),
                layout.activation_codes().byte_len(),
            ),
            (
                layout.activation_scales().offset_bytes(),
                layout.activation_scales().byte_len(),
            ),
            (
                layout.weight_codes().offset_bytes(),
                layout.weight_codes().byte_len(),
            ),
            (
                layout.weight_scales().offset_bytes(),
                layout.weight_scales().byte_len(),
            ),
            (layout.logits().offset_bytes(), layout.logits().byte_len()),
        ]
    }

    #[test]
    fn qwen_endpoint_byte_accounting_is_exact() {
        let layout = EndpointLayout::build::<Qwen38_27B>().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 1_271_905_280);
        assert_eq!(layout.workspace_bytes(), 4_177_952);
        assert_eq!(layout.owner_bytes(), 1_276_083_232);
        assert_eq!(layout.arena_bytes(), 1_276_083_456);
        assert_eq!(layout.arena_bytes() - layout.owner_bytes(), 224);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_inside_the_arena() {
        let layout = EndpointLayout::build::<Qwen38_27B>().unwrap();
        let regions = spans(&layout);
        let mut ordered = regions;
        ordered.sort_unstable_by_key(|(offset, _)| *offset);

        for (offset, bytes) in regions {
            assert_eq!(offset % ALIGNMENT, 0);
            assert!(offset + bytes <= layout.arena_bytes());
        }
        for adjacent in ordered.windows(2) {
            assert!(adjacent[0].0 + adjacent[0].1 <= adjacent[1].0);
        }
    }

    #[test]
    fn geometry_flows_from_the_architecture() {
        let layout = EndpointLayout::build::<TestArch>().unwrap();

        assert_eq!(layout.input().byte_len(), MAX_BATCH * 128 * 2);
        assert_eq!(layout.final_norm_weight().byte_len(), 128 * 2);
        assert_eq!(layout.activation_codes().byte_len(), MAX_BATCH * 128);
        assert_eq!(layout.weight_codes().byte_len(), 512 * 128);
        assert_eq!(layout.weight_scales().byte_len(), 512 * 2);
        assert_eq!(layout.logits().byte_len(), MAX_BATCH * 512 * 2);
    }
}

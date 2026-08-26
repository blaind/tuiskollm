//! Address-stable Qwen3.5 text-endpoint layout.

use crate::common::math::product;
use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen35_9B};

const ALIGNMENT: usize = 256;

/// Checked regions owned by the resident Qwen3.5 text endpoint.
#[derive(Clone, Debug)]
pub struct Qwen35TextEndpointLayout {
    builder: ArenaLayout,
    input: ArenaRegion<u16>,
    final_norm_weight: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    lm_head_weight: ArenaRegion<u16>,
    logits: ArenaRegion<u16>,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen35TextEndpointLayout {
    /// Reserves every Qwen3.5 endpoint plane in one allocation.
    pub fn build() -> EngineResult<Self> {
        let batch_hidden = product(
            "Qwen3.5 batch-hidden elements",
            MAX_BATCH,
            Qwen35_9B::HIDDEN,
        )?;
        let lm_head = product(
            "Qwen3.5 LM-head elements",
            Qwen35_9B::VOCAB,
            Qwen35_9B::HIDDEN,
        )?;
        let batch_logits = product("Qwen3.5 batch-logit elements", MAX_BATCH, Qwen35_9B::VOCAB)?;
        let mut builder = ArenaLayout::new();
        let input = builder.reserve(batch_hidden, ALIGNMENT)?;
        let final_norm_weight = builder.reserve(Qwen35_9B::HIDDEN, ALIGNMENT)?;
        let normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let lm_head_weight = builder.reserve(lm_head, ALIGNMENT)?;
        let logits = builder.reserve(batch_logits, ALIGNMENT)?;
        let resident_weight_bytes = final_norm_weight
            .byte_len()
            .checked_add(lm_head_weight.byte_len())
            .ok_or_else(|| EngineError::layout("Qwen3.5 endpoint weight bytes overflow"))?;
        let workspace_bytes = input
            .byte_len()
            .checked_add(normalized.byte_len())
            .and_then(|bytes| bytes.checked_add(logits.byte_len()))
            .ok_or_else(|| EngineError::layout("Qwen3.5 endpoint workspace bytes overflow"))?;

        Ok(Self {
            builder,
            input,
            final_norm_weight,
            normalized,
            lm_head_weight,
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

    /// Source-backed final-norm and BF16 LM-head bytes.
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Address-stable input, normalized, and logit bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Resident weights plus workspace, excluding padding.
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

    pub(crate) const fn lm_head_weight(&self) -> ArenaRegion<u16> {
        self.lm_head_weight
    }

    pub(crate) const fn logits(&self) -> ArenaRegion<u16> {
        self.logits
    }
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, Qwen35TextEndpointLayout};

    #[test]
    fn source_and_workspace_accounting_are_exact() {
        let layout = Qwen35TextEndpointLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 2_034_245_632);
        assert_eq!(layout.workspace_bytes(), 4_104_192);
        assert_eq!(layout.owner_bytes(), 2_038_349_824);
        assert_eq!(layout.arena_bytes(), 2_038_349_824);
    }

    #[test]
    fn regions_are_aligned_disjoint_and_complete() {
        let layout = Qwen35TextEndpointLayout::build().unwrap();
        let regions = [
            layout.input(),
            layout.final_norm_weight(),
            layout.normalized(),
            layout.lm_head_weight(),
            layout.logits(),
        ];
        let mut end = 0usize;
        for region in regions {
            assert_eq!(region.offset_bytes() % ALIGNMENT, 0);
            assert!(end <= region.offset_bytes());
            end = region.offset_bytes() + region.byte_len();
        }
        assert_eq!(end, layout.arena_bytes());
    }
}

//! Address-stable Qwen3.6 text-endpoint layout.

use crate::common::math::product;
use crate::{EngineError, EngineResult, MAX_BATCH};
use tuisko_gpu::{ArenaLayout, ArenaRegion};
use tuisko_model::{Arch, Qwen36Moe35B};

const ALIGNMENT: usize = 256;
const NVFP4_GROUP: usize = 16;

/// Checked regions owned by the resident Qwen3.6 text endpoint.
#[derive(Clone, Debug)]
pub struct Qwen36TextEndpointLayout {
    builder: ArenaLayout,
    input: ArenaRegion<u16>,
    final_norm_weight: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    lm_head_weight_codes: ArenaRegion<u8>,
    lm_head_weight_scales: ArenaRegion<u8>,
    logits: ArenaRegion<u16>,
    resident_weight_bytes: usize,
    workspace_bytes: usize,
}

impl Qwen36TextEndpointLayout {
    /// Reserves every Qwen3.6 endpoint plane in one allocation.
    pub fn build() -> EngineResult<Self> {
        let batch_hidden = product(
            "Qwen3.6 endpoint batch-hidden",
            MAX_BATCH,
            Qwen36Moe35B::HIDDEN,
        )?;
        let weight_codes = product(
            "Qwen3.6 endpoint packed LM-head weights",
            Qwen36Moe35B::VOCAB,
            Qwen36Moe35B::HIDDEN / 2,
        )?;
        let weight_scales = product(
            "Qwen3.6 endpoint LM-head scales",
            Qwen36Moe35B::VOCAB,
            Qwen36Moe35B::HIDDEN / NVFP4_GROUP,
        )?;
        let batch_logits = product(
            "Qwen3.6 endpoint batch logits",
            MAX_BATCH,
            Qwen36Moe35B::VOCAB,
        )?;
        let mut builder = ArenaLayout::new();
        let input = builder.reserve(batch_hidden, ALIGNMENT)?;
        let final_norm_weight = builder.reserve(Qwen36Moe35B::HIDDEN, ALIGNMENT)?;
        let normalized = builder.reserve(batch_hidden, ALIGNMENT)?;
        let lm_head_weight_codes = builder.reserve(weight_codes, ALIGNMENT)?;
        let lm_head_weight_scales = builder.reserve(weight_scales, ALIGNMENT)?;
        let logits = builder.reserve(batch_logits, ALIGNMENT)?;
        let resident_weight_bytes = final_norm_weight
            .byte_len()
            .checked_add(lm_head_weight_codes.byte_len())
            .and_then(|bytes| bytes.checked_add(lm_head_weight_scales.byte_len()))
            .ok_or_else(|| EngineError::layout("Qwen3.6 endpoint weight bytes overflow"))?;
        let workspace_bytes = input
            .byte_len()
            .checked_add(normalized.byte_len())
            .and_then(|bytes| bytes.checked_add(logits.byte_len()))
            .ok_or_else(|| EngineError::layout("Qwen3.6 endpoint workspace bytes overflow"))?;

        Ok(Self {
            builder,
            input,
            final_norm_weight,
            normalized,
            lm_head_weight_codes,
            lm_head_weight_scales,
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

    /// Source-backed final-norm and NVFP4 LM-head bytes.
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

    pub(crate) const fn lm_head_weight_codes(&self) -> ArenaRegion<u8> {
        self.lm_head_weight_codes
    }

    pub(crate) const fn lm_head_weight_scales(&self) -> ArenaRegion<u8> {
        self.lm_head_weight_scales
    }

    pub(crate) const fn logits(&self) -> ArenaRegion<u16> {
        self.logits
    }
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, Qwen36TextEndpointLayout};

    #[test]
    fn source_and_workspace_accounting_are_exact() {
        let layout = Qwen36TextEndpointLayout::build().unwrap();

        assert_eq!(layout.resident_weight_bytes(), 286_068_736);
        assert_eq!(layout.workspace_bytes(), 4_038_656);
        assert_eq!(layout.owner_bytes(), 290_107_392);
        assert_eq!(layout.arena_bytes(), 290_107_392);
    }

    #[test]
    fn every_region_is_aligned_and_disjoint() {
        let layout = Qwen36TextEndpointLayout::build().unwrap();
        let spans = [
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
                layout.lm_head_weight_codes().offset_bytes(),
                layout.lm_head_weight_codes().byte_len(),
            ),
            (
                layout.lm_head_weight_scales().offset_bytes(),
                layout.lm_head_weight_scales().byte_len(),
            ),
            (layout.logits().offset_bytes(), layout.logits().byte_len()),
        ];

        for (index, &(offset, bytes)) in spans.iter().enumerate() {
            assert_eq!(offset % ALIGNMENT, 0);
            if let Some(next) = spans.get(index + 1) {
                assert!(offset + bytes <= next.0);
            }
        }
        let (offset, bytes) = spans.last().unwrap();
        assert_eq!(offset + bytes, layout.arena_bytes());
    }
}

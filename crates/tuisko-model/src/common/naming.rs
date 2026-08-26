//! Checkpoint-wide tensor key constants shared by every admitted target.

pub(crate) const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
pub(crate) const FINAL_NORM: &str = "model.language_model.norm.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
pub(crate) const LM_HEAD_SCALE: &str = "lm_head.weight_scale";
pub(crate) const MTP_LAYER: usize = 0;

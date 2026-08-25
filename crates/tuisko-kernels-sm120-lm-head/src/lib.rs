//! Exact language-model head operators for SM120.
//!
//! Neither route shares a device body with another family, so they are grouped
//! by the vocabulary projection they both perform.

mod bf16_lm_head;
mod qwen36_nvfp4_lm_head;

pub use bf16_lm_head::Qwen35Bf16LmHeadOp;
pub use qwen36_nvfp4_lm_head::Qwen36Nvfp4LmHeadOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    bf16_lm_head::qwen35_bf16_lm_head_ptx_names()
        .into_iter()
        .chain(qwen36_nvfp4_lm_head::qwen36_nvfp4_lm_head_ptx_names())
        .collect()
}

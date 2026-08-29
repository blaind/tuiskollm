//! Source-native dense-FP8 projection operators for SM120.
//!
//! Every operator here shares the `device::fp8_projection` body, so they are
//! codegen'd together: splitting them would re-prepare that body per crate.

mod attention_output;
mod device;
mod gdn_input_tma;
mod gdn_output;
mod gdn_output_tma;
mod projection;
mod qkv_tma;
mod qwen36_attention_output;
mod qwen36_fp8_qkv;
mod qwen36_gdn_input;
mod qwen36_gdn_output;
mod qwen38_flash_next_attention_output;

pub use attention_output::AttentionOutputOp;
pub use gdn_input_tma::DenseFp8GdnInputTmaMaps;
pub use gdn_output::GdnOutputProjectionOp;
pub use gdn_output_tma::DenseFp8GdnOutputTmaMaps;
pub use projection::{FullAttentionQkvOp, GdnInputProjectionOp, LmHeadOp};
pub use qkv_tma::DenseFp8QkvTmaMaps;
pub use qwen36_attention_output::Qwen36AttentionOutputOp;
pub use qwen36_fp8_qkv::Qwen36Fp8QkvOp;
pub use qwen36_gdn_input::Qwen36GdnInputOp;
pub use qwen36_gdn_output::Qwen36GdnOutputOp;
pub use qwen38_flash_next_attention_output::Qwen38FlashNextAttentionGateOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    attention_output::attention_output_ptx_names()
        .into_iter()
        .chain(projection::fp8_qkv_ptx_names())
        .chain(projection::fp8_gdn_input_ptx_names())
        .chain(projection::fp8_lm_head_ptx_names())
        .chain(gdn_output::gdn_output_ptx_names())
        .chain(qwen36_gdn_input::qwen36_gdn_input_ptx_names())
        .chain(qwen36_gdn_output::qwen36_gdn_output_ptx_names())
        .chain(qwen36_fp8_qkv::qwen36_fp8_qkv_ptx_names())
        .chain(qwen36_attention_output::qwen36_attention_output_ptx_names())
        .chain(qwen38_flash_next_attention_output::qwen38_flash_next_attention_output_ptx_names())
        .collect()
}

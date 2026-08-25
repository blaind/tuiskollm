//! MTP BF16 speculative-decode operators for SM120.
//!
//! Two of these kernels launch entries around device bodies owned by the
//! full-attention crate, and the fusion operator composes the RMSNorm owner,
//! so this crate depends on both rather than copying either.

mod mtp_bf16_attention_output;
mod mtp_bf16_fusion;
mod mtp_bf16_mlp;
mod mtp_bf16_paged_gqa;
mod mtp_bf16_qk_prepare;
mod mtp_bf16_qkv;

pub use mtp_bf16_attention_output::{
    MtpBf16AttentionOutputOp, Qwen35MtpBf16AttentionOutputOp, Qwen36MtpBf16AttentionOutputOp,
};
pub use mtp_bf16_fusion::{MtpBf16FusionOp, Qwen35MtpBf16FusionOp, Qwen36MtpBf16FusionOp};
pub use mtp_bf16_mlp::{MtpBf16MlpOp, Qwen35MtpBf16MlpOp};
pub use mtp_bf16_paged_gqa::{MtpBf16PagedGqaOp, Qwen35MtpBf16PagedGqaOp};
pub use mtp_bf16_qk_prepare::{MtpBf16QkPrepareOp, Qwen35MtpBf16QkPrepareOp};
pub use mtp_bf16_qkv::{MtpBf16QkvOp, Qwen35MtpBf16QkvOp, Qwen36MtpBf16QkvOp};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    mtp_bf16_fusion::mtp_bf16_fusion_ptx_names()
        .into_iter()
        .chain(mtp_bf16_fusion::mtp_bf16_fusion_prefill_ptx_names())
        .chain(mtp_bf16_fusion::qwen35_mtp_bf16_fusion_ptx_names())
        .chain(mtp_bf16_fusion::qwen36_mtp_bf16_fusion_ptx_names())
        .chain(mtp_bf16_mlp::mtp_bf16_mlp_ptx_names())
        .chain(mtp_bf16_mlp::qwen35_mtp_bf16_mlp_ptx_names())
        .chain(mtp_bf16_attention_output::mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_attention_output::qwen35_mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_attention_output::qwen36_mtp_bf16_attention_output_ptx_names())
        .chain(mtp_bf16_qkv::mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qkv::mtp_bf16_qkv_prefill_ptx_names())
        .chain(mtp_bf16_qkv::qwen35_mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qkv::qwen36_mtp_bf16_qkv_ptx_names())
        .chain(mtp_bf16_qk_prepare::mtp_bf16_qk_prepare_ptx_names())
        .chain(mtp_bf16_qk_prepare::mtp_bf16_qk_prepare_prefill_ptx_names())
        .chain(mtp_bf16_qk_prepare::qwen35_mtp_bf16_qk_prepare_ptx_names())
        .chain(mtp_bf16_paged_gqa::mtp_bf16_paged_gqa_ptx_names())
        .chain(mtp_bf16_paged_gqa::qwen35_mtp_bf16_paged_gqa_ptx_names())
        .collect()
}

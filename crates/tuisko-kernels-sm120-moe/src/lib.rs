//! Exact Qwen3.6 mixture-of-experts router and expert operators for SM120.

mod experts;
mod qwen36_mtp_bf16_moe;
mod router;

pub use experts::Qwen36MoeExpertsOp;
pub use qwen36_mtp_bf16_moe::Qwen36MtpBf16MoeOp;
pub use router::Qwen36MoeRouterOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    router::qwen36_moe_router_ptx_names()
        .into_iter()
        .chain(experts::qwen36_moe_experts_ptx_names())
        .chain(qwen36_mtp_bf16_moe::qwen36_mtp_bf16_moe_ptx_names())
        .collect()
}

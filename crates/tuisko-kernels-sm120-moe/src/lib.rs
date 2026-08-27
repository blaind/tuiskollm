//! Exact mixture-of-experts router and expert operators for SM120.
//!
//! Two targets share this family behind sealed per-target operators: Qwen3.6,
//! whose expert pool is device-resident and expert-major, and
//! Qwen3.8-Flash-Next, whose streaming pool uses device-visible slot indices.

mod experts;
mod qwen36_mtp_bf16_moe;
mod qwen38_flash_next_experts;
mod qwen38_flash_next_router;
mod router;

pub use experts::Qwen36MoeExpertsOp;
pub use qwen36_mtp_bf16_moe::Qwen36MtpBf16MoeOp;
pub use qwen38_flash_next_experts::{
    QWEN38_FLASH_NEXT_ABSENT_SLOT, QWEN38_FLASH_NEXT_EXPERT_SLOT_BYTES,
    Qwen38FlashNextExpertDispatch, Qwen38FlashNextMoeExpertsOp, Qwen38FlashNextSlotPlane,
    qwen38_flash_next_expert_slot_plane,
};
pub use qwen38_flash_next_router::Qwen38FlashNextMoeRouterOp;
pub use router::Qwen36MoeRouterOp;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    router::qwen36_moe_router_ptx_names()
        .into_iter()
        .chain(experts::qwen36_moe_experts_ptx_names())
        .chain(qwen36_mtp_bf16_moe::qwen36_mtp_bf16_moe_ptx_names())
        .chain(qwen38_flash_next_router::qwen38_flash_next_moe_router_ptx_names())
        .chain(qwen38_flash_next_experts::qwen38_flash_next_moe_experts_ptx_names())
        .collect()
}

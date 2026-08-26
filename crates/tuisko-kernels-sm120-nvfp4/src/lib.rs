//! NVFP4 W4A4 SwiGLU, down, QKV and output operators for SM120.
//!
//! Every operator here shares the `device::nvfp4_prefill` W4A4 prefill body,
//! including the Qwen3.5 attention-output and GDN-output routes whose model
//! stage sits with full attention.

mod device;
mod nvfp4_down;
mod nvfp4_gdn_input;
mod nvfp4_output;
mod nvfp4_qkv;
mod nvfp4_swiglu;

pub use nvfp4_down::{
    Nvfp4DownA16Route, Nvfp4DownEntries, Nvfp4DownOp, Nvfp4DownPrefillRoute, PreparedBatchOneRoute,
    PreparedBatchRoute, PreparedPrefillRoute, PreparedQwen35BatchRoute, PreparedQwen35PrefillRoute,
    Qwen35Nvfp4DownEntries, Qwen35Nvfp4DownOp, Qwen38Nvfp4DownEntries,
};
pub use nvfp4_gdn_input::Qwen35Nvfp4GdnInputOp;
pub use nvfp4_output::{Qwen35Nvfp4AttentionOutputOp, Qwen35Nvfp4GdnOutputOp};
pub use nvfp4_qkv::Qwen35Nvfp4QkvOp;
pub use nvfp4_swiglu::{
    A16Slot, Nvfp4SwiGluEntries, Nvfp4SwiGluOp, PreparedA16Routes, PreparedQwen35A16Routes,
    PreparedQwen35W4a4Route, PreparedW4a4Route, Qwen35Nvfp4SwiGluEntries, Qwen35Nvfp4SwiGluOp,
    Qwen38Nvfp4SwiGluEntries, SwiGluA16Routes, SwiGluRoute, SwiGluW4a4Route, UnadmittedRoute,
};

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    nvfp4_output::qwen35_nvfp4_attention_output_ptx_names()
        .into_iter()
        .chain(nvfp4_swiglu::nvfp4_swiglu_ptx_names())
        .chain(nvfp4_swiglu::qwen35_nvfp4_swiglu_ptx_names())
        .chain(nvfp4_down::nvfp4_down_ptx_names())
        .chain(nvfp4_down::qwen35_nvfp4_down_ptx_names())
        .chain(nvfp4_qkv::qwen35_nvfp4_qkv_ptx_names())
        .chain(nvfp4_gdn_input::qwen35_nvfp4_gdn_input_ptx_names())
        .collect()
}

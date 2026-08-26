//! Source-native dense-FP8 SwiGLU and down-projection operators for SM120.

mod device;
mod down;
mod down_tma;
mod swiglu;
mod swiglu_tma;

pub use down::DenseFp8DownOp;
pub use down_tma::DenseFp8DownTmaMaps;
pub use swiglu::DenseFp8SwiGluOp;
pub use swiglu_tma::DenseFp8SwiGluTmaMaps;

/// Semantic inventory of every entry this family emits.
pub fn kernel_ptx_names() -> Vec<&'static str> {
    swiglu::fp8_swiglu_ptx_names()
        .into_iter()
        .chain(down::fp8_down_ptx_names())
        .collect()
}

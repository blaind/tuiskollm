//! Operators consuming source-native NVFP4 weight planes.

mod swiglu;

pub use swiglu::Nvfp4SwiGluOp;
pub(crate) use swiglu::nvfp4_swiglu_ptx_names;

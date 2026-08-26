//! Architecture admission and device primitives shared by the SM120 kernel crates.
//!
//! The kernel families are split across crates so that editing one family
//! re-runs cuda-oxide device codegen only for that family. Anything placed
//! here is re-prepared once per dependent kernel crate, so this crate holds
//! only the sealed architecture bound and the leaf device bodies that more
//! than one family calls.

pub mod attention_output;
pub mod device;

use tuisko_model::{Arch, Qwen38_27B};

mod private {
    pub trait Sealed {}

    impl Sealed for tuisko_model::Qwen38_27B {}
}

/// Model architecture admitted by this compiled SM120 kernel artifact.
///
/// Device bodies and prepared owners remain parameterized by [`Arch`], while
/// this sealed bound prevents constructing an owner for a model whose exact
/// entries have not been emitted and qualified. Concrete artifact anchors
/// still instantiate the current target and therefore do not admit a model.
pub trait Sm120Arch: Arch + private::Sealed {}

impl Sm120Arch for Qwen38_27B {}

/// Synthetic architecture used by the kernel crates' host-side geometry tests.
///
/// It deliberately does not implement [`Sm120Arch`]: no entry is emitted for
/// it, so it may never reach a prepared launch.
#[derive(Clone, Copy)]
pub struct TestArch;

impl Arch for TestArch {
    const MODEL_ID: &'static str = "test/sm120-arch";
    const REVISION: &'static str = "test-revision";
    const HIDDEN: usize = 1_024;
    const RMS_NORM_EPSILON: f32 = 1.0e-5;
    const INTERMEDIATE: usize = 512;
    const VOCAB: usize = 512;
    const LAYERS: usize = 4;
    const FULL_ATTENTION_INTERVAL: usize = 2;
    const NUM_ATTENTION_HEADS: usize = 4;
    const NUM_KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 64;
    const LINEAR_KEY_HEADS: usize = 2;
    const LINEAR_VALUE_HEADS: usize = 4;
    const LINEAR_HEAD_DIM: usize = 32;
    const LINEAR_CONV_KERNEL_DIM: usize = 4;
    const MTP_LAYERS: usize = 1;
    const MTP_USES_DEDICATED_EMBEDDINGS: bool = false;
    const VISION_DEPTH: usize = 2;
    const VISION_HIDDEN: usize = 64;
    const VISION_INTERMEDIATE: usize = 128;
    const VISION_NUM_HEADS: usize = 4;
    const VISION_POSITIONS: usize = 16;
    const VISION_OUTPUT_HIDDEN: usize = 1_024;
    const VISION_INPUT_CHANNELS: usize = 3;
    const VISION_PATCH_SIZE: usize = 8;
    const VISION_SPATIAL_MERGE_SIZE: usize = 2;
    const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
}

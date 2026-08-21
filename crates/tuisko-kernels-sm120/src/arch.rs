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

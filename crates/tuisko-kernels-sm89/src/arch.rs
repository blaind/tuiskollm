use tuisko_model::{Arch, Qwen38_27B};

mod private {
    pub trait Sealed {}

    impl Sealed for tuisko_model::Qwen38_27B {}
}

/// Model architecture admitted by this compiled SM89 kernel artifact.
pub trait Sm89Arch: Arch + private::Sealed {}

impl Sm89Arch for Qwen38_27B {}

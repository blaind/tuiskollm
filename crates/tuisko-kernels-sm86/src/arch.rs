use tuisko_model::{Arch, Qwen38_27B};

mod private {
    pub trait Sealed {}

    impl Sealed for tuisko_model::Qwen38_27B {}
}

/// Model architecture admitted by this compiled SM86 kernel artifact.
pub trait Sm86Arch: Arch + private::Sealed {}

impl Sm86Arch for Qwen38_27B {}

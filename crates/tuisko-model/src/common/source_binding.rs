//! Two-phase lifecycle shared by every admitted decoder-layer source binding.

use crate::common::inventory::CheckpointSnapshot;
use crate::common::materialized::MaterializedMemory;
use crate::{Arch, CheckpointResult};

pub mod sealed {
    /// Restricts `SourceLayerBinding` to this crate's admitted decoder-layer bindings.
    pub trait Sealed {}
}

/// Zero-copy source binding for one decoder layer and its lossless host materialization.
///
/// Sealed through `sealed::Sealed`, whose module is unreachable outside this crate: no
/// downstream type can implement it and no downstream trait can extend it. The `Sized`
/// supertrait also makes it dyn-incompatible, so `dyn SourceLayerBinding` does not compile
/// and every use is monomorphized with statically dispatched calls and no vtable.
///
/// The `A` parameter is the admission gate. A binding whose sources exist in exactly one
/// checkpoint implements this for that one `Arch` alone, so the type system refuses to bind it
/// against another target's snapshot; a binding whose inherent `bind` already admits any `A`
/// implements it generically and keeps its existing runtime contract check. Both methods
/// forward to the inherent constructors, so implementing this trait admits exactly what the
/// inherent `bind` and `materialize` already admitted.
pub trait SourceLayerBinding<'a, A: Arch>: sealed::Sealed + Sized {
    /// Runtime-native host layout this binding materializes into.
    type Materialized: MaterializedMemory;

    /// Binds zero-copy views for decoder `layer` from the immutable snapshot.
    fn bind(snapshot: &'a CheckpointSnapshot<A>, layer: usize) -> CheckpointResult<Self>;

    /// Losslessly transforms the bound source views into runtime-native host memory.
    fn materialize(self) -> CheckpointResult<Self::Materialized>;
}

#[cfg(test)]
mod tests {
    use super::SourceLayerBinding;
    use crate::common::nvfp4::{Nvfp4DownBindings, Nvfp4GateUpBindings};
    use crate::qwen35::bindings::{
        ModelOptNvfp4AttentionBindings, ModelOptNvfp4GdnBindings, ModelOptNvfp4MlpBindings,
    };
    use crate::qwen36::bindings::{
        Qwen36FullAttentionBindings, Qwen36GdnBindings, Qwen36MoeLayerBindings,
    };
    use crate::qwen38::bindings::FullAttentionQkvBindings;
    use crate::{Arch, Qwen35_9B, Qwen36Moe35B, Qwen38_27B};

    fn admits<'a, A: Arch, B: SourceLayerBinding<'a, A>>() {}

    #[test]
    fn every_admitted_layer_binding_shares_the_two_phase_lifecycle() {
        admits::<Qwen38_27B, FullAttentionQkvBindings<'_>>();
        admits::<Qwen38_27B, Nvfp4GateUpBindings<'_>>();
        admits::<Qwen38_27B, Nvfp4DownBindings<'_>>();

        admits::<Qwen35_9B, ModelOptNvfp4MlpBindings<'_>>();
        admits::<Qwen35_9B, ModelOptNvfp4AttentionBindings<'_>>();
        admits::<Qwen35_9B, ModelOptNvfp4GdnBindings<'_>>();
        admits::<Qwen35_9B, Nvfp4GateUpBindings<'_>>();
        admits::<Qwen35_9B, Nvfp4DownBindings<'_>>();

        admits::<Qwen36Moe35B, Qwen36MoeLayerBindings<'_>>();
        admits::<Qwen36Moe35B, Qwen36GdnBindings<'_>>();
        admits::<Qwen36Moe35B, Qwen36FullAttentionBindings<'_>>();
    }
}

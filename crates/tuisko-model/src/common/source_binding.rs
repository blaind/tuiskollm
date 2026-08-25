//! Two-phase lifecycle shared by every admitted decoder-layer source binding.
//!
//! The lifecycle is shared here; the layer binding structs stay concrete and per-target. A
//! generic layer struct over the ModelOpt pair's linear representation was considered and
//! rejected against this tree:
//!
//! 1. The GDN control planes are not the same source shape. Qwen3.5 binds `in_proj_a` and
//!    `in_proj_b` as four-tensor ModelOpt NVFP4 linears (packed weight, E4M3 block scale,
//!    `input_scale`, `weight_scale_2`); Qwen3.6 binds one BF16 `in_proj_a.weight`. One linear
//!    parameter cannot express slots that are a linear in one checkpoint and a raw plane in
//!    the other, and a second parameter would make source shapes neither checkpoint has
//!    constructible.
//! 2. The two targets re-validate their layer route from different state. Qwen3.5 carries
//!    `layer_count` and `full_attention_interval` on the binding; Qwen3.6 re-derives them from
//!    its own `Arch` constants at materialization. Merging changes admission strictness on one
//!    target.
//! 3. The sibling-scale admission errors are per-target text (`query/key input_scale` against
//!    `Q/K input_scale`), so a shared body would change the observable admission surface of two
//!    mature targets as a deduplication side effect.

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

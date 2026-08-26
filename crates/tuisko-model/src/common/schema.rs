//! Per-target checkpoint admission schemas and the single source-contract dispatcher.
//!
//! Before this module, `config_util` and `inventory` each carried their own
//! `A::CHECKPOINT_CONTRACT` match and imported the three targets' config structs, validators,
//! and inventory specs directly. Both matches now live here, each arm is one generic call, and
//! every target's admission surface stays inside that target's directory.
//!
//! The inventory half of the trait is `open_snapshot`, not an expected-tensor map: the two
//! ModelOpt targets generate a complete `ExpectedTensor` map, while the compressed-tensors
//! target admits its two shards through the index weight map and its pinned entry, tensor, and
//! byte counts. Requiring a shared expected-tensor map would mean inventing an inventory the
//! product target does not have.

use crate::common::inventory::CheckpointSnapshot;
use crate::qwen35::ModelOptNvfp4Schema;
use crate::qwen36::ModelOptNvfp4MoeSchema;
use crate::qwen38::CompressedTensorsSchema;
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use std::path::Path;

/// Complete config and inventory admission surface of one source contract.
///
/// Sealed by visibility: the trait is `pub(crate)`, so no other crate can name it, implement
/// it, or bound on it. Monomorphized only — every dispatch arm below resolves to one concrete
/// schema type with statically dispatched calls and no vtable.
///
/// `CONTRACT` is the admission gate. `require_contract` refuses any pairing whose schema does
/// not admit the target's own `CHECKPOINT_CONTRACT`, so a mis-wired dispatch arm fails at the
/// first admission rather than validating a checkpoint against another contract's schema.
pub(crate) trait CheckpointSchema<A: Arch> {
    /// Source contract this schema admits.
    const CONTRACT: CheckpointContract;

    /// Validates `config.json` schema and quantization metadata for this target.
    fn validate_config(path: &Path) -> CheckpointResult<()>;

    /// Opens and validates the pinned snapshot's complete tensor inventory.
    fn open_snapshot(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>>;
}

/// Validates a checkpoint config against the selected architecture.
pub fn validate_config<A: Arch>(path: &Path) -> CheckpointResult<()> {
    match A::CHECKPOINT_CONTRACT {
        CheckpointContract::CompressedTensors => admit_config::<A, CompressedTensorsSchema>(path),
        CheckpointContract::ModelOptNvfp4 => admit_config::<A, ModelOptNvfp4Schema>(path),
        CheckpointContract::ModelOptNvfp4Moe => admit_config::<A, ModelOptNvfp4MoeSchema>(path),
    }
}

pub(crate) fn open_snapshot<A: Arch>(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>> {
    match A::CHECKPOINT_CONTRACT {
        CheckpointContract::CompressedTensors => admit_snapshot::<A, CompressedTensorsSchema>(root),
        CheckpointContract::ModelOptNvfp4 => admit_snapshot::<A, ModelOptNvfp4Schema>(root),
        CheckpointContract::ModelOptNvfp4Moe => admit_snapshot::<A, ModelOptNvfp4MoeSchema>(root),
    }
}

fn admit_config<A: Arch, S: CheckpointSchema<A>>(path: &Path) -> CheckpointResult<()> {
    require_contract::<A, S>()?;

    S::validate_config(path)
}

fn admit_snapshot<A: Arch, S: CheckpointSchema<A>>(
    root: &Path,
) -> CheckpointResult<CheckpointSnapshot<A>> {
    require_contract::<A, S>()?;

    S::open_snapshot(root)
}

fn require_contract<A: Arch, S: CheckpointSchema<A>>() -> CheckpointResult<()> {
    if S::CONTRACT != A::CHECKPOINT_CONTRACT {
        return Err(CheckpointError::config(format!(
            "{} declares checkpoint contract {:?} but was routed to a {:?} schema",
            A::MODEL_ID,
            A::CHECKPOINT_CONTRACT,
            S::CONTRACT
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CheckpointSchema, require_contract};
    use crate::qwen35::ModelOptNvfp4Schema;
    use crate::qwen36::ModelOptNvfp4MoeSchema;
    use crate::qwen38::CompressedTensorsSchema;
    use crate::{
        Arch, CheckpointContract, CheckpointErrorCode, Qwen35_9B, Qwen36Moe35B, Qwen38_27B,
    };

    #[test]
    fn each_target_is_routed_to_the_schema_that_admits_its_contract() {
        assert_eq!(
            <CompressedTensorsSchema as CheckpointSchema<Qwen38_27B>>::CONTRACT,
            Qwen38_27B::CHECKPOINT_CONTRACT
        );
        assert_eq!(
            <ModelOptNvfp4Schema as CheckpointSchema<Qwen35_9B>>::CONTRACT,
            Qwen35_9B::CHECKPOINT_CONTRACT
        );
        assert_eq!(
            <ModelOptNvfp4MoeSchema as CheckpointSchema<Qwen36Moe35B>>::CONTRACT,
            Qwen36Moe35B::CHECKPOINT_CONTRACT
        );

        require_contract::<Qwen38_27B, CompressedTensorsSchema>().unwrap();
        require_contract::<Qwen35_9B, ModelOptNvfp4Schema>().unwrap();
        require_contract::<Qwen36Moe35B, ModelOptNvfp4MoeSchema>().unwrap();
    }

    #[test]
    fn refuses_a_schema_that_does_not_admit_the_target_contract() {
        for (error, expected) in [
            (
                require_contract::<Qwen35_9B, CompressedTensorsSchema>()
                    .err()
                    .unwrap(),
                CheckpointContract::CompressedTensors,
            ),
            (
                require_contract::<Qwen38_27B, ModelOptNvfp4MoeSchema>()
                    .err()
                    .unwrap(),
                CheckpointContract::ModelOptNvfp4Moe,
            ),
        ] {
            assert_eq!(error.code(), CheckpointErrorCode::Config);
            assert!(
                error
                    .to_string()
                    .contains(&format!("routed to a {expected:?} schema")),
                "{error}"
            );
        }
    }
}

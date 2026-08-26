//! Source admission for `nvidia/Qwen3.6-35B-A3B-NVFP4` (ModelOpt NVFP4 MoE).
//!
//! Holds only items no other target reaches: the routed-expert and mixed FP8/NVFP4 bindings and
//! their materialization, the MoE MTP and text endpoints, the mixed-precision config schema, and
//! the indexed three-shard inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

use crate::common::inventory::CheckpointSnapshot;
use crate::common::schema::CheckpointSchema;
use crate::qwen36::config::{Qwen36Config, validate_qwen36, validate_qwen36_hf_quantization};
use crate::qwen36::inventory::QWEN36_INVENTORY;
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use std::fs;
use std::path::Path;

/// Config and inventory admission for the ModelOpt mixed FP8/NVFP4 MoE contract.
pub(crate) struct ModelOptNvfp4MoeSchema;

impl<A: Arch> CheckpointSchema<A> for ModelOptNvfp4MoeSchema {
    const CONTRACT: CheckpointContract = CheckpointContract::ModelOptNvfp4Moe;

    fn validate_config(path: &Path) -> CheckpointResult<()> {
        let bytes =
            fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;
        let config: Qwen36Config =
            serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))?;

        validate_qwen36::<A>(path, &config)?;
        validate_qwen36_hf_quantization::<A>(path)
    }

    fn open_snapshot(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>> {
        CheckpointSnapshot::open_indexed(root, QWEN36_INVENTORY)
    }
}

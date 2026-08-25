//! Source admission for `AxionML/Qwen3.5-9B-NVFP4` (ModelOpt NVFP4).
//!
//! Holds only items no other target reaches: the ModelOpt attention, GDN, and MLP bindings and
//! their materialization, the BF16 text endpoint, the ModelOpt config schema, and the single-shard
//! inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

use crate::common::inventory::CheckpointSnapshot;
use crate::common::schema::CheckpointSchema;
use crate::qwen35::config::{ModelOptConfig, validate_modelopt};
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use std::fs;
use std::path::Path;

/// Config and inventory admission for the ModelOpt NVFP4 dense contract.
pub(crate) struct ModelOptNvfp4Schema;

impl<A: Arch> CheckpointSchema<A> for ModelOptNvfp4Schema {
    const CONTRACT: CheckpointContract = CheckpointContract::ModelOptNvfp4;

    fn validate_config(path: &Path) -> CheckpointResult<()> {
        let bytes =
            fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;
        let config: ModelOptConfig =
            serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))?;

        validate_modelopt::<A>(path, &config)
    }

    fn open_snapshot(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>> {
        CheckpointSnapshot::open_modelopt(root)
    }
}

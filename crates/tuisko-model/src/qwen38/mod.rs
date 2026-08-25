//! Source admission for `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors).
//!
//! Holds only items no other target reaches: the dense-FP8 and packed-NVFP4 MLP bindings, the
//! FP8 full-attention and GDN families, the FP8 MTP and text endpoints, the compressed-tensors
//! config schema, and the split-shard inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

use crate::common::inventory::CheckpointSnapshot;
use crate::common::schema::CheckpointSchema;
use crate::qwen38::config::{ModelConfig, validate_compressed};
use crate::qwen38::inventory::QWEN38_INVENTORY;
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use std::fs;
use std::path::Path;

/// Config and inventory admission for the compressed-tensors mixed FP8/NVFP4 contract.
pub(crate) struct CompressedTensorsSchema;

impl<A: Arch> CheckpointSchema<A> for CompressedTensorsSchema {
    const CONTRACT: CheckpointContract = CheckpointContract::CompressedTensors;

    fn validate_config(path: &Path) -> CheckpointResult<()> {
        let bytes =
            fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;
        let config: ModelConfig =
            serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))?;

        validate_compressed::<A>(path, &config)
    }

    fn open_snapshot(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>> {
        CheckpointSnapshot::open_split_with_spec(root, QWEN38_INVENTORY)
    }
}

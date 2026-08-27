//! Source admission for the pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod engram;
pub(crate) mod engram_hash;
pub(crate) mod inventory;
pub(crate) mod materialize;

use crate::common::inventory::CheckpointSnapshot;
use crate::common::schema::CheckpointSchema;
use crate::qwen38_flash_next::config::{
    Qwen38FlashNextConfig, validate_qwen38_flash_next, validate_qwen38_flash_next_hf_quantization,
};
use crate::qwen38_flash_next::inventory::QWEN38_FLASH_NEXT_INVENTORY;
use crate::{Arch, CheckpointContract, CheckpointError, CheckpointResult};
use std::fs;
use std::path::Path;

/// Config and inventory admission for this target's ModelOpt NVFP4 contract.
pub(crate) struct Qwen38FlashNextModelOptNvfp4Schema;

impl<A: Arch> CheckpointSchema<A> for Qwen38FlashNextModelOptNvfp4Schema {
    const CONTRACT: CheckpointContract = CheckpointContract::Qwen38FlashNextModelOptNvfp4;

    fn validate_config(path: &Path) -> CheckpointResult<()> {
        let bytes =
            fs::read(path).map_err(|source| CheckpointError::io("reading", path, source))?;
        let config: Qwen38FlashNextConfig =
            serde_json::from_slice(&bytes).map_err(|source| CheckpointError::json(path, source))?;

        validate_qwen38_flash_next::<A>(path, &config)?;
        validate_qwen38_flash_next_hf_quantization(path)
    }

    fn open_snapshot(root: &Path) -> CheckpointResult<CheckpointSnapshot<A>> {
        CheckpointSnapshot::open_sharded(root, QWEN38_FLASH_NEXT_INVENTORY)
    }
}

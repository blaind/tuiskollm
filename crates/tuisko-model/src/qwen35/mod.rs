//! Source admission for `AxionML/Qwen3.5-9B-NVFP4` (ModelOpt NVFP4).
//!
//! Holds only items no other target reaches: the ModelOpt attention, GDN, and MLP bindings and
//! their materialization, the BF16 text endpoint, the ModelOpt config schema, and the single-shard
//! inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

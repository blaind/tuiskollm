//! Source admission for `nvidia/Qwen3.6-35B-A3B-NVFP4` (ModelOpt NVFP4 MoE).
//!
//! Holds only items no other target reaches: the routed-expert and mixed FP8/NVFP4 bindings and
//! their materialization, the MoE MTP and text endpoints, the mixed-precision config schema, and
//! the indexed three-shard inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

//! Source admission for `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors).
//!
//! Holds only items no other target reaches: the dense-FP8 and packed-NVFP4 MLP bindings, the
//! FP8 full-attention and GDN families, the FP8 MTP and text endpoints, the compressed-tensors
//! config schema, and the split-shard inventory spec.

pub(crate) mod bindings;
pub(crate) mod config;
pub(crate) mod inventory;
pub(crate) mod materialize;

//! Checkpoint items shared by more than one admitted target.
//!
//! Classification rule applied to the split of `bindings.rs`, `materialize.rs`, `config.rs`, and
//! `inventory.rs`: an item is shared when items belonging to two or more targets reach it, or when
//! a target-independent module reaches it. An item reached by exactly one target lives in that
//! target's directory. Three shared sets are load-bearing and would have been mis-filed by name
//! alone: the NVFP4 gate/up and down carriers in `nvfp4`, which the compressed-tensors MLP and the
//! ModelOpt MLP both use; `ModelOptNvfp4LinearBindings` in `modelopt_codec`, which Qwen3.5 dense
//! linears and Qwen3.6 routed experts both use; and `add_vision_expected_tensors` in `inventory`,
//! which both ModelOpt inventories call.

pub(crate) mod config_util;
pub(crate) mod inventory;
pub(crate) mod modelopt_codec;
pub(crate) mod mtp;
pub(crate) mod naming;
pub(crate) mod nvfp4;
pub(crate) mod routes;
pub(crate) mod scale_swizzle;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod vision;

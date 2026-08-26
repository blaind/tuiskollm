//! Qwen3.6-35B-A3B MoE target layer, endpoint, and generation programs.

pub(crate) mod batch_generation;
pub(crate) mod full_attention_layer;
pub(crate) mod full_attention_layer_layout;
pub(crate) mod gdn_moe_layer;
pub(crate) mod gdn_moe_layer_layout;
pub(crate) mod long_context_kv;
pub(crate) mod long_context_kv_layout;
pub(crate) mod mtp_layer;
pub(crate) mod mtp_layer_layout;
pub(crate) mod resident_model;
pub(crate) mod text_endpoint;
pub(crate) mod text_endpoint_layout;
pub(crate) mod text_generation;

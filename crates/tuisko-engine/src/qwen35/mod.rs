//! Qwen3.5-9B target layer, endpoint, MTP, and generation programs.

pub(crate) mod batch_generation;
pub(crate) mod full_attention_layer;
pub(crate) mod full_attention_layer_layout;
pub(crate) mod gdn_layer;
pub(crate) mod gdn_layer_layout;
pub(crate) mod long_context_kv;
pub(crate) mod long_context_kv_layout;
pub(crate) mod mtp_batch_generation;
pub(crate) mod mtp_generation;
pub(crate) mod mtp_kv;
pub(crate) mod mtp_kv_layout;
pub(crate) mod mtp_layer;
pub(crate) mod mtp_layer_layout;
pub(crate) mod nvfp4_mlp;
pub(crate) mod resident_model;
pub(crate) mod resident_mtp;
pub(crate) mod resident_mtp_layout;
pub(crate) mod text_endpoint;
pub(crate) mod text_endpoint_layout;

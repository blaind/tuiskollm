//! Qwen3.8-27B target layer, endpoint, and MTP programs.

pub(crate) mod dense_fp8_gdn_layer;
pub(crate) mod dense_fp8_gdn_layer_layout;
pub(crate) mod dense_fp8_mlp;
pub(crate) mod dense_fp8_mlp_layout;
pub(crate) mod full_attention_layer;
pub(crate) mod full_attention_layer_layout;
pub(crate) mod long_context_kv_layout;
pub(crate) mod mtp_layer;
pub(crate) mod mtp_layer_layout;
pub(crate) mod mtp_prompt_prime;
pub(crate) mod mtp_prompt_prime_layout;
pub(crate) mod nvfp4_mlp;
pub(crate) mod nvfp4_mlp_layout;
pub(crate) mod resident_model;
pub(crate) mod resident_model_layout;
pub(crate) mod resident_mtp;
pub(crate) mod resident_mtp_batch_generation;
pub(crate) mod resident_mtp_generation;
pub(crate) mod resident_mtp_layout;
pub(crate) mod upload_plan;

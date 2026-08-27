//! Repository build and qualification gates.

mod gpu_target;
mod perf_artifact;
mod perf_iteration;
mod performance;
mod qwen38_flash_next_server_qual;
mod remote;
mod server_bench;
mod server_performance;
mod server_qual;
mod server_qualification;

use gpu_target::BuildTargetProfile;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const RESIDUAL_NORM_RESOURCE_BASELINE: &str = "qual/baselines/residual-norm-sm120.txt";
const QWEN38_FLASH_NEXT_HYPER_CONNECTION_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-hyper-connection-sm120.txt";
const QWEN38_FLASH_NEXT_PLE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-ple-sm120.txt";
const QWEN38_FLASH_NEXT_PLE_TEST_FILTER: &str = "qwen38_flash_next_ple_suite_";
const QWEN38_FLASH_NEXT_PROJECTION_TEST_FILTER: &str = "qwen38_flash_next_projection";
const QWEN38_FLASH_NEXT_LM_HEAD_TEST_FILTER: &str = "qwen38_flash_next_lm_head";
const QWEN38_FLASH_NEXT_GDN_LAYER_TEST_FILTER: &str = "qwen38_flash_next_gdn_moe_layer";
const QWEN38_FLASH_NEXT_QSA_LAYER_TEST_FILTER: &str = "qwen38_flash_next_qsa_moe_layer";
const QWEN38_FLASH_NEXT_RESIDENT_MODEL_TEST_FILTER: &str = "qwen38_flash_next_resident_model";
const QWEN38_FLASH_NEXT_GENERATION_TEST_FILTER: &str = "qwen38_flash_next_generation";
const QWEN38_FLASH_NEXT_PROMPT_PRIME_TEST_FILTER: &str = "qwen38_flash_next_prompt_prime";
const QWEN38_FLASH_NEXT_MTP_GENERATION_TEST_FILTER: &str = "qwen38_flash_next_mtp_generation";
const QWEN38_FLASH_NEXT_MTP_ORACLE_TEST_FILTER: &str = "qwen38_flash_next_mtp_oracle";
const QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-residual-norm-sm120.txt";
const QWEN35_RESIDUAL_NORM_TEST_FILTER: &str = "qwen35_residual_norm";
const QWEN35_LONG_CONTEXT_KV_TEST_FILTER: &str = "qwen35_long_context_kv::tests";
const QWEN35_TEXT_ENDPOINT_TEST_FILTER: &str = "qwen35_text_endpoint_suite_";
const QWEN35_RESIDENT_MODEL_TEST_FILTER: &str = "qwen35_resident_model_suite_";
const QWEN35_RESIDENT_MTP_TEST_FILTER: &str = "qwen35_resident_mtp_suite_";
const QWEN35_MTP_GENERATION_TEST_FILTER: &str = "qwen35_mtp_generation_suite_";
const QWEN35_MTP_BATCH_GENERATION_TEST_FILTER: &str = "qwen35_mtp_batch_generation_suite_";
const QWEN36_MTP_LAYER_TEST_FILTER: &str = "qwen36_mtp_layer_suite_";
const QWEN36_LONG_CONTEXT_KV_TEST_FILTER: &str = "qwen36_long_context_kv::tests";
const STREAMING_WEIGHT_POOL_TEST_FILTER: &str = "streaming_weight_pool_suite_";
const QWEN38_FLASH_NEXT_ENGRAM_STAGING_TEST_FILTER: &str =
    "qwen38_flash_next_engram_staging_suite_";
const MTP_BF16_PAGED_GQA_BENCHMARK_FILTER: &str =
    "bf16_paged_gqa_benchmark::tests::mtp_bf16_paged_gqa_";
const MTP_LAYER_TEST_FILTER: &str = "mtp_layer::tests::mtp_layer_suite_";
const MTP_LAYER_BENCHMARK_FILTER: &str = "mtp_layer_benchmark::tests::mtp_layer_suite_";
const QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-residual-norm-sm120.txt";
const QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-swiglu-sm120.txt";
const QWEN35_NVFP4_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-down-sm120.txt";
const QWEN35_NVFP4_QKV_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-nvfp4-qkv-sm120.txt";
const QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-bf16-lm-head-sm120.txt";
const QWEN36_MOE_ROUTER_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-moe-router-sm120.txt";
const QWEN36_MOE_EXPERTS_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-moe-experts-sm120.txt";
const QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-nvfp4-lm-head-sm120.txt";
const QWEN36_FP8_QKV_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-fp8-qkv-sm120.txt";
const QWEN36_GDN_INPUT_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-gdn-input-sm120.txt";
const QWEN36_GDN_OUTPUT_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-gdn-output-sm120.txt";
const QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-attention-output-sm120.txt";
const QWEN36_RESIDENT_MODEL_TEST_FILTER: &str = "qwen36_resident_model";
const QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-nvfp4-gdn-input-sm120.txt";
const QWEN35_GDN_PREPARE_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-gdn-prepare-sm120.txt";
const QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-gdn-recurrence-sm120.txt";
const QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-nvfp4-attention-output-sm120.txt";
const FP8_QKV_RESOURCE_BASELINE: &str = "qual/baselines/fp8-qkv-sm120.txt";
const FP8_GDN_INPUT_RESOURCE_BASELINE: &str = "qual/baselines/fp8-gdn-input-sm120.txt";
const FP8_LM_HEAD_RESOURCE_BASELINE: &str = "qual/baselines/fp8-lm-head-sm120.txt";
const FP8_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/fp8-swiglu-sm120.txt";
const FP8_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/fp8-down-sm120.txt";
const NVFP4_SWIGLU_RESOURCE_BASELINE: &str = "qual/baselines/nvfp4-swiglu-sm120.txt";
const NVFP4_DOWN_RESOURCE_BASELINE: &str = "qual/baselines/nvfp4-down-sm120.txt";
const GDN_PREPARE_RESOURCE_BASELINE: &str = "qual/baselines/gdn-prepare-sm120.txt";
const GDN_RECURRENCE_RESOURCE_BASELINE: &str = "qual/baselines/gdn-recurrence-sm120.txt";
const GDN_STATE_SNAPSHOT_RESOURCE_BASELINE: &str = "qual/baselines/gdn-state-snapshot-sm120.txt";
const GDN_OUTPUT_RESOURCE_BASELINE: &str = "qual/baselines/gdn-output-sm120.txt";
const QWEN38_FLASH_NEXT_GDN_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-gdn-prepare-sm120.txt";
const QWEN38_FLASH_NEXT_GDN_RECURRENCE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-gdn-recurrence-sm120.txt";
const QWEN38_FLASH_NEXT_QSA_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-qsa-prepare-sm120.txt";
const QWEN38_FLASH_NEXT_QSA_ATTENTION_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-qsa-attention-sm120.txt";
const QWEN38_FLASH_NEXT_QSA_SELECTION_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-qsa-selection-sm120.txt";
const QWEN38_FLASH_NEXT_MOE_ROUTER_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-moe-router-sm120.txt";
const QWEN38_FLASH_NEXT_MOE_EXPERTS_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-moe-experts-sm120.txt";
const QWEN38_FLASH_NEXT_PROJECTION_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-projection-sm120.txt";
const QWEN38_FLASH_NEXT_LM_HEAD_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen38-flash-next-lm-head-sm120.txt";
const ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/attention-qk-prepare-sm120.txt";
const QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen35-attention-qk-prepare-sm120.txt";
const QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-attention-qk-prepare-sm120.txt";
const QWEN36_FP8_ATTENTION_QK_PREPARE_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-fp8-attention-qk-prepare-sm120.txt";
const PAGED_GQA_RESOURCE_BASELINE: &str = "qual/baselines/paged-gqa-sm120.txt";
const QWEN35_PAGED_GQA_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-paged-gqa-sm120.txt";
const QWEN36_PAGED_GQA_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-paged-gqa-sm120.txt";
const QWEN36_FP8_PAGED_GQA_RESOURCE_BASELINE: &str =
    "qual/baselines/qwen36-fp8-paged-gqa-sm120.txt";
const LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE: &str =
    "qual/baselines/long-context-paged-gqa-sm120.txt";
const ATTENTION_OUTPUT_RESOURCE_BASELINE: &str = "qual/baselines/attention-output-sm120.txt";
const MTP_BF16_FUSION_RESOURCE_BASELINE: &str = "qual/baselines/mtp-bf16-fusion-sm120.txt";
const MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE: &str =
    "qual/baselines/mtp-bf16-attention-output-sm120.txt";
const MTP_BF16_MLP_RESOURCE_BASELINE: &str = "qual/baselines/mtp-bf16-mlp-sm120.txt";
const MTP_BF16_QKV_RESOURCE_BASELINE: &str = "qual/baselines/mtp-bf16-qkv-sm120.txt";
const MTP_BF16_QK_PREPARE_RESOURCE_BASELINE: &str = "qual/baselines/mtp-bf16-qk-prepare-sm120.txt";
const MTP_BF16_PAGED_GQA_RESOURCE_BASELINE: &str = "qual/baselines/mtp-bf16-paged-gqa-sm120.txt";
const QWEN35_MTP_RESOURCE_BASELINE: &str = "qual/baselines/qwen35-mtp-sm120.txt";
const QWEN36_MTP_RESOURCE_BASELINE: &str = "qual/baselines/qwen36-mtp-sm120.txt";
const RESIDENT_MODEL_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_STATE_SNAPSHOT_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
];
const SM120_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_QKV_RESOURCE_BASELINE,
    QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE,
    QWEN36_MOE_ROUTER_RESOURCE_BASELINE,
    QWEN36_MOE_EXPERTS_RESOURCE_BASELINE,
    QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE,
    QWEN36_FP8_QKV_RESOURCE_BASELINE,
    QWEN36_GDN_INPUT_RESOURCE_BASELINE,
    QWEN36_GDN_OUTPUT_RESOURCE_BASELINE,
    QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_STATE_SNAPSHOT_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_QSA_PREPARE_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_QSA_ATTENTION_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_QSA_SELECTION_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_MOE_ROUTER_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_MOE_EXPERTS_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_PROJECTION_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_LM_HEAD_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN36_FP8_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    QWEN35_PAGED_GQA_RESOURCE_BASELINE,
    QWEN36_PAGED_GQA_RESOURCE_BASELINE,
    QWEN36_FP8_PAGED_GQA_RESOURCE_BASELINE,
    LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
    MTP_BF16_FUSION_RESOURCE_BASELINE,
    MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    MTP_BF16_MLP_RESOURCE_BASELINE,
    MTP_BF16_QKV_RESOURCE_BASELINE,
    MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
    MTP_BF16_PAGED_GQA_RESOURCE_BASELINE,
    QWEN35_MTP_RESOURCE_BASELINE,
    QWEN36_MTP_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_HYPER_CONNECTION_RESOURCE_BASELINE,
    QWEN38_FLASH_NEXT_PLE_RESOURCE_BASELINE,
];
const NVFP4_MLP_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
];
const DENSE_FP8_MLP_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const DENSE_FP8_GDN_LAYER_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const FULL_ATTENTION_LAYER_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
];
const MTP_LAYER_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    MTP_BF16_FUSION_RESOURCE_BASELINE,
    MTP_BF16_QKV_RESOURCE_BASELINE,
    MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
    MTP_BF16_PAGED_GQA_RESOURCE_BASELINE,
    MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    MTP_BF16_MLP_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
];
const QWEN35_MTP_LAYER_RESOURCE_BASELINES: &[&str] = &[
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_MTP_RESOURCE_BASELINE,
];
const MTP_PROMPT_PRIME_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    MTP_BF16_FUSION_RESOURCE_BASELINE,
    MTP_BF16_QKV_RESOURCE_BASELINE,
    MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
];
const RESIDENT_MTP_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_QKV_RESOURCE_BASELINE,
    FP8_GDN_INPUT_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
    FP8_SWIGLU_RESOURCE_BASELINE,
    FP8_DOWN_RESOURCE_BASELINE,
    NVFP4_SWIGLU_RESOURCE_BASELINE,
    NVFP4_DOWN_RESOURCE_BASELINE,
    GDN_PREPARE_RESOURCE_BASELINE,
    GDN_RECURRENCE_RESOURCE_BASELINE,
    GDN_STATE_SNAPSHOT_RESOURCE_BASELINE,
    GDN_OUTPUT_RESOURCE_BASELINE,
    ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    PAGED_GQA_RESOURCE_BASELINE,
    LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
    ATTENTION_OUTPUT_RESOURCE_BASELINE,
    MTP_BF16_FUSION_RESOURCE_BASELINE,
    MTP_BF16_QKV_RESOURCE_BASELINE,
    MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
    MTP_BF16_PAGED_GQA_RESOURCE_BASELINE,
    MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    MTP_BF16_MLP_RESOURCE_BASELINE,
];
const QWEN35_FULL_ATTENTION_LAYER_RESOURCE_BASELINES: &[&str] = &[
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_NVFP4_QKV_RESOURCE_BASELINE,
    QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN35_PAGED_GQA_RESOURCE_BASELINE,
    QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
];
const QWEN35_GDN_LAYER_RESOURCE_BASELINES: &[&str] = &[
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
];
const QWEN36_GDN_MOE_LAYER_RESOURCE_BASELINES: &[&str] = &[
    QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN36_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN36_GDN_OUTPUT_RESOURCE_BASELINE,
    QWEN36_MOE_ROUTER_RESOURCE_BASELINE,
    QWEN36_MOE_EXPERTS_RESOURCE_BASELINE,
];
const QWEN36_FULL_ATTENTION_LAYER_RESOURCE_BASELINES: &[&str] = &[
    QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN36_FP8_QKV_RESOURCE_BASELINE,
    QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN36_PAGED_GQA_RESOURCE_BASELINE,
    QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN36_GDN_OUTPUT_RESOURCE_BASELINE,
    QWEN36_MOE_ROUTER_RESOURCE_BASELINE,
    QWEN36_MOE_EXPERTS_RESOURCE_BASELINE,
];
const QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINES: &[&str] = &[
    QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN36_GDN_OUTPUT_RESOURCE_BASELINE,
];
const QWEN35_RESIDENT_MODEL_RESOURCE_BASELINES: &[&str] = &[
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_QKV_RESOURCE_BASELINE,
    QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE,
    QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN35_PAGED_GQA_RESOURCE_BASELINE,
];
const QWEN36_RESIDENT_MODEL_RESOURCE_BASELINES: &[&str] = &[
    QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN36_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN36_GDN_OUTPUT_RESOURCE_BASELINE,
    QWEN36_MOE_ROUTER_RESOURCE_BASELINE,
    QWEN36_MOE_EXPERTS_RESOURCE_BASELINE,
    QWEN36_FP8_QKV_RESOURCE_BASELINE,
    QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN36_PAGED_GQA_RESOURCE_BASELINE,
    QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE,
];
const TEXT_ENDPOINT_RESOURCE_BASELINES: &[&str] = &[
    RESIDUAL_NORM_RESOURCE_BASELINE,
    FP8_LM_HEAD_RESOURCE_BASELINE,
];
/// Qwen 3.5 resident MTP baselines: the resident model set followed by the
/// MTP layer set. The repeated residual-norm entry is deliberate; the
/// concatenation is hashed positionally and is never deduplicated.
const QWEN35_RESIDENT_MTP_RESOURCE_BASELINES: &[&str] = &[
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
    QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
    QWEN35_NVFP4_QKV_RESOURCE_BASELINE,
    QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE,
    QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE,
    QWEN35_GDN_PREPARE_RESOURCE_BASELINE,
    QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE,
    QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE,
    QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
    QWEN35_PAGED_GQA_RESOURCE_BASELINE,
    QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
    QWEN35_MTP_RESOURCE_BASELINE,
];
/// Ordered resource baselines for each canonical `bench-device` suite. The
/// order feeds `TUISKO_GENERATOR_BASELINE_SHA256` and binds the benchmark's
/// measurement identity, so entries are never reordered or deduplicated.
const BENCH_DEVICE_BASELINES: &[(&str, &[&str])] = &[
    (
        "qwen38-flash-next-hyper-connection",
        &[QWEN38_FLASH_NEXT_HYPER_CONNECTION_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-ple",
        &[QWEN38_FLASH_NEXT_PLE_RESOURCE_BASELINE],
    ),
    (
        "qwen35-residual-norm",
        &[QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE],
    ),
    (
        "qwen36-residual-norm",
        &[QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE],
    ),
    (
        "qwen35-nvfp4-swiglu",
        &[QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE],
    ),
    ("qwen35-nvfp4-down", &[QWEN35_NVFP4_DOWN_RESOURCE_BASELINE]),
    ("qwen35-nvfp4-qkv", &[QWEN35_NVFP4_QKV_RESOURCE_BASELINE]),
    ("qwen36-moe-router", &[QWEN36_MOE_ROUTER_RESOURCE_BASELINE]),
    (
        "qwen36-moe-experts",
        &[QWEN36_MOE_EXPERTS_RESOURCE_BASELINE],
    ),
    (
        "qwen36-nvfp4-lm-head",
        &[QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE],
    ),
    ("qwen36-fp8-qkv", &[QWEN36_FP8_QKV_RESOURCE_BASELINE]),
    ("qwen36-gdn-input", &[QWEN36_GDN_INPUT_RESOURCE_BASELINE]),
    ("qwen36-gdn-output", &[QWEN36_GDN_OUTPUT_RESOURCE_BASELINE]),
    (
        "qwen36-attention-output",
        QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINES,
    ),
    (
        "qwen36-gdn-prepare",
        &[QWEN35_GDN_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-gdn-prepare",
        &[QWEN38_FLASH_NEXT_GDN_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-gdn-recurrence",
        &[QWEN38_FLASH_NEXT_GDN_RECURRENCE_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-qsa-prepare",
        &[QWEN38_FLASH_NEXT_QSA_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-qsa-attention",
        &[QWEN38_FLASH_NEXT_QSA_ATTENTION_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-qsa-selection",
        &[QWEN38_FLASH_NEXT_QSA_SELECTION_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-moe-router",
        &[QWEN38_FLASH_NEXT_MOE_ROUTER_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-moe-experts",
        &[QWEN38_FLASH_NEXT_MOE_EXPERTS_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-projections",
        &[QWEN38_FLASH_NEXT_PROJECTION_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-lm-head",
        &[QWEN38_FLASH_NEXT_LM_HEAD_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-gdn-layer",
        &[QWEN38_FLASH_NEXT_GDN_RECURRENCE_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-qsa-layer",
        &[QWEN38_FLASH_NEXT_QSA_ATTENTION_RESOURCE_BASELINE],
    ),
    (
        "qwen38-flash-next-ple-layer",
        &[QWEN38_FLASH_NEXT_PLE_RESOURCE_BASELINE],
    ),
    (
        "qwen36-gdn-recurrence",
        &[QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE],
    ),
    (
        "qwen35-nvfp4-gdn-input",
        &[QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE],
    ),
    (
        "qwen35-gdn-prepare",
        &[QWEN35_GDN_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen35-gdn-recurrence",
        &[QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE],
    ),
    (
        "qwen35-nvfp4-gdn-output",
        &[QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE],
    ),
    (
        "qwen35-nvfp4-attention-output",
        &[QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE],
    ),
    (
        "qwen35-nvfp4-mlp",
        &[
            QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
            QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE,
            QWEN35_NVFP4_DOWN_RESOURCE_BASELINE,
        ],
    ),
    (
        "qwen35-attention-qk-prepare",
        &[QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen36-attention-qk-prepare",
        &[QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "qwen36-fp8-attention-qk-prepare",
        &[QWEN36_FP8_ATTENTION_QK_PREPARE_RESOURCE_BASELINE],
    ),
    (
        "nvfp4-mlp",
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            NVFP4_SWIGLU_RESOURCE_BASELINE,
            NVFP4_DOWN_RESOURCE_BASELINE,
        ],
    ),
    ("qwen35-paged-gqa", &[QWEN35_PAGED_GQA_RESOURCE_BASELINE]),
    ("qwen36-paged-gqa", &[QWEN36_PAGED_GQA_RESOURCE_BASELINE]),
    (
        "qwen36-fp8-paged-gqa",
        &[QWEN36_FP8_PAGED_GQA_RESOURCE_BASELINE],
    ),
    (
        "dense-fp8-mlp",
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_SWIGLU_RESOURCE_BASELINE,
            FP8_DOWN_RESOURCE_BASELINE,
        ],
    ),
    (
        "dense-fp8-gdn-layer",
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_GDN_INPUT_RESOURCE_BASELINE,
            GDN_PREPARE_RESOURCE_BASELINE,
            GDN_RECURRENCE_RESOURCE_BASELINE,
            GDN_OUTPUT_RESOURCE_BASELINE,
            FP8_SWIGLU_RESOURCE_BASELINE,
            FP8_DOWN_RESOURCE_BASELINE,
        ],
    ),
    (
        "full-attention-layer",
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_QKV_RESOURCE_BASELINE,
            ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
            PAGED_GQA_RESOURCE_BASELINE,
            ATTENTION_OUTPUT_RESOURCE_BASELINE,
            FP8_SWIGLU_RESOURCE_BASELINE,
            FP8_DOWN_RESOURCE_BASELINE,
        ],
    ),
    ("mtp-layer", MTP_LAYER_RESOURCE_BASELINES),
    ("qwen35-mtp-layer", QWEN35_MTP_LAYER_RESOURCE_BASELINES),
    ("qwen36-mtp-layer", &[QWEN36_MTP_RESOURCE_BASELINE]),
    (
        "qwen35-resident-mtp",
        QWEN35_RESIDENT_MTP_RESOURCE_BASELINES,
    ),
    (
        "qwen35-mtp-generation",
        QWEN35_RESIDENT_MTP_RESOURCE_BASELINES,
    ),
    (
        "qwen35-mtp-batch-generation",
        QWEN35_RESIDENT_MTP_RESOURCE_BASELINES,
    ),
    ("target-mtp-verify", RESIDENT_MODEL_RESOURCE_BASELINES),
    ("mtp-prompt-prime", MTP_PROMPT_PRIME_RESOURCE_BASELINES),
    ("resident-mtp", RESIDENT_MTP_RESOURCE_BASELINES),
    ("generation-mtp-greedy", RESIDENT_MTP_RESOURCE_BASELINES),
    ("generation-mtp-sampling", RESIDENT_MTP_RESOURCE_BASELINES),
    ("generation-mtp-batch", RESIDENT_MTP_RESOURCE_BASELINES),
    (
        "qwen35-full-attention-layer",
        QWEN35_FULL_ATTENTION_LAYER_RESOURCE_BASELINES,
    ),
    ("qwen35-gdn-layer", QWEN35_GDN_LAYER_RESOURCE_BASELINES),
    (
        "qwen36-gdn-moe-layer",
        QWEN36_GDN_MOE_LAYER_RESOURCE_BASELINES,
    ),
    (
        "qwen36-full-attention-layer",
        QWEN36_FULL_ATTENTION_LAYER_RESOURCE_BASELINES,
    ),
    ("resident-model", RESIDENT_MODEL_RESOURCE_BASELINES),
    ("resident-prefill", RESIDENT_MODEL_RESOURCE_BASELINES),
    (
        "resident-long-context-model",
        RESIDENT_MODEL_RESOURCE_BASELINES,
    ),
    (
        "text-endpoint",
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_LM_HEAD_RESOURCE_BASELINE,
        ],
    ),
    (
        "qwen35-text-endpoint",
        &[
            QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE,
            QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE,
        ],
    ),
    (
        "qwen36-text-endpoint",
        &[
            QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE,
            QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE,
        ],
    ),
    (
        "qwen35-resident-model",
        QWEN35_RESIDENT_MODEL_RESOURCE_BASELINES,
    ),
    (
        "qwen36-resident-model",
        QWEN36_RESIDENT_MODEL_RESOURCE_BASELINES,
    ),
];
/// Device-codegen crates whose PTX modules make up the SM120 artifact.
///
/// The kernel families are separate crates so an edit re-runs cuda-oxide
/// device codegen only for the family it touched. cargo-oxide accepts the
/// owners as one comma-separated list and emits one PTX module per crate.
pub(crate) const SM120_DEVICE_CODEGEN_CRATES: &str = "tuisko-kernels-sm120-attention,tuisko-kernels-sm120-engram,tuisko-kernels-sm120-fp8-mlp,tuisko-kernels-sm120-fp8-projection,tuisko-kernels-sm120-gdn,tuisko-kernels-sm120-hyper-connection,tuisko-kernels-sm120-lm-head,tuisko-kernels-sm120-moe,tuisko-kernels-sm120-mtp,tuisko-kernels-sm120-norm,tuisko-kernels-sm120-nvfp4,tuisko-kernels-sm120-qwen38-flash-next-projection";
/// Every module the SM120 device build emits, in `SM120_DEVICE_CODEGEN_CRATES`
/// order. The resource gates read the concatenation: entry names are unique
/// across the whole artifact, and every module is compiled on its own so the
/// reported shared-memory footprint is the family's alone.
const SM120_PTX_MODULES: [&str; 12] = [
    "target/cuda/tuisko_kernels_sm120_attention.ptx",
    "target/cuda/tuisko_kernels_sm120_engram.ptx",
    "target/cuda/tuisko_kernels_sm120_fp8_mlp.ptx",
    "target/cuda/tuisko_kernels_sm120_fp8_projection.ptx",
    "target/cuda/tuisko_kernels_sm120_gdn.ptx",
    "target/cuda/tuisko_kernels_sm120_hyper_connection.ptx",
    "target/cuda/tuisko_kernels_sm120_lm_head.ptx",
    "target/cuda/tuisko_kernels_sm120_moe.ptx",
    "target/cuda/tuisko_kernels_sm120_mtp.ptx",
    "target/cuda/tuisko_kernels_sm120_norm.ptx",
    "target/cuda/tuisko_kernels_sm120_nvfp4.ptx",
    "target/cuda/tuisko_kernels_sm120_qwen38_flash_next_projection.ptx",
];
const CUDA_OXIDE_BUILD_TARGET: &str = "target/cuda-oxide-build-sm120";
const CUDA_OXIDE_TEST_TARGET: &str = "target/cuda-oxide-test";
/// Trailing `tuisko-qual` harness flags: ignored device tests, unbuffered
/// output, and one test thread so sequential CUDA contexts never race.
const QUALIFICATION_IGNORED_SERIAL_FLAGS: &[&str] =
    &["--include-ignored", "--nocapture", "--test-threads=1"];
/// Trailing harness flags for ignored device tests at the harness default
/// thread count.
const QUALIFICATION_IGNORED_FLAGS: &[&str] = &["--include-ignored", "--nocapture"];
/// Trailing harness flags for suites whose tests are not `#[ignore]`d.
const QUALIFICATION_NOCAPTURE_FLAGS: &[&str] = &["--nocapture"];
const CUDA_OXIDE_REPOSITORY: &str = "https://github.com/blaind/cuda-oxide.git";
const CUDA_OXIDE_REVISION: &str = "0199e55572ee78cd2cea97335e5b7392a3f9be4a";
const MAX_IDLE_DEVICE_MEMORY_MIB: u64 = 2_048;
const IDLE_DEVICE_UTILIZATION_LIMIT_PERCENT: u32 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PerformanceSuite {
    ResidualNorm,
    Fp8Qkv,
    Fp8GdnInput,
    Fp8LmHead,
    Fp8SwiGlu,
    Fp8Down,
    Nvfp4SwiGlu,
    Nvfp4Down,
    GdnPrepare,
    GdnRecurrence,
    GdnOutput,
    AttentionQkPrepare,
    PagedGqa,
    LongContextPagedGqa,
    AttentionOutput,
    MtpBf16Fusion,
    MtpBf16AttentionOutput,
    MtpBf16Mlp,
    MtpBf16Qkv,
    MtpBf16QkPrepare,
    MtpBf16PagedGqa,
}

const PERFORMANCE_SUITES: [PerformanceSuite; 21] = [
    PerformanceSuite::ResidualNorm,
    PerformanceSuite::Fp8Qkv,
    PerformanceSuite::Fp8GdnInput,
    PerformanceSuite::Fp8LmHead,
    PerformanceSuite::Fp8SwiGlu,
    PerformanceSuite::Fp8Down,
    PerformanceSuite::Nvfp4SwiGlu,
    PerformanceSuite::Nvfp4Down,
    PerformanceSuite::GdnPrepare,
    PerformanceSuite::GdnRecurrence,
    PerformanceSuite::GdnOutput,
    PerformanceSuite::AttentionQkPrepare,
    PerformanceSuite::PagedGqa,
    PerformanceSuite::LongContextPagedGqa,
    PerformanceSuite::AttentionOutput,
    PerformanceSuite::MtpBf16Fusion,
    PerformanceSuite::MtpBf16Qkv,
    PerformanceSuite::MtpBf16QkPrepare,
    PerformanceSuite::MtpBf16PagedGqa,
    PerformanceSuite::MtpBf16AttentionOutput,
    PerformanceSuite::MtpBf16Mlp,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OptimizationSuite {
    Leaf(PerformanceSuite),
    Nvfp4Mlp,
    DenseFp8Mlp,
    DenseFp8GdnLayer,
    FullAttentionLayer,
    MtpLayer,
    TextEndpoint,
    ResidentModel,
    ResidentPrefill,
    ResidentLongContextModel,
}

const COMPOSED_PERFORMANCE_SUITES: [OptimizationSuite; 9] = [
    OptimizationSuite::Nvfp4Mlp,
    OptimizationSuite::DenseFp8Mlp,
    OptimizationSuite::DenseFp8GdnLayer,
    OptimizationSuite::FullAttentionLayer,
    OptimizationSuite::MtpLayer,
    OptimizationSuite::TextEndpoint,
    OptimizationSuite::ResidentModel,
    OptimizationSuite::ResidentPrefill,
    OptimizationSuite::ResidentLongContextModel,
];

impl PerformanceSuite {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::ResidualNorm => "residual-norm",
            Self::Fp8Qkv => "fp8-qkv",
            Self::Fp8GdnInput => "fp8-gdn-input",
            Self::Fp8LmHead => "fp8-lm-head",
            Self::Fp8SwiGlu => "fp8-swiglu",
            Self::Fp8Down => "fp8-down",
            Self::Nvfp4SwiGlu => "nvfp4-swiglu",
            Self::Nvfp4Down => "nvfp4-down",
            Self::GdnPrepare => "gdn-prepare",
            Self::GdnRecurrence => "gdn-recurrence",
            Self::GdnOutput => "gdn-output",
            Self::AttentionQkPrepare => "attention-qk-prepare",
            Self::PagedGqa => "paged-gqa",
            Self::LongContextPagedGqa => "long-context-paged-gqa",
            Self::AttentionOutput => "attention-output",
            Self::MtpBf16Fusion => "mtp-bf16-fusion",
            Self::MtpBf16AttentionOutput => "mtp-bf16-attention-output",
            Self::MtpBf16Mlp => "mtp-bf16-mlp",
            Self::MtpBf16Qkv => "mtp-bf16-qkv",
            Self::MtpBf16QkPrepare => "mtp-bf16-qk-prepare",
            Self::MtpBf16PagedGqa => "mtp-bf16-paged-gqa",
        }
    }

    const fn resource_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => RESIDUAL_NORM_RESOURCE_BASELINE,
            Self::Fp8Qkv => FP8_QKV_RESOURCE_BASELINE,
            Self::Fp8GdnInput => FP8_GDN_INPUT_RESOURCE_BASELINE,
            Self::Fp8LmHead => FP8_LM_HEAD_RESOURCE_BASELINE,
            Self::Fp8SwiGlu => FP8_SWIGLU_RESOURCE_BASELINE,
            Self::Fp8Down => FP8_DOWN_RESOURCE_BASELINE,
            Self::Nvfp4SwiGlu => NVFP4_SWIGLU_RESOURCE_BASELINE,
            Self::Nvfp4Down => NVFP4_DOWN_RESOURCE_BASELINE,
            Self::GdnPrepare => GDN_PREPARE_RESOURCE_BASELINE,
            Self::GdnRecurrence => GDN_RECURRENCE_RESOURCE_BASELINE,
            Self::GdnOutput => GDN_OUTPUT_RESOURCE_BASELINE,
            Self::AttentionQkPrepare => ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
            Self::PagedGqa => PAGED_GQA_RESOURCE_BASELINE,
            Self::LongContextPagedGqa => LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE,
            Self::AttentionOutput => ATTENTION_OUTPUT_RESOURCE_BASELINE,
            Self::MtpBf16Fusion => MTP_BF16_FUSION_RESOURCE_BASELINE,
            Self::MtpBf16AttentionOutput => MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE,
            Self::MtpBf16Mlp => MTP_BF16_MLP_RESOURCE_BASELINE,
            Self::MtpBf16Qkv => MTP_BF16_QKV_RESOURCE_BASELINE,
            Self::MtpBf16QkPrepare => MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
            Self::MtpBf16PagedGqa => MTP_BF16_PAGED_GQA_RESOURCE_BASELINE,
        }
    }

    const fn performance_baseline(self) -> &'static str {
        match self {
            Self::ResidualNorm => "qual/baselines/residual-norm-sm120.json",
            Self::Fp8Qkv => "qual/baselines/fp8-qkv-sm120.json",
            Self::Fp8GdnInput => "qual/baselines/fp8-gdn-input-sm120.json",
            Self::Fp8LmHead => "qual/baselines/fp8-lm-head-sm120.json",
            Self::Fp8SwiGlu => "qual/baselines/fp8-swiglu-sm120.json",
            Self::Fp8Down => "qual/baselines/fp8-down-sm120.json",
            Self::Nvfp4SwiGlu => "qual/baselines/nvfp4-swiglu-sm120.json",
            Self::Nvfp4Down => "qual/baselines/nvfp4-down-sm120.json",
            Self::GdnPrepare => "qual/baselines/gdn-prepare-sm120.json",
            Self::GdnRecurrence => "qual/baselines/gdn-recurrence-sm120.json",
            Self::GdnOutput => "qual/baselines/gdn-output-sm120.json",
            Self::AttentionQkPrepare => "qual/baselines/attention-qk-prepare-sm120.json",
            Self::PagedGqa => "qual/baselines/paged-gqa-sm120.json",
            Self::LongContextPagedGqa => "qual/baselines/long-context-paged-gqa-sm120.json",
            Self::AttentionOutput => "qual/baselines/attention-output-sm120.json",
            Self::MtpBf16Fusion => "qual/baselines/mtp-bf16-fusion-sm120.json",
            Self::MtpBf16AttentionOutput => "qual/baselines/mtp-bf16-attention-output-sm120.json",
            Self::MtpBf16Mlp => "qual/baselines/mtp-bf16-mlp-sm120.json",
            Self::MtpBf16Qkv => "qual/baselines/mtp-bf16-qkv-sm120.json",
            Self::MtpBf16QkPrepare => "qual/baselines/mtp-bf16-qk-prepare-sm120.json",
            Self::MtpBf16PagedGqa => "qual/baselines/mtp-bf16-paged-gqa-sm120.json",
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "residual-norm" => Ok(Self::ResidualNorm),
            "fp8-qkv" => Ok(Self::Fp8Qkv),
            "fp8-gdn-input" => Ok(Self::Fp8GdnInput),
            "fp8-lm-head" => Ok(Self::Fp8LmHead),
            "fp8-swiglu" => Ok(Self::Fp8SwiGlu),
            "fp8-down" => Ok(Self::Fp8Down),
            "nvfp4-swiglu" => Ok(Self::Nvfp4SwiGlu),
            "nvfp4-down" => Ok(Self::Nvfp4Down),
            "gdn-prepare" => Ok(Self::GdnPrepare),
            "gdn-recurrence" => Ok(Self::GdnRecurrence),
            "gdn-output" => Ok(Self::GdnOutput),
            "attention-qk-prepare" => Ok(Self::AttentionQkPrepare),
            "paged-gqa" => Ok(Self::PagedGqa),
            "long-context-paged-gqa" => Ok(Self::LongContextPagedGqa),
            "attention-output" => Ok(Self::AttentionOutput),
            "mtp-bf16-fusion" => Ok(Self::MtpBf16Fusion),
            "mtp-bf16-attention-output" => Ok(Self::MtpBf16AttentionOutput),
            "mtp-bf16-mlp" => Ok(Self::MtpBf16Mlp),
            "mtp-bf16-qkv" => Ok(Self::MtpBf16Qkv),
            "mtp-bf16-qk-prepare" => Ok(Self::MtpBf16QkPrepare),
            "mtp-bf16-paged-gqa" => Ok(Self::MtpBf16PagedGqa),
            _ => Err(format!("unknown performance suite `{value}`").into()),
        }
    }

    const fn requires_snapshot(self) -> bool {
        matches!(
            self,
            Self::MtpBf16Fusion
                | Self::MtpBf16AttentionOutput
                | Self::MtpBf16Mlp
                | Self::MtpBf16Qkv
                | Self::MtpBf16QkPrepare
        )
    }

    fn qualify(self, root: &Path, snapshot: Option<&OsStr>) -> Result<(), Box<dyn Error>> {
        match self {
            Self::ResidualNorm => qualify_residual_norm(root),
            Self::Fp8Qkv => qualify_fp8_qkv(root),
            Self::Fp8GdnInput => qualify_fp8_gdn_input(root),
            Self::Fp8LmHead => qualify_fp8_lm_head(root),
            Self::Fp8SwiGlu => qualify_fp8_swiglu(root),
            Self::Fp8Down => qualify_fp8_down(root),
            Self::Nvfp4SwiGlu => qualify_nvfp4_swiglu(root),
            Self::Nvfp4Down => qualify_nvfp4_down(root),
            Self::GdnPrepare => qualify_gdn_prepare(root),
            Self::GdnRecurrence => qualify_gdn_recurrence(root),
            Self::GdnOutput => qualify_gdn_output(root),
            Self::AttentionQkPrepare => qualify_attention_qk_prepare(root),
            Self::PagedGqa => qualify_paged_gqa(root),
            Self::LongContextPagedGqa => qualify_long_context_paged_gqa(root),
            Self::AttentionOutput => qualify_attention_output(root),
            Self::MtpBf16Fusion => qualify_mtp_bf16_fusion(
                root,
                &[snapshot
                    .ok_or("mtp-bf16-fusion qualification requires the admitted snapshot path")?
                    .to_os_string()],
            ),
            Self::MtpBf16AttentionOutput => qualify_mtp_bf16_attention_output(
                root,
                &[snapshot
                    .ok_or(
                        "mtp-bf16-attention-output qualification requires the admitted snapshot path",
                    )?
                    .to_os_string()],
            ),
            Self::MtpBf16Mlp => qualify_mtp_bf16_mlp(
                root,
                &[snapshot
                    .ok_or("mtp-bf16-mlp qualification requires the admitted snapshot path")?
                    .to_os_string()],
            ),
            Self::MtpBf16Qkv => qualify_mtp_bf16_qkv(
                root,
                &[snapshot
                    .ok_or("mtp-bf16-qkv qualification requires the admitted snapshot path")?
                    .to_os_string()],
            ),
            Self::MtpBf16QkPrepare => qualify_mtp_bf16_qk_prepare(
                root,
                &[snapshot
                    .ok_or("mtp-bf16-qk-prepare qualification requires the admitted snapshot path")?
                    .to_os_string()],
            ),
            Self::MtpBf16PagedGqa => qualify_mtp_bf16_paged_gqa(root),
        }
    }
}

impl OptimizationSuite {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        if let Ok(suite) = PerformanceSuite::parse(value) {
            return Ok(Self::Leaf(suite));
        }
        COMPOSED_PERFORMANCE_SUITES
            .iter()
            .copied()
            .find(|suite| suite.name() == value)
            .ok_or_else(|| format!("unknown optimization suite `{value}`").into())
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Leaf(suite) => suite.name(),
            Self::Nvfp4Mlp => "nvfp4-mlp",
            Self::DenseFp8Mlp => "dense-fp8-mlp",
            Self::DenseFp8GdnLayer => "dense-fp8-gdn-layer",
            Self::FullAttentionLayer => "full-attention-layer",
            Self::MtpLayer => "mtp-layer",
            Self::TextEndpoint => "text-endpoint",
            Self::ResidentModel => "resident-model",
            Self::ResidentPrefill => "resident-prefill",
            Self::ResidentLongContextModel => "resident-long-context-model",
        }
    }

    const fn requires_snapshot(self) -> bool {
        match self {
            Self::Leaf(suite) => suite.requires_snapshot(),
            _ => true,
        }
    }

    fn resource_baselines(self) -> Vec<&'static str> {
        match self {
            Self::Leaf(suite) => vec![suite.resource_baseline()],
            Self::Nvfp4Mlp => NVFP4_MLP_RESOURCE_BASELINES.to_vec(),
            Self::DenseFp8Mlp => DENSE_FP8_MLP_RESOURCE_BASELINES.to_vec(),
            Self::DenseFp8GdnLayer => DENSE_FP8_GDN_LAYER_RESOURCE_BASELINES.to_vec(),
            Self::FullAttentionLayer => FULL_ATTENTION_LAYER_RESOURCE_BASELINES.to_vec(),
            Self::MtpLayer => MTP_LAYER_RESOURCE_BASELINES.to_vec(),
            Self::TextEndpoint => TEXT_ENDPOINT_RESOURCE_BASELINES.to_vec(),
            Self::ResidentModel | Self::ResidentPrefill | Self::ResidentLongContextModel => {
                RESIDENT_MODEL_RESOURCE_BASELINES.to_vec()
            }
        }
    }

    const fn performance_baseline(self) -> &'static str {
        match self {
            Self::Leaf(suite) => suite.performance_baseline(),
            Self::Nvfp4Mlp => "qual/baselines/nvfp4-mlp-sm120.json",
            Self::DenseFp8Mlp => "qual/baselines/dense-fp8-mlp-sm120.json",
            Self::DenseFp8GdnLayer => "qual/baselines/dense-fp8-gdn-layer-sm120.json",
            Self::FullAttentionLayer => "qual/baselines/full-attention-layer-sm120.json",
            Self::MtpLayer => "qual/baselines/mtp-layer-sm120.json",
            Self::TextEndpoint => "qual/baselines/text-endpoint-sm120.json",
            Self::ResidentModel => "qual/baselines/resident-model-sm120.json",
            Self::ResidentPrefill => "qual/baselines/resident-prefill-sm120.json",
            Self::ResidentLongContextModel => {
                "qual/baselines/resident-long-context-model-sm120.json"
            }
        }
    }

    fn qualify(self, root: &Path, snapshot: Option<&OsStr>) -> Result<(), Box<dyn Error>> {
        let snapshot_arguments = || -> Result<[std::ffi::OsString; 1], Box<dyn Error>> {
            Ok([snapshot
                .ok_or_else(|| format!("{} requires the admitted snapshot path", self.name()))?
                .to_os_string()])
        };
        match self {
            Self::Leaf(suite) => suite.qualify(root, snapshot),
            Self::Nvfp4Mlp => qualify_nvfp4_mlp(root, &snapshot_arguments()?),
            Self::DenseFp8Mlp => qualify_dense_fp8_mlp(root, &snapshot_arguments()?),
            Self::DenseFp8GdnLayer => qualify_dense_fp8_gdn_layer(root, &snapshot_arguments()?),
            Self::FullAttentionLayer => qualify_full_attention_layer(root, &snapshot_arguments()?),
            Self::MtpLayer => qualify_mtp_layer(root, &snapshot_arguments()?),
            Self::TextEndpoint => qualify_text_endpoint(root, &snapshot_arguments()?),
            Self::ResidentModel | Self::ResidentPrefill | Self::ResidentLongContextModel => {
                qualify_resident_model(root, &snapshot_arguments()?)
            }
        }
    }

    fn dependency_cone(self) -> Vec<Self> {
        use OptimizationSuite::{
            DenseFp8GdnLayer, DenseFp8Mlp, FullAttentionLayer, MtpLayer, Nvfp4Mlp,
            ResidentLongContextModel, ResidentModel, ResidentPrefill, TextEndpoint,
        };
        use PerformanceSuite::{
            AttentionOutput, AttentionQkPrepare, Fp8Down, Fp8GdnInput, Fp8LmHead, Fp8Qkv,
            Fp8SwiGlu, GdnOutput, GdnPrepare, GdnRecurrence, LongContextPagedGqa,
            MtpBf16AttentionOutput, MtpBf16Fusion, MtpBf16Mlp, MtpBf16PagedGqa, MtpBf16QkPrepare,
            MtpBf16Qkv, Nvfp4Down, Nvfp4SwiGlu, PagedGqa, ResidualNorm,
        };

        let downstream = match self {
            Self::MtpLayer
            | Self::ResidentModel
            | Self::ResidentPrefill
            | Self::ResidentLongContextModel => &[][..],
            Self::Leaf(LongContextPagedGqa) => &[ResidentLongContextModel],
            Self::Leaf(
                MtpBf16Fusion
                | MtpBf16Qkv
                | MtpBf16QkPrepare
                | MtpBf16PagedGqa
                | MtpBf16AttentionOutput
                | MtpBf16Mlp,
            ) => &[MtpLayer],
            Self::Leaf(Nvfp4SwiGlu | Nvfp4Down) | Self::Nvfp4Mlp => &[
                Nvfp4Mlp,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::Leaf(Fp8LmHead) => &[
                MtpLayer,
                TextEndpoint,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::TextEndpoint => &[
                TextEndpoint,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::Leaf(Fp8GdnInput | GdnPrepare | GdnRecurrence | GdnOutput)
            | Self::DenseFp8GdnLayer => &[
                DenseFp8GdnLayer,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::Leaf(Fp8Qkv | AttentionQkPrepare | PagedGqa | AttentionOutput)
            | Self::FullAttentionLayer => &[
                FullAttentionLayer,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::Leaf(Fp8SwiGlu | Fp8Down) | Self::DenseFp8Mlp => &[
                DenseFp8Mlp,
                DenseFp8GdnLayer,
                FullAttentionLayer,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
            Self::Leaf(ResidualNorm) => &[
                Nvfp4Mlp,
                DenseFp8Mlp,
                DenseFp8GdnLayer,
                FullAttentionLayer,
                MtpLayer,
                TextEndpoint,
                ResidentModel,
                ResidentPrefill,
                ResidentLongContextModel,
            ],
        };
        let mut cone = vec![self];
        for suite in downstream {
            if !cone.contains(suite) {
                cone.push(*suite);
            }
        }

        cone
    }
}

type NoArgumentHandler = fn(&Path) -> Result<(), Box<dyn Error>>;
type ForwardingHandler = fn(&Path, &[std::ffi::OsString]) -> Result<(), Box<dyn Error>>;

/// How a subcommand consumes what follows its name. The policy is table data,
/// so `dispatch` enforces it in one place and no row can omit the check.
#[derive(Clone, Copy)]
enum Handler {
    NoArguments(NoArgumentHandler),
    Forwarded(ForwardingHandler),
}

/// One CLI subcommand: the name `main` matches and the handler it reaches.
struct Subcommand {
    name: &'static str,
    run: Handler,
}

const fn no_args(name: &'static str, run: NoArgumentHandler) -> Subcommand {
    Subcommand {
        name,
        run: Handler::NoArguments(run),
    }
}

const fn forwarded(name: &'static str, run: ForwardingHandler) -> Subcommand {
    Subcommand {
        name,
        run: Handler::Forwarded(run),
    }
}

/// The complete subcommand set in stable dispatch order.
const SUBCOMMANDS: &[Subcommand] = &[
    no_args("bootstrap-cuda-oxide", bootstrap_cuda_oxide),
    no_args("build-sm120", build_sm120),
    forwarded("build-residual-norm", build_residual_norm),
    forwarded("build-residual-bench", build_residual_bench),
    no_args("build-server", build_server),
    forwarded("qualify-frontend", qualify_frontend),
    forwarded("qualify-generation", qualify_generation),
    forwarded("qualify-server", server_qual::run),
    forwarded("qualify-server-long-context", server_qual::run_long_context),
    no_args("qualify-host", qualify_host),
    no_args(
        "qualify-qwen38-flash-next-hyper-connection",
        qualify_qwen38_flash_next_hyper_connection,
    ),
    no_args(
        "qualify-qwen38-flash-next-ple",
        qualify_qwen38_flash_next_ple,
    ),
    no_args("qualify-residual-norm", qualify_residual_norm),
    no_args("qualify-qwen35-residual-norm", qualify_qwen35_residual_norm),
    no_args("qualify-qwen36-residual-norm", qualify_qwen36_residual_norm),
    no_args("qualify-qwen35-nvfp4-swiglu", qualify_qwen35_nvfp4_swiglu),
    no_args("qualify-qwen35-nvfp4-down", qualify_qwen35_nvfp4_down),
    no_args("qualify-qwen35-nvfp4-qkv", qualify_qwen35_nvfp4_qkv),
    no_args("qualify-qwen36-moe-router", qualify_qwen36_moe_router),
    no_args("qualify-qwen36-moe-experts", qualify_qwen36_moe_experts),
    no_args("qualify-qwen36-nvfp4-lm-head", qualify_qwen36_nvfp4_lm_head),
    no_args("qualify-qwen36-fp8-qkv", qualify_qwen36_fp8_qkv),
    no_args("qualify-qwen36-gdn-input", qualify_qwen36_gdn_input),
    no_args("qualify-qwen36-gdn-output", qualify_qwen36_gdn_output),
    no_args(
        "qualify-qwen36-attention-output",
        qualify_qwen36_attention_output,
    ),
    no_args("qualify-qwen36-gdn-prepare", qualify_qwen36_gdn_prepare),
    no_args(
        "qualify-qwen36-gdn-recurrence",
        qualify_qwen36_gdn_recurrence,
    ),
    forwarded("qualify-qwen35-bf16-lm-head", qualify_qwen35_bf16_lm_head),
    forwarded(
        "qualify-qwen35-mtp-bf16-fusion",
        qualify_qwen35_mtp_bf16_fusion,
    ),
    forwarded(
        "qualify-qwen35-mtp-bf16-attention",
        qualify_qwen35_mtp_bf16_attention,
    ),
    forwarded(
        "qualify-qwen36-mtp-bf16-attention",
        qualify_qwen36_mtp_bf16_attention,
    ),
    no_args("qualify-qwen36-mtp-bf16-moe", qualify_qwen36_mtp_bf16_moe),
    forwarded("qualify-qwen35-mtp-bf16-mlp", qualify_qwen35_mtp_bf16_mlp),
    forwarded("qualify-qwen35-text-endpoint", qualify_qwen35_text_endpoint),
    forwarded("qualify-qwen36-text-endpoint", qualify_qwen36_text_endpoint),
    forwarded(
        "qualify-qwen36-resident-model",
        qualify_qwen36_resident_model,
    ),
    forwarded("qualify-qwen36-generation", qualify_qwen36_generation),
    forwarded("qualify-qwen36-server", qualify_qwen36_server),
    forwarded(
        "qualify-qwen38-flash-next-server",
        qwen38_flash_next_server_qual::run,
    ),
    forwarded("qualify-qwen35-server", qualify_qwen35_server),
    forwarded(
        "qualify-qwen35-resident-model",
        qualify_qwen35_resident_model,
    ),
    forwarded("qualify-qwen35-resident-mtp", qualify_qwen35_resident_mtp),
    forwarded(
        "qualify-qwen35-mtp-generation",
        qualify_qwen35_mtp_generation,
    ),
    forwarded(
        "qualify-qwen35-mtp-batch-generation",
        qualify_qwen35_mtp_batch_generation,
    ),
    forwarded("qualify-qwen35-generation", qualify_qwen35_generation),
    no_args(
        "qualify-qwen35-long-context-kv",
        qualify_qwen35_long_context_kv,
    ),
    no_args(
        "qualify-qwen36-long-context-kv",
        qualify_qwen36_long_context_kv,
    ),
    no_args("qualify-streaming-pool", qualify_streaming_pool),
    no_args(
        "qualify-qwen38-flash-next-engram-staging",
        qualify_qwen38_flash_next_engram_staging,
    ),
    no_args(
        "qualify-qwen35-nvfp4-gdn-input",
        qualify_qwen35_nvfp4_gdn_input,
    ),
    no_args("qualify-qwen35-gdn-prepare", qualify_qwen35_gdn_prepare),
    no_args(
        "qualify-qwen35-gdn-recurrence",
        qualify_qwen35_gdn_recurrence,
    ),
    no_args(
        "qualify-qwen35-nvfp4-gdn-output",
        qualify_qwen35_nvfp4_gdn_output,
    ),
    no_args(
        "qualify-qwen35-nvfp4-attention-output",
        qualify_qwen35_nvfp4_attention_output,
    ),
    forwarded("qualify-qwen35-nvfp4-mlp", qualify_qwen35_nvfp4_mlp),
    no_args(
        "qualify-qwen35-attention-qk-prepare",
        qualify_qwen35_attention_qk_prepare,
    ),
    no_args(
        "qualify-qwen36-attention-qk-prepare",
        qualify_qwen36_attention_qk_prepare,
    ),
    no_args(
        "qualify-qwen36-fp8-attention-qk-prepare",
        qualify_qwen36_fp8_attention_qk_prepare,
    ),
    no_args("qualify-fp8-qkv", qualify_fp8_qkv),
    no_args("qualify-fp8-gdn-input", qualify_fp8_gdn_input),
    no_args("qualify-fp8-lm-head", qualify_fp8_lm_head),
    no_args("qualify-fp8-swiglu", qualify_fp8_swiglu),
    no_args("qualify-fp8-down", qualify_fp8_down),
    no_args("qualify-nvfp4-swiglu", qualify_nvfp4_swiglu),
    no_args("qualify-nvfp4-down", qualify_nvfp4_down),
    forwarded("qualify-nvfp4-mlp", qualify_nvfp4_mlp),
    no_args("qualify-gdn-prepare", qualify_gdn_prepare),
    no_args(
        "qualify-qwen38-flash-next-gdn-prepare",
        qualify_qwen38_flash_next_gdn_prepare,
    ),
    no_args(
        "qualify-qwen38-flash-next-gdn-recurrence",
        qualify_qwen38_flash_next_gdn_recurrence,
    ),
    no_args(
        "qualify-qwen38-flash-next-qsa-prepare",
        qualify_qwen38_flash_next_qsa_prepare,
    ),
    no_args(
        "qualify-qwen38-flash-next-qsa-attention",
        qualify_qwen38_flash_next_qsa_attention,
    ),
    no_args(
        "qualify-qwen38-flash-next-qsa-selection",
        qualify_qwen38_flash_next_qsa_selection,
    ),
    no_args(
        "qualify-qwen38-flash-next-moe-router",
        qualify_qwen38_flash_next_moe_router,
    ),
    no_args(
        "qualify-qwen38-flash-next-moe-experts",
        qualify_qwen38_flash_next_moe_experts,
    ),
    no_args(
        "qualify-qwen38-flash-next-projections",
        qualify_qwen38_flash_next_projections,
    ),
    no_args(
        "qualify-qwen38-flash-next-lm-head",
        qualify_qwen38_flash_next_lm_head,
    ),
    forwarded(
        "qualify-qwen38-flash-next-gdn-layer",
        qualify_qwen38_flash_next_gdn_layer,
    ),
    forwarded(
        "qualify-qwen38-flash-next-mtp-oracle",
        qualify_qwen38_flash_next_mtp_oracle,
    ),
    forwarded(
        "qualify-qwen38-flash-next-mtp-generation",
        qualify_qwen38_flash_next_mtp_generation,
    ),
    forwarded(
        "qualify-qwen38-flash-next-generation",
        qualify_qwen38_flash_next_generation,
    ),
    forwarded(
        "qualify-qwen38-flash-next-prompt-prime",
        qualify_qwen38_flash_next_prompt_prime,
    ),
    forwarded(
        "qualify-qwen38-flash-next-compact-generation",
        qualify_qwen38_flash_next_compact_generation,
    ),
    forwarded(
        "qualify-qwen38-flash-next-qsa-layer",
        qualify_qwen38_flash_next_qsa_layer,
    ),
    forwarded(
        "qualify-qwen38-flash-next-resident-model",
        qualify_qwen38_flash_next_resident_model,
    ),
    no_args("qualify-gdn-recurrence", qualify_gdn_recurrence),
    no_args("qualify-gdn-output", qualify_gdn_output),
    no_args("qualify-attention-qk-prepare", qualify_attention_qk_prepare),
    no_args("qualify-paged-gqa", qualify_paged_gqa),
    no_args("qualify-qwen35-paged-gqa", qualify_qwen35_paged_gqa),
    no_args("qualify-qwen36-paged-gqa", qualify_qwen36_paged_gqa),
    no_args("qualify-qwen36-fp8-paged-gqa", qualify_qwen36_fp8_paged_gqa),
    no_args(
        "qualify-long-context-paged-gqa",
        qualify_long_context_paged_gqa,
    ),
    no_args("qualify-attention-output", qualify_attention_output),
    forwarded("qualify-mtp-bf16-fusion", qualify_mtp_bf16_fusion),
    forwarded(
        "qualify-mtp-bf16-attention-output",
        qualify_mtp_bf16_attention_output,
    ),
    forwarded("qualify-mtp-bf16-mlp", qualify_mtp_bf16_mlp),
    forwarded("qualify-mtp-bf16-qkv", qualify_mtp_bf16_qkv),
    forwarded("qualify-mtp-bf16-qk-prepare", qualify_mtp_bf16_qk_prepare),
    no_args("qualify-mtp-bf16-paged-gqa", qualify_mtp_bf16_paged_gqa),
    forwarded("qualify-dense-fp8-mlp", qualify_dense_fp8_mlp),
    forwarded("qualify-dense-fp8-gdn-layer", qualify_dense_fp8_gdn_layer),
    forwarded("qualify-full-attention-layer", qualify_full_attention_layer),
    forwarded("qualify-mtp-layer", qualify_mtp_layer),
    forwarded("qualify-qwen35-mtp-layer", qualify_qwen35_mtp_layer),
    forwarded("qualify-qwen36-mtp-layer", qualify_qwen36_mtp_layer),
    forwarded("qualify-target-mtp-verify", qualify_target_mtp_verify),
    forwarded("qualify-mtp-prompt-prime", qualify_mtp_prompt_prime),
    forwarded("qualify-resident-mtp", qualify_resident_mtp),
    forwarded(
        "qualify-generation-mtp-greedy",
        qualify_generation_mtp_greedy,
    ),
    forwarded(
        "qualify-generation-mtp-sampling",
        qualify_generation_mtp_sampling,
    ),
    forwarded("qualify-generation-mtp-batch", qualify_generation_mtp_batch),
    forwarded(
        "qualify-qwen35-full-attention-layer",
        qualify_qwen35_full_attention_layer,
    ),
    forwarded(
        "qualify-qwen36-full-attention-layer",
        qualify_qwen36_full_attention_layer,
    ),
    forwarded("qualify-qwen35-gdn-layer", qualify_qwen35_gdn_layer),
    forwarded("qualify-qwen36-gdn-moe-layer", qualify_qwen36_gdn_moe_layer),
    forwarded("qualify-resident-model", qualify_resident_model),
    forwarded("qualify-resident-generation", qualify_resident_generation),
    forwarded(
        "qualify-resident-batch-generation",
        qualify_resident_batch_generation,
    ),
    forwarded("qualify-text-endpoint", qualify_text_endpoint),
    forwarded("bench-startup", bench_startup),
    forwarded("bench-server", server_bench::run),
    forwarded(
        "bench-qwen38-flash-next-server",
        qwen38_flash_next_server_qual::bench,
    ),
    forwarded(
        "bench-qwen38-flash-next-hyper-connection",
        bench_qwen38_flash_next_hyper_connection,
    ),
    forwarded("bench-qwen38-flash-next-ple", bench_qwen38_flash_next_ple),
    forwarded("bench-residual-norm", bench_residual_norm),
    forwarded("bench-qwen35-residual-norm", bench_qwen35_residual_norm),
    forwarded("bench-qwen36-residual-norm", bench_qwen36_residual_norm),
    forwarded("bench-qwen35-nvfp4-swiglu", bench_qwen35_nvfp4_swiglu),
    forwarded("bench-qwen35-nvfp4-down", bench_qwen35_nvfp4_down),
    forwarded("bench-qwen35-nvfp4-qkv", bench_qwen35_nvfp4_qkv),
    forwarded("bench-qwen36-moe-router", bench_qwen36_moe_router),
    forwarded("bench-qwen36-moe-experts", bench_qwen36_moe_experts),
    forwarded("bench-qwen36-nvfp4-lm-head", bench_qwen36_nvfp4_lm_head),
    forwarded("bench-qwen36-fp8-qkv", bench_qwen36_fp8_qkv),
    forwarded("bench-qwen36-gdn-input", bench_qwen36_gdn_input),
    forwarded("bench-qwen36-gdn-output", bench_qwen36_gdn_output),
    forwarded(
        "bench-qwen36-attention-output",
        bench_qwen36_attention_output,
    ),
    forwarded("bench-qwen36-gdn-prepare", bench_qwen36_gdn_prepare),
    forwarded("bench-qwen36-gdn-recurrence", bench_qwen36_gdn_recurrence),
    forwarded("bench-qwen35-nvfp4-gdn-input", bench_qwen35_nvfp4_gdn_input),
    forwarded("bench-qwen35-gdn-prepare", bench_qwen35_gdn_prepare),
    forwarded("bench-qwen35-gdn-recurrence", bench_qwen35_gdn_recurrence),
    forwarded(
        "bench-qwen35-nvfp4-gdn-output",
        bench_qwen35_nvfp4_gdn_output,
    ),
    forwarded(
        "bench-qwen35-nvfp4-attention-output",
        bench_qwen35_nvfp4_attention_output,
    ),
    forwarded("bench-qwen35-nvfp4-mlp", bench_qwen35_nvfp4_mlp),
    forwarded(
        "bench-qwen35-attention-qk-prepare",
        bench_qwen35_attention_qk_prepare,
    ),
    forwarded(
        "bench-qwen36-attention-qk-prepare",
        bench_qwen36_attention_qk_prepare,
    ),
    forwarded(
        "bench-qwen36-fp8-attention-qk-prepare",
        bench_qwen36_fp8_attention_qk_prepare,
    ),
    forwarded("bench-fp8-qkv", bench_fp8_qkv),
    forwarded("bench-fp8-gdn-input", bench_fp8_gdn_input),
    forwarded("bench-fp8-lm-head", bench_fp8_lm_head),
    forwarded("bench-fp8-swiglu", bench_fp8_swiglu),
    forwarded("bench-fp8-down", bench_fp8_down),
    forwarded("bench-nvfp4-swiglu", bench_nvfp4_swiglu),
    forwarded("bench-nvfp4-down", bench_nvfp4_down),
    forwarded("bench-nvfp4-mlp", bench_nvfp4_mlp),
    forwarded("bench-gdn-prepare", bench_gdn_prepare),
    forwarded(
        "bench-qwen38-flash-next-gdn-prepare",
        bench_qwen38_flash_next_gdn_prepare,
    ),
    forwarded(
        "bench-qwen38-flash-next-gdn-recurrence",
        bench_qwen38_flash_next_gdn_recurrence,
    ),
    forwarded(
        "bench-qwen38-flash-next-qsa-prepare",
        bench_qwen38_flash_next_qsa_prepare,
    ),
    forwarded(
        "bench-qwen38-flash-next-qsa-attention",
        bench_qwen38_flash_next_qsa_attention,
    ),
    forwarded(
        "bench-qwen38-flash-next-qsa-selection",
        bench_qwen38_flash_next_qsa_selection,
    ),
    forwarded(
        "bench-qwen38-flash-next-moe-router",
        bench_qwen38_flash_next_moe_router,
    ),
    forwarded(
        "bench-qwen38-flash-next-moe-experts",
        bench_qwen38_flash_next_moe_experts,
    ),
    forwarded(
        "bench-qwen38-flash-next-projections",
        bench_qwen38_flash_next_projections,
    ),
    forwarded(
        "bench-qwen38-flash-next-lm-head",
        bench_qwen38_flash_next_lm_head,
    ),
    forwarded(
        "bench-qwen38-flash-next-gdn-layer",
        bench_qwen38_flash_next_gdn_layer,
    ),
    forwarded("bench-qwen38-flash-next", bench_qwen38_flash_next),
    forwarded(
        "bench-qwen38-flash-next-generation",
        bench_qwen38_flash_next_generation,
    ),
    forwarded(
        "bench-qwen38-flash-next-prompt-prime",
        bench_qwen38_flash_next_prompt_prime,
    ),
    forwarded(
        "bench-qwen38-flash-next-qsa-layer",
        bench_qwen38_flash_next_qsa_layer,
    ),
    forwarded(
        "bench-qwen38-flash-next-ple-layer",
        bench_qwen38_flash_next_ple_layer,
    ),
    forwarded(
        "bench-qwen38-flash-next-resident-model",
        bench_qwen38_flash_next_resident_model,
    ),
    forwarded("bench-gdn-recurrence", bench_gdn_recurrence),
    forwarded("bench-gdn-output", bench_gdn_output),
    forwarded("bench-attention-qk-prepare", bench_attention_qk_prepare),
    forwarded("bench-paged-gqa", bench_paged_gqa),
    forwarded("bench-qwen35-paged-gqa", bench_qwen35_paged_gqa),
    forwarded("bench-qwen36-paged-gqa", bench_qwen36_paged_gqa),
    forwarded("bench-qwen36-fp8-paged-gqa", bench_qwen36_fp8_paged_gqa),
    forwarded("bench-long-context-paged-gqa", bench_long_context_paged_gqa),
    forwarded("bench-attention-output", bench_attention_output),
    forwarded("bench-mtp-bf16-fusion", bench_mtp_bf16_fusion),
    forwarded(
        "bench-mtp-bf16-attention-output",
        bench_mtp_bf16_attention_output,
    ),
    forwarded("bench-mtp-bf16-mlp", bench_mtp_bf16_mlp),
    forwarded("bench-mtp-bf16-qkv", bench_mtp_bf16_qkv),
    forwarded("bench-mtp-bf16-qk-prepare", bench_mtp_bf16_qk_prepare),
    forwarded("bench-mtp-bf16-paged-gqa", bench_mtp_bf16_paged_gqa),
    forwarded("bench-dense-fp8-mlp", bench_dense_fp8_mlp),
    forwarded("bench-dense-fp8-gdn-layer", bench_dense_fp8_gdn_layer),
    forwarded("bench-full-attention-layer", bench_full_attention_layer),
    forwarded("bench-mtp-layer", bench_mtp_layer),
    forwarded("bench-qwen35-mtp-layer", bench_qwen35_mtp_layer),
    forwarded("bench-qwen36-mtp-layer", bench_qwen36_mtp_layer),
    forwarded("bench-target-mtp-verify", bench_target_mtp_verify),
    forwarded("bench-mtp-prompt-prime", bench_mtp_prompt_prime),
    forwarded("bench-resident-mtp", bench_resident_mtp),
    forwarded("bench-qwen35-resident-mtp", bench_qwen35_resident_mtp),
    forwarded("bench-qwen35-mtp-generation", bench_qwen35_mtp_generation),
    forwarded(
        "bench-qwen35-mtp-batch-generation",
        bench_qwen35_mtp_batch_generation,
    ),
    forwarded("bench-generation-mtp-greedy", bench_generation_mtp_greedy),
    forwarded(
        "bench-generation-mtp-sampling",
        bench_generation_mtp_sampling,
    ),
    forwarded("bench-generation-mtp-batch", bench_generation_mtp_batch),
    forwarded(
        "bench-qwen35-full-attention-layer",
        bench_qwen35_full_attention_layer,
    ),
    forwarded("bench-qwen35-gdn-layer", bench_qwen35_gdn_layer),
    forwarded("bench-qwen36-gdn-moe-layer", bench_qwen36_gdn_moe_layer),
    forwarded(
        "bench-qwen36-full-attention-layer",
        bench_qwen36_full_attention_layer,
    ),
    forwarded("bench-qwen35-text-endpoint", bench_qwen35_text_endpoint),
    forwarded("bench-qwen36-text-endpoint", bench_qwen36_text_endpoint),
    forwarded("bench-qwen35-resident-model", bench_qwen35_resident_model),
    forwarded("bench-qwen36-resident-model", bench_qwen36_resident_model),
    forwarded("bench-resident-model", bench_resident_model),
    forwarded("bench-resident-prefill", bench_resident_prefill),
    forwarded(
        "bench-resident-long-context-model",
        bench_resident_long_context_model,
    ),
    forwarded("bench-text-endpoint", bench_text_endpoint),
    no_args(
        "gate-qwen38-flash-next-hyper-connection",
        gate_qwen38_flash_next_hyper_connection,
    ),
    no_args("gate-qwen38-flash-next-ple", gate_qwen38_flash_next_ple),
    no_args("gate-residual-norm", gate_residual_norm),
    no_args("gate-qwen35-residual-norm", gate_qwen35_residual_norm),
    no_args("gate-qwen36-residual-norm", gate_qwen36_residual_norm),
    no_args("gate-qwen35-nvfp4-swiglu", gate_qwen35_nvfp4_swiglu),
    no_args("gate-qwen35-nvfp4-down", gate_qwen35_nvfp4_down),
    no_args("gate-qwen35-nvfp4-qkv", gate_qwen35_nvfp4_qkv),
    no_args("gate-qwen35-bf16-lm-head", gate_qwen35_bf16_lm_head),
    no_args("gate-qwen36-moe-router", gate_qwen36_moe_router),
    no_args("gate-qwen36-moe-experts", gate_qwen36_moe_experts),
    no_args("gate-qwen36-nvfp4-lm-head", gate_qwen36_nvfp4_lm_head),
    no_args("gate-qwen36-fp8-qkv", gate_qwen36_fp8_qkv),
    no_args("gate-qwen36-gdn-input", gate_qwen36_gdn_input),
    no_args("gate-qwen36-gdn-output", gate_qwen36_gdn_output),
    no_args("gate-qwen35-nvfp4-gdn-input", gate_qwen35_nvfp4_gdn_input),
    no_args("gate-qwen35-gdn-prepare", gate_qwen35_gdn_prepare),
    no_args("gate-qwen35-gdn-recurrence", gate_qwen35_gdn_recurrence),
    no_args(
        "gate-qwen35-nvfp4-attention-output",
        gate_qwen35_nvfp4_attention_output,
    ),
    no_args(
        "gate-qwen35-attention-qk-prepare",
        gate_qwen35_attention_qk_prepare,
    ),
    no_args(
        "gate-qwen36-attention-qk-prepare",
        gate_qwen36_attention_qk_prepare,
    ),
    no_args(
        "gate-qwen36-fp8-attention-qk-prepare",
        gate_qwen36_fp8_attention_qk_prepare,
    ),
    no_args("gate-fp8-qkv", gate_fp8_qkv),
    no_args("gate-fp8-gdn-input", gate_fp8_gdn_input),
    no_args("gate-fp8-lm-head", gate_fp8_lm_head),
    no_args("gate-fp8-swiglu", gate_fp8_swiglu),
    no_args("gate-fp8-down", gate_fp8_down),
    no_args("gate-nvfp4-swiglu", gate_nvfp4_swiglu),
    no_args("gate-nvfp4-down", gate_nvfp4_down),
    no_args("gate-gdn-prepare", gate_gdn_prepare),
    no_args(
        "gate-qwen38-flash-next-gdn-prepare",
        gate_qwen38_flash_next_gdn_prepare,
    ),
    no_args(
        "gate-qwen38-flash-next-gdn-recurrence",
        gate_qwen38_flash_next_gdn_recurrence,
    ),
    no_args(
        "gate-qwen38-flash-next-qsa-prepare",
        gate_qwen38_flash_next_qsa_prepare,
    ),
    no_args(
        "gate-qwen38-flash-next-qsa-attention",
        gate_qwen38_flash_next_qsa_attention,
    ),
    no_args(
        "gate-qwen38-flash-next-qsa-selection",
        gate_qwen38_flash_next_qsa_selection,
    ),
    no_args(
        "gate-qwen38-flash-next-moe-router",
        gate_qwen38_flash_next_moe_router,
    ),
    no_args(
        "gate-qwen38-flash-next-moe-experts",
        gate_qwen38_flash_next_moe_experts,
    ),
    no_args(
        "gate-qwen38-flash-next-projections",
        gate_qwen38_flash_next_projections,
    ),
    no_args(
        "gate-qwen38-flash-next-lm-head",
        gate_qwen38_flash_next_lm_head,
    ),
    no_args("gate-gdn-recurrence", gate_gdn_recurrence),
    no_args("gate-gdn-output", gate_gdn_output),
    no_args("gate-attention-qk-prepare", gate_attention_qk_prepare),
    no_args("gate-paged-gqa", gate_paged_gqa),
    no_args("gate-qwen35-paged-gqa", gate_qwen35_paged_gqa),
    no_args("gate-qwen36-paged-gqa", gate_qwen36_paged_gqa),
    no_args("gate-qwen36-fp8-paged-gqa", gate_qwen36_fp8_paged_gqa),
    no_args("gate-qwen36-attention-output", gate_qwen36_attention_output),
    no_args("gate-long-context-paged-gqa", gate_long_context_paged_gqa),
    no_args("gate-attention-output", gate_attention_output),
    no_args("gate-mtp-bf16-fusion", gate_mtp_bf16_fusion),
    no_args(
        "gate-mtp-bf16-attention-output",
        gate_mtp_bf16_attention_output,
    ),
    no_args("gate-mtp-bf16-mlp", gate_mtp_bf16_mlp),
    no_args("gate-mtp-bf16-qkv", gate_mtp_bf16_qkv),
    no_args("gate-mtp-bf16-qk-prepare", gate_mtp_bf16_qk_prepare),
    no_args("gate-mtp-bf16-paged-gqa", gate_mtp_bf16_paged_gqa),
    no_args("gate-qwen35-mtp-resources", gate_qwen35_mtp_resources),
    no_args("gate-qwen36-mtp-resources", gate_qwen36_mtp_resources),
    forwarded("perf", perf),
    forwarded("profile", profile),
    // Billable: the subcommand is real in every build, but only the feature
    // links the runner. Without it the handler names the flag.
    #[cfg(feature = "remote")]
    forwarded("remote", remote::run),
    #[cfg(not(feature = "remote"))]
    forwarded("remote", remote::unavailable),
];

/// Reject arguments a subcommand does not take.
fn require_no_args(arguments: &[std::ffi::OsString], name: &str) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Ok(());
    }

    Err(format!("`{name}` takes no arguments").into())
}

/// Resolve one command name against `SUBCOMMANDS` and run its handler.
fn dispatch(
    root: &Path,
    command: &OsStr,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let subcommand = command
        .to_str()
        .and_then(|name| {
            SUBCOMMANDS
                .iter()
                .find(|subcommand| subcommand.name == name)
        })
        .ok_or_else(|| format!("unknown xtask command `{}`", command.to_string_lossy()))?;

    match subcommand.run {
        Handler::NoArguments(run) => {
            require_no_args(arguments, subcommand.name)?;
            run(root)
        }
        Handler::Forwarded(run) => run(root, arguments),
    }
}

/// Test-only interception of the two canonical device spawners. The dispatch
/// tests drive real `SUBCOMMANDS` rows and observe the suite, filter, flags and
/// snapshot variable each handler passes; the recorded call aborts its handler,
/// so neither a build nor the gates behind it ever run.
#[cfg(test)]
mod dispatch_probe {
    use std::cell::RefCell;
    use std::error::Error;
    use std::ffi::{OsStr, OsString};

    pub(super) const ABORTED: &str = "dispatch probe intercepted the device spawn";

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(super) enum Spawn {
        BenchDevice {
            suite: String,
            arguments: Vec<OsString>,
            baselines: Vec<&'static str>,
        },
        Qualification {
            filter: String,
            trailing: Vec<String>,
            environment: Option<(String, OsString)>,
        },
    }

    thread_local! {
        static RECORDER: RefCell<Option<Option<Spawn>>> = const { RefCell::new(None) };
    }

    /// Record the first spawn of an armed dispatch and abort its handler.
    /// Unarmed, it yields `None` and the caller spawns as usual.
    pub(super) fn intercept(spawn: Spawn) -> Option<Result<(), Box<dyn Error>>> {
        RECORDER.with(|recorder| {
            let mut recorder = recorder.borrow_mut();
            let slot = recorder.as_mut()?;
            if slot.is_none() {
                *slot = Some(spawn);
            }

            Some(Err(ABORTED.into()))
        })
    }

    /// Run one subcommand through the real router and return the device spawn
    /// its handler reached.
    pub(super) fn observe(command: &str, arguments: &[OsString]) -> Spawn {
        RECORDER.with(|recorder| *recorder.borrow_mut() = Some(None));
        let outcome = super::dispatch(
            super::workspace_root().unwrap(),
            OsStr::new(command),
            arguments,
        );
        let recorded = RECORDER.with(|recorder| recorder.borrow_mut().take().flatten());

        match outcome {
            Err(error) if error.to_string() == ABORTED => {}
            Err(error) => panic!("`{command}` failed before its device spawn: {error}"),
            Ok(()) => panic!("`{command}` returned without a device spawn"),
        }

        recorded.unwrap_or_else(|| panic!("`{command}` recorded no device spawn"))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(command) = arguments.next() else {
        return Err("usage: cargo run -p xtask -- <bootstrap-cuda-oxide|build-sm120|build-server|qualify-...|bench-...|gate-...|perf|profile|remote>".into());
    };
    let remaining = arguments.collect::<Vec<_>>();
    let root = workspace_root()?;

    dispatch(root, &command, &remaining)?;

    require_consumed_baseline_keys()
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest has no workspace parent".into())
}

fn bootstrap_cuda_oxide(root: &Path) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    if !source.join(".git").is_dir() {
        fs::create_dir_all(
            source
                .parent()
                .ok_or("cuda-oxide source path has no parent")?,
        )?;
        run_visible(
            Command::new("git")
                .args(["clone", "--no-checkout", CUDA_OXIDE_REPOSITORY])
                .arg(&source),
        )?;
        run_visible(Command::new("git").arg("-C").arg(&source).args([
            "checkout",
            "--detach",
            CUDA_OXIDE_REVISION,
        ]))?;
    }
    require_cuda_oxide_revision(&source)?;

    let driver_target = root.join("target/cuda-oxide-driver");
    run_visible(
        Command::new("cargo")
            .arg("+nightly-2026-04-03")
            .arg("build")
            .arg("--manifest-path")
            .arg(source.join("Cargo.toml"))
            .args(["--package", "cargo-oxide", "--target-dir"])
            .arg(&driver_target)
            .env("CARGO_HOME", task_cargo_home(root)),
    )?;
    let wrapper = driver_target.join("debug/cargo-oxide");
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    run_visible(
        Command::new(&wrapper)
            .arg("setup")
            .current_dir(&source)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS"),
    )?;

    println!(
        "cuda-oxide ready: {} at {}",
        CUDA_OXIDE_REVISION,
        wrapper.display()
    );
    Ok(())
}

fn build_sm120(root: &Path) -> Result<(), Box<dyn Error>> {
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let resource_baselines =
        perf_artifact::resource_baselines_sha256(root, SM120_RESOURCE_BASELINES)?;
    if perf_artifact::local_build_is_current(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    )? {
        println!("reusing verified worktree-local SM120 benchmark artifacts");
        return gate_sm120_resources(root);
    }
    if let Some(source) = perf_artifact::restore_build_from_worktrees(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    )? {
        println!(
            "restored verified SM120 benchmark artifacts from {}",
            source.display()
        );
        gate_sm120_resources(root)?;
        return Ok(());
    }

    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-device",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)?;
    perf_artifact::record_build(root, device_inputs, resource_baselines, CUDA_OXIDE_REVISION)
}

fn build_sm120_for_performance(root: &Path) -> Result<(), Box<dyn Error>> {
    require_performance_device_idle()?;
    build_sm120(root)?;
    wait_for_device_idle()
}

/// Concatenates the resource baselines binding one benchmark's measurement
/// identity. The given order is part of that identity: it feeds
/// `TUISKO_GENERATOR_BASELINE_SHA256`, so it is never sorted or deduplicated.
fn concatenated_resource_baselines(
    root: &Path,
    baseline_paths: &[&str],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut baselines = Vec::new();
    for baseline in baseline_paths {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    Ok(baselines)
}

/// The declared baselines for one canonical `bench-device` suite. An
/// undeclared suite is a wiring error, never a silently empty hash.
fn bench_device_baselines(suite_name: &str) -> Result<&'static [&'static str], Box<dyn Error>> {
    BENCH_DEVICE_BASELINES
        .iter()
        .find(|(suite, _)| *suite == suite_name)
        .map(|(_, baselines)| *baselines)
        .ok_or_else(|| {
            format!("no resource baselines are declared for benchmark suite `{suite_name}`").into()
        })
}

/// The canonical `bench-device` command line: the suite name, then the
/// caller's arguments verbatim, then the concatenated baseline hash.
fn bench_device_command(
    executable: &Path,
    suite_name: &str,
    arguments: &[std::ffi::OsString],
    baseline_hash: &str,
) -> Command {
    let mut command = Command::new(executable);
    command
        .arg(suite_name)
        .args(arguments)
        .env("TUISKO_GENERATOR_BASELINE_SHA256", baseline_hash);
    command
}

/// Runs one suite with an already-built `bench-device` executable.
fn run_prebuilt_bench_device(
    root: &Path,
    suite_name: &str,
    baseline_paths: &[&str],
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let baselines = concatenated_resource_baselines(root, baseline_paths)?;
    run_visible(&mut bench_device_command(
        &executable,
        suite_name,
        arguments,
        &sha256(&baselines),
    ))
}

/// Run one canonical `bench-device` suite. The device build precedes the
/// baseline reads so a stale executable fails before any hashing.
fn run_bench_device(
    root: &Path,
    suite_name: &str,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let baseline_paths = bench_device_baselines(suite_name)?;

    #[cfg(test)]
    if let Some(intercepted) = dispatch_probe::intercept(dispatch_probe::Spawn::BenchDevice {
        suite: suite_name.to_owned(),
        arguments: arguments.to_vec(),
        baselines: baseline_paths.to_vec(),
    }) {
        return intercepted;
    }

    build_sm120_for_performance(root)?;
    run_prebuilt_bench_device(root, suite_name, baseline_paths, arguments)
}

fn build_startup_benchmark(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-startup",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)
}

fn build_qwen38_flash_next_benchmark(root: &Path) -> Result<(), Box<dyn Error>> {
    build_sm120(root)?;
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-qwen38-flash-next",
            "--release",
        ],
    )?;
    build_sm120(root)
}

fn build_qwen38_flash_next_prompt_prime_benchmark(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-qwen38-flash-next-prompt-prime",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)
}

fn build_server(root: &Path) -> Result<(), Box<dyn Error>> {
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_BUILD_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuiskollm",
            "--release",
        ],
    )?;
    gate_sm120_resources(root)?;

    let binary = root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm");
    if !binary.is_file() {
        return Err(format!("server build omitted `{}`", binary.display()).into());
    }
    println!("server binary: {}", binary.display());
    Ok(())
}

/// Reconciles `tuisko_kernels_sm120::kernel_ptx_names()` against the entries
/// the build emitted. It runs as a test of the kernel crate rather than from
/// here because that crate cannot be an xtask dependency: its CUDA bindings
/// need a toolkit the host CI runner that builds xtask does not have.
fn gate_kernel_inventory(root: &Path) -> Result<(), Box<dyn Error>> {
    run_visible(Command::new("cargo").current_dir(root).args([
        "test",
        "--package",
        "tuisko-kernels-sm120",
        "--lib",
        "--",
        "--include-ignored",
    ]))
}

fn gate_sm120_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_kernel_inventory(root)?;
    gate_residual_norm(root)?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen36_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_fp8_gdn_input(root)?;
    gate_fp8_lm_head(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen36_moe_router(root)?;
    gate_qwen36_moe_experts(root)?;
    gate_qwen36_nvfp4_lm_head(root)?;
    gate_qwen36_fp8_qkv(root)?;
    gate_qwen36_gdn_input(root)?;
    gate_qwen36_gdn_output(root)?;
    gate_qwen36_attention_output(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_qsa_selection(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)?;
    gate_gdn_state_snapshot(root)?;
    gate_gdn_output(root)?;
    gate_attention_qk_prepare(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen36_attention_qk_prepare(root)?;
    gate_qwen36_fp8_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen36_paged_gqa(root)?;
    gate_qwen36_fp8_paged_gqa(root)?;
    gate_long_context_paged_gqa(root)?;
    gate_attention_output(root)?;
    gate_mtp_bf16_fusion(root)?;
    gate_mtp_bf16_attention_output(root)?;
    gate_mtp_bf16_mlp(root)?;
    gate_mtp_bf16_qkv(root)?;
    gate_mtp_bf16_qk_prepare(root)?;
    gate_mtp_bf16_paged_gqa(root)?;
    gate_qwen35_mtp_resources(root)?;
    gate_qwen36_mtp_resources(root)?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_ple(root)
}

fn build_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let gpu = parse_build_gpu(arguments, "build-residual-norm")?;
    let prepared = prepare_remote_qualify(root, gpu, "residual_norm_suite_")?;
    gate_residual_norm_target(root, gpu)?;
    println!(
        "{} ({}, compute capability {}) residual-norm qualification artifact: {}",
        gpu.key(),
        gpu.device_name(),
        gpu.compute_capability_text(),
        prepared.executable.display()
    );
    Ok(())
}

fn build_residual_bench(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let gpu = parse_build_gpu(arguments, "build-residual-bench")?;
    build_residual_benchmark_target(root, gpu)?;
    let prepared = prepare_remote_benchmark(root, gpu, PerformanceSuite::ResidualNorm)?;
    println!(
        "{} residual-norm benchmark artifact: {} (resource baseline {})",
        gpu.key(),
        prepared.executable.display(),
        prepared.generator_baseline_sha256,
    );
    Ok(())
}

fn parse_build_gpu(
    arguments: &[std::ffi::OsString],
    command: &str,
) -> Result<gpu_target::GpuTarget, Box<dyn Error>> {
    let [flag, value] = arguments else {
        return Err(
            format!("usage: cargo run -p xtask -- {command} --gpu <5090|4090|3090>").into(),
        );
    };
    if flag != "--gpu" {
        return Err(format!("{command} requires `--gpu <5090|4090|3090>`").into());
    }
    Ok(gpu_target::GpuTarget::parse(&value.to_string_lossy())?)
}

pub(crate) fn build_residual_benchmark_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        return build_sm120(root);
    }
    run_oxide(
        root,
        &[
            "build",
            "--arch",
            gpu.oxide_arch(),
            "--cargo-target-dir",
            gpu.oxide_build_target(),
            "--device-codegen-crate",
            gpu.device_codegen_crates(),
            "--",
            "--package",
            "tuisko-qual",
            "--bin",
            "bench-device",
            "--release",
            "--no-default-features",
            "--features",
            gpu.qualification_feature(),
        ],
    )?;
    gate_residual_norm_target(root, gpu)
}

fn qualify_frontend(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-frontend SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--bin",
                "qualify-frontend",
                "--",
            ])
            .arg(snapshot),
    )
}

fn qualify_generation(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation SNAPSHOT".into());
    };
    run_visible(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "run",
                "--package",
                "tuisko-qual",
                "--no-default-features",
                "--features",
                "engine",
                "--bin",
                "qualify-generation",
                "--",
            ])
            .arg(snapshot),
    )
}

/// The sm89/sm86 qualification configurations have no engine consumer, so this
/// host check is the only thing keeping them compiling.
fn check_portable_qualification(root: &Path) -> Result<(), Box<dyn Error>> {
    for arch in ["sm89", "sm86"] {
        run_visible(Command::new("cargo").current_dir(root).args([
            "check",
            "--package",
            "tuisko-qual",
            "--no-default-features",
            "--features",
            arch,
        ]))?;
    }

    Ok(())
}

/// Runs every non-`#[ignore]` (host-side) tuisko-qual unit test with the GPU
/// hidden, so stale host asserts fail before any device work.
fn qualify_host(root: &Path) -> Result<(), Box<dyn Error>> {
    check_portable_qualification(root)?;
    run_oxide_with_env(
        root,
        &[
            "test",
            "--arch",
            "sm_120a",
            "--cargo-target-dir",
            CUDA_OXIDE_TEST_TARGET,
            "--device-codegen-crate",
            SM120_DEVICE_CODEGEN_CRATES,
            "--",
            "--package",
            "tuisko-qual",
            "--release",
            "--lib",
        ],
        Some(("CUDA_VISIBLE_DEVICES", OsStr::new(""))),
    )
}

fn qualify_qwen38_flash_next_hyper_connection(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_hyper_connection_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_hyper_connection(root)
}

fn qualify_qwen38_flash_next_ple(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_PLE_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_ple(root)
}

fn qualify_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "residual_norm_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_residual_norm(root)
}

fn qualify_qwen35_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN35_RESIDUAL_NORM_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen35_residual_norm(root)
}

fn qualify_qwen36_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_residual_norm",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_residual_norm(root)
}

fn qualify_qwen35_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_nvfp4_swiglu",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_qwen35_nvfp4_swiglu(root)
}

fn qualify_qwen35_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_nvfp4_down",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_qwen35_nvfp4_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(root, "qwen35_nvfp4_qkv", QUALIFICATION_IGNORED_FLAGS, None)?;
    gate_qwen35_nvfp4_qkv(root)
}

fn qualify_qwen36_moe_router(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_moe_router::tests::exact_routes_match_independent_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_moe_router(root)
}

fn qualify_qwen36_moe_experts(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_moe_experts::tests::exact_routes_match_independent_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_moe_experts(root)
}

fn qualify_qwen36_nvfp4_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_nvfp4_lm_head",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_nvfp4_lm_head(root)
}

fn qualify_qwen36_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_gdn_input",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_gdn_input(root)
}

fn qualify_qwen36_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_fp8_qkv",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_fp8_qkv(root)
}

fn qualify_qwen36_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_gdn_output::tests::exact_routes_match_independent_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_gdn_output(root)
}

fn qualify_qwen36_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_attention_output",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_gdn_output(root)?;
    gate_qwen36_attention_output(root)
}

fn qualify_qwen36_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_gdn_prepare::tests::qwen36_exact_routes_match_shared_independent_oracle",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen35_gdn_prepare(root)
}

fn qualify_qwen36_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_gdn_recurrence::tests::qwen36_exact_routes_match_shared_independent_oracle",
        &[
            "--exact",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        None,
    )?;
    gate_qwen35_gdn_recurrence(root)
}

fn qualify_qwen35_bf16_lm_head(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-bf16-lm-head SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_bf16_lm_head::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_bf16_lm_head(root)
}

fn qualify_qwen35_mtp_bf16_fusion(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-mtp-bf16-fusion SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_fusion_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen35_mtp_bf16_attention(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen35-mtp-bf16-attention SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "qwen35_mtp_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen36_mtp_bf16_attention(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen36-mtp-bf16-attention SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "qwen36_mtp_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen36_mtp_resources(root)
}

fn qualify_qwen36_mtp_bf16_moe(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen36_mtp_bf16_moe_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen36_mtp_resources(root)
}

fn qualify_qwen35_mtp_bf16_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-mtp-bf16-mlp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_mtp_mlp_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen35_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-text-endpoint SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN35_TEXT_ENDPOINT_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_bf16_lm_head(root)
}

fn qualify_qwen36_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-text-endpoint SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen36_text_endpoint::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_nvfp4_lm_head(root)
}

fn qualify_qwen36_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-resident-model SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN36_RESIDENT_MODEL_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen36_gdn_output(root)?;
    gate_qwen36_moe_router(root)?;
    gate_qwen36_moe_experts(root)?;
    gate_qwen36_fp8_qkv(root)?;
    gate_qwen36_fp8_attention_qk_prepare(root)?;
    gate_qwen36_fp8_paged_gqa(root)?;
    gate_qwen36_attention_output(root)?;
    gate_qwen36_nvfp4_lm_head(root)
}

fn qualify_qwen36_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-generation SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen36_generation::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen36_gdn_output(root)?;
    gate_qwen36_moe_router(root)?;
    gate_qwen36_moe_experts(root)?;
    gate_qwen36_fp8_qkv(root)?;
    gate_qwen36_fp8_attention_qk_prepare(root)?;
    gate_qwen36_fp8_paged_gqa(root)?;
    gate_qwen36_attention_output(root)?;
    gate_qwen36_nvfp4_lm_head(root)
}

fn qualify_qwen36_server(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-server SNAPSHOT".into());
    };
    require_performance_device_idle()?;
    build_server(root)?;
    require_performance_device_idle()?;
    let executable = root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm");
    server_qualification::qualify_qwen36(&executable, Path::new(snapshot))
}

fn qualify_qwen35_server(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-server SNAPSHOT".into());
    };
    require_performance_device_idle()?;
    build_server(root)?;
    require_performance_device_idle()?;
    let executable = root.join(CUDA_OXIDE_BUILD_TARGET).join("release/tuiskollm");
    server_qualification::qualify_qwen35(&executable, Path::new(snapshot))
}

fn qualify_qwen35_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-resident-model SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN35_RESIDENT_MODEL_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)
}

fn qualify_qwen35_resident_mtp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-resident-mtp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN35_RESIDENT_MTP_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen35_mtp_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-mtp-generation SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN35_MTP_GENERATION_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen35_mtp_batch_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen35-mtp-batch-generation SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN35_MTP_BATCH_GENERATION_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen35_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-generation SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_generation::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_bf16_lm_head(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)
}

fn qualify_qwen35_long_context_kv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN35_LONG_CONTEXT_KV_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )
}

fn qualify_qwen36_long_context_kv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN36_LONG_CONTEXT_KV_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )
}

fn qualify_streaming_pool(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        STREAMING_WEIGHT_POOL_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )
}

fn qualify_qwen38_flash_next_engram_staging(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_ENGRAM_STAGING_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )
}

fn qualify_qwen35_nvfp4_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_nvfp4_gdn_input",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_qwen35_nvfp4_gdn_input(root)
}

fn qualify_qwen35_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_gdn_prepare",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen35_gdn_prepare(root)
}

fn qualify_qwen35_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_gdn_recurrence",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen35_gdn_recurrence(root)
}

fn qualify_qwen35_nvfp4_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_nvfp4_gdn_output",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_qwen35_nvfp4_attention_output(root)
}

fn qualify_qwen35_nvfp4_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen35_nvfp4_attention_output::tests::exact_batches_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "qwen35_nvfp4_attention_output_benchmark::tests::accounting_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen35_nvfp4_attention_output(root)
}

fn qualify_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "fp8_qkv",
        &[
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
            "--skip",
            "qwen36_fp8_qkv",
        ],
        None,
    )?;
    wait_for_device_idle()?;
    run_qualification_test(
        root,
        "qwen36_fp8_qkv",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_fp8_qkv(root)
}

fn qualify_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "fp8_swiglu_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_fp8_swiglu(root)
}

fn qualify_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "fp8_down_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_fp8_down(root)
}

fn qualify_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(root, "nvfp4_swiglu", QUALIFICATION_IGNORED_FLAGS, None)?;
    gate_nvfp4_swiglu(root)
}

fn qualify_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(root, "nvfp4_down", QUALIFICATION_IGNORED_FLAGS, None)?;
    gate_nvfp4_down(root)
}

fn qualify_nvfp4_mlp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)
}

fn qualify_qwen35_nvfp4_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-nvfp4-mlp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "nvfp4_mlp::tests::qwen35_source_layer0_matches_complete_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "gdn_prepare::tests",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_gdn_prepare(root)
}

fn qualify_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    for test in [
        "gdn_recurrence::tests::route_inventory_and_arena_accounting_are_exact",
        "gdn_recurrence::tests::exact_routes_match_independent_oracles_and_graph_replay",
    ] {
        run_qualification_test(
            root,
            test,
            &[
                "--exact",
                "--include-ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            None,
        )?;
    }
    gate_gdn_recurrence(root)
}

/// Selects the prepare oracle and benchmark-accounting tests by shared prefix.
fn qualify_qwen38_flash_next_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_gdn_prepare",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)
}

/// Selects the recurrence oracle and benchmark-accounting tests by shared prefix.
fn qualify_qwen38_flash_next_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_gdn_recurrence",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_gdn_recurrence(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)
}

/// Runs the QSA prepare oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_qsa_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_qsa_prepare",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_qsa_prepare(root)
}

/// Runs the QSA attention oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_qsa_attention(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_qsa_attention",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_qsa_attention(root)
}

/// Runs the QSA selection oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_qsa_selection(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_qsa_selection",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_qsa_selection(root)
}

/// Runs the MoE router oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_moe_router(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_moe_router",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_moe_router(root)
}

/// Runs the source-backed MTP oracle and every composed artifact gate.
fn qualify_qwen38_flash_next_mtp_oracle(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-mtp-oracle SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_MTP_ORACLE_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)
}

/// Runs the source-backed MTP identity and every composed artifact gate.
fn qualify_qwen38_flash_next_mtp_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-mtp-generation SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_MTP_GENERATION_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)?;
    gate_qwen38_flash_next_ple(root)
}

/// Runs the MoE expert oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_moe_experts(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "qwen38_flash_next_moe_experts",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_moe_experts(root)
}

/// Runs the backbone projection oracles, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_projections(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_PROJECTION_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_projections(root)
}

/// Runs the BF16 LM-head oracle, accounting tests, and artifact gate.
fn qualify_qwen38_flash_next_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_LM_HEAD_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_qwen38_flash_next_lm_head(root)
}

/// Runs the source-backed generation gate and every resource family it composes.
fn qualify_qwen38_flash_next_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-generation SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_GENERATION_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_qsa_selection(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)?;
    gate_qwen38_flash_next_ple(root)
}

/// Runs the source-backed compact-scheduler gate.
fn qualify_qwen38_flash_next_compact_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-compact-generation SNAPSHOT"
                .into(),
        );
    };
    run_qualification_test(
        root,
        "qwen38_flash_next_compact_generation",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )
}

/// Runs grouped prompt-prime exactness, accounting, and composed resource gates.
fn qualify_qwen38_flash_next_prompt_prime(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-prompt-prime SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_PROMPT_PRIME_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)?;
    gate_qwen38_flash_next_ple(root)
}

/// Runs the source-backed GDN/MoE layer gate and every leaf resource gate it composes.
fn qualify_qwen38_flash_next_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-gdn-layer SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_GDN_LAYER_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)
}

/// Runs the whole-model gate and every resource family it composes.
fn qualify_qwen38_flash_next_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-resident-model SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_RESIDENT_MODEL_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_gdn_prepare(root)?;
    gate_qwen38_flash_next_gdn_recurrence(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_qsa_selection(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)?;
    gate_qwen38_flash_next_lm_head(root)?;
    gate_qwen38_flash_next_ple(root)
}

/// Runs the source-backed QSA/MoE layer gate and every leaf resource gate it composes.
fn qualify_qwen38_flash_next_qsa_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen38-flash-next-qsa-layer SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        QWEN38_FLASH_NEXT_QSA_LAYER_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen38_flash_next_hyper_connection(root)?;
    gate_qwen38_flash_next_qsa_prepare(root)?;
    gate_qwen38_flash_next_qsa_attention(root)?;
    gate_qwen38_flash_next_qsa_selection(root)?;
    gate_qwen38_flash_next_moe_router(root)?;
    gate_qwen38_flash_next_moe_experts(root)?;
    gate_qwen38_flash_next_projections(root)
}

fn qualify_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "gdn_output::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_gdn_output(root)
}

fn qualify_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "attention_qk_prepare",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_attention_qk_prepare(root)
}

fn qualify_qwen35_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "attention_qk_prepare::tests::qwen35_",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "attention_qk_prepare_benchmark::tests::qwen35_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen35_attention_qk_prepare(root)
}

fn qualify_qwen36_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "attention_qk_prepare::tests::qwen36_exact_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "attention_qk_prepare_benchmark::tests::qwen36_bf16_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_attention_qk_prepare(root)
}

fn qualify_qwen36_fp8_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "attention_qk_prepare::tests::qwen36_fp8_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "attention_qk_prepare_benchmark::tests::qwen36_fp8_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_fp8_attention_qk_prepare(root)
}

fn qualify_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "paged_gqa_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    gate_paged_gqa(root)
}

fn qualify_qwen36_fp8_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "paged_gqa::tests::qwen36_fp8_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "bf16_paged_gqa_benchmark::tests::qwen36_fp8_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_fp8_paged_gqa(root)
}

fn qualify_qwen35_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "paged_gqa::tests::qwen35_bf16_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "bf16_paged_gqa_benchmark::tests::qwen35_bf16_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen35_paged_gqa(root)
}

fn qualify_qwen36_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "paged_gqa::tests::qwen36_bf16_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "bf16_paged_gqa_benchmark::tests::qwen36_bf16_",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_paged_gqa(root)
}

fn qualify_long_context_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "long_context_paged_gqa",
        QUALIFICATION_IGNORED_FLAGS,
        None,
    )?;
    gate_long_context_paged_gqa(root)
}

fn qualify_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "attention_output::tests::attention_output_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "attention_output_prefill::tests::attention_output_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        "attention_output_benchmark::tests::attention_output_suite_",
        &["--nocapture", "--test-threads=1"],
        None,
    )?;
    gate_attention_output(root)
}

fn qualify_mtp_bf16_fusion(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-bf16-fusion SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "mtp_bf16_fusion_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_mtp_bf16_fusion(root)
}

fn qualify_mtp_bf16_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-mtp-bf16-attention-output SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "mtp_bf16_attention_output_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_mtp_bf16_attention_output(root)
}

fn qualify_mtp_bf16_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-bf16-mlp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "mtp_bf16_mlp_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_mtp_bf16_mlp(root)
}

fn qualify_mtp_bf16_qkv(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-bf16-qkv SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "mtp_bf16_qkv_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_mtp_bf16_qkv(root)
}

fn qualify_mtp_bf16_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-bf16-qk-prepare SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "mtp_bf16_qk_prepare_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_mtp_bf16_qk_prepare(root)
}

fn qualify_mtp_bf16_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(
        root,
        "mtp_bf16_paged_gqa_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        None,
    )?;
    run_qualification_test(
        root,
        MTP_BF16_PAGED_GQA_BENCHMARK_FILTER,
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_mtp_bf16_paged_gqa(root)
}

fn qualify_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-dense-fp8-mlp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "dense_fp8_mlp_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_dense_fp8_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-dense-fp8-gdn-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "dense_fp8_gdn_layer::tests::source_layer60_matches_complete_seam_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_gdn_input(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_gdn_output(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-full-attention-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "full_attention_layer_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_attention_output(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)
}

fn qualify_mtp_layer(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        MTP_LAYER_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    run_qualification_test(
        root,
        MTP_LAYER_BENCHMARK_FILTER,
        &["--nocapture", "--test-threads=1"],
        None,
    )?;
    gate_mtp_layer(root)
}

fn qualify_qwen35_mtp_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-mtp-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_mtp_layer_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    for baseline in QWEN35_MTP_LAYER_RESOURCE_BASELINES {
        if !root.join(baseline).is_file() {
            return Err(format!("missing Qwen3.5 MTP layer resource baseline `{baseline}`").into());
        }
    }
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_mtp_resources(root)
}

fn qualify_qwen36_mtp_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-mtp-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        QWEN36_MTP_LAYER_TEST_FILTER,
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_mtp_resources(root)
}

fn qualify_target_mtp_verify(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-target-mtp-verify SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "target_mtp_verify::tests::exact_target_verify_and_commit_match_source_oracles",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn qualify_mtp_prompt_prime(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-mtp-prompt-prime SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "mtp_prompt_prime_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_mtp_prompt_prime(root)
}

fn qualify_resident_mtp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-resident-mtp SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_mtp_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_mtp(root)
}

fn qualify_generation_mtp_greedy(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation-mtp-greedy SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_mtp_generation_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_mtp(root)
}

fn qualify_generation_mtp_sampling(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation-mtp-sampling SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_mtp_sampling_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_mtp(root)
}

fn qualify_generation_mtp_batch(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-generation-mtp-batch SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_mtp_batch_suite_",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_mtp(root)
}

pub(crate) fn gate_mtp_prompt_prime(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm(root)?;
    gate_mtp_bf16_fusion(root)?;
    gate_mtp_bf16_qkv(root)?;
    gate_mtp_bf16_qk_prepare(root)
}

pub(crate) fn gate_resident_mtp(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_resident_model_resources(root)?;
    gate_mtp_layer(root)
}

pub(crate) fn gate_mtp_layer(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm(root)?;
    gate_mtp_bf16_fusion(root)?;
    gate_mtp_bf16_qkv(root)?;
    gate_mtp_bf16_qk_prepare(root)?;
    gate_mtp_bf16_paged_gqa(root)?;
    gate_mtp_bf16_attention_output(root)?;
    gate_mtp_bf16_mlp(root)?;
    gate_fp8_lm_head(root)
}

fn qualify_qwen35_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen35-full-attention-layer SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "qwen35_full_attention_layer::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    run_qualification_test(
        root,
        "qwen35_full_attention_layer_benchmark::tests",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_qkv(root)?;
    gate_qwen35_attention_qk_prepare(root)?;
    gate_qwen35_paged_gqa(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_qwen35_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen35-gdn-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen35_gdn_layer::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN35_SNAPSHOT", snapshot.as_os_str())),
    )?;
    run_qualification_test(
        root,
        "qwen35_gdn_layer_benchmark::tests",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen35_residual_norm(root)?;
    gate_qwen35_nvfp4_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen35_nvfp4_attention_output(root)?;
    gate_qwen35_nvfp4_swiglu(root)?;
    gate_qwen35_nvfp4_down(root)
}

fn qualify_qwen36_gdn_moe_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-qwen36-gdn-moe-layer SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "qwen36_gdn_moe_layer::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    run_qualification_test(
        root,
        "qwen36_gdn_moe_layer_benchmark::tests",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_gdn_input(root)?;
    gate_qwen35_gdn_prepare(root)?;
    gate_qwen35_gdn_recurrence(root)?;
    gate_qwen36_gdn_output(root)?;
    gate_qwen36_moe_router(root)?;
    gate_qwen36_moe_experts(root)
}

fn qualify_qwen36_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-qwen36-full-attention-layer SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "qwen36_full_attention_layer::tests",
        QUALIFICATION_IGNORED_SERIAL_FLAGS,
        Some(("TUISKO_QWEN36_SNAPSHOT", snapshot.as_os_str())),
    )?;
    run_qualification_test(
        root,
        "qwen36_full_attention_layer_benchmark::tests",
        QUALIFICATION_NOCAPTURE_FLAGS,
        None,
    )?;
    gate_qwen36_residual_norm(root)?;
    gate_qwen36_fp8_qkv(root)?;
    gate_qwen36_fp8_attention_qk_prepare(root)?;
    gate_qwen36_fp8_paged_gqa(root)?;
    gate_qwen36_attention_output(root)?;
    gate_qwen36_moe_router(root)?;
    gate_qwen36_moe_experts(root)
}

fn qualify_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-resident-model SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn qualify_resident_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-resident-generation SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn qualify_resident_batch_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT".into(),
        );
    };
    run_qualification_test(
        root,
        "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_resident_model_resources(root)
}

fn gate_resident_model_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm(root)?;
    gate_fp8_qkv(root)?;
    gate_fp8_gdn_input(root)?;
    gate_fp8_lm_head(root)?;
    gate_fp8_swiglu(root)?;
    gate_fp8_down(root)?;
    gate_nvfp4_swiglu(root)?;
    gate_nvfp4_down(root)?;
    gate_gdn_prepare(root)?;
    gate_gdn_recurrence(root)?;
    gate_gdn_state_snapshot(root)?;
    gate_gdn_output(root)?;
    gate_attention_qk_prepare(root)?;
    gate_paged_gqa(root)?;
    gate_long_context_paged_gqa(root)?;
    gate_attention_output(root)
}

fn qualify_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    run_qualification_test(root, "fp8_gdn_input", QUALIFICATION_IGNORED_FLAGS, None)?;
    gate_fp8_gdn_input(root)
}

fn qualify_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    build_sm120(root)?;
    run_visible(
        Command::new(
            root.join(CUDA_OXIDE_BUILD_TARGET)
                .join("release/bench-device"),
        )
        .arg("qualify-fp8-lm-head"),
    )?;
    gate_fp8_lm_head(root)
}

fn qualify_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err("usage: cargo run -p xtask -- qualify-text-endpoint SNAPSHOT".into());
    };
    run_qualification_test(
        root,
        "text_endpoint::tests::source_endpoint_matches_independent_oracles_and_graph_replay",
        QUALIFICATION_IGNORED_FLAGS,
        Some(("TUISKO_SNAPSHOT", snapshot.as_os_str())),
    )?;
    gate_residual_norm(root)?;
    gate_fp8_lm_head(root)
}

fn bench_qwen38_flash_next_hyper_connection(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-hyper-connection", arguments)
}

fn bench_qwen38_flash_next_ple(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-ple", arguments)
}

fn bench_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::ResidualNorm, arguments)
}

fn bench_qwen35_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-residual-norm", arguments)
}

fn bench_qwen36_residual_norm(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-residual-norm", arguments)
}

fn bench_qwen35_nvfp4_swiglu(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-swiglu", arguments)
}

fn bench_qwen35_nvfp4_down(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-down", arguments)
}

fn bench_qwen35_nvfp4_qkv(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-qkv", arguments)
}

fn bench_qwen36_moe_router(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-moe-router", arguments)
}

fn bench_qwen36_moe_experts(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-moe-experts", arguments)
}

fn bench_qwen36_nvfp4_lm_head(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-nvfp4-lm-head", arguments)
}

fn bench_qwen36_fp8_qkv(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-fp8-qkv", arguments)
}

fn bench_qwen36_gdn_input(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-gdn-input", arguments)
}

fn bench_qwen36_gdn_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-gdn-output", arguments)
}

fn bench_qwen36_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-attention-output", arguments)
}

fn bench_qwen36_gdn_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-gdn-prepare", arguments)
}

fn bench_qwen38_flash_next_gdn_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-gdn-prepare", arguments)
}

fn bench_qwen38_flash_next_gdn_recurrence(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-gdn-recurrence", arguments)
}

fn bench_qwen38_flash_next_qsa_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-qsa-prepare", arguments)
}

fn bench_qwen38_flash_next_qsa_attention(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-qsa-attention", arguments)
}

fn bench_qwen38_flash_next_qsa_selection(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-qsa-selection", arguments)
}

fn bench_qwen38_flash_next_moe_router(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-moe-router", arguments)
}

fn bench_qwen38_flash_next_moe_experts(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-moe-experts", arguments)
}

fn bench_qwen38_flash_next_projections(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-projections", arguments)
}

fn bench_qwen38_flash_next_lm_head(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-lm-head", arguments)
}

fn bench_qwen38_flash_next_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-gdn-layer", arguments)
}

fn bench_qwen38_flash_next_qsa_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-qsa-layer", arguments)
}

fn bench_qwen38_flash_next_ple_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen38-flash-next-ple-layer", arguments)
}

fn bench_qwen36_gdn_recurrence(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-gdn-recurrence", arguments)
}

fn bench_qwen35_nvfp4_gdn_input(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-gdn-input", arguments)
}

fn bench_qwen35_gdn_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-gdn-prepare", arguments)
}

fn bench_qwen35_gdn_recurrence(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-gdn-recurrence", arguments)
}

fn bench_qwen35_nvfp4_gdn_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-gdn-output", arguments)
}

fn bench_qwen35_nvfp4_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-nvfp4-attention-output", arguments)
}

fn bench_qwen35_nvfp4_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-nvfp4-mlp SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-nvfp4-mlp", arguments)
}

fn bench_qwen35_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-attention-qk-prepare", arguments)
}

fn bench_qwen36_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-attention-qk-prepare", arguments)
}

fn bench_qwen36_fp8_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-fp8-attention-qk-prepare", arguments)
}

fn bench_fp8_qkv(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Qkv, arguments)
}

fn bench_fp8_gdn_input(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8GdnInput, arguments)
}

fn bench_fp8_lm_head(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8LmHead, arguments)
}

fn bench_fp8_swiglu(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8SwiGlu, arguments)
}

fn bench_fp8_down(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Fp8Down, arguments)
}

fn bench_nvfp4_swiglu(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Nvfp4SwiGlu, arguments)
}

fn bench_nvfp4_down(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::Nvfp4Down, arguments)
}

fn bench_nvfp4_mlp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err("usage: cargo run -p xtask -- bench-nvfp4-mlp SNAPSHOT [options]".into());
    }
    run_bench_device(root, "nvfp4-mlp", arguments)
}

fn bench_gdn_prepare(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnPrepare, arguments)
}

fn bench_gdn_recurrence(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnRecurrence, arguments)
}

fn bench_gdn_output(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::GdnOutput, arguments)
}

fn bench_attention_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::AttentionQkPrepare, arguments)
}

fn bench_paged_gqa(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::PagedGqa, arguments)
}

fn bench_qwen35_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen35-paged-gqa", arguments)
}

fn bench_qwen36_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-paged-gqa", arguments)
}

fn bench_qwen36_fp8_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_bench_device(root, "qwen36-fp8-paged-gqa", arguments)
}

fn bench_long_context_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::LongContextPagedGqa, arguments)
}

fn bench_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::AttentionOutput, arguments)
}

fn bench_mtp_bf16_fusion(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16Fusion, arguments)
}

fn bench_mtp_bf16_attention_output(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16AttentionOutput, arguments)
}

fn bench_mtp_bf16_mlp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16Mlp, arguments)
}

fn bench_mtp_bf16_qkv(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16Qkv, arguments)
}

fn bench_mtp_bf16_qk_prepare(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16QkPrepare, arguments)
}

fn bench_mtp_bf16_paged_gqa(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_suite(root, PerformanceSuite::MtpBf16PagedGqa, arguments)
}

fn bench_dense_fp8_mlp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err("usage: cargo run -p xtask -- bench-dense-fp8-mlp SNAPSHOT [options]".into());
    }
    run_bench_device(root, "dense-fp8-mlp", arguments)
}

fn bench_dense_fp8_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-dense-fp8-gdn-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "dense-fp8-gdn-layer", arguments)
}

fn bench_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-full-attention-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "full-attention-layer", arguments)
}

fn bench_mtp_layer(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err("usage: cargo run -p xtask -- bench-mtp-layer SNAPSHOT [options]".into());
    }
    run_bench_device(root, "mtp-layer", arguments)
}

fn bench_qwen35_mtp_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-mtp-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-mtp-layer", arguments)
}

fn bench_qwen36_mtp_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen36-mtp-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen36-mtp-layer", arguments)
}

fn bench_qwen35_resident_mtp(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-resident-mtp SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-resident-mtp", arguments)
}

fn bench_qwen35_mtp_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-mtp-generation SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-mtp-generation", arguments)
}

fn bench_qwen35_mtp_batch_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-mtp-batch-generation SNAPSHOT [options]"
                .into(),
        );
    }
    run_bench_device(root, "qwen35-mtp-batch-generation", arguments)
}

fn bench_target_mtp_verify(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-target-mtp-verify SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "target-mtp-verify", arguments)
}

fn bench_mtp_prompt_prime(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-mtp-prompt-prime SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "mtp-prompt-prime", arguments)
}

fn bench_resident_mtp(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err("usage: cargo run -p xtask -- bench-resident-mtp SNAPSHOT [options]".into());
    }
    run_bench_device(root, "resident-mtp", arguments)
}

fn bench_generation_mtp_greedy(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-generation-mtp-greedy SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "generation-mtp-greedy", arguments)
}

fn bench_generation_mtp_sampling(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-generation-mtp-sampling SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "generation-mtp-sampling", arguments)
}

fn bench_generation_mtp_batch(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-generation-mtp-batch SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "generation-mtp-batch", arguments)
}

fn bench_qwen35_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-full-attention-layer SNAPSHOT [options]"
                .into(),
        );
    }
    run_bench_device(root, "qwen35-full-attention-layer", arguments)
}

fn bench_qwen35_gdn_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-gdn-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-gdn-layer", arguments)
}

fn bench_qwen36_gdn_moe_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen36-gdn-moe-layer SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen36-gdn-moe-layer", arguments)
}

fn bench_qwen36_full_attention_layer(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen36-full-attention-layer SNAPSHOT [options]"
                .into(),
        );
    }
    run_bench_device(root, "qwen36-full-attention-layer", arguments)
}

fn bench_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_resident_model_variant(root, arguments, "bench-resident-model", "resident-model")
}

fn bench_resident_prefill(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_resident_model_variant(
        root,
        arguments,
        "bench-resident-prefill",
        "resident-prefill",
    )
}

/// Runs requested Qwen3.8 Flash-Next sweeps from one resident model construction.
fn bench_qwen38_flash_next(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen38-flash-next SNAPSHOT [--sweeps resident,generation] [options]"
                .into(),
        );
    };
    require_performance_device_idle()?;
    build_qwen38_flash_next_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-qwen38-flash-next");
    if !executable.is_file() {
        return Err(format!(
            "Qwen3.8 Flash-Next benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut command = Command::new(executable);
    command
        .arg(snapshot)
        .args(options)
        .arg("--cuda-oxide-commit")
        .arg(CUDA_OXIDE_REVISION);
    append_benchmark_cache_source_identity(root, options, &mut command)?;
    run_visible(&mut command)
}

fn append_benchmark_cache_source_identity(
    root: &Path,
    arguments: &[std::ffi::OsString],
    command: &mut Command,
) -> Result<(), Box<dyn Error>> {
    if !arguments
        .iter()
        .any(|argument| argument == "--baseline-cache")
    {
        return Ok(());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--base-sha" || argument == "--device-input-sha256")
    {
        return Err("benchmark cache source stamps are supplied by xtask".into());
    }
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let base_sha = String::from_utf8(output.stdout)?.trim().to_string();
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    command
        .arg("--base-sha")
        .arg(base_sha)
        .arg("--device-input-sha256")
        .arg(device_inputs);

    Ok(())
}

/// Times whole generation requests with production cache state.
fn bench_qwen38_flash_next_generation(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen38-flash-next-generation SNAPSHOT [options]"
                .into(),
        );
    };
    require_performance_device_idle()?;
    build_qwen38_flash_next_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-qwen38-flash-next");
    if !executable.is_file() {
        return Err(format!(
            "Qwen3.8 Flash-Next generation benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(
        Command::new(executable)
            .arg(snapshot)
            .args(["--sweeps", "generation"])
            .args(options)
            .args(["--cuda-oxide-commit", CUDA_OXIDE_REVISION]),
    )
}

/// Times sequential and grouped admission on the production owner.
fn bench_qwen38_flash_next_prompt_prime(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let [snapshot] = arguments else {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen38-flash-next-prompt-prime SNAPSHOT".into(),
        );
    };
    require_performance_device_idle()?;
    build_qwen38_flash_next_prompt_prime_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-qwen38-flash-next-prompt-prime");
    if !executable.is_file() {
        return Err(format!(
            "Qwen3.8 Flash-Next prompt-prime benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(Command::new(executable).arg(snapshot))
}

/// Times the full host-observed resident step outside the single-graph benchmark harness.
fn bench_qwen38_flash_next_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen38-flash-next-resident-model SNAPSHOT [options]".into(),
        );
    };
    require_performance_device_idle()?;
    build_qwen38_flash_next_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-qwen38-flash-next");
    if !executable.is_file() {
        return Err(format!(
            "Qwen3.8 Flash-Next resident benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let mut command = Command::new(executable);
    command
        .arg(snapshot)
        .args(["--sweeps", "resident"])
        .args(options)
        .args(["--cuda-oxide-commit", CUDA_OXIDE_REVISION]);
    append_benchmark_cache_source_identity(root, options, &mut command)?;
    run_visible(&mut command)
}

fn bench_startup(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some((snapshot, options)) = arguments.split_first() else {
        return Err("usage: cargo run -p xtask -- bench-startup SNAPSHOT [options]".into());
    };
    require_performance_device_idle()?;
    build_startup_benchmark(root)?;
    wait_for_device_idle()?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-startup");
    if !executable.is_file() {
        return Err(format!(
            "startup benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    run_visible(Command::new(executable).arg(snapshot).args(options))
}

fn bench_resident_long_context_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    bench_resident_model_variant(
        root,
        arguments,
        "bench-resident-long-context-model",
        "resident-long-context-model",
    )
}

fn bench_resident_model_variant(
    root: &Path,
    arguments: &[std::ffi::OsString],
    command: &str,
    suite: &str,
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(format!("usage: cargo run -p xtask -- {command} SNAPSHOT [options]").into());
    }
    run_bench_device(root, suite, arguments)
}

fn bench_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err("usage: cargo run -p xtask -- bench-text-endpoint SNAPSHOT [options]".into());
    }
    run_bench_device(root, "text-endpoint", arguments)
}

fn bench_qwen35_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-text-endpoint SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-text-endpoint", arguments)
}

fn bench_qwen36_text_endpoint(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen36-text-endpoint SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen36-text-endpoint", arguments)
}

fn bench_qwen35_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen35-resident-model SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen35-resident-model", arguments)
}

fn bench_qwen36_resident_model(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(
            "usage: cargo run -p xtask -- bench-qwen36-resident-model SNAPSHOT [options]".into(),
        );
    }
    run_bench_device(root, "qwen36-resident-model", arguments)
}

fn bench_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    build_sm120_for_performance(root)?;
    run_prebuilt_performance_suite(root, suite, arguments)
}

fn run_prebuilt_performance_suite(
    root: &Path,
    suite: PerformanceSuite,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    run_prebuilt_bench_device(root, suite.name(), &[suite.resource_baseline()], arguments)
}

fn run_optimization_benchmark(
    root: &Path,
    suite: OptimizationSuite,
    snapshot: Option<&OsStr>,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    if let OptimizationSuite::Leaf(leaf) = suite {
        return run_prebuilt_performance_suite(root, leaf, arguments);
    }
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let snapshot =
        snapshot.ok_or_else(|| format!("{} requires the admitted snapshot path", suite.name()))?;
    let mut baselines = Vec::new();
    for baseline in suite.resource_baselines() {
        baselines.extend_from_slice(&fs::read(root.join(baseline))?);
    }
    let mut command = Command::new(executable);
    command
        .arg(suite.name())
        .arg(snapshot)
        .args(arguments)
        .env("TUISKO_GENERATOR_BASELINE_SHA256", sha256(&baselines));
    run_visible(&mut command)
}

fn profile(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let [scope, snapshot, remaining @ ..] = arguments else {
        return Err("usage: cargo run -p xtask -- profile <resident-model|resident-prefill> SNAPSHOT [--batch B | --tokens T] [--replays N] [--tool nsys|ncu] [--kernel REGEX] [--output-dir PATH]".into());
    };
    let prefill = match scope.to_str() {
        Some("resident-model") => false,
        Some("resident-prefill") => true,
        _ => {
            return Err(format!("unknown profile scope `{}`", scope.to_string_lossy()).into());
        }
    };
    let mut batch = 1u32;
    let mut tokens = 1_024u32;
    let mut replays = 3u64;
    let mut tool = "nsys";
    let mut kernel = None;
    let mut output_dir = None;
    let mut options = remaining.iter();
    while let Some(argument) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| format!("`{}` requires a value", argument.to_string_lossy()))?;
        match argument.to_str().ok_or("profile argument is not UTF-8")? {
            "--batch" if prefill => {
                return Err("resident-prefill profiling does not accept `--batch`".into());
            }
            "--batch" => batch = value.to_str().ok_or("batch is not UTF-8")?.parse()?,
            "--tokens" if !prefill => {
                return Err("resident-model profiling does not accept `--tokens`".into());
            }
            "--tokens" => tokens = value.to_str().ok_or("tokens is not UTF-8")?.parse()?,
            "--replays" => replays = value.to_str().ok_or("replays is not UTF-8")?.parse()?,
            "--tool" => tool = value.to_str().ok_or("profile tool is not UTF-8")?,
            "--kernel" => kernel = Some(value.to_str().ok_or("kernel filter is not UTF-8")?),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            option => return Err(format!("unknown profile option `{option}`").into()),
        }
    }
    if !(1..=8).contains(&batch) || replays == 0 {
        return Err("resident profile requires `--batch 1..=8` and nonzero `--replays`".into());
    }
    if prefill && tokens != 1_024 {
        return Err("resident prefill profile currently requires `--tokens 1024`".into());
    }
    if !matches!(tool, "nsys" | "ncu") {
        return Err("resident profile tool must be `nsys` or `ncu`".into());
    }
    if tool == "ncu" && kernel.is_none() {
        return Err("resident `ncu` profiling requires `--kernel REGEX`".into());
    }

    build_sm120_for_performance(root)?;
    let executable = root
        .join(CUDA_OXIDE_BUILD_TARGET)
        .join("release/bench-device");
    if !executable.is_file() {
        return Err(format!(
            "benchmark executable is missing at {}",
            executable.display()
        )
        .into());
    }
    let stem = if prefill {
        format!("resident-prefill-t{tokens}")
    } else {
        format!("resident-model-b{batch}")
    };
    let output_dir = output_dir.unwrap_or_else(|| root.join(format!("target/profiles/{stem}")));
    fs::create_dir_all(&output_dir)?;
    let graph_dot = output_dir.join(format!("{stem}-graph.dot"));
    let manifest = output_dir.join(format!("{stem}-semantic.json"));
    let profile_prefix = output_dir.join(format!("{stem}-{tool}"));
    let warmup_launches = if tool == "ncu" { 1 } else { 16 };
    let profile_arguments = [
        if prefill {
            "profile-resident-prefill".into()
        } else {
            "profile-resident-model".into()
        },
        snapshot.clone(),
        if prefill {
            "--tokens".into()
        } else {
            "--batch".into()
        },
        if prefill {
            tokens.to_string().into()
        } else {
            batch.to_string().into()
        },
        "--warmup-launches".into(),
        warmup_launches.to_string().into(),
        "--captured-replays".into(),
        replays.to_string().into(),
        "--graph-dot".into(),
        graph_dot.as_os_str().to_os_string(),
        "--manifest".into(),
        manifest.as_os_str().to_os_string(),
    ];
    let tool_path = cuda_tool(tool);
    let mut command = Command::new(&tool_path);
    if tool == "nsys" {
        command.args([
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
        ]);
        command.arg("--output").arg(&profile_prefix);
    } else {
        command.args([
            "--profile-from-start=off",
            "--target-processes=all",
            "--set=full",
            "--launch-count=1",
            "--force-overwrite",
        ]);
        command
            .arg("--kernel-name")
            .arg(kernel.expect("ncu kernel filter checked"))
            .arg("--export")
            .arg(&profile_prefix);
    }
    command.arg(&executable).args(&profile_arguments);
    run_visible(&mut command)?;

    if tool == "nsys" {
        let report = profile_prefix.with_extension("nsys-rep");
        let sqlite = output_dir.join(format!("{stem}-nsys.sqlite"));
        run_visible(
            Command::new(&tool_path)
                .args(["export", "--type=sqlite", "--force-overwrite=true"])
                .arg("--output")
                .arg(&sqlite)
                .arg(&report),
        )?;
        postprocess_resident_nsys(&sqlite, &manifest, &output_dir, &stem)?;
    }

    let tool_identity = run_captured(Command::new(&tool_path).arg("--version"))?;
    let git_commit = command_text("git", &["-C", path_text(root)?, "rev-parse", "HEAD"])?;
    let git_status = command_text("git", &["-C", path_text(root)?, "status", "--short"])?;
    let device = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw.instant",
            "--format=csv,noheader",
        ],
    )?;
    let metadata = serde_json::json!({
        "schema_version": 1,
        "scope": if prefill { "resident-prefill" } else { "resident-model" },
        "batch_size": (!prefill).then_some(batch),
        "prompt_tokens": prefill.then_some(tokens),
        "captured_replays": replays,
        "warmup_launches": warmup_launches,
        "tool": tool,
        "tool_path": tool_path,
        "tool_identity": tool_identity.trim(),
        "binary_sha256": sha256(&fs::read(&executable)?),
        "git_commit": git_commit.trim(),
        "git_status": git_status.lines().collect::<Vec<_>>(),
        "device_after_capture": device.trim(),
        "graph_dot": graph_dot,
        "semantic_manifest": manifest,
    });
    let metadata_path = output_dir.join(format!("{stem}-{tool}-metadata.json"));
    let mut json = serde_json::to_vec_pretty(&metadata)?;
    json.push(b'\n');
    fs::write(&metadata_path, json)?;
    println!("profile artifacts: {}", output_dir.display());

    Ok(())
}

#[derive(Clone)]
struct ProfileKernelSample {
    start_ns: u64,
    end_ns: u64,
}

struct ProfileNodeSamples {
    graph_node_id: u64,
    kernel: String,
    samples: Vec<ProfileKernelSample>,
}

struct SemanticProfileNode {
    layer: Option<usize>,
    component: String,
    source_route: String,
    kernel_family: String,
}

fn postprocess_resident_nsys(
    sqlite: &Path,
    manifest: &Path,
    output_dir: &Path,
    stem: &str,
) -> Result<(), Box<dyn Error>> {
    let manifest_json: serde_json::Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let expected_nodes = manifest_json["graph_kernel_nodes"]
        .as_u64()
        .ok_or("resident semantic manifest omits graph_kernel_nodes")?
        as usize;
    let expected_replays = manifest_json["captured_replays"]
        .as_u64()
        .ok_or("resident semantic manifest omits captured_replays")?
        as usize;
    let semantic_nodes = semantic_profile_nodes(&manifest_json)?;
    if semantic_nodes.len() != expected_nodes {
        return Err(format!(
            "semantic manifest expands to {} nodes, expected {expected_nodes}",
            semantic_nodes.len()
        )
        .into());
    }
    let connection = Connection::open_with_flags(sqlite, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let graph_id = connection
        .query_row(
            "SELECT graphId
             FROM CUPTI_ACTIVITY_KIND_KERNEL
             WHERE graphId IS NOT NULL
             GROUP BY graphId
             HAVING COUNT(DISTINCT graphNodeId) = ?1
             ORDER BY graphId
             LIMIT 1",
            [expected_nodes as i64],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .ok_or_else(|| format!("Nsight SQLite has no graph with {expected_nodes} kernel nodes"))?;
    let mut statement = connection.prepare(
        "SELECT k.graphNodeId, k.start, k.end, s.value
         FROM CUPTI_ACTIVITY_KIND_KERNEL k
         JOIN StringIds s ON s.id = k.shortName
         WHERE k.graphId = ?1
         ORDER BY k.graphNodeId, k.start",
    )?;
    let rows = statement.query_map([graph_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    let mut by_node = BTreeMap::<u64, ProfileNodeSamples>::new();
    for row in rows {
        let (graph_node_id, start_ns, end_ns, kernel) = row?;
        let graph_node_id = u64::try_from(graph_node_id)?;
        let start_ns = u64::try_from(start_ns)?;
        let end_ns = u64::try_from(end_ns)?;
        let entry = by_node
            .entry(graph_node_id)
            .or_insert_with(|| ProfileNodeSamples {
                graph_node_id,
                kernel: kernel.clone(),
                samples: Vec::new(),
            });
        if entry.kernel != kernel {
            return Err(format!(
                "graph node {graph_node_id} changed kernel from `{}` to `{kernel}`",
                entry.kernel
            )
            .into());
        }
        entry.samples.push(ProfileKernelSample { start_ns, end_ns });
    }
    let nodes = by_node.into_values().collect::<Vec<_>>();
    if nodes.len() != expected_nodes {
        return Err(format!(
            "Nsight graph {graph_id} contains {} kernel nodes, expected {expected_nodes}",
            nodes.len()
        )
        .into());
    }
    for (ordinal, (node, semantic)) in nodes.iter().zip(&semantic_nodes).enumerate() {
        if node.samples.len() != expected_replays {
            return Err(format!(
                "graph node {} has {} captured samples, expected {expected_replays}",
                node.graph_node_id,
                node.samples.len()
            )
            .into());
        }
        if !node.kernel.contains(&semantic.kernel_family) {
            return Err(format!(
                "semantic node {} expects kernel family `{}`, Nsight observed `{}`",
                ordinal + 1,
                semantic.kernel_family,
                node.kernel
            )
            .into());
        }
    }

    let mut graph_spans = vec![0.0; expected_replays];
    let mut graph_kernel_sums = vec![0.0; expected_replays];
    for replay in 0..expected_replays {
        let start = nodes
            .iter()
            .map(|node| node.samples[replay].start_ns)
            .min()
            .expect("nonempty graph node inventory");
        let end = nodes
            .iter()
            .map(|node| node.samples[replay].end_ns)
            .max()
            .expect("nonempty graph node inventory");
        graph_spans[replay] = (end - start) as f64 / 1_000.0;
        graph_kernel_sums[replay] = nodes
            .iter()
            .map(|node| {
                let sample = &node.samples[replay];
                (sample.end_ns - sample.start_ns) as f64 / 1_000.0
            })
            .sum();
    }
    write_profile_node_csv(output_dir, stem, &nodes, &semantic_nodes)?;
    write_profile_stage_csv(
        output_dir,
        stem,
        &nodes,
        &semantic_nodes,
        expected_replays,
        mean(&graph_spans),
    )?;
    write_profile_layer_csv(
        output_dir,
        stem,
        &nodes,
        &semantic_nodes,
        expected_replays,
        mean(&graph_spans),
    )?;
    let mut replay_csv = String::from("replay,graph_span_us,kernel_sum_us,gaps_us,gap_percent\n");
    for replay in 0..expected_replays {
        let gaps = graph_spans[replay] - graph_kernel_sums[replay];
        replay_csv.push_str(&format!(
            "{},{:.3},{:.3},{:.3},{:.4}\n",
            replay + 1,
            graph_spans[replay],
            graph_kernel_sums[replay],
            gaps,
            gaps / graph_spans[replay] * 100.0,
        ));
    }
    fs::write(
        output_dir.join(format!("{stem}-replay-timings.csv")),
        replay_csv,
    )?;
    println!(
        "profile closure: {:.3} us graph, {:.3} us kernels, {:.3} us gaps ({:.2}%)",
        mean(&graph_spans),
        mean(&graph_kernel_sums),
        mean(&graph_spans) - mean(&graph_kernel_sums),
        (mean(&graph_spans) - mean(&graph_kernel_sums)) / mean(&graph_spans) * 100.0,
    );

    Ok(())
}

fn semantic_profile_nodes(
    manifest: &serde_json::Value,
) -> Result<Vec<SemanticProfileNode>, Box<dyn Error>> {
    let stages = manifest["stages"]
        .as_array()
        .ok_or("resident semantic manifest omits stages")?;
    let mut nodes = Vec::new();
    for stage in stages {
        let layer = stage["layer"].as_u64().map(|layer| layer as usize);
        let component = stage["component"]
            .as_str()
            .ok_or("resident semantic stage omits component")?;
        let source_route = stage["source_route"]
            .as_str()
            .ok_or("resident semantic stage omits source_route")?;
        let families = stage["kernel_families"]
            .as_array()
            .ok_or("resident semantic stage omits kernel_families")?;
        for family in families {
            nodes.push(SemanticProfileNode {
                layer,
                component: component.to_string(),
                source_route: source_route.to_string(),
                kernel_family: family
                    .as_str()
                    .ok_or("resident semantic kernel family is not text")?
                    .to_string(),
            });
        }
    }

    Ok(nodes)
}

fn write_profile_node_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "ordinal,graph_node_id,layer,component,source_route,kernel,mean_us,median_us,min_us,max_us\n",
    );
    for (ordinal, (node, semantic)) in nodes.iter().zip(semantic).enumerate() {
        let values = node
            .samples
            .iter()
            .map(|sample| (sample.end_ns - sample.start_ns) as f64 / 1_000.0)
            .collect::<Vec<_>>();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}\n",
            ordinal + 1,
            node.graph_node_id,
            semantic
                .layer
                .map_or_else(|| "-".to_string(), |layer| layer.to_string()),
            semantic.component,
            semantic.source_route,
            csv_field(&node.kernel),
            mean(&values),
            median(&values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
        ));
    }
    fs::write(output_dir.join(format!("{stem}-node-timings.csv")), csv)?;
    Ok(())
}

fn write_profile_stage_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
    replays: usize,
    graph_mean_us: f64,
) -> Result<(), Box<dyn Error>> {
    let mut groups = Vec::<(Option<usize>, String, String, Vec<f64>, usize)>::new();
    for (node, semantic) in nodes.iter().zip(semantic) {
        let new_group = groups.last().is_none_or(|(layer, component, route, _, _)| {
            *layer != semantic.layer
                || component != &semantic.component
                || route != &semantic.source_route
        });
        if new_group {
            groups.push((
                semantic.layer,
                semantic.component.clone(),
                semantic.source_route.clone(),
                vec![0.0; replays],
                0,
            ));
        }
        let group = groups.last_mut().expect("stage group was inserted");
        group.4 += 1;
        for replay in 0..replays {
            group.3[replay] +=
                (node.samples[replay].end_ns - node.samples[replay].start_ns) as f64 / 1_000.0;
        }
    }
    let mut csv = String::from(
        "ordinal,layer,component,source_route,kernel_nodes,mean_us,median_us,min_us,max_us,graph_share_percent\n",
    );
    for (ordinal, (layer, component, route, values, kernel_nodes)) in groups.iter().enumerate() {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.4}\n",
            ordinal + 1,
            layer.map_or_else(|| "-".to_string(), |layer| layer.to_string()),
            component,
            route,
            kernel_nodes,
            mean(values),
            median(values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
            mean(values) / graph_mean_us * 100.0,
        ));
    }
    fs::write(output_dir.join(format!("{stem}-stage-timings.csv")), csv)?;
    Ok(())
}

fn write_profile_layer_csv(
    output_dir: &Path,
    stem: &str,
    nodes: &[ProfileNodeSamples],
    semantic: &[SemanticProfileNode],
    replays: usize,
    graph_mean_us: f64,
) -> Result<(), Box<dyn Error>> {
    let mut layers = BTreeMap::<i32, (String, Vec<f64>)>::new();
    for (node, semantic) in nodes.iter().zip(semantic) {
        let layer = semantic.layer.map_or_else(
            || {
                if semantic.component == "input_norm" {
                    -1
                } else {
                    64
                }
            },
            |layer| layer as i32,
        );
        let entry = layers
            .entry(layer)
            .or_insert_with(|| (semantic.source_route.clone(), vec![0.0; replays]));
        for replay in 0..replays {
            entry.1[replay] +=
                (node.samples[replay].end_ns - node.samples[replay].start_ns) as f64 / 1_000.0;
        }
    }
    let mut csv =
        String::from("layer,source_route,mean_us,median_us,min_us,max_us,graph_share_percent\n");
    for (layer, (route, values)) in layers {
        let label = match layer {
            -1 => "input".to_string(),
            64 => "endpoint".to_string(),
            _ => layer.to_string(),
        };
        csv.push_str(&format!(
            "{label},{route},{:.3},{:.3},{:.3},{:.3},{:.4}\n",
            mean(&values),
            median(&values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(0.0, f64::max),
            mean(&values) / graph_mean_us * 100.0,
        ));
    }
    fs::write(output_dir.join(format!("{stem}-layer-timings.csv")), csv)?;
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn perf(root: &Path, arguments: &[std::ffi::OsString]) -> Result<(), Box<dyn Error>> {
    let Some(mode) = arguments.first() else {
        return Err(
            "usage: cargo run -p xtask -- perf <smoke|leaf|energy|gate|candidate|check|bless|iterate|diagnose-diff> ...".into(),
        );
    };
    let mode = mode.to_str().ok_or("perf mode is not UTF-8")?;
    if mode == "diagnose-diff" {
        return diagnose_performance_report(root, &arguments[1..]);
    }
    if mode == "iterate" {
        let options = parse_performance_iteration(&arguments[1..])?;
        return run_performance_iteration(root, options);
    }
    if mode == "bless" {
        let suite = arguments
            .get(1)
            .ok_or("usage: cargo run -p xtask -- perf bless SUITE [SNAPSHOT]")?;
        let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
        let snapshot = if suite.requires_snapshot() {
            let [_, _, snapshot] = arguments else {
                return Err(format!("perf bless {} requires SNAPSHOT", suite.name()).into());
            };
            Some(snapshot.as_os_str())
        } else {
            if arguments.len() != 2 {
                return Err(format!("perf bless {} takes no SNAPSHOT", suite.name()).into());
            }
            None
        };
        require_performance_device_idle()?;
        return bless_optimization_suite(root, suite, snapshot);
    }
    if matches!(mode, "candidate" | "check") {
        let suite = arguments.get(1).ok_or(
            "usage: cargo run -p xtask -- perf candidate|check SUITE [SNAPSHOT] [options]",
        )?;
        let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
        let cone = suite.dependency_cone();
        let needs_snapshot = cone.iter().any(|suite| suite.requires_snapshot());
        let mut remaining = &arguments[2..];
        let snapshot = if needs_snapshot {
            let (snapshot, options) = remaining.split_first().ok_or_else(|| {
                format!(
                    "perf {mode} {} requires SNAPSHOT for its composed dependency cone",
                    suite.name()
                )
            })?;
            remaining = options;
            Some(snapshot.as_os_str())
        } else {
            None
        };
        if mode == "check" && !remaining.is_empty() {
            return Err("`perf check` uses the complete authoritative suite defaults".into());
        }
        if mode == "check" {
            preflight_performance_baselines(
                root,
                cone.iter().map(|suite| suite.performance_baseline()),
            )?;
        }
        require_performance_device_idle()?;
        return run_optimization_cone(root, mode, suite, &cone, snapshot, remaining);
    }
    let gate_snapshot = match (mode, arguments) {
        ("gate", [_, snapshot]) => Some(snapshot.as_os_str()),
        ("gate", [_]) => {
            return Err("`perf gate` requires SNAPSHOT for source-backed MTP qualification".into());
        }
        (_, [_]) => None,
        _ => return Err(format!("`perf {mode}` takes no additional arguments").into()),
    };

    let options = match mode {
        "smoke" => vec!["--samples".into(), "3".into()],
        "leaf" | "gate" => Vec::new(),
        "energy" => vec!["--energy-seconds".into(), "2".into()],
        _ => return Err(format!("unknown perf mode `{mode}`").into()),
    };
    if mode == "gate" {
        preflight_performance_baselines(
            root,
            PERFORMANCE_SUITES
                .into_iter()
                .map(PerformanceSuite::performance_baseline),
        )?;
    }
    require_performance_device_idle()?;
    if mode == "gate" {
        qualify_host(root)?;
        for suite in PERFORMANCE_SUITES {
            suite.qualify(root, gate_snapshot)?;
        }
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    run_performance_suites(root, mode, &options, mode == "gate")
}

struct PerformanceIterationOptions {
    suite: PerformanceSuite,
    snapshot: Option<std::ffi::OsString>,
    batch_size: u32,
    hypothesis: String,
}

fn parse_performance_iteration(
    arguments: &[std::ffi::OsString],
) -> Result<PerformanceIterationOptions, Box<dyn Error>> {
    let Some((suite, remaining)) = arguments.split_first() else {
        return Err(
            "usage: cargo run -p xtask -- perf iterate SUITE [SNAPSHOT] --batch B --hypothesis TEXT"
                .into(),
        );
    };
    let suite = PerformanceSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
    let (snapshot, remaining) = if suite.requires_snapshot() {
        let (snapshot, remaining) = remaining
            .split_first()
            .ok_or_else(|| format!("perf iterate {} requires SNAPSHOT", suite.name()))?;
        if snapshot.to_string_lossy().starts_with("--") {
            return Err(format!("perf iterate {} requires SNAPSHOT", suite.name()).into());
        }
        (Some(snapshot.clone()), remaining)
    } else {
        (None, remaining)
    };
    let mut batch_size = None;
    let mut hypothesis = None;
    let mut options = remaining.iter();
    while let Some(option) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| format!("`{}` requires a value", option.to_string_lossy()))?;
        match option.to_str().ok_or("perf iterate option is not UTF-8")? {
            "--batch" if batch_size.is_none() => {
                batch_size = Some(value.to_str().ok_or("batch is not UTF-8")?.parse()?);
            }
            "--hypothesis" if hypothesis.is_none() => {
                hypothesis = Some(
                    value
                        .to_str()
                        .ok_or("performance hypothesis is not UTF-8")?
                        .trim()
                        .to_string(),
                );
            }
            option => {
                return Err(format!("unknown or duplicate perf iterate option `{option}`").into());
            }
        }
    }
    let batch_size = batch_size.ok_or("perf iterate requires `--batch B`")?;
    if !(1..=8).contains(&batch_size) {
        return Err("perf iterate requires `--batch 1..=8`".into());
    }
    let hypothesis = hypothesis.ok_or("perf iterate requires `--hypothesis TEXT`")?;
    if hypothesis.is_empty() {
        return Err("perf iterate requires a nonempty hypothesis".into());
    }

    Ok(PerformanceIterationOptions {
        suite,
        snapshot,
        batch_size,
        hypothesis,
    })
}

fn run_performance_iteration(
    root: &Path,
    options: PerformanceIterationOptions,
) -> Result<(), Box<dyn Error>> {
    let suite = options.suite;
    let baseline = root.join(suite.performance_baseline());
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let mut recorder = perf_iteration::IterationRecorder::start(
        root,
        suite.name(),
        options.batch_size,
        options.hypothesis.clone(),
        device_inputs.clone(),
        &baseline,
    )?;

    let started = Instant::now();
    let device_identity = match require_performance_device_idle()
        .and_then(|()| performance_device_identity_sha256())
    {
        Ok(identity) => {
            recorder.record_stage("preflight", "passed", started.elapsed());
            identity
        }
        Err(error) => {
            recorder.record_stage("preflight", "refused", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    };

    let started = Instant::now();
    match perf_artifact::qualification_is_current(
        root,
        suite.name(),
        &device_inputs,
        &device_identity,
    ) {
        Ok(true) => recorder.record_stage("qualification", "reused", started.elapsed()),
        Ok(false) => {
            if let Err(error) = suite
                .qualify(root, options.snapshot.as_deref())
                .and_then(|()| {
                    perf_artifact::record_qualification(
                        root,
                        suite.name(),
                        device_inputs.clone(),
                        device_identity.clone(),
                    )
                })
            {
                recorder.record_stage("qualification", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
            recorder.record_stage("qualification", "passed", started.elapsed());
        }
        Err(error) => {
            recorder.record_stage("qualification", "failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    }

    let started = Instant::now();
    let resource_baselines =
        match perf_artifact::resource_baselines_sha256(root, SM120_RESOURCE_BASELINES) {
            Ok(value) => value,
            Err(error) => {
                recorder.record_stage("build", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
        };
    match perf_artifact::local_build_is_current(
        root,
        &device_inputs,
        &resource_baselines,
        CUDA_OXIDE_REVISION,
    ) {
        Ok(true) => recorder.record_stage("build", "reused", started.elapsed()),
        Ok(false) => {
            if let Err(error) = build_sm120(root) {
                recorder.record_stage("build", "failed", started.elapsed());
                return fail_performance_iteration(recorder, error);
            }
            recorder.record_stage("build", "passed", started.elapsed());
        }
        Err(error) => {
            recorder.record_stage("build", "failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    }

    let started = Instant::now();
    let benchmark_arguments = [
        "--batch".into(),
        options.batch_size.to_string().into(),
        "--json".into(),
        recorder.report_path().into_os_string(),
    ];
    if let Err(error) = wait_for_device_idle()
        .and_then(|()| run_prebuilt_performance_suite(root, suite, &benchmark_arguments))
    {
        recorder.record_stage("benchmark", "refused_or_failed", started.elapsed());
        return fail_performance_iteration(recorder, error);
    }
    recorder.record_stage("benchmark", "passed", started.elapsed());

    let started = Instant::now();
    let diagnostic = match performance::diagnose(&recorder.report_path(), &baseline) {
        Ok(diagnostic) => diagnostic,
        Err(error) => {
            recorder.record_stage("comparison", "refused_or_failed", started.elapsed());
            return fail_performance_iteration(recorder, error);
        }
    };
    recorder.record_stage("comparison", "diagnostic_only", started.elapsed());
    let output = recorder.succeed(diagnostic)?;
    println!(
        "diagnostic optimization iteration preserved at {}",
        output.display()
    );
    Ok(())
}

fn fail_performance_iteration(
    recorder: perf_iteration::IterationRecorder,
    error: Box<dyn Error>,
) -> Result<(), Box<dyn Error>> {
    let error_text = error.to_string();
    match recorder.fail(error.as_ref()) {
        Ok(output) => eprintln!(
            "failed or refused optimization iteration preserved at {}",
            output.display()
        ),
        Err(record_error) => {
            return Err(format!(
                "{error_text}; additionally failed to preserve the iteration record: {record_error}"
            )
            .into());
        }
    }
    Err(error)
}

fn diagnose_performance_report(
    root: &Path,
    arguments: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let (suite, report, output) = match arguments {
        [suite, report] => (suite, report, None),
        [suite, report, option, output] if option == "--json" => (suite, report, Some(output)),
        _ => {
            return Err(
                "usage: cargo run -p xtask -- perf diagnose-diff SUITE REPORT [--json OUTPUT]"
                    .into(),
            );
        }
    };
    let suite = OptimizationSuite::parse(suite.to_str().ok_or("perf suite is not UTF-8")?)?;
    let report = resolve_repository_path(root, report);
    let output = match output {
        Some(path) => resolve_target_output(root, path)?,
        None => root.join(format!(
            "target/benchmarks/perf-diagnostic/{}.json",
            suite.name()
        )),
    };
    let diagnostic = performance::diagnose(&report, &root.join(suite.performance_baseline()))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(&diagnostic)?;
    json.push(b'\n');
    fs::write(&output, json)?;
    println!("diagnostic comparison JSON: {}", output.display());
    Ok(())
}

fn resolve_repository_path(root: &Path, path: &OsStr) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn resolve_target_output(root: &Path, path: &OsStr) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        || !path.starts_with("target")
    {
        return Err("diagnostic output must be a repository-relative path under `target/`".into());
    }
    Ok(root.join(path))
}

fn preflight_performance_baselines(
    root: &Path,
    baselines: impl IntoIterator<Item = &'static str>,
) -> Result<(), Box<dyn Error>> {
    let mut failures = Vec::new();
    for baseline in baselines {
        if let Err(error) = performance::preflight_baseline(&root.join(baseline)) {
            failures.push(format!("{baseline}: {error}"));
        }
    }
    if failures.is_empty() {
        return Ok(());
    }

    Err(format!(
        "performance baselines are missing or invalid; bless each suite before comparing:\n{}",
        failures.join("\n")
    )
    .into())
}

fn run_performance_suites(
    root: &Path,
    mode: &str,
    options: &[std::ffi::OsString],
    compare: bool,
) -> Result<(), Box<dyn Error>> {
    for (index, suite) in PERFORMANCE_SUITES.into_iter().enumerate() {
        if index != 0 {
            wait_for_device_idle()?;
        }
        let report = performance_report_path(root, mode, suite);
        let mut arguments = options.to_vec();
        arguments.push("--json".into());
        arguments.push(path_text(&report)?.into());
        run_prebuilt_performance_suite(root, suite, &arguments)?;
    }
    if compare {
        for suite in PERFORMANCE_SUITES {
            let report = performance_report_path(root, mode, suite);
            performance::compare(&report, &root.join(suite.performance_baseline()))?;
        }
    }

    Ok(())
}

fn run_optimization_cone(
    root: &Path,
    mode: &str,
    root_suite: OptimizationSuite,
    cone: &[OptimizationSuite],
    snapshot: Option<&OsStr>,
    options: &[std::ffi::OsString],
) -> Result<(), Box<dyn Error>> {
    let device_inputs = perf_artifact::device_input_sha256(root)?;
    let device_identity = performance_device_identity_sha256()?;
    if mode == "check" {
        let mut qualified = Vec::new();
        for suite in cone.iter().copied() {
            let authority = if matches!(
                suite,
                OptimizationSuite::ResidentPrefill | OptimizationSuite::ResidentLongContextModel
            ) {
                OptimizationSuite::ResidentModel
            } else {
                suite
            };
            if !qualified.contains(&authority) {
                authority.qualify(root, snapshot)?;
                if let OptimizationSuite::Leaf(leaf) = authority {
                    perf_artifact::record_qualification(
                        root,
                        leaf.name(),
                        device_inputs.clone(),
                        device_identity.clone(),
                    )?;
                }
                qualified.push(authority);
            }
        }
    } else {
        root_suite.qualify(root, snapshot)?;
        if let OptimizationSuite::Leaf(leaf) = root_suite {
            perf_artifact::record_qualification(root, leaf.name(), device_inputs, device_identity)?;
        }
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    for (index, suite) in cone.iter().copied().enumerate() {
        if index != 0 {
            wait_for_device_idle()?;
        }
        let report = optimization_report_path(root, mode, root_suite, suite);
        let mut arguments = options.to_vec();
        arguments.push("--json".into());
        arguments.push(path_text(&report)?.into());
        run_optimization_benchmark(root, suite, snapshot, &arguments)?;
        if mode == "check" {
            performance::compare(&report, &root.join(suite.performance_baseline()))?;
        }
    }

    Ok(())
}

fn wait_for_device_idle() -> Result<(), Box<dyn Error>> {
    let timeout = Duration::from_secs(60);
    let deadline = Instant::now() + timeout;
    loop {
        let (utilization, memory_mib, pids) = device_idle_evidence("device idle wait")?;
        if device_is_idle(utilization, memory_mib, &pids) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "device zero remained busy for {} seconds: utilization={utilization}%, memory={memory_mib} MiB, compute processes={pids:?}",
                timeout.as_secs()
            )
            .into());
        }

        std::thread::sleep(Duration::from_millis(100));
    }
}

fn require_performance_device_idle() -> Result<(), Box<dyn Error>> {
    wait_for_device_idle()
}

fn require_device_idle(activity: &str) -> Result<(), Box<dyn Error>> {
    let (utilization, memory_mib, pids) = device_idle_evidence(activity)?;
    if utilization >= IDLE_DEVICE_UTILIZATION_LIMIT_PERCENT
        || memory_mib > MAX_IDLE_DEVICE_MEMORY_MIB
    {
        return Err(format!(
            "device zero is busy before {activity}: utilization={utilization}%, memory={memory_mib} MiB"
        )
        .into());
    }
    if !pids.is_empty() {
        return Err(format!(
            "device zero has foreign compute processes before {activity}: {pids:?}"
        )
        .into());
    }

    Ok(())
}

fn device_idle_evidence(activity: &str) -> Result<(u32, u64, Vec<u32>), Box<dyn Error>> {
    if env::var_os("CUDA_VISIBLE_DEVICES").is_some_and(|value| value != "0") {
        return Err(
            format!("{activity} requires CUDA_VISIBLE_DEVICES to be unset or exactly `0`").into(),
        );
    }
    let row = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let (device, utilization, memory_mib) = parse_performance_device_sample(&row)?;
    let expected = gpu_target::GpuTarget::Sm120.device_name();
    if device != expected {
        return Err(format!(
            "{activity} requires device zero to be `{expected}`, found `{device}`"
        )
        .into());
    }
    let processes = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let pids = parse_compute_pids(&processes)?;
    Ok((utilization, memory_mib, pids))
}

fn device_is_idle(utilization: u32, memory_mib: u64, pids: &[u32]) -> bool {
    utilization < IDLE_DEVICE_UTILIZATION_LIMIT_PERCENT
        && memory_mib <= MAX_IDLE_DEVICE_MEMORY_MIB
        && pids.is_empty()
}

fn performance_device_identity_sha256() -> Result<String, Box<dyn Error>> {
    let identity = command_text(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )?;
    Ok(sha256(identity.trim().as_bytes()))
}

fn parse_performance_device_sample(row: &str) -> Result<(String, u32, u64), Box<dyn Error>> {
    let fields = row.trim().split(',').map(str::trim).collect::<Vec<_>>();
    let [device, utilization, memory_mib] = fields.as_slice() else {
        return Err(format!(
            "unexpected nvidia-smi performance preflight row `{}`",
            row.trim()
        )
        .into());
    };

    Ok((
        (*device).to_string(),
        utilization.parse()?,
        memory_mib.parse()?,
    ))
}

fn parse_compute_pids(output: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != "No running processes found")
        .map(|line| {
            line.parse().map_err(|error| {
                format!("unexpected nvidia-smi compute PID `{line}`: {error}").into()
            })
        })
        .collect()
}

fn sass_function_body<'a>(sass: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("Function : {name}");
    let start = sass
        .match_indices(&marker)
        .map(|(index, _)| index + marker.len())
        .find(|&end| sass[end..].lines().next().unwrap_or("").trim().is_empty())?;
    let body = &sass[start..];

    Some(body.split("\n\t\tFunction :").next().unwrap_or(body))
}

fn bless_optimization_suite(
    root: &Path,
    suite: OptimizationSuite,
    snapshot: Option<&OsStr>,
) -> Result<(), Box<dyn Error>> {
    suite.qualify(root, snapshot)?;
    if let OptimizationSuite::Leaf(leaf) = suite {
        perf_artifact::record_qualification(
            root,
            leaf.name(),
            perf_artifact::device_input_sha256(root)?,
            performance_device_identity_sha256()?,
        )?;
    }
    build_sm120(root)?;
    wait_for_device_idle()?;
    let report = optimization_report_path(root, "bless", suite, suite);
    run_optimization_benchmark(
        root,
        suite,
        snapshot,
        &["--json".into(), path_text(&report)?.into()],
    )?;
    performance::bless(&report, &root.join(suite.performance_baseline()))
}

fn performance_report_path(root: &Path, mode: &str, suite: PerformanceSuite) -> PathBuf {
    root.join(format!(
        "target/benchmarks/perf-{mode}/{}.json",
        suite.name()
    ))
}

fn optimization_report_path(
    root: &Path,
    mode: &str,
    root_suite: OptimizationSuite,
    suite: OptimizationSuite,
) -> PathBuf {
    root.join(format!(
        "target/benchmarks/perf-{mode}/{}/{}.json",
        root_suite.name(),
        suite.name()
    ))
}

fn run_oxide(root: &Path, arguments: &[&str]) -> Result<(), Box<dyn Error>> {
    run_oxide_with_env(root, arguments, None)
}

fn run_oxide_with_env(
    root: &Path,
    arguments: &[&str],
    environment: Option<(&str, &OsStr)>,
) -> Result<(), Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    require_cuda_oxide_revision(&source)?;
    let wrapper = root.join("target/cuda-oxide-driver/debug/cargo-oxide");
    if !wrapper.is_file() {
        return Err(
            "cuda-oxide is not bootstrapped; run `cargo run -p xtask -- bootstrap-cuda-oxide`"
                .into(),
        );
    }
    let backend = local_backend(root)?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    fs::create_dir_all(root.join("target/tmp"))?;
    let mut command = Command::new(wrapper);
    command
        .args(arguments)
        .current_dir(root)
        .env("CARGO_HOME", task_cargo_home(root))
        .env("CUDA_OXIDE_BACKEND", backend)
        .env("CUDA_OXIDE_SOURCE", source)
        .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
        .env_remove("RUSTFLAGS")
        .env("TMPDIR", root.join("target/tmp"));
    if let Some((name, value)) = environment {
        command.env(name, value);
    }

    run_visible(&mut command)
}

/// Canonical SM120 device qualification argv. `trailing` follows the filter verbatim because
/// test-harness argument order is part of the suite contract.
fn qualification_test_arguments<'a>(test_filter: &'a str, trailing: &[&'a str]) -> Vec<&'a str> {
    let mut arguments = vec![
        "test",
        "--arch",
        "sm_120a",
        "--cargo-target-dir",
        CUDA_OXIDE_TEST_TARGET,
        "--device-codegen-crate",
        SM120_DEVICE_CODEGEN_CRATES,
        "--",
        "--package",
        "tuisko-qual",
        "--release",
        "--lib",
        "--",
        test_filter,
    ];
    arguments.extend_from_slice(trailing);
    arguments
}

/// Run one canonical `tuisko-qual` device test invocation. Gates stay with the
/// caller: a suite's gate list is its contract, not boilerplate.
fn run_qualification_test(
    root: &Path,
    test_filter: &str,
    trailing: &[&str],
    environment: Option<(&str, &OsStr)>,
) -> Result<(), Box<dyn Error>> {
    #[cfg(test)]
    if let Some(intercepted) = dispatch_probe::intercept(dispatch_probe::Spawn::Qualification {
        filter: test_filter.to_owned(),
        trailing: trailing.iter().map(|flag| (*flag).to_owned()).collect(),
        environment: environment.map(|(key, value)| (key.to_owned(), value.to_owned())),
    }) {
        return intercepted;
    }

    let arguments = qualification_test_arguments(test_filter, trailing);
    run_oxide_with_env(root, &arguments, environment)
}

/// Same as `run_oxide` but captures stdout so the caller can locate
/// build artifacts (remote prepare builds with `--no-run`).
fn run_oxide_capture(root: &Path, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let source = local_cuda_oxide_source(root);
    require_cuda_oxide_revision(&source)?;
    let wrapper = root.join("target/cuda-oxide-driver/debug/cargo-oxide");
    if !wrapper.is_file() {
        return Err(
            "cuda-oxide is not bootstrapped; run `cargo run -p xtask -- bootstrap-cuda-oxide`"
                .into(),
        );
    }
    let backend = local_backend(root)?;
    let backend_rustflags = encoded_backend_rustflags(root, &source)?;
    fs::create_dir_all(root.join("target/tmp"))?;
    run_captured(
        Command::new(wrapper)
            .args(arguments)
            .current_dir(root)
            .env("CARGO_HOME", task_cargo_home(root))
            .env("CUDA_OXIDE_BACKEND", backend)
            .env("CUDA_OXIDE_SOURCE", source)
            .env("CARGO_ENCODED_RUSTFLAGS", backend_rustflags)
            .env_remove("RUSTFLAGS")
            .env("TMPDIR", root.join("target/tmp")),
    )
}

/// Runs a command with stdout piped (stderr inherited) and returns stdout.
fn run_captured(command: &mut Command) -> Result<String, Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let output = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(format!("{program} failed with {}", output.status).into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A compiled qual test executable ready to ship to a gate pod.
pub struct RemoteQualify {
    /// Path to the compiled qual test executable.
    pub executable: PathBuf,
    /// Arguments to run the executable with (libtest filters + flags).
    pub test_args: Vec<String>,
}

/// A compiled benchmark executable and its matching resource identity.
pub(crate) struct RemoteBenchmark {
    /// Path to the compiled `bench-device` executable.
    pub executable: PathBuf,
    /// Hash of the suite's checked static-resource baseline.
    pub generator_baseline_sha256: String,
}

/// Locates the already-built benchmark executable and binds it to one suite's resources.
pub(crate) fn prepare_remote_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    suite: PerformanceSuite,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    let resource_baseline = match suite {
        PerformanceSuite::ResidualNorm => gpu.residual_resource_baseline(),
        PerformanceSuite::Nvfp4SwiGlu => gpu
            .nvfp4_swiglu_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no NVFP4 SwiGLU resource baseline", gpu.key()))?,
        PerformanceSuite::Nvfp4Down => gpu
            .nvfp4_down_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no NVFP4 down resource baseline", gpu.key()))?,
        PerformanceSuite::Fp8Qkv => gpu
            .fp8_qkv_resource_baseline()
            .ok_or_else(|| format!("GPU {} has no FP8 QKV resource baseline", gpu.key()))?,
        _ => suite.resource_baseline(),
    };
    prepare_remote_benchmark_with_baselines(root, gpu, &[resource_baseline])
}

/// Locates the composed NVFP4 MLP benchmark and binds it to all leaf resources it launches.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_nvfp4_mlp_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(
        root,
        gpu,
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            NVFP4_SWIGLU_RESOURCE_BASELINE,
            NVFP4_DOWN_RESOURCE_BASELINE,
        ],
    )
}

/// Locates the composed full-attention benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_full_attention_layer_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(
        root,
        gpu,
        &[
            RESIDUAL_NORM_RESOURCE_BASELINE,
            FP8_QKV_RESOURCE_BASELINE,
            ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
            PAGED_GQA_RESOURCE_BASELINE,
            ATTENTION_OUTPUT_RESOURCE_BASELINE,
            FP8_SWIGLU_RESOURCE_BASELINE,
            FP8_DOWN_RESOURCE_BASELINE,
        ],
    )
}

/// Locates the composed MTP-layer benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_mtp_layer_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(root, gpu, MTP_LAYER_RESOURCE_BASELINES)
}

/// Locates the prompt-prime benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_mtp_prompt_prime_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(root, gpu, MTP_PROMPT_PRIME_RESOURCE_BASELINES)
}

/// Locates the resident MTP benchmark and binds every target and MTP leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_resident_mtp_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(root, gpu, RESIDENT_MTP_RESOURCE_BASELINES)
}

/// Locates the complete resident-model benchmark and binds every launched leaf resource.
#[cfg(feature = "remote")]
pub(crate) fn prepare_remote_resident_model_benchmark(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    prepare_remote_benchmark_with_baselines(root, gpu, RESIDENT_MODEL_RESOURCE_BASELINES)
}

fn prepare_remote_benchmark_with_baselines(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    baselines: &[&str],
) -> Result<RemoteBenchmark, Box<dyn Error>> {
    let built = root
        .join(gpu.oxide_build_target())
        .join("release/bench-device");
    if !built.is_file() {
        return Err(format!("benchmark executable is missing at {}", built.display()).into());
    }
    let artifact_name = format!("bench-device-{}", gpu.key());
    let executable = strip_remote_artifact(root, &built, &artifact_name)?;
    let mut resources = Vec::new();
    for baseline in baselines {
        resources.extend_from_slice(&fs::read(root.join(baseline))?);
    }

    Ok(RemoteBenchmark {
        executable,
        generator_baseline_sha256: sha256(&resources),
    })
}

/// Compiles the qual test executable for a remote `qualify-*` gate.
///
/// No GPU is needed locally: the cuda-oxide codegen and ptxas steps run
/// at compile time; the pod only needs the driver to execute the binary.
pub(crate) fn prepare_remote_qualify(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    test_filter: &str,
) -> Result<RemoteQualify, Box<dyn Error>> {
    let test_args = [
        test_filter,
        "--include-ignored",
        "--nocapture",
        "--test-threads=1",
    ];
    let arguments = vec![
        "test".to_string(),
        "--arch".to_string(),
        gpu.oxide_arch().to_string(),
        "--cargo-target-dir".to_string(),
        gpu.oxide_test_target().to_string(),
        "--device-codegen-crate".to_string(),
        gpu.device_codegen_crates().to_string(),
        "--".to_string(),
        "--package".to_string(),
        "tuisko-qual".to_string(),
        "--release".to_string(),
        "--lib".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        gpu.qualification_feature().to_string(),
        "--message-format=json-render-diagnostics".to_string(),
        "--no-run".to_string(),
        "--".to_string(),
        test_args[0].to_string(),
        test_args[1].to_string(),
        test_args[2].to_string(),
        test_args[3].to_string(),
    ];
    let argument_refs = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    let messages = run_oxide_capture(root, &argument_refs)?;
    let built = messages
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message.get("reason").and_then(serde_json::Value::as_str) == Some("compiler-artifact")
        })
        .filter(|message| {
            message
                .pointer("/target/name")
                .and_then(serde_json::Value::as_str)
                == Some("tuisko_qual")
        })
        .filter_map(|message| {
            message
                .get("executable")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from)
        })
        .next_back()
        .ok_or("cargo did not report the tuisko-qual test executable")?;
    if !built.is_file() {
        return Err(format!(
            "cargo reported a missing qualification executable at {}",
            built.display()
        )
        .into());
    }
    let artifact_name = format!("tuisko-qual-{}", gpu.key());
    let executable = strip_remote_artifact(root, &built, &artifact_name)?;
    println!("remote test binary: {}", executable.display());
    Ok(RemoteQualify {
        executable,
        test_args: test_args.into_iter().map(String::from).collect(),
    })
}

fn strip_remote_artifact(
    root: &Path,
    source: &Path,
    name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let directory = root.join("target/remote-artifacts");
    fs::create_dir_all(&directory)?;
    let output = directory.join(name);
    run_visible(
        Command::new("strip")
            .args(["--strip-all", "-o"])
            .arg(&output)
            .arg(source),
    )?;
    let source_bytes = fs::metadata(source)?.len();
    let output_bytes = fs::metadata(&output)?.len();
    println!(
        "remote artifact: {} -> {} bytes ({:.1}% smaller)",
        source_bytes,
        output_bytes,
        100.0 * (source_bytes.saturating_sub(output_bytes)) as f64 / source_bytes as f64,
    );

    Ok(output)
}

fn run_visible(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(format!("{program} failed with {status}").into());
    }

    Ok(())
}

fn require_cuda_oxide_revision(source: &Path) -> Result<(), Box<dyn Error>> {
    let commit = command_text("git", &["-C", path_text(source)?, "rev-parse", "HEAD"])?;
    if commit.trim() != CUDA_OXIDE_REVISION {
        return Err(format!(
            "cuda-oxide source is at {}, expected {}",
            commit.trim(),
            CUDA_OXIDE_REVISION
        )
        .into());
    }

    Ok(())
}

/// The single-module PTX path of a target that emits one.
///
/// SM120 emits one module per kernel family, so its gates go through
/// `sm120_gate_module` instead of this path.
fn single_ptx_path(gpu: gpu_target::GpuTarget) -> Result<&'static str, Box<dyn Error>> {
    gpu.ptx_path().ok_or_else(|| {
        format!(
            "GPU {} emits one PTX module per kernel family and has no single module path",
            gpu.key()
        )
        .into()
    })
}

/// Pins the Qwen3.8-Flash-Next PLE launch and resource contract.
fn gate_qwen38_flash_next_ple(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_PLE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let artifact = sm120_gate_artifact(root)?;
    let sass = artifact.sass()?;
    let dequant_ptx = ["cvt.rn.f16x2.e4m3x2", "cvt.rn.bf16x2.f32"];
    let dequant_sass = ["F2FP.BF16.F32.PACK_AB"];
    let project_ptx = ["shfl.sync.down.b32", "st.global.b16"];
    let project_sass = ["SHFL"];
    let gate_ptx = ["sqrt.rn.f32", "div.rn.f32", "ex2.approx.f32"];
    let gate_sass = ["MUFU.EX2"];
    let convolution_ptx = ["fma.rn.f32", "ex2.approx.f32"];
    let convolution_sass = ["MUFU.EX2"];
    let inject_ptx = ["cvt.rn.bf16x2.f32"];
    let inject_sass = ["F2FP.BF16.F32.PACK_AB"];
    let families = [
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE dequantization",
            prefix: "qwen38_flash_next_ple_dequant_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "dequant_registers",
            ptx_instructions: &dequant_ptx,
            sass_instructions: &dequant_sass,
            forbidden_sass: &["SHFL"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE dequantization prefill",
            prefix: "qwen38_flash_next_ple_dequant_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "dequant_prefill_registers",
            ptx_instructions: &dequant_ptx,
            sass_instructions: &dequant_sass,
            forbidden_sass: &["SHFL"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE projection",
            prefix: "qwen38_flash_next_ple_project_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "project_registers",
            ptx_instructions: &project_ptx,
            sass_instructions: &project_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE projection prefill",
            prefix: "qwen38_flash_next_ple_project_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "project_prefill_registers",
            ptx_instructions: &project_ptx,
            sass_instructions: &project_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE gate",
            prefix: "qwen38_flash_next_ple_gate_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "gate_registers",
            ptx_instructions: &gate_ptx,
            sass_instructions: &gate_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE gate prefill",
            prefix: "qwen38_flash_next_ple_gate_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "gate_prefill_registers",
            ptx_instructions: &gate_ptx,
            sass_instructions: &gate_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE convolution",
            prefix: "qwen38_flash_next_ple_convolution_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "convolution_registers",
            ptx_instructions: &convolution_ptx,
            sass_instructions: &convolution_sass,
            forbidden_sass: &["SHFL"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE convolution prefill",
            prefix: "qwen38_flash_next_ple_convolution_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "convolution_prefill_registers",
            ptx_instructions: &convolution_ptx,
            sass_instructions: &convolution_sass,
            forbidden_sass: &["SHFL"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE history publication",
            prefix: "qwen38_flash_next_ple_convolution_prefill_history_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "history_registers",
            ptx_instructions: &["st.global.b16"],
            sass_instructions: &[],
            forbidden_sass: &["SHFL", "MUFU.EX2"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE injection",
            prefix: "qwen38_flash_next_ple_inject_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "inject_registers",
            ptx_instructions: &inject_ptx,
            sass_instructions: &inject_sass,
            forbidden_sass: &["SHFL", "MUFU.EX2"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE injection prefill",
            prefix: "qwen38_flash_next_ple_inject_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "inject_prefill_registers",
            ptx_instructions: &inject_ptx,
            sass_instructions: &inject_sass,
            forbidden_sass: &["SHFL", "MUFU.EX2"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next PLE state copy",
            prefix: "qwen38_flash_next_ple_state_copy_exact_TID_",
            count: 2,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "state_copy_registers",
            ptx_instructions: &["ld.global.v2.b64", "st.global.v2.b64"],
            sass_instructions: &[],
            forbidden_sass: &["SHFL", "MUFU.EX2"],
        },
    ];

    let entry_count = gate_exact_resource_families(
        &baseline,
        entries,
        artifact,
        sass,
        &families,
        SharedFootprint::PerEntry,
    )?;
    println!("Qwen3.8-Flash-Next PLE resource gate passed: {entry_count} entries, STACK:0 LOCAL:0");
    Ok(())
}

/// Pins the Qwen3.8-Flash-Next hyper-connection family's launch shapes, spill-freedom,
/// register envelopes, and shared arena.
///
/// The grouped reduction, two GEMVs, gated fold, and elementwise injection
/// retain independent resource groups and register keys.
/// Decode and prefill stay separate groups because they are separate symbols.
fn gate_qwen38_flash_next_hyper_connection(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_HYPER_CONNECTION_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let artifact = sm120_gate_artifact(root)?;
    let sass = artifact.sass()?;
    // The grouped norm is the only stage that takes a reciprocal square root,
    // and both it and the write-back must keep the packed BF16 store epilogue.
    let norm_ptx = ["rsqrt.approx.f32", "cvt.rn.bf16x2.f32"];
    let norm_sass = ["MUFU.RSQ", "F2FP.BF16.F32.PACK_AB"];
    // Both projections reduce one output row inside one warp and round the
    // projection through BF16 before their nonlinearity.
    let projection_ptx = ["shfl.sync.down.b32", "ex2.approx.f32", "st.global.b16"];
    let projection_sass = ["SHFL", "MUFU.EX2"];
    // The write-back is elementwise: no reduction, so a shuffle here would mean
    // the injection had grown a cross-lane dependency it must not have.
    let write_back_ptx = ["cvt.rn.bf16x2.f32"];
    let write_back_sass = ["F2FP.BF16.F32.PACK_AB"];
    let families = [
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection norm",
            prefix: "qwen38_flash_next_hyper_connection_norm_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "norm_registers",
            ptx_instructions: &norm_ptx,
            sass_instructions: &norm_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection norm prefill",
            prefix: "qwen38_flash_next_hyper_connection_norm_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "norm_prefill_registers",
            ptx_instructions: &norm_ptx,
            sass_instructions: &norm_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection mix projection",
            prefix: "qwen38_flash_next_hyper_connection_mix_down_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "mix_down_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection mix projection prefill",
            prefix: "qwen38_flash_next_hyper_connection_mix_down_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "mix_down_prefill_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection mixer projection",
            prefix: "qwen38_flash_next_hyper_connection_final_down_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "final_down_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection mixer projection prefill",
            prefix: "qwen38_flash_next_hyper_connection_final_down_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "final_down_prefill_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection fold",
            prefix: "qwen38_flash_next_hyper_connection_mix_up_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "mix_up_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection fold prefill",
            prefix: "qwen38_flash_next_hyper_connection_mix_up_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "mix_up_prefill_registers",
            ptx_instructions: &projection_ptx,
            sass_instructions: &projection_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection write-back",
            prefix: "qwen38_flash_next_hyper_connection_write_back_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "write_back_registers",
            ptx_instructions: &write_back_ptx,
            sass_instructions: &write_back_sass,
            forbidden_sass: &["SHFL"],
        },
        ExactResourceFamily {
            label: "Qwen3.8-Flash-Next hyper-connection write-back prefill",
            prefix: "qwen38_flash_next_hyper_connection_write_back_prefill_TID_",
            count: 4,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "write_back_prefill_registers",
            ptx_instructions: &write_back_ptx,
            sass_instructions: &write_back_sass,
            forbidden_sass: &["SHFL"],
        },
    ];

    // Only the grouped norm reduces through a shared arena, so this family's
    // footprint is genuinely per entry: a projection or write-back entry that
    // grew one would move a zero in the pinned list.
    let entry_count = gate_exact_resource_families(
        &baseline,
        entries,
        artifact,
        sass,
        &families,
        SharedFootprint::PerEntry,
    )?;
    println!(
        "Qwen3.8-Flash-Next hyper-connection resource gate passed: {entry_count} entries, STACK:0 LOCAL:0, SHARED 1072 on the twelve grouped-norm entries and 0 elsewhere; launch bounds, warp-reduction paths, and register envelopes retained"
    );
    Ok(())
}

fn gate_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_residual_norm_target(root, gpu_target::GpuTarget::Sm120)
}

pub(crate) fn gate_residual_norm_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(gpu.residual_resource_baseline()),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let fallback_ptx;
    let fallback_entries;
    let entries = if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        &sm120_gate_module(root)?.entries
    } else {
        let ptx_path = root.join(single_ptx_path(gpu)?);
        fallback_ptx = fs::read_to_string(&ptx_path).map_err(|error| {
            format!(
                "could not read {}: {error}; run the pinned release device build first",
                ptx_path.display()
            )
        })?;
        fallback_entries = parse_entries(&fallback_ptx);
        &fallback_entries
    };
    // Generic entry names encode the Rust type only in their hash. The pinned
    // compiler leaves Qwen3.8's exact 5,120 divisor in each decode body.
    let plain = entries
        .iter()
        .filter(|entry| {
            entry.name == "rms_norm_b1"
                || (entry.name.starts_with("rms_norm_TID_") && entry.body.contains("0f45A00000"))
        })
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("residual_rms_norm_TID_") && entry.body.contains("0f45A00000")
        })
        .collect::<Vec<_>>();
    let prefill_plain = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("rms_norm_prefill_TID_") && entry.body.contains("0f45A00000")
        })
        .collect::<Vec<_>>();
    let prefill_residual = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("residual_rms_norm_prefill_TID_")
                && entry.body.contains("0f45A00000")
        })
        .collect::<Vec<_>>();
    require_count("plain RMSNorm", plain.len(), 8)?;
    require_count("residual RMSNorm", residual.len(), 8)?;
    let expected_prefill = usize::from(matches!(gpu, gpu_target::GpuTarget::Sm120)) * 4;
    require_count(
        "plain RMSNorm prefill",
        prefill_plain.len(),
        expected_prefill,
    )?;
    require_count(
        "residual RMSNorm prefill",
        prefill_residual.len(),
        expected_prefill,
    )?;

    for entry in plain
        .iter()
        .chain(&residual)
        .chain(&prefill_plain)
        .chain(&prefill_residual)
    {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        let artifact = sm120_gate_artifact(root)?;
        let sass = artifact.sass()?;
        for entry in prefill_plain.iter().chain(&prefill_residual) {
            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["SHFL.BFLY", "MUFU.RSQ", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }

        artifact.resources.clone()
    } else {
        let temporary = root.join("target/tmp");
        fs::create_dir_all(&temporary)?;
        let cubin = temporary.join(format!("residual-norm-{}-gate.cubin", gpu.key()));
        let ptx_path = root.join(single_ptx_path(gpu)?);
        require_success(
            &cuda_tool("ptxas"),
            &[
                OsStr::new("-O3"),
                OsStr::new("--gpu-name"),
                OsStr::new(gpu.oxide_arch()),
                ptx_path.as_os_str(),
                OsStr::new("--output-file"),
                cubin.as_os_str(),
            ],
        )?;
        let output = require_success(
            &cuda_tool("cuobjdump"),
            &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
        )?;
        parse_resources(&String::from_utf8(output.stdout)?)?
    };
    let mut plain_registers = Vec::new();
    let mut residual_registers = Vec::new();
    let mut prefill_plain_registers = Vec::new();
    let mut prefill_residual_registers = Vec::new();
    let mut shared = Vec::new();
    let mut prefill_shared = Vec::new();

    for entry in plain {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted plain entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        plain_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    for entry in residual {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted residual entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        residual_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    for entry in prefill_plain {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted plain prefill entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_plain_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    for entry in prefill_residual {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted residual prefill entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_residual_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    prefill_plain_registers.sort_unstable();
    prefill_residual_registers.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;
    if matches!(gpu, gpu_target::GpuTarget::Sm120) {
        require_registers(
            &baseline,
            "prefill_plain_registers",
            &prefill_plain_registers,
        )?;
        require_registers(
            &baseline,
            "prefill_residual_registers",
            &prefill_residual_registers,
        )?;
        require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;
    }

    println!(
        "{} residual-norm gate passed: 8 plain + 8 residual decode and {} + {} prefill entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}, SHFL/RSQ/BF16 present",
        gpu.key(),
        expected_prefill,
        expected_prefill,
        plain_registers,
        residual_registers,
        prefill_plain_registers,
        prefill_residual_registers,
        shared,
        prefill_shared,
    );
    Ok(())
}

fn gate_qwen35_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_RESIDUAL_NORM_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let plain = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_rms_norm_TID_"))
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_residual_rms_norm_TID_"))
        .collect::<Vec<_>>();
    // The pinned compiler folds Qwen3.5's exact 1/4,096 factor to 0x39800000.
    let prefill_plain = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("rms_norm_prefill_TID_") && entry.body.contains("0f39800000")
        })
        .collect::<Vec<_>>();
    let prefill_residual = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("residual_rms_norm_prefill_TID_")
                && entry.body.contains("0f39800000")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.5 plain RMSNorm", plain.len(), 8)?;
    require_count("Qwen3.5 residual RMSNorm", residual.len(), 8)?;
    require_count("Qwen3.5 plain RMSNorm prefill", prefill_plain.len(), 3)?;
    require_count(
        "Qwen3.5 residual RMSNorm prefill",
        prefill_residual.len(),
        3,
    )?;

    for entry in plain
        .iter()
        .chain(&residual)
        .chain(&prefill_plain)
        .chain(&prefill_residual)
    {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut plain_registers = Vec::with_capacity(plain.len());
    let mut residual_registers = Vec::with_capacity(residual.len());
    let mut prefill_plain_registers = Vec::with_capacity(prefill_plain.len());
    let mut prefill_residual_registers = Vec::with_capacity(prefill_residual.len());
    let mut shared = Vec::with_capacity(
        plain.len() + residual.len() + prefill_plain.len() + prefill_residual.len(),
    );

    for (family, entries, registers) in [
        ("plain", &plain, &mut plain_registers),
        ("residual", &residual, &mut residual_registers),
        (
            "plain prefill",
            &prefill_plain,
            &mut prefill_plain_registers,
        ),
        (
            "residual prefill",
            &prefill_residual,
            &mut prefill_residual_registers,
        ),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!("cuobjdump omitted Qwen3.5 {family} entry `{}`", entry.name)
            })?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["MUFU.RSQ", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    prefill_plain_registers.sort_unstable();
    prefill_residual_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    if baseline.contains_key("prefill_plain_registers") {
        require_registers(
            &baseline,
            "prefill_plain_registers",
            &prefill_plain_registers,
        )?;
    }
    if baseline.contains_key("prefill_residual_registers") {
        require_registers(
            &baseline,
            "prefill_residual_registers",
            &prefill_residual_registers,
        )?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5 residual-norm gate passed: 8 decode + 3 prefill plain and residual entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, RSQ/BF16 pack present",
        plain_registers,
        residual_registers,
        prefill_plain_registers,
        prefill_residual_registers,
        shared
    );
    Ok(())
}

fn gate_qwen36_residual_norm(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_RESIDUAL_NORM_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    // The pinned compiler folds Qwen3.6's exact 1/2,048 factor to 0x3A000000.
    // This separates its generic symbols from the Qwen3.8 generic family.
    let plain = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("rms_norm_TID_") && entry.body.contains("0f3A000000")
        })
        .collect::<Vec<_>>();
    let residual = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("residual_rms_norm_TID_") && entry.body.contains("0f3A000000")
        })
        .collect::<Vec<_>>();
    let prefill_plain = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("rms_norm_prefill_TID_") && entry.body.contains("0f3A000000")
        })
        .collect::<Vec<_>>();
    let prefill_residual = entries
        .iter()
        .filter(|entry| {
            entry.name.starts_with("residual_rms_norm_prefill_TID_")
                && entry.body.contains("0f3A000000")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 plain RMSNorm", plain.len(), 8)?;
    require_count("Qwen3.6 residual RMSNorm", residual.len(), 8)?;
    require_count("Qwen3.6 plain RMSNorm prefill", prefill_plain.len(), 3)?;
    require_count(
        "Qwen3.6 residual RMSNorm prefill",
        prefill_residual.len(),
        3,
    )?;

    for entry in plain
        .iter()
        .chain(&residual)
        .chain(&prefill_plain)
        .chain(&prefill_residual)
    {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut plain_registers = Vec::with_capacity(plain.len());
    let mut residual_registers = Vec::with_capacity(residual.len());
    let mut prefill_plain_registers = Vec::with_capacity(prefill_plain.len());
    let mut prefill_residual_registers = Vec::with_capacity(prefill_residual.len());
    let mut shared = Vec::with_capacity(
        plain.len() + residual.len() + prefill_plain.len() + prefill_residual.len(),
    );

    for (family, entries, registers) in [
        ("plain", &plain, &mut plain_registers),
        ("residual", &residual, &mut residual_registers),
        (
            "plain prefill",
            &prefill_plain,
            &mut prefill_plain_registers,
        ),
        (
            "residual prefill",
            &prefill_residual,
            &mut prefill_residual_registers,
        ),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!("cuobjdump omitted Qwen3.6 {family} entry `{}`", entry.name)
            })?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["MUFU.RSQ", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    plain_registers.sort_unstable();
    residual_registers.sort_unstable();
    prefill_plain_registers.sort_unstable();
    prefill_residual_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "plain_registers", &plain_registers)?;
    require_registers(&baseline, "residual_registers", &residual_registers)?;
    if baseline.contains_key("prefill_plain_registers") {
        require_registers(
            &baseline,
            "prefill_plain_registers",
            &prefill_plain_registers,
        )?;
    }
    if baseline.contains_key("prefill_residual_registers") {
        require_registers(
            &baseline,
            "prefill_residual_registers",
            &prefill_residual_registers,
        )?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 residual-norm gate passed: 8 decode + 3 prefill plain and residual entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, RSQ/BF16 pack present",
        plain_registers,
        residual_registers,
        prefill_plain_registers,
        prefill_residual_registers,
        shared
    );
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn gate_nvfp4_swiglu_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    if gpu == gpu_target::GpuTarget::Sm120 {
        return gate_nvfp4_swiglu(root);
    }

    let baseline_path = gpu
        .nvfp4_swiglu_resource_baseline()
        .ok_or_else(|| format!("GPU {} has no NVFP4 SwiGLU resource baseline", gpu.key()))?;
    gate_nvfp4_a16_target(
        root,
        gpu,
        baseline_path,
        "nvfp4_swiglu_a16_b1",
        "nvfp4_swiglu_a16_TID_",
        "nvfp4-swiglu",
        "NVFP4 SwiGLU",
    )
}

#[cfg(feature = "remote")]
pub(crate) fn gate_nvfp4_down_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
) -> Result<(), Box<dyn Error>> {
    if gpu == gpu_target::GpuTarget::Sm120 {
        return gate_nvfp4_down(root);
    }

    let baseline_path = gpu
        .nvfp4_down_resource_baseline()
        .ok_or_else(|| format!("GPU {} has no NVFP4 down resource baseline", gpu.key()))?;
    gate_nvfp4_a16_target(
        root,
        gpu,
        baseline_path,
        "nvfp4_down_a16_b1",
        "nvfp4_down_a16_TID_",
        "nvfp4-down",
        "NVFP4 down",
    )
}

#[cfg(feature = "remote")]
#[allow(clippy::too_many_arguments)]
fn gate_nvfp4_a16_target(
    root: &Path,
    gpu: gpu_target::GpuTarget,
    baseline_path: &str,
    singleton_name: &str,
    route_prefix: &str,
    artifact_stem: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(single_ptx_path(gpu)?);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned {} release build first",
            ptx_path.display(),
            gpu.oxide_arch()
        )
    })?;
    let entries = parse_entries(&ptx);
    let routes = entries
        .iter()
        .filter(|entry| entry.name == singleton_name || entry.name.starts_with(route_prefix))
        .collect::<Vec<_>>();
    require_count(label, routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 1") {
            return Err(format!(
                "entry `{}` lost its 256-thread/one-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join(format!("{artifact_stem}-{}-gate.cubin", gpu.key()));
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new(gpu.oxide_arch()),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();

    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted {} NVFP4 entry `{}`",
                gpu.key(),
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "{} {label} gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        gpu.key(),
        registers,
        shared
    );
    Ok(())
}

fn gate_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_QKV_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name == "quantize_activation_e4m3")
        .collect::<Vec<_>>();
    let qkv = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    let qkv_t16 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_qkv_mma_t16")
        .collect::<Vec<_>>();
    let qkv_prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_mma_TID_"))
        .collect::<Vec<_>>();
    let qkv_t1024 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_qkv_mma_t1024")
        .collect::<Vec<_>>();
    require_count("FP8 activation quantization", quantize.len(), 1)?;
    require_count("FP8 QKV", qkv.len(), 8)?;
    require_count("FP8 QKV T=16", qkv_t16.len(), 1)?;
    require_count("FP8 QKV T=32/64/128", qkv_prefill.len(), 3)?;
    require_count("FP8 QKV T=1024", qkv_t1024.len(), 1)?;

    for entry in quantize.iter().chain(&qkv) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !qkv_t16[0].body.contains(".reqntid 64, 1, 1")
        || !qkv_t16[0].body.contains(".minnctapersm 4")
    {
        return Err("FP8 QKV T=16 lost its 64-thread/four-CTA launch bounds".into());
    }
    for entry in qkv_prefill.iter().chain(&qkv_t1024) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA prefill launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in qkv_t16.iter().chain(&qkv_prefill).chain(&qkv_t1024) {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV MMA entry `{}`", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "FP8 QKV MMA entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted FP8 activation quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut qkv_registers = Vec::new();
    let mut qkv_shared = Vec::new();
    for entry in qkv {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_registers.push(resource.registers);
        qkv_shared.push(resource.shared);
    }
    qkv_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &qkv_registers)?;
    let qkv_t16_resource = resources
        .get(qkv_t16[0].name)
        .ok_or("cuobjdump omitted FP8 QKV T=16")?;
    require_spill_free(qkv_t16[0].name, qkv_t16_resource)?;
    require_registers(
        &baseline,
        "qkv_t16_registers",
        &[qkv_t16_resource.registers],
    )?;
    let mut qkv_prefill_registers = Vec::new();
    let mut qkv_prefill_shared = Vec::new();
    for entry in qkv_prefill.iter().chain(&qkv_t1024) {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 QKV prefill entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_prefill_registers.push(resource.registers);
        qkv_prefill_shared.push(resource.shared);
    }
    qkv_prefill_registers.sort_unstable();
    if baseline.contains_key("qkv_prefill_registers") {
        require_registers(&baseline, "qkv_prefill_registers", &qkv_prefill_registers)?;
    }

    println!(
        "FP8 QKV gate passed: 1 quantize + 8 decode + 1 T=16 + 4 prefill projection entries, REG {} / {:?} / {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?} / {} / {:?}",
        quantize_resource.registers,
        qkv_registers,
        qkv_t16_resource.registers,
        qkv_prefill_registers,
        quantize_resource.shared,
        qkv_shared,
        qkv_t16_resource.shared,
        qkv_prefill_shared,
    );
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn gate_fp8_qkv_sm89(root: &Path) -> Result<(), Box<dyn Error>> {
    let gpu = gpu_target::GpuTarget::Sm89;
    let baseline_path = gpu
        .fp8_qkv_resource_baseline()
        .ok_or("SM89 has no FP8 QKV resource baseline")?;
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;

    let ptx_path = root.join(single_ptx_path(gpu)?);
    let ptx = fs::read_to_string(&ptx_path).map_err(|error| {
        format!(
            "could not read {}: {error}; run the pinned SM89 release build first",
            ptx_path.display()
        )
    })?;
    let entries = parse_entries(&ptx);
    let quantize = entries
        .iter()
        .filter(|entry| entry.name == "quantize_activation_e4m3")
        .collect::<Vec<_>>();
    let qkv = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    require_count("SM89 FP8 activation quantization", quantize.len(), 1)?;
    require_count("SM89 FP8 QKV", qkv.len(), 8)?;

    for entry in quantize.iter().chain(&qkv) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let temporary = root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let cubin = temporary.join("fp8-qkv-sm89-gate.cubin");
    let ptxas = cuda_tool("ptxas");
    require_success(
        &ptxas,
        &[
            OsStr::new("-O3"),
            OsStr::new("--gpu-name"),
            OsStr::new(gpu.oxide_arch()),
            ptx_path.as_os_str(),
            OsStr::new("--output-file"),
            cubin.as_os_str(),
        ],
    )?;
    let cuobjdump = cuda_tool("cuobjdump");
    let resources = require_success(
        &cuobjdump,
        &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
    )?;
    let resources = parse_resources(&String::from_utf8(resources.stdout)?)?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted SM89 FP8 activation quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;

    let mut qkv_registers = Vec::new();
    let mut qkv_shared = Vec::new();
    for entry in qkv {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted SM89 FP8 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        qkv_registers.push(resource.registers);
        qkv_shared.push(resource.shared);
    }
    qkv_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &qkv_registers)?;

    println!(
        "4090 FP8 QKV gate passed: 1 quantize + 8 decode entries, REG {} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?}",
        quantize_resource.registers, qkv_registers, quantize_resource.shared, qkv_shared,
    );
    Ok(())
}

fn gate_fp8_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_GDN_INPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let gdn_input = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_gdn_input_TID_"))
        .collect::<Vec<_>>();
    let gdn_input_prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_gdn_input_mma_TID_"))
        .collect::<Vec<_>>();
    let gdn_input_t1024 = entries
        .iter()
        .filter(|entry| entry.name == "fp8_gdn_input_tma_t1024")
        .collect::<Vec<_>>();
    require_count("FP8 GDN input", gdn_input.len(), 8)?;
    require_count("FP8 GDN input T=32/64/128", gdn_input_prefill.len(), 3)?;
    require_count("FP8 GDN input T=1024", gdn_input_t1024.len(), 1)?;

    for entry in gdn_input.iter().chain(&gdn_input_prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &gdn_input_t1024 {
        if !entry.body.contains(".reqntid 288, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 288-thread/two-CTA TMA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in gdn_input_prefill.iter().chain(&gdn_input_t1024) {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 GDN input MMA entry `{}`", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "FP8 GDN input MMA entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in gdn_input {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 GDN input entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "gdn_input_registers", &registers)?;
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in gdn_input_prefill.iter().chain(&gdn_input_t1024) {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted FP8 GDN input prefill entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    if baseline.contains_key("gdn_input_prefill_registers") {
        require_registers(&baseline, "gdn_input_prefill_registers", &prefill_registers)?;
    }

    println!(
        "FP8 GDN input gate passed: 8 decode + 4 prefill projection entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}",
        registers, prefill_registers, shared, prefill_shared
    );
    Ok(())
}

fn gate_fp8_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let lm_head = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_lm_head_TID_"))
        .collect::<Vec<_>>();
    require_count("FP8 LM head", lm_head.len(), 8)?;

    for entry in &lm_head {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let resources = &sm120_gate_artifact(root)?.resources;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in lm_head {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted FP8 LM-head entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;

    println!(
        "FP8 LM-head gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers, shared
    );
    Ok(())
}

fn gate_fp8_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(FP8_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let module = sm120_gate_module(root)?;
    let entries = &module.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_quantize_TID_"))
        .collect::<Vec<_>>();
    let decode = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_swiglu_decode_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.name,
                "fp8_swiglu_mma_t32" | "fp8_swiglu_mma_t64" | "fp8_swiglu_mma_t128"
            )
        })
        .collect::<Vec<_>>();
    let tma = entries
        .iter()
        .filter(|entry| entry.name == "fp8_swiglu_tma_t1024")
        .collect::<Vec<_>>();
    require_count("dense-FP8 SwiGLU quantization", quantize.len(), 1)?;
    require_count("dense-FP8 SwiGLU decode", decode.len(), 8)?;
    require_count("dense-FP8 SwiGLU prefill", prefill.len(), 3)?;
    require_count("dense-FP8 SwiGLU TMA prefill", tma.len(), 1)?;

    for entry in quantize.iter().chain(&decode).chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !tma[0].body.contains(".reqntid 288, 1, 1")
        || !tma[0].body.contains(".minnctapersm 2")
        || tma[0].body.contains(".reqnctapercluster")
    {
        return Err(
            "dense-FP8 SwiGLU TMA lost its 288-thread/two-CTA single-CTA launch contract".into(),
        );
    }
    if !module
        .ptx
        .contains(".extern .shared .align 128 .b8 __dynamic_smem_fp8_swiglu_tma_t1024[];")
    {
        return Err("dense-FP8 SwiGLU TMA lost its 128-byte dynamic-shared alignment".into());
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in &prefill {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("QMMA.16832.F32.E4M3.E4M3") {
            return Err(format!(
                "dense-FP8 SwiGLU entry `{}` lost its native E4M3 tensor-core instruction",
                entry.name
            )
            .into());
        }
    }
    let tma_sass = sass_function_body(sass, tma[0].name)
        .ok_or("cuobjdump omitted dense-FP8 SwiGLU TMA SASS")?;
    for instruction in ["QMMA.16832.F32.E4M3.E4M3", "UTMALDG.2D", "MUFU.RCP"] {
        if !tma_sass.contains(instruction) {
            return Err(format!("dense-FP8 SwiGLU TMA lost required `{instruction}` SASS").into());
        }
    }
    if tma_sass.contains("CALL.") {
        return Err("dense-FP8 SwiGLU TMA regained an out-of-line device call".into());
    }

    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 SwiGLU quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    require_uniform_value(
        &baseline,
        "quantize_shared_bytes",
        &[quantize_resource.shared],
    )?;

    let mut decode_registers = Vec::new();
    let mut decode_shared = Vec::new();
    for entry in &decode {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        decode_registers.push(resource.registers);
        decode_shared.push(resource.shared);
    }
    decode_registers.sort_unstable();
    require_registers(&baseline, "decode_registers", &decode_registers)?;
    require_uniform_value(&baseline, "decode_shared_bytes", &decode_shared)?;

    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in &prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prefill_registers", &prefill_registers)?;
    require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;

    let tma_resource = resources
        .get(tma[0].name)
        .ok_or("cuobjdump omitted dense-FP8 SwiGLU TMA resources")?;
    require_spill_free(tma[0].name, tma_resource)?;
    require_registers(&baseline, "tma_registers", &[tma_resource.registers])?;
    require_uniform_value(&baseline, "tma_shared_bytes", &[tma_resource.shared])?;

    println!(
        "dense-FP8 SwiGLU gate passed: 1 quantize + 8 decode + 3 tiled prefill + 1 TMA prefill entries, REG {} / {:?} / {:?} / {}, STACK:0 LOCAL:0, SHARED {} / {:?} / {:?} / {}, QMMA/TMA/RCP present",
        quantize_resource.registers,
        decode_registers,
        prefill_registers,
        tma_resource.registers,
        quantize_resource.shared,
        decode_shared,
        prefill_shared,
        tma_resource.shared,
    );
    Ok(())
}

fn gate_fp8_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(FP8_DOWN_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;

    let module = sm120_gate_module(root)?;
    let entries = &module.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_quantize_TID_"))
        .collect::<Vec<_>>();
    let down = entries
        .iter()
        .filter(|entry| entry.name.starts_with("fp8_down_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.name,
                "fp8_down_mma_t32" | "fp8_down_mma_t64" | "fp8_down_mma_t128"
            )
        })
        .collect::<Vec<_>>();
    let tma = entries
        .iter()
        .filter(|entry| entry.name == "fp8_down_tma_t1024")
        .collect::<Vec<_>>();
    require_count("dense-FP8 down quantization", quantize.len(), 1)?;
    require_count("dense-FP8 down projection", down.len(), 8)?;
    require_count("dense-FP8 down tail prefill", prefill.len(), 3)?;
    require_count("dense-FP8 down TMA prefill", tma.len(), 1)?;

    for entry in quantize.iter().chain(&down) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        let (threads, minimum_ctas) = if entry.name == "fp8_down_mma_t32" {
            (128, 4)
        } else {
            (256, 2)
        };
        if !entry.body.contains(&format!(".reqntid {threads}, 1, 1"))
            || !entry
                .body
                .contains(&format!(".minnctapersm {minimum_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its exact {threads}-thread/{minimum_ctas}-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !tma[0].body.contains(".reqntid 288, 1, 1")
        || !tma[0].body.contains(".minnctapersm 2")
        || tma[0].body.contains(".reqnctapercluster")
    {
        return Err(
            "dense-FP8 down TMA lost its 288-thread/two-CTA single-CTA launch contract".into(),
        );
    }
    if !module
        .ptx
        .contains(".extern .shared .align 128 .b8 __dynamic_smem_fp8_down_tma_t1024[];")
    {
        return Err("dense-FP8 down TMA lost its 128-byte dynamic-shared alignment".into());
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in &prefill {
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["QMMA.16832.F32.E4M3.E4M3", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(format!(
                    "dense-FP8 down entry `{}` lost required `{instruction}` SASS",
                    entry.name
                )
                .into());
            }
        }
    }
    let tma_sass =
        sass_function_body(sass, tma[0].name).ok_or("cuobjdump omitted dense-FP8 down TMA SASS")?;
    for instruction in ["QMMA.16832.F32.E4M3.E4M3", "UTMALDG.2D"] {
        if !tma_sass.contains(instruction) {
            return Err(format!("dense-FP8 down TMA lost required `{instruction}` SASS").into());
        }
    }
    if tma_sass.contains("CALL.") {
        return Err("dense-FP8 down TMA regained an out-of-line device call".into());
    }
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted dense-FP8 down quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    require_uniform_value(
        &baseline,
        "quantize_shared_bytes",
        &[quantize_resource.shared],
    )?;

    let mut down_registers = Vec::new();
    let mut shared = Vec::new();
    for entry in down {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        down_registers.push(resource.registers);
        shared.push(resource.shared);
    }
    down_registers.sort_unstable();
    require_registers(&baseline, "down_registers", &down_registers)?;
    require_uniform_value(&baseline, "down_shared_bytes", &shared)?;

    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in &prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prefill_registers", &prefill_registers)?;
    require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;

    let tma_resource = resources
        .get(tma[0].name)
        .ok_or("cuobjdump omitted dense-FP8 down TMA resources")?;
    require_spill_free(tma[0].name, tma_resource)?;
    require_registers(&baseline, "tma_registers", &[tma_resource.registers])?;
    require_uniform_value(&baseline, "tma_shared_bytes", &[tma_resource.shared])?;

    println!(
        "dense-FP8 down gate passed: 1 quantize + 8 decode + 3 tiled prefill + 1 TMA prefill entries, REG {} / {:?} / {:?} / {}, STACK:0 LOCAL:0, SHARED {} / {:?} / {:?} / {}, QMMA/LDGSTS/TMA present",
        quantize_resource.registers,
        down_registers,
        prefill_registers,
        tma_resource.registers,
        quantize_resource.shared,
        shared,
        prefill_shared,
        tma_resource.shared,
    );
    Ok(())
}

fn gate_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(NVFP4_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let a16 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_swiglu_a16_t"))
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_swiglu_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("NVFP4 SwiGLU A16", a16.len(), 4)?;
    require_count("NVFP4 activation quantization", quantize.len(), 9)?;
    require_count("NVFP4 SwiGLU W4A4", w4a4.len(), 9)?;

    for entry in &a16 {
        let minimum_ctas = if entry.name == "nvfp4_swiglu_a16_t1" {
            2
        } else {
            1
        };
        if !entry.body.contains(".reqntid 256, 1, 1")
            || !entry
                .body
                .contains(&format!(".minnctapersm {minimum_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its retained 256-thread/{minimum_ctas}-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its retained 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;

    let mut a16_registers = Vec::new();
    let mut a16_shared = Vec::new();
    for entry in a16 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 A16 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        a16_registers.push(resource.registers);
        a16_shared.push(resource.shared);
    }
    a16_registers.sort_unstable();
    a16_shared.sort_unstable();
    require_registers(&baseline, "a16_registers", &a16_registers)?;
    require_registers(&baseline, "a16_shared_bytes", &a16_shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted NVFP4 quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_uniform_value(&baseline, "quantize_shared_bytes", &quantize_shared)?;

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 W4A4 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 W4A4 SASS `{}`", entry.name))?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    require_registers(&baseline, "w4a4_registers", &w4a4_registers)?;
    require_uniform_value(&baseline, "w4a4_shared_bytes", &w4a4_shared)?;

    println!(
        "NVFP4 SwiGLU gate passed: 4 A16 + 9 quantize (5 decode/4 prefill) + 9 W4A4 (5 decode/4 prefill) entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        a16_registers, quantize_registers, w4a4_registers, a16_shared, quantize_shared, w4a4_shared,
    );
    Ok(())
}

fn gate_qwen35_nvfp4_swiglu(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_SWIGLU_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let a16 = entries
        .iter()
        .filter(|entry| {
            entry.name == "qwen35_nvfp4_swiglu_a16_t1"
                || entry.name == "qwen35_nvfp4_swiglu_a16_t2"
                || entry.name.starts_with("qwen35_nvfp4_swiglu_a16_TID_")
        })
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_swiglu_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 SwiGLU A16", a16.len(), 4)?;
    require_count("Qwen3.5 NVFP4 activation quantization", quantize.len(), 12)?;
    require_count("Qwen3.5 NVFP4 SwiGLU W4A4", w4a4.len(), 12)?;

    for entry in &a16 {
        let minimum_ctas = if entry.name == "qwen35_nvfp4_swiglu_a16_t1" {
            2
        } else {
            1
        };
        if !entry.body.contains(".reqntid 256, 1, 1")
            || !entry
                .body
                .contains(&format!(".minnctapersm {minimum_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its retained 256-thread/{minimum_ctas}-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its retained 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;

    let mut a16_registers = Vec::new();
    let mut a16_shared = Vec::new();
    for entry in a16 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 A16 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        a16_registers.push(resource.registers);
        a16_shared.push(resource.shared);
    }
    a16_registers.sort_unstable();
    a16_shared.sort_unstable();
    require_registers(&baseline, "a16_registers", &a16_registers)?;
    require_registers(&baseline, "a16_shared_bytes", &a16_shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    quantize_shared.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_uniform_value(&baseline, "quantize_shared_bytes", &quantize_shared)?;

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 W4A4 entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 W4A4 SASS `{}`", entry.name))?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    w4a4_shared.sort_unstable();
    require_registers(&baseline, "w4a4_registers", &w4a4_registers)?;
    require_uniform_value(&baseline, "w4a4_shared_bytes", &w4a4_shared)?;

    println!(
        "Qwen3.5 NVFP4 SwiGLU gate passed: 4 A16 + 12 quantize (8 decode/4 prefill) + 12 W4A4 (8 decode/4 prefill) entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        a16_registers, quantize_registers, w4a4_registers, a16_shared, quantize_shared, w4a4_shared,
    );
    Ok(())
}

fn gate_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_PREPARE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let control = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_control_exact_TID_"))
        .collect::<Vec<_>>();
    let convolution = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_convolution_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill_control = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_control_prefill_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill_convolution = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_convolution_prefill_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill_history = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("gdn_convolution_prefill_history_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("GDN control", control.len(), 8)?;
    require_count("GDN convolution", convolution.len(), 8)?;
    require_count("GDN causal/prefill control", prefill_control.len(), 8)?;
    require_count(
        "GDN causal/prefill convolution",
        prefill_convolution.len(),
        8,
    )?;
    require_count(
        "GDN causal/prefill history publication",
        prefill_history.len(),
        8,
    )?;

    for entry in control.iter().chain(&prefill_control) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in convolution
        .iter()
        .chain(&prefill_convolution)
        .chain(&prefill_history)
    {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in prefill_control
        .iter()
        .chain(&prefill_convolution)
        .chain(&prefill_history)
    {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(
                format!("cuobjdump omitted GDN prefill SASS entry `{}`", entry.name).into(),
            );
        }
    }
    let mut control_registers = Vec::new();
    let mut convolution_registers = Vec::new();
    let mut prefill_control_registers = Vec::new();
    let mut prefill_convolution_registers = Vec::new();
    let mut prefill_history_registers = Vec::new();
    let mut control_shared = Vec::new();
    let mut convolution_shared = Vec::new();
    let mut prefill_control_shared = Vec::new();
    let mut prefill_convolution_shared = Vec::new();
    let mut prefill_history_shared = Vec::new();

    for entry in control {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        control_registers.push(resource.registers);
        control_shared.push(resource.shared);
    }
    for entry in convolution {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        convolution_registers.push(resource.registers);
        convolution_shared.push(resource.shared);
    }
    for (entries, registers, shared) in [
        (
            &prefill_control,
            &mut prefill_control_registers,
            &mut prefill_control_shared,
        ),
        (
            &prefill_convolution,
            &mut prefill_convolution_registers,
            &mut prefill_convolution_shared,
        ),
        (
            &prefill_history,
            &mut prefill_history_registers,
            &mut prefill_history_shared,
        ),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    control_registers.sort_unstable();
    convolution_registers.sort_unstable();
    prefill_control_registers.sort_unstable();
    prefill_convolution_registers.sort_unstable();
    prefill_history_registers.sort_unstable();
    require_registers(&baseline, "control_registers", &control_registers)?;
    require_registers(&baseline, "convolution_registers", &convolution_registers)?;
    for (key, registers) in [
        ("prefill_control_registers", &prefill_control_registers),
        (
            "prefill_convolution_registers",
            &prefill_convolution_registers,
        ),
        ("prefill_history_registers", &prefill_history_registers),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    for (key, shared) in [
        ("prefill_control_shared_bytes", &prefill_control_shared),
        (
            "prefill_convolution_shared_bytes",
            &prefill_convolution_shared,
        ),
        ("prefill_history_shared_bytes", &prefill_history_shared),
    ] {
        if baseline.contains_key(key) {
            require_uniform_value(&baseline, key, shared)?;
        }
    }

    println!(
        "GDN prepare gate passed: 8 decode control + 8 decode convolution + 4 causal/4 prefill control + 4 causal/4 prefill convolution + 4 causal/4 prefill history entries, REG {:?} / {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?} / {:?} / {:?}, SASS present",
        control_registers,
        convolution_registers,
        prefill_control_registers,
        prefill_convolution_registers,
        prefill_history_registers,
        control_shared,
        convolution_shared,
        prefill_control_shared,
        prefill_convolution_shared,
        prefill_history_shared,
    );
    Ok(())
}

/// Pins the target-specific control entries and their reused convolution dependency.
fn gate_qwen38_flash_next_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_GDN_PREPARE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let control = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_gdn_control_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_control = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_gdn_control_prefill_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.8-Flash-Next GDN control", control.len(), 8)?;
    require_count(
        "Qwen3.8-Flash-Next GDN causal/prefill control",
        prefill_control.len(),
        8,
    )?;
    // Reused convolution entries remain part of the admitted route.
    require_count(
        "Qwen3.8-Flash-Next GDN reused convolution",
        entries
            .iter()
            .filter(|entry| entry.name.starts_with("gdn_convolution_exact_TID_"))
            .count(),
        8,
    )?;

    for entry in control.iter().chain(&prefill_control) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        // Pin the softplus and beta-sigmoid math.
        if !entry.body.contains("lg2.approx.f32")
            || !entry.body.contains("ex2.approx.f32")
            || !entry.body.contains("rcp.rn.f32")
        {
            return Err(format!(
                "entry `{}` lost its softplus/decay/beta control math",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in control.iter().chain(&prefill_control) {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next GDN control SASS `{}`",
                entry.name
            )
            .into());
        }
    }

    let mut control_registers = Vec::new();
    let mut control_shared = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for (entries, registers, shared) in [
        (&control, &mut control_registers, &mut control_shared),
        (
            &prefill_control,
            &mut prefill_registers,
            &mut prefill_shared,
        ),
    ] {
        for entry in entries.iter() {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "control_registers", &control_registers)?;
    require_uniform_value(&baseline, "control_shared_bytes", &control_shared)?;
    require_registers(&baseline, "prefill_control_registers", &prefill_registers)?;
    require_uniform_value(&baseline, "prefill_control_shared_bytes", &prefill_shared)?;

    println!(
        "Qwen3.8-Flash-Next GDN prepare gate passed: 8 control + 8 causal/prefill control entries reusing the qualified convolution leaves, REG {control_registers:?} / {prefill_registers:?}, STACK:0 LOCAL:0, SHARED {control_shared:?} / {prefill_shared:?}, softplus/beta math and SASS present"
    );

    Ok(())
}

/// Pins sigmoid recurrence entries by their reciprocal and lack of SiLU division.
fn gate_qwen38_flash_next_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_GDN_RECURRENCE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let recurrence = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_gdn_recurrence_exact_TID_")
        })
        .collect::<Vec<_>>();
    let epilogue = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_gdn_recurrence_prefill_epilogue_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.8-Flash-Next GDN recurrence", recurrence.len(), 8)?;
    require_count(
        "Qwen3.8-Flash-Next GDN recurrence prefill epilogue",
        epilogue.len(),
        8,
    )?;
    // The reused serial pass remains part of the admitted route.
    require_count(
        "Qwen3.8-Flash-Next GDN reused serial prefill",
        entries
            .iter()
            .filter(|entry| entry.name.starts_with("gdn_recurrence_prefill_exact_TID_"))
            .count(),
        8,
    )?;

    for entry in recurrence.iter().chain(&epilogue) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("rsqrt.approx.f32") || !entry.body.contains("ex2.approx.f32") {
            return Err(format!(
                "entry `{}` lost its RMS normalization or gate exponential",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("rcp.rn.f32") || entry.body.contains("div.rn.f32") {
            return Err(format!(
                "entry `{}` is not the sigmoid gated-norm variant: a SiLU epilogue divides by the gate denominator instead of taking its reciprocal",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in recurrence.iter().chain(&epilogue) {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next GDN recurrence SASS `{}`",
                entry.name
            )
            .into());
        }
    }

    let mut recurrence_registers = Vec::new();
    let mut recurrence_shared = Vec::new();
    let mut epilogue_registers = Vec::new();
    let mut epilogue_shared = Vec::new();
    for (entries, registers, shared) in [
        (
            &recurrence,
            &mut recurrence_registers,
            &mut recurrence_shared,
        ),
        (&epilogue, &mut epilogue_registers, &mut epilogue_shared),
    ] {
        for entry in entries.iter() {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "recurrence_registers", &recurrence_registers)?;
    require_uniform_value(&baseline, "recurrence_shared_bytes", &recurrence_shared)?;
    require_registers(&baseline, "prefill_epilogue_registers", &epilogue_registers)?;
    require_uniform_value(&baseline, "prefill_epilogue_shared_bytes", &epilogue_shared)?;

    println!(
        "Qwen3.8-Flash-Next GDN recurrence gate passed: 8 decode + 8 prefill epilogue entries reusing the qualified serial prefill pass, REG {recurrence_registers:?} / {epilogue_registers:?}, STACK:0 LOCAL:0, SHARED {recurrence_shared:?} / {epilogue_shared:?}, sigmoid reciprocal present and SiLU division absent"
    );

    Ok(())
}

/// Pins all QSA prepare entries and their target-specific 26-head division.
fn gate_qwen38_flash_next_qsa_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_QSA_PREPARE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let prepare = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_attention_qk_prepare_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_prepare = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_attention_qk_prepare_prefill_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.8-Flash-Next QSA prepare", prepare.len(), 8)?;
    require_count(
        "Qwen3.8-Flash-Next QSA causal/prefill prepare",
        prefill_prepare.len(),
        4,
    )?;

    for entry in prepare.iter().chain(&prefill_prepare) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        // Require normalization, MRoPE exchange, and represented cache append.
        if !entry.body.contains("rsqrt.approx.f32")
            || !entry.body.contains("shfl.sync")
            || !entry.body.contains("cvt.rn.satfinite.e4m3x2.f32")
        {
            return Err(format!(
                "entry `{}` lost its RMS reciprocal, warp/MRoPE exchange, or E4M3 cache append",
                entry.name
            )
            .into());
        }
        if !contains_immediate_operand(entry.body, "26")
            || contains_immediate_operand(entry.body, "28")
        {
            return Err(format!(
                "entry `{}` is not the 24/2 QSA head-warp division",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in prepare.iter().chain(&prefill_prepare) {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next QSA prepare SASS `{}`",
                entry.name
            )
            .into());
        }
    }

    let mut prepare_registers = Vec::new();
    let mut prepare_shared = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for (entries, registers, shared) in [
        (&prepare, &mut prepare_registers, &mut prepare_shared),
        (
            &prefill_prepare,
            &mut prefill_registers,
            &mut prefill_shared,
        ),
    ] {
        for entry in entries.iter() {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "prepare_registers", &prepare_registers)?;
    require_uniform_value(&baseline, "prepare_shared_bytes", &prepare_shared)?;
    require_registers(&baseline, "prefill_prepare_registers", &prefill_registers)?;
    require_uniform_value(&baseline, "prefill_prepare_shared_bytes", &prefill_shared)?;

    println!(
        "Qwen3.8-Flash-Next QSA prepare gate passed: 8 decode + 4 causal/prefill prepare entries, REG {prepare_registers:?} / {prefill_registers:?}, STACK:0 LOCAL:0, SHARED {prepare_shared:?} / {prefill_shared:?}, RMS reciprocal, MRoPE exchange, E4M3 cache append, the 24/2 head-warp division and SASS present"
    );

    Ok(())
}

/// Pins all QSA attention entries and the sigmoid gate instruction shape.
fn gate_qwen38_flash_next_qsa_attention(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_QSA_ATTENTION_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let decode = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_paged_gqa_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_paged_gqa_prefill_shared_exact_TID_")
        })
        .collect::<Vec<_>>();
    let gate = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_attention_output_gate_bf16_TID_")
        })
        .collect::<Vec<_>>();
    let gate_prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_attention_output_gate_bf16_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.8-Flash-Next QSA decode attention", decode.len(), 8)?;
    require_count("Qwen3.8-Flash-Next QSA prefill attention", prefill.len(), 4)?;
    require_count("Qwen3.8-Flash-Next QSA output gate", gate.len(), 8)?;
    require_count(
        "Qwen3.8-Flash-Next QSA causal/prefill output gate",
        gate_prefill.len(),
        4,
    )?;

    for entry in decode.iter() {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        // The online softmax and the represented E4M3 cache load are what make
        // this the paged decode pass over the quantized cache.
        if !entry.body.contains("ex2.approx.f32") || !entry.body.contains("cvt.rn.f16x2.e4m3x2") {
            return Err(format!(
                "entry `{}` lost its online softmax or E4M3 cache load",
                entry.name
            )
            .into());
        }
    }

    for entry in prefill.iter() {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        // `cp.async` is the shared K/V tile fill this pass exists for.
        if !entry.body.contains("ex2.approx.f32") || !entry.body.contains("cp.async") {
            return Err(format!(
                "entry `{}` lost its online softmax or shared K/V tile fill",
                entry.name
            )
            .into());
        }
    }

    for entry in gate.iter().chain(&gate_prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("ex2.approx.f32")
            || !entry.body.contains("rcp.rn.f32")
            || entry.body.contains("div.rn.f32")
        {
            return Err(format!(
                "entry `{}` is not the sigmoid output gate: a SiLU gate divides by the gate denominator instead of taking its reciprocal",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in decode
        .iter()
        .chain(&prefill)
        .chain(&gate)
        .chain(&gate_prefill)
    {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next QSA attention SASS `{}`",
                entry.name
            )
            .into());
        }
    }

    let mut decode_registers = Vec::new();
    let mut decode_shared = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    let mut gate_registers = Vec::new();
    let mut gate_shared = Vec::new();
    let mut gate_prefill_registers = Vec::new();
    let mut gate_prefill_shared = Vec::new();
    for (entries, registers, shared) in [
        (&decode, &mut decode_registers, &mut decode_shared),
        (&prefill, &mut prefill_registers, &mut prefill_shared),
        (&gate, &mut gate_registers, &mut gate_shared),
        (
            &gate_prefill,
            &mut gate_prefill_registers,
            &mut gate_prefill_shared,
        ),
    ] {
        for entry in entries.iter() {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "decode_registers", &decode_registers)?;
    require_uniform_value(&baseline, "decode_shared_bytes", &decode_shared)?;
    require_registers(&baseline, "prefill_registers", &prefill_registers)?;
    require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;
    require_registers(&baseline, "gate_registers", &gate_registers)?;
    require_uniform_value(&baseline, "gate_shared_bytes", &gate_shared)?;
    require_registers(&baseline, "gate_prefill_registers", &gate_prefill_registers)?;
    require_uniform_value(&baseline, "gate_prefill_shared_bytes", &gate_prefill_shared)?;

    println!(
        "Qwen3.8-Flash-Next QSA attention gate passed: 8 decode + 4 prefill scoring entries and 8 + 4 output gate entries, REG {decode_registers:?} / {prefill_registers:?} / {gate_registers:?} / {gate_prefill_registers:?}, STACK:0 LOCAL:0, SHARED {decode_shared:?} / {prefill_shared:?} / {gate_shared:?} / {gate_prefill_shared:?}, online softmax, E4M3 cache load, shared K/V tile fill, sigmoid reciprocal present with SiLU division absent, and SASS present"
    );

    Ok(())
}

/// Pins all QSA selection entries and their defining instruction shapes.
fn gate_qwen38_flash_next_qsa_selection(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_QSA_SELECTION_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let family = |prefix: &str| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    };
    let prepare = family("qwen38_flash_next_indexer_prepare_exact_TID_");
    let prepare_prefill = family("qwen38_flash_next_indexer_prepare_prefill_exact_TID_");
    let compress = family("qwen38_flash_next_indexer_block_compress_exact_TID_");
    let score = family("qwen38_flash_next_indexer_score_exact_TID_");
    let select_pass = family("qwen38_flash_next_indexer_select_pass_exact_TID_");
    let select_expand = family("qwen38_flash_next_indexer_select_expand_exact_TID_");
    let attention = family("qwen38_flash_next_paged_gqa_selected_exact_TID_");
    let attention_prefill = family("qwen38_flash_next_paged_gqa_prefill_selected_exact_TID_");
    require_count("Qwen3.8-Flash-Next indexer prepare", prepare.len(), 8)?;
    require_count(
        "Qwen3.8-Flash-Next indexer prefill prepare",
        prepare_prefill.len(),
        4,
    )?;
    require_count(
        "Qwen3.8-Flash-Next indexer block compression",
        compress.len(),
        12,
    )?;
    require_count("Qwen3.8-Flash-Next indexer scoring", score.len(), 10)?;
    require_count(
        "Qwen3.8-Flash-Next indexer selection pass",
        select_pass.len(),
        10,
    )?;
    require_count(
        "Qwen3.8-Flash-Next indexer selection expansion",
        select_expand.len(),
        10,
    )?;
    require_count(
        "Qwen3.8-Flash-Next selected decode attention",
        attention.len(),
        8,
    )?;
    require_count(
        "Qwen3.8-Flash-Next selected prefill attention",
        attention_prefill.len(),
        4,
    )?;

    for entry in prepare.iter().chain(&prepare_prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("rsqrt.approx.f32") || !entry.body.contains("shfl.sync.bfly.b32") {
            return Err(format!(
                "entry `{}` lost its RMS reciprocal or MRoPE exchange",
                entry.name
            )
            .into());
        }
        if entry.body.contains("e4m3") {
            return Err(
                format!("entry `{}` quantizes the raw indexer key plane", entry.name).into(),
            );
        }
    }

    for entry in compress.iter() {
        if !entry.body.contains("rsqrt.approx.f32") || !entry.body.contains("cvt.rn.bf16x2.f32") {
            return Err(format!(
                "entry `{}` lost its pooled-key norm or BF16 store",
                entry.name
            )
            .into());
        }
    }

    for entry in score.iter() {
        if !entry.body.contains("max.f32") {
            return Err(format!("entry `{}` lost its ReLU", entry.name).into());
        }
        if entry.body.contains("ex2.approx.f32") {
            return Err(format!("entry `{}` exponentiates indexer scores", entry.name).into());
        }
    }

    for entry in &select_pass {
        if !entry.body.contains("match.any.sync.b32") {
            return Err(format!("entry `{}` lost its conflict-free histogram", entry.name).into());
        }
    }

    for entry in select_pass.iter().chain(&select_expand) {
        if names_opcode(entry.body, "atom.") || names_opcode(entry.body, "red.") {
            return Err(format!("entry `{}` uses atomic selection accounting", entry.name).into());
        }
        if !entry.body.contains("bar.sync") || !entry.body.contains("shfl.sync.up.b32") {
            return Err(format!(
                "entry `{}` lost its CTA-wide ascending prefix count",
                entry.name
            )
            .into());
        }
    }

    for entry in attention.iter().chain(&attention_prefill) {
        if !entry.body.contains("ex2.approx.f32") || !entry.body.contains("cvt.rn.f16x2.e4m3x2") {
            return Err(format!(
                "entry `{}` lost its online softmax or E4M3 cache load",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in prepare
        .iter()
        .chain(&prepare_prefill)
        .chain(&compress)
        .chain(&score)
        .chain(&select_pass)
        .chain(&select_expand)
        .chain(&attention)
        .chain(&attention_prefill)
    {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next QSA selection SASS `{}`",
                entry.name
            )
            .into());
        }
    }

    let mut measured = Vec::new();
    for (label, group) in [
        ("prepare", &prepare),
        ("prepare_prefill", &prepare_prefill),
        ("compress", &compress),
        ("score", &score),
        ("select_pass", &select_pass),
        ("select_expand", &select_expand),
        ("attention", &attention),
        ("attention_prefill", &attention_prefill),
    ] {
        let mut registers = Vec::new();
        let mut shared = Vec::new();
        for entry in group.iter() {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
        require_registers(&baseline, &format!("{label}_registers"), &registers)?;
        require_uniform_value(&baseline, &format!("{label}_shared_bytes"), &shared)?;
        measured.push((label, registers, shared[0]));
    }

    println!(
        "Qwen3.8-Flash-Next QSA selection gate passed: 66 entries, STACK:0 LOCAL:0, {measured:?}, exact indexer instruction shapes and SASS present"
    );

    Ok(())
}

/// Pins every Qwen3.8-Flash-Next MoE router entry and its 512-way softmax.
fn gate_qwen38_flash_next_moe_router(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_MOE_ROUTER_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let family = |prefix: &str| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    };
    let logits = family("qwen38_flash_next_moe_router_logits_TID_");
    let logits_prefill = family("qwen38_flash_next_moe_router_logits_prefill_TID_");
    let select = family("qwen38_flash_next_moe_router_select_TID_");
    let select_prefill = family("qwen38_flash_next_moe_router_select_prefill_TID_");
    require_count(
        "Qwen3.8-Flash-Next MoE router decode logits",
        logits.len(),
        8,
    )?;
    require_count(
        "Qwen3.8-Flash-Next MoE router prefill logits",
        logits_prefill.len(),
        4,
    )?;
    require_count(
        "Qwen3.8-Flash-Next MoE router decode selection",
        select.len(),
        8,
    )?;
    require_count(
        "Qwen3.8-Flash-Next MoE router prefill selection",
        select_prefill.len(),
        4,
    )?;

    for entry in logits.iter().chain(&logits_prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in select.iter().chain(&select_prefill) {
        if !entry.body.contains(".reqntid 32, 1, 1") {
            return Err(format!("entry `{}` lost its one-warp launch bounds", entry.name).into());
        }
        let exponentials = entry.body.matches("ex2.approx.f32").count();
        let divisions = entry.body.matches("div.rn.f32").count();
        if exponentials != 16 || divisions != 26 {
            return Err(format!(
                "entry `{}` has {exponentials} exponentials and {divisions} divisions; expected 16 and 26 for the 512-way softmax",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut logit_registers = Vec::new();
    let mut select_registers = Vec::new();
    for (group, registers) in [
        (
            logits.iter().chain(&logits_prefill).collect::<Vec<_>>(),
            &mut logit_registers,
        ),
        (
            select.iter().chain(&select_prefill).collect::<Vec<_>>(),
            &mut select_registers,
        ),
    ] {
        for entry in group {
            if sass_function_body(sass, entry.name).is_none() {
                return Err(format!(
                    "cuobjdump omitted Qwen3.8-Flash-Next MoE router SASS `{}`",
                    entry.name
                )
                .into());
            }
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "logit_registers", &logit_registers)?;
    require_registers(&baseline, "select_registers", &select_registers)?;

    println!(
        "Qwen3.8-Flash-Next MoE router gate passed: 12 logit + 12 selection entries, REG {logit_registers:?} / {select_registers:?}, STACK:0 LOCAL:0, 512-way softmax and SASS present"
    );
    Ok(())
}

/// Pins routed NVFP4, resident BF16, and combine expert entry families.
fn gate_qwen38_flash_next_moe_experts(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_MOE_EXPERTS_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let family = |prefix: &str| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    };
    let routed = [
        family("qwen38_flash_next_moe_expert_gate_up_TID_"),
        family("qwen38_flash_next_moe_expert_gate_up_prefill_TID_"),
        family("qwen38_flash_next_moe_expert_down_TID_"),
        family("qwen38_flash_next_moe_expert_down_prefill_TID_"),
    ];
    let shared = [
        family("qwen38_flash_next_moe_shared_expert_gate_up_TID_"),
        family("qwen38_flash_next_moe_shared_expert_gate_up_prefill_TID_"),
        family("qwen38_flash_next_moe_shared_expert_down_TID_"),
        family("qwen38_flash_next_moe_shared_expert_down_prefill_TID_"),
    ];
    let combine = [
        family("qwen38_flash_next_moe_expert_combine_TID_"),
        family("qwen38_flash_next_moe_expert_combine_prefill_TID_"),
    ];
    let mtp = [
        family("qwen38_flash_next_mtp_expert_gate_up_TID_"),
        family("qwen38_flash_next_mtp_expert_gate_up_prefill_TID_"),
        family("qwen38_flash_next_mtp_expert_down_TID_"),
        family("qwen38_flash_next_mtp_expert_down_prefill_TID_"),
    ];
    for (label, group) in [
        ("routed gate/up decode", &routed[0]),
        ("routed gate/up prefill", &routed[1]),
        ("routed down decode", &routed[2]),
        ("routed down prefill", &routed[3]),
        ("shared gate/up decode", &shared[0]),
        ("shared gate/up prefill", &shared[1]),
        ("shared down decode", &shared[2]),
        ("shared down prefill", &shared[3]),
        ("combine decode", &combine[0]),
        ("combine prefill", &combine[1]),
        ("MTP gate/up decode", &mtp[0]),
        ("MTP gate/up prefill", &mtp[1]),
        ("MTP down decode", &mtp[2]),
        ("MTP down prefill", &mtp[3]),
    ] {
        require_count(
            &format!("Qwen3.8-Flash-Next MoE {label}"),
            group.len(),
            if label.ends_with("prefill") { 4 } else { 8 },
        )?;
    }

    for entry in routed.iter().flatten() {
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("routed entry `{}` lost its E2M1 decode", entry.name).into());
        }
    }
    for entry in shared.iter().flatten() {
        if entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!(
                "shared-expert entry `{}` decodes E2M1 from its BF16 plane",
                entry.name
            )
            .into());
        }
    }
    for entry in mtp.iter().flatten() {
        if entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(
                format!("MTP entry `{}` decodes E2M1 from its BF16 pool", entry.name).into(),
            );
        }
        if !entry.body.contains("9830400") {
            return Err(format!(
                "MTP entry `{}` lost its 9,830,400-byte slot stride",
                entry.name
            )
            .into());
        }
    }
    for entry in mtp[2..].iter().flatten() {
        if !entry.body.contains("6553600") {
            return Err(format!(
                "MTP down entry `{}` lost its 6,553,600-byte plane offset",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut routed_registers = Vec::new();
    let mut shared_registers = Vec::new();
    let mut combine_registers = Vec::new();
    let mut mtp_registers = Vec::new();
    for (groups, registers) in [
        (&routed[..], &mut routed_registers),
        (&shared[..], &mut shared_registers),
        (&combine[..], &mut combine_registers),
        (&mtp[..], &mut mtp_registers),
    ] {
        for entry in groups.iter().flatten() {
            if sass_function_body(sass, entry.name).is_none() {
                return Err(format!(
                    "cuobjdump omitted Qwen3.8-Flash-Next MoE expert SASS `{}`",
                    entry.name
                )
                .into());
            }
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
        }
        registers.sort_unstable();
    }
    require_registers(&baseline, "routed_registers", &routed_registers)?;
    require_registers(&baseline, "shared_registers", &shared_registers)?;
    require_registers(&baseline, "combine_registers", &combine_registers)?;
    require_registers(&baseline, "mtp_registers", &mtp_registers)?;

    println!(
        "Qwen3.8-Flash-Next MoE expert gate passed: 24 routed + 24 shared + 12 combine + 24 MTP entries, REG {routed_registers:?} / {shared_registers:?} / {combine_registers:?} / {mtp_registers:?}, STACK:0 LOCAL:0, E2M1 routed and BF16 shared/MTP representations, MTP slot stride, and SASS present"
    );
    Ok(())
}

/// Pins the plain BF16 MMA and store contract for all backbone projections.
fn gate_qwen38_flash_next_projections(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_PROJECTION_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let named = |prefix: &str| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    };
    let shapes = [
        (
            "GDN input",
            named("qwen38_flash_next_gdn_input_projection_TID_"),
            named("qwen38_flash_next_gdn_input_projection_prefill_TID_"),
            "gdn_input",
            256,
            8,
        ),
        (
            "QSA QKV",
            named("qwen38_flash_next_qsa_qkv_projection_TID_"),
            named("qwen38_flash_next_qsa_qkv_projection_prefill_TID_"),
            "qsa_qkv",
            256,
            8,
        ),
        (
            "indexer QK",
            named("qwen38_flash_next_indexer_qk_projection_TID_"),
            named("qwen38_flash_next_indexer_qk_projection_prefill_TID_"),
            "indexer_qk",
            128,
            8,
        ),
        (
            "block output",
            named("qwen38_flash_next_block_output_projection_TID_"),
            named("qwen38_flash_next_block_output_projection_prefill_TID_"),
            "block_output",
            128,
            8,
        ),
        (
            "MTP fusion",
            named("qwen38_flash_next_mtp_fusion_projection_TID_"),
            named("qwen38_flash_next_mtp_fusion_projection_prefill_TID_"),
            "mtp_fusion",
            256,
            1,
        ),
    ];

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for (label, decode, prefill, key, threads, decode_entries) in &shapes {
        require_count(
            &format!("Qwen3.8-Flash-Next {label} projection decode"),
            decode.len(),
            *decode_entries,
        )?;
        require_count(
            &format!("Qwen3.8-Flash-Next {label} projection prefill"),
            prefill.len(),
            4,
        )?;

        let mut registers = Vec::with_capacity(12);
        for entry in decode.iter().chain(prefill) {
            if !entry.body.contains(&format!(".reqntid {threads}, 1, 1")) {
                return Err(format!(
                    "entry `{}` lost its {threads}-thread launch bounds",
                    entry.name
                )
                .into());
            }
            if !entry.body.contains("mma.sync.aligned.m16n8k16")
                || !entry.body.contains("cvt.rn.bf16x2.f32")
            {
                return Err(format!(
                    "entry `{}` lost its BF16 MMA or nearest-rounded store",
                    entry.name
                )
                .into());
            }
            for forbidden in ["ex2.approx.f32", "lg2.approx.f32", "rcp.rn.f32"] {
                if entry.body.contains(forbidden) {
                    return Err(format!(
                        "plain projection `{}` contains epilogue instruction `{forbidden}`",
                        entry.name
                    )
                    .into());
                }
            }
            if sass_function_body(sass, entry.name).is_none() {
                return Err(format!(
                    "cuobjdump omitted Qwen3.8-Flash-Next projection SASS `{}`",
                    entry.name
                )
                .into());
            }
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            if resource.shared != 0 {
                return Err(format!(
                    "entry `{}` reserved {} shared bytes",
                    entry.name, resource.shared
                )
                .into());
            }
            registers.push(resource.registers);
        }
        registers.sort_unstable();
        require_registers(&baseline, &format!("{key}_registers"), &registers)?;
    }

    println!(
        "Qwen3.8-Flash-Next backbone projection gate passed: 53 entries, STACK:0 LOCAL:0 SHARED:0, BF16 MMA/store and SASS present"
    );
    Ok(())
}

/// Pins the exact BF16 LM-head entry inventory and plain projection contract.
fn gate_qwen38_flash_next_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN38_FLASH_NEXT_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen38_flash_next_bf16_lm_head_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.8-Flash-Next BF16 LM head", routes.len(), 8)?;
    require_count(
        "Qwen3.5 BF16 LM head",
        entries
            .iter()
            .filter(|entry| entry.name.starts_with("qwen35_bf16_lm_head_TID_"))
            .count(),
        8,
    )?;

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::with_capacity(routes.len());
    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("mma.sync.aligned.m16n8k16")
            || !entry.body.contains("cvt.rn.bf16x2.f32")
        {
            return Err(format!(
                "entry `{}` lost its BF16 MMA or nearest-rounded store",
                entry.name
            )
            .into());
        }
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.8-Flash-Next BF16 LM-head SASS `{}`",
                entry.name
            )
            .into());
        }
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        if resource.shared != 0 {
            return Err(format!(
                "entry `{}` reserved {} shared bytes",
                entry.name, resource.shared
            )
            .into());
        }
        registers.push(resource.registers);
    }
    registers.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;

    println!(
        "Qwen3.8-Flash-Next BF16 LM-head gate passed: 8 entries, REG {registers:?}, STACK:0 LOCAL:0 SHARED:0, BF16 MMA/store and SASS present"
    );
    Ok(())
}

fn gate_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_RECURRENCE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let recurrence = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_recurrence_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_recurrence_prefill_exact_TID_"))
        .collect::<Vec<_>>();
    let epilogue = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("gdn_recurrence_prefill_epilogue_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("GDN recurrence", recurrence.len(), 8)?;
    require_count("GDN recurrence causal/prefill", prefill.len(), 8)?;
    require_count("GDN recurrence prefill epilogue", epilogue.len(), 8)?;
    for entry in &recurrence {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/split-head launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &epilogue {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA epilogue launch bounds",
                entry.name
            )
            .into());
        }
    }
    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    for entry in recurrence.iter().chain(&prefill).chain(&epilogue) {
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!("cuobjdump omitted GDN recurrence SASS `{}`", entry.name).into());
        }
    }
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in recurrence {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "recurrence_registers", &registers)?;
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    if baseline.contains_key("prefill_recurrence_registers") {
        require_registers(
            &baseline,
            "prefill_recurrence_registers",
            &prefill_registers,
        )?;
    }
    if baseline.contains_key("prefill_recurrence_shared_bytes") {
        require_uniform_value(
            &baseline,
            "prefill_recurrence_shared_bytes",
            &prefill_shared,
        )?;
    }
    let mut epilogue_registers = Vec::new();
    let mut epilogue_shared = Vec::new();
    for entry in epilogue {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        epilogue_registers.push(resource.registers);
        epilogue_shared.push(resource.shared);
    }
    epilogue_registers.sort_unstable();
    if baseline.contains_key("prefill_epilogue_registers") {
        require_registers(&baseline, "prefill_epilogue_registers", &epilogue_registers)?;
    }
    if baseline.contains_key("prefill_epilogue_shared_bytes") {
        require_uniform_value(&baseline, "prefill_epilogue_shared_bytes", &epilogue_shared)?;
    }
    println!(
        "GDN recurrence gate passed: 8 decode + 4 causal + 4 prefill + 4 causal/4 prefill epilogue entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}, SASS present",
        registers, prefill_registers, epilogue_registers, shared, prefill_shared, epilogue_shared,
    );
    Ok(())
}

fn gate_gdn_state_snapshot(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_STATE_SNAPSHOT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let snapshots = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_state_snapshot_exact_TID_"))
        .collect::<Vec<_>>();
    require_count("GDN state snapshot", snapshots.len(), 1)?;
    let entry = snapshots[0];
    if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
        return Err(format!(
            "entry `{}` lost its 256-thread/two-CTA launch bounds",
            entry.name
        )
        .into());
    }
    if !entry.body.contains("ld.global.v2.b64") || !entry.body.contains("st.global.v2.b64") {
        return Err(format!(
            "entry `{}` lost its represented 16-byte load/store path",
            entry.name
        )
        .into());
    }

    let artifact = sm120_gate_artifact(root)?;
    let resource = artifact
        .resources
        .get(entry.name)
        .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
    require_spill_free(entry.name, resource)?;
    require_registers(&baseline, "snapshot_registers", &[resource.registers])?;
    require_uniform_value(&baseline, "snapshot_shared_bytes", &[resource.shared])?;
    if sass_function_body(artifact.sass()?, entry.name).is_none() {
        return Err(format!("cuobjdump omitted GDN snapshot SASS `{}`", entry.name).into());
    }
    println!(
        "GDN state snapshot gate passed: 1 exact entry, REG [{}], STACK:0 LOCAL:0, SHARED [{}], vectorized PTX and SASS present",
        resource.registers, resource.shared
    );
    Ok(())
}

fn gate_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(GDN_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_output_quantize"))
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("gdn_output_projection_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("gdn_output_projection_mma_exact_TID_")
        })
        .collect::<Vec<_>>();
    let macro_prefill = entries
        .iter()
        .filter(|entry| entry.name == "gdn_output_projection_mma_t1024")
        .collect::<Vec<_>>();
    require_count("GDN output quantization", quantize.len(), 1)?;
    require_count("GDN output projection", projection.len(), 8)?;
    require_count("GDN output T=32/64/128 projection", prefill.len(), 3)?;
    require_count("GDN output T=1024 projection", macro_prefill.len(), 1)?;
    for entry in quantize.iter().chain(&projection) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 64, 1, 1") || !entry.body.contains(".minnctapersm 4") {
            return Err(format!(
                "entry `{}` lost its 64-thread/four-CTA prefill launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !macro_prefill[0].body.contains(".reqntid 128, 1, 1")
        || !macro_prefill[0].body.contains(".minnctapersm 4")
    {
        return Err("GDN output T=1024 lost its 128-thread/four-CTA launch bounds".into());
    }
    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted GDN output quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    let quantize_sass = sass_function_body(sass, quantize[0].name)
        .ok_or("cuobjdump omitted GDN output quantization SASS")?;
    for instruction in ["SHFL.BFLY", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C"] {
        if !quantize_sass.contains(instruction) {
            return Err(format!(
                "entry `{}` lost required `{instruction}` SASS",
                quantize[0].name
            )
            .into());
        }
    }
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    let mut projection_registers = Vec::new();
    let mut projection_shared = Vec::new();
    for entry in projection {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.DOWN"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        projection_registers.push(resource.registers);
        projection_shared.push(resource.shared);
    }
    projection_registers.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;

    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in prefill.iter().chain(&macro_prefill) {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["QMMA.16832.F32.E4M3.E4M3", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    if baseline.contains_key("prefill_projection_registers") {
        require_registers(
            &baseline,
            "prefill_projection_registers",
            &prefill_registers,
        )?;
    }
    if baseline.contains_key("prefill_projection_shared_bytes") {
        require_uniform_value(
            &baseline,
            "prefill_projection_shared_bytes",
            &prefill_shared,
        )?;
    }
    println!(
        "GDN output gate passed: 1 quantize + 8 decode + 4 prefill projection entries, REG {} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?} / {:?}, E4M3/SHFL/QMMA/LDGSTS present",
        quantize_resource.registers,
        projection_registers,
        prefill_registers,
        quantize_resource.shared,
        projection_shared,
        prefill_shared,
    );
    Ok(())
}

fn gate_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "attention_qk_prepare_exact_TID_",
        Some(("attention_qk_prepare_prefill_exact_TID_", 4)),
        "attention Q/K prepare",
        "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
        "E4M3",
    )
}

fn gate_qwen35_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        QWEN35_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "qwen35_attention_qk_prepare_exact_TID_",
        Some(("qwen35_attention_qk_prepare_prefill_exact_TID_", 4)),
        "Qwen3.5 attention Q/K prepare",
        "F2FP.BF16.F32.PACK_AB",
        "BF16",
    )
}

fn gate_mtp_bf16_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        MTP_BF16_QK_PREPARE_RESOURCE_BASELINE,
        "mtp_bf16_qk_prepare_TID_",
        Some(("mtp_bf16_qk_prepare_prefill_TID_", 4)),
        "MTP BF16 Q/K prepare",
        "F2FP.BF16.F32.PACK_AB",
        "BF16",
    )
}

fn gate_qwen36_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        QWEN36_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "qwen36_attention_qk_prepare_exact_TID_",
        Some(("qwen36_attention_qk_prepare_prefill_exact_TID_", 3)),
        "Qwen3.6 attention Q/K prepare",
        "F2FP.BF16.F32.PACK_AB",
        "BF16",
    )
}

fn gate_qwen36_fp8_attention_qk_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_attention_qk_prepare_target(
        root,
        QWEN36_FP8_ATTENTION_QK_PREPARE_RESOURCE_BASELINE,
        "qwen36_fp8_attention_qk_prepare_exact_TID_",
        Some(("qwen36_fp8_attention_qk_prepare_prefill_exact_TID_", 3)),
        "Qwen3.6 FP8 attention Q/K prepare",
        "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
        "E4M3",
    )
}

fn gate_attention_qk_prepare_target(
    root: &Path,
    baseline_path: &str,
    entry_prefix: &str,
    prefill_inventory: Option<(&str, usize)>,
    label: &str,
    cache_instruction: &str,
    cache_label: &str,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let prepare = entries
        .iter()
        .filter(|entry| entry.name.starts_with(entry_prefix))
        .collect::<Vec<_>>();
    let prefill = prefill_inventory.map_or_else(Vec::new, |(prefix, _)| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    });
    require_count(label, prepare.len(), 8)?;
    if let Some((_, expected)) = prefill_inventory {
        require_count("attention Q/K prefill preparation", prefill.len(), expected)?;
    }
    for entry in prepare.iter().chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut decode_registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, registers) in [
        (&prepare, &mut decode_registers),
        (&prefill, &mut prefill_registers),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["MUFU.RSQ", "SHFL.BFLY", cache_instruction] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    decode_registers.sort_unstable();
    prefill_registers.sort_unstable();
    require_registers(&baseline, "prepare_registers", &decode_registers)?;
    if baseline.contains_key("prefill_prepare_registers") {
        require_registers(&baseline, "prefill_prepare_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "{label} gate passed: {} decode + {} prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, RSQ/SHFL/{cache_label} present",
        prepare.len(),
        prefill.len(),
        decode_registers,
        prefill_registers,
        shared
    );
    Ok(())
}

fn gate_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(PAGED_GQA_RESOURCE_BASELINE))?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let attention = entries
        .iter()
        .filter(|entry| entry.name.starts_with("paged_gqa_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_shared_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_partials = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_flash_p8_exact_TID_")
                || entry
                    .name
                    .starts_with("paged_gqa_prefill_flash_p16_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_reductions = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_partitioned_reduce_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_macro_partials = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_flash_macro_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_macro_reductions = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("paged_gqa_prefill_macro_reduce_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("paged GQA", attention.len(), 8)?;
    require_count("shared prefill paged GQA", prefill.len(), 3)?;
    require_count("partitioned prefill paged GQA", prefill_partials.len(), 2)?;
    require_count(
        "partitioned prefill paged GQA reduction",
        prefill_reductions.len(),
        2,
    )?;
    require_count(
        "macro prefill paged GQA partition",
        prefill_macro_partials.len(),
        1,
    )?;
    require_count(
        "macro prefill paged GQA reduction",
        prefill_macro_reductions.len(),
        5,
    )?;
    for entry in &attention {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!("entry `{}` lost dynamic shared memory", entry.name).into());
        }
    }
    for entry in &prefill_partials {
        let expected_ctas = if entry.name.contains("_p8_") { 1 } else { 2 };
        if !entry.body.contains(".reqntid 256, 1, 1")
            || !entry
                .body
                .contains(&format!(".minnctapersm {expected_ctas}"))
        {
            return Err(format!(
                "entry `{}` lost its 256-thread/{expected_ctas}-CTA launch bounds",
                entry.name,
            )
            .into());
        }
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!("entry `{}` lost dynamic shared memory", entry.name).into());
        }
    }
    for entry in &prefill_reductions {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill_macro_partials {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name,
            )
            .into());
        }
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!("entry `{}` lost dynamic shared memory", entry.name).into());
        }
    }
    for entry in &prefill_macro_reductions {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_partial_registers = Vec::new();
    let mut prefill_reduce_registers = Vec::new();
    let mut prefill_macro_partial_registers = Vec::new();
    let mut prefill_macro_reduce_registers = Vec::new();
    let mut decode_shared = Vec::new();
    let mut shared = Vec::new();
    for entry in attention {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        decode_shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill_partials {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_partial_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in [
            "F2FP.F16.E4M3.UNPACK_B",
            "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
            "QMMA.16832.F32.E4M3.E4M3",
            "HMMA.16816.F32",
            "SHFL.BFLY",
            "MUFU.EX2",
            "LDGSTS",
        ] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill_reductions {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_reduce_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("MUFU.EX2") {
            return Err(format!("entry `{}` lost required `MUFU.EX2` SASS", entry.name).into());
        }
    }
    for entry in prefill_macro_partials {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_macro_partial_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in [
            "F2FP.F16.E4M3.UNPACK_B",
            "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C",
            "QMMA.16832.F32.E4M3.E4M3",
            "HMMA.16816.F32",
            "SHFL.BFLY",
            "MUFU.EX2",
            "LDGSTS",
        ] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
    }
    for entry in prefill_macro_reductions {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        prefill_macro_reduce_registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        if !body.contains("MUFU.EX2") {
            return Err(format!("entry `{}` lost required `MUFU.EX2` SASS", entry.name).into());
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    prefill_partial_registers.sort_unstable();
    prefill_reduce_registers.sort_unstable();
    prefill_macro_partial_registers.sort_unstable();
    prefill_macro_reduce_registers.sort_unstable();
    require_registers(&baseline, "attention_registers", &registers)?;
    if baseline.contains_key("prefill_shared_registers") {
        require_registers(&baseline, "prefill_shared_registers", &prefill_registers)?;
    }
    if baseline.contains_key("prefill_flash_partition_registers") {
        require_registers(
            &baseline,
            "prefill_flash_partition_registers",
            &prefill_partial_registers,
        )?;
    }
    if baseline.contains_key("prefill_reduce_registers") {
        require_registers(
            &baseline,
            "prefill_reduce_registers",
            &prefill_reduce_registers,
        )?;
    }
    if baseline.contains_key("prefill_flash_macro_registers") {
        require_registers(
            &baseline,
            "prefill_flash_macro_registers",
            &prefill_macro_partial_registers,
        )?;
    }
    if baseline.contains_key("prefill_macro_reduce_registers") {
        require_registers(
            &baseline,
            "prefill_macro_reduce_registers",
            &prefill_macro_reduce_registers,
        )?;
    }
    require_uniform_value(&baseline, "decode_shared_bytes", &decode_shared)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "paged GQA gate passed: 8 decode + 3 shared + 2 flash partition + 2 reduction + 1 macro flash + 5 macro reduction entries, REG {:?} / {:?} / {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, DECODE SHARED {:?}, SHARED {:?}, FP8-QMMA/F16-HMMA/E4M3/SHFL/EX2/LDGSTS present",
        registers,
        prefill_registers,
        prefill_partial_registers,
        prefill_reduce_registers,
        prefill_macro_partial_registers,
        prefill_macro_reduce_registers,
        decode_shared,
        shared
    );
    Ok(())
}

fn gate_qwen35_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_paged_gqa_target(
        root,
        QWEN35_PAGED_GQA_RESOURCE_BASELINE,
        "qwen35_paged_gqa_exact_TID_",
        Some(("qwen35_paged_gqa_prefill_shared_exact_TID_", 3, 128)),
        "Qwen3.5",
        false,
    )
}

fn gate_qwen36_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_paged_gqa_target(
        root,
        QWEN36_PAGED_GQA_RESOURCE_BASELINE,
        "qwen36_paged_gqa_exact_TID_",
        Some(("qwen36_paged_gqa_prefill_shared_exact_TID_", 3, 256)),
        "Qwen3.6",
        false,
    )
}

fn gate_qwen36_fp8_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    gate_paged_gqa_target(
        root,
        QWEN36_FP8_PAGED_GQA_RESOURCE_BASELINE,
        "qwen36_fp8_paged_gqa_exact_TID_",
        Some(("qwen36_fp8_paged_gqa_prefill_shared_exact_TID_", 3, 256)),
        "Qwen3.6",
        true,
    )
}

fn gate_paged_gqa_target(
    root: &Path,
    baseline_path: &str,
    entry_prefix: &str,
    prefill_inventory: Option<(&str, usize, u32)>,
    target: &str,
    e4m3_cache: bool,
) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(root.join(baseline_path))?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let attention = entries
        .iter()
        .filter(|entry| entry.name.starts_with(entry_prefix))
        .collect::<Vec<_>>();
    let prefill = prefill_inventory.map_or_else(Vec::new, |(prefix, _, _)| {
        entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect::<Vec<_>>()
    });
    let cache = if e4m3_cache { "FP8" } else { "BF16" };
    require_count(&format!("{target} {cache} paged GQA"), attention.len(), 8)?;
    if let Some((_, expected, _)) = prefill_inventory {
        require_count(
            &format!("{target} {cache} paged GQA prefill"),
            prefill.len(),
            expected,
        )?;
    }
    for entry in &attention {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        let threads = prefill_inventory.map_or(0, |(_, _, threads)| threads);
        if !entry.body.contains(&format!(".reqntid {threads}, 1, 1"))
            || !entry.body.contains(".minnctapersm 1")
        {
            return Err(format!(
                "entry `{}` lost its {threads}-thread/one-CTA launch bounds",
                entry.name,
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    let decode_instructions = if e4m3_cache {
        &["F2FP.F16.E4M3.UNPACK_B", "LDGSTS", "SHFL.BFLY", "MUFU.EX2"][..]
    } else {
        &["LDGSTS", "SHFL.BFLY", "MUFU.EX2"][..]
    };
    let prefill_instructions = if e4m3_cache {
        &["F2FP.F16.E4M3.UNPACK_B", "LDGSTS", "SHFL.BFLY", "MUFU.EX2"][..]
    } else {
        &["LDGSTS", "SHFL.BFLY", "MUFU.EX2"][..]
    };
    for (entries, registers, shared, instructions) in [
        (&attention, &mut registers, &mut shared, decode_instructions),
        (
            &prefill,
            &mut prefill_registers,
            &mut prefill_shared,
            prefill_instructions,
        ),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            if !e4m3_cache && body.contains("F2FP.F16.E4M3.UNPACK_B") {
                return Err(format!(
                    "entry `{}` unexpectedly decodes {target} cache as E4M3",
                    entry.name,
                )
                .into());
            }
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    require_registers(&baseline, "attention_registers", &registers)?;
    if baseline.contains_key("prefill_attention_registers") {
        require_registers(&baseline, "prefill_attention_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;
    if baseline.contains_key("prefill_shared_bytes") {
        require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;
    }

    println!(
        "{target} {cache} paged GQA gate passed: 8 decode + {} prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}, {}/LDGSTS/SHFL/EX2 present",
        prefill.len(),
        registers,
        prefill_registers,
        shared,
        prefill_shared,
        if e4m3_cache { "E4M3" } else { "U16" },
    );
    Ok(())
}

fn gate_mtp_bf16_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(MTP_BF16_PAGED_GQA_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let attention = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_paged_gqa_TID_"))
        .collect::<Vec<_>>();
    require_count("MTP BF16 paged GQA", attention.len(), 8)?;
    for entry in &attention {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in attention {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        registers.push(resource.registers);
        shared.push(resource.shared);

        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["LDG.E.U16", "SHFL.BFLY", "MUFU.EX2"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        if body.contains("F2FP.F16.E4M3.UNPACK_B") {
            return Err(format!(
                "entry `{}` unexpectedly decodes the MTP cache as E4M3",
                entry.name
            )
            .into());
        }
    }
    registers.sort_unstable();
    require_registers(&baseline, "attention_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "MTP BF16 paged GQA gate passed: 8 decode entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}, U16/SHFL/EX2 present and E4M3 absent",
        registers, shared
    );
    Ok(())
}

fn gate_long_context_paged_gqa(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(LONG_CONTEXT_PAGED_GQA_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let partials = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("long_context_paged_gqa_partial_exact_TID_")
        })
        .collect::<Vec<_>>();
    let reductions = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("long_context_paged_gqa_reduce_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("long-context paged GQA partial", partials.len(), 8)?;
    require_count("long-context paged GQA reduction", reductions.len(), 8)?;
    for entry in partials.iter().chain(&reductions) {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 16") {
            return Err(format!(
                "entry `{}` lost its 32-thread/sixteen-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &reductions {
        if !entry.body.contains("__dynamic_smem__") {
            return Err(format!(
                "entry `{}` lost its dynamic partition-weight workspace",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut partial_registers = Vec::new();
    let mut reduction_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, registers, instructions) in [
        (
            &partials,
            &mut partial_registers,
            &["F2FP.F16.E4M3.UNPACK_B", "SHFL.BFLY", "MUFU.EX2"][..],
        ),
        (
            &reductions,
            &mut reduction_registers,
            &["SHFL.BFLY", "MUFU.EX2", "STS", "LDS", "BAR.SYNC"][..],
        ),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            shared.push(resource.shared);

            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }
    partial_registers.sort_unstable();
    reduction_registers.sort_unstable();
    require_registers(&baseline, "partial_registers", &partial_registers)?;
    require_registers(&baseline, "reduction_registers", &reduction_registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "long-context paged GQA gate passed: 8 partial + 8 reduction entries, REG partial {:?}, reduction {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/SHFL/EX2/dynamic-shared present",
        partial_registers, reduction_registers, shared
    );
    Ok(())
}

fn gate_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(ATTENTION_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("attention_gate_quantize_exact_TID_"))
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("attention_output_projection_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("attention_output_projection_mma_exact_TID_")
        })
        .collect::<Vec<_>>();
    let macro_prefill = entries
        .iter()
        .filter(|entry| entry.name == "attention_output_projection_mma_t1024")
        .collect::<Vec<_>>();
    require_count("attention-output gate quantization", quantize.len(), 1)?;
    require_count("attention-output projection", projection.len(), 8)?;
    require_count("attention-output T=32/64/128 projection", prefill.len(), 3)?;
    require_count("attention-output T=1024 projection", macro_prefill.len(), 1)?;
    for entry in quantize.iter().chain(&projection) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 64, 1, 1") || !entry.body.contains(".minnctapersm 4") {
            return Err(format!(
                "entry `{}` lost its 64-thread/four-CTA prefill launch bounds",
                entry.name
            )
            .into());
        }
    }
    if !macro_prefill[0].body.contains(".reqntid 128, 1, 1")
        || !macro_prefill[0].body.contains(".minnctapersm 4")
    {
        return Err("attention-output T=1024 lost its 128-thread/four-CTA launch bounds".into());
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let quantize_resource = resources
        .get(quantize[0].name)
        .ok_or("cuobjdump omitted attention-output gate quantization")?;
    require_spill_free(quantize[0].name, quantize_resource)?;
    let quantize_sass = sass_function_body(sass, quantize[0].name)
        .ok_or("cuobjdump omitted attention-output gate quantization SASS")?;
    for instruction in ["MUFU.EX2", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C"] {
        if !quantize_sass.contains(instruction) {
            return Err(format!(
                "entry `{}` lost required `{instruction}` SASS",
                quantize[0].name
            )
            .into());
        }
    }
    require_registers(
        &baseline,
        "quantize_registers",
        &[quantize_resource.registers],
    )?;
    require_uniform_value(
        &baseline,
        "quantize_shared_bytes",
        &[quantize_resource.shared],
    )?;

    let mut projection_registers = Vec::new();
    let mut projection_shared = Vec::new();
    for entry in projection {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["F2FP.F16.E4M3.UNPACK_B", "SHFL.DOWN"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        projection_registers.push(resource.registers);
        projection_shared.push(resource.shared);
    }
    projection_registers.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_uniform_value(&baseline, "projection_shared_bytes", &projection_shared)?;

    let mut prefill_registers = Vec::new();
    let mut prefill_shared = Vec::new();
    for entry in prefill.iter().chain(&macro_prefill) {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["QMMA.16832.F32.E4M3.E4M3", "LDGSTS"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        prefill_registers.push(resource.registers);
        prefill_shared.push(resource.shared);
    }
    prefill_registers.sort_unstable();
    if baseline.contains_key("prefill_projection_registers") {
        require_registers(
            &baseline,
            "prefill_projection_registers",
            &prefill_registers,
        )?;
    }
    if baseline.contains_key("prefill_projection_shared_bytes") {
        require_uniform_value(
            &baseline,
            "prefill_projection_shared_bytes",
            &prefill_shared,
        )?;
    }

    println!(
        "attention output gate passed: 1 quantize + 8 decode + 4 prefill projection entries, REG {} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {} / {:?} / {:?}, EX2/E4M3/SHFL/QMMA/LDGSTS present",
        quantize_resource.registers,
        projection_registers,
        prefill_registers,
        quantize_resource.shared,
        projection_shared,
        prefill_shared,
    );
    Ok(())
}

struct ExactResourceFamily<'a> {
    label: &'a str,
    prefix: &'a str,
    count: usize,
    threads: u32,
    minimum_ctas_per_sm: u32,
    register_key: &'a str,
    ptx_instructions: &'a [&'a str],
    sass_instructions: &'a [&'a str],
    forbidden_sass: &'a [&'a str],
}

fn gate_qwen35_mtp_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_MTP_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let artifact = sm120_gate_artifact(root)?;
    let sass = artifact.sass()?;
    let bf16_mma_ptx = [
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "cvt.rn.bf16x2.f32",
    ];
    let bf16_mma_sass = ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"];
    let families = [
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 fusion",
            prefix: "qwen35_mtp_bf16_fusion_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "fusion_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 fusion prefill",
            prefix: "qwen35_mtp_bf16_fusion_prefill_TID_",
            count: 3,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "fusion_prefill_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 QKV",
            prefix: "qwen35_mtp_bf16_qkv_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qkv_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 QKV prefill",
            prefix: "qwen35_mtp_bf16_qkv_prefill_TID_",
            count: 3,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qkv_prefill_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 Q/K preparation",
            prefix: "qwen35_mtp_bf16_qk_prepare_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qk_prepare_registers",
            ptx_instructions: &[],
            sass_instructions: &["MUFU.RSQ", "SHFL.BFLY", "F2FP.BF16.F32.PACK_AB"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 Q/K preparation prefill",
            prefix: "qwen35_mtp_bf16_qk_prepare_prefill_TID_",
            count: 3,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qk_prepare_prefill_registers",
            ptx_instructions: &[],
            sass_instructions: &["MUFU.RSQ", "SHFL.BFLY", "F2FP.BF16.F32.PACK_AB"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 paged GQA",
            prefix: "qwen35_mtp_bf16_paged_gqa_TID_",
            count: 8,
            threads: 32,
            minimum_ctas_per_sm: 16,
            register_key: "paged_gqa_registers",
            ptx_instructions: &[],
            sass_instructions: &["LDGSTS", "SHFL.BFLY", "MUFU.EX2"],
            forbidden_sass: &["F2FP.F16.E4M3.UNPACK_B"],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 attention gate",
            prefix: "qwen35_mtp_bf16_attention_gate_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "attention_gate_registers",
            ptx_instructions: &["ex2.approx.f32", "rcp.rn.f32", "st.global.b16"],
            sass_instructions: &["MUFU.EX2", "MUFU.RCP", "STG.E.U16"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 attention output",
            prefix: "qwen35_mtp_bf16_attention_output_TID_",
            count: 8,
            threads: 128,
            minimum_ctas_per_sm: 4,
            register_key: "attention_output_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 SwiGLU",
            prefix: "qwen35_mtp_bf16_swiglu_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "swiglu_registers",
            ptx_instructions: &[
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                "ex2.approx.f32",
                "cvt.rn.bf16x2.f32",
            ],
            sass_instructions: &["HMMA.16816.F32.BF16", "MUFU.EX2", "F2FP.BF16.F32.PACK_AB"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.5 MTP BF16 down",
            prefix: "qwen35_mtp_bf16_down_TID_",
            count: 8,
            threads: 128,
            minimum_ctas_per_sm: 4,
            register_key: "down_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
    ];

    let entry_count = gate_exact_resource_families(
        &baseline,
        entries,
        artifact,
        sass,
        &families,
        SharedFootprint::Uniform,
    )?;
    println!(
        "Qwen3.5 MTP resource gate passed: {entry_count} entries, STACK:0 LOCAL:0, SHARED:1024; launch bounds, BF16 HMMA/cache paths, and register envelopes retained"
    );
    Ok(())
}

/// How a family group pins its per-entry shared-memory footprint.
///
/// `Uniform` is the stronger claim and is kept wherever it is still true. `PerEntry` exists
/// because the Qwen3.6 MTP family spans two device-codegen crates after the kernel split
/// (`-mtp` and `-moe`), so its entries no longer share one module arena and their real
/// footprints differ. The Qwen3.5 MTP family lives entirely in `-mtp` and stays uniform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SharedFootprint {
    Uniform,
    PerEntry,
}

fn gate_exact_resource_families(
    baseline: &Baseline,
    entries: &[Entry<'_>],
    artifact: &Sm120GateArtifact,
    sass: &str,
    families: &[ExactResourceFamily<'_>],
    shared_footprint: SharedFootprint,
) -> Result<usize, Box<dyn Error>> {
    let mut all_shared = Vec::new();
    for family in families {
        let matched = entries
            .iter()
            .filter(|entry| entry.name.starts_with(family.prefix))
            .collect::<Vec<_>>();
        require_count(family.label, matched.len(), family.count)?;
        let launch = format!(".reqntid {}, 1, 1", family.threads);
        let residency = format!(".minnctapersm {}", family.minimum_ctas_per_sm);
        let mut registers = Vec::with_capacity(matched.len());
        for entry in matched {
            if !entry.body.contains(&launch) || !entry.body.contains(&residency) {
                return Err(format!(
                    "entry `{}` lost its {}-thread/{}-CTA launch bounds",
                    entry.name, family.threads, family.minimum_ctas_per_sm
                )
                .into());
            }
            for instruction in family.ptx_instructions {
                if !entry.body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` PTX",
                        entry.name
                    )
                    .into());
                }
            }

            let resource = artifact
                .resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            registers.push(resource.registers);
            all_shared.push(resource.shared);
            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in family.sass_instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            for instruction in family.forbidden_sass {
                if body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` unexpectedly contains `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
        }
        registers.sort_unstable();
        require_registers(baseline, family.register_key, &registers)?;
    }
    match shared_footprint {
        SharedFootprint::Uniform => require_uniform_value(baseline, "shared_bytes", &all_shared)?,
        SharedFootprint::PerEntry => require_registers(baseline, "shared_bytes", &all_shared)?,
    }
    Ok(all_shared.len())
}

fn gate_qwen36_mtp_resources(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_MTP_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let artifact = sm120_gate_artifact(root)?;
    let sass = artifact.sass()?;
    let bf16_mma_ptx = [
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "cvt.rn.bf16x2.f32",
    ];
    let bf16_mma_sass = ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"];
    let families = [
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 fusion",
            prefix: "qwen36_mtp_bf16_fusion_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "fusion_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 fusion prefill",
            prefix: "qwen36_mtp_bf16_fusion_prefill_TID_",
            count: 3,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "fusion_prefill_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 QKV",
            prefix: "qwen36_mtp_bf16_qkv_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qkv_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 QKV prefill",
            prefix: "qwen36_mtp_bf16_qkv_prefill_TID_",
            count: 3,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "qkv_prefill_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 attention gate",
            prefix: "qwen36_mtp_bf16_attention_gate_TID_",
            count: 8,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "attention_gate_registers",
            ptx_instructions: &["ex2.approx.f32", "rcp.rn.f32", "st.global.b16"],
            sass_instructions: &["MUFU.EX2", "MUFU.RCP", "STG.E.U16"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 attention output",
            prefix: "qwen36_mtp_bf16_attention_output_TID_",
            count: 8,
            threads: 128,
            minimum_ctas_per_sm: 4,
            register_key: "attention_output_registers",
            ptx_instructions: &bf16_mma_ptx,
            sass_instructions: &bf16_mma_sass,
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 expert gate/up",
            prefix: "qwen36_mtp_bf16_expert_gate_up_TID_",
            count: 11,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "expert_gate_up_registers",
            ptx_instructions: &[
                "fma.rn.f32",
                "shfl.sync.down.b32",
                "ex2.approx.f32",
                "div.rn.f32",
                "st.global.b16",
            ],
            sass_instructions: &["FFMA", "SHFL.DOWN", "MUFU.EX2", "MUFU.RCP", "STG.E.U16"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 expert down",
            prefix: "qwen36_mtp_bf16_expert_down_TID_",
            count: 11,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "expert_down_registers",
            ptx_instructions: &["fma.rn.f32", "shfl.sync.down.b32", "st.global.b16"],
            sass_instructions: &["FFMA", "SHFL.DOWN", "STG.E.U16"],
            forbidden_sass: &[],
        },
        ExactResourceFamily {
            label: "Qwen3.6 MTP BF16 expert combine",
            prefix: "qwen36_mtp_bf16_expert_combine_TID_",
            count: 11,
            threads: 256,
            minimum_ctas_per_sm: 2,
            register_key: "expert_combine_registers",
            ptx_instructions: &[
                "fma.rn.f32",
                "ex2.approx.f32",
                "rcp.rn.f32",
                "st.global.b16",
            ],
            sass_instructions: &["FFMA", "MUFU.EX2", "MUFU.RCP", "STG.E.U16"],
            forbidden_sass: &[],
        },
    ];

    let entry_count = gate_exact_resource_families(
        &baseline,
        entries,
        artifact,
        sass,
        &families,
        SharedFootprint::PerEntry,
    )?;
    println!(
        "Qwen3.6 MTP resource gate passed: {entry_count} entries, STACK:0 LOCAL:0, per-entry SHARED pinned; launch bounds, BF16 tensor/scalar paths, and register envelopes retained"
    );
    Ok(())
}

fn gate_mtp_bf16_fusion(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(MTP_BF16_FUSION_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_fusion_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_fusion_prefill_TID_"))
        .collect::<Vec<_>>();
    require_count("MTP BF16 fusion", routes.len(), 8)?;
    require_count("MTP BF16 fusion prefill", prefill.len(), 4)?;
    for entry in routes.iter().chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, entry_registers) in [
        (&routes, &mut registers),
        (&prefill, &mut prefill_registers),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            entry_registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    require_registers(&baseline, "fusion_registers", &registers)?;
    if baseline.contains_key("prefill_fusion_registers") {
        require_registers(&baseline, "prefill_fusion_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "fusion_shared_bytes", &shared)?;

    println!(
        "MTP BF16 fusion gate passed: 8 decode + 4 prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, BF16 HMMA/pack present",
        registers, prefill_registers, shared
    );
    Ok(())
}

fn gate_mtp_bf16_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(MTP_BF16_ATTENTION_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let gates = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_attention_gate_TID_"))
        .collect::<Vec<_>>();
    let projections = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_attention_output_TID_"))
        .collect::<Vec<_>>();
    require_count("MTP BF16 attention gate", gates.len(), 8)?;
    require_count("MTP BF16 attention-output projection", projections.len(), 8)?;
    for entry in &gates {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["ex2.approx.f32", "rcp.rn.f32", "st.global.b16"] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }
    for entry in &projections {
        if !entry.body.contains(".reqntid 128, 1, 1") || !entry.body.contains(".minnctapersm 4") {
            return Err(format!(
                "entry `{}` lost its 128-thread/four-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut gate_registers = Vec::new();
    let mut gate_shared = Vec::new();
    for entry in gates {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["MUFU.EX2", "MUFU.RCP", "STG.E.U16"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        gate_registers.push(resource.registers);
        gate_shared.push(resource.shared);
    }
    gate_registers.sort_unstable();
    require_registers(&baseline, "gate_registers", &gate_registers)?;
    require_uniform_value(&baseline, "gate_shared_bytes", &gate_shared)?;

    let mut projection_registers = Vec::new();
    let mut projection_shared = Vec::new();
    for entry in projections {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        projection_registers.push(resource.registers);
        projection_shared.push(resource.shared);
    }
    projection_registers.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_uniform_value(&baseline, "projection_shared_bytes", &projection_shared)?;

    println!(
        "MTP BF16 attention-output gate passed: 8 gate + 8 projection entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}, EX2/BF16 HMMA/pack present",
        gate_registers, projection_registers, gate_shared, projection_shared
    );
    Ok(())
}

fn gate_mtp_bf16_mlp(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(MTP_BF16_MLP_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let swiglu = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_swiglu_TID_"))
        .collect::<Vec<_>>();
    let down = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_down_TID_"))
        .collect::<Vec<_>>();
    require_count("MTP BF16 SwiGLU", swiglu.len(), 8)?;
    require_count("MTP BF16 down", down.len(), 8)?;
    for entry in &swiglu {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "ex2.approx.f32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }
    for entry in &down {
        if !entry.body.contains(".reqntid 128, 1, 1") || !entry.body.contains(".minnctapersm 4") {
            return Err(format!(
                "entry `{}` lost its 128-thread/four-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut swiglu_registers = Vec::new();
    let mut swiglu_shared = Vec::new();
    for entry in swiglu {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["HMMA.16816.F32.BF16", "MUFU.EX2", "F2FP.BF16.F32.PACK_AB"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        swiglu_registers.push(resource.registers);
        swiglu_shared.push(resource.shared);
    }
    swiglu_registers.sort_unstable();
    require_registers(&baseline, "swiglu_registers", &swiglu_registers)?;
    require_uniform_value(&baseline, "swiglu_shared_bytes", &swiglu_shared)?;

    let mut down_registers = Vec::new();
    let mut down_shared = Vec::new();
    for entry in down {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
        for instruction in ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        down_registers.push(resource.registers);
        down_shared.push(resource.shared);
    }
    down_registers.sort_unstable();
    require_registers(&baseline, "down_registers", &down_registers)?;
    require_uniform_value(&baseline, "down_shared_bytes", &down_shared)?;

    println!(
        "MTP BF16 MLP gate passed: 8 SwiGLU + 8 down entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?}, BF16 HMMA/EX2/pack present",
        swiglu_registers, down_registers, swiglu_shared, down_shared
    );
    Ok(())
}

fn gate_mtp_bf16_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(MTP_BF16_QKV_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_qkv_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| entry.name.starts_with("mtp_bf16_qkv_prefill_TID_"))
        .collect::<Vec<_>>();
    require_count("MTP BF16 QKV", routes.len(), 8)?;
    require_count("MTP BF16 QKV prefill", prefill.len(), 4)?;
    for entry in routes.iter().chain(&prefill) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut shared = Vec::new();
    for (entries, entry_registers) in [
        (&routes, &mut registers),
        (&prefill, &mut prefill_registers),
    ] {
        for entry in entries {
            let resource = resources
                .get(entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}`", entry.name))?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name)
                .ok_or_else(|| format!("cuobjdump omitted `{}` SASS", entry.name))?;
            for instruction in ["HMMA.16816.F32.BF16", "F2FP.BF16.F32.PACK_AB"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            entry_registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    require_registers(&baseline, "qkv_registers", &registers)?;
    if baseline.contains_key("prefill_qkv_registers") {
        require_registers(&baseline, "prefill_qkv_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "qkv_shared_bytes", &shared)?;

    println!(
        "MTP BF16 QKV gate passed: 8 decode + 4 prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, BF16 HMMA/pack present",
        registers, prefill_registers, shared
    );
    Ok(())
}

fn gate_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(NVFP4_DOWN_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;

    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| {
            entry.name == "nvfp4_down_a16_b1" || entry.name.starts_with("nvfp4_down_a16_TID_")
        })
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_down_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("nvfp4_down_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("NVFP4 down", routes.len(), 8)?;
    require_count("NVFP4 down prefill quantization", quantize.len(), 4)?;
    require_count("NVFP4 down W4A4 prefill", w4a4.len(), 4)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 down entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!("cuobjdump omitted NVFP4 down SASS `{}`", entry.name).into());
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted NVFP4 down quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted NVFP4 down quantization SASS `{}`",
                entry.name
            )
            .into());
        }
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    require_registers(&baseline, "prefill_quantize_registers", &quantize_registers)?;
    require_uniform_value(&baseline, "prefill_quantize_shared_bytes", &quantize_shared)?;

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 down W4A4 entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name)
            .ok_or_else(|| format!("cuobjdump omitted NVFP4 down W4A4 SASS `{}`", entry.name))?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    require_registers(&baseline, "prefill_w4a4_registers", &w4a4_registers)?;
    require_uniform_value(&baseline, "prefill_w4a4_shared_bytes", &w4a4_shared)?;

    println!(
        "NVFP4 down gate passed: 8 A16 + 4 prefill quantize + 4 W4A4 entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        registers, quantize_registers, w4a4_registers, shared, quantize_shared, w4a4_shared
    );
    Ok(())
}

fn gate_qwen35_nvfp4_down(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_DOWN_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_down_a16_TID_"))
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_down_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_down_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 down", routes.len(), 8)?;
    require_count("Qwen3.5 NVFP4 down prefill quantization", quantize.len(), 4)?;
    require_count("Qwen3.5 NVFP4 down W4A4 prefill", w4a4.len(), 4)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(
                format!("cuobjdump omitted Qwen3.5 NVFP4 down SASS `{}`", entry.name).into(),
            );
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down quantization SASS `{}`",
                entry.name
            )
            .into());
        }
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    if baseline.contains_key("prefill_quantize_registers") {
        require_registers(&baseline, "prefill_quantize_registers", &quantize_registers)?;
        require_uniform_value(&baseline, "prefill_quantize_shared_bytes", &quantize_shared)?;
    }

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down W4A4 entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 down W4A4 SASS `{}`",
                entry.name
            )
        })?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    if baseline.contains_key("prefill_w4a4_registers") {
        require_registers(&baseline, "prefill_w4a4_registers", &w4a4_registers)?;
        require_uniform_value(&baseline, "prefill_w4a4_shared_bytes", &w4a4_shared)?;
    }

    println!(
        "Qwen3.5 NVFP4 down gate passed: 8 A16 + 4 prefill quantize + 4 W4A4 entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        registers, quantize_registers, w4a4_registers, shared, quantize_shared, w4a4_shared
    );
    Ok(())
}

fn gate_qwen35_nvfp4_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_QKV_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_qkv_a16_TID_"))
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_qkv_quantize_TID_"))
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_qkv_w4a4_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 QKV", routes.len(), 8)?;
    require_count("Qwen3.5 NVFP4 QKV prefill quantization", quantize.len(), 4)?;
    require_count("Qwen3.5 NVFP4 QKV W4A4 prefill", w4a4.len(), 4)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources
            .get(entry.name)
            .ok_or_else(|| format!("cuobjdump omitted Qwen3.5 NVFP4 QKV entry `{}`", entry.name))?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(
                format!("cuobjdump omitted Qwen3.5 NVFP4 QKV SASS `{}`", entry.name).into(),
            );
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 QKV quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.5 NVFP4 QKV quantization SASS `{}`",
                entry.name
            )
            .into());
        }
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    if baseline.contains_key("prefill_quantize_registers") {
        require_registers(&baseline, "prefill_quantize_registers", &quantize_registers)?;
        require_uniform_value(&baseline, "prefill_quantize_shared_bytes", &quantize_shared)?;
    }

    let mut w4a4_registers = Vec::new();
    let mut w4a4_shared = Vec::new();
    for entry in w4a4 {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 QKV W4A4 entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 QKV W4A4 SASS `{}`",
                entry.name
            )
        })?;
        if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
            return Err(format!(
                "entry `{}` lost native Blackwell NVFP4 MMA selection",
                entry.name
            )
            .into());
        }
        w4a4_registers.push(resource.registers);
        w4a4_shared.push(resource.shared);
    }
    w4a4_registers.sort_unstable();
    if baseline.contains_key("prefill_w4a4_registers") {
        require_registers(&baseline, "prefill_w4a4_registers", &w4a4_registers)?;
        require_uniform_value(&baseline, "prefill_w4a4_shared_bytes", &w4a4_shared)?;
    }

    println!(
        "Qwen3.5 NVFP4 QKV gate passed: 8 A16 + 4 prefill quantize + 4 W4A4 entries, REG {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}",
        registers, quantize_registers, w4a4_registers, shared, quantize_shared, w4a4_shared
    );
    Ok(())
}

fn gate_qwen35_bf16_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_BF16_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_bf16_lm_head_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.5 BF16 LM head", routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["fma.rn.f32", "shfl.sync.down.b32"] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::with_capacity(routes.len());
    let mut shared = Vec::with_capacity(routes.len());
    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 BF16 LM-head entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 BF16 LM-head SASS `{}`",
                entry.name
            )
        })?;
        for instruction in ["FFMA", "SHFL.DOWN", "STG.E.U16"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5 BF16 LM-head gate passed: 8 projection entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}, FFMA/SHFL/BF16-store present",
        registers, shared
    );
    Ok(())
}

fn gate_qwen36_moe_router(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_MOE_ROUTER_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_moe_router_logits_TID_"))
        .collect::<Vec<_>>();
    let selection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_moe_router_select_TID_"))
        .collect::<Vec<_>>();
    let prefill_projection = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_moe_router_logits_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_selection = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_moe_router_select_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 MoE router projection", projection.len(), 8)?;
    require_count("Qwen3.6 MoE router selection", selection.len(), 8)?;
    require_count(
        "Qwen3.6 MoE prompt router projection",
        prefill_projection.len(),
        3,
    )?;
    require_count(
        "Qwen3.6 MoE prompt router selection",
        prefill_selection.len(),
        3,
    )?;

    for entry in projection.iter().chain(&prefill_projection) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["fma.rn.f32", "shfl.sync.down.b32"] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }
    for entry in selection.iter().chain(&prefill_selection) {
        if !entry.body.contains(".reqntid 32, 1, 1") || !entry.body.contains(".minnctapersm 1") {
            return Err(format!("entry `{}` lost its one-warp launch bounds", entry.name).into());
        }
        if !entry.body.contains("ex2.approx.f32") {
            return Err(format!("entry `{}` lost top-eight exponential PTX", entry.name).into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut projection_registers = Vec::with_capacity(projection.len());
    let mut prefill_projection_registers = Vec::with_capacity(prefill_projection.len());
    let mut selection_registers = Vec::with_capacity(selection.len());
    let mut prefill_selection_registers = Vec::with_capacity(prefill_selection.len());
    let mut shared = Vec::with_capacity(
        projection.len() + prefill_projection.len() + selection.len() + prefill_selection.len(),
    );
    for (role, routes, instructions, registers) in [
        (
            "projection",
            projection,
            &["FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut projection_registers,
        ),
        (
            "selection",
            selection,
            &["MUFU.EX2", "LDG.E.U16", "STG.E.U16"][..],
            &mut selection_registers,
        ),
        (
            "prefill projection",
            prefill_projection,
            &["FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut prefill_projection_registers,
        ),
        (
            "prefill selection",
            prefill_selection,
            &["MUFU.EX2", "LDG.E.U16", "STG.E.U16"][..],
            &mut prefill_selection_registers,
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 MoE router {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 MoE router {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    projection_registers.sort_unstable();
    prefill_projection_registers.sort_unstable();
    selection_registers.sort_unstable();
    prefill_selection_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_registers(&baseline, "selection_registers", &selection_registers)?;
    for (key, registers) in [
        (
            "prefill_projection_registers",
            prefill_projection_registers.as_slice(),
        ),
        (
            "prefill_selection_registers",
            prefill_selection_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 MoE router gate passed: 8 decode + 3 prefill routes, REG projection {:?} / {:?}, selection {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, FFMA/SHFL/EX2/BF16-store present",
        projection_registers,
        prefill_projection_registers,
        selection_registers,
        prefill_selection_registers,
        shared
    );
    Ok(())
}

fn gate_qwen36_moe_experts(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_MOE_EXPERTS_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let gate_up = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_moe_expert_gate_up_TID_"))
        .collect::<Vec<_>>();
    let down = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_moe_expert_down_TID_"))
        .collect::<Vec<_>>();
    let combine = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_moe_expert_combine_TID_"))
        .collect::<Vec<_>>();
    let prefill_gate_up = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_moe_expert_gate_up_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_down = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_moe_expert_down_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_combine = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_moe_expert_combine_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 MoE expert gate/up", gate_up.len(), 8)?;
    require_count("Qwen3.6 MoE expert down", down.len(), 8)?;
    require_count("Qwen3.6 MoE expert combine", combine.len(), 8)?;
    require_count(
        "Qwen3.6 MoE prompt expert gate/up",
        prefill_gate_up.len(),
        3,
    )?;
    require_count("Qwen3.6 MoE prompt expert down", prefill_down.len(), 3)?;
    require_count(
        "Qwen3.6 MoE prompt expert combine",
        prefill_combine.len(),
        3,
    )?;

    for (role, routes, instructions) in [
        (
            "gate/up",
            gate_up
                .iter()
                .chain(&prefill_gate_up)
                .copied()
                .collect::<Vec<_>>(),
            &[
                "cvt.rn.f16x2.e2m1x2",
                "shfl.sync.down.b32",
                "ex2.approx.f32",
            ][..],
        ),
        (
            "down",
            down.iter()
                .chain(&prefill_down)
                .copied()
                .collect::<Vec<_>>(),
            &["cvt.rn.f16x2.e2m1x2", "shfl.sync.down.b32"][..],
        ),
        (
            "combine",
            combine
                .iter()
                .chain(&prefill_combine)
                .copied()
                .collect::<Vec<_>>(),
            &["fma.rn.f32", "ex2.approx.f32"][..],
        ),
    ] {
        for entry in &routes {
            if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2")
            {
                return Err(format!(
                    "entry `{}` lost its 256-thread/two-CTA launch bounds",
                    entry.name
                )
                .into());
            }
            for instruction in instructions {
                if !entry.body.contains(instruction) {
                    return Err(format!(
                        "Qwen3.6 MoE expert {role} entry `{}` lost `{instruction}` PTX",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut gate_up_registers = Vec::with_capacity(gate_up.len());
    let mut prefill_gate_up_registers = Vec::with_capacity(prefill_gate_up.len());
    let mut down_registers = Vec::with_capacity(down.len());
    let mut prefill_down_registers = Vec::with_capacity(prefill_down.len());
    let mut combine_registers = Vec::with_capacity(combine.len());
    let mut prefill_combine_registers = Vec::with_capacity(prefill_combine.len());
    let mut gate_up_shared = Vec::with_capacity(gate_up.len() + prefill_gate_up.len());
    let mut prefill_gate_up_shared = Vec::with_capacity(prefill_gate_up.len());
    let mut down_shared = Vec::with_capacity(down.len() + prefill_down.len());
    let mut prefill_down_shared = Vec::with_capacity(prefill_down.len());
    let mut combine_shared = Vec::with_capacity(combine.len() + prefill_combine.len());
    let mut prefill_combine_shared = Vec::with_capacity(prefill_combine.len());

    for (role, routes, instructions, registers, shared) in [
        (
            "gate/up",
            gate_up,
            &["F2FP.F16.E2M1", "SHFL.DOWN", "MUFU.EX2", "STG.E.U16"][..],
            &mut gate_up_registers,
            &mut gate_up_shared,
        ),
        (
            "prompt gate/up",
            prefill_gate_up,
            &["F2FP.F16.E2M1", "SHFL.DOWN", "MUFU.EX2", "STG.E.U16"][..],
            &mut prefill_gate_up_registers,
            &mut prefill_gate_up_shared,
        ),
        (
            "down",
            down,
            &["F2FP.F16.E2M1", "SHFL.DOWN", "STG.E.U16"][..],
            &mut down_registers,
            &mut down_shared,
        ),
        (
            "prompt down",
            prefill_down,
            &["F2FP.F16.E2M1", "SHFL.DOWN", "STG.E.U16"][..],
            &mut prefill_down_registers,
            &mut prefill_down_shared,
        ),
        (
            "combine",
            combine,
            &["MUFU.EX2", "FFMA", "STG.E.U16"][..],
            &mut combine_registers,
            &mut combine_shared,
        ),
        (
            "prompt combine",
            prefill_combine,
            &["MUFU.EX2", "FFMA", "STG.E.U16"][..],
            &mut prefill_combine_registers,
            &mut prefill_combine_shared,
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 MoE expert {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 MoE expert {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    gate_up_registers.sort_unstable();
    prefill_gate_up_registers.sort_unstable();
    down_registers.sort_unstable();
    prefill_down_registers.sort_unstable();
    combine_registers.sort_unstable();
    prefill_combine_registers.sort_unstable();
    gate_up_shared.extend(prefill_gate_up_shared);
    down_shared.extend(prefill_down_shared);
    combine_shared.extend(prefill_combine_shared);
    gate_up_shared.sort_unstable();
    down_shared.sort_unstable();
    combine_shared.sort_unstable();
    require_registers(&baseline, "gate_up_registers", &gate_up_registers)?;
    require_registers(&baseline, "down_registers", &down_registers)?;
    require_registers(&baseline, "combine_registers", &combine_registers)?;
    for (key, registers) in [
        (
            "prefill_gate_up_registers",
            prefill_gate_up_registers.as_slice(),
        ),
        ("prefill_down_registers", prefill_down_registers.as_slice()),
        (
            "prefill_combine_registers",
            prefill_combine_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "gate_up_shared_bytes", &gate_up_shared)?;
    require_uniform_value(&baseline, "down_shared_bytes", &down_shared)?;
    require_uniform_value(&baseline, "combine_shared_bytes", &combine_shared)?;

    println!(
        "Qwen3.6 MoE expert gate passed: 8 decode + 3 prefill routes, REG gate/up {:?} / {:?}, down {:?} / {:?}, combine {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}, E2M1/SHFL/EX2/BF16-store present",
        gate_up_registers,
        prefill_gate_up_registers,
        down_registers,
        prefill_down_registers,
        combine_registers,
        prefill_combine_registers,
        gate_up_shared,
        down_shared,
        combine_shared
    );
    Ok(())
}

fn gate_qwen36_nvfp4_lm_head(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_NVFP4_LM_HEAD_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_nvfp4_lm_head_a16_TID_"))
        .collect::<Vec<_>>();
    require_count("Qwen3.6 NVFP4 LM head", routes.len(), 8)?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["cvt.rn.f16x2.e2m1x2", "shfl.sync.down.b32", "st.global.b16"] {
            if !entry.body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` PTX", entry.name).into(),
                );
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::with_capacity(routes.len());
    let mut shared = Vec::with_capacity(routes.len());
    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.6 NVFP4 LM-head entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        let body = sass_function_body(sass, entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.6 NVFP4 LM-head SASS `{}`",
                entry.name
            )
        })?;
        for instruction in ["F2FP.F16.E2M1", "SHFL.DOWN", "STG.E.U16"] {
            if !body.contains(instruction) {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "lm_head_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 NVFP4 LM-head gate passed: 8 A16 entries, REG {:?}, STACK:0 LOCAL:0, SHARED {:?}, E2M1/SHFL/BF16-store present",
        registers, shared
    );
    Ok(())
}

fn gate_qwen36_fp8_qkv(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_FP8_QKV_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_attention_fp8_quantize_TID_"))
        .collect::<Vec<_>>();
    let prefill_quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_attention_fp8_quantize_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_fp8_qkv_TID_"))
        .collect::<Vec<_>>();
    let prefill_projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_fp8_qkv_prefill_TID_"))
        .collect::<Vec<_>>();
    require_count(
        "Qwen3.6 attention static FP8 quantization",
        quantize.len(),
        8,
    )?;
    require_count(
        "Qwen3.6 attention prefill static FP8 quantization",
        prefill_quantize.len(),
        3,
    )?;
    require_count("Qwen3.6 FP8 QKV projection", projection.len(), 8)?;
    require_count(
        "Qwen3.6 FP8 QKV prefill projection",
        prefill_projection.len(),
        3,
    )?;

    for (role, routes, instructions) in [
        (
            "static quantization",
            quantize.as_slice(),
            &["div.rn.f32", "cvt.rn.satfinite.e4m3x2.f32"][..],
        ),
        (
            "QKV projection",
            projection.as_slice(),
            &[
                "cvt.rn.f16x2.e4m3x2",
                "fma.rn.f32",
                "shfl.sync.down.b32",
                "cvt.rn.bf16x2.f32",
            ][..],
        ),
        (
            "prefill static quantization",
            prefill_quantize.as_slice(),
            &["div.rn.f32", "cvt.rn.satfinite.e4m3x2.f32"][..],
        ),
        (
            "QKV prefill projection",
            prefill_projection.as_slice(),
            &[][..],
        ),
    ] {
        for entry in routes {
            if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2")
            {
                return Err(format!(
                    "entry `{}` lost its 256-thread/two-CTA launch bounds",
                    entry.name
                )
                .into());
            }
            for instruction in instructions {
                if !entry.body.contains(instruction) {
                    return Err(format!(
                        "Qwen3.6 attention {role} entry `{}` lost `{instruction}` PTX",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut quantize_registers = Vec::with_capacity(quantize.len());
    let mut prefill_quantize_registers = Vec::with_capacity(prefill_quantize.len());
    let mut projection_registers = Vec::with_capacity(projection.len());
    let mut prefill_projection_registers = Vec::with_capacity(prefill_projection.len());
    let mut shared = Vec::with_capacity(
        quantize.len() + prefill_quantize.len() + projection.len() + prefill_projection.len(),
    );
    for (role, routes, instructions, registers) in [
        (
            "static quantization",
            quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut quantize_registers,
        ),
        (
            "QKV projection",
            projection,
            &["F2FP.F16.E4M3", "FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut projection_registers,
        ),
        (
            "prefill static quantization",
            prefill_quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut prefill_quantize_registers,
        ),
        (
            "QKV prefill projection",
            prefill_projection,
            &["QMMA.16832.F32.E4M3.E4M3", "LDGSTS", "STG.E.U16"][..],
            &mut prefill_projection_registers,
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 attention {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 attention {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    quantize_registers.sort_unstable();
    prefill_quantize_registers.sort_unstable();
    projection_registers.sort_unstable();
    prefill_projection_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    for (key, registers) in [
        (
            "prefill_quantize_registers",
            prefill_quantize_registers.as_slice(),
        ),
        (
            "prefill_projection_registers",
            prefill_projection_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 FP8 QKV gate passed: 8 decode + 3 prefill routes, REG quantize {:?} / {:?}, projection {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/QMMA/FFMA/SHFL/BF16-store present",
        quantize_registers,
        prefill_quantize_registers,
        projection_registers,
        prefill_projection_registers,
        shared
    );
    Ok(())
}

fn gate_qwen36_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_GDN_INPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_static_fp8_quantize_TID_"))
        .collect::<Vec<_>>();
    let prefill_quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_static_fp8_quantize_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_fp8_gdn_input_TID_"))
        .collect::<Vec<_>>();
    let prefill_projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_fp8_gdn_input_prefill_TID_"))
        .collect::<Vec<_>>();
    let controls = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_bf16_gdn_control_TID_"))
        .collect::<Vec<_>>();
    let prefill_controls = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_bf16_gdn_control_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 static FP8 quantization", quantize.len(), 8)?;
    require_count(
        "Qwen3.6 prefill static FP8 quantization",
        prefill_quantize.len(),
        3,
    )?;
    require_count("Qwen3.6 FP8 GDN input projection", projection.len(), 8)?;
    require_count(
        "Qwen3.6 FP8 GDN input prefill projection",
        prefill_projection.len(),
        3,
    )?;
    require_count("Qwen3.6 BF16 GDN controls", controls.len(), 8)?;
    require_count(
        "Qwen3.6 BF16 GDN prefill controls",
        prefill_controls.len(),
        3,
    )?;

    for (role, routes, instructions) in [
        (
            "static quantization",
            quantize.as_slice(),
            &["div.rn.f32", "cvt.rn.satfinite.e4m3x2.f32"][..],
        ),
        (
            "FP8 projection",
            projection.as_slice(),
            &[
                "cvt.rn.f16x2.e4m3x2",
                "fma.rn.f32",
                "shfl.sync.down.b32",
                "cvt.rn.bf16x2.f32",
            ][..],
        ),
        (
            "BF16 controls",
            controls.as_slice(),
            &["fma.rn.f32", "shfl.sync.down.b32", "cvt.rn.bf16x2.f32"][..],
        ),
        (
            "prefill static quantization",
            prefill_quantize.as_slice(),
            &["div.rn.f32", "cvt.rn.satfinite.e4m3x2.f32"][..],
        ),
        (
            "FP8 prefill projection",
            prefill_projection.as_slice(),
            &[][..],
        ),
        (
            "BF16 prefill controls",
            prefill_controls.as_slice(),
            &["fma.rn.f32", "shfl.sync.down.b32", "cvt.rn.bf16x2.f32"][..],
        ),
    ] {
        for entry in routes {
            if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2")
            {
                return Err(format!(
                    "entry `{}` lost its 256-thread/two-CTA launch bounds",
                    entry.name
                )
                .into());
            }
            for instruction in instructions {
                if !entry.body.contains(instruction) {
                    return Err(format!(
                        "Qwen3.6 GDN input {role} entry `{}` lost `{instruction}` PTX",
                        entry.name
                    )
                    .into());
                }
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut quantize_registers = Vec::with_capacity(quantize.len());
    let mut prefill_quantize_registers = Vec::with_capacity(prefill_quantize.len());
    let mut projection_registers = Vec::with_capacity(projection.len());
    let mut prefill_projection_registers = Vec::with_capacity(prefill_projection.len());
    let mut control_registers = Vec::with_capacity(controls.len());
    let mut prefill_control_registers = Vec::with_capacity(prefill_controls.len());
    let mut shared = Vec::with_capacity(
        quantize.len()
            + prefill_quantize.len()
            + projection.len()
            + prefill_projection.len()
            + controls.len()
            + prefill_controls.len(),
    );
    for (role, routes, instructions, registers) in [
        (
            "static quantization",
            quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut quantize_registers,
        ),
        (
            "FP8 projection",
            projection,
            &["F2FP.F16.E4M3", "FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut projection_registers,
        ),
        (
            "BF16 controls",
            controls,
            &["FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut control_registers,
        ),
        (
            "prefill static quantization",
            prefill_quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut prefill_quantize_registers,
        ),
        (
            "FP8 prefill projection",
            prefill_projection,
            &["QMMA.16832.F32.E4M3.E4M3", "LDGSTS", "STG.E.U16"][..],
            &mut prefill_projection_registers,
        ),
        (
            "BF16 prefill controls",
            prefill_controls,
            &["FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut prefill_control_registers,
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 GDN input {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 GDN input {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            if role.starts_with("BF16") && body.contains("E4M3") {
                return Err(format!(
                    "Qwen3.6 BF16 control entry `{}` unexpectedly contains E4M3 conversion",
                    entry.name
                )
                .into());
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    quantize_registers.sort_unstable();
    prefill_quantize_registers.sort_unstable();
    projection_registers.sort_unstable();
    prefill_projection_registers.sort_unstable();
    control_registers.sort_unstable();
    prefill_control_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_registers(&baseline, "control_registers", &control_registers)?;
    for (key, registers) in [
        (
            "prefill_quantize_registers",
            prefill_quantize_registers.as_slice(),
        ),
        (
            "prefill_projection_registers",
            prefill_projection_registers.as_slice(),
        ),
        (
            "prefill_control_registers",
            prefill_control_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 GDN input gate passed: 8 decode + 3 prefill routes, REG quantize {:?} / {:?}, projection {:?} / {:?}, control {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/QMMA/FFMA/SHFL/BF16-store present",
        quantize_registers,
        prefill_quantize_registers,
        projection_registers,
        prefill_projection_registers,
        control_registers,
        prefill_control_registers,
        shared
    );
    Ok(())
}

fn gate_qwen36_gdn_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_GDN_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_gdn_output_static_quantize_TID_")
        })
        .collect::<Vec<_>>();
    let projection = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen36_gdn_output_projection_TID_"))
        .collect::<Vec<_>>();
    let prefill_quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_gdn_output_static_quantize_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_projection = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_gdn_output_projection_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 GDN output static quantization", quantize.len(), 8)?;
    require_count("Qwen3.6 GDN output projection", projection.len(), 8)?;
    require_count(
        "Qwen3.6 GDN output prefill static quantization",
        prefill_quantize.len(),
        3,
    )?;
    require_count(
        "Qwen3.6 GDN output prefill projection",
        prefill_projection.len(),
        3,
    )?;

    for entry in quantize.iter().chain(prefill_quantize.iter()) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["div.rn.f32", "cvt.rn.satfinite.e4m3x2.f32"] {
            if !entry.body.contains(instruction) {
                return Err(format!(
                    "Qwen3.6 GDN output quantization `{}` lost `{instruction}` PTX",
                    entry.name
                )
                .into());
            }
        }
    }
    for entry in &projection {
        if !entry.body.contains(".reqntid 128, 1, 1") || !entry.body.contains(".minnctapersm 4") {
            return Err(format!(
                "entry `{}` lost its 128-thread/four-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in [
            "cvt.rn.f16x2.e4m3x2",
            "fma.rn.f32",
            "shfl.sync.down.b32",
            "cvt.rn.bf16x2.f32",
        ] {
            if !entry.body.contains(instruction) {
                return Err(format!(
                    "Qwen3.6 GDN output projection `{}` lost `{instruction}` PTX",
                    entry.name
                )
                .into());
            }
        }
    }
    for entry in &prefill_projection {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut quantize_registers = Vec::with_capacity(quantize.len());
    let mut prefill_quantize_registers = Vec::with_capacity(prefill_quantize.len());
    let mut projection_registers = Vec::with_capacity(projection.len());
    let mut prefill_projection_registers = Vec::with_capacity(prefill_projection.len());
    let mut shared = Vec::with_capacity(
        quantize.len() + prefill_quantize.len() + projection.len() + prefill_projection.len(),
    );
    for (role, routes, instructions, registers) in [
        (
            "static quantization",
            quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut quantize_registers,
        ),
        (
            "projection",
            projection,
            &["F2FP.F16.E4M3", "FFMA", "SHFL.DOWN", "STG.E.U16"][..],
            &mut projection_registers,
        ),
        (
            "prefill static quantization",
            prefill_quantize,
            &["F2FP.SATFINITE.E4M3", "STG.E.U16"][..],
            &mut prefill_quantize_registers,
        ),
        (
            "prefill projection",
            prefill_projection,
            &["QMMA.16832.F32.E4M3.E4M3", "LDGSTS", "STG.E.U16"][..],
            &mut prefill_projection_registers,
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 GDN output {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 GDN output {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in instructions {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    quantize_registers.sort_unstable();
    prefill_quantize_registers.sort_unstable();
    projection_registers.sort_unstable();
    prefill_projection_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "quantize_registers", &quantize_registers)?;
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    for (key, registers) in [
        (
            "prefill_quantize_registers",
            prefill_quantize_registers.as_slice(),
        ),
        (
            "prefill_projection_registers",
            prefill_projection_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 GDN output gate passed: 8 decode + 3 prefill routes, REG quantize {:?} / {:?}, projection {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, E4M3/QMMA/FFMA/SHFL/BF16-store present",
        quantize_registers,
        prefill_quantize_registers,
        projection_registers,
        prefill_projection_registers,
        shared
    );
    Ok(())
}

fn gate_qwen36_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN36_ATTENTION_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let gates = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_attention_output_gate_bf16_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_gates = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen36_attention_output_gate_bf16_prefill_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.6 attention-output gate", gates.len(), 8)?;
    require_count(
        "Qwen3.6 attention-output prefill gate",
        prefill_gates.len(),
        3,
    )?;

    for entry in gates.iter().chain(prefill_gates.iter()) {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        for instruction in ["ex2.approx.f32", "st.global.b16"] {
            if !entry.body.contains(instruction) {
                return Err(format!(
                    "Qwen3.6 attention-output gate `{}` lost `{instruction}` PTX",
                    entry.name
                )
                .into());
            }
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::with_capacity(gates.len());
    let mut prefill_registers = Vec::with_capacity(prefill_gates.len());
    let mut shared = Vec::with_capacity(gates.len() + prefill_gates.len());
    for (role, routes, role_registers) in [
        ("decode", gates, &mut registers),
        ("prefill", prefill_gates, &mut prefill_registers),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 attention-output {role} gate `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.6 attention-output {role} gate SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in ["MUFU.EX2", "STG.E.U16"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            role_registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    registers.sort_unstable();
    prefill_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "gate_registers", &registers)?;
    if baseline.contains_key("prefill_gate_registers") {
        require_registers(&baseline, "prefill_gate_registers", &prefill_registers)?;
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.6 attention-output gate passed: 8 decode + 3 prefill entries, REG {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}, EX2/BF16-store present",
        registers, prefill_registers, shared
    );
    Ok(())
}

fn gate_qwen35_nvfp4_gdn_input(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_GDN_INPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_nvfp4_gdn_input_a16_TID_"))
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_gdn_input_quantize_TID_")
        })
        .collect::<Vec<_>>();
    let projected_w4a4 = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_gdn_input_projected_w4a4_TID_")
        })
        .collect::<Vec<_>>();
    let control_w4a4 = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_gdn_input_control_w4a4_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 GDN input", routes.len(), 8)?;
    require_count(
        "Qwen3.5 NVFP4 GDN input prefill quantization",
        quantize.len(),
        3,
    )?;
    require_count(
        "Qwen3.5 NVFP4 projected GDN input prefill",
        projected_w4a4.len(),
        3,
    )?;
    require_count(
        "Qwen3.5 NVFP4 control GDN input prefill",
        control_w4a4.len(),
        3,
    )?;

    for entry in &routes {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }
    for entry in &quantize {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in projected_w4a4.iter().chain(&control_w4a4) {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut shared = Vec::new();
    for entry in routes {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 GDN input entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.5 NVFP4 GDN input SASS `{}`",
                entry.name
            )
            .into());
        }
        registers.push(resource.registers);
        shared.push(resource.shared);
    }
    registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "nvfp4_registers", &registers)?;
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    let mut quantize_registers = Vec::new();
    let mut quantize_shared = Vec::new();
    for entry in quantize {
        let resource = resources.get(entry.name).ok_or_else(|| {
            format!(
                "cuobjdump omitted Qwen3.5 NVFP4 GDN input quantization entry `{}`",
                entry.name
            )
        })?;
        require_spill_free(entry.name, resource)?;
        if sass_function_body(sass, entry.name).is_none() {
            return Err(format!(
                "cuobjdump omitted Qwen3.5 NVFP4 GDN input quantization SASS `{}`",
                entry.name
            )
            .into());
        }
        quantize_registers.push(resource.registers);
        quantize_shared.push(resource.shared);
    }
    quantize_registers.sort_unstable();
    if baseline.contains_key("prefill_quantize_registers") {
        require_registers(&baseline, "prefill_quantize_registers", &quantize_registers)?;
        require_uniform_value(&baseline, "prefill_quantize_shared_bytes", &quantize_shared)?;
    }

    let mut projected_registers = Vec::new();
    let mut projected_shared = Vec::new();
    let mut control_registers = Vec::new();
    let mut control_shared = Vec::new();
    for (role, entries, registers, shared_bytes) in [
        (
            "projected",
            projected_w4a4,
            &mut projected_registers,
            &mut projected_shared,
        ),
        (
            "control",
            control_w4a4,
            &mut control_registers,
            &mut control_shared,
        ),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 {role} GDN input entry `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 {role} GDN input SASS `{}`",
                    entry.name
                )
            })?;
            if !body.contains("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") {
                return Err(format!(
                    "entry `{}` lost native Blackwell NVFP4 MMA selection",
                    entry.name
                )
                .into());
            }
            registers.push(resource.registers);
            shared_bytes.push(resource.shared);
        }
    }
    projected_registers.sort_unstable();
    control_registers.sort_unstable();
    for (key, registers) in [
        (
            "prefill_projected_registers",
            projected_registers.as_slice(),
        ),
        ("prefill_control_registers", control_registers.as_slice()),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    for (key, shared) in [
        (
            "prefill_projected_shared_bytes",
            projected_shared.as_slice(),
        ),
        ("prefill_control_shared_bytes", control_shared.as_slice()),
    ] {
        if baseline.contains_key(key) {
            require_uniform_value(&baseline, key, shared)?;
        }
    }

    println!(
        "Qwen3.5 NVFP4 GDN input gate passed: 8 A16 + 3 prefill quantize + 3 projected W4A4 + 3 control W4A4 entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?} / {:?}",
        registers,
        quantize_registers,
        projected_registers,
        control_registers,
        shared,
        quantize_shared,
        projected_shared,
        control_shared
    );
    Ok(())
}

fn gate_qwen35_gdn_prepare(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_GDN_PREPARE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_gdn_prepare_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_prepare_prefill_exact_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_history = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_prepare_prefill_history_exact_TID_")
        })
        .collect::<Vec<_>>();
    let causal = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_prepare_causal_exact_TID_")
        })
        .collect::<Vec<_>>();
    let causal_history = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_prepare_causal_history_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.5 GDN prepare", routes.len(), 8)?;
    require_count("Qwen3.5 GDN prepare causal", causal.len(), 3)?;
    require_count(
        "Qwen3.5 GDN prepare causal history",
        causal_history.len(),
        3,
    )?;
    require_count("Qwen3.5/Qwen3.6 GDN prepare prefill", prefill.len(), 3)?;
    require_count(
        "Qwen3.5/Qwen3.6 GDN prepare prefill history",
        prefill_history.len(),
        3,
    )?;

    for entry in routes
        .iter()
        .chain(&causal)
        .chain(&causal_history)
        .chain(&prefill)
        .chain(&prefill_history)
    {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry.name.contains("history")
            && (!entry.body.contains("ex2.approx.f32") || !entry.body.contains("lg2.approx.f32"))
        {
            return Err(format!(
                "entry `{}` lost the Qwen3.5 GDN control transforms",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut causal_registers = Vec::new();
    let mut causal_history_registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut prefill_history_registers = Vec::new();
    let mut shared = Vec::new();
    for (role, entries, role_registers) in [
        ("decode", routes, &mut registers),
        ("causal", causal, &mut causal_registers),
        (
            "causal history",
            causal_history,
            &mut causal_history_registers,
        ),
        ("prefill", prefill, &mut prefill_registers),
        (
            "prefill history",
            prefill_history,
            &mut prefill_history_registers,
        ),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 GDN prepare {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 GDN prepare {role} SASS `{}`",
                    entry.name
                )
            })?;
            if !body.contains("STG.E") {
                return Err(format!("entry `{}` lost its represented stores", entry.name).into());
            }
            role_registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    registers.sort_unstable();
    causal_registers.sort_unstable();
    causal_history_registers.sort_unstable();
    prefill_registers.sort_unstable();
    prefill_history_registers.sort_unstable();
    shared.sort_unstable();
    require_registers(&baseline, "prepare_registers", &registers)?;
    for (key, registers) in [
        ("causal_registers", causal_registers.as_slice()),
        (
            "causal_history_registers",
            causal_history_registers.as_slice(),
        ),
        ("prefill_registers", prefill_registers.as_slice()),
        (
            "prefill_history_registers",
            prefill_history_registers.as_slice(),
        ),
    ] {
        if baseline.contains_key(key) {
            require_registers(&baseline, key, registers)?;
        }
    }
    require_uniform_value(&baseline, "shared_bytes", &shared)?;

    println!(
        "Qwen3.5/Qwen3.6 GDN prepare gate passed: 8 decode + 3 causal/3 history + 3 prefill/3 history entries, REG {:?} / {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?}",
        registers,
        causal_registers,
        causal_history_registers,
        prefill_registers,
        prefill_history_registers,
        shared
    );
    Ok(())
}

fn gate_qwen35_gdn_recurrence(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_GDN_RECURRENCE_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let routes = entries
        .iter()
        .filter(|entry| entry.name.starts_with("qwen35_gdn_recurrence_exact_TID_"))
        .collect::<Vec<_>>();
    let prefill = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_recurrence_prefill_exact_TID_")
        })
        .collect::<Vec<_>>();
    let epilogue = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_recurrence_prefill_epilogue_exact_TID_")
        })
        .collect::<Vec<_>>();
    let causal = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_gdn_recurrence_causal_exact_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.5 GDN recurrence", routes.len(), 8)?;
    require_count("Qwen3.5 GDN recurrence causal", causal.len(), 3)?;
    require_count("Qwen3.5/Qwen3.6 GDN recurrence prefill", prefill.len(), 3)?;
    require_count(
        "Qwen3.5/Qwen3.6 GDN recurrence prefill epilogue",
        epilogue.len(),
        3,
    )?;

    for entry in routes.iter().chain(&causal) {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &prefill {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 1") {
            return Err(format!(
                "entry `{}` lost its 512-thread/shared-resident-state launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in &epilogue {
        if !entry.body.contains(".reqntid 512, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 512-thread/two-CTA epilogue launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in routes
        .iter()
        .chain(&causal)
        .chain(&prefill)
        .chain(&epilogue)
    {
        if !entry.body.contains("rsqrt.approx.f32") || !entry.body.contains("ex2.approx.f32") {
            return Err(format!(
                "entry `{}` lost normalization or recurrent decay",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut registers = Vec::new();
    let mut causal_registers = Vec::new();
    let mut prefill_registers = Vec::new();
    let mut shared = Vec::new();
    let mut causal_shared = Vec::new();
    let mut prefill_shared = Vec::new();
    let mut epilogue_registers = Vec::new();
    let mut epilogue_shared = Vec::new();
    for (role, entries, role_registers, role_shared) in [
        ("decode", routes, &mut registers, &mut shared),
        ("causal", causal, &mut causal_registers, &mut causal_shared),
        (
            "prefill",
            prefill,
            &mut prefill_registers,
            &mut prefill_shared,
        ),
        (
            "prefill epilogue",
            epilogue,
            &mut epilogue_registers,
            &mut epilogue_shared,
        ),
    ] {
        for entry in entries {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 GDN recurrence {role} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 GDN recurrence {role} SASS `{}`",
                    entry.name
                )
            })?;
            for instruction in ["MUFU.RSQ", "MUFU.EX2"] {
                if !body.contains(instruction) {
                    return Err(format!(
                        "entry `{}` lost required `{instruction}` SASS",
                        entry.name
                    )
                    .into());
                }
            }
            role_registers.push(resource.registers);
            role_shared.push(resource.shared);
        }
    }
    registers.sort_unstable();
    causal_registers.sort_unstable();
    prefill_registers.sort_unstable();
    shared.sort_unstable();
    causal_shared.sort_unstable();
    prefill_shared.sort_unstable();
    epilogue_registers.sort_unstable();
    epilogue_shared.sort_unstable();
    require_registers(&baseline, "recurrence_registers", &registers)?;
    if baseline.contains_key("causal_registers") {
        require_registers(&baseline, "causal_registers", &causal_registers)?;
    }
    if baseline.contains_key("prefill_registers") {
        require_registers(&baseline, "prefill_registers", &prefill_registers)?;
    }
    let mut decode_and_causal_shared = shared.clone();
    decode_and_causal_shared.extend_from_slice(&causal_shared);
    decode_and_causal_shared.sort_unstable();
    require_uniform_value(&baseline, "shared_bytes", &decode_and_causal_shared)?;
    if baseline.contains_key("prefill_shared_bytes") {
        require_uniform_value(&baseline, "prefill_shared_bytes", &prefill_shared)?;
    }
    if baseline.contains_key("prefill_epilogue_registers") {
        require_registers(&baseline, "prefill_epilogue_registers", &epilogue_registers)?;
    }
    if baseline.contains_key("prefill_epilogue_shared_bytes") {
        require_uniform_value(&baseline, "prefill_epilogue_shared_bytes", &epilogue_shared)?;
    }

    println!(
        "Qwen3.5/Qwen3.6 GDN recurrence gate passed: 8 decode + 3 causal + 3 prefill + 3 prefill epilogue entries, REG {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?}, RSQ/EX2 present",
        registers,
        causal_registers,
        prefill_registers,
        epilogue_registers,
        decode_and_causal_shared,
        prefill_shared,
        epilogue_shared,
    );
    Ok(())
}

fn gate_qwen35_nvfp4_attention_output(root: &Path) -> Result<(), Box<dyn Error>> {
    let baseline = parse_baseline(&fs::read_to_string(
        root.join(QWEN35_NVFP4_ATTENTION_OUTPUT_RESOURCE_BASELINE),
    )?)?;
    verify_generator_stamp(root, &baseline)?;
    let entries = &sm120_gate_module(root)?.entries;
    let gates = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_attention_output_gate_bf16_TID_")
        })
        .collect::<Vec<_>>();
    let projections = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_attention_output_a16_TID_")
        })
        .collect::<Vec<_>>();
    let prefill_gates = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_attention_output_gate_bf16_prefill_TID_")
        })
        .collect::<Vec<_>>();
    let quantize = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_attention_output_quantize_TID_")
        })
        .collect::<Vec<_>>();
    let w4a4 = entries
        .iter()
        .filter(|entry| {
            entry
                .name
                .starts_with("qwen35_nvfp4_attention_output_w4a4_TID_")
        })
        .collect::<Vec<_>>();
    require_count("Qwen3.5 NVFP4 attention-output gate", gates.len(), 8)?;
    require_count(
        "Qwen3.5 NVFP4 attention-output projection",
        projections.len(),
        8,
    )?;
    require_count(
        "Qwen3.5 NVFP4 attention-output prefill gate",
        prefill_gates.len(),
        3,
    )?;
    require_count(
        "Qwen3.5 NVFP4 attention-output quantization",
        quantize.len(),
        3,
    )?;
    require_count("Qwen3.5 NVFP4 attention-output W4A4", w4a4.len(), 3)?;

    for entry in gates
        .iter()
        .chain(&projections)
        .chain(&prefill_gates)
        .chain(&quantize)
    {
        if !entry.body.contains(".reqntid 256, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 256-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
    }
    for entry in gates.iter().chain(&prefill_gates) {
        if !entry.body.contains("ex2.approx.f32") {
            return Err(format!("entry `{}` lost sigmoid EX2", entry.name).into());
        }
    }
    for entry in &projections {
        if !entry.body.contains("cvt.rn.f16x2.e2m1x2") {
            return Err(format!("entry `{}` lost represented E2M1 conversion", entry.name).into());
        }
    }
    for entry in &w4a4 {
        if !entry.body.contains(".reqntid 384, 1, 1") || !entry.body.contains(".minnctapersm 2") {
            return Err(format!(
                "entry `{}` lost its 384-thread/two-CTA launch bounds",
                entry.name
            )
            .into());
        }
        if !entry
            .body
            .contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X")
        {
            return Err(format!(
                "entry `{}` lost its exact NVFP4 inline PTX instruction",
                entry.name
            )
            .into());
        }
    }

    let artifact = sm120_gate_artifact(root)?;
    let resources = &artifact.resources;
    let sass = artifact.sass()?;
    let mut gate_registers = Vec::new();
    let mut projection_registers = Vec::new();
    let mut gate_shared = Vec::new();
    let mut projection_shared = Vec::new();
    for (routes, registers, shared, label) in [
        (&gates, &mut gate_registers, &mut gate_shared, "gate"),
        (
            &projections,
            &mut projection_registers,
            &mut projection_shared,
            "projection",
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 attention-output {label} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            if sass_function_body(sass, entry.name).is_none() {
                return Err(format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 attention-output {label} SASS `{}`",
                    entry.name
                )
                .into());
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
    }
    gate_registers.sort_unstable();
    projection_registers.sort_unstable();
    gate_shared.sort_unstable();
    projection_shared.sort_unstable();
    require_registers(&baseline, "gate_registers", &gate_registers)?;
    require_registers(&baseline, "projection_registers", &projection_registers)?;
    require_uniform_value(&baseline, "gate_shared_bytes", &gate_shared)?;
    require_uniform_value(&baseline, "projection_shared_bytes", &projection_shared)?;

    let mut prefill_gate_registers = Vec::new();
    let mut prefill_quantize_registers = Vec::new();
    let mut prefill_w4a4_registers = Vec::new();
    let mut prefill_gate_shared = Vec::new();
    let mut prefill_quantize_shared = Vec::new();
    let mut prefill_w4a4_shared = Vec::new();
    for (routes, registers, shared, label, required_sass) in [
        (
            &prefill_gates,
            &mut prefill_gate_registers,
            &mut prefill_gate_shared,
            "prefill gate",
            None,
        ),
        (
            &quantize,
            &mut prefill_quantize_registers,
            &mut prefill_quantize_shared,
            "prefill quantization",
            None,
        ),
        (
            &w4a4,
            &mut prefill_w4a4_registers,
            &mut prefill_w4a4_shared,
            "W4A4 prefill projection",
            Some("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"),
        ),
    ] {
        for entry in routes {
            let resource = resources.get(entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 attention-output {label} `{}`",
                    entry.name
                )
            })?;
            require_spill_free(entry.name, resource)?;
            let body = sass_function_body(sass, entry.name).ok_or_else(|| {
                format!(
                    "cuobjdump omitted Qwen3.5 NVFP4 attention-output {label} SASS `{}`",
                    entry.name
                )
            })?;
            if let Some(instruction) = required_sass
                && !body.contains(instruction)
            {
                return Err(
                    format!("entry `{}` lost required `{instruction}` SASS", entry.name).into(),
                );
            }
            registers.push(resource.registers);
            shared.push(resource.shared);
        }
        registers.sort_unstable();
        shared.sort_unstable();
    }
    if baseline.contains_key("prefill_gate_registers") {
        require_registers(&baseline, "prefill_gate_registers", &prefill_gate_registers)?;
        require_uniform_value(&baseline, "prefill_gate_shared_bytes", &prefill_gate_shared)?;
        require_registers(
            &baseline,
            "prefill_quantize_registers",
            &prefill_quantize_registers,
        )?;
        require_uniform_value(
            &baseline,
            "prefill_quantize_shared_bytes",
            &prefill_quantize_shared,
        )?;
        require_registers(&baseline, "prefill_w4a4_registers", &prefill_w4a4_registers)?;
        require_uniform_value(&baseline, "prefill_w4a4_shared_bytes", &prefill_w4a4_shared)?;
    }

    println!(
        "Qwen3.5 NVFP4 attention-output gate passed: 8 gate + 8 A16 + 3 prefill gate + 3 quantize + 3 W4A4 entries, REG {:?} / {:?} / {:?} / {:?} / {:?}, STACK:0 LOCAL:0, SHARED {:?} / {:?} / {:?} / {:?} / {:?}, EX2/E2M1/OMMA present",
        gate_registers,
        projection_registers,
        prefill_gate_registers,
        prefill_quantize_registers,
        prefill_w4a4_registers,
        gate_shared,
        projection_shared,
        prefill_gate_shared,
        prefill_quantize_shared,
        prefill_w4a4_shared,
    );
    Ok(())
}

/// The generator identity every resource baseline is stamped against.
struct GeneratorIdentity {
    cuda_oxide_commit: String,
    rustc_release: String,
    rustc_commit: String,
    toolkit: CudaToolkitIdentity,
    lock: String,
}

/// Collects the generator identity once per process.
///
/// Every gate compares the same identity against its own baseline, so the git,
/// rustc and CUDA Toolkit probes behind it run a single time. The clean-checkout
/// requirement is part of collecting the identity and therefore still holds
/// before any stamp is compared.
fn generator_identity(root: &Path) -> Result<&'static GeneratorIdentity, Box<dyn Error>> {
    static IDENTITY: OnceLock<GeneratorIdentity> = OnceLock::new();
    if let Some(identity) = IDENTITY.get() {
        return Ok(identity);
    }

    let backend = backend_path(root)?;
    let source = cuda_oxide_source(root, &backend)?;
    let commit = command_text("git", &["-C", path_text(&source)?, "rev-parse", "HEAD"])?;
    let changes = command_text(
        "git",
        &[
            "-C",
            path_text(&source)?,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )?;
    if !changes.trim().is_empty() {
        return Err("cuda-oxide source has tracked changes; restore the pinned checkout".into());
    }
    let rustc = require_success(Path::new("rustc"), &[OsStr::new("-vV")])?;
    let (rustc_release, rustc_commit) = parse_rustc_identity(&String::from_utf8(rustc.stdout)?)?;

    let ptxas = cuda_tool("ptxas");
    let cuobjdump = cuda_tool("cuobjdump");
    let ptxas_identity = cuda_toolkit_identity(&ptxas)?;
    let cuobjdump_identity = cuda_toolkit_identity(&cuobjdump)?;
    if ptxas_identity != cuobjdump_identity {
        return Err(format!(
            "CUDA tools come from different Toolkit versions: {} reports release {} / V{}, while {} reports release {} / V{}",
            ptxas.display(),
            ptxas_identity.release,
            ptxas_identity.version,
            cuobjdump.display(),
            cuobjdump_identity.release,
            cuobjdump_identity.version,
        )
        .into());
    }

    let lock = fs::read_to_string(root.join("Cargo.lock"))?;

    Ok(IDENTITY.get_or_init(|| GeneratorIdentity {
        cuda_oxide_commit: commit.trim().to_string(),
        rustc_release,
        rustc_commit,
        toolkit: ptxas_identity,
        lock,
    }))
}

fn verify_generator_stamp(root: &Path, baseline: &Baseline) -> Result<(), Box<dyn Error>> {
    let identity = generator_identity(root)?;
    require_stamp(baseline, "cuda_oxide_commit", &identity.cuda_oxide_commit)?;
    require_stamp(baseline, "rustc_release", &identity.rustc_release)?;
    require_stamp(baseline, "rustc_commit", &identity.rustc_commit)?;
    require_stamp(baseline, "cuda_toolkit_release", &identity.toolkit.release)?;
    require_stamp(baseline, "cuda_toolkit_version", &identity.toolkit.version)?;

    let expected_commit = baseline
        .get("cuda_oxide_commit")
        .ok_or("baseline is missing `cuda_oxide_commit`")?;
    if !identity.lock.contains(&format!("rev={expected_commit}")) {
        return Err("Cargo.lock does not contain the stamped cuda-oxide revision".into());
    }

    Ok(())
}

fn backend_path(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_BACKEND") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = local_backend(root) {
        return Ok(path);
    }
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .ok_or("set CUDA_OXIDE_BACKEND or CARGO_HOME")?;

    Ok(cargo_home
        .join("cuda-oxide")
        .join("librustc_codegen_cuda.so"))
}

fn cuda_oxide_source(root: &Path, backend: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("CUDA_OXIDE_SOURCE") {
        return Ok(PathBuf::from(path));
    }
    for ancestor in backend.ancestors() {
        if ancestor.join(".git").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }
    if let Some(parent) = backend.parent() {
        let cached = parent.join("src");
        if cached.join(".git").exists() {
            return Ok(cached);
        }
    }
    let local = local_cuda_oxide_source(root);
    if local.join(".git").exists() {
        return Ok(local);
    }

    Err("could not locate cuda-oxide source; set CUDA_OXIDE_SOURCE".into())
}

fn local_cuda_oxide_source(root: &Path) -> PathBuf {
    root.join("target/cuda-oxide-source")
}

fn task_cargo_home(root: &Path) -> PathBuf {
    root.join("target/cargo-home")
}

fn encoded_backend_rustflags(root: &Path, source: &Path) -> Result<String, Box<dyn Error>> {
    let sysroot = command_text("rustc", &["--print", "sysroot"])?;
    let cargo_home = task_cargo_home(root);
    let prefixes = [
        (source, "/cuda-oxide"),
        (cargo_home.as_path(), "/cargo-home"),
        (Path::new(sysroot.trim()), "/rust-toolchain"),
        (root, "/tuiskollm"),
    ];
    let flags = prefixes
        .into_iter()
        .map(|(path, replacement)| {
            Ok(format!(
                "--remap-path-prefix={}={replacement}",
                path_text(path)?
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(flags.join("\u{1f}"))
}

fn local_backend(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let rustc = command_text("rustc", &["-vV"])?;
    let host = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or("rustc -vV omitted its host triple")?;
    let path = local_cuda_oxide_source(root)
        .join("crates/rustc-codegen-cuda/target")
        .join(host)
        .join("debug/librustc_codegen_cuda.so");
    if !path.is_file() {
        return Err(format!("cuda-oxide backend does not exist at {}", path.display()).into());
    }

    Ok(path)
}

fn cuda_tool(name: &str) -> PathBuf {
    let home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let candidate = home.join("bin").join(name);
    if candidate.is_file() {
        candidate
    } else {
        PathBuf::from(name)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CudaToolkitIdentity {
    release: String,
    version: String,
}

fn cuda_toolkit_identity(tool: &Path) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let output = require_success(tool, &[OsStr::new("--version")])?;

    parse_cuda_toolkit_identity(&String::from_utf8(output.stdout)?)
}

fn parse_cuda_toolkit_identity(text: &str) -> Result<CudaToolkitIdentity, Box<dyn Error>> {
    let identity = text
        .lines()
        .find_map(|line| line.strip_prefix("Cuda compilation tools, release "))
        .ok_or("CUDA tool omitted its release identity")?;
    let (release, version) = identity
        .split_once(", V")
        .ok_or("CUDA tool emitted an invalid release identity")?;

    Ok(CudaToolkitIdentity {
        release: release.to_string(),
        version: version.to_string(),
    })
}

fn parse_rustc_identity(text: &str) -> Result<(String, String), Box<dyn Error>> {
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name}: ")))
            .map(str::to_string)
            .ok_or_else(|| format!("rustc -vV omitted `{name}`"))
    };

    Ok((field("release")?, field("commit-hash")?))
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let arguments = arguments.iter().map(OsStr::new).collect::<Vec<_>>();
    let output = require_success(Path::new(program), &arguments)?;

    Ok(String::from_utf8(output.stdout)?)
}

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| format!("path `{}` is not UTF-8", path.display()).into())
}

fn require_success(program: &Path, arguments: &[&OsStr]) -> Result<Output, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            program.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Ok(output)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

thread_local! {
    static UNCONSUMED_BASELINE_KEYS: RefCell<BTreeSet<String>> =
        const { RefCell::new(BTreeSet::new()) };
}

struct Baseline {
    fields: BTreeMap<String, String>,
    consumed: RefCell<BTreeSet<String>>,
}

impl Baseline {
    fn contains_key(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    fn get(&self, key: &str) -> Option<&String> {
        self.consumed.borrow_mut().insert(key.to_string());
        self.fields.get(key)
    }
}

impl Drop for Baseline {
    fn drop(&mut self) {
        let consumed = self.consumed.borrow();
        UNCONSUMED_BASELINE_KEYS.with(|unconsumed| {
            unconsumed.borrow_mut().extend(
                self.fields
                    .keys()
                    .filter(|key| !consumed.contains(*key))
                    .cloned(),
            );
        });
    }
}

fn require_consumed_baseline_keys() -> Result<(), Box<dyn Error>> {
    let unconsumed = UNCONSUMED_BASELINE_KEYS.with(|keys| std::mem::take(&mut *keys.borrow_mut()));
    if unconsumed.is_empty() {
        return Ok(());
    }

    Err(format!(
        "resource baseline keys were never consumed by their gate (misspelled key?): {}",
        unconsumed.into_iter().collect::<Vec<_>>().join(", ")
    )
    .into())
}

fn parse_baseline(text: &str) -> Result<Baseline, Box<dyn Error>> {
    require_consumed_baseline_keys()?;
    let mut fields = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid baseline line `{line}`"))?;
        fields.insert(key.to_string(), value.to_string());
    }

    Ok(Baseline {
        fields,
        consumed: RefCell::new(BTreeSet::new()),
    })
}

fn require_stamp(baseline: &Baseline, key: &str, actual: &str) -> Result<(), Box<dyn Error>> {
    let expected = baseline
        .get(key)
        .ok_or_else(|| format!("baseline is missing `{key}`"))?;
    if expected != actual {
        return Err(format!(
            "generator stamp `{key}` is `{actual}`, expected `{expected}`; re-baseline separately"
        )
        .into());
    }

    Ok(())
}

struct Entry<'a> {
    name: &'a str,
    body: &'a str,
}

fn parse_entries(ptx: &str) -> Vec<Entry<'_>> {
    let marker = ".visible .entry ";
    let offsets = ptx.match_indices(marker).collect::<Vec<_>>();
    offsets
        .iter()
        .enumerate()
        .filter_map(|(index, (offset, _))| {
            let begin = offset + marker.len();
            let end = offsets
                .get(index + 1)
                .map(|(offset, _)| *offset)
                .unwrap_or(ptx.len());
            let body = &ptx[begin..end];
            let name_end = body.find('(')?;

            Some(Entry {
                name: body[..name_end].trim(),
                body,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Resource {
    registers: u32,
    stack: u32,
    shared: u32,
    local: u32,
}

/// The SM120 PTX modules and their entry table, read and parsed once per process.
///
/// Every resource gate filters the same entry table out of the same artifact,
/// so the text and its entries are shared the way the compiled artifact already
/// is. Entry names are unique across the split modules, which is what lets one
/// concatenated table serve every family.
struct Sm120GateModule {
    root: PathBuf,
    ptx_sha256: String,
    ptx: &'static str,
    entries: Vec<Entry<'static>>,
}

static SM120_GATE_MODULE: Mutex<Option<&'static Sm120GateModule>> = Mutex::new(None);

fn sm120_gate_module(root: &Path) -> Result<&'static Sm120GateModule, Box<dyn Error>> {
    let mut cached = SM120_GATE_MODULE
        .lock()
        .map_err(|_| "SM120 gate module cache is poisoned")?;
    if let Some(module) = *cached {
        if module.root != root {
            return Err(format!(
                "one xtask process cannot resource-check SM120 artifacts from both `{}` and `{}`",
                module.root.display(),
                root.display()
            )
            .into());
        }
        // perf gate regenerates the PTX after qualification seeds this cache
        if perf_artifact::ptx_modules_exist(root)
            && module.ptx_sha256 == perf_artifact::ptx_modules_sha256(root)?
        {
            return Ok(module);
        }
    }
    let ptx_sha256 = perf_artifact::ptx_modules_sha256(root)?;
    let mut text = String::new();
    for path in SM120_PTX_MODULES.map(|module| root.join(module)) {
        text.push_str(&fs::read_to_string(&path).map_err(|error| {
            format!(
                "could not read {}: {error}; run the pinned release device build first",
                path.display()
            )
        })?);
        text.push('\n');
    }
    // leaked so the entry table and every gate borrow 'static text; rebuilds are rare
    let text: &'static str = Box::leak(text.into_boxed_str());
    let module = &*Box::leak(Box::new(Sm120GateModule {
        root: root.to_path_buf(),
        ptx_sha256,
        ptx: text,
        entries: parse_entries(text),
    }));
    *cached = Some(module);
    Ok(module)
}

struct Sm120GateArtifact {
    module: &'static Sm120GateModule,
    cubins: Vec<PathBuf>,
    resources: BTreeMap<String, Resource>,
    sass: OnceLock<Result<String, String>>,
}

static SM120_GATE_ARTIFACT: Mutex<Option<&'static Sm120GateArtifact>> = Mutex::new(None);

fn sm120_gate_artifact(root: &Path) -> Result<&'static Sm120GateArtifact, Box<dyn Error>> {
    let module = sm120_gate_module(root)?;
    let mut cached = SM120_GATE_ARTIFACT
        .lock()
        .map_err(|_| "SM120 gate artifact cache is poisoned")?;
    // the module cache already rejected a second root and revalidated the PTX
    if let Some(artifact) = *cached
        && std::ptr::eq(artifact.module, module)
    {
        return Ok(artifact);
    }
    // leaked so gate call sites keep borrowing a 'static artifact; rebuilds are rare
    let artifact = &*Box::leak(Box::new(build_sm120_gate_artifact(module)?));
    *cached = Some(artifact);
    Ok(artifact)
}

/// Compiles every emitted module on its own and merges the reported resources.
///
/// Compiling per module is what makes the reported shared-memory footprint
/// truthful: `ptxas` charges a module's largest shared arena to every entry in
/// it, so a family only ever accounts for its own.
fn build_sm120_gate_artifact(
    module: &'static Sm120GateModule,
) -> Result<Sm120GateArtifact, Box<dyn Error>> {
    let temporary = module.root.join("target/tmp");
    fs::create_dir_all(&temporary)?;
    let mut cubins = Vec::with_capacity(SM120_PTX_MODULES.len());
    let mut resources = BTreeMap::new();
    for path in SM120_PTX_MODULES {
        let ptx = module.root.join(path);
        let stem = Path::new(path)
            .file_stem()
            .ok_or_else(|| format!("PTX module path `{path}` has no file stem"))?;
        let cubin = temporary.join(format!("{}-resource-gates.cubin", stem.to_string_lossy()));
        require_success(
            &cuda_tool("ptxas"),
            &[
                OsStr::new("-O3"),
                OsStr::new("--gpu-name"),
                OsStr::new("sm_120a"),
                ptx.as_os_str(),
                OsStr::new("--output-file"),
                cubin.as_os_str(),
            ],
        )?;
        let output = require_success(
            &cuda_tool("cuobjdump"),
            &[OsStr::new("--dump-resource-usage"), cubin.as_os_str()],
        )?;
        for (name, resource) in parse_resources(&String::from_utf8(output.stdout)?)? {
            if let Some(previous) = resources.insert(name.clone(), resource) {
                return Err(format!(
                    "entry `{name}` is emitted by more than one SM120 module ({previous:?})"
                )
                .into());
            }
        }
        cubins.push(cubin);
    }

    Ok(Sm120GateArtifact {
        module,
        cubins,
        resources,
        sass: OnceLock::new(),
    })
}

impl Sm120GateArtifact {
    fn sass(&self) -> Result<&str, Box<dyn Error>> {
        let sass = self.sass.get_or_init(|| {
            let mut merged = String::new();
            for cubin in &self.cubins {
                let output = require_success(
                    &cuda_tool("cuobjdump"),
                    &[OsStr::new("--dump-sass"), cubin.as_os_str()],
                )
                .and_then(|output| String::from_utf8(output.stdout).map_err(Into::into))
                .map_err(|error: Box<dyn Error>| error.to_string())?;
                merged.push_str(&output);
                merged.push('\n');
            }

            Ok(merged)
        });
        match sass {
            Ok(sass) => Ok(sass),
            Err(error) => Err(error.clone().into()),
        }
    }
}

fn parse_resources(text: &str) -> Result<BTreeMap<String, Resource>, Box<dyn Error>> {
    let mut resources = BTreeMap::new();
    let mut function = None;
    for line in text.lines() {
        if let Some(name) = line
            .trim()
            .strip_prefix("Function ")
            .and_then(|name| name.strip_suffix(':'))
        {
            function = Some(name.to_string());
            continue;
        }
        let Some(name) = function.take() else {
            continue;
        };
        let fields = line
            .split_whitespace()
            .filter_map(|field| field.split_once(':'))
            .collect::<BTreeMap<_, _>>();
        let field = |key: &str| -> Result<u32, Box<dyn Error>> {
            Ok(fields
                .get(key)
                .ok_or_else(|| format!("resource line for `{name}` is missing `{key}`"))?
                .parse()?)
        };
        let resource = Resource {
            registers: field("REG")?,
            stack: field("STACK")?,
            shared: field("SHARED")?,
            local: field("LOCAL")?,
        };
        resources.insert(name, resource);
    }

    Ok(resources)
}

fn names_opcode(body: &str, opcode: &str) -> bool {
    body.lines().any(|line| {
        line.split_whitespace()
            .next()
            .is_some_and(|first| first.starts_with(opcode))
    })
}

fn require_count(family: &str, actual: usize, expected: usize) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(format!(
            "{family} emitted {actual} entries, expected {expected}; zero entries is a silent generic-instantiation failure"
        )
        .into());
    }

    Ok(())
}

/// Reports whether `immediate` appears in `body` as a whole-word arithmetic
/// operand rather than as a byte displacement inside a `[...]` memory operand.
///
/// A structural constant such as a head-warp count reaches the artifact as a
/// stride multiplier, never as an addressing offset, and every `qk_prepare`
/// family in this artifact carries `+24` and `+28` displacements regardless of
/// how many head-warps it divides a token into. Matching the raw body would
/// therefore find `28` in the Qwen3.8-Flash-Next entries too and prove nothing; masking
/// memory operands out first is what makes the count discriminating. The
/// word-boundary check keeps register names such as `%rd26` from matching.
fn contains_immediate_operand(body: &str, immediate: &str) -> bool {
    let mut masked = String::with_capacity(body.len());
    let mut inside_memory_operand = false;
    for character in body.chars() {
        match character {
            '[' => {
                inside_memory_operand = true;
                masked.push(' ');
            }
            ']' => {
                inside_memory_operand = false;
                masked.push(' ');
            }
            _ if inside_memory_operand => masked.push(' '),
            _ => masked.push(character),
        }
    }

    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let bytes = masked.as_bytes();
    masked.match_indices(immediate).any(|(offset, matched)| {
        !offset
            .checked_sub(1)
            .is_some_and(|index| word(bytes[index]))
            && !bytes
                .get(offset + matched.len())
                .is_some_and(|byte| word(*byte))
    })
}

fn require_spill_free(name: &str, resource: &Resource) -> Result<(), Box<dyn Error>> {
    if resource.stack != 0 || resource.local != 0 {
        return Err(format!(
            "entry `{name}` uses STACK:{} LOCAL:{}",
            resource.stack, resource.local
        )
        .into());
    }

    Ok(())
}

fn require_registers(baseline: &Baseline, key: &str, actual: &[u32]) -> Result<(), Box<dyn Error>> {
    let actual = actual
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    require_stamp(baseline, key, &actual)
}

fn require_uniform_value(
    baseline: &Baseline,
    key: &str,
    actual: &[u32],
) -> Result<(), Box<dyn Error>> {
    let Some((&first, remaining)) = actual.split_first() else {
        return Err(format!("resource inventory `{key}` is empty").into());
    };
    if remaining.iter().any(|value| *value != first) {
        return Err(format!("resource inventory `{key}` is not uniform: {actual:?}").into());
    }

    require_stamp(baseline, key, &first.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        BENCH_DEVICE_BASELINES, COMPOSED_PERFORMANCE_SUITES, CUDA_OXIDE_TEST_TARGET, Handler,
        MAX_IDLE_DEVICE_MEMORY_MIB, MTP_BF16_PAGED_GQA_BENCHMARK_FILTER,
        MTP_LAYER_BENCHMARK_FILTER, MTP_LAYER_RESOURCE_BASELINES, MTP_LAYER_TEST_FILTER,
        OptimizationSuite, PERFORMANCE_SUITES, PerformanceSuite,
        QWEN35_LONG_CONTEXT_KV_TEST_FILTER, QWEN35_MTP_BATCH_GENERATION_TEST_FILTER,
        QWEN35_MTP_GENERATION_TEST_FILTER, QWEN35_RESIDENT_MODEL_TEST_FILTER,
        QWEN35_RESIDENT_MTP_TEST_FILTER, QWEN35_RESIDUAL_NORM_TEST_FILTER,
        QWEN35_TEXT_ENDPOINT_TEST_FILTER, QWEN36_LONG_CONTEXT_KV_TEST_FILTER,
        QWEN36_MTP_LAYER_TEST_FILTER, QWEN36_RESIDENT_MODEL_TEST_FILTER,
        QWEN38_FLASH_NEXT_ENGRAM_STAGING_TEST_FILTER, QWEN38_FLASH_NEXT_GDN_LAYER_TEST_FILTER,
        QWEN38_FLASH_NEXT_GENERATION_TEST_FILTER, QWEN38_FLASH_NEXT_LM_HEAD_TEST_FILTER,
        QWEN38_FLASH_NEXT_MTP_GENERATION_TEST_FILTER, QWEN38_FLASH_NEXT_MTP_ORACLE_TEST_FILTER,
        QWEN38_FLASH_NEXT_PLE_TEST_FILTER, QWEN38_FLASH_NEXT_PROJECTION_TEST_FILTER,
        QWEN38_FLASH_NEXT_PROMPT_PRIME_TEST_FILTER, QWEN38_FLASH_NEXT_QSA_LAYER_TEST_FILTER,
        QWEN38_FLASH_NEXT_RESIDENT_MODEL_TEST_FILTER, SM120_DEVICE_CODEGEN_CRATES,
        SM120_RESOURCE_BASELINES, STREAMING_WEIGHT_POOL_TEST_FILTER, SUBCOMMANDS,
        bench_device_baselines, bench_device_command, concatenated_resource_baselines,
        contains_immediate_operand, device_is_idle, dispatch, dispatch_probe, names_opcode,
        parse_baseline, parse_compute_pids, parse_cuda_toolkit_identity, parse_entries,
        parse_performance_device_sample, parse_performance_iteration, parse_resources,
        parse_rustc_identity, preflight_performance_baselines, qualification_test_arguments,
        require_consumed_baseline_keys, require_count, require_registers, require_uniform_value,
        resolve_target_output, sass_function_body, workspace_root,
    };
    use std::collections::BTreeSet;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::path::Path;

    #[test]
    fn parses_hashed_and_concrete_entries() {
        let ptx = ".visible .entry rms_norm_b1()\n.reqntid 512, 1, 1\n\
                   .visible .entry residual_rms_norm_TID_abc()\n.reqntid 512, 1, 1\n";
        let entries = parse_entries(ptx);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "rms_norm_b1");
        assert_eq!(entries[1].name, "residual_rms_norm_TID_abc");
    }

    #[test]
    fn qwen36_resident_filter_selects_oracle_and_accounting() {
        for test in [
            "qwen36_resident_model::tests::whole_model_matches_endpoint_oracles_and_graph_replay",
            "qwen36_resident_model_benchmark::tests::accounting_covers_every_layer_endpoint_and_selected_expert",
        ] {
            assert!(test.contains(QWEN36_RESIDENT_MODEL_TEST_FILTER));
        }
    }

    #[test]
    fn qwen35_residual_norm_filter_selects_oracle_and_accounting() {
        for test in [
            "residual_norm::tests::qwen35_residual_norm_exact_routes_match_independent_oracles_and_graph_replay",
            "residual_norm_benchmark::tests::qwen35_residual_norm_benchmark_arena_accounting_exposes_every_byte",
        ] {
            assert!(test.contains(QWEN35_RESIDUAL_NORM_TEST_FILTER));
        }
    }

    #[test]
    fn streaming_pool_filter_selects_oracles_and_accounting() {
        for test in [
            "streaming_weight_pool::tests::streaming_weight_pool_suite_byte_accounting_is_exact",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_lru_eviction_order_is_deterministic",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_every_cache_state_holds_identical_bits",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_a_miss_stalls_instead_of_serving_a_stale_slot",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_replay_coexists_with_slot_streaming",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_a_round_window_holds_identical_bits",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_uploads_overlap_an_in_flight_replay",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_the_bounce_path_reproduces_the_pinned_path",
            "streaming_weight_pool::tests::streaming_weight_pool_suite_a_failed_upload_never_commits_residency",
            "qwen38_flash_next_streaming_weight_pool_benchmark::tests::streaming_weight_pool_suite_benchmark_inventory_and_accounting_are_exact",
            "qwen38_flash_next_streaming_weight_pool_benchmark::tests::streaming_weight_pool_suite_benchmark_both_host_postures_are_exact",
            "qwen38_flash_next_streaming_weight_pool_benchmark::tests::streaming_weight_pool_suite_benchmark_big_pool_reports_pinning_and_upload_rate",
        ] {
            assert!(test.contains(STREAMING_WEIGHT_POOL_TEST_FILTER));
        }
    }

    #[test]
    fn qwen38_flash_next_engram_staging_filter_selects_oracles_and_accounting() {
        for test in [
            "qwen38_flash_next_engram_staging::tests::qwen38_flash_next_engram_staging_suite_host_gather_matches_literal_rows",
            "qwen38_flash_next_engram_staging::tests::qwen38_flash_next_engram_staging_suite_refusal_is_transactional",
            "qwen38_flash_next_engram_staging::tests::qwen38_flash_next_engram_staging_suite_device_plane_matches_eager_and_graph_consumers",
            "qwen38_flash_next_engram_staging_benchmark::tests::qwen38_flash_next_engram_staging_suite_benchmark_accounting_is_exact",
            "qwen38_flash_next_engram_staging_benchmark::tests::qwen38_flash_next_engram_staging_suite_benchmark_times_the_source_backed_owner",
        ] {
            assert!(test.contains(QWEN38_FLASH_NEXT_ENGRAM_STAGING_TEST_FILTER));
        }
    }

    #[test]
    fn qwen38_flash_next_mtp_generation_filter_selects_identity_and_accounting() {
        for test in [
            "qwen38_flash_next_mtp_generation::tests::qwen38_flash_next_mtp_generation_list_identity_is_exact",
            "qwen38_flash_next_mtp_generation::tests::qwen38_flash_next_mtp_generation_case_inventory_is_exact",
            "qwen38_flash_next_mtp_generation::tests::qwen38_flash_next_mtp_generation_benchmark_accounting_is_pinned",
            "qwen38_flash_next_mtp_generation::device_tests::qwen38_flash_next_mtp_generation_source_backed_identity",
        ] {
            assert!(test.contains(QWEN38_FLASH_NEXT_MTP_GENERATION_TEST_FILTER));
        }
    }

    #[test]
    fn qwen38_flash_next_ple_filter_selects_oracles_and_accounting() {
        for test in [
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_signed_root_keeps_the_sign_and_the_zero",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_probe_discriminates_the_gate_sign",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_fixture_keeps_both_residual_terms_visible",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_every_convolution_slot_is_distinct",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_engram_codes_and_scale_are_pinned",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_route_and_arena_inventory_is_exact",
            "qwen38_flash_next_ple::tests::qwen38_flash_next_ple_suite_exact_routes_match_independent_oracles_and_graph_replay",
            "qwen38_flash_next_ple_benchmark::tests::qwen38_flash_next_ple_suite_benchmark_arena_accounting_exposes_every_byte",
            "qwen38_flash_next_ple_benchmark::tests::qwen38_flash_next_ple_suite_benchmark_byte_accounting_covers_every_read_and_write_plane",
        ] {
            assert!(test.contains(QWEN38_FLASH_NEXT_PLE_TEST_FILTER));
        }
    }

    #[test]
    fn qwen38_flash_next_projection_filters_select_oracles_and_accounting() {
        for (filter, tests) in [
            (
                QWEN38_FLASH_NEXT_PROJECTION_TEST_FILTER,
                &[
                    "qwen38_flash_next_projection::tests::qwen38_flash_next_projection_suite_routes_match_independent_oracles_and_graph_replay",
                    "qwen38_flash_next_projection_benchmark::tests::qwen38_flash_next_projection_benchmark_inventory_and_accounting_are_exact",
                ][..],
            ),
            (
                QWEN38_FLASH_NEXT_LM_HEAD_TEST_FILTER,
                &[
                    "qwen38_flash_next_lm_head::tests::qwen38_flash_next_lm_head_suite_routes_match_independent_oracles_and_graph_replay",
                    "qwen38_flash_next_lm_head_benchmark::tests::qwen38_flash_next_lm_head_benchmark_inventory_and_accounting_are_exact",
                ][..],
            ),
        ] {
            for test in tests {
                assert!(test.contains(filter));
            }
        }
    }

    #[test]
    fn qwen38_flash_next_layer_filters_select_oracles_and_accounting() {
        for (filter, tests) in [
            (
                QWEN38_FLASH_NEXT_GDN_LAYER_TEST_FILTER,
                &[
                    "qwen38_flash_next_gdn_moe_layer::tests::source_layer0_matches_the_layer_oracle_and_graph_replay",
                    "qwen38_flash_next_gdn_moe_layer_benchmark::tests::accounting_grows_with_every_admitted_route",
                ][..],
            ),
            (
                QWEN38_FLASH_NEXT_QSA_LAYER_TEST_FILTER,
                &[
                    "qwen38_flash_next_qsa_moe_layer::tests::source_layer3_matches_the_layer_oracle_and_graph_replay",
                    "qwen38_flash_next_qsa_moe_layer_benchmark::tests::accounting_grows_with_every_admitted_route",
                ][..],
            ),
        ] {
            for test in tests {
                assert!(test.contains(filter), "`{filter}` does not select `{test}`");
            }
        }
    }

    #[test]
    fn qwen38_flash_next_resident_filter_selects_oracle_and_accounting() {
        for test in [
            "qwen38_flash_next_resident_model::tests::the_source_backed_resident_model_captures_and_decodes",
            "qwen38_flash_next_resident_model_benchmark::tests::qwen38_flash_next_resident_model_benchmark_accounting_covers_every_route_and_boundary",
            "qwen38_flash_next_resident_model_oracle::device::qwen38_flash_next_resident_model_oracle_matches_device_selection",
        ] {
            assert!(
                test.contains(QWEN38_FLASH_NEXT_RESIDENT_MODEL_TEST_FILTER),
                "resident filter does not select `{test}`"
            );
        }
    }

    #[test]
    fn qwen38_flash_next_generation_filter_selects_oracle_and_accounting() {
        for test in [
            "qwen38_flash_next_generation::tests::qwen38_flash_next_generation_matches_the_external_selection_contract",
            "qwen38_flash_next_generation_benchmark::tests::qwen38_flash_next_generation_benchmark_accounting_matches_each_swept_route",
        ] {
            assert!(
                test.contains(QWEN38_FLASH_NEXT_GENERATION_TEST_FILTER),
                "generation filter does not select `{test}`"
            );
        }
    }

    #[test]
    fn qwen38_flash_next_prompt_prime_filter_selects_exactness_and_accounting() {
        for test in [
            "qwen38_flash_next_prompt_prime_benchmark::tests::qwen38_flash_next_prompt_prime_wide_group_matches_sequential_outputs",
            "qwen38_flash_next_prompt_prime_benchmark::tests::qwen38_flash_next_prompt_prime_benchmark_accounting_covers_every_funded_width",
        ] {
            assert!(
                test.contains(QWEN38_FLASH_NEXT_PROMPT_PRIME_TEST_FILTER),
                "prompt-prime filter does not select `{test}`"
            );
        }
    }

    #[test]
    fn qwen35_long_context_filter_selects_lifecycle_and_accounting() {
        for test in [
            "qwen35_long_context_kv::tests::qwen35_long_context_kv_suite_byte_accounting_is_exact",
            "qwen35_long_context_kv::tests::qwen35_long_context_kv_suite_lifecycle_is_address_stable",
        ] {
            assert!(test.contains(QWEN35_LONG_CONTEXT_KV_TEST_FILTER));
        }
    }

    #[test]
    fn qwen35_composed_filters_select_oracles_and_accounting() {
        for (filter, tests) in [
            (
                QWEN35_TEXT_ENDPOINT_TEST_FILTER,
                &[
                    "qwen35_text_endpoint::tests::qwen35_text_endpoint_suite_token_and_logit_samples_cover_boundaries",
                    "qwen35_text_endpoint::tests::qwen35_text_endpoint_suite_source_endpoint_matches_complete_oracles_and_graph_replay",
                    "qwen35_text_endpoint_benchmark::tests::qwen35_text_endpoint_suite_benchmark_accounting_covers_bf16_endpoint_operations",
                ][..],
            ),
            (
                QWEN35_RESIDENT_MODEL_TEST_FILTER,
                &[
                    "qwen35_resident_model::tests::qwen35_resident_model_suite_samples_cover_exact_boundaries",
                    "qwen35_resident_model::tests::qwen35_resident_model_suite_whole_model_matches_endpoint_oracles_and_graph_replay",
                    "qwen35_resident_model_benchmark::tests::qwen35_resident_model_suite_benchmark_accounting_covers_every_layer_endpoint_and_route",
                ][..],
            ),
            (
                QWEN35_RESIDENT_MTP_TEST_FILTER,
                &[
                    "qwen35_resident_mtp::tests::qwen35_resident_mtp_suite_inventory_is_exact",
                    "qwen35_resident_mtp::tests::qwen35_resident_mtp_suite_composes_draft_prompt_and_mirrored_lifecycle",
                    "qwen35_resident_mtp_benchmark::tests::qwen35_resident_mtp_suite_benchmark_inventory_and_accounting_are_exact",
                ][..],
            ),
            (
                QWEN35_MTP_GENERATION_TEST_FILTER,
                &[
                    "qwen35_mtp_generation::tests::qwen35_mtp_generation_suite_inventory_selects_every_k",
                    "qwen35_mtp_generation::tests::qwen35_mtp_generation_suite_matches_target_only_greedy",
                    "qwen35_mtp_generation_benchmark::tests::qwen35_mtp_generation_suite_benchmark_uses_one_complete_k4_request",
                ][..],
            ),
            (
                QWEN35_MTP_BATCH_GENERATION_TEST_FILTER,
                &[
                    "qwen35_mtp_batch_generation::tests::qwen35_mtp_batch_generation_suite_compact_scheduler_matches_target_and_reuses_rejected_slots",
                    "qwen35_mtp_batch_generation_benchmark::tests::qwen35_mtp_batch_generation_suite_benchmark_inventory_is_exact",
                ][..],
            ),
        ] {
            for test in tests {
                assert!(test.contains(filter), "`{filter}` does not select `{test}`");
            }
        }
    }

    #[test]
    fn qwen36_mtp_layer_filter_selects_oracle_and_accounting() {
        for test in [
            "qwen36_mtp_layer::tests::qwen36_mtp_layer_suite_route_and_byte_inventory_is_exact",
            "qwen36_mtp_layer::tests::qwen36_mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes",
            "qwen36_mtp_layer_benchmark::tests::qwen36_mtp_layer_suite_benchmark_inventory_and_accounting_are_exact",
        ] {
            assert!(
                test.contains(QWEN36_MTP_LAYER_TEST_FILTER),
                "`{QWEN36_MTP_LAYER_TEST_FILTER}` does not select `{test}`"
            );
        }
    }

    #[test]
    fn qwen36_long_context_filter_selects_lifecycle_and_accounting() {
        for test in [
            "qwen36_long_context_kv::tests::qwen36_long_context_kv_suite_byte_accounting_is_exact",
            "qwen36_long_context_kv::tests::qwen36_long_context_kv_suite_lifecycle_is_address_stable",
        ] {
            assert!(test.contains(QWEN36_LONG_CONTEXT_KV_TEST_FILTER));
        }
    }

    #[test]
    fn mtp_paged_gqa_benchmark_filter_selects_both_accounting_tests() {
        for test in [
            "bf16_paged_gqa_benchmark::tests::mtp_bf16_paged_gqa_byte_accounting_covers_every_query_head_cache_read",
            "bf16_paged_gqa_benchmark::tests::mtp_bf16_paged_gqa_arena_accounting_exposes_every_padding_byte",
        ] {
            assert!(test.contains(MTP_BF16_PAGED_GQA_BENCHMARK_FILTER));
        }
    }

    #[test]
    fn mtp_layer_filters_exclude_qwen35_and_select_q38_accounting() {
        assert!(
            "mtp_layer::tests::mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes"
                .contains(MTP_LAYER_TEST_FILTER)
        );
        assert!(
            "mtp_layer_benchmark::tests::mtp_layer_suite_benchmark_inventory_and_accounting_are_exact"
                .contains(MTP_LAYER_BENCHMARK_FILTER)
        );
        assert!(!"qwen35_mtp_layer::tests::qwen35_mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes".contains(MTP_LAYER_TEST_FILTER));
    }

    #[test]
    fn parses_cuobjdump_resource_lines() {
        let resources = parse_resources(
            " Function rms_norm_b1:\n  REG:20 STACK:0 SHARED:1088 LOCAL:0 CONSTANT[0]:920\n",
        )
        .unwrap();
        let resource = resources["rms_norm_b1"];

        assert_eq!(resource.registers, 20);
        assert_eq!(resource.stack, 0);
        assert_eq!(resource.shared, 1_088);
        assert_eq!(resource.local, 0);
    }

    #[test]
    fn zero_generic_entries_fail_loudly() {
        let error = require_count("plain RMSNorm", 0, 8).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("silent generic-instantiation failure")
        );
    }

    #[test]
    fn parses_readable_compiler_identities() {
        let rustc = parse_rustc_identity(
            "rustc 1.96.0-nightly (55e86c996 2026-04-02)\n\
             commit-hash: 55e86c996809902e8bbad512cfb4d2c18be446d9\n\
             release: 1.96.0-nightly\n",
        )
        .unwrap();
        let cuda = parse_cuda_toolkit_identity(
            "Cuda compilation tools, release 13.3, V13.3.73\n\
             Build cuda_13.3.r13.3/compiler.38244171_0\n",
        )
        .unwrap();

        assert_eq!(rustc.0, "1.96.0-nightly");
        assert_eq!(rustc.1, "55e86c996809902e8bbad512cfb4d2c18be446d9");
        assert_eq!(cuda.release, "13.3");
        assert_eq!(cuda.version, "13.3.73");
    }

    #[test]
    fn shared_memory_contract_requires_one_value() {
        let baseline = parse_baseline("shared_bytes=1088\n").unwrap();

        require_uniform_value(&baseline, "shared_bytes", &[1_088; 16]).unwrap();
        assert!(require_uniform_value(&baseline, "shared_bytes", &[1_088, 1_024]).is_err());
    }

    /// The head-warp gate reads a structural constant out of PTX, and PTX spells
    /// byte displacements the same way it spells multipliers. This pins the one
    /// distinction the gate rests on: only operands outside `[...]` count.
    #[test]
    fn immediate_operands_exclude_addressing_offsets() {
        // the shapes the Qwen3.8-Flash-Next and Qwen3.8-27B prepare entries actually emit
        let qwen38_flash_next =
            "\tmad.lo.s64 \t%rd15, %rd14, -26, %rd13;\n\tld.global.b32 \t%r52, [%rd26+28];\n";
        assert!(contains_immediate_operand(qwen38_flash_next, "26"));
        assert!(!contains_immediate_operand(qwen38_flash_next, "28"));

        let qwen38 =
            "\tmad.lo.s64 \t%rd15, %rd14, -28, %rd13;\n\tld.global.b32 \t%r52, [%rd26+28];\n";
        assert!(contains_immediate_operand(qwen38, "28"));
        assert!(!contains_immediate_operand(qwen38, "26"));

        // a register name is not an immediate
        assert!(!contains_immediate_operand(
            "\tmov.u64 \t%rd26, %rd128;\n",
            "26"
        ));
    }

    #[test]
    fn opcode_prefix_does_not_match_shared_memory_suffixes() {
        let store = "\tst.shared.b32 \t[%r1], %r2;\n";
        assert!(!names_opcode(store, "red."));
        assert!(!names_opcode(store, "atom."));
        assert!(names_opcode(
            "\tred.global.add.u32 \t[%rd1], %r2;\n",
            "red."
        ));
        assert!(names_opcode(
            "\tatom.shared.add.u32 \t%r1, [%r2], %r3;\n",
            "atom."
        ));
    }

    #[test]
    fn baseline_preflight_lists_every_missing_file() {
        let root = workspace_root().unwrap();
        preflight_performance_baselines(root, ["qual/baselines/nvfp4-down-sm120.json"]).unwrap();

        let error = preflight_performance_baselines(
            root,
            [
                "qual/baselines/never-blessed-a.json",
                "qual/baselines/nvfp4-down-sm120.json",
                "qual/baselines/never-blessed-b.json",
            ],
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("never-blessed-a.json"));
        assert!(error.to_string().contains("never-blessed-b.json"));
        assert!(!error.to_string().contains("nvfp4-down-sm120.json"));
    }

    #[test]
    fn unconsumed_baseline_keys_are_rejected() {
        let baseline = parse_baseline("alpha_registers=80\nbeta_registers=96\n").unwrap();
        assert!(!baseline.contains_key("beta_regs"));
        require_registers(&baseline, "alpha_registers", &[80]).unwrap();
        drop(baseline);

        let error = require_consumed_baseline_keys().err().unwrap();
        assert!(error.to_string().contains("beta_registers"));
        assert!(!error.to_string().contains("alpha_registers"));
        require_consumed_baseline_keys().unwrap();

        let baseline = parse_baseline("alpha_registers=80\n").unwrap();
        require_registers(&baseline, "alpha_registers", &[80]).unwrap();
        drop(baseline);
        require_consumed_baseline_keys().unwrap();
    }

    #[test]
    fn performance_suite_names_select_the_complete_inventory() {
        let names = PERFORMANCE_SUITES
            .iter()
            .map(|suite| suite.name())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "residual-norm",
                "fp8-qkv",
                "fp8-gdn-input",
                "fp8-lm-head",
                "fp8-swiglu",
                "fp8-down",
                "nvfp4-swiglu",
                "nvfp4-down",
                "gdn-prepare",
                "gdn-recurrence",
                "gdn-output",
                "attention-qk-prepare",
                "paged-gqa",
                "long-context-paged-gqa",
                "attention-output",
                "mtp-bf16-fusion",
                "mtp-bf16-qkv",
                "mtp-bf16-qk-prepare",
                "mtp-bf16-paged-gqa",
                "mtp-bf16-attention-output",
                "mtp-bf16-mlp",
            ]
        );
        for suite in PERFORMANCE_SUITES {
            assert_eq!(
                PerformanceSuite::parse(suite.name()).unwrap().name(),
                suite.name()
            );
        }
        assert!(PerformanceSuite::parse("unknown").is_err());
    }

    #[test]
    fn sm120_build_receipt_covers_every_resource_gate() {
        assert_eq!(
            SM120_RESOURCE_BASELINES,
            [
                "qual/baselines/residual-norm-sm120.txt",
                "qual/baselines/qwen35-residual-norm-sm120.txt",
                "qual/baselines/qwen36-residual-norm-sm120.txt",
                "qual/baselines/fp8-qkv-sm120.txt",
                "qual/baselines/fp8-gdn-input-sm120.txt",
                "qual/baselines/fp8-lm-head-sm120.txt",
                "qual/baselines/fp8-swiglu-sm120.txt",
                "qual/baselines/fp8-down-sm120.txt",
                "qual/baselines/nvfp4-swiglu-sm120.txt",
                "qual/baselines/qwen35-nvfp4-swiglu-sm120.txt",
                "qual/baselines/nvfp4-down-sm120.txt",
                "qual/baselines/qwen35-nvfp4-down-sm120.txt",
                "qual/baselines/qwen35-nvfp4-qkv-sm120.txt",
                "qual/baselines/qwen35-bf16-lm-head-sm120.txt",
                "qual/baselines/qwen36-moe-router-sm120.txt",
                "qual/baselines/qwen36-moe-experts-sm120.txt",
                "qual/baselines/qwen36-nvfp4-lm-head-sm120.txt",
                "qual/baselines/qwen36-fp8-qkv-sm120.txt",
                "qual/baselines/qwen36-gdn-input-sm120.txt",
                "qual/baselines/qwen36-gdn-output-sm120.txt",
                "qual/baselines/qwen36-attention-output-sm120.txt",
                "qual/baselines/qwen35-nvfp4-gdn-input-sm120.txt",
                "qual/baselines/qwen35-gdn-prepare-sm120.txt",
                "qual/baselines/qwen35-gdn-recurrence-sm120.txt",
                "qual/baselines/qwen35-nvfp4-attention-output-sm120.txt",
                "qual/baselines/gdn-prepare-sm120.txt",
                "qual/baselines/gdn-recurrence-sm120.txt",
                "qual/baselines/gdn-state-snapshot-sm120.txt",
                "qual/baselines/gdn-output-sm120.txt",
                "qual/baselines/qwen38-flash-next-gdn-prepare-sm120.txt",
                "qual/baselines/qwen38-flash-next-gdn-recurrence-sm120.txt",
                "qual/baselines/qwen38-flash-next-qsa-prepare-sm120.txt",
                "qual/baselines/qwen38-flash-next-qsa-attention-sm120.txt",
                "qual/baselines/qwen38-flash-next-qsa-selection-sm120.txt",
                "qual/baselines/qwen38-flash-next-moe-router-sm120.txt",
                "qual/baselines/qwen38-flash-next-moe-experts-sm120.txt",
                "qual/baselines/qwen38-flash-next-projection-sm120.txt",
                "qual/baselines/qwen38-flash-next-lm-head-sm120.txt",
                "qual/baselines/attention-qk-prepare-sm120.txt",
                "qual/baselines/qwen35-attention-qk-prepare-sm120.txt",
                "qual/baselines/qwen36-attention-qk-prepare-sm120.txt",
                "qual/baselines/qwen36-fp8-attention-qk-prepare-sm120.txt",
                "qual/baselines/paged-gqa-sm120.txt",
                "qual/baselines/qwen35-paged-gqa-sm120.txt",
                "qual/baselines/qwen36-paged-gqa-sm120.txt",
                "qual/baselines/qwen36-fp8-paged-gqa-sm120.txt",
                "qual/baselines/long-context-paged-gqa-sm120.txt",
                "qual/baselines/attention-output-sm120.txt",
                "qual/baselines/mtp-bf16-fusion-sm120.txt",
                "qual/baselines/mtp-bf16-attention-output-sm120.txt",
                "qual/baselines/mtp-bf16-mlp-sm120.txt",
                "qual/baselines/mtp-bf16-qkv-sm120.txt",
                "qual/baselines/mtp-bf16-qk-prepare-sm120.txt",
                "qual/baselines/mtp-bf16-paged-gqa-sm120.txt",
                "qual/baselines/qwen35-mtp-sm120.txt",
                "qual/baselines/qwen36-mtp-sm120.txt",
                "qual/baselines/qwen38-flash-next-hyper-connection-sm120.txt",
                "qual/baselines/qwen38-flash-next-ple-sm120.txt",
            ]
        );
    }

    #[test]
    fn composed_performance_inventory_and_dependency_cones_are_exact() {
        let names = COMPOSED_PERFORMANCE_SUITES
            .iter()
            .map(|suite| suite.name())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "nvfp4-mlp",
                "dense-fp8-mlp",
                "dense-fp8-gdn-layer",
                "full-attention-layer",
                "mtp-layer",
                "text-endpoint",
                "resident-model",
                "resident-prefill",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("nvfp4-down")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            [
                "nvfp4-down",
                "nvfp4-mlp",
                "resident-model",
                "resident-prefill",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("fp8-down")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            [
                "fp8-down",
                "dense-fp8-mlp",
                "dense-fp8-gdn-layer",
                "full-attention-layer",
                "resident-model",
                "resident-prefill",
                "resident-long-context-model",
            ]
        );

        let cone = OptimizationSuite::parse("long-context-paged-gqa")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(
            cone,
            ["long-context-paged-gqa", "resident-long-context-model"]
        );

        let cone = OptimizationSuite::parse("mtp-bf16-qkv")
            .unwrap()
            .dependency_cone()
            .into_iter()
            .map(OptimizationSuite::name)
            .collect::<Vec<_>>();
        assert_eq!(cone, ["mtp-bf16-qkv", "mtp-layer"]);

        let mtp_layer = OptimizationSuite::parse("mtp-layer").unwrap();
        assert_eq!(mtp_layer.resource_baselines(), MTP_LAYER_RESOURCE_BASELINES);
        assert!(mtp_layer.requires_snapshot());
    }

    #[test]
    fn idle_evidence_requires_all_three_signals() {
        assert!(device_is_idle(0, 234, &[]));
        assert!(device_is_idle(9, 234, &[]));
        assert!(device_is_idle(0, MAX_IDLE_DEVICE_MEMORY_MIB, &[]));
        assert!(!device_is_idle(10, 234, &[]));
        assert!(!device_is_idle(0, MAX_IDLE_DEVICE_MEMORY_MIB + 1, &[]));
        assert!(!device_is_idle(0, 234, &[123]));
    }

    #[test]
    fn parses_exact_performance_preflight_samples_and_processes() {
        assert_eq!(
            parse_performance_device_sample("NVIDIA GeForce RTX 5090, 0, 503\n").unwrap(),
            ("NVIDIA GeForce RTX 5090".to_string(), 0, 503)
        );
        assert!(parse_performance_device_sample("RTX 5090, 0").is_err());
        assert_eq!(parse_compute_pids("\n").unwrap(), Vec::<u32>::new());
        assert_eq!(parse_compute_pids("123\n456\n").unwrap(), vec![123, 456]);
        assert!(parse_compute_pids("N/A\n").is_err());
    }

    #[test]
    fn perf_iteration_requires_one_exact_batch_and_hypothesis() {
        let options = parse_performance_iteration(&[
            OsString::from("nvfp4-down"),
            OsString::from("--batch"),
            OsString::from("1"),
            OsString::from("--hypothesis"),
            OsString::from("coalesce B=1 loads"),
        ])
        .unwrap();
        assert_eq!(options.suite, PerformanceSuite::Nvfp4Down);
        assert_eq!(options.snapshot, None);
        assert_eq!(options.batch_size, 1);
        assert_eq!(options.hypothesis, "coalesce B=1 loads");

        let options = parse_performance_iteration(&[
            OsString::from("mtp-bf16-fusion"),
            OsString::from("/snapshot"),
            OsString::from("--batch"),
            OsString::from("8"),
            OsString::from("--hypothesis"),
            OsString::from("reuse normalized planes"),
        ])
        .unwrap();
        assert_eq!(options.snapshot, Some(OsString::from("/snapshot")));

        assert!(
            parse_performance_iteration(&[
                OsString::from("nvfp4-down"),
                OsString::from("--batch"),
                OsString::from("9"),
                OsString::from("--hypothesis"),
                OsString::from("invalid batch"),
            ])
            .is_err()
        );
    }

    #[test]
    fn diagnostic_output_is_confined_to_the_ignored_target_tree() {
        let root = std::path::Path::new("/repository");
        assert_eq!(
            resolve_target_output(root, std::ffi::OsStr::new("target/report.json")).unwrap(),
            root.join("target/report.json")
        );
        assert!(resolve_target_output(root, std::ffi::OsStr::new("AGENTS.md")).is_err());
        assert!(resolve_target_output(root, std::ffi::OsStr::new("target/../AGENTS.md")).is_err());
        assert!(resolve_target_output(root, std::ffi::OsStr::new("/tmp/report.json")).is_err());
    }

    #[test]
    fn isolates_one_sass_function() {
        let sass = "Function : first\nQMMA.16832.F32.E4M3.E4M3 R0, R1, R2, R3;\n\
                    \t\tFunction : second\nNOP;\n";

        assert!(
            sass_function_body(sass, "first")
                .unwrap()
                .contains("QMMA.16832.F32.E4M3.E4M3")
        );
        assert!(!sass_function_body(sass, "second").unwrap().contains("QMMA"));
        assert!(sass_function_body(sass, "missing").is_none());
    }

    #[test]
    fn prefix_extension_names_never_alias() {
        let sass = "\t\tFunction : foo_bar\nNOP;\n\
                    \t\tFunction : foo\nQMMA.16832.F32.E4M3.E4M3 R0, R1, R2, R3;\n";

        let body = sass_function_body(sass, "foo").unwrap();
        assert!(body.contains("QMMA"));
        assert!(!body.contains("NOP"));
        assert!(sass_function_body(sass, "foo_bar").unwrap().contains("NOP"));
        assert!(sass_function_body(sass, "foo_").is_none());
    }

    #[test]
    fn qualification_arguments_preserve_filter_and_harness_order() {
        let trailing = ["--exact", "--include-ignored", "--nocapture"];

        assert_eq!(
            qualification_test_arguments("suite_filter", &trailing),
            [
                "test",
                "--arch",
                "sm_120a",
                "--cargo-target-dir",
                CUDA_OXIDE_TEST_TARGET,
                "--device-codegen-crate",
                SM120_DEVICE_CODEGEN_CRATES,
                "--",
                "--package",
                "tuisko-qual",
                "--release",
                "--lib",
                "--",
                "suite_filter",
                "--exact",
                "--include-ignored",
                "--nocapture",
            ]
        );
    }

    #[test]
    fn benchmark_command_preserves_suite_arguments_and_hash() {
        let arguments = [OsString::from("--batch"), OsString::from("8")];
        let command = bench_device_command(
            Path::new("/tmp/bench-device"),
            "example-suite",
            &arguments,
            "baseline-hash",
        );

        assert_eq!(command.get_program(), OsStr::new("/tmp/bench-device"));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            [
                OsStr::new("example-suite"),
                OsStr::new("--batch"),
                OsStr::new("8"),
            ]
        );
        assert_eq!(
            command.get_envs().collect::<Vec<_>>(),
            [(
                OsStr::new("TUISKO_GENERATOR_BASELINE_SHA256"),
                Some(OsStr::new("baseline-hash")),
            )]
        );

        let root =
            std::env::temp_dir().join(format!("xtask-bench-baselines-{}", std::process::id()));
        fs::create_dir_all(root.join("qual/baselines")).unwrap();
        fs::write(root.join("qual/baselines/first.txt"), b"alpha").unwrap();
        fs::write(root.join("qual/baselines/second.txt"), b"beta").unwrap();
        let forward = ["qual/baselines/first.txt", "qual/baselines/second.txt"];
        let reversed = ["qual/baselines/second.txt", "qual/baselines/first.txt"];
        assert_eq!(
            concatenated_resource_baselines(&root, &forward).unwrap(),
            b"alphabeta"
        );
        assert_eq!(
            concatenated_resource_baselines(&root, &reversed).unwrap(),
            b"betaalpha"
        );
        assert!(bench_device_baselines("no-such-suite").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn subcommand_table_is_unique_and_enforces_argument_policy() {
        let names = SUBCOMMANDS
            .iter()
            .map(|subcommand| subcommand.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), SUBCOMMANDS.len(), "the table repeats a name");

        let root = workspace_root().unwrap();
        let stray = [OsString::from("--stray")];
        for subcommand in SUBCOMMANDS {
            if matches!(subcommand.run, Handler::NoArguments(_)) {
                let error = dispatch(root, OsStr::new(subcommand.name), &stray).unwrap_err();
                assert_eq!(
                    error.to_string(),
                    format!("`{}` takes no arguments", subcommand.name)
                );
            }
        }

        let error = dispatch(root, OsStr::new("qualify-nothing"), &[]).unwrap_err();
        assert_eq!(error.to_string(), "unknown xtask command `qualify-nothing`");

        let error = dispatch(root, OsStr::new("remote"), &[]).unwrap_err();
        #[cfg(feature = "remote")]
        assert!(
            error
                .to_string()
                .starts_with("usage: cargo run -p xtask --features remote -- remote <"),
            "`remote` reached {error}"
        );
        #[cfg(not(feature = "remote"))]
        assert_eq!(
            error.to_string(),
            "remote execution requires `cargo run -p xtask --features remote -- remote ...`"
        );
    }

    /// Every canonical `bench-device` subcommand must reach the handler that
    /// names its own suite, forwards its arguments unchanged, and resolves the
    /// baselines that suite declares. Subcommand, handler and suite are bound
    /// in one chain so naming a sibling suite fails visibly.
    #[test]
    fn bench_device_subcommands_bind_their_suite() {
        // The `bench-*` subcommands that do not run a `bench-device` suite:
        // the startup and server harnesses and the `PerformanceSuite`
        // comparator commands, transcribed from their handlers.
        const HOST_BENCH_SUBCOMMANDS: &[&str] = &[
            "bench-qwen38-flash-next-server",
            "bench-qwen38-flash-next",
            "bench-qwen38-flash-next-generation",
            "bench-qwen38-flash-next-prompt-prime",
            "bench-qwen38-flash-next-resident-model",
            "bench-startup",
            "bench-server",
            "bench-residual-norm",
            "bench-fp8-qkv",
            "bench-fp8-gdn-input",
            "bench-fp8-lm-head",
            "bench-fp8-swiglu",
            "bench-fp8-down",
            "bench-nvfp4-swiglu",
            "bench-nvfp4-down",
            "bench-gdn-prepare",
            "bench-gdn-recurrence",
            "bench-gdn-output",
            "bench-attention-qk-prepare",
            "bench-paged-gqa",
            "bench-long-context-paged-gqa",
            "bench-attention-output",
            "bench-mtp-bf16-fusion",
            "bench-mtp-bf16-attention-output",
            "bench-mtp-bf16-mlp",
            "bench-mtp-bf16-qkv",
            "bench-mtp-bf16-qk-prepare",
            "bench-mtp-bf16-paged-gqa",
        ];

        let arguments = [
            OsString::from("/snapshot"),
            OsString::from("--batch"),
            OsString::from("8"),
        ];

        for (suite, baselines) in BENCH_DEVICE_BASELINES {
            let command = format!("bench-{suite}");
            assert_eq!(
                dispatch_probe::observe(&command, &arguments),
                dispatch_probe::Spawn::BenchDevice {
                    suite: (*suite).to_owned(),
                    arguments: arguments.to_vec(),
                    baselines: baselines.to_vec(),
                },
                "`{command}`"
            );
        }

        // No other `bench-*` row may reach `run_bench_device`: every device
        // suite is named by exactly one subcommand, and that subcommand is
        // `bench-` followed by the suite.
        let device = BENCH_DEVICE_BASELINES
            .iter()
            .map(|(suite, _)| format!("bench-{suite}"))
            .collect::<BTreeSet<_>>();
        assert_eq!(device.len(), BENCH_DEVICE_BASELINES.len());
        let rows = SUBCOMMANDS
            .iter()
            .map(|subcommand| subcommand.name)
            .filter(|name| name.starts_with("bench-"))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            rows,
            device
                .union(
                    &HOST_BENCH_SUBCOMMANDS
                        .iter()
                        .map(|name| (*name).to_owned())
                        .collect()
                )
                .cloned()
                .collect()
        );
    }

    /// Every canonical `qualify-*` subcommand must reach the handler that runs
    /// its own test filter, under its own harness flags and its own snapshot
    /// variable. This binds the whole chain from subcommand to spawned argv.
    #[test]
    fn qualification_subcommands_bind_their_test_filter() {
        // Keep expected flags independent from the production constants.
        const SERIAL: &[&str] = &["--include-ignored", "--nocapture", "--test-threads=1"];
        const IGNORED: &[&str] = &["--include-ignored", "--nocapture"];
        const EXACT_SERIAL: &[&str] = &[
            "--exact",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ];
        const SERIAL_SKIPPING_FP8_QKV: &[&str] = &[
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
            "--skip",
            "qwen36_fp8_qkv",
        ];
        const NO_SNAPSHOT: Option<&str> = None;
        const SNAPSHOT: Option<&str> = Some("TUISKO_SNAPSHOT");
        const QWEN35_SNAPSHOT: Option<&str> = Some("TUISKO_QWEN35_SNAPSHOT");
        const QWEN36_SNAPSHOT: Option<&str> = Some("TUISKO_QWEN36_SNAPSHOT");
        const QWEN38_FLASH_NEXT_SNAPSHOT: Option<&str> = Some("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT");

        // (subcommand, first filter, harness flags, snapshot variable)
        const EXPECTED_QUALIFICATION_ROUTES: &[(&str, &str, &[&str], Option<&str>)] = &[
            (
                "qualify-residual-norm",
                "residual_norm_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-residual-norm",
                "qwen35_residual_norm",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-residual-norm",
                "qwen36_residual_norm",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-swiglu",
                "qwen35_nvfp4_swiglu",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-down",
                "qwen35_nvfp4_down",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-qkv",
                "qwen35_nvfp4_qkv",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-moe-router",
                "qwen36_moe_router::tests::exact_routes_match_independent_oracles_and_graph_replay",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-moe-experts",
                "qwen36_moe_experts::tests::exact_routes_match_independent_oracles_and_graph_replay",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-nvfp4-lm-head",
                "qwen36_nvfp4_lm_head",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-fp8-qkv",
                "qwen36_fp8_qkv",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-gdn-input",
                "qwen36_gdn_input",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-gdn-output",
                "qwen36_gdn_output::tests::exact_routes_match_independent_oracles_and_graph_replay",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-attention-output",
                "qwen36_attention_output",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-gdn-prepare",
                "qwen35_gdn_prepare::tests::qwen36_exact_routes_match_shared_independent_oracle",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-gdn-recurrence",
                "qwen35_gdn_recurrence::tests::qwen36_exact_routes_match_shared_independent_oracle",
                EXACT_SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-bf16-lm-head",
                "qwen35_bf16_lm_head::tests",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-bf16-fusion",
                "qwen35_fusion_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-bf16-attention",
                "qwen35_mtp_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen36-mtp-bf16-attention",
                "qwen36_mtp_",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-qwen36-mtp-bf16-moe",
                "qwen36_mtp_bf16_moe_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-bf16-mlp",
                "qwen35_mtp_mlp_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-text-endpoint",
                "qwen35_text_endpoint_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen36-text-endpoint",
                "qwen36_text_endpoint::tests",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-qwen36-resident-model",
                "qwen36_resident_model",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-qwen36-generation",
                "qwen36_generation::tests",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-qwen35-resident-model",
                "qwen35_resident_model_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-resident-mtp",
                "qwen35_resident_mtp_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-generation",
                "qwen35_mtp_generation_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-batch-generation",
                "qwen35_mtp_batch_generation_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-generation",
                "qwen35_generation::tests",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-long-context-kv",
                "qwen35_long_context_kv::tests",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-long-context-kv",
                "qwen36_long_context_kv::tests",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-streaming-pool",
                "streaming_weight_pool_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-engram-staging",
                "qwen38_flash_next_engram_staging_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-gdn-input",
                "qwen35_nvfp4_gdn_input",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-gdn-prepare",
                "qwen35_gdn_prepare",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-gdn-recurrence",
                "qwen35_gdn_recurrence",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-gdn-output",
                "qwen35_nvfp4_gdn_output",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-attention-output",
                "qwen35_nvfp4_attention_output::tests::exact_batches_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen35-nvfp4-mlp",
                "nvfp4_mlp::tests::qwen35_source_layer0_matches_complete_oracles_and_graph_replay",
                IGNORED,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen35-attention-qk-prepare",
                "attention_qk_prepare::tests::qwen35_",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-attention-qk-prepare",
                "attention_qk_prepare::tests::qwen36_exact_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-fp8-attention-qk-prepare",
                "attention_qk_prepare::tests::qwen36_fp8_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-fp8-qkv",
                "fp8_qkv",
                SERIAL_SKIPPING_FP8_QKV,
                NO_SNAPSHOT,
            ),
            (
                "qualify-fp8-gdn-input",
                "fp8_gdn_input",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-fp8-swiglu",
                "fp8_swiglu_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            ("qualify-fp8-down", "fp8_down_suite_", SERIAL, NO_SNAPSHOT),
            ("qualify-nvfp4-swiglu", "nvfp4_swiglu", IGNORED, NO_SNAPSHOT),
            ("qualify-nvfp4-down", "nvfp4_down", IGNORED, NO_SNAPSHOT),
            (
                "qualify-nvfp4-mlp",
                "nvfp4_mlp::tests::source_layer55_matches_complete_oracles_and_graph_replay",
                IGNORED,
                SNAPSHOT,
            ),
            (
                "qualify-gdn-prepare",
                "gdn_prepare::tests",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-gdn-prepare",
                "qwen38_flash_next_gdn_prepare",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-gdn-recurrence",
                "qwen38_flash_next_gdn_recurrence",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-hyper-connection",
                "qwen38_flash_next_hyper_connection_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-ple",
                "qwen38_flash_next_ple_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-qsa-prepare",
                "qwen38_flash_next_qsa_prepare",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-qsa-attention",
                "qwen38_flash_next_qsa_attention",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-qsa-selection",
                "qwen38_flash_next_qsa_selection",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-moe-router",
                "qwen38_flash_next_moe_router",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-moe-experts",
                "qwen38_flash_next_moe_experts",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-projections",
                "qwen38_flash_next_projection",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-lm-head",
                "qwen38_flash_next_lm_head",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-generation",
                "qwen38_flash_next_generation",
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-prompt-prime",
                QWEN38_FLASH_NEXT_PROMPT_PRIME_TEST_FILTER,
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-compact-generation",
                "qwen38_flash_next_compact_generation",
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-gdn-layer",
                "qwen38_flash_next_gdn_moe_layer",
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-qsa-layer",
                "qwen38_flash_next_qsa_moe_layer",
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-resident-model",
                QWEN38_FLASH_NEXT_RESIDENT_MODEL_TEST_FILTER,
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-mtp-oracle",
                QWEN38_FLASH_NEXT_MTP_ORACLE_TEST_FILTER,
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-qwen38-flash-next-mtp-generation",
                QWEN38_FLASH_NEXT_MTP_GENERATION_TEST_FILTER,
                SERIAL,
                QWEN38_FLASH_NEXT_SNAPSHOT,
            ),
            (
                "qualify-gdn-recurrence",
                "gdn_recurrence::tests::route_inventory_and_arena_accounting_are_exact",
                EXACT_SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-gdn-output",
                "gdn_output::tests",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-attention-qk-prepare",
                "attention_qk_prepare",
                IGNORED,
                NO_SNAPSHOT,
            ),
            ("qualify-paged-gqa", "paged_gqa_suite_", SERIAL, NO_SNAPSHOT),
            (
                "qualify-qwen35-paged-gqa",
                "paged_gqa::tests::qwen35_bf16_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-paged-gqa",
                "paged_gqa::tests::qwen36_bf16_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-qwen36-fp8-paged-gqa",
                "paged_gqa::tests::qwen36_fp8_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-long-context-paged-gqa",
                "long_context_paged_gqa",
                IGNORED,
                NO_SNAPSHOT,
            ),
            (
                "qualify-attention-output",
                "attention_output::tests::attention_output_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-fusion",
                "mtp_bf16_fusion_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-attention-output",
                "mtp_bf16_attention_output_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-mlp",
                "mtp_bf16_mlp_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-qkv",
                "mtp_bf16_qkv_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-qk-prepare",
                "mtp_bf16_qk_prepare_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-bf16-paged-gqa",
                "mtp_bf16_paged_gqa_suite_",
                SERIAL,
                NO_SNAPSHOT,
            ),
            (
                "qualify-dense-fp8-mlp",
                "dense_fp8_mlp_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-dense-fp8-gdn-layer",
                "dense_fp8_gdn_layer::tests::source_layer60_matches_complete_seam_oracles_and_graph_replay",
                IGNORED,
                SNAPSHOT,
            ),
            (
                "qualify-full-attention-layer",
                "full_attention_layer_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-layer",
                "mtp_layer::tests::mtp_layer_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-qwen35-mtp-layer",
                "qwen35_mtp_layer_suite_",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen36-mtp-layer",
                "qwen36_mtp_layer_suite_",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-target-mtp-verify",
                "target_mtp_verify::tests::exact_target_verify_and_commit_match_source_oracles",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-mtp-prompt-prime",
                "mtp_prompt_prime_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-resident-mtp",
                "resident_mtp_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-generation-mtp-greedy",
                "resident_mtp_generation_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-generation-mtp-sampling",
                "resident_mtp_sampling_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-generation-mtp-batch",
                "resident_mtp_batch_suite_",
                SERIAL,
                SNAPSHOT,
            ),
            (
                "qualify-qwen35-full-attention-layer",
                "qwen35_full_attention_layer::tests",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen36-full-attention-layer",
                "qwen36_full_attention_layer::tests",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-qwen35-gdn-layer",
                "qwen35_gdn_layer::tests",
                SERIAL,
                QWEN35_SNAPSHOT,
            ),
            (
                "qualify-qwen36-gdn-moe-layer",
                "qwen36_gdn_moe_layer::tests",
                SERIAL,
                QWEN36_SNAPSHOT,
            ),
            (
                "qualify-resident-model",
                "resident_model::tests::source_model_matches_final_oracle_and_exact_graph_replay",
                IGNORED,
                SNAPSHOT,
            ),
            (
                "qualify-resident-generation",
                "resident_generation::tests::source_frontend_generation_matches_vllm_tokens_and_streaming",
                IGNORED,
                SNAPSHOT,
            ),
            (
                "qualify-resident-batch-generation",
                "resident_batch_generation::tests::compact_scheduler_matches_sequential_requests_and_recycles_holes",
                IGNORED,
                SNAPSHOT,
            ),
            (
                "qualify-text-endpoint",
                "text_endpoint::tests::source_endpoint_matches_independent_oracles_and_graph_replay",
                IGNORED,
                SNAPSHOT,
            ),
        ];

        assert_eq!(EXPECTED_QUALIFICATION_ROUTES.len(), 105);

        let snapshot = OsString::from("/snapshot");
        for &(command, filter, trailing, variable) in EXPECTED_QUALIFICATION_ROUTES {
            let arguments: &[OsString] = match variable {
                Some(_) => std::slice::from_ref(&snapshot),
                None => &[],
            };
            assert_eq!(
                dispatch_probe::observe(command, arguments),
                dispatch_probe::Spawn::Qualification {
                    filter: filter.to_owned(),
                    trailing: trailing.iter().map(|flag| (*flag).to_owned()).collect(),
                    environment: variable.map(|key| (key.to_owned(), snapshot.clone())),
                },
                "`{command}`"
            );
        }
    }
}

//! Independent operator qualification for one compiled GPU target.

#![cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]

#[cfg(any(
    all(feature = "device", feature = "sm89"),
    all(feature = "device", feature = "sm86"),
    all(feature = "sm89", feature = "sm86")
))]
compile_error!("select exactly one device target feature");

#[cfg(feature = "device")]
mod attention_output;
#[cfg(feature = "device")]
mod attention_output_benchmark;
#[cfg(feature = "device")]
mod attention_output_prefill;
#[cfg(feature = "device")]
mod attention_qk_prepare;
#[cfg(feature = "device")]
mod attention_qk_prepare_benchmark;
#[cfg(feature = "device")]
mod bf16_paged_gqa_benchmark;
#[cfg(feature = "device")]
mod dense_fp8_gdn_layer;
#[cfg(feature = "device")]
mod dense_fp8_gdn_layer_benchmark;
#[cfg(feature = "device")]
mod dense_fp8_mlp;
#[cfg(feature = "device")]
mod dense_fp8_mlp_benchmark;
mod device_benchmark;
#[cfg(feature = "device")]
mod fp8_down;
#[cfg(feature = "device")]
mod fp8_down_benchmark;
#[cfg(feature = "device")]
mod fp8_gdn_input;
#[cfg(feature = "device")]
mod fp8_gdn_input_benchmark;
#[cfg(feature = "device")]
mod fp8_lm_head;
#[cfg(feature = "device")]
mod fp8_lm_head_benchmark;
#[cfg(any(feature = "device", feature = "sm89"))]
mod fp8_projection_oracle;
#[cfg(any(feature = "device", feature = "sm89"))]
mod fp8_qkv;
#[cfg(any(feature = "device", feature = "sm89"))]
mod fp8_qkv_benchmark;
#[cfg(feature = "device")]
mod fp8_swiglu;
#[cfg(feature = "device")]
mod fp8_swiglu_benchmark;
#[cfg(feature = "device")]
mod full_attention_layer;
#[cfg(feature = "device")]
mod full_attention_layer_benchmark;
#[cfg(feature = "device")]
mod gdn_output;
#[cfg(feature = "device")]
mod gdn_output_benchmark;
#[cfg(feature = "device")]
mod gdn_prepare;
#[cfg(feature = "device")]
mod gdn_prepare_benchmark;
#[cfg(feature = "device")]
mod gdn_recurrence;
#[cfg(feature = "device")]
mod gdn_recurrence_benchmark;
#[cfg(feature = "device")]
mod gdn_state_snapshot;
mod harness;
#[cfg(feature = "device")]
mod long_context_paged_gqa;
#[cfg(feature = "device")]
mod long_context_paged_gqa_benchmark;
#[cfg(feature = "device")]
mod mtp_bf16_attention_output;
#[cfg(feature = "device")]
mod mtp_bf16_attention_output_benchmark;
#[cfg(feature = "device")]
mod mtp_bf16_fusion;
#[cfg(feature = "device")]
mod mtp_bf16_fusion_benchmark;
#[cfg(feature = "device")]
mod mtp_bf16_mlp;
#[cfg(feature = "device")]
mod mtp_bf16_mlp_benchmark;
#[cfg(feature = "device")]
mod mtp_bf16_qkv;
#[cfg(feature = "device")]
mod mtp_bf16_qkv_benchmark;
#[cfg(feature = "device")]
mod mtp_layer;
#[cfg(feature = "device")]
mod mtp_layer_benchmark;
#[cfg(feature = "device")]
mod mtp_prompt_prime;
#[cfg(feature = "device")]
mod mtp_prompt_prime_benchmark;
#[cfg(feature = "sm89")]
mod nvfp4_down;
#[cfg(feature = "sm89")]
mod nvfp4_down_benchmark;
#[cfg(feature = "device")]
mod nvfp4_down_benchmark_sm120;
#[cfg(feature = "device")]
mod nvfp4_down_sm120;
#[cfg(feature = "device")]
mod nvfp4_mlp;
#[cfg(feature = "device")]
mod nvfp4_mlp_benchmark;
#[cfg(any(feature = "sm89", feature = "sm86"))]
mod nvfp4_swiglu;
#[cfg(any(feature = "sm89", feature = "sm86"))]
mod nvfp4_swiglu_benchmark;
#[cfg(feature = "device")]
mod nvfp4_swiglu_benchmark_sm120;
#[cfg(feature = "device")]
mod nvfp4_swiglu_sm120;
mod oracles;
#[cfg(feature = "device")]
mod paged_gqa;
#[cfg(feature = "device")]
mod paged_gqa_benchmark;
#[cfg(feature = "device")]
mod paged_gqa_macro_prefill;
#[cfg(feature = "device")]
mod paged_gqa_partitioned_prefill;
#[cfg(feature = "device")]
mod paged_gqa_prefill;
#[cfg(feature = "device")]
mod qwen35_bf16_lm_head;
#[cfg(feature = "device")]
mod qwen35_full_attention_layer;
#[cfg(feature = "device")]
mod qwen35_full_attention_layer_benchmark;
#[cfg(feature = "device")]
mod qwen35_gdn_layer;
#[cfg(feature = "device")]
mod qwen35_gdn_layer_benchmark;
#[cfg(feature = "device")]
mod qwen35_gdn_prepare;
#[cfg(feature = "device")]
mod qwen35_gdn_prepare_benchmark;
#[cfg(feature = "device")]
mod qwen35_gdn_recurrence;
#[cfg(feature = "device")]
mod qwen35_gdn_recurrence_benchmark;
#[cfg(feature = "device")]
mod qwen35_generation;
#[cfg(feature = "device")]
mod qwen35_long_context_kv;
#[cfg(feature = "device")]
mod qwen35_mtp_batch_generation;
#[cfg(feature = "device")]
mod qwen35_mtp_batch_generation_benchmark;
#[cfg(feature = "device")]
mod qwen35_mtp_generation;
#[cfg(feature = "device")]
mod qwen35_mtp_generation_benchmark;
#[cfg(feature = "device")]
mod qwen35_mtp_layer;
#[cfg(feature = "device")]
mod qwen35_mtp_layer_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_attention_output;
#[cfg(feature = "device")]
mod qwen35_nvfp4_attention_output_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_down;
#[cfg(feature = "device")]
mod qwen35_nvfp4_down_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_gdn_input;
#[cfg(feature = "device")]
mod qwen35_nvfp4_gdn_input_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_gdn_output;
#[cfg(feature = "device")]
mod qwen35_nvfp4_gdn_output_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_mlp_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_qkv;
#[cfg(feature = "device")]
mod qwen35_nvfp4_qkv_benchmark;
#[cfg(feature = "device")]
mod qwen35_nvfp4_swiglu;
#[cfg(feature = "device")]
mod qwen35_nvfp4_swiglu_benchmark;
#[cfg(feature = "device")]
mod qwen35_resident_model;
#[cfg(feature = "device")]
mod qwen35_resident_model_benchmark;
#[cfg(feature = "device")]
mod qwen35_resident_mtp;
#[cfg(feature = "device")]
mod qwen35_resident_mtp_benchmark;
#[cfg(feature = "device")]
mod qwen35_text_endpoint;
#[cfg(feature = "device")]
mod qwen35_text_endpoint_benchmark;
#[cfg(feature = "device")]
mod qwen36_attention_output;
#[cfg(feature = "device")]
mod qwen36_attention_output_benchmark;
#[cfg(feature = "device")]
mod qwen36_fp8_qkv;
#[cfg(feature = "device")]
mod qwen36_fp8_qkv_benchmark;
#[cfg(feature = "device")]
mod qwen36_full_attention_layer;
#[cfg(feature = "device")]
mod qwen36_full_attention_layer_benchmark;
#[cfg(feature = "device")]
mod qwen36_gdn_input;
#[cfg(feature = "device")]
mod qwen36_gdn_input_benchmark;
#[cfg(feature = "device")]
mod qwen36_gdn_moe_layer;
#[cfg(feature = "device")]
mod qwen36_gdn_moe_layer_benchmark;
#[cfg(feature = "device")]
mod qwen36_gdn_output;
#[cfg(feature = "device")]
mod qwen36_gdn_output_benchmark;
#[cfg(feature = "device")]
mod qwen36_generation;
#[cfg(feature = "device")]
mod qwen36_long_context_kv;
#[cfg(feature = "device")]
mod qwen36_moe_experts;
#[cfg(feature = "device")]
mod qwen36_moe_experts_benchmark;
#[cfg(feature = "device")]
mod qwen36_moe_router;
#[cfg(feature = "device")]
mod qwen36_moe_router_benchmark;
#[cfg(feature = "device")]
mod qwen36_mtp_bf16_moe;
#[cfg(feature = "device")]
mod qwen36_mtp_layer;
#[cfg(feature = "device")]
mod qwen36_mtp_layer_benchmark;
#[cfg(feature = "device")]
mod qwen36_nvfp4_lm_head;
#[cfg(feature = "device")]
mod qwen36_nvfp4_lm_head_benchmark;
#[cfg(feature = "device")]
mod qwen36_resident_model;
#[cfg(feature = "device")]
mod qwen36_resident_model_benchmark;
#[cfg(feature = "device")]
mod qwen36_text_endpoint;
#[cfg(feature = "device")]
mod qwen36_text_endpoint_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_compact_generation;
#[cfg(feature = "device")]
mod qwen38_flash_next_engram_staging;
#[cfg(feature = "device")]
mod qwen38_flash_next_engram_staging_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_moe_layer;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_moe_layer_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_prepare;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_prepare_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_recurrence;
#[cfg(feature = "device")]
mod qwen38_flash_next_gdn_recurrence_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_generation;
#[cfg(feature = "device")]
mod qwen38_flash_next_generation_benchmark;
mod qwen38_flash_next_golden;
#[cfg(feature = "device")]
mod qwen38_flash_next_hyper_connection;
#[cfg(feature = "device")]
mod qwen38_flash_next_hyper_connection_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_layer_oracle;
#[cfg(feature = "device")]
mod qwen38_flash_next_lm_head;
#[cfg(feature = "device")]
mod qwen38_flash_next_lm_head_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_moe_experts;
#[cfg(feature = "device")]
mod qwen38_flash_next_moe_experts_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_moe_router;
#[cfg(feature = "device")]
mod qwen38_flash_next_moe_router_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_ple;
#[cfg(feature = "device")]
mod qwen38_flash_next_ple_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_projection;
#[cfg(feature = "device")]
mod qwen38_flash_next_projection_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_attention;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_attention_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_moe_layer;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_moe_layer_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_prepare_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_selection;
#[cfg(feature = "device")]
mod qwen38_flash_next_qsa_selection_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_resident_model;
#[cfg(feature = "device")]
mod qwen38_flash_next_resident_model_benchmark;
#[cfg(feature = "device")]
mod qwen38_flash_next_resident_model_oracle;
#[cfg(feature = "device")]
mod qwen38_flash_next_streaming_weight_pool_benchmark;
#[cfg(feature = "device")]
mod resident_batch_generation;
#[cfg(feature = "device")]
mod resident_generation;
#[cfg(feature = "device")]
mod resident_model;
#[cfg(feature = "device")]
mod resident_model_benchmark;
#[cfg(feature = "device")]
mod resident_mtp;
#[cfg(feature = "device")]
mod resident_mtp_batch_generation;
#[cfg(feature = "device")]
mod resident_mtp_batch_generation_benchmark;
#[cfg(feature = "device")]
mod resident_mtp_benchmark;
#[cfg(feature = "device")]
mod resident_mtp_generation;
#[cfg(feature = "device")]
mod resident_mtp_generation_benchmark;
#[cfg(feature = "device")]
mod resident_mtp_sampling;
#[cfg(feature = "device")]
mod resident_mtp_sampling_benchmark;
mod residual_norm;
mod residual_norm_benchmark;
#[cfg(feature = "device")]
mod residual_norm_prefill;
#[cfg(feature = "engine")]
mod speculative_sampling;
#[cfg(feature = "device")]
mod startup_benchmark;
#[cfg(feature = "device")]
mod startup_h2d;
#[cfg(feature = "device")]
mod streaming_weight_pool;
mod target;
#[cfg(feature = "device")]
mod target_mtp_verify;
#[cfg(feature = "device")]
mod target_mtp_verify_benchmark;
#[cfg(feature = "device")]
mod text_endpoint;
#[cfg(feature = "device")]
mod text_endpoint_benchmark;

#[cfg(feature = "device")]
pub use attention_output::{
    AttentionOutputQualification, AttentionOutputQualificationError, qualify_attention_output,
};
#[cfg(feature = "device")]
pub use attention_output_benchmark::benchmark_attention_output;
#[cfg(feature = "device")]
pub use attention_output_prefill::{
    AttentionOutputPrefillQualification, AttentionOutputPrefillQualificationError,
    qualify_attention_output_prefill,
};
#[cfg(feature = "device")]
pub use attention_qk_prepare::{
    AttentionQkPrepareQualification, AttentionQkPrepareQualificationError,
    qualify_attention_qk_prepare, qualify_mtp_bf16_qk_prepare, qualify_qwen35_attention_qk_prepare,
    qualify_qwen35_mtp_bf16_qk_prepare, qualify_qwen36_attention_qk_prepare,
    qualify_qwen36_fp8_attention_qk_prepare, qualify_qwen38_flash_next_qsa_prepare,
};
#[cfg(feature = "device")]
pub use attention_qk_prepare_benchmark::{
    benchmark_attention_qk_prepare, benchmark_mtp_bf16_qk_prepare,
    benchmark_qwen35_attention_qk_prepare, benchmark_qwen36_attention_qk_prepare,
    benchmark_qwen36_fp8_attention_qk_prepare,
};
#[cfg(feature = "device")]
pub use bf16_paged_gqa_benchmark::{
    benchmark_mtp_bf16_paged_gqa, benchmark_qwen35_paged_gqa, benchmark_qwen36_fp8_paged_gqa,
    benchmark_qwen36_paged_gqa,
};
#[cfg(feature = "device")]
pub use dense_fp8_gdn_layer::{
    DenseFp8GdnLayerQualification, DenseFp8GdnLayerQualificationError, qualify_dense_fp8_gdn_layer,
};
#[cfg(feature = "device")]
pub use dense_fp8_gdn_layer_benchmark::benchmark_dense_fp8_gdn_layer;
#[cfg(feature = "device")]
pub use dense_fp8_mlp::{
    DenseFp8MlpQualification, DenseFp8MlpQualificationError, qualify_dense_fp8_mlp,
};
#[cfg(feature = "device")]
pub use dense_fp8_mlp_benchmark::benchmark_dense_fp8_mlp;
pub use device_benchmark::{
    BenchmarkExecution, BenchmarkMeasurement, BenchmarkMemoryKind, BenchmarkMemoryMeasurement,
    BenchmarkPhase, BenchmarkScope, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkMetric,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, DeviceCacheRegime, DeviceEnergyMetric,
    DeviceMemoryMetric, DeviceMemoryReport, DeviceMemorySnapshot, MemoryComparison,
    PrefixCacheRegime,
};
#[cfg(feature = "device")]
pub use fp8_down::{Fp8DownQualification, Fp8DownQualificationError, qualify_fp8_down};
#[cfg(feature = "device")]
pub use fp8_down_benchmark::benchmark_fp8_down;
#[cfg(feature = "device")]
pub use fp8_gdn_input::{
    Fp8GdnInputQualification, Fp8GdnInputQualificationError, qualify_fp8_gdn_input,
};
#[cfg(feature = "device")]
pub use fp8_gdn_input_benchmark::benchmark_fp8_gdn_input;
#[cfg(feature = "device")]
pub use fp8_lm_head::{Fp8LmHeadQualification, Fp8LmHeadQualificationError, qualify_fp8_lm_head};
#[cfg(feature = "device")]
pub use fp8_lm_head_benchmark::benchmark_fp8_lm_head;
#[cfg(any(feature = "device", feature = "sm89"))]
pub use fp8_qkv::{Fp8QkvQualification, Fp8QkvQualificationError, qualify_fp8_qkv};
#[cfg(any(feature = "device", feature = "sm89"))]
pub use fp8_qkv_benchmark::benchmark_fp8_qkv;
#[cfg(feature = "device")]
pub use fp8_swiglu::{Fp8SwiGluQualification, Fp8SwiGluQualificationError, qualify_fp8_swiglu};
#[cfg(feature = "device")]
pub use fp8_swiglu_benchmark::benchmark_fp8_swiglu;
#[cfg(feature = "device")]
pub use full_attention_layer::{
    FullAttentionLayerQualification, FullAttentionLayerQualificationError,
    qualify_full_attention_layer,
};
#[cfg(feature = "device")]
pub use full_attention_layer_benchmark::benchmark_full_attention_layer;
#[cfg(feature = "device")]
pub use gdn_output::{GdnOutputQualification, GdnOutputQualificationError, qualify_gdn_output};
#[cfg(feature = "device")]
pub use gdn_output_benchmark::benchmark_gdn_output;
#[cfg(feature = "device")]
pub use gdn_prepare::{GdnPrepareQualification, GdnPrepareQualificationError, qualify_gdn_prepare};
#[cfg(feature = "device")]
pub use gdn_prepare_benchmark::benchmark_gdn_prepare;
#[cfg(feature = "device")]
pub use gdn_recurrence::{
    GdnRecurrenceQualification, GdnRecurrenceQualificationError, qualify_gdn_recurrence,
};
#[cfg(feature = "device")]
pub use gdn_recurrence_benchmark::benchmark_gdn_recurrence;
#[cfg(feature = "device")]
pub use gdn_state_snapshot::{
    GdnStateSnapshotQualification, GdnStateSnapshotQualificationError, qualify_gdn_state_snapshot,
};
#[cfg(feature = "device")]
pub use long_context_paged_gqa::{
    LongContextPagedGqaQualification, LongContextPagedGqaQualificationError,
    qualify_long_context_paged_gqa,
};
#[cfg(feature = "device")]
pub use long_context_paged_gqa_benchmark::benchmark_long_context_paged_gqa;
#[cfg(feature = "device")]
pub use mtp_bf16_attention_output::{
    MtpBf16AttentionOutputQualification, MtpBf16AttentionOutputQualificationError,
    qualify_mtp_bf16_attention_output, qualify_qwen35_mtp_bf16_attention_output,
    qualify_qwen36_mtp_bf16_attention_output,
};
#[cfg(feature = "device")]
pub use mtp_bf16_attention_output_benchmark::benchmark_mtp_bf16_attention_output;
#[cfg(feature = "device")]
pub use mtp_bf16_fusion::{
    MtpBf16FusionQualification, MtpBf16FusionQualificationError, qualify_mtp_bf16_fusion,
    qualify_qwen35_mtp_bf16_fusion, qualify_qwen36_mtp_bf16_fusion,
};
#[cfg(feature = "device")]
pub use mtp_bf16_fusion_benchmark::benchmark_mtp_bf16_fusion;
#[cfg(feature = "device")]
pub use mtp_bf16_mlp::{
    MtpBf16MlpQualification, MtpBf16MlpQualificationError, qualify_mtp_bf16_mlp,
    qualify_qwen35_mtp_bf16_mlp,
};
#[cfg(feature = "device")]
pub use mtp_bf16_mlp_benchmark::benchmark_mtp_bf16_mlp;
#[cfg(feature = "device")]
pub use mtp_bf16_qkv::{
    MtpBf16QkvQualification, MtpBf16QkvQualificationError, qualify_mtp_bf16_qkv,
    qualify_qwen35_mtp_bf16_qkv, qualify_qwen36_mtp_bf16_qkv,
};
#[cfg(feature = "device")]
pub use mtp_bf16_qkv_benchmark::benchmark_mtp_bf16_qkv;
#[cfg(feature = "device")]
pub use mtp_layer::{MtpLayerQualification, MtpLayerQualificationError, qualify_mtp_layer};
#[cfg(feature = "device")]
pub use mtp_layer_benchmark::benchmark_mtp_layer;
#[cfg(feature = "device")]
pub use mtp_prompt_prime::{
    MtpPromptPrimeQualification, MtpPromptPrimeQualificationError, qualify_mtp_prompt_prime,
};
#[cfg(feature = "device")]
pub use mtp_prompt_prime_benchmark::benchmark_mtp_prompt_prime;
#[cfg(feature = "sm89")]
pub use nvfp4_down::{Nvfp4DownQualification, Nvfp4DownQualificationError, qualify_nvfp4_down};
#[cfg(feature = "sm89")]
pub use nvfp4_down_benchmark::benchmark_nvfp4_down;
#[cfg(feature = "device")]
pub use nvfp4_down_benchmark_sm120::benchmark_nvfp4_down;
#[cfg(feature = "device")]
pub use nvfp4_down_sm120::{
    Nvfp4DownQualification, Nvfp4DownQualificationError, qualify_nvfp4_down,
};
#[cfg(feature = "device")]
pub use nvfp4_mlp::{
    Nvfp4MlpQualification, Nvfp4MlpQualificationError, qualify_nvfp4_mlp, qualify_qwen35_nvfp4_mlp,
};
#[cfg(feature = "device")]
pub use nvfp4_mlp_benchmark::benchmark_nvfp4_mlp;
#[cfg(any(feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu::{
    Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError, qualify_nvfp4_swiglu,
};
#[cfg(any(feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu_benchmark::benchmark_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use nvfp4_swiglu_benchmark_sm120::benchmark_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use nvfp4_swiglu_sm120::{
    Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError, qualify_nvfp4_swiglu,
};
#[cfg(feature = "device")]
pub use paged_gqa::{
    PagedGqaQualification, PagedGqaQualificationError, qualify_mtp_bf16_paged_gqa,
    qualify_paged_gqa, qualify_qwen35_mtp_bf16_paged_gqa, qualify_qwen35_paged_gqa,
    qualify_qwen36_fp8_paged_gqa, qualify_qwen36_paged_gqa,
};
#[cfg(feature = "device")]
pub use paged_gqa_benchmark::benchmark_paged_gqa;
#[cfg(feature = "device")]
pub use paged_gqa_macro_prefill::{
    PagedGqaMacroPrefillQualification, PagedGqaMacroPrefillQualificationError,
    qualify_paged_gqa_macro_prefill,
};
#[cfg(feature = "device")]
pub use paged_gqa_partitioned_prefill::{
    PagedGqaPartitionedPrefillQualification, PagedGqaPartitionedPrefillQualificationError,
    qualify_paged_gqa_partitioned_prefill,
};
#[cfg(feature = "device")]
pub use paged_gqa_prefill::{
    PagedGqaPrefillQualification, PagedGqaPrefillQualificationError, qualify_paged_gqa_prefill,
};
#[cfg(feature = "device")]
pub use qwen35_bf16_lm_head::{
    Qwen35Bf16LmHeadQualification, Qwen35Bf16LmHeadQualificationError, qualify_qwen35_bf16_lm_head,
};
#[cfg(feature = "device")]
pub use qwen35_full_attention_layer::{
    Qwen35FullAttentionLayerQualification, Qwen35FullAttentionLayerQualificationError,
    qualify_qwen35_full_attention_layer,
};
#[cfg(feature = "device")]
pub use qwen35_full_attention_layer_benchmark::benchmark_qwen35_full_attention_layer;
#[cfg(feature = "device")]
pub use qwen35_gdn_layer::{
    Qwen35GdnLayerQualification, Qwen35GdnLayerQualificationError, qualify_qwen35_gdn_layer,
};
#[cfg(feature = "device")]
pub use qwen35_gdn_layer_benchmark::benchmark_qwen35_gdn_layer;
#[cfg(feature = "device")]
pub use qwen35_gdn_prepare::{
    Qwen35GdnPrepareQualification, Qwen35GdnPrepareQualificationError,
    Qwen36GdnPrepareQualification, Qwen36GdnPrepareQualificationError, qualify_qwen35_gdn_prepare,
    qualify_qwen36_gdn_prepare,
};
#[cfg(feature = "device")]
pub use qwen35_gdn_prepare_benchmark::{
    benchmark_qwen35_gdn_prepare, benchmark_qwen36_gdn_prepare,
};
#[cfg(feature = "device")]
pub use qwen35_gdn_recurrence::{
    Qwen35GdnRecurrenceQualification, Qwen35GdnRecurrenceQualificationError,
    Qwen36GdnRecurrenceQualification, Qwen36GdnRecurrenceQualificationError,
    qualify_qwen35_gdn_recurrence, qualify_qwen36_gdn_recurrence,
};
#[cfg(feature = "device")]
pub use qwen35_gdn_recurrence_benchmark::{
    benchmark_qwen35_gdn_recurrence, benchmark_qwen36_gdn_recurrence,
};
#[cfg(feature = "device")]
pub use qwen35_generation::{
    Qwen35CompactGenerationQualification, Qwen35GenerationQualification,
    Qwen35GenerationQualificationError, qualify_qwen35_compact_generation,
    qualify_qwen35_generation,
};
#[cfg(feature = "device")]
pub use qwen35_mtp_batch_generation::{
    Qwen35MtpBatchQualification, Qwen35MtpBatchQualificationError,
    qualify_qwen35_mtp_batch_generation,
};
#[cfg(feature = "device")]
pub use qwen35_mtp_batch_generation_benchmark::benchmark_qwen35_mtp_batch_generation;
#[cfg(feature = "device")]
pub use qwen35_mtp_generation::{
    Qwen35MtpGenerationQualification, Qwen35MtpGenerationQualificationError,
    qualify_qwen35_mtp_generation,
};
#[cfg(feature = "device")]
pub use qwen35_mtp_generation_benchmark::benchmark_qwen35_mtp_generation;
#[cfg(feature = "device")]
pub use qwen35_mtp_layer::{
    Qwen35MtpLayerQualification, Qwen35MtpLayerQualificationError, qualify_qwen35_mtp_layer,
};
#[cfg(feature = "device")]
pub use qwen35_mtp_layer_benchmark::benchmark_qwen35_mtp_layer;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_attention_output::{
    Qwen35Nvfp4AttentionOutputQualification, Qwen35Nvfp4AttentionOutputQualificationError,
    qualify_qwen35_nvfp4_attention_output,
};
#[cfg(feature = "device")]
pub use qwen35_nvfp4_attention_output_benchmark::benchmark_qwen35_nvfp4_attention_output;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_down::qualify_qwen35_nvfp4_down;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_down_benchmark::benchmark_qwen35_nvfp4_down;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_gdn_input::{
    Qwen35Nvfp4GdnInputQualification, Qwen35Nvfp4GdnInputQualificationError,
    qualify_qwen35_nvfp4_gdn_input,
};
#[cfg(feature = "device")]
pub use qwen35_nvfp4_gdn_input_benchmark::benchmark_qwen35_nvfp4_gdn_input;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_gdn_output::{
    Qwen35Nvfp4GdnOutputQualification, Qwen35Nvfp4GdnOutputQualificationError,
    qualify_qwen35_nvfp4_gdn_output,
};
#[cfg(feature = "device")]
pub use qwen35_nvfp4_gdn_output_benchmark::benchmark_qwen35_nvfp4_gdn_output;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_mlp_benchmark::benchmark_qwen35_nvfp4_mlp;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_qkv::{
    Qwen35Nvfp4QkvQualification, Qwen35Nvfp4QkvQualificationError, qualify_qwen35_nvfp4_qkv,
};
#[cfg(feature = "device")]
pub use qwen35_nvfp4_qkv_benchmark::benchmark_qwen35_nvfp4_qkv;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_swiglu::qualify_qwen35_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use qwen35_nvfp4_swiglu_benchmark::benchmark_qwen35_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use qwen35_resident_model::{
    Qwen35ResidentModelQualification, Qwen35ResidentModelQualificationError,
    qualify_qwen35_resident_model,
};
#[cfg(feature = "device")]
pub use qwen35_resident_model_benchmark::benchmark_qwen35_resident_model;
#[cfg(feature = "device")]
pub use qwen35_resident_mtp::{
    Qwen35ResidentMtpQualification, Qwen35ResidentMtpQualificationError,
    qualify_qwen35_resident_mtp,
};
#[cfg(feature = "device")]
pub use qwen35_resident_mtp_benchmark::benchmark_qwen35_resident_mtp;
#[cfg(feature = "device")]
pub use qwen35_text_endpoint::{
    Qwen35TextEndpointQualification, Qwen35TextEndpointQualificationError,
    qualify_qwen35_text_endpoint,
};
#[cfg(feature = "device")]
pub use qwen35_text_endpoint_benchmark::benchmark_qwen35_text_endpoint;
#[cfg(feature = "device")]
pub use qwen36_attention_output::{
    Qwen36AttentionOutputQualification, Qwen36AttentionOutputQualificationError,
    qualify_qwen36_attention_output,
};
#[cfg(feature = "device")]
pub use qwen36_attention_output_benchmark::benchmark_qwen36_attention_output;
#[cfg(feature = "device")]
pub use qwen36_fp8_qkv::{
    Qwen36Fp8QkvQualification, Qwen36Fp8QkvQualificationError, qualify_qwen36_fp8_qkv,
};
#[cfg(feature = "device")]
pub use qwen36_fp8_qkv_benchmark::benchmark_qwen36_fp8_qkv;
#[cfg(feature = "device")]
pub use qwen36_full_attention_layer::{
    Qwen36FullAttentionLayerQualification, Qwen36FullAttentionLayerQualificationError,
    qualify_qwen36_full_attention_layer,
};
#[cfg(feature = "device")]
pub use qwen36_full_attention_layer_benchmark::benchmark_qwen36_full_attention_layer;
#[cfg(feature = "device")]
pub use qwen36_gdn_input::{
    Qwen36GdnInputQualification, Qwen36GdnInputQualificationError, qualify_qwen36_gdn_input,
};
#[cfg(feature = "device")]
pub use qwen36_gdn_input_benchmark::benchmark_qwen36_gdn_input;
#[cfg(feature = "device")]
pub use qwen36_gdn_moe_layer::{
    Qwen36GdnMoeLayerQualification, Qwen36GdnMoeLayerQualificationError,
    qualify_qwen36_gdn_moe_layer,
};
#[cfg(feature = "device")]
pub use qwen36_gdn_moe_layer_benchmark::benchmark_qwen36_gdn_moe_layer;
#[cfg(feature = "device")]
pub use qwen36_gdn_output::{
    Qwen36GdnOutputQualification, Qwen36GdnOutputQualificationError, qualify_qwen36_gdn_output,
};
#[cfg(feature = "device")]
pub use qwen36_gdn_output_benchmark::benchmark_qwen36_gdn_output;
#[cfg(feature = "device")]
pub use qwen36_generation::{
    Qwen36CompactGenerationQualification, Qwen36GenerationQualification,
    Qwen36GenerationQualificationError, qualify_qwen36_compact_generation,
    qualify_qwen36_generation,
};
#[cfg(feature = "device")]
pub use qwen36_moe_experts::{
    Qwen36MoeExpertsQualification, Qwen36MoeExpertsQualificationError, qualify_qwen36_moe_experts,
};
#[cfg(feature = "device")]
pub use qwen36_moe_experts_benchmark::benchmark_qwen36_moe_experts;
#[cfg(feature = "device")]
pub use qwen36_moe_router::{
    Qwen36MoeRouterQualification, Qwen36MoeRouterQualificationError, qualify_qwen36_moe_router,
};
#[cfg(feature = "device")]
pub use qwen36_moe_router_benchmark::benchmark_qwen36_moe_router;
#[cfg(feature = "device")]
pub use qwen36_mtp_bf16_moe::{Qwen36MtpBf16MoeQualification, qualify_qwen36_mtp_bf16_moe};
#[cfg(feature = "device")]
pub use qwen36_mtp_layer::{
    Qwen36MtpLayerQualification, Qwen36MtpLayerQualificationError, qualify_qwen36_mtp_layer,
};
#[cfg(feature = "device")]
pub use qwen36_mtp_layer_benchmark::benchmark_qwen36_mtp_layer;
#[cfg(feature = "device")]
pub use qwen36_nvfp4_lm_head::{
    Qwen36Nvfp4LmHeadQualification, Qwen36Nvfp4LmHeadQualificationError,
    qualify_qwen36_nvfp4_lm_head,
};
#[cfg(feature = "device")]
pub use qwen36_nvfp4_lm_head_benchmark::benchmark_qwen36_nvfp4_lm_head;
#[cfg(feature = "device")]
pub use qwen36_resident_model::{
    Qwen36ResidentModelQualification, Qwen36ResidentModelQualificationError,
    qualify_qwen36_resident_model,
};
#[cfg(feature = "device")]
pub use qwen36_resident_model_benchmark::benchmark_qwen36_resident_model;
#[cfg(feature = "device")]
pub use qwen36_text_endpoint::{
    Qwen36TextEndpointQualification, Qwen36TextEndpointQualificationError,
    qualify_qwen36_text_endpoint,
};
#[cfg(feature = "device")]
pub use qwen36_text_endpoint_benchmark::benchmark_qwen36_text_endpoint;
#[cfg(feature = "device")]
pub use qwen38_flash_next_compact_generation::{
    Qwen38FlashNextCompactGenerationQualification,
    Qwen38FlashNextCompactGenerationQualificationError,
    print_qwen38_flash_next_compact_generation_report,
    qualify_qwen38_flash_next_compact_generation,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_moe_layer::{
    Qwen38FlashNextGdnMoeLayerQualification, Qwen38FlashNextGdnMoeLayerQualificationError,
    qualify_qwen38_flash_next_gdn_moe_layer,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_moe_layer_benchmark::benchmark_qwen38_flash_next_gdn_moe_layer;
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_prepare::{
    Qwen38FlashNextGdnPrepareQualification, Qwen38FlashNextGdnPrepareQualificationError,
    qualify_qwen38_flash_next_gdn_prepare,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_prepare_benchmark::benchmark_qwen38_flash_next_gdn_prepare;
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_recurrence::{
    Qwen38FlashNextGdnRecurrenceQualification, Qwen38FlashNextGdnRecurrenceQualificationError,
    qualify_qwen38_flash_next_gdn_recurrence,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_gdn_recurrence_benchmark::benchmark_qwen38_flash_next_gdn_recurrence;
#[cfg(feature = "device")]
pub use qwen38_flash_next_generation::{
    Qwen38FlashNextBoundaryVerdict, Qwen38FlashNextCaptureVerdict,
    Qwen38FlashNextGenerationQualification, Qwen38FlashNextGenerationQualificationError,
    print_qwen38_flash_next_generation_report, qualify_qwen38_flash_next_generation,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_generation_benchmark::{
    Qwen38FlashNextGenerationBenchmarkReport, Qwen38FlashNextGenerationRouteReport,
    benchmark_qwen38_flash_next_generation, print_qwen38_flash_next_generation_benchmark,
};
pub use qwen38_flash_next_golden::{
    QWEN38_FLASH_NEXT_GOLDEN_BOUNDARIES, QWEN38_FLASH_NEXT_GOLDEN_DIRECTORY,
    QWEN38_FLASH_NEXT_GOLDEN_ENV, QWEN38_FLASH_NEXT_GOLDEN_PROMPTS, Qwen38FlashNextGoldenCapture,
    Qwen38FlashNextGoldenError, Qwen38FlashNextGoldenMeta, Qwen38FlashNextGoldenStep,
    load_qwen38_flash_next_golden_boundary, load_qwen38_flash_next_golden_capture,
    load_qwen38_flash_next_golden_meta, qwen38_flash_next_golden_directory,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_hyper_connection::{
    Qwen38FlashNextHyperConnectionQualification, Qwen38FlashNextHyperConnectionQualificationError,
    qualify_qwen38_flash_next_hyper_connection,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_hyper_connection_benchmark::benchmark_qwen38_flash_next_hyper_connection;
#[cfg(feature = "device")]
pub use qwen38_flash_next_lm_head::{
    Qwen38FlashNextLmHeadQualification, Qwen38FlashNextLmHeadQualificationError,
    qualify_qwen38_flash_next_lm_head,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_lm_head_benchmark::benchmark_qwen38_flash_next_lm_head;
#[cfg(feature = "device")]
pub use qwen38_flash_next_moe_experts::{
    Qwen38FlashNextMoeExpertsQualification, Qwen38FlashNextMoeExpertsQualificationError,
    SlotAssignment, qualify_qwen38_flash_next_moe_experts,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_moe_experts_benchmark::benchmark_qwen38_flash_next_moe_experts;
#[cfg(feature = "device")]
pub use qwen38_flash_next_moe_router::{
    Qwen38FlashNextMoeRouterQualification, Qwen38FlashNextMoeRouterQualificationError,
    qualify_qwen38_flash_next_moe_router,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_moe_router_benchmark::benchmark_qwen38_flash_next_moe_router;
#[cfg(feature = "device")]
pub use qwen38_flash_next_ple::{
    Qwen38FlashNextPleQualification, Qwen38FlashNextPleQualificationError,
    qualify_qwen38_flash_next_ple,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_ple_benchmark::benchmark_qwen38_flash_next_ple;
#[cfg(feature = "device")]
pub use qwen38_flash_next_projection::{
    Qwen38FlashNextProjectionQualification, Qwen38FlashNextProjectionQualificationError,
    qualify_qwen38_flash_next_projections,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_projection_benchmark::benchmark_qwen38_flash_next_projections;
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_attention::{
    Qwen38FlashNextQsaAttentionQualification, Qwen38FlashNextQsaAttentionQualificationError,
    qualify_qwen38_flash_next_qsa_attention,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_attention_benchmark::benchmark_qwen38_flash_next_qsa_attention;
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_moe_layer::{
    Qwen38FlashNextQsaMoeLayerQualification, Qwen38FlashNextQsaMoeLayerQualificationError,
    qualify_qwen38_flash_next_qsa_moe_layer,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_moe_layer_benchmark::benchmark_qwen38_flash_next_qsa_moe_layer;
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_prepare_benchmark::benchmark_qwen38_flash_next_qsa_prepare;
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_selection::{
    Qwen38FlashNextQsaSelectionQualification, Qwen38FlashNextQsaSelectionQualificationError,
    qualify_qwen38_flash_next_qsa_selection,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_qsa_selection_benchmark::benchmark_qwen38_flash_next_qsa_selection;
#[cfg(feature = "device")]
pub use qwen38_flash_next_resident_model::{
    Qwen38FlashNextResidentModelQualification, Qwen38FlashNextResidentModelQualificationError,
    Qwen38FlashNextRouteMeasurement, print_qwen38_flash_next_resident_model_report,
    qualify_qwen38_flash_next_resident_model,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_resident_model_benchmark::{
    Qwen38FlashNextResidentBenchmarkReport, Qwen38FlashNextResidentRouteReport,
    benchmark_qwen38_flash_next_resident_model, print_qwen38_flash_next_resident_benchmark,
};
#[cfg(feature = "device")]
pub use qwen38_flash_next_resident_model_oracle::{
    Qwen38FlashNextModelOracle, Qwen38FlashNextModelOracleError,
    print_qwen38_flash_next_model_oracle, qwen38_flash_next_model_oracle,
};
#[cfg(feature = "device")]
pub use resident_batch_generation::{
    ResidentBatchGenerationQualification, ResidentBatchGenerationQualificationError,
    qualify_resident_batch_generation,
};
#[cfg(feature = "device")]
pub use resident_generation::{
    ResidentGenerationQualification, ResidentGenerationQualificationError,
    qualify_resident_generation,
};
#[cfg(feature = "device")]
pub use resident_model::{
    ResidentModelQualification, ResidentModelQualificationError, qualify_resident_model,
};
#[cfg(feature = "device")]
pub use resident_model_benchmark::{
    ResidentModelProfileManifest, ResidentProfileStage, benchmark_resident_long_context_model,
    benchmark_resident_model, benchmark_resident_prefill, profile_resident_model,
    profile_resident_prefill,
};
#[cfg(feature = "device")]
pub use resident_mtp::{
    ResidentMtpQualification, ResidentMtpQualificationError, qualify_resident_mtp,
};
#[cfg(feature = "device")]
pub use resident_mtp_batch_generation::{
    ResidentMtpBatchGenerationQualification, ResidentMtpBatchGenerationQualificationError,
    qualify_resident_mtp_batch_generation,
};
#[cfg(feature = "device")]
pub use resident_mtp_batch_generation_benchmark::benchmark_resident_mtp_batch_generation;
#[cfg(feature = "device")]
pub use resident_mtp_benchmark::benchmark_resident_mtp;
#[cfg(feature = "device")]
pub use resident_mtp_generation::{
    ResidentMtpGenerationQualification, ResidentMtpGenerationQualificationError,
    qualify_resident_mtp_generation,
};
#[cfg(feature = "device")]
pub use resident_mtp_generation_benchmark::benchmark_resident_mtp_generation;
#[cfg(feature = "device")]
pub use resident_mtp_sampling::{
    ResidentMtpSamplingQualification, ResidentMtpSamplingQualificationError,
    qualify_resident_mtp_sampling,
};
#[cfg(feature = "device")]
pub use resident_mtp_sampling_benchmark::benchmark_resident_mtp_sampling;
pub use residual_norm::{
    ResidualNormQualification, ResidualNormQualificationError, qualify_residual_norm,
};
#[cfg(feature = "device")]
pub use residual_norm::{qualify_qwen35_residual_norm, qualify_qwen36_residual_norm};
pub use residual_norm_benchmark::benchmark_residual_norm;
#[cfg(feature = "device")]
pub use residual_norm_benchmark::{benchmark_qwen35_residual_norm, benchmark_qwen36_residual_norm};
#[cfg(feature = "device")]
pub use residual_norm_prefill::{
    ResidualNormPrefillQualification, ResidualNormPrefillQualificationError,
    qualify_residual_norm_prefill,
};
#[cfg(feature = "engine")]
pub use speculative_sampling::{SpeculativeSamplingQualification, qualify_speculative_sampling};
#[cfg(feature = "device")]
pub use startup_benchmark::run_startup_benchmark_cli;
#[cfg(feature = "device")]
pub use target_mtp_verify::{
    TargetMtpVerifyQualification, TargetMtpVerifyQualificationError, qualify_target_mtp_verify,
};
#[cfg(feature = "device")]
pub use target_mtp_verify_benchmark::benchmark_target_mtp_verify;
#[cfg(feature = "device")]
pub use text_endpoint::{
    TextEndpointQualification, TextEndpointQualificationError, qualify_text_endpoint,
};
#[cfg(feature = "device")]
pub use text_endpoint_benchmark::benchmark_text_endpoint;

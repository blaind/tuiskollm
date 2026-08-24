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
#[cfg(feature = "device")]
#[path = "nvfp4_down_sm120.rs"]
mod nvfp4_down;
#[cfg(feature = "sm89")]
mod nvfp4_down_benchmark;
#[cfg(feature = "device")]
#[path = "nvfp4_down_benchmark_sm120.rs"]
mod nvfp4_down_benchmark;
#[cfg(feature = "device")]
mod nvfp4_mlp;
#[cfg(feature = "device")]
mod nvfp4_mlp_benchmark;
#[cfg(any(feature = "sm89", feature = "sm86"))]
mod nvfp4_swiglu;
#[cfg(feature = "device")]
#[path = "nvfp4_swiglu_sm120.rs"]
mod nvfp4_swiglu;
#[cfg(any(feature = "sm89", feature = "sm86"))]
mod nvfp4_swiglu_benchmark;
#[cfg(feature = "device")]
#[path = "nvfp4_swiglu_benchmark_sm120.rs"]
mod nvfp4_swiglu_benchmark;
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
mod qwen35_paged_gqa_benchmark;
#[cfg(feature = "device")]
mod qwen35_resident_model;
#[cfg(feature = "device")]
mod qwen35_resident_model_benchmark;
#[cfg(feature = "device")]
mod qwen35_text_endpoint;
#[cfg(feature = "device")]
mod qwen35_text_endpoint_benchmark;
#[cfg(feature = "device")]
mod qwen36_gdn_input;
#[cfg(feature = "device")]
mod qwen36_gdn_input_benchmark;
#[cfg(feature = "device")]
mod qwen36_gdn_output;
#[cfg(feature = "device")]
mod qwen36_gdn_output_benchmark;
#[cfg(feature = "device")]
mod qwen36_moe_experts;
#[cfg(feature = "device")]
mod qwen36_moe_experts_benchmark;
#[cfg(feature = "device")]
mod qwen36_moe_router;
#[cfg(feature = "device")]
mod qwen36_moe_router_benchmark;
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
};
#[cfg(feature = "device")]
pub use attention_qk_prepare_benchmark::{
    benchmark_attention_qk_prepare, benchmark_mtp_bf16_qk_prepare,
    benchmark_qwen35_attention_qk_prepare,
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
    qualify_mtp_bf16_attention_output,
};
#[cfg(feature = "device")]
pub use mtp_bf16_attention_output_benchmark::benchmark_mtp_bf16_attention_output;
#[cfg(feature = "device")]
pub use mtp_bf16_fusion::{
    MtpBf16FusionQualification, MtpBf16FusionQualificationError, qualify_mtp_bf16_fusion,
};
#[cfg(feature = "device")]
pub use mtp_bf16_fusion_benchmark::benchmark_mtp_bf16_fusion;
#[cfg(feature = "device")]
pub use mtp_bf16_mlp::{
    MtpBf16MlpQualification, MtpBf16MlpQualificationError, qualify_mtp_bf16_mlp,
};
#[cfg(feature = "device")]
pub use mtp_bf16_mlp_benchmark::benchmark_mtp_bf16_mlp;
#[cfg(feature = "device")]
pub use mtp_bf16_qkv::{
    MtpBf16QkvQualification, MtpBf16QkvQualificationError, qualify_mtp_bf16_qkv,
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
#[cfg(any(feature = "device", feature = "sm89"))]
pub use nvfp4_down::{Nvfp4DownQualification, Nvfp4DownQualificationError, qualify_nvfp4_down};
#[cfg(any(feature = "device", feature = "sm89"))]
pub use nvfp4_down_benchmark::benchmark_nvfp4_down;
#[cfg(feature = "device")]
pub use nvfp4_mlp::{
    Nvfp4MlpQualification, Nvfp4MlpQualificationError, qualify_nvfp4_mlp, qualify_qwen35_nvfp4_mlp,
};
#[cfg(feature = "device")]
pub use nvfp4_mlp_benchmark::benchmark_nvfp4_mlp;
#[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu::{
    Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError, qualify_nvfp4_swiglu,
};
#[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu_benchmark::benchmark_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use paged_gqa::{
    PagedGqaQualification, PagedGqaQualificationError, qualify_mtp_bf16_paged_gqa,
    qualify_paged_gqa, qualify_qwen35_paged_gqa,
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
    Qwen35GenerationQualification, Qwen35GenerationQualificationError, qualify_qwen35_generation,
};
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
pub use qwen35_paged_gqa_benchmark::{benchmark_mtp_bf16_paged_gqa, benchmark_qwen35_paged_gqa};
#[cfg(feature = "device")]
pub use qwen35_resident_model::{
    Qwen35ResidentModelQualification, Qwen35ResidentModelQualificationError,
    qualify_qwen35_resident_model,
};
#[cfg(feature = "device")]
pub use qwen35_resident_model_benchmark::benchmark_qwen35_resident_model;
#[cfg(feature = "device")]
pub use qwen35_text_endpoint::{
    Qwen35TextEndpointQualification, Qwen35TextEndpointQualificationError,
    qualify_qwen35_text_endpoint,
};
#[cfg(feature = "device")]
pub use qwen35_text_endpoint_benchmark::benchmark_qwen35_text_endpoint;
#[cfg(feature = "device")]
pub use qwen36_gdn_input::{
    Qwen36GdnInputQualification, Qwen36GdnInputQualificationError, qualify_qwen36_gdn_input,
};
#[cfg(feature = "device")]
pub use qwen36_gdn_input_benchmark::benchmark_qwen36_gdn_input;
#[cfg(feature = "device")]
pub use qwen36_gdn_output::{
    Qwen36GdnOutputQualification, Qwen36GdnOutputQualificationError, qualify_qwen36_gdn_output,
};
#[cfg(feature = "device")]
pub use qwen36_gdn_output_benchmark::benchmark_qwen36_gdn_output;
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

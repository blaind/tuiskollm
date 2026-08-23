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
mod long_context_paged_gqa;
#[cfg(feature = "device")]
mod long_context_paged_gqa_benchmark;
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
mod paged_gqa_partitioned_prefill;
#[cfg(feature = "device")]
mod paged_gqa_prefill;
#[cfg(feature = "device")]
mod resident_batch_generation;
#[cfg(feature = "device")]
mod resident_generation;
#[cfg(feature = "device")]
mod resident_model;
#[cfg(feature = "device")]
mod resident_model_benchmark;
mod residual_norm;
mod residual_norm_benchmark;
#[cfg(feature = "device")]
mod startup_benchmark;
mod target;
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
pub use attention_qk_prepare::{
    AttentionQkPrepareQualification, AttentionQkPrepareQualificationError,
    qualify_attention_qk_prepare,
};
#[cfg(feature = "device")]
pub use attention_qk_prepare_benchmark::benchmark_attention_qk_prepare;
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
pub use long_context_paged_gqa::{
    LongContextPagedGqaQualification, LongContextPagedGqaQualificationError,
    qualify_long_context_paged_gqa,
};
#[cfg(feature = "device")]
pub use long_context_paged_gqa_benchmark::benchmark_long_context_paged_gqa;
#[cfg(any(feature = "device", feature = "sm89"))]
pub use nvfp4_down::{Nvfp4DownQualification, Nvfp4DownQualificationError, qualify_nvfp4_down};
#[cfg(any(feature = "device", feature = "sm89"))]
pub use nvfp4_down_benchmark::benchmark_nvfp4_down;
#[cfg(feature = "device")]
pub use nvfp4_mlp::{Nvfp4MlpQualification, Nvfp4MlpQualificationError, qualify_nvfp4_mlp};
#[cfg(feature = "device")]
pub use nvfp4_mlp_benchmark::benchmark_nvfp4_mlp;
#[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu::{
    Nvfp4SwiGluQualification, Nvfp4SwiGluQualificationError, qualify_nvfp4_swiglu,
};
#[cfg(any(feature = "device", feature = "sm89", feature = "sm86"))]
pub use nvfp4_swiglu_benchmark::benchmark_nvfp4_swiglu;
#[cfg(feature = "device")]
pub use paged_gqa::{PagedGqaQualification, PagedGqaQualificationError, qualify_paged_gqa};
#[cfg(feature = "device")]
pub use paged_gqa_benchmark::benchmark_paged_gqa;
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
    benchmark_resident_model, profile_resident_model,
};
pub use residual_norm::{
    ResidualNormQualification, ResidualNormQualificationError, qualify_residual_norm,
};
pub use residual_norm_benchmark::benchmark_residual_norm;
#[cfg(feature = "device")]
pub use startup_benchmark::run_startup_benchmark_cli;
#[cfg(feature = "device")]
pub use text_endpoint::{
    TextEndpointQualification, TextEndpointQualificationError, qualify_text_endpoint,
};
#[cfg(feature = "device")]
pub use text_endpoint_benchmark::benchmark_text_endpoint;

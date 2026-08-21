//! Independent operator qualification for the exact SM120 target.

#![cfg(feature = "device")]

mod device_benchmark;
mod fp8_gdn_input;
mod fp8_gdn_input_benchmark;
mod fp8_lm_head;
mod fp8_lm_head_benchmark;
mod fp8_projection_oracle;
mod fp8_qkv;
mod fp8_qkv_benchmark;
mod fp8_swiglu;
mod fp8_swiglu_benchmark;
mod residual_norm;
mod residual_norm_benchmark;
mod text_endpoint;
mod text_endpoint_benchmark;

pub use device_benchmark::{
    BenchmarkExecution, BenchmarkMeasurement, BenchmarkMemoryKind, BenchmarkMemoryMeasurement,
    BenchmarkPhase, BenchmarkScope, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkMetric,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, DeviceCacheRegime, DeviceEnergyMetric,
    DeviceMemoryMetric, DeviceMemoryReport, DeviceMemorySnapshot, MemoryComparison,
    PrefixCacheRegime,
};
pub use fp8_gdn_input::{
    Fp8GdnInputQualification, Fp8GdnInputQualificationError, qualify_fp8_gdn_input,
};
pub use fp8_gdn_input_benchmark::benchmark_fp8_gdn_input;
pub use fp8_lm_head::{Fp8LmHeadQualification, Fp8LmHeadQualificationError, qualify_fp8_lm_head};
pub use fp8_lm_head_benchmark::benchmark_fp8_lm_head;
pub use fp8_qkv::{Fp8QkvQualification, Fp8QkvQualificationError, qualify_fp8_qkv};
pub use fp8_qkv_benchmark::benchmark_fp8_qkv;
pub use fp8_swiglu::{Fp8SwiGluQualification, Fp8SwiGluQualificationError, qualify_fp8_swiglu};
pub use fp8_swiglu_benchmark::benchmark_fp8_swiglu;
pub use residual_norm::{
    ResidualNormQualification, ResidualNormQualificationError, qualify_residual_norm,
};
pub use residual_norm_benchmark::benchmark_residual_norm;
pub use text_endpoint::{
    TextEndpointQualification, TextEndpointQualificationError, qualify_text_endpoint,
};
pub use text_endpoint_benchmark::benchmark_text_endpoint;

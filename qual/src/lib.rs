//! Independent operator qualification for the exact SM120 target.

mod device_benchmark;
mod residual_norm;
mod residual_norm_benchmark;

pub use device_benchmark::{
    BenchmarkExecution, BenchmarkMeasurement, BenchmarkMemoryKind, BenchmarkMemoryMeasurement,
    BenchmarkPhase, BenchmarkScope, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkMetric,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, DeviceCacheRegime, DeviceEnergyMetric,
    DeviceMemoryMetric, DeviceMemoryReport, DeviceMemorySnapshot, MemoryComparison,
    PrefixCacheRegime,
};
pub use residual_norm::{
    ResidualNormQualification, ResidualNormQualificationError, qualify_residual_norm,
};
pub use residual_norm_benchmark::benchmark_residual_norm;

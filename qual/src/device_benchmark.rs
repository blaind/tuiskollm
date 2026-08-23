//! Shared controls and reports for exclusive device measurements.

use crate::target::{
    CLOCK_LOCK_COMMAND, EXPECTED_COMPUTE_CAPABILITY_TEXT, EXPECTED_DEVICE_NAME,
    MAX_MEMORY_CLOCK_SPREAD_MHZ, MAX_SM_CLOCK_SPREAD_MHZ,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::{Command, Output};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tuisko_gpu::{CudaGraph, CudaStream, GpuTimer};

const DEVICE_INDEX: &str = "0";
const MAX_IDLE_MEMORY_MIB: u32 = 1_024;
const MIN_TELEMETRY_SAMPLES: usize = 3;
const LOADED_CLOCK_PROBE_DURATION: Duration = Duration::from_secs(2);
const MAX_LOADED_CLOCK_PROBE_REPLAYS: u64 = 1_000_000;
const DIAGNOSTIC_CLOCK_ENV: &str = "TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT";
const CLOCK_RESET_COMMAND: &str =
    "sudo nvidia-smi -i 0 --reset-gpu-clocks && sudo nvidia-smi -i 0 --reset-memory-clocks";

/// Controls the statistical size of one device benchmark run.
#[derive(Clone, Copy, Debug)]
pub struct DeviceBenchmarkOptions {
    /// Number of rotated samples collected for every route.
    pub samples: usize,
    /// Operations bracketed by each paired timing interval.
    pub launches_per_sample: u64,
    /// Production-graph replays used to establish warmed state before timing.
    pub warmup_launches: u64,
    /// Optional exact decode batch retained for a diagnostic subset run.
    pub batch_size: Option<u32>,
    /// Dedicated power-sampling window per route, when requested.
    pub energy_seconds: Option<f64>,
}

impl DeviceBenchmarkOptions {
    /// Defaults for microsecond-scale operator graphs.
    pub const fn short_graph() -> Self {
        Self {
            samples: 40,
            launches_per_sample: 256,
            warmup_launches: 1_024,
            batch_size: None,
            energy_seconds: None,
        }
    }

    /// Defaults for graph boundaries whose single replay already takes hundreds of microseconds.
    pub const fn long_graph() -> Self {
        Self {
            samples: 40,
            // A 32-replay interval is several milliseconds for the admitted composed owners,
            // amortizing the timer without turning one sample into a long thermal phase.
            launches_per_sample: 32,
            warmup_launches: 128,
            batch_size: None,
            energy_seconds: None,
        }
    }

    /// Defaults for the complete resident model graph.
    pub const fn resident_model() -> Self {
        Self {
            samples: 40,
            // One 17+ ms graph already exceeds CUDA-event resolution by four orders of magnitude.
            launches_per_sample: 1,
            warmup_launches: 16,
            batch_size: None,
            energy_seconds: None,
        }
    }
}

impl Default for DeviceBenchmarkOptions {
    fn default() -> Self {
        Self::short_graph()
    }
}

/// Clock used for one reported performance metric.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMeasurement {
    /// Rust time spent calling the asynchronous graph-launch API.
    HostSubmit,
    /// Rust time from submission through device completion.
    HostCompletion,
    /// CUDA-event time around repeated production graph replays.
    DeviceGraph,
    /// CUDA-event time per operation within one repeated-operation graph replay.
    DevicePath,
}

impl BenchmarkMeasurement {
    /// Stable spelling used in tables and baseline keys.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostSubmit => "host_submit",
            Self::HostCompletion => "host_completion",
            Self::DeviceGraph => "device_graph",
            Self::DevicePath => "device_path",
        }
    }

    const fn is_device(self) -> bool {
        matches!(self, Self::DeviceGraph | Self::DevicePath)
    }
}

/// Production boundary exercised by a benchmark case.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkScope {
    /// One closed operator.
    Operator,
    /// A model input or output endpoint.
    Endpoint,
    /// One composed model layer.
    Layer,
    /// The resident model graph.
    Model,
    /// The externally visible serving path.
    Server,
}

/// Inference phase exercised by a benchmark case.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPhase {
    /// Model loading, materialization, upload, or prewarm.
    Startup,
    /// Prompt processing.
    Prefill,
    /// Ordinary autoregressive decoding.
    Decode,
    /// Speculative draft and verification work.
    Mtp,
    /// A complete externally observed request.
    Request,
}

/// Device-cache state established before measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCacheRegime {
    /// Cache state is deliberately displaced before the case.
    Cold,
    /// The case is warmed before timing.
    Warm,
    /// Device cache state does not apply to the measured boundary.
    NotApplicable,
}

/// Prefix-cache state of a serving workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixCacheRegime {
    /// No reusable prefix is present.
    Miss,
    /// A strict prefix is reusable.
    PartialHit,
    /// The complete eligible prefix is reusable.
    FullHit,
}

/// Submission mechanism used by a benchmark case.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkExecution {
    /// Direct eager launches.
    Eager,
    /// Replay of a captured CUDA Graph.
    CudaGraph,
    /// The complete server boundary.
    Server,
}

/// Exact workload dimensions that make a timing comparable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BenchmarkWorkload {
    /// Production boundary under measurement.
    pub scope: BenchmarkScope,
    /// Inference phase under measurement.
    pub phase: BenchmarkPhase,
    /// Active compiled batch, when applicable.
    pub batch_size: Option<u32>,
    /// Concurrent externally active requests, when applicable.
    pub concurrency: Option<u32>,
    /// Rows or tokens processed by the measured operation.
    pub active_tokens: Option<u64>,
    /// Complete rendered prompt length, when applicable.
    pub prompt_tokens: Option<u64>,
    /// Context length visible to the measured operation.
    pub context_tokens: Option<u64>,
    /// Requested generated-token count, when applicable.
    pub output_tokens: Option<u64>,
    /// Device-cache state established before measurement.
    pub device_cache: DeviceCacheRegime,
    /// Prefix-cache state, when a serving cache is involved.
    pub prefix_cache: Option<PrefixCacheRegime>,
    /// Submission mechanism used by the case.
    pub execution: BenchmarkExecution,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
impl BenchmarkWorkload {
    pub(crate) fn warm_operator_decode(batch_size: u32) -> Self {
        Self {
            scope: BenchmarkScope::Operator,
            phase: BenchmarkPhase::Decode,
            batch_size: Some(batch_size),
            concurrency: None,
            active_tokens: Some(u64::from(batch_size)),
            prompt_tokens: None,
            context_tokens: None,
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_operator_prefill(active_tokens: u64) -> Self {
        Self {
            scope: BenchmarkScope::Operator,
            phase: BenchmarkPhase::Prefill,
            batch_size: None,
            concurrency: None,
            active_tokens: Some(active_tokens),
            prompt_tokens: Some(active_tokens),
            context_tokens: Some(active_tokens),
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_operator_mtp(active_tokens: u64) -> Self {
        Self {
            scope: BenchmarkScope::Operator,
            phase: BenchmarkPhase::Mtp,
            batch_size: None,
            concurrency: None,
            active_tokens: Some(active_tokens),
            prompt_tokens: None,
            context_tokens: None,
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_endpoint_decode(batch_size: u32) -> Self {
        Self {
            scope: BenchmarkScope::Endpoint,
            phase: BenchmarkPhase::Decode,
            batch_size: Some(batch_size),
            concurrency: None,
            active_tokens: Some(u64::from(batch_size)),
            prompt_tokens: None,
            context_tokens: None,
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_layer_decode(batch_size: u32) -> Self {
        Self {
            scope: BenchmarkScope::Layer,
            phase: BenchmarkPhase::Decode,
            batch_size: Some(batch_size),
            concurrency: None,
            active_tokens: Some(u64::from(batch_size)),
            prompt_tokens: None,
            context_tokens: None,
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_attention_layer_decode(batch_size: u32, context_tokens: u64) -> Self {
        Self {
            scope: BenchmarkScope::Layer,
            phase: BenchmarkPhase::Decode,
            batch_size: Some(batch_size),
            concurrency: None,
            active_tokens: Some(u64::from(batch_size)),
            prompt_tokens: None,
            context_tokens: Some(context_tokens),
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }

    pub(crate) fn warm_model_decode(batch_size: u32, context_tokens: u64) -> Self {
        Self {
            scope: BenchmarkScope::Model,
            phase: BenchmarkPhase::Decode,
            batch_size: Some(batch_size),
            concurrency: None,
            active_tokens: Some(u64::from(batch_size)),
            prompt_tokens: None,
            context_tokens: Some(context_tokens),
            output_tokens: None,
            device_cache: DeviceCacheRegime::Warm,
            prefix_cache: None,
            execution: BenchmarkExecution::CudaGraph,
        }
    }
}

/// Kind of resident memory attributed by a benchmark owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMemoryKind {
    /// Model weights.
    Weights,
    /// Key/value cache storage.
    KvCache,
    /// Address-stable activation or scratch storage.
    Workspace,
    /// Captured graph storage with a known size.
    Graph,
    /// CUDA/runtime storage with a known size.
    Runtime,
    /// Storage that does not fit the categories above.
    Other,
}

/// Direction used when comparing a memory quantity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryComparison {
    /// Smaller byte counts are preferable.
    AtMost,
    /// Larger byte counts are preferable.
    AtLeast,
}

/// Source and meaning of one memory quantity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMemoryMeasurement {
    /// Bytes owned and attributed by production owners.
    Owned,
    /// Whole-device usage added between preflight and warmed setup.
    SetupDeviceDelta,
    /// Highest whole-device usage during timed work, relative to preflight.
    TimedPeakDeviceDelta,
    /// Highest whole-device usage during timed work above the warmed setup.
    TimedGrowthAfterWarmup,
    /// Lowest whole-device free-memory headroom during timed work.
    MinimumDeviceHeadroom,
    /// Peak resident host memory reported by the process.
    ProcessPeakRss,
    /// Framebuffer capacity reserved by the driver according to NVML.
    DeviceReserved,
    /// Setup delta not explained by attributed production owners.
    UnattributedSetupDelta,
}

impl BenchmarkMemoryMeasurement {
    /// Stable spelling used in tables and baseline keys.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Owned => "owned",
            Self::SetupDeviceDelta => "setup_device_delta",
            Self::TimedPeakDeviceDelta => "timed_peak_device_delta",
            Self::TimedGrowthAfterWarmup => "timed_growth_after_warmup",
            Self::MinimumDeviceHeadroom => "minimum_device_headroom",
            Self::ProcessPeakRss => "process_peak_rss",
            Self::DeviceReserved => "device_reserved",
            Self::UnattributedSetupDelta => "unattributed_setup_delta",
        }
    }
}

/// One byte-count suitable for reporting and baseline comparison.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceMemoryMetric {
    /// Stable owner or summary name.
    pub name: String,
    /// Source and meaning of the byte count.
    pub measurement: BenchmarkMemoryMeasurement,
    /// Resident-memory class for an owned quantity.
    pub kind: Option<BenchmarkMemoryKind>,
    /// How the owner scales, such as `max_batch=8`.
    pub scaling: Option<String>,
    /// Measured or accounted byte count.
    pub bytes: u64,
    /// Preferred comparison direction.
    pub comparison: MemoryComparison,
}

/// Memory state captured at one benchmark phase boundary.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceMemorySnapshot {
    /// Stable phase name.
    pub phase: String,
    /// Whole-device memory used according to NVML.
    pub device_used_bytes: u64,
    /// Whole-device free memory according to NVML.
    pub device_free_bytes: u64,
    /// Whole-device driver-reserved memory according to NVML.
    pub device_reserved_bytes: u64,
    /// Current process resident host memory.
    pub process_rss_bytes: u64,
    /// Peak process resident host memory.
    pub process_peak_rss_bytes: u64,
}

/// Capacity evidence captured around one benchmark session.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceMemoryReport {
    /// Physical framebuffer capacity reported by NVML.
    pub device_total_bytes: u64,
    /// Phase-boundary memory snapshots.
    pub snapshots: Vec<DeviceMemorySnapshot>,
    /// Comparable attributed and observed memory quantities.
    pub metrics: Vec<DeviceMemoryMetric>,
}

/// One named timing from an exact performance route.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceBenchmarkMetric {
    /// Stable slash-separated operator route.
    pub route: &'static str,
    /// Exact compiled shape, such as `B=1`.
    pub shape: String,
    /// Workload dimensions required for a valid comparison.
    pub workload: BenchmarkWorkload,
    /// Timer and execution boundary used for the sample.
    pub measurement: BenchmarkMeasurement,
    /// Median time for one operation.
    pub median_microseconds: f64,
    /// Tenth-percentile time for one operation.
    pub p10_microseconds: f64,
    /// Ninetieth-percentile time for one operation.
    pub p90_microseconds: f64,
    /// Operations represented by each raw timing interval.
    pub operations_per_interval: u64,
    /// Minimum logical bytes read and written by one operation.
    pub logical_bytes_per_operation: u64,
    /// Logical throughput for a device timing.
    pub logical_gib_per_second: Option<f64>,
}

/// Estimated whole-board energy for one sustained exact route.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceEnergyMetric {
    /// Stable slash-separated operator route.
    pub route: &'static str,
    /// Exact compiled shape, such as `B=1`.
    pub shape: String,
    /// Workload dimensions required for a valid comparison.
    pub workload: BenchmarkWorkload,
    /// Unit used for normalization, such as `token`.
    pub unit: &'static str,
    /// Graph operations executed in the sustained window.
    pub operations: u64,
    /// Logical units completed by every operation.
    pub units_per_operation: u64,
    /// CUDA-event duration of the sustained window.
    pub device_seconds: f64,
    /// Arithmetic mean of sampled instantaneous board power.
    pub average_board_watts: f64,
    /// Arithmetic mean of sampled board power while the warmed device is idle.
    pub idle_board_watts: f64,
    /// Workload power above the sampled idle-board baseline.
    pub dynamic_board_watts: f64,
    /// Estimated board energy divided by completed logical units.
    pub estimated_board_joules_per_unit: f64,
    /// Estimated board energy above idle divided by completed logical units.
    pub estimated_dynamic_joules_per_unit: f64,
    /// Completed logical units per estimated whole-board joule.
    pub estimated_units_per_board_joule: f64,
    /// Power samples captured during the sustained window.
    pub telemetry_samples: usize,
}

/// Machine-readable result of one exclusive device benchmark run.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceBenchmarkReport {
    /// Report schema revision.
    pub schema_version: u32,
    /// Registered exact-target suite.
    pub suite: &'static str,
    /// Scope of the result; the first suite is a performance-sensitive leaf.
    pub classification: &'static str,
    /// CUDA device name reported by `nvidia-smi`.
    pub device: String,
    /// Stable physical GPU identity reported by `nvidia-smi`.
    pub device_uuid: String,
    /// NVIDIA driver version used for the run.
    pub driver_version: String,
    /// Device index used by both CUDA and `nvidia-smi`.
    pub device_index: u32,
    /// Compute capability admitted by the runtime.
    pub compute_capability: String,
    /// Whether clock spread was enforced or merely recorded for a diagnostic run.
    pub clock_policy: &'static str,
    /// Hash of the benchmark executable.
    pub binary_sha256: String,
    /// Hash of the complete checked generator/resource baseline.
    pub generator_baseline_sha256: String,
    /// Lowest SM clock sampled during timed work.
    pub sm_clock_min_mhz: u32,
    /// Median SM clock sampled during timed work.
    pub sm_clock_median_mhz: u32,
    /// Highest SM clock sampled during timed work.
    pub sm_clock_max_mhz: u32,
    /// Lowest memory clock sampled during timed work.
    pub memory_clock_min_mhz: u32,
    /// Median memory clock sampled during timed work.
    pub memory_clock_median_mhz: u32,
    /// Highest memory clock sampled during timed work.
    pub memory_clock_max_mhz: u32,
    /// Lowest GPU temperature sampled during timed work.
    pub temperature_min_celsius: u32,
    /// Highest GPU temperature sampled during timed work.
    pub temperature_max_celsius: u32,
    /// Lowest instantaneous board power sampled during timed work.
    pub power_min_watts: f64,
    /// Arithmetic mean of instantaneous board power samples.
    pub power_mean_watts: f64,
    /// Median board power sampled during timed work.
    pub power_median_watts: f64,
    /// Highest board power sampled during timed work.
    pub power_max_watts: f64,
    /// Number of in-window telemetry samples.
    pub telemetry_samples: usize,
    /// Number of rotated samples collected for every metric.
    pub samples: usize,
    /// Leaf graph replays included in each paired timing interval.
    pub launches_per_sample: u64,
    /// Production-graph replays used to establish warmed state before timing.
    pub warmup_launches: u64,
    /// Whether the report covers the complete suite inventory or a diagnostic subset.
    pub case_policy: &'static str,
    /// Exact decode batch selected for a diagnostic subset report.
    pub selected_batch_size: Option<u32>,
    /// Boundaries represented by the four timing kinds.
    pub timing_scope: &'static str,
    /// Telemetry field and physical scope used for power estimates.
    pub power_scope: &'static str,
    /// Named timings sorted in execution-path order.
    pub metrics: Vec<DeviceBenchmarkMetric>,
    /// Optional sustained whole-board energy estimates.
    pub energy_metrics: Vec<DeviceEnergyMetric>,
    /// Device and host capacity evidence for the complete session.
    pub memory: DeviceMemoryReport,
}

pub(crate) struct BenchmarkReportSpec {
    pub(crate) suite: &'static str,
    pub(crate) classification: &'static str,
    pub(crate) timing_scope: &'static str,
}

/// Failure to establish or measure a comparable device benchmark.
#[derive(Debug, thiserror::Error)]
pub enum DeviceBenchmarkError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] tuisko_gpu::GpuError),

    /// Resident engine ownership or execution failure.
    #[cfg(feature = "engine")]
    #[error(transparent)]
    Engine(#[from] tuisko_engine::EngineError),

    /// Snapshot admission or source binding failure.
    #[error(transparent)]
    Checkpoint(#[from] tuisko_model::CheckpointError),

    /// Host filesystem or process failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The machine was not in the exact comparable state.
    #[error("device benchmark precondition failed: {0}")]
    Precondition(String),
}

pub(crate) struct DeviceIdentity {
    pub(crate) name: String,
    pub(crate) uuid: String,
    pub(crate) driver_version: String,
}

pub(crate) struct TelemetryEvidence {
    pub(crate) clock_comparable: bool,
    pub(crate) sm_minimum_mhz: u32,
    pub(crate) sm_median_mhz: u32,
    pub(crate) sm_maximum_mhz: u32,
    pub(crate) memory_minimum_mhz: u32,
    pub(crate) memory_median_mhz: u32,
    pub(crate) memory_maximum_mhz: u32,
    pub(crate) temperature_minimum_celsius: u32,
    pub(crate) temperature_maximum_celsius: u32,
    pub(crate) power_minimum_watts: f64,
    pub(crate) power_mean_watts: f64,
    pub(crate) power_median_watts: f64,
    pub(crate) power_maximum_watts: f64,
    pub(crate) device_memory_maximum_used_bytes: u64,
    pub(crate) device_memory_minimum_free_bytes: u64,
    pub(crate) samples: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct RepeatedGraph<'a> {
    graph: &'a CudaGraph,
    operations: u64,
}

pub(crate) struct OperationAccounting {
    logical_bytes: usize,
    units_per_operation: u64,
    unit: &'static str,
}

impl OperationAccounting {
    pub(crate) fn new(logical_bytes: usize, units_per_operation: u64, unit: &'static str) -> Self {
        Self {
            logical_bytes,
            units_per_operation,
            unit,
        }
    }
}

impl<'a> RepeatedGraph<'a> {
    pub(crate) fn new(graph: &'a CudaGraph, operations: u64) -> Self {
        Self { graph, operations }
    }
}

#[derive(Clone)]
pub(crate) struct ExactDeviceCase<'a> {
    route: &'static str,
    shape: String,
    workload: BenchmarkWorkload,
    logical_bytes: usize,
    units_per_operation: u64,
    unit: &'static str,
    leaf_graph: &'a CudaGraph,
    preparation_graph: Option<&'a CudaGraph>,
    repeated: Option<RepeatedGraph<'a>>,
}

impl<'a> ExactDeviceCase<'a> {
    pub(crate) fn new(
        route: &'static str,
        shape: String,
        workload: BenchmarkWorkload,
        accounting: OperationAccounting,
        leaf_graph: &'a CudaGraph,
        repeated: Option<RepeatedGraph<'a>>,
    ) -> Self {
        Self {
            route,
            shape,
            workload,
            logical_bytes: accounting.logical_bytes,
            units_per_operation: accounting.units_per_operation,
            unit: accounting.unit,
            leaf_graph,
            preparation_graph: None,
            repeated,
        }
    }

    pub(crate) fn with_preparation(mut self, graph: &'a CudaGraph) -> Self {
        self.preparation_graph = Some(graph);
        self
    }
}

#[derive(Default)]
struct CaseSamples {
    host_submit: Vec<f64>,
    host_completion: Vec<f64>,
    device_graph: Vec<f64>,
    device_path: Vec<f64>,
}

#[derive(Clone, Copy)]
enum MeasurementTask {
    Leaf(usize),
    Repeated(usize),
}

#[derive(Clone, Copy)]
struct TelemetrySample {
    sm_clock_mhz: u32,
    memory_clock_mhz: u32,
    temperature_celsius: u32,
    power_watts: f64,
    device_memory_used_mib: u32,
    device_memory_free_mib: u32,
}

pub(crate) struct TelemetrySampler {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<Result<Vec<TelemetrySample>, String>>>,
}

impl TelemetrySampler {
    pub(crate) fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let mut samples = Vec::new();
            loop {
                thread::sleep(Duration::from_millis(10));
                if thread_stop.load(Ordering::Acquire) {
                    break;
                }
                samples.push(query_telemetry()?);
            }

            Ok(samples)
        });

        Self {
            stop,
            thread: Some(thread),
        }
    }

    fn samples(mut self) -> Result<Vec<TelemetrySample>, DeviceBenchmarkError> {
        self.stop.store(true, Ordering::Release);
        self.thread
            .take()
            .ok_or_else(|| {
                DeviceBenchmarkError::Precondition(
                    "telemetry sampler thread was already consumed".to_string(),
                )
            })?
            .join()
            .map_err(|_| {
                DeviceBenchmarkError::Precondition("telemetry sampler thread panicked".to_string())
            })?
            .map_err(DeviceBenchmarkError::Precondition)
    }

    pub(crate) fn finish(self) -> Result<TelemetryEvidence, DeviceBenchmarkError> {
        telemetry_evidence(self.samples()?, diagnostic_clock_drift_allowed())
    }

    fn finish_preserving_clock_drift(self) -> Result<TelemetryEvidence, DeviceBenchmarkError> {
        telemetry_evidence(self.samples()?, true)
    }
}

impl Drop for TelemetrySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct DeviceSnapshot {
    identity: DeviceIdentity,
    utilization_percent: u32,
    memory_used_mib: u32,
    memory_free_mib: u32,
    memory_reserved_mib: u32,
    memory_total_mib: u32,
}

pub(crate) struct DevicePreflight {
    pub(crate) identity: DeviceIdentity,
    memory: DeviceSnapshot,
}

pub(crate) struct MemoryRecorder {
    device_uuid: String,
    preflight_used_bytes: u64,
    total_bytes: u64,
    snapshots: Vec<DeviceMemorySnapshot>,
    owned: Vec<DeviceMemoryMetric>,
}

impl MemoryRecorder {
    pub(crate) fn new(preflight: &DevicePreflight) -> Result<Self, DeviceBenchmarkError> {
        let mut recorder = Self {
            device_uuid: preflight.identity.uuid.clone(),
            preflight_used_bytes: mib_to_bytes(preflight.memory.memory_used_mib),
            total_bytes: mib_to_bytes(preflight.memory.memory_total_mib),
            snapshots: Vec::new(),
            owned: Vec::new(),
        };
        recorder.capture_snapshot("before_context", &preflight.memory)?;

        Ok(recorder)
    }

    pub(crate) fn register_owned(
        &mut self,
        name: &'static str,
        kind: BenchmarkMemoryKind,
        bytes: usize,
        scaling: &'static str,
    ) -> Result<(), DeviceBenchmarkError> {
        let bytes = u64::try_from(bytes).map_err(|_| {
            DeviceBenchmarkError::Precondition(format!(
                "memory owner `{name}` exceeds the report width"
            ))
        })?;
        if self.owned.iter().any(|metric| metric.name == name) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "memory owner `{name}` is registered more than once"
            )));
        }
        self.owned.push(DeviceMemoryMetric {
            name: name.to_string(),
            measurement: BenchmarkMemoryMeasurement::Owned,
            kind: Some(kind),
            scaling: Some(scaling.to_string()),
            bytes,
            comparison: MemoryComparison::AtMost,
        });

        Ok(())
    }

    pub(crate) fn capture(&mut self, phase: &'static str) -> Result<(), DeviceBenchmarkError> {
        let snapshot = device_snapshot()?;
        self.capture_snapshot(phase, &snapshot)
    }

    pub(crate) fn finish(
        mut self,
        telemetry: &TelemetryEvidence,
    ) -> Result<DeviceMemoryReport, DeviceBenchmarkError> {
        self.capture("after_measurement")?;

        self.into_report(telemetry)
    }

    fn into_report(
        self,
        telemetry: &TelemetryEvidence,
    ) -> Result<DeviceMemoryReport, DeviceBenchmarkError> {
        let warmed = self
            .snapshots
            .iter()
            .find(|snapshot| snapshot.phase == "after_warmup")
            .ok_or_else(|| {
                DeviceBenchmarkError::Precondition(
                    "memory recorder is missing the after_warmup snapshot".to_string(),
                )
            })?;
        let accounted = self.owned.iter().try_fold(0u64, |total, metric| {
            total.checked_add(metric.bytes).ok_or_else(|| {
                DeviceBenchmarkError::Precondition(
                    "accounted resident memory overflows the report width".to_string(),
                )
            })
        })?;
        let setup_delta = warmed
            .device_used_bytes
            .saturating_sub(self.preflight_used_bytes);
        let timed_peak_delta = telemetry
            .device_memory_maximum_used_bytes
            .saturating_sub(self.preflight_used_bytes);
        let timed_growth_after_warmup = telemetry
            .device_memory_maximum_used_bytes
            .saturating_sub(warmed.device_used_bytes);
        let minimum_headroom = telemetry.device_memory_minimum_free_bytes;
        let process_peak_rss = self
            .snapshots
            .iter()
            .map(|snapshot| snapshot.process_peak_rss_bytes)
            .max()
            .unwrap_or(0);
        let mut metrics = self.owned;
        metrics.extend([
            summary_memory_metric(
                "summary/accounted_resident",
                BenchmarkMemoryMeasurement::Owned,
                accounted,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/setup_device_delta",
                BenchmarkMemoryMeasurement::SetupDeviceDelta,
                setup_delta,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/timed_peak_device_delta",
                BenchmarkMemoryMeasurement::TimedPeakDeviceDelta,
                timed_peak_delta,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/timed_growth_after_warmup",
                BenchmarkMemoryMeasurement::TimedGrowthAfterWarmup,
                timed_growth_after_warmup,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/minimum_device_headroom",
                BenchmarkMemoryMeasurement::MinimumDeviceHeadroom,
                minimum_headroom,
                MemoryComparison::AtLeast,
            ),
            summary_memory_metric(
                "summary/process_peak_rss",
                BenchmarkMemoryMeasurement::ProcessPeakRss,
                process_peak_rss,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/device_reserved",
                BenchmarkMemoryMeasurement::DeviceReserved,
                warmed.device_reserved_bytes,
                MemoryComparison::AtMost,
            ),
            summary_memory_metric(
                "summary/unattributed_setup_delta",
                BenchmarkMemoryMeasurement::UnattributedSetupDelta,
                setup_delta.saturating_sub(accounted),
                MemoryComparison::AtMost,
            ),
        ]);

        Ok(DeviceMemoryReport {
            device_total_bytes: self.total_bytes,
            snapshots: self.snapshots,
            metrics,
        })
    }

    fn capture_snapshot(
        &mut self,
        phase: &'static str,
        device: &DeviceSnapshot,
    ) -> Result<(), DeviceBenchmarkError> {
        if device.identity.uuid != self.device_uuid {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "memory snapshot `{phase}` came from GPU {}, expected {}",
                device.identity.uuid, self.device_uuid
            )));
        }
        if mib_to_bytes(device.memory_total_mib) != self.total_bytes {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device memory capacity changed while capturing `{phase}`"
            )));
        }
        let (process_rss_bytes, process_peak_rss_bytes) = process_memory_bytes()?;
        self.snapshots.push(DeviceMemorySnapshot {
            phase: phase.to_string(),
            device_used_bytes: mib_to_bytes(device.memory_used_mib),
            device_free_bytes: mib_to_bytes(device.memory_free_mib),
            device_reserved_bytes: mib_to_bytes(device.memory_reserved_mib),
            process_rss_bytes,
            process_peak_rss_bytes,
        });

        Ok(())
    }
}

pub(crate) fn preflight() -> Result<DevicePreflight, DeviceBenchmarkError> {
    require_unmapped_ordinal_zero()?;
    let snapshot = device_snapshot()?;
    if snapshot.identity.name != EXPECTED_DEVICE_NAME {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "device zero is `{}`, expected `{EXPECTED_DEVICE_NAME}`",
            snapshot.identity.name
        )));
    }
    if snapshot.utilization_percent != 0 || snapshot.memory_used_mib > MAX_IDLE_MEMORY_MIB {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "device zero is not idle: utilization={}%, memory={} MiB",
            snapshot.utilization_percent, snapshot.memory_used_mib
        )));
    }
    require_compute_process_count(0)?;

    Ok(DevicePreflight {
        identity: DeviceIdentity {
            name: snapshot.identity.name.clone(),
            uuid: snapshot.identity.uuid.clone(),
            driver_version: snapshot.identity.driver_version.clone(),
        },
        memory: snapshot,
    })
}

pub(crate) fn require_current_process_exclusive() -> Result<(), DeviceBenchmarkError> {
    // Preflight admitted an empty GPU immediately before Session created one persistent CUDA
    // context. Requiring exactly one process here proves exclusivity without assuming that
    // nvidia-smi's host PID namespace matches the container's PID namespace.
    require_compute_process_count(1)
}

pub(crate) fn warmup_launches(
    options: DeviceBenchmarkOptions,
) -> Result<u64, DeviceBenchmarkError> {
    if options.samples < 3 || options.launches_per_sample == 0 || options.warmup_launches == 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "at least three samples, one launch per sample, and one warmup launch are required"
                .to_string(),
        ));
    }

    Ok(options.warmup_launches)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_report(
    spec: BenchmarkReportSpec,
    preflight: DevicePreflight,
    generator_baseline_sha256: String,
    options: DeviceBenchmarkOptions,
    metrics: Vec<DeviceBenchmarkMetric>,
    energy_metrics: Vec<DeviceEnergyMetric>,
    telemetry: TelemetryEvidence,
    memory: DeviceMemoryReport,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let identity = preflight.identity;

    Ok(DeviceBenchmarkReport {
        schema_version: 7,
        suite: spec.suite,
        classification: spec.classification,
        device: identity.name,
        device_uuid: identity.uuid,
        driver_version: identity.driver_version,
        device_index: 0,
        compute_capability: EXPECTED_COMPUTE_CAPABILITY_TEXT.to_string(),
        clock_policy: if diagnostic_clock_drift_allowed() || !telemetry.clock_comparable {
            "diagnostic_uncontrolled"
        } else {
            "controlled"
        },
        binary_sha256: executable_sha256()?,
        generator_baseline_sha256,
        sm_clock_min_mhz: telemetry.sm_minimum_mhz,
        sm_clock_median_mhz: telemetry.sm_median_mhz,
        sm_clock_max_mhz: telemetry.sm_maximum_mhz,
        memory_clock_min_mhz: telemetry.memory_minimum_mhz,
        memory_clock_median_mhz: telemetry.memory_median_mhz,
        memory_clock_max_mhz: telemetry.memory_maximum_mhz,
        temperature_min_celsius: telemetry.temperature_minimum_celsius,
        temperature_max_celsius: telemetry.temperature_maximum_celsius,
        power_min_watts: telemetry.power_minimum_watts,
        power_mean_watts: telemetry.power_mean_watts,
        power_median_watts: telemetry.power_median_watts,
        power_max_watts: telemetry.power_maximum_watts,
        telemetry_samples: telemetry.samples,
        samples: options.samples,
        launches_per_sample: options.launches_per_sample,
        warmup_launches: options.warmup_launches,
        case_policy: if options.batch_size.is_some() {
            "diagnostic_subset"
        } else {
            "complete_inventory"
        },
        selected_batch_size: options.batch_size,
        timing_scope: spec.timing_scope,
        power_scope: "nvidia-smi power.draw.instant, whole board",
        metrics,
        energy_metrics,
        memory,
    })
}

pub(crate) fn measure_cases(
    stream: &CudaStream,
    timer: &GpuTimer,
    cases: &[ExactDeviceCase<'_>],
    options: DeviceBenchmarkOptions,
) -> Result<
    (
        Vec<DeviceBenchmarkMetric>,
        Vec<DeviceEnergyMetric>,
        TelemetryEvidence,
    ),
    DeviceBenchmarkError,
> {
    let cases = selected_cases(cases, options.batch_size)?;

    validate_loaded_clock_policy(stream, timer, &cases)?;

    let mut tasks = Vec::with_capacity(cases.len() * 2);
    for (index, case) in cases.iter().enumerate() {
        tasks.push(MeasurementTask::Leaf(index));
        if case.repeated.is_some() {
            tasks.push(MeasurementTask::Repeated(index));
        }
    }
    let mut samples = (0..cases.len())
        .map(|_| CaseSamples::default())
        .collect::<Vec<_>>();
    let telemetry_sampler = TelemetrySampler::start();
    for sample in 0..options.samples {
        for task_index in measurement_order(sample, tasks.len()) {
            match tasks[task_index] {
                MeasurementTask::Leaf(case_index) => {
                    let case = &cases[case_index];
                    let timing = if let Some(preparation) = case.preparation_graph {
                        let mut timing = tuisko_gpu::GpuTiming {
                            device: Duration::ZERO,
                            host_submit: Duration::ZERO,
                            host_completion: Duration::ZERO,
                        };
                        for _ in 0..options.launches_per_sample {
                            preparation.launch(stream)?;
                            stream.synchronize().map_err(tuisko_gpu::GpuError::from)?;
                            let operation = timer
                                .measure_with_host(stream, || case.leaf_graph.launch(stream))?;
                            timing.device += operation.device;
                            timing.host_submit += operation.host_submit;
                            timing.host_completion += operation.host_completion;
                        }
                        timing
                    } else {
                        timer.measure_with_host(stream, || {
                            for _ in 0..options.launches_per_sample {
                                case.leaf_graph.launch(stream)?;
                            }

                            Ok(())
                        })?
                    };
                    let divisor = options.launches_per_sample;
                    samples[case_index]
                        .host_submit
                        .push(microseconds_per(timing.host_submit, divisor));
                    samples[case_index]
                        .host_completion
                        .push(microseconds_per(timing.host_completion, divisor));
                    samples[case_index]
                        .device_graph
                        .push(microseconds_per(timing.device, divisor));
                }
                MeasurementTask::Repeated(case_index) => {
                    let repeated = cases[case_index]
                        .repeated
                        .as_ref()
                        .expect("repeated task requires a repeated graph");
                    let elapsed = timer.measure(stream, || repeated.graph.launch(stream))?;
                    samples[case_index]
                        .device_path
                        .push(microseconds_per(elapsed, repeated.operations));
                }
            }
        }
    }
    let telemetry = telemetry_sampler.finish_preserving_clock_drift()?;
    require_current_process_exclusive()?;

    let mut metrics = Vec::with_capacity(tasks.len() * 2);
    let mut device_graph_medians = Vec::with_capacity(cases.len());
    for (case, samples) in cases.iter().zip(samples) {
        metrics.push(metric(
            case,
            BenchmarkMeasurement::HostSubmit,
            options.launches_per_sample,
            samples.host_submit,
        )?);
        metrics.push(metric(
            case,
            BenchmarkMeasurement::HostCompletion,
            options.launches_per_sample,
            samples.host_completion,
        )?);
        let device_graph = metric(
            case,
            BenchmarkMeasurement::DeviceGraph,
            options.launches_per_sample,
            samples.device_graph,
        )?;
        device_graph_medians.push(device_graph.median_microseconds);
        metrics.push(device_graph);
        if let Some(repeated) = &case.repeated {
            metrics.push(metric(
                case,
                BenchmarkMeasurement::DevicePath,
                repeated.operations,
                samples.device_path,
            )?);
        }
    }

    let energy_metrics = if let Some(seconds) = options.energy_seconds {
        measure_energy(stream, timer, &cases, &device_graph_medians, seconds)?
    } else {
        Vec::new()
    };

    Ok((metrics, energy_metrics, telemetry))
}

fn selected_cases<'a>(
    cases: &[ExactDeviceCase<'a>],
    batch_size: Option<u32>,
) -> Result<Vec<ExactDeviceCase<'a>>, DeviceBenchmarkError> {
    let selected = cases
        .iter()
        .filter(|case| batch_size.is_none_or(|batch| case.workload.batch_size == Some(batch)))
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        let selection = batch_size.map_or_else(
            || "the complete inventory".to_string(),
            |batch| format!("diagnostic B={batch}"),
        );
        return Err(DeviceBenchmarkError::Precondition(format!(
            "device benchmark has no case matching {selection}"
        )));
    }

    Ok(selected)
}

fn validate_loaded_clock_policy(
    stream: &CudaStream,
    timer: &GpuTimer,
    cases: &[ExactDeviceCase<'_>],
) -> Result<(), DeviceBenchmarkError> {
    let case = cases
        .iter()
        .max_by_key(|case| case.logical_bytes)
        .expect("nonempty cases checked by caller");
    let sampler = TelemetrySampler::start();
    let mut measured = Duration::ZERO;
    let mut replays = 1;
    let started = Instant::now();
    while measured < LOADED_CLOCK_PROBE_DURATION || started.elapsed() < LOADED_CLOCK_PROBE_DURATION
    {
        let elapsed = timer.measure(stream, || {
            for _ in 0..replays {
                launch_clock_probe_unit(stream, case)?;
            }

            Ok(())
        })?;
        measured += elapsed;
        if measured < LOADED_CLOCK_PROBE_DURATION {
            replays = loaded_clock_probe_replays(
                LOADED_CLOCK_PROBE_DURATION - measured,
                elapsed,
                replays,
            )?;
        }
    }
    let telemetry = sampler.finish().map_err(|error| match error {
        DeviceBenchmarkError::Precondition(reason) => DeviceBenchmarkError::Precondition(format!(
            "loaded clock probe refused before full timing: {reason}"
        )),
        other => other,
    })?;
    require_current_process_exclusive()?;
    eprintln!(
        "loaded clock probe passed: {} {}, SM {}..{} MHz (median {}), memory {}..{} MHz, {} samples",
        case.route,
        case.shape,
        telemetry.sm_minimum_mhz,
        telemetry.sm_maximum_mhz,
        telemetry.sm_median_mhz,
        telemetry.memory_minimum_mhz,
        telemetry.memory_maximum_mhz,
        telemetry.samples,
    );

    Ok(())
}

fn launch_clock_probe_unit(
    stream: &CudaStream,
    case: &ExactDeviceCase<'_>,
) -> Result<(), tuisko_gpu::GpuError> {
    if let Some(repeated) = &case.repeated {
        repeated.graph.launch(stream)
    } else {
        if let Some(preparation) = case.preparation_graph {
            preparation.launch(stream)?;
        }
        case.leaf_graph.launch(stream)
    }
}

fn loaded_clock_probe_replays(
    remaining: Duration,
    elapsed: Duration,
    completed_replays: u64,
) -> Result<u64, DeviceBenchmarkError> {
    let elapsed_nanos = elapsed.as_nanos();
    if elapsed_nanos == 0 || completed_replays == 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "loaded clock probe produced zero work or device duration".to_string(),
        ));
    }
    let replays = remaining
        .as_nanos()
        .saturating_mul(u128::from(completed_replays))
        .div_ceil(elapsed_nanos);

    Ok(u64::try_from(replays)
        .unwrap_or(u64::MAX)
        .clamp(1, MAX_LOADED_CLOCK_PROBE_REPLAYS))
}

fn measure_energy(
    stream: &CudaStream,
    timer: &GpuTimer,
    cases: &[ExactDeviceCase<'_>],
    device_graph_medians: &[f64],
    target_seconds: f64,
) -> Result<Vec<DeviceEnergyMetric>, DeviceBenchmarkError> {
    if !target_seconds.is_finite() || target_seconds < 2.0 {
        return Err(DeviceBenchmarkError::Precondition(
            "energy windows must be finite and at least two seconds".to_string(),
        ));
    }

    stream.synchronize().map_err(tuisko_gpu::GpuError::from)?;
    let idle_sampler = TelemetrySampler::start();
    thread::sleep(Duration::from_secs_f64(target_seconds));
    let idle_telemetry = idle_sampler.finish()?;
    require_current_process_exclusive()?;

    let mut metrics = Vec::with_capacity(cases.len());
    for (case, &median_microseconds) in cases.iter().zip(device_graph_medians) {
        let operations = (target_seconds * 1_000_000.0 / median_microseconds)
            .ceil()
            .clamp(1.0, u64::MAX as f64) as u64;
        let sampler = TelemetrySampler::start();
        let timing = timer.measure_with_host(stream, || {
            for _ in 0..operations {
                if let Some(preparation) = case.preparation_graph {
                    preparation.launch(stream)?;
                }
                case.leaf_graph.launch(stream)?;
            }

            Ok(())
        })?;
        let telemetry = sampler.finish()?;
        require_current_process_exclusive()?;
        let device_seconds = timing.device.as_secs_f64();
        let completed_units = operations
            .checked_mul(case.units_per_operation)
            .ok_or_else(|| {
                DeviceBenchmarkError::Precondition(
                    "energy normalization unit count overflows".to_string(),
                )
            })?;
        let (
            dynamic_board_watts,
            estimated_board_joules_per_unit,
            estimated_dynamic_joules_per_unit,
            estimated_units_per_board_joule,
        ) = energy_estimates(
            telemetry.power_mean_watts,
            idle_telemetry.power_mean_watts,
            device_seconds,
            completed_units,
        )?;

        metrics.push(DeviceEnergyMetric {
            route: case.route,
            shape: case.shape.clone(),
            workload: case.workload.clone(),
            unit: case.unit,
            operations,
            units_per_operation: case.units_per_operation,
            device_seconds,
            average_board_watts: telemetry.power_mean_watts,
            idle_board_watts: idle_telemetry.power_mean_watts,
            dynamic_board_watts,
            estimated_board_joules_per_unit,
            estimated_dynamic_joules_per_unit,
            estimated_units_per_board_joule,
            telemetry_samples: telemetry.samples,
        });
    }

    Ok(metrics)
}

fn energy_estimates(
    average_board_watts: f64,
    idle_board_watts: f64,
    device_seconds: f64,
    completed_units: u64,
) -> Result<(f64, f64, f64, f64), DeviceBenchmarkError> {
    if !average_board_watts.is_finite() || average_board_watts <= 0.0 {
        return Err(DeviceBenchmarkError::Precondition(
            "average board power must be finite and positive".to_string(),
        ));
    }
    if !idle_board_watts.is_finite() || idle_board_watts < 0.0 {
        return Err(DeviceBenchmarkError::Precondition(
            "idle board power must be finite and nonnegative".to_string(),
        ));
    }
    if !device_seconds.is_finite() || device_seconds <= 0.0 || completed_units == 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "energy work and duration must be positive".to_string(),
        ));
    }

    let dynamic_board_watts = (average_board_watts - idle_board_watts).max(0.0);
    let estimated_board_joules_per_unit =
        average_board_watts * device_seconds / completed_units as f64;
    let estimated_dynamic_joules_per_unit =
        dynamic_board_watts * device_seconds / completed_units as f64;
    let estimated_units_per_board_joule = 1.0 / estimated_board_joules_per_unit;
    if [
        dynamic_board_watts,
        estimated_board_joules_per_unit,
        estimated_dynamic_joules_per_unit,
        estimated_units_per_board_joule,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(DeviceBenchmarkError::Precondition(
            "energy estimate is non-finite or negative".to_string(),
        ));
    }

    Ok((
        dynamic_board_watts,
        estimated_board_joules_per_unit,
        estimated_dynamic_joules_per_unit,
        estimated_units_per_board_joule,
    ))
}

pub(crate) fn measurement_order(sample: usize, metric_count: usize) -> Vec<usize> {
    if metric_count == 0 {
        return Vec::new();
    }
    let mut order = (0..metric_count).collect::<Vec<_>>();
    if sample % 2 == 1 {
        order.reverse();
    }
    order.rotate_left(sample % metric_count);

    order
}

fn metric(
    case: &ExactDeviceCase<'_>,
    measurement: BenchmarkMeasurement,
    operations_per_interval: u64,
    mut samples: Vec<f64>,
) -> Result<DeviceBenchmarkMetric, DeviceBenchmarkError> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "{} {} {} produced a non-finite or non-positive timing",
            case.route,
            case.shape,
            measurement.as_str()
        )));
    }

    samples.sort_by(f64::total_cmp);
    let percentile = |numerator: usize| {
        let index = (samples.len() - 1) * numerator / 10;
        samples[index]
    };
    let median = percentile(5);
    let logical_bytes_per_operation = case.logical_bytes as u64;
    let logical_gib_per_second = measurement.is_device().then(|| {
        logical_bytes_per_operation as f64 / median / (1024.0 * 1024.0 * 1024.0) * 1_000_000.0
    });
    if logical_gib_per_second.is_some_and(|throughput| !throughput.is_finite() || throughput <= 0.0)
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "{} {} {} produced invalid logical throughput",
            case.route,
            case.shape,
            measurement.as_str()
        )));
    }

    Ok(DeviceBenchmarkMetric {
        route: case.route,
        shape: case.shape.clone(),
        workload: case.workload.clone(),
        measurement,
        median_microseconds: median,
        p10_microseconds: percentile(1),
        p90_microseconds: percentile(9),
        operations_per_interval,
        logical_bytes_per_operation,
        logical_gib_per_second,
    })
}

pub(crate) fn executable_sha256() -> Result<String, DeviceBenchmarkError> {
    Ok(sha256(&fs::read(env::current_exe()?)?))
}

pub(crate) fn generator_baseline_sha256() -> Result<String, DeviceBenchmarkError> {
    env::var("TUISKO_GENERATOR_BASELINE_SHA256").map_err(|_| {
        DeviceBenchmarkError::Precondition(
            "run through the matching `cargo run -p xtask -- bench-...` command".to_string(),
        )
    })
}

fn telemetry_evidence(
    samples: Vec<TelemetrySample>,
    allow_clock_drift: bool,
) -> Result<TelemetryEvidence, DeviceBenchmarkError> {
    if samples.len() < MIN_TELEMETRY_SAMPLES {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "only {} in-window telemetry samples were captured, expected at least {MIN_TELEMETRY_SAMPLES}; increase the benchmark size",
            samples.len()
        )));
    }

    let mut sm = samples
        .iter()
        .map(|sample| sample.sm_clock_mhz)
        .collect::<Vec<_>>();
    let mut memory = samples
        .iter()
        .map(|sample| sample.memory_clock_mhz)
        .collect::<Vec<_>>();
    let mut temperature = samples
        .iter()
        .map(|sample| sample.temperature_celsius)
        .collect::<Vec<_>>();
    let mut power = samples
        .iter()
        .map(|sample| sample.power_watts)
        .collect::<Vec<_>>();
    let mut device_memory = samples
        .iter()
        .map(|sample| mib_to_bytes(sample.device_memory_used_mib))
        .collect::<Vec<_>>();
    let mut device_free = samples
        .iter()
        .map(|sample| mib_to_bytes(sample.device_memory_free_mib))
        .collect::<Vec<_>>();
    sm.sort_unstable();
    memory.sort_unstable();
    temperature.sort_unstable();
    power.sort_by(f64::total_cmp);
    device_memory.sort_unstable();
    device_free.sort_unstable();

    let sm_spread = sm[sm.len() - 1] - sm[0];
    let sm_comparable = sm_spread <= MAX_SM_CLOCK_SPREAD_MHZ;
    if !sm_comparable && !allow_clock_drift {
        return Err(DeviceBenchmarkError::Precondition(clock_drift_message(
            "SM",
            sm[0],
            sm[sm.len() - 1],
        )));
    }
    let memory_spread = memory[memory.len() - 1] - memory[0];
    let memory_comparable = memory_spread <= MAX_MEMORY_CLOCK_SPREAD_MHZ;
    if !memory_comparable && !allow_clock_drift {
        return Err(DeviceBenchmarkError::Precondition(clock_drift_message(
            "memory",
            memory[0],
            memory[memory.len() - 1],
        )));
    }

    Ok(TelemetryEvidence {
        clock_comparable: sm_comparable && memory_comparable,
        sm_minimum_mhz: sm[0],
        sm_median_mhz: sm[sm.len() / 2],
        sm_maximum_mhz: sm[sm.len() - 1],
        memory_minimum_mhz: memory[0],
        memory_median_mhz: memory[memory.len() / 2],
        memory_maximum_mhz: memory[memory.len() - 1],
        temperature_minimum_celsius: temperature[0],
        temperature_maximum_celsius: temperature[temperature.len() - 1],
        power_minimum_watts: power[0],
        power_mean_watts: power.iter().sum::<f64>() / power.len() as f64,
        power_median_watts: power[power.len() / 2],
        power_maximum_watts: power[power.len() - 1],
        device_memory_maximum_used_bytes: device_memory[device_memory.len() - 1],
        device_memory_minimum_free_bytes: device_free[0],
        samples: samples.len(),
    })
}

fn clock_drift_message(kind: &str, minimum_mhz: u32, maximum_mhz: u32) -> String {
    if let Some(lock) = CLOCK_LOCK_COMMAND {
        format!(
            "{kind} clock moved from {minimum_mhz} to {maximum_mhz} MHz\nlock target clocks before the run:\n  {lock}\nreset them afterward:\n  {CLOCK_RESET_COMMAND}\nfor an explicitly non-authoritative diagnostic report, set `{DIAGNOSTIC_CLOCK_ENV}=1`; diagnostic reports cannot be blessed"
        )
    } else {
        format!(
            "{kind} clock moved from {minimum_mhz} to {maximum_mhz} MHz\nthis target has no blessed clock-lock profile; use `{DIAGNOSTIC_CLOCK_ENV}=1` only for exploratory tuning"
        )
    }
}

fn diagnostic_clock_drift_allowed() -> bool {
    env::var(DIAGNOSTIC_CLOCK_ENV).as_deref() == Ok("1")
}

fn summary_memory_metric(
    name: &'static str,
    measurement: BenchmarkMemoryMeasurement,
    bytes: u64,
    comparison: MemoryComparison,
) -> DeviceMemoryMetric {
    DeviceMemoryMetric {
        name: name.to_string(),
        measurement,
        kind: None,
        scaling: None,
        bytes,
        comparison,
    }
}

fn mib_to_bytes(mib: u32) -> u64 {
    u64::from(mib) * 1024 * 1024
}

fn process_memory_bytes() -> Result<(u64, u64), DeviceBenchmarkError> {
    parse_process_memory(&fs::read_to_string("/proc/self/status")?).map_err(|message| {
        DeviceBenchmarkError::Precondition(format!(
            "could not read process memory from /proc/self/status: {message}"
        ))
    })
}

fn parse_process_memory(status: &str) -> Result<(u64, u64), String> {
    let value = |name: &str| -> Result<u64, String> {
        let line = status
            .lines()
            .find(|line| line.starts_with(name))
            .ok_or_else(|| format!("missing {name}"))?;
        let mut fields = line[name.len()..].split_whitespace();
        let kib = fields
            .next()
            .ok_or_else(|| format!("missing {name} value"))?
            .parse::<u64>()
            .map_err(|_| format!("invalid {name} value"))?;
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(format!("unexpected {name} unit"));
        }

        kib.checked_mul(1024)
            .ok_or_else(|| format!("{name} byte count overflows"))
    };

    Ok((value("VmRSS:")?, value("VmHWM:")?))
}

fn microseconds_per(duration: Duration, operations: u64) -> f64 {
    duration.as_secs_f64() * 1_000_000.0 / operations as f64
}

fn require_unmapped_ordinal_zero() -> Result<(), DeviceBenchmarkError> {
    match env::var("CUDA_VISIBLE_DEVICES") {
        Err(env::VarError::NotPresent) => Ok(()),
        Ok(value) if value == DEVICE_INDEX => Ok(()),
        Ok(value) => Err(DeviceBenchmarkError::Precondition(format!(
            "CUDA_VISIBLE_DEVICES is `{value}`; unset it or set it to `0` so CUDA and nvidia-smi identify the same GPU"
        ))),
        Err(env::VarError::NotUnicode(_)) => Err(DeviceBenchmarkError::Precondition(
            "CUDA_VISIBLE_DEVICES is not UTF-8".to_string(),
        )),
    }
}

fn device_snapshot() -> Result<DeviceSnapshot, DeviceBenchmarkError> {
    let output = require_command(
        "nvidia-smi",
        &[
            "-i",
            DEVICE_INDEX,
            "--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,memory.free,memory.reserved,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let text = String::from_utf8_lossy(&output.stdout);
    let fields = text.trim().split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 8 {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "unexpected nvidia-smi device row `{}`",
            text.trim()
        )));
    }

    Ok(DeviceSnapshot {
        identity: DeviceIdentity {
            name: fields[0].to_string(),
            uuid: fields[1].to_string(),
            driver_version: fields[2].to_string(),
        },
        utilization_percent: parse_nvidia_u32("utilization.gpu", fields[3])?,
        memory_used_mib: parse_nvidia_u32("memory.used", fields[4])?,
        memory_free_mib: parse_nvidia_u32("memory.free", fields[5])?,
        memory_reserved_mib: parse_nvidia_u32("memory.reserved", fields[6])?,
        memory_total_mib: parse_nvidia_u32("memory.total", fields[7])?,
    })
}

fn query_telemetry() -> Result<TelemetrySample, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            DEVICE_INDEX,
            "--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw.instant,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("nvidia-smi telemetry query failed: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi telemetry query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let fields = text.trim().split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 6 {
        return Err(format!(
            "unexpected nvidia-smi telemetry row `{}`",
            text.trim()
        ));
    }

    Ok(TelemetrySample {
        sm_clock_mhz: parse_telemetry_u32("clocks.current.sm", fields[0])?,
        memory_clock_mhz: parse_telemetry_u32("clocks.current.memory", fields[1])?,
        temperature_celsius: parse_telemetry_u32("temperature.gpu", fields[2])?,
        power_watts: fields[3].parse().map_err(|_| {
            format!(
                "nvidia-smi power.draw.instant value `{}` is not a number",
                fields[3]
            )
        })?,
        device_memory_used_mib: parse_telemetry_u32("memory.used", fields[4])?,
        device_memory_free_mib: parse_telemetry_u32("memory.free", fields[5])?,
    })
}

fn parse_telemetry_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("nvidia-smi {name} value `{value}` is not an integer"))
}

fn parse_nvidia_u32(name: &str, value: &str) -> Result<u32, DeviceBenchmarkError> {
    value.parse().map_err(|_| {
        DeviceBenchmarkError::Precondition(format!(
            "nvidia-smi `{name}` value `{value}` is not an integer"
        ))
    })
}

fn require_compute_process_count(expected: usize) -> Result<(), DeviceBenchmarkError> {
    let output = require_command(
        "nvidia-smi",
        &[
            "-i",
            DEVICE_INDEX,
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let text = String::from_utf8_lossy(&output.stdout);
    let processes = parse_compute_processes(&text).map_err(DeviceBenchmarkError::Precondition)?;
    validate_compute_process_count(&processes, expected).map_err(DeviceBenchmarkError::Precondition)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComputeProcess {
    pid: u32,
    executable: String,
}

fn parse_compute_processes(text: &str) -> Result<Vec<ComputeProcess>, String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (pid, executable) = line
                .split_once(',')
                .ok_or_else(|| format!("unexpected nvidia-smi compute-process row `{line}`"))?;
            let pid = pid.trim().parse::<u32>().map_err(|_| {
                format!(
                    "nvidia-smi compute-process PID `{}` is not an integer",
                    pid.trim()
                )
            })?;
            let executable = executable.trim();
            if executable.is_empty() {
                return Err(format!(
                    "nvidia-smi compute-process {pid} has no executable name"
                ));
            }
            Ok(ComputeProcess {
                pid,
                executable: executable.to_owned(),
            })
        })
        .collect()
}

fn validate_compute_process_count(
    processes: &[ComputeProcess],
    expected: usize,
) -> Result<(), String> {
    if processes.len() == expected {
        return Ok(());
    }

    let observed = processes
        .iter()
        .map(|process| format!("{} ({})", process.pid, process.executable))
        .collect::<Vec<_>>()
        .join(", ");
    let observed = if observed.is_empty() {
        "none".to_owned()
    } else {
        observed
    };
    Err(format!(
        "expected {expected} compute process(es), observed {}: {observed}",
        processes.len()
    ))
}

fn require_command(program: &str, arguments: &[&str]) -> Result<Output, DeviceBenchmarkError> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    Ok(output)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        BenchmarkMemoryKind, BenchmarkMemoryMeasurement, ComputeProcess, DeviceBenchmarkOptions,
        DeviceMemoryMetric, DeviceMemorySnapshot, MemoryComparison, MemoryRecorder,
        TelemetryEvidence, TelemetrySample, loaded_clock_probe_replays, measurement_order,
        parse_compute_processes, parse_process_memory, telemetry_evidence,
        validate_compute_process_count, warmup_launches,
    };
    use std::time::Duration;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn benchmark_budgets_match_graph_duration_classes() {
        let short = DeviceBenchmarkOptions::short_graph();
        let long = DeviceBenchmarkOptions::long_graph();
        let resident = DeviceBenchmarkOptions::resident_model();

        assert_eq!(
            (short.launches_per_sample, short.warmup_launches),
            (256, 1_024)
        );
        assert_eq!((long.launches_per_sample, long.warmup_launches), (32, 128));
        assert_eq!(
            (resident.launches_per_sample, resident.warmup_launches),
            (1, 16)
        );
        assert_eq!(warmup_launches(resident).unwrap(), 16);
    }

    #[test]
    fn telemetry_statistics_are_deterministic() {
        let evidence = telemetry_evidence(
            vec![
                TelemetrySample {
                    sm_clock_mhz: 2_190,
                    memory_clock_mhz: 14_000,
                    temperature_celsius: 51,
                    power_watts: 220.0,
                    device_memory_used_mib: 100,
                    device_memory_free_mib: 900,
                },
                TelemetrySample {
                    sm_clock_mhz: 2_200,
                    memory_clock_mhz: 14_010,
                    temperature_celsius: 53,
                    power_watts: 240.0,
                    device_memory_used_mib: 102,
                    device_memory_free_mib: 898,
                },
                TelemetrySample {
                    sm_clock_mhz: 2_195,
                    memory_clock_mhz: 14_005,
                    temperature_celsius: 52,
                    power_watts: 230.0,
                    device_memory_used_mib: 101,
                    device_memory_free_mib: 899,
                },
            ],
            false,
        )
        .unwrap();

        assert_eq!(evidence.sm_median_mhz, 2_195);
        assert_eq!(evidence.memory_median_mhz, 14_005);
        assert_eq!(evidence.temperature_maximum_celsius, 53);
        assert_eq!(evidence.power_median_watts, 230.0);
        assert_eq!(evidence.device_memory_maximum_used_bytes, 102 * 1024 * 1024);
        assert_eq!(evidence.device_memory_minimum_free_bytes, 898 * 1024 * 1024);
    }

    #[test]
    fn loaded_clock_probe_targets_two_seconds_without_unbounded_submission() {
        assert_eq!(
            loaded_clock_probe_replays(Duration::from_secs(2), Duration::from_millis(500), 1,)
                .unwrap(),
            4
        );
        assert_eq!(
            loaded_clock_probe_replays(Duration::from_secs(1), Duration::from_micros(2), 1)
                .unwrap(),
            500_000
        );
        assert_eq!(
            loaded_clock_probe_replays(Duration::from_secs(2), Duration::from_micros(1), 1)
                .unwrap(),
            1_000_000
        );
        assert!(loaded_clock_probe_replays(Duration::from_secs(1), Duration::ZERO, 1).is_err());
        assert!(
            loaded_clock_probe_replays(Duration::from_secs(1), Duration::from_millis(1), 0)
                .is_err()
        );
    }

    #[test]
    #[cfg(feature = "device")]
    fn target_clock_p_state_steps_are_admitted() {
        let sample = |sm_clock_mhz, memory_clock_mhz| TelemetrySample {
            sm_clock_mhz,
            memory_clock_mhz,
            temperature_celsius: 50,
            power_watts: 220.0,
            device_memory_used_mib: 100,
            device_memory_free_mib: 900,
        };
        let evidence = telemetry_evidence(
            vec![
                sample(2_160, 13_801),
                sample(2_197, 14_001),
                sample(2_190, 13_801),
            ],
            false,
        )
        .expect("the target's measured clock P-state steps are comparable");

        assert!(evidence.clock_comparable);
        assert_eq!(evidence.sm_minimum_mhz, 2_160);
        assert_eq!(evidence.sm_maximum_mhz, 2_197);
        assert_eq!(evidence.memory_minimum_mhz, 13_801);
        assert_eq!(evidence.memory_maximum_mhz, 14_001);
    }

    #[test]
    fn larger_sm_clock_movement_is_refused() {
        let sample = |sm_clock_mhz| TelemetrySample {
            sm_clock_mhz,
            memory_clock_mhz: 14_001,
            temperature_celsius: 50,
            power_watts: 220.0,
            device_memory_used_mib: 100,
            device_memory_free_mib: 900,
        };
        let error =
            match telemetry_evidence(vec![sample(2_100), sample(2_200), sample(2_200)], false) {
                Ok(_) => panic!("larger SM clock movement must be refused"),
                Err(error) => error.to_string(),
            };

        assert!(error.contains("SM clock moved from 2100 to 2200 MHz"));
    }

    #[test]
    #[cfg(feature = "device")]
    fn clock_refusal_prints_lock_and_reset_commands() {
        let sample = |memory_clock_mhz| TelemetrySample {
            sm_clock_mhz: 2_200,
            memory_clock_mhz,
            temperature_celsius: 50,
            power_watts: 220.0,
            device_memory_used_mib: 100,
            device_memory_free_mib: 900,
        };
        let error =
            match telemetry_evidence(vec![sample(13_601), sample(14_001), sample(14_001)], false) {
                Ok(_) => panic!("clock spread must be refused"),
                Err(error) => error.to_string(),
            };

        assert!(error.contains("--lock-gpu-clocks=2200,2200"));
        assert!(error.contains("--lock-memory-clocks=14001,14001"));
        assert!(error.contains("--reset-gpu-clocks"));
        assert!(error.contains("--reset-memory-clocks"));
        assert!(error.contains("TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT=1"));
        assert!(error.contains("diagnostic reports cannot be blessed"));

        let evidence =
            telemetry_evidence(vec![sample(13_601), sample(14_001), sample(14_001)], true)
                .expect("diagnostic runs retain drifting clock evidence");
        assert!(!evidence.clock_comparable);
        assert_eq!(evidence.memory_minimum_mhz, 13_601);
        assert_eq!(evidence.memory_maximum_mhz, 14_001);
    }

    #[test]
    #[cfg(not(feature = "device"))]
    fn unblessed_target_clock_refusal_is_explicitly_diagnostic() {
        let sample = |memory_clock_mhz| TelemetrySample {
            sm_clock_mhz: 2_200,
            memory_clock_mhz,
            temperature_celsius: 50,
            power_watts: 220.0,
            device_memory_used_mib: 100,
            device_memory_free_mib: 900,
        };
        let error =
            match telemetry_evidence(vec![sample(13_801), sample(14_001), sample(14_001)], false) {
                Ok(_) => panic!("clock spread must be refused"),
                Err(error) => error.to_string(),
            };

        assert!(error.contains("no blessed clock-lock profile"));
        assert!(error.contains("TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT=1"));
    }

    #[test]
    fn process_memory_parser_requires_named_kib_fields() {
        let status = "Name:\ttest\nVmHWM:\t4096 kB\nVmRSS:\t2048 kB\n";
        assert_eq!(
            parse_process_memory(status).unwrap(),
            (2_097_152, 4_194_304)
        );
        assert!(parse_process_memory("VmRSS:\t1 MB\nVmHWM:\t2 kB\n").is_err());
        assert!(parse_process_memory("VmRSS:\t1 kB\n").is_err());
    }

    #[test]
    fn compute_process_gate_requires_the_empty_to_singleton_transition() {
        let namespaced = parse_compute_processes("3863838, /tmp/tuiskollm/bench-device\n").unwrap();
        assert!(validate_compute_process_count(&[], 0).is_ok());
        assert!(validate_compute_process_count(&namespaced, 0).is_err());
        assert!(validate_compute_process_count(&namespaced, 1).is_ok());

        let hidden = parse_compute_processes("3549378, [Not Found]\n").unwrap();
        assert!(validate_compute_process_count(&hidden, 1).is_ok());

        let peer = ComputeProcess {
            pid: 99,
            executable: "/tmp/tuiskollm/bench-device".to_owned(),
        };
        assert!(validate_compute_process_count(&[namespaced[0].clone(), peer], 1).is_err());
        assert!(validate_compute_process_count(&[], 1).is_err());
        assert!(parse_compute_processes("not-a-pid, bench-device\n").is_err());
    }

    #[test]
    fn workload_serialization_pins_comparison_dimensions() {
        let workload = super::BenchmarkWorkload::warm_operator_decode(4);
        let json = serde_json::to_value(workload).unwrap();

        assert_eq!(json["scope"], "operator");
        assert_eq!(json["phase"], "decode");
        assert_eq!(json["batch_size"], 4);
        assert_eq!(json["active_tokens"], 4);
        assert_eq!(json["device_cache"], "warm");
        assert_eq!(json["execution"], "cuda_graph");
        assert!(json["context_tokens"].is_null());
        assert!(json["prefix_cache"].is_null());
    }

    #[test]
    fn memory_report_keeps_free_reserved_and_attributed_bytes_distinct() {
        let recorder = MemoryRecorder {
            device_uuid: "GPU-test".to_string(),
            preflight_used_bytes: 100 * MIB,
            total_bytes: 1_000 * MIB,
            snapshots: vec![DeviceMemorySnapshot {
                phase: "after_warmup".to_string(),
                device_used_bytes: 300 * MIB,
                device_free_bytes: 600 * MIB,
                device_reserved_bytes: 100 * MIB,
                process_rss_bytes: 20 * MIB,
                process_peak_rss_bytes: 25 * MIB,
            }],
            owned: vec![DeviceMemoryMetric {
                name: "owner".to_string(),
                measurement: BenchmarkMemoryMeasurement::Owned,
                kind: Some(BenchmarkMemoryKind::Workspace),
                scaling: Some("fixed".to_string()),
                bytes: 150 * MIB,
                comparison: MemoryComparison::AtMost,
            }],
        };
        let telemetry = TelemetryEvidence {
            clock_comparable: true,
            sm_minimum_mhz: 2_200,
            sm_median_mhz: 2_200,
            sm_maximum_mhz: 2_200,
            memory_minimum_mhz: 14_000,
            memory_median_mhz: 14_000,
            memory_maximum_mhz: 14_000,
            temperature_minimum_celsius: 50,
            temperature_maximum_celsius: 50,
            power_minimum_watts: 200.0,
            power_mean_watts: 200.0,
            power_median_watts: 200.0,
            power_maximum_watts: 200.0,
            device_memory_maximum_used_bytes: 350 * MIB,
            device_memory_minimum_free_bytes: 550 * MIB,
            samples: 3,
        };
        let report = recorder.into_report(&telemetry).unwrap();
        let bytes = |name: &str| {
            report
                .metrics
                .iter()
                .find(|metric| metric.name == name)
                .unwrap()
                .bytes
        };

        assert_eq!(bytes("summary/accounted_resident"), 150 * MIB);
        assert_eq!(bytes("summary/setup_device_delta"), 200 * MIB);
        assert_eq!(bytes("summary/timed_peak_device_delta"), 250 * MIB);
        assert_eq!(bytes("summary/timed_growth_after_warmup"), 50 * MIB);
        assert_eq!(bytes("summary/minimum_device_headroom"), 550 * MIB);
        assert_eq!(bytes("summary/device_reserved"), 100 * MIB);
        assert_eq!(bytes("summary/unattributed_setup_delta"), 50 * MIB);
    }

    #[test]
    fn energy_estimates_distinguish_board_and_above_idle_work() {
        let (dynamic_watts, board_joules, dynamic_joules, units_per_joule) =
            super::energy_estimates(75.0, 55.0, 2.0, 1_000).unwrap();

        assert_eq!(dynamic_watts, 20.0);
        assert_eq!(board_joules, 0.15);
        assert_eq!(dynamic_joules, 0.04);
        assert!((units_per_joule - 6.666666666666667).abs() < f64::EPSILON);

        let (_, _, zero_dynamic, _) = super::energy_estimates(50.0, 55.0, 2.0, 1_000).unwrap();
        assert_eq!(zero_dynamic, 0.0);
        assert!(super::energy_estimates(f64::NAN, 55.0, 2.0, 1_000).is_err());
    }

    #[test]
    fn measurement_order_rotates_and_reverses_complete_metric_set() {
        assert!(measurement_order(0, 0).is_empty());
        assert_eq!(measurement_order(0, 4), vec![0, 1, 2, 3]);
        assert_eq!(measurement_order(1, 4), vec![2, 1, 0, 3]);
        assert_eq!(measurement_order(2, 4), vec![2, 3, 0, 1]);

        for sample in 0..16 {
            let mut order = measurement_order(sample, 16);
            order.sort_unstable();
            assert_eq!(order, (0..16).collect::<Vec<_>>());
        }
    }
}

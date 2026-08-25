//! Direct timing for one source-backed Qwen3.5 GDN layer.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35GdnLayerProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const SOURCE_LAYER: usize = 0;
const CONTROL_STRIDE: usize = 128;
const NVFP4_GROUP: usize = 16;
const MAX_ROWS: usize = 128;
const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];

struct RouteGraph {
    rows: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: Qwen35GdnLayerProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = Qwen35GdnLayerProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
        program.reset_state(&stream)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| {
                Ok(RouteGraph {
                    rows,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        rows,
                        repeated_operations,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;

        Ok(Self {
            routes,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for rows in EXACT_ROUTES {
                // SAFETY: the program retains every captured layer allocation through this replay.
                unsafe { self.program.qualification_graph(rows)?.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    fn cases(
        &self,
        repeated_operations: u64,
    ) -> Result<Vec<ExactDeviceCase<'_>>, DeviceBenchmarkError> {
        self.routes
            .iter()
            .map(|route| {
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_layer_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_layer_prefill(route.rows as u64),
                    )
                };
                Ok(ExactDeviceCase::new(
                    "qwen35_9b/gdn/layer0",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    self.program.qualification_graph(route.rows)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
            })
            .collect()
    }
}

fn benchmark_input() -> Vec<u16> {
    const PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
    (0..MAX_ROWS * Qwen35_9B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen35_9B::HIDDEN) & 7]))
        .collect()
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let input_rows = Qwen35_9B::GDN_INPUT_ROWS;
    let qkv_rows = Qwen35_9B::GDN_QKV_ROWS;
    let value_rows = Qwen35_9B::GDN_VALUE_ROWS;
    let controls = Qwen35_9B::GDN_CONTROL_ROWS;
    let state = controls * Qwen35_9B::LINEAR_HEAD_DIM * Qwen35_9B::LINEAR_HEAD_DIM;
    let weights = 123_068_800;
    let plain_norm = 4 * hidden;
    let hidden_scratch = hidden / 2 + hidden / NVFP4_GROUP;
    let intermediate_scratch = intermediate / 2 + intermediate / NVFP4_GROUP;
    let input_projection = if rows <= MAX_BATCH {
        4 * hidden + 2 * (input_rows + CONTROL_STRIDE)
    } else {
        2 * hidden + 3 * hidden_scratch + 2 * (input_rows + CONTROL_STRIDE)
    };
    let prepare = 4 * controls + 16 * qkv_rows + 8 * controls;
    let recurrence = 2 * qkv_rows + 2 * value_rows + 8 * controls + 8 * state + 2 * value_rows;
    let output_projection = if rows <= MAX_BATCH {
        2 * (value_rows + hidden)
    } else {
        2 * value_rows + 2 * hidden_scratch + 2 * hidden
    };
    let fused_residual = 8 * hidden;
    let swiglu = 2 * (hidden + intermediate) + usize::from(uses_w4a4(rows)) * 2 * hidden_scratch;
    let down = if rows <= MAX_BATCH {
        2 * (intermediate + hidden)
    } else {
        2 * intermediate + 2 * intermediate_scratch + 2 * hidden
    };
    let per_token = plain_norm
        + input_projection
        + prepare
        + recurrence
        + output_projection
        + 2 * fused_residual
        + swiglu
        + down;

    weights + rows * per_token
}

fn uses_w4a4(rows: usize) -> bool {
    rows > MAX_BATCH || rows == 1 || rows >= 3
}

/// Measures every exact graph of one source-backed Qwen3.5 GDN layer.
pub fn benchmark_qwen35_gdn_layer(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "qwen35_9b/gdn/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=0 source-native parameters and losslessly materialized NVFP4 planes",
    )?;
    memory.register_owned(
        "qwen35_9b/gdn/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=128 plus eight-slot causal history and FP32 state",
    )?;
    memory.register_owned(
        "qwen35_9b/gdn/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.workspace_bytes(),
        "single 256-byte-aligned arena",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-gdn-layer",
            classification: "performance_sensitive_layer",
            timing_scope: "paired Rust submission/completion, repeated production graph, and repeated-operation graph",
        },
        preflight,
        baseline_sha256,
        options,
        metrics,
        energy_metrics,
        telemetry,
        memory,
    )
}

#[cfg(test)]
mod tests {
    use super::{EXACT_ROUTES, MAX_BATCH, logical_bytes, uses_w4a4};

    #[test]
    fn accounting_covers_recurrent_state_and_both_mlp_routes() {
        assert_eq!(logical_bytes(1), 127_637_248);
        assert_eq!(logical_bytes(2), 132_196_480);
        assert_eq!(logical_bytes(MAX_BATCH), 159_616_384);
        assert_eq!(logical_bytes(32), 269_808_000);
        assert_eq!(logical_bytes(128), 710_025_600);
        assert_eq!(
            EXACT_ROUTES.map(uses_w4a4),
            [
                true, false, true, true, true, true, true, true, true, true, true
            ]
        );
    }
}

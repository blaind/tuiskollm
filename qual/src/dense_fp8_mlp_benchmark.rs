//! Direct timing for one source-backed dense-FP8 MLP owner.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::DenseFp8MlpProgram;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const SOURCE_LAYER: usize = 60;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];

struct RouteGraph {
    rows: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: DenseFp8MlpProgram,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(root: &Path, repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let program = DenseFp8MlpProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_ROWS, &benchmark_input())?;
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
        let timer = GpuTimer::new(&context)?;

        Ok(Self {
            routes,
            timer,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for rows in EXACT_ROUTES {
                let graph = self.program.qualification_graph(rows)?;
                // SAFETY: this Session's program owns the graph and every
                // allocation it captured, outliving the replay and synchronize.
                unsafe { graph.launch(&self.stream) }?;
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
                let (shape, workload) = if route.rows <= 8 {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_layer_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_operator_prefill(route.rows as u64),
                    )
                };
                Ok(ExactDeviceCase::new(
                    "dense_fp8_mlp/layer60",
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
    (0..MAX_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen38_27B::HIDDEN) & 7]))
        .collect()
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let weights = 6 * hidden + 2 * intermediate * (hidden + 2) + hidden * intermediate;
    let per_token = 22 * hidden + 6 * intermediate + 2 * size_of::<f32>() * 2;

    weights + batch * per_token
}

/// Measures every exact graph of one source-backed dense-FP8 MLP owner.
pub fn benchmark_dense_fp8_mlp(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "dense_fp8_mlp/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=60 source-native norms, gate/up, and down",
    )?;
    memory.register_owned(
        "dense_fp8_mlp/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 exact decode and prefill routes",
    )?;
    memory.register_owned(
        "dense_fp8_mlp/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.program.arena_bytes()
            - session.program.resident_weight_bytes()
            - session.program.workspace_bytes(),
        "single 256-byte-aligned arena",
    )?;
    memory.register_owned(
        "dense_fp8_mlp/address_bound_tensor_maps",
        BenchmarkMemoryKind::Other,
        session.program.descriptor_bytes(),
        "four 128-byte gate/up and down tensor maps",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample)?;
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-dense-fp8-mlp",
            classification: "performance_sensitive_route",
            timing_scope: "paired Rust production-graph submission/completion and repeated whole-MLP production route",
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

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{EXACT_ROUTES, MAX_ROWS, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn dense_fp8_mlp_suite_byte_accounting_covers_the_complete_mlp_graph() {
        let hidden = Qwen38_27B::HIDDEN;
        let intermediate = Qwen38_27B::INTERMEDIATE;
        let weights = 6 * hidden + 2 * intermediate * (hidden + 2) + hidden * intermediate;
        let per_token = 22 * hidden + 6 * intermediate + 2 * size_of::<f32>() * 2;

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_ROWS), weights + MAX_ROWS * per_token);
    }

    #[test]
    fn dense_fp8_mlp_suite_benchmark_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
    }
}

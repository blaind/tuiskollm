//! Direct timing for one source-backed NVFP4 MLP owner.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::oracles::codecs::f32_to_bf16;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Nvfp4MlpProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

const SOURCE_LAYER: usize = 55;
const GROUP: usize = 16;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];

struct RouteGraph {
    rows: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: Nvfp4MlpProgram,
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
        let program = Nvfp4MlpProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
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
                    "nvfp4_mlp/layer55",
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

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let weights = 4 * hidden
        + intermediate * hidden
        + 2 * intermediate * (hidden / GROUP)
        + hidden * (intermediate / 2)
        + hidden * (intermediate / GROUP);
    // Count each represented boundary plane once per producer/consumer.
    let a16_per_token = 20 * hidden + 4 * intermediate;
    let gate_up_w4a4_scratch = hidden + hidden / 8;
    let down_w4a4_scratch = intermediate + intermediate / 8;
    let per_token = a16_per_token
        + if uses_gate_w4a4(rows) {
            gate_up_w4a4_scratch
        } else {
            0
        }
        + if rows > MAX_BATCH {
            down_w4a4_scratch
        } else {
            0
        };

    weights + rows * per_token
}

fn uses_gate_w4a4(rows: usize) -> bool {
    rows > MAX_BATCH || rows == 1 || rows >= 5
}

/// Measures every exact graph of one source-backed NVFP4 MLP owner.
pub fn benchmark_nvfp4_mlp(
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
        "nvfp4_mlp/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=55 source-native norms and losslessly materialized projections",
    )?;
    memory.register_owned(
        "nvfp4_mlp/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_rows=1024 exact decode and prefill scratch and BF16 seams",
    )?;
    memory.register_owned(
        "nvfp4_mlp/alignment_padding",
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
            suite: "bench-nvfp4-mlp",
            classification: "performance_sensitive_route",
            timing_scope: "paired Rust production-graph submission/completion and repeated eager whole-MLP path",
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
    use super::{EXACT_ROUTES, MAX_BATCH, MAX_ROWS, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_both_production_route_kinds() {
        let hidden = Qwen38_27B::HIDDEN;
        let intermediate = Qwen38_27B::INTERMEDIATE;
        let weights = 4 * hidden
            + intermediate * hidden
            + 2 * intermediate * (hidden / 16)
            + hidden * (intermediate / 2)
            + hidden * (intermediate / 16);
        let a16 = 20 * hidden + 4 * intermediate;
        let gate_up_w4a4 = hidden + hidden / 8;
        let down_w4a4 = intermediate + intermediate / 8;

        assert_eq!(logical_bytes(1), weights + a16 + gate_up_w4a4);
        assert_eq!(logical_bytes(2), weights + 2 * a16);
        assert_eq!(
            logical_bytes(MAX_BATCH),
            weights + MAX_BATCH * (a16 + gate_up_w4a4)
        );
        assert_eq!(
            logical_bytes(MAX_ROWS),
            weights + MAX_ROWS * (a16 + gate_up_w4a4 + down_w4a4)
        );
    }

    #[test]
    fn benchmark_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
    }
}

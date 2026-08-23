//! Direct timing for one source-backed Qwen3.5 NVFP4 MLP owner.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, Qwen35Nvfp4MlpProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen35_9B};

const SOURCE_LAYER: usize = 0;
const GROUP: usize = 16;

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    timer: GpuTimer,
    program: Qwen35Nvfp4MlpProgram,
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
        let program = Qwen35Nvfp4MlpProgram::from_snapshot(&context, snapshot, SOURCE_LAYER)?;
        program.load_residual(&stream, MAX_BATCH, &benchmark_input())?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                Ok(RouteGraph {
                    batch,
                    repeated: program.qualification_repeated_graph(
                        &stream,
                        batch,
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
            for batch in 1..=MAX_BATCH {
                self.program
                    .qualification_graph(batch)?
                    .launch(&self.stream)?;
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
                Ok(ExactDeviceCase::new(
                    "qwen35_9b/nvfp4_mlp/layer0",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_layer_decode(route.batch as u32),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    self.program.qualification_graph(route.batch)?,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                ))
            })
            .collect()
    }
}

fn benchmark_input() -> Vec<u16> {
    const PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
    (0..MAX_BATCH * Qwen35_9B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + index / Qwen35_9B::HIDDEN) & 7]))
        .collect()
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen35_9B::HIDDEN;
    let intermediate = Qwen35_9B::INTERMEDIATE;
    let weights = 4 * hidden
        + intermediate * hidden
        + 2 * intermediate * (hidden / GROUP)
        + hidden * (intermediate / 2)
        + hidden * (intermediate / GROUP);
    let a16_per_token = 20 * hidden + 4 * intermediate;
    let w4a4_scratch = hidden + hidden / 8;
    let per_token = if uses_w4a4(batch) {
        a16_per_token + w4a4_scratch
    } else {
        a16_per_token
    };

    weights + batch * per_token
}

fn uses_w4a4(batch: usize) -> bool {
    batch == 1 || batch >= 3
}

/// Measures every exact graph of one source-backed Qwen3.5 NVFP4 MLP owner.
pub fn benchmark_qwen35_nvfp4_mlp(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(root, options.launches_per_sample)?;
    memory.register_owned(
        "qwen35_9b/nvfp4_mlp/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "layer=0 source-native norms and losslessly materialized projections",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_mlp/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8 exact route scratch and BF16 seams",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_mlp/alignment_padding",
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
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-nvfp4-mlp",
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

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, logical_bytes, uses_w4a4};
    use tuisko_model::{Arch, Qwen35_9B};

    #[test]
    fn byte_accounting_covers_both_qwen35_route_kinds() {
        let hidden = Qwen35_9B::HIDDEN;
        let intermediate = Qwen35_9B::INTERMEDIATE;
        let weights = 4 * hidden
            + intermediate * hidden
            + 2 * intermediate * (hidden / 16)
            + hidden * (intermediate / 2)
            + hidden * (intermediate / 16);
        let a16 = 20 * hidden + 4 * intermediate;
        let w4a4 = a16 + hidden + hidden / 8;

        assert_eq!(weights, 84_951_040);
        assert_eq!(logical_bytes(1), weights + w4a4);
        assert_eq!(logical_bytes(2), weights + 2 * a16);
        assert_eq!(logical_bytes(3), weights + 3 * w4a4);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * w4a4);
        assert_eq!(
            (1..=MAX_BATCH).map(uses_w4a4).collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true]
        );
    }
}

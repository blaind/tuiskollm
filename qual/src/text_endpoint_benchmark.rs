//! Paired timings for the resident text endpoint.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{MAX_BATCH, TextEndpointProgram};
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, GpuError, GpuTimer};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

struct RouteGraph {
    batch: usize,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraph>,
    program: TextEndpointProgram,
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
        let mut program = TextEndpointProgram::from_snapshot(&context, snapshot)?;
        let token_ids = benchmark_token_ids();
        program.stage_embeddings(&stream, &token_ids)?;
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

        Ok(Self {
            routes,
            program,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for batch in 1..=MAX_BATCH {
                let graph = self.program.qualification_graph(batch)?;
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
                Ok(ExactDeviceCase::new(
                    "text_endpoint/final_norm_lm_head",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_endpoint_decode(route.batch as u32),
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

fn benchmark_token_ids() -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| ((17 + row * 65_537) % Qwen38_27B::VOCAB) as u32)
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let vocab = Qwen38_27B::VOCAB;
    let weights = 2 * hidden + vocab * hidden + 2 * vocab;
    let per_token = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;

    weights + batch * per_token
}

/// Measures every exact final-norm plus LM-head graph over source weights.
pub fn benchmark_text_endpoint(
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
        "text_endpoint/resident_weights",
        BenchmarkMemoryKind::Weights,
        session.program.resident_weight_bytes(),
        "final norm and source-native LM head",
    )?;
    memory.register_owned(
        "text_endpoint/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.program.workspace_bytes(),
        "max_batch=8",
    )?;
    memory.register_owned(
        "text_endpoint/alignment_padding",
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
            suite: "bench-text-endpoint",
            classification: "performance_sensitive_route",
            timing_scope: "paired Rust production-graph submission/completion and repeated eager endpoint path",
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
    use super::{MAX_BATCH, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_both_endpoint_operations() {
        let hidden = Qwen38_27B::HIDDEN;
        let vocab = Qwen38_27B::VOCAB;
        let weights = 2 * hidden + vocab * hidden + 2 * vocab;
        let per_token = 8 * hidden + 2 * size_of::<f32>() + 2 * vocab;

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * per_token);
    }
}

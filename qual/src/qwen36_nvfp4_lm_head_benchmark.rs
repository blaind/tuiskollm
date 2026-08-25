//! Paired timings for the exact Qwen3.6 NVFP4 LM head.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_nvfp4_lm_head::{MAX_BATCH, Regions, launch, layout, make_fixture};
use crate::target::Qwen36Nvfp4LmHeadOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuTimer};
use tuisko_model::{Arch, Qwen36Moe35B};

struct RouteGraph {
    batch: usize,
    graph: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    arena: DeviceArena,
    routes: Vec<RouteGraph>,
    regions: Regions,
    stream: Arc<CudaStream>,
    _op: Qwen36Nvfp4LmHeadOp,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            )));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let op = Qwen36Nvfp4LmHeadOp::new(&context)?;
        let fixture = make_fixture();
        arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
        arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
        arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                Ok(RouteGraph {
                    batch,
                    graph: CudaGraph::capture(&stream, || {
                        launch(&op, &arena, &stream, regions, batch)
                    })?,
                    repeated: CudaGraph::capture(&stream, || {
                        for _ in 0..repeated_operations {
                            launch(&op, &arena, &stream, regions, batch)?;
                        }
                        Ok(())
                    })?,
                })
            })
            .collect::<Result<Vec<_>, DeviceBenchmarkError>>()?;

        Ok(Self {
            arena,
            routes,
            regions,
            stream,
            _op: op,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> Result<(), DeviceBenchmarkError> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.graph.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)?;

        Ok(())
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    "qwen36_35b_a3b/nvfp4_lm_head/a16",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_endpoint_decode(route.batch as u32),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    &route.graph,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen36Moe35B::HIDDEN;
    let vocab = Qwen36Moe35B::VOCAB;
    let weight_bytes = vocab * (hidden / 2 + hidden / 16);
    weight_bytes + batch * (hidden * size_of::<u16>() + vocab * size_of::<u16>())
}

/// Measures every exact Qwen3.6 NVFP4 LM-head graph.
pub fn benchmark_qwen36_nvfp4_lm_head(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    memory.register_owned(
        "qwen36_35b_a3b/nvfp4_lm_head/weights",
        BenchmarkMemoryKind::Weights,
        session.regions.weight_bytes(),
        "packed E2M1 codes plus swizzled E4M3 block scales",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/nvfp4_lm_head/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.regions.payload_bytes() - session.regions.weight_bytes(),
        "max_batch=8 input and full-vocabulary BF16 logits",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/nvfp4_lm_head/alignment_padding",
        BenchmarkMemoryKind::Other,
        session.arena.byte_len() - session.regions.payload_bytes(),
        "single 256-byte-aligned arena",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen36-nvfp4-lm-head",
            classification: "performance_sensitive_route",
            timing_scope: "paired Rust production-graph submission/completion and repeated eager Qwen3.6 NVFP4 LM-head path",
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
    use super::*;

    #[test]
    fn byte_accounting_covers_source_nvfp4_weights_and_logits() {
        assert_eq!(logical_bytes(1), 286_565_376);
        assert_eq!(logical_bytes(MAX_BATCH), 290_070_528);
    }
}

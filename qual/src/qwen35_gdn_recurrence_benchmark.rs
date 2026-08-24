//! Paired timings for exact Qwen3.5 GDN recurrence routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_gdn_recurrence::{
    HEAD_DIM, MAX_BATCH, Regions, STATE_PER_ROW, VALUE_HEADS, VALUE_WIDTH, launch, layout,
    make_fixture, upload_fixture,
};
use crate::target::Qwen35GdnRecurrenceOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};
use tuisko_model::{Arch, Qwen35_9B};

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

#[derive(Clone, Copy)]
enum Target {
    Qwen35,
    Qwen36,
}

impl Target {
    fn route(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35_9b/gdn_recurrence/state_gate_norm",
            Self::Qwen36 => "qwen36_35b_a3b/gdn_recurrence/state_gate_norm",
        }
    }

    fn suite(self) -> &'static str {
        match self {
            Self::Qwen35 => "bench-qwen35-gdn-recurrence",
            Self::Qwen36 => "bench-qwen36-gdn-recurrence",
        }
    }

    fn weight(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35_9b/gdn_recurrence/norm_weight",
            Self::Qwen36 => "qwen36_35b_a3b/gdn_recurrence/norm_weight",
        }
    }

    fn workspace(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35_9b/gdn_recurrence/address_stable_state_workspace",
            Self::Qwen36 => "qwen36_35b_a3b/gdn_recurrence/address_stable_state_workspace",
        }
    }

    fn padding(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35_9b/gdn_recurrence/alignment_padding",
            Self::Qwen36 => "qwen36_35b_a3b/gdn_recurrence/alignment_padding",
        }
    }
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen35GdnRecurrenceOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
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
        let fixture = make_fixture();
        upload_fixture(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = Qwen35GdnRecurrenceOp::new(&context)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| capture_route(&op, &arena, &stream, regions, batch, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            timer: GpuTimer::new(&context)?,
            _op: op,
            arena,
            regions,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, target: Target, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    target.route(),
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_decode(route.batch as u32),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen35GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, arena, stream, regions, batch)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        batch,
        leaf,
        repeated,
    })
}

fn logical_bytes(batch: usize) -> usize {
    let per_token = Qwen35_9B::GDN_QKV_ROWS * size_of::<u16>()
        + VALUE_WIDTH * size_of::<u16>()
        + 2 * VALUE_HEADS * size_of::<f32>()
        + size_of::<u32>()
        + 2 * STATE_PER_ROW * size_of::<f32>()
        + VALUE_WIDTH * size_of::<u16>();

    HEAD_DIM * size_of::<u16>() + batch * per_token
}

/// Measures every exact Qwen3.5 recurrence route with paired timings.
pub fn benchmark_qwen35_gdn_recurrence(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark(Target::Qwen35, options)
}

/// Measures Qwen3.6 through the shared exact-geometry recurrence routes.
pub fn benchmark_qwen36_gdn_recurrence(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark(Target::Qwen36, options)
}

fn benchmark(
    target: Target,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        target.weight(),
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "one BF16 head-width norm vector",
    )?;
    memory.register_owned(
        target.workspace(),
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "eight FP32 state rows and exact-B workspace",
    )?;
    memory.register_owned(
        target.padding(),
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(target, options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: target.suite(),
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph",
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
    fn accounting_covers_the_complete_state_transition() {
        let per_token = 4_227_332;

        assert_eq!(logical_bytes(1), 256 + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), 256 + MAX_BATCH * per_token);
    }
}

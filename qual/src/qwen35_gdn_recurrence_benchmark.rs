//! Paired timings for exact Qwen3.5/Qwen3.6 GDN recurrence routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_gdn_recurrence::{
    EXACT_ROUTES, HEAD_DIM, MAX_BATCH, Regions, STATE_PER_ROW, VALUE_HEADS, VALUE_WIDTH, launch,
    layout, make_fixture, upload_fixture,
};
use crate::target::Qwen35GdnRecurrenceOp;
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer,
    PinnedHostBuffer,
};
use tuisko_model::{Arch, Qwen35_9B};

struct RouteGraphs {
    rows: usize,
    preparation: CudaGraph,
    leaf: CudaGraph,
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
    _state_seed: PinnedHostBuffer<f32>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new() -> Result<Self, DeviceBenchmarkError> {
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
        let state_seed =
            PinnedHostBuffer::from_slice(&context, &fixture.state).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = Qwen35GdnRecurrenceOp::new(&context)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| capture_route(&op, &arena, &stream, regions, &state_seed, rows))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            timer: GpuTimer::new(&context)?,
            _op: op,
            arena,
            regions,
            _state_seed: state_seed,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.preparation.launch(&self.stream) }?;
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, target: Target) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                let (shape, workload) = if route.rows <= MAX_BATCH {
                    (
                        format!("B={}", route.rows),
                        BenchmarkWorkload::warm_operator_decode(route.rows as u32),
                    )
                } else {
                    (
                        format!("T={}", route.rows),
                        BenchmarkWorkload::warm_operator_prefill(route.rows as u64),
                    )
                };
                ExactDeviceCase::new(
                    target.route(),
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    None,
                )
                .with_preparation(&route.preparation)
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen35GdnRecurrenceOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    state_seed: &PinnedHostBuffer<f32>,
    rows: usize,
) -> GpuResult<RouteGraphs> {
    let preparation = CudaGraph::capture(stream, || {
        // SAFETY: Session owns the immutable pinned seed through every graph
        // replay, and the region covers the complete eight-row state owner.
        unsafe {
            arena.copy_prefix_from_pinned_host_async(
                stream,
                regions.state,
                state_seed,
                state_seed.len(),
            )
        }
    })?;
    let leaf = CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows))?;

    Ok(RouteGraphs {
        rows,
        preparation,
        leaf,
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
    let session = Session::new()?;
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
        "eight FP32 state rows and max_rows=128 workspace",
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
    let cases = session.cases(target);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: target.suite(),
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion and production graph after untimed exact-state restore",
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
    use crate::qwen35_gdn_recurrence::MAX_ROWS;

    #[test]
    fn accounting_covers_the_complete_state_transition() {
        let per_token = 4_227_332;

        assert_eq!(logical_bytes(1), 256 + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), 256 + MAX_BATCH * per_token);
        assert_eq!(logical_bytes(MAX_ROWS), 256 + MAX_ROWS * per_token);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
    }
}

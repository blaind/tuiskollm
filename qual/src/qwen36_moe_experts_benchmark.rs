//! Paired timings for every exact Qwen3.6 routed/shared expert route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_moe_experts::{
    HIDDEN, INTERMEDIATE, MAX_BATCH, Regions, SLOTS, TOP_K, copy_fixture, launch, layout,
    make_fixture,
};
use crate::target::Qwen36MoeExpertsOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

const GATE_UP_CODE_BYTES: usize = 2 * INTERMEDIATE * HIDDEN / 2;
const GATE_UP_SCALE_BYTES: usize = 2 * INTERMEDIATE * (HIDDEN / 16);
const DOWN_CODE_BYTES: usize = HIDDEN * INTERMEDIATE / 2;
const DOWN_SCALE_BYTES: usize = HIDDEN * (INTERMEDIATE / 16);

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen36MoeExpertsOp,
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
        copy_fixture(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = Qwen36MoeExpertsOp::new(&context)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                capture_route(
                    &op,
                    &arena,
                    &stream,
                    regions,
                    batch,
                    repeated_operations,
                    &fixture,
                )
            })
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

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    "qwen36_35b_a3b/moe_experts/nvfp4_top8_shared",
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

#[allow(clippy::too_many_arguments)]
fn capture_route(
    op: &Qwen36MoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
    repeated_operations: u64,
    fixture: &crate::qwen36_moe_experts::Fixture,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || {
        launch(op, arena, stream, regions, batch, fixture)
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, arena, stream, regions, batch, fixture)?;
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
    let selected_weight_bytes = batch
        * SLOTS
        * (GATE_UP_CODE_BYTES + GATE_UP_SCALE_BYTES + DOWN_CODE_BYTES + DOWN_SCALE_BYTES);
    let input = batch * HIDDEN * size_of::<u16>();
    let routing = batch * TOP_K * 2 * size_of::<u16>();
    let shared_gate_weight = batch * HIDDEN * size_of::<u16>();
    let intermediate = batch * SLOTS * INTERMEDIATE * 2 * size_of::<u16>();
    let expert_output = batch * SLOTS * HIDDEN * 2 * size_of::<u16>();
    let output = batch * HIDDEN * size_of::<u16>();

    selected_weight_bytes
        + input
        + routing
        + shared_gate_weight
        + intermediate
        + expert_output
        + output
}

/// Measures every exact Qwen3.6 routed/shared expert and combine route.
pub fn benchmark_qwen36_moe_experts(
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
        "qwen36_35b_a3b/moe_experts/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "256 numeric-order routed experts, one shared expert, and shared-gate BF16 weights",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/moe_experts/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_batch=8 input, top-eight routing, nine expert slots, and combined output",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/moe_experts/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen36-moe-experts",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, repeated leaf graph, and repeated-operation graph",
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
    fn accounting_covers_selected_experts_and_every_workspace_seam() {
        assert_eq!(logical_bytes(1), 16_029_728);
        assert_eq!(logical_bytes(MAX_BATCH), 128_237_824);

        let (layout, regions) = layout().unwrap();
        assert_eq!(regions.weight_bytes(), 454_760_448);
        assert_eq!(layout.byte_len(), 455_195_392);
    }
}

//! Paired timings for every exact Qwen3.6 MoE router route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_moe_router::{
    EXACT_ROUTES, EXPERTS, HIDDEN, MAX_BATCH, Regions, TOP_K, launch, layout, make_fixture,
};
use crate::target::Qwen36MoeRouterOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen36MoeRouterOp,
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
        arena.copy_from_host(&stream, regions.input, &fixture.input)?;
        arena.copy_from_host(&stream, regions.weights, &fixture.weights)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen36MoeRouterOp::new(&context)?;
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| capture_route(&op, &arena, &stream, regions, rows, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
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
                    "qwen36_35b_a3b/moe_router/bf16_top8",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen36MoeRouterOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, arena, stream, regions, rows)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        leaf,
        repeated,
    })
}

fn logical_bytes(rows: usize) -> usize {
    let weights = EXPERTS * HIDDEN * size_of::<u16>();
    let input = rows * HIDDEN * size_of::<u16>();
    let logits = rows * EXPERTS * size_of::<u16>();
    let selected = rows * TOP_K * 2 * size_of::<u16>();

    weights + input + logits + selected
}

/// Measures every exact Qwen3.6 BF16 router and top-eight selection route.
pub fn benchmark_qwen36_moe_router(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen36_35b_a3b/moe_router/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source BF16 [256,2048] router matrix",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/moe_router/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=128 logits, top-eight indices, and normalized weights",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/moe_router/alignment_padding",
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
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen36-moe-router",
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
    use crate::qwen36_moe_router::MAX_ROWS;

    #[test]
    fn accounting_covers_router_projection_and_selection() {
        let (layout, regions) = layout().unwrap();
        let weights = EXPERTS * HIDDEN * size_of::<u16>();
        let b8 = weights
            + MAX_BATCH
                * (HIDDEN * size_of::<u16>()
                    + EXPERTS * size_of::<u16>()
                    + 2 * TOP_K * size_of::<u16>());

        assert_eq!(logical_bytes(MAX_BATCH), b8);
        assert_eq!(logical_bytes(MAX_ROWS), 1_642_496);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(layout.byte_len(), 1_642_496);
        assert_eq!(regions.weight_bytes(), 1_048_576);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 593_920);
    }
}

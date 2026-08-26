//! Paired timings for exact Qwen3.5 NVFP4 SwiGLU routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_nvfp4_swiglu::{
    CODE_BYTES_PER_ROW, GATE_UP_ROWS, GROUPS_PER_ROW, HIDDEN, INPUT_SCALE_DIVISOR, MAX_BATCH,
    OUTPUT_ROWS, PREFILL_ROWS, Regions, WEIGHT_SCALE_DIVISOR, layout, make_fixture,
};
use crate::target::Qwen35Nvfp4SwiGluOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

#[derive(Clone, Copy)]
enum Schedule {
    A16,
    W4a4,
}

impl Schedule {
    fn route(self) -> &'static str {
        match self {
            Self::A16 => "qwen35_9b/nvfp4_swiglu/a16",
            Self::W4a4 => "qwen35_9b/nvfp4_swiglu/w4a4",
        }
    }
}

struct RouteGraphs {
    rows: usize,
    schedule: Schedule,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Addresses {
    input: *const u16,
    activation_codes: *mut u8,
    activation_scales: *mut u8,
    weight_codes: *const u8,
    weight_scales: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen35Nvfp4SwiGluOp,
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
        let fixture = make_fixture().map_err(|error| {
            DeviceBenchmarkError::Precondition(format!(
                "Qwen3.5 benchmark fixture construction failed: {error}"
            ))
        })?;
        arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
        arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
        arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = Qwen35Nvfp4SwiGluOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        let mut routes = Vec::with_capacity(16);
        for batch in 1..=4 {
            routes.push(capture_route(
                &op,
                &stream,
                &addresses,
                batch,
                Schedule::A16,
                repeated_operations,
            )?);
        }
        for batch in 1..=MAX_BATCH {
            routes.push(capture_route(
                &op,
                &stream,
                &addresses,
                batch,
                Schedule::W4a4,
                repeated_operations,
            )?);
        }
        for rows in PREFILL_ROWS {
            routes.push(capture_route(
                &op,
                &stream,
                &addresses,
                rows,
                Schedule::W4a4,
                repeated_operations,
            )?);
        }

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
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
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
                    route.schedule.route(),
                    shape,
                    workload,
                    OperationAccounting::new(
                        logical_bytes(route.rows, route.schedule),
                        route.rows as u64,
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
    op: &Qwen35Nvfp4SwiGluOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
    schedule: Schedule,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, rows, schedule))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, rows, schedule)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        schedule,
        leaf,
        repeated,
    })
}

fn launch(
    op: &Qwen35Nvfp4SwiGluOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
    schedule: Schedule,
) -> GpuResult<()> {
    // SAFETY: each address names its aligned maximum-batch arena region.
    unsafe {
        match (schedule, rows <= MAX_BATCH) {
            (Schedule::A16, _) => op.launch_a16(
                stream,
                rows,
                addresses.input,
                addresses.weight_codes,
                addresses.weight_scales,
                WEIGHT_SCALE_DIVISOR,
                addresses.output,
            ),
            (Schedule::W4a4, true) => op.launch_w4a4(
                stream,
                rows,
                addresses.input,
                addresses.activation_codes,
                addresses.activation_scales,
                addresses.weight_codes,
                addresses.weight_scales,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                addresses.output,
            ),
            (Schedule::W4a4, false) => op.launch_prefill(
                stream,
                rows,
                addresses.input,
                addresses.activation_codes,
                addresses.activation_scales,
                addresses.weight_codes,
                addresses.weight_scales,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                addresses.output,
            ),
        }
    }
}

fn logical_bytes(rows: usize, schedule: Schedule) -> usize {
    let weights = GATE_UP_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let input = rows * HIDDEN * size_of::<u16>();
    let output = rows * OUTPUT_ROWS * size_of::<u16>();
    let scratch = match schedule {
        Schedule::A16 => 0,
        Schedule::W4a4 => 2 * rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW),
    };

    weights + input + output + scratch
}

/// Measures all qualified Qwen3.5 A16/W4A4 routes.
pub fn benchmark_qwen35_nvfp4_swiglu(
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
        "qwen35_9b/nvfp4_swiglu/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "packed fused gate/up plus swizzled block scales",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_swiglu/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=1024",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_swiglu/alignment_padding",
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
            suite: "bench-qwen35-nvfp4-swiglu",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired candidate CUDA graphs and repeated-operation graphs",
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
    fn accounting_covers_exact_qwen35_candidates() {
        let (layout, regions) = layout().unwrap();
        let weights = GATE_UP_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);

        assert_eq!(layout.byte_len(), 92_536_832);
        assert_eq!(regions.weight_bytes(), weights);
        assert_eq!(
            logical_bytes(4, Schedule::A16),
            weights + 4 * (HIDDEN + OUTPUT_ROWS) * 2
        );
        assert_eq!(
            logical_bytes(8, Schedule::W4a4),
            weights + 8 * (HIDDEN + OUTPUT_ROWS) * 2 + 16 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
        assert_eq!(
            logical_bytes(1_024, Schedule::W4a4),
            weights
                + 1_024 * (HIDDEN + OUTPUT_ROWS) * 2
                + 2_048 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
        );
    }
}

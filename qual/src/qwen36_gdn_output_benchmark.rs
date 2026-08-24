//! Paired timings for every exact Qwen3.6 GDN output route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_gdn_output::{
    EXACT_ROUTES, INPUT_COLUMNS, INPUT_SCALE, MAX_BATCH, OUTPUT_ROWS, Regions, WEIGHT_SCALE,
    layout, make_fixture,
};
use crate::target::Qwen36GdnOutputOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Addresses {
    input: *const u16,
    activation_codes: *mut u8,
    weight_codes: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen36GdnOutputOp,
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
            DeviceBenchmarkError::Precondition(format!("building Qwen3.6 fixture: {error}"))
        })?;
        arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
        arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen36GdnOutputOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            weight_codes: arena.address(regions.weight_codes)?,
            output: arena.address(regions.output)?,
        };
        let routes = EXACT_ROUTES
            .iter()
            .map(|&rows| capture_route(&op, &stream, &addresses, rows, repeated_operations))
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
                    "qwen36_35b_a3b/gdn_output/static_fp8",
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
    op: &Qwen36GdnOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, rows)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        rows,
        leaf,
        repeated,
    })
}

fn launch(
    op: &Qwen36GdnOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            rows,
            addresses.input,
            addresses.activation_codes,
            INPUT_SCALE,
            addresses.weight_codes,
            WEIGHT_SCALE,
            addresses.output,
        )
    }
}

fn logical_bytes(rows: usize) -> usize {
    let weights = OUTPUT_ROWS * INPUT_COLUMNS;
    let input = rows * INPUT_COLUMNS * size_of::<u16>();
    let activation_codes = rows * INPUT_COLUMNS;
    let output = rows * OUTPUT_ROWS * size_of::<u16>();

    weights + input + activation_codes + output
}

/// Measures every exact Qwen3.6 static-FP8 GDN output route.
pub fn benchmark_qwen36_gdn_output(
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
        "qwen36_35b_a3b/gdn_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source E4M3 [2048,4096] output matrix",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=128 BF16 input, static codes, and BF16 output",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_output/alignment_padding",
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
            suite: "bench-qwen36-gdn-output",
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
    fn accounting_covers_static_quantize_and_projection() {
        let (layout, regions) = layout().unwrap();
        let max_rows = regions.weight_bytes()
            + 128
                * (INPUT_COLUMNS * (size_of::<u16>() + size_of::<u8>())
                    + OUTPUT_ROWS * size_of::<u16>());

        assert_eq!(logical_bytes(128), max_rows);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert_eq!(layout.byte_len(), 10_485_760);
        assert_eq!(regions.weight_bytes(), 8_388_608);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 2_097_152);
    }
}

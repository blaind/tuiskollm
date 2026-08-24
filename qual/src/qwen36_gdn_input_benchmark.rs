//! Paired timings for every exact Qwen3.6 GDN input route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_gdn_input::{
    CONTROL_ROWS, INPUT_COLUMNS, INPUT_SCALE, MAX_BATCH, PROJECTED_ROWS, QKV_WEIGHT_SCALE, Regions,
    Z_WEIGHT_SCALE, layout, make_fixture,
};
use crate::target::Qwen36GdnInputOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Addresses {
    input: *const u16,
    activation_codes: *mut u8,
    projected_weight_codes: *const u8,
    control_weight_bf16: *const u16,
    projected_output: *mut u16,
    control_output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen36GdnInputOp,
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
        arena.copy_from_host(
            &stream,
            regions.projected_weight_codes,
            &fixture.projected_weight_codes,
        )?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_bf16,
            &fixture.control_weight_bf16,
        )?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen36GdnInputOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            projected_weight_codes: arena.address(regions.projected_weight_codes)?,
            control_weight_bf16: arena.address(regions.control_weight_bf16)?,
            projected_output: arena.address(regions.projected_output)?,
            control_output: arena.address(regions.control_output)?,
        };
        let routes = (1..=MAX_BATCH)
            .map(|batch| capture_route(&op, &stream, &addresses, batch, repeated_operations))
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
                    "qwen36_35b_a3b/gdn_input/static_fp8_bf16",
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
    op: &Qwen36GdnInputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, batch))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, stream, addresses, batch)?;
        }
        Ok(())
    })?;

    Ok(RouteGraphs {
        batch,
        leaf,
        repeated,
    })
}

fn launch(
    op: &Qwen36GdnInputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.input,
            addresses.activation_codes,
            INPUT_SCALE,
            addresses.projected_weight_codes,
            QKV_WEIGHT_SCALE,
            Z_WEIGHT_SCALE,
            addresses.control_weight_bf16,
            addresses.projected_output,
            addresses.control_output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = PROJECTED_ROWS * INPUT_COLUMNS + CONTROL_ROWS * INPUT_COLUMNS * size_of::<u16>();
    let input = batch * INPUT_COLUMNS * size_of::<u16>();
    let activation_codes = batch * INPUT_COLUMNS;
    let output = batch * (PROJECTED_ROWS + CONTROL_ROWS) * size_of::<u16>();

    weights + input + activation_codes + output
}

/// Measures every exact Qwen3.6 static-FP8/BF16 GDN input route.
pub fn benchmark_qwen36_gdn_input(
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
        "qwen36_35b_a3b/gdn_input/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source E4M3 Q/K/V/Z and BF16 A/B control planes",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_input/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_batch=8 static codes and BF16 outputs",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/gdn_input/alignment_padding",
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
            suite: "bench-qwen36-gdn-input",
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
    fn accounting_covers_static_quantize_and_both_projection_families() {
        let (layout, regions) = layout().unwrap();
        let b8 = regions.weight_bytes()
            + MAX_BATCH
                * (INPUT_COLUMNS * (size_of::<u16>() + size_of::<u8>())
                    + (PROJECTED_ROWS + CONTROL_ROWS) * size_of::<u16>());

        assert_eq!(logical_bytes(MAX_BATCH), b8);
        assert_eq!(layout.byte_len(), 25_674_752);
        assert_eq!(regions.weight_bytes(), 25_427_968);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 246_784);
    }
}

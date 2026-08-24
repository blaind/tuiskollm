//! Paired timings for every exact Qwen3.5 NVFP4 GDN input route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_nvfp4_gdn_input::{
    CODE_BYTES_PER_ROW, CONTROL_WEIGHT_SCALE_DIVISOR, EXACT_ROUTES, GROUPS_PER_ROW, INPUT_COLUMNS,
    INPUT_SCALE_DIVISOR, MAX_BATCH, PADDED_CONTROL_ROWS, PROJECTED_ROWS,
    PROJECTED_WEIGHT_SCALE_DIVISOR, Regions, layout, make_fixture,
};
use crate::target::Qwen35Nvfp4GdnInputOp;
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
    activation_scales: *mut u8,
    projected_weight_codes: *const u8,
    projected_weight_scales: *const u8,
    control_weight_codes: *const u8,
    control_weight_scales: *const u8,
    projected_output: *mut u16,
    control_output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen35Nvfp4GdnInputOp,
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
                "Qwen3.5 GDN input fixture construction failed: {error}"
            ))
        })?;
        arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
        arena.copy_from_host(
            &stream,
            regions.projected_weight_codes,
            &fixture.projected_weight_codes,
        )?;
        arena.copy_from_host(
            &stream,
            regions.projected_weight_scales,
            &fixture.projected_weight_scales,
        )?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_codes,
            &fixture.control_weight_codes,
        )?;
        arena.copy_from_host(
            &stream,
            regions.control_weight_scales,
            &fixture.control_weight_scales,
        )?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen35Nvfp4GdnInputOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            projected_weight_codes: arena.address(regions.projected_weight_codes)?,
            projected_weight_scales: arena.address(regions.projected_weight_scales)?,
            control_weight_codes: arena.address(regions.control_weight_codes)?,
            control_weight_scales: arena.address(regions.control_weight_scales)?,
            projected_output: arena.address(regions.projected_output)?,
            control_output: arena.address(regions.control_output)?,
        };
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| capture_route(&op, &stream, &addresses, rows, repeated_operations))
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
                    "qwen35_9b/nvfp4_gdn_input/production",
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
    op: &Qwen35Nvfp4GdnInputOp,
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
    op: &Qwen35Nvfp4GdnInputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        if rows <= MAX_BATCH {
            op.launch(
                stream,
                rows,
                addresses.input,
                addresses.projected_weight_codes,
                addresses.projected_weight_scales,
                PROJECTED_WEIGHT_SCALE_DIVISOR,
                addresses.control_weight_codes,
                addresses.control_weight_scales,
                CONTROL_WEIGHT_SCALE_DIVISOR,
                addresses.projected_output,
                addresses.control_output,
            )
        } else {
            op.launch_prefill(
                stream,
                rows,
                addresses.input,
                addresses.activation_codes,
                addresses.activation_scales,
                addresses.projected_weight_codes,
                addresses.projected_weight_scales,
                PROJECTED_WEIGHT_SCALE_DIVISOR,
                addresses.control_weight_codes,
                addresses.control_weight_scales,
                CONTROL_WEIGHT_SCALE_DIVISOR,
                INPUT_SCALE_DIVISOR,
                addresses.projected_output,
                addresses.control_output,
            )
        }
    }
}

fn logical_bytes(rows: usize) -> usize {
    let weights = (PROJECTED_ROWS + PADDED_CONTROL_ROWS) * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let input = rows * INPUT_COLUMNS * size_of::<u16>();
    let output = rows * (PROJECTED_ROWS + PADDED_CONTROL_ROWS) * size_of::<u16>();
    let scratch = if rows > MAX_BATCH {
        2 * rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    } else {
        0
    };

    weights + input + output + scratch
}

/// Measures every exact Qwen3.5 NVFP4 GDN input route with paired timings.
pub fn benchmark_qwen35_nvfp4_gdn_input(
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
        "qwen35_9b/nvfp4_gdn_input/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "packed Q/K/V/Z and padded A/B controls plus swizzled block scales",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_gdn_input/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=128 activation quantization and both output seams",
    )?;
    memory.register_owned(
        "qwen35_9b/nvfp4_gdn_input/alignment_padding",
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
            suite: "bench-qwen35-nvfp4-gdn-input",
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
    fn accounting_covers_every_qwen35_gdn_input_decode_and_prefill_byte() {
        let (layout, regions) = layout().unwrap();
        let weights =
            (PROJECTED_ROWS + PADDED_CONTROL_ROWS) * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
        let t128 = weights
            + crate::qwen35_nvfp4_gdn_input::MAX_ROWS
                * (INPUT_COLUMNS * size_of::<u16>()
                    + (PROJECTED_ROWS + PADDED_CONTROL_ROWS) * size_of::<u16>()
                    + 2 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW));

        assert_eq!(logical_bytes(crate::qwen35_nvfp4_gdn_input::MAX_ROWS), t128);
        assert_eq!(layout.byte_len(), 33_128_448);
        assert_eq!(regions.weight_bytes(), 28_606_464);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 4_521_984);
    }
}

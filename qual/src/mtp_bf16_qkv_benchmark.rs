//! Direct timings for every exact source-BF16 MTP Q/gate/K/V route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use crate::mtp_bf16_qkv::{MAX_BATCH, OUTPUT_ROWS, Regions, layout};
use crate::target::MtpBf16QkvOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};
use tuisko_model::{Arch, Qwen38_27B};

const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.5, -0.375, 0.25, -0.125, 0.0625, -0.03125];

struct Addresses {
    input: *const u16,
    weight: *const u16,
    output: *mut u16,
}

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: MtpBf16QkvOp,
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
        load_fixture(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = MtpBf16QkvOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            weight: arena.address(regions.weight)?,
            output: arena.address(regions.output)?,
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
                route.leaf.launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    "qwen3_8/mtp/bf16_qkv",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_mtp(route.batch as u64),
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

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let input = (0..MAX_BATCH * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(INPUT_PATTERN[(index * 3 + index / Qwen38_27B::HIDDEN) & 7]))
        .collect::<Vec<_>>();
    // A nonzero BF16 plane forces every gathered source-shaped word through the timed path.
    let weight = vec![f32_to_bf16(0.015625); OUTPUT_ROWS * Qwen38_27B::HIDDEN];

    arena.copy_from_host(stream, regions.input, &input)?;
    arena.copy_from_host(stream, regions.weight, &weight)
}

fn capture_route(
    op: &MtpBf16QkvOp,
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
    op: &MtpBf16QkvOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: production op receives the complete aligned max-B arena and source-shaped weight.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.input,
            addresses.weight,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weight = OUTPUT_ROWS * Qwen38_27B::HIDDEN * size_of::<u16>();
    let per_token = (Qwen38_27B::HIDDEN + OUTPUT_ROWS) * size_of::<u16>();

    weight + batch * per_token
}

/// Measures every exact `B=1..=8` MTP BF16 Q/gate/K/V graph with production allocations.
pub fn benchmark_mtp_bf16_qkv(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen3_8/mtp/bf16_qkv/weights",
        BenchmarkMemoryKind::Weights,
        session.regions.weight_bytes(),
        "losslessly gathered [14336,5120] source-BF16 Q/gate/K/V plane",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_qkv/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.regions.workspace_bytes(),
        "max_batch=8 normalized input and fused BF16 output",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_qkv/alignment_padding",
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
            suite: "bench-mtp-bf16-qkv",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, complete production graph, and repeated-operation graph for gathered source-BF16 Q/gate/K/V projection",
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
    use super::{MAX_BATCH, layout, logical_bytes};

    #[test]
    fn mtp_bf16_qkv_suite_benchmark_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 146_800_640);
        assert_eq!(regions.workspace_bytes(), 311_296);
        assert_eq!(regions.payload_bytes(), 147_111_936);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(logical_bytes(1), 146_839_552);
        assert_eq!(logical_bytes(MAX_BATCH), 147_111_936);
    }
}

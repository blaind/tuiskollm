//! Direct timings for every exact source-BF16 MTP MLP route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::mtp_bf16_mlp::{HIDDEN, INTERMEDIATE, MAX_BATCH, Regions, layout};
use crate::oracles::codecs::f32_to_bf16;
use crate::target::MtpBf16MlpOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};

const INPUT_VALUE: f32 = 0.03125;
const WEIGHT_VALUE: f32 = 0.000_976_562_5;

struct Addresses {
    input: *const u16,
    gate_up_weight: *const u16,
    activation: *mut u16,
    down_weight: *const u16,
    output: *mut u16,
}

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: MtpBf16MlpOp,
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
        let op = MtpBf16MlpOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            gate_up_weight: arena.address(regions.gate_up_weight)?,
            activation: arena.address(regions.activation)?,
            down_weight: arena.address(regions.down_weight)?,
            output: arena.address(regions.output)?,
        };
        let routes = (1..=MAX_BATCH)
            .map(|batch| capture_route(&op, &stream, &addresses, batch, repeated_operations))
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
        for route in &self.routes {
            // SAFETY: this Session owns the repeated route graph and every captured
            // allocation until after this synchronized replay.
            unsafe { route.repeated.launch(&self.stream) }?;
        }
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
                ExactDeviceCase::new(
                    "qwen3_8/mtp/bf16_mlp",
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
    let input = vec![f32_to_bf16(INPUT_VALUE); MAX_BATCH * HIDDEN];
    // Both source-shaped matrices are nonzero so the timed route must consume
    // every represented weight byte and publish a finite nonzero down result.
    let gate_up_weight = vec![f32_to_bf16(WEIGHT_VALUE); 2 * INTERMEDIATE * HIDDEN];
    let down_weight = vec![f32_to_bf16(WEIGHT_VALUE); HIDDEN * INTERMEDIATE];

    arena.copy_from_host(stream, regions.input, &input)?;
    arena.copy_from_host(stream, regions.gate_up_weight, &gate_up_weight)?;
    arena.copy_from_host(stream, regions.down_weight, &down_weight)
}

fn capture_route(
    op: &MtpBf16MlpOp,
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
    op: &MtpBf16MlpOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: the production op receives the complete aligned max-B arena.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.input,
            addresses.gate_up_weight,
            addresses.activation,
            addresses.down_weight,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * size_of::<u16>();
    let per_token = 4 * (HIDDEN + INTERMEDIATE);

    weights + batch * per_token
}

/// Measures every exact source-BF16 MTP MLP production graph.
pub fn benchmark_mtp_bf16_mlp(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen3_8/mtp/bf16_mlp/weights",
        BenchmarkMemoryKind::Weights,
        session.regions.weight_bytes(),
        "unchanged [34816,5120] gate/up and [5120,17408] down source-BF16 matrices",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_mlp/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.regions.workspace_bytes(),
        "max_batch=8 input, represented SwiGLU activation, and down output",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_mlp/alignment_padding",
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
            suite: "bench-mtp-bf16-mlp",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, complete production graph, and repeated-operation graph for source-BF16 gate/up SwiGLU and down projection",
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
    fn mtp_bf16_mlp_suite_benchmark_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 534_773_760);
        assert_eq!(regions.workspace_bytes(), 442_368);
        assert_eq!(regions.payload_bytes(), 535_216_128);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(logical_bytes(1), 534_863_872);
        assert_eq!(logical_bytes(MAX_BATCH), 535_494_656);
    }
}

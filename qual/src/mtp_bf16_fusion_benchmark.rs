//! Direct timings for every exact source-BF16 MTP fusion route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use crate::mtp_bf16_fusion::{Regions, layout};
use crate::target::MtpBf16FusionOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.5, -0.375, 0.25, -0.125, 0.0625, -0.03125];

struct Addresses {
    embedding: *const u16,
    hidden: *const u16,
    embedding_norm_weight: *const u16,
    hidden_norm_weight: *const u16,
    normalized_embedding: *mut u16,
    normalized_hidden: *mut u16,
    projection_weight: *const u16,
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
    _op: MtpBf16FusionOp,
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
        let op = MtpBf16FusionOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
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
                    "qwen3_8/mtp/bf16_input_fusion",
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
    let hidden = Qwen38_27B::HIDDEN;
    let embedding = (0..MAX_BATCH * hidden)
        .map(|index| f32_to_bf16(INPUT_PATTERN[(index + index / hidden) & 7]))
        .collect::<Vec<_>>();
    let hidden_input = (0..MAX_BATCH * hidden)
        .map(|index| f32_to_bf16(INPUT_PATTERN[(index * 3 + index / hidden + 1) & 7]))
        .collect::<Vec<_>>();
    let norm = vec![0u16; hidden];
    // A nonzero represented BF16 plane forces the timed production kernel to read
    // every source word while keeping setup deterministic and snapshot-independent.
    let projection = vec![f32_to_bf16(0.015625); hidden * 2 * hidden];

    arena.copy_from_host(stream, regions.embedding, &embedding)?;
    arena.copy_from_host(stream, regions.hidden, &hidden_input)?;
    arena.copy_from_host(stream, regions.embedding_norm_weight, &norm)?;
    arena.copy_from_host(stream, regions.hidden_norm_weight, &norm)?;
    arena.copy_from_host(stream, regions.projection_weight, &projection)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        embedding: arena.address(regions.embedding)?,
        hidden: arena.address(regions.hidden)?,
        embedding_norm_weight: arena.address(regions.embedding_norm_weight)?,
        hidden_norm_weight: arena.address(regions.hidden_norm_weight)?,
        normalized_embedding: arena.address(regions.normalized_embedding)?,
        normalized_hidden: arena.address(regions.normalized_hidden)?,
        projection_weight: arena.address(regions.projection_weight)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &MtpBf16FusionOp,
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
    op: &MtpBf16FusionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: the benchmark uses the production op with the complete aligned
    // max-B allocations and immutable source-shaped weights.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.embedding,
            addresses.hidden,
            addresses.embedding_norm_weight,
            addresses.hidden_norm_weight,
            addresses.normalized_embedding,
            addresses.normalized_hidden,
            addresses.projection_weight,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let weights = (2 * hidden + hidden * 2 * hidden) * size_of::<u16>();
    // Per token: two raw reads, two normalized writes, two normalized projection
    // reads, and one projected write. Matrix and norm weights are counted once.
    let per_token = 7 * hidden * size_of::<u16>();

    weights + batch * per_token
}

/// Measures every exact `B=1..=8` MTP BF16 fusion graph with production allocations.
pub fn benchmark_mtp_bf16_fusion(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen3_8/mtp/bf16_fusion/weights",
        BenchmarkMemoryKind::Weights,
        session.regions.weight_bytes(),
        "two source-BF16 zero-centered norms and [5120,10240] source-BF16 projection",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_fusion/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.regions.workspace_bytes(),
        "max_batch=8 embedding, hidden, both normalized seams, and projected output",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_fusion/alignment_padding",
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
            suite: "bench-mtp-bf16-fusion",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, complete production graph, and repeated-operation graph for two BF16 norms plus source-BF16 fusion projection",
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
    fn mtp_bf16_fusion_suite_benchmark_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 104_878_080);
        assert_eq!(regions.workspace_bytes(), 409_600);
        assert_eq!(regions.payload_bytes(), 105_287_680);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(logical_bytes(1), 104_949_760);
        assert_eq!(logical_bytes(MAX_BATCH), 105_451_520);
    }
}

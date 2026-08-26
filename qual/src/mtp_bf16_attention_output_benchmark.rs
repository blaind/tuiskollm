//! Direct timings for every exact gated source-BF16 MTP attention-output route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use crate::mtp_bf16_attention_output::{COLUMNS, MAX_BATCH, OUTPUT_ROWS, Regions, layout};
use crate::target::MtpBf16AttentionOutputOp;
use std::sync::Arc;
use tuisko_gpu::{CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer};
use tuisko_model::{Arch, Qwen38_27B};

const ATTENTION_VALUES: [f32; 8] = [1.0, -0.875, 0.75, -0.625, 0.5, -0.375, 0.25, -0.125];
const REPLAY_STABLE_GATE: f32 = 128.0;

struct Addresses {
    attention: *mut f32,
    qkv: *const u16,
    activation: *mut u16,
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
    _op: MtpBf16AttentionOutputOp,
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
        let op = MtpBf16AttentionOutputOp::new(&context)?;
        let addresses = Addresses {
            attention: arena.address(regions.attention)?,
            qkv: arena.address(regions.qkv)?,
            activation: arena.address(regions.activation)?,
            weight: arena.address(regions.weight)?,
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
                    "qwen3_8/mtp/bf16_attention_output",
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
    let attention = (0..MAX_BATCH * COLUMNS)
        .map(|index| ATTENTION_VALUES[(index + index / COLUMNS) & 7])
        .collect::<Vec<_>>();
    let mut qkv = vec![0u16; MAX_BATCH * Qwen38_27B::ATTENTION_QKV_ROWS];
    // This positive gate rounds sigmoid to one. Repeated production graphs
    // therefore retain identical in-place FP32 inputs without timed restoration.
    for token in 0..MAX_BATCH {
        for head in 0..Qwen38_27B::NUM_ATTENTION_HEADS {
            for dimension in 0..Qwen38_27B::HEAD_DIM {
                let gate = token * Qwen38_27B::ATTENTION_QKV_ROWS
                    + head * 2 * Qwen38_27B::HEAD_DIM
                    + Qwen38_27B::HEAD_DIM
                    + dimension;
                qkv[gate] = f32_to_bf16(REPLAY_STABLE_GATE);
            }
        }
    }
    // A nonzero source-shaped BF16 plane forces all 60 MiB through the timed path.
    let weight = vec![f32_to_bf16(0.015625); OUTPUT_ROWS * COLUMNS];

    arena.copy_from_host(stream, regions.attention, &attention)?;
    arena.copy_from_host(stream, regions.qkv, &qkv)?;
    arena.copy_from_host(stream, regions.weight, &weight)
}

fn capture_route(
    op: &MtpBf16AttentionOutputOp,
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
    op: &MtpBf16AttentionOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: production op receives the complete aligned max-B arena and source-shaped weight.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.attention,
            addresses.qkv,
            addresses.activation,
            addresses.weight,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = OUTPUT_ROWS * COLUMNS * size_of::<u16>();
    let per_token = 14 * COLUMNS + 2 * OUTPUT_ROWS;

    weights + batch * per_token
}

/// Measures every exact MTP gated BF16 attention-output production graph.
pub fn benchmark_mtp_bf16_attention_output(
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
        "qwen3_8/mtp/bf16_attention_output/weights",
        BenchmarkMemoryKind::Weights,
        session.regions.weight_bytes(),
        "unchanged [5120,6144] source-BF16 output matrix",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_attention_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.regions.workspace_bytes(),
        "max_batch=8 attention, QKV, gated BF16 activation, and output seams",
    )?;
    memory.register_owned(
        "qwen3_8/mtp/bf16_attention_output/alignment_padding",
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
            suite: "bench-mtp-bf16-attention-output",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, complete production graph, and repeated-operation graph for sigmoid gate, BF16 seam, and source-BF16 projection",
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
    fn mtp_bf16_attention_output_suite_benchmark_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 62_914_560);
        assert_eq!(regions.workspace_bytes(), 606_208);
        assert_eq!(regions.payload_bytes(), 63_520_768);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(logical_bytes(1), 63_010_816);
        assert_eq!(logical_bytes(MAX_BATCH), 63_684_608);
    }
}

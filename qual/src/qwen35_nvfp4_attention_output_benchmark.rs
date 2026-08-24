//! Paired timings for exact Qwen3.5 gated NVFP4 attention output.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_nvfp4_attention_output::{
    CODE_BYTES_PER_ROW, COLUMNS, GROUPS_PER_ROW, MAX_BATCH, OUTPUT_ROWS, Regions,
    WEIGHT_SCALE_DIVISOR, layout, make_fixture,
};
use crate::target::Qwen35Nvfp4AttentionOutputOp;
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer,
    PinnedHostBuffer,
};

struct RouteGraphs {
    batch: usize,
    preparation: CudaGraph,
    operation: CudaGraph,
}

struct Addresses {
    attention: *mut f32,
    qkv: *const u16,
    activation: *mut u16,
    weight_codes: *const u8,
    weight_scales: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen35Nvfp4AttentionOutputOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _attention_source: PinnedHostBuffer<f32>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new() -> Result<Self, DeviceBenchmarkError> {
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
        arena.copy_from_host(&stream, regions.attention, &fixture.attention)?;
        arena.copy_from_host(&stream, regions.qkv, &fixture.qkv)?;
        arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
        arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
        let attention_source =
            PinnedHostBuffer::from_slice(&context, &fixture.attention).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen35Nvfp4AttentionOutputOp::new(&context)?;
        let addresses = Addresses {
            attention: arena.address(regions.attention)?,
            qkv: arena.address(regions.qkv)?,
            activation: arena.address(regions.activation)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        let routes = (1..=MAX_BATCH)
            .map(|batch| {
                capture_route(
                    &op,
                    &arena,
                    regions,
                    &stream,
                    &attention_source,
                    &addresses,
                    batch,
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
            _attention_source: attention_source,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.preparation.launch(&self.stream) }?;
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.operation.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self) -> Vec<ExactDeviceCase<'_>> {
        self.routes
            .iter()
            .map(|route| {
                ExactDeviceCase::new(
                    "qwen35_9b/attention_output/gate_bf16_nvfp4",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_decode(route.batch as u32),
                    OperationAccounting::new(
                        logical_bytes(route.batch),
                        route.batch as u64,
                        "token",
                    ),
                    &route.operation,
                    None,
                )
                .with_preparation(&route.preparation)
            })
            .collect()
    }
}

fn capture_route(
    op: &Qwen35Nvfp4AttentionOutputOp,
    arena: &DeviceArena,
    regions: Regions,
    stream: &CudaStream,
    attention_source: &PinnedHostBuffer<f32>,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<RouteGraphs> {
    let preparation = CudaGraph::capture(stream, || unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.attention,
            attention_source,
            batch * COLUMNS,
        )
    })?;
    let operation = CudaGraph::capture(stream, || launch(op, stream, addresses, batch))?;

    Ok(RouteGraphs {
        batch,
        preparation,
        operation,
    })
}

fn launch(
    op: &Qwen35Nvfp4AttentionOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.attention,
            addresses.qkv,
            addresses.activation,
            addresses.weight_codes,
            addresses.weight_scales,
            WEIGHT_SCALE_DIVISOR,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let per_token = 14 * COLUMNS + 2 * OUTPUT_ROWS;

    weights + batch * per_token
}

/// Measures every complete Qwen3.5 gated NVFP4 attention-output graph.
pub fn benchmark_qwen35_nvfp4_attention_output(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new()?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen35_9b/attention_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "packed output weights plus swizzled block scales",
    )?;
    memory.register_owned(
        "qwen35_9b/attention_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_batch=8",
    )?;
    memory.register_owned(
        "qwen35_9b/attention_output/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases();
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-nvfp4-attention-output",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion and complete gate-BF16-plus-NVFP4 graph; pinned input restoration is outside every timed operation",
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
    fn accounting_covers_the_complete_output_path() {
        let (layout, regions) = layout().unwrap();
        let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
        let b8 = weights + MAX_BATCH * (14 * COLUMNS + 2 * OUTPUT_ROWS);

        assert_eq!(logical_bytes(MAX_BATCH), b8);
        assert_eq!(crate::qwen35_nvfp4_attention_output::QKV_ROWS, 10_240);
        assert_eq!(layout.byte_len(), 9_863_168);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 425_984);
    }
}

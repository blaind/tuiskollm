//! Paired timings for exact Qwen3.6 gated static-FP8 attention output.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::qwen36_attention_output::{
    COLUMNS, INPUT_SCALE, MAX_BATCH, OUTPUT_ROWS, Regions, WEIGHT_SCALE, layout, make_fixture,
};
use crate::target::Qwen36AttentionOutputOp;
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
    activation_codes: *mut u8,
    weight_codes: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Qwen36AttentionOutputOp,
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
        let fixture = make_fixture().map_err(|error| {
            DeviceBenchmarkError::Precondition(format!("building Qwen3.6 fixture: {error}"))
        })?;
        arena.copy_from_host(&stream, regions.attention, &fixture.attention)?;
        arena.copy_from_host(&stream, regions.qkv, &fixture.qkv)?;
        arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
        let attention_source =
            PinnedHostBuffer::from_slice(&context, &fixture.attention).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen36AttentionOutputOp::new(&context)?;
        let addresses = Addresses {
            attention: arena.address(regions.attention)?,
            qkv: arena.address(regions.qkv)?,
            activation: arena.address(regions.activation)?,
            activation_codes: arena.address(regions.activation_codes)?,
            weight_codes: arena.address(regions.weight_codes)?,
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
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
                unsafe { route.preparation.launch(&self.stream) }?;
                // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
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
                    "qwen36_35b_a3b/attention_output/gate_static_fp8",
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
    op: &Qwen36AttentionOutputOp,
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
    op: &Qwen36AttentionOutputOp,
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
            addresses.activation_codes,
            INPUT_SCALE,
            addresses.weight_codes,
            WEIGHT_SCALE,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = OUTPUT_ROWS * COLUMNS;
    let per_token = 15 * COLUMNS + 2 * OUTPUT_ROWS;

    weights + batch * per_token
}

/// Measures every complete Qwen3.6 gated static-FP8 attention-output graph.
pub fn benchmark_qwen36_attention_output(
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
        "qwen36_35b_a3b/attention_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source E4M3 [2048,4096] output matrix",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/attention_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_batch=8 attention, QKV, BF16 staging, static codes, and output",
    )?;
    memory.register_owned(
        "qwen36_35b_a3b/attention_output/alignment_padding",
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
            suite: "bench-qwen36-attention-output",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion and complete gate-BF16-plus-static-FP8 graph; pinned attention restoration is outside every timed operation",
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
        let b8 = regions.weight_bytes() + MAX_BATCH * (15 * COLUMNS + 2 * OUTPUT_ROWS);

        assert_eq!(logical_bytes(MAX_BATCH), b8);
        assert_eq!(crate::qwen36_attention_output::QKV_ROWS, 9_216);
        assert_eq!(layout.byte_len(), 8_798_208);
        assert_eq!(regions.weight_bytes(), 8_388_608);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 409_600);
    }
}

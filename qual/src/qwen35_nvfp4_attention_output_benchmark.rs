//! Paired timings for exact Qwen3.5 gated NVFP4 attention output.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::qwen35_nvfp4_attention_output::{
    CODE_BYTES_PER_ROW, COLUMNS, EXACT_ROUTES, GROUPS_PER_ROW, INPUT_SCALE_DIVISOR, MAX_BATCH,
    OUTPUT_ROWS, Regions, WEIGHT_SCALE_DIVISOR, layout, make_fixture,
};
use crate::target::Qwen35Nvfp4AttentionOutputOp;
use std::sync::Arc;
use tuisko_gpu::{
    CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer,
    PinnedHostBuffer,
};

struct RouteGraphs {
    rows: usize,
    preparation: CudaGraph,
    operation: CudaGraph,
}

struct Addresses {
    attention: *mut f32,
    qkv: *const u16,
    activation: *mut u16,
    activation_codes: *mut u8,
    activation_scales: *mut u8,
    weight_codes: *const u8,
    weight_scales: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
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
        let fixture = make_fixture().map_err(|error| {
            DeviceBenchmarkError::Precondition(format!(
                "Qwen3.5 attention-output fixture construction failed: {error}"
            ))
        })?;
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
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| {
                capture_route(
                    &op,
                    &arena,
                    regions,
                    &stream,
                    &attention_source,
                    &addresses,
                    rows,
                )
            })
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
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
                    "qwen35_9b/attention_output/gate_bf16_nvfp4",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
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
    rows: usize,
) -> GpuResult<RouteGraphs> {
    let preparation = CudaGraph::capture(stream, || unsafe {
        arena.copy_prefix_from_pinned_host_async(
            stream,
            regions.attention,
            attention_source,
            rows * COLUMNS,
        )
    })?;
    let operation = CudaGraph::capture(stream, || launch(op, stream, addresses, rows))?;

    Ok(RouteGraphs {
        rows,
        preparation,
        operation,
    })
}

fn launch(
    op: &Qwen35Nvfp4AttentionOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        if rows <= MAX_BATCH {
            op.launch(
                stream,
                rows,
                addresses.attention,
                addresses.qkv,
                addresses.activation,
                addresses.weight_codes,
                addresses.weight_scales,
                WEIGHT_SCALE_DIVISOR,
                addresses.output,
            )
        } else {
            op.launch_prefill(
                stream,
                rows,
                addresses.attention,
                addresses.qkv,
                addresses.activation,
                addresses.activation_codes,
                addresses.activation_scales,
                addresses.weight_codes,
                addresses.weight_scales,
                INPUT_SCALE_DIVISOR,
                WEIGHT_SCALE_DIVISOR,
                addresses.output,
            )
        }
    }
}

fn logical_bytes(rows: usize) -> usize {
    let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let per_token = 14 * COLUMNS + 2 * OUTPUT_ROWS;
    let scratch = if rows > MAX_BATCH {
        2 * rows * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW)
    } else {
        0
    };

    weights + rows * per_token + scratch
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
    let mut timer = GpuTimer::new(session.stream.context())?;
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
        "max_rows=128 gated, quantized, and output seams",
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
        measure_cases(&session.stream, &mut timer, &cases, options)?;
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
        let t128 = weights
            + crate::qwen35_nvfp4_attention_output::MAX_ROWS
                * (14 * COLUMNS + 2 * OUTPUT_ROWS + 2 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW));

        assert_eq!(
            logical_bytes(crate::qwen35_nvfp4_attention_output::MAX_ROWS),
            t128
        );
        assert_eq!(crate::qwen35_nvfp4_attention_output::QKV_ROWS, 10_240);
        assert_eq!(layout.byte_len(), 16_547_840);
        assert_eq!(regions.weight_bytes(), 9_437_184);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 7_110_656);
    }
}

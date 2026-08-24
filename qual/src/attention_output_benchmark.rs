//! Direct timings for every exact gated FP8 attention-output graph route.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::{SCALE_VALUES, WEIGHT_CODES, f32_to_bf16};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::AttentionOutputOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
const ATTENTION_VALUES: [f32; 8] = [1.0, -0.875, 0.75, -0.625, 0.5, -0.375, 0.25, -0.125];
const REPLAY_STABLE_GATE: f32 = 128.0;

#[derive(Clone, Copy)]
struct Regions {
    attention: ArenaRegion<f32>,
    qkv: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.attention.byte_len()
            + self.qkv.byte_len()
            + self.activation_codes.byte_len()
            + self.activation_scales.byte_len()
            + self.weight_codes.byte_len()
            + self.weight_scales.byte_len()
            + self.output.byte_len()
    }

    fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }
}

struct Addresses {
    attention: *mut f32,
    qkv: *const u16,
    activation_codes: *mut u8,
    activation_scales: *mut f32,
    weight_codes: *const u8,
    weight_scales: *const u16,
    output: *mut u16,
}

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: AttentionOutputOp,
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
        let op = AttentionOutputOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| capture_route(&op, &stream, &addresses, rows, repeated_operations))
            .collect::<GpuResult<Vec<_>>>()?;
        let timer = GpuTimer::new(&context)?;

        Ok(Self {
            routes,
            timer,
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
                    "attention_output/sigmoid_quantize_projection",
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

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let attention = layout.reserve(MAX_ROWS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let qkv = layout.reserve(MAX_ROWS * Qwen38_27B::ATTENTION_QKV_ROWS, ALIGNMENT)?;
    let activation_codes =
        layout.reserve(MAX_ROWS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(
        Qwen38_27B::HIDDEN * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ALIGNMENT,
    )?;
    let weight_scales = layout.reserve(Qwen38_27B::HIDDEN, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * Qwen38_27B::HIDDEN, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            attention,
            qkv,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let attention = (0..MAX_ROWS * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| ATTENTION_VALUES[(index + index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS) & 7])
        .collect::<Vec<_>>();
    let mut qkv = vec![0u16; MAX_ROWS * Qwen38_27B::ATTENTION_QKV_ROWS];
    // A large positive BF16 gate makes `1 + exp(-gate)` round to exactly one.
    // Repeated graphs therefore time identical production inputs even though the
    // public gated-FP32 seam is written in place by every operation.
    for token in 0..MAX_ROWS {
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
    let weight_codes = (0..Qwen38_27B::HIDDEN * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS)
        .map(|index| {
            let row = index / Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
            WEIGHT_CODES[(row + index) & 3]
        })
        .collect::<Vec<_>>();
    let weight_scales = (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.attention, &attention)?;
    arena.copy_from_host(stream, regions.qkv, &qkv)?;
    arena.copy_from_host(stream, regions.weight_codes, &weight_codes)?;
    arena.copy_from_host(stream, regions.weight_scales, &weight_scales)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        attention: arena.address(regions.attention)?,
        qkv: arena.address(regions.qkv)?,
        activation_codes: arena.address(regions.activation_codes)?,
        activation_scales: arena.address(regions.activation_scales)?,
        weight_codes: arena.address(regions.weight_codes)?,
        weight_scales: arena.address(regions.weight_scales)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &AttentionOutputOp,
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
    op: &AttentionOutputOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every pointer names its complete, aligned maximum-batch arena region.
    unsafe {
        op.launch(
            stream,
            rows,
            addresses.attention,
            addresses.qkv,
            addresses.activation_codes,
            addresses.activation_scales,
            addresses.weight_codes,
            addresses.weight_scales,
            addresses.output,
        )
    }
}

fn logical_bytes(rows: usize) -> usize {
    let columns = Qwen38_27B::ATTENTION_OUTPUT_COLUMNS;
    let output_rows = Qwen38_27B::HIDDEN;
    let weights = output_rows * (columns + size_of::<u16>());
    let per_token = 16 * columns + 3 * size_of::<f32>() + output_rows * size_of::<u16>();

    weights + rows * per_token
}

/// Measures every exact attention-output decode and prefill route.
pub fn benchmark_attention_output(
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
        "attention_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source-native [5120,6144] E4M3 output projection and BF16 row scales",
    )?;
    memory.register_owned(
        "attention_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=1024 gated attention, fused QKV, quantization, and output seams",
    )?;
    memory.register_owned(
        "attention_output/alignment_padding",
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
            suite: "bench-attention-output",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph for sigmoid gate, E4M3 quantization, and projection",
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
    use super::{MAX_BATCH, MAX_ROWS, layout, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn attention_output_suite_benchmark_byte_accounting_covers_decode_and_prefill() {
        let weights =
            Qwen38_27B::HIDDEN * (Qwen38_27B::ATTENTION_OUTPUT_COLUMNS + size_of::<u16>());
        let per_token = 16 * Qwen38_27B::ATTENTION_OUTPUT_COLUMNS
            + 3 * size_of::<f32>()
            + Qwen38_27B::HIDDEN * size_of::<u16>();

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * per_token);
        assert_eq!(logical_bytes(MAX_ROWS), weights + MAX_ROWS * per_token);
    }

    #[test]
    fn attention_output_suite_benchmark_arena_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
        assert_eq!(layout.byte_len(), 102_774_784);
        assert_eq!(regions.payload_bytes(), 102_774_784);
    }
}

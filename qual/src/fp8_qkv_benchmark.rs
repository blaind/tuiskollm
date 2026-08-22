//! Paired timings for the exact FP8 QKV graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, FullAttentionQkvOp};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
#[cfg(feature = "device")]
const MAX_ROWS: usize = 16;
#[cfg(feature = "sm89")]
const MAX_ROWS: usize = MAX_BATCH;
#[cfg(feature = "device")]
const EXACT_ROUTES: [usize; MAX_BATCH + 1] = [1, 2, 3, 4, 5, 6, 7, 8, 16];
#[cfg(feature = "sm89")]
const EXACT_ROUTES: [usize; MAX_BATCH] = [1, 2, 3, 4, 5, 6, 7, 8];
#[cfg(feature = "device")]
const MAX_ROWS_DESCRIPTION: &str = "max_rows=16";
#[cfg(feature = "sm89")]
const MAX_ROWS_DESCRIPTION: &str = "max_rows=8";
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const TOKEN_FACTORS: [f32; 16] = [
    1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125, 0.75, -0.75, 0.375, -0.375, 0.1875, -0.1875,
    0.09375, -0.09375,
];
const WEIGHT_CODES: [u8; 4] = [0x38, 0xb0, 0x30, 0x28];
const SCALE_VALUES: [f32; 4] = [1.0, 0.5, 0.25, 2.0];

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    activation_codes: ArenaRegion<u8>,
    activation_scales: ArenaRegion<f32>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

struct RouteGraphs {
    rows: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Addresses {
    input: *const u16,
    activation_codes: *mut u8,
    activation_scales: *mut f32,
    weight_codes: *const u8,
    weight_scales: *const u16,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: FullAttentionQkvOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        let capability = context.compute_capability().map_err(GpuError::from)?;
        if capability != EXPECTED_COMPUTE_CAPABILITY {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "device zero has compute capability {}.{}, expected {}.{}",
                capability.0,
                capability.1,
                EXPECTED_COMPUTE_CAPABILITY.0,
                EXPECTED_COMPUTE_CAPABILITY.1,
            )));
        }

        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        arena.copy_from_host(&stream, regions.input, &make_input())?;
        arena.copy_from_host(&stream, regions.weight_codes, &make_weight_codes())?;
        arena.copy_from_host(&stream, regions.weight_scales, &make_weight_scales())?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = FullAttentionQkvOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        let mut routes = Vec::with_capacity(EXACT_ROUTES.len());
        for rows in EXACT_ROUTES {
            routes.push(capture_route(
                &op,
                &stream,
                &addresses,
                rows,
                repeated_operations,
            )?);
        }
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
                route.leaf.launch(&self.stream)?;
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
                        BenchmarkWorkload::warm_operator_mtp(route.rows as u64),
                    )
                };
                ExactDeviceCase::new(
                    "fp8_qkv/quantize_projection",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    Some(RepeatedGraph::new(&route.repeated, repeated_operations)),
                )
            })
            .collect()
    }

    fn weight_bytes(&self) -> usize {
        self.regions.weight_codes.byte_len() + self.regions.weight_scales.byte_len()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let rows = Qwen38_27B::ATTENTION_QKV_ROWS;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * hidden, ALIGNMENT)?;
    let weight_scales = layout.reserve(rows, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * rows, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn make_input() -> Vec<u16> {
    (0..MAX_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & 7] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn make_weight_codes() -> Vec<u8> {
    let rows = Qwen38_27B::ATTENTION_QKV_ROWS;
    let mut codes = vec![0; rows * Qwen38_27B::HIDDEN];
    for (row, values) in codes
        .as_mut_slice()
        .as_chunks_mut::<{ Qwen38_27B::HIDDEN }>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[row & 3]);
    }

    codes
}

fn make_weight_scales() -> Vec<u16> {
    (0..Qwen38_27B::ATTENTION_QKV_ROWS)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect()
}

fn capture_route(
    op: &FullAttentionQkvOp,
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
    op: &FullAttentionQkvOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every pointer names its complete, aligned, maximum-row arena region.
    unsafe {
        op.launch(
            stream,
            rows,
            addresses.input,
            addresses.activation_codes,
            addresses.activation_scales,
            addresses.weight_codes,
            addresses.weight_scales,
            addresses.output,
        )
    }
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let output_rows = Qwen38_27B::ATTENTION_QKV_ROWS;
    let activation = rows * (2 * hidden + 2 * hidden + 2 * size_of::<f32>());
    let weights = output_rows * hidden + output_rows * size_of::<u16>();
    let output = rows * output_rows * size_of::<u16>();

    activation + weights + output
}

/// Measures every admitted FP8 QKV route with paired timings.
pub fn benchmark_fp8_qkv(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let weight_bytes = session.weight_bytes();
    memory.register_owned(
        "fp8_qkv/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source-native fused QKV",
    )?;
    memory.register_owned(
        "fp8_qkv/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes,
        MAX_ROWS_DESCRIPTION,
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
            suite: "bench-fp8-qkv",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, repeated production graph, and repeated-operation graph",
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

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, MAX_ROWS, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_the_complete_quantize_projection_path() {
        let weights = Qwen38_27B::ATTENTION_QKV_ROWS * (Qwen38_27B::HIDDEN + size_of::<u16>());
        let per_token =
            4 * Qwen38_27B::HIDDEN + 2 * size_of::<f32>() + 2 * Qwen38_27B::ATTENTION_QKV_ROWS;

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * per_token);
        assert_eq!(logical_bytes(MAX_ROWS), weights + MAX_ROWS * per_token);
    }
}

//! Paired timings for exact dense-FP8 down graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::{DenseFp8DownOp, DenseFp8DownTmaMaps};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 1_024];
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const TOKEN_FACTORS: [f32; 8] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];
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

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
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
    input: *const u16,
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
    _op: DenseFp8DownOp,
    maps: DenseFp8DownTmaMaps,
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
        arena.copy_from_host(&stream, regions.input, &make_input())?;
        arena.copy_from_host(&stream, regions.weight_codes, &make_weight_codes())?;
        arena.copy_from_host(&stream, regions.weight_scales, &make_weight_scales())?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = DenseFp8DownOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        // SAFETY: this session owns the stable maximum activation and source weight planes.
        let maps = unsafe {
            DenseFp8DownTmaMaps::new(
                &stream,
                addresses.activation_codes.cast_const(),
                addresses.weight_codes,
            )?
        };
        let mut routes = Vec::with_capacity(EXACT_ROUTES.len());
        for rows in EXACT_ROUTES {
            routes.push(capture_route(
                &op,
                &maps,
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
            maps,
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
                let (shape, workload) = if route.rows <= 8 {
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
                    "fp8_down/quantize_projection",
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
        self.regions.weight_bytes()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * intermediate, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * intermediate, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(hidden * intermediate, ALIGNMENT)?;
    let weight_scales = layout.reserve(hidden, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * hidden, ALIGNMENT)?;

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
    (0..MAX_ROWS * Qwen38_27B::INTERMEDIATE)
        .map(|index| {
            let token = index / Qwen38_27B::INTERMEDIATE;
            f32_to_bf16(INPUT_PATTERN[index & 7] * TOKEN_FACTORS[token & 7])
        })
        .collect()
}

fn make_weight_codes() -> Vec<u8> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    (0..hidden * intermediate)
        .map(|index| {
            let row = index / intermediate;
            let column = index - row * intermediate;
            WEIGHT_CODES[(row + column) & 3]
        })
        .collect()
}

fn make_weight_scales() -> Vec<u16> {
    (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect()
}

fn capture_route(
    op: &DenseFp8DownOp,
    maps: &DenseFp8DownTmaMaps,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let leaf = CudaGraph::capture(stream, || launch(op, maps, stream, addresses, rows))?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch(op, maps, stream, addresses, rows)?;
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
    op: &DenseFp8DownOp,
    maps: &DenseFp8DownTmaMaps,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: all pointers cover their complete aligned maximum-batch arena regions.
    unsafe {
        if rows == MAX_ROWS {
            op.launch_macro_prefill(
                stream,
                addresses.input,
                addresses.activation_codes,
                addresses.activation_scales,
                addresses.weight_codes,
                addresses.weight_scales,
                addresses.output,
                maps,
            )
        } else {
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
}

fn logical_bytes(batch: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let activation = batch * (4 * intermediate + 2 * size_of::<f32>());
    let weights = hidden * (intermediate + size_of::<u16>());
    let output = batch * hidden * size_of::<u16>();

    activation + weights + output
}

/// Measures all exact dense-FP8 down routes with paired graph timings.
pub fn benchmark_fp8_down(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let weight_bytes = session.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "fp8_down/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source-native down planes",
    )?;
    memory.register_owned(
        "fp8_down/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=1024 quantization and down output seams",
    )?;
    memory.register_owned(
        "fp8_down/tma_descriptors",
        BenchmarkMemoryKind::Other,
        session.maps.byte_len(),
        "two address-bound 128-byte tensor maps",
    )?;
    memory.register_owned(
        "fp8_down/alignment_padding",
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
            suite: "bench-fp8-down",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, production graph, and repeated-operation graph",
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
    use super::{EXACT_ROUTES, MAX_ROWS, layout, logical_bytes};
    use tuisko_kernels_sm120::DenseFp8DownTmaMaps;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn fp8_down_suite_benchmark_byte_accounting_covers_every_route() {
        let weights = Qwen38_27B::HIDDEN * (Qwen38_27B::INTERMEDIATE + size_of::<u16>());
        let per_token = 4 * Qwen38_27B::INTERMEDIATE
            + 2 * size_of::<f32>()
            + Qwen38_27B::HIDDEN * size_of::<u16>();

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_ROWS), weights + MAX_ROWS * per_token);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 1_024]);
    }

    #[test]
    fn fp8_down_suite_benchmark_arena_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
        assert_eq!(layout.byte_len(), 153_106_432);
        assert_eq!(regions.payload_bytes(), 153_106_432);
        assert_eq!(DenseFp8DownTmaMaps::BYTE_LEN, 256);
    }
}

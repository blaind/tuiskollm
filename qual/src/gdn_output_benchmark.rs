//! Paired timings for every exact FP8 GDN output graph route.

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
use tuisko_kernels_sm120::{DenseFp8GdnOutputTmaMaps, GdnOutputProjectionOp};
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const ALIGNMENT: usize = 256;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const TOKEN_FACTORS: [f32; 16] = [
    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, -1.0, -0.875, -0.75, -0.625, -0.5, -0.375,
    -0.25, -0.125,
];

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
    _op: GdnOutputProjectionOp,
    _maps: DenseFp8GdnOutputTmaMaps,
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

        let op = GdnOutputProjectionOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            activation_codes: arena.address(regions.activation_codes)?,
            activation_scales: arena.address(regions.activation_scales)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        // SAFETY: the arena owns exact stable T=1024 activation and weight planes.
        let maps = unsafe {
            DenseFp8GdnOutputTmaMaps::new(
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

        Ok(Self {
            routes,
            _op: op,
            _maps: maps,
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
                    "gdn_output/quantize_projection",
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

    fn workspace_bytes(&self) -> usize {
        self.regions.payload_bytes() - self.weight_bytes()
    }

    fn padding_bytes(&self) -> usize {
        self.arena.byte_len() - self.regions.payload_bytes()
    }
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
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let rows = Qwen38_27B::HIDDEN;
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * columns, ALIGNMENT)?;
    let activation_codes = layout.reserve(MAX_ROWS * columns, ALIGNMENT)?;
    let activation_scales = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let weight_codes = layout.reserve(rows * columns, ALIGNMENT)?;
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
    (0..MAX_ROWS * Qwen38_27B::GDN_VALUE_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::GDN_VALUE_ROWS;
            f32_to_bf16(INPUT_PATTERN[index & 7] * TOKEN_FACTORS[token & 15])
        })
        .collect()
}

fn make_weight_codes() -> Vec<u8> {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let mut codes = vec![0; Qwen38_27B::HIDDEN * columns];
    for (row, values) in codes
        .as_mut_slice()
        .as_chunks_mut::<{ Qwen38_27B::GDN_VALUE_ROWS }>()
        .0
        .iter_mut()
        .enumerate()
    {
        values.fill(WEIGHT_CODES[row & 3]);
    }

    codes
}

fn make_weight_scales() -> Vec<u16> {
    (0..Qwen38_27B::HIDDEN)
        .map(|row| f32_to_bf16(SCALE_VALUES[row & 3]))
        .collect()
}

fn capture_route(
    op: &GdnOutputProjectionOp,
    maps: &DenseFp8GdnOutputTmaMaps,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
    repeated_operations: u64,
) -> GpuResult<RouteGraphs> {
    let launch_once = || -> GpuResult<()> {
        if rows == MAX_ROWS {
            // SAFETY: every pointer names its complete, aligned arena region.
            return unsafe {
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
            };
        }
        launch(op, stream, addresses, rows)
    };
    let leaf = CudaGraph::capture(stream, launch_once)?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch_once()?;
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
    op: &GdnOutputProjectionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every pointer names its complete, aligned maximum-batch arena region.
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

fn logical_bytes(rows_count: usize) -> usize {
    let columns = Qwen38_27B::GDN_VALUE_ROWS;
    let rows = Qwen38_27B::HIDDEN;
    let activation = rows_count * (2 * columns + 2 * columns + 2 * size_of::<f32>());
    let weights = rows * columns + rows * size_of::<u16>();
    let output = rows_count * rows * size_of::<u16>();

    activation + weights + output
}

/// Measures every exact FP8 GDN output batch with paired host/device timings.
pub fn benchmark_gdn_output(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.weight_bytes();
    let workspace_bytes = session.workspace_bytes();
    let padding_bytes = session.padding_bytes();
    memory.register_owned(
        "gdn_output/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "source-native GDN output projection",
    )?;
    memory.register_owned(
        "gdn_output/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_rows=1024 inputs, activation scratch, and output",
    )?;
    memory.register_owned(
        "gdn_output/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte region alignment",
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
            suite: "bench-gdn-output",
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

#[cfg(test)]
mod tests {
    use super::{EXACT_ROUTES, MAX_BATCH, MAX_ROWS, layout, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_covers_the_complete_quantize_projection_path() {
        let weights = Qwen38_27B::HIDDEN * (Qwen38_27B::GDN_VALUE_ROWS + size_of::<u16>());
        let per_token =
            4 * Qwen38_27B::GDN_VALUE_ROWS + 2 * size_of::<f32>() + 2 * Qwen38_27B::HIDDEN;

        assert_eq!(logical_bytes(1), weights + per_token);
        assert_eq!(logical_bytes(MAX_BATCH), weights + MAX_BATCH * per_token);
        assert_eq!(logical_bytes(MAX_ROWS), weights + MAX_ROWS * per_token);
    }

    #[test]
    fn benchmark_route_inventory_and_owner_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        let (layout, regions) = layout().unwrap();

        assert_eq!(
            regions.weight_codes.byte_len() + regions.weight_scales.byte_len(),
            31_467_520
        );
        assert_eq!(regions.payload_bytes(), 60_831_744);
        assert_eq!(layout.byte_len(), 60_831_744);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }
}

//! Paired timings for every exact SM120 NVFP4 A16 down projection route.

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
use tuisko_kernels_sm120::Nvfp4DownOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const INPUT_COLUMNS: usize = Qwen38_27B::INTERMEDIATE;
const OUTPUT_ROWS: usize = Qwen38_27B::HIDDEN;
const GROUP: usize = 16;
const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
const WEIGHT_SCALE_DIVISOR: f32 = 0.125;
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    weight_codes: ArenaRegion<u8>,
    weight_scales: ArenaRegion<u8>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    fn payload_bytes(self) -> usize {
        self.input.byte_len() + self.weight_bytes() + self.output.byte_len()
    }
}

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Addresses {
    input: *const u16,
    weight_codes: *const u8,
    weight_scales: *const u8,
    output: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: Nvfp4DownOp,
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
                capability.0, capability.1,
            )));
        }

        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        arena.copy_from_host(&stream, regions.input, &make_input())?;
        arena.copy_from_host(&stream, regions.weight_codes, &make_weight_codes())?;
        arena.copy_from_host(&stream, regions.weight_scales, &make_weight_scales())?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Nvfp4DownOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            weight_codes: arena.address(regions.weight_codes)?,
            weight_scales: arena.address(regions.weight_scales)?,
            output: arena.address(regions.output)?,
        };
        let mut routes = Vec::with_capacity(MAX_BATCH);
        for batch in 1..=MAX_BATCH {
            routes.push(capture_route(
                &op,
                &stream,
                &addresses,
                batch,
                repeated_operations,
            )?);
        }

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
                    "nvfp4_down/a16",
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_decode(route.batch as u32),
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

    fn weight_bytes(&self) -> usize {
        self.regions.weight_bytes()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * INPUT_COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn make_input() -> Vec<u16> {
    (0..MAX_BATCH * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn make_weight_codes() -> Vec<u8> {
    const BASE: [u8; 8] = [0xf7, 0xd5, 0xb3, 0x70, 0x5f, 0x3d, 0x0b, 0xf7];
    const SPARSE: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
    let negative = BASE.map(|byte| byte ^ 0x88);
    let mut codes = vec![0u8; OUTPUT_ROWS * CODE_BYTES_PER_ROW];

    for row in 0..OUTPUT_ROWS {
        let base_is_base = row & 1 == 0;
        let base = if base_is_base { &BASE } else { &negative };
        let exceptional = if base_is_base { &SPARSE } else { &BASE };
        let exceptional_group = exceptional_group(row);
        for group in 0..GROUPS_PER_ROW {
            let begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
            let pattern = if group == exceptional_group {
                exceptional
            } else {
                base
            };
            codes[begin..begin + GROUP / 2].copy_from_slice(pattern);
        }
    }

    codes
}

fn make_weight_scales() -> Vec<u8> {
    const SCALE_CODES: [u8; 4] = [0x38, 0x01, 0x40, 0x01];
    let mut scales = vec![0u8; OUTPUT_ROWS * GROUPS_PER_ROW];

    for row in 0..OUTPUT_ROWS {
        for group in 0..GROUPS_PER_ROW {
            let scale_index = if group == exceptional_group(row) {
                (row + 1) & 3
            } else {
                row & 3
            };
            scales[scale_offset(row, group)] = SCALE_CODES[scale_index];
        }
    }

    scales
}

fn scale_offset(row: usize, group: usize) -> usize {
    let tile = row / 128;
    let row_in_tile = row & 127;
    let scale_tile = group / 4;
    let scale_lane = group & 3;
    let row_mod32 = row_in_tile & 31;
    let row_quartile = row_in_tile >> 5;

    (tile * (GROUPS_PER_ROW / 4) + scale_tile) * 512
        + row_mod32 * 16
        + row_quartile * 4
        + scale_lane
}

fn exceptional_group(row: usize) -> usize {
    (row * 17 + row / 128 * 13) % GROUPS_PER_ROW
}

fn capture_route(
    op: &Nvfp4DownOp,
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
    op: &Nvfp4DownOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: every pointer names its complete, aligned maximum-batch region.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.input,
            addresses.weight_codes,
            addresses.weight_scales,
            WEIGHT_SCALE_DIVISOR,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    let input = batch * INPUT_COLUMNS * size_of::<u16>();
    let output = batch * OUTPUT_ROWS * size_of::<u16>();

    weights + input + output
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

/// Measures every exact-target NVFP4 down projection route with paired timings.
pub fn benchmark_nvfp4_down(
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
        "nvfp4_down/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "packed down weights plus swizzled block scales",
    )?;
    memory.register_owned(
        "nvfp4_down/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_batch=8",
    )?;
    memory.register_owned(
        "nvfp4_down/alignment_padding",
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
            suite: "bench-nvfp4-down",
            classification: "performance_sensitive_leaf",
            timing_scope: "paired Rust submission/completion, repeated leaf graph, and repeated-operation graph",
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
    use super::{
        CODE_BYTES_PER_ROW, GROUPS_PER_ROW, INPUT_COLUMNS, MAX_BATCH, OUTPUT_ROWS, layout,
        logical_bytes,
    };

    #[test]
    fn byte_accounting_covers_the_complete_a16_path() {
        let weights = OUTPUT_ROWS * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
        let b8 = weights
            + MAX_BATCH * (INPUT_COLUMNS * size_of::<u16>() + OUTPUT_ROWS * size_of::<u16>());

        assert_eq!(logical_bytes(MAX_BATCH), b8);
    }

    #[test]
    fn arena_accounting_exposes_every_owned_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(layout.byte_len(), 50_495_488);
        assert_eq!(regions.weight_bytes(), 50_135_040);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 360_448);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }
}

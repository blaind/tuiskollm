//! Paired timings for the exact Qwen3.8-Flash-Next hyper-connection graph routes.
//!
//! The three measured operations are the production ones: the combining gated
//! residual every sublayer runs, the model-level mixer that is this target's
//! final norm, and the raw-stream write-back. Each is timed on the production
//! stream, out of the same address-stable arena, in the warm cache regime, at
//! every admitted decode batch and prefill tile.
//!
//! No case needs a preparation graph: the benchmark drives the write-back in
//! its disjoint form, so no measured owner's output aliases its next input.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, Qwen38FlashNextHyperConnectionOp};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const MAX_ROWS: usize = 1_024;
const ALIGNMENT: usize = 256;

const BRANCHES: usize = Qwen38FlashNext::HC_COUNT;
const BRANCH: usize = Qwen38FlashNext::HIDDEN;
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
const RANK: usize = Qwen38FlashNext::HC_LOWRANK;

const STREAM_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];
const PROJECTION_PATTERN: [f32; 8] = [
    0.001953125,
    -0.0009765625,
    0.00048828125,
    0.0,
    -0.001953125,
    0.0009765625,
    -0.00048828125,
    0.0009765625,
];

struct GraphPair {
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct RouteGraphs {
    rows: usize,
    input_mix: GraphPair,
    final_mix: GraphPair,
    write_back: GraphPair,
}

#[derive(Clone, Copy)]
struct Regions {
    residual: ArenaRegion<u16>,
    block_output: ArenaRegion<u16>,
    norm_weight: ArenaRegion<u16>,
    down: ArenaRegion<u16>,
    up: ArenaRegion<u16>,
    inject: ArenaRegion<u16>,
    mixer_norm_weight: ArenaRegion<u16>,
    mixer_down: ArenaRegion<u16>,
    mixer_up: ArenaRegion<u16>,
    normalized: ArenaRegion<u16>,
    low_rank: ArenaRegion<u16>,
    mixed: ArenaRegion<u16>,
    write_gate: ArenaRegion<u16>,
    mixer_normalized: ArenaRegion<u16>,
    mixer_low_rank: ArenaRegion<u16>,
    mixer_mixed: ArenaRegion<u16>,
    injected: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.norm_weight.byte_len()
            + self.down.byte_len()
            + self.up.byte_len()
            + self.inject.byte_len()
            + self.mixer_norm_weight.byte_len()
            + self.mixer_down.byte_len()
            + self.mixer_up.byte_len()
    }

    fn payload_bytes(self) -> usize {
        self.weight_bytes()
            + self.residual.byte_len()
            + self.block_output.byte_len()
            + self.normalized.byte_len()
            + self.low_rank.byte_len()
            + self.mixed.byte_len()
            + self.write_gate.byte_len()
            + self.mixer_normalized.byte_len()
            + self.mixer_low_rank.byte_len()
            + self.mixer_mixed.byte_len()
            + self.injected.byte_len()
    }
}

struct Addresses {
    residual: *const u16,
    block_output: *const u16,
    norm_weight: *const u16,
    down: *const u16,
    up: *const u16,
    inject: *const u16,
    mixer_norm_weight: *const u16,
    mixer_down: *const u16,
    mixer_up: *const u16,
    normalized: *mut u16,
    low_rank: *mut u16,
    mixed: *mut u16,
    write_gate: *mut u16,
    mixer_normalized: *mut u16,
    mixer_low_rank: *mut u16,
    mixer_mixed: *mut u16,
    injected: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: Qwen38FlashNextHyperConnectionOp,
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
        let stream_host = (0..MAX_ROWS * WIDTH)
            .map(|index| f32_to_bf16(STREAM_PATTERN[(index + index / WIDTH) & 7]))
            .collect::<Vec<_>>();
        let block_host = (0..MAX_ROWS * BRANCH)
            .map(|index| f32_to_bf16(STREAM_PATTERN[(index * 3) & 7]))
            .collect::<Vec<_>>();
        let norm_host = (0..WIDTH)
            .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
            .collect::<Vec<_>>();
        let projection_host = (0..RANK * WIDTH)
            .map(|index| f32_to_bf16(PROJECTION_PATTERN[(index + index / WIDTH) & 7]))
            .collect::<Vec<_>>();
        let inject_host = (0..BRANCHES * WIDTH)
            .map(|index| f32_to_bf16(PROJECTION_PATTERN[(index * 3) & 7]))
            .collect::<Vec<_>>();
        arena.copy_from_host(&stream, regions.residual, &stream_host)?;
        arena.copy_from_host(&stream, regions.block_output, &block_host)?;
        arena.copy_from_host(&stream, regions.norm_weight, &norm_host)?;
        arena.copy_from_host(&stream, regions.mixer_norm_weight, &norm_host)?;
        arena.copy_from_host(&stream, regions.down, &projection_host)?;
        arena.copy_from_host(&stream, regions.up, &projection_host)?;
        arena.copy_from_host(&stream, regions.mixer_down, &projection_host)?;
        arena.copy_from_host(&stream, regions.mixer_up, &projection_host)?;
        arena.copy_from_host(&stream, regions.inject, &inject_host)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = Qwen38FlashNextHyperConnectionOp::new(&context)?;
        let addresses = Addresses {
            residual: arena.address(regions.residual)?,
            block_output: arena.address(regions.block_output)?,
            norm_weight: arena.address(regions.norm_weight)?,
            down: arena.address(regions.down)?,
            up: arena.address(regions.up)?,
            inject: arena.address(regions.inject)?,
            mixer_norm_weight: arena.address(regions.mixer_norm_weight)?,
            mixer_down: arena.address(regions.mixer_down)?,
            mixer_up: arena.address(regions.mixer_up)?,
            normalized: arena.address(regions.normalized)?,
            low_rank: arena.address(regions.low_rank)?,
            mixed: arena.address(regions.mixed)?,
            write_gate: arena.address(regions.write_gate)?,
            mixer_normalized: arena.address(regions.mixer_normalized)?,
            mixer_low_rank: arena.address(regions.mixer_low_rank)?,
            mixer_mixed: arena.address(regions.mixer_mixed)?,
            injected: arena.address(regions.injected)?,
        };

        // The write-back reads the gates the mix publishes, so the gate plane
        // is primed once before capture and every measured replay then reads
        // the same production values from the same address.
        // SAFETY: every pointer names a complete, aligned arena region.
        unsafe {
            launch_input_mix(&op, &stream, &addresses, MAX_ROWS)?;
        }
        stream.synchronize().map_err(GpuError::from)?;

        let mut routes = Vec::with_capacity(ROUTES.len());
        for rows in ROUTES {
            routes.push(RouteGraphs {
                rows,
                input_mix: capture(repeated_operations, &stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_input_mix(&op, &stream, &addresses, rows) }
                })?,
                final_mix: capture(repeated_operations, &stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_final_mix(&op, &stream, &addresses, rows) }
                })?,
                write_back: capture(repeated_operations, &stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_write_back(&op, &stream, &addresses, rows) }
                })?,
            });
        }
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
                for graph in [
                    &route.input_mix.leaf,
                    &route.final_mix.leaf,
                    &route.write_back.leaf,
                ] {
                    // SAFETY: this Session owns every route graph and everything it
                    // captured, dropping the graphs first.
                    unsafe { graph.launch(&self.stream) }?;
                }
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        let mut cases = Vec::with_capacity(self.routes.len() * 3);
        for route in &self.routes {
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
            for (name, pair, values) in [
                (
                    "qwen38_flash_next/hyper_connection/input_mix",
                    &route.input_mix,
                    input_mix_values(route.rows),
                ),
                (
                    "qwen38_flash_next/hyper_connection/final_mix",
                    &route.final_mix,
                    final_mix_values(route.rows),
                ),
                (
                    "qwen38_flash_next/hyper_connection/write_back",
                    &route.write_back,
                    write_back_values(route.rows),
                ),
            ] {
                cases.push(ExactDeviceCase::new(
                    name,
                    shape.clone(),
                    workload.clone(),
                    OperationAccounting::new(values * size_of::<u16>(), route.rows as u64, "token"),
                    &pair.leaf,
                    Some(RepeatedGraph::new(&pair.repeated, repeated_operations)),
                ));
            }
        }

        cases
    }
}

fn capture(
    repeated_operations: u64,
    stream: &CudaStream,
    launch: impl Fn() -> GpuResult<()>,
) -> GpuResult<GraphPair> {
    let leaf = CudaGraph::capture(stream, &launch)?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            launch()?;
        }

        Ok(())
    })?;

    Ok(GraphPair { leaf, repeated })
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_input_mix(
    op: &Qwen38FlashNextHyperConnectionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged.
    unsafe {
        op.launch_input_mix(
            stream,
            rows,
            addresses.residual,
            addresses.norm_weight,
            addresses.down,
            addresses.up,
            addresses.inject,
            addresses.normalized,
            addresses.low_rank,
            addresses.mixed,
            addresses.write_gate,
        )
    }
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_final_mix(
    op: &Qwen38FlashNextHyperConnectionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged.
    unsafe {
        op.launch_final_mix(
            stream,
            rows,
            addresses.residual,
            addresses.mixer_norm_weight,
            addresses.mixer_down,
            addresses.mixer_up,
            addresses.mixer_normalized,
            addresses.mixer_low_rank,
            addresses.mixer_mixed,
        )
    }
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_write_back(
    op: &Qwen38FlashNextHyperConnectionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged, and
    // the benchmark drives the disjoint form, so the output never aliases the
    // stream a measured replay must restore.
    unsafe {
        op.launch_write_back(
            stream,
            rows,
            addresses.residual,
            addresses.block_output,
            addresses.write_gate,
            addresses.injected,
        )
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let stream_values = MAX_ROWS * WIDTH;
    let branch_values = MAX_ROWS * BRANCH;
    let projection_values = RANK * WIDTH;
    let residual = layout.reserve(stream_values, ALIGNMENT)?;
    let block_output = layout.reserve(branch_values, ALIGNMENT)?;
    let norm_weight = layout.reserve(WIDTH, ALIGNMENT)?;
    let down = layout.reserve(projection_values, ALIGNMENT)?;
    let up = layout.reserve(projection_values, ALIGNMENT)?;
    let inject = layout.reserve(BRANCHES * WIDTH, ALIGNMENT)?;
    let mixer_norm_weight = layout.reserve(WIDTH, ALIGNMENT)?;
    let mixer_down = layout.reserve(projection_values, ALIGNMENT)?;
    let mixer_up = layout.reserve(projection_values, ALIGNMENT)?;
    let normalized = layout.reserve(stream_values, ALIGNMENT)?;
    let low_rank = layout.reserve(MAX_ROWS * RANK, ALIGNMENT)?;
    let mixed = layout.reserve(branch_values, ALIGNMENT)?;
    let write_gate = layout.reserve(MAX_ROWS * BRANCHES, ALIGNMENT)?;
    let mixer_normalized = layout.reserve(stream_values, ALIGNMENT)?;
    let mixer_low_rank = layout.reserve(MAX_ROWS * RANK, ALIGNMENT)?;
    let mixer_mixed = layout.reserve(branch_values, ALIGNMENT)?;
    let injected = layout.reserve(stream_values, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            residual,
            block_output,
            norm_weight,
            down,
            up,
            inject,
            mixer_norm_weight,
            mixer_down,
            mixer_up,
            normalized,
            low_rank,
            mixed,
            write_gate,
            mixer_normalized,
            mixer_low_rank,
            mixer_mixed,
            injected,
        },
    ))
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

/// Every value the combining gated residual reads or writes, counted per
/// staged entry so the accounting follows the route that actually runs.
fn input_mix_values(rows: usize) -> usize {
    // norm: reads the stream and the gamma row, writes the normalized stream.
    let norm = 2 * rows * WIDTH + WIDTH;
    // mix projection: reads the normalized stream, the down plane, and the
    // inject plane; writes the low rank and the four write gates.
    let down = rows * WIDTH + (RANK + BRANCHES) * WIDTH + rows * (RANK + BRANCHES);
    // fold: reads the normalized stream, the up plane, and the low rank;
    // writes the mixed block input.
    let up = rows * WIDTH + RANK * WIDTH + rows * RANK + rows * BRANCH;

    norm + down + up
}

/// Every value the model-level mixer reads or writes. It is the same module
/// without `block_inject_weight`, so it drops the inject plane and the gates.
fn final_mix_values(rows: usize) -> usize {
    let norm = 2 * rows * WIDTH + WIDTH;
    let down = rows * WIDTH + RANK * WIDTH + rows * RANK;
    let up = rows * WIDTH + RANK * WIDTH + rows * RANK + rows * BRANCH;

    norm + down + up
}

/// Every value the raw-stream write-back reads or writes.
fn write_back_values(rows: usize) -> usize {
    2 * rows * WIDTH + rows * BRANCH + rows * BRANCHES
}

/// Measures every admitted hyper-connection decode batch and prefill tile.
pub fn benchmark_qwen38_flash_next_hyper_connection(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen38_flash_next/hyper_connection/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "two gamma rows, four low-rank planes, and one inject plane",
    )?;
    memory.register_owned(
        "qwen38_flash_next/hyper_connection/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        "max_rows=1024,hc_width=10240,hc_lowrank=320",
    )?;
    memory.register_owned(
        "qwen38_flash_next/hyper_connection/alignment_padding",
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
            suite: "bench-qwen38-flash-next-hyper-connection",
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
        BRANCH, BRANCHES, MAX_ROWS, RANK, ROUTES, WIDTH, final_mix_values, input_mix_values,
        layout, write_back_values,
    };

    #[test]
    fn qwen38_flash_next_hyper_connection_suite_benchmark_arena_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(layout.byte_len(), 127_270_912);
        assert_eq!(
            regions.weight_bytes(),
            (2 * WIDTH + 4 * RANK * WIDTH + BRANCHES * WIDTH) * size_of::<u16>()
        );
        assert_eq!(
            layout.byte_len() - regions.weight_bytes(),
            (4 * MAX_ROWS * WIDTH
                + 3 * MAX_ROWS * BRANCH
                + 2 * MAX_ROWS * RANK
                + MAX_ROWS * BRANCHES)
                * size_of::<u16>()
        );
    }

    /// Byte accounting must name every plane each staged entry touches, or a
    /// per-token throughput is measured against a traffic the route never had.
    #[test]
    fn qwen38_flash_next_hyper_connection_suite_benchmark_byte_accounting_covers_every_read_and_write_plane()
     {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);

        for rows in ROUTES {
            assert_eq!(
                input_mix_values(rows),
                4 * rows * WIDTH
                    + 2 * rows * RANK
                    + rows * BRANCHES
                    + rows * BRANCH
                    + WIDTH * (1 + RANK + BRANCHES + RANK)
            );
            assert_eq!(
                final_mix_values(rows),
                4 * rows * WIDTH + 2 * rows * RANK + rows * BRANCH + WIDTH * (1 + 2 * RANK)
            );
            assert_eq!(
                write_back_values(rows),
                rows * (2 * WIDTH + BRANCH + BRANCHES)
            );
            // The mixer differs from the combining module by exactly the inject
            // plane it does not own and the gates it does not publish.
            assert_eq!(
                input_mix_values(rows) - final_mix_values(rows),
                rows * BRANCHES + BRANCHES * WIDTH
            );
        }
    }
}

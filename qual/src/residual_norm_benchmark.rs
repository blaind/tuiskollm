//! Paired timings for the exact residual-norm graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::residual_norm::ResidualNormLauncher;
#[cfg(feature = "device")]
use crate::target::Qwen35ResidualNormOp;
use crate::target::{EXPECTED_COMPUTE_CAPABILITY, ResidualNormOp};
use std::marker::PhantomData;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer,
};
use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

const MAX_BATCH: usize = 8;
const DECODE_ROUTES: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
#[cfg(feature = "device")]
const QWEN38_MAX_ROWS: usize = 1_024;
#[cfg(not(feature = "device"))]
const QWEN38_MAX_ROWS: usize = MAX_BATCH;
#[cfg(feature = "device")]
const QWEN38_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
#[cfg(not(feature = "device"))]
const QWEN38_ROUTES: [usize; 8] = DECODE_ROUTES;
const INPUT_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const BRANCH_PATTERN: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, -0.25, 0.125, -0.0625, 0.03125,
];
const WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];

struct GraphPair {
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct RouteGraphs {
    batch: usize,
    plain: GraphPair,
    residual: GraphPair,
}

struct Addresses {
    input: *const u16,
    branch: *const u16,
    weight: *const u16,
    plain: *mut u16,
    residual: *mut u16,
    normalized: *mut u16,
}

#[derive(Clone, Copy)]
struct Regions {
    input: tuisko_gpu::ArenaRegion<u16>,
    branch: tuisko_gpu::ArenaRegion<u16>,
    weight: tuisko_gpu::ArenaRegion<u16>,
    plain: tuisko_gpu::ArenaRegion<u16>,
    residual: tuisko_gpu::ArenaRegion<u16>,
    normalized: tuisko_gpu::ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.branch.byte_len()
            + self.weight.byte_len()
            + self.plain.byte_len()
            + self.residual.byte_len()
            + self.normalized.byte_len()
    }

    fn weight_bytes(self) -> usize {
        self.weight.byte_len()
    }
}

struct Session<A: Arch, O: ResidualNormLauncher> {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: O,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
    _arch: PhantomData<A>,
}

impl<A: Arch, O: ResidualNormLauncher> Session<A, O> {
    fn new(
        repeated_operations: u64,
        prepare: fn(&Arc<CudaContext>) -> GpuResult<O>,
        exact_routes: &[usize],
        max_rows: usize,
    ) -> Result<Self, DeviceBenchmarkError> {
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
        let (layout, regions) = layout::<A>(max_rows)?;
        let rows = max_rows * A::HIDDEN;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let input_host = (0..rows)
            .map(|index| f32_to_bf16(INPUT_PATTERN[(index + index / A::HIDDEN) & 7]))
            .collect::<Vec<_>>();
        let branch_host = (0..rows)
            .map(|index| f32_to_bf16(BRANCH_PATTERN[(index * 3) & 7]))
            .collect::<Vec<_>>();
        let weight_host = (0..A::HIDDEN)
            .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
            .collect::<Vec<_>>();
        arena.copy_from_host(&stream, regions.input, &input_host)?;
        arena.copy_from_host(&stream, regions.branch, &branch_host)?;
        arena.copy_from_host(&stream, regions.weight, &weight_host)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = prepare(&context)?;
        let addresses = Addresses {
            input: arena.address(regions.input)?,
            branch: arena.address(regions.branch)?,
            weight: arena.address(regions.weight)?,
            plain: arena.address(regions.plain)?,
            residual: arena.address(regions.residual)?,
            normalized: arena.address(regions.normalized)?,
        };
        let mut routes = Vec::with_capacity(exact_routes.len());
        for &batch in exact_routes {
            routes.push(RouteGraphs {
                batch,
                plain: capture_plain(&op, &stream, &addresses, batch, repeated_operations)?,
                residual: capture_residual(&op, &stream, &addresses, batch, repeated_operations)?,
            });
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
            _arch: PhantomData,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.plain.leaf.launch(&self.stream) }?;
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.residual.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(
        &self,
        repeated_operations: u64,
        plain_route: &'static str,
        residual_route: &'static str,
    ) -> Vec<ExactDeviceCase<'_>> {
        let mut cases = Vec::with_capacity(self.routes.len() * 2);
        for route in &self.routes {
            let (shape, workload) = if route.batch <= MAX_BATCH {
                (
                    format!("B={}", route.batch),
                    BenchmarkWorkload::warm_operator_decode(route.batch as u32),
                )
            } else {
                (
                    format!("T={}", route.batch),
                    BenchmarkWorkload::warm_operator_prefill(route.batch as u64),
                )
            };
            cases.push(ExactDeviceCase::new(
                plain_route,
                shape.clone(),
                workload.clone(),
                OperationAccounting::new(
                    logical_bytes::<A>(route.batch, false),
                    route.batch as u64,
                    "token",
                ),
                &route.plain.leaf,
                Some(RepeatedGraph::new(
                    &route.plain.repeated,
                    repeated_operations,
                )),
            ));
            cases.push(ExactDeviceCase::new(
                residual_route,
                shape,
                workload,
                OperationAccounting::new(
                    logical_bytes::<A>(route.batch, true),
                    route.batch as u64,
                    "token",
                ),
                &route.residual.leaf,
                Some(RepeatedGraph::new(
                    &route.residual.repeated,
                    repeated_operations,
                )),
            ));
        }

        cases
    }
}

fn layout<A: Arch>(max_rows: usize) -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let rows = max_rows * A::HIDDEN;
    let input = layout.reserve(rows, 256)?;
    let branch = layout.reserve(rows, 256)?;
    let weight = layout.reserve(A::HIDDEN, 256)?;
    let plain = layout.reserve(rows, 256)?;
    let residual = layout.reserve(rows, 256)?;
    let normalized = layout.reserve(rows, 256)?;

    Ok((
        layout,
        Regions {
            input,
            branch,
            weight,
            plain,
            residual,
            normalized,
        },
    ))
}

fn capture_plain<O: ResidualNormLauncher>(
    op: &O,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
    repeated_operations: u64,
) -> GpuResult<GraphPair> {
    let leaf = CudaGraph::capture(stream, || {
        // SAFETY: every pointer names a complete, aligned arena region.
        unsafe {
            op.launch_plain(
                stream,
                batch,
                addresses.input,
                addresses.weight,
                addresses.plain,
            )
        }
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            // SAFETY: every pointer names a complete, aligned arena region.
            unsafe {
                op.launch_plain(
                    stream,
                    batch,
                    addresses.input,
                    addresses.weight,
                    addresses.plain,
                )?;
            }
        }

        Ok(())
    })?;

    Ok(GraphPair { leaf, repeated })
}

fn capture_residual<O: ResidualNormLauncher>(
    op: &O,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
    repeated_operations: u64,
) -> GpuResult<GraphPair> {
    let leaf = CudaGraph::capture(stream, || {
        // SAFETY: every pointer names a complete, aligned arena region.
        unsafe {
            op.launch_residual(
                stream,
                batch,
                addresses.input,
                addresses.branch,
                addresses.weight,
                addresses.residual,
                addresses.normalized,
            )
        }
    })?;
    let repeated = CudaGraph::capture(stream, || {
        for _ in 0..repeated_operations {
            // SAFETY: every pointer names a complete, aligned arena region.
            unsafe {
                op.launch_residual(
                    stream,
                    batch,
                    addresses.input,
                    addresses.branch,
                    addresses.weight,
                    addresses.residual,
                    addresses.normalized,
                )?;
            }
        }

        Ok(())
    })?;

    Ok(GraphPair { leaf, repeated })
}

/// Measures every admitted decode and target-specific prefill route directly.
pub fn benchmark_residual_norm(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen38_27B, ResidualNormOp>(
        options,
        ResidualNormOp::new,
        "bench-residual-norm",
        "residual_norm/plain",
        "residual_norm/fused_residual",
        "residual_norm/address_stable_workspace",
        "max_rows=1024,hidden=5120",
        "residual_norm/weights",
        "residual_norm/alignment_padding",
        &QWEN38_ROUTES,
        QWEN38_MAX_ROWS,
    )
}

/// Measures the exact Qwen3.5 residual-norm routes on SM120.
#[cfg(feature = "device")]
pub fn benchmark_qwen35_residual_norm(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    benchmark_target::<Qwen35_9B, Qwen35ResidualNormOp>(
        options,
        Qwen35ResidualNormOp::new,
        "bench-qwen35-residual-norm",
        "qwen35_9b/residual_norm/plain",
        "qwen35_9b/residual_norm/fused_residual",
        "qwen35_9b/residual_norm/address_stable_workspace",
        "max_batch=8,hidden=4096",
        "qwen35_9b/residual_norm/weights",
        "qwen35_9b/residual_norm/alignment_padding",
        &DECODE_ROUTES,
        MAX_BATCH,
    )
}

#[allow(clippy::too_many_arguments)]
fn benchmark_target<A: Arch, O: ResidualNormLauncher>(
    options: DeviceBenchmarkOptions,
    prepare: fn(&Arc<CudaContext>) -> GpuResult<O>,
    suite: &'static str,
    plain_route: &'static str,
    residual_route: &'static str,
    workspace_name: &'static str,
    arena_scaling: &'static str,
    weight_name: &'static str,
    padding_name: &'static str,
    exact_routes: &[usize],
    max_rows: usize,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session =
        Session::<A, O>::new(options.launches_per_sample, prepare, exact_routes, max_rows)?;
    let weight_bytes = session.regions.weight_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        weight_name,
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "one zero-centered BF16 hidden-width row",
    )?;
    memory.register_owned(
        workspace_name,
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - padding_bytes,
        arena_scaling,
    )?;
    memory.register_owned(
        padding_name,
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases(options.launches_per_sample, plain_route, residual_route);
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite,
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

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

fn logical_bytes<A: Arch>(batch: usize, fused_residual: bool) -> usize {
    let planes = if fused_residual { 5 } else { 3 };

    planes * batch * A::HIDDEN * size_of::<u16>()
}

#[cfg(test)]
mod tests {
    use super::{MAX_BATCH, QWEN38_MAX_ROWS, Qwen35_9B, Qwen38_27B, layout, logical_bytes};
    use tuisko_model::Arch;

    #[test]
    fn byte_accounting_covers_every_read_and_write_plane() {
        assert_eq!(logical_bytes::<Qwen38_27B>(8, false), 6 * 8 * 5_120);
        assert_eq!(logical_bytes::<Qwen38_27B>(8, true), 10 * 8 * 5_120);
        assert_eq!(logical_bytes::<Qwen35_9B>(8, false), 6 * 8 * 4_096);
        assert_eq!(logical_bytes::<Qwen35_9B>(8, true), 10 * 8 * 4_096);
    }

    #[test]
    fn residual_norm_suite_benchmark_arena_accounting_exposes_every_byte() {
        let (qwen38_layout, regions) = layout::<Qwen38_27B>(QWEN38_MAX_ROWS).unwrap();
        assert_eq!(qwen38_layout.byte_len(), regions.payload_bytes());
        assert_eq!(regions.weight_bytes(), 10_240);
        assert_eq!(
            qwen38_layout.byte_len(),
            5 * QWEN38_MAX_ROWS * Qwen38_27B::HIDDEN * size_of::<u16>() + regions.weight_bytes()
        );

        let (qwen35_layout, qwen35_regions) = layout::<Qwen35_9B>(MAX_BATCH).unwrap();
        assert_eq!(qwen35_layout.byte_len(), qwen35_regions.payload_bytes());
        assert_eq!(qwen35_regions.weight_bytes(), 8_192);
    }
}

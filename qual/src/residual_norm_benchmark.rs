//! Paired timings for the exact residual-norm graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult, GpuTimer,
};
use tuisko_kernels_sm120::ResidualNormOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
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

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: ResidualNormOp,
    _arena: DeviceArena,
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
        let mut layout = ArenaLayout::new();
        let rows = MAX_BATCH * Qwen38_27B::HIDDEN;
        let input = layout.reserve::<u16>(rows, 256)?;
        let branch = layout.reserve::<u16>(rows, 256)?;
        let weight = layout.reserve::<u16>(Qwen38_27B::HIDDEN, 256)?;
        let plain = layout.reserve::<u16>(rows, 256)?;
        let residual = layout.reserve::<u16>(rows, 256)?;
        let normalized = layout.reserve::<u16>(rows, 256)?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let input_host = (0..rows)
            .map(|index| f32_to_bf16(INPUT_PATTERN[(index + index / Qwen38_27B::HIDDEN) & 7]))
            .collect::<Vec<_>>();
        let branch_host = (0..rows)
            .map(|index| f32_to_bf16(BRANCH_PATTERN[(index * 3) & 7]))
            .collect::<Vec<_>>();
        let weight_host = (0..Qwen38_27B::HIDDEN)
            .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
            .collect::<Vec<_>>();
        arena.copy_from_host(&stream, input, &input_host)?;
        arena.copy_from_host(&stream, branch, &branch_host)?;
        arena.copy_from_host(&stream, weight, &weight_host)?;
        stream.synchronize().map_err(GpuError::from)?;

        let op = ResidualNormOp::new(&context)?;
        let addresses = Addresses {
            input: arena.address(input)?,
            branch: arena.address(branch)?,
            weight: arena.address(weight)?,
            plain: arena.address(plain)?,
            residual: arena.address(residual)?,
            normalized: arena.address(normalized)?,
        };
        let mut routes = Vec::with_capacity(MAX_BATCH);
        for batch in 1..=MAX_BATCH {
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
            _arena: arena,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                route.plain.leaf.launch(&self.stream)?;
                route.residual.leaf.launch(&self.stream)?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self, repeated_operations: u64) -> Vec<ExactDeviceCase<'_>> {
        let mut cases = Vec::with_capacity(self.routes.len() * 2);
        for route in &self.routes {
            let shape = format!("B={}", route.batch);
            cases.push(ExactDeviceCase::new(
                "residual_norm/plain",
                shape.clone(),
                BenchmarkWorkload::warm_operator_decode(route.batch as u32),
                OperationAccounting::new(
                    6 * route.batch * Qwen38_27B::HIDDEN,
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
                "residual_norm/fused_residual",
                shape,
                BenchmarkWorkload::warm_operator_decode(route.batch as u32),
                OperationAccounting::new(
                    10 * route.batch * Qwen38_27B::HIDDEN,
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

fn capture_plain(
    op: &ResidualNormOp,
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

fn capture_residual(
    op: &ResidualNormOp,
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

/// Measures all exact batches with paired host/device and repeated-path timings.
pub fn benchmark_residual_norm(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    memory.register_owned(
        "residual_norm/address_stable_arena",
        BenchmarkMemoryKind::Workspace,
        session._arena.byte_len(),
        "max_batch=8",
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
            suite: "bench-residual-norm",
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

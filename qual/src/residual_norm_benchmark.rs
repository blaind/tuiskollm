//! Paired timings for the exact residual-norm graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkOptions,
    DeviceBenchmarkReport, ExactDeviceCase, IntrinsicGraph, MemoryRecorder, OperationAccounting,
    executable_sha256, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive,
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
    intrinsic: CudaGraph,
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
    fn new(intrinsic_nodes: u64) -> Result<Self, DeviceBenchmarkError> {
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
                plain: capture_plain(&op, &stream, &addresses, batch, intrinsic_nodes)?,
                residual: capture_residual(&op, &stream, &addresses, batch, intrinsic_nodes)?,
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

    fn cases(&self, intrinsic_nodes: u64) -> Vec<ExactDeviceCase<'_>> {
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
                Some(IntrinsicGraph::new(&route.plain.intrinsic, intrinsic_nodes)),
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
                Some(IntrinsicGraph::new(
                    &route.residual.intrinsic,
                    intrinsic_nodes,
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
    intrinsic_nodes: u64,
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
    let intrinsic = CudaGraph::capture(stream, || {
        for _ in 0..intrinsic_nodes {
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

    Ok(GraphPair { leaf, intrinsic })
}

fn capture_residual(
    op: &ResidualNormOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
    intrinsic_nodes: u64,
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
    let intrinsic = CudaGraph::capture(stream, || {
        for _ in 0..intrinsic_nodes {
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

    Ok(GraphPair { leaf, intrinsic })
}

/// Measures all exact batches with paired host/device and repeated-node timings.
pub fn benchmark_residual_norm(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.samples < 3 || options.launches_per_sample == 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "at least three samples and one launch per sample are required".to_string(),
        ));
    }

    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = options
        .launches_per_sample
        .checked_mul(4)
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition("warmup launch count overflows".to_string())
        })?
        .max(1_024);
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
    let identity = preflight.identity;

    Ok(DeviceBenchmarkReport {
        schema_version: 4,
        suite: "bench-residual-norm",
        classification: "performance_sensitive_leaf",
        device: identity.name,
        device_uuid: identity.uuid,
        driver_version: identity.driver_version,
        device_index: 0,
        compute_capability: "12.0".to_string(),
        binary_sha256: executable_sha256()?,
        generator_baseline_sha256: baseline_sha256,
        sm_clock_min_mhz: telemetry.sm_minimum_mhz,
        sm_clock_median_mhz: telemetry.sm_median_mhz,
        sm_clock_max_mhz: telemetry.sm_maximum_mhz,
        memory_clock_min_mhz: telemetry.memory_minimum_mhz,
        memory_clock_median_mhz: telemetry.memory_median_mhz,
        memory_clock_max_mhz: telemetry.memory_maximum_mhz,
        temperature_min_celsius: telemetry.temperature_minimum_celsius,
        temperature_max_celsius: telemetry.temperature_maximum_celsius,
        power_min_watts: telemetry.power_minimum_watts,
        power_mean_watts: telemetry.power_mean_watts,
        power_median_watts: telemetry.power_median_watts,
        power_max_watts: telemetry.power_maximum_watts,
        telemetry_samples: telemetry.samples,
        samples: options.samples,
        launches_per_sample: options.launches_per_sample,
        timing_scope: "paired Rust submission/completion, repeated leaf graph, and repeated-node graph",
        power_scope: "nvidia-smi power.draw.instant, whole board",
        metrics,
        energy_metrics,
        memory,
    })
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

//! Paired timings for exact FP32 GDN recurrence graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, RepeatedGraph, finish_report, generator_baseline_sha256, measure_cases,
    preflight, require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer,
};
use tuisko_kernels_sm120::GdnRecurrenceOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const HEAD_DIM: usize = 128;
const VALUE_HEADS: usize = 48;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const STATE_ROWS: [u32; MAX_BATCH] = [7, 0, 5, 2, 6, 1, 4, 3];
const VALUES: [f32; 8] = [0.75, -0.625, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625];

#[derive(Clone, Copy)]
struct Regions {
    qkv: ArenaRegion<u16>,
    projected: ArenaRegion<u16>,
    log_decay: ArenaRegion<f32>,
    beta: ArenaRegion<f32>,
    norm_weight: ArenaRegion<u16>,
    state_rows: ArenaRegion<u32>,
    state: ArenaRegion<f32>,
    output: ArenaRegion<u16>,
}

struct Addresses {
    qkv: *const u16,
    projected: *const u16,
    log_decay: *const f32,
    beta: *const f32,
    norm_weight: *const u16,
    state_rows: *const u32,
    state: *mut f32,
    output: *mut u16,
}

struct RouteGraphs {
    batch: usize,
    leaf: CudaGraph,
    repeated: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: GdnRecurrenceOp,
    arena: DeviceArena,
    regions: Regions,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new(repeated_operations: u64) -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(
                "device zero is not the exact compute-capability 12.0 target".into(),
            ));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        load_fixture(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = GdnRecurrenceOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = (1..=MAX_BATCH)
            .map(|batch| capture_route(&op, &stream, &addresses, batch, repeated_operations))
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
                    "gdn_recurrence/state_gate_norm",
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
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_BATCH * VALUE_HEADS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_BATCH * VALUE_HEADS, ALIGNMENT)?;
    let norm_weight = layout.reserve(HEAD_DIM, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let state = layout.reserve(MAX_BATCH * STATE_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * VALUE_WIDTH, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            qkv,
            projected,
            log_decay,
            beta,
            norm_weight,
            state_rows,
            state,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    let qkv = (0..MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS)
        .map(|index| f32_to_bf16(VALUES[(3 * index + index / Qwen38_27B::GDN_QKV_ROWS) & 7]))
        .collect::<Vec<_>>();
    let projected = (0..MAX_BATCH * Qwen38_27B::GDN_INPUT_ROWS)
        .map(|index| f32_to_bf16(VALUES[(5 * index + 1) & 7]))
        .collect::<Vec<_>>();
    let log_decay = (0..MAX_BATCH * VALUE_HEADS)
        .map(|index| -0.125 - (index & 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let beta = (0..MAX_BATCH * VALUE_HEADS)
        .map(|index| 0.25 + (index & 3) as f32 * 0.125)
        .collect::<Vec<_>>();
    let norm = (0..HEAD_DIM)
        .map(|index| f32_to_bf16(0.75 + (index & 3) as f32 * 0.125))
        .collect::<Vec<_>>();
    let state = (0..MAX_BATCH * STATE_PER_ROW)
        .map(|index| (((13 * index) & 31) as f32 - 15.5) / 2048.0)
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.qkv, &qkv)?;
    arena.copy_from_host(stream, regions.projected, &projected)?;
    arena.copy_from_host(stream, regions.log_decay, &log_decay)?;
    arena.copy_from_host(stream, regions.beta, &beta)?;
    arena.copy_from_host(stream, regions.norm_weight, &norm)?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)?;
    arena.copy_from_host(stream, regions.state, &state)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        qkv: arena.address(regions.qkv)?,
        projected: arena.address(regions.projected)?,
        log_decay: arena.address(regions.log_decay)?,
        beta: arena.address(regions.beta)?,
        norm_weight: arena.address(regions.norm_weight)?,
        state_rows: arena.address(regions.state_rows)?,
        state: arena.address(regions.state)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &GdnRecurrenceOp,
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
    op: &GdnRecurrenceOp,
    stream: &CudaStream,
    addresses: &Addresses,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: pointers cover maximum extents and mapped rows are below eight.
    unsafe {
        op.launch(
            stream,
            batch,
            addresses.qkv,
            addresses.projected,
            addresses.log_decay,
            addresses.beta,
            addresses.norm_weight,
            addresses.state_rows,
            addresses.state,
            addresses.output,
        )
    }
}

fn logical_bytes(batch: usize) -> usize {
    let per_token = 2 * Qwen38_27B::GDN_QKV_ROWS
        + 2 * VALUE_WIDTH
        + 2 * VALUE_HEADS * size_of::<f32>()
        + size_of::<u32>()
        + 2 * STATE_PER_ROW * size_of::<f32>()
        + VALUE_WIDTH * size_of::<u16>();

    HEAD_DIM * size_of::<u16>() + batch * per_token
}

/// Measures every exact recurrence route with paired host/device timings.
pub fn benchmark_gdn_recurrence(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new(options.launches_per_sample)?;
    memory.register_owned(
        "gdn_recurrence/norm_weight",
        BenchmarkMemoryKind::Weights,
        session.regions.norm_weight.byte_len(),
        "one BF16 head-width norm vector",
    )?;
    memory.register_owned(
        "gdn_recurrence/address_stable_state_workspace",
        BenchmarkMemoryKind::Other,
        session.arena.byte_len() - session.regions.norm_weight.byte_len(),
        "eight FP32 state rows and exact-B workspace",
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
            suite: "bench-gdn-recurrence",
            classification: "performance_sensitive_stateful_leaf",
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

#[cfg(test)]
mod tests {
    use super::{HEAD_DIM, MAX_BATCH, STATE_PER_ROW, VALUE_HEADS, VALUE_WIDTH, logical_bytes};
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    fn byte_accounting_includes_the_complete_state_transition() {
        let per_token = 2 * Qwen38_27B::GDN_QKV_ROWS
            + 2 * VALUE_WIDTH
            + 2 * VALUE_HEADS * 4
            + 4
            + 2 * STATE_PER_ROW * 4
            + VALUE_WIDTH * 2;
        assert_eq!(logical_bytes(1), HEAD_DIM * 2 + per_token);
        assert_eq!(
            logical_bytes(MAX_BATCH),
            HEAD_DIM * 2 + MAX_BATCH * per_token
        );
    }
}

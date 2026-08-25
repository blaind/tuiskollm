//! Paired timings for exact FP32 GDN recurrence graph routes.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::fp8_projection_oracle::f32_to_bf16;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer, PinnedHostBuffer,
};
use tuisko_kernels_sm120::GdnRecurrenceOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const ALIGNMENT: usize = 256;
const HEAD_DIM: usize = 128;
const VALUE_HEADS: usize = 48;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const STATE_ROWS: [u32; MAX_BATCH] = [0, 1, 2, 3, 4, 5, 6, 7];
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
    recurrent_plane: ArenaRegion<f32>,
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
    recurrent_plane: *mut f32,
    output: *mut u16,
}

struct RouteGraphs {
    rows: usize,
    preparation: CudaGraph,
    leaf: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _op: GdnRecurrenceOp,
    arena: DeviceArena,
    regions: Regions,
    _state_seed: PinnedHostBuffer<f32>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new() -> Result<Self, DeviceBenchmarkError> {
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err(DeviceBenchmarkError::Precondition(
                "device zero is not the exact compute-capability 12.0 target".into(),
            ));
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let (layout, regions) = layout()?;
        let arena = DeviceArena::zeroed(&stream, &layout)?;
        let state = load_fixture(&arena, &stream, regions)?;
        let state_seed = PinnedHostBuffer::from_slice(&context, &state).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = GdnRecurrenceOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| capture_route(&op, &stream, &arena, regions, &state_seed, &addresses, rows))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            routes,
            _op: op,
            arena,
            regions,
            _state_seed: state_seed,
            stream,
            _context: context,
        })
    }

    fn warm(&self, launches: u64) -> GpuResult<()> {
        for _ in 0..launches {
            for route in &self.routes {
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.preparation.launch(&self.stream) }?;
                // SAFETY: this Session owns both these route graphs and everything they
                // captured (arena, maps, op modules), dropping the graphs first.
                unsafe { route.leaf.launch(&self.stream) }?;
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self) -> Vec<ExactDeviceCase<'_>> {
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
                    "gdn_recurrence/state_gate_norm",
                    shape,
                    workload,
                    OperationAccounting::new(logical_bytes(route.rows), route.rows as u64, "token"),
                    &route.leaf,
                    None,
                )
                .with_preparation(&route.preparation)
            })
            .collect()
    }

    fn weight_bytes(&self) -> usize {
        self.regions.norm_weight.byte_len()
    }

    fn state_bytes(&self) -> usize {
        self.regions.state.byte_len()
    }

    fn workspace_bytes(&self) -> usize {
        self.regions.payload_bytes() - self.weight_bytes() - self.state_bytes()
    }

    fn padding_bytes(&self) -> usize {
        self.arena.byte_len() - self.regions.payload_bytes()
    }
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.qkv.byte_len()
            + self.projected.byte_len()
            + self.log_decay.byte_len()
            + self.beta.byte_len()
            + self.norm_weight.byte_len()
            + self.state_rows.byte_len()
            + self.state.byte_len()
            + self.output.byte_len()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let qkv = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let log_decay = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * VALUE_HEADS, ALIGNMENT)?;
    let norm_weight = layout.reserve(HEAD_DIM, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let state = layout.reserve(MAX_BATCH * STATE_PER_ROW, ALIGNMENT)?;
    let recurrent_plane = layout.reserve(MAX_ROWS * VALUE_WIDTH, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * VALUE_WIDTH, ALIGNMENT)?;

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
            recurrent_plane,
            output,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Vec<f32>> {
    let qkv = (0..MAX_ROWS * Qwen38_27B::GDN_QKV_ROWS)
        .map(|index| f32_to_bf16(VALUES[(3 * index + index / Qwen38_27B::GDN_QKV_ROWS) & 7]))
        .collect::<Vec<_>>();
    let projected = (0..MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS)
        .map(|index| f32_to_bf16(VALUES[(5 * index + 1) & 7]))
        .collect::<Vec<_>>();
    let log_decay = (0..MAX_ROWS * VALUE_HEADS)
        .map(|index| -0.125 - (index & 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let beta = (0..MAX_ROWS * VALUE_HEADS)
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
    arena.copy_from_host(stream, regions.state, &state)?;

    Ok(state)
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
        recurrent_plane: arena.address(regions.recurrent_plane)?,
        output: arena.address(regions.output)?,
    })
}

fn capture_route(
    op: &GdnRecurrenceOp,
    stream: &CudaStream,
    arena: &DeviceArena,
    regions: Regions,
    state_seed: &PinnedHostBuffer<f32>,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<RouteGraphs> {
    let state_rows = if rows <= MAX_BATCH { rows } else { 1 };
    let state_values = state_rows * STATE_PER_ROW;
    let preparation = CudaGraph::capture(stream, || {
        // SAFETY: Session owns the immutable pinned seed through every graph
        // replay, and both regions cover the exact active-state prefix.
        unsafe {
            arena.copy_prefix_from_pinned_host_async(
                stream,
                regions.state,
                state_seed,
                state_values,
            )
        }
    })?;
    let leaf = CudaGraph::capture(stream, || launch(op, stream, addresses, rows))?;

    Ok(RouteGraphs {
        rows,
        preparation,
        leaf,
    })
}

fn launch(
    op: &GdnRecurrenceOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: pointers cover maximum extents and mapped rows are below eight.
    unsafe {
        op.launch(
            stream,
            rows,
            addresses.qkv,
            addresses.projected,
            addresses.log_decay,
            addresses.beta,
            addresses.norm_weight,
            addresses.state_rows,
            addresses.state,
            addresses.recurrent_plane,
            addresses.output,
        )
    }
}

fn logical_bytes(rows: usize) -> usize {
    let per_token = 2 * Qwen38_27B::GDN_QKV_ROWS
        + 2 * VALUE_WIDTH
        + 2 * VALUE_HEADS * size_of::<f32>()
        + size_of::<u32>()
        + 2 * STATE_PER_ROW * size_of::<f32>()
        + VALUE_WIDTH * size_of::<u16>();

    HEAD_DIM * size_of::<u16>() + rows * per_token
}

/// Measures every exact recurrence route with paired host/device timings.
pub fn benchmark_gdn_recurrence(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new()?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.weight_bytes();
    let state_bytes = session.state_bytes();
    let workspace_bytes = session.workspace_bytes();
    let padding_bytes = session.padding_bytes();
    memory.register_owned(
        "gdn_recurrence/norm_weight",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "one BF16 head-width norm vector",
    )?;
    memory.register_owned(
        "gdn_recurrence/address_stable_state",
        BenchmarkMemoryKind::Other,
        state_bytes,
        "eight FP32 recurrent state rows",
    )?;
    memory.register_owned(
        "gdn_recurrence/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_rows=1024 inputs, controls, mapping, and output",
    )?;
    memory.register_owned(
        "gdn_recurrence/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte region alignment",
    )?;
    memory.capture("after_setup")?;
    session.warm(warmup_launches)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;
    let cases = session.cases();
    let (metrics, energy_metrics, telemetry) =
        measure_cases(&session.stream, &mut timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-gdn-recurrence",
            classification: "performance_sensitive_stateful_leaf",
            timing_scope: "paired Rust submission/completion and production graph after untimed exact-state restore",
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
        EXACT_ROUTES, HEAD_DIM, MAX_BATCH, MAX_ROWS, STATE_PER_ROW, VALUE_HEADS, VALUE_WIDTH,
        layout, logical_bytes,
    };
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
        assert_eq!(logical_bytes(MAX_ROWS), HEAD_DIM * 2 + MAX_ROWS * per_token);
    }

    #[test]
    fn benchmark_route_inventory_and_owner_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.norm_weight.byte_len(), 256);
        assert_eq!(regions.state.byte_len(), 25_165_824);
        assert_eq!(regions.payload_bytes(), 92_668_192);
        assert_eq!(layout.byte_len(), 92_668_416);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 224);
    }
}

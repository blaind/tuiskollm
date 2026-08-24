//! Paired timings for exact GDN control and convolution graph routes.

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
use tuisko_kernels_sm120::GdnPrepareOp;
use tuisko_model::{Arch, Qwen38_27B};

const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, MAX_BATCH, 32, 64, 128, MAX_ROWS];
const ALIGNMENT: usize = 256;
const HISTORY_TAPS: usize = 3;
const STATE_ROWS: [u32; MAX_BATCH] = [0, 1, 2, 3, 4, 5, 6, 7];
const VALUES: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, 0.1875, -0.09375, 0.046875, 0.0,
];

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    control_weights: ArenaRegion<u16>,
    a_log: ArenaRegion<u16>,
    dt_bias: ArenaRegion<u16>,
    projected: ArenaRegion<u16>,
    convolution_weights: ArenaRegion<u16>,
    state_rows: ArenaRegion<u32>,
    history: ArenaRegion<u16>,
    log_decay: ArenaRegion<f32>,
    beta: ArenaRegion<f32>,
    convolved: ArenaRegion<u16>,
}

struct Addresses {
    input: *const u16,
    control_weights: *const u16,
    a_log: *const u16,
    dt_bias: *const u16,
    projected: *const u16,
    convolution_weights: *const u16,
    state_rows: *const u32,
    history: *mut u16,
    log_decay: *mut f32,
    beta: *mut f32,
    convolved: *mut u16,
}

struct RouteGraphs {
    rows: usize,
    preparation: CudaGraph,
    leaf: CudaGraph,
}

struct Session {
    routes: Vec<RouteGraphs>,
    timer: GpuTimer,
    _op: GdnPrepareOp,
    arena: DeviceArena,
    regions: Regions,
    _history_seed: PinnedHostBuffer<u16>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new() -> Result<Self, DeviceBenchmarkError> {
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
        let history = load_fixture(&arena, &stream, regions)?;
        let history_seed =
            PinnedHostBuffer::from_slice(&context, &history).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;
        let op = GdnPrepareOp::new(&context)?;
        let addresses = addresses(&arena, regions)?;
        let routes = EXACT_ROUTES
            .into_iter()
            .map(|rows| {
                capture_route(
                    &op,
                    &stream,
                    &arena,
                    regions,
                    &history_seed,
                    &addresses,
                    rows,
                )
            })
            .collect::<GpuResult<Vec<_>>>()?;
        let timer = GpuTimer::new(&context)?;

        Ok(Self {
            routes,
            timer,
            _op: op,
            arena,
            regions,
            _history_seed: history_seed,
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
                    "gdn_prepare/control_convolution",
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
        self.regions.control_weights.byte_len()
            + self.regions.a_log.byte_len()
            + self.regions.dt_bias.byte_len()
            + self.regions.convolution_weights.byte_len()
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
            + self.control_weights.byte_len()
            + self.a_log.byte_len()
            + self.dt_bias.byte_len()
            + self.projected.byte_len()
            + self.convolution_weights.byte_len()
            + self.state_rows.byte_len()
            + self.history.byte_len()
            + self.log_decay.byte_len()
            + self.beta.byte_len()
            + self.convolved.byte_len()
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * Qwen38_27B::HIDDEN, ALIGNMENT)?;
    let control_weights = layout.reserve(
        2 * Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::HIDDEN,
        ALIGNMENT,
    )?;
    let a_log = layout.reserve(Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let dt_bias = layout.reserve(Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let projected = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS, ALIGNMENT)?;
    let convolution_weights = layout.reserve(Qwen38_27B::GDN_QKV_ROWS * 4, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_BATCH, ALIGNMENT)?;
    let history = layout.reserve(
        MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS,
        ALIGNMENT,
    )?;
    let log_decay = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let beta = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_CONTROL_ROWS, ALIGNMENT)?;
    let convolved = layout.reserve(MAX_ROWS * Qwen38_27B::GDN_QKV_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            control_weights,
            a_log,
            dt_bias,
            projected,
            convolution_weights,
            state_rows,
            history,
            log_decay,
            beta,
            convolved,
        },
    ))
}

fn load_fixture(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Vec<u16>> {
    let input = (0..MAX_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| {
            let token = index / Qwen38_27B::HIDDEN;
            f32_to_bf16(VALUES[(index + token) & 7])
        })
        .collect::<Vec<_>>();
    let control_weights = (0..2 * Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(VALUES[(3 * index) & 7] / 64.0))
        .collect::<Vec<_>>();
    let a_log = vec![f32_to_bf16(-2.0); Qwen38_27B::GDN_CONTROL_ROWS];
    let dt_bias = vec![f32_to_bf16(0.03125); Qwen38_27B::GDN_CONTROL_ROWS];
    let projected = (0..MAX_ROWS * Qwen38_27B::GDN_INPUT_ROWS)
        .map(|index| {
            let token = index / Qwen38_27B::GDN_INPUT_ROWS;
            f32_to_bf16(VALUES[(5 * index + token) & 7])
        })
        .collect::<Vec<_>>();
    let convolution_weights = (0..Qwen38_27B::GDN_QKV_ROWS * 4)
        .map(|index| f32_to_bf16([0.5, -0.25, 0.125, 0.25][index & 3]))
        .collect::<Vec<_>>();
    let history = (0..MAX_BATCH * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS)
        .map(|index| f32_to_bf16(VALUES[(7 * index) & 7] * 0.5))
        .collect::<Vec<_>>();

    arena.copy_from_host(stream, regions.input, &input)?;
    arena.copy_from_host(stream, regions.control_weights, &control_weights)?;
    arena.copy_from_host(stream, regions.a_log, &a_log)?;
    arena.copy_from_host(stream, regions.dt_bias, &dt_bias)?;
    arena.copy_from_host(stream, regions.projected, &projected)?;
    arena.copy_from_host(stream, regions.convolution_weights, &convolution_weights)?;
    arena.copy_from_host(stream, regions.state_rows, &STATE_ROWS)?;
    arena.copy_from_host(stream, regions.history, &history)?;

    Ok(history)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Addresses> {
    Ok(Addresses {
        input: arena.address(regions.input)?,
        control_weights: arena.address(regions.control_weights)?,
        a_log: arena.address(regions.a_log)?,
        dt_bias: arena.address(regions.dt_bias)?,
        projected: arena.address(regions.projected)?,
        convolution_weights: arena.address(regions.convolution_weights)?,
        state_rows: arena.address(regions.state_rows)?,
        history: arena.address(regions.history)?,
        log_decay: arena.address(regions.log_decay)?,
        beta: arena.address(regions.beta)?,
        convolved: arena.address(regions.convolved)?,
    })
}

fn capture_route(
    op: &GdnPrepareOp,
    stream: &CudaStream,
    arena: &DeviceArena,
    regions: Regions,
    history_seed: &PinnedHostBuffer<u16>,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<RouteGraphs> {
    let state_rows = if rows <= MAX_BATCH { rows } else { 1 };
    let history_values = state_rows * Qwen38_27B::GDN_QKV_ROWS * HISTORY_TAPS;
    let preparation = CudaGraph::capture(stream, || {
        // SAFETY: the pinned seed remains immutable and owned by Session through
        // every replay; both source and destination cover `history_values`.
        unsafe {
            arena.copy_prefix_from_pinned_host_async(
                stream,
                regions.history,
                history_seed,
                history_values,
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
    op: &GdnPrepareOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: pointers cover aligned maximum-batch regions and all mapped state
    // rows are below the eight-row history capacity.
    unsafe {
        op.launch(
            stream,
            rows,
            addresses.input,
            addresses.control_weights,
            addresses.a_log,
            addresses.dt_bias,
            addresses.projected,
            addresses.convolution_weights,
            addresses.state_rows,
            addresses.history,
            addresses.log_decay,
            addresses.beta,
            addresses.convolved,
        )
    }
}

fn logical_bytes(rows: usize) -> usize {
    let hidden = Qwen38_27B::HIDDEN;
    let controls = Qwen38_27B::GDN_CONTROL_ROWS;
    let qkv = Qwen38_27B::GDN_QKV_ROWS;
    let weights = 2 * controls * hidden * size_of::<u16>()
        + 2 * controls * size_of::<u16>()
        + qkv * 4 * size_of::<u16>();
    let control_per_token =
        hidden * size_of::<u16>() + 2 * controls * size_of::<f32>() + size_of::<u32>();
    let convolution = if rows <= MAX_BATCH {
        rows * (8 * qkv * size_of::<u16>())
    } else {
        rows * (5 * qkv * size_of::<u16>()) + 6 * qkv * size_of::<u16>() + size_of::<u32>()
    };

    weights + rows * control_per_token + convolution
}

/// Measures every exact GDN prepare batch with paired host/device timings.
pub fn benchmark_gdn_prepare(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new()?;
    let weight_bytes = session.weight_bytes();
    let workspace_bytes = session.workspace_bytes();
    let padding_bytes = session.padding_bytes();
    memory.register_owned(
        "gdn_prepare/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "BF16 A/B controls and width-4 convolution",
    )?;
    memory.register_owned(
        "gdn_prepare/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        workspace_bytes,
        "max_rows=1024,eight mapped history rows",
    )?;
    memory.register_owned(
        "gdn_prepare/alignment_padding",
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
        measure_cases(&session.stream, &session.timer, &cases, options)?;
    let memory = memory.finish(&telemetry)?;

    finish_report(
        BenchmarkReportSpec {
            suite: "bench-gdn-prepare",
            classification: "performance_sensitive_leaf_pair",
            timing_scope: "paired Rust submission/completion and production graph after untimed exact-history restore",
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
    fn byte_accounting_covers_controls_convolution_and_history() {
        let controls = Qwen38_27B::GDN_CONTROL_ROWS;
        let qkv = Qwen38_27B::GDN_QKV_ROWS;
        let weights = 2 * controls * Qwen38_27B::HIDDEN * 2 + 2 * controls * 2 + qkv * 4 * 2;
        let control_per_token = Qwen38_27B::HIDDEN * 2 + 2 * controls * 4 + 4;
        let decode_convolution_per_token = 8 * qkv * 2;
        let prefill_convolution_per_token = 5 * qkv * 2;
        let publication = 6 * qkv * 2 + 4;

        assert_eq!(
            logical_bytes(1),
            weights + control_per_token + decode_convolution_per_token
        );
        assert_eq!(
            logical_bytes(MAX_BATCH),
            weights + MAX_BATCH * (control_per_token + decode_convolution_per_token)
        );
        assert_eq!(
            logical_bytes(32),
            weights + 32 * (control_per_token + prefill_convolution_per_token) + publication
        );
        assert_eq!(
            logical_bytes(MAX_ROWS),
            weights + MAX_ROWS * (control_per_token + prefill_convolution_per_token) + publication
        );
    }

    #[test]
    fn benchmark_route_inventory_and_owner_accounting_are_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        let (layout, regions) = layout().unwrap();
        let weights = regions.control_weights.byte_len()
            + regions.a_log.byte_len()
            + regions.dt_bias.byte_len()
            + regions.convolution_weights.byte_len();

        assert_eq!(weights, 1_065_152);
        assert_eq!(layout.byte_len(), 66_962_176);
        assert_eq!(regions.payload_bytes(), 66_961_632);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 544);
    }
}

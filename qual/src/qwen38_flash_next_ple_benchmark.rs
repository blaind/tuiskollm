//! Timings for the exact Qwen3.8-Flash-Next PLE graph routes.
//!
//! Whole-module, convolution, and injection graphs use the production stream,
//! arena, warm-cache regime, and every admitted width. An untimed preparation
//! graph restores convolution history before each measured replay.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, ExactDeviceCase, MemoryRecorder,
    OperationAccounting, finish_report, generator_baseline_sha256, measure_cases, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::target::{
    EXPECTED_COMPUTE_CAPABILITY, Qwen38FlashNextEngramOp, Qwen38FlashNextEngramSources,
    Qwen38FlashNextEngramWorkspace, Qwen38FlashNextHyperConnectionOp,
};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    GpuTimer, PinnedHostBuffer,
};
use tuisko_model::{Arch, Qwen38FlashNext};

const MAX_BATCH: usize = 8;
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const MAX_ROWS: usize = 1_024;
const SLOTS: usize = MAX_BATCH;
const ALIGNMENT: usize = 256;

const BRANCH: usize = Qwen38FlashNext::HIDDEN;
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
const EMBED: usize = Qwen38FlashNext::PLE_EMBED_DIM;
const CONV_TAPS: usize = Qwen38FlashNext::PLE_CONV_KERNEL;
const CONV_STATE: usize = Qwen38FlashNext::PLE_CONV_STATE_LEN;
const TABLE_SCALE_BITS: u16 = 0x3951;

const STREAM_PATTERN: [f32; 8] = [0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625];
const WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];
const PROJECTION_PATTERN: [f32; 8] = [2.0, -1.5, 1.0, -0.75, 0.5, -0.375, 0.25, -0.125];
const CODE_PATTERN: [u8; 8] = [0x38, 0xb8, 0x3c, 0xbc, 0x40, 0xc0, 0x34, 0xb4];

struct RouteGraphs {
    rows: usize,
    /// Untimed restore of the seeded convolution history.
    preparation: CudaGraph,
    engram: CudaGraph,
    convolution: CudaGraph,
    inject: CudaGraph,
}

#[derive(Clone, Copy)]
struct Regions {
    codes: ArenaRegion<u8>,
    hidden: ArenaRegion<u16>,
    key_proj: ArenaRegion<u16>,
    value_proj: ArenaRegion<u16>,
    norm_key: ArenaRegion<u16>,
    norm_query: ArenaRegion<u16>,
    norm_conv: ArenaRegion<u16>,
    conv_weight: ArenaRegion<u16>,
    state_rows: ArenaRegion<u32>,
    embedding: ArenaRegion<u16>,
    key: ArenaRegion<u16>,
    key_normed: ArenaRegion<u16>,
    query_normed: ArenaRegion<u16>,
    value: ArenaRegion<u16>,
    gated: ArenaRegion<u16>,
    gated_normed: ArenaRegion<u16>,
    delta: ArenaRegion<u16>,
    injected: ArenaRegion<u16>,
    conv_state: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.key_proj.byte_len()
            + self.value_proj.byte_len()
            + self.norm_key.byte_len()
            + self.norm_query.byte_len()
            + self.norm_conv.byte_len()
            + self.conv_weight.byte_len()
    }

    fn state_bytes(self) -> usize {
        self.conv_state.byte_len()
    }

    fn payload_bytes(self) -> usize {
        self.weight_bytes()
            + self.state_bytes()
            + self.codes.byte_len()
            + self.hidden.byte_len()
            + self.state_rows.byte_len()
            + self.embedding.byte_len()
            + self.key.byte_len()
            + self.key_normed.byte_len()
            + self.query_normed.byte_len()
            + self.value.byte_len()
            + self.gated.byte_len()
            + self.gated_normed.byte_len()
            + self.delta.byte_len()
            + self.injected.byte_len()
    }
}

struct Addresses {
    codes: *const u8,
    hidden: *const u16,
    sources: Qwen38FlashNextEngramSources,
    workspace: Qwen38FlashNextEngramWorkspace,
    state_rows: *const u32,
    conv_state: *mut u16,
    injected: *mut u16,
}

struct Session {
    routes: Vec<RouteGraphs>,
    _engram: Qwen38FlashNextEngramOp,
    _norm: Qwen38FlashNextHyperConnectionOp,
    arena: DeviceArena,
    regions: Regions,
    _state_seed: PinnedHostBuffer<u16>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
}

impl Session {
    fn new() -> Result<Self, DeviceBenchmarkError> {
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
        let codes_host = (0..MAX_ROWS * EMBED)
            .map(|index| CODE_PATTERN[(index + index / EMBED) & 7])
            .collect::<Vec<_>>();
        let stream_host = (0..MAX_ROWS * WIDTH)
            .map(|index| f32_to_bf16(STREAM_PATTERN[(index + index / WIDTH) & 7]))
            .collect::<Vec<_>>();
        let key_host = (0..WIDTH * EMBED)
            .map(|index| f32_to_bf16(PROJECTION_PATTERN[(index + index / EMBED) & 7]))
            .collect::<Vec<_>>();
        let value_host = (0..EMBED * EMBED)
            .map(|index| f32_to_bf16(PROJECTION_PATTERN[(index * 3 + index / EMBED) & 7] * 64.0))
            .collect::<Vec<_>>();
        let norm_host = (0..WIDTH)
            .map(|index| f32_to_bf16(WEIGHT_PATTERN[index & 7]))
            .collect::<Vec<_>>();
        let conv_host = (0..WIDTH * CONV_TAPS)
            .map(|index| f32_to_bf16(WEIGHT_PATTERN[(index * 3) & 7]))
            .collect::<Vec<_>>();
        let state_host = (0..SLOTS * WIDTH * CONV_STATE)
            .map(|index| f32_to_bf16(STREAM_PATTERN[(index * 5) & 7]))
            .collect::<Vec<_>>();
        let rows_host = (0..MAX_ROWS)
            .map(|row| (row % SLOTS) as u32)
            .collect::<Vec<_>>();
        arena.copy_from_host(&stream, regions.codes, &codes_host)?;
        arena.copy_from_host(&stream, regions.hidden, &stream_host)?;
        arena.copy_from_host(&stream, regions.key_proj, &key_host)?;
        arena.copy_from_host(&stream, regions.value_proj, &value_host)?;
        arena.copy_from_host(&stream, regions.norm_key, &norm_host)?;
        arena.copy_from_host(&stream, regions.norm_query, &norm_host)?;
        arena.copy_from_host(&stream, regions.norm_conv, &norm_host)?;
        arena.copy_from_host(&stream, regions.conv_weight, &conv_host)?;
        arena.copy_from_host(&stream, regions.conv_state, &state_host)?;
        arena.copy_from_host(&stream, regions.state_rows, &rows_host)?;
        let state_seed =
            PinnedHostBuffer::from_slice(&context, &state_host).map_err(GpuError::from)?;
        stream.synchronize().map_err(GpuError::from)?;

        let engram = Qwen38FlashNextEngramOp::new(&context)?;
        let norm = Qwen38FlashNextHyperConnectionOp::new(&context)?;
        let addresses = Addresses {
            codes: arena.address(regions.codes)?,
            hidden: arena.address(regions.hidden)?,
            sources: Qwen38FlashNextEngramSources {
                key_proj: arena.address(regions.key_proj)?,
                value_proj: arena.address(regions.value_proj)?,
                norm_key: arena.address(regions.norm_key)?,
                norm_query: arena.address(regions.norm_query)?,
                norm_conv: arena.address(regions.norm_conv)?,
                convolution: arena.address(regions.conv_weight)?,
                table_scale_bits: TABLE_SCALE_BITS,
            },
            workspace: Qwen38FlashNextEngramWorkspace {
                embedding: arena.address(regions.embedding)?,
                key: arena.address(regions.key)?,
                key_normed: arena.address(regions.key_normed)?,
                query_normed: arena.address(regions.query_normed)?,
                value: arena.address(regions.value)?,
                gated: arena.address(regions.gated)?,
                gated_normed: arena.address(regions.gated_normed)?,
                delta: arena.address(regions.delta)?,
            },
            state_rows: arena.address(regions.state_rows)?,
            conv_state: arena.address(regions.conv_state)?,
            injected: arena.address(regions.injected)?,
        };

        // The convolution and the injection read planes the whole module
        // publishes, so those planes are primed once before capture and every
        // measured replay then reads the same production values from the same
        // address.
        // SAFETY: every pointer names a complete, aligned arena region.
        unsafe {
            launch_engram(&engram, &norm, &stream, &addresses, MAX_ROWS)?;
        }
        stream.synchronize().map_err(GpuError::from)?;

        let mut routes = Vec::with_capacity(ROUTES.len());
        for rows in ROUTES {
            let preparation = CudaGraph::capture(&stream, || {
                // SAFETY: the pinned seed stays immutable and owned by Session
                // through every replay, and covers the whole slot plane.
                unsafe {
                    arena.copy_prefix_from_pinned_host_async(
                        &stream,
                        regions.conv_state,
                        &state_seed,
                        SLOTS * WIDTH * CONV_STATE,
                    )
                }
            })?;
            routes.push(RouteGraphs {
                rows,
                preparation,
                engram: CudaGraph::capture(&stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_engram(&engram, &norm, &stream, &addresses, rows) }
                })?,
                convolution: CudaGraph::capture(&stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_convolution(&engram, &stream, &addresses, rows) }
                })?,
                inject: CudaGraph::capture(&stream, || {
                    // SAFETY: every pointer names a complete, aligned arena region.
                    unsafe { launch_inject(&engram, &stream, &addresses, rows) }
                })?,
            });
        }
        Ok(Self {
            routes,
            _engram: engram,
            _norm: norm,
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
                for graph in [
                    &route.preparation,
                    &route.engram,
                    &route.preparation,
                    &route.convolution,
                    &route.preparation,
                    &route.inject,
                ] {
                    // SAFETY: this Session owns every route graph and everything it
                    // captured, dropping the graphs first.
                    unsafe { graph.launch(&self.stream) }?;
                }
            }
        }
        self.stream.synchronize().map_err(GpuError::from)
    }

    fn cases(&self) -> Vec<ExactDeviceCase<'_>> {
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
            for (name, graph, values) in [
                (
                    "qwen38_flash_next/engram/module",
                    &route.engram,
                    engram_values(route.rows),
                ),
                (
                    "qwen38_flash_next/engram/convolution",
                    &route.convolution,
                    convolution_values(route.rows),
                ),
                (
                    "qwen38_flash_next/engram/inject",
                    &route.inject,
                    inject_values(route.rows),
                ),
            ] {
                cases.push(
                    ExactDeviceCase::new(
                        name,
                        shape.clone(),
                        workload.clone(),
                        OperationAccounting::new(values, route.rows as u64, "token"),
                        graph,
                        None,
                    )
                    .with_preparation(&route.preparation),
                );
            }
        }

        cases
    }
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_engram(
    engram: &Qwen38FlashNextEngramOp,
    norm: &Qwen38FlashNextHyperConnectionOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged.
    unsafe {
        engram.launch_engram(
            norm,
            stream,
            rows,
            addresses.codes,
            addresses.hidden,
            addresses.sources,
            addresses.workspace,
            addresses.state_rows,
            addresses.conv_state,
            addresses.injected,
        )
    }
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_convolution(
    engram: &Qwen38FlashNextEngramOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged.
    unsafe {
        engram.launch_convolution(
            stream,
            rows,
            addresses.workspace.gated,
            addresses.workspace.gated_normed,
            addresses.sources.convolution,
            addresses.state_rows,
            addresses.conv_state,
            addresses.workspace.delta,
        )
    }
}

/// # Safety
///
/// Every address must name a complete, aligned, context-local arena region.
unsafe fn launch_inject(
    engram: &Qwen38FlashNextEngramOp,
    stream: &CudaStream,
    addresses: &Addresses,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: the caller's contract is this method's contract unchanged, and
    // the benchmark drives the disjoint form, so the output never aliases the
    // stream a measured replay must restore.
    unsafe {
        engram.launch_inject(
            stream,
            rows,
            addresses.hidden,
            addresses.workspace.delta,
            addresses.injected,
        )
    }
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let stream_values = MAX_ROWS * WIDTH;
    let slot_values = WIDTH * CONV_STATE;
    let codes = layout.reserve(MAX_ROWS * EMBED, ALIGNMENT)?;
    let hidden = layout.reserve(stream_values, ALIGNMENT)?;
    let key_proj = layout.reserve(WIDTH * EMBED, ALIGNMENT)?;
    let value_proj = layout.reserve(EMBED * EMBED, ALIGNMENT)?;
    let norm_key = layout.reserve(WIDTH, ALIGNMENT)?;
    let norm_query = layout.reserve(WIDTH, ALIGNMENT)?;
    let norm_conv = layout.reserve(WIDTH, ALIGNMENT)?;
    let conv_weight = layout.reserve(WIDTH * CONV_TAPS, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let embedding = layout.reserve(MAX_ROWS * EMBED, ALIGNMENT)?;
    let key = layout.reserve(stream_values, ALIGNMENT)?;
    let key_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let query_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let value = layout.reserve(MAX_ROWS * BRANCH, ALIGNMENT)?;
    let gated = layout.reserve(stream_values, ALIGNMENT)?;
    let gated_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let delta = layout.reserve(stream_values, ALIGNMENT)?;
    let injected = layout.reserve(stream_values, ALIGNMENT)?;
    let conv_state = layout.reserve(SLOTS * slot_values, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            codes,
            hidden,
            key_proj,
            value_proj,
            norm_key,
            norm_query,
            norm_conv,
            conv_weight,
            state_rows,
            embedding,
            key,
            key_normed,
            query_normed,
            value,
            gated,
            gated_normed,
            delta,
            injected,
            conv_state,
        },
    ))
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));

    (rounded >> 16) as u16
}

/// Every byte the complete PLE module reads or writes, counted per staged entry
/// so the accounting follows the route that actually runs.
fn engram_values(rows: usize) -> usize {
    // Dequantization: reads the code plane, writes the embedding.
    let dequant = rows * EMBED * size_of::<u8>() + rows * EMBED * size_of::<u16>();
    // Projection: reads the embedding and both weight planes, writes the key
    // and value planes.
    let project =
        (rows * EMBED + (WIDTH + EMBED) * EMBED + rows * (WIDTH + BRANCH)) * size_of::<u16>();
    // Three grouped norms: each reads one stream and one gamma row and writes
    // one stream.
    let norms = (3 * (2 * rows * WIDTH + WIDTH)) * size_of::<u16>();
    // Gate: reads both normalized streams and the value row, writes the gated
    // stream.
    let gate = (2 * rows * WIDTH + rows * BRANCH + rows * WIDTH) * size_of::<u16>();
    // Convolution and injection carry their own accounting.
    dequant + project + norms + gate + convolution_values(rows) + inject_values(rows)
}

/// Every byte the stateful convolution reads or writes.
///
/// Decode advances one nine-column history per token; a prefill tile reads one
/// history and republishes nine columns once.
fn convolution_values(rows: usize) -> usize {
    let taps = 2 * rows * WIDTH + WIDTH * CONV_TAPS;
    let state = if rows <= MAX_BATCH {
        2 * rows * WIDTH * CONV_STATE
    } else {
        rows.min(CONV_STATE) * WIDTH + 2 * WIDTH * CONV_STATE
    };

    (taps + rows * WIDTH + state) * size_of::<u16>()
}

/// Every byte the injection reads or writes.
fn inject_values(rows: usize) -> usize {
    3 * rows * WIDTH * size_of::<u16>()
}

/// Measures every admitted engram decode batch and prefill tile.
pub fn benchmark_qwen38_flash_next_ple(
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup_launches = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let session = Session::new()?;
    let mut timer = GpuTimer::new(session.stream.context())?;
    let weight_bytes = session.regions.weight_bytes();
    let state_bytes = session.regions.state_bytes();
    let padding_bytes = session.arena.byte_len() - session.regions.payload_bytes();
    memory.register_owned(
        "qwen38_flash_next/engram/weights",
        BenchmarkMemoryKind::Weights,
        weight_bytes,
        "two projections, three gamma rows, and the dilated convolution",
    )?;
    memory.register_owned(
        "qwen38_flash_next/engram/convolution_state",
        BenchmarkMemoryKind::Workspace,
        state_bytes,
        "slots=8,hc_width=10240,state_len=9",
    )?;
    memory.register_owned(
        "qwen38_flash_next/engram/address_stable_workspace",
        BenchmarkMemoryKind::Workspace,
        session.arena.byte_len() - weight_bytes - state_bytes - padding_bytes,
        "max_rows=1024,hc_width=10240,ple_embed_dim=2560",
    )?;
    memory.register_owned(
        "qwen38_flash_next/engram/alignment_padding",
        BenchmarkMemoryKind::Other,
        padding_bytes,
        "256-byte arena region alignment",
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
            suite: "bench-flashnext-ple",
            classification: "performance_sensitive_leaf",
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
    use super::{
        BRANCH, CONV_STATE, CONV_TAPS, EMBED, MAX_BATCH, MAX_ROWS, ROUTES, SLOTS, WIDTH,
        convolution_values, engram_values, inject_values, layout,
    };
    use tuisko_model::Qwen38FlashNext;

    #[test]
    fn qwen38_flash_next_ple_suite_benchmark_arena_accounting_exposes_every_byte() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(layout.byte_len(), 248_037_376);
        assert_eq!(
            regions.weight_bytes(),
            ((WIDTH + EMBED) * EMBED + 3 * WIDTH + WIDTH * CONV_TAPS) * size_of::<u16>()
        );
        assert_eq!(
            regions.state_bytes(),
            SLOTS * WIDTH * CONV_STATE * size_of::<u16>()
        );
    }

    /// Byte accounting must name every plane each staged entry touches, or a
    /// per-token throughput is measured against a traffic the route never had.
    #[test]
    fn qwen38_flash_next_ple_suite_benchmark_byte_accounting_covers_every_read_and_write_plane() {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(WIDTH, Qwen38FlashNext::HC_COUNT * BRANCH);
        assert_eq!(MAX_ROWS, 1_024);

        for rows in ROUTES {
            assert_eq!(inject_values(rows), 3 * rows * WIDTH * size_of::<u16>());
            // The convolution's state traffic is what separates decode from
            // prefill: decode reads and rewrites one history per token, a tile
            // reads one and republishes it once.
            if rows <= MAX_BATCH {
                assert_eq!(
                    convolution_values(rows),
                    (3 * rows * WIDTH + WIDTH * CONV_TAPS + 2 * rows * WIDTH * CONV_STATE)
                        * size_of::<u16>()
                );
            } else {
                assert_eq!(
                    convolution_values(rows),
                    (3 * rows * WIDTH
                        + WIDTH * CONV_TAPS
                        + CONV_STATE * WIDTH
                        + 2 * WIDTH * CONV_STATE)
                        * size_of::<u16>()
                );
            }
            // The module is every stage, so its accounting must strictly
            // exceed the two stages timed on their own.
            assert!(
                engram_values(rows) > convolution_values(rows) + inject_values(rows),
                "rows={rows}"
            );
        }
    }
}

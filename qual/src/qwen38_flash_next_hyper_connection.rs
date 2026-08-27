//! Exact Qwen3.8-Flash-Next hyper-connection (gated-residual) qualification.
//!
//! The independent oracle accumulates in FP64 and reproduces every stored BF16
//! intermediate and product. It checks each norm, contraction, four-way mean,
//! and elementwise write-back independently of the device implementation.

use crate::device_benchmark;
use crate::residual_norm::{bf16_to_f32, f32_to_bf16};
use crate::{DeviceBenchmarkError, target::Qwen38FlashNextHyperConnectionOp};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Exact decode batches and prefill tiles this family admits.
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const MAX_ROWS: usize = 1_024;
const ALIGNMENT: usize = 256;
const INACTIVE_SENTINEL: u16 = 0xa5a5;

const BRANCHES: usize = Qwen38FlashNext::HC_COUNT;
const BRANCH: usize = Qwen38FlashNext::HIDDEN;
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
const RANK: usize = Qwen38FlashNext::HC_LOWRANK;
const EPSILON: f32 = Qwen38FlashNext::RMS_NORM_EPSILON;

// Fixture tables. Every value is exactly representable in BF16 so the fixture
// itself contributes no rounding and all observed error belongs to the kernel.
const RESIDUAL_PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];
const BLOCK_PATTERN: [f32; 8] = [
    0.25, -0.125, 0.0625, -0.03125, -0.25, 0.125, -0.0625, 0.03125,
];
const NORM_WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];
// The contraction is 10,240 wide, so the down plane is scaled to keep
// `projection / 4` inside the informative part of both nonlinearities instead
// of saturating every sigmoid to one.
const DOWN_PATTERN: [f32; 16] = [
    0.001953125,
    -0.0009765625,
    0.00048828125,
    0.0,
    -0.001953125,
    0.0009765625,
    -0.00048828125,
    0.0009765625,
    0.00146484375,
    -0.00048828125,
    0.001953125,
    -0.00146484375,
    0.0,
    0.00048828125,
    -0.0009765625,
    0.00146484375,
];
// The expansion contracts over 320 ranks only, so its plane is two orders
// larger than the down plane for the same reason.
const UP_PATTERN: [f32; 16] = [
    0.015625,
    -0.0078125,
    0.00390625,
    0.0,
    -0.015625,
    0.0078125,
    -0.00390625,
    0.0078125,
    0.01171875,
    -0.00390625,
    0.015625,
    -0.01171875,
    0.0,
    0.00390625,
    -0.0078125,
    0.01171875,
];

/// Failure of exact SM120 hyper-connection qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextHyperConnectionQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively under the checked clock policy.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("hyper-connection qualification failed: {0}")]
    Mismatch(String),
}

type QualificationError = Qwen38FlashNextHyperConnectionQualificationError;

/// Complete observable accounting across every admitted route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextHyperConnectionQualification {
    /// Grouped `hc_norm` BF16 values compared with the FP64 oracle.
    pub normalized_values: usize,
    /// Low-rank BF16 values compared with the FP64 oracle.
    pub low_rank_values: usize,
    /// Mixed block-input BF16 values compared with the FP64 oracle.
    pub mixed_values: usize,
    /// Per-branch write-gate BF16 values compared with the FP64 oracle.
    pub write_gate_values: usize,
    /// Model-level mixer BF16 values compared with the FP64 oracle.
    pub mixer_values: usize,
    /// Injected raw-stream BF16 values compared bit-exactly.
    pub injected_values: usize,
    /// Injected values reproduced bit-exactly with the output aliasing the input.
    pub in_place_values: usize,
    /// Values proved identical across every route that shares a token.
    pub route_independent_values: usize,
    /// Mutable arena values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Inactive sentinel values proved untouched.
    pub inactive_values: usize,
    /// Read-only stream and weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Smallest per-branch write gate the fixture produced, in `(0, 2)`.
    pub minimum_write_gate: f32,
    /// Largest per-branch write gate the fixture produced, in `(0, 2)`.
    pub maximum_write_gate: f32,
    /// Largest absolute difference from any oracle.
    pub maximum_absolute_error: f32,
    /// Largest per-value tolerance applied by the oracle comparisons.
    pub maximum_tolerance: f32,
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
    fn payload_bytes(self) -> usize {
        self.immutable()
            .iter()
            .map(|region| region.1)
            .sum::<usize>()
            + self.mutable().iter().map(|region| region.1).sum::<usize>()
    }

    fn immutable(self) -> [(ArenaRegion<u16>, usize); 9] {
        [
            (self.residual, self.residual.byte_len()),
            (self.block_output, self.block_output.byte_len()),
            (self.norm_weight, self.norm_weight.byte_len()),
            (self.down, self.down.byte_len()),
            (self.up, self.up.byte_len()),
            (self.inject, self.inject.byte_len()),
            (self.mixer_norm_weight, self.mixer_norm_weight.byte_len()),
            (self.mixer_down, self.mixer_down.byte_len()),
            (self.mixer_up, self.mixer_up.byte_len()),
        ]
    }

    fn mutable(self) -> [(ArenaRegion<u16>, usize); 8] {
        [
            (self.normalized, self.normalized.byte_len()),
            (self.low_rank, self.low_rank.byte_len()),
            (self.mixed, self.mixed.byte_len()),
            (self.write_gate, self.write_gate.byte_len()),
            (self.mixer_normalized, self.mixer_normalized.byte_len()),
            (self.mixer_low_rank, self.mixer_low_rank.byte_len()),
            (self.mixer_mixed, self.mixer_mixed.byte_len()),
            (self.injected, self.injected.byte_len()),
        ]
    }
}

struct Fixture {
    residual: Vec<u16>,
    block_output: Vec<u16>,
    norm_weight: Vec<u16>,
    down: Vec<u16>,
    up: Vec<u16>,
    inject: Vec<u16>,
    mixer_norm_weight: Vec<u16>,
    mixer_down: Vec<u16>,
    mixer_up: Vec<u16>,
}

/// The whole-fixture oracle, computed once because every value a route
/// produces for a token depends only on that token's stream row.
struct Oracle {
    normalized: Vec<u16>,
    low_rank: Vec<u16>,
    mixed: Vec<u16>,
    write_gate: Vec<u16>,
    mixer_normalized: Vec<u16>,
    mixer_low_rank: Vec<u16>,
    mixer_mixed: Vec<u16>,
}

/// The eight mutable planes, read back in `Regions::mutable` order.
struct Observed {
    normalized: Vec<u16>,
    low_rank: Vec<u16>,
    mixed: Vec<u16>,
    write_gate: Vec<u16>,
    mixer_normalized: Vec<u16>,
    mixer_low_rank: Vec<u16>,
    mixer_mixed: Vec<u16>,
    injected: Vec<u16>,
}

impl Observed {
    fn planes(&self) -> [(&'static str, &Vec<u16>, usize); 8] {
        [
            ("normalized", &self.normalized, WIDTH),
            ("low_rank", &self.low_rank, RANK),
            ("mixed", &self.mixed, BRANCH),
            ("write_gate", &self.write_gate, BRANCHES),
            ("mixer_normalized", &self.mixer_normalized, WIDTH),
            ("mixer_low_rank", &self.mixer_low_rank, RANK),
            ("mixer_mixed", &self.mixer_mixed, BRANCH),
            ("injected", &self.injected, WIDTH),
        ]
    }
}

/// Qualifies every exact hyper-connection route and public seam.
pub fn qualify_qwen38_flash_next_hyper_connection()
-> Result<Qwen38FlashNextHyperConnectionQualification, QualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != crate::target::EXPECTED_COMPUTE_CAPABILITY {
        return Err(QualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected {}.{}",
            capability.0,
            capability.1,
            crate::target::EXPECTED_COMPUTE_CAPABILITY.0,
            crate::target::EXPECTED_COMPUTE_CAPABILITY.1,
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let fixture = fixture();
    load_immutable(&arena, &stream, regions, &fixture)?;
    let oracle = oracle(&fixture);
    let op = Qwen38FlashNextHyperConnectionOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let (minimum_write_gate, maximum_write_gate) = write_gate_extent(&oracle);
    let mut report = Qwen38FlashNextHyperConnectionQualification {
        normalized_values: 0,
        low_rank_values: 0,
        mixed_values: 0,
        write_gate_values: 0,
        mixer_values: 0,
        injected_values: 0,
        in_place_values: 0,
        route_independent_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        minimum_write_gate,
        maximum_write_gate,
        maximum_absolute_error: 0.0,
        maximum_tolerance: 0.0,
    };
    let mut first_token: Option<Observed> = None;

    for rows in ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, &fixture, &oracle, &eager, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
        verify_route_independence(rows, &first_token, &eager, &mut report)?;
        if first_token.is_none() {
            first_token = Some(truncate(&eager, 1));
        }

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replays and the synchronize that follows.
        unsafe { graph.launch(&stream) }?;
        let replay = observe(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;

        if addresses(&arena, regions)? != stable_addresses {
            return Err(QualificationError::Mismatch(format!(
                "device addresses changed while qualifying rows={rows}"
            )));
        }
    }

    verify_in_place_write_back(&op, &arena, &stream, regions, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
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

fn fixture() -> Fixture {
    // Branch `c` is scaled and rotated so the four branches carry genuinely
    // different sums of squares. The reference initializes the stream as four
    // identical embedding copies, but the engram layer adds its delta into the
    // widened stream before the first mix, so no admitted route may assume
    // branch symmetry.
    let residual = (0..MAX_ROWS * WIDTH)
        .map(|index| {
            let token = index / WIDTH;
            let column = index % WIDTH;
            let branch = column / BRANCH;
            let scale = 1.0 + branch as f32 * 0.25;
            f32_to_bf16(
                RESIDUAL_PATTERN[(column + 5 * branch + token) & 15]
                    * scale
                    * (1.0 - (token & 7) as f32 / 32.0),
            )
        })
        .collect();
    let block_output = (0..MAX_ROWS * BRANCH)
        .map(|index| {
            let token = index / BRANCH;
            f32_to_bf16(BLOCK_PATTERN[(index * 3 + token) & 7])
        })
        .collect();
    let norm_weight = (0..WIDTH)
        .map(|index| f32_to_bf16(NORM_WEIGHT_PATTERN[index & 7]))
        .collect();
    let mixer_norm_weight = (0..WIDTH)
        .map(|index| f32_to_bf16(NORM_WEIGHT_PATTERN[(index * 3 + 1) & 7]))
        .collect();
    let down = projection_plane(&DOWN_PATTERN, RANK, WIDTH, 7);
    let mixer_down = projection_plane(&DOWN_PATTERN, RANK, WIDTH, 11);
    let up = projection_plane(&UP_PATTERN, WIDTH, RANK, 5);
    let mixer_up = projection_plane(&UP_PATTERN, WIDTH, RANK, 9);
    let inject = projection_plane(&DOWN_PATTERN, BRANCHES, WIDTH, 3);

    Fixture {
        residual,
        block_output,
        norm_weight,
        down,
        up,
        inject,
        mixer_norm_weight,
        mixer_down,
        mixer_up,
    }
}

/// Builds one projection plane, rotating the pattern per row so different rows
/// land at different points of both nonlinearities.
fn projection_plane(pattern: &[f32; 16], rows: usize, columns: usize, stride: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|index| {
            let row = index / columns;
            let column = index % columns;
            f32_to_bf16(pattern[(column + stride * row) & 15] * (1.0 - (row % 5) as f32 / 8.0))
        })
        .collect()
}

fn load_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.residual, &fixture.residual)?;
    arena.copy_from_host(stream, regions.block_output, &fixture.block_output)?;
    arena.copy_from_host(stream, regions.norm_weight, &fixture.norm_weight)?;
    arena.copy_from_host(stream, regions.down, &fixture.down)?;
    arena.copy_from_host(stream, regions.up, &fixture.up)?;
    arena.copy_from_host(stream, regions.inject, &fixture.inject)?;
    arena.copy_from_host(
        stream,
        regions.mixer_norm_weight,
        &fixture.mixer_norm_weight,
    )?;
    arena.copy_from_host(stream, regions.mixer_down, &fixture.mixer_down)?;
    arena.copy_from_host(stream, regions.mixer_up, &fixture.mixer_up)
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    for (region, _) in regions.mutable() {
        arena.fill(stream, region, 0xa5)?;
    }

    Ok(())
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Vec<usize>> {
    let mut addresses = Vec::with_capacity(17);
    for (region, _) in regions.immutable() {
        addresses.push(arena.address(region)?.addr());
    }
    for (region, _) in regions.mutable() {
        addresses.push(arena.address(region)?.addr());
    }

    Ok(addresses)
}

fn launch(
    op: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every region is aligned, disjoint, context-local, and covers
    // `MAX_ROWS`, which bounds every admitted route.
    unsafe {
        op.launch_input_mix(
            stream,
            rows,
            arena.address(regions.residual)?,
            arena.address(regions.norm_weight)?,
            arena.address(regions.down)?,
            arena.address(regions.up)?,
            arena.address(regions.inject)?,
            arena.address(regions.normalized)?,
            arena.address(regions.low_rank)?,
            arena.address(regions.mixed)?,
            arena.address(regions.write_gate)?,
        )?;
        op.launch_final_mix(
            stream,
            rows,
            arena.address(regions.residual)?,
            arena.address(regions.mixer_norm_weight)?,
            arena.address(regions.mixer_down)?,
            arena.address(regions.mixer_up)?,
            arena.address(regions.mixer_normalized)?,
            arena.address(regions.mixer_low_rank)?,
            arena.address(regions.mixer_mixed)?,
        )?;
        op.launch_write_back(
            stream,
            rows,
            arena.address(regions.residual)?,
            arena.address(regions.block_output)?,
            arena.address(regions.write_gate)?,
            arena.address(regions.injected)?,
        )
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        normalized: arena.copy_to_host(stream, regions.normalized)?,
        low_rank: arena.copy_to_host(stream, regions.low_rank)?,
        mixed: arena.copy_to_host(stream, regions.mixed)?,
        write_gate: arena.copy_to_host(stream, regions.write_gate)?,
        mixer_normalized: arena.copy_to_host(stream, regions.mixer_normalized)?,
        mixer_low_rank: arena.copy_to_host(stream, regions.mixer_low_rank)?,
        mixer_mixed: arena.copy_to_host(stream, regions.mixer_mixed)?,
        injected: arena.copy_to_host(stream, regions.injected)?,
    })
}

fn truncate(observed: &Observed, rows: usize) -> Observed {
    Observed {
        normalized: observed.normalized[..rows * WIDTH].to_vec(),
        low_rank: observed.low_rank[..rows * RANK].to_vec(),
        mixed: observed.mixed[..rows * BRANCH].to_vec(),
        write_gate: observed.write_gate[..rows * BRANCHES].to_vec(),
        mixer_normalized: observed.mixer_normalized[..rows * WIDTH].to_vec(),
        mixer_low_rank: observed.mixer_low_rank[..rows * RANK].to_vec(),
        mixer_mixed: observed.mixer_mixed[..rows * BRANCH].to_vec(),
        injected: observed.injected[..rows * WIDTH].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Independent oracle
// ---------------------------------------------------------------------------

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// Rounds through the BF16 grid the reference's intermediate tensors carry.
fn through_bf16(value: f64) -> f64 {
    f64::from(bf16_to_f32(f32_to_bf16(value as f32)))
}

fn widen(values: &[u16]) -> Vec<f64> {
    values
        .iter()
        .map(|&bits| f64::from(bf16_to_f32(bits)))
        .collect()
}

/// `hc_norm`: four independent 2,560-wide RMSNorms, flattened, then one
/// 10,240-wide `(1 + w)`. The checkpoint ships the gamma unfolded.
fn grouped_rms_norm_oracle(row: &[f64], weight: &[f64]) -> Vec<u16> {
    let mut normalized = vec![0u16; WIDTH];
    for branch in 0..BRANCHES {
        let begin = branch * BRANCH;
        let squares = row[begin..begin + BRANCH]
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let inverse = 1.0 / (squares / BRANCH as f64 + f64::from(EPSILON)).sqrt();
        for column in 0..BRANCH {
            let index = begin + column;
            normalized[index] = f32_to_bf16((row[index] * inverse * (1.0 + weight[index])) as f32);
        }
    }

    normalized
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
}

/// `silu(down(hn) / 4)` for every rank.
fn low_rank_oracle(normalized: &[f64], down: &[f64]) -> Vec<u16> {
    (0..RANK)
        .map(|rank| {
            let scaled = through_bf16(dot(&down[rank * WIDTH..(rank + 1) * WIDTH], normalized))
                / BRANCHES as f64;
            f32_to_bf16((scaled * logistic(scaled)) as f32)
        })
        .collect()
}

/// `2 * sigmoid(inject(hn) / 4)` for every branch.
fn write_gate_oracle(normalized: &[f64], inject: &[f64]) -> Vec<u16> {
    (0..BRANCHES)
        .map(|branch| {
            let scaled = through_bf16(dot(
                &inject[branch * WIDTH..(branch + 1) * WIDTH],
                normalized,
            )) / BRANCHES as f64;
            let gate = through_bf16(logistic(scaled));
            f32_to_bf16((2.0 * gate) as f32)
        })
        .collect()
}

/// `mean_c(sigmoid(up(t)) * hn)` for every mixed column.
fn mixed_oracle(normalized: &[f64], up: &[f64], low_rank: &[u16]) -> Vec<u16> {
    let low_rank = widen(low_rank);
    (0..BRANCH)
        .map(|column| {
            let mut total = 0.0f64;
            for branch in 0..BRANCHES {
                let row = branch * BRANCH + column;
                let read = through_bf16(logistic(through_bf16(dot(
                    &up[row * RANK..(row + 1) * RANK],
                    &low_rank,
                ))));
                total += through_bf16(read * normalized[row]);
            }

            f32_to_bf16((total / BRANCHES as f64) as f32)
        })
        .collect()
}

/// `h + broadcast_c(block_output) * w_inj`, into the raw stream.
///
/// Every operand is BF16 and there is no reduction, so this arm is exact in
/// FP32 and is compared bit-exactly rather than with a tolerance.
fn write_back_oracle(residual: &[u16], block_output: &[u16], write_gate: &[u16]) -> Vec<u16> {
    let mut injected = vec![0u16; WIDTH];
    for branch in 0..BRANCHES {
        let gate = bf16_to_f32(write_gate[branch]);
        for column in 0..BRANCH {
            let index = branch * BRANCH + column;
            let injection = f32_to_bf16(bf16_to_f32(block_output[column]) * gate);
            injected[index] = f32_to_bf16(bf16_to_f32(residual[index]) + bf16_to_f32(injection));
        }
    }

    injected
}

fn oracle(fixture: &Fixture) -> Oracle {
    let norm_weight = widen(&fixture.norm_weight);
    let mixer_norm_weight = widen(&fixture.mixer_norm_weight);
    let down = widen(&fixture.down);
    let up = widen(&fixture.up);
    let inject = widen(&fixture.inject);
    let mixer_down = widen(&fixture.mixer_down);
    let mixer_up = widen(&fixture.mixer_up);
    let mut oracle = Oracle {
        normalized: Vec::with_capacity(MAX_ROWS * WIDTH),
        low_rank: Vec::with_capacity(MAX_ROWS * RANK),
        mixed: Vec::with_capacity(MAX_ROWS * BRANCH),
        write_gate: Vec::with_capacity(MAX_ROWS * BRANCHES),
        mixer_normalized: Vec::with_capacity(MAX_ROWS * WIDTH),
        mixer_low_rank: Vec::with_capacity(MAX_ROWS * RANK),
        mixer_mixed: Vec::with_capacity(MAX_ROWS * BRANCH),
    };

    for token in 0..MAX_ROWS {
        let row = widen(&fixture.residual[token * WIDTH..(token + 1) * WIDTH]);

        let normalized = grouped_rms_norm_oracle(&row, &norm_weight);
        let widened = widen(&normalized);
        let low_rank = low_rank_oracle(&widened, &down);
        oracle.mixed.extend(mixed_oracle(&widened, &up, &low_rank));
        oracle
            .write_gate
            .extend(write_gate_oracle(&widened, &inject));
        oracle.low_rank.extend(low_rank);
        oracle.normalized.extend(normalized);

        let normalized = grouped_rms_norm_oracle(&row, &mixer_norm_weight);
        let widened = widen(&normalized);
        let low_rank = low_rank_oracle(&widened, &mixer_down);
        oracle
            .mixer_mixed
            .extend(mixed_oracle(&widened, &mixer_up, &low_rank));
        oracle.mixer_low_rank.extend(low_rank);
        oracle.mixer_normalized.extend(normalized);
    }

    oracle
}

fn write_gate_extent(oracle: &Oracle) -> (f32, f32) {
    oracle
        .write_gate
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        })
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify_eager(
    rows: usize,
    fixture: &Fixture,
    oracle: &Oracle,
    observed: &Observed,
    report: &mut Qwen38FlashNextHyperConnectionQualification,
) -> Result<(), QualificationError> {
    for (name, plane, expected, width) in [
        ("hc_norm", &observed.normalized, &oracle.normalized, WIDTH),
        ("low rank", &observed.low_rank, &oracle.low_rank, RANK),
        ("mixed", &observed.mixed, &oracle.mixed, BRANCH),
        (
            "write gate",
            &observed.write_gate,
            &oracle.write_gate,
            BRANCHES,
        ),
        (
            "mixer hc_norm",
            &observed.mixer_normalized,
            &oracle.mixer_normalized,
            WIDTH,
        ),
        (
            "mixer low rank",
            &observed.mixer_low_rank,
            &oracle.mixer_low_rank,
            RANK,
        ),
        (
            "mixer mixed",
            &observed.mixer_mixed,
            &oracle.mixer_mixed,
            BRANCH,
        ),
    ] {
        for index in 0..rows * width {
            check_close(
                name,
                rows,
                index / width,
                index % width,
                plane[index],
                expected[index],
                &mut report.maximum_absolute_error,
                &mut report.maximum_tolerance,
            )?;
        }
    }

    // The injection arm consumes the gates the device produced, which the
    // block above has already checked against the oracle. Given those gates
    // the algebra is exact, so this arm is a bitwise contract.
    for token in 0..rows {
        let expected = write_back_oracle(
            &fixture.residual[token * WIDTH..(token + 1) * WIDTH],
            &fixture.block_output[token * BRANCH..(token + 1) * BRANCH],
            &observed.write_gate[token * BRANCHES..(token + 1) * BRANCHES],
        );
        for column in 0..WIDTH {
            let index = token * WIDTH + column;
            if observed.injected[index] != expected[column] {
                return Err(QualificationError::Mismatch(format!(
                    "write-back at rows={rows}, row={token}, column={column}: device={:#06x}, oracle={:#06x}",
                    observed.injected[index], expected[column]
                )));
            }
        }
    }

    verify_inactive(rows, observed)?;
    report.normalized_values += rows * WIDTH;
    report.low_rank_values += rows * RANK;
    report.mixed_values += rows * BRANCH;
    report.write_gate_values += rows * BRANCHES;
    report.mixer_values += rows * (WIDTH + RANK + BRANCH);
    report.injected_values += rows * WIDTH;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn verify_inactive(rows: usize, observed: &Observed) -> Result<(), QualificationError> {
    for (name, plane, width) in observed.planes() {
        let active = rows * width;
        if let Some(relative) = plane[active..]
            .iter()
            .position(|&value| value != INACTIVE_SENTINEL)
        {
            let index = active + relative;
            return Err(QualificationError::Mismatch(format!(
                "rows={rows} {name} route modified inactive value {index}: device={:#06x}",
                plane[index]
            )));
        }
    }

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (3 * WIDTH + 2 * RANK + 2 * BRANCH + BRANCHES)
}

fn verify_route_independence(
    rows: usize,
    reference: &Option<Observed>,
    observed: &Observed,
    report: &mut Qwen38FlashNextHyperConnectionQualification,
) -> Result<(), QualificationError> {
    let Some(reference) = reference else {
        return Ok(());
    };

    // Every entry keeps one output's whole reduction inside one warp, so a
    // token's bits must not depend on which route produced them.
    for ((name, plane, width), (_, expected, _)) in
        observed.planes().into_iter().zip(reference.planes())
    {
        if let Some(index) = plane[..width]
            .iter()
            .zip(&expected[..width])
            .position(|(actual, expected)| actual != expected)
        {
            return Err(QualificationError::Mismatch(format!(
                "rows={rows} {name} differs from the B=1 route at column {index}: rows={:#06x}, B=1={:#06x}",
                plane[index], expected[index]
            )));
        }
        report.route_independent_values += width;
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextHyperConnectionQualification,
) -> Result<(), QualificationError> {
    for (name, region, expected) in [
        ("residual", regions.residual, &fixture.residual),
        ("block output", regions.block_output, &fixture.block_output),
        ("hc_norm weight", regions.norm_weight, &fixture.norm_weight),
        ("down", regions.down, &fixture.down),
        ("up", regions.up, &fixture.up),
        ("inject", regions.inject, &fixture.inject),
        (
            "mixer hc_norm weight",
            regions.mixer_norm_weight,
            &fixture.mixer_norm_weight,
        ),
        ("mixer down", regions.mixer_down, &fixture.mixer_down),
        ("mixer up", regions.mixer_up, &fixture.mixer_up),
    ] {
        let actual = arena.copy_to_host(stream, region)?;
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(QualificationError::Mismatch(format!(
                "read-only {name} changed at index {index}"
            )));
        }
        report.immutable_input_values += actual.len();
    }

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Observed,
    replay: &Observed,
    report: &mut Qwen38FlashNextHyperConnectionQualification,
) -> Result<(), QualificationError> {
    for ((name, actual, _), (_, expected, _)) in replay.planes().into_iter().zip(eager.planes()) {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(QualificationError::Mismatch(format!(
                "rows={rows} {name} graph replay differs from eager at value {index}: replay={:#06x}, eager={:#06x}",
                actual[index], expected[index]
            )));
        }
        report.graph_replay_values += actual.len();
    }

    verify_inactive(rows, replay)?;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

/// Proves the documented aliasing contract: the write-back may publish into the
/// stream it reads.
fn verify_in_place_write_back(
    op: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    report: &mut Qwen38FlashNextHyperConnectionQualification,
) -> Result<(), QualificationError> {
    reset_outputs(arena, stream, regions)?;
    launch(op, arena, stream, regions, MAX_ROWS)?;
    let expected = arena.copy_to_host(stream, regions.injected)?;
    let raw = arena.copy_to_host(stream, regions.residual)?;
    arena.copy_from_host(stream, regions.injected, &raw)?;
    // SAFETY: `injected` is the documented in-place form: the output aliases
    // the input exactly, and every thread writes only the pair it read.
    unsafe {
        op.launch_write_back(
            stream,
            MAX_ROWS,
            arena.address(regions.injected)?,
            arena.address(regions.block_output)?,
            arena.address(regions.write_gate)?,
            arena.address(regions.injected)?,
        )?;
    }
    let actual = arena.copy_to_host(stream, regions.injected)?;

    if let Some(index) = actual
        .iter()
        .zip(&expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(QualificationError::Mismatch(format!(
            "in-place write-back differs from the disjoint form at value {index}: in-place={:#06x}, disjoint={:#06x}",
            actual[index], expected[index]
        )));
    }
    report.in_place_values += actual.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), QualificationError> {
    let mut graphs = Vec::with_capacity(ROUTES.len());
    for rows in ROUTES {
        reset_outputs(arena, stream, regions)?;
        graphs.push(CudaGraph::capture(stream, || {
            launch(op, arena, stream, regions, rows)
        })?);
    }
    for graph in &graphs {
        // SAFETY: the qualification owner retains every allocation captured by these graphs.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification owner retains every allocation captured by these graphs.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(QualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_close(
    operation: &str,
    rows: usize,
    token: usize,
    column: usize,
    actual_bits: u16,
    oracle_bits: u16,
    maximum_absolute_error: &mut f32,
    maximum_tolerance: &mut f32,
) -> Result<(), QualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_absolute_error = maximum_absolute_error.max(error);
    let tolerance = 0.015625f32.max(oracle.abs() * 0.005);
    *maximum_tolerance = maximum_tolerance.max(tolerance);
    if error > tolerance {
        return Err(QualificationError::Mismatch(format!(
            "{operation} at rows={rows}, row={token}, column={column}: device={actual}, oracle={oracle}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BRANCH, BRANCHES, MAX_ROWS, RANK, ROUTES, WIDTH, bf16_to_f32, fixture, layout,
        qualify_qwen38_flash_next_hyper_connection,
    };

    /// The four branches must carry genuinely different sums of squares.
    ///
    /// The reference widens the stream as four identical embedding copies, but
    /// the engram injection at layer one adds its delta into the same widened
    /// stream before the first mix, so a route that assumed branch symmetry
    /// would be wrong on the real model. This fixture would catch it.
    #[test]
    fn qwen38_flash_next_hyper_connection_suite_fixture_breaks_branch_symmetry() {
        let fixture = fixture();
        let mut roots = Vec::with_capacity(BRANCHES);
        for branch in 0..BRANCHES {
            let begin = branch * BRANCH;
            let squares = fixture.residual[begin..begin + BRANCH]
                .iter()
                .map(|&bits| {
                    let value = f64::from(bf16_to_f32(bits));
                    value * value
                })
                .sum::<f64>();
            roots.push((squares / BRANCH as f64).sqrt());
        }

        for (index, root) in roots.iter().enumerate() {
            assert!(*root > 0.0, "branch {index} is entirely zero");
            for other in &roots[index + 1..] {
                assert!(
                    (root - other).abs() > 0.05,
                    "branch root-mean-squares are not distinct: {roots:?}"
                );
            }
        }
    }

    #[test]
    fn qwen38_flash_next_hyper_connection_suite_route_and_arena_inventory_is_exact() {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(WIDTH, BRANCHES * BRANCH);
        assert_eq!(RANK, 320);

        let (arena, regions) = layout().unwrap();
        assert_eq!(arena.byte_len(), regions.payload_bytes());
        assert_eq!(arena.byte_len(), 127_270_912);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_hyper_connection_suite_exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::Qwen38FlashNextHyperConnectionQualificationError> {
        let report = qualify_qwen38_flash_next_hyper_connection()?;
        let rows = ROUTES.into_iter().sum::<usize>();
        let immutable =
            MAX_ROWS * WIDTH + MAX_ROWS * BRANCH + 2 * WIDTH + 4 * RANK * WIDTH + BRANCHES * WIDTH;

        assert_eq!(report.normalized_values, rows * WIDTH);
        assert_eq!(report.low_rank_values, rows * RANK);
        assert_eq!(report.mixed_values, rows * BRANCH);
        assert_eq!(report.write_gate_values, rows * BRANCHES);
        assert_eq!(report.mixer_values, rows * (WIDTH + RANK + BRANCH));
        assert_eq!(report.injected_values, rows * WIDTH);
        assert_eq!(report.in_place_values, MAX_ROWS * WIDTH);
        assert_eq!(
            report.route_independent_values,
            (ROUTES.len() - 1) * (3 * WIDTH + 2 * RANK + 2 * BRANCH + BRANCHES)
        );
        assert_eq!(
            report.graph_replay_values,
            ROUTES.len() * MAX_ROWS * (3 * WIDTH + 2 * RANK + 2 * BRANCH + BRANCHES)
        );
        assert_eq!(
            report.inactive_values,
            2 * ROUTES
                .into_iter()
                .map(|rows| (MAX_ROWS - rows) * (3 * WIDTH + 2 * RANK + 2 * BRANCH + BRANCHES))
                .sum::<usize>()
        );
        assert_eq!(report.immutable_input_values, 2 * ROUTES.len() * immutable);

        // A fixture whose gates all saturated would qualify the plumbing and
        // none of the algebra.
        assert!(report.minimum_write_gate > 0.0);
        assert!(report.maximum_write_gate < 2.0);
        assert!(report.maximum_write_gate - report.minimum_write_gate > 0.125);
        assert!(report.maximum_absolute_error <= report.maximum_tolerance);

        let (arena, regions) = layout()?;
        assert_eq!(report.padding_bytes, 0);
        assert_eq!(report.arena_bytes, 127_270_912);
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(regions.payload_bytes(), arena.byte_len());

        Ok(())
    }
}

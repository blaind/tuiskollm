//! Independent qualification for the Qwen3.8-Flash-Next PLE module.
//!
//! The oracle decodes E4M3 from bits and uses f64 for transcendental math. It
//! reproduces every represented BF16 boundary and the device's declared FP32
//! accumulation order: warp-strided projections, grouped norms, a branch gate
//! over rounded products, and convolution taps `t-9, t-6, t-3, t`.

use crate::device_benchmark;
use crate::residual_norm::{bf16_to_f32, f32_to_bf16};
use crate::{
    DeviceBenchmarkError,
    target::{
        Qwen38FlashNextEngramOp, Qwen38FlashNextEngramSources, Qwen38FlashNextEngramWorkspace,
        Qwen38FlashNextHyperConnectionOp, Qwen38FlashNextPleStateSnapshotOp,
    },
};
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// Exact decode batches and prefill tiles this family admits.
const ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const MAX_BATCH: usize = 8;
const MAX_ROWS: usize = 1_024;
/// Persistent convolution slots, one per compact decode row.
const SLOTS: usize = MAX_BATCH;
const ALIGNMENT: usize = 256;
const INACTIVE_SENTINEL: u16 = 0xa5a5;

const BRANCHES: usize = Qwen38FlashNext::HC_COUNT;
const BRANCH: usize = Qwen38FlashNext::HIDDEN;
const WIDTH: usize = Qwen38FlashNext::HC_WIDTH;
const EMBED: usize = Qwen38FlashNext::PLE_EMBED_DIM;
const CONV_TAPS: usize = Qwen38FlashNext::PLE_CONV_KERNEL;
const CONV_DILATION: usize = Qwen38FlashNext::PLE_CONV_DILATION;
const CONV_STATE: usize = Qwen38FlashNext::PLE_CONV_STATE_LEN;
const EPSILON: f32 = Qwen38FlashNext::RMS_NORM_EPSILON;
const GATE_FLOOR: f32 = Qwen38FlashNext::PLE_GATE_FLOOR;
/// The checkpoint's exact BF16 source word for the engram table multiplier.
const TABLE_SCALE_BITS: u16 = 0x3951;

// Fixture tables. Every BF16 value is exactly representable so the fixture
// itself contributes no rounding and all observed error belongs to the kernel.
const HIDDEN_PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];
const NORM_WEIGHT_PATTERN: [f32; 8] = [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25];
/// E4M3 codes spanning both signs, three exponents, and every fraction bit.
const CODE_PATTERN: [u8; 16] = [
    0x38, 0xb8, 0x3c, 0xbc, 0x40, 0xc0, 0x34, 0xb4, 0x39, 0xb9, 0x3d, 0xbd, 0x41, 0xc1, 0x35, 0xb5,
];
// The key contraction is 2,560 wide over an embedding the table scale keeps
// near 2e-4, so the key plane is scaled to leave the grouped norm something to
// normalize rather than a row of denormals.
const KEY_PATTERN: [f32; 16] = [
    2.0, -1.5, 1.0, -0.75, 0.5, -0.375, 0.25, -0.125, -2.0, 1.5, -1.0, 0.75, -0.5, 0.375, -0.25,
    0.125,
];
// The value plane is two orders larger so the gated tensor and the convolution
// output are the same order of magnitude: the reference adds them, and a
// fixture where one term vanished would qualify a route that dropped it.
const VALUE_PATTERN: [f32; 16] = [
    192.0, -160.0, 128.0, -96.0, 64.0, -48.0, 32.0, -24.0, -192.0, 160.0, -128.0, 96.0, -64.0,
    48.0, -32.0, 24.0,
];
const CONV_PATTERN: [f32; 8] = [0.5, -0.375, 0.25, -0.125, -0.5, 0.375, -0.25, 0.125];
const CONV_STATE_PATTERN: [f32; 8] = [0.75, -0.625, 0.5, -0.375, 0.25, -0.125, -0.75, 0.625];
/// Constant key magnitude the signed-root probe multiplies against a unit query.
///
/// `5/64` is exact in BF16 and `2560 * 5/64 = 200` is exact in FP32, so the
/// probe's branch dot products are the same bits on the device and in the
/// oracle no matter how the reduction is ordered.
const PROBE_MAGNITUDE: f32 = 0.078_125;

/// Failure of exact SM120 engram qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextPleQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively under the checked clock policy.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent mathematical contract.
    #[error("engram qualification failed: {0}")]
    Mismatch(String),
}

type QualificationError = Qwen38FlashNextPleQualificationError;

/// Complete observable accounting across every admitted route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextPleQualification {
    /// Dequantized embedding BF16 values compared bit-exactly.
    pub embedding_values: usize,
    /// Projected key and value BF16 values compared with the FP64 oracle.
    pub projected_values: usize,
    /// Normalized key and query BF16 values compared with the FP64 oracle.
    pub normalized_values: usize,
    /// Gated and flattened BF16 values compared with the FP64 oracle.
    pub gated_values: usize,
    /// Convolution delta BF16 values compared with the FP64 oracle.
    pub delta_values: usize,
    /// Injected stream BF16 values compared bit-exactly.
    pub injected_values: usize,
    /// Published convolution history BF16 values compared bit-exactly.
    pub conv_state_values: usize,
    /// Convolution slots proved untouched by a narrower route.
    pub untouched_slot_values: usize,
    /// Injected values reproduced bit-exactly with the output aliasing the input.
    pub in_place_values: usize,
    /// Values proved identical across every route that shares a token.
    pub route_independent_values: usize,
    /// Mutable arena values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Inactive sentinel values proved untouched.
    pub inactive_values: usize,
    /// Read-only source values proved unchanged.
    pub immutable_input_values: usize,
    /// Slot values captured and put back bit-exactly by the snapshot pair.
    pub snapshot_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact alignment padding bytes in that arena.
    pub padding_bytes: usize,
    /// Smallest per-branch gate activation the fixture produced, in `(0, 1)`.
    pub minimum_gate_activation: f32,
    /// Largest per-branch gate activation the fixture produced, in `(0, 1)`.
    pub maximum_gate_activation: f32,
    /// Gate activation the signed-root probe produced on its zero branch.
    pub zero_branch_activation: f32,
    /// Gate activations the probe produced on its positive and negative branches.
    pub signed_branch_activations: (f32, f32),
    /// Largest absolute difference from any oracle.
    pub maximum_absolute_error: f32,
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
    conv_state_seed: ArenaRegion<u16>,
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
    snapshot: ArenaRegion<u16>,
}

impl Regions {
    fn payload_bytes(self) -> usize {
        self.immutable_bytes() + self.state_bytes() + self.token_planes_bytes()
    }

    fn immutable_bytes(self) -> usize {
        self.codes.byte_len()
            + self.hidden.byte_len()
            + self.key_proj.byte_len()
            + self.value_proj.byte_len()
            + self.norm_key.byte_len()
            + self.norm_query.byte_len()
            + self.norm_conv.byte_len()
            + self.conv_weight.byte_len()
            + self.conv_state_seed.byte_len()
            + self.state_rows.byte_len()
    }

    fn state_bytes(self) -> usize {
        self.conv_state.byte_len() + self.snapshot.byte_len()
    }

    /// The nine per-token planes the inactive-tail and replay checks walk.
    fn token_planes(self) -> [(ArenaRegion<u16>, usize); 9] {
        [
            (self.embedding, EMBED),
            (self.key, WIDTH),
            (self.key_normed, WIDTH),
            (self.query_normed, WIDTH),
            (self.value, BRANCH),
            (self.gated, WIDTH),
            (self.gated_normed, WIDTH),
            (self.delta, WIDTH),
            (self.injected, WIDTH),
        ]
    }

    fn token_planes_bytes(self) -> usize {
        self.token_planes()
            .iter()
            .map(|(region, _)| region.byte_len())
            .sum()
    }
}

struct Fixture {
    codes: Vec<u8>,
    hidden: Vec<u16>,
    key_proj: Vec<u16>,
    value_proj: Vec<u16>,
    norm_key: Vec<u16>,
    norm_query: Vec<u16>,
    norm_conv: Vec<u16>,
    conv_weight: Vec<u16>,
    conv_state_seed: Vec<u16>,
    state_rows: Vec<u32>,
}

/// The token-local part of the oracle, computed once because every value up to
/// the convolution depends only on that token's own row.
struct Oracle {
    embedding: Vec<u16>,
    key: Vec<u16>,
    key_normed: Vec<u16>,
    query_normed: Vec<u16>,
    value: Vec<u16>,
    gated: Vec<u16>,
    gated_normed: Vec<u16>,
    /// Every per-branch gate activation, `rows * BRANCHES`.
    activation: Vec<f32>,
}

/// The nine per-token planes plus the convolution state, read back together.
struct Observed {
    embedding: Vec<u16>,
    key: Vec<u16>,
    key_normed: Vec<u16>,
    query_normed: Vec<u16>,
    value: Vec<u16>,
    gated: Vec<u16>,
    gated_normed: Vec<u16>,
    delta: Vec<u16>,
    injected: Vec<u16>,
    conv_state: Vec<u16>,
}

impl Observed {
    fn planes(&self) -> [(&'static str, &Vec<u16>, usize); 9] {
        [
            ("embedding", &self.embedding, EMBED),
            ("key", &self.key, WIDTH),
            ("key_normed", &self.key_normed, WIDTH),
            ("query_normed", &self.query_normed, WIDTH),
            ("value", &self.value, BRANCH),
            ("gated", &self.gated, WIDTH),
            ("gated_normed", &self.gated_normed, WIDTH),
            ("delta", &self.delta, WIDTH),
            ("injected", &self.injected, WIDTH),
        ]
    }
}

/// Qualifies every exact engram route and public seam.
pub fn qualify_qwen38_flash_next_ple() -> Result<Qwen38FlashNextPleQualification, QualificationError>
{
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
    let engram = Qwen38FlashNextEngramOp::new(&context)?;
    let norm = Qwen38FlashNextHyperConnectionOp::new(&context)?;
    let snapshot = Qwen38FlashNextPleStateSnapshotOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let (minimum_gate_activation, maximum_gate_activation) = activation_extent(&oracle);
    let mut report = Qwen38FlashNextPleQualification {
        embedding_values: 0,
        projected_values: 0,
        normalized_values: 0,
        gated_values: 0,
        delta_values: 0,
        injected_values: 0,
        conv_state_values: 0,
        untouched_slot_values: 0,
        in_place_values: 0,
        route_independent_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        snapshot_values: 0,
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        minimum_gate_activation,
        maximum_gate_activation,
        zero_branch_activation: 0.0,
        signed_branch_activations: (0.0, 0.0),
        maximum_absolute_error: 0.0,
    };
    let mut first_token: Option<Observed> = None;

    for rows in ROUTES {
        reset_outputs(&arena, &stream, regions, &fixture)?;
        launch(&engram, &norm, &arena, &stream, regions, rows)?;
        let eager = observe(&arena, &stream, regions)?;
        verify_eager(rows, &fixture, &oracle, &eager, &mut report)?;
        verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
        verify_route_independence(rows, &first_token, &eager, &mut report)?;
        if first_token.is_none() {
            first_token = Some(truncate(&eager, 1));
        }

        reset_outputs(&arena, &stream, regions, &fixture)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&engram, &norm, &arena, &stream, regions, rows)
        })?;
        // The convolution is stateful, so a second replay would advance the
        // history a second time. Replay agreement is a one-launch contract
        // against the same reset state, which is what eager produced.
        // SAFETY: every allocation this graph captured is owned by this scope or
        // its caller and outlives the replay and the synchronize that follows.
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

    verify_signed_root_edge(&engram, &arena, &stream, regions, &mut report)?;
    verify_in_place_injection(
        &engram,
        &norm,
        &arena,
        &stream,
        regions,
        &fixture,
        &mut report,
    )?;
    verify_snapshot_round_trip(&snapshot, &arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &engram, &norm, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let stream_values = MAX_ROWS * WIDTH;
    let embed_values = MAX_ROWS * EMBED;
    let slot_values = WIDTH * CONV_STATE;
    let codes = layout.reserve(MAX_ROWS * EMBED, ALIGNMENT)?;
    let hidden = layout.reserve(stream_values, ALIGNMENT)?;
    let key_proj = layout.reserve(WIDTH * EMBED, ALIGNMENT)?;
    let value_proj = layout.reserve(EMBED * EMBED, ALIGNMENT)?;
    let norm_key = layout.reserve(WIDTH, ALIGNMENT)?;
    let norm_query = layout.reserve(WIDTH, ALIGNMENT)?;
    let norm_conv = layout.reserve(WIDTH, ALIGNMENT)?;
    let conv_weight = layout.reserve(WIDTH * CONV_TAPS, ALIGNMENT)?;
    let conv_state_seed = layout.reserve(SLOTS * slot_values, ALIGNMENT)?;
    let state_rows = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let embedding = layout.reserve(embed_values, ALIGNMENT)?;
    let key = layout.reserve(stream_values, ALIGNMENT)?;
    let key_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let query_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let value = layout.reserve(MAX_ROWS * BRANCH, ALIGNMENT)?;
    let gated = layout.reserve(stream_values, ALIGNMENT)?;
    let gated_normed = layout.reserve(stream_values, ALIGNMENT)?;
    let delta = layout.reserve(stream_values, ALIGNMENT)?;
    let injected = layout.reserve(stream_values, ALIGNMENT)?;
    let conv_state = layout.reserve(SLOTS * slot_values, ALIGNMENT)?;
    let snapshot = layout.reserve(slot_values, ALIGNMENT)?;

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
            conv_state_seed,
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
            snapshot,
        },
    ))
}

fn fixture() -> Fixture {
    let codes = (0..MAX_ROWS * EMBED)
        .map(|index| {
            let token = index / EMBED;
            CODE_PATTERN[(index + 3 * token) & 15]
        })
        .collect();
    // Branch `c` is scaled and rotated so the four gate dot products differ.
    // The reference widens the stream as four identical embedding copies, but
    // the engram delta is added into that stream before the first mix, so no
    // Distinct branches prevent an implementation from assuming symmetry.
    let hidden = (0..MAX_ROWS * WIDTH)
        .map(|index| {
            let token = index / WIDTH;
            let column = index % WIDTH;
            let branch = column / BRANCH;
            let scale = 1.0 + branch as f32 * 0.25;
            f32_to_bf16(
                HIDDEN_PATTERN[(column + 5 * branch + token) & 15]
                    * scale
                    * (1.0 - (token & 7) as f32 / 32.0),
            )
        })
        .collect();
    let key_proj = projection_plane(&KEY_PATTERN, WIDTH, EMBED, 7);
    let value_proj = projection_plane(&VALUE_PATTERN, EMBED, EMBED, 11);
    let norm_key = (0..WIDTH)
        .map(|index| f32_to_bf16(NORM_WEIGHT_PATTERN[index & 7]))
        .collect();
    let norm_query = (0..WIDTH)
        .map(|index| f32_to_bf16(NORM_WEIGHT_PATTERN[(index * 3 + 1) & 7]))
        .collect();
    let norm_conv = (0..WIDTH)
        .map(|index| f32_to_bf16(NORM_WEIGHT_PATTERN[(index * 5 + 3) & 7]))
        .collect();
    let conv_weight = (0..WIDTH * CONV_TAPS)
        .map(|index| f32_to_bf16(CONV_PATTERN[(index + index / CONV_TAPS) & 7]))
        .collect();
    // Every slot carries a distinct history so a route that read the wrong one,
    // or read a fixed one, cannot pass.
    let conv_state_seed = (0..SLOTS * WIDTH * CONV_STATE)
        .map(|index| {
            let slot = index / (WIDTH * CONV_STATE);
            f32_to_bf16(CONV_STATE_PATTERN[(index + 3 * slot) & 7] * (1.0 + slot as f32 / 8.0))
        })
        .collect();
    // Decode row `t` owns slot `t`; a prefill tile is one sequence and reads
    // only the first entry.
    let state_rows = (0..MAX_ROWS).map(|row| (row % SLOTS) as u32).collect();

    Fixture {
        codes,
        hidden,
        key_proj,
        value_proj,
        norm_key,
        norm_query,
        norm_conv,
        conv_weight,
        conv_state_seed,
        state_rows,
    }
}

/// Builds one projection plane, rotating the pattern per row so different rows
/// land at different points of the gate's nonlinearity.
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
    arena.copy_from_host(stream, regions.codes, &fixture.codes)?;
    arena.copy_from_host(stream, regions.hidden, &fixture.hidden)?;
    arena.copy_from_host(stream, regions.key_proj, &fixture.key_proj)?;
    arena.copy_from_host(stream, regions.value_proj, &fixture.value_proj)?;
    arena.copy_from_host(stream, regions.norm_key, &fixture.norm_key)?;
    arena.copy_from_host(stream, regions.norm_query, &fixture.norm_query)?;
    arena.copy_from_host(stream, regions.norm_conv, &fixture.norm_conv)?;
    arena.copy_from_host(stream, regions.conv_weight, &fixture.conv_weight)?;
    arena.copy_from_host(stream, regions.conv_state_seed, &fixture.conv_state_seed)?;
    arena.copy_from_host(stream, regions.state_rows, &fixture.state_rows)
}

/// Resets the staged planes to the sentinel and the convolution history to its
/// seed, because the convolution mutates the history it reads.
fn reset_outputs(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    for (region, _) in regions.token_planes() {
        arena.fill(stream, region, 0xa5)?;
    }
    arena.fill(stream, regions.snapshot, 0xa5)?;
    arena.copy_from_host(stream, regions.conv_state, &fixture.conv_state_seed)
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Vec<usize>> {
    let mut addresses = vec![
        arena.address(regions.codes)?.addr(),
        arena.address(regions.state_rows)?.addr(),
        arena.address(regions.conv_state)?.addr(),
        arena.address(regions.snapshot)?.addr(),
    ];
    for region in [
        regions.hidden,
        regions.key_proj,
        regions.value_proj,
        regions.norm_key,
        regions.norm_query,
        regions.norm_conv,
        regions.conv_weight,
        regions.conv_state_seed,
    ] {
        addresses.push(arena.address(region)?.addr());
    }
    for (region, _) in regions.token_planes() {
        addresses.push(arena.address(region)?.addr());
    }

    Ok(addresses)
}

fn sources(arena: &DeviceArena, regions: Regions) -> GpuResult<Qwen38FlashNextEngramSources> {
    Ok(Qwen38FlashNextEngramSources {
        key_proj: arena.address(regions.key_proj)?,
        value_proj: arena.address(regions.value_proj)?,
        norm_key: arena.address(regions.norm_key)?,
        norm_query: arena.address(regions.norm_query)?,
        norm_conv: arena.address(regions.norm_conv)?,
        convolution: arena.address(regions.conv_weight)?,
        table_scale_bits: TABLE_SCALE_BITS,
    })
}

fn workspace(arena: &DeviceArena, regions: Regions) -> GpuResult<Qwen38FlashNextEngramWorkspace> {
    Ok(Qwen38FlashNextEngramWorkspace {
        embedding: arena.address(regions.embedding)?,
        key: arena.address(regions.key)?,
        key_normed: arena.address(regions.key_normed)?,
        query_normed: arena.address(regions.query_normed)?,
        value: arena.address(regions.value)?,
        gated: arena.address(regions.gated)?,
        gated_normed: arena.address(regions.gated_normed)?,
        delta: arena.address(regions.delta)?,
    })
}

fn launch(
    engram: &Qwen38FlashNextEngramOp,
    norm: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    // SAFETY: every region is aligned, disjoint, context-local, and covers
    // `MAX_ROWS`, which bounds every admitted route.
    unsafe {
        engram.launch_engram(
            norm,
            stream,
            rows,
            arena.address(regions.codes)?,
            arena.address(regions.hidden)?,
            sources(arena, regions)?,
            workspace(arena, regions)?,
            arena.address(regions.state_rows)?,
            arena.address(regions.conv_state)?,
            arena.address(regions.injected)?,
        )
    }
}

fn observe(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Observed> {
    Ok(Observed {
        embedding: arena.copy_to_host(stream, regions.embedding)?,
        key: arena.copy_to_host(stream, regions.key)?,
        key_normed: arena.copy_to_host(stream, regions.key_normed)?,
        query_normed: arena.copy_to_host(stream, regions.query_normed)?,
        value: arena.copy_to_host(stream, regions.value)?,
        gated: arena.copy_to_host(stream, regions.gated)?,
        gated_normed: arena.copy_to_host(stream, regions.gated_normed)?,
        delta: arena.copy_to_host(stream, regions.delta)?,
        injected: arena.copy_to_host(stream, regions.injected)?,
        conv_state: arena.copy_to_host(stream, regions.conv_state)?,
    })
}

fn truncate(observed: &Observed, rows: usize) -> Observed {
    Observed {
        embedding: observed.embedding[..rows * EMBED].to_vec(),
        key: observed.key[..rows * WIDTH].to_vec(),
        key_normed: observed.key_normed[..rows * WIDTH].to_vec(),
        query_normed: observed.query_normed[..rows * WIDTH].to_vec(),
        value: observed.value[..rows * BRANCH].to_vec(),
        gated: observed.gated[..rows * WIDTH].to_vec(),
        gated_normed: observed.gated_normed[..rows * WIDTH].to_vec(),
        delta: observed.delta[..rows * WIDTH].to_vec(),
        injected: observed.injected[..rows * WIDTH].to_vec(),
        conv_state: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Independent oracle
// ---------------------------------------------------------------------------

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f64) -> f64 {
    value * logistic(value)
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

/// Decodes one E4M3FN code straight from its bits.
///
/// The engram table is the only place this target reads FP8 outside a ModelOpt
/// scale, so the decode is transcribed here rather than borrowed from a
/// quantized-projection oracle.
fn decode_e4m3fn(code: u8) -> f64 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let fraction = code & 0x07;
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0.0,
        (0, fraction) => f64::from(fraction) * 2.0f64.powi(-9),
        (15, 7) => f64::NAN,
        (exponent, fraction) => {
            (1.0 + f64::from(fraction) / 8.0) * 2.0f64.powi(i32::from(exponent) - 7)
        }
    };

    sign * magnitude
}

/// `E = code * table_scale`. Every E4M3 value is exact in BF16, so the
/// product is the only rounding site.
fn dequant_oracle(codes: &[u8], scale: f64) -> Vec<u16> {
    codes
        .iter()
        .map(|&code| f32_to_bf16((decode_e4m3fn(code) * scale) as f32))
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
}

/// One `nn.Linear` over the dequantized embedding, rounded to BF16 once.
fn projection_oracle(embedding: &[f64], weight: &[f64], rows: usize) -> Vec<u16> {
    (0..rows)
        .map(|row| f32_to_bf16(dot(&weight[row * EMBED..(row + 1) * EMBED], embedding) as f32))
        .collect()
}

/// `Qwen4ExpTextRMSNorm(10240, group_size=2560)`: four independent 2,560-wide
/// RMSNorms, flattened, then one 10,240-wide `(1 + w)`.
///
/// This is the same law the hyper-connection family's `hc_norm` implements, and
/// the engram routes launch that entry rather than a fourth copy of it. The
/// oracle is written here independently so the reuse is proved rather than
/// assumed.
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

/// The checkpoint's signed square root.
///
/// `clamp_min` guards the square root and does **not** restore a sign, so a dot
/// product of exactly zero gates on `sigmoid(0) = 0.5`.
fn signed_root(scaled: f64) -> f64 {
    let magnitude = through_bf16(scaled.abs().max(through_bf16(f64::from(GATE_FLOOR))));
    let root = through_bf16(magnitude.sqrt());

    if scaled > 0.0 {
        root
    } else if scaled < 0.0 {
        -root
    } else {
        0.0
    }
}

/// One branch's gate activation: `sigmoid(signed_sqrt(dot / sqrt(2560)))`.
///
/// The reference materializes `key_normed * query_normed` as a BF16 tensor
/// before `sum(-1)`, so every product is rounded before the accumulation.
fn gate_activation(key_normed: &[f64], query_normed: &[f64]) -> f64 {
    let total = key_normed
        .iter()
        .zip(query_normed)
        .map(|(key, query)| through_bf16(key * query))
        .sum::<f64>();
    let scaled = through_bf16(through_bf16(total) / f64::from((BRANCH as f32).sqrt()));

    through_bf16(logistic(signed_root(scaled)))
}

/// `sigmoid(gate) * value` broadcast into the four branches, then flattened.
fn gated_oracle(activation: &[f64], value: &[f64]) -> Vec<u16> {
    let mut gated = vec![0u16; WIDTH];
    for branch in 0..BRANCHES {
        for column in 0..BRANCH {
            gated[branch * BRANCH + column] =
                f32_to_bf16((activation[branch] * value[column]) as f32);
        }
    }

    gated
}

/// `gated + silu(conv1d(window))` for one channel.
///
/// `F.conv1d` returns a BF16 tensor before `silu` reads it, and the residual add
/// is a second BF16 tensor op, so both round.
fn conv_oracle(weights: &[f64], window: [f64; CONV_TAPS], residual: f64) -> u16 {
    let sum = weights
        .iter()
        .zip(window)
        .map(|(weight, value)| weight * value)
        .sum::<f64>();

    f32_to_bf16((residual + through_bf16(silu(through_bf16(sum)))) as f32)
}

fn oracle(fixture: &Fixture) -> Oracle {
    let scale = f64::from(f32::from_bits(u32::from(TABLE_SCALE_BITS) << 16));
    let key_proj = widen(&fixture.key_proj);
    let value_proj = widen(&fixture.value_proj);
    let norm_key = widen(&fixture.norm_key);
    let norm_query = widen(&fixture.norm_query);
    let norm_conv = widen(&fixture.norm_conv);
    let mut oracle = Oracle {
        embedding: Vec::with_capacity(MAX_ROWS * EMBED),
        key: Vec::with_capacity(MAX_ROWS * WIDTH),
        key_normed: Vec::with_capacity(MAX_ROWS * WIDTH),
        query_normed: Vec::with_capacity(MAX_ROWS * WIDTH),
        value: Vec::with_capacity(MAX_ROWS * BRANCH),
        gated: Vec::with_capacity(MAX_ROWS * WIDTH),
        gated_normed: Vec::with_capacity(MAX_ROWS * WIDTH),
        activation: Vec::with_capacity(MAX_ROWS * BRANCHES),
    };

    for token in 0..MAX_ROWS {
        let embedding = dequant_oracle(&fixture.codes[token * EMBED..(token + 1) * EMBED], scale);
        let widened = widen(&embedding);
        let key = projection_oracle(&widened, &key_proj, WIDTH);
        let value = projection_oracle(&widened, &value_proj, EMBED);
        let key_normed = grouped_rms_norm_oracle(&widen(&key), &norm_key);
        let query_normed = grouped_rms_norm_oracle(
            &widen(&fixture.hidden[token * WIDTH..(token + 1) * WIDTH]),
            &norm_query,
        );
        let normed_key = widen(&key_normed);
        let normed_query = widen(&query_normed);
        let activation = (0..BRANCHES)
            .map(|branch| {
                let begin = branch * BRANCH;
                gate_activation(
                    &normed_key[begin..begin + BRANCH],
                    &normed_query[begin..begin + BRANCH],
                )
            })
            .collect::<Vec<_>>();
        let gated = gated_oracle(&activation, &widen(&value));
        let gated_normed = grouped_rms_norm_oracle(&widen(&gated), &norm_conv);

        oracle.embedding.extend(embedding);
        oracle.key.extend(key);
        oracle.value.extend(value);
        oracle.key_normed.extend(key_normed);
        oracle.query_normed.extend(query_normed);
        oracle.gated.extend(gated);
        oracle.gated_normed.extend(gated_normed);
        oracle
            .activation
            .extend(activation.into_iter().map(|value| value as f32));
    }

    oracle
}

fn activation_extent(oracle: &Oracle) -> (f32, f32) {
    oracle
        .activation
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
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
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    // The dequantization has no reduction: every value is one E4M3 decode and
    // one product, so it is a bitwise contract rather than a tolerance.
    for index in 0..rows * EMBED {
        if observed.embedding[index] != oracle.embedding[index] {
            return Err(QualificationError::Mismatch(format!(
                "dequantization at rows={rows}, row={}, column={}: device={:#06x}, oracle={:#06x}",
                index / EMBED,
                index % EMBED,
                observed.embedding[index],
                oracle.embedding[index]
            )));
        }
    }

    for (name, plane, expected, width) in [
        ("key", &observed.key, &oracle.key, WIDTH),
        ("value", &observed.value, &oracle.value, BRANCH),
        (
            "key_normed",
            &observed.key_normed,
            &oracle.key_normed,
            WIDTH,
        ),
        (
            "query_normed",
            &observed.query_normed,
            &oracle.query_normed,
            WIDTH,
        ),
        ("gated", &observed.gated, &oracle.gated, WIDTH),
        (
            "gated_normed",
            &observed.gated_normed,
            &oracle.gated_normed,
            WIDTH,
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
            )?;
        }
    }

    verify_convolution(rows, fixture, observed, report)?;
    verify_injection(rows, fixture, observed)?;
    verify_inactive(rows, observed)?;
    verify_untouched_slots(rows, fixture, observed, report)?;

    report.embedding_values += rows * EMBED;
    report.projected_values += rows * (WIDTH + BRANCH);
    report.normalized_values += rows * 2 * WIDTH;
    report.gated_values += rows * 2 * WIDTH;
    report.delta_values += rows * WIDTH;
    report.injected_values += rows * WIDTH;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

/// Checks the convolution delta against the oracle and the published history
/// bit-exactly, both driven by the normalized plane the device produced.
fn verify_convolution(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let decode = rows <= MAX_BATCH;
    let gated_normed = widen(&observed.gated_normed[..rows * WIDTH]);
    let gated = widen(&observed.gated[..rows * WIDTH]);
    let weights = widen(&fixture.conv_weight);
    let seed = widen(&fixture.conv_state_seed);

    for token in 0..rows {
        let slot = if decode {
            fixture.state_rows[token] as usize
        } else {
            fixture.state_rows[0] as usize
        };
        for channel in 0..WIDTH {
            let history = |column: usize| seed[(slot * WIDTH + channel) * CONV_STATE + column];
            let window = if decode {
                [
                    history(0),
                    history(CONV_DILATION),
                    history(2 * CONV_DILATION),
                    gated_normed[token * WIDTH + channel],
                ]
            } else {
                let tap = |distance: usize| {
                    if token >= distance {
                        gated_normed[(token - distance) * WIDTH + channel]
                    } else {
                        history(CONV_STATE - distance + token)
                    }
                };
                [
                    tap(3 * CONV_DILATION),
                    tap(2 * CONV_DILATION),
                    tap(CONV_DILATION),
                    gated_normed[token * WIDTH + channel],
                ]
            };
            let expected = conv_oracle(
                &weights[channel * CONV_TAPS..(channel + 1) * CONV_TAPS],
                window,
                gated[token * WIDTH + channel],
            );
            check_close(
                "delta",
                rows,
                token,
                channel,
                observed.delta[token * WIDTH + channel],
                expected,
                &mut report.maximum_absolute_error,
            )?;
        }
    }

    // The published history is a permutation of values the device already
    // emitted, so it is a bitwise contract.
    let touched = if decode { rows.min(SLOTS) } else { 1 };
    for row in 0..touched {
        let slot = if decode {
            fixture.state_rows[row] as usize
        } else {
            fixture.state_rows[0] as usize
        };
        for channel in 0..WIDTH {
            for column in 0..CONV_STATE {
                let index = (slot * WIDTH + channel) * CONV_STATE + column;
                let expected = if decode {
                    if column + 1 < CONV_STATE {
                        fixture.conv_state_seed[index + 1]
                    } else {
                        observed.gated_normed[row * WIDTH + channel]
                    }
                } else {
                    observed.gated_normed[(rows - CONV_STATE + column) * WIDTH + channel]
                };
                if observed.conv_state[index] != expected {
                    return Err(QualificationError::Mismatch(format!(
                        "convolution history at rows={rows}, slot={slot}, channel={channel}, column={column}: device={:#06x}, oracle={:#06x}",
                        observed.conv_state[index], expected
                    )));
                }
            }
        }
        report.conv_state_values += WIDTH * CONV_STATE;
    }
    Ok(())
}

/// Given the delta the device produced, the injection is exact in FP32 and has
/// no reduction, so it is a bitwise contract.
fn verify_injection(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
) -> Result<(), QualificationError> {
    for index in 0..rows * WIDTH {
        let expected =
            f32_to_bf16(bf16_to_f32(fixture.hidden[index]) + bf16_to_f32(observed.delta[index]));
        if observed.injected[index] != expected {
            return Err(QualificationError::Mismatch(format!(
                "injection at rows={rows}, row={}, column={}: device={:#06x}, oracle={:#06x}",
                index / WIDTH,
                index % WIDTH,
                observed.injected[index],
                expected
            )));
        }
    }

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
    (MAX_ROWS - rows) * (EMBED + BRANCH + 7 * WIDTH)
}

/// A narrower route must leave every convolution slot it does not name exactly
/// as the seed left it.
fn verify_untouched_slots(
    rows: usize,
    fixture: &Fixture,
    observed: &Observed,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let touched = if rows <= MAX_BATCH {
        rows.min(SLOTS)
    } else {
        1
    };
    let slot_values = WIDTH * CONV_STATE;
    for slot in touched..SLOTS {
        let begin = slot * slot_values;
        if let Some(offset) = observed.conv_state[begin..begin + slot_values]
            .iter()
            .zip(&fixture.conv_state_seed[begin..begin + slot_values])
            .position(|(actual, expected)| actual != expected)
        {
            return Err(QualificationError::Mismatch(format!(
                "rows={rows} modified unnamed convolution slot {slot} at value {offset}"
            )));
        }
        report.untouched_slot_values += slot_values;
    }

    Ok(())
}

fn verify_route_independence(
    rows: usize,
    reference: &Option<Observed>,
    observed: &Observed,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let Some(reference) = reference else {
        return Ok(());
    };

    // Every entry keeps one output's whole reduction inside one warp or one
    // block, and decode row zero and a prefill tile's token zero both name slot
    // zero with the same nine carried columns, so token zero's bits must not
    // depend on which route produced them.
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
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let codes = arena.copy_to_host(stream, regions.codes)?;
    if let Some(index) = codes
        .iter()
        .zip(&fixture.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(QualificationError::Mismatch(format!(
            "read-only engram codes changed at index {index}"
        )));
    }
    report.immutable_input_values += codes.len();

    for (name, region, expected) in [
        ("hidden", regions.hidden, &fixture.hidden),
        ("key_proj", regions.key_proj, &fixture.key_proj),
        ("value_proj", regions.value_proj, &fixture.value_proj),
        ("norm_key", regions.norm_key, &fixture.norm_key),
        ("norm_query", regions.norm_query, &fixture.norm_query),
        ("norm_conv", regions.norm_conv, &fixture.norm_conv),
        ("conv weight", regions.conv_weight, &fixture.conv_weight),
        (
            "conv state seed",
            regions.conv_state_seed,
            &fixture.conv_state_seed,
        ),
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
    report: &mut Qwen38FlashNextPleQualification,
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

    if let Some(index) = replay
        .conv_state
        .iter()
        .zip(&eager.conv_state)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(QualificationError::Mismatch(format!(
            "rows={rows} convolution history graph replay differs from eager at value {index}"
        )));
    }
    report.graph_replay_values += replay.conv_state.len();

    verify_inactive(rows, replay)?;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

/// Proves the `sign(0) = 0` edge and signed-root discrimination.
///
/// The gate entry is driven directly with a crafted normalized pair so the four
/// branches land on an exactly-zero dot product, a large positive one, its
/// negation, and a small positive one that leaves the `clamp_min` inert.
fn verify_signed_root_edge(
    engram: &Qwen38FlashNextEngramOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let (key, query) = signed_root_probe();
    arena.copy_from_host(stream, regions.key_normed, &key)?;
    arena.copy_from_host(stream, regions.query_normed, &query)?;
    arena.fill(stream, regions.gated, 0xa5)?;
    // SAFETY: every region is aligned, disjoint, context-local, and covers one
    // complete row.
    unsafe {
        engram.launch_gate(
            stream,
            1,
            arena.address(regions.key_normed)?,
            arena.address(regions.query_normed)?,
            arena.address(regions.value)?,
            arena.address(regions.gated)?,
        )?;
    }
    let gated = arena.copy_to_host(stream, regions.gated)?;
    let value = arena.copy_to_host(stream, regions.value)?;
    let expected = (0..BRANCHES)
        .map(|branch| {
            let begin = branch * BRANCH;
            gate_activation(
                &widen(&key[begin..begin + BRANCH]),
                &widen(&query[begin..begin + BRANCH]),
            )
        })
        .collect::<Vec<_>>();

    for (branch, activation) in expected.iter().enumerate() {
        for (column, &value_bits) in value[..BRANCH].iter().enumerate() {
            let index = branch * BRANCH + column;
            let oracle_bits = f32_to_bf16((activation * f64::from(bf16_to_f32(value_bits))) as f32);
            check_close(
                "signed-root probe",
                1,
                branch,
                column,
                gated[index],
                oracle_bits,
                &mut report.maximum_absolute_error,
            )?;
        }
    }

    // Branch zero's dot product is exactly zero, so the reference's `sign` is
    // exactly zero and the activation is `sigmoid(0)`, not `sigmoid(sqrt(1e-6))`
    // - the two coincide in BF16, so what this pins bitwise is the represented
    // value, while branches one and two are what discriminate the sign.
    let zero_activation = f64::from(bf16_to_f32(f32_to_bf16(0.5)));
    if expected[0] != zero_activation {
        return Err(QualificationError::Mismatch(format!(
            "signed-root probe branch zero gated on {} rather than sigmoid(0)",
            expected[0]
        )));
    }
    for column in 0..BRANCH {
        let oracle_bits = f32_to_bf16((0.5 * f64::from(bf16_to_f32(value[column]))) as f32);
        if gated[column] != oracle_bits {
            return Err(QualificationError::Mismatch(format!(
                "signed-root probe zero branch at column {column}: device={:#06x}, oracle={:#06x}",
                gated[column], oracle_bits
            )));
        }
    }

    report.zero_branch_activation = expected[0] as f32;
    report.signed_branch_activations = (expected[1] as f32, expected[2] as f32);

    Ok(())
}

/// Builds the crafted normalized pair the signed-root probe drives.
///
/// The query row is all ones and the key row carries `+/-PROBE_MAGNITUDE`, so
/// every product is exact in BF16 and every branch dot product is an exact FP32
/// value no reduction order can perturb.
fn signed_root_probe() -> (Vec<u16>, Vec<u16>) {
    let query = vec![f32_to_bf16(1.0); MAX_ROWS * WIDTH];
    let mut key = vec![f32_to_bf16(0.0); MAX_ROWS * WIDTH];
    for column in 0..BRANCH {
        // Branch 0: alternating signs, so the dot product is exactly zero.
        key[column] = f32_to_bf16(if column % 2 == 0 {
            PROBE_MAGNITUDE
        } else {
            -PROBE_MAGNITUDE
        });
        // Branch 1 and 2: the same magnitude with opposite signs, which a route
        // that dropped `gate.sign()` would collapse onto one activation.
        key[BRANCH + column] = f32_to_bf16(PROBE_MAGNITUDE);
        key[2 * BRANCH + column] = f32_to_bf16(-PROBE_MAGNITUDE);
        // Branch 3: far above the 1e-6 floor, so `clamp_min` stays inert.
        key[3 * BRANCH + column] = f32_to_bf16(PROBE_MAGNITUDE / 64.0);
    }

    (key, query)
}

/// Proves the documented aliasing contract: the injection may publish into the
/// stream it reads.
fn verify_in_place_injection(
    engram: &Qwen38FlashNextEngramOp,
    norm: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    reset_outputs(arena, stream, regions, fixture)?;
    launch(engram, norm, arena, stream, regions, MAX_ROWS)?;
    let expected = arena.copy_to_host(stream, regions.injected)?;
    arena.copy_from_host(stream, regions.injected, &fixture.hidden)?;
    // SAFETY: `injected` is the documented in-place form: the output aliases
    // the input exactly, and every thread writes only the pair it read.
    unsafe {
        engram.launch_inject(
            stream,
            MAX_ROWS,
            arena.address(regions.injected)?,
            arena.address(regions.delta)?,
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
            "in-place injection differs from the disjoint form at value {index}: in-place={:#06x}, disjoint={:#06x}",
            actual[index], expected[index]
        )));
    }
    report.in_place_values += actual.len();

    Ok(())
}

/// Proves the cancellation discipline: a captured slot put back is the slot the
/// capture read, bit for bit, and nothing else moved.
fn verify_snapshot_round_trip(
    snapshot: &Qwen38FlashNextPleStateSnapshotOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextPleQualification,
) -> Result<(), QualificationError> {
    let slot = 3usize;
    let slot_values = WIDTH * CONV_STATE;
    arena.copy_from_host(stream, regions.conv_state, &fixture.conv_state_seed)?;
    arena.fill(stream, regions.snapshot, 0xa5)?;
    let selected = arena.address(regions.state_rows)?;
    // SAFETY: `state_rows[slot] == slot as u32` by construction, and both slot
    // planes are aligned, disjoint, and context-local.
    unsafe {
        snapshot.launch_snapshot(
            stream,
            selected.wrapping_add(slot).cast_const(),
            arena.address(regions.conv_state)?,
            arena.address(regions.snapshot)?,
        )?;
    }
    let captured = arena.copy_to_host(stream, regions.snapshot)?;
    let expected = &fixture.conv_state_seed[slot * slot_values..(slot + 1) * slot_values];
    if let Some(index) = captured
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(QualificationError::Mismatch(format!(
            "convolution slot capture differs from the seeded slot at value {index}"
        )));
    }

    // Overwrite the slot the way a provisional step would, then cancel it. The
    // neighbouring slots keep their seed so the restore is proved to touch one
    // slot and no more.
    let mut scribbled = fixture.conv_state_seed.clone();
    scribbled[slot * slot_values..(slot + 1) * slot_values].fill(INACTIVE_SENTINEL);
    arena.copy_from_host(stream, regions.conv_state, &scribbled)?;
    // SAFETY: the restore arm carries the capture's contract with the roles
    // reversed.
    unsafe {
        snapshot.launch_restore(
            stream,
            selected.wrapping_add(slot).cast_const(),
            arena.address(regions.conv_state)?,
            arena.address(regions.snapshot)?,
        )?;
    }
    let restored = arena.copy_to_host(stream, regions.conv_state)?;
    if let Some(index) = restored
        .iter()
        .zip(&fixture.conv_state_seed)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(QualificationError::Mismatch(format!(
            "convolution slot restore left the plane different at value {index}"
        )));
    }
    report.snapshot_values += captured.len() + restored.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    engram: &Qwen38FlashNextEngramOp,
    norm: &Qwen38FlashNextHyperConnectionOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), QualificationError> {
    let mut graphs = Vec::with_capacity(ROUTES.len());
    for rows in ROUTES {
        graphs.push(CudaGraph::capture(stream, || {
            launch(engram, norm, arena, stream, regions, rows)
        })?);
    }
    // Warmup replays the exact pattern the measurement replays, because the
    // driver sizes its graph-launch storage from what it has actually seen: a
    // forward pass alone leaves the reverse order's first launch to allocate.
    let replay = || -> GpuResult<()> {
        for _ in 0..4 {
            for graph in graphs.iter().rev() {
                // SAFETY: the qualification owner retains every allocation
                // captured by these graphs.
                unsafe { graph.launch(stream) }?;
            }
        }

        stream.synchronize()?;
        Ok(())
    };
    for graph in &graphs {
        // SAFETY: the qualification owner retains every allocation captured by these graphs.
        unsafe { graph.launch(stream) }?;
    }
    replay()?;
    let before = device_memory_info(context)?;
    replay()?;
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
) -> Result<(), QualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_absolute_error = maximum_absolute_error.max(error);
    let tolerance = 0.015625f32.max(oracle.abs() * 0.005);
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
        BRANCH, BRANCHES, CONV_DILATION, CONV_STATE, CONV_TAPS, EMBED, MAX_ROWS, ROUTES, SLOTS,
        TABLE_SCALE_BITS, WIDTH, decode_e4m3fn, fixture, gate_activation, layout,
        qualify_qwen38_flash_next_ple, signed_root, signed_root_probe, widen,
    };

    /// `sign(0) = 0` forces `sigmoid(0) = 0.5`; the `clamp_min`
    /// guards the square root and never restores a sign.
    #[test]
    fn qwen38_flash_next_ple_suite_signed_root_keeps_the_sign_and_the_zero() {
        assert_eq!(signed_root(0.0), 0.0);
        assert_eq!(signed_root(-0.0), 0.0);
        assert_eq!(signed_root(4.0), 2.0);
        assert_eq!(signed_root(-4.0), -2.0);
        // Below the floor the magnitude is clamped, but the sign survives.
        assert!(signed_root(-1.0e-12) < 0.0);
        assert!(signed_root(1.0e-12) > 0.0);
        assert_eq!(signed_root(1.0e-12), -signed_root(-1.0e-12));
    }

    /// The probe must actually separate the two signs, or it would qualify a
    /// route that computed `sqrt(|gate|)` and dropped `gate.sign()`.
    #[test]
    fn qwen38_flash_next_ple_suite_probe_discriminates_the_gate_sign() {
        let (key, query) = signed_root_probe();
        let activations = (0..BRANCHES)
            .map(|branch| {
                let begin = branch * BRANCH;
                gate_activation(
                    &widen(&key[begin..begin + BRANCH]),
                    &widen(&query[begin..begin + BRANCH]),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(activations[0], 0.5);
        assert!(activations[1] > 0.8, "{activations:?}");
        assert!(activations[2] < 0.2, "{activations:?}");
        assert!(activations[1] - activations[2] > 0.6, "{activations:?}");
        // Branch three stays far above the floor, so the clamp is inert there
        // and its activation is neither saturated nor the zero branch's.
        assert!(
            activations[3] > 0.5 && activations[3] < 0.7,
            "{activations:?}"
        );
    }

    /// The residual add must have two visible terms. A fixture whose
    /// `gated` term vanished beside the convolution output would qualify a
    /// route that dropped it.
    #[test]
    fn qwen38_flash_next_ple_suite_fixture_keeps_both_residual_terms_visible() {
        let fixture = fixture();
        let hidden = widen(&fixture.hidden[..WIDTH]);
        let roots = (0..BRANCHES)
            .map(|branch| {
                let begin = branch * BRANCH;
                (hidden[begin..begin + BRANCH]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    / BRANCH as f64)
                    .sqrt()
            })
            .collect::<Vec<_>>();

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

    /// Every slot must carry a distinct history, or a route that read a fixed
    /// slot would pass.
    #[test]
    fn qwen38_flash_next_ple_suite_every_convolution_slot_is_distinct() {
        let fixture = fixture();
        let slot_values = WIDTH * CONV_STATE;

        assert_eq!(fixture.conv_state_seed.len(), SLOTS * slot_values);
        for slot in 0..SLOTS {
            for other in slot + 1..SLOTS {
                assert_ne!(
                    fixture.conv_state_seed[slot * slot_values..(slot + 1) * slot_values],
                    fixture.conv_state_seed[other * slot_values..(other + 1) * slot_values],
                    "slots {slot} and {other} carry the same history"
                );
            }
        }
        for row in 0..SLOTS {
            assert_eq!(fixture.state_rows[row], row as u32);
        }
    }

    /// The engram codes are the only FP8 this target reads outside a ModelOpt
    /// scale, so the decode and the admitted multiplier are pinned here.
    #[test]
    fn qwen38_flash_next_ple_suite_engram_codes_and_scale_are_pinned() {
        assert_eq!(TABLE_SCALE_BITS, 0x3951);
        assert_eq!(
            f32::from_bits(u32::from(TABLE_SCALE_BITS) << 16),
            1.993_179_3e-4
        );
        assert_eq!(decode_e4m3fn(0x38), 1.0);
        assert_eq!(decode_e4m3fn(0xb8), -1.0);
        assert_eq!(decode_e4m3fn(0x3c), 1.5);
        assert_eq!(decode_e4m3fn(0x00), 0.0);

        let fixture = fixture();
        let distinct = fixture.codes[..EMBED]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert!(distinct.len() >= 8, "the code fixture is degenerate");
    }

    #[test]
    fn qwen38_flash_next_ple_suite_route_and_arena_inventory_is_exact() {
        assert_eq!(ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
        assert_eq!(WIDTH, BRANCHES * BRANCH);
        assert_eq!(EMBED, BRANCH);
        assert_eq!(CONV_STATE, (CONV_TAPS - 1) * CONV_DILATION);

        let (arena, regions) = layout().unwrap();
        assert_eq!(arena.byte_len(), regions.payload_bytes());
        assert_eq!(arena.byte_len(), 249_696_256);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_ple_suite_exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), super::Qwen38FlashNextPleQualificationError> {
        let report = qualify_qwen38_flash_next_ple()?;
        let rows = ROUTES.into_iter().sum::<usize>();

        assert_eq!(report.embedding_values, rows * EMBED);
        assert_eq!(report.projected_values, rows * (WIDTH + BRANCH));
        assert_eq!(report.normalized_values, rows * 2 * WIDTH);
        assert_eq!(report.gated_values, rows * 2 * WIDTH);
        assert_eq!(report.delta_values, rows * WIDTH);
        assert_eq!(report.injected_values, rows * WIDTH);
        assert_eq!(report.in_place_values, MAX_ROWS * WIDTH);
        assert_eq!(
            report.route_independent_values,
            (ROUTES.len() - 1) * (EMBED + BRANCH + 7 * WIDTH)
        );
        assert_eq!(report.snapshot_values, (SLOTS + 1) * WIDTH * CONV_STATE);

        // A fixture whose gates all saturated would qualify the plumbing and
        // none of the algebra.
        assert!(report.minimum_gate_activation > 0.0);
        assert!(report.maximum_gate_activation < 1.0);
        assert!(report.maximum_gate_activation - report.minimum_gate_activation > 0.0625);
        assert_eq!(report.zero_branch_activation, 0.5);
        assert!(report.signed_branch_activations.0 > 0.8);
        assert!(report.signed_branch_activations.1 < 0.2);
        assert!(report.maximum_absolute_error <= 0.015625);

        let (arena, regions) = layout()?;
        assert_eq!(report.padding_bytes, 0);
        assert_eq!(report.arena_bytes, 249_696_256);
        assert_eq!(report.arena_bytes, arena.byte_len());
        assert_eq!(regions.payload_bytes(), arena.byte_len());

        Ok(())
    }
}

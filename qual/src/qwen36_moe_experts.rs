//! Qwen3.6 represented-value qualification for routed and shared experts.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::oracles::codecs::decode_e2m1;
use crate::target::Qwen36MoeExpertsOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 128;
pub(crate) const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
pub(crate) const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
pub(crate) const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
pub(crate) const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
pub(crate) const SLOTS: usize = TOP_K + 1;
const GROUP: usize = 16;
const GATE_UP_ROWS: usize = 2 * INTERMEDIATE;
const GATE_UP_CODE_BYTES: usize = GATE_UP_ROWS * HIDDEN / 2;
const GATE_UP_SCALE_BYTES: usize = GATE_UP_ROWS * (HIDDEN / GROUP);
const DOWN_CODE_BYTES: usize = HIDDEN * INTERMEDIATE / 2;
const DOWN_SCALE_BYTES: usize = HIDDEN * (INTERMEDIATE / GROUP);
const BF16_SENTINEL: u16 = 0xa5a5;
const SCALE_CODES: [u8; 4] = [0x28, 0x30, 0x34, 0x38];
const ROUTING_WEIGHTS: [f32; TOP_K] = [
    0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875,
];

/// Failure of exact Qwen3.6 routed/shared expert qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36MoeExpertsQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.6 MoE expert qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst represented-value error across exact batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36MoeExpertsQualification {
    /// Fused gate/up outputs compared against the independent oracle.
    pub intermediate_values: usize,
    /// Per-slot down-projection outputs compared against the oracle.
    pub expert_output_values: usize,
    /// Shared-expert gate logits compared against the BF16 oracle.
    pub shared_gate_values: usize,
    /// Final routed plus shared outputs compared against the oracle.
    pub combined_values: usize,
    /// Active values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside every active route extent.
    pub inactive_values: usize,
    /// Read-only source bytes proved unchanged.
    pub immutable_bytes: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact resident bytes read as weights or expert scalars.
    pub weight_bytes: usize,
    /// Exact address-stable input/output workspace bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference over represented BF16 outputs.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) routing_weights: ArenaRegion<u16>,
    pub(crate) routed_gate_up_codes: ArenaRegion<u8>,
    pub(crate) routed_gate_up_scales: ArenaRegion<u8>,
    pub(crate) routed_gate_up_weight_scales_2: ArenaRegion<f32>,
    pub(crate) routed_down_codes: ArenaRegion<u8>,
    pub(crate) routed_down_scales: ArenaRegion<u8>,
    pub(crate) routed_down_weight_scales_2: ArenaRegion<f32>,
    pub(crate) shared_gate_up_codes: ArenaRegion<u8>,
    pub(crate) shared_gate_up_scales: ArenaRegion<u8>,
    pub(crate) shared_down_codes: ArenaRegion<u8>,
    pub(crate) shared_down_scales: ArenaRegion<u8>,
    pub(crate) shared_gate_weight: ArenaRegion<u16>,
    pub(crate) intermediate: ArenaRegion<u16>,
    pub(crate) expert_output: ArenaRegion<u16>,
    pub(crate) shared_gate: ArenaRegion<u16>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.routed_gate_up_codes.byte_len()
            + self.routed_gate_up_scales.byte_len()
            + self.routed_gate_up_weight_scales_2.byte_len()
            + self.routed_down_codes.byte_len()
            + self.routed_down_scales.byte_len()
            + self.routed_down_weight_scales_2.byte_len()
            + self.shared_gate_up_codes.byte_len()
            + self.shared_gate_up_scales.byte_len()
            + self.shared_down_codes.byte_len()
            + self.shared_down_scales.byte_len()
            + self.shared_gate_weight.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes()
            + self.input.byte_len()
            + self.expert_indices.byte_len()
            + self.routing_weights.byte_len()
            + self.intermediate.byte_len()
            + self.expert_output.byte_len()
            + self.shared_gate.byte_len()
            + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) expert_indices: Vec<u16>,
    pub(crate) routing_weights: Vec<u16>,
    pub(crate) routed_gate_up_codes: Vec<u8>,
    pub(crate) routed_gate_up_scales: Vec<u8>,
    pub(crate) routed_gate_up_weight_scales_2: Vec<f32>,
    pub(crate) routed_down_codes: Vec<u8>,
    pub(crate) routed_down_scales: Vec<u8>,
    pub(crate) routed_down_weight_scales_2: Vec<f32>,
    pub(crate) shared_gate_up_codes: Vec<u8>,
    pub(crate) shared_gate_up_scales: Vec<u8>,
    pub(crate) shared_gate_up_weight_scale_2: Vec<f32>,
    pub(crate) shared_down_codes: Vec<u8>,
    pub(crate) shared_down_scales: Vec<u8>,
    pub(crate) shared_down_weight_scale_2: Vec<f32>,
    pub(crate) shared_gate_weight: Vec<u16>,
    expected_intermediate: Vec<u16>,
    expected_expert_output: Vec<u16>,
    expected_shared_gate: Vec<u16>,
    expected_output: Vec<u16>,
}

struct Outputs {
    intermediate: Vec<u16>,
    expert_output: Vec<u16>,
    shared_gate: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies eager and captured routed/shared expert execution at every exact route.
pub fn qualify_qwen36_moe_experts()
-> Result<Qwen36MoeExpertsQualification, Qwen36MoeExpertsQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen36MoeExpertsOp::new(&context)?;
    let fixture = make_fixture();
    copy_fixture(&arena, &stream, regions, &fixture)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen36MoeExpertsQualification {
        intermediate_values: 0,
        expert_output_values: 0,
        shared_gate_values: 0,
        combined_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_bytes: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        fill_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows, &fixture)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_eager(rows, &fixture, &eager, &mut report)?;

        fill_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&op, &arena, &stream, regions, rows, &fixture)
        })?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "device addresses changed while qualifying rows={rows}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions, &fixture)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_ROWS * HIDDEN, ALIGNMENT)?;
    let expert_indices = layout.reserve(MAX_ROWS * TOP_K, ALIGNMENT)?;
    let routing_weights = layout.reserve(MAX_ROWS * TOP_K, ALIGNMENT)?;
    let routed_gate_up_codes = layout.reserve(EXPERTS * GATE_UP_CODE_BYTES, ALIGNMENT)?;
    let routed_gate_up_scales = layout.reserve(EXPERTS * GATE_UP_SCALE_BYTES, ALIGNMENT)?;
    let routed_gate_up_weight_scales_2 = layout.reserve(EXPERTS, ALIGNMENT)?;
    let routed_down_codes = layout.reserve(EXPERTS * DOWN_CODE_BYTES, ALIGNMENT)?;
    let routed_down_scales = layout.reserve(EXPERTS * DOWN_SCALE_BYTES, ALIGNMENT)?;
    let routed_down_weight_scales_2 = layout.reserve(EXPERTS, ALIGNMENT)?;
    let shared_gate_up_codes = layout.reserve(GATE_UP_CODE_BYTES, ALIGNMENT)?;
    let shared_gate_up_scales = layout.reserve(GATE_UP_SCALE_BYTES, ALIGNMENT)?;
    let shared_down_codes = layout.reserve(DOWN_CODE_BYTES, ALIGNMENT)?;
    let shared_down_scales = layout.reserve(DOWN_SCALE_BYTES, ALIGNMENT)?;
    let shared_gate_weight = layout.reserve(HIDDEN, ALIGNMENT)?;
    let intermediate = layout.reserve(MAX_ROWS * SLOTS * INTERMEDIATE, ALIGNMENT)?;
    let expert_output = layout.reserve(MAX_ROWS * SLOTS * HIDDEN, ALIGNMENT)?;
    let shared_gate = layout.reserve(MAX_ROWS, ALIGNMENT)?;
    let output = layout.reserve(MAX_ROWS * HIDDEN, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            expert_indices,
            routing_weights,
            routed_gate_up_codes,
            routed_gate_up_scales,
            routed_gate_up_weight_scales_2,
            routed_down_codes,
            routed_down_scales,
            routed_down_weight_scales_2,
            shared_gate_up_codes,
            shared_gate_up_scales,
            shared_down_codes,
            shared_down_scales,
            shared_gate_weight,
            intermediate,
            expert_output,
            shared_gate,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Vec<usize>> {
    macro_rules! address {
        ($region:expr) => {
            arena.address($region)?.addr()
        };
    }

    Ok(vec![
        address!(regions.input),
        address!(regions.expert_indices),
        address!(regions.routing_weights),
        address!(regions.routed_gate_up_codes),
        address!(regions.routed_gate_up_scales),
        address!(regions.routed_gate_up_weight_scales_2),
        address!(regions.routed_down_codes),
        address!(regions.routed_down_scales),
        address!(regions.routed_down_weight_scales_2),
        address!(regions.shared_gate_up_codes),
        address!(regions.shared_gate_up_scales),
        address!(regions.shared_down_codes),
        address!(regions.shared_down_scales),
        address!(regions.shared_gate_weight),
        address!(regions.intermediate),
        address!(regions.expert_output),
        address!(regions.shared_gate),
        address!(regions.output),
    ])
}

pub(crate) fn launch(
    op: &Qwen36MoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
    fixture: &Fixture,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.expert_indices)?,
            arena.address(regions.routing_weights)?,
            arena.address(regions.routed_gate_up_codes)?,
            arena.address(regions.routed_gate_up_scales)?,
            arena.address(regions.routed_gate_up_weight_scales_2)?,
            arena.address(regions.routed_down_codes)?,
            arena.address(regions.routed_down_scales)?,
            arena.address(regions.routed_down_weight_scales_2)?,
            arena.address(regions.shared_gate_up_codes)?,
            arena.address(regions.shared_gate_up_scales)?,
            fixture.shared_gate_up_weight_scale_2[0],
            arena.address(regions.shared_down_codes)?,
            arena.address(regions.shared_down_scales)?,
            fixture.shared_down_weight_scale_2[0],
            arena.address(regions.shared_gate_weight)?,
            arena.address(regions.intermediate)?,
            arena.address(regions.expert_output)?,
            arena.address(regions.shared_gate)?,
            arena.address(regions.output)?,
        )
    }
}

pub(crate) fn copy_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    macro_rules! copy {
        ($region:expr, $values:expr) => {
            arena.copy_from_host(stream, $region, $values)?
        };
    }

    copy!(regions.input, &fixture.input);
    copy!(regions.expert_indices, &fixture.expert_indices);
    copy!(regions.routing_weights, &fixture.routing_weights);
    copy!(regions.routed_gate_up_codes, &fixture.routed_gate_up_codes);
    copy!(
        regions.routed_gate_up_scales,
        &fixture.routed_gate_up_scales
    );
    copy!(
        regions.routed_gate_up_weight_scales_2,
        &fixture.routed_gate_up_weight_scales_2
    );
    copy!(regions.routed_down_codes, &fixture.routed_down_codes);
    copy!(regions.routed_down_scales, &fixture.routed_down_scales);
    copy!(
        regions.routed_down_weight_scales_2,
        &fixture.routed_down_weight_scales_2
    );
    copy!(regions.shared_gate_up_codes, &fixture.shared_gate_up_codes);
    copy!(
        regions.shared_gate_up_scales,
        &fixture.shared_gate_up_scales
    );
    copy!(regions.shared_down_codes, &fixture.shared_down_codes);
    copy!(regions.shared_down_scales, &fixture.shared_down_scales);
    copy!(regions.shared_gate_weight, &fixture.shared_gate_weight);

    Ok(())
}

fn fill_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.intermediate, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.expert_output, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.shared_gate, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.output, BF16_SENTINEL as u8)
}

fn read_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Outputs> {
    Ok(Outputs {
        intermediate: arena.copy_to_host(stream, regions.intermediate)?,
        expert_output: arena.copy_to_host(stream, regions.expert_output)?,
        shared_gate: arena.copy_to_host(stream, regions.shared_gate)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

pub(crate) fn make_fixture() -> Fixture {
    let input = (0..MAX_ROWS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let column = index % HIDDEN;
            let pattern = token & (MAX_BATCH - 1);
            f32_to_bf16((pattern as f32 + 1.0) * (column as i32 % 17 - 8) as f32 / 256.0)
        })
        .collect::<Vec<_>>();
    let expert_indices = (0..MAX_ROWS).flat_map(selected_experts).collect::<Vec<_>>();
    let routing_weights = (0..MAX_ROWS)
        .flat_map(|_| ROUTING_WEIGHTS.map(f32_to_bf16))
        .collect::<Vec<_>>();
    let selected = selected_mask(&expert_indices);
    let mut routed_gate_up_codes = vec![0; EXPERTS * GATE_UP_CODE_BYTES];
    let mut routed_gate_up_scales = vec![0; EXPERTS * GATE_UP_SCALE_BYTES];
    let mut routed_down_codes = vec![0; EXPERTS * DOWN_CODE_BYTES];
    let mut routed_down_scales = vec![0; EXPERTS * DOWN_SCALE_BYTES];

    for expert in 0..EXPERTS {
        if selected[expert] {
            fill_expert_plane(
                &mut routed_gate_up_codes
                    [expert * GATE_UP_CODE_BYTES..(expert + 1) * GATE_UP_CODE_BYTES],
                &mut routed_gate_up_scales
                    [expert * GATE_UP_SCALE_BYTES..(expert + 1) * GATE_UP_SCALE_BYTES],
                expert,
                GATE_UP_ROWS,
                HIDDEN,
                0,
            );
            fill_expert_plane(
                &mut routed_down_codes[expert * DOWN_CODE_BYTES..(expert + 1) * DOWN_CODE_BYTES],
                &mut routed_down_scales[expert * DOWN_SCALE_BYTES..(expert + 1) * DOWN_SCALE_BYTES],
                expert,
                HIDDEN,
                INTERMEDIATE,
                1,
            );
        }
    }

    let routed_gate_up_weight_scales_2 = (0..EXPERTS)
        .map(|expert| (1.0 + (expert % 5) as f32 * 0.125) / 512.0)
        .collect::<Vec<_>>();
    let routed_down_weight_scales_2 = (0..EXPERTS)
        .map(|expert| (1.0 + (expert % 7) as f32 * 0.0625) / 512.0)
        .collect::<Vec<_>>();
    let mut shared_gate_up_codes = vec![0; GATE_UP_CODE_BYTES];
    let mut shared_gate_up_scales = vec![0; GATE_UP_SCALE_BYTES];
    let mut shared_down_codes = vec![0; DOWN_CODE_BYTES];
    let mut shared_down_scales = vec![0; DOWN_SCALE_BYTES];
    fill_expert_plane(
        &mut shared_gate_up_codes,
        &mut shared_gate_up_scales,
        263,
        GATE_UP_ROWS,
        HIDDEN,
        2,
    );
    fill_expert_plane(
        &mut shared_down_codes,
        &mut shared_down_scales,
        263,
        HIDDEN,
        INTERMEDIATE,
        3,
    );
    let shared_gate_up_weight_scale_2 = vec![1.0 / 448.0];
    let shared_down_weight_scale_2 = vec![1.0 / 480.0];
    let shared_gate_weight = (0..HIDDEN)
        .map(|column| f32_to_bf16((column as i32 % 13 - 6) as f32 / 1_024.0))
        .collect::<Vec<_>>();
    let mut fixture = Fixture {
        input,
        expert_indices,
        routing_weights,
        routed_gate_up_codes,
        routed_gate_up_scales,
        routed_gate_up_weight_scales_2,
        routed_down_codes,
        routed_down_scales,
        routed_down_weight_scales_2,
        shared_gate_up_codes,
        shared_gate_up_scales,
        shared_gate_up_weight_scale_2,
        shared_down_codes,
        shared_down_scales,
        shared_down_weight_scale_2,
        shared_gate_weight,
        expected_intermediate: vec![0; MAX_ROWS * SLOTS * INTERMEDIATE],
        expected_expert_output: vec![0; MAX_ROWS * SLOTS * HIDDEN],
        expected_shared_gate: vec![0; MAX_ROWS],
        expected_output: vec![0; MAX_ROWS * HIDDEN],
    };
    build_oracle(&mut fixture);

    fixture
}

fn selected_experts(token: usize) -> [u16; TOP_K] {
    let token = (token & (MAX_BATCH - 1)) as u16;
    [
        255 - token,
        token,
        17 + token,
        49 + token,
        81 + token,
        113 + token,
        145 + token,
        177 + token,
    ]
}

fn selected_mask(indices: &[u16]) -> [bool; EXPERTS] {
    let mut selected = [false; EXPERTS];
    for &expert in indices {
        selected[expert as usize] = true;
    }

    selected
}

fn fill_expert_plane(
    codes: &mut [u8],
    scales: &mut [u8],
    expert: usize,
    rows: usize,
    columns: usize,
    plane: usize,
) {
    let code_bytes = columns / 2;
    let groups = columns / GROUP;

    for row in 0..rows {
        let row_codes = &mut codes[row * code_bytes..(row + 1) * code_bytes];
        for (byte, value) in row_codes.iter_mut().enumerate() {
            let low = (expert * 3 + row * 5 + byte * 7 + plane * 11) & 15;
            let high = (expert * 13 + row * 7 + byte * 3 + plane * 5 + 1) & 15;
            *value = low as u8 | (high as u8) << 4;
        }
        for group in 0..groups {
            let offset = scale_offset(row, group, groups);
            scales[offset] = SCALE_CODES[(expert + row * 3 + group * 5 + plane) & 3];
        }
    }
}

fn scale_offset(row: usize, group: usize, groups: usize) -> usize {
    let tile_base = (row / 128) * (groups / 4) * 512;
    let group_tile = (group / 4) * 512;
    let row_lane = (row % 32) * 16 + ((row % 128) / 32) * 4;

    tile_base + group_tile + row_lane + group % 4
}

fn build_oracle(fixture: &mut Fixture) {
    for token in 0..MAX_ROWS {
        if token >= MAX_BATCH {
            let pattern = token & (MAX_BATCH - 1);
            fixture.expected_intermediate.copy_within(
                pattern * SLOTS * INTERMEDIATE..(pattern + 1) * SLOTS * INTERMEDIATE,
                token * SLOTS * INTERMEDIATE,
            );
            fixture.expected_expert_output.copy_within(
                pattern * SLOTS * HIDDEN..(pattern + 1) * SLOTS * HIDDEN,
                token * SLOTS * HIDDEN,
            );
            fixture
                .expected_shared_gate
                .copy_within(pattern..pattern + 1, token);
            fixture
                .expected_output
                .copy_within(pattern * HIDDEN..(pattern + 1) * HIDDEN, token * HIDDEN);
            continue;
        }

        let input = &fixture.input[token * HIDDEN..(token + 1) * HIDDEN];
        for position in 0..SLOTS {
            let slot = token * SLOTS + position;
            let routed = position < TOP_K;
            let expert = if routed {
                fixture.expert_indices[token * TOP_K + position] as usize
            } else {
                0
            };
            let (gate_up_codes, gate_up_scales, gate_up_scale_2, gate_up_expert) = if routed {
                (
                    fixture.routed_gate_up_codes.as_slice(),
                    fixture.routed_gate_up_scales.as_slice(),
                    fixture.routed_gate_up_weight_scales_2[expert],
                    expert,
                )
            } else {
                (
                    fixture.shared_gate_up_codes.as_slice(),
                    fixture.shared_gate_up_scales.as_slice(),
                    fixture.shared_gate_up_weight_scale_2[0],
                    0,
                )
            };
            for row in 0..INTERMEDIATE {
                let gate = nvfp4_dot(
                    input,
                    gate_up_codes,
                    gate_up_scales,
                    gate_up_expert,
                    row,
                    GATE_UP_ROWS,
                    HIDDEN,
                    gate_up_scale_2,
                );
                let up = nvfp4_dot(
                    input,
                    gate_up_codes,
                    gate_up_scales,
                    gate_up_expert,
                    row + INTERMEDIATE,
                    GATE_UP_ROWS,
                    HIDDEN,
                    gate_up_scale_2,
                );
                fixture.expected_intermediate[slot * INTERMEDIATE + row] =
                    f32_to_bf16((gate / (1.0 + (-gate).exp())) * up);
            }

            let intermediate =
                &fixture.expected_intermediate[slot * INTERMEDIATE..(slot + 1) * INTERMEDIATE];
            let (down_codes, down_scales, down_scale_2, down_expert) = if routed {
                (
                    fixture.routed_down_codes.as_slice(),
                    fixture.routed_down_scales.as_slice(),
                    fixture.routed_down_weight_scales_2[expert],
                    expert,
                )
            } else {
                (
                    fixture.shared_down_codes.as_slice(),
                    fixture.shared_down_scales.as_slice(),
                    fixture.shared_down_weight_scale_2[0],
                    0,
                )
            };
            for row in 0..HIDDEN {
                fixture.expected_expert_output[slot * HIDDEN + row] = f32_to_bf16(nvfp4_dot(
                    intermediate,
                    down_codes,
                    down_scales,
                    down_expert,
                    row,
                    HIDDEN,
                    INTERMEDIATE,
                    down_scale_2,
                ));
            }
        }

        let shared_gate = fixture.shared_gate_weight.iter().zip(input).fold(
            0.0f64,
            |sum, (&weight, &activation)| {
                sum + f64::from(bf16_to_f32(weight)) * f64::from(bf16_to_f32(activation))
            },
        );
        fixture.expected_shared_gate[token] = f32_to_bf16(shared_gate as f32);
        let shared_multiplier = 1.0 / (1.0 + (-(shared_gate as f32)).exp());
        for column in 0..HIDDEN {
            let mut sum = 0.0f32;
            for position in 0..TOP_K {
                let expert = bf16_to_f32(
                    fixture.expected_expert_output[(token * SLOTS + position) * HIDDEN + column],
                );
                let weight = bf16_to_f32(fixture.routing_weights[token * TOP_K + position]);
                sum = expert.mul_add(weight, sum);
            }
            let shared = bf16_to_f32(
                fixture.expected_expert_output[(token * SLOTS + TOP_K) * HIDDEN + column],
            );
            fixture.expected_output[token * HIDDEN + column] =
                f32_to_bf16(shared.mul_add(shared_multiplier, sum));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn nvfp4_dot(
    input: &[u16],
    codes: &[u8],
    scales: &[u8],
    expert: usize,
    row: usize,
    rows: usize,
    columns: usize,
    weight_scale_2: f32,
) -> f32 {
    let code_bytes = columns / 2;
    let groups = columns / GROUP;
    let code_stride = rows * code_bytes;
    let scale_stride = rows * groups;
    let code_row = expert * code_stride + row * code_bytes;
    let scale_plane = expert * scale_stride;
    let mut sum = 0.0f64;

    for group in 0..groups {
        let scale = decode_e4m3(scales[scale_plane + scale_offset(row, group, groups)]);
        let mut group_sum = 0.0f64;
        for column in 0..GROUP {
            let packed = codes[code_row + group * (GROUP / 2) + column / 2];
            let code = if column & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            group_sum += f64::from(bf16_to_f32(input[group * GROUP + column]))
                * f64::from(decode_e2m1(code));
        }
        sum += group_sum * f64::from(scale * weight_scale_2);
    }

    sum as f32
}

fn decode_e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;

    if exponent == 0 {
        fraction as f32 / 512.0
    } else {
        f32::from_bits(((exponent as u32 + 120) << 23) | ((fraction as u32) << 20))
    }
}

fn verify_eager(
    rows: usize,
    fixture: &Fixture,
    observed: &Outputs,
    report: &mut Qwen36MoeExpertsQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    let intermediate = rows * SLOTS * INTERMEDIATE;
    let expert_output = rows * SLOTS * HIDDEN;
    let shared_gate = rows;
    let output = rows * HIDDEN;
    compare_bf16(
        rows,
        "intermediate",
        &observed.intermediate[..intermediate],
        &fixture.expected_intermediate[..intermediate],
        0.02,
        report,
    )?;
    compare_bf16(
        rows,
        "expert output",
        &observed.expert_output[..expert_output],
        &fixture.expected_expert_output[..expert_output],
        0.04,
        report,
    )?;
    compare_bf16(
        rows,
        "shared gate",
        &observed.shared_gate[..shared_gate],
        &fixture.expected_shared_gate[..shared_gate],
        0.002,
        report,
    )?;
    compare_bf16(
        rows,
        "combined output",
        &observed.output[..output],
        &fixture.expected_output[..output],
        0.08,
        report,
    )?;
    verify_inactive(rows, observed)?;
    report.intermediate_values += intermediate;
    report.expert_output_values += expert_output;
    report.shared_gate_values += shared_gate;
    report.combined_values += output;
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn compare_bf16(
    rows: usize,
    role: &str,
    actual: &[u16],
    expected: &[u16],
    tolerance: f32,
    report: &mut Qwen36MoeExpertsQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = bf16_to_f32(actual);
        let expected = bf16_to_f32(expected);
        let error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(error);
        if !actual.is_finite() || error > tolerance {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "rows={rows} {role} {index}: device={actual}, oracle={expected}, error={error}"
            )));
        }
    }

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Outputs,
    replay: &Outputs,
    report: &mut Qwen36MoeExpertsQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    if eager.intermediate != replay.intermediate
        || eager.expert_output != replay.expert_output
        || eager.shared_gate != replay.shared_gate
        || eager.output != replay.output
    {
        return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
            "rows={rows} graph replay differs from eager execution"
        )));
    }
    verify_inactive(rows, replay)?;
    report.graph_replay_values += rows * (SLOTS * INTERMEDIATE + SLOTS * HIDDEN + 1 + HIDDEN);
    report.inactive_values += inactive_values(rows);

    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    (MAX_ROWS - rows) * (SLOTS * INTERMEDIATE + SLOTS * HIDDEN + 1 + HIDDEN)
}

fn verify_inactive(
    rows: usize,
    observed: &Outputs,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for (role, values, begin) in [
        (
            "intermediate",
            observed.intermediate.as_slice(),
            rows * SLOTS * INTERMEDIATE,
        ),
        (
            "expert output",
            observed.expert_output.as_slice(),
            rows * SLOTS * HIDDEN,
        ),
        ("shared gate", observed.shared_gate.as_slice(), rows),
        ("combined output", observed.output.as_slice(), rows * HIDDEN),
    ] {
        if let Some(relative) = values[begin..]
            .iter()
            .position(|&value| value != BF16_SENTINEL)
        {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "rows={rows} modified inactive {role} value {}",
                begin + relative
            )));
        }
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen36MoeExpertsQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    macro_rules! same {
        ($region:expr, $expected:expr, $role:literal) => {{
            let observed = arena.copy_to_host(stream, $region)?;
            if observed != *$expected {
                return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                    "read-only {} changed",
                    $role
                )));
            }
            report.immutable_bytes += $region.byte_len();
        }};
    }

    same!(regions.input, &fixture.input, "input");
    same!(
        regions.expert_indices,
        &fixture.expert_indices,
        "expert indices"
    );
    same!(
        regions.routing_weights,
        &fixture.routing_weights,
        "routing weights"
    );
    same!(
        regions.routed_gate_up_codes,
        &fixture.routed_gate_up_codes,
        "routed gate/up codes"
    );
    same!(
        regions.routed_gate_up_scales,
        &fixture.routed_gate_up_scales,
        "routed gate/up scales"
    );
    same!(
        regions.routed_gate_up_weight_scales_2,
        &fixture.routed_gate_up_weight_scales_2,
        "routed gate/up scalar scales"
    );
    same!(
        regions.routed_down_codes,
        &fixture.routed_down_codes,
        "routed down codes"
    );
    same!(
        regions.routed_down_scales,
        &fixture.routed_down_scales,
        "routed down scales"
    );
    same!(
        regions.routed_down_weight_scales_2,
        &fixture.routed_down_weight_scales_2,
        "routed down scalar scales"
    );
    same!(
        regions.shared_gate_up_codes,
        &fixture.shared_gate_up_codes,
        "shared gate/up codes"
    );
    same!(
        regions.shared_gate_up_scales,
        &fixture.shared_gate_up_scales,
        "shared gate/up scales"
    );
    same!(
        regions.shared_down_codes,
        &fixture.shared_down_codes,
        "shared down codes"
    );
    same!(
        regions.shared_down_scales,
        &fixture.shared_down_scales,
        "shared down scales"
    );
    same!(
        regions.shared_gate_weight,
        &fixture.shared_gate_weight,
        "shared gate weights"
    );

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen36MoeExpertsOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    let graphs = EXACT_ROUTES
        .iter()
        .map(|&rows| {
            CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows, fixture))
        })
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..2 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_arena_accounting_covers_all_expert_planes() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(regions.weight_bytes(), 454_760_448);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 6_951_168);
        assert_eq!(layout.byte_len(), 461_711_616);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
        for token in 0..MAX_BATCH {
            let selected = selected_experts(token);
            assert_eq!(
                selected
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                TOP_K
            );
        }
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen36MoeExpertsQualificationError> {
        let report = qualify_qwen36_moe_experts()?;
        let active = EXACT_ROUTES.iter().sum::<usize>();
        let inactive = EXACT_ROUTES
            .iter()
            .map(|&rows| inactive_values(rows))
            .sum::<usize>();

        assert_eq!(report.intermediate_values, active * SLOTS * INTERMEDIATE);
        assert_eq!(report.expert_output_values, active * SLOTS * HIDDEN);
        assert_eq!(report.shared_gate_values, active);
        assert_eq!(report.combined_values, active * HIDDEN);
        assert_eq!(
            report.graph_replay_values,
            active * (SLOTS * INTERMEDIATE + SLOTS * HIDDEN + 1 + HIDDEN)
        );
        assert_eq!(report.inactive_values, 2 * inactive);
        assert_eq!(report.immutable_bytes, 455_288_832);
        assert_eq!(report.arena_bytes, 461_711_616);
        assert_eq!(report.weight_bytes, 454_760_448);
        assert_eq!(report.workspace_bytes, 6_951_168);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

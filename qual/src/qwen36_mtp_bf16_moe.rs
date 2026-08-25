//! Represented-value qualification for the Qwen3.6 MTP BF16 experts.

use crate::Qwen36MoeExpertsQualificationError;
use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::target::Qwen36MtpBf16MoeOp;
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

const MAX_ROWS: usize = 128;
const ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];
const ALIGNMENT: usize = 256;
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const SLOTS: usize = TOP_K + 1;
const GATE_UP_VALUES_PER_EXPERT: usize = 2 * INTERMEDIATE * HIDDEN;
const DOWN_VALUES_PER_EXPERT: usize = HIDDEN * INTERMEDIATE;
const BF16_SENTINEL: u16 = 0xa5a5;
const ROUTING_WEIGHTS: [f32; TOP_K] = [
    0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875,
];

/// Observable counts and ownership for exact Qwen3.6 MTP BF16 expert routes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36MtpBf16MoeQualification {
    /// Active gate/up values checked across every route.
    pub intermediate_values: usize,
    /// Active per-expert down outputs checked across every route.
    pub expert_output_values: usize,
    /// Active fixed-order output values checked across every route.
    pub output_values: usize,
    /// B=1 values checked against the independent source formula.
    pub source_values: usize,
    /// Active output values reproduced exactly by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Inactive sentinel values preserved across every route.
    pub inactive_values: usize,
    /// Selected, shared, and sampled unselected source values proved immutable.
    pub immutable_values: usize,
    /// Exact resident source-weight bytes in the one-allocation arena.
    pub weight_bytes: usize,
    /// Exact address-stable route workspace bytes.
    pub workspace_bytes: usize,
    /// Complete arena bytes including alignment padding.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute error against the independent formula.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    expert_indices: ArenaRegion<u16>,
    routing_weights: ArenaRegion<u16>,
    routed_gate_up: ArenaRegion<u16>,
    routed_down: ArenaRegion<u16>,
    shared_gate: ArenaRegion<u16>,
    shared_up: ArenaRegion<u16>,
    shared_down: ArenaRegion<u16>,
    shared_gate_weight: ArenaRegion<u16>,
    intermediate: ArenaRegion<u16>,
    expert_output: ArenaRegion<u16>,
    shared_gate_output: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    fn weight_bytes(self) -> usize {
        self.routed_gate_up.byte_len()
            + self.routed_down.byte_len()
            + self.shared_gate.byte_len()
            + self.shared_up.byte_len()
            + self.shared_down.byte_len()
            + self.shared_gate_weight.byte_len()
    }

    fn workspace_bytes(self) -> usize {
        self.input.byte_len()
            + self.expert_indices.byte_len()
            + self.routing_weights.byte_len()
            + self.intermediate.byte_len()
            + self.expert_output.byte_len()
            + self.shared_gate_output.byte_len()
            + self.output.byte_len()
    }
}

struct Fixture {
    input: Vec<u16>,
    expert_indices: Vec<u16>,
    routing_weights: Vec<u16>,
    routed_gate_up: Vec<Vec<u16>>,
    routed_down: Vec<Vec<u16>>,
    shared_gate: Vec<u16>,
    shared_up: Vec<u16>,
    shared_down: Vec<u16>,
    shared_gate_weight: Vec<u16>,
}

#[derive(Clone)]
struct Outputs {
    intermediate: Vec<u16>,
    expert_output: Vec<u16>,
    shared_gate: Vec<u16>,
    output: Vec<u16>,
}

/// Qualifies every exact Qwen3.6 MTP BF16 expert route on the target device.
pub fn qualify_qwen36_mtp_bf16_moe()
-> Result<Qwen36MtpBf16MoeQualification, Qwen36MoeExpertsQualificationError> {
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
    let fixture = make_fixture();
    upload_fixture(&arena, &stream, regions, &fixture)?;
    let op = Qwen36MtpBf16MoeOp::new(&context)?;
    let stable_addresses = addresses(&arena, regions)?;
    let source = source_oracle(&fixture);

    reset_outputs(&arena, &stream, regions)?;
    launch(&op, &arena, &stream, regions, 1)?;
    let reference = read_outputs(&arena, &stream, regions)?;
    let mut report = Qwen36MtpBf16MoeQualification {
        intermediate_values: 0,
        expert_output_values: 0,
        output_values: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.weight_bytes() - regions.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };
    verify_source(&reference, &source, &mut report)?;

    for rows in ROUTES {
        reset_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, rows)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_route(rows, &reference, &eager, &mut report)?;

        reset_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, rows))?;
        // SAFETY: the arena and loaded module outlive both replays and synchronization.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the arena and loaded module outlive both replays and synchronization.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay(rows, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "Qwen3.6 MTP BF16 expert addresses changed at rows={rows}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_growth(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let regions = Regions {
        input: layout.reserve(MAX_ROWS * HIDDEN, ALIGNMENT)?,
        expert_indices: layout.reserve(MAX_ROWS * TOP_K, ALIGNMENT)?,
        routing_weights: layout.reserve(MAX_ROWS * TOP_K, ALIGNMENT)?,
        routed_gate_up: layout.reserve(EXPERTS * GATE_UP_VALUES_PER_EXPERT, ALIGNMENT)?,
        routed_down: layout.reserve(EXPERTS * DOWN_VALUES_PER_EXPERT, ALIGNMENT)?,
        shared_gate: layout.reserve(INTERMEDIATE * HIDDEN, ALIGNMENT)?,
        shared_up: layout.reserve(INTERMEDIATE * HIDDEN, ALIGNMENT)?,
        shared_down: layout.reserve(HIDDEN * INTERMEDIATE, ALIGNMENT)?,
        shared_gate_weight: layout.reserve(HIDDEN, ALIGNMENT)?,
        intermediate: layout.reserve(MAX_ROWS * SLOTS * INTERMEDIATE, ALIGNMENT)?,
        expert_output: layout.reserve(MAX_ROWS * SLOTS * HIDDEN, ALIGNMENT)?,
        shared_gate_output: layout.reserve(MAX_ROWS, ALIGNMENT)?,
        output: layout.reserve(MAX_ROWS * HIDDEN, ALIGNMENT)?,
    };
    Ok((layout, regions))
}

fn make_fixture() -> Fixture {
    let input_row = (0..HIDDEN)
        .map(|column| pattern(column, 3, 1.0 / 64.0))
        .collect::<Vec<_>>();
    Fixture {
        input: (0..MAX_ROWS)
            .flat_map(|_| input_row.iter().copied())
            .collect(),
        expert_indices: (0..MAX_ROWS)
            .flat_map(|_| (0..TOP_K).map(|expert| expert as u16))
            .collect(),
        routing_weights: (0..MAX_ROWS)
            .flat_map(|_| ROUTING_WEIGHTS.map(f32_to_bf16))
            .collect(),
        routed_gate_up: (0..TOP_K)
            .map(|expert| {
                (0..GATE_UP_VALUES_PER_EXPERT)
                    .map(|index| pattern(index, 11 + expert, 1.0 / 2_048.0))
                    .collect()
            })
            .collect(),
        routed_down: (0..TOP_K)
            .map(|expert| {
                (0..DOWN_VALUES_PER_EXPERT)
                    .map(|index| pattern(index, 31 + expert, 1.0 / 1_024.0))
                    .collect()
            })
            .collect(),
        shared_gate: (0..INTERMEDIATE * HIDDEN)
            .map(|index| pattern(index, 43, 1.0 / 2_048.0))
            .collect(),
        shared_up: (0..INTERMEDIATE * HIDDEN)
            .map(|index| pattern(index, 47, 1.0 / 2_048.0))
            .collect(),
        shared_down: (0..HIDDEN * INTERMEDIATE)
            .map(|index| pattern(index, 53, 1.0 / 1_024.0))
            .collect(),
        shared_gate_weight: (0..HIDDEN)
            .map(|index| pattern(index, 59, 1.0 / 1_024.0))
            .collect(),
    }
}

fn pattern(index: usize, salt: usize, scale: f32) -> u16 {
    let signed = ((index.wrapping_mul(13).wrapping_add(salt * 7)) % 17) as i32 - 8;
    f32_to_bf16(signed as f32 * scale)
}

fn upload_fixture(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
) -> GpuResult<()> {
    arena.copy_from_host(stream, regions.input, &fixture.input)?;
    arena.copy_from_host(stream, regions.expert_indices, &fixture.expert_indices)?;
    arena.copy_from_host(stream, regions.routing_weights, &fixture.routing_weights)?;
    for expert in 0..TOP_K {
        arena.copy_slice_from_host(
            stream,
            regions.routed_gate_up,
            expert * GATE_UP_VALUES_PER_EXPERT,
            &fixture.routed_gate_up[expert],
        )?;
        arena.copy_slice_from_host(
            stream,
            regions.routed_down,
            expert * DOWN_VALUES_PER_EXPERT,
            &fixture.routed_down[expert],
        )?;
    }
    arena.copy_from_host(stream, regions.shared_gate, &fixture.shared_gate)?;
    arena.copy_from_host(stream, regions.shared_up, &fixture.shared_up)?;
    arena.copy_from_host(stream, regions.shared_down, &fixture.shared_down)?;
    arena.copy_from_host(
        stream,
        regions.shared_gate_weight,
        &fixture.shared_gate_weight,
    )?;
    Ok(())
}

fn reset_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    for region in [regions.intermediate, regions.expert_output, regions.output] {
        arena.fill(stream, region, 0xa5)?;
    }
    arena.fill(stream, regions.shared_gate_output, 0xa5)
}

fn launch(
    op: &Qwen36MtpBf16MoeOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            rows,
            arena.address(regions.input)?.cast_const(),
            arena.address(regions.expert_indices)?.cast_const(),
            arena.address(regions.routing_weights)?.cast_const(),
            arena.address(regions.routed_gate_up)?.cast_const(),
            arena.address(regions.routed_down)?.cast_const(),
            arena.address(regions.shared_gate)?.cast_const(),
            arena.address(regions.shared_up)?.cast_const(),
            arena.address(regions.shared_down)?.cast_const(),
            arena.address(regions.shared_gate_weight)?.cast_const(),
            arena.address(regions.intermediate)?,
            arena.address(regions.expert_output)?,
            arena.address(regions.shared_gate_output)?,
            arena.address(regions.output)?,
        )
    }
}

fn read_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Outputs> {
    Ok(Outputs {
        intermediate: arena.copy_to_host(stream, regions.intermediate)?,
        expert_output: arena.copy_to_host(stream, regions.expert_output)?,
        shared_gate: arena.copy_to_host(stream, regions.shared_gate_output)?,
        output: arena.copy_to_host(stream, regions.output)?,
    })
}

fn source_oracle(fixture: &Fixture) -> Outputs {
    let input = &fixture.input[..HIDDEN];
    let mut intermediate = vec![0; SLOTS * INTERMEDIATE];
    let mut expert_output = vec![0; SLOTS * HIDDEN];
    let mut shared_gate = vec![0; 1];
    let mut output = vec![0; HIDDEN];
    for slot in 0..SLOTS {
        let (gate, up, down) = if slot < TOP_K {
            let gate_up = &fixture.routed_gate_up[slot];
            (
                &gate_up[..INTERMEDIATE * HIDDEN],
                &gate_up[INTERMEDIATE * HIDDEN..],
                fixture.routed_down[slot].as_slice(),
            )
        } else {
            (
                fixture.shared_gate.as_slice(),
                fixture.shared_up.as_slice(),
                fixture.shared_down.as_slice(),
            )
        };
        for row in 0..INTERMEDIATE {
            let gate = dot(input, &gate[row * HIDDEN..(row + 1) * HIDDEN]);
            let up = dot(input, &up[row * HIDDEN..(row + 1) * HIDDEN]);
            intermediate[slot * INTERMEDIATE + row] =
                f32_to_bf16((gate / (1.0 + (-gate).exp())) * up);
        }
        let activation = &intermediate[slot * INTERMEDIATE..(slot + 1) * INTERMEDIATE];
        for row in 0..HIDDEN {
            expert_output[slot * HIDDEN + row] = f32_to_bf16(dot(
                activation,
                &down[row * INTERMEDIATE..(row + 1) * INTERMEDIATE],
            ));
        }
    }
    shared_gate[0] = f32_to_bf16(dot(input, &fixture.shared_gate_weight));
    let shared_coefficient = 1.0 / (1.0 + (-bf16_to_f32(shared_gate[0])).exp());
    for column in 0..HIDDEN {
        let routed = (0..TOP_K).fold(0.0f32, |sum, slot| {
            bf16_to_f32(expert_output[slot * HIDDEN + column])
                .mul_add(bf16_to_f32(fixture.routing_weights[slot]), sum)
        });
        output[column] = f32_to_bf16(
            bf16_to_f32(expert_output[TOP_K * HIDDEN + column]).mul_add(shared_coefficient, routed),
        );
    }
    Outputs {
        intermediate,
        expert_output,
        shared_gate,
        output,
    }
}

fn dot(input: &[u16], weights: &[u16]) -> f32 {
    input
        .iter()
        .zip(weights)
        .fold(0.0f64, |sum, (&input, &weight)| {
            f64::from(bf16_to_f32(input)) * f64::from(bf16_to_f32(weight)) + sum
        }) as f32
}

fn verify_source(
    observed: &Outputs,
    expected: &Outputs,
    report: &mut Qwen36MtpBf16MoeQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for (role, actual, expected) in [
        (
            "intermediate",
            &observed.intermediate[..SLOTS * INTERMEDIATE],
            expected.intermediate.as_slice(),
        ),
        (
            "expert output",
            &observed.expert_output[..SLOTS * HIDDEN],
            expected.expert_output.as_slice(),
        ),
        (
            "shared gate",
            &observed.shared_gate[..1],
            expected.shared_gate.as_slice(),
        ),
        (
            "combined output",
            &observed.output[..HIDDEN],
            expected.output.as_slice(),
        ),
    ] {
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (bf16_to_f32(actual) - bf16_to_f32(expected)).abs();
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            if !error.is_finite() || error > 0.03125 {
                return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                    "{role}[{index}] is {} but source formula is {}",
                    bf16_to_f32(actual),
                    bf16_to_f32(expected)
                )));
            }
            report.source_values += 1;
        }
    }
    Ok(())
}

fn verify_route(
    rows: usize,
    reference: &Outputs,
    observed: &Outputs,
    report: &mut Qwen36MtpBf16MoeQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    verify_repeated(
        "intermediate",
        rows,
        SLOTS * INTERMEDIATE,
        &reference.intermediate,
        &observed.intermediate,
        &mut report.intermediate_values,
        &mut report.inactive_values,
    )?;
    verify_repeated(
        "expert output",
        rows,
        SLOTS * HIDDEN,
        &reference.expert_output,
        &observed.expert_output,
        &mut report.expert_output_values,
        &mut report.inactive_values,
    )?;
    verify_repeated(
        "shared gate",
        rows,
        1,
        &reference.shared_gate,
        &observed.shared_gate,
        &mut report.output_values,
        &mut report.inactive_values,
    )?;
    verify_repeated(
        "combined output",
        rows,
        HIDDEN,
        &reference.output,
        &observed.output,
        &mut report.output_values,
        &mut report.inactive_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_repeated(
    role: &str,
    rows: usize,
    stride: usize,
    reference: &[u16],
    observed: &[u16],
    active_count: &mut usize,
    inactive_count: &mut usize,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for row in 0..rows {
        if observed[row * stride..(row + 1) * stride] != reference[..stride] {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "{role} route rows={rows} disagrees with B=1 at row {row}"
            )));
        }
        *active_count += stride;
    }
    for (index, &value) in observed[rows * stride..].iter().enumerate() {
        if value != BF16_SENTINEL {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "{role} inactive value {index} changed at rows={rows}"
            )));
        }
        *inactive_count += 1;
    }
    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Outputs,
    replay: &Outputs,
    report: &mut Qwen36MtpBf16MoeQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for (role, active, eager, replay) in [
        (
            "intermediate",
            rows * SLOTS * INTERMEDIATE,
            eager.intermediate.as_slice(),
            replay.intermediate.as_slice(),
        ),
        (
            "expert output",
            rows * SLOTS * HIDDEN,
            eager.expert_output.as_slice(),
            replay.expert_output.as_slice(),
        ),
        (
            "shared gate",
            rows,
            eager.shared_gate.as_slice(),
            replay.shared_gate.as_slice(),
        ),
        (
            "combined output",
            rows * HIDDEN,
            eager.output.as_slice(),
            replay.output.as_slice(),
        ),
    ] {
        if eager != replay {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "{role} graph replay disagrees at rows={rows}"
            )));
        }
        report.graph_replay_values += active;
    }
    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen36MtpBf16MoeQualification,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    for expert in 0..TOP_K {
        let gate_up = arena.copy_slice_to_host(
            stream,
            regions.routed_gate_up,
            expert * GATE_UP_VALUES_PER_EXPERT,
            GATE_UP_VALUES_PER_EXPERT,
        )?;
        let down = arena.copy_slice_to_host(
            stream,
            regions.routed_down,
            expert * DOWN_VALUES_PER_EXPERT,
            DOWN_VALUES_PER_EXPERT,
        )?;
        if gate_up != fixture.routed_gate_up[expert] || down != fixture.routed_down[expert] {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "selected expert {expert} source weights changed"
            )));
        }
        report.immutable_values += gate_up.len() + down.len();
    }
    let unselected = TOP_K;
    let gate_up = arena.copy_slice_to_host(
        stream,
        regions.routed_gate_up,
        unselected * GATE_UP_VALUES_PER_EXPERT,
        GATE_UP_VALUES_PER_EXPERT,
    )?;
    let down = arena.copy_slice_to_host(
        stream,
        regions.routed_down,
        unselected * DOWN_VALUES_PER_EXPERT,
        DOWN_VALUES_PER_EXPERT,
    )?;
    if gate_up.iter().any(|&word| word != 0) || down.iter().any(|&word| word != 0) {
        return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
            "unselected expert {unselected} source weights changed"
        )));
    }
    report.immutable_values += gate_up.len() + down.len();
    for (role, actual, expected) in [
        (
            "shared gate",
            arena.copy_to_host(stream, regions.shared_gate)?,
            &fixture.shared_gate,
        ),
        (
            "shared up",
            arena.copy_to_host(stream, regions.shared_up)?,
            &fixture.shared_up,
        ),
        (
            "shared down",
            arena.copy_to_host(stream, regions.shared_down)?,
            &fixture.shared_down,
        ),
        (
            "shared gate weight",
            arena.copy_to_host(stream, regions.shared_gate_weight)?,
            &fixture.shared_gate_weight,
        ),
    ] {
        if actual != *expected {
            return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
                "{role} source weights changed"
            )));
        }
        report.immutable_values += actual.len();
    }
    Ok(())
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<Vec<usize>> {
    Ok(vec![
        arena.address(regions.input)?.addr(),
        arena.address(regions.expert_indices)?.addr(),
        arena.address(regions.routing_weights)?.addr(),
        arena.address(regions.routed_gate_up)?.addr(),
        arena.address(regions.routed_down)?.addr(),
        arena.address(regions.shared_gate)?.addr(),
        arena.address(regions.shared_up)?.addr(),
        arena.address(regions.shared_down)?.addr(),
        arena.address(regions.shared_gate_weight)?.addr(),
        arena.address(regions.intermediate)?.addr(),
        arena.address(regions.expert_output)?.addr(),
        arena.address(regions.shared_gate_output)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn verify_no_growth(
    context: &Arc<CudaContext>,
    op: &Qwen36MtpBf16MoeOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen36MoeExpertsQualificationError> {
    launch(op, arena, stream, regions, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for rows in [1, 8, 32, 128, 3, 64] {
        launch(op, arena, stream, regions, rows)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen36MoeExpertsQualificationError::Mismatch(format!(
            "post-warmup launches changed device memory from {before:?} to {after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen36_mtp_bf16_moe_suite_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();
        assert_eq!(regions.weight_bytes(), 1_616_908_288);
        assert_eq!(regions.workspace_bytes(), 6_951_168);
        assert_eq!(layout.byte_len(), 1_623_859_456);
        assert_eq!(ROUTES.iter().sum::<usize>(), 260);
    }

    #[test]
    #[ignore = "requires an exclusive SM120 device"]
    fn qwen36_mtp_bf16_moe_suite_matches_source_routes_and_graphs() {
        let report = qualify_qwen36_mtp_bf16_moe().expect("Qwen3.6 MTP BF16 MoE qualification");
        assert_eq!(
            report.source_values,
            SLOTS * (INTERMEDIATE + HIDDEN) + 1 + HIDDEN
        );
        assert_eq!(report.weight_bytes, 1_616_908_288);
        assert_eq!(report.workspace_bytes, 6_951_168);
        assert_eq!(report.arena_bytes, 1_623_859_456);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());
    }
}

//! Qwen3.6 represented-value qualification for BF16 router top-8 selection.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::target::Qwen36MoeRouterOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
pub(crate) const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
pub(crate) const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const BF16_SENTINEL: u16 = 0xa5a5;
const INDEX_SENTINEL: u16 = u16::MAX;
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25];

/// Failure of the exact Qwen3.6 MoE router qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36MoeRouterQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.6 MoE router qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst routing-weight error from every exact batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36MoeRouterQualification {
    /// BF16 router logits compared bit-exactly.
    pub logit_values: usize,
    /// Top-eight indices compared exactly.
    pub selected_experts: usize,
    /// BF16 normalized routing weights compared with the independent oracle.
    pub routing_weights: usize,
    /// Active values reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active route extent.
    pub inactive_values: usize,
    /// Read-only input and router-weight values proved unchanged.
    pub immutable_input_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact resident router-weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute routing-weight difference.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) weights: ArenaRegion<u16>,
    pub(crate) logits: ArenaRegion<u16>,
    pub(crate) expert_indices: ArenaRegion<u16>,
    pub(crate) expert_weights: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weights.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len()
            + self.weights.byte_len()
            + self.logits.byte_len()
            + self.expert_indices.byte_len()
            + self.expert_weights.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) weights: Vec<u16>,
    expected_logits: Vec<u16>,
    expected_indices: Vec<u16>,
    expected_weights: Vec<u16>,
}

/// Qualifies eager and captured Qwen3.6 router execution at exact `B=1..=8`.
pub fn qualify_qwen36_moe_router()
-> Result<Qwen36MoeRouterQualification, Qwen36MoeRouterQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen36MoeRouterOp::new(&context)?;
    let fixture = make_fixture();
    arena.copy_from_host(&stream, regions.input, &fixture.input)?;
    arena.copy_from_host(&stream, regions.weights, &fixture.weights)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen36MoeRouterQualification {
        logit_values: 0,
        selected_experts: 0,
        routing_weights: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        fill_outputs(&arena, &stream, regions)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        fill_outputs(&arena, &stream, regions)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = read_outputs(&arena, &stream, regions)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;
    let weights = layout.reserve(EXPERTS * HIDDEN, ALIGNMENT)?;
    let logits = layout.reserve(MAX_BATCH * EXPERTS, ALIGNMENT)?;
    let expert_indices = layout.reserve(MAX_BATCH * TOP_K, ALIGNMENT)?;
    let expert_weights = layout.reserve(MAX_BATCH * TOP_K, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weights,
            logits,
            expert_indices,
            expert_weights,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 5]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weights)?.addr(),
        arena.address(regions.logits)?.addr(),
        arena.address(regions.expert_indices)?.addr(),
        arena.address(regions.expert_weights)?.addr(),
    ])
}

pub(crate) fn launch(
    op: &Qwen36MoeRouterOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.weights)?,
            arena.address(regions.logits)?,
            arena.address(regions.expert_indices)?,
            arena.address(regions.expert_weights)?,
        )
    }
}

fn fill_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<()> {
    arena.fill(stream, regions.logits, BF16_SENTINEL as u8)?;
    arena.fill(stream, regions.expert_indices, INDEX_SENTINEL as u8)?;
    arena.fill(stream, regions.expert_weights, BF16_SENTINEL as u8)
}

struct Outputs {
    logits: Vec<u16>,
    indices: Vec<u16>,
    weights: Vec<u16>,
}

fn read_outputs(arena: &DeviceArena, stream: &CudaStream, regions: Regions) -> GpuResult<Outputs> {
    Ok(Outputs {
        logits: arena.copy_to_host(stream, regions.logits)?,
        indices: arena.copy_to_host(stream, regions.expert_indices)?,
        weights: arena.copy_to_host(stream, regions.expert_weights)?,
    })
}

pub(crate) fn make_fixture() -> Fixture {
    let input = (0..MAX_BATCH * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let column = index % HIDDEN;
            let value = if column == 0 {
                TOKEN_FACTORS[token]
            } else if column & 1 == 0 {
                0.5
            } else {
                -0.5
            };
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();
    let weights = (0..EXPERTS * HIDDEN)
        .map(|index| {
            let expert = index / HIDDEN;
            let column = index % HIDDEN;
            let value = if column == 0 {
                (expert as f32 - 127.5) / 32.0
            } else {
                0.125
            };
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();
    let mut expected_logits = vec![0; MAX_BATCH * EXPERTS];
    let mut expected_indices = vec![0; MAX_BATCH * TOP_K];
    let mut expected_weights = vec![0; MAX_BATCH * TOP_K];

    for token in 0..MAX_BATCH {
        for expert in 0..EXPERTS {
            let mut sum = 0.0f64;
            for column in 0..HIDDEN {
                sum += f64::from(bf16_to_f32(input[token * HIDDEN + column]))
                    * f64::from(bf16_to_f32(weights[expert * HIDDEN + column]));
            }
            expected_logits[token * EXPERTS + expert] = f32_to_bf16(sum as f32);
        }

        let mut ranking = (0..EXPERTS).collect::<Vec<_>>();
        ranking.sort_unstable_by(|&left, &right| {
            bf16_to_f32(expected_logits[token * EXPERTS + right])
                .total_cmp(&bf16_to_f32(expected_logits[token * EXPERTS + left]))
                .then_with(|| left.cmp(&right))
        });
        let maximum = f64::from(bf16_to_f32(expected_logits[token * EXPERTS + ranking[0]]));
        let mut exponentials = [0.0f64; TOP_K];
        let mut denominator = 0.0f64;

        for position in 0..TOP_K {
            let expert = ranking[position];
            let value = f64::from(bf16_to_f32(expected_logits[token * EXPERTS + expert]));
            let exponential = (value - maximum).exp();
            expected_indices[token * TOP_K + position] = expert as u16;
            exponentials[position] = exponential;
            denominator += exponential;
        }
        for position in 0..TOP_K {
            expected_weights[token * TOP_K + position] =
                f32_to_bf16((exponentials[position] / denominator) as f32);
        }
    }

    Fixture {
        input,
        weights,
        expected_logits,
        expected_indices,
        expected_weights,
    }
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    observed: &Outputs,
    report: &mut Qwen36MoeRouterQualification,
) -> Result<(), Qwen36MoeRouterQualificationError> {
    let logits = batch * EXPERTS;
    if observed.logits[..logits] != fixture.expected_logits[..logits] {
        let index = observed.logits[..logits]
            .iter()
            .zip(&fixture.expected_logits[..logits])
            .position(|(actual, expected)| actual != expected)
            .expect("different slices contain one mismatch");
        return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
            "B={batch} logit {index}: device={:#06x}, oracle={:#06x}",
            observed.logits[index], fixture.expected_logits[index]
        )));
    }

    let selected = batch * TOP_K;
    if observed.indices[..selected] != fixture.expected_indices[..selected] {
        let index = observed.indices[..selected]
            .iter()
            .zip(&fixture.expected_indices[..selected])
            .position(|(actual, expected)| actual != expected)
            .expect("different slices contain one mismatch");
        return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
            "B={batch} selected expert {index}: device={}, oracle={}",
            observed.indices[index], fixture.expected_indices[index]
        )));
    }

    for index in 0..selected {
        let actual = bf16_to_f32(observed.weights[index]);
        let expected = bf16_to_f32(fixture.expected_weights[index]);
        let absolute_error = (actual - expected).abs();
        report.maximum_absolute_error = report.maximum_absolute_error.max(absolute_error);
        if absolute_error > 0.001 {
            return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
                "B={batch} routing weight {index}: device={actual}, oracle={expected}"
            )));
        }
    }

    verify_inactive(batch, observed)?;
    report.logit_values += logits;
    report.selected_experts += selected;
    report.routing_weights += selected;
    report.inactive_values += (MAX_BATCH - batch) * (EXPERTS + 2 * TOP_K);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Outputs,
    replay: &Outputs,
    report: &mut Qwen36MoeRouterQualification,
) -> Result<(), Qwen36MoeRouterQualificationError> {
    if eager.logits != replay.logits
        || eager.indices != replay.indices
        || eager.weights != replay.weights
    {
        return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * (EXPERTS + 2 * TOP_K);
    report.inactive_values += (MAX_BATCH - batch) * (EXPERTS + 2 * TOP_K);

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Outputs,
) -> Result<(), Qwen36MoeRouterQualificationError> {
    for (role, values, begin, sentinel) in [
        (
            "logits",
            observed.logits.as_slice(),
            batch * EXPERTS,
            BF16_SENTINEL,
        ),
        (
            "indices",
            observed.indices.as_slice(),
            batch * TOP_K,
            INDEX_SENTINEL,
        ),
        (
            "weights",
            observed.weights.as_slice(),
            batch * TOP_K,
            BF16_SENTINEL,
        ),
    ] {
        if let Some(relative) = values[begin..].iter().position(|&value| value != sentinel) {
            return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} value {}",
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
    report: &mut Qwen36MoeRouterQualification,
) -> Result<(), Qwen36MoeRouterQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weights = arena.copy_to_host(stream, regions.weights)?;
    if input != fixture.input || weights != fixture.weights {
        return Err(Qwen36MoeRouterQualificationError::Mismatch(
            "read-only input or router-weight plane changed".to_string(),
        ));
    }
    report.immutable_input_values = input.len() + weights.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen36MoeRouterOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen36MoeRouterQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graphs[batch - 1].launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen36MoeRouterQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_routes_both_expert_orderings_and_accounts_exactly() {
        let fixture = make_fixture();
        let (layout, regions) = layout().unwrap();

        assert_eq!(
            &fixture.expected_indices[..TOP_K],
            &[255, 254, 253, 252, 251, 250, 249, 248]
        );
        assert_eq!(
            &fixture.expected_indices[TOP_K..2 * TOP_K],
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(regions.weight_bytes(), 1_048_576);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 37_120);
        assert_eq!(layout.byte_len(), 1_085_824);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 128);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen36MoeRouterQualificationError> {
        let report = qualify_qwen36_moe_router()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.logit_values, active_rows * EXPERTS);
        assert_eq!(report.selected_experts, active_rows * TOP_K);
        assert_eq!(report.routing_weights, active_rows * TOP_K);
        assert_eq!(
            report.graph_replay_values,
            active_rows * (EXPERTS + 2 * TOP_K)
        );
        assert_eq!(
            report.inactive_values,
            2 * inactive_rows * (EXPERTS + 2 * TOP_K)
        );
        assert_eq!(report.immutable_input_values, 540_672);
        assert_eq!(report.arena_bytes, 1_085_824);
        assert_eq!(report.weight_bytes, 1_048_576);
        assert_eq!(report.workspace_bytes, 37_120);
        assert_eq!(report.padding_bytes, 128);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

//! Qwen3.8-Flash-Next represented-value qualification for the BF16 router and its
//! normalized top-10 selection.
//!
//! The oracle is structurally independent of the kernel: it accumulates every
//! router dot product in `f64`, takes the **full 512-way softmax first**, then
//! the top ten, then renormalizes. It never reproduces the kernel's shortcut,
//! so a kernel that silently reverted to Qwen3.6's "select on logits, softmax the
//! winners" form would have to agree with a differently-derived number to pass.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{bf16_to_f32, f32_to_bf16};
use crate::target::Qwen38FlashNextMoeRouterOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_ROWS: usize = 1_024;
pub(crate) const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];
const ALIGNMENT: usize = 256;
pub(crate) const HIDDEN: usize = Qwen38FlashNext::HIDDEN;
pub(crate) const EXPERTS: usize = Qwen38FlashNext::NUM_EXPERTS;
pub(crate) const TOP_K: usize = Qwen38FlashNext::NUM_EXPERTS_PER_TOKEN;
const BF16_SENTINEL: u16 = 0xa5a5;
const INDEX_SENTINEL: u16 = u16::MAX;
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25];

/// Ten experts given the highest expert's router row verbatim, so eleven
/// experts share one bit-identical top logit and only ten seats exist.
const TIE_FIRST_EXPERT: usize = 100;
/// The expert whose row the tied group copies, and the one the lowest-index
/// rule must drop.
const TIE_DONOR_EXPERT: usize = EXPERTS - 1;

/// Failure of the exact Qwen3.8-Flash-Next MoE router qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextMoeRouterQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.8-Flash-Next MoE router qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst routing-weight error from every exact route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextMoeRouterQualification {
    /// BF16 router logits compared bit-exactly.
    pub logit_values: usize,
    /// Selected expert ids compared exactly, in ascending order.
    pub selected_experts: usize,
    /// BF16 renormalized routing weights compared with the independent oracle.
    pub routing_weights: usize,
    /// Selections whose published order was verified strictly ascending.
    pub ascending_selections: usize,
    /// Tie-broken selections that matched the pinned lowest-index rule.
    pub tie_broken_selections: usize,
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
    pub(crate) expected_logits: Vec<u16>,
    pub(crate) expected_indices: Vec<u16>,
    pub(crate) expected_weights: Vec<u16>,
    /// Tokens whose selection is decided by the tie-break rule.
    pub(crate) tie_tokens: Vec<usize>,
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve::<u16>(MAX_ROWS * HIDDEN, ALIGNMENT)?;
    let weights = layout.reserve::<u16>(EXPERTS * HIDDEN, ALIGNMENT)?;
    let logits = layout.reserve::<u16>(MAX_ROWS * EXPERTS, ALIGNMENT)?;
    let expert_indices = layout.reserve::<u16>(MAX_ROWS * TOP_K, ALIGNMENT)?;
    let expert_weights = layout.reserve::<u16>(MAX_ROWS * TOP_K, ALIGNMENT)?;

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

/// The independent law: a full 512-way `f64` softmax, then the top ten, then a
/// renormalization of exactly those ten.
///
/// Ties resolve to the lowest expert index, which this target pins because
/// `torch.topk` guarantees no order of its own.
fn expected_route(logits: &[u16]) -> (Vec<u16>, Vec<u16>, bool) {
    let values = logits
        .iter()
        .map(|&bits| f64::from(bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f64>();
    let probabilities = exponentials
        .iter()
        .map(|exponential| exponential / denominator)
        .collect::<Vec<_>>();

    let mut ranking = (0..EXPERTS).collect::<Vec<_>>();
    ranking.sort_unstable_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    let selected = &ranking[..TOP_K];
    // A tie decides this token when the tenth and eleventh probabilities are
    // equal, or when any two selected probabilities are.
    let boundary_tie = probabilities[ranking[TOP_K - 1]] == probabilities[ranking[TOP_K]];
    let interior_tie = selected
        .windows(2)
        .any(|pair| probabilities[pair[0]] == probabilities[pair[1]]);

    let selected_sum = selected
        .iter()
        .map(|&expert| probabilities[expert])
        .sum::<f64>();
    let mut ascending = selected.to_vec();
    ascending.sort_unstable();
    let indices = ascending.iter().map(|&expert| expert as u16).collect();
    let weights = ascending
        .iter()
        .map(|&expert| f32_to_bf16((probabilities[expert] / selected_sum) as f32))
        .collect();

    (indices, weights, boundary_tie || interior_tie)
}

pub(crate) fn make_fixture() -> Fixture {
    let input = (0..MAX_ROWS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let column = index % HIDDEN;
            let value = if column == 0 {
                TOKEN_FACTORS[token & (MAX_BATCH - 1)]
            } else if column & 1 == 0 {
                0.5
            } else {
                -0.5
            };
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();
    let mut weights = (0..EXPERTS * HIDDEN)
        .map(|index| {
            let expert = index / HIDDEN;
            let column = index % HIDDEN;
            // `(e - 256) / 64` is a multiple of 2^-6 in [-4, 4), so every
            // value is exactly representable in BF16 and no two experts round
            // together. The earlier `- 255.5` form landed on round-half-to-even
            // boundaries and tied adjacent experts by accident, which would
            // have made the deliberate tie below indistinguishable from noise.
            let value = if column == 0 {
                (expert as f32 - 256.0) / 64.0
            } else {
                0.125
            };
            f32_to_bf16(value)
        })
        .collect::<Vec<_>>();

    // The tie case, built into the router matrix so the device reaches it by
    // arithmetic rather than by a doctored expectation. Ten experts are given
    // the *highest* expert's row verbatim, so eleven experts carry one
    // bit-identical logit -- the eleven rows are equal word for word, so the
    // dot products agree exactly rather than approximately -- and only ten
    // seats exist. That makes it simultaneously the interior sweep (all ten
    // selected probabilities equal) and the boundary case (the eleventh is
    // dropped), which is the only place a tie can change the routed set.
    //
    // The rule under test is lowest-index-wins, so the selection must be
    // exactly `TIE_EXPERTS` and must exclude `TIE_DONOR_EXPERT`. It only binds
    // for tokens whose factor is positive; the negative-factor tokens invert
    // the ordering and route elsewhere, which `tie_tokens` records rather than
    // assumes.
    let (donor, rest) = weights.split_at_mut(TIE_DONOR_EXPERT * HIDDEN);
    let donor_row = &rest[..HIDDEN];
    for offset in 0..TOP_K {
        let expert = TIE_FIRST_EXPERT + offset;
        donor[expert * HIDDEN..(expert + 1) * HIDDEN].copy_from_slice(donor_row);
    }

    let mut expected_logits = vec![0u16; MAX_ROWS * EXPERTS];
    for token in 0..MAX_ROWS {
        for expert in 0..EXPERTS {
            let mut sum = 0.0f64;
            for column in 0..HIDDEN {
                sum += f64::from(bf16_to_f32(input[token * HIDDEN + column]))
                    * f64::from(bf16_to_f32(weights[expert * HIDDEN + column]));
            }
            expected_logits[token * EXPERTS + expert] = f32_to_bf16(sum as f32);
        }
    }

    let mut expected_indices = vec![0u16; MAX_ROWS * TOP_K];
    let mut expected_weights = vec![0u16; MAX_ROWS * TOP_K];
    let mut tie_tokens = Vec::new();
    for token in 0..MAX_ROWS {
        let (indices, weights, tied) =
            expected_route(&expected_logits[token * EXPERTS..(token + 1) * EXPERTS]);
        expected_indices[token * TOP_K..(token + 1) * TOP_K].copy_from_slice(&indices);
        expected_weights[token * TOP_K..(token + 1) * TOP_K].copy_from_slice(&weights);
        if tied {
            tie_tokens.push(token);
        }
    }

    Fixture {
        input,
        weights,
        expected_logits,
        expected_indices,
        expected_weights,
        tie_tokens,
    }
}

pub(crate) fn launch(
    op: &Qwen38FlashNextMoeRouterOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    rows: usize,
) -> GpuResult<()> {
    unsafe {
        op.launch(
            stream,
            rows,
            arena.address(regions.input)?,
            arena.address(regions.weights)?,
            arena.address(regions.logits)?,
            arena.address(regions.expert_indices)?,
            arena.address(regions.expert_weights)?,
        )
    }
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

/// Qualifies eager and captured Qwen3.8-Flash-Next router execution at every exact route.
pub fn qualify_qwen38_flash_next_moe_router()
-> Result<Qwen38FlashNextMoeRouterQualification, Qwen38FlashNextMoeRouterQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let operator = Qwen38FlashNextMoeRouterOp::new(&context)?;
    let fixture = make_fixture();

    arena.copy_from_host(&stream, regions.input, &fixture.input)?;
    arena.copy_from_host(&stream, regions.weights, &fixture.weights)?;
    stream.synchronize().map_err(GpuError::from)?;

    let baseline_addresses = addresses(&arena, regions)?;

    let mut report = Qwen38FlashNextMoeRouterQualification {
        logit_values: 0,
        selected_experts: 0,
        routing_weights: 0,
        ascending_selections: 0,
        tie_broken_selections: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_input_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        fill_outputs(&arena, &stream, regions)?;
        launch(&operator, &arena, &stream, regions, rows)?;
        stream.synchronize().map_err(GpuError::from)?;
        let eager = read_outputs(&arena, &stream, regions)?;
        verify_route(rows, &fixture, &eager, &mut report)?;

        // Eager and replay must agree over every observable boundary.
        fill_outputs(&arena, &stream, regions)?;
        let graph = CudaGraph::capture(&stream, || {
            launch(&operator, &arena, &stream, regions, rows)
        })?;
        unsafe { graph.launch(&stream) }?;
        stream.synchronize().map_err(GpuError::from)?;
        let replayed = read_outputs(&arena, &stream, regions)?;

        if replayed.logits != eager.logits
            || replayed.indices != eager.indices
            || replayed.weights != eager.weights
        {
            return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                format!("graph replay diverged from eager execution at rows={rows}"),
            ));
        }
        report.graph_replay_values += rows * (EXPERTS + 2 * TOP_K);

        // Every value past the active extent kept its sentinel.
        for token in rows..MAX_ROWS {
            for expert in 0..EXPERTS {
                if eager.logits[token * EXPERTS + expert] != BF16_SENTINEL {
                    return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                        format!("rows={rows} wrote logit token {token} expert {expert}"),
                    ));
                }
                report.inactive_values += 1;
            }
            for position in 0..TOP_K {
                if eager.indices[token * TOP_K + position] != INDEX_SENTINEL
                    || eager.weights[token * TOP_K + position] != BF16_SENTINEL
                {
                    return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                        format!("rows={rows} wrote selection token {token} position {position}"),
                    ));
                }
                report.inactive_values += 2;
            }
        }

        // The read-only planes are unchanged.
        let observed_input = arena.copy_to_host(&stream, regions.input)?;
        let observed_weights = arena.copy_to_host(&stream, regions.weights)?;
        if observed_input != fixture.input || observed_weights != fixture.weights {
            return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                format!("rows={rows} modified a read-only plane"),
            ));
        }
        report.immutable_input_values += observed_input.len() + observed_weights.len();

        if addresses(&arena, regions)? != baseline_addresses {
            return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                format!("rows={rows} moved an address-stable region"),
            ));
        }
    }

    verify_no_post_warmup_allocation(&context, &operator, &arena, &stream, regions)?;

    Ok(report)
}

/// Graph replay after warmup must not allocate.
///
/// Measured around *replays only*: capturing and instantiating a graph
/// legitimately allocates, so a span that included the captures would report a
/// leak that is really the harness building its own fixtures.
fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen38FlashNextMoeRouterOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen38FlashNextMoeRouterQualificationError> {
    let graphs = EXACT_ROUTES
        .iter()
        .map(|&rows| CudaGraph::capture(stream, || launch(op, arena, stream, regions, rows)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for graph in graphs.iter().rev() {
            // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
            unsafe { graph.launch(stream) }?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
            format!("device memory changed after warmup: before={before:?}, after={after:?}"),
        ));
    }

    Ok(())
}

fn verify_route(
    rows: usize,
    fixture: &Fixture,
    observed: &Outputs,
    report: &mut Qwen38FlashNextMoeRouterQualification,
) -> Result<(), Qwen38FlashNextMoeRouterQualificationError> {
    for token in 0..rows {
        for expert in 0..EXPERTS {
            let index = token * EXPERTS + expert;
            if observed.logits[index] != fixture.expected_logits[index] {
                return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                    format!(
                        "rows={rows} token {token} expert {expert} logit {:#06x} != {:#06x}",
                        observed.logits[index], fixture.expected_logits[index]
                    ),
                ));
            }
            report.logit_values += 1;
        }

        let selection = &observed.indices[token * TOP_K..(token + 1) * TOP_K];
        let expected = &fixture.expected_indices[token * TOP_K..(token + 1) * TOP_K];
        if selection != expected {
            return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                format!(
                    "rows={rows} token {token} selected {selection:?}, oracle says {expected:?}"
                ),
            ));
        }
        report.selected_experts += TOP_K;

        // Published order defines the downstream accumulation order, so it is
        // asserted rather than assumed.
        if !selection.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                format!(
                    "rows={rows} token {token} published {selection:?}, which is not ascending"
                ),
            ));
        }
        report.ascending_selections += 1;
        if fixture.tie_tokens.contains(&token) {
            report.tie_broken_selections += 1;
        }

        for position in 0..TOP_K {
            let index = token * TOP_K + position;
            let observed_weight = bf16_to_f32(observed.weights[index]);
            let expected_weight = bf16_to_f32(fixture.expected_weights[index]);
            let error = (observed_weight - expected_weight).abs();
            if error > weight_tolerance(expected_weight) {
                return Err(Qwen38FlashNextMoeRouterQualificationError::Mismatch(
                    format!(
                        "rows={rows} token {token} position {position} weight {observed_weight} != \
                     {expected_weight}"
                    ),
                ));
            }
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            report.routing_weights += 1;
        }
    }

    Ok(())
}

/// One BF16 ulp at the observed magnitude, widened by the `ex2.approx.f32`
/// contract the softmax exponential is computed with. A routing weight lies in
/// `(0, 1]`, so the floor covers the smallest representable normal step.
fn weight_tolerance(expected: f32) -> f32 {
    (expected.abs() * 0.01).max(1.0e-3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_takes_the_full_softmax_before_selecting() {
        // Two logits equal to the maximum and 510 far below it: if the oracle
        // renormalized without the 512-way denominator it would still get this
        // right, so the discriminating check is that the ten weights sum to one
        // while the underlying probabilities do not.
        let mut logits = vec![f32_to_bf16(-8.0); EXPERTS];
        for (offset, expert) in (0..TOP_K).enumerate() {
            logits[expert] = f32_to_bf16(offset as f32 * 0.5);
        }
        let (indices, weights, tied) = expected_route(&logits);

        assert!(!tied);
        assert_eq!(indices, (0..TOP_K as u16).collect::<Vec<_>>());
        let total = weights.iter().map(|&bits| bf16_to_f32(bits)).sum::<f32>();
        assert!((total - 1.0).abs() < 1.0e-2, "weights summed to {total}");
    }

    #[test]
    fn oracle_breaks_ties_toward_the_lowest_expert_index() {
        // Eleven experts share the top probability; exactly the ten lowest
        // indices must be selected, and the eleventh must be dropped.
        let mut logits = vec![f32_to_bf16(-8.0); EXPERTS];
        for expert in 40..=50 {
            logits[expert] = f32_to_bf16(1.0);
        }
        let (indices, _, tied) = expected_route(&logits);

        assert!(tied, "a boundary tie must be reported");
        assert_eq!(indices, (40..50).collect::<Vec<_>>());
    }

    #[test]
    fn fixture_pins_the_tie_case_and_accounts_exactly() {
        let fixture = make_fixture();
        let (layout, regions) = layout().unwrap();

        // Token 0 carries the +1.0 factor, so its eleven tied experts hold the
        // row maximum: the ten lowest ids must take every seat and the donor
        // must be dropped. That is the whole tie-break rule in one assertion.
        let tied = (TIE_FIRST_EXPERT as u16..(TIE_FIRST_EXPERT + TOP_K) as u16).collect::<Vec<_>>();
        assert_eq!(&fixture.expected_indices[..TOP_K], tied.as_slice());
        assert!(!fixture.expected_indices[..TOP_K].contains(&(TIE_DONOR_EXPERT as u16)));
        assert!(fixture.tie_tokens.contains(&0));

        // The negative-factor tokens invert the ordering and route elsewhere,
        // so the fixture carries an untied selection beside the tied one.
        assert!(!fixture.tie_tokens.contains(&1));
        assert_eq!(
            &fixture.expected_indices[TOP_K..2 * TOP_K],
            (0..TOP_K as u16).collect::<Vec<_>>().as_slice()
        );

        // Every published selection is ascending by construction.
        for token in 0..MAX_ROWS {
            let selection = &fixture.expected_indices[token * TOP_K..(token + 1) * TOP_K];
            assert!(
                selection.windows(2).all(|pair| pair[0] < pair[1]),
                "token {token} published {selection:?}"
            );
        }

        assert_eq!(regions.weight_bytes(), 2_621_440);
        assert_eq!(layout.byte_len(), regions.payload_bytes());
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_routes_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen38FlashNextMoeRouterQualificationError> {
        let report = qualify_qwen38_flash_next_moe_router()?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        let inactive_rows = EXACT_ROUTES
            .iter()
            .map(|rows| MAX_ROWS - rows)
            .sum::<usize>();

        assert_eq!(report.logit_values, active_rows * EXPERTS);
        assert_eq!(report.selected_experts, active_rows * TOP_K);
        assert_eq!(report.routing_weights, active_rows * TOP_K);
        assert_eq!(report.ascending_selections, active_rows);
        assert_eq!(
            report.graph_replay_values,
            active_rows * (EXPERTS + 2 * TOP_K)
        );
        assert_eq!(
            report.inactive_values,
            inactive_rows * (EXPERTS + 2 * TOP_K)
        );
        assert_eq!(report.padding_bytes, 0);
        // Both tie tokens are inside every route from B=3 upward.
        assert!(report.tie_broken_selections >= EXACT_ROUTES.len() - 2);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

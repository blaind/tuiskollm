//! Qualification for the exact Qwen3.8-Flash-Next BF16 language-model head.
//!
//! `lm_head: Linear(2560 -> 248320, bias=False)`, untied, reading the collapsed
//! stream the hyper-connection mixer publishes. There is no final RMSNorm in this
//! target, so the head's input is the mixer's output unmodified and the head is
//! the plain projection the backbone shapes are, at the vocabulary width.
//!
//! Decode routes only: the reference reads logits for the rows it samples, and
//! a prompt-tile route would publish 496,640 B for every row of a tile to keep
//! one of them.

use crate::device_benchmark;
use crate::fp8_projection_oracle::{BF16_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::qwen38_flash_next_projection::{oracle_row, output_tolerance, probe_row, probe_value};
use crate::target::Qwen38FlashNextBf16LmHeadOp;
use crate::{DeviceBenchmarkError, harness::immutable_sentinel::first_difference};
use std::sync::Arc;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen38FlashNext};

/// The head's admitted decode routes.
pub(crate) const EXACT_ROUTES: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// Largest admitted decode batch, and the row capacity every plane is sized for.
pub(crate) const MAX_BATCH: usize = 8;
/// Collapsed stream width the head contracts over.
pub(crate) const HIDDEN: usize = <Qwen38FlashNext as Arch>::HIDDEN;
/// Untied vocabulary rows the head publishes.
pub(crate) const VOCAB: usize = <Qwen38FlashNext as Arch>::VOCAB;

const ALIGNMENT: usize = 256;
/// Exactly representable BF16 values, so every fixture product is exact in FP32.
const PATTERN: [f32; 16] = [
    0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125, -0.0625, -0.875, 0.75, -0.625, 0.5, -0.375,
    0.25, -0.125, 0.0625,
];
/// BF16 word `1.0078125`: the nearest-rounding of the probe's exact dot product.
const PROBE_NEAREST: u16 = 0x3f81;

/// Failure of the Qwen3.8-Flash-Next BF16 LM-head qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextLmHeadQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behaviour disagreed with the independent projection law.
    #[error("Qwen3.8-Flash-Next BF16 LM-head qualification failed: {0}")]
    Mismatch(String),
}

type QualificationResult<T> = Result<T, Qwen38FlashNextLmHeadQualificationError>;

/// Observable counts, ownership, and worst oracle error.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Qwen38FlashNextLmHeadQualification {
    /// Active BF16 logits checked against their exact single-row reference.
    pub logit_values: usize,
    /// Active BF16 logits checked against the independent `f64` projection law.
    pub oracle_values: usize,
    /// Active BF16 logits reproduced bit-exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact batch's extent.
    pub inactive_values: usize,
    /// Read-only input and weight words proved unchanged.
    pub immutable_values: usize,
    /// Probe logits whose word separates nearest rounding from truncation.
    pub rne_separated_values: usize,
    /// Untied source-BF16 vocabulary plane bytes.
    pub weight_bytes: usize,
    /// Address-stable stream and logit bytes.
    pub workspace_bytes: usize,
    /// One-allocation arena bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Largest absolute difference from the `f64` projection law.
    pub maximum_absolute_error: f32,
}

/// The head's planes, in launch order.
#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) weight: ArenaRegion<u16>,
    pub(crate) logits: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight.byte_len()
    }

    pub(crate) fn workspace_bytes(self) -> usize {
        self.input.byte_len() + self.logits.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.weight_bytes() + self.workspace_bytes()
    }
}

/// The head's host-side stream and vocabulary planes.
pub(crate) struct Fixture {
    pub(crate) input: Vec<u16>,
    pub(crate) replacement_input: Vec<u16>,
    pub(crate) weight: Vec<u16>,
}

/// A non-periodic mix of two indices, so distinct vocabulary rows carry
/// distinct logits and a per-row reference comparison is load-bearing.
fn mix(first: usize, second: usize) -> usize {
    let seed = first
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(second.wrapping_mul(0x85EB_CA77));

    (seed ^ (seed >> 15)).wrapping_mul(0xC2B2_AE35) >> 16
}

pub(crate) fn make_input(salt: usize) -> Vec<u16> {
    (0..MAX_BATCH * HIDDEN)
        .map(|index| {
            let row = index / HIDDEN;
            let column = index - row * HIDDEN;
            f32_to_bf16(PATTERN[mix(row + salt * 0x2545, column + salt) & 15])
        })
        .collect()
}

pub(crate) fn make_fixture() -> Fixture {
    Fixture {
        input: make_input(0),
        replacement_input: make_input(1),
        weight: (0..VOCAB * HIDDEN)
            .map(|index| {
                let row = index / HIDDEN;
                let column = index - row * HIDDEN;
                f32_to_bf16(PATTERN[mix(row + 0x51ED, column) & 15])
            })
            .collect(),
    }
}

pub(crate) fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;
    let weight = layout.reserve(VOCAB * HIDDEN, ALIGNMENT)?;
    let logits = layout.reserve(MAX_BATCH * VOCAB, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight,
            logits,
        },
    ))
}

pub(crate) fn launch(
    op: &Qwen38FlashNextBf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    batch: usize,
) -> GpuResult<()> {
    // SAFETY: aligned, non-overlapping regions cover every admitted batch.
    unsafe {
        op.launch(
            stream,
            batch,
            arena.address(regions.input)?,
            arena.address(regions.weight)?,
            arena.address(regions.logits)?,
        )
    }
}

fn launch_single_row(
    op: &Qwen38FlashNextBf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    row: usize,
) -> GpuResult<()> {
    // SAFETY: `row` stays inside the reserved batch capacity, and each offset
    // selects one complete row of its own plane.
    unsafe {
        op.launch(
            stream,
            1,
            arena.address(regions.input)?.add(row * HIDDEN),
            arena.address(regions.weight)?,
            arena.address(regions.logits)?.add(row * VOCAB),
        )
    }
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 3]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weight)?.addr(),
        arena.address(regions.logits)?.addr(),
    ])
}

/// The independent projection law over the vocabulary plane.
pub(crate) fn oracle(input: &[u16], weight: &[u16]) -> Vec<f64> {
    weight
        .chunks_exact(HIDDEN)
        .map(|row| oracle_row(input, row))
        .collect()
}

/// Qualifies every admitted decode route of the untied BF16 head.
pub fn qualify_qwen38_flash_next_lm_head() -> QualificationResult<Qwen38FlashNextLmHeadQualification>
{
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen38FlashNextBf16LmHeadOp::new(&context)?;
    let mut report = Qwen38FlashNextLmHeadQualification {
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.workspace_bytes(),
        arena_bytes: layout.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        ..Qwen38FlashNextLmHeadQualification::default()
    };

    verify_rounding_site(&op, &arena, &stream, regions, &mut report)?;

    let fixture = make_fixture();
    arena.copy_from_host(&stream, regions.weight, &fixture.weight)?;
    arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
    let stable_addresses = addresses(&arena, regions)?;
    let single_row_reference = single_row_references(&op, &arena, &stream, regions)?;
    let expected = oracle(&fixture.replacement_input[..HIDDEN], &fixture.weight);

    for &batch in &EXACT_ROUTES {
        arena.copy_from_host(&stream, regions.input, &fixture.input)?;
        arena.fill(&stream, regions.logits, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let first = arena.copy_to_host(&stream, regions.logits)?;

        arena.copy_from_host(&stream, regions.input, &fixture.replacement_input)?;
        arena.fill(&stream, regions.logits, BF16_SENTINEL as u8)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: every allocation this graph captured is owned by this scope
        // and outlives both replays and the copy that follows them.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: as above; a second replay proves the graph is re-runnable.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.logits)?;

        arena.fill(&stream, regions.logits, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.logits)?;

        verify_rows(
            batch,
            &single_row_reference,
            &expected,
            &replay,
            &mut report,
        )?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement(batch, &first, &replay)?;
        verify_inactive(batch, &eager, &mut report)?;
        verify_inactive(batch, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &fixture, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

/// Proves the head's store rounds to nearest rather than truncating.
///
/// The probe is the backbone family's: one activation row and one vocabulary
/// row carrying `[1, 2^-4, 2^-8, 0, ...]`, whose exact dot product lies strictly
/// above the BF16 midpoint between `1.0` and `1.0078125`.
fn verify_rounding_site(
    op: &Qwen38FlashNextBf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    report: &mut Qwen38FlashNextLmHeadQualification,
) -> QualificationResult<()> {
    let mut weight = vec![0u16; VOCAB * HIDDEN];
    weight[..HIDDEN].copy_from_slice(&probe_row(HIDDEN));
    arena.copy_from_host(stream, regions.weight, &weight)?;
    drop(weight);

    let mut input = vec![0u16; MAX_BATCH * HIDDEN];
    for row in 0..MAX_BATCH {
        input[row * HIDDEN..(row + 1) * HIDDEN].copy_from_slice(&probe_row(HIDDEN));
    }
    arena.copy_from_host(stream, regions.input, &input)?;

    arena.fill(stream, regions.logits, BF16_SENTINEL as u8)?;
    launch(op, arena, stream, regions, MAX_BATCH)?;
    let observed = arena.copy_to_host(stream, regions.logits)?;
    for row in 0..MAX_BATCH {
        let word = observed[row * VOCAB];
        if word != PROBE_NEAREST {
            return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
                "B={MAX_BATCH} row={row} store published {word:#06x}, expected \
                 {PROBE_NEAREST:#06x} for the exact value {}",
                probe_value()
            )));
        }
        report.rne_separated_values += 1;
    }

    Ok(())
}

fn single_row_references(
    op: &Qwen38FlashNextBf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> GpuResult<Vec<u16>> {
    arena.fill(stream, regions.logits, BF16_SENTINEL as u8)?;
    for row in 0..MAX_BATCH {
        launch_single_row(op, arena, stream, regions, row)?;
    }
    arena.copy_to_host(stream, regions.logits)
}

fn verify_rows(
    batch: usize,
    single_row_reference: &[u16],
    expected: &[f64],
    observed: &[u16],
    report: &mut Qwen38FlashNextLmHeadQualification,
) -> QualificationResult<()> {
    for row in 0..batch {
        let begin = row * VOCAB;
        let end = begin + VOCAB;
        if let Some(index) =
            first_difference(&observed[begin..end], &single_row_reference[begin..end])
        {
            return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
                "B={batch}, row={row} differs from its single-row projection at logit {index}"
            )));
        }
    }

    if batch == 1 {
        for (logit, (&actual, &expected)) in observed[..VOCAB].iter().zip(expected).enumerate() {
            let actual = bf16_to_f32(actual);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_absolute_error = report.maximum_absolute_error.max(error);
            let tolerance = output_tolerance(expected);
            if !actual.is_finite() || error > tolerance {
                return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
                    "logit={logit}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        report.oracle_values += VOCAB;
    }
    report.logit_values += batch * VOCAB;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Qwen38FlashNextLmHeadQualification,
) -> QualificationResult<()> {
    let active = batch * VOCAB;
    if let Some(index) = first_difference(&eager[..active], &replay[..active]) {
        return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
            "B={batch} eager and graph logits differ at value {index}"
        )));
    }
    report.graph_replay_values += active;

    Ok(())
}

fn verify_replacement(batch: usize, first: &[u16], replacement: &[u16]) -> QualificationResult<()> {
    let active = batch * VOCAB;
    if first[..active] == replacement[..active] {
        return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe the replacement stream"
        )));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &[u16],
    report: &mut Qwen38FlashNextLmHeadQualification,
) -> QualificationResult<()> {
    let begin = batch * VOCAB;
    if let Some(index) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
            "B={batch} modified inactive logit {index}"
        )));
    }
    report.inactive_values += (MAX_BATCH - batch) * VOCAB;

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen38FlashNextLmHeadQualification,
) -> QualificationResult<()> {
    for (role, actual, expected) in [
        (
            "stream input",
            arena.copy_to_host(stream, regions.input)?,
            &fixture.replacement_input,
        ),
        (
            "vocabulary weights",
            arena.copy_to_host(stream, regions.weight)?,
            &fixture.weight,
        ),
    ] {
        if let Some(index) = first_difference(&actual, expected) {
            return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
                "read-only {role} value {index} changed"
            )));
        }
        report.immutable_values += actual.len();
    }

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &Arc<CudaContext>,
    op: &Qwen38FlashNextBf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> QualificationResult<()> {
    launch(op, arena, stream, regions, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        for &batch in &EXACT_ROUTES {
            launch(op, arena, stream, regions, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen38FlashNextLmHeadQualificationError::Mismatch(format!(
            "post-warmup launches changed device memory from {before:?} to {after:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, HIDDEN, MAX_BATCH, PROBE_NEAREST, VOCAB, layout, make_input, oracle,
        qualify_qwen38_flash_next_lm_head,
    };
    use crate::qwen38_flash_next_projection::probe_row;
    use tuisko_model::{Arch, Qwen35_9B, Qwen38FlashNext};

    #[test]
    fn qwen38_flash_next_lm_head_suite_route_and_byte_inventory_is_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(MAX_BATCH, 8);
        assert_eq!(HIDDEN, 2_560);
        assert_eq!(VOCAB, 248_320);
        assert_eq!(regions.weight_bytes(), 1_271_398_400);
        assert_eq!(regions.workspace_bytes(), 4_014_080);
        assert_eq!(regions.payload_bytes(), 1_275_412_480);
        // Every reserved byte is a payload byte.
        assert_eq!(layout.byte_len(), regions.payload_bytes());
    }

    /// Decode-only is a contract, not an omission: a prompt tile would publish
    /// half a gigabyte of logits to keep one row's worth.
    #[test]
    fn qwen38_flash_next_lm_head_suite_admits_decode_rows_only() {
        assert_eq!(EXACT_ROUTES.len(), MAX_BATCH);
        assert_eq!(EXACT_ROUTES.iter().max(), Some(&MAX_BATCH));
        assert_eq!(VOCAB * size_of::<u16>(), 496_640);
        assert_eq!(1_024 * VOCAB * size_of::<u16>(), 508_559_360);
    }

    /// The head shares a vocabulary with the other BF16 head and nothing else,
    /// which is why this target needs its own entries.
    #[test]
    fn qwen38_flash_next_lm_head_suite_differs_from_the_other_bf16_head_in_its_stream_width() {
        assert_eq!(<Qwen38FlashNext as Arch>::VOCAB, <Qwen35_9B as Arch>::VOCAB);
        assert_ne!(HIDDEN, <Qwen35_9B as Arch>::HIDDEN);
    }

    /// The probe row selects the nearest word, and distinct vocabulary rows
    /// carry distinct logits so the per-row reference is load-bearing.
    #[test]
    fn qwen38_flash_next_lm_head_suite_the_probe_and_the_fixture_are_decisive() {
        let probe = probe_row(HIDDEN);
        assert_eq!(probe.len(), HIDDEN);
        assert_eq!(oracle(&probe, &probe), vec![super::probe_value()]);
        assert_eq!(PROBE_NEAREST, 0x3f81);

        let input = make_input(0);
        let replacement = make_input(1);
        assert_eq!(input.len(), MAX_BATCH * HIDDEN);
        for row in 0..MAX_BATCH {
            let begin = row * HIDDEN;
            assert_ne!(
                input[begin..begin + HIDDEN],
                replacement[begin..begin + HIDDEN]
            );
        }
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen38_flash_next_lm_head_suite_routes_match_independent_oracles_and_graph_replay() {
        let report = qualify_qwen38_flash_next_lm_head()
            .expect("Qwen3.8-Flash-Next BF16 LM-head qualification");
        let active = EXACT_ROUTES.iter().sum::<usize>();
        let inactive = EXACT_ROUTES
            .iter()
            .map(|&batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.logit_values, active * VOCAB);
        assert_eq!(report.oracle_values, VOCAB);
        assert_eq!(report.graph_replay_values, active * VOCAB);
        assert_eq!(report.inactive_values, 2 * inactive * VOCAB);
        assert_eq!(report.rne_separated_values, MAX_BATCH);
        assert_eq!(report.weight_bytes, 1_271_398_400);
        assert_eq!(report.workspace_bytes, 4_014_080);
        assert_eq!(report.arena_bytes, 1_275_412_480);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.immutable_values > 0);
        assert!(report.maximum_absolute_error.is_finite());
    }
}

//! Represented-value qualification for the Qwen3.6 NVFP4 LM head.

use crate::device_benchmark;
use crate::nvfp4_down_sm120::{bf16_to_f32, decode_e2m1, decode_e4m3fn, f32_to_bf16};
use crate::target::Qwen36Nvfp4LmHeadOp;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{Arch, Qwen36Moe35B};

pub(crate) const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
pub(crate) const INPUT_COLUMNS: usize = Qwen36Moe35B::HIDDEN;
pub(crate) const OUTPUT_ROWS: usize = Qwen36Moe35B::VOCAB;
const GROUP: usize = 16;
pub(crate) const GROUPS_PER_ROW: usize = INPUT_COLUMNS / GROUP;
pub(crate) const CODE_BYTES_PER_ROW: usize = INPUT_COLUMNS / 2;
pub(crate) const WEIGHT_SCALE_2: f32 = 0.25;
const BF16_SENTINEL: u16 = 0xa5a5;
const ORACLE_ROWS: usize = 128;
const ORACLE_BOUNDARIES: [usize; 8] = [0, 31, 32, 127, 128, 129, OUTPUT_ROWS - 2, OUTPUT_ROWS - 1];
const INPUT_PATTERN: [f32; GROUP] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.5,
    -0.5,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, 1.0, 0.5, 0.25, 0.125];

/// Failure of the exact Qwen3.6 NVFP4 LM-head qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36Nvfp4LmHeadQualificationError {
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] crate::DeviceBenchmarkError),
    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.6 NVFP4 LM-head qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36Nvfp4LmHeadQualification {
    /// Representative BF16 logits compared with the independent full dot.
    pub sampled_logits: usize,
    /// All active logits checked as finite and published.
    pub published_logits: usize,
    /// Active BF16 logits reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each active route extent.
    pub inactive_values: usize,
    /// Read-only input and represented-weight values proved unchanged.
    pub immutable_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact packed-weight and block-scale bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from the independent oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Regions {
    pub(crate) input: ArenaRegion<u16>,
    pub(crate) weight_codes: ArenaRegion<u8>,
    pub(crate) weight_scales: ArenaRegion<u8>,
    pub(crate) output: ArenaRegion<u16>,
}

impl Regions {
    pub(crate) fn weight_bytes(self) -> usize {
        self.weight_codes.byte_len() + self.weight_scales.byte_len()
    }

    pub(crate) fn payload_bytes(self) -> usize {
        self.input.byte_len() + self.weight_bytes() + self.output.byte_len()
    }
}

pub(crate) struct Fixture {
    pub(crate) input_bf16: Vec<u16>,
    input_f32: Vec<f32>,
    pub(crate) weight_codes: Vec<u8>,
    pub(crate) weight_scales: Vec<u8>,
}

/// Qualifies eager and captured Qwen3.6 A16 LM-head routes at `B=1..=8`.
pub fn qualify_qwen36_nvfp4_lm_head()
-> Result<Qwen36Nvfp4LmHeadQualification, Qwen36Nvfp4LmHeadQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }

    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen36Nvfp4LmHeadOp::new(&context)?;
    let fixture = make_fixture();
    arena.copy_from_host(&stream, regions.input, &fixture.input_bf16)?;
    arena.copy_from_host(&stream, regions.weight_codes, &fixture.weight_codes)?;
    arena.copy_from_host(&stream, regions.weight_scales, &fixture.weight_scales)?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen36Nvfp4LmHeadQualification {
        sampled_logits: 0,
        published_logits: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weight_bytes(),
        workspace_bytes: regions.payload_bytes() - regions.weight_bytes(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_eager(batch, &fixture, &eager, &mut report)?;

        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        stream.synchronize().map_err(GpuError::from)?;
        let graph = CudaGraph::capture(&stream, || launch(&op, &arena, &stream, regions, batch))?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(&stream) }?;
        let replay = arena.copy_to_host(&stream, regions.output)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        if addresses(&arena, regions)? != stable_addresses {
            return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
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
    let input = layout.reserve(MAX_BATCH * INPUT_COLUMNS, ALIGNMENT)?;
    let weight_codes = layout.reserve(OUTPUT_ROWS * CODE_BYTES_PER_ROW, ALIGNMENT)?;
    let weight_scales = layout.reserve(OUTPUT_ROWS * GROUPS_PER_ROW, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * OUTPUT_ROWS, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weight_codes,
            weight_scales,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 4]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weight_codes)?.addr(),
        arena.address(regions.weight_scales)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

pub(crate) fn launch(
    op: &Qwen36Nvfp4LmHeadOp,
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
            arena.address(regions.weight_codes)?,
            arena.address(regions.weight_scales)?,
            WEIGHT_SCALE_2,
            arena.address(regions.output)?,
        )
    }
}

pub(crate) fn make_fixture() -> Fixture {
    let input_bf16 = (0..MAX_BATCH * INPUT_COLUMNS)
        .map(|index| {
            let token = index / INPUT_COLUMNS;
            f32_to_bf16(INPUT_PATTERN[index & (GROUP - 1)] * TOKEN_FACTORS[token])
        })
        .collect::<Vec<_>>();
    let input_f32 = input_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let (weight_codes, weight_scales) = make_weights();

    Fixture {
        input_bf16,
        input_f32,
        weight_codes,
        weight_scales,
    }
}

fn make_weights() -> (Vec<u8>, Vec<u8>) {
    const SCALE_CODES: [u8; 4] = [0x38, 0x30, 0x28, 0x20];
    let mut codes = vec![0u8; OUTPUT_ROWS * CODE_BYTES_PER_ROW];
    let mut scales = vec![0u8; OUTPUT_ROWS * GROUPS_PER_ROW];

    for row in 0..OUTPUT_ROWS {
        let base_code = if row & 1 == 0 { 0x21 } else { 0xa9 };
        let exceptional = exceptional_group(row);
        let row_begin = row * CODE_BYTES_PER_ROW;
        codes[row_begin..row_begin + CODE_BYTES_PER_ROW].fill(base_code);
        codes[row_begin + exceptional * (GROUP / 2)..row_begin + (exceptional + 1) * (GROUP / 2)]
            .fill(if row & 1 == 0 { 0x10 } else { 0x98 });
        for group in 0..GROUPS_PER_ROW {
            let scale_index = if group == exceptional {
                (row + 1) & 3
            } else {
                row & 3
            };
            scales[scale_offset(row, group)] = SCALE_CODES[scale_index];
        }
    }

    (codes, scales)
}

fn verify_eager(
    batch: usize,
    fixture: &Fixture,
    observed: &[u16],
    report: &mut Qwen36Nvfp4LmHeadQualification,
) -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
    for token in 0..batch {
        let token_output = &observed[token * OUTPUT_ROWS..(token + 1) * OUTPUT_ROWS];
        if let Some(row) = token_output
            .iter()
            .position(|word| *word == BF16_SENTINEL || !bf16_is_finite(*word))
        {
            return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
                "B={batch} output token={token}, row={row} was not published as a finite value"
            )));
        }
        for row in sampled_rows() {
            let expected = dot_oracle(token, row, fixture)?;
            let actual = f64::from(bf16_to_f32(token_output[row]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.25f64.max(expected.abs() * 0.025);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
                    "B={batch} logit token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
            report.sampled_logits += 1;
        }
        report.published_logits += OUTPUT_ROWS;
    }

    verify_inactive(batch, observed)?;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Qwen36Nvfp4LmHeadQualification,
) -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
    if eager != replay {
        return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * OUTPUT_ROWS;
    report.inactive_values += (MAX_BATCH - batch) * OUTPUT_ROWS;

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &[u16],
) -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
    let begin = batch * OUTPUT_ROWS;
    if let Some(relative) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
            "B={batch} modified inactive output {}",
            begin + relative
        )));
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    fixture: &Fixture,
    report: &mut Qwen36Nvfp4LmHeadQualification,
) -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
    let input = arena.copy_to_host(stream, regions.input)?;
    let weight_codes = arena.copy_to_host(stream, regions.weight_codes)?;
    let weight_scales = arena.copy_to_host(stream, regions.weight_scales)?;
    if input != fixture.input_bf16
        || weight_codes != fixture.weight_codes
        || weight_scales != fixture.weight_scales
    {
        return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(
            "read-only input or represented-weight plane changed".to_string(),
        ));
    }
    report.immutable_values = input.len() + weight_codes.len() + weight_scales.len();

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen36Nvfp4LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
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
        return Err(Qwen36Nvfp4LmHeadQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn dot_oracle(
    token: usize,
    row: usize,
    fixture: &Fixture,
) -> Result<f64, Qwen36Nvfp4LmHeadQualificationError> {
    let exceptional = exceptional_group(row);
    let ordinary = (exceptional + 1) % GROUPS_PER_ROW;
    let ordinary_dot = group_dot(token, row, ordinary, fixture);
    let exceptional_dot = group_dot(token, row, exceptional, fixture);
    let ordinary_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, ordinary)])
        .map_err(|error| Qwen36Nvfp4LmHeadQualificationError::Mismatch(error.to_string()))?;
    let exceptional_scale = decode_e4m3fn(fixture.weight_scales[scale_offset(row, exceptional)])
        .map_err(|error| Qwen36Nvfp4LmHeadQualificationError::Mismatch(error.to_string()))?;

    Ok(
        ((GROUPS_PER_ROW - 1) as f64 * ordinary_dot * f64::from(ordinary_scale)
            + exceptional_dot * f64::from(exceptional_scale))
            * f64::from(WEIGHT_SCALE_2),
    )
}

fn group_dot(token: usize, row: usize, group: usize, fixture: &Fixture) -> f64 {
    let weight_begin = row * CODE_BYTES_PER_ROW + group * (GROUP / 2);
    let input_begin = token * INPUT_COLUMNS + group * GROUP;
    let mut sum = 0.0f64;
    for column in 0..GROUP {
        let packed = fixture.weight_codes[weight_begin + column / 2];
        let code = if column & 1 == 0 {
            packed & 15
        } else {
            packed >> 4
        };
        sum += f64::from(fixture.input_f32[input_begin + column]) * f64::from(decode_e2m1(code));
    }

    sum
}

fn sampled_rows() -> [usize; ORACLE_ROWS] {
    core::array::from_fn(|index| {
        if index < ORACLE_BOUNDARIES.len() {
            ORACLE_BOUNDARIES[index]
        } else {
            (index - ORACLE_BOUNDARIES.len()) * (OUTPUT_ROWS - 1)
                / (ORACLE_ROWS - ORACLE_BOUNDARIES.len() - 1)
        }
    })
}

fn exceptional_group(row: usize) -> usize {
    (row * 17 + row / 128 * 13) % GROUPS_PER_ROW
}

pub(crate) fn scale_offset(row: usize, group: usize) -> usize {
    let tile = row / 128;
    let row_in_tile = row & 127;
    let scale_tile = group / 4;
    let scale_lane = group & 3;
    let row_mod32 = row_in_tile & 31;
    let row_quartile = row_in_tile >> 5;

    (tile * (GROUPS_PER_ROW / 4) + scale_tile) * 512
        + row_mod32 * 16
        + row_quartile * 4
        + scale_lane
}

fn bf16_is_finite(word: u16) -> bool {
    word & 0x7f80 != 0x7f80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_swizzle_and_samples_cover_exact_geometry() {
        let (layout, regions) = layout().unwrap();
        let rows = sampled_rows();

        assert_eq!(scale_offset(0, 0), 0);
        assert_eq!(scale_offset(127, 127), 16_383);
        assert_eq!(scale_offset(128, 0), 16_384);
        assert_eq!(rows[0], 0);
        assert_eq!(&rows[..ORACLE_BOUNDARIES.len()], &ORACLE_BOUNDARIES);
        assert_eq!(rows[ORACLE_ROWS - 1], OUTPUT_ROWS - 1);
        assert_eq!(regions.weight_bytes(), 286_064_640);
        assert_eq!(regions.payload_bytes() - regions.weight_bytes(), 4_005_888);
        assert_eq!(layout.byte_len(), 290_070_528);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires an exclusive NVIDIA compute-capability 12.0 device"]
    fn exact_batches_match_independent_oracles_and_graph_replay()
    -> Result<(), Qwen36Nvfp4LmHeadQualificationError> {
        let report = qualify_qwen36_nvfp4_lm_head()?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();
        let inactive_rows = (1..=MAX_BATCH)
            .map(|batch| MAX_BATCH - batch)
            .sum::<usize>();

        assert_eq!(report.sampled_logits, active_rows * ORACLE_ROWS);
        assert_eq!(report.published_logits, active_rows * OUTPUT_ROWS);
        assert_eq!(report.graph_replay_values, active_rows * OUTPUT_ROWS);
        assert_eq!(report.inactive_values, 2 * inactive_rows * OUTPUT_ROWS);
        assert_eq!(report.immutable_values, 286_081_024);
        assert_eq!(report.arena_bytes, 290_070_528);
        assert_eq!(report.weight_bytes, 286_064_640);
        assert_eq!(report.workspace_bytes, 4_005_888);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

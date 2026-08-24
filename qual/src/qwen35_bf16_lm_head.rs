//! Source-backed qualification for the Qwen3.5 BF16 language-model head.

use crate::fp8_projection_oracle::{BF16_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::target::Qwen35Bf16LmHeadOp;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use tuisko_gpu::{
    ArenaLayout, ArenaRegion, CudaContext, CudaGraph, CudaStream, DeviceArena, GpuError, GpuResult,
    device_memory_info,
};
use tuisko_model::{
    Arch, Bf16TextEndpointBindings, CheckpointError, CheckpointSnapshot, Qwen35_9B,
};

const MAX_BATCH: usize = 8;
const ALIGNMENT: usize = 256;
const HIDDEN: usize = Qwen35_9B::HIDDEN;
const VOCAB: usize = Qwen35_9B::VOCAB;
const LOGIT_SAMPLES: usize = 64;
const INPUT_PATTERN: [f32; 16] = [
    0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0, 0.75, -0.75, 0.375, -0.375, 0.0625, -0.0625, 1.0,
    -1.0, 0.0,
];
const TOKEN_FACTORS: [f32; MAX_BATCH] = [1.0, 0.5, 0.25, 0.125, -1.0, -0.5, -0.25, -0.125];

/// Failure of the exact Qwen3.5 BF16 LM-head gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35Bf16LmHeadQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// GPU ownership, launch, or driver failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 BF16 LM-head qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from every exact batch route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35Bf16LmHeadQualification {
    /// Sampled source-backed logits compared with an FP64 dot product.
    pub sampled_logits: usize,
    /// Active BF16 logits reproduced bit-exactly by graph replay.
    pub graph_replay_values: usize,
    /// Sentinel values verified outside each active route extent.
    pub inactive_values: usize,
    /// Read-only input and sampled source-weight words proved unchanged.
    pub immutable_values: usize,
    /// Exact bytes in the one-allocation qualification arena.
    pub arena_bytes: usize,
    /// Exact resident BF16 weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable input and output bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from the independent FP64 oracle.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct Regions {
    input: ArenaRegion<u16>,
    weights: ArenaRegion<u16>,
    output: ArenaRegion<u16>,
}

impl Regions {
    const fn payload_bytes(self) -> usize {
        self.input.byte_len() + self.weights.byte_len() + self.output.byte_len()
    }
}

/// Qualifies eager and captured source-backed BF16 LM-head routes at `B=1..=8`.
pub fn qualify_qwen35_bf16_lm_head(
    root: &Path,
) -> Result<Qwen35Bf16LmHeadQualification, Qwen35Bf16LmHeadQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = CheckpointSnapshot::<Qwen35_9B>::open(root)?;
    let bindings = Bf16TextEndpointBindings::bind(&snapshot)?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let (layout, regions) = layout()?;
    let arena = DeviceArena::zeroed(&stream, &layout)?;
    let op = Qwen35Bf16LmHeadOp::new(&context)?;
    let input = make_input();
    arena.copy_from_host(&stream, regions.input, &input)?;
    arena.copy_region_bytes_from_host(&stream, regions.weights, bindings.lm_head.bytes())?;
    let stable_addresses = addresses(&arena, regions)?;
    let mut report = Qwen35Bf16LmHeadQualification {
        sampled_logits: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: layout.byte_len(),
        weight_bytes: regions.weights.byte_len(),
        workspace_bytes: regions.input.byte_len() + regions.output.byte_len(),
        padding_bytes: layout.byte_len() - regions.payload_bytes(),
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        arena.fill(&stream, regions.output, BF16_SENTINEL as u8)?;
        launch(&op, &arena, &stream, regions, batch)?;
        let eager = arena.copy_to_host(&stream, regions.output)?;
        verify_source(batch, &input, bindings, &eager, &mut report)?;

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
            return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
                "device addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_immutable(&arena, &stream, regions, &input, bindings, &mut report)?;
    verify_no_post_warmup_allocation(&context, &op, &arena, &stream, regions)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn layout() -> GpuResult<(ArenaLayout, Regions)> {
    let mut layout = ArenaLayout::new();
    let input = layout.reserve(MAX_BATCH * HIDDEN, ALIGNMENT)?;
    let weights = layout.reserve(VOCAB * HIDDEN, ALIGNMENT)?;
    let output = layout.reserve(MAX_BATCH * VOCAB, ALIGNMENT)?;

    Ok((
        layout,
        Regions {
            input,
            weights,
            output,
        },
    ))
}

fn addresses(arena: &DeviceArena, regions: Regions) -> GpuResult<[usize; 3]> {
    Ok([
        arena.address(regions.input)?.addr(),
        arena.address(regions.weights)?.addr(),
        arena.address(regions.output)?.addr(),
    ])
}

fn launch(
    op: &Qwen35Bf16LmHeadOp,
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
            arena.address(regions.output)?,
        )
    }
}

fn make_input() -> Vec<u16> {
    (0..MAX_BATCH * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            f32_to_bf16(INPUT_PATTERN[index & 15] * TOKEN_FACTORS[token])
        })
        .collect()
}

fn verify_source(
    batch: usize,
    input: &[u16],
    bindings: Bf16TextEndpointBindings<'_>,
    observed: &[u16],
    report: &mut Qwen35Bf16LmHeadQualification,
) -> Result<(), Qwen35Bf16LmHeadQualificationError> {
    for token in 0..batch {
        for row in sampled_rows() {
            let expected = dot_oracle(token, row, input, bindings)?;
            let actual = f64::from(bf16_to_f32(observed[token * VOCAB + row]));
            let absolute_error = (actual - expected).abs();
            let tolerance = 0.0625f64.max(expected.abs() * 0.01);
            report.maximum_absolute_error =
                report.maximum_absolute_error.max(absolute_error as f32);
            if absolute_error > tolerance {
                return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
                    "B={batch} logit token={token}, row={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
            report.sampled_logits += 1;
        }
    }
    verify_inactive(batch, observed)?;
    report.inactive_values += (MAX_BATCH - batch) * VOCAB;

    Ok(())
}

fn dot_oracle(
    token: usize,
    row: usize,
    input: &[u16],
    bindings: Bf16TextEndpointBindings<'_>,
) -> Result<f64, Qwen35Bf16LmHeadQualificationError> {
    let input = &input[token * HIDDEN..(token + 1) * HIDDEN];
    let begin = row * HIDDEN;
    let mut sum = 0.0f64;
    for (column, &activation) in input.iter().enumerate() {
        let weight = bindings.lm_head.word(begin + column).ok_or_else(|| {
            Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
                "LM-head word {} is outside its source view",
                begin + column
            ))
        })?;
        sum += f64::from(bf16_to_f32(activation)) * f64::from(bf16_to_f32(weight));
    }

    Ok(sum)
}

fn verify_replay(
    batch: usize,
    eager: &[u16],
    replay: &[u16],
    report: &mut Qwen35Bf16LmHeadQualification,
) -> Result<(), Qwen35Bf16LmHeadQualificationError> {
    if eager != replay {
        return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
            "B={batch} graph replay differs from eager execution"
        )));
    }
    verify_inactive(batch, replay)?;
    report.graph_replay_values += batch * VOCAB;
    report.inactive_values += (MAX_BATCH - batch) * VOCAB;

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &[u16],
) -> Result<(), Qwen35Bf16LmHeadQualificationError> {
    let begin = batch * VOCAB;
    if let Some(relative) = observed[begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
            "B={batch} modified inactive logit {}",
            begin + relative
        )));
    }

    Ok(())
}

fn verify_immutable(
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
    input: &[u16],
    bindings: Bf16TextEndpointBindings<'_>,
    report: &mut Qwen35Bf16LmHeadQualification,
) -> Result<(), Qwen35Bf16LmHeadQualificationError> {
    if arena.copy_to_host(stream, regions.input)? != input {
        return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(
            "read-only input changed".to_string(),
        ));
    }
    report.immutable_values += input.len();
    for row in sampled_rows() {
        let observed = arena.copy_slice_to_host(stream, regions.weights, row * HIDDEN, HIDDEN)?;
        for (column, actual) in observed.into_iter().enumerate() {
            let expected = bindings
                .lm_head
                .word(row * HIDDEN + column)
                .ok_or_else(|| {
                    Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
                        "LM-head word {} is outside its source view",
                        row * HIDDEN + column
                    ))
                })?;
            if actual != expected {
                return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
                    "read-only LM-head row={row}, column={column} changed"
                )));
            }
        }
        report.immutable_values += HIDDEN;
    }

    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &CudaContext,
    op: &Qwen35Bf16LmHeadOp,
    arena: &DeviceArena,
    stream: &CudaStream,
    regions: Regions,
) -> Result<(), Qwen35Bf16LmHeadQualificationError> {
    let graphs = (1..=MAX_BATCH)
        .map(|batch| CudaGraph::capture(stream, || launch(op, arena, stream, regions, batch)))
        .collect::<GpuResult<Vec<_>>>()?;
    for graph in &graphs {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graph.launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for &batch in &[1usize, 8, 3, 6, 2, 7, 4, 5] {
        // SAFETY: the qualification harness retains every captured allocation through this synchronized replay.
        unsafe { graphs[batch - 1].launch(stream) }?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(Qwen35Bf16LmHeadQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn sampled_rows() -> [usize; LOGIT_SAMPLES] {
    core::array::from_fn(|index| index * (VOCAB - 1) / (LOGIT_SAMPLES - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_and_sample_inventory_are_exact() {
        let (layout, regions) = layout().unwrap();

        assert_eq!(sampled_rows()[0], 0);
        assert_eq!(sampled_rows()[LOGIT_SAMPLES - 1], VOCAB - 1);
        assert_eq!(regions.weights.byte_len(), 2_034_237_440);
        assert_eq!(
            regions.input.byte_len() + regions.output.byte_len(),
            4_038_656
        );
        assert_eq!(layout.byte_len(), 2_038_276_096);
        assert_eq!(layout.byte_len() - regions.payload_bytes(), 0);
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive compute-capability 12.0 device"]
    fn exact_batches_match_source_oracles_and_graph_replay()
    -> Result<(), Qwen35Bf16LmHeadQualificationError> {
        let root = std::env::var("TUISKO_QWEN35_SNAPSHOT").map_err(|_| {
            Qwen35Bf16LmHeadQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is not set".to_string(),
            )
        })?;
        let report = qualify_qwen35_bf16_lm_head(Path::new(&root))?;

        assert_eq!(report.sampled_logits, 2_304);
        assert_eq!(report.graph_replay_values, 8_939_520);
        assert_eq!(report.inactive_values, 13_905_920);
        assert_eq!(report.immutable_values, 294_912);
        assert_eq!(report.arena_bytes, 2_038_276_096);
        assert_eq!(report.weight_bytes, 2_034_237_440);
        assert_eq!(report.workspace_bytes, 4_038_656);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

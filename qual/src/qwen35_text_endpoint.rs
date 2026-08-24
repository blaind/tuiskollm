//! Source-backed qualification for the resident Qwen3.5 text endpoint.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{EngineError, Qwen35EndpointObservables, Qwen35TextEndpointProgram};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, Bf16TextEndpointBindings, Bf16View, CheckpointError, CheckpointSnapshot, Qwen35_9B,
};

const MAX_BATCH: usize = 8;
const LOGIT_SAMPLES: usize = 64;

/// Failure of the source-backed Qwen3.5 text-endpoint gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35TextEndpointQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// Device behavior disagreed with the independent contract.
    #[error("Qwen3.5 text-endpoint qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst errors from the endpoint gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35TextEndpointQualification {
    /// Source embedding words compared bit-exactly.
    pub embedding_values: usize,
    /// Final-normalized values compared with the FP64 formula.
    pub normalized_values: usize,
    /// Sampled full-width BF16 logits compared with FP64 dots.
    pub sampled_logits: usize,
    /// Complete observable values reproduced by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside every active exact batch.
    pub inactive_values: usize,
    /// Sampled immutable BF16 LM-head words proved unchanged.
    pub immutable_values: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable device workspace bytes.
    pub workspace_bytes: usize,
    /// Complete one-allocation owner bytes.
    pub arena_bytes: usize,
    /// Largest final-normalization absolute error.
    pub maximum_normalization_error: f32,
    /// Largest sampled-logit absolute error.
    pub maximum_logit_error: f32,
}

/// Qualifies the resident Qwen3.5 endpoint against one admitted snapshot.
pub fn qualify_qwen35_text_endpoint(
    root: &Path,
) -> Result<Qwen35TextEndpointQualification, Qwen35TextEndpointQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let bindings = Bf16TextEndpointBindings::bind(snapshot.as_ref())?;
    let final_norm_weight = bindings.final_norm.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = Qwen35TextEndpointProgram::from_snapshot(&context, Arc::clone(&snapshot))?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let mut report = Qwen35TextEndpointQualification {
        embedding_values: 0,
        normalized_values: 0,
        sampled_logits: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        weight_bytes: program.resident_weight_bytes(),
        workspace_bytes: program.workspace_bytes(),
        arena_bytes: program.arena_bytes(),
        maximum_normalization_error: 0.0,
        maximum_logit_error: 0.0,
    };

    verify_immutable(&program, &stream, bindings, &mut report)?;
    for batch in 1..=MAX_BATCH {
        let first_ids = token_ids(batch, 0);
        program.stage_embeddings(&stream, &first_ids[..batch])?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let first = program.qualification_observables(&stream)?;

        let second_ids = token_ids(batch, 1);
        program.stage_embeddings(&stream, &second_ids[..batch])?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.replay(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;

        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;

        verify_source(
            batch,
            &second_ids[..batch],
            bindings,
            &final_norm_weight,
            &replay,
            &mut report,
        )?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement_input(batch, &first, &replay)?;
        verify_inactive(batch, &replay, &mut report)?;
        verify_inactive(batch, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
                "endpoint addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&mut program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn verify_source(
    batch: usize,
    token_ids: &[u32],
    bindings: Bf16TextEndpointBindings<'_>,
    final_norm_weight: &[u16],
    observed: &Qwen35EndpointObservables,
    report: &mut Qwen35TextEndpointQualification,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    for (token, &token_id) in token_ids.iter().enumerate() {
        let input = embedding_row(bindings.embedding, token_id as usize)?;
        let begin = token * Qwen35_9B::HIDDEN;
        let end = begin + Qwen35_9B::HIDDEN;
        if let Some(column) = observed.input[begin..end]
            .iter()
            .zip(&input)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
                "embedding at B={batch}, token={token}, column={column} differs from source"
            )));
        }
        report.embedding_values += Qwen35_9B::HIDDEN;

        let normalized = rms_norm_oracle::<Qwen35_9B>(&input, final_norm_weight);
        for (column, (&actual, &expected)) in observed.normalized[begin..end]
            .iter()
            .zip(&normalized)
            .enumerate()
        {
            check_close(
                "final RMSNorm",
                batch,
                token,
                column,
                actual,
                expected,
                0.005,
                0.015625,
                &mut report.maximum_normalization_error,
            )?;
        }
        report.normalized_values += Qwen35_9B::HIDDEN;

        for row in sampled_rows() {
            let expected = logit_oracle(row, &normalized, bindings)?;
            let actual = bf16_to_f32(observed.logits[token * Qwen35_9B::VOCAB + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_logit_error = report.maximum_logit_error.max(error);
            let tolerance = 0.0625f32.max(expected.abs() as f32 * 0.01);
            if error > tolerance {
                return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
                    "logit at B={batch}, token={token}, vocabulary={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
            report.sampled_logits += 1;
        }
    }

    Ok(())
}

fn logit_oracle(
    row: usize,
    normalized: &[u16],
    bindings: Bf16TextEndpointBindings<'_>,
) -> Result<f64, Qwen35TextEndpointQualificationError> {
    let begin = row * Qwen35_9B::HIDDEN;
    let mut sum = 0.0f64;
    for (column, &activation) in normalized.iter().enumerate() {
        let weight = bindings.lm_head.word(begin + column).ok_or_else(|| {
            Qwen35TextEndpointQualificationError::Mismatch(format!(
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
    eager: &Qwen35EndpointObservables,
    replay: &Qwen35EndpointObservables,
    report: &mut Qwen35TextEndpointQualification,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    compare_words(batch, "input", &eager.input, &replay.input)?;
    compare_words(batch, "normalized", &eager.normalized, &replay.normalized)?;
    compare_words(batch, "logits", &eager.logits, &replay.logits)?;
    report.graph_replay_values += batch * (2 * Qwen35_9B::HIDDEN + Qwen35_9B::VOCAB);

    Ok(())
}

fn verify_replacement_input(
    batch: usize,
    first: &Qwen35EndpointObservables,
    second: &Qwen35EndpointObservables,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    let active = batch * Qwen35_9B::HIDDEN;
    if first.input[..active] == second.input[..active]
        || first.normalized[..active] == second.normalized[..active]
    {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement embeddings"
        )));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Qwen35EndpointObservables,
    report: &mut Qwen35TextEndpointQualification,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    let normalized_begin = batch * Qwen35_9B::HIDDEN;
    let logits_begin = batch * Qwen35_9B::VOCAB;
    require_sentinel(
        batch,
        "normalized",
        &observed.normalized[normalized_begin..],
    )?;
    require_sentinel(batch, "logits", &observed.logits[logits_begin..])?;
    report.inactive_values += (MAX_BATCH - batch) * (Qwen35_9B::HIDDEN + Qwen35_9B::VOCAB);

    Ok(())
}

fn verify_immutable(
    program: &Qwen35TextEndpointProgram,
    stream: &CudaStream,
    bindings: Bf16TextEndpointBindings<'_>,
    report: &mut Qwen35TextEndpointQualification,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    for row in sampled_rows() {
        let observed = program.qualification_lm_head_row(stream, row)?;
        for (column, actual) in observed.into_iter().enumerate() {
            let expected = bindings
                .lm_head
                .word(row * Qwen35_9B::HIDDEN + column)
                .ok_or_else(|| {
                    Qwen35TextEndpointQualificationError::Mismatch(format!(
                        "LM-head word {} is outside its source view",
                        row * Qwen35_9B::HIDDEN + column
                    ))
                })?;
            if actual != expected {
                return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
                    "resident LM-head row={row}, column={column} differs from source"
                )));
            }
        }
        report.immutable_values += Qwen35_9B::HIDDEN;
    }

    Ok(())
}

fn verify_no_post_warmup_allocation(
    program: &mut Qwen35TextEndpointProgram,
    stream: &CudaStream,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    let warm = token_ids(MAX_BATCH, 2);
    program.stage_embeddings(stream, &warm)?;
    program.replay(stream, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for round in 0..4 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            let ids = token_ids(batch, round + 3);
            program.stage_embeddings(stream, &ids[..batch])?;
            program.replay(stream, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn embedding_row(
    embedding: Bf16View<'_, 2>,
    token: usize,
) -> Result<Vec<u16>, Qwen35TextEndpointQualificationError> {
    let begin = token * Qwen35_9B::HIDDEN;
    (begin..begin + Qwen35_9B::HIDDEN)
        .map(|index| {
            embedding.word(index).ok_or_else(|| {
                Qwen35TextEndpointQualificationError::Mismatch(format!(
                    "embedding word {index} is outside its source view"
                ))
            })
        })
        .collect()
}

fn token_ids(batch: usize, salt: usize) -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| {
        ((17 + batch * 7_919 + salt * 31_337 + row * 65_537) % Qwen35_9B::VOCAB) as u32
    })
}

fn sampled_rows() -> [usize; LOGIT_SAMPLES] {
    core::array::from_fn(|index| index * (Qwen35_9B::VOCAB - 1) / (LOGIT_SAMPLES - 1))
}

#[allow(clippy::too_many_arguments)]
fn check_close(
    operation: &str,
    batch: usize,
    token: usize,
    column: usize,
    actual_bits: u16,
    oracle_bits: u16,
    relative_tolerance: f32,
    absolute_tolerance: f32,
    maximum_error: &mut f32,
) -> Result<(), Qwen35TextEndpointQualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_error = maximum_error.max(error);
    let tolerance = absolute_tolerance.max(oracle.abs() * relative_tolerance);
    if error > tolerance {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "{operation} at B={batch}, token={token}, column={column}: device={actual}, oracle={oracle}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

fn compare_words(
    batch: usize,
    name: &str,
    expected: &[u16],
    actual: &[u16],
) -> Result<(), Qwen35TextEndpointQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "B={batch} {name} graph value {index} differs: replay={:#06x}, eager={:#06x}",
            actual[index], expected[index]
        )));
    }

    Ok(())
}

fn require_sentinel(
    batch: usize,
    name: &str,
    values: &[u16],
) -> Result<(), Qwen35TextEndpointQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != BF16_SENTINEL) {
        return Err(Qwen35TextEndpointQualificationError::Mismatch(format!(
            "B={batch} modified inactive {name} value {index}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_and_logit_samples_cover_boundaries() {
        assert_eq!(sampled_rows()[0], 0);
        assert_eq!(sampled_rows()[LOGIT_SAMPLES - 1], Qwen35_9B::VOCAB - 1);
        for batch in 1..=MAX_BATCH {
            assert!(token_ids(batch, 0).iter().all(|&id| id < 248_320));
        }
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive compute-capability 12.0 device"]
    fn source_endpoint_matches_complete_oracles_and_graph_replay()
    -> Result<(), Qwen35TextEndpointQualificationError> {
        let root = std::env::var("TUISKO_QWEN35_SNAPSHOT").map_err(|_| {
            Qwen35TextEndpointQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is not set".to_string(),
            )
        })?;
        let report = qualify_qwen35_text_endpoint(Path::new(&root))?;

        assert_eq!(report.embedding_values, 147_456);
        assert_eq!(report.normalized_values, 147_456);
        assert_eq!(report.sampled_logits, 2_304);
        assert_eq!(report.graph_replay_values, 9_234_432);
        assert_eq!(report.inactive_values, 14_135_296);
        assert_eq!(report.immutable_values, 262_144);
        assert_eq!(report.weight_bytes, 2_034_245_632);
        assert_eq!(report.workspace_bytes, 4_104_192);
        assert_eq!(report.arena_bytes, 2_038_349_824);
        assert!(report.maximum_normalization_error.is_finite());
        assert!(report.maximum_logit_error.is_finite());

        Ok(())
    }
}

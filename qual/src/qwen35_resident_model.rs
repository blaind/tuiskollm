//! Source-backed qualification for the complete resident Qwen3.5 text model.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen35ResidentModelObservables, Qwen35ResidentModelProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, Bf16TextEndpointBindings, CheckpointError, CheckpointSnapshot, Qwen35_9B,
};

const ROTARY_PAIRS: usize = 32;
const LOGIT_SAMPLES: usize = 64;

/// Failure of the source-backed Qwen3.5 resident-model gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35ResidentModelQualificationError {
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
    #[error("Qwen3.5 resident-model qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst errors from the whole-model gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35ResidentModelQualification {
    /// Endpoint values checked against independent BF16/FP64 formulas.
    pub oracle_values: usize,
    /// Complete active final planes reproduced by eager and graph execution.
    pub graph_replay_values: usize,
    /// Full-vocabulary BF16 logits checked for finite values.
    pub finite_logits: usize,
    /// Endpoint values preserved outside every active exact batch.
    pub inactive_values: usize,
    /// Exact batches whose replacement embeddings changed the final residual.
    pub replacement_cases: usize,
    /// Stable layer and endpoint arena addresses retained by the owner.
    pub arena_addresses: usize,
    /// Exact resident weight bytes.
    pub weight_bytes: usize,
    /// Exact BF16 K/V cache bytes.
    pub cache_bytes: usize,
    /// Exact address-stable state and workspace bytes.
    pub workspace_bytes: usize,
    /// Complete allocated device bytes including alignment.
    pub arena_bytes: usize,
    /// Largest final-normalization absolute error.
    pub maximum_normalization_error: f32,
    /// Largest sampled-logit absolute error.
    pub maximum_logit_error: f32,
}

/// Qualifies all exact-B whole-model graphs against eager replay and endpoint formulas.
pub fn qualify_qwen35_resident_model(
    root: &Path,
) -> Result<Qwen35ResidentModelQualification, Qwen35ResidentModelQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let bindings = Bf16TextEndpointBindings::bind(snapshot.as_ref())?;
    let final_norm_weight = bindings.final_norm.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = Qwen35ResidentModelProgram::from_snapshot(&context, Arc::clone(&snapshot))?;
    let stable_addresses = program.base_addresses();
    let layout = program.layout();
    let mut report = Qwen35ResidentModelQualification {
        oracle_values: 0,
        graph_replay_values: 0,
        finite_logits: 0,
        inactive_values: 0,
        replacement_cases: 0,
        arena_addresses: stable_addresses.len(),
        weight_bytes: layout.resident_weight_bytes(),
        cache_bytes: layout.cache_bytes(),
        workspace_bytes: layout.workspace_bytes(),
        arena_bytes: layout.arena_bytes(),
        maximum_normalization_error: 0.0,
        maximum_logit_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        prepare_run(&mut program, &stream, batch, 0)?;
        program.replay(&stream, batch)?;
        let first = program.qualification_observables(&stream)?;

        prepare_run(&mut program, &stream, batch, 1)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;

        prepare_run(&mut program, &stream, batch, 1)?;
        program.replay(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;

        verify_replacement(batch, &first, &replay, &mut report)?;
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_endpoint_oracle(batch, bindings, &final_norm_weight, &replay, &mut report)?;
        verify_finite_logits(batch, &replay, &mut report)?;
        verify_inactive(batch, &eager, &mut report)?;
        verify_inactive(batch, &replay, &mut report)?;

        if program.base_addresses() != stable_addresses {
            return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
                "resident arena addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_post_warmup_allocation(&mut program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn prepare_run(
    program: &mut Qwen35ResidentModelProgram,
    stream: &CudaStream,
    batch: usize,
    salt: usize,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    program.reset_state(stream)?;
    let ids = token_ids(batch, salt);
    program.stage_embeddings(stream, &ids[..batch])?;
    let positions = vec![0u32; batch];
    let rope_cos = vec![1.0f32; batch * ROTARY_PAIRS];
    let rope_sin = vec![0.0f32; batch * ROTARY_PAIRS];
    program.load_decode_state(stream, batch, &positions, &rope_cos, &rope_sin)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

fn verify_replacement(
    batch: usize,
    first: &Qwen35ResidentModelObservables,
    second: &Qwen35ResidentModelObservables,
    report: &mut Qwen35ResidentModelQualification,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    let active = batch * Qwen35_9B::HIDDEN;
    if first.final_residual[..active] == second.final_residual[..active] {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "B={batch} whole-model graph did not observe replacement embeddings"
        )));
    }
    report.replacement_cases += 1;

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Qwen35ResidentModelObservables,
    replay: &Qwen35ResidentModelObservables,
    report: &mut Qwen35ResidentModelQualification,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    let hidden = batch * Qwen35_9B::HIDDEN;
    let logits = batch * Qwen35_9B::VOCAB;
    compare_words(
        batch,
        "final residual",
        &eager.final_residual[..hidden],
        &replay.final_residual[..hidden],
    )?;
    compare_words(
        batch,
        "final normalization",
        &eager.normalized[..hidden],
        &replay.normalized[..hidden],
    )?;
    compare_words(
        batch,
        "logits",
        &eager.logits[..logits],
        &replay.logits[..logits],
    )?;
    report.graph_replay_values += 2 * hidden + logits;

    Ok(())
}

fn verify_endpoint_oracle(
    batch: usize,
    bindings: Bf16TextEndpointBindings<'_>,
    final_norm_weight: &[u16],
    observed: &Qwen35ResidentModelObservables,
    report: &mut Qwen35ResidentModelQualification,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    for token in 0..batch {
        let begin = token * Qwen35_9B::HIDDEN;
        let end = begin + Qwen35_9B::HIDDEN;
        let normalized =
            rms_norm_oracle::<Qwen35_9B>(&observed.final_residual[begin..end], final_norm_weight);
        for (column, (&actual_bits, &expected_bits)) in observed.normalized[begin..end]
            .iter()
            .zip(&normalized)
            .enumerate()
        {
            let actual = bf16_to_f32(actual_bits);
            let expected = bf16_to_f32(expected_bits);
            let error = (actual - expected).abs();
            report.maximum_normalization_error = report.maximum_normalization_error.max(error);
            let tolerance = 0.015625f32.max(expected.abs() * 0.005);
            if error > tolerance {
                return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
                    "final RMSNorm at B={batch}, token={token}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
        }
        report.oracle_values += Qwen35_9B::HIDDEN;

        for row in sampled_rows() {
            let expected = logit_oracle(row, &normalized, bindings)?;
            let actual = bf16_to_f32(observed.logits[token * Qwen35_9B::VOCAB + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_logit_error = report.maximum_logit_error.max(error);
            let tolerance = 0.0625f32.max(expected.abs() as f32 * 0.01);
            if error > tolerance {
                return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
                    "logit at B={batch}, token={token}, vocabulary={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
            report.oracle_values += 1;
        }
    }

    Ok(())
}

fn logit_oracle(
    row: usize,
    normalized: &[u16],
    bindings: Bf16TextEndpointBindings<'_>,
) -> Result<f64, Qwen35ResidentModelQualificationError> {
    let begin = row * Qwen35_9B::HIDDEN;
    let mut sum = 0.0f64;
    for (column, &activation) in normalized.iter().enumerate() {
        let weight = bindings.lm_head.word(begin + column).ok_or_else(|| {
            Qwen35ResidentModelQualificationError::Mismatch(format!(
                "LM-head word {} is outside its source view",
                begin + column
            ))
        })?;
        sum += f64::from(bf16_to_f32(activation)) * f64::from(bf16_to_f32(weight));
    }

    Ok(sum)
}

fn verify_finite_logits(
    batch: usize,
    observed: &Qwen35ResidentModelObservables,
    report: &mut Qwen35ResidentModelQualification,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    let active = batch * Qwen35_9B::VOCAB;
    if let Some(index) = observed.logits[..active]
        .iter()
        .position(|bits| bits & 0x7f80 == 0x7f80)
    {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "B={batch} logit {index} is not finite"
        )));
    }
    report.finite_logits += active;

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Qwen35ResidentModelObservables,
    report: &mut Qwen35ResidentModelQualification,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    let normalized_begin = batch * Qwen35_9B::HIDDEN;
    let logits_begin = batch * Qwen35_9B::VOCAB;
    require_sentinel(
        batch,
        "normalization",
        &observed.normalized[normalized_begin..],
    )?;
    require_sentinel(batch, "logits", &observed.logits[logits_begin..])?;
    report.inactive_values += (MAX_BATCH - batch) * (Qwen35_9B::HIDDEN + Qwen35_9B::VOCAB);

    Ok(())
}

fn verify_no_post_warmup_allocation(
    program: &mut Qwen35ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(), Qwen35ResidentModelQualificationError> {
    prepare_run(program, stream, MAX_BATCH, 2)?;
    program.replay(stream, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for round in 0..2 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            prepare_run(program, stream, batch, round + 3)?;
            program.replay(stream, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn compare_words(
    batch: usize,
    name: &str,
    expected: &[u16],
    actual: &[u16],
) -> Result<(), Qwen35ResidentModelQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "B={batch} {name} value {index} differs: replay={:#06x}, eager={:#06x}",
            actual[index], expected[index]
        )));
    }

    Ok(())
}

fn require_sentinel(
    batch: usize,
    name: &str,
    values: &[u16],
) -> Result<(), Qwen35ResidentModelQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != BF16_SENTINEL) {
        return Err(Qwen35ResidentModelQualificationError::Mismatch(format!(
            "B={batch} modified inactive {name} value {index}"
        )));
    }

    Ok(())
}

fn token_ids(batch: usize, salt: usize) -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| {
        ((17 + batch * 7_919 + salt * 31_337 + row * 65_537) % Qwen35_9B::VOCAB) as u32
    })
}

fn sampled_rows() -> [usize; LOGIT_SAMPLES] {
    core::array::from_fn(|index| index * (Qwen35_9B::VOCAB - 1) / (LOGIT_SAMPLES - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_cover_exact_boundaries() {
        assert_eq!(sampled_rows()[0], 0);
        assert_eq!(sampled_rows()[LOGIT_SAMPLES - 1], Qwen35_9B::VOCAB - 1);
        assert!(
            (1..=MAX_BATCH)
                .flat_map(|batch| token_ids(batch, 0))
                .all(|id| id < Qwen35_9B::VOCAB as u32)
        );
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive compute-capability 12.0 device"]
    fn whole_model_matches_endpoint_oracles_and_graph_replay()
    -> Result<(), Qwen35ResidentModelQualificationError> {
        let root = std::env::var("TUISKO_QWEN35_SNAPSHOT").map_err(|_| {
            Qwen35ResidentModelQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is not set".to_string(),
            )
        })?;
        let report = qualify_qwen35_resident_model(Path::new(&root))?;

        assert_eq!(report.oracle_values, 149_760);
        assert_eq!(report.graph_replay_values, 9_234_432);
        assert_eq!(report.finite_logits, 8_939_520);
        assert_eq!(report.inactive_values, 14_135_296);
        assert_eq!(report.replacement_cases, 8);
        assert_eq!(report.arena_addresses, 33);
        assert_eq!(report.weight_bytes, 5_931_820_032);
        assert_eq!(report.cache_bytes, 50_331_648);
        assert_eq!(report.workspace_bytes, 897_919_232);
        assert_eq!(report.arena_bytes, 6_880_092_160);
        assert!(report.maximum_normalization_error.is_finite());
        assert!(report.maximum_logit_error.is_finite());

        Ok(())
    }
}

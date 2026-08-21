//! Source-backed qualification for the resident text endpoint.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, quantize_oracle,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{EndpointObservables, EngineError, TextEndpointProgram};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{
    Arch, Bf16View, CheckpointError, CheckpointSnapshot, Qwen38_27B, TextEndpointBindings,
};

const MAX_BATCH: usize = 8;
const LOGIT_SAMPLES: usize = 64;

/// Failure of the source-backed text-endpoint gate.
#[derive(Debug, thiserror::Error)]
pub enum TextEndpointQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Resident engine setup or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),

    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// The exact-target device was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),

    /// Device behavior disagreed with the independent contract.
    #[error("text-endpoint qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors from the endpoint gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextEndpointQualification {
    /// Source embedding words compared bit-exactly.
    pub embedding_values: usize,
    /// Final-normalized values compared with the FP64 formula.
    pub normalized_values: usize,
    /// Dynamic E4M3 activation codes compared bit-exactly.
    pub activation_codes: usize,
    /// Dynamic FP32 activation scales compared bit-exactly.
    pub activation_scales: usize,
    /// Sampled full-width source-weight logits compared with the FP64 formula.
    pub sampled_logits: usize,
    /// Active values reproduced bit-exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact batch.
    pub inactive_values: usize,
    /// Largest final-normalization absolute error.
    pub maximum_normalization_error: f32,
    /// Largest sampled-logit absolute error.
    pub maximum_logit_error: f32,
}

/// Qualifies the resident endpoint against one admitted snapshot.
pub fn qualify_text_endpoint(
    root: &Path,
) -> Result<TextEndpointQualification, TextEndpointQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let bindings = TextEndpointBindings::bind(snapshot.as_ref())?;
    let final_norm_weight = bindings.final_norm.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = TextEndpointProgram::from_snapshot(&context, snapshot.clone())?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let mut report = TextEndpointQualification {
        embedding_values: 0,
        normalized_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        sampled_logits: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_normalization_error: 0.0,
        maximum_logit_error: 0.0,
    };

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
        verify_live_input(batch, &first, &replay)?;
        verify_inactive(batch, &replay, &mut report)?;
        verify_inactive(batch, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(TextEndpointQualificationError::Mismatch(format!(
                "endpoint addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_device_allocation(&mut program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn verify_source(
    batch: usize,
    token_ids: &[u32],
    bindings: TextEndpointBindings<'_>,
    final_norm_weight: &[u16],
    observed: &EndpointObservables,
    report: &mut TextEndpointQualification,
) -> Result<(), TextEndpointQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let vocab = Qwen38_27B::VOCAB;
    let sampled_rows = sampled_logit_rows();

    for (token, &token_id) in token_ids.iter().enumerate() {
        let input = embedding_row(bindings.embedding, token_id as usize)?;
        let begin = token * hidden;
        let end = begin + hidden;
        if let Some(column) = observed.input[begin..end]
            .iter()
            .zip(&input)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(TextEndpointQualificationError::Mismatch(format!(
                "embedding at B={batch}, row={token}, column={column}: device={:#06x}, source={:#06x}",
                observed.input[begin + column],
                input[column]
            )));
        }
        report.embedding_values += hidden;

        let normalized = rms_norm_oracle::<Qwen38_27B>(&input, final_norm_weight);
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
        report.normalized_values += hidden;

        let activation =
            quantize_oracle(&normalized).map_err(TextEndpointQualificationError::Mismatch)?;
        if let Some(column) = observed.activation_codes[begin..end]
            .iter()
            .zip(&activation.codes)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(TextEndpointQualificationError::Mismatch(format!(
                "activation code at B={batch}, row={token}, column={column}: device={:#04x}, oracle={:#04x}",
                observed.activation_codes[begin + column],
                activation.codes[column]
            )));
        }
        if observed.activation_scales[token].to_bits() != activation.scale.to_bits() {
            return Err(TextEndpointQualificationError::Mismatch(format!(
                "activation scale at B={batch}, row={token}: device={:#010x}, oracle={:#010x}",
                observed.activation_scales[token].to_bits(),
                activation.scale.to_bits()
            )));
        }
        report.activation_codes += hidden;
        report.activation_scales += 1;

        for &row in &sampled_rows {
            let expected = logit_oracle(row, &activation.codes, activation.scale, bindings)?;
            let actual = bf16_to_f32(observed.logits[token * vocab + row]);
            let error = (f64::from(actual) - expected).abs() as f32;
            report.maximum_logit_error = report.maximum_logit_error.max(error);
            let tolerance = 0.0625f32.max(expected.abs() as f32 * 0.01);
            if error > tolerance {
                return Err(TextEndpointQualificationError::Mismatch(format!(
                    "logit at B={batch}, row={token}, vocabulary={row}: device={actual}, oracle={expected}, tolerance={tolerance}"
                )));
            }
            report.sampled_logits += 1;
        }
    }

    Ok(())
}

fn logit_oracle(
    row: usize,
    activation_codes: &[u8],
    activation_scale: f32,
    bindings: TextEndpointBindings<'_>,
) -> Result<f64, TextEndpointQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let begin = row * hidden;
    let end = begin + hidden;
    let weight_codes = &bindings.lm_head.codes()[begin..end];
    let mut sum = 0.0f64;
    for (&activation, &weight) in activation_codes.iter().zip(weight_codes) {
        let activation =
            decode_e4m3fn(activation).map_err(TextEndpointQualificationError::Mismatch)?;
        let weight = decode_e4m3fn(weight).map_err(TextEndpointQualificationError::Mismatch)?;
        sum += f64::from(activation) * f64::from(weight);
    }
    let weight_scale = bindings
        .lm_head_scale
        .word(row)
        .ok_or_else(|| {
            TextEndpointQualificationError::Mismatch(format!(
                "LM-head scale row {row} is outside its source view"
            ))
        })
        .map(bf16_to_f32)?;

    Ok(sum * f64::from(activation_scale) * f64::from(weight_scale))
}

fn verify_replay(
    batch: usize,
    eager: &EndpointObservables,
    replay: &EndpointObservables,
    report: &mut TextEndpointQualification,
) -> Result<(), TextEndpointQualificationError> {
    compare_words(batch, "staged input", &eager.input, &replay.input)?;
    compare_words(batch, "normalized", &eager.normalized, &replay.normalized)?;
    compare_bytes(
        batch,
        "activation codes",
        &eager.activation_codes,
        &replay.activation_codes,
    )?;
    compare_f32_bits(
        batch,
        "activation scales",
        &eager.activation_scales,
        &replay.activation_scales,
    )?;
    compare_words(batch, "logits", &eager.logits, &replay.logits)?;

    report.graph_replay_values += batch * (2 * Qwen38_27B::HIDDEN + 1 + Qwen38_27B::VOCAB);

    Ok(())
}

fn verify_live_input(
    batch: usize,
    first: &EndpointObservables,
    second: &EndpointObservables,
) -> Result<(), TextEndpointQualificationError> {
    let active = batch * Qwen38_27B::HIDDEN;
    if first.input[..active] == second.input[..active]
        || first.normalized[..active] == second.normalized[..active]
    {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe the replacement embedding rows"
        )));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &EndpointObservables,
    report: &mut TextEndpointQualification,
) -> Result<(), TextEndpointQualificationError> {
    let hidden_begin = batch * Qwen38_27B::HIDDEN;
    let logits_begin = batch * Qwen38_27B::VOCAB;
    require_sentinel_words(batch, "normalized", &observed.normalized[hidden_begin..])?;
    require_sentinel_bytes(
        batch,
        "activation codes",
        &observed.activation_codes[hidden_begin..],
    )?;
    if let Some(relative) = observed.activation_scales[batch..]
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} modified inactive activation scale {}",
            batch + relative
        )));
    }
    require_sentinel_words(batch, "logits", &observed.logits[logits_begin..])?;
    report.inactive_values +=
        (MAX_BATCH - batch) * (2 * Qwen38_27B::HIDDEN + 1 + Qwen38_27B::VOCAB);

    Ok(())
}

fn verify_no_device_allocation(
    program: &mut TextEndpointProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), TextEndpointQualificationError> {
    let warm_ids = token_ids(MAX_BATCH, 2);
    program.stage_embeddings(stream, &warm_ids)?;
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
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn embedding_row(
    embedding: Bf16View<'_, 2>,
    token: usize,
) -> Result<Vec<u16>, TextEndpointQualificationError> {
    let begin = token * Qwen38_27B::HIDDEN;
    (begin..begin + Qwen38_27B::HIDDEN)
        .map(|index| {
            embedding.word(index).ok_or_else(|| {
                TextEndpointQualificationError::Mismatch(format!(
                    "embedding word {index} is outside its source view"
                ))
            })
        })
        .collect()
}

fn token_ids(batch: usize, salt: usize) -> [u32; MAX_BATCH] {
    core::array::from_fn(|row| {
        let token = (17 + batch * 7_919 + salt * 31_337 + row * 65_537) % Qwen38_27B::VOCAB;
        token as u32
    })
}

fn sampled_logit_rows() -> [usize; LOGIT_SAMPLES] {
    core::array::from_fn(|index| index * (Qwen38_27B::VOCAB - 1) / (LOGIT_SAMPLES - 1))
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
) -> Result<(), TextEndpointQualificationError> {
    let actual = bf16_to_f32(actual_bits);
    let oracle = bf16_to_f32(oracle_bits);
    let error = (actual - oracle).abs();
    *maximum_error = maximum_error.max(error);
    let tolerance = absolute_tolerance.max(oracle.abs() * relative_tolerance);
    if error > tolerance {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "{operation} at B={batch}, row={token}, column={column}: device={actual}, oracle={oracle}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

fn compare_words(
    batch: usize,
    name: &str,
    expected: &[u16],
    actual: &[u16],
) -> Result<(), TextEndpointQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} {name} graph value {index} differs: replay={:#06x}, eager={:#06x}",
            actual[index], expected[index]
        )));
    }

    Ok(())
}

fn compare_bytes(
    batch: usize,
    name: &str,
    expected: &[u8],
    actual: &[u8],
) -> Result<(), TextEndpointQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} {name} graph value {index} differs: replay={:#04x}, eager={:#04x}",
            actual[index], expected[index]
        )));
    }

    Ok(())
}

fn compare_f32_bits(
    batch: usize,
    name: &str,
    expected: &[f32],
    actual: &[f32],
) -> Result<(), TextEndpointQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} {name} graph value {index} differs: replay={:#010x}, eager={:#010x}",
            actual[index].to_bits(),
            expected[index].to_bits()
        )));
    }

    Ok(())
}

fn require_sentinel_words(
    batch: usize,
    name: &str,
    values: &[u16],
) -> Result<(), TextEndpointQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != BF16_SENTINEL) {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} modified inactive {name} value {index}"
        )));
    }

    Ok(())
}

fn require_sentinel_bytes(
    batch: usize,
    name: &str,
    values: &[u8],
) -> Result<(), TextEndpointQualificationError> {
    if let Some(index) = values.iter().position(|&value| value != BYTE_SENTINEL) {
        return Err(TextEndpointQualificationError::Mismatch(format!(
            "B={batch} modified inactive {name} value {index}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LOGIT_SAMPLES, MAX_BATCH, Qwen38_27B, qualify_text_endpoint, sampled_logit_rows};
    use std::path::PathBuf;
    use tuisko_model::Arch;

    #[test]
    fn sampled_rows_cover_the_full_vocabulary_without_duplicates() {
        let rows = sampled_logit_rows();

        assert_eq!(rows.len(), LOGIT_SAMPLES);
        assert_eq!(rows[0], 0);
        assert_eq!(rows[LOGIT_SAMPLES - 1], Qwen38_27B::VOCAB - 1);
        assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    #[ignore = "requires an idle sm_120a device and TUISKO_SNAPSHOT"]
    fn source_endpoint_matches_independent_oracles_and_graph_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = PathBuf::from(
            std::env::var_os("TUISKO_SNAPSHOT")
                .ok_or("TUISKO_SNAPSHOT must name the admitted Hugging Face snapshot")?,
        );
        let report = qualify_text_endpoint(&root)?;
        let active_rows = (1..=MAX_BATCH).sum::<usize>();

        assert_eq!(report.embedding_values, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.normalized_values, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.activation_codes, active_rows * Qwen38_27B::HIDDEN);
        assert_eq!(report.activation_scales, active_rows);
        assert_eq!(report.sampled_logits, active_rows * LOGIT_SAMPLES);
        assert!(report.maximum_normalization_error <= 0.015625);

        Ok(())
    }
}

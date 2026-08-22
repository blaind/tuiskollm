//! Source-backed qualification for one resident dense-FP8 MLP owner.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, TokenOracle, bf16_to_f32, decode_e4m3fn,
    f32_to_bf16, quantize_oracle,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{DenseFp8MlpObservables, DenseFp8MlpProgram, EngineError, MAX_BATCH};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, DenseFp8MlpBindings, Qwen38_27B};

const SOURCE_LAYER: usize = 60;

/// Failure of the complete dense-FP8 MLP qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum DenseFp8MlpQualificationError {
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
    /// Device behavior disagreed with the independent source formula.
    #[error("dense-FP8 MLP qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst errors from the complete MLP boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseFp8MlpQualification {
    /// Pre-MLP normalized BF16 values checked at every exact batch.
    pub normalized_values: usize,
    /// Gate/up and down dynamic E4M3 codes checked bit-exactly.
    pub activation_codes: usize,
    /// Gate/up and down FP32 activation scales checked bit-exactly.
    pub activation_scales: usize,
    /// B=1 SwiGLU values checked against the complete source-weight formula.
    pub source_swiglu_values: usize,
    /// B=1 down values checked against the complete source-weight formula.
    pub source_branch_values: usize,
    /// Published residual and next-normalized values checked at every batch.
    pub boundary_values: usize,
    /// Active values reproduced exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside exact batch extents.
    pub inactive_values: usize,
    /// Largest absolute difference at a represented BF16 seam.
    pub maximum_absolute_error: f32,
}

/// Qualifies source-backed layer 60 at every exact decode batch.
pub fn qualify_dense_fp8_mlp(
    root: &Path,
) -> Result<DenseFp8MlpQualification, DenseFp8MlpQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let bindings = DenseFp8MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let input_norm = bindings.input_norm.words().collect::<Vec<_>>();
    let next_norm = bindings.next_norm.words().collect::<Vec<_>>();
    let gate_up_scales = little_endian_words(bindings.gate_up.scale_bf16)?;
    let down_scales = bindings.down.scale.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = DenseFp8MlpProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let mut report = DenseFp8MlpQualification {
        normalized_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        source_swiglu_values: 0,
        source_branch_values: 0,
        boundary_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        let first_input = make_input(batch, 0);
        program.load_residual(&stream, batch, &first_input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(batch, 1);
        program.load_residual(&stream, batch, &input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.replay(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;

        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;

        verify_seams(batch, &input, &input_norm, &next_norm, &replay, &mut report)?;
        if batch == 1 {
            verify_source_formula(
                bindings,
                &gate_up_scales,
                &down_scales,
                &replay,
                &mut report,
            )?;
        }
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement_input(batch, &first, &replay)?;
        verify_inactive(batch, &replay, &mut report)?;
        verify_inactive(batch, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn make_input(batch: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..batch * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / Qwen38_27B::HIDDEN) & 15]))
        .collect()
}

fn verify_seams(
    batch: usize,
    input: &[u16],
    input_norm: &[u16],
    next_norm: &[u16],
    observed: &DenseFp8MlpObservables,
    report: &mut DenseFp8MlpQualification,
) -> Result<(), DenseFp8MlpQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;

    for token in 0..batch {
        let hidden_begin = token * hidden;
        let hidden_end = hidden_begin + hidden;
        let intermediate_begin = token * intermediate;
        let intermediate_end = intermediate_begin + intermediate;
        let normalized =
            rms_norm_oracle::<Qwen38_27B>(&input[hidden_begin..hidden_end], input_norm);
        compare_close_slice(
            "pre-MLP RMSNorm",
            batch,
            token,
            &observed.normalized[hidden_begin..hidden_end],
            &normalized,
            &mut report.maximum_absolute_error,
        )?;

        let gate_up_oracle = quantize_oracle(&observed.normalized[hidden_begin..hidden_end])
            .map_err(DenseFp8MlpQualificationError::Mismatch)?;
        compare_codes(
            "gate/up activation",
            batch,
            token,
            &observed.gate_up_activation_codes[hidden_begin..hidden_end],
            &gate_up_oracle,
            observed.gate_up_activation_scales[token],
        )?;

        let down_oracle = quantize_oracle(&observed.swiglu[intermediate_begin..intermediate_end])
            .map_err(DenseFp8MlpQualificationError::Mismatch)?;
        compare_codes(
            "down activation",
            batch,
            token,
            &observed.down_activation_codes[intermediate_begin..intermediate_end],
            &down_oracle,
            observed.down_activation_scales[token],
        )?;

        let residual = input[hidden_begin..hidden_end]
            .iter()
            .zip(&observed.branch[hidden_begin..hidden_end])
            .map(|(&input, &branch)| f32_to_bf16(bf16_to_f32(input) + bf16_to_f32(branch)))
            .collect::<Vec<_>>();
        if observed.residual_output[hidden_begin..hidden_end] != residual {
            let relative = observed.residual_output[hidden_begin..hidden_end]
                .iter()
                .zip(&residual)
                .position(|(actual, expected)| actual != expected)
                .expect("slices differ");
            return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                "residual publication at B={batch}, row={token}, column={relative} differs"
            )));
        }
        let next = rms_norm_oracle::<Qwen38_27B>(&residual, next_norm);
        compare_close_slice(
            "next RMSNorm",
            batch,
            token,
            &observed.next_normalized[hidden_begin..hidden_end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.normalized_values += batch * hidden;
    report.activation_codes += batch * (hidden + intermediate);
    report.activation_scales += batch * 2;
    report.boundary_values += batch * hidden * 2;

    Ok(())
}

fn verify_source_formula(
    bindings: DenseFp8MlpBindings<'_>,
    gate_up_scales: &[u16],
    down_scales: &[u16],
    observed: &DenseFp8MlpObservables,
    report: &mut DenseFp8MlpQualification,
) -> Result<(), DenseFp8MlpQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let intermediate = Qwen38_27B::INTERMEDIATE;
    let activation = quantize_oracle(&observed.normalized[..hidden])
        .map_err(DenseFp8MlpQualificationError::Mismatch)?;
    for row in 0..intermediate {
        let gate_begin = row * hidden;
        let up_begin = (intermediate + row) * hidden;
        let gate = fp8_dot(
            &activation,
            &bindings.gate_up.weight_e4m3[gate_begin..gate_begin + hidden],
            gate_up_scales[row],
        )?;
        let up = fp8_dot(
            &activation,
            &bindings.gate_up.weight_e4m3[up_begin..up_begin + hidden],
            gate_up_scales[intermediate + row],
        )?;
        let expected = gate / (1.0 + (-gate).exp()) * up;
        let actual = bf16_to_f32(observed.swiglu[row]);
        require_close(
            "source SwiGLU",
            row,
            actual,
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }

    let down_activation = quantize_oracle(&observed.swiglu[..intermediate])
        .map_err(DenseFp8MlpQualificationError::Mismatch)?;
    for (row, &weight_scale) in down_scales.iter().enumerate().take(hidden) {
        let begin = row * intermediate;
        let expected = fp8_dot(
            &down_activation,
            &bindings.down.weight.codes()[begin..begin + intermediate],
            weight_scale,
        )?;
        let actual = bf16_to_f32(observed.branch[row]);
        require_close(
            "source down projection",
            row,
            actual,
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.source_swiglu_values += intermediate;
    report.source_branch_values += hidden;

    Ok(())
}

fn fp8_dot(
    activation: &TokenOracle,
    weights: &[u8],
    weight_scale: u16,
) -> Result<f64, DenseFp8MlpQualificationError> {
    let sum = activation
        .codes
        .iter()
        .zip(weights)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            let activation = decode_e4m3fn(activation)?;
            let weight = decode_e4m3fn(weight)?;
            Ok::<_, String>(sum + f64::from(activation) * f64::from(weight))
        })
        .map_err(DenseFp8MlpQualificationError::Mismatch)?;

    Ok(sum * f64::from(activation.scale) * f64::from(bf16_to_f32(weight_scale)))
}

fn compare_codes(
    role: &str,
    batch: usize,
    token: usize,
    actual_codes: &[u8],
    oracle: &TokenOracle,
    actual_scale: f32,
) -> Result<(), DenseFp8MlpQualificationError> {
    if let Some(column) = actual_codes
        .iter()
        .zip(&oracle.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "{role} code at B={batch}, row={token}, column={column}: device={:#04x}, oracle={:#04x}",
            actual_codes[column], oracle.codes[column]
        )));
    }
    if actual_scale.to_bits() != oracle.scale.to_bits() {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "{role} scale at B={batch}, row={token}: device={:#010x}, oracle={:#010x}",
            actual_scale.to_bits(),
            oracle.scale.to_bits()
        )));
    }

    Ok(())
}

fn compare_close_slice(
    role: &str,
    batch: usize,
    token: usize,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), DenseFp8MlpQualificationError> {
    for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = bf16_to_f32(actual);
        let expected = f64::from(bf16_to_f32(expected));
        require_close(
            &format!("{role} at B={batch}, row={token}"),
            column,
            actual,
            expected,
            maximum,
        )?;
    }

    Ok(())
}

fn require_close(
    role: &str,
    column: usize,
    actual: f32,
    expected: f64,
    maximum: &mut f32,
) -> Result<(), DenseFp8MlpQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.125f32.max(expected.abs() as f32 * 0.015);
    if error > tolerance {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "{role}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &DenseFp8MlpObservables,
    replay: &DenseFp8MlpObservables,
    report: &mut DenseFp8MlpQualification,
) -> Result<(), DenseFp8MlpQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                    "B={batch} graph plane `{}` differs at value {index}",
                    stringify!($field)
                )));
            }
        };
    }
    same!(residual_input);
    same!(normalized);
    same!(gate_up_activation_codes);
    same!(gate_up_activation_scales);
    same!(swiglu);
    same!(down_activation_codes);
    same!(down_activation_scales);
    same!(branch);
    same!(residual_output);
    same!(next_normalized);

    report.graph_replay_values +=
        batch * (6 * Qwen38_27B::HIDDEN + 2 * Qwen38_27B::INTERMEDIATE + 2);

    Ok(())
}

fn verify_replacement_input(
    batch: usize,
    first: &DenseFp8MlpObservables,
    replay: &DenseFp8MlpObservables,
) -> Result<(), DenseFp8MlpQualificationError> {
    let active = batch * Qwen38_27B::HIDDEN;
    if first.residual_input[..active] == replay.residual_input[..active]
        || first.residual_output[..active] == replay.residual_output[..active]
    {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement residual rows"
        )));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &DenseFp8MlpObservables,
    report: &mut DenseFp8MlpQualification,
) -> Result<(), DenseFp8MlpQualificationError> {
    let hidden_begin = batch * Qwen38_27B::HIDDEN;
    let intermediate_begin = batch * Qwen38_27B::INTERMEDIATE;
    for (role, values) in [
        ("normalized", &observed.normalized[hidden_begin..]),
        ("branch", &observed.branch[hidden_begin..]),
        ("residual output", &observed.residual_output[hidden_begin..]),
        ("next normalized", &observed.next_normalized[hidden_begin..]),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BF16_SENTINEL) {
            return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} value {relative}"
            )));
        }
    }
    if let Some(relative) = observed.swiglu[intermediate_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "B={batch} modified inactive SwiGLU value {relative}"
        )));
    }
    for (role, values) in [
        (
            "gate/up activation",
            &observed.gate_up_activation_codes[hidden_begin..],
        ),
        (
            "down activation",
            &observed.down_activation_codes[intermediate_begin..],
        ),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BYTE_SENTINEL) {
            return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} code {relative}"
            )));
        }
    }
    for (role, values) in [
        (
            "gate/up activation",
            &observed.gate_up_activation_scales[batch..],
        ),
        ("down activation", &observed.down_activation_scales[batch..]),
    ] {
        if let Some(relative) = values
            .iter()
            .position(|value| value.to_bits() != F32_SENTINEL_BITS)
        {
            return Err(DenseFp8MlpQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} scale {relative}"
            )));
        }
    }

    report.inactive_values +=
        (MAX_BATCH - batch) * (5 * Qwen38_27B::HIDDEN + 2 * Qwen38_27B::INTERMEDIATE + 2);

    Ok(())
}

fn verify_no_device_allocation(
    program: &DenseFp8MlpProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), DenseFp8MlpQualificationError> {
    program.replay(stream, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..4 {
        for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
            program.replay(stream, batch)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(DenseFp8MlpQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn little_endian_words(bytes: &[u8]) -> Result<Vec<u16>, DenseFp8MlpQualificationError> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(DenseFp8MlpQualificationError::Mismatch(
            "source BF16 scale plane has an odd byte length".to_string(),
        ));
    }

    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{DenseFp8MlpQualificationError, qualify_dense_fp8_mlp};

    #[test]
    #[ignore = "requires TUISKO_SNAPSHOT and an idle NVIDIA compute-capability 12.0 device"]
    fn source_layer60_matches_complete_oracles_and_graph_replay()
    -> Result<(), DenseFp8MlpQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            DenseFp8MlpQualificationError::Mismatch(
                "TUISKO_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_dense_fp8_mlp(std::path::Path::new(&root))?;

        assert_eq!(report.source_swiglu_values, 17_408);
        assert_eq!(report.source_branch_values, 5_120);
        assert!(report.normalized_values > 0);
        assert!(report.activation_codes > 0);
        assert!(report.activation_scales > 0);
        assert!(report.boundary_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.maximum_absolute_error <= 0.125);

        Ok(())
    }
}

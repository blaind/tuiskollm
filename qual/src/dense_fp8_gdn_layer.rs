//! Source-backed qualification for one resident dense-FP8 GDN layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, TokenOracle, bf16_to_f32, decode_e4m3fn,
    f32_to_bf16, quantize_oracle,
};
use crate::oracles::norm::residual_oracle;
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{DenseFp8GdnLayerObservables, DenseFp8GdnLayerProgram, EngineError, MAX_BATCH};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, DenseFp8MlpBindings, GdnBindings, Qwen38_27B,
};

const SOURCE_LAYER: usize = 60;
const MAX_ROWS: usize = 1_024;
const EXACT_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, MAX_ROWS];
const HEAD_DIM: usize = 128;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const RMS_EPSILON: f64 = 1.0e-6;
const DELTA_SCALE: f64 = 0.088_388_35;

/// Failure of the complete dense-FP8 GDN layer qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum DenseFp8GdnLayerQualificationError {
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
    /// Device behavior disagreed with an independently derived seam.
    #[error("dense-FP8 GDN layer qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts and worst error from one complete source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseFp8GdnLayerQualification {
    /// Residual and normalization values checked at every exact route.
    pub boundary_values: usize,
    /// Dynamic E4M3 codes checked bit-exactly at all four quantization seams.
    pub activation_codes: usize,
    /// Dynamic FP32 scales checked bit-exactly at all four quantization seams.
    pub activation_scales: usize,
    /// Real-source mixer and MLP values checked through the complete B=1 formula.
    pub source_values: usize,
    /// Active working, history, and state values reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Inactive values verified unchanged.
    pub inactive_values: usize,
    /// Immutable tensor-map words checked after every route.
    pub immutable_descriptor_words: usize,
    /// Exact source-backed device weight bytes.
    pub resident_weight_bytes: usize,
    /// Exact address-stable workspace and state bytes.
    pub workspace_bytes: usize,
    /// Exact resident weights plus workspace, excluding padding.
    pub owner_bytes: usize,
    /// Complete arena allocation bytes.
    pub arena_bytes: usize,
    /// Alignment bytes not assigned to an owner plane.
    pub padding_bytes: usize,
    /// Four address-bound tensor-map descriptor bytes.
    pub descriptor_bytes: usize,
    /// Largest absolute difference from a represented-value or FP64 seam oracle.
    pub maximum_absolute_error: f32,
}

struct SourcePlanes {
    input_scales: Vec<u16>,
    control_weights: Vec<u16>,
    convolution_weights: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    recurrent_norm: Vec<u16>,
    output_scales: Vec<u16>,
    input_norm: Vec<u16>,
    post_norm: Vec<u16>,
    gate_up_scales: Vec<u16>,
    down_scales: Vec<u16>,
    next_norm: Vec<u16>,
}

/// Qualifies source-backed layer 60 at every exact decode and prefill route.
pub fn qualify_dense_fp8_gdn_layer(
    root: &Path,
) -> Result<DenseFp8GdnLayerQualification, DenseFp8GdnLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let gdn = GdnBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let mlp = DenseFp8MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let sources = source_planes(gdn, mlp)?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = DenseFp8GdnLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let stable_descriptors = program.qualification_descriptors(&stream)?;
    if stable_addresses.len() != 47 {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "owner exposes {} addresses, expected 47",
            stable_addresses.len()
        )));
    }
    if program.resident_weight_bytes() != 383_949_248
        || program.workspace_bytes() != 272_482_336
        || program.arena_bytes() != 656_432_128
        || program.descriptor_bytes() != 768
    {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "owner byte accounting differs from the admitted layout: weights {} workspace {} arena {} descriptors {}",
            program.resident_weight_bytes(),
            program.workspace_bytes(),
            program.arena_bytes(),
            program.descriptor_bytes(),
        )));
    }
    let mut report = DenseFp8GdnLayerQualification {
        boundary_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_descriptor_words: 0,
        resident_weight_bytes: program.resident_weight_bytes(),
        workspace_bytes: program.workspace_bytes(),
        owner_bytes: program.owner_bytes(),
        arena_bytes: program.arena_bytes(),
        padding_bytes: program.arena_bytes() - program.owner_bytes(),
        descriptor_bytes: program.descriptor_bytes(),
        maximum_absolute_error: 0.0,
    };

    for rows in EXACT_ROUTES {
        let first_input = make_input(rows, 0);
        program.reset_state(&stream)?;
        program.load_residual(&stream, rows, &first_input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(rows, 1);
        program.reset_state(&stream)?;
        program.load_residual(&stream, rows, &input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;

        program.reset_state(&stream)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;

        verify_boundaries(rows, &input, &sources, &replay, &mut report)?;
        verify_quantization(rows, &replay, &mut report)?;
        if rows == 1 {
            verify_source_formula(gdn, mlp, &sources, &replay, &mut report)?;
        }
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &replay, &mut report)?;
        verify_inactive(rows, &eager, &mut report)?;
        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying rows={rows}"
            )));
        }
        let descriptors = program.qualification_descriptors(&stream)?;
        if descriptors != stable_descriptors {
            return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                "tensor-map descriptors changed while qualifying rows={rows}"
            )));
        }
        report.immutable_descriptor_words += descriptors.iter().map(Vec::len).sum::<usize>();
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn source_planes(
    gdn: GdnBindings<'_>,
    mlp: DenseFp8MlpBindings<'_>,
) -> Result<SourcePlanes, DenseFp8GdnLayerQualificationError> {
    let mut control_weights = gdn.a_control_weight.words().collect::<Vec<_>>();
    control_weights.extend(gdn.b_control_weight.words());
    Ok(SourcePlanes {
        input_scales: little_endian_words(gdn.input_scale_bf16)?,
        control_weights,
        convolution_weights: gdn.convolution_weight.words().collect(),
        a_log: gdn.a_log.words().collect(),
        dt_bias: gdn.dt_bias.words().collect(),
        recurrent_norm: gdn.norm.words().collect(),
        output_scales: gdn.output_scale.words().collect(),
        input_norm: gdn.input_norm.words().collect(),
        post_norm: gdn.post_attention_norm.words().collect(),
        gate_up_scales: little_endian_words(mlp.gate_up.scale_bf16)?,
        down_scales: mlp.down.scale.words().collect(),
        next_norm: mlp.next_norm.words().collect(),
    })
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

fn verify_boundaries(
    batch: usize,
    input: &[u16],
    sources: &SourcePlanes,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for token in 0..batch {
        let begin = token * hidden;
        let end = begin + hidden;
        let mixer_normalized =
            rms_norm_oracle::<Qwen38_27B>(&input[begin..end], &sources.input_norm);
        compare_bf16_slice(
            "mixer RMSNorm",
            &observed.mixer_normalized[begin..end],
            &mixer_normalized,
            &mut report.maximum_absolute_error,
        )?;
        let mixer_residual =
            residual_oracle(&input[begin..end], &observed.mixer_branch[begin..end]);
        compare_exact(
            "mixer residual",
            &observed.mixer_residual[begin..end],
            &mixer_residual,
        )?;
        let mlp_normalized = rms_norm_oracle::<Qwen38_27B>(&mixer_residual, &sources.post_norm);
        compare_bf16_slice(
            "post-mixer RMSNorm",
            &observed.mlp_normalized[begin..end],
            &mlp_normalized,
            &mut report.maximum_absolute_error,
        )?;
        let residual = residual_oracle(
            &observed.mixer_residual[begin..end],
            &observed.mlp_branch[begin..end],
        );
        compare_exact(
            "layer residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        let next = rms_norm_oracle::<Qwen38_27B>(&residual, &sources.next_norm);
        compare_bf16_slice(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }
    report.boundary_values += batch * hidden * 5;
    Ok(())
}

fn verify_quantization(
    batch: usize,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    for token in 0..batch {
        check_quantized(
            "GDN input",
            token,
            Qwen38_27B::HIDDEN,
            &observed.mixer_normalized,
            &observed.input_activation_codes,
            &observed.input_activation_scales,
        )?;
        check_quantized(
            "GDN output",
            token,
            Qwen38_27B::GDN_VALUE_ROWS,
            &observed.recurrent_output,
            &observed.output_activation_codes,
            &observed.output_activation_scales,
        )?;
        check_quantized(
            "gate/up",
            token,
            Qwen38_27B::HIDDEN,
            &observed.mlp_normalized,
            &observed.gate_up_activation_codes,
            &observed.gate_up_activation_scales,
        )?;
        check_quantized(
            "down",
            token,
            Qwen38_27B::INTERMEDIATE,
            &observed.swiglu,
            &observed.down_activation_codes,
            &observed.down_activation_scales,
        )?;
    }
    report.activation_codes +=
        batch * (2 * Qwen38_27B::HIDDEN + Qwen38_27B::GDN_VALUE_ROWS + Qwen38_27B::INTERMEDIATE);
    report.activation_scales += batch * 4;
    Ok(())
}

fn check_quantized(
    role: &str,
    token: usize,
    width: usize,
    input: &[u16],
    codes: &[u8],
    scales: &[f32],
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let begin = token * width;
    let oracle = quantize_oracle(&input[begin..begin + width])
        .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    if let Some(column) = codes[begin..begin + width]
        .iter()
        .zip(&oracle.codes)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "{role} code at token={token}, column={column} differs"
        )));
    }
    if scales[token].to_bits() != oracle.scale.to_bits() {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "{role} scale at token={token} differs"
        )));
    }
    Ok(())
}

fn verify_source_formula(
    gdn: GdnBindings<'_>,
    mlp: DenseFp8MlpBindings<'_>,
    sources: &SourcePlanes,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let input_activation = quantize_oracle(&observed.mixer_normalized[..Qwen38_27B::HIDDEN])
        .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    verify_fp8_projection(
        "GDN input projection",
        &input_activation,
        gdn.input_weight_e4m3,
        &sources.input_scales,
        &observed.projected[..Qwen38_27B::GDN_INPUT_ROWS],
        Qwen38_27B::HIDDEN,
        &mut report.maximum_absolute_error,
    )?;
    verify_controls(sources, observed, report)?;
    verify_convolution(sources, observed, report)?;
    verify_recurrence(sources, observed, report)?;

    let output_activation =
        quantize_oracle(&observed.recurrent_output[..Qwen38_27B::GDN_VALUE_ROWS])
            .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    verify_fp8_projection(
        "GDN output projection",
        &output_activation,
        gdn.output_weight.codes(),
        &sources.output_scales,
        &observed.mixer_branch[..Qwen38_27B::HIDDEN],
        Qwen38_27B::GDN_VALUE_ROWS,
        &mut report.maximum_absolute_error,
    )?;

    let gate_up_activation = quantize_oracle(&observed.mlp_normalized[..Qwen38_27B::HIDDEN])
        .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    for row in 0..Qwen38_27B::INTERMEDIATE {
        let gate_begin = row * Qwen38_27B::HIDDEN;
        let up_begin = (Qwen38_27B::INTERMEDIATE + row) * Qwen38_27B::HIDDEN;
        let gate = fp8_dot(
            &gate_up_activation,
            &mlp.gate_up.weight_e4m3[gate_begin..gate_begin + Qwen38_27B::HIDDEN],
            sources.gate_up_scales[row],
        )?;
        let up = fp8_dot(
            &gate_up_activation,
            &mlp.gate_up.weight_e4m3[up_begin..up_begin + Qwen38_27B::HIDDEN],
            sources.gate_up_scales[Qwen38_27B::INTERMEDIATE + row],
        )?;
        require_close(
            "source SwiGLU",
            row,
            bf16_to_f32(observed.swiglu[row]),
            gate / (1.0 + (-gate).exp()) * up,
            &mut report.maximum_absolute_error,
        )?;
    }
    let down_activation = quantize_oracle(&observed.swiglu[..Qwen38_27B::INTERMEDIATE])
        .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    verify_fp8_projection(
        "dense-FP8 down projection",
        &down_activation,
        mlp.down.weight.codes(),
        &sources.down_scales,
        &observed.mlp_branch[..Qwen38_27B::HIDDEN],
        Qwen38_27B::INTERMEDIATE,
        &mut report.maximum_absolute_error,
    )?;

    report.source_values += Qwen38_27B::GDN_INPUT_ROWS
        + 2 * Qwen38_27B::GDN_CONTROL_ROWS
        + 4 * Qwen38_27B::GDN_QKV_ROWS
        + STATE_PER_ROW
        + Qwen38_27B::GDN_VALUE_ROWS
        + Qwen38_27B::HIDDEN
        + Qwen38_27B::INTERMEDIATE
        + Qwen38_27B::HIDDEN;
    Ok(())
}

fn verify_fp8_projection(
    role: &str,
    activation: &TokenOracle,
    weights: &[u8],
    scales: &[u16],
    actual: &[u16],
    columns: usize,
    maximum: &mut f32,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    for (row, (&scale, &actual)) in scales.iter().zip(actual).enumerate() {
        let begin = row * columns;
        let expected = fp8_dot(activation, &weights[begin..begin + columns], scale)?;
        require_close(role, row, bf16_to_f32(actual), expected, maximum)?;
    }
    Ok(())
}

fn fp8_dot(
    activation: &TokenOracle,
    weights: &[u8],
    weight_scale: u16,
) -> Result<f64, DenseFp8GdnLayerQualificationError> {
    let sum = activation
        .codes
        .iter()
        .zip(weights)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            Ok::<_, String>(
                sum + f64::from(decode_e4m3fn(activation)?) * f64::from(decode_e4m3fn(weight)?),
            )
        })
        .map_err(DenseFp8GdnLayerQualificationError::Mismatch)?;
    Ok(sum * f64::from(activation.scale) * f64::from(bf16_to_f32(weight_scale)))
}

fn verify_controls(
    sources: &SourcePlanes,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    for row in 0..2 * VALUE_HEADS {
        let begin = row * Qwen38_27B::HIDDEN;
        let sum = observed.mixer_normalized[..Qwen38_27B::HIDDEN]
            .iter()
            .zip(&sources.control_weights[begin..begin + Qwen38_27B::HIDDEN])
            .map(|(&input, &weight)| f64::from(bf16_to_f32(input)) * f64::from(bf16_to_f32(weight)))
            .sum::<f64>();
        let (actual, expected) = if row < VALUE_HEADS {
            let control = sum + f64::from(bf16_to_f32(sources.dt_bias[row]));
            let softplus = if control > 20.0 {
                control
            } else {
                (1.0 + control.exp()).ln()
            };
            (
                observed.log_decay[row],
                -f64::from(bf16_to_f32(sources.a_log[row])).exp() * softplus,
            )
        } else {
            (observed.beta[row - VALUE_HEADS], 1.0 / (1.0 + (-sum).exp()))
        };
        require_f32_close("GDN control", row, actual, expected, 0.002, report)?;
    }
    Ok(())
}

fn verify_convolution(
    sources: &SourcePlanes,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    for channel in 0..Qwen38_27B::GDN_QKV_ROWS {
        let current = bf16_to_f32(observed.projected[channel]);
        let weight = bf16_to_f32(sources.convolution_weights[channel * 4 + 3]);
        let sum = f64::from(current) * f64::from(weight);
        let expected = sum / (1.0 + (-sum).exp());
        require_close(
            "GDN convolution",
            channel,
            bf16_to_f32(observed.convolved[channel]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
        let history = channel * 3;
        if observed.history[history] != 0
            || observed.history[history + 1] != 0
            || observed.history[history + 2] != observed.projected[channel]
        {
            return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                "causal history differs at channel {channel}"
            )));
        }
    }
    Ok(())
}

fn verify_recurrence(
    sources: &SourcePlanes,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let mut state = vec![0.0f64; STATE_PER_ROW];
    let mut output = vec![0.0f64; Qwen38_27B::GDN_VALUE_ROWS];
    let mut query = vec![[0.0f64; HEAD_DIM]; KEY_HEADS];
    let mut key = vec![[0.0f64; HEAD_DIM]; KEY_HEADS];
    for head in 0..KEY_HEADS {
        for (plane, destination) in [(0, &mut query), (1, &mut key)] {
            let begin = plane * QK_WIDTH + head * HEAD_DIM;
            let sum = observed.convolved[begin..begin + HEAD_DIM]
                .iter()
                .map(|&bits| f64::from(bf16_to_f32(bits)).powi(2))
                .sum::<f64>();
            let inverse = 1.0 / (sum + RMS_EPSILON).sqrt();
            for (column, value) in destination[head].iter_mut().enumerate() {
                *value = f64::from(bf16_to_f32(observed.convolved[begin + column])) * inverse;
            }
        }
    }
    for value_head in 0..VALUE_HEADS {
        let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
        let decay = f64::from(observed.log_decay[value_head]).exp();
        let beta = f64::from(observed.beta[value_head]);
        let state_begin = value_head * HEAD_DIM * HEAD_DIM;
        let value_begin = 2 * QK_WIDTH + value_head * HEAD_DIM;
        let mut recurrent = [0.0f64; HEAD_DIM];
        for (row, recurrent_value) in recurrent.iter_mut().enumerate() {
            let row_begin = state_begin + row * HEAD_DIM;
            let state_key = (0..HEAD_DIM)
                .map(|column| state[row_begin + column] * key[key_head][column])
                .sum::<f64>();
            let update = beta
                * (f64::from(bf16_to_f32(observed.convolved[value_begin + row]))
                    - decay * state_key);
            for column in 0..HEAD_DIM {
                state[row_begin + column] =
                    decay * state[row_begin + column] + update * key[key_head][column];
                *recurrent_value += state[row_begin + column] * query[key_head][column];
            }
            *recurrent_value *= DELTA_SCALE;
        }
        let rms = (recurrent.iter().map(|value| value * value).sum::<f64>() / HEAD_DIM as f64
            + RMS_EPSILON)
            .sqrt();
        let gate_begin = Qwen38_27B::GDN_QKV_ROWS + value_head * HEAD_DIM;
        let output_begin = value_head * HEAD_DIM;
        for row in 0..HEAD_DIM {
            let gate = f64::from(bf16_to_f32(observed.projected[gate_begin + row]));
            output[output_begin + row] =
                recurrent[row] / rms * f64::from(bf16_to_f32(sources.recurrent_norm[row])) * gate
                    / (1.0 + (-gate).exp());
        }
    }
    for (index, (&actual, &expected)) in observed.state[..STATE_PER_ROW]
        .iter()
        .zip(&state)
        .enumerate()
    {
        require_f32_close("GDN state", index, actual, expected, 2.0e-4, report)?;
    }
    for (index, (&actual, &expected)) in observed.recurrent_output[..Qwen38_27B::GDN_VALUE_ROWS]
        .iter()
        .zip(&output)
        .enumerate()
    {
        require_close(
            "GDN recurrent output",
            index,
            bf16_to_f32(actual),
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }
    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &DenseFp8GdnLayerObservables,
    replay: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                    "rows={rows} graph plane `{}` differs at value {index}",
                    stringify!($field)
                )));
            }
        };
    }
    same!(mixer_normalized);
    same!(input_activation_codes);
    same!(input_activation_scales);
    same!(projected);
    same!(log_decay);
    same!(beta);
    same!(convolved);
    same!(history);
    same!(state);
    same!(recurrent_output);
    same!(output_activation_codes);
    same!(output_activation_scales);
    same!(mixer_branch);
    same!(mixer_residual);
    same!(mlp_normalized);
    same!(gate_up_activation_codes);
    same!(gate_up_activation_scales);
    same!(swiglu);
    same!(down_activation_codes);
    same!(down_activation_scales);
    same!(mlp_branch);
    same!(residual_output);
    same!(next_normalized);
    report.graph_replay_values += active_values(rows);
    Ok(())
}

fn active_state_rows(rows: usize) -> usize {
    if rows <= MAX_BATCH { rows } else { 1 }
}

fn active_values(rows: usize) -> usize {
    rows * (9 * Qwen38_27B::HIDDEN
        + Qwen38_27B::GDN_INPUT_ROWS
        + Qwen38_27B::GDN_QKV_ROWS
        + 2 * Qwen38_27B::GDN_VALUE_ROWS
        + 2 * Qwen38_27B::INTERMEDIATE
        + 2 * Qwen38_27B::GDN_CONTROL_ROWS
        + 4)
        + active_state_rows(rows) * (3 * Qwen38_27B::GDN_QKV_ROWS + STATE_PER_ROW)
}

fn verify_replacement_input(
    rows: usize,
    first: &DenseFp8GdnLayerObservables,
    replay: &DenseFp8GdnLayerObservables,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let active = rows * Qwen38_27B::HIDDEN;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "rows={rows} graph ignored replacement input"
        )));
    }
    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &DenseFp8GdnLayerObservables,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_u8 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BYTE_SENTINEL)
            {
                return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_f32 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
                    "rows={rows} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    let mut inactive = 0;
    inactive += sentinel_u16!(mixer_normalized, Qwen38_27B::HIDDEN);
    inactive += sentinel_u8!(input_activation_codes, Qwen38_27B::HIDDEN);
    inactive += sentinel_f32!(input_activation_scales, 1);
    inactive += sentinel_u16!(projected, Qwen38_27B::GDN_INPUT_ROWS);
    inactive += sentinel_f32!(log_decay, Qwen38_27B::GDN_CONTROL_ROWS);
    inactive += sentinel_f32!(beta, Qwen38_27B::GDN_CONTROL_ROWS);
    inactive += sentinel_u16!(convolved, Qwen38_27B::GDN_QKV_ROWS);
    inactive += sentinel_u16!(recurrent_output, Qwen38_27B::GDN_VALUE_ROWS);
    inactive += sentinel_u8!(output_activation_codes, Qwen38_27B::GDN_VALUE_ROWS);
    inactive += sentinel_f32!(output_activation_scales, 1);
    inactive += sentinel_u16!(mixer_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mixer_residual, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mlp_normalized, Qwen38_27B::HIDDEN);
    inactive += sentinel_u8!(gate_up_activation_codes, Qwen38_27B::HIDDEN);
    inactive += sentinel_f32!(gate_up_activation_scales, 1);
    inactive += sentinel_u16!(swiglu, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_u8!(down_activation_codes, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_f32!(down_activation_scales, 1);
    inactive += sentinel_u16!(mlp_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(residual_output, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(next_normalized, Qwen38_27B::HIDDEN);
    let state_begin = active_state_rows(rows) * STATE_PER_ROW;
    if observed.state[state_begin..]
        .iter()
        .any(|&value| value != 0.0)
    {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "rows={rows} modified inactive recurrent state"
        )));
    }
    let history_width = Qwen38_27B::GDN_QKV_ROWS * 3;
    let history_begin = active_state_rows(rows) * history_width;
    if observed.history[history_begin..]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "rows={rows} modified inactive causal history"
        )));
    }
    inactive += observed.state.len() - state_begin + observed.history.len() - history_begin;
    let expected = inactive_values(rows);
    if inactive != expected {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "rows={rows} inactive accounting is {inactive}, expected {expected}"
        )));
    }
    report.inactive_values += inactive;
    Ok(())
}

fn inactive_values(rows: usize) -> usize {
    let working_per_row = 9 * Qwen38_27B::HIDDEN
        + Qwen38_27B::GDN_INPUT_ROWS
        + Qwen38_27B::GDN_QKV_ROWS
        + 2 * Qwen38_27B::GDN_VALUE_ROWS
        + 2 * Qwen38_27B::INTERMEDIATE
        + 2 * Qwen38_27B::GDN_CONTROL_ROWS
        + 4;
    let state_per_row = 3 * Qwen38_27B::GDN_QKV_ROWS + STATE_PER_ROW;

    (MAX_ROWS - rows) * working_per_row + (MAX_BATCH - active_state_rows(rows)) * state_per_row
}

fn compare_exact(
    role: &str,
    actual: &[u16],
    expected: &[u16],
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn compare_bf16_slice(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            maximum,
        )?;
    }
    Ok(())
}

fn require_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    maximum: &mut f32,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.5f32.max(expected.abs() as f32 * 0.03);
    if error > tolerance {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn require_f32_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    absolute: f32,
    report: &mut DenseFp8GdnLayerQualification,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    report.maximum_absolute_error = report.maximum_absolute_error.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * 0.005);
    if error > tolerance {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn verify_no_device_allocation(
    program: &DenseFp8GdnLayerProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), DenseFp8GdnLayerQualificationError> {
    program.replay(stream, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in [1, 32, 8, 64, 3, 128, 6, MAX_ROWS, 2, 7, 4, 5] {
            program.replay(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

fn little_endian_words(bytes: &[u8]) -> Result<Vec<u16>, DenseFp8GdnLayerQualificationError> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(DenseFp8GdnLayerQualificationError::Mismatch(
            "source BF16 plane has an odd byte length".to_string(),
        ));
    }
    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        DenseFp8GdnLayerQualificationError, EXACT_ROUTES, MAX_ROWS, SOURCE_LAYER, STATE_PER_ROW,
        active_values, inactive_values, qualify_dense_fp8_gdn_layer,
    };
    use std::path::PathBuf;
    use tuisko_model::{Arch, Qwen38_27B};

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn source_layer60_matches_complete_seam_oracles_and_graph_replay()
    -> Result<(), DenseFp8GdnLayerQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            DenseFp8GdnLayerQualificationError::Mismatch(
                "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_dense_fp8_gdn_layer(&PathBuf::from(root))?;
        let active_rows = EXACT_ROUTES.iter().sum::<usize>();
        assert_eq!(SOURCE_LAYER, 60);
        assert_eq!(report.boundary_values, active_rows * 5 * Qwen38_27B::HIDDEN);
        assert_eq!(
            report.activation_codes,
            active_rows
                * (2 * Qwen38_27B::HIDDEN + Qwen38_27B::GDN_VALUE_ROWS + Qwen38_27B::INTERMEDIATE)
        );
        assert_eq!(report.activation_scales, 4 * active_rows);
        assert_eq!(
            report.source_values,
            Qwen38_27B::GDN_INPUT_ROWS
                + 2 * Qwen38_27B::GDN_CONTROL_ROWS
                + 4 * Qwen38_27B::GDN_QKV_ROWS
                + STATE_PER_ROW
                + Qwen38_27B::GDN_VALUE_ROWS
                + 2 * Qwen38_27B::HIDDEN
                + Qwen38_27B::INTERMEDIATE
        );
        assert_eq!(
            report.graph_replay_values,
            EXACT_ROUTES
                .iter()
                .map(|&rows| active_values(rows))
                .sum::<usize>()
        );
        assert_eq!(
            report.inactive_values,
            2 * EXACT_ROUTES
                .iter()
                .map(|&rows| inactive_values(rows))
                .sum::<usize>()
        );
        assert_eq!(report.immutable_descriptor_words, 1_152);
        assert_eq!(report.resident_weight_bytes, 383_949_248);
        assert_eq!(report.workspace_bytes, 272_482_336);
        assert_eq!(report.owner_bytes, 656_431_584);
        assert_eq!(report.arena_bytes, 656_432_128);
        assert_eq!(report.padding_bytes, 544);
        assert_eq!(report.descriptor_bytes, 768);
        assert!(report.maximum_absolute_error.is_finite());
        Ok(())
    }

    #[test]
    fn dense_fp8_gdn_layer_suite_route_inventory_is_exact() {
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024]);
        assert_eq!(MAX_ROWS, 1_024);
    }
}

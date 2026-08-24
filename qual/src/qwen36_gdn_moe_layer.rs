//! Source-backed qualification for one Qwen3.6 GDN plus MoE decoder layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, decode_e4m3fn, encode_e4m3fn,
    f32_to_bf16,
};
use crate::qwen36_moe_experts::nvfp4_dot;
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen36GdnMoeLayerImmutable, Qwen36GdnMoeLayerInputs,
    Qwen36GdnMoeLayerObservables, Qwen36GdnMoeLayerProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MaterializedQwen36Gdn, MaterializedQwen36MoeLayer,
    Qwen36GdnBindings, Qwen36Moe35B, Qwen36MoeLayerBindings,
};

const SOURCE_LAYER: usize = 0;
const HIDDEN: usize = Qwen36Moe35B::HIDDEN;
const INPUT_ROWS: usize = Qwen36Moe35B::GDN_INPUT_ROWS;
const QKV_ROWS: usize = Qwen36Moe35B::GDN_QKV_ROWS;
const VALUE_ROWS: usize = Qwen36Moe35B::GDN_VALUE_ROWS;
const CONTROL_ROWS: usize = Qwen36Moe35B::GDN_CONTROL_ROWS;
const HEAD_DIM: usize = Qwen36Moe35B::LINEAR_HEAD_DIM;
const KEY_HEADS: usize = Qwen36Moe35B::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = Qwen36Moe35B::LINEAR_VALUE_HEADS;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const EXPERTS: usize = Qwen36Moe35B::NUM_EXPERTS;
const TOP_K: usize = Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN;
const SLOTS: usize = TOP_K + 1;
const INTERMEDIATE: usize = Qwen36Moe35B::INTERMEDIATE;
const RMS_EPSILON: f64 = 1.0e-6;
const QUERY_SCALE: f64 = 0.088_388_35;
const MAX_ROWS: usize = 128;
const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, MAX_ROWS];

/// Failure of the complete source-backed Qwen3.6 GDN/MoE layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen36GdnMoeLayerQualificationError {
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

    /// Device behavior disagreed with an independent represented-value formula.
    #[error("Qwen3.6 GDN/MoE layer qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36GdnMoeLayerQualification {
    /// Residual and normalization values checked at every exact batch.
    pub boundary_values: usize,
    /// Real-source GDN and MoE values checked through B=1.
    pub source_values: usize,
    /// Mutable owner values reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and state values verified unchanged.
    pub inactive_values: usize,
    /// Immutable source/materialized device values proved unchanged.
    pub immutable_values: usize,
    /// Runtime-owned graph-input values proved unchanged.
    pub runtime_input_values: usize,
    /// Complete one-allocation owner bytes.
    pub arena_bytes: usize,
    /// Exact source-backed device weight bytes.
    pub weight_bytes: usize,
    /// Exact address-stable workspace and state bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the owner arena.
    pub padding_bytes: usize,
    /// Largest absolute difference from an accepted represented-value formula.
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Copy)]
struct SourceBindings<'a> {
    gdn: Qwen36GdnBindings<'a>,
    moe: &'a Qwen36MoeLayerBindings<'a>,
}

struct SourceMaterialized<'a> {
    gdn: MaterializedQwen36Gdn<'a>,
    moe: MaterializedQwen36MoeLayer<'a>,
}

/// Qualifies source-backed Qwen3.6 layer 0 at every exact decode and prefill route.
pub fn qualify_qwen36_gdn_moe_layer(
    root: &Path,
) -> Result<Qwen36GdnMoeLayerQualification, Qwen36GdnMoeLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen36Moe35B>::open(root)?);
    let gdn_binding = Qwen36GdnBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let moe_binding = Qwen36MoeLayerBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let source = SourceMaterialized {
        gdn: gdn_binding.materialize()?,
        moe: moe_binding.clone().materialize()?,
    };
    let bindings = SourceBindings {
        gdn: gdn_binding,
        moe: &moe_binding,
    };
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        Qwen36GdnMoeLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 47 {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "Qwen3.6 layer owner exposes {} addresses, expected 47",
            stable_addresses.len()
        )));
    }
    let mut report = Qwen36GdnMoeLayerQualification {
        boundary_values: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        runtime_input_values: 0,
        arena_bytes: program.arena_bytes(),
        weight_bytes: program.resident_weight_bytes(),
        workspace_bytes: program.workspace_bytes(),
        padding_bytes: program.arena_bytes()
            - program.resident_weight_bytes()
            - program.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };

    verify_scales(&program, &source)?;

    for rows in EXACT_ROUTES {
        let first_input = make_input(rows, 0);
        prepare_run(&program, &stream, rows, &first_input)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(rows, 1);
        prepare_run(&program, &stream, rows, &input)?;
        let replay_inputs = program.qualification_runtime_inputs(&stream)?;
        verify_runtime_input_contract(rows, &input, &replay_inputs)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;
        let replay_inputs_after = program.qualification_runtime_inputs(&stream)?;
        report.runtime_input_values +=
            verify_runtime_inputs_unchanged(rows, &replay_inputs, &replay_inputs_after)?;

        prepare_run(&program, &stream, rows, &input)?;
        let eager_inputs = program.qualification_runtime_inputs(&stream)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;
        let eager_inputs_after = program.qualification_runtime_inputs(&stream)?;
        report.runtime_input_values +=
            verify_runtime_inputs_unchanged(rows, &eager_inputs, &eager_inputs_after)?;

        verify_boundaries(rows, &input, bindings, &replay, &mut report)?;
        if rows == 1 {
            verify_source_formula(bindings, &source, &replay, &mut report)?;
        }
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &replay, &mut report)?;
        verify_inactive(rows, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                "Qwen3.6 layer owner addresses changed while qualifying {}",
                route_label(rows)
            )));
        }
    }

    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &source,
        &mut report,
    )?;
    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

fn make_input(rows: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..rows * HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / HIDDEN) & 15]))
        .collect()
}

fn prepare_run(
    program: &Qwen36GdnMoeLayerProgram,
    stream: &CudaStream,
    rows: usize,
    input: &[u16],
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    program.reset_state(stream)?;
    program.load_residual(stream, rows, input)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

fn verify_runtime_input_contract(
    rows: usize,
    input: &[u16],
    actual: &Qwen36GdnMoeLayerInputs,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    compare_exact(
        "runtime residual input",
        &actual.residual_input[..rows * HIDDEN],
        input,
    )?;
    compare_exact(
        "runtime state rows",
        &actual.state_rows,
        &(0..8u32).collect::<Vec<_>>(),
    )?;

    Ok(())
}

fn verify_runtime_inputs_unchanged(
    rows: usize,
    before: &Qwen36GdnMoeLayerInputs,
    after: &Qwen36GdnMoeLayerInputs,
) -> Result<usize, Qwen36GdnMoeLayerQualificationError> {
    compare_exact(
        &format!("{} immutable residual input", route_label(rows)),
        &after.residual_input,
        &before.residual_input,
    )?;
    compare_exact(
        &format!("{} immutable state rows", route_label(rows)),
        &after.state_rows,
        &before.state_rows,
    )?;

    Ok(after.residual_input.len() + after.state_rows.len())
}

fn verify_scales(
    program: &Qwen36GdnMoeLayerProgram,
    source: &SourceMaterialized<'_>,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let expected = [
        source.gdn.input_scale,
        source.gdn.input_weight_scales[0],
        source.gdn.input_weight_scales[1],
        source.gdn.output.input_scale,
        source.gdn.output.weight_scale,
        source.moe.shared_expert.gate_up_weight_scales_2[0],
        source.moe.shared_expert.down_weight_scales_2[0],
    ]
    .map(f32::to_bits);
    if program.qualification_source_scales().map(f32::to_bits) != expected {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(
            "resident static scales differ from the materialized source contract".to_string(),
        ));
    }

    Ok(())
}

fn verify_boundaries(
    batch: usize,
    input: &[u16],
    sources: SourceBindings<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let input_norm = sources.gdn.input_norm.words().collect::<Vec<_>>();
    let post_norm = sources.gdn.post_attention_norm.words().collect::<Vec<_>>();
    let next_norm = sources.moe.next_norm.words().collect::<Vec<_>>();

    for token in 0..batch {
        let begin = token * HIDDEN;
        let end = begin + HIDDEN;
        let mixer_normalized = rms_norm_oracle::<Qwen36Moe35B>(&input[begin..end], &input_norm);
        compare_bf16(
            "input RMSNorm",
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
        let moe_normalized = rms_norm_oracle::<Qwen36Moe35B>(&mixer_residual, &post_norm);
        compare_bf16(
            "post-mixer RMSNorm",
            &observed.moe_normalized[begin..end],
            &moe_normalized,
            &mut report.maximum_absolute_error,
        )?;

        let residual = residual_oracle(
            &observed.mixer_residual[begin..end],
            &observed.moe_branch[begin..end],
        );
        compare_exact(
            "layer residual",
            &observed.residual_output[begin..end],
            &residual,
        )?;
        let next = rms_norm_oracle::<Qwen36Moe35B>(&residual, &next_norm);
        compare_bf16(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }
    report.boundary_values += batch * HIDDEN * 5;

    Ok(())
}

fn verify_source_formula(
    bindings: SourceBindings<'_>,
    source: &SourceMaterialized<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let input = &observed.mixer_normalized[..HIDDEN];
    let input_codes = quantize_static(input, source.gdn.input_scale)?;
    compare_exact(
        "GDN input activation codes",
        &observed.input_activation_codes[..HIDDEN],
        &input_codes,
    )?;
    for row in 0..INPUT_ROWS {
        let scale = if row < QKV_ROWS {
            source.gdn.input_weight_scales[0]
        } else {
            source.gdn.input_weight_scales[1]
        };
        let expected = fp8_dot(
            &input_codes,
            &source.gdn.input_weight_e4m3[row * HIDDEN..(row + 1) * HIDDEN],
            source.gdn.input_scale,
            scale,
        )?;
        require_close(
            "source GDN input projection",
            row,
            bf16_to_f32(observed.projected[row]),
            expected,
            0.25,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }

    for (role, weight, begin) in [
        ("A control", bindings.gdn.a_control, 0),
        ("B control", bindings.gdn.b_control, CONTROL_ROWS),
    ] {
        let words = weight.words().collect::<Vec<_>>();
        for row in 0..CONTROL_ROWS {
            let expected = bf16_dot(input, &words[row * HIDDEN..(row + 1) * HIDDEN]);
            require_close(
                role,
                row,
                bf16_to_f32(observed.projected_controls[begin + row]),
                expected,
                0.25,
                0.025,
                &mut report.maximum_absolute_error,
            )?;
        }
    }
    verify_controls(bindings.gdn, observed, report)?;
    verify_convolution(bindings.gdn, observed, report)?;
    verify_recurrence(bindings.gdn, observed, report)?;

    let output_codes = quantize_static(
        &observed.recurrent_output[..VALUE_ROWS],
        source.gdn.output.input_scale,
    )?;
    compare_exact(
        "GDN output activation codes",
        &observed.output_activation_codes[..VALUE_ROWS],
        &output_codes,
    )?;
    for row in 0..HIDDEN {
        let expected = fp8_dot(
            &output_codes,
            &source.gdn.output.weight_e4m3[row * VALUE_ROWS..(row + 1) * VALUE_ROWS],
            source.gdn.output.input_scale,
            source.gdn.output.weight_scale,
        )?;
        require_close(
            "source GDN output projection",
            row,
            bf16_to_f32(observed.mixer_branch[row]),
            expected,
            0.25,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }
    verify_router(bindings.moe, observed, report)?;
    verify_experts(&source.moe, observed, report)?;

    report.source_values += HIDDEN
        + INPUT_ROWS
        + 2 * CONTROL_ROWS
        + 4 * QKV_ROWS
        + STATE_PER_ROW
        + 2 * VALUE_ROWS
        + HIDDEN
        + EXPERTS
        + 2 * TOP_K
        + SLOTS * (INTERMEDIATE + HIDDEN)
        + 1
        + HIDDEN;

    Ok(())
}

fn quantize_static(
    values: &[u16],
    scale: f32,
) -> Result<Vec<u8>, Qwen36GdnMoeLayerQualificationError> {
    values
        .iter()
        .map(|&bits| {
            encode_e4m3fn(bf16_to_f32(bits) / scale)
                .map_err(Qwen36GdnMoeLayerQualificationError::Mismatch)
        })
        .collect()
}

fn fp8_dot(
    activation: &[u8],
    weight: &[u8],
    activation_scale: f32,
    weight_scale: f32,
) -> Result<f64, Qwen36GdnMoeLayerQualificationError> {
    activation
        .iter()
        .zip(weight)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            let activation =
                decode_e4m3fn(activation).map_err(Qwen36GdnMoeLayerQualificationError::Mismatch)?;
            let weight =
                decode_e4m3fn(weight).map_err(Qwen36GdnMoeLayerQualificationError::Mismatch)?;
            Ok(sum + f64::from(activation) * f64::from(weight))
        })
        .map(|sum| sum * f64::from(activation_scale * weight_scale))
}

fn bf16_dot(left: &[u16], right: &[u16]) -> f64 {
    left.iter().zip(right).fold(0.0f64, |sum, (&left, &right)| {
        sum + f64::from(bf16_to_f32(left)) * f64::from(bf16_to_f32(right))
    })
}

fn verify_controls(
    source: Qwen36GdnBindings<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let a_log = source.a_log.words().collect::<Vec<_>>();
    let dt_bias = source.dt_bias.words().collect::<Vec<_>>();
    for row in 0..2 * VALUE_HEADS {
        let raw = f64::from(bf16_to_f32(observed.projected_controls[row]));
        let (actual, expected) = if row < VALUE_HEADS {
            let control = raw + f64::from(bf16_to_f32(dt_bias[row]));
            let softplus = if control > 20.0 {
                control
            } else {
                (1.0 + control.exp()).ln()
            };
            (
                observed.log_decay[row],
                -f64::from(bf16_to_f32(a_log[row])).exp() * softplus,
            )
        } else {
            (observed.beta[row - VALUE_HEADS], 1.0 / (1.0 + (-raw).exp()))
        };
        require_close(
            "GDN control",
            row,
            actual,
            expected,
            0.002,
            0.002,
            &mut report.maximum_absolute_error,
        )?;
    }

    Ok(())
}

fn verify_convolution(
    source: Qwen36GdnBindings<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let weights = source.convolution_weight.words().collect::<Vec<_>>();
    for channel in 0..QKV_ROWS {
        let sum = f64::from(bf16_to_f32(observed.projected[channel]))
            * f64::from(bf16_to_f32(weights[channel * 4 + 3]));
        let expected = sum / (1.0 + (-sum).exp());
        require_close(
            "GDN convolution",
            channel,
            bf16_to_f32(observed.convolved[channel]),
            expected,
            0.002,
            0.002,
            &mut report.maximum_absolute_error,
        )?;
        let history = channel * 3;
        if observed.history[history] != 0
            || observed.history[history + 1] != 0
            || observed.history[history + 2] != observed.projected[channel]
        {
            return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                "causal history differs at channel {channel}"
            )));
        }
    }

    Ok(())
}

fn verify_recurrence(
    source: Qwen36GdnBindings<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let norm = source.norm.words().collect::<Vec<_>>();
    let mut state = vec![0.0f64; STATE_PER_ROW];
    let mut output = vec![0.0f64; VALUE_ROWS];
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
            *recurrent_value *= QUERY_SCALE;
        }
        let rms = (recurrent.iter().map(|value| value * value).sum::<f64>() / HEAD_DIM as f64
            + RMS_EPSILON)
            .sqrt();
        let gate_begin = QKV_ROWS + value_head * HEAD_DIM;
        let output_begin = value_head * HEAD_DIM;
        for row in 0..HEAD_DIM {
            let gate = f64::from(bf16_to_f32(observed.projected[gate_begin + row]));
            output[output_begin + row] =
                recurrent[row] / rms * f64::from(bf16_to_f32(norm[row])) * gate
                    / (1.0 + (-gate).exp());
        }
    }
    for (index, (&actual, &expected)) in observed.state[..STATE_PER_ROW]
        .iter()
        .zip(&state)
        .enumerate()
    {
        require_close(
            "GDN state",
            index,
            actual,
            expected,
            2.0e-4,
            0.002,
            &mut report.maximum_absolute_error,
        )?;
    }
    for (index, (&actual, &expected)) in observed.recurrent_output[..VALUE_ROWS]
        .iter()
        .zip(&output)
        .enumerate()
    {
        require_close(
            "GDN recurrent output",
            index,
            bf16_to_f32(actual),
            expected,
            0.015_625,
            0.01,
            &mut report.maximum_absolute_error,
        )?;
    }

    Ok(())
}

fn verify_router(
    source: &Qwen36MoeLayerBindings<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let input = &observed.moe_normalized[..HIDDEN];
    let weights = source.router_weight.words().collect::<Vec<_>>();
    let mut logits = vec![0u16; EXPERTS];
    for expert in 0..EXPERTS {
        logits[expert] =
            f32_to_bf16(bf16_dot(input, &weights[expert * HIDDEN..(expert + 1) * HIDDEN]) as f32);
    }
    compare_exact("router logits", &observed.router_logits[..EXPERTS], &logits)?;
    let mut ranking = (0..EXPERTS).collect::<Vec<_>>();
    ranking.sort_unstable_by(|&left, &right| {
        bf16_to_f32(logits[right])
            .total_cmp(&bf16_to_f32(logits[left]))
            .then_with(|| left.cmp(&right))
    });
    let maximum = f64::from(bf16_to_f32(logits[ranking[0]]));
    let mut exponentials = [0.0f64; TOP_K];
    let mut denominator = 0.0f64;
    for position in 0..TOP_K {
        let expert = ranking[position];
        exponentials[position] = (f64::from(bf16_to_f32(logits[expert])) - maximum).exp();
        denominator += exponentials[position];
        if observed.expert_indices[position] != expert as u16 {
            return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                "router position {position}: device={}, oracle={expert}",
                observed.expert_indices[position]
            )));
        }
    }
    for (position, exponential) in exponentials.into_iter().enumerate() {
        let expected = f32_to_bf16((exponential / denominator) as f32);
        require_close(
            "routing weight",
            position,
            bf16_to_f32(observed.routing_weights[position]),
            f64::from(bf16_to_f32(expected)),
            0.001,
            0.001,
            &mut report.maximum_absolute_error,
        )?;
    }

    Ok(())
}

fn verify_experts(
    source: &MaterializedQwen36MoeLayer<'_>,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let input = &observed.moe_normalized[..HIDDEN];
    let mut expected_intermediate = vec![0u16; SLOTS * INTERMEDIATE];
    let mut expected_output = vec![0u16; SLOTS * HIDDEN];
    for position in 0..SLOTS {
        let routed = position < TOP_K;
        let expert = if routed {
            observed.expert_indices[position] as usize
        } else {
            0
        };
        let (gate_codes, gate_scales, gate_scale, gate_expert) = if routed {
            (
                source.experts.gate_up_weight_e2m1.as_slice(),
                source.experts.gate_up_scale_e4m3_swizzled.as_slice(),
                source.experts.gate_up_weight_scales_2[expert],
                expert,
            )
        } else {
            (
                source.shared_expert.gate_up_weight_e2m1.as_slice(),
                source.shared_expert.gate_up_scale_e4m3_swizzled.as_slice(),
                source.shared_expert.gate_up_weight_scales_2[0],
                0,
            )
        };
        for row in 0..INTERMEDIATE {
            let gate = nvfp4_dot(
                input,
                gate_codes,
                gate_scales,
                gate_expert,
                row,
                2 * INTERMEDIATE,
                HIDDEN,
                gate_scale,
            );
            let up = nvfp4_dot(
                input,
                gate_codes,
                gate_scales,
                gate_expert,
                row + INTERMEDIATE,
                2 * INTERMEDIATE,
                HIDDEN,
                gate_scale,
            );
            expected_intermediate[position * INTERMEDIATE + row] =
                f32_to_bf16((gate / (1.0 + (-gate).exp())) * up);
        }
        let intermediate =
            &expected_intermediate[position * INTERMEDIATE..(position + 1) * INTERMEDIATE];
        let (down_codes, down_scales, down_scale, down_expert) = if routed {
            (
                source.experts.down_weight_e2m1.as_slice(),
                source.experts.down_scale_e4m3_swizzled.as_slice(),
                source.experts.down_weight_scales_2[expert],
                expert,
            )
        } else {
            (
                source.shared_expert.down_weight_e2m1.as_slice(),
                source.shared_expert.down_scale_e4m3_swizzled.as_slice(),
                source.shared_expert.down_weight_scales_2[0],
                0,
            )
        };
        for row in 0..HIDDEN {
            expected_output[position * HIDDEN + row] = f32_to_bf16(nvfp4_dot(
                intermediate,
                down_codes,
                down_scales,
                down_expert,
                row,
                HIDDEN,
                INTERMEDIATE,
                down_scale,
            ));
        }
    }
    compare_bf16_tolerance(
        "expert intermediate",
        &observed.expert_intermediate[..SLOTS * INTERMEDIATE],
        &expected_intermediate,
        0.02,
        report,
    )?;
    compare_bf16_tolerance(
        "expert output",
        &observed.expert_output[..SLOTS * HIDDEN],
        &expected_output,
        0.04,
        report,
    )?;

    let gate_weights = source.shared_expert_gate_weight.words().collect::<Vec<_>>();
    let shared_gate = bf16_dot(input, &gate_weights);
    require_close(
        "shared expert gate",
        0,
        bf16_to_f32(observed.shared_gate[0]),
        shared_gate,
        0.002,
        0.002,
        &mut report.maximum_absolute_error,
    )?;
    let multiplier = 1.0 / (1.0 + (-(shared_gate as f32)).exp());
    for column in 0..HIDDEN {
        let mut sum = 0.0f32;
        for position in 0..TOP_K {
            let expert = bf16_to_f32(expected_output[position * HIDDEN + column]);
            let weight = bf16_to_f32(observed.routing_weights[position]);
            sum = expert.mul_add(weight, sum);
        }
        let shared = bf16_to_f32(expected_output[TOP_K * HIDDEN + column]);
        let expected = f32_to_bf16(shared.mul_add(multiplier, sum));
        require_close(
            "combined MoE output",
            column,
            bf16_to_f32(observed.moe_branch[column]),
            f64::from(bf16_to_f32(expected)),
            0.08,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }

    Ok(())
}

fn verify_replay(
    rows: usize,
    eager: &Qwen36GdnMoeLayerObservables,
    replay: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            compare_exact(
                &format!("{} graph plane `{}`", route_label(rows), stringify!($field)),
                &replay.$field,
                &eager.$field,
            )?;
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            compare_exact(
                &format!("{} graph plane `{}`", route_label(rows), stringify!($field)),
                &replay
                    .$field
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                &eager
                    .$field
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            )?;
        };
    }

    same!(mixer_normalized);
    same!(input_activation_codes);
    same!(projected);
    same!(projected_controls);
    same_f32!(log_decay);
    same_f32!(beta);
    same!(convolved);
    same!(history);
    same_f32!(state);
    same!(recurrent_output);
    same!(output_activation_codes);
    same!(mixer_branch);
    same!(mixer_residual);
    same!(moe_normalized);
    same!(router_logits);
    same!(expert_indices);
    same!(routing_weights);
    same!(expert_intermediate);
    same!(expert_output);
    same!(shared_gate);
    same!(moe_branch);
    same!(residual_output);
    same!(next_normalized);
    report.graph_replay_values += observable_values(replay);

    Ok(())
}

fn observable_values(values: &Qwen36GdnMoeLayerObservables) -> usize {
    values.mixer_normalized.len()
        + values.input_activation_codes.len()
        + values.projected.len()
        + values.projected_controls.len()
        + values.log_decay.len()
        + values.beta.len()
        + values.convolved.len()
        + values.history.len()
        + values.state.len()
        + values.recurrent_output.len()
        + values.output_activation_codes.len()
        + values.mixer_branch.len()
        + values.mixer_residual.len()
        + values.moe_normalized.len()
        + values.router_logits.len()
        + values.expert_indices.len()
        + values.routing_weights.len()
        + values.expert_intermediate.len()
        + values.expert_output.len()
        + values.shared_gate.len()
        + values.moe_branch.len()
        + values.residual_output.len()
        + values.next_normalized.len()
}

fn verify_replacement_input(
    rows: usize,
    first: &Qwen36GdnMoeLayerObservables,
    replay: &Qwen36GdnMoeLayerObservables,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let active = rows * HIDDEN;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{} graph ignored replacement residual rows",
            route_label(rows)
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Qwen36GdnMoeLayerObservables,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                    "{} modified inactive `{}` value",
                    route_label(rows),
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
                return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                    "{} modified inactive `{}` value",
                    route_label(rows),
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
                return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
                    "{} modified inactive `{}` value",
                    route_label(rows),
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }

    let mut inactive = 0;
    inactive += sentinel_u16!(mixer_normalized, HIDDEN);
    inactive += sentinel_u8!(input_activation_codes, HIDDEN);
    inactive += sentinel_u16!(projected, INPUT_ROWS);
    inactive += sentinel_u16!(projected_controls, 2 * CONTROL_ROWS);
    inactive += sentinel_f32!(log_decay, CONTROL_ROWS);
    inactive += sentinel_f32!(beta, CONTROL_ROWS);
    inactive += sentinel_u16!(convolved, QKV_ROWS);
    inactive += sentinel_u16!(recurrent_output, VALUE_ROWS);
    inactive += sentinel_u8!(output_activation_codes, VALUE_ROWS);
    inactive += sentinel_u16!(mixer_branch, HIDDEN);
    inactive += sentinel_u16!(mixer_residual, HIDDEN);
    inactive += sentinel_u16!(moe_normalized, HIDDEN);
    inactive += sentinel_u16!(router_logits, EXPERTS);
    inactive += sentinel_u16!(expert_indices, TOP_K);
    inactive += sentinel_u16!(routing_weights, TOP_K);
    inactive += sentinel_u16!(expert_intermediate, SLOTS * INTERMEDIATE);
    inactive += sentinel_u16!(expert_output, SLOTS * HIDDEN);
    inactive += sentinel_u16!(shared_gate, 1);
    inactive += sentinel_u16!(moe_branch, HIDDEN);
    inactive += sentinel_u16!(residual_output, HIDDEN);
    inactive += sentinel_u16!(next_normalized, HIDDEN);

    let state_slots = if rows <= MAX_BATCH { rows } else { 1 };
    let history_begin = state_slots * QKV_ROWS * 3;
    if observed.history[history_begin..]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{} modified inactive history",
            route_label(rows)
        )));
    }
    inactive += observed.history.len() - history_begin;
    let state_begin = state_slots * STATE_PER_ROW;
    if observed.state[state_begin..]
        .iter()
        .any(|&value| value.to_bits() != 0)
    {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{} modified inactive recurrent state",
            route_label(rows)
        )));
    }
    inactive += observed.state.len() - state_begin;
    report.inactive_values += inactive;

    Ok(())
}

fn verify_immutable(
    actual: &Qwen36GdnMoeLayerImmutable,
    source: &SourceMaterialized<'_>,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let control_words = bytes_to_words(&source.gdn.control_weight_bf16)?;
    macro_rules! same {
        ($field:ident, $expected:expr) => {{
            let expected = $expected;
            compare_exact(stringify!($field), &actual.$field, expected)?;
            report.immutable_values += actual.$field.len();
        }};
    }

    same!(
        input_norm,
        &source.gdn.input_norm.words().collect::<Vec<_>>()
    );
    same!(input_weight_codes, &source.gdn.input_weight_e4m3);
    same!(control_weight_bf16, &control_words);
    same!(a_log, &source.gdn.a_log.words().collect::<Vec<_>>());
    same!(dt_bias, &source.gdn.dt_bias.words().collect::<Vec<_>>());
    same!(
        convolution_weights,
        &source.gdn.convolution_weight.words().collect::<Vec<_>>()
    );
    same!(recurrent_norm, &source.gdn.norm.words().collect::<Vec<_>>());
    same!(output_weight_codes, source.gdn.output.weight_e4m3);
    same!(
        post_attention_norm,
        &source.gdn.post_attention_norm.words().collect::<Vec<_>>()
    );
    same!(
        router_weight,
        &source.moe.router_weight.words().collect::<Vec<_>>()
    );
    same!(
        routed_gate_up_codes,
        &source.moe.experts.gate_up_weight_e2m1
    );
    same!(
        routed_gate_up_scales,
        &source.moe.experts.gate_up_scale_e4m3_swizzled
    );
    same!(
        routed_gate_up_weight_scales_2,
        &source.moe.experts.gate_up_weight_scales_2
    );
    same!(routed_down_codes, &source.moe.experts.down_weight_e2m1);
    same!(
        routed_down_scales,
        &source.moe.experts.down_scale_e4m3_swizzled
    );
    same!(
        routed_down_weight_scales_2,
        &source.moe.experts.down_weight_scales_2
    );
    same!(
        shared_gate_up_codes,
        &source.moe.shared_expert.gate_up_weight_e2m1
    );
    same!(
        shared_gate_up_scales,
        &source.moe.shared_expert.gate_up_scale_e4m3_swizzled
    );
    same!(
        shared_down_codes,
        &source.moe.shared_expert.down_weight_e2m1
    );
    same!(
        shared_down_scales,
        &source.moe.shared_expert.down_scale_e4m3_swizzled
    );
    same!(
        shared_gate_weight,
        &source
            .moe
            .shared_expert_gate_weight
            .words()
            .collect::<Vec<_>>()
    );
    same!(next_norm, &source.moe.next_norm.words().collect::<Vec<_>>());

    Ok(())
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u16>, Qwen36GdnMoeLayerQualificationError> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(
            "BF16 source plane has an odd byte length".to_string(),
        ));
    }
    Ok(words.iter().map(|word| u16::from_le_bytes(*word)).collect())
}

fn verify_no_device_allocation(
    program: &Qwen36GdnMoeLayerProgram,
    stream: &CudaStream,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    program.replay(stream, MAX_ROWS)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..2 {
        for rows in [128, 1, 64, 8, 32, 3, 6, 2, 7, 4, 5] {
            program.replay(stream, rows)?;
        }
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn residual_oracle(input: &[u16], branch: &[u16]) -> Vec<u16> {
    input
        .iter()
        .zip(branch)
        .map(|(&input, &branch)| f32_to_bf16(bf16_to_f32(input) + bf16_to_f32(branch)))
        .collect()
}

fn route_label(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    require_same_len(role, actual.len(), expected.len())?;
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }

    Ok(())
}

fn compare_bf16(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    maximum: &mut f32,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    require_same_len(role, actual.len(), expected.len())?;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            0.25,
            0.025,
            maximum,
        )?;
    }

    Ok(())
}

fn compare_bf16_tolerance(
    role: &str,
    actual: &[u16],
    expected: &[u16],
    tolerance: f32,
    report: &mut Qwen36GdnMoeLayerQualification,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    require_same_len(role, actual.len(), expected.len())?;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            role,
            index,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
            tolerance,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }

    Ok(())
}

fn require_same_len(
    role: &str,
    actual: usize,
    expected: usize,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    if actual != expected {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{role} has {actual} values, expected {expected}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    absolute: f32,
    relative: f32,
    maximum: &mut f32,
) -> Result<(), Qwen36GdnMoeLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * relative);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen36GdnMoeLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_layer_and_owner_accounting_are_exact() {
        assert_eq!(SOURCE_LAYER, 0);
        assert_eq!(Qwen36Moe35B::GDN_INPUT_ROWS, 12_288);
        assert_eq!(Qwen36Moe35B::NUM_EXPERTS_PER_TOKEN, 8);
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer0_matches_complete_oracles_and_graph_replay()
    -> Result<(), Qwen36GdnMoeLayerQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT").ok_or_else(|| {
            Qwen36GdnMoeLayerQualificationError::Mismatch(
                "TUISKO_QWEN36_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_qwen36_gdn_moe_layer(Path::new(&root))?;

        assert_eq!(report.boundary_values, 2_662_400);
        assert_eq!(report.weight_bytes, 489_703_808);
        assert_eq!(report.workspace_bytes, 34_459_936);
        assert_eq!(report.arena_bytes, 524_164_352);
        assert_eq!(report.padding_bytes, 608);
        assert!(report.source_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.runtime_input_values > 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

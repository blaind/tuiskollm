//! Source-backed qualification for one Qwen3.5 GDN layer.

use crate::fp8_projection_oracle::{
    BF16_SENTINEL, BYTE_SENTINEL, F32_SENTINEL_BITS, bf16_to_f32, f32_to_bf16,
};
use crate::oracles::norm::residual_oracle;
use crate::qwen35_full_attention_layer::{
    QuantizedActivation, nvfp4_dot_w4a4, quantize_oracle, verify_a16_projection,
};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Qwen35GdnLayerImmutable, Qwen35GdnLayerObservables,
    Qwen35GdnLayerProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MaterializedModelOptNvfp4Gdn,
    MaterializedModelOptNvfp4Mlp, ModelOptNvfp4GdnBindings, ModelOptNvfp4MlpBindings, Qwen35_9B,
};

const SOURCE_LAYER: usize = 0;
const GROUP: usize = 16;
const CONTROL_STRIDE: usize = 128;
const HEAD_DIM: usize = Qwen35_9B::LINEAR_HEAD_DIM;
const KEY_HEADS: usize = Qwen35_9B::LINEAR_KEY_HEADS;
const VALUE_HEADS: usize = Qwen35_9B::LINEAR_VALUE_HEADS;
const QK_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const STATE_PER_ROW: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const RMS_EPSILON: f64 = 1.0e-6;
const QUERY_SCALE: f64 = 0.088_388_35;
const W4A4_BATCHES: [bool; MAX_BATCH] = [true, false, true, true, true, true, true, true];
const EXACT_ROUTES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128];

/// Failure of the complete source-backed Qwen3.5 GDN-layer gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35GdnLayerQualificationError {
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
    #[error("Qwen3.5 GDN-layer qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst error from one source-backed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35GdnLayerQualification {
    /// Residual and normalization values checked at every exact route.
    pub boundary_values: usize,
    /// Exact activation codes and scales checked on every W4A4 stage.
    pub activation_values: usize,
    /// Complete real-source mixer and MLP values checked through B=1.
    pub source_values: usize,
    /// Mutable owner values reproduced by CUDA Graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and state values verified unchanged.
    pub inactive_values: usize,
    /// Immutable source/materialized device values proved unchanged.
    pub immutable_values: usize,
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
    gdn: ModelOptNvfp4GdnBindings<'a>,
    mlp: ModelOptNvfp4MlpBindings<'a>,
}

struct SourceMaterialized<'a> {
    gdn: MaterializedModelOptNvfp4Gdn<'a>,
    mlp: MaterializedModelOptNvfp4Mlp<'a>,
}

/// Qualifies source-backed Qwen3.5 layer 0 at every exact row route.
pub fn qualify_qwen35_gdn_layer(
    root: &Path,
) -> Result<Qwen35GdnLayerQualification, Qwen35GdnLayerQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let bindings = SourceBindings {
        gdn: ModelOptNvfp4GdnBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?,
        mlp: ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?,
    };
    let materialized = SourceMaterialized {
        gdn: bindings.gdn.materialize()?,
        mlp: bindings.mlp.materialize()?,
    };
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = Qwen35GdnLayerProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    if stable_addresses.len() != 44 {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "Qwen3.5 GDN owner exposes {} addresses, expected 44",
            stable_addresses.len()
        )));
    }
    let mut report = Qwen35GdnLayerQualification {
        boundary_values: 0,
        activation_values: 0,
        source_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        immutable_values: 0,
        arena_bytes: program.arena_bytes(),
        weight_bytes: program.resident_weight_bytes(),
        workspace_bytes: program.workspace_bytes(),
        padding_bytes: program.arena_bytes()
            - program.resident_weight_bytes()
            - program.workspace_bytes(),
        maximum_absolute_error: 0.0,
    };

    verify_scales(&program, &materialized)?;
    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &materialized,
        &mut report,
    )?;

    for rows in EXACT_ROUTES {
        let first_input = make_input(rows, 0);
        prepare_run(&program, &stream, rows, &first_input)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(rows, 1);
        prepare_run(&program, &stream, rows, &input)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;

        prepare_run(&program, &stream, rows, &input)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;

        verify_boundaries(rows, &input, bindings, &replay, &mut report)?;
        verify_activation_quantization(rows, &materialized, &replay, &mut report)?;
        if rows == 1 {
            verify_source_formula(bindings, &materialized, &replay, &mut report)?;
        }
        verify_replay(rows, &eager, &replay, &mut report)?;
        verify_replacement_input(rows, &first, &replay)?;
        verify_inactive(rows, &replay, &mut report)?;
        verify_inactive(rows, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "Qwen3.5 GDN owner addresses changed while qualifying {}",
                route_label(rows)
            )));
        }
    }

    verify_slot_routing(&program, &stream, &mut report)?;

    verify_immutable(
        &program.qualification_immutable(&stream)?,
        &materialized,
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
    (0..rows * Qwen35_9B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / Qwen35_9B::HIDDEN) & 15]))
        .collect()
}

fn prepare_run(
    program: &Qwen35GdnLayerProgram,
    stream: &CudaStream,
    rows: usize,
    input: &[u16],
) -> Result<(), Qwen35GdnLayerQualificationError> {
    program.reset_state(stream)?;
    program.load_residual(stream, rows, input)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;

    Ok(())
}

fn verify_slot_routing(
    program: &Qwen35GdnLayerProgram,
    stream: &CudaStream,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let decode_slots = [7usize, 2];
    let decode_rows = decode_slots.len();
    program.reset_state(stream)?;
    program.load_slot_routes(stream, &decode_slots)?;
    program.load_residual(stream, decode_rows, &make_input(decode_rows, 5))?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    program.replay(stream, decode_rows)?;
    let before_reset = program.qualification_observables(stream)?;
    compare_exact(
        "compact GDN state rows",
        &before_reset.state_rows[..decode_rows],
        &[7, 2],
    )?;
    verify_selected_state_slots("decode", &before_reset, &decode_slots, report)?;

    program.reset_slot(stream, decode_slots[0])?;
    let after_reset = program.qualification_observables(stream)?;
    verify_reset_state_slot(
        "history",
        &before_reset.history,
        &after_reset.history,
        decode_slots[0],
        Qwen35_9B::GDN_QKV_ROWS * 3,
        report,
    )?;
    verify_reset_f32_state_slot(
        &before_reset.state,
        &after_reset.state,
        decode_slots[0],
        report,
    )?;

    const PREFILL_SLOT: usize = 5;
    const PREFILL_TOKENS: usize = 32;
    let input = make_input(PREFILL_TOKENS, 6);
    program.reset_state(stream)?;
    program.load_prefill_slot(stream, PREFILL_SLOT)?;
    program.load_residual(stream, PREFILL_TOKENS, &input)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    program.replay(stream, PREFILL_TOKENS)?;
    let mapped = program.qualification_observables(stream)?;
    compare_exact("prompt GDN state row", &mapped.state_rows[..1], &[5])?;
    verify_selected_state_slots("prefill", &mapped, &[PREFILL_SLOT], report)?;

    program.reset_state(stream)?;
    program.load_prefill_slot(stream, 0)?;
    program.load_residual(stream, PREFILL_TOKENS, &input)?;
    program.qualification_reset_outputs(stream, BYTE_SENTINEL)?;
    program.replay(stream, PREFILL_TOKENS)?;
    let row_zero = program.qualification_observables(stream)?;
    compare_exact(
        "slot-independent prompt residual",
        &mapped.residual_output[..PREFILL_TOKENS * Qwen35_9B::HIDDEN],
        &row_zero.residual_output[..PREFILL_TOKENS * Qwen35_9B::HIDDEN],
    )?;
    let history_width = Qwen35_9B::GDN_QKV_ROWS * 3;
    compare_exact(
        "slot-independent prompt history",
        &mapped.history[PREFILL_SLOT * history_width..(PREFILL_SLOT + 1) * history_width],
        &row_zero.history[..history_width],
    )?;
    let state_begin = PREFILL_SLOT * STATE_PER_ROW;
    compare_f32_bits(
        "slot-independent prompt state",
        &mapped.state[state_begin..state_begin + STATE_PER_ROW],
        &row_zero.state[..STATE_PER_ROW],
    )?;
    report.graph_replay_values +=
        PREFILL_TOKENS * Qwen35_9B::HIDDEN + history_width + STATE_PER_ROW;

    Ok(())
}

fn verify_selected_state_slots(
    route: &str,
    observed: &Qwen35GdnLayerObservables,
    selected: &[usize],
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let history_width = Qwen35_9B::GDN_QKV_ROWS * 3;
    for slot in 0..MAX_BATCH {
        let history = &observed.history[slot * history_width..(slot + 1) * history_width];
        let state = &observed.state[slot * STATE_PER_ROW..(slot + 1) * STATE_PER_ROW];
        let is_selected = selected.contains(&slot);
        if is_selected
            && (history.iter().all(|&value| value == 0)
                || state.iter().all(|&value| value.to_bits() == 0))
        {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "{route} did not advance selected physical slot {slot}"
            )));
        }
        if !is_selected
            && (history.iter().any(|&value| value != 0)
                || state.iter().any(|&value| value.to_bits() != 0))
        {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "{route} modified inactive physical slot {slot}"
            )));
        }
        report.inactive_values += history.len() + state.len();
    }
    Ok(())
}

fn verify_reset_state_slot(
    role: &str,
    before: &[u16],
    after: &[u16],
    reset_slot: usize,
    width: usize,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    for index in 0..after.len() {
        let slot = index / width;
        let expected = if slot == reset_slot { 0 } else { before[index] };
        if after[index] != expected {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "reset {role} differs at value {index} for slot {reset_slot}"
            )));
        }
    }
    report.inactive_values += after.len();
    Ok(())
}

fn verify_reset_f32_state_slot(
    before: &[f32],
    after: &[f32],
    reset_slot: usize,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    for index in 0..after.len() {
        let slot = index / STATE_PER_ROW;
        let expected = if slot == reset_slot {
            0
        } else {
            before[index].to_bits()
        };
        if after[index].to_bits() != expected {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "reset recurrent state differs at value {index} for slot {reset_slot}"
            )));
        }
    }
    report.inactive_values += after.len();
    Ok(())
}

fn compare_f32_bits(
    role: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), Qwen35GdnLayerQualificationError> {
    if actual.len() != expected.len() {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{role} has {} values, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{role} differs at value {index}"
        )));
    }
    Ok(())
}

fn verify_scales(
    program: &Qwen35GdnLayerProgram,
    source: &SourceMaterialized<'_>,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let expected_source = [
        source.gdn.input_scale,
        source.gdn.input_weight_scale_2,
        source.gdn.control_input_scale,
        source.gdn.control_weight_scale_2,
        source.gdn.output.input_scale,
        source.gdn.output.weight_scale_2,
        source.mlp.gate_up_input_scale,
        source.mlp.gate_up_weight_scale_2,
        source.mlp.down_input_scale,
        source.mlp.down_weight_scale_2,
    ]
    .map(f32::to_bits);
    let expected_divisors = [
        source.gdn.input_scale_divisor,
        source.gdn.input_weight_scale_divisor,
        source.gdn.control_input_scale_divisor,
        source.gdn.control_weight_scale_divisor,
        source.gdn.output.input_scale_divisor,
        source.gdn.output.weight_scale_divisor,
        source.mlp.gate_up.input_scale_divisor,
        source.mlp.gate_up.weight_scale_divisor,
        source.mlp.down.input_scale_divisor,
        source.mlp.down.weight_scale_divisor,
    ]
    .map(f32::to_bits);

    if program.qualification_source_scales().map(f32::to_bits) != expected_source
        || program.qualification_divisors().map(f32::to_bits) != expected_divisors
    {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(
            "resident ModelOpt scales differ from the materialized source contract".to_string(),
        ));
    }

    Ok(())
}

fn verify_immutable(
    actual: &Qwen35GdnLayerImmutable,
    source: &SourceMaterialized<'_>,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let input_norm = source.gdn.input_norm.words().collect::<Vec<_>>();
    let a_log = source.gdn.a_log.words().collect::<Vec<_>>();
    let dt_bias = source.gdn.dt_bias.words().collect::<Vec<_>>();
    let convolution_weights = source.gdn.convolution_weight.words().collect::<Vec<_>>();
    let recurrent_norm = source.gdn.norm.words().collect::<Vec<_>>();
    let post_attention_norm = source.gdn.post_attention_norm.words().collect::<Vec<_>>();
    let next_norm = source.mlp.next_norm.words().collect::<Vec<_>>();

    macro_rules! same {
        ($field:ident, $expected:expr) => {{
            let expected = $expected;
            if let Some(index) = actual
                .$field
                .iter()
                .zip(expected)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                    "immutable plane `{}` differs at value {index}",
                    stringify!($field)
                )));
            }
            report.immutable_values += actual.$field.len();
        }};
    }

    same!(input_norm, &input_norm);
    same!(input_weight_codes, &source.gdn.input_weight_e2m1);
    same!(input_weight_scales, &source.gdn.input_scale_e4m3_swizzled);
    same!(control_weight_codes, &source.gdn.control_weight_e2m1_padded);
    same!(
        control_weight_scales,
        &source.gdn.control_scale_e4m3_swizzled
    );
    same!(a_log, &a_log);
    same!(dt_bias, &dt_bias);
    same!(convolution_weights, &convolution_weights);
    same!(recurrent_norm, &recurrent_norm);
    same!(output_weight_codes, source.gdn.output.weight_e2m1);
    same!(output_weight_scales, &source.gdn.output.scale_e4m3_swizzled);
    same!(post_attention_norm, &post_attention_norm);
    same!(gate_weight_codes, source.mlp.gate_up.gate_weight_e2m1);
    same!(up_weight_codes, source.mlp.gate_up.up_weight_e2m1);
    same!(
        gate_up_weight_scales,
        &source.mlp.gate_up.scale_e4m3_swizzled
    );
    same!(down_weight_codes, source.mlp.down.weight_e2m1);
    same!(down_weight_scales, &source.mlp.down.scale_e4m3_swizzled);
    same!(next_norm, &next_norm);

    Ok(())
}

fn verify_boundaries(
    rows: usize,
    input: &[u16],
    sources: SourceBindings<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let input_norm = sources.gdn.input_norm.words().collect::<Vec<_>>();
    let post_norm = sources.gdn.post_attention_norm.words().collect::<Vec<_>>();
    let next_norm = sources.mlp.next_norm.words().collect::<Vec<_>>();

    for token in 0..rows {
        let begin = token * Qwen35_9B::HIDDEN;
        let end = begin + Qwen35_9B::HIDDEN;
        let mixer_normalized = rms_norm_oracle::<Qwen35_9B>(&input[begin..end], &input_norm);
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
        let mlp_normalized = rms_norm_oracle::<Qwen35_9B>(&mixer_residual, &post_norm);
        compare_bf16(
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
        let next = rms_norm_oracle::<Qwen35_9B>(&residual, &next_norm);
        compare_bf16(
            "next RMSNorm",
            &observed.next_normalized[begin..end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.boundary_values += rows * Qwen35_9B::HIDDEN * 5;

    Ok(())
}

fn verify_activation_quantization(
    rows: usize,
    source: &SourceMaterialized<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    if rows > MAX_BATCH {
        verify_quantized_rows(
            "GDN input",
            rows,
            Qwen35_9B::HIDDEN,
            &observed.mixer_normalized,
            &observed.input_activation_codes,
            &observed.input_activation_scales,
            source.gdn.input_scale_divisor,
            report,
        )?;
        verify_quantized_rows(
            "GDN output",
            rows,
            Qwen35_9B::GDN_VALUE_ROWS,
            &observed.recurrent_output,
            &observed.output_activation_codes,
            &observed.output_activation_scales,
            source.gdn.output.input_scale_divisor,
            report,
        )?;
        verify_quantized_rows(
            "gate/up",
            rows,
            Qwen35_9B::HIDDEN,
            &observed.mlp_normalized,
            &observed.gate_up_activation_codes,
            &observed.gate_up_activation_scales,
            source.mlp.gate_up.input_scale_divisor,
            report,
        )?;
        verify_quantized_rows(
            "down",
            rows,
            Qwen35_9B::INTERMEDIATE,
            &observed.swiglu,
            &observed.down_activation_codes,
            &observed.down_activation_scales,
            source.mlp.down.input_scale_divisor,
            report,
        )?;
    } else if W4A4_BATCHES[rows - 1] {
        verify_quantized_rows(
            "gate/up",
            rows,
            Qwen35_9B::HIDDEN,
            &observed.mlp_normalized,
            &observed.gate_up_activation_codes,
            &observed.gate_up_activation_scales,
            source.mlp.gate_up.input_scale_divisor,
            report,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_quantized_rows(
    role: &str,
    rows: usize,
    columns: usize,
    input: &[u16],
    observed_codes: &[u8],
    observed_scales: &[u8],
    divisor: f32,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let code_width = columns / 2;
    let scale_width = columns / GROUP;
    for token in 0..rows {
        let input_begin = token * columns;
        let (codes, scales) = quantize_oracle(&input[input_begin..input_begin + columns], divisor)
            .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
        compare_exact(
            &format!("{role} activation codes"),
            &observed_codes[token * code_width..(token + 1) * code_width],
            &codes,
        )?;
        compare_exact(
            &format!("{role} activation scales"),
            &observed_scales[token * scale_width..(token + 1) * scale_width],
            &scales,
        )?;
    }
    report.activation_values += rows * (code_width + scale_width);

    Ok(())
}

fn verify_source_formula(
    bindings: SourceBindings<'_>,
    materialized: &SourceMaterialized<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let input = &observed.mixer_normalized[..Qwen35_9B::HIDDEN];
    let mut offset = 0;
    for (role, binding) in [
        ("GDN Q/K/V projection", bindings.gdn.qkv),
        ("GDN Z projection", bindings.gdn.z),
    ] {
        let rows = binding.weight.shape()[0] as usize;
        verify_a16_projection(
            role,
            input,
            binding,
            materialized.gdn.input_weight_scale_divisor,
            &observed.projected[offset..offset + rows],
            &mut report.maximum_absolute_error,
        )
        .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
        offset += rows;
    }
    if offset != Qwen35_9B::GDN_INPUT_ROWS {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "source GDN input rows total {offset}, expected {}",
            Qwen35_9B::GDN_INPUT_ROWS
        )));
    }
    for (role, binding, begin) in [
        ("GDN A-control projection", bindings.gdn.a_control, 0),
        (
            "GDN B-control projection",
            bindings.gdn.b_control,
            Qwen35_9B::GDN_CONTROL_ROWS,
        ),
    ] {
        verify_a16_projection(
            role,
            input,
            binding,
            materialized.gdn.control_weight_scale_divisor,
            &observed.projected_controls[begin..begin + Qwen35_9B::GDN_CONTROL_ROWS],
            &mut report.maximum_absolute_error,
        )
        .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
    }
    verify_controls(bindings.gdn, observed, report)?;
    verify_convolution(bindings.gdn, observed, report)?;
    verify_recurrence(bindings.gdn, observed, report)?;
    verify_a16_projection(
        "GDN output projection",
        &observed.recurrent_output[..Qwen35_9B::GDN_VALUE_ROWS],
        bindings.gdn.output,
        materialized.gdn.output.weight_scale_divisor,
        &observed.mixer_branch[..Qwen35_9B::HIDDEN],
        &mut report.maximum_absolute_error,
    )
    .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;

    let (activation_codes, activation_scales) = quantize_oracle(
        &observed.mlp_normalized[..Qwen35_9B::HIDDEN],
        materialized.mlp.gate_up.input_scale_divisor,
    )
    .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
    let activation = QuantizedActivation {
        codes: &activation_codes,
        scales: &activation_scales,
        scale_divisor: materialized.mlp.gate_up.input_scale_divisor,
    };
    for row in 0..Qwen35_9B::INTERMEDIATE {
        let gate = nvfp4_dot_w4a4(
            activation,
            bindings.mlp.gate.weight.bytes(),
            bindings.mlp.gate.block_scale.codes(),
            row,
            Qwen35_9B::HIDDEN,
            materialized.mlp.gate_up.weight_scale_divisor,
        )
        .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
        let up = nvfp4_dot_w4a4(
            activation,
            bindings.mlp.up.weight.bytes(),
            bindings.mlp.up.block_scale.codes(),
            row,
            Qwen35_9B::HIDDEN,
            materialized.mlp.gate_up.weight_scale_divisor,
        )
        .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;
        let gate = f64::from(bf16_to_f32(f32_to_bf16(gate as f32)));
        let up = f64::from(bf16_to_f32(f32_to_bf16(up as f32)));
        let expected = gate / (1.0 + (-gate).exp()) * up;
        require_close(
            "source SwiGLU",
            row,
            bf16_to_f32(observed.swiglu[row]),
            expected,
            0.25,
            0.025,
            &mut report.maximum_absolute_error,
        )?;
    }
    verify_a16_projection(
        "source down projection",
        &observed.swiglu[..Qwen35_9B::INTERMEDIATE],
        bindings.mlp.down,
        materialized.mlp.down.weight_scale_divisor,
        &observed.mlp_branch[..Qwen35_9B::HIDDEN],
        &mut report.maximum_absolute_error,
    )
    .map_err(|error| Qwen35GdnLayerQualificationError::Mismatch(error.to_string()))?;

    report.source_values += Qwen35_9B::GDN_INPUT_ROWS
        + 2 * Qwen35_9B::GDN_CONTROL_ROWS
        + 4 * Qwen35_9B::GDN_QKV_ROWS
        + STATE_PER_ROW
        + Qwen35_9B::GDN_VALUE_ROWS
        + Qwen35_9B::HIDDEN
        + Qwen35_9B::INTERMEDIATE
        + Qwen35_9B::HIDDEN;

    Ok(())
}

fn verify_controls(
    source: ModelOptNvfp4GdnBindings<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
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
        require_f32_close("GDN control", row, actual, expected, 0.002, report)?;
    }

    Ok(())
}

fn verify_convolution(
    source: ModelOptNvfp4GdnBindings<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let weights = source.convolution_weight.words().collect::<Vec<_>>();
    for channel in 0..Qwen35_9B::GDN_QKV_ROWS {
        let current = bf16_to_f32(observed.projected[channel]);
        let weight = bf16_to_f32(weights[channel * 4 + 3]);
        let sum = f64::from(current) * f64::from(weight);
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
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "causal history differs at channel {channel}"
            )));
        }
    }

    Ok(())
}

fn verify_recurrence(
    source: ModelOptNvfp4GdnBindings<'_>,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let norm = source.norm.words().collect::<Vec<_>>();
    let mut state = vec![0.0f64; STATE_PER_ROW];
    let mut output = vec![0.0f64; Qwen35_9B::GDN_VALUE_ROWS];
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
        let gate_begin = Qwen35_9B::GDN_QKV_ROWS + value_head * HEAD_DIM;
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
        require_f32_close("GDN state", index, actual, expected, 2.0e-4, report)?;
    }
    for (index, (&actual, &expected)) in observed.recurrent_output[..Qwen35_9B::GDN_VALUE_ROWS]
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

fn verify_replay(
    rows: usize,
    eager: &Qwen35GdnLayerObservables,
    replay: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                    "{} graph plane `{}` differs at value {index}",
                    route_label(rows),
                    stringify!($field)
                )));
            }
        };
    }
    macro_rules! same_f32 {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .map(|value| value.to_bits())
                .zip(eager.$field.iter().map(|value| value.to_bits()))
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                    "{} graph plane `{}` differs at value {index}",
                    route_label(rows),
                    stringify!($field)
                )));
            }
        };
    }

    same!(mixer_normalized);
    same!(input_activation_codes);
    same!(input_activation_scales);
    same!(projected);
    same!(projected_controls);
    same_f32!(log_decay);
    same_f32!(beta);
    same!(convolved);
    same!(history);
    same_f32!(state);
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
    report.graph_replay_values += observable_values(replay);

    Ok(())
}

fn observable_values(values: &Qwen35GdnLayerObservables) -> usize {
    values.mixer_normalized.len()
        + values.input_activation_codes.len()
        + values.input_activation_scales.len()
        + values.projected.len()
        + values.projected_controls.len()
        + values.log_decay.len()
        + values.beta.len()
        + values.convolved.len()
        + values.history.len()
        + values.state.len()
        + values.recurrent_output.len()
        + values.output_activation_codes.len()
        + values.output_activation_scales.len()
        + values.mixer_branch.len()
        + values.mixer_residual.len()
        + values.mlp_normalized.len()
        + values.gate_up_activation_codes.len()
        + values.gate_up_activation_scales.len()
        + values.swiglu.len()
        + values.down_activation_codes.len()
        + values.down_activation_scales.len()
        + values.mlp_branch.len()
        + values.residual_output.len()
        + values.next_normalized.len()
}

fn verify_replacement_input(
    rows: usize,
    first: &Qwen35GdnLayerObservables,
    replay: &Qwen35GdnLayerObservables,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let active = rows * Qwen35_9B::HIDDEN;
    if first.residual_output[..active] == replay.residual_output[..active] {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{} graph ignored replacement residual rows",
            route_label(rows)
        )));
    }

    Ok(())
}

fn verify_inactive(
    rows: usize,
    observed: &Qwen35GdnLayerObservables,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = rows * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
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
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                    "{} modified inactive `{}` value",
                    route_label(rows),
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_u8 {
        ($field:ident, $begin:expr) => {{
            let begin = $begin;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BYTE_SENTINEL)
            {
                return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                    "{} modified inactive `{}` value",
                    route_label(rows),
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }

    let mut inactive = 0;
    inactive += sentinel_u16!(mixer_normalized, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(projected, Qwen35_9B::GDN_INPUT_ROWS);
    inactive += sentinel_u16!(projected_controls, CONTROL_STRIDE);
    inactive += sentinel_f32!(log_decay, VALUE_HEADS);
    inactive += sentinel_f32!(beta, VALUE_HEADS);
    inactive += sentinel_u16!(convolved, Qwen35_9B::GDN_QKV_ROWS);
    inactive += sentinel_u16!(recurrent_output, Qwen35_9B::GDN_VALUE_ROWS);
    inactive += sentinel_u16!(mixer_branch, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(mixer_residual, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(mlp_normalized, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(swiglu, Qwen35_9B::INTERMEDIATE);
    inactive += sentinel_u16!(mlp_branch, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(residual_output, Qwen35_9B::HIDDEN);
    inactive += sentinel_u16!(next_normalized, Qwen35_9B::HIDDEN);

    for token in 0..rows {
        let padding = &observed.projected_controls
            [token * CONTROL_STRIDE + 2 * VALUE_HEADS..(token + 1) * CONTROL_STRIDE];
        if padding.iter().any(|&value| value != 0) {
            return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
                "{} active control padding is not zero for token {token}",
                route_label(rows)
            )));
        }
    }

    let active_state_rows = if rows <= MAX_BATCH { rows } else { 1 };
    let history_begin = active_state_rows * Qwen35_9B::GDN_QKV_ROWS * 3;
    if observed.history[history_begin..]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{} modified inactive history",
            route_label(rows)
        )));
    }
    inactive += observed.history.len() - history_begin;
    let state_begin = active_state_rows * STATE_PER_ROW;
    if observed.state[state_begin..]
        .iter()
        .any(|&value| value.to_bits() != 0)
    {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{} modified inactive recurrent state",
            route_label(rows)
        )));
    }
    inactive += observed.state.len() - state_begin;

    let hidden_codes = Qwen35_9B::HIDDEN / 2;
    let hidden_scales = Qwen35_9B::HIDDEN / GROUP;
    let intermediate_codes = Qwen35_9B::INTERMEDIATE / 2;
    let intermediate_scales = Qwen35_9B::INTERMEDIATE / GROUP;
    let prompt = rows > MAX_BATCH;
    let prompt_hidden_code_begin = if prompt { rows * hidden_codes } else { 0 };
    let prompt_hidden_scale_begin = if prompt { rows * hidden_scales } else { 0 };
    let gate_code_begin = if uses_w4a4(rows) {
        rows * hidden_codes
    } else {
        0
    };
    let gate_scale_begin = if uses_w4a4(rows) {
        rows * hidden_scales
    } else {
        0
    };
    let down_code_begin = if prompt { rows * intermediate_codes } else { 0 };
    let down_scale_begin = if prompt {
        rows * intermediate_scales
    } else {
        0
    };
    inactive += sentinel_u8!(input_activation_codes, prompt_hidden_code_begin);
    inactive += sentinel_u8!(input_activation_scales, prompt_hidden_scale_begin);
    inactive += sentinel_u8!(output_activation_codes, prompt_hidden_code_begin);
    inactive += sentinel_u8!(output_activation_scales, prompt_hidden_scale_begin);
    inactive += sentinel_u8!(gate_up_activation_codes, gate_code_begin);
    inactive += sentinel_u8!(gate_up_activation_scales, gate_scale_begin);
    inactive += sentinel_u8!(down_activation_codes, down_code_begin);
    inactive += sentinel_u8!(down_activation_scales, down_scale_begin);

    report.inactive_values += inactive;

    Ok(())
}

fn verify_no_device_allocation(
    program: &Qwen35GdnLayerProgram,
    stream: &CudaStream,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    program.replay(stream, 128)?;
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
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn uses_w4a4(rows: usize) -> bool {
    rows > MAX_BATCH || W4A4_BATCHES[rows - 1]
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
) -> Result<(), Qwen35GdnLayerQualificationError> {
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
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
) -> Result<(), Qwen35GdnLayerQualificationError> {
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

fn require_f32_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    tolerance: f32,
    report: &mut Qwen35GdnLayerQualification,
) -> Result<(), Qwen35GdnLayerQualificationError> {
    require_close(
        role,
        index,
        actual,
        expected,
        tolerance,
        0.002,
        &mut report.maximum_absolute_error,
    )
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
) -> Result<(), Qwen35GdnLayerQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = absolute.max(expected.abs() as f32 * relative);
    if !actual.is_finite() || error > tolerance {
        return Err(Qwen35GdnLayerQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_ROUTES, Qwen35GdnLayerQualificationError, SOURCE_LAYER, W4A4_BATCHES,
        qualify_qwen35_gdn_layer, uses_w4a4,
    };

    #[test]
    fn source_layer_and_swiglu_routes_are_exact() {
        assert_eq!(SOURCE_LAYER, 0);
        assert_eq!(
            W4A4_BATCHES,
            [true, false, true, true, true, true, true, true]
        );
        assert_eq!(EXACT_ROUTES, [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128]);
        assert!(
            EXACT_ROUTES
                .into_iter()
                .all(|rows| rows == 2 || uses_w4a4(rows))
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer0_matches_complete_oracles_and_graph_replay()
    -> Result<(), Qwen35GdnLayerQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            Qwen35GdnLayerQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_qwen35_gdn_layer(std::path::Path::new(&root))?;

        assert_eq!(report.boundary_values, 5_324_800);
        assert_eq!(report.activation_values, 3_174_912);
        assert_eq!(report.weight_bytes, 123_068_800);
        assert_eq!(report.workspace_bytes, 41_074_720);
        assert_eq!(report.arena_bytes, 164_144_128);
        assert_eq!(report.padding_bytes, 608);
        assert!(report.source_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.maximum_absolute_error <= 0.5);

        Ok(())
    }
}

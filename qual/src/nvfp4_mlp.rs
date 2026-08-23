//! Source-backed qualification for one resident NVFP4 MLP owner.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Nvfp4MlpImmutable, Nvfp4MlpObservables, Nvfp4MlpProgram,
};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Nvfp4MlpBindings, Qwen38_27B};

const SOURCE_LAYER: usize = 55;
const GROUP: usize = 16;
const HIDDEN: usize = Qwen38_27B::HIDDEN;
const INTERMEDIATE: usize = Qwen38_27B::INTERMEDIATE;
const HIDDEN_GROUPS: usize = HIDDEN / GROUP;
const HIDDEN_CODE_BYTES: usize = HIDDEN / 2;

/// Failure of the complete NVFP4 MLP qualification gate.
#[derive(Debug, thiserror::Error)]
pub enum Nvfp4MlpQualificationError {
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
    #[error("NVFP4 MLP qualification failed: {0}")]
    Mismatch(String),
}

/// Observable counts, ownership, and worst error from the complete MLP boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nvfp4MlpQualification {
    /// Pre-MLP normalized BF16 values checked at every exact batch.
    pub normalized_values: usize,
    /// Dynamic E2M1 codes checked bit-exactly on every W4A4 route.
    pub activation_codes: usize,
    /// Dynamic E4M3 scales checked bit-exactly on every W4A4 route.
    pub activation_scales: usize,
    /// B=1 SwiGLU values checked against the complete source-weight formula.
    pub source_swiglu_values: usize,
    /// B=1 down values checked against the complete source-weight formula.
    pub source_branch_values: usize,
    /// Published residual and next-normalized values checked at every batch.
    pub boundary_values: usize,
    /// Active values reproduced exactly by eager and graph execution.
    pub graph_replay_values: usize,
    /// Sentinel values preserved outside each exact route extent.
    pub inactive_values: usize,
    /// Immutable source/materialized device values proved unchanged.
    pub immutable_values: usize,
    /// Complete one-allocation owner bytes.
    pub arena_bytes: usize,
    /// Exact source-backed norm and projection bytes.
    pub weight_bytes: usize,
    /// Exact address-stable working-plane bytes.
    pub workspace_bytes: usize,
    /// Alignment padding bytes in the owner arena.
    pub padding_bytes: usize,
    /// Largest absolute difference at an accepted BF16 seam.
    pub maximum_absolute_error: f32,
}

/// Qualifies source-backed layer 55 at every exact decode batch.
pub fn qualify_nvfp4_mlp(root: &Path) -> Result<Nvfp4MlpQualification, Nvfp4MlpQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let bindings = Nvfp4MlpBindings::bind(snapshot.as_ref(), SOURCE_LAYER)?;
    let gate_up_materialized = bindings.gate_up.materialize()?;
    let down_materialized = bindings.down.materialize()?;
    let input_norm = bindings.input_norm.words().collect::<Vec<_>>();
    let next_norm = bindings.next_norm.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program = Nvfp4MlpProgram::from_snapshot(&context, snapshot.clone(), SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let expected_immutable = ExpectedImmutable {
        input_norm: &input_norm,
        gate_weight_codes: gate_up_materialized.gate_weight_e2m1,
        up_weight_codes: gate_up_materialized.up_weight_e2m1,
        gate_up_weight_scales: &gate_up_materialized.scale_e4m3_swizzled,
        down_weight_codes: down_materialized.weight_e2m1,
        down_weight_scales: &down_materialized.scale_e4m3_swizzled,
        next_norm: &next_norm,
    };
    let mut report = Nvfp4MlpQualification {
        normalized_values: 0,
        activation_codes: 0,
        activation_scales: 0,
        source_swiglu_values: 0,
        source_branch_values: 0,
        boundary_values: 0,
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

    require_divisors(&program, bindings)?;
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
        verify_immutable(
            batch,
            &program.qualification_immutable(&stream)?,
            expected_immutable,
            &mut report,
        )?;

        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;
        verify_immutable(
            batch,
            &program.qualification_immutable(&stream)?,
            expected_immutable,
            &mut report,
        )?;

        verify_seams(
            batch,
            &input,
            &input_norm,
            &next_norm,
            bindings.gate_up.input_scale_divisor,
            &replay,
            &mut report,
        )?;
        if batch == 1 {
            verify_source_formula(bindings, &replay, &mut report)?;
        }
        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_replacement_input(batch, &first, &replay)?;
        verify_inactive(batch, &replay, &mut report)?;
        verify_inactive(batch, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

#[derive(Clone, Copy)]
struct ExpectedImmutable<'a> {
    input_norm: &'a [u16],
    gate_weight_codes: &'a [u8],
    up_weight_codes: &'a [u8],
    gate_up_weight_scales: &'a [u8],
    down_weight_codes: &'a [u8],
    down_weight_scales: &'a [u8],
    next_norm: &'a [u16],
}

fn require_divisors(
    program: &Nvfp4MlpProgram,
    bindings: Nvfp4MlpBindings<'_>,
) -> Result<(), Nvfp4MlpQualificationError> {
    let actual = program.qualification_divisors().map(f32::to_bits);
    let expected = [
        bindings.gate_up.input_scale_divisor,
        bindings.gate_up.weight_scale_divisor,
        bindings.down.input_scale_divisor,
        bindings.down.weight_scale_divisor,
    ]
    .map(f32::to_bits);
    if actual != expected {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "resident divisors differ: device-owner={actual:?}, source={expected:?}"
        )));
    }

    Ok(())
}

fn make_input(batch: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..batch * HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / HIDDEN) & 15]))
        .collect()
}

fn verify_seams(
    batch: usize,
    input: &[u16],
    input_norm: &[u16],
    next_norm: &[u16],
    input_scale_divisor: f32,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    for token in 0..batch {
        let hidden_begin = token * HIDDEN;
        let hidden_end = hidden_begin + HIDDEN;
        let intermediate_begin = token * INTERMEDIATE;
        let intermediate_end = intermediate_begin + INTERMEDIATE;
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

        if uses_w4a4(batch) {
            let (codes, scales) = quantize_oracle(
                &observed.normalized[hidden_begin..hidden_end],
                input_scale_divisor,
            )?;
            require_equal_u8(
                "gate/up activation code",
                batch,
                token,
                &observed.gate_up_activation_codes
                    [token * HIDDEN_CODE_BYTES..(token + 1) * HIDDEN_CODE_BYTES],
                &codes,
            )?;
            require_equal_u8(
                "gate/up activation scale",
                batch,
                token,
                &observed.gate_up_activation_scales
                    [token * HIDDEN_GROUPS..(token + 1) * HIDDEN_GROUPS],
                &scales,
            )?;
            report.activation_codes += HIDDEN_CODE_BYTES;
            report.activation_scales += HIDDEN_GROUPS;
        }

        require_active_written(
            "SwiGLU",
            batch,
            token,
            &observed.swiglu[intermediate_begin..intermediate_end],
        )?;
        require_active_written(
            "down branch",
            batch,
            token,
            &observed.branch[hidden_begin..hidden_end],
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
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
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

    report.normalized_values += batch * HIDDEN;
    report.boundary_values += batch * HIDDEN * 2;

    Ok(())
}

fn verify_source_formula(
    bindings: Nvfp4MlpBindings<'_>,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    let (activation_codes, activation_scales) = quantize_oracle(
        &observed.normalized[..HIDDEN],
        bindings.gate_up.input_scale_divisor,
    )?;
    let activation = QuantizedActivation {
        codes: &activation_codes,
        scales: &activation_scales,
        scale_divisor: bindings.gate_up.input_scale_divisor,
    };
    for row in 0..INTERMEDIATE {
        let gate = nvfp4_dot_w4a4(
            activation,
            bindings.gate_up.gate_weight.bytes(),
            bindings.gate_up.gate_scale.codes(),
            row,
            HIDDEN,
            bindings.gate_up.weight_scale_divisor,
        )?;
        let up = nvfp4_dot_w4a4(
            activation,
            bindings.gate_up.up_weight.bytes(),
            bindings.gate_up.up_scale.codes(),
            row,
            HIDDEN,
            bindings.gate_up.weight_scale_divisor,
        )?;
        let gate = f64::from(bf16_to_f32(f32_to_bf16(gate as f32)));
        let up = f64::from(bf16_to_f32(f32_to_bf16(up as f32)));
        let expected = gate / (1.0 + (-gate).exp()) * up;
        require_close(
            "source SwiGLU",
            row,
            bf16_to_f32(observed.swiglu[row]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }

    for row in 0..HIDDEN {
        let expected = nvfp4_dot_a16(
            &observed.swiglu[..INTERMEDIATE],
            bindings.down.weight.bytes(),
            bindings.down.scale.codes(),
            row,
            INTERMEDIATE,
            bindings.down.weight_scale_divisor,
        )?;
        require_close(
            "source down projection",
            row,
            bf16_to_f32(observed.branch[row]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.source_swiglu_values += INTERMEDIATE;
    report.source_branch_values += HIDDEN;

    Ok(())
}

#[derive(Clone, Copy)]
struct QuantizedActivation<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    scale_divisor: f32,
}

fn nvfp4_dot_w4a4(
    activation: QuantizedActivation<'_>,
    weights: &[u8],
    scales: &[u8],
    row: usize,
    columns: usize,
    weight_scale_divisor: f32,
) -> Result<f64, Nvfp4MlpQualificationError> {
    let groups = columns / GROUP;
    let code_bytes = columns / 2;
    let weight_begin = row * code_bytes;
    let mut sum = 0.0f64;

    for group in 0..groups {
        let activation_scale = decode_e4m3fn(activation.scales[group])?;
        let weight_scale = decode_e4m3fn(scales[row * groups + group])?;
        let mut group_sum = 0.0f64;
        for pair in 0..GROUP / 2 {
            let activation_pair = activation.codes[group * (GROUP / 2) + pair];
            let weight_pair = weights[weight_begin + group * (GROUP / 2) + pair];
            for nibble in 0..2 {
                let shift = nibble * 4;
                group_sum += f64::from(decode_e2m1((activation_pair >> shift) & 15))
                    * f64::from(decode_e2m1((weight_pair >> shift) & 15));
            }
        }
        sum += group_sum
            * f64::from(activation_scale / activation.scale_divisor)
            * f64::from(weight_scale / weight_scale_divisor);
    }

    Ok(sum)
}

fn nvfp4_dot_a16(
    activation: &[u16],
    weights: &[u8],
    scales: &[u8],
    row: usize,
    columns: usize,
    weight_scale_divisor: f32,
) -> Result<f64, Nvfp4MlpQualificationError> {
    let groups = columns / GROUP;
    let code_bytes = columns / 2;
    let weight_begin = row * code_bytes;
    let mut sum = 0.0f64;

    for group in 0..groups {
        let scale = decode_e4m3fn(scales[row * groups + group])?;
        let mut group_sum = 0.0f64;
        for column in 0..GROUP {
            let packed = weights[weight_begin + group * (GROUP / 2) + column / 2];
            let code = if column & 1 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            group_sum += f64::from(bf16_to_f32(activation[group * GROUP + column]))
                * f64::from(decode_e2m1(code));
        }
        sum += group_sum * f64::from(scale / weight_scale_divisor);
    }

    Ok(sum)
}

fn quantize_oracle(
    input: &[u16],
    input_scale_divisor: f32,
) -> Result<(Vec<u8>, Vec<u8>), Nvfp4MlpQualificationError> {
    let mut codes = vec![0u8; HIDDEN_CODE_BYTES];
    let mut scales = vec![0u8; HIDDEN_GROUPS];

    for group in 0..HIDDEN_GROUPS {
        let begin = group * GROUP;
        let maximum = input[begin..begin + GROUP]
            .iter()
            .map(|&value| bf16_to_f32(value).abs())
            .fold(0.0f32, f32::max);
        let scale = encode_e4m3fn(input_scale_divisor * maximum / 6.0)?;
        scales[group] = scale;
        if scale == 0 {
            continue;
        }

        let decoded_scale = decode_e4m3fn(scale)?;
        for pair in 0..GROUP / 2 {
            let low = encode_e2m1(
                bf16_to_f32(input[begin + 2 * pair]) * input_scale_divisor / decoded_scale,
            );
            let high = encode_e2m1(
                bf16_to_f32(input[begin + 2 * pair + 1]) * input_scale_divisor / decoded_scale,
            );
            codes[group * (GROUP / 2) + pair] = low | (high << 4);
        }
    }

    Ok((codes, scales))
}

fn verify_immutable(
    batch: usize,
    actual: &Nvfp4MlpImmutable,
    expected: ExpectedImmutable<'_>,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = actual
                .$field
                .iter()
                .zip(expected.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                    "B={batch} immutable plane `{}` differs at value {index}",
                    stringify!($field),
                )));
            }
            report.immutable_values += actual.$field.len();
        };
    }
    same!(input_norm);
    same!(gate_weight_codes);
    same!(up_weight_codes);
    same!(gate_up_weight_scales);
    same!(down_weight_codes);
    same!(down_weight_scales);
    same!(next_norm);

    Ok(())
}

fn verify_replay(
    batch: usize,
    eager: &Nvfp4MlpObservables,
    replay: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                    "B={batch} graph plane `{}` differs at value {index}",
                    stringify!($field),
                )));
            }
        };
    }
    same!(residual_input);
    same!(normalized);
    same!(gate_up_activation_codes);
    same!(gate_up_activation_scales);
    same!(swiglu);
    same!(branch);
    same!(residual_output);
    same!(next_normalized);

    report.graph_replay_values += batch * (5 * HIDDEN + INTERMEDIATE);
    if uses_w4a4(batch) {
        report.graph_replay_values += batch * (HIDDEN_CODE_BYTES + HIDDEN_GROUPS);
    }

    Ok(())
}

fn verify_replacement_input(
    batch: usize,
    first: &Nvfp4MlpObservables,
    replay: &Nvfp4MlpObservables,
) -> Result<(), Nvfp4MlpQualificationError> {
    let active = batch * HIDDEN;
    if first.residual_input[..active] == replay.residual_input[..active]
        || first.residual_output[..active] == replay.residual_output[..active]
    {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "B={batch} graph replay did not observe replacement residual rows"
        )));
    }

    Ok(())
}

fn verify_inactive(
    batch: usize,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    let hidden_begin = batch * HIDDEN;
    let intermediate_begin = batch * INTERMEDIATE;
    for (role, values) in [
        ("normalized", &observed.normalized[hidden_begin..]),
        ("branch", &observed.branch[hidden_begin..]),
        ("residual output", &observed.residual_output[hidden_begin..]),
        ("next normalized", &observed.next_normalized[hidden_begin..]),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BF16_SENTINEL) {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} value {relative}"
            )));
        }
    }
    if let Some(relative) = observed.swiglu[intermediate_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "B={batch} modified inactive SwiGLU value {relative}"
        )));
    }

    let code_begin = if uses_w4a4(batch) {
        batch * HIDDEN_CODE_BYTES
    } else {
        0
    };
    let scale_begin = if uses_w4a4(batch) {
        batch * HIDDEN_GROUPS
    } else {
        0
    };
    for (role, values) in [
        (
            "gate/up activation code",
            &observed.gate_up_activation_codes[code_begin..],
        ),
        (
            "gate/up activation scale",
            &observed.gate_up_activation_scales[scale_begin..],
        ),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BYTE_SENTINEL) {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "B={batch} modified inactive {role} value {relative}"
            )));
        }
    }

    let inactive_scratch =
        (MAX_BATCH * HIDDEN_CODE_BYTES - code_begin) + (MAX_BATCH * HIDDEN_GROUPS - scale_begin);
    report.inactive_values += (MAX_BATCH - batch) * (4 * HIDDEN + INTERMEDIATE) + inactive_scratch;

    Ok(())
}

fn verify_no_device_allocation(
    program: &Nvfp4MlpProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), Nvfp4MlpQualificationError> {
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
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn uses_w4a4(batch: usize) -> bool {
    batch == 1 || batch >= 5
}

fn require_active_written(
    role: &str,
    batch: usize,
    token: usize,
    values: &[u16],
) -> Result<(), Nvfp4MlpQualificationError> {
    if let Some(column) = values.iter().position(|&value| value == BF16_SENTINEL) {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "B={batch} {role} row={token}, column={column} retained its sentinel"
        )));
    }

    Ok(())
}

fn require_equal_u8(
    role: &str,
    batch: usize,
    token: usize,
    actual: &[u8],
    expected: &[u8],
) -> Result<(), Nvfp4MlpQualificationError> {
    if let Some(column) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "{role} at B={batch}, row={token}, column={column}: device={:#04x}, oracle={:#04x}",
            actual[column], expected[column],
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
) -> Result<(), Nvfp4MlpQualificationError> {
    for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        require_close(
            &format!("{role} at B={batch}, row={token}"),
            column,
            bf16_to_f32(actual),
            f64::from(bf16_to_f32(expected)),
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
) -> Result<(), Nvfp4MlpQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.25f32.max(expected.abs() as f32 * 0.025);
    if error > tolerance {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "{role}, column={column}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }

    Ok(())
}

fn encode_e2m1(value: f32) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    let candidates = if value.is_sign_negative() {
        8u8..16
    } else {
        0u8..8
    };

    for code in candidates {
        let distance = (value - decode_e2m1(code)).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    best
}

fn decode_e2m1(code: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(code & 7) as usize];

    if code & 8 == 0 { magnitude } else { -magnitude }
}

fn encode_e4m3fn(value: f32) -> Result<u8, Nvfp4MlpQualificationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(Nvfp4MlpQualificationError::Mismatch(
            "oracle E4M3 scale is not finite and non-negative".to_string(),
        ));
    }

    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=0x7e {
        let represented = decode_e4m3fn(code)?;
        let distance = (value - represented).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best = code;
            best_distance = distance;
        }
    }

    Ok(best)
}

fn decode_e4m3fn(word: u8) -> Result<f32, Nvfp4MlpQualificationError> {
    let sign = if word & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (word >> 3) & 15;
    let fraction = word & 7;
    let magnitude = match (exponent, fraction) {
        (0, 0) => 0.0,
        (0, fraction) => f32::from(fraction) * 2.0f32.powi(-9),
        (15, 7) => {
            return Err(Nvfp4MlpQualificationError::Mismatch(
                "oracle encountered an E4M3FN NaN".to_string(),
            ));
        }
        (exponent, fraction) => {
            (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
        }
    };

    Ok(sign * magnitude)
}

#[cfg(test)]
mod tests {
    use super::{
        Nvfp4MlpQualificationError, SOURCE_LAYER, decode_e2m1, decode_e4m3fn, qualify_nvfp4_mlp,
        uses_w4a4,
    };

    #[test]
    fn independent_codecs_and_route_table_are_pinned() {
        assert_eq!(decode_e2m1(0x07), 6.0);
        assert_eq!(decode_e2m1(0x0f), -6.0);
        assert_eq!(decode_e4m3fn(0x01).unwrap(), 2.0f32.powi(-9));
        assert_eq!(decode_e4m3fn(0x38).unwrap(), 1.0);
        assert_eq!(decode_e4m3fn(0x40).unwrap(), 2.0);
        assert_eq!(
            (1..=8).map(uses_w4a4).collect::<Vec<_>>(),
            [true, false, false, false, true, true, true, true],
        );
        assert_eq!(SOURCE_LAYER, 55);
    }

    #[test]
    #[ignore = "requires TUISKO_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn source_layer55_matches_complete_oracles_and_graph_replay()
    -> Result<(), Nvfp4MlpQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            Nvfp4MlpQualificationError::Mismatch(
                "TUISKO_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_nvfp4_mlp(std::path::Path::new(&root))?;

        assert_eq!(report.source_swiglu_values, 17_408);
        assert_eq!(report.source_branch_values, 5_120);
        assert_eq!(report.weight_bytes, 150_425_600);
        assert_eq!(report.workspace_bytes, 711_168);
        assert_eq!(report.arena_bytes, 151_136_768);
        assert_eq!(report.padding_bytes, 0);
        assert!(report.normalized_values > 0);
        assert!(report.activation_codes > 0);
        assert!(report.activation_scales > 0);
        assert!(report.boundary_values > 0);
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.immutable_values > 0);
        assert!(report.maximum_absolute_error.is_finite());

        Ok(())
    }
}

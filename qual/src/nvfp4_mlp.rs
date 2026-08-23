//! Source-backed qualification for one resident NVFP4 MLP owner.

use crate::fp8_projection_oracle::{BF16_SENTINEL, BYTE_SENTINEL, bf16_to_f32, f32_to_bf16};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, Nvfp4MlpImmutable, Nvfp4MlpObservables, Nvfp4MlpProgram,
    Qwen35Nvfp4MlpProgram,
};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, MaterializedModelOptNvfp4Mlp,
    ModelOptNvfp4MlpBindings, Nvfp4MlpBindings, Qwen35_9B, Qwen38_27B,
};

const SOURCE_LAYER: usize = 55;
const QWEN35_SOURCE_LAYER: usize = 0;
const GROUP: usize = 16;
const QWEN38_ROUTES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 32, 64, 128, 1_024];

#[derive(Clone, Copy)]
struct MlpGeometry {
    hidden: usize,
    intermediate: usize,
    max_rows: usize,
    w4a4_batches: [bool; MAX_BATCH],
}

impl MlpGeometry {
    const fn uses_gate_w4a4(self, rows: usize) -> bool {
        if rows > MAX_BATCH {
            true
        } else {
            rows > 0 && self.w4a4_batches[rows - 1]
        }
    }

    const fn uses_down_w4a4(self, rows: usize) -> bool {
        rows > MAX_BATCH
    }

    const fn hidden_groups(self) -> usize {
        self.hidden / GROUP
    }

    const fn hidden_code_bytes(self) -> usize {
        self.hidden / 2
    }

    const fn intermediate_groups(self) -> usize {
        self.intermediate / GROUP
    }

    const fn intermediate_code_bytes(self) -> usize {
        self.intermediate / 2
    }
}

const QWEN38_GEOMETRY: MlpGeometry = MlpGeometry {
    hidden: Qwen38_27B::HIDDEN,
    intermediate: Qwen38_27B::INTERMEDIATE,
    max_rows: 1_024,
    w4a4_batches: [true, false, false, false, true, true, true, true],
};
const QWEN35_GEOMETRY: MlpGeometry = MlpGeometry {
    hidden: Qwen35_9B::HIDDEN,
    intermediate: Qwen35_9B::INTERMEDIATE,
    max_rows: MAX_BATCH,
    w4a4_batches: [true, false, true, true, true, true, true, true],
};

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
    /// Pre-MLP normalized BF16 values checked at every exact route.
    pub normalized_values: usize,
    /// Dynamic E2M1 codes checked bit-exactly on every W4A4 route.
    pub activation_codes: usize,
    /// Dynamic E4M3 scales checked bit-exactly on every W4A4 route.
    pub activation_scales: usize,
    /// SwiGLU values checked against complete or sampled source-weight formulas.
    pub source_swiglu_values: usize,
    /// Down values checked against complete or sampled source-weight formulas.
    pub source_branch_values: usize,
    /// Published residual and next-normalized values checked at every exact route.
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

/// Qualifies source-backed layer 55 at every exact decode and prefill width.
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
    let source_formula = SourceFormula {
        gate_weight: bindings.gate_up.gate_weight.bytes(),
        up_weight: bindings.gate_up.up_weight.bytes(),
        gate_scale: bindings.gate_up.gate_scale.codes(),
        up_scale: bindings.gate_up.up_scale.codes(),
        down_weight: bindings.down.weight.bytes(),
        down_scale: bindings.down.scale.codes(),
        gate_up_input_divisor: bindings.gate_up.input_scale_divisor,
        gate_up_weight_divisor: bindings.gate_up.weight_scale_divisor,
        down_input_divisor: bindings.down.input_scale_divisor,
        down_weight_divisor: bindings.down.weight_scale_divisor,
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
    for rows in QWEN38_ROUTES {
        let first_input = make_input(QWEN38_GEOMETRY, rows, 0);
        program.load_residual(&stream, rows, &first_input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, rows)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(QWEN38_GEOMETRY, rows, 1);
        program.load_residual(&stream, rows, &input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.replay(&stream, rows)?;
        let replay = program.qualification_observables(&stream)?;
        verify_immutable(
            rows,
            &program.qualification_immutable(&stream)?,
            expected_immutable,
            &mut report,
        )?;

        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, rows)?;
        let eager = program.qualification_observables(&stream)?;
        verify_immutable(
            rows,
            &program.qualification_immutable(&stream)?,
            expected_immutable,
            &mut report,
        )?;

        verify_seams::<Qwen38_27B>(
            QWEN38_GEOMETRY,
            rows,
            BoundarySource {
                input: &input,
                input_norm: &input_norm,
                next_norm: &next_norm,
                gate_up_input_scale_divisor: bindings.gate_up.input_scale_divisor,
                down_input_scale_divisor: bindings.down.input_scale_divisor,
            },
            &replay,
            &mut report,
        )?;
        if rows == 1 {
            verify_source_formula(QWEN38_GEOMETRY, source_formula, &replay, &mut report)?;
        } else if rows > MAX_BATCH {
            verify_prefill_source_samples(
                QWEN38_GEOMETRY,
                rows,
                source_formula,
                &replay,
                &mut report,
            )?;
        }
        verify_replay(QWEN38_GEOMETRY, rows, &eager, &replay, &mut report)?;
        verify_replacement_input(QWEN38_GEOMETRY, rows, &first, &replay)?;
        verify_inactive(QWEN38_GEOMETRY, rows, &replay, &mut report)?;
        verify_inactive(QWEN38_GEOMETRY, rows, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying {}",
                route_name(rows)
            )));
        }
    }

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;

    Ok(report)
}

/// Qualifies source-backed Qwen3.5 layer 0 at every exact decode batch.
pub fn qualify_qwen35_nvfp4_mlp(
    root: &Path,
) -> Result<Nvfp4MlpQualification, Nvfp4MlpQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let bindings = ModelOptNvfp4MlpBindings::bind(snapshot.as_ref(), QWEN35_SOURCE_LAYER)?;
    let materialized = bindings.materialize()?;
    let input_norm = materialized.input_norm.words().collect::<Vec<_>>();
    let next_norm = materialized.next_norm.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let program =
        Qwen35Nvfp4MlpProgram::from_snapshot(&context, snapshot.clone(), QWEN35_SOURCE_LAYER)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses()?;
    let expected_immutable = ExpectedImmutable {
        input_norm: &input_norm,
        gate_weight_codes: materialized.gate_up.gate_weight_e2m1,
        up_weight_codes: materialized.gate_up.up_weight_e2m1,
        gate_up_weight_scales: &materialized.gate_up.scale_e4m3_swizzled,
        down_weight_codes: materialized.down.weight_e2m1,
        down_weight_scales: &materialized.down.scale_e4m3_swizzled,
        next_norm: &next_norm,
    };
    let source = SourceFormula {
        gate_weight: bindings.gate.weight.bytes(),
        up_weight: bindings.up.weight.bytes(),
        gate_scale: bindings.gate.block_scale.codes(),
        up_scale: bindings.up.block_scale.codes(),
        down_weight: bindings.down.weight.bytes(),
        down_scale: bindings.down.block_scale.codes(),
        gate_up_input_divisor: materialized.gate_up.input_scale_divisor,
        gate_up_weight_divisor: materialized.gate_up.weight_scale_divisor,
        down_input_divisor: materialized.down.input_scale_divisor,
        down_weight_divisor: materialized.down.weight_scale_divisor,
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

    require_qwen35_scales(&program, &materialized)?;
    for batch in 1..=MAX_BATCH {
        let first_input = make_input(QWEN35_GEOMETRY, batch, 0);
        program.load_residual(&stream, batch, &first_input)?;
        program.qualification_reset_outputs(&stream, BYTE_SENTINEL)?;
        program.launch_eager(&stream, batch)?;
        let first = program.qualification_observables(&stream)?;

        let input = make_input(QWEN35_GEOMETRY, batch, 1);
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

        verify_seams::<Qwen35_9B>(
            QWEN35_GEOMETRY,
            batch,
            BoundarySource {
                input: &input,
                input_norm: &input_norm,
                next_norm: &next_norm,
                gate_up_input_scale_divisor: materialized.gate_up.input_scale_divisor,
                down_input_scale_divisor: materialized.down.input_scale_divisor,
            },
            &replay,
            &mut report,
        )?;
        if batch == 1 {
            verify_source_formula(QWEN35_GEOMETRY, source, &replay, &mut report)?;
        }
        verify_replay(QWEN35_GEOMETRY, batch, &eager, &replay, &mut report)?;
        verify_replacement_input(QWEN35_GEOMETRY, batch, &first, &replay)?;
        verify_inactive(QWEN35_GEOMETRY, batch, &replay, &mut report)?;
        verify_inactive(QWEN35_GEOMETRY, batch, &eager, &mut report)?;

        if program.base_address() != stable_base
            || program.qualification_addresses()? != stable_addresses
        {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "Qwen3.5 owner addresses changed while qualifying B={batch}"
            )));
        }
    }

    verify_no_qwen35_device_allocation(&program, &stream)?;
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

fn require_qwen35_scales(
    program: &Qwen35Nvfp4MlpProgram,
    materialized: &MaterializedModelOptNvfp4Mlp<'_>,
) -> Result<(), Nvfp4MlpQualificationError> {
    let actual_source = program.qualification_source_scales().map(f32::to_bits);
    let expected_source = [
        materialized.gate_up_input_scale,
        materialized.gate_up_weight_scale_2,
        materialized.down_input_scale,
        materialized.down_weight_scale_2,
    ]
    .map(f32::to_bits);
    if actual_source != expected_source {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "Qwen3.5 source scales differ: device-owner={actual_source:?}, source={expected_source:?}"
        )));
    }

    let actual_divisors = program.qualification_divisors().map(f32::to_bits);
    let expected_divisors = [
        materialized.gate_up.input_scale_divisor,
        materialized.gate_up.weight_scale_divisor,
        materialized.down.input_scale_divisor,
        materialized.down.weight_scale_divisor,
    ]
    .map(f32::to_bits);
    if actual_divisors != expected_divisors {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "Qwen3.5 runtime divisors differ: device-owner={actual_divisors:?}, materialized={expected_divisors:?}"
        )));
    }

    Ok(())
}

fn make_input(geometry: MlpGeometry, batch: usize, salt: usize) -> Vec<u16> {
    const PATTERN: [f32; 16] = [
        0.875, -0.875, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
        0.0, 0.5, -0.25, 0.125,
    ];
    (0..batch * geometry.hidden)
        .map(|index| f32_to_bf16(PATTERN[(index + salt * 5 + index / geometry.hidden) & 15]))
        .collect()
}

#[derive(Clone, Copy)]
struct BoundarySource<'a> {
    input: &'a [u16],
    input_norm: &'a [u16],
    next_norm: &'a [u16],
    gate_up_input_scale_divisor: f32,
    down_input_scale_divisor: f32,
}

fn verify_seams<A: Arch>(
    geometry: MlpGeometry,
    batch: usize,
    source: BoundarySource<'_>,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    for token in 0..batch {
        let hidden_begin = token * geometry.hidden;
        let hidden_end = hidden_begin + geometry.hidden;
        let intermediate_begin = token * geometry.intermediate;
        let intermediate_end = intermediate_begin + geometry.intermediate;
        let normalized =
            rms_norm_oracle::<A>(&source.input[hidden_begin..hidden_end], source.input_norm);
        compare_close_slice(
            "pre-MLP RMSNorm",
            batch,
            token,
            &observed.normalized[hidden_begin..hidden_end],
            &normalized,
            &mut report.maximum_absolute_error,
        )?;

        if geometry.uses_gate_w4a4(batch) {
            let (codes, scales) = quantize_oracle(
                &observed.normalized[hidden_begin..hidden_end],
                source.gate_up_input_scale_divisor,
            )?;
            require_equal_u8(
                "gate/up activation code",
                batch,
                token,
                &observed.gate_up_activation_codes[token * geometry.hidden_code_bytes()
                    ..(token + 1) * geometry.hidden_code_bytes()],
                &codes,
            )?;
            require_equal_u8(
                "gate/up activation scale",
                batch,
                token,
                &observed.gate_up_activation_scales
                    [token * geometry.hidden_groups()..(token + 1) * geometry.hidden_groups()],
                &scales,
            )?;
            report.activation_codes += geometry.hidden_code_bytes();
            report.activation_scales += geometry.hidden_groups();
        }

        require_active_written(
            "SwiGLU",
            batch,
            token,
            &observed.swiglu[intermediate_begin..intermediate_end],
        )?;
        if geometry.uses_down_w4a4(batch) {
            let (codes, scales) = quantize_oracle(
                &observed.swiglu[intermediate_begin..intermediate_end],
                source.down_input_scale_divisor,
            )?;
            require_equal_u8(
                "down activation code",
                batch,
                token,
                &observed.down_activation_codes[token * geometry.intermediate_code_bytes()
                    ..(token + 1) * geometry.intermediate_code_bytes()],
                &codes,
            )?;
            require_equal_u8(
                "down activation scale",
                batch,
                token,
                &observed.down_activation_scales[token * geometry.intermediate_groups()
                    ..(token + 1) * geometry.intermediate_groups()],
                &scales,
            )?;
            report.activation_codes += geometry.intermediate_code_bytes();
            report.activation_scales += geometry.intermediate_groups();
        }
        require_active_written(
            "down branch",
            batch,
            token,
            &observed.branch[hidden_begin..hidden_end],
        )?;
        let residual = source.input[hidden_begin..hidden_end]
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
                "residual publication at {}, row={token}, column={relative} differs",
                route_name(batch)
            )));
        }
        let next = rms_norm_oracle::<A>(&residual, source.next_norm);
        compare_close_slice(
            "next RMSNorm",
            batch,
            token,
            &observed.next_normalized[hidden_begin..hidden_end],
            &next,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.normalized_values += batch * geometry.hidden;
    report.boundary_values += batch * geometry.hidden * 2;

    Ok(())
}

#[derive(Clone, Copy)]
struct SourceFormula<'a> {
    gate_weight: &'a [u8],
    up_weight: &'a [u8],
    gate_scale: &'a [u8],
    up_scale: &'a [u8],
    down_weight: &'a [u8],
    down_scale: &'a [u8],
    gate_up_input_divisor: f32,
    gate_up_weight_divisor: f32,
    down_input_divisor: f32,
    down_weight_divisor: f32,
}

fn verify_source_formula(
    geometry: MlpGeometry,
    source: SourceFormula<'_>,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    let (activation_codes, activation_scales) = quantize_oracle(
        &observed.normalized[..geometry.hidden],
        source.gate_up_input_divisor,
    )?;
    let activation = QuantizedActivation {
        codes: &activation_codes,
        scales: &activation_scales,
        scale_divisor: source.gate_up_input_divisor,
    };
    for row in 0..geometry.intermediate {
        let gate = nvfp4_dot_w4a4(
            activation,
            source.gate_weight,
            source.gate_scale,
            row,
            geometry.hidden,
            source.gate_up_weight_divisor,
        )?;
        let up = nvfp4_dot_w4a4(
            activation,
            source.up_weight,
            source.up_scale,
            row,
            geometry.hidden,
            source.gate_up_weight_divisor,
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

    for row in 0..geometry.hidden {
        let expected = nvfp4_dot_a16(
            &observed.swiglu[..geometry.intermediate],
            source.down_weight,
            source.down_scale,
            row,
            geometry.intermediate,
            source.down_weight_divisor,
        )?;
        require_close(
            "source down projection",
            row,
            bf16_to_f32(observed.branch[row]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }

    report.source_swiglu_values += geometry.intermediate;
    report.source_branch_values += geometry.hidden;

    Ok(())
}

fn verify_prefill_source_samples(
    geometry: MlpGeometry,
    rows: usize,
    source: SourceFormula<'_>,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    let token = rows - 1;
    let hidden_code_begin = token * geometry.hidden_code_bytes();
    let hidden_scale_begin = token * geometry.hidden_groups();
    let gate_activation = QuantizedActivation {
        codes: &observed.gate_up_activation_codes
            [hidden_code_begin..hidden_code_begin + geometry.hidden_code_bytes()],
        scales: &observed.gate_up_activation_scales
            [hidden_scale_begin..hidden_scale_begin + geometry.hidden_groups()],
        scale_divisor: source.gate_up_input_divisor,
    };
    for column in sample_columns(geometry.intermediate) {
        let gate = nvfp4_dot_w4a4(
            gate_activation,
            source.gate_weight,
            source.gate_scale,
            column,
            geometry.hidden,
            source.gate_up_weight_divisor,
        )?;
        let up = nvfp4_dot_w4a4(
            gate_activation,
            source.up_weight,
            source.up_scale,
            column,
            geometry.hidden,
            source.gate_up_weight_divisor,
        )?;
        let gate = f64::from(bf16_to_f32(f32_to_bf16(gate as f32)));
        let up = f64::from(bf16_to_f32(f32_to_bf16(up as f32)));
        let expected = gate / (1.0 + (-gate).exp()) * up;
        require_close(
            &format!("source prefill SwiGLU at {}", route_name(rows)),
            column,
            bf16_to_f32(observed.swiglu[token * geometry.intermediate + column]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
        report.source_swiglu_values += 1;
    }

    let intermediate_code_begin = token * geometry.intermediate_code_bytes();
    let intermediate_scale_begin = token * geometry.intermediate_groups();
    let down_activation = QuantizedActivation {
        codes: &observed.down_activation_codes
            [intermediate_code_begin..intermediate_code_begin + geometry.intermediate_code_bytes()],
        scales: &observed.down_activation_scales
            [intermediate_scale_begin..intermediate_scale_begin + geometry.intermediate_groups()],
        scale_divisor: source.down_input_divisor,
    };
    for column in sample_columns(geometry.hidden) {
        let expected = nvfp4_dot_w4a4(
            down_activation,
            source.down_weight,
            source.down_scale,
            column,
            geometry.intermediate,
            source.down_weight_divisor,
        )?;
        require_close(
            &format!("source prefill down projection at {}", route_name(rows)),
            column,
            bf16_to_f32(observed.branch[token * geometry.hidden + column]),
            expected,
            &mut report.maximum_absolute_error,
        )?;
        report.source_branch_values += 1;
    }

    Ok(())
}

const fn sample_columns(width: usize) -> [usize; 8] {
    [
        0,
        1,
        width / 8,
        width / 4,
        width / 2,
        3 * width / 4,
        width - 2,
        width - 1,
    ]
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
    if !input.len().is_multiple_of(GROUP) {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "oracle input width {} is not divisible by {GROUP}",
            input.len()
        )));
    }
    let groups = input.len() / GROUP;
    let mut codes = vec![0u8; input.len() / 2];
    let mut scales = vec![0u8; groups];

    for group in 0..groups {
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
    rows: usize,
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
                    "{} immutable plane `{}` differs at value {index}",
                    route_name(rows),
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
    geometry: MlpGeometry,
    rows: usize,
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
                    "{} graph plane `{}` differs at value {index}",
                    route_name(rows),
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
    same!(down_activation_codes);
    same!(down_activation_scales);
    same!(branch);
    same!(residual_output);
    same!(next_normalized);

    report.graph_replay_values += rows * (5 * geometry.hidden + geometry.intermediate);
    if geometry.uses_gate_w4a4(rows) {
        report.graph_replay_values +=
            rows * (geometry.hidden_code_bytes() + geometry.hidden_groups());
    }
    if geometry.uses_down_w4a4(rows) {
        report.graph_replay_values +=
            rows * (geometry.intermediate_code_bytes() + geometry.intermediate_groups());
    }

    Ok(())
}

fn verify_replacement_input(
    geometry: MlpGeometry,
    rows: usize,
    first: &Nvfp4MlpObservables,
    replay: &Nvfp4MlpObservables,
) -> Result<(), Nvfp4MlpQualificationError> {
    let active = rows * geometry.hidden;
    if first.residual_input[..active] == replay.residual_input[..active]
        || first.residual_output[..active] == replay.residual_output[..active]
    {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "{} graph replay did not observe replacement residual rows",
            route_name(rows)
        )));
    }

    Ok(())
}

fn verify_inactive(
    geometry: MlpGeometry,
    rows: usize,
    observed: &Nvfp4MlpObservables,
    report: &mut Nvfp4MlpQualification,
) -> Result<(), Nvfp4MlpQualificationError> {
    let hidden_begin = rows * geometry.hidden;
    let intermediate_begin = rows * geometry.intermediate;
    for (role, values) in [
        ("normalized", &observed.normalized[hidden_begin..]),
        ("branch", &observed.branch[hidden_begin..]),
        ("residual output", &observed.residual_output[hidden_begin..]),
        ("next normalized", &observed.next_normalized[hidden_begin..]),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BF16_SENTINEL) {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "{} modified inactive {role} value {relative}",
                route_name(rows)
            )));
        }
    }
    if let Some(relative) = observed.swiglu[intermediate_begin..]
        .iter()
        .position(|&value| value != BF16_SENTINEL)
    {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "{} modified inactive SwiGLU value {relative}",
            route_name(rows)
        )));
    }

    let gate_code_begin = if geometry.uses_gate_w4a4(rows) {
        rows * geometry.hidden_code_bytes()
    } else {
        0
    };
    let gate_scale_begin = if geometry.uses_gate_w4a4(rows) {
        rows * geometry.hidden_groups()
    } else {
        0
    };
    let down_code_begin = if geometry.uses_down_w4a4(rows) {
        rows * geometry.intermediate_code_bytes()
    } else {
        0
    };
    let down_scale_begin = if geometry.uses_down_w4a4(rows) {
        rows * geometry.intermediate_groups()
    } else {
        0
    };
    for (role, values) in [
        (
            "gate/up activation code",
            &observed.gate_up_activation_codes[gate_code_begin..],
        ),
        (
            "gate/up activation scale",
            &observed.gate_up_activation_scales[gate_scale_begin..],
        ),
        (
            "down activation code",
            &observed.down_activation_codes[down_code_begin..],
        ),
        (
            "down activation scale",
            &observed.down_activation_scales[down_scale_begin..],
        ),
    ] {
        if let Some(relative) = values.iter().position(|&value| value != BYTE_SENTINEL) {
            return Err(Nvfp4MlpQualificationError::Mismatch(format!(
                "{} modified inactive {role} value {relative}",
                route_name(rows)
            )));
        }
    }

    let inactive_scratch = observed.gate_up_activation_codes.len() - gate_code_begin
        + observed.gate_up_activation_scales.len()
        - gate_scale_begin
        + observed.down_activation_codes.len()
        - down_code_begin
        + observed.down_activation_scales.len()
        - down_scale_begin;
    report.inactive_values += (geometry.max_rows - rows)
        * (4 * geometry.hidden + geometry.intermediate)
        + inactive_scratch;

    Ok(())
}

fn verify_no_device_allocation(
    program: &Nvfp4MlpProgram,
    stream: &tuisko_gpu::CudaStream,
) -> Result<(), Nvfp4MlpQualificationError> {
    program.replay(stream, 1_024)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for _ in 0..4 {
        for rows in QWEN38_ROUTES {
            program.replay(stream, rows)?;
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

fn route_name(rows: usize) -> String {
    if rows <= MAX_BATCH {
        format!("B={rows}")
    } else {
        format!("T={rows}")
    }
}

fn verify_no_qwen35_device_allocation(
    program: &Qwen35Nvfp4MlpProgram,
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
            "Qwen3.5 device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    Ok(())
}

fn require_active_written(
    role: &str,
    batch: usize,
    token: usize,
    values: &[u16],
) -> Result<(), Nvfp4MlpQualificationError> {
    if let Some(column) = values.iter().position(|&value| value == BF16_SENTINEL) {
        return Err(Nvfp4MlpQualificationError::Mismatch(format!(
            "{} {role} row={token}, column={column} retained its sentinel",
            route_name(batch)
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
            "{role} at {}, row={token}, column={column}: device={:#04x}, oracle={:#04x}",
            route_name(batch),
            actual[column],
            expected[column],
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
            &format!("{role} at {}, row={token}", route_name(batch)),
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
        Nvfp4MlpQualificationError, QWEN35_GEOMETRY, QWEN35_SOURCE_LAYER, QWEN38_GEOMETRY,
        SOURCE_LAYER, decode_e2m1, decode_e4m3fn, qualify_nvfp4_mlp, qualify_qwen35_nvfp4_mlp,
    };

    #[test]
    fn independent_codecs_and_route_table_are_pinned() {
        assert_eq!(decode_e2m1(0x07), 6.0);
        assert_eq!(decode_e2m1(0x0f), -6.0);
        assert_eq!(decode_e4m3fn(0x01).unwrap(), 2.0f32.powi(-9));
        assert_eq!(decode_e4m3fn(0x38).unwrap(), 1.0);
        assert_eq!(decode_e4m3fn(0x40).unwrap(), 2.0);
        assert_eq!(
            (1..=8)
                .map(|batch| QWEN38_GEOMETRY.uses_gate_w4a4(batch))
                .collect::<Vec<_>>(),
            [true, false, false, false, true, true, true, true],
        );
        assert_eq!(
            [32, 64, 128, 1_024].map(|rows| QWEN38_GEOMETRY.uses_gate_w4a4(rows)),
            [true; 4],
        );
        assert_eq!(
            [32, 64, 128, 1_024].map(|rows| QWEN38_GEOMETRY.uses_down_w4a4(rows)),
            [true; 4],
        );
        assert_eq!(SOURCE_LAYER, 55);
        assert_eq!(
            (1..=8)
                .map(|batch| QWEN35_GEOMETRY.uses_gate_w4a4(batch))
                .collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true],
        );
        assert_eq!(QWEN35_SOURCE_LAYER, 0);
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

        assert_eq!(report.source_swiglu_values, 17_440);
        assert_eq!(report.source_branch_values, 5_152);
        assert_eq!(report.weight_bytes, 150_425_600);
        assert_eq!(report.workspace_bytes, 101_056_512);
        assert_eq!(report.arena_bytes, 251_482_112);
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

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT and an exclusive NVIDIA compute-capability 12.0 device"]
    fn qwen35_source_layer0_matches_complete_oracles_and_graph_replay()
    -> Result<(), Nvfp4MlpQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            Nvfp4MlpQualificationError::Mismatch(
                "TUISKO_QWEN35_SNAPSHOT is required for the source-backed gate".to_string(),
            )
        })?;
        let report = qualify_qwen35_nvfp4_mlp(std::path::Path::new(&root))?;

        assert_eq!(report.source_swiglu_values, 12_288);
        assert_eq!(report.source_branch_values, 4_096);
        assert_eq!(report.weight_bytes, 84_951_040);
        assert_eq!(report.workspace_bytes, 542_720);
        assert_eq!(report.arena_bytes, 85_493_760);
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

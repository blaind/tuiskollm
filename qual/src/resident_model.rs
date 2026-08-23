//! Source-backed qualification for the complete resident text model.

use crate::fp8_projection_oracle::{bf16_to_f32, decode_e4m3fn, quantize_oracle};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{EngineError, MAX_BATCH, ResidentModelObservables, ResidentModelProgram};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, FullAttentionPostBindings, Nvfp4DownBindings,
    Nvfp4GateUpBindings, Qwen38_27B, TextEndpointBindings,
};

const SENTINEL: u8 = 0x7d;
const BF16_SENTINEL: u16 = 0x7d7d;
const F32_SENTINEL_BITS: u32 = 0x7d7d7d7d;
const ROTARY_PAIRS: usize = 32;
const TABLE_STRIDE: usize = 3;
const SELECTED_LOGIT_ROWS: [usize; 5] = [0, 1, 31_337, 131_071, Qwen38_27B::VOCAB - 1];

/// Failure of the complete source-backed resident-model gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentModelQualificationError {
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
    /// Device behavior disagreed with the independent source-backed formula.
    #[error("resident-model qualification failed: {0}")]
    Mismatch(String),
}

/// Counts and worst numeric error from the complete resident-model gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentModelQualification {
    /// Exact source scalar values checked before execution.
    pub source_scalars: usize,
    /// Final residual, normalization, quantization, and selected-logit values checked.
    pub oracle_values: usize,
    /// Complete mutable owner values reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Inactive workspace and persistent values verified unchanged.
    pub inactive_values: usize,
    /// Persistent values checked across physical routing and one-slot reset.
    pub slot_control_values: usize,
    /// Largest absolute difference from a BF16 or FP64 oracle.
    pub maximum_absolute_error: f32,
}

/// Qualifies every complete-model `B=1..=8` graph against eager replay and source formulas.
pub fn qualify_resident_model(
    root: &Path,
) -> Result<ResidentModelQualification, ResidentModelQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let endpoint = TextEndpointBindings::bind(snapshot.as_ref())?;
    let final_norm = endpoint.final_norm.words().collect::<Vec<_>>();
    let lm_head_codes = endpoint.lm_head.codes();
    let lm_head_scales = endpoint.lm_head_scale.words().collect::<Vec<_>>();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = ResidentModelProgram::from_snapshot(&context, snapshot.clone())?;
    verify_owner(&program)?;
    let stable_base = program.base_address();
    let stable_addresses = program.qualification_addresses();
    verify_scalars(&program, snapshot.as_ref())?;
    let mut report = ResidentModelQualification {
        source_scalars: 16 * 2 + 56 * 4,
        oracle_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        slot_control_values: 0,
        maximum_absolute_error: 0.0,
    };

    for batch in 1..=MAX_BATCH {
        let slots = route_slots(batch);
        prepare_run(&mut program, &stream, batch, slots)?;
        program.launch_eager(&stream, batch)?;
        let eager = program.qualification_observables(&stream)?;

        prepare_run(&mut program, &stream, batch, slots)?;
        program.replay(&stream, batch)?;
        let replay = program.qualification_observables(&stream)?;

        verify_replay(batch, &eager, &replay, &mut report)?;
        verify_final_oracle(
            batch,
            &final_norm,
            lm_head_codes,
            &lm_head_scales,
            &replay,
            &mut report,
        )?;
        verify_inactive(batch, slots, &replay, &mut report)?;
        if program.base_address() != stable_base
            || program.qualification_addresses() != stable_addresses
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying B={batch}"
            )));
        }
    }
    verify_slot_reset(&mut program, &stream, &mut report)?;

    verify_no_device_allocation(&program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn verify_owner(program: &ResidentModelProgram) -> Result<(), ResidentModelQualificationError> {
    if program.resident_weight_bytes() != 19_103_682_560
        || program.history_bytes() != 23_592_960
        || program.state_bytes() != 1_207_959_552
        || program.cache_bytes() != 50_331_648
        || program.workspace_bytes() != 5_910_272
        || program.padding_bytes() != 16_640
        || program.arena_bytes() != 20_391_493_632
        || program.host_stager_bytes() != 81_920
        || program.batch_capacity() != 8
        || program.context_capacity() != 192
        || program.persistent_slot_bytes() != 160_235_520
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "owner byte or route accounting differs from the admitted layout".to_string(),
        ));
    }
    let addresses = program.qualification_addresses();
    if addresses.len() != 1_126 || addresses.iter().copied().collect::<BTreeSet<_>>().len() != 1_126
    {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "owner exposes {} addresses, expected 1,126 unique addresses",
            addresses.len()
        )));
    }
    Ok(())
}

fn verify_scalars(
    program: &ResidentModelProgram,
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> Result<(), ResidentModelQualificationError> {
    let expected_cache = (0..Qwen38_27B::LAYERS)
        .filter(|layer| (layer + 1).is_multiple_of(Qwen38_27B::FULL_ATTENTION_INTERVAL))
        .map(|layer| {
            let source = FullAttentionPostBindings::bind(snapshot, layer)?;
            Ok([
                bf16_to_f32(source.key_cache_scale_bf16),
                bf16_to_f32(source.value_cache_scale_bf16),
            ])
        })
        .collect::<Result<Vec<_>, CheckpointError>>()?;
    let actual_cache = program.qualification_cache_scales();
    if !same_f32_arrays(&actual_cache, &expected_cache) {
        return Err(ResidentModelQualificationError::Mismatch(
            "attention cache scales differ from their BF16 sources".to_string(),
        ));
    }

    let expected_nvfp4 = (0..56)
        .map(|layer| {
            let gate_up = Nvfp4GateUpBindings::bind(snapshot, layer)?;
            let down = Nvfp4DownBindings::bind(snapshot, layer)?;
            Ok([
                gate_up.input_scale_divisor,
                gate_up.weight_scale_divisor,
                down.input_scale_divisor,
                down.weight_scale_divisor,
            ])
        })
        .collect::<Result<Vec<_>, CheckpointError>>()?;
    let actual_nvfp4 = program.qualification_nvfp4_divisors();
    if !same_f32_arrays(&actual_nvfp4, &expected_nvfp4) {
        return Err(ResidentModelQualificationError::Mismatch(
            "NVFP4 divisors differ from their F32 sources".to_string(),
        ));
    }
    Ok(())
}

fn same_f32_arrays<const N: usize>(actual: &[[f32; N]], expected: &[[f32; N]]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual
                .iter()
                .zip(expected)
                .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits())
        })
}

fn prepare_run(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    batch: usize,
    slots: &[usize],
) -> Result<(), ResidentModelQualificationError> {
    program.reset_state(stream)?;
    program.qualification_reset_workspace(stream, SENTINEL)?;
    let token_ids = (0..batch)
        .map(|slot| 100u32 + slot as u32 * 17)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    program.load_slot_routes(stream, slots)?;
    program.load_decode_state(
        stream,
        batch,
        &[0; MAX_BATCH][..batch],
        &[1.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
        &[0.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
    )?;
    Ok(())
}

fn route_slots(batch: usize) -> &'static [usize] {
    const ROUTES: [usize; MAX_BATCH] = [7, 0, 6, 1, 5, 2, 4, 3];
    &ROUTES[..batch]
}

fn verify_final_oracle(
    batch: usize,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    observed: &ResidentModelObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    for token in 0..batch {
        let begin = token * hidden;
        let end = begin + hidden;
        let residual = observed.mixer_residual[begin..end]
            .iter()
            .zip(&observed.mlp_branch[begin..end])
            .map(|(&residual, &branch)| f32_to_bf16(bf16_to_f32(residual) + bf16_to_f32(branch)))
            .collect::<Vec<_>>();
        compare_exact(
            "final residual",
            &observed.residual_a[begin..end],
            &residual,
        )?;
        let normalized = rms_norm_oracle::<Qwen38_27B>(&residual, final_norm);
        compare_bf16(
            "final RMSNorm",
            &observed.mixer_normalized[begin..end],
            &normalized,
            &mut report.maximum_absolute_error,
        )?;
        let quantized =
            quantize_oracle(&normalized).map_err(ResidentModelQualificationError::Mismatch)?;
        compare_exact(
            "endpoint activation codes",
            &observed.activation_codes[begin..end],
            &quantized.codes,
        )?;
        if observed.activation_scales[token].to_bits() != quantized.scale.to_bits() {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "endpoint activation scale differs at token {token}"
            )));
        }
        for &row in &SELECTED_LOGIT_ROWS {
            let weight_begin = row * hidden;
            let expected = fp8_dot(
                &quantized.codes,
                quantized.scale,
                &lm_head_codes[weight_begin..weight_begin + hidden],
                lm_head_scales[row],
            )?;
            let actual = bf16_to_f32(observed.logits[token * Qwen38_27B::VOCAB + row]);
            require_close(
                "selected LM-head row",
                token * SELECTED_LOGIT_ROWS.len() + row,
                actual,
                expected,
                &mut report.maximum_absolute_error,
            )?;
        }
    }
    report.oracle_values += batch * (3 * hidden + 1 + SELECTED_LOGIT_ROWS.len());
    Ok(())
}

fn fp8_dot(
    activations: &[u8],
    activation_scale: f32,
    weights: &[u8],
    weight_scale: u16,
) -> Result<f64, ResidentModelQualificationError> {
    let dot = activations
        .iter()
        .zip(weights)
        .try_fold(0.0f64, |sum, (&activation, &weight)| {
            Ok::<_, String>(
                sum + f64::from(decode_e4m3fn(activation)?) * f64::from(decode_e4m3fn(weight)?),
            )
        })
        .map_err(ResidentModelQualificationError::Mismatch)?;
    Ok(dot * f64::from(activation_scale) * f64::from(bf16_to_f32(weight_scale)))
}

fn verify_replay(
    batch: usize,
    eager: &ResidentModelObservables,
    replay: &ResidentModelObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    macro_rules! same {
        ($field:ident) => {
            if let Some(index) = replay
                .$field
                .iter()
                .zip(&eager.$field)
                .position(|(actual, expected)| actual != expected)
            {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} graph plane `{}` differs at value {index}",
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
                .zip(&eager.$field)
                .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} graph plane `{}` differs at value {index}",
                    stringify!($field)
                )));
            }
        };
    }
    same!(residual_a);
    same!(residual_b);
    same!(mixer_residual);
    same!(mixer_normalized);
    same!(mlp_normalized);
    same!(activation_codes);
    same_f32!(activation_scales);
    same!(nvfp4_activation_codes);
    same!(nvfp4_activation_scales);
    same!(projected);
    same_f32!(log_decay);
    same_f32!(beta);
    same!(convolved);
    same!(recurrent_output);
    same_f32!(query);
    same_f32!(attention);
    same!(mixer_branch);
    same!(swiglu);
    same!(mlp_branch);
    same!(logits);
    same!(history);
    same_f32!(state);
    same!(key_pages);
    same!(value_pages);
    report.graph_replay_values += observable_values(replay);
    Ok(())
}

fn observable_values(observed: &ResidentModelObservables) -> usize {
    observed.residual_a.len()
        + observed.residual_b.len()
        + observed.mixer_residual.len()
        + observed.mixer_normalized.len()
        + observed.mlp_normalized.len()
        + observed.activation_codes.len()
        + observed.activation_scales.len()
        + observed.nvfp4_activation_codes.len()
        + observed.nvfp4_activation_scales.len()
        + observed.projected.len()
        + observed.log_decay.len()
        + observed.beta.len()
        + observed.convolved.len()
        + observed.recurrent_output.len()
        + observed.query.len()
        + observed.attention.len()
        + observed.mixer_branch.len()
        + observed.swiglu.len()
        + observed.mlp_branch.len()
        + observed.logits.len()
        + observed.history.len()
        + observed.state.len()
        + observed.key_pages.len()
        + observed.value_pages.len()
}

fn verify_inactive(
    batch: usize,
    slots: &[usize],
    observed: &ResidentModelObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    macro_rules! sentinel_u16 {
        ($field:ident, $width:expr) => {{
            let begin = batch * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != BF16_SENTINEL)
            {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_u8 {
        ($field:ident, $width:expr) => {{
            let begin = batch * $width;
            if observed.$field[begin..]
                .iter()
                .any(|&value| value != SENTINEL)
            {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    macro_rules! sentinel_f32 {
        ($field:ident, $width:expr) => {{
            let begin = batch * $width;
            if observed.$field[begin..]
                .iter()
                .any(|value| value.to_bits() != F32_SENTINEL_BITS)
            {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified inactive `{}` value",
                    stringify!($field)
                )));
            }
            observed.$field.len() - begin
        }};
    }
    let mut inactive = 0;
    inactive += sentinel_u16!(residual_a, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(residual_b, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mixer_residual, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mixer_normalized, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(mlp_normalized, Qwen38_27B::HIDDEN);
    // Shared FP8 scratch retains the widest earlier writer beyond the
    // endpoint prefix: the last dense down route writes INTERMEDIATE codes.
    inactive += sentinel_u8!(activation_codes, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_f32!(activation_scales, 1);
    inactive += sentinel_u8!(nvfp4_activation_codes, Qwen38_27B::HIDDEN / 2);
    inactive += sentinel_u8!(nvfp4_activation_scales, Qwen38_27B::HIDDEN / 16);
    // Layer 63 overwrites the attention prefix, while the wider layer-62 GDN
    // input projection remains observable in the tail.
    inactive += sentinel_u16!(projected, Qwen38_27B::GDN_INPUT_ROWS);
    inactive += sentinel_f32!(log_decay, Qwen38_27B::GDN_CONTROL_ROWS);
    inactive += sentinel_f32!(beta, Qwen38_27B::GDN_CONTROL_ROWS);
    inactive += sentinel_u16!(convolved, Qwen38_27B::GDN_QKV_ROWS);
    inactive += sentinel_u16!(recurrent_output, Qwen38_27B::GDN_VALUE_ROWS);
    inactive += sentinel_f32!(query, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_f32!(attention, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_u16!(mixer_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(swiglu, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_u16!(mlp_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(logits, Qwen38_27B::VOCAB);
    let persistent = verify_persistent_inactive(batch, slots, observed)?;
    inactive += persistent;
    report.slot_control_values += persistent;
    report.inactive_values += inactive;
    Ok(())
}

fn verify_persistent_inactive(
    batch: usize,
    active_slots: &[usize],
    observed: &ResidentModelObservables,
) -> Result<usize, ResidentModelQualificationError> {
    let history_per_slot = Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
    let state_per_slot =
        Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::LINEAR_HEAD_DIM * Qwen38_27B::LINEAR_HEAD_DIM;
    let cache_per_slot =
        TABLE_STRIDE * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let mut inactive = 0;
    for (layer, values) in observed
        .history
        .chunks_exact(MAX_BATCH * history_per_slot)
        .enumerate()
    {
        for slot in 0..MAX_BATCH {
            if active_slots.contains(&slot) {
                continue;
            }
            let begin = slot * history_per_slot;
            let end = begin + history_per_slot;
            if values[begin..end].iter().any(|&value| value != 0) {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified inactive GDN history slot {slot} at inventory layer {layer}"
                )));
            }
            inactive += history_per_slot;
        }
    }
    for (layer, values) in observed
        .state
        .chunks_exact(MAX_BATCH * state_per_slot)
        .enumerate()
    {
        for slot in 0..MAX_BATCH {
            if active_slots.contains(&slot) {
                continue;
            }
            let begin = slot * state_per_slot;
            let end = begin + state_per_slot;
            if values[begin..end].iter().any(|value| value.to_bits() != 0) {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified inactive GDN state slot {slot} at inventory layer {layer}"
                )));
            }
            inactive += state_per_slot;
        }
    }
    for (role, all) in [
        ("key", &observed.key_pages),
        ("value", &observed.value_pages),
    ] {
        for (layer, values) in all.chunks_exact(MAX_BATCH * cache_per_slot).enumerate() {
            for slot in 0..MAX_BATCH {
                if active_slots.contains(&slot) {
                    continue;
                }
                let begin = slot * cache_per_slot;
                let end = begin + cache_per_slot;
                if values[begin..end].iter().any(|&value| value != 0) {
                    return Err(ResidentModelQualificationError::Mismatch(format!(
                        "B={batch} modified inactive {role} cache slot {slot} at inventory layer {layer}"
                    )));
                }
                inactive += cache_per_slot;
            }
        }
    }
    Ok(inactive)
}

fn verify_slot_reset(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    prepare_run(program, stream, MAX_BATCH, route_slots(MAX_BATCH))?;
    program.replay(stream, MAX_BATCH)?;
    let before = program.qualification_observables(stream)?;
    let reset = [1, 6];
    for slot in reset {
        program.reset_slot(stream, slot)?;
    }
    let after = program.qualification_observables(stream)?;

    let history_per_slot = Qwen38_27B::GDN_QKV_ROWS * (Qwen38_27B::LINEAR_CONV_KERNEL_DIM - 1);
    let state_per_slot =
        Qwen38_27B::GDN_CONTROL_ROWS * Qwen38_27B::LINEAR_HEAD_DIM * Qwen38_27B::LINEAR_HEAD_DIM;
    let cache_per_slot =
        TABLE_STRIDE * Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let mut checked = 0;
    checked += verify_reset_u16(
        "GDN history",
        &before.history,
        &after.history,
        history_per_slot,
        &reset,
    )?;
    checked += verify_reset_f32(
        "GDN state",
        &before.state,
        &after.state,
        state_per_slot,
        &reset,
    )?;
    checked += verify_reset_u8(
        "key cache",
        &before.key_pages,
        &after.key_pages,
        cache_per_slot,
        &reset,
    )?;
    checked += verify_reset_u8(
        "value cache",
        &before.value_pages,
        &after.value_pages,
        cache_per_slot,
        &reset,
    )?;
    report.slot_control_values += checked;
    Ok(())
}

fn verify_reset_u8(
    role: &str,
    before: &[u8],
    after: &[u8],
    slot_width: usize,
    reset: &[usize],
) -> Result<usize, ResidentModelQualificationError> {
    verify_reset_chunks(role, before, after, slot_width, reset, |value| *value == 0)
}

fn verify_reset_u16(
    role: &str,
    before: &[u16],
    after: &[u16],
    slot_width: usize,
    reset: &[usize],
) -> Result<usize, ResidentModelQualificationError> {
    verify_reset_chunks(role, before, after, slot_width, reset, |value| *value == 0)
}

fn verify_reset_f32(
    role: &str,
    before: &[f32],
    after: &[f32],
    slot_width: usize,
    reset: &[usize],
) -> Result<usize, ResidentModelQualificationError> {
    verify_reset_chunks(role, before, after, slot_width, reset, |value| {
        value.to_bits() == 0
    })
}

fn verify_reset_chunks<T: PartialEq>(
    role: &str,
    before: &[T],
    after: &[T],
    slot_width: usize,
    reset: &[usize],
    is_zero: impl Fn(&T) -> bool,
) -> Result<usize, ResidentModelQualificationError> {
    if before.len() != after.len() || !(before.len()).is_multiple_of(MAX_BATCH * slot_width) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "{role} reset inventory has incompatible lengths"
        )));
    }
    for (layer, (before, after)) in before
        .chunks_exact(MAX_BATCH * slot_width)
        .zip(after.chunks_exact(MAX_BATCH * slot_width))
        .enumerate()
    {
        for slot in 0..MAX_BATCH {
            let begin = slot * slot_width;
            let end = begin + slot_width;
            if reset.contains(&slot) {
                if after[begin..end].iter().any(|value| !is_zero(value)) {
                    return Err(ResidentModelQualificationError::Mismatch(format!(
                        "{role} reset left nonzero slot {slot} at inventory layer {layer}"
                    )));
                }
            } else if after[begin..end] != before[begin..end] {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "{role} reset changed surviving slot {slot} at inventory layer {layer}"
                )));
            }
        }
    }
    Ok(after.len())
}

fn compare_exact<T: PartialEq>(
    role: &str,
    actual: &[T],
    expected: &[T],
) -> Result<(), ResidentModelQualificationError> {
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
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
) -> Result<(), ResidentModelQualificationError> {
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
) -> Result<(), ResidentModelQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    *maximum = maximum.max(error);
    let tolerance = 0.5f32.max(expected.abs() as f32 * 0.03);
    if !actual.is_finite() || error > tolerance {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "{role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn verify_no_device_allocation(
    program: &ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(), ResidentModelQualificationError> {
    program.replay(stream, MAX_BATCH)?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
        program.replay(stream, batch)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SELECTED_LOGIT_ROWS, qualify_resident_model};
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn source_model_matches_final_oracle_and_exact_graph_replay()
    -> Result<(), super::ResidentModelQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            super::ResidentModelQualificationError::Mismatch(
                "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_resident_model(&PathBuf::from(root))?;
        let active = (1..=8).sum::<usize>();
        assert_eq!(report.source_scalars, 256);
        assert_eq!(
            report.oracle_values,
            active * (3 * 5_120 + 1 + SELECTED_LOGIT_ROWS.len())
        );
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.slot_control_values > 0);
        assert!(report.maximum_absolute_error.is_finite());
        Ok(())
    }
}

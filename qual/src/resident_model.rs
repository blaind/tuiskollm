//! Source-backed qualification for the complete resident text model.

use crate::fp8_projection_oracle::{bf16_to_f32, decode_e4m3fn, quantize_oracle};
use crate::residual_norm::rms_norm_oracle;
use crate::{DeviceBenchmarkError, device_benchmark};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, MAX_BATCH, PagedKvSlotState, ResidentDecodeRoute, ResidentLoadMode,
    ResidentLongContextObservables, ResidentModelObservables, ResidentModelProgram,
    ResidentPrefillRoute,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::{
    ATTENTION_PAGE_SIZE, LONG_CONTEXT_GQA_MAX_PARTITIONS, LONG_CONTEXT_GQA_PARTITION_SIZE,
};
use tuisko_model::{
    Arch, CheckpointError, CheckpointSnapshot, FullAttentionPostBindings, Nvfp4DownBindings,
    Nvfp4GateUpBindings, Qwen38_27B, TextEndpointBindings,
};

const SENTINEL: u8 = 0x7d;
const BF16_SENTINEL: u16 = 0x7d7d;
const F32_SENTINEL_BITS: u32 = 0x7d7d7d7d;
const ROTARY_PAIRS: usize = 32;
const TABLE_STRIDE: usize = 3;
const LONG_ROUTE_LENGTHS: [usize; 6] = [193, 1_025, 4_097, 16_385, 65_537, 131_073];
const LONG_ORACLE_DIMENSIONS: [usize; 8] = [0, 31, 32, 127, 128, 223, 224, 255];
const PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1_024];
const PREFILL_TAIL_ROUTES: [(usize, usize, Option<usize>); 5] = [
    (32, 160, None),
    (64, 192, None),
    (128, 1, Some(8)),
    (128, 32_768, Some(16)),
    (1_024, 1_024, Some(4)),
];
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
    /// Long-context whole-model eager/graph cases checked.
    pub long_route_cases: usize,
    /// Independent long-attention seam values checked.
    pub long_oracle_values: usize,
    /// Exact from-empty and nonzero-prefix prompt routes checked eagerly and by graph replay.
    pub prefill_route_cases: usize,
    /// Largest absolute difference from a BF16 or FP64 oracle.
    pub maximum_absolute_error: f32,
}

/// Qualifies every complete-model `B=1..=8` graph against eager replay and source formulas.
pub fn qualify_resident_model(
    root: &Path,
) -> Result<ResidentModelQualification, ResidentModelQualificationError> {
    qualify_resident_model_with_mode(root, ResidentLoadMode::Selective)
}

fn qualify_resident_model_with_mode(
    root: &Path,
    load_mode: ResidentLoadMode,
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
    let mut program = match load_mode {
        ResidentLoadMode::Legacy => {
            ResidentModelProgram::from_snapshot_legacy(&context, snapshot.clone())?
        }
        ResidentLoadMode::Selective => {
            ResidentModelProgram::from_snapshot_selective(&context, snapshot.clone())?
        }
    };
    initialize_short_routes(&mut program, &stream)?;
    verify_owner(&program)?;
    let stable_base = program.base_address();
    let stable_kv_base = program.kv_base_address();
    let stable_addresses = program.qualification_addresses();
    let stable_route_addresses = program.qualification_kv_route_addresses();
    verify_scalars(&program, snapshot.as_ref())?;
    let mut report = ResidentModelQualification {
        source_scalars: 16 * 2 + 56 * 4,
        oracle_values: 0,
        graph_replay_values: 0,
        inactive_values: 0,
        slot_control_values: 0,
        long_route_cases: 0,
        long_oracle_values: 0,
        prefill_route_cases: 0,
        maximum_absolute_error: 0.0,
    };
    report.slot_control_values += verify_block_tables(&program, &stream)?;

    for batch in 1..=MAX_BATCH {
        let slots = route_slots(batch);
        let eager_route = prepare_run(&mut program, &stream, batch, slots)?;
        program.launch_eager(&stream, eager_route)?;
        let eager = program.qualification_observables(&stream)?;

        let replay_route = prepare_run(&mut program, &stream, batch, slots)?;
        program.replay(&stream, replay_route)?;
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
            || program.kv_base_address() != stable_kv_base
            || program.qualification_addresses() != stable_addresses
            || program.qualification_kv_route_addresses() != stable_route_addresses
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying B={batch}"
            )));
        }
    }
    verify_prefill_routes(
        &mut program,
        &stream,
        &final_norm,
        lm_head_codes,
        &lm_head_scales,
        stable_base,
        stable_kv_base,
        &stable_addresses,
        &stable_route_addresses,
        &mut report,
    )?;
    verify_prefill_tail_routes(
        &mut program,
        &stream,
        &final_norm,
        lm_head_codes,
        &lm_head_scales,
        stable_base,
        stable_kv_base,
        &stable_addresses,
        &stable_route_addresses,
        &mut report,
    )?;
    verify_slot_reset(&mut program, &stream, &mut report)?;
    report.slot_control_values += verify_block_tables(&program, &stream)?;
    report.slot_control_values += verify_dynamic_page_routes(&mut program, &stream)?;
    verify_long_context_routes(&mut program, &stream, &mut report)?;

    verify_no_device_allocation(&mut program, &stream)?;
    if program.qualification_kv_route_addresses() != stable_route_addresses {
        return Err(ResidentModelQualificationError::Mismatch(
            "page-router host addresses changed after route mutation".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn verify_prefill_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    stable_base: u64,
    stable_kv_base: u64,
    stable_addresses: &[usize],
    stable_route_addresses: &[usize; 2],
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let update = program.reserve_kv_slot_tokens(stream, 0, 1_024)?;
    if update.first_entry() != 3
        || update.entry_count() != 13
        || program.qualification_kv_page_count(0)? != 16
    {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "resident prefill slot owns {} pages after update {update:?}, expected 16 after adding entries 3..16",
            program.qualification_kv_page_count(0)?
        )));
    }

    for tokens in PREFILL_ROUTES {
        let eager_route = prepare_prefill_run(program, stream, tokens, 0)?;
        program.launch_prefill_eager(stream, eager_route)?;
        let eager = program.qualification_observables(stream)?;
        let eager_pages = read_prefill_cache_pages(program, stream, 0, tokens)?;

        let replay_route = prepare_prefill_run(program, stream, tokens, 0)?;
        if replay_route != eager_route {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} eager and replay route tokens differ"
            )));
        }
        program.replay_prefill(stream, replay_route)?;
        let replay = program.qualification_observables(stream)?;
        let replay_pages = read_prefill_cache_pages(program, stream, 0, tokens)?;

        verify_replay(tokens, &eager, &replay, report)?;
        if replay_pages != eager_pages {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} represented cache pages differ under graph replay"
            )));
        }
        if replay_pages.last().is_none_or(|(key, value)| {
            key.iter().all(|&code| code == 0) || value.iter().all(|&code| code == 0)
        }) {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} did not publish represented K/V values through its final cache page"
            )));
        }
        verify_prefill_final_oracle(
            tokens,
            final_norm,
            lm_head_codes,
            lm_head_scales,
            &replay,
            report,
        )?;
        verify_prefill_inactive(replay_route, &replay, report)?;
        report.graph_replay_values += replay_pages
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>();
        report.prefill_route_cases += 1;

        if program.base_address() != stable_base
            || program.kv_base_address() != stable_kv_base
            || program.qualification_addresses() != stable_addresses
            || program.qualification_kv_route_addresses() != *stable_route_addresses
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying T={tokens}"
            )));
        }
    }

    let tables = program.qualification_block_tables(stream)?;
    let used = tables
        .iter()
        .copied()
        .filter(|&page| page != u32::MAX)
        .collect::<BTreeSet<_>>();
    let guard_page = (0..3_438u32)
        .find(|page| !used.contains(page))
        .ok_or_else(|| {
            ResidentModelQualificationError::Mismatch(
                "resident prefill qualification has no unassigned cache guard page".to_string(),
            )
        })? as usize;
    let (key_guard, value_guard) = program.qualification_cache_page(stream, guard_page)?;
    if key_guard.iter().any(|&value| value != 0) || value_guard.iter().any(|&value| value != 0) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "resident prefill modified unassigned physical cache page {guard_page}"
        )));
    }
    report.inactive_values += key_guard.len() + value_guard.len();

    let released = program.truncate_kv_slot_tokens(stream, 0, 192)?;
    if released != 13 || program.qualification_kv_page_count(0)? != 3 {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "resident prefill slot retained {} pages after releasing {released}, expected 3 after releasing 13",
            program.qualification_kv_page_count(0)?
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_prefill_tail_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    stable_base: u64,
    stable_kv_base: u64,
    stable_addresses: &[usize],
    stable_route_addresses: &[usize; 2],
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let maximum_context = PREFILL_TAIL_ROUTES
        .iter()
        .map(|&(tokens, first_position, _)| first_position + tokens)
        .max()
        .expect("tail route inventory is nonempty");
    program.reserve_kv_slot_tokens(stream, 0, maximum_context)?;

    for (tokens, first_position, expected_partitions) in PREFILL_TAIL_ROUTES {
        let eager_route = prepare_prefill_tail_run(program, stream, tokens, first_position, 0)?;
        if eager_route.tokens() != tokens
            || eager_route.first_position() != first_position
            || eager_route.context_tokens() != first_position + tokens
            || eager_route.partition_capacity() != expected_partitions
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} first={first_position} selected an incorrect tail route"
            )));
        }
        program.launch_prefill_eager(stream, eager_route)?;
        let eager = program.qualification_observables(stream)?;
        let eager_pages =
            read_prefill_tail_cache_pages(program, stream, 0, first_position, tokens)?;

        let replay_route = prepare_prefill_tail_run(program, stream, tokens, first_position, 0)?;
        if replay_route != eager_route {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} first={first_position} eager and replay tail routes differ"
            )));
        }
        program.replay_prefill(stream, replay_route)?;
        let replay = program.qualification_observables(stream)?;
        let replay_pages =
            read_prefill_tail_cache_pages(program, stream, 0, first_position, tokens)?;

        verify_replay(tokens, &eager, &replay, report)?;
        if replay_pages != eager_pages {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} first={first_position} represented cache tail differs under graph replay"
            )));
        }
        if replay_pages.last().is_none_or(|(key, value)| {
            key.iter().all(|&code| code == 0) || value.iter().all(|&code| code == 0)
        }) {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "T={tokens} first={first_position} did not publish its represented cache tail"
            )));
        }
        verify_prefill_final_oracle(
            tokens,
            final_norm,
            lm_head_codes,
            lm_head_scales,
            &replay,
            report,
        )?;
        verify_prefill_inactive(replay_route, &replay, report)?;
        report.graph_replay_values += replay_pages
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>();
        report.prefill_route_cases += 1;

        if program.base_address() != stable_base
            || program.kv_base_address() != stable_kv_base
            || program.qualification_addresses() != stable_addresses
            || program.qualification_kv_route_addresses() != *stable_route_addresses
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "owner addresses changed while qualifying T={tokens} first={first_position}"
            )));
        }
    }

    program.truncate_kv_slot_tokens(stream, 0, 192)?;
    Ok(())
}

fn prepare_prefill_run(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    tokens: usize,
    slot: usize,
) -> Result<ResidentPrefillRoute, ResidentModelQualificationError> {
    program.reset_state(stream)?;
    program.qualification_reset_workspace(stream, SENTINEL)?;
    let token_ids = (0..tokens)
        .map(|token| 100u32 + (token % 251) as u32)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    let (rope_cos, rope_sin) = prefill_rope(tokens);
    Ok(program.load_prefill_state(stream, tokens, slot, &rope_cos, &rope_sin)?)
}

fn prepare_prefill_tail_run(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    tokens: usize,
    first_position: usize,
    slot: usize,
) -> Result<ResidentPrefillRoute, ResidentModelQualificationError> {
    program.reset_state(stream)?;
    program.qualification_reset_workspace(stream, SENTINEL)?;
    let token_ids = (0..tokens)
        .map(|token| 100u32 + ((first_position + token) % 251) as u32)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    let (rope_cos, rope_sin) = prefill_rope_at(first_position, tokens);
    Ok(program.load_prefill_tile_state(
        stream,
        tokens,
        slot,
        first_position,
        &rope_cos,
        &rope_sin,
    )?)
}

fn prefill_rope(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    prefill_rope_at(0, tokens)
}

fn prefill_rope_at(first_position: usize, tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; tokens * ROTARY_PAIRS];
    let mut sine = vec![0.0; tokens * ROTARY_PAIRS];
    for token in 0..tokens {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let angle = (first_position + token) as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            cosine[token * ROTARY_PAIRS + pair] = cos as f32;
            sine[token * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn read_prefill_cache_pages(
    program: &ResidentModelProgram,
    stream: &CudaStream,
    slot: usize,
    tokens: usize,
) -> Result<CachePages, ResidentModelQualificationError> {
    (0..tokens.div_ceil(ATTENTION_PAGE_SIZE))
        .map(|logical_page| {
            let position = logical_page * ATTENTION_PAGE_SIZE;
            let physical = usize::try_from(program.qualification_kv_physical_page(slot, position)?)
                .map_err(|_| {
                    ResidentModelQualificationError::Mismatch(
                        "resident prefill physical page exceeds host width".to_string(),
                    )
                })?;
            Ok(program.qualification_cache_page(stream, physical)?)
        })
        .collect()
}

fn read_prefill_tail_cache_pages(
    program: &ResidentModelProgram,
    stream: &CudaStream,
    slot: usize,
    first_position: usize,
    tokens: usize,
) -> Result<CachePages, ResidentModelQualificationError> {
    let first_page = first_position / ATTENTION_PAGE_SIZE;
    let final_page = (first_position + tokens).div_ceil(ATTENTION_PAGE_SIZE);
    (first_page..final_page)
        .map(|logical_page| {
            let physical = usize::try_from(
                program.qualification_kv_physical_page(slot, logical_page * ATTENTION_PAGE_SIZE)?,
            )
            .map_err(|_| {
                ResidentModelQualificationError::Mismatch(
                    "resident prefill tail physical page exceeds host width".to_string(),
                )
            })?;
            Ok(program.qualification_cache_page(stream, physical)?)
        })
        .collect()
}

fn verify_dynamic_page_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
) -> Result<usize, ResidentModelQualificationError> {
    if program.recycle_kv_slot(stream, 0)? != TABLE_STRIDE
        || program.qualification_kv_slot_state(0)? != PagedKvSlotState::Vacant
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "slot 0 recycling did not release its exact initial page inventory".to_string(),
        ));
    }

    let extension = program.reserve_kv_slot_tokens(stream, 1, 2 * 192)?;
    if (
        extension.slot(),
        extension.first_entry(),
        extension.entry_count(),
    ) != (1, TABLE_STRIDE, TABLE_STRIDE)
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "slot 1 did not reuse the three released physical pages as its logical tail"
                .to_string(),
        ));
    }
    for (position, expected) in [(192usize, 0u32), (256, 1), (320, 2)] {
        if program.qualification_kv_physical_page(1, position)? != expected {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "slot 1 position {position} did not route to reassigned page {expected}"
            )));
        }
    }
    let (reassigned_key, reassigned_value) = program.qualification_cache_page(stream, 0)?;
    if reassigned_key.iter().any(|&value| value != 0)
        || reassigned_value.iter().any(|&value| value != 0)
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "a reassigned physical page retained values from its prior slot owner".to_string(),
        ));
    }

    program.activate_kv_slot(0)?;
    let replacement = program.reserve_kv_slot_tokens(stream, 0, 192)?;
    if (
        replacement.slot(),
        replacement.first_entry(),
        replacement.entry_count(),
    ) != (0, 0, TABLE_STRIDE)
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "slot 0 replacement route has the wrong logical inventory".to_string(),
        ));
    }
    for (position, expected) in [(0usize, 24u32), (64, 25), (128, 26)] {
        if program.qualification_kv_physical_page(0, position)? != expected {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "slot 0 position {position} did not route to replacement page {expected}"
            )));
        }
    }

    let tables = program.qualification_block_tables(stream)?;
    let row0 = &tables[..3_438];
    let row1 = &tables[3_438..2 * 3_438];
    if row0[..TABLE_STRIDE] != [24, 25, 26]
        || row0[TABLE_STRIDE..].iter().any(|&page| page != u32::MAX)
        || row1[..2 * TABLE_STRIDE] != [3, 4, 5, 0, 1, 2]
        || row1[2 * TABLE_STRIDE..]
            .iter()
            .any(|&page| page != u32::MAX)
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "device block-table rows disagree with the independent remapping oracle".to_string(),
        ));
    }

    let eager_route = prepare_run(program, stream, 1, &[0])?;
    program.launch_eager(stream, eager_route)?;
    let eager = program.qualification_cache_page(stream, 24)?;
    if eager.0.iter().all(|&value| value == 0) || eager.1.iter().all(|&value| value == 0) {
        return Err(ResidentModelQualificationError::Mismatch(
            "eager replay did not append K/V values through the remapped slot row".to_string(),
        ));
    }
    let replay_route = prepare_run(program, stream, 1, &[0])?;
    program.replay(stream, replay_route)?;
    let replay = program.qualification_cache_page(stream, 24)?;
    if replay != eager {
        return Err(ResidentModelQualificationError::Mismatch(
            "graph replay differs from eager cache append through a remapped slot row".to_string(),
        ));
    }

    Ok(tables.len()
        + reassigned_key.len()
        + reassigned_value.len()
        + eager.0.len()
        + eager.1.len()
        + replay.0.len()
        + replay.1.len())
}

fn verify_long_context_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    vacate_all_slots(program, stream)?;
    program.activate_kv_slot(7)?;
    program.reserve_kv_slot_tokens(stream, 7, 220_000)?;
    verify_long_context_case(program, stream, &[7], &[219_999], report)?;

    vacate_all_slots(program, stream)?;
    for slot in 0..MAX_BATCH {
        program.activate_kv_slot(slot)?;
    }
    program.reserve_kv_slot_tokens(stream, 7, LONG_ROUTE_LENGTHS[5])?;
    for slot in 0..MAX_BATCH - 1 {
        program.reserve_kv_slot_tokens(stream, slot, 1)?;
    }
    for maximum_length in LONG_ROUTE_LENGTHS {
        for batch in 1..=MAX_BATCH {
            let mut positions = [0u32; MAX_BATCH];
            positions[0] = u32::try_from(maximum_length - 1).map_err(|_| {
                ResidentModelQualificationError::Mismatch(
                    "long-context qualification position exceeds u32".to_string(),
                )
            })?;
            verify_long_context_case(
                program,
                stream,
                route_slots(batch),
                &positions[..batch],
                report,
            )?;
        }
    }
    Ok(())
}

fn vacate_all_slots(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(), ResidentModelQualificationError> {
    for slot in 0..MAX_BATCH {
        if program.qualification_kv_slot_state(slot)? != PagedKvSlotState::Vacant {
            program.recycle_kv_slot(stream, slot)?;
        }
    }
    Ok(())
}

fn verify_long_context_case(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    slots: &[usize],
    positions: &[u32],
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let eager_route = prepare_long_context_run(program, stream, slots, positions)?;
    program.launch_eager(stream, eager_route)?;
    let eager = program.qualification_long_context_observables(stream, eager_route)?;
    let eager_pages = read_active_cache_pages(program, stream, slots, positions)?;

    let replay_route = prepare_long_context_run(program, stream, slots, positions)?;
    if replay_route != eager_route {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "long-context eager route {eager_route:?} differs from replay route {replay_route:?}"
        )));
    }
    program.replay(stream, replay_route)?;
    let replay = program.qualification_long_context_observables(stream, replay_route)?;
    let replay_pages = read_active_cache_pages(program, stream, slots, positions)?;

    compare_long_context_replay(replay_route, &eager, &replay)?;
    if eager_pages != replay_pages {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "B={} long-context cache append differs under graph replay",
            replay_route.batch()
        )));
    }
    verify_long_context_oracle(
        replay_route,
        positions,
        &replay_pages,
        program
            .qualification_cache_scales()
            .last()
            .copied()
            .ok_or_else(|| {
                ResidentModelQualificationError::Mismatch(
                    "resident model has no attention cache scale".to_string(),
                )
            })?,
        &replay,
        report,
    )?;
    report.graph_replay_values += long_context_observable_values(&replay)
        + replay_pages
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>();
    report.long_route_cases += 1;
    Ok(())
}

fn prepare_long_context_run(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    slots: &[usize],
    positions: &[u32],
) -> Result<ResidentDecodeRoute, ResidentModelQualificationError> {
    if slots.len() != positions.len() {
        return Err(ResidentModelQualificationError::Mismatch(
            "long-context slot and position inventories differ".to_string(),
        ));
    }
    let batch = slots.len();
    program.reset_state(stream)?;
    program.qualification_reset_workspace(stream, SENTINEL)?;
    let token_ids = (0..batch)
        .map(|row| 700u32 + row as u32 * 29)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    program.load_slot_routes(stream, slots)?;
    Ok(program.load_decode_state(
        stream,
        batch,
        positions,
        &[1.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
        &[0.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
    )?)
}

type CachePages = Vec<(Vec<u8>, Vec<u8>)>;

fn read_active_cache_pages(
    program: &ResidentModelProgram,
    stream: &CudaStream,
    slots: &[usize],
    positions: &[u32],
) -> Result<CachePages, ResidentModelQualificationError> {
    slots
        .iter()
        .zip(positions)
        .map(|(&slot, &position)| {
            let physical =
                usize::try_from(program.qualification_kv_physical_page(slot, position as usize)?)
                    .map_err(|_| {
                    ResidentModelQualificationError::Mismatch(
                        "long-context physical page exceeds host width".to_string(),
                    )
                })?;
            Ok(program.qualification_cache_page(stream, physical)?)
        })
        .collect()
}

fn compare_long_context_replay(
    route: ResidentDecodeRoute,
    eager: &ResidentLongContextObservables,
    replay: &ResidentLongContextObservables,
) -> Result<(), ResidentModelQualificationError> {
    compare_exact(
        &format!("B={} long-context graph `projected`", route.batch()),
        &replay.projected,
        &eager.projected,
    )?;
    for (role, actual, expected) in [
        ("query", &replay.query, &eager.query),
        (
            "partial_maximum",
            &replay.partial_maximum,
            &eager.partial_maximum,
        ),
        (
            "partial_denominator",
            &replay.partial_denominator,
            &eager.partial_denominator,
        ),
        (
            "partial_numerator",
            &replay.partial_numerator,
            &eager.partial_numerator,
        ),
        ("attention", &replay.attention, &eager.attention),
    ] {
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "B={} long-context graph `{role}` differs at value {index}",
                route.batch()
            )));
        }
    }
    for (role, actual, expected) in [
        ("mixer_branch", &replay.mixer_branch, &eager.mixer_branch),
        ("residual_a", &replay.residual_a, &eager.residual_a),
        ("logits", &replay.logits, &eager.logits),
    ] {
        compare_exact(
            &format!("B={} long-context graph `{role}`", route.batch()),
            actual,
            expected,
        )?;
    }
    Ok(())
}

fn verify_long_context_oracle(
    route: ResidentDecodeRoute,
    positions: &[u32],
    pages: &CachePages,
    cache_scales: [f32; 2],
    observed: &ResidentLongContextObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let page_values = Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM;
    let attention_layers = Qwen38_27B::LAYERS / Qwen38_27B::FULL_ATTENTION_INTERVAL;
    let layer_offset = (attention_layers - 1) * page_values;
    let dimensions = Qwen38_27B::HEAD_DIM;
    let heads = Qwen38_27B::NUM_ATTENTION_HEADS;
    let queries_per_kv = heads / Qwen38_27B::NUM_KV_HEADS;
    let launched_partitions = route.partition_capacity().ok_or_else(|| {
        ResidentModelQualificationError::Mismatch(
            "long-context oracle received a short graph route".to_string(),
        )
    })?;

    for token in 0..route.batch() {
        let length = positions[token] as usize + 1;
        let active_partitions = length.div_ceil(LONG_CONTEXT_GQA_PARTITION_SIZE);
        if active_partitions > launched_partitions {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "B={} needs {active_partitions} partials but graph owns {launched_partitions}",
                route.batch()
            )));
        }
        let page_position = (length - 1) & (ATTENTION_PAGE_SIZE - 1);
        let key = &pages[token].0[layer_offset..layer_offset + page_values];
        let value = &pages[token].1[layer_offset..layer_offset + page_values];
        for query_head in 0..heads {
            let head_token = token * heads + query_head;
            let query_base = head_token * dimensions;
            let kv_head = query_head / queries_per_kv;
            let cache_base = (kv_head * ATTENTION_PAGE_SIZE + page_position) * dimensions;
            let mut score = 0.0f64;
            for dimension in 0..dimensions {
                score += f64::from(observed.query[query_base + dimension])
                    * f64::from(
                        decode_e4m3fn(key[cache_base + dimension])
                            .map_err(ResidentModelQualificationError::Mismatch)?,
                    )
                    * f64::from(cache_scales[0]);
            }
            score *= 0.0625;

            let partial_base = head_token * LONG_CONTEXT_GQA_MAX_PARTITIONS;
            for partition in 0..active_partitions {
                let first = partition * LONG_CONTEXT_GQA_PARTITION_SIZE;
                let end = (first + LONG_CONTEXT_GQA_PARTITION_SIZE).min(length);
                let values = end - first;
                let contains_current = (first..end).contains(&(length - 1));
                let (maximum, denominator, current_weight) = if contains_current {
                    let maximum = score.max(0.0);
                    let current_weight = (score - maximum).exp();
                    let denominator = (values - 1) as f64 * (-maximum).exp() + current_weight;
                    (maximum, denominator, current_weight)
                } else {
                    (0.0, values as f64, 0.0)
                };
                let partial = partial_base + partition;
                require_long_close(
                    "partial maximum",
                    partial,
                    observed.partial_maximum[partial],
                    maximum,
                    0.004,
                    0.004,
                )?;
                require_long_close(
                    "partial denominator",
                    partial,
                    observed.partial_denominator[partial],
                    denominator,
                    0.02,
                    0.008,
                )?;
                report.long_oracle_values += 2;
                for dimension in LONG_ORACLE_DIMENSIONS {
                    let expected = if contains_current {
                        f64::from(
                            decode_e4m3fn(value[cache_base + dimension])
                                .map_err(ResidentModelQualificationError::Mismatch)?,
                        ) * f64::from(cache_scales[1])
                            * current_weight
                    } else {
                        0.0
                    };
                    let index = partial * dimensions + dimension;
                    require_long_close(
                        "partial numerator",
                        index,
                        observed.partial_numerator[index],
                        expected,
                        0.02,
                        0.008,
                    )?;
                    report.long_oracle_values += 1;
                }
            }
            for partition in active_partitions..LONG_CONTEXT_GQA_MAX_PARTITIONS {
                let partial = partial_base + partition;
                if observed.partial_maximum[partial].to_bits() != F32_SENTINEL_BITS
                    || observed.partial_denominator[partial].to_bits() != F32_SENTINEL_BITS
                {
                    return Err(ResidentModelQualificationError::Mismatch(format!(
                        "B={} inactive long partial {partial} was modified",
                        route.batch()
                    )));
                }
                for dimension in LONG_ORACLE_DIMENSIONS {
                    let index = partial * dimensions + dimension;
                    if observed.partial_numerator[index].to_bits() != F32_SENTINEL_BITS {
                        return Err(ResidentModelQualificationError::Mismatch(format!(
                            "B={} inactive long numerator {index} was modified",
                            route.batch()
                        )));
                    }
                }
            }
            let maximum = score.max(0.0);
            let current_weight = (score - maximum).exp();
            let denominator = (length - 1) as f64 * (-maximum).exp() + current_weight;
            for dimension in LONG_ORACLE_DIMENSIONS {
                let current_value = f64::from(
                    decode_e4m3fn(value[cache_base + dimension])
                        .map_err(ResidentModelQualificationError::Mismatch)?,
                ) * f64::from(cache_scales[1]);
                let gate_index = token * Qwen38_27B::ATTENTION_QKV_ROWS
                    + query_head * 2 * dimensions
                    + dimensions
                    + dimension;
                let gate = f64::from(bf16_to_f32(observed.projected[gate_index]));
                let reduced = current_value * current_weight / denominator;
                let expected = reduced / (1.0 + (-gate).exp());
                require_long_close(
                    "reduced and gated attention",
                    query_base + dimension,
                    observed.attention[query_base + dimension],
                    expected,
                    0.004,
                    0.006,
                )?;
                report.long_oracle_values += 1;
            }
        }
    }
    Ok(())
}

fn require_long_close(
    role: &str,
    index: usize,
    actual: f32,
    expected: f64,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<(), ResidentModelQualificationError> {
    let error = (f64::from(actual) - expected).abs() as f32;
    let tolerance = absolute_tolerance.max(expected.abs() as f32 * relative_tolerance);
    if !actual.is_finite() || error > tolerance {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "long-context {role} at value {index}: device={actual}, oracle={expected}, tolerance={tolerance}"
        )));
    }
    Ok(())
}

fn long_context_observable_values(observed: &ResidentLongContextObservables) -> usize {
    observed.projected.len()
        + observed.query.len()
        + observed.partial_maximum.len()
        + observed.partial_denominator.len()
        + observed.partial_numerator.len()
        + observed.attention.len()
        + observed.mixer_branch.len()
        + observed.residual_a.len()
        + observed.logits.len()
}

fn initialize_short_routes(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(), ResidentModelQualificationError> {
    for slot in 0..MAX_BATCH {
        program.activate_kv_slot(slot)?;
        program.reserve_kv_slot_tokens(stream, slot, 192)?;
    }
    Ok(())
}

fn verify_block_tables(
    program: &ResidentModelProgram,
    stream: &CudaStream,
) -> Result<usize, ResidentModelQualificationError> {
    let tables = program.qualification_block_tables(stream)?;
    let expected = MAX_BATCH * 3_438;
    if tables.len() != expected {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "resident block tables contain {} entries, expected {expected}",
            tables.len()
        )));
    }
    for slot in 0..MAX_BATCH {
        let row = &tables[slot * 3_438..(slot + 1) * 3_438];
        for (logical_page, &physical_page) in row.iter().enumerate() {
            let expected = if logical_page < TABLE_STRIDE {
                (slot * TABLE_STRIDE + logical_page) as u32
            } else {
                u32::MAX
            };
            if physical_page != expected {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "resident slot {slot} table entry {logical_page} is {physical_page}, expected {expected}"
                )));
            }
        }
    }
    Ok(tables.len())
}

fn verify_owner(program: &ResidentModelProgram) -> Result<(), ResidentModelQualificationError> {
    if program.resident_weight_bytes() != 19_103_682_560
        || program.history_bytes() != 23_592_960
        || program.state_bytes() != 1_207_959_552
        || program.cache_bytes() != 7_210_008_576
        || program.kv_table_bytes() != 110_016
        || program.workspace_bytes() != 923_695_108
        || program.descriptor_bytes() != 4_096
        || program.padding_bytes() != 15_676
        || program.resident_arena_bytes() != 21_258_945_792
        || program.kv_arena_bytes() != 7_210_118_656
        || program.arena_bytes() != 28_469_064_448
        || program.host_stager_bytes() != 10_485_760
        || program.kv_route_host_bytes() != 113_454
        || program.batch_capacity() != 8
        || program.row_capacity() != 1_024
        || program.context_capacity() != 220_000
        || program.persistent_slot_bytes() != 153_944_064
    {
        return Err(ResidentModelQualificationError::Mismatch(
            "owner byte or route accounting differs from the admitted layout".to_string(),
        ));
    }
    let addresses = program.qualification_addresses();
    if addresses.len() != 1_168 || addresses.iter().copied().collect::<BTreeSet<_>>().len() != 1_168
    {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "owner exposes {} addresses, expected 1,168 unique addresses",
            addresses.len()
        )));
    }
    let host_addresses = program.qualification_kv_route_addresses();
    if host_addresses[0] == 0 || host_addresses[1] == 0 || host_addresses[0] == host_addresses[1] {
        return Err(ResidentModelQualificationError::Mismatch(
            "page-router host addresses are null or aliased".to_string(),
        ));
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
) -> Result<ResidentDecodeRoute, ResidentModelQualificationError> {
    program.reset_state(stream)?;
    program.qualification_reset_workspace(stream, SENTINEL)?;
    let token_ids = (0..batch)
        .map(|slot| 100u32 + slot as u32 * 17)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    program.load_slot_routes(stream, slots)?;
    Ok(program.load_decode_state(
        stream,
        batch,
        &[0; MAX_BATCH][..batch],
        &[1.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
        &[0.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
    )?)
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

fn verify_prefill_final_oracle(
    tokens: usize,
    final_norm: &[u16],
    lm_head_codes: &[u8],
    lm_head_scales: &[u16],
    observed: &ResidentModelObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let hidden = Qwen38_27B::HIDDEN;
    let begin = (tokens - 1) * hidden;
    let end = begin + hidden;
    let residual = observed.mixer_residual[begin..end]
        .iter()
        .zip(&observed.mlp_branch[begin..end])
        .map(|(&residual, &branch)| f32_to_bf16(bf16_to_f32(residual) + bf16_to_f32(branch)))
        .collect::<Vec<_>>();
    compare_exact(
        &format!("T={tokens} final residual"),
        &observed.residual_a[begin..end],
        &residual,
    )?;
    let normalized = rms_norm_oracle::<Qwen38_27B>(&residual, final_norm);
    compare_bf16(
        &format!("T={tokens} final RMSNorm"),
        &observed.mixer_normalized[begin..end],
        &normalized,
        &mut report.maximum_absolute_error,
    )?;
    let quantized =
        quantize_oracle(&normalized).map_err(ResidentModelQualificationError::Mismatch)?;
    compare_exact(
        &format!("T={tokens} endpoint activation codes"),
        &observed.activation_codes[..hidden],
        &quantized.codes,
    )?;
    if observed.activation_scales[0].to_bits() != quantized.scale.to_bits() {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "T={tokens} endpoint activation scale differs"
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
        let actual = bf16_to_f32(observed.logits[row]);
        require_close(
            &format!("T={tokens} selected LM-head row"),
            row,
            actual,
            expected,
            &mut report.maximum_absolute_error,
        )?;
    }
    report.oracle_values += 3 * hidden + 1 + SELECTED_LOGIT_ROWS.len();
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
    same_f32!(partial_maximum);
    same_f32!(partial_denominator);
    same_f32!(partial_numerator);
    same_f32!(prefill_partials);
    same_f32!(attention);
    same!(mixer_branch);
    same!(swiglu);
    same!(mlp_branch);
    same!(logits);
    same!(history);
    same_f32!(state);
    same!(key_pages);
    same!(value_pages);
    same!(key_guard_pages);
    same!(value_guard_pages);
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
        + observed.partial_maximum.len()
        + observed.partial_denominator.len()
        + observed.partial_numerator.len()
        + observed.prefill_partials.len()
        + observed.attention.len()
        + observed.mixer_branch.len()
        + observed.swiglu.len()
        + observed.mlp_branch.len()
        + observed.logits.len()
        + observed.history.len()
        + observed.state.len()
        + observed.key_pages.len()
        + observed.value_pages.len()
        + observed.key_guard_pages.len()
        + observed.value_guard_pages.len()
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
    for (role, values) in [
        ("partial_maximum", &observed.partial_maximum),
        ("partial_denominator", &observed.partial_denominator),
        ("partial_numerator", &observed.partial_numerator),
        ("prefill_partials", &observed.prefill_partials),
    ] {
        if values
            .iter()
            .any(|value| value.to_bits() != F32_SENTINEL_BITS)
        {
            return Err(ResidentModelQualificationError::Mismatch(format!(
                "B={batch} short graph modified `{role}` scratch"
            )));
        }
        inactive += values.len();
    }
    inactive += sentinel_f32!(attention, Qwen38_27B::ATTENTION_OUTPUT_COLUMNS);
    inactive += sentinel_u16!(mixer_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(swiglu, Qwen38_27B::INTERMEDIATE);
    inactive += sentinel_u16!(mlp_branch, Qwen38_27B::HIDDEN);
    inactive += sentinel_u16!(logits, Qwen38_27B::VOCAB);
    let persistent = verify_persistent_inactive(batch, slots, observed, true)?;
    inactive += persistent;
    report.slot_control_values += persistent;
    report.inactive_values += inactive;
    Ok(())
}

fn verify_prefill_inactive(
    route: ResidentPrefillRoute,
    observed: &ResidentModelObservables,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let tokens = route.tokens();
    let mut inactive = 0usize;
    for (role, values, width) in [
        ("residual_a", &observed.residual_a, Qwen38_27B::HIDDEN),
        ("residual_b", &observed.residual_b, Qwen38_27B::HIDDEN),
        (
            "mixer_residual",
            &observed.mixer_residual,
            Qwen38_27B::HIDDEN,
        ),
        (
            "mixer_normalized",
            &observed.mixer_normalized,
            Qwen38_27B::HIDDEN,
        ),
        (
            "mlp_normalized",
            &observed.mlp_normalized,
            Qwen38_27B::HIDDEN,
        ),
        ("projected", &observed.projected, Qwen38_27B::GDN_INPUT_ROWS),
        ("convolved", &observed.convolved, Qwen38_27B::GDN_QKV_ROWS),
        (
            "recurrent_output",
            &observed.recurrent_output,
            Qwen38_27B::GDN_VALUE_ROWS,
        ),
        ("mixer_branch", &observed.mixer_branch, Qwen38_27B::HIDDEN),
        ("swiglu", &observed.swiglu, Qwen38_27B::INTERMEDIATE),
        ("mlp_branch", &observed.mlp_branch, Qwen38_27B::HIDDEN),
    ] {
        inactive += require_bf16_sentinel_tail(
            &format!("T={tokens} inactive {role}"),
            values,
            tokens * width,
        )?;
    }
    for (role, values, width) in [
        (
            "activation_codes",
            &observed.activation_codes,
            Qwen38_27B::INTERMEDIATE,
        ),
        (
            "nvfp4_activation_codes",
            &observed.nvfp4_activation_codes,
            Qwen38_27B::INTERMEDIATE / 2,
        ),
        (
            "nvfp4_activation_scales",
            &observed.nvfp4_activation_scales,
            Qwen38_27B::INTERMEDIATE / 16,
        ),
    ] {
        inactive += require_u8_sentinel_tail(
            &format!("T={tokens} inactive {role}"),
            values,
            tokens * width,
        )?;
    }
    for (role, values, width) in [
        ("activation_scales", &observed.activation_scales, 1),
        (
            "log_decay",
            &observed.log_decay,
            Qwen38_27B::GDN_CONTROL_ROWS,
        ),
        ("beta", &observed.beta, Qwen38_27B::GDN_CONTROL_ROWS),
        (
            "query",
            &observed.query,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ),
        (
            "attention",
            &observed.attention,
            Qwen38_27B::ATTENTION_OUTPUT_COLUMNS,
        ),
    ] {
        inactive += require_f32_sentinel_tail(
            &format!("T={tokens} inactive {role}"),
            values,
            tokens * width,
        )?;
    }
    for (role, values) in [
        ("partial_maximum", &observed.partial_maximum),
        ("partial_denominator", &observed.partial_denominator),
        ("partial_numerator", &observed.partial_numerator),
    ] {
        inactive += require_f32_sentinel_tail(&format!("T={tokens} untouched {role}"), values, 0)?;
    }
    let active_prefill_partials = route.partition_capacity().map_or(0, |partitions| {
        tokens * Qwen38_27B::NUM_ATTENTION_HEADS * partitions * (Qwen38_27B::HEAD_DIM + 2)
    });
    inactive += require_f32_sentinel_tail(
        &format!("T={tokens} inactive prefill_partials"),
        &observed.prefill_partials,
        active_prefill_partials,
    )?;
    inactive += require_bf16_sentinel_tail(
        &format!("T={tokens} inactive logits"),
        &observed.logits,
        Qwen38_27B::VOCAB,
    )?;
    let persistent = verify_persistent_inactive(tokens, &[0], observed, false)?;
    report.slot_control_values += persistent;
    report.inactive_values += inactive + persistent;
    Ok(())
}

fn require_u8_sentinel_tail(
    role: &str,
    values: &[u8],
    begin: usize,
) -> Result<usize, ResidentModelQualificationError> {
    let tail = values.get(begin..).ok_or_else(|| {
        ResidentModelQualificationError::Mismatch(format!(
            "{role} begins at {begin}, beyond {} values",
            values.len()
        ))
    })?;
    if let Some(index) = tail.iter().position(|&value| value != SENTINEL) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "{role} changed at value {}",
            begin + index
        )));
    }
    Ok(tail.len())
}

fn require_bf16_sentinel_tail(
    role: &str,
    values: &[u16],
    begin: usize,
) -> Result<usize, ResidentModelQualificationError> {
    let tail = values.get(begin..).ok_or_else(|| {
        ResidentModelQualificationError::Mismatch(format!(
            "{role} begins at {begin}, beyond {} values",
            values.len()
        ))
    })?;
    if let Some(index) = tail.iter().position(|&value| value != BF16_SENTINEL) {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "{role} changed at value {}",
            begin + index
        )));
    }
    Ok(tail.len())
}

fn require_f32_sentinel_tail(
    role: &str,
    values: &[f32],
    begin: usize,
) -> Result<usize, ResidentModelQualificationError> {
    let tail = values.get(begin..).ok_or_else(|| {
        ResidentModelQualificationError::Mismatch(format!(
            "{role} begins at {begin}, beyond {} values",
            values.len()
        ))
    })?;
    if let Some(index) = tail
        .iter()
        .position(|value| value.to_bits() != F32_SENTINEL_BITS)
    {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "{role} changed at value {}",
            begin + index
        )));
    }
    Ok(tail.len())
}

fn verify_persistent_inactive(
    batch: usize,
    active_slots: &[usize],
    observed: &ResidentModelObservables,
    check_guard_pages: bool,
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
    if check_guard_pages {
        for (role, guards) in [
            ("key", &observed.key_guard_pages),
            ("value", &observed.value_guard_pages),
        ] {
            if guards.iter().any(|&value| value != 0) {
                return Err(ResidentModelQualificationError::Mismatch(format!(
                    "B={batch} modified the first unassigned {role} cache page"
                )));
            }
            inactive += guards.len();
        }
    }
    Ok(inactive)
}

fn verify_slot_reset(
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
    report: &mut ResidentModelQualification,
) -> Result<(), ResidentModelQualificationError> {
    let route = prepare_run(program, stream, MAX_BATCH, route_slots(MAX_BATCH))?;
    program.replay(stream, route)?;
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
    program: &mut ResidentModelProgram,
    stream: &CudaStream,
) -> Result<(), ResidentModelQualificationError> {
    let routes: [ResidentDecodeRoute; MAX_BATCH] = (1..=MAX_BATCH)
        .map(|batch| {
            program.load_decode_state(
                stream,
                batch,
                &[0; MAX_BATCH][..batch],
                &[1.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
                &[0.0; MAX_BATCH * ROTARY_PAIRS][..batch * ROTARY_PAIRS],
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| {
            ResidentModelQualificationError::Mismatch(
                "allocation route inventory has wrong cardinality".to_string(),
            )
        })?;
    program.replay(stream, routes[MAX_BATCH - 1])?;
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for batch in [1, 8, 3, 6, 2, 7, 4, 5] {
        program.replay(stream, routes[batch - 1])?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }

    let token_ids = (0..1_024)
        .map(|token| 100u32 + (token % 251) as u32)
        .collect::<Vec<_>>();
    program.stage_embeddings(stream, &token_ids)?;
    let graph_routes = [
        (32, 0),
        (64, 0),
        (128, 0),
        (128, 1),
        (128, 32_768),
        (1_024, 1_024),
    ];
    let reserved = program
        .qualification_kv_page_count(7)?
        .checked_mul(ATTENTION_PAGE_SIZE)
        .ok_or_else(|| {
            ResidentModelQualificationError::Mismatch(
                "prefill allocation warmup reservation overflows".to_string(),
            )
        })?;
    if reserved < 32_896 {
        program.reserve_kv_slot_tokens(stream, 7, 32_896)?;
    }
    for (tokens, first_position) in graph_routes {
        let (rope_cos, rope_sin) = prefill_rope_at(first_position, tokens);
        let route = program.load_prefill_tile_state(
            stream,
            tokens,
            7,
            first_position,
            &rope_cos,
            &rope_sin,
        )?;
        program.replay_prefill(stream, route)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(program.context())?;
    for (tokens, first_position) in graph_routes {
        let (rope_cos, rope_sin) = prefill_rope_at(first_position, tokens);
        let route = program.load_prefill_tile_state(
            stream,
            tokens,
            7,
            first_position,
            &rope_cos,
            &rope_sin,
        )?;
        program.replay_prefill(stream, route)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(program.context())?;
    if before != after {
        return Err(ResidentModelQualificationError::Mismatch(format!(
            "device memory changed after prefill warmup: before={before:?}, after={after:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ResidentModelQualification, SELECTED_LOGIT_ROWS, qualify_resident_model,
        qualify_resident_model_with_mode,
    };
    use std::path::PathBuf;
    use tuisko_engine::ResidentLoadMode;

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn source_model_matches_final_oracle_and_exact_graph_replay()
    -> Result<(), super::ResidentModelQualificationError> {
        let report = qualify_resident_model(&snapshot_root()?)?;
        assert_complete_report(report);
        Ok(())
    }

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn legacy_loader_matches_final_oracle_and_exact_graph_replay()
    -> Result<(), super::ResidentModelQualificationError> {
        let report = qualify_resident_model_with_mode(&snapshot_root()?, ResidentLoadMode::Legacy)?;
        assert_complete_report(report);
        Ok(())
    }

    fn snapshot_root() -> Result<PathBuf, super::ResidentModelQualificationError> {
        std::env::var_os("TUISKO_SNAPSHOT")
            .map(PathBuf::from)
            .ok_or_else(|| {
                super::ResidentModelQualificationError::Mismatch(
                    "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
                )
            })
    }

    fn assert_complete_report(report: ResidentModelQualification) {
        let active = (1..=8).sum::<usize>();
        assert_eq!(report.source_scalars, 256);
        assert_eq!(
            report.oracle_values,
            (active + 9) * (3 * 5_120 + 1 + SELECTED_LOGIT_ROWS.len())
        );
        assert!(report.graph_replay_values > 0);
        assert!(report.inactive_values > 0);
        assert!(report.slot_control_values > 0);
        assert_eq!(report.long_route_cases, 49);
        assert_eq!(report.prefill_route_cases, 9);
        assert!(report.long_oracle_values > 0);
        assert!(report.maximum_absolute_error.is_finite());
    }
}

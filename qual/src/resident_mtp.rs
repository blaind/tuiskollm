//! Source-backed qualification for the resident long-context Qwen3.8 MTP owner.

use crate::DeviceBenchmarkError;
use crate::device_benchmark;
use crate::fp8_projection_oracle::f32_to_bf16;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    EngineError, LONG_CONTEXT_PHYSICAL_PAGES, MAX_BATCH, ResidentMtpObservables, ResidentMtpProgram,
};
use tuisko_gpu::{CudaContext, CudaStream, GpuError, device_memory_info};
use tuisko_kernels_sm120::ATTENTION_PAGE_SIZE;
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38_27B, TextEndpointBindings};

const PROMPT_ROUTES: [usize; 5] = [1, 32, 64, 128, 1_024];
const REALIGN_ROUTES: usize = 4;
const ROTARY_PAIRS: usize = 32;

/// Failure of the source-backed resident MTP owner gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentMtpQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Resident ownership or execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership, launch, or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// The resident owner disagreed with an exact route or lifecycle contract.
    #[error("resident MTP qualification failed: {0}")]
    Mismatch(String),
}

/// Exact source-oracle, route, graph, cache, lifecycle, and owner counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentMtpQualification {
    /// Independent complete MTP-layer and prompt-prime source suites completed first.
    pub source_oracle_suites: usize,
    /// Exact prompt routes checked eagerly and by graph replay.
    pub prompt_routes: usize,
    /// Exact compact draft routes checked eagerly and by graph replay.
    pub draft_routes: usize,
    /// Exact compact residual-continuation routes checked eagerly and by graph replay.
    pub continuation_routes: usize,
    /// Exact compact staged-hidden continuation routes checked eagerly and by graph replay.
    pub staged_continuation_routes: usize,
    /// Exact prime-only realignment routes checked eagerly and by graph replay.
    pub prime_routes: usize,
    /// Exact full realignment routes checked eagerly and by graph replay.
    pub realign_routes: usize,
    /// Repeated scalar prompt tail lengths checked without padding.
    pub tail_routes: usize,
    /// Active seam values reproduced by graph replay.
    pub graph_replay_values: usize,
    /// Appended and untouched cache values checked at page boundaries.
    pub cache_values: usize,
    /// Shared lifecycle transitions exercised.
    pub lifecycle_transitions: usize,
    /// Exact unchanged source-BF16 MTP weights.
    pub resident_weight_bytes: usize,
    /// Exact represented long-context BF16 MTP cache.
    pub cache_bytes: usize,
    /// Exact typed route workspace.
    pub workspace_bytes: usize,
    /// Complete incremental MTP device allocation.
    pub owner_bytes: usize,
    /// Exact alignment padding.
    pub padding_bytes: usize,
    /// Page-locked graph source bytes.
    pub host_stager_bytes: usize,
    /// Exact prompt, seeded draft, same-round/staged continuation, prime, and realignment graphs.
    pub graph_count: usize,
}

struct CachePage {
    physical: usize,
    key: Vec<u16>,
    value: Vec<u16>,
}

/// Qualifies shared lifecycle ownership and every admitted resident MTP graph.
pub fn qualify_resident_mtp(
    root: &Path,
) -> Result<ResidentMtpQualification, ResidentMtpQualificationError> {
    run_source_oracles()?;
    let _preflight = device_benchmark::preflight()?;

    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(ResidentMtpQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let stream = context.new_stream().map_err(GpuError::from)?;
    let mut program = ResidentMtpProgram::from_snapshot(&context, snapshot.clone())?;
    for slot in 0..MAX_BATCH {
        program.activate_kv_slot(slot)?;
        program.reserve_kv_slot_tokens(&stream, slot, 1_024)?;
    }
    verify_owner(&program)?;
    let stable_bases = (program.base_address(), program.cache_base_address());
    let stable_addresses = program.qualification_addresses()?;
    let stable_host_stagers = program.qualification_host_stager_addresses();
    if stable_addresses.len() != 44
        || stable_addresses
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != stable_addresses.len()
    {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "resident MTP exposes {} addresses, expected 44 unique addresses",
            stable_addresses.len()
        )));
    }
    if stable_host_stagers
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != stable_host_stagers.len()
    {
        return Err(ResidentMtpQualificationError::Mismatch(
            "resident MTP pinned graph sources do not have seven unique addresses".to_string(),
        ));
    }

    let embedding = TextEndpointBindings::bind_embedding(snapshot.as_ref())?.bytes();
    let mut report = ResidentMtpQualification {
        source_oracle_suites: 2,
        prompt_routes: 0,
        draft_routes: 0,
        continuation_routes: 0,
        staged_continuation_routes: 0,
        prime_routes: 0,
        realign_routes: 0,
        tail_routes: 0,
        graph_replay_values: 0,
        cache_values: 0,
        lifecycle_transitions: 0,
        resident_weight_bytes: program.resident_weight_bytes(),
        cache_bytes: program.cache_bytes(),
        workspace_bytes: program.workspace_bytes(),
        owner_bytes: program.owner_bytes(),
        padding_bytes: program.padding_bytes(),
        host_stager_bytes: program.host_stager_bytes(),
        graph_count: program.graph_count(),
    };

    verify_prompt_routes(
        &mut program,
        &stream,
        embedding,
        stable_bases,
        &stable_addresses,
        &stable_host_stagers,
        &mut report,
    )?;
    verify_draft_routes(
        &mut program,
        &stream,
        embedding,
        stable_bases,
        &stable_addresses,
        &stable_host_stagers,
        &mut report,
    )?;
    verify_continuation_route(
        &mut program,
        &stream,
        embedding,
        stable_bases,
        &stable_addresses,
        &stable_host_stagers,
        &mut report,
    )?;
    verify_realign_routes(
        &mut program,
        &stream,
        embedding,
        stable_bases,
        &stable_addresses,
        &stable_host_stagers,
        &mut report,
    )?;
    verify_scalar_tails(&mut program, &stream, &mut report)?;
    verify_lifecycle(&mut program, &stream, &mut report)?;
    verify_no_post_warmup_allocation(&context, &mut program, &stream)?;
    device_benchmark::require_current_process_exclusive()?;
    Ok(report)
}

fn run_source_oracles() -> Result<(), ResidentMtpQualificationError> {
    const TESTS: [(&str, &str); 2] = [
        (
            "complete independent MTP layer oracle",
            "mtp_layer::tests::mtp_layer_suite_source_owner_matches_all_draft_prime_and_realign_routes",
        ),
        (
            "independent MTP prompt-prime oracle",
            "mtp_prompt_prime::tests::mtp_prompt_prime_suite_source_values_match_every_route_and_tail",
        ),
    ];
    let executable = std::env::current_exe().map_err(|error| {
        ResidentMtpQualificationError::Mismatch(format!(
            "locating the qualification executable failed: {error}"
        ))
    })?;
    for (role, test) in TESTS {
        let listing = std::process::Command::new(&executable)
            .args([test, "--exact", "--list"])
            .output()
            .map_err(|error| {
                ResidentMtpQualificationError::Mismatch(format!(
                    "listing the {role} target test failed: {error}"
                ))
            })?;
        if !listing.status.success()
            || !String::from_utf8_lossy(&listing.stdout).contains(&format!("{test}: test"))
        {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "the {role} target test `{test}` is missing from this executable"
            )));
        }
        let status = std::process::Command::new(&executable)
            .args([
                test,
                "--exact",
                "--include-ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .status()
            .map_err(|error| {
                ResidentMtpQualificationError::Mismatch(format!(
                    "launching the {role} in an isolated process failed: {error}"
                ))
            })?;
        if !status.success() {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "{role} failed in its isolated process with {status}"
            )));
        }
        device_benchmark::wait_for_owned_child_cleanup()?;
    }
    Ok(())
}

fn verify_owner(program: &ResidentMtpProgram) -> Result<(), ResidentMtpQualificationError> {
    if program.resident_weight_bytes() != 849_398_784
        || program.cache_bytes() != 901_251_072
        || program.workspace_bytes() != 122_904_032
        || program.owner_bytes() != 1_873_554_176
        || program.padding_bytes() != 288
        || program.host_stager_bytes() != 10_842_112
        || program.graph_count() != 37
    {
        return Err(ResidentMtpQualificationError::Mismatch(
            "resident MTP byte or graph accounting differs from the admitted layout".to_string(),
        ));
    }
    Ok(())
}

fn verify_prompt_routes(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    embedding: &[u8],
    stable_bases: (u64, u64),
    stable_addresses: &[usize],
    stable_host_stagers: &[usize; 7],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    for rows in PROMPT_ROUTES {
        let hidden = hidden_fixture(rows, rows);
        let tokens = token_ids(rows, rows);
        let positions = positions(0, rows)?;
        let (cosine, sine) = rope(&positions);
        program.reset_slot(stream, 0)?;
        program.target().load_residual(stream, rows, &hidden)?;
        let route = program.stage_prompt(stream, rows, 0, 0, &tokens, &cosine, &sine)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.qualification_launch_eager_prompt(stream, route)?;
        let eager = program.qualification_observables(stream, rows, false)?;
        let eager_cache = read_cache(program, stream, 0, 0, rows)?;
        verify_inputs(
            program, stream, embedding, &tokens, &hidden, &[0; 0], &positions, &cosine, &sine,
            &eager,
        )?;
        verify_cache(rows, 0, &positions, &eager, &eager_cache, report)?;

        program.reset_slot(stream, 0)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.replay_prompt(stream, route)?;
        let replay = program.qualification_observables(stream, rows, false)?;
        let replay_cache = read_cache(program, stream, 0, 0, rows)?;
        compare_observables("prompt", rows, &eager, &replay, report)?;
        compare_cache("prompt", rows, &eager_cache, &replay_cache)?;
        verify_stable(
            program,
            stable_bases,
            stable_addresses,
            stable_host_stagers,
            "prompt",
            rows,
        )?;
        report.prompt_routes += 1;
    }
    Ok(())
}

fn verify_draft_routes(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    embedding: &[u8],
    stable_bases: (u64, u64),
    stable_addresses: &[usize],
    stable_host_stagers: &[usize; 7],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    for batch in 1..=MAX_BATCH {
        let slots = (0..batch).collect::<Vec<_>>();
        let positions = (0..batch).map(|row| 130 + row as u32).collect::<Vec<_>>();
        let tokens = token_ids(2_000 + batch, batch);
        let hidden = hidden_fixture(batch, 2_000 + batch);
        let (cosine, sine) = rope(&positions);
        for &slot in &slots {
            program.reset_slot(stream, slot)?;
        }
        program.target().load_residual(stream, batch, &hidden)?;
        let route = program.stage_draft(stream, &slots, &positions, &tokens, &cosine, &sine)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.qualification_launch_eager_draft(stream, route)?;
        let eager = program.qualification_observables(stream, batch, true)?;
        let eager_cache = read_draft_cache(program, stream, &slots, &positions)?;
        verify_inputs(
            program, stream, embedding, &tokens, &hidden, &slots, &positions, &cosine, &sine,
            &eager,
        )?;
        verify_draft_cache(batch, &slots, &positions, &eager, &eager_cache, report)?;

        for &slot in &slots {
            program.reset_slot(stream, slot)?;
        }
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.replay_draft(stream, route)?;
        let replay = program.qualification_observables(stream, batch, true)?;
        let replay_cache = read_draft_cache(program, stream, &slots, &positions)?;
        compare_observables("draft", batch, &eager, &replay, report)?;
        compare_cache("draft", batch, &eager_cache, &replay_cache)?;
        verify_stable(
            program,
            stable_bases,
            stable_addresses,
            stable_host_stagers,
            "draft",
            batch,
        )?;
        report.draft_routes += 1;
    }
    Ok(())
}

fn verify_continuation_route(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    embedding: &[u8],
    stable_bases: (u64, u64),
    stable_addresses: &[usize],
    stable_host_stagers: &[usize; 7],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    const SEED_POSITION: u32 = 130;
    const CONTINUATION_POSITION: u32 = SEED_POSITION + 1;
    for batch in 1..=MAX_BATCH {
        let slots = (0..batch).collect::<Vec<_>>();
        let seed_positions = vec![SEED_POSITION; batch];
        let continuation_positions = vec![CONTINUATION_POSITION; batch];
        let seed_tokens = token_ids(8_100 + 2 * batch, batch);
        let continuation_tokens = token_ids(8_200 + 2 * batch, batch);
        let seed_hidden = hidden_fixture(batch, 8_100 + batch);
        let (seed_cosine, seed_sine) = rope(&seed_positions);
        let (continuation_cosine, continuation_sine) = rope(&continuation_positions);

        let run_seed = |program: &mut ResidentMtpProgram| {
            for &slot in &slots {
                program.reset_slot(stream, slot)?;
            }
            program
                .target()
                .load_residual(stream, batch, &seed_hidden)?;
            let route = program.stage_draft(
                stream,
                &slots,
                &seed_positions,
                &seed_tokens,
                &seed_cosine,
                &seed_sine,
            )?;
            program.replay_draft(stream, route)?;
            program.qualification_observables(stream, batch, true)
        };

        let seed = run_seed(program)?;
        let prior_residual = seed.residual_output.ok_or_else(|| {
            ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP B={batch} seed route did not publish its residual"
            ))
        })?;
        let route = program.stage_draft(
            stream,
            &slots,
            &continuation_positions,
            &continuation_tokens,
            &continuation_cosine,
            &continuation_sine,
        )?;
        program.qualification_launch_eager_continue_draft(stream, route)?;
        let eager = program.qualification_observables(stream, batch, true)?;
        let eager_cache = read_draft_cache(program, stream, &slots, &continuation_positions)?;
        verify_inputs(
            program,
            stream,
            embedding,
            &continuation_tokens,
            &prior_residual,
            &slots,
            &continuation_positions,
            &continuation_cosine,
            &continuation_sine,
            &eager,
        )?;

        let replay_seed = run_seed(program)?;
        if replay_seed.residual_output.as_deref() != Some(prior_residual.as_slice()) {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP B={batch} seed residual changed before graph continuation"
            )));
        }
        let route = program.stage_draft(
            stream,
            &slots,
            &continuation_positions,
            &continuation_tokens,
            &continuation_cosine,
            &continuation_sine,
        )?;
        program.replay_continue_draft(stream, route)?;
        let replay = program.qualification_observables(stream, batch, true)?;
        let replay_cache = read_draft_cache(program, stream, &slots, &continuation_positions)?;
        compare_observables("continuation", batch, &eager, &replay, report)?;
        compare_cache("continuation", batch, &eager_cache, &replay_cache)?;

        let staged_seed = run_seed(program)?;
        if staged_seed.residual_output.as_deref() != Some(prior_residual.as_slice()) {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP B={batch} seed residual changed before staged eager continuation"
            )));
        }
        let staged_route = program.stage_continuation_draft(
            stream,
            &slots,
            &continuation_positions,
            &continuation_tokens,
            &prior_residual,
            &continuation_cosine,
            &continuation_sine,
        )?;
        program.qualification_launch_eager_staged_continue_draft(stream, staged_route)?;
        let staged_eager = program.qualification_observables(stream, batch, true)?;
        let staged_eager_cache =
            read_draft_cache(program, stream, &slots, &continuation_positions)?;
        if staged_eager != eager {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP B={batch} staged and prior-residual eager continuations differ"
            )));
        }
        compare_cache(
            "staged continuation eager",
            batch,
            &eager_cache,
            &staged_eager_cache,
        )?;
        verify_inputs(
            program,
            stream,
            embedding,
            &continuation_tokens,
            &prior_residual,
            &slots,
            &continuation_positions,
            &continuation_cosine,
            &continuation_sine,
            &staged_eager,
        )?;

        let staged_replay_seed = run_seed(program)?;
        if staged_replay_seed.residual_output.as_deref() != Some(prior_residual.as_slice()) {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP B={batch} seed residual changed before staged graph continuation"
            )));
        }
        let staged_route = program.stage_continuation_draft(
            stream,
            &slots,
            &continuation_positions,
            &continuation_tokens,
            &prior_residual,
            &continuation_cosine,
            &continuation_sine,
        )?;
        program.replay_staged_continue_draft(stream, staged_route)?;
        let staged_replay = program.qualification_observables(stream, batch, true)?;
        let staged_replay_cache =
            read_draft_cache(program, stream, &slots, &continuation_positions)?;
        compare_observables(
            "staged continuation graph",
            batch,
            &staged_eager,
            &staged_replay,
            report,
        )?;
        compare_cache(
            "staged continuation graph",
            batch,
            &staged_eager_cache,
            &staged_replay_cache,
        )?;
        verify_stable(
            program,
            stable_bases,
            stable_addresses,
            stable_host_stagers,
            "continuation",
            batch,
        )?;
        report.continuation_routes += 1;
        report.staged_continuation_routes += 1;
    }
    Ok(())
}

fn verify_realign_routes(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    embedding: &[u8],
    stable_bases: (u64, u64),
    stable_addresses: &[usize],
    stable_host_stagers: &[usize; 7],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    for tokens_count in 1..=REALIGN_ROUTES {
        let first = 300 + 8 * tokens_count;
        let positions = positions(first, tokens_count)?;
        let tokens = token_ids(4_000 + tokens_count, tokens_count);
        let hidden = hidden_fixture(tokens_count, 4_000 + tokens_count);
        let (cosine, sine) = rope(&positions);
        program.reset_slot(stream, 0)?;
        program
            .target()
            .load_residual(stream, tokens_count, &hidden)?;
        let route =
            program.stage_realign(stream, tokens_count, 0, first, &tokens, &cosine, &sine)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.qualification_launch_eager_prime(stream, route)?;
        let eager = program.qualification_observables(stream, tokens_count, false)?;
        let eager_cache = read_cache(program, stream, 0, first, tokens_count)?;
        verify_cache(tokens_count, 0, &positions, &eager, &eager_cache, report)?;
        program.reset_slot(stream, 0)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.replay_prime(stream, route)?;
        let replay = program.qualification_observables(stream, tokens_count, false)?;
        let replay_cache = read_cache(program, stream, 0, first, tokens_count)?;
        compare_observables("prime", tokens_count, &eager, &replay, report)?;
        compare_cache("prime", tokens_count, &eager_cache, &replay_cache)?;
        report.prime_routes += 1;

        program.reset_slot(stream, 0)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.qualification_launch_eager_realign(stream, route)?;
        let eager = program.qualification_observables(stream, tokens_count, true)?;
        let eager_cache = read_cache(program, stream, 0, first, tokens_count)?;
        verify_inputs(
            program,
            stream,
            embedding,
            &tokens,
            &hidden,
            &vec![0; tokens_count],
            &positions,
            &cosine,
            &sine,
            &eager,
        )?;
        program.reset_slot(stream, 0)?;
        program.qualification_reset_outputs(stream, 0xa5)?;
        program.replay_realign(stream, route)?;
        let replay = program.qualification_observables(stream, tokens_count, true)?;
        let replay_cache = read_cache(program, stream, 0, first, tokens_count)?;
        let mut selected_logits = vec![0u16; Qwen38_27B::VOCAB];
        program.read_logit_row_into(stream, tokens_count - 1, &mut selected_logits)?;
        let mut selected_residual = vec![0u16; Qwen38_27B::HIDDEN];
        program.read_residual_row_into(stream, tokens_count - 1, &mut selected_residual)?;
        let logits = replay.logits.as_ref().ok_or_else(|| {
            ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP K={tokens_count} realignment published no logits"
            ))
        })?;
        let begin = (tokens_count - 1) * Qwen38_27B::VOCAB;
        if selected_logits != logits[begin..begin + Qwen38_27B::VOCAB] {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP K={tokens_count} selected the wrong realignment logit row"
            )));
        }
        let residuals = replay.residual_output.as_ref().ok_or_else(|| {
            ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP K={tokens_count} realignment published no final residual"
            ))
        })?;
        let residual_begin = (tokens_count - 1) * Qwen38_27B::HIDDEN;
        if selected_residual != residuals[residual_begin..residual_begin + Qwen38_27B::HIDDEN] {
            return Err(ResidentMtpQualificationError::Mismatch(format!(
                "resident MTP K={tokens_count} selected the wrong realignment residual row"
            )));
        }
        compare_observables("realign", tokens_count, &eager, &replay, report)?;
        compare_cache("realign", tokens_count, &eager_cache, &replay_cache)?;
        verify_stable(
            program,
            stable_bases,
            stable_addresses,
            stable_host_stagers,
            "realign",
            tokens_count,
        )?;
        report.realign_routes += 1;
    }
    Ok(())
}

fn verify_scalar_tails(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    for tail in 1..=31 {
        let mut eager_pages = Vec::new();
        for graph in [false, true] {
            program.reset_slot(stream, 0)?;
            for position in 0..tail {
                let positions = [position as u32];
                let (cosine, sine) = rope(&positions);
                program
                    .target()
                    .load_residual(stream, 1, &hidden_fixture(1, tail + position))?;
                let route = program.stage_prompt(
                    stream,
                    1,
                    0,
                    position,
                    &[token_id(tail + position)],
                    &cosine,
                    &sine,
                )?;
                if graph {
                    program.replay_prompt(stream, route)?;
                } else {
                    program.qualification_launch_eager_prompt(stream, route)?;
                }
            }
            let pages = read_cache(program, stream, 0, 0, tail)?;
            if graph {
                compare_cache("tail", tail, &eager_pages, &pages)?;
            } else {
                verify_tail_written(program, tail, &pages)?;
                eager_pages = pages;
            }
        }
        report.cache_values += eager_pages
            .iter()
            .map(|page| page.key.len() + page.value.len())
            .sum::<usize>();
        report.tail_routes += 1;
    }
    Ok(())
}

fn verify_lifecycle(
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    let slot = MAX_BATCH - 1;
    program.reset_slot(stream, slot)?;
    let physical = usize::try_from(program.target().qualification_kv_physical_page(slot, 0)?)
        .map_err(|_| {
            ResidentMtpQualificationError::Mismatch("physical page exceeds usize".into())
        })?;
    program
        .target()
        .load_residual(stream, 1, &hidden_fixture(1, 9_000))?;
    let (cosine, sine) = rope(&[0]);
    let route = program.stage_prompt(stream, 1, slot, 0, &[token_id(9_000)], &cosine, &sine)?;
    program.replay_prompt(stream, route)?;
    let written = program.qualification_cache_page(stream, physical)?;
    if written.0.iter().all(|&word| word == 0) || written.1.iter().all(|&word| word == 0) {
        return Err(ResidentMtpQualificationError::Mismatch(
            "resident MTP lifecycle fixture did not write both cache planes".to_string(),
        ));
    }

    program.truncate_kv_slot_tokens(stream, slot, 0)?;
    require_zero_page(program, stream, physical, "truncate")?;
    report.lifecycle_transitions += 1;
    program.reserve_kv_slot_tokens(stream, slot, 64)?;
    let reused = usize::try_from(program.target().qualification_kv_physical_page(slot, 0)?)
        .map_err(|_| ResidentMtpQualificationError::Mismatch("reused page exceeds usize".into()))?;
    require_zero_page(program, stream, reused, "reassignment")?;
    report.lifecycle_transitions += 1;
    program.retain_kv_slot(slot)?;
    program.activate_kv_slot(slot)?;
    report.lifecycle_transitions += 2;
    program.recycle_kv_slot(stream, slot)?;
    require_zero_page(program, stream, reused, "recycle")?;
    report.lifecycle_transitions += 1;
    program.activate_kv_slot(slot)?;
    program.reserve_kv_slot_tokens(stream, slot, 1_024)?;
    report.lifecycle_transitions += 2;
    Ok(())
}

fn require_zero_page(
    program: &ResidentMtpProgram,
    stream: &CudaStream,
    physical: usize,
    role: &str,
) -> Result<(), ResidentMtpQualificationError> {
    let (key, value) = program.qualification_cache_page(stream, physical)?;
    if key.iter().any(|&word| word != 0) || value.iter().any(|&word| word != 0) {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "{role} left represented values in MTP physical page {physical}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_inputs(
    program: &ResidentMtpProgram,
    stream: &CudaStream,
    embedding_source: &[u8],
    tokens: &[u32],
    hidden: &[u16],
    slots: &[usize],
    positions: &[u32],
    cosine: &[f32],
    sine: &[f32],
    observed: &ResidentMtpObservables,
) -> Result<(), ResidentMtpQualificationError> {
    let rows = tokens.len();
    let mut expected_embedding = Vec::with_capacity(rows * Qwen38_27B::HIDDEN);
    for &token in tokens {
        let token = token as usize;
        let begin = token * Qwen38_27B::HIDDEN * 2;
        let end = begin + Qwen38_27B::HIDDEN * 2;
        expected_embedding.extend(
            embedding_source[begin..end]
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(u16::from_le_bytes),
        );
    }
    let expected_slots = if slots.is_empty() {
        vec![0u32; rows]
    } else {
        slots.iter().map(|&slot| slot as u32).collect()
    };
    let target_tables = program.target().qualification_block_tables(stream)?;
    if observed.embedding != expected_embedding
        || observed.target_hidden != hidden
        || observed.table_rows != expected_slots
        || observed.cache_positions != positions
        || observed.lengths
            != positions
                .iter()
                .map(|position| position + 1)
                .collect::<Vec<_>>()
        || observed.rope_cos != cosine
        || observed.rope_sin != sine
        || observed.block_tables != target_tables
    {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "resident MTP staged input or target handoff differs for rows={rows}"
        )));
    }
    Ok(())
}

fn verify_cache(
    rows: usize,
    slot: usize,
    positions: &[u32],
    observed: &ResidentMtpObservables,
    pages: &[CachePage],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    let page_values = cache_page_values();
    for page in pages {
        let mut touched = vec![false; page_values];
        for (row, &position) in positions.iter().enumerate() {
            let position = position as usize;
            let physical = observed.block_tables
                [slot * LONG_CONTEXT_PHYSICAL_PAGES + position / ATTENTION_PAGE_SIZE]
                as usize;
            if physical != page.physical {
                continue;
            }
            let value_begin = row * Qwen38_27B::ATTENTION_QKV_ROWS
                + Qwen38_27B::ATTENTION_QUERY_ROWS
                + Qwen38_27B::ATTENTION_KV_ROWS;
            for head in 0..Qwen38_27B::NUM_KV_HEADS {
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let index = Qwen38_27B::HEAD_DIM
                        * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * head)
                        + dimension;
                    touched[index] = true;
                    let expected =
                        observed.qkv[value_begin + head * Qwen38_27B::HEAD_DIM + dimension];
                    if page.value[index] != expected {
                        return Err(ResidentMtpQualificationError::Mismatch(format!(
                            "resident MTP rows={rows} value cache differs at page {}, index {index}",
                            page.physical
                        )));
                    }
                }
            }
        }
        for (index, (&key, &value)) in page.key.iter().zip(&page.value).enumerate() {
            if !touched[index] && (key != 0 || value != 0) {
                return Err(ResidentMtpQualificationError::Mismatch(format!(
                    "resident MTP rows={rows} modified inactive page {}, index {index}",
                    page.physical
                )));
            }
        }
        report.cache_values += 2 * page_values;
    }
    Ok(())
}

fn verify_draft_cache(
    batch: usize,
    slots: &[usize],
    positions: &[u32],
    observed: &ResidentMtpObservables,
    pages: &[CachePage],
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    let page_values = cache_page_values();
    for page in pages {
        let mut touched = vec![false; page_values];
        for (row, (&slot, &position)) in slots.iter().zip(positions).enumerate() {
            let position = position as usize;
            let physical = observed.block_tables
                [slot * LONG_CONTEXT_PHYSICAL_PAGES + position / ATTENTION_PAGE_SIZE]
                as usize;
            if physical != page.physical {
                continue;
            }
            let value_begin = row * Qwen38_27B::ATTENTION_QKV_ROWS
                + Qwen38_27B::ATTENTION_QUERY_ROWS
                + Qwen38_27B::ATTENTION_KV_ROWS;
            for head in 0..Qwen38_27B::NUM_KV_HEADS {
                for dimension in 0..Qwen38_27B::HEAD_DIM {
                    let index = Qwen38_27B::HEAD_DIM
                        * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * head)
                        + dimension;
                    touched[index] = true;
                    let expected =
                        observed.qkv[value_begin + head * Qwen38_27B::HEAD_DIM + dimension];
                    if page.value[index] != expected {
                        return Err(ResidentMtpQualificationError::Mismatch(format!(
                            "resident MTP B={batch} value cache differs at page {}, index {index}",
                            page.physical
                        )));
                    }
                }
            }
        }
        for (index, (&key, &value)) in page.key.iter().zip(&page.value).enumerate() {
            if !touched[index] && (key != 0 || value != 0) {
                return Err(ResidentMtpQualificationError::Mismatch(format!(
                    "resident MTP B={batch} modified inactive page {}, index {index}",
                    page.physical
                )));
            }
        }
        report.cache_values += 2 * page_values;
    }
    Ok(())
}

fn verify_tail_written(
    program: &ResidentMtpProgram,
    tail: usize,
    pages: &[CachePage],
) -> Result<(), ResidentMtpQualificationError> {
    for position in 0..tail {
        let physical = usize::try_from(
            program
                .target()
                .qualification_kv_physical_page(0, position)?,
        )
        .map_err(|_| ResidentMtpQualificationError::Mismatch("tail page exceeds usize".into()))?;
        let page = pages
            .iter()
            .find(|page| page.physical == physical)
            .ok_or_else(|| {
                ResidentMtpQualificationError::Mismatch(format!(
                    "tail={tail} omitted physical page {physical}"
                ))
            })?;
        for head in 0..Qwen38_27B::NUM_KV_HEADS {
            let begin = Qwen38_27B::HEAD_DIM
                * (position % ATTENTION_PAGE_SIZE + ATTENTION_PAGE_SIZE * head);
            let end = begin + Qwen38_27B::HEAD_DIM;
            if page.key[begin..end].iter().all(|&word| word == 0)
                || page.value[begin..end].iter().all(|&word| word == 0)
            {
                return Err(ResidentMtpQualificationError::Mismatch(format!(
                    "tail={tail} position={position} left an MTP K/V head unwritten"
                )));
            }
        }
    }
    Ok(())
}

fn compare_observables(
    role: &str,
    rows: usize,
    eager: &ResidentMtpObservables,
    replay: &ResidentMtpObservables,
    report: &mut ResidentMtpQualification,
) -> Result<(), ResidentMtpQualificationError> {
    if eager != replay {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "resident MTP {role} rows={rows} eager and graph seams differ"
        )));
    }
    report.graph_replay_values += observable_values(eager);
    Ok(())
}

fn compare_cache(
    role: &str,
    rows: usize,
    eager: &[CachePage],
    replay: &[CachePage],
) -> Result<(), ResidentMtpQualificationError> {
    if eager.len() != replay.len()
        || eager.iter().zip(replay).any(|(eager, replay)| {
            eager.physical != replay.physical
                || eager.key != replay.key
                || eager.value != replay.value
        })
    {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "resident MTP {role} rows={rows} eager and graph cache values differ"
        )));
    }
    Ok(())
}

fn verify_stable(
    program: &ResidentMtpProgram,
    bases: (u64, u64),
    addresses: &[usize],
    host_stagers: &[usize; 7],
    role: &str,
    rows: usize,
) -> Result<(), ResidentMtpQualificationError> {
    if (program.base_address(), program.cache_base_address()) != bases
        || program.qualification_addresses()? != addresses
        || program.qualification_host_stager_addresses() != *host_stagers
    {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "resident MTP addresses changed after {role} rows={rows}"
        )));
    }
    Ok(())
}

fn verify_no_post_warmup_allocation(
    context: &Arc<CudaContext>,
    program: &mut ResidentMtpProgram,
    stream: &CudaStream,
) -> Result<(), ResidentMtpQualificationError> {
    let slots = (0..MAX_BATCH).collect::<Vec<_>>();
    let positions = vec![130u32; MAX_BATCH];
    let tokens = token_ids(12_000, MAX_BATCH);
    let hidden = hidden_fixture(MAX_BATCH, 12_000);
    let (cosine, sine) = rope(&positions);
    program.target().load_residual(stream, MAX_BATCH, &hidden)?;
    let draft = program.stage_draft(stream, &slots, &positions, &tokens, &cosine, &sine)?;
    let staged = program
        .stage_continuation_draft(stream, &slots, &positions, &tokens, &hidden, &cosine, &sine)?;
    for _ in 0..2 {
        program.replay_draft(stream, draft)?;
        program.replay_continue_draft(stream, draft)?;
        program.replay_staged_continue_draft(stream, staged)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let before = device_memory_info(context)?;
    for _ in 0..4 {
        program.replay_draft(stream, draft)?;
        program.replay_continue_draft(stream, draft)?;
        program.replay_staged_continue_draft(stream, staged)?;
    }
    stream.synchronize().map_err(GpuError::from)?;
    let after = device_memory_info(context)?;
    if before != after {
        return Err(ResidentMtpQualificationError::Mismatch(format!(
            "post-warmup resident MTP graphs changed device memory from {before:?} to {after:?}"
        )));
    }
    Ok(())
}

fn read_cache(
    program: &ResidentMtpProgram,
    stream: &CudaStream,
    slot: usize,
    first: usize,
    rows: usize,
) -> Result<Vec<CachePage>, ResidentMtpQualificationError> {
    let mut pages = BTreeSet::new();
    for position in first..first + rows {
        pages.insert(
            usize::try_from(
                program
                    .target()
                    .qualification_kv_physical_page(slot, position)?,
            )
            .map_err(|_| {
                ResidentMtpQualificationError::Mismatch("physical page exceeds usize".into())
            })?,
        );
    }
    pages
        .into_iter()
        .map(|physical| {
            let (key, value) = program.qualification_cache_page(stream, physical)?;
            Ok(CachePage {
                physical,
                key,
                value,
            })
        })
        .collect()
}

fn read_draft_cache(
    program: &ResidentMtpProgram,
    stream: &CudaStream,
    slots: &[usize],
    positions: &[u32],
) -> Result<Vec<CachePage>, ResidentMtpQualificationError> {
    let mut pages = BTreeSet::new();
    for (&slot, &position) in slots.iter().zip(positions) {
        pages.insert(
            usize::try_from(
                program
                    .target()
                    .qualification_kv_physical_page(slot, position as usize)?,
            )
            .map_err(|_| {
                ResidentMtpQualificationError::Mismatch("physical page exceeds usize".into())
            })?,
        );
    }
    pages
        .into_iter()
        .map(|physical| {
            let (key, value) = program.qualification_cache_page(stream, physical)?;
            Ok(CachePage {
                physical,
                key,
                value,
            })
        })
        .collect()
}

fn hidden_fixture(rows: usize, seed: usize) -> Vec<u16> {
    const PATTERN: [f32; 8] = [
        0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125,
    ];
    (0..rows * Qwen38_27B::HIDDEN)
        .map(|index| f32_to_bf16(PATTERN[(index.wrapping_mul(3) + seed) & 7]))
        .collect()
}

fn token_ids(first: usize, rows: usize) -> Vec<u32> {
    (first..first + rows).map(token_id).collect()
}

fn token_id(position: usize) -> u32 {
    ((position.wrapping_mul(7_919).wrapping_add(101)) % Qwen38_27B::VOCAB) as u32
}

fn positions(first: usize, rows: usize) -> Result<Vec<u32>, ResidentMtpQualificationError> {
    (first..first + rows)
        .map(|position| {
            u32::try_from(position).map_err(|_| {
                ResidentMtpQualificationError::Mismatch("position exceeds u32".to_string())
            })
        })
        .collect()
}

fn rope(positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; positions.len() * ROTARY_PAIRS];
    let mut sine = vec![0.0; positions.len() * ROTARY_PAIRS];
    for (row, &position) in positions.iter().enumerate() {
        for pair in 0..ROTARY_PAIRS {
            let frequency = 10_000_000.0f64.powf(-((2 * pair) as f64) / 64.0);
            let (sin, cos) = (f64::from(position) * frequency).sin_cos();
            cosine[row * ROTARY_PAIRS + pair] = cos as f32;
            sine[row * ROTARY_PAIRS + pair] = sin as f32;
        }
    }
    (cosine, sine)
}

fn cache_page_values() -> usize {
    Qwen38_27B::NUM_KV_HEADS * ATTENTION_PAGE_SIZE * Qwen38_27B::HEAD_DIM
}

fn observable_values(observed: &ResidentMtpObservables) -> usize {
    observed.embedding.len()
        + observed.target_hidden.len()
        + observed.normalized_embedding.len()
        + observed.normalized_hidden.len()
        + observed.residual.len()
        + observed.attention_normalized.len()
        + observed.qkv.len()
        + observed.rope_cos.len()
        + observed.rope_sin.len()
        + observed.block_tables.len()
        + observed.table_rows.len()
        + observed.cache_positions.len()
        + observed.lengths.len()
        + observed.query.len()
        + observed.attention.as_deref().map_or(0, <[f32]>::len)
        + observed
            .attention_activation
            .as_deref()
            .map_or(0, <[u16]>::len)
        + observed.attention_branch.as_deref().map_or(0, <[u16]>::len)
        + observed
            .post_attention_residual
            .as_deref()
            .map_or(0, <[u16]>::len)
        + observed.mlp_normalized.as_deref().map_or(0, <[u16]>::len)
        + observed.swiglu.as_deref().map_or(0, <[u16]>::len)
        + observed.mlp_branch.as_deref().map_or(0, <[u16]>::len)
        + observed.residual_output.as_deref().map_or(0, <[u16]>::len)
        + observed.final_normalized.as_deref().map_or(0, <[u16]>::len)
        + observed
            .lm_head_activation_codes
            .as_deref()
            .map_or(0, <[u8]>::len)
        + observed
            .lm_head_activation_scales
            .as_deref()
            .map_or(0, <[f32]>::len)
        + observed.logits.as_deref().map_or(0, <[u16]>::len)
}

#[cfg(test)]
mod tests {
    use super::{PROMPT_ROUTES, REALIGN_ROUTES};

    #[test]
    fn resident_mtp_suite_inventory_is_exact() {
        assert_eq!(PROMPT_ROUTES, [1, 32, 64, 128, 1_024]);
        assert_eq!(REALIGN_ROUTES, 4);
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_suite_source_values_match_every_route_and_lifecycle() {
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let report = super::qualify_resident_mtp(Path::new(&root)).unwrap();
        assert_eq!(report.source_oracle_suites, 2);
        assert_eq!(report.prompt_routes, 5);
        assert_eq!(report.draft_routes, 8);
        assert_eq!(report.continuation_routes, 8);
        assert_eq!(report.staged_continuation_routes, 8);
        assert_eq!(report.prime_routes, 4);
        assert_eq!(report.realign_routes, 4);
        assert_eq!(report.tail_routes, 31);
        assert_eq!(report.lifecycle_transitions, 7);
        assert!(report.graph_replay_values > 0);
        assert!(report.cache_values > 0);
    }

    use std::path::Path;
}

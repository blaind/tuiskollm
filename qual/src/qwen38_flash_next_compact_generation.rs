//! Source-backed gate for eight-slot compact Qwen3.8 Flash-Next generation.
//!
//! Each sequence must match its solo output while mixed lengths exercise dense widths,
//! noncontiguous survivors, retirement, recycled slots, and independent carries.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, GeneratedText, Qwen38FlashNextBatchTelemetry,
    Qwen38FlashNextResidentBatchGenerator, ResidentBatchEvent, ResidentBatchEvents,
    ResidentRequestId, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{
    CheckpointError, CheckpointSnapshot, Qwen38FlashNext, Qwen38FlashNextEngramCarry,
};

/// Physical slots the Flash-Next page pool funds.
const SLOTS: usize = 8;

/// Prompt width of the tiled fixture, which is one whole `T = 128` prefill route.
const TILED_PROMPT_TOKENS: usize = 128;

/// Shared budget that keeps the carry scenario batched.
const CARRY_BUDGET: usize = 4;

/// Distinct prompts with unsorted budgets, placing the first retirement inside the active order.
const MIXED_FIXTURES: [(&str, usize); SLOTS] = [
    ("Name one primary color.", 5),
    ("Say hello.", 2),
    ("Describe a river in one sentence.", 8),
    ("What is two plus two?", 3),
    ("List three fruits, separated by commas.", 9),
    ("Give one fact about the moon.", 4),
    ("Write one short sentence about rain.", 7),
    ("Name the largest ocean.", 6),
];

/// Failure of the compact Flash-Next scheduler gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen38FlashNextCompactGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Tokenizer or chat-template admission failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An externally visible scheduling boundary differed.
    #[error("Flash-Next compact generation qualification failed: {0}")]
    Mismatch(String),
}

type QualResult<T> = Result<T, Qwen38FlashNextCompactGenerationQualificationError>;

/// Compact routes, lifecycle seams, and batch independence checked for Flash-Next serving.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextCompactGenerationQualification {
    /// Exact decode widths exercised by identical concurrent requests, `B=1` through `B=8`.
    pub route_batches: usize,
    /// Sequences whose batched output was compared byte for byte with a solo run.
    pub independent_sequences: usize,
    /// Decode widths the mixed drain actually reached, in round order.
    pub mixed_widths: Vec<usize>,
    /// Physical slot recycled and refilled while other requests stayed live.
    pub recycled_slot: usize,
    /// Native prompt tokens admitted into a hole while other requests were active.
    pub concurrent_prefill_tokens: usize,
    /// Active cancellation boundaries exercised mid-batch.
    pub cancellations: usize,
    /// Admissions refused because every physical slot was already active.
    pub capacity_refusals: usize,
    /// Per-slot engram carries compared against their own sequence's tokens.
    pub engram_carry_checks: usize,
    /// Exact bytes across all retained device arenas.
    pub arena_bytes: usize,
    /// Exact page-locked staging and double-logit-bank bytes.
    pub host_stager_bytes: usize,
    /// Stable retained device and pinned addresses.
    pub stable_addresses: usize,
    /// Decode evidence split by the width of the round that produced it.
    pub telemetry: Qwen38FlashNextBatchTelemetry,
}

/// Qualifies the compact Flash-Next scheduler against solo execution of every sequence.
pub fn qualify_qwen38_flash_next_compact_generation(
    root: &Path,
) -> QualResult<Qwen38FlashNextCompactGenerationQualification> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38FlashNext>::open(root)?);
    let frontend = TextFrontend::open(snapshot.as_ref())?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let mut generator =
        Qwen38FlashNextResidentBatchGenerator::from_snapshot(&context, snapshot, None)?;
    verify_compact_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();

    let mixed = MIXED_FIXTURES.map(|(content, budget)| greedy_request(content, budget));
    let tiled = exact_prompt_request(&frontend, TILED_PROMPT_TOKENS)?;
    let short = greedy_request("Say hello.", 2);

    // Build every authority on an otherwise empty scheduler.
    let solo = mixed
        .iter()
        .map(|request| run_alone(&mut generator, request))
        .collect::<QualResult<Vec<_>>>()?;
    let tiled_solo = run_alone(&mut generator, &tiled)?;
    let short_solo = run_alone(&mut generator, &short)?;
    if generator.active_requests() != 0 {
        return Err(mismatch("solo runs left a slot active"));
    }

    let before = device_memory_info(generator.context())?;
    generator.reset_telemetry();
    verify_route_inventory(&mut generator, &short, &short_solo)?;
    let mixed_widths = verify_batch_independence(&mut generator, &mixed, &solo)?;
    let engram_carry_checks = verify_engram_carries(&mut generator, &frontend)?;
    let recycled_slot =
        verify_hole_reuse(&mut generator, &short, &short_solo, &tiled, &tiled_solo)?;
    let capacity_refusals = verify_capacity_refusal(&mut generator, &short, &short_solo)?;
    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(mismatch(format!(
            "device memory changed across compact scheduling: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(mismatch("Flash-Next compact owner addresses changed"));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen38FlashNextCompactGenerationQualification {
        route_batches: SLOTS,
        independent_sequences: SLOTS,
        mixed_widths,
        recycled_slot,
        concurrent_prefill_tokens: TILED_PROMPT_TOKENS,
        cancellations: 1,
        capacity_refusals,
        engram_carry_checks,
        arena_bytes: generator.arena_bytes()?,
        host_stager_bytes: generator.host_stager_bytes(),
        stable_addresses: stable_addresses.len(),
        telemetry: generator.batch_telemetry(),
    })
}

fn mismatch(message: impl Into<String>) -> Qwen38FlashNextCompactGenerationQualificationError {
    Qwen38FlashNextCompactGenerationQualificationError::Mismatch(message.into())
}

fn greedy_request(content: &str, maximum: usize) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", content)]);
    // Disable thinking so short budgets reach visible content.
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = maximum;
    request
}

/// A request whose rendered prompt is exactly `target_tokens` long.
fn exact_prompt_request(
    frontend: &TextFrontend,
    target_tokens: usize,
) -> QualResult<ChatGenerationRequest> {
    let mut lower = 1usize;
    let mut upper = target_tokens;
    while lower < upper {
        let words = lower + (upper - lower) / 2;
        let request = greedy_request(&vec!["x"; words].join(" "), 1);
        let tokens = frontend
            .encode_chat(&request.messages, &request.template)?
            .len();
        if tokens < target_tokens {
            lower = words + 1;
        } else {
            upper = words;
        }
    }
    let request = greedy_request(&vec!["x"; lower].join(" "), 2);
    let actual = frontend
        .encode_chat(&request.messages, &request.template)?
        .len();
    if actual != target_tokens {
        return Err(mismatch(format!(
            "could not construct an exact {target_tokens}-token prompt; the deterministic fixture \
             renders {actual}"
        )));
    }

    Ok(request)
}

/// Drives one request to completion on an otherwise empty scheduler.
fn run_alone(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    request: &ChatGenerationRequest,
) -> QualResult<GeneratedText> {
    if generator.active_requests() != 0 {
        return Err(mismatch("a solo run started beside a live request"));
    }
    let admission = generator.admit(request)?;
    if let Some(output) = admission.completed {
        return Ok(output);
    }
    loop {
        let events = generator.step()?;
        if events.len() != 1 {
            return Err(mismatch(format!(
                "a solo round returned {} events",
                events.len()
            )));
        }
        let event = events.iter().next().expect("one solo event");
        if let Some(output) = &event.completed {
            return Ok(output.clone());
        }
    }
}

/// Runs every decode width with identical rows to isolate route selection.
fn verify_route_inventory(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
) -> QualResult<()> {
    if expected.token_ids.len() != 2 {
        return Err(mismatch(format!(
            "the route fixture generated {} tokens rather than its two-token budget",
            expected.token_ids.len()
        )));
    }
    for batch in 1..=SLOTS {
        let mut requests = [None; SLOTS];
        for (row, request_id) in requests[..batch].iter_mut().enumerate() {
            let admission = generator.admit(request)?;
            if generator.qualification_slot(admission.request_id) != Some(row) {
                return Err(mismatch(format!(
                    "B={batch} admission {row} did not take physical slot {row}"
                )));
            }
            *request_id = Some(admission.request_id);
        }
        // Fresh primes hold logits and have no pending replay.
        if !generator.qualification_round()?.is_empty() {
            return Err(mismatch(format!(
                "B={batch} had a pending round before its first sampling step"
            )));
        }
        let first = generator.step()?;
        let round = generator.qualification_round()?;
        if round.rows() != batch || round.slots() != (0..batch).collect::<Vec<_>>() {
            return Err(mismatch(format!(
                "B={batch} planned a {}-row round over slots {:?}",
                round.rows(),
                round.slots()
            )));
        }
        let second = generator.step()?;
        if first.len() != batch || second.len() != batch {
            return Err(mismatch(format!(
                "B={batch} did not preserve its compact event inventory"
            )));
        }
        for (row, (first, second)) in first.iter().zip(second.iter()).enumerate() {
            let request_id = requests[row].expect("every admitted row has an identity");
            require_event(first, request_id, expected, 0)?;
            require_event(second, request_id, expected, 1)?;
        }
        if generator.active_requests() != 0 {
            return Err(mismatch(format!(
                "B={batch} left completed requests active"
            )));
        }
    }

    Ok(())
}

/// Drains eight distinct sequences and returns each decode width used.
fn verify_batch_independence(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    requests: &[ChatGenerationRequest; SLOTS],
    solo: &[GeneratedText],
) -> QualResult<Vec<usize>> {
    let mut identities = Vec::with_capacity(SLOTS);
    for (index, request) in requests.iter().enumerate() {
        let admission = generator.admit(request)?;
        if generator.qualification_slot(admission.request_id) != Some(index) {
            return Err(mismatch(format!(
                "mixed admission {index} did not take physical slot {index}"
            )));
        }
        identities.push(admission.request_id);
    }
    let mut streamed = vec![Vec::new(); SLOTS];
    let mut finished = vec![None; SLOTS];
    let mut widths = Vec::new();
    while generator.active_requests() > 0 {
        let planned = generator.qualification_round()?;
        if !planned.is_empty() {
            widths.push(planned.rows());
            // A planned round names each pending live slot once.
            let mut named = planned.slots().to_vec();
            named.sort_unstable();
            named.dedup();
            if named.len() != planned.rows() {
                return Err(mismatch(format!(
                    "a planned round named a slot twice: {:?}",
                    planned.slots()
                )));
            }
        }
        let events = generator.step()?;
        for event in events.iter() {
            let lane = identities
                .iter()
                .position(|identity| *identity == event.request_id)
                .ok_or_else(|| mismatch("a round produced an event for an unknown request"))?;
            streamed[lane].push(event.step.token_id);
            if let Some(output) = &event.completed {
                if finished[lane].is_some() {
                    return Err(mismatch(format!("lane {lane} completed twice")));
                }
                finished[lane] = Some(output.clone());
            }
        }
    }

    for (lane, expected) in solo.iter().enumerate() {
        let actual = finished[lane]
            .as_ref()
            .ok_or_else(|| mismatch(format!("lane {lane} never completed")))?;
        // Compare the complete externally visible result.
        if actual.prompt.token_ids != expected.prompt.token_ids
            || actual.prompt.rendered_bytes != expected.prompt.rendered_bytes
            || actual.token_ids != expected.token_ids
            || actual.text != expected.text
            || actual.finish_reason != expected.finish_reason
        {
            return Err(mismatch(format!(
                "lane {lane} run beside seven others differs from the same request run alone: \
                 alone {:?} -> {:?}, batched {:?} -> {:?}",
                expected.prompt.token_ids.len(),
                expected.token_ids,
                actual.prompt.token_ids.len(),
                actual.token_ids
            )));
        }
        if streamed[lane] != actual.token_ids {
            return Err(mismatch(format!(
                "lane {lane} streamed {:?} but completed with {:?}",
                streamed[lane], actual.token_ids
            )));
        }
    }
    if widths.first() != Some(&SLOTS) {
        return Err(mismatch(format!(
            "the mixed drain never reached a full eight-row round: widths {widths:?}"
        )));
    }
    // Greedy stops may retire multiple rows together, but widths cannot grow.
    if widths.windows(2).any(|pair| pair[0] < pair[1]) {
        return Err(mismatch(format!(
            "a round of the mixed drain was wider than the round before it: widths {widths:?}"
        )));
    }
    if widths.last() >= widths.first() {
        return Err(mismatch(format!(
            "the mixed drain never narrowed, so no round ran beside a hole: widths {widths:?}"
        )));
    }

    Ok(widths)
}

/// Checks each slot carry against its own sequence while rows and slots differ.
fn verify_engram_carries(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    frontend: &TextFrontend,
) -> QualResult<usize> {
    // Retired slots restart while surviving carries advance independently.
    let requests = MIXED_FIXTURES.map(|(content, _)| greedy_request(content, CARRY_BUDGET));
    let restart = Qwen38FlashNextEngramCarry::start().previous();
    let mut retired = [false; SLOTS];
    let mut checks = 0;
    let mut identities = Vec::with_capacity(SLOTS);
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(SLOTS);
    for (index, request) in requests.iter().enumerate() {
        let admission = generator.admit(request)?;
        identities.push(admission.request_id);
        let prompt = frontend.encode_chat(&request.messages, &request.template)?;
        // Prime must update only the admitted slot's carry.
        for (earlier, prompt) in prompts.iter().enumerate() {
            require_carry(generator, earlier, tail_of(prompt)?)?;
            checks += 1;
        }
        require_carry(generator, index, tail_of(&prompt)?)?;
        checks += 1;
        prompts.push(prompt);
    }

    let first = generator.step()?;
    if first.len() != SLOTS {
        return Err(mismatch(format!(
            "the carry scenario's first round returned {} events",
            first.len()
        )));
    }
    let sampled = first
        .iter()
        .map(|event| event.step.token_id)
        .collect::<Vec<_>>();
    for (slot, event) in first.iter().enumerate() {
        retired[slot] = event.completed.is_some();
    }
    // Sampling prime logits does not replay or advance carries.
    for (slot, prompt) in prompts.iter().enumerate() {
        let expected = if retired[slot] {
            restart
        } else {
            tail_of(prompt)?
        };
        require_carry(generator, slot, expected)?;
        checks += 1;
    }

    generator.step()?;
    // One replay advances each surviving carry by its own sampled token.
    for (slot, prompt) in prompts.iter().enumerate() {
        if retired[slot] {
            require_carry(generator, slot, restart)?;
            checks += 1;
            continue;
        }
        let mut advanced = prompt.clone();
        advanced.push(sampled[slot]);
        let expected = tail_of(&advanced)?;
        let expected = if generator.qualification_slot(identities[slot]).is_some() {
            expected
        } else {
            // Retirement clears the carry after replay.
            restart
        };
        require_carry(generator, slot, expected)?;
        checks += 1;
    }

    for identity in identities {
        if generator.qualification_slot(identity).is_some() {
            generator.cancel(identity)?;
        }
    }
    // Every recycled slot returns to the segment boundary.
    for slot in 0..SLOTS {
        require_carry(generator, slot, restart)?;
        checks += 1;
    }
    if generator.active_requests() != 0 {
        return Err(mismatch("the carry scenario left a request active"));
    }

    Ok(checks)
}

/// The carry a sequence holds once its whole prompt has been staged.
fn tail_of(prompt: &[u32]) -> QualResult<[u32; 2]> {
    Ok(Qwen38FlashNextEngramCarry::after(prompt)?.previous())
}

fn require_carry(
    generator: &Qwen38FlashNextResidentBatchGenerator,
    slot: usize,
    expected: [u32; 2],
) -> QualResult<()> {
    let actual = generator.qualification_engram_carry(slot)?;
    if actual != expected {
        return Err(mismatch(format!(
            "slot {slot} carries {actual:?}, expected {expected:?}"
        )));
    }

    Ok(())
}

/// Cancels a middle request, then admits a tiled prompt into the hole it left.
fn verify_hole_reuse(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    short: &ChatGenerationRequest,
    short_solo: &GeneratedText,
    tiled: &ChatGenerationRequest,
    tiled_solo: &GeneratedText,
) -> QualResult<usize> {
    let free_before = generator.qualification_free_pages();
    let first = generator.admit(short)?;
    let cancelled = generator.admit(short)?;
    let third = generator.admit(short)?;
    if generator.qualification_slot(first.request_id) != Some(0)
        || generator.qualification_slot(cancelled.request_id) != Some(1)
        || generator.qualification_slot(third.request_id) != Some(2)
    {
        return Err(mismatch(
            "cold compact admissions did not fill slots 0, 1, 2",
        ));
    }
    let first_round = generator.step()?;
    require_round(
        &first_round,
        &[
            (first.request_id, short_solo, 0),
            (cancelled.request_id, short_solo, 0),
            (third.request_id, short_solo, 0),
        ],
    )?;

    let cancellation = generator.cancel(cancelled.request_id)?;
    if cancellation.device_retained_tokens != 0
        || cancellation.output.token_ids != short_solo.token_ids[..1]
    {
        return Err(mismatch(
            "cancellation changed its host boundary or retained device state",
        ));
    }
    // Cancellation releases only the cancelled slot's pages.
    if generator.qualification_slot(first.request_id) != Some(0)
        || generator.qualification_slot(third.request_id) != Some(2)
    {
        return Err(mismatch("cancellation moved a surviving request's slot"));
    }
    if generator.qualification_slot_tokens(1)? != 0 {
        return Err(mismatch("a cancelled slot kept its committed length"));
    }

    let joined = generator.admit(tiled)?;
    if joined.native_prefill_tokens != TILED_PROMPT_TOKENS
        || generator.qualification_slot(joined.request_id) != Some(1)
    {
        return Err(mismatch(format!(
            "the concurrent tiled admission took native={} slot={:?}",
            joined.native_prefill_tokens,
            generator.qualification_slot(joined.request_id)
        )));
    }
    // The newcomer joins survivors after its private prime.
    let planned = generator.qualification_round()?;
    if planned.slots() != [0, 2] {
        return Err(mismatch(format!(
            "the round after a mid-flight admission planned slots {:?}",
            planned.slots()
        )));
    }
    let second_round = generator.step()?;
    require_round(
        &second_round,
        &[
            (first.request_id, short_solo, 1),
            (third.request_id, short_solo, 1),
            (joined.request_id, tiled_solo, 0),
        ],
    )?;
    while generator.active_requests() > 0 {
        generator.step()?;
    }
    // Finished and cancelled requests must return every reserved page.
    if generator.qualification_free_pages() != free_before {
        return Err(mismatch(format!(
            "the shared pool holds {} free pages after four admissions and one cancellation, and \
             held {free_before} before them",
            generator.qualification_free_pages()
        )));
    }

    Ok(1)
}

/// A ninth admission is refused, and the eight it could not join are undisturbed.
fn verify_capacity_refusal(
    generator: &mut Qwen38FlashNextResidentBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
) -> QualResult<usize> {
    let mut identities = Vec::with_capacity(SLOTS);
    for _ in 0..SLOTS {
        identities.push(generator.admit(request)?.request_id);
    }
    let free_before = generator.qualification_free_pages();
    let refusal = generator
        .admit(request)
        .err()
        .ok_or_else(|| mismatch("a ninth request was admitted into eight funded slots"))?;
    if !refusal.to_string().contains("slots are active") {
        return Err(mismatch(format!(
            "the capacity refusal did not name the slots: {refusal}"
        )));
    }
    // Refusal must leave all eight live sequences untouched.
    if generator.qualification_free_pages() != free_before {
        return Err(mismatch("a refused admission took pages from the pool"));
    }
    if generator.active_requests() != SLOTS {
        return Err(mismatch("a refused admission changed the active count"));
    }

    let first = generator.step()?;
    let second = generator.step()?;
    for (row, (first, second)) in first.iter().zip(second.iter()).enumerate() {
        require_event(first, identities[row], expected, 0)?;
        require_event(second, identities[row], expected, 1)?;
    }
    if generator.active_requests() != 0 {
        return Err(mismatch("the capacity scenario left a request active"));
    }

    Ok(1)
}

fn require_round(
    events: &ResidentBatchEvents,
    expected: &[(ResidentRequestId, &GeneratedText, usize)],
) -> QualResult<()> {
    if events.len() != expected.len() {
        return Err(mismatch(format!(
            "a compact round returned {} events, expected {}",
            events.len(),
            expected.len()
        )));
    }
    for (event, &(request, output, token)) in events.iter().zip(expected) {
        require_event(event, request, output, token)?;
    }

    Ok(())
}

fn require_event(
    event: &ResidentBatchEvent,
    request: ResidentRequestId,
    expected: &GeneratedText,
    token: usize,
) -> QualResult<()> {
    if event.request_id != request || event.step.token_id != expected.token_ids[token] {
        return Err(mismatch(format!(
            "request {} differs at token {token}",
            request.get()
        )));
    }
    let terminal = token + 1 == expected.token_ids.len();
    if terminal {
        if event.completed.as_ref().is_none_or(|actual| {
            actual.prompt.token_ids != expected.prompt.token_ids
                || actual.prompt.rendered_bytes != expected.prompt.rendered_bytes
                || actual.token_ids != expected.token_ids
                || actual.text != expected.text
                || actual.finish_reason != expected.finish_reason
        }) {
            return Err(mismatch(format!(
                "request {} terminal output differs from its solo run",
                request.get()
            )));
        }
    } else if event.completed.is_some() {
        return Err(mismatch(format!(
            "request {} completed before token {token}",
            request.get()
        )));
    }

    Ok(())
}

fn verify_compact_owner(generator: &Qwen38FlashNextResidentBatchGenerator) -> QualResult<()> {
    if generator.slot_capacity() != SLOTS {
        return Err(mismatch(format!(
            "the compact owner funds {} slots",
            generator.slot_capacity()
        )));
    }
    if generator.context_capacity() != 2_051 {
        return Err(mismatch(format!(
            "the compact owner admits {} context tokens rather than the proven dense band",
            generator.context_capacity()
        )));
    }
    if !generator.mapped_primary() {
        return Err(mismatch(
            "the compact owner did not open under the mapped-primary host posture",
        ));
    }
    let addresses = generator.qualification_addresses();
    let mut unique = addresses.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(mismatch(
            "the compact owner's retained addresses are invalid",
        ));
    }

    Ok(())
}

/// Prints the compact scheduler verdict and its per-width decode evidence.
pub fn print_qwen38_flash_next_compact_generation_report(
    report: &Qwen38FlashNextCompactGenerationQualification,
) {
    println!(
        "PASS qwen38_flash_next-compact-generation: B=1..{} routes; {} sequences byte-identical to their \
         solo runs; mixed drain widths {:?}; slot {} recycled and refilled with a {}-token tiled \
         prompt; {} cancellation; {} capacity refusal; {} engram carry checks",
        report.route_batches,
        report.independent_sequences,
        report.mixed_widths,
        report.recycled_slot,
        report.concurrent_prefill_tokens,
        report.cancellations,
        report.capacity_refusals,
        report.engram_carry_checks,
    );
    println!(
        "  arenas {} B, pinned {} B, {} stable addresses",
        report.arena_bytes, report.host_stager_bytes, report.stable_addresses
    );
    println!(
        "  {:>5}  {:>7}  {:>8}  {:>10}  {:>10}  {:>12}",
        "width", "rounds", "tokens", "round ms", "tok/s", "expert hit"
    );
    for width in 1..=SLOTS {
        let evidence = report
            .telemetry
            .at(width)
            .expect("every admitted width is addressable");
        if evidence.rounds() == 0 {
            continue;
        }
        println!(
            "  {width:>5}  {:>7}  {:>8}  {:>10.2}  {:>10.2}  {:>12.4}",
            evidence.rounds(),
            evidence.tokens(),
            evidence.round_ms(),
            evidence.tokens_per_second(),
            evidence.expert_hit_rate(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MIXED_FIXTURES, Qwen38FlashNextCompactGenerationQualificationError, SLOTS,
        print_qwen38_flash_next_compact_generation_report,
        qualify_qwen38_flash_next_compact_generation,
    };
    use std::path::PathBuf;

    #[test]
    fn the_mixed_fixtures_retire_from_the_middle_of_the_active_order() {
        // The shortest budget opens an interior hole.
        let budgets = MIXED_FIXTURES.map(|(_, budget)| budget);
        let shortest = budgets
            .iter()
            .enumerate()
            .min_by_key(|(_, budget)| **budget)
            .expect("the fixture table is not empty")
            .0;
        assert!(shortest > 0 && shortest < SLOTS - 1, "slot {shortest}");

        let mut sorted = budgets;
        sorted.sort_unstable();
        assert_eq!(sorted, [2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(budgets.len(), SLOTS);

        let mut prompts = MIXED_FIXTURES.map(|(content, _)| content).to_vec();
        let distinct = prompts.len();
        prompts.sort_unstable();
        prompts.dedup();
        assert_eq!(prompts.len(), distinct, "the fixtures share a prompt");
    }

    #[test]
    #[ignore = "requires the pinned Flash-Next snapshot and an exclusive SM120 device"]
    fn compact_scheduling_is_byte_identical_to_running_each_sequence_alone()
    -> Result<(), Qwen38FlashNextCompactGenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT").ok_or_else(|| {
            Qwen38FlashNextCompactGenerationQualificationError::Mismatch(
                "set TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT to the admitted revision".into(),
            )
        })?;
        let report = qualify_qwen38_flash_next_compact_generation(&PathBuf::from(root))?;
        print_qwen38_flash_next_compact_generation_report(&report);

        assert_eq!(report.route_batches, SLOTS);
        assert_eq!(report.independent_sequences, SLOTS);
        assert_eq!(report.mixed_widths.first(), Some(&SLOTS));
        assert_eq!(report.recycled_slot, 1);
        assert_eq!(report.concurrent_prefill_tokens, 128);
        assert_eq!(report.cancellations, 1);
        assert_eq!(report.capacity_refusals, 1);
        assert_eq!(report.engram_carry_checks, 36 + SLOTS + SLOTS + SLOTS);
        assert_eq!(report.stable_addresses, 3);
        assert_eq!(report.arena_bytes, 30_675_307_776);
        assert_eq!(report.host_stager_bytes, 139_399_168);
        assert!(report.telemetry.at(SLOTS)?.rounds() > 0);
        Ok(())
    }
}

//! Source-backed gate for compact target-plus-MTP generation.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, EngineErrorCode, GeneratedText, ResidentMtpBatchGenerator,
    ResidentRequestId, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen38_27B};

const EXACT_BATCHES: std::ops::RangeInclusive<usize> = 1..=8;
const EXACT_VERIFY_TOKENS: std::ops::RangeInclusive<usize> = 1..=4;

/// Failure of the compact MTP generation gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentMtpBatchGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact device was unavailable exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An observable scheduler or mathematical boundary differed.
    #[error("resident MTP batch generation qualification failed: {0}")]
    Mismatch(String),
}

/// Exact route, lifecycle, sampling, and ownership evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMtpBatchGenerationQualification {
    /// Every exact `(B, K)` target transaction exercised.
    pub verification_routes: usize,
    /// Every exact compact seed/continuation batch exercised.
    pub draft_batches: usize,
    /// Full-vocabulary target rows checked by an independent BF16 argmax.
    pub oracle_rows: usize,
    /// Cancellation boundaries exercised while other slots remained live.
    pub cancellations: usize,
    /// Exact full-prefix restores exercised.
    pub exact_prefix_reuses: usize,
    /// Completed raw generations restored through their complete-message fallback.
    pub message_boundary_fallbacks: usize,
    /// Retained spans rejected after divergence before their complete boundary.
    pub safe_cold_fallbacks: usize,
    /// Full-pool admissions that evicted an unrelated inactive prefix.
    pub page_pressure_evictions: usize,
    /// Active-owner page shortages refused without losing retained state.
    pub page_pressure_refusals: usize,
    /// Sampled scheduler lanes completed with deterministic per-request RNG state.
    pub sampled_lanes: usize,
    /// Complete greedy lanes compared with the independent B=1 sequence.
    pub greedy_invariant_lanes: usize,
    /// Complete target-plus-MTP device ownership.
    pub device_owner_bytes: usize,
    /// Complete page-locked program and scheduler ownership.
    pub host_stager_bytes: usize,
    /// Page-locked cancellation snapshots across all eight slots.
    pub message_boundary_snapshot_bytes: usize,
    /// Exact shared page-route host ownership.
    pub kv_route_host_bytes: usize,
}

/// Qualifies compact greedy/sampled scheduling and shared slot lifecycle on the pinned source.
pub fn qualify_resident_mtp_batch_generation(
    root: &Path,
) -> Result<ResidentMtpBatchGenerationQualification, ResidentMtpBatchGenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses()?;

    // Warm every owned transfer before the memory stability observation.
    let warm = generator.admit(&greedy_request("Warm the exact MTP owner.", 5))?;
    let _ = generator.step()?;
    let _ = generator.step()?;
    if generator
        .active_request_ids()
        .any(|request| request == warm.request_id)
    {
        let _ = generator.cancel(warm.request_id)?;
    }
    generator.qualification_clear_retained()?;
    let before = device_memory_info(generator.context())?;

    let mut oracle_rows = 0;
    for tokens in EXACT_VERIFY_TOKENS {
        for batch in EXACT_BATCHES {
            oracle_rows += qualify_exact_route(&mut generator, batch, tokens)?;
            generator.qualification_clear_retained()?;
        }
    }
    let (cancellations, exact_prefix_reuses, safe_cold_fallbacks) =
        qualify_reuse_cancellation_and_recycling(&mut generator)?;
    generator.qualification_clear_retained()?;
    let message_boundary_fallbacks = qualify_completed_message_fallback(&mut generator)?;
    generator.qualification_clear_retained()?;
    let (page_pressure_evictions, page_pressure_refusals) =
        qualify_page_pressure_eviction(&mut generator)?;
    generator.qualification_clear_retained()?;
    let sampled_lanes = qualify_sampled_batch(&mut generator)?;
    generator.qualification_clear_retained()?;
    let greedy_invariant_lanes = qualify_greedy_batch_invariance(&mut generator)?;
    generator.qualification_clear_retained()?;

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!(
                "device memory changed after compact MTP warmup: before={before:?}, after={after:?}"
            ),
        ));
    }
    if generator.qualification_addresses()? != stable_addresses {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "compact MTP owner addresses changed after warmup".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(ResidentMtpBatchGenerationQualification {
        verification_routes: 32,
        draft_batches: 8,
        oracle_rows,
        cancellations,
        exact_prefix_reuses,
        message_boundary_fallbacks,
        safe_cold_fallbacks,
        page_pressure_evictions,
        page_pressure_refusals,
        sampled_lanes,
        greedy_invariant_lanes,
        device_owner_bytes: generator.device_owner_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        message_boundary_snapshot_bytes: generator.message_boundary_snapshot_bytes(),
        kv_route_host_bytes: generator.kv_route_host_bytes(),
    })
}

fn qualify_completed_message_fallback(
    generator: &mut ResidentMtpBatchGenerator,
) -> Result<usize, ResidentMtpBatchGenerationQualificationError> {
    let opening = greedy_request("Give a brief greeting.", 8);
    let mut divergent = greedy_request("Give a brief greeting.", 8);
    divergent.messages.push(ChatMessage::new(
        "assistant",
        "This deliberately differs from the generated assistant turn.",
    ));
    divergent
        .messages
        .push(ChatMessage::new("user", "Now name one primary color."));

    let cold = generator.admit(&divergent)?;
    let cold_anchor = one_token(&generator.step()?, cold.request_id)?;
    let _ = generator.cancel(cold.request_id)?;
    generator.qualification_clear_retained()?;

    let admitted = generator.admit(&opening)?;
    let slot = generator
        .qualification_slot(admitted.request_id)
        .ok_or_else(|| mismatch("completed-prefix seed has no physical slot"))?;
    let mut output = None;
    while generator
        .active_request_ids()
        .any(|request| request == admitted.request_id)
    {
        let events = generator.step()?;
        if let Some(completed) = events
            .iter()
            .find(|event| event.request_id == admitted.request_id)
            .and_then(|event| event.completed.as_ref())
        {
            output = Some(completed.clone());
        }
    }
    let output = output.ok_or_else(|| mismatch("completed-prefix seed returned no output"))?;
    let boundary = output.prompt.message_boundary_tokens;
    let retained = generator
        .qualification_retained_tokens(slot)
        .ok_or_else(|| mismatch("normal completion retained no processed prefix"))?;
    if retained <= boundary
        || generator.qualification_retained_message_boundary(slot) != Some(boundary)
    {
        return Err(mismatch(format!(
            "normal completion retained {retained} tokens without its {boundary}-token message boundary"
        )));
    }

    let resumed = generator.admit(&divergent)?;
    if resumed.device_reused_tokens != boundary
        || generator.qualification_slot(resumed.request_id) != Some(slot)
    {
        return Err(mismatch(format!(
            "divergent history reused {}/{} tokens on slot {:?}/{slot}",
            resumed.device_reused_tokens,
            boundary,
            generator.qualification_slot(resumed.request_id)
        )));
    }
    let warm_anchor = one_token(&generator.step()?, resumed.request_id)?;
    if warm_anchor != cold_anchor {
        return Err(mismatch(format!(
            "message-boundary fallback changed the next token: cold {cold_anchor}, warm {warm_anchor}"
        )));
    }
    let _ = generator.cancel(resumed.request_id)?;

    Ok(1)
}

fn mismatch(message: impl Into<String>) -> ResidentMtpBatchGenerationQualificationError {
    ResidentMtpBatchGenerationQualificationError::Mismatch(message.into())
}

fn qualify_exact_route(
    generator: &mut ResidentMtpBatchGenerator,
    batch: usize,
    tokens: usize,
) -> Result<usize, ResidentMtpBatchGenerationQualificationError> {
    let request = greedy_request("Name one primary color.", tokens + 1);
    let mut requests = [None; 8];
    for destination in &mut requests[..batch] {
        *destination = Some(generator.admit(&request)?.request_id);
    }
    let anchors = generator.step()?;
    if anchors.len() != batch {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!(
                "B={batch} anchor transaction returned {} events",
                anchors.len()
            ),
        ));
    }
    for (lane, event) in anchors.iter().enumerate() {
        if event.request_id != requests[lane].expect("route request exists")
            || event.len() != 1
            || event.completed.is_some()
        {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                format!("B={batch} lane {lane} changed its anchor seam"),
            ));
        }
    }

    let verified = generator.step()?;
    if verified.len() != batch {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!("B={batch} K={tokens} returned {} events", verified.len()),
        ));
    }
    let logits = generator.qualification_target_logits(batch * tokens)?;
    let mut checked = 0;
    for (lane, event) in verified.iter().enumerate() {
        if event.request_id != requests[lane].expect("route request exists")
            || event.stats.verification_routes[tokens - 1] != 1
        {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                format!("B={batch} lane {lane} did not select target K={tokens}"),
            ));
        }
        for (row, step) in event.steps().enumerate() {
            let source_row = lane * tokens + row;
            let begin = source_row * Qwen38_27B::VOCAB;
            let expected = independent_bf16_argmax(&logits[begin..begin + Qwen38_27B::VOCAB])?;
            if step.token_id != expected {
                return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                    format!(
                        "B={batch} K={tokens} lane {lane} output {row} selected {}, independent target argmax is {expected}",
                        step.token_id
                    ),
                ));
            }
            checked += 1;
        }
    }
    let active = generator.active_request_ids().collect::<Vec<_>>();
    for request in active {
        let _ = generator.cancel(request)?;
    }
    if generator.active_requests() != 0 {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!("B={batch} K={tokens} left active requests after cleanup"),
        ));
    }
    Ok(checked)
}

fn qualify_reuse_cancellation_and_recycling(
    generator: &mut ResidentMtpBatchGenerator,
) -> Result<(usize, usize, usize), ResidentMtpBatchGenerationQualificationError> {
    let request = greedy_request("Give a brief greeting.", 8);
    let a = generator.admit(&request)?;
    let b = generator.admit(&request)?;
    let c = generator.admit(&request)?;
    require_slot(generator, a.request_id, 0)?;
    require_slot(generator, b.request_id, 1)?;
    require_slot(generator, c.request_id, 2)?;
    let anchors = generator.step()?;
    let b_anchor = one_token(&anchors, b.request_id)?;
    let cancelled = generator.cancel(b.request_id)?;
    let boundary_tokens = cancelled.output.prompt.message_boundary_tokens;
    if boundary_tokens >= b.prompt_tokens
        || cancelled.device_retained_tokens != boundary_tokens
        || generator.qualification_retained_tokens(1) != Some(boundary_tokens)
        || !generator.qualification_message_boundary_matches(1)?
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "middle-slot cancellation did not restore every message-boundary seam".to_string(),
        ));
    }

    let reused = generator.admit(&request)?;
    if reused.device_reused_tokens != boundary_tokens {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "identical prompt did not restore the cancelled message boundary".to_string(),
        ));
    }
    require_slot(generator, reused.request_id, 1)?;
    let joined = generator.step()?;
    if one_token(&joined, reused.request_id)? != b_anchor {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "hidden-boundary restoration changed the next anchor".to_string(),
        ));
    }
    if generator.active_request_ids().collect::<Vec<_>>()
        != [a.request_id, c.request_id, reused.request_id]
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "middle-slot reuse changed compact survivor order".to_string(),
        ));
    }

    let reused_cancelled = generator.cancel(reused.request_id)?;
    if reused_cancelled.device_retained_tokens != boundary_tokens {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "reused prompt did not retain the same message boundary".to_string(),
        ));
    }
    let followup = followup_request(8);
    let followup_admission = generator.admit(&followup)?;
    if followup_admission.device_reused_tokens != boundary_tokens {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "a divergent next turn did not reuse the cancelled message boundary".to_string(),
        ));
    }
    require_slot(generator, followup_admission.request_id, 1)?;
    let _ = generator.cancel(followup_admission.request_id)?;

    for request in [a.request_id, c.request_id] {
        let _ = generator.cancel(request)?;
    }
    generator.qualification_clear_retained()?;

    let longer = greedy_request("Give a brief greeting and name one color.", 8);
    let cold = generator.admit(&longer)?;
    let _ = generator.step()?;
    let _ = generator.step()?;
    let _ = generator.cancel(cold.request_id)?;
    let original = generator.admit(&request)?;
    if original.device_reused_tokens != 0 {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "a prompt ending before the retained span reused stale MTP state".to_string(),
        ));
    }
    let _ = generator.cancel(original.request_id)?;
    Ok((7, 2, 1))
}

fn qualify_page_pressure_eviction(
    generator: &mut ResidentMtpBatchGenerator,
) -> Result<(usize, usize), ResidentMtpBatchGenerationQualificationError> {
    let displaced_request = greedy_request("Describe the color amber briefly.", 1);
    let displaced = generator.admit(&displaced_request)?;
    let displaced_slot = generator
        .qualification_slot(displaced.request_id)
        .expect("admitted request owns a slot");

    let selected_request = greedy_request("Name one primary color.", 1);
    let selected = generator.admit(&selected_request)?;
    let selected_slot = generator
        .qualification_slot(selected.request_id)
        .expect("admitted request owns a slot");
    let selected_retained = generator.cancel(selected.request_id)?;
    let maximum = generator
        .context_capacity()
        .checked_sub(selected.prompt_tokens)
        .filter(|&tokens| tokens != 0)
        .ok_or_else(|| {
            ResidentMtpBatchGenerationQualificationError::Mismatch(
                "page-pressure fixture prompt fills the resident context".to_string(),
            )
        })?;

    let full_request = greedy_request("Name one primary color.", maximum);
    let error = match generator.admit(&full_request) {
        Ok(_) => {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                "full-pool admission ignored pages owned by an active request".to_string(),
            ));
        }
        Err(error) => error,
    };
    if error.code() != Some(EngineErrorCode::Capacity)
        || generator.qualification_retained_tokens(selected_slot)
            != Some(selected_retained.device_retained_tokens)
        || generator.qualification_slot(displaced.request_id) != Some(displaced_slot)
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "active page pressure did not refuse admission without changing slot state".to_string(),
        ));
    }

    let displaced_retained = generator.cancel(displaced.request_id)?;
    if generator.qualification_retained_tokens(displaced_slot)
        != Some(displaced_retained.device_retained_tokens)
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "page-pressure fixture did not retain its displacement candidate".to_string(),
        ));
    }
    let admitted = generator.admit(&full_request)?;
    if admitted.device_reused_tokens != selected_retained.device_retained_tokens
        || generator.qualification_slot(admitted.request_id) != Some(selected_slot)
        || generator
            .qualification_retained_tokens(displaced_slot)
            .is_some()
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "full-pool admission did not preserve its prefix and evict the LRU inactive owner"
                .to_string(),
        ));
    }
    let _ = generator.cancel(admitted.request_id)?;
    Ok((1, 1))
}

fn qualify_sampled_batch(
    generator: &mut ResidentMtpBatchGenerator,
) -> Result<usize, ResidentMtpBatchGenerationQualificationError> {
    let request = sampled_request("Write a short salutation.", 7, 0x5a17);
    let admissions = [
        generator.admit(&request)?,
        generator.admit(&request)?,
        generator.admit(&request)?,
    ];
    let mut deltas = [String::new(), String::new(), String::new()];
    let mut outputs: [Option<GeneratedText>; 3] = std::array::from_fn(|_| None);
    while generator.active_requests() != 0 {
        let events = generator.step()?;
        for event in events.iter() {
            let lane = admissions
                .iter()
                .position(|admission| admission.request_id == event.request_id)
                .ok_or_else(|| {
                    ResidentMtpBatchGenerationQualificationError::Mismatch(
                        "sampled batch returned an unknown request".to_string(),
                    )
                })?;
            for step in event.steps() {
                if let Some(delta) = &step.delta {
                    deltas[lane].push_str(delta);
                }
            }
            if let Some(output) = &event.completed {
                outputs[lane] = Some(output.clone());
            }
        }
    }
    for lane in 0..3 {
        let output = outputs[lane].as_ref().ok_or_else(|| {
            ResidentMtpBatchGenerationQualificationError::Mismatch(format!(
                "sampled lane {lane} did not complete"
            ))
        })?;
        if deltas[lane] != output.text || output.token_ids.len() != 7 {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                format!("sampled lane {lane} changed its streaming or length boundary"),
            ));
        }
    }
    let expected = outputs[0].as_ref().expect("sampled lane zero completed");
    for (lane, actual) in outputs.iter().enumerate().skip(1) {
        let actual = actual.as_ref().expect("sampled lane completed");
        if actual.prompt.rendered_bytes != expected.prompt.rendered_bytes
            || actual.prompt.token_ids != expected.prompt.token_ids
            || actual.token_ids != expected.token_ids
            || actual.text != expected.text
            || actual.finish_reason != expected.finish_reason
        {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                format!(
                    "identical sampled lane {lane} diverged: {:?} versus {:?}",
                    actual.token_ids, expected.token_ids
                ),
            ));
        }
    }
    Ok(3)
}

fn qualify_greedy_batch_invariance(
    generator: &mut ResidentMtpBatchGenerator,
) -> Result<usize, ResidentMtpBatchGenerationQualificationError> {
    let request = greedy_request("Reply with exactly the word blue.", 8);
    let mut expected: Option<GeneratedText> = None;
    let mut checked = 0;

    for batch in EXACT_BATCHES {
        let mut admissions = Vec::with_capacity(batch);
        for _ in 0..batch {
            admissions.push(generator.admit(&request)?);
        }
        let mut streamed_tokens: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        let mut streamed_text: [String; 8] = std::array::from_fn(|_| String::new());
        let mut outputs: [Option<GeneratedText>; 8] = std::array::from_fn(|_| None);

        while generator.active_requests() != 0 {
            let events = generator.step()?;
            for event in events.iter() {
                let lane = admissions
                    .iter()
                    .position(|admission| admission.request_id == event.request_id)
                    .ok_or_else(|| {
                        ResidentMtpBatchGenerationQualificationError::Mismatch(format!(
                            "B={batch} greedy transaction returned an unknown request"
                        ))
                    })?;
                for step in event.steps() {
                    streamed_tokens[lane].push(step.token_id);
                    if let Some(delta) = &step.delta {
                        streamed_text[lane].push_str(delta);
                    }
                }
                if let Some(output) = &event.completed {
                    outputs[lane] = Some(output.clone());
                }
            }
        }

        for lane in 0..batch {
            let actual = outputs[lane].as_ref().ok_or_else(|| {
                ResidentMtpBatchGenerationQualificationError::Mismatch(format!(
                    "B={batch} greedy lane {lane} did not complete"
                ))
            })?;
            if streamed_tokens[lane] != actual.token_ids || streamed_text[lane] != actual.text {
                return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                    format!("B={batch} greedy lane {lane} changed its streaming boundary"),
                ));
            }
            if let Some(expected) = &expected {
                if actual.prompt.rendered_bytes != expected.prompt.rendered_bytes
                    || actual.prompt.token_ids != expected.prompt.token_ids
                    || actual.token_ids != expected.token_ids
                    || actual.text != expected.text
                    || actual.finish_reason != expected.finish_reason
                {
                    return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                        format!(
                            "B={batch} greedy lane {lane} diverged from B=1: {:?} versus {:?}",
                            actual.token_ids, expected.token_ids
                        ),
                    ));
                }
            } else {
                expected = Some(actual.clone());
            }
            checked += 1;
        }
        generator.qualification_clear_retained()?;
    }

    Ok(checked)
}

fn one_token(
    events: &tuisko_engine::ResidentMtpBatchEvents,
    request: ResidentRequestId,
) -> Result<u32, ResidentMtpBatchGenerationQualificationError> {
    let event = events
        .iter()
        .find(|event| event.request_id == request)
        .ok_or_else(|| {
            ResidentMtpBatchGenerationQualificationError::Mismatch(format!(
                "request {} produced no event",
                request.get()
            ))
        })?;
    if event.len() != 1 {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!(
                "request {} produced {} tokens, expected one anchor",
                request.get(),
                event.len()
            ),
        ));
    }
    Ok(event.steps().next().expect("one step exists").token_id)
}

fn require_slot(
    generator: &ResidentMtpBatchGenerator,
    request: ResidentRequestId,
    expected: usize,
) -> Result<(), ResidentMtpBatchGenerationQualificationError> {
    if generator.qualification_slot(request) != Some(expected) {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!("request {} did not own slot {expected}", request.get()),
        ));
    }
    Ok(())
}

fn independent_bf16_argmax(
    logits: &[u16],
) -> Result<u32, ResidentMtpBatchGenerationQualificationError> {
    if logits.len() != Qwen38_27B::VOCAB {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "independent argmax received a partial vocabulary row".to_string(),
        ));
    }
    let mut best = 0usize;
    let mut best_value = f32::from_bits(u32::from(logits[0]) << 16);
    if !best_value.is_finite() {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "target logits contain a non-finite first value".to_string(),
        ));
    }
    for (token, &word) in logits.iter().enumerate().skip(1) {
        let value = f32::from_bits(u32::from(word) << 16);
        if !value.is_finite() {
            return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
                format!("target logit {token} is non-finite"),
            ));
        }
        if value > best_value {
            best = token;
            best_value = value;
        }
    }
    u32::try_from(best).map_err(|_| {
        ResidentMtpBatchGenerationQualificationError::Mismatch(
            "independent argmax exceeds u32".to_string(),
        )
    })
}

fn greedy_request(content: &str, maximum: usize) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", content)]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = maximum;
    request
}

fn sampled_request(content: &str, maximum: usize, seed: u64) -> ChatGenerationRequest {
    let mut request = greedy_request(content, maximum);
    request.sampling = SamplingOptions {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 20,
        seed,
        ..SamplingOptions::default()
    };
    request
}

fn followup_request(maximum: usize) -> ChatGenerationRequest {
    let mut request = greedy_request("Give a brief greeting.", maximum);
    request
        .messages
        .push(ChatMessage::new("user", "Now name one primary color."));
    request
}

fn verify_owner(
    generator: &ResidentMtpBatchGenerator,
) -> Result<(), ResidentMtpBatchGenerationQualificationError> {
    if generator.device_owner_bytes() != 30_342_618_624
        || generator.host_stager_bytes() != 1_281_019_904
        || generator.message_boundary_snapshot_bytes() != 1_231_634_432
        || generator.kv_route_host_bytes() != 113_454
        || generator.context_capacity() != 220_000
    {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            format!(
                "compact MTP owner accounting changed: device={}, host={}, boundary={}, routes={}, capacity={}",
                generator.device_owner_bytes(),
                generator.host_stager_bytes(),
                generator.message_boundary_snapshot_bytes(),
                generator.kv_route_host_bytes(),
                generator.context_capacity()
            ),
        ));
    }
    let addresses = generator.qualification_addresses()?;
    let unique = addresses.iter().copied().collect::<BTreeSet<_>>();
    if addresses.contains(&0) || unique.len() != addresses.len() {
        return Err(ResidentMtpBatchGenerationQualificationError::Mismatch(
            "compact MTP owner addresses are not unique and nonzero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EXACT_BATCHES, EXACT_VERIFY_TOKENS, greedy_request, one_token,
        qualify_page_pressure_eviction,
    };
    use std::path::Path;
    use std::sync::Arc;
    use tuisko_engine::ResidentMtpBatchGenerator;
    use tuisko_gpu::{CudaContext, device_memory_info};
    use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

    #[test]
    fn resident_mtp_batch_route_inventory_is_exact() {
        assert_eq!(
            EXACT_BATCHES.collect::<Vec<_>>(),
            (1..=8).collect::<Vec<_>>()
        );
        assert_eq!(
            EXACT_VERIFY_TOKENS.collect::<Vec<_>>(),
            (1..=4).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_batch_suite_covers_routes_oracles_and_slot_lifecycle() {
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let report = super::qualify_resident_mtp_batch_generation(Path::new(&root)).unwrap();
        assert_eq!(report.verification_routes, 32);
        assert_eq!(report.draft_batches, 8);
        assert!(report.oracle_rows >= 32);
        assert_eq!(report.cancellations, 7);
        assert_eq!(report.exact_prefix_reuses, 2);
        assert_eq!(report.message_boundary_fallbacks, 1);
        assert_eq!(report.safe_cold_fallbacks, 1);
        assert_eq!(report.page_pressure_evictions, 1);
        assert_eq!(report.page_pressure_refusals, 1);
        assert_eq!(report.sampled_lanes, 3);
        assert_eq!(report.greedy_invariant_lanes, 36);
        assert_eq!(report.device_owner_bytes, 30_342_618_624);
        assert_eq!(report.host_stager_bytes, 1_281_019_904);
        assert_eq!(report.message_boundary_snapshot_bytes, 1_231_634_432);
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_batch_suite_shared_prefix_scoring_matches_independent_scoring() {
        let _preflight = crate::device_benchmark::preflight().unwrap();
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let snapshot = Arc::new(
            CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&root))
                .expect("snapshot must be admitted"),
        );
        let context = CudaContext::new(0).unwrap();
        let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot).unwrap();

        for common in [31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025] {
            let prefix = (0..common)
                .map(|position| 100 + u32::try_from(position % 1_000).unwrap())
                .collect::<Vec<_>>();
            let mut one_token_suffix = prefix.clone();
            one_token_suffix.push(2_001);
            let mut unequal_suffix = prefix.clone();
            unequal_suffix.extend([2_002, 2_003, 2_004]);
            let prompts = vec![prefix.clone(), prefix, one_token_suffix, unequal_suffix];

            let shared = generator.score_prompts(&prompts).unwrap();
            let independent = prompts
                .iter()
                .map(|prompt| generator.score_prompt(prompt).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                shared, independent,
                "shared-prefix scoring differed at common length {common}"
            );
        }
        crate::device_benchmark::require_current_process_exclusive().unwrap();
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_batch_suite_reclaims_inactive_pages_before_refusing_capacity() {
        let _preflight = crate::device_benchmark::preflight().unwrap();
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let snapshot = Arc::new(
            CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&root))
                .expect("snapshot must be admitted"),
        );
        let context = CudaContext::new(0).unwrap();
        let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot).unwrap();

        assert_eq!(
            qualify_page_pressure_eviction(&mut generator).unwrap(),
            (1, 1)
        );
        generator.qualification_clear_retained().unwrap();
        crate::device_benchmark::require_current_process_exclusive().unwrap();
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_batch_suite_park_releases_all_arenas_and_restores_prefix() {
        let _preflight = crate::device_benchmark::preflight().unwrap();
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let snapshot = Arc::new(
            CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&root))
                .expect("snapshot must be admitted"),
        );
        let context = CudaContext::new(0).unwrap();
        let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot).unwrap();
        let request = greedy_request("Give a brief greeting.", 8);

        let first = generator.admit(&request).unwrap();
        let first_events = generator.step().unwrap();
        let _ = one_token(&first_events, first.request_id).unwrap();
        let cancelled = generator.cancel(first.request_id).unwrap();
        let retained_tokens = cancelled.device_retained_tokens;
        assert_ne!(retained_tokens, 0);

        let replay = generator.admit(&request).unwrap();
        assert_eq!(replay.device_reused_tokens, retained_tokens);
        let replay_events = generator.step().unwrap();
        let expected_anchor = one_token(&replay_events, replay.request_id).unwrap();
        let replay_cancelled = generator.cancel(replay.request_id).unwrap();
        assert_eq!(replay_cancelled.device_retained_tokens, retained_tokens);

        let loaded_bytes = generator.device_owner_bytes();
        let addresses = generator.qualification_addresses().unwrap();
        let before_park = device_memory_info(&context).unwrap();
        let (parked, stats) = match generator.park() {
            Ok(parked) => parked,
            Err((_, error)) => panic!("park failed: {error}"),
        };
        let after_park = device_memory_info(&context).unwrap();
        assert_eq!(stats.released_device_bytes, loaded_bytes);
        assert_eq!(parked.remaining_device_bytes(), 0);
        assert!(stats.host_bytes > 0);
        assert!(after_park.free_bytes >= before_park.free_bytes + loaded_bytes);
        crate::device_benchmark::require_current_process_exclusive().unwrap();

        let mut generator = match parked.resume() {
            Ok(generator) => generator,
            Err((_, error)) => panic!("resume failed: {error}"),
        };
        assert_eq!(generator.device_owner_bytes(), loaded_bytes);
        assert_eq!(generator.qualification_addresses().unwrap(), addresses);
        let restored = generator.admit(&request).unwrap();
        assert_eq!(restored.device_reused_tokens, retained_tokens);
        let restored_events = generator.step().unwrap();
        assert_eq!(
            one_token(&restored_events, restored.request_id).unwrap(),
            expected_anchor
        );
        let _ = generator.cancel(restored.request_id).unwrap();
        generator.qualification_clear_retained().unwrap();
        crate::device_benchmark::require_current_process_exclusive().unwrap();
    }
}

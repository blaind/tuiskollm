//! Source-backed integration gate for compact multi-request resident generation.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, GeneratedText, ResidentBatchGenerator, ResidentRequestId,
    SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen38_27B};

const NATIVE_PREFILL_ROUTES: [usize; 4] = [32, 64, 128, 1024];

/// Failure of the compact resident-generation integration gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentBatchGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// Tokenizer or streaming decode failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An externally visible scheduler boundary differed.
    #[error("resident-batch-generation qualification failed: {0}")]
    Mismatch(String),
}

/// Compact scheduling, recycling, and ownership boundaries checked by the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentBatchGenerationQualification {
    /// Independent sequential request outputs compared with compact scheduling.
    pub requests: usize,
    /// Compact scheduler rounds exercised.
    pub rounds: usize,
    /// Exact pending replay batches exercised across B=1..8.
    pub route_batches: usize,
    /// Exact from-empty prefill routes exercised through batch admission.
    pub native_prefill_routes: usize,
    /// Physical hole recycled while surviving requests remained active.
    pub recycled_slot: usize,
    /// Active cancellation boundaries exercised.
    pub cancellations: usize,
    /// Complete retained prefixes restored without device replay.
    pub exact_prefix_reuses: usize,
    /// Divergent retained spans correctly rejected for device reuse.
    pub safe_cold_fallbacks: usize,
    /// Exact device arena bytes shared by every request.
    pub arena_bytes: usize,
    /// Exact page-locked embedding and double-logit-bank bytes.
    pub host_stager_bytes: usize,
    /// Exact allocation-free host page-routing bytes.
    pub kv_route_host_bytes: usize,
}

/// Qualifies mixed-length compact scheduling against independent sequential execution.
pub fn qualify_resident_batch_generation(
    root: &Path,
) -> Result<ResidentBatchGenerationQualification, ResidentBatchGenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let oracle_frontend = TextFrontend::open(snapshot.as_ref())?;
    let native_requests = NATIVE_PREFILL_ROUTES
        .into_iter()
        .map(|tokens| exact_prompt_request(&oracle_frontend, tokens))
        .collect::<Result<Vec<_>, _>>()?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            format!(
                "device zero has compute capability {}.{}, expected 12.0",
                capability.0, capability.1
            ),
        ));
    }
    let mut generator = ResidentBatchGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();
    let requests = [
        greedy_request("Hello", 2),
        greedy_request("Name one color.", 1),
        greedy_request("Reply with one word.", 2),
        greedy_request("Name one color.", 2),
    ];
    let mut expected = Vec::with_capacity(requests.len());
    for request in &requests {
        expected.push(run_alone(&mut generator, request)?);
    }
    verify_native_prefill_inventory(&mut generator, &native_requests)?;
    generator.qualification_clear_retained()?;
    verify_exact_batch_inventory(&mut generator, &requests[0], &expected[0])?;
    generator.qualification_clear_retained()?;
    let before = device_memory_info(generator.context())?;

    let a = generator.admit(&requests[0])?.request_id;
    let b_admission = generator.admit(&requests[1])?;
    let b = b_admission.request_id;
    let c = generator.admit(&requests[2])?.request_id;
    require_slot(&generator, a, 0)?;
    require_slot(&generator, b, 1)?;
    require_slot(&generator, c, 2)?;
    let first = generator.step()?;
    require_round(
        &first,
        &[
            (a, &expected[0], 0),
            (b, &expected[1], 0),
            (c, &expected[2], 0),
        ],
    )?;
    if generator.active_requests() != 2 || generator.qualification_slot(b).is_some() {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "one-token request did not leave a recyclable middle slot".to_string(),
        ));
    }

    let d_admission = generator.admit(&requests[3])?;
    if d_admission.device_reused_tokens != b_admission.prompt_tokens {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "recycled middle slot did not restore its complete retained prompt".to_string(),
        ));
    }
    let d = d_admission.request_id;
    require_slot(&generator, d, 1)?;
    let second = generator.step()?;
    require_round(
        &second,
        &[
            (a, &expected[0], 1),
            (c, &expected[2], 1),
            (d, &expected[3], 0),
        ],
    )?;
    if generator.active_request_ids().collect::<Vec<_>>() != [d] {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "terminal survivors did not compact to the recycled request".to_string(),
        ));
    }
    let third = generator.step()?;
    require_round(&third, &[(d, &expected[3], 1)])?;
    if generator.active_requests() != 0 {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "completed compact schedule retained an active request".to_string(),
        ));
    }
    generator.qualification_clear_retained()?;
    verify_prefix_reuse_and_cancellation(
        &mut generator,
        &oracle_frontend,
        &requests[0],
        &expected[0],
        &requests[2],
        &expected[2],
    )?;

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            format!(
                "device memory changed after scheduler warmup: before={before:?}, after={after:?}"
            ),
        ));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "compact scheduler owner addresses changed".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(ResidentBatchGenerationQualification {
        requests: requests.len(),
        rounds: 3,
        route_batches: 8,
        native_prefill_routes: NATIVE_PREFILL_ROUTES.len(),
        recycled_slot: 1,
        cancellations: 2,
        exact_prefix_reuses: 1,
        safe_cold_fallbacks: 1,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        kv_route_host_bytes: generator.kv_route_host_bytes(),
    })
}

fn exact_prompt_request(
    frontend: &TextFrontend,
    target_tokens: usize,
) -> Result<ChatGenerationRequest, ResidentBatchGenerationQualificationError> {
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
    let request = greedy_request(&vec!["x"; lower].join(" "), 1);
    let actual = frontend
        .encode_chat(&request.messages, &request.template)?
        .len();
    if actual != target_tokens {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            format!(
                "could not construct exact T={target_tokens} rendered prompt; nearest deterministic fixture has T={actual}"
            ),
        ));
    }
    Ok(request)
}

fn verify_native_prefill_inventory(
    generator: &mut ResidentBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> Result<(), ResidentBatchGenerationQualificationError> {
    for (&tokens, request) in NATIVE_PREFILL_ROUTES.iter().zip(requests) {
        let admission = generator.admit(request)?;
        if admission.prompt_tokens != tokens
            || admission.device_reused_tokens != 0
            || admission.native_prefill_tokens != tokens
        {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!(
                    "T={tokens} batch admission reported prompt={}, reused={}, native={}",
                    admission.prompt_tokens,
                    admission.device_reused_tokens,
                    admission.native_prefill_tokens
                ),
            ));
        }
        let events = generator.step()?;
        let event = events.iter().next().ok_or_else(|| {
            ResidentBatchGenerationQualificationError::Mismatch(format!(
                "T={tokens} native prefill produced no generation event"
            ))
        })?;
        if events.len() != 1
            || event.request_id != admission.request_id
            || event.completed.is_none()
        {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!("T={tokens} native prefill changed its terminal scheduler seam"),
            ));
        }
        let native_output = event
            .completed
            .as_ref()
            .expect("native terminal output was checked")
            .clone();

        let reused = generator.admit(request)?;
        if reused.prompt_tokens != tokens
            || reused.device_reused_tokens != tokens
            || reused.native_prefill_tokens != 0
        {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!(
                    "T={tokens} retained admission reported prompt={}, reused={}, native={}",
                    reused.prompt_tokens, reused.device_reused_tokens, reused.native_prefill_tokens
                ),
            ));
        }
        let reused_events = generator.step()?;
        let reused_event = reused_events.iter().next().ok_or_else(|| {
            ResidentBatchGenerationQualificationError::Mismatch(format!(
                "T={tokens} retained admission produced no generation event"
            ))
        })?;
        let reused_output = reused_event.completed.as_ref();
        if reused_events.len() != 1
            || reused_event.request_id != reused.request_id
            || reused_output.is_none_or(|output| {
                output.prompt.token_ids != native_output.prompt.token_ids
                    || output.token_ids != native_output.token_ids
                    || output.text != native_output.text
                    || output.finish_reason != native_output.finish_reason
            })
        {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!("T={tokens} retained admission changed its terminal scheduler seam"),
            ));
        }
    }
    Ok(())
}

fn verify_prefix_reuse_and_cancellation(
    generator: &mut ResidentBatchGenerator,
    frontend: &TextFrontend,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
    survivor_request: &ChatGenerationRequest,
    survivor_expected: &GeneratedText,
) -> Result<(), ResidentBatchGenerationQualificationError> {
    let cold = generator.admit(request)?;
    let survivor = generator.admit(survivor_request)?;
    if cold.device_reused_tokens != 0 || survivor.device_reused_tokens != 0 {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "cold cancellation fixture unexpectedly reused device state".to_string(),
        ));
    }
    require_slot(generator, cold.request_id, 0)?;
    require_slot(generator, survivor.request_id, 1)?;
    let first = generator.step()?;
    require_round(
        &first,
        &[
            (cold.request_id, expected, 0),
            (survivor.request_id, survivor_expected, 0),
        ],
    )?;

    let cancelled = generator.cancel(cold.request_id)?;
    let expected_cancelled_text = frontend.decode(&expected.token_ids[..1], true)?;
    if cancelled.request_id != cold.request_id
        || cancelled.device_retained_tokens != cold.prompt_tokens
        || cancelled.output.prompt.token_ids != expected.prompt.token_ids
        || cancelled.output.token_ids != expected.token_ids[..1]
        || cancelled.output.text != expected_cancelled_text
        || generator.qualification_retained_tokens(0) != Some(cold.prompt_tokens)
        || generator.active_request_ids().collect::<Vec<_>>() != [survivor.request_id]
    {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "cancellation did not preserve its exact host output, processed device span, or survivor order"
                .to_string(),
        ));
    }

    let reused = generator.admit(request)?;
    if reused.device_reused_tokens != cold.prompt_tokens {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "identical prompt did not restore its complete retained device prefix".to_string(),
        ));
    }
    require_slot(generator, reused.request_id, 0)?;
    let joined = generator.step()?;
    require_round(
        &joined,
        &[
            (survivor.request_id, survivor_expected, 1),
            (reused.request_id, expected, 0),
        ],
    )?;
    let terminal = generator.step()?;
    require_round(&terminal, &[(reused.request_id, expected, 1)])?;
    if generator.active_requests() != 0
        || generator.qualification_retained_tokens(0) != Some(cold.prompt_tokens + 1)
    {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "reused request did not retain exactly the token span processed by the device"
                .to_string(),
        ));
    }

    // The retained state now includes one generated token. The original prompt is
    // shorter, so treating it as a reusable prefix would silently cross divergence.
    let divergent = generator.admit(request)?;
    if divergent.device_reused_tokens != 0 {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "a prompt that diverged before the full retained span reused device state".to_string(),
        ));
    }
    let divergent_first = generator.step()?;
    require_round(&divergent_first, &[(divergent.request_id, expected, 0)])?;
    let second_cancel = generator.cancel(divergent.request_id)?;
    if second_cancel.device_retained_tokens != divergent.prompt_tokens
        || second_cancel.output.token_ids != expected.token_ids[..1]
        || generator.active_requests() != 0
    {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "cold fallback cancellation changed its exact processed boundary".to_string(),
        ));
    }
    Ok(())
}

fn verify_exact_batch_inventory(
    generator: &mut ResidentBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
) -> Result<(), ResidentBatchGenerationQualificationError> {
    if expected.token_ids.len() != 2 {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "two-token route oracle terminated before the exact batch sweep".to_string(),
        ));
    }
    for batch in 1..=8 {
        let mut request_ids = [None; 8];
        for request_id in &mut request_ids[..batch] {
            *request_id = Some(generator.admit(request)?.request_id);
        }
        let first = generator.step()?;
        let second = generator.step()?;
        if first.len() != batch || second.len() != batch {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!("B={batch} did not preserve its compact event inventory"),
            ));
        }
        for (index, (first, second)) in first.iter().zip(second.iter()).enumerate() {
            let request_id = request_ids[index].expect("exact batch request ID exists");
            if first.request_id != request_id
                || second.request_id != request_id
                || first.step.token_id != expected.token_ids[0]
                || second.step.token_id != expected.token_ids[1]
                || first.completed.is_some()
            {
                return Err(ResidentBatchGenerationQualificationError::Mismatch(
                    format!("B={batch} request row {index} differs from sequential execution"),
                ));
            }
            let Some(output) = &second.completed else {
                return Err(ResidentBatchGenerationQualificationError::Mismatch(
                    format!("B={batch} request row {index} did not complete"),
                ));
            };
            if output.token_ids != expected.token_ids
                || output.text != expected.text
                || output.finish_reason != expected.finish_reason
            {
                return Err(ResidentBatchGenerationQualificationError::Mismatch(
                    format!("B={batch} request row {index} output changed"),
                ));
            }
        }
        if generator.active_requests() != 0 {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!("B={batch} left completed requests active"),
            ));
        }
    }
    Ok(())
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

fn run_alone(
    generator: &mut ResidentBatchGenerator,
    request: &ChatGenerationRequest,
) -> Result<GeneratedText, ResidentBatchGenerationQualificationError> {
    let admission = generator.admit(request)?;
    if let Some(output) = admission.completed {
        return Ok(output);
    }
    loop {
        let events = generator.step()?;
        if events.len() != 1 {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                "sequential scheduler route produced the wrong event count".to_string(),
            ));
        }
        let event = events.iter().next().expect("one sequential event");
        if let Some(output) = &event.completed {
            return Ok(output.clone());
        }
    }
}

fn require_slot(
    generator: &ResidentBatchGenerator,
    request: ResidentRequestId,
    expected: usize,
) -> Result<(), ResidentBatchGenerationQualificationError> {
    if generator.qualification_slot(request) != Some(expected) {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            format!(
                "request {} did not own physical slot {expected}",
                request.get()
            ),
        ));
    }
    Ok(())
}

fn require_round(
    events: &tuisko_engine::ResidentBatchEvents,
    expected: &[(ResidentRequestId, &GeneratedText, usize)],
) -> Result<(), ResidentBatchGenerationQualificationError> {
    if events.len() != expected.len() {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            format!(
                "scheduler round returned {} events, expected {}",
                events.len(),
                expected.len()
            ),
        ));
    }
    for (event, &(request, output, token_index)) in events.iter().zip(expected) {
        let token = output.token_ids.get(token_index).ok_or_else(|| {
            ResidentBatchGenerationQualificationError::Mismatch(
                "sequential oracle has too few output tokens".to_string(),
            )
        })?;
        if event.request_id != request || event.step.token_id != *token {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!(
                    "request {} differs from its sequential token {token_index}",
                    request.get()
                ),
            ));
        }
        let terminal = token_index + 1 == output.token_ids.len();
        if terminal {
            let Some(actual) = &event.completed else {
                return Err(ResidentBatchGenerationQualificationError::Mismatch(
                    format!(
                        "request {} did not complete at its oracle boundary",
                        request.get()
                    ),
                ));
            };
            if actual.prompt.token_ids != output.prompt.token_ids
                || actual.token_ids != output.token_ids
                || actual.text != output.text
                || actual.finish_reason != output.finish_reason
            {
                return Err(ResidentBatchGenerationQualificationError::Mismatch(
                    format!(
                        "request {} output differs from sequential execution",
                        request.get()
                    ),
                ));
            }
        } else if event.completed.is_some() {
            return Err(ResidentBatchGenerationQualificationError::Mismatch(
                format!(
                    "request {} completed before its sequential boundary",
                    request.get()
                ),
            ));
        }
    }
    Ok(())
}

fn verify_owner(
    generator: &ResidentBatchGenerator,
) -> Result<(), ResidentBatchGenerationQualificationError> {
    if generator.arena_bytes() != 28_380_566_016
        || generator.host_stager_bytes() != 18_432_000
        || generator.kv_route_host_bytes() != 113_454
    {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "compact scheduler owner byte accounting changed".to_string(),
        ));
    }
    let addresses = generator.qualification_addresses();
    if addresses.contains(&0)
        || addresses[0] == addresses[1]
        || addresses[0] == addresses[2]
        || addresses[1] == addresses[2]
    {
        return Err(ResidentBatchGenerationQualificationError::Mismatch(
            "compact scheduler owner addresses are invalid".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::qualify_resident_batch_generation;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn compact_scheduler_matches_sequential_requests_and_recycles_holes()
    -> Result<(), super::ResidentBatchGenerationQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            super::ResidentBatchGenerationQualificationError::Mismatch(
                "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_resident_batch_generation(&PathBuf::from(root))?;
        assert_eq!(report.requests, 4);
        assert_eq!(report.rounds, 3);
        assert_eq!(report.route_batches, 8);
        assert_eq!(report.native_prefill_routes, 4);
        assert_eq!(report.recycled_slot, 1);
        assert_eq!(report.cancellations, 2);
        assert_eq!(report.exact_prefix_reuses, 1);
        assert_eq!(report.safe_cold_fallbacks, 1);
        assert_eq!(report.arena_bytes, 28_380_566_016);
        assert_eq!(report.host_stager_bytes, 18_432_000);
        Ok(())
    }
}

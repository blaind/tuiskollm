//! Source-backed integration gate for the Qwen3.5 frontend and resident text graph.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, GeneratedText, Qwen35ResidentBatchGenerator,
    Qwen35ResidentTextGenerator, ResidentBatchEvents, ResidentRequestId, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{Arch, CheckpointError, CheckpointSnapshot, Qwen35_9B};

// Transformers 5.2.0 `apply_chat_template` and tokenizer output from the pinned snapshot.
const HELLO_THINKING: [u32; 11] = [
    248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
];
const HELLO_NO_THINKING: [u32; 13] = [
    248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 271, 248_069, 271,
];
const PREFILL_ROUTE_CASES: [(usize, usize); 7] = [
    (31, 0),
    (32, 32),
    (33, 32),
    (64, 64),
    (65, 64),
    (128, 128),
    (129, 128),
];

/// Failure of the concrete Qwen3.5 generation integration gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35GenerationQualificationError {
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
    /// An externally visible generation boundary differed.
    #[error("Qwen3.5 generation qualification failed: {0}")]
    Mismatch(String),
}

/// Frontend and streaming boundaries checked by the integration gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GenerationQualification {
    /// Independent Transformers prompt fixtures checked.
    pub prompt_cases: usize,
    /// Production chat steps streamed and reassembled.
    pub chat_steps: usize,
    /// Exact selected tokens from the production request.
    pub generated_tokens: Vec<u32>,
    /// Native prompt-prefix widths selected by the synthetic routing matrix.
    pub native_prefill_tokens: Vec<usize>,
    /// Exact bytes across all retained device arenas.
    pub arena_bytes: usize,
    /// Exact page-locked embedding and logit staging bytes.
    pub host_stager_bytes: usize,
    /// Number of stable retained device and host addresses.
    pub stable_addresses: usize,
}

/// Exact compact routes and lifecycle seams checked for Qwen3.5 serving.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35CompactGenerationQualification {
    /// Exact pending decode batches exercised from B=1 through B=8.
    pub route_batches: usize,
    /// Concurrent requests compared with sequential generation.
    pub requests: usize,
    /// Physical slot recycled while two other requests remained active.
    pub recycled_slot: usize,
    /// Native prompt tokens admitted while another request was active.
    pub concurrent_prefill_tokens: usize,
    /// Active cancellation boundaries exercised.
    pub cancellations: usize,
    /// Exact bytes across all retained device arenas.
    pub arena_bytes: usize,
    /// Exact page-locked embedding and double-logit-bank bytes.
    pub host_stager_bytes: usize,
    /// Stable retained device and host addresses.
    pub stable_addresses: usize,
}

/// Qualifies the exact single-slot Qwen3.5 frontend-to-device generation path.
pub fn qualify_qwen35_generation(
    root: &Path,
) -> Result<Qwen35GenerationQualification, Qwen35GenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let oracle_frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
    verify_prompt_fixtures(&oracle_frontend)?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let mut generator = Qwen35ResidentTextGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();

    let warm_tokens = route_tokens(128, 0);
    let (_, warm_native) = generator.qualification_greedy_after_tokens_with_route(&warm_tokens)?;
    if warm_native != 128 {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "T=128 warmup selected a {warm_native}-token native prefix"
        )));
    }
    let before = device_memory_info(generator.context())?;
    let mut native_prefill_tokens = Vec::with_capacity(PREFILL_ROUTE_CASES.len());
    for (case, (tokens, expected)) in PREFILL_ROUTE_CASES.into_iter().enumerate() {
        let (_, actual) = generator
            .qualification_greedy_after_tokens_with_route(&route_tokens(tokens, case + 1))?;
        if actual != expected {
            return Err(Qwen35GenerationQualificationError::Mismatch(format!(
                "{tokens}-token prompt selected native prefix {actual}, expected {expected}"
            )));
        }
        native_prefill_tokens.push(actual);
    }
    let first = generator.qualification_greedy_after_tokens(&HELLO_THINKING)?;
    let repeated = generator.qualification_greedy_after_tokens(&HELLO_THINKING)?;
    if first != repeated {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "reset replay selected {first} then {repeated} for the same prompt"
        )));
    }

    let messages = vec![ChatMessage::new("user", "Hello")];
    let mut request = ChatGenerationRequest::new(messages);
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = 2;
    let stop_ids = oracle_frontend.stop_ids();
    let mut expected_tokens = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut processed = HELLO_THINKING.to_vec();
        processed.extend_from_slice(&expected_tokens);
        let token = generator.qualification_greedy_after_tokens(&processed)?;
        expected_tokens.push(token);
        if stop_ids.contains(&token) {
            break;
        }
    }

    let mut session = generator.start(&request)?;
    if session.prompt_token_ids() != HELLO_THINKING {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "generation bridge changed the Transformers prompt fixture".into(),
        ));
    }
    if session.native_prefill_tokens() != 0 {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "11-token chat prompt selected a {}-token native prefix",
            session.native_prefill_tokens()
        )));
    }
    let mut streamed = String::new();
    let mut step_tokens = Vec::new();
    while session.finish_reason().is_none() {
        let step = session.step()?;
        step_tokens.push(step.token_id);
        if let Some(delta) = step.delta {
            streamed.push_str(&delta);
        }
    }
    let output = session.into_output()?;
    let expected_reason = if expected_tokens
        .last()
        .is_some_and(|token| stop_ids.contains(token))
    {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    if step_tokens != expected_tokens
        || output.token_ids != expected_tokens
        || output.text != streamed
        || output.finish_reason != expected_reason
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "streamed output differs from independent raw-token replay".into(),
        ));
    }

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device memory changed after generation warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "resident generation owner addresses changed".into(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen35GenerationQualification {
        prompt_cases: 2,
        chat_steps: step_tokens.len(),
        generated_tokens: step_tokens,
        native_prefill_tokens,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        stable_addresses: stable_addresses.len(),
    })
}

/// Qualifies compact Qwen3.5 decode, admission, cancellation, and slot reuse.
pub fn qualify_qwen35_compact_generation(
    root: &Path,
) -> Result<Qwen35CompactGenerationQualification, Qwen35GenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let mut generator = Qwen35ResidentBatchGenerator::from_snapshot(&context, snapshot)?;
    verify_compact_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();

    let request = greedy_request("Hello", 2);
    let alternate = greedy_request("Name one color.", 2);
    let exact_prefill = exact_prompt_request(&frontend, 32)?;
    let expected = run_compact_alone(&mut generator, &request)?;
    let alternate_expected = run_compact_alone(&mut generator, &alternate)?;
    let prefill_expected = run_compact_alone(&mut generator, &exact_prefill)?;
    if expected.token_ids.len() != 2 || alternate_expected.token_ids.len() != 2 {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "compact route fixtures stopped before their two-token boundary".into(),
        ));
    }

    verify_compact_route_inventory(&mut generator, &request, &expected)?;
    let before = device_memory_info(generator.context())?;
    let recycled_slot = verify_hole_reuse(
        &mut generator,
        &request,
        &expected,
        &alternate,
        &alternate_expected,
        &exact_prefill,
        &prefill_expected,
    )?;
    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device memory changed after compact warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 compact owner addresses changed".into(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen35CompactGenerationQualification {
        route_batches: 8,
        requests: 4,
        recycled_slot,
        concurrent_prefill_tokens: 32,
        cancellations: 1,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        stable_addresses: stable_addresses.len(),
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

fn exact_prompt_request(
    frontend: &TextFrontend,
    target_tokens: usize,
) -> Result<ChatGenerationRequest, Qwen35GenerationQualificationError> {
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
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "could not construct exact T={target_tokens} prompt; deterministic fixture has T={actual}"
        )));
    }
    Ok(request)
}

fn run_compact_alone(
    generator: &mut Qwen35ResidentBatchGenerator,
    request: &ChatGenerationRequest,
) -> Result<GeneratedText, Qwen35GenerationQualificationError> {
    let admission = generator.admit(request)?;
    if let Some(output) = admission.completed {
        return Ok(output);
    }
    loop {
        let events = generator.step()?;
        if events.len() != 1 {
            return Err(Qwen35GenerationQualificationError::Mismatch(
                "sequential compact route returned the wrong event count".into(),
            ));
        }
        let event = events.iter().next().expect("one compact event");
        if let Some(output) = &event.completed {
            return Ok(output.clone());
        }
    }
}

fn verify_compact_route_inventory(
    generator: &mut Qwen35ResidentBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
) -> Result<(), Qwen35GenerationQualificationError> {
    for batch in 1..=8 {
        let mut requests = [None; 8];
        for request_id in &mut requests[..batch] {
            *request_id = Some(generator.admit(request)?.request_id);
        }
        let first = generator.step()?;
        let second = generator.step()?;
        if first.len() != batch || second.len() != batch {
            return Err(Qwen35GenerationQualificationError::Mismatch(format!(
                "B={batch} did not preserve its compact event inventory"
            )));
        }
        for (row, (first, second)) in first.iter().zip(second.iter()).enumerate() {
            let request_id = requests[row].expect("exact batch request exists");
            require_compact_event(first, request_id, expected, 0)?;
            require_compact_event(second, request_id, expected, 1)?;
        }
        if generator.active_requests() != 0 {
            return Err(Qwen35GenerationQualificationError::Mismatch(format!(
                "B={batch} left completed requests active"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_hole_reuse(
    generator: &mut Qwen35ResidentBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
    alternate: &ChatGenerationRequest,
    alternate_expected: &GeneratedText,
    prefill: &ChatGenerationRequest,
    prefill_expected: &GeneratedText,
) -> Result<usize, Qwen35GenerationQualificationError> {
    let first = generator.admit(request)?;
    let cancelled = generator.admit(alternate)?;
    let third = generator.admit(request)?;
    if generator.qualification_slot(first.request_id) != Some(0)
        || generator.qualification_slot(cancelled.request_id) != Some(1)
        || generator.qualification_slot(third.request_id) != Some(2)
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "cold compact admissions did not fill slots 0,1,2".into(),
        ));
    }
    let first_round = generator.step()?;
    require_compact_round(
        &first_round,
        &[
            (first.request_id, expected, 0),
            (cancelled.request_id, alternate_expected, 0),
            (third.request_id, expected, 0),
        ],
    )?;
    let cancellation = generator.cancel(cancelled.request_id)?;
    if cancellation.device_retained_tokens != 0
        || cancellation.output.token_ids != alternate_expected.token_ids[..1]
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 cancellation changed its host boundary or retained device state".into(),
        ));
    }

    let joined = generator.admit(prefill)?;
    if joined.native_prefill_tokens != 32
        || generator.qualification_slot(joined.request_id) != Some(1)
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "concurrent T=32 admission selected native={} slot={:?}",
            joined.native_prefill_tokens,
            generator.qualification_slot(joined.request_id)
        )));
    }
    let second_round = generator.step()?;
    require_compact_round(
        &second_round,
        &[
            (first.request_id, expected, 1),
            (third.request_id, expected, 1),
            (joined.request_id, prefill_expected, 0),
        ],
    )?;
    if generator.active_requests() != 0 {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "hole-reuse round left terminal requests active".into(),
        ));
    }

    Ok(1)
}

fn require_compact_round(
    events: &ResidentBatchEvents,
    expected: &[(ResidentRequestId, &GeneratedText, usize)],
) -> Result<(), Qwen35GenerationQualificationError> {
    if events.len() != expected.len() {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "compact round returned {} events, expected {}",
            events.len(),
            expected.len()
        )));
    }
    for (event, &(request, output, token)) in events.iter().zip(expected) {
        require_compact_event(event, request, output, token)?;
    }
    Ok(())
}

fn require_compact_event(
    event: &tuisko_engine::ResidentBatchEvent,
    request: ResidentRequestId,
    expected: &GeneratedText,
    token: usize,
) -> Result<(), Qwen35GenerationQualificationError> {
    if event.request_id != request || event.step.token_id != expected.token_ids[token] {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "request {} differs at token {token}",
            request.get()
        )));
    }
    let terminal = token + 1 == expected.token_ids.len();
    if terminal {
        if event.completed.as_ref().is_none_or(|actual| {
            actual.prompt.token_ids != expected.prompt.token_ids
                || actual.token_ids != expected.token_ids
                || actual.text != expected.text
                || actual.finish_reason != expected.finish_reason
        }) {
            return Err(Qwen35GenerationQualificationError::Mismatch(format!(
                "request {} terminal output differs from sequential execution",
                request.get()
            )));
        }
    } else if event.completed.is_some() {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "request {} completed before token {token}",
            request.get()
        )));
    }
    Ok(())
}

fn route_tokens(tokens: usize, salt: usize) -> Vec<u32> {
    (0..tokens)
        .map(|position| ((101 + salt * 31_337 + position * 65_537) % Qwen35_9B::VOCAB) as u32)
        .collect()
}

fn verify_prompt_fixtures(
    frontend: &TextFrontend,
) -> Result<(), Qwen35GenerationQualificationError> {
    let messages = [ChatMessage::new("user", "Hello")];
    let thinking = frontend.encode_chat(&messages, &ChatTemplateOptions::default())?;
    let no_thinking = frontend.encode_chat(
        &messages,
        &ChatTemplateOptions {
            enable_thinking: Some(false),
            ..ChatTemplateOptions::default()
        },
    )?;
    if thinking != HELLO_THINKING || no_thinking != HELLO_NO_THINKING {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "prompt IDs differ from Transformers: thinking={thinking:?}, no-thinking={no_thinking:?}"
        )));
    }
    Ok(())
}

fn verify_owner(
    generator: &Qwen35ResidentTextGenerator,
) -> Result<(), Qwen35GenerationQualificationError> {
    if generator.arena_bytes() != 7_039_870_976
        || generator.host_stager_bytes() != 1_610_752
        || generator.context_capacity() != 192
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 generation owner bytes or capacity changed".into(),
        ));
    }
    let addresses = generator.qualification_addresses();
    let mut unique = addresses.clone();
    unique.sort_unstable();
    unique.dedup();
    if addresses.len() != 34 || unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 generation owner addresses are invalid".into(),
        ));
    }
    Ok(())
}

fn verify_compact_owner(
    generator: &Qwen35ResidentBatchGenerator,
) -> Result<(), Qwen35GenerationQualificationError> {
    if generator.arena_bytes() != 7_039_870_976
        || generator.host_stager_bytes() != 9_060_352
        || generator.context_capacity() != 192
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 compact owner bytes or capacity changed".into(),
        ));
    }
    let addresses = generator.qualification_addresses();
    let mut unique = addresses.clone();
    unique.sort_unstable();
    unique.dedup();
    if addresses.len() != 34 || unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 compact owner addresses are invalid".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{qualify_qwen35_compact_generation, qualify_qwen35_generation};
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive SM120 device"]
    fn source_frontend_generation_matches_transformers_and_streaming()
    -> Result<(), super::Qwen35GenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            super::Qwen35GenerationQualificationError::Mismatch(
                "set TUISKO_QWEN35_SNAPSHOT to the admitted revision".into(),
            )
        })?;
        let report = qualify_qwen35_generation(&PathBuf::from(root))?;
        assert_eq!(report.prompt_cases, 2);
        assert!((1..=2).contains(&report.chat_steps));
        assert_eq!(report.native_prefill_tokens, [0, 32, 32, 64, 64, 128, 128]);
        assert_eq!(report.arena_bytes, 7_039_870_976);
        assert_eq!(report.host_stager_bytes, 1_610_752);
        assert_eq!(report.stable_addresses, 34);
        eprintln!("Qwen3.5 generation qualification passed: {report:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive SM120 device"]
    fn compact_scheduler_matches_sequential_routes_and_recycles_a_hole()
    -> Result<(), super::Qwen35GenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            super::Qwen35GenerationQualificationError::Mismatch(
                "set TUISKO_QWEN35_SNAPSHOT to the admitted revision".into(),
            )
        })?;
        let report = qualify_qwen35_compact_generation(&PathBuf::from(root))?;
        assert_eq!(report.route_batches, 8);
        assert_eq!(report.requests, 4);
        assert_eq!(report.recycled_slot, 1);
        assert_eq!(report.concurrent_prefill_tokens, 32);
        assert_eq!(report.cancellations, 1);
        assert_eq!(report.arena_bytes, 7_039_870_976);
        assert_eq!(report.host_stager_bytes, 9_060_352);
        assert_eq!(report.stable_addresses, 34);
        eprintln!("Qwen3.5 compact generation qualification passed: {report:?}");
        Ok(())
    }
}

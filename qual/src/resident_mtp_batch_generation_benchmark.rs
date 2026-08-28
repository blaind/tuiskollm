//! Direct host-completion timing for exact compact MTP scheduler transactions.

use crate::device_benchmark::{
    BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkOptions,
    DeviceBenchmarkReport, MemoryRecorder, TelemetrySampler, finish_report,
    generator_baseline_sha256, host_completion_metric, measurement_order, preflight,
    require_current_process_exclusive, validate_loaded_host_clock_policy, warmup_launches,
};
use crate::resident_mtp_generation_benchmark::register_program_memory;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{ChatGenerationRequest, ResidentMtpBatchGenerator, SamplingOptions};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const OUTPUT_TOKENS: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RoundOutcome {
    committed: usize,
    token_ids: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CancellationOutcome {
    prompt_tokens: usize,
    message_boundary_tokens: usize,
    followup_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompletionFallbackOutcome {
    prompt_tokens: usize,
    retained_tokens: usize,
    message_boundary_tokens: usize,
    followup_tokens: usize,
}

/// Measures every exact compact B=1..8 draft-three/K=4 scheduler transaction directly.
pub fn benchmark_resident_mtp_batch_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "compact MTP scheduler energy is deferred to the full-server gate".to_string(),
        ));
    }
    let batches = selected_batches(options.batch_size)?;
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(DeviceBenchmarkError::Precondition(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot)?;
    register_program_memory(&mut memory, generator.qualification_program())?;
    memory.capture("after_setup")?;
    let request = request();

    let mut references = Vec::with_capacity(batches.len());
    let mut prompt_tokens = Vec::with_capacity(batches.len());
    for &batch in &batches {
        let mut reference = None;
        let mut prompt = None;
        for _ in 0..warmup {
            let (_, outcome, current_prompt) = run_round(&mut generator, &request, batch)?;
            require_reference(batch, reference.as_ref(), &outcome)?;
            reference.get_or_insert(outcome);
            match prompt {
                Some(expected) if expected != current_prompt => {
                    return Err(DeviceBenchmarkError::Precondition(format!(
                        "compact MTP B={batch} prompt length changed during warmup"
                    )));
                }
                None => prompt = Some(current_prompt),
                Some(_) => {}
            }
        }
        references.push(reference.ok_or_else(|| {
            DeviceBenchmarkError::Precondition(format!(
                "compact MTP B={batch} requires at least one warmup"
            ))
        })?);
        prompt_tokens.push(prompt.expect("warmup established prompt length"));
    }
    let followup = followup_request();
    let mut cancellation_reference = None;
    for _ in 0..warmup {
        let (_, outcome) = run_cancellation_resume(&mut generator, &request, &followup)?;
        if cancellation_reference.is_some_and(|expected| expected != outcome) {
            return Err(DeviceBenchmarkError::Precondition(
                "cancellation-resume boundary changed during warmup".to_string(),
            ));
        }
        cancellation_reference = Some(outcome);
    }
    let cancellation_reference = cancellation_reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "cancellation-resume benchmark requires at least one warmup".to_string(),
        )
    })?;
    let completed_followup = completed_followup_request();
    let mut completion_fallback_reference = None;
    for _ in 0..warmup {
        let (_, outcome) = run_completion_fallback(&mut generator, &request, &completed_followup)?;
        if completion_fallback_reference.is_some_and(|expected| expected != outcome) {
            return Err(DeviceBenchmarkError::Precondition(
                "completed-prefix fallback boundary changed during warmup".to_string(),
            ));
        }
        completion_fallback_reference = Some(outcome);
    }
    let completion_fallback_reference = completion_fallback_reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "completed-prefix fallback benchmark requires at least one warmup".to_string(),
        )
    })?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;

    let loaded_batch = *batches
        .last()
        .expect("selected batch inventory is nonempty");
    let loaded_reference = references
        .last()
        .expect("selected batch reference exists")
        .clone();
    validate_loaded_host_clock_policy("qwen3_8/generation/mtp_batch_round", || {
        let (_, outcome, _) = run_round(&mut generator, &request, loaded_batch)?;
        require_reference(loaded_batch, Some(&loaded_reference), &outcome)?;
        let (_, cancellation) = run_cancellation_resume(&mut generator, &request, &followup)?;
        if cancellation != cancellation_reference {
            return Err(DeviceBenchmarkError::Precondition(
                "loaded cancellation-resume probe changed its exact boundary".to_string(),
            ));
        }
        let (_, completion_fallback) =
            run_completion_fallback(&mut generator, &request, &completed_followup)?;
        if completion_fallback != completion_fallback_reference {
            return Err(DeviceBenchmarkError::Precondition(
                "loaded completed-prefix fallback probe changed its exact boundary".to_string(),
            ));
        }
        Ok(())
    })?;

    let sampler = TelemetrySampler::start();
    let mut samples = batches
        .iter()
        .map(|_| Vec::with_capacity(options.samples))
        .collect::<Vec<_>>();
    let mut cancellation_samples = Vec::with_capacity(options.samples);
    let mut completion_fallback_samples = Vec::with_capacity(options.samples);
    for sample in 0..options.samples {
        for case in measurement_order(sample, batches.len()) {
            let batch = batches[case];
            let mut elapsed = Duration::ZERO;
            for _ in 0..options.launches_per_sample {
                let (round_elapsed, outcome, current_prompt) =
                    run_round(&mut generator, &request, batch)?;
                elapsed += round_elapsed;
                require_reference(batch, Some(&references[case]), &outcome)?;
                if current_prompt != prompt_tokens[case] {
                    return Err(DeviceBenchmarkError::Precondition(format!(
                        "compact MTP B={batch} prompt length changed between samples"
                    )));
                }
            }
            samples[case]
                .push(elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
        }
        let mut elapsed = Duration::ZERO;
        for _ in 0..options.launches_per_sample {
            let (iteration, outcome) =
                run_cancellation_resume(&mut generator, &request, &followup)?;
            elapsed += iteration;
            if outcome != cancellation_reference {
                return Err(DeviceBenchmarkError::Precondition(
                    "cancellation-resume boundary changed between samples".to_string(),
                ));
            }
        }
        cancellation_samples
            .push(elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
        let mut elapsed = Duration::ZERO;
        for _ in 0..options.launches_per_sample {
            let (iteration, outcome) =
                run_completion_fallback(&mut generator, &request, &completed_followup)?;
            elapsed += iteration;
            if outcome != completion_fallback_reference {
                return Err(DeviceBenchmarkError::Precondition(
                    "completed-prefix fallback boundary changed between samples".to_string(),
                ));
            }
        }
        completion_fallback_samples
            .push(elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
    }
    let telemetry = sampler.finish()?;
    require_current_process_exclusive()?;

    let mut metrics = Vec::with_capacity(batches.len() + 1);
    for (case, &batch) in batches.iter().enumerate() {
        let committed = references[case].committed as u64;
        let prompt = prompt_tokens[case] as u64;
        metrics.push(host_completion_metric(
            "qwen3_8/generation/mtp_batch_round",
            format!(
                "B={batch},context={},draft=3,K=4,committed={committed}",
                prompt + 4
            ),
            BenchmarkWorkload::warm_model_mtp_batch_round(
                u32::try_from(batch).map_err(|_| {
                    DeviceBenchmarkError::Precondition("compact MTP batch exceeds u32".to_string())
                })?,
                committed,
                prompt + 4,
            ),
            options.launches_per_sample,
            0,
            std::mem::take(&mut samples[case]),
        )?);
    }
    metrics.push(host_completion_metric(
        "qwen3_8/generation/cancellation_resume",
        format!(
            "B=1,prompt={},boundary={},followup={}",
            cancellation_reference.prompt_tokens,
            cancellation_reference.message_boundary_tokens,
            cancellation_reference.followup_tokens
        ),
        BenchmarkWorkload::warm_model_cancellation_resume(
            cancellation_reference.prompt_tokens as u64,
            cancellation_reference.message_boundary_tokens as u64,
            cancellation_reference.followup_tokens as u64,
        ),
        options.launches_per_sample,
        0,
        cancellation_samples,
    )?);
    metrics.push(host_completion_metric(
        "qwen3_8/generation/completed_prefix_fallback",
        format!(
            "B=1,prompt={},retained={},boundary={},followup={}",
            completion_fallback_reference.prompt_tokens,
            completion_fallback_reference.retained_tokens,
            completion_fallback_reference.message_boundary_tokens,
            completion_fallback_reference.followup_tokens
        ),
        BenchmarkWorkload::warm_model_completed_prefix_fallback(
            completion_fallback_reference.prompt_tokens as u64,
            completion_fallback_reference.message_boundary_tokens as u64,
            completion_fallback_reference.followup_tokens as u64,
            OUTPUT_TOKENS as u64,
        ),
        options.launches_per_sample,
        0,
        completion_fallback_samples,
    )?);
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-generation-mtp-batch",
            classification: "performance_sensitive_model",
            timing_scope: "direct Rust host completion for compact proposal continuation, cancellation restore, and completed-prefix message-boundary fallback through the production scheduler",
        },
        preflight,
        baseline_sha256,
        options,
        metrics,
        Vec::new(),
        telemetry,
        memory,
    )
}

fn run_completion_fallback(
    generator: &mut ResidentMtpBatchGenerator,
    request: &ChatGenerationRequest,
    followup: &ChatGenerationRequest,
) -> Result<(Duration, CompletionFallbackOutcome), DeviceBenchmarkError> {
    generator.qualification_clear_retained()?;
    let started = Instant::now();
    let admission = generator.admit(request)?;
    if admission.device_reused_tokens != 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "completed-prefix benchmark initial request unexpectedly reused a prefix".to_string(),
        ));
    }
    let slot = generator
        .qualification_slot(admission.request_id)
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition(
                "completed-prefix benchmark seed has no physical slot".to_string(),
            )
        })?;
    let mut output = None;
    while generator
        .active_request_ids()
        .any(|request_id| request_id == admission.request_id)
    {
        let events = generator.step()?;
        if let Some(completed) = events
            .iter()
            .find(|event| event.request_id == admission.request_id)
            .and_then(|event| event.completed.as_ref())
        {
            output = Some(completed.clone());
        }
    }
    let output = output.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "completed-prefix benchmark seed returned no output".to_string(),
        )
    })?;
    let message_boundary_tokens = output.prompt.message_boundary_tokens;
    let retained_tokens = generator
        .qualification_retained_tokens(slot)
        .ok_or_else(|| {
            DeviceBenchmarkError::Precondition(
                "completed-prefix benchmark seed retained no device state".to_string(),
            )
        })?;
    if retained_tokens <= message_boundary_tokens
        || generator.qualification_retained_message_boundary(slot) != Some(message_boundary_tokens)
    {
        return Err(DeviceBenchmarkError::Precondition(
            "completed-prefix benchmark retained the wrong fallback boundary".to_string(),
        ));
    }
    let resumed = generator.admit(followup)?;
    if resumed.device_reused_tokens != message_boundary_tokens
        || generator.qualification_slot(resumed.request_id) != Some(slot)
    {
        return Err(DeviceBenchmarkError::Precondition(
            "completed-prefix benchmark did not restore the divergent follow-up".to_string(),
        ));
    }
    let elapsed = started.elapsed();
    let outcome = CompletionFallbackOutcome {
        prompt_tokens: admission.prompt_tokens,
        retained_tokens,
        message_boundary_tokens,
        followup_tokens: resumed.prompt_tokens,
    };
    let _ = generator.cancel(resumed.request_id)?;
    generator.qualification_clear_retained()?;

    Ok((elapsed, outcome))
}

fn run_cancellation_resume(
    generator: &mut ResidentMtpBatchGenerator,
    request: &ChatGenerationRequest,
    followup: &ChatGenerationRequest,
) -> Result<(Duration, CancellationOutcome), DeviceBenchmarkError> {
    generator.qualification_clear_retained()?;
    let started = Instant::now();
    let admission = generator.admit(request)?;
    if admission.device_reused_tokens != 0 {
        return Err(DeviceBenchmarkError::Precondition(
            "cancellation benchmark initial request unexpectedly reused a prefix".to_string(),
        ));
    }
    let cancelled = generator.cancel(admission.request_id)?;
    let message_boundary_tokens = cancelled.device_retained_tokens;
    if message_boundary_tokens != cancelled.output.prompt.message_boundary_tokens
        || message_boundary_tokens >= admission.prompt_tokens
    {
        return Err(DeviceBenchmarkError::Precondition(
            "cancellation benchmark retained the wrong message boundary".to_string(),
        ));
    }
    let resumed = generator.admit(followup)?;
    if resumed.device_reused_tokens != message_boundary_tokens {
        return Err(DeviceBenchmarkError::Precondition(
            "cancellation benchmark did not restore the divergent follow-up prefix".to_string(),
        ));
    }
    let elapsed = started.elapsed();
    let outcome = CancellationOutcome {
        prompt_tokens: admission.prompt_tokens,
        message_boundary_tokens,
        followup_tokens: resumed.prompt_tokens,
    };
    let _ = generator.cancel(resumed.request_id)?;
    generator.qualification_clear_retained()?;
    Ok((elapsed, outcome))
}

fn run_round(
    generator: &mut ResidentMtpBatchGenerator,
    request: &ChatGenerationRequest,
    batch: usize,
) -> Result<(Duration, RoundOutcome, usize), DeviceBenchmarkError> {
    generator.qualification_clear_retained()?;
    let mut prompt_tokens = None;
    for _ in 0..batch {
        let admission = generator.admit(request)?;
        if admission.device_reused_tokens != 0 {
            return Err(DeviceBenchmarkError::Precondition(
                "compact MTP benchmark setup unexpectedly reused a prefix".to_string(),
            ));
        }
        match prompt_tokens {
            Some(expected) if expected != admission.prompt_tokens => {
                return Err(DeviceBenchmarkError::Precondition(
                    "compact MTP benchmark lanes have different prompt lengths".to_string(),
                ));
            }
            None => prompt_tokens = Some(admission.prompt_tokens),
            Some(_) => {}
        }
    }
    let anchors = generator.step()?;
    if anchors.len() != batch
        || anchors
            .iter()
            .any(|event| event.len() != 1 || event.completed.is_some())
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "compact MTP B={batch} setup did not stop at one proposal-ready anchor per lane"
        )));
    }

    let started = Instant::now();
    let events = generator.step()?;
    let elapsed = started.elapsed();
    if events.len() != batch {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "compact MTP B={batch} returned {} timed events",
            events.len()
        )));
    }
    let mut token_ids = Vec::with_capacity(batch);
    let mut committed = 0;
    for event in events.iter() {
        if event.stats.verification_routes[3] != 1 || event.stats.draft_proposals < 3 {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "compact MTP B={batch} did not execute draft-three/K=4"
            )));
        }
        let tokens = event.steps().map(|step| step.token_id).collect::<Vec<_>>();
        committed += tokens.len();
        token_ids.push(tokens);
    }
    let active = generator.active_request_ids().collect::<Vec<_>>();
    for request_id in active {
        let _ = generator.cancel(request_id)?;
    }
    generator.qualification_clear_retained()?;
    Ok((
        elapsed,
        RoundOutcome {
            committed,
            token_ids,
        },
        prompt_tokens.expect("nonempty exact batch established prompt length"),
    ))
}

fn require_reference(
    batch: usize,
    expected: Option<&RoundOutcome>,
    actual: &RoundOutcome,
) -> Result<(), DeviceBenchmarkError> {
    if actual.committed < batch || actual.committed > 4 * batch {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "compact MTP B={batch} committed {} outputs, expected B..4B",
            actual.committed
        )));
    }
    if let Some(expected) = expected
        && expected != actual
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "compact MTP B={batch} output changed between benchmark replays"
        )));
    }
    Ok(())
}

fn selected_batches(batch_size: Option<u32>) -> Result<Vec<usize>, DeviceBenchmarkError> {
    match batch_size {
        None => Ok((1..=8).collect()),
        Some(batch @ 1..=8) => Ok(vec![usize::try_from(batch).map_err(|_| {
            DeviceBenchmarkError::Precondition("compact MTP batch exceeds host width".to_string())
        })?]),
        Some(batch) => Err(DeviceBenchmarkError::Precondition(format!(
            "compact MTP scheduler batch {batch} is outside 1..=8"
        ))),
    }
}

fn request() -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = OUTPUT_TOKENS;
    request
}

fn followup_request() -> ChatGenerationRequest {
    let mut request = request();
    request.messages.push(ChatMessage::new(
        "user",
        "Continue with a different primary color.",
    ));
    request
}

fn completed_followup_request() -> ChatGenerationRequest {
    let mut request = request();
    request.messages.push(ChatMessage::new(
        "assistant",
        "This deliberately differs from the generated assistant turn.",
    ));
    request.messages.push(ChatMessage::new(
        "user",
        "Continue with a different primary color.",
    ));
    request
}

#[cfg(test)]
mod tests {
    #[test]
    fn resident_mtp_batch_suite_benchmark_inventory_is_exact() {
        assert_eq!(
            super::selected_batches(None).unwrap(),
            (1..=8).collect::<Vec<_>>()
        );
        for batch in 1..=8 {
            assert_eq!(
                super::selected_batches(Some(batch)).unwrap(),
                [batch as usize]
            );
        }
        assert!(super::selected_batches(Some(0)).is_err());
        assert!(super::selected_batches(Some(9)).is_err());
    }
}

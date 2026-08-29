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
use tuisko_engine::{
    ChatGenerationRequest, PromptLogprobs, ResidentMtpBatchGenerator, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const OUTPUT_TOKENS: usize = 8;
const SCORING_BATCH: u32 = 4;
const SHARED_SCORING_METRIC: &str = "qwen3_8/scoring/prompt_batch_shared_prefix";
const INDEPENDENT_SCORING_METRIC: &str = "qwen3_8/scoring/prompt_batch_independent";

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

/// Directly measures the production four-choice scoring owner and its independent reference.
pub fn benchmark_resident_mtp_prompt_scoring(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring energy is deferred to the full-server gate".to_string(),
        ));
    }
    if options.batch_size != Some(SCORING_BATCH) {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "prompt scoring benchmark requires production batch_size={SCORING_BATCH}"
        )));
    }
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let frontend = TextFrontend::open(snapshot.as_ref()).map_err(|error| {
        DeviceBenchmarkError::Precondition(format!(
            "prompt scoring benchmark frontend admission failed: {error}"
        ))
    })?;
    let prompts = scoring_prompts(&frontend)?;
    let (prompt_tokens, common_tokens) = scoring_prompt_shape(&prompts)?;
    let input_reference = prompts.clone();
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(DeviceBenchmarkError::Precondition(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let mut generator = ResidentMtpBatchGenerator::from_snapshot(&context, snapshot)?;
    register_program_memory(&mut memory, generator.qualification_program())?;
    memory.capture("after_setup")?;
    let stable_addresses = generator.qualification_addresses()?;

    let mut reference = None;
    for _ in 0..warmup {
        let (_, shared) = run_shared_scoring(&mut generator, &prompts)?;
        let (_, independent) = run_independent_scoring(&mut generator, &prompts)?;
        require_scoring_output(reference.as_ref(), &shared, &independent)?;
        reference.get_or_insert(shared);
        require_scoring_input(&prompts, &input_reference)?;
    }
    let reference = reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "prompt scoring benchmark requires at least one warmup".to_string(),
        )
    })?;
    require_stable_addresses(&generator, &stable_addresses)?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;

    validate_loaded_host_clock_policy(SHARED_SCORING_METRIC, || {
        let (_, shared) = run_shared_scoring(&mut generator, &prompts)?;
        let (_, independent) = run_independent_scoring(&mut generator, &prompts)?;
        require_scoring_output(Some(&reference), &shared, &independent)?;
        require_stable_addresses(&generator, &stable_addresses)
    })?;

    let sampler = TelemetrySampler::start();
    let mut samples = [
        Vec::with_capacity(options.samples),
        Vec::with_capacity(options.samples),
    ];
    for sample in 0..options.samples {
        for case in measurement_order(sample, samples.len()) {
            let mut elapsed = Duration::ZERO;
            for _ in 0..options.launches_per_sample {
                let (iteration, outcome) = if case == 0 {
                    run_shared_scoring(&mut generator, &prompts)?
                } else {
                    run_independent_scoring(&mut generator, &prompts)?
                };
                elapsed += iteration;
                if outcome != reference {
                    return Err(DeviceBenchmarkError::Precondition(format!(
                        "prompt scoring case {case} changed output between samples"
                    )));
                }
            }
            samples[case]
                .push(elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
            require_scoring_input(&prompts, &input_reference)?;
            require_stable_addresses(&generator, &stable_addresses)?;
        }
    }
    let telemetry = sampler.finish()?;
    require_current_process_exclusive()?;
    memory.capture("after_measurement")?;

    let shape = format!(
        "B={SCORING_BATCH},choices={SCORING_BATCH},prompt={prompt_tokens},common={common_tokens}"
    );
    let workload = BenchmarkWorkload::warm_model_prompt_scoring(
        SCORING_BATCH,
        u64::try_from(prompt_tokens).map_err(|_| {
            DeviceBenchmarkError::Precondition("prompt length exceeds u64".to_string())
        })?,
    );
    let metrics = vec![
        host_completion_metric(
            SHARED_SCORING_METRIC,
            shape.clone(),
            workload.clone(),
            options.launches_per_sample,
            0,
            std::mem::take(&mut samples[0]),
        )?,
        host_completion_metric(
            INDEPENDENT_SCORING_METRIC,
            shape,
            workload,
            options.launches_per_sample,
            0,
            std::mem::take(&mut samples[1]),
        )?,
    ];
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-prompt-scoring",
            classification: "performance_sensitive_model",
            timing_scope: "separate direct Rust host completion for one production four-choice shared-prefix batch and the complete four-prompt independent reference",
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

fn scoring_prompts(frontend: &TextFrontend) -> Result<Vec<Vec<u32>>, DeviceBenchmarkError> {
    let stem = "The following is a multiple choice question. Select the best answer.\n\nWhich property must every prime number greater than two have?\nA. It is even.\nB. It is odd.\nC. It is divisible by three.\nD. It is a perfect square.\nAnswer:";
    let mut prompts = Vec::with_capacity(SCORING_BATCH as usize);
    for choice in [" A", " B", " C", " D"] {
        let prompt = frontend
            .encode(&format!("{stem}{choice}"))
            .map_err(|error| {
                DeviceBenchmarkError::Precondition(format!(
                    "prompt scoring benchmark full choice encoding failed: {error}"
                ))
            })?;
        prompts.push(prompt);
    }
    scoring_prompt_shape(&prompts)?;
    Ok(prompts)
}

fn scoring_prompt_shape(prompts: &[Vec<u32>]) -> Result<(usize, usize), DeviceBenchmarkError> {
    if prompts.len() != SCORING_BATCH as usize {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring benchmark requires exactly four choices".to_string(),
        ));
    }
    let prompt_tokens = prompts[0].len();
    if prompt_tokens < 2 || prompts.iter().any(|prompt| prompt.len() != prompt_tokens) {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring benchmark choices are not equal nontrivial lengths".to_string(),
        ));
    }
    let common_tokens = (0..prompt_tokens)
        .take_while(|&position| {
            prompts[1..]
                .iter()
                .all(|prompt| prompt[position] == prompts[0][position])
        })
        .count();
    if common_tokens != prompt_tokens - 1 {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "prompt scoring benchmark common prefix is {common_tokens}, expected {}",
            prompt_tokens - 1
        )));
    }
    let final_choices = prompts
        .iter()
        .map(|prompt| prompt[prompt_tokens - 1])
        .collect::<std::collections::BTreeSet<_>>();
    if final_choices.len() != SCORING_BATCH as usize {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring benchmark final choice token IDs are not distinct".to_string(),
        ));
    }
    Ok((prompt_tokens, common_tokens))
}

fn run_shared_scoring(
    generator: &mut ResidentMtpBatchGenerator,
    prompts: &[Vec<u32>],
) -> Result<(Duration, Vec<PromptLogprobs>), DeviceBenchmarkError> {
    let started = Instant::now();
    let output = generator.score_prompts(prompts)?;
    Ok((started.elapsed(), output))
}

fn run_independent_scoring(
    generator: &mut ResidentMtpBatchGenerator,
    prompts: &[Vec<u32>],
) -> Result<(Duration, Vec<PromptLogprobs>), DeviceBenchmarkError> {
    let started = Instant::now();
    let output = prompts
        .iter()
        .map(|prompt| generator.score_prompt(prompt))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((started.elapsed(), output))
}

fn require_scoring_output(
    expected: Option<&Vec<PromptLogprobs>>,
    shared: &[PromptLogprobs],
    independent: &[PromptLogprobs],
) -> Result<(), DeviceBenchmarkError> {
    if shared != independent || expected.is_some_and(|expected| expected != shared) {
        return Err(DeviceBenchmarkError::Precondition(
            "shared and independent prompt scoring outputs are not invariant".to_string(),
        ));
    }
    Ok(())
}

fn require_scoring_input(
    prompts: &[Vec<u32>],
    expected: &[Vec<u32>],
) -> Result<(), DeviceBenchmarkError> {
    if prompts != expected {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring benchmark input changed between replays".to_string(),
        ));
    }
    Ok(())
}

fn require_stable_addresses(
    generator: &ResidentMtpBatchGenerator,
    expected: &[usize],
) -> Result<(), DeviceBenchmarkError> {
    if generator.qualification_addresses()? != expected {
        return Err(DeviceBenchmarkError::Precondition(
            "prompt scoring owner addresses changed after setup".to_string(),
        ));
    }
    Ok(())
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

    #[test]
    fn resident_mtp_batch_suite_prompt_scoring_benchmark_accounting_is_exact() {
        use crate::device_benchmark::{BenchmarkWorkload, DeviceBenchmarkOptions};

        let options = DeviceBenchmarkOptions::prompt_scoring();
        assert_eq!(
            (
                options.samples,
                options.launches_per_sample,
                options.warmup_launches,
                options.batch_size,
            ),
            (9, 1, 1, Some(super::SCORING_BATCH))
        );
        let workload = BenchmarkWorkload::warm_model_prompt_scoring(super::SCORING_BATCH, 96);
        assert_eq!(workload.batch_size, Some(4));
        assert_eq!(workload.active_tokens, Some(384));
        assert_eq!(workload.prompt_tokens, Some(96));
        assert_eq!(workload.output_tokens, Some(4));
        assert_ne!(
            super::SHARED_SCORING_METRIC,
            super::INDEPENDENT_SCORING_METRIC
        );
    }

    #[test]
    fn resident_mtp_batch_suite_prompt_scoring_shape_requires_one_distinct_choice_token() {
        let valid = vec![
            vec![10, 11, 20],
            vec![10, 11, 21],
            vec![10, 11, 22],
            vec![10, 11, 23],
        ];
        assert_eq!(super::scoring_prompt_shape(&valid).unwrap(), (3, 2));

        let mut duplicate = valid.clone();
        duplicate[3][2] = 22;
        assert!(super::scoring_prompt_shape(&duplicate).is_err());

        let mut short_prefix = valid.clone();
        short_prefix[3][1] = 12;
        assert!(super::scoring_prompt_shape(&short_prefix).is_err());

        let mut unequal = valid;
        unequal[3].push(24);
        assert!(super::scoring_prompt_shape(&unequal).is_err());
    }
}

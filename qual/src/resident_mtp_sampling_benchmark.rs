//! Direct host-completion timing for production sampled MTP generation.

use crate::device_benchmark::{
    BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkOptions,
    DeviceBenchmarkReport, MemoryRecorder, TelemetrySampler, finish_report,
    generator_baseline_sha256, host_completion_metric, preflight,
    require_current_process_exclusive, warmup_launches,
};
use crate::resident_mtp_generation_benchmark::register_memory;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{
    ChatGenerationRequest, GeneratedText, ResidentMtpGenerationStats, ResidentMtpTextGenerator,
    SamplingOptions, SamplingPenalties,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const OUTPUT_TOKENS: usize = 8;
const SAMPLE_SEED: u64 = 29;

/// Measures one sampled round plus identity- and penalty-conditioned complete requests.
pub fn benchmark_resident_mtp_sampling(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.batch_size.is_some_and(|batch| batch != 1) {
        return Err(DeviceBenchmarkError::Precondition(
            "resident sampled MTP benchmark admits only B=1".to_string(),
        ));
    }
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "resident sampled MTP energy is deferred to the full-server gate".to_string(),
        ));
    }
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
    let mut generator = ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
    register_memory(&mut memory, &generator)?;
    memory.capture("after_setup")?;
    let request = sampled_request(SamplingPenalties::identity());
    let penalized = sampled_request(SamplingPenalties {
        presence: 1.5,
        frequency: 0.5,
        repetition: 1.1,
    });
    let mut request_reference = None;
    let mut penalty_reference = None;
    let mut round_outputs = None;
    for _ in 0..warmup {
        let (output, stats) = run_request(&mut generator, &request)?;
        require_k4(stats)?;
        retain_reference(&mut request_reference, &output)?;
        let (output, stats) = run_request(&mut generator, &penalized)?;
        require_k4(stats)?;
        retain_reference(&mut penalty_reference, &output)?;
        let (_, stats) = run_round(&mut generator, &request)?;
        require_k4(stats)?;
        match round_outputs {
            Some(expected) if expected != stats.verified_outputs => {
                return Err(DeviceBenchmarkError::Precondition(
                    "sampled MTP round committed count changed during warmup".to_string(),
                ));
            }
            None => round_outputs = Some(stats.verified_outputs),
            Some(_) => {}
        }
    }
    let request_reference = request_reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "sampled MTP benchmark requires at least one warmup request".to_string(),
        )
    })?;
    let penalty_reference = penalty_reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "penalty-conditioned MTP benchmark requires a warmup request".to_string(),
        )
    })?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;

    let sampler = TelemetrySampler::start();
    let mut request_samples = Vec::with_capacity(options.samples);
    let mut penalty_samples = Vec::with_capacity(options.samples);
    let mut round_samples = Vec::with_capacity(options.samples);
    for sample in 0..options.samples {
        let mut elapsed = [Duration::ZERO; 3];
        for offset in 0..3 {
            let task = (sample + offset) % 3;
            for _ in 0..options.launches_per_sample {
                match task {
                    0 => {
                        let started = Instant::now();
                        let (output, stats) = run_request(&mut generator, &request)?;
                        elapsed[0] += started.elapsed();
                        require_same_output(&request_reference, &output)?;
                        require_k4(stats)?;
                    }
                    1 => {
                        let started = Instant::now();
                        let (output, stats) = run_request(&mut generator, &penalized)?;
                        elapsed[1] += started.elapsed();
                        require_same_output(&penalty_reference, &output)?;
                        require_k4(stats)?;
                    }
                    _ => {
                        let (duration, stats) = run_round(&mut generator, &request)?;
                        elapsed[2] += duration;
                        require_k4(stats)?;
                        if Some(stats.verified_outputs) != round_outputs {
                            return Err(DeviceBenchmarkError::Precondition(
                                "sampled MTP round committed count changed between samples"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
        }
        let divisor = options.launches_per_sample as f64;
        request_samples.push(elapsed[0].as_secs_f64() * 1_000_000.0 / divisor);
        penalty_samples.push(elapsed[1].as_secs_f64() * 1_000_000.0 / divisor);
        round_samples.push(elapsed[2].as_secs_f64() * 1_000_000.0 / divisor);
    }
    let telemetry = sampler.finish()?;
    require_current_process_exclusive()?;
    let prompt_tokens = request_reference.prompt.token_ids.len() as u64;
    let output_tokens = request_reference.token_ids.len() as u64;
    let context_tokens = prompt_tokens + output_tokens - 1;
    let committed = round_outputs.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "sampled MTP benchmark did not warm one speculative round".to_string(),
        )
    })? as u64;
    let metrics = vec![
        host_completion_metric(
            "qwen3_8/generation/mtp_sampled_round",
            format!(
                "B=1,context={},draft=3,K=4,committed={committed},seed={SAMPLE_SEED}",
                prompt_tokens + 4
            ),
            BenchmarkWorkload::warm_model_mtp_round(committed, prompt_tokens + 4),
            options.launches_per_sample,
            0,
            round_samples,
        )?,
        host_completion_metric(
            "qwen3_8/generation/mtp_sampled_request",
            format!("B=1,prompt={prompt_tokens},output={output_tokens},draft=3,seed={SAMPLE_SEED}"),
            BenchmarkWorkload::warm_model_mtp_request(prompt_tokens, output_tokens, context_tokens),
            options.launches_per_sample,
            0,
            request_samples,
        )?,
        host_completion_metric(
            "qwen3_8/generation/mtp_sampled_penalties",
            format!(
                "B=1,prompt={prompt_tokens},output={},draft=3,seed={SAMPLE_SEED},presence=1.5,frequency=0.5,repetition=1.1",
                penalty_reference.token_ids.len()
            ),
            BenchmarkWorkload::warm_model_mtp_request(
                prompt_tokens,
                penalty_reference.token_ids.len() as u64,
                prompt_tokens + penalty_reference.token_ids.len() as u64 - 1,
            ),
            options.launches_per_sample,
            0,
            penalty_samples,
        )?,
    ];
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-generation-mtp-sampling",
            classification: "performance_sensitive_model",
            timing_scope: "direct Rust host completion for unbiased sampled draft-three, target verify/commit, residual correction, penalty conditioning, MTP realignment, and streaming control through the production owner",
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

fn sampled_request(penalties: SamplingPenalties) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions {
        seed: SAMPLE_SEED,
        penalties,
        ..SamplingOptions::default()
    };
    request.max_new_tokens = OUTPUT_TOKENS;
    request
}

fn run_request(
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(GeneratedText, ResidentMtpGenerationStats), DeviceBenchmarkError> {
    let mut session = generator.start(request)?;
    while session.finish_reason().is_none() {
        let _ = session.step()?;
    }
    let stats = session.stats();
    Ok((session.into_output()?, stats))
}

fn run_round(
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(Duration, ResidentMtpGenerationStats), DeviceBenchmarkError> {
    let mut session = generator.start(request)?;
    let _ = session.step()?;
    let started = Instant::now();
    let _ = session.step()?;
    Ok((started.elapsed(), session.stats()))
}

fn retain_reference(
    reference: &mut Option<GeneratedText>,
    output: &GeneratedText,
) -> Result<(), DeviceBenchmarkError> {
    if let Some(reference) = reference {
        require_same_output(reference, output)
    } else {
        *reference = Some(output.clone());
        Ok(())
    }
}

fn require_same_output(
    expected: &GeneratedText,
    actual: &GeneratedText,
) -> Result<(), DeviceBenchmarkError> {
    if actual.prompt.token_ids != expected.prompt.token_ids
        || actual.token_ids != expected.token_ids
        || actual.text != expected.text
        || actual.finish_reason != expected.finish_reason
    {
        return Err(DeviceBenchmarkError::Precondition(
            "seeded sampled MTP output changed between benchmark iterations".to_string(),
        ));
    }
    Ok(())
}

fn require_k4(stats: ResidentMtpGenerationStats) -> Result<(), DeviceBenchmarkError> {
    if stats.verification_routes[3] == 0 || stats.draft_proposals < 3 {
        return Err(DeviceBenchmarkError::Precondition(
            "sampled MTP benchmark did not execute a draft-three/K=4 round".to_string(),
        ));
    }
    Ok(())
}

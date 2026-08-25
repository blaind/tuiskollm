//! Direct host-completion timing for one production Qwen3.5 MTP request.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, MemoryRecorder, TelemetrySampler, finish_report,
    generator_baseline_sha256, host_completion_metric, preflight,
    require_current_process_exclusive, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{
    ChatGenerationRequest, GeneratedText, Qwen35ResidentMtpTextGenerator,
    ResidentMtpGenerationStats, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen35_9B};

const OUTPUT_TOKENS: usize = 8;

/// Measures full Qwen3.5 MTP requests and one draft-three transaction directly.
pub fn benchmark_qwen35_mtp_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.batch_size.is_some_and(|batch| batch != 1) {
        return Err(DeviceBenchmarkError::Precondition(
            "Qwen3.5 MTP generation benchmark admits only B=1".to_string(),
        ));
    }
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "Qwen3.5 MTP request energy belongs to the full-server gate".to_string(),
        ));
    }
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let mut memory = MemoryRecorder::new(&preflight)?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(DeviceBenchmarkError::Precondition(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let mut generator = Qwen35ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
    register_memory(&mut memory, &generator)?;
    memory.capture("after_setup")?;
    let request = request();
    let mut reference = None;
    let mut round_outputs = None;
    for _ in 0..warmup {
        let (output, stats) = run_request(&mut generator, &request)?;
        require_k4(stats)?;
        if let Some(expected) = &reference {
            require_same_output(expected, &output)?;
        } else {
            reference = Some(output);
        }
        let (_, stats) = run_round(&mut generator, &request)?;
        require_k4(stats)?;
        match round_outputs {
            Some(expected) if expected != stats.verified_outputs => {
                return Err(DeviceBenchmarkError::Precondition(
                    "Qwen3.5 MTP committed width changed during warmup".to_string(),
                ));
            }
            None => round_outputs = Some(stats.verified_outputs),
            Some(_) => {}
        }
    }
    let reference = reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "Qwen3.5 MTP benchmark requires one warmup request".to_string(),
        )
    })?;
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;

    let sampler = TelemetrySampler::start();
    let mut request_samples = Vec::with_capacity(options.samples);
    let mut round_samples = Vec::with_capacity(options.samples);
    for sample in 0..options.samples {
        let mut request_elapsed = Duration::ZERO;
        let mut round_elapsed = Duration::ZERO;
        for task in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
            for _ in 0..options.launches_per_sample {
                if task == 0 {
                    let started = Instant::now();
                    let (output, stats) = run_request(&mut generator, &request)?;
                    request_elapsed += started.elapsed();
                    require_same_output(&reference, &output)?;
                    require_k4(stats)?;
                } else {
                    let (elapsed, stats) = run_round(&mut generator, &request)?;
                    round_elapsed += elapsed;
                    require_k4(stats)?;
                    if Some(stats.verified_outputs) != round_outputs {
                        return Err(DeviceBenchmarkError::Precondition(
                            "Qwen3.5 MTP committed width changed between samples".to_string(),
                        ));
                    }
                }
            }
        }
        request_samples
            .push(request_elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
        round_samples
            .push(round_elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
    }
    let telemetry = sampler.finish()?;
    require_current_process_exclusive()?;
    let prompt_tokens = reference.prompt.token_ids.len() as u64;
    let output_tokens = reference.token_ids.len() as u64;
    let committed = round_outputs.ok_or_else(|| {
        DeviceBenchmarkError::Precondition("Qwen3.5 MTP round was not warmed".to_string())
    })? as u64;
    let round_metric = host_completion_metric(
        "qwen3_5/generation/mtp_greedy_round",
        format!(
            "B=1,context={},draft=3,K=4,committed={committed}",
            prompt_tokens + 4
        ),
        BenchmarkWorkload::warm_model_mtp_round(committed, prompt_tokens + 4),
        options.launches_per_sample,
        0,
        round_samples,
    )?;
    let request_metric = host_completion_metric(
        "qwen3_5/generation/mtp_greedy_request",
        format!("B=1,prompt={prompt_tokens},output={output_tokens},draft=3"),
        BenchmarkWorkload::warm_model_mtp_request(
            prompt_tokens,
            output_tokens,
            prompt_tokens + output_tokens - 1,
        ),
        options.launches_per_sample,
        0,
        request_samples,
    )?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-mtp-generation",
            classification: "performance_sensitive_model",
            timing_scope: "direct Rust host completion for Qwen3.5 prompt prime, draft-three, exact-K target verification/commit, MTP realignment, and streaming control",
        },
        preflight,
        baseline_sha256,
        options,
        vec![round_metric, request_metric],
        Vec::new(),
        telemetry,
        memory,
    )
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

fn run_request(
    generator: &mut Qwen35ResidentMtpTextGenerator,
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
    generator: &mut Qwen35ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(Duration, ResidentMtpGenerationStats), DeviceBenchmarkError> {
    let mut session = generator.start(request)?;
    let _ = session.step()?;
    let started = Instant::now();
    let _ = session.step()?;
    let elapsed = started.elapsed();
    Ok((elapsed, session.stats()))
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
            "Qwen3.5 MTP output changed between samples".to_string(),
        ));
    }
    Ok(())
}

fn require_k4(stats: ResidentMtpGenerationStats) -> Result<(), DeviceBenchmarkError> {
    if stats.verification_routes[3] == 0 || stats.draft_proposals < 3 {
        return Err(DeviceBenchmarkError::Precondition(
            "Qwen3.5 benchmark did not execute a draft-three/K=4 round".to_string(),
        ));
    }
    Ok(())
}

fn register_memory(
    memory: &mut MemoryRecorder,
    generator: &Qwen35ResidentMtpTextGenerator,
) -> Result<(), DeviceBenchmarkError> {
    let layout = generator.qualification_program().layout();
    for (name, kind, bytes, description) in [
        (
            "qwen35_generation_mtp/weights",
            BenchmarkMemoryKind::Weights,
            layout.resident_weight_bytes(),
            "32 target layers, one BF16 MTP layer, and one shared BF16 endpoint",
        ),
        (
            "qwen35_generation_mtp/cache",
            BenchmarkMemoryKind::KvCache,
            layout.cache_bytes(),
            "target and mirrored MTP 262,144-position BF16 K/V pools",
        ),
        (
            "qwen35_generation_mtp/workspace",
            BenchmarkMemoryKind::Workspace,
            layout.workspace_bytes(),
            "target/MTP working planes plus exact device-resident GDN snapshots",
        ),
        (
            "qwen35_generation_mtp/padding",
            BenchmarkMemoryKind::Other,
            layout.padding_bytes(),
            "256-byte arena alignment",
        ),
    ] {
        memory.register_owned(name, kind, bytes, description)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::OUTPUT_TOKENS;

    #[test]
    fn qwen35_mtp_generation_suite_benchmark_uses_one_complete_k4_request() {
        assert_eq!(OUTPUT_TOKENS, 8);
    }
}

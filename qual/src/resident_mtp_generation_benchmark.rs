//! Direct host-completion timing for one production greedy MTP request.

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
    ChatGenerationRequest, GeneratedText, ResidentMtpGreedyStats, ResidentMtpTextGenerator,
    SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const OUTPUT_TOKENS: usize = 8;

/// Measures complete frontend-to-target/MTP greedy requests without summing leaf medians.
pub fn benchmark_resident_mtp_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.batch_size.is_some_and(|batch| batch != 1) {
        return Err(DeviceBenchmarkError::Precondition(
            "resident greedy MTP generation benchmark admits only B=1".to_string(),
        ));
    }
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "resident greedy MTP request energy is deferred to the full-server gate".to_string(),
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
    let request = request();
    let mut reference = None;
    let mut round_outputs = None;
    for _ in 0..warmup {
        let (output, stats, _) = run_request(&mut generator, &request)?;
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
                    "resident greedy MTP round committed count changed during warmup".to_string(),
                ));
            }
            None => round_outputs = Some(stats.verified_outputs),
            Some(_) => {}
        }
    }
    let reference = reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "resident greedy MTP benchmark requires at least one warmup request".to_string(),
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
                    let (output, stats, _) = run_request(&mut generator, &request)?;
                    request_elapsed += started.elapsed();
                    require_same_output(&reference, &output)?;
                    require_k4(stats)?;
                } else {
                    let (elapsed, stats) = run_round(&mut generator, &request)?;
                    round_elapsed += elapsed;
                    require_k4(stats)?;
                    if Some(stats.verified_outputs) != round_outputs {
                        return Err(DeviceBenchmarkError::Precondition(
                            "resident greedy MTP round committed count changed between samples"
                                .to_string(),
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
    let context_tokens = prompt_tokens + output_tokens - 1;
    let committed = round_outputs.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(
            "resident greedy MTP benchmark did not warm one speculative round".to_string(),
        )
    })? as u64;
    let round_metric = host_completion_metric(
        "qwen3_8/generation/mtp_greedy_round",
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
        "qwen3_8/generation/mtp_greedy_request",
        format!("B=1,prompt={prompt_tokens},output={output_tokens},draft=3"),
        BenchmarkWorkload::warm_model_mtp_request(prompt_tokens, output_tokens, context_tokens),
        options.launches_per_sample,
        0,
        request_samples,
    )?;
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-generation-mtp-greedy",
            classification: "performance_sensitive_model",
            timing_scope: "direct Rust host completion for prompt prime, greedy draft-three, target verify/commit, MTP realignment, and streaming control through the production owner",
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
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(GeneratedText, ResidentMtpGreedyStats, usize), DeviceBenchmarkError> {
    let mut session = generator.start(request)?;
    let mut steps = 0;
    while session.finish_reason().is_none() {
        let _ = session.step()?;
        steps += 1;
    }
    let stats = session.stats();
    Ok((session.into_output()?, stats, steps))
}

fn run_round(
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(Duration, ResidentMtpGreedyStats), DeviceBenchmarkError> {
    let mut session = generator.start(request)?;
    let _anchor = session.step()?;
    let started = Instant::now();
    let _first_committed = session.step()?;
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
            "resident greedy MTP benchmark output changed between samples".to_string(),
        ));
    }
    Ok(())
}

fn require_k4(stats: ResidentMtpGreedyStats) -> Result<(), DeviceBenchmarkError> {
    if stats.verification_routes[3] == 0 || stats.draft_proposals < 3 {
        return Err(DeviceBenchmarkError::Precondition(
            "resident greedy MTP benchmark did not execute a draft-three/K=4 round".to_string(),
        ));
    }
    Ok(())
}

fn register_memory(
    memory: &mut MemoryRecorder,
    generator: &ResidentMtpTextGenerator,
) -> Result<(), DeviceBenchmarkError> {
    let program = generator.qualification_program();
    let target = program.target();
    for (name, kind, bytes, description) in [
        (
            "generation_mtp/target_weights",
            BenchmarkMemoryKind::Weights,
            target.resident_weight_bytes(),
            "64 exact target layers plus shared final norm and LM head",
        ),
        (
            "generation_mtp/target_gdn_history",
            BenchmarkMemoryKind::Other,
            target.history_bytes(),
            "48 layers * 8 persistent causal-history slots",
        ),
        (
            "generation_mtp/target_gdn_state",
            BenchmarkMemoryKind::Other,
            target.state_bytes(),
            "48 layers * 8 persistent recurrent-state slots",
        ),
        (
            "generation_mtp/target_kv_cache",
            BenchmarkMemoryKind::KvCache,
            target.cache_bytes(),
            "16 target layers sharing the exact 3,438-page pool",
        ),
        (
            "generation_mtp/target_kv_tables",
            BenchmarkMemoryKind::Other,
            target.kv_table_bytes(),
            "8 target slot rows * 3,438 page entries",
        ),
        (
            "generation_mtp/target_workspace",
            BenchmarkMemoryKind::Workspace,
            target.workspace_bytes(),
            "target address-stable decode, prefill, and verification workspace",
        ),
        (
            "generation_mtp/target_tensor_maps",
            BenchmarkMemoryKind::Other,
            target.descriptor_bytes(),
            "eight dense target layers * four address-bound tensor maps",
        ),
        (
            "generation_mtp/target_padding",
            BenchmarkMemoryKind::Other,
            target.padding_bytes(),
            "target resident and KV arena alignment",
        ),
        (
            "generation_mtp/mtp_weights",
            BenchmarkMemoryKind::Weights,
            program.resident_weight_bytes(),
            "one unchanged source-BF16 MTP weight set sharing the target endpoint",
        ),
        (
            "generation_mtp/mtp_kv_cache",
            BenchmarkMemoryKind::KvCache,
            program.cache_bytes(),
            "one BF16 MTP K/V mirror using the target page lifecycle",
        ),
        (
            "generation_mtp/mtp_workspace",
            BenchmarkMemoryKind::Workspace,
            program.workspace_bytes(),
            "prompt, continuation, verification realignment, and LM-head seams",
        ),
        (
            "generation_mtp/mtp_padding",
            BenchmarkMemoryKind::Other,
            program.padding_bytes(),
            "two 256-byte-aligned MTP arenas",
        ),
    ] {
        memory.register_owned(name, kind, bytes, description)?;
    }
    Ok(())
}

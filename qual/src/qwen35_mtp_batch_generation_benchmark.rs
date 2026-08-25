//! Direct host-completion timing for exact compact Qwen3.5 MTP transactions.

use crate::device_benchmark::{
    BenchmarkMemoryKind, BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError,
    DeviceBenchmarkOptions, DeviceBenchmarkReport, MemoryRecorder, TelemetrySampler, finish_report,
    generator_baseline_sha256, host_completion_metric, measurement_order, preflight,
    require_current_process_exclusive, validate_loaded_host_clock_policy, warmup_launches,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{ChatGenerationRequest, Qwen35ResidentMtpBatchGenerator, SamplingOptions};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};
use tuisko_model::{CheckpointSnapshot, Qwen35_9B};

const OUTPUT_TOKENS: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RoundOutcome {
    committed: usize,
    token_ids: Vec<Vec<u32>>,
}

/// Measures singleton MTP and every selected compact B=2..8 target-alignment transaction.
pub fn benchmark_qwen35_mtp_batch_generation(
    root: &Path,
    options: DeviceBenchmarkOptions,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            "compact Qwen3.5 MTP energy belongs to the full-server gate".to_string(),
        ));
    }
    let batches = selected_batches(options.batch_size)?;
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
    let mut generator = Qwen35ResidentMtpBatchGenerator::from_snapshot(&context, snapshot)?;
    register_memory(&mut memory, &generator)?;
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
                        "Qwen3.5 compact MTP B={batch} prompt length changed during warmup"
                    )));
                }
                None => prompt = Some(current_prompt),
                Some(_) => {}
            }
        }
        references.push(reference.ok_or_else(|| {
            DeviceBenchmarkError::Precondition(format!(
                "Qwen3.5 compact MTP B={batch} requires at least one warmup"
            ))
        })?);
        prompt_tokens.push(prompt.expect("warmup established prompt length"));
    }
    memory.capture("after_warmup")?;
    require_current_process_exclusive()?;

    let loaded_batch = *batches
        .last()
        .expect("selected batch inventory is nonempty");
    let loaded_reference = references
        .last()
        .expect("selected batch reference exists")
        .clone();
    validate_loaded_host_clock_policy("qwen3_5/generation/mtp_batch_round", || {
        let (_, outcome, _) = run_round(&mut generator, &request, loaded_batch)?;
        require_reference(loaded_batch, Some(&loaded_reference), &outcome)
    })?;

    let sampler = TelemetrySampler::start();
    let mut samples = batches
        .iter()
        .map(|_| Vec::with_capacity(options.samples))
        .collect::<Vec<_>>();
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
                        "Qwen3.5 compact MTP B={batch} prompt length changed between samples"
                    )));
                }
            }
            samples[case]
                .push(elapsed.as_secs_f64() * 1_000_000.0 / options.launches_per_sample as f64);
        }
    }
    let telemetry = sampler.finish()?;
    require_current_process_exclusive()?;

    let mut metrics = Vec::with_capacity(batches.len());
    for (case, &batch) in batches.iter().enumerate() {
        let committed = references[case].committed as u64;
        let prompt = prompt_tokens[case] as u64;
        let (mode, context) = if batch == 1 {
            ("mtp-draft-3,K=4", prompt + 4)
        } else {
            ("compact-target,K=1", prompt + 1)
        };
        metrics.push(host_completion_metric(
            "qwen3_5/generation/mtp_batch_round",
            format!("B={batch},context={context},{mode},committed={committed}"),
            BenchmarkWorkload::warm_model_mtp_batch_round(
                u32::try_from(batch).map_err(|_| {
                    DeviceBenchmarkError::Precondition(
                        "Qwen3.5 compact MTP batch exceeds u32".to_string(),
                    )
                })?,
                committed,
                context,
            ),
            options.launches_per_sample,
            0,
            std::mem::take(&mut samples[case]),
        )?);
    }
    let memory = memory.finish(&telemetry)?;
    finish_report(
        BenchmarkReportSpec {
            suite: "bench-qwen35-mtp-batch-generation",
            classification: "performance_sensitive_model",
            timing_scope: "direct Rust host completion for Qwen3.5 singleton draft-three verification and B=2..8 compact target decode plus aligned compact MTP continuation",
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

fn run_round(
    generator: &mut Qwen35ResidentMtpBatchGenerator,
    request: &ChatGenerationRequest,
    batch: usize,
) -> Result<(Duration, RoundOutcome, usize), DeviceBenchmarkError> {
    let mut prompt_tokens = None;
    for _ in 0..batch {
        let admission = generator.admit(request)?;
        match prompt_tokens {
            Some(expected) if expected != admission.prompt_tokens => {
                return Err(DeviceBenchmarkError::Precondition(
                    "Qwen3.5 compact MTP lanes have different prompt lengths".to_string(),
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
            "Qwen3.5 compact MTP B={batch} setup did not stop at one anchor per lane"
        )));
    }

    let started = Instant::now();
    let events = generator.step()?;
    let elapsed = started.elapsed();
    if events.len() != batch {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "Qwen3.5 compact MTP B={batch} returned {} timed events",
            events.len()
        )));
    }
    let mut token_ids = Vec::with_capacity(batch);
    let mut committed = 0;
    for event in events.iter() {
        let expected_route = if batch == 1 { 3 } else { 0 };
        if event.stats.verification_routes[expected_route] != 1
            || (batch == 1 && event.stats.draft_proposals < 3)
        {
            return Err(DeviceBenchmarkError::Precondition(format!(
                "Qwen3.5 scheduler B={batch} did not execute its exact route"
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
    let expected_committed = if batch == 1 { 4 } else { batch };
    if actual.committed != expected_committed {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "Qwen3.5 scheduler B={batch} committed {}, expected {expected_committed}",
            actual.committed,
        )));
    }
    if let Some(expected) = expected
        && expected != actual
    {
        return Err(DeviceBenchmarkError::Precondition(format!(
            "Qwen3.5 compact MTP B={batch} output changed between replays"
        )));
    }
    Ok(())
}

fn selected_batches(batch_size: Option<u32>) -> Result<Vec<usize>, DeviceBenchmarkError> {
    match batch_size {
        None => Ok((1..=8).collect()),
        Some(batch @ 1..=8) => Ok(vec![usize::try_from(batch).map_err(|_| {
            DeviceBenchmarkError::Precondition(
                "Qwen3.5 compact MTP batch exceeds host width".to_string(),
            )
        })?]),
        Some(batch) => Err(DeviceBenchmarkError::Precondition(format!(
            "Qwen3.5 compact MTP batch {batch} is outside 1..=8"
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

fn register_memory(
    memory: &mut MemoryRecorder,
    generator: &Qwen35ResidentMtpBatchGenerator,
) -> Result<(), DeviceBenchmarkError> {
    let layout = generator.qualification_program().layout();
    for (name, kind, bytes, description) in [
        (
            "qwen35_generation_mtp_batch/weights",
            BenchmarkMemoryKind::Weights,
            layout.resident_weight_bytes(),
            "32 target layers, one BF16 MTP layer, and one shared BF16 endpoint",
        ),
        (
            "qwen35_generation_mtp_batch/cache",
            BenchmarkMemoryKind::KvCache,
            layout.cache_bytes(),
            "target and mirrored MTP 262,144-position BF16 K/V pools",
        ),
        (
            "qwen35_generation_mtp_batch/workspace",
            BenchmarkMemoryKind::Workspace,
            layout.workspace_bytes(),
            "target/MTP workspaces plus exact device-resident GDN rollback snapshots",
        ),
        (
            "qwen35_generation_mtp_batch/padding",
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
    #[test]
    fn qwen35_compact_mtp_benchmark_inventory_is_exact() {
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

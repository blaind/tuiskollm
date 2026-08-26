//! Generic host-completion benchmark session for the greedy MTP generation suites.
//!
//! Part V §3.B: the session owns the warmup, reference-identity, and paired-sample mechanics
//! shared by the Qwen3.5 and Qwen3.8 greedy MTP request benchmarks. Everything that identifies a
//! measurement — metric route names, report suite/classification/timing scope, refusal texts, the
//! generated output width, and every count carried in `DeviceBenchmarkOptions` — is bound per
//! suite through `MtpGreedyBenchmarkSpec` and never normalized here.
//!
//! The generator bound is sealed (Part I §3 Bound C): only the two production MTP text
//! generators implement it, and the trait drives their sessions internally so no borrowed
//! session type crosses the bound.

use crate::device_benchmark::{
    BenchmarkReportSpec, BenchmarkWorkload, DeviceBenchmarkError, DeviceBenchmarkOptions,
    DeviceBenchmarkReport, DevicePreflight, MemoryRecorder, TelemetrySampler, finish_report,
    generator_baseline_sha256, host_completion_metric, preflight,
    require_current_process_exclusive, warmup_launches,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tuisko_engine::{
    ChatGenerationRequest, GeneratedText, Qwen35ResidentMtpTextGenerator,
    ResidentMtpGenerationStats, ResidentMtpTextGenerator, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError};

mod private {
    pub(crate) trait Sealed {}

    impl Sealed for super::Qwen35ResidentMtpTextGenerator {}
    impl Sealed for super::ResidentMtpTextGenerator {}
}

/// Production MTP text generator a greedy host-completion benchmark drives.
///
/// Both methods start their own session so the borrow of `self` ends with the call; the session
/// type never appears in the bound.
pub(crate) trait MtpGreedyGenerator: private::Sealed {
    /// Runs one complete request to its finish reason.
    fn run_request(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<(GeneratedText, ResidentMtpGenerationStats), DeviceBenchmarkError>;

    /// Primes one anchor step, then times exactly one speculative round.
    fn run_round(
        &mut self,
        request: &ChatGenerationRequest,
    ) -> Result<(Duration, ResidentMtpGenerationStats), DeviceBenchmarkError>;
}

/// Both generators expose the same session surface; the sessions differ only in type.
macro_rules! impl_mtp_greedy_generator {
    ($generator:ty) => {
        impl MtpGreedyGenerator for $generator {
            fn run_request(
                &mut self,
                request: &ChatGenerationRequest,
            ) -> Result<(GeneratedText, ResidentMtpGenerationStats), DeviceBenchmarkError> {
                let mut session = self.start(request)?;
                while session.finish_reason().is_none() {
                    let _ = session.step()?;
                }
                let stats = session.stats();

                Ok((session.into_output()?, stats))
            }

            fn run_round(
                &mut self,
                request: &ChatGenerationRequest,
            ) -> Result<(Duration, ResidentMtpGenerationStats), DeviceBenchmarkError> {
                let mut session = self.start(request)?;
                let _anchor = session.step()?;
                let started = Instant::now();
                let _first_committed = session.step()?;
                let elapsed = started.elapsed();

                Ok((elapsed, session.stats()))
            }
        }
    };
}

impl_mtp_greedy_generator!(Qwen35ResidentMtpTextGenerator);
impl_mtp_greedy_generator!(ResidentMtpTextGenerator);

/// Per-suite measurement identity for one greedy MTP host-completion benchmark.
///
/// Every field is transcribed from the suite it replaces; none of them may be shared or
/// defaulted across suites.
pub(crate) struct MtpGreedyBenchmarkSpec {
    /// Refusal raised when a batch size other than one is requested.
    pub(crate) batch_refusal: &'static str,
    /// Refusal raised when request energy is asked of a single-request benchmark.
    pub(crate) energy_refusal: &'static str,
    /// Refusal raised when the committed round width moves during warmup.
    pub(crate) warmup_width_drift: &'static str,
    /// Refusal raised when the committed round width moves between samples.
    pub(crate) sample_width_drift: &'static str,
    /// Refusal raised when no warmup request established a reference output.
    pub(crate) missing_warmup_request: &'static str,
    /// Refusal raised when no warmup round established a committed width.
    pub(crate) unwarmed_round: &'static str,
    /// Refusal raised when a measured request disagrees with the reference output.
    pub(crate) output_drift: &'static str,
    /// Refusal raised when a round did not execute a draft-three/K=4 verification.
    pub(crate) k4_refusal: &'static str,
    /// Metric route name for the timed speculative round.
    pub(crate) round_metric: &'static str,
    /// Metric route name for the timed complete request.
    pub(crate) request_metric: &'static str,
    /// Report suite name carried into `finish_report`.
    pub(crate) suite: &'static str,
    /// Report classification carried into `finish_report`.
    pub(crate) classification: &'static str,
    /// Report timing scope carried into `finish_report`.
    pub(crate) timing_scope: &'static str,
}

/// Evidence established before a greedy MTP benchmark opens its checkpoint.
pub(crate) struct MtpGreedyPreamble {
    /// Generator baseline digest carried into the report.
    baseline_sha256: String,
    /// Admitted warmup request count for this run.
    warmup: u64,
    /// Device identity and capacity evidence for the session.
    preflight: DevicePreflight,
    /// Recorder the suite registers its owned regions with.
    pub(crate) memory: MemoryRecorder,
}

/// Refuses unmeasurable option shapes, then establishes the pre-checkpoint session evidence.
///
/// The order — admission, baseline digest, warmup admission, preflight, recorder — is the order
/// each migrated suite used, so the first refusal a bad invocation sees is unchanged.
pub(crate) fn admit_greedy_mtp_benchmark(
    options: DeviceBenchmarkOptions,
    spec: &MtpGreedyBenchmarkSpec,
) -> Result<MtpGreedyPreamble, DeviceBenchmarkError> {
    if options.batch_size.is_some_and(|batch| batch != 1) {
        return Err(DeviceBenchmarkError::Precondition(
            spec.batch_refusal.to_string(),
        ));
    }
    if options.energy_seconds.is_some() {
        return Err(DeviceBenchmarkError::Precondition(
            spec.energy_refusal.to_string(),
        ));
    }
    let baseline_sha256 = generator_baseline_sha256()?;
    let warmup = warmup_launches(options)?;
    let preflight = preflight()?;
    let memory = MemoryRecorder::new(&preflight)?;

    Ok(MtpGreedyPreamble {
        baseline_sha256,
        warmup,
        preflight,
        memory,
    })
}

/// Opens device zero and refuses anything other than compute capability 12.0.
pub(crate) fn open_device_zero() -> Result<Arc<CudaContext>, DeviceBenchmarkError> {
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(DeviceBenchmarkError::Precondition(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }

    Ok(context)
}

/// Builds the greedy single-turn request both suites measure.
pub(crate) fn greedy_request(output_tokens: usize) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = output_tokens;
    request
}

/// Warms, measures, and reports one greedy MTP generation benchmark.
///
/// The caller has already admitted `options` through `admit_greedy_mtp_benchmark`, opened the
/// snapshot, built `generator`, and registered its owned memory regions. This runs the warmup
/// identity checks, the alternating request/round samples, and the report construction. Sample
/// ordering alternates `[0, 1]` and `[1, 0]` by sample index exactly as the suites did before the
/// migration.
pub(crate) fn run_greedy_mtp_benchmark<G: MtpGreedyGenerator>(
    generator: &mut G,
    request: &ChatGenerationRequest,
    preamble: MtpGreedyPreamble,
    options: DeviceBenchmarkOptions,
    spec: &MtpGreedyBenchmarkSpec,
) -> Result<DeviceBenchmarkReport, DeviceBenchmarkError> {
    let MtpGreedyPreamble {
        baseline_sha256,
        warmup,
        preflight,
        mut memory,
    } = preamble;
    let mut reference = None;
    let mut round_outputs = None;
    for _ in 0..warmup {
        let (output, stats) = generator.run_request(request)?;
        require_k4(stats, spec)?;
        if let Some(expected) = &reference {
            require_same_output(expected, &output, spec)?;
        } else {
            reference = Some(output);
        }
        let (_, stats) = generator.run_round(request)?;
        require_k4(stats, spec)?;
        match round_outputs {
            Some(expected) if expected != stats.verified_outputs => {
                return Err(DeviceBenchmarkError::Precondition(
                    spec.warmup_width_drift.to_string(),
                ));
            }
            None => round_outputs = Some(stats.verified_outputs),
            Some(_) => {}
        }
    }
    let reference = reference.ok_or_else(|| {
        DeviceBenchmarkError::Precondition(spec.missing_warmup_request.to_string())
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
                    let (output, stats) = generator.run_request(request)?;
                    request_elapsed += started.elapsed();
                    require_same_output(&reference, &output, spec)?;
                    require_k4(stats, spec)?;
                } else {
                    let (elapsed, stats) = generator.run_round(request)?;
                    round_elapsed += elapsed;
                    require_k4(stats, spec)?;
                    if Some(stats.verified_outputs) != round_outputs {
                        return Err(DeviceBenchmarkError::Precondition(
                            spec.sample_width_drift.to_string(),
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
    let committed = round_outputs
        .ok_or_else(|| DeviceBenchmarkError::Precondition(spec.unwarmed_round.to_string()))?
        as u64;
    let round_metric = host_completion_metric(
        spec.round_metric,
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
        spec.request_metric,
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
            suite: spec.suite,
            classification: spec.classification,
            timing_scope: spec.timing_scope,
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

fn require_same_output(
    expected: &GeneratedText,
    actual: &GeneratedText,
    spec: &MtpGreedyBenchmarkSpec,
) -> Result<(), DeviceBenchmarkError> {
    if actual.prompt.token_ids != expected.prompt.token_ids
        || actual.token_ids != expected.token_ids
        || actual.text != expected.text
        || actual.finish_reason != expected.finish_reason
    {
        return Err(DeviceBenchmarkError::Precondition(
            spec.output_drift.to_string(),
        ));
    }

    Ok(())
}

fn require_k4(
    stats: ResidentMtpGenerationStats,
    spec: &MtpGreedyBenchmarkSpec,
) -> Result<(), DeviceBenchmarkError> {
    if stats.verification_routes[3] == 0 || stats.draft_proposals < 3 {
        return Err(DeviceBenchmarkError::Precondition(
            spec.k4_refusal.to_string(),
        ));
    }

    Ok(())
}

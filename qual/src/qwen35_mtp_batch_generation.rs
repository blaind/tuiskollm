//! Source-backed gate for compact Qwen3.5 target-plus-MTP generation.

use crate::{DeviceBenchmarkError, device_benchmark, qualify_speculative_sampling};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, GeneratedText,
    Qwen35ResidentMtpBatchGenerator, Qwen35ResidentTextGenerator, ResidentMtpGenerationStats,
    SamplingOptions, SamplingPenalties,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen35_9B};

const LIMIT_CASES: [usize; 4] = [2, 3, 4, 8];

/// Failure of compact Qwen3.5 target-plus-MTP qualification.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35MtpBatchQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Exact tokenizer or chat-template admission failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An exact scheduler boundary differed.
    #[error("Qwen3.5 compact MTP qualification failed: {0}")]
    Mismatch(String),
}

/// Exact-batch, target-agreement, rollback, and slot-lifecycle evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MtpBatchQualification {
    /// Exact compact draft widths B=1..8 exercised.
    pub route_batches: usize,
    /// Independent target-only outputs compared with MTP outputs.
    pub target_agreement_cases: usize,
    /// K=1,2,3,4 target verification routes observed.
    pub verification_routes: [usize; 4],
    /// Draft proposals evaluated across all cases.
    pub draft_proposals: usize,
    /// Draft proposals licensed by the target.
    pub accepted_drafts: usize,
    /// A cancelled physical slot immediately reused by another prompt.
    pub recycled_slot: usize,
    /// Concurrent native-prefill tokens processed in the recycled slot.
    pub concurrent_prefill_tokens: usize,
    /// Fixed-seed sampled rollback cases replayed deterministically.
    pub sampled_cases: usize,
    /// Complete target, MTP, and mirror allocation bytes.
    pub device_owner_bytes: usize,
    /// Fixed page-locked scheduler staging bytes.
    pub host_stager_bytes: usize,
    /// Stable retained device and pinned-host addresses.
    pub stable_addresses: usize,
}

/// Qualifies compact MTP output, routing, cancellation, and rollback behavior.
pub fn qualify_qwen35_mtp_batch_generation(
    root: &Path,
) -> Result<Qwen35MtpBatchQualification, Qwen35MtpBatchQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    qualify_speculative_sampling().map_err(|error| {
        Qwen35MtpBatchQualificationError::Mismatch(format!(
            "independent speculative-sampling oracle failed: {error}"
        ))
    })?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }

    let requests = LIMIT_CASES.map(|maximum| greedy_request("Hello", maximum));
    let alternate = greedy_request("Name one color.", 8);
    let exact_prefill = exact_prompt_request(&frontend, 192)?;
    let mut target = Qwen35ResidentTextGenerator::from_snapshot(&context, Arc::clone(&snapshot))?;
    let mut expected = Vec::with_capacity(requests.len());
    for request in &requests {
        expected.push(run_target(&mut target, request)?);
    }
    let alternate_expected = run_target(&mut target, &alternate)?;
    let prefill_expected = run_target(&mut target, &exact_prefill)?;
    drop(target);

    let mut generator = Qwen35ResidentMtpBatchGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let addresses = generator.qualification_addresses()?;
    let unique = addresses
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if addresses.len() != 77 || unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "compact MTP owner has {}/{} retained/unique addresses, expected 77",
            addresses.len(),
            unique.len()
        )));
    }

    let _ = run_batch(&mut generator, &[requests[3].clone()])?;
    let before = device_memory_info(generator.context())?;
    let mut routes = [0usize; 4];
    let mut draft_proposals = 0;
    let mut accepted_drafts = 0;
    for (request, expected) in requests.iter().zip(&expected) {
        let actual = run_batch(&mut generator, std::slice::from_ref(request))?
            .pop()
            .expect("one compact result");
        compare_output(expected, &actual.output)?;
        accumulate_stats(
            &mut routes,
            &mut draft_proposals,
            &mut accepted_drafts,
            actual.stats,
        );
    }
    for batch in 1..=8 {
        let batch_requests = vec![requests[2].clone(); batch];
        let actual = run_batch(&mut generator, &batch_requests)?;
        if actual.len() != batch {
            return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
                "B={batch} returned {} complete outputs",
                actual.len()
            )));
        }
        for output in actual {
            compare_output(&expected[2], &output.output)?;
            accumulate_stats(
                &mut routes,
                &mut draft_proposals,
                &mut accepted_drafts,
                output.stats,
            );
        }
    }

    let recycled_slot = verify_hole_reuse(
        &mut generator,
        &requests[3],
        &expected[3],
        &alternate,
        &alternate_expected,
        &exact_prefill,
        &prefill_expected,
    )?;
    let penalties = SamplingPenalties {
        presence: 1.5,
        frequency: 0.5,
        repetition: 1.1,
    };
    let mut sampled_proposals = 0;
    let mut sampled_accepted = 0;
    let mut sampled_cases = 0;
    for seed in 1..=64 {
        let request = sampled_request(16, seed, penalties);
        let sampled_a = run_batch(&mut generator, std::slice::from_ref(&request))?;
        let sampled_b = run_batch(&mut generator, &[request])?;
        if sampled_a.len() != sampled_b.len()
            || sampled_a.iter().zip(&sampled_b).any(|(left, right)| {
                left.output.token_ids != right.output.token_ids || left.stats != right.stats
            })
        {
            return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
                "fixed-seed compact sampled execution differs for seed {seed}"
            )));
        }
        sampled_cases += sampled_a.len();
        sampled_proposals += sampled_a
            .iter()
            .map(|result| result.stats.draft_proposals)
            .sum::<usize>();
        sampled_accepted += sampled_a
            .iter()
            .map(|result| result.stats.accepted_drafts)
            .sum::<usize>();
        if sampled_accepted < sampled_proposals {
            break;
        }
    }
    if sampled_proposals == 0 || sampled_accepted >= sampled_proposals {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "sampled compact case did not exercise rollback: {sampled_accepted}/{sampled_proposals} accepted"
        )));
    }

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "device memory changed after compact MTP warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses()? != addresses {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "compact MTP owner addresses changed".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen35MtpBatchQualification {
        route_batches: 8,
        target_agreement_cases: LIMIT_CASES.len() + 8,
        verification_routes: routes,
        draft_proposals,
        accepted_drafts,
        recycled_slot,
        concurrent_prefill_tokens: 160,
        sampled_cases,
        device_owner_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        stable_addresses: addresses.len(),
    })
}

struct BatchResult {
    output: GeneratedText,
    stats: ResidentMtpGenerationStats,
}

fn run_batch(
    generator: &mut Qwen35ResidentMtpBatchGenerator,
    requests: &[ChatGenerationRequest],
) -> Result<Vec<BatchResult>, Qwen35MtpBatchQualificationError> {
    let mut order = Vec::with_capacity(requests.len());
    let mut streamed = BTreeMap::new();
    for request in requests {
        let admission = generator.admit(request)?;
        if admission.completed.is_some() {
            return Err(Qwen35MtpBatchQualificationError::Mismatch(
                "nonempty compact fixture completed at admission".to_string(),
            ));
        }
        order.push(admission.request_id);
        streamed.insert(admission.request_id.get(), String::new());
    }
    let mut completed = BTreeMap::new();
    while generator.active_requests() != 0 {
        let events = generator.step()?;
        if events.is_empty() || events.len() > requests.len() {
            return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
                "compact scheduler returned {} events for {} requests",
                events.len(),
                requests.len()
            )));
        }
        for event in events.iter() {
            if event.is_empty() || event.len() > 4 {
                return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
                    "request {} committed {} steps",
                    event.request_id.get(),
                    event.len()
                )));
            }
            let text = streamed.get_mut(&event.request_id.get()).ok_or_else(|| {
                Qwen35MtpBatchQualificationError::Mismatch(format!(
                    "unknown compact request {} produced an event",
                    event.request_id.get()
                ))
            })?;
            for step in event.steps() {
                if let Some(delta) = &step.delta {
                    text.push_str(delta);
                }
            }
            if let Some(output) = &event.completed {
                if text != &output.text || output.token_ids.len() < event.len() {
                    return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
                        "request {} streaming output differs",
                        event.request_id.get()
                    )));
                }
                completed.insert(
                    event.request_id.get(),
                    BatchResult {
                        output: output.clone(),
                        stats: event.stats,
                    },
                );
            }
        }
    }
    order
        .into_iter()
        .map(|request| {
            completed.remove(&request.get()).ok_or_else(|| {
                Qwen35MtpBatchQualificationError::Mismatch(format!(
                    "request {} has no terminal output",
                    request.get()
                ))
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn verify_hole_reuse(
    generator: &mut Qwen35ResidentMtpBatchGenerator,
    request: &ChatGenerationRequest,
    expected: &GeneratedText,
    alternate: &ChatGenerationRequest,
    alternate_expected: &GeneratedText,
    prefill: &ChatGenerationRequest,
    prefill_expected: &GeneratedText,
) -> Result<usize, Qwen35MtpBatchQualificationError> {
    let first = generator.admit(request)?;
    let cancelled = generator.admit(alternate)?;
    let third = generator.admit(request)?;
    if generator.qualification_slot(first.request_id) != Some(0)
        || generator.qualification_slot(cancelled.request_id) != Some(1)
        || generator.qualification_slot(third.request_id) != Some(2)
    {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "cold compact MTP admissions did not fill slots 0,1,2".to_string(),
        ));
    }
    let first_round = generator.step()?;
    if first_round.len() != 3 {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "three compact MTP anchors did not produce three events".to_string(),
        ));
    }
    let cancellation = generator.cancel(cancelled.request_id)?;
    if cancellation.device_retained_tokens != 0
        || cancellation.output.token_ids != alternate_expected.token_ids[..1]
    {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "Qwen3.5 MTP cancellation changed its round boundary".to_string(),
        ));
    }
    let joined = generator.admit(prefill)?;
    if joined.native_prefill_tokens != 160
        || generator.qualification_slot(joined.request_id) != Some(1)
    {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "concurrent T=192 admission selected native={} slot={:?}",
            joined.native_prefill_tokens,
            generator.qualification_slot(joined.request_id)
        )));
    }
    let mut expected_by_request = BTreeMap::from([
        (first.request_id.get(), expected),
        (third.request_id.get(), expected),
        (joined.request_id.get(), prefill_expected),
    ]);
    while generator.active_requests() != 0 {
        for event in generator.step()?.iter() {
            if let Some(output) = &event.completed {
                let expected = expected_by_request
                    .remove(&event.request_id.get())
                    .ok_or_else(|| {
                        Qwen35MtpBatchQualificationError::Mismatch(
                            "unexpected hole-reuse completion".to_string(),
                        )
                    })?;
                compare_output(expected, output)?;
            }
        }
    }
    if !expected_by_request.is_empty() {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(
            "hole-reuse requests did not all complete".to_string(),
        ));
    }
    Ok(1)
}

fn run_target(
    generator: &mut Qwen35ResidentTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<GeneratedText, Qwen35MtpBatchQualificationError> {
    let mut session = generator.start(request)?;
    while session.finish_reason().is_none() {
        let _ = session.step()?;
    }
    Ok(session.into_output()?)
}

fn compare_output(
    expected: &GeneratedText,
    actual: &GeneratedText,
) -> Result<(), Qwen35MtpBatchQualificationError> {
    if actual.prompt.token_ids != expected.prompt.token_ids
        || actual.token_ids != expected.token_ids
        || actual.text != expected.text
        || actual.finish_reason != expected.finish_reason
        || actual.finish_reason != FinishReason::Length
    {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "target/MTP output differs: target={:?}/{:?}, MTP={:?}/{:?}",
            expected.token_ids, expected.finish_reason, actual.token_ids, actual.finish_reason
        )));
    }
    Ok(())
}

fn accumulate_stats(
    routes: &mut [usize; 4],
    proposals: &mut usize,
    accepted: &mut usize,
    stats: ResidentMtpGenerationStats,
) {
    for (total, count) in routes.iter_mut().zip(stats.verification_routes) {
        *total += count;
    }
    *proposals += stats.draft_proposals;
    *accepted += stats.accepted_drafts;
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

fn sampled_request(
    maximum: usize,
    seed: u64,
    penalties: SamplingPenalties,
) -> ChatGenerationRequest {
    let mut request = greedy_request("Hello", maximum);
    request.sampling = SamplingOptions {
        seed,
        penalties,
        ..SamplingOptions::default()
    };
    request
}

fn exact_prompt_request(
    frontend: &TextFrontend,
    target_tokens: usize,
) -> Result<ChatGenerationRequest, Qwen35MtpBatchQualificationError> {
    let mut lower = 1usize;
    let mut upper = target_tokens;
    while lower < upper {
        let words = lower + (upper - lower) / 2;
        let request = greedy_request(&vec!["x"; words].join(" "), 4);
        let tokens = frontend
            .encode_chat(&request.messages, &request.template)?
            .len();
        if tokens < target_tokens {
            lower = words + 1;
        } else {
            upper = words;
        }
    }
    let request = greedy_request(&vec!["x"; lower].join(" "), 4);
    let actual = frontend
        .encode_chat(&request.messages, &request.template)?
        .len();
    if actual != target_tokens {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "could not construct exact T={target_tokens} prompt; fixture has T={actual}"
        )));
    }
    Ok(request)
}

fn verify_owner(
    generator: &Qwen35ResidentMtpBatchGenerator,
) -> Result<(), Qwen35MtpBatchQualificationError> {
    if generator.arena_bytes() != 17_253_733_120
        || generator.host_stager_bytes() != 30_433_280
        || generator.context_capacity() != 262_144
    {
        return Err(Qwen35MtpBatchQualificationError::Mismatch(format!(
            "compact MTP owner changed: device={} host={} context={}",
            generator.arena_bytes(),
            generator.host_stager_bytes(),
            generator.context_capacity()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::qualify_qwen35_mtp_batch_generation;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive SM120 device"]
    fn compact_scheduler_matches_target_and_reuses_rejected_slots()
    -> Result<(), super::Qwen35MtpBatchQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            super::Qwen35MtpBatchQualificationError::Mismatch(
                "set TUISKO_QWEN35_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_qwen35_mtp_batch_generation(&PathBuf::from(root))?;
        assert_eq!(report.route_batches, 8);
        assert_eq!(report.target_agreement_cases, 12);
        assert!(report.verification_routes.iter().all(|&routes| routes != 0));
        assert!(report.draft_proposals >= report.accepted_drafts);
        assert_eq!(report.recycled_slot, 1);
        assert_eq!(report.concurrent_prefill_tokens, 160);
        assert!((1..=64).contains(&report.sampled_cases));
        assert_eq!(report.device_owner_bytes, 17_253_733_120);
        assert_eq!(report.host_stager_bytes, 30_433_280);
        assert_eq!(report.stable_addresses, 77);
        eprintln!("Qwen3.5 compact MTP qualification passed: {report:?}");
        Ok(())
    }
}

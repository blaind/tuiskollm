//! Source-backed gate for exact single-slot Qwen3.5 MTP generation.

use crate::{
    DeviceBenchmarkError, device_benchmark, qualify_qwen35_resident_mtp,
    qualify_speculative_sampling,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, GeneratedText,
    Qwen35ResidentMtpTextGenerator, Qwen35ResidentTextGenerator, ResidentMtpGenerationStats,
    SamplingOptions, SamplingPenalties,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen35_9B};

const LIMIT_CASES: [usize; 4] = [2, 3, 4, 8];
const SAMPLED_CASES: [(usize, u64); 4] = [(2, 11), (3, 17), (4, 23), (8, 29)];

/// Failure of the exact Qwen3.5 target-plus-MTP generation gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35MtpGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An exact transaction boundary differed.
    #[error("Qwen3.5 MTP generation qualification failed: {0}")]
    Mismatch(String),
}

/// Exact target agreement, route, streaming, and owner evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MtpGenerationQualification {
    /// Independent resident target/MTP owner suite completed first.
    pub source_owner_suites: usize,
    /// Complete target-only outputs compared with MTP outputs.
    pub target_agreement_cases: usize,
    /// K=1,2,3,4 target verification routes observed.
    pub verification_routes: [usize; 4],
    /// Draft proposals evaluated across all cases.
    pub draft_proposals: usize,
    /// Draft proposals licensed by the target.
    pub accepted_drafts: usize,
    /// Streaming steps reassembled into complete outputs.
    pub streaming_steps: usize,
    /// Fixed-seed sampled cases replayed deterministically.
    pub sampled_cases: usize,
    /// Non-identity penalty-conditioned cases replayed deterministically.
    pub penalty_cases: usize,
    /// Complete target, MTP, and mirror allocation bytes.
    pub device_owner_bytes: usize,
    /// Fixed page-locked transaction staging bytes.
    pub host_stager_bytes: usize,
    /// Fixed target-plus-MTP host page-routing bytes.
    pub kv_route_host_bytes: usize,
}

/// Qualifies draft-three generation against the target-only Qwen3.5 route.
pub fn qualify_qwen35_mtp_generation(
    root: &Path,
) -> Result<Qwen35MtpGenerationQualification, Qwen35MtpGenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    qualify_speculative_sampling().map_err(|error| {
        Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "independent speculative-sampling oracle failed: {error}"
        ))
    })?;
    qualify_qwen35_resident_mtp(root).map_err(|error| {
        Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "independent resident owner suite failed: {error}"
        ))
    })?;

    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let requests = LIMIT_CASES.map(greedy_request);
    let mut target = Qwen35ResidentTextGenerator::from_snapshot(&context, Arc::clone(&snapshot))?;
    let mut expected = Vec::with_capacity(LIMIT_CASES.len());
    for request in &requests {
        expected.push(run_target(&mut target, request)?);
    }
    drop(target);

    let mut generator = Qwen35ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let addresses = generator.qualification_addresses();
    let unique = addresses
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if addresses.len() != 38 || unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "Qwen3.5 MTP generation has {}/{} retained/unique addresses, expected 38",
            addresses.len(),
            unique.len()
        )));
    }

    let _ = run_mtp(&mut generator, &requests[3])?;
    let before = device_memory_info(generator.context())?;
    let mut routes = [0usize; 4];
    let mut draft_proposals = 0;
    let mut accepted_drafts = 0;
    let mut streaming_steps = 0;
    for (index, (request, expected)) in requests.iter().zip(&expected).enumerate() {
        let (actual, stats, steps) = run_mtp(&mut generator, request)?;
        compare_output(LIMIT_CASES[index], expected, &actual)?;
        let required_route = LIMIT_CASES[index].saturating_sub(2).min(3);
        if stats.verification_routes[required_route] == 0 {
            return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
                "max_new_tokens={} did not select target verification K={}",
                LIMIT_CASES[index],
                required_route + 1
            )));
        }
        for (total, count) in routes.iter_mut().zip(stats.verification_routes) {
            *total += count;
        }
        draft_proposals += stats.draft_proposals;
        accepted_drafts += stats.accepted_drafts;
        streaming_steps += steps;
    }
    let mut sampled_proposals = 0;
    let mut sampled_accepted = 0;
    for (maximum, seed) in SAMPLED_CASES {
        let request = sampled_request(maximum, seed, SamplingPenalties::identity());
        let first = run_mtp(&mut generator, &request)?;
        let second = run_mtp(&mut generator, &request)?;
        compare_seeded(maximum, &first, &second)?;
        sampled_proposals += first.1.draft_proposals;
        sampled_accepted += first.1.accepted_drafts;
    }
    let penalty_request = sampled_request(
        8,
        37,
        SamplingPenalties {
            presence: 1.5,
            frequency: 0.5,
            repetition: 1.1,
        },
    );
    let penalized_a = run_mtp(&mut generator, &penalty_request)?;
    let penalized_b = run_mtp(&mut generator, &penalty_request)?;
    compare_seeded(8, &penalized_a, &penalized_b)?;
    sampled_proposals += penalized_a.1.draft_proposals;
    sampled_accepted += penalized_a.1.accepted_drafts;
    if sampled_proposals == 0 || sampled_accepted >= sampled_proposals {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "sampled cases did not exercise rejection and rollback: {sampled_accepted}/{sampled_proposals} accepted"
        )));
    }
    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != addresses {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(
            "retained Qwen3.5 MTP generation addresses changed".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen35MtpGenerationQualification {
        source_owner_suites: 1,
        target_agreement_cases: LIMIT_CASES.len(),
        verification_routes: routes,
        draft_proposals,
        accepted_drafts,
        streaming_steps,
        sampled_cases: SAMPLED_CASES.len(),
        penalty_cases: 1,
        device_owner_bytes: generator.device_owner_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        kv_route_host_bytes: generator.kv_route_host_bytes(),
    })
}

fn greedy_request(maximum_new_tokens: usize) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = maximum_new_tokens;
    request
}

fn sampled_request(
    maximum_new_tokens: usize,
    seed: u64,
    penalties: SamplingPenalties,
) -> ChatGenerationRequest {
    let mut request = greedy_request(maximum_new_tokens);
    request.sampling = SamplingOptions {
        seed,
        penalties,
        ..SamplingOptions::default()
    };
    request
}

fn run_target(
    generator: &mut Qwen35ResidentTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<GeneratedText, Qwen35MtpGenerationQualificationError> {
    let mut session = generator.start(request)?;
    while session.finish_reason().is_none() {
        let _ = session.step()?;
    }
    Ok(session.into_output()?)
}

fn run_mtp(
    generator: &mut Qwen35ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(GeneratedText, ResidentMtpGenerationStats, usize), Qwen35MtpGenerationQualificationError>
{
    let mut session = generator.start(request)?;
    let mut streamed = String::new();
    let mut steps = 0;
    while session.finish_reason().is_none() {
        let step = session.step()?;
        if let Some(delta) = step.delta {
            streamed.push_str(&delta);
        }
        steps += 1;
    }
    let stats = session.stats();
    let output = session.into_output()?;
    if streamed != output.text || steps != output.token_ids.len() {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(
            "streaming deltas did not reassemble the complete output".to_string(),
        ));
    }
    Ok((output, stats, steps))
}

fn compare_output(
    limit: usize,
    expected: &GeneratedText,
    actual: &GeneratedText,
) -> Result<(), Qwen35MtpGenerationQualificationError> {
    if actual.prompt.token_ids != expected.prompt.token_ids
        || actual.token_ids != expected.token_ids
        || actual.text != expected.text
        || actual.finish_reason != expected.finish_reason
        || actual.finish_reason != FinishReason::Length
    {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "max_new_tokens={limit} differs: target={:?}/{:?}, MTP={:?}/{:?}",
            expected.token_ids, expected.finish_reason, actual.token_ids, actual.finish_reason
        )));
    }
    Ok(())
}

fn compare_seeded(
    maximum: usize,
    first: &(GeneratedText, ResidentMtpGenerationStats, usize),
    second: &(GeneratedText, ResidentMtpGenerationStats, usize),
) -> Result<(), Qwen35MtpGenerationQualificationError> {
    if first.0.prompt.token_ids != second.0.prompt.token_ids
        || first.0.token_ids != second.0.token_ids
        || first.0.text != second.0.text
        || first.0.finish_reason != second.0.finish_reason
        || first.1 != second.1
        || first.2 != second.2
    {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "seeded max_new_tokens={maximum} replay differed: first={:?}/{:?}, second={:?}/{:?}",
            first.0.token_ids, first.1, second.0.token_ids, second.1
        )));
    }
    Ok(())
}

fn verify_owner(
    generator: &Qwen35ResidentMtpTextGenerator,
) -> Result<(), Qwen35MtpGenerationQualificationError> {
    if generator.device_owner_bytes() != 17_253_733_120
        || generator.host_stager_bytes() != 4_678_656
        || generator.kv_route_host_bytes() != 270_336
        || generator.context_capacity() != 262_144
    {
        return Err(Qwen35MtpGenerationQualificationError::Mismatch(format!(
            "owner changed: device={}, host={}, routes={}, capacity={}",
            generator.device_owner_bytes(),
            generator.host_stager_bytes(),
            generator.kv_route_host_bytes(),
            generator.context_capacity()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LIMIT_CASES, SAMPLED_CASES};
    use std::path::Path;

    #[test]
    fn qwen35_mtp_generation_inventory_selects_every_k() {
        assert_eq!(LIMIT_CASES, [2, 3, 4, 8]);
        assert_eq!(LIMIT_CASES.map(|limit| (limit - 1).min(4)), [1, 2, 3, 4]);
        assert_eq!(
            SAMPLED_CASES.map(|(limit, _)| (limit - 1).min(4)),
            [1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "requires the admitted Qwen3.5 snapshot and an exclusive RTX 5090"]
    fn qwen35_mtp_generation_matches_target_only_greedy() {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT")
            .expect("TUISKO_QWEN35_SNAPSHOT must name the admitted Qwen3.5 snapshot");
        let report = super::qualify_qwen35_mtp_generation(Path::new(&root)).unwrap();
        assert_eq!(report.source_owner_suites, 1);
        assert_eq!(report.target_agreement_cases, 4);
        assert!(report.verification_routes.iter().all(|&routes| routes > 0));
        assert!(report.draft_proposals > 0);
        assert!(report.streaming_steps >= 8);
        assert_eq!(report.sampled_cases, 4);
        assert_eq!(report.penalty_cases, 1);
        assert_eq!(report.device_owner_bytes, 17_253_733_120);
        assert_eq!(report.host_stager_bytes, 4_678_656);
    }
}

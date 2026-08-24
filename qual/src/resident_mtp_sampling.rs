//! Source-backed gate for unbiased single-slot MTP sampling.

use crate::{
    DeviceBenchmarkError, SpeculativeSamplingQualification, device_benchmark,
    qualify_resident_mtp_generation, qualify_speculative_sampling,
};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, GeneratedText, ResidentMtpGenerationStats,
    ResidentMtpTextGenerator, SamplingOptions, SamplingPenalties,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen38_27B};

const LIMIT_CASES: [(usize, u64); 4] = [(2, 11), (3, 17), (4, 23), (8, 29)];

/// Failure of the sampled target-plus-MTP generation gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentMtpSamplingQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Frontend, sampling, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA ownership or observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Device preconditions were not satisfied.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An exact law, route, conditioning, or streaming boundary differed.
    #[error("resident MTP sampling qualification failed: {0}")]
    Mismatch(String),
}

/// Exact mathematical, source, route, and owner evidence produced by the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMtpSamplingQualification {
    /// Independent exact-law oracle evidence.
    pub mathematical_oracle: SpeculativeSamplingQualification,
    /// Existing source-backed greedy-degeneration suite completed first.
    pub greedy_degenerate_suites: usize,
    /// Seeded sampled cases replayed deterministically.
    pub deterministic_cases: usize,
    /// Non-identity penalty-conditioned cases executed.
    pub penalty_cases: usize,
    /// K=1,2,3,4 target verification routes observed.
    pub verification_routes: [usize; 4],
    /// Draft tokens proposed across sampled cases.
    pub draft_proposals: usize,
    /// Draft tokens accepted by the target law.
    pub accepted_drafts: usize,
    /// Complete target-plus-MTP device ownership.
    pub device_owner_bytes: usize,
    /// Complete page-locked staging ownership.
    pub host_stager_bytes: usize,
}

/// Qualifies sampled MTP generation against independent exact-law and source-backed authorities.
pub fn qualify_resident_mtp_sampling(
    root: &Path,
) -> Result<ResidentMtpSamplingQualification, ResidentMtpSamplingQualificationError> {
    let mathematical_oracle = qualify_speculative_sampling().map_err(|error| {
        ResidentMtpSamplingQualificationError::Mismatch(format!(
            "independent speculative-sampling oracle failed: {error}"
        ))
    })?;
    let _greedy = qualify_resident_mtp_generation(root).map_err(|error| {
        ResidentMtpSamplingQualificationError::Mismatch(format!(
            "greedy degeneration suite failed: {error}"
        ))
    })?;
    let _preflight = device_benchmark::preflight()?;

    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let mut generator = ResidentMtpTextGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();
    if stable_addresses.contains(&0)
        || stable_addresses
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != stable_addresses.len()
    {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(
            "resident sampled MTP addresses are not five unique nonzero values".to_string(),
        ));
    }

    let warm = sampled_request(8, 29, SamplingPenalties::identity());
    let _ = run_mtp(&mut generator, &warm)?;
    let before = device_memory_info(generator.context())?;
    let mut verification_routes = [0usize; 4];
    let mut draft_proposals = 0;
    let mut accepted_drafts = 0;
    for (maximum, seed) in LIMIT_CASES {
        let request = sampled_request(maximum, seed, SamplingPenalties::identity());
        let (first, first_stats, first_steps) = run_mtp(&mut generator, &request)?;
        let (second, second_stats, second_steps) = run_mtp(&mut generator, &request)?;
        compare_seeded(
            maximum,
            &first,
            first_stats,
            first_steps,
            &second,
            second_stats,
            second_steps,
        )?;
        let required_route = maximum.saturating_sub(2).min(3);
        if first_stats.verification_routes[required_route] == 0 {
            return Err(ResidentMtpSamplingQualificationError::Mismatch(format!(
                "max_new_tokens={maximum} did not select sampled target verification K={}",
                required_route + 1
            )));
        }
        for (total, count) in verification_routes
            .iter_mut()
            .zip(first_stats.verification_routes)
        {
            *total += count;
        }
        draft_proposals += first_stats.draft_proposals;
        accepted_drafts += first_stats.accepted_drafts;
    }
    if accepted_drafts == 0 || accepted_drafts >= draft_proposals {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(format!(
            "sampled source cases did not cover both acceptance and rejection: accepted={accepted_drafts}, proposals={draft_proposals}"
        )));
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
    let (penalized_a, stats_a, steps_a) = run_mtp(&mut generator, &penalty_request)?;
    let (penalized_b, stats_b, steps_b) = run_mtp(&mut generator, &penalty_request)?;
    compare_seeded(
        8,
        &penalized_a,
        stats_a,
        steps_a,
        &penalized_b,
        stats_b,
        steps_b,
    )?;
    if stats_a.verification_routes[3] == 0 {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(
            "penalty-conditioned source case did not execute K=4 verification".to_string(),
        ));
    }

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(format!(
            "resident sampled MTP device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(
            "resident sampled MTP addresses changed after replay".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(ResidentMtpSamplingQualification {
        mathematical_oracle,
        greedy_degenerate_suites: 1,
        deterministic_cases: LIMIT_CASES.len(),
        penalty_cases: 1,
        verification_routes,
        draft_proposals,
        accepted_drafts,
        device_owner_bytes: generator.device_owner_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
    })
}

fn sampled_request(
    maximum_new_tokens: usize,
    seed: u64,
    penalties: SamplingPenalties,
) -> ChatGenerationRequest {
    let mut request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);
    request.template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    request.sampling = SamplingOptions {
        seed,
        penalties,
        ..SamplingOptions::default()
    };
    request.max_new_tokens = maximum_new_tokens;
    request
}

fn run_mtp(
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(GeneratedText, ResidentMtpGenerationStats, usize), ResidentMtpSamplingQualificationError>
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
        return Err(ResidentMtpSamplingQualificationError::Mismatch(
            "sampled MTP streaming deltas did not reassemble the complete output".to_string(),
        ));
    }
    Ok((output, stats, steps))
}

#[allow(clippy::too_many_arguments)]
fn compare_seeded(
    maximum: usize,
    first: &GeneratedText,
    first_stats: ResidentMtpGenerationStats,
    first_steps: usize,
    second: &GeneratedText,
    second_stats: ResidentMtpGenerationStats,
    second_steps: usize,
) -> Result<(), ResidentMtpSamplingQualificationError> {
    if first.prompt.token_ids != second.prompt.token_ids
        || first.prompt.rendered_bytes != second.prompt.rendered_bytes
        || first.token_ids != second.token_ids
        || first.text != second.text
        || first.finish_reason != second.finish_reason
        || first.finish_reason != FinishReason::Length
        || first_stats != second_stats
        || first_steps != second_steps
    {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(format!(
            "seeded max_new_tokens={maximum} sampled replay differed: first={:?}/{first_stats:?}, second={:?}/{second_stats:?}",
            first.token_ids, second.token_ids
        )));
    }
    Ok(())
}

fn verify_owner(
    generator: &ResidentMtpTextGenerator,
) -> Result<(), ResidentMtpSamplingQualificationError> {
    if generator.device_owner_bytes() != 30_342_618_624
        || generator.host_stager_bytes() != 23_811_072
        || generator.kv_route_host_bytes() != 113_454
        || generator.context_capacity() != 220_000
    {
        return Err(ResidentMtpSamplingQualificationError::Mismatch(format!(
            "resident sampled MTP owner accounting changed: device={}, host={}, routes={}, capacity={}",
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
    use super::LIMIT_CASES;
    use std::path::Path;

    #[test]
    fn resident_mtp_sampling_suite_inventory_is_exact() {
        assert_eq!(LIMIT_CASES, [(2, 11), (3, 17), (4, 23), (8, 29)]);
        assert_eq!(
            LIMIT_CASES.map(|(limit, _)| (limit - 1).min(4)),
            [1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_sampling_suite_is_unbiased_and_conditioned() {
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let report = super::qualify_resident_mtp_sampling(Path::new(&root)).unwrap();
        assert_eq!(report.mathematical_oracle.induced_law_cases, 4);
        assert_eq!(report.greedy_degenerate_suites, 1);
        assert_eq!(report.deterministic_cases, 4);
        assert_eq!(report.penalty_cases, 1);
        assert!(report.verification_routes.iter().all(|&routes| routes > 0));
        assert!(report.accepted_drafts > 0);
        assert!(report.accepted_drafts < report.draft_proposals);
        assert_eq!(report.device_owner_bytes, 30_342_618_624);
    }
}

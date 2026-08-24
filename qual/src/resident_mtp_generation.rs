//! Source-backed gate for exact single-slot greedy MTP generation.

use crate::{DeviceBenchmarkError, device_benchmark, qualify_resident_mtp};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, GeneratedText, ResidentMtpGreedyStats,
    ResidentMtpTextGenerator, ResidentTextGenerator, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen38_27B};

const LIMIT_CASES: [usize; 4] = [2, 3, 4, 8];

/// Failure of the exact greedy target-plus-MTP generation gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentMtpGenerationQualificationError {
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
    /// An exact route, fallback, or streaming boundary differed.
    #[error("resident greedy MTP generation qualification failed: {0}")]
    Mismatch(String),
}

/// Exact fallback, route, streaming, and owner evidence produced by the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMtpGenerationQualification {
    /// Independent source-backed resident-MTP owner suites completed first.
    pub source_owner_suites: usize,
    /// Complete target-only outputs compared with MTP outputs.
    pub fallback_cases: usize,
    /// K=1,2,3,4 target verification routes observed.
    pub verification_routes: [usize; 4],
    /// Draft proposals evaluated across all cases.
    pub draft_proposals: usize,
    /// Draft proposals licensed by equal target argmax decisions.
    pub accepted_drafts: usize,
    /// Streaming steps reassembled into complete outputs.
    pub streaming_steps: usize,
    /// Complete target-plus-MTP device ownership.
    pub device_owner_bytes: usize,
    /// Complete page-locked staging ownership.
    pub host_stager_bytes: usize,
    /// Exact allocation-free host page-routing bytes.
    pub kv_route_host_bytes: usize,
}

/// Qualifies greedy draft-three generation against the existing target-only production route.
pub fn qualify_resident_mtp_generation(
    root: &Path,
) -> Result<ResidentMtpGenerationQualification, ResidentMtpGenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    qualify_resident_mtp(root).map_err(|error| {
        ResidentMtpGenerationQualificationError::Mismatch(format!(
            "independent resident MTP owner suite failed: {error}"
        ))
    })?;

    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
        return Err(ResidentMtpGenerationQualificationError::Mismatch(
            "device zero is not compute capability 12.0".to_string(),
        ));
    }
    let requests = LIMIT_CASES.map(greedy_request);
    let mut baseline = ResidentTextGenerator::from_snapshot(&context, snapshot.clone())?;
    let mut expected = Vec::with_capacity(LIMIT_CASES.len());
    for request in &requests {
        expected.push(run_target(&mut baseline, request)?);
    }
    drop(baseline);

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
        return Err(ResidentMtpGenerationQualificationError::Mismatch(
            "resident greedy MTP owner addresses are not five unique nonzero values".to_string(),
        ));
    }

    // Warm the complete K=4 transaction and all owned transfers before the memory observation.
    let _ = run_mtp(&mut generator, &requests[3])?;
    let before = device_memory_info(generator.context())?;
    let mut routes = [0usize; 4];
    let mut draft_proposals = 0;
    let mut accepted_drafts = 0;
    let mut streaming_steps = 0;
    for (case, (request, expected)) in requests.iter().zip(&expected).enumerate() {
        let (actual, stats, steps) = run_mtp(&mut generator, request)?;
        compare_output(LIMIT_CASES[case], expected, &actual)?;
        let required_route = LIMIT_CASES[case].saturating_sub(2).min(3);
        if stats.verification_routes[required_route] == 0 {
            return Err(ResidentMtpGenerationQualificationError::Mismatch(format!(
                "max_new_tokens={} did not select target verification K={}",
                LIMIT_CASES[case],
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
    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(ResidentMtpGenerationQualificationError::Mismatch(format!(
            "resident greedy MTP device memory changed after warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(ResidentMtpGenerationQualificationError::Mismatch(
            "resident greedy MTP owner addresses changed after replay".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(ResidentMtpGenerationQualification {
        source_owner_suites: 1,
        fallback_cases: LIMIT_CASES.len(),
        verification_routes: routes,
        draft_proposals,
        accepted_drafts,
        streaming_steps,
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

fn run_target(
    generator: &mut ResidentTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<GeneratedText, ResidentMtpGenerationQualificationError> {
    let mut session = generator.start(request)?;
    while session.finish_reason().is_none() {
        let _ = session.step()?;
    }
    Ok(session.into_output()?)
}

fn run_mtp(
    generator: &mut ResidentMtpTextGenerator,
    request: &ChatGenerationRequest,
) -> Result<(GeneratedText, ResidentMtpGreedyStats, usize), ResidentMtpGenerationQualificationError>
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
        return Err(ResidentMtpGenerationQualificationError::Mismatch(
            "resident greedy MTP streaming deltas did not reassemble the complete output"
                .to_string(),
        ));
    }
    Ok((output, stats, steps))
}

fn compare_output(
    maximum_new_tokens: usize,
    expected: &GeneratedText,
    actual: &GeneratedText,
) -> Result<(), ResidentMtpGenerationQualificationError> {
    if actual.prompt.token_ids != expected.prompt.token_ids
        || actual.prompt.rendered_bytes != expected.prompt.rendered_bytes
        || actual.token_ids != expected.token_ids
        || actual.text != expected.text
        || actual.finish_reason != expected.finish_reason
        || actual.finish_reason != FinishReason::Length
    {
        return Err(ResidentMtpGenerationQualificationError::Mismatch(format!(
            "max_new_tokens={maximum_new_tokens} differs from the target-only fallback: target={:?}/{:?}, MTP={:?}/{:?}",
            expected.token_ids, expected.finish_reason, actual.token_ids, actual.finish_reason
        )));
    }
    Ok(())
}

fn verify_owner(
    generator: &ResidentMtpTextGenerator,
) -> Result<(), ResidentMtpGenerationQualificationError> {
    if generator.device_owner_bytes() != 30_263_692_800
        || generator.host_stager_bytes() != 23_729_152
        || generator.kv_route_host_bytes() != 113_454
        || generator.context_capacity() != 220_000
    {
        return Err(ResidentMtpGenerationQualificationError::Mismatch(
            "resident greedy MTP owner bytes or capacity changed".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::LIMIT_CASES;
    use std::path::Path;

    #[test]
    fn resident_mtp_generation_suite_inventory_is_exact() {
        assert_eq!(LIMIT_CASES, [2, 3, 4, 8]);
        assert_eq!(LIMIT_CASES.map(|limit| (limit - 1).min(4)), [1, 2, 3, 4]);
    }

    #[test]
    #[ignore = "requires the admitted source snapshot and an exclusive RTX 5090"]
    fn resident_mtp_generation_suite_matches_target_only_greedy() {
        let root = std::env::var_os("TUISKO_SNAPSHOT")
            .expect("TUISKO_SNAPSHOT must name the admitted snapshot");
        let report = super::qualify_resident_mtp_generation(Path::new(&root)).unwrap();
        assert_eq!(report.source_owner_suites, 1);
        assert_eq!(report.fallback_cases, 4);
        assert!(report.verification_routes.iter().all(|&routes| routes > 0));
        assert!(report.draft_proposals > 0);
        assert!(report.streaming_steps >= 8);
        assert_eq!(report.device_owner_bytes, 30_263_692_800);
        assert_eq!(report.host_stager_bytes, 23_729_152);
    }
}

//! Source-backed integration gate for frontend, generation control, and the resident model.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, ResidentTextGenerator, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen38_27B};

// vLLM 0.27.1 reference captures documented by decompressed fixture digests
// 8f20cb4fdf7ab2e5ff9def3598b433f4cafcd9c02aa62d9cfa19eee400bf225a and
// 4ff0853747ac857814a12455869dc4f111eb7d40e2af544a1389a8c73e107041.
const VLLM_CASES: [(&[u32], u32); 2] = [(&[151_643], 198), (&[151_643, 151_644], 30_350)];
const PREFILL_PLAN_CASES: [(usize, usize); 13] = [
    (31, 0),
    (32, 32),
    (63, 32),
    (64, 64),
    (96, 96),
    (127, 96),
    (128, 128),
    (256, 256),
    (1_024, 1_024),
    (1_055, 1_024),
    (1_056, 1_056),
    (1_152, 1_152),
    (32_896, 32_896),
];

/// Failure of the concrete resident-generation integration gate.
#[derive(Debug, thiserror::Error)]
pub enum ResidentGenerationQualificationError {
    /// Snapshot admission or source binding failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// Tokenizer or chat-template admission failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Frontend, generation, or resident execution failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// CUDA context or memory observation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The exact target was not available exclusively.
    #[error(transparent)]
    Precondition(#[from] DeviceBenchmarkError),
    /// An externally visible generation boundary differed.
    #[error("resident-generation qualification failed: {0}")]
    Mismatch(String),
}

/// Reference and streaming boundaries checked by the integration gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentGenerationQualification {
    /// Independent vLLM next-token fixtures checked.
    pub reference_cases: usize,
    /// Production chat steps streamed and reassembled.
    pub chat_steps: usize,
    /// Exact and composed prefill plans selected through production dispatch.
    pub native_prefill_plans: usize,
    /// Exact device arena bytes owned by the generator.
    pub arena_bytes: usize,
    /// Exact page-locked embedding and logit staging bytes.
    pub host_stager_bytes: usize,
    /// Exact allocation-free host page-routing bytes.
    pub kv_route_host_bytes: usize,
}

/// Qualifies the exact single-slot frontend-to-device generation path.
pub fn qualify_resident_generation(
    root: &Path,
) -> Result<ResidentGenerationQualification, ResidentGenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen38_27B>::open(root)?);
    let oracle_frontend = TextFrontend::open(snapshot.as_ref())?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(ResidentGenerationQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let mut generator = ResidentTextGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();

    // Warm every host/device transfer path before observing allocation stability.
    let _ = generator.qualification_greedy_after_tokens(VLLM_CASES[0].0)?;
    let mut native_tokens = [0u32; PREFILL_PLAN_CASES.len()];
    for (index, (tokens, expected_native)) in PREFILL_PLAN_CASES.into_iter().enumerate() {
        let fixture = vec![151_643; tokens];
        let (token, selected) = generator.qualification_greedy_after_tokens_with_route(&fixture)?;
        if selected != expected_native {
            return Err(ResidentGenerationQualificationError::Mismatch(format!(
                "T={tokens} production dispatch selected {selected} native prefill tokens, expected {expected_native}"
            )));
        }
        native_tokens[index] = token;
    }
    let before = device_memory_info(generator.context())?;
    for (tokens, expected) in VLLM_CASES {
        let actual = generator.qualification_greedy_after_tokens(tokens)?;
        if actual != expected {
            return Err(ResidentGenerationQualificationError::Mismatch(format!(
                "vLLM next-token fixture {tokens:?} selected {actual}, expected {expected}"
            )));
        }
    }
    for (index, (tokens, expected_native)) in PREFILL_PLAN_CASES.into_iter().enumerate() {
        let fixture = vec![151_643; tokens];
        let (token, selected) = generator.qualification_greedy_after_tokens_with_route(&fixture)?;
        if selected != expected_native || token != native_tokens[index] {
            return Err(ResidentGenerationQualificationError::Mismatch(format!(
                "T={tokens} composed prefill replay changed dispatch or greedy output"
            )));
        }
    }

    let messages = vec![ChatMessage::new("user", "Hello")];
    let template = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    let expected_prompt = oracle_frontend.encode_chat(&messages, &template)?;
    let stop_ids = oracle_frontend.stop_ids();
    let mut expected_tokens = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut processed = expected_prompt.clone();
        processed.extend_from_slice(&expected_tokens);
        let token = generator.qualification_greedy_after_tokens(&processed)?;
        expected_tokens.push(token);
        if stop_ids.contains(&token) {
            break;
        }
    }

    let mut request = ChatGenerationRequest::new(messages);
    request.template = template;
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = 2;
    let mut session = generator.start(&request)?;
    if session.prompt_token_ids() != expected_prompt {
        return Err(ResidentGenerationQualificationError::Mismatch(
            "generation bridge changed the admitted prompt encoding".to_string(),
        ));
    }
    let mut streamed = String::new();
    let mut step_tokens = Vec::new();
    while session.finish_reason().is_none() {
        let step = session.step()?;
        step_tokens.push(step.token_id);
        if let Some(delta) = step.delta {
            streamed.push_str(&delta);
        }
    }
    let output = session.into_output()?;
    let expected_reason = if expected_tokens
        .last()
        .is_some_and(|token| stop_ids.contains(token))
    {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    if step_tokens != expected_tokens
        || output.token_ids != expected_tokens
        || output.text != streamed
        || output.finish_reason != expected_reason
    {
        return Err(ResidentGenerationQualificationError::Mismatch(
            "streamed chat output differs from independently replayed token control".to_string(),
        ));
    }

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(ResidentGenerationQualificationError::Mismatch(format!(
            "device memory changed after generation warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(ResidentGenerationQualificationError::Mismatch(
            "resident generation owner addresses changed".to_string(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(ResidentGenerationQualification {
        reference_cases: VLLM_CASES.len(),
        chat_steps: step_tokens.len(),
        native_prefill_plans: PREFILL_PLAN_CASES.len(),
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        kv_route_host_bytes: generator.kv_route_host_bytes(),
    })
}

fn verify_owner(
    generator: &ResidentTextGenerator,
) -> Result<(), ResidentGenerationQualificationError> {
    if generator.arena_bytes() != 28_494_230_272
        || generator.host_stager_bytes() != 10_982_400
        || generator.kv_route_host_bytes() != 113_454
        || generator.context_capacity() != 220_000
    {
        return Err(ResidentGenerationQualificationError::Mismatch(
            "resident generation owner bytes or capacity changed".to_string(),
        ));
    }
    let addresses = generator.qualification_addresses();
    if addresses.contains(&0)
        || addresses[0] == addresses[1]
        || addresses[0] == addresses[2]
        || addresses[1] == addresses[2]
    {
        return Err(ResidentGenerationQualificationError::Mismatch(
            "resident generation owner addresses are invalid".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::qualify_resident_generation;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned snapshot and an exclusive SM120 device"]
    fn source_frontend_generation_matches_vllm_tokens_and_streaming()
    -> Result<(), super::ResidentGenerationQualificationError> {
        let root = std::env::var_os("TUISKO_SNAPSHOT").ok_or_else(|| {
            super::ResidentGenerationQualificationError::Mismatch(
                "set TUISKO_SNAPSHOT to the admitted revision".to_string(),
            )
        })?;
        let report = qualify_resident_generation(&PathBuf::from(root))?;
        assert_eq!(report.reference_cases, 2);
        assert!((1..=2).contains(&report.chat_steps));
        assert_eq!(report.native_prefill_plans, 13);
        assert_eq!(report.arena_bytes, 28_494_230_272);
        assert_eq!(report.host_stager_bytes, 10_982_400);
        Ok(())
    }
}

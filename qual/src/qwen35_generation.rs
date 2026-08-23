//! Source-backed integration gate for the Qwen3.5 frontend and resident text graph.

use crate::{DeviceBenchmarkError, device_benchmark};
use std::path::Path;
use std::sync::Arc;
use tuisko_engine::{
    ChatGenerationRequest, EngineError, FinishReason, Qwen35ResidentTextGenerator, SamplingOptions,
};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, FrontendError, TextFrontend};
use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
use tuisko_model::{CheckpointError, CheckpointSnapshot, Qwen35_9B};

// Transformers 5.2.0 `apply_chat_template` and tokenizer output from the pinned snapshot.
const HELLO_THINKING: [u32; 11] = [
    248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
];
const HELLO_NO_THINKING: [u32; 13] = [
    248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 271, 248_069, 271,
];

/// Failure of the concrete Qwen3.5 generation integration gate.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35GenerationQualificationError {
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
    #[error("Qwen3.5 generation qualification failed: {0}")]
    Mismatch(String),
}

/// Frontend and streaming boundaries checked by the integration gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GenerationQualification {
    /// Independent Transformers prompt fixtures checked.
    pub prompt_cases: usize,
    /// Production chat steps streamed and reassembled.
    pub chat_steps: usize,
    /// Exact selected tokens from the production request.
    pub generated_tokens: Vec<u32>,
    /// Exact bytes across all retained device arenas.
    pub arena_bytes: usize,
    /// Exact page-locked embedding and logit staging bytes.
    pub host_stager_bytes: usize,
    /// Number of stable retained device and host addresses.
    pub stable_addresses: usize,
}

/// Qualifies the exact single-slot Qwen3.5 frontend-to-device generation path.
pub fn qualify_qwen35_generation(
    root: &Path,
) -> Result<Qwen35GenerationQualification, Qwen35GenerationQualificationError> {
    let _preflight = device_benchmark::preflight()?;
    let snapshot = Arc::new(CheckpointSnapshot::<Qwen35_9B>::open(root)?);
    let oracle_frontend = TextFrontend::open_qwen35(snapshot.as_ref())?;
    verify_prompt_fixtures(&oracle_frontend)?;
    let context = CudaContext::new(0).map_err(GpuError::from)?;
    let capability = context.compute_capability().map_err(GpuError::from)?;
    if capability != (12, 0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device zero has compute capability {}.{}, expected 12.0",
            capability.0, capability.1
        )));
    }
    let mut generator = Qwen35ResidentTextGenerator::from_snapshot(&context, snapshot)?;
    verify_owner(&generator)?;
    let stable_addresses = generator.qualification_addresses();

    let _ = generator.qualification_greedy_after_tokens(&HELLO_THINKING)?;
    let before = device_memory_info(generator.context())?;
    let first = generator.qualification_greedy_after_tokens(&HELLO_THINKING)?;
    let repeated = generator.qualification_greedy_after_tokens(&HELLO_THINKING)?;
    if first != repeated {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "reset replay selected {first} then {repeated} for the same prompt"
        )));
    }

    let messages = vec![ChatMessage::new("user", "Hello")];
    let mut request = ChatGenerationRequest::new(messages);
    request.sampling = SamplingOptions::greedy();
    request.max_new_tokens = 2;
    let stop_ids = oracle_frontend.stop_ids();
    let mut expected_tokens = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut processed = HELLO_THINKING.to_vec();
        processed.extend_from_slice(&expected_tokens);
        let token = generator.qualification_greedy_after_tokens(&processed)?;
        expected_tokens.push(token);
        if stop_ids.contains(&token) {
            break;
        }
    }

    let mut session = generator.start(&request)?;
    if session.prompt_token_ids() != HELLO_THINKING {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "generation bridge changed the Transformers prompt fixture".into(),
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
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "streamed output differs from independent raw-token replay".into(),
        ));
    }

    let after = device_memory_info(generator.context())?;
    if before != after {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "device memory changed after generation warmup: before={before:?}, after={after:?}"
        )));
    }
    if generator.qualification_addresses() != stable_addresses {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "resident generation owner addresses changed".into(),
        ));
    }
    device_benchmark::require_current_process_exclusive()?;

    Ok(Qwen35GenerationQualification {
        prompt_cases: 2,
        chat_steps: step_tokens.len(),
        generated_tokens: step_tokens,
        arena_bytes: generator.arena_bytes(),
        host_stager_bytes: generator.host_stager_bytes(),
        stable_addresses: stable_addresses.len(),
    })
}

fn verify_prompt_fixtures(
    frontend: &TextFrontend,
) -> Result<(), Qwen35GenerationQualificationError> {
    let messages = [ChatMessage::new("user", "Hello")];
    let thinking = frontend.encode_chat(&messages, &ChatTemplateOptions::default())?;
    let no_thinking = frontend.encode_chat(
        &messages,
        &ChatTemplateOptions {
            enable_thinking: Some(false),
            ..ChatTemplateOptions::default()
        },
    )?;
    if thinking != HELLO_THINKING || no_thinking != HELLO_NO_THINKING {
        return Err(Qwen35GenerationQualificationError::Mismatch(format!(
            "prompt IDs differ from Transformers: thinking={thinking:?}, no-thinking={no_thinking:?}"
        )));
    }
    Ok(())
}

fn verify_owner(
    generator: &Qwen35ResidentTextGenerator,
) -> Result<(), Qwen35GenerationQualificationError> {
    if generator.arena_bytes() != 6_435_512_320
        || generator.host_stager_bytes() != 562_176
        || generator.context_capacity() != 192
    {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 generation owner bytes or capacity changed".into(),
        ));
    }
    let addresses = generator.qualification_addresses();
    let mut unique = addresses.clone();
    unique.sort_unstable();
    unique.dedup();
    if addresses.len() != 34 || unique.len() != addresses.len() || addresses.contains(&0) {
        return Err(Qwen35GenerationQualificationError::Mismatch(
            "Qwen3.5 generation owner addresses are invalid".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::qualify_qwen35_generation;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires the pinned Qwen3.5 snapshot and an exclusive SM120 device"]
    fn source_frontend_generation_matches_transformers_and_streaming()
    -> Result<(), super::Qwen35GenerationQualificationError> {
        let root = std::env::var_os("TUISKO_QWEN35_SNAPSHOT").ok_or_else(|| {
            super::Qwen35GenerationQualificationError::Mismatch(
                "set TUISKO_QWEN35_SNAPSHOT to the admitted revision".into(),
            )
        })?;
        let report = qualify_qwen35_generation(&PathBuf::from(root))?;
        assert_eq!(report.prompt_cases, 2);
        assert!((1..=2).contains(&report.chat_steps));
        assert_eq!(report.arena_bytes, 6_435_512_320);
        assert_eq!(report.host_stager_bytes, 562_176);
        assert_eq!(report.stable_addresses, 34);
        eprintln!("Qwen3.5 generation qualification passed: {report:?}");
        Ok(())
    }
}

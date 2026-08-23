//! Source-backed host generation-state qualification.

use std::env;
use std::error::Error;
use std::path::Path;
use tuisko_engine::{ChatGenerationRequest, FinishReason, GenerationSession, SamplingOptions};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, TextFrontend};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(snapshot) = arguments.next() else {
        return Err("usage: qualify-generation SNAPSHOT".into());
    };
    if arguments.next().is_some() {
        return Err("usage: qualify-generation SNAPSHOT".into());
    }

    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&snapshot))?;
    let frontend = TextFrontend::open(&snapshot)?;
    let messages = vec![ChatMessage::new("user", "Hello")];
    let expected_prompt = frontend.encode_chat(
        &messages,
        ChatTemplateOptions {
            enable_thinking: Some(false),
        },
    )?;
    let expected_text = "Hello, 世界!";
    let expected_tokens = frontend.encode(expected_text)?;

    let mut stop_request = ChatGenerationRequest::new(messages.clone());
    stop_request.template.enable_thinking = Some(false);
    stop_request.sampling = SamplingOptions::greedy();
    stop_request.max_new_tokens = expected_tokens.len() + 1;
    let mut stop_session = GenerationSession::start(&frontend, &stop_request)?;
    if stop_session.prompt_token_ids() != expected_prompt {
        return Err("generation session changed the exact prompt encoding".into());
    }
    let mut streamed = String::new();
    for &token in &expected_tokens {
        let step = stop_session.accept_logits(&logits_selecting(token))?;
        if step.finish_reason.is_some() {
            return Err("generation session stopped before the selected EOS token".into());
        }
        if let Some(delta) = step.delta {
            streamed.push_str(&delta);
        }
    }
    let eos = frontend.stop_ids()[0];
    let stop = stop_session.accept_logits(&logits_selecting(eos))?;
    if stop.finish_reason != Some(FinishReason::Stop) || stop.delta.is_some() {
        return Err("selected EOS did not finish without emitting special-token text".into());
    }
    let stop_output = stop_session.into_output()?;
    let mut expected_with_eos = expected_tokens.clone();
    expected_with_eos.push(eos);
    if stop_output.token_ids != expected_with_eos
        || stop_output.text != expected_text
        || streamed != expected_text
        || stop_output.finish_reason != FinishReason::Stop
    {
        return Err("stop-terminated generation output differs from its selected tokens".into());
    }

    let mut length_request = ChatGenerationRequest::new(messages);
    length_request.template.enable_thinking = Some(false);
    length_request.sampling = SamplingOptions::greedy();
    length_request.max_new_tokens = expected_tokens.len();
    let mut length_session = GenerationSession::start(&frontend, &length_request)?;
    for (index, &token) in expected_tokens.iter().enumerate() {
        let step = length_session.accept_logits(&logits_selecting(token))?;
        let expected_reason = (index + 1 == expected_tokens.len()).then_some(FinishReason::Length);
        if step.finish_reason != expected_reason {
            return Err("length termination occurred at the wrong token boundary".into());
        }
    }
    let length_output = length_session.into_output()?;
    if length_output.token_ids != expected_tokens
        || length_output.text != expected_text
        || length_output.finish_reason != FinishReason::Length
    {
        return Err("length-terminated generation output differs from its selected tokens".into());
    }

    let mut empty_request = stop_request;
    empty_request.max_new_tokens = 0;
    let mut empty_session = GenerationSession::start(&frontend, &empty_request)?;
    drop(frontend);
    if empty_session.finish_reason() != Some(FinishReason::Length) {
        return Err("zero-token request did not finish at initialization".into());
    }
    if empty_session
        .accept_logits(&logits_selecting(expected_tokens[0]))
        .is_ok()
    {
        return Err("zero-token request accepted an unnecessary logit row".into());
    }
    let empty_output = empty_session.into_output()?;
    if !empty_output.token_ids.is_empty()
        || !empty_output.text.is_empty()
        || empty_output.finish_reason != FinishReason::Length
    {
        return Err("zero-token generation output is not empty".into());
    }

    println!(
        "generation qualification passed: {} prompt IDs, {} generated IDs, stop and length routes",
        expected_prompt.len(),
        expected_tokens.len()
    );
    Ok(())
}

fn logits_selecting(token: u32) -> Vec<u16> {
    let mut logits = vec![0xbf80; Qwen38_27B::VOCAB];
    logits[token as usize] = 0x3f80;
    logits
}

//! Host generation state over production logit rows.

use crate::{
    EngineError, EngineResult, SampleDecision, Sampler, SamplingDistribution, SamplingOptions,
};
use std::collections::HashMap;
use tuisko_frontend::{
    ChatMessage, ChatTemplateOptions, PromptEncoding, PromptEncodingMetrics, StreamingDecoder,
    TextFrontend,
};
use tuisko_model::{Arch, Qwen38_27B};

const DEFAULT_MAX_NEW_TOKENS: usize = 128;

/// One text-generation request after transport parsing.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatGenerationRequest {
    /// Ordered text-only chat messages.
    pub messages: Vec<ChatMessage>,
    /// Chat-template controls.
    pub template: ChatTemplateOptions,
    /// Token selection controls.
    pub sampling: SamplingOptions,
    /// Maximum generated tokens, including a selected stop token.
    pub max_new_tokens: usize,
}

impl ChatGenerationRequest {
    /// Creates a request with the admitted checkpoint defaults.
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            template: ChatTemplateOptions::default(),
            sampling: SamplingOptions::default(),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

/// Why a generation session finished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// An admitted EOS token was selected.
    Stop,
    /// The request reached `max_new_tokens`.
    Length,
}

impl FinishReason {
    /// OpenAI-compatible spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Observable result of one consumed logit row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationStep {
    /// Selected token.
    pub token_id: u32,
    /// Newly complete UTF-8 text, when this token emitted any.
    pub delta: Option<String>,
    /// Terminal reason when this step completed the request.
    pub finish_reason: Option<FinishReason>,
}

/// Completed generation output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedText {
    /// Prompt encoding and prefix-cache accounting.
    pub prompt: PromptEncoding,
    /// Generated tokens, including a selected stop token.
    pub token_ids: Vec<u32>,
    /// Complete decoded text, excluding special tokens.
    pub text: String,
    /// Terminal reason.
    pub finish_reason: FinishReason,
}

/// Host-visible output retained when an active request is cancelled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancelledText {
    /// Prompt encoding and frontend prefix-cache accounting.
    pub prompt: PromptEncoding,
    /// Tokens emitted before cancellation.
    pub token_ids: Vec<u32>,
    /// Complete text decoded before cancellation.
    pub text: String,
}

/// Per-request host state consuming one exact BF16 vocabulary row at a time.
pub struct GenerationSession {
    prompt: PromptEncoding,
    prompt_metrics: PromptEncodingMetrics,
    sampler: Sampler,
    decoder: StreamingDecoder,
    generated: Vec<u32>,
    occurrences: HashMap<u32, u32>,
    max_new_tokens: usize,
    finish_reason: Option<FinishReason>,
}

impl GenerationSession {
    /// Renders and tokenizes the prompt and initializes sampling state.
    pub fn start(frontend: &TextFrontend, request: &ChatGenerationRequest) -> EngineResult<Self> {
        let (prompt, prompt_metrics) =
            frontend.encode_chat_with_metrics(&request.messages, &request.template)?;
        let sampler = Sampler::new(request.sampling, frontend.stop_ids())?;

        Ok(Self {
            prompt,
            prompt_metrics,
            sampler,
            decoder: frontend.streaming_decoder(),
            generated: Vec::with_capacity(request.max_new_tokens.min(4_096)),
            occurrences: HashMap::with_capacity(if request.sampling.penalties.is_identity() {
                0
            } else {
                request.max_new_tokens.min(Qwen38_27B::VOCAB)
            }),
            max_new_tokens: request.max_new_tokens,
            finish_reason: (request.max_new_tokens == 0).then_some(FinishReason::Length),
        })
    }

    /// Starts a qualification session from already-tokenized prompt IDs.
    #[cfg(feature = "qualification")]
    pub fn qualification_from_tokens(
        frontend: &TextFrontend,
        token_ids: &[u32],
        max_new_tokens: usize,
        sampling: SamplingOptions,
    ) -> EngineResult<Self> {
        let sampler = Sampler::new(sampling, frontend.stop_ids())?;

        Ok(Self {
            prompt: PromptEncoding {
                token_ids: token_ids.to_vec(),
                message_boundary_tokens: token_ids.len(),
                reused_tokens: 0,
                rendered_bytes: 0,
                fresh_bytes: 0,
            },
            prompt_metrics: PromptEncodingMetrics::default(),
            sampler,
            decoder: frontend.streaming_decoder(),
            generated: Vec::with_capacity(max_new_tokens.min(4_096)),
            occurrences: HashMap::with_capacity(if sampling.penalties.is_identity() {
                0
            } else {
                max_new_tokens.min(Qwen38_27B::VOCAB)
            }),
            max_new_tokens,
            finish_reason: (max_new_tokens == 0).then_some(FinishReason::Length),
        })
    }

    /// Exact prompt IDs to prefill before the first logit row is consumed.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt.token_ids
    }

    /// Exact prompt prefix through the last complete chat message.
    pub fn message_boundary_token_ids(&self) -> &[u32] {
        &self.prompt.token_ids[..self.prompt.message_boundary_tokens]
    }

    /// Prompt-cache accounting for request instrumentation.
    pub const fn prompt_encoding(&self) -> &PromptEncoding {
        &self.prompt
    }

    /// Observation-only frontend timings and prefix-lookup detail.
    pub const fn prompt_metrics(&self) -> &PromptEncodingMetrics {
        &self.prompt_metrics
    }

    /// Tokens selected so far.
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated
    }

    /// Current terminal state.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Consumes one complete production logit row.
    pub fn accept_logits(&mut self, logits: &[u16]) -> EngineResult<GenerationStep> {
        if self.finish_reason.is_some() {
            return Err(EngineError::generation(
                "cannot consume logits after generation finished",
            ));
        }

        let decision = self
            .sampler
            .sample_with_counts(logits, &self.occurrences, &[])?;
        self.accept_decision(decision)
    }

    pub(crate) fn propose_logits(
        &mut self,
        logits: &[u16],
        provisional: &[u32],
    ) -> EngineResult<SampleDecision> {
        self.sampler
            .sample_with_counts(logits, &self.occurrences, provisional)
    }

    pub(crate) fn sampling_distribution(
        &self,
        logits: &[u16],
        provisional: &[u32],
    ) -> EngineResult<SamplingDistribution> {
        self.sampler
            .distribution(logits, &self.occurrences, provisional)
    }

    pub(crate) fn draw_distribution(
        &mut self,
        distribution: &SamplingDistribution,
    ) -> EngineResult<u32> {
        self.sampler.draw(distribution)
    }

    pub(crate) fn random_unit(&mut self) -> f64 {
        self.sampler.unit_f64()
    }

    pub(crate) fn accept_token(&mut self, token_id: u32) -> EngineResult<GenerationStep> {
        if self.finish_reason.is_some() {
            return Err(EngineError::generation(
                "cannot consume a token after generation finished",
            ));
        }
        let decision = self.sampler.decision_for_token(token_id)?;
        self.accept_decision(decision)
    }

    fn accept_decision(&mut self, decision: SampleDecision) -> EngineResult<GenerationStep> {
        let mut delta = if decision.stopped {
            None
        } else {
            self.decoder.push(decision.token_id)?
        };
        self.generated.push(decision.token_id);
        if !self.sampler.options().penalties.is_identity() {
            *self.occurrences.entry(decision.token_id).or_insert(0) += 1;
        }

        let finish_reason =
            completion_reason(decision.stopped, self.generated.len(), self.max_new_tokens);
        if finish_reason.is_some()
            && let Some(tail) = self.decoder.finish()
        {
            match &mut delta {
                Some(delta) => delta.push_str(&tail),
                None => delta = Some(tail),
            }
        }
        self.finish_reason = finish_reason;

        Ok(GenerationStep {
            token_id: decision.token_id,
            delta,
            finish_reason,
        })
    }

    /// Converts a terminal session into its owned output.
    pub fn into_output(self) -> EngineResult<GeneratedText> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            EngineError::generation("cannot take output before generation finishes")
        })?;

        Ok(GeneratedText {
            prompt: self.prompt,
            token_ids: self.generated,
            text: self.decoder.text().to_owned(),
            finish_reason,
        })
    }

    /// Finishes text decoding and takes the observable state of one active request.
    pub fn cancel(mut self) -> EngineResult<CancelledText> {
        if self.finish_reason.is_some() {
            return Err(EngineError::generation(
                "cannot cancel a generation session after it finished",
            ));
        }
        let _ = self.decoder.finish();
        Ok(CancelledText {
            prompt: self.prompt,
            token_ids: self.generated,
            text: self.decoder.text().to_owned(),
        })
    }
}

fn completion_reason(
    stopped: bool,
    generated_tokens: usize,
    max_new_tokens: usize,
) -> Option<FinishReason> {
    if stopped {
        Some(FinishReason::Stop)
    } else if generated_tokens >= max_new_tokens {
        Some(FinishReason::Length)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatGenerationRequest, DEFAULT_MAX_NEW_TOKENS, FinishReason, completion_reason};
    use tuisko_frontend::ChatMessage;

    #[test]
    fn request_defaults_are_the_checkpoint_defaults() {
        let request = ChatGenerationRequest::new(vec![ChatMessage::new("user", "Hello")]);

        assert_eq!(request.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(request.sampling.temperature, 1.0);
        assert_eq!(request.sampling.top_p, 0.95);
        assert_eq!(request.sampling.top_k, 20);
    }

    #[test]
    fn completion_route_covers_stop_and_length_boundaries() {
        let cases = [
            (false, 0, 4, None),
            (false, 3, 4, None),
            (false, 4, 4, Some(FinishReason::Length)),
            (false, 5, 4, Some(FinishReason::Length)),
            (true, 1, 4, Some(FinishReason::Stop)),
            (true, 4, 4, Some(FinishReason::Stop)),
        ];
        for (stopped, generated, maximum, expected) in cases {
            assert_eq!(
                completion_reason(stopped, generated, maximum),
                expected,
                "route ({stopped}, {generated}, {maximum})"
            );
        }
    }

    #[test]
    fn finish_reason_spellings_are_stable() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
    }
}

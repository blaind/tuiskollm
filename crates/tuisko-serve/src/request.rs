//! OpenAI chat-completion request admission.

use serde::Deserialize;
use serde_json::Value;
use tuisko_engine::{ChatGenerationRequest, SamplingOptions};
use tuisko_frontend::{ChatMessage, ChatTemplateOptions};
use tuisko_model::{Arch, Qwen38_27B};

const DEFAULT_MAX_NEW_TOKENS: usize = 128;

/// Only model identity served by this exact-target process.
pub const SERVED_MODEL: &str = Qwen38_27B::MODEL_ID;

/// OpenAI chat-completion request fields admitted by the current text product.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    preserve_thinking: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

/// Request after transport validation and mapping to the concrete generation owner.
#[derive(Debug)]
pub struct PreparedChatRequest {
    /// Exact host-generation request consumed by the resident scheduler.
    pub generation: ChatGenerationRequest,
    /// Whether the HTTP response uses server-sent events.
    pub stream: bool,
    /// Whether reasoning is split from assistant content in the response.
    pub split_reasoning: bool,
    /// Whether generated Qwen tool XML is parsed into OpenAI tool calls.
    pub parse_tools: bool,
}

/// Rejection at the OpenAI transport boundary.
#[derive(Debug, thiserror::Error)]
pub enum ChatRequestError {
    /// Requested model is not the one resident in this process.
    #[error("model `{requested}` is not served by this process")]
    ModelNotFound {
        /// Model identity supplied by the caller.
        requested: String,
    },
    /// Request option has no faithful current generation route.
    #[error("{0}")]
    Invalid(String),
}

impl ChatCompletionRequest {
    /// Validates and maps this wire request, using `default_seed` only when no seed was supplied.
    pub fn prepare(self, default_seed: u64) -> Result<PreparedChatRequest, ChatRequestError> {
        if self.model != SERVED_MODEL {
            return Err(ChatRequestError::ModelNotFound {
                requested: self.model,
            });
        }
        if self.messages.is_empty() {
            return Err(ChatRequestError::Invalid(
                "messages must not be empty".into(),
            ));
        }
        if self.n.is_some_and(|choices| choices != 1) {
            return Err(ChatRequestError::Invalid(
                "only one completion choice is admitted".into(),
            ));
        }
        require_default_float("presence_penalty", self.presence_penalty, 0.0)?;
        require_default_float("frequency_penalty", self.frequency_penalty, 0.0)?;
        require_default_float("repetition_penalty", self.repetition_penalty, 1.0)?;
        if self.stop.as_ref().is_some_and(|stop| !stop.is_null()) {
            return Err(ChatRequestError::Invalid(
                "custom stop sequences are not admitted; the checkpoint EOS set is fixed".into(),
            ));
        }
        if let Some(effort) = self.chat_template_kwargs.reasoning_effort.as_deref()
            && !matches!(effort, "xhigh" | "high" | "medium" | "low")
        {
            return Err(ChatRequestError::Invalid(format!(
                "reasoning_effort `{effort}` is not xhigh, high, medium, or low"
            )));
        }

        let mut tools = self.tools;
        match self.tool_choice.as_ref() {
            None | Some(Value::Null) => {}
            Some(Value::String(choice)) if choice == "auto" => {}
            Some(Value::String(choice)) if choice == "none" => tools.clear(),
            Some(_) => {
                return Err(ChatRequestError::Invalid(
                    "required or named tool_choice is not admitted without constrained decoding"
                        .into(),
                ));
            }
        }

        let max_new_tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let split_reasoning = self.chat_template_kwargs.enable_thinking.unwrap_or(true);
        let parse_tools = !tools.is_empty();
        let mut generation = ChatGenerationRequest::new(self.messages);
        generation.template = ChatTemplateOptions {
            enable_thinking: self.chat_template_kwargs.enable_thinking,
            preserve_thinking: self.chat_template_kwargs.preserve_thinking,
            reasoning_effort: self.chat_template_kwargs.reasoning_effort,
            tools,
        };
        generation.sampling = SamplingOptions {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k.unwrap_or(20),
            seed: self.seed.unwrap_or(default_seed),
        };
        generation.max_new_tokens = max_new_tokens;

        Ok(PreparedChatRequest {
            generation,
            stream: self.stream,
            split_reasoning,
            parse_tools,
        })
    }
}

fn require_default_float(
    name: &str,
    value: Option<f32>,
    default: f32,
) -> Result<(), ChatRequestError> {
    if value.is_some_and(|value| value != default) {
        return Err(ChatRequestError::Invalid(format!(
            "{name} is not implemented by the current sampler"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionRequest, ChatRequestError, SERVED_MODEL};

    fn request(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn maps_the_admitted_openai_request_to_the_real_generation_type() {
        let prepared = request(&format!(
            r#"{{
                "model":"{SERVED_MODEL}",
                "messages":[{{"role":"user","content":[
                    {{"type":"text","text":"hel"}},
                    {{"type":"text","text":"lo"}}
                ]}}],
                "tools":[{{"type":"function","function":{{"name":"bash"}}}}],
                "max_tokens":7,
                "max_completion_tokens":5,
                "temperature":0.7,
                "top_p":0.8,
                "top_k":9,
                "seed":17,
                "stream":true,
                "chat_template_kwargs":{{
                    "enable_thinking":true,
                    "preserve_thinking":false,
                    "reasoning_effort":"medium"
                }}
            }}"#
        ))
        .prepare(99)
        .unwrap();

        assert_eq!(prepared.generation.messages[0].content, "hello");
        assert_eq!(prepared.generation.max_new_tokens, 5);
        assert_eq!(prepared.generation.sampling.temperature, 0.7);
        assert_eq!(prepared.generation.sampling.top_p, 0.8);
        assert_eq!(prepared.generation.sampling.top_k, 9);
        assert_eq!(prepared.generation.sampling.seed, 17);
        assert_eq!(
            prepared.generation.template.reasoning_effort.as_deref(),
            Some("medium")
        );
        assert!(!prepared.generation.template.preserve_thinking.unwrap());
        assert_eq!(prepared.generation.template.tools.len(), 1);
        assert!(prepared.stream);
        assert!(prepared.split_reasoning);
        assert!(prepared.parse_tools);
    }

    #[test]
    fn applies_defaults_and_honors_tool_choice_none() {
        let prepared = request(&format!(
            r#"{{
                "model":"{SERVED_MODEL}",
                "messages":[{{"role":"user","content":"hello"}}],
                "tools":[{{"type":"function","function":{{"name":"bash"}}}}],
                "tool_choice":"none"
            }}"#
        ))
        .prepare(91)
        .unwrap();

        assert_eq!(prepared.generation.max_new_tokens, 128);
        assert_eq!(prepared.generation.sampling.seed, 91);
        assert!(prepared.generation.template.tools.is_empty());
        assert!(!prepared.parse_tools);
        assert!(prepared.split_reasoning);
    }

    #[test]
    fn rejects_routes_that_would_otherwise_be_silently_ignored() {
        let cases = [
            (r#""messages":[]"#, "messages must not be empty"),
            (
                r#""messages":[{"role":"user","content":"x"}],"n":2"#,
                "one completion",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"presence_penalty":1.5"#,
                "presence_penalty",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"stop":"done""#,
                "custom stop",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"tool_choice":"required""#,
                "tool_choice",
            ),
        ];
        for (fields, expected) in cases {
            let error = request(&format!(r#"{{"model":"{SERVED_MODEL}",{fields}}}"#))
                .prepare(1)
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn rejects_a_different_model_and_unimplemented_images() {
        let error =
            request(r#"{"model":"other/model","messages":[{"role":"user","content":"x"}]}"#)
                .prepare(1)
                .unwrap_err();
        assert!(matches!(error, ChatRequestError::ModelNotFound { .. }));

        let image = format!(
            r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"x"}}}}]}}]}}"#
        );
        let error = serde_json::from_str::<ChatCompletionRequest>(&image).unwrap_err();
        assert!(error.to_string().contains("vision tower"));
    }
}

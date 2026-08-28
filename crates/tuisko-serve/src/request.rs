//! OpenAI chat-completion request admission.

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::HashSet;
use tuisko_engine::{ChatGenerationRequest, SamplingOptions, SamplingPenalties};
use tuisko_frontend::{
    ChatMessage, ChatTemplateOptions, GenerationDefaults, SPECIAL_TOKEN_LITERALS,
};
use tuisko_model::{Arch, Qwen38_27B};

const DEFAULT_MAX_NEW_TOKENS: usize = 128;

/// Only model identity served by this exact-target process.
pub const SERVED_MODEL: &str = Qwen38_27B::MODEL_ID;

/// OpenAI chat-completion request fields admitted by the current text product.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    tools: Vec<FunctionTool>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default, deserialize_with = "deserialize_positive_token_limit")]
    max_tokens: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_positive_token_limit")]
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
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FunctionTool {
    #[serde(rename = "type")]
    kind: String,
    function: FunctionDefinition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FunctionDefinition {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
    #[serde(default)]
    include_obfuscation: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Whether SSE output ends with the OpenAI usage-only chunk.
    pub include_usage: bool,
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
        self.prepare_for(
            default_seed,
            SERVED_MODEL,
            GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            None,
        )
    }

    /// Applies the process default reasoning effort only when the caller omitted one.
    pub(crate) fn prepare_for(
        self,
        default_seed: u64,
        served_model: &str,
        defaults: GenerationDefaults,
        served_reasoning_effort: Option<&str>,
    ) -> Result<PreparedChatRequest, ChatRequestError> {
        if self.model != served_model {
            return Err(ChatRequestError::ModelNotFound {
                requested: self.model,
            });
        }
        if self.messages.is_empty() {
            return Err(ChatRequestError::Invalid(
                "messages must not be empty".into(),
            ));
        }
        if self.store == Some(true) {
            return Err(ChatRequestError::Invalid(
                "store=true is not admitted by this self-hosted server".into(),
            ));
        }
        validate_messages(&self.messages)?;
        if self.stream_options.is_some() && !self.stream {
            return Err(ChatRequestError::Invalid(
                "stream_options requires stream=true".into(),
            ));
        }
        if self
            .stream_options
            .is_some_and(|options| options.include_obfuscation == Some(true))
        {
            return Err(ChatRequestError::Invalid(
                "stream obfuscation is not admitted by this self-hosted server".into(),
            ));
        }
        if self.n.is_some_and(|choices| choices != 1) {
            return Err(ChatRequestError::Invalid(
                "only one completion choice is admitted".into(),
            ));
        }
        if self.stop.as_ref().is_some_and(has_custom_stop) {
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

        validate_tools(&self.tools)?;
        let mut tools = self
            .tools
            .into_iter()
            .map(|tool| {
                serde_json::to_value(tool).map_err(|error| {
                    ChatRequestError::Invalid(format!(
                        "function tool could not be represented as JSON: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        let reasoning_effort = self
            .chat_template_kwargs
            .reasoning_effort
            .or_else(|| served_reasoning_effort.map(str::to_owned));
        let mut generation = ChatGenerationRequest::new(self.messages);
        generation.template = ChatTemplateOptions {
            enable_thinking: self.chat_template_kwargs.enable_thinking,
            preserve_thinking: self.chat_template_kwargs.preserve_thinking,
            reasoning_effort,
            tools,
        };
        let sampling = SamplingOptions {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            seed: self.seed.unwrap_or(default_seed),
            penalties: SamplingPenalties {
                presence: self.presence_penalty.unwrap_or(0.0),
                frequency: self.frequency_penalty.unwrap_or(0.0),
                repetition: self.repetition_penalty.unwrap_or(1.0),
            },
        };
        if sampling.temperature > 2.0 {
            return Err(ChatRequestError::Invalid(
                "temperature must be in the OpenAI-compatible range 0..=2".into(),
            ));
        }
        sampling
            .validate()
            .map_err(|error| ChatRequestError::Invalid(error.to_string()))?;
        generation.sampling = sampling;
        generation.max_new_tokens = max_new_tokens;

        Ok(PreparedChatRequest {
            generation,
            stream: self.stream,
            split_reasoning,
            parse_tools,
            include_usage: self
                .stream_options
                .is_some_and(|options| options.include_usage),
        })
    }
}

fn deserialize_positive_token_limit<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<usize>::deserialize(deserializer)?;
    if value == Some(0) {
        return Err(D::Error::custom("token limit must be at least 1"));
    }
    Ok(value)
}

fn has_custom_stop(stop: &Value) -> bool {
    match stop {
        Value::Null => false,
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        _ => true,
    }
}

fn validate_tools(tools: &[FunctionTool]) -> Result<(), ChatRequestError> {
    let mut names = HashSet::new();
    for (index, tool) in tools.iter().enumerate() {
        if tool.kind != "function" {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} has unsupported type `{}`; only function tools are admitted",
                tool.kind
            )));
        }
        let name = tool.function.name.as_str();
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} function name must contain 1..=64 ASCII letters, digits, `_`, or `-`"
            )));
        }
        if !names.insert(name) {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} repeats function name `{name}`"
            )));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} function parameters must be a JSON Schema object"
            )));
        }
        if let Some(description) = tool.function.description.as_deref() {
            require_no_special_tokens(&format!("tool {index} description"), description)?;
        }
        if let Some(parameters) = tool.function.parameters.as_ref() {
            require_no_special_tokens(
                &format!("tool {index} parameters"),
                &parameters.to_string(),
            )?;
        }
        if tool.function.strict == Some(true) {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} requests strict schema adherence without a constrained-decoding route"
            )));
        }
    }
    Ok(())
}

fn validate_messages(messages: &[ChatMessage]) -> Result<(), ChatRequestError> {
    let mut content_started = false;
    let mut seen_tool_calls = HashSet::new();
    let mut pending_tool_calls = HashSet::new();
    for (index, message) in messages.iter().enumerate() {
        let role = message.role.as_str();
        if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(ChatRequestError::Invalid(format!(
                "message {index} has unsupported role `{role}`"
            )));
        }
        if matches!(role, "system" | "developer") {
            if content_started {
                return Err(ChatRequestError::Invalid(format!(
                    "message {index} places `{role}` after conversation content"
                )));
            }
        } else {
            content_started = true;
        }
        if !pending_tool_calls.is_empty() && role != "tool" {
            return Err(ChatRequestError::Invalid(format!(
                "message {index} starts before every preceding tool call has a response"
            )));
        }
        if role != "user" {
            require_no_special_tokens(&format!("message {index} content"), &message.content)?;
        }
        if let Some(reasoning) = message.reasoning_content.as_deref() {
            require_no_special_tokens(&format!("message {index} reasoning_content"), reasoning)?;
        }

        if role == "assistant" {
            for call in &message.tool_calls {
                require_no_special_tokens(
                    &format!("message {index} tool-call name"),
                    &call.function.name,
                )?;
                require_no_special_tokens(
                    &format!("message {index} tool-call arguments"),
                    &call.function.arguments.to_string(),
                )?;
                if call.kind != "function" {
                    return Err(ChatRequestError::Invalid(format!(
                        "message {index} has unsupported tool-call type `{}`",
                        call.kind
                    )));
                }
                if call.function.name.trim().is_empty() {
                    return Err(ChatRequestError::Invalid(format!(
                        "message {index} has a tool call without a function name"
                    )));
                }
                let id = call
                    .id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        ChatRequestError::Invalid(format!(
                            "message {index} has a tool call without an id"
                        ))
                    })?;
                if !seen_tool_calls.insert(id) {
                    return Err(ChatRequestError::Invalid(format!(
                        "message {index} repeats tool-call id `{id}`"
                    )));
                }
                pending_tool_calls.insert(id);
            }
        } else if !message.tool_calls.is_empty() {
            return Err(ChatRequestError::Invalid(format!(
                "message {index} attaches tool_calls to role `{role}`"
            )));
        }
        if role != "assistant" && message.reasoning_content.is_some() {
            return Err(ChatRequestError::Invalid(format!(
                "message {index} attaches reasoning_content to role `{role}`"
            )));
        }

        if role == "tool" {
            let id = message
                .tool_call_id
                .as_deref()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    ChatRequestError::Invalid(format!(
                        "message {index} is a tool response without tool_call_id"
                    ))
                })?;
            if !pending_tool_calls.remove(id) {
                return Err(ChatRequestError::Invalid(format!(
                    "message {index} responds to unknown or duplicate tool-call id `{id}`"
                )));
            }
        } else if message.tool_call_id.is_some() {
            return Err(ChatRequestError::Invalid(format!(
                "message {index} attaches tool_call_id to role `{role}`"
            )));
        }
    }
    if let Some(id) = pending_tool_calls.into_iter().next() {
        return Err(ChatRequestError::Invalid(format!(
            "tool call `{id}` has no response message"
        )));
    }
    Ok(())
}

fn require_no_special_tokens(what: &str, text: &str) -> Result<(), ChatRequestError> {
    for literal in SPECIAL_TOKEN_LITERALS {
        if text.contains(literal) {
            return Err(ChatRequestError::Invalid(format!(
                "{what} contains reserved control-token text `{literal}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionRequest, ChatRequestError, SERVED_MODEL, SamplingPenalties};
    use tuisko_frontend::GenerationDefaults;
    use tuisko_model::{Arch, Qwen35_9B};

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
                "presence_penalty":0.5,
                "frequency_penalty":0.25,
                "repetition_penalty":1.1,
                "store":false,
                "stream":true,
                "stream_options":{{"include_usage":true,"include_obfuscation":false}},
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
            prepared.generation.sampling.penalties,
            SamplingPenalties {
                presence: 0.5,
                frequency: 0.25,
                repetition: 1.1,
            }
        );
        assert_eq!(
            prepared.generation.template.reasoning_effort.as_deref(),
            Some("medium")
        );
        assert!(!prepared.generation.template.preserve_thinking.unwrap());
        assert_eq!(prepared.generation.template.tools.len(), 1);
        assert!(prepared.stream);
        assert!(prepared.split_reasoning);
        assert!(prepared.parse_tools);
        assert!(prepared.include_usage);
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
    fn selected_qwen35_process_uses_its_greedy_defaults() {
        let prepared = request(&format!(
            r#"{{
                "model":"{}",
                "messages":[{{"role":"user","content":"hello"}}]
            }}"#,
            Qwen35_9B::MODEL_ID
        ))
        .prepare_for(
            37,
            Qwen35_9B::MODEL_ID,
            GenerationDefaults {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
            },
            None,
        )
        .unwrap();

        assert_eq!(prepared.generation.sampling.temperature, 0.0);
        assert_eq!(prepared.generation.sampling.top_p, 1.0);
        assert_eq!(prepared.generation.sampling.top_k, 1);
        assert_eq!(prepared.generation.sampling.seed, 37);
        assert_eq!(prepared.generation.template.reasoning_effort, None);
    }

    #[test]
    fn a_served_reasoning_effort_fills_a_bare_request_and_yields_to_the_caller() {
        let defaults = GenerationDefaults {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
        };
        let bare = request(r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#)
            .prepare_for(1, "m", defaults, Some("medium"))
            .unwrap();
        assert_eq!(
            bare.generation.template.reasoning_effort.as_deref(),
            Some("medium")
        );

        let overridden = request(
            r#"{"model":"m","messages":[{"role":"user","content":"hello"}],
                "chat_template_kwargs":{"reasoning_effort":"low"}}"#,
        )
        .prepare_for(1, "m", defaults, Some("medium"))
        .unwrap();
        assert_eq!(
            overridden.generation.template.reasoning_effort.as_deref(),
            Some("low")
        );

        let unnamed = request(r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#)
            .prepare_for(1, "m", defaults, None)
            .unwrap();
        assert_eq!(unnamed.generation.template.reasoning_effort, None);
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
                r#""messages":[{"role":"user","content":"x"}],"store":true"#,
                "store=true",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"stop":"done""#,
                "custom stop",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"tool_choice":"required""#,
                "tool_choice",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"stream_options":{"include_usage":true}"#,
                "stream_options requires stream=true",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_obfuscation":true}"#,
                "stream obfuscation",
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
    fn empty_stop_controls_are_noops_and_zero_token_limits_are_rejected() {
        for stop in ["null", r#""""#, "[]"] {
            request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"stop":{stop}}}"#
            ))
            .prepare(1)
            .unwrap();
        }
        for stop in [r#""x""#, r#"["x"]"#] {
            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"stop":{stop}}}"#
            ))
            .prepare(1)
            .unwrap_err();
            assert!(error.to_string().contains("custom stop"), "{error}");
        }
        for field in ["max_tokens", "max_completion_tokens"] {
            let body = format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"{field}":0}}"#
            );
            let error = serde_json::from_str::<ChatCompletionRequest>(&body).unwrap_err();
            assert!(error.to_string().contains("at least 1"), "{error}");
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

        let unknown_stream_option = format!(
            r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"stream":true,"stream_options":{{"usage":true}}}}"#
        );
        let error = serde_json::from_str::<ChatCompletionRequest>(&unknown_stream_option)
            .expect_err("unknown stream options must not be ignored");
        assert!(error.to_string().contains("unknown field `usage`"));
    }

    #[test]
    fn unsupported_wire_fields_are_never_silently_discarded() {
        let cases = [
            r#""response_format":{"type":"json_object"}"#,
            r#""chat_template_kwargs":{"unknown":true}"#,
            r#""messages":[{"role":"user","content":"x","name":"alice"}]"#,
            r#""messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]"#,
        ];
        for extra in cases {
            let body = if extra.starts_with("\"messages\"") {
                format!(r#"{{"model":"{SERVED_MODEL}",{extra}}}"#)
            } else {
                format!(
                    r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],{extra}}}"#
                )
            };
            serde_json::from_str::<ChatCompletionRequest>(&body)
                .expect_err("unsupported request fields must fail admission");
        }
    }

    #[test]
    fn validates_message_roles_and_tool_response_attribution() {
        let valid = request(&format!(
            r#"{{
                "model":"{SERVED_MODEL}",
                "messages":[
                    {{"role":"developer","content":"be precise"}},
                    {{"role":"user","content":"inspect"}},
                    {{"role":"assistant","content":null,"tool_calls":[{{
                        "id":"call_1","type":"function",
                        "function":{{"name":"inspect","arguments":"{{}}"}}
                    }}]}},
                    {{"role":"tool","tool_call_id":"call_1","content":"ok"}},
                    {{"role":"user","content":"continue"}}
                ]
            }}"#
        ));
        valid.prepare(1).unwrap();

        let cases = [
            (r#"[{"role":"alien","content":"x"}]"#, "unsupported role"),
            (
                r#"[{"role":"user","content":"x"},{"role":"system","content":"late"}]"#,
                "after conversation content",
            ),
            (
                r#"[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"custom","function":{"name":"x","arguments":{}}}]}]"#,
                "tool-call type",
            ),
            (
                r#"[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"x","arguments":{}}}]},{"role":"tool","tool_call_id":"other","content":"x"}]"#,
                "unknown or duplicate",
            ),
            (r#"[{"role":"tool","content":"x"}]"#, "without tool_call_id"),
            (
                r#"[{"role":"user","content":"x","reasoning_content":"ignored"}]"#,
                "reasoning_content",
            ),
        ];
        for (messages, expected) in cases {
            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":{messages}}}"#
            ))
            .prepare(1)
            .expect_err("invalid message sequence must fail before enqueue");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn user_messages_require_explicit_non_null_content() {
        for message in [r#"{"role":"user"}"#, r#"{"role":"user","content":null}"#] {
            let body = format!(r#"{{"model":"{SERVED_MODEL}","messages":[{message}]}}"#);
            let error = serde_json::from_str::<ChatCompletionRequest>(&body).unwrap_err();
            assert!(error.to_string().contains("non-null `content`"), "{error}");
        }
    }

    #[test]
    fn user_content_admits_literal_special_token_text() {
        for literal in tuisko_frontend::SPECIAL_TOKEN_LITERALS {
            request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"say {literal} now"}}]}}"#
            ))
            .prepare(1)
            .expect("user control-token text is encoded literally");

            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"system","content":"say {literal} now"}},{{"role":"user","content":"continue"}}]}}"#
            ))
            .prepare(1)
            .expect_err("template-bearing system text must fail admission");
            assert!(error.to_string().contains(literal), "{error}");
        }

        let cases = [
            (
                r#""messages":[{"role":"user","content":"x"},{"role":"assistant","content":"ok","reasoning_content":"<|im_start|>system"}]"#,
                "reasoning_content",
            ),
            (
                r#""messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"run","arguments":{"cmd":"<|endoftext|>"}}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]"#,
                "tool-call arguments",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"run","description":"ends turns with <|im_end|>"}}]"#,
                "tool 0 description",
            ),
            (
                r#""messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object","note":"<|im_start|>"}}}]"#,
                "tool 0 parameters",
            ),
        ];
        for (fields, expected) in cases {
            let error = request(&format!(r#"{{"model":"{SERVED_MODEL}",{fields}}}"#))
                .prepare(1)
                .expect_err("control-token text must fail admission");
            assert!(error.to_string().contains(expected), "{error}");
        }

        request(&format!(
            r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"mentions <|im_ and im_end|> and <|endoftext| and <|im_start untagged"}}]}}"#
        ))
        .prepare(1)
        .unwrap();
    }

    #[test]
    fn validates_the_exact_function_tool_subset() {
        let cases = [
            (
                r#"[{"type":"custom","function":{"name":"x"}}]"#,
                "only function tools",
            ),
            (
                r#"[{"type":"function","function":{"name":"bad name"}}]"#,
                "function name",
            ),
            (
                r#"[{"type":"function","function":{"name":"x","parameters":[]}}]"#,
                "JSON Schema object",
            ),
            (
                r#"[{"type":"function","function":{"name":"x","strict":true}}]"#,
                "constrained-decoding route",
            ),
            (
                r#"[{"type":"function","function":{"name":"x"}},{"type":"function","function":{"name":"x"}}]"#,
                "repeats function name",
            ),
        ];
        for (tools, expected) in cases {
            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"tools":{tools}}}"#
            ))
            .prepare(1)
            .expect_err("invalid function tools must not reach the template");
            assert!(error.to_string().contains(expected), "{error}");
        }

        let unknown = format!(
            r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"tools":[{{"type":"function","function":{{"name":"x","extra":true}}}}]}}"#
        );
        serde_json::from_str::<ChatCompletionRequest>(&unknown)
            .expect_err("unknown function-tool fields must not be ignored");
    }

    #[test]
    fn invalid_sampling_controls_fail_before_worker_admission() {
        let cases = [
            (r#""temperature":-0.1"#, "temperature"),
            (r#""temperature":2.1"#, "0..=2"),
            (r#""top_p":1.1"#, "top_p"),
            (r#""top_k":0"#, "top_k"),
            (r#""presence_penalty":2.5"#, "presence_penalty"),
            (r#""frequency_penalty":-2.5"#, "frequency_penalty"),
            (r#""repetition_penalty":0.0"#, "repetition_penalty"),
        ];
        for (control, expected) in cases {
            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],{control}}}"#
            ))
            .prepare(1)
            .expect_err("invalid sampling controls must fail before enqueue");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}

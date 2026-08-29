//! OpenAI chat-completion request admission.

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;
use tuisko_engine::{ChatGenerationRequest, SamplingOptions, SamplingPenalties};
use tuisko_frontend::{
    ChatMessage, ChatTemplateOptions, GenerationDefaults, SPECIAL_TOKEN_LITERALS,
    ToolCallConstraintSpec, ToolConstraintSpec, ToolParameterSpec,
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

/// Token-ID-only OpenAI completions request admitted for lm-eval prompt scoring.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    model: String,
    prompt: CompletionPrompt,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    logprobs: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    echo: bool,
}

/// Token-ID-only native continuation log-likelihood request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoglikelihoodRequest {
    model: String,
    context: Vec<u32>,
    continuations: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
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
    /// Immutable tool-call contract shared with response validation.
    pub tool_constraint: Option<Arc<ToolCallConstraintSpec>>,
    /// Whether SSE output ends with the OpenAI usage-only chunk.
    pub include_usage: bool,
}

/// Exact token-ID prompts admitted by the evaluation scoring route.
#[derive(Debug, Eq, PartialEq)]
pub struct PreparedCompletionRequest {
    /// One to eight nonempty token-ID prompts in response-choice order.
    pub prompts: Vec<Vec<u32>>,
}

/// Exact token-ID context and continuation branches admitted for native evaluation.
#[derive(Debug, Eq, PartialEq)]
pub struct PreparedLoglikelihoodRequest {
    /// Nonempty shared context.
    pub context: Vec<u32>,
    /// One to eight nonempty continuations in response order.
    pub continuations: Vec<Vec<u32>>,
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

        let mut tool_constraint = validate_tools(&self.tools)?;
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
            Some(Value::String(choice)) if choice == "none" => {
                tools.clear();
                tool_constraint = None;
            }
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
        generation.tool_constraint = tool_constraint.clone();

        Ok(PreparedChatRequest {
            generation,
            stream: self.stream,
            split_reasoning,
            parse_tools,
            tool_constraint,
            include_usage: self
                .stream_options
                .is_some_and(|options| options.include_usage),
        })
    }
}

impl CompletionRequest {
    /// Admits only the echo-plus-one-greedy-token contract used by lm-eval loglikelihood.
    pub fn prepare_for(
        self,
        served_model: &str,
    ) -> Result<PreparedCompletionRequest, ChatRequestError> {
        if self.model != served_model {
            return Err(ChatRequestError::ModelNotFound {
                requested: self.model,
            });
        }
        if self
            .temperature
            .is_some_and(|temperature| temperature != 0.0)
        {
            return Err(ChatRequestError::Invalid(
                "prompt scoring requires temperature=0".into(),
            ));
        }
        if self.max_tokens != Some(1) {
            return Err(ChatRequestError::Invalid(
                "prompt scoring requires max_tokens=1".into(),
            ));
        }
        if self.logprobs != Some(1) {
            return Err(ChatRequestError::Invalid(
                "prompt scoring requires logprobs=1".into(),
            ));
        }
        if !self.echo {
            return Err(ChatRequestError::Invalid(
                "prompt scoring requires echo=true".into(),
            ));
        }
        let _ = self.seed;
        let prompts = match self.prompt {
            CompletionPrompt::TokenIds(prompt) => vec![prompt],
            CompletionPrompt::TokenIdBatch(prompts) => prompts,
        };
        if prompts.is_empty() || prompts.len() > 8 {
            return Err(ChatRequestError::Invalid(
                "prompt scoring requires 1..=8 prompts".into(),
            ));
        }
        if let Some(index) = prompts.iter().position(Vec::is_empty) {
            return Err(ChatRequestError::Invalid(format!(
                "prompt scoring prompt {index} is empty"
            )));
        }
        Ok(PreparedCompletionRequest { prompts })
    }
}

impl LoglikelihoodRequest {
    /// Validates the exact model, vocabulary, branch count, and resident position capacity.
    pub fn prepare_for(
        self,
        served_model: &str,
        context_capacity: usize,
    ) -> Result<PreparedLoglikelihoodRequest, ChatRequestError> {
        if self.model != served_model {
            return Err(ChatRequestError::ModelNotFound {
                requested: self.model,
            });
        }
        if served_model != SERVED_MODEL {
            return Err(ChatRequestError::Invalid(
                "native loglikelihood is unsupported for this exact target".into(),
            ));
        }
        if self.context.is_empty() {
            return Err(ChatRequestError::Invalid(
                "loglikelihood context must not be empty".into(),
            ));
        }
        if self.continuations.is_empty() || self.continuations.len() > 8 {
            return Err(ChatRequestError::Invalid(
                "loglikelihood requires 1..=8 continuations".into(),
            ));
        }
        validate_token_ids("context", 0, &self.context)?;
        for (index, continuation) in self.continuations.iter().enumerate() {
            if continuation.is_empty() {
                return Err(ChatRequestError::Invalid(format!(
                    "loglikelihood continuation {index} is empty"
                )));
            }
            validate_token_ids("continuation", index, continuation)?;
            let positions = self
                .context
                .len()
                .checked_add(continuation.len())
                .ok_or_else(|| {
                    ChatRequestError::Invalid("loglikelihood token count overflows".into())
                })?;
            if positions > context_capacity {
                return Err(ChatRequestError::Invalid(format!(
                    "loglikelihood continuation {index} requires {positions} positions, current resident capacity is {context_capacity}"
                )));
            }
        }
        Ok(PreparedLoglikelihoodRequest {
            context: self.context,
            continuations: self.continuations,
        })
    }
}

fn validate_token_ids(
    kind: &str,
    branch: usize,
    token_ids: &[u32],
) -> Result<(), ChatRequestError> {
    if let Some((position, token)) =
        token_ids.iter().copied().enumerate().find(|(_, token)| {
            usize::try_from(*token).map_or(true, |token| token >= Qwen38_27B::VOCAB)
        })
    {
        return Err(ChatRequestError::Invalid(format!(
            "loglikelihood token {token} in {kind} {branch} at position {position} is outside vocabulary 0..{}",
            Qwen38_27B::VOCAB
        )));
    }
    Ok(())
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

fn validate_tools(
    tools: &[FunctionTool],
) -> Result<Option<Arc<ToolCallConstraintSpec>>, ChatRequestError> {
    let mut names = HashSet::new();
    let mut constrained_tools = Vec::with_capacity(tools.len());
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
                "tool {index} requests strict JSON-Schema value adherence; TuiskoLLM currently constrains tool structure and declared names only"
            )));
        }
        constrained_tools.push(tool_constraint_spec(index, tool)?);
    }
    if constrained_tools.is_empty() {
        Ok(None)
    } else {
        ToolCallConstraintSpec::new(constrained_tools)
            .map(Arc::new)
            .map(Some)
            .map_err(|error| ChatRequestError::Invalid(error.to_string()))
    }
}

fn tool_constraint_spec(
    index: usize,
    tool: &FunctionTool,
) -> Result<ToolConstraintSpec, ChatRequestError> {
    let schema = tool.function.parameters.as_ref().and_then(Value::as_object);
    if let Some(kind) = schema.and_then(|schema| schema.get("type"))
        && kind != "object"
    {
        return Err(ChatRequestError::Invalid(format!(
            "tool {index} parameters type must be `object` when present"
        )));
    }
    let properties = match schema.and_then(|schema| schema.get("properties")) {
        None => serde_json::Map::new(),
        Some(Value::Object(properties)) => properties.clone(),
        Some(_) => {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} parameters properties must be an object"
            )));
        }
    };
    let required = match schema.and_then(|schema| schema.get("required")) {
        None => HashSet::new(),
        Some(Value::Array(required)) => {
            let mut names = HashSet::with_capacity(required.len());
            for value in required {
                let Some(name) = value.as_str() else {
                    return Err(ChatRequestError::Invalid(format!(
                        "tool {index} required parameters must be strings"
                    )));
                };
                if !names.insert(name.to_owned()) {
                    return Err(ChatRequestError::Invalid(format!(
                        "tool {index} repeats required parameter `{name}`"
                    )));
                }
            }
            names
        }
        Some(_) => {
            return Err(ChatRequestError::Invalid(format!(
                "tool {index} required parameters must be an array"
            )));
        }
    };
    if let Some(missing) = required
        .iter()
        .find(|name| !properties.contains_key(name.as_str()))
    {
        return Err(ChatRequestError::Invalid(format!(
            "tool {index} requires undeclared parameter `{missing}`"
        )));
    }
    let parameters = properties
        .keys()
        .map(|name| {
            ToolParameterSpec::new(name.clone(), required.contains(name))
                .map_err(|error| ChatRequestError::Invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ToolConstraintSpec::new(tool.function.name.clone(), parameters)
        .map_err(|error| ChatRequestError::Invalid(error.to_string()))
}

fn validate_messages(messages: &[ChatMessage]) -> Result<(), ChatRequestError> {
    let mut content_started = false;
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
                if !pending_tool_calls.insert(id) {
                    return Err(ChatRequestError::Invalid(format!(
                        "message {index} repeats pending tool-call id `{id}`"
                    )));
                }
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
    use super::{
        ChatCompletionRequest, ChatRequestError, CompletionRequest, LoglikelihoodRequest,
        SERVED_MODEL, SamplingPenalties,
    };
    use tuisko_frontend::GenerationDefaults;
    use tuisko_model::{Arch, Qwen35_9B, Qwen38_27B};

    fn request(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).unwrap()
    }

    fn completion(body: &str) -> CompletionRequest {
        serde_json::from_str(body).unwrap()
    }

    fn loglikelihood(body: &str) -> LoglikelihoodRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn admits_native_loglikelihood_and_rejects_every_wire_boundary() {
        let admitted = loglikelihood(&format!(
            r#"{{"model":"{SERVED_MODEL}","context":[1,2],"continuations":[[3],[4,5]]}}"#
        ))
        .prepare_for(SERVED_MODEL, 4)
        .unwrap();
        assert_eq!(admitted.context, vec![1, 2]);
        assert_eq!(admitted.continuations, vec![vec![3], vec![4, 5]]);

        for (context, continuations, capacity, expected) in [
            ("[]", "[[1]]", 8, "context must not be empty"),
            ("[1]", "[]", 8, "1..=8 continuations"),
            ("[1]", "[[]]", 8, "continuation 0 is empty"),
            ("[1,2]", "[[3,4,5]]", 4, "requires 5 positions"),
        ] {
            let error = loglikelihood(&format!(
                r#"{{"model":"{SERVED_MODEL}","context":{context},"continuations":{continuations}}}"#
            ))
            .prepare_for(SERVED_MODEL, capacity)
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        let unknown = format!(
            r#"{{"model":"{SERVED_MODEL}","context":[1],"continuations":[[2]],"text":"no"}}"#
        );
        assert!(serde_json::from_str::<LoglikelihoodRequest>(&unknown).is_err());

        let too_many = vec![vec![1]; 9];
        let body = serde_json::json!({
            "model": SERVED_MODEL,
            "context": [1],
            "continuations": too_many,
        });
        let error = serde_json::from_value::<LoglikelihoodRequest>(body)
            .unwrap()
            .prepare_for(SERVED_MODEL, 16)
            .unwrap_err();
        assert!(error.to_string().contains("1..=8 continuations"));

        let outside = u32::try_from(Qwen38_27B::VOCAB).unwrap();
        let error = loglikelihood(&format!(
            r#"{{"model":"{SERVED_MODEL}","context":[{outside}],"continuations":[[1]]}}"#
        ))
        .prepare_for(SERVED_MODEL, 4)
        .unwrap_err();
        assert!(error.to_string().contains("outside vocabulary"));
    }

    #[test]
    fn admits_the_exact_lm_eval_echo_logprob_contract() {
        let single = completion(&format!(
            r#"{{"model":"{SERVED_MODEL}","prompt":[1,2,3],"temperature":0,"max_tokens":1,"logprobs":1,"seed":1234,"echo":true}}"#
        ))
        .prepare_for(SERVED_MODEL)
        .unwrap();
        assert_eq!(single.prompts, vec![vec![1, 2, 3]]);

        let batch = completion(&format!(
            r#"{{"model":"{SERVED_MODEL}","prompt":[[1,2],[3,4,5]],"temperature":0,"max_tokens":1,"logprobs":1,"echo":true}}"#
        ))
        .prepare_for(SERVED_MODEL)
        .unwrap();
        assert_eq!(batch.prompts, vec![vec![1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn rejects_completion_modes_that_are_not_prompt_scoring() {
        let cases = [
            (
                r#""prompt":[1],"temperature":1,"max_tokens":1,"logprobs":1,"echo":true"#,
                "temperature=0",
            ),
            (
                r#""prompt":[1],"temperature":0,"max_tokens":2,"logprobs":1,"echo":true"#,
                "max_tokens=1",
            ),
            (
                r#""prompt":[1],"temperature":0,"max_tokens":1,"logprobs":2,"echo":true"#,
                "logprobs=1",
            ),
            (
                r#""prompt":[1],"temperature":0,"max_tokens":1,"logprobs":1,"echo":false"#,
                "echo=true",
            ),
            (
                r#""prompt":[],"temperature":0,"max_tokens":1,"logprobs":1,"echo":true"#,
                "prompt 0 is empty",
            ),
        ];
        for (fields, expected) in cases {
            let error = completion(&format!(r#"{{"model":"{SERVED_MODEL}",{fields}}}"#))
                .prepare_for(SERVED_MODEL)
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }

        let error = completion(&format!(
            r#"{{"model":"other/model","prompt":[1],"temperature":0,"max_tokens":1,"logprobs":1,"echo":true}}"#
        ))
        .prepare_for(SERVED_MODEL)
        .unwrap_err();
        assert!(matches!(error, ChatRequestError::ModelNotFound { .. }));
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
        let constraint = prepared.tool_constraint.as_ref().unwrap();
        assert_eq!(constraint.tools()[0].name(), "bash");
        assert!(prepared.generation.tool_constraint.is_some());
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
        assert!(prepared.tool_constraint.is_none());
        assert!(prepared.generation.tool_constraint.is_none());
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
    fn compiles_declared_and_required_tool_parameters() {
        let prepared = request(&format!(
            r#"{{
                "model":"{SERVED_MODEL}",
                "messages":[{{"role":"user","content":"inspect"}}],
                "tools":[{{"type":"function","function":{{
                    "name":"bash",
                    "parameters":{{
                        "type":"object",
                        "properties":{{"command":{{"type":"string"}},"timeout":{{"type":"number"}}}},
                        "required":["command"]
                    }}
                }}}}]
            }}"#
        ))
        .prepare(1)
        .unwrap();

        let constraint = prepared.tool_constraint.unwrap();
        let tool = &constraint.tools()[0];
        assert!(tool.parameter("command").unwrap().required());
        assert!(!tool.parameter("timeout").unwrap().required());
    }

    #[test]
    fn rejects_ambiguous_structural_tool_schemas() {
        let cases = [
            (r#"{"type":"array"}"#, "parameters type must be `object`"),
            (r#"{"properties":[]}"#, "properties must be an object"),
            (
                r#"{"properties":{"command":{"type":"string"}},"required":["missing"]}"#,
                "requires undeclared parameter",
            ),
            (
                r#"{"properties":{"bad/name":{"type":"string"}}}"#,
                "name must contain",
            ),
        ];
        for (schema, expected) in cases {
            let error = request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}}],"tools":[{{"type":"function","function":{{"name":"bash","parameters":{schema}}}}}]}}"#
            ))
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
                r#"[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"x","arguments":{}}},{"id":"call_1","type":"function","function":{"name":"y","arguments":{}}}]}]"#,
                "repeats pending tool-call id",
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
    fn completed_tool_turns_may_reuse_opaque_ids() {
        let request = request(&format!(
            r#"{{
                "model":"{SERVED_MODEL}",
                "messages":[
                    {{"role":"user","content":"inspect"}},
                    {{"role":"assistant","content":null,"tool_calls":[{{
                        "id":"call_restarted_17","type":"function",
                        "function":{{"name":"inspect","arguments":"{{}}"}}
                    }}]}},
                    {{"role":"tool","tool_call_id":"call_restarted_17","content":"ok"}},
                    {{"role":"assistant","content":null,"tool_calls":[{{
                        "id":"call_restarted_17","type":"function",
                        "function":{{"name":"inspect_again","arguments":"{{}}"}}
                    }}]}},
                    {{"role":"tool","tool_call_id":"call_restarted_17","content":"still ok"}},
                    {{"role":"user","content":"continue"}}
                ]
            }}"#
        ));

        request.prepare(1).unwrap();
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
    fn message_text_admits_literal_special_token_text() {
        for literal in tuisko_frontend::SPECIAL_TOKEN_LITERALS {
            request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"say {literal} now"}}]}}"#
            ))
            .prepare(1)
            .expect("message control-token text is encoded literally");

            request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"system","content":"say {literal} now"}},{{"role":"user","content":"continue"}}]}}"#
            ))
            .prepare(1)
            .expect("system control-token text is encoded literally");

            request(&format!(
                r#"{{"model":"{SERVED_MODEL}","messages":[{{"role":"user","content":"x"}},{{"role":"assistant","content":"quoted {literal}","reasoning_content":"consider {literal} literally"}},{{"role":"user","content":"continue"}}]}}"#
            ))
            .prepare(1)
            .expect("assistant control-token text is encoded literally");
        }

        let cases = [
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
                "strict JSON-Schema value adherence",
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

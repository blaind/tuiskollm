//! Exact tokenizer and text-only chat-template boundary.

mod error;

use minijinja::{Environment, Error as TemplateError, ErrorKind};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tuisko_model::{
    Arch, CheckpointSnapshot, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext,
};

pub use error::{FrontendError, FrontendErrorCode, FrontendResult};

const TOKENIZER_FILE: &str = "tokenizer.json";
const TEMPLATE_FILE: &str = "chat_template.jinja";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";
const IM_START_ID: u32 = 248_045;
const IM_END_ID: u32 = 248_046;
const END_OF_TEXT_ID: u32 = 248_044;
const PROMPT_BLOCK_START: &str = SPECIAL_TOKEN_LITERALS[0];
const GENERATION_BLOCK_START: &str = "<|im_start|>assistant";

/// Literal strings the pinned tokenizer always extracts as control tokens from raw text.
pub const SPECIAL_TOKEN_LITERALS: [&str; 3] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"];

/// Pinned tokenizer identity of every control literal an admitted schema may list.
const CONTROL_TOKEN_IDS: [(&str, u32); 3] = [
    (SPECIAL_TOKEN_LITERALS[0], IM_START_ID),
    (SPECIAL_TOKEN_LITERALS[1], IM_END_ID),
    (SPECIAL_TOKEN_LITERALS[2], END_OF_TEXT_ID),
];

mod private {
    /// Seals `TokenizedSchema` to the targets this crate has admitted tokenizers for.
    pub trait Sealed {}

    impl Sealed for tuisko_model::Qwen38_27B {}

    impl Sealed for tuisko_model::Qwen35_9B {}

    impl Sealed for tuisko_model::Qwen36Moe35B {}

    impl Sealed for tuisko_model::Qwen38FlashNext {}
}

/// Keeps frontend stopping and engram segmentation on the same end-of-text token.
const _: () = assert!(Qwen38FlashNext::EOS_TOKEN_ID == END_OF_TEXT_ID);

/// Shape the pinned `generation_config.json` uses to state sampling defaults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationAdmission {
    /// `do_sample` is true and `temperature`, `top_p`, and `top_k` are all present.
    Sampled,
    /// `do_sample` is false or absent and every sampling field is absent.
    Greedy,
}

/// Tokenizer, stop-token, and generation metadata of one admitted target.
///
/// Sealed because `tuisko-frontend` owns tokenizer admission. Only targets whose pinned
/// tokenizer, template, and generation config are fully admitted can implement it.
pub trait TokenizedSchema: Arch + private::Sealed {
    /// Entries in `tokenizer.json`, distinct from `Arch::VOCAB`'s padded LM-head width.
    const TOKENIZER_ENTRIES: usize;
    /// Stop token IDs registered in `generation_config.json`, in file order.
    const EOS_IDS: &'static [u32];
    /// Shape this target's `generation_config.json` must take.
    const GENERATION_ADMISSION: GenerationAdmission;
    /// Sampling defaults the pinned `generation_config.json` must state.
    const DEFAULT_GENERATION: GenerationDefaults;
    /// Control token literals extracted before BPE tokenization. Required, with no trait
    /// default: the Qwen chat-control tokens are coincidental family metadata, not an
    /// invariant, and a future family must never inherit `<|im_start|>` silently.
    const SPECIAL_TOKENS: &'static [&'static str];
}

impl TokenizedSchema for Qwen38_27B {
    const TOKENIZER_ENTRIES: usize = 248_077;
    const EOS_IDS: &'static [u32] = &[IM_END_ID, END_OF_TEXT_ID];
    const GENERATION_ADMISSION: GenerationAdmission = GenerationAdmission::Sampled;
    const DEFAULT_GENERATION: GenerationDefaults = GenerationDefaults {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 20,
    };
    const SPECIAL_TOKENS: &'static [&'static str] = &SPECIAL_TOKEN_LITERALS;
}

impl TokenizedSchema for Qwen35_9B {
    const TOKENIZER_ENTRIES: usize = 248_070;
    const EOS_IDS: &'static [u32] = &[END_OF_TEXT_ID];
    const GENERATION_ADMISSION: GenerationAdmission = GenerationAdmission::Greedy;
    const DEFAULT_GENERATION: GenerationDefaults = GenerationDefaults {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
    };
    const SPECIAL_TOKENS: &'static [&'static str] = &SPECIAL_TOKEN_LITERALS;
}

impl TokenizedSchema for Qwen36Moe35B {
    const TOKENIZER_ENTRIES: usize = 248_070;
    const EOS_IDS: &'static [u32] = &[IM_END_ID, END_OF_TEXT_ID];
    const GENERATION_ADMISSION: GenerationAdmission = GenerationAdmission::Sampled;
    const DEFAULT_GENERATION: GenerationDefaults = GenerationDefaults {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 20,
    };
    const SPECIAL_TOKENS: &'static [&'static str] = &SPECIAL_TOKEN_LITERALS;
}

/// Exact frontend metadata for the pinned Qwen3.8-Flash-Next revision.
///
/// Its tokenizer happens to match [`Qwen38_27B`], but the values remain independently pinned.
/// The admitted template is multimodal; the current transport supplies text content only.
impl TokenizedSchema for Qwen38FlashNext {
    const TOKENIZER_ENTRIES: usize = 248_077;
    const EOS_IDS: &'static [u32] = &[IM_END_ID, END_OF_TEXT_ID];
    const GENERATION_ADMISSION: GenerationAdmission = GenerationAdmission::Sampled;
    const DEFAULT_GENERATION: GenerationDefaults = GenerationDefaults {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 20,
    };
    const SPECIAL_TOKENS: &'static [&'static str] = &SPECIAL_TOKEN_LITERALS;
}

/// One text message supplied to the checkpoint chat template.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChatMessage {
    /// Template role such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Text content, accepting OpenAI text parts at the transport boundary.
    pub content: String,
    /// Earlier reasoning supplied when the template is configured to preserve it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant tool calls supplied in conversation history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    /// Tool-call identity associated with a `tool` response message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// `deny_unknown_fields` deliberately rejects OpenAI's optional `name`: the pinned chat
// template never renders it, so admitting it would silently drop caller intent.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireChatMessage {
    role: String,
    #[serde(default, deserialize_with = "deserialize_wire_chat_content")]
    content: WireMessageContent,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Default)]
enum WireMessageContent {
    #[default]
    Missing,
    Null,
    Text(String),
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let message = WireChatMessage::deserialize(deserializer)?;
        if message.role == "user" && !matches!(message.content, WireMessageContent::Text(_)) {
            return Err(D::Error::custom(
                "user message must include non-null `content`",
            ));
        }
        let content = match message.content {
            WireMessageContent::Missing | WireMessageContent::Null => String::new(),
            WireMessageContent::Text(content) => content,
        };
        Ok(Self {
            role: message.role,
            content,
            reasoning_content: message.reasoning_content,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id,
        })
    }
}

/// One OpenAI-compatible function call in conversation history.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolCall {
    /// Optional transport identity for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool-call kind; the admitted template expects `function`.
    #[serde(rename = "type", default = "default_tool_call_type")]
    pub kind: String,
    /// Function name and represented JSON arguments.
    pub function: ChatFunctionCall,
}

/// Function name and arguments carried by one historical tool call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatFunctionCall {
    /// Function name exposed to the model.
    pub name: String,
    /// JSON object accepted either directly or as an encoded JSON string.
    #[serde(
        default = "empty_tool_arguments",
        deserialize_with = "deserialize_tool_arguments"
    )]
    pub arguments: Value,
}

impl ChatMessage {
    /// Creates a text-only chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

fn default_tool_call_type() -> String {
    "function".into()
}

fn empty_tool_arguments() -> Value {
    Value::Object(Map::new())
}

fn deserialize_tool_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    let value = match value {
        None => return Ok(empty_tool_arguments()),
        Some(Value::String(encoded)) if encoded.trim().is_empty() => {
            return Ok(empty_tool_arguments());
        }
        Some(Value::String(encoded)) => serde_json::from_str(&encoded).map_err(|source| {
            D::Error::custom(format!("invalid tool-call arguments JSON: {source}"))
        })?,
        Some(value) => value,
    };
    if !value.is_object() {
        return Err(D::Error::custom(
            "tool-call arguments must encode a JSON object",
        ));
    }
    Ok(value)
}

#[derive(Deserialize)]
struct WireChatContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

fn deserialize_wire_chat_content<'de, D>(deserializer: D) -> Result<WireMessageContent, D::Error>
where
    D: Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Null => Ok(WireMessageContent::Null),
        Value::String(text) => Ok(WireMessageContent::Text(text)),
        Value::Array(parts) => {
            let mut text = String::new();
            for (index, part) in parts.into_iter().enumerate() {
                let part = WireChatContentPart::deserialize(part).map_err(|source| {
                    D::Error::custom(format!("chat content part {index} is malformed: {source}"))
                })?;
                if part.kind != "text" {
                    let detail = if part.kind == "image_url" || part.kind == "image" {
                        "image parts are not served yet: the vision tower has no device implementation"
                    } else {
                        "TuiskoLLM currently accepts text parts only"
                    };
                    return Err(D::Error::custom(format!(
                        "unsupported chat content part `{}`; {detail}",
                        part.kind
                    )));
                }
                if let Some(field) = part.extra.keys().next() {
                    return Err(D::Error::custom(format!(
                        "unsupported text content field `{field}`"
                    )));
                }
                text.push_str(
                    part.text
                        .as_deref()
                        .ok_or_else(|| D::Error::custom("text content part is missing `text`"))?,
                );
            }
            Ok(WireMessageContent::Text(text))
        }
        other => Err(D::Error::custom(format!(
            "message `content` must be a string, an array of content parts, or null, not {}",
            json_type_name(&other)
        ))),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Per-request options admitted by the current text template boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    /// Overrides the checkpoint's default thinking mode when present.
    pub enable_thinking: Option<bool>,
    /// Preserves earlier assistant reasoning when requested by the caller.
    pub preserve_thinking: Option<bool>,
    /// Checkpoint-specific reasoning budget name.
    pub reasoning_effort: Option<String>,
    /// OpenAI-compatible function definitions rendered by the checkpoint template.
    pub tools: Vec<Value>,
}

/// Startup options for the text frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextFrontendOptions {
    /// Maximum number of rendered prompts retained for prefix reuse.
    pub prompt_cache_capacity: usize,
}

impl Default for TextFrontendOptions {
    fn default() -> Self {
        Self {
            prompt_cache_capacity: 4,
        }
    }
}

/// Result and cache accounting for one encoded chat prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptEncoding {
    /// Exact prompt token IDs.
    pub token_ids: Vec<u32>,
    /// Tokens through the last complete message, before the generated assistant header.
    pub message_boundary_tokens: usize,
    /// Token IDs reused from an earlier rendering.
    pub reused_tokens: usize,
    /// Bytes in the complete rendered prompt.
    pub rendered_bytes: usize,
    /// Bytes passed through BPE during this call.
    pub fresh_bytes: usize,
}

/// Observation-only frontend timing and prefix-lookup detail for one prompt.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PromptEncodingMetrics {
    /// Wall time spent rendering the checkpoint chat template.
    pub render_us: u64,
    /// Wall time spent on prefix lookup and tokenization.
    pub encode_us: u64,
    /// Stable cache-miss reason, empty when any exact prefix was reused.
    pub miss_reason: String,
}

/// Sampling defaults admitted from `generation_config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationDefaults {
    /// Default softmax temperature.
    pub temperature: f32,
    /// Default nucleus probability.
    pub top_p: f32,
    /// Default top-k candidate count.
    pub top_k: usize,
}

struct PromptPrefixEntry {
    rendered: String,
    token_ids: Vec<u32>,
    token_ends: Vec<usize>,
}

struct CachedPrefix {
    split: usize,
    token_ids: Vec<u32>,
    token_ends: Vec<usize>,
}

/// Admitted tokenizer, chat template, and generation stop-token metadata.
pub struct TextFrontend {
    decode: Arc<DecodeState>,
    template: String,
    stop_ids: Vec<u32>,
    generation_defaults: GenerationDefaults,
    prompt_cache_capacity: usize,
    prompt_cache: Mutex<VecDeque<PromptPrefixEntry>>,
}

struct DecodeState {
    tokenizer: Tokenizer,
    literal_tokenizer: Tokenizer,
    byte_table: HashMap<char, u8>,
    special_decode_ids: HashSet<u32>,
    special_encode_tokens: Vec<(String, u32)>,
}

struct MarkedMessageText {
    start: String,
    end: String,
    content: String,
}

/// Incremental decoder for one generated text sequence.
pub struct StreamingDecoder {
    decode: Arc<DecodeState>,
    text: String,
    pending: Vec<u8>,
    finished: bool,
}

impl TextFrontend {
    /// Loads and validates tokenizer, template, and generation metadata for an admitted schema.
    pub fn open<A: TokenizedSchema>(snapshot: &CheckpointSnapshot<A>) -> FrontendResult<Self> {
        Self::open_with_options(snapshot, TextFrontendOptions::default())
    }

    /// Loads the frontend for an admitted schema with explicit startup options.
    pub fn open_with_options<A: TokenizedSchema>(
        snapshot: &CheckpointSnapshot<A>,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        Self::open_root::<A>(snapshot.root(), options)
    }

    /// Loads and validates the pinned Qwen3.5 tokenizer, template, and generation metadata.
    pub fn open_qwen35(snapshot: &CheckpointSnapshot<Qwen35_9B>) -> FrontendResult<Self> {
        Self::open(snapshot)
    }

    /// Loads the Qwen3.5 frontend with explicit startup options.
    pub fn open_qwen35_with_options(
        snapshot: &CheckpointSnapshot<Qwen35_9B>,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        Self::open_with_options(snapshot, options)
    }

    /// Loads and validates the pinned Qwen3.6 tokenizer, template, and generation metadata.
    pub fn open_qwen36(snapshot: &CheckpointSnapshot<Qwen36Moe35B>) -> FrontendResult<Self> {
        Self::open(snapshot)
    }

    /// Loads the Qwen3.6 frontend with explicit startup options.
    pub fn open_qwen36_with_options(
        snapshot: &CheckpointSnapshot<Qwen36Moe35B>,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        Self::open_with_options(snapshot, options)
    }

    fn open_root<A: TokenizedSchema>(
        root: &Path,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        let tokenizer_path = root.join(TOKENIZER_FILE);
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(tokenizer_error(
            &format!("could not load {}", tokenizer_path.display()),
        ))?;
        tokenizer
            .with_truncation(None)
            .map_err(tokenizer_error("could not disable truncation"))?;
        let byte_table = byte_level_table();
        validate_tokenizer::<A>(&tokenizer, &byte_table)?;

        let template_path = root.join(TEMPLATE_FILE);
        let template = read_string(&template_path)?;

        let generation_path = root.join(GENERATION_CONFIG_FILE);
        let generation = read_json(&generation_path)?;
        let stop_ids = parse_stop_ids::<A>(&generation)?;
        let generation_defaults = parse_generation_defaults::<A>(&generation)?;
        let mut special_encode_tokens = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter_map(|(&id, token)| token.special.then_some((token.content.clone(), id)))
            .collect::<Vec<_>>();
        special_encode_tokens.sort_unstable_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        let special_decode_ids = special_encode_tokens.iter().map(|(_, id)| *id).collect();
        let mut literal_tokenizer = tokenizer.clone();
        literal_tokenizer.set_encode_special_tokens(true);

        Ok(Self {
            decode: Arc::new(DecodeState {
                tokenizer,
                literal_tokenizer,
                byte_table,
                special_decode_ids,
                special_encode_tokens,
            }),
            template,
            stop_ids,
            generation_defaults,
            prompt_cache_capacity: options.prompt_cache_capacity,
            prompt_cache: Mutex::new(VecDeque::new()),
        })
    }

    /// Returns the pinned generation stop-token IDs.
    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    /// Returns the pinned sampling defaults.
    pub const fn generation_defaults(&self) -> GenerationDefaults {
        self.generation_defaults
    }

    /// Renders the checkpoint's text-only chat template.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        options: &ChatTemplateOptions,
    ) -> FrontendResult<String> {
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, TemplateError> {
                Err(TemplateError::new(ErrorKind::InvalidOperation, message))
            },
        );

        let mut context = Map::new();
        context.insert(
            "messages".into(),
            serde_json::to_value(messages).expect("chat messages serialize"),
        );
        context.insert(
            "add_generation_prompt".into(),
            Value::Bool(add_generation_prompt),
        );
        context.insert("add_vision_id".into(), Value::Bool(false));
        context.insert("tools".into(), Value::Array(options.tools.clone()));
        if let Some(enable_thinking) = options.enable_thinking {
            context.insert("enable_thinking".into(), Value::Bool(enable_thinking));
        }
        if let Some(preserve_thinking) = options.preserve_thinking {
            context.insert("preserve_thinking".into(), Value::Bool(preserve_thinking));
        }
        if let Some(reasoning_effort) = &options.reasoning_effort {
            context.insert(
                "reasoning_effort".into(),
                Value::String(reasoning_effort.clone()),
            );
        }

        environment
            .render_str(&self.template, Value::Object(context))
            .map_err(FrontendError::from)
    }

    /// Encodes raw text without extracting tokenizer-defined special tokens.
    pub fn encode(&self, text: &str) -> FrontendResult<Vec<u32>> {
        self.decode
            .literal_tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(tokenizer_error("could not encode text"))
    }

    /// Renders and encodes one generation prompt.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> FrontendResult<Vec<u32>> {
        self.encode_chat_with_report(messages, options)
            .map(|encoding| encoding.token_ids)
    }

    /// Renders and encodes one prompt with prefix-cache accounting.
    pub fn encode_chat_with_report(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> FrontendResult<PromptEncoding> {
        self.encode_chat_with_metrics(messages, options)
            .map(|(encoding, _)| encoding)
    }

    /// Renders and encodes one prompt with separate observation-only metrics.
    pub fn encode_chat_with_metrics(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> FrontendResult<(PromptEncoding, PromptEncodingMetrics)> {
        let render_start = std::time::Instant::now();
        if self.has_literal_message_specials(messages) {
            let (rendered, literal_ranges) =
                self.render_chat_with_literal_message_ranges(messages, options)?;
            let render_us = elapsed_microseconds(render_start);
            let encode_start = std::time::Instant::now();
            let boundary_bytes = message_boundary_bytes(&rendered)?;
            if literal_ranges.iter().any(|&(_, end)| end > boundary_bytes) {
                return Err(FrontendError::Contract(
                    "chat template placed historical message text after its generation header"
                        .into(),
                ));
            }
            let (encoding, miss_reason) = self.encode_rendered_with_literal_ranges(
                &rendered,
                boundary_bytes,
                &literal_ranges,
            )?;
            return Ok((
                encoding,
                PromptEncodingMetrics {
                    render_us,
                    encode_us: elapsed_microseconds(encode_start),
                    miss_reason: miss_reason.into(),
                },
            ));
        }
        let rendered = self.render_chat(messages, true, options)?;
        let render_us = elapsed_microseconds(render_start);
        let boundary_bytes = message_boundary_bytes(&rendered)?;
        let encode_start = std::time::Instant::now();
        let (encoding, miss_reason) =
            self.encode_rendered_with_prefix(&rendered, boundary_bytes)?;
        Ok((
            encoding,
            PromptEncodingMetrics {
                render_us,
                encode_us: elapsed_microseconds(encode_start),
                miss_reason: miss_reason.into(),
            },
        ))
    }

    /// Decodes token IDs using the admitted tokenizer.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> FrontendResult<String> {
        self.decode
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(tokenizer_error("could not decode token IDs"))
    }

    /// Starts a special-token-skipping streaming decoder.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder {
            decode: self.decode.clone(),
            text: String::new(),
            pending: Vec::new(),
            finished: false,
        }
    }

    fn encode_rendered_with_prefix(
        &self,
        rendered: &str,
        boundary_bytes: usize,
    ) -> FrontendResult<(PromptEncoding, &'static str)> {
        // Added tokens split before BPE, so restarting at a shared `<|im_start|>`
        // preserves the full-encode token sequence. Other splits fall back.
        let (cached, miss_reason) = {
            let cache = self
                .prompt_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            best_cached_prefix(&cache, rendered)
        };

        if let Some(cached) = cached {
            let split = cached.split;
            if split == rendered.len() {
                let message_boundary_tokens =
                    message_boundary_tokens(&cached.token_ends, boundary_bytes)?;
                return Ok((
                    PromptEncoding {
                        reused_tokens: cached.token_ids.len(),
                        token_ids: cached.token_ids,
                        message_boundary_tokens,
                        rendered_bytes: rendered.len(),
                        fresh_bytes: 0,
                    },
                    "",
                ));
            }

            let tail = &rendered[split..];
            let encoding = self
                .decode
                .tokenizer
                .encode(tail, false)
                .map_err(tokenizer_error("could not encode prompt tail"))?;
            let reused_tokens = cached.token_ids.len();
            let mut token_ids = cached.token_ids;
            token_ids.extend_from_slice(encoding.get_ids());
            let mut token_ends = cached.token_ends;
            token_ends.extend(encoding.get_offsets().iter().map(|&(_, end)| split + end));
            let message_boundary_tokens = message_boundary_tokens(&token_ends, boundary_bytes)?;
            self.push_prompt_entry(rendered, token_ids.clone(), token_ends);

            return Ok((
                PromptEncoding {
                    token_ids,
                    message_boundary_tokens,
                    reused_tokens,
                    rendered_bytes: rendered.len(),
                    fresh_bytes: tail.len(),
                },
                "",
            ));
        }

        let encoding = self
            .decode
            .tokenizer
            .encode(rendered, false)
            .map_err(tokenizer_error("could not encode prompt"))?;
        let token_ids = encoding.get_ids().to_vec();
        let token_ends = encoding
            .get_offsets()
            .iter()
            .map(|&(_, end)| end)
            .collect::<Vec<_>>();
        let message_boundary_tokens = message_boundary_tokens(&token_ends, boundary_bytes)?;
        self.push_prompt_entry(rendered, token_ids.clone(), token_ends);

        Ok((
            PromptEncoding {
                token_ids,
                message_boundary_tokens,
                reused_tokens: 0,
                rendered_bytes: rendered.len(),
                fresh_bytes: rendered.len(),
            },
            miss_reason,
        ))
    }

    fn has_literal_message_specials(&self, messages: &[ChatMessage]) -> bool {
        messages.iter().any(|message| {
            self.contains_special_token(&message.content)
                || message
                    .reasoning_content
                    .as_deref()
                    .is_some_and(|reasoning| self.contains_special_token(reasoning))
        })
    }

    fn contains_special_token(&self, text: &str) -> bool {
        self.decode
            .special_encode_tokens
            .iter()
            .any(|(token, _)| text.contains(token))
    }

    fn render_chat_with_literal_message_ranges(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> FrontendResult<(String, Vec<(usize, usize)>)> {
        let mut occupied = self.template.clone();
        occupied.push_str(
            &serde_json::to_string(messages).expect("chat messages serialize for marker admission"),
        );
        occupied.push_str(&Value::Array(options.tools.clone()).to_string());
        if let Some(reasoning_effort) = &options.reasoning_effort {
            occupied.push_str(reasoning_effort);
        }
        let marker_prefix = (0usize..)
            .map(|nonce| format!("\u{e000}tuisko-message-{nonce}-"))
            .find(|prefix| !occupied.contains(prefix))
            .expect("an unused finite marker prefix exists");

        let mut marked_messages = messages.to_vec();
        let mut marked = Vec::new();
        for (index, message) in marked_messages.iter_mut().enumerate() {
            if self.contains_special_token(&message.content) {
                let start = format!("{marker_prefix}{index}-content-start\u{e001}");
                let end = format!("{marker_prefix}{index}-content-end\u{e001}");
                let content = std::mem::take(&mut message.content);
                message.content = format!("{start}{content}{end}");
                marked.push(MarkedMessageText {
                    start,
                    end,
                    content,
                });
            }
            if let Some(reasoning) = message.reasoning_content.as_mut()
                && self.contains_special_token(reasoning)
            {
                let start = format!("{marker_prefix}{index}-reasoning-start\u{e001}");
                let end = format!("{marker_prefix}{index}-reasoning-end\u{e001}");
                let content = std::mem::take(reasoning);
                *reasoning = format!("{start}{content}{end}");
                marked.push(MarkedMessageText {
                    start,
                    end,
                    content,
                });
            }
        }

        let marked_rendered = self.render_chat(&marked_messages, true, options)?;
        let mut placed = marked
            .into_iter()
            .map(|marker| {
                let start = marked_rendered.find(&marker.start).ok_or_else(|| {
                    FrontendError::Contract("chat template omitted a message-text marker".into())
                })?;
                let content_start = start + marker.start.len();
                let content_end = marked_rendered[content_start..]
                    .find(&marker.end)
                    .map(|offset| content_start + offset)
                    .ok_or_else(|| {
                        FrontendError::Contract(
                            "chat template omitted a message-text boundary".into(),
                        )
                    })?;
                if marked_rendered[content_start..content_end] != marker.content {
                    return Err(FrontendError::Contract(
                        "chat template transformed marked message text".into(),
                    ));
                }
                Ok((start, content_start, content_end, marker))
            })
            .collect::<FrontendResult<Vec<_>>>()?;
        placed.sort_by_key(|&(start, _, _, _)| start);

        let mut rendered = String::with_capacity(marked_rendered.len());
        let mut literal_ranges = Vec::with_capacity(placed.len());
        let mut cursor = 0;
        for (start, _, content_end, marker) in placed {
            if start < cursor {
                return Err(FrontendError::Contract(
                    "chat template overlapped message-text markers".into(),
                ));
            }
            rendered.push_str(&marked_rendered[cursor..start]);
            let literal_start = rendered.len();
            rendered.push_str(&marker.content);
            literal_ranges.push((literal_start, rendered.len()));
            cursor = content_end + marker.end.len();
        }
        rendered.push_str(&marked_rendered[cursor..]);
        Ok((rendered, literal_ranges))
    }

    fn encode_rendered_with_literal_ranges(
        &self,
        rendered: &str,
        boundary_bytes: usize,
        literal_ranges: &[(usize, usize)],
    ) -> FrontendResult<(PromptEncoding, &'static str)> {
        let (cached, miss_reason) = {
            let cache = self
                .prompt_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            best_cached_prefix(&cache, rendered)
        };
        if let Some(cached) = cached {
            let split = cached.split;
            if split == rendered.len() {
                let message_boundary_tokens =
                    message_boundary_tokens(&cached.token_ends, boundary_bytes)?;
                return Ok((
                    PromptEncoding {
                        reused_tokens: cached.token_ids.len(),
                        token_ids: cached.token_ids,
                        message_boundary_tokens,
                        rendered_bytes: rendered.len(),
                        fresh_bytes: 0,
                    },
                    "",
                ));
            }

            let tail_ranges = literal_ranges
                .iter()
                .filter_map(|&(start, end)| {
                    (end > split).then_some((start.saturating_sub(split), end - split))
                })
                .collect::<Vec<_>>();
            let (tail_ids, tail_ends) =
                self.encode_rendered_segments(&rendered[split..], &tail_ranges)?;
            let reused_tokens = cached.token_ids.len();
            let mut token_ids = cached.token_ids;
            token_ids.extend(tail_ids);
            let mut token_ends = cached.token_ends;
            token_ends.extend(tail_ends.into_iter().map(|end| split + end));
            let message_boundary_tokens = message_boundary_tokens(&token_ends, boundary_bytes)?;
            self.push_prompt_entry(rendered, token_ids.clone(), token_ends);
            return Ok((
                PromptEncoding {
                    token_ids,
                    message_boundary_tokens,
                    reused_tokens,
                    rendered_bytes: rendered.len(),
                    fresh_bytes: rendered.len() - split,
                },
                "",
            ));
        }

        let (token_ids, token_ends) = self.encode_rendered_segments(rendered, literal_ranges)?;
        let message_boundary_tokens = message_boundary_tokens(&token_ends, boundary_bytes)?;
        self.push_prompt_entry(rendered, token_ids.clone(), token_ends);
        Ok((
            PromptEncoding {
                token_ids,
                message_boundary_tokens,
                reused_tokens: 0,
                rendered_bytes: rendered.len(),
                fresh_bytes: rendered.len(),
            },
            miss_reason,
        ))
    }

    fn encode_rendered_segments(
        &self,
        rendered: &str,
        literal_ranges: &[(usize, usize)],
    ) -> FrontendResult<(Vec<u32>, Vec<usize>)> {
        let mut token_ids = Vec::new();
        let mut token_ends = Vec::new();
        let mut text_start = 0;
        let mut search_start = 0;
        while let Some((start, token, id)) = self.next_special_token(rendered, search_start) {
            let end = start + token.len();
            if literal_ranges
                .iter()
                .any(|&(literal_start, literal_end)| start < literal_end && end > literal_start)
            {
                search_start = end;
                continue;
            }
            self.append_literal_encoding(
                &rendered[text_start..start],
                text_start,
                &mut token_ids,
                &mut token_ends,
            )?;
            token_ids.push(id);
            token_ends.push(end);
            text_start = end;
            search_start = end;
        }
        self.append_literal_encoding(
            &rendered[text_start..],
            text_start,
            &mut token_ids,
            &mut token_ends,
        )?;
        Ok((token_ids, token_ends))
    }

    fn next_special_token<'a>(
        &'a self,
        rendered: &'a str,
        search_start: usize,
    ) -> Option<(usize, &'a str, u32)> {
        self.decode
            .special_encode_tokens
            .iter()
            .filter_map(|(token, id)| {
                rendered[search_start..]
                    .find(token)
                    .map(|offset| (search_start + offset, token.as_str(), *id))
            })
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.len().cmp(&left.1.len()))
            })
    }

    fn append_literal_encoding(
        &self,
        text: &str,
        offset: usize,
        token_ids: &mut Vec<u32>,
        token_ends: &mut Vec<usize>,
    ) -> FrontendResult<()> {
        if text.is_empty() {
            return Ok(());
        }
        let encoding = self
            .decode
            .literal_tokenizer
            .encode(text, false)
            .map_err(tokenizer_error("could not encode literal prompt text"))?;
        token_ids.extend_from_slice(encoding.get_ids());
        token_ends.extend(encoding.get_offsets().iter().map(|&(_, end)| offset + end));
        Ok(())
    }

    fn push_prompt_entry(&self, rendered: &str, token_ids: Vec<u32>, token_ends: Vec<usize>) {
        debug_assert_eq!(token_ids.len(), token_ends.len());
        if self.prompt_cache_capacity == 0 {
            return;
        }

        let mut cache = self
            .prompt_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.push_back(PromptPrefixEntry {
            rendered: rendered.into(),
            token_ids,
            token_ends,
        });
        while cache.len() > self.prompt_cache_capacity {
            cache.pop_front();
        }
    }
}

impl StreamingDecoder {
    /// Decodes one generated token and returns any newly complete text.
    pub fn push(&mut self, token_id: u32) -> FrontendResult<Option<String>> {
        if self.finished {
            return Err(FrontendError::Contract(
                "cannot push a token after finishing the stream".into(),
            ));
        }
        if self.decode.special_decode_ids.contains(&token_id) {
            return Ok(None);
        }

        let content = self.decode.tokenizer.id_to_token(token_id).ok_or_else(|| {
            FrontendError::Tokenizer(format!("token ID {token_id} has no tokenizer entry"))
        })?;
        let mut delta = String::new();
        for character in content.chars() {
            let byte = self
                .decode
                .byte_table
                .get(&character)
                .copied()
                .ok_or_else(|| {
                    FrontendError::Tokenizer(format!(
                        "token ID {token_id} contains non-byte-level character {character:?}"
                    ))
                })?;
            push_stream_byte(&mut self.pending, &mut delta, byte);
        }

        if delta.is_empty() {
            Ok(None)
        } else {
            self.text.push_str(&delta);
            Ok(Some(delta))
        }
    }

    /// Finishes the stream and flushes an incomplete UTF-8 suffix.
    pub fn finish(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        self.finished = true;

        let delta = finish_pending(&mut self.pending);
        if delta.is_empty() {
            None
        } else {
            self.text.push_str(&delta);
            Some(delta)
        }
    }

    /// Returns all text emitted by this decoder.
    pub fn text(&self) -> &str {
        &self.text
    }
}

fn byte_level_table() -> HashMap<char, u8> {
    let mut table = HashMap::with_capacity(256);
    let mut replacement = 0u32;
    for byte in 0..=u8::MAX {
        let character = if (0x21..=0x7e).contains(&byte)
            || (0xa1..=0xac).contains(&byte)
            || (0xae..=0xff).contains(&byte)
        {
            char::from(byte)
        } else {
            let character = char::from_u32(0x100 + replacement).expect("byte-level character");
            replacement += 1;
            character
        };
        table.insert(character, byte);
    }
    table
}

fn push_stream_byte(pending: &mut Vec<u8>, output: &mut String, byte: u8) {
    if pending.is_empty() {
        if byte < 0x80 {
            output.push(char::from(byte));
        } else if (0x80..=0xbf).contains(&byte) {
            output.push('\u{fffd}');
        } else if utf8_sequence_length(byte).is_some() {
            pending.push(byte);
        } else {
            output.push('\u{fffd}');
        }
        return;
    }

    let mut candidate = pending.clone();
    candidate.push(byte);
    if (0x80..=0xbf).contains(&byte) && is_valid_utf8_prefix(&candidate) {
        // `is_valid_utf8_prefix` accepts exactly the well-formed prefixes, so a candidate that
        // does not decode is always an incomplete sequence.
        if let Ok(text) = std::str::from_utf8(&candidate) {
            output.push_str(text);
            pending.clear();
        } else {
            *pending = candidate;
        }
        return;
    }

    output.push('\u{fffd}');
    pending.clear();
    push_stream_byte(pending, output, byte);
}

fn finish_pending(pending: &mut Vec<u8>) -> String {
    let delta = String::from_utf8_lossy(pending).into_owned();
    pending.clear();
    delta
}

fn message_boundary_bytes(rendered: &str) -> FrontendResult<usize> {
    rendered.rfind(GENERATION_BLOCK_START).ok_or_else(|| {
        FrontendError::Contract(
            "rendered chat prompt has no generated-assistant message boundary".into(),
        )
    })
}

fn message_boundary_tokens(token_ends: &[usize], boundary_bytes: usize) -> FrontendResult<usize> {
    let tokens = token_ends.partition_point(|&end| end <= boundary_bytes);
    if tokens == 0 || token_ends[tokens - 1] != boundary_bytes {
        return Err(FrontendError::Contract(
            "generated-assistant message boundary is not token-aligned".into(),
        ));
    }
    Ok(tokens)
}

fn utf8_sequence_length(lead: u8) -> Option<u8> {
    match lead {
        0x00..=0x7f => Some(1),
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

fn is_valid_utf8_prefix(bytes: &[u8]) -> bool {
    if bytes.is_empty() || bytes.len() > usize::from(utf8_sequence_length(bytes[0]).unwrap_or(0)) {
        return false;
    }
    if bytes.len() >= 2 {
        let second_is_valid = match bytes[0] {
            0xc2..=0xdf => (0x80..=0xbf).contains(&bytes[1]),
            0xe0 => (0xa0..=0xbf).contains(&bytes[1]),
            0xe1..=0xec | 0xee..=0xef => (0x80..=0xbf).contains(&bytes[1]),
            0xed => (0x80..=0x9f).contains(&bytes[1]),
            0xf0..=0xf3 => (0x90..=0xbf).contains(&bytes[1]),
            0xf4 => (0x80..=0x8f).contains(&bytes[1]),
            _ => false,
        };
        if !second_is_valid {
            return false;
        }
    }

    bytes
        .iter()
        .skip(2)
        .all(|byte| (0x80..=0xbf).contains(byte))
}

fn best_cached_prefix(
    cache: &VecDeque<PromptPrefixEntry>,
    rendered: &str,
) -> (Option<CachedPrefix>, &'static str) {
    if cache.is_empty() {
        return (None, "cache-empty");
    }

    let mut best = None;
    let mut saw_block_start = false;
    for entry in cache {
        if entry.rendered == rendered {
            best = Some(CachedPrefix {
                split: rendered.len(),
                token_ids: entry.token_ids.clone(),
                token_ends: entry.token_ends.clone(),
            });
            continue;
        }

        let common = common_prefix_bytes(&entry.rendered, rendered);
        let Some(split) = rendered[..common].rfind(PROMPT_BLOCK_START) else {
            continue;
        };
        saw_block_start = true;
        let Some(last_token) = entry.token_ends.iter().position(|&end| end == split) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|cached: &CachedPrefix| cached.split < split)
        {
            let reused_tokens = last_token + 1;
            best = Some(CachedPrefix {
                split,
                token_ids: entry.token_ids[..reused_tokens].to_vec(),
                token_ends: entry.token_ends[..reused_tokens].to_vec(),
            });
        }
    }
    let miss_reason = if best.is_some() {
        ""
    } else if saw_block_start {
        "block-start-not-token-end"
    } else {
        "no-shared-block-start"
    };
    (best, miss_reason)
}

fn elapsed_microseconds(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    let limit = left.len().min(right.len());
    let left_bytes = left.as_bytes();
    let right_bytes = right.as_bytes();
    let mut length = 0;
    while length < limit && left_bytes[length] == right_bytes[length] {
        length += 1;
    }
    while length > 0 && !(left.is_char_boundary(length) && right.is_char_boundary(length)) {
        length -= 1;
    }
    length
}

fn tokenizer_error<E: std::fmt::Display>(context: &str) -> impl FnOnce(E) -> FrontendError + '_ {
    move |source| FrontendError::Tokenizer(format!("{context}: {source}"))
}

fn validate_tokenizer<A: TokenizedSchema>(
    tokenizer: &Tokenizer,
    byte_table: &HashMap<char, u8>,
) -> FrontendResult<()> {
    let entries = tokenizer.get_vocab_size(true);
    let expected_entries = A::TOKENIZER_ENTRIES;
    if entries != expected_entries {
        return Err(FrontendError::Contract(format!(
            "tokenizer has {entries} entries, expected {expected_entries}"
        )));
    }

    for &token in A::SPECIAL_TOKENS {
        let expected = CONTROL_TOKEN_IDS
            .iter()
            .find_map(|&(literal, id)| (literal == token).then_some(id))
            .ok_or_else(|| {
                FrontendError::Contract(format!("control token `{token}` has no pinned ID"))
            })?;
        let actual = tokenizer.token_to_id(token);
        if actual != Some(expected) {
            return Err(FrontendError::Contract(format!(
                "tokenizer maps `{token}` to {actual:?}, expected {expected}"
            )));
        }
    }

    validate_added_token_alphabet(tokenizer, byte_table)
}

/// Pins the streaming decoder's assumption that every decodable added token is byte-level.
fn validate_added_token_alphabet(
    tokenizer: &Tokenizer,
    byte_table: &HashMap<char, u8>,
) -> FrontendResult<()> {
    let offender = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| !token.special)
        .filter_map(|(id, token)| {
            token
                .content
                .chars()
                .find(|character| !byte_table.contains_key(character))
                .map(|character| (id, character))
        })
        .min_by_key(|&(id, _)| id);
    if let Some((id, character)) = offender {
        return Err(FrontendError::Contract(format!(
            "added token {id} contains non-byte-level character {character:?}"
        )));
    }

    Ok(())
}

fn read_string(path: &Path) -> FrontendResult<String> {
    fs::read_to_string(path).map_err(|source| FrontendError::Io {
        path: path.to_owned(),
        source,
    })
}

fn read_json(path: &Path) -> FrontendResult<Value> {
    let bytes = fs::read(path).map_err(|source| FrontendError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| FrontendError::Json {
        path: path.to_owned(),
        source,
    })
}

fn parse_stop_ids<A: TokenizedSchema>(generation: &Value) -> FrontendResult<Vec<u32>> {
    let value = generation.get("eos_token_id").ok_or_else(|| {
        FrontendError::Contract("generation_config.json is missing `eos_token_id`".into())
    })?;
    let values = match value {
        Value::Array(values) => values.clone(),
        value => vec![value.clone()],
    };
    let stop_ids = values
        .iter()
        .map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json contains a non-u32 stop ID".into())
        })?;
    let expected = A::EOS_IDS;
    if stop_ids != expected {
        return Err(FrontendError::Contract(format!(
            "generation stop IDs {stop_ids:?} do not match {expected:?}"
        )));
    }

    Ok(stop_ids)
}

fn parse_generation_defaults<A: TokenizedSchema>(
    generation: &Value,
) -> FrontendResult<GenerationDefaults> {
    match A::GENERATION_ADMISSION {
        GenerationAdmission::Sampled => {
            parse_sampled_generation_defaults(generation, A::DEFAULT_GENERATION)
        }
        GenerationAdmission::Greedy => {
            parse_greedy_generation_defaults(generation, A::DEFAULT_GENERATION)
        }
    }
}

fn parse_sampled_generation_defaults(
    generation: &Value,
    expected: GenerationDefaults,
) -> FrontendResult<GenerationDefaults> {
    if generation.get("do_sample").and_then(Value::as_bool) != Some(true) {
        return Err(FrontendError::Contract(
            "generation_config.json `do_sample` must be true".into(),
        ));
    }
    let temperature = generation
        .get("temperature")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json `temperature` must be a number".into())
        })? as f32;
    let top_p = generation
        .get("top_p")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json `top_p` must be a number".into())
        })? as f32;
    let top_k = generation
        .get("top_k")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json `top_k` must be a usize".into())
        })?;
    let defaults = GenerationDefaults {
        temperature,
        top_p,
        top_k,
    };
    if defaults != expected {
        return Err(FrontendError::Contract(format!(
            "generation defaults {defaults:?} do not match {expected:?}"
        )));
    }

    Ok(defaults)
}

fn parse_greedy_generation_defaults(
    generation: &Value,
    defaults: GenerationDefaults,
) -> FrontendResult<GenerationDefaults> {
    if generation
        .get("do_sample")
        .is_some_and(|value| value.as_bool() != Some(false))
    {
        return Err(FrontendError::Contract(
            "generation_config.json `do_sample` must be false or absent".into(),
        ));
    }
    for field in ["temperature", "top_p", "top_k"] {
        if generation.get(field).is_some() {
            return Err(FrontendError::Contract(format!(
                "generation_config.json `{field}` must be absent when sampling is disabled"
            )));
        }
    }

    Ok(defaults)
}

#[cfg(test)]
mod tests {
    use super::{
        CONTROL_TOKEN_IDS, CachedPrefix, ChatFunctionCall, ChatMessage, ChatTemplateOptions,
        ChatToolCall, END_OF_TEXT_ID, FrontendErrorCode, FrontendResult, GENERATION_CONFIG_FILE,
        GenerationAdmission, GenerationDefaults, IM_START_ID, PROMPT_BLOCK_START,
        PromptPrefixEntry, SPECIAL_TOKEN_LITERALS, TextFrontend, TextFrontendOptions,
        TokenizedSchema, best_cached_prefix, byte_level_table, common_prefix_bytes, finish_pending,
        is_valid_utf8_prefix, parse_generation_defaults, parse_stop_ids, push_stream_byte,
        read_json, utf8_sequence_length, validate_added_token_alphabet, validate_tokenizer,
    };
    use serde_json::json;
    use std::collections::{HashSet, VecDeque};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer};
    use tuisko_model::{
        Arch, CheckpointSnapshot, Qwen35_9B, Qwen36Moe35B, Qwen38_27B, Qwen38FlashNext,
    };

    // Transformers 5.2.0 `apply_chat_template` and tokenizer output from the pinned snapshot.
    const QWEN36_HELLO_THINKING: [u32; 11] = [
        248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
    ];
    const QWEN36_HELLO_NO_THINKING: [u32; 13] = [
        248_045, 846, 198, 9_419, 248_046, 198, 248_045, 74_455, 198, 248_068, 271, 248_069, 271,
    ];

    #[test]
    fn chat_message_serializes_for_the_template() {
        let message = ChatMessage::new("user", "Hello");
        assert_eq!(
            serde_json::to_value(message).unwrap(),
            json!({"role": "user", "content": "Hello"})
        );
        assert_eq!(ChatTemplateOptions::default().enable_thinking, None);
    }

    #[test]
    fn chat_message_accepts_openai_text_content_shapes() {
        let scalar: ChatMessage =
            serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        let parts: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hel"},{"type":"text","text":"lo"}]}"#,
        )
        .unwrap();
        let null: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
        let missing: ChatMessage = serde_json::from_str(r#"{"role":"assistant"}"#).unwrap();

        assert_eq!(scalar.content, "hello");
        assert_eq!(parts.content, "hello");
        assert_eq!(null.content, "");
        assert_eq!(missing.content, "");
    }

    #[test]
    fn user_message_requires_explicit_non_null_content() {
        for message in [r#"{"role":"user"}"#, r#"{"role":"user","content":null}"#] {
            let error = serde_json::from_str::<ChatMessage>(message).unwrap_err();
            assert!(error.to_string().contains("non-null `content`"), "{error}");
        }
    }

    #[test]
    fn chat_message_refuses_unimplemented_image_content() {
        let error = serde_json::from_str::<ChatMessage>(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}"#,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("image_url"), "{error}");
        assert!(error.contains("not served yet"), "{error}");
        assert!(error.contains("device"), "{error}");
    }

    #[test]
    fn malformed_content_parts_keep_their_own_diagnostics() {
        for (message, expected) in [
            (
                r#"{"role":"user","content":[{"text":"hello"}]}"#,
                "chat content part 0 is malformed: missing field `type`",
            ),
            (
                r#"{"role":"user","content":[{"type":"text","text":"hi"},{"type":7}]}"#,
                "chat content part 1 is malformed",
            ),
            (
                r#"{"role":"user","content":7}"#,
                "message `content` must be a string, an array of content parts, or null, not a number",
            ),
        ] {
            let error = serde_json::from_str::<ChatMessage>(message).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn historical_tool_calls_accept_openai_argument_strings() {
        let message: ChatMessage = serde_json::from_str(
            r#"{
                "role":"assistant",
                "content":null,
                "reasoning_content":"inspect the directory",
                "tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"bash","arguments":"{\"command\":\"ls -la\"}"}
                }]
            }"#,
        )
        .unwrap();

        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("inspect the directory")
        );
        assert_eq!(message.tool_calls[0].function.name, "bash");
        assert_eq!(
            message.tool_calls[0].function.arguments,
            json!({"command": "ls -la"})
        );
        assert!(
            serde_json::to_value(message).unwrap()["tool_calls"][0]["function"]["arguments"]
                .is_object()
        );
    }

    #[test]
    fn historical_tool_calls_reject_non_object_arguments() {
        let error = serde_json::from_str::<ChatMessage>(
            r#"{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"bash","arguments":"[]"}}]}"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("must encode a JSON object"));
    }

    #[test]
    fn stop_ids_are_exact() {
        let qwen38 = json!({"eos_token_id": [248046, 248044]});
        let qwen35 = json!({"eos_token_id": 248044});
        let qwen36 = json!({"eos_token_id": [248046, 248044]});
        assert_eq!(
            parse_stop_ids::<Qwen38_27B>(&qwen38).unwrap(),
            [248046, 248044]
        );
        assert_eq!(parse_stop_ids::<Qwen35_9B>(&qwen35).unwrap(), [248044]);
        assert_eq!(
            parse_stop_ids::<Qwen36Moe35B>(&qwen36).unwrap(),
            [248046, 248044]
        );
        assert_eq!(
            parse_stop_ids::<Qwen38FlashNext>(&json!({"eos_token_id": [248046, 248044]})).unwrap(),
            [248046, 248044]
        );
    }

    #[test]
    fn stop_id_shape_and_values_are_rejected() {
        for generation in [
            json!({}),
            json!({"eos_token_id": 248046}),
            json!({"eos_token_id": [248046]}),
            json!({"eos_token_id": [248046, 248043]}),
            json!({"eos_token_id": [248046, -1]}),
        ] {
            let error = parse_stop_ids::<Qwen38_27B>(&generation).unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
        for generation in [
            json!({}),
            json!({"eos_token_id": [248044, 248046]}),
            json!({"eos_token_id": -1}),
        ] {
            let error = parse_stop_ids::<Qwen35_9B>(&generation).unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
        for generation in [
            json!({}),
            json!({"eos_token_id": 248046}),
            json!({"eos_token_id": [248044, 248046]}),
        ] {
            let error = parse_stop_ids::<Qwen36Moe35B>(&generation).unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
        for generation in [
            json!({}),
            json!({"eos_token_id": 248044}),
            json!({"eos_token_id": [248044, 248046]}),
            json!({"eos_token_id": [248046, 248044, 248069]}),
        ] {
            let error = parse_stop_ids::<Qwen38FlashNext>(&generation).unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
    }

    #[test]
    fn streaming_utf8_matches_batch_lossy_decode() {
        let cases = [
            vec![],
            vec![0x41, 0x42, 0x43],
            vec![0xe4],
            vec![0xe4, 0x41],
            vec![0xe4, 0xb0, 0x80],
            vec![0xf0, 0x9f],
            vec![0xf0, 0x9f, 0x9a, 0x80],
            vec![0x80],
            vec![0xc0, 0x41],
            vec![0xe4, 0xe4],
            vec![0xf0, 0x80, 0x80, 0x80],
            vec![0xed, 0xa0, 0x80],
            vec![0x41, 0xe4, 0xb0, 0x80, 0x42],
        ];

        for bytes in cases {
            let mut pending = Vec::new();
            let mut actual = String::new();
            for &byte in &bytes {
                push_stream_byte(&mut pending, &mut actual, byte);
            }
            actual.push_str(&finish_pending(&mut pending));

            assert_eq!(actual, String::from_utf8_lossy(&bytes));
            assert!(pending.is_empty());
        }
    }

    #[test]
    fn accepted_full_length_utf8_candidates_always_decode() {
        fn walk(candidate: &mut Vec<u8>, length: usize) {
            if candidate.len() == length {
                assert!(std::str::from_utf8(candidate).is_ok(), "{candidate:02x?}");
                return;
            }
            for byte in 0x80..=0xbfu8 {
                candidate.push(byte);
                if is_valid_utf8_prefix(candidate) {
                    walk(candidate, length);
                }
                candidate.pop();
            }
        }

        for lead in 0..=u8::MAX {
            let Some(length) = utf8_sequence_length(lead) else {
                continue;
            };
            walk(&mut vec![lead], usize::from(length));
        }
    }

    #[test]
    fn added_tokens_outside_the_byte_level_alphabet_fail_admission() {
        let vocab = [("<unk>".to_owned(), 0), ("北".to_owned(), 1)]
            .into_iter()
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let byte_table = byte_level_table();

        let mut tokenizer = Tokenizer::new(model);
        tokenizer
            .add_special_tokens([AddedToken::from("北", true)])
            .unwrap();
        validate_added_token_alphabet(&tokenizer, &byte_table).unwrap();

        tokenizer
            .add_tokens([AddedToken::from("北", false)])
            .unwrap();
        let error = validate_added_token_alphabet(&tokenizer, &byte_table).unwrap_err();
        assert_eq!(error.code(), FrontendErrorCode::Contract);
        assert!(
            error.to_string().contains("non-byte-level character"),
            "{error}"
        );
    }

    #[test]
    fn streaming_decoder_skips_specials_across_utf8_boundaries() {
        let byte_table = byte_level_table();
        let token = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| {
                    byte_table
                        .iter()
                        .find_map(|(&character, value)| (*value == *byte).then_some(character))
                        .unwrap()
                })
                .collect::<String>()
        };
        let vocab = [
            (token(b"H"), 0),
            (token(&[0xf0]), 1),
            (token(&[0x9f, 0x9a, 0x80]), 2),
            ("<eos>".into(), 3),
            ("<unk>".into(), 4),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        let mut literal_tokenizer = tokenizer.clone();
        literal_tokenizer.set_encode_special_tokens(true);
        let frontend = TextFrontend {
            decode: std::sync::Arc::new(super::DecodeState {
                tokenizer,
                literal_tokenizer,
                byte_table,
                special_decode_ids: HashSet::from([3]),
                special_encode_tokens: vec![("<special>".into(), 3)],
            }),
            template: String::new(),
            stop_ids: vec![3],
            generation_defaults: super::GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            prompt_cache_capacity: 0,
            prompt_cache: std::sync::Mutex::new(VecDeque::new()),
        };

        let mut decoder = frontend.streaming_decoder();
        drop(frontend);
        assert_eq!(decoder.push(0).unwrap().as_deref(), Some("H"));
        assert_eq!(decoder.push(1).unwrap(), None);
        assert_eq!(decoder.push(3).unwrap(), None);
        assert_eq!(decoder.push(2).unwrap().as_deref(), Some("🚀"));
        assert_eq!(decoder.finish(), None);
        assert_eq!(decoder.text(), "H🚀");
        assert_eq!(decoder.finish(), None);

        let error = decoder.push(0).unwrap_err();
        assert_eq!(error.code(), FrontendErrorCode::Contract);
    }

    #[test]
    fn common_prefix_is_a_character_boundary() {
        assert_eq!(common_prefix_bytes("abc", "abc"), 3);
        assert_eq!(common_prefix_bytes("abc", "abd"), 2);
        assert_eq!(common_prefix_bytes("abc北", "abc南"), 3);
        assert_eq!(common_prefix_bytes("🚀a", "🚀b"), 4);
    }

    #[test]
    fn generation_defaults_are_exact() {
        let exact = json!({
            "do_sample": true,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20
        });
        let defaults = parse_generation_defaults::<Qwen38_27B>(&exact).unwrap();

        assert_eq!(defaults.temperature, 1.0);
        assert_eq!(defaults.top_p, 0.95);
        assert_eq!(defaults.top_k, 20);
        assert_eq!(
            parse_generation_defaults::<Qwen36Moe35B>(&exact).unwrap(),
            defaults
        );
        assert_eq!(
            parse_generation_defaults::<Qwen38FlashNext>(&exact).unwrap(),
            defaults
        );

        for changed in [
            json!({"do_sample": false, "temperature": 1.0, "top_p": 0.95, "top_k": 20}),
            json!({"do_sample": true, "temperature": 0.8, "top_p": 0.95, "top_k": 20}),
            json!({"do_sample": true, "temperature": 1.0, "top_p": 0.9, "top_k": 20}),
            json!({"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 40}),
        ] {
            assert!(parse_generation_defaults::<Qwen38_27B>(&changed).is_err());
            assert!(parse_generation_defaults::<Qwen38FlashNext>(&changed).is_err());
        }

        let defaults =
            parse_generation_defaults::<Qwen35_9B>(&json!({"eos_token_id": 248044})).unwrap();
        assert_eq!(defaults.temperature, 0.0);
        assert_eq!(defaults.top_p, 1.0);
        assert_eq!(defaults.top_k, 1);
        for changed in [
            json!({"do_sample": true}),
            json!({"temperature": 1.0}),
            json!({"top_p": 0.95}),
            json!({"top_k": 20}),
        ] {
            assert!(parse_generation_defaults::<Qwen35_9B>(&changed).is_err());
        }
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.6 snapshot"]
    fn qwen36_snapshot_matches_transformers_prompt_fixtures() {
        let root = std::env::var_os("TUISKO_QWEN36_SNAPSHOT")
            .expect("set TUISKO_QWEN36_SNAPSHOT to the admitted revision");
        let snapshot =
            CheckpointSnapshot::<Qwen36Moe35B>::open(std::path::Path::new(&root)).unwrap();
        let frontend = TextFrontend::open_qwen36(&snapshot).unwrap();
        let messages = [ChatMessage::new("user", "Hello")];

        assert_eq!(
            frontend
                .encode_chat(&messages, &ChatTemplateOptions::default())
                .unwrap(),
            QWEN36_HELLO_THINKING
        );
        assert_eq!(
            frontend
                .encode_chat(
                    &messages,
                    &ChatTemplateOptions {
                        enable_thinking: Some(false),
                        ..ChatTemplateOptions::default()
                    },
                )
                .unwrap(),
            QWEN36_HELLO_NO_THINKING
        );
        assert_eq!(frontend.stop_ids(), [248_046, 248_044]);
        assert_eq!(
            frontend.generation_defaults(),
            super::GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            }
        );
    }

    #[test]
    fn cache_selects_the_latest_shared_message_boundary() {
        let first = "prefix<|im_start|>one<|im_start|>old";
        let second_split = first.rfind(PROMPT_BLOCK_START).unwrap();
        let cache = VecDeque::from([PromptPrefixEntry {
            rendered: first.into(),
            token_ids: vec![10, 11],
            token_ends: vec![6, second_split],
        }]);

        let (cached, miss_reason) =
            best_cached_prefix(&cache, "prefix<|im_start|>one<|im_start|>new");
        let CachedPrefix {
            split,
            token_ids,
            token_ends,
        } = cached.unwrap();
        assert_eq!(split, second_split);
        assert_eq!(token_ids, [10, 11]);
        assert_eq!(token_ends, [6, second_split]);
        assert_eq!(miss_reason, "");
    }

    #[test]
    fn cache_miss_reasons_distinguish_lookup_failures() {
        let (cached, reason) = best_cached_prefix(&VecDeque::new(), "new");
        assert!(cached.is_none());
        assert_eq!(reason, "cache-empty");

        let cache = VecDeque::from([PromptPrefixEntry {
            rendered: "unrelated".into(),
            token_ids: vec![10],
            token_ends: vec![9],
        }]);
        let (cached, reason) = best_cached_prefix(&cache, "different");
        assert!(cached.is_none());
        assert_eq!(reason, "no-shared-block-start");

        let rendered = "prefix<|im_start|>old";
        let cache = VecDeque::from([PromptPrefixEntry {
            rendered: rendered.into(),
            token_ids: vec![10],
            token_ends: vec![rendered.len()],
        }]);
        let (cached, reason) = best_cached_prefix(&cache, "prefix<|im_start|>new");
        assert!(cached.is_none());
        assert_eq!(reason, "block-start-not-token-end");
    }

    #[test]
    fn compatibility_constructor_signatures_still_resolve() {
        let _: fn(&CheckpointSnapshot<Qwen38_27B>) -> FrontendResult<TextFrontend> =
            TextFrontend::open;
        let _: fn(
            &CheckpointSnapshot<Qwen38_27B>,
            TextFrontendOptions,
        ) -> FrontendResult<TextFrontend> = TextFrontend::open_with_options;
        let _: fn(&CheckpointSnapshot<Qwen35_9B>) -> FrontendResult<TextFrontend> =
            TextFrontend::open_qwen35;
        let _: fn(
            &CheckpointSnapshot<Qwen35_9B>,
            TextFrontendOptions,
        ) -> FrontendResult<TextFrontend> = TextFrontend::open_qwen35_with_options;
        let _: fn(&CheckpointSnapshot<Qwen36Moe35B>) -> FrontendResult<TextFrontend> =
            TextFrontend::open_qwen36;
        let _: fn(
            &CheckpointSnapshot<Qwen36Moe35B>,
            TextFrontendOptions,
        ) -> FrontendResult<TextFrontend> = TextFrontend::open_qwen36_with_options;
    }

    fn pinned_snapshot<A: TokenizedSchema>(variable: &str) -> CheckpointSnapshot<A> {
        let root = std::env::var_os(variable)
            .unwrap_or_else(|| panic!("{variable} is required for the source-backed gate"));

        CheckpointSnapshot::<A>::open(std::path::Path::new(&root)).unwrap()
    }

    /// Admits pinned frontend files through `open::<A>` and its compatibility alias.
    fn assert_pinned_admission<A: TokenizedSchema>(
        snapshot: &CheckpointSnapshot<A>,
        aliased: &TextFrontend,
    ) {
        let frontend = TextFrontend::open(snapshot).unwrap();
        let messages = [ChatMessage::new("user", "Hello")];
        let options = ChatTemplateOptions::default();

        assert_eq!(frontend.stop_ids(), A::EOS_IDS);
        assert_eq!(frontend.generation_defaults(), A::DEFAULT_GENERATION);
        assert_eq!(aliased.stop_ids(), frontend.stop_ids());
        assert_eq!(
            aliased.generation_defaults(),
            frontend.generation_defaults()
        );
        assert_eq!(
            aliased.render_chat(&messages, true, &options).unwrap(),
            frontend.render_chat(&messages, true, &options).unwrap()
        );
        assert_eq!(
            aliased.encode_chat(&messages, &options).unwrap(),
            frontend.encode_chat(&messages, &options).unwrap()
        );
    }

    #[test]
    #[ignore = "requires TUISKO_CHECKPOINT with the pinned complete Qwen3.8 checkpoint"]
    fn unified_open_admits_the_pinned_qwen38_snapshot() {
        let snapshot = pinned_snapshot::<Qwen38_27B>("TUISKO_CHECKPOINT");
        let aliased =
            TextFrontend::open_with_options(&snapshot, TextFrontendOptions::default()).unwrap();

        assert_pinned_admission(&snapshot, &aliased);
    }

    #[test]
    #[ignore = "requires TUISKO_CHECKPOINT with the pinned complete Qwen3.8 checkpoint"]
    fn qwen38_snapshot_literalizes_assistant_reasoning_controls_with_prefix_reuse() {
        let snapshot = pinned_snapshot::<Qwen38_27B>("TUISKO_CHECKPOINT");
        let frontend = TextFrontend::open(&snapshot).unwrap();
        let options = ChatTemplateOptions::default();
        let opening = vec![ChatMessage::new("user", "First question")];
        let follow_up = vec![
            ChatMessage::new("user", "First question"),
            ChatMessage {
                reasoning_content: Some("quoted <|im_start|> literally".into()),
                ..ChatMessage::new("assistant", "First answer")
            },
            ChatMessage::new("user", "Second question"),
        ];

        let opening = frontend
            .encode_chat_with_report(&opening, &options)
            .unwrap();
        let (extended, metrics) = frontend
            .encode_chat_with_metrics(&follow_up, &options)
            .unwrap();
        assert_eq!(extended.reused_tokens, opening.message_boundary_tokens);
        assert!(metrics.miss_reason.is_empty());
        assert!(
            frontend
                .decode(&extended.token_ids, true)
                .unwrap()
                .contains("quoted <|im_start|> literally")
        );

        let repeated = frontend
            .encode_chat_with_report(&follow_up, &options)
            .unwrap();
        assert_eq!(repeated.reused_tokens, repeated.token_ids.len());
        assert_eq!(repeated.token_ids, extended.token_ids);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN35_SNAPSHOT with the pinned complete Qwen3.5 checkpoint"]
    fn unified_open_admits_the_pinned_qwen35_snapshot() {
        let snapshot = pinned_snapshot::<Qwen35_9B>("TUISKO_QWEN35_SNAPSHOT");
        let aliased = TextFrontend::open_qwen35(&snapshot).unwrap();

        assert_pinned_admission(&snapshot, &aliased);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN36_SNAPSHOT with the pinned complete Qwen3.6 checkpoint"]
    fn unified_open_admits_the_pinned_qwen36_snapshot() {
        let snapshot = pinned_snapshot::<Qwen36Moe35B>("TUISKO_QWEN36_SNAPSHOT");
        let aliased = TextFrontend::open_qwen36(&snapshot).unwrap();

        assert_pinned_admission(&snapshot, &aliased);
    }

    /// Vision controls that the text-only path must not render.
    const QWEN38_FLASH_NEXT_VISION_TOKENS: [(&str, u32); 5] = [
        ("<|vision_start|>", 248_053),
        ("<|vision_end|>", 248_054),
        ("<|vision_pad|>", 248_055),
        ("<|image_pad|>", 248_056),
        ("<|video_pad|>", 248_057),
    ];

    /// Number of tokens the generated-assistant header contributes when thinking stays open.
    const QWEN38_FLASH_NEXT_THINKING_HEADER_TOKENS: usize = 5;

    /// One byte-exact text-only prompt fixture.
    struct PromptFixture {
        name: &'static str,
        messages: Vec<ChatMessage>,
        options: ChatTemplateOptions,
        rendered: &'static str,
        token_ids: &'static [u32],
    }

    fn reasoning_effort(effort: &str) -> ChatTemplateOptions {
        ChatTemplateOptions {
            reasoning_effort: Some(effort.into()),
            ..ChatTemplateOptions::default()
        }
    }

    fn multi_turn_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", "You are terse."),
            ChatMessage::new("user", "First question"),
            ChatMessage {
                reasoning_content: Some("earlier thought".into()),
                ..ChatMessage::new("assistant", "First answer")
            },
            ChatMessage::new("user", "Second question"),
        ]
    }

    fn tool_flow_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("user", "Weather in Oulu?"),
            ChatMessage {
                tool_calls: vec![ChatToolCall {
                    id: Some("call_1".into()),
                    kind: "function".into(),
                    function: ChatFunctionCall {
                        name: "get_weather".into(),
                        arguments: json!({"city": "Oulu"}),
                    },
                }],
                ..ChatMessage::new("assistant", "")
            },
            ChatMessage {
                tool_call_id: Some("call_1".into()),
                ..ChatMessage::new("tool", "{\"c\": 3}")
            },
        ]
    }

    /// Golden renders and token IDs from the pinned files and Transformers 5.2.0.
    fn qwen38_flash_next_prompt_fixtures() -> Vec<PromptFixture> {
        vec![
            PromptFixture {
                name: "hello-xhigh",
                messages: vec![ChatMessage::new("user", "Hello")],
                options: ChatTemplateOptions::default(),
                rendered: "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 8678, 198, 24_342, 286, 4879, 369, 716, 310, 830, 11_553, 13, 5044,
                    1683, 15_060, 1472, 279, 3274, 11, 9307, 1328, 30_800, 11, 2814, 47_675,
                    25_605, 11, 321, 60_445, 55_404, 11, 27_224, 11, 321, 30_246, 303, 279, 1534,
                    4087, 13, 248_046, 198, 248_045, 846, 198, 9419, 248_046, 198, 248_045, 74_455,
                    198, 248_068, 198,
                ],
            },
            PromptFixture {
                name: "hello-no-thinking",
                messages: vec![ChatMessage::new("user", "Hello")],
                options: ChatTemplateOptions {
                    enable_thinking: Some(false),
                    ..ChatTemplateOptions::default()
                },
                rendered: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                token_ids: &[
                    248_045, 846, 198, 9419, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
                    248_069, 271,
                ],
            },
            PromptFixture {
                name: "hello-low",
                messages: vec![ChatMessage::new("user", "Hello")],
                options: reasoning_effort("low"),
                rendered: "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 8678, 198, 24_342, 286, 4879, 369, 716, 310, 3238, 13, 13_262, 678,
                    7047, 9522, 321, 10_419, 11, 6992, 5774, 310, 279, 16_198, 1973, 24_366,
                    24_150, 362, 13, 248_046, 198, 248_045, 846, 198, 9419, 248_046, 198, 248_045,
                    74_455, 198, 248_068, 198,
                ],
            },
            PromptFixture {
                name: "hello-medium",
                messages: vec![ChatMessage::new("user", "Hello")],
                options: reasoning_effort("medium"),
                rendered: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 846, 198, 9419, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
                ],
            },
            PromptFixture {
                name: "multi-turn",
                messages: multi_turn_messages(),
                options: reasoning_effort("medium"),
                rendered: "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\n<think>\nearlier thought\n</think>\n\nFirst answer<|im_end|>\n<|im_start|>user\nSecond question<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 8678, 198, 2523, 513, 48_834, 13, 248_046, 198, 248_045, 846, 198,
                    5170, 3296, 248_046, 198, 248_045, 74_455, 198, 248_068, 198, 664, 5446, 3272,
                    198, 248_069, 271, 5170, 4087, 248_046, 198, 248_045, 846, 198, 15_207, 3296,
                    248_046, 198, 248_045, 74_455, 198, 248_068, 198,
                ],
            },
            PromptFixture {
                name: "multi-turn-dropped-thinking",
                messages: multi_turn_messages(),
                options: ChatTemplateOptions {
                    preserve_thinking: Some(false),
                    ..reasoning_effort("medium")
                },
                rendered: "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\nFirst answer<|im_end|>\n<|im_start|>user\nSecond question<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 8678, 198, 2523, 513, 48_834, 13, 248_046, 198, 248_045, 846, 198,
                    5170, 3296, 248_046, 198, 248_045, 74_455, 198, 5170, 4087, 248_046, 198,
                    248_045, 846, 198, 15_207, 3296, 248_046, 198, 248_045, 74_455, 198, 248_068,
                    198,
                ],
            },
            PromptFixture {
                name: "tool-flow",
                messages: tool_flow_messages(),
                options: reasoning_effort("medium"),
                rendered: "<|im_start|>user\nWeather in Oulu?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOulu\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"c\": 3}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 846, 198, 28_034, 303, 496, 23_639, 30, 248_046, 198, 248_045, 74_455,
                    198, 248_068, 271, 248_069, 271, 248_058, 198, 27, 1628, 27_362, 67_017, 29,
                    198, 27, 15_704, 28, 8656, 29, 198, 46, 23_639, 198, 510, 15_704, 29, 198, 510,
                    1628, 29, 198, 248_059, 248_046, 198, 248_045, 846, 198, 248_066, 198, 4754,
                    66, 763, 220, 18, 92, 198, 248_067, 248_046, 198, 248_045, 74_455, 198,
                    248_068, 198,
                ],
            },
            PromptFixture {
                name: "unicode",
                messages: vec![ChatMessage::new("user", "Hei 🚀 北")],
                options: reasoning_effort("medium"),
                rendered: "<|im_start|>user\nHei 🚀 北<|im_end|>\n<|im_start|>assistant\n<think>\n",
                token_ids: &[
                    248_045, 846, 198, 1465, 72, 10_838, 248, 222, 220, 96_443, 248_046, 198,
                    248_045, 74_455, 198, 248_068, 198,
                ],
            },
        ]
    }

    #[test]
    fn qwen38_flash_next_pins_its_frontend_contract() {
        assert_eq!(Qwen38FlashNext::TOKENIZER_ENTRIES, 248_077);
        assert_eq!(Qwen38FlashNext::EOS_IDS, [248_046, 248_044]);
        assert_eq!(
            Qwen38FlashNext::GENERATION_ADMISSION,
            GenerationAdmission::Sampled
        );
        assert_eq!(
            Qwen38FlashNext::DEFAULT_GENERATION,
            GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            }
        );
        assert_eq!(Qwen38FlashNext::SPECIAL_TOKENS, SPECIAL_TOKEN_LITERALS);

        // Equal widths are checkpoint facts; neither target derives its value from the other.
        assert_eq!(Qwen38FlashNext::VOCAB, Qwen38_27B::VOCAB);
        assert_eq!(
            Qwen38FlashNext::TOKENIZER_ENTRIES,
            Qwen38_27B::TOKENIZER_ENTRIES
        );
        const {
            assert!(Qwen38FlashNext::TOKENIZER_ENTRIES < Qwen38FlashNext::VOCAB);
        }

        assert_eq!(Qwen38FlashNext::EOS_TOKEN_ID, END_OF_TEXT_ID);
        assert!(Qwen38FlashNext::EOS_IDS.contains(&Qwen38FlashNext::EOS_TOKEN_ID));
    }

    fn tokenizer_with_inventory(entries: usize, control_shift: u32) -> Tokenizer {
        let entries = u32::try_from(entries).expect("test inventory fits u32");
        let vocab = (0..entries)
            .map(|id| {
                let control = CONTROL_TOKEN_IDS.iter().find_map(|&(literal, pinned)| {
                    (pinned + control_shift == id).then_some(literal)
                });
                (control.map_or_else(|| id.to_string(), str::to_owned), id)
            })
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("0".into())
            .build()
            .unwrap();

        Tokenizer::new(model)
    }

    #[test]
    fn qwen38_flash_next_rejects_a_different_tokenizer_inventory() {
        let byte_table = byte_level_table();

        validate_tokenizer::<Qwen38FlashNext>(&tokenizer_with_inventory(248_077, 0), &byte_table)
            .unwrap();
        for (entries, shift) in [(248_070, 0), (248_320, 0), (248_077, 1)] {
            let error = validate_tokenizer::<Qwen38FlashNext>(
                &tokenizer_with_inventory(entries, shift),
                &byte_table,
            )
            .unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
    }

    fn qwen38_flash_next_snapshot() -> CheckpointSnapshot<Qwen38FlashNext> {
        pinned_snapshot::<Qwen38FlashNext>("TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT")
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_admits_the_frontend_contract() {
        let snapshot = qwen38_flash_next_snapshot();
        let aliased =
            TextFrontend::open_with_options(&snapshot, TextFrontendOptions::default()).unwrap();

        assert_pinned_admission(&snapshot, &aliased);
        assert_eq!(aliased.stop_ids(), [248_046, 248_044]);
        assert_eq!(
            aliased.generation_defaults(),
            GenerationDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            }
        );
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_matches_text_fixtures() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();

        for fixture in qwen38_flash_next_prompt_fixtures() {
            let rendered = frontend
                .render_chat(&fixture.messages, true, &fixture.options)
                .unwrap();
            assert_eq!(rendered, fixture.rendered, "{}", fixture.name);

            let encoding = frontend
                .encode_chat_with_report(&fixture.messages, &fixture.options)
                .unwrap();
            assert_eq!(encoding.token_ids, fixture.token_ids, "{}", fixture.name);
            assert_eq!(encoding.rendered_bytes, fixture.rendered.len());

            // Round-trip: the byte-level tokenizer must reproduce the prompt exactly, and the
            // message boundary must land on the generated-assistant header.
            assert_eq!(
                frontend.decode(&encoding.token_ids, false).unwrap(),
                fixture.rendered,
                "{}",
                fixture.name
            );
            let boundary = fixture
                .rendered
                .rfind(super::GENERATION_BLOCK_START)
                .unwrap();
            assert_eq!(
                frontend
                    .decode(
                        &encoding.token_ids[..encoding.message_boundary_tokens],
                        false
                    )
                    .unwrap(),
                fixture.rendered[..boundary],
                "{}",
                fixture.name
            );
        }
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_keeps_vision_controls_out_of_text() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();

        for (literal, id) in QWEN38_FLASH_NEXT_VISION_TOKENS {
            // The multimodal family is present in the pinned tokenizer: absence from prompts is
            // the template's text-only path, not a missing entry.
            assert_eq!(frontend.decode(&[id], false).unwrap(), literal);
            // Nothing generated on a vision branch can leak into streamed text either.
            assert_eq!(frontend.streaming_decoder().push(id).unwrap(), None);
        }

        for fixture in qwen38_flash_next_prompt_fixtures() {
            let encoding = frontend
                .encode_chat_with_report(&fixture.messages, &fixture.options)
                .unwrap();
            for (literal, id) in QWEN38_FLASH_NEXT_VISION_TOKENS {
                assert!(
                    !fixture.rendered.contains(literal),
                    "{} rendered {literal}",
                    fixture.name
                );
                assert!(
                    !encoding.token_ids.contains(&id),
                    "{} encoded {literal}",
                    fixture.name
                );
            }
        }
    }

    /// Exact tool-call syntax emitted by the pinned template.
    const QWEN38_FLASH_NEXT_TOOL_CALL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n";

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_renders_tool_definitions() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();
        let options = ChatTemplateOptions {
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            })],
            ..reasoning_effort("medium")
        };
        let messages = [ChatMessage::new("user", "Weather in Oulu?")];
        let rendered = frontend.render_chat(&messages, true, &options).unwrap();

        assert!(
            rendered.starts_with(
                "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            ),
            "{rendered}"
        );
        // Pin minijinja's compact and escaped `tojson` output separately from Transformers.
        assert!(
            rendered.contains(
                "{\"function\":{\"description\":\"Get weather\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}},\"type\":\"function\"}\n</tools>"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains(QWEN38_FLASH_NEXT_TOOL_CALL_INSTRUCTIONS),
            "{rendered}"
        );
        assert!(
            rendered.ends_with(
                "</IMPORTANT><|im_end|>\n<|im_start|>user\nWeather in Oulu?<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ),
            "{rendered}"
        );

        // A tool list never opens a vision branch either.
        for (literal, _) in QWEN38_FLASH_NEXT_VISION_TOKENS {
            assert!(!rendered.contains(literal), "{rendered}");
        }
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_prepends_no_bos() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();
        let generation = read_json(&snapshot.root().join(GENERATION_CONFIG_FILE)).unwrap();

        // `bos_token_id` restates the pad and end-of-text identity; `tokenizer_config.json` sets
        // `add_bos_token: false` with a null `bos_token`, so no admitted path prepends it.
        assert_eq!(generation["bos_token_id"], json!(END_OF_TEXT_ID));
        assert_eq!(generation["pad_token_id"], json!(END_OF_TEXT_ID));

        for fixture in qwen38_flash_next_prompt_fixtures() {
            let encoding = frontend
                .encode_chat_with_report(&fixture.messages, &fixture.options)
                .unwrap();
            assert_eq!(encoding.token_ids[0], IM_START_ID, "{}", fixture.name);
            assert!(
                !encoding.token_ids.contains(&END_OF_TEXT_ID),
                "{}",
                fixture.name
            );
        }
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_streams_utf8() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();
        let generated = "Sää Oulussa: 3 °C 🚀 北風";
        let token_ids = frontend.encode(generated).unwrap();
        assert_eq!(frontend.decode(&token_ids, false).unwrap(), generated);

        let mut decoder = frontend.streaming_decoder();
        let mut streamed = String::new();
        for (index, &token_id) in token_ids.iter().enumerate() {
            // A stop token mid-stream must be skipped without disturbing a pending sequence.
            if index == token_ids.len() / 2 {
                assert_eq!(decoder.push(248_046).unwrap(), None);
            }
            if let Some(delta) = decoder.push(token_id).unwrap() {
                streamed.push_str(&delta);
            }
        }
        if let Some(delta) = decoder.finish() {
            streamed.push_str(&delta);
        }

        assert_eq!(streamed, generated);
        assert_eq!(decoder.text(), generated);
    }

    #[test]
    #[ignore = "requires TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT with the pinned complete checkpoint"]
    fn qwen38_flash_next_snapshot_reuses_prompt_prefix() {
        let snapshot = qwen38_flash_next_snapshot();
        let frontend = TextFrontend::open(&snapshot).unwrap();
        let options = reasoning_effort("medium");
        let first = vec![ChatMessage::new("user", "First question")];
        let follow_up = vec![
            ChatMessage::new("user", "First question"),
            ChatMessage {
                reasoning_content: Some("earlier thought".into()),
                ..ChatMessage::new("assistant", "First answer")
            },
            ChatMessage::new("user", "Second question"),
        ];

        let (opening, metrics) = frontend.encode_chat_with_metrics(&first, &options).unwrap();
        assert_eq!(opening.reused_tokens, 0);
        assert_eq!(metrics.miss_reason, "cache-empty");
        assert_eq!(
            opening.message_boundary_tokens + QWEN38_FLASH_NEXT_THINKING_HEADER_TOKENS,
            opening.token_ids.len()
        );

        let (extended, metrics) = frontend
            .encode_chat_with_metrics(&follow_up, &options)
            .unwrap();
        assert_eq!(metrics.miss_reason, "");
        // The shared block ends at the `<|im_start|>` that opened the first prompt's generated
        // turn, so every token through the first user message is reused verbatim.
        assert_eq!(extended.reused_tokens, opening.message_boundary_tokens);
        assert!(extended.fresh_bytes < extended.rendered_bytes);
        assert_eq!(
            extended.token_ids[..extended.reused_tokens],
            opening.token_ids[..extended.reused_tokens]
        );

        // The identical prompt is served entirely from the cache.
        let (repeated, metrics) = frontend
            .encode_chat_with_metrics(&follow_up, &options)
            .unwrap();
        assert_eq!(metrics.miss_reason, "");
        assert_eq!(repeated.token_ids, extended.token_ids);
        assert_eq!(repeated.reused_tokens, repeated.token_ids.len());
        assert_eq!(repeated.fresh_bytes, 0);
    }
}

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
use tuisko_model::{CheckpointSnapshot, Qwen35_9B, Qwen38_27B};

pub use error::{FrontendError, FrontendErrorCode, FrontendResult};

const TOKENIZER_FILE: &str = "tokenizer.json";
const TEMPLATE_FILE: &str = "chat_template.jinja";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";
const QWEN38_TOKENIZER_ENTRIES: usize = 248_077;
const QWEN35_TOKENIZER_ENTRIES: usize = 248_070;
const IM_START_ID: u32 = 248_045;
const IM_END_ID: u32 = 248_046;
const END_OF_TEXT_ID: u32 = 248_044;
const QWEN38_EOS_IDS: [u32; 2] = [IM_END_ID, END_OF_TEXT_ID];
const QWEN35_EOS_IDS: [u32; 1] = [END_OF_TEXT_ID];
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_TOP_K: usize = 20;
const PROMPT_BLOCK_START: &str = SPECIAL_TOKEN_LITERALS[0];

/// Literal strings the pinned tokenizer always extracts as control tokens from raw text.
pub const SPECIAL_TOKEN_LITERALS: [&str; 3] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"];

#[derive(Clone, Copy)]
enum FrontendContract {
    Qwen38,
    Qwen35,
}

impl FrontendContract {
    const fn tokenizer_entries(self) -> usize {
        match self {
            Self::Qwen38 => QWEN38_TOKENIZER_ENTRIES,
            Self::Qwen35 => QWEN35_TOKENIZER_ENTRIES,
        }
    }

    const fn eos_ids(self) -> &'static [u32] {
        match self {
            Self::Qwen38 => &QWEN38_EOS_IDS,
            Self::Qwen35 => &QWEN35_EOS_IDS,
        }
    }
}

/// One text message supplied to the checkpoint chat template.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// Template role such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Text content, accepting OpenAI text parts at the transport boundary.
    #[serde(default, deserialize_with = "deserialize_chat_content")]
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
#[serde(untagged)]
enum WireChatContent {
    Text(String),
    Parts(Vec<WireChatContentPart>),
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

fn deserialize_chat_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<WireChatContent>::deserialize(deserializer)? {
        None => Ok(String::new()),
        Some(WireChatContent::Text(text)) => Ok(text),
        Some(WireChatContent::Parts(parts)) => {
            let mut text = String::new();
            for part in parts {
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
            Ok(text)
        }
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
    /// Token IDs reused from an earlier rendering.
    pub reused_tokens: usize,
    /// Bytes in the complete rendered prompt.
    pub rendered_bytes: usize,
    /// Bytes passed through BPE during this call.
    pub fresh_bytes: usize,
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
    byte_table: HashMap<char, u8>,
    special_decode_ids: HashSet<u32>,
}

/// Incremental decoder for one generated text sequence.
pub struct StreamingDecoder {
    decode: Arc<DecodeState>,
    text: String,
    pending: Vec<u8>,
    finished: bool,
}

impl TextFrontend {
    /// Loads and validates frontend files from an admitted snapshot.
    pub fn open(snapshot: &CheckpointSnapshot<Qwen38_27B>) -> FrontendResult<Self> {
        Self::open_with_options(snapshot, TextFrontendOptions::default())
    }

    /// Loads the frontend with explicit startup options.
    pub fn open_with_options(
        snapshot: &CheckpointSnapshot<Qwen38_27B>,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        Self::open_root(snapshot.root(), options, FrontendContract::Qwen38)
    }

    /// Loads and validates the pinned Qwen3.5 tokenizer, template, and generation metadata.
    pub fn open_qwen35(snapshot: &CheckpointSnapshot<Qwen35_9B>) -> FrontendResult<Self> {
        Self::open_qwen35_with_options(snapshot, TextFrontendOptions::default())
    }

    /// Loads the Qwen3.5 frontend with explicit startup options.
    pub fn open_qwen35_with_options(
        snapshot: &CheckpointSnapshot<Qwen35_9B>,
        options: TextFrontendOptions,
    ) -> FrontendResult<Self> {
        Self::open_root(snapshot.root(), options, FrontendContract::Qwen35)
    }

    fn open_root(
        root: &Path,
        options: TextFrontendOptions,
        contract: FrontendContract,
    ) -> FrontendResult<Self> {
        let tokenizer_path = root.join(TOKENIZER_FILE);
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|source| {
            FrontendError::Tokenizer(format!(
                "could not load {}: {source}",
                tokenizer_path.display()
            ))
        })?;
        tokenizer
            .with_truncation(None)
            .map_err(|source| FrontendError::Tokenizer(source.to_string()))?;
        validate_tokenizer(&tokenizer, contract)?;

        let template_path = root.join(TEMPLATE_FILE);
        let template = read_string(&template_path)?;

        let generation_path = root.join(GENERATION_CONFIG_FILE);
        let generation = read_json(&generation_path)?;
        let stop_ids = parse_stop_ids(&generation, contract)?;
        let generation_defaults = parse_generation_defaults(&generation, contract)?;
        let special_decode_ids = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter_map(|(&id, token)| token.special.then_some(id))
            .collect();

        Ok(Self {
            decode: Arc::new(DecodeState {
                tokenizer,
                byte_table: byte_level_table(),
                special_decode_ids,
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

    /// Encodes text without adding tokenizer-defined special tokens.
    pub fn encode(&self, text: &str) -> FrontendResult<Vec<u32>> {
        self.decode
            .tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|source| FrontendError::Tokenizer(format!("could not encode text: {source}")))
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
        let rendered = self.render_chat(messages, true, options)?;
        self.encode_rendered_with_prefix(&rendered)
    }

    /// Decodes token IDs using the admitted tokenizer.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> FrontendResult<String> {
        self.decode
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|source| {
                FrontendError::Tokenizer(format!("could not decode token IDs: {source}"))
            })
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

    fn encode_rendered_with_prefix(&self, rendered: &str) -> FrontendResult<PromptEncoding> {
        // Added tokens split before BPE, so restarting at a shared `<|im_start|>`
        // preserves the full-encode token sequence. Other splits fall back.
        let cached = {
            let cache = self
                .prompt_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            best_cached_prefix(&cache, rendered)
        };

        if let Some(cached) = cached {
            let split = cached.split;
            if split == rendered.len() {
                return Ok(PromptEncoding {
                    reused_tokens: cached.token_ids.len(),
                    token_ids: cached.token_ids,
                    rendered_bytes: rendered.len(),
                    fresh_bytes: 0,
                });
            }

            let tail = &rendered[split..];
            let encoding = self
                .decode
                .tokenizer
                .encode(tail, false)
                .map_err(|source| {
                    FrontendError::Tokenizer(format!("could not encode prompt tail: {source}"))
                })?;
            let reused_tokens = cached.token_ids.len();
            let mut token_ids = cached.token_ids;
            token_ids.extend_from_slice(encoding.get_ids());
            let mut token_ends = cached.token_ends;
            token_ends.extend(encoding.get_offsets().iter().map(|&(_, end)| split + end));
            self.push_prompt_entry(rendered, token_ids.clone(), token_ends);

            return Ok(PromptEncoding {
                token_ids,
                reused_tokens,
                rendered_bytes: rendered.len(),
                fresh_bytes: tail.len(),
            });
        }

        let encoding = self
            .decode
            .tokenizer
            .encode(rendered, false)
            .map_err(|source| {
                FrontendError::Tokenizer(format!("could not encode prompt: {source}"))
            })?;
        let token_ids = encoding.get_ids().to_vec();
        let token_ends = encoding.get_offsets().iter().map(|&(_, end)| end).collect();
        self.push_prompt_entry(rendered, token_ids.clone(), token_ends);

        Ok(PromptEncoding {
            token_ids,
            reused_tokens: 0,
            rendered_bytes: rendered.len(),
            fresh_bytes: rendered.len(),
        })
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
        if let Ok(text) = std::str::from_utf8(&candidate) {
            output.push_str(text);
            pending.clear();
        } else if candidate.len() == usize::from(utf8_sequence_length(candidate[0]).unwrap()) {
            output.push_str(&"\u{fffd}".repeat(candidate.len()));
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

fn best_cached_prefix(cache: &VecDeque<PromptPrefixEntry>, rendered: &str) -> Option<CachedPrefix> {
    let mut best = None;
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
    best
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

fn validate_tokenizer(tokenizer: &Tokenizer, contract: FrontendContract) -> FrontendResult<()> {
    let entries = tokenizer.get_vocab_size(true);
    let expected_entries = contract.tokenizer_entries();
    if entries != expected_entries {
        return Err(FrontendError::Contract(format!(
            "tokenizer has {entries} entries, expected {expected_entries}"
        )));
    }

    let [im_start, im_end, end_of_text] = SPECIAL_TOKEN_LITERALS;
    for (token, expected) in [
        (im_start, IM_START_ID),
        (im_end, IM_END_ID),
        (end_of_text, END_OF_TEXT_ID),
    ] {
        let actual = tokenizer.token_to_id(token);
        if actual != Some(expected) {
            return Err(FrontendError::Contract(format!(
                "tokenizer maps `{token}` to {actual:?}, expected {expected}"
            )));
        }
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

fn parse_stop_ids(generation: &Value, contract: FrontendContract) -> FrontendResult<Vec<u32>> {
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
    let expected = contract.eos_ids();
    if stop_ids != expected {
        return Err(FrontendError::Contract(format!(
            "generation stop IDs {stop_ids:?} do not match {expected:?}"
        )));
    }

    Ok(stop_ids)
}

fn parse_generation_defaults(
    generation: &Value,
    contract: FrontendContract,
) -> FrontendResult<GenerationDefaults> {
    match contract {
        FrontendContract::Qwen38 => parse_qwen38_generation_defaults(generation),
        FrontendContract::Qwen35 => parse_qwen35_generation_defaults(generation),
    }
}

fn parse_qwen38_generation_defaults(generation: &Value) -> FrontendResult<GenerationDefaults> {
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
    let expected = GenerationDefaults {
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        top_k: DEFAULT_TOP_K,
    };
    if defaults != expected {
        return Err(FrontendError::Contract(format!(
            "generation defaults {defaults:?} do not match {expected:?}"
        )));
    }

    Ok(defaults)
}

fn parse_qwen35_generation_defaults(generation: &Value) -> FrontendResult<GenerationDefaults> {
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

    Ok(GenerationDefaults {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CachedPrefix, ChatMessage, ChatTemplateOptions, FrontendContract, FrontendErrorCode,
        PROMPT_BLOCK_START, PromptPrefixEntry, TextFrontend, best_cached_prefix, byte_level_table,
        common_prefix_bytes, finish_pending, parse_generation_defaults, parse_stop_ids,
        push_stream_byte,
    };
    use serde_json::json;
    use std::collections::{HashSet, VecDeque};
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

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
        assert_eq!(
            parse_stop_ids(&qwen38, FrontendContract::Qwen38).unwrap(),
            [248046, 248044]
        );
        assert_eq!(
            parse_stop_ids(&qwen35, FrontendContract::Qwen35).unwrap(),
            [248044]
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
            let error = parse_stop_ids(&generation, FrontendContract::Qwen38).unwrap_err();
            assert_eq!(error.code(), FrontendErrorCode::Contract);
        }
        for generation in [
            json!({}),
            json!({"eos_token_id": [248044, 248046]}),
            json!({"eos_token_id": -1}),
        ] {
            let error = parse_stop_ids(&generation, FrontendContract::Qwen35).unwrap_err();
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
        let frontend = TextFrontend {
            decode: std::sync::Arc::new(super::DecodeState {
                tokenizer: Tokenizer::new(model),
                byte_table,
                special_decode_ids: HashSet::from([3]),
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
        let defaults = parse_generation_defaults(&exact, FrontendContract::Qwen38).unwrap();

        assert_eq!(defaults.temperature, 1.0);
        assert_eq!(defaults.top_p, 0.95);
        assert_eq!(defaults.top_k, 20);

        for changed in [
            json!({"do_sample": false, "temperature": 1.0, "top_p": 0.95, "top_k": 20}),
            json!({"do_sample": true, "temperature": 0.8, "top_p": 0.95, "top_k": 20}),
            json!({"do_sample": true, "temperature": 1.0, "top_p": 0.9, "top_k": 20}),
            json!({"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 40}),
        ] {
            assert!(parse_generation_defaults(&changed, FrontendContract::Qwen38).is_err());
        }

        let defaults =
            parse_generation_defaults(&json!({"eos_token_id": 248044}), FrontendContract::Qwen35)
                .unwrap();
        assert_eq!(defaults.temperature, 0.0);
        assert_eq!(defaults.top_p, 1.0);
        assert_eq!(defaults.top_k, 1);
        for changed in [
            json!({"do_sample": true}),
            json!({"temperature": 1.0}),
            json!({"top_p": 0.95}),
            json!({"top_k": 20}),
        ] {
            assert!(parse_generation_defaults(&changed, FrontendContract::Qwen35).is_err());
        }
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

        let CachedPrefix {
            split,
            token_ids,
            token_ends,
        } = best_cached_prefix(&cache, "prefix<|im_start|>one<|im_start|>new").unwrap();
        assert_eq!(split, second_split);
        assert_eq!(token_ids, [10, 11]);
        assert_eq!(token_ends, [6, second_split]);
    }
}

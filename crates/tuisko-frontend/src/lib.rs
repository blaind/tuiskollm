//! Exact tokenizer and text-only chat-template boundary.

mod error;

use minijinja::{Environment, Error as TemplateError, ErrorKind};
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

pub use error::{FrontendError, FrontendErrorCode, FrontendResult};

const TOKENIZER_FILE: &str = "tokenizer.json";
const TEMPLATE_FILE: &str = "chat_template.jinja";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";
const TOKENIZER_ENTRIES: usize = 248_077;
const IM_START_ID: u32 = 248_045;
const IM_END_ID: u32 = 248_046;
const END_OF_TEXT_ID: u32 = 248_044;
const DEFAULT_EOS_IDS: [u32; 2] = [IM_END_ID, END_OF_TEXT_ID];

/// One text message supplied to the checkpoint chat template.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChatMessage {
    /// Template role such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// Creates a text-only chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Per-request options admitted by the current text template boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    /// Overrides the checkpoint's default thinking mode when present.
    pub enable_thinking: Option<bool>,
}

/// Admitted tokenizer, chat template, and generation stop-token metadata.
pub struct TextFrontend {
    tokenizer: Tokenizer,
    template: String,
    stop_ids: [u32; 2],
    byte_table: HashMap<char, u8>,
    special_decode_ids: HashSet<u32>,
}

/// Incremental decoder for one generated text sequence.
pub struct StreamingDecoder<'a> {
    frontend: &'a TextFrontend,
    text: String,
    pending: Vec<u8>,
    finished: bool,
}

impl TextFrontend {
    /// Loads and validates frontend files from an admitted snapshot.
    pub fn open(snapshot: &CheckpointSnapshot<Qwen38_27B>) -> FrontendResult<Self> {
        Self::open_root(snapshot.root())
    }

    fn open_root(root: &Path) -> FrontendResult<Self> {
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
        validate_tokenizer(&tokenizer)?;

        let template_path = root.join(TEMPLATE_FILE);
        let template = read_string(&template_path)?;

        let generation_path = root.join(GENERATION_CONFIG_FILE);
        let generation = read_json(&generation_path)?;
        let stop_ids = parse_stop_ids(&generation)?;
        let special_decode_ids = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter_map(|(&id, token)| token.special.then_some(id))
            .collect();

        Ok(Self {
            tokenizer,
            template,
            stop_ids,
            byte_table: byte_level_table(),
            special_decode_ids,
        })
    }

    /// Returns the pinned generation stop-token IDs.
    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    /// Renders the checkpoint's text-only chat template.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        options: ChatTemplateOptions,
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
        context.insert("tools".into(), Value::Array(Vec::new()));
        if let Some(enable_thinking) = options.enable_thinking {
            context.insert("enable_thinking".into(), Value::Bool(enable_thinking));
        }

        environment
            .render_str(&self.template, Value::Object(context))
            .map_err(FrontendError::from)
    }

    /// Encodes text without adding tokenizer-defined special tokens.
    pub fn encode(&self, text: &str) -> FrontendResult<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|source| FrontendError::Tokenizer(format!("could not encode text: {source}")))
    }

    /// Renders and encodes one generation prompt.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> FrontendResult<Vec<u32>> {
        let rendered = self.render_chat(messages, true, options)?;
        self.encode(&rendered)
    }

    /// Decodes token IDs using the admitted tokenizer.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> FrontendResult<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|source| {
                FrontendError::Tokenizer(format!("could not decode token IDs: {source}"))
            })
    }

    /// Starts a special-token-skipping streaming decoder.
    pub fn streaming_decoder(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            frontend: self,
            text: String::new(),
            pending: Vec::new(),
            finished: false,
        }
    }
}

impl StreamingDecoder<'_> {
    /// Decodes one generated token and returns any newly complete text.
    pub fn push(&mut self, token_id: u32) -> FrontendResult<Option<String>> {
        if self.finished {
            return Err(FrontendError::Contract(
                "cannot push a token after finishing the stream".into(),
            ));
        }
        if self.frontend.special_decode_ids.contains(&token_id) {
            return Ok(None);
        }

        let content = self
            .frontend
            .tokenizer
            .id_to_token(token_id)
            .ok_or_else(|| {
                FrontendError::Tokenizer(format!("token ID {token_id} has no tokenizer entry"))
            })?;
        let mut delta = String::new();
        for character in content.chars() {
            let byte = self
                .frontend
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

fn validate_tokenizer(tokenizer: &Tokenizer) -> FrontendResult<()> {
    let entries = tokenizer.get_vocab_size(true);
    if entries != TOKENIZER_ENTRIES {
        return Err(FrontendError::Contract(format!(
            "tokenizer has {entries} entries, expected {TOKENIZER_ENTRIES}"
        )));
    }

    for (token, expected) in [
        ("<|im_start|>", IM_START_ID),
        ("<|im_end|>", IM_END_ID),
        ("<|endoftext|>", END_OF_TEXT_ID),
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

fn parse_stop_ids(generation: &Value) -> FrontendResult<[u32; 2]> {
    let values = generation
        .get("eos_token_id")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json `eos_token_id` must be an array".into())
        })?;
    let stop_ids = values
        .iter()
        .map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            FrontendError::Contract("generation_config.json contains a non-u32 stop ID".into())
        })?;
    let stop_ids: [u32; 2] = stop_ids.try_into().map_err(|actual: Vec<u32>| {
        FrontendError::Contract(format!(
            "generation stop IDs {actual:?} do not match {DEFAULT_EOS_IDS:?}"
        ))
    })?;
    if stop_ids != DEFAULT_EOS_IDS {
        return Err(FrontendError::Contract(format!(
            "generation stop IDs {stop_ids:?} do not match {DEFAULT_EOS_IDS:?}"
        )));
    }

    Ok(stop_ids)
}

#[cfg(test)]
mod tests {
    use super::{
        ChatMessage, ChatTemplateOptions, FrontendErrorCode, TextFrontend, byte_level_table,
        finish_pending, parse_stop_ids, push_stream_byte,
    };
    use serde_json::json;
    use std::collections::HashSet;
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
    fn stop_ids_are_exact() {
        let generation = json!({"eos_token_id": [248046, 248044]});
        assert_eq!(parse_stop_ids(&generation).unwrap(), [248046, 248044]);
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
            let error = parse_stop_ids(&generation).unwrap_err();
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
            tokenizer: Tokenizer::new(model),
            template: String::new(),
            stop_ids: [3, 3],
            byte_table,
            special_decode_ids: HashSet::from([3]),
        };

        let mut decoder = frontend.streaming_decoder();
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
}

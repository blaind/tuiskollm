//! Exact tokenizer and text-only chat-template boundary.

mod error;

use minijinja::{Environment, Error as TemplateError, ErrorKind};
use serde::Serialize;
use serde_json::{Map, Value};
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

        Ok(Self {
            tokenizer,
            template,
            stop_ids,
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
    use super::{ChatMessage, ChatTemplateOptions, FrontendErrorCode, parse_stop_ids};
    use serde_json::json;

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
}

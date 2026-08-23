//! Python bindings for the exact tokenizer and text-only chat template.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::path::Path;
use tuisko_frontend::{
    ChatMessage, ChatTemplateOptions, GenerationDefaults as RustGenerationDefaults,
    PromptEncoding as RustPromptEncoding, TextFrontend, TextFrontendOptions,
};
use tuisko_model::{Arch, CheckpointSnapshot, Qwen38_27B};

create_exception!(
    _native,
    TuiskoError,
    PyException,
    "Base class for errors raised by the TuiskoLLM Python package."
);
create_exception!(
    _native,
    CheckpointError,
    TuiskoError,
    "The path is not the admitted checkpoint snapshot."
);
create_exception!(
    _native,
    FrontendError,
    TuiskoError,
    "Tokenizer, template, or frontend metadata failure."
);

#[pyclass(module = "tuisko.llm._native", frozen, get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
struct PromptEncoding {
    token_ids: Vec<u32>,
    reused_tokens: usize,
    rendered_bytes: usize,
    fresh_bytes: usize,
}

impl From<RustPromptEncoding> for PromptEncoding {
    fn from(encoding: RustPromptEncoding) -> Self {
        Self {
            token_ids: encoding.token_ids,
            reused_tokens: encoding.reused_tokens,
            rendered_bytes: encoding.rendered_bytes,
            fresh_bytes: encoding.fresh_bytes,
        }
    }
}

#[pymethods]
impl PromptEncoding {
    fn __repr__(&self) -> String {
        format!(
            "PromptEncoding(tokens={}, reused_tokens={}, rendered_bytes={}, fresh_bytes={})",
            self.token_ids.len(),
            self.reused_tokens,
            self.rendered_bytes,
            self.fresh_bytes
        )
    }
}

#[pyclass(module = "tuisko.llm._native", frozen, get_all, skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
struct GenerationDefaults {
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

impl From<RustGenerationDefaults> for GenerationDefaults {
    fn from(defaults: RustGenerationDefaults) -> Self {
        Self {
            temperature: defaults.temperature,
            top_p: defaults.top_p,
            top_k: defaults.top_k,
        }
    }
}

#[pyclass(module = "tuisko.llm._native")]
struct Frontend {
    inner: TextFrontend,
}

#[pymethods]
impl Frontend {
    #[classmethod]
    #[pyo3(signature = (checkpoint, prompt_cache_capacity = 4))]
    fn open(
        _class: &Bound<'_, PyType>,
        checkpoint: &str,
        prompt_cache_capacity: usize,
    ) -> PyResult<Self> {
        let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(Path::new(checkpoint))
            .map_err(|error| CheckpointError::new_err(error.to_string()))?;
        let inner = TextFrontend::open_with_options(
            &snapshot,
            TextFrontendOptions {
                prompt_cache_capacity,
            },
        )
        .map_err(|error| FrontendError::new_err(error.to_string()))?;

        Ok(Self { inner })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner
            .encode(text)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    #[pyo3(signature = (token_ids, skip_special_tokens = true))]
    fn decode(&self, token_ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner
            .decode(&token_ids, skip_special_tokens)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    #[pyo3(signature = (messages, add_generation_prompt = true, enable_thinking = None))]
    fn render_chat(
        &self,
        messages: Vec<(String, String)>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> PyResult<String> {
        let options = ChatTemplateOptions {
            enable_thinking,
            ..ChatTemplateOptions::default()
        };
        self.inner
            .render_chat(&chat_messages(messages), add_generation_prompt, &options)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    #[pyo3(signature = (messages, enable_thinking = None))]
    fn encode_chat(
        &self,
        messages: Vec<(String, String)>,
        enable_thinking: Option<bool>,
    ) -> PyResult<Vec<u32>> {
        let options = ChatTemplateOptions {
            enable_thinking,
            ..ChatTemplateOptions::default()
        };
        self.inner
            .encode_chat(&chat_messages(messages), &options)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    #[pyo3(signature = (messages, enable_thinking = None))]
    fn encode_chat_with_report(
        &self,
        messages: Vec<(String, String)>,
        enable_thinking: Option<bool>,
    ) -> PyResult<PromptEncoding> {
        let options = ChatTemplateOptions {
            enable_thinking,
            ..ChatTemplateOptions::default()
        };
        self.inner
            .encode_chat_with_report(&chat_messages(messages), &options)
            .map(PromptEncoding::from)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    fn stop_ids(&self) -> Vec<u32> {
        self.inner.stop_ids().to_vec()
    }

    fn generation_defaults(&self) -> GenerationDefaults {
        self.inner.generation_defaults().into()
    }
}

fn chat_messages(messages: Vec<(String, String)>) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|(role, content)| ChatMessage::new(role, content))
        .collect()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let module_name = "tuisko.llm._native";
    let tuisko_error = module.py().get_type::<TuiskoError>();
    let checkpoint_error = module.py().get_type::<CheckpointError>();
    let frontend_error = module.py().get_type::<FrontendError>();

    tuisko_error.setattr("__module__", module_name)?;
    checkpoint_error.setattr("__module__", module_name)?;
    frontend_error.setattr("__module__", module_name)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("MODEL_ID", Qwen38_27B::MODEL_ID)?;
    module.add("MODEL_REVISION", Qwen38_27B::REVISION)?;
    module.add("VOCAB_SIZE", Qwen38_27B::VOCAB)?;
    module.add("TuiskoError", tuisko_error)?;
    module.add("CheckpointError", checkpoint_error)?;
    module.add("FrontendError", frontend_error)?;
    module.add_class::<PromptEncoding>()?;
    module.add_class::<GenerationDefaults>()?;
    module.add_class::<Frontend>()?;
    Ok(())
}

//! Python bindings for the exact tokenizer and text-only chat template.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use serde_json::Value;
use std::path::Path;
use tuisko_frontend::{
    ChatFunctionCall as RustChatFunctionCall, ChatMessage as RustChatMessage, ChatTemplateOptions,
    ChatToolCall as RustChatToolCall, GenerationDefaults as RustGenerationDefaults,
    PromptEncoding as RustPromptEncoding, StreamingDecoder as RustStreamingDecoder, TextFrontend,
    TextFrontendOptions,
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
    message_boundary_tokens: usize,
    reused_tokens: usize,
    rendered_bytes: usize,
    fresh_bytes: usize,
}

impl From<RustPromptEncoding> for PromptEncoding {
    fn from(encoding: RustPromptEncoding) -> Self {
        Self {
            token_ids: encoding.token_ids,
            message_boundary_tokens: encoding.message_boundary_tokens,
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
            "PromptEncoding(tokens={}, message_boundary_tokens={}, reused_tokens={}, rendered_bytes={}, fresh_bytes={})",
            self.token_ids.len(),
            self.message_boundary_tokens,
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

#[pyclass(
    name = "ChatFunctionCall",
    module = "tuisko.llm._native",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct ChatFunctionCall {
    inner: RustChatFunctionCall,
}

#[pymethods]
impl ChatFunctionCall {
    #[new]
    #[pyo3(signature = (name, arguments = None))]
    fn new(name: String, arguments: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let arguments = match arguments {
            Some(arguments) => python_json(arguments, "arguments")?,
            None => Value::Object(Default::default()),
        };
        if !arguments.is_object() {
            return Err(PyValueError::new_err(
                "ChatFunctionCall arguments must be a JSON object",
            ));
        }
        Ok(Self {
            inner: RustChatFunctionCall { name, arguments },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_python(py, &self.inner.arguments)
    }

    fn __repr__(&self) -> String {
        format!("ChatFunctionCall(name={:?})", self.inner.name)
    }
}

#[pyclass(
    name = "ChatToolCall",
    module = "tuisko.llm._native",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct ChatToolCall {
    inner: RustChatToolCall,
}

#[pymethods]
impl ChatToolCall {
    #[new]
    #[pyo3(signature = (function, id = None, kind = "function"))]
    fn new(py: Python<'_>, function: Py<ChatFunctionCall>, id: Option<String>, kind: &str) -> Self {
        Self {
            inner: RustChatToolCall {
                id,
                kind: kind.into(),
                function: function.borrow(py).inner.clone(),
            },
        }
    }

    #[getter]
    fn id(&self) -> Option<&str> {
        self.inner.id.as_deref()
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }

    #[getter]
    fn function(&self, py: Python<'_>) -> PyResult<Py<ChatFunctionCall>> {
        Py::new(
            py,
            ChatFunctionCall {
                inner: self.inner.function.clone(),
            },
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatToolCall(id={:?}, kind={:?}, function={:?})",
            self.inner.id, self.inner.kind, self.inner.function.name
        )
    }
}

#[pyclass(
    name = "ChatMessage",
    module = "tuisko.llm._native",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct ChatMessage {
    inner: RustChatMessage,
}

#[pymethods]
impl ChatMessage {
    #[new]
    #[pyo3(signature = (
        role,
        content = "",
        *,
        reasoning_content = None,
        tool_calls = None,
        tool_call_id = None
    ))]
    fn new(
        py: Python<'_>,
        role: String,
        content: &str,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Py<ChatToolCall>>>,
        tool_call_id: Option<String>,
    ) -> Self {
        let tool_calls = tool_calls
            .unwrap_or_default()
            .iter()
            .map(|call| call.borrow(py).inner.clone())
            .collect();
        Self {
            inner: RustChatMessage {
                role,
                content: content.into(),
                reasoning_content,
                tool_calls,
                tool_call_id,
            },
        }
    }

    #[getter]
    fn role(&self) -> &str {
        &self.inner.role
    }

    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }

    #[getter]
    fn reasoning_content(&self) -> Option<&str> {
        self.inner.reasoning_content.as_deref()
    }

    #[getter]
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Vec<Py<ChatToolCall>>> {
        self.inner
            .tool_calls
            .iter()
            .cloned()
            .map(|inner| Py::new(py, ChatToolCall { inner }))
            .collect()
    }

    #[getter]
    fn tool_call_id(&self) -> Option<&str> {
        self.inner.tool_call_id.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessage(role={:?}, content={:?})",
            self.inner.role, self.inner.content
        )
    }
}

#[pyclass(
    name = "StreamingDecoder",
    module = "tuisko.llm._native",
    skip_from_py_object
)]
struct StreamingDecoder {
    inner: RustStreamingDecoder,
}

#[pymethods]
impl StreamingDecoder {
    fn push(&mut self, token_id: u32) -> PyResult<Option<String>> {
        self.inner
            .push(token_id)
            .map_err(|error| FrontendError::new_err(error.to_string()))
    }

    fn finish(&mut self) -> Option<String> {
        self.inner.finish()
    }

    #[getter]
    fn text(&self) -> &str {
        self.inner.text()
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
        class: &Bound<'_, PyType>,
        checkpoint: &str,
        prompt_cache_capacity: usize,
    ) -> PyResult<Self> {
        class.py().detach(|| {
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
        })
    }

    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.detach(|| {
            self.inner
                .encode(text)
                .map_err(|error| FrontendError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (token_ids, skip_special_tokens = true))]
    fn decode(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .decode(&token_ids, skip_special_tokens)
                .map_err(|error| FrontendError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (
        messages,
        add_generation_prompt = true,
        enable_thinking = None,
        preserve_thinking = None,
        reasoning_effort = None,
        tools = None
    ))]
    fn render_chat(
        &self,
        messages: &Bound<'_, PyAny>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        preserve_thinking: Option<bool>,
        reasoning_effort: Option<String>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let py = messages.py();
        let messages = chat_messages(messages)?;
        let options =
            chat_template_options(enable_thinking, preserve_thinking, reasoning_effort, tools)?;
        py.detach(|| {
            self.inner
                .render_chat(&messages, add_generation_prompt, &options)
                .map_err(|error| FrontendError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (
        messages,
        enable_thinking = None,
        preserve_thinking = None,
        reasoning_effort = None,
        tools = None
    ))]
    fn encode_chat(
        &self,
        messages: &Bound<'_, PyAny>,
        enable_thinking: Option<bool>,
        preserve_thinking: Option<bool>,
        reasoning_effort: Option<String>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<u32>> {
        let py = messages.py();
        let messages = chat_messages(messages)?;
        let options =
            chat_template_options(enable_thinking, preserve_thinking, reasoning_effort, tools)?;
        py.detach(|| {
            self.inner
                .encode_chat(&messages, &options)
                .map_err(|error| FrontendError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (
        messages,
        enable_thinking = None,
        preserve_thinking = None,
        reasoning_effort = None,
        tools = None
    ))]
    fn encode_chat_with_report(
        &self,
        messages: &Bound<'_, PyAny>,
        enable_thinking: Option<bool>,
        preserve_thinking: Option<bool>,
        reasoning_effort: Option<String>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PromptEncoding> {
        let py = messages.py();
        let messages = chat_messages(messages)?;
        let options =
            chat_template_options(enable_thinking, preserve_thinking, reasoning_effort, tools)?;
        py.detach(|| {
            self.inner
                .encode_chat_with_report(&messages, &options)
                .map(PromptEncoding::from)
                .map_err(|error| FrontendError::new_err(error.to_string()))
        })
    }

    fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder {
            inner: self.inner.streaming_decoder(),
        }
    }

    fn stop_ids(&self) -> Vec<u32> {
        self.inner.stop_ids().to_vec()
    }

    fn generation_defaults(&self) -> GenerationDefaults {
        self.inner.generation_defaults().into()
    }
}

fn chat_messages(messages: &Bound<'_, PyAny>) -> PyResult<Vec<RustChatMessage>> {
    let iterator = messages.try_iter().map_err(|_| {
        PyTypeError::new_err("messages must be a sequence of ChatMessage or (role, content) pairs")
    })?;
    iterator
        .enumerate()
        .map(|(index, item)| {
            let item = item?;
            if let Ok(message) = item.extract::<PyRef<'_, ChatMessage>>() {
                return Ok(message.inner.clone());
            }
            if let Ok((role, content)) = item.extract::<(String, String)>() {
                return Ok(RustChatMessage::new(role, content));
            }
            Err(PyTypeError::new_err(format!(
                "messages[{index}] must be ChatMessage or a (role, content) pair"
            )))
        })
        .collect()
}

fn chat_template_options(
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
    reasoning_effort: Option<String>,
    tools: Option<&Bound<'_, PyAny>>,
) -> PyResult<ChatTemplateOptions> {
    let tools = match tools {
        Some(tools) => {
            let value = python_json(tools, "tools")?;
            let Value::Array(tools) = value else {
                return Err(PyTypeError::new_err(
                    "tools must be a sequence of JSON objects",
                ));
            };
            if tools.iter().any(|tool| !tool.is_object()) {
                return Err(PyValueError::new_err(
                    "each tool definition must be a JSON object",
                ));
            }
            tools
        }
        None => Vec::new(),
    };
    Ok(ChatTemplateOptions {
        enable_thinking,
        preserve_thinking,
        reasoning_effort,
        tools,
    })
}

fn python_json(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Value> {
    let encoded: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&encoded)
        .map_err(|error| PyValueError::new_err(format!("invalid JSON in {field}: {error}")))
}

fn json_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let encoded = serde_json::to_string(value).expect("serde_json::Value serializes");
    Ok(py
        .import("json")?
        .call_method1("loads", (encoded,))?
        .unbind())
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
    module.add_class::<ChatFunctionCall>()?;
    module.add_class::<ChatToolCall>()?;
    module.add_class::<ChatMessage>()?;
    module.add_class::<PromptEncoding>()?;
    module.add_class::<GenerationDefaults>()?;
    module.add_class::<StreamingDecoder>()?;
    module.add_class::<Frontend>()?;
    Ok(())
}

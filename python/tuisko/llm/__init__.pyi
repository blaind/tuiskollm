"""Exact tokenizer and chat-template frontend for TuiskoLLM."""

from collections.abc import Sequence
from typing import final

MODEL_ID: str
MODEL_REVISION: str
VOCAB_SIZE: int
__version__: str

class TuiskoError(Exception):
    """Base class for errors raised by this package."""

class CheckpointError(TuiskoError):
    """The path is not the admitted checkpoint snapshot."""

class FrontendError(TuiskoError):
    """Tokenizer, template, or frontend metadata failure."""

@final
class ChatFunctionCall:
    """A function name and represented JSON-object arguments."""

    def __init__(self, name: str, arguments: dict[str, object] | None = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> dict[str, object]: ...

@final
class ChatToolCall:
    """An OpenAI-compatible historical function call."""

    def __init__(
        self,
        function: ChatFunctionCall,
        id: str | None = None,
        kind: str = "function",
    ) -> None: ...
    @property
    def id(self) -> str | None: ...
    @property
    def kind(self) -> str: ...
    @property
    def function(self) -> ChatFunctionCall: ...

@final
class ChatMessage:
    """One structured message supplied to the checkpoint chat template."""

    def __init__(
        self,
        role: str,
        content: str = "",
        *,
        reasoning_content: str | None = None,
        tool_calls: Sequence[ChatToolCall] | None = None,
        tool_call_id: str | None = None,
    ) -> None: ...
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...
    @property
    def reasoning_content(self) -> str | None: ...
    @property
    def tool_calls(self) -> list[ChatToolCall]: ...
    @property
    def tool_call_id(self) -> str | None: ...

type _ChatInput = ChatMessage | tuple[str, str]

@final
class PromptEncoding:
    """Encoded prompt and prefix-cache accounting."""

    @property
    def token_ids(self) -> list[int]: ...
    @property
    def reused_tokens(self) -> int: ...
    @property
    def rendered_bytes(self) -> int: ...
    @property
    def fresh_bytes(self) -> int: ...

@final
class GenerationDefaults:
    """Sampling defaults from the admitted generation config."""

    @property
    def temperature(self) -> float: ...
    @property
    def top_p(self) -> float: ...
    @property
    def top_k(self) -> int: ...

@final
class StreamingDecoder:
    """Incremental decoder for one generated token stream."""

    def push(self, token_id: int) -> str | None: ...
    def finish(self) -> str | None: ...
    @property
    def text(self) -> str: ...

@final
class Frontend:
    """Exact tokenizer and text-only chat template."""

    @classmethod
    def open(cls, checkpoint: str, prompt_cache_capacity: int = 4) -> Frontend: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def render_chat(
        self,
        messages: Sequence[_ChatInput],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> str: ...
    def encode_chat(
        self,
        messages: Sequence[_ChatInput],
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> list[int]: ...
    def encode_chat_with_report(
        self,
        messages: Sequence[_ChatInput],
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> PromptEncoding: ...
    def streaming_decoder(self) -> StreamingDecoder: ...
    def stop_ids(self) -> list[int]: ...
    def generation_defaults(self) -> GenerationDefaults: ...

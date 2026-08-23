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
class Frontend:
    """Exact tokenizer and text-only chat template."""

    @classmethod
    def open(cls, checkpoint: str, prompt_cache_capacity: int = 4) -> Frontend: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def render_chat(
        self,
        messages: Sequence[tuple[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str: ...
    def encode_chat(
        self,
        messages: Sequence[tuple[str, str]],
        enable_thinking: bool | None = None,
    ) -> list[int]: ...
    def encode_chat_with_report(
        self,
        messages: Sequence[tuple[str, str]],
        enable_thinking: bool | None = None,
    ) -> PromptEncoding: ...
    def stop_ids(self) -> list[int]: ...
    def generation_defaults(self) -> GenerationDefaults: ...

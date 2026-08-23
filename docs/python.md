# Python frontend

The optional `tuisko-llm` wheel exposes the exact tokenizer, text-only chat template, generation
defaults, and prompt-prefix accounting already owned by `tuisko-frontend`.

It deliberately does not expose CUDA objects, endpoint kernels, sessions, sampling, or an
in-process inference engine. Those ownership and scheduling contracts remain in the Rust server;
Python applications can use its OpenAI-compatible HTTP API for inference.

## Build and test

Python 3.12 or newer, uv, and a Rust toolchain are required to build the development wheel. CUDA
and an NVIDIA driver are not required.

```bash
uv sync --locked
uv run --locked ruff check python
uv run --locked ruff format --check python
uv run --locked mypy
uv run --locked pytest
uv run --locked maturin build --release --locked --out dist
uv run --locked maturin sdist --out dist
```

Tests using the real tokenizer and template are explicit and skipped when the pinned snapshot is
unavailable:

```bash
TUISKO_CHECKPOINT=/path/to/16b6615af3548b88e2d8e382457bc705b00479cf \
  uv run --locked pytest -m checkpoint
```

## Surface

```python
from tuisko.llm import ChatMessage, Frontend

frontend = Frontend.open("/path/to/16b6615af3548b88e2d8e382457bc705b00479cf")
messages = [ChatMessage("system", "You are concise."), ChatMessage("user", "Hello")]
prompt = frontend.encode_chat_with_report(messages, enable_thinking=False)

print(prompt.token_ids)
print(prompt.reused_tokens)
```

The structured message types also carry prior reasoning, tool calls, and tool results. Template
methods expose the checkpoint's thinking controls and tool definitions. `Frontend.streaming_decoder`
incrementally decodes generated token IDs without splitting UTF-8 text. Existing `(role, content)`
pairs remain accepted for text-only messages.

`Frontend.open` first admits the complete pinned checkpoint inventory. Errors retain the Rust
boundary's stable category in their message and are raised as either `CheckpointError` or
`FrontendError`, both subclasses of `TuiskoError`.

## Release

The `Python release` workflow builds and inspects both distributions on a manual run. A tag named
`tuisko-llm-vX.Y.Z`, exactly matching the workspace version, additionally publishes them to PyPI
and creates a GitHub release.

Before the first release, add a pending PyPI trusted publisher for project `tuisko-llm` with owner
`blaind`, repository `tuiskollm`, workflow `python-release.yml`, and environment `pypi`. The
matching GitHub environment should be protected before the first tag is pushed. No PyPI token is
stored in GitHub.

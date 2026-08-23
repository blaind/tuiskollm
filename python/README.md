# Tuisko LLM Python

`tuisko-llm` provides the exact host-side frontend for
`unsloth/Qwen3.8-27B-NVFP4` at its pinned revision. It supports:

- checkpoint admission, tokenizer encode/decode, stop IDs, and generation defaults;
- structured chat messages, reasoning history, tool calls, and template controls;
- prompt-prefix accounting and incremental UTF-8-safe token decoding.

It does not initialize CUDA or run inference in process. Use the TuiskoLLM server's
OpenAI-compatible HTTP API for generation.

## Install

```bash
pip install tuisko-llm
```

## Example

```python
from tuisko.llm import ChatMessage, Frontend

frontend = Frontend.open("/path/to/16b6615af3548b88e2d8e382457bc705b00479cf")
messages = [ChatMessage("system", "You are concise."), ChatMessage("user", "Hello")]
prompt = frontend.encode_chat_with_report(messages, enable_thinking=False)

print(prompt.token_ids)
print(prompt.reused_tokens)
```

`ChatMessage` also represents prior reasoning, function calls, and tool results. The three chat
methods accept both structured messages and existing `(role, content)` pairs. Their options expose
thinking controls and OpenAI-compatible tool definitions. `Frontend.streaming_decoder()` decodes
generated token IDs without splitting UTF-8 text.

`Frontend.open()` first admits the complete pinned checkpoint inventory. It raises
`CheckpointError` for snapshot admission failures and `FrontendError` for tokenizer or template
failures; both derive from `TuiskoError`.

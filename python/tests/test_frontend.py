from __future__ import annotations

import os
from pathlib import Path

import pytest
import tuisko.llm as llm

CHECKPOINT = os.environ.get("TUISKO_CHECKPOINT")


def test_module_identifies_the_exact_target() -> None:
    assert llm.MODEL_ID == "unsloth/Qwen3.8-27B-NVFP4"
    assert llm.MODEL_REVISION == "16b6615af3548b88e2d8e382457bc705b00479cf"
    assert llm.VOCAB_SIZE == 248_320


def test_error_hierarchy() -> None:
    assert issubclass(llm.CheckpointError, llm.TuiskoError)
    assert issubclass(llm.FrontendError, llm.TuiskoError)
    assert llm.Frontend.__module__ == "tuisko.llm._native"
    assert llm.TuiskoError.__module__ == "tuisko.llm._native"


def test_structured_messages_preserve_tool_fields() -> None:
    function = llm.ChatFunctionCall("lookup_weather", {"city": "Helsinki"})
    call = llm.ChatToolCall(function, id="call-1")
    assistant = llm.ChatMessage(
        "assistant",
        reasoning_content="I should inspect the weather.",
        tool_calls=[call],
    )
    tool = llm.ChatMessage("tool", "Snow", tool_call_id="call-1")

    assert assistant.role == "assistant"
    assert assistant.reasoning_content == "I should inspect the weather."
    assert assistant.tool_calls[0].id == "call-1"
    assert assistant.tool_calls[0].function.name == "lookup_weather"
    assert assistant.tool_calls[0].function.arguments == {"city": "Helsinki"}
    assert tool.tool_call_id == "call-1"


def test_tool_arguments_must_be_json_objects() -> None:
    with pytest.raises(ValueError, match="arguments must be a JSON object"):
        llm.ChatFunctionCall("bad", [1, 2, 3])  # type: ignore[arg-type]


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / llm.MODEL_REVISION

    with pytest.raises(llm.CheckpointError, match="checkpoint.io"):
        llm.Frontend.open(str(missing))


@pytest.fixture(scope="module")
def frontend() -> llm.Frontend:
    if CHECKPOINT is None:
        pytest.skip("TUISKO_CHECKPOINT is not set")
    return llm.Frontend.open(CHECKPOINT)


@pytest.mark.checkpoint
def test_encode_decode_round_trip(frontend: llm.Frontend) -> None:
    text = "The quick brown fox"
    token_ids = frontend.encode(text)

    assert token_ids
    assert frontend.decode(token_ids) == text


@pytest.mark.checkpoint
def test_chat_report_exposes_real_prefix_reuse(frontend: llm.Frontend) -> None:
    messages = [("system", "You are concise."), ("user", "Hello")]

    cold = frontend.encode_chat_with_report(messages, enable_thinking=False)
    warm = frontend.encode_chat_with_report(messages, enable_thinking=False)

    assert cold.token_ids == warm.token_ids
    assert cold.reused_tokens == 0
    assert warm.reused_tokens == len(warm.token_ids)
    assert warm.fresh_bytes == 0


@pytest.mark.checkpoint
def test_structured_messages_retain_tuple_compatibility(frontend: llm.Frontend) -> None:
    tuples = [("system", "You are concise."), ("user", "Hello")]
    structured = [
        llm.ChatMessage("system", "You are concise."),
        llm.ChatMessage("user", "Hello"),
    ]

    assert frontend.render_chat(tuples, enable_thinking=False) == frontend.render_chat(
        structured, enable_thinking=False
    )


@pytest.mark.checkpoint
def test_tool_definitions_and_history_reach_the_template(frontend: llm.Frontend) -> None:
    function = llm.ChatFunctionCall("lookup_weather", {"city": "Helsinki"})
    messages = [
        llm.ChatMessage("user", "What is the weather?"),
        llm.ChatMessage("assistant", tool_calls=[llm.ChatToolCall(function, id="call-1")]),
        llm.ChatMessage("tool", "Snow", tool_call_id="call-1"),
    ]
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get current weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    rendered = frontend.render_chat(
        messages,
        enable_thinking=False,
        preserve_thinking=False,
        reasoning_effort="high",
        tools=tools,
    )

    assert "lookup_weather" in rendered
    assert "Helsinki" in rendered
    assert "Snow" in rendered


@pytest.mark.checkpoint
def test_streaming_decoder_matches_complete_decode(frontend: llm.Frontend) -> None:
    token_ids = frontend.encode("Hello, maailma ✨")
    decoder = frontend.streaming_decoder()
    deltas = [delta for token_id in token_ids if (delta := decoder.push(token_id)) is not None]
    final = decoder.finish()
    if final is not None:
        deltas.append(final)

    assert "".join(deltas) == frontend.decode(token_ids)
    assert decoder.text == frontend.decode(token_ids)
    assert decoder.finish() is None
    with pytest.raises(llm.FrontendError, match="after finishing"):
        decoder.push(token_ids[0])


@pytest.mark.checkpoint
def test_generation_metadata_is_admitted(frontend: llm.Frontend) -> None:
    defaults = frontend.generation_defaults()

    assert defaults.temperature == 1.0
    assert defaults.top_p == pytest.approx(0.95)
    assert defaults.top_k == 20
    assert len(frontend.stop_ids()) == 2

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
def test_generation_metadata_is_admitted(frontend: llm.Frontend) -> None:
    defaults = frontend.generation_defaults()

    assert defaults.temperature == 1.0
    assert defaults.top_p == pytest.approx(0.95)
    assert defaults.top_k == 20
    assert len(frontend.stop_ids()) == 2

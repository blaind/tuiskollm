#!/usr/bin/env python3
"""lm-eval 0.4.12 launcher for TuiskoLLM's native scoring route."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from typing import Any


MODEL = "unsloth/Qwen3.8-27B-NVFP4"
REVISION = "16b6615af3548b88e2d8e382457bc705b00479cf"


def encode_pair(tokenizer: Any, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Preserve lm-eval 0.4.12's context/continuation token boundary."""
    if context == "":
        continuation_ids = list(tokenizer.encode(continuation, add_special_tokens=False))
        if not continuation_ids:
            return [], []
        prefix = int(tokenizer.eos_token_id)
        if continuation_ids[0] == prefix:
            return [continuation_ids[0]], continuation_ids[1:]
        return [prefix], continuation_ids

    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    context_ids = list(tokenizer.encode(context, add_special_tokens=False))
    whole_ids = list(tokenizer.encode(context + continuation, add_special_tokens=False))
    return context_ids, whole_ids[len(context_ids) :]


def fit_context(
    context: list[int], continuation: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    """Apply lm-eval's left-context truncation while retaining one context token."""
    if len(continuation) >= max_length:
        raise ValueError(
            f"continuation length {len(continuation)} leaves no context at max_length={max_length}"
        )
    return context[-(max_length - len(continuation)) :], continuation


def parse_batch_size(batch_size: int | str) -> int:
    if isinstance(batch_size, str) and batch_size.startswith("auto"):
        return 8
    value = int(batch_size)
    if value < 1:
        raise ValueError("batch_size must be positive")
    return min(value, 8)


def adjacent_context_groups(
    encoded: Sequence[tuple[list[int], list[int]]], maximum: int = 8
) -> Iterable[tuple[list[int], list[tuple[int, list[int]]]]]:
    """Group only adjacent equal contexts, preserving result positions."""
    cursor = 0
    while cursor < len(encoded):
        context = encoded[cursor][0]
        group: list[tuple[int, list[int]]] = []
        while (
            cursor < len(encoded)
            and encoded[cursor][0] == context
            and len(group) < maximum
        ):
            group.append((cursor, encoded[cursor][1]))
            cursor += 1
        yield context, group


def post_loglikelihood(
    base_url: str,
    model: str,
    context: list[int],
    continuations: list[list[int]],
    timeout: float,
) -> list[tuple[float, bool]]:
    payload = json.dumps(
        {"model": model, "context": context, "continuations": continuations}
    ).encode()
    request = urllib.request.Request(
        base_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"TuiskoLLM scoring failed with HTTP {error.code}: {detail}") from error
    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(continuations):
        raise RuntimeError("TuiskoLLM returned the wrong continuation result count")
    results: list[tuple[float, bool]] = []
    for index, item in enumerate(data):
        if item.get("index") != index:
            raise RuntimeError("TuiskoLLM returned continuation results out of order")
        results.append((float(item["logprob"]), bool(item["is_greedy"])))
    return results


def install_model() -> None:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from transformers import AutoTokenizer

    @register_model("tuisko-native")
    class TuiskoNative(LM):
        def __init__(
            self,
            model: str = MODEL,
            base_url: str = "http://127.0.0.1:8000/v1/evals/loglikelihood",
            tokenizer: str = MODEL,
            revision: str = REVISION,
            max_length: int | str = 220000,
            timeout: float | str = 600,
            batch_size: int | str = 8,
            max_batch_size: int | str | None = None,
            device: str | None = None,
            **kwargs: Any,
        ) -> None:
            if kwargs:
                raise ValueError(f"unsupported tuisko-native model arguments: {sorted(kwargs)}")
            if model != MODEL:
                raise ValueError(f"tuisko-native requires model={MODEL}")
            if revision != REVISION:
                raise ValueError(f"tuisko-native requires revision={REVISION}")
            if tokenizer != MODEL:
                raise ValueError(f"tuisko-native requires tokenizer={MODEL}")
            super().__init__()
            self._model = model
            self._base_url = base_url
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)
            self._max_length = int(max_length)
            if self._max_length < 2:
                raise ValueError("max_length must leave room for context and continuation")
            self._timeout = float(timeout)
            self._batch_size = parse_batch_size(batch_size)
            if max_batch_size is not None:
                self._batch_size = min(
                    self._batch_size, parse_batch_size(max_batch_size)
                )
            if device not in (None, "cpu"):
                raise ValueError("tuisko-native is an HTTP client and admits only device=cpu")

        @property
        def eot_token_id(self) -> int:
            return int(self._tokenizer.eos_token_id)

        @property
        def max_length(self) -> int:
            return self._max_length

        @property
        def max_gen_toks(self) -> int:
            return 0

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self) -> str:
            return "cpu"

        @property
        def tokenizer_name(self) -> str:
            return f"{MODEL}@{REVISION}"

        def tok_encode(self, string: str, **_: Any) -> list[int]:
            return list(self._tokenizer.encode(string, add_special_tokens=False))

        def tok_decode(self, tokens: Sequence[int]) -> str:
            return str(self._tokenizer.decode(tokens, skip_special_tokens=False))

        def loglikelihood(
            self, requests: Sequence[Any], disable_tqdm: bool = False
        ) -> list[tuple[float, bool]]:
            del disable_tqdm
            encoded = [
                fit_context(*encode_pair(self._tokenizer, *request.args), self.max_length)
                for request in requests
            ]
            for index, (context, continuation) in enumerate(encoded):
                if not context or not continuation:
                    raise ValueError(
                        f"native loglikelihood request {index} has an empty token boundary"
                    )
            results: list[tuple[float, bool] | None] = [None] * len(encoded)
            for context, group in adjacent_context_groups(
                encoded, maximum=self.batch_size
            ):
                continuations = [continuation for _, continuation in group]
                scores = post_loglikelihood(
                    self._base_url,
                    self._model,
                    context,
                    continuations,
                    self._timeout,
                )
                for (index, _), score in zip(group, scores, strict=True):
                    results[index] = score
                    self.cache_hook.add_partial(
                        "loglikelihood", requests[index].args, score
                    )
            if any(result is None for result in results):
                raise RuntimeError("native loglikelihood left a request unscored")
            return [result for result in results if result is not None]

        def loglikelihood_rolling(
            self, requests: Sequence[Any], disable_tqdm: bool = False
        ) -> list[float]:
            del requests, disable_tqdm
            raise NotImplementedError(
                "tuisko-native does not support rolling log-likelihood tasks"
            )

        def generate_until(
            self, requests: Sequence[Any], disable_tqdm: bool = False
        ) -> list[str]:
            del requests, disable_tqdm
            raise NotImplementedError("tuisko-native does not support generation tasks")


def main() -> None:
    install_model()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()

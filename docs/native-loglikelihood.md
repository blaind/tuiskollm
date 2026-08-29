# Native continuation log-likelihood

## Decision

TuiskoLLM exposes a model-specific scoring route at `POST /v1/evals/loglikelihood`. The route is
not part of the OpenAI API. It is the native evaluation boundary used by an lm-eval adapter; the
OpenAI-compatible `/v1/completions` echo-logprob route remains available as a compatibility path.

The native route scores one nonempty token-ID context against one to eight nonempty token-ID
continuations:

```json
{
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "context": [151644, 872, 198],
  "continuations": [[362], [426], [4267], [423]]
}
```

Text is deliberately not admitted. The client owns tokenization with the pinned checkpoint
tokenizer, including the context/continuation boundary. Unknown fields, an empty context, an empty
continuation, more than eight continuations, out-of-vocabulary IDs, and any context-plus-
continuation beyond the resident capacity are rejected.

The response preserves request order:

```json
{
  "id": "eval-tuisko-...",
  "object": "eval.loglikelihood",
  "created": 0,
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "data": [
    {
      "index": 0,
      "logprob": -0.75,
      "is_greedy": true,
      "tokens": [
        {
          "token_id": 362,
          "logprob": -0.75,
          "top_token_id": 362,
          "top_logprob": -0.75
        }
      ]
    }
  ],
  "usage": {
    "context_tokens": 3,
    "continuation_tokens": 4,
    "evaluated_tokens": 7
  }
}
```

`logprob` is the FP64 host sum of the per-token natural-log probabilities. Each token probability
is computed from the represented BF16 production logits using the existing stable FP64
normalizer. `is_greedy` is true only when every teacher-forced continuation token equals the
existing first-argmax result for its row. The response does not contain a sampled token, decoded
text, prompt-token scores, or a fabricated completion.

## Exact execution

The route runs only while the compact scheduler is idle and uses the same scratch-slot ownership,
KV pages, target prefill/decode graphs, LM head, BF16 logit download, and host normalizer as the
existing exact scoring route.

1. Reserve enough slot-zero capacity for the widest context-plus-continuation.
2. Replay the context once with the existing native prefill tiles and B=1 residual decode path.
3. Project and download only the final context row, then score every continuation's first token
   from that one normalized row.
4. Snapshot the context KV/GDN boundary. For each multi-token continuation, restore that boundary
   and teacher-force all tokens except the final token through the existing B=1 exact path,
   scoring the following token after each replay.
5. Stop after scoring the final continuation token. Do not replay it merely to obtain an unused
   next-token row.

The first implementation shares only the explicit context. A future continuation trie may also
share identical continuation prefixes, but it must preserve the independent B=1 observable
results before admission. The native route must not use the numerically different B-wide target
generation graphs as a substitute for independent B=1 scoring.

This is the same model-quality path as normal generation from token admission through the target
transformer, LM head, and represented-logit normalization. It intentionally does not exercise
sampling, MTP acceptance, text decoding, streaming, or stop handling; those remain covered by
generation qualification and server smoke tests.

## lm-eval integration

The repository adapter implements lm-eval's native `loglikelihood(context, continuation)` method.
It uses the pinned Hugging Face tokenizer, preserves lm-eval's context/continuation tokenization
boundary, groups adjacent requests with identical context IDs up to eight continuations, and sends
those groups to `/v1/evals/loglikelihood`. Generation-only and rolling-loglikelihood tasks are
rejected with an explicit unsupported-operation error in this initial adapter.

On a shared GPU the adapter issues one HTTP request at a time. One request can still score all four
MMLU alternatives because the alternatives are branches within that request, not concurrent GPU
jobs.

## Qualification

Host tests cover wire admission, response order and totals, error mapping, and the adapter's
tokenization/grouping behavior. The source-backed device acceptance test compares native results
with complete independent B=1 prompt scoring at route seams and for one-token plus multi-token
continuations. It checks every token ID, selected logprob, greedy token, greedy logprob, total, and
`is_greedy` result exactly.

The sibling benchmark times the complete four-choice production operation. It reports the native
boundary directly and does not infer its latency from prefill or LM-head leaf medians. Device
qualification remains pending whenever the exact RTX 5090 is occupied; local tests must be run
sequentially and must not retain the GPU after the bounded check.

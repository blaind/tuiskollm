# Live-server evaluation

TuiskoLLM exposes a scoring-only subset of `POST /v1/completions` for the
`local-completions` adapter in lm-evaluation-harness. This is separate from generation, which
remains owned by `POST /v1/chat/completions`.

## Exact request contract

The scoring route admits the payload emitted by lm-eval for loglikelihood requests:

```json
{
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "prompt": [[151644, 872, 198], [151644, 872, 220]],
  "temperature": 0,
  "max_tokens": 1,
  "logprobs": 1,
  "seed": 1234,
  "echo": true
}
```

`prompt` must be one nonempty token-ID array or a batch of 1..=8 nonempty token-ID arrays. The
server returns one indexed choice per prompt. Each choice contains prompt-aligned
`token_logprobs` and `top_logprobs`, followed by one greedy completion token. The first prompt
token has a null logprob because it has no causal predecessor. Unsupported completions modes are
rejected rather than silently mapped to chat generation.

The route requires an idle generation scheduler. It evicts retained prefix slot zero for scratch
ownership. Other inactive prefixes survive when the shared KV pool has enough free pages; the
normal inactive-prefix eviction policy applies under page pressure. A scoring request received
during active generation returns a retryable capacity error.

## lm-eval smoke run

Use the exact pinned checkpoint tokenizer and tokenized requests. Start with a small limit because
multiple-choice tasks score every answer alternative:

```bash
lm_eval \
  --model local-completions \
  --model_args model=unsloth/Qwen3.8-27B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=unsloth/Qwen3.8-27B-NVFP4,revision=16b6615af3548b88e2d8e382457bc705b00479cf,tokenizer_backend=huggingface,tokenized_requests=true,max_length=220000,num_concurrent=1 \
  --tasks hellaswag \
  --batch_size 4 \
  --limit 10 \
  --log_samples \
  --output_path target/lm-eval/hellaswag-smoke
```

Keep `num_concurrent=1`: one HTTP scoring job may carry up to eight prompts, while concurrent
jobs would contend for the one scoring scratch slot. Full MMLU and HellaSwag runs contain tens of
thousands of alternative prompts and can take hours; establish a measured examples/second rate
with `--limit 10`, then `--limit 100`, before scheduling a full suite.

Generation-only tasks continue to use `local-chat-completions` and
`http://127.0.0.1:8000/v1/chat/completions`.

## Scoring implementation

Normal serving prefill projects only its final row through the LM head. Scoring preserves that
production graph and consumes the final-normalized rows it already leaves in the address-stable
workspace:

1. Replay each admitted 32/64/128/1024-token target prefill tile.
2. Project retained normalized rows through the existing exact LM-head B=1..8 routes.
3. Compute the selected-token and greedy-token natural-log probabilities from represented BF16
   logits with a stable FP64 host reduction.
4. Score the residual tail shorter than 32 tokens with existing B=1 decode graphs.
5. Replay the final prompt token once and return its greedy next token as the one completion token
   expected by lm-eval.

No checkpoint tensor is converted and no new device kernel or generated symbol is introduced.
The scoring path adds LM-head launches and device-to-host logit traffic only while evaluation is
explicitly requested; it does not alter chat TTFT or decode.

## Qualification before merge

The source-backed device gate must cover prompt lengths around every route seam: 1, 2, 31, 32,
33, 63, 64, 65, 127, 128, 129, 1023, 1024, and 1025 tokens, plus one long-attention context.
For every scored row, compare the selected logprob, greedy token, and greedy logprob with an
independent host softmax over the production BF16 logits. Time the complete scoring owner for
representative short multiple-choice prompts and 1K/8K prompts; report prompts/s, scored tokens/s,
LM-head launches, and downloaded bytes. These are scoring baselines and must remain separate from
chat prefill and decode baselines.

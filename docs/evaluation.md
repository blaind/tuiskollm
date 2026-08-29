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

Every admitted scoring request emits a content-free terminal log after success or failure. The
line reports the request ID, batch size, total prompt and output-token counts, prompt-length range,
common-prefix length, queue and scoring durations, scoring throughput, total latency, and route.
It never includes prompt text, token IDs, or answer contents. For example:

```text
REQUEST 11      B4 · 244 prompt · 4 output · 0.6s · length
                accepted +5.2s · queue 0.1ms · score 0.6s (386 tok/s) · lengths 61..61 · common 60/61 (98.4%) · route prompt-scoring
```

`common` describes the request shape, not proof that reuse was admitted; the route restrictions
below still decide whether shared-prefix replay is safe.

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

For an equal-length prompt batch, the resident scorer may replay the longest exact shared route
prefix once, capture its recurrent state, and restore that state for each suffix. Reuse requires
the complete prompt reservations and route segments to be identical, and stops before the final
completion replay. Unequal lengths and batches whose shared route would include the T=1024 macro
tile use independent scoring. This keeps the optimization on the qualified MMLU-sized route while
the separate ignored `resident_mtp_batch_t1024_macro_scoring_repeatability_acceptance` documents
the ordinary macro-scoring repeatability condition that remains open.

## Qualification before merge

The base scoring gate covers prompt lengths around every route seam: 1, 2, 31, 32, 33, 63, 64,
65, 127, 128, 129, 1023, 1024, and 1025 tokens, plus one long-attention context. For every scored
row, compare the selected logprob, greedy token, and greedy logprob with an independent host
softmax over the production BF16 logits.

Shared-prefix qualification separately compares the optimized batch with complete independent
scoring at the admitted seams through a 1023-token common prefix. It includes equal-length
multi-token suffixes and requires exact equality at every observable response field. The suite
filter also selects its benchmark-accounting sibling:

```bash
cargo run -p xtask -- qualify-generation-mtp-batch "$SNAPSHOT"
cargo run -p xtask -- bench-prompt-scoring "$SNAPSHOT"
```

The benchmark directly times one production four-choice `score_prompts` call and the complete
sequence of four production `score_prompt` calls. Do not infer either boundary by adding leaf
medians, and keep scoring reports separate from chat prefill and decode baselines.

## Shared-prefix timing evidence

The 2026-08-29 RTX 5090 diagnostic used three samples, one warmup, and one operation per sample.
For four 61-token prompts with a 60-token common prefix, loaded clocks stayed at 2197 MHz and the
complete host-synchronized medians were:

| Boundary | Median | Relative to independent |
| --- | ---: | ---: |
| One shared-prefix batch | 631.902 ms | 3.56x faster |
| Four independent prompts | 2252.483 ms | reference |

The run had zero timed device-memory growth and is stored under
`target/benchmarks/prompt-scoring-shared-prefix-samples-3.json`; it is diagnostic evidence, not a
blessed baseline.

A finalized-server MMLU abstract-algebra zero-shot `--limit 10 --batch_size 4` smoke completed in
9.65 seconds wall time, versus 25.78 seconds before prefix reuse. API request progress took about
5.1 seconds, down from about 17 seconds. The aggregate accuracy remained 7/10, and all ten logged
sample prompts, loglikelihood responses, targets, and correctness fields matched the earlier run
exactly. Limited runs are timing and integration checks, not publishable quality estimates.

### Shared-row normalizer reuse

The shared boundary produces one represented-BF16 logit row for every answer alternative. Its
sequential token-order FP64 normalizer and argmax are now computed once, then reused to derive each
selected-token score. An adversarial full-vocabulary host test preserves exact f32 response bits,
including the `0.0` result for one `0.0` logit followed by 248,319 represented `-37.0` logits. The
source-backed shared-versus-independent device gate also remained bitwise exact at every admitted
route seam; its focused device phase completed in 37.12 seconds.

On the same three-sample four-choice diagnostic shape, shared-prefix scoring moved from 631.902 ms
to 629.809 ms, a 2.093 ms or 0.33% reduction. The independent reference was 2254.145 ms and is not
changed by this optimization. Loaded clocks were 2190..2197 MHz with memory at 13801 MHz, and timed
device-memory growth remained zero. The report is
`target/benchmarks/prompt-scoring-shared-normalizer-samples-3.json`; three samples remain diagnostic
and are not baseline authority.

## Rejected device-side normalizer

A second 2026-08-29 hypothesis proposed replacing full-vocabulary downloads and the host FP64
normalizer with a parallel device reduction. It was rejected before implementation or device use
because the current API result binds the sequential token-order sum.

One finite represented-BF16 counterexample has vocabulary width 248,320, token zero at `0.0`, and
every remaining token at `-37.0`. The host loop adds each `exp(-37)` after `1.0`; every contribution
is below half an FP64 ULP there, so the denominator remains exactly `1.0` and the f32 top logprob is
`0.0` (`0x00000000`). A representative 256-lane partial/tree reduction first groups those small
terms, producing denominator `1.0000000000211064` and f32 top logprob `-2.110645e-11`
(`0xadb9a780`). That is a material response change, not a one-ULP deviation. CUDA and host
transcendental implementations also have distinct identity risk.

A future compact normalizer therefore needs either an explicitly revised response-numerics
contract or an exact token-order algorithm with a directly measured production-owner win. The
rejected evidence is preserved at
`target/benchmarks/prompt-scoring-device-reduction-rejected.json`.

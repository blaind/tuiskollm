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

Use the exact pinned checkpoint tokenizer and tokenized requests. The verified lean environment is
lm-evaluation-harness 0.4.12 with its API dependencies and tokenizer-only Transformers support;
PyTorch is not required because the Rust server owns inference:

```bash
python3 -m venv target/lm-eval-venv
target/lm-eval-venv/bin/python -m pip install \
  'lm_eval[api]==0.4.12' 'transformers==5.16.1'
```

The harness writes task data and tokenizer metadata through the Hugging Face cache. Keep that cache
under the ignored `target/` tree when the global cache is read-only or when evaluation artifacts
must remain worktree-local:

```bash
export HF_HOME="$PWD/target/lm-eval-hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

Stage the tokenizer and HellaSwag before loading the resident model. This avoids holding the GPU
while the harness downloads or prepares host-only inputs:

```bash
target/lm-eval-venv/bin/python - <<'PY'
from datasets import load_dataset
from transformers import AutoTokenizer

revision = "16b6615af3548b88e2d8e382457bc705b00479cf"
AutoTokenizer.from_pretrained("unsloth/Qwen3.8-27B-NVFP4", revision=revision)
load_dataset("Rowan/hellaswag")
PY
```

Then start the server and begin with one example. Multiple-choice tasks score every answer
alternative, so one HellaSwag example produces four prompt-scoring requests:

```bash
target/lm-eval-venv/bin/lm_eval run \
  --model local-completions \
  --model_args model=unsloth/Qwen3.8-27B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=unsloth/Qwen3.8-27B-NVFP4,revision=16b6615af3548b88e2d8e382457bc705b00479cf,tokenizer_backend=huggingface,tokenized_requests=true,max_length=220000,num_concurrent=1 \
  --tasks hellaswag \
  --batch_size 4 \
  --limit 1 \
  --log_samples \
  --output_path target/lm-eval/hellaswag-smoke-limit-1
```

Keep `num_concurrent=1`: one HTTP scoring job may carry up to eight prompts, while concurrent
jobs would contend for the one scoring scratch slot. Full MMLU and HellaSwag runs contain tens of
thousands of alternative prompts and can take hours. A limited result proves plumbing only; its
accuracy is not evaluation authority. On an exclusive GPU, establish a measured examples/second
rate with `--limit 10`, then `--limit 100`, before scheduling a full suite. On a shared GPU, keep
the cases sequential and bounded, then stop the server promptly to release resident memory.

### MMLU planning estimate

The harness's `mmlu` group contains 14,042 test questions across 57 subjects and scores four
choices per question. On the exact RTX 5090 target, a ten-question `mmlu_abstract_algebra` pilot
with `batch_size=4` and `num_concurrent=1` measured:

| Setting | Complete pilot wall time | API scoring time | Full-suite planning estimate |
| --- | ---: | ---: | ---: |
| zero-shot | 25.78 s | about 17 s | about 6.5–7 hours |
| five-shot | 37.30 s | about 31 s | about 12–13 hours |

These are single-subject diagnostic extrapolations, not controlled performance results. Subject
prompt lengths vary substantially, so reserve additional time and run the full group only during
an exclusive GPU window. Pin `--num_fewshot 0` or `5` explicitly; otherwise a harness-default
change can alter both the quality result and duration.

MMLU time is prompt scoring, not a one-token generation loop. Each of a question's four choice
prompts is scored independently. The engine replays native prefill tiles, uses B=1 target decode
graphs only for the residual tail shorter than 32 tokens and the final greedy-token row, and then
projects and downloads a full 248,320-entry BF16 logit row for every causal prompt position. Each
download synchronizes the scoring stream, after which the host performs the stable FP64 softmax.

Reconstructing the exact ten-question pilot prompts from the harness samples gives this diagnostic
work inventory:

| Setting | Mean tokens/choice | Prefill tiles/choice | Decode replays/choice | Downloaded logits | Host-softmax rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 83.4 | 1.2 | 19.4 | 1.54 GiB | 3,336 |
| five-shot | 369.4 | 3.6 | 17.4 | 6.83 GiB | 14,776 |

The five-shot API phase increased from about 17 to 31 seconds while its residual decode count
slightly decreased. The incremental cost is therefore tiled prefill plus the LM-head,
device-to-host, and host-softmax scoring path. An exact percentage attribution still requires a
directly instrumented scoring-owner benchmark; do not infer it from these request-level timings.

### Short quality runs

For a shared GPU, prefer one complete MMLU subject over a limited slice of the 57-subject group.
Ten subjects have exactly 100 test questions, including `mmlu_abstract_algebra`,
`mmlu_college_computer_science`, `mmlu_computer_security`, `mmlu_medical_genetics`, and
`mmlu_global_facts`. The measured abstract-algebra pilot projects to about three minutes zero-shot
or five to six minutes five-shot for its complete subject. Other subjects vary with prompt length.

Run subjects separately and pin the shot count, for example:

```bash
target/lm-eval-venv/bin/lm_eval run \
  --model local-completions \
  --model_args model=unsloth/Qwen3.8-27B-NVFP4,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=unsloth/Qwen3.8-27B-NVFP4,revision=16b6615af3548b88e2d8e382457bc705b00479cf,tokenizer_backend=huggingface,tokenized_requests=true,max_length=220000,num_concurrent=1 \
  --tasks mmlu_abstract_algebra \
  --num_fewshot 0 \
  --batch_size 4 \
  --log_samples \
  --output_path target/lm-eval/mmlu-abstract-algebra-zero-shot
```

A complete single-subject result measures that domain but is not an aggregate MMLU score. Keep
subjects, shot counts, and output paths separate rather than presenting a hand-selected subset as
the official 57-subject metric.

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

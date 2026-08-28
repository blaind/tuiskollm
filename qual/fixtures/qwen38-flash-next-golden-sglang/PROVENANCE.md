# Qwen3.8 Flash-Next SGLang capture provenance

These captures are an external selection cross-check, not a performance or numerical baseline.
Operator and whole-model oracles remain the represented-value authority.

| | |
| :-- | :-- |
| checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| engine | SGLang `0.0.0.dev1+gd91c3682b` |
| tensor parallel | 1 |
| hardware | 1x NVIDIA B300 SXM6 AC |
| sampling | greedy |
| recorded per step | top 32 candidates with logprobs |
| context width | 8,192 |

`prompt-00..07.json` are eight unrelated prompts with 64 greedy tokens each. Each records
`prompt_ids`, `generated_ids`, the decoded `text`, and one step per generated token holding
`top_ids`, `top_logprobs`, and the `chosen` token.

`boundary-{2047,2048,2049,2050,2051,2052,2056,2100}.json` are eight prompts chosen around the
dense-QSA ceiling, with 8 greedy tokens each. The name records
`total_length_at_first_step`, one more than the prompt length.

Prompt captures use parallel `top_ids` and `top_logprobs`; boundary captures use
`[token, logprob]` pairs. `TUISKO_QWEN38_FLASH_NEXT_GOLDEN` may override this directory.

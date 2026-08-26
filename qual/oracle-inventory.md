# Reference oracle inventory (`qual/`)

Built by grep at `main` @ `67c079ad` over the 153 `.rs` files / 88,395 lines of `qual/`, per
`docs/architecture-refactoring.md` Part V §1.2 and §3.A. This file records *what was measured*;
`qual/src/oracles/` records what was extracted, and `qual/src/oracles/diff_tests.rs` proves each
absorbed site is reproduced bit-for-bit.

## How the inventory was built

```bash
# 1. every function definition name, ranked by how often it is defined
grep -rhoP '^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+\K[a-zA-Z0-9_]+' \
  --include='*.rs' qual/src | sort | uniq -c | sort -rn

# 2. the defining files for one candidate
grep -rlP '^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+<name>\b' --include='*.rs' qual/src

# 3. the distinct bodies behind one name (brace-matched extraction, whitespace-normalized,
#    grouped by hash) — duplicate *names* are not duplicate *math*
```

## Absorbed into `qual/src/oracles/` (value-identical)

| Oracle | Defining sites | Distinct bodies | All value-identical | Home |
|:--|--:|--:|:--|:--|
| `f32_to_bf16` | 30 | 5 | yes (3 spellings of one RNE expression) | `codecs.rs` |
| `bf16_to_f32` | 7 | 2 | yes (visibility only) | `codecs.rs` |
| `f32_to_f16` | 1 | 1 | pure move | `codecs.rs` |
| `f16_to_f32` | 1 | 1 | pure move | `codecs.rs` |
| `decode_e2m1` | 6 | 2 | yes (visibility only) | `codecs.rs` |
| `encode_e2m1` | 8 | 2 | yes (visibility only) | `codecs.rs` |
| `decode_e4m3fn` | 6 | 6 | yes — the six differ only in the NaN branch's error type and in `0x0f`/`15` mask spelling | `codecs.rs` |
| `encode_e4m3fn` (scale) | 7 | 7 | yes — non-negative search over `0x00..=0x7e`; sites differ only in error type and message | `codecs.rs::encode_e4m3fn_scale` |
| `encode_e4m3fn` (signed) | 1 | 1 | pure move — clamped `±FP8_MAX` search preserving signed zero | `codecs.rs::encode_e4m3fn` |
| `decode_e4m3` (f64) | 5 | 1 | yes | `codecs.rs::decode_e4m3fn_f64` |
| `residual_oracle` | 9 | 2 | yes (parameter names only) | `norm.rs` |
| `rope` | 8 | 3 | yes — `ROTARY_PAIRS = 32`, rotary dim `64`, theta `1e7` at every site | `attention.rs::rope_tables` |
| `prefill_rope` / `prefill_rope_at` | 7 | 3 | yes — same constants | `attention.rs::prefill_rope_tables` |

`pairs`, `rotary_dim`, and `theta` stay caller-supplied so no per-suite shape contract is
normalized into a harness default (Part V §3.F). Sites that spelled the rotary dimension as the
literal `64.0` now pass `2 * ROTARY_PAIRS`; sites that had a `ROTARY_DIM` constant pass it.

`fp8_projection_oracle`, `nvfp4_down_sm120`, `nvfp4_swiglu_sm120`, and `residual_norm` are
imported as codec sources by other suites, so they keep their public names as `pub(crate) use`
re-exports of `crate::oracles::codecs` (plus thin error-mapping adapters where the site's
`Result` error type is local). Repointing those ~65 importers at `crate::oracles::codecs`
directly is a namespace move for the Part V §2 directory reorganization, not part of this
extraction.

## Inventoried and deliberately NOT absorbed

| Oracle | Sites | Distinct bodies | Why it stays local |
|:--|--:|--:|:--|
| `dot_oracle` | 10 | 9 | Bound to per-suite `Fixture`, `Schedule`, seed, and weight-scale divisor; two suites even fold the group loop in-line rather than calling `group_dot`. |
| `group_dot` | 8 | 7 | Same — per-suite fixture layout and A16/W4A4 schedule branching. |
| `fp8_dot` | 7 | 7 | Per-suite accumulation shapes: some take a `TokenOracle`, others raw code slices; error types and `try_fold` closures differ. |
| `quantize_oracle` | 10 | 9 | Two unrelated operations share the name: the FP8 per-token quantizer (`fp8_projection_oracle`) and the NVFP4 per-group quantizer, itself parameterized by suite constants. |
| `decode_e4m3` in `qwen36_moe_experts.rs` | 1 | 1 | Genuinely different math from the other five: returns `f32`, reconstructs by bit assembly, and ignores the sign bit. Documented divergence, not a duplicate. |
| `rms_norm_oracle` | 1 | 1 | Confirms Part V §1.2's correction — exactly **one** definition site (`residual_norm.rs`), consumed by ~20 suites. Nothing to de-duplicate; relocating it is a namespace move for Part V §2. |
| `prefill_rope` in `resident_model.rs` | 1 | 1 | A one-line delegation to `prefill_rope_at`, whose math is absorbed. |

## Result

59 files changed, 163 insertions, 1,069 deletions — a net 906 lines of copy-pasted reference
math removed, against §3.A's ~2,200-line estimate. The remaining §3.A headroom is in the
fixture-bound `dot_oracle`/`group_dot`/`fp8_dot`/`quantize_oracle` families, which need the
Part V §3.C layer-harness fixture seam before they can be centralized without a rewrite.

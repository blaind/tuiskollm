# Qual benchmark suite -> SM120 kernel-family map (Part XI task Q3)

Derived at `main` @ `67c079ad` from the operator symbols each `qual/src/*_benchmark*.rs` file
actually names, resolved through `qual/src/target.rs` and
`crates/tuisko-kernels-sm120/src/lib.rs`'s re-export list. Part XI's Q3 row requires this table
before any benchmark-session migration, because each migrated file gates behind the K task that
owns its family.

## 1. Family membership (Part III is the authority)

| Family | Part XI letter | SM120 source | Exported operators |
|:--|:--|:--|:--|
| 1 | K1 (`-norm`) | `residual_norm.rs` | `ResidualNormOp`, `Qwen35ResidualNormOp`, `Qwen36ResidualNormOp` |
| 2 | K2 (`-attention`) | `attention/qk_prepare.rs` | `AttentionQkPrepareOp`, `Qwen35AttentionQkPrepareOp`, `Qwen36AttentionQkPrepareOp`, `Qwen36Fp8AttentionQkPrepareOp` |
| 3 | K3 (`-attention`) | `attention/paged_gqa.rs`, `attention/long_context_paged_gqa.rs` | `PagedGqaOp`, `Qwen35PagedGqaOp`, `Qwen36PagedGqaOp`, `Qwen36Fp8PagedGqaOp`, `LongContextPagedGqaOp`, `ATTENTION_PAGE_SIZE`, the `PAGED_GQA_PREFILL_*` / `LONG_CONTEXT_GQA_*` constants, `paged_gqa_prefill_partitions` |
| 4 | K4 (`-mtp`) | the six `mtp_bf16_*.rs` files | `MtpBf16{Fusion,Mlp,QkPrepare,Qkv,AttentionOutput,PagedGqa}Op` and their `Qwen35*` / `Qwen36*` siblings |
| 5 | K5 (`-nvfp4`) | `nvfp4_swiglu.rs`, `nvfp4_down.rs` | `Nvfp4SwiGluOp`, `Qwen35Nvfp4SwiGluOp`, `Nvfp4DownOp`, `Qwen35Nvfp4DownOp` |
| 6 | none (contract only) | `gdn/*`, `nvfp4_gdn_input.rs`, `qwen36_gdn_{input,output}.rs`, `fp8/gdn_*`, `attention/nvfp4_output.rs` | the GDN prepare/recurrence/snapshot/input/output operators |

Three membership calls that a name-prefix guess gets wrong, and that this table follows Part III
on instead:

- `Qwen35Nvfp4QkvOp` (`nvfp4_qkv.rs`) is **not** Family 5. Part III scopes Family 5 to
  `nvfp4_swiglu.rs` and `nvfp4_down.rs` only.
- `Qwen35Nvfp4GdnInputOp` and `Qwen35Nvfp4GdnOutputOp` are **Family 6**, not Family 5.
- `Qwen36MtpBf16MoeOp` is **not** Family 4 — Part III §Family 4 states it is qwen36-specific and
  stays concrete.

`none` below means the file's operators belong to no K task (FP8 dense projections, MoE, LM
heads, attention output, `nvfp4_qkv`). `indirect` means the benchmark drives a
`tuisko-engine` resident/layer program and names no SM120 operator: it gates on whichever K
tasks own the kernels that program launches, not on a symbol this file imports.

## 2. Per-file map (all 66 benchmark files, `device_benchmark.rs` excluded as the shared library)

| Benchmark file | Family | Evidence (first symbols matched) |
|:--|:--|:--|
| `attention_output_benchmark.rs` | none | `AttentionOutputOp` |
| `attention_qk_prepare_benchmark.rs` | K2, K3, K4 | `AttentionQkPrepareOp`, `Qwen35AttentionQkPrepareOp`, `Qwen36AttentionQkPrepareOp` |
| `bf16_paged_gqa_benchmark.rs` | K3, K4 | `ATTENTION_PAGE_SIZE`, `Qwen35PagedGqaOp`, `Qwen36Fp8PagedGqaOp` |
| `dense_fp8_gdn_layer_benchmark.rs` | indirect | engine program only |
| `dense_fp8_mlp_benchmark.rs` | indirect | engine program only |
| `fp8_down_benchmark.rs` | none | `DenseFp8DownOp`, `DenseFp8DownTmaMaps` |
| `fp8_gdn_input_benchmark.rs` | F6 | `DenseFp8GdnInputTmaMaps`, `GdnInputProjectionOp` |
| `fp8_lm_head_benchmark.rs` | none | `LmHeadOp` |
| `fp8_qkv_benchmark.rs` | none | `FullAttentionQkvOp` |
| `fp8_swiglu_benchmark.rs` | none | `DenseFp8SwiGluOp`, `DenseFp8SwiGluTmaMaps` |
| `full_attention_layer_benchmark.rs` | indirect | engine program only |
| `gdn_output_benchmark.rs` | F6 | `GdnOutputProjectionOp` |
| `gdn_prepare_benchmark.rs` | F6 | `GdnPrepareOp` |
| `gdn_recurrence_benchmark.rs` | F6 | `GdnRecurrenceOp` |
| `long_context_paged_gqa_benchmark.rs` | K3 | `ATTENTION_PAGE_SIZE`, `LONG_CONTEXT_GQA_MAX_PARTITIONS`, `LONG_CONTEXT_GQA_MAX_TOKENS` |
| `mtp_bf16_attention_output_benchmark.rs` | K4 | `MtpBf16AttentionOutputOp` |
| `mtp_bf16_fusion_benchmark.rs` | K4 | `MtpBf16FusionOp` |
| `mtp_bf16_mlp_benchmark.rs` | K4 | `MtpBf16MlpOp` |
| `mtp_bf16_qkv_benchmark.rs` | K4 | `MtpBf16QkvOp` |
| `mtp_layer_benchmark.rs` | indirect | engine program only |
| `mtp_prompt_prime_benchmark.rs` | indirect | engine program only |
| `nvfp4_down_benchmark.rs` | K5 | `Nvfp4DownOp` |
| `nvfp4_down_benchmark_sm120.rs` | K5 | `Nvfp4DownOp` |
| `nvfp4_mlp_benchmark.rs` | indirect | engine program only |
| `nvfp4_swiglu_benchmark.rs` | K5 | `Nvfp4SwiGluOp` |
| `nvfp4_swiglu_benchmark_sm120.rs` | K5 | `Nvfp4SwiGluOp` |
| `paged_gqa_benchmark.rs` | K3 | `ATTENTION_PAGE_SIZE`, `PAGED_GQA_PREFILL_LONG_PARTITION_MIN_CONTEXT`, `PAGED_GQA_PREFILL_MACRO_PARTIAL_BYTES` |
| `qwen35_full_attention_layer_benchmark.rs` | indirect | engine program only |
| `qwen35_gdn_layer_benchmark.rs` | indirect | engine program only |
| `qwen35_gdn_prepare_benchmark.rs` | F6 | `Qwen35GdnPrepareOp` |
| `qwen35_gdn_recurrence_benchmark.rs` | F6 | `Qwen35GdnRecurrenceOp` |
| `qwen35_mtp_batch_generation_benchmark.rs` | indirect | engine program only |
| `qwen35_mtp_generation_benchmark.rs` | indirect | engine program only |
| `qwen35_mtp_layer_benchmark.rs` | indirect | engine program only |
| `qwen35_nvfp4_attention_output_benchmark.rs` | none | `Qwen35Nvfp4AttentionOutputOp` |
| `qwen35_nvfp4_down_benchmark.rs` | K5 | `Qwen35Nvfp4DownOp` |
| `qwen35_nvfp4_gdn_input_benchmark.rs` | F6 | `Qwen35Nvfp4GdnInputOp` |
| `qwen35_nvfp4_gdn_output_benchmark.rs` | F6 | `Qwen35Nvfp4GdnOutputOp` |
| `qwen35_nvfp4_mlp_benchmark.rs` | indirect | engine program only |
| `qwen35_nvfp4_qkv_benchmark.rs` | none | `Qwen35Nvfp4QkvOp` |
| `qwen35_nvfp4_swiglu_benchmark.rs` | K5 | `Qwen35Nvfp4SwiGluOp` |
| `qwen35_resident_model_benchmark.rs` | indirect | engine program only |
| `qwen35_resident_mtp_benchmark.rs` | indirect | engine program only |
| `qwen35_text_endpoint_benchmark.rs` | indirect | engine program only |
| `qwen36_attention_output_benchmark.rs` | none | `Qwen36AttentionOutputOp` |
| `qwen36_fp8_qkv_benchmark.rs` | none | `Qwen36Fp8QkvOp` |
| `qwen36_full_attention_layer_benchmark.rs` | indirect | engine program only |
| `qwen36_gdn_input_benchmark.rs` | F6 | `Qwen36GdnInputOp` |
| `qwen36_gdn_moe_layer_benchmark.rs` | indirect | engine program only |
| `qwen36_gdn_output_benchmark.rs` | F6 | `Qwen36GdnOutputOp` |
| `qwen36_moe_experts_benchmark.rs` | none | `Qwen36MoeExpertsOp` |
| `qwen36_moe_router_benchmark.rs` | none | `Qwen36MoeRouterOp` |
| `qwen36_mtp_layer_benchmark.rs` | indirect | engine program only |
| `qwen36_nvfp4_lm_head_benchmark.rs` | none | `Qwen36Nvfp4LmHeadOp` |
| `qwen36_resident_model_benchmark.rs` | indirect | engine program only |
| `qwen36_text_endpoint_benchmark.rs` | indirect | engine program only |
| `resident_model_benchmark.rs` | K3 | `LONG_CONTEXT_GQA_PARTITION_BUCKETS`, `LONG_CONTEXT_GQA_PARTITION_SIZE` |
| `resident_mtp_batch_generation_benchmark.rs` | indirect | engine program only |
| `resident_mtp_benchmark.rs` | indirect | engine program only |
| `resident_mtp_generation_benchmark.rs` | indirect | engine program only |
| `resident_mtp_sampling_benchmark.rs` | indirect | engine program only |
| `residual_norm_benchmark.rs` | K1 | `Qwen35ResidualNormOp`, `Qwen36ResidualNormOp`, `ResidualNormOp` |
| `startup_benchmark.rs` | indirect | engine program only |
| `target_mtp_verify_benchmark.rs` | indirect | engine program only |
| `text_endpoint_benchmark.rs` | indirect | engine program only |

## 3. `struct Session` contention ledger

59 of the 66 benchmark files carry the near-identical `struct Session` / `struct RouteGraph`
pair Part V §3.B targets (verified by grep at `67c079ad`; the §3.B figure of 59 is exact).
§3.B admits a file for migration only when no open branch modifies it. Blockers below are
unmerged local branches that **modify** the file relative to `main` (an add-only diff from a
stale merge base is not a conflict and is excluded).

| `struct Session` file | Modifying unmerged branches |
|:--|:--|
| `attention_output_benchmark.rs` | `fix/gpu-api-guardrails`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `attention_qk_prepare_benchmark.rs` | `fable-merge-preview`, `fable-mtp-unroll-ac367ff`, `fix/gpu-api-guardrails`, `perf/qwen35-attention-qk-prepare-resources`, `perf/qwen36-gdn-recurrence-prefill-resources`, `refactor/gpu-arena-helpers` |
| `bf16_paged_gqa_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `dense_fp8_gdn_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `dense_fp8_mlp_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `fp8_down_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `fp8_gdn_input_benchmark.rs` | `fix/gpu-api-guardrails`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `fp8_lm_head_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `fp8_qkv_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `feat/fp8-qkv-t16`, `feat/sm89-fp8-qkv`, `fix/gpu-api-guardrails`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `fp8_swiglu_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `full_attention_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `gdn_output_benchmark.rs` | `fix/gpu-api-guardrails`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `gdn_prepare_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `gdn_recurrence_benchmark.rs` | `fix/gpu-api-guardrails`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `long_context_paged_gqa_benchmark.rs` | `fable-merge-preview`, `fix/gpu-api-guardrails`, `perf/optimize-qwen38`, `refactor/gpu-arena-helpers` |
| `mtp_bf16_attention_output_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `mtp_bf16_fusion_benchmark.rs` | `fix/gpu-api-guardrails`, `fix/mtp-bench-batch-layout`, `refactor/gpu-arena-helpers` |
| `mtp_bf16_mlp_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `mtp_bf16_qkv_benchmark.rs` | `fix/gpu-api-guardrails`, `fix/mtp-bench-batch-layout`, `refactor/gpu-arena-helpers` |
| `mtp_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `mtp_prompt_prime_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `nvfp4_down_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `nvfp4_down_benchmark_sm120.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `nvfp4_mlp_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `nvfp4_swiglu_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `feat/sm86-nvfp4-swiglu`, `feat/sm89-nvfp4-down`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `nvfp4_swiglu_benchmark_sm120.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `paged_gqa_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_full_attention_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_gdn_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_gdn_prepare_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_gdn_recurrence_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_mtp_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_attention_output_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_down_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_gdn_input_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_gdn_output_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_mlp_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_qkv_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_nvfp4_swiglu_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_resident_model_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_resident_mtp_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen35_text_endpoint_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_attention_output_benchmark.rs` | `docs/qwen36-text-support`, `feat/qwen36-attention-output-prefill`, `feat/qwen36-full-attention-prefill`, `feat/qwen36-gdn-moe-layer-prefill`, `feat/qwen36-resident-prefill`, `feat/qwen36-residual-norm-prefill`, `feat/qwen36-server-native-prefill`, `fix/gpu-api-guardrails`, `fix/qwen36-resident-gate-accounting`, `perf/qwen36-attention-output-prefill-resources`, `perf/qwen36-residual-norm-prefill-resources`, `refactor/gpu-arena-helpers` |
| `qwen36_fp8_qkv_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_full_attention_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `docs/qwen36-text-support`, `feat/qwen36-full-attention-prefill`, `feat/qwen36-gdn-moe-layer-prefill`, `feat/qwen36-resident-prefill`, `feat/qwen36-server-native-prefill`, `fix/gpu-api-guardrails`, `fix/qwen36-resident-gate-accounting`, `refactor/gpu-arena-helpers` |
| `qwen36_gdn_input_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_gdn_moe_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `docs/qwen36-text-support`, `feat/qwen36-gdn-moe-layer-prefill`, `feat/qwen36-resident-prefill`, `feat/qwen36-server-native-prefill`, `fix/gpu-api-guardrails`, `fix/qwen36-resident-gate-accounting`, `refactor/gpu-arena-helpers` |
| `qwen36_gdn_output_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_moe_experts_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_moe_router_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_mtp_layer_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction` |
| `qwen36_nvfp4_lm_head_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `qwen36_resident_model_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `docs/qwen36-text-support`, `feat/qwen36-resident-prefill`, `feat/qwen36-server-native-prefill`, `fix/gpu-api-guardrails`, `fix/qwen36-benchmark-prefill-scaling`, `fix/qwen36-resident-gate-accounting`, `refactor/gpu-arena-helpers` |
| `qwen36_text_endpoint_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `resident_model_benchmark.rs` | `fable-merge-preview`, `fix/gpu-api-guardrails`, `perf/optimize-qwen38`, `perf/qwen38-prefill-opt`, `refactor/gpu-arena-helpers` |
| `resident_mtp_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `fix/resident-mtp-benchmark-accounting`, `refactor/gpu-arena-helpers` |
| `residual_norm_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `docs/qwen36-text-support`, `fable-merge-preview`, `feat/qwen36-full-attention-prefill`, `feat/qwen36-gdn-moe-layer-prefill`, `feat/qwen36-resident-prefill`, `feat/qwen36-residual-norm-prefill`, `feat/qwen36-server-native-prefill`, `feat/residual-norm-sm89-sm86`, `fix/gpu-api-guardrails`, `fix/qwen36-resident-gate-accounting`, `perf/optimize-qwen38`, `perf/qwen36-gdn-recurrence-prefill-resources`, `perf/qwen36-residual-norm-prefill-resources`, `refactor/gpu-arena-helpers` |
| `target_mtp_verify_benchmark.rs` | `chore/qual-harness-helpers`, `chore/qual-oracle-extraction`, `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |
| `text_endpoint_benchmark.rs` | `fix/gpu-api-guardrails`, `refactor/gpu-arena-helpers` |

## 4. Migrated in this task

Only two benchmark files carry no modifying unmerged branch **and** duplicate a whole measured
lifecycle with a sibling, so only those two moved onto the shared session:

| File | Family | Sibling |
|:--|:--|:--|
| `qwen35_mtp_generation_benchmark.rs` | indirect (Qwen3.5 MTP generator) | pair |
| `resident_mtp_generation_benchmark.rs` | indirect (Qwen3.8 MTP generator) | pair |

`harness/benchmark_session.rs` carries the mechanics; `MtpGreedyBenchmarkSpec` carries every
identity field (metric route names, report suite/classification/timing scope, and all eight
refusal texts) per suite, and each suite pins its own identity in a unit test. Warmup counts,
sample counts, launches per sample, and the alternating `[0, 1]` / `[1, 0]` task order are
unchanged.

Everything else in section 3 is deferred to whichever branch lands first. The two heaviest
blockers are `fix/gpu-api-guardrails` and its stacked `refactor/gpu-arena-helpers`, which move
`GpuTimer` out of `struct Session` in 58 of the 59 files — the exact lines a §3.B migration
rewrites. Sequencing that pair before the remaining §3.B work removes most of this ledger.

`resident_mtp_sampling_benchmark.rs` is unblocked but deliberately **not** migrated: it measures
three rotating tasks against two references (identity- and penalty-conditioned), so it deviates
from the paired-task lifecycle and keeps its bespoke form per §3.G's migration rule.

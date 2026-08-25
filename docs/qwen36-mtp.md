# Qwen3.6 MTP implementation contract

This document fixes the implementation boundary for the single MTP layer in
`nvidia/Qwen3.6-35B-A3B-NVFP4` revision
`491c2f1ea524c639598bf8fa787a93fed5a6fbce`. It does not generalize MTP across
models or GPU targets.

## Reference behavior

The draft consumes the target model's hidden state and the embedding of the
current token. It independently applies zero-centered RMSNorm to both inputs,
concatenates them, projects `[4096] -> [2048]`, and runs one full-attention
Qwen3.5-MoE decoder layer. A final zero-centered RMSNorm feeds the target's
shared LM head.

The decoder layer is exact Qwen3.6 geometry:

- BF16 query-plus-gate `[8192,2048]`, K/V `[512,2048]`, and output
  `[2048,4096]` projections;
- 16 query heads, two KV heads, width 256, 3-axis MRoPE, and gated attention;
- a BF16 `[256,2048]` router selecting and renormalizing exactly eight experts;
- 256 BF16 routed experts with gate/up `[1024,2048]` and down `[2048,512]`;
- one BF16 width-512 shared expert multiplied by
  `sigmoid(shared_expert_gate)`; and
- the same zero-centered `1e-6` normalization and residual order as a target
  full-attention layer.

This is independently encoded from the pinned checkpoint and qualified against
plain target transitions. The current upstream executable reference is vLLM's
[`Qwen3_5MoeMTP`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py);
Transformers deliberately ignores the checkpoint's `mtp.*` tensors in its base
model loader.

## Source contract

The admitted snapshot contains 19 BF16 MTP tensors in shard three, totaling
1,689,281,536 bytes:

| Family | Shape | Bytes |
| --- | ---: | ---: |
| input fusion | `[2048,4096]` | 16,777,216 |
| Q/gate, K, V | `[8192,2048]`, two `[512,2048]` | 37,748,736 |
| attention output | `[2048,4096]` | 16,777,216 |
| routed router | `[256,2048]` | 1,048,576 |
| routed gate/up | `[256,1024,2048]` | 1,073,741,824 |
| routed down | `[256,2048,512]` | 536,870,912 |
| shared gate/up/down and shared gate | two `[512,2048]`, `[2048,512]`, `[1,2048]` | 6,295,552 |
| seven normalization planes | five `[2048]`, two `[256]` | 21,504 |

The existing dense `MtpBindings` is not this contract. Qwen3.6 receives a
separate `Qwen36MtpBindings` that validates all names, ranks, shapes, dtypes,
and exact bytes. It binds the routed 3-D tensors directly. The source orders
the shared expert as down, gate, up and does not authorize an adjacency span;
any runtime fusion is an explicit lossless gather with a byte-for-byte test.

MTP weights remain BF16 because the pinned quantization exclusion covers
`mtp.layers.0*` and `mtp*`. They are never passed through the target's NVFP4 or
FP8 weight paths. The target embedding and LM head are shared rather than
duplicated.

## Cache prerequisite

The admitted ModelOpt metadata selects an FP8 KV cache. The draft model's
cache configuration inherits the target cache dtype in the upstream MTP
contract. Target and MTP therefore own separate E4M3 K/V storage but share one
logical page lifecycle and the same stable per-slot page mappings.

The current Qwen3.6 resident program's 192-position BF16 cache is qualification
scaffolding. It is not the production cache and must be replaced before MTP or
compact serving is admitted. The replacement has:

- one shared physical page pool across at most eight active or retained slots;
- ten target full-attention K/V layer pairs and one separate MTP K/V pair;
- 64 positions per page, two KV heads, width 256, and represented E4M3 bytes;
- per-request reserve, truncate, retain, recycle, and cancellation at scheduler
  round boundaries; and
- a logical maximum of 262,144 tokens without reserving eight independent
  full-context caches.

At the full 4,096-page capacity, target cache data is 2,684,354,560 bytes and
the MTP mirror is 268,435,456 bytes. Page tables, alignment, CUDA context,
graphs, MTP weights, and workspaces remain additional. A real resident-layout
and device-headroom gate must pass before the server route can be enabled.

Every target page-table mutation publishes the identical mapping to the MTP
table. Storage never aliases. A mismatched route, token count, or ownership
state is an `engine.layout` or `engine.route` error before graph replay.

## Device program

Qwen3.5's speculative transaction and page-owner mechanics are reusable, but
its device entries are not. Qwen3.6 receives separately named SM120 entries for
2,048 hidden columns, two KV heads, and the BF16 top-8 MoE.

The exact leaf inventory is:

1. embedding/hidden RMSNorm plus BF16 input fusion at `B=1..8`;
2. BF16 QKV, Q/K norm, MRoPE, FP8 cache append, paged GQA, and gated BF16
   attention output at `B=1..8`;
3. residual publication, BF16 top-8 router, routed and shared expert execution,
   fixed-order reduction, and final RMSNorm at `B=1..8`;
4. causal prime and rollback/realignment routes for `K=1..4`; and
5. prompt cache priming at exact `T=32,64,128` tiles.

The already-qualified Qwen3.6 RMSNorm, Q/K preparation arithmetic, paged-GQA
geometry, BF16 router formula, and endpoint may be reused only through the
Qwen3.6 symbols whose represented semantics match. New BF16 QKV/output and
BF16 expert projections require their own entries. Shared low-level helpers may
be extracted after the second exact call site proves identical accumulation
order; there is no runtime backend trait.

Every entry requires an independent represented-value oracle, eager/CUDA Graph
agreement, inactive-region and immutable-source checks, stable addresses, no
post-warmup allocation, and zero stack/local memory. Router ties, top-eight
ordering, renormalization, and expert accumulation order are pinned in tests.

## Resident and generation behavior

The concrete owner composes the target model, the separate MTP layer, their
mirrored FP8 cache owners, and the shared endpoint. It captures exact graph
inventories over address-stable allocations; the decode hot path performs only
checked exact-route selection.

Initial production scope is:

- singleton draft/verify with `K=1..4`;
- compact target and draft rows over stable physical slots at `B=1..8`;
- target verification, accepted-prefix publication, and rejected-suffix
  rollback for every GDN state/history plane and both cache owners;
- MTP cache priming after each exact native-prefill tile and scalar tail;
- immediate physical-slot reuse after a completion or cancellation boundary;
  and
- context growth through the shared page pool up to the admitted 262,144-token
  logical limit.

The route policy is a pure table-tested function. Qwen3.5 currently uses
singleton MTP and compact plain-target execution for `B=2..8`; Qwen3.6 starts
with the same safe policy but does not retain it as performance authority until
direct target-plus-draft-plus-verification measurements establish the winning
route for each batch.

Acceptance uses the existing independent speculative-sampling law. Qualification
checks accepted IDs, emitted deltas, target and MTP positions, GDN rollback,
final hidden/logit realignment, holes and slot reuse, eager/graph agreement, and
zero post-warmup growth. Plain target output with the same seed remains the
correctness authority.

Vision is not part of this stack. Multimodal prompt embeddings may later feed
the same admitted text transition, but Vision ownership and qualification stay
separate.

## MR sequence

1. `docs/qwen36-mtp-design`
2. `feat/qwen36-mtp-source-bindings`
3. `feat/qwen36-long-context-fp8-kv`
4. `feat/qwen36-mtp-bf16-fusion-attention`
5. `feat/qwen36-mtp-bf16-moe`
6. `perf/qwen36-mtp-resources`
7. `feat/qwen36-mtp-layer`
8. `feat/qwen36-resident-mtp`
9. `feat/qwen36-mtp-target-verify`
10. `perf/qwen36-mtp-target-verify-resources`
11. `feat/qwen36-mtp-generation`
12. `feat/qwen36-mtp-compact-batching`
13. `feat/qwen36-mtp-serving`

The cache branch includes target long-context ownership because production MTP
cannot be correct over the current short BF16 cache. If its implementation
reveals independently reviewable codec and lifecycle slices, split them without
changing this ordering dependency.

A measured performance baseline remains a separate, explicitly approved
`perf/qwen36-mtp-baseline` MR. Leaf medians are diagnostic; the performance
claim uses directly timed target-plus-draft-plus-verification transactions and
the complete server at the clock recorded in each artifact.

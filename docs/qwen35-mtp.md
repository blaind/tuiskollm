# Qwen3.5 MTP implementation contract

This document fixes the implementation boundary for the single MTP layer in
`AxionML/Qwen3.5-9B-NVFP4` revision
`97aef92393f126bf649f310cd40861be8dad3279`. It does not generalize MTP across
models or GPU targets.

## Source contract

The admitted checkpoint contains one non-dedicated-embedding MTP layer and 15
BF16 tensors. `MtpBindings` already validates their exact Qwen3.5 names,
shapes, dtypes, and gate/up adjacency.

The runtime owner preserves these BF16 words exactly:

- input fusion `[4096, 8192]`;
- Q/gate, K, and V gathered losslessly into `[10240, 4096]`;
- attention output `[4096, 4096]`;
- adjacent gate/up `[24576, 4096]` and down `[4096, 12288]`;
- seven 4,096- or 256-wide normalization planes.

The MTP-only weights total 486,581,248 bytes. The target model's BF16 embedding
source and resident BF16 LM head are shared; they must not be copied into a
second multi-gigabyte owner.

## Device program

Qwen3.8's MTP algorithm is reusable, but its kernels are not: they are compiled
for 5,120 hidden columns, 24 query heads, 6,144 attention columns, and 17,408
MLP columns. Qwen3.5 receives separately named SM120 entries specialized for
4,096 hidden columns, 16 query heads, 4,096 attention columns, and 12,288 MLP
columns.

Shared low-level BF16 MMA helpers may be parameterized over the sealed
`Sm120Arch` geometry. Existing Qwen3.8 entry names, schedules, generated
resource envelopes, and performance authorities remain unchanged. There is no
runtime backend trait and no model dispatch inside a kernel launch.

The exact leaf inventory is:

1. embedding/hidden RMSNorm plus BF16 input fusion at `B=1..8`;
2. BF16 QKV, Q/K norm, MRoPE, cache append, paged GQA, and gated BF16 attention
   output at `B=1..8`;
3. residual publication, BF16 gate/up SwiGLU, BF16 down, final RMSNorm, and the
   shared Qwen3.5 BF16 LM head at `B=1..8`;
4. causal prime/realign routes for `K=1..4`;
5. prompt cache priming at the target's exact `T=32,64,128` tile inventory.

Every entry requires an independent represented-value oracle, eager/CUDA Graph
agreement, inactive-region checks, stable addresses, and zero stack/local
memory before composition.

## Cache ownership

MTP owns a separate BF16 K/V mirror because its attention state is not target
attention state. At 4,096 physical pages, four KV heads, 64 positions per page,
and width 256, the complete mirror is 1,073,741,824 bytes.

It reuses the target's host page ownership and stable page-table rows. Target
reservation, truncation, retention, and recycling publish the identical page
mapping to the MTP table. The two cache arenas therefore share lifecycle and
logical routes but never storage. A mismatch is a generation error before
replay.

## Resident and generation behavior

The concrete owner is `Qwen35ResidentMtpProgram { target, mtp }`. Selection
happens when the Qwen3.5 worker starts; the decode hot path contains only exact
graph lookup.

Initial production scope matches the qualified Qwen3.8 mechanism:

- singleton draft/verify with `K=1..4`;
- compact `B=1..8` draft rows over stable physical slots;
- target verification and accepted-prefix realignment;
- prompt priming after each exact Qwen3.5 native-prefill tile;
- cancellation and completion only at scheduler round boundaries.

Acceptance uses the existing independent speculative-sampling law. A gate must
check accepted token IDs, emitted deltas, target/MTP cache positions, final
hidden/logit realignment, slot reuse, and zero post-warmup allocation. Plain
target output remains the correctness authority for a fixed random seed.

## MR sequence

1. `docs/qwen35-mtp-design`
2. `feat/qwen35-mtp-bf16-fusion`
3. `feat/qwen35-mtp-bf16-attention`
4. `feat/qwen35-mtp-bf16-mlp`
5. `perf/qwen35-mtp-resources`
6. `feat/qwen35-mtp-layer`
7. `feat/qwen35-resident-mtp`
8. `feat/qwen35-mtp-generation`
9. `feat/qwen35-mtp-serving`

A measured performance baseline remains a separate, explicitly approved
`perf/qwen35-mtp-baseline` MR. Leaf medians are diagnostic; the performance
claim uses the directly timed target-plus-draft-plus-verification transaction.

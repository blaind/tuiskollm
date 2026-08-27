# Qwen3.8-Flash-Next GDN routes

This slice admits the pinned target's GDN prepare and recurrence operators on SM120. It does not
compose a model layer or widen `Sm120Arch`.

## Route ownership

`Qwen38FlashNext` and `Qwen38_27B` share the GDN QKV, control, head, and convolution geometry.
Compile-time assertions bind every reused route to those exact equalities.

| Boundary | Route |
| --- | --- |
| Control projection, `B=1..8` | New target-qualified entries because hidden width is 2,560 |
| Control projection, `T=1..4,32,64,128,1024` | New target-qualified entries |
| Causal convolution and history publication | Existing Qwen3.8-27B entries |
| Serial recurrence prefill | Existing Qwen3.8-27B entries before the output gate |
| Decode recurrence, `B=1..8` | New sigmoid-gated entries |
| Prefill recurrence epilogue, `T=1..4,32,64,128,1024` | New sigmoid-gated entries |
| State snapshot | Existing Qwen3.8-27B entry with identical byte extents |

The target adds 32 entries to `tuisko_kernels_sm120_gdn`: eight decode control, eight prefill
control, eight decode recurrence, and eight prefill recurrence epilogue entries. The family grows
from 102 to 134 entries. The aggregate inventory derives its total from the per-family table.

## Numerical boundary

The target pins `output_gate_type = "sigmoid"`; Qwen3.8-27B uses SiLU. The independent recurrence
oracle computes sigmoid from its definition and carries a separate SiLU result so the fixture
proves the activation difference. The artifact gate requires reciprocal instructions and refuses
division instructions in every new gated entry.

Both suites require eager and CUDA Graph replay agreement, restored mutable state, immutable inputs
and weights, inactive-tail sentinels, stable addresses, and unchanged device allocation after
warmup. Their benchmark-accounting tests share the qualification filter.

## Commands

```bash
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-qwen38-flash-next-gdn-prepare
cargo run -p xtask -- qualify-qwen38-flash-next-gdn-recurrence
cargo run -p xtask -- bench-qwen38-flash-next-gdn-prepare
cargo run -p xtask -- bench-qwen38-flash-next-gdn-recurrence
```

The resource gates require preserved launch bounds, zero stack and local memory, exact shared-memory
and register baselines, and the expected PTX/SASS math instructions.

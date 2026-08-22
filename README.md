# TuiskoLLM

TuiskoLLM is an exact-target Rust/SM120 inference server for
`unsloth/Qwen3.8-27B-NVFP4` at revision
`16b6615af3548b88e2d8e382457bc705b00479cf` on one NVIDIA GeForce RTX 5090.

## Development

Requires Linux, Git, rustup, CUDA Toolkit 13.3.73, and Clang/libclang 21. Device commands also
require the NVIDIA driver and an RTX 5090. Rust is pinned by `rust-toolchain.toml`; set `CUDA_HOME`
only for a nonstandard Toolkit location.

## Build

Build the top-level executable with:

```bash
cargo build --release --bin tuiskollm
```

The output is `target/release/tuiskollm`. The executable is currently a scaffold; serving and its
checkpoint command line have not landed yet.

The optional `tuisko-llm` Python package exposes the admitted tokenizer and chat-template frontend.
It does not claim an in-process inference API; see [`docs/python.md`](docs/python.md).

`tuisko-engine` owns the exact 64-layer resident text program: all source-native weights, 48 GDN
history/state pairs, 16 current 192-token-per-slot attention KV caches, endpoint weights, one shared
workspace, and immutable whole-model CUDA Graphs for every `B=1..8` route. Compact active rows can
address any distinct physical state/cache slots, and one slot can be reset without touching its
survivors. Server wiring has not landed. Concrete single-slot and compact eight-request generation
owners connect the admitted frontend, sampling, streaming decode, and resident graphs for prompts
within the current 192-token cache. The compact owner preserves the final emitted token as pending,
packs only requests needing device work, cancels without advancing that pending token, and recycles
holes without moving survivor state. Inactive slots retain their exact processed token span and may
skip only a prefix that the next prompt contains in full; divergence falls back to cold priming.
Prompt priming uses the exact B=1 decode route until optimized prefill routes are admitted.

## Current device slice

Bootstrap the pinned cuda-oxide toolchain, then build and qualify the current device slice:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-residual-norm
cargo run -p xtask -- qualify-fp8-qkv
cargo run -p xtask -- qualify-fp8-gdn-input
cargo run -p xtask -- qualify-fp8-lm-head
cargo run -p xtask -- qualify-nvfp4-swiglu
cargo run -p xtask -- qualify-nvfp4-down
cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT
cargo run -p xtask -- qualify-resident-model SNAPSHOT
cargo run -p xtask -- qualify-resident-generation SNAPSHOT
cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT
cargo run -p xtask -- perf smoke
```

`xtask` keeps its cuda-oxide checkout, backend, and nested Cargo home under the ignored `target/`
tree; no shell-level `CARGO_HOME` override is needed.

See [`docs/performance.md`](docs/performance.md) for the command reference, metric definitions,
memory, energy, baseline, and refusal contracts.

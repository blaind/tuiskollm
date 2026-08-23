# TuiskoLLM

TuiskoLLM is an exact-target Rust/SM120 inference server for
`unsloth/Qwen3.8-27B-NVFP4` at revision
`16b6615af3548b88e2d8e382457bc705b00479cf` on one NVIDIA GeForce RTX 5090.

## Development

Requires Linux, Git, rustup, CUDA Toolkit 13.3.73, and Clang/libclang 21. Device commands also
require the NVIDIA driver and an RTX 5090. Rust is pinned by `rust-toolchain.toml`; set `CUDA_HOME`
only for a nonstandard Toolkit location.

## Build

Build the top-level executable through the pinned CUDA compiler and run every resource gate with:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-server
```

The output is `target/cuda-oxide-build-sm120/release/tuiskollm`. A plain `cargo build` cannot
finalize the embedded device artifacts. Start the exact resident server with the pinned snapshot
directory and an optional numeric listen address:

```bash
target/cuda-oxide-build-sm120/release/tuiskollm serve SNAPSHOT 127.0.0.1:8000
```

Tagged releases attach a stripped Linux x86-64 server and checksum. The archive requires glibc
2.35 or newer, the NVIDIA driver, and the exact RTX 5090, but not Rust or the CUDA Toolkit.

It exposes `GET /health`, `GET /v1/models`, and OpenAI-compatible blocking or SSE
`POST /v1/chat/completions`. The server loads and admits the complete checkpoint before binding the
listener, owns one bounded resident scheduling queue, and refuses a different model identity or
request options that the current product cannot honor.

The optional `tuisko-llm` Python package exposes the admitted tokenizer and chat-template frontend.
It does not claim an in-process inference API; see [`python/README.md`](python/README.md).

`tuisko-engine` owns the exact 64-layer resident text program: all source-native weights, 48 GDN
history/state pairs, one shared 3,438-page E4M3 KV pool across 16 attention layers, endpoint weights,
one shared workspace, and immutable whole-model CUDA Graphs for every `B=1..8` route. The current
decode path reserves three pages per active slot, so request admission remains 192 tokens while the
remaining long-context pages stay unassigned. An allocation-free 113,454-byte host owner maintains
the eight stable device-table rows, recycles physical pages between slots, and clears a reassigned
page before publishing its new route. Compact active rows can address any distinct physical
state/cache slots, and one slot can be reset without touching its survivors. The HTTP worker owns
that scheduler and disconnecting a response cancels its resident
request without moving survivors. Concrete single-slot and compact eight-request generation owners
connect the admitted frontend, sampling, streaming decode, and resident graphs for prompts within
the current 192-token cache. The compact owner preserves the final emitted token as pending, packs
only requests needing device work, cancels without advancing that pending token, and recycles holes
without moving survivor state. Inactive slots retain their exact processed token span and may skip
only a prefix that the next prompt contains in full; divergence falls back to cold priming. Prompt
priming uses the exact B=1 decode route until optimized prefill routes are admitted. Vision inputs,
MTP generation, and prefill routes are not served yet and are rejected or remain outside the HTTP
contract rather than silently taking another route.

The standalone SM120 operator inventory also includes partitioned paged GQA through 220,000
positions at every exact `B=1..8` route. Its resident-program integration is a later slice; its
presence does not expand the server's current 192-token admission limit.

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
cargo run -p xtask -- qualify-long-context-paged-gqa
cargo run -p xtask -- qualify-resident-model SNAPSHOT
cargo run -p xtask -- qualify-resident-generation SNAPSHOT
cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT
cargo run -p xtask -- perf smoke
```

`xtask` keeps its cuda-oxide checkout, backend, and nested Cargo home under the ignored `target/`
tree; no shell-level `CARGO_HOME` override is needed.

See [`docs/performance.md`](docs/performance.md) for the command reference, metric definitions,
memory, energy, baseline, and refusal contracts.

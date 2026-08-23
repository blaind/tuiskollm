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
one shared workspace, and immutable whole-model CUDA Graphs for every `B=1..8` route. Decode keeps
the short graph through 192 positions and selects one of six partitioned graph buckets above it,
with an exact 220,000-position per-request ceiling. The shared 3,438-page pool is divided among
active slots, so aggregate admission may refuse concurrent requests whose rounded page counts
exceed the pool. An allocation-free 113,454-byte host owner maintains the eight stable device-table
rows, recycles physical pages between slots, and clears a reassigned page before publishing its new
route. Compact active rows can address any distinct physical state/cache slots, and one slot can be
reset without touching its survivors. The HTTP worker owns that scheduler and disconnecting a
response cancels its resident request without moving survivors. Concrete single-slot and compact
eight-request generation owners connect the admitted frontend, sampling, streaming decode, and
resident graphs through the 220,000-position ceiling. The compact owner preserves the final emitted
token as pending, packs only requests needing device work, cancels without advancing that pending
token, and recycles holes without moving survivor state. Inactive slots retain their exact processed
token span and may skip only a prefix that the next prompt contains in full; divergence falls back
to cold priming. Prompt
priming uses the exact B=1 decode route until optimized prefill routes are admitted, so long prompts
are a correctness path rather than a production-TTFT path. Vision inputs,
MTP generation, and prefill routes are not served yet and are rejected or remain outside the HTTP
contract rather than silently taking another route.

The standalone SM120 operator inventory also includes partitioned paged GQA through 220,000
positions at every exact `B=1..8` route. The resident program owns its maximum-B partial workspace
once and captures all six partition buckets without changing addresses after warmup. Its
zero-centered RMSNorm leaves retain separate exact `T=32,64,128,1024` symbols for plain input and
the fused BF16 residual-publication seam. The
source-native full-attention QKV owner admits exact `T=32,64,128,1024` prefill projections in
addition to `B=1..8` decode and `T=16` MTP. Q/K zero-centered normalization, MRoPE, and represented
E4M3 cache append admit the same four prefill widths alongside `B=1..8` decode. Paged GQA and
attention now also admits shared-cache causal `T=32,64,128` early-context tails: one 384-thread CTA
shares each 64-position E4M3 K/V tile across two tokens and their twelve grouped-query warps. Exact
deep `T=128` tails use FP8/F16 flash attention: one 256-thread CTA owns 32 query rows, one query
head, and one partition; dynamically represented E4M3 Q and source E4M3 K feed QK Tensor Cores,
while represented F16 probabilities and V feed PV Tensor Cores. P8 uses 64-position tiles through
32,768 positions, P16 uses 32-position tiles through 220,000, and both publish complete FP32
softmax states to the existing reducer. The `T=1024` macro leaf reuses the two-CTA K32 producer
through exact `P=1,2,4,8,16` routes and has a separately specialized reducer for each; the
source-backed full-attention owner selects P4. Gated attention output admits exact
`T=32,64,128,1024` routes:
one CTA per token publishes the sigmoid-gated FP32 seam and its dynamic E4M3 representation, then
32x32 or 64x32 native E4M3 MMA tiles project through the source-native output matrix. Dense-FP8
gate/up SwiGLU also admits exact `T=1024`: a 128x64x64 three-stage TMA route retains the represented
E4M3 activation and source-weight planes and owns two stable tensor-map descriptors. Dense-FP8
down projection admits exact `T=32,64,128` K=128 MMA tails plus a separate `T=1024` 128x64x64
three-stage TMA route over its source-native `[5120,17408]` weight plane with the same explicit
descriptor ownership. The source-backed dense-FP8 MLP owner composes residual norms, gate/up,
SwiGLU, down projection, and residual publication into directly qualified graphs at every
`B=1..8` and `T=32,64,128,1024`. The source-backed full-attention owner composes that MLP with
input norm, QKV, Q/K preparation and cache append, paged GQA, gated output projection, and both
residual seams at the same exact widths. Its prefill graphs own separate from-empty causal metadata,
the shared 24-page cache row, P4 macro partials, and stable MLP tensor maps. The source-native
dense-FP8 GDN Q/K/V/Z input projection separately admits the same four prefill widths through
64x64 E4M3 MMA tiles. GDN control and width-4 causal-convolution preparation admits the same
widths, retaining one mapped history row and publishing its final three represented values only
after every parallel convolution reader completes. GDN recurrence admits the same widths with one
causally advanced mapped FP32 state row; 48 value-head CTAs retain the decode reduction order while
each CTA advances tokens serially. GDN output retains one token-owned dynamic quantization CTA per
row and projects `T=32,64,128` with 32x32 native E4M3 MMA tiles or `T=1024` with 64x32 macro tiles.
GDN layer composition and resident/server routing remain separate slices, so these routes do not
yet change server priming.

## Current device slice

Bootstrap the pinned cuda-oxide toolchain, then build and qualify the current device slice:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-residual-norm
cargo run -p xtask -- qualify-fp8-qkv
cargo run -p xtask -- qualify-fp8-gdn-input
cargo run -p xtask -- qualify-fp8-lm-head
cargo run -p xtask -- qualify-fp8-swiglu
cargo run -p xtask -- qualify-fp8-down
cargo run -p xtask -- qualify-nvfp4-swiglu
cargo run -p xtask -- qualify-nvfp4-down
cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT
cargo run -p xtask -- qualify-attention-qk-prepare
cargo run -p xtask -- qualify-paged-gqa
cargo run -p xtask -- qualify-long-context-paged-gqa
cargo run -p xtask -- qualify-resident-model SNAPSHOT
cargo run -p xtask -- bench-resident-long-context-model SNAPSHOT
cargo run -p xtask -- qualify-resident-generation SNAPSHOT
cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT
cargo run -p xtask -- perf smoke
```

`xtask` keeps its cuda-oxide checkout, backend, and nested Cargo home under the ignored `target/`
tree; no shell-level `CARGO_HOME` override is needed.

See [`docs/performance.md`](docs/performance.md) for the command reference, metric definitions,
memory, energy, baseline, and refusal contracts.

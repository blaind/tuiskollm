# TuiskoLLM

<p align="center">
  <img src="assets/tuiskollm-hero.webp" alt="TuiskoLLM" width="100%">
</p>

<p align="center">
  <strong>Hyper-optimized LLM inference. One binary.</strong>
</p>

<p align="center">
  Qwen3.8-27B-NVFP4 · RTX 5090 · NVIDIA driver only
</p>

---

TuiskoLLM is a native Rust inference server for a **small, predefined set of models and NVIDIA
GPUs**.

Instead of supporting everything, Tuisko specializes the entire execution path for each
model x GPU target: kernels, memory layout, CUDA Graphs, scheduling, and qualification.

**No PyTorch. No CUDA Toolkit at runtime. No Triton. No JIT. No model conversion.**

## Quick start

A tagged release is one stripped Linux x86-64 executable with its device code embedded. It needs
the NVIDIA driver, the exact target GPU, and glibc 2.35 or newer, and nothing else.

Name the model, and the pinned revision is resolved from the local snapshot cache:

```bash
./tuiskollm serve unsloth/Qwen3.8-27B-NVFP4
```

A directory holding that snapshot works too. The server never downloads: fetch the checkpoint once
with any tool you like, at the revision listed under [Models](#models).

The server loads and admits the complete checkpoint before it binds the listener, so a successful
bind means the model is resident. Then point any OpenAI-compatible client at:

```text
http://127.0.0.1:8000/v1
```

It exposes `GET /health`, `GET /v1/models`, and blocking or SSE `POST /v1/chat/completions`. An
optional second argument replaces the default `127.0.0.1:8000` listen address. Another model
identity, or a request option the current product cannot honor, is refused rather than silently
served by another route.

## Models

Served today:

* **Qwen3.8-27B-NVFP4**, from `unsloth/Qwen3.8-27B-NVFP4` at revision
  `16b6615af3548b88e2d8e382457bc705b00479cf`

Admitted at the checkpoint and model layer, not served yet:

* **Qwen3.5-9B-NVFP4**, from `AxionML/Qwen3.5-9B-NVFP4`

GPU targets:

* **RTX 5090 / SM120**, the complete product target and the only one that serves
* **RTX 4090 / SM89** and **RTX 3090 / SM86**, partial operator inventories with diagnostic-only
  feasibility sweeps, not usable servers

Tuisko only runs combinations that have an explicit execution target. There is no generic fallback
backend.

## Performance

The checkpoint is consumed directly in its upstream NVFP4 representation, so there is no
Tuisko-specific repack and no converted model artifact. Decode runs as immutable whole-model CUDA
Graphs over resident weights at every exact `B=1..8` route, with paged GQA through 220,000
positions over one shared E4M3 KV pool.

Prompt priming still uses the exact `B=1` decode route until optimized prefill routes are admitted,
so long prompts are a correctness path rather than a production TTFT path. End-to-end throughput
and TTFT figures are not published yet; a formal benchmark will state prompt length, context, MTP
acceptance, clocks, and build revision.

See [`docs/performance.md`](docs/performance.md) for the benchmark commands, metric definitions,
and the baseline and refusal contracts.

## Why?

General inference engines optimize for breadth.

Tuisko deliberately trades breadth for:

* model-specific fused kernels
* GPU-generation-specific execution
* very small deployment
* fast startup
* low interactive latency
* reproducible qualification

The goal is simple:

> **If Tuisko supports a model x GPU pair, make that pair run exceptionally well.**

## Build from source

Requires Linux, Git, rustup, CUDA Toolkit 13.3.73, and Clang/libclang 21. Device commands also
require the NVIDIA driver and an RTX 5090. Rust is pinned by `rust-toolchain.toml`; set `CUDA_HOME`
only for a nonstandard Toolkit location.

Build the top-level executable through the pinned CUDA compiler and run every resource gate:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-server
```

The output is `target/cuda-oxide-build-sm120/release/tuiskollm`. A plain `cargo build` cannot
finalize the embedded device artifacts. `xtask` keeps its cuda-oxide checkout, backend, and nested
Cargo home under the ignored `target/` tree; no shell-level `CARGO_HOME` override is needed.

Build and qualify the current device slice:

```bash
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
cargo run -p xtask -- bench-resident-long-context-model SNAPSHOT
cargo run -p xtask -- qualify-resident-generation SNAPSHOT
cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT
cargo run -p xtask -- perf smoke
```

The optional `tuisko-llm` Python package exposes the admitted tokenizer and chat-template frontend.
It does not claim an in-process inference API; see [`python/README.md`](python/README.md).

## Status

TuiskoLLM is experimental and under active development.

Text generation is served, through one bounded resident scheduling queue with an exact 220,000
position per-request ceiling; disconnecting a response cancels its resident request without moving
the survivors. Vision inputs, MTP generation, and optimized prefill routes are rejected or stay
outside the HTTP contract rather than silently taking another route.

Performance, supported targets, and APIs may change before 1.0.

## License

MIT OR Apache-2.0, at your option. See [`LICENSE-MIT`](LICENSE-MIT) and
[`LICENSE-APACHE`](LICENSE-APACHE).

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

## Current device slice

Bootstrap the pinned cuda-oxide toolchain, then build and qualify the current device slice:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-residual-norm
cargo run -p xtask -- qualify-fp8-qkv
cargo run -p xtask -- qualify-fp8-gdn-input
cargo run -p xtask -- perf smoke
```

`xtask` keeps its cuda-oxide checkout, backend, and nested Cargo home under the ignored `target/`
tree; no shell-level `CARGO_HOME` override is needed.

See [`docs/performance.md`](docs/performance.md) for the command reference, metric definitions,
memory, energy, baseline, and refusal contracts.

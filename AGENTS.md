# TuiskoLLM engineering contract

TuiskoLLM is an exact-target Rust/SM120 inference server for
`unsloth/Qwen3.8-27B-NVFP4` at revision
`16b6615af3548b88e2d8e382457bc705b00479cf` on one NVIDIA GeForce RTX 5090.

## Scope

- Preserve the checkpoint's represented FP4, FP8, and BF16 values. Layout conversion may reorder
  source words; it must not decode and requantize them.
- Do not add arbitrary model discovery, generic Transformer graphs, GGUF, or a backend abstraction.
- The terminal deliverable is one Rust-owned OpenAI-compatible server with Text, MTP, and the
  checkpoint's Vision behavior. The current implemented slice is documented in `README.md`.

## Ownership

- `tuisko-model` owns checkpoint admission, typed source views, and lossless materialization.
- `tuisko-gpu` owns raw CUDA resources and model-independent checked wrappers.
- `tuisko-kernels-sm120` owns device code and prepared concrete `*Op` launchers. Operators do not
  allocate their inputs, outputs, weights, or scratch space.
- `tuisko-engine` owns resident weights, address-stable workspaces, CUDA Graphs, slots, and
  scheduling. Qualification code never becomes a production dependency.
- `qual` owns independent oracles, fixtures, probes, and benchmarks. `xtask` owns device builds,
  artifact inspection, resource gates, and explicit baseline blessing.

## Device changes

An admitted device route requires all of the following in the same feature:

1. a structurally independent represented-value or mathematical oracle;
2. eager and CUDA Graph replay agreement over every observable boundary;
3. an explicit inventory for every admitted exact route, including all `B=1..8` entries;
4. generated PTX/SASS checks, preserved launch bounds, and zero stack/local memory; and
5. a benchmark using the production operation, allocations, stream, cache regime, and shape.

Do not infer composed or end-to-end wall time by adding leaf medians. Do not bless a performance
baseline to hide a regression. Resource or performance baseline changes remain separate commits.
The runner must refuse a busy device or incomparable clocks; it never changes clock or power state.
See `docs/performance.md` for commands and measurement semantics.

## Change discipline

- Use focused branches such as `feat/...`, `fix/...`, `perf/...`, `docs/...`, or `chore/...` and keep
  MRs independently reviewable.
- Keep comments brief. Document contracts, safety, and measured launch-shape rationale; do not
  narrate obvious code. Preserve blank lines between logical blocks.
- Preserve unrelated working-tree changes. Keep generated PTX, SASS, cubins, profiles, model files,
  benchmark reports, and build products out of Git; repository-local outputs belong under `target/`.
- Do not commit or push unless the user explicitly requests it.
- Count skips separately from passes. A deferred fix needs an `#[ignore]`d acceptance test that
  states the missing condition.

## Verification

Run the host checks relevant to every change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude tuisko-gpu --exclude tuisko-kernels-sm120 \
  --exclude tuisko-qual --all-targets -- -D warnings
cargo test --workspace --exclude tuisko-gpu --exclude tuisko-kernels-sm120 \
  --exclude tuisko-qual
cargo deny --workspace --all-features check
```

For a device change, also run:

```bash
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-<changed-suite>
```

If an exclusive RTX 5090 is unavailable, report the device gate as pending rather than passed.

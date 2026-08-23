# TuiskoLLM engineering contract

TuiskoLLM is an exact-model Rust inference server, with SM120 as its complete product target, for
`unsloth/Qwen3.8-27B-NVFP4` at revision
`16b6615af3548b88e2d8e382457bc705b00479cf` on one NVIDIA GeForce RTX 5090.
SM89 and SM86 remain partial qualification targets until their complete inventories close.

## Scope

- Preserve the checkpoint's represented FP4, FP8, and BF16 values. Layout conversion may reorder
  source words; it must not decode and requantize them.
- Do not add arbitrary model discovery, generic Transformer graphs, GGUF, or a backend abstraction.
- The terminal deliverable is one Rust-owned OpenAI-compatible server with Text, MTP, and the
  checkpoint's Vision behavior. The current implemented slice is documented in `README.md`.

## Ownership

- `tuisko-model` owns checkpoint admission, typed source views, and lossless materialization.
- `tuisko-frontend` owns tokenizer admission, chat templates, prompt encoding, and streaming text.
- `tuisko-provision` owns pinned Hugging Face cache resolution, download, and snapshot admission.
- `tuisko-gpu` owns raw CUDA resources and model-independent checked wrappers.
- `tuisko-targets` owns exact GPU identity and startup profile selection.
- Each `tuisko-kernels-sm*` crate owns one architecture's device code, inventories, tuning, and
  prepared concrete `*Op` launchers. Operators do not allocate inputs, outputs, weights, or scratch.
- `tuisko-engine` owns resident weights, address-stable workspaces, CUDA Graphs, slots, and
  scheduling. Qualification code never becomes a production dependency.
- `qual` owns independent oracles, fixtures, probes, and benchmarks. `xtask` owns device builds,
  artifact inspection, resource gates, and explicit baseline blessing.

## Target expansion

- Admit a new exact target in this order: pinned config, complete tensor inventory, typed source
  bindings, lossless materialization, qualified kernels, composed layers, resident program, then
  serving.
- Admit device support per operator and exact route. Do not widen a model- or GPU-level sealed
  trait if that makes unfinished operators constructible.
- Add a second exact implementation beside mature code first. Extract a shared helper only after
  both real call sites prove identical represented-value semantics, layout, and accumulation order.
- A shared device-code change reruns every consuming product's numerical, resource, and performance
  sentinels. A new target must not silently alter an existing product.
- Generated symbol hashes may change when a CUDA module gains entries. Compare semantic
  inventories and normalized function bodies; symbol-name stability alone is not artifact
  stability.
- Document quantization-convention conversion at the checkpoint-adapter boundary and independently
  test source words, scale permutations, and scalar bits. Do not bury the conversion in a kernel
  launcher.
- Qualify synthetic represented-value cases before real checkpoint ownership, then require a real
  source-backed layer gate before whole-model composition.
- Describe partial support precisely. Source admission or one qualified kernel is not model
  inference support.

## Device changes

An admitted device route requires all of the following in the same feature:

1. a structurally independent represented-value or mathematical oracle;
2. eager and CUDA Graph replay agreement over every observable boundary;
3. an explicit inventory for every admitted exact route, including all `B=1..8` entries;
4. generated PTX/SASS checks, preserved launch bounds, and zero stack/local memory; and
5. a benchmark using the production operation, allocations, stream, cache regime, and shape.

An `xtask qualify-*` filter must select both the suite's numerical/device oracle and its sibling
benchmark-accounting tests. A green oracle with those accounting tests filtered out is incomplete.
If that filter selects multiple device tests, run them with one test thread or sequence them in one
test; parallel preflights race their CUDA contexts and can falsely report a busy device. Between
sequential contexts, let process-free residual utilization settle to exactly zero without relaxing
the memory, process-count, or clock gates.
Do not infer composed or end-to-end wall time by adding leaf medians. Do not bless a performance
baseline to hide a regression. Resource or performance baseline changes remain separate commits.
Run the device preflight before opening a large source snapshot. Numerical summary assertions must
respect the represented output dtype and magnitude; they cannot be stricter than the per-value
acceptance contract at the observed ULP.
Remote source-backed fixtures must stage and explicitly admit every pinned artifact opened by the
production path, including frontend files when tokenizer or template ownership is exercised.
For a reused workspace, inactive-tail checks begin after the widest surviving writer, not
necessarily the final writer. If an owner's output aliases its next input, a repeated benchmark
must restore the production input before every measured replay.
The runner must refuse a busy device or incomparable clocks; it never changes clock or power state.
After owner warmup, validate clocks under sustained production-graph load before starting a long
timing matrix. A failed loaded-clock probe ends the run early. If clocks drift only after that
probe, preserve the completed medians as diagnostic evidence before refusing the run. Explicit
uncontrolled-clock reports are diagnostic only and can never become performance authority.
See `docs/performance.md` for commands and measurement semantics.

## Optimization loop

- Change one measured hypothesis per iteration. Record both agent-loop wall time and command/device
  wall time, and preserve accepted, rejected, refused, and failed evidence under `target/` with its
  clocks, medians, and resources.
- Refuse a wrong or occupied local device before qualification or build work. Reuse device evidence
  or build artifacts only through ignored exact-input receipts bound to the applicable device,
  toolchain, resource authorities, and artifact hashes; never share a mutable Cargo target between
  worktrees.
- An exact route or `B`-only run is an inner-loop diagnostic. Record the selection in the report
  and never compare or bless it as the complete admitted inventory.
- Scale warmup and timing repetitions to the production boundary's measured duration. A resident
  model graph must not inherit a microsecond leaf's repetition count; reports and baselines bind
  the selected counts as part of measurement identity.
- After the changed numerical and resource gate passes, directly time every affected boundary in
  the checked leaf-to-owner-to-resident/server dependency cone. A leaf win is not a model win, and
  no composed result may be inferred by summing constituent medians.
- Before comparison or blessing, repeat the dependency cone with every admitted exact route and
  its authoritative defaults. Keep each composed boundary's baseline independent from leaf
  resource and performance baselines.
- Revert a rejected implementation before beginning the next hypothesis. A noisy composed result
  cannot rescue a clear leaf regression, and changed resource authority remains a separate slice.
- Profile only the production owner or CUDA Graph after allocation and warmup. Attribute profiler
  nodes through an exact semantic owner/stage manifest, require the observed graph inventory to
  match it, and close graph span against kernel time and gaps before drawing Amdahl conclusions.
  Profiles remain diagnostic artifacts under `target/`, never checked performance authority.

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
- When implementation reveals a durable repository-wide invariant or recurring failure mode not
  covered here, propose a concise `AGENTS.md` update in the handoff. Branch-specific status,
  measurements, and one-off implementation details belong in the relevant design or performance
  document.

## Releases

- A manual `Release` workflow run builds without publishing; only a matching `vX.Y.Z` tag publishes.
- Before tagging, qualify the downloaded server archive on the exact RTX 5090 and pinned snapshot.

## Verification

Run the host checks relevant to every change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude tuiskollm --exclude tuisko-engine \
  --exclude tuisko-gpu --exclude tuisko-kernels-sm86 \
  --exclude tuisko-kernels-sm89 --exclude tuisko-kernels-sm120 \
  --exclude tuisko-qual --exclude tuisko-serve --all-targets -- -D warnings
cargo test --workspace --exclude tuiskollm --exclude tuisko-engine \
  --exclude tuisko-gpu --exclude tuisko-kernels-sm86 \
  --exclude tuisko-kernels-sm89 --exclude tuisko-kernels-sm120 \
  --exclude tuisko-qual --exclude tuisko-serve
cargo deny --workspace --all-features check
```

For a device change, also run:

```bash
cargo run -p xtask -- build-sm120
cargo run -p xtask -- qualify-<changed-suite>
```

If an exclusive RTX 5090 is unavailable, report the device gate as pending rather than passed.
With explicit user permission, an agent may instead use `xtask remote`; see
`docs/remote-gates.md`. Remote runs create billable RunPod resources, and this new runner may still
have lifecycle bugs. After every attempt, verify that no gate pod remains with
`cargo run -p xtask --features remote -- remote check`; use the corresponding `remote sweep` if
needed and report any retained pod immediately. Cleanup is worktree-owned until another runner's
declared lease and grace period expire; never replace it with prefix-wide deletion. Remote
qualification can satisfy numerical and
resource gates, but remote benchmark timings are diagnostic and cannot bless a performance
baseline. Treat provisioning, upload, source staging, and artifact-load errors before the test or
timing process starts as infrastructure failures, not device evidence. After the same
pre-execution failure repeats on two billable fresh pods, stop renting equivalent pods; increasing
`--max-minutes` alone is not a diagnosed change. Ephemeral source pods do not resume the pinned
23.4 GB snapshot transfer, so another attempt requires a materially different, verified staging
path. Report that the test ran zero times and diagnose locally.

# Performance and capacity qualification

TuiskoLLM uses a custom device runner for GPU measurements and Criterion for pure host work. GPU
results are valid only on the exact RTX 5090 target under an exclusive, recorded environment.

The available device suites cover zero-centered residual/RMSNorm at exact `B=1..8`, and
dynamic-quantize FP8 QKV at exact `B=1..8` plus `T=16`, GDN Q/K/V/Z input projection at exact
`B=1..8`, the full-vocabulary FP8 LM head at exact `B=1..8`, dense-FP8 gate/up SwiGLU at exact
`B=1..8` and `T=32,64,128`, dense-FP8 down and GDN control/convolution at exact `B=1..8`, and the
GDN recurrence and source-native output projection at exact `B=1..8`, plus the source-backed
dense-FP8 MLP, complete layer-60 GDN, and final-norm plus LM-head owners. NVFP4 gate/up SwiGLU
uses the exact retained A16 and W4A4 decode schedules at `B=1..8`; NVFP4 down projection consumes
the represented E2M1/E4M3 source planes through exact A16 routes at `B=1..8`. Full-attention Q/K
preparation covers zero-centered normalization, the 64-wide three-axis MRoPE, and represented E4M3
KV-cache append at exact `B=1..8`. Short-context paged GQA covers exact 24-query/4-KV-head,
256-wide online-softmax decode across page boundaries at `B=1..8`; long-context partitioned decode
and prefill remain separate future routes. The resident text owner composes all 48 GDN layers, 16
attention layers, source-routed MLPs, and the LM head into one directly timed graph at every exact
`B=1..8`; serving cases remain future work.

SM89 has separate remote-only diagnostic suites for the exact `[34816,5120]` NVFP4 gate/up,
`[5120,17408]` down, and dynamic-quantize FP8 `[14336,5120]` QKV owners at `B=1..8`. The NVFP4
routes decode the admitted E2M1 words and swizzled E4M3 scales inside A16 kernels; none creates a
requantized weight artifact or implies that the partial SM89 inventory is a usable server. The
SM89 QKV inventory deliberately excludes the Blackwell T=16 specialization. These numerical and
static-resource gates are authoritative, while their uncontrolled-clock RunPod timings are
feasibility evidence only.

The first RTX 4090 sweep measured 109.984 us / 849.45 logical GiB/s at `B=1` and 184.320 us /
508.46 logical GiB/s at `B=8`, with observed SM clocks of 2,625--2,700 MHz and a 10,251 MHz memory
clock. This establishes a viable source-native SM89 decode path, but the falling multi-row
bandwidth leaves `B=5..8` open for an Ada-specific reuse schedule before the target inventory can
make a performance-complete claim.

The first SM89 down sweep measured 46.304 us / 1,009.28 logical GiB/s at `B=1` and 122.176 us /
384.92 logical GiB/s at `B=8`, with fixed observed clocks of 2,775 MHz SM and 10,251 MHz memory.
The route is a viable source-native leaf, but its falling multi-row bandwidth likewise leaves an
Ada-specific reuse schedule open.

The first SM89 QKV sweep measured 24.800 us / 2,759.35 logical GiB/s at `B=1` and 39.872 us /
1,724.33 logical GiB/s at `B=8`, with fixed observed clocks of 2,520 MHz SM and 10,251 MHz memory.
This admits the exact decode route and its graph topology; it is not evidence for the absent T=16
prefill route or a complete SM89 attention owner.

The corresponding RTX 3090 feasibility sweep measured 177.408 us / 526.62 logical GiB/s at `B=1`
and 412.896 us / 226.98 logical GiB/s at `B=8`, with fixed observed clocks of 1,800 MHz SM and
9,501 MHz memory. The represented path is correct, but this first Ampere schedule is not efficient
enough to justify expanding the complete SM86 inventory before a target-specific retune.

## Quick start

Run commands from the repository root. Bootstrap cuda-oxide once:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
```

`xtask` places its cuda-oxide checkout, backend, and nested Cargo home under `target/`; do not
override the outer Cargo invocation's home.

Check that the GPU is idle and has no foreign compute process:

```bash
nvidia-smi
```

Then run a quick benchmark:

```bash
cargo run -p xtask -- perf smoke
```

The human-readable table goes to stderr. The machine-readable report is written to:

```text
target/benchmarks/perf-smoke/residual-norm.json
target/benchmarks/perf-smoke/fp8-qkv.json
target/benchmarks/perf-smoke/fp8-gdn-input.json
target/benchmarks/perf-smoke/fp8-lm-head.json
target/benchmarks/perf-smoke/fp8-swiglu.json
target/benchmarks/perf-smoke/fp8-down.json
target/benchmarks/perf-smoke/nvfp4-swiglu.json
target/benchmarks/perf-smoke/nvfp4-down.json
target/benchmarks/perf-smoke/gdn-prepare.json
target/benchmarks/perf-smoke/gdn-recurrence.json
target/benchmarks/perf-smoke/gdn-output.json
target/benchmarks/perf-smoke/attention-qk-prepare.json
target/benchmarks/perf-smoke/paged-gqa.json
target/benchmarks/perf-smoke/attention-output.json
```

Every performance command also executes the release SM120 build and checks the PTX/SASS entry and
resource inventory before launching the benchmark.

## Command reference

| Command | Purpose | Output |
|---|---|---|
| `cargo run -p xtask -- build-sm120` | Build the release device artifact and check entries, registers, stack, local, and shared bytes | terminal |
| `cargo run -p xtask -- qualify-frontend SNAPSHOT` | Check exact template, tokenizer, streaming, and prefix-cache behavior | terminal |
| `cargo run -p xtask -- qualify-generation SNAPSHOT` | Check prompt-to-sampling-to-streaming state over exact BF16 logit rows | terminal |
| `cargo run -p xtask -- qualify-residual-norm` | Run the independent numerical and graph-replay oracle | terminal |
| `cargo run -p xtask -- qualify-fp8-qkv` | Run the independent represented-value QKV oracle and benchmark-accounting test | terminal |
| `cargo run -p xtask -- qualify-fp8-gdn-input` | Run the independent represented-value GDN input oracle and benchmark-accounting test | terminal |
| `cargo run -p xtask -- qualify-fp8-lm-head` | Run the independent represented-value full-vocabulary LM-head oracle and benchmark-accounting test | terminal |
| `cargo run -p xtask -- qualify-fp8-swiglu` | Run the exhaustive represented-value gate/up SwiGLU oracle and graph-replay gate | terminal |
| `cargo run -p xtask -- qualify-fp8-down` | Run the exhaustive represented-value dense-FP8 down oracle and graph-replay gate | terminal |
| `cargo run -p xtask -- qualify-nvfp4-swiglu` | Check represented E2M1/E4M3 seams, A16/W4A4 production routing, immutable weights, graph replay, stable addresses, and post-warmup allocation at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-nvfp4-down` | Check represented E2M1/E4M3 down projection, immutable input/weights, graph replay, stable addresses, and post-warmup allocation at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT` | Check source layer 55, route-specific A16/W4A4 scratch, every observable seam, exact-B graphs, immutable weights, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-gdn-prepare` | Check the two control formulas, mapped width-4 convolution/history updates, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-gdn-recurrence` | Check mapped FP32 state transitions, gated normalization, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-gdn-output` | Check dynamic E4M3 quantization, source-native output projection, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-attention-qk-prepare` | Check Q/K zero-centered normalization, three-axis MRoPE, represented E4M3 cache append, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-paged-gqa` | Check exact page lookup, grouped-head mapping, represented E4M3 online softmax, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-attention-output` | Check sigmoid gating, the published FP32 seam, dynamic E4M3 quantization, source-native projection, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-dense-fp8-mlp SNAPSHOT` | Check source layer 60, every exact-B graph, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-dense-fp8-gdn-layer SNAPSHOT` | Check the complete source layer-60 mixer/MLP seams, persistent state, exact-B graphs, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-full-attention-layer SNAPSHOT` | Check complete source layer-63 attention/MLP seams, represented KV cache, exact-B graphs, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-resident-model SNAPSHOT` | Check all 64 source routes, final source-backed formulas, dynamic page recycling/remapping and isolated reset, persistent state/cache, exact-B whole-model graphs, stable device/host addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-resident-generation SNAPSHOT` | Check pinned vLLM next-token fixtures plus frontend, greedy control, streaming decode, stable ownership, and zero post-warmup device allocation | terminal |
| `cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT` | Compare compact mixed-length scheduling with sequential requests, including every B=1..8 route, noncontiguous survivor replay, cancellation, exact retained-prefix reuse, divergence fallback, slot recycling, stable ownership, and zero post-warmup device allocation | terminal |
| `cargo run -p xtask -- qualify-text-endpoint SNAPSHOT` | Check source embeddings, final norm, sampled full-formula logits, graph replay, stable addresses, and post-warmup allocation | terminal |
| `cargo run -p xtask -- bench-gdn-prepare` | Measure every exact control-plus-convolution graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-gdn-recurrence` | Measure every exact stateful recurrence graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-gdn-output` | Measure every exact output quantize-plus-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-attention-qk-prepare` | Measure every exact Q/K prepare and cache-append graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-paged-gqa` | Measure every exact paged GQA graph at a 130-token, three-page context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-attention-output` | Measure every exact sigmoid-gate, quantize, and output-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-swiglu` | Measure every exact retained A16/W4A4 NVFP4 gate/up SwiGLU graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-down` | Measure every exact represented-weight A16 NVFP4 down-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-mlp SNAPSHOT` | Measure every complete source-backed layer-55 MLP graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-dense-fp8-gdn-layer SNAPSHOT` | Measure every complete source-backed layer-60 graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-full-attention-layer SNAPSHOT` | Measure every complete source-backed layer-63 graph at a 131-token, three-page context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-resident-model SNAPSHOT` | Directly measure every complete 64-layer plus LM-head graph at a 131-token context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-text-endpoint SNAPSHOT` | Measure every source-backed final-norm plus LM-head graph | terminal or `--json PATH` |
| `cargo run -p xtask -- perf smoke` | Three-sample harness and environment smoke test for every suite | `target/benchmarks/perf-smoke/*.json` |
| `cargo run -p xtask -- perf leaf` | Full registered leaf timing and memory reports | `target/benchmarks/perf-leaf/*.json` |
| `cargo run -p xtask -- perf energy` | Full leaf reports plus a sustained power window per route | `target/benchmarks/perf-energy/*.json` |
| `cargo run -p xtask -- perf gate` | Run every oracle, measure every suite, and compare checked baselines | `target/benchmarks/perf-gate/*.json` |
| `cargo run -p xtask -- perf bless SUITE` | Run one oracle and explicitly replace that suite's baseline | `qual/baselines/SUITE-sm120.json` |

Host text paths use Criterion with the real snapshot loaded once outside measurement:

```bash
TUISKO_SNAPSHOT=/path/to/snapshots/16b6615af3548b88e2d8e382457bc705b00479cf \
  cargo bench --package tuisko-frontend --bench text
```

This measures chat-template rendering, short and long prompt encoding, disabled/partial/identical
prompt-cache routes, batched decoding, and streaming decoding. Criterion output remains under
ignored `target/criterion`; it is diagnostic until a checked host-baseline comparator is added.

The complete BF16 vocabulary sampling routes are host-only too:

```bash
cargo bench --package tuisko-engine --bench sampling
```

This measures greedy and checkpoint-default top-k-20/top-p-0.95 selection over all 248,320 logits.

The composed prompt-preparation and one-row generation-control paths use the pinned snapshot:

```bash
TUISKO_SNAPSHOT=/path/to/snapshots/16b6615af3548b88e2d8e382457bc705b00479cf \
  cargo bench --package tuisko-engine --bench generation
```

`perf gate` cannot run before each suite has an explicit baseline. A baseline update is a reviewed
source change; blessing one suite at a time keeps that diff independent, and the command never
commits it.

The leaf executable can also be controlled directly through `xtask`:

```bash
cargo run -p xtask -- bench-residual-norm \
  --samples 40 \
  --launches-per-sample 256 \
  --json target/benchmarks/residual-norm.json
```

Use `cargo run -p xtask -- bench-fp8-qkv`, `bench-fp8-gdn-input`, `bench-fp8-lm-head`,
`bench-fp8-swiglu`, `bench-fp8-down`, `bench-gdn-prepare`, `bench-gdn-recurrence`, or
`bench-gdn-output`, `bench-nvfp4-swiglu`, `bench-nvfp4-down`, `bench-attention-qk-prepare`,
`bench-paged-gqa`, or `bench-attention-output` with the same options for one operator suite only.

`bench-text-endpoint SNAPSHOT` accepts the same options. It is intentionally separate from the
leaf-wide `perf` commands until its first reviewed baseline is blessed.

`bench-dense-fp8-mlp SNAPSHOT` measures the complete source-backed layer-60 MLP graph with the same
options. It stays outside leaf-wide `perf` until the source-backed route receives a reviewed
baseline.

`bench-nvfp4-mlp SNAPSHOT` directly measures the complete source-backed layer-55 MLP graph. Its
`B=1,5..8` routes include production E2M1 activation quantization and W4A4 gate/up projection;
`B=2..4` preserve the BF16 gate/up activation, while every route uses the represented-weight A16
down projection. It remains outside leaf-wide `perf` until a locked-clock local baseline is
reviewed.

`bench-dense-fp8-gdn-layer SNAPSHOT` measures the complete stateful layer-60 graph. Repeated samples
advance its persistent history and FP32 recurrence exactly as serial decode rounds do; setup and
allocation remain outside the timed region.

`bench-full-attention-layer SNAPSHOT` measures the complete layer-63 decode graph directly. Its
131-token warm cache crosses both 64-token page seams; repeated paths overwrite the same admitted
cache position so the timed geometry stays invariant.

`bench-resident-model SNAPSHOT` times the complete production graph directly; it never derives a
model latency from leaf medians. The current 131-token route exercises all 64 layers and the LM
head with resident weights, shared workspace, recurrent state, and represented KV caches. It omits
the repeated-operation graph because one complete graph is already long enough for CUDA-event
resolution and duplicating hundreds of model nodes would measure a different owner. The production
embedding-staging graph restores represented input rows before each sample and remains outside the
timed whole-model replay.

Add `--energy-seconds 2` for sustained energy sampling. At least three samples, one launch per
sample, and a two-second energy window are required.

## What one timing means

Each exact route reports three boundaries and short routes also report a fourth:

| Measurement | Boundary |
|---|---|
| `host_submit` | Rust time spent submitting repeated CUDA Graph replays |
| `host_completion` | Rust time from submission through device completion |
| `device_graph` | CUDA-event time per production graph replay |
| `device_path` | CUDA-event time per complete operation inside one graph containing many repetitions |

`device_graph` is the production graph-replay cost. Optional `device_path` reduces CUDA-event timer
quantization for a short operation; it is not a different production route. A residual-norm path is
one kernel. An FP8 projection path is its production quantization and projection pair. The text
endpoint path is final RMSNorm followed by dynamic activation quantization and the LM-head
projection over the production graph's stable addresses.

The reusable timer records two events around a repeated interval and synchronizes once after it.
It does not insert events into or mutate the production graph, and its fixed boundary cost is
amortized across the reported operation count. The production server does not call the benchmark
timer.

Every metric records median, p10, p90, operations per measured interval, logical bytes per
operation, and logical GiB/s for device timings. Logical GiB/s uses the operation's minimum declared
reads and writes. It is not an `ncu` measurement of physical DRAM traffic.

Cases are measured in rotating and reversing order so a fixed route sequence cannot absorb clock or
thermal drift. All exact routes share one context, stream, prepared operation, and address-stable
arena for the complete session.

## Workload identity

A timing key is more than `(route, shape)`. The report and baseline bind each metric to:

- scope: `operator`, `endpoint`, `layer`, `model`, or `server`;
- phase: `startup`, `prefill`, `decode`, `mtp`, or `request`;
- compiled batch size;
- external concurrency;
- active, prompt, context, and requested output token counts;
- device-cache regime;
- prefix-cache regime; and
- execution mode: eager, CUDA Graph, or server.

Unused dimensions are `null`, not zero. A comparison refuses when any workload dimension or metric
inventory differs. This prevents, for example, a warm short-context operator result from being
compared with a cold long-context model result that shares a route name.

The residual-norm, FP8-QKV `B=1..8`, FP8-GDN-input, FP8-LM-head, dense-FP8-SwiGLU `B=1..8`,
dense-FP8-down, NVFP4-SwiGLU, and NVFP4-down cases are `operator/decode`, warm-cache, CUDA-Graph
workloads. They set batch and active tokens to the exact batch. FP8-QKV `T=16` is an
`operator/mtp` case.
Dense-FP8-SwiGLU `T=32,64,128` cases are `operator/prefill` cases with prompt and context lengths
equal to the active rows. Concurrency, output, and prefix cache do not apply to these leaf suites.

## Memory and capacity

Memory is reported from two independent views:

1. production owners attribute exact address-stable byte counts; and
2. NVML and `/proc/self/status` observe the whole process and device.

The report captures snapshots at `before_context`, `after_setup`, `after_warmup`, and
`after_measurement`. Each snapshot contains:

- whole-device used bytes;
- whole-device free bytes;
- driver-reserved device bytes;
- current process RSS; and
- peak process RSS.

It also emits comparable memory metrics:

| Metric | Meaning | Direction |
|---|---|---|
| named owner | Exact bytes registered by one production owner | at most |
| `summary/accounted_resident` | Sum of registered owner bytes | at most |
| `summary/setup_device_delta` | NVML used-memory increase from preflight through warm setup | at most |
| `summary/timed_peak_device_delta` | Highest in-window NVML usage relative to preflight | at most |
| `summary/timed_growth_after_warmup` | Highest in-window NVML usage above warmed setup | at most |
| `summary/minimum_device_headroom` | Lowest in-window `memory.free` value | at least |
| `summary/device_reserved` | NVML `memory.reserved` after warmup | at most |
| `summary/process_peak_rss` | Process `VmHWM` | at most |
| `summary/unattributed_setup_delta` | Setup delta not explained by registered owners | at most |

Free headroom is sampled from `memory.free` directly. Do not derive it as `total - used`: the driver
reports reserved framebuffer memory separately, and on the target card that difference is hundreds
of MiB. NVML memory observations have MiB resolution; owner accounting remains the exact authority
for allocations the program controls.

The residual-norm suite attributes its single address-stable arena. Each FP8 projection suite,
including the 1.27 GB full-vocabulary head, separately attributes source-native weights and the
remainder of its single address-stable arena as workspace. CUDA context, module, and graph storage
remain visible in the setup delta and unattributed remainder because their exact allocation sizes
are not owned by the program.

Owned quantities and post-warmup growth are enforced by default with zero slack. Setup delta,
timed peak relative to preflight, headroom, driver reservation, RSS, and unattributed setup remain
in the checked artifact but default to informational: those observations include volatile driver
and CUDA-context state. A later model or server suite may deliberately enforce a product memory
budget, such as minimum headroom, by changing that metric in a baseline-only review.

Future resident owners should register weights, KV cache, workspaces, graph storage, and runtime
storage separately when their exact byte counts are known. A missing attribution must remain visible
as unattributed memory rather than being assigned to a convenient category.

### Host/device transfers

The arena's synchronous `copy_from_host` and `copy_to_host` helpers currently serve setup and oracle
readback, so they are not performance cases. Benchmark transfers when a production owner depends on
them, using its actual pinned staging buffer, direction, payload sizes, and overlap policy. Such a
case should record the warmed negotiated PCIe generation and width alongside effective GiB/s;
resident compute cases such as residual norm do not depend on PCIe link state.

## Power and energy

The telemetry sampler records instantaneous whole-board power during timed work. `perf energy`
first samples the synchronized, warmed device at idle, then creates a separate sustained window of
at least two seconds for every exact route. It reports whole-board, above-idle, and reciprocal
efficiency estimates:

```text
board joules per unit = mean board watts × CUDA-event seconds / completed logical units
dynamic joules per unit = max(mean board watts - idle board watts, 0)
                         × CUDA-event seconds / completed logical units
units per board joule = 1 / board joules per unit
```

The current unit is a token/active row. Each leaf estimate comes from continuous graph replay, not
one power sample around a microsecond operation, but it is still a whole-board diagnostic dominated
by clock, static, dispatch, and runtime power. It is not energy physically attributable to one leaf
and is not a blessed regression metric. Use energy most confidently for sustained full graphs or
serving workloads under the same machine and environment.

Future model reports must distinguish joules per prompt token, generated token, and MTP committed
token. Those denominators describe different work and cannot share a baseline.

## Environment controls and refusal

Before measurement, the runner requires:

- device zero is exactly `NVIDIA GeForce RTX 5090`;
- `CUDA_VISIBLE_DEVICES` is unset or exactly `0`;
- GPU utilization is zero;
- at most 1,024 MiB is already used;
- no foreign compute PID is present; and
- the runtime compute capability is 12.0.

During measurement it samples SM clock, memory clock, temperature, board power, used memory, and
free memory every 10 ms. A run is refused when:

- fewer than three telemetry samples are captured;
- the SM clock spreads by more than 50 MHz, admitting the target's measured 2,160-to-2,197 MHz
  light-load P-state step;
- the memory clock spreads by more than 250 MHz, admitting the target's measured 13,801-to-14,001
  MHz loaded P-state step;
- a timing or derived throughput is non-finite or non-positive; or
- another compute process appears.

The runner records clocks; it does not change clock or power settings. Clock locking, if used, is a
machine-administration step outside the repository and must remain identical between baseline and
candidate runs.

Multi-suite `perf` commands wait up to ten seconds between their own benchmark processes for NVML's
utilization sample to return idle. Direct single-suite commands retain immediate refusal behavior.

For the retained 2,200 MHz target clock and the card's 14,001 MHz memory clock, lock both before the
run:

```bash
sudo nvidia-smi -i 0 --lock-gpu-clocks=2200,2200 && \
sudo nvidia-smi -i 0 --lock-memory-clocks=14001,14001
```

Reset the administrator-controlled clock state afterward:

```bash
sudo nvidia-smi -i 0 --reset-gpu-clocks && \
sudo nvidia-smi -i 0 --reset-memory-clocks
```

The runner prints these complete commands when it refuses a drifting-clock result. It does not run
them automatically because changing device clocks requires explicit machine authority.

## Baselines and regression gates

The checked baseline stores one record per timing and memory key. Comparison additionally requires
the same:

- suite and complete metric inventories;
- target GPU name, driver, compute capability, and memory capacity;
- cuda-oxide generator/resource stamp;
- SM and memory clock bands;
- minimum sample count;
- timing and power scopes.

The generator stamp records readable cuda-oxide, Rust, and CUDA Toolkit identities. Both `ptxas`
and `cuobjdump` must report the same exact Toolkit release and patch version so a mixed installation
is refused. Locally linked compiler-backend bytes are not a portable equality gate.

The raw report records the physical UUID so CUDA execution, NVML telemetry, and memory snapshots can
be proven to refer to the same device during one run. The checked baseline neither stores nor
compares that UUID: it describes the declared target class rather than identifying one physical
runner. It retains the blessed executable hash as provenance, not as a candidate comparison key.

New device timing metrics default to the larger of 5% or 0.05 us of upper slack. Host metrics
default to the larger of 15% or 0.10 us and are informational unless deliberately enabled. Exact
owned-memory and post-warmup-growth quantities default to zero slack. Volatile NVML and RSS
quantities initially receive 16 MiB of absolute slack and remain informational unless a reviewed
product budget enables them. Each value, tolerance, direction, and enforcement flag is visible in
the baseline diff and can be reviewed independently.

`perf bless SUITE` retains reviewed tolerances and enforcement flags for that suite's existing keys.
It updates measured references but does not silently loosen a gate. Adding or removing a workload
or metric changes the inventory and therefore requires an explicit baseline review.

Do not bless a regression merely to make the gate green. First establish whether the change is an
intentional schedule improvement, a measurement-environment change, or a real regression. Timings
from different clocks, drivers, devices, or cache regimes are not comparable.

## Correctness and performance order

Performance does not admit a kernel. The required order is:

1. independent represented-value or mathematical oracle;
2. eager-versus-graph replay equivalence;
3. generated entry and resource gate;
4. exclusive performance measurement; and
5. explicit baseline comparison or blessing.

`perf gate` mechanizes this order for registered suites. A skipped numerical oracle is not a pass.

Plain `cargo test --workspace` cannot link the SM120 qualification crate because its cuda-oxide
artifact anchor is produced by the device build. Use the `xtask` qualification command for device
tests. Ordinary host tests and Criterion benchmarks remain normal Cargo commands.

## Adding another benchmark case

A new production owner should extend the existing runner rather than create an unrelated timing
loop:

1. prepare one context, stream, operation owner, and address-stable arena for the suite;
2. capture its production leaf/full graph and, only for timer resolution, an optional
   repeated-operation graph;
3. register an exact workload identity and logical-byte/unit accounting;
4. register every exactly known resident allocation with its owner and scaling rule;
5. warm or displace caches according to the declared regime;
6. keep numerical qualification independent from production helpers;
7. add the exact route to the expected inventory; and
8. bless a baseline only after the correctness and resource gates pass.

Time composed layers, the whole model, and server requests directly. Do not estimate their wall time
by summing leaf medians: graph concurrency, cache reuse, scheduling, and host work can make that sum
wrong in either direction.

Criterion remains the harness for pure host work such as checkpoint admission, tokenization,
template rendering, prefix lookup, sampling, and streaming detokenization. CUDA-event timing,
exclusive-device controls, NVML telemetry, and device baselines remain in this runner.

## Current limitations

- Decode operator coverage is exact `B=1..8`; prefill remains limited to the explicitly listed
  dense-FP8 SwiGLU routes, and long-context attention needs separate routes.
- The suite labels warm cache; it does not yet implement a generic cold-cache displacement protocol.
- There is no TTFT, inter-token-latency, concurrency, long-context, or end-to-end MTP benchmark in
  this repository yet.
- Power and energy are reportable for the resident model; full-server energy remains future work.
- `ncu` and Nsight Systems traces are diagnostic artifacts, not produced by the regression runner.

These are scope statements, not inferred passes. New cases should be added with their production
owners and independent oracles.

## Troubleshooting

`device zero is not idle`
: Stop or wait for the foreign GPU workload. Do not bypass the exclusivity check.

`foreign compute process IDs`
: Another CUDA process appeared after setup. Re-run under an exclusive reservation.

`SM clock moved ... lock clocks before comparing`
: Cool or stabilize the card, run the complete lock command printed by the refusal, and repeat under
  the same clock policy as the baseline. Run the printed reset command when finished.

`performance baseline ... could not read`
: No reviewed baseline exists. Run `perf leaf`, inspect the report, then use
  `perf bless SUITE` explicitly for one of the registered suites in the command reference.

`performance report and baseline metric inventories differ`
: A route, workload dimension, timing boundary, or memory owner changed. Review the inventory change;
  do not edit the generated report.

`cuda_oxide_artifact_anchor` is undefined
: A device crate was linked through plain Cargo. Use `cargo run -p xtask -- build-sm120` or the
  qualification/performance commands above.

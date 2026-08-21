# Performance and capacity qualification

TuiskoLLM uses a custom device runner for GPU measurements and Criterion for pure host work. GPU
results are valid only on the exact RTX 5090 target under an exclusive, recorded environment.

The currently registered device suite covers the zero-centered residual/RMSNorm leaf at every exact
`B=1..8` route. The report schema is already shaped for future layer, whole-model, and serving cases,
but those measurements do not exist until their production owners land.

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
target/benchmarks/perf-smoke.json
```

Every performance command also executes the release SM120 build and checks the PTX/SASS entry and
resource inventory before launching the benchmark.

## Command reference

| Command | Purpose | Output |
|---|---|---|
| `cargo run -p xtask -- build-sm120` | Build the release device artifact and check entries, registers, stack, local, and shared bytes | terminal |
| `cargo run -p xtask -- qualify-residual-norm` | Run the independent numerical and graph-replay oracle | terminal |
| `cargo run -p xtask -- perf smoke` | Three-sample harness and environment smoke test | `target/benchmarks/perf-smoke.json` |
| `cargo run -p xtask -- perf leaf` | Full registered leaf timing and memory report | `target/benchmarks/perf-leaf.json` |
| `cargo run -p xtask -- perf energy` | Full leaf report plus a sustained power window per route | `target/benchmarks/perf-energy.json` |
| `cargo run -p xtask -- perf gate` | Run correctness, measure, and compare with the checked baseline | `target/benchmarks/perf-gate.json` |
| `cargo run -p xtask -- perf bless` | Run correctness and explicitly replace the checked baseline | `qual/baselines/sm120-qwen38.json` |

`perf gate` cannot run before the first explicit `perf bless`. A baseline update is a reviewed source
change; the command never commits it.

The leaf executable can also be controlled directly through `xtask`:

```bash
cargo run -p xtask -- bench-residual-norm \
  --samples 40 \
  --launches-per-sample 256 \
  --json target/benchmarks/residual-norm.json
```

Add `--energy-seconds 2` for sustained energy sampling. At least three samples, one launch per
sample, and a two-second energy window are required.

## What one timing means

Each exact route reports four boundaries:

| Measurement | Boundary |
|---|---|
| `host_submit` | Rust time spent submitting repeated CUDA Graph replays |
| `host_completion` | Rust time from submission through device completion |
| `device_graph` | CUDA-event time per production graph replay; currently a one-node leaf graph |
| `device_node` | CUDA-event time per leaf node inside one graph containing many repeated nodes |

`device_graph` is the production graph-replay cost. `device_node` reduces CUDA-event timer
quantization for a short kernel; it is not a different production route.

The reusable timer records two events around a repeated interval and synchronizes once after it.
It does not insert events into or mutate the production graph, and its fixed boundary cost is
amortized across the reported operation count. The production server does not call the benchmark
timer.

Every metric records median, p10, p90, operations per measured interval, logical bytes per
operation, and logical GiB/s for device timings. Logical GiB/s uses the operation's minimum declared
reads and writes. It is not an `ncu` measurement of physical DRAM traffic.

Cases are measured in rotating and reversing order so a fixed `B=1..8` sequence cannot absorb clock
or thermal drift. All exact routes share one context, stream, prepared operation, and address-stable
arena for the complete session.

## Workload identity

A timing key is more than `(route, shape)`. The report and baseline bind each metric to:

- scope: `operator`, `layer`, `model`, or `server`;
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

The current residual-norm suite is an `operator/decode`, warm-cache, CUDA-Graph workload. It sets
batch and active tokens to each exact `B=1..8`; context, prompt, concurrency, output, and prefix
cache do not apply.

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

The residual-norm suite currently attributes its single address-stable arena. CUDA context, module,
and graph storage remain visible in the setup delta and unattributed remainder because their exact
allocation sizes are not owned by the program.

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
one power sample around a microsecond kernel, but it is still a whole-board diagnostic dominated by
clock, static, dispatch, and runtime power. It is not energy physically attributable to RMSNorm and
is not a blessed regression metric. Use energy most confidently for sustained full graphs or
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
- the SM clock spreads by more than 30 MHz;
- the memory clock spreads by more than 100 MHz;
- a timing or derived throughput is non-finite or non-positive; or
- another compute process appears.

The runner records clocks; it does not change clock or power settings. Clock locking, if used, is a
machine-administration step outside the repository and must remain identical between baseline and
candidate runs.

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

`perf bless` retains reviewed tolerances and enforcement flags for existing keys. It updates measured
references but does not silently loosen a gate. Adding or removing a workload or metric changes the
inventory and therefore requires an explicit baseline review.

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
2. capture its production leaf/full graph and, only for timer resolution, an optional repeated-node
   graph;
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

- Only residual/RMSNorm `B=1..8` leaf cases are registered.
- The suite labels warm cache; it does not yet implement a generic cold-cache displacement protocol.
- There is no whole-layer, whole-model, TTFT, inter-token-latency, concurrency, long-context, prefix-
  cache, or end-to-end MTP benchmark in this repository yet.
- Power and energy are reportable, but energy is most meaningful after sustained model/server cases
  land.
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
: No reviewed baseline exists. Run `perf leaf`, inspect the report, then use `perf bless` explicitly.

`performance report and baseline metric inventories differ`
: A route, workload dimension, timing boundary, or memory owner changed. Review the inventory change;
  do not edit the generated report.

`cuda_oxide_artifact_anchor` is undefined
: A device crate was linked through plain Cargo. Use `cargo run -p xtask -- build-sm120` or the
  qualification/performance commands above.

# Performance qualification

This document defines how TuiskoLLM measures and accepts performance. Product-level results live
in [`README.md`](../README.md); checked SM120 references live in
[`qual/baselines/`](../qual/baselines/).

Only an exclusive RTX 5090 run with a comparable environment can create performance authority.
SM89, SM86, uncontrolled-clock, subset, profiler, and remote timings are diagnostic even when their
numerical and resource gates pass.

## Commands

Run device commands from the repository root. Bootstrap the pinned cuda-oxide toolchain once:

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
```

The main workflows are:

| Command | Use |
| --- | --- |
| `cargo run -p xtask -- build-sm120` | Build the release device artifact and check its entry, launch-bound, register, stack, local-memory, shared-memory, PTX, and SASS contracts. |
| `cargo run -p xtask -- qualify-<suite> [SNAPSHOT]` | Run the suite's independent numerical oracle, eager/graph checks, allocation checks, and benchmark-accounting tests. |
| `cargo run -p xtask -- bench-<suite> [SNAPSHOT] [options]` | Measure one complete suite. An explicit `--json` path must be repository-relative and under `target/`. |
| `cargo run -p xtask -- perf smoke` | Run every registered performance suite with three samples to validate the harness and environment. |
| `cargo run -p xtask -- perf leaf` | Measure the complete registered leaf inventory. |
| `cargo run -p xtask -- perf energy` | Measure the leaf inventory with a sustained energy window. |
| `cargo run -p xtask -- perf gate SNAPSHOT` | Qualify, measure, and compare every registered suite with its checked baseline. |
| `cargo run -p xtask -- perf candidate SUITE [SNAPSHOT] [options]` | Qualify the changed boundary and directly measure its dependency cone. Diagnostic. |
| `cargo run -p xtask -- perf check SUITE [SNAPSHOT]` | Requalify and compare the complete authoritative dependency cone. |
| `cargo run -p xtask -- perf bless SUITE [SNAPSHOT]` | Replace one suite's checked baseline after a reviewed run. It never commits the result. |
| `cargo run -p xtask -- perf iterate SUITE [SNAPSHOT] --batch B --hypothesis TEXT` | Run one exact-batch optimization iteration with reusable exact-input receipts. Diagnostic. |

The `xtask` command registry is the source of truth for suite names. Do not copy its evolving leaf,
layer, model, and target inventory into prose. Common complete boundaries include
`resident-model`, `resident-prefill`, `resident-long-context-model`, and `server`.

Device filters are part of qualification design: `qualify-<suite>` must select both the numerical
device test and the suite's benchmark-accounting tests. If a filter selects multiple device tests,
run them serially so their CUDA preflights do not race.

Pure host work uses Criterion. For example:

```bash
TUISKO_SNAPSHOT=/path/to/snapshot cargo bench -p tuisko-frontend --bench text
cargo bench -p tuisko-engine --bench sampling
TUISKO_SNAPSHOT=/path/to/snapshot cargo bench -p tuisko-engine --bench generation
```

Criterion output under `target/criterion/` is diagnostic until a checked host comparator exists.

## Required order

Performance never admits an implementation. An exact device route must pass, in order:

1. an independent represented-value or mathematical oracle;
2. eager and CUDA Graph replay agreement;
3. complete generated-entry and resource checks;
4. exclusive-device measurement of the production boundary; and
5. explicit baseline comparison or blessing.

A skipped oracle is not a pass. Run device preflight before opening a large checkpoint. Do not
change or bless a baseline to hide a correctness, resource, or performance regression.

## What is timed

Every benchmark uses the production operation, allocations, stream, graph, cache regime, shapes,
and address-stable workspace. Setup, allocation, snapshot loading, and fixture restoration stay
outside the timed interval unless they are part of the production boundary being measured.

The runner reports:

| Measurement | Boundary |
| --- | --- |
| `host_submit` | Rust time spent submitting production CUDA Graph replays. |
| `host_completion` | Rust time from submission through device completion. |
| `device_graph` | CUDA-event time per production graph replay. This is the primary device boundary. |
| `device_path` | CUDA-event time per operation in an optional repeated-operation graph, used only to resolve very short paths. |

The reusable GPU timer brackets repeated work with two CUDA events and synchronizes after the
interval. It does not mutate the production graph. Reports include median, p10, p90, operation
count, declared logical bytes, and logical GiB/s. Logical GiB/s is derived from minimum semantic
reads and writes; it is not physical DRAM traffic from a profiler.

Cases rotate and reverse order to avoid assigning fixed thermal or clock drift to one route. The
default budgets match the duration of the timed boundary:

| Boundary | Samples | Replays per sample | Warmup replays |
| --- | ---: | ---: | ---: |
| Short operator graph | 40 | 256 | 1,024 |
| Long/composed graph | 40 | 32 | 128 |
| Resident model graph | 40 | 1 | 16 |

These counts are part of measurement identity. A resident graph must not inherit a microsecond
leaf's replay count.

Stateful cases restore production input outside every measured replay when an output aliases the
next input. Inactive-tail checks begin after the widest surviving writer, not automatically after
the final writer.

Composed layers, resident models, and HTTP requests are timed directly. Never estimate their wall
time by adding leaf medians: graph scheduling, cache reuse, memory pressure, and host work make the
sum invalid in either direction.

## Measurement identity

A comparable report binds all of the following:

- suite, route, metric inventory, scope, and phase;
- exact checkpoint and revision when source-backed;
- batch, active/prompt/context/output tokens, external concurrency, and execution mode;
- device-cache and prefix-cache regimes;
- operation count, sample count, warmup count, replay count, timing scope, and power scope;
- GPU name, compute capability, framebuffer capacity, driver, and controlled clock bands;
- cuda-oxide generator/resource provenance and CUDA toolchain identity; and
- exact owned-memory metrics and their scaling rules.

Unused workload dimensions are `null`, not zero. Comparison refuses when identities or inventories
differ. An exact `--batch B` run records `case_policy: diagnostic_subset`; it cannot be compared or
blessed as the complete admitted `B=1..8` inventory.

## Device control and refusal

The local SM120 runner requires:

- device zero to be exactly `NVIDIA GeForce RTX 5090`, compute capability 12.0;
- `CUDA_VISIBLE_DEVICES` to be unset or exactly `0`;
- no foreign compute process;
- utilization settled below 10%; and
- at most 2,048 MiB already used before the run.

Post-build and benchmark preflights wait up to 60 seconds for process-free desktop utilization to
settle below that threshold. The memory, process-count, and clock requirements remain unchanged.

The runner records clocks and power but never changes them. After owner warmup it drives the
production graph for at least two seconds and validates loaded clocks before the long timing
matrix. It continues sampling clocks, temperature, power, and memory every 10 ms during the run.
SM120 leaf/model runs admit at most 50 MHz SM-clock spread and 250 MHz memory-clock spread; the
complete HTTP boundary admits 75 MHz and 250 MHz respectively.

Locking clocks is an explicit machine-administration action:

```bash
sudo nvidia-smi -i 0 --lock-gpu-clocks=2200,2200 \
  && sudo nvidia-smi -i 0 --lock-memory-clocks=14001,14001
```

Reset them afterward:

```bash
sudo nvidia-smi -i 0 --reset-gpu-clocks \
  && sudo nvidia-smi -i 0 --reset-memory-clocks
```

The run refuses a wrong or occupied device, a foreign process, insufficient telemetry, invalid
timings, or incomparable clocks. If the loaded-clock probe fails, timing stops early. If clocks
drift only after the probe, completed medians are retained as diagnostic evidence before refusal.

For an intentionally exploratory run:

```bash
TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT=1 cargo run -p xtask -- bench-<suite> ...
```

Its report records `clock_policy: diagnostic_uncontrolled` and can never be blessed. Remote GPU
timings are subject to the same non-authoritative rule; remote qualification can satisfy numerical
and resource gates, but not a performance baseline.

## Memory and capacity

Reports combine two independent views:

1. exact byte attribution from production owners; and
2. whole-process observations from NVML and `/proc/self/status`.

Snapshots are taken before context creation, after setup, after warmup, and after measurement.
They record device used/free/reserved memory and current/peak process RSS. Exact owner allocation
and post-warmup growth default to zero slack. Volatile CUDA-context, driver, NVML, and RSS metrics
remain visible but informational unless a reviewed product budget enables them.

Use NVML's reported free memory for headroom; do not derive it as total minus used because the
driver reports reserved framebuffer memory separately. Missing ownership remains unattributed—it
must not be assigned to a convenient category.

Capacity gates are correctness gates. They must exercise the real shared-pool, slot, reclaim, and
retry behavior at the admitted boundary; a benchmark's healthy headroom is not proof of capacity.

## Energy

`perf energy` measures a synchronized idle reference, then continuously replays each warmed route
for at least two seconds while sampling whole-board power. It reports board joules per logical unit,
above-idle joules per unit, and units per board joule. The unit must state what completed work means,
such as an active row, prompt token, generated token, or committed MTP token.

This is whole-board evidence, not energy physically attributable to one leaf. It is most useful for
sustained resident or server boundaries under an identical environment and remains diagnostic
unless an explicit checked energy baseline is introduced.

## Baselines

Checked timing baselines are `qual/baselines/<suite>-sm120.json`. Resource authorities remain
separate from timing baselines. Raw reports, profiler output, receipts, and optimization history
belong under ignored `target/` paths.

A report is blessable only when it has:

- the complete admitted case and metric inventory;
- controlled, comparable clocks on the exact local target;
- matching device, driver, generator, toolchain, sampling, workload, and memory identity; and
- passing numerical and resource gates.

`perf bless` retains reviewed tolerances and enforcement flags for existing keys; it does not
silently loosen them. New device timings initially use the larger of 5% or 0.05 us upper slack.
Host timings initially use the larger of 15% or 0.10 us and are informational. Exact owned memory
and post-warmup growth use zero slack; volatile observations begin informational with 16 MiB slack.

Adding or removing a case or metric is an inventory change and requires review. Baseline changes
are separate source commits. Diagnose an environment change or regression before blessing; never
bless merely to make a gate green.

## Optimization loop

Change one measured hypothesis per iteration. Start with an exact route only for fast diagnosis,
then qualify the changed numerical/resource boundary and directly time every affected boundary in
its leaf-to-owner-to-resident/server dependency cone. `perf candidate` and `perf check` use the
checked cone registry for this relationship.

A leaf win is not a model win. Before accepting a change, repeat the cone with complete suite
defaults and every admitted exact route. Keep each composed boundary's baseline independent from
leaf resource and timing baselines. Revert a rejected implementation before testing the next
hypothesis.

`perf iterate` records the hypothesis, Git/input identity, reports, refusals, and phase wall times
under `target/optimization/`. It can reuse ignored qualification and build receipts only when their
complete source, device, driver, toolchain, resource, executable, and PTX identities match. It may
copy verified immutable artifacts from another registered worktree; it never shares a mutable
Cargo target directory.

Direct `bench-*` and profiling commands use the same exact-input build receipt. A matching receipt
proves that the complete SM120 resource audit already passed for the verified executable and PTX;
these commands therefore proceed to their device preflight without rerunning that unrelated audit.
`build-sm120` remains the explicit command that reruns the complete product resource audit.

## Profiling

Profile only the production owner or CUDA Graph after allocation and warmup. Nsight Systems is the
first step:

```bash
cargo run -p xtask -- profile resident-model SNAPSHOT \
  --batch 1 --replays 3 --tool nsys
```

The command requires the observed kernel sequence to match the exact semantic owner/stage manifest
and closes complete graph span against kernel time and gaps. Reports, SQLite exports, graph
inventories, semantic manifests, and timing CSVs are written under `target/profiles/`.

Use Nsight Compute only for a kernel family selected from that graph-level evidence:

```bash
cargo run -p xtask -- profile resident-model SNAPSHOT \
  --batch 1 --replays 1 --tool ncu --kernel 'REGEX'
```

Profiler timings are perturbed diagnostics, never regression references. Occupancy and transaction
warnings select hypotheses; they do not prove an optimization. Production timing of the complete
affected boundary decides whether a change survives.

## Troubleshooting

`device zero is not idle` or `foreign compute process IDs`
: Wait for or stop the other GPU workload. Do not bypass exclusivity.

`SM clock moved ...`
: Stabilize or lock the card, then repeat under the baseline's clock policy. Use the diagnostic
  environment variable only when non-authoritative evidence is sufficient.

`performance baseline ... could not read`
: The suite has no reviewed authority. Produce and inspect a complete run, then invoke
  `perf bless SUITE` explicitly.

`performance report and baseline metric inventories differ`
: A workload, timing boundary, or memory owner changed. Review the inventory; do not hand-edit the
  generated report.

`cuda_oxide_artifact_anchor is undefined`
: A device crate was linked through plain Cargo. Use `build-sm120`, a qualification command, or a
  performance command so `xtask` finalizes the device artifact.

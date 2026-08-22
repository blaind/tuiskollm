# Remote qualification

`xtask remote` probes a selected GPU or runs a locally built qualification or benchmark executable
on a fresh secure RunPod pod. RTX 5090 remains the default and the only GPU with a complete kernel
inventory. RTX 4090 admits residual norm, represented-weight NVFP4 A16 feasibility routes, and FP8
QKV decode; RTX 3090 admits residual norm and NVFP4 gate/up. Other target/suite pairs reject before
provisioning.

Remote benchmark reports are diagnostic. They record GPU identity, driver, clocks, telemetry,
binary and resource hashes with `clock_policy: diagnostic_uncontrolled`. Baseline blessing rejects
that policy, so locked local runs remain the performance authority.

## Credentials

Set these in the environment or the nearest gitignored `.env`:

```text
RUNPOD_API_KEY=...
RUNPOD_SSH_KEY_FILE=/absolute/path/to/a/private/key
```

The private key must correspond to an SSH public key registered in the RunPod account. The runner
checks both credentials before compiling or creating a billable pod.

The Rust runner owns SSH, SFTP, and upload compression. The local machine only needs `strip`; the
selected pod image supplies its SSH server and `gzip` decoder.

## Commands

```sh
cargo run -p xtask --features remote -- remote check
cargo run -p xtask --features remote -- remote probe --gpu 4090
cargo run -p xtask --features remote -- remote probe --gpu 3090
cargo run -p xtask -- build-residual-norm --gpu 4090
cargo run -p xtask -- build-residual-norm --gpu 3090
cargo run -p xtask -- build-residual-bench --gpu 4090
cargo run -p xtask -- build-residual-bench --gpu 3090
cargo run -p xtask --features remote -- remote qualify-residual-norm
cargo run -p xtask --features remote -- remote qualify-residual-norm --gpu 4090
cargo run -p xtask --features remote -- remote qualify-residual-norm --gpu 3090
cargo run -p xtask --features remote -- remote qualify-nvfp4-swiglu --gpu 4090
cargo run -p xtask --features remote -- remote qualify-nvfp4-swiglu --gpu 3090
cargo run -p xtask --features remote -- remote qualify-nvfp4-down --gpu 4090
cargo run -p xtask --features remote -- remote qualify-fp8-qkv
cargo run -p xtask --features remote -- remote qualify-fp8-qkv --gpu 4090
cargo run -p xtask --features remote -- remote qualify-fp8-gdn-input
cargo run -p xtask --features remote -- remote qualify-fp8-lm-head
cargo run -p xtask --features remote -- remote bench-residual-norm
cargo run -p xtask --features remote -- remote bench-residual-norm --gpu 4090
cargo run -p xtask --features remote -- remote bench-residual-norm --gpu 3090
cargo run -p xtask --features remote -- remote bench-nvfp4-swiglu --gpu 4090
cargo run -p xtask --features remote -- remote bench-nvfp4-swiglu --gpu 3090
cargo run -p xtask --features remote -- remote bench-nvfp4-down --gpu 4090
cargo run -p xtask --features remote -- remote bench-fp8-qkv
cargo run -p xtask --features remote -- remote bench-fp8-qkv --gpu 4090
cargo run -p xtask --features remote -- remote bench-fp8-gdn-input
cargo run -p xtask --features remote -- remote bench-fp8-lm-head
cargo run -p xtask --features remote -- remote sweep
```

Qualification accepts `--max-minutes N`, `--image NAME`, and `--keep-on-fail`. The last option
retains a failed, billable pod until a sweep from the creating worktree or manual deletion.
`remote sweep` immediately deletes pod IDs recorded under the current worktree's `target/` and may
also delete another worktree's pod only after its encoded run budget plus five-minute cleanup grace
has expired. It does not infer staleness from the shared `tuiskollm-gate` prefix.

All provisioning commands accept `--gpu 5090|4090|3090`. The command/target decision table admits
only implemented inventories; remaining non-SM120 operators are separate qualification work.
`probe` validates the requested device name, compute capability,
userland, direct SSH, and cleanup without uploading a CUDA artifact. The runner does not fall back
to RunPod's proxy shell when a host fails to expose direct SSH.

Benchmarks also accept `--samples N`, `--launches-per-sample N`, and `--energy-seconds N`. RunPod
does not grant clock-control permission, so remote reports retain the complete observed clock range
without claiming comparability. The runner downloads `benchmark.out` plus `benchmark.json` under
`target/remote-reports/`.

The residual-norm benchmark sweeps both route families at every exact `B=1..8` in one process.
Non-SM120 NVFP4 benchmarks sweep their source-word-preserving A16 routes over the same exact
batches; SM89 includes gate/up and down while SM86 includes gate/up only. SM89 FP8 QKV also sweeps
its dynamic-quantize decode routes at exact `B=1..8`; the Blackwell-only T=16 route is not part of
the partial Ada inventory. Retune only the selected architecture crate, rerun its numerical gate,
then compare diagnostic JSON from the same GPU target. SM89 and SM86 do not have blessed clock
profiles yet, so their reports cannot become checked performance baselines until controlled-clock
evidence is established.

Each run:

1. checks credentials and builds the selected architecture artifact locally;
2. verifies static resource gates and prepares the selected executable;
3. creates one secure pod for the selected exact GPU through the RunPod v2 API;
4. starts a detached cleanup watchdog;
5. connects to the API-provided direct SSH endpoint through the Rust client, uploads through SFTP,
   and runs through SSH exec under a deadline;
6. verifies the exact GPU and userland and executes the selected route;
7. writes reports under `target/remote-reports/`; and
8. deletes the pod and verifies its absence through the API.

The client accepts the API-provided endpoint's first host key for the single pod session and rejects
any key change during that session.

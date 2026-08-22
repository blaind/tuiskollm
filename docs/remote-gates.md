# Remote qualification

`xtask remote` probes a selected GPU or runs a locally built qualification or benchmark executable
on a fresh secure RunPod pod. RTX 5090 remains the default and the only GPU with a complete kernel
inventory. RTX 4090 and RTX 3090 have separate residual-norm artifacts; the other suites reject
those targets before provisioning.

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
cargo run -p xtask --features remote -- remote qualify-residual-norm
cargo run -p xtask --features remote -- remote qualify-residual-norm --gpu 4090
cargo run -p xtask --features remote -- remote qualify-residual-norm --gpu 3090
cargo run -p xtask --features remote -- remote qualify-fp8-qkv
cargo run -p xtask --features remote -- remote qualify-fp8-gdn-input
cargo run -p xtask --features remote -- remote qualify-fp8-lm-head
cargo run -p xtask --features remote -- remote bench-residual-norm
cargo run -p xtask --features remote -- remote bench-fp8-qkv
cargo run -p xtask --features remote -- remote bench-fp8-gdn-input
cargo run -p xtask --features remote -- remote bench-fp8-lm-head
cargo run -p xtask --features remote -- remote sweep
```

Qualification accepts `--max-minutes N`, `--image NAME`, and `--keep-on-fail`. The last option
retains a failed, billable pod until `remote sweep` or manual deletion.

All provisioning commands accept `--gpu 5090|4090|3090`. Non-SM120 targets currently admit only
`qualify-residual-norm`; their benchmark loop and remaining operator inventories are separate
qualification work. `probe` validates the requested device name, compute capability,
userland, direct SSH, and cleanup without uploading a CUDA artifact. The runner does not fall back
to RunPod's proxy shell when a host fails to expose direct SSH.

Benchmarks also accept `--samples N`, `--launches-per-sample N`, and `--energy-seconds N`. RunPod
does not grant clock-control permission, so remote reports retain the complete observed clock range
without claiming comparability. The runner downloads `benchmark.out` plus `benchmark.json` under
`target/remote-reports/`.

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

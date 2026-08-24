# Performance and capacity qualification

TuiskoLLM uses a custom device runner for GPU measurements and Criterion for pure host work. GPU
results are valid only on the exact RTX 5090 target under an exclusive, recorded environment.

The available device suites cover zero-centered residual/RMSNorm at exact `B=1..8` and
`T=32,64,128,1024`, and
dynamically-quantized FP8 QKV at exact `B=1..8`, `T=16` MTP, and `T=32,64,128,1024` prefill
widths, GDN Q/K/V/Z input projection at exact `B=1..8` and `T=32,64,128,1024`, the
full-vocabulary FP8 LM head at exact `B=1..8`,
dense-FP8 gate/up SwiGLU at exact
`B=1..8` and `T=32,64,128,1024`, dense-FP8 down at exact `B=1..8` and `T=32,64,128,1024`, GDN
control/convolution and recurrence at exact `B=1..8` and `T=32,64,128,1024`, and the GDN
source-native output projection at exact `B=1..8`, plus the source-backed
dense-FP8 MLP at exact `B=1..8` and `T=32,64,128,1024`, complete layer-60 GDN, and final-norm plus
LM-head owners. NVFP4 gate/up SwiGLU
uses the exact retained A16 and W4A4 decode schedules at `B=1..8` plus W4A4 prefill at
`T=32,64,128,1024`; NVFP4 down projection consumes the represented E2M1/E4M3 source planes
through exact A16 routes at `B=1..8` and native W4A4 prefill at `T=32,64,128,1024`.
Full-attention Q/K
preparation covers zero-centered normalization, the 64-wide three-axis MRoPE, and represented E4M3
KV-cache append at exact `B=1..8` and `T=32,64,128,1024`. Short-context paged GQA covers exact
24-query/4-KV-head, 256-wide online-softmax decode across page boundaries at `B=1..8` plus causal
shared-cache prefill tails at `T=32,64,128`. Each prefill CTA stages one 64-position represented
E4M3 K/V tile once for two adjacent tokens and their twelve grouped-query warps. Deep `T=128`
prefill tails use 256-thread FP8/F16 flash CTAs over 32 query rows and one query head. P8 uses
64-position tiles for contexts 129 through 32,768; P16 uses 32-position tiles through 220,000 so
its 43,520-byte single buffer preserves two-CTA residency. Both publish complete FP32
maximum/denominator/numerator states and reduce them into the public output seam. Long-context
`T=1024` macro prefill uses the two-CTA K32 flash producer at exact `P=1,2,4,8,16`, with one
compile-time reducer per partition count; the intended resident route is P4. Long-context paged
GQA retains the same represented cache contract and partitions contexts through 220,000 positions
into 256-position partial softmaxes plus one exact reduction at `B=1..8`. Gated attention output
publishes its FP32 and dynamic E4M3 seams and applies the source-native projection at exact
`B=1..8` and `T=32,64,128,1024`; the prefill projection uses exact 32x32 tiles through T=128 and
64x32 tiles at T=1024. The resident text owner
composes all 48 GDN layers, 16 attention layers, source-routed MLPs, and the LM head into one
directly timed graph at every exact `B=1..8` and one from-empty graph at each exact
`T=32,64,128,1024`; server routing remains separate.
The source-backed layer-63 full-attention owner separately composes exact `B=1..8` decode and
from-empty causal `T=32,64,128,1024` prefill graphs; T=1024 selects the admitted P4 macro GQA route.

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
target/benchmarks/perf-smoke/long-context-paged-gqa.json
target/benchmarks/perf-smoke/attention-output.json
```

Every performance command also executes the release SM120 build and checks the PTX/SASS entry and
resource inventory before launching the benchmark.

A complete in-process resource sweep compiles the generated SM120 PTX into one shared cubin, parses
its resource inventory once, and lazily dumps its SASS once. Every suite still independently checks
its exact entries, launch bounds, registers, stack, local memory, shared memory, and required
instructions. Reusing the identical compiler artifact removes repeated `ptxas` and `cuobjdump`
work; it does not turn the suite checks into one aggregate pass.

After owner warmup, the runner sustains the production graph for at least two seconds and applies
the checked clock-spread policy before collecting the full rotated sample matrix. An unlocked or
otherwise incomparable device therefore refuses before a long suite spends its timing window.
The same clock policy remains in force over the complete measurement; the probe is an early guard,
not a substitute for final telemetry.

To retain exploratory timings on an intentionally uncontrolled clock, set
`TUISKO_DIAGNOSTIC_ALLOW_CLOCK_DRIFT=1`. The resulting JSON records
`clock_policy: diagnostic_uncontrolled`; `perf bless` refuses it, so diagnostic evidence cannot
silently become performance authority.

If clocks pass the loaded probe and drift only later during a long measurement, the runner still
writes the completed medians with `clock_policy: diagnostic_uncontrolled` and then returns a
refusal. This preserves tuning evidence without weakening the gate or making it blessable.

Benchmark repetition budgets are selected by the timed boundary rather than inherited from one
global leaf default:

| Duration class | Samples | Replays per sample | Warmup replays | Typical boundary |
|---|---:|---:|---:|---|
| short graph | 40 | 256 | 1,024 | microsecond-scale operator |
| long graph | 40 | 32 | 128 | LM head or composed owner |
| resident model | 40 | 1 | 16 | complete 64-layer graph |

One resident replay already takes tens of milliseconds, so repeating it 256 times provides no
useful timer-resolution benefit and creates an avoidable thermal phase. Reports bind the warmup and
replay counts into their performance identity; a baseline comparison refuses when either changes.

## Command reference

| Command | Purpose | Output |
|---|---|---|
| `cargo run -p xtask -- build-sm120` | Build the release device artifact and check entries, registers, stack, local, and shared bytes | terminal |
| `cargo run -p xtask -- qualify-frontend SNAPSHOT` | Check exact template, tokenizer, streaming, and prefix-cache behavior | terminal |
| `cargo run -p xtask -- qualify-generation SNAPSHOT` | Check prompt-to-sampling-to-streaming state over exact BF16 logit rows | terminal |
| `cargo run -p xtask -- qualify-residual-norm` | Run the independent numerical and graph-replay oracle | terminal |
| `cargo run -p xtask -- qualify-fp8-qkv` | Check represented activation codes/scales and QKV output for B=1..8, T=16, and T=32/64/128/1024 prefill, including padded T=32 reads and graph replay | terminal |
| `cargo run -p xtask -- qualify-fp8-gdn-input` | Check represented activation codes/scales and GDN Q/K/V/Z output for B=1..8 and T=32/64/128/1024 prefill, including padded T=32 reads and graph replay | terminal |
| `cargo run -p xtask -- qualify-fp8-lm-head` | Run the independent represented-value full-vocabulary LM-head oracle and benchmark-accounting test | terminal |
| `cargo run -p xtask -- qualify-fp8-swiglu` | Run the exhaustive represented-value gate/up SwiGLU oracle and graph-replay gate | terminal |
| `cargo run -p xtask -- qualify-fp8-down` | Run the exhaustive represented-value dense-FP8 down oracle and graph-replay gate | terminal |
| `cargo run -p xtask -- qualify-nvfp4-swiglu` | Check represented E2M1/E4M3 seams, A16/W4A4 production routing, immutable weights, graph replay, stable addresses, and post-warmup allocation at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-nvfp4-down` | Check represented E2M1/E4M3 activation and down-projection seams, immutable input/weights, graph replay, stable addresses, and post-warmup allocation at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-nvfp4-mlp SNAPSHOT` | Check source layer 55, route-specific A16/W4A4 scratch, every observable seam, exact B=1..8 and T=32/64/128/1024 graphs, immutable weights, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-qwen35-nvfp4-mlp SNAPSHOT` | Check Qwen3.5 source layer 0, ModelOpt scale conversion, route-specific A16/W4A4 scratch, every observable seam, exact-B graphs, immutable weights, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-qwen35-nvfp4-qkv` | Check Qwen3.5 fused Q/gate, K, and V represented values with three weight-scale divisors at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-qwen35-nvfp4-attention-output` | Check Qwen3.5 sigmoid gating, BF16 projection seam, represented NVFP4 output, immutable inputs, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-qwen35-attention-qk-prepare` | Check Qwen3.5 Q/K zero-centered normalization, three-axis MRoPE, represented BF16 cache append, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-gdn-prepare` | Check the two control formulas, mapped width-4 convolution/history updates, immutable seams, stable ownership, and graph replay at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-gdn-recurrence` | Check mapped FP32 state transitions, causal prefill, gated normalization, immutable seams, stable ownership, and graph replay at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-gdn-output` | Check dynamic E4M3 quantization, source-native output projection, immutable seams, stable ownership, and graph replay at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-attention-qk-prepare` | Check Q/K zero-centered normalization, three-axis MRoPE, represented E4M3 cache append, and graph replay at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-paged-gqa` | Check exact page lookup, grouped-head mapping, represented E4M3 online softmax, immutable seams, graph replay, stable addresses, and allocation behavior at B=1..8, shared T=32/64/128 prefill, and partitioned T=128 P8/P16 deep tails | terminal |
| `cargo run -p xtask -- qualify-qwen35-paged-gqa` | Check Qwen3.5 exact page lookup, grouped-head mapping, represented BF16 online softmax, immutable seams, graph replay, stable addresses, and allocation behavior at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-long-context-paged-gqa` | Check every partition bucket through 220,000 positions, all partial/reduction seams, untouched scratch, and graph replay at B=1..8 | terminal |
| `cargo run -p xtask -- qualify-attention-output` | Check sigmoid gating, the published FP32 seam, dynamic E4M3 quantization, source-native projection, and graph replay at B=1..8 and T=32/64/128/1024 | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-fusion SNAPSHOT` | Check both source zero-centered normalization seams, the complete source-BF16 `[5120,10240]` projection, exact B=1..8 routes, immutable inputs/weights, graph replay, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-qkv SNAPSHOT` | Check lossless Q/gate/K/V gathering, the complete source-BF16 `[14336,5120]` projection, exact B=1..8 routes, immutable inputs/weights, graph replay, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-qk-prepare SNAPSHOT` | Check source Q/K norms, independent zero-centered normalization and three-axis MRoPE formulas, represented BF16 K/V append, exact B=1..8 routes, immutable inputs/weights/metadata, graph replay, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-paged-gqa` | Check exact MTP page lookup, 24-query/4-KV-head grouping, represented BF16 online softmax, immutable query/cache/metadata seams, exact B=1..8 eager and graph replay, stable addresses, and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-attention-output SNAPSHOT` | Check mathematical sigmoid gating, published FP32 and BF16 seams, the complete source-BF16 `[5120,6144]` projection, exact B=1..8 routes, immutable QKV/weights, graph replay, stable addresses, and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-bf16-mlp SNAPSHOT` | Check the complete source-BF16 gate/up SwiGLU and `[5120,17408]` down formulas, the represented BF16 activation/output seams, exact B=1..8 routes, immutable inputs/weights, graph replay, stable addresses, and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-mtp-layer SNAPSHOT` | Check the complete source-backed MTP owner through exact draft B=1..8, prime-only K=1..4, and causal realignment K=1..4; rerun every independent leaf/source oracle; compare every mutable seam under eager and graph replay; and verify stable ownership, exact byte accounting, inactive boundaries, and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-resident-mtp SNAPSHOT` | Check the long-context MTP owner through exact prompt, scalar-tail, seeded-draft B=1..8, residual-continuation B=1..8, prime K=1..4, and realignment K=1..4 routes; rerun the complete source-backed MTP authorities; compare eager and graph seams; and verify shared page reset, truncation, retention, reuse, recycling, stable addresses, exact bytes, and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-generation-mtp-greedy SNAPSHOT` | Compare single-slot greedy draft-three generation with the target-only production fallback; force target verification K=1..4 through exact output limits; and check continuation, commit, realignment, streaming, stable addresses, exact ownership, and zero post-warmup device growth | terminal |
| `cargo run -p xtask -- qualify-generation-mtp-sampling SNAPSHOT` | Gate unbiased draft-three sampling with an independent induced-law and sequence oracle that rejects known mutants; force target verification K=1..4; exercise non-identity output-history penalties; and check deterministic seeds, streaming, stable addresses, exact ownership, and zero post-warmup device growth | terminal |
| `cargo run -p xtask -- qualify-generation-mtp-batch SNAPSHOT` | Gate compact target-plus-MTP scheduling over every exact `(B=1..8, K=1..4)` transaction; compare every committed greedy output with an independent BF16 target-logit argmax; exercise sampled lanes, cancellation, exact hidden-boundary prefix restoration, divergence fallback, slot recycling, stable addresses, exact ownership, and zero post-warmup device growth | terminal |
| `cargo run -p xtask -- qualify-dense-fp8-mlp SNAPSHOT` | Check source layer 60, every exact B=1..8 and T=32/64/128/1024 graph, all working and residual seams, tensor-map immutability, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-dense-fp8-gdn-layer SNAPSHOT` | Check the complete source layer-60 mixer/MLP seams, persistent state, exact B=1..8 and T=32/64/128/1024 graphs, tensor-map immutability, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-full-attention-layer SNAPSHOT` | Check complete source layer-63 attention/MLP seams, represented KV cache, exact B=1..8 and T=32/64/128/1024 graphs, P4 macro partials, immutable tensor maps, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-qwen35-full-attention-layer SNAPSHOT` | Check complete Qwen3.5 source layer-31 attention/MLP seams, BF16 KV cache, exact-B graphs, immutable weights, stable addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-resident-model SNAPSHOT` | Check all 64 source routes, final source-backed formulas, dynamic page recycling/remapping and isolated reset, persistent state/cache, short plus six-bucket exact-B whole-model graphs, six exact prefill graph specializations across from-empty and nonzero-prefix metadata, independent long-attention seam formulas, stable device/host addresses, and owner allocation | terminal |
| `cargo run -p xtask -- qualify-target-mtp-verify SNAPSHOT` | Check singleton K=1..4 plus every exact lane-major `(B=1..8, K=1..4)` target verification; provisional per-lane GDN rollback; distinct accepted-prefix replay; source endpoint mathematics; every target/cache seam; graph agreement; stable ownership; exact bytes; and zero post-warmup allocation | terminal |
| `cargo run -p xtask -- qualify-resident-generation SNAPSHOT` | Check pinned vLLM next-token fixtures, exact and composed T1024/T128/T64/T32 prefill plans through the P16 context band, scalar tail boundaries, frontend, greedy control, streaming decode, stable ownership, and zero post-warmup device allocation | terminal |
| `cargo run -p xtask -- qualify-resident-batch-generation SNAPSHOT` | Compare compact mixed-length scheduling with sequential requests, including every B=1..8 decode route, exact and composed prefill plans, scalar tail boundaries, noncontiguous survivor replay, cancellation, exact retained-prefix reuse, divergence fallback, slot recycling, stable ownership, and zero post-warmup device allocation | terminal |
| `cargo run -p xtask -- qualify-text-endpoint SNAPSHOT` | Check source embeddings, final norm, sampled full-formula logits, graph replay, stable addresses, and post-warmup allocation | terminal |
| `cargo run -p xtask -- bench-gdn-prepare` | Measure every exact control-plus-convolution graph after an untimed exact-history restore | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-gdn-recurrence` | Measure every exact stateful recurrence graph after an untimed exact-state restore | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-gdn-output` | Measure every exact decode and prefill output quantize-plus-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-attention-qk-prepare` | Measure every exact Q/K prepare and cache-append graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-paged-gqa` | Measure exact B=1..8 graphs at a 130-token context, causal shared T=32/64/128 graphs, partitioned T=128 tails, and production-P4 T=1024 macro graphs at contexts 32,768 and 98,304 | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-paged-gqa` | Measure every exact Qwen3.5 B=1..8 BF16 paged-GQA graph at a 130-token context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-long-context-paged-gqa` | Measure every exact two-stage paged GQA graph with the complete 3,438-page pool divided among active slots | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-attention-output` | Measure every exact sigmoid-gate, quantize, and output-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-fusion` | Measure every exact B=1..8 production graph for both BF16 norms plus the source-BF16 MTP fusion projection | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-qkv` | Measure every exact B=1..8 gathered source-BF16 MTP Q/gate/K/V projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-qk-prepare` | Measure every exact B=1..8 MTP source-norm Q/K preparation and BF16 cache-append graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-paged-gqa` | Measure every exact B=1..8 production MTP BF16 paged-GQA graph at a 130-token context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-attention-output` | Measure every exact B=1..8 production graph for sigmoid gating, represented-BF16 activation, and the source-BF16 output projection | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-bf16-mlp` | Measure every exact B=1..8 production graph for source-BF16 gate/up SwiGLU and down projection | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-mtp-layer SNAPSHOT` | Directly measure every complete source-backed MTP draft B=1..8 production graph at a 131-token, three-page BF16 context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-resident-mtp SNAPSHOT` | Directly measure every resident long-context MTP seeded-draft, same-round residual-continuation, and explicit hidden-handoff B=1..8 graph, including pinned input upload, the respective target/prior-residual handoff, shared target LM head, and the production cache regime | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-target-mtp-verify SNAPSHOT` | Directly measure every exact lane-major `(B=1..8, K=1..4)` target verification and verification-plus-full-prefix commit graph at a 132-token context, with matching production metadata restored outside timing | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-generation-mtp-greedy SNAPSHOT` | Directly measure host completion for one proposal-ready draft-three/K=4/realignment round and one production prompt-plus-eight-output greedy MTP request; prompt preparation for the round is outside its timing rather than estimated from leaf medians | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-generation-mtp-sampling SNAPSHOT` | Directly measure host completion for one seeded sampled draft-three/K=4 round, one complete sampled request, and the same request with non-identity output-history penalties; the round excludes prompt preparation and no boundary is estimated from leaf medians | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-generation-mtp-batch SNAPSHOT` | Directly measure the production compact scheduler's proposal-ready draft-three/K=4 transaction at every exact B=1..8, including host control, lane-major target verification/commit, per-lane MTP realignment, transfers, and synchronization; prompt preparation is untimed and no constituent median is summed | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-swiglu` | Measure every exact retained A16/W4A4 NVFP4 gate/up SwiGLU graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-down` | Measure every exact A16 decode and W4A4 prefill NVFP4 down-projection graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-nvfp4-mlp SNAPSHOT` | Measure every complete source-backed layer-55 decode and prefill MLP graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-nvfp4-mlp SNAPSHOT` | Measure every complete source-backed Qwen3.5 layer-0 MLP graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-nvfp4-qkv` | Measure every exact Qwen3.5 fused NVFP4 QKV graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-nvfp4-attention-output` | Measure every complete Qwen3.5 sigmoid-gate, BF16-stage, and NVFP4 output graph with input restoration outside timing | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-attention-qk-prepare` | Measure every exact Qwen3.5 Q/K prepare and cache-append graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-dense-fp8-gdn-layer SNAPSHOT` | Measure every complete source-backed layer-60 graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-full-attention-layer SNAPSHOT` | Measure every complete source-backed layer-63 B=1..8 graph at a 131-token context and every from-empty T=32/64/128/1024 prefill graph | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-qwen35-full-attention-layer SNAPSHOT` | Measure every complete Qwen3.5 source-backed layer-31 graph at a 131-token, three-page BF16 context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-resident-model SNAPSHOT` | Directly measure every complete 64-layer plus LM-head graph at a 131-token context | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-resident-prefill SNAPSHOT` | Directly measure complete T=32/64/128/1024 resident prompt graphs across from-empty, shared-tail, P8/P16 T128, and P4 macro-tail contexts with final-token-only LM head | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-resident-long-context-model SNAPSHOT` | Directly measure every complete 64-layer plus LM-head long graph with one 131,073-token row and compact one-token survivors | terminal or `--json PATH` |
| `cargo run -p xtask -- bench-text-endpoint SNAPSHOT` | Measure every source-backed final-norm plus LM-head graph | terminal or `--json PATH` |
| `cargo run -p xtask -- perf smoke` | Three-sample harness and environment smoke test for every suite | `target/benchmarks/perf-smoke/*.json` |
| `cargo run -p xtask -- perf leaf` | Full registered leaf timing and memory reports | `target/benchmarks/perf-leaf/*.json` |
| `cargo run -p xtask -- perf energy` | Full leaf reports plus a sustained power window per route | `target/benchmarks/perf-energy/*.json` |
| `cargo run -p xtask -- perf gate` | Run every oracle, measure every suite, and compare checked baselines | `target/benchmarks/perf-gate/*.json` |
| `cargo run -p xtask -- perf candidate SUITE [SNAPSHOT] [options]` | Qualify the changed suite, then directly time its exact downstream owner/model cone | `target/benchmarks/perf-candidate/SUITE/*.json` |
| `cargo run -p xtask -- perf check SUITE [SNAPSHOT]` | Measure the complete authoritative dependency cone and compare each checked baseline | `target/benchmarks/perf-check/SUITE/*.json` |
| `cargo run -p xtask -- perf bless SUITE [SNAPSHOT]` | Run one oracle and explicitly replace that leaf or composed suite's baseline | `qual/baselines/SUITE-sm120.json` |
| `cargo run -p xtask -- perf iterate SUITE --batch B --hypothesis TEXT` | Run one exact leaf route through verified qualification/build receipts, timing, and a diagnostic-only comparison | `target/optimization/SUITE/` |
| `cargo run -p xtask -- perf diagnose-diff SUITE REPORT [--json OUTPUT]` | Compare a complete or exact-B report diagnostically while admitting only case-policy and generator-provenance differences | `target/benchmarks/perf-diagnostic/` |
| `cargo run -p xtask -- profile resident-model SNAPSHOT --batch B --replays N --tool nsys` | Capture the production graph after warmup and attribute every node to semantic owner, stage, and layer | `target/profiles/resident-model-bB/` |
| `cargo run -p xtask -- profile resident-model SNAPSHOT --batch B --tool ncu --kernel REGEX` | Collect hardware counters for one selected production kernel family | `target/profiles/resident-model-bB/` |

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
  --warmup-launches 1024 \
  --json target/benchmarks/residual-norm.json
```

Use `--batch B` for a fast exact-route diagnostic. The report records
`case_policy: diagnostic_subset` and `selected_batch_size: B`; it cannot be blessed or compared as
the complete authority. This is the intended inner loop for a B-specific retile. Remove the option
before final comparison so every admitted `B=1..8` route is timed.

For the qualified inner loop, use the leaf-only wrapper and state the single measured hypothesis:

```bash
export TUISKO_AGENT_ITERATION_STARTED_UNIX_MILLISECONDS="$(date +%s%3N)"
# Make the one-hypothesis source change.
cargo run -p xtask -- perf iterate nvfp4-down \
  --batch 1 \
  --hypothesis "coalesce B=1 weight sectors"
```

Before any qualification or build work, the wrapper requires device zero to be the exact RTX 5090,
idle below the admitted memory threshold, free of compute processes, and selected by an unset or
exactly `0` `CUDA_VISIBLE_DEVICES`. The benchmark process still performs its complete independent
preflight and telemetry checks immediately before timing.

`perf iterate` reuses a numerical qualification only when an ignored receipt matches the complete
device-input fingerprint and the hashed physical-device/driver identity. It reuses a build only
when the input, cuda-oxide revision, complete resource-baseline digest, executable digest, and PTX
digest match. It may copy those two verified build artifacts from another registered Git worktree;
it never shares a mutable Cargo target directory. A normal `build-sm120` still reruns all resource
gates after local or cross-worktree reuse, while the inner loop may trust a receipt that was written
only after those gates passed. A stale, malformed, or hash-mismatched ignored receipt is a cache
miss and forces fresh evidence; it never blocks the rebuild or weakens validation.

Each attempt, including a controlled refusal or failure, gets a JSON manifest and raw benchmark
report under `target/optimization/SUITE/`. A bundled-SQLite index at
`target/optimization/iterations.sqlite3` records the hypothesis, Git/input identity, result, and
wall time for preflight, qualification, build, benchmark, comparison, the whole command, and—when
the environment variable above is set—the complete agent loop. These files are ignored diagnostic
evidence, not checked authority.

The diagnostic comparator requires the same suite, target, driver, compute capability, controlled
clock policy and band, sampling identity, timing/power scope, workload keys, operation counts, and
memory contract as the checked baseline. It permits only the expected complete-versus-exact-B case
selection and generator/resource-provenance lag. Its JSON always records `authoritative: false`, it
does not fail merely because a metric regressed, and no diagnostic path calls baseline blessing.
An explicit diagnostic `--json` output must be a repository-relative path under `target/`.

Use `cargo run -p xtask -- bench-fp8-qkv`, `bench-fp8-gdn-input`, `bench-fp8-lm-head`,
`bench-fp8-swiglu`, `bench-fp8-down`, `bench-gdn-prepare`, `bench-gdn-recurrence`, or
`bench-gdn-output`, `bench-nvfp4-swiglu`, `bench-nvfp4-down`, `bench-attention-qk-prepare`,
`bench-paged-gqa`, `bench-long-context-paged-gqa`, `bench-attention-output`, or
`bench-mtp-bf16-fusion`, `bench-mtp-bf16-qkv`, `bench-mtp-bf16-qk-prepare`, or
`bench-mtp-bf16-paged-gqa` with the same options for one operator suite only.

`bench-text-endpoint SNAPSHOT` accepts the same options. It is intentionally separate from the
leaf-wide `perf` commands until its first reviewed baseline is blessed.

`bench-dense-fp8-mlp SNAPSHOT` directly measures the complete source-backed layer-60 MLP graph at
every exact `B=1..8` and `T=32,64,128,1024` with the same options. It does not infer composition
time from leaf medians and stays outside leaf-wide `perf` until the source-backed route receives a
reviewed baseline.

`bench-nvfp4-mlp SNAPSHOT` directly measures the complete source-backed layer-55 MLP graph at every
exact `B=1..8` and `T=32,64,128,1024`. Its `B=1,5..8` routes include production E2M1 activation
quantization and W4A4 gate/up projection; `B=2..4` preserve the BF16 gate/up activation, and all
decode routes use the represented-weight A16 down projection. Prefill uses W4A4 for both
projections and retains distinct represented gate/up and down activation seams. It remains outside
leaf-wide `perf` until a locked-clock local baseline is reviewed.

`bench-dense-fp8-gdn-layer SNAPSHOT` measures each complete stateful layer-60 decode and prefill
graph after an untimed production-owner reset of its history and FP32 recurrence. It reports exact
`B=1..8` and `T=32,64,128,1024` routes; setup and allocation remain outside the timed region.

`bench-full-attention-layer SNAPSHOT` directly measures the complete layer-63 graphs rather than
summing leaf medians. Decode uses a 131-token warm cache that crosses both 64-token page seams;
prefill uses one from-empty shared table row and reports the exact T=32/64/128/1024 causal routes,
including the production P4 macro workspace at T=1024.

`bench-qwen35-full-attention-layer SNAPSHOT` applies the same direct boundary to Qwen3.5 layer 31.
Its accounting distinguishes the B=2 A16 MLP path from the W4A4 paths and records the BF16 cache
separately from resident weights and workspace.

`bench-resident-model SNAPSHOT` times the complete production graph directly; it never derives a
model latency from leaf medians. The current 131-token route exercises all 64 layers and the LM
head with resident weights, shared workspace, recurrent state, and represented KV caches. It omits
the repeated-operation graph because one complete graph is already long enough for CUDA-event
resolution and duplicating hundreds of model nodes would measure a different owner. The production
embedding-staging graph restores represented input rows before each sample and remains outside the
timed whole-model replay.

`bench-resident-prefill SNAPSHOT` directly times the same resident owner at exact
`T=32,64,128,1024`. Its complete inventory includes the four from-empty prompts plus shared
T32/T64 tails, both absolute-context T128 partition bands, and a nonzero-prefix P4 T1024 macro
tile. All 64 layers advance one mapped GDN state/history row and one paged-attention table row
causally. Only the final normalized tile row enters the LM head; represented embedding and exact
metadata restoration remain outside the timed graph.

`bench-resident-long-context-model SNAPSHOT` uses the same production owner and direct graph timing.
Its shared-pool profile assigns 131,073 positions to the first compact row and one position to each
survivor, selecting the 860-partition graph for every exact `B=1..8` route without inventing eight
copies of the physical KV pool. Prompt preparation and metadata uploads remain outside timing.

Add `--energy-seconds 2` for sustained energy sampling. At least three samples, one launch per
sample, and a two-second energy window are required.

## Optimization dependency cones

An optimization starts at one exact suite and moves outward through directly affected production
owners. `perf candidate` and `perf check` use a checked registry for that relationship; they do not
estimate a composed boundary by adding leaf medians. Examples include:

```text
nvfp4-down -> nvfp4-mlp -> resident-model + resident-prefill + resident-long-context-model
fp8-down -> dense-fp8-mlp -> dense-fp8-gdn-layer + full-attention-layer
         -> resident-model + resident-prefill + resident-long-context-model
long-context-paged-gqa -> resident-long-context-model
```

The candidate mode is diagnostic: it runs the changed suite's oracle once, builds and resource-
checks once, and measures the direct dependency cone. The check mode requalifies each distinct
correctness boundary in the cone, uses complete suite defaults, and compares every cone report with
its independent baseline. The three resident timing profiles share one resident-model oracle. A
source-backed composed suite needs the admitted snapshot path even when the selected root is a
synthetic leaf.

Keep resource and timing authorities distinct. A leaf resource change is reviewed in its text
baseline; each directly measured composed boundary has its own JSON performance baseline. A leaf
improvement is not a model win until the directly timed resident report shows it.

## Resident graph profiling

The profiling command uses the same resident owner, allocations, stream, warmed cache state, exact
batch graph, and 131-token context as the production short-context benchmark. Model loading,
materialization, graph construction, and warmup finish before the application calls the public CUDA
profiler-control API. Nsight therefore captures only explicit embedding preparation and complete
resident graph replays.

For Nsight Systems:

```bash
cargo run -p xtask -- profile resident-model SNAPSHOT \
  --batch 1 --replays 3 --tool nsys
```

The command exports the trace to SQLite and processes it in Rust. It refuses unless the observed
kernel sequence matches the semantic manifest exactly. The output directory contains:

- the native `.nsys-rep` and exported `.sqlite` reports;
- CUDA's verbose graph `.dot` inventory;
- a semantic JSON manifest mapping graph-node ranges to exact layers, owners, and source routes;
- per-node, per-stage, per-layer, and per-replay CSV timings; and
- tool, Git status, executable hash, and device provenance metadata.

The replay CSV closes complete graph span against the sum of kernel durations and inter-kernel
gaps. Treat it as profiler evidence, not a regression median: tracing perturbs timing, and profiler
clocks may differ from the checked benchmark clock band.

Use Nsight Compute only after the Systems trace identifies an Amdahl-relevant kernel family:

```bash
cargo run -p xtask -- profile resident-model SNAPSHOT \
  --batch 1 --replays 1 --tool ncu --kernel 'nvfp4_down_a16_b1'
```

The NCU report diagnoses physical memory transactions, stalls, occupancy, and instruction-pipeline
use. Its isolated replay duration is not directly comparable with an uninstrumented resident-model
baseline, and local speedup estimates must not be added across kernels.

Treat occupancy, wave-tail, and excessive-transaction warnings as experiment selectors rather than
optimization goals. Before a retile, account for the invariant useful rows or warps and every fixed
per-CTA cost, including staging, barriers, inactive lanes, and tail guards. Making a grid divide the
SM count evenly can duplicate that work or weaken a branch-free mapping even when reported
occupancy rises. Direct production timing decides whether to retain the change; before-and-after
counters support a causal explanation but do not replace that timing.

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

The residual-norm `B=1..8`, FP8-QKV `B=1..8`, FP8-GDN-input `B=1..8`, FP8-LM-head, dense-FP8-SwiGLU `B=1..8`,
dense-FP8-down, NVFP4-SwiGLU, and NVFP4-down cases are `operator/decode`, warm-cache, CUDA-Graph
workloads. They set batch and active tokens to the exact batch. FP8-QKV `T=16` is an
`operator/mtp` case; residual norm, FP8-QKV, FP8-GDN-input, GDN-prepare, and GDN-recurrence
`T=32,64,128,1024` routes are
`operator/prefill` cases. Both projection T=32 routes read a padded 64-row activation-code tile,
and their logical-byte accounting includes those immutable padding reads.
GDN-prepare samples restore the exact mapped history outside the timed interval; the timed graph
contains only the production control, parallel causal-convolution, and ordered history-publication
launches.
GDN-recurrence samples likewise restore mapped FP32 state outside timing. Prefill exposes the 48
independent value-head CTAs and advances tokens serially inside each CTA because every token reads
the state produced by its predecessor.
Paged GQA `B=1..8` cases are `operator/decode` at context 130. Its shared `T=32,64,128` cases are
`operator/prefill`; token `i` attends a two-token prefix plus the causal span through `i`. Logical
bytes charge each K/V tile once per adjacent-token/KV-head group rather than duplicating its six
query-head consumers. The partitioned `T=128` P8/P16 cases are also `operator/prefill`; accounting
includes one query read per active partition, one K/V load per 32-row/query-head group, exact
length/table/page metadata reads, and both the producer writes and reducer reads of every complete
FP32 partial state. Macro `T=1024/P=4` cases use the same accounting at contexts 32,768 and 98,304;
their resident maximum workspace still covers every qualified `P=1,2,4,8,16` route.
Dense-FP8-SwiGLU, dense-FP8-down, NVFP4-SwiGLU, and NVFP4-down `T=32,64,128,1024`
cases are `operator/prefill` with prompt and context lengths equal to the active rows. Each
dense-FP8 T=1024 owner has
two 128-byte address-bound TMA descriptors; the NVFP4 route uses its prepared W4A4 launches without
descriptors. Concurrency, output, and prefix cache do not apply to these leaf suites.

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
- warmup replay count and complete-inventory case policy;
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

For tuning, first use an exact route subset and profiler artifacts to select the change. Then run
the changed oracle and resource gate, directly measure its composed dependency cone, and finally
repeat with the complete inventory before comparison or blessing. Preserve profiler reports under
`target/`; they are diagnostic build products, not checked source.

Criterion remains the harness for pure host work such as checkpoint admission, tokenization,
template rendering, prefix lookup, sampling, and streaming detokenization. CUDA-event timing,
exclusive-device controls, NVML telemetry, and device baselines remain in this runner.

## Current limitations

- Decode operator coverage is exact `B=1..8`; prefill remains limited to the explicitly listed
  residual norm, FP8-QKV, Q/K preparation/cache-append, shared early-context, partitioned deep-tail
  and macro paged-GQA, attention-output, dense-FP8 MLP, NVFP4 SwiGLU, NVFP4 down,
  full-attention-layer, FP8-GDN-input, GDN-prepare, GDN-recurrence, GDN-output, and dense-FP8
  GDN-layer routes. Server admission composes only exact T1024/T128/T64/T32 whole-model graphs for
  cold prompts and reused-prefix suffixes; an unmatched final span below 32 tokens primes through
  exact B1 decode.
- The suite labels warm cache; it does not yet implement a generic cold-cache displacement protocol.
- There is no full-server TTFT, inter-token-latency, concurrency, prefix-reuse, or end-to-end MTP
  benchmark in this repository yet. Direct long-context operator and resident-model timing exists.
- Power and energy are reportable for the resident model; full-server energy remains future work.
- `ncu` and Nsight Systems traces are first-class diagnostic artifacts, but remain outside checked
  regression comparison and baseline blessing.

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

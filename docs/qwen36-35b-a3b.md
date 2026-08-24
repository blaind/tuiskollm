# Qwen3.6 35B-A3B NVFP4 integration

This document records the admission plan for
[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
at immutable revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`. It is a third concrete product,
not a widening of the Qwen3.8 or Qwen3.5 hot paths into a generic model backend.

## Feasibility decision

Single-GPU support on the 32 GB RTX 5090 is feasible enough to implement. The exact snapshot has
124,468 tensors and 23,407,580,856 tensor-payload bytes across three shards. A text-first resident
owner can leave the 1,017,118,720-byte BF16 embedding in its host mmap and retain
19,808,038,104 bytes (18.448 GiB) of text weights on the device.

At the full 262,144-token context, one FP8 KV slot needs 2,684,354,560 bytes: ten attention layers,
two K/V planes, two KV heads, 256 columns, and one byte per value. Thirty recurrent layers add
62,914,560 state bytes and 1,474,560 history bytes. Adding the complete BF16 Vision and MTP weights
gives this pre-workspace budget:

| Owner | Bytes | GiB |
| --- | ---: | ---: |
| Text weights, excluding the host embedding | 19,808,038,104 | 18.448 |
| One full-context FP8 KV slot | 2,684,354,560 | 2.500 |
| Recurrent FP32 state and BF16 history | 64,389,120 | 0.060 |
| Vision weights | 893,142,496 | 0.832 |
| MTP weights | 1,689,281,536 | 1.573 |
| Total before workspaces, graphs, and CUDA context | 25,139,205,816 | 23.413 |

The local target reports 31.843 GiB of framebuffer memory, leaving 8.430 GiB before workspaces,
graphs, allocation alignment, and CUDA context ownership. Feasibility therefore remains conditional
on an exact resident-layout test and a real peak/headroom gate. It does not authorize CPU expert
offload, demand paging, or partial model residency.

## Exact checkpoint contract

The pinned config describes a 40-layer `qwen3_5_moe` hybrid:

- hidden width 2,048, RMSNorm epsilon `1e-6`, and vocabulary 248,320;
- 30 linear-attention layers and 10 full-attention layers at interval four;
- 16 query heads, two KV heads, and width-256 full-attention heads;
- the same 16 Q/K and 32 value heads of width 128 used by the admitted Qwen3.5 recurrence;
- 256 routed experts, exactly eight selected per token, and width 512 for both routed and shared
  experts;
- one BF16 MTP layer, the existing 27-block BF16 Vision encoder, and 262,144 positions.

The 40 routers are BF16 `[256,2048]` matrices. Each layer stores 256 separate ModelOpt NVFP4
experts. Every expert owns gate/up `[512,2048]` and down `[2048,512]` represented weights, E4M3
block scales, and scalar input/divisor metadata. Gate/up words are adjacent per expert, but the
source orders expert identifiers lexicographically. Materialization must reorder complete source
words into a documented numeric expert-major layout without decoding and requantizing them.

Linear-attention and full-attention projections are source E4M3 with scalar FP32 scales. The LM
head is ModelOpt NVFP4 `[248320,2048]`. The external quantization metadata specifies an FP8 KV
cache. MTP is intentionally unquantized BF16 and accounts for 1.573 GiB by itself.

The tokenizer is byte-identical to the admitted Qwen3.5 tokenizer
(`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`), but the chat template and
generation config are not. The frontend may reuse the tokenizer codec only; it must separately pin
the Qwen3.6 template, its two EOS identifiers, and sampled defaults `temperature=1.0`,
`top_p=0.95`, `top_k=20`.

## Reuse boundary

Reuse is allowed where represented semantics and arithmetic are already identical:

- mmap-backed safetensors, typed BF16/F32/E4M3/U8 views, ModelOpt scalar checks, and lossless scale
  swizzling;
- the 32-value-head GDN recurrence and convolution math after an independent geometry check;
- CUDA arena, prepared launch, graph, telemetry, benchmark, and resource-gate infrastructure;
- tokenizer decoding and streaming UTF-8 ownership because the tokenizer artifact is identical.

New concrete routes are required for:

- BF16 router logits, FP32 softmax, deterministic top-8 selection, and normalized route weights;
- expert-major NVFP4 gate/up, SwiGLU, down projection, route-weight application, and fixed-order
  reduction;
- the width-2,048 residual and projection families;
- two-KV-head Q/K preparation, grouped-query attention with eight query heads per KV head, and the
  exact FP8 cache codec;
- the NVFP4 LM head and Qwen3.6 endpoint;
- MoE layer, resident model, MTP, Vision, and server owners.

The router oracle follows the reference inference formula: BF16 matrix logits, FP32 softmax,
top eight, renormalization over those eight probabilities, routed expert output, plus
`sigmoid(shared_expert_gate) * shared_expert_output`. Ties and the eight-expert accumulation order
must be explicitly pinned rather than inherited from an unstable library primitive.

The current partial slice admits the complete source inventory and lossless expert materialization,
then qualifies BF16 router logits, normalized top-eight selection, all selected routed experts, the
always-active shared expert, and their fixed-order combination at every exact `B=1..8`. The expert
owner retains all 256 numeric-order expert planes in 454,760,448 weight bytes and uses 434,448
address-stable workspace bytes. Its 24 gate/up, down, and combine entries use zero stack/local
memory and pass complete eager/oracle plus CUDA Graph agreement.

At a measured 2,182 MHz median SM clock, the warm expert-only repeated path is 12.726 us at B=1
and 76.390 us at B=8. The rise after B=6 coincides with the selected expert working set exceeding
the target's cache, so these are leaf diagnostics rather than a cold whole-layer or model claim.
The 2,048-wide zero-centered RMSNorm and fused residual-publication routes also cover every exact
`B=1..8`, pass 786,432 oracle/graph/sentinel observations, and measure 1.761/1.858 us plain plus
1.887/1.948 us fused at B=1/8 with a locked 2,197 MHz SM clock. The first GDN projection family
preserves the checkpoint's static FP8 contract: BF16 inputs are quantized with the admitted scalar
scale, source E4M3 Q/K/V/Z weights retain their two scalar scales, and the 64 BF16 A/B control rows
remain a separate exact projection. Exact `T=32/64/128` prompt routes reuse the same represented
scales and switch only the E4M3 projection to the native `m16n8k32` tensor-core tile. Across every
decode and prompt route, the gate passes 532,480 exact activation-code, 3,211,520 FP64-formula
output, 3,744,000 graph-replay, and 33,062,400 inactive-sentinel comparisons with immutable sources
and no post-warmup device growth. Its complete three-node path measures 12.443/33.309 us at B=1/8
and a locked 2,197 MHz SM clock; the unblessed prompt diagnostics measure
102.416/104.407/132.575 us at T=32/64/128 with a 2,182 MHz median SM clock. The decode and prompt
timings are unblessed leaf diagnostics. The following control/convolution stage reuses the Qwen3.5
binary entries because both profiles have the exact same 32 control rows, 8,192 Q/K/V rows, 4,096
value rows, 16 Q/K heads, 32 value heads, width-128 state, and width-four history. Its exact
`T=32/64/128` route computes each four-tap causal output from one mapped prior history without racing
cross-token updates, then publishes the final three represented BF16 values in a separate kernel.
Every decode and prompt route passes the independent control/convolution oracle, graph replay,
inactive sentinels, immutable inputs, stable-address, and zero-growth checks. All 14 entries use
zero stack/local memory. At a diagnostic 2,115 MHz median SM clock, its unblessed intrinsic prompt
path measures 3.739/4.689/6.259 us and its complete graph measures 6.140/6.148/8.192 us at
T=32/64/128. The next FP32 recurrence stage can share the same geometry only after its causal
prompt oracle closes that arithmetic separately. Compile-time geometry assertions and separate
Qwen3.6 oracle entry points make shared binaries executable rather than conventional. The final
GDN projection preserves its source E4M3 `[2048,4096]` plane and static
FP8 scales. Its exact `B=1..8` routes pass 147,456 activation-code, 73,728 FP64-formula output,
221,184 graph-replay, and 344,064 inactive-sentinel comparisons while retaining immutable sources,
stable addresses, and zero post-warmup growth. All 16 entries have zero stack/local memory. At a
2,190 MHz median SM clock, the unblessed warm repeated path measures 9.543/22.031 us at B=1/8;
the complete two-node graph measures 10.852/24.567 us. One source-backed layer-0 owner now
composes both residual boundaries, the complete GDN path, router, eight routed experts, and the
shared expert into eight immutable exact-B graphs over 47 stable addresses. It owns 489,703,808
weight bytes and 18,251,056 workspace/state bytes in one 507,955,968-byte arena. Every batch passes
eager/graph agreement, inactive seam checks, immutable source checks, and zero post-warmup growth;
B=1 additionally passes the complete represented source formula. At a diagnostic 2,107 MHz median
SM clock, its direct graph measures 67.869/201.387 us at B=1/8. Full-attention layers, the endpoint,
exact prefill experts, and model inference remain unimplemented. The full-attention source seam is
now admitted separately: Q, K, and V preserve their E4M3 bytes and three scalar FP32 weight scales
while losslessly gathering into one `[9216,2048]` Q/K/V plane; the source-native
`[2048,4096]` output plane remains zero-copy. The first full-attention compute leaf statically
quantizes BF16 inputs with the checkpoint's shared scalar input scale and projects every exact
`B=1..8` route. It passes 73,728 activation-code, 331,776 FP64-formula output, 405,504 graph-replay,
and 630,784 inactive-sentinel comparisons with immutable sources, stable addresses, and zero
post-warmup growth. Its eight quantizers retain 25 registers, the eight projections retain sorted
register counts `[31,32,35,38,38,40,44,45]`, and every entry uses zero stack/local memory. At a
diagnostic 2,167 MHz median SM clock, its unblessed warm repeated path
measures 8.064/21.350 us at B=1/8; the production graph boundary measures 10.240/22.544 us. The
following BF16 Q/K seam reuses the exact width-256 normalization and 64-wide `[11,11,10]`
interleaved MRoPE arithmetic already qualified for Qwen3.5, but owns separate Qwen3.6 symbols and
the narrower two-KV-head page layout. Its eight routes pass 147,456 prepared-query values, 36,864
exact BF16 cache values, complete eager/graph agreement, untouched pages, immutable inputs, stable
addresses, and zero post-warmup growth in a 2,380,288-byte arena. All eight entries retain 54
registers, zero stack/local memory, and 1,024 bytes of shared memory. At a diagnostic locked
2,197 MHz SM clock, its warm repeated path measures 2.825/3.141 us at B=1/8; the 4.104-us graph
boundary is dispatch-limited throughout this small leaf. The following BF16 paged-GQA leaf owns a
separate 8:1 query/KV-head route at every `B=1..8`. Its independent FP64 page/online-softmax oracle
passes 147,456 active outputs, 114,688 inactive sentinels, 262,144 graph-replay values, and complete
read-only input checks in a 3,408,640-byte arena. All eight generated entries use 48 registers,
zero stack/local memory, and 1,024 bytes shared. Timing remains unreported because the available
diagnostic run failed the exclusive-device precondition. The gated attention-output leaf then
applies the query-paired sigmoid gate, publishes the BF16 projection seam, statically quantizes it
with the admitted scalar scale, and consumes the source-native E4M3 `[2048,4096]` output plane at
every exact `B=1..8`. Its independent oracle passes 147,456 gated FP32 values, 147,456 exact BF16
staging values, 147,456 exact E4M3 codes, 73,728 FP64-formula outputs, 516,096 graph-replay values,
and 802,816 inactive sentinels with immutable sources, stable addresses, and zero post-warmup
growth in an 8,798,208-byte arena. The eight gate entries use 26 registers, zero stack/local
memory, and 1,024 bytes shared; the reused static-FP8 projection retains its separately checked
resource contract. Timing remains unreported because the available run failed the exclusive-device
precondition after qualification. One source-backed layer-3 owner now composes these attention
leaves, both residual seams, and the routed/shared MoE boundary into eight exact-B graphs over one
487,394,048-byte arena. Its B=1 source oracle and all-batch lifecycle gate pass with 483,085,312
resident weight bytes, 3,145,728 BF16 cache bytes, and 1,161,680 workspace bytes. Its unblessed
direct diagnostic measures 122.901/255.794 us at B=1/8 while clocks span 2,070--2,197 MHz.

The NVFP4 endpoint prerequisite now owns eight exact A16 LM-head routes over the source-represented
`[248320,2048]` plane. It retains BF16 activations rather than requantizing them, reads packed E2M1
codes plus losslessly swizzled E4M3 scales, and applies the exact second-stage source weight scale.
Every `B=1..8` route publishes and graph-replays 8,939,520 finite logits, checks 4,608 independent
represented-value dots including scale-layout seams, preserves 13,905,920 inactive sentinels, and
proves its 286,064,640 weight bytes immutable with stable addresses and no post-warmup growth.
The eight A16 entries use 40--111 registers, 9,216 bytes shared, and zero stack/local memory; their
pinned PTX/SASS retains represented E2M1 conversion, warp reduction, and BF16 publication. Direct
timing remains unreported because the available card was occupied. Exact endpoint bindings retain
the mmap-backed BF16 embeddings/final norm and losslessly swizzle the LM-head E4M3 plane while
borrowing its packed E2M1 words. A single-allocation resident owner now stages embedding rows
without rescanning the LM-head source and captures final-norm plus NVFP4 LM-head graphs at every
exact `B=1..8`. Its source-backed gate passes 73,728 exact embedding values, 73,728 final-norm
values, 2,304 sampled full-formula represented NVFP4 logits, complete eager/graph agreement,
inactive-row sentinels, immutable source planes, stable addresses, and zero post-warmup device
growth in one 290,107,392-byte arena. At locked 2,197/13,801 MHz SM/memory clocks, its unblessed
direct graph measures 199.142/448.010 us at B=1/8; the repeated intrinsic path measures
197.736/446.026 us and the B=1 route reads the 286 MB endpoint at 1,349.78 GiB/s.

The initial resident text layout then composes 30 GDN/MoE layers, ten full-attention/MoE layers,
and the endpoint into 41 address-stable arenas. It accounts 19,808,036,096 device weight bytes,
31,457,280 short-context BF16 cache bytes, 563,187,136 workspace/state bytes, and 46,400 alignment
bytes for a 20,402,726,912-byte allocation. Eight whole-model decode graphs chain each layer's
BF16 publication directly into the next owner. The real checkpoint passes all eight eager/graph
routes, 76,032 represented endpoint-oracle values, 8,939,520 finite logits, inactive-row and
replacement-input checks, all 41 stable addresses, zero post-warmup growth, and the complete
downstream spill/resource cone. An unblessed direct diagnostic at an observed 2,100--2,137 MHz SM
and 13,801 MHz memory clock measures 5.051/9.364 ms at B=1/8, or about 198/854 aggregate rows per
second. Production-graph and repeated complete-path medians agree within 0.01%, and the timed
region has zero device-memory growth. Native prompt graphs will be reconciled with the
resident-prefill stack after that stack lands.

The text frontend separately admits the snapshot's 248,070-entry tokenizer, Qwen3.6 chat template,
ordered stop IDs `[248046, 248044]`, and sampled defaults `temperature=1`, `top_p=0.95`, and
`top_k=20`. Thinking and no-thinking `Hello` prompts match the retained Transformers 5.2.0 token
fixtures exactly; this contract is not aliased to Qwen3.5 even though their tokenizer files match.
A concrete single-slot generation owner now joins that frontend to the complete resident model.
For the exact thinking-mode `Hello` prompt, production streaming and separately driven raw-token
transitions select the same two greedy tokens `[8160, 579]`; reset replay is deterministic, all 42
device/host addresses remain stable, and device memory does not grow after warmup. This is
frontend-to-device state-transition evidence, not an external same-model logit-parity claim. The
initial route serially evaluates prompts through B=1 decode and loudly enforces its 192-position
capacity until the native resident-prefill stack is reconciled.
The server now selects this concrete target from the pinned revision directory and publishes
`nvidia/Qwen3.6-35B-A3B-NVFP4` through the OpenAI health, models, blocking chat, and SSE routes. A
real localhost thinking-mode `Hello` request emits the two-token reasoning text `Here's`; blocking
and SSE responses both report 11 prompt tokens, two completion tokens, `finish_reason=length`, and
the exact model identity. `xtask qualify-qwen36-server` makes those public-boundary checks
repeatable and stops only the server child it starts. Startup exposes one slot and the same
192-position limit rather than implying compact batching or native prefill support.

## Implementation order

1. Add the pinned architecture profile and validate both `config.json` and
   `hf_quant_config.json`.
2. Extend the exact snapshot owner to three named shards and admit all 124,468 tensor descriptors
   bijectively, including header lengths, file lengths, and index ownership.
3. Add typed MoE bindings and lossless per-layer materialization into numeric expert-major layout.
4. Qualify router/top-8 behavior and the represented NVFP4 expert/shared-expert operation at every
   exact `B=1..8` route, then establish spill/resource and direct performance evidence.
5. Add the narrower GDN and full-attention routes, reusing recurrence code only after its complete
   oracle proves identical arithmetic.
6. Compose and qualify one real linear-attention layer, one real full-attention layer, the NVFP4
   endpoint, and then the 40-layer text owner.
7. Start with one resident slot and a fixed shared KV budget. Compact batching follows only after
   expert dispatch, recurrent state, and KV ownership have exact physical-slot mappings.
8. Admit the separate chat template and serve the pinned model identity. Add optimized prefill,
   MTP, and Vision as separately qualified slices.

Each step reruns the existing Qwen3.8 and Qwen3.5 consumers in its dependency cone. A shared helper
is extracted only when the second concrete route proves identical layout, represented values, and
accumulation order.

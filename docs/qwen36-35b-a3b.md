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
remain a separate exact projection. Every `B=1..8` route passes 73,728 exact activation-code,
444,672 FP64-formula output, 518,400 graph-replay, and 806,400 inactive-sentinel comparisons with
immutable sources and no post-warmup device growth. Its complete three-node path measures
12.443/33.309 us at B=1/8 and a locked 2,197 MHz SM clock. These are unblessed leaf diagnostics;
the next control/convolution and FP32 recurrence stages reuse the already qualified Qwen3.5 binary
entries because both profiles have the exact same 32 control rows, 8,192 Q/K/V rows, 4,096 value
rows, 16 Q/K heads, 32 value heads, width-128 state, and width-four history. Compile-time geometry
assertions and separate Qwen3.6 oracle entry points make that reuse executable rather than
conventional. The final GDN projection preserves its source E4M3 `[2048,4096]` plane and static
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
post-warmup growth. At a diagnostic 2,167 MHz median SM clock, its unblessed warm repeated path
measures 8.064/21.350 us at B=1/8; the production graph boundary measures 10.240/22.544 us. Q/K
normalization and RoPE, BF16 cache ownership, attention, gated output, and the full layer owner
remain separate qualification slices.

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

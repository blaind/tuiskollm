# Qwen3.8-Flash-Next source contract

This document covers `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

The implemented slice binds and materializes the base model's text, Vision, routed-expert,
hyper-connection, sparse-attention, gated DeltaNet, and engram sources. It does not bind MTP and
does not constitute model inference support.

## Represented source values

The checkpoint has three distinct quantization conventions:

- Routed experts use ModelOpt NVFP4. Each projection stores packed E2M1 weights as U8, E4M3 block
  scales, and rank-zero F32 `input_scale` and `weight_scale_2` values.
- Unquantized text and Vision tensors are BF16. Sparse-attention, gated DeltaNet, shared-expert,
  hyper-connection, endpoint, and engram projection tensors remain in this class.
- The engram table stores E4M3 codes in 128 shards with one BF16 multiplier. The multiplier is a
  plain scale, not a ModelOpt reciprocal divisor.

The engram also stores three I64 hash buffers. Admission independently recomputes the SplitMix64
multipliers, prime head vocabularies, and exclusive offsets, then requires every source word to
match. The table has 320,001,446 addressable rows padded to 320,001,536 physical rows.

## Lossless materialization

Materialization may reorder bytes for a runtime layout, but never decodes and requantizes source
values.

| Source plane | Runtime treatment |
| --- | --- |
| Routed-expert packed E2M1 weights | Borrowed without copying |
| Routed-expert E4M3 block scales | Byte-permuted into `BlockScaleK16M128x4` order |
| ModelOpt F32 scalars | Source values retained; reciprocal divisors derived separately |
| Engram E4M3 table | Borrowed shard-by-shard without decoding |
| Engram BF16 multiplier | Exact BF16 bits retained and widened directly |
| BF16 GDN and sparse-attention inputs | Gathered into the fused runtime projection order |
| Other BF16 planes | Borrowed without copying |

Routed experts are described in numeric expert order even though shard payloads use lexicographic
tensor order. One layer stages 157,286,400 bytes of expert scales while borrowing 1,258,291,200
bytes of packed expert weights. The engram table borrows 51,200,245,760 bytes. If every base layer
were materialized simultaneously, the staged scale and BF16 gather planes would total
11,405,230,080 bytes; production ownership may impose a smaller lifetime.

The engram multiplier in this checkpoint has BF16 source bits `0x3951`. It is admitted as a finite,
positive multiplier and never passed through the ModelOpt scalar conversion.

## Deferred boundary

`mtp.*` remains inventory-admitted but has no `Qwen38FlashNextMtpBindings` or materialized layout.
The ignored acceptance test
`qwen38_flash_next_mtp_block_binds_its_fused_bf16_expert_pool` names that missing condition.

Synthetic tests cover each source family, represented scalar bits, scale permutations, engram hash
constants, EOS segment behavior, borrowed table addressing, and staged-byte accounting. The ignored
snapshot tests require `TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT` and exercise the pinned complete source.

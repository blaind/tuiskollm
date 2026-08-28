# TuiskoLLM

<p align="center">
  <img src="assets/tuiskollm-hero.webp" alt="TuiskoLLM" width="100%">
</p>

<p align="center">
  <strong>Exact-model inference, built all the way down for the RTX 5090.</strong>
</p>

<p align="center">
  Rust · NVFP4 · CUDA Graphs · OpenAI-compatible HTTP
</p>

TuiskoLLM specializes the full inference stack for exact checkpoint × GPU targets and fails closed
on unsupported combinations. Its OpenAI-compatible server and CUDA kernels are Rust, built with
[cuda-oxide](https://github.com/NVlabs/cuda-oxide). Releases are single stripped executables with
embedded device code: no PyTorch, Triton, JIT, model conversion, or runtime CUDA Toolkit.

## Download and run

**[Download the Linux x86-64 binary](https://github.com/blaind/tuiskollm/releases)**

Requires Linux/glibc 2.34+, an NVIDIA driver, and an RTX 5090. Download the archive, then run:

```bash
tar -xzf tuiskollm-*-linux-x86_64*.tar.gz
cd tuiskollm-*-linux-x86_64*/
./tuiskollm serve unsloth/Qwen3.8-27B-NVFP4
```

TuiskoLLM resolves the selected model's pinned Hugging Face revision, downloading and verifying any
missing files. Pass `--snapshot PATH` to use an explicit checkpoint directory.

After validating and loading the checkpoint, the server listens on `127.0.0.1:8000` by default:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [{"role": "user", "content": "Reply with hi."}],
    "max_tokens": 32
  }'
```

## Models and performance

| Model | Served route | Context boundary | RTX 5090 decode (B=1 / B=8 aggregate) |
| --- | --- | ---: | --- |
| [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Text + draft-three MTP · 8 slots | 220,000 | 56.6 / 380.4 tok/s @ 131<br>36.7 / 261.5 tok/s @ 131,073 |
| [`AxionML/Qwen3.5-9B-NVFP4`](https://huggingface.co/AxionML/Qwen3.5-9B-NVFP4) | Text + source-BF16 MTP · 8 slots | 262,144 rounded | Not yet blessed |
| [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | Text · compact B=1..8 | 262,144 | Not yet blessed |

Decode rates are controlled-clock target-graph medians at the measured context; see
[`docs/performance.md`](docs/performance.md) for methods and checked baselines.

RTX 5090 / SM120 is the complete product target. SM89 and SM86 remain partial qualification
targets, not fallback servers. Vision is not served yet.

## What the server owns

- Source-native represented weights: no decode-and-requantize checkpoint conversion.
- Address-stable resident arenas and immutable whole-model CUDA Graphs.
- Exact compact scheduling for every admitted B=1..8 route.
- Paged KV ownership, retained-prefix reuse, cancellation, and retryable overload.
- Blocking and SSE `POST /v1/chat/completions`, plus `GET /health`, `GET /ready`, and
  `GET /v1/models`.
- Qwen3.8 lifecycle administration: `/v1/unload`/`load` release and recreate the complete model;
  `/v1/park`/`resume` preserve retained-prefix KV and recurrent state in pinned host memory while
  releasing all target/MTP arena backing. Resume reloads represented weights from the admitted
  snapshot at the retained graph addresses before restoring the mirror. Exact-model park/resume
  device qualification remains pending.
- Request logs with queue-inclusive latency, TTFT, decode rate, cache reuse, and route identity.

## Build from source

Building requires Linux, Git, rustup, CUDA Toolkit 13.3.73, Clang/libclang 21, the NVIDIA driver,
and an RTX 5090. Rust is pinned by `rust-toolchain.toml`.

```bash
cargo run -p xtask -- bootstrap-cuda-oxide
cargo run -p xtask -- build-server
```

The executable is written to `target/cuda-oxide-build-sm120/release/tuiskollm`. A plain
`cargo build` does not finalize the embedded device artifacts. Start it by selecting one exact
model, for example:

```bash
target/cuda-oxide-build-sm120/release/tuiskollm serve \
  unsloth/Qwen3.8-27B-NVFP4
```

`--address ADDRESS` overrides the default `127.0.0.1:8000` listener.
The lifecycle routes are enabled only for the exact Qwen3.8 target. A non-loopback listener also
requires `--admin-token-env NAME`; requests then authenticate with `Authorization: Bearer ...`.
Park is accepted only after previously admitted chats drain. The first chat submitted while parked
starts one resume attempt; concurrent chats pend behind that same attempt. Explicit `/v1/resume`
remains available for administrative warmup.

For qualification commands and engineering constraints, see
[`docs/performance.md`](docs/performance.md) and [`AGENTS.md`](AGENTS.md). The optional
[`tuisko-llm` Python distribution](python/) exposes the admitted tokenizer and chat-template
frontend; its standards-normalized wheel and sdist filenames use `tuisko_llm`. It is not an
in-process inference API.

## Status

TuiskoLLM is experimental and under active development. Performance, supported targets, and APIs
may change before 1.0.

## License

MIT OR Apache-2.0, at your option. See [`LICENSE-MIT`](LICENSE-MIT) and
[`LICENSE-APACHE`](LICENSE-APACHE).

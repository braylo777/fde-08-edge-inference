# fde-08-edge-inference

**Forward Deployed Engineering — Stage 8 of 12: Edge & On-Prem Local Inference**

Decide what runs on the customer's own hardware: read the model file, pick a quantization that fits, keep quality with activation-aware int4, and route requests safely when the network or a server is down.

**Live demo:** https://fde-08-edge-inference.braylo777.workers.dev — the Rust core runs in your browser as WebAssembly. No server, no tracking.

## Why this matters

Some customers cannot send a single token to a cloud API: a factory floor with no internet, a hospital, a defense site. The FDE then owns hard, physical questions. Will an 8B model fit on this 16 GB card with 32k context? How much quality does 4-bit cost? What happens when the on-prem GPU box goes down? This repo answers each with code and numbers, not guesses.

## What it does

- GGUF v2/v3 header reader: metadata, tensor table, quantization mix, parameter count, bits per weight; bounds-checked against truncated or hostile files
- Int4 group-wise RTN quantization and AWQ-style activation-aware scaling with an alpha sweep on calibration data
- Memory planner: weights + KV cache (f16 / fp8 / q4) + overhead for 8 formats, best fit and max context
- Offline-first router: local PII screen before any network decision, rule-based complexity triage, health / latency / network fallbacks, queue instead of policy breach
- CLI: gguf_file <model.gguf> reads only the header of a real multi-GB model

## What it does not do (yet)

- No vLLM / TensorRT-LLM / llama.cpp runtime here: this repo plans and routes; it does not serve tokens
- The AWQ demo uses a synthetic layer with planted outlier channels, not a real model checkpoint
- Triage is rule-based; a production system would use a small local classifier model
- No GPU provisioning (drivers, CUDA, MIG partitioning)

## Operations

Every operation takes JSON in and gives JSON out. The CLI, the tests and the demo page all use the same entry point.

| Operation | What it does |
|---|---|
| `gguf_inspect` | A synthetic header shaped like an 8B Llama-3-class model at Q4_K_M (Q4_K with some Q6_K layers). Parameter count and bits per weight come from the tensor table. |
| `quantize_compare` | A 64x256 layer where 4 input channels carry 20x larger activations. AWQ scales those weight columns before 4-bit rounding. Set outlier_channels to 0 and the gain disappears. |
| `memory_plan` | An 8B model with GQA (8 KV heads) on a 16 GB card, 32k context, one user: Q8_0 fits. Set concurrent_sequences to 4 and nothing fits until you use kv_cache_dtype fp8 or q4_0. |
| `route` | The on-prem server is down. A hard request with patient data cannot go to the cloud, so it is queued. Remove the patient details and it falls back to the cloud. |

## Run it

You need Rust (stable). No Node, no npm.

```bash
git clone https://github.com/braylo-ai/fde-08-edge-inference
cd fde-08-edge-inference
cargo test
cargo run -- ops
```

Try the first demo example from the command line:

```bash
python3 -c 'import json;print(json.dumps(json.load(open("web/examples.json"))["ops"][0]["input"]))' | cargo run -q -- gguf_inspect
```

## Inspect a real model file

```bash
cargo run --release -- gguf_file ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Only the first 64 MB are read (header + tensor table), so it is instant even for a 40 GB file.

## Build and deploy the demo

```bash
./build.sh    # tests, then compiles the core to web/app.wasm
./deploy.sh   # serves web/ as a Cloudflare Workers static-assets site (needs a Cloudflare login)
```

`web/` is a plain static site (`index.html`, `app.js`, `app.wasm`). It has no build step, so any static host works.

## Layout

```
src/lib.rs        operation list and router
src/api.rs        JSON entry point + the 3 WASM exports (alloc, dealloc, run)
src/main.rs       CLI
src/*.rs          one module per capability, each with unit tests
tests/            integration tests (every demo example must pass)
web/              demo page; examples.json drives both the page and this README
```

## Design choices

- **Rust, few dependencies.** Only `serde`, `serde_json`, `sha2` and `hmac`, pinned to exact versions with a committed `Cargo.lock`.
- **Deterministic.** Time, randomness and keys are inputs, so every result can be replayed in a test or an audit.
- **Fail closed.** Unknown input, missing fields or an unmatched policy gives an error or a denial, never a silent pass.

## Part of a series

This is one of 12 repos that work through a public Forward Deployed Engineering roadmap ([source post](https://x.com/suraj_sharma14/status/2124827808535048566)). All 12: [github.com/braylo-ai?q=fde-](https://github.com/braylo-ai?tab=repositories&q=fde-).

## License

MIT — see [LICENSE](LICENSE). Built by [Braylo SI](https://braylo.si).

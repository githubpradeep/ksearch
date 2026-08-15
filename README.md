# ksearch

Run **Gemma 4** on Apple Silicon with kernels that are **compiled**, not hand-pasted as Metal shaders.

ksearch is a local Metal inference stack: load a GGUF, generate text, or serve an OpenAI-compatible chat API. Under the hood it lowers the model to a tinygrad-shaped IR, emits MSL from that IR, and uses BEAM search to pick tilings — so you get a maintainable compiler path instead of a growing pile of named `.metal` kernels.

**Requires:** macOS on Apple Silicon, Rust (`cargo`), and a dense Gemma 4 GGUF (E2B or E4B).

## Why use it

- **Local Gemma 4 on Mac** — Q4_K-style GGUFs, chat template, generate or HTTP serve
- **Compiler, not a shader farm** — ops expand to primitives; Metal is rendered from an AST
- **Dense E2B and E4B** — sizes and GQA/MQA come from GGUF metadata (MoE A4B is not supported yet)
- **OpenAI-style server** — `/v1/chat/completions` for tools and apps that already speak that API

If you want the fastest possible hand-tuned Metal Gemma stack today, compare against projects that ship expert shaders. ksearch’s bet is the same math with a searchable, regenerable kernel pipeline.

## Install

```bash
git clone https://github.com/githubpradeep/ksearch.git
cd ksearch
cargo build -p ksearch_cli --release
```

Put a GGUF somewhere convenient, for example:

- `~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf`
- `~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf`

First load warms BEAM plans (can take a minute). Later runs reuse `~/.cache/ksearch/beam` unless you force a re-search.

## Generate text

```bash
cargo run -p ksearch_cli --release -- generate \
  --gguf ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --prompt "Hi" --n-predict 32 --max-seq 64
```

Same command works for E4B — pass the E4B GGUF path. `--prompt` is wrapped in the Gemma 4 chat template and BPE-encoded.

## Chat server

```bash
cargo run -p ksearch_cli --release -- serve \
  --gguf ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
  --port 8080 --max-seq 16384 --slots 4
```

Then:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

`--slots` is concurrent KV capacity (like llama.cpp `--parallel`). GPU work is still serial; slots hold separate caches.

## Supported models

| Model | Status |
|-------|--------|
| Gemma 4 **E2B** (dense, MQA) | Supported |
| Gemma 4 **E4B** (dense, GQA) | Supported |
| Gemma 4 **A4B** (MoE) / vision | Not supported |
| Other architectures | Not supported |

## Check that it works

```bash
# Short correctness + tok/s gate (default GGUF path is E2B; override with --gguf)
cargo run -p ksearch_cli --release -- bench

# Optional: tiny Metal smoke tests
cargo run -p ksearch_cli --release -- elem-add --n 1048576
cargo run -p ksearch_cli --release -- matvec --rows 4096 --cols 4096 --beam
```

`bench` expects a `"Hi"` reply that contains `Hi!` and `help`.

## Useful flags and env

| CLI | Meaning |
|-----|---------|
| `--gguf` | Path to Gemma 4 GGUF |
| `--max-seq` | KV length to allocate (serve default 16k; model ctx can be larger) |
| `--slots` | Concurrent sequences for `serve` |
| `--n-predict` | Max new tokens for `generate` |

| Env | Effect |
|-----|--------|
| `KSEARCH_SKIP_BEAM` | Skip BEAM warmup (faster first load, slower/untuned matvecs) |
| `KSEARCH_BEAM_FORCE` | Ignore cached plans and search again |
| `KSEARCH_BEAM_CACHE` | Plan cache dir (default `~/.cache/ksearch/beam`) |
| `KSEARCH_PROFILE` | Per-section GPU timings |
| `KSEARCH_CHIP` | Plan-cache chip key (default: Metal device name) |

## How it works (short)

```
GGUF → Graph (primitives + schedule hints)
    → Kernel IR AST → MSL → Metal
    → prefill / decode with Q4_0 KV cache
```

Weights stay packed (Q4_K / Q6_K); dequant happens at load in the generated kernels. Design notes: [docs/DESIGN.md](docs/DESIGN.md).

## Dig deeper

- [docs/README.md](docs/README.md) — curriculum if you want to reimplement or contribute
- [docs/07-gemma-runtime.md](docs/07-gemma-runtime.md) — load / decode / serve internals
- [docs/FINDINGS.md](docs/FINDINGS.md) — why this shape vs hand kernels / other compilers

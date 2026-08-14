# ksearch

A **Metal kernel compiler** for Gemma-class LLM inference on Apple GPUs.

ksearch does not ship a farm of hand-written `rmsnorm.metal` / FlashAttention / Q4 fusion shaders. Models lower to a tinygrad-shaped IR; kernels are **generated** from that IR; **BEAM search** picks tilings. Speed is measured against `reference/metal-llm-server` (the oracle scoreboard), not copied from it.

```
primitives → scheduler invents CALL boundaries → Kernel IR AST
  → generic MSL renderer (from AST only) → BEAM tilings → Metal
```

**Requires:** macOS with Apple Silicon (Metal). Rust stable (`cargo`).

## Quick start

```bash
# Sanity: generate an add kernel, compile it, check CPU vs GPU
cargo run -p ksearch_cli --release -- elem-add --n 1048576

# Generate a matvec kernel (optionally BEAM-search tilings)
cargo run -p ksearch_cli --release -- matvec --rows 4096 --cols 4096 --beam

# Run Gemma 4 E2B or E4B from a GGUF (dense; sizes come from metadata)
cargo run -p ksearch_cli --release -- generate \
  --gguf ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --prompt "Hi" --n-predict 32 --max-seq 64

cargo run -p ksearch_cli --release -- generate \
  --gguf ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
  --prompt "Hi" --n-predict 32 --max-seq 64

# Correctness + tok/s regression (Hi gate + essay; default GGUF is E2B)
cargo run -p ksearch_cli --release -- bench

# OpenAI-compatible HTTP server (point --gguf at E2B or E4B)
cargo run -p ksearch_cli --release -- serve \
  --gguf ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf --port 8080
```

Dense Gemma 4 only: **E2B** (MQA, `n_kv=1`) and **E4B** (GQA, `n_kv=2` in current GGUFs). MoE **A4B** is out of scope. `bench` prints Hi pass/fail plus prefill/decode tok/s; the Hi gate expects the reply to contain `Hi!` and `help`.

## What this project is (and is not)

| Is | Is not |
|----|--------|
| A compiler: Graph → schedule → Kernel IR → MSL | A catalog of named fused Graph ops (`Op::MatVecQ4K`, `Op::FlashAttn`) |
| Tinygrad-shaped: `matmul` is mul+sum, RMSNorm is `x * rsqrt(mean(x²)+eps) * w` | A paste of metal-llm-server `.metal` shaders |
| Q4_K/Q6_K weights stay packed; the renderer expands `Load(dtype=Q4K)` to float | Hand `q4k_load` helpers pasted as named matvec kernels |
| BEAM searches **tilings** (`tg` / `vec` / `unroll` / `nr0`) on an already-scheduled AST | Search that invents FlashAttention from primops |

If you want the research that locked this thesis, start at [docs/FINDINGS.md](docs/FINDINGS.md). The design lock is [docs/DESIGN.md](docs/DESIGN.md).

## Study path (start here)

A beginner who wants to **reimplement this** should read in order:

0. **Basics (diagrams + video)**
   - [docs/00-transformers.md](docs/00-transformers.md) — decoder block, attention, KV cache
   - [docs/00-metal.md](docs/00-metal.md) — threads, buffers, command encoder
   - [docs/00-gemma-architecture.md](docs/00-gemma-architecture.md) — SWA, PLE, sequences
   - [docs/animations/README.md](docs/animations/README.md) — Manim scenes (`manim -pql docs/animations/scenes.py TransformerBlock`)
1. [docs/README.md](docs/README.md) — full curriculum
2. [docs/01-mental-model.md](docs/01-mental-model.md) — kernel compiler pipeline
3. [docs/02-graph-ir.md](docs/02-graph-ir.md) through [docs/08-implement-from-scratch.md](docs/08-implement-from-scratch.md)

Interactive diagrams (open beside chat): [ksearch basics](/Users/PRADEEP.BORADO/.cursor/projects/Users-PRADEEP-BORADO-Documents-misc-ksearch/canvases/ksearch-basics.canvas.tsx)

## Crate map

```
ksearch/
  crates/
    ksearch_ir        Graph primitives, FuseHint, Kernel IR, OptSchedule
    ksearch_codegen   schedule, rewrite, lower, render, BEAM, plan cache
    ksearch_metal     device, compile, buffers, launch
    ksearch_kernels   Eng: build Graph → lower_to_metal → run
    ksearch_gguf      mmap GGUF + tokenizer + CPU dequant helpers
    ksearch_gemma     GemmaPrimModel (Gemma 4 forward)
    ksearch_cli       elem-add / matvec / generate / bench / serve
  docs/               design + beginner curriculum
  reference/          tinygrad, luminal, metal-llm-server (local study copies)
```

Data flow at inference time:

```
GGUF mmap
  → GemmaPrimModel builds Graphs for each op (via Eng)
  → schedule invents one kernel
  → lower builds KirStmt/KirExpr AST
  → renderer emits MSL (F16/Q4K Load expand to float)
  → Metal compiles + launches
  → KvPool / chunked prefill / decode
```

## Environment

| Variable | Effect |
|----------|--------|
| `KSEARCH_CHIP` | Plan-cache chip key (default: Metal device name) |
| `KSEARCH_BEAM_FORCE` | Ignore cached BEAM plans and search again |
| `KSEARCH_BEAM_CACHE` | Directory for on-disk schedules (default `~/.cache/ksearch/beam`) |
| `KSEARCH_PROFILE` | Per-section GPU timings in `GemmaPrimModel` |

## References

- `reference/tinygrad` — UOp `Ops`, scheduler CALLs, BEAM OptOps, MetalRenderer
- `reference/luminal` — e-graph architecture (do **not** copy FlashInfer matching)
- `reference/metal-llm-server` — tok/s and Gemma semantics scoreboard only

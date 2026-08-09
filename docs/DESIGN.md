# ksearch design — Thesis A (compiler-pure)

**Status:** locked for implementation (2026-08)  
**Findings:** [FINDINGS.md](./FINDINGS.md)

---

## Thesis

Build a Metal inference **compiler**: models lower to an IR; kernels are **generated** from that IR; **search** picks schedules (and later fusion equivalents). Measure against metal-llm-server. Do **not** paste oracle shaders as the product.

MLX shows Mac LLM inference can be strong without FlashInfer-style flash theater. Thesis A asks: how far can **generated** kernels go on the same insight (matvecs + fusion + decent attention), with search instead of a hand farm.

---

## Non-goals (v1)

- Training / autograd
- CUDA / Triton (no Triton on Metal)
- Calling FlashInfer or shipping metal-llm-server MSL as hot path
- Matching oracle tok/s on day one
- A4B MoE / vision

---

## Architecture

```
GGUF / tensors
    → Frontend builds Graph (lazy ops)
    → Scheduler: kernel boundaries (fuse vs materialize)
    → Per-kernel: lower to UOp-like body → BEAM tile search → render MSL → MTLLibrary
    → Plan cache (model hash + chip + dim bucket)
    → Runtime execute (later: KvPool + scheduler for B≥1)
```

| Layer | Role |
|-------|------|
| **Graph IR** | Tensor ops with symbolic dims `b`, `s`, `c` |
| **Kernel IR** | Loop/axis typed body after scheduling |
| **Search** | Discrete OptOps (LOCAL, UPCAST, UNROLL, TG size, …); time on device; disk cache |
| **Renderer** | Kernel IR → MSL string |
| **Runtime** | Buffers, CB encode, plan load |

---

## Attention & quant (Thesis A stance)

| Topic | Stance |
|-------|--------|
| **Attention v1** | Naive SDPA from matmul+softmax **generated** kernels. No flash library. Improve via fusion + schedule; add flash-**structure** only as IR rewrites that still **codegen** MSL (not a pasted blob). |
| **Quant v1→v2** | Start f16 (or dequant-to-f16) for correctness. Then **Q4_K as IR dtype** so fused dequant+matvec can be scheduled/searched — the real decode fight. |
| **Serving** | Types for `SlotId` + `b` from early on; KvPool after generate works. |

---

## Success metrics

| Gate | Metric |
|------|--------|
| P0 | Generated Metal kernel from IR runs; matches CPU |
| P1 | BEAM improves dense matvec/matmul over untuned |
| P2 | Gemma (f16 or dequant) generates coherent text vs oracle meaning |
| P3 | Fusion cuts launch count; report tok/s vs oracle |
| P4 | Q4_K in IR; fused path under search; close bandwidth gap |
| Product | KvPool / B≥1; never claim “discovered FA” unless FA structure is codegen’d from IR |

Hand MSL allowed only as **temporary debt** with a removal ticket.

---

## Crate layout

```
ksearch/
  Cargo.toml                 # workspace
  crates/
    ksearch_ir/              # graph + kernel IR types
    ksearch_codegen/         # lower, BEAM, MSL render
    ksearch_metal/           # device, compile, buffers, launch
    ksearch_gguf/            # mmap loader (later)
    ksearch_gemma/           # model builder (later)
    ksearch_cli/             # benches + generate
  docs/
  reference/                 # tinygrad, luminal, metal-llm-server
```

---

## Implementation phases

| Phase | Deliverable | Kill / note |
|-------|-------------|-------------|
| **P0** | IR → MSL for add/mul/sum; Metal run; CPU check | ✅ |
| **P1** | Matmul as mul+sum; BEAM tiles; cache | ✅ matvec BEAM |
| **P2** | Minimal Gemma graph f16; generate text | ✅ full decode+PLE; chat reply OK |
| **P3** | Stronger fusion / e-graph lite | Launch tax |
| **P4** | Q4_K dtype + fused search | Vs oracle bandwidth |
| **P5** | KvPool + batch buckets | Server shape |

---

## Oracle policy

**Allowed:** GGUF layout knowledge, Gemma math/attrs, benchmarks, serving ideas.  
**Forbidden as hot path:** copying `shaders/*.metal` fusion stacks into ksearch.

---

## References

- `reference/tinygrad` — UOp, BEAM, MetalRenderer  
- `reference/luminal` — egglog, dim buckets (steal mechanisms, not FlashInfer lie)  
- `reference/metal-llm-server` — tok/s + correctness bar  
- MLX — existence proof for strong Mac LLM without FA theater

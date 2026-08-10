# ksearch design — Thesis A (compiler-pure)

**Status:** locked (2026-08) — tinygrad-only IR purge complete  
**Findings:** [FINDINGS.md](./FINDINGS.md)

---

## Thesis

Build a Metal inference **compiler**: models lower to an IR; kernels are **generated** from that IR; **search** picks schedules. Measure against metal-llm-server. Do **not** paste oracle shaders as the product.

ksearch is **tinygrad-shaped**: Tensor/nn sugar expands to ALU + movement + REDUCE; the scheduler invents kernel boundaries; BEAM searches OptOps-like tilings. There is **no** fused-kernel `Op` catalog.

---

## Tinygrad IR map (studied in `reference/tinygrad`)

| Layer | Role |
|-------|------|
| **Tensor / nn sugar** | `matmul`, `gelu`, `RMSNorm`, `scaled_dot_product_attention` — **not** the IR product |
| **UOp DAG** | Real IR: ALU (`ADD`/`MUL`/`…`), movement (`RESHAPE`/`EXPAND`/…), `REDUCE`+arg, `LOAD`/`STORE` |
| **Scheduler** | Invents CALL / kernel boundaries |
| **BEAM OptOps** | `LOCAL` / `UPCAST` / `UNROLL` / … on an already-scheduled AST |

There is **no** `Ops.MATMUL`, `Ops.RMSNORM`, `Ops.FLASH_ATTN`, `Ops.GELU` in tinygrad’s UOp enum.

| Sugar | Expansion |
|-------|-----------|
| matmul / dot | reshape/transpose + **mul + sum** |
| RMSNorm | `x * rsqrt(mean(x²)+eps) * w` |
| gelu (tanh) | `0.5 * x * (1 + tanh(…))` |
| SDPA | `Q@Kᵀ` → softmax → `@V` (no FlashAttention) |

### ksearch mapping

| tinygrad | ksearch |
|----------|---------|
| Tensor sugar | `Graph` helpers (`rmsnorm_expand`, `gelu_tanh`, `softcap`, `matvec_prim`, `sdpa_naive`) |
| UOp ALU + movement + reduce | `Op::{Add,Mul,ScaleConst,Rsqrt,Tanh,Exp,SumReduce,Expand,…}` |
| Scheduler fusion | `schedule` → `KernelKind` / `KirBody` (incl. fused RMSNorm/GeLU/RoPE regions) |
| OptOps BEAM | `OptSchedule` TG/VEC/UNROLL (+ Q4 NSG/NR0) |
| Q4 | **dtype** on BUFFER/Input; dequant fused at matvec **render** |

**Deleted:** `debt/`, `GemmaModel`, `KSEARCH_DEBT`, catalog Ops (`MatVecQ4K*`, `AttnGqa*`, …).

---

## Non-goals (v1)

- Training / autograd
- CUDA / Triton (no Triton on Metal)
- Calling FlashInfer or shipping metal-llm-server MSL as hot path
- Matching oracle tok/s on day one
- A4B MoE / vision
- Reintroducing fused catalog Ops to chase tok/s

---

## Architecture

```
GGUF / tensors
    → Frontend builds Graph (primitive ops + sugar expand)
    → Scheduler: kernel boundaries (fuse vs materialize)
    → Per-kernel: Kernel IR → BEAM OptSchedule → render MSL → MTLLibrary
    → Plan cache (model hash + chip + dim bucket)
    → Runtime execute (KvPool for serving)
```

| Layer | Role |
|-------|------|
| **Graph IR** | Primitives only (ALU / movement / reduce / load) |
| **Kernel IR** | Scheduled body (`KirBody`); layer sugar may fuse to one region |
| **Search** | OptOps-like discrete schedules; time on device; disk cache |
| **Renderer** | Kernel IR → MSL string |
| **Runtime** | Buffers, CB encode (`GemmaPrimModel` + `Eng`) |

---

## Attention & quant

| Topic | Stance |
|-------|--------|
| **Attention** | Naive SDPA (`SdpaNaive`) = matmul+softmax composed; no flash catalog |
| **Quant** | Q4_K as IR **dtype**; fused dequant+matvec at render under BEAM |
| **Serving** | `KvPool::new_f32`; chunked prefill in `generate_timed` |

---

## Success metrics

| Gate | Metric |
|------|--------|
| P0 | Generated Metal kernel from IR runs |
| P1 | BEAM improves dense matvec over untuned |
| P2 | `"Hi"` → text contains `Hi!` and `help` on GemmaPrim |
| P3 | Schedule rewrites / sugar fusion (not catalog Ops) |
| P4 | Q4_K dtype fusion; no debt Eng |
| Product | KvPool / B≥1; never claim “discovered FA” |

Hand MSL is **not** architecture. Layer `KirBody` fusion is scheduled sugar (tinygrad CALL regions), not a Graph `Op` zoo.

---

## Crate layout

```
ksearch/
  crates/
    ksearch_ir/              # Graph primitives + Kernel IR + OptSchedule
    ksearch_codegen/         # schedule, rewrite, BEAM, render, layer Kir builders
    ksearch_metal/           # device, compile, buffers, launch
    ksearch_kernels/         # Eng (Thesis A methods only)
    ksearch_gguf/            # mmap loader
    ksearch_gemma/           # GemmaPrimModel only
    ksearch_cli/             # benches + generate
  docs/
  reference/                 # tinygrad, luminal, metal-llm-server (oracle scoreboard)
```

---

## Implementation status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **P0** | Graph → schedule → Kernel IR → MSL | ✅ |
| **P1** | OptOp BEAM + disk cache | ✅ |
| **P2** | GemmaPrim `"Hi"` gate | ✅ |
| **P3** | Schedule rewrites / sugar expand | ✅ |
| **P4** | Q4_K dtype fusion; **debt deleted** | ✅ |
| **P5** | KvPool + chunked prefill | ✅ |
| **Purge** | Tinygrad-only IR; no `GemmaModel` / `debt/` | ✅ |

---

## Oracle policy

**Allowed:** GGUF layout, Gemma math/attrs, benchmarks, serving structure hints.  
**Forbidden as hot path:** copying `shaders/*.metal` fusion stacks; catalog Ops named after oracle kernels.

---

## References

- `reference/tinygrad` — UOp `Ops` / `GroupOp`, BEAM OptOps, MetalRenderer, `nn.RMSNorm`, `mixin/op.py` dot  
- `reference/luminal` — egglog (do **not** copy FlashInfer matching)  
- `reference/metal-llm-server` — tok/s + correctness scoreboard only  
- MLX — Mac LLM without FA theater

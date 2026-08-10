# ksearch design — Thesis A (compiler-pure)

**Status:** locked (2026-08) — tinygrad-only IR purge complete  
**Findings:** [FINDINGS.md](./FINDINGS.md)

---

## Thesis

Build a Metal inference **compiler**: models lower to an IR; kernels are **generated** from that IR; **search** picks schedules. Measure against metal-llm-server. Do **not** paste oracle shaders as the product.

ksearch is **tinygrad-shaped**: Tensor/nn sugar expands to ALU + movement + REDUCE; the scheduler invents CALL boundaries; **lower builds a Kernel IR AST**; the **renderer emits MSL only from that AST**; BEAM searches tilings. There is **no** fused-kernel `Op` catalog and **no** hand Metal templates (`rmsnorm.metal`, etc.).

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

### Quant (tinygrad CUDA parity)

tinygrad `ggml_data_to_tensor` turns Q4_K into a float Tensor, then typically `.half()`. ksearch: **CPU `dequant_to_f16_bytes` → F16 buffers → generic AST** (ALU in float, store half). Argmax index stays F32 (vocab ids). No `q4k_load`.

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
    → Frontend builds Graph (primitive ops + sugar expand + FuseHint)
    → Scheduler: invents CALL/kernel boundaries (hints + matvec/elemwise patterns)
    → Per-kernel: Kernel IR → OptSchedule (plan cache / BEAM) → render MSL → MTLLibrary
    → Runtime Eng execute (GemmaPrimModel + KvPool)
```

| Layer | Role |
|-------|------|
| **Graph IR** | Primitives (`Add`/`Mul`/`Reduce`/`Expand`/`Reshape`/`Permute`/…) + `Call` |
| **FuseHint** | Sugar metadata so scheduler invents fused `KirBody` (not Graph catalog Ops) |
| **Kernel IR** | Scheduled body; BEAM/plan cache picks `OptSchedule` |
| **Renderer** | Kernel IR → MSL |
| **Runtime** | `GemmaPrimModel` + `Eng` (Graph→lower only) |

---

## Attention & quant

| Topic | Stance |
|-------|--------|
| **Attention** | SDPA sugar → `Call` + `FuseHint::SdpaNaive` (Q@Kᵀ→softmax→@V fused at schedule) |
| **Quant** | Like tinygrad: dequant Q4→float then **`.half()` (F16)** buffers; generic AST kernels only |
| **Serving** | `KvPool::new_f16`; chunked prefill; B≥1 pool types ready |

---

## Success metrics

| Gate | Metric | Status |
|------|--------|--------|
| P0 | Generated Metal kernel from IR runs | ✅ |
| P1 | BEAM improves dense matvec over untuned | ✅ |
| P2 | `"Hi"` → `Hi!` + `help` on GemmaPrim | ✅ |
| P3 | Sugar expand + schedule FuseHint fusion | ✅ |
| P4 | Q4→F16 dequant (tinygrad `.half()`); no debt Eng; no `q4k_load` | ✅ |
| Arch | Eng = Graph→schedule only; no `Op::SdpaNaive` catalog | ✅ |
| Product | KvPool / B≥1 decode speed vs oracle | open (scoreboard; not blocking IR) |

Hand MSL is **not** architecture. `KirBody` fusion is scheduled sugar (tinygrad CALL), not a Graph `Op` zoo.

---

## Crate layout

```
ksearch/
  crates/
    ksearch_ir/              # Graph primitives + FuseHint + Kernel IR + OptSchedule
    ksearch_codegen/         # schedule, rewrite, BEAM, plan_cache, render, layer
    ksearch_metal/           # device, compile, buffers, launch
    ksearch_kernels/         # Eng (Graph → lower_to_metal only)
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
| **P3** | Sugar expand + FuseHint schedule fusion | ✅ |
| **P4** | Q4→F16 dequant (tinygrad `.half()`); debt deleted; no `q4k_load` | ✅ |
| **P5** | KvPool + chunked prefill | ✅ |
| **Purge** | Tinygrad-shaped IR; no `GemmaModel` / `debt/` | ✅ |
| **Arch wire** | Eng Graph-only; `Call`+hints; Reshape/Permute; plan_cache | ✅ |

**Still scoreboard work (not IR gaps):** F16 + FuseHints + **device-keyed F16 BEAM plan warm on load** are in. Dense F16 GEMV tilings are near local plateau (~4 tok/s); next real gain is LOCAL (threadgroup) staging of the matvec `x` vector, not more TG/VEC search.

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

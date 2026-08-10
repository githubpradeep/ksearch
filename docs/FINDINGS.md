# Research findings: kernel compilers & LLM inference

Survey for **ksearch** (2026-08). Local references: `reference/tinygrad`, `reference/luminal`, `reference/metal-llm-server`.

Interactive summary: Cursor canvas `ksearch-landscape` / `ksearch-design`.

---

## 1. Why this research

Goal: run Gemma 4 on Apple Metal with a **compiler that generates and searches kernels** (Hotz / tinygrad / luminal bet), measured against **metal-llm-server** (~45 tok/s E2B Q4_K on M1 Pro), without maintaining a hand fusion farm.

Question that kept coming up: if metal-llm-server already has hand FlashAttention + Q4_K, what does “search” buy? Answer depends on whether search means **tiling the same math** vs **inventing algorithms** vs **calling libraries**.

---

## 2. Four honest sources of GPU speed

| Source | What it is | Who leans on it |
|--------|------------|-----------------|
| **A. Dispatch / JIT** | Fewer launches, less host overhead (TinyJit, CUDA graphs, Metal ICB, megakernels) | tinygrad’s loudest claim |
| **B. Schedule search** | Same algorithm, better tiles (BEAM, Ansor, Triton autotune) | tinygrad BEAM, TVM |
| **C. Library / template match** | Pattern → FlashInfer / FA shell / cuBLAS | luminal “discover FA” |
| **D. Hand expert kernels** | Humans write Metal/CUDA | llama.cpp, MLX, metal-llm-server |

Marketing often blurs A–D. That is not fake benchmarks; it is **wrong race on the poster**.

---

## 3. tinygrad (studied in code)

**Pipeline:** Tensor → UOp graph → scheduler fuses CALLs → `hand_coded_optimizations` (default) or **BEAM** → render MSL → Metal compile.

**Matmul:** reshape + mul + sum (no hand GEMM body).

**Attention:** `scaled_dot_product_attention` = `Q@Kᵀ` → scale → mask → softmax → `@V`. **No FlashAttention.**

**Quant:** GGUF dequant expressed as Tensor ops; typically cast/realize to f16, then generic kernels — **not** fused Q4 Metal matvec codegen.

**What BEAM searches:** LOCAL, UPCAST, UNROLL, GROUP, TC, … — **tilings**, not fusion policy, not new algorithms.

**LLM speed without flash:** short-ctx BS=1 decode is mostly **weight matvecs**. JIT + decent GEMV can look fine; long ctx exposes naive attention. Community Metal numbers still trail llama.cpp on Q6_K until fused quant exists.

**Verdict:** Real codegen + schedule search. Does **not** prove “we beat hand Metal LLM servers.” Overclaim risk: selling framework/JIT wins as algorithmic LLM wins.

---

## 4. luminal (studied in code)

**Pipeline:** GraphTensor → HLIR (~15 primops) → egglog saturate → genetic genome search → profile → LLIR → CUDA/Metal.

**“Discover FlashAttention” in code:** egglog **pattern-matches** paged GQA attention → unions **`FlashInferAttention`** (hand CUDA library). Search picks FlashInfer vs naive HLIR. README in `luminal_cuda_lite/.../flashinfer/` states this explicitly.

**KernelOps:** many are still hand `format!` CUDA strings; search selects among them. Tile sizes mostly fixed constants, not free search.

**Gemma examples:** CUDA + bf16 + FlashInfer matching. **No GGUF Q4.** Metal backend thin (MPS + elementwise MSL).

**Verdict:** Strong architecture (e-graph + profiled extract). Attention speed is **library matching**, not synthesis. Same “poster vs garage” issue as tinygrad, different garage (FlashInfer).

---

## 5. Broader landscape

| Family | Examples | Core trick |
|--------|----------|------------|
| Schedule search / codegen | tinygrad, TVM Ansor/MetaSchedule | Emit code; tune tiles |
| E-graph / superopt | luminal, Mirage µGraph | Equivalence + pick / discover fusions |
| DSL + templates | Triton, FlexAttention, FlashInfer, ThunderKittens | Hand shell + fill-in + autotune |
| Hand backends | llama.cpp, MLX, TensorRT-LLM, metal-llm-server | Expert kernels |
| Agent codegen | KernelAgent, KernelEvolve, GEAK | LLM writes Triton/CUDA; verify/profile |
| Megakernel | Mirage MPK, Hazy megakernels | Whole-model GPU residency |

**Metal reality:** no public Triton-on-Metal, no FlashInfer. Serious Apple paths: **MLX**, **llama.cpp Metal**, **metal-llm-server**, plus weaker tinygrad Metal codegen.

**FlexAttention (honest template story):** hand FA template; Inductor injects `score_mod` / `mask_mod`. Does not pretend to invent flash from primops.

**MLX:** lazy graph + **hand** fast SDPA / quant matmul; `mx.fast.metal_kernel()` for custom MSL. Competitive Mac LLM without being a “search discovers FA” system — supports the idea that **strong matvecs + decent attention + Apple stack** can get far **without** FlashInfer-class flash, especially at moderate context.

---

## 6. metal-llm-server (oracle)

Hand Metal + fused decode + GGUF Q4_K + Q4_0 KV + SWA/shared-KV/PLE + **KV pool / continuous batching** (B≥1, default decode batch max 4).

Role for ksearch: **scoreboard + Gemma semantics**, not shader source of truth for Thesis A.

---

## 7. What’s actually hard (research gaps)

1. **Flash-style attention** — algorithmic (online softmax + tiling), not a tiling of naive SDPA. tinygrad doesn’t have it; luminal calls FlashInfer; Metal has no drop-in.
2. **Fused Q4_K matvec from IR** — decode bandwidth; neither tinygrad nor luminal ships llama.cpp-class search-codegen for this.
3. **Fusion / scheduler search** — tinygrad admits weakness; luminal multi-op fusion grow was disabled pending legality.
4. **Metal ecosystem gap** — more must be built in-house than on CUDA.

---

## 8. Decision recorded

**Implementation thesis: A — compiler-pure.**

- IR → generate MSL → search schedules/fusion.
- No FlashInfer; no shipping metal-llm-server `.metal` as the hot path.
- Expect an initial gap vs oracle; measure honestly.
- Motivation: MLX-class stacks show Mac LLM can be useful without FA theater; we push how far **generated** kernels get.

See [DESIGN.md](./DESIGN.md).

---

## 9. Tinygrad-only purge (2026-08)

Locked IR study against local `reference/tinygrad` UOp model:

- Graph product = ALU + movement + REDUCE (+ `Call`); sugar expands + `FuseHint`
- Scheduler invents CALL boundaries → **lower to KirStmt/KirExpr AST**
- **Generic MSL renderer** walks the AST only (dtype expand on `Load` e.g. Q4_K) — no hand `rmsnorm.metal`
- BEAM / plan_cache = OptOps tilings on that AST
- Eng builds **only** Graphs → `lower_to_metal`
- **Removed:** `debt/`, `GemmaModel`, `KSEARCH_DEBT`, hand `KirBody::*` Metal templates

Scoreboard: tok/s will rise via BEAM/parallel reduces on the AST — not via reintroducing hand kernels.
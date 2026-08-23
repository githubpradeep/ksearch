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

See [DESIGN.md](./DESIGN.md) and the beginner curriculum starting at [README.md](./README.md).

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
---

## 10. Q4K GEMV bandwidth ceiling (measured, 2026-08)

Question: decode is weight-bound; the coop Q4_K GEMV runs at ~19–25 GB/s on
M1 Pro while DRAM peaks ~200 GB/s. Is the gap a *layout* problem (searchable)
or an *instruction recipe* problem?

`ksearch q4k-frag-bench --rows 8960 --cols 1536` times hand-written MSL
variants against the pipeline kernel (all numerically verified vs dequant CPU ref):

| Variant | GB/s |
|---|---|
| pipeline `frag_nr4_dev` (ushort nibble-word trick, quad K-split) | **20–25** |
| coalesced (lane-contiguous 32B chunks) | 17–21 |
| coalesced nsg=1 | 17–19 |
| x-in-registers (x hoisted per superblock) | ~18–23 |
| one-row-per-lane (max threads) | 17–19 |
| pure-load probe (no dequant/FMA), same buffer | ~33 |

Findings:

- Load layout / coalescing / x-reuse move it ±15%, not ×5. The pipeline's
  ggml-style nibble-word recipe is already at the practical ceiling of this
  dequant pattern; the pure-load bound (~33 GB/s here) says memory-side
  headroom exists but requires a fundamentally lighter instruction mix.
- For calibration: llama.cpp-class Metal GEMV reaches ~60–90 GB/s effective on
  similar shapes — the remaining ~3× lives in their exact instruction schedule,
  not in tiling search. That is source-of-truth work for the renderer, not a
  tuner knob (Thesis A compliant either way).
- E2B decode streams ~261 MB of weights/token → today's ~62 tok/s ≈ 16 GB/s
  effective across the whole token (attn/PLE/logits included). A 2× GEMV win
  would land ~95–105 tok/s; that remains the biggest known lever.

### Round 3 (dispatch audit + row-ratio search, 2026-08)

- **Dispatch audit**: decode issues **428 dispatches/token** but host CPU
  enqueue is only **~0.8 ms/token** — not CPU-bound. Kernel-count reduction
  has no guaranteed win without GPU-side bubble timestamps.
- **Row-ratio 8** (`nr0 = nsg*8`: two Nr4 bodies per superblock, doubles FMA
  chains + halves x re-reads): implemented, verified across shapes, added to
  the tuner candidate grid. The tuner picks it for **zero** model shapes —
  within-process medians show it never beats ratio 4. Confirms short-K GEMV
  sits at a latency floor that neither layout nor ILP tuning moves.

### Round 4 (cold-cache + floor calibration, 2026-08)

Three measurement corrections that reframe rounds 1–3:

1. **Machine ceiling confirmed: 151.9 GB/s** (saturating probe, 262k threads,
   4×16B independent loads). User instinct correct — M1 Pro has headroom.
2. **Big-matrix GEMV is already excellent**: the pipeline coop kernel sustains
   **114–124 GB/s** on the 227 MB lm_head shape (75–82% of peak). The ggml
   verbatim recipe ties it.
3. **Dispatch+sync floor: 0.164 ms/call** (empty-kernel control). All small-
   buffer bench numbers were floor-dominated: true cold work at 8960×1536 is
   ~0.17 ms ≈ 46 GB/s (not 18), warm ~70 GB/s. Split-K (KS=2/4) pays the
   floor twice and loses even accounting for it — negative result, prototype
   kept in the harness.

Corrected decode model: 1.45 GB weights/token ÷ 16.1 ms = **90 GB/s system
aggregate**, against a 152 GB/s machine and 114+ GB/s proven-kernel rate.
The residual losses live in (a) per-layer 5–10 MB matrices that never reach
steady-state streaming, (b) attention/PLE/logits glue, (c) unknown — resolvable
only with GPU timestamp counters, not host timing.

Path to 90 tok/s unchanged in direction but sharpened: speculative decoding
(amortizes the whole per-token stream) remains the only lever with >1.3×
potential; per-kernel micro-optimization is measured out.

### Round 5 (end-to-end ablation — the decisive attribution, 2026-08)

`KSEARCH_ABLATE=noattn|nomlp|nople|nohead` skips each component wholesale;
Δms/token over a 48-token decode attributes real time without any sync
distortion (E2B Q4_K_M, pos < 64):

| Skipped | ms/token | Δ | Share |
|---|---|---|---|
| baseline | 19.0 | — | — |
| **MLP** (gate∥up + gelu + down ×35) | 8.8 | **10.2** | **54%** |
| attention (qkv+rope+sdpa+o ×35) | 14.2 | 4.9 | 26% |
| head (lm_head 226 MB + argmax) | 16.6 | 2.4 | 13% |
| PLE (prepass + per-layer proj) | 17.2 | 1.9 | 10% |

Cross-check: MLP ≈ 842 MB/token ÷ 10.2 ms = **82 GB/s achieved** vs **114 GB/s**
the same kernel family sustains at large scale → bounded gap ≈ 2.4 ms.

Strategic conclusions:

1. **Full-token bandwidth ceiling**: every byte at machine peak (152 GB/s)
   gives 1450 MB → 9.5 ms → **~105 tok/s is the hard physical limit** for
   single-stream E2B decode on M1 Pro.
2. Closing *every* measured kernel gap (MLP→114, rest→~100 GB/s) lands
   ~13.5 ms ≈ **74 tok/s**. That is the realistic engineering ceiling of
   per-kernel optimization.
3. Therefore **90 tok/s single-stream requires reducing bytes-per-generated-
   token**, i.e. speculative/MTP decoding (verify k tokens per weight stream),
   or a lighter quantization. All per-kernel levers are measured out.

### Round 6 (fused-sweep port — final kernel avenue closed, 2026-08)

Ported the llama.cpp loop-owning structure into the renderer
(`KSEARCH_Q4K_SWEEP=1`: helper owns `for (ib = lane/8; ib < nb; ib += 4)` for
plain + dual gate∥up paths; lowering drops the enclosing ForRange when
single-token/exact-rows). Numerically verified across shapes. Long-run A/B
(200-token generations, alternating):

| | steady decode |
|---|---|
| classic (call per superblock) | 60.5 / 61.0 tok/s |
| fused sweep | 59.1 / 59.2 tok/s |

The fused structure is ~2 tok/s **slower** end-to-end — Metal's inliner already
flattens the per-block call, and the sweep's register residency slightly hurts
the dual-stream kernels. `KSEARCH_Q4K_SWEEP` kept as opt-in for future
regression checks.

**Kernel avenues are now exhausted with measurements on every one**: layouts,
coalescing, x-reuse, ILP ratio, split-K, LOCAL staging, dispatch, loop fusion.
Single-stream E2B on M1 Pro sits at its practical ceiling (~62 tok/s ≈ 90 GB/s
aggregate); the paths beyond are speculative decoding and lighter quantization.

### Round 7 (MTP draft integration — Phase 1 complete, acceptance unresolved)

`--mtp <draft.gguf>` loads the 77M `gemma4-assistant` F16 draft (4 layers,
hidden 256→base 1536 via nextn pre/post projections, own tied 256-dim vocab
head) and runs one draft step per decoded token, reporting greedy
acceptance vs the base model's actual next token.

Status: full plumbing works end-to-end (weight loading via new
`Gguf::tensor_raw`, projections, 4-layer stack with fused per-head-norm+rope,
GQA attention over the BASE model's Q4_0 KV cache at mapped layers, tied-head
logits + softcap argmax, host-side seed chaining). Verified no crashes,
coherent base generation unaffected without `--mtp`.

**Open problem: acceptance = 0%** (127/127 misses on essay prompt). Draft
logits are structured (top-5 gaps of 1–4 logits) but dominated by
context-independent high-id tokens; first-call logits are unwritten zeros.

Debugged and ruled out: buffer swap feeding roped-Q into attn_output (found +
fixed), KV mapping shallowness (0→13/14 deep anchors, no change), scale
application (verified correct — scales are O(0.04–0.4)), embedding row
dequant, F16 concat overflow, python loader alignment bugs (three separate
parser bugs found while building the cross-check tooling).

Next steps for whoever picks this up:
1. Run reference `metal-llm-server` with `DRAFT_DEBUG=1 DRAFT_READ_ATTN=1` on
   the SAME prompt; diff its per-stage intermediates against
   `MTP_DUMP_ALL=1 seq.json/xh.json/h_preproj.json` dumps from ksearch.
2. Verify exporter semantics against llama.cpp's gemma4-assistant MTP PR
   (concat order, projection orientation, rope freq-factor direction, KV
   mapping anchors 33/34 vs shared-tail resolution).
3. Check whether draft expects base hidden BEFORE final norm (x, not x2).

Ablation hooks remain available (`KSEARCH_ABLATE=…`) as standing diagnostics.

Harness: `ksearch q4k-frag-bench`. Experimental TG-staged x toggle:
`KSEARCH_Q4K_LOCAL_X=1` (no win at these shapes).

### Round 2 (same-process A/B, 2026-08 later)

Added llama.cpp's verbatim `mul_vec_q4_K_f32_impl` recipe as a bench variant
(`ggmlkq_nsg2/nsg4`) plus a pure-load probe. Stable within-process numbers
(8960×1536 Q4K, best-of interleaved medians):

| Kernel | GB/s |
|---|---|
| pipeline coop frag | 28–29 |
| **ggml verbatim recipe (nsg2)** | **29.5–31** |
| coalesced / rowlane layouts | 19–21 |
| pure-load probe (no dequant) | ~33–39 |
| any kernel @ long K (nb=32) | ~55 |

Refined conclusions:

1. **Cross-process timing varies ±30%** (GPU clocks/thermal). Every claim in
   this file now requires same-process interleaved A/B; several earlier
   "wins" (including a 1.5× for the ggml recipe) evaporated under it.
2. The ggml recipe's real edge over our frag is **~5%** (its outer loop owns
   all superblocks; ours emits a call per block from the AST ForRange).
   Fusing that loop into codegen was prototyped and reverted: the sweep must
   own the loop start (`ix = lane/8`), which needs lowering changes for a
   ≈+1 tok/s end-to-end payoff. Documented here instead.
3. Short-K shapes (E2B nb=6) are latency-bound: every layout lands 20–30 GB/s
   while long-K hits ~55 GB/s with the same kernels. Structural escapes are
   token batching (MTP-style verify), not per-kernel micro-opt.
4. The load probe (~33–39 GB/s single-stream grid-stride) bounds what ANY
   dequant GEMV of this pattern can reach on this machine class without
   deeper memory-system work.

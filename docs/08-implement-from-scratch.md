# 8. Implement this from scratch

This is a rebuild plan, not a second copy of ksearch. After each stage you should have a **runnable checkpoint**. Use this repo as an oracle: when stuck, read the cited file, do not paste it blindly.

Assumptions: Rust, macOS, Apple GPU. You may use the `metal` crate the same way ksearch does.

## Stage 0 — Prerequisites (1–2 evenings)

**Know** (read these, do not skip):

- [00-transformers.md](./00-transformers.md) — residual stream, RMSNorm, SDPA, MLP, prefill vs decode
- [00-metal.md](./00-metal.md) — `gid` / `lid`, threadgroups, buffers, encoder
- [00-gemma-architecture.md](./00-gemma-architecture.md) — SWA, shared-KV, PLE

**Watch:**

```bash
manim -pql docs/animations/scenes.py TransformerBlock
manim -pql docs/animations/scenes.py MetalDispatch
manim -pql docs/animations/scenes.py GemmaLayer
manim -pql docs/animations/scenes.py DecodeTokenSeq
```

See [animations/README.md](./animations/README.md) for all scenes.

**Do:**

- Read [01-mental-model.md](./01-mental-model.md) and [DESIGN.md](./DESIGN.md).
- Run `cargo run -p ksearch_cli --release -- elem-add --n 1024` and read the printed MSL.

**Skip for now:** egglog, FlashAttention papers, serving.

## Stage 1 — Graph + elementwise (checkpoint: CPU)

Build crate `ir`:

```text
TensorId, Shape, DType::{F32,F16}
enum Op { Input, Add, Mul, ScaleConst }
struct Graph { nodes: Vec<Node> }
```

Implement `input`, `add`, `mul` with shape checks.

**Checkpoint:** unit test — `add` then `mul` produces the right `Op` chain and shapes. No GPU.

ksearch: `crates/ksearch_ir/src/graph.rs` (`enum Op`, `fn add`).

## Stage 2 — Kernel IR for elementwise (checkpoint: print AST)

```text
enum KirExpr { ConstF32, Gid, Load{buf,idx,dtype}, Bin{Add,Mul}, ... }
enum KirStmt { Let, Store }
struct KernelIr { n_inputs, body, launch: Elementwise{n} }
```

`schedule(Add)` → `ElemExpr` tree → `lower_elemwise` = one `Store(out, gid, expr)`.

**Checkpoint:** pretty-print the AST for `a+b`. You should see `Store(2, Gid, Add(Load(0,Gid), Load(1,Gid)))`.

ksearch: `schedule.rs` (`build_elem_expr`), `lower.rs` (`lower_elemwise`).

## Stage 3 — Renderer + Metal (checkpoint: GPU add matches CPU)

Walk the AST to MSL (the `elem-add` skeleton in [04-render-and-metal.md](./04-render-and-metal.md)). Compile with `new_library_with_source`. Bind buffers. Dispatch `n` threads.

**Checkpoint:** `max|gpu-cpu| < 1e-5` on F32 add of 1e6 elements. Print MSL in the test so you never fear the compiler.

If Metal compile fails, your printer has a bug. If it runs but numbers differ, your Load/Store indexing is wrong.

ksearch: `render.rs` (`render_msl`), `ksearch_metal`, CLI `elem_add`.

## Stage 4 — Matvec as mul + sum (checkpoint: F32 GEMV)

Graph:

```text
MulBroadcastRow(W[rows,cols], x[cols]) → [rows,cols]
SumReduce(axis=1) → [rows]
```

Scheduler: **recognize this pattern** → `KernelKind::Matvec`. Do not add `Op::MatMul`.

Lowering v1 (slow, clear): one thread per row, sequential `for c in 0..cols`.

**Checkpoint:** `4096×4096` F32 (or smaller) matches CPU; dump MSL and confirm a K loop exists.

Then lowering v2: `RowsParallel`, `tg` threads, each does `cols/tg` dots, `ThreadgroupReduce` / `SimdSum`, `lid==0` stores.

**Checkpoint:** same answers, faster.

ksearch: `Graph::matvec_prim`, `schedule` `SumReduce` arm, `lower_matvec_generic`.

## Stage 5 — OptSchedule + BEAM (checkpoint: faster than untuned)

Add `OptSchedule { tg, vec, unroll, nr0 }`. Lowering reads it (vector width, rows per TG).

Search 20–50 candidates, median of 3 launches, save `~/.cache/...`.

**Checkpoint:** `matvec --beam` prints a speedup vs `untuned`. Cache hit on second run.

ksearch: [05-beam-and-plans.md](./05-beam-and-plans.md).

## Stage 6 — RMSNorm sugar + FuseHint (checkpoint: one kernel)

Implement `rmsnorm_expand` as primitives (square, sum, scale, add eps, rsqrt, expand, muls) **and** `FuseHint::RmsNorm`.

Scheduler: if hint, `KernelKind::RmsNorm`. Lower: reduce `x²`, `rsqrt(mean+eps)`, store `x*inv*w`.

**Checkpoint:** fused kernel matches a two-kernel (or CPU) RMSNorm. MSL has one `kernel void`, a reduce, then a store loop — no host round-trip.

Resist adding `Op::RmsNorm`.

ksearch: `Graph::rmsnorm_expand`, `lower_rmsnorm`.

## Stage 7 — Eng cache (checkpoint: two ops in a row)

`Eng` builds Graphs, `lower_to_metal`, caches pipelines by key. Residual add after RMSNorm as a second kernel.

**Checkpoint:** `y = x + rmsnorm(x,w)` on GPU. First call compiles; second is launch-only.

ksearch: `crates/ksearch_kernels/src/lib.rs`.

## Stage 8 — F16 Load expand (checkpoint: half activations)

ALU stays float. `Load(F16)` → `float(half)`. `Store` → `half(val)`.

**Checkpoint:** matvec and RMSNorm in F16 match F32 within ~1e-2 relative (typical half).

## Stage 9 — Q4_K Load expand (checkpoint: packed GEMV)

Read [06-quant.md](./06-quant.md). Scalar `ksearch_load_q4k` first. Graph still mul+sum.

**Checkpoint:** 256×256 (one superblock per row) matches CPU `dequantize_row_q4_K` + dot. Then scale up. Then optional coop AST (`Q4kCoopFrag`) if scalar is too slow.

Forbidden checkpoint: a file `matvec_q4k.metal` that Eng launches by name.

## Stage 10 — Naive SDPA (checkpoint: tiny attention)

Graph `Call` + `FuseHint::SdpaNaive`. Lower online softmax: for each head, loop `t`, scores, running `m`/`l`, accumulate `O`. Causal `tlen`. F16 first, then `Load(Q40)` for K/V.

**Checkpoint:** `n_q=2, hd=32, tlen=8` matches a NumPy `softmax(QKᵀ)V` with mask.

Do **not** start with FlashAttention.

ksearch: `Graph::sdpa_naive`, `lower_sdpa_online`.

## Stage 11 — One transformer layer (checkpoint: residual doesn’t blow up)

F16 or Q4 weights, no PLE, no SWA:

```text
x = x + Attn(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

RoPE: host table `cos||sin` per position; kernel rotates pairs.

**Checkpoint:** random weights, one token, finite outputs; attention changes if you permute K.

## Stage 12 — GGUF + generate (checkpoint: Hi gate)

mmap GGUF. Parse Gemma 4 metadata. Load Q4_K weights. Implement layer extras in this order:

1. SWA window + two RoPE tables
2. Q4_0 KV append as `[seq, n_kv, hd]` + SDPA Q40 loads (MQA `n_kv=1`, GQA for E4B)
3. Shared-KV (`owns_kv` / `kv_source`)
4. PLE residual (required for E2B/E4B-it quality)
5. Tied lm_head + softcap argmax
6. Chat template + tokenizer

**Checkpoint:** this repo’s bench: `"Hi"` → text contains `Hi!` and `help`. Optional: same prompt on an E4B GGUF.

ksearch: `gemma_prim.rs` `forward_token`, [07-gemma-runtime.md](./07-gemma-runtime.md).

## Stage 13 — Prefill + serving (optional)

Chunked prefill (`matvec_batch`, batch SDPA). `KvPool` + decode-before-prefill scheduler. HTTP last.

## Design rules while you code (pin these)

1. Graph product = ALU + movement + REDUCE + `Call`. Sugar expands or hints.
2. Renderer walks AST only. Dtype expand on `Load` is allowed. Named shader products are not.
3. BEAM retunes `OptSchedule`. It does not add algorithms.
4. When a fuse is faster as two launches on one encoder, that is still legal (`Eng::matvec_rmsnorm_add_scale`). Both launches must still come from Graph→lower.
5. If a change would add `Op::FlashAttn` or paste `reference/metal-llm-server/shaders`, stop and re-read DESIGN.md.

## How to use this repo while rebuilding

| You are stuck on… | Read |
|-------------------|------|
| What ops exist | `ksearch_ir/src/graph.rs` `enum Op` |
| How fusion is requested | `FuseHint` in `kernel.rs` |
| How a region becomes a kernel | `schedule.rs` `fn schedule` |
| How loops are built | `lower.rs` `lower_elemwise` then `lower_rmsnorm` then `lower_matvec` |
| How MSL is printed | `render.rs` `fn render_msl` |
| How a token is computed | `gemma_prim.rs` `fn forward_token` |
| Why not copy oracle shaders | [FINDINGS.md](./FINDINGS.md) §2 and §8 |

Copy **tests and CLI shapes**, not shader text. When your `elem-add` MSL looks like ksearch’s, Stage 3 is done even if the pretty-printer differs.

## Suggested milestone timeline

| Weeks | Stages | You have |
|-------|--------|----------|
| 1 | 0–3 | GPU add |
| 2 | 4–6 | Tuned matvec + RMSNorm |
| 3 | 7–9 | F16 + Q4_K GEMV |
| 4 | 10–11 | One layer + naive attention |
| 5–7 | 12 | E2B `"Hi"` |
| later | 13 | tok/s and serving |

If `"Hi"` is wrong, bisect: embed row vs CPU dequant; RMSNorm vs CPU; matvec vs CPU; SDPA vs NumPy on a 4×4; then PLE. Do not tune BEAM until numbers are right.

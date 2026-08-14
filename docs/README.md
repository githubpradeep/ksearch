# ksearch documentation

This folder is the curriculum. After reading it in order, you should be able to sit down and implement a smaller clone: Graph → Kernel IR → MSL → run a matvec, then a transformer layer, then generate tokens.

You do **not** need to have written a compiler or GPU kernels before. Start at **0a–0c** if Metal, transformers, or Gemma are new. Diagrams are mermaid in those pages; videos are Manim in [animations/](./animations/README.md). Interactive map: open the basics canvas beside chat.

## Reading order

| # | Doc | What you should be able to do after it |
|---|-----|----------------------------------------|
| 0a | [00-transformers.md](./00-transformers.md) | Draw a decoder block; explain prefill vs decode and KV cache |
| 0b | [00-metal.md](./00-metal.md) | Explain gid/lid, threadgroups, buffers, one encoder per token |
| 0c | [00-gemma-architecture.md](./00-gemma-architecture.md) | Trace SWA, shared-KV, PLE, and a generate sequence |
| — | [animations/README.md](./animations/README.md) | Render Manim scenes for the diagrams above |
| — | [DESIGN.md](./DESIGN.md) | State the thesis in one sentence and name the forbidden shortcuts |
| — | [FINDINGS.md](./FINDINGS.md) | Explain why tinygrad / luminal / metal-llm-server were studied |
| 1 | [01-mental-model.md](./01-mental-model.md) | Draw the compiler pipeline and map each box to a crate |
| 2 | [02-graph-ir.md](./02-graph-ir.md) | Write `rmsnorm` as primitives + a `FuseHint` |
| 3 | [03-schedule-and-kernel-ir.md](./03-schedule-and-kernel-ir.md) | Turn `SumReduce(MulBroadcastRow(W, x))` into a matvec AST |
| 4 | [04-render-and-metal.md](./04-render-and-metal.md) | Walk an AST node to MSL and launch it |
| 5 | [05-beam-and-plans.md](./05-beam-and-plans.md) | Say what BEAM searches (tilings) and what it does not |
| 6 | [06-quant.md](./06-quant.md) | Explain dequant-at-load vs eager CPU dequant |
| 7 | [07-gemma-runtime.md](./07-gemma-runtime.md) | Trace one decode token through embed → layers → logits |
| 8 | [08-implement-from-scratch.md](./08-implement-from-scratch.md) | Rebuild the stack in stages, with checkpoints |

Read 0a–0c until the pictures are boring. Then DESIGN.md once. Then 01–08. Come back to FINDINGS.md when you wonder “why not just copy llama.cpp kernels?”

## How to study (not just read)

Each chapter points at **real files**. Open them. For every claim, find the enum variant or function. The code is the spec; the docs are a guided tour.

Suggested loop for each chapter:

1. Read the chapter once without the repo.
2. Open the cited files and skim signatures.
3. Run the matching CLI command (`elem-add`, then `matvec`, then `generate`).
4. Do the checkpoint in [08-implement-from-scratch.md](./08-implement-from-scratch.md) for that stage.

## Map: docs ↔ code

| Idea | Code |
|------|------|
| Graph primitives, sugar, `FuseHint` | `crates/ksearch_ir/src/graph.rs`, `kernel.rs` (`FuseHint`) |
| Dtypes, Q4_K byte counts | `crates/ksearch_ir/src/lib.rs` |
| Kernel IR AST | `crates/ksearch_ir/src/kernel.rs` (`KirExpr`, `KirStmt`, `KernelKind`) |
| Scheduler (CALL boundaries) | `crates/ksearch_codegen/src/schedule.rs` |
| Pattern checks (no catalog Ops) | `crates/ksearch_codegen/src/rewrite.rs` |
| KernelKind → AST | `crates/ksearch_codegen/src/lower.rs` |
| AST → MSL | `crates/ksearch_codegen/src/render.rs` |
| BEAM + disk cache | `crates/ksearch_codegen/src/beam.rs`, `plan_cache.rs` |
| `lower_to_metal` entry | `crates/ksearch_codegen/src/lib.rs` |
| Metal device / launch | `crates/ksearch_metal/src/lib.rs` |
| Runtime that builds Graphs | `crates/ksearch_kernels/src/lib.rs` (`Eng`) |
| Gemma 4 forward | `crates/ksearch_gemma/src/gemma_prim.rs` |
| GGUF mmap | `crates/ksearch_gguf/src/gguf_impl.rs` |
| CLI demos | `crates/ksearch_cli/src/main.rs` |

## Thesis A in one paragraph

tinygrad’s product is not `Ops.MATMUL`. It is ALU + movement + REDUCE. High-level ops (`matmul`, `RMSNorm`, `gelu`, SDPA) **expand** to those primitives. The scheduler invents kernel (CALL) boundaries. A **generic renderer** walks the resulting AST and emits Metal. Quantization is a **Load dtype**: `Load(Q4K)` becomes float in the renderer, the same way `Load(F16)` becomes float. BEAM only retunes loop tiles on that AST. Hand named shaders and Graph catalog Ops named after oracle kernels are out of scope.

That rule lives in `.cursor/rules/thesis-a-compiler.mdc` and wins over speed work.

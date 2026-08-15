# 5. BEAM search and plan cache

BEAM is **not** “AI that writes kernels.” It is a grid search over `OptSchedule` integers. Same AST family, different tile sizes, pick the fastest on this chip.

tinygrad’s BEAM applies OptOps (`LOCAL`, `UPCAST`, `UNROLL`, …) to an already-scheduled kernel. ksearch copies that idea with a small struct.

## What is searched

```rust
OptSchedule { tg, vec, unroll, nsg, nr0 }
```

| Field | Effect on matvec |
|-------|------------------|
| `tg` | Threads cooperating on a row (more threads → shorter K-loop, more reduce) |
| `vec` | Load 1/2/4 elements per inner step (float path) |
| `unroll` | Duplicate the K inner body |
| `nsg` | Simdgroups per threadgroup (Q4 coop; `threads = nsg*32`) |
| `nr0` | How many output rows one TG computes (amortizes loading `x`) |

`beam_matvec_candidates()` (float): `tg ∈ {32,64,128}`, `vec ∈ {2,4}`, `unroll ∈ {1,2}`, `nr0 ∈ {1,2,4,8}` — about 48 kernels.

`beam_matvec_q4k_candidates()`: `tg ∈ {8,16,32,64}`, `unroll ∈ {1,2}`, `nr0 ∈ {4,8,16}`. Lowering then **clamps** Q4/Q6 to ggml-like shapes (`apply_matvec_sched`), so some candidates collapse.

What is **not** searched: fusion policy (that is FuseHint), FlashAttention vs naive SDPA, whether to use Q4_K. Those are frontend / scheduler decisions.

## When it runs

`beam_search_matvec(graph, out, chip, time_ms)` in `ksearch_codegen/src/lib.rs`:

1. Classify plan kind from weight dtype: `matvec_f16_nr`, `matvec_q4k`, `matvec_q6k`, `matvec_f32`.
2. If a plan exists on disk and `KSEARCH_BEAM_FORCE` is unset → return it (after one timing).
3. Else time `OptSchedule::untuned()`, then every candidate; keep min median ms.
4. `save_plan(...)`.

`Eng::beam_matvec` builds the Graph, allocates scratch `x`/`y`, compiles each candidate, runs a warmup + 3 timed launches, takes the median.

`Eng::warm_matvec_plans` is called at model load for mid-size shapes so the first generate token is not a 50-kernel search. Huge tensors (lm_head) skip the warm cap (`MAX_ELEMS`).

CLI: `cargo run -p ksearch_cli --release -- matvec --rows 4096 --cols 4096 --beam`

## Plan cache on disk

`plan_key(kind, dims, chip)` hashes kind + `[rows,cols]` + Metal device name → `~/.cache/ksearch/beam/{key}.txt` (override with `KSEARCH_BEAM_CACHE`).

`lower_to_metal_chip` **loads** the plan before lowering, so even without an explicit BEAM call, the second process on the same machine uses the tuned tiles.

`KSEARCH_CHIP` overrides the chip string if you want to share caches (usually you should not).

## How to think about speed

Short-context decode is **weight-bandwidth bound**. Tiling helps by:

- Staging `x` in LOCAL so each weight row does not re-read activations from DRAM.
- Computing several rows per TG (`nr0`) so `x` is loaded once for 4–16 outputs.
- Using simdgroup-friendly Q4 unpack (256-wide superblocks).

BEAM will not close a 10× gap vs a hand FlashAttention + fused Q4 kernel that uses a different algorithm. DESIGN.md records this: dense F16 GEMV is near a local plateau; next real gains are better LOCAL staging / fusion, not more TG search.

## Implementing BEAM yourself (minimum)

```text
candidates = list of OptSchedule
best = inf
for s in candidates:
    kir = lower(kind, s)
    msl = render(kir)
    pipe = compile(msl)
    t = median of 3 launches
    if t < best: save s
write s to disk keyed by (op, rows, cols, device)
```

Correctness: always check the tuned kernel against the untuned kernel (or CPU) before saving. ksearch’s CLI `matvec --beam` compares to a CPU reference.

Next: [06-quant.md](./06-quant.md).

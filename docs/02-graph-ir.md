# 2. Graph IR

The Graph is the language you write models in. It lives in `crates/ksearch_ir/src/graph.rs`.

A Graph is a **append-only list of tensors**. Each `TensorId` is an index into `Graph.nodes`. There is no mutation of old nodes. There is no autograd.

```rust
pub struct Graph {
    pub nodes: Vec<Node>,
    pub fuse_hints: HashMap<u32, FuseHint>,
}

pub struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
}
```

## Dtypes

From `crates/ksearch_ir/src/lib.rs`:

| `DType` | Role |
|---------|------|
| `F32` | Rare on device (argmax index, SDPA meta, RoPE table, MWG partials) |
| `F16` | Activations, most norms, float weights after dequant |
| `Q4K` / `Q6K` | Packed **weights** (256 elems / 144 or 210 bytes). Logical shape is still `[rows, cols]` in **elements**, not bytes |
| `Q5K` | Packed weights used for some embeddings (PLE); Load expand → F16 |
| `Q40` | Packed **KV cache** (32 elems / 18 bytes per block) |
| `BF16` | Loader helper; not the hot-path act dtype |

`DType::size_bytes()` returns `0` for packed quants: you cannot `n * size_bytes()` a Q4_K tensor. Use `q4k_nbytes(nelem)`, `q6k_nbytes`, `q40_nbytes`.

Logical vs physical is the whole trick: Graph math talks about **elements**. The renderer talks about **bytes** only when expanding a Load.

## Primitive `Op`s (the product)

These are the tinygrad-like UOps:

| Op | Meaning |
|----|---------|
| `Input` | Placeholder; runtime binds a Metal buffer |
| `Const` | Broadcast scalar |
| `Add` / `Mul` | Same-shape elementwise |
| `ScaleConst` | `x * c` |
| `Rsqrt` / `Tanh` / `Exp` | Unary |
| `SumReduce` / `MaxReduce` | Drop one axis |
| `Expand` / `Reshape` / `Permute` | Movement (no data change in the math) |
| `MulBroadcastRow` | `out[r,c] = left[r,c] * row[c]` — the mul inside matvec |
| `CopySlice` | Contiguous copy with offsets |
| `Call` | “This region is one scheduled kernel”; algorithm is the `FuseHint` |

There is **no** `Op::MatMul`, `Op::RmsNorm`, `Op::Gelu`, `Op::Sdpa`. If you feel the urge to add one, stop: either expand to primitives, or `Call` + hint.

## How you build graphs

Every method returns a `TensorId` (or `Result`). Shapes and dtypes are checked at build time.

```rust
let mut g = Graph::new();
let x = g.input(Shape(vec![n]), DType::F16);
let w = g.input(Shape(vec![n]), DType::F16);
let y = g.rmsnorm_expand(x, w, 1e-6)?;
```

`Eng` (the runtime) does exactly this for every kernel, then throws the Graph away after lowering. Graphs are **not** a persistent whole-model DAG at runtime. Each `Eng` method builds a 2–6 node graph for one launch. That is a pragmatic split: the compiler is per-kernel; the model loop in `GemmaPrimModel` is Rust.

When you reimplement, you can start the same way. A whole-model Graph (tinygrad style) is a later upgrade, not required for correctness.

## Sugar that **expands** (primitives + hint)

These functions write several primitive nodes, then `hint(out, FuseHint::…)`. The scheduler sees the hint and fuses; if you ignored the hint you could still run the expanded ops as many kernels (slower, same math).

### RMSNorm

tinygrad: `x * rsqrt(mean(x²)+eps) * w`

```text
sq = x * x
sum = sum(sq, axis=0)          # shape [1]
mean = sum * (1/n)
inv = rsqrt(mean + eps)
out = (x * expand(inv)) * w
hint: FuseHint::RmsNorm { n, eps, x, w }
```

See `Graph::rmsnorm_expand`. Variants:

- `rmsnorm_add_expand` — residual add after RMSNorm
- `rmsnorm_add_scale_expand` — then multiply by a layer scale
- `gelu_tanh` — `0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * x³)))`
- `gelu_mul_at` — `gelu(gate) * up[off:]`

### Matvec as primitives

```text
m = MulBroadcastRow(W, x)   # [rows, cols]
y = SumReduce(m, axis=1)    # [rows]
```

`Graph::matvec_prim` is that pair. **This is the pattern the scheduler matches** for `KernelKind::Matvec`. Q4_K works because `MulBroadcastRow` allows `Q4K × F16 → F16` when `cols % 256 == 0`. The mul does not eagerly dequant on the host; dtype on the node is the **output** (F16). Weights stay Q4K on the input node.

## Sugar that is a `Call` (hint is the body)

Some regions are awkward to expand as a pile of primitives that the current scheduler can fuse (per-head RMS + RoPE + Q4_0 pack, SDPA, fused QKV matvecs). Those go through `Graph::call(inputs, shape, dtype, hint)`.

`Op::Call` without a hint is a scheduler error. The Call is not a catalog of algorithms; it is a **region marker**. The algorithm is `FuseHint`.

Important Call families (see `FuseHint` in `kernel.rs`):

| Hint | Math (one launch) |
|------|-------------------|
| `MatvecQkv` | `Q=Wq@x`, `K=Wk@x`, `V=Wv@x` (3 outputs; LOCAL-stage `x`) |
| `MatvecGateUpGelu` | `out[i] = gelu(Wg[i]·x) * (Wu[i]·x)` |
| `RmsNormMatvec` | RMSNorm into LOCAL `x_hat`, then `W @ x_hat` |
| `SdpaNaive` | `softmax(QKᵀ / √d) V` with causal `meta` (tlen, start) |
| `SdpaMwgPart` + `SdpaMwgReduce` | Split KV over workgroups, merge online-softmax partials |
| `RmsNormPerHeadRope` | Per-head RMS + RoPE |
| `RmsNormPerHeadQkvQ40` | Q RMS+RoPE (F16) + K RMS+RoPE+Q40 + V RMS+Q40 |
| `CopyScaleIndexed` | Embedding gather: `out[i] = scale * src[id * n + i]` |
| `SoftcapArgmax` | `argmax(cap * tanh(logits/cap))` — output is **F32 index** (half cannot hold vocab ids) |
| `QuantizeQ40` | Pack F16 → Q4_0 for KV append |

Prefill twins (`MatvecBatch`, `RmsNormRows`, `SdpaNaiveBatch`, …) are the same math with a token batch dimension.

## `FuseHint` vs `KernelKind`

They look similar. That is intentional.

- **`FuseHint`** is Graph metadata (“please fuse this”).
- **`KernelKind`** is the scheduler’s decision (“this kernel’s body is X”).

`schedule.rs` function `sk_from_hint` is almost a field-for-field copy. Extra fields appear on `KernelKind` (e.g. `weight_dtype`) that the scheduler **reads from the Graph**, not from the hint.

When you add a new fused region:

1. Add a `FuseHint` variant.
2. Add a Graph constructor (`fn my_op(...) -> TensorId`).
3. Map it in `sk_from_hint` to a `KernelKind`.
4. Lower that `KernelKind` to `KirStmt` in `lower.rs`.
5. Renderer should already handle the stmts you emit. If you need new AST nodes, add `KirExpr`/`KirStmt` first, then render them **generically**.

Do not add `Op::MyOp`.

## Shape rules you will hit

- `Add`/`Mul`: shapes and dtypes must match.
- `SumReduce(axis)`: that axis is removed; rank-0 becomes `[1]`.
- `MulBroadcastRow`: left rank-2, row rank-1, `left.cols == row.len`.
- Packed matvec: `cols % 256 == 0` (Q4_K/Q6_K superblock).
- Q4_0 pack: `n % 32 == 0` and usually `hd % 32 == 0`.

If `Eng` fails with `ShapeMismatch` at Graph build, you never got to Metal. Fix the Graph.

## Mental exercise

Write (on paper) the Graph for:

```
y = RMSNorm(x, w_norm)
o = W @ y
```

Two legal answers:

1. **Unfused:** `rmsnorm_expand` then `matvec_prim` — two kernels.
2. **Fused:** `rmsnorm_matvec(W, x, w_norm, eps)` — one `Call` + `FuseHint::RmsNormMatvec`.

ksearch uses (2) on the decode path when it is faster (LOCAL `x_hat` reused by all rows). The Graph still *could* be (1). Fusion is a schedule choice with a hint, not a new primitive.

Next: [03-schedule-and-kernel-ir.md](./03-schedule-and-kernel-ir.md).

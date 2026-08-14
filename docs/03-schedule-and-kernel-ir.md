# 3. Scheduler and Kernel IR

This chapter is the compiler’s middle: Graph region → `ScheduledKernel` → `KernelIr` AST.

Files:

- `crates/ksearch_codegen/src/schedule.rs`
- `crates/ksearch_codegen/src/lower.rs`
- `crates/ksearch_ir/src/kernel.rs`

## `schedule(graph, out) → Vec<ScheduledKernel>`

Today this returns **one** kernel (the region rooted at `out`). A whole-model scheduler that splits a giant Graph into many CALLs is the tinygrad endgame; ksearch’s `Eng` already presents one region per launch, so a single-kernel schedule is enough.

Algorithm (`schedule.rs`):

1. `rewrite::validate_q4_matvec_pattern` — confirm Q4 matvec is still `SumReduce(MulBroadcastRow(Q4K, F16))`, not a catalog Op.
2. If `graph.fuse_hint(out)` exists → `sk_from_hint` (Call or expanded sugar).
3. Else match the root `Op`:
   - `Add`/`Mul`/`ScaleConst` → fold a small elementwise tree into `ElemExpr`, `KernelKind::Elementwise`
   - `SumReduce` of `MulBroadcastRow` → `KernelKind::Matvec`
   - other last-axis `SumReduce` → `KernelKind::SumLast`
   - `CopySlice` → `KernelKind::CopySlice`
   - movement-only (`Reshape`/`Permute`/`Expand`) → identity `Elementwise` (Load)
   - `Call` without hint → error

That is how CALL boundaries are “invented”: **pattern + hint**, not a human naming `rmsnorm.metal`.

### Elementwise trees

`build_elem_expr` walks `Add`/`Mul`/`ScaleConst`/`Input` and produces:

```text
ElemExpr::Add(Load(0), Mul(Load(1), Scale(Load(2), 0.5)))
```

`Load(i)` means “input buffer i at this thread’s `gid`.” All inputs must be the same length. This is how `add_scale` (`scale * (a+b)`) becomes one kernel without a hint.

### Matvec pattern (the one to memorize)

```text
out = SumReduce(MulBroadcastRow(W, x), axis=last)
```

Scheduler records:

```rust
KernelKind::Matvec {
    rows, cols,
    matrix: W,
    vector: x,
    weight_dtype: dtype_of(W),  // F16 or Q4K or Q6K
}
```

Same Graph ops for Q4_K and F16. The dtype on `W` is the only difference. Lowering picks a cooperative Q4 loop vs a float `VecMulSum` loop.

## `KernelKind` is not Metal

`KernelKind` is a **tag for lowering**. It says *which AST builder* to run. It does not contain MSL.

Think of it as LLVM’s “this is a reduction over axis 1 with these buffers.” `lower_to_kir` in `lower.rs` is a large `match` that returns `(KirLaunch, Vec<KirStmt>, n_outputs)`.

## Kernel IR AST

Defined in `crates/ksearch_ir/src/kernel.rs`. This is the language the renderer understands.

### `KernelIr`

```text
name, n_inputs, n_outputs, out_shape, out_dtype, launch, body, next_id
```

`n_outputs > 1` for fused QKV (three buffers) and `RmsNormAddThenRmsNorm` (residual stream + FFN-normalized activations).

### `KirLaunch` (grid)

| Variant | GPU mapping |
|---------|-------------|
| `Elementwise { n }` | 1-D grid, one thread per element, `gid` |
| `Rows { rows }` | one thread per output row |
| `RowsParallel { rows, tg }` | one **threadgroup** per row (or per `nr0` rows); `lid` inside TG |
| `RowsParallelSg { rows, nsg }` | ggml-style: `threadsPerThreadgroup = (32, nsg)` simdgroups |
| `MulMm { … }` | prefill GEMM tiles (64×32 simdgroup MMA) |

Matvec uses parallel launches so 32–256 threads cooperate on one (or a few) output rows.

### `KirExpr` (values)

The ALU:

- `ConstF32` / `ConstU32`
- `Gid` / `Lid` / `ForVar` / `Var` / `UVar`
- `Bin { Add, Sub, Mul, Div, Max, Min }`
- `Unary { Neg, Exp, Tanh, Rsqrt, Sqrt, Floor }`
- `CmpGt` / `CmpEq`
- `CastU32ToF32` / `CastF32ToU32`
- `SimdSum` — Metal `simd_sum` (warp/simdgroup reduction)

Memory:

- `Load { buf, idx, dtype }` — **logical element load**. Renderer expands dtype (`half` → `float`, `Q4K` → dequant to `float`).
- `VecMulSum` — `dot(floatN(a[i..]), floatN(b[j..]))` (UPCAST)
- `TgLoad` / `ThreadLoad` — LOCAL / register arrays
- `Q4kCoopFrag` / `Q6kCoopFrag` — one simdgroup lane’s partial for a 256-element superblock (still AST nodes, not a named Graph Op)

### `KirStmt` (effects)

- `Let` / `LetU32` / `Assign`
- `For { n, body }` / `ForRange` (K-loop with stride)
- `Store { buf, idx, val }`
- `TgDeclF32` / `TgStore` / `Barrier` — threadgroup (LOCAL) memory
- `ThreadDeclF32` / `ThreadStore` — per-thread arrays
- `If`
- `ThreadgroupReduce` / `ThreadgroupArgmax`
- Quant packing/MMA helpers: `Q4kCoopNr4`, `Q4kMulMm`, `Q40PackBlock`, `Q40DequantToTg`, …

If you need new GPU behavior, **add a Kir node** and teach the renderer to print it. Do not `format!(msl_string)` inside `lower.rs`.

## Lowering a matvec (float)

`lower_matvec` → `lower_matvec_generic` roughly:

1. Optionally RMSNorm `x` into threadgroup `tg_x` (`lower_stage_xhat_local`).
2. Each threadgroup owns `nr0` output rows (`OptSchedule.nr0`).
3. Loop K with stride `tg * vec`:
   - `VecMulSum` of width `vec` (1/2/4)
   - accumulate into `acc`
4. `ThreadgroupReduce` / `SimdSum`
5. `lid==0` stores `out[row]`

`OptSchedule` only changes `tg`, `vec`, `unroll`, `nr0`. Same AST shape.

## Lowering a matvec (Q4_K)

If `weight_dtype == Q4K && cols % 256 == 0`, `lower_matvec_q4k_coop` emits ggml-shaped cooperative loops:

- Threadgroup of 64 or 128 (`nsg` simdgroups × 32 lanes)
- Each superblock (256 weights) is one `Q4kCoopFrag` or `Q4kCoopNr4` (4 consecutive rows, one weight load)
- Activations from device `half` or from LOCAL `x`

The renderer expands those nodes using `ksearch_load_q4k` / coop helpers. The Graph was still `mul + sum`.

## Lowering RMSNorm (fused hint)

`lower_rmsnorm`:

1. Each thread loads `x[i]`, `w[i]`.
2. Partial `x²` + `simd_sum` / threadgroup reduce → `sum`.
3. `inv = rsqrt(sum/n + eps)`.
4. Store `x[i] * inv * w[i]`.

One kernel, no round-trip to DRAM for the mean. That fusion is why the hint exists.

## IDs in the AST

`next_id` allocates SSA-like `Let` ids (`v0`, `v1`, … in MSL). Buffers are numbered: inputs `0..n_inputs-1`, outputs after that (`out` or `out0..`). Lowering hardcodes “buffer 0 is W, 1 is x” to match `Eng` bind order. When you add a kernel, **input order in `ScheduledKernel.inputs` must match `Eng::run` and Graph `call` inputs**.

## `OptSchedule`

```rust
pub struct OptSchedule {
    pub tg: u64,     // threads per threadgroup (LOCAL / parallelism)
    pub vec: u32,    // vector width (UPCAST): 1, 2, 4
    pub unroll: u32, // K-loop unroll
    pub nsg: u32,    // simdgroups per TG (Q4 coop)
    pub nr0: u32,    // output rows per threadgroup
}
```

tinygrad names: `LOCAL ≈ tg`, `UPCAST ≈ vec`, `UNROLL ≈ unroll`. Defaults live on `OptSchedule::default()` / `q4k_default()`. `apply_matvec_sched` in `lower.rs` clamps Q4/Q6 to ggml-like `{tg:64, nr0:8, nsg:2}` (wider TG for mid-size Q4 row counts).

BEAM searches these integers; it does not change `KernelKind`.

## Checkpoint

You should now be able to:

1. Look at `schedule()` and point to the matvec match.
2. Look at `lower_elemwise` (~8 lines) and see `Store(out, gid, expr)`.
3. Explain why `KernelKind::RmsNorm` is allowed but `Op::RmsNorm` is not.

Next: [04-render-and-metal.md](./04-render-and-metal.md).

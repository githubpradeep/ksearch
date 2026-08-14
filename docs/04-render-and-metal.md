# 4. Renderer and Metal runtime

Lowering produced an AST. This chapter prints it as Metal Shading Language and runs it.

Files:

- `crates/ksearch_codegen/src/render.rs` — `render_msl`
- `crates/ksearch_metal/src/lib.rs` — `MetalContext`
- `crates/ksearch_kernels/src/lib.rs` — `Eng`

## Renderer contract

`render_msl(kir, sched) -> MetalKernelSource`

The renderer **only walks the AST**. It does not know “this is RMSNorm.” If the body is loads, a reduce, and stores, the MSL will look like RMSNorm. That is the point.

`OptSchedule` is already baked into the AST (loop bounds, vector widths). `render_msl` currently ignores `sched` except for the signature (`_sched`). Tilings change lowering, not a second rewrite in the renderer.

### Output

```rust
pub struct MetalKernelSource {
    pub name: String,
    pub source: String,      // full .metal translation unit
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: LaunchHint,  // how Eng sets threadgroups
}
```

### Skeleton it emits

```metal
#include <metal_stdlib>
using namespace metal;

// optional: Q4K/Q6K/Q40 load helpers if the body needs them

kernel void k_elem_2(
  device const float* in0 [[buffer(0)]],
  device const float* in1 [[buffer(1)]],
  device float* out [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= 1048576u) return;
  out[gid] = (float(in0[gid]) + float(in1[gid]));
}
```

`LaunchHint::Elementwise { n }` → Eng launches `n` threads (with a threadgroup size of 256).

For `RowsParallel`, the signature uses `threadgroup_position_in_grid` (`gid` = row/TG index) and `thread_index_in_threadgroup` (`lid`).

## Load expand (the only “magic”)

`KirExpr::Load { buf, idx, dtype }` does **not** mean “raw bytes at idx.” It means “logical element `idx` as **float** in the ALU.”

| dtype | MSL idea |
|-------|----------|
| `F32` | `inN[idx]` |
| `F16` | `float(inN[idx])` where `inN` is `device const half*` |
| `Q4K` | `ksearch_load_q4k((device const uchar*)inN, idx)` |
| `Q6K` | `ksearch_load_q6k(...)` |
| `Q40` | dequant one nibble block to float (KV) |

Helpers are prepended **only if** the body contains those dtypes (`body_needs_q4`, etc.). They are dtype expansion, analogous to `float(half)`. They are not a product called `matvec_q4k.metal`.

`VecMulSum` is the vectorized form: load 4 weights, load 4 acts, `dot`. For Q4_K it uses `ksearch_load_q4k4` when indices share a superblock.

Stores go the other way: ALU is float, `out` may be `half` (`half(val)`).

## Walking statements

`render.rs` recursively prints:

| KirStmt | MSL |
|---------|-----|
| `Let { id, expr }` | `float v{id} = {expr};` |
| `For { id, n, body }` | `for (uint f{id} = 0; f{id} < n; f{id}++) { ... }` |
| `Store` | `out[idx] = half(val);` (or float / packed) |
| `TgDeclF32 { id, n }` | `threadgroup float tg{id}[n];` |
| `Barrier` | `threadgroup_barrier(mem_flags::mem_threadgroup);` |
| `ThreadgroupReduce` | `simd_sum` and/or LOCAL tree reduce |
| `If` | `if (cond) { ... }` |

If you add a `KirStmt` and forget to render it, you get a compile error in Rust (`non-exhaustive match`). That is the safety rail: new GPU behavior must be explicit in both AST and printer.

## Metal runtime (`MetalContext`)

Thin wrapper over the `metal` crate:

- `new()` — system default device + command queue
- `compile(kernel)` — `new_library_with_source` + pipeline state (fast math on)
- `buffer_f32` / `buffer_bytes` / `buffer_empty_f16` — **shared** storage (CPU and GPU see the same pages; fine on Apple Silicon)
- `encode` / `encode_multi` / `encode_offsets` — bind buffers, dispatch
- `synchronize` / `flush_async` / `wait_inflight_at_most` — CPU/GPU overlap

### One encoder, many dispatches

Decode wants **one command buffer per token**, many kernels, no CPU round-trip. `MetalContext` keeps a pending compute encoder. `Eng::run` appends a dispatch. `flush_async` commits without waiting. `GemmaPrimModel::generate_timed` ping-pongs token-id buffers so the next gather overlaps the in-flight token (`wait_inflight_at_most(1)`).

Offsets (`encode_offsets`) implement slices: RoPE table at `pos * hd`, KV append at `pos * q40_row_bytes(hd)`, PLE context at `layer * ple_dim`. Same pipeline, different byte offset.

## `Eng`: the only way the model talks to the compiler

`crates/ksearch_kernels/src/lib.rs`.

Pattern for every op:

```text
key = "mv_q4k_{rows}x{cols}"
if key not cached:
    Graph { input W: Q4K, input x: F16, out = matvec_prim }
    src = lower_to_metal_chip(&g, out, device_name)
    compile, insert cache
run(inputs, output)
```

Properties:

- **Cache key** includes dtypes and shapes (and eps bits for RMSNorm). One pipeline per unique kernel.
- **Chip** is `ctx.device_name()` so BEAM plans are per GPU.
- **No MSL in Eng.** If you catch yourself concatenating shader strings here, you are violating Thesis A.
- Some “fuses” are **two dispatches on the same encoder** (`matvec_rmsnorm_add_scale`): a Graph FuseHint exists, but a single-TG fused AST lost to multi-TG coop matvec + tiny RMSNorm. That is a performance choice, still Graph→lower for each piece.

`weight_cache_tag` maps `DType` → `"q4k"` / `"f16"` / … for keys.

## Launch dimensions

`Eng::tg_for`:

| `LaunchHint` | threads per threadgroup |
|--------------|-------------------------|
| Elementwise / Rows | 256 |
| `RowsParallel { tg }` | `tg` |
| `RowsParallelSg { nsg }` | `nsg * 32` |
| `MulMm { tw, nsg }` | `tw * nsg` |

Wrong TG size → silent wrong answers or Metal errors. The hint on `MetalKernelSource` must match what lowering assumed.

## How to debug a kernel

1. `cargo run -p ksearch_cli --release -- elem-add` — print MSL, check `max\|err\|`.
2. Dump MSL for a Graph: `ksearch_codegen::layer::render_rmsnorm` or examples under `crates/ksearch_codegen/examples/`.
3. If Metal compile fails, the error includes the source (`compile` maps the compiler message).
4. If GPU runs but numbers are wrong: compare a tiny CPU loop (as `elem_add` / `matvec` in the CLI do).
5. `KSEARCH_PROFILE=1` on generate: per-section ms in `forward_token`.

## Exercise

Take the `elem_add` CLI. Change `g.add` to `g.mul`, run it, read the MSL. You should see `*` instead of `+` and nothing else structural. That is the compiler doing its job.

Next: [05-beam-and-plans.md](./05-beam-and-plans.md).

# 6. Quantization as Load expand

Goal: run Q4_K GGUF weights **without** a Graph op named `MatVecQ4K` and **without** converting the whole model to F16 in RAM.

tinygrad’s usual path is `ggml_data_to_tensor` → float Tensor → `.half()`. That is correct and simple; it also costs ~2× memory vs packed Q4_K and loses the packed-bandwidth win on decode.

ksearch’s Thesis A quant rule:

- **Allowed:** `Load(dtype=Q4K|Q6K|Q40)` expands in the **generic renderer** to float, like `Load(F16)` → float. Packed buffers stay packed.
- **Forbidden:** hand `q4k_load` helpers pasted as named matvec kernels / Graph catalog `Op::MatVecQ4K*`.
- Activations stay **F16**. Argmax index stays **F32**.

## Physical layouts (GGML)

You need these numbers; they show up in asserts and byte offsets.

| Type | Block | Bytes / block | Helper |
|------|-------|----------------|--------|
| Q4_K | 256 elements | 144 | `q4k_nbytes(nelem)` |
| Q5_K | 256 elements | 176 | `q5k_nbytes` |
| Q6_K | 256 elements | 210 | `q6k_nbytes` |
| Q4_0 | 32 elements | 18 | `q40_row_bytes(hd)`, `q40_nbytes(max_t, hd)` |

Graph **shape** is always element counts: `token_embd` is `[vocab, hidden]`, not `[vocab, q4k_row_bytes]`. Metal buffer length is `q4k_nbytes(vocab * hidden)`.

Q4_K block (ggml): `half d`, `half dmin`, 12 scale/min bytes, 128 bytes of 4-bit `qs`. Superblock scale `d` and min `dmin` plus 8 groups of 32 weights. The renderer helper `ksearch_q4k_at` implements that formula. You do not need to memorize bit packing to use the compiler; you do need it if you reimplement Load expand.

## Three places quant appears

### 1. Weights (Q4_K / Q6_K / mixed)

`GemmaPrimModel` uploads GGUF rows **raw** when the tensor type is Q4_K or Q6_K (`WeightBuf::Q4K`). Norms and a few small tensors dequant on CPU to F16 (`dequant_to_f16_bytes`) because they are tiny.

Matvec Graph:

```text
W: Input [rows, cols] Q4K
x: Input [cols] F16
y = sum(W ⊙ x, axis=1)     # out dtype F16
```

`MulBroadcastRow` allows `(Q4K|Q6K, F16) → F16` if `cols % 256 == 0`. Scheduler sets `KernelKind::Matvec { weight_dtype: Q4K }`. Lowering emits `Load { dtype: Q4K }` / `Q4kCoopFrag`. Renderer prepends `ksearch_load_q4k`.

Mixed QKV (Q4_K Q, Q4_K K, Q6_K V) is allowed: `qkv_weight_dtypes_ok` requires all packed K-quants or all the same float.

### 2. KV cache (Q4_0)

KV is written every token and read every later token. F16 KV is simple; Q4_0 is ~4× smaller and is what the decode SDPA loads.

Write path: fused `RmsNormPerHeadQkvQ40` — per-head RMS + RoPE, then `Q40PackBlock` into `kv_k[pos]`, `kv_v[pos]`. Offset: `pos * q40_row_bytes(hd)`.

Read path: `SdpaNaive` / MWG with `kv_dtype: Q40`. K/V `Load` expands to float inside the attention loops. Q stays F16.

`QuantizeQ40` exists as a standalone Call if you need pack without RMS.

### 3. Eager CPU dequant (everything else)

`Gguf::dequant_to_f16_bytes` for types that are not worth a Load expand yet, or that are small (RMS weights, scales). `Q5_K` PLE embeddings use Load expand on gather (`CopyScaleIndexed` with `src_dtype: Q5K`).

DESIGN.md historically said “CPU dequant Q4→F16, no q4k_load.” The **locked** rule (and the code) is packed Q4_K + renderer Load expand. CPU dequant remains the fallback for other ggml types.

## What “generic” means

A beginner often implements Q4 matvec as a second shader file. Thesis A forbids that as the architecture.

Legal structure:

```text
render(Load(buf, idx, dtype)):
    match dtype:
        F32 -> buf[idx]
        F16 -> float(buf[idx])
        Q4K -> ksearch_load_q4k(buf, idx)
```

Every kernel that loads weights — copy-scale embed, matvec, gate-up-gelu, lm_head — gets Q4 “for free” if lowering emits `Load` with that dtype.

Coop nodes (`Q4kCoopNr4`) are **specialized AST** for a known superblock layout (like `VecMulSum` is specialized float4). They still go through the generic printer. They are not Graph Ops.

## Prefill GEMM

When `batch > 8`, `MatvecBatch` can lower as `Q4kMulMm` / `Q6kMulMm`: tile 64×32, simdgroup MMA, Load-expand a K-tile of weights to LOCAL. Same weights, different launch (`KirLaunch::MulMm`). Decode (`batch=1`) stays GEMV (matvec).

## Implementing Load expand yourself

1. Keep Graph shapes in elements.
2. Add `DType::Q4K` and `q4k_nbytes`.
3. Allow `MulBroadcastRow` Q4K×F16.
4. In the renderer, one function `emit_load(dtype, buf, idx) -> String`.
5. Port ggml’s `dequantize_row_q4_K` to MSL as `ksearch_load_q4k` (scalar first).
6. Only then add vector/coop AST nodes if profiling says scalar Load is the bottleneck.

Checkpoint: a 256×256 Q4_K matvec matches a CPU dequant+dot within a few ulps of F16.

Next: [07-gemma-runtime.md](./07-gemma-runtime.md).

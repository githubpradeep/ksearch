# 0c. Gemma 4 architecture

Vanilla decoder (previous chapter) plus dense Gemma 4 specifics that ksearch actually runs (**E2B** and **E4B**). Numbers come from GGUF metadata (`GemmaConfig`); the diagrams below use E2B-shaped counts as a teaching default. E4B is the same family with larger hidden size and **GQA** (`n_kv > 1`). MoE A4B is not supported.

Watch: Manim scenes `Gemma4Stack`, `GemmaLayer`, `DecodeTokenSeq` ([animations](./animations/README.md)). Runtime details: [07-gemma-runtime.md](./07-gemma-runtime.md).

## Whole-model block diagram

```mermaid
flowchart TB
  TOK["token id"]
  EMB["token_embd row gather\nQ4_K Load expand → F16 x"]
  PLE0["PLE prepass\nper_layer_token_embd → ple_ctx"]
  L0["Layer 0  SWA  owns KV"]
  L1["Layer 1  SWA  owns KV"]
  DOT["…"]
  LF["Layer 5  full attn  owns KV"]
  SH["Last shared-KV layers\nQ only; read earlier K,V"]
  ON["output_norm RMS"]
  LM["lm_head = token_embdᵀ\nsoftcap + argmax"]

  TOK --> EMB --> PLE0
  PLE0 --> L0 --> L1 --> DOT --> LF --> SH --> ON --> LM
```

Tied embeddings: `token_embd.weight` is both the lookup table and the lm_head matrix.

## One layer (owner)

This is the block `forward_token` runs for a layer that **writes** KV.

```mermaid
flowchart TB
  X["x  residual F16 hidden"]
  AN["attn_norm RMS"]
  QKV["Wq,Wk,Wv @ x_hat\n3 outputs"]
  QN["per-head RMS + RoPE on Q"]
  KN["per-head RMS + RoPE on K\npack Q4_0 → kv_k[pos, n_kv]"]
  VN["per-head RMS on V\npack Q4_0 → kv_v[pos, n_kv]"]
  SDPA["SDPA hybrid\nQ F16 × K,V Q40\n1 TG per KV head (GQA)"]
  O["Wo @ attn"]
  PA["post_attn RMS + residual add"]
  FN["ffn_norm RMS"]
  GU["gelu(W_gate x) ⊙ (W_up x)"]
  D["W_down"]
  PF["post_ffw RMS + residual"]
  PLE["PLE: gelu(W_gate x) ⊙ ctx[layer]\nthen W_proj + RMS + layer_scale"]

  X --> AN --> QKV
  QKV --> QN
  QKV --> KN
  QKV --> VN
  QN --> SDPA
  KN --> SDPA
  VN --> SDPA
  SDPA --> O --> PA --> FN --> GU --> D --> PF --> PLE
  X --> PA
  PF --> PLE
```

Non-owners skip K/V projections and pack; they still run Q, SDPA against `kv_source(layer)`’s cache, then MLP + PLE.

## Sliding window vs full attention

Default pattern: `(layer + 1) % 6 != 0` → SWA; every 6th layer is **full**.

```mermaid
flowchart LR
  subgraph swa ["SWA layer  hd=256  θ=10k"]
    S["query at pos p"]
    S --> W["keys in (p-window, p]"]
  end
  subgraph full ["Full layer  hd=512  θ=1e6  partial RoPE"]
    F["query at pos p"]
    F --> A["keys in [0, p]"]
  end
```

- SWA: cheap attention, local context, smaller head dim.
- Full: global context, larger `hd`, only a prefix of dims get RoPE (`partial_rotary = 0.25`).

`meta` buffer per layer: `(tlen, start)` so the SDPA kernel does not recompile when `pos` changes.

## Shared KV

Last `shared_kv_layers` layers do not allocate K/V. They reuse the nearest earlier owner with the same SWA/full type.

```mermaid
flowchart TB
  O0["owner 0  writes KV0"]
  O1["owner 1  writes KV1"]
  O2["owner 2  writes KV2"]
  S0["shared  Q only  reads KV?"]

  O0 --> O1 --> O2 --> S0
  O2 -.->|"kv_source"| S0
```

## PLE (per-layer embedding)

Not in a vanilla GPT block. Each token has an extra embedding used **inside every layer**:

```text
u = gelu(W_inp_gate @ x) ⊙ ple_ctx[layer]     # ple_dim (e.g. 256)
x = layer_scale * (x + RMSNorm(W_proj @ u))
```

`ple_ctx` is computed once per token (prepass) from `per_layer_token_embd`. Skip this and E2B-it will fail the `"Hi"` gate even if attention is perfect.

## Data types on the wire

```mermaid
flowchart LR
  subgraph packed ["Packed on GPU"]
    W["weights Q4_K / Q6_K"]
    KV["KV cache Q4_0"]
    PLE["PLE embd often Q5_K"]
  end
  subgraph f16 ["F16 activations"]
    X["residual x"]
    Q["Q after RoPE"]
    N["norm weights"]
  end
  subgraph f32 ["F32"]
    M["SDPA meta"]
    R["RoPE cos/sin"]
    I["argmax index"]
  end
  W -->|"Load expand"| X
  KV -->|"Load Q40 in SDPA"| Q
```

## Sequence: `generate(prompt, n_predict)`

```mermaid
sequenceDiagram
  actor User
  participant CLI as ksearch CLI
  participant Tok as GGUF tokenizer
  participant M as GemmaPrimModel
  participant GPU as Metal

  User->>CLI: --prompt "Hi"
  CLI->>Tok: gemma4 chat template + BPE
  Tok-->>CLI: prompt ids
  CLI->>M: generate(ids, n_predict)

  Note over M,GPU: Prefill (no logits except last)
  loop chunks of 256
    M->>GPU: embed batch + all layers + KV append
    GPU-->>M: async
  end

  Note over M,GPU: Decode
  loop n_predict or EOS
    M->>GPU: forward_token (logits)
    GPU-->>M: argmax id in tok_idx
    M-->>CLI: token
  end
  CLI->>Tok: decode ids
  Tok-->>User: text
```

## Sequence: one decode token (compiler + GPU)

```mermaid
sequenceDiagram
  participant M as GemmaPrimModel
  participant Eng
  participant CG as codegen
  participant MTL as MetalContext
  participant GPU

  M->>Eng: embed_from_idx(tok)
  alt pipeline missing
    Eng->>Eng: Graph copy_scale_indexed
    Eng->>CG: lower_to_metal_chip
    CG->>CG: schedule → lower → render MSL
    CG->>MTL: compile library
    MTL-->>Eng: pipeline cached
  end
  Eng->>MTL: encode dispatch (buffer offsets)
  Note over MTL,GPU: encoder stays open

  loop each layer
    M->>Eng: rmsnorm_matvec_qkv / SDPA / MLP / PLE
    Eng->>MTL: more dispatches
  end

  M->>Eng: rmsnorm + lm_head + softcap_argmax
  M->>MTL: flush_async
  MTL->>GPU: commit
  M->>MTL: wait_inflight_at_most(1)
  MTL-->>M: next token id
```

The **first** time a unique `(op, shape, dtype)` runs, you pay MSL compile. Later tokens hit `Eng`’s `HashMap`.

## Sequence: serving (`ksearch serve`)

```mermaid
sequenceDiagram
  participant HTTP as axum /v1/chat/completions
  participant Q as job queue
  participant Sch as scheduler thread
  participant Pool as KvPool
  participant M as GemmaPrimModel

  HTTP->>Q: tokenize + InferenceRequest
  Q->>Sch: recv

  Note over Sch: decode-before-prefill tick
  loop occupied slots
    alt phase Decode
      Sch->>Pool: bind_slot
      Sch->>M: decode_token
      M-->>HTTP: StreamEvent.Token
    else phase Prefill
      Sch->>M: prefill_chunk
    end
  end
```

GPU work is **serial** (one model, shared `x` scratch). Parallelism is **N KV slots** in memory.

## Map onto code

| Diagram box | Code |
|-------------|------|
| Embed gather | `Eng::copy_scale_indexed_wd` |
| PLE prepass | `ple_prepass` in `gemma_prim.rs` |
| QKV + RMS | `Eng::rmsnorm_matvec_qkv_wds` |
| Pack KV | `Eng::rmsnorm_per_head_qkv_q40_off` |
| SDPA | `Eng::sdpa_hybrid_kv` |
| MLP | `matvec_gate_up_gelu_wd` |
| PLE residual | `matvec_gelu_mul_at` + `matvec_rmsnorm_add_scale` |
| lm_head | `matvec_wd` on `token_embd` |
| Sample | `softcap_argmax` or `sample_softcap_min_p` |

## Checkpoint

Explain to a rubber duck, without notes:

1. Why decode is matvecs, not matmuls.
2. What SWA changes in the SDPA loop bound.
3. Why shared-KV layers still run Q and SDPA.
4. Why PLE is required for the Hi gate.
5. Why the Metal encoder stays open for a whole token.

Then start the compiler path: [01-mental-model.md](./01-mental-model.md).

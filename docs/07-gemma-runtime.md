# 7. Gemma 4 runtime

The compiler generates kernels. This chapter is the **model loop** that asks for those kernels: load GGUF, run a token, keep a KV cache, sample.

Block diagrams and sequences for the architecture (before this runtime wiring): [00-gemma-architecture.md](./00-gemma-architecture.md). Videos: `Gemma4Stack`, `GemmaLayer`, `DecodeTokenSeq`.

Files:

- `crates/ksearch_gemma/src/lib.rs` — `GemmaConfig` from GGUF metadata
- `crates/ksearch_gemma/src/gemma_prim.rs` — `GemmaPrimModel`
- `crates/ksearch_gemma/src/kv_pool.rs` — serving slots
- `crates/ksearch_gguf/` — mmap + tokenizer
- `crates/ksearch_cli/src/main.rs` — generate / bench
- `crates/ksearch_cli/src/serve.rs` + `scheduler.rs` — HTTP + continuous batching

## Gemma 4 E2B shape

`GemmaConfig::from_gguf` reads keys like `gemma4.block_count`, `gemma4.embedding_length`, …

| Field | Role |
|-------|------|
| `n_layers` | Transformer blocks (`blk.{i}.*` in GGUF) |
| `hidden` | Residual stream width (E2B: 1536) |
| `n_heads` / `n_kv` | Q heads vs KV heads. E2B is **MQA**: `n_kv` is typically 1 |
| `head_dim_swa` / `head_dim_full` | Sliding-window layers vs full-attn layers (256 vs 512) |
| `swa_pattern` | Which layers are sliding-window (often 5 SWA + 1 full) |
| `sliding_window` | SWA attends only the last W tokens |
| `shared_kv_layers` | Last N layers **reuse** earlier K/V (no KV write) |
| `ffn[i]` | Per-layer MLP inner size |
| `ple_dim` | Per-layer embedding width (Gemma 4 PLE) |
| `softcap` | Final logit `cap * tanh(x/cap)` |
| `rms_eps` | RMSNorm epsilon |
| `rope_theta_*` / `partial_rotary` | RoPE bases; full-attn rotates only a prefix of `hd` (0.25) |

`owns_kv(layer)` / `kv_source(layer)` implement shared-KV: a non-owner still runs Q projection + SDPA, but reads another layer’s packed K/V.

SDPA streams **one** K/V sequence `[tlen, hd]` across all Q heads (MQA bandwidth). Packed KV per owning layer is Q4_0 with logical shape `[max_seq, hd]`.

## GGUF and tokenizer

`ksearch_gguf` mmaps the file. Tensor lookup is by name (`token_embd.weight`, `blk.0.attn_q.weight`, …). Metadata is a key/value table (`gemma4.block_count`, …).

- `tensor_raw(name)` — packed bytes (upload Q4_K as-is)
- `dequant_to_f16_bytes(name)` — CPU dequant for small tensors
- `Vocab::from_gguf` / `build_tokenizer_from_gguf` — decode ids → text; encode uses the `tokenizers` crate plus `gemma4_chat_prompt` so `"Hi"` becomes the instruction-tuned chat template, not raw characters

CLI `generate --prompt` always goes through that template. `--tokens 1,2,3` skips it.

## Load

`GemmaPrimModel::load(path, max_seq)`:

1. `Gguf::open` — mmap; tensor names + ggml types + byte offsets.
2. Parse `GemmaConfig`.
3. `MetalContext::new`.
4. Upload `token_embd.weight` packed if Q4_K (tied lm_head uses the same buffer), else dequant F16.
5. Dequant all `*.norm.weight` to F16 (small).
6. Lazy-load large weights on first use (`ensure_weight`) into `HashMap<String, WeightBuf>`.
7. Allocate scratch: residual `x`/`x2`, `tmp_q/k/v/o`, FFN temps, logits, PLE buffers, RoPE tables, Q4_0 KV for each owning layer, SDPA `meta`, MWG scratch. Prefill scratch is sized for `PREFILL_CHUNK` (256) token rows.
8. Warm BEAM plans for mid-size Q4 matvecs unless `KSEARCH_SKIP_BEAM` is set.

`max_seq` is the **KV length you allocate** (bench might use 64–1024; `serve` defaults 16k). The GGUF context can be 128k; you do not have to allocate that.

## One decode token (`forward_token`)

`pos` is the current sequence index. After the token, `pos += 1`.

```text
1. Embed
   tok_idx holds the token id as F32
   copy_scale_indexed: x = scale * token_embd[id]
   (Q4K Load expand if the embedding is packed)

2. PLE load
   gather per_layer_token_embd[id] → ple_tok
   ple_prepass: project to ple_ctx[n_layers * ple_dim]

3. For each layer:
   a. Attn input RMS + QKV matvecs
      owner:     rmsnorm_matvec_qkv → tmp_q, tmp_k, tmp_v
      non-owner: rmsnorm_matvec on Q only
   b. Per-head RMS + RoPE
      owner: pack K/V as Q4_0 into kv_k/v at byte offset pos * q40_row_bytes(hd)
      non-owner: RMS+RoPE on Q only
   c. SDPA hybrid
      meta = (tlen, start)   # SWA: last window; full: 0..pos+1
      if tlen < 128: naive online softmax (one TG streams K/V for all Q heads)
      else: MWG pass1 (16 partitions) + pass2 reduce
      K/V Load dtype = Q40
   d. o-proj:  tmp_o → x2
   e. MLP
      rmsnorm_add_then_rmsnorm   (residual + post-attn RMS + ffn RMS, 2 outputs)
      matvec_gate_up_gelu        (or unfused if gate/up dtypes differ)
      down-proj
      rmsnorm_add residual
   f. PLE residual
      u = gelu(W_gate @ x) * ple_ctx[layer]
      x = layer_scale * (x + rms(W_proj @ u))

4. If want_logits:
      RMSNorm(x, output_norm)
      logits = token_embd @ x          # tied weights
      softcap_argmax → next id in F32
```

Every bullet is an `Eng` method that builds a Graph and `lower_to_metal`. The model file contains **no MSL**.

Prefill (`forward_prefill_chunk`) is the same math with `n_tok` rows: `matvec_batch`, `sdpa_*_batch`, `rmsnorm_per_head_qkv_q40_batch`. Long prompts are split into 256-token chunks so scratch buffers stay bounded.

## PLE (per-layer embedding)

Gemma 4 is not a vanilla decoder-only block. Each token has an extra embedding used **inside each layer** as a gated residual:

```text
u = gelu(W_inp_gate @ x) ⊙ ple_ctx[layer]
x = scale * (x + RMSNorm(W_proj @ u))
```

`ple_prepass` runs once per token (or chunk) so `ple_ctx` is ready before the layer loop. When you reimplement “a transformer,” you can skip PLE and still learn the compiler. When you reimplement **this** model, PLE is required for the Hi gate.

## Generate loop

`generate` / `generate_timed`:

1. **Prefill** all prompt tokens except the last (`want_logits=false`, keep GPU busy, `flush_async`).
2. **Prefill last prompt token** with logits (or decode from there).
3. **Decode** `n_predict` tokens: `forward_token(tok, true)`, read argmax, stop on EOS ids `1` and `106`.

Decode overlaps: ping-pong `tok_idx[0|1]`, `wait_inflight_at_most(1)` so the CPU reads token *i−1* while the GPU runs token *i*.

Sampling: temperature 0 → GPU `softcap_argmax`. Else read F16 logits and `sample_softcap_min_p` on CPU (`serve` path).

## Serving

`ksearch serve --port 8080`:

- HTTP (axum) tokenizes OpenAI-style `/v1/chat/completions` and queues a job.
- A dedicated **scheduler thread** owns the GPU (Metal is not multi-threaded here).
- `KvPool` holds `slots` independent KV caches (`--slots 4`, like llama.cpp `--parallel`).
- Policy: **decode-before-prefill**. Each tick: one token for every decoding slot, then a prefill chunk for prefilling slots. GPU work is serial (shared `x` scratch); concurrency is **memory** (N KV slots).

`GemmaPrimModel::bind_slot` swaps the model’s `kv_k`/`kv_v` pointers to the pool buffers and sets `pos`.

## Correctness gate

`ksearch bench` encodes `"Hi"` with the chat template, generates, and requires the decoded text to contain `Hi!` and `help`. That is the P2 gate in DESIGN.md. If you break RMSNorm, RoPE, SDPA causality, or PLE, this fails even when kernels still launch.

## Reimplementing the model (order)

1. One layer, F16 weights, no SWA/PLE/shared-KV: RMSNorm → QKV → RoPE → naive SDPA → o-proj → MLP.
2. Add GGUF load + Q4_K matvec.
3. Add SWA + RoPE tables.
4. Add Q4_0 KV + hybrid SDPA.
5. Add PLE + shared-KV.
6. Add chunked prefill + tied lm_head + softcap argmax.
7. Only then KvPool / HTTP.

Next: [08-implement-from-scratch.md](./08-implement-from-scratch.md).

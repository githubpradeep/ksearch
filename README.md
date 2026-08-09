# ksearch

Metal **kernel compiler** for Gemma-class LLM inference (Thesis A).

IR → generate MSL → search schedules. Measured against `reference/metal-llm-server`.
Does **not** ship hand FlashAttention / Q4 fusion farms as the product.

## Docs

- [docs/FINDINGS.md](docs/FINDINGS.md) — research (tinygrad, luminal, landscape)
- [docs/DESIGN.md](docs/DESIGN.md) — Thesis A design lock

## Quick start

```bash
cargo run -p ksearch_cli --release -- bench
cargo run -p ksearch_cli --release -- generate \
  --gguf ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --prompt "Hi" --n-predict 32 --max-seq 64
```

`bench` prints Hi pass/fail plus prefill/decode tok/s for Hi and the essay prompt.

## References

`reference/tinygrad`, `reference/luminal`, `reference/metal-llm-server`

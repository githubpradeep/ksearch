# ksearch

Metal **kernel compiler** for Gemma-class LLM inference (Thesis A).

IR → generate MSL → search schedules. Measured against `reference/metal-llm-server`.
Does **not** ship hand FlashAttention / Q4 fusion farms as the product.

## Docs

- [docs/FINDINGS.md](docs/FINDINGS.md) — research (tinygrad, luminal, landscape)
- [docs/DESIGN.md](docs/DESIGN.md) — Thesis A design lock

## Quick start

```bash
cargo run -p ksearch_cli --release -- elem-add
cargo run -p ksearch_cli --release -- matvec --beam
```

## References

`reference/tinygrad`, `reference/luminal`, `reference/metal-llm-server`

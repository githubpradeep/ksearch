# Manim animations

Short videos for the basics chapters. They are **not** the spec — the markdown + code are. Use these when a diagram on the page is not enough.

Requires [Manim Community](https://www.manim.community/) (Python 3.10+). On macOS:

```bash
brew install py3cairo ffmpeg
pip install -r docs/animations/requirements.txt
```

## Render

From the repo root:

```bash
# low-res preview (fast)
manim -pql docs/animations/scenes.py TransformerBlock

# 1080p
manim -pqh docs/animations/scenes.py TransformerBlock
```

`-p` opens the player when done. Output is under `docs/animations/media/videos/`.

Render every scene:

```bash
for s in TransformerBlock SdpaNaive MetalDispatch MatvecMulSum \
         CompilerPipeline Gemma4Stack GemmaLayer DecodeTokenSeq PrefillVsDecode
do
  manim -ql docs/animations/scenes.py "$s"
done
```

## Scene index

| Scene | What it shows | Read with |
|-------|----------------|-----------|
| `TransformerBlock` | Residual stream, attn + MLP | [00-transformers.md](../00-transformers.md) |
| `SdpaNaive` | Q, K, V → scores → softmax → O | [00-transformers.md](../00-transformers.md) |
| `PrefillVsDecode` | Prompt pass vs one-token loop + KV | [00-transformers.md](../00-transformers.md) |
| `MetalDispatch` | Grid → threadgroup → simdgroup → thread | [00-metal.md](../00-metal.md) |
| `MatvecMulSum` | `y = sum(W * x)` as the compiler sees it | [00-metal.md](../00-metal.md), [02-graph-ir.md](../02-graph-ir.md) |
| `CompilerPipeline` | Graph → schedule → KIR → MSL → GPU | [01-mental-model.md](../01-mental-model.md) |
| `Gemma4Stack` | Embed, PLE, SWA/full layers, lm_head | [00-gemma-architecture.md](../00-gemma-architecture.md) |
| `GemmaLayer` | One owner layer including PLE | [00-gemma-architecture.md](../00-gemma-architecture.md) |
| `DecodeTokenSeq` | Sequence diagram of one decode token | [00-gemma-architecture.md](../00-gemma-architecture.md) |

## Quality settings

| Flag | Use |
|------|-----|
| `-ql` | 480p15 — iterate on timing |
| `-qm` | 720p30 |
| `-qh` | 1080p60 — share / record |

If Cairo/FFmpeg are missing, Manim errors on import; install the brew packages above, not only `pip install manim`.

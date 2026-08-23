# 9. The tuner: a learned cost model guides the search

BEAM (chapter 5) times **every** candidate tiling. That is ~48 kernels per
float matvec shape and ~36 per Q4_K shape — paid at first model load, per
shape, per chip. The tuner replaces most of those timings with **predictions**
from a cost model trained on timings this machine already produced.

Same idea as TVM's XGBoost cost model (AutoTVM), with two twists:

1. **No Python, no XGBoost.** The model is a from-scratch histogram
   gradient-boosted tree ensemble (~200 lines) in
   `crates/ksearch_tuner/src/gbdt.rs`.
2. **The dataset collects itself.** Every BEAM/tuner timing anywhere in ksearch
   appends one JSONL line. Using the compiler *is* data collection.

## Data flow

```text
any beam/tuner timing ──▶ ~/.cache/ksearch/tuner/samples.jsonl   (append-only)
                              │
               GBDT regression on log(ms)
               features: (kind, rows, cols, tg, vec, unroll, nsg, nr0,
                          threads/TG, K-work/thread, TG count,
                          per-TG traffic, total traffic, flop:byte)
                              │
             ~/.cache/ksearch/tuner/cost_model_{chip}.json
                              │
     guided search: measure only top-1 predicted + ε-greedy explorers,
     stop early when the measured best confirms the predicted optimum
```

One sample = `{kind, rows, cols, chip, sched{tg,vec,unroll,nsg,nr0}, ms, ts}`.
Models are keyed by chip (like plan caches); features are chip-independent
hardware proxies, so nothing conceptually blocks cross-chip transfer later.

## Why log(ms) and trees

Kernel time spans ~0.05–5 ms and is roughly multiplicative in its factors
(halve the traffic → halve the time). Squared loss on `log(ms)` therefore
optimizes *relative* error, which is what ranking candidates needs; absolute-ms
squared loss would let big shapes dominate small ones. Trees handle the
discrete knobs (tg/nr0/…) and interactions (traffic vs occupancy) without any
feature scaling, on tabular data of a few thousand rows — exactly where a
neural net would be fussy and XGBoost would be overkill.

## Guided search as a bandit

`GuidedSearch` (`crates/ksearch_tuner/src/search.rs`) treats one search as a
single-step decision problem:

- **state** — problem features + which candidates are already measured
- **action** — which unmeasured schedule to time next
- **reward** — −ms of the final pick

Policy: greedy on the model (top-1 predicted unmeasured) plus `explore`
random picks per round (ε-greedy), under a measurement `budget`. Stop when the
budget is spent or the measured best is within `early_stop_rel` of the
predicted global minimum — i.e. measurement confirmed the model's optimum.
This is deliberately the same loop an agent would run; swapping the GBDT for a
learned Q-function later changes one file.

Safety rules baked into `beam_search_matvec`:

- The untuned baseline is always candidate 0 and always measured.
- Identical rendered MSL is timed once (Q4/Q6 clamping collapses many seeds).
- No model / thin dataset ⇒ classic full grid (which also logs samples).
- Plans are still verified against the plan cache flow from chapter 5.

## Measured on an M1 Pro (F16 4096×4096)

```text
exhaustive: 48 kernels → best 0.518 ms (tg=128 nr0=8)
guided:      6 timings (12% of grid) → 0.520 ms  +0.002 ms vs best
```

And on a shape the model had never seen (6144×4096): 6 timings picked the same
schedule the full grid picks. That is the point of shared features — shapes
transfer, so new layers/shapes are tuned almost for free once a few shapes have
been searched.

Holdout metrics after ~200 samples (`ksearch tune`): mean Spearman ρ ≈ 0.87,
mean regret ≈ 3%.

## Using it

```bash
# 1. Collect (passive): every --beam search logs samples.
KSEARCH_BEAM_FORCE=1 cargo run -p ksearch_cli --release -- matvec --rows 4096 --cols 4096 --beam

# 2. Train + report holdout ranking metrics.
cargo run -p ksearch_cli --release -- tune

# 3. Replay guided vs exhaustive on a live shape (GPU ground truth).
cargo run -p ksearch_cli --release -- tune --live-rows 4096 --live-cols 4096

# 4. Turn guided mode on for real searches (model load warmup included).
KSEARCH_TUNER=1 cargo run -p ksearch_cli --release -- generate --gguf …
```

| Env | Meaning |
|-----|---------|
| `KSEARCH_TUNER=1` | Enable guided search (default off: log-only + full grid) |
| `KSEARCH_TUNER_BUDGET` | Max timings per search (default 12) |
| `KSEARCH_TUNER_EXPLORE` | Random explorers per round (default 2) |
| `KSEARCH_TUNER_MIN_SAMPLES` | Dataset size before guiding activates (default 128) |
| `KSEARCH_TUNER_EARLY_STOP` | Confirm-optimum tolerance (default 0.03) |
| `KSEARCH_TUNER_LOG=0` | Disable sample logging |
| `KSEARCH_TUNER_CACHE` | Tuner dir (default `~/.cache/ksearch/tuner`) |

Retraining happens automatically after searches once ≥96 samples exist for the
chip (throttled to every ≥24 new samples per process).

## What the tuner does not do

It does not invent schedules outside the BEAM candidate grids, choose fusion
policy, or pick algorithms (FlashAttention vs naive SDPA). It makes the
existing search cheap enough to run everywhere — and its dataset is exactly
the training set a future RL policy would need.

Next: [06-quant.md](./06-quant.md).

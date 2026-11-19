# 10 — CUDA (H100 / RTX 40-50) backend

After this chapter you should be able to: name the three crates that make up the
CUDA path, pick the backend with `--backend` / `KSEARCH_BACKEND`, read a
generated `.cu` file, and say exactly which parts of the stack are verified on a
machine with no NVIDIA GPU.

Prerequisites: [04-render-and-metal.md](./04-render-and-metal.md) (the renderer
and the launch contract) and [07-gemma-runtime.md](./07-gemma-runtime.md) (the
model loop that now runs on either backend).

## The one-sentence version

The Kernel IR is backend-neutral; CUDA is a second **target** in the same
renderer plus a second **runtime** under the same `Context` facade — the model
code does not know which one it is running on.

```
Graph ──► schedule ──► Kernel IR ──┬──► render(Target::Metal) ──► MSL  ──► ksearch_metal
                                   └──► render(Target::Cuda)  ──► CUDA ──► ksearch_cuda
                                             ▲                                 ▲
                                             └────── ksearch_compute::Context ─┘
```

| Crate | Role |
|-------|------|
| `crates/ksearch_codegen` | `Target::Cuda`, `cuda_prelude.rs` (dialect shim), `render_cuda`, `lower_to_cuda_chip` |
| `crates/ksearch_cuda` | `libcuda` FFI (`ffi.rs`), `CudaContext`, `Buffer`, cubin build, `launch_config` |
| `crates/ksearch_compute` | `Context` / `Buffer` / `Pipeline`: the only thing the model layer touches |

## Picking the backend

`ksearch_compute::Context::new()` is the single decision point:

1. `KSEARCH_BACKEND=metal|cuda` — explicit, wins over everything.
2. Apple host — Metal.
3. Everything else — CUDA (H100 `sm_90`, RTX 5090 `sm_120`, ...).
4. Nothing usable — an error naming every backend it tried.

On the CLI:

```bash
# model loops: auto by default (CUDA on an NVIDIA host), or pin it
ksearch generate --gguf model.gguf --prompt "Hi" --backend cuda
ksearch serve    --gguf model.gguf --slots 4    --backend cuda
KSEARCH_BACKEND=cuda ksearch bench --gguf model.gguf

# kernel/R&D commands
ksearch cuda                     # device, sm_XX, nvcc path, grid/block/smem per kernel
ksearch cuda --out /tmp/cu       # write every generated .cu
ksearch cuda --compile           # ...and nvcc --cubin each one for this device
ksearch matvec --backend cuda --rows 4096 --cols 4096 --dtype q4k --beam
```

`Backend::parse` also accepts `mtl` / `nvidia`. `--backend` on `generate`,
`bench` and `serve` is exported as `KSEARCH_BACKEND` before the model opens the
GPU, so the two spellings cannot disagree. Naming a backend the binary was not
built with fails immediately with the feature to rebuild.

## Building for an NVIDIA-only box

The `metal` dependency links `Metal.framework`, which does not exist on
Linux/Windows, so the backends are cargo features and the default is both:

| Build | Command | Backends |
|-------|---------|----------|
| Everything (default, macOS dev) | `cargo build --release -p ksearch_cli` | Metal + CUDA |
| NVIDIA box (H100 / A100 / RTX 40-50) | `cargo build --release -p ksearch_cli --no-default-features --features cuda` | CUDA only |
| Apple GPU only | `cargo build --release -p ksearch_cli --no-default-features --features metal` | Metal only |

On the CUDA-only build the `metal` crate is not compiled or linked at all: the
binary has no `Metal.framework` in `otool -L` (Linux: no `-lMetal`). The
Metal-only R&D commands stay in the CLI and fail fast, before touching a GPU:

| Command on a CUDA-only build | Result |
|------------------------------|--------|
| `elem-add` | `` `elem-add` needs the Metal backend: rebuild with `--features metal` `` |
| `q4k-frag-bench` | same, named for that command |
| `matvec --backend metal` | same, named for that command |
| `tune --live-rows R --live-cols C` | `` `tune-live` needs the Metal backend: ... `` |
| `tune` (dataset ranking, no GPU) | still works |
| `cuda` (render / `--compile`) | still works |

`KSEARCH_BACKEND=metal` on a CUDA-only binary fails in `Context::open_strict`
(`this build has no Metal backend (feature \`metal\` off)`, exit 1) when the GPU
is opened — after the GGUF header is read, before any inference — and
`ksearch_compute` refuses to build with neither feature (`compile_error!`).

**Feature plumbing rule.** Cargo ignores a member's `default-features = false`
unless `[workspace.dependencies]` declares it too, so the backends are pinned at
the workspace level:

```toml
ksearch_compute = { path = "crates/ksearch_compute", default-features = false }
ksearch_gemma   = { path = "crates/ksearch_gemma",   default-features = false }
ksearch_kernels = { path = "crates/ksearch_kernels", default-features = false }
```

Each layer re-exports the choice (`ksearch_cli/metal → ksearch_gemma/metal →
ksearch_kernels/metal → ksearch_compute/metal`), so `--features cuda` really
does switch Metal off everywhere. Without this, `cargo build
--no-default-features --features cuda` still links `Metal.framework` and cannot
build on Linux. Check it with `otool -L target/debug/ksearch | grep -c Metal`
(expect `0`).

## The dialect shim

The renderer emits one small MSL-flavoured IR for both targets; for CUDA it
prepends [`CUDA_PRELUDE`](../crates/ksearch_codegen/src/cuda_prelude.rs), which
maps that dialect onto CUDA C++:

| MSL | CUDA |
|-----|------|
| `kernel void` | `extern "C" __global__ void` |
| `[[buffer(n)]]` | plain pointers, `in0..inN` then `out` |
| `thread_position_in_grid` | `blockIdx.x * blockDim.x + threadIdx.x` |
| `threadgroup_position_in_grid` | `blockIdx.x` |
| `thread_index_in_simdgroup` | `threadIdx.x` (block shape `(32, nsg)`) |
| `simdgroup_index_in_threadgroup` | `threadIdx.y` (block shape `(32, nsg)`) |
| `simd_sum` / `simd_shuffle_*` | `__shfl_xor_sync` / `__shfl_down_sync` butterflies |
| `threadgroup_barrier` | `__syncthreads()` |
| `threadgroup` | `__shared__` |
| `simdgroup_matrix` MMA | shared-memory tiled FMA GEMM (`mul_mm`, bounds-checked) |
| `xcrun metal` | `nvcc --cubin -arch=sm_XX` |

When `__CUDACC__` is undefined the prelude substitutes stubs for the CUDA
builtins, so the very same translation unit still parses under
`clang++ -x c++ -fsyntax-only`. Nothing on that path ever executes — it exists so
generated kernels can be validated in CI without a CUDA toolchain.

## The runtime

- **Driver**: `libcuda` is loaded with `dlopen` (`libcuda.so.1`, `libcuda.so`,
  `nvcuda.dll`, `libcuda.dylib`), so the workspace builds and tests on machines
  with no toolkit. A missing library is a `CudaError::DriverUnavailable`, never
  a link error or a panic.
- **Buffers**: `cuMemAlloc` + `cuMemcpyHtoD` / `DtoH` / `DtoD`; `zero` is
  `cuMemsetD8`, and `blit_copy_bytes` is the DtoD copy the Metal path spells as
  a blit encoder. `Buffer` is `Clone` and shares the allocation, matching Metal
  semantics the model layer was written against.
- **Kernels**: `nvcc --cubin -std=c++17 -O3 -arch=sm_XX -o out.cubin src.cu`,
  then `cuModuleLoadData` + `cuModuleGetFunction`, cached by source hash.
  `CudaContext::compile` returns the cubin bytes, which is what
  `ksearch cuda --compile` reports.
- **Launches**: `launch_config(&LaunchHint, tg)` maps each IR launch kind to
  `(grid, block, smem)`; `LaunchHint::MulMm` carries the tile and shared-memory
  size, everything else derives from the element/row count and `KSEARCH_TG`.
- **Telemetry**: `Context::phase_mark` / `resolve_phase_times_ms` record
  **device events** on CUDA (`cuEventRecord` on the stream the bracketed work
  was enqueued on), so `KSEARCH_PHASE_GPUTIME` reports real GPU milliseconds
  — one event per mark, freed at resolve. `KSEARCH_CUDA_EVENTS=0` falls back
  to host instants. `take_telemetry()` stays host-side enqueue accounting.
  Replayed decode graphs cannot be phase-marked from inside `forward_token`;
  the bench path brackets the whole replay loop instead and prints
  `[gputime] decode replay: N tokens in X ms -> Y ms/token (device events)`.

## Prefill: the dequant-once F16 weight cache

The Q4_K/Q6_K prefill tile dequantizes every weight **inside the K-loop** —
per chunk, per repeat. llama.cpp's pp path does not: since the MMQ work its
default GEMM keeps quantized loads out of the critical path, and the older
dequant-to-F16-plus-cuBLAS route cached the dequantized copy once. ksearch
now does the same, compiler-style:

- `matvec_w_batch` (batch > 8, CUDA) dequantizes each packed weight tensor
  **once** with a standalone runtime kernel (`ks_dequant_q4k_f16` /
  `ks_dequant_q6k_f16` — the exact `ksearch_load_*` recipes as plain CUDA)
  and caches the F16 copy (`GemmaPrimModel::f16_bufs`).
- Every later chunk runs `KirStmt::MulMmF16`: the same `m16n16k16` WMMA core
  as the Q4_K tile, on a **64×128 output tile with BK=32 and a two-stage
  `cp.async` double buffer on both tiles** — no dequant in the loop, so the
  weight tile streams exactly like the activation tile.
- Both tile dimensions are A10 measurements, not defaults from a table:
  - **BN=128**: a 64-wide column tile runs two column blocks at a 128-token
    pass and streams **every weight byte twice**. Removing the dequant alone
    (64×64 tile) bought only +8.6% — the kernel was traffic-bound, not
    dequant-bound. BN=128 reads each cached weight once.
  - **BM=64, two stages (30.7 KB smem)**: the 128×128 variant (61.4 KB, one
    block per SM) collapsed the grid — a `rows=256` GEMM was 2 blocks on
    72 SMs — and prefill regressed 4,164 → 2,486 tok/s. BM=64 restores 4× the
    blocks and the two-resident-blocks-per-SM overlap.
  - **32×32 register-blocked warp tile** (2×2 accumulator fragments in a
    2×4 warp grid): with the aspect-ratio problem fixed by split-K, every big
    shape measured a uniform ~30 TFLOPS. That is the shared-memory bandwidth
    ceiling of a 16-row warp tile — 2.5 KB of `ldmatrix` traffic per 4 MMAs.
    The 32×32 warp tile cuts bytes per MAC by 5× and moves the limit toward
    the tensor cores. The first attempt used 32×64 warp tiles, which
    **over-cover** the 64×128 block (8 warps for 4 slots): the extra four
    warps read columns 128–255 of a 128-column B stage and faulted the
    context — the async error then surfaced as
    `cuModuleLoadData: ILLEGAL_ADDRESS` at the next kernel compile. The
    `static_assert`s in the helper now make "the warp grid must tile the
    block" a compile-time contract, checked by nvcc and by the headless
    clang parse.

**Where the tile search stopped (five variants, one winner).** The
64×128 / 30.7 KB / two-resident-blocks-per-SM configuration is the measured
optimum of this WMMA family on the A10, at a uniform ~30 TFLOPS — close to
the weight-stream fill rate, not the tensor peak. Every configuration that
spent more shared memory lost: 128×128 at 61.4 KB (−40%), the 128-row tall
tile at 41 KB (−12% end-to-end, tall shapes 1.26–2.07× per call), even
though both halved the activation re-reads the byte model blamed. Those
re-reads were L2 hits all along (the 384 KB X strip is resident); what
actually hides the per-block fill latency is the **second resident block**,
and 30.7 KB is the budget that keeps two. Treat smem > 30 KB on this kernel
family as an occupancy regression until measured otherwise
(`KSEARCH_MM_TALL=1` re-enables the tall tile for A/B on other parts).
Beyond this point the prefill gap lives in attention (4.3 ms by phase
split, untouched) and small-GEMM latency floors — not the tile.

**The epilogue was the hidden 2×.** The C store is `C[n * M + m]`
(token-major), and the original lane mapping varied `n` across consecutive
lanes — so every 2-byte store landed in its own 32-byte sector: a 64×128
tile (16 KB useful) emitted 256 KB of write traffic per block. The A10
per-shape times fit `compute floor + epilogue sectors` almost exactly
(`6144×1536`: 38 + 41 ≈ 85 µs; lm_head: 56 + 60 ≈ 118 µs). Swapping the
mapping so consecutive lanes vary `m` (the contiguous axis) coalesces the
writes — a two-line change per epilogue, applied to all five tile helpers.
This is the largest remaining GEMM lever and it is orthogonal to the tile
shape.
- Kernels with ≥ 24 KB dynamic smem request a max carveout hint
  (`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`).
- **Split-K for wide-K / short-M shapes** (`rows <= 2048 && cols >= 2048`,
  `KSEARCH_MM_SPLIT` overrides, default 4): the down-projections. A10
  per-shape profiling showed the aspect-ratio signature — tall-M shapes
  reached 33–57 TFLOPS while `1536×12288` ran at 13.7, because `M = 1536` is
  only 24 blocks on 72 SMs with a 384-step K-loop. `grid.z = split` gives
  each block a K-slice into F32 partials `[split, batch, rows]`, and the
  runtime `ks_f16_mm_reduce` sums the planes into the F16 result. The graph
  op is `Graph::mul_mm_f16_split` (CUDA-only; MSL render refuses it).
- Numerics: the dequant tile already rounded weights to half in shared memory,
  so the cache stores the same values the WMMA path consumed. The `"Hi"`
  gate is unchanged.
- Cost: ~2.6 B of VRAM per weight element (E2B ≈ +1.6 GB, E4B ≈ +3.6 GB on
  top of the packed weights) and a few ms of one-time dequant per tensor,
  paid on the first large prefill. Opt out with `KSEARCH_PREFILL_F16=0`.

Decode is untouched by design: a gemv must read the packed 4.5-bit form
(3.5× fewer bytes than F16), so the cache is built lazily and only the
batch > 8 GEMM path consults it.

## Attribution on device (`KSEARCH_MM_PROF`)

The A10 answered the plateau question. `KSEARCH_MM_PROF=1` brackets every
prefill GEMM with device events and prints calls / total ms / ms-per-call /
effective TFLOPS per shape, sorted by time. The measured table was not a
uniform ceiling: tall-M shapes ran at 33–57 TFLOPS while their transposes
(the down-projections, `1536×12288` at 13.7) collapsed — 24 blocks on 72 SMs
with a 384-step K-loop. That finding became the split-K path above. What
remains after split-K lands is judged the same way:

- still-slow big shapes → the WMMA tile itself (`mma.sync` / `ldmatrix`
  route),
- small shapes at per-call floors (k/v at 2.4 TFLOPS × 59 calls, PLE) →
  fusing q/k/v at dequant-cache time.

Related knobs: `KSEARCH_PREFILL_PROF=1` prints the coarse per-phase split
(qkv / attn / o / ffn / ple), and `KSEARCH_PHASE_GPUTIME=1` reports decode
phases from device events. The A10 also showed ~10% run-to-run clock variance
and refuses `nvidia-smi -lgc` without root — treat cross-run deltas under
~10% as noise and prefer same-run A/Bs (`KSEARCH_MM_SPLIT`,
`KSEARCH_PREFILL_F16`).

## Attention knobs

The partitioned MWG SDPA (pass 1 + reduce) takes **two independent** KV-slice
counts, because the A10 measured them pulling in opposite directions at a
128-256-token context:

| Path | Knob | Default |
|---|---|---|
| decode (`sdpa_mwg_kv`) | `KSEARCH_SDPA_NWG` | 16 |
| prefill batch (`sdpa_mwg_kv_batch`) | `KSEARCH_SDPA_NWG_PREFILL` | 8 |

Prefill at NWG=8 measured +11% pp128 over 16; decode at 16 measured +4% over
8. Both clamp to 1..=64; the shared partials buffer is sized for the larger
of the two. Callers that size `tmp` read `Eng::sdpa_mwg_nwg()` /
`Eng::sdpa_mwg_nwg_prefill()`, not the const.

## What is verified, and where

Headless (any machine, including this repo's Apple dev box):

```bash
cargo test --workspace                              # 61 tests, no GPU required
cargo test -p ksearch_codegen --test cuda_codegen   # dialect, host C++ parse, determinism
cargo test -p ksearch_cuda                          # launch mapping, nvcc -arch targeting
cargo run -p ksearch_cli --release -- cuda --out /tmp/cu
cargo build -p ksearch_cli --no-default-features --features cuda   # NVIDIA-only build
cargo build -p ksearch_cli --no-default-features --features metal  # Apple-only build
otool -L target/debug/ksearch | grep -c Metal.framework            # 0 after the cuda-only build
```

The `nvcc` tests drive the real `nvcc_cubin` entry point with a **stub** `nvcc`
that records its argv, which is what proves `-arch=sm_90` for an H100 and
`-arch=sm_120` for an RTX 5090 without either card present.

On a CUDA box, the same tests additionally open the driver: `gpu_round_trip_when_available`
in `ksearch_cuda`, `gpu_copy_round_trip_when_available` in
`codegen_launch.rs`, and the `matvec --backend cuda` CPU-reference gate. The
model commands are the end-to-end check:

```bash
ksearch generate --backend cuda --gguf model.gguf --prompt "Hi" --n-predict 8
ksearch serve    --backend cuda --gguf model.gguf --slots 4   # then curl the endpoint
```

Honest limits:

- The prefill tensor-core tiles (Q4_K/Q6_K dequant-in-loop and the F16 cache
  tile) are WMMA `m16n16k16`, not hand-tuned `mma.sync` with `ldmatrix` and a
  swizzled shared-memory layout — llama.cpp's MMQ and cuBLAS still hold a
  per-GEMM edge there. That is the next prefill lever if the F16 cache is not
  enough.
- `KSEARCH_PROFILE` numbers are host-side; `KSEARCH_PHASE_GPUTIME` is the
  GPU-side one (device events on CUDA, counters on Metal).
- This repository's CI/dev machine is an Apple GPU host: CUDA correctness is
  covered by codegen, launch mapping and the `nvcc` targeting tests above, but
  the driver round-trips only run on an NVIDIA box. The F16 cache path follows
  the same rule — it is verified by the clang++ type-check of the rendered
  translation unit here, and by the A10 run where the numbers live.

## Checkpoint

1. Which env var forces CUDA for `serve`? Which two CLI spellings set it?
2. Where does `thread_index_in_simdgroup` land in CUDA, and why is the block
   shape `(32, nsg)`?
3. Why is `libcuda` loaded with `dlopen` instead of linked?
4. You are on an H100 and `ksearch cuda --compile` fails. What is the first
   thing to check, and which env var overrides it?

Answers live in [../crates/ksearch_cuda/src/lib.rs](../crates/ksearch_cuda/src/lib.rs)
and [../crates/ksearch_codegen/src/cuda_prelude.rs](../crates/ksearch_codegen/src/cuda_prelude.rs).

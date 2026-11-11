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
- **Telemetry**: there are no GPU counters here. `Context::phase_mark` /
  `resolve_phase_times_ms` record host-side phase times instead, and
  `take_telemetry()` reports dispatches with a host-measured enqueue time. Do
  not compare those numbers against Metal's GPU timings.

## What is verified, and where

Headless (any machine, including this repo's Apple dev box):

```bash
cargo test --workspace                              # 56 tests, no GPU required
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

- The generated GEMM is a tiled FMA kernel, not a tensor-core (MMA) kernel; the
  Metal path uses `simdgroup_matrix` where the CUDA path cannot.
- CUDA has no GPU-side timers in this stack, so `KSEARCH_PROFILE` numbers are
  host-side.
- This repository's CI/dev machine is an Apple GPU host: CUDA correctness is
  covered by codegen, launch mapping and the `nvcc` targeting tests above, but
  the driver round-trips only run on an NVIDIA box.

## Checkpoint

1. Which env var forces CUDA for `serve`? Which two CLI spellings set it?
2. Where does `thread_index_in_simdgroup` land in CUDA, and why is the block
   shape `(32, nsg)`?
3. Why is `libcuda` loaded with `dlopen` instead of linked?
4. You are on an H100 and `ksearch cuda --compile` fails. What is the first
   thing to check, and which env var overrides it?

Answers live in [../crates/ksearch_cuda/src/lib.rs](../crates/ksearch_cuda/src/lib.rs)
and [../crates/ksearch_codegen/src/cuda_prelude.rs](../crates/ksearch_codegen/src/cuda_prelude.rs).

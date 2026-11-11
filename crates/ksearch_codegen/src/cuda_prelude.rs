//! CUDA dialect shim + CUDA-native helper kernels.
//!
//! The emitter is shared with the Metal backend: kernels are written in a small
//! MSL-flavoured dialect (address-space qualifiers, `[[buffer(n)]]` attributes,
//! `simd_*` intrinsics, MSL vector types such as `float4`/`half4`/`uchar2`) and
//! [`CUDA_PRELUDE`] maps that dialect onto CUDA C++.
//!
//! When `__CUDACC__` is undefined — i.e. a plain host C++ compile such as
//! `clang++ -x c++ -fsyntax-only` — a small set of stubs stands in for the CUDA
//! builtins so the very same translation unit still parses and type-checks.
//! That path is never executed; it exists so CI can validate generated kernels
//! without a GPU toolchain.
//!
//! Thread-mapping contract (see `render_kernel` in `render.rs`):
//!
//! | MSL                                   | CUDA                                        |
//! |---------------------------------------|---------------------------------------------|
//! | `thread_position_in_grid`             | `blockIdx.x * blockDim.x + threadIdx.x`      |
//! | `threadgroup_position_in_grid`        | `blockIdx.x`                                |
//! | `thread_index_in_threadgroup`         | `threadIdx.x`                               |
//! | `thread_index_in_simdgroup`           | `threadIdx.x` (threadgroup is `(32, nsg)`)   |
//! | `simdgroup_index_in_threadgroup`      | `threadIdx.y` (threadgroup is `(32, nsg)`)   |
//! | `simd_sum`                            | warp `__shfl_xor_sync` butterfly             |
//! | `simd_shuffle_xor/down`               | `__shfl_xor_sync` / `__shfl_down_sync`       |
//! | `threadgroup_barrier`                 | `__syncthreads`                              |

/// MSL → CUDA dialect shim, emitted at the top of every generated CUDA kernel.
pub const CUDA_PRELUDE: &str = r#"
/* ===================== ksearch CUDA dialect shim ====================== */
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
/* `__device__` only. Call sites already say `KS_DEVICE inline`.
   `__forceinline__` is itself an inline specifier, and nvcc rejects
   `__forceinline__ inline` as a duplicate specifier. */
#define KS_DEVICE __device__
#define KS_KERNEL extern "C" __global__
#else
/* Host-only syntax-check build: no GPU toolchain required. Never executed. */
#include <math.h>
#include <stdint.h>
#define KS_DEVICE
#define KS_KERNEL extern "C"
#define __device__
#define __host__
#define __global__
#define __shared__
#define __constant__
#define __restrict__
/* `__align__(N)` marks a declaration alignment in CUDA. */
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ inline
#define __syncthreads() ((void)0)
#define __syncwarp() ((void)0)
/* `rsqrtf` is a CUDA device builtin, not standard C. */
static inline float rsqrtf(float v) { return 1.0f / sqrtf(v); }
struct ks_uint3_host { unsigned int x, y, z; };
extern const ks_uint3_host blockIdx, threadIdx, blockDim, gridDim;
KS_DEVICE inline float __shfl_xor_sync(unsigned m, float v, int) { (void)m; return v; }
KS_DEVICE inline unsigned int __shfl_xor_sync(unsigned m, unsigned int v, int) { (void)m; return v; }
KS_DEVICE inline float __shfl_down_sync(unsigned m, float v, int) { (void)m; return v; }
KS_DEVICE inline unsigned int __shfl_down_sync(unsigned m, unsigned int v, int) { (void)m; return v; }
#endif

/* MSL address-space words (`device`, `thread`, `threadgroup`, `constant`,
   `precise`) are erased by the renderer before this file is compiled.
   A preprocessor macro named `device` is illegal here: nvcc expands it
   inside the `__device__` keyword and every helper becomes a host function.
   Shared-memory declarations: emitted as `KS_SMEM float tgN[N];` by the
   backend so the same statement works for `threadgroup` and `__shared__`.
   16-byte aligned because the bodies reinterpret these tiles as `float4`. */
#define KS_SMEM __shared__ __align__(16)

/* ---- MSL scalar spellings ---- */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

/* ---- __half (host syntax-check stub) ---- */
#if !defined(__CUDACC__)
struct __half {
  unsigned short bits;
  __half() : bits(0u) {}
  __half(float f) : bits(0u) { (void)f; }
  operator float() const { return 0.0f; }
};
#endif
typedef __half half;

/* ---- MSL vector types (POD, identical size/alignment to MSL) ---- */
struct ks_half2;
struct ks_half4;

struct ks_uint2 {
  uint x, y;
  KS_DEVICE ks_uint2() : x(0u), y(0u) {}
  KS_DEVICE ks_uint2(uint a, uint b) : x(a), y(b) {}
};

struct ks_uint4 {
  uint x, y, z, w;
  KS_DEVICE ks_uint4() : x(0u), y(0u), z(0u), w(0u) {}
  KS_DEVICE ks_uint4(uint a) : x(a), y(a), z(a), w(a) {}
  KS_DEVICE ks_uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
  KS_DEVICE uint& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const uint& operator[](int i) const { return (&x)[i]; }
};

struct ks_int4 {
  int x, y, z, w;
  KS_DEVICE ks_int4() : x(0), y(0), z(0), w(0) {}
  KS_DEVICE ks_int4(int a) : x(a), y(a), z(a), w(a) {}
  KS_DEVICE ks_int4(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
  KS_DEVICE ks_int4(const ks_uint4& u) : x((int)u.x), y((int)u.y), z((int)u.z), w((int)u.w) {}
  KS_DEVICE int& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const int& operator[](int i) const { return (&x)[i]; }
};

struct ks_uchar2 {
  uchar x, y;
  KS_DEVICE ks_uchar2() : x(0u), y(0u) {}
  KS_DEVICE ks_uchar2(uchar a, uchar b) : x(a), y(b) {}
  KS_DEVICE uchar& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const uchar& operator[](int i) const { return (&x)[i]; }
};

struct ks_packed_uchar4 {
  uchar x, y, z, w;
  KS_DEVICE ks_packed_uchar4() : x(0u), y(0u), z(0u), w(0u) {}
  KS_DEVICE ks_packed_uchar4(uchar a, uchar b, uchar c, uchar d) : x(a), y(b), z(c), w(d) {}
  KS_DEVICE uchar& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const uchar& operator[](int i) const { return (&x)[i]; }
};

struct ks_ushort4 {
  ushort x, y, z, w;
  KS_DEVICE ks_ushort4() : x(0u), y(0u), z(0u), w(0u) {}
  KS_DEVICE ks_ushort4(ushort a, ushort b, ushort c, ushort d) : x(a), y(b), z(c), w(d) {}
  KS_DEVICE ushort& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const ushort& operator[](int i) const { return (&x)[i]; }
};

struct ks_float2 {
  float x, y;
  KS_DEVICE ks_float2() : x(0.f), y(0.f) {}
  KS_DEVICE ks_float2(float a) : x(a), y(a) {}
  KS_DEVICE ks_float2(float a, float b) : x(a), y(b) {}
  KS_DEVICE ks_float2(const ks_half2& h);
  KS_DEVICE float& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const float& operator[](int i) const { return (&x)[i]; }
};

struct ks_float4 {
  float x, y, z, w;
  KS_DEVICE ks_float4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
  KS_DEVICE ks_float4(float a) : x(a), y(a), z(a), w(a) {}
  KS_DEVICE ks_float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
  KS_DEVICE ks_float4(const ks_half4& h);
  KS_DEVICE ks_float4(const ks_int4& v)
      : x((float)v.x), y((float)v.y), z((float)v.z), w((float)v.w) {}
  KS_DEVICE float& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const float& operator[](int i) const { return (&x)[i]; }
};

struct ks_half2 {
  __half x, y;
  KS_DEVICE ks_half2() : x(__half(0.f)), y(__half(0.f)) {}
  KS_DEVICE ks_half2(__half a, __half b) : x(a), y(b) {}
  KS_DEVICE ks_half2(const ks_float2& f) : x(__half(f.x)), y(__half(f.y)) {}
};

struct ks_half4 {
  __half x, y, z, w;
  KS_DEVICE ks_half4()
      : x(__half(0.f)), y(__half(0.f)), z(__half(0.f)), w(__half(0.f)) {}
  KS_DEVICE ks_half4(const ks_float4& f)
      : x(__half(f.x)), y(__half(f.y)), z(__half(f.z)), w(__half(f.w)) {}
  KS_DEVICE __half& operator[](int i) { return (&x)[i]; }
  KS_DEVICE const __half& operator[](int i) const { return (&x)[i]; }
};

struct ks_half2x4 { __half v[8]; };

struct ks_float4x4 {
  ks_float4 cols[4];
  KS_DEVICE ks_float4& operator[](int i) { return cols[i]; }
  KS_DEVICE const ks_float4& operator[](int i) const { return cols[i]; }
};

KS_DEVICE inline ks_float2::ks_float2(const ks_half2& h)
    : x((float)h.x), y((float)h.y) {}
KS_DEVICE inline ks_float4::ks_float4(const ks_half4& h)
    : x((float)h.x), y((float)h.y), z((float)h.z), w((float)h.w) {}

/* ---- float4 arithmetic ---- */
KS_DEVICE inline ks_float4 operator+(ks_float4 a, ks_float4 b) {
  return ks_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
KS_DEVICE inline ks_float4 operator-(ks_float4 a, ks_float4 b) {
  return ks_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
KS_DEVICE inline ks_float4 operator*(ks_float4 a, ks_float4 b) {
  return ks_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
KS_DEVICE inline ks_float4 operator/(ks_float4 a, ks_float4 b) {
  return ks_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
KS_DEVICE inline ks_float4 operator*(float s, ks_float4 a) {
  return ks_float4(s * a.x, s * a.y, s * a.z, s * a.w);
}
KS_DEVICE inline ks_float4 operator*(ks_float4 a, float s) {
  return ks_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
KS_DEVICE inline ks_float4 operator/(ks_float4 a, float s) {
  return ks_float4(a.x / s, a.y / s, a.z / s, a.w / s);
}
KS_DEVICE inline ks_float4 operator-(ks_float4 a) {
  return ks_float4(-a.x, -a.y, -a.z, -a.w);
}
KS_DEVICE inline ks_float4& operator+=(ks_float4& a, ks_float4 b) { a = a + b; return a; }
KS_DEVICE inline ks_float4& operator-=(ks_float4& a, ks_float4 b) { a = a - b; return a; }
KS_DEVICE inline ks_float4& operator*=(ks_float4& a, ks_float4 b) { a = a * b; return a; }
KS_DEVICE inline ks_float4& operator/=(ks_float4& a, ks_float4 b) { a = a / b; return a; }
KS_DEVICE inline bool operator==(ks_float4 a, ks_float4 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
KS_DEVICE inline bool operator!=(ks_float4 a, ks_float4 b) { return !(a == b); }

/* ---- float2 arithmetic ---- */
KS_DEVICE inline ks_float2 operator+(ks_float2 a, ks_float2 b) {
  return ks_float2(a.x + b.x, a.y + b.y);
}
KS_DEVICE inline ks_float2 operator-(ks_float2 a, ks_float2 b) {
  return ks_float2(a.x - b.x, a.y - b.y);
}
KS_DEVICE inline ks_float2 operator*(ks_float2 a, ks_float2 b) {
  return ks_float2(a.x * b.x, a.y * b.y);
}
KS_DEVICE inline ks_float2 operator*(float s, ks_float2 a) {
  return ks_float2(s * a.x, s * a.y);
}
KS_DEVICE inline ks_float2& operator+=(ks_float2& a, ks_float2 b) { a = a + b; return a; }

/* ---- integer vector bit ops ---- */
KS_DEVICE inline ks_uint4 operator&(ks_uint4 a, ks_uint4 b) {
  return ks_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}
KS_DEVICE inline ks_uint4 operator&(ks_uint4 a, uint b) {
  return ks_uint4(a.x & b, a.y & b, a.z & b, a.w & b);
}
KS_DEVICE inline ks_uint4 operator|(ks_uint4 a, ks_uint4 b) {
  return ks_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}
KS_DEVICE inline ks_uint4 operator^(ks_uint4 a, ks_uint4 b) {
  return ks_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}
KS_DEVICE inline ks_uint4 operator>>(ks_uint4 a, uint s) {
  return ks_uint4(a.x >> s, a.y >> s, a.z >> s, a.w >> s);
}
KS_DEVICE inline ks_uint4 operator<<(ks_uint4 a, uint s) {
  return ks_uint4(a.x << s, a.y << s, a.z << s, a.w << s);
}
KS_DEVICE inline ks_uint4 operator~(ks_uint4 a) {
  return ks_uint4(~a.x, ~a.y, ~a.z, ~a.w);
}
KS_DEVICE inline ks_uint4 operator+(ks_uint4 a, ks_uint4 b) {
  return ks_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
KS_DEVICE inline ks_int4 operator&(ks_int4 a, int b) {
  return ks_int4(a.x & b, a.y & b, a.z & b, a.w & b);
}
KS_DEVICE inline ks_int4 operator>>(ks_int4 a, int s) {
  return ks_int4(a.x >> s, a.y >> s, a.z >> s, a.w >> s);
}
KS_DEVICE inline ks_int4 operator*(ks_int4 a, ks_int4 b) {
  return ks_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
KS_DEVICE inline ks_int4 operator-(ks_int4 a, ks_int4 b) {
  return ks_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/* ---- vector helpers ---- */
KS_DEVICE inline float ks_dot(ks_float4 a, ks_float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
KS_DEVICE inline float ks_dot(ks_float2 a, ks_float2 b) {
  return a.x * b.x + a.y * b.y;
}

/* ---- scalar math ---- */
KS_DEVICE inline float ks_round(float x) { return roundf(x); }
KS_DEVICE inline float ks_floor(float x) { return floorf(x); }
KS_DEVICE inline float ks_ceil(float x) { return ceilf(x); }
KS_DEVICE inline float ks_exp(float x) { return expf(x); }
KS_DEVICE inline float ks_sqrt(float x) { return sqrtf(x); }
KS_DEVICE inline float ks_rsqrt(float x) { return rsqrtf(x); }
KS_DEVICE inline float ks_tanh(float x) { return tanhf(x); }
KS_DEVICE inline float ks_fabs(float x) { return fabsf(x); }
KS_DEVICE inline float ks_abs(float x) { return fabsf(x); }
KS_DEVICE inline int ks_abs(int x) { return x < 0 ? -x : x; }

template <class A, class B>
KS_DEVICE inline A ks_min(A a, B b) { return a < (A)b ? a : (A)b; }
template <class A, class B>
KS_DEVICE inline A ks_max(A a, B b) { return a > (A)b ? a : (A)b; }
template <class T, class L, class H>
KS_DEVICE inline T ks_clamp(T v, L lo, H hi) {
  return v < (T)lo ? (T)lo : (v > (T)hi ? (T)hi : v);
}

/* ---- warp intrinsics (MSL simd_* → CUDA warp shuffles) ---- */
KS_DEVICE inline void ks_block_barrier() { __syncthreads(); }
KS_DEVICE inline void ks_warp_barrier() { __syncwarp(); }

template <class T>
KS_DEVICE inline T ks_simd_sum(T v) {
  float f = (float)v;
#if defined(__CUDACC__)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    f = f + __shfl_xor_sync(0xffffffffu, f, o);
  }
#endif
  return (T)f;
}

KS_DEVICE inline float ks_simd_shuffle_down(float v, uint d) {
  return __shfl_down_sync(0xffffffffu, v, (int)d);
}
KS_DEVICE inline uint ks_simd_shuffle_down(uint v, uint d) {
  return __shfl_down_sync(0xffffffffu, v, (int)d);
}
KS_DEVICE inline float ks_simd_shuffle_xor(float v, uint s) {
  return __shfl_xor_sync(0xffffffffu, v, (int)s);
}
KS_DEVICE inline uint ks_simd_shuffle_xor(uint v, uint s) {
  return __shfl_xor_sync(0xffffffffu, v, (int)s);
}

/* ---- MSL spellings → shim names ---- */
#define uchar2 ks_uchar2
#define packed_uchar4 ks_packed_uchar4
#define ushort4 ks_ushort4
#define float2 ks_float2
#define float4 ks_float4
#define int4 ks_int4
#define uint2 ks_uint2
#define uint4 ks_uint4
#define half2 ks_half2
#define half4 ks_half4
#define half2x4 ks_half2x4
#define float4x4 ks_float4x4
#define dot(a, b) ks_dot((a), (b))
#define simd_sum(x) ks_simd_sum(x)
#define simd_shuffle_down(v, d) ks_simd_shuffle_down((v), (d))
#define simd_shuffle_xor(v, s) ks_simd_shuffle_xor((v), (s))
#define simdgroup_barrier(x) ks_warp_barrier()
#define threadgroup_barrier(x) ks_block_barrier()
#define clamp(v, lo, hi) ks_clamp((v), (lo), (hi))
#define min(a, b) ks_min((a), (b))
#define max(a, b) ks_max((a), (b))
#define abs(x) ks_abs(x)
#define fabs(x) ks_fabs(x)
#define round(x) ks_round(x)
#define floor(x) ks_floor(x)
#define ceil(x) ks_ceil(x)
#define exp(x) ks_exp(x)
#define sqrt(x) ks_sqrt(x)
#define rsqrt(x) ks_rsqrt(x)
#define tanh(x) ks_tanh(x)
"#;

/// Q4_K prefill GEMM, CUDA flavour.
///
/// `C[n * M + m]`, `W` is `[M, K]` Q4_K, `X` is `[N, K]` F16. A 64×64 output
/// block: each warp owns 16 rows and 32 columns and multiplies with
/// `wmma::mma_sync` (`m16n16k16`). Every thread dequantizes 16 Q4_K weights
/// (one scale, one vector load) into the shared `half` tile, and the
/// activation tile is copied with `cp.async` so that copy overlaps the
/// dequant. Two stages are in flight. Q4_K stays a load expand.
pub const Q4K_MUL_MM_CUDA_HELPER: &str = r#"
#define KS_MM_BM 64u
#define KS_MM_BN 64u
#define KS_MM_BK 64u
#define KS_MM_LD (KS_MM_BK + 8u)

#ifndef KS_QLOAD_FN
#define KS_QLOAD_FN
typedef void (*ks_dequant16_fn)(device const uchar*, uint, half*);
typedef float (*ks_qload1_fn)(device const uchar*, uint);
#endif

#ifndef KS_MM_GEMM_DEFINED
#define KS_MM_GEMM_DEFINED
KS_DEVICE void ks_dequant_q4k16(device const uchar* A, uint idx, half* dst);
KS_DEVICE inline void ks_cp_async16(half* dst, device const half* src) {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  unsigned smem = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(src));
#else
  for (int i = 0; i < 8; ++i) dst[i] = src[i];
#endif
}
KS_DEVICE inline void ks_cp_commit() {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
KS_DEVICE inline void ks_cp_wait() {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

KS_DEVICE inline void ks_mm_fill(
    device const uchar* W, device const half* X,
    uint M, uint K, uint N, uint m0, uint n0, uint k0,
    half* As, half* Bs, uint tid,
    ks_dequant16_fn dequant, ks_qload1_fn load1) {
  const uint bparts = KS_MM_BK / 8u;
  const uint ncp = KS_MM_BN * bparts;
  for (uint c = tid; c < ncp; c += blockDim.x) {
    const uint ni = c / bparts;
    const uint part = c - ni * bparts;
    const uint n = n0 + ni;
    half* dst = Bs + ni * KS_MM_LD + part * 8u;
    if (n < N && k0 + KS_MM_BK <= K) {
      ks_cp_async16(dst, X + n * K + k0 + part * 8u);
    } else {
      for (uint t = 0u; t < 8u; ++t) {
        const uint k = k0 + part * 8u + t;
        dst[t] = (n < N && k < K) ? X[n * K + k] : half(0.0f);
      }
    }
  }
  ks_cp_commit();
  const uint ap = KS_MM_BK / 16u;
  const uint nch = KS_MM_BM * ap;
  const bool full = (k0 + KS_MM_BK) <= K;
  for (uint c = tid; c < nch; c += blockDim.x) {
    const uint row = c / ap;
    const uint part = c - row * ap;
    const uint m = m0 + row;
    half* dst = As + row * KS_MM_LD + part * 16u;
    if (m < M && full) {
      ks_dequant_q4k16(W, m * K + k0 + part * 16u, dst);
    } else {
      for (uint t = 0u; t < 16u; ++t) {
        const uint k = k0 + part * 16u + t;
        dst[t] = (m < M && k < K) ? half(ksearch_load_q4k(W, m * K + k)) : half(0.0f);
      }
    }
  }
}

/* 8 warps. Warp (warp_m, warp_n) owns 16 rows and 32 columns (two m16n16). */
KS_DEVICE inline void ks_mm_gemm(
    device const uchar* W, device const half* X, device half* C,
    uint M, uint K, uint N,
    uint2 tgpig, ushort tiitg, ushort sgitg, threadgroup char* shmem,
    ks_dequant16_fn dequant, ks_qload1_fn load1) {
  (void)sgitg;
  const uint tid = (uint)tiitg;
  const uint m0 = tgpig.y * KS_MM_BM;
  const uint n0 = tgpig.x * KS_MM_BN;
  const uint stage_h = (KS_MM_BM + KS_MM_BN) * KS_MM_LD;
  half* buf0 = (half*)shmem;
  half* buf1 = buf0 + stage_h;
#if defined(__CUDACC__)
  const uint lane = tid & 31u;
  const uint warp = tid >> 5;
  const uint warp_m = warp & 3u;
  const uint warp_n = warp >> 2;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cfrag[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) nvcuda::wmma::fill_fragment(cfrag[j], 0.0f);
#endif
  const uint nsteps = (K + KS_MM_BK - 1u) / KS_MM_BK;
  uint stage = 0u;
  if (nsteps > 0u) {
    half* As = buf0;
    half* Bs = As + KS_MM_BM * KS_MM_LD;
    ks_mm_fill(W, X, M, K, N, m0, n0, 0u, As, Bs, tid, dequant, load1);
    ks_cp_wait();
    ks_block_barrier();
  }
  for (uint step = 0u; step < nsteps; ++step) {
    half* cur = (stage == 0u) ? buf0 : buf1;
    half* nxt = (stage == 0u) ? buf1 : buf0;
    const uint next = step + 1u;
    if (next < nsteps) {
      ks_mm_fill(W, X, M, K, N, m0, n0, next * KS_MM_BK, nxt, nxt + KS_MM_BM * KS_MM_LD, tid, dequant, load1);
    }
#if defined(__CUDACC__)
    half* As = cur;
    half* Bs = As + KS_MM_BM * KS_MM_LD;
#pragma unroll
    for (int kstep = 0; kstep < (int)(KS_MM_BK / 16u); ++kstep) {
      const uint kk = (uint)kstep * 16u;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag;
      nvcuda::wmma::load_matrix_sync(a_frag, As + (warp_m * 16u) * KS_MM_LD + kk, KS_MM_LD);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint col = warp_n * 32u + (uint)j * 16u;
        nvcuda::wmma::load_matrix_sync(b_frag, Bs + col * KS_MM_LD + kk, KS_MM_LD);
        nvcuda::wmma::mma_sync(cfrag[j], a_frag, b_frag, cfrag[j]);
      }
    }
#endif
    if (next < nsteps) {
      ks_cp_wait();
      ks_block_barrier();
      stage ^= 1u;
    }
  }
#if defined(__CUDACC__)
  ks_block_barrier();
  float* mine = (float*)shmem + warp * 256u;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    nvcuda::wmma::store_matrix_sync(mine, cfrag[j], 16, nvcuda::wmma::mem_row_major);
    for (uint e = lane; e < 256u; e += 32u) {
      const uint i = e >> 4;
      const uint col = e & 15u;
      const uint m = m0 + warp_m * 16u + i;
      const uint n = n0 + warp_n * 32u + (uint)j * 16u + col;
      if (m < M && n < N) C[n * M + m] = half(mine[i * 16u + col]);
    }
  }
#else
  (void)C;
  (void)tid;
#endif
}
#endif

KS_DEVICE void ks_dequant_q4k16(device const uchar* A, uint idx, half* dst) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  device const uchar* blk = A + (idx / QK) * BPB;
  float d = float(*(device const half*)blk);
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  const uint j = idx % QK;
  const uint group = j / 64u;
  const uint jl = j % 64u;
  const uint is = group * 2u + (jl >= 32u ? 1u : 0u);
  const uint qbase = group * 32u + (jl & 16u);
  uchar sc, mn;
  ksearch_get_scale_min_k4(is, scales, sc, mn);
  const float scale = d * float(sc);
  const float m = dmin * float(mn);
  const bool hi = jl >= 32u;
#pragma unroll
  for (uint l = 0u; l < 16u; l += 4u) {
    const uint bits = *(device const uint*)(qs + qbase + l);
    float n0, n1, n2, n3;
    if (!hi) {
      n0 = float(bits & 0xFu);
      n1 = float((bits >> 8) & 0xFu);
      n2 = float((bits >> 16) & 0xFu);
      n3 = float((bits >> 24) & 0xFu);
    } else {
      n0 = float((bits >> 4) & 0xFu);
      n1 = float((bits >> 12) & 0xFu);
      n2 = float((bits >> 20) & 0xFu);
      n3 = float((bits >> 28) & 0xFu);
    }
    dst[l + 0u] = half(scale * n0 - m);
    dst[l + 1u] = half(scale * n1 - m);
    dst[l + 2u] = half(scale * n2 - m);
    dst[l + 3u] = half(scale * n3 - m);
  }
}

KS_DEVICE inline void ksearch_q4k_mul_mm(
    device const uchar* W, device const half* X, device half* C,
    uint M, uint K, uint N,
    uint2 tgpig, ushort tiitg, ushort sgitg, threadgroup char* shmem) {
  ks_mm_gemm(W, X, C, M, K, N, tgpig, tiitg, sgitg, shmem, ks_dequant_q4k16, ksearch_load_q4k);
}
"#;

/// Q6_K prefill GEMM, CUDA flavour (see [`Q4K_MUL_MM_CUDA_HELPER`]).
pub const Q6K_MUL_MM_CUDA_HELPER: &str = r#"
#define KS_MM_BM 64u
#define KS_MM_BN 64u
#define KS_MM_BK 64u
#define KS_MM_LD (KS_MM_BK + 8u)

#ifndef KS_QLOAD_FN
#define KS_QLOAD_FN
typedef void (*ks_dequant16_fn)(device const uchar*, uint, half*);
typedef float (*ks_qload1_fn)(device const uchar*, uint);
#endif

#ifndef KS_MM_GEMM_DEFINED
#define KS_MM_GEMM_DEFINED
KS_DEVICE inline void ks_cp_async16(half* dst, device const half* src) {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  unsigned smem = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(src));
#else
  for (int i = 0; i < 8; ++i) dst[i] = src[i];
#endif
}
KS_DEVICE inline void ks_cp_commit() {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
KS_DEVICE inline void ks_cp_wait() {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}
KS_DEVICE inline void ks_mm_fill(
    device const uchar* W, device const half* X,
    uint M, uint K, uint N, uint m0, uint n0, uint k0,
    half* As, half* Bs, uint tid,
    ks_dequant16_fn dequant, ks_qload1_fn load1) {
  const uint bparts = KS_MM_BK / 8u;
  const uint ncp = KS_MM_BN * bparts;
  for (uint c = tid; c < ncp; c += blockDim.x) {
    const uint ni = c / bparts;
    const uint part = c - ni * bparts;
    const uint n = n0 + ni;
    half* dst = Bs + ni * KS_MM_LD + part * 8u;
    if (n < N && k0 + KS_MM_BK <= K) {
      ks_cp_async16(dst, X + n * K + k0 + part * 8u);
    } else {
      for (uint t = 0u; t < 8u; ++t) {
        const uint k = k0 + part * 8u + t;
        dst[t] = (n < N && k < K) ? X[n * K + k] : half(0.0f);
      }
    }
  }
  ks_cp_commit();
  const uint ap = KS_MM_BK / 16u;
  const uint nch = KS_MM_BM * ap;
  const bool full = (k0 + KS_MM_BK) <= K;
  for (uint c = tid; c < nch; c += blockDim.x) {
    const uint row = c / ap;
    const uint part = c - row * ap;
    const uint m = m0 + row;
    half* dst = As + row * KS_MM_LD + part * 16u;
    if (m < M && full) {
      dequant(W, m * K + k0 + part * 16u, dst);
    } else {
      for (uint t = 0u; t < 16u; ++t) {
        const uint k = k0 + part * 16u + t;
        dst[t] = (m < M && k < K) ? half(load1(W, m * K + k)) : half(0.0f);
      }
    }
  }
}
KS_DEVICE inline void ks_mm_gemm(
    device const uchar* W, device const half* X, device half* C,
    uint M, uint K, uint N,
    uint2 tgpig, ushort tiitg, ushort sgitg, threadgroup char* shmem,
    ks_dequant16_fn dequant, ks_qload1_fn load1) {
  (void)sgitg;
  const uint tid = (uint)tiitg;
  const uint m0 = tgpig.y * KS_MM_BM;
  const uint n0 = tgpig.x * KS_MM_BN;
  const uint stage_h = (KS_MM_BM + KS_MM_BN) * KS_MM_LD;
  half* buf0 = (half*)shmem;
  half* buf1 = buf0 + stage_h;
#if defined(__CUDACC__)
  const uint lane = tid & 31u;
  const uint warp = tid >> 5;
  const uint warp_m = warp & 3u;
  const uint warp_n = warp >> 2;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cfrag[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) nvcuda::wmma::fill_fragment(cfrag[j], 0.0f);
#endif
  const uint nsteps = (K + KS_MM_BK - 1u) / KS_MM_BK;
  uint stage = 0u;
  if (nsteps > 0u) {
    half* As = buf0;
    half* Bs = As + KS_MM_BM * KS_MM_LD;
    ks_mm_fill(W, X, M, K, N, m0, n0, 0u, As, Bs, tid, dequant, load1);
    ks_cp_wait();
    ks_block_barrier();
  }
  for (uint step = 0u; step < nsteps; ++step) {
    half* cur = (stage == 0u) ? buf0 : buf1;
    half* nxt = (stage == 0u) ? buf1 : buf0;
    const uint next = step + 1u;
    if (next < nsteps) {
      ks_mm_fill(W, X, M, K, N, m0, n0, next * KS_MM_BK, nxt, nxt + KS_MM_BM * KS_MM_LD, tid, dequant, load1);
    }
#if defined(__CUDACC__)
    half* As = cur;
    half* Bs = As + KS_MM_BM * KS_MM_LD;
#pragma unroll
    for (int kstep = 0; kstep < (int)(KS_MM_BK / 16u); ++kstep) {
      const uint kk = (uint)kstep * 16u;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag;
      nvcuda::wmma::load_matrix_sync(a_frag, As + (warp_m * 16u) * KS_MM_LD + kk, KS_MM_LD);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint col = warp_n * 32u + (uint)j * 16u;
        nvcuda::wmma::load_matrix_sync(b_frag, Bs + col * KS_MM_LD + kk, KS_MM_LD);
        nvcuda::wmma::mma_sync(cfrag[j], a_frag, b_frag, cfrag[j]);
      }
    }
#endif
    if (next < nsteps) {
      ks_cp_wait();
      ks_block_barrier();
      stage ^= 1u;
    }
  }
#if defined(__CUDACC__)
  ks_block_barrier();
  float* mine = (float*)shmem + warp * 256u;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    nvcuda::wmma::store_matrix_sync(mine, cfrag[j], 16, nvcuda::wmma::mem_row_major);
    for (uint e = lane; e < 256u; e += 32u) {
      const uint i = e >> 4;
      const uint col = e & 15u;
      const uint m = m0 + warp_m * 16u + i;
      const uint n = n0 + warp_n * 32u + (uint)j * 16u + col;
      if (m < M && n < N) C[n * M + m] = half(mine[i * 16u + col]);
    }
  }
#else
  (void)C;
  (void)tid;
#endif
}
#endif

KS_DEVICE inline void ks_dequant_q6k16(device const uchar* A, uint idx, half* dst) {
#pragma unroll
  for (uint l = 0u; l < 16u; ++l) dst[l] = half(ksearch_load_q6k(A, idx + l));
}

KS_DEVICE inline void ksearch_q6k_mul_mm(
    device const uchar* W, device const half* X, device half* C,
    uint M, uint K, uint N,
    uint2 tgpig, ushort tiitg, ushort sgitg, threadgroup char* shmem) {
  ks_mm_gemm(W, X, C, M, K, N, tgpig, tiitg, sgitg, shmem, ks_dequant_q6k16, ksearch_load_q6k);
}
"#;

/// Dynamic shared memory for a CUDA `mul_mm` block: two stages, each a
/// `BM × BK` half weight tile plus a `BN × BK` half activation tile.
pub const CUDA_MUL_MM_BM: usize = 64;
pub const CUDA_MUL_MM_BN: usize = 64;
pub const CUDA_MUL_MM_BK: usize = 64;
pub const CUDA_MUL_MM_SMEM_BYTES: u64 =
    2 * (CUDA_MUL_MM_BM as u64 + CUDA_MUL_MM_BN as u64) * (CUDA_MUL_MM_BK as u64 + 8) * 2;

/// Threads per block for a CUDA `mul_mm` kernel.
pub const CUDA_MUL_MM_THREADS: u64 = 256;

/// Padded block size of a CUDA `mul_mm` kernel: low 32 lanes on `threadIdx.x`,
/// `nsg` groups on `threadIdx.y` (mirrors the Metal `(32, nsg)` threadgroup so
/// the `simd_shuffle_*` reductions in the `*_mul_mv_ext` helpers line up with
/// CUDA warp lanes).
pub const CUDA_MUL_MM_TW: u64 = 32;

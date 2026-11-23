//! Hand CUDA Q4_K × Q8_1 matvec (mmvq) and tiled GEMM (mmq-lite).
//!
//! Adapted from [llama.cpp](https://github.com/ggml-org/llama.cpp) b12283
//! (`ggml/src/ggml-cuda/{mmvq.cu,vecdotq.cuh,quantize.cu,mmq.cuh}`), MIT License.
//! See top-level `NOTICE`. Thesis A is suspended for these money ops.

use crate::ffi::CudaError;
use crate::launch::LaunchConfig;
use crate::{Buffer, Kernel, KernelArgs};

/// Compiled hand-mm entry points (one cubin TU).
#[derive(Clone, Copy)]
pub struct HandMmKernels {
    pub quant_q8_1: Kernel,
    pub mmvq_q4k: Kernel,
    pub mmq_q4k: Kernel,
    pub gelu_mul_f16: Kernel,
}

pub const HAND_MM_SRC: &str = r#"
// Adapted from llama.cpp b12283 ggml-cuda (MIT).
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256
#define QR4_K 2
#define QI4_K 32
#define QK8_1 32
#define QI8_1 8
#define VDR_Q4_K_Q8_1_MMVQ 2

struct block_q4_K {
  half2 dm;
  uint8_t scales[12];
  uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K");

struct block_q8_1 {
  half2 ds;
  int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1");

static __device__ __forceinline__ int ks_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const int8_t *aa = (const int8_t *)&a, *bb = (const int8_t *)&b;
  return c + aa[0]*bb[0] + aa[1]*bb[1] + aa[2]*bb[2] + aa[3]*bb[3];
#endif
}

static __device__ __forceinline__ float warp_sum_f(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, mask);
  return x;
}

static __device__ __forceinline__ float warp_max_f(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask));
  return x;
}

static __device__ __forceinline__ int warp_sum_i(int x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, mask);
  return x;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const half2 &dm4, const float *__restrict__ d8) {
  float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = ks_dp4a(v1i, u[2 * i + 1], ks_dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = ks_dp4a(0x01010101, u[2 * i + 1], ks_dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int kbx, const int iqs) {
  const block_q4_K *bq4_K = (const block_q4_K *)vbq + kbx;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];
  const uint16_t *scales = (const uint16_t *)bq4_K->scales;
  const int j = bq8_offset / 2;
  const int jm = j & 1;
  const uint32_t s0 = scales[jm + 0];
  const uint32_t s2 = scales[jm + 2];
  const uint32_t s4 = scales[jm + 4];
  const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
  uint16_t aux[2];
  aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) |
                      ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
  aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) |
                      ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

// F16 row -> Q8_1. One warp owns one Q8_1 block (32 elems).
extern "C" __global__ void ks_quantize_q8_1_f16(
    const half *__restrict__ x, block_q8_1 *__restrict__ y, unsigned ne) {
  const unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 >= ((ne + 31u) & ~31u)) return;
  const unsigned ib = i0 / QK8_1;
  const unsigned iqs = i0 % QK8_1;
  const float xi = (i0 < ne) ? __half2float(x[i0]) : 0.0f;
  float amax = fabsf(xi);
  amax = warp_max_f(amax);
  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
  const int sum = warp_sum_i((int)q);
  y[ib].qs[iqs] = q;
  if (iqs == 0) y[ib].ds = __floats2half2_rn(d, d * (float)sum);
}

// Q4_K × Q8_1 GEMV. block=(32,4), grid=(nrows). Adapted from llama.cpp mmvq.
extern "C" __global__ void ks_mmvq_q4k(
    const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
    half *__restrict__ dst, unsigned ncols, unsigned nrows) {
  constexpr int nwarps = 4;
  constexpr int warp_size = 32;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  constexpr int qi = QI4_K;
  constexpr int qk = QK_K;
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi; // 8

  const int tid = warp_size * threadIdx.y + threadIdx.x;
  const unsigned row = blockIdx.x;
  if (row >= nrows) return;

  const int blocks_per_row = (int)(ncols / qk);
  const int stride_row = blocks_per_row;
  float tmp = 0.0f;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));
    tmp += vec_dot_q4_K_q8_1(vx, vy + kby, row * stride_row + kbx, kqs);
  }

  __shared__ float tmp_shared[nwarps - 1][warp_size];
  if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;

#pragma unroll
  for (int l = 0; l < nwarps - 1; ++l) tmp += tmp_shared[l][threadIdx.x];
  tmp = warp_sum_f(tmp);
  if (threadIdx.x == 0) dst[row] = __float2half(tmp);
}

// Prefill MMQ-lite: one CUDA block per output row; K is streamed once while
// token groups of 8 reuse the same weight loads (L1/regs hot). Much less HBM
// traffic than a (nrows × nbatch/8) grid. Grid (nrows), block (32,4).
#define KS_MMQ_BN 8

extern "C" __global__ void ks_mmq_q4k(
    const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
    half *__restrict__ dst, unsigned ncols, unsigned nrows, unsigned nbatch) {
  constexpr int nwarps = 4;
  constexpr int warp_size = 32;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  constexpr int qi = QI4_K;
  constexpr int qk = QK_K;
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
  constexpr int bn = KS_MMQ_BN;

  const unsigned row = blockIdx.x;
  if (row >= nrows) return;

  const int tid = warp_size * threadIdx.y + threadIdx.x;
  const int blocks_per_row = (int)(ncols / qk);
  const int y_row_stride = (int)(ncols / QK8_1);
  const int row_kb = (int)row * blocks_per_row;

  for (unsigned col0 = 0; col0 < nbatch; col0 += bn) {
    float tmp[bn];
#pragma unroll
    for (int j = 0; j < bn; ++j) tmp[j] = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
      for (int j = 0; j < bn; ++j) {
        const unsigned tok = col0 + (unsigned)j;
        if (tok < nbatch) {
          tmp[j] += vec_dot_q4_K_q8_1(
              vx, vy + (size_t)tok * y_row_stride + kby, row_kb + kbx, kqs);
        }
      }
    }

    __shared__ float tmp_shared[nwarps - 1][bn][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
      for (int j = 0; j < bn; ++j)
        tmp_shared[threadIdx.y - 1][j][threadIdx.x] = tmp[j];
    }
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int j = 0; j < bn; ++j) {
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l)
          tmp[j] += tmp_shared[l][j][threadIdx.x];
        tmp[j] = warp_sum_f(tmp[j]);
        const unsigned tok = col0 + (unsigned)j;
        if (threadIdx.x == 0 && tok < nbatch)
          dst[(size_t)tok * nrows + row] = __float2half(tmp[j]);
      }
    }
    __syncthreads();
  }
}

// y = gelu(gate) * up, all F16, length n.
extern "C" __global__ void ks_gelu_mul_f16(
    const half *__restrict__ gate, const half *__restrict__ up,
    half *__restrict__ y, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = __half2float(gate[i]);
  // tanh-approx gelu
  const float c = 0.7978845608028654f; // sqrt(2/pi)
  float x3 = x * x * x;
  float t = tanhf(c * (x + 0.044715f * x3));
  float g = 0.5f * x * (1.0f + t);
  y[i] = __float2half(g * __half2float(up[i]));
}
"#;

/// Bytes for a Q8_1 row covering `cols` F16 activations (`cols` multiple of 32).
pub fn q8_1_bytes(cols: usize) -> usize {
    (cols / 32) * 36
}

impl super::CudaContext {
    pub(crate) fn hand_mm_kernels(&mut self) -> Result<HandMmKernels, CudaError> {
        if let Some(k) = self.hand_mm {
            return Ok(k);
        }
        let src = HAND_MM_SRC;
        let k = HandMmKernels {
            quant_q8_1: self.load_source("ks_quantize_q8_1_f16", src)?,
            mmvq_q4k: self.load_source("ks_mmvq_q4k", src)?,
            mmq_q4k: self.load_source("ks_mmq_q4k", src)?,
            gelu_mul_f16: self.load_source("ks_gelu_mul_f16", src)?,
        };
        self.hand_mm = Some(k);
        Ok(k)
    }

    /// F16 `[ne]` → Q8_1 packed `[ceil(ne/32)]` blocks.
    pub fn quantize_q8_1_f16(&mut self, x: &Buffer, y: &Buffer, ne: u32) -> Result<(), CudaError> {
        let k = self.hand_mm_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(x.as_ptr()).ptr(y.as_ptr()).u32(ne);
        let threads = 256u32;
        let padded = (ne + 31) & !31;
        let grid = (padded + threads - 1) / threads;
        self.launch_with(
            k.quant_q8_1,
            LaunchConfig::new((grid.max(1), 1), (threads, 1), 0),
            &mut args,
        )
    }

    /// Q4_K `[nrows, ncols]` × Q8_1 `[ncols]` → F16 `[nrows]`.
    pub fn mmvq_q4k(
        &mut self,
        w: &Buffer,
        x_q8: &Buffer,
        y: &Buffer,
        nrows: u32,
        ncols: u32,
    ) -> Result<(), CudaError> {
        let k = self.hand_mm_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(w.as_ptr())
            .ptr(x_q8.as_ptr())
            .ptr(y.as_ptr())
            .u32(ncols)
            .u32(nrows);
        // block (32, 4) = 4 warps
        self.launch_with(
            k.mmvq_q4k,
            LaunchConfig::new((nrows.max(1), 1), (32, 4), 0),
            &mut args,
        )
    }

    /// Prefill: Q4_K `[nrows,ncols]` × Q8_1 `[nbatch, ncols/32]` → F16 `[nbatch, nrows]`.
    pub fn mmq_q4k(
        &mut self,
        w: &Buffer,
        x_q8: &Buffer,
        y: &Buffer,
        nrows: u32,
        ncols: u32,
        nbatch: u32,
    ) -> Result<(), CudaError> {
        let k = self.hand_mm_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(w.as_ptr())
            .ptr(x_q8.as_ptr())
            .ptr(y.as_ptr())
            .u32(ncols)
            .u32(nrows)
            .u32(nbatch);
        self.launch_with(
            k.mmq_q4k,
            LaunchConfig::new((nrows.max(1), 1), (32, 4), 0),
            &mut args,
        )
    }

    pub fn gelu_mul_f16(
        &mut self,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
        n: u32,
    ) -> Result<(), CudaError> {
        let k = self.hand_mm_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(gate.as_ptr())
            .ptr(up.as_ptr())
            .ptr(y.as_ptr())
            .u32(n);
        let threads = 256u32;
        let grid = (n + threads - 1) / threads;
        self.launch_with(
            k.gelu_mul_f16,
            LaunchConfig::new((grid.max(1), 1), (threads, 1), 0),
            &mut args,
        )
    }
}

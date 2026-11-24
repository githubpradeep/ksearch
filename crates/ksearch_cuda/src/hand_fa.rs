//! Hand CUDA Flash Attention — llama.cpp `fattn-vec` port (Q4_0×Q4_0).
//!
//! Adapted from llama.cpp `fattn-vec.cuh` / `fattn-common.cuh` (MIT; see `NOTICE`).
//! Thesis A suspended (same class as [`crate::hand_mm`]).
//!
//! Matches llama.cpp dispatch on A10 for quantized KV:
//! - D≤256 → fattn-vec (this file)
//! - D=512 → fattn-mma-f16 ([`crate::hand_fa_mma`]; no vec instance for 512)
//!
//! Launch shape matches llama decode (ncols=1): **one block per Q head**,
//! `grid = n_tok * n_q * n_parts`, GQA via `kv_head = q_head / gqa_ratio`.

use crate::ffi::CudaError;
use crate::launch::LaunchConfig;
use crate::{Buffer, Kernel, KernelArgs};

#[derive(Clone, Copy)]
pub struct HandFaKernels {
    pub flash_q40_vec_d256: Kernel,
    pub flash_q40_vec: Kernel,
    pub flash_q40_combine: Kernel,
}

/// llama.cpp A10 quantized-KV policy: MMA for D=512 (no Q4 vec); vec for ≤256.
/// `KSEARCH_FLASH_ATTN_MMA=1` forces MMA on 256|512; `=0` forces vec.
pub fn prefer_fa_mma(hd: u32) -> bool {
    match std::env::var("KSEARCH_FLASH_ATTN_MMA").ok().as_deref() {
        Some("0") => false,
        Some("1") => hd == 256 || hd == 512,
        _ => hd == 512,
    }
}

pub const HAND_FA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_0 32
#define QK8_1 32
#define QI4_0 4
#define QI8_1 8
#define WARP_SIZE 32
#define NWARPS 4
#define NTHREADS (WARP_SIZE * NWARPS)
#define V_ROWS 4

struct alignas(2) block_q4_0 {
  half d;
  uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0");

static __device__ __forceinline__ int ks_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const int8_t *aa = (const int8_t *)&a, *bb = (const int8_t *)&b;
  return c + aa[0] * bb[0] + aa[1] * bb[1] + aa[2] * bb[2] + aa[3] * bb[3];
#endif
}

static __device__ __forceinline__ float warp_sum_f(float x) {
#pragma unroll
  for (int m = WARP_SIZE / 2; m > 0; m >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, m);
  return x;
}

template <int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float *__restrict__ x, int *__restrict__ yq32, float2 *__restrict__ yds) {
  float vals[4];
#pragma unroll
  for (int l = 0; l < 4; ++l)
    vals[l] = (threadIdx.x < ni) ? x[4 * threadIdx.x + l] : 0.0f;

  float amax = fabsf(vals[0]);
  float sum = vals[0];
#pragma unroll
  for (int l = 1; l < 4; ++l) {
    amax = fmaxf(amax, fabsf(vals[l]));
    sum += vals[l];
  }
#pragma unroll
  for (int mask = QI8_1 / 2; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }

  const float d = amax / 127.0f;
  int q32 = 0;
  int8_t *q8 = (int8_t *)&q32;
  if (d != 0.0f) {
#pragma unroll
    for (int l = 0; l < 4; ++l)
      q8[l] = (int8_t)roundf(vals[l] / d);
  }
  if (threadIdx.x < ni)
    yq32[threadIdx.x] = q32;
  if ((threadIdx.x % QI8_1) == 0 && threadIdx.x < ni)
    yds[threadIdx.x / QI8_1] = make_float2(d, sum);
}

template <int D>
static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const block_q4_0 *__restrict__ K_row, const int *__restrict__ Q_q8,
    const float2 *__restrict__ Q_ds) {
  constexpr int n_qpack = D / (4 * WARP_SIZE);
  float sum = 0.0f;
#pragma unroll
  for (int i0 = 0; i0 < n_qpack; ++i0) {
    const int k = i0 * WARP_SIZE + threadIdx.x;
    const int ib = k / QI8_1;
    const int iqs4 = k % QI4_0;
    const int shift = k & (QI8_1 / 2);
    int v;
    const uint8_t *p = K_row[ib].qs + sizeof(int) * iqs4;
    v = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
    v = (v >> shift) & 0x0F0F0F0F;
    const int sumi = ks_dp4a(v, Q_q8[i0], 0);
    const float2 ds = Q_ds[i0];
    sum += __half2float(K_row[ib].d) *
           (sumi * ds.x - (8.0f / (float)QI8_1) * ds.y);
  }
  return sum;
}

static __device__ __forceinline__ void dequant_v_q4_0(
    const block_q4_0 *__restrict__ row, int i0, float *dst) {
  const int ib = i0 / QK4_0;
  const int iqs = i0 % (QK4_0 / 2);
  const int shift = (i0 % QK4_0) / (QK4_0 / 2);
  int q;
  const uint8_t *p = row[ib].qs + iqs;
  q = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
  q >>= 4 * shift;
  q &= 0x0F0F0F0F;
  const float d = __half2float(row[ib].d);
  const int8_t *q8 = (const int8_t *)&q;
#pragma unroll
  for (int l = 0; l < V_ROWS; ++l)
    dst[l] = d * (float)(q8[l] - 8);
}

// One Q head per block (llama fattn-vec ncols=1).
template <int D>
static __device__ void flash_attn_vec_impl(
    const half *__restrict__ Q, const uint8_t *__restrict__ K,
    const uint8_t *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned max_t, unsigned n_parts) {
  constexpr int n_qpack = D / (4 * WARP_SIZE);
  constexpr int ne = D / WARP_SIZE;
  constexpr int n_chunks = D / (WARP_SIZE * V_ROWS);

  const int part = (int)(blockIdx.x % n_parts);
  const int q_head = (int)((blockIdx.x / n_parts) % n_q);
  const int tok = (int)(blockIdx.x / (n_parts * n_q));
  if (tok >= (int)n_tok) return;

  const int gqa = (int)(n_q / max(n_kv, 1u));
  const int kv_head = q_head / max(gqa, 1);
  const int row_bytes = (D / QK4_0) * (int)sizeof(block_q4_0);
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

  int tlen, start;
  if (n_tok == 1) {
    tlen = (int)meta[0];
    start = (int)meta[1];
  } else {
    const int abs_t = (int)meta[0] + tok;
    const int win = (meta[2] > 0.0f) ? (int)meta[2] : (int)max_t;
    tlen = min(abs_t, win);
    start = abs_t - tlen;
  }

  __shared__ float q_f32[D];
  __shared__ int q_i32_sh[D / 4];
  __shared__ float2 q_ds_sh[D / QK8_1];
  __shared__ float KQ[NTHREADS];
  __shared__ float tile_max;
  __shared__ float l_warp[NWARPS];
  __shared__ float l_bcast;
  __shared__ float a_bcast;
  __shared__ float o_red[NWARPS][WARP_SIZE][D / WARP_SIZE];

  int Q_i32[n_qpack];
  float2 Q_ds[n_qpack];
  float o_reg[ne];
  float m_i = -1.0e30f;
  float l_i = 0.0f;

#pragma unroll
  for (int i = 0; i < ne; ++i)
    o_reg[i] = 0.0f;

  {
    const int q_off = (tok * (int)n_q + q_head) * D;
    for (int i = tid; i < D; i += NTHREADS)
      q_f32[i] = __half2float(Q[q_off + i]);
    __syncthreads();

    if (threadIdx.y == 0) {
      const int n_ints = D / 4;
      for (int i0 = 0; i0 < n_ints; i0 += WARP_SIZE) {
        quantize_q8_1_to_shared<WARP_SIZE>(q_f32 + i0 * 4, q_i32_sh + i0,
                                           q_ds_sh + i0 / QI8_1);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i0 = 0; i0 < n_qpack; ++i0) {
      const int i = i0 * WARP_SIZE + threadIdx.x;
      Q_i32[i0] = q_i32_sh[i];
      Q_ds[i0] = q_ds_sh[i / QI8_1];
    }
  }

  if (tlen <= 0) {
    if (n_parts == 1 && threadIdx.y == 0) {
      const int o_off = (tok * (int)n_q + q_head) * D;
#pragma unroll
      for (int c = 0; c < n_chunks; ++c)
#pragma unroll
        for (int j = 0; j < V_ROWS; ++j)
          O[o_off + c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS + j] =
              __float2half(0.0f);
    }
    return;
  }

  for (int t0 = part * NTHREADS; t0 < tlen; t0 += (int)n_parts * NTHREADS) {
#pragma unroll
    for (int i_KQ = 0; i_KQ < WARP_SIZE; ++i_KQ) {
      const int kv_idx = t0 + threadIdx.y * WARP_SIZE + i_KQ;
      const bool valid = kv_idx < tlen;
      float score = -1.0e30f;
      if (valid) {
        const int row = (start + kv_idx) * (int)n_kv + kv_head;
        const block_q4_0 *k_row =
            (const block_q4_0 *)(K + (size_t)row * (size_t)row_bytes);
        score = vec_dot_q4_0_q8_1<D>(k_row, Q_i32, Q_ds);
        score = warp_sum_f(score);
      }
      if (!valid) score = -1.0e30f;
      if (threadIdx.x == i_KQ)
        KQ[threadIdx.y * WARP_SIZE + i_KQ] = score;
    }
    __syncthreads();

    if (tid == 0) {
      float tm = m_i;
#pragma unroll
      for (int s = 0; s < NTHREADS; ++s)
        tm = fmaxf(tm, KQ[s]);
      tile_max = tm;
    }
    __syncthreads();

    {
      const float m2 = tile_max;
      const float alpha = __expf(m_i - m2);
      if (tid == 0) a_bcast = alpha;
      m_i = m2;
      float s = (t0 + tid < tlen) ? __expf(KQ[tid] - m2) : 0.0f;
      KQ[tid] = s;
      float l_add = warp_sum_f(s);
      if (threadIdx.x == 0)
        l_warp[threadIdx.y] = l_add;
#pragma unroll
      for (int i = 0; i < ne; ++i)
        o_reg[i] *= alpha;
    }
    __syncthreads();

    if (tid == 0) {
      float ls = 0.0f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w)
        ls += l_warp[w];
      l_i = l_i * a_bcast + ls;
      l_bcast = l_i;
    }
    __syncthreads();
    l_i = l_bcast;

    float o_tile[ne];
#pragma unroll
    for (int i = 0; i < ne; ++i)
      o_tile[i] = 0.0f;

#pragma unroll
    for (int i_KQ = 0; i_KQ < WARP_SIZE; ++i_KQ) {
      const int slot = threadIdx.y * WARP_SIZE + i_KQ;
      const int kv_idx = t0 + slot;
      if (kv_idx >= tlen) continue;
      const int row = (start + kv_idx) * (int)n_kv + kv_head;
      const block_q4_0 *v_row =
          (const block_q4_0 *)(V + (size_t)row * (size_t)row_bytes);
      const float w = KQ[slot];
#pragma unroll
      for (int c = 0; c < n_chunks; ++c) {
        float vv[V_ROWS];
        dequant_v_q4_0(v_row, c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS,
                       vv);
#pragma unroll
        for (int j = 0; j < V_ROWS; ++j)
          o_tile[c * V_ROWS + j] += w * vv[j];
      }
    }

#pragma unroll
    for (int i = 0; i < ne; ++i)
      o_red[threadIdx.y][threadIdx.x][i] = o_tile[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ne; ++i) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w)
        sum += o_red[w][threadIdx.x][i];
      o_reg[i] += sum;
    }
    __syncthreads();
  }

  const int stride = D + 2;
  if (n_parts == 1) {
    if (threadIdx.y == 0) {
      const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
      const int o_off = (tok * (int)n_q + q_head) * D;
#pragma unroll
      for (int c = 0; c < n_chunks; ++c)
#pragma unroll
        for (int j = 0; j < V_ROWS; ++j)
          O[o_off + c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS + j] =
              __float2half(o_reg[c * V_ROWS + j] * inv);
    }
  } else if (threadIdx.y == 0) {
    float *dst =
        tmp + ((size_t)(tok * (int)n_q + q_head) * n_parts + (size_t)part) *
                  (size_t)stride;
#pragma unroll
    for (int c = 0; c < n_chunks; ++c)
#pragma unroll
      for (int j = 0; j < V_ROWS; ++j)
        dst[c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS + j] =
            o_reg[c * V_ROWS + j];
    if (threadIdx.x == 0) {
      dst[D] = m_i;
      dst[D + 1] = l_i;
    }
  }
}

extern "C" __global__ void ks_flash_attn_q40_vec_d256(
    const half *__restrict__ Q, const uint8_t *__restrict__ K,
    const uint8_t *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned max_t, unsigned n_parts) {
  flash_attn_vec_impl<256>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                           n_parts);
}

extern "C" __global__ void ks_flash_attn_q40_vec(
    const half *__restrict__ Q, const uint8_t *__restrict__ K,
    const uint8_t *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned hd, unsigned max_t,
    unsigned n_parts) {
  if (hd == 256) {
    flash_attn_vec_impl<256>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                             n_parts);
  } else if (hd == 128) {
    flash_attn_vec_impl<128>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                             n_parts);
  } else if (hd == 384) {
    flash_attn_vec_impl<384>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                             n_parts);
  } else if (hd == 512) {
    flash_attn_vec_impl<512>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                             n_parts);
  }
}

extern "C" __global__ void ks_flash_attn_q40_combine(
    const float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned hd, unsigned n_parts) {
  const int idx = (int)blockIdx.x;
  const int n = (int)(n_tok * n_q);
  if (idx >= n) return;
  const int stride = (int)hd + 2;
  float m = -1.0e30f;
  float l = 0.0f;
  for (int p = 0; p < (int)n_parts; ++p) {
    const float *src = tmp + ((size_t)idx * n_parts + (size_t)p) * (size_t)stride;
    const float mp = src[hd];
    const float lp = src[hd + 1];
    const float m2 = fmaxf(m, mp);
    l = __expf(m - m2) * l + __expf(mp - m2) * lp;
    m = m2;
  }
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  half *out = O + (size_t)idx * hd;
  for (int i = threadIdx.x; i < (int)hd; i += WARP_SIZE) {
    float acc = 0.0f;
    for (int p = 0; p < (int)n_parts; ++p) {
      const float *src =
          tmp + ((size_t)idx * n_parts + (size_t)p) * (size_t)stride;
      acc += src[i] * __expf(src[hd] - m);
    }
    out[i] = __float2half(acc * inv);
  }
}
"#;

impl super::CudaContext {
    pub(crate) fn hand_fa_kernels(&mut self) -> Result<HandFaKernels, CudaError> {
        if let Some(k) = self.hand_fa {
            return Ok(k);
        }
        let k = HandFaKernels {
            flash_q40_vec_d256: self.load_source("ks_flash_attn_q40_vec_d256", HAND_FA_SRC)?,
            flash_q40_vec: self.load_source("ks_flash_attn_q40_vec", HAND_FA_SRC)?,
            flash_q40_combine: self.load_source("ks_flash_attn_q40_combine", HAND_FA_SRC)?,
        };
        self.hand_fa = Some(k);
        Ok(k)
    }

    /// FlashAttention over Q4_0 KV.
    ///
    /// llama.cpp A10 quantized-KV policy: **vec for D≤256**, **MMA-F16 for D=512**.
    pub fn flash_attn_q40(
        &mut self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        tmp: &Buffer,
        out: &Buffer,
        n_tok: u32,
        n_q: u32,
        n_kv: u32,
        hd: u32,
        max_t: u32,
        n_parts: u32,
        kv_len: u32,
    ) -> Result<(), CudaError> {
        if prefer_fa_mma(hd) {
            match self.flash_attn_q40_mma(
                q, k, v, meta, tmp, out, n_tok, n_q, n_kv, hd, max_t, n_parts, kv_len,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!("[fa] MMA unavailable ({e}); falling back to vec");
                }
            }
        }
        if hd % 128 != 0 || hd > 512 || hd < 128 {
            return Err(CudaError::Other(format!(
                "flash_attn_q40: hd={hd} unsupported (need 128|256|384|512)"
            )));
        }
        if n_kv == 0 || n_q % n_kv != 0 {
            return Err(CudaError::Other(format!(
                "flash_attn_q40: n_q={n_q} n_kv={n_kv} unsupported"
            )));
        }
        let n_parts = n_parts.max(1);
        let kern = self.hand_fa_kernels()?;
        let grid = n_tok.saturating_mul(n_q).saturating_mul(n_parts).max(1);
        if hd == 256 {
            let mut args = KernelArgs::new();
            args.ptr(q.as_ptr())
                .ptr(k.as_ptr())
                .ptr(v.as_ptr())
                .ptr(meta.as_ptr())
                .ptr(tmp.as_ptr())
                .ptr(out.as_ptr())
                .u32(n_tok)
                .u32(n_q)
                .u32(n_kv)
                .u32(max_t)
                .u32(n_parts);
            self.launch_with(
                kern.flash_q40_vec_d256,
                LaunchConfig::new((grid, 1), (32, 4), 0),
                &mut args,
            )?;
        } else {
            let mut args = KernelArgs::new();
            args.ptr(q.as_ptr())
                .ptr(k.as_ptr())
                .ptr(v.as_ptr())
                .ptr(meta.as_ptr())
                .ptr(tmp.as_ptr())
                .ptr(out.as_ptr())
                .u32(n_tok)
                .u32(n_q)
                .u32(n_kv)
                .u32(hd)
                .u32(max_t)
                .u32(n_parts);
            self.launch_with(
                kern.flash_q40_vec,
                LaunchConfig::new((grid, 1), (32, 4), 0),
                &mut args,
            )?;
        }
        if n_parts > 1 {
            let mut cargs = KernelArgs::new();
            cargs
                .ptr(tmp.as_ptr())
                .ptr(out.as_ptr())
                .u32(n_tok)
                .u32(n_q)
                .u32(hd)
                .u32(n_parts);
            let cgrid = n_tok.saturating_mul(n_q).max(1);
            self.launch_with(
                kern.flash_q40_combine,
                LaunchConfig::new((cgrid, 1), (32, 1), 0),
                &mut cargs,
            )?;
        }
        Ok(())
    }
}

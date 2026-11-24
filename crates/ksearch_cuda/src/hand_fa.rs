//! Hand CUDA Flash Attention over Q4_0 KV — llama.cpp fattn-vec shaped.
//!
//! Adapted from [llama.cpp](https://github.com/ggml-org/llama.cpp) `fattn-vec.cuh` /
//! `fattn-common.cuh` (MIT; see `NOTICE`). Thesis A suspended (same class as
//! [`crate::hand_mm`]).
//!
//! - block = (32, 4) → 128 threads / 128 KV per tile
//! - Q → Q8_1 + dp4a against Q4_0 K
//! - V: vectorized Q4_0 dequant (4-wide), warp-local KV slice + cross-warp reduce
//! - `n_parts` parallel KV slices + online-softmax combine
//! - one block per `(tok, kv_head, part)`; GQA/MQA shares K/V in-block
//!
//! Math matches AST SDPA: score = Q·K (no 1/√d).

use crate::ffi::CudaError;
use crate::launch::LaunchConfig;
use crate::{Buffer, Kernel, KernelArgs};

#[derive(Clone, Copy)]
pub struct HandFaKernels {
    pub flash_q40_vec: Kernel,
    pub flash_q40_combine: Kernel,
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
#define MAX_GROUP 16
#define MAX_D 512
#define V_ROWS 4
#define MAX_NE (MAX_D / WARP_SIZE)           // floats per lane (D/32)
#define MAX_QPACK (MAX_D / (4 * WARP_SIZE))  // int32 packs per lane for K-dot

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

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const block_q4_0 *__restrict__ K_row, const int *__restrict__ Q_q8,
    const float2 *__restrict__ Q_ds, int n_qpack) {
  float sum = 0.0f;
#pragma unroll
  for (int i0 = 0; i0 < MAX_QPACK; ++i0) {
    if (i0 >= n_qpack) break;
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

// llama.cpp dequantize_V_q4_0: load `V_ROWS` consecutive elems starting at i0.
static __device__ __forceinline__ void dequant_v_q4_0(
    const block_q4_0 *__restrict__ row, int i0, float *dst) {
  const int ib = i0 / QK4_0;
  const int iqs = i0 % (QK4_0 / 2);
  const int shift = (i0 % QK4_0) / (QK4_0 / 2);
  int q;
  const uint8_t *p = row[ib].qs + iqs;
  // V_ROWS==4 → read 4 packed nibbles as one int (low or high nibble plane).
  q = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
  q >>= 4 * shift;
  q &= 0x0F0F0F0F;
  const float d = __half2float(row[ib].d);
  const int8_t *q8 = (const int8_t *)&q;
#pragma unroll
  for (int l = 0; l < V_ROWS; ++l)
    dst[l] = d * (float)(q8[l] - 8);
}

// Lane tx owns dims c*128 + tx*4 + 0..3  (c = chunk). o_reg index = c*4 + j.
static __device__ __forceinline__ int o_dim(int c, int j) {
  return c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS + j;
}

extern "C" __global__ void ks_flash_attn_q40_vec(
    const half *__restrict__ Q, const uint8_t *__restrict__ K,
    const uint8_t *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned hd, unsigned max_t,
    unsigned n_parts) {
  const int part = (int)(blockIdx.x % n_parts);
  const int kv_head = (int)((blockIdx.x / n_parts) % n_kv);
  const int tok = (int)(blockIdx.x / (n_parts * n_kv));
  if (tok >= (int)n_tok) return;

  const int n_groups = (int)(n_q / max(n_kv, 1u));
  const int q_head0 = kv_head * n_groups;
  const int row_bytes = (int)(hd / QK4_0) * (int)sizeof(block_q4_0);
  const int n_qpack = (int)hd / (4 * WARP_SIZE);
  const int ne = (int)hd / WARP_SIZE; // floats per lane
  const int n_chunks = (int)hd / (WARP_SIZE * V_ROWS);
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

  __shared__ float q_f32[MAX_D];
  __shared__ int q_i32_sh[MAX_D / 4];
  __shared__ float2 q_ds_sh[MAX_D / QK8_1];
  __shared__ float KQ[MAX_GROUP * NTHREADS];
  __shared__ float tile_max[MAX_GROUP];
  __shared__ float l_warp[MAX_GROUP][NWARPS];
  __shared__ float l_bcast[MAX_GROUP];
  __shared__ float a_bcast[MAX_GROUP];
  // Per-lane cross-warp O reduce: [warp][lane][ne]
  __shared__ float o_red[NWARPS][WARP_SIZE][MAX_NE];

  int Q_i32[MAX_GROUP][MAX_QPACK];
  float2 Q_ds[MAX_GROUP][MAX_QPACK];
  float o_reg[MAX_GROUP][MAX_NE];
  float m_i[MAX_GROUP];
  float l_i[MAX_GROUP];

#pragma unroll
  for (int g = 0; g < MAX_GROUP; ++g) {
    m_i[g] = -1.0e30f;
    l_i[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_NE; ++i)
      o_reg[g][i] = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_QPACK; ++i) {
      Q_i32[g][i] = 0;
      Q_ds[g][i] = make_float2(0.0f, 0.0f);
    }
  }

  for (int g = 0; g < n_groups; ++g) {
    const int q_off = (tok * (int)n_q + q_head0 + g) * (int)hd;
    for (int i = tid; i < (int)hd; i += NTHREADS)
      q_f32[i] = __half2float(Q[q_off + i]);
    __syncthreads();

    if (threadIdx.y == 0) {
      const int n_ints = (int)hd / 4;
      for (int i0 = 0; i0 < n_ints; i0 += WARP_SIZE) {
        quantize_q8_1_to_shared<WARP_SIZE>(q_f32 + i0 * 4, q_i32_sh + i0,
                                           q_ds_sh + i0 / QI8_1);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i0 = 0; i0 < MAX_QPACK; ++i0) {
      if (i0 >= n_qpack) break;
      const int i = i0 * WARP_SIZE + threadIdx.x;
      Q_i32[g][i0] = q_i32_sh[i];
      Q_ds[g][i0] = q_ds_sh[i / QI8_1];
    }
    __syncthreads();
  }

  if (tlen <= 0) {
    if (n_parts == 1 && threadIdx.y == 0) {
      for (int g = 0; g < n_groups; ++g) {
        const int o_off = (tok * (int)n_q + q_head0 + g) * (int)hd;
#pragma unroll
        for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
          if (c >= n_chunks) break;
#pragma unroll
          for (int j = 0; j < V_ROWS; ++j)
            O[o_off + o_dim(c, j)] = __float2half(0.0f);
        }
      }
    }
    return;
  }

  for (int t0 = part * NTHREADS; t0 < tlen; t0 += (int)n_parts * NTHREADS) {
    // --- KQ tile ---
#pragma unroll
    for (int i_KQ = 0; i_KQ < WARP_SIZE; ++i_KQ) {
      const int kv_idx = t0 + threadIdx.y * WARP_SIZE + i_KQ;
      const bool valid = kv_idx < tlen;
      const block_q4_0 *k_row = nullptr;
      if (valid) {
        const int row = (start + kv_idx) * (int)n_kv + kv_head;
        k_row = (const block_q4_0 *)(K + (size_t)row * (size_t)row_bytes);
      }
#pragma unroll
      for (int g = 0; g < MAX_GROUP; ++g) {
        if (g >= n_groups) break;
        float score = -1.0e30f;
        if (valid)
          score = vec_dot_q4_0_q8_1(k_row, Q_i32[g], Q_ds[g], n_qpack);
        score = warp_sum_f(score);
        if (!valid) score = -1.0e30f;
        if (threadIdx.x == i_KQ)
          KQ[g * NTHREADS + threadIdx.y * WARP_SIZE + i_KQ] = score;
      }
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < n_groups) {
      float tm = m_i[threadIdx.x];
#pragma unroll
      for (int s = 0; s < NTHREADS; ++s)
        tm = fmaxf(tm, KQ[threadIdx.x * NTHREADS + s]);
      tile_max[threadIdx.x] = tm;
    }
    __syncthreads();

#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      if (g >= n_groups) break;
      const float m2 = tile_max[g];
      const float alpha = __expf(m_i[g] - m2);
      a_bcast[g] = alpha;
      m_i[g] = m2;
      float s = (t0 + tid < tlen) ? __expf(KQ[g * NTHREADS + tid] - m2) : 0.0f;
      KQ[g * NTHREADS + tid] = s;
      float l_add = warp_sum_f(s);
      if (threadIdx.x == 0)
        l_warp[g][threadIdx.y] = l_add;
#pragma unroll
      for (int i = 0; i < MAX_NE; ++i) {
        if (i >= ne) break;
        o_reg[g][i] *= alpha;
      }
    }
    __syncthreads();

    if (tid < n_groups) {
      float ls = 0.0f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w)
        ls += l_warp[tid][w];
      l_i[tid] = l_i[tid] * a_bcast[tid] + ls;
      l_bcast[tid] = l_i[tid];
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      if (g >= n_groups) break;
      l_i[g] = l_bcast[g];
    }

    // --- V: each warp accumulates its 32 KV slots into o_tile, then reduce ---
    float o_tile[MAX_GROUP][MAX_NE];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g)
#pragma unroll
      for (int i = 0; i < MAX_NE; ++i)
        o_tile[g][i] = 0.0f;

#pragma unroll
    for (int i_KQ = 0; i_KQ < WARP_SIZE; ++i_KQ) {
      const int slot = threadIdx.y * WARP_SIZE + i_KQ;
      const int kv_idx = t0 + slot;
      if (kv_idx >= tlen) continue;
      const int row = (start + kv_idx) * (int)n_kv + kv_head;
      const block_q4_0 *v_row =
          (const block_q4_0 *)(V + (size_t)row * (size_t)row_bytes);

#pragma unroll
      for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
        if (c >= n_chunks) break;
        float vv[V_ROWS];
        dequant_v_q4_0(v_row, c * (WARP_SIZE * V_ROWS) + threadIdx.x * V_ROWS,
                       vv);
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) {
          if (g >= n_groups) break;
          const float w = KQ[g * NTHREADS + slot];
#pragma unroll
          for (int j = 0; j < V_ROWS; ++j)
            o_tile[g][c * V_ROWS + j] += w * vv[j];
        }
      }
    }

    // Cross-warp reduce per lane.
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      if (g >= n_groups) break;
#pragma unroll
      for (int i = 0; i < MAX_NE; ++i) {
        if (i >= ne) break;
        o_red[threadIdx.y][threadIdx.x][i] = o_tile[g][i];
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < MAX_NE; ++i) {
        if (i >= ne) break;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w)
          sum += o_red[w][threadIdx.x][i];
        o_reg[g][i] += sum;
      }
      __syncthreads();
    }
  }

  const int stride = (int)hd + 2;
  if (n_parts == 1) {
    if (threadIdx.y == 0) {
#pragma unroll
      for (int g = 0; g < MAX_GROUP; ++g) {
        if (g >= n_groups) break;
        const float inv = (l_i[g] > 0.0f) ? (1.0f / l_i[g]) : 0.0f;
        const int o_off = (tok * (int)n_q + q_head0 + g) * (int)hd;
#pragma unroll
        for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
          if (c >= n_chunks) break;
#pragma unroll
          for (int j = 0; j < V_ROWS; ++j)
            O[o_off + o_dim(c, j)] =
                __float2half(o_reg[g][c * V_ROWS + j] * inv);
        }
      }
    }
  } else if (threadIdx.y == 0) {
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      if (g >= n_groups) break;
      float *dst = tmp + ((size_t)(tok * (int)n_q + q_head0 + g) * n_parts +
                          (size_t)part) *
                             (size_t)stride;
#pragma unroll
      for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
        if (c >= n_chunks) break;
#pragma unroll
        for (int j = 0; j < V_ROWS; ++j)
          dst[o_dim(c, j)] = o_reg[g][c * V_ROWS + j];
      }
      if (threadIdx.x == 0) {
        dst[hd] = m_i[g];
        dst[hd + 1] = l_i[g];
      }
    }
  }
}

extern "C" __global__ void ks_flash_attn_q40_combine(
    const float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned hd, unsigned n_parts) {
  const int q_head = (int)(blockIdx.x % n_q);
  const int tok = (int)(blockIdx.x / n_q);
  if (tok >= (int)n_tok) return;
  const int tid = threadIdx.x;
  const int ne = (int)hd / WARP_SIZE;
  const int n_chunks = (int)hd / (WARP_SIZE * V_ROWS);
  const int stride = (int)hd + 2;

  float m = -1.0e30f;
  float l = 0.0f;
  float o[MAX_NE];
#pragma unroll
  for (int i = 0; i < MAX_NE; ++i)
    o[i] = 0.0f;

  for (int p = 0; p < (int)n_parts; ++p) {
    const float *src =
        tmp + ((size_t)(tok * (int)n_q + q_head) * n_parts + (size_t)p) *
                  (size_t)stride;
    const float mp = src[hd];
    const float lp = src[hd + 1];
    const float m2 = fmaxf(m, mp);
    const float a = __expf(m - m2);
    const float b = __expf(mp - m2);
    l = l * a + lp * b;
    m = m2;
#pragma unroll
    for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
      if (c >= n_chunks) break;
#pragma unroll
      for (int j = 0; j < V_ROWS; ++j) {
        const int di = c * (WARP_SIZE * V_ROWS) + tid * V_ROWS + j;
        o[c * V_ROWS + j] = o[c * V_ROWS + j] * a + src[di] * b;
      }
    }
  }
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  const int o_off = (tok * (int)n_q + q_head) * (int)hd;
#pragma unroll
  for (int c = 0; c < MAX_D / (WARP_SIZE * V_ROWS); ++c) {
    if (c >= n_chunks) break;
#pragma unroll
    for (int j = 0; j < V_ROWS; ++j) {
      const int di = c * (WARP_SIZE * V_ROWS) + tid * V_ROWS + j;
      O[o_off + di] = __float2half(o[c * V_ROWS + j] * inv);
    }
  }
  (void)ne;
}
"#;

impl super::CudaContext {
    pub(crate) fn hand_fa_kernels(&mut self) -> Result<HandFaKernels, CudaError> {
        if let Some(k) = self.hand_fa {
            return Ok(k);
        }
        let k = HandFaKernels {
            flash_q40_vec: self.load_source("ks_flash_attn_q40_vec", HAND_FA_SRC)?,
            flash_q40_combine: self.load_source("ks_flash_attn_q40_combine", HAND_FA_SRC)?,
        };
        self.hand_fa = Some(k);
        Ok(k)
    }

    /// FlashAttention over Q4_0 KV. Prefers WMMA (D=256|512); else warp-vec.
    ///
    /// `tmp` when `n_parts > 1`: `[n_tok][n_q][n_parts][hd+2]` F32 (partial O, m, l).
    /// `KSEARCH_FLASH_ATTN_MMA=0` forces the vec path.
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
        // Prefer WMMA for D=256|512: Q4→F16 once then F16 TC (llama.cpp
        // fattn-mma-f16). KSEARCH_FLASH_ATTN_MMA=0 forces the vec path.
        let prefer_mma = (hd == 256 || hd == 512)
            && std::env::var("KSEARCH_FLASH_ATTN_MMA")
                .map(|v| v != "0")
                .unwrap_or(true);
        if prefer_mma {
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
        if n_kv == 0 || n_q % n_kv != 0 || (n_q / n_kv) > 16 {
            return Err(CudaError::Other(format!(
                "flash_attn_q40: n_q={n_q} n_kv={n_kv} unsupported"
            )));
        }
        let n_parts = n_parts.max(1);
        let kern = self.hand_fa_kernels()?;
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
        let grid = n_tok.saturating_mul(n_kv).saturating_mul(n_parts).max(1);
        self.launch_with(
            kern.flash_q40_vec,
            LaunchConfig::new((grid, 1), (32, 4), 0),
            &mut args,
        )?;
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

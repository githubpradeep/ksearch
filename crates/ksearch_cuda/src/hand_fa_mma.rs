//! Hand CUDA Flash Attention — WMMA / tensor-core path (llama.cpp fattn-mma-f16).
//!
//! llama.cpp does **not** run MMA on Q4_0 in-kernel. For MMA it:
//! 1. converts K/V → F16 once (`to_fp16`), then
//! 2. runs F16 tensor-core FlashAttention.
//!
//! We mirror that: a vectorized Q4_0→F16 pass over the used KV rows, then
//! WMMA Q·Kᵀ / P·V over F16 tiles. Thesis A suspended (same class as
//! [`crate::hand_fa`]). MIT NOTICE: patterns from llama.cpp fattn-mma-f16.

use crate::ffi::CudaError;
use crate::launch::LaunchConfig;
use crate::{Buffer, Kernel, KernelArgs};

#[derive(Clone, Copy)]
pub struct HandFaMmaKernels {
    pub dequant: Kernel,
    pub mma_d256: Kernel,
    pub mma_d512: Kernel,
    pub combine: Kernel,
}

pub const BR: u32 = 16;
/// KV tile rows. 32 keeps D=512 smem under A10's ~99 KiB/block cap.
pub const BC: u32 = 32;

pub fn mma_smem_bytes(hd: u32) -> u32 {
    // Q half[BR*D] + K half[BC*D] + S float[BR*BC] + O float[BR*D] + P half[BR*BC]
    let q = BR * hd * 2;
    let k = BC * hd * 2;
    let s = BR * BC * 4;
    let o = BR * hd * 4;
    let p = BR * BC * 2;
    q + k + s + o + p
}

pub const HAND_FA_MMA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cstdint>
#include <mma.h>
using namespace nvcuda;

#define QK4_0 32
#define WARP_SIZE 32
#define BR 16
#define BC 32

struct alignas(2) block_q4_0 {
  half d;
  uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0");

// ---- one-shot Q4_0 → F16 (llama.cpp to_fp16 before fattn-mma) ----------------

extern "C" __global__ void ks_q40_rows_to_f16(
    const uint8_t *__restrict__ src, half *__restrict__ dst, unsigned n_rows,
    unsigned hd) {
  const int row = (int)blockIdx.x;
  if (row >= (int)n_rows) return;
  const int row_blocks = (int)(hd / QK4_0);
  const block_q4_0 *brow =
      (const block_q4_0 *)(src + (size_t)row * (size_t)row_blocks *
                                    (size_t)sizeof(block_q4_0));
  half *out = dst + (size_t)row * (size_t)hd;
  // 32 threads × 1 Q4 block apiece per step; vectorize within a block.
  for (int ib = threadIdx.x; ib < row_blocks; ib += (int)blockDim.x) {
    const block_q4_0 &b = brow[ib];
    const float d = __half2float(b.d);
    half *o = out + ib * QK4_0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint8_t q = b.qs[j];
      o[j] = __float2half((float(int(q & 0x0f) - 8)) * d);
      o[j + 16] = __float2half((float(int(q >> 4) - 8)) * d);
    }
  }
}

// ---- F16 WMMA flash attn ----------------------------------------------------

static __device__ __forceinline__ float block_max(float v, int tid, int nthreads,
                                                   float *red) {
#pragma unroll
  for (int m = WARP_SIZE / 2; m > 0; m >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  if ((tid & 31) == 0)
    red[tid / 32] = v;
  __syncthreads();
  if (tid < 32) {
    float x = (tid < (nthreads / 32)) ? red[tid] : -1.0e30f;
#pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1)
      x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, m));
    if (tid == 0)
      red[0] = x;
  }
  __syncthreads();
  return red[0];
}

static __device__ __forceinline__ float block_sum(float v, int tid, int nthreads,
                                                   float *red) {
#pragma unroll
  for (int m = WARP_SIZE / 2; m > 0; m >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, m);
  if ((tid & 31) == 0)
    red[tid / 32] = v;
  __syncthreads();
  if (tid < 32) {
    float x = (tid < (nthreads / 32)) ? red[tid] : 0.0f;
#pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1)
      x += __shfl_xor_sync(0xffffffff, x, m);
    if (tid == 0)
      red[0] = x;
  }
  __syncthreads();
  return red[0];
}

template <int D>
static __device__ __forceinline__ void load_kv_tile_f16(
    const half *__restrict__ src, half *__restrict__ dst, int row0, int n_kv,
    int kv_head, int tile_n, int tid, int nthreads) {
  for (int t = 0; t < BC; ++t) {
    half *row_dst = dst + t * D;
    if (t < tile_n) {
      const half *row_src =
          src + ((size_t)(row0 + t) * (size_t)n_kv + (size_t)kv_head) * (size_t)D;
      // vectorized 8-half (16B) moves when D%8==0 (always here).
      for (int d = tid * 8; d < D; d += nthreads * 8) {
        *((uint4 *)(row_dst + d)) = *((const uint4 *)(row_src + d));
      }
    } else {
      for (int d = tid; d < D; d += nthreads)
        row_dst[d] = __float2half(0.0f);
    }
  }
}

template <int D>
static __device__ void flash_attn_mma_f16_impl(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned max_t, unsigned n_parts) {
  const int part = (int)(blockIdx.x % n_parts);
  const int kv_head = (int)((blockIdx.x / n_parts) % n_kv);
  const int tok = (int)(blockIdx.x / (n_parts * n_kv));
  if (tok >= (int)n_tok) return;

  const int n_groups = (int)(n_q / max(n_kv, 1u));
  const int q_head0 = kv_head * n_groups;
  const int nrows = n_groups;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;

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

  extern __shared__ char smem_raw[];
  half *Q_s = (half *)smem_raw;
  half *K_s = Q_s + BR * D;
  float *S_s = (float *)(K_s + BC * D);
  float *O_s = S_s + BR * BC;
  half *P_h = (half *)(O_s + BR * D);
  __shared__ float red[32];

  for (int r = 0; r < BR; ++r) {
    half *dst = Q_s + r * D;
    if (r < nrows) {
      const half *src = Q + (size_t)(tok * (int)n_q + q_head0 + r) * (size_t)D;
      for (int d = tid * 8; d < D; d += nthreads * 8)
        *((uint4 *)(dst + d)) = *((const uint4 *)(src + d));
    } else {
      for (int d = tid; d < D; d += nthreads)
        dst[d] = __float2half(0.0f);
    }
  }
  __syncthreads();

  float m_i[BR];
  float l_i[BR];
#pragma unroll
  for (int r = 0; r < BR; ++r) {
    m_i[r] = -1.0e30f;
    l_i[r] = 0.0f;
  }
  for (int i = tid; i < BR * D; i += nthreads)
    O_s[i] = 0.0f;
  __syncthreads();

  if (tlen <= 0) {
    if (n_parts == 1) {
      for (int r = 0; r < nrows; ++r) {
        const int o_off = (tok * (int)n_q + q_head0 + r) * D;
        for (int d = tid; d < D; d += nthreads)
          O[o_off + d] = __float2half(0.0f);
      }
    }
    return;
  }

  for (int t0 = part * BC; t0 < tlen; t0 += (int)n_parts * BC) {
    const int tile_n = min(BC, tlen - t0);

    load_kv_tile_f16<D>(K, K_s, start + t0, (int)n_kv, kv_head, tile_n, tid,
                        nthreads);
    __syncthreads();

    for (int i = tid; i < BR * BC; i += nthreads)
      S_s[i] = 0.0f;
    __syncthreads();

    {
      const int warp = tid / WARP_SIZE;
      const int nwarps = nthreads / WARP_SIZE;
      const int tiles_n = BC / 16;
      const int ntiles = (BR / 16) * tiles_n;
      for (int tile = warp; tile < ntiles; tile += nwarps) {
        const int tm = tile / tiles_n;
        const int tn = tile % tiles_n;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
        wmma::fill_fragment(cfrag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> afrag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bfrag;
          wmma::load_matrix_sync(afrag, Q_s + tm * 16 * D + kk, D);
          wmma::load_matrix_sync(bfrag, K_s + tn * 16 * D + kk, D);
          wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
        }
        wmma::store_matrix_sync(S_s + tm * 16 * BC + tn * 16, cfrag, BC,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Softmax over valid KV; OOB → 0 before P·V (llama.cpp fattn-mma).
    for (int r = 0; r < nrows; ++r) {
      float row_max = m_i[r];
      for (int c = tid; c < tile_n; c += nthreads)
        row_max = fmaxf(row_max, S_s[r * BC + c]);
      const float m2 = block_max(row_max, tid, nthreads, red);
      const float alpha = __expf(m_i[r] - m2);
      m_i[r] = m2;

      float row_sum = 0.0f;
      for (int c = tid; c < BC; c += nthreads) {
        float e = 0.0f;
        if (c < tile_n) {
          e = __expf(S_s[r * BC + c] - m2);
          row_sum += e;
        }
        S_s[r * BC + c] = e;
      }
      const float rs = block_sum(row_sum, tid, nthreads, red);
      l_i[r] = l_i[r] * alpha + rs;
      for (int d = tid; d < D; d += nthreads)
        O_s[r * D + d] *= alpha;
      __syncthreads();
    }
    for (int i = tid; i < BR * BC; i += nthreads) {
      if ((i / BC) >= nrows)
        S_s[i] = 0.0f;
    }
    __syncthreads();

    load_kv_tile_f16<D>(V, K_s, start + t0, (int)n_kv, kv_head, tile_n, tid,
                        nthreads);
    __syncthreads();

    for (int i = tid; i < BR * BC; i += nthreads)
      P_h[i] = __float2half(S_s[i]);
    __syncthreads();

    {
      const int warp = tid / WARP_SIZE;
      const int nwarps = nthreads / WARP_SIZE;
      const int tiles_n = D / 16;
      const int ntiles = (BR / 16) * tiles_n;
      for (int tile = warp; tile < ntiles; tile += nwarps) {
        const int tm = tile / tiles_n;
        const int tn = tile % tiles_n;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
        wmma::load_matrix_sync(cfrag, O_s + tm * 16 * D + tn * 16, D,
                               wmma::mem_row_major);
#pragma unroll
        for (int kk = 0; kk < BC; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> afrag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bfrag;
          wmma::load_matrix_sync(afrag, P_h + tm * 16 * BC + kk, BC);
          wmma::load_matrix_sync(bfrag, K_s + kk * D + tn * 16, D);
          wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
        }
        wmma::store_matrix_sync(O_s + tm * 16 * D + tn * 16, cfrag, D,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  const int stride = D + 2;
  if (n_parts == 1) {
    for (int r = 0; r < nrows; ++r) {
      const float inv = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;
      const int o_off = (tok * (int)n_q + q_head0 + r) * D;
      for (int d = tid; d < D; d += nthreads)
        O[o_off + d] = __float2half(O_s[r * D + d] * inv);
    }
  } else {
    for (int r = 0; r < nrows; ++r) {
      float *dst =
          tmp + ((size_t)(tok * (int)n_q + q_head0 + r) * n_parts + (size_t)part) *
                    (size_t)stride;
      for (int d = tid; d < D; d += nthreads)
        dst[d] = O_s[r * D + d];
      if (tid == 0) {
        dst[D] = m_i[r];
        dst[D + 1] = l_i[r];
      }
    }
  }
}

extern "C" __global__ void ks_flash_attn_f16_mma_d256(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned max_t, unsigned n_parts) {
  flash_attn_mma_f16_impl<256>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                               n_parts);
}

extern "C" __global__ void ks_flash_attn_f16_mma_d512(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, const float *__restrict__ meta,
    float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned n_kv, unsigned max_t, unsigned n_parts) {
  flash_attn_mma_f16_impl<512>(Q, K, V, meta, tmp, O, n_tok, n_q, n_kv, max_t,
                               n_parts);
}

extern "C" __global__ void ks_flash_attn_q40_mma_combine(
    const float *__restrict__ tmp, half *__restrict__ O, unsigned n_tok,
    unsigned n_q, unsigned hd, unsigned n_parts) {
  const int q_head = (int)(blockIdx.x % n_q);
  const int tok = (int)(blockIdx.x / n_q);
  if (tok >= (int)n_tok) return;
  const int tid = threadIdx.x;
  const int stride = (int)hd + 2;
  const int o_off = (tok * (int)n_q + q_head) * (int)hd;

  for (int d0 = tid; d0 < (int)hd; d0 += WARP_SIZE) {
    float o = 0.0f;
    float mm = -1.0e30f;
    float ll = 0.0f;
    for (int p = 0; p < (int)n_parts; ++p) {
      const float *src =
          tmp + ((size_t)(tok * (int)n_q + q_head) * n_parts + (size_t)p) *
                    (size_t)stride;
      const float mp = src[hd];
      const float lp = src[hd + 1];
      const float m2 = fmaxf(mm, mp);
      const float a = __expf(mm - m2);
      const float b = __expf(mp - m2);
      o = o * a + src[d0] * b;
      ll = ll * a + lp * b;
      mm = m2;
    }
    const float inv = (ll > 0.0f) ? (1.0f / ll) : 0.0f;
    O[o_off + d0] = __float2half(o * inv);
  }
}
"#;

impl super::CudaContext {
    pub(crate) fn hand_fa_mma_kernels(&mut self) -> Result<HandFaMmaKernels, CudaError> {
        if let Some(k) = self.hand_fa_mma {
            return Ok(k);
        }
        let k = HandFaMmaKernels {
            dequant: self.load_source("ks_q40_rows_to_f16", HAND_FA_MMA_SRC)?,
            mma_d256: self.load_source("ks_flash_attn_f16_mma_d256", HAND_FA_MMA_SRC)?,
            mma_d512: self.load_source("ks_flash_attn_f16_mma_d512", HAND_FA_MMA_SRC)?,
            combine: self.load_source("ks_flash_attn_q40_mma_combine", HAND_FA_MMA_SRC)?,
        };
        self.hand_fa_mma = Some(k);
        Ok(k)
    }

    /// WMMA FlashAttention: Q4_0 KV → F16 once, then F16 tensor-core FA.
    /// `kv_len` = absolute filled KV positions (not the SWA window).
    pub fn flash_attn_q40_mma(
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
        if n_kv == 0 || n_q % n_kv != 0 || (n_q / n_kv) > 16 {
            return Err(CudaError::Other(format!(
                "flash_attn_q40_mma: n_q={n_q} n_kv={n_kv} unsupported"
            )));
        }
        let n_parts = n_parts.max(1);
        let kv_len = kv_len.min(max_t).max(1);
        let kern = self.hand_fa_mma_kernels()?;
        let entry = match hd {
            256 => kern.mma_d256,
            512 => kern.mma_d512,
            _ => {
                return Err(CudaError::Other(format!(
                    "flash_attn_q40_mma: hd={hd} need 256|512"
                )))
            }
        };

        let n_rows = kv_len.saturating_mul(n_kv);
        let elems = n_rows as usize * hd as usize;
        // Dequant into workspace, then MMA reads F16.
        {
            let (k_f16, v_f16) = {
                let bytes = elems.max(1) * 2;
                let need = match &self.fa_kv_f16 {
                    Some((_, _, cap)) => *cap < bytes,
                    None => true,
                };
                if need {
                    let kf = self.alloc(bytes)?;
                    let vf = self.alloc(bytes)?;
                    self.fa_kv_f16 = Some((kf, vf, bytes));
                }
                let (kf, vf, _) = self.fa_kv_f16.as_ref().unwrap();
                (kf.clone(), vf.clone())
            };
            let mut dargs = KernelArgs::new();
            dargs
                .ptr(k.as_ptr())
                .ptr(k_f16.as_ptr())
                .u32(n_rows)
                .u32(hd);
            self.launch_with(
                kern.dequant,
                LaunchConfig::new((n_rows.max(1), 1), (64, 1), 0),
                &mut dargs,
            )?;
            let mut dargs = KernelArgs::new();
            dargs
                .ptr(v.as_ptr())
                .ptr(v_f16.as_ptr())
                .u32(n_rows)
                .u32(hd);
            self.launch_with(
                kern.dequant,
                LaunchConfig::new((n_rows.max(1), 1), (64, 1), 0),
                &mut dargs,
            )?;

            let smem = mma_smem_bytes(hd);
            let mut args = KernelArgs::new();
            args.ptr(q.as_ptr())
                .ptr(k_f16.as_ptr())
                .ptr(v_f16.as_ptr())
                .ptr(meta.as_ptr())
                .ptr(tmp.as_ptr())
                .ptr(out.as_ptr())
                .u32(n_tok)
                .u32(n_q)
                .u32(n_kv)
                .u32(max_t)
                .u32(n_parts);
            let grid = n_tok.saturating_mul(n_kv).saturating_mul(n_parts).max(1);
            self.launch_with(
                entry,
                LaunchConfig::new((grid, 1), (32, 8), smem),
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
                kern.combine,
                LaunchConfig::new((cgrid, 1), (32, 1), 0),
                &mut cargs,
            )?;
        }
        Ok(())
    }
}

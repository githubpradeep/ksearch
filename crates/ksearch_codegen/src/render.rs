//! Generic MSL renderer: Kernel IR AST → Metal. No named hand kernels.

use crate::{CodegenError, LaunchHint, MetalKernelSource};
use ksearch_ir::{BinOp, DType, KirExpr, KirLaunch, KirStmt, KernelIr, OptSchedule, UnaryOp};

/// Generic Q4_K Load / VecMulSum expand (not a named matvec kernel).
const Q4K_LOAD_HELPER: &str = r#"
inline void ksearch_get_scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {
  if (j < 4u) { d = q[j] & 63u; m = q[j + 4u] & 63u; }
  else { d = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u); m = (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u); }
}
inline float ksearch_q4k_at(device const uchar* blk, float d, float dmin, uint j) {
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  uint is = (j / 64u) * 2u;
  uint qoff = (j / 64u) * 32u;
  uint jl = j % 64u;
  uchar sc, mn;
  if (jl < 32u) {
    ksearch_get_scale_min_k4(is, scales, sc, mn);
    return d * float(sc) * float(qs[qoff + jl] & 0x0Fu) - dmin * float(mn);
  } else {
    ksearch_get_scale_min_k4(is + 1u, scales, sc, mn);
    return d * float(sc) * float(qs[qoff + (jl - 32u)] >> 4u) - dmin * float(mn);
  }
}
inline float ksearch_load_q4k(device const uchar* A, uint idx) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  device const uchar* blk = A + (idx / QK) * BPB;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  return ksearch_q4k_at(blk, d, dmin, idx % QK);
}
// Shared super-block metadata for consecutive idx..idx+3 (vec Load expand).
inline float4 ksearch_load_q4k4(device const uchar* A, uint idx) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  uint b0 = idx / QK;
  if ((idx + 3u) / QK != b0) {
    return float4(ksearch_load_q4k(A, idx), ksearch_load_q4k(A, idx + 1u),
                  ksearch_load_q4k(A, idx + 2u), ksearch_load_q4k(A, idx + 3u));
  }
  device const uchar* blk = A + b0 * BPB;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  uint j = idx % QK;
  return float4(ksearch_q4k_at(blk, d, dmin, j),
                ksearch_q4k_at(blk, d, dmin, j + 1u),
                ksearch_q4k_at(blk, d, dmin, j + 2u),
                ksearch_q4k_at(blk, d, dmin, j + 3u));
}
inline float2 ksearch_load_q4k2(device const uchar* A, uint idx) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  uint b0 = idx / QK;
  if ((idx + 1u) / QK != b0) {
    return float2(ksearch_load_q4k(A, idx), ksearch_load_q4k(A, idx + 1u));
  }
  device const uchar* blk = A + b0 * BPB;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  uint j = idx % QK;
  return float2(ksearch_q4k_at(blk, d, dmin, j), ksearch_q4k_at(blk, d, dmin, j + 1u));
}
// One 6-bit scale/min governs 32 weights — vectorized nibble unpack + float4 dots.
inline float ksearch_dot_q4k32(device const uchar* A, uint widx, threadgroup const float* x, uint xidx) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  device const uchar* blk = A + (widx / QK) * BPB;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  uint j = widx % QK;
  uint is = (j / 64u) * 2u;
  uint qoff = (j / 64u) * 32u;
  uint jl = j % 64u;
  uchar sc, mn;
  float4 x0 = *(threadgroup const float4*)(x + xidx + 0u);
  float4 x1 = *(threadgroup const float4*)(x + xidx + 4u);
  float4 x2 = *(threadgroup const float4*)(x + xidx + 8u);
  float4 x3 = *(threadgroup const float4*)(x + xidx + 12u);
  float4 x4 = *(threadgroup const float4*)(x + xidx + 16u);
  float4 x5 = *(threadgroup const float4*)(x + xidx + 20u);
  float4 x6 = *(threadgroup const float4*)(x + xidx + 24u);
  float4 x7 = *(threadgroup const float4*)(x + xidx + 28u);
  packed_uchar4 qp0 = *(device const packed_uchar4*)(qs + qoff + 0u);
  packed_uchar4 qp1 = *(device const packed_uchar4*)(qs + qoff + 4u);
  packed_uchar4 qp2 = *(device const packed_uchar4*)(qs + qoff + 8u);
  packed_uchar4 qp3 = *(device const packed_uchar4*)(qs + qoff + 12u);
  packed_uchar4 qp4 = *(device const packed_uchar4*)(qs + qoff + 16u);
  packed_uchar4 qp5 = *(device const packed_uchar4*)(qs + qoff + 20u);
  packed_uchar4 qp6 = *(device const packed_uchar4*)(qs + qoff + 24u);
  packed_uchar4 qp7 = *(device const packed_uchar4*)(qs + qoff + 28u);
  uint4 q0 = uint4(qp0[0], qp0[1], qp0[2], qp0[3]);
  uint4 q1 = uint4(qp1[0], qp1[1], qp1[2], qp1[3]);
  uint4 q2 = uint4(qp2[0], qp2[1], qp2[2], qp2[3]);
  uint4 q3 = uint4(qp3[0], qp3[1], qp3[2], qp3[3]);
  uint4 q4 = uint4(qp4[0], qp4[1], qp4[2], qp4[3]);
  uint4 q5 = uint4(qp5[0], qp5[1], qp5[2], qp5[3]);
  uint4 q6 = uint4(qp6[0], qp6[1], qp6[2], qp6[3]);
  uint4 q7 = uint4(qp7[0], qp7[1], qp7[2], qp7[3]);
  float4 acc4 = float4(0.0f);
  if (jl < 32u) {
    ksearch_get_scale_min_k4(is, scales, sc, mn);
    float d1 = d * float(sc);
    float4 m = float4(dmin * float(mn));
    acc4 += (d1 * float4(int4(q0 & 0xFu)) - m) * x0;
    acc4 += (d1 * float4(int4(q1 & 0xFu)) - m) * x1;
    acc4 += (d1 * float4(int4(q2 & 0xFu)) - m) * x2;
    acc4 += (d1 * float4(int4(q3 & 0xFu)) - m) * x3;
    acc4 += (d1 * float4(int4(q4 & 0xFu)) - m) * x4;
    acc4 += (d1 * float4(int4(q5 & 0xFu)) - m) * x5;
    acc4 += (d1 * float4(int4(q6 & 0xFu)) - m) * x6;
    acc4 += (d1 * float4(int4(q7 & 0xFu)) - m) * x7;
  } else {
    ksearch_get_scale_min_k4(is + 1u, scales, sc, mn);
    float d2 = d * float(sc);
    float4 m = float4(dmin * float(mn));
    acc4 += (d2 * float4(int4(q0 >> 4u)) - m) * x0;
    acc4 += (d2 * float4(int4(q1 >> 4u)) - m) * x1;
    acc4 += (d2 * float4(int4(q2 >> 4u)) - m) * x2;
    acc4 += (d2 * float4(int4(q3 >> 4u)) - m) * x3;
    acc4 += (d2 * float4(int4(q4 >> 4u)) - m) * x4;
    acc4 += (d2 * float4(int4(q5 >> 4u)) - m) * x5;
    acc4 += (d2 * float4(int4(q6 >> 4u)) - m) * x6;
    acc4 += (d2 * float4(int4(q7 >> 4u)) - m) * x7;
  }
  return acc4.x + acc4.y + acc4.z + acc4.w;
}
// Full Q4_K super-block (256 weights) — one d/dmin load, 8 scale groups.
inline float ksearch_dot_q4k256(device const uchar* A, uint widx, threadgroup const float* x, uint xidx) {
  constexpr uint BPB = 144u;
  device const uchar* blk = A + (widx / 256u) * BPB;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  float acc = 0.0f;
  for (uint g = 0u; g < 4u; g++) {
    uchar sc0, mn0, sc1, mn1;
    ksearch_get_scale_min_k4(g * 2u, scales, sc0, mn0);
    ksearch_get_scale_min_k4(g * 2u + 1u, scales, sc1, mn1);
    float d0 = d * float(sc0);
    float m0 = dmin * float(mn0);
    float d1 = d * float(sc1);
    float m1 = dmin * float(mn1);
    uint qoff = g * 32u;
    uint x0 = xidx + g * 64u;
    for (uint l = 0u; l < 32u; l += 4u) {
      float4 ql = float4(float(qs[qoff + l] & 0x0Fu), float(qs[qoff + l + 1u] & 0x0Fu),
                         float(qs[qoff + l + 2u] & 0x0Fu), float(qs[qoff + l + 3u] & 0x0Fu));
      float4 qh = float4(float(qs[qoff + l] >> 4u), float(qs[qoff + l + 1u] >> 4u),
                         float(qs[qoff + l + 2u] >> 4u), float(qs[qoff + l + 3u] >> 4u));
      acc += dot(d0 * ql - float4(m0), *(threadgroup const float4*)(x + x0 + l));
      acc += dot(d1 * qh - float4(m1), *(threadgroup const float4*)(x + x0 + 32u + l));
    }
  }
  return acc;
}
inline float ksearch_dot_q4k32_dev(device const uchar* A, uint widx, device const half* x, uint xidx) {
  float acc = 0.0f;
  for (uint l = 0u; l < 32u; l++) {
    acc += ksearch_load_q4k(A, widx + l) * float(x[xidx + l]);
  }
  return acc;
}
inline float ksearch_dot_q4k256_dev(device const uchar* A, uint widx, device const half* x, uint xidx) {
  float acc = 0.0f;
  for (uint l = 0u; l < 256u; l++) {
    acc += ksearch_load_q4k(A, widx + l) * float(x[xidx + l]);
  }
  return acc;
}
// ggml-style simdgroup lane partial for one Q4_K superblock (mul_vec_q4_K layout).
inline float ksearch_q4k_coop_frag(
    device const uchar* A, uint row_base, uint cols, uint ib,
    threadgroup const float* x, uint lane) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  (void)cols;
  device const uchar* blk = A + (row_base / QK + ib) * BPB;
  float yl[16];
  float yh[16];
  threadgroup const float* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy = float4(0.0f);
  for (uint i = 0u; i < 8u; i++) {
    yl[i + 0u] = y4[i + 0u];   sumy[0] += yl[i + 0u];
    yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
    yh[i + 0u] = y4[i + 128u]; sumy[2] += yh[i + 0u];
    yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
  }
  device const uint16_t* sc = ((device const uint16_t*)(blk + 4)) + iq;
  device const uint16_t* q1 = ((device const uint16_t*)(blk + 16)) + 16u * iq + 4u * ir;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  sc16[0] = sc[0] & kmask1;
  sc16[1] = sc[2] & kmask1;
  sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
  sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
  device const uint16_t* q2 = q1 + 32;
  float4 acc1 = float4(0.0f);
  float4 acc2 = float4(0.0f);
  for (uint i = 0u; i < 4u; i++) {
    acc1[0] += yl[2u * i + 0u] * float(q1[i] & 0x000Fu);
    acc1[1] += yl[2u * i + 1u] * float(q1[i] & 0x0F00u);
    acc1[2] += yl[2u * i + 8u] * float(q1[i] & 0x00F0u);
    acc1[3] += yl[2u * i + 9u] * float(q1[i] & 0xF000u);
    acc2[0] += yh[2u * i + 0u] * float(q2[i] & 0x000Fu);
    acc2[1] += yh[2u * i + 1u] * float(q2[i] & 0x0F00u);
    acc2[2] += yh[2u * i + 8u] * float(q2[i] & 0x00F0u);
    acc2[3] += yh[2u * i + 9u] * float(q2[i] & 0xF000u);
  }
  return d * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
              (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
              (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
              (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
         dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                 sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
}
// One y-load; accumulate 4 rows for both gate and up weight streams.
inline void ksearch_q4k_coop_frag_nr4_dual(
    device const uchar* Ag, device const uchar* Au,
    uint row0_base, uint cols, uint ib,
    threadgroup const float* x, uint lane,
    thread float* g0, thread float* g1, thread float* g2, thread float* g3,
    thread float* u0, thread float* u1, thread float* u2, thread float* u3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  uint bpr = cols / QK;
  float yl[16];
  float yh[16];
  threadgroup const float* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy = float4(0.0f);
  for (uint i = 0u; i < 8u; i++) {
    yl[i + 0u] = y4[i + 0u];   sumy[0] += yl[i + 0u];
    yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
    yh[i + 0u] = y4[i + 128u]; sumy[2] += yh[i + 0u];
    yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
  }
  uint row_bytes_u16 = (bpr * BPB) / 2u;
  device const uchar* blk_g = Ag + (row0_base / QK + ib) * BPB;
  device const uchar* blk_u = Au + (row0_base / QK + ib) * BPB;
  device const uint16_t* sc_g = ((device const uint16_t*)(blk_g + 4)) + iq;
  device const uint16_t* q1_g = ((device const uint16_t*)(blk_g + 16)) + 16u * iq + 4u * ir;
  device const half* dh_g = (device const half*)(blk_g);
  device const uint16_t* sc_u = ((device const uint16_t*)(blk_u + 4)) + iq;
  device const uint16_t* q1_u = ((device const uint16_t*)(blk_u + 16)) + 16u * iq + 4u * ir;
  device const half* dh_u = (device const half*)(blk_u);
  float sumg[4] = {0.f, 0.f, 0.f, 0.f};
  float sumu[4] = {0.f, 0.f, 0.f, 0.f};
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  for (uint row = 0u; row < 4u; row++) {
    float4 a1g = float4(0.0f), a2g = float4(0.0f);
    float4 a1u = float4(0.0f), a2u = float4(0.0f);
    sc16[0] = sc_g[0] & kmask1;
    sc16[1] = sc_g[2] & kmask1;
    sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
    sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);
    device const uint16_t* q2_g = q1_g + 32;
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
      a1g[0] += yl[2u * i + 0u] * float(q1_g[i] & 0x000Fu);
      a1g[1] += yl[2u * i + 1u] * float(q1_g[i] & 0x0F00u);
      a1g[2] += yl[2u * i + 8u] * float(q1_g[i] & 0x00F0u);
      a1g[3] += yl[2u * i + 9u] * float(q1_g[i] & 0xF000u);
      a2g[0] += yh[2u * i + 0u] * float(q2_g[i] & 0x000Fu);
      a2g[1] += yh[2u * i + 1u] * float(q2_g[i] & 0x0F00u);
      a2g[2] += yh[2u * i + 8u] * float(q2_g[i] & 0x00F0u);
      a2g[3] += yh[2u * i + 9u] * float(q2_g[i] & 0xF000u);
    }
    sumg[row] += float(dh_g[0]) * ((a1g[0] + (1.0f / 256.0f) * a1g[1]) * float(sc8[0]) +
                                   (a1g[2] + (1.0f / 256.0f) * a1g[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                                   (a2g[0] + (1.0f / 256.0f) * a2g[1]) * float(sc8[4]) +
                                   (a2g[2] + (1.0f / 256.0f) * a2g[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                float(dh_g[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                  sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    sc16[0] = sc_u[0] & kmask1;
    sc16[1] = sc_u[2] & kmask1;
    sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
    sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);
    device const uint16_t* q2_u = q1_u + 32;
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
      a1u[0] += yl[2u * i + 0u] * float(q1_u[i] & 0x000Fu);
      a1u[1] += yl[2u * i + 1u] * float(q1_u[i] & 0x0F00u);
      a1u[2] += yl[2u * i + 8u] * float(q1_u[i] & 0x00F0u);
      a1u[3] += yl[2u * i + 9u] * float(q1_u[i] & 0xF000u);
      a2u[0] += yh[2u * i + 0u] * float(q2_u[i] & 0x000Fu);
      a2u[1] += yh[2u * i + 1u] * float(q2_u[i] & 0x0F00u);
      a2u[2] += yh[2u * i + 8u] * float(q2_u[i] & 0x00F0u);
      a2u[3] += yh[2u * i + 9u] * float(q2_u[i] & 0xF000u);
    }
    sumu[row] += float(dh_u[0]) * ((a1u[0] + (1.0f / 256.0f) * a1u[1]) * float(sc8[0]) +
                                   (a1u[2] + (1.0f / 256.0f) * a1u[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                                   (a2u[0] + (1.0f / 256.0f) * a2u[1]) * float(sc8[4]) +
                                   (a2u[2] + (1.0f / 256.0f) * a2u[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                float(dh_u[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                  sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    q1_g += row_bytes_u16; sc_g += row_bytes_u16; dh_g += row_bytes_u16;
    q1_u += row_bytes_u16; sc_u += row_bytes_u16; dh_u += row_bytes_u16;
  }
  *g0 += sumg[0]; *g1 += sumg[1]; *g2 += sumg[2]; *g3 += sumg[3];
  *u0 += sumu[0]; *u1 += sumu[1]; *u2 += sumu[2]; *u3 += sumu[3];
}
inline void ksearch_q4k_coop_frag_nr4(
    device const uchar* A, uint row0_base, uint cols, uint ib,
    threadgroup const float* x, uint lane,
    thread float* a0, thread float* a1, thread float* a2, thread float* a3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  uint bpr = cols / QK;
  float yl[16];
  float yh[16];
  threadgroup const float* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy = float4(0.0f);
  for (uint i = 0u; i < 8u; i++) {
    yl[i + 0u] = y4[i + 0u];   sumy[0] += yl[i + 0u];
    yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
    yh[i + 0u] = y4[i + 128u]; sumy[2] += yh[i + 0u];
    yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
  }
  device const uchar* blk0 = A + (row0_base / QK + ib) * BPB;
  device const uint16_t* sc = ((device const uint16_t*)(blk0 + 4)) + iq;
  device const uint16_t* q1 = ((device const uint16_t*)(blk0 + 16)) + 16u * iq + 4u * ir;
  device const half* dh = (device const half*)(blk0);
  thread float* accs[4] = { a0, a1, a2, a3 };
  uint row_bytes_u16 = (bpr * BPB) / 2u;
  for (uint row = 0u; row < 4u; row++) {
    uint16_t sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
    device const uint16_t* q2 = q1 + 32;
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    for (uint i = 0u; i < 4u; i++) {
      acc1[0] += yl[2u * i + 0u] * float(q1[i] & 0x000Fu);
      acc1[1] += yl[2u * i + 1u] * float(q1[i] & 0x0F00u);
      acc1[2] += yl[2u * i + 8u] * float(q1[i] & 0x00F0u);
      acc1[3] += yl[2u * i + 9u] * float(q1[i] & 0xF000u);
      acc2[0] += yh[2u * i + 0u] * float(q2[i] & 0x000Fu);
      acc2[1] += yh[2u * i + 1u] * float(q2[i] & 0x0F00u);
      acc2[2] += yh[2u * i + 8u] * float(q2[i] & 0x00F0u);
      acc2[3] += yh[2u * i + 9u] * float(q2[i] & 0xF000u);
    }
    float d = float(dh[0]);
    float dmin = float(dh[1]);
    *accs[row] += d * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                       (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                       (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                       (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                  dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                          sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    q1 += row_bytes_u16;
    sc += row_bytes_u16;
    dh += row_bytes_u16;
  }
}


// Device-half y variants (no TG staging — oracle mul_vec streams device activations).
inline void ksearch_q4k_load_y_dev(device const half* y4, thread float* yl, thread float* yh, thread float4* sumy) {
  float4 yl0 = float4(*(device const half4*)(y4 + 0u));
  float4 yl1 = float4(*(device const half4*)(y4 + 4u));
  float4 yl2 = float4(*(device const half4*)(y4 + 32u));
  float4 yl3 = float4(*(device const half4*)(y4 + 36u));
  float4 yh0 = float4(*(device const half4*)(y4 + 128u));
  float4 yh1 = float4(*(device const half4*)(y4 + 132u));
  float4 yh2 = float4(*(device const half4*)(y4 + 160u));
  float4 yh3 = float4(*(device const half4*)(y4 + 164u));
  ((thread float4*)yl)[0] = yl0;
  ((thread float4*)yl)[1] = yl1;
  ((thread float4*)yl)[2] = yl2;
  ((thread float4*)yl)[3] = yl3;
  ((thread float4*)yh)[0] = yh0;
  ((thread float4*)yh)[1] = yh1;
  ((thread float4*)yh)[2] = yh2;
  ((thread float4*)yh)[3] = yh3;
  *sumy = float4(
      yl0[0] + yl0[1] + yl0[2] + yl0[3] + yl1[0] + yl1[1] + yl1[2] + yl1[3],
      yl2[0] + yl2[1] + yl2[2] + yl2[3] + yl3[0] + yl3[1] + yl3[2] + yl3[3],
      yh0[0] + yh0[1] + yh0[2] + yh0[3] + yh1[0] + yh1[1] + yh1[2] + yh1[3],
      yh2[0] + yh2[1] + yh2[2] + yh2[3] + yh3[0] + yh3[1] + yh3[2] + yh3[3]);
}
inline float ksearch_q4k_coop_frag_dev(
    device const uchar* A, uint row_base, uint cols, uint ib,
    device const half* x, uint lane) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  (void)cols;
  device const uchar* blk = A + (row_base / QK + ib) * BPB;
  float yl[16];
  float yh[16];
  device const half* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy;
  ksearch_q4k_load_y_dev(y4, yl, yh, &sumy);
  device const uint16_t* sc = ((device const uint16_t*)(blk + 4)) + iq;
  device const uint16_t* q1 = ((device const uint16_t*)(blk + 16)) + 16u * iq + 4u * ir;
  float2 dd = float2(*(device const half2*)blk);
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  sc16[0] = sc[0] & kmask1;
  sc16[1] = sc[2] & kmask1;
  sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
  sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
  ushort4 q1v = *(device const ushort4*)q1;
  ushort4 q2v = *(device const ushort4*)(q1 + 32);
  float4 acc1 = float4(0.0f);
  float4 acc2 = float4(0.0f);
  #pragma unroll
  for (uint i = 0u; i < 4u; i++) {
    acc1[0] += yl[2u * i + 0u] * float(q1v[i] & 0x000Fu);
    acc1[1] += yl[2u * i + 1u] * float(q1v[i] & 0x0F00u);
    acc1[2] += yl[2u * i + 8u] * float(q1v[i] & 0x00F0u);
    acc1[3] += yl[2u * i + 9u] * float(q1v[i] & 0xF000u);
    acc2[0] += yh[2u * i + 0u] * float(q2v[i] & 0x000Fu);
    acc2[1] += yh[2u * i + 1u] * float(q2v[i] & 0x0F00u);
    acc2[2] += yh[2u * i + 8u] * float(q2v[i] & 0x00F0u);
    acc2[3] += yh[2u * i + 9u] * float(q2v[i] & 0xF000u);
  }
  return dd[0] * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                  (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                  (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                  (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
         dd[1] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                  sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
}
inline void ksearch_q4k_coop_frag_nr4_dev(
    device const uchar* A, uint row0_base, uint cols, uint ib,
    device const half* x, uint lane,
    thread float* a0, thread float* a1, thread float* a2, thread float* a3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  uint bpr = cols / QK;
  float yl[16];
  float yh[16];
  device const half* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy;
  ksearch_q4k_load_y_dev(y4, yl, yh, &sumy);
  device const uchar* blk0 = A + (row0_base / QK + ib) * BPB;
  device const uint16_t* sc = ((device const uint16_t*)(blk0 + 4)) + iq;
  device const uint16_t* q1 = ((device const uint16_t*)(blk0 + 16)) + 16u * iq + 4u * ir;
  device const half* dh = (device const half*)(blk0);
  float sumf[4] = {0.f, 0.f, 0.f, 0.f};
  uint row_bytes_u16 = (bpr * BPB) / 2u;
  for (uint row = 0u; row < 4u; row++) {
    uint16_t sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
    ushort4 q1v = *(device const ushort4*)q1;
    ushort4 q2v = *(device const ushort4*)(q1 + 32);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
      acc1[0] += yl[2u * i + 0u] * float(q1v[i] & 0x000Fu);
      acc1[1] += yl[2u * i + 1u] * float(q1v[i] & 0x0F00u);
      acc1[2] += yl[2u * i + 8u] * float(q1v[i] & 0x00F0u);
      acc1[3] += yl[2u * i + 9u] * float(q1v[i] & 0xF000u);
      acc2[0] += yh[2u * i + 0u] * float(q2v[i] & 0x000Fu);
      acc2[1] += yh[2u * i + 1u] * float(q2v[i] & 0x0F00u);
      acc2[2] += yh[2u * i + 8u] * float(q2v[i] & 0x00F0u);
      acc2[3] += yh[2u * i + 9u] * float(q2v[i] & 0xF000u);
    }
    float2 dd = float2(*(device const half2*)dh);
    sumf[row] += dd[0] * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                           (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                           (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                           (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                  dd[1] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                           sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    q1 += row_bytes_u16;
    sc += row_bytes_u16;
    dh += row_bytes_u16;
  }
  *a0 += sumf[0]; *a1 += sumf[1]; *a2 += sumf[2]; *a3 += sumf[3];
}
inline void ksearch_q4k_coop_frag_nr4_dual_dev(
    device const uchar* Ag, device const uchar* Au,
    uint row0_base, uint cols, uint ib,
    device const half* x, uint lane,
    thread float* g0, thread float* g1, thread float* g2, thread float* g3,
    thread float* u0, thread float* u1, thread float* u2, thread float* u3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  uint it = lane % 8u;
  uint iq = it / 4u;
  uint ir = it % 4u;
  uint bpr = cols / QK;
  float yl[16];
  float yh[16];
  device const half* y4 = x + ib * QK + 64u * iq + 8u * ir;
  float4 sumy;
  ksearch_q4k_load_y_dev(y4, yl, yh, &sumy);
  uint row_bytes_u16 = (bpr * BPB) / 2u;
  device const uchar* blk_g = Ag + (row0_base / QK + ib) * BPB;
  device const uchar* blk_u = Au + (row0_base / QK + ib) * BPB;
  device const uint16_t* sc_g = ((device const uint16_t*)(blk_g + 4)) + iq;
  device const uint16_t* q1_g = ((device const uint16_t*)(blk_g + 16)) + 16u * iq + 4u * ir;
  device const half* dh_g = (device const half*)(blk_g);
  device const uint16_t* sc_u = ((device const uint16_t*)(blk_u + 4)) + iq;
  device const uint16_t* q1_u = ((device const uint16_t*)(blk_u + 16)) + 16u * iq + 4u * ir;
  device const half* dh_u = (device const half*)(blk_u);
  float sumg[4] = {0.f, 0.f, 0.f, 0.f};
  float sumu[4] = {0.f, 0.f, 0.f, 0.f};
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  for (uint row = 0u; row < 4u; row++) {
    float4 a1g = float4(0.0f), a2g = float4(0.0f);
    float4 a1u = float4(0.0f), a2u = float4(0.0f);
    sc16[0] = sc_g[0] & kmask1;
    sc16[1] = sc_g[2] & kmask1;
    sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
    sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);
    ushort4 q1gv = *(device const ushort4*)q1_g;
    ushort4 q2gv = *(device const ushort4*)(q1_g + 32);
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
      a1g[0] += yl[2u * i + 0u] * float(q1gv[i] & 0x000Fu);
      a1g[1] += yl[2u * i + 1u] * float(q1gv[i] & 0x0F00u);
      a1g[2] += yl[2u * i + 8u] * float(q1gv[i] & 0x00F0u);
      a1g[3] += yl[2u * i + 9u] * float(q1gv[i] & 0xF000u);
      a2g[0] += yh[2u * i + 0u] * float(q2gv[i] & 0x000Fu);
      a2g[1] += yh[2u * i + 1u] * float(q2gv[i] & 0x0F00u);
      a2g[2] += yh[2u * i + 8u] * float(q2gv[i] & 0x00F0u);
      a2g[3] += yh[2u * i + 9u] * float(q2gv[i] & 0xF000u);
    }
    float2 ddg = float2(*(device const half2*)dh_g);
    sumg[row] += ddg[0] * ((a1g[0] + (1.0f / 256.0f) * a1g[1]) * float(sc8[0]) +
                           (a1g[2] + (1.0f / 256.0f) * a1g[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                           (a2g[0] + (1.0f / 256.0f) * a2g[1]) * float(sc8[4]) +
                           (a2g[2] + (1.0f / 256.0f) * a2g[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                ddg[1] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                          sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    sc16[0] = sc_u[0] & kmask1;
    sc16[1] = sc_u[2] & kmask1;
    sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
    sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);
    ushort4 q1uv = *(device const ushort4*)q1_u;
    ushort4 q2uv = *(device const ushort4*)(q1_u + 32);
    #pragma unroll
    for (uint i = 0u; i < 4u; i++) {
      a1u[0] += yl[2u * i + 0u] * float(q1uv[i] & 0x000Fu);
      a1u[1] += yl[2u * i + 1u] * float(q1uv[i] & 0x0F00u);
      a1u[2] += yl[2u * i + 8u] * float(q1uv[i] & 0x00F0u);
      a1u[3] += yl[2u * i + 9u] * float(q1uv[i] & 0xF000u);
      a2u[0] += yh[2u * i + 0u] * float(q2uv[i] & 0x000Fu);
      a2u[1] += yh[2u * i + 1u] * float(q2uv[i] & 0x0F00u);
      a2u[2] += yh[2u * i + 8u] * float(q2uv[i] & 0x00F0u);
      a2u[3] += yh[2u * i + 9u] * float(q2uv[i] & 0xF000u);
    }
    float2 ddu = float2(*(device const half2*)dh_u);
    sumu[row] += ddu[0] * ((a1u[0] + (1.0f / 256.0f) * a1u[1]) * float(sc8[0]) +
                           (a1u[2] + (1.0f / 256.0f) * a1u[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                           (a2u[0] + (1.0f / 256.0f) * a2u[1]) * float(sc8[4]) +
                           (a2u[2] + (1.0f / 256.0f) * a2u[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                ddu[1] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                          sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    q1_g += row_bytes_u16; sc_g += row_bytes_u16; dh_g += row_bytes_u16;
    q1_u += row_bytes_u16; sc_u += row_bytes_u16; dh_u += row_bytes_u16;
  }
  *g0 += sumg[0]; *g1 += sumg[1]; *g2 += sumg[2]; *g3 += sumg[3];
  *u0 += sumu[0]; *u1 += sumu[1]; *u2 += sumu[2]; *u3 += sumu[3];
}



"#;

/// Generic Q6_K Load / coop expand (not a named matvec kernel).
const Q6K_LOAD_HELPER: &str = r#"
// --- Q6_K Load expand + ggml mul_vec_q6_K-shaped coop ---
inline float ksearch_load_q6k(device const uchar* A, uint idx) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 210u;
  device const uchar* blk = A + (idx / QK) * BPB;
  uint j = idx % QK;
  uint hpart = j / 128u;
  uint jl = j % 128u;
  device const uchar* ql = blk + hpart * 64u;
  device const uchar* qh = blk + 128u + hpart * 32u;
  device const char* sc = (device const char*)(blk + 192u) + hpart * 8u;
  float d = float(*(device const half*)(blk + 208));
  uint l = jl % 32u;
  uint is = l / 16u;
  int q;
  if (jl < 32u) {
    q = int((ql[l] & 0x0Fu) | (((qh[l] >> 0u) & 3u) << 4u)) - 32;
    return d * float(sc[is]) * float(q);
  } else if (jl < 64u) {
    q = int((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2u) & 3u) << 4u)) - 32;
    return d * float(sc[is + 2]) * float(q);
  } else if (jl < 96u) {
    q = int((ql[l] >> 4u) | (((qh[l] >> 4u) & 3u) << 4u)) - 32;
    return d * float(sc[is + 4]) * float(q);
  } else {
    q = int((ql[l + 32u] >> 4u) | (((qh[l] >> 6u) & 3u) << 4u)) - 32;
    return d * float(sc[is + 6]) * float(q);
  }
}
inline float ksearch_q6k_coop_frag_regs(
    device const uchar* A, uint row_base, uint cols, uint ib,
    thread float* yl, uint lane) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 210u;
  constexpr uchar kmask1 = 0x03u;
  constexpr uchar kmask2 = 0x0Cu;
  constexpr uchar kmask3 = 0x30u;
  constexpr uchar kmask4 = 0xC0u;
  (void)cols;
  uint tid = lane / 2u;
  uint ip = tid / 8u;
  uint il = tid % 8u;
  uint l0 = 4u * il;
  uint is = 8u * ip + l0 / 16u;
  uint q_offset_l = 64u * ip + l0;
  uint q_offset_h = 32u * ip + l0;
  device const uchar* blk = A + (row_base / QK + ib) * BPB;
  device const uchar* q1 = blk + q_offset_l;
  device const uchar* q2 = q1 + 32u;
  device const uchar* qh = blk + 128u + q_offset_h;
  device const char* sc = (device const char*)(blk + 192u) + is;
  float d = float(*(device const half*)(blk + 208));
  float4 sums = float4(0.0f);
  #pragma unroll
  for (uint l = 0u; l < 4u; l++) {
    sums[0] += yl[4u * l + 0u] * float(char((q1[l] & 0x0Fu) | ((qh[l] & kmask1) << 4u)) - 32);
    sums[1] += yl[4u * l + 1u] * float(char((q2[l] & 0x0Fu) | ((qh[l] & kmask2) << 2u)) - 32);
    sums[2] += yl[4u * l + 2u] * float(char((q1[l] >> 4u) | ((qh[l] & kmask3) << 0u)) - 32);
    sums[3] += yl[4u * l + 3u] * float(char((q2[l] >> 4u) | ((qh[l] & kmask4) >> 2u)) - 32);
  }
  return d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
}
inline void ksearch_q6k_load_y_dev(device const half* y, thread float* yl) {
  float4 ya = float4(*(device const half4*)(y + 0u));
  float4 yb = float4(*(device const half4*)(y + 32u));
  float4 yc = float4(*(device const half4*)(y + 64u));
  float4 yd = float4(*(device const half4*)(y + 96u));
  #pragma unroll
  for (uint l = 0u; l < 4u; l++) {
    yl[4u * l + 0u] = ya[l];
    yl[4u * l + 1u] = yb[l];
    yl[4u * l + 2u] = yc[l];
    yl[4u * l + 3u] = yd[l];
  }
}
inline float ksearch_q6k_coop_frag(
    device const uchar* A, uint row_base, uint cols, uint ib,
    threadgroup const float* x, uint x_off, uint lane) {
  constexpr uint QK = 256u;
  uint tid = lane / 2u;
  uint ip = tid / 8u;
  uint il = tid % 8u;
  uint l0 = 4u * il;
  uint y_offset = 128u * ip + l0;
  threadgroup const float* y = x + (ib * QK - x_off) + y_offset;
  float yl[16];
  for (uint l = 0u; l < 4u; l++) {
    yl[4u * l + 0u] = y[l + 0u];
    yl[4u * l + 1u] = y[l + 32u];
    yl[4u * l + 2u] = y[l + 64u];
    yl[4u * l + 3u] = y[l + 96u];
  }
  return ksearch_q6k_coop_frag_regs(A, row_base, cols, ib, yl, lane);
}
inline float ksearch_q6k_coop_frag_dev(
    device const uchar* A, uint row_base, uint cols, uint ib,
    device const half* x, uint lane) {
  constexpr uint QK = 256u;
  uint tid = lane / 2u;
  uint ip = tid / 8u;
  uint il = tid % 8u;
  uint l0 = 4u * il;
  uint y_offset = 128u * ip + l0;
  device const half* y = x + ib * QK + y_offset;
  float yl[16];
  ksearch_q6k_load_y_dev(y, yl);
  return ksearch_q6k_coop_frag_regs(A, row_base, cols, ib, yl, lane);
}
inline void ksearch_q6k_coop_frag_nr4(
    device const uchar* A, uint row0_base, uint cols, uint ib,
    threadgroup const float* x, uint x_off, uint lane,
    thread float* a0, thread float* a1, thread float* a2, thread float* a3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 210u;
  constexpr uchar kmask1 = 0x03u;
  constexpr uchar kmask2 = 0x0Cu;
  constexpr uchar kmask3 = 0x30u;
  constexpr uchar kmask4 = 0xC0u;
  uint tid = lane / 2u;
  uint ip = tid / 8u;
  uint il = tid % 8u;
  uint l0 = 4u * il;
  uint is = 8u * ip + l0 / 16u;
  uint y_offset = 128u * ip + l0;
  uint q_offset_l = 64u * ip + l0;
  uint q_offset_h = 32u * ip + l0;
  uint bpr = cols / QK;
  uint row_bytes = bpr * BPB;
  threadgroup const float* y = x + (ib * QK - x_off) + y_offset;
  float yl[16];
  for (uint l = 0u; l < 4u; l++) {
    yl[4u * l + 0u] = y[l + 0u];
    yl[4u * l + 1u] = y[l + 32u];
    yl[4u * l + 2u] = y[l + 64u];
    yl[4u * l + 3u] = y[l + 96u];
  }
  device const uchar* blk = A + (row0_base / QK + ib) * BPB;
  device const uchar* q1 = blk + q_offset_l;
  device const uchar* q2 = q1 + 32u;
  device const uchar* qh = blk + 128u + q_offset_h;
  device const char* sc = (device const char*)(blk + 192u) + is;
  device const half* dh = (device const half*)(blk + 208);
  float sumf[4] = {0.f, 0.f, 0.f, 0.f};
  for (uint row = 0u; row < 4u; row++) {
    float4 sums = float4(0.0f);
    #pragma unroll
    for (uint l = 0u; l < 4u; l++) {
      sums[0] += yl[4u * l + 0u] * float(char((q1[l] & 0x0Fu) | ((qh[l] & kmask1) << 4u)) - 32);
      sums[1] += yl[4u * l + 1u] * float(char((q2[l] & 0x0Fu) | ((qh[l] & kmask2) << 2u)) - 32);
      sums[2] += yl[4u * l + 2u] * float(char((q1[l] >> 4u) | ((qh[l] & kmask3) << 0u)) - 32);
      sums[3] += yl[4u * l + 3u] * float(char((q2[l] >> 4u) | ((qh[l] & kmask4) >> 2u)) - 32);
    }
    sumf[row] += float(dh[0]) * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) +
                                 sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    q1 += row_bytes; q2 += row_bytes; qh += row_bytes; sc += row_bytes; dh += row_bytes / 2u;
  }
  *a0 += sumf[0]; *a1 += sumf[1]; *a2 += sumf[2]; *a3 += sumf[3];
}
inline void ksearch_q6k_coop_frag_nr4_dev(
    device const uchar* A, uint row0_base, uint cols, uint ib,
    device const half* x, uint lane,
    thread float* a0, thread float* a1, thread float* a2, thread float* a3) {
  constexpr uint QK = 256u;
  constexpr uint BPB = 210u;
  constexpr uchar kmask1 = 0x03u;
  constexpr uchar kmask2 = 0x0Cu;
  constexpr uchar kmask3 = 0x30u;
  constexpr uchar kmask4 = 0xC0u;
  uint tid = lane / 2u;
  uint ip = tid / 8u;
  uint il = tid % 8u;
  uint l0 = 4u * il;
  uint is = 8u * ip + l0 / 16u;
  uint y_offset = 128u * ip + l0;
  uint q_offset_l = 64u * ip + l0;
  uint q_offset_h = 32u * ip + l0;
  uint bpr = cols / QK;
  uint row_bytes = bpr * BPB;
  device const half* y = x + ib * QK + y_offset;
  float yl[16];
  ksearch_q6k_load_y_dev(y, yl);
  device const uchar* blk = A + (row0_base / QK + ib) * BPB;
  device const uchar* q1 = blk + q_offset_l;
  device const uchar* q2 = q1 + 32u;
  device const uchar* qh = blk + 128u + q_offset_h;
  device const char* sc = (device const char*)(blk + 192u) + is;
  device const half* dh = (device const half*)(blk + 208);
  float sumf[4] = {0.f, 0.f, 0.f, 0.f};
  for (uint row = 0u; row < 4u; row++) {
    float4 sums = float4(0.0f);
    #pragma unroll
    for (uint l = 0u; l < 4u; l++) {
      sums[0] += yl[4u * l + 0u] * float(char((q1[l] & 0x0Fu) | ((qh[l] & kmask1) << 4u)) - 32);
      sums[1] += yl[4u * l + 1u] * float(char((q2[l] & 0x0Fu) | ((qh[l] & kmask2) << 2u)) - 32);
      sums[2] += yl[4u * l + 2u] * float(char((q1[l] >> 4u) | ((qh[l] & kmask3) << 0u)) - 32);
      sums[3] += yl[4u * l + 3u] * float(char((q2[l] >> 4u) | ((qh[l] & kmask4) >> 2u)) - 32);
    }
    sumf[row] += float(dh[0]) * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) +
                                 sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    q1 += row_bytes; q2 += row_bytes; qh += row_bytes; sc += row_bytes; dh += row_bytes / 2u;
  }
  *a0 += sumf[0]; *a1 += sumf[1]; *a2 += sumf[2]; *a3 += sumf[3];
}



"#;

/// Generic Q4_0 Load / pack expand (KV cache; same pattern as Q4_K Load expand).
const Q40_LOAD_HELPER: &str = r#"
constant uint Q40_QK = 32u;
constant uint Q40_BS = 18u;
inline float ksearch_q40_at(device const uchar* blk, float d, uint j) {
  uchar q = blk[2u + (j & 15u)];
  int qv = (j < 16u) ? int(q & 0x0Fu) - 8 : int(q >> 4u) - 8;
  return float(qv) * d;
}
inline float ksearch_load_q40(device const uchar* A, uint idx) {
  uint block = idx / Q40_QK;
  uint j = idx % Q40_QK;
  device const uchar* blk = A + block * Q40_BS;
  float d = float(*(device const half*)blk);
  return ksearch_q40_at(blk, d, j);
}
inline float4 ksearch_load_q404(device const uchar* A, uint idx) {
  uint block = idx / Q40_QK;
  uint j = idx % Q40_QK;
  device const uchar* blk = A + block * Q40_BS;
  float d = float(*(device const half*)blk);
  return float4(ksearch_q40_at(blk, d, j),
                ksearch_q40_at(blk, d, j + 1u),
                ksearch_q40_at(blk, d, j + 2u),
                ksearch_q40_at(blk, d, j + 3u));
}
inline void ksearch_pack_q40(
    device uchar* out, uint block, device const half* src, uint src_elem) {
  float vals[32];
  float max_abs = 0.0f;
  for (uint d = 0u; d < 32u; d++) {
    float v = float(src[src_elem + d]);
    vals[d] = v;
    max_abs = max(max_abs, fabs(v));
  }
  float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
  float inv = 1.0f / scale;
  device uchar* blk = out + block * Q40_BS;
  *((device half*)blk) = half(scale);
  for (uint i = 0u; i < 16u; i++) {
    int q_lo = clamp(int(round(vals[i] * inv)) + 8, 0, 15);
    int q_hi = clamp(int(round(vals[i + 16u] * inv)) + 8, 0, 15);
    blk[2u + i] = uchar(q_lo | (q_hi << 4));
  }
}
inline void ksearch_pack_q40_th(
    device uchar* out, uint block, thread const float* src, uint src_off) {
  float vals[32];
  float max_abs = 0.0f;
  for (uint d = 0u; d < 32u; d++) {
    // Match F16 store/load round-trip before pack.
    float v = float(half(src[src_off + d]));
    vals[d] = v;
    max_abs = max(max_abs, fabs(v));
  }
  float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
  float inv = 1.0f / scale;
  device uchar* blk = out + block * Q40_BS;
  *((device half*)blk) = half(scale);
  for (uint i = 0u; i < 16u; i++) {
    int q_lo = clamp(int(round(vals[i] * inv)) + 8, 0, 15);
    int q_hi = clamp(int(round(vals[i + 16u] * inv)) + 8, 0, 15);
    blk[2u + i] = uchar(q_lo | (q_hi << 4));
  }
}
inline void ksearch_dequant_q40_to_tg(
    threadgroup float* dst, device const uchar* A, uint src_elem) {
  uint block = src_elem / Q40_QK;
  device const uchar* blk = A + block * Q40_BS;
  float d = float(*(device const half*)blk);
  for (uint j = 0u; j < 32u; j++) {
    dst[j] = ksearch_q40_at(blk, d, j);
  }
}
"#;

pub fn render_msl(kir: &KernelIr, _sched: OptSchedule) -> Result<MetalKernelSource, CodegenError> {
    // Q4_K is a Load dtype like F16: generic expand to float (tinygrad ggml_data_to_tensor ops).
    // No named matvec_q4k kernel template — only dtype expansion on Load / VecMulSum.
    let mut src = String::from("#include <metal_stdlib>\nusing namespace metal;\n\n");
    if body_needs_q4(&kir.body) {
        src.push_str(Q4K_LOAD_HELPER);
    }
    if body_needs_q6(&kir.body) {
        src.push_str(Q6K_LOAD_HELPER);
    }
    if body_needs_q40(&kir.body) {
        src.push_str(Q40_LOAD_HELPER);
    }
    let n_in = kir.n_inputs as u32;
    let (params, launch, guard) = match &kir.launch {
        KirLaunch::Elementwise { n } => {
            let p = buffer_params(kir);
            (
                format!("{p}  uint gid [[thread_position_in_grid]]"),
                LaunchHint::Elementwise { n: (*n).max(1) },
                format!("  if (gid >= {}u) return;\n", (*n).max(1)),
            )
        }
        KirLaunch::Rows { rows } => {
            let p = buffer_params(kir);
            (
                format!("{p}  uint gid [[thread_position_in_grid]]"),
                LaunchHint::Elementwise {
                    n: (*rows).max(1),
                },
                format!("  if (gid >= {}u) return;\n", (*rows).max(1)),
            )
        }
        KirLaunch::RowsParallel { rows, tg } => {
            let p = buffer_params(kir);
            (
                format!(
                    "{p}  uint gid [[threadgroup_position_in_grid]],\n  uint lid [[thread_index_in_threadgroup]]"
                ),
                LaunchHint::RowsParallel {
                    rows: (*rows).max(1),
                    tg: (*tg).max(1),
                },
                format!("  if (gid >= {}u) return;\n", (*rows).max(1)),
            )
        }
        KirLaunch::RowsParallelSg { rows, nsg } => {
            let p = buffer_params(kir);
            (
                format!(
                    "{p}  uint gid [[threadgroup_position_in_grid]],\n  uint tiisg [[thread_index_in_simdgroup]],\n  uint sgitg [[simdgroup_index_in_threadgroup]]"
                ),
                LaunchHint::RowsParallelSg {
                    rows: (*rows).max(1),
                    nsg: (*nsg).max(1),
                },
                format!(
                    "  if (gid >= {}u) return;\n  uint lid = sgitg * 32u + tiisg;\n",
                    (*rows).max(1)
                ),
            )
        }
    };
    let body = emit_stmts(&kir.body, 1, n_in, kir.n_outputs as u32, kir.out_dtype)?;
    src.push_str(&format!(
        "kernel void {name}(\n{params}\n) {{\n{guard}{body}}}\n",
        name = kir.name,
        params = params,
        guard = guard,
        body = body,
    ));
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source: src,
        n_inputs: kir.n_inputs,
        n_outputs: kir.n_outputs.max(1),
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch,
    })
}

fn buffer_params(kir: &KernelIr) -> String {
    let mut in_dt = vec![kir.out_dtype; kir.n_inputs];
    infer_load_dtypes(&kir.body, &mut in_dt);
    let mut p = String::new();
    for i in 0..kir.n_inputs {
        let ty = in_dt[i].msl();
        p.push_str(&format!(
            "  device const {ty}* in{i} [[buffer({i})]],\n"
        ));
    }
    let out_ty = kir.out_dtype.msl();
    let n_out = kir.n_outputs.max(1);
    if n_out == 1 {
        p.push_str(&format!(
            "  device {out_ty}* out [[buffer({})]],\n",
            kir.n_inputs
        ));
    } else {
        for o in 0..n_out {
            p.push_str(&format!(
                "  device {out_ty}* out{o} [[buffer({})]],\n",
                kir.n_inputs + o
            ));
        }
    }
    p
}

fn infer_load_dtypes(stmts: &[KirStmt], in_dt: &mut [DType]) {
    for s in stmts {
        match s {
            KirStmt::For { body, .. } | KirStmt::If { body, .. } | KirStmt::ForRange { body, .. } => {
                infer_load_dtypes(body, in_dt);
            }
            KirStmt::Let { expr, .. } | KirStmt::LetU32 { expr, .. } | KirStmt::Assign { expr, .. } => {
                infer_load_expr(expr, in_dt);
            }
            KirStmt::Store { idx, val, .. }
            | KirStmt::TgStore { idx, val, .. }
            | KirStmt::ThreadStore { idx, val, .. } => {
                infer_load_expr(idx, in_dt);
                infer_load_expr(val, in_dt);
            }
            KirStmt::TgDeclF32 { .. }
            | KirStmt::ThreadDeclF32 { .. }
            | KirStmt::Barrier
            | KirStmt::ThreadgroupReduce { .. }
            | KirStmt::ThreadgroupArgmax { .. } => {}
            KirStmt::Q4kCoopNr4 {
                w_buf,
                row0_base,
                ib,
                lane,
                b_from_tg,
                x_buf,
                ..
            } => {
                if (*w_buf as usize) < in_dt.len() {
                    in_dt[*w_buf as usize] = DType::Q4K;
                }
                if b_from_tg.is_none() && (*x_buf as usize) < in_dt.len() {
                    in_dt[*x_buf as usize] = DType::F16;
                }
                infer_load_expr(row0_base, in_dt);
                infer_load_expr(ib, in_dt);
                infer_load_expr(lane, in_dt);
            }
            KirStmt::Q4kCoopNr4Dual {
                row0_base,
                ib,
                lane,
                b_from_tg,
                x_buf,
                ..
            } => {
                if in_dt.len() > 0 {
                    in_dt[0] = DType::Q4K;
                }
                if in_dt.len() > 1 {
                    in_dt[1] = DType::Q4K;
                }
                if b_from_tg.is_none() && (*x_buf as usize) < in_dt.len() {
                    in_dt[*x_buf as usize] = DType::F16;
                }
                infer_load_expr(row0_base, in_dt);
                infer_load_expr(ib, in_dt);
                infer_load_expr(lane, in_dt);
            }
            KirStmt::Q6kCoopNr4 {
                w_buf,
                row0_base,
                ib,
                lane,
                b_from_tg,
                x_buf,
                ..
            } => {
                if (*w_buf as usize) < in_dt.len() {
                    in_dt[*w_buf as usize] = DType::Q6K;
                }
                if b_from_tg.is_none() && (*x_buf as usize) < in_dt.len() {
                    in_dt[*x_buf as usize] = DType::F16;
                }
                infer_load_expr(row0_base, in_dt);
                infer_load_expr(ib, in_dt);
                infer_load_expr(lane, in_dt);
            }
            KirStmt::Q40PackBlock {
                src_buf,
                block,
                src_elem,
                ..
            } => {
                if (*src_buf as usize) < in_dt.len() {
                    in_dt[*src_buf as usize] = DType::F16;
                }
                infer_load_expr(block, in_dt);
                infer_load_expr(src_elem, in_dt);
            }
            KirStmt::Q40PackFromThread { block, th_off, .. } => {
                infer_load_expr(block, in_dt);
                infer_load_expr(th_off, in_dt);
            }
            KirStmt::Q40DequantToTg {
                src_buf,
                tg_off,
                src_elem,
                ..
            } => {
                if (*src_buf as usize) < in_dt.len() {
                    in_dt[*src_buf as usize] = DType::Q40;
                }
                infer_load_expr(tg_off, in_dt);
                infer_load_expr(src_elem, in_dt);
            }
            KirStmt::TgStoreF4FromLoad {
                src_buf,
                tg_off,
                src_elem,
                dtype,
                ..
            } => {
                if (*src_buf as usize) < in_dt.len() {
                    in_dt[*src_buf as usize] = *dtype;
                }
                infer_load_expr(tg_off, in_dt);
                infer_load_expr(src_elem, in_dt);
            }
        }
    }
}

fn infer_load_expr(e: &KirExpr, in_dt: &mut [DType]) {
    match e {
        KirExpr::Load { buf, idx, dtype } => {
            if (*buf as usize) < in_dt.len() {
                in_dt[*buf as usize] = *dtype;
            }
            infer_load_expr(idx, in_dt);
        }
        KirExpr::VecMulSum {
            a_buf,
            a_idx,
            b_buf,
            b_idx,
            dtype,
            b_from_tg,
            ..
        } => {
            if (*a_buf as usize) < in_dt.len() {
                in_dt[*a_buf as usize] = *dtype;
            }
            // B is always activation float/half (never quantized packing).
            if b_from_tg.is_none() {
                if (*b_buf as usize) < in_dt.len() && dtype.is_float() {
                    in_dt[*b_buf as usize] = *dtype;
                }
            }
            infer_load_expr(a_idx, in_dt);
            infer_load_expr(b_idx, in_dt);
        }
        KirExpr::TgLoad { idx, .. } | KirExpr::ThreadLoad { idx, .. } => infer_load_expr(idx, in_dt),
        KirExpr::CastU32ToF32(e) => infer_load_expr(e, in_dt),
        KirExpr::CastF32ToU32(e) => infer_load_expr(e, in_dt),
        KirExpr::SimdSum(a) | KirExpr::Unary { a, .. } => infer_load_expr(a, in_dt),
        KirExpr::Q4kCoopFrag {
            w_buf,
            row_base,
            ib,
            lane,
            ..
        } => {
            if (*w_buf as usize) < in_dt.len() {
                in_dt[*w_buf as usize] = DType::Q4K;
            }
            infer_load_expr(row_base, in_dt);
            infer_load_expr(ib, in_dt);
            infer_load_expr(lane, in_dt);
        }
        KirExpr::Q6kCoopFrag {
            w_buf,
            row_base,
            ib,
            lane,
            b_from_tg,
            x_buf,
            ..
        } => {
            if (*w_buf as usize) < in_dt.len() {
                in_dt[*w_buf as usize] = DType::Q6K;
            }
            if b_from_tg.is_none() && (*x_buf as usize) < in_dt.len() {
                in_dt[*x_buf as usize] = DType::F16;
            }
            infer_load_expr(row_base, in_dt);
            infer_load_expr(ib, in_dt);
            infer_load_expr(lane, in_dt);
        }
        KirExpr::Bin { a, b, .. } | KirExpr::CmpGt { a, b } | KirExpr::CmpEq { a, b } => {
            infer_load_expr(a, in_dt);
            infer_load_expr(b, in_dt);
        }
        _ => {}
    }
}

fn buf_name(buf: u32, n_in: u32, n_out: u32) -> String {
    if buf < n_in {
        format!("in{buf}")
    } else if n_out <= 1 {
        "out".into()
    } else {
        format!("out{}", buf - n_in)
    }
}

fn walk_expr_q4(e: &KirExpr, buf: u32) -> bool {
    match e {
        KirExpr::Load {
            buf: b,
            dtype: DType::Q4K,
            ..
        } if *b == buf => true,
        KirExpr::Load { idx, .. } => walk_expr_q4(idx, buf),
        KirExpr::VecMulSum {
            a_buf,
            dtype: DType::Q4K,
            ..
        } if *a_buf == buf => true,
        KirExpr::VecMulSum {
            a_idx,
            b_idx,
            ..
        } => walk_expr_q4(a_idx, buf) || walk_expr_q4(b_idx, buf),
        KirExpr::TgLoad { idx, .. } | KirExpr::ThreadLoad { idx, .. } => walk_expr_q4(idx, buf),
        KirExpr::CastU32ToF32(e) => walk_expr_q4(e, buf),
        KirExpr::CastF32ToU32(e) => walk_expr_q4(e, buf),
        KirExpr::SimdSum(a) | KirExpr::Unary { a, .. } => walk_expr_q4(a, buf),
        KirExpr::Q4kCoopFrag {
            w_buf,
            row_base,
            ib,
            lane,
            ..
        } => {
            *w_buf == buf
                || walk_expr_q4(row_base, buf)
                || walk_expr_q4(ib, buf)
                || walk_expr_q4(lane, buf)
        }
        KirExpr::Q6kCoopFrag {
            w_buf,
            row_base,
            ib,
            x_off,
            lane,
            ..
        } => {
            // Q6 frag does not imply Q4 helpers; only track buffer uses for walk completeness.
            let _ = (w_buf, buf);
            walk_expr_q4(row_base, buf)
                || walk_expr_q4(ib, buf)
                || walk_expr_q4(x_off, buf)
                || walk_expr_q4(lane, buf)
        }
        KirExpr::Bin { a, b, .. }
        | KirExpr::CmpGt { a, b }
        | KirExpr::CmpEq { a, b } => walk_expr_q4(a, buf) || walk_expr_q4(b, buf),
        _ => false,
    }
}

fn buf_is_q4(stmts: &[KirStmt], buf: u32) -> bool {
    for s in stmts {
        match s {
            KirStmt::For { body, .. } | KirStmt::If { body, .. } => {
                if buf_is_q4(body, buf) {
                    return true;
                }
            }
            KirStmt::ForRange {
                limit_off,
                bound,
                step,
                body,
                ..
            } => {
                if walk_expr_q4(limit_off, buf)
                    || walk_expr_q4(bound, buf)
                    || walk_expr_q4(step, buf)
                    || buf_is_q4(body, buf)
                {
                    return true;
                }
            }
            KirStmt::Let { expr, .. }
            | KirStmt::LetU32 { expr, .. }
            | KirStmt::Assign { expr, .. } => {
                if walk_expr_q4(expr, buf) {
                    return true;
                }
            }
            KirStmt::Store { idx, val, .. }
            | KirStmt::TgStore { idx, val, .. }
            | KirStmt::ThreadStore { idx, val, .. } => {
                if walk_expr_q4(idx, buf) || walk_expr_q4(val, buf) {
                    return true;
                }
            }
            KirStmt::TgDeclF32 { .. }
            | KirStmt::ThreadDeclF32 { .. }
            | KirStmt::Barrier
            | KirStmt::ThreadgroupReduce { .. }
            | KirStmt::ThreadgroupArgmax { .. } => {}
            KirStmt::Q4kCoopNr4 {
                w_buf,
                row0_base,
                ib,
                lane,
                ..
            } => {
                if *w_buf == buf
                    || walk_expr_q4(row0_base, buf)
                    || walk_expr_q4(ib, buf)
                    || walk_expr_q4(lane, buf)
                {
                    return true;
                }
            }
            KirStmt::Q4kCoopNr4Dual {
                row0_base,
                ib,
                lane,
                ..
            } => {
                if buf <= 1
                    || walk_expr_q4(row0_base, buf)
                    || walk_expr_q4(ib, buf)
                    || walk_expr_q4(lane, buf)
                {
                    return true;
                }
            }
            KirStmt::Q6kCoopNr4 {
                row0_base,
                ib,
                x_off,
                lane,
                ..
            } => {
                if walk_expr_q4(row0_base, buf)
                    || walk_expr_q4(ib, buf)
                    || walk_expr_q4(x_off, buf)
                    || walk_expr_q4(lane, buf)
                {
                    return true;
                }
            }
            KirStmt::Q40PackBlock {
                block,
                src_elem,
                ..
            }
            | KirStmt::Q40DequantToTg {
                tg_off: block,
                src_elem,
                ..
            } => {
                if walk_expr_q4(block, buf) || walk_expr_q4(src_elem, buf) {
                    return true;
                }
            }
            KirStmt::Q40PackFromThread { block, th_off, .. } => {
                if walk_expr_q4(block, buf) || walk_expr_q4(th_off, buf) {
                    return true;
                }
            }
            KirStmt::TgStoreF4FromLoad { tg_off, src_elem, .. } => {
                if walk_expr_q4(tg_off, buf) || walk_expr_q4(src_elem, buf) {
                    return true;
                }
            }
        }
    }
    false
}

fn body_needs_q4(stmts: &[KirStmt]) -> bool {
    (0..8).any(|b| buf_is_q4(stmts, b))
}

fn walk_expr_q6(e: &KirExpr, buf: u32) -> bool {
    match e {
        KirExpr::Load {
            buf: b,
            dtype: DType::Q6K,
            ..
        } => *b == buf,
        KirExpr::Q6kCoopFrag {
            w_buf,
            row_base,
            ib,
            x_off,
            lane,
            ..
        } => {
            *w_buf == buf
                || walk_expr_q6(row_base, buf)
                || walk_expr_q6(ib, buf)
                || walk_expr_q6(x_off, buf)
                || walk_expr_q6(lane, buf)
        }
        KirExpr::Load { idx, .. }
        | KirExpr::TgLoad { idx, .. }
        | KirExpr::ThreadLoad { idx, .. }
        | KirExpr::CastU32ToF32(idx)
        | KirExpr::CastF32ToU32(idx)
        | KirExpr::SimdSum(idx)
        | KirExpr::Unary { a: idx, .. } => walk_expr_q6(idx, buf),
        KirExpr::VecMulSum {
            a_buf,
            a_idx,
            b_idx,
            dtype,
            ..
        } => {
            (*dtype == DType::Q6K && *a_buf == buf)
                || walk_expr_q6(a_idx, buf)
                || walk_expr_q6(b_idx, buf)
        }
        KirExpr::Q4kCoopFrag {
            row_base, ib, lane, ..
        } => {
            walk_expr_q6(row_base, buf) || walk_expr_q6(ib, buf) || walk_expr_q6(lane, buf)
        }
        KirExpr::Bin { a, b, .. } | KirExpr::CmpGt { a, b } | KirExpr::CmpEq { a, b } => {
            walk_expr_q6(a, buf) || walk_expr_q6(b, buf)
        }
        _ => false,
    }
}

fn buf_is_q6(stmts: &[KirStmt], buf: u32) -> bool {
    for s in stmts {
        match s {
            KirStmt::For { body, .. } | KirStmt::If { body, .. } => {
                if buf_is_q6(body, buf) {
                    return true;
                }
            }
            KirStmt::ForRange {
                limit_off,
                bound,
                step,
                body,
                ..
            } => {
                if walk_expr_q6(limit_off, buf)
                    || walk_expr_q6(bound, buf)
                    || walk_expr_q6(step, buf)
                    || buf_is_q6(body, buf)
                {
                    return true;
                }
            }
            KirStmt::Let { expr, .. }
            | KirStmt::LetU32 { expr, .. }
            | KirStmt::Assign { expr, .. } => {
                if walk_expr_q6(expr, buf) {
                    return true;
                }
            }
            KirStmt::Store { idx, val, .. }
            | KirStmt::TgStore { idx, val, .. }
            | KirStmt::ThreadStore { idx, val, .. } => {
                if walk_expr_q6(idx, buf) || walk_expr_q6(val, buf) {
                    return true;
                }
            }
            KirStmt::TgDeclF32 { .. }
            | KirStmt::ThreadDeclF32 { .. }
            | KirStmt::Barrier
            | KirStmt::ThreadgroupReduce { .. }
            | KirStmt::ThreadgroupArgmax { .. } => {}
            KirStmt::Q4kCoopNr4 {
                row0_base, ib, lane, ..
            }
            | KirStmt::Q4kCoopNr4Dual {
                row0_base, ib, lane, ..
            } => {
                if walk_expr_q6(row0_base, buf)
                    || walk_expr_q6(ib, buf)
                    || walk_expr_q6(lane, buf)
                {
                    return true;
                }
            }
            KirStmt::Q6kCoopNr4 {
                w_buf,
                row0_base,
                ib,
                x_off,
                lane,
                ..
            } => {
                if *w_buf == buf
                    || walk_expr_q6(row0_base, buf)
                    || walk_expr_q6(ib, buf)
                    || walk_expr_q6(x_off, buf)
                    || walk_expr_q6(lane, buf)
                {
                    return true;
                }
            }
            KirStmt::Q40PackBlock {
                block,
                src_elem,
                ..
            }
            | KirStmt::Q40DequantToTg {
                tg_off: block,
                src_elem,
                ..
            } => {
                if walk_expr_q6(block, buf) || walk_expr_q6(src_elem, buf) {
                    return true;
                }
            }
            KirStmt::Q40PackFromThread { block, th_off, .. } => {
                if walk_expr_q6(block, buf) || walk_expr_q6(th_off, buf) {
                    return true;
                }
            }
            KirStmt::TgStoreF4FromLoad { tg_off, src_elem, .. } => {
                if walk_expr_q6(tg_off, buf) || walk_expr_q6(src_elem, buf) {
                    return true;
                }
            }
        }
    }
    false
}

fn body_needs_q6(stmts: &[KirStmt]) -> bool {
    (0..8).any(|b| buf_is_q6(stmts, b))
}

fn body_needs_q40(stmts: &[KirStmt]) -> bool {
    fn walk(stmts: &[KirStmt]) -> bool {
        for s in stmts {
            match s {
                KirStmt::For { body, .. }
                | KirStmt::If { body, .. }
                | KirStmt::ForRange { body, .. } => {
                    if walk(body) {
                        return true;
                    }
                }
                KirStmt::Q40PackBlock { .. } | KirStmt::Q40PackFromThread { .. } | KirStmt::Q40DequantToTg { .. } | KirStmt::TgStoreF4FromLoad { dtype: DType::Q40, .. } => return true,
                KirStmt::Let { expr, .. }
                | KirStmt::LetU32 { expr, .. }
                | KirStmt::Assign { expr, .. } => {
                    if expr_needs_q40(expr) {
                        return true;
                    }
                }
                KirStmt::Store { idx, val, .. }
                | KirStmt::TgStore { idx, val, .. }
                | KirStmt::ThreadStore { idx, val, .. } => {
                    if expr_needs_q40(idx) || expr_needs_q40(val) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    fn expr_needs_q40(e: &KirExpr) -> bool {
        match e {
            KirExpr::Load {
                dtype: DType::Q40, ..
            }
            | KirExpr::VecMulSum {
                dtype: DType::Q40, ..
            } => true,
            KirExpr::Load { idx, .. }
            | KirExpr::TgLoad { idx, .. }
            | KirExpr::ThreadLoad { idx, .. }
            | KirExpr::CastU32ToF32(idx)
            | KirExpr::CastF32ToU32(idx)
            | KirExpr::SimdSum(idx)
            | KirExpr::Unary { a: idx, .. } => expr_needs_q40(idx),
            KirExpr::VecMulSum { a_idx, b_idx, .. } => {
                expr_needs_q40(a_idx) || expr_needs_q40(b_idx)
            }
            KirExpr::Bin { a, b, .. } | KirExpr::CmpGt { a, b } | KirExpr::CmpEq { a, b } => {
                expr_needs_q40(a) || expr_needs_q40(b)
            }
            KirExpr::Q4kCoopFrag {
                row_base, ib, lane, ..
            } => {
                expr_needs_q40(row_base) || expr_needs_q40(ib) || expr_needs_q40(lane)
            }
            KirExpr::Q6kCoopFrag {
                row_base,
                ib,
                x_off,
                lane,
                ..
            } => {
                expr_needs_q40(row_base)
                    || expr_needs_q40(ib)
                    || expr_needs_q40(x_off)
                    || expr_needs_q40(lane)
            }
            _ => false,
        }
    }
    walk(stmts)
}

fn emit_stmts(
    stmts: &[KirStmt],
    indent: usize,
    n_in: u32,
    n_out: u32,
    elem: DType,
) -> Result<String, CodegenError> {
    let pad = "  ".repeat(indent);
    let mut s = String::new();
    for st in stmts {
        match st {
            KirStmt::For { id, n, body } => {
                s.push_str(&format!(
                    "{pad}for (uint f{id} = 0u; f{id} < {n}u; f{id}++) {{\n"
                ));
                s.push_str(&emit_stmts(body, indent + 1, n_in, n_out, elem)?);
                s.push_str(&format!("{pad}}}\n"));
            }
            KirStmt::ForRange {
                id,
                limit_off,
                bound,
                step,
                body,
            } => {
                let lo = emit_expr_ty(limit_off, n_in, n_out, false, elem)?;
                let bd = emit_expr_ty(bound, n_in, n_out, false, elem)?;
                let sp = emit_expr_ty(step, n_in, n_out, false, elem)?;
                s.push_str(&format!(
                    "{pad}for (; u{id} + ({lo}) < ({bd}); u{id} += ({sp})) {{\n"
                ));
                s.push_str(&emit_stmts(body, indent + 1, n_in, n_out, elem)?);
                s.push_str(&format!("{pad}}}\n"));
            }
            KirStmt::Let { id, expr } => {
                s.push_str(&format!(
                    "{pad}float v{id} = {};\n",
                    emit_expr_ty(expr, n_in, n_out, true, elem)?
                ));
            }
            KirStmt::LetU32 { id, expr } => {
                s.push_str(&format!(
                    "{pad}uint u{id} = {};\n",
                    emit_expr_ty(expr, n_in, n_out, false, elem)?
                ));
            }
            KirStmt::Assign { id, expr } => {
                s.push_str(&format!(
                    "{pad}v{id} = {};\n",
                    emit_expr_ty(expr, n_in, n_out, true, elem)?
                ));
            }
            KirStmt::Store { buf, idx, val } => {
                let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
                let idx_u = if is_uintish(idx) {
                    idx_s
                } else {
                    format!("uint({idx_s})")
                };
                let vs = emit_expr_ty(val, n_in, n_out, true, elem)?;
                let store = match elem {
                    DType::F16 => format!("half({vs})"),
                    _ => vs,
                };
                s.push_str(&format!(
                    "{pad}{}[{}] = {};\n",
                    buf_name(*buf, n_in, n_out),
                    idx_u,
                    store
                ));
            }
            KirStmt::TgDeclF32 { id, n } => {
                s.push_str(&format!("{pad}threadgroup float tg{id}[{n}];\n"));
            }
            KirStmt::TgStore { id, idx, val } => {
                let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
                let idx_u = if is_uintish(idx) {
                    idx_s
                } else {
                    format!("uint({idx_s})")
                };
                let vs = emit_expr_ty(val, n_in, n_out, true, elem)?;
                s.push_str(&format!("{pad}tg{id}[{idx_u}] = {vs};\n"));
            }
            KirStmt::ThreadDeclF32 { id, n } => {
                s.push_str(&format!("{pad}thread float th{id}[{n}];\n"));
            }
            KirStmt::ThreadStore { id, idx, val } => {
                let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
                let idx_u = if is_uintish(idx) {
                    idx_s
                } else {
                    format!("uint({idx_s})")
                };
                let vs = emit_expr_ty(val, n_in, n_out, true, elem)?;
                s.push_str(&format!("{pad}th{id}[{idx_u}] = {vs};\n"));
            }
            KirStmt::Barrier => {
                s.push_str(&format!(
                    "{pad}threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                ));
            }
            KirStmt::If { cond, body } => {
                s.push_str(&format!(
                    "{pad}if ({}) {{\n",
                    emit_expr_ty(cond, n_in, n_out, true, elem)?
                ));
                s.push_str(&emit_stmts(body, indent + 1, n_in, n_out, elem)?);
                s.push_str(&format!("{pad}}}\n"));
            }
            KirStmt::ThreadgroupReduce { acc_id, tg } => {
                if *tg <= 32 {
                    s.push_str(&format!("{pad}v{acc_id} = simd_sum(v{acc_id});\n"));
                } else {
                    // Hierarchical: simd_sum per SIMD-group, then short sum over ≤tg/32 partials.
                    let nsg = (*tg + 31) / 32;
                    s.push_str(&format!(
                        "{pad}v{acc_id} = simd_sum(v{acc_id});\n\
                         {pad}threadgroup float red_{acc_id}[{nsg}];\n\
                         {pad}if ((lid & 31u) == 0u) red_{acc_id}[lid / 32u] = v{acc_id};\n\
                         {pad}threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}if (lid == 0u) {{\n\
                         {pad}  float s = red_{acc_id}[0];\n"
                    ));
                    for i in 1..nsg {
                        s.push_str(&format!("{pad}  s += red_{acc_id}[{i}];\n"));
                    }
                    s.push_str(&format!(
                        "{pad}  red_{acc_id}[0] = s;\n\
                         {pad}}}\n\
                         {pad}threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}v{acc_id} = red_{acc_id}[0];\n"
                    ));
                }
            }
            KirStmt::ThreadgroupArgmax { val_id, idx_id, tg } => {
                // Butterfly max within SIMD-group, then ≤tg/32 partial merge.
                s.push_str(&format!(
                    "{pad}{{\n\
                     {pad}  float am_v = v{val_id};\n\
                     {pad}  uint am_i = uint(v{idx_id});\n\
                     {pad}  for (uint am_s = 16u; am_s > 0u; am_s >>= 1u) {{\n\
                     {pad}    float ov = simd_shuffle_xor(am_v, am_s);\n\
                     {pad}    uint oi = simd_shuffle_xor(am_i, am_s);\n\
                     {pad}    if (ov > am_v || (ov == am_v && oi < am_i)) {{ am_v = ov; am_i = oi; }}\n\
                     {pad}  }}\n"
                ));
                if *tg <= 32 {
                    s.push_str(&format!(
                        "{pad}  v{val_id} = am_v;\n\
                         {pad}  v{idx_id} = float(am_i);\n\
                         {pad}}}\n"
                    ));
                } else {
                    let nsg = (*tg + 31) / 32;
                    s.push_str(&format!(
                        "{pad}  threadgroup float am_rv[{nsg}];\n\
                         {pad}  threadgroup float am_ri[{nsg}];\n\
                         {pad}  if ((lid & 31u) == 0u) {{ am_rv[lid / 32u] = am_v; am_ri[lid / 32u] = float(am_i); }}\n\
                         {pad}  threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}  if (lid == 0u) {{\n\
                         {pad}    float bv = am_rv[0]; uint bi = uint(am_ri[0]);\n"
                    ));
                    for i in 1..nsg {
                        s.push_str(&format!(
                            "{pad}    {{ float ov = am_rv[{i}]; uint oi = uint(am_ri[{i}]);\n\
                             {pad}      if (ov > bv || (ov == bv && oi < bi)) {{ bv = ov; bi = oi; }} }}\n"
                        ));
                    }
                    s.push_str(&format!(
                        "{pad}    am_rv[0] = bv; am_ri[0] = float(bi);\n\
                         {pad}  }}\n\
                         {pad}  threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}  v{val_id} = am_rv[0];\n\
                         {pad}  v{idx_id} = am_ri[0];\n\
                         {pad}}}\n"
                    ));
                }
            }
            KirStmt::Q4kCoopNr4 {
                w_buf,
                row0_base,
                cols,
                ib,
                b_from_tg,
                x_buf,
                lane,
                acc_ids,
            } => {
                let a = buf_name(*w_buf, n_in, n_out);
                let rb = emit_expr_ty(row0_base, n_in, n_out, false, elem)?;
                let ib_s = emit_expr_ty(ib, n_in, n_out, false, elem)?;
                let lane_s = emit_expr_ty(lane, n_in, n_out, false, elem)?;
                let call = if let Some(tg) = b_from_tg {
                    format!(
                        "ksearch_q4k_coop_frag_nr4({a}, ({rb}), {cols}u, ({ib_s}), tg{tg}, ({lane_s}), &v{}, &v{}, &v{}, &v{})",
                        acc_ids[0], acc_ids[1], acc_ids[2], acc_ids[3]
                    )
                } else {
                    let xb = buf_name(*x_buf, n_in, n_out);
                    format!(
                        "ksearch_q4k_coop_frag_nr4_dev({a}, ({rb}), {cols}u, ({ib_s}), {xb}, ({lane_s}), &v{}, &v{}, &v{}, &v{})",
                        acc_ids[0], acc_ids[1], acc_ids[2], acc_ids[3]
                    )
                };
                s.push_str(&format!("{pad}{call};\n"));
            }
            KirStmt::Q4kCoopNr4Dual {
                row0_base,
                cols,
                ib,
                b_from_tg,
                x_buf,
                lane,
                acc_g,
                acc_u,
            } => {
                let rb = emit_expr_ty(row0_base, n_in, n_out, false, elem)?;
                let ib_s = emit_expr_ty(ib, n_in, n_out, false, elem)?;
                let lane_s = emit_expr_ty(lane, n_in, n_out, false, elem)?;
                let call = if let Some(tg) = b_from_tg {
                    format!(
                        "ksearch_q4k_coop_frag_nr4_dual(in0, in1, ({rb}), {cols}u, ({ib_s}), tg{tg}, ({lane_s}), &v{}, &v{}, &v{}, &v{}, &v{}, &v{}, &v{}, &v{})",
                        acc_g[0], acc_g[1], acc_g[2], acc_g[3],
                        acc_u[0], acc_u[1], acc_u[2], acc_u[3]
                    )
                } else {
                    let xb = buf_name(*x_buf, n_in, n_out);
                    format!(
                        "ksearch_q4k_coop_frag_nr4_dual_dev(in0, in1, ({rb}), {cols}u, ({ib_s}), {xb}, ({lane_s}), &v{}, &v{}, &v{}, &v{}, &v{}, &v{}, &v{}, &v{})",
                        acc_g[0], acc_g[1], acc_g[2], acc_g[3],
                        acc_u[0], acc_u[1], acc_u[2], acc_u[3]
                    )
                };
                s.push_str(&format!("{pad}{call};\n"));
            }
            KirStmt::Q6kCoopNr4 {
                w_buf,
                row0_base,
                cols,
                ib,
                b_from_tg,
                x_buf,
                x_off,
                lane,
                acc_ids,
            } => {
                let a = buf_name(*w_buf, n_in, n_out);
                let rb = emit_expr_ty(row0_base, n_in, n_out, false, elem)?;
                let ib_s = emit_expr_ty(ib, n_in, n_out, false, elem)?;
                let lane_s = emit_expr_ty(lane, n_in, n_out, false, elem)?;
                let call = if let Some(tg) = b_from_tg {
                    let xo = emit_expr_ty(x_off, n_in, n_out, false, elem)?;
                    format!(
                        "ksearch_q6k_coop_frag_nr4({a}, ({rb}), {cols}u, ({ib_s}), tg{tg}, ({xo}), ({lane_s}), &v{}, &v{}, &v{}, &v{})",
                        acc_ids[0], acc_ids[1], acc_ids[2], acc_ids[3]
                    )
                } else {
                    let xb = buf_name(*x_buf, n_in, n_out);
                    format!(
                        "ksearch_q6k_coop_frag_nr4_dev({a}, ({rb}), {cols}u, ({ib_s}), {xb}, ({lane_s}), &v{}, &v{}, &v{}, &v{})",
                        acc_ids[0], acc_ids[1], acc_ids[2], acc_ids[3]
                    )
                };
                s.push_str(&format!("{pad}{call};\n"));
            }
            KirStmt::Q40PackBlock {
                dst_buf,
                block,
                src_buf,
                src_elem,
            } => {
                let dst = buf_name(*dst_buf, n_in, n_out);
                let src = buf_name(*src_buf, n_in, n_out);
                let blk = emit_expr_ty(block, n_in, n_out, false, elem)?;
                let se = emit_expr_ty(src_elem, n_in, n_out, false, elem)?;
                s.push_str(&format!(
                    "{pad}ksearch_pack_q40({dst}, ({blk}), {src}, ({se}));\n"
                ));
            }
            KirStmt::Q40PackFromThread {
                dst_buf,
                block,
                th_id,
                th_off,
            } => {
                let dst = buf_name(*dst_buf, n_in, n_out);
                let blk = emit_expr_ty(block, n_in, n_out, false, elem)?;
                let off = emit_expr_ty(th_off, n_in, n_out, false, elem)?;
                s.push_str(&format!(
                    "{pad}ksearch_pack_q40_th({dst}, ({blk}), th{th_id}, ({off}));\n"
                ));
            }
            KirStmt::Q40DequantToTg {
                tg_id,
                tg_off,
                src_buf,
                src_elem,
            } => {
                let src = buf_name(*src_buf, n_in, n_out);
                let off = emit_expr_ty(tg_off, n_in, n_out, false, elem)?;
                let se = emit_expr_ty(src_elem, n_in, n_out, false, elem)?;
                s.push_str(&format!(
                    "{pad}ksearch_dequant_q40_to_tg(tg{tg_id} + ({off}), {src}, ({se}));\n"
                ));
            }
            KirStmt::TgStoreF4FromLoad {
                tg_id,
                tg_off,
                src_buf,
                src_elem,
                dtype,
            } => {
                let src = buf_name(*src_buf, n_in, n_out);
                let off = emit_expr_ty(tg_off, n_in, n_out, false, elem)?;
                let se = emit_expr_ty(src_elem, n_in, n_out, false, elem)?;
                let load = match dtype {
                    DType::Q40 => format!("ksearch_load_q404({src}, ({se}))"),
                    DType::F16 => format!("float4(*(device const half4*)({src} + ({se})))"),
                    _ => format!("float4(0.0f)"),
                };
                s.push_str(&format!(
                    "{pad}{{ float4 _t4 = {load}; *(threadgroup float4*)(tg{tg_id} + ({off})) = _t4; }}\n"
                ));
            }
        }
    }
    Ok(s)
}

fn emit_expr_ty(
    e: &KirExpr,
    n_in: u32,
    n_out: u32,
    as_float: bool,
    elem: DType,
) -> Result<String, CodegenError> {
    Ok(match e {
        KirExpr::ConstF32(f) => format!("{f:?}f"),
        KirExpr::ConstU32(u) => {
            if as_float {
                format!("{u}.0f")
            } else {
                format!("{u}u")
            }
        }
        KirExpr::Gid => {
            if as_float {
                "float(gid)".into()
            } else {
                "gid".into()
            }
        }
        KirExpr::Lid => {
            if as_float {
                "float(lid)".into()
            } else {
                "lid".into()
            }
        }
        KirExpr::ForVar(id) => {
            if as_float {
                format!("float(f{id})")
            } else {
                format!("f{id}")
            }
        }
        KirExpr::Var(id) => format!("v{id}"),
        KirExpr::UVar(id) => {
            if as_float {
                format!("float(u{id})")
            } else {
                format!("u{id}")
            }
        }
        KirExpr::Load { buf, idx, dtype } => {
            let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
            let idx_u = if is_uintish(idx) {
                idx_s
            } else {
                format!("uint({idx_s})")
            };
            let load = format!("{}[{}]", buf_name(*buf, n_in, n_out), idx_u);
            match dtype {
                DType::F32 => load,
                DType::F16 => format!("float({load})"),
                DType::Q4K => format!(
                    "ksearch_load_q4k({}, {})",
                    buf_name(*buf, n_in, n_out),
                    idx_u
                ),
                DType::Q6K => format!(
                    "ksearch_load_q6k({}, {})",
                    buf_name(*buf, n_in, n_out),
                    idx_u
                ),
                DType::Q40 => format!(
                    "ksearch_load_q40({}, {})",
                    buf_name(*buf, n_in, n_out),
                    idx_u
                ),
                other => {
                    return Err(CodegenError::Msg(format!(
                        "render load: unsupported dtype {other:?}"
                    )))
                }
            }
        }
        KirExpr::TgLoad { id, idx } => {
            let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
            let idx_u = if is_uintish(idx) {
                idx_s
            } else {
                format!("uint({idx_s})")
            };
            format!("tg{id}[{idx_u}]")
        }
        KirExpr::ThreadLoad { id, idx } => {
            let idx_s = emit_expr_ty(idx, n_in, n_out, false, elem)?;
            let idx_u = if is_uintish(idx) {
                idx_s
            } else {
                format!("uint({idx_s})")
            };
            format!("th{id}[{idx_u}]")
        }
        KirExpr::VecMulSum {
            a_buf,
            a_idx,
            b_buf,
            b_idx,
            width,
            dtype,
            b_from_tg,
        } => {
            let ai = emit_expr_ty(a_idx, n_in, n_out, false, elem)?;
            let bi = emit_expr_ty(b_idx, n_in, n_out, false, elem)?;
            let a = buf_name(*a_buf, n_in, n_out);
            if let Some(tg_id) = b_from_tg {
                // A from device (dtype); B from threadgroup float.
                match width {
                    1 if *dtype == DType::Q4K => format!(
                        "(ksearch_load_q4k({a}, {ai}) * tg{tg_id}[{bi}])"
                    ),
                    1 if *dtype == DType::Q6K => format!(
                        "(ksearch_load_q6k({a}, {ai}) * tg{tg_id}[{bi}])"
                    ),
                    1 if *dtype == DType::Q40 => format!(
                        "(ksearch_load_q40({a}, {ai}) * tg{tg_id}[{bi}])"
                    ),
                    2 if *dtype == DType::Q4K => format!(
                        "dot(ksearch_load_q4k2({a}, ({ai})), *(threadgroup const float2*)(tg{tg_id} + ({bi})))"
                    ),
                    4 if *dtype == DType::Q4K => format!(
                        "dot(ksearch_load_q4k4({a}, ({ai})), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    4 if *dtype == DType::Q40 => format!(
                        "dot(ksearch_load_q404({a}, ({ai})), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    32 if *dtype == DType::Q4K => format!(
                        "ksearch_dot_q4k32({a}, ({ai}), tg{tg_id}, ({bi}))"
                    ),
                    256 if *dtype == DType::Q4K => format!(
                        "ksearch_dot_q4k256({a}, ({ai}), tg{tg_id}, ({bi}))"
                    ),
                    4 if *dtype == DType::F16 => format!(
                        "dot(float4(*(device const half4*)({a} + ({ai}))), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    2 if *dtype == DType::F16 => format!(
                        "(float((*(device const half2*)({a} + ({ai}))).x) * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).x + \
                         float((*(device const half2*)({a} + ({ai}))).y) * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).y)"
                    ),
                    1 if *dtype == DType::F16 => {
                        format!("(float({a}[{ai}]) * tg{tg_id}[{bi}])")
                    },
                    4 => format!(
                        "dot(*(device const float4*)({a} + ({ai})), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    2 => format!(
                        "((*(device const float2*)({a} + ({ai}))).x * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).x + \
                         (*(device const float2*)({a} + ({ai}))).y * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).y)"
                    ),
                    1 => format!("{a}[{ai}] * tg{tg_id}[{bi}]"),
                    w => {
                        eprintln!(
                            "[codegen] VecMulSum unsupported width={w} dtype={dtype:?} tg={b_from_tg:?}"
                        );
                        return Err(CodegenError::Msg(format!(
                            "render VecMulSum: unsupported width {w} dtype={dtype:?} tg={b_from_tg:?}"
                        )))
                    }
                }
            } else {
                let b = buf_name(*b_buf, n_in, n_out);
                let (vn, cast) = match dtype {
                    DType::F16 => ("half", "float"),
                    _ => ("float", ""),
                };
                match width {
                    1 if *dtype == DType::Q4K => format!(
                        "(ksearch_load_q4k({a}, {ai}) * float({b}[{bi}]))"
                    ),
                    1 if *dtype == DType::Q6K => format!(
                        "(ksearch_load_q6k({a}, {ai}) * float({b}[{bi}]))"
                    ),
                    1 if *dtype == DType::Q40 => format!(
                        "(ksearch_load_q40({a}, {ai}) * float({b}[{bi}]))"
                    ),
                    2 if *dtype == DType::Q4K => format!(
                        "dot(ksearch_load_q4k2({a}, ({ai})), float2(float({b}[({bi})]), float({b}[({bi}) + 1u])))"
                    ),
                    4 if *dtype == DType::Q4K => format!(
                        "dot(ksearch_load_q4k4({a}, ({ai})), float4(float({b}[({bi})]), float({b}[({bi}) + 1u]), float({b}[({bi}) + 2u]), float({b}[({bi}) + 3u])))"
                    ),
                    4 if *dtype == DType::Q40 => format!(
                        "dot(ksearch_load_q404({a}, ({ai})), float4(float({b}[({bi})]), float({b}[({bi}) + 1u]), float({b}[({bi}) + 2u]), float({b}[({bi}) + 3u])))"
                    ),
                    32 if *dtype == DType::Q4K => format!(
                        "ksearch_dot_q4k32_dev({a}, ({ai}), {b}, ({bi}))"
                    ),
                    256 if *dtype == DType::Q4K => format!(
                        "ksearch_dot_q4k256_dev({a}, ({ai}), {b}, ({bi}))"
                    ),
                    4 if *dtype == DType::F16 => format!(
                        "dot(float4(*(device const half4*)({a} + ({ai}))), float4(*(device const half4*)({b} + ({bi}))))"
                    ),
                    2 if *dtype == DType::F16 => format!(
                        "(float((*(device const half2*)({a} + ({ai}))).x) * float((*(device const half2*)({b} + ({bi}))).x) + \
                         float((*(device const half2*)({a} + ({ai}))).y) * float((*(device const half2*)({b} + ({bi}))).y))"
                    ),
                    1 if *dtype == DType::F16 => {
                        format!("(float({a}[{ai}]) * float({b}[{bi}]))")
                    }
                    4 => format!(
                        "dot(*(device const float4*)({a} + ({ai})), *(device const float4*)({b} + ({bi})))"
                    ),
                    2 => format!(
                        "((*(device const float2*)({a} + ({ai}))).x * (*(device const float2*)({b} + ({bi}))).x + \
                         (*(device const float2*)({a} + ({ai}))).y * (*(device const float2*)({b} + ({bi}))).y)"
                    ),
                    1 => format!("{a}[{ai}] * {b}[{bi}]"),
                    w => {
                        let _ = (vn, cast);
                        eprintln!(
                            "[codegen] VecMulSum unsupported width={w} dtype={dtype:?} (device)"
                        );
                        return Err(CodegenError::Msg(format!(
                            "render VecMulSum: unsupported width {w} dtype={dtype:?} (device)"
                        )))
                    }
                }
            }
        }
        KirExpr::SimdSum(a) => format!("simd_sum({})", emit_expr_ty(a, n_in, n_out, true, elem)?),
        KirExpr::Q4kCoopFrag {
            w_buf,
            row_base,
            cols,
            ib,
            b_from_tg,
            x_buf,
            lane,
        } => {
            let a = buf_name(*w_buf, n_in, n_out);
            let rb = emit_expr_ty(row_base, n_in, n_out, false, elem)?;
            let ib_s = emit_expr_ty(ib, n_in, n_out, false, elem)?;
            let lane_s = emit_expr_ty(lane, n_in, n_out, false, elem)?;
            if let Some(tg) = b_from_tg {
                format!(
                    "ksearch_q4k_coop_frag({a}, ({rb}), {cols}u, ({ib_s}), tg{tg}, ({lane_s}))"
                )
            } else {
                let xb = buf_name(*x_buf, n_in, n_out);
                format!(
                    "ksearch_q4k_coop_frag_dev({a}, ({rb}), {cols}u, ({ib_s}), {xb}, ({lane_s}))"
                )
            }
        }
        KirExpr::Q6kCoopFrag {
            w_buf,
            row_base,
            cols,
            ib,
            b_from_tg,
            x_buf,
            x_off,
            lane,
        } => {
            let a = buf_name(*w_buf, n_in, n_out);
            let rb = emit_expr_ty(row_base, n_in, n_out, false, elem)?;
            let ib_s = emit_expr_ty(ib, n_in, n_out, false, elem)?;
            let lane_s = emit_expr_ty(lane, n_in, n_out, false, elem)?;
            if let Some(tg) = b_from_tg {
                let xo = emit_expr_ty(x_off, n_in, n_out, false, elem)?;
                format!(
                    "ksearch_q6k_coop_frag({a}, ({rb}), {cols}u, ({ib_s}), tg{tg}, ({xo}), ({lane_s}))"
                )
            } else {
                let xb = buf_name(*x_buf, n_in, n_out);
                format!(
                    "ksearch_q6k_coop_frag_dev({a}, ({rb}), {cols}u, ({ib_s}), {xb}, ({lane_s}))"
                )
            }
        }
        KirExpr::Bin { op, a, b } => {
            if matches!(
                op,
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Max | BinOp::Min
            ) && is_uintish(a)
                && is_uintish(b)
                && !as_float
            {
                let as_ = emit_expr_ty(a, n_in, n_out, false, elem)?;
                let bs = emit_expr_ty(b, n_in, n_out, false, elem)?;
                match op {
                    BinOp::Add => format!("({as_} + {bs})"),
                    BinOp::Sub => format!("({as_} - {bs})"),
                    BinOp::Mul => format!("({as_} * {bs})"),
                    BinOp::Div => format!("({as_} / {bs})"),
                    BinOp::Max => format!("max({as_}, {bs})"),
                    BinOp::Min => format!("min({as_}, {bs})"),
                }
            } else {
                let (as_, bs) = (
                    emit_expr_ty(a, n_in, n_out, true, elem)?,
                    emit_expr_ty(b, n_in, n_out, true, elem)?,
                );
                match op {
                    BinOp::Add => format!("({as_} + {bs})"),
                    BinOp::Sub => format!("({as_} - {bs})"),
                    BinOp::Mul => format!("({as_} * {bs})"),
                    BinOp::Div => format!("({as_} / {bs})"),
                    BinOp::Max => format!("max({as_}, {bs})"),
                    BinOp::Min => format!("min({as_}, {bs})"),
                }
            }
        }
        KirExpr::Unary { op, a } => {
            let x = emit_expr_ty(a, n_in, n_out, true, elem)?;
            match op {
                UnaryOp::Neg => format!("(-{x})"),
                UnaryOp::Exp => format!("exp({x})"),
                UnaryOp::Tanh => format!("precise::tanh({x})"),
                UnaryOp::Rsqrt => format!("rsqrt({x})"),
                UnaryOp::Sqrt => format!("sqrt({x})"),
                UnaryOp::Floor => format!("floor({x})"),
            }
        }
        KirExpr::CmpGt { a, b } => {
            if is_uintish(a) && is_uintish(b) {
                format!(
                    "(({}) > ({}))",
                    emit_expr_ty(a, n_in, n_out, false, elem)?,
                    emit_expr_ty(b, n_in, n_out, false, elem)?
                )
            } else {
                format!(
                    "(({}) > ({}))",
                    emit_expr_ty(a, n_in, n_out, true, elem)?,
                    emit_expr_ty(b, n_in, n_out, true, elem)?
                )
            }
        }
        KirExpr::CmpEq { a, b } => {
            if is_uintish(a) && is_uintish(b) {
                format!(
                    "(({}) == ({}))",
                    emit_expr_ty(a, n_in, n_out, false, elem)?,
                    emit_expr_ty(b, n_in, n_out, false, elem)?
                )
            } else {
                format!(
                    "(({}) == ({}))",
                    emit_expr_ty(a, n_in, n_out, true, elem)?,
                    emit_expr_ty(b, n_in, n_out, true, elem)?
                )
            }
        }
        KirExpr::CastU32ToF32(e) => {
            format!("float({})", emit_expr_ty(e, n_in, n_out, false, elem)?)
        }
        KirExpr::CastF32ToU32(e) => {
            format!("uint({})", emit_expr_ty(e, n_in, n_out, true, elem)?)
        }
    })
}

fn is_uintish(e: &KirExpr) -> bool {
    match e {
        KirExpr::Gid
        | KirExpr::Lid
        | KirExpr::ForVar(_)
        | KirExpr::ConstU32(_)
        | KirExpr::UVar(_)
        | KirExpr::CastF32ToU32(_) => true,
        KirExpr::Bin {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Max | BinOp::Min,
            a,
            b,
        } => is_uintish(a) && is_uintish(b),
        _ => false,
    }
}

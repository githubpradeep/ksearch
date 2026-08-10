//! Render MSL from Kernel IR + OptSchedule (not hand GEMV template matching).

use crate::{CodegenError, LaunchHint, MetalKernelSource};
use ksearch_ir::{DType, ElemExpr, KirBody, KernelIr, OptSchedule};

pub fn render_msl(kir: &KernelIr, sched: OptSchedule) -> Result<MetalKernelSource, CodegenError> {
    match &kir.body {
        KirBody::Elementwise { n, expr } => render_elementwise(kir, *n, expr),
        KirBody::Matvec {
            rows,
            cols,
            weight_dtype,
        } => match weight_dtype {
            DType::F32 => render_matvec_f32(kir, *rows, *cols, sched),
            DType::Q4K => render_matvec_q4k(kir, *rows, *cols, sched),
            other => Err(CodegenError::Msg(format!(
                "matvec render: unsupported weight dtype {other:?}"
            ))),
        },
        KirBody::SumLast { rows, cols } => render_sum_last(kir, *rows, *cols, sched),
        KirBody::SdpaNaive { n_q, hd, max_t } => render_sdpa_naive(kir, *n_q, *hd, *max_t),
        KirBody::RmsNorm { n, eps } => render_rmsnorm(kir, *n, *eps),
        KirBody::RmsNormAdd { n, eps } => render_rmsnorm_add(kir, *n, *eps),
        KirBody::RmsNormAddScale { n, eps, scale } => {
            render_rmsnorm_add_scale(kir, *n, *eps, *scale)
        }
        KirBody::RmsNormPerHead {
            n_heads,
            hd,
            eps,
            with_weight,
        } => render_rmsnorm_per_head(kir, *n_heads, *hd, *eps, *with_weight),
        KirBody::Rope { n_heads, hd } => render_rope(kir, *n_heads, *hd),
        KirBody::GeluMul { n, up_off } => render_gelu_mul(kir, *n, *up_off),
        KirBody::CopySlice {
            src_off,
            dst_off,
            n,
        } => render_copy_slice(kir, *src_off, *dst_off, *n),
        KirBody::SoftcapArgmax { n, cap } => render_softcap_argmax(kir, *n, *cap),
    }
}

fn render_elementwise(
    kir: &KernelIr,
    n: usize,
    expr: &ElemExpr,
) -> Result<MetalKernelSource, CodegenError> {
    let ty = kir.out_dtype.msl();
    let mut params = String::new();
    for i in 0..kir.n_inputs {
        params.push_str(&format!(
            "  device const {ty}* in{i} [[buffer({i})]],\n"
        ));
    }
    params.push_str(&format!(
        "  device {ty}* out [[buffer({})]],\n",
        kir.n_inputs
    ));
    params.push_str("  uint gid [[thread_position_in_grid]]");
    let body = emit_elem_expr(expr);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params}
) {{
  if (gid >= {n}u) return;
  out[gid] = {body};
}}
"#,
        name = kir.name,
        params = params,
        n = n,
        body = body,
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: kir.n_inputs,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn emit_elem_expr(expr: &ElemExpr) -> String {
    match expr {
        ElemExpr::Load(bi) => format!("in{bi}[gid]"),
        ElemExpr::Add(a, b) => format!("({} + {})", emit_elem_expr(a), emit_elem_expr(b)),
        ElemExpr::Mul(a, b) => format!("({} * {})", emit_elem_expr(a), emit_elem_expr(b)),
        ElemExpr::Scale(inner, s) => format!("({s:?}f * {})", emit_elem_expr(inner)),
    }
}

fn render_matvec_f32(
    kir: &KernelIr,
    rows: usize,
    cols: usize,
    sched: OptSchedule,
) -> Result<MetalKernelSource, CodegenError> {
    let ty = kir.out_dtype.msl();
    let tg = sched.tg;
    let vec = sched.vec;
    let unroll = sched.unroll;
    if !matches!(vec, 1 | 2 | 4) {
        return Err(CodegenError::Msg(format!("unsupported VEC={vec}")));
    }
    if !matches!(unroll, 1 | 2 | 4 | 8) {
        return Err(CodegenError::Msg(format!("unsupported UNROLL={unroll}")));
    }
    if tg == 0 || tg > 1024 {
        return Err(CodegenError::Msg(format!("unsupported TG={tg}")));
    }

    let k_body = emit_k_loop(cols, tg, vec, unroll);
    let reduce = emit_tg_reduce(tg);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const {ty}* A [[buffer(0)]],
  device const {ty}* x [[buffer(1)]],
  device {ty}* y [[buffer(2)]],
  uint row [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]]
) {{
  if (row >= {rows}u) return;
  device const {ty}* a = A + row * {cols}u;
  float acc = 0.0f;
{k_body}
{reduce}
  if (lid == 0u) y[row] = acc;
}}
"#,
        name = kir.name,
        ty = ty,
        rows = rows,
        cols = cols,
        k_body = k_body,
        reduce = reduce,
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows, tg },
    })
}

/// Q4_K dtype fusion: dequant+matvec in one Kernel IR region (searched via nsg/nr0).
fn render_matvec_q4k(
    kir: &KernelIr,
    rows: usize,
    cols: usize,
    sched: OptSchedule,
) -> Result<MetalKernelSource, CodegenError> {
    if cols % 256 != 0 {
        return Err(CodegenError::Msg("Q4_K matvec cols must be multiple of 256".into()));
    }
    let nsg = match sched.nsg {
        1 | 2 | 4 => sched.nsg as u64,
        _ => return Err(CodegenError::Msg(format!("unsupported NSG={}", sched.nsg))),
    };
    let nr0 = match sched.nr0 {
        2 | 4 | 8 => sched.nr0 as u64,
        _ => return Err(CodegenError::Msg(format!("unsupported NR0={}", sched.nr0))),
    };
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb = cols / 256;
    let nb01 = nb * 144;
    let name = &kir.name;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const uchar* A [[buffer(0)]],
  device const float* x [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr short NSG = {nsg};
  constexpr short NR0 = {nr0};
  constexpr uint QK = 256u;
  constexpr uint ROW_BYTES = {nb01}u;
  constexpr uint16_t kmask1 = 0x3f3fu;
  constexpr uint16_t kmask2 = 0x0f0fu;
  constexpr uint16_t kmask3 = 0xc0c0u;

  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;

  const int first_row = int((tgpig * NSG + sgitg) * NR0);
  device const uchar* row0 = A + (ulong)first_row * ROW_BYTES;

  float yl[16];
  float yh[16];
  float sumf[NR0] = {{0.f}};

  device const float* y4 = x + ix * QK + 64 * iq + 8 * ir;

  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;

  for (int ib = ix; ib < {nb}; ib += 4) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    for (short i = 0; i < 8; ++i) {{
      yl[i + 0] = y4[i + 0];
      sumy[0] += yl[i + 0];
      yl[i + 8] = y4[i + 32];
      sumy[1] += yl[i + 8];
      yh[i + 0] = y4[i + 128];
      sumy[2] += yh[i + 0];
      yh[i + 8] = y4[i + 160];
      sumy[3] += yh[i + 8];
    }}

    device const uchar* blk = row0 + (ulong)ib * 144u;
    device const uint16_t* sc = (device const uint16_t*)(blk + 4) + iq;
    device const uint16_t* q1 = (device const uint16_t*)(blk + 16) + 16 * iq + 4 * ir;
    device const half* dh = (device const half*)blk;

    for (short row = 0; row < NR0; row++) {{
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

      device const uint16_t* q2 = q1 + 32;

      float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
      #pragma unroll
      for (short i = 0; i < 4; ++i) {{
        acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000Fu);
        acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00u);
        acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0u);
        acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000u);
        acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000Fu);
        acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00u);
        acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0u);
        acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000u);
      }}

      sumf[row] += float(dh[0]) * ((acc1[0] + (1.f / 256.f) * acc1[1]) * float(sc8[0]) +
                                   (acc1[2] + (1.f / 256.f) * acc1[3]) * float(sc8[1]) * (1.f / 16.f) +
                                   (acc2[0] + (1.f / 256.f) * acc2[1]) * float(sc8[4]) +
                                   (acc2[2] + (1.f / 256.f) * acc2[3]) * float(sc8[5]) * (1.f / 16.f)) -
                  float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                  sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

      q1 += ROW_BYTES / 2u;
      sc += ROW_BYTES / 2u;
      dh += ROW_BYTES / 2u;
    }}
    y4 += 4 * QK;
  }}

  for (int row = 0; row < NR0; ++row) {{
    if (first_row + row >= {rows}) break;
    float sum_all = simd_sum(sumf[row]);
    if (tiisg == 0) y[first_row + row] = sum_all;
  }}
}}
"#
    );
    let n_tg = (rows as u64 + rows_per_tg - 1) / rows_per_tg;
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}

fn render_sum_last(
    kir: &KernelIr,
    rows: usize,
    cols: usize,
    sched: OptSchedule,
) -> Result<MetalKernelSource, CodegenError> {
    let ty = kir.out_dtype.msl();
    let tg = sched.tg.max(1);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const {ty}* inp [[buffer(0)]],
  device {ty}* out [[buffer(1)]],
  uint row [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]]
) {{
  if (row >= {rows}u) return;
  device const {ty}* a = inp + row * {cols}u;
  float acc = 0.0f;
  for (uint k = lid; k < {cols}u; k += {tg}u) {{
    acc += a[k];
  }}
{reduce}
  if (lid == 0u) out[row] = acc;
}}
"#,
        name = kir.name,
        ty = ty,
        rows = rows,
        cols = cols,
        tg = tg,
        reduce = emit_tg_reduce(tg),
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 1,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows, tg },
    })
}

fn emit_k_loop(cols: usize, tg: u64, vec: u32, unroll: u32) -> String {
    let stride = tg * u64::from(vec);
    let step = stride * u64::from(unroll);
    let mut s = String::new();
    s.push_str(&format!(
        "  uint k = lid * {vec}u;\n  const uint stride = {stride}u;\n",
        vec = vec,
        stride = stride,
    ));
    // Main loop: UNROLL vector chunks spaced by `tg * vec` (no cross-lane overlap).
    let last_u = u64::from(unroll.saturating_sub(1));
    s.push_str(&format!(
        "  for (; k + {last_u}u * stride + {vec}u - 1u < {cols}u; k += {step}u) {{\n",
        last_u = last_u,
        vec = vec,
        cols = cols,
        step = step,
    ));
    for u in 0..unroll {
        let off = if u == 0 {
            "k".to_string()
        } else {
            format!("k + {u}u * stride")
        };
        s.push_str(&emit_vec_chunk(&off, vec));
    }
    s.push_str("  }\n");
    // Remainder: one vector chunk per iteration.
    s.push_str(&format!(
        "  for (; k + {vec}u - 1u < {cols}u; k += stride) {{\n",
        vec = vec,
        cols = cols,
    ));
    s.push_str(&emit_vec_chunk("k", vec));
    s.push_str("  }\n");
    // Scalar tail.
    s.push_str(&format!(
        "  for (; k < {cols}u; k += {tg}u) {{\n    acc += a[k] * x[k];\n  }}\n",
        cols = cols,
        tg = tg,
    ));
    s
}

fn emit_vec_chunk(off: &str, vec: u32) -> String {
    match vec {
        4 => format!(
            "    {{\n      float4 av = *(device const float4*)(a + {off});\n      float4 xv = *(device const float4*)(x + {off});\n      acc += dot(av, xv);\n    }}\n"
        ),
        2 => format!(
            "    {{\n      float2 av = *(device const float2*)(a + {off});\n      float2 xv = *(device const float2*)(x + {off});\n      acc += av.x * xv.x + av.y * xv.y;\n    }}\n"
        ),
        _ => format!("    acc += a[{off}] * x[{off}];\n"),
    }
}

fn emit_tg_reduce(tg: u64) -> String {
    if tg <= 32 {
        "  acc = simd_sum(acc);\n".to_string()
    } else {
        format!(
            r#"  threadgroup float red[{tg}];
  red[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = {tg}u / 2u; stride > 0u; stride >>= 1u) {{
    if (lid < stride) red[lid] += red[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  acc = red[0];
"#,
            tg = tg
        )
    }
}

/// Naive MQA SDPA: per-head Q@Kᵀ → softmax → @V (matches Gemma AttnGqa score scale: raw dots).
fn render_sdpa_naive(
    kir: &KernelIr,
    n_q: usize,
    hd: usize,
    max_t: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    // One thread per head (correct, simple; Hi prompts are short).
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const float* q [[buffer(0)]],
  device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]],
  device const uint* meta [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint head [[thread_position_in_grid]]
) {{
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  if (T == 0u) return;
  if (T > {max_t}u) T = {max_t}u;
  device const float* qh = q + head * {hd}u;
  device float* oh = out + head * {hd}u;

  float m = -INFINITY;
  for (uint t = 0u; t < T; t++) {{
    device const float* kt = k + (start + t) * {hd}u;
    float s = 0.0f;
    for (uint d = 0u; d < {hd}u; d++) s += qh[d] * kt[d];
    m = max(m, s);
  }}
  float l = 0.0f;
  for (uint t = 0u; t < T; t++) {{
    device const float* kt = k + (start + t) * {hd}u;
    float s = 0.0f;
    for (uint d = 0u; d < {hd}u; d++) s += qh[d] * kt[d];
    l += exp(s - m);
  }}
  float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  for (uint d = 0u; d < {hd}u; d++) oh[d] = 0.0f;
  for (uint t = 0u; t < T; t++) {{
    device const float* kt = k + (start + t) * {hd}u;
    device const float* vt = v + (start + t) * {hd}u;
    float s = 0.0f;
    for (uint d = 0u; d < {hd}u; d++) s += qh[d] * kt[d];
    float p = exp(s - m) * inv_l;
    for (uint d = 0u; d < {hd}u; d++) oh[d] += p * vt[d];
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 4,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise { n: n_q.max(1) },
    })
}

fn render_rmsnorm(
    kir: &KernelIr,
    n: usize,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let tg = 256u64;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* w [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint lid [[thread_index_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]
) {{
  threadgroup float sh[{tg}];
  float local = 0.0f;
  for (uint i = lid; i < {n}u; i += tpg) local += x[i] * x[i];
  sh[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2u; s > 0u; s >>= 1u) {{
    if (lid < s) sh[lid] += sh[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  float inv = rsqrt(sh[0] / float({n}) + {eps:?}f);
  for (uint i = lid; i < {n}u; i += tpg) {{
    y[i] = x[i] * inv * w[i];
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

fn render_rmsnorm_add(
    kir: &KernelIr,
    n: usize,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* w [[buffer(1)]],
  device const float* residual [[buffer(2)]],
  device float* y [[buffer(3)]],
  uint lid [[thread_index_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]
) {{
  threadgroup float sh[256];
  float local = 0.0f;
  for (uint i = lid; i < {n}u; i += tpg) local += x[i] * x[i];
  sh[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2u; s > 0u; s >>= 1u) {{
    if (lid < s) sh[lid] += sh[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  float inv = rsqrt(sh[0] / float({n}) + {eps:?}f);
  for (uint i = lid; i < {n}u; i += tpg) {{
    y[i] = x[i] * inv * w[i] + residual[i];
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 3,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows: 1, tg: 256 },
    })
}

fn render_rmsnorm_add_scale(
    kir: &KernelIr,
    n: usize,
    eps: f32,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let tg = 256u64;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* w [[buffer(1)]],
  device const float* residual [[buffer(2)]],
  device float* y [[buffer(3)]],
  uint lid [[thread_index_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]
) {{
  threadgroup float sh[{tg}];
  float local = 0.0f;
  for (uint i = lid; i < {n}u; i += tpg) local += x[i] * x[i];
  sh[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2u; s > 0u; s >>= 1u) {{
    if (lid < s) sh[lid] += sh[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  float inv = rsqrt(sh[0] / float({n}) + {eps:?}f);
  float sc = {scale:?}f;
  for (uint i = lid; i < {n}u; i += tpg) {{
    y[i] = sc * (x[i] * inv * w[i] + residual[i]);
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 3,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

fn render_rmsnorm_per_head(
    kir: &KernelIr,
    n_heads: usize,
    hd: usize,
    eps: f32,
    with_weight: bool,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let body = if with_weight {
        "y[base + i] = h[i] * inv * w[i];"
    } else {
        "y[base + i] = h[i] * inv;"
    };
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* w [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint head [[thread_position_in_grid]]
) {{
  if (head >= {n_heads}u) return;
  uint base = head * {hd}u;
  device const float* h = x + base;
  float ss = 0.0;
  for (uint i = 0; i < {hd}u; i++) ss += h[i] * h[i];
  float inv = rsqrt(ss / {hd}.0f + {eps:?}f);
  for (uint i = 0; i < {hd}u; i++) {{
    {body}
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise {
            n: n_heads.max(1),
        },
    })
}

fn render_rope(
    kir: &KernelIr,
    n_heads: usize,
    hd: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* cos_sin [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint head [[thread_position_in_grid]]
) {{
  if (head >= {n_heads}u) return;
  device const float* h = x + head * {hd}u;
  device float* o = y + head * {hd}u;
  uint half_d = {hd}u / 2u;
  for (uint i = 0; i < half_d; i++) {{
    float c = cos_sin[i];
    float s = cos_sin[half_d + i];
    float u = h[i];
    float v = h[i + half_d];
    o[i] = u * c - v * s;
    o[i + half_d] = u * s + v * c;
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise { n: n_heads.max(1) },
    })
}

fn render_gelu_mul(
    kir: &KernelIr,
    n: usize,
    up_off: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* gate [[buffer(0)]],
  device const float* up [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint i [[thread_position_in_grid]]
) {{
  if (i >= {n}u) return;
  float x = gate[i];
  float ax = clamp(x, -20.0f, 20.0f);
  float u = 0.79788456f * (ax + 0.044715f * ax * ax * ax);
  float g = 0.5f * ax * (1.0f + precise::tanh(u));
  float outv = g * up[{up_off}u + i];
  y[i] = isnan(outv) ? 0.0f : outv;
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 2,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn render_copy_slice(
    kir: &KernelIr,
    src_off: usize,
    dst_off: usize,
    n: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device float* y [[buffer(1)]],
  uint i [[thread_position_in_grid]]
) {{
  if (i >= {n}u) return;
  y[{dst_off}u + i] = x[{src_off}u + i];
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 1,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn render_softcap_argmax(
    kir: &KernelIr,
    n: usize,
    cap: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let name = &kir.name;
    let tg = 256u64;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint lid [[thread_index_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]
) {{
  uint best = lid;
  float m = -INFINITY;
  for (uint i = lid; i < {n}u; i += tpg) {{
    float v = {cap:?}f * tanh(x[i] / {cap:?}f);
    if (v > m) {{ m = v; best = i; }}
  }}
  threadgroup float shv[{tg}];
  threadgroup uint shi[{tg}];
  shv[lid] = m;
  shi[lid] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2u; s > 0u; s >>= 1u) {{
    if (lid < s) {{
      if (shv[lid + s] > shv[lid]) {{
        shv[lid] = shv[lid + s];
        shi[lid] = shi[lid + s];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  if (lid == 0u) out[0] = (float)shi[0];
}}
"#
    );
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source,
        n_inputs: 1,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

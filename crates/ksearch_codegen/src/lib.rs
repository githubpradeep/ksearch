//! Lower IR regions to MSL (Thesis A: generated kernels).

use ksearch_ir::{DType, Graph, IrError, Op, Shape, TensorId};

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ir(#[from] IrError),
    #[error("{0}")]
    Msg(String),
}

/// A single Metal kernel ready to compile.
#[derive(Clone, Debug)]
pub struct MetalKernelSource {
    pub name: String,
    pub source: String,
    /// Number of input buffers (binding 0..n-1), output at binding n.
    pub n_inputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: LaunchHint,
}

#[derive(Clone, Debug)]
pub enum LaunchHint {
    Elementwise { n: usize },
    Rows { rows: usize, cols: usize },
    /// One threadgroup per row; `tg` threads cooperate on that row (matvec reduce).
    RowsParallel { rows: usize, tg: u64 },
}

/// Lower `out` and its dependencies into one Metal kernel.
pub fn lower_to_metal(graph: &Graph, out: TensorId) -> Result<MetalKernelSource, CodegenError> {
    let node = graph.node(out)?;
    match &node.op {
        Op::Add { .. } | Op::Mul { .. } => lower_elementwise(graph, out),
        Op::SumReduce { inp, axis } if *axis == graph.node(*inp)?.shape.rank().saturating_sub(1) => {
            if let Op::MulBroadcastRow { left, row } = &graph.node(*inp)?.op {
                return lower_matvec(graph, out, *left, *row);
            }
            lower_sum_last(graph, out, *inp)
        }
        Op::MatVecQ4K { w, x } => lower_matvec_q4k(graph, out, *w, *x),
        Op::MatVecQ6K { w, x } => lower_matvec_q6k(graph, out, *w, *x),
        Op::MatVecBF16 { w, x } => lower_matvec_bf16(graph, out, *w, *x),
        Op::MatVecQ4KGateUpGelu { gate, up, x } => lower_matvec_q4k_gate_up_gelu(graph, out, *gate, *up, *x),
        Op::MatVecQ4KRmsGateUpGelu { gate, up, x, w, inv, eps } => {
            lower_matvec_q4k_rms_gate_up_gelu(graph, out, *gate, *up, *x, *w, *inv, *eps)
        }
        Op::MatVecQ4KRms { w, x, nw, inv, eps } => {
            lower_matvec_q4k_rms(graph, out, *w, *x, *nw, *inv, *eps)
        }
        Op::InvRms { x, eps } => lower_inv_rms(graph, out, *x, *eps),
        Op::ScaleConst { x, scale } => lower_scale_const(graph, out, *x, *scale),
        Op::ScaleBuf { x, s } => lower_scale_buf(graph, out, *x, *s),
        Op::Softcap { x, cap } => lower_softcap(graph, out, *x, *cap),
        Op::GeluMul { gate, up, up_off } => lower_gelu_mul(graph, out, *gate, *up, *up_off),
        Op::ArgMax { x } => lower_argmax(graph, out, *x),
        Op::RmsNorm { x, w, eps } => lower_rmsnorm(graph, out, *x, *w, *eps),
        Op::RmsNormAdd { x, w, residual, eps } => lower_rmsnorm_add(graph, out, *x, *w, *residual, *eps),
        Op::RmsNormAddScale { x, w, residual, eps, scale } => lower_rmsnorm_add_scale(graph, out, *x, *w, *residual, *eps, *scale),
        Op::RmsNormPerHead {
            x,
            w,
            n_heads,
            hd,
            eps,
            with_weight,
        } => lower_rmsnorm_per_head(graph, out, *x, *w, *n_heads, *hd, *eps, *with_weight),
        Op::Rope {
            x,
            cos_sin,
            n_heads,
            hd,
        } => lower_rope(graph, out, *x, *cos_sin, *n_heads, *hd),
        Op::AttnGqa {
            q,
            k,
            v,
            meta,
            n_q,
            hd,
            max_t,
        } => lower_attn_gqa(graph, out, *q, *k, *v, *meta, *n_q, *hd, *max_t),
        Op::AttnGqaQ4 {
            q,
            k,
            v,
            meta,
            n_q,
            hd,
            max_t,
        } => lower_attn_gqa_q4(graph, out, *q, *k, *v, *meta, *n_q, *hd, *max_t),
        Op::AttnGqaQ4Split {
            q,
            k,
            v,
            meta,
            n_q,
            hd,
            max_t,
            nwg,
        } => lower_attn_gqa_q4_split(graph, out, *q, *k, *v, *meta, *n_q, *hd, *max_t, *nwg),
        Op::AttnGqaQ4Reduce {
            partials,
            n_q,
            hd,
            nwg,
        } => lower_attn_gqa_q4_reduce(graph, out, *partials, *n_q, *hd, *nwg),
        Op::AttnGqaQ4Fused {
            q,
            k,
            v,
            k_new,
            v_new,
            meta,
            pos,
            n_q,
            hd,
            max_t,
        } => lower_attn_gqa_q4_fused(
            graph, out, *q, *k, *v, *k_new, *v_new, *meta, *pos, *n_q, *hd, *max_t,
        ),
        Op::AttnGqaQ4QFused {
            q,
            q_norm,
            cos_sin,
            k_cache,
            v_cache,
            meta,
            n_q,
            hd,
            max_t,
            eps,
        } => lower_attn_gqa_q4_q_fused(
            graph, out, *q, *q_norm, *cos_sin, *k_cache, *v_cache, *meta, *n_q, *hd, *max_t, *eps,
        ),
        Op::KvAppendQ4 {
            src,
            pos,
            hd,
            max_t,
        } => lower_kv_append_q4(graph, out, *src, *pos, *hd, *max_t),
        Op::CopySlice {
            src,
            src_off,
            dst_off,
            n,
        } => lower_copy_slice(graph, out, *src, *src_off, *dst_off, *n),
        Op::GatherQ4KRow {
            w,
            row_idx,
            cols,
            scale,
        } => lower_gather_q4k_row(graph, out, *w, *row_idx, *cols, *scale),
        Op::GatherQ5KRow {
            w,
            row_idx,
            cols,
            scale,
        } => lower_gather_q5k_row(graph, out, *w, *row_idx, *cols, *scale),
        Op::SoftcapArgmax { x, cap } => lower_softcap_argmax(graph, out, *x, *cap),
        Op::Input { .. } => Err(CodegenError::Msg("cannot lower Input alone".into())),
        other => Err(CodegenError::Msg(format!(
            "unsupported root op: {:?}",
            other
        ))),
    }
}

fn push_deps(op: &Op, stack: &mut Vec<TensorId>) {
    match op {
        Op::Input { .. } => {}
        Op::Add { a, b } | Op::Mul { a, b } => {
            stack.push(*a);
            stack.push(*b);
        }
        Op::SumReduce { inp, .. } => stack.push(*inp),
        Op::MulBroadcastRow { left, row } => {
            stack.push(*left);
            stack.push(*row);
        }
        Op::MatVecQ4K { w, x } | Op::MatVecQ6K { w, x } | Op::MatVecBF16 { w, x } => {
            stack.push(*w);
            stack.push(*x);
        }
        Op::MatVecQ4KGateUpGelu { gate, up, x } => {
            stack.push(*gate);
            stack.push(*up);
            stack.push(*x);
        }
        Op::MatVecQ4KRmsGateUpGelu { gate, up, x, w, inv, .. } => {
            stack.push(*gate);
            stack.push(*up);
            stack.push(*x);
            stack.push(*w);
            stack.push(*inv);
        }
        Op::MatVecQ4KRms { w, x, nw, inv, .. } => {
            stack.push(*w);
            stack.push(*x);
            stack.push(*nw);
            stack.push(*inv);
        }
        Op::InvRms { x, .. } => stack.push(*x),
        Op::ScaleConst { x, .. } | Op::Softcap { x, .. } | Op::ArgMax { x } => stack.push(*x),
        Op::ScaleBuf { x, s } => {
            stack.push(*x);
            stack.push(*s);
        }
        Op::GeluMul { gate, up, .. } => {
            stack.push(*gate);
            stack.push(*up);
        }
        Op::RmsNorm { x, w, .. } => {
            stack.push(*x);
            stack.push(*w);
        }
        Op::RmsNormAdd { x, w, residual, .. } | Op::RmsNormAddScale { x, w, residual, .. } => {
            stack.push(*x);
            stack.push(*w);
            stack.push(*residual);
        }
        Op::RmsNormPerHead { x, w, .. } => {
            stack.push(*x);
            stack.push(*w);
        }
        Op::Rope { x, cos_sin, .. } => {
            stack.push(*x);
            stack.push(*cos_sin);
        }
        Op::AttnGqa { q, k, v, meta, .. }
        | Op::AttnGqaQ4 { q, k, v, meta, .. }
        | Op::AttnGqaQ4Split { q, k, v, meta, .. } => {
            stack.push(*q);
            stack.push(*k);
            stack.push(*v);
            stack.push(*meta);
        }
        Op::AttnGqaQ4Reduce { partials, .. } => stack.push(*partials),
        Op::AttnGqaQ4Fused {
            q,
            k,
            v,
            k_new,
            v_new,
            meta,
            pos,
            ..
        } => {
            stack.push(*q);
            stack.push(*k);
            stack.push(*v);
            stack.push(*k_new);
            stack.push(*v_new);
            stack.push(*meta);
            stack.push(*pos);
        }
        Op::AttnGqaQ4QFused {
            q,
            q_norm,
            cos_sin,
            k_cache,
            v_cache,
            meta,
            ..
        } => {
            stack.push(*q);
            stack.push(*q_norm);
            stack.push(*cos_sin);
            stack.push(*k_cache);
            stack.push(*v_cache);
            stack.push(*meta);
        }
        Op::KvAppendQ4 { src, pos, .. } => {
            stack.push(*src);
            stack.push(*pos);
        }
        Op::CopySlice { src, .. } => stack.push(*src),
        Op::GatherQ4KRow { w, row_idx, .. } | Op::GatherQ5KRow { w, row_idx, .. } => {
            stack.push(*w);
            stack.push(*row_idx);
        }
        Op::SoftcapArgmax { x, .. } => stack.push(*x),
    }
}

fn collect_inputs(graph: &Graph, root: TensorId) -> Result<Vec<TensorId>, CodegenError> {
    let mut inputs = Vec::new();
    let mut stack = vec![root];
    let mut seen = vec![false; graph.nodes.len()];
    while let Some(id) = stack.pop() {
        if seen[id.0 as usize] {
            continue;
        }
        seen[id.0 as usize] = true;
        let op = &graph.node(id)?.op;
        match op {
            Op::Input { .. } => inputs.push(id),
            _ => push_deps(op, &mut stack),
        }
    }
    inputs.sort_by_key(|t| t.0);
    inputs.dedup();
    Ok(inputs)
}

fn reaches(graph: &Graph, from: TensorId, to: TensorId) -> Result<bool, CodegenError> {
    if from == to {
        return Ok(true);
    }
    let mut stack = vec![to];
    let mut seen = vec![false; graph.nodes.len()];
    while let Some(id) = stack.pop() {
        if seen[id.0 as usize] {
            continue;
        }
        seen[id.0 as usize] = true;
        if id == from {
            return Ok(true);
        }
        push_deps(&graph.node(id)?.op, &mut stack);
    }
    Ok(false)
}

fn expr_load(binding: &std::collections::HashMap<TensorId, usize>, id: TensorId) -> String {
    let b = binding[&id];
    if b < 1000 {
        format!("in{b}[i]")
    } else {
        format!("t{}", id.0)
    }
}

fn lower_elementwise(graph: &Graph, out: TensorId) -> Result<MetalKernelSource, CodegenError> {
    let inputs = collect_inputs(graph, out)?;
    let (out_shape, out_dtype) = graph.shape_dtype(out)?;
    let n = out_shape.numel();
    let name = format!("k_elem_{}", out.0);

    let mut binding = std::collections::HashMap::new();
    for (i, id) in inputs.iter().enumerate() {
        binding.insert(*id, i);
    }

    let mut body = String::new();
    body.push_str("  uint i = tgp;\n");
    body.push_str(&format!("  if (i >= {n}u) return;\n"));

    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = TensorId(idx as u32);
        if !reaches(graph, id, out)? && id != out {
            continue;
        }
        match &node.op {
            Op::Input { .. } => {}
            Op::Add { a, b } => {
                let ea = expr_load(&binding, *a);
                let eb = expr_load(&binding, *b);
                body.push_str(&format!("  {ty} t{idx} = {ea} + {eb};\n", ty = out_dtype.msl()));
                binding.insert(id, 1000 + idx);
            }
            Op::Mul { a, b } => {
                let ea = expr_load(&binding, *a);
                let eb = expr_load(&binding, *b);
                body.push_str(&format!("  {ty} t{idx} = {ea} * {eb};\n", ty = out_dtype.msl()));
                binding.insert(id, 1000 + idx);
            }
            _ => {
                return Err(CodegenError::Msg(
                    "elementwise lower only supports Add/Mul chains".into(),
                ));
            }
        }
    }

    let out_expr = expr_load(&binding, out);
    let n_inputs = inputs.len();
    body.push_str(&format!("  out[i] = {out_expr};\n"));

    let mut params = String::new();
    for i in 0..n_inputs {
        params.push_str(&format!(
            "  device const {ty}* in{i} [[buffer({i})]],\n",
            ty = out_dtype.msl()
        ));
    }
    params.push_str(&format!(
        "  device {ty}* out [[buffer({n_inputs})]]",
        ty = out_dtype.msl()
    ));

    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params},
  uint tgp [[thread_position_in_grid]]
) {{
{body}}}
"#
    );

    Ok(MetalKernelSource {
        name,
        source,
        n_inputs,
        out_shape,
        out_dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_matvec(
    graph: &Graph,
    out: TensorId,
    matrix: TensorId,
    vector: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(matrix)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms.rank() != 2 || vs.rank() != 1 || ms.0[1] != vs.0[0] || md != vd {
        return Err(CodegenError::Msg("matvec shape error".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let name = format!("k_matvec_{}", out.0);
    let ty = md.msl();
    // Simdgroup schedule: 1 TG/row, 32 threads reduce over K with float4.
    let tg = 32u64;
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
  uint k = lid * 4u;
  for (; k + 3u < {cols}u; k += {tg}u * 4u) {{
    float4 av = *(device const float4*)(a + k);
    float4 xv = *(device const float4*)(x + k);
    acc += dot(av, xv);
  }}
  for (; k < {cols}u; k += {tg}u) {{
    acc += a[k] * x[k];
  }}
  acc = simd_sum(acc);
  if (lid == 0u) y[row] = acc;
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: md,
        launch: LaunchHint::RowsParallel { rows, tg },
    })
}


fn lower_matvec_bf16(
    graph: &Graph,
    out: TensorId,
    matrix: TensorId,
    vector: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(matrix)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::BF16
        || vd != DType::F32
    {
        return Err(CodegenError::Msg("matvec_bf16 shape/dtype error".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let name = format!("k_matvec_bf16_{}", out.0);
    let tg = 32u64;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline float bf16_to_f32(ushort h) {{
  return as_type<float>((uint)h << 16);
}}

kernel void {name}(
  device const ushort* A [[buffer(0)]],
  device const float* x [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint row [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]]
) {{
  if (row >= {rows}u) return;
  device const ushort* a = A + row * {cols}u;
  float acc = 0.0f;
  uint k = lid * 4u;
  for (; k + 3u < {cols}u; k += {tg}u * 4u) {{
    ushort4 h = *(device const ushort4*)(a + k);
    float4 av = float4(bf16_to_f32(h[0]), bf16_to_f32(h[1]), bf16_to_f32(h[2]), bf16_to_f32(h[3]));
    float4 xv = *(device const float4*)(x + k);
    acc += dot(av, xv);
  }}
  for (; k < {cols}u; k += {tg}u) {{
    acc += bf16_to_f32(a[k]) * x[k];
  }}
  acc = simd_sum(acc);
  if (lid == 0u) y[row] = acc;
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel { rows, tg },
    })
}

fn lower_matvec_q4k(
    graph: &Graph,
    out: TensorId,
    matrix: TensorId,
    vector: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(matrix)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::Q4K
        || vd != DType::F32
        || ms.0[1] % 256 != 0
    {
        return Err(CodegenError::Msg("matvec_q4k shape/dtype error".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let nb = cols / 256;
    let name = format!("k_matvec_q4k_{}", out.0);

    // ggml-style simdgroup schedule for all Q4_K matvecs (pattern-matched lower).
    let nsg: u64 = 2;
    let nr0: u64 = 4;
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb01 = nb * 144; // bytes per weight row
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
    let _ = cols; // used via nb
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}

fn lower_matvec_q6k(
    graph: &Graph,
    out: TensorId,
    matrix: TensorId,
    vector: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(matrix)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::Q6K
        || vd != DType::F32
        || ms.0[1] % 256 != 0
    {
        return Err(CodegenError::Msg("matvec_q6k shape/dtype error".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let nb = cols / 256;
    let name = format!("k_matvec_q6k_{}", out.0);
    let nsg: u64 = 2;
    let nr0: u64 = 4;
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb01 = nb * 210;
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
  constexpr uchar kmask1 = 0x03u;
  constexpr uchar kmask2 = 0x0Cu;
  constexpr uchar kmask3 = 0x30u;
  constexpr uchar kmask4 = 0xC0u;

  const int first_row = int((tgpig * NSG + sgitg) * NR0);
  device const uchar* row0 = A + (ulong)first_row * ROW_BYTES;

  float sumf[NR0] = {{0.f}};
  float yl[16];

  const short tid = tiisg / 2;
  const short ix = tiisg % 2;
  const short ip = tid / 8;
  const short il = tid % 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0 / 16;
  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;

  for (int i = ix; i < {nb}; i += 2) {{
    device const uchar* blk = row0 + (ulong)i * 210u;
    device const uchar* q1 = blk + q_offset_l;
    device const uchar* q2 = q1 + 32;
    device const uchar* qh = blk + 128 + q_offset_h;
    device const char* sc = (device const char*)(blk + 192) + is;
    device const half* dh = (device const half*)(blk + 208);
    device const float* yy = x + i * QK + y_offset;

    for (short l = 0; l < 4; ++l) {{
      yl[4 * l + 0] = yy[l + 0];
      yl[4 * l + 1] = yy[l + 32];
      yl[4 * l + 2] = yy[l + 64];
      yl[4 * l + 3] = yy[l + 96];
    }}

    for (short row = 0; row < NR0; ++row) {{
      float4 sums = {{0.f, 0.f, 0.f, 0.f}};
      #pragma unroll
      for (short l = 0; l < 4; ++l) {{
        sums[0] += yl[4 * l + 0] * float(char((q1[l] & 0xFu) | ((qh[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4 * l + 1] * float(char((q2[l] & 0xFu) | ((qh[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4 * l + 2] * float(char((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4 * l + 3] * float(char((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
      }}
      sumf[row] += float(dh[0]) * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) +
                                   sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
      q1 += ROW_BYTES;
      q2 += ROW_BYTES;
      qh += ROW_BYTES;
      sc += ROW_BYTES;
      dh += ROW_BYTES / 2u;
    }}
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
    let _ = cols;
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}


fn lower_matvec_q4k_gate_up_gelu(
    graph: &Graph,
    out: TensorId,
    gate: TensorId,
    up: TensorId,
    vector: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(gate)?;
    let (us, ud) = graph.shape_dtype(up)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms != us
        || ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::Q4K
        || ud != DType::Q4K
        || vd != DType::F32
        || ms.0[1] % 256 != 0
    {
        return Err(CodegenError::Msg("matvec_q4k_gate_up_gelu shape/dtype".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let nb = cols / 256;
    let name = format!("k_mv_q4k_gug_{}", out.0);
    let nsg: u64 = 2;
    let nr0: u64 = 4;
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb01 = nb * 144;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const uchar* Wg [[buffer(0)]],
  device const uchar* Wu [[buffer(1)]],
  device const float* x [[buffer(2)]],
  device float* y [[buffer(3)]],
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
  device const uchar* row_g = Wg + (ulong)first_row * ROW_BYTES;
  device const uchar* row_u = Wu + (ulong)first_row * ROW_BYTES;

  float yl[16];
  float yh[16];
  float sumf_g[NR0] = {{0.f}};
  float sumf_u[NR0] = {{0.f}};

  device const float* y4 = x + ix * QK + 64 * iq + 8 * ir;
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;

  for (int ib = ix; ib < {nb}; ib += 4) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    for (short i = 0; i < 8; ++i) {{
      yl[i + 0] = y4[i + 0]; sumy[0] += yl[i + 0];
      yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
      yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
      yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
    }}

    device const uchar* blk_g = row_g + (ulong)ib * 144u;
    device const uchar* blk_u = row_u + (ulong)ib * 144u;
    device const uint16_t* sc_g = (device const uint16_t*)(blk_g + 4) + iq;
    device const uint16_t* q1_g = (device const uint16_t*)(blk_g + 16) + 16 * iq + 4 * ir;
    device const half* dh_g = (device const half*)blk_g;
    device const uint16_t* sc_u = (device const uint16_t*)(blk_u + 4) + iq;
    device const uint16_t* q1_u = (device const uint16_t*)(blk_u + 16) + 16 * iq + 4 * ir;
    device const half* dh_u = (device const half*)blk_u;

    for (short row = 0; row < NR0; row++) {{
      float4 acc1_g = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2_g = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc1_u = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2_u = {{0.f, 0.f, 0.f, 0.f}};

      sc16[0] = sc_g[0] & kmask1;
      sc16[1] = sc_g[2] & kmask1;
      sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
      sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);
      device const uint16_t* q2_g = q1_g + 32;
      #pragma unroll
      for (short i = 0; i < 4; ++i) {{
        acc1_g[0] += yl[2 * i + 0] * float(q1_g[i] & 0x000Fu);
        acc1_g[1] += yl[2 * i + 1] * float(q1_g[i] & 0x0F00u);
        acc1_g[2] += yl[2 * i + 8] * float(q1_g[i] & 0x00F0u);
        acc1_g[3] += yl[2 * i + 9] * float(q1_g[i] & 0xF000u);
        acc2_g[0] += yh[2 * i + 0] * float(q2_g[i] & 0x000Fu);
        acc2_g[1] += yh[2 * i + 1] * float(q2_g[i] & 0x0F00u);
        acc2_g[2] += yh[2 * i + 8] * float(q2_g[i] & 0x00F0u);
        acc2_g[3] += yh[2 * i + 9] * float(q2_g[i] & 0xF000u);
      }}
      sumf_g[row] += float(dh_g[0]) * ((acc1_g[0] + (1.f / 256.f) * acc1_g[1]) * float(sc8[0]) +
                                       (acc1_g[2] + (1.f / 256.f) * acc1_g[3]) * float(sc8[1]) * (1.f / 16.f) +
                                       (acc2_g[0] + (1.f / 256.f) * acc2_g[1]) * float(sc8[4]) +
                                       (acc2_g[2] + (1.f / 256.f) * acc2_g[3]) * float(sc8[5]) * (1.f / 16.f)) -
                     float(dh_g[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                       sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

      sc16[0] = sc_u[0] & kmask1;
      sc16[1] = sc_u[2] & kmask1;
      sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
      sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);
      device const uint16_t* q2_u = q1_u + 32;
      #pragma unroll
      for (short i = 0; i < 4; ++i) {{
        acc1_u[0] += yl[2 * i + 0] * float(q1_u[i] & 0x000Fu);
        acc1_u[1] += yl[2 * i + 1] * float(q1_u[i] & 0x0F00u);
        acc1_u[2] += yl[2 * i + 8] * float(q1_u[i] & 0x00F0u);
        acc1_u[3] += yl[2 * i + 9] * float(q1_u[i] & 0xF000u);
        acc2_u[0] += yh[2 * i + 0] * float(q2_u[i] & 0x000Fu);
        acc2_u[1] += yh[2 * i + 1] * float(q2_u[i] & 0x0F00u);
        acc2_u[2] += yh[2 * i + 8] * float(q2_u[i] & 0x00F0u);
        acc2_u[3] += yh[2 * i + 9] * float(q2_u[i] & 0xF000u);
      }}
      sumf_u[row] += float(dh_u[0]) * ((acc1_u[0] + (1.f / 256.f) * acc1_u[1]) * float(sc8[0]) +
                                       (acc1_u[2] + (1.f / 256.f) * acc1_u[3]) * float(sc8[1]) * (1.f / 16.f) +
                                       (acc2_u[0] + (1.f / 256.f) * acc2_u[1]) * float(sc8[4]) +
                                       (acc2_u[2] + (1.f / 256.f) * acc2_u[3]) * float(sc8[5]) * (1.f / 16.f)) -
                     float(dh_u[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                       sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

      q1_g += ROW_BYTES / 2u; sc_g += ROW_BYTES / 2u; dh_g += ROW_BYTES / 2u;
      q1_u += ROW_BYTES / 2u; sc_u += ROW_BYTES / 2u; dh_u += ROW_BYTES / 2u;
    }}
    y4 += 4 * QK;
  }}

  for (int row = 0; row < NR0; ++row) {{
    if (first_row + row >= {rows}) break;
    float gate = simd_sum(sumf_g[row]);
    float upv = simd_sum(sumf_u[row]);
    float ax = clamp(gate, -20.0f, 20.0f);
    float u = 0.79788456f * (ax + 0.044715f * ax * ax * ax);
    float g = 0.5f * ax * (1.0f + precise::tanh(u));
    float outv = g * upv;
    if (tiisg == 0) y[first_row + row] = isnan(outv) ? 0.0f : outv;
  }}
}}
"#
    );
    let n_tg = (rows as u64 + rows_per_tg - 1) / rows_per_tg;
    let _ = cols;
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 3,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}


fn lower_matvec_q4k_rms_gate_up_gelu(
    graph: &Graph,
    out: TensorId,
    gate: TensorId,
    up: TensorId,
    vector: TensorId,
    _nw: TensorId,
    _inv: TensorId,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(gate)?;
    let (us, ud) = graph.shape_dtype(up)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms != us
        || ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::Q4K
        || ud != DType::Q4K
        || vd != DType::F32
        || ms.0[1] % 256 != 0
    {
        return Err(CodegenError::Msg("matvec_q4k_rms_gate_up_gelu shape".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let nb = cols / 256;
    let name = format!("k_mv_q4k_rms_gug_{}", out.0);
    let nsg: u64 = 2;
    let nr0: u64 = 4;
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb01 = nb * 144;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const uchar* Wg [[buffer(0)]],
  device const uchar* Wu [[buffer(1)]],
  device const float* x [[buffer(2)]],
  device const float* nw [[buffer(3)]],
  device const float* inv_rms [[buffer(4)]],
  device float* y [[buffer(5)]],
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

  float inv = inv_rms[0];

  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;

  const int first_row = int((tgpig * NSG + sgitg) * NR0);
  device const uchar* row_g = Wg + (ulong)first_row * ROW_BYTES;
  device const uchar* row_u = Wu + (ulong)first_row * ROW_BYTES;

  float yl[16];
  float yh[16];
  float sumf_g[NR0] = {{0.f}};
  float sumf_u[NR0] = {{0.f}};

  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  int k_base = int(ix * QK + 64 * iq + 8 * ir);

  for (int ib = ix; ib < {nb}; ib += 4) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    for (short i = 0; i < 8; ++i) {{
      uint i0 = uint(k_base + i);
      uint i1 = uint(k_base + i + 32);
      uint i2 = uint(k_base + i + 128);
      uint i3 = uint(k_base + i + 160);
      yl[i + 0] = x[i0] * inv * nw[i0]; sumy[0] += yl[i + 0];
      yl[i + 8] = x[i1] * inv * nw[i1]; sumy[1] += yl[i + 8];
      yh[i + 0] = x[i2] * inv * nw[i2]; sumy[2] += yh[i + 0];
      yh[i + 8] = x[i3] * inv * nw[i3]; sumy[3] += yh[i + 8];
    }}

    device const uchar* blk_g = row_g + (ulong)ib * 144u;
    device const uchar* blk_u = row_u + (ulong)ib * 144u;
    device const uint16_t* sc_g = (device const uint16_t*)(blk_g + 4) + iq;
    device const uint16_t* q1_g = (device const uint16_t*)(blk_g + 16) + 16 * iq + 4 * ir;
    device const half* dh_g = (device const half*)blk_g;
    device const uint16_t* sc_u = (device const uint16_t*)(blk_u + 4) + iq;
    device const uint16_t* q1_u = (device const uint16_t*)(blk_u + 16) + 16 * iq + 4 * ir;
    device const half* dh_u = (device const half*)blk_u;

    for (short row = 0; row < NR0; row++) {{
      float4 acc1_g = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2_g = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc1_u = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2_u = {{0.f, 0.f, 0.f, 0.f}};

      sc16[0] = sc_g[0] & kmask1;
      sc16[1] = sc_g[2] & kmask1;
      sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
      sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);
      device const uint16_t* q2_g = q1_g + 32;
      #pragma unroll
      for (short i = 0; i < 4; ++i) {{
        acc1_g[0] += yl[2 * i + 0] * float(q1_g[i] & 0x000Fu);
        acc1_g[1] += yl[2 * i + 1] * float(q1_g[i] & 0x0F00u);
        acc1_g[2] += yl[2 * i + 8] * float(q1_g[i] & 0x00F0u);
        acc1_g[3] += yl[2 * i + 9] * float(q1_g[i] & 0xF000u);
        acc2_g[0] += yh[2 * i + 0] * float(q2_g[i] & 0x000Fu);
        acc2_g[1] += yh[2 * i + 1] * float(q2_g[i] & 0x0F00u);
        acc2_g[2] += yh[2 * i + 8] * float(q2_g[i] & 0x00F0u);
        acc2_g[3] += yh[2 * i + 9] * float(q2_g[i] & 0xF000u);
      }}
      sumf_g[row] += float(dh_g[0]) * ((acc1_g[0] + (1.f / 256.f) * acc1_g[1]) * float(sc8[0]) +
                                       (acc1_g[2] + (1.f / 256.f) * acc1_g[3]) * float(sc8[1]) * (1.f / 16.f) +
                                       (acc2_g[0] + (1.f / 256.f) * acc2_g[1]) * float(sc8[4]) +
                                       (acc2_g[2] + (1.f / 256.f) * acc2_g[3]) * float(sc8[5]) * (1.f / 16.f)) -
                     float(dh_g[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                       sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

      sc16[0] = sc_u[0] & kmask1;
      sc16[1] = sc_u[2] & kmask1;
      sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
      sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);
      device const uint16_t* q2_u = q1_u + 32;
      #pragma unroll
      for (short i = 0; i < 4; ++i) {{
        acc1_u[0] += yl[2 * i + 0] * float(q1_u[i] & 0x000Fu);
        acc1_u[1] += yl[2 * i + 1] * float(q1_u[i] & 0x0F00u);
        acc1_u[2] += yl[2 * i + 8] * float(q1_u[i] & 0x00F0u);
        acc1_u[3] += yl[2 * i + 9] * float(q1_u[i] & 0xF000u);
        acc2_u[0] += yh[2 * i + 0] * float(q2_u[i] & 0x000Fu);
        acc2_u[1] += yh[2 * i + 1] * float(q2_u[i] & 0x0F00u);
        acc2_u[2] += yh[2 * i + 8] * float(q2_u[i] & 0x00F0u);
        acc2_u[3] += yh[2 * i + 9] * float(q2_u[i] & 0xF000u);
      }}
      sumf_u[row] += float(dh_u[0]) * ((acc1_u[0] + (1.f / 256.f) * acc1_u[1]) * float(sc8[0]) +
                                       (acc1_u[2] + (1.f / 256.f) * acc1_u[3]) * float(sc8[1]) * (1.f / 16.f) +
                                       (acc2_u[0] + (1.f / 256.f) * acc2_u[1]) * float(sc8[4]) +
                                       (acc2_u[2] + (1.f / 256.f) * acc2_u[3]) * float(sc8[5]) * (1.f / 16.f)) -
                     float(dh_u[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                       sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

      q1_g += ROW_BYTES / 2u; sc_g += ROW_BYTES / 2u; dh_g += ROW_BYTES / 2u;
      q1_u += ROW_BYTES / 2u; sc_u += ROW_BYTES / 2u; dh_u += ROW_BYTES / 2u;
    }}
    k_base += 4 * int(QK);
  }}

  for (int row = 0; row < NR0; ++row) {{
    if (first_row + row >= {rows}) break;
    float gatev = simd_sum(sumf_g[row]);
    float upv = simd_sum(sumf_u[row]);
    float ax = clamp(gatev, -20.0f, 20.0f);
    float u = 0.79788456f * (ax + 0.044715f * ax * ax * ax);
    float g = 0.5f * ax * (1.0f + precise::tanh(u));
    float outv = g * upv;
    if (tiisg == 0) y[first_row + row] = isnan(outv) ? 0.0f : outv;
  }}
}}
"#
    );
    let n_tg = (rows as u64 + rows_per_tg - 1) / rows_per_tg;
    let _ = eps;
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 5,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}

fn lower_matvec_q4k_rms(
    graph: &Graph,
    out: TensorId,
    matrix: TensorId,
    vector: TensorId,
    _nw: TensorId,
    _inv: TensorId,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let (ms, md) = graph.shape_dtype(matrix)?;
    let (vs, vd) = graph.shape_dtype(vector)?;
    if ms.rank() != 2
        || vs.rank() != 1
        || ms.0[1] != vs.0[0]
        || md != DType::Q4K
        || vd != DType::F32
        || ms.0[1] % 256 != 0
    {
        return Err(CodegenError::Msg("matvec_q4k_rms shape".into()));
    }
    let rows = ms.0[0];
    let cols = ms.0[1];
    let nb = cols / 256;
    let name = format!("k_mv_q4k_rms_{}", out.0);
    let nsg: u64 = 2;
    let nr0: u64 = 4;
    let tg = nsg * 32;
    let rows_per_tg = nsg * nr0;
    let nb01 = nb * 144;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const uchar* A [[buffer(0)]],
  device const float* x [[buffer(1)]],
  device const float* nw [[buffer(2)]],
  device const float* inv_rms [[buffer(3)]],
  device float* y [[buffer(4)]],
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

  float inv = inv_rms[0];

  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;
  const int first_row = int((tgpig * NSG + sgitg) * NR0);
  device const uchar* row0 = A + (ulong)first_row * ROW_BYTES;

  float yl[16];
  float yh[16];
  float sumf[NR0] = {{0.f}};
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  int k_base = int(ix * QK + 64 * iq + 8 * ir);

  for (int ib = ix; ib < {nb}; ib += 4) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    for (short i = 0; i < 8; ++i) {{
      uint i0 = uint(k_base + i);
      uint i1 = uint(k_base + i + 32);
      uint i2 = uint(k_base + i + 128);
      uint i3 = uint(k_base + i + 160);
      yl[i + 0] = x[i0] * inv * nw[i0]; sumy[0] += yl[i + 0];
      yl[i + 8] = x[i1] * inv * nw[i1]; sumy[1] += yl[i + 8];
      yh[i + 0] = x[i2] * inv * nw[i2]; sumy[2] += yh[i + 0];
      yh[i + 8] = x[i3] * inv * nw[i3]; sumy[3] += yh[i + 8];
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
      q1 += ROW_BYTES / 2u; sc += ROW_BYTES / 2u; dh += ROW_BYTES / 2u;
    }}
    k_base += 4 * int(QK);
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
    let _ = eps;
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 4,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel {
            rows: n_tg as usize,
            tg,
        },
    })
}


fn lower_inv_rms(
    graph: &Graph,
    out: TensorId,
    x: TensorId,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let n = graph.shape_dtype(x)?.0.numel();
    let name = format!("k_inv_rms_{}", out.0);
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
  threadgroup float sh[{tg}];
  float local = 0.0f;
  for (uint i = lid; i < {n}u; i += tpg) local += x[i] * x[i];
  sh[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2u; s > 0u; s >>= 1u) {{
    if (lid < s) sh[lid] += sh[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  if (lid == 0u) out[0] = rsqrt(sh[0] / float({n}) + {eps:?}f);
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: Shape(vec![1]),
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

fn lower_sum_last(
    graph: &Graph,
    out: TensorId,
    inp: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (s, d) = graph.shape_dtype(inp)?;
    if s.rank() != 2 {
        return Err(CodegenError::Msg("sum_reduce expects rank-2".into()));
    }
    let rows = s.0[0];
    let cols = s.0[1];
    let name = format!("k_sum_{}", out.0);
    let ty = d.msl();
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
  device const {ty}* inp [[buffer(0)]],
  device {ty}* out [[buffer(1)]],
  uint row [[thread_position_in_grid]]
) {{
  if (row >= {rows}u) return;
  {ty} acc = 0.0;
  uint base = row * {cols}u;
  for (uint k = 0u; k < {cols}u; k++) {{
    acc += inp[base + k];
  }}
  out[row] = acc;
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: d,
        launch: LaunchHint::Rows { rows, cols },
    })
}

fn lower_scale_const(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let n = shape.numel();
    let name = format!("k_sc_{}_{}", out.0, scale.to_bits());
    let lit = format!("{scale:?}");
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device float* y [[buffer(1)]],
  uint i [[thread_position_in_grid]]
) {{
  if (i >= {n}u) return;
  y[i] = x[i] * {lit}f;
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_scale_buf(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _s: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let n = shape.numel();
    let name = format!("k_scale_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device const float* s [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint i [[thread_position_in_grid]]
) {{
  if (i >= {n}u) return;
  y[i] = x[i] * s[0];
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_softcap(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    cap: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let n = shape.numel();
    let name = format!("k_softcap_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device float* y [[buffer(1)]],
  uint i [[thread_position_in_grid]]
) {{
  if (i >= {n}u) return;
  y[i] = {cap:?}f * tanh(x[i] / {cap:?}f);
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_gelu_mul(
    graph: &Graph,
    out: TensorId,
    _gate: TensorId,
    _up: TensorId,
    up_off: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let n = shape.numel();
    let name = format!("k_gelu_mul_{}", out.0);
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
        name,
        source,
        n_inputs: 2,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_argmax(
    graph: &Graph,
    out: TensorId,
    x: TensorId,
) -> Result<MetalKernelSource, CodegenError> {
    let n = graph.shape_dtype(x)?.0.numel();
    let name = format!("k_argmax_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* x [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint tid [[thread_position_in_grid]]
) {{
  if (tid > 0u) return;
  uint best = 0;
  float m = x[0];
  for (uint i = 1; i < {n}u; i++) {{
    if (x[i] > m) {{ m = x[i]; best = i; }}
  }}
  out[0] = (float)best;
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: Shape(vec![1]),
        out_dtype: DType::F32,
        launch: LaunchHint::Elementwise { n: 1 },
    })
}

fn lower_rmsnorm(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _w: TensorId,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let n = shape.numel();
    let name = format!("k_rmsnorm_{}", out.0);
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
        name,
        source,
        n_inputs: 2,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

fn lower_rmsnorm_add(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _w: TensorId,
    _residual: TensorId,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let n = graph.shape_dtype(out)?.0.numel();
    let name = format!("k_rms_add_{}", out.0);
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
        name,
        source,
        n_inputs: 3,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel { rows: 1, tg: 256 },
    })
}

fn lower_rmsnorm_add_scale(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _w: TensorId,
    _residual: TensorId,
    eps: f32,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let n = graph.shape_dtype(out)?.0.numel();
    let name = format!("k_rms_add_sc_{}", out.0);
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
        name,
        source,
        n_inputs: 3,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

fn lower_rmsnorm_per_head(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _w: TensorId,
    n_heads: usize,
    hd: usize,
    eps: f32,
    with_weight: bool,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_rms_ph_{}_{}_{}", out.0, n_heads, with_weight as u8);
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
        name,
        source,
        n_inputs: 2,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise {
            n: n_heads.max(1),
        },
    })
}

fn lower_rope(
    graph: &Graph,
    out: TensorId,
    _x: TensorId,
    _cos_sin: TensorId,
    n_heads: usize,
    hd: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_rope_{}", out.0);
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
        name,
        source,
        n_inputs: 2,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::Elementwise { n: n_heads.max(1) },
    })
}

fn lower_attn_gqa(
    graph: &Graph,
    out: TensorId,
    _q: TensorId,
    _k: TensorId,
    _v: TensorId,
    _meta: TensorId,
    n_q: usize,
    hd: usize,
    max_t: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_{}", out.0);
    // Flash decode: online softmax over KV tiles. One TG per Q head.
    // TILE=256 covers SWA window (≤512) in ≤2 tiles; TG=256 → 8 simdgroups.
    let tg = 256u64;
    let tile = 256usize;
    let simd = 32u64;
    let nsg = tg / simd;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* q [[buffer(0)]],
  device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]],
  device const uint* meta [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint head [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint sgid [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {{
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  if (T == 0u) return;
  if (T > {max_t}u) T = {max_t}u;
  device const float* qh = q + head * {hd}u;
  device float* oh = out + head * {hd}u;

  threadgroup float sq[{hd}];
  threadgroup float scores[{tile}];
  threadgroup float exps[{tile}];
  threadgroup float upd[4]; // m, l, old_factor, inv_l

  if (tid == 0u) {{
    upd[0] = -INFINITY;
    upd[1] = 0.0f;
  }}
  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
    *(device float4*)(oh + d) = float4(0.0f);
  }}
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
    oh[d] = 0.0f;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = tid; d < {hd}u; d += {tg}u) {{
    sq[d] = qh[d];
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile0 = 0u; tile0 < T; tile0 += {tile}u) {{
    uint tc = min((uint){tile}u, T - tile0);

    for (uint wave = 0u; wave < tc; wave += {nsg}u) {{
      uint kv_off = wave + sgid;
      if (kv_off < tc) {{
        device const float* kt = k + (start + tile0 + kv_off) * {hd}u;
        float partial = 0.0f;
        for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u) {{
          float4 qv = *(threadgroup float4*)(sq + d);
          float4 kv = *(device const float4*)(kt + d);
          partial += dot(qv, kv);
        }}
        for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u) {{
          partial += sq[d] * kt[d];
        }}
        partial = simd_sum(partial);
        if (lane == 0u) scores[kv_off] = partial;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
      float m_old = upd[0];
      float l_old = upd[1];
      float m_new = m_old;
      for (uint i = 0u; i < tc; i++) m_new = max(m_new, scores[i]);
      float tile_sum = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        float e = exp(scores[i] - m_new);
        exps[i] = e;
        tile_sum += e;
      }}
      float scale = exp(m_old - m_new);
      float l_new = l_old * scale + tile_sum;
      upd[0] = m_new;
      upd[1] = l_new;
      upd[2] = l_new > 0.0f ? (l_old * scale) / l_new : 0.0f;
      upd[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float old_factor = upd[2];
    float inv_l = upd[3];
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
      float4 ov = *(device float4*)(oh + d);
      float4 acc = float4(0.0f);
      for (uint i = 0u; i < tc; i++) {{
        device const float* vt = v + (start + tile0 + i) * {hd}u;
        acc += exps[i] * *(device const float4*)(vt + d);
      }}
      ov = ov * old_factor + acc * inv_l;
      *(device float4*)(oh + d) = ov;
    }}
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
      float acc = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        acc += exps[i] * v[(start + tile0 + i) * {hd}u + d];
      }}
      oh[d] = oh[d] * old_factor + acc * inv_l;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 4,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: n_q.max(1),
            tg,
        },
    })
}

fn lower_kv_append_q4(
    graph: &Graph,
    out: TensorId,
    _src: TensorId,
    _pos: TensorId,
    hd: usize,
    max_t: usize,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg("KvAppendQ4 hd must be multiple of 32".into()));
    }
    let groups = hd / 32;
    let row_bytes = groups * 18;
    let name = format!("k_kv_q4_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* src [[buffer(0)]],
  device const uint* pos_buf [[buffer(1)]],
  device uchar* cache [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {{
  if (gid >= {groups}u) return;
  uint pos = pos_buf[0];
  if (pos >= {max_t}u) return;
  uint g = gid;
  float max_abs = 0.0f;
  for (uint d = 0u; d < 32u; d++) {{
    float a = fabs(src[g * 32u + d]);
    if (a > max_abs) max_abs = a;
  }}
  float scale = max_abs / 7.0f;
  if (max_abs == 0.0f) scale = 1.0f;
  float inv_scale = 1.0f / scale;
  uint base = pos * {row_bytes}u + g * 18u;
  *reinterpret_cast<device half*>(&cache[base]) = half(scale);
  for (uint i = 0u; i < 16u; i++) {{
    float v_lo = src[g * 32u + i];
    float v_hi = src[g * 32u + i + 16u];
    int q_lo = clamp(int(round(v_lo * inv_scale)) + 8, 0, 15);
    int q_hi = clamp(int(round(v_hi * inv_scale)) + 8, 0, 15);
    cache[base + 2u + i] = uchar(q_lo | (q_hi << 4));
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::Q40,
        launch: LaunchHint::Elementwise { n: groups.max(1) },
    })
}

fn lower_attn_gqa_q4(
    graph: &Graph,
    out: TensorId,
    _q: TensorId,
    _k: TensorId,
    _v: TensorId,
    _meta: TensorId,
    n_q: usize,
    hd: usize,
    max_t: usize,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg("AttnGqaQ4 hd must be multiple of 32".into()));
    }
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_q4_{}", out.0);
    let tg = 256u64;
    // Small tiles fit K+V in TG mem and shorten the V-accum inner loop.
    let tile = 32usize;
    let simd = 32u64;
    let nsg = tg / simd;
    let row_bytes = (hd / 32) * 18;
    let tile_bytes = tile * row_bytes;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline float q4_0_read_row(threadgroup const uchar* row, uint d) {{{{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = g * 18u;
  float scale = float(*reinterpret_cast<threadgroup const half*>(&row[offset]));
  if (e < 16u) return float(int(row[offset + 2u + e] & 0xFu) - 8) * scale;
  return float(int(row[offset + 2u + e - 16u] >> 4) - 8) * scale;
}}}}

inline float4 q4_0_read4_row(threadgroup const uchar* row, uint d) {{{{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = g * 18u;
  float scale = float(*reinterpret_cast<threadgroup const half*>(&row[offset]));
  if (e + 3u < 16u) {{{{
    return float4(
      float(int(row[offset + 2u + e + 0u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 1u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 2u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 3u] & 0xFu) - 8) * scale);
  }}}}
  if (e >= 16u && e + 3u < 32u) {{{{
    uint b = e - 16u;
    return float4(
      float(int(row[offset + 2u + b + 0u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 1u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 2u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 3u] >> 4) - 8) * scale);
  }}}}
  return float4(q4_0_read_row(row, d), q4_0_read_row(row, d+1u), q4_0_read_row(row, d+2u), q4_0_read_row(row, d+3u));
}}}}

inline void load_q4_tile(
  device const uchar* cache,
  uint start_pos,
  uint tc,
  uint row_bytes,
  uint tid,
  uint tg,
  threadgroup uchar* tile
) {{{{
  uint nbytes = tc * row_bytes;
  uint base = start_pos * row_bytes;
  for (uint i = tid * 4u; i + 3u < nbytes; i += tg * 4u) {{{{
    *reinterpret_cast<threadgroup uint*>(&tile[i]) =
      *reinterpret_cast<device const uint*>(&cache[base + i]);
  }}}}
  for (uint i = (nbytes / 4u) * 4u + tid; i < nbytes; i += tg) {{{{
    tile[i] = cache[base + i];
  }}}}
}}}}

kernel void {name}(
  device const float* q [[buffer(0)]],
  device const uchar* k [[buffer(1)]],
  device const uchar* v [[buffer(2)]],
  device const uint* meta [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint head [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint sgid [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {{
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  if (T == 0u) return;
  if (T > {max_t}u) T = {max_t}u;
  device const float* qh = q + head * {hd}u;
  device float* oh = out + head * {hd}u;
  const uint row_bytes = {row_bytes}u;

  threadgroup float sq[{hd}];
  threadgroup float scores[{tile}];
  threadgroup float exps[{tile}];
  threadgroup float upd[4];
  threadgroup uchar k_tile[{tile_bytes}];
  threadgroup uchar v_tile[{tile_bytes}];

  if (tid == 0u) {{
    upd[0] = -INFINITY;
    upd[1] = 0.0f;
  }}
  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
    *(device float4*)(oh + d) = float4(0.0f);
  }}
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) oh[d] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = tid; d < {hd}u; d += {tg}u) sq[d] = qh[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile0 = 0u; tile0 < T; tile0 += {tile}u) {{
    uint tc = min((uint){tile}u, T - tile0);
    load_q4_tile(k, start + tile0, tc, row_bytes, tid, {tg}u, k_tile);
    load_q4_tile(v, start + tile0, tc, row_bytes, tid, {tg}u, v_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wave = 0u; wave < tc; wave += {nsg}u) {{
      uint kv_off = wave + sgid;
      if (kv_off < tc) {{
        threadgroup const uchar* kr = k_tile + kv_off * row_bytes;
        float partial = 0.0f;
        for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u) {{
          float4 qv = *(threadgroup float4*)(sq + d);
          partial += dot(qv, q4_0_read4_row(kr, d));
        }}
        for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u) {{
          partial += sq[d] * q4_0_read_row(kr, d);
        }}
        partial = simd_sum(partial);
        if (lane == 0u) scores[kv_off] = partial;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
      float m_old = upd[0];
      float l_old = upd[1];
      float m_new = m_old;
      for (uint i = 0u; i < tc; i++) m_new = max(m_new, scores[i]);
      float tile_sum = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        float e = exp(scores[i] - m_new);
        exps[i] = e;
        tile_sum += e;
      }}
      float scale = exp(m_old - m_new);
      float l_new = l_old * scale + tile_sum;
      upd[0] = m_new;
      upd[1] = l_new;
      upd[2] = l_new > 0.0f ? (l_old * scale) / l_new : 0.0f;
      upd[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float old_factor = upd[2];
    float inv_l = upd[3];
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
      float4 ov = *(device float4*)(oh + d);
      float4 acc = float4(0.0f);
      for (uint i = 0u; i < tc; i++) {{
        threadgroup const uchar* vr = v_tile + i * row_bytes;
        acc += exps[i] * q4_0_read4_row(vr, d);
      }}
      ov = ov * old_factor + acc * inv_l;
      *(device float4*)(oh + d) = ov;
    }}
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
      float acc = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        threadgroup const uchar* vr = v_tile + i * row_bytes;
        acc += exps[i] * q4_0_read_row(vr, d);
      }}
      oh[d] = oh[d] * old_factor + acc * inv_l;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 4,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: n_q.max(1),
            tg,
        },
    })
}

fn lower_attn_gqa_q4_split(
    graph: &Graph,
    out: TensorId,
    _q: TensorId,
    _k: TensorId,
    _v: TensorId,
    _meta: TensorId,
    n_q: usize,
    hd: usize,
    max_t: usize,
    nwg: usize,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg(
            "AttnGqaQ4Split hd must be multiple of 32".into(),
        ));
    }
    if nwg == 0 {
        return Err(CodegenError::Msg("AttnGqaQ4Split nwg must be > 0".into()));
    }
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_q4_split_{}", out.0);
    // One simdgroup per WG (ggml flash_attn_ext_vec style); O stays in TG mem.
    let tg = 32u64;
    let simd = 32u64;
    let row_bytes = (hd / 32) * 18;
    let stride = hd + 2;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline float q4_0_read(device const uchar* cache, uint pos, uint row_bytes, uint d) {{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = pos * row_bytes + g * 18u;
  float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
  device const uchar* qs = cache + offset + 2u;
  if (e < 16u) return float(int(qs[e] & 0xFu) - 8) * scale;
  return float(int(qs[e - 16u] >> 4) - 8) * scale;
}}

inline float4 q4_0_read4(device const uchar* cache, uint pos, uint row_bytes, uint d) {{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = pos * row_bytes + g * 18u;
  float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
  device const uchar* qs = cache + offset + 2u;
  if (e + 3u < 16u) {{
    return float4(
      float(int(qs[e + 0u] & 0xFu) - 8) * scale,
      float(int(qs[e + 1u] & 0xFu) - 8) * scale,
      float(int(qs[e + 2u] & 0xFu) - 8) * scale,
      float(int(qs[e + 3u] & 0xFu) - 8) * scale
    );
  }}
  if (e >= 16u && e + 3u < 32u) {{
    uint b = e - 16u;
    return float4(
      float(int(qs[b + 0u] >> 4) - 8) * scale,
      float(int(qs[b + 1u] >> 4) - 8) * scale,
      float(int(qs[b + 2u] >> 4) - 8) * scale,
      float(int(qs[b + 3u] >> 4) - 8) * scale
    );
  }}
  return float4(
    q4_0_read(cache, pos, row_bytes, d),
    q4_0_read(cache, pos, row_bytes, d + 1u),
    q4_0_read(cache, pos, row_bytes, d + 2u),
    q4_0_read(cache, pos, row_bytes, d + 3u)
  );
}}

kernel void {name}(
  device const float* q [[buffer(0)]],
  device const uchar* k [[buffer(1)]],
  device const uchar* v [[buffer(2)]],
  device const uint* meta [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {{
  uint head = tgid / {nwg}u;
  uint wg = tgid % {nwg}u;
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  if (T > {max_t}u) T = {max_t}u;
  const uint row_bytes = {row_bytes}u;
  const uint stride = {stride}u;
  device float* po = out + (head * {nwg}u + wg) * stride;

  if (T == 0u) {{
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u)
      *(device float4*)(po + d) = float4(0.0f);
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) po[d] = 0.0f;
    if (tid == 0u) {{ po[{hd}u] = -INFINITY; po[{hd}u + 1u] = 0.0f; }}
    return;
  }}
  uint chunk = (T + {nwg}u - 1u) / {nwg}u;
  uint t0 = wg * chunk;
  uint t1 = min(T, t0 + chunk);
  if (t0 >= T) {{
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u)
      *(device float4*)(po + d) = float4(0.0f);
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) po[d] = 0.0f;
    if (tid == 0u) {{ po[{hd}u] = -INFINITY; po[{hd}u + 1u] = 0.0f; }}
    return;
  }}

  device const float* qh = q + head * {hd}u;
  threadgroup float sq[{hd}];
  threadgroup float so[{hd}];
  for (uint d = tid; d < {hd}u; d += {tg}u) {{
    sq[d] = qh[d];
    so[d] = 0.0f;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m = -INFINITY;
  float l = 0.0f;
  for (uint t = t0; t < t1; t++) {{
    uint pos = start + t;
    float partial = 0.0f;
    for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u)
      partial += dot(*(threadgroup float4*)(sq + d), q4_0_read4(k, pos, row_bytes, d));
    for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u)
      partial += sq[d] * q4_0_read(k, pos, row_bytes, d);
    float score = simd_sum(partial);

    float m_new = max(m, score);
    float e = exp(score - m_new);
    float scale = exp(m - m_new);
    float l_new = l * scale + e;
    float old_factor = l_new > 0.0f ? (l * scale) / l_new : 0.0f;
    float inv_l = l_new > 0.0f ? 1.0f / l_new : 0.0f;
    m = m_new;
    l = l_new;

    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
      float4 ov = *(threadgroup float4*)(so + d);
      *(threadgroup float4*)(so + d) =
        ov * old_factor + e * inv_l * q4_0_read4(v, pos, row_bytes, d);
    }}
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u)
      so[d] = so[d] * old_factor + e * inv_l * q4_0_read(v, pos, row_bytes, d);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u)
    *(device float4*)(po + d) = *(threadgroup float4*)(so + d);
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) po[d] = so[d];
  if (tid == 0u) {{
    po[{hd}u] = m;
    po[{hd}u + 1u] = l;
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 4,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: (n_q * nwg).max(1),
            tg,
        },
    })
}

fn lower_attn_gqa_q4_reduce(
    graph: &Graph,
    out: TensorId,
    _partials: TensorId,
    n_q: usize,
    hd: usize,
    nwg: usize,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg(
            "AttnGqaQ4Reduce hd must be multiple of 32".into(),
        ));
    }
    if nwg == 0 {
        return Err(CodegenError::Msg("AttnGqaQ4Reduce nwg must be > 0".into()));
    }
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_q4_reduce_{}", out.0);
    let tg = 32u64;
    let stride = hd + 2;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {name}(
  device const float* partials [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint head [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {{
  if (head >= {n_q}u) return;
  const uint stride = {stride}u;
  device float* oh = out + head * {hd}u;

  float m = -INFINITY;
  for (uint wg = 0u; wg < {nwg}u; wg++) {{
    float mw = partials[(head * {nwg}u + wg) * stride + {hd}u];
    m = max(m, mw);
  }}

  float l = 0.0f;
  for (uint wg = 0u; wg < {nwg}u; wg++) {{
    uint base = (head * {nwg}u + wg) * stride;
    float mw = partials[base + {hd}u];
    float lw = partials[base + {hd}u + 1u];
    if (lw > 0.0f && isfinite(mw))
      l += lw * exp(mw - m);
  }}

  float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
    float4 acc = float4(0.0f);
    for (uint wg = 0u; wg < {nwg}u; wg++) {{
      uint base = (head * {nwg}u + wg) * stride;
      float mw = partials[base + {hd}u];
      float lw = partials[base + {hd}u + 1u];
      if (!(lw > 0.0f && isfinite(mw))) continue;
      float w = lw * exp(mw - m) * inv_l;
      acc += w * *(device const float4*)(partials + base + d);
    }}
    *(device float4*)(oh + d) = acc;
  }}
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
    float acc = 0.0f;
    for (uint wg = 0u; wg < {nwg}u; wg++) {{
      uint base = (head * {nwg}u + wg) * stride;
      float mw = partials[base + {hd}u];
      float lw = partials[base + {hd}u + 1u];
      if (!(lw > 0.0f && isfinite(mw))) continue;
      float w = lw * exp(mw - m) * inv_l;
      acc += w * partials[base + d];
    }}
    oh[d] = acc;
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 1,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: n_q.max(1),
            tg,
        },
    })
}


fn lower_attn_gqa_q4_fused(
    graph: &Graph,
    out: TensorId,
    _q: TensorId,
    _k: TensorId,
    _v: TensorId,
    _k_new: TensorId,
    _v_new: TensorId,
    _meta: TensorId,
    _pos: TensorId,
    n_q: usize,
    hd: usize,
    max_t: usize,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg(
            "AttnGqaQ4Fused hd must be multiple of 32".into(),
        ));
    }
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_q4f_{}", out.0);
    let tg = 256u64;
    let tile = 32usize;
    let simd = 32u64;
    let nsg = tg / simd;
    let groups = hd / 32;
    let row_bytes = groups * 18;
    let tile_bytes = tile * row_bytes;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline float q4_0_read_row(threadgroup const uchar* row, uint d) {{{{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = g * 18u;
  float scale = float(*reinterpret_cast<threadgroup const half*>(&row[offset]));
  if (e < 16u) return float(int(row[offset + 2u + e] & 0xFu) - 8) * scale;
  return float(int(row[offset + 2u + e - 16u] >> 4) - 8) * scale;
}}}}

inline float4 q4_0_read4_row(threadgroup const uchar* row, uint d) {{{{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = g * 18u;
  float scale = float(*reinterpret_cast<threadgroup const half*>(&row[offset]));
  if (e + 3u < 16u) {{{{
    return float4(
      float(int(row[offset + 2u + e + 0u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 1u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 2u] & 0xFu) - 8) * scale,
      float(int(row[offset + 2u + e + 3u] & 0xFu) - 8) * scale);
  }}}}
  if (e >= 16u && e + 3u < 32u) {{{{
    uint b = e - 16u;
    return float4(
      float(int(row[offset + 2u + b + 0u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 1u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 2u] >> 4) - 8) * scale,
      float(int(row[offset + 2u + b + 3u] >> 4) - 8) * scale);
  }}}}
  return float4(q4_0_read_row(row, d), q4_0_read_row(row, d+1u), q4_0_read_row(row, d+2u), q4_0_read_row(row, d+3u));
}}}}

inline void load_q4_tile(
  device const uchar* cache,
  uint start_pos,
  uint tc,
  uint row_bytes,
  uint tid,
  uint tg,
  threadgroup uchar* tile
) {{{{
  uint nbytes = tc * row_bytes;
  uint base = start_pos * row_bytes;
  for (uint i = tid * 4u; i + 3u < nbytes; i += tg * 4u) {{{{
    *reinterpret_cast<threadgroup uint*>(&tile[i]) =
      *reinterpret_cast<device const uint*>(&cache[base + i]);
  }}}}
  for (uint i = (nbytes / 4u) * 4u + tid; i < nbytes; i += tg) {{{{
    tile[i] = cache[base + i];
  }}}}
}}}}

inline void q4_0_append_group(
  device const float* src,
  device uchar* cache,
  uint pos,
  uint row_bytes,
  uint g
) {{
  uint base = pos * row_bytes + g * 18u;
  float max_abs = 0.0f;
  for (uint d = 0u; d < 32u; d++) {{
    float a = fabs(src[g * 32u + d]);
    if (a > max_abs) max_abs = a;
  }}
  float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
  float inv_scale = 1.0f / scale;
  *reinterpret_cast<device half*>(&cache[base]) = half(scale);
  for (uint i = 0u; i < 16u; i++) {{
    int q_lo = clamp(int(round(src[g * 32u + i] * inv_scale)) + 8, 0, 15);
    int q_hi = clamp(int(round(src[g * 32u + i + 16u] * inv_scale)) + 8, 0, 15);
    cache[base + 2u + i] = uchar(q_lo | (q_hi << 4));
  }}
}}

kernel void {name}(
  device const float* q [[buffer(0)]],
  device uchar* k [[buffer(1)]],
  device uchar* v [[buffer(2)]],
  device const float* k_new [[buffer(3)]],
  device const float* v_new [[buffer(4)]],
  device const uint* meta [[buffer(5)]],
  device const uint* pos_buf [[buffer(6)]],
  device float* out [[buffer(7)]],
  uint head [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint sgid [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {{
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  uint cur = pos_buf[0];
  if (T == 0u) return;
  if (T > {max_t}u) T = {max_t}u;
  device const float* qh = q + head * {hd}u;
  device float* oh = out + head * {hd}u;
  const uint row_bytes = {row_bytes}u;

  threadgroup float sq[{hd}];
  threadgroup float scores[{tile}];
  threadgroup float exps[{tile}];
  threadgroup float upd[4];
  threadgroup uchar k_tile[{tile_bytes}];
  threadgroup uchar v_tile[{tile_bytes}];

  if (tid == 0u) {{
    upd[0] = -INFINITY;
    upd[1] = 0.0f;
  }}
  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
    *(device float4*)(oh + d) = float4(0.0f);
  }}
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) oh[d] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = tid; d < {hd}u; d += {tg}u) sq[d] = qh[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile0 = 0u; tile0 < T; tile0 += {tile}u) {{
    uint tc = min((uint){tile}u, T - tile0);
    // Load prior tokens from Q4; current token stays f32 (not yet in cache).
    load_q4_tile(k, start + tile0, tc, row_bytes, tid, {tg}u, k_tile);
    load_q4_tile(v, start + tile0, tc, row_bytes, tid, {tg}u, v_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wave = 0u; wave < tc; wave += {nsg}u) {{
      uint kv_off = wave + sgid;
      if (kv_off < tc) {{
        uint pos = start + tile0 + kv_off;
        float partial = 0.0f;
        if (pos == cur) {{
          for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u) {{
            float4 qv = *(threadgroup float4*)(sq + d);
            float4 kv = *(device const float4*)(k_new + d);
            partial += dot(qv, kv);
          }}
          for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u) {{
            partial += sq[d] * k_new[d];
          }}
        }} else {{
          threadgroup const uchar* kr = k_tile + kv_off * row_bytes;
          for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u) {{
            float4 qv = *(threadgroup float4*)(sq + d);
            partial += dot(qv, q4_0_read4_row(kr, d));
          }}
          for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u) {{
            partial += sq[d] * q4_0_read_row(kr, d);
          }}
        }}
        partial = simd_sum(partial);
        if (lane == 0u) scores[kv_off] = partial;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
      float m_old = upd[0];
      float l_old = upd[1];
      float m_new = m_old;
      for (uint i = 0u; i < tc; i++) m_new = max(m_new, scores[i]);
      float tile_sum = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        float e = exp(scores[i] - m_new);
        exps[i] = e;
        tile_sum += e;
      }}
      float scale = exp(m_old - m_new);
      float l_new = l_old * scale + tile_sum;
      upd[0] = m_new;
      upd[1] = l_new;
      upd[2] = l_new > 0.0f ? (l_old * scale) / l_new : 0.0f;
      upd[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float old_factor = upd[2];
    float inv_l = upd[3];
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
      float4 ov = *(device float4*)(oh + d);
      float4 acc = float4(0.0f);
      for (uint i = 0u; i < tc; i++) {{
        uint pos = start + tile0 + i;
        if (pos == cur) {{
          acc += exps[i] * *(device const float4*)(v_new + d);
        }} else {{
          threadgroup const uchar* vr = v_tile + i * row_bytes;
          acc += exps[i] * q4_0_read4_row(vr, d);
        }}
      }}
      ov = ov * old_factor + acc * inv_l;
      *(device float4*)(oh + d) = ov;
    }}
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
      float acc = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        uint pos = start + tile0 + i;
        if (pos == cur) {{
          acc += exps[i] * v_new[d];
        }} else {{
          threadgroup const uchar* vr = v_tile + i * row_bytes;
          acc += exps[i] * q4_0_read_row(vr, d);
        }}
      }}
      oh[d] = oh[d] * old_factor + acc * inv_l;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  if (head == 0u && cur < {max_t}u) {{
    for (uint g = tid; g < {groups}u; g += {tg}u) {{
      q4_0_append_group(k_new, k, cur, row_bytes, g);
      q4_0_append_group(v_new, v, cur, row_bytes, g);
    }}
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 7,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: n_q.max(1),
            tg,
        },
    })
}


fn lower_attn_gqa_q4_q_fused(
    graph: &Graph,
    out: TensorId,
    _q: TensorId,
    _q_norm: TensorId,
    _cos_sin: TensorId,
    _k_cache: TensorId,
    _v_cache: TensorId,
    _meta: TensorId,
    n_q: usize,
    hd: usize,
    max_t: usize,
    eps: f32,
) -> Result<MetalKernelSource, CodegenError> {
    if hd % 32 != 0 {
        return Err(CodegenError::Msg(
            "AttnGqaQ4QFused hd must be multiple of 32".into(),
        ));
    }
    let (shape, dtype) = graph.shape_dtype(out)?;
    let name = format!("k_attn_q4qf_qprep_{}", out.0);
    let tg = 256u64;
    let tile = 256usize;
    let simd = 32u64;
    let nsg = tg / simd;
    let groups = hd / 32;
    let row_bytes = groups * 18;
    let _ = groups;
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline float q4_0_read(device const uchar* cache, uint pos, uint row_bytes, uint d) {{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = pos * row_bytes + g * 18u;
  float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
  device const uchar* qs = cache + offset + 2u;
  if (e < 16u) return float(int(qs[e] & 0xFu) - 8) * scale;
  return float(int(qs[e - 16u] >> 4) - 8) * scale;
}}

inline float4 q4_0_read4(device const uchar* cache, uint pos, uint row_bytes, uint d) {{
  uint g = d / 32u;
  uint e = d % 32u;
  uint offset = pos * row_bytes + g * 18u;
  float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
  device const uchar* qs = cache + offset + 2u;
  if (e + 3u < 16u) {{
    return float4(
      float(int(qs[e + 0u] & 0xFu) - 8) * scale,
      float(int(qs[e + 1u] & 0xFu) - 8) * scale,
      float(int(qs[e + 2u] & 0xFu) - 8) * scale,
      float(int(qs[e + 3u] & 0xFu) - 8) * scale
    );
  }}
  if (e >= 16u && e + 3u < 32u) {{
    uint b = e - 16u;
    return float4(
      float(int(qs[b + 0u] >> 4) - 8) * scale,
      float(int(qs[b + 1u] >> 4) - 8) * scale,
      float(int(qs[b + 2u] >> 4) - 8) * scale,
      float(int(qs[b + 3u] >> 4) - 8) * scale
    );
  }}
  return float4(
    q4_0_read(cache, pos, row_bytes, d),
    q4_0_read(cache, pos, row_bytes, d + 1u),
    q4_0_read(cache, pos, row_bytes, d + 2u),
    q4_0_read(cache, pos, row_bytes, d + 3u)
  );
}}

kernel void {name}(
  device const float* q_raw [[buffer(0)]],
  device const float* q_norm [[buffer(1)]],
  device const float* cos_sin [[buffer(2)]],
  device const uchar* k_cache [[buffer(3)]],
  device const uchar* v_cache [[buffer(4)]],
  device const uint* meta [[buffer(5)]],
  device float* out [[buffer(6)]],
  uint head [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint sgid [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {{
  if (head >= {n_q}u) return;
  uint T = meta[0];
  uint start = meta[1];
  if (T == 0u) return;
  if (T > {max_t}u) T = {max_t}u;
  device float* oh = out + head * {hd}u;
  const uint row_bytes = {row_bytes}u;
  const uint half_d = {hd}u / 2u;

  threadgroup float sq[{hd}];
  threadgroup float scores[{tile}];
  threadgroup float exps[{tile}];
  threadgroup float upd[4];
  threadgroup float tmp[8];

  {{
    device const float* qr = q_raw + head * {hd}u;
    for (uint d = tid; d < {hd}u; d += {tg}u) sq[d] = qr[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float partial = 0.0f;
    for (uint d = tid; d < {hd}u; d += {tg}u) partial += sq[d] * sq[d];
    partial = simd_sum(partial);
    if (lane == 0u) tmp[sgid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
      float ss = 0.0f;
      for (uint i = 0u; i < {nsg}u; i++) ss += tmp[i];
      tmp[0] = rsqrt(ss / {hd}.0f + {eps:?}f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = tmp[0];
    for (uint d = tid; d < {hd}u; d += {tg}u) sq[d] = sq[d] * inv * q_norm[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < half_d; i += {tg}u) {{
      float c = cos_sin[i];
      float s = cos_sin[half_d + i];
      float u = sq[i];
      float vv = sq[i + half_d];
      sq[i] = u * c - vv * s;
      sq[i + half_d] = u * s + vv * c;
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {{
    upd[0] = -INFINITY;
    upd[1] = 0.0f;
  }}
  for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u)
    *(device float4*)(oh + d) = float4(0.0f);
  for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) oh[d] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile0 = 0u; tile0 < T; tile0 += {tile}u) {{
    uint tc = min((uint){tile}u, T - tile0);
    for (uint wave = 0u; wave < tc; wave += {nsg}u) {{
      uint kv_off = wave + sgid;
      if (kv_off < tc) {{
        uint pos = start + tile0 + kv_off;
        float partial = 0.0f;
        for (uint d = lane * 4u; d + 3u < {hd}u; d += {simd}u * 4u)
          partial += dot(*(threadgroup float4*)(sq + d), q4_0_read4(k_cache, pos, row_bytes, d));
        for (uint d = ({hd}u / 4u) * 4u + lane; d < {hd}u; d += {simd}u)
          partial += sq[d] * q4_0_read(k_cache, pos, row_bytes, d);
        partial = simd_sum(partial);
        if (lane == 0u) scores[kv_off] = partial;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
      float m_old = upd[0];
      float l_old = upd[1];
      float m_new = m_old;
      for (uint i = 0u; i < tc; i++) m_new = max(m_new, scores[i]);
      float tile_sum = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        float e = exp(scores[i] - m_new);
        exps[i] = e;
        tile_sum += e;
      }}
      float scale = exp(m_old - m_new);
      float l_new = l_old * scale + tile_sum;
      upd[0] = m_new;
      upd[1] = l_new;
      upd[2] = l_new > 0.0f ? (l_old * scale) / l_new : 0.0f;
      upd[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float old_factor = upd[2];
    float inv_l = upd[3];
    for (uint d = tid * 4u; d + 3u < {hd}u; d += {tg}u * 4u) {{
      float4 ov = *(device float4*)(oh + d);
      float4 acc = float4(0.0f);
      for (uint i = 0u; i < tc; i++) {{
        uint pos = start + tile0 + i;
        acc += exps[i] * q4_0_read4(v_cache, pos, row_bytes, d);
      }}
      *(device float4*)(oh + d) = ov * old_factor + acc * inv_l;
    }}
    for (uint d = ({hd}u / 4u) * 4u + tid; d < {hd}u; d += {tg}u) {{
      float acc = 0.0f;
      for (uint i = 0u; i < tc; i++) {{
        uint pos = start + tile0 + i;
        acc += exps[i] * q4_0_read(v_cache, pos, row_bytes, d);
      }}
      oh[d] = oh[d] * old_factor + acc * inv_l;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 6,
        out_shape: shape,
        out_dtype: dtype,
        launch: LaunchHint::RowsParallel {
            rows: n_q.max(1),
            tg,
        },
    })
}

fn lower_copy_slice(
    graph: &Graph,
    out: TensorId,
    _src: TensorId,
    src_off: usize,
    dst_off: usize,
    n: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let name = format!("k_copy_slice_{}", out.0);
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
        name,
        source,
        n_inputs: 1,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::Elementwise { n },
    })
}

fn lower_gather_q4k_row(
    _graph: &Graph,
    out: TensorId,
    _w: TensorId,
    _row_idx: TensorId,
    cols: usize,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let nb = cols / 256;
    let name = format!("k_gather_q4k_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline void get_scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {{
  if (j < 4u) {{
    d = q[j] & 63u;
    m = q[j + 4u] & 63u;
  }} else {{
    d = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u);
    m = (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u);
  }}
}}

kernel void {name}(
  device const uchar* A [[buffer(0)]],
  device const float* row_f [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint b [[thread_position_in_grid]]
) {{
  if (b >= {nb}u) return;
  uint row = (uint)row_f[0];
  constexpr uint QK = 256u;
  constexpr uint BPB = 144u;
  device const uchar* blk = A + row * {nb}u * BPB + b * BPB;
  float scale = {scale:?}f;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  uint is = 0u;
  uint qoff = 0u;
  uint yy = 0u;
  while (yy < QK) {{
    uchar sc1, m1, sc2, m2;
    get_scale_min_k4(is, scales, sc1, m1);
    get_scale_min_k4(is + 1u, scales, sc2, m2);
    float d1 = d * float(sc1);
    float mn1 = dmin * float(m1);
    float d2 = d * float(sc2);
    float mn2 = dmin * float(m2);
    uint ybase = b * QK + yy;
    for (uint l = 0u; l < 32u; l++) {{
      y[ybase + l] = scale * (d1 * float(qs[qoff + l] & 0x0Fu) - mn1);
    }}
    for (uint l = 0u; l < 32u; l++) {{
      y[ybase + 32u + l] = scale * (d2 * float(qs[qoff + l] >> 4u) - mn2);
    }}
    qoff += 32u;
    is += 2u;
    yy += 64u;
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: Shape(vec![cols]),
        out_dtype: DType::F32,
        launch: LaunchHint::Elementwise { n: nb.max(1) },
    })
}

fn lower_gather_q5k_row(
    _graph: &Graph,
    out: TensorId,
    _w: TensorId,
    _row_idx: TensorId,
    cols: usize,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let nb = cols / 256;
    let name = format!("k_gather_q5k_{}", out.0);
    let source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline void get_scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {{
  if (j < 4u) {{
    d = q[j] & 63u;
    m = q[j + 4u] & 63u;
  }} else {{
    d = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u);
    m = (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u);
  }}
}}

kernel void {name}(
  device const uchar* A [[buffer(0)]],
  device const float* row_f [[buffer(1)]],
  device float* y [[buffer(2)]],
  uint b [[thread_position_in_grid]]
) {{
  if (b >= {nb}u) return;
  uint row = (uint)row_f[0];
  constexpr uint QK = 256u;
  constexpr uint BPB = 176u;
  device const uchar* blk = A + row * {nb}u * BPB + b * BPB;
  float scale = {scale:?}f;
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qh = blk + 16;
  device const uchar* qs = blk + 48;
  uint is = 0u;
  uint qoff = 0u;
  uint yy = 0u;
  uchar u1 = 1u;
  uchar u2 = 2u;
  while (yy < QK) {{
    uchar sc1, m1, sc2, m2;
    get_scale_min_k4(is, scales, sc1, m1);
    get_scale_min_k4(is + 1u, scales, sc2, m2);
    float d1 = d * float(sc1);
    float mn1 = dmin * float(m1);
    float d2 = d * float(sc2);
    float mn2 = dmin * float(m2);
    uint ybase = b * QK + yy;
    for (uint l = 0u; l < 32u; l++) {{
      float hi = (qh[l] & u1) ? 16.0f : 0.0f;
      y[ybase + l] = scale * (d1 * (float(qs[qoff + l] & 0x0Fu) + hi) - mn1);
    }}
    for (uint l = 0u; l < 32u; l++) {{
      float hi = (qh[l] & u2) ? 16.0f : 0.0f;
      y[ybase + 32u + l] = scale * (d2 * (float(qs[qoff + l] >> 4u) + hi) - mn2);
    }}
    qoff += 32u;
    is += 2u;
    yy += 64u;
    u1 <<= 2u;
    u2 <<= 2u;
  }}
}}
"#
    );
    Ok(MetalKernelSource {
        name,
        source,
        n_inputs: 2,
        out_shape: Shape(vec![cols]),
        out_dtype: DType::F32,
        launch: LaunchHint::Elementwise { n: nb.max(1) },
    })
}

fn lower_softcap_argmax(
    graph: &Graph,
    out: TensorId,
    x: TensorId,
    cap: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let n = graph.shape_dtype(x)?.0.numel();
    let name = format!("k_sca_{}", out.0);
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
        name,
        source,
        n_inputs: 1,
        out_shape: graph.shape_dtype(out)?.0,
        out_dtype: DType::F32,
        launch: LaunchHint::RowsParallel { rows: 1, tg },
    })
}

/// Untuned vs candidate TG sizes for BEAM — returns candidates to time.
pub fn beam_tg_candidates(n_threads: usize) -> Vec<u64> {
    let mut v = vec![32, 64, 128, 256, 512];
    v.retain(|&tg| tg <= n_threads.max(1) as u64 || n_threads < 32);
    if v.is_empty() {
        v.push(64);
    }
    v
}

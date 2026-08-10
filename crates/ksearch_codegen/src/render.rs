//! Generic MSL renderer: Kernel IR AST → Metal. No named hand kernels.

use crate::{CodegenError, LaunchHint, MetalKernelSource};
use ksearch_ir::{BinOp, DType, KirExpr, KirLaunch, KirStmt, KernelIr, OptSchedule, UnaryOp};

pub fn render_msl(kir: &KernelIr, _sched: OptSchedule) -> Result<MetalKernelSource, CodegenError> {
    // Tinygrad-style: no hand q4k_load. Quant → float via Graph/CPU dequant, then F32 AST only.
    if body_needs_q4(&kir.body) {
        return Err(CodegenError::Msg(
            "render: Q4_K Load in AST — dequant to F32 first (tinygrad ggml_data_to_tensor style)"
                .into(),
        ));
    }
    let mut src = String::from("#include <metal_stdlib>\nusing namespace metal;\n\n");
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
            KirStmt::Store { idx, val, .. } | KirStmt::TgStore { idx, val, .. } => {
                infer_load_expr(idx, in_dt);
                infer_load_expr(val, in_dt);
            }
            KirStmt::TgDeclF32 { .. } | KirStmt::Barrier | KirStmt::ThreadgroupReduce { .. } => {}
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
            // B from threadgroup float — do not overwrite device buffer dtype from VecMulSum.
            if b_from_tg.is_none() {
                if (*b_buf as usize) < in_dt.len() {
                    in_dt[*b_buf as usize] = *dtype;
                }
            }
            infer_load_expr(a_idx, in_dt);
            infer_load_expr(b_idx, in_dt);
        }
        KirExpr::TgLoad { idx, .. } => infer_load_expr(idx, in_dt),
        KirExpr::CastU32ToF32(e) => infer_load_expr(e, in_dt),
        KirExpr::SimdSum(a) | KirExpr::Unary { a, .. } => infer_load_expr(a, in_dt),
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
            a_idx,
            b_idx,
            ..
        } => walk_expr_q4(a_idx, buf) || walk_expr_q4(b_idx, buf),
        KirExpr::TgLoad { idx, .. } => walk_expr_q4(idx, buf),
        KirExpr::CastU32ToF32(e) => walk_expr_q4(e, buf),
        KirExpr::SimdSum(a) | KirExpr::Unary { a, .. } => walk_expr_q4(a, buf),
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
            KirStmt::Store { idx, val, .. } | KirStmt::TgStore { idx, val, .. } => {
                if walk_expr_q4(idx, buf) || walk_expr_q4(val, buf) {
                    return true;
                }
            }
            KirStmt::TgDeclF32 { .. } | KirStmt::Barrier | KirStmt::ThreadgroupReduce { .. } => {}
        }
    }
    false
}

fn body_needs_q4(stmts: &[KirStmt]) -> bool {
    (0..8).any(|b| buf_is_q4(stmts, b))
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
                    s.push_str(&format!(
                        "{pad}threadgroup float red_{acc_id}[{tg}];\n\
                         {pad}red_{acc_id}[lid] = v{acc_id};\n\
                         {pad}threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}for (uint stride = {tg}u / 2u; stride > 0u; stride >>= 1u) {{\n\
                         {pad}  if (lid < stride) red_{acc_id}[lid] += red_{acc_id}[lid + stride];\n\
                         {pad}  threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                         {pad}}}\n\
                         {pad}v{acc_id} = red_{acc_id}[0];\n"
                    ));
                }
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
                other => {
                    return Err(CodegenError::Msg(format!(
                        "render load: dtype {other:?} — dequant to float first (tinygrad style)"
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
                    4 if *dtype == DType::F16 => format!(
                        "dot(float4(*(device const half4*)({a} + ({ai}))), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    2 if *dtype == DType::F16 => format!(
                        "(float((*(device const half2*)({a} + ({ai}))).x) * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).x + \
                         float((*(device const half2*)({a} + ({ai}))).y) * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).y)"
                    ),
                    1 if *dtype == DType::F16 => {
                        format!("(float({a}[{ai}]) * tg{tg_id}[{bi}])")
                    }
                    4 => format!(
                        "dot(*(device const float4*)({a} + ({ai})), *(threadgroup const float4*)(tg{tg_id} + ({bi})))"
                    ),
                    2 => format!(
                        "((*(device const float2*)({a} + ({ai}))).x * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).x + \
                         (*(device const float2*)({a} + ({ai}))).y * (*(threadgroup const float2*)(tg{tg_id} + ({bi}))).y)"
                    ),
                    1 => format!("{a}[{ai}] * tg{tg_id}[{bi}]"),
                    w => {
                        return Err(CodegenError::Msg(format!(
                            "render VecMulSum: unsupported width {w}"
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
                        return Err(CodegenError::Msg(format!(
                            "render VecMulSum: unsupported width {w}"
                        )))
                    }
                }
            }
        }
        KirExpr::SimdSum(a) => format!("simd_sum({})", emit_expr_ty(a, n_in, n_out, true, elem)?),
        KirExpr::Bin { op, a, b } => {
            if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul)
                && is_uintish(a)
                && is_uintish(b)
                && !as_float
            {
                let op_s = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    _ => unreachable!(),
                };
                format!(
                    "({} {} {})",
                    emit_expr_ty(a, n_in, n_out, false, elem)?,
                    op_s,
                    emit_expr_ty(b, n_in, n_out, false, elem)?
                )
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
        KirExpr::CmpGt { a, b } => format!(
            "(({}) > ({}))",
            emit_expr_ty(a, n_in, n_out, true, elem)?,
            emit_expr_ty(b, n_in, n_out, true, elem)?
        ),
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
    })
}

fn is_uintish(e: &KirExpr) -> bool {
    match e {
        KirExpr::Gid | KirExpr::Lid | KirExpr::ForVar(_) | KirExpr::ConstU32(_) | KirExpr::UVar(_) => {
            true
        }
        KirExpr::Bin {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul,
            a,
            b,
        } => is_uintish(a) && is_uintish(b),
        _ => false,
    }
}

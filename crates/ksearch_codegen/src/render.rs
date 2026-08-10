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
    let (params, launch, grid_n) = match &kir.launch {
        KirLaunch::Elementwise { n } => {
            let p = buffer_params(kir);
            (
                format!("{p}  uint gid [[thread_position_in_grid]]"),
                LaunchHint::Elementwise { n: (*n).max(1) },
                *n,
            )
        }
        KirLaunch::Rows { rows } => {
            let p = buffer_params(kir);
            (
                format!("{p}  uint gid [[thread_position_in_grid]]"),
                LaunchHint::Elementwise {
                    n: (*rows).max(1),
                },
                *rows,
            )
        }
    };
    let body = emit_stmts(&kir.body, 1, n_in)?;
    src.push_str(&format!(
        "kernel void {name}(\n{params}\n) {{\n  if (gid >= {grid_n}u) return;\n{body}}}\n",
        name = kir.name,
        params = params,
        grid_n = grid_n.max(1),
        body = body,
    ));
    Ok(MetalKernelSource {
        name: kir.name.clone(),
        source: src,
        n_inputs: kir.n_inputs,
        out_shape: kir.out_shape.clone(),
        out_dtype: kir.out_dtype,
        launch,
    })
}

fn buffer_params(kir: &KernelIr) -> String {
    let mut p = String::new();
    for i in 0..kir.n_inputs {
        p.push_str(&format!(
            "  device const float* in{i} [[buffer({i})]],\n"
        ));
    }
    p.push_str(&format!(
        "  device float* out [[buffer({})]],\n",
        kir.n_inputs
    ));
    p
}

fn buf_name(buf: u32, n_in: u32) -> String {
    if buf == n_in {
        "out".into()
    } else {
        format!("in{buf}")
    }
}

fn buf_is_q4(stmts: &[KirStmt], buf: u32) -> bool {
    fn walk(stmts: &[KirStmt], buf: u32) -> bool {
        for s in stmts {
            match s {
                KirStmt::For { body, .. } | KirStmt::If { body, .. } => {
                    if walk(body, buf) {
                        return true;
                    }
                }
                KirStmt::Let { expr, .. } | KirStmt::Assign { expr, .. } => {
                    if expr_q4(expr, buf) {
                        return true;
                    }
                }
                KirStmt::Store { idx, val, .. } => {
                    if expr_q4(idx, buf) || expr_q4(val, buf) {
                        return true;
                    }
                }
            }
        }
        false
    }
    fn expr_q4(e: &KirExpr, buf: u32) -> bool {
        match e {
            KirExpr::Load {
                buf: b,
                dtype: DType::Q4K,
                ..
            } if *b == buf => true,
            KirExpr::Load { idx, .. } => expr_q4(idx, buf),
            KirExpr::Bin { a, b, .. } | KirExpr::CmpGt { a, b } => {
                expr_q4(a, buf) || expr_q4(b, buf)
            }
            KirExpr::Unary { a, .. } => expr_q4(a, buf),
            _ => false,
        }
    }
    walk(stmts, buf)
}

fn body_needs_q4(stmts: &[KirStmt]) -> bool {
    (0..8).any(|b| buf_is_q4(stmts, b))
}

fn emit_stmts(stmts: &[KirStmt], indent: usize, n_in: u32) -> Result<String, CodegenError> {
    let pad = "  ".repeat(indent);
    let mut s = String::new();
    for st in stmts {
        match st {
            KirStmt::For { id, n, body } => {
                s.push_str(&format!(
                    "{pad}for (uint f{id} = 0u; f{id} < {n}u; f{id}++) {{\n"
                ));
                s.push_str(&emit_stmts(body, indent + 1, n_in)?);
                s.push_str(&format!("{pad}}}\n"));
            }
            KirStmt::Let { id, expr } => {
                s.push_str(&format!(
                    "{pad}float v{id} = {};\n",
                    emit_expr_ty(expr, n_in, true)?
                ));
            }
            KirStmt::Assign { id, expr } => {
                s.push_str(&format!(
                    "{pad}v{id} = {};\n",
                    emit_expr_ty(expr, n_in, true)?
                ));
            }
            KirStmt::Store { buf, idx, val } => {
                let idx_s = emit_expr_ty(idx, n_in, false)?;
                let idx_u = if is_uintish(idx) {
                    idx_s
                } else {
                    format!("uint({idx_s})")
                };
                s.push_str(&format!(
                    "{pad}{}[{}] = {};\n",
                    buf_name(*buf, n_in),
                    idx_u,
                    emit_expr_ty(val, n_in, true)?
                ));
            }
            KirStmt::If { cond, body } => {
                s.push_str(&format!(
                    "{pad}if ({}) {{\n",
                    emit_expr_ty(cond, n_in, true)?
                ));
                s.push_str(&emit_stmts(body, indent + 1, n_in)?);
                s.push_str(&format!("{pad}}}\n"));
            }
        }
    }
    Ok(s)
}

fn emit_expr_ty(e: &KirExpr, n_in: u32, as_float: bool) -> Result<String, CodegenError> {
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
        KirExpr::ForVar(id) => {
            if as_float {
                format!("float(f{id})")
            } else {
                format!("f{id}")
            }
        }
        KirExpr::Var(id) => format!("v{id}"),
        KirExpr::Load { buf, idx, dtype } => {
            let idx_s = emit_expr_ty(idx, n_in, false)?;
            let idx_u = if is_uintish(idx) {
                idx_s
            } else {
                format!("uint({idx_s})")
            };
            match dtype {
                DType::F32 => format!("{}[{}]", buf_name(*buf, n_in), idx_u),
                other => {
                    return Err(CodegenError::Msg(format!(
                        "render load: dtype {other:?} — dequant to F32 first (tinygrad style)"
                    )))
                }
            }
        }
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
                    emit_expr_ty(a, n_in, false)?,
                    op_s,
                    emit_expr_ty(b, n_in, false)?
                )
            } else {
                let (as_, bs) = (
                    emit_expr_ty(a, n_in, true)?,
                    emit_expr_ty(b, n_in, true)?,
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
            let x = emit_expr_ty(a, n_in, true)?;
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
            emit_expr_ty(a, n_in, true)?,
            emit_expr_ty(b, n_in, true)?
        ),
    })
}

fn is_uintish(e: &KirExpr) -> bool {
    match e {
        KirExpr::Gid | KirExpr::ForVar(_) | KirExpr::ConstU32(_) => true,
        KirExpr::Bin {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul,
            a,
            b,
        } => is_uintish(a) && is_uintish(b),
        _ => false,
    }
}

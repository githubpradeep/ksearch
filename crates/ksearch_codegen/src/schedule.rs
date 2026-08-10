//! Graph → ScheduledKernel: invent CALL/kernel boundaries from primitives + FuseHint.

use crate::rewrite;
use crate::CodegenError;
use ksearch_ir::{ElemExpr, FuseHint, Graph, KernelKind, Op, ScheduledKernel, TensorId};

pub fn is_primitive_region(graph: &Graph, out: TensorId) -> Result<bool, CodegenError> {
    if graph.fuse_hint(out).is_some() {
        return Ok(true);
    }
    let node = graph.node(out)?;
    Ok(matches!(
        &node.op,
        Op::Add { .. }
            | Op::Mul { .. }
            | Op::SumReduce { .. }
            | Op::MulBroadcastRow { .. }
            | Op::ScaleConst { .. }
            | Op::CopySlice { .. }
            | Op::Rsqrt { .. }
            | Op::Tanh { .. }
            | Op::Exp { .. }
            | Op::Const { .. }
            | Op::Expand { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            | Op::MaxReduce { .. }
            | Op::Call { .. }
    ))
}

fn sk_from_hint(graph: &Graph, out: TensorId, hint: &FuseHint) -> Result<ScheduledKernel, CodegenError> {
    Ok(match hint {
        FuseHint::RmsNorm { n, eps, x, w } => ScheduledKernel {
            name: format!("k_rmsnorm_{}", out.0),
            inputs: vec![*x, *w],
            output: out,
            kind: KernelKind::RmsNorm {
                n: *n,
                eps: *eps,
                x: *x,
                w: *w,
            },
        },
        FuseHint::RmsNormAdd {
            n,
            eps,
            x,
            w,
            residual,
        } => ScheduledKernel {
            name: format!("k_rms_add_{}", out.0),
            inputs: vec![*x, *w, *residual],
            output: out,
            kind: KernelKind::RmsNormAdd {
                n: *n,
                eps: *eps,
                x: *x,
                w: *w,
                residual: *residual,
            },
        },
        FuseHint::RmsNormAddScale {
            n,
            eps,
            scale,
            x,
            w,
            residual,
        } => ScheduledKernel {
            name: format!("k_rms_add_sc_{}", out.0),
            inputs: vec![*x, *w, *residual],
            output: out,
            kind: KernelKind::RmsNormAddScale {
                n: *n,
                eps: *eps,
                scale: *scale,
                x: *x,
                w: *w,
                residual: *residual,
            },
        },
        FuseHint::RmsNormPerHead {
            n_heads,
            hd,
            eps,
            with_weight,
            x,
            w,
        } => ScheduledKernel {
            name: format!("k_rms_ph_{}", out.0),
            inputs: vec![*x, *w],
            output: out,
            kind: KernelKind::RmsNormPerHead {
                n_heads: *n_heads,
                hd: *hd,
                eps: *eps,
                with_weight: *with_weight,
                x: *x,
                w: *w,
            },
        },
        FuseHint::RmsNormPerHeadRope {
            n_heads,
            hd,
            eps,
            with_weight,
            x,
            w,
            cos_sin,
        } => ScheduledKernel {
            name: format!("k_rms_ph_rope_{}", out.0),
            inputs: vec![*x, *w, *cos_sin],
            output: out,
            kind: KernelKind::RmsNormPerHeadRope {
                n_heads: *n_heads,
                hd: *hd,
                eps: *eps,
                with_weight: *with_weight,
                x: *x,
                w: *w,
                cos_sin: *cos_sin,
            },
        },
        FuseHint::Rope {
            n_heads,
            hd,
            x,
            cos_sin,
        } => ScheduledKernel {
            name: format!("k_rope_{}", out.0),
            inputs: vec![*x, *cos_sin],
            output: out,
            kind: KernelKind::Rope {
                n_heads: *n_heads,
                hd: *hd,
                x: *x,
                cos_sin: *cos_sin,
            },
        },
        FuseHint::CopyScale {
            src_off,
            dst_off,
            n,
            scale,
            src,
        } => ScheduledKernel {
            name: format!("k_csl_sc_{}", out.0),
            inputs: vec![*src],
            output: out,
            kind: KernelKind::CopyScale {
                src_off: *src_off,
                dst_off: *dst_off,
                n: *n,
                scale: *scale,
                src: *src,
            },
        },
        FuseHint::GeluMul {
            n,
            up_off,
            gate,
            up,
        } => ScheduledKernel {
            name: format!("k_gelu_mul_{}", out.0),
            inputs: vec![*gate, *up],
            output: out,
            kind: KernelKind::GeluMul {
                n: *n,
                up_off: *up_off,
                gate: *gate,
                up: *up,
            },
        },
        FuseHint::MatvecGateUpGelu {
            rows,
            cols,
            gate,
            up,
            x,
        } => {
            let wd = graph.shape_dtype(*gate)?.1;
            ScheduledKernel {
                name: format!("k_mv_gate_up_gelu_{}", out.0),
                inputs: vec![*gate, *up, *x],
                output: out,
                kind: KernelKind::MatvecGateUpGelu {
                    rows: *rows,
                    cols: *cols,
                    gate: *gate,
                    up: *up,
                    x: *x,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::MatvecQkv {
            q_rows,
            kv_rows,
            cols,
            wq,
            wk,
            wv,
            x,
        } => {
            let wd = graph.shape_dtype(*wq)?.1;
            ScheduledKernel {
                name: format!("k_mv_qkv_{}", out.0),
                inputs: vec![*wq, *wk, *wv, *x],
                output: out,
                kind: KernelKind::MatvecQkv {
                    q_rows: *q_rows,
                    kv_rows: *kv_rows,
                    cols: *cols,
                    wq: *wq,
                    wk: *wk,
                    wv: *wv,
                    x: *x,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::RmsNormMatvec {
            n,
            eps,
            rows,
            cols,
            x,
            w_norm,
            w_mat,
        } => {
            let wd = graph.shape_dtype(*w_mat)?.1;
            ScheduledKernel {
                name: format!("k_rms_mv_{}", out.0),
                inputs: vec![*w_mat, *x, *w_norm],
                output: out,
                kind: KernelKind::RmsNormMatvec {
                    n: *n,
                    eps: *eps,
                    rows: *rows,
                    cols: *cols,
                    x: *x,
                    w_norm: *w_norm,
                    w_mat: *w_mat,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::RmsNormMatvecGateUpGelu {
            n,
            eps,
            rows,
            cols,
            x,
            w_norm,
            gate,
            up,
        } => {
            let wd = graph.shape_dtype(*gate)?.1;
            ScheduledKernel {
                name: format!("k_rms_mv_gate_up_gelu_{}", out.0),
                inputs: vec![*gate, *up, *x, *w_norm],
                output: out,
                kind: KernelKind::RmsNormMatvecGateUpGelu {
                    n: *n,
                    eps: *eps,
                    rows: *rows,
                    cols: *cols,
                    x: *x,
                    w_norm: *w_norm,
                    gate: *gate,
                    up: *up,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::RmsNormMatvecQkv {
            n,
            eps,
            q_rows,
            kv_rows,
            cols,
            x,
            w_norm,
            wq,
            wk,
            wv,
        } => {
            let wd = graph.shape_dtype(*wq)?.1;
            ScheduledKernel {
                name: format!("k_rms_mv_qkv_{}", out.0),
                inputs: vec![*wq, *wk, *wv, *x, *w_norm],
                output: out,
                kind: KernelKind::RmsNormMatvecQkv {
                    n: *n,
                    eps: *eps,
                    q_rows: *q_rows,
                    kv_rows: *kv_rows,
                    cols: *cols,
                    x: *x,
                    w_norm: *w_norm,
                    wq: *wq,
                    wk: *wk,
                    wv: *wv,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::SdpaNaive {
            n_q,
            hd,
            max_t,
            q,
            k,
            v,
            meta,
        } => ScheduledKernel {
            name: format!("k_sdpa_naive_{}", out.0),
            inputs: vec![*q, *k, *v, *meta],
            output: out,
            kind: KernelKind::SdpaNaive {
                n_q: *n_q,
                hd: *hd,
                max_t: *max_t,
                q: *q,
                k: *k,
                v: *v,
                meta: *meta,
            },
        },
        FuseHint::SoftcapArgmax { n, cap, x } => ScheduledKernel {
            name: format!("k_sca_{}", out.0),
            inputs: vec![*x],
            output: out,
            kind: KernelKind::SoftcapArgmax {
                n: *n,
                cap: *cap,
                x: *x,
            },
        },
    })
}

/// Schedule one kernel covering `out` (fuse hints / matvec / elemwise / copy).
pub fn schedule(graph: &Graph, out: TensorId) -> Result<Vec<ScheduledKernel>, CodegenError> {
    let _ = rewrite::validate_q4_matvec_pattern(graph, out)?;

    if let Some(hint) = graph.fuse_hint(out) {
        return Ok(vec![sk_from_hint(graph, out, hint)?]);
    }

    let node = graph.node(out)?;
    let sk = match &node.op {
        Op::Call { .. } => {
            return Err(CodegenError::Msg(
                "schedule: Call without FuseHint".into(),
            ));
        }
        Op::ScaleConst { x, scale } => {
            let n = graph.shape_dtype(out)?.0.numel();
            ScheduledKernel {
                name: format!("k_scale_{}", out.0),
                inputs: vec![*x],
                output: out,
                kind: KernelKind::Elementwise {
                    n,
                    expr: ElemExpr::Scale(Box::new(ElemExpr::Load(0)), *scale),
                },
            }
        }
        Op::Add { .. } | Op::Mul { .. } => {
            let mut inputs = Vec::new();
            let expr = build_elem_expr(graph, out, &mut inputs)?;
            let n = graph.shape_dtype(out)?.0.numel();
            ScheduledKernel {
                name: format!("k_elem_{}", out.0),
                inputs,
                output: out,
                kind: KernelKind::Elementwise { n, expr },
            }
        }
        Op::Reshape { inp, .. } | Op::Permute { inp, .. } | Op::Expand { inp, .. } => {
            // Movement-only: schedule as identity copy of underlying buffer view via scale 1.
            // Real views are handled when fused into producers; standalone → copy via Load.
            let n = graph.shape_dtype(out)?.0.numel();
            ScheduledKernel {
                name: format!("k_move_{}", out.0),
                inputs: vec![*inp],
                output: out,
                kind: KernelKind::Elementwise {
                    n,
                    expr: ElemExpr::Load(0),
                },
            }
        }
        Op::SumReduce { inp, axis } => {
            let last = graph.node(*inp)?.shape.rank().saturating_sub(1);
            if *axis != last {
                return Err(CodegenError::Msg(
                    "schedule: only last-axis SumReduce supported".into(),
                ));
            }
            if let Op::MulBroadcastRow { left, row } = &graph.node(*inp)?.op {
                let (ms, wd) = graph.shape_dtype(*left)?;
                if ms.rank() != 2 {
                    return Err(CodegenError::Msg("matvec matrix must be rank-2".into()));
                }
                ScheduledKernel {
                    name: format!("k_matvec_{}", out.0),
                    inputs: vec![*left, *row],
                    output: out,
                    kind: KernelKind::Matvec {
                        rows: ms.0[0],
                        cols: ms.0[1],
                        matrix: *left,
                        vector: *row,
                        weight_dtype: wd,
                    },
                }
            } else {
                let (is, _) = graph.shape_dtype(*inp)?;
                if is.rank() != 2 {
                    return Err(CodegenError::Msg("sum_last input must be rank-2".into()));
                }
                ScheduledKernel {
                    name: format!("k_sumlast_{}", out.0),
                    inputs: vec![*inp],
                    output: out,
                    kind: KernelKind::SumLast {
                        rows: is.0[0],
                        cols: is.0[1],
                        inp: *inp,
                    },
                }
            }
        }
        Op::CopySlice {
            src,
            src_off,
            dst_off,
            n,
        } => ScheduledKernel {
            name: format!("k_copy_slice_{}", out.0),
            inputs: vec![*src],
            output: out,
            kind: KernelKind::CopySlice {
                src_off: *src_off,
                dst_off: *dst_off,
                n: *n,
                src: *src,
            },
        },
        other => {
            return Err(CodegenError::Msg(format!(
                "schedule: not a Thesis A primitive region ({other:?})"
            )));
        }
    };
    Ok(vec![sk])
}

pub fn lower_kernel(
    graph: &Graph,
    sk: &ScheduledKernel,
    sched: ksearch_ir::OptSchedule,
) -> Result<ksearch_ir::KernelIr, CodegenError> {
    let (out_shape, out_dtype) = graph.shape_dtype(sk.output)?;
    crate::lower::lower_to_kir(sk, out_shape, out_dtype, sched)
}

fn build_elem_expr(
    graph: &Graph,
    id: TensorId,
    inputs: &mut Vec<TensorId>,
) -> Result<ElemExpr, CodegenError> {
    let node = graph.node(id)?;
    match &node.op {
        Op::Input { .. } => {
            let bi = inputs.iter().position(|t| *t == id).unwrap_or_else(|| {
                inputs.push(id);
                inputs.len() - 1
            });
            Ok(ElemExpr::Load(bi))
        }
        Op::Add { a, b } => Ok(ElemExpr::Add(
            Box::new(build_elem_expr(graph, *a, inputs)?),
            Box::new(build_elem_expr(graph, *b, inputs)?),
        )),
        Op::Mul { a, b } => Ok(ElemExpr::Mul(
            Box::new(build_elem_expr(graph, *a, inputs)?),
            Box::new(build_elem_expr(graph, *b, inputs)?),
        )),
        Op::ScaleConst { x, scale } => Ok(ElemExpr::Scale(
            Box::new(build_elem_expr(graph, *x, inputs)?),
            *scale,
        )),
        _ => Err(CodegenError::Msg(
            "elementwise region may only contain Add/Mul/ScaleConst/Input".into(),
        )),
    }
}

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
        FuseHint::RmsNormAddThenRmsNorm {
            n,
            eps,
            y,
            w_post,
            residual,
            w_ffn,
        } => ScheduledKernel {
            name: format!("k_rms_add_then_rms_{}", out.0),
            inputs: vec![*y, *w_post, *residual, *w_ffn],
            output: out,
            kind: KernelKind::RmsNormAddThenRmsNorm {
                n: *n,
                eps: *eps,
                y: *y,
                w_post: *w_post,
                residual: *residual,
                w_ffn: *w_ffn,
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
        FuseHint::RmsNormPerHeadRopeQ40 {
            n_heads,
            hd,
            eps,
            with_weight,
            x,
            w,
            cos_sin,
        } => ScheduledKernel {
            name: format!("k_rms_ph_rope_q40_{}", out.0),
            inputs: vec![*x, *w, *cos_sin],
            output: out,
            kind: KernelKind::RmsNormPerHeadRopeQ40 {
                n_heads: *n_heads,
                hd: *hd,
                eps: *eps,
                with_weight: *with_weight,
                x: *x,
                w: *w,
                cos_sin: *cos_sin,
            },
        },
        FuseHint::RmsNormPerHeadQ40 {
            n_heads,
            hd,
            eps,
            with_weight,
            x,
            w,
        } => ScheduledKernel {
            name: format!("k_rms_ph_q40_{}", out.0),
            inputs: vec![*x, *w],
            output: out,
            kind: KernelKind::RmsNormPerHeadQ40 {
                n_heads: *n_heads,
                hd: *hd,
                eps: *eps,
                with_weight: *with_weight,
                x: *x,
                w: *w,
            },
        },
        FuseHint::RmsNormPerHeadQkvQ40 {
            n_q,
            n_kv,
            hd,
            eps,
            q,
            qw,
            cos_sin,
            k,
            kw,
            v,
        } => ScheduledKernel {
            name: format!("k_rms_ph_qkv_q40_{}", out.0),
            inputs: vec![*q, *qw, *cos_sin, *k, *kw, *v],
            output: out,
            kind: KernelKind::RmsNormPerHeadQkvQ40 {
                n_q: *n_q,
                n_kv: *n_kv,
                hd: *hd,
                eps: *eps,
                q: *q,
                qw: *qw,
                cos_sin: *cos_sin,
                k: *k,
                kw: *kw,
                v: *v,
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
            src_dtype,
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
                src_dtype: *src_dtype,
            },
        },
        FuseHint::CopyScaleIndexed {
            n,
            scale,
            src,
            idx,
            src_dtype,
        } => ScheduledKernel {
            name: format!("k_csl_sc_idx_{}", out.0),
            inputs: vec![*src, *idx],
            output: out,
            kind: KernelKind::CopyScaleIndexed {
                n: *n,
                scale: *scale,
                src: *src,
                idx: *idx,
                src_dtype: *src_dtype,
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
        FuseHint::MatvecGeluMul {
            rows,
            cols,
            ctx_off,
            w,
            x,
            ctx,
        } => {
            let wd = graph.shape_dtype(*w)?.1;
            ScheduledKernel {
                name: format!("k_mv_gelu_mul_{}", out.0),
                inputs: vec![*w, *x, *ctx],
                output: out,
                kind: KernelKind::MatvecGeluMul {
                    rows: *rows,
                    cols: *cols,
                    ctx_off: *ctx_off,
                    w: *w,
                    x: *x,
                    ctx: *ctx,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::MatvecGeluMulProjRmsAddScale {
            gate_rows,
            cols,
            proj_rows,
            ctx_off,
            eps,
            scale,
            w_gate,
            x,
            ctx,
            w_proj,
            w_norm,
            residual,
        } => {
            let wd = graph.shape_dtype(*w_gate)?.1;
            ScheduledKernel {
                name: format!("k_mv_gelu_proj_rms_{}", out.0),
                inputs: vec![*w_gate, *x, *ctx, *w_proj, *w_norm, *residual],
                output: out,
                kind: KernelKind::MatvecGeluMulProjRmsAddScale {
                    gate_rows: *gate_rows,
                    cols: *cols,
                    proj_rows: *proj_rows,
                    ctx_off: *ctx_off,
                    eps: *eps,
                    scale: *scale,
                    w_gate: *w_gate,
                    x: *x,
                    ctx: *ctx,
                    w_proj: *w_proj,
                    w_norm: *w_norm,
                    residual: *residual,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::MatvecRmsNormAdd {
            rows,
            cols,
            eps,
            w_mat,
            x,
            w_norm,
            residual,
        } => {
            let wd = graph.shape_dtype(*w_mat)?.1;
            ScheduledKernel {
                name: format!("k_mv_rms_add_{}", out.0),
                inputs: vec![*w_mat, *x, *w_norm, *residual],
                output: out,
                kind: KernelKind::MatvecRmsNormAdd {
                    rows: *rows,
                    cols: *cols,
                    eps: *eps,
                    w_mat: *w_mat,
                    x: *x,
                    w_norm: *w_norm,
                    residual: *residual,
                    weight_dtype: wd,
                },
            }
        }
        FuseHint::MatvecRmsNormAddScale {
            rows,
            cols,
            eps,
            scale,
            w_mat,
            x,
            w_norm,
            residual,
        } => {
            let wd = graph.shape_dtype(*w_mat)?.1;
            ScheduledKernel {
                name: format!("k_mv_rms_add_sc_{}", out.0),
                inputs: vec![*w_mat, *x, *w_norm, *residual],
                output: out,
                kind: KernelKind::MatvecRmsNormAddScale {
                    rows: *rows,
                    cols: *cols,
                    eps: *eps,
                    scale: *scale,
                    w_mat: *w_mat,
                    x: *x,
                    w_norm: *w_norm,
                    residual: *residual,
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
            let wq_dtype = graph.shape_dtype(*wq)?.1;
            let wk_dtype = graph.shape_dtype(*wk)?.1;
            let wv_dtype = graph.shape_dtype(*wv)?.1;
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
                    wq_dtype,
                    wk_dtype,
                    wv_dtype,
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
            let wq_dtype = graph.shape_dtype(*wq)?.1;
            let wk_dtype = graph.shape_dtype(*wk)?.1;
            let wv_dtype = graph.shape_dtype(*wv)?.1;
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
                    wq_dtype,
                    wk_dtype,
                    wv_dtype,
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
            kv_dtype,
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
                kv_dtype: *kv_dtype,
            },
        },
        FuseHint::SdpaMwgPart {
            n_q,
            hd,
            max_t,
            nwg,
            q,
            k,
            v,
            meta,
            kv_dtype,
        } => ScheduledKernel {
            name: format!("k_sdpa_mwg_part_{}", out.0),
            inputs: vec![*q, *k, *v, *meta],
            output: out,
            kind: KernelKind::SdpaMwgPart {
                n_q: *n_q,
                hd: *hd,
                max_t: *max_t,
                nwg: *nwg,
                q: *q,
                k: *k,
                v: *v,
                meta: *meta,
                kv_dtype: *kv_dtype,
            },
        },
        FuseHint::SdpaMwgReduce {
            n_q,
            hd,
            nwg,
            tmp,
        } => ScheduledKernel {
            name: format!("k_sdpa_mwg_reduce_{}", out.0),
            inputs: vec![*tmp],
            output: out,
            kind: KernelKind::SdpaMwgReduce {
                n_q: *n_q,
                hd: *hd,
                nwg: *nwg,
                tmp: *tmp,
            },
        },
        FuseHint::QuantizeQ40 { n, src } => ScheduledKernel {
            name: format!("k_q40_{}", out.0),
            inputs: vec![*src],
            output: out,
            kind: KernelKind::QuantizeQ40 {
                n: *n,
                src: *src,
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
            // Fold scale(add/mul(...)) into one elementwise (PLE combine).
            let (expr, inputs) = match &graph.node(*x)?.op {
                Op::Add { .. } | Op::Mul { .. } => {
                    let mut inputs = Vec::new();
                    let inner = build_elem_expr(graph, *x, &mut inputs)?;
                    (
                        ElemExpr::Scale(Box::new(inner), *scale),
                        inputs,
                    )
                }
                _ => (
                    ElemExpr::Scale(Box::new(ElemExpr::Load(0)), *scale),
                    vec![*x],
                ),
            };
            ScheduledKernel {
                name: format!("k_scale_{}", out.0),
                inputs,
                output: out,
                kind: KernelKind::Elementwise { n, expr },
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

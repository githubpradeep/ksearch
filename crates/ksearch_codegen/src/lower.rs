//! Lower KernelKind → Kernel IR AST (no Metal strings).

use crate::CodegenError;
use ksearch_ir::{
    BinOp, DType, ElemExpr, KirExpr, KirLaunch, KirStmt, KernelIr, KernelKind, OptSchedule,
    ScheduledKernel, UnaryOp,
};

pub fn lower_to_kir(
    sk: &ScheduledKernel,
    out_shape: ksearch_ir::Shape,
    out_dtype: DType,
    sched: OptSchedule,
) -> Result<KernelIr, CodegenError> {
    let mut next = 0u32;
    let (launch, body) = match &sk.kind {
        KernelKind::Elementwise { n, expr } => (
            KirLaunch::Elementwise { n: *n },
            lower_elemwise(expr, sk.inputs.len() as u32, out_dtype),
        ),
        KernelKind::Matvec {
            rows,
            cols,
            weight_dtype,
            ..
        } => {
            if !weight_dtype.is_float() {
                return Err(CodegenError::Msg(
                    "lower matvec: only float weights (dequant first)".into(),
                ));
            }
            validate_sched(sched)?;
            (
                KirLaunch::RowsParallel {
                    rows: *rows,
                    tg: sched.tg,
                },
                lower_matvec(*cols, *weight_dtype, sched, &mut next)?,
            )
        }
        KernelKind::SumLast { rows, cols, .. } => {
            validate_sched(sched)?;
            (
                KirLaunch::RowsParallel {
                    rows: *rows,
                    tg: sched.tg,
                },
                lower_sum_last(*cols, out_dtype, sched, &mut next)?,
            )
        }
        KernelKind::CopySlice {
            src_off,
            dst_off,
            n,
            ..
        } => (
            KirLaunch::Elementwise { n: *n },
            lower_copy_slice(*src_off, *dst_off, sk.inputs.len() as u32, out_dtype),
        ),
        KernelKind::RmsNorm { n, eps, .. } => (
            KirLaunch::Elementwise { n: 1 },
            lower_rmsnorm(*n, *eps, false, None, 2, out_dtype, &mut next),
        ),
        KernelKind::RmsNormAdd { n, eps, .. } => (
            KirLaunch::Elementwise { n: 1 },
            lower_rmsnorm(*n, *eps, true, None, 3, out_dtype, &mut next),
        ),
        KernelKind::RmsNormAddScale {
            n, eps, scale, ..
        } => (
            KirLaunch::Elementwise { n: 1 },
            lower_rmsnorm(*n, *eps, true, Some(*scale), 3, out_dtype, &mut next),
        ),
        KernelKind::RmsNormPerHead {
            n_heads,
            hd,
            eps,
            with_weight,
            ..
        } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rmsnorm_per_head(*hd, *eps, *with_weight, 0, 1, 2, out_dtype, &mut next),
        ),
        KernelKind::RmsNormPerHeadRope {
            n_heads,
            hd,
            eps,
            with_weight,
            ..
        } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rmsnorm_per_head_rope(*hd, *eps, *with_weight, out_dtype, &mut next),
        ),
        KernelKind::Rope { n_heads, hd, .. } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rope(*hd, 0, 1, 2, out_dtype, &mut next),
        ),
        KernelKind::CopyScale {
            src_off,
            dst_off,
            n,
            scale,
            ..
        } => (
            KirLaunch::Elementwise { n: *n },
            lower_copy_scale(*src_off, *dst_off, *scale, sk.inputs.len() as u32, out_dtype),
        ),
        KernelKind::GeluMul { n, up_off, .. } => (
            KirLaunch::Elementwise { n: *n },
            lower_gelu_mul(*up_off, sk.inputs.len() as u32, out_dtype, &mut next),
        ),
        KernelKind::SdpaNaive { n_q, hd, max_t, .. } => (
            KirLaunch::Elementwise { n: *n_q },
            lower_sdpa(*hd, *max_t, out_dtype, &mut next),
        ),
        KernelKind::SoftcapArgmax { n, cap, .. } => (
            // Gemma prim uses F16 logits; out_dtype is F32 for the index.
            KirLaunch::Elementwise { n: 1 },
            lower_softcap_argmax(*n, *cap, DType::F16, &mut next),
        ),
    };
    Ok(KernelIr {
        name: sk.name.clone(),
        n_inputs: sk.inputs.len(),
        out_shape,
        out_dtype,
        launch,
        body,
        next_id: next,
    })
}

fn fresh(next: &mut u32) -> u32 {
    let i = *next;
    *next += 1;
    i
}
fn c(f: f32) -> KirExpr {
    KirExpr::ConstF32(f)
}
fn cu(u: u32) -> KirExpr {
    KirExpr::ConstU32(u)
}
fn gid() -> KirExpr {
    KirExpr::Gid
}
fn v(id: u32) -> KirExpr {
    KirExpr::Var(id)
}
fn uv(id: u32) -> KirExpr {
    KirExpr::UVar(id)
}
fn fv(id: u32) -> KirExpr {
    KirExpr::ForVar(id)
}
fn bin(op: BinOp, a: KirExpr, b: KirExpr) -> KirExpr {
    KirExpr::Bin {
        op,
        a: Box::new(a),
        b: Box::new(b),
    }
}
fn un(op: UnaryOp, a: KirExpr) -> KirExpr {
    KirExpr::Unary {
        op,
        a: Box::new(a),
    }
}
fn ld(buf: u32, idx: KirExpr, dtype: DType) -> KirExpr {
    KirExpr::Load {
        buf,
        idx: Box::new(idx),
        dtype,
    }
}
fn st(buf: u32, idx: KirExpr, val: KirExpr) -> KirStmt {
    KirStmt::Store { buf, idx, val }
}
fn gt(a: KirExpr, b: KirExpr) -> KirExpr {
    KirExpr::CmpGt {
        a: Box::new(a),
        b: Box::new(b),
    }
}

fn elem_kir(expr: &ElemExpr, g: &KirExpr, dt: DType) -> KirExpr {
    match expr {
        ElemExpr::Load(bi) => ld(*bi as u32, g.clone(), dt),
        ElemExpr::Add(a, b) => bin(BinOp::Add, elem_kir(a, g, dt), elem_kir(b, g, dt)),
        ElemExpr::Mul(a, b) => bin(BinOp::Mul, elem_kir(a, g, dt), elem_kir(b, g, dt)),
        ElemExpr::Scale(i, s) => bin(BinOp::Mul, c(*s), elem_kir(i, g, dt)),
    }
}

fn lid() -> KirExpr {
    KirExpr::Lid
}
fn eq(a: KirExpr, b: KirExpr) -> KirExpr {
    KirExpr::CmpEq {
        a: Box::new(a),
        b: Box::new(b),
    }
}
fn vec_dot(a_buf: u32, a_idx: KirExpr, b_buf: u32, b_idx: KirExpr, width: u32, dt: DType) -> KirExpr {
    KirExpr::VecMulSum {
        a_buf,
        a_idx: Box::new(a_idx),
        b_buf,
        b_idx: Box::new(b_idx),
        width,
        dtype: dt,
    }
}

fn validate_sched(sched: OptSchedule) -> Result<(), CodegenError> {
    if !matches!(sched.vec, 1 | 2 | 4) {
        return Err(CodegenError::Msg(format!("unsupported VEC={}", sched.vec)));
    }
    if !matches!(sched.unroll, 1 | 2 | 4 | 8) {
        return Err(CodegenError::Msg(format!(
            "unsupported UNROLL={}",
            sched.unroll
        )));
    }
    if sched.tg == 0 || sched.tg > 1024 {
        return Err(CodegenError::Msg(format!("unsupported TG={}", sched.tg)));
    }
    Ok(())
}

fn lower_elemwise(expr: &ElemExpr, n_in: u32, dt: DType) -> Vec<KirStmt> {
    vec![st(n_in, gid(), elem_kir(expr, &gid(), dt))]
}

/// Tinygrad-style LOCAL/UPCAST/UNROLL as AST: lid-strided K, vec loads, tg reduce.
fn lower_matvec(
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let tg = sched.tg;
    let vec = sched.vec;
    let unroll = sched.unroll;
    let stride = tg * u64::from(vec);
    let step = stride * u64::from(unroll);
    let last_u = u64::from(unroll.saturating_sub(1));
    let acc = fresh(next);
    let k = fresh(next);
    let row = gid();
    let base = fresh(next);

    let mut stmts = vec![
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row.clone(), cu(cols as u32)),
        },
        KirStmt::Let {
            id: acc,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: k,
            expr: bin(BinOp::Mul, lid(), cu(vec)),
        },
    ];

    // Main: for (; k + (unroll-1)*stride + vec - 1 < cols; k += step)
    let main_off = (last_u * stride + u64::from(vec) - 1) as u32;
    let mut main_body = Vec::new();
    for u in 0..unroll {
        let off = if u == 0 {
            uv(k)
        } else {
            bin(BinOp::Add, uv(k), cu((u as u64 * stride) as u32))
        };
        main_body.push(KirStmt::Assign {
            id: acc,
            expr: bin(
                BinOp::Add,
                v(acc),
                vec_dot(
                    0,
                    bin(BinOp::Add, uv(base), off.clone()),
                    1,
                    off,
                    vec,
                    weight_dtype,
                ),
            ),
        });
    }
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(main_off),
        bound: cu(cols as u32),
        step: cu(step as u32),
        body: main_body,
    });

    // Remainder vector: for (; k + vec - 1 < cols; k += stride)
    let rem_off = vec.saturating_sub(1);
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(rem_off),
        bound: cu(cols as u32),
        step: cu(stride as u32),
        body: vec![KirStmt::Assign {
            id: acc,
            expr: bin(
                BinOp::Add,
                v(acc),
                vec_dot(0, bin(BinOp::Add, uv(base), uv(k)), 1, uv(k), vec, weight_dtype),
            ),
        }],
    });

    // Scalar tail: for (; k < cols; k += tg)
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(0),
        bound: cu(cols as u32),
        step: cu(tg as u32),
        body: vec![KirStmt::Assign {
            id: acc,
            expr: bin(
                BinOp::Add,
                v(acc),
                bin(
                    BinOp::Mul,
                    ld(0, bin(BinOp::Add, uv(base), uv(k)), weight_dtype),
                    ld(1, uv(k), weight_dtype),
                ),
            ),
        }],
    });

    stmts.push(KirStmt::ThreadgroupReduce { acc_id: acc, tg });
    stmts.push(KirStmt::If {
        cond: eq(lid(), cu(0)),
        body: vec![st(2, row, v(acc))],
    });
    Ok(stmts)
}

fn lower_sum_last(
    cols: usize,
    dt: DType,
    sched: OptSchedule,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let tg = sched.tg;
    let acc = fresh(next);
    let k = fresh(next);
    let row = gid();
    let base = fresh(next);
    Ok(vec![
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row.clone(), cu(cols as u32)),
        },
        KirStmt::Let {
            id: acc,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: k,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: k,
            limit_off: cu(0),
            bound: cu(cols as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::Assign {
                id: acc,
                expr: bin(
                    BinOp::Add,
                    v(acc),
                    ld(0, bin(BinOp::Add, uv(base), uv(k)), dt),
                ),
            }],
        },
        KirStmt::ThreadgroupReduce { acc_id: acc, tg },
        KirStmt::If {
            cond: eq(lid(), cu(0)),
            body: vec![st(1, row, v(acc))],
        },
    ])
}

fn lower_copy_slice(src_off: usize, dst_off: usize, n_in: u32, dt: DType) -> Vec<KirStmt> {
    let g = gid();
    vec![st(
        n_in,
        bin(BinOp::Add, cu(dst_off as u32), g.clone()),
        ld(0, bin(BinOp::Add, cu(src_off as u32), g), dt),
    )]
}

fn lower_copy_scale(
    src_off: usize,
    dst_off: usize,
    scale: f32,
    n_in: u32,
    dt: DType,
) -> Vec<KirStmt> {
    let g = gid();
    vec![st(
        n_in,
        bin(BinOp::Add, cu(dst_off as u32), g.clone()),
        bin(
            BinOp::Mul,
            c(scale),
            ld(0, bin(BinOp::Add, cu(src_off as u32), g), dt),
        ),
    )]
}

fn lower_rmsnorm(
    n: usize,
    eps: f32,
    add_res: bool,
    scale: Option<f32>,
    out_buf: u32,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let j = fresh(next);
    let mut body = vec![
        KirStmt::Let {
            id: ss,
            expr: c(0.0),
        },
        KirStmt::For {
            id: i,
            n,
            body: vec![KirStmt::Assign {
                id: ss,
                expr: bin(
                    BinOp::Add,
                    v(ss),
                    bin(
                        BinOp::Mul,
                        ld(0, fv(i), dt),
                        ld(0, fv(i), dt),
                    ),
                ),
            }],
        },
        KirStmt::Let {
            id: inv,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss), c(n as f32)), c(eps)),
            ),
        },
    ];
    let mut val = bin(
        BinOp::Mul,
        bin(BinOp::Mul, ld(0, fv(j), dt), v(inv)),
        ld(1, fv(j), dt),
    );
    if add_res {
        val = bin(BinOp::Add, val, ld(2, fv(j), dt));
    }
    if let Some(sc) = scale {
        val = bin(BinOp::Mul, c(sc), val);
    }
    body.push(KirStmt::For {
        id: j,
        n,
        body: vec![st(out_buf, fv(j), val)],
    });
    body
}

fn lower_rmsnorm_per_head(
    hd: usize,
    eps: f32,
    with_weight: bool,
    x_buf: u32,
    w_buf: u32,
    out_buf: u32,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    let head = gid();
    let base = fresh(next);
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let j = fresh(next);
    let mut stmts = vec![
        KirStmt::Let {
            id: base,
            expr: bin(BinOp::Mul, head, cu(hd as u32)),
        },
        KirStmt::Let {
            id: ss,
            expr: c(0.0),
        },
        KirStmt::For {
            id: i,
            n: hd,
            body: vec![KirStmt::Assign {
                id: ss,
                expr: bin(
                    BinOp::Add,
                    v(ss),
                    bin(
                        BinOp::Mul,
                        ld(x_buf, bin(BinOp::Add, v(base), fv(i)), dt),
                        ld(x_buf, bin(BinOp::Add, v(base), fv(i)), dt),
                    ),
                ),
            }],
        },
        KirStmt::Let {
            id: inv,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss), c(hd as f32)), c(eps)),
            ),
        },
    ];
    let mut val = bin(
        BinOp::Mul,
        ld(x_buf, bin(BinOp::Add, v(base), fv(j)), dt),
        v(inv),
    );
    if with_weight {
        val = bin(BinOp::Mul, val, ld(w_buf, fv(j), dt));
    }
    stmts.push(KirStmt::For {
        id: j,
        n: hd,
        body: vec![st(out_buf, bin(BinOp::Add, v(base), fv(j)), val)],
    });
    stmts
}

fn lower_rmsnorm_per_head_rope(
    hd: usize,
    eps: f32,
    with_weight: bool,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    // inputs: x=0, w=1, cos_sin=2; out=3
    let mut body = lower_rmsnorm_per_head(hd, eps, with_weight, 0, 1, 3, dt, next);
    body.extend(lower_rope(hd, 3, 2, 3, dt, next));
    body
}

fn lower_rope(
    hd: usize,
    x_buf: u32,
    cos_buf: u32,
    out_buf: u32,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    let half = hd / 2;
    let base = fresh(next);
    let i = fresh(next);
    let u = fresh(next);
    let vv = fresh(next);
    let cos = fresh(next);
    let sin = fresh(next);
    vec![
        KirStmt::Let {
            id: base,
            expr: bin(BinOp::Mul, gid(), cu(hd as u32)),
        },
        KirStmt::For {
            id: i,
            n: half,
            body: vec![
                KirStmt::Let {
                    id: cos,
                    expr: ld(cos_buf, fv(i), dt),
                },
                KirStmt::Let {
                    id: sin,
                    expr: ld(cos_buf, bin(BinOp::Add, cu(half as u32), fv(i)), dt),
                },
                KirStmt::Let {
                    id: u,
                    expr: ld(x_buf, bin(BinOp::Add, v(base), fv(i)), dt),
                },
                KirStmt::Let {
                    id: vv,
                    expr: ld(
                        x_buf,
                        bin(BinOp::Add, v(base), bin(BinOp::Add, cu(half as u32), fv(i))),
                        dt,
                    ),
                },
                st(
                    out_buf,
                    bin(BinOp::Add, v(base), fv(i)),
                    bin(
                        BinOp::Sub,
                        bin(BinOp::Mul, v(u), v(cos)),
                        bin(BinOp::Mul, v(vv), v(sin)),
                    ),
                ),
                st(
                    out_buf,
                    bin(BinOp::Add, v(base), bin(BinOp::Add, cu(half as u32), fv(i))),
                    bin(
                        BinOp::Add,
                        bin(BinOp::Mul, v(u), v(sin)),
                        bin(BinOp::Mul, v(vv), v(cos)),
                    ),
                ),
            ],
        },
    ]
}

fn lower_gelu_mul(up_off: usize, n_in: u32, dt: DType, next: &mut u32) -> Vec<KirStmt> {
    let g = gid();
    let x = fresh(next);
    let ax = fresh(next);
    let u = fresh(next);
    let gelu = fresh(next);
    vec![
        KirStmt::Let {
            id: x,
            expr: ld(0, g.clone(), dt),
        },
        KirStmt::Let {
            id: ax,
            expr: bin(BinOp::Min, c(20.0), bin(BinOp::Max, c(-20.0), v(x))),
        },
        KirStmt::Let {
            id: u,
            expr: bin(
                BinOp::Mul,
                c(0.79788456),
                bin(
                    BinOp::Add,
                    v(ax),
                    bin(
                        BinOp::Mul,
                        c(0.044715),
                        bin(BinOp::Mul, bin(BinOp::Mul, v(ax), v(ax)), v(ax)),
                    ),
                ),
            ),
        },
        KirStmt::Let {
            id: gelu,
            expr: bin(
                BinOp::Mul,
                bin(BinOp::Mul, c(0.5), v(ax)),
                bin(BinOp::Add, c(1.0), un(UnaryOp::Tanh, v(u))),
            ),
        },
        st(
            n_in,
            g.clone(),
            bin(
                BinOp::Mul,
                v(gelu),
                ld(1, bin(BinOp::Add, cu(up_off as u32), g), dt),
            ),
        ),
    ]
}

fn dot_qkv(
    head: &KirExpr,
    start: &KirExpr,
    t: KirExpr,
    hd: usize,
    s_id: u32,
    d_id: u32,
    k_buf: u32,
    dt: DType,
) -> Vec<KirStmt> {
    vec![
        KirStmt::Let {
            id: s_id,
            expr: c(0.0),
        },
        KirStmt::For {
            id: d_id,
            n: hd,
            body: vec![KirStmt::Assign {
                id: s_id,
                expr: bin(
                    BinOp::Add,
                    v(s_id),
                    bin(
                        BinOp::Mul,
                        ld(
                            0,
                            bin(BinOp::Add, bin(BinOp::Mul, head.clone(), cu(hd as u32)), fv(d_id)),
                            dt,
                        ),
                        ld(
                            k_buf,
                            bin(
                                BinOp::Add,
                                bin(BinOp::Mul, bin(BinOp::Add, start.clone(), t), cu(hd as u32)),
                                fv(d_id),
                            ),
                            dt,
                        ),
                    ),
                ),
            }],
        },
    ]
}

fn lower_sdpa(hd: usize, max_t: usize, dt: DType, next: &mut u32) -> Vec<KirStmt> {
    let head = gid();
    let tlen = fresh(next);
    let start = fresh(next);
    let mmax = fresh(next);
    let t = fresh(next);
    let s = fresh(next);
    let d = fresh(next);
    let lsum = fresh(next);
    let inv_l = fresh(next);

    let mut stmts = vec![
        KirStmt::Let {
            id: tlen,
            expr: ld(3, cu(0), dt),
        },
        KirStmt::Let {
            id: start,
            expr: ld(3, cu(1), dt),
        },
        KirStmt::Let {
            id: mmax,
            expr: c(-1.0e30),
        },
    ];

    let mut pass1 = dot_qkv(&head, &v(start), fv(t), hd, s, d, 1, dt);
    pass1.push(KirStmt::If {
        cond: gt(v(tlen), fv(t)),
        body: vec![KirStmt::Assign {
            id: mmax,
            expr: bin(BinOp::Max, v(mmax), v(s)),
        }],
    });
    stmts.push(KirStmt::For {
        id: t,
        n: max_t,
        body: pass1,
    });

    stmts.push(KirStmt::Let {
        id: lsum,
        expr: c(0.0),
    });
    let t2 = fresh(next);
    let s2 = fresh(next);
    let d2 = fresh(next);
    let mut pass2 = dot_qkv(&head, &v(start), fv(t2), hd, s2, d2, 1, dt);
    pass2.push(KirStmt::If {
        cond: gt(v(tlen), fv(t2)),
        body: vec![KirStmt::Assign {
            id: lsum,
            expr: bin(
                BinOp::Add,
                v(lsum),
                un(UnaryOp::Exp, bin(BinOp::Sub, v(s2), v(mmax))),
            ),
        }],
    });
    stmts.push(KirStmt::For {
        id: t2,
        n: max_t,
        body: pass2,
    });
    stmts.push(KirStmt::Let {
        id: inv_l,
        expr: bin(BinOp::Div, c(1.0), v(lsum)),
    });

    let dout = fresh(next);
    stmts.push(KirStmt::For {
        id: dout,
        n: hd,
        body: vec![st(
            4,
            bin(BinOp::Add, bin(BinOp::Mul, head.clone(), cu(hd as u32)), fv(dout)),
            c(0.0),
        )],
    });

    let t3 = fresh(next);
    let s3 = fresh(next);
    let d3 = fresh(next);
    let p = fresh(next);
    let d4 = fresh(next);
    let mut pass3 = dot_qkv(&head, &v(start), fv(t3), hd, s3, d3, 1, dt);
    pass3.push(KirStmt::If {
        cond: gt(v(tlen), fv(t3)),
        body: {
            let mut b = vec![KirStmt::Let {
                id: p,
                expr: bin(
                    BinOp::Mul,
                    un(UnaryOp::Exp, bin(BinOp::Sub, v(s3), v(mmax))),
                    v(inv_l),
                ),
            }];
            b.push(KirStmt::For {
                id: d4,
                n: hd,
                body: {
                    let oi = bin(
                        BinOp::Add,
                        bin(BinOp::Mul, head.clone(), cu(hd as u32)),
                        fv(d4),
                    );
                    vec![st(
                        4,
                        oi.clone(),
                        bin(
                            BinOp::Add,
                            ld(4, oi, dt),
                            bin(
                                BinOp::Mul,
                                v(p),
                                ld(
                                    2,
                                    bin(
                                        BinOp::Add,
                                        bin(
                                            BinOp::Mul,
                                            bin(BinOp::Add, v(start), fv(t3)),
                                            cu(hd as u32),
                                        ),
                                        fv(d4),
                                    ),
                                    dt,
                                ),
                            ),
                        ),
                    )]
                },
            });
            b
        },
    });
    stmts.push(KirStmt::For {
        id: t3,
        n: max_t,
        body: pass3,
    });
    stmts
}

fn lower_softcap_argmax(n: usize, cap: f32, dt: DType, next: &mut u32) -> Vec<KirStmt> {
    let best = fresh(next);
    let bestv = fresh(next);
    let i = fresh(next);
    let val = fresh(next);
    vec![
        KirStmt::Let {
            id: best,
            expr: c(0.0),
        },
        KirStmt::Let {
            id: bestv,
            expr: c(-1.0e30),
        },
        KirStmt::For {
            id: i,
            n,
            body: vec![
                KirStmt::Let {
                    id: val,
                    expr: bin(
                        BinOp::Mul,
                        c(cap),
                        un(
                            UnaryOp::Tanh,
                            bin(BinOp::Div, ld(0, fv(i), dt), c(cap)),
                        ),
                    ),
                },
                KirStmt::If {
                    cond: gt(v(val), v(bestv)),
                    body: vec![
                        KirStmt::Assign {
                            id: bestv,
                            expr: v(val),
                        },
                        KirStmt::Assign {
                            id: best,
                            expr: fv(i),
                        },
                    ],
                },
            ],
        },
        st(1, cu(0), v(best)),
    ]
}

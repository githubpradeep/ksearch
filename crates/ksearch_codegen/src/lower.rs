//! Lower KernelKind → Kernel IR AST (no Metal strings).

use crate::CodegenError;
use ksearch_ir::{
    BinOp, DType, ElemExpr, KirExpr, KirLaunch, KirStmt, KernelIr, KernelKind, OptSchedule,
    ScheduledKernel, UnaryOp,
};

fn packed_k_quant(d: DType) -> bool {
    matches!(d, DType::Q4K | DType::Q6K)
}
fn allow_matvec_weight(d: DType) -> bool {
    d.is_float() || packed_k_quant(d)
}

pub fn lower_to_kir(
    sk: &ScheduledKernel,
    out_shape: ksearch_ir::Shape,
    out_dtype: DType,
    sched: OptSchedule,
) -> Result<KernelIr, CodegenError> {
    let mut next = 0u32;
    let (launch, body, n_outputs) = match &sk.kind {
        KernelKind::Elementwise { n, expr } => (
            KirLaunch::Elementwise { n: *n },
            lower_elemwise(expr, sk.inputs.len() as u32, out_dtype),
            1,
        ),
        KernelKind::Matvec {
            rows,
            cols,
            weight_dtype,
            ..
        } => {
            if !allow_matvec_weight(*weight_dtype) {
                return Err(CodegenError::Msg(
                    "lower matvec: weight dtype must be float, Q4K, or Q6K".into(),
                ));
            }
            validate_sched(sched)?;
            let sched = apply_matvec_sched(*weight_dtype, *cols, sched);
            (
                matvec_launch(*weight_dtype, *rows, sched),
                lower_matvec(*rows, *cols, *weight_dtype, sched, 1, 2, None, &mut next)?,
                1,
            )
        }
        KernelKind::MatvecGateUpGelu {
            rows,
            cols,
            weight_dtype,
            ..
        } => {
            if !allow_matvec_weight(*weight_dtype) {
                return Err(CodegenError::Msg(
                    "lower matvec_gate_up_gelu: weight dtype must be float or Q4K".into(),
                ));
            }
            validate_sched(sched)?;
            let mut sched = apply_matvec_sched(*weight_dtype, *cols, sched);
            // More SGs/TG for dual gate∥up (measure: nsg=4 vs ggml-default 2).
            if *weight_dtype == DType::Q4K && *cols % 256 == 0 {
                sched.nsg = 4;
                sched.tg = 128;
                sched.nr0 = 16;
            }
            (
                matvec_launch(*weight_dtype, *rows, sched),
                lower_matvec_gate_up_gelu(
                    *rows,
                    *cols,
                    *weight_dtype,
                    sched,
                    2,
                    3,
                    None,
                    &mut next,
                )?,
                1,
            )
        }
        KernelKind::MatvecQkv {
            q_rows,
            kv_rows,
            cols,
            wq_dtype,
            wk_dtype,
            wv_dtype,
            ..
        } => {
            if !(allow_matvec_weight(*wq_dtype) && allow_matvec_weight(*wk_dtype) && allow_matvec_weight(*wv_dtype)) {
                return Err(CodegenError::Msg(
                    "lower matvec_qkv: weight dtype must be float, Q4K, or Q6K".into(),
                ));
            }
            if !(*wq_dtype == *wk_dtype && *wk_dtype == *wv_dtype) {
                // Homogeneous only on HEAD lower; mixed uses wq for schedule, per-buf via separate path later.
            }
            let wd = *wq_dtype;
            validate_sched(sched)?;
            let sched = apply_matvec_sched(wd, *cols, sched);
            let max_rows = (*q_rows).max(*kv_rows);
            (
                matvec_launch(wd, max_rows, sched),
                lower_matvec_qkv(
                    *q_rows,
                    *kv_rows,
                    *cols,
                    *wq_dtype,
                    *wk_dtype,
                    *wv_dtype,
                    sched,
                    3,
                    4,
                    5,
                    6,
                    None,
                    &mut next,
                )?,
                3,
            )
        }
        KernelKind::RmsNormMatvec {
            eps,
            rows,
            cols,
            weight_dtype,
            ..
        } => {
            if !allow_matvec_weight(*weight_dtype) {
                return Err(CodegenError::Msg(
                    "lower rmsnorm_matvec: weight dtype must be float, Q4K, or Q6K".into(),
                ));
            }
            validate_sched(sched)?;
            let sched = apply_matvec_sched(*weight_dtype, *cols, sched);
            (
                matvec_launch(*weight_dtype, *rows, sched),
                lower_matvec(
                    *rows,
                    *cols,
                    *weight_dtype,
                    sched,
                    1,
                    3,
                    Some((2, *eps)),
                    &mut next,
                )?,
                1,
            )
        }
        KernelKind::RmsNormMatvecGateUpGelu {
            eps,
            rows,
            cols,
            weight_dtype,
            ..
        } => {
            if !allow_matvec_weight(*weight_dtype) {
                return Err(CodegenError::Msg(
                    "lower rmsnorm_matvec_gate_up_gelu: weight dtype must be float or Q4K".into(),
                ));
            }
            validate_sched(sched)?;
            let mut sched = apply_matvec_sched(*weight_dtype, *cols, sched);
            if *weight_dtype == DType::Q4K && *cols % 256 == 0 {
                sched.nsg = 4;
                sched.tg = 128;
                sched.nr0 = 16;
            }
            (
                matvec_launch(*weight_dtype, *rows, sched),
                lower_matvec_gate_up_gelu(
                    *rows,
                    *cols,
                    *weight_dtype,
                    sched,
                    2,
                    4,
                    Some((3, *eps)),
                    &mut next,
                )?,
                1,
            )
        }
        KernelKind::RmsNormMatvecQkv {
            eps,
            q_rows,
            kv_rows,
            cols,
            wq_dtype,
            wk_dtype,
            wv_dtype,
            ..
        } => {
            if !(allow_matvec_weight(*wq_dtype)
                && allow_matvec_weight(*wk_dtype)
                && allow_matvec_weight(*wv_dtype))
            {
                return Err(CodegenError::Msg(
                    "lower rmsnorm_matvec_qkv: weight dtype must be float, Q4K, or Q6K".into(),
                ));
            }
            let wd = *wq_dtype;
            validate_sched(sched)?;
            let sched = apply_matvec_sched(wd, *cols, sched);
            let max_rows = (*q_rows).max(*kv_rows);
            (
                matvec_launch(wd, max_rows, sched),
                lower_matvec_qkv(
                    *q_rows,
                    *kv_rows,
                    *cols,
                    *wq_dtype,
                    *wk_dtype,
                    *wv_dtype,
                    sched,
                    3,
                    5,
                    6,
                    7,
                    Some((4, *eps)),
                    &mut next,
                )?,
                3,
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
                1,
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
            1,
        ),
        KernelKind::RmsNorm { n, eps, .. } => (
            KirLaunch::RowsParallel {
                rows: 1,
                tg: 256,
            },
            lower_rmsnorm(*n, *eps, false, None, 2, out_dtype, 256, &mut next),
            1,
        ),
        KernelKind::RmsNormAdd { n, eps, .. } => (
            KirLaunch::RowsParallel {
                rows: 1,
                tg: 256,
            },
            lower_rmsnorm(*n, *eps, true, None, 3, out_dtype, 256, &mut next),
            1,
        ),
        KernelKind::RmsNormAddScale {
            n, eps, scale, ..
        } => (
            KirLaunch::RowsParallel {
                rows: 1,
                tg: 256,
            },
            lower_rmsnorm(*n, *eps, true, Some(*scale), 3, out_dtype, 256, &mut next),
            1,
        ),
        KernelKind::RmsNormAddThenRmsNorm { n, eps, .. } => {
            const TG: u64 = 256;
            (
                KirLaunch::RowsParallel {
                    rows: 1,
                    tg: TG,
                },
                lower_rmsnorm_add_then_rmsnorm(*n, *eps, out_dtype, TG, &mut next)?,
                2,
            )
        }
        KernelKind::RmsNormPerHead {
            n_heads,
            hd,
            eps,
            with_weight,
            ..
        } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rmsnorm_per_head(*hd, *eps, *with_weight, 0, 1, 2, out_dtype, &mut next),
            1,
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
            1,
        ),
        KernelKind::RmsNormPerHeadRopeQ40 {
            n_heads,
            hd,
            eps,
            with_weight,
            ..
        } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rmsnorm_per_head_rope_q40(*hd, *eps, *with_weight, DType::F16, &mut next),
            1,
        ),
        KernelKind::RmsNormPerHeadQ40 {
            n_heads,
            hd,
            eps,
            with_weight,
            ..
        } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rmsnorm_per_head_q40(*hd, *eps, *with_weight, DType::F16, &mut next),
            1,
        ),
        KernelKind::Rope { n_heads, hd, .. } => (
            KirLaunch::Elementwise { n: *n_heads },
            lower_rope(*hd, 0, 1, 2, out_dtype, &mut next),
            1,
        ),
        KernelKind::CopyScale {
            src_off,
            dst_off,
            n,
            scale,
            src_dtype,
            ..
        } => (
            KirLaunch::Elementwise { n: *n },
            // Load uses src_dtype (Q4K/Q6K expand); store uses Call out dtype (F16).
            lower_copy_scale(*src_off, *dst_off, *scale, sk.inputs.len() as u32, *src_dtype),
            1,
        ),
        KernelKind::GeluMul { n, up_off, .. } => (
            KirLaunch::Elementwise { n: *n },
            lower_gelu_mul(*up_off, sk.inputs.len() as u32, out_dtype, &mut next),
            1,
        ),
        KernelKind::SdpaNaive {
            n_q,
            hd,
            max_t,
            kv_dtype,
            ..
        } => {
            // One TG streams shared K/V once for all Q heads (MQA/GQA bandwidth).
            let tg = (*n_q as u64).saturating_mul(32).max(32);
            (
                KirLaunch::RowsParallel { rows: 1, tg },
                lower_sdpa_online(*n_q, *hd, *max_t, out_dtype, *kv_dtype, tg, &mut next),
                1,
            )
        },
        KernelKind::QuantizeQ40 { n, .. } => (
            KirLaunch::Elementwise { n: *n / 32 },
            lower_quantize_q40(sk.inputs.len() as u32, &mut next),
            1,
        ),

        KernelKind::MatvecGeluMul {
            rows,
            cols,
            ctx_off,
            weight_dtype,
            ..
        } => {
            validate_sched(sched)?;
            let sched = apply_matvec_sched(*weight_dtype, *cols, sched);
            let epi = MatvecEpi::GeluMulCtx {
                ctx_buf: 2,
                ctx_off: *ctx_off as u32,
                dt: DType::F16,
            };
            (
                matvec_launch(*weight_dtype, *rows, sched),
                lower_matvec_epi(
                    *rows, *cols, *weight_dtype, sched, 1, 3, None, epi, &mut next,
                )?,
                1,
            )
        }
        KernelKind::MatvecGeluMulProjRmsAddScale {
            gate_rows,
            cols,
            proj_rows,
            ctx_off,
            eps,
            scale,
            weight_dtype,
            ..
        } => {
            let (launch, body) = lower_ple_gelu_proj_rms_fused(
                *gate_rows,
                *cols,
                *proj_rows,
                *ctx_off,
                *eps,
                *scale,
                *weight_dtype,
                &mut next,
            )?;
            (launch, body, 1)
        }
        KernelKind::MatvecRmsNormAdd {
            rows,
            cols,
            eps,
            weight_dtype,
            ..
        } => {
            let (launch, body) =
                lower_matvec_rmsnorm_fused(*rows, *cols, *weight_dtype, *eps, None, &mut next)?;
            (launch, body, 1)
        }
        KernelKind::MatvecRmsNormAddScale {
            rows,
            cols,
            eps,
            scale,
            weight_dtype,
            ..
        } => {
            let (launch, body) = lower_matvec_rmsnorm_fused(
                *rows,
                *cols,
                *weight_dtype,
                *eps,
                Some(*scale),
                &mut next,
            )?;
            (launch, body, 1)
        }

        KernelKind::SoftcapArgmax { n, cap, .. } => (
            // One threadgroup, lid-strided scan + tg reduce (not 1 thread × vocab).
            KirLaunch::RowsParallel {
                rows: 1,
                tg: 1024,
            },
            lower_softcap_argmax(*n, *cap, DType::F16, 1024, &mut next),
            1,
        ),
    };
    Ok(KernelIr {
        name: sk.name.clone(),
        n_inputs: sk.inputs.len(),
        n_outputs,
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
fn vec_dot(
    a_buf: u32,
    a_idx: KirExpr,
    b_buf: u32,
    b_idx: KirExpr,
    width: u32,
    dt: DType,
    b_from_tg: Option<u32>,
) -> KirExpr {
    KirExpr::VecMulSum {
        a_buf,
        a_idx: Box::new(a_idx),
        b_buf,
        b_idx: Box::new(b_idx),
        width,
        dtype: dt,
        b_from_tg,
    }
}

fn tg_load(id: u32, idx: KirExpr) -> KirExpr {
    KirExpr::TgLoad {
        id,
        idx: Box::new(idx),
    }
}
fn th_load(id: u32, idx: KirExpr) -> KirExpr {
    KirExpr::ThreadLoad {
        id,
        idx: Box::new(idx),
    }
}
fn th_store(id: u32, idx: KirExpr, val: KirExpr) -> KirStmt {
    KirStmt::ThreadStore { id, idx, val }
}

/// Metal threadgroup mem budget for staging float `x`: cols * sizeof(float) ≤ 32 KiB.
const TG_X_BYTES_MAX: usize = 32768;

/// Cooperative RMSNorm into LOCAL: `tg0[i] = x[i] * rsqrt(mean(x²)+eps) * w[i]`.
/// All lanes hold the reduced sum-of-squares after [`KirStmt::ThreadgroupReduce`].
fn lower_stage_xhat_local(
    n: usize,
    eps: f32,
    x_buf: u32,
    w_buf: u32,
    tg: u64,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    let tg_id = 0u32;
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let i2 = fresh(next);
    vec![
        KirStmt::TgDeclF32 { id: tg_id, n },
        KirStmt::Let {
            id: ss,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: i,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::Assign {
                id: ss,
                expr: bin(
                    BinOp::Add,
                    v(ss),
                    bin(
                        BinOp::Mul,
                        ld(x_buf, uv(i), dt),
                        ld(x_buf, uv(i), dt),
                    ),
                ),
            }],
        },
        KirStmt::ThreadgroupReduce { acc_id: ss, tg },
        KirStmt::Let {
            id: inv,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss), c(n as f32)), c(eps)),
            ),
        },
        KirStmt::LetU32 {
            id: i2,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i2,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::TgStore {
                id: tg_id,
                idx: uv(i2),
                val: bin(
                    BinOp::Mul,
                    bin(BinOp::Mul, ld(x_buf, uv(i2), dt), v(inv)),
                    ld(w_buf, uv(i2), dt),
                ),
            }],
        },
        KirStmt::Barrier,
    ]
}

/// Stage raw `x` into LOCAL when it fits; or RMSNorm→`x_hat` when `rms` is set (requires LOCAL).
/// Activation loads use `dt` (F16 when weights are Q4K — never load x as Q4K).
fn stage_local_x(
    cols: usize,
    x_buf: u32,
    tg: u64,
    dt: DType,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<(Vec<KirStmt>, Option<u32>), CodegenError> {
    let local_ok = cols.saturating_mul(4) <= TG_X_BYTES_MAX;
    match rms {
        Some((w_norm, eps)) => {
            if !local_ok {
                return Err(CodegenError::Msg(
                    "rmsnorm+matvec fuse requires LOCAL staging of x_hat".into(),
                ));
            }
            Ok((
                lower_stage_xhat_local(cols, eps, x_buf, w_norm, tg, dt, next),
                Some(0),
            ))
        }
        None if local_ok => {
            let tg_id = 0u32;
            let i = fresh(next);
            Ok((
                vec![
                    KirStmt::TgDeclF32 { id: tg_id, n: cols },
                    KirStmt::LetU32 {
                        id: i,
                        expr: lid(),
                    },
                    KirStmt::ForRange {
                        id: i,
                        limit_off: cu(0),
                        bound: cu(cols as u32),
                        step: cu(tg as u32),
                        body: vec![KirStmt::TgStore {
                            id: tg_id,
                            idx: uv(i),
                            val: ld(x_buf, uv(i), dt),
                        }],
                    },
                    KirStmt::Barrier,
                ],
                Some(tg_id),
            ))
        }
        None => Ok((vec![], None)),
    }
}

/// Activations stay float; Q4K applies only to weight loads.
fn x_dtype(weight_dtype: DType) -> DType {
    match weight_dtype {
        DType::Q4K | DType::Q6K => DType::F16,
        d if d.is_float() => d,
        _ => DType::F16,
    }
}

fn apply_matvec_sched(weight_dtype: DType, cols: usize, mut sched: OptSchedule) -> OptSchedule {
    if matches!(weight_dtype, DType::Q4K | DType::Q6K) && cols % 256 == 0 {
        // ggml mul_vec_q{4,6}_K: TG=64 (2 SG), 8 rows/TG (4 per SG).
        sched.tg = 64;
        sched.nr0 = 8;
        sched.nsg = 2;
    }
    sched
}

fn matvec_launch(weight_dtype: DType, rows: usize, sched: OptSchedule) -> KirLaunch {
    let nr = sched.nr0.max(1) as usize;
    let n_tg = rows.saturating_add(nr - 1) / nr;
    // Q4/Q6 coop AST uses `sg = lid()/32` with ggml-style TG=(32, nsg).
    if matches!(weight_dtype, DType::Q4K | DType::Q6K) && sched.nsg >= 1 {
        KirLaunch::RowsParallelSg {
            rows: n_tg,
            nsg: sched.nsg.max(1) as u64,
        }
    } else {
        KirLaunch::RowsParallel {
            rows: n_tg,
            tg: sched.tg,
        }
    }
}

fn effective_vec(weight_dtype: DType, cols: usize, sched_vec: u32) -> u32 {
    match weight_dtype {
        DType::Q6K => 1,
        DType::Q4K => q4k_vec_width(cols),
        _ => sched_vec,
    }
}

fn effective_tg(weight_dtype: DType, cols: usize, sched_tg: u64) -> u64 {
    if matches!(weight_dtype, DType::Q4K | DType::Q6K) && cols % 256 == 0 {
        64
    } else {
        sched_tg
    }
}

fn q4k_vec_width(cols: usize) -> u32 {
    let _ = cols;
    32
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

#[derive(Clone, Copy)]
enum MatvecEpi {
    None,
    /// `out = gelu(acc) * ctx[ctx_off + row]` (PLE gate fuse).
    GeluMulCtx { ctx_buf: u32, ctx_off: u32, dt: DType },
}

fn emit_gelu_of(x: KirExpr, next: &mut u32) -> (Vec<KirStmt>, u32) {
    let ax = fresh(next);
    let u = fresh(next);
    let gelu = fresh(next);
    let stmts = vec![
        KirStmt::Let {
            id: ax,
            expr: bin(BinOp::Min, c(20.0), bin(BinOp::Max, c(-20.0), x)),
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
    ];
    (stmts, gelu)
}

fn store_row_with_epi(
    out_buf: u32,
    row: KirExpr,
    acc: u32,
    epi: MatvecEpi,
    next: &mut u32,
) -> Vec<KirStmt> {
    match epi {
        MatvecEpi::None => vec![st(out_buf, row, v(acc))],
        MatvecEpi::GeluMulCtx {
            ctx_buf,
            ctx_off,
            dt,
        } => {
            let (mut stmts, gelu) = emit_gelu_of(v(acc), next);
            let out = fresh(next);
            stmts.push(KirStmt::Let {
                id: out,
                expr: bin(
                    BinOp::Mul,
                    v(gelu),
                    ld(ctx_buf, bin(BinOp::Add, cu(ctx_off), row.clone()), dt),
                ),
            });
            stmts.push(st(out_buf, row, v(out)));
            stmts
        }
    }
}

/// Short-K single-TG fuse: `y = W@x` into LOCAL, then `out = scale*(residual + rms(y)*w)`.
/// Buffers: `0=W`, `1=x`, `2=w_norm`, `3=residual`, `4=out`. PLE proj is Q4×256×1536.
fn lower_matvec_rmsnorm_fused(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    eps: f32,
    scale: Option<f32>,
    next: &mut u32,
) -> Result<(KirLaunch, Vec<KirStmt>), CodegenError> {
    const TG: u64 = 256;
    let y_bytes = rows.saturating_mul(4);
    let x_bytes = cols.saturating_mul(4);
    if y_bytes.saturating_add(x_bytes) > TG_X_BYTES_MAX {
        return Err(CodegenError::Msg(format!(
            "matvec+rmsnorm fuse: LOCAL y+x too large ({rows}x{cols})"
        )));
    }
    if weight_dtype != DType::Q4K || cols % 256 != 0 {
        return Err(CodegenError::Msg(format!(
            "matvec+rmsnorm fuse: need Q4K cols%256==0 (got {weight_dtype:?} cols={cols})"
        )));
    }

    let (mut stmts, tg_x) = stage_local_x(cols, 1, TG, DType::F16, None, next)?;
    let tg_x = tg_x.ok_or_else(|| {
        CodegenError::Msg("matvec+rmsnorm fuse requires LOCAL staging of x".into())
    })?;
    let tg_y = 1u32;
    stmts.push(KirStmt::TgDeclF32 { id: tg_y, n: rows });

    // Per-lane rows: full Q4K row dots against LOCAL x (no TG reduce — one thread owns the row).
    let row = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: row,
        expr: lid(),
    });
    let acc = fresh(next);
    let base = fresh(next);
    let k = fresh(next);
    let mv_body = vec![
        KirStmt::Let {
            id: acc,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, uv(row), cu(cols as u32)),
        },
        KirStmt::LetU32 {
            id: k,
            expr: cu(0),
        },
        KirStmt::ForRange {
            id: k,
            limit_off: cu(0),
            bound: cu(cols as u32),
            step: cu(256),
            body: vec![KirStmt::Assign {
                id: acc,
                expr: bin(
                    BinOp::Add,
                    v(acc),
                    vec_dot(
                        0,
                        bin(BinOp::Add, uv(base), uv(k)),
                        1,
                        uv(k),
                        256,
                        DType::Q4K,
                        Some(tg_x),
                    ),
                ),
            }],
        },
        KirStmt::TgStore {
            id: tg_y,
            idx: uv(row),
            val: v(acc),
        },
    ];
    stmts.push(KirStmt::ForRange {
        id: row,
        limit_off: cu(0),
        bound: cu(rows as u32),
        step: cu(TG as u32),
        body: mv_body,
    });
    stmts.push(KirStmt::Barrier);

    // RMS over LOCAL y, then scale*(residual + y*inv*w).
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let j = fresh(next);
    stmts.push(KirStmt::Let {
        id: ss,
        expr: c(0.0),
    });
    stmts.push(KirStmt::LetU32 {
        id: i,
        expr: lid(),
    });
    stmts.push(KirStmt::ForRange {
        id: i,
        limit_off: cu(0),
        bound: cu(rows as u32),
        step: cu(TG as u32),
        body: vec![KirStmt::Assign {
            id: ss,
            expr: bin(
                BinOp::Add,
                v(ss),
                bin(BinOp::Mul, tg_load(tg_y, uv(i)), tg_load(tg_y, uv(i))),
            ),
        }],
    });
    stmts.push(KirStmt::ThreadgroupReduce { acc_id: ss, tg: TG });
    stmts.push(KirStmt::Let {
        id: inv,
        expr: un(
            UnaryOp::Rsqrt,
            bin(BinOp::Add, bin(BinOp::Div, v(ss), c(rows as f32)), c(eps)),
        ),
    });
    stmts.push(KirStmt::LetU32 {
        id: j,
        expr: lid(),
    });
    let mut val = bin(
        BinOp::Mul,
        bin(BinOp::Mul, tg_load(tg_y, uv(j)), v(inv)),
        ld(2, uv(j), DType::F16),
    );
    val = bin(BinOp::Add, val, ld(3, uv(j), DType::F16));
    if let Some(sc) = scale {
        val = bin(BinOp::Mul, c(sc), val);
    }
    stmts.push(KirStmt::ForRange {
        id: j,
        limit_off: cu(0),
        bound: cu(rows as u32),
        step: cu(TG as u32),
        body: vec![st(4, uv(j), val)],
    });

    Ok((
        KirLaunch::RowsParallel {
            rows: 1,
            tg: TG,
        },
        stmts,
    ))
}

/// Full PLE one TG: gate gelu*ctx → proj → rmsnorm_add_scale.
/// Buffers: `0=Wg`, `1=x`, `2=ctx`, `3=Wp`, `4=w_norm`, `5=residual`, `6=out`.
/// LOCAL: x[cols] + u[gate_rows] + y[proj_rows].
fn lower_ple_gelu_proj_rms_fused(
    gate_rows: usize,
    cols: usize,
    proj_rows: usize,
    ctx_off: usize,
    eps: f32,
    scale: f32,
    weight_dtype: DType,
    next: &mut u32,
) -> Result<(KirLaunch, Vec<KirStmt>), CodegenError> {
    const TG: u64 = 256;
    if weight_dtype != DType::Q4K || cols % 256 != 0 || gate_rows % 256 != 0 {
        return Err(CodegenError::Msg(format!(
            "PLE full fuse: need Q4K, cols%256==0, gate_rows%256==0 (wd={weight_dtype:?} cols={cols} gate={gate_rows})"
        )));
    }
    let local_elems = cols.saturating_add(gate_rows).saturating_add(proj_rows);
    if local_elems.saturating_mul(4) > TG_X_BYTES_MAX {
        return Err(CodegenError::Msg(format!(
            "PLE full fuse: LOCAL too large ({local_elems} floats)"
        )));
    }

    let (mut stmts, tg_x) = stage_local_x(cols, 1, TG, DType::F16, None, next)?;
    let tg_x = tg_x.ok_or_else(|| CodegenError::Msg("PLE fuse needs LOCAL x".into()))?;
    let tg_u = 1u32;
    let tg_y = 2u32;
    stmts.push(KirStmt::TgDeclF32 {
        id: tg_u,
        n: gate_rows,
    });
    stmts.push(KirStmt::TgDeclF32 {
        id: tg_y,
        n: proj_rows,
    });

    // Phase 1: u[r] = gelu(Wg[r]·x) * ctx[ctx_off+r]
    let row = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: row,
        expr: lid(),
    });
    let acc = fresh(next);
    let base = fresh(next);
    let k = fresh(next);
    let (gelu_stmts, gelu_id) = emit_gelu_of(v(acc), next);
    let out_u = fresh(next);
    let mut gate_body = vec![
        KirStmt::Let {
            id: acc,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, uv(row), cu(cols as u32)),
        },
        KirStmt::LetU32 {
            id: k,
            expr: cu(0),
        },
        KirStmt::ForRange {
            id: k,
            limit_off: cu(0),
            bound: cu(cols as u32),
            step: cu(256),
            body: vec![KirStmt::Assign {
                id: acc,
                expr: bin(
                    BinOp::Add,
                    v(acc),
                    vec_dot(
                        0,
                        bin(BinOp::Add, uv(base), uv(k)),
                        1,
                        uv(k),
                        256,
                        DType::Q4K,
                        Some(tg_x),
                    ),
                ),
            }],
        },
    ];
    gate_body.extend(gelu_stmts);
    gate_body.push(KirStmt::Let {
        id: out_u,
        expr: bin(
            BinOp::Mul,
            v(gelu_id),
            ld(
                2,
                bin(BinOp::Add, cu(ctx_off as u32), uv(row)),
                DType::F16,
            ),
        ),
    });
    gate_body.push(KirStmt::TgStore {
        id: tg_u,
        idx: uv(row),
        val: v(out_u),
    });
    stmts.push(KirStmt::ForRange {
        id: row,
        limit_off: cu(0),
        bound: cu(gate_rows as u32),
        step: cu(TG as u32),
        body: gate_body,
    });
    stmts.push(KirStmt::Barrier);

    // Phase 2: y[r] = Wp[r]·u  (proj cols = gate_rows)
    let row2 = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: row2,
        expr: lid(),
    });
    let acc2 = fresh(next);
    let base2 = fresh(next);
    let k2 = fresh(next);
    let proj_body = vec![
        KirStmt::Let {
            id: acc2,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: base2,
            expr: bin(BinOp::Mul, uv(row2), cu(gate_rows as u32)),
        },
        KirStmt::LetU32 {
            id: k2,
            expr: cu(0),
        },
        KirStmt::ForRange {
            id: k2,
            limit_off: cu(0),
            bound: cu(gate_rows as u32),
            step: cu(256),
            body: vec![KirStmt::Assign {
                id: acc2,
                expr: bin(
                    BinOp::Add,
                    v(acc2),
                    vec_dot(
                        3,
                        bin(BinOp::Add, uv(base2), uv(k2)),
                        1, // unused when b_from_tg set
                        uv(k2),
                        256,
                        DType::Q4K,
                        Some(tg_u),
                    ),
                ),
            }],
        },
        KirStmt::TgStore {
            id: tg_y,
            idx: uv(row2),
            val: v(acc2),
        },
    ];
    stmts.push(KirStmt::ForRange {
        id: row2,
        limit_off: cu(0),
        bound: cu(proj_rows as u32),
        step: cu(TG as u32),
        body: proj_body,
    });
    stmts.push(KirStmt::Barrier);

    // Phase 3: rmsnorm_add_scale on LOCAL y
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let j = fresh(next);
    stmts.push(KirStmt::Let {
        id: ss,
        expr: c(0.0),
    });
    stmts.push(KirStmt::LetU32 {
        id: i,
        expr: lid(),
    });
    stmts.push(KirStmt::ForRange {
        id: i,
        limit_off: cu(0),
        bound: cu(proj_rows as u32),
        step: cu(TG as u32),
        body: vec![KirStmt::Assign {
            id: ss,
            expr: bin(
                BinOp::Add,
                v(ss),
                bin(BinOp::Mul, tg_load(tg_y, uv(i)), tg_load(tg_y, uv(i))),
            ),
        }],
    });
    stmts.push(KirStmt::ThreadgroupReduce { acc_id: ss, tg: TG });
    stmts.push(KirStmt::Let {
        id: inv,
        expr: un(
            UnaryOp::Rsqrt,
            bin(
                BinOp::Add,
                bin(BinOp::Div, v(ss), c(proj_rows as f32)),
                c(eps),
            ),
        ),
    });
    stmts.push(KirStmt::LetU32 {
        id: j,
        expr: lid(),
    });
    let val = bin(
        BinOp::Mul,
        c(scale),
        bin(
            BinOp::Add,
            ld(5, uv(j), DType::F16),
            bin(
                BinOp::Mul,
                bin(BinOp::Mul, tg_load(tg_y, uv(j)), v(inv)),
                ld(4, uv(j), DType::F16),
            ),
        ),
    );
    stmts.push(KirStmt::ForRange {
        id: j,
        limit_off: cu(0),
        bound: cu(proj_rows as u32),
        step: cu(TG as u32),
        body: vec![st(6, uv(j), val)],
    });

    Ok((
        KirLaunch::RowsParallel {
            rows: 1,
            tg: TG,
        },
        stmts,
    ))
}

/// Tinygrad-style LOCAL/UPCAST/UNROLL as AST: lid-strided K, vec loads, tg reduce.
/// `nr = max(1, sched.nr0)` output rows per TG share one LOCAL `x` (or `x_hat`) stage.
/// K is outer so all NR rows reuse each weight/x chunk (oracle-style x amortization).
/// Buffers: `0=W`, `x_buf=x`, `out_buf=out`; optional `rms=(w_norm_buf, eps)` stages x_hat.
fn lower_matvec(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    lower_matvec_epi(
        rows,
        cols,
        weight_dtype,
        sched,
        x_buf,
        out_buf,
        rms,
        MatvecEpi::None,
        next,
    )
}

fn lower_matvec_epi(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    epi: MatvecEpi,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    if weight_dtype == DType::Q4K && cols % 256 == 0 {
        match lower_matvec_q4k_coop(rows, cols, sched, x_buf, out_buf, rms, epi, next) {
            Ok(s) => return Ok(s),
            Err(_) => {
                // Fall back if LOCAL x cannot be staged (cols too large for TG mem).
            }
        }
    }
    if weight_dtype == DType::Q6K && cols % 256 == 0 {
        match lower_matvec_q6k_coop(rows, cols, sched, x_buf, out_buf, rms, epi, next) {
            Ok(s) => return Ok(s),
            Err(_) => {}
        }
    }
    lower_matvec_generic(rows, cols, weight_dtype, sched, x_buf, out_buf, rms, epi, next)
}

/// ggml `mul_vec_q4_K`-shaped tiling: 2 simdgroups/TG, each owns `nr` rows;
/// lanes cooperate on Q4_K superblocks via [`KirExpr::Q4kCoopFrag`].
fn lower_matvec_q4k_coop(
    rows: usize,
    cols: usize,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    epi: MatvecEpi,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let nsg = sched.nsg.max(1);
    let tg = (nsg as u64) * 32;
    // Force 4 rows/SG (ggml KQ_NR0) so nr4 y-amortized expand applies.
    let nr = 4u32;
    let nb = (cols / 256) as u32;
    let exact_rows = rows % ((nsg * nr) as usize) == 0;

    // Device-x (oracle-style): stream device half x unless fused rms needs LOCAL x_hat.
    let prefer_local = false;
    let (mut stmts, tg_x) = if rms.is_some() || prefer_local {
        let (s, tg) = stage_local_x(cols, x_buf, tg, DType::F16, rms, next)?;
        if rms.is_some() {
            let tg = tg.ok_or_else(|| {
                CodegenError::Msg("q4k coop+rmsnorm requires LOCAL staging of x_hat".into())
            })?;
            (s, Some(tg))
        } else {
            (s, tg)
        }
    } else {
        (vec![], None)
    };

    let lane = fresh(next);
    let sg = fresh(next);
    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: sg,
        expr: bin(BinOp::Div, lid(), cu(32)),
    });
    stmts.push(KirStmt::LetU32 {
        id: lane,
        expr: bin(BinOp::Sub, lid(), bin(BinOp::Mul, uv(sg), cu(32))),
    });
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(
            BinOp::Mul,
            bin(BinOp::Add, bin(BinOp::Mul, gid(), cu(nsg)), uv(sg)),
            cu(nr),
        ),
    });
    let ix = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: ix,
        expr: bin(BinOp::Div, uv(lane), cu(8)),
    });

    let mut accs = Vec::with_capacity(nr as usize);
    let mut bases = Vec::with_capacity(nr as usize);
    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = fresh(next);
        let base = fresh(next);
        stmts.push(KirStmt::Let {
            id: acc,
            expr: c(0.0),
        });
        stmts.push(KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row, cu(cols as u32)),
        });
        accs.push(acc);
        bases.push(base);
    }

    let ib = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: ib,
        expr: uv(ix),
    });
    let mut body = Vec::new();
    if nr == 4 {
        let nr4 = KirStmt::Q4kCoopNr4 {
            w_buf: 0,
            row0_base: uv(bases[0]),
            cols: cols as u32,
            ib: uv(ib),
            b_from_tg: tg_x,
            x_buf,
            lane: uv(lane),
            acc_ids: [accs[0], accs[1], accs[2], accs[3]],
        };
        if exact_rows {
            body.push(nr4);
        } else {
            let last = bin(BinOp::Add, uv(first), cu(3));
            body.push(KirStmt::If {
                cond: gt(cu(rows as u32), last.clone()),
                body: vec![nr4],
            });
            let mut partial = Vec::new();
            for r in 0..nr {
                let row = if r == 0 {
                    uv(first)
                } else {
                    bin(BinOp::Add, uv(first), cu(r))
                };
                let acc = accs[r as usize];
                let base = bases[r as usize];
                partial.push(KirStmt::If {
                    cond: gt(cu(rows as u32), row),
                    body: vec![KirStmt::Assign {
                        id: acc,
                        expr: bin(
                            BinOp::Add,
                            v(acc),
                            KirExpr::Q4kCoopFrag {
                                w_buf: 0,
                                row_base: Box::new(uv(base)),
                                cols: cols as u32,
                                ib: Box::new(uv(ib)),
                                b_from_tg: tg_x,
                                x_buf,
                                lane: Box::new(uv(lane)),
                            },
                        ),
                    }],
                });
            }
            let rows_m1 = (rows as u32).saturating_sub(1);
            body.push(KirStmt::If {
                cond: gt(last, cu(rows_m1)),
                body: partial,
            });
        }
    } else {
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc = accs[r as usize];
            let base = bases[r as usize];
            body.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![KirStmt::Assign {
                    id: acc,
                    expr: bin(
                        BinOp::Add,
                        v(acc),
                        KirExpr::Q4kCoopFrag {
                            w_buf: 0,
                            row_base: Box::new(uv(base)),
                            cols: cols as u32,
                            ib: Box::new(uv(ib)),
                            b_from_tg: tg_x,
                            x_buf,
                            lane: Box::new(uv(lane)),
                        },
                    ),
                }],
            });
        }
    }
    stmts.push(KirStmt::ForRange {
        id: ib,
        limit_off: cu(0),
        bound: cu(nb),
        step: cu(4),
        body,
    });

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = accs[r as usize];
        let mut reduce_store = vec![KirStmt::Assign {
            id: acc,
            expr: KirExpr::SimdSum(Box::new(v(acc))),
        }];
        let store_body = store_row_with_epi(out_buf, row.clone(), acc, epi, next);
        reduce_store.push(KirStmt::If {
            cond: eq(uv(lane), cu(0)),
            body: store_body,
        });
        if exact_rows {
            stmts.extend(reduce_store);
        } else {
            stmts.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: reduce_store,
            });
        }
    }
    Ok(stmts)
}


/// ggml `mul_vec_q6_K`-shaped tiling: 2 simdgroups/TG, lanes stripe by 2.
fn lower_matvec_q6k_coop(
    rows: usize,
    cols: usize,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    epi: MatvecEpi,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let nsg = 2u32;
    let tg = 64u64;
    let nr = 4u32;
    let _ = sched;
    let nb = (cols / 256) as u32;
    let exact_rows = rows % ((nsg * nr) as usize) == 0;
    let tile = (TG_X_BYTES_MAX / 4) & !255;
    // Device-x streams any K from device half*; only tile when LOCAL staging is required
    // and x does not fit in TG memory.
    let prefer_local = false;
    let need_tile =
        prefer_local && cols.saturating_mul(4) > TG_X_BYTES_MAX;
    let (mut stmts, tg_x, tiled) = if need_tile {
        if rms.is_some() {
            return Err(CodegenError::Msg(
                "q6k coop+rmsnorm requires LOCAL staging of x_hat".into(),
            ));
        }
        let tg_id = 0u32;
        (
            vec![KirStmt::TgDeclF32 { id: tg_id, n: tile }],
            Some(tg_id),
            true,
        )
    } else if rms.is_some() || prefer_local {
        let (s, tg_opt) = stage_local_x(cols, x_buf, tg, DType::F16, rms, next)?;
        if rms.is_some() {
            let tg = tg_opt.ok_or_else(|| {
                CodegenError::Msg("q6k coop+rmsnorm requires LOCAL staging of x_hat".into())
            })?;
            (s, Some(tg), false)
        } else {
            (s, tg_opt, false)
        }
    } else {
        (vec![], None, false)
    };

    let lane = fresh(next);
    let sg = fresh(next);
    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: sg,
        expr: bin(BinOp::Div, lid(), cu(32)),
    });
    stmts.push(KirStmt::LetU32 {
        id: lane,
        expr: bin(BinOp::Sub, lid(), bin(BinOp::Mul, uv(sg), cu(32))),
    });
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(
            BinOp::Mul,
            bin(BinOp::Add, bin(BinOp::Mul, gid(), cu(nsg)), uv(sg)),
            cu(nr),
        ),
    });
    // ggml q6: ix = lane % 2; stripe superblocks by 2.
    let ix = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: ix,
        expr: bin(
            BinOp::Sub,
            uv(lane),
            bin(BinOp::Mul, bin(BinOp::Div, uv(lane), cu(2)), cu(2)),
        ),
    });

    let mut accs = Vec::with_capacity(nr as usize);
    let mut bases = Vec::with_capacity(nr as usize);
    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = fresh(next);
        let base = fresh(next);
        stmts.push(KirStmt::Let {
            id: acc,
            expr: c(0.0),
        });
        stmts.push(KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row, cu(cols as u32)),
        });
        accs.push(acc);
        bases.push(base);
    }

    let make_ib_body = |ib_expr: KirExpr, x_off: KirExpr| -> Vec<KirStmt> {
        let nr4 = KirStmt::Q6kCoopNr4 {
            w_buf: 0,
            row0_base: uv(bases[0]),
            cols: cols as u32,
            ib: ib_expr.clone(),
            b_from_tg: tg_x,
            x_buf,
            x_off: x_off.clone(),
            lane: uv(lane),
            acc_ids: [accs[0], accs[1], accs[2], accs[3]],
        };
        if exact_rows {
            return vec![nr4];
        }
        let last = bin(BinOp::Add, uv(first), cu(3));
        let mut body = vec![KirStmt::If {
            cond: gt(cu(rows as u32), last.clone()),
            body: vec![nr4],
        }];
        let mut partial = Vec::new();
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc = accs[r as usize];
            let base = bases[r as usize];
            partial.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![KirStmt::Assign {
                    id: acc,
                    expr: bin(
                        BinOp::Add,
                        v(acc),
                        KirExpr::Q6kCoopFrag {
                            w_buf: 0,
                            row_base: Box::new(uv(base)),
                            cols: cols as u32,
                            ib: Box::new(ib_expr.clone()),
                            b_from_tg: tg_x,
                            x_buf,
                            x_off: Box::new(x_off.clone()),
                            lane: Box::new(uv(lane)),
                        },
                    ),
                }],
            });
        }
        let rows_m1 = (rows as u32).saturating_sub(1);
        body.push(KirStmt::If {
            cond: gt(last, cu(rows_m1)),
            body: partial,
        });
        body
    };

    if tiled {
        let tg_id = tg_x.expect("tiled q6 requires TG");
        let tile_u = tile as u32;
        let n_tiles = ((cols + tile - 1) / tile) as u32;
        let tile_i = fresh(next);
        stmts.push(KirStmt::LetU32 {
            id: tile_i,
            expr: cu(0),
        });
        let mut tile_body = Vec::new();
        let tile_off = fresh(next);
        tile_body.push(KirStmt::LetU32 {
            id: tile_off,
            expr: bin(BinOp::Mul, uv(tile_i), cu(tile_u)),
        });
        let tile_end = fresh(next);
        tile_body.push(KirStmt::LetU32 {
            id: tile_end,
            expr: bin(
                BinOp::Min,
                bin(BinOp::Add, uv(tile_off), cu(tile_u)),
                cu(cols as u32),
            ),
        });
        let tile_len = fresh(next);
        tile_body.push(KirStmt::LetU32 {
            id: tile_len,
            expr: bin(BinOp::Sub, uv(tile_end), uv(tile_off)),
        });
        let j = fresh(next);
        tile_body.push(KirStmt::LetU32 {
            id: j,
            expr: lid(),
        });
        tile_body.push(KirStmt::ForRange {
            id: j,
            limit_off: cu(0),
            bound: uv(tile_len),
            step: cu(tg as u32),
            body: vec![KirStmt::TgStore {
                id: tg_id,
                idx: uv(j),
                val: ld(
                    x_buf,
                    bin(BinOp::Add, uv(tile_off), uv(j)),
                    DType::F16,
                ),
            }],
        });
        tile_body.push(KirStmt::Barrier);
        let ib = fresh(next);
        tile_body.push(KirStmt::LetU32 {
            id: ib,
            expr: bin(
                BinOp::Add,
                bin(BinOp::Div, uv(tile_off), cu(256)),
                uv(ix),
            ),
        });
        tile_body.push(KirStmt::ForRange {
            id: ib,
            limit_off: cu(0),
            bound: bin(BinOp::Div, uv(tile_end), cu(256)),
            step: cu(2),
            body: make_ib_body(uv(ib), uv(tile_off)),
        });
        stmts.push(KirStmt::ForRange {
            id: tile_i,
            limit_off: cu(0),
            bound: cu(n_tiles),
            step: cu(1),
            body: tile_body,
        });
    } else {
        let ib = fresh(next);
        stmts.push(KirStmt::LetU32 {
            id: ib,
            expr: uv(ix),
        });
        stmts.push(KirStmt::ForRange {
            id: ib,
            limit_off: cu(0),
            bound: cu(nb),
            step: cu(2),
            body: make_ib_body(uv(ib), cu(0)),
        });
    }

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = accs[r as usize];
        let mut reduce_store = vec![KirStmt::Assign {
            id: acc,
            expr: KirExpr::SimdSum(Box::new(v(acc))),
        }];
        let store_body = store_row_with_epi(out_buf, row.clone(), acc, epi, next);
        reduce_store.push(KirStmt::If {
            cond: eq(uv(lane), cu(0)),
            body: store_body,
        });
        if exact_rows {
            stmts.extend(reduce_store);
        } else {
            stmts.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: reduce_store,
            });
        }
    }
    Ok(stmts)
}

fn lower_matvec_generic(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    epi: MatvecEpi,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let tg = effective_tg(weight_dtype, cols, sched.tg);
    let vec = effective_vec(weight_dtype, cols, sched.vec);
    let unroll = sched.unroll;
    let nr = sched.nr0.max(1);
    let stride = tg * u64::from(vec);
    let step = stride * u64::from(unroll);
    let last_u = u64::from(unroll.saturating_sub(1));
    let main_off = (last_u * stride + u64::from(vec) - 1) as u32;
    let rem_off = vec.saturating_sub(1);

    let (mut stmts, tg_x) = stage_local_x(cols, x_buf, tg, x_dtype(weight_dtype), rms, next)?;

    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(BinOp::Mul, gid(), cu(nr)),
    });

    let mut accs = Vec::with_capacity(nr as usize);
    let mut bases = Vec::with_capacity(nr as usize);
    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = fresh(next);
        let base = fresh(next);
        stmts.push(KirStmt::Let {
            id: acc,
            expr: c(0.0),
        });
        // OOB rows keep base=0; accumulates are skipped via row guards below.
        stmts.push(KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row, cu(cols as u32)),
        });
        accs.push(acc);
        bases.push(base);
    }

    let k = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: k,
        expr: bin(BinOp::Mul, lid(), cu(vec)),
    });

    let accumulate_chunk = |off: KirExpr, next_body: &mut Vec<KirStmt>| {
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc = accs[r as usize];
            let base = bases[r as usize];
            next_body.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![KirStmt::Assign {
                    id: acc,
                    expr: bin(
                        BinOp::Add,
                        v(acc),
                        vec_dot(
                            0,
                            bin(BinOp::Add, uv(base), off.clone()),
                            x_buf,
                            off.clone(),
                            vec,
                            weight_dtype,
                            tg_x,
                        ),
                    ),
                }],
            });
        }
    };

    let mut main_body = Vec::new();
    for u in 0..unroll {
        let off = if u == 0 {
            uv(k)
        } else {
            bin(BinOp::Add, uv(k), cu((u as u64 * stride) as u32))
        };
        accumulate_chunk(off, &mut main_body);
    }
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(main_off),
        bound: cu(cols as u32),
        step: cu(step as u32),
        body: main_body,
    });

    let mut rem_body = Vec::new();
    accumulate_chunk(uv(k), &mut rem_body);
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(rem_off),
        bound: cu(cols as u32),
        step: cu(stride as u32),
        body: rem_body,
    });

    if weight_dtype != DType::Q4K {
        let b_scalar = if let Some(tg_id) = tg_x {
            tg_load(tg_id, uv(k))
        } else {
            ld(x_buf, uv(k), x_dtype(weight_dtype))
        };
        let mut tail = Vec::new();
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc = accs[r as usize];
            let base = bases[r as usize];
            tail.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![KirStmt::Assign {
                    id: acc,
                    expr: bin(
                        BinOp::Add,
                        v(acc),
                        bin(
                            BinOp::Mul,
                            ld(0, bin(BinOp::Add, uv(base), uv(k)), weight_dtype),
                            b_scalar.clone(),
                        ),
                    ),
                }],
            });
        }
        stmts.push(KirStmt::ForRange {
            id: k,
            limit_off: cu(0),
            bound: cu(cols as u32),
            step: cu(tg as u32),
            body: tail,
        });
    }

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc = accs[r as usize];
        let store_body = store_row_with_epi(out_buf, row.clone(), acc, epi, next);
        stmts.push(KirStmt::If {
            cond: gt(cu(rows as u32), row),
            body: vec![
                KirStmt::ThreadgroupReduce { acc_id: acc, tg },
                KirStmt::If {
                    cond: eq(lid(), cu(0)),
                    body: store_body,
                },
            ],
        });
    }
    Ok(stmts)
}

/// Fused gate/up matvecs + gelu*mul. Buffers: 0=W_gate, 1=W_up, `x_buf`=x, `out_buf`=out.
/// LOCAL-stages `x` (or rmsnorm `x_hat`) once; NR rows share one K pass (dual dots per chunk).
fn lower_matvec_gate_up_gelu(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    if weight_dtype == DType::Q4K && cols % 256 == 0 {
        match lower_matvec_gate_up_q4k_coop(rows, cols, sched, x_buf, out_buf, rms, next) {
            Ok(s) => return Ok(s),
            Err(_) => {}
        }
    }
    lower_matvec_gate_up_generic(rows, cols, weight_dtype, sched, x_buf, out_buf, rms, next)
}

fn lower_matvec_gate_up_q4k_coop(
    rows: usize,
    cols: usize,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let nsg = sched.nsg.max(1);
    let tg = (nsg as u64) * 32;
    let nr = 4u32;
    let nb = (cols / 256) as u32;
    let exact_rows = rows % ((nsg * nr) as usize) == 0;

    let prefer_local = false;
    let (mut stmts, tg_x) = if rms.is_some() || prefer_local {
        let (s, tg) = stage_local_x(cols, x_buf, tg, DType::F16, rms, next)?;
        if rms.is_some() {
            let tg = tg.ok_or_else(|| {
                CodegenError::Msg(
                    "q4k coop gate_up+rmsnorm requires LOCAL staging of x_hat".into(),
                )
            })?;
            (s, Some(tg))
        } else {
            (s, tg)
        }
    } else {
        (vec![], None)
    };

    let lane = fresh(next);
    let sg = fresh(next);
    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: sg,
        expr: bin(BinOp::Div, lid(), cu(32)),
    });
    stmts.push(KirStmt::LetU32 {
        id: lane,
        expr: bin(BinOp::Sub, lid(), bin(BinOp::Mul, uv(sg), cu(32))),
    });
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(
            BinOp::Mul,
            bin(BinOp::Add, bin(BinOp::Mul, gid(), cu(nsg)), uv(sg)),
            cu(nr),
        ),
    });
    let ix = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: ix,
        expr: bin(BinOp::Div, uv(lane), cu(8)),
    });

    let mut acc_gs = Vec::with_capacity(nr as usize);
    let mut acc_us = Vec::with_capacity(nr as usize);
    let mut bases = Vec::with_capacity(nr as usize);
    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc_g = fresh(next);
        let acc_u = fresh(next);
        let base = fresh(next);
        stmts.push(KirStmt::Let {
            id: acc_g,
            expr: c(0.0),
        });
        stmts.push(KirStmt::Let {
            id: acc_u,
            expr: c(0.0),
        });
        stmts.push(KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row, cu(cols as u32)),
        });
        acc_gs.push(acc_g);
        acc_us.push(acc_u);
        bases.push(base);
    }

    let ib = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: ib,
        expr: uv(ix),
    });
    let mut body = Vec::new();
    // Prefer nr=4 path: amortize y across rows (two nr4 calls: gate then up).
    let nr_gate = nr;
    if nr_gate == 4 {
        let nr4 = KirStmt::Q4kCoopNr4Dual {
            row0_base: uv(bases[0]),
            cols: cols as u32,
            ib: uv(ib),
            b_from_tg: tg_x,
            x_buf,
            lane: uv(lane),
            acc_g: [acc_gs[0], acc_gs[1], acc_gs[2], acc_gs[3]],
            acc_u: [acc_us[0], acc_us[1], acc_us[2], acc_us[3]],
        };
        if exact_rows {
            body.push(nr4);
        } else {
            let last = bin(BinOp::Add, uv(first), cu(3));
            body.push(KirStmt::If {
                cond: gt(cu(rows as u32), last.clone()),
                body: vec![nr4],
            });
            let mut partial = Vec::new();
            for r in 0..nr {
                let row = if r == 0 {
                    uv(first)
                } else {
                    bin(BinOp::Add, uv(first), cu(r))
                };
                let acc_g = acc_gs[r as usize];
                let acc_u = acc_us[r as usize];
                let base = bases[r as usize];
                partial.push(KirStmt::If {
                    cond: gt(cu(rows as u32), row),
                    body: vec![
                        KirStmt::Assign {
                            id: acc_g,
                            expr: bin(
                                BinOp::Add,
                                v(acc_g),
                                KirExpr::Q4kCoopFrag {
                                    w_buf: 0,
                                    row_base: Box::new(uv(base)),
                                    cols: cols as u32,
                                    ib: Box::new(uv(ib)),
                                    b_from_tg: tg_x,
                                    x_buf,
                                    lane: Box::new(uv(lane)),
                                },
                            ),
                        },
                        KirStmt::Assign {
                            id: acc_u,
                            expr: bin(
                                BinOp::Add,
                                v(acc_u),
                                KirExpr::Q4kCoopFrag {
                                    w_buf: 1,
                                    row_base: Box::new(uv(base)),
                                    cols: cols as u32,
                                    ib: Box::new(uv(ib)),
                                    b_from_tg: tg_x,
                                    x_buf,
                                    lane: Box::new(uv(lane)),
                                },
                            ),
                        },
                    ],
                });
            }
            let rows_m1 = (rows as u32).saturating_sub(1);
            body.push(KirStmt::If {
                cond: gt(last, cu(rows_m1)),
                body: partial,
            });
        }
    } else {
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc_g = acc_gs[r as usize];
            let acc_u = acc_us[r as usize];
            let base = bases[r as usize];
            body.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![
                    KirStmt::Assign {
                        id: acc_g,
                        expr: bin(
                            BinOp::Add,
                            v(acc_g),
                            KirExpr::Q4kCoopFrag {
                                w_buf: 0,
                                row_base: Box::new(uv(base)),
                                cols: cols as u32,
                                ib: Box::new(uv(ib)),
                                b_from_tg: tg_x,
                                x_buf,
                                lane: Box::new(uv(lane)),
                            },
                        ),
                    },
                    KirStmt::Assign {
                        id: acc_u,
                        expr: bin(
                            BinOp::Add,
                            v(acc_u),
                            KirExpr::Q4kCoopFrag {
                                w_buf: 1,
                                row_base: Box::new(uv(base)),
                                cols: cols as u32,
                                ib: Box::new(uv(ib)),
                                b_from_tg: tg_x,
                                x_buf,
                                lane: Box::new(uv(lane)),
                            },
                        ),
                    },
                ],
            });
        }
    }
    stmts.push(KirStmt::ForRange {
        id: ib,
        limit_off: cu(0),
        bound: cu(nb),
        step: cu(4),
        body,
    });

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc_g = acc_gs[r as usize];
        let acc_u = acc_us[r as usize];
        let ax = fresh(next);
        let u = fresh(next);
        let gelu = fresh(next);
        let reduce_store = vec![
            KirStmt::Assign {
                id: acc_g,
                expr: KirExpr::SimdSum(Box::new(v(acc_g))),
            },
            KirStmt::Assign {
                id: acc_u,
                expr: KirExpr::SimdSum(Box::new(v(acc_u))),
            },
            KirStmt::If {
                cond: eq(uv(lane), cu(0)),
                body: vec![
                    KirStmt::Let {
                        id: ax,
                        expr: bin(BinOp::Min, c(20.0), bin(BinOp::Max, c(-20.0), v(acc_g))),
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
                    st(out_buf, row.clone(), bin(BinOp::Mul, v(gelu), v(acc_u))),
                ],
            },
        ];
        if exact_rows {
            stmts.extend(reduce_store);
        } else {
            stmts.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: reduce_store,
            });
        }
    }
    Ok(stmts)
}

fn lower_matvec_gate_up_generic(
    rows: usize,
    cols: usize,
    weight_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_buf: u32,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let tg = effective_tg(weight_dtype, cols, sched.tg);
    let vec = effective_vec(weight_dtype, cols, sched.vec);
    let unroll = sched.unroll;
    let nr = sched.nr0.max(1);
    let stride = tg * u64::from(vec);
    let step = stride * u64::from(unroll);
    let last_u = u64::from(unroll.saturating_sub(1));
    let main_off = (last_u * stride + u64::from(vec) - 1) as u32;
    let rem_off = vec.saturating_sub(1);

    let (mut stmts, tg_x) = stage_local_x(cols, x_buf, tg, x_dtype(weight_dtype), rms, next)?;

    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(BinOp::Mul, gid(), cu(nr)),
    });

    let mut acc_gs = Vec::with_capacity(nr as usize);
    let mut acc_us = Vec::with_capacity(nr as usize);
    let mut bases = Vec::with_capacity(nr as usize);
    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc_g = fresh(next);
        let acc_u = fresh(next);
        let base = fresh(next);
        stmts.push(KirStmt::Let {
            id: acc_g,
            expr: c(0.0),
        });
        stmts.push(KirStmt::Let {
            id: acc_u,
            expr: c(0.0),
        });
        stmts.push(KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, row, cu(cols as u32)),
        });
        acc_gs.push(acc_g);
        acc_us.push(acc_u);
        bases.push(base);
    }

    let k = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: k,
        expr: bin(BinOp::Mul, lid(), cu(vec)),
    });

    let accumulate_chunk = |off: KirExpr, body: &mut Vec<KirStmt>| {
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc_g = acc_gs[r as usize];
            let acc_u = acc_us[r as usize];
            let base = bases[r as usize];
            let a_idx = bin(BinOp::Add, uv(base), off.clone());
            body.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![
                    KirStmt::Assign {
                        id: acc_g,
                        expr: bin(
                            BinOp::Add,
                            v(acc_g),
                            vec_dot(0, a_idx.clone(), x_buf, off.clone(), vec, weight_dtype, tg_x),
                        ),
                    },
                    KirStmt::Assign {
                        id: acc_u,
                        expr: bin(
                            BinOp::Add,
                            v(acc_u),
                            vec_dot(1, a_idx, x_buf, off.clone(), vec, weight_dtype, tg_x),
                        ),
                    },
                ],
            });
        }
    };

    let mut main_body = Vec::new();
    for u in 0..unroll {
        let off = if u == 0 {
            uv(k)
        } else {
            bin(BinOp::Add, uv(k), cu((u as u64 * stride) as u32))
        };
        accumulate_chunk(off, &mut main_body);
    }
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(main_off),
        bound: cu(cols as u32),
        step: cu(step as u32),
        body: main_body,
    });

    let mut rem_body = Vec::new();
    accumulate_chunk(uv(k), &mut rem_body);
    stmts.push(KirStmt::ForRange {
        id: k,
        limit_off: cu(rem_off),
        bound: cu(cols as u32),
        step: cu(stride as u32),
        body: rem_body,
    });

    if weight_dtype != DType::Q4K {
        let b_scalar = if let Some(tg_id) = tg_x {
            tg_load(tg_id, uv(k))
        } else {
            ld(x_buf, uv(k), x_dtype(weight_dtype))
        };
        let mut tail = Vec::new();
        for r in 0..nr {
            let row = if r == 0 {
                uv(first)
            } else {
                bin(BinOp::Add, uv(first), cu(r))
            };
            let acc_g = acc_gs[r as usize];
            let acc_u = acc_us[r as usize];
            let base = bases[r as usize];
            let a_idx = bin(BinOp::Add, uv(base), uv(k));
            tail.push(KirStmt::If {
                cond: gt(cu(rows as u32), row),
                body: vec![
                    KirStmt::Assign {
                        id: acc_g,
                        expr: bin(
                            BinOp::Add,
                            v(acc_g),
                            bin(BinOp::Mul, ld(0, a_idx.clone(), weight_dtype), b_scalar.clone()),
                        ),
                    },
                    KirStmt::Assign {
                        id: acc_u,
                        expr: bin(
                            BinOp::Add,
                            v(acc_u),
                            bin(BinOp::Mul, ld(1, a_idx, weight_dtype), b_scalar.clone()),
                        ),
                    },
                ],
            });
        }
        stmts.push(KirStmt::ForRange {
            id: k,
            limit_off: cu(0),
            bound: cu(cols as u32),
            step: cu(tg as u32),
            body: tail,
        });
    }

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };
        let acc_g = acc_gs[r as usize];
        let acc_u = acc_us[r as usize];
        let ax = fresh(next);
        let u = fresh(next);
        let gelu = fresh(next);
        stmts.push(KirStmt::If {
            cond: gt(cu(rows as u32), row.clone()),
            body: vec![
                KirStmt::ThreadgroupReduce {
                    acc_id: acc_g,
                    tg,
                },
                KirStmt::ThreadgroupReduce {
                    acc_id: acc_u,
                    tg,
                },
                KirStmt::If {
                    cond: eq(lid(), cu(0)),
                    body: vec![
                        KirStmt::Let {
                            id: ax,
                            expr: bin(BinOp::Min, c(20.0), bin(BinOp::Max, c(-20.0), v(acc_g))),
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
                        st(out_buf, row, bin(BinOp::Mul, v(gelu), v(acc_u))),
                    ],
                },
            ],
        });
    }
    Ok(stmts)
}

/// Fused Q/K/V matvecs. Buffers: 0=Wq, 1=Wk, 2=Wv, `x_buf`=x, then outQ/outK/outV.
/// LOCAL-stages `x` (or rmsnorm `x_hat`) once; NR rows over max(q_rows, kv_rows).
fn lower_matvec_qkv(
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    wq_dtype: DType,
    wk_dtype: DType,
    wv_dtype: DType,
    sched: OptSchedule,
    x_buf: u32,
    out_q: u32,
    out_k: u32,
    out_v: u32,
    rms: Option<(u32, f32)>,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    let weight_dtype = wq_dtype;
    let tg = effective_tg(weight_dtype, cols, sched.tg);
    let vec = if packed_k_quant(wq_dtype) || packed_k_quant(wk_dtype) || packed_k_quant(wv_dtype) {
        if matches!(wq_dtype, DType::Q6K) || matches!(wk_dtype, DType::Q6K) || matches!(wv_dtype, DType::Q6K) {
            1
        } else {
            effective_vec(weight_dtype, cols, sched.vec)
        }
    } else {
        effective_vec(weight_dtype, cols, sched.vec)
    };
    let unroll = sched.unroll;
    let nr = sched.nr0.max(1);
    let stride = tg * u64::from(vec);
    let step = stride * u64::from(unroll);
    let last_u = u64::from(unroll.saturating_sub(1));
    let main_off = (last_u * stride + u64::from(vec) - 1) as u32;
    let rem_off = vec.saturating_sub(1);
    let max_rows = q_rows.max(kv_rows);

    let (mut stmts, tg_x) = stage_local_x(cols, x_buf, tg, x_dtype(weight_dtype), rms, next)?;

    let first = fresh(next);
    stmts.push(KirStmt::LetU32 {
        id: first,
        expr: bin(BinOp::Mul, gid(), cu(nr)),
    });

    for r in 0..nr {
        let row = if r == 0 {
            uv(first)
        } else {
            bin(BinOp::Add, uv(first), cu(r))
        };

        let mut max_body = Vec::new();

        // Q matvec → out_q
        {
            let acc = fresh(next);
            let k = fresh(next);
            let base = fresh(next);
            let mut q_body = Vec::new();
            q_body.push(KirStmt::LetU32 {
                id: base,
                expr: bin(BinOp::Mul, row.clone(), cu(cols as u32)),
            });
            q_body.push(KirStmt::Let {
                id: acc,
                expr: c(0.0),
            });
            q_body.push(KirStmt::LetU32 {
                id: k,
                expr: bin(BinOp::Mul, lid(), cu(vec)),
            });

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
                            x_buf,
                            off,
                            vec,
                            weight_dtype,
                            tg_x,
                        ),
                    ),
                });
            }
            q_body.push(KirStmt::ForRange {
                id: k,
                limit_off: cu(main_off),
                bound: cu(cols as u32),
                step: cu(step as u32),
                body: main_body,
            });
            q_body.push(KirStmt::ForRange {
                id: k,
                limit_off: cu(rem_off),
                bound: cu(cols as u32),
                step: cu(stride as u32),
                body: vec![KirStmt::Assign {
                    id: acc,
                    expr: bin(
                        BinOp::Add,
                        v(acc),
                        vec_dot(
                            0,
                            bin(BinOp::Add, uv(base), uv(k)),
                            x_buf,
                            uv(k),
                            vec,
                            weight_dtype,
                            tg_x,
                        ),
                    ),
                }],
            });
            if !packed_k_quant(weight_dtype) {
                let b_scalar = if let Some(tg_id) = tg_x {
                    tg_load(tg_id, uv(k))
                } else {
                    ld(x_buf, uv(k), x_dtype(weight_dtype))
                };
                q_body.push(KirStmt::ForRange {
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
                                ld(0, bin(BinOp::Add, uv(base), uv(k)), wq_dtype),
                                b_scalar,
                            ),
                        ),
                    }],
                });
            }
            q_body.push(KirStmt::ThreadgroupReduce { acc_id: acc, tg });
            q_body.push(KirStmt::If {
                cond: eq(lid(), cu(0)),
                body: vec![st(out_q, row.clone(), v(acc))],
            });
            max_body.push(KirStmt::If {
                cond: gt(cu(q_rows as u32), row.clone()),
                body: q_body,
            });
        }

        // Dual K+V matvecs → out_k/out_v; share one K loop like gate/up.
        {
            let acc_k = fresh(next);
            let acc_v = fresh(next);
            let k = fresh(next);
            let base = fresh(next);
            let mut kv_body = Vec::new();
            kv_body.push(KirStmt::LetU32 {
                id: base,
                expr: bin(BinOp::Mul, row.clone(), cu(cols as u32)),
            });
            kv_body.push(KirStmt::Let {
                id: acc_k,
                expr: c(0.0),
            });
            kv_body.push(KirStmt::Let {
                id: acc_v,
                expr: c(0.0),
            });
            kv_body.push(KirStmt::LetU32 {
                id: k,
                expr: bin(BinOp::Mul, lid(), cu(vec)),
            });

            let mut main_body = Vec::new();
            for u in 0..unroll {
                let off = if u == 0 {
                    uv(k)
                } else {
                    bin(BinOp::Add, uv(k), cu((u as u64 * stride) as u32))
                };
                let a_idx = bin(BinOp::Add, uv(base), off.clone());
                main_body.push(KirStmt::Assign {
                    id: acc_k,
                    expr: bin(
                        BinOp::Add,
                        v(acc_k),
                        vec_dot(1, a_idx.clone(), x_buf, off.clone(), vec, wq_dtype, tg_x),
                    ),
                });
                main_body.push(KirStmt::Assign {
                    id: acc_v,
                    expr: bin(
                        BinOp::Add,
                        v(acc_v),
                        vec_dot(2, a_idx, x_buf, off, vec, wk_dtype, tg_x),
                    ),
                });
            }
            kv_body.push(KirStmt::ForRange {
                id: k,
                limit_off: cu(main_off),
                bound: cu(cols as u32),
                step: cu(step as u32),
                body: main_body,
            });
            kv_body.push(KirStmt::ForRange {
                id: k,
                limit_off: cu(rem_off),
                bound: cu(cols as u32),
                step: cu(stride as u32),
                body: {
                    let a_idx = bin(BinOp::Add, uv(base), uv(k));
                    vec![
                        KirStmt::Assign {
                            id: acc_k,
                            expr: bin(
                                BinOp::Add,
                                v(acc_k),
                                vec_dot(1, a_idx.clone(), x_buf, uv(k), vec, wk_dtype, tg_x),
                            ),
                        },
                        KirStmt::Assign {
                            id: acc_v,
                            expr: bin(
                                BinOp::Add,
                                v(acc_v),
                                vec_dot(2, a_idx, x_buf, uv(k), vec, wv_dtype, tg_x),
                            ),
                        },
                    ]
                },
            });
            if !packed_k_quant(weight_dtype) {
                let b_scalar = if let Some(tg_id) = tg_x {
                    tg_load(tg_id, uv(k))
                } else {
                    ld(x_buf, uv(k), x_dtype(weight_dtype))
                };
                kv_body.push(KirStmt::ForRange {
                    id: k,
                    limit_off: cu(0),
                    bound: cu(cols as u32),
                    step: cu(tg as u32),
                    body: {
                        let a_idx = bin(BinOp::Add, uv(base), uv(k));
                        vec![
                            KirStmt::Assign {
                                id: acc_k,
                                expr: bin(
                                    BinOp::Add,
                                    v(acc_k),
                                    bin(BinOp::Mul, ld(1, a_idx.clone(), wk_dtype), b_scalar.clone()),
                                ),
                            },
                            KirStmt::Assign {
                                id: acc_v,
                                expr: bin(
                                    BinOp::Add,
                                    v(acc_v),
                                    bin(BinOp::Mul, ld(2, a_idx, wv_dtype), b_scalar),
                                ),
                            },
                        ]
                    },
                });
            }
            kv_body.push(KirStmt::ThreadgroupReduce {
                acc_id: acc_k,
                tg,
            });
            kv_body.push(KirStmt::ThreadgroupReduce {
                acc_id: acc_v,
                tg,
            });
            kv_body.push(KirStmt::If {
                cond: eq(lid(), cu(0)),
                body: vec![
                    st(out_k, row.clone(), v(acc_k)),
                    st(out_v, row.clone(), v(acc_v)),
                ],
            });
            max_body.push(KirStmt::If {
                cond: gt(cu(kv_rows as u32), row.clone()),
                body: kv_body,
            });
        }

        stmts.push(KirStmt::If {
            cond: gt(cu(max_rows as u32), row),
            body: max_body,
        });
    }
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

/// `out0 = residual + rms(y)*w_post`, `out1 = rms(out0)*w_ffn`.
/// Stages residual stream in LOCAL between the two ThreadgroupReduce phases.
/// Inputs: 0=y, 1=w_post, 2=residual, 3=w_ffn; outs: n_in+0, n_in+1.
fn lower_rmsnorm_add_then_rmsnorm(
    n: usize,
    eps: f32,
    dt: DType,
    tg: u64,
    next: &mut u32,
) -> Result<Vec<KirStmt>, CodegenError> {
    if n.saturating_mul(4) > TG_X_BYTES_MAX {
        return Err(CodegenError::Msg(
            "rmsnorm_add_then_rmsnorm requires LOCAL staging of residual stream".into(),
        ));
    }
    let n_in = 4u32;
    let out_x = n_in;
    let out_x2 = n_in + 1;
    let tg_id = 0u32;
    let ss = fresh(next);
    let i = fresh(next);
    let inv = fresh(next);
    let j = fresh(next);
    let ss2 = fresh(next);
    let i2 = fresh(next);
    let inv2 = fresh(next);
    let j2 = fresh(next);
    let body = vec![
        KirStmt::TgDeclF32 { id: tg_id, n },
        KirStmt::Let {
            id: ss,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: i,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::Assign {
                id: ss,
                expr: bin(
                    BinOp::Add,
                    v(ss),
                    bin(BinOp::Mul, ld(0, uv(i), dt), ld(0, uv(i), dt)),
                ),
            }],
        },
        KirStmt::ThreadgroupReduce { acc_id: ss, tg },
        KirStmt::Let {
            id: inv,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss), c(n as f32)), c(eps)),
            ),
        },
        KirStmt::LetU32 {
            id: j,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: j,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: {
                let val = bin(
                    BinOp::Add,
                    ld(2, uv(j), dt),
                    bin(
                        BinOp::Mul,
                        bin(BinOp::Mul, ld(0, uv(j), dt), v(inv)),
                        ld(1, uv(j), dt),
                    ),
                );
                vec![
                    KirStmt::TgStore {
                        id: tg_id,
                        idx: uv(j),
                        val: val.clone(),
                    },
                    st(out_x, uv(j), val),
                ]
            },
        },
        KirStmt::Barrier,
        KirStmt::Let {
            id: ss2,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: i2,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i2,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::Assign {
                id: ss2,
                expr: bin(
                    BinOp::Add,
                    v(ss2),
                    bin(BinOp::Mul, tg_load(tg_id, uv(i2)), tg_load(tg_id, uv(i2))),
                ),
            }],
        },
        KirStmt::ThreadgroupReduce { acc_id: ss2, tg },
        KirStmt::Let {
            id: inv2,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss2), c(n as f32)), c(eps)),
            ),
        },
        KirStmt::LetU32 {
            id: j2,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: j2,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![st(
                out_x2,
                uv(j2),
                bin(
                    BinOp::Mul,
                    bin(BinOp::Mul, tg_load(tg_id, uv(j2)), v(inv2)),
                    ld(3, uv(j2), dt),
                ),
            )],
        },
    ];
    Ok(body)
}

fn lower_rmsnorm(
    n: usize,
    eps: f32,
    add_res: bool,
    scale: Option<f32>,
    out_buf: u32,
    dt: DType,
    tg: u64,
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
        KirStmt::LetU32 {
            id: i,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![KirStmt::Assign {
                id: ss,
                expr: bin(
                    BinOp::Add,
                    v(ss),
                    bin(BinOp::Mul, ld(0, uv(i), dt), ld(0, uv(i), dt)),
                ),
            }],
        },
        KirStmt::ThreadgroupReduce { acc_id: ss, tg },
        KirStmt::Let {
            id: inv,
            expr: un(
                UnaryOp::Rsqrt,
                bin(BinOp::Add, bin(BinOp::Div, v(ss), c(n as f32)), c(eps)),
            ),
        },
        KirStmt::LetU32 {
            id: j,
            expr: lid(),
        },
    ];
    let mut val = bin(
        BinOp::Mul,
        bin(BinOp::Mul, ld(0, uv(j), dt), v(inv)),
        ld(1, uv(j), dt),
    );
    if add_res {
        val = bin(BinOp::Add, val, ld(2, uv(j), dt));
    }
    if let Some(sc) = scale {
        val = bin(BinOp::Mul, c(sc), val);
    }
    body.push(KirStmt::ForRange {
        id: j,
        limit_off: cu(0),
        bound: cu(n as u32),
        step: cu(tg as u32),
        body: vec![st(out_buf, uv(j), val)],
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


fn lower_rmsnorm_per_head_to_thread(
    hd: usize,
    eps: f32,
    with_weight: bool,
    x_buf: u32,
    w_buf: u32,
    th_id: u32,
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
        body: vec![th_store(th_id, fv(j), val)],
    });
    stmts
}

fn lower_rope_on_thread(
    hd: usize,
    th_id: u32,
    cos_buf: u32,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    let half = hd / 2;
    let i = fresh(next);
    let u = fresh(next);
    let vv = fresh(next);
    let cos = fresh(next);
    let sin = fresh(next);
    vec![KirStmt::For {
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
                expr: th_load(th_id, fv(i)),
            },
            KirStmt::Let {
                id: vv,
                expr: th_load(th_id, bin(BinOp::Add, cu(half as u32), fv(i))),
            },
            th_store(
                th_id,
                fv(i),
                bin(
                    BinOp::Sub,
                    bin(BinOp::Mul, v(u), v(cos)),
                    bin(BinOp::Mul, v(vv), v(sin)),
                ),
            ),
            th_store(
                th_id,
                bin(BinOp::Add, cu(half as u32), fv(i)),
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, v(u), v(sin)),
                    bin(BinOp::Mul, v(vv), v(cos)),
                ),
            ),
        ],
    }]
}

fn lower_pack_thread_q40(hd: usize, th_id: u32, dst_buf: u32, next: &mut u32) -> Vec<KirStmt> {
    assert!(hd % 32 == 0);
    let n_blk = hd / 32;
    let b = fresh(next);
    vec![KirStmt::For {
        id: b,
        n: n_blk,
        body: vec![KirStmt::Q40PackFromThread {
            dst_buf,
            block: bin(BinOp::Add, bin(BinOp::Mul, gid(), cu(n_blk as u32)), fv(b)),
            th_id,
            th_off: bin(BinOp::Mul, fv(b), cu(32)),
        }],
    }]
}

fn lower_rmsnorm_per_head_rope_q40(
    hd: usize,
    eps: f32,
    with_weight: bool,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    // inputs: x=0, w=1, cos_sin=2; out Q40 = 3
    let th = 0u32;
    let mut body = vec![KirStmt::ThreadDeclF32 { id: th, n: hd }];
    body.extend(lower_rmsnorm_per_head_to_thread(
        hd, eps, with_weight, 0, 1, th, dt, next,
    ));
    body.extend(lower_rope_on_thread(hd, th, 2, dt, next));
    body.extend(lower_pack_thread_q40(hd, th, 3, next));
    body
}

fn lower_rmsnorm_per_head_q40(
    hd: usize,
    eps: f32,
    with_weight: bool,
    dt: DType,
    next: &mut u32,
) -> Vec<KirStmt> {
    // inputs: x=0, w=1; out Q40 = 2
    let th = 0u32;
    let mut body = vec![KirStmt::ThreadDeclF32 { id: th, n: hd }];
    body.extend(lower_rmsnorm_per_head_to_thread(
        hd, eps, with_weight, 0, 1, th, dt, next,
    ));
    body.extend(lower_pack_thread_q40(hd, th, 2, next));
    body
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

fn lower_quantize_q40(n_in: u32, next: &mut u32) -> Vec<KirStmt> {
    let _ = next;
    let blk = gid();
    vec![KirStmt::Q40PackBlock {
        dst_buf: n_in, // out
        block: blk.clone(),
        src_buf: 0,
        src_elem: bin(BinOp::Mul, blk, cu(32)),
    }]
}

fn lower_sdpa_online(
    n_q: usize,
    hd: usize,
    max_t: usize,
    q_dt: DType,
    kv_dt: DType,
    tg: u64,
    next: &mut u32,
) -> Vec<KirStmt> {
    let _ = (max_t, n_q);
    // Shared K/V across Q heads. Contiguous lane slices.
    // Max TILE that fits K+V float TG (2 * tile * hd * 4 ≤ 32KB): hd=256→8, hd=512→8.
    assert!(hd % 32 == 0, "sdpa hd must be multiple of 32");
    let tile = (32768 / (2 * hd * 4)).min(8).max(1);
    assert!(2 * tile * hd * 4 <= 32768, "SDPA K+V tile exceeds TG budget");
    let n_own = hd / 32;
    let k_tg = 0u32;
    let v_tg = 1u32;
    let q_th = 0u32;
    let o_th = 1u32;

    let head = fresh(next);
    let lane = fresh(next);
    let base = fresh(next);
    let tlen_f = fresh(next);
    let start_f = fresh(next);
    let tlen = fresh(next);
    let start = fresh(next);
    let m = fresh(next);
    let lsum = fresh(next);
    let t_base = fresh(next);
    let s = fresh(next);
    let m2 = fresh(next);
    let alpha = fresh(next);
    let e = fresh(next);
    let iq = fresh(next);
    let io = fresh(next);
    let iw = fresh(next);
    let q_base = fresh(next);
    let kv_base = fresh(next);
    let inv_l = fresh(next);
    let mut d_load = Vec::with_capacity(tile);
    let mut d_score = Vec::with_capacity(tile);
    for _ in 0..tile {
        d_load.push(fresh(next));
        d_score.push(fresh(next));
    }

    let mut stmts = vec![
        KirStmt::TgDeclF32 {
            id: k_tg,
            n: tile * hd,
        },
        KirStmt::TgDeclF32 {
            id: v_tg,
            n: tile * hd,
        },
        KirStmt::ThreadDeclF32 { id: q_th, n: n_own },
        KirStmt::ThreadDeclF32 { id: o_th, n: n_own },
        KirStmt::LetU32 {
            id: head,
            expr: bin(BinOp::Div, lid(), cu(32)),
        },
        KirStmt::LetU32 {
            id: lane,
            expr: bin(BinOp::Sub, lid(), bin(BinOp::Mul, uv(head), cu(32))),
        },
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, uv(lane), cu(n_own as u32)),
        },
        KirStmt::Let {
            id: tlen_f,
            expr: ld(3, cu(0), q_dt),
        },
        KirStmt::Let {
            id: start_f,
            expr: ld(3, cu(1), q_dt),
        },
        KirStmt::LetU32 {
            id: tlen,
            expr: KirExpr::CastF32ToU32(Box::new(v(tlen_f))),
        },
        KirStmt::LetU32 {
            id: start,
            expr: KirExpr::CastF32ToU32(Box::new(v(start_f))),
        },
        KirStmt::LetU32 {
            id: q_base,
            expr: bin(BinOp::Mul, uv(head), cu(hd as u32)),
        },
        KirStmt::Let {
            id: m,
            expr: c(-1.0e30),
        },
        KirStmt::Let {
            id: lsum,
            expr: c(0.0),
        },
        KirStmt::For {
            id: iq,
            n: n_own,
            body: vec![
                th_store(
                    q_th,
                    fv(iq),
                    ld(
                        0,
                        bin(BinOp::Add, uv(q_base), bin(BinOp::Add, uv(base), fv(iq))),
                        q_dt,
                    ),
                ),
                th_store(o_th, fv(iq), c(0.0)),
            ],
        },
        KirStmt::LetU32 {
            id: t_base,
            expr: cu(0),
        },
    ];

    let mut pass = Vec::new();
    for r in 0..tile {
        let row_off = (r * hd) as u32;
        let row_base = bin(
            BinOp::Mul,
            bin(
                BinOp::Add,
                uv(start),
                bin(BinOp::Add, uv(t_base), cu(r as u32)),
            ),
            cu(hd as u32),
        );
        let load_body = {
            let hd4 = (hd as u32) / 4;
            let d = d_load[r];
            let off4 = |base: KirExpr| {
                bin(BinOp::Add, base, bin(BinOp::Mul, uv(d), cu(4)))
            };
            vec![
                KirStmt::LetU32 {
                    id: d,
                    expr: lid(),
                },
                KirStmt::ForRange {
                    id: d,
                    limit_off: cu(0),
                    bound: cu(hd4),
                    step: cu(tg as u32),
                    body: vec![
                        KirStmt::TgStoreF4FromLoad {
                            tg_id: k_tg,
                            tg_off: off4(cu(row_off)),
                            src_buf: 1,
                            src_elem: off4(row_base.clone()),
                            dtype: kv_dt,
                        },
                        KirStmt::TgStoreF4FromLoad {
                            tg_id: v_tg,
                            tg_off: off4(cu(row_off)),
                            src_buf: 2,
                            src_elem: off4(row_base.clone()),
                            dtype: kv_dt,
                        },
                    ],
                },
            ]
        };
        if tile == 1 {
            pass.extend(load_body);
        } else {
            pass.push(KirStmt::If {
                cond: gt(uv(tlen), bin(BinOp::Add, uv(t_base), cu(r as u32))),
                body: load_body,
            });
        }
    }
    pass.push(KirStmt::Barrier);

    for r in 0..tile {
        let row_off = (r * hd) as u32;
        let mut body = Vec::new();
        body.push(KirStmt::LetU32 {
            id: kv_base,
            expr: bin(
                BinOp::Mul,
                bin(
                    BinOp::Add,
                    uv(start),
                    bin(BinOp::Add, uv(t_base), cu(r as u32)),
                ),
                cu(hd as u32),
            ),
        });
        body.push(KirStmt::Let {
            id: s,
            expr: c(0.0),
        });
        if n_own % 4 == 0 {
            let n4 = n_own / 4;
            body.push(KirStmt::For {
                id: d_score[r],
                n: n4,
                body: vec![KirStmt::Assign {
                    id: s,
                    expr: bin(
                        BinOp::Add,
                        v(s),
                        bin(
                            BinOp::Add,
                            bin(
                                BinOp::Add,
                                bin(
                                    BinOp::Mul,
                                    th_load(q_th, bin(BinOp::Mul, fv(d_score[r]), cu(4))),
                                    tg_load(
                                        k_tg,
                                        bin(
                                            BinOp::Add,
                                            cu(row_off),
                                            bin(
                                                BinOp::Add,
                                                uv(base),
                                                bin(BinOp::Mul, fv(d_score[r]), cu(4)),
                                            ),
                                        ),
                                    ),
                                ),
                                bin(
                                    BinOp::Mul,
                                    th_load(
                                        q_th,
                                        bin(BinOp::Add, bin(BinOp::Mul, fv(d_score[r]), cu(4)), cu(1)),
                                    ),
                                    tg_load(
                                        k_tg,
                                        bin(
                                            BinOp::Add,
                                            cu(row_off),
                                            bin(
                                                BinOp::Add,
                                                uv(base),
                                                bin(
                                                    BinOp::Add,
                                                    bin(BinOp::Mul, fv(d_score[r]), cu(4)),
                                                    cu(1),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            bin(
                                BinOp::Add,
                                bin(
                                    BinOp::Mul,
                                    th_load(
                                        q_th,
                                        bin(BinOp::Add, bin(BinOp::Mul, fv(d_score[r]), cu(4)), cu(2)),
                                    ),
                                    tg_load(
                                        k_tg,
                                        bin(
                                            BinOp::Add,
                                            cu(row_off),
                                            bin(
                                                BinOp::Add,
                                                uv(base),
                                                bin(
                                                    BinOp::Add,
                                                    bin(BinOp::Mul, fv(d_score[r]), cu(4)),
                                                    cu(2),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                bin(
                                    BinOp::Mul,
                                    th_load(
                                        q_th,
                                        bin(BinOp::Add, bin(BinOp::Mul, fv(d_score[r]), cu(4)), cu(3)),
                                    ),
                                    tg_load(
                                        k_tg,
                                        bin(
                                            BinOp::Add,
                                            cu(row_off),
                                            bin(
                                                BinOp::Add,
                                                uv(base),
                                                bin(
                                                    BinOp::Add,
                                                    bin(BinOp::Mul, fv(d_score[r]), cu(4)),
                                                    cu(3),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                }],
            });
        } else {
            body.push(KirStmt::For {
                id: d_score[r],
                n: n_own,
                body: vec![KirStmt::Assign {
                    id: s,
                    expr: bin(
                        BinOp::Add,
                        v(s),
                        bin(
                            BinOp::Mul,
                            th_load(q_th, fv(d_score[r])),
                            tg_load(
                                k_tg,
                                bin(BinOp::Add, cu(row_off), bin(BinOp::Add, uv(base), fv(d_score[r]))),
                            ),
                        ),
                    ),
                }],
            });
        }
        body.push(KirStmt::Assign {
            id: s,
            expr: KirExpr::SimdSum(Box::new(v(s))),
        });
        body.push(KirStmt::Let {
            id: m2,
            expr: bin(BinOp::Max, v(m), v(s)),
        });
        body.push(KirStmt::Let {
            id: alpha,
            expr: un(UnaryOp::Exp, bin(BinOp::Sub, v(m), v(m2))),
        });
        body.push(KirStmt::Let {
            id: e,
            expr: un(UnaryOp::Exp, bin(BinOp::Sub, v(s), v(m2))),
        });
        body.push(KirStmt::Assign {
            id: lsum,
            expr: bin(BinOp::Add, bin(BinOp::Mul, v(lsum), v(alpha)), v(e)),
        });
        body.push(KirStmt::Assign {
            id: m,
            expr: v(m2),
        });
        body.push(KirStmt::For {
            id: io,
            n: n_own,
            body: vec![th_store(
                o_th,
                fv(io),
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, th_load(o_th, fv(io)), v(alpha)),
                    bin(
                        BinOp::Mul,
                        v(e),
                        tg_load(
                            v_tg,
                            bin(BinOp::Add, cu(row_off), bin(BinOp::Add, uv(base), fv(io))),
                        ),
                    ),
                ),
            )],
        });
        if tile == 1 {
            pass.extend(body);
        } else {
            pass.push(KirStmt::If {
                cond: gt(uv(tlen), bin(BinOp::Add, uv(t_base), cu(r as u32))),
                body,
            });
        }
    }

    stmts.push(KirStmt::ForRange {
        id: t_base,
        limit_off: cu(0),
        bound: uv(tlen),
        step: cu(tile as u32),
        body: pass,
    });
    stmts.push(KirStmt::Let {
        id: inv_l,
        expr: bin(BinOp::Div, c(1.0), v(lsum)),
    });
    stmts.push(KirStmt::For {
        id: iw,
        n: n_own,
        body: vec![st(
            4,
            bin(BinOp::Add, uv(q_base), bin(BinOp::Add, uv(base), fv(iw))),
            bin(BinOp::Mul, th_load(o_th, fv(iw)), v(inv_l)),
        )],
    });
    stmts
}

fn lower_sdpa(n_q: usize, hd: usize, max_t: usize, dt: DType, next: &mut u32) -> Vec<KirStmt> {
    lower_sdpa_online(n_q, hd, max_t, dt, dt, (n_q as u64) * 32, next)
}


fn lower_softcap_argmax(
    n: usize,
    cap: f32,
    dt: DType,
    tg: u64,
    next: &mut u32,
) -> Vec<KirStmt> {
    let best_i = fresh(next);
    let best_v = fresh(next);
    let i = fresh(next);
    let val = fresh(next);
    vec![
        KirStmt::Let {
            id: best_v,
            expr: c(-1.0e30),
        },
        KirStmt::Let {
            id: best_i,
            expr: c(0.0),
        },
        KirStmt::LetU32 {
            id: i,
            expr: lid(),
        },
        KirStmt::ForRange {
            id: i,
            limit_off: cu(0),
            bound: cu(n as u32),
            step: cu(tg as u32),
            body: vec![
                KirStmt::Let {
                    id: val,
                    expr: bin(
                        BinOp::Mul,
                        c(cap),
                        un(
                            UnaryOp::Tanh,
                            bin(BinOp::Div, ld(0, uv(i), dt), c(cap)),
                        ),
                    ),
                },
                KirStmt::If {
                    cond: gt(v(val), v(best_v)),
                    body: vec![
                        KirStmt::Assign {
                            id: best_v,
                            expr: v(val),
                        },
                        KirStmt::Assign {
                            id: best_i,
                            expr: KirExpr::CastU32ToF32(Box::new(uv(i))),
                        },
                    ],
                },
            ],
        },
        KirStmt::ThreadgroupArgmax {
            val_id: best_v,
            idx_id: best_i,
            tg,
        },
        KirStmt::If {
            cond: eq(lid(), cu(0)),
            body: vec![st(1, cu(0), v(best_i))],
        },
    ]
}


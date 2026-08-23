//! Small-batch SDPA: one TG per KV head; K/V tiles shared across query tokens.
//! Thesis A: online softmax + TG tiles from AST — not a hand flash_attn paste.

use super::{
    bin, c, cu, fresh, fv, gid, gt, ld, lid, st, tg_load, th_load, th_store, un, uv, v,
};
use ksearch_ir::{BinOp, DType, KirExpr, KirStmt, UnaryOp};

/// `n_tok` in 2..=8. Launch grid must be `n_kv` (not `n_tok * n_kv`).
pub(crate) fn lower_sdpa_online_shared_q(
    n_q: usize,
    n_kv: usize,
    hd: usize,
    q_dt: DType,
    kv_dt: DType,
    tg: u64,
    n_tok: usize,
    next: &mut u32,
) -> Vec<KirStmt> {
    assert!(
        (2..=8).contains(&n_tok),
        "shared-q SDPA for MTP verify batches only"
    );
    assert!(hd % 32 == 0, "sdpa hd must be multiple of 32");
    assert!(n_q % n_kv == 0);
    let n_groups = n_q / n_kv;
    let n_kv_u = n_kv as u32;
    let tile = (32768 / (2 * hd * 4)).min(8).max(1);
    assert!(2 * tile * hd * 4 <= 32768, "SDPA K+V tile exceeds TG budget");
    let n_own = hd / 32;
    let k_tg = 0u32;
    let v_tg = 1u32;
    let q_th = 0u32;
    let o_th = 1u32;
    let m_th = 2u32;
    let l_th = 3u32;

    let kv_head = fresh(next);
    let local_head = fresh(next);
    let head = fresh(next);
    let lane = fresh(next);
    let base = fresh(next);
    let t0_f = fresh(next);
    let win_f = fresh(next);
    let t0 = fresh(next);
    let win = fresh(next);
    let start_cover = fresh(next);
    let cover_len = fresh(next);
    let t_base = fresh(next);
    let tok = fresh(next);
    let abs_t = fresh(next);
    let tlen_t = fresh(next);
    let start_t = fresh(next);
    let abs_pos = fresh(next);
    let s = fresh(next);
    let m = fresh(next);
    let lsum = fresh(next);
    let m2 = fresh(next);
    let alpha = fresh(next);
    let e = fresh(next);
    let iq = fresh(next);
    let io = fresh(next);
    let iw = fresh(next);
    let inv_l = fresh(next);
    let q_base = fresh(next);
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
        KirStmt::ThreadDeclF32 {
            id: q_th,
            n: n_tok * n_own,
        },
        KirStmt::ThreadDeclF32 {
            id: o_th,
            n: n_tok * n_own,
        },
        KirStmt::ThreadDeclF32 {
            id: m_th,
            n: n_tok,
        },
        KirStmt::ThreadDeclF32 {
            id: l_th,
            n: n_tok,
        },
        KirStmt::LetU32 {
            id: kv_head,
            expr: gid(),
        },
        KirStmt::LetU32 {
            id: local_head,
            expr: bin(BinOp::Div, lid(), cu(32)),
        },
        KirStmt::LetU32 {
            id: lane,
            expr: bin(BinOp::Sub, lid(), bin(BinOp::Mul, uv(local_head), cu(32))),
        },
        KirStmt::LetU32 {
            id: head,
            expr: bin(
                BinOp::Add,
                bin(BinOp::Mul, uv(kv_head), cu(n_groups as u32)),
                uv(local_head),
            ),
        },
        KirStmt::LetU32 {
            id: base,
            expr: bin(BinOp::Mul, uv(lane), cu(n_own as u32)),
        },
        KirStmt::Let {
            id: t0_f,
            expr: ld(3, cu(0), DType::F32),
        },
        KirStmt::Let {
            id: win_f,
            expr: ld(3, cu(2), DType::F32),
        },
        KirStmt::LetU32 {
            id: t0,
            expr: KirExpr::CastF32ToU32(Box::new(v(t0_f))),
        },
        KirStmt::LetU32 {
            id: win,
            expr: KirExpr::CastF32ToU32(Box::new(v(win_f))),
        },
        KirStmt::LetU32 {
            id: start_cover,
            expr: bin(BinOp::Sub, uv(t0), bin(BinOp::Min, uv(t0), uv(win))),
        },
        KirStmt::LetU32 {
            id: cover_len,
            expr: bin(
                BinOp::Sub,
                bin(BinOp::Add, uv(t0), cu(n_tok as u32 - 1)),
                uv(start_cover),
            ),
        },
    ];

    stmts.push(KirStmt::For {
        id: tok,
        n: n_tok,
        body: {
            let q_slot = bin(BinOp::Mul, fv(tok), cu(n_own as u32));
            vec![
                KirStmt::LetU32 {
                    id: q_base,
                    expr: bin(
                        BinOp::Add,
                        bin(BinOp::Mul, fv(tok), cu((n_q * hd) as u32)),
                        bin(BinOp::Mul, uv(head), cu(hd as u32)),
                    ),
                },
                th_store(m_th, fv(tok), c(-1.0e30)),
                th_store(l_th, fv(tok), c(0.0)),
                KirStmt::For {
                    id: iq,
                    n: n_own,
                    body: vec![
                        th_store(
                            q_th,
                            bin(BinOp::Add, q_slot.clone(), fv(iq)),
                            ld(
                                0,
                                bin(BinOp::Add, uv(q_base), bin(BinOp::Add, uv(base), fv(iq))),
                                q_dt,
                            ),
                        ),
                        th_store(o_th, bin(BinOp::Add, q_slot, fv(iq)), c(0.0)),
                    ],
                },
            ]
        },
    });

    stmts.push(KirStmt::LetU32 {
        id: t_base,
        expr: cu(0),
    });

    let mut pass = Vec::new();
    for r in 0..tile {
        let row_off = (r * hd) as u32;
        let abs_row = bin(
            BinOp::Add,
            uv(start_cover),
            bin(BinOp::Add, uv(t_base), cu(r as u32)),
        );
        let row_base = bin(
            BinOp::Mul,
            bin(
                BinOp::Add,
                bin(BinOp::Mul, abs_row, cu(n_kv_u)),
                uv(kv_head),
            ),
            cu(hd as u32),
        );
        let hd4 = (hd as u32) / 4;
        let d = d_load[r];
        let off4 = |base: KirExpr| bin(BinOp::Add, base, bin(BinOp::Mul, uv(d), cu(4)));
        let load_body = vec![
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
                        src_elem: off4(row_base),
                        dtype: kv_dt,
                    },
                ],
            },
        ];
        pass.push(KirStmt::If {
            cond: gt(uv(cover_len), bin(BinOp::Add, uv(t_base), cu(r as u32))),
            body: load_body,
        });
    }
    pass.push(KirStmt::Barrier);

    for r in 0..tile {
        let row_off = (r * hd) as u32;
        let q_slot = bin(BinOp::Mul, fv(tok), cu(n_own as u32));

        let score_body = vec![
            KirStmt::Let {
                id: s,
                expr: c(0.0),
            },
            KirStmt::Let {
                id: m,
                expr: th_load(m_th, fv(tok)),
            },
            KirStmt::Let {
                id: lsum,
                expr: th_load(l_th, fv(tok)),
            },
            KirStmt::For {
                id: d_score[r],
                n: n_own,
                body: vec![KirStmt::Assign {
                    id: s,
                    expr: bin(
                        BinOp::Add,
                        v(s),
                        bin(
                            BinOp::Mul,
                            th_load(q_th, bin(BinOp::Add, q_slot.clone(), fv(d_score[r]))),
                            tg_load(
                                k_tg,
                                bin(
                                    BinOp::Add,
                                    cu(row_off),
                                    bin(BinOp::Add, uv(base), fv(d_score[r])),
                                ),
                            ),
                        ),
                    ),
                }],
            },
            KirStmt::Assign {
                id: s,
                expr: KirExpr::SimdSum(Box::new(v(s))),
            },
            KirStmt::Let {
                id: m2,
                expr: bin(BinOp::Max, v(m), v(s)),
            },
            KirStmt::Let {
                id: alpha,
                expr: un(UnaryOp::Exp, bin(BinOp::Sub, v(m), v(m2))),
            },
            KirStmt::Let {
                id: e,
                expr: un(UnaryOp::Exp, bin(BinOp::Sub, v(s), v(m2))),
            },
            KirStmt::Assign {
                id: lsum,
                expr: bin(BinOp::Add, bin(BinOp::Mul, v(lsum), v(alpha)), v(e)),
            },
            th_store(m_th, fv(tok), v(m2)),
            th_store(l_th, fv(tok), v(lsum)),
            KirStmt::For {
                id: io,
                n: n_own,
                body: vec![th_store(
                    o_th,
                    bin(BinOp::Add, q_slot.clone(), fv(io)),
                    bin(
                        BinOp::Add,
                        bin(
                            BinOp::Mul,
                            th_load(o_th, bin(BinOp::Add, q_slot.clone(), fv(io))),
                            v(alpha),
                        ),
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
            },
        ];

        // abs_pos < abs_t  &&  abs_pos >= start_t
        //   ≡  abs_t > abs_pos  &&  (abs_pos + 1) > start_t
        let causal = gt(uv(abs_t), uv(abs_pos));
        let in_win = gt(bin(BinOp::Add, uv(abs_pos), cu(1)), uv(start_t));
        let masked = KirStmt::If {
            cond: causal,
            body: vec![KirStmt::If {
                cond: in_win,
                body: score_body,
            }],
        };

        let tok_body = vec![
            KirStmt::LetU32 {
                id: abs_t,
                expr: bin(BinOp::Add, uv(t0), fv(tok)),
            },
            KirStmt::LetU32 {
                id: tlen_t,
                expr: bin(BinOp::Min, uv(abs_t), uv(win)),
            },
            KirStmt::LetU32 {
                id: start_t,
                expr: bin(BinOp::Sub, uv(abs_t), uv(tlen_t)),
            },
            KirStmt::LetU32 {
                id: abs_pos,
                expr: bin(
                    BinOp::Add,
                    uv(start_cover),
                    bin(BinOp::Add, uv(t_base), cu(r as u32)),
                ),
            },
            masked,
        ];

        pass.push(KirStmt::If {
            cond: gt(uv(cover_len), bin(BinOp::Add, uv(t_base), cu(r as u32))),
            body: vec![KirStmt::For {
                id: tok,
                n: n_tok,
                body: tok_body,
            }],
        });
    }

    stmts.push(KirStmt::ForRange {
        id: t_base,
        limit_off: cu(0),
        bound: uv(cover_len),
        step: cu(tile as u32),
        body: pass,
    });

    stmts.push(KirStmt::For {
        id: tok,
        n: n_tok,
        body: {
            let q_slot = bin(BinOp::Mul, fv(tok), cu(n_own as u32));
            vec![
                KirStmt::LetU32 {
                    id: q_base,
                    expr: bin(
                        BinOp::Add,
                        bin(BinOp::Mul, fv(tok), cu((n_q * hd) as u32)),
                        bin(BinOp::Mul, uv(head), cu(hd as u32)),
                    ),
                },
                KirStmt::Let {
                    id: inv_l,
                    expr: bin(BinOp::Div, c(1.0), th_load(l_th, fv(tok))),
                },
                KirStmt::For {
                    id: iw,
                    n: n_own,
                    body: vec![st(
                        4,
                        bin(BinOp::Add, uv(q_base), bin(BinOp::Add, uv(base), fv(iw))),
                        bin(
                            BinOp::Mul,
                            th_load(o_th, bin(BinOp::Add, q_slot, fv(iw))),
                            v(inv_l),
                        ),
                    )],
                },
            ]
        },
    });

    stmts
}

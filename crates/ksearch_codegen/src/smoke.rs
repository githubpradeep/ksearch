//! A small zoo of representative `KernelIr` values, one per launch kind and
//! helper family.
//!
//! These are hand-built (not lowered from a graph) so they exercise the
//! renderer's helper selection and signature/prologue generation in isolation:
//! `cargo test -p ksearch_codegen --test cuda_codegen` renders and syntax-checks
//! them, and `ksearch emit` writes them out for `nvcc`/`metal` on a real
//! toolchain. Keep them in sync with the statement families in
//! [`ksearch_ir::KirStmt`].

use ksearch_ir::{DType, KernelIr, KirExpr, KirLaunch, KirStmt, Shape};

fn kir(name: &str, n_inputs: usize, launch: KirLaunch, body: Vec<KirStmt>) -> KernelIr {
    KernelIr {
        name: name.to_string(),
        n_inputs,
        n_outputs: 1,
        out_shape: Shape(vec![256]),
        out_dtype: DType::F16,
        launch,
        body,
        next_id: 1024,
    }
}

fn load(buf: u32, idx: KirExpr) -> KirExpr {
    KirExpr::Load {
        buf,
        idx: Box::new(idx),
        dtype: DType::F16,
    }
}

fn zero(id: u32) -> KirStmt {
    KirStmt::Let {
        id,
        expr: KirExpr::ConstF32(0.0),
    }
}

/// One kernel per launch kind / helper family.
pub fn smoke_kernels() -> Vec<KernelIr> {
    vec![
        // Elementwise: `[[thread_position_in_grid]]` → `blockIdx.x * blockDim.x + threadIdx.x`.
        kir(
            "k_copy",
            1,
            KirLaunch::Elementwise { n: 256 },
            vec![KirStmt::Store {
                buf: 1,
                idx: KirExpr::Gid,
                val: load(0, KirExpr::Gid),
            }],
        ),
        // Per-row kernel with a threadgroup reduction (warp shuffles + shared scratch).
        kir(
            "k_rmsnorm",
            2,
            KirLaunch::RowsParallelSg { rows: 128, nsg: 4 },
            vec![
                KirStmt::Let {
                    id: 0,
                    expr: load(0, KirExpr::Lid),
                },
                KirStmt::ThreadgroupReduce { acc_id: 0, tg: 128 },
                KirStmt::Store {
                    buf: 2,
                    idx: KirExpr::Gid,
                    val: KirExpr::Var(0),
                },
            ],
        ),
        // Q4_K cooperative matvec: shared staging, Q4_K dequant helpers, lane layout.
        // `nsg: 4` launches `(32, 4)` threads, so `lid` covers 128 entries — the
        // shared tile is sized to match.
        kir(
            "k_q4k_coop",
            3,
            KirLaunch::RowsParallelSg { rows: 64, nsg: 4 },
            vec![
                KirStmt::TgDeclF32 { id: 0, n: 128 },
                KirStmt::TgStore {
                    id: 0,
                    idx: KirExpr::Lid,
                    val: load(2, KirExpr::Lid),
                },
                KirStmt::Barrier,
                zero(10),
                zero(11),
                zero(12),
                zero(13),
                KirStmt::Q4kCoopNr4 {
                    w_buf: 0,
                    row0_base: KirExpr::Gid,
                    cols: 256,
                    ib: KirExpr::ConstU32(0),
                    b_from_tg: Some(0),
                    x_buf: 2,
                    x_off: KirExpr::ConstU32(0),
                    lane: KirExpr::Lid,
                    acc_ids: [10, 11, 12, 13],
                    n_tok: 1,
                    x_stride: 256,
                    acc_th: None,
                },
                KirStmt::Store {
                    buf: 3,
                    idx: KirExpr::Gid,
                    val: KirExpr::Var(10),
                },
            ],
        ),
        // Prefill Q4_K GEMM (`mul_mm`).
        kir(
            "k_q4k_mul_mm",
            2,
            KirLaunch::MulMm {
                tg_x: 2,
                tg_y: 1,
                tw: 256,
                nsg: 1,
                smem: 8192,
            },
            vec![KirStmt::Q4kMulMm {
                rows: 64,
                cols: 512,
                batch: 64,
                w_buf: 0,
                x_buf: 1,
                out_buf: 2,
            }],
        ),
        // Prefill Q6_K GEMM.
        kir(
            "k_q6k_mul_mm",
            2,
            KirLaunch::MulMm {
                tg_x: 1,
                tg_y: 1,
                tw: 256,
                nsg: 1,
                smem: 8192,
            },
            vec![KirStmt::Q6kMulMm {
                rows: 64,
                cols: 512,
                batch: 32,
                w_buf: 0,
                x_buf: 1,
                out_buf: 2,
            }],
        ),
        // Small-batch matvec (`mul_mv_ext`): warp-lane layout, no shared memory.
        kir(
            "k_q4k_mul_mv_ext",
            2,
            KirLaunch::MulMm {
                tg_x: 1,
                tg_y: 8,
                tw: 32,
                nsg: 4,
                smem: 32,
            },
            vec![KirStmt::Q4kMulMvExt {
                rows: 256,
                cols: 256,
                batch: 4,
                w_buf: 0,
                x_buf: 1,
                out_buf: 2,
            }],
        ),
        // Q4_0 quantiser: thread-local staging + device-side packing.
        kir(
            "k_q40_pack",
            1,
            KirLaunch::Rows { rows: 32 },
            vec![
                KirStmt::ThreadDeclF32 { id: 0, n: 32 },
                KirStmt::ThreadStore {
                    id: 0,
                    idx: KirExpr::ConstU32(0),
                    val: load(0, KirExpr::ConstU32(0)),
                },
                KirStmt::Q40PackFromThread {
                    dst_buf: 1,
                    block: KirExpr::Gid,
                    th_id: 0,
                    th_off: KirExpr::ConstU32(0),
                },
            ],
        ),
    ]
}

/// Look up a smoke kernel by name.
pub fn smoke_kernel(name: &str) -> Option<KernelIr> {
    smoke_kernels().into_iter().find(|k| k.name == name)
}

//! Model-kernel CUDA parity.
//!
//! `ksearch_codegen::smoke` is a hand-built zoo (one kernel per *statement*
//! family). This test covers the other axis: every `Graph` region the model
//! actually launches on an NVIDIA part. The regions below are transcribed from
//! the `Eng` wrappers in `ksearch_kernels` that the Gemma path calls, with
//! Gemma-ish shapes, and each one is
//!
//!   1. lowered with `lower_to_cuda_chip(.., "sm_90")` (H100; the same entry
//!      point `Context::lower` selects for `Backend::Cuda`),
//!   2. checked for CUDA structure (no Metal spellings, `__global__` entry),
//!   3. and, when a host C++ compiler is present, fed through
//!      `clang++ -x c++ -fsyntax-only` together with the dialect shim, which
//!      type-checks the whole translation unit (prelude + helpers + kernel).
//!
//! That is the strongest check available without an NVIDIA device: a region
//! that renders and type-checks here cannot fail on the device for dialect or
//! signature reasons. `ksearch emit` writes the same sources for `nvcc`.
//!
//! Set `KSEARCH_SKIP_CUDA_SYNTAX=1` to skip the `clang++` step.

use ksearch_codegen::lower_to_cuda_chip;
use ksearch_ir::{DType, Graph, IrError, Shape, TensorId};
use std::io::Write;
use std::process::Command;

// Gemma-shaped dimensions. Weight rows/cols are kept multiples of 256 so the
// Q4_K quantiser paths are exercised.
const HID: usize = 1536;
const INTER: usize = 512;
const NHEADS: usize = 8;
const NKV: usize = 1;
const HD: usize = 256;
const NTOK: usize = 4;
const ROWS: usize = 4;
const MAXT: usize = 64;
const NWG: usize = 16;
const VOCAB: usize = 4096;
const EPS: f32 = 1e-6;
/// H100 (sm_90). A 5090 would be `sm_120`; the renderer keys helpers on this.
const CHIP: &str = "sm_90";

struct Case {
    name: &'static str,
    build: fn(&mut Graph) -> Result<TensorId, IrError>,
}

fn f16(g: &mut Graph, n: usize) -> TensorId {
    g.input(Shape(vec![n]), DType::F16)
}
fn f32(g: &mut Graph, n: usize) -> TensorId {
    g.input(Shape(vec![n]), DType::F32)
}
fn w(g: &mut Graph, rows: usize, cols: usize, dt: DType) -> TensorId {
    g.input(Shape(vec![rows, cols]), dt)
}

/// Every `Graph` region the model launches (via `ksearch_kernels::Eng`).
fn cases() -> Vec<Case> {
    use DType::{F16, Q40, Q4K};
    vec![
        Case {
            name: "add",
            build: |g| {
                let a = f16(g, HID);
                let b = f16(g, HID);
                g.add(a, b)
            },
        },
        Case {
            name: "scale_const",
            build: |g| {
                let x = f16(g, HID);
                g.scale_const(x, 0.5)
            },
        },
        Case {
            name: "rmsnorm_expand",
            build: |g| {
                let x = f16(g, HID);
                let ww = f16(g, HID);
                g.rmsnorm_expand(x, ww, EPS)
            },
        },
        Case {
            name: "rmsnorm_add_expand",
            build: |g| {
                let x = f16(g, HID);
                let ww = f16(g, HID);
                let r = f16(g, HID);
                g.rmsnorm_add_expand(x, ww, r, EPS)
            },
        },
        Case {
            name: "rmsnorm_add_scale_expand",
            build: |g| {
                let x = f16(g, HID);
                let ww = f16(g, HID);
                let r = f16(g, HID);
                g.rmsnorm_add_scale_expand(x, ww, r, EPS, 0.5)
            },
        },
        Case {
            name: "rmsnorm_add_then_rmsnorm",
            build: |g| {
                let y = f16(g, HID);
                let wp = f16(g, HID);
                let r = f16(g, HID);
                let wf = f16(g, HID);
                g.rmsnorm_add_then_rmsnorm(y, wp, r, wf, EPS)
            },
        },
        Case {
            name: "rmsnorm_rows",
            build: |g| {
                let x = f16(g, ROWS * HID);
                let ww = f16(g, HID);
                g.rmsnorm_rows(x, ww, HID, ROWS, EPS)
            },
        },
        Case {
            name: "rmsnorm_add_rows",
            build: |g| {
                let x = f16(g, ROWS * HID);
                let ww = f16(g, HID);
                let r = f16(g, ROWS * HID);
                g.rmsnorm_add_rows(x, ww, r, HID, ROWS, EPS)
            },
        },
        Case {
            name: "rmsnorm_add_scale_rows",
            build: |g| {
                let x = f16(g, ROWS * HID);
                let ww = f16(g, HID);
                let r = f16(g, ROWS * HID);
                g.rmsnorm_add_scale_rows(x, ww, r, HID, ROWS, EPS, 0.5)
            },
        },
        Case {
            name: "rmsnorm_add_then_rmsnorm_rows",
            build: |g| {
                let y = f16(g, ROWS * HID);
                let wp = f16(g, HID);
                let r = f16(g, ROWS * HID);
                let wf = f16(g, HID);
                g.rmsnorm_add_then_rmsnorm_rows(y, wp, r, wf, HID, ROWS, EPS)
            },
        },
        Case {
            name: "rmsnorm_per_head",
            build: |g| {
                let x = f16(g, NHEADS * HD);
                let ww = f16(g, HD);
                g.rmsnorm_per_head(x, ww, NHEADS, HD, EPS, true)
            },
        },
        Case {
            name: "rmsnorm_per_head_rope",
            build: |g| {
                let x = f16(g, NHEADS * HD);
                let ww = f16(g, HD);
                let c = f32(g, HD);
                g.rmsnorm_per_head_rope(x, ww, c, NHEADS, HD, EPS, true)
            },
        },
        Case {
            name: "rmsnorm_per_head_rope_batch",
            build: |g| {
                let x = f16(g, NTOK * NHEADS * HD);
                let ww = f16(g, HD);
                let c = f32(g, NTOK * HD);
                g.rmsnorm_per_head_rope_batch(x, ww, c, NHEADS, HD, NTOK, EPS, true)
            },
        },
        Case {
            name: "rmsnorm_per_head_q40",
            build: |g| {
                let x = f16(g, NHEADS * HD);
                let ww = f16(g, HD);
                g.rmsnorm_per_head_q40(x, ww, NHEADS, HD, EPS, false)
            },
        },
        Case {
            name: "rmsnorm_per_head_rope_q40",
            build: |g| {
                let x = f16(g, NHEADS * HD);
                let ww = f16(g, HD);
                let c = f32(g, HD);
                g.rmsnorm_per_head_rope_q40(x, ww, c, NHEADS, HD, EPS, true)
            },
        },
        Case {
            name: "rmsnorm_per_head_qkv_q40",
            build: |g| {
                let q = f16(g, NHEADS * HD);
                let qw = f16(g, HD);
                let c = f32(g, HD);
                let k = f16(g, NKV * HD);
                let kw = f16(g, HD);
                let v = f16(g, NKV * HD);
                g.rmsnorm_per_head_qkv_q40(q, qw, c, k, kw, v, NHEADS, NKV, HD, EPS)
            },
        },
        Case {
            name: "rmsnorm_per_head_qkv_q40_batch",
            build: |g| {
                let q = f16(g, NTOK * NHEADS * HD);
                let qw = f16(g, HD);
                let c = f32(g, NTOK * HD);
                let k = f16(g, NTOK * NKV * HD);
                let kw = f16(g, HD);
                let v = f16(g, NTOK * NKV * HD);
                g.rmsnorm_per_head_qkv_q40_batch(q, qw, c, k, kw, v, NHEADS, NKV, HD, NTOK, EPS)
            },
        },
        // Dense matvec, F16 and Q4_K weights.
        Case {
            name: "matvec_f16",
            build: |g| {
                let ww = w(g, INTER, HID, F16);
                let x = f16(g, HID);
                g.matvec_prim(ww, x)
            },
        },
        Case {
            name: "matvec_q4k",
            build: |g| {
                let ww = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                g.matvec_prim(ww, x)
            },
        },
        Case {
            name: "matvec_batch_q4k",
            build: |g| {
                let ww = w(g, INTER, HID, Q4K);
                let x = f16(g, NTOK * HID);
                g.matvec_batch(ww, x, NTOK)
            },
        },
        Case {
            name: "matvec_gate_up_gelu_q4k",
            build: |g| {
                let gate = w(g, INTER, HID, Q4K);
                let up = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                g.matvec_gate_up_gelu(gate, up, x)
            },
        },
        Case {
            name: "matvec_gate_up_gelu_batch_q4k",
            build: |g| {
                let gate = w(g, INTER, HID, Q4K);
                let up = w(g, INTER, HID, Q4K);
                let x = f16(g, NTOK * HID);
                g.matvec_gate_up_gelu_batch(gate, up, x, NTOK)
            },
        },
        Case {
            name: "matvec_gelu_mul_q4k",
            build: |g| {
                let ww = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let c = f16(g, INTER);
                g.matvec_gelu_mul(ww, x, c, 0)
            },
        },
        Case {
            name: "matvec_gelu_mul_proj_rms_add_scale_q4k",
            build: |g| {
                let wg = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let c = f16(g, INTER);
                let wp = w(g, HD, INTER, Q4K);
                let wn = f16(g, HD);
                let r = f16(g, HD);
                g.matvec_gelu_mul_proj_rms_add_scale(wg, x, c, 0, wp, wn, r, EPS, 1.0)
            },
        },
        Case {
            name: "matvec_qkv_q4k",
            build: |g| {
                let wq = w(g, NHEADS * HD, HID, Q4K);
                let wk = w(g, NKV * HD, HID, Q4K);
                let wv = w(g, NKV * HD, HID, Q4K);
                let x = f16(g, HID);
                g.matvec_qkv(wq, wk, wv, x)
            },
        },
        Case {
            name: "matvec_qkv_batch_q4k",
            build: |g| {
                let wq = w(g, NHEADS * HD, HID, Q4K);
                let wk = w(g, NKV * HD, HID, Q4K);
                let wv = w(g, NKV * HD, HID, Q4K);
                let x = f16(g, NTOK * HID);
                g.matvec_qkv_batch(wq, wk, wv, x, NTOK)
            },
        },
        Case {
            name: "matvec_rmsnorm_add_q4k",
            build: |g| {
                let wm = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let wn = f16(g, INTER);
                let r = f16(g, INTER);
                g.matvec_rmsnorm_add(wm, x, wn, r, EPS)
            },
        },
        Case {
            name: "matvec_rmsnorm_add_scale_q4k",
            build: |g| {
                let wm = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let wn = f16(g, INTER);
                let r = f16(g, INTER);
                g.matvec_rmsnorm_add_scale(wm, x, wn, r, EPS, 0.5)
            },
        },
        Case {
            name: "rmsnorm_matvec_q4k",
            build: |g| {
                let wm = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let wn = f16(g, HID);
                g.rmsnorm_matvec(wm, x, wn, EPS)
            },
        },
        Case {
            name: "rmsnorm_matvec_qkv_q4k",
            build: |g| {
                let wq = w(g, NHEADS * HD, HID, Q4K);
                let wk = w(g, NKV * HD, HID, Q4K);
                let wv = w(g, NKV * HD, HID, Q4K);
                let x = f16(g, HID);
                let wn = f16(g, HID);
                g.rmsnorm_matvec_qkv(wq, wk, wv, x, wn, EPS)
            },
        },
        Case {
            name: "rmsnorm_matvec_gate_up_gelu_q4k",
            build: |g| {
                let gate = w(g, INTER, HID, Q4K);
                let up = w(g, INTER, HID, Q4K);
                let x = f16(g, HID);
                let wn = f16(g, HID);
                g.rmsnorm_matvec_gate_up_gelu(gate, up, x, wn, EPS)
            },
        },
        Case {
            name: "gelu_mul_at",
            build: |g| {
                let gate = f16(g, INTER);
                let up = f16(g, 2 * INTER);
                g.gelu_mul_at(gate, up, INTER)
            },
        },
        Case {
            name: "gelu_mul_strided",
            build: |g| {
                let gate = f16(g, NTOK * INTER);
                let up = f16(g, NTOK * INTER);
                g.gelu_mul_strided(gate, up, INTER, NTOK, 0, INTER)
            },
        },
        Case {
            name: "rope",
            build: |g| {
                let x = f16(g, NHEADS * HD);
                let c = f32(g, HD);
                g.rope(x, c, NHEADS, HD)
            },
        },
        // Attention: naive (decode, F16 and Q4_0 KV) and partitioned MWG.
        Case {
            name: "sdpa_naive_f16",
            build: |g| {
                let q = f16(g, NHEADS * HD);
                let k = f16(g, MAXT * NKV * HD);
                let v = f16(g, MAXT * NKV * HD);
                let m = f32(g, 4);
                g.sdpa_naive(q, k, v, m, NHEADS, HD, MAXT)
            },
        },
        Case {
            name: "sdpa_naive_q40",
            build: |g| {
                let q = f16(g, NHEADS * HD);
                let k = g.input(Shape(vec![MAXT * NKV * HD]), Q40);
                let v = g.input(Shape(vec![MAXT * NKV * HD]), Q40);
                let m = f32(g, 4);
                g.sdpa_naive(q, k, v, m, NHEADS, HD, MAXT)
            },
        },
        Case {
            name: "sdpa_naive_batch",
            build: |g| {
                let q = f16(g, NTOK * NHEADS * HD);
                let k = f16(g, MAXT * NKV * HD);
                let v = f16(g, MAXT * NKV * HD);
                let m = f32(g, 4);
                g.sdpa_naive_batch(q, k, v, m, NHEADS, NTOK, HD, MAXT)
            },
        },
        Case {
            name: "sdpa_mwg_part",
            build: |g| {
                let q = f16(g, NHEADS * HD);
                let k = f16(g, MAXT * NKV * HD);
                let v = f16(g, MAXT * NKV * HD);
                let m = f32(g, 4);
                g.sdpa_mwg_part(q, k, v, m, NHEADS, HD, MAXT, NWG)
            },
        },
        Case {
            name: "sdpa_mwg_reduce",
            build: |g| {
                let t = f32(g, NHEADS * NWG * (HD + 2));
                g.sdpa_mwg_reduce(t, NHEADS, HD, NWG)
            },
        },
        Case {
            name: "sdpa_mwg_part_batch",
            build: |g| {
                let q = f16(g, NTOK * NHEADS * HD);
                let k = f16(g, MAXT * NKV * HD);
                let v = f16(g, MAXT * NKV * HD);
                let m = f32(g, 4);
                g.sdpa_mwg_part_batch(q, k, v, m, NHEADS, NTOK, HD, MAXT, NWG)
            },
        },
        Case {
            name: "sdpa_mwg_reduce_batch",
            build: |g| {
                let t = f32(g, NTOK * NHEADS * NWG * (HD + 2));
                g.sdpa_mwg_reduce_batch(t, NHEADS, NTOK, HD, NWG)
            },
        },
        // KV-cache quantiser and gathers.
        Case {
            name: "quantize_q40",
            build: |g| {
                let x = f16(g, HD);
                g.quantize_q40(x, HD)
            },
        },
        Case {
            name: "copy_slice",
            build: |g| {
                let x = f16(g, 2 * HID);
                g.copy_slice(x, HID, 0, HID)
            },
        },
        Case {
            name: "copy_scale",
            build: |g| {
                let x = f16(g, HID);
                g.copy_scale(x, 0, 0, HID, 0.5)
            },
        },
        Case {
            name: "copy_scale_indexed",
            build: |g| {
                let x = f16(g, HID);
                let i = f32(g, 1);
                g.copy_scale_indexed(x, i, HID, 0.5)
            },
        },
        Case {
            name: "copy_scale_indexed_batch",
            build: |g| {
                let x = f16(g, HID);
                let i = f32(g, ROWS);
                g.copy_scale_indexed_batch(x, i, HID, ROWS, 0.5)
            },
        },
        // Sampling.
        Case {
            name: "softcap_argmax",
            build: |g| {
                let x = f16(g, VOCAB);
                g.softcap_argmax(x, 30.0)
            },
        },
        Case {
            name: "softcap_argmax_rows",
            build: |g| {
                let x = f16(g, ROWS * VOCAB);
                g.softcap_argmax_rows(x, VOCAB, ROWS, 30.0)
            },
        },
    ]
}

fn lower(c: &Case) -> String {
    let mut g = Graph::new();
    let out = (c.build)(&mut g).unwrap_or_else(|e| panic!("{}: graph build failed: {e:?}", c.name));
    let src = lower_to_cuda_chip(&g, out, CHIP)
        .unwrap_or_else(|e| panic!("{}: CUDA lowering failed: {e:?}", c.name));
    src.source
}

/// Every model region lowers to CUDA C++ with the expected structure.
#[test]
fn model_kernels_lower_to_cuda() {
    for c in cases() {
        let src = lower(&c);
        assert!(
            src.contains("ksearch CUDA dialect shim"),
            "{}: CUDA prelude missing",
            c.name
        );
        assert!(
            src.contains("KS_KERNEL void") || src.contains("__global__"),
            "{}: no CUDA entry point",
            c.name
        );
        assert!(
            !src.contains("[[buffer(") && !src.contains("[[thread_position_in_grid]]"),
            "{}: Metal attributes leaked into CUDA",
            c.name
        );
        assert!(
            !src.contains("#include <metal_stdlib>"),
            "{}: Metal header leaked into CUDA",
            c.name
        );
    }
}

/// The model's region set spans the launch kinds the CUDA host code switches on.
#[test]
fn model_kernels_cover_every_launch_kind() {
    use ksearch_codegen::LaunchHint;
    let mut seen = std::collections::BTreeSet::new();
    for c in cases() {
        let mut g = Graph::new();
        let out = (c.build)(&mut g).expect("graph build");
        let src = lower_to_cuda_chip(&g, out, CHIP).expect("lower");
        let kind = match src.launch {
            LaunchHint::Elementwise { .. } => "Elementwise",
            LaunchHint::Rows { .. } => "Rows",
            LaunchHint::RowsParallel { .. } => "RowsParallel",
            LaunchHint::RowsParallelSg { .. } => "RowsParallelSg",
            LaunchHint::RowsParallel2D { .. } => "RowsParallel2D",
            LaunchHint::MulMm { .. } => "MulMm",
        };
        seen.insert(kind);
    }
    // Decode (rows-parallel), prefill (simdgroup), GEMM and elementwise all
    // occur. `KirLaunch::Rows` collapses to `Elementwise` in the renderer, so
    // these four are the kinds the model reaches the host launcher through.
    for want in ["Elementwise", "RowsParallel", "RowsParallelSg", "MulMm"] {
        assert!(
            seen.contains(want),
            "no model kernel uses launch kind {want}: {seen:?}"
        );
    }
}

/// Host-C++ type check of the generated CUDA for every model region.
#[test]
fn model_kernels_parse_as_cxx() {
    if std::env::var("KSEARCH_SKIP_CUDA_SYNTAX").is_ok() {
        eprintln!("skipped: KSEARCH_SKIP_CUDA_SYNTAX set");
        return;
    }
    let clang = ["clang++", "c++"].into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    });
    let Some(clang) = clang else {
        eprintln!("skipped: no C++ compiler found");
        return;
    };

    let dir = std::env::temp_dir().join(format!("ksearch_cuda_model_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mk temp dir");

    for c in cases() {
        let src = lower(&c);
        let path = dir.join(format!("{}.cu.cpp", c.name));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(src.as_bytes()))
            .expect("write source");
        let out = Command::new(clang)
            .args(["-std=c++17", "-x", "c++", "-fsyntax-only"])
            .args(["-Wno-unused-variable", "-Wno-unused-parameter"])
            .arg("-Wno-unused-but-set-variable")
            .arg("-Wno-unknown-pragmas")
            .arg(&path)
            .output()
            .expect("run compiler");
        assert!(
            out.status.success(),
            "{}: generated CUDA failed the host syntax check:\n{}\n--- source ---\n{}",
            c.name,
            String::from_utf8_lossy(&out.stderr),
            src
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

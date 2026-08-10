//! P3 schedule rewrites (e-graph lite): fold Graph patterns before Kernel IR.
//!
//! Thesis A: fusion is discovered here / in [`crate::schedule`], not by adding
//! fused catalog `Op` variants.

use crate::CodegenError;
use ksearch_ir::{DType, Graph, Op, TensorId};

/// Apply local rewrites rooted at `out`. Returns the (possibly new) root id.
///
/// Current rewrites:
/// 1. **Q4 matvec discoverability** — `SumReduce(MulBroadcastRow(Q4K, F32))` is left
///    intact so the scheduler emits one Kernel IR matvec with `weight_dtype=Q4K`
///    (dequant fused at render). No `Op::MatVecQ4K` is introduced.
/// 2. **ScaleConst folding into Mul** — `Mul(ScaleConst(x,s), y)` → keep structure but
///    mark as fuseable elemwise (handled by existing Add/Mul chain fuse in schedule).
/// 3. **Dead identity ScaleConst(x, 1.0)** — elided by returning `x` when safe.
pub fn rewrite_region(graph: &mut Graph, out: TensorId) -> Result<TensorId, CodegenError> {
    let node = graph.node(out)?.clone();
    match &node.op {
        Op::ScaleConst { x, scale } if (*scale - 1.0).abs() < 1e-12 => {
            // Identity scale — use input as root.
            return Ok(*x);
        }
        Op::SumReduce { inp, axis } => {
            // Ensure MulBroadcastRow+Q4K is recognized (no Graph rewrite needed).
            let _ = (inp, axis);
            let _ = validate_q4_matvec_pattern(graph, out)?;
            Ok(out)
        }
        _ => Ok(out),
    }
}

/// True when `out` is (or rewrites to) a Q4_K weight matvec composed of primitives.
pub fn validate_q4_matvec_pattern(graph: &Graph, out: TensorId) -> Result<bool, CodegenError> {
    let node = graph.node(out)?;
    let Op::SumReduce { inp, axis } = &node.op else {
        return Ok(false);
    };
    let last = graph.node(*inp)?.shape.rank().saturating_sub(1);
    if *axis != last {
        return Ok(false);
    }
    let Op::MulBroadcastRow { left, row } = &graph.node(*inp)?.op else {
        return Ok(false);
    };
    let (_, wd) = graph.shape_dtype(*left)?;
    let (_, xd) = graph.shape_dtype(*row)?;
    Ok(wd == DType::Q4K && xd == DType::F32)
}

/// Classify weight dtype for a scheduled matvec root (`SumReduce` of `MulBroadcastRow`
/// or fused matvec / rmsnorm+matvec Call).
pub fn matvec_weight_dtype(graph: &Graph, out: TensorId) -> Result<Option<DType>, CodegenError> {
    if let Some(ksearch_ir::FuseHint::MatvecGateUpGelu { gate, .. }) = graph.fuse_hint(out) {
        return Ok(Some(graph.shape_dtype(*gate)?.1));
    }
    if let Some(ksearch_ir::FuseHint::MatvecQkv { wq, .. }) = graph.fuse_hint(out) {
        return Ok(Some(graph.shape_dtype(*wq)?.1));
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvec { w_mat, .. }) = graph.fuse_hint(out) {
        return Ok(Some(graph.shape_dtype(*w_mat)?.1));
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvecGateUpGelu { gate, .. }) = graph.fuse_hint(out)
    {
        return Ok(Some(graph.shape_dtype(*gate)?.1));
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvecQkv { wq, .. }) = graph.fuse_hint(out) {
        return Ok(Some(graph.shape_dtype(*wq)?.1));
    }
    let node = graph.node(out)?;
    let Op::SumReduce { inp, .. } = &node.op else {
        return Ok(None);
    };
    let Op::MulBroadcastRow { left, .. } = &graph.node(*inp)?.op else {
        return Ok(None);
    };
    Ok(Some(graph.shape_dtype(*left)?.1))
}

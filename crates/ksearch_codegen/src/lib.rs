//! Thesis A codegen: Graph → schedule → Kernel IR → OptOps → MSL.

mod beam;
pub mod layer;
mod render;
mod rewrite;
mod schedule;

pub use beam::{
    beam_cache_dir, beam_cache_key, beam_matvec_candidates, beam_matvec_q4k_candidates,
    load_beam_cache, save_beam_cache, BeamCacheEntry, BeamSearchResult,
};
pub use render::render_msl;
pub use rewrite::{matvec_weight_dtype, rewrite_region, validate_q4_matvec_pattern};
pub use schedule::{is_primitive_region, lower_kernel, schedule};

pub use ksearch_ir::OptSchedule;

use ksearch_ir::{DType, Graph, IrError, Op, Shape, TensorId};

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ir(#[from] IrError),
    #[error("{0}")]
    Msg(String),
}

/// A single Metal kernel ready to compile.
#[derive(Clone, Debug)]
pub struct MetalKernelSource {
    pub name: String,
    pub source: String,
    /// Number of input buffers (binding 0..n-1), output at binding n.
    pub n_inputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: LaunchHint,
}

#[derive(Clone, Debug)]
pub enum LaunchHint {
    Elementwise { n: usize },
    Rows { rows: usize, cols: usize },
    /// One threadgroup per row; `tg` threads cooperate on that row (matvec reduce).
    RowsParallel { rows: usize, tg: u64 },
    /// 2D grid: `rows` × `batch` threadgroups (batch matvecs).
    RowsParallel2D { rows: usize, batch: usize, tg: u64 },
    /// llama.cpp mul_mm: grid `(tg_x, tg_y)`, threads `(tw, nsg)`, shared `smem` bytes.
    MulMm {
        tg_x: u64,
        tg_y: u64,
        tw: u64,
        nsg: u64,
        smem: u64,
    },
}

/// Lower Graph region via schedule → Kernel IR → MSL (default OptSchedule).
pub fn lower_to_metal(graph: &Graph, out: TensorId) -> Result<MetalKernelSource, CodegenError> {
    if !is_primitive_region(graph, out)? {
        return Err(CodegenError::Msg(format!(
            "lower_to_metal: not a primitive region ({:?})",
            graph.node(out)?.op
        )));
    }
    let sched = match matvec_weight_dtype(graph, out)? {
        Some(DType::Q4K) => OptSchedule::q4k_default(),
        _ => OptSchedule::default(),
    };
    lower_with_schedule(graph, out, sched)
}

/// Graph → schedule → Kernel IR → MSL with an explicit OptSchedule (BEAM uses this).
pub fn lower_with_schedule(
    graph: &Graph,
    out: TensorId,
    sched: OptSchedule,
) -> Result<MetalKernelSource, CodegenError> {
    let kernels = schedule(graph, out)?;
    let sk = kernels
        .into_iter()
        .next()
        .ok_or_else(|| CodegenError::Msg("empty schedule".into()))?;
    let kir = lower_kernel(graph, &sk)?;
    render_msl(&kir, sched)
}

/// Deprecated stub name kept for CLI until callers switch to [`beam_matvec_candidates`].
pub fn beam_tg_candidates(rows: usize) -> Vec<u64> {
    let _ = rows;
    beam_matvec_candidates()
        .into_iter()
        .map(|s| s.tg)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Run BEAM over OptSchedules; `time_ms` should compile+time one candidate (caller owns Metal).
pub fn beam_search_matvec<F>(
    graph: &Graph,
    out: TensorId,
    chip: &str,
    mut time_ms: F,
) -> Result<BeamSearchResult, CodegenError>
where
    F: FnMut(&MetalKernelSource) -> Result<f64, CodegenError>,
{
    let node = graph.node(out)?;
    let (rows, cols) = match &node.op {
        Op::SumReduce { inp, .. } => {
            if let Op::MulBroadcastRow { left, .. } = &graph.node(*inp)?.op {
                let sh = graph.shape_dtype(*left)?.0;
                (sh.0[0], sh.0[1])
            } else {
                return Err(CodegenError::Msg("beam_search_matvec: not matvec".into()));
            }
        }
        _ => return Err(CodegenError::Msg("beam_search_matvec: not matvec".into())),
    };
    let key = beam_cache_key("matvec", rows, cols, chip);
    if let Some(cached) = load_beam_cache(&key) {
        let sched = cached.schedule;
        let kernel = lower_with_schedule(graph, out, sched)?;
        let ms = time_ms(&kernel)?;
        return Ok(BeamSearchResult {
            schedule: sched,
            ms,
            from_cache: true,
            kernel,
        });
    }

    let untuned = OptSchedule::untuned();
    let mut best_sched = untuned;
    let mut best_ms = f64::INFINITY;
    let mut best_kernel = lower_with_schedule(graph, out, untuned)?;

    for sched in beam_matvec_candidates() {
        let kernel = lower_with_schedule(graph, out, sched)?;
        let ms = time_ms(&kernel)?;
        if ms < best_ms {
            best_ms = ms;
            best_sched = sched;
            best_kernel = kernel;
        }
    }

    save_beam_cache(
        &key,
        &BeamCacheEntry {
            schedule: best_sched,
            ms: best_ms,
        },
    );
    Ok(BeamSearchResult {
        schedule: best_sched,
        ms: best_ms,
        from_cache: false,
        kernel: best_kernel,
    })
}

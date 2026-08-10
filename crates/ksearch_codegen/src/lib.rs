//! Thesis A codegen: Graph → schedule → Kernel IR → OptOps → MSL.

mod beam;
pub mod layer;
mod plan_cache;
mod render;
mod rewrite;
mod schedule;

pub use beam::{
    beam_cache_dir, beam_cache_key, beam_matvec_candidates, beam_matvec_q4k_candidates,
    load_beam_cache, save_beam_cache, BeamCacheEntry, BeamSearchResult,
};
pub use plan_cache::{load_plan, plan_cache_dir, plan_key, save_plan};
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
    pub n_inputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: LaunchHint,
}

#[derive(Clone, Debug)]
pub enum LaunchHint {
    Elementwise { n: usize },
    Rows { rows: usize, cols: usize },
    RowsParallel { rows: usize, tg: u64 },
    RowsParallel2D { rows: usize, batch: usize, tg: u64 },
    MulMm {
        tg_x: u64,
        tg_y: u64,
        tw: u64,
        nsg: u64,
        smem: u64,
    },
}

/// Lower Graph region via schedule → Kernel IR → MSL (default OptSchedule / plan cache).
pub fn lower_to_metal(graph: &Graph, out: TensorId) -> Result<MetalKernelSource, CodegenError> {
    if !is_primitive_region(graph, out)? {
        return Err(CodegenError::Msg(format!(
            "lower_to_metal: not a primitive region ({:?})",
            graph.node(out)?.op
        )));
    }
    let chip = std::env::var("KSEARCH_CHIP").unwrap_or_else(|_| "default".into());
    let sched = if let Some(DType::Q4K) = matvec_weight_dtype(graph, out)? {
        load_plan("matvec_q4k", &matvec_dims(graph, out)?, &chip)
            .unwrap_or_else(OptSchedule::q4k_default)
    } else if matvec_weight_dtype(graph, out)?.is_some() {
        load_plan("matvec_f32", &matvec_dims(graph, out)?, &chip)
            .unwrap_or_default()
    } else {
        OptSchedule::default()
    };
    lower_with_schedule(graph, out, sched)
}

fn matvec_dims(graph: &Graph, out: TensorId) -> Result<Vec<usize>, CodegenError> {
    let node = graph.node(out)?;
    if let Op::SumReduce { inp, .. } = &node.op {
        if let Op::MulBroadcastRow { left, .. } = &graph.node(*inp)?.op {
            let sh = graph.shape_dtype(*left)?.0;
            return Ok(vec![sh.0[0], sh.0[1]]);
        }
    }
    Ok(vec![out.0 as usize])
}

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

pub fn beam_tg_candidates(rows: usize) -> Vec<u64> {
    let _ = rows;
    beam_matvec_candidates()
        .into_iter()
        .map(|s| s.tg)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

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

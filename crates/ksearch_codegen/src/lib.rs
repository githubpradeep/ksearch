//! Thesis A codegen: Graph → schedule → Kernel IR → OptOps → MSL.

mod beam;
pub mod cuda_prelude;
pub mod layer;
mod lower;
mod plan_cache;
mod render;
mod rewrite;
mod schedule;
pub mod smoke;

pub use beam::{
    beam_cache_dir, beam_cache_key, beam_matvec_candidates, beam_matvec_q4k_candidates,
    load_beam_cache, save_beam_cache, BeamCacheEntry, BeamSearchResult,
};
pub use plan_cache::{load_plan, plan_cache_dir, plan_key, save_plan};
pub use render::{render_cuda, render_for, render_msl, Target};
pub use rewrite::{matvec_weight_dtype, rewrite_region, validate_q4_matvec_pattern};
pub use schedule::{is_primitive_region, lower_kernel, schedule};

pub use ksearch_ir::OptSchedule;

use ksearch_ir::{DType, Graph, IrError, Op, Shape, TensorId};
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ir(#[from] IrError),
    #[error("{0}")]
    Msg(String),
}

/// A single Metal kernel ready to compile.
/// The name predates the CUDA backend: the lowered kernel plus its launch plan,
/// valid for either [`Target`]. Prefer this alias in new code.
pub type KernelSource = MetalKernelSource;

#[derive(Clone, Debug)]
pub struct MetalKernelSource {
    pub name: String,
    pub source: String,
    pub n_inputs: usize,
    /// Number of device output buffers (`out` or `out0..out{n-1}`).
    pub n_outputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: LaunchHint,
}

#[derive(Clone, Debug)]
pub enum LaunchHint {
    Elementwise {
        n: usize,
    },
    Rows {
        rows: usize,
        cols: usize,
    },
    RowsParallel {
        rows: usize,
        tg: u64,
    },
    /// ggml-style: `threads_per_threadgroup = (32, nsg)`; `rows` = #threadgroups.
    RowsParallelSg {
        rows: usize,
        nsg: u64,
    },
    RowsParallel2D {
        rows: usize,
        batch: usize,
        tg: u64,
    },
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
    let chip = std::env::var("KSEARCH_CHIP").unwrap_or_else(|_| "default".into());
    lower_to_metal_chip(graph, out, &chip)
}

/// Lower Graph region via schedule → Kernel IR → CUDA (default OptSchedule / plan cache).
pub fn lower_to_cuda(graph: &Graph, out: TensorId) -> Result<MetalKernelSource, CodegenError> {
    let chip = std::env::var("KSEARCH_CHIP").unwrap_or_else(|_| "default".into());
    lower_to_cuda_chip(graph, out, &chip)
}

/// Same as [`lower_to_cuda`] but with an explicit chip key for the plan cache.
pub fn lower_to_cuda_chip(
    graph: &Graph,
    out: TensorId,
    chip: &str,
) -> Result<MetalKernelSource, CodegenError> {
    lower_to_target_chip(graph, out, chip, Target::Cuda)
}

/// Same as [`lower_to_cuda`] but with an explicit chip key for plan cache (use device name).
pub fn lower_to_metal_chip(
    graph: &Graph,
    out: TensorId,
    chip: &str,
) -> Result<MetalKernelSource, CodegenError> {
    lower_to_target_chip(graph, out, chip, Target::Metal)
}

/// Shared graph → schedule → Kernel IR → [`Target`] lowering path.
pub fn lower_to_target_chip(
    graph: &Graph,
    out: TensorId,
    chip: &str,
    target: Target,
) -> Result<MetalKernelSource, CodegenError> {
    if !is_primitive_region(graph, out)? {
        return Err(CodegenError::Msg(format!(
            "lower_to_metal: not a primitive region ({:?})",
            graph.node(out)?.op
        )));
    }
    let sched = match matvec_weight_dtype(graph, out)? {
        Some(DType::Q4K) => load_plan("matvec_q4k_v2", &matvec_dims(graph, out)?, chip)
            .unwrap_or_else(OptSchedule::q4k_default),
        Some(DType::Q6K) => load_plan("matvec_q6k_v2", &matvec_dims(graph, out)?, chip)
            .unwrap_or_else(OptSchedule::q4k_default),
        Some(DType::F16) => {
            load_plan("matvec_f16_nr", &matvec_dims(graph, out)?, chip).unwrap_or_default()
        }
        Some(DType::F32) => {
            load_plan("matvec_f32", &matvec_dims(graph, out)?, chip).unwrap_or_default()
        }
        Some(_) | None => OptSchedule::default(),
    };
    lower_with_schedule_target(graph, out, sched, target)
}

fn matvec_dims(graph: &Graph, out: TensorId) -> Result<Vec<usize>, CodegenError> {
    if let Some(ksearch_ir::FuseHint::MatvecGateUpGelu { rows, cols, .. }) = graph.fuse_hint(out) {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::MatvecGeluMul { rows, cols, .. }) = graph.fuse_hint(out) {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::MatvecGeluMulProjRmsAddScale {
        gate_rows,
        cols,
        proj_rows,
        ..
    }) = graph.fuse_hint(out)
    {
        return Ok(vec![*proj_rows.max(gate_rows), *cols]);
    }
    if let Some(ksearch_ir::FuseHint::MatvecRmsNormAdd { rows, cols, .. }) = graph.fuse_hint(out) {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::MatvecRmsNormAddScale { rows, cols, .. }) =
        graph.fuse_hint(out)
    {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvecGateUpGelu { rows, cols, .. }) =
        graph.fuse_hint(out)
    {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvec { rows, cols, .. }) = graph.fuse_hint(out) {
        return Ok(vec![*rows, *cols]);
    }
    if let Some(ksearch_ir::FuseHint::MatvecQkv {
        q_rows,
        kv_rows,
        cols,
        ..
    }) = graph.fuse_hint(out)
    {
        return Ok(vec![(*q_rows).max(*kv_rows), *cols]);
    }
    if let Some(ksearch_ir::FuseHint::RmsNormMatvecQkv {
        q_rows,
        kv_rows,
        cols,
        ..
    }) = graph.fuse_hint(out)
    {
        return Ok(vec![(*q_rows).max(*kv_rows), *cols]);
    }
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
    lower_with_schedule_target(graph, out, sched, Target::Metal)
}

pub fn lower_with_schedule_target(
    graph: &Graph,
    out: TensorId,
    sched: OptSchedule,
    target: Target,
) -> Result<MetalKernelSource, CodegenError> {
    let kernels = schedule(graph, out)?;
    let sk = kernels
        .into_iter()
        .next()
        .ok_or_else(|| CodegenError::Msg("empty schedule".into()))?;
    let kir = lower_kernel(graph, &sk, sched)?;
    render_for(&kir, sched, target)
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

/// Append one timing to the tuner dataset (unless `KSEARCH_TUNER_LOG=0`).
fn log_tuner_sample(kind: &str, rows: usize, cols: usize, chip: &str, sched: OptSchedule, ms: f64) {
    if std::env::var("KSEARCH_TUNER_LOG").is_ok_and(|v| v == "0") {
        return;
    }
    let _ = ksearch_tuner::dataset::append_sample(&ksearch_tuner::Sample {
        kind: kind.to_string(),
        rows,
        cols,
        chip: chip.to_string(),
        sched: sched.into(),
        ms,
        ts: ksearch_tuner::now_unix(),
    });
}

/// Retrain + persist the cost model once enough fresh samples accumulated.
/// Process-global counter keeps `warm_matvec_plans` from retraining per shape.
fn tuner_maybe_retrain(chip: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LAST_TRAINED: AtomicUsize = AtomicUsize::new(0);
    const MIN_TRAIN: usize = 96;
    const RETRAIN_EVERY: usize = 24;
    let samples = ksearch_tuner::dataset::load_samples(Some(chip));
    let total = samples.len();
    if total < MIN_TRAIN
        || total.saturating_sub(LAST_TRAINED.load(Ordering::Relaxed)) < RETRAIN_EVERY
    {
        return;
    }
    let model = ksearch_tuner::CostModel::train(
        &samples,
        chip,
        &ksearch_tuner::gbdt::GbdtConfig::default(),
    );
    match model.save() {
        Ok(p) => {
            LAST_TRAINED.store(total, Ordering::Relaxed);
            eprintln!(
                "[tuner] cost model updated: {total} samples → {}",
                p.display()
            );
        }
        Err(e) => eprintln!("[tuner] model save failed: {e}"),
    }
}

pub fn beam_search_matvec<F>(
    graph: &Graph,
    out: TensorId,
    chip: &str,
    target: Target,
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
    let dims = [rows, cols];
    let dtype = matvec_weight_dtype(graph, out)?;
    // `_v2` cache keys: schedules are honored as-is (no clamp), so pre-tuner
    // plans must not be reused.
    let plan_kind = match dtype {
        Some(DType::F16) => "matvec_f16_nr",
        Some(DType::Q4K) => "matvec_q4k_v2",
        Some(DType::Q6K) => "matvec_q6k_v2",
        _ => "matvec_f32",
    };
    // Dataset/feature kind keeps the historical name.
    let sample_kind = plan_kind.strip_suffix("_v2").unwrap_or(plan_kind);
    let force = std::env::var("KSEARCH_BEAM_FORCE").is_ok();
    if !force {
        if let Some(sched) = load_plan(plan_kind, &dims, chip) {
            let kernel = lower_with_schedule_target(graph, out, sched, target)?;
            let ms = time_ms(&kernel)?;
            return Ok(BeamSearchResult {
                schedule: sched,
                ms,
                from_cache: true,
                kernel,
            });
        }
    }

    // Candidate space; untuned baseline first so every mode has a floor.
    let mut candidates = match dtype {
        Some(DType::Q4K) | Some(DType::Q6K) => beam_matvec_q4k_candidates(),
        _ => beam_matvec_candidates(),
    };
    candidates.insert(0, OptSchedule::untuned());

    // Lower once per candidate; drop schedules that render identically AND
    // launch identically (Q4/Q6 clamping collapses many seeds into one kernel;
    // the launch hint depends on the schedule, so hash both).
    let mut uniq: Vec<(OptSchedule, MetalKernelSource)> = Vec::with_capacity(candidates.len());
    let mut seen: std::collections::HashSet<u64> = Default::default();
    for sched in candidates {
        let kernel = lower_with_schedule_target(graph, out, sched, target)?;
        let mut h = DefaultHasher::new();
        kernel.source.hash(&mut h);
        sched.hash(&mut h);
        if seen.insert(h.finish()) {
            uniq.push((sched, kernel));
        }
    }

    let scheds: Vec<ksearch_tuner::Sched> = uniq.iter().map(|(s, _)| (*s).into()).collect();
    let ctx = ksearch_tuner::SearchCtx {
        kind: sample_kind,
        rows,
        cols,
        chip,
    };
    let tuner_cfg = ksearch_tuner::TunerConfig::default();
    let mut gs = ksearch_tuner::GuidedSearch::new(tuner_cfg.clone(), ctx, &scheds);

    let (best_sched, best_ms, best_kernel) = if gs.should_guide() {
        eprintln!(
            "[tuner] guiding {plan_kind} {rows}x{cols}: {} unique kernels, budget {}",
            uniq.len(),
            tuner_cfg.budget
        );
        while !gs.should_stop() {
            let batch = gs.next_batch();
            if batch.is_empty() {
                break;
            }
            for idx in batch {
                let (sched, kernel) = &uniq[idx];
                let ms = time_ms(kernel)?;
                log_tuner_sample(sample_kind, rows, cols, chip, *sched, ms);
                gs.record(idx, ms);
            }
        }
        let (idx, ms) = gs
            .best()
            .ok_or_else(|| CodegenError::Msg("tuner measured nothing".into()))?;
        eprintln!(
            "[tuner] {rows}x{cols} → tg={} vec={} unroll={} nr0={} after {} timings (grid would be {})",
            uniq[idx].0.tg,
            uniq[idx].0.vec,
            uniq[idx].0.unroll,
            uniq[idx].0.nr0,
            gs.n_measured(),
            uniq.len()
        );
        (uniq[idx].0, ms, uniq[idx].1.clone())
    } else {
        // Classic grid over unique kernels — also feeds the dataset.
        let mut best_sched = OptSchedule::untuned();
        let mut best_ms = f64::INFINITY;
        let mut best_kernel = uniq[0].1.clone();
        for (sched, kernel) in &uniq {
            let ms = time_ms(kernel)?;
            log_tuner_sample(sample_kind, rows, cols, chip, *sched, ms);
            if ms < best_ms {
                best_ms = ms;
                best_sched = *sched;
                best_kernel = kernel.clone();
            }
        }
        (best_sched, best_ms, best_kernel)
    };

    save_plan(plan_kind, &dims, chip, best_sched, best_ms);
    tuner_maybe_retrain(chip);
    Ok(BeamSearchResult {
        schedule: best_sched,
        ms: best_ms,
        from_cache: false,
        kernel: best_kernel,
    })
}

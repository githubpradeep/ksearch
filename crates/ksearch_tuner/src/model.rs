//! CostModel: (kind, dims, schedule) features -> predicted kernel ms.

use crate::dataset;
use crate::features::{self, Sample, Sched};
use crate::gbdt::{Gbdt, GbdtConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub const MODEL_FORMAT: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostModel {
    pub format: u32,
    pub chip: String,
    pub n_samples: usize,
    pub trained_unix: u64,
    pub n_features: u32,
    /// Trained on log(ms); predictions are exponentiated.
    pub gbdt: Gbdt,
}

impl CostModel {
    pub fn train(samples: &[Sample], chip: &str, cfg: &GbdtConfig) -> CostModel {
        let x: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| features::features(&s.kind, s.rows, s.cols, &s.sched))
            .collect();
        let y: Vec<f32> = samples.iter().map(|s| (s.ms as f32).ln()).collect();
        let gbdt = Gbdt::train(&x, &y, cfg);
        CostModel {
            format: MODEL_FORMAT,
            chip: chip.to_string(),
            n_samples: samples.len(),
            trained_unix: crate::now_unix(),
            n_features: features::N_FEATURES as u32,
            gbdt,
        }
    }

    pub fn predict_ms(&self, kind: &str, rows: usize, cols: usize, sched: &Sched) -> f64 {
        let x = features::features(kind, rows, cols, sched);
        self.gbdt.predict_log(&x).exp().max(1e-6) as f64
    }

    pub fn model_path(chip: &str) -> PathBuf {
        dataset::tuner_dir().join(format!("cost_model_{}.json", slug(chip)))
    }

    pub fn save(&self) -> std::io::Result<PathBuf> {
        let dir = dataset::tuner_dir();
        std::fs::create_dir_all(&dir)?;
        let path = Self::model_path(&self.chip);
        let json = serde_json::to_string(self).expect("model serializes");
        std::fs::write(&path, json)?;
        Ok(path)
    }

    pub fn load(chip: &str) -> Option<CostModel> {
        let text = std::fs::read_to_string(Self::model_path(chip)).ok()?;
        let m: CostModel = serde_json::from_str(&text).ok()?;
        (m.format == MODEL_FORMAT && m.n_features == features::N_FEATURES as u32).then_some(m)
    }
}

/// Filesystem-safe chip tag.
pub fn slug(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Evaluation: how good is the model at the actual job — picking the fastest
// schedule without measuring everything?
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct GroupMetrics {
    pub kind: String,
    pub rows: usize,
    pub cols: usize,
    pub n: usize,
    /// Spearman rank correlation pred vs actual within this shape group.
    pub spearman: f64,
    /// Did the model's argmin match the true argmin?
    pub top1_hit: bool,
    /// (actual ms of model-picked config) / (true best ms) - 1.
    pub regret: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EvalSummary {
    pub groups: Vec<GroupMetrics>,
    pub mean_spearman: f64,
    pub top1_rate: f64,
    /// Mean relative regret across groups (0 = always picks the true best).
    pub mean_regret: f64,
    pub n_groups: usize,
    pub n_samples: usize,
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (
        ra.iter().sum::<f64>() / n,
        rb.iter().sum::<f64>() / n,
    );
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let (xa, xb) = (ra[i] - ma, rb[i] - mb);
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Evaluate per (kind, rows, cols) group with at least `min_group` samples.
pub fn evaluate(model: &CostModel, samples: &[Sample], min_group: usize) -> EvalSummary {
    let mut groups: HashMap<(String, usize, usize), Vec<&Sample>> = Default::default();
    for s in samples {
        groups
            .entry((s.kind.clone(), s.rows, s.cols))
            .or_default()
            .push(s);
    }
    let mut summary = EvalSummary::default();
    for ((kind, rows, cols), grp) in groups {
        if grp.len() < min_group {
            continue;
        }
        let preds: Vec<f64> = grp
            .iter()
            .map(|s| model.predict_ms(&kind, rows, cols, &s.sched))
            .collect();
        let acts: Vec<f64> = grp.iter().map(|s| s.ms).collect();
        let best_pred = preds
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_act = acts
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        summary.groups.push(GroupMetrics {
            kind,
            rows,
            cols,
            n: grp.len(),
            spearman: spearman(&preds, &acts),
            top1_hit: best_pred == best_act,
            regret: if acts[best_act] > 0.0 {
                acts[best_pred] / acts[best_act] - 1.0
            } else {
                0.0
            },
        });
    }
    summary.n_groups = summary.groups.len();
    summary.n_samples = samples.len();
    if !summary.groups.is_empty() {
        let g = &summary.groups;
        summary.mean_spearman = g.iter().map(|m| m.spearman).sum::<f64>() / g.len() as f64;
        summary.top1_rate =
            g.iter().filter(|m| m.top1_hit).count() as f64 / g.len() as f64;
        summary.mean_regret = g.iter().map(|m| m.regret).sum::<f64>() / g.len() as f64;
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_basics() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((spearman(&a, &a) - 1.0).abs() < 1e-9);
        let rev = [4.0, 3.0, 2.0, 1.0];
        assert!((spearman(&a, &rev) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn model_roundtrip_and_predict() {
        let mk = |tg: u64, nr0: u32, ms: f64| Sample {
            kind: "matvec_f16_nr".into(),
            rows: 4096,
            cols: 4096,
            chip: "test".into(),
            sched: Sched {
                tg,
                vec: 4,
                unroll: 1,
                nsg: 2,
                nr0,
            },
            ms,
            ts: 0,
        };
        // Synthetic "hardware": time grows with tg and shrinks with nr0.
        let samples: Vec<Sample> = (0..40)
            .flat_map(|i| {
                let tg = [32u64, 64, 128][i % 3];
                let nr0 = [1u32, 2, 4, 8][(i / 3) % 4];
                let ms = 0.02 * (tg as f64) / (nr0 as f64).sqrt() * (1.0 + 0.01 * (i % 7) as f64);
                [mk(tg, nr0, ms)]
            })
            .collect();
        let model = CostModel::train(&samples, "test", &GbdtConfig::default());
        let good = Sched {
            tg: 32,
            vec: 4,
            unroll: 1,
            nsg: 2,
            nr0: 8,
        };
        let bad = Sched {
            tg: 128,
            vec: 4,
            unroll: 1,
            nsg: 2,
            nr0: 1,
        };
        let p_good = model.predict_ms("matvec_f16_nr", 4096, 4096, &good);
        let p_bad = model.predict_ms("matvec_f16_nr", 4096, 4096, &bad);
        assert!(
            p_good < p_bad,
            "model should rank fast config lower: {p_good} vs {p_bad}"
        );

        let dir = std::env::temp_dir().join(format!("ksearch_model_test_{}", std::process::id()));
        std::env::set_var("KSEARCH_TUNER_CACHE", &dir);
        model.save().unwrap();
        let loaded = CostModel::load("test").expect("loads");
        assert_eq!(loaded.n_samples, samples.len());
        let p_loaded = loaded.predict_ms("matvec_f16_nr", 4096, 4096, &good);
        assert!((p_loaded - p_good).abs() < 1e-9);

        let ev = evaluate(&loaded, &samples, 5);
        assert!(ev.mean_regret < 0.10, "regret should be tiny: {:?}", ev.mean_regret);
        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("KSEARCH_TUNER_CACHE");
    }
}

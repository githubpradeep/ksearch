//! From-scratch histogram gradient-boosted trees (squared loss on log-ms).
//!
//! TVM uses XGBoost for the same job; this is a ~200-line Rust stand-in that is
//! plenty for a few thousand tabular samples with 18 features. Trained on
//! `log(ms)` so squared loss optimizes relative error, which is what ranking
//! candidates cares about.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct GbdtConfig {
    pub n_trees: usize,
    pub learning_rate: f32,
    pub max_depth: u8,
    pub min_samples_leaf: usize,
    pub l2: f32,
    /// Fraction of rows sampled per tree (LightGBM-style bagging).
    pub subsample: f32,
    pub n_bins: usize,
    pub seed: u64,
}

impl Default for GbdtConfig {
    fn default() -> Self {
        Self {
            n_trees: 96,
            learning_rate: 0.12,
            max_depth: 6,
            min_samples_leaf: 3,
            l2: 1.0,
            subsample: 0.8,
            n_bins: 64,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Deterministic splitmix64 — no external rand dep.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x2545_F491_4F6C_DD1D)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Flat node array; `feat < 0` marks a leaf.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeSer {
    pub feat: i32,
    /// Go left when `x[feat] <= thresh`.
    pub thresh: f32,
    pub left: i32,
    pub right: i32,
    pub value: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSer {
    pub nodes: Vec<NodeSer>,
}

impl TreeSer {
    pub fn predict(&self, x: &[f32]) -> f32 {
        let mut i = 0usize;
        loop {
            let n = &self.nodes[i];
            if n.feat < 0 {
                return n.value;
            }
            i = if x[n.feat as usize] <= n.thresh {
                n.left as usize
            } else {
                n.right as usize
            };
        }
    }
}

/// Ensemble predicting **log-ms**; exponentiate at the call site.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gbdt {
    pub bias: f32,
    pub lr: f32,
    pub trees: Vec<TreeSer>,
}

impl Gbdt {
    /// Bias-only model when there is not enough data to split.
    pub fn trivial(mean_log_ms: f32) -> Self {
        Self {
            bias: mean_log_ms,
            lr: 0.0,
            trees: Vec::new(),
        }
    }

    pub fn train(x: &[Vec<f32>], y_log: &[f32], cfg: &GbdtConfig) -> Self {
        let n = x.len();
        assert_eq!(n, y_log.len());
        if n == 0 {
            return Self::trivial(0.0);
        }
        let bias = y_log.iter().sum::<f32>() / n as f32;
        if n < 8 {
            return Self::trivial(bias);
        }
        let n_features = x[0].len();
        let bins = Bins::build(x, n_features, cfg.n_bins);
        let x_bin: Vec<Vec<u16>> = x
            .iter()
            .map(|row| {
                (0..n_features)
                    .map(|f| bins.bin_of(f, row[f]) as u16)
                    .collect()
            })
            .collect();

        let mut model = Self {
            bias,
            lr: cfg.learning_rate,
            trees: Vec::with_capacity(cfg.n_trees),
        };
        let mut pred = vec![bias; n];
        let mut rng = Rng::new(cfg.seed);
        for _ in 0..cfg.n_trees {
            // Squared-loss gradient of pred vs target.
            let g: Vec<f32> = pred
                .iter()
                .zip(y_log.iter())
                .map(|(&p, &y)| p - y)
                .collect();
            // Partial Fisher-Yates row subsample.
            let take = ((n as f32 * cfg.subsample).ceil() as usize).clamp(4.min(n), n);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..take {
                let j = i + rng.below(n - i);
                idx.swap(i, j);
            }
            let rows = idx[..take].to_vec();

            let mut nodes = Vec::new();
            build_tree(rows, &x_bin, &g, 0, cfg, &bins, &mut nodes);
            let tree = TreeSer { nodes };
            for (p, row) in pred.iter_mut().zip(x.iter()) {
                *p += model.lr * tree.predict(row);
            }
            model.trees.push(tree);
        }
        model
    }

    pub fn predict_log(&self, x: &[f32]) -> f32 {
        let mut p = self.bias;
        for t in &self.trees {
            p += self.lr * t.predict(x);
        }
        p
    }
}

struct Bins {
    /// Per feature: ascending quantile edges (deduped).
    edges: Vec<Vec<f32>>,
}

impl Bins {
    fn build(x: &[Vec<f32>], n_features: usize, n_bins: usize) -> Self {
        let mut edges = vec![Vec::new(); n_features];
        for f in 0..n_features {
            let mut col: Vec<f32> = x.iter().map(|r| r[f]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col.len();
            if n < 2 {
                continue;
            }
            let mut e = Vec::with_capacity(n_bins.saturating_sub(1));
            for b in 1..n_bins {
                let idx = (b * n / n_bins).min(n - 1);
                let v = col[idx];
                if e.last().copied().is_none_or(|last| v > last + 1e-9) {
                    e.push(v);
                }
            }
            edges[f] = e;
        }
        Self { edges }
    }

    /// Bin index such that train-side `{bin <= b}` matches predict-side `{x <= edges[b]}`.
    fn bin_of(&self, f: usize, v: f32) -> usize {
        self.edges[f].partition_point(|&e| e < v)
    }
}

#[derive(Clone, Copy)]
struct HistSum {
    sum_g: f64,
    cnt: u32,
}

impl HistSum {
    fn zeroed(n: usize) -> Vec<Self> {
        vec![Self { sum_g: 0.0, cnt: 0 }; n]
    }
}

fn best_split(
    rows: &[usize],
    x_bin: &[Vec<u16>],
    g: &[f32],
    bins: &Bins,
    cfg: &GbdtConfig,
) -> Option<(usize, usize)> {
    let n_features = bins.edges.len();
    let mut hist: Vec<Vec<HistSum>> = bins
        .edges
        .iter()
        .map(|e| HistSum::zeroed(e.len() + 1))
        .collect();
    for &r in rows {
        let xr = &x_bin[r];
        for f in 0..n_features {
            let h = &mut hist[f][xr[f] as usize];
            h.sum_g += g[r] as f64;
            h.cnt += 1;
        }
    }
    let mut total = HistSum { sum_g: 0.0, cnt: 0 };
    for hf in &hist {
        for h in hf {
            total.sum_g += h.sum_g;
            total.cnt += h.cnt;
        }
    }
    if (total.cnt as usize) < 2 * cfg.min_samples_leaf {
        return None;
    }
    let parent_score = total.sum_g * total.sum_g / (total.cnt as f64 + cfg.l2 as f64);
    let mut best: Option<(usize, usize)> = None;
    let mut best_gain = 1e-12f64;
    for (f, hf) in hist.iter().enumerate() {
        if bins.edges[f].is_empty() {
            continue; // constant feature carries no signal
        }
        let mut left = HistSum { sum_g: 0.0, cnt: 0 };
        let nbins = hf.len();
        for (b, h) in hf.iter().enumerate().take(nbins - 1) {
            left.sum_g += h.sum_g;
            left.cnt += h.cnt;
            if left.cnt == 0 {
                continue;
            }
            let rc = total.cnt - left.cnt;
            if rc == 0 {
                break;
            }
            if (left.cnt as usize) < cfg.min_samples_leaf || (rc as usize) < cfg.min_samples_leaf {
                continue;
            }
            let rsum = total.sum_g - left.sum_g;
            let score = left.sum_g * left.sum_g / (left.cnt as f64 + cfg.l2 as f64)
                + rsum * rsum / (rc as f64 + cfg.l2 as f64);
            let gain = score - parent_score;
            if gain > best_gain {
                best_gain = gain;
                best = Some((f, b));
            }
        }
    }
    best
}

fn build_tree(
    rows: Vec<usize>,
    x_bin: &[Vec<u16>],
    g: &[f32],
    depth: u8,
    cfg: &GbdtConfig,
    bins: &Bins,
    nodes: &mut Vec<NodeSer>,
) -> usize {
    let sum_g: f64 = rows.iter().map(|&r| g[r] as f64).sum();
    let leaf_value = -(sum_g / (rows.len() as f64 + cfg.l2 as f64)) as f32;

    let make_leaf = |nodes: &mut Vec<NodeSer>| -> usize {
        nodes.push(NodeSer {
            feat: -1,
            thresh: 0.0,
            left: -1,
            right: -1,
            value: leaf_value,
        });
        nodes.len() - 1
    };

    if depth >= cfg.max_depth || rows.len() < 2 * cfg.min_samples_leaf {
        return make_leaf(nodes);
    }
    let Some((feat, bin)) = best_split(&rows, x_bin, g, bins, cfg) else {
        return make_leaf(nodes);
    };
    let thresh = bins.edges[feat][bin];
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for &r in &rows {
        if (x_bin[r][feat] as usize) <= bin {
            left_rows.push(r);
        } else {
            right_rows.push(r);
        }
    }
    if left_rows.is_empty() || right_rows.is_empty() {
        return make_leaf(nodes);
    }
    let idx = nodes.len();
    nodes.push(NodeSer {
        feat: feat as i32,
        thresh,
        left: -1,
        right: -1,
        value: 0.0,
    });
    let li = build_tree(left_rows, x_bin, g, depth + 1, cfg, bins, nodes);
    let ri = build_tree(right_rows, x_bin, g, depth + 1, cfg, bins, nodes);
    nodes[idx].left = li as i32;
    nodes[idx].right = ri as i32;
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn learns_smooth_function() {
        // y = ln(ms)-style target: 2*log2(x0) + 0.5*x1 + interaction bump.
        let mut rng = Rng::new(42);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..600 {
            let x0 = 1.0 + rng.next_f32() * 15.0; // e.g. tg
            let x1 = rng.next_f32() * 4.0; // e.g. nr0
            let yv = 2.0 * x0.log2() + 0.5 * x1 + ((x0 * x1) * 0.05).sin() * 0.3;
            x.push(vec![x0, x1]);
            y.push(yv);
        }
        let cfg = GbdtConfig::default();
        let m = Gbdt::train(&x, &y, &cfg);
        assert!(!m.trees.is_empty());
        let base_rmse = (y.iter()
            .map(|&yi| (yi - y.iter().sum::<f32>() / y.len() as f32).powi(2))
            .sum::<f32>()
            / y.len() as f32)
            .sqrt();
        let rmse = (x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| (m.predict_log(xi) - yi).powi(2))
            .sum::<f32>()
            / y.len() as f32)
            .sqrt();
        assert!(
            rmse < base_rmse * 0.35,
            "model should fit well: rmse={rmse} base={base_rmse}"
        );
    }

    #[test]
    fn binning_matches_predict_thresholds() {
        // Step function at x=5 must be recovered exactly on both sides.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let v = (i % 20) as f32 + 0.5;
            x.push(vec![v]);
            y.push(if v <= 5.0 { 1.0 } else { 10.0 });
        }
        let cfg = GbdtConfig {
            n_trees: 32,
            max_depth: 3,
            ..Default::default()
        };
        let m = Gbdt::train(&x, &y, &cfg);
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let p = m.predict_log(xi);
            assert!(
                approx(p, yi, 0.75),
                "x={} y={yi} pred={p}",
                xi[0]
            );
        }
    }

    #[test]
    fn tiny_dataset_is_bias_only() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let m = Gbdt::train(&x, &y, &GbdtConfig::default());
        assert!(m.trees.is_empty());
        assert!(approx(m.bias, 2.0, 1e-6));
    }
}

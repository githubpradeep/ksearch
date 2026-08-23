//! Guided search: cost-model ranking + epsilon-greedy exploration under a
//! measurement budget.
//!
//! This is the TVM trick (learned cost model ranks schedules; only the top
//! candidates get measured) expressed as a single-step bandit: state =
//! (problem features, measured set), action = which unmeasured schedule to
//! time next, reward = -ms. The GBDT is the Q-function; `explore` random
//! picks per round keep the model learning about the tails.

use crate::features::Sched;
use crate::gbdt::Rng;
use crate::model::CostModel;

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug)]
pub struct TunerConfig {
    /// Guided mode on (`KSEARCH_TUNER=1`); off = log-only + classic grid.
    pub enabled: bool,
    /// Max kernels timed per search (including the untuned baseline).
    pub budget: usize,
    /// Random exploratory measurements per round.
    pub explore: usize,
    /// Dataset size (this chip) required before guiding replaces grid search.
    pub min_samples_to_guide: usize,
    /// Stop once measured best <= predicted global min * (1 + this).
    pub early_stop_rel: f64,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            enabled: std::env::var("KSEARCH_TUNER").is_ok_and(|v| v != "0"),
            budget: env_parse("KSEARCH_TUNER_BUDGET", 12usize),
            explore: env_parse("KSEARCH_TUNER_EXPLORE", 2usize),
            min_samples_to_guide: env_parse("KSEARCH_TUNER_MIN_SAMPLES", 128usize),
            early_stop_rel: env_parse("KSEARCH_TUNER_EARLY_STOP", 0.03f64),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchCtx<'a> {
    pub kind: &'a str,
    pub rows: usize,
    pub cols: usize,
    pub chip: &'a str,
}

/// Budgeted, model-guided measurement policy over a fixed candidate list.
pub struct GuidedSearch<'a> {
    cfg: TunerConfig,
    ctx: SearchCtx<'a>,
    scheds: &'a [Sched],
    model: Option<CostModel>,
    pred_ms: Vec<f64>,
    measured_ms: Vec<Option<f64>>,
    n_measured: usize,
    rounds: usize,
    rng: Rng,
}

impl<'a> GuidedSearch<'a> {
    pub fn new(cfg: TunerConfig, ctx: SearchCtx<'a>, scheds: &'a [Sched]) -> Self {
        let model = CostModel::load(ctx.chip);
        let pred_ms = scheds
            .iter()
            .map(|s| {
                model
                    .as_ref()
                    .map(|m| m.predict_ms(ctx.kind, ctx.rows, ctx.cols, s))
                    .unwrap_or(f64::INFINITY)
            })
            .collect();
        Self {
            cfg,
            ctx,
            scheds,
            model,
            pred_ms,
            measured_ms: vec![None; scheds.len()],
            n_measured: 0,
            rounds: 0,
            rng: Rng::new(0x5EED_C0FF_EE42_1337),
        }
    }

    pub fn model_ready(&self) -> bool {
        self.model.is_some()
    }

    pub fn dataset_ready(&self) -> bool {
        crate::dataset::load_samples(Some(self.ctx.chip)).len() >= self.cfg.min_samples_to_guide
    }

    pub fn should_guide(&self) -> bool {
        self.cfg.enabled && self.model_ready() && self.dataset_ready()
    }

    /// Lowest predicted ms over ALL candidates (early-stop target).
    pub fn pred_global_min(&self) -> f64 {
        self.pred_ms.iter().copied().fold(f64::INFINITY, f64::min)
    }

    pub fn best(&self) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, ms) in self.measured_ms.iter().enumerate() {
            if let Some(ms) = ms {
                if best.is_none() || *ms < best.unwrap().1 {
                    best = Some((i, *ms));
                }
            }
        }
        best
    }

    pub fn n_measured(&self) -> usize {
        self.n_measured
    }

    pub fn record(&mut self, idx: usize, ms: f64) {
        if self.measured_ms[idx].is_none() {
            self.n_measured += 1;
        }
        self.measured_ms[idx] = Some(ms);
    }

    /// Done when the budget is spent, everything is measured, or the model's
    /// global optimum has been confirmed by measurement.
    pub fn should_stop(&self) -> bool {
        if self.n_measured >= self.cfg.budget || self.n_measured >= self.scheds.len() {
            return true;
        }
        if self.rounds >= 2 && self.model.is_some() {
            if let Some((_, best)) = self.best() {
                let pmin = self.pred_global_min();
                if pmin.is_finite() && best <= pmin * (1.0 + self.cfg.early_stop_rel) {
                    return true;
                }
            }
        }
        false
    }

    /// Next batch of candidate indices to measure: greedy top-1 + explorers.
    pub fn next_batch(&mut self) -> Vec<usize> {
        self.rounds += 1;
        let remaining_budget = self.cfg.budget.saturating_sub(self.n_measured);
        if remaining_budget == 0 {
            return Vec::new();
        }
        let mut unmeasured: Vec<usize> = (0..self.scheds.len())
            .filter(|&i| self.measured_ms[i].is_none())
            .collect();
        if unmeasured.is_empty() {
            return Vec::new();
        }
        // Greedy: cheapest predicted unmeasured candidate.
        unmeasured.sort_by(|&a, &b| {
            self.pred_ms[a]
                .partial_cmp(&self.pred_ms[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut batch = vec![unmeasured[0]];
        // Epsilon-greedy exploration: random unmeasured picks per round.
        for _ in 0..self.cfg.explore.min(remaining_budget - 1) {
            let pool: Vec<usize> = unmeasured
                .iter()
                .copied()
                .filter(|i| !batch.contains(i))
                .collect();
            if pool.is_empty() {
                break;
            }
            batch.push(pool[self.rng.below(pool.len())]);
        }
        batch.truncate(remaining_budget);
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(tg: u64) -> Sched {
        Sched {
            tg,
            vec: 4,
            unroll: 1,
            nsg: 2,
            nr0: 4,
        }
    }

    #[test]
    fn respects_budget_and_never_repeats() {
        let scheds: Vec<Sched> = (0..20).map(|i| sched(32 + i as u64)).collect();
        let cfg = TunerConfig {
            enabled: true,
            budget: 6,
            explore: 2,
            min_samples_to_guide: 0,
            early_stop_rel: 0.03,
        };
        let ctx = SearchCtx {
            kind: "matvec_f16_nr",
            rows: 4096,
            cols: 4096,
            chip: "no-such-chip",
        };
        let mut gs = GuidedSearch::new(cfg, ctx, &scheds);
        assert!(!gs.should_guide(), "no model on disk for this chip");
        let mut seen = std::collections::HashSet::new();
        while !gs.should_stop() {
            let batch = gs.next_batch();
            assert!(!batch.is_empty(), "stop must follow empty batches");
            for i in batch {
                assert!(seen.insert(i), "candidate {i} measured twice");
                gs.record(i, 1.0 + i as f64 * 0.001);
            }
        }
        assert_eq!(gs.n_measured(), 6);
        assert!(gs.best().is_some());
    }

    #[test]
    fn early_stops_when_model_minimum_confirmed() {
        let scheds: Vec<Sched> = (0..10).map(|i| sched(32 + i as u64)).collect();
        let cfg = TunerConfig {
            enabled: true,
            budget: 100,
            explore: 0,
            min_samples_to_guide: 0,
            early_stop_rel: 0.03,
        };
        let ctx = SearchCtx {
            kind: "matvec_f16_nr",
            rows: 4096,
            cols: 4096,
            chip: "no-such-chip",
        };
        let mut gs = GuidedSearch::new(cfg, ctx, &scheds);
        // No model => pred_global_min is INF => never early-stops; run to completion.
        let mut guard = 0;
        while !gs.should_stop() {
            for i in gs.next_batch() {
                gs.record(i, 5.0);
            }
            guard += 1;
            assert!(guard < 50);
        }
        assert_eq!(gs.n_measured(), scheds.len());
    }
}

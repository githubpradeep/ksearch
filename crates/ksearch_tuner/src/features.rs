//! Feature extraction: (op kind, problem dims, schedule) -> dense f32 vector.

use ksearch_ir::OptSchedule;
use serde::{Deserialize, Serialize};

/// Serde-friendly mirror of [`OptSchedule`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sched {
    pub tg: u64,
    pub vec: u32,
    pub unroll: u32,
    pub nsg: u32,
    pub nr0: u32,
}

impl From<OptSchedule> for Sched {
    fn from(s: OptSchedule) -> Self {
        Self {
            tg: s.tg,
            vec: s.vec,
            unroll: s.unroll,
            nsg: s.nsg,
            nr0: s.nr0,
        }
    }
}

impl From<Sched> for OptSchedule {
    fn from(s: Sched) -> Self {
        Self {
            tg: s.tg,
            vec: s.vec,
            unroll: s.unroll,
            nsg: s.nsg,
            nr0: s.nr0,
        }
    }
}

/// One timed measurement: everything needed to re-train and to score new shapes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    /// Plan kind: `matvec_f16_nr` | `matvec_q4k` | `matvec_q6k` | `matvec_f32`.
    pub kind: String,
    pub rows: usize,
    pub cols: usize,
    #[serde(default)]
    pub chip: String,
    pub sched: Sched,
    /// Median kernel time in milliseconds.
    pub ms: f64,
    #[serde(default)]
    pub ts: u64,
}

impl Sample {
    pub fn sched_opt(&self) -> OptSchedule {
        OptSchedule {
            tg: self.sched.tg,
            vec: self.sched.vec,
            unroll: self.sched.unroll,
            nsg: self.sched.nsg,
            nr0: self.sched.nr0,
        }
    }
}

/// Layout is pinned in saved models (`n_features`); never reorder, only append.
pub const N_FEATURES: usize = 18;

fn log2f(x: f32) -> f32 {
    x.max(1.0).log2()
}

/// Weight bytes per element by plan kind (Q4_K ~144B/256, Q6_K ~210B/256).
fn bytes_per_elem(kind: &str) -> f32 {
    match kind {
        "matvec_q4k" => 144.0 / 256.0,
        "matvec_q6k" => 210.0 / 256.0,
        "matvec_f16_nr" => 2.0,
        _ => 4.0,
    }
}

fn kind_onehot(kind: &str) -> [f32; 4] {
    match kind {
        "matvec_f16_nr" => [0.0, 1.0, 0.0, 0.0],
        "matvec_q4k" => [0.0, 0.0, 1.0, 0.0],
        "matvec_q6k" => [0.0, 0.0, 0.0, 1.0],
        _ => [1.0, 0.0, 0.0, 0.0],
    }
}

/// Dense feature vector handed to the cost model.
///
/// Mixes raw knobs (tg/vec/unroll/nsg/nr0) with hardware-shaped proxies the
/// trees cannot easily invent: threads per TG, K-work per thread, TG count,
/// per-TG DRAM traffic, total traffic, and flop:byte balance.
pub fn features(kind: &str, rows: usize, cols: usize, s: &Sched) -> Vec<f32> {
    let oh = kind_onehot(kind);
    let tg = s.tg.max(1) as f32;
    let nsg = s.nsg.max(1) as f32;
    let nr0 = s.nr0.max(1) as f32;
    let vec = s.vec.max(1) as f32;
    let unroll = s.unroll.max(1) as f32;

    let threads_per_tg = tg * nsg;
    let threadgroups = (rows as f32 / nr0).ceil().max(1.0);
    let elems_per_thread = cols as f32 / (threads_per_tg * vec * unroll);
    let bpe = bytes_per_elem(kind);
    let traffic_per_tg = bpe * cols as f32 * nr0 + 2.0 * cols as f32;
    let total_traffic = bpe * rows as f32 * cols as f32 + 2.0 * cols as f32 + 2.0 * rows as f32;
    let flops = 2.0 * rows as f32 * cols as f32;

    vec![
        oh[0],
        oh[1],
        oh[2],
        oh[3],
        log2f(rows as f32),
        log2f(cols as f32),
        log2f((rows * cols) as f32),
        log2f(tg),
        vec,
        unroll,
        nsg,
        log2f(nr0),
        log2f(threads_per_tg),
        log2f(elems_per_thread),
        log2f(threadgroups),
        log2f(traffic_per_tg),
        log2f(total_traffic),
        flops / total_traffic.max(1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_layout_is_stable() {
        let s = Sched {
            tg: 64,
            vec: 4,
            unroll: 2,
            nsg: 2,
            nr0: 8,
        };
        let f = features("matvec_q4k", 4096, 3072, &s);
        assert_eq!(f.len(), N_FEATURES);
        assert!(f.iter().all(|v| v.is_finite()));
        // kind one-hot prefix
        assert_eq!(&f[0..4], &[0.0, 0.0, 1.0, 0.0]);
    }
}

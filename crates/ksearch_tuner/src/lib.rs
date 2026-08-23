//! ksearch_tuner: learned cost model for Metal kernel autotuning.
//!
//! Pipeline:
//!
//! ```text
//! every BEAM/tuner timing ──▶ samples.jsonl (append-only)
//!                              │
//!              histogram-GBDT on log(ms)   ← from scratch, no Python
//!                              │
//!     guided search: measure only top-K predicted + ε-greedy explorers
//! ```
//!
//! Same idea as TVM's XGBoost cost model, but trained purely on timings this
//! machine produced, and it transfers across matvec shapes: a new shape is
//! scored instantly instead of grid-searching ~48–144 kernels.

pub mod dataset;
pub mod features;
pub mod gbdt;
pub mod model;
pub mod search;

pub use features::{Sample, Sched, N_FEATURES};
pub use model::{CostModel, EvalSummary};
pub use search::{GuidedSearch, SearchCtx, TunerConfig};

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

//! Whole-model / kernel plan cache: hash(op + dims + chip) → OptSchedule + optional MSL fingerprint.

use crate::beam::{beam_cache_dir, load_beam_cache, save_beam_cache, BeamCacheEntry};
use ksearch_ir::OptSchedule;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Plan key for any scheduled kernel (matvec, rmsnorm, sdpa, …).
pub fn plan_key(kind: &str, dims: &[usize], chip: &str) -> String {
    let mut h = DefaultHasher::new();
    kind.hash(&mut h);
    for d in dims {
        d.hash(&mut h);
    }
    chip.hash(&mut h);
    let dim_s = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    format!("{kind}_{dim_s}_{:016x}", h.finish())
}

pub fn load_plan(kind: &str, dims: &[usize], chip: &str) -> Option<OptSchedule> {
    let key = plan_key(kind, dims, chip);
    load_beam_cache(&key).map(|e| e.schedule)
}

pub fn save_plan(kind: &str, dims: &[usize], chip: &str, schedule: OptSchedule, ms: f64) {
    let key = plan_key(kind, dims, chip);
    save_beam_cache(
        &key,
        &BeamCacheEntry {
            schedule,
            ms,
        },
    );
}

pub fn plan_cache_dir() -> std::path::PathBuf {
    beam_cache_dir().parent().map(|p| p.join("plan")).unwrap_or_else(|| {
        beam_cache_dir()
    })
}

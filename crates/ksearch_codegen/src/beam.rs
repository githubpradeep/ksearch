//! OptOp BEAM search helpers + on-disk schedule cache.

use crate::MetalKernelSource;
use ksearch_ir::OptSchedule;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct BeamCacheEntry {
    pub schedule: OptSchedule,
    pub ms: f64,
}

#[derive(Clone, Debug)]
pub struct BeamSearchResult {
    pub schedule: OptSchedule,
    pub ms: f64,
    pub from_cache: bool,
    pub kernel: MetalKernelSource,
}

/// Discrete TG / VEC / UNROLL candidates for dense F32 matvec.
pub fn beam_matvec_candidates() -> Vec<OptSchedule> {
    let mut out = Vec::new();
    for &tg in &[32u64, 64, 128] {
        for &vec in &[1u32, 2, 4] {
            for &unroll in &[1u32, 2, 4] {
                out.push(OptSchedule {
                    tg,
                    vec,
                    unroll,
                    nsg: 2,
                    nr0: 4,
                });
            }
        }
    }
    out
}

/// Discrete NSG / NR0 candidates for Q4_K dtype-fused matvec.
pub fn beam_matvec_q4k_candidates() -> Vec<OptSchedule> {
    let mut out = Vec::new();
    for &nsg in &[1u32, 2, 4] {
        for &nr0 in &[2u32, 4, 8] {
            out.push(OptSchedule {
                tg: (nsg as u64) * 32,
                vec: 1,
                unroll: 1,
                nsg,
                nr0,
            });
        }
    }
    out
}

pub fn beam_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("KSEARCH_BEAM_CACHE") {
        return PathBuf::from(p);
    }
    dirs_fallback()
}

fn dirs_fallback() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("ksearch").join("beam");
    }
    PathBuf::from(".ksearch_beam_cache")
}

pub fn beam_cache_key(op: &str, rows: usize, cols: usize, chip: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    op.hash(&mut h);
    rows.hash(&mut h);
    cols.hash(&mut h);
    chip.hash(&mut h);
    format!("{op}_{rows}x{cols}_{:016x}", h.finish())
}

pub fn load_beam_cache(key: &str) -> Option<BeamCacheEntry> {
    let path = beam_cache_dir().join(format!("{key}.txt"));
    let text = fs::read_to_string(path).ok()?;
    let mut tg = None;
    let mut vec = None;
    let mut unroll = None;
    let mut nsg = None;
    let mut nr0 = None;
    let mut ms = None;
    for line in text.lines() {
        let mut parts = line.splitn(2, '=');
        let k = parts.next()?.trim();
        let v = parts.next()?.trim();
        match k {
            "tg" => tg = v.parse().ok(),
            "vec" => vec = v.parse().ok(),
            "unroll" => unroll = v.parse().ok(),
            "nsg" => nsg = v.parse().ok(),
            "nr0" => nr0 = v.parse().ok(),
            "ms" => ms = v.parse().ok(),
            _ => {}
        }
    }
    Some(BeamCacheEntry {
        schedule: OptSchedule {
            tg: tg?,
            vec: vec?,
            unroll: unroll?,
            nsg: nsg.unwrap_or(2),
            nr0: nr0.unwrap_or(4),
        },
        ms: ms?,
    })
}

pub fn save_beam_cache(key: &str, entry: &BeamCacheEntry) {
    let dir = beam_cache_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(format!("{key}.txt"));
    let body = format!(
        "tg={}\nvec={}\nunroll={}\nnsg={}\nnr0={}\nms={:.6}\n",
        entry.schedule.tg,
        entry.schedule.vec,
        entry.schedule.unroll,
        entry.schedule.nsg,
        entry.schedule.nr0,
        entry.ms
    );
    let _ = fs::write(path, body);
}

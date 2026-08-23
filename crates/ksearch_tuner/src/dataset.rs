//! Append-only JSONL dataset of kernel timings + load helpers.
//!
//! Every BEAM/tuner measurement lands here, so the cost model keeps improving
//! passively as the compiler is used. One line = one [`Sample`].

use crate::features::Sample;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

pub fn tuner_dir() -> PathBuf {
    if let Ok(p) = std::env::var("KSEARCH_TUNER_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("ksearch").join("tuner");
    }
    PathBuf::from(".ksearch_tuner")
}

pub fn dataset_path() -> PathBuf {
    tuner_dir().join("samples.jsonl")
}

pub fn append_sample(sample: &Sample) -> std::io::Result<()> {
    let dir = tuner_dir();
    fs::create_dir_all(&dir)?;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dataset_path())?;
    let mut line = serde_json::to_string(sample).expect("sample serializes");
    line.push('\n');
    f.write_all(line.as_bytes())
}

/// Load samples, optionally restricted to one chip; malformed lines are skipped.
pub fn load_samples(chip_filter: Option<&str>) -> Vec<Sample> {
    let Ok(f) = File::open(dataset_path()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let Ok(line) = line else { continue };
        let Ok(s) = serde_json::from_str::<Sample>(line.trim()) else {
            continue;
        };
        if let Some(c) = chip_filter {
            if s.chip != c {
                continue;
            }
        }
        if !s.ms.is_finite() || s.ms <= 0.0 || s.rows == 0 || s.cols == 0 {
            continue;
        }
        out.push(s);
    }
    out
}

/// Keep only the newest `keep` lines (disk hygiene); returns lines dropped.
pub fn truncate_dataset(keep: usize) -> std::io::Result<usize> {
    let path = dataset_path();
    let Ok(f) = File::open(&path) else {
        return Ok(0);
    };
    let lines: Vec<String> = BufReader::new(f).lines().map_while(Result::ok).collect();
    if lines.len() <= keep {
        return Ok(0);
    }
    let drop = lines.len() - keep;
    fs::create_dir_all(tuner_dir())?;
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        for l in &lines[drop..] {
            writeln!(w, "{l}")?;
        }
    }
    fs::rename(&tmp, &path)?;
    Ok(drop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Sched;

    fn sample(rows: usize, ms: f64) -> Sample {
        Sample {
            kind: "matvec_q4k".into(),
            rows,
            cols: 3072,
            chip: "test-chip".into(),
            sched: Sched {
                tg: 32,
                vec: 1,
                unroll: 1,
                nsg: 2,
                nr0: 4,
            },
            ms,
            ts: 0,
        }
    }

    #[test]
    fn append_load_truncate_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ksearch_tuner_test_{}", std::process::id()));
        std::env::set_var("KSEARCH_TUNER_CACHE", &dir);
        let _ = fs::remove_dir_all(&dir);

        assert!(load_samples(Some("test-chip")).is_empty());
        append_sample(&sample(1024, 0.05)).unwrap();
        append_sample(&sample(2048, 0.09)).unwrap();
        // A different chip must be filterable out.
        let mut other = sample(4096, 0.2);
        other.chip = "other".into();
        append_sample(&other).unwrap();

        let mine = load_samples(Some("test-chip"));
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].rows, 1024);

        assert_eq!(truncate_dataset(1).unwrap(), 2);
        assert_eq!(load_samples(None).len(), 1);

        let _ = fs::remove_dir_all(&dir);
        std::env::remove_var("KSEARCH_TUNER_CACHE");
    }
}

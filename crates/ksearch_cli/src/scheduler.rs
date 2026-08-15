//! Decode-before-prefill slot scheduler (llama.cpp / metal-llm-server serving split).
//!
//! GPU work is serial (shared activation scratch). Concurrency is N KV slots:
//! each request owns a slot; a tick runs one decode token per decoding slot,
//! then chunked prefill on prefilling slots.

use anyhow::Result;
use ksearch_gemma::{GemmaPrimModel, KvPool, SlotId, PREFILL_CHUNK_SIZE};
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

pub const EOS_IDS: [u32; 2] = [1, 106];
pub const BUILTIN_STOPS: [&str; 3] = ["<turn|>", "<|turn>", "<tool_call|>"];

#[derive(Clone, Debug)]
pub enum StreamEvent {
    Token(u32),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

pub struct InferenceRequest {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub temperature: f32,
    pub min_p: f32,
    pub seed: u32,
    pub events: UnboundedSender<StreamEvent>,
}

enum Phase {
    Prefill { cursor: usize },
    Decode { last_token: u32 },
}

struct Active {
    slot: SlotId,
    prompt_ids: Vec<u32>,
    generated: Vec<u32>,
    text: String,
    max_tokens: usize,
    stop: Vec<String>,
    temperature: f32,
    min_p: f32,
    rng: u32,
    phase: Phase,
    events: UnboundedSender<StreamEvent>,
    finished: bool,
    admitted_at: Instant,
    prefill_s: f64,
    decode_s: f64,
}

pub struct SchedulerConfig {
    pub slots: usize,
    pub prefill_chunk: usize,
    pub prefill_tokens_per_tick: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            slots: 4,
            prefill_chunk: PREFILL_CHUNK_SIZE,
            // Long prompts: spend a full tick on prefill when no decode work.
            prefill_tokens_per_tick: 2048,
        }
    }
}

pub fn spawn_scheduler(
    mut model: GemmaPrimModel,
    mut pool: KvPool,
    cfg: SchedulerConfig,
    jobs: Receiver<InferenceRequest>,
) -> Result<()> {
    let mut waiting: VecDeque<InferenceRequest> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut prefill_cursor = 0usize;

    loop {
        loop {
            match jobs.try_recv() {
                Ok(req) => waiting.push_back(req),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(()),
            }
        }

        while pool.active_batch().len() < cfg.slots {
            let Some(req) = waiting.pop_front() else {
                break;
            };
            if req.prompt_ids.is_empty() {
                let _ = req.events.send(StreamEvent::Error("empty prompt".into()));
                continue;
            }
            let events = req.events.clone();
            match admit(&mut pool, req) {
                Ok(a) => active.push(a),
                Err(e) => {
                    let _ = events.send(StreamEvent::Error(format!("admit failed: {e}")));
                }
            }
        }

        if active.is_empty() {
            match jobs.recv_timeout(Duration::from_millis(50)) {
                Ok(req) => waiting.push_back(req),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            continue;
        }

        decode_round(&mut model, &mut pool, &mut active)?;
        prefill_round(
            &mut model,
            &mut pool,
            &mut active,
            &cfg,
            &mut prefill_cursor,
        )?;
        reap_finished(&mut pool, &mut active);
    }
}

fn admit(pool: &mut KvPool, req: InferenceRequest) -> Result<Active> {
    let slot = pool.alloc()?;
    pool.clear_slot(slot)?;
    let mut stop = req.stop;
    for s in BUILTIN_STOPS {
        if !stop.iter().any(|x| x == s) {
            stop.push(s.to_string());
        }
    }
    Ok(Active {
        slot,
        prompt_ids: req.prompt_ids,
        generated: Vec::new(),
        text: String::new(),
        max_tokens: req.max_tokens.max(1),
        stop,
        temperature: req.temperature,
        min_p: req.min_p,
        rng: req.seed | 1,
        phase: Phase::Prefill { cursor: 0 },
        events: req.events,
        finished: false,
        admitted_at: Instant::now(),
        prefill_s: 0.0,
        decode_s: 0.0,
    })
}

fn decode_round(
    model: &mut GemmaPrimModel,
    pool: &mut KvPool,
    active: &mut [Active],
) -> Result<()> {
    for req in active.iter_mut() {
        if req.finished {
            continue;
        }
        let Phase::Decode { last_token } = req.phase else {
            continue;
        };
        if req.generated.len() >= req.max_tokens {
            finish(req, "length");
            continue;
        }
        if pool.seq_len(req.slot).unwrap_or(0) >= model.max_seq() {
            finish(req, "length");
            continue;
        }
        model.bind_slot(pool, req.slot)?;
        let t0 = Instant::now();
        req.rng = req.rng.wrapping_add(0x9E3779B9);
        match model.decode_token_sampled(last_token, req.temperature, req.min_p, req.rng) {
            Ok(next) => {
                req.decode_s += t0.elapsed().as_secs_f64();
                if let Err(e) = pool.bump_len(req.slot) {
                    let _ = req.events.send(StreamEvent::Error(e.to_string()));
                    finish(req, "error");
                    continue;
                }
                req.phase = Phase::Decode { last_token: next };
                if EOS_IDS.contains(&next) {
                    finish(req, "stop");
                    continue;
                }
                let piece = model
                    .vocab
                    .as_ref()
                    .map(|v| v.decode(&[next], false))
                    .unwrap_or_default();
                req.text.push_str(&piece);
                if let Some(reason) = hit_stop(&req.text, &req.stop) {
                    trim_stop(&mut req.text, &reason);
                    finish(req, "stop");
                    continue;
                }
                req.generated.push(next);
                if repeating_token(&req.generated, 24) {
                    finish(req, "stop");
                    continue;
                }
                if req.events.send(StreamEvent::Token(next)).is_err() {
                    finish(req, "cancelled");
                    continue;
                }
                if req.generated.len() >= req.max_tokens {
                    finish(req, "length");
                }
            }
            Err(e) => {
                req.decode_s += t0.elapsed().as_secs_f64();
                let _ = req.events.send(StreamEvent::Error(e.to_string()));
                finish(req, "error");
            }
        }
    }
    Ok(())
}

fn prefill_round(
    model: &mut GemmaPrimModel,
    pool: &mut KvPool,
    active: &mut [Active],
    cfg: &SchedulerConfig,
    next_idx: &mut usize,
) -> Result<()> {
    let n = active.len();
    if n == 0
        || !active
            .iter()
            .any(|a| matches!(a.phase, Phase::Prefill { .. }))
    {
        return Ok(());
    }
    let decoding = active.iter().any(|a| {
        !a.finished && matches!(a.phase, Phase::Decode { .. })
    });
    let mut budget = cfg.prefill_tokens_per_tick.max(cfg.prefill_chunk);
    let mut idx = *next_idx % n;
    for _ in 0..n {
        if budget == 0 {
            break;
        }
        let req = &mut active[idx];
        idx = (idx + 1) % n;
        if req.finished {
            continue;
        }
        let Phase::Prefill { cursor } = req.phase else {
            continue;
        };
        // Last prompt token is the first decode input (logits), not a prefill step.
        let prefill_end = req.prompt_ids.len().saturating_sub(1);
        if cursor >= prefill_end {
            enter_decode(req);
            continue;
        }
        model.bind_slot(pool, req.slot)?;
        // Fairness vs decode: small chunks when mixed; otherwise eat the budget.
        let cap = if decoding {
            cfg.prefill_chunk
        } else {
            budget
        };
        let take = budget.min(cap).min(prefill_end - cursor);
        let t0 = Instant::now();
        let slice = &req.prompt_ids[cursor..cursor + take];
        if let Err(e) = model.prefill_chunk(slice) {
            req.prefill_s += t0.elapsed().as_secs_f64();
            let _ = req.events.send(StreamEvent::Error(e.to_string()));
            finish(req, "error");
        } else if let Err(e) = pool.bump_len_by(req.slot, take) {
            req.prefill_s += t0.elapsed().as_secs_f64();
            let _ = req.events.send(StreamEvent::Error(e.to_string()));
            finish(req, "error");
        } else {
            req.prefill_s += t0.elapsed().as_secs_f64();
            budget = budget.saturating_sub(take);
            let c = cursor + take;
            if matches!(req.phase, Phase::Prefill { .. }) {
                if c >= prefill_end {
                    enter_decode(req);
                } else {
                    req.phase = Phase::Prefill { cursor: c };
                }
            }
        }
    }
    *next_idx = idx;
    Ok(())
}

fn reap_finished(pool: &mut KvPool, active: &mut Vec<Active>) {
    active.retain(|req| {
        if req.finished {
            let _ = pool.free(req.slot);
            false
        } else {
            true
        }
    });
}

fn enter_decode(req: &mut Active) {
    let last = req.prompt_ids[req.prompt_ids.len() - 1];
    req.phase = Phase::Decode { last_token: last };
    let n = req.prompt_ids.len().saturating_sub(1);
    eprintln!(
        "[serve] slot={} prefill={} tok {:.1} tok/s ({:.2}s)",
        req.slot.0,
        n,
        n as f64 / req.prefill_s.max(1e-6),
        req.prefill_s
    );
}

fn finish(req: &mut Active, reason: &str) {
    if req.finished {
        return;
    }
    req.finished = true;
    if reason != "error" {
        let _ = req.events.send(StreamEvent::Done {
            finish_reason: reason.to_string(),
            prompt_tokens: req.prompt_ids.len(),
            completion_tokens: req.generated.len(),
        });
    }
    let prefill_n = req.prompt_ids.len().saturating_sub(1);
    let decode_n = req.generated.len();
    eprintln!(
        "[serve] slot={} prefill={} tok {:.1} tok/s  decode={} tok {:.1} tok/s  reason={reason} e2e={:.2}s",
        req.slot.0,
        prefill_n,
        prefill_n as f64 / req.prefill_s.max(1e-6),
        decode_n,
        decode_n as f64 / req.decode_s.max(1e-6),
        req.admitted_at.elapsed().as_secs_f64()
    );
}

fn repeating_token(ids: &[u32], n: usize) -> bool {
    if ids.len() < n {
        return false;
    }
    let last = *ids.last().unwrap();
    ids[ids.len() - n..].iter().all(|&t| t == last)
}

fn hit_stop(text: &str, stops: &[String]) -> Option<String> {
    stops
        .iter()
        .find(|s| !s.is_empty() && text.contains(s.as_str()))
        .cloned()
}

fn trim_stop(text: &mut String, stop: &str) {
    if let Some(i) = text.find(stop) {
        text.truncate(i);
    }
}

/// Channel used by HTTP handlers to enqueue work.
pub type JobSender = SyncSender<InferenceRequest>;

//! Decode-before-prefill slot scheduler (llama.cpp / metal-llm-server serving split).
//!
//! GPU work is serial (shared activation scratch). Concurrency is N KV slots:
//! each request owns a slot; a tick runs one decode token per decoding slot,
//! then chunked prefill on prefilling slots.

use anyhow::Result;
use ksearch_gemma::{GemmaPrimModel, KvPool, SlotId, PREFILL_CHUNK_SIZE};
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

pub const EOS_IDS: [u32; 2] = [1, 106];
pub const BUILTIN_STOPS: [&str; 2] = ["<turn|>", "<|turn>"];

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
    phase: Phase,
    events: UnboundedSender<StreamEvent>,
    finished: bool,
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
            prefill_tokens_per_tick: 256,
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
        phase: Phase::Prefill { cursor: 0 },
        events: req.events,
        finished: false,
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
        match model.decode_token(last_token) {
            Ok(next) => {
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
                    .map(|v| v.decode(&[next], true))
                    .unwrap_or_default();
                req.text.push_str(&piece);
                if let Some(reason) = hit_stop(&req.text, &req.stop) {
                    trim_stop(&mut req.text, &reason);
                    finish(req, "stop");
                    continue;
                }
                req.generated.push(next);
                if req.events.send(StreamEvent::Token(next)).is_err() {
                    finish(req, "cancelled");
                    continue;
                }
                if req.generated.len() >= req.max_tokens {
                    finish(req, "length");
                }
            }
            Err(e) => {
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
            let last = req.prompt_ids[req.prompt_ids.len() - 1];
            req.phase = Phase::Decode { last_token: last };
            continue;
        }
        model.bind_slot(pool, req.slot)?;
        let take = budget.min(cfg.prefill_chunk).min(prefill_end - cursor);
        let mut c = cursor;
        for _ in 0..take {
            if let Err(e) = model.prefill_token(req.prompt_ids[c]) {
                let _ = req.events.send(StreamEvent::Error(e.to_string()));
                finish(req, "error");
                c = prefill_end;
                break;
            }
            if let Err(e) = pool.bump_len(req.slot) {
                let _ = req.events.send(StreamEvent::Error(e.to_string()));
                finish(req, "error");
                c = prefill_end;
                break;
            }
            c += 1;
            budget = budget.saturating_sub(1);
        }
        if matches!(req.phase, Phase::Prefill { .. }) {
            if c >= prefill_end {
                let last = req.prompt_ids[req.prompt_ids.len() - 1];
                req.phase = Phase::Decode { last_token: last };
            } else {
                req.phase = Phase::Prefill { cursor: c };
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
    eprintln!(
        "[serve] slot={} prompt={} completion={} reason={reason}",
        req.slot.0,
        req.prompt_ids.len(),
        req.generated.len()
    );
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

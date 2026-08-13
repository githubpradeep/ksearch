//! OpenAI-compatible HTTP server (`/v1/chat/completions`).
//!
//! HTTP thread tokenizes and queues; a dedicated scheduler thread owns the GPU
//! (KV slots, chunked prefill, decode-before-prefill).

use crate::gemma4_chat::{
    apply_chat_template, awaits_tool_call, infer_tool_calls_without_generation, resolve_tool_calls,
    split_reasoning_and_content, ChatMessage, Tool, ToolCall,
};
use crate::scheduler::{
    spawn_scheduler, InferenceRequest, JobSender, SchedulerConfig, StreamEvent,
};
use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, StreamExt};
use ksearch_gemma::{GemmaConfig, GemmaPrimModel};
use ksearch_gguf::{build_tokenizer_from_gguf, encode_prompt, Gguf};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::convert::Infallible;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::mpsc::{self, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

const DEFAULT_MAX_TOKENS: usize = 1024;

fn default_temperature() -> f32 {
    1.0
}

fn default_min_p() -> f32 {
    0.05
}

#[derive(Clone)]
struct AppState {
    jobs: JobSender,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    max_seq: usize,
    model_ctx: usize,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_min_p")]
    min_p: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopArg>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    #[serde(default)]
    tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopArg {
    One(String),
    Many(Vec<String>),
}

impl StopArg {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopArg::One(s) => vec![s],
            StopArg::Many(v) => v,
        }
    }
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: AssistantMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Clone)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    code: String,
}

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Value>,
}

pub struct ServeArgs {
    pub gguf: PathBuf,
    pub port: u16,
    pub max_seq: usize,
    pub slots: usize,
}

pub async fn run_server(args: ServeArgs) -> Result<()> {
    let model_id = args
        .gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("ksearch-gemma")
        .to_string();
    let g = Gguf::open(&args.gguf);
    let tokenizer = build_tokenizer_from_gguf(&g).map_err(|e| anyhow::anyhow!(e))?;
    let tokenizer = Arc::new(tokenizer);
    let cfg = GemmaConfig::from_gguf(&g)?;
    let model_ctx = cfg.context_length.max(256);
    let max_seq = args.max_seq.max(256);
    if max_seq > model_ctx {
        eprintln!(
            "[serve] warning: --max-seq {max_seq} > model context_length {model_ctx} (RoPE past training length)"
        );
    }
    let slots = args.slots.max(1);
    let kv_mb = cfg.kv_q40_bytes(max_seq, slots) as f64 / 1e6;
    eprintln!("[serve] ctx={max_seq} model_ctx={model_ctx} slots={slots} kv_q40≈{kv_mb:.0} MB");

    let (job_tx, job_rx) = mpsc::sync_channel::<InferenceRequest>(32);
    let gguf = args.gguf.clone();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("ksearch-sched".into())
        .spawn(move || {
            let loaded = GemmaPrimModel::load(&gguf, max_seq);
            match loaded {
                Ok(model) => match model.make_kv_pool(slots) {
                    Ok(pool) => {
                        let _ = ready_tx.send(Ok(()));
                        let cfg = SchedulerConfig {
                            slots,
                            ..SchedulerConfig::default()
                        };
                        if let Err(e) = spawn_scheduler(model, pool, cfg, job_rx) {
                            eprintln!("[serve] scheduler exited: {e}");
                        }
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                },
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .context("spawn scheduler")?;
    ready_rx.recv().context("scheduler handshake")??;

    let state = AppState {
        jobs: job_tx,
        tokenizer,
        model_id: model_id.clone(),
        max_seq,
        model_ctx,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    eprintln!(
        "[serve] OpenAI chat at http://{addr}/v1/chat/completions  model={model_id} slots={slots} max_seq={max_seq}"
    );
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(st): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: st.model_id.clone(),
            object: "model",
            owned_by: "local",
        }],
    })
}

async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    match handle_chat(st, req).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

struct ApiError {
    status: StatusCode,
    code: String,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorBody {
            error: ErrorDetail {
                message: self.message,
                kind: "invalid_request_error",
                code: self.code,
            },
        });
        (self.status, body).into_response()
    }
}

fn api_err(status: StatusCode, code: &str, message: impl Into<String>) -> ApiError {
    ApiError {
        status,
        code: code.into(),
        message: message.into(),
    }
}

async fn handle_chat(st: AppState, req: ChatCompletionRequest) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "empty_messages",
            "messages must be non-empty",
        ));
    }
    if !req.temperature.is_finite() || !(0.0..=5.0).contains(&req.temperature) {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "invalid_temperature",
            "temperature must be between 0 and 5",
        ));
    }
    if !req.min_p.is_finite() || !(0.0..=1.0).contains(&req.min_p) {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "invalid_min_p",
            "min_p must be between 0 and 1",
        ));
    }
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .max(1);
    let prompt = apply_chat_template(
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
    );
    let ids = encode_prompt(st.tokenizer.as_ref(), &prompt, true)
        .map_err(|e| api_err(StatusCode::BAD_REQUEST, "tokenizer_error", e))?;
    if ids.is_empty() {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "tokenizer_error",
            "prompt encoded to zero tokens",
        ));
    }
    if ids.len() + 1 > st.max_seq {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!(
                "prompt is {} tokens; max_seq is {} (need room for at least one decode step). Restart with --max-seq N (model context_length is {}).",
                ids.len(),
                st.max_seq,
                st.model_ctx
            ),
        ));
    }
    let room = st.max_seq.saturating_sub(ids.len());
    let max_tokens = max_tokens.min(room).max(1);
    let stop = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let model_name = req.model.clone().unwrap_or_else(|| st.model_id.clone());
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(true);

    let n_tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let roles: Vec<&str> = req.messages.iter().map(|m| m.role.as_str()).collect();
    let chars: Vec<usize> = req
        .messages
        .iter()
        .map(|m| m.content.as_deref().map(|c| c.len()).unwrap_or(0))
        .collect();
    eprintln!(
        "[serve] roles={roles:?} chars={chars:?} tools={n_tools} prompt_tokens={}",
        ids.len()
    );

    let inferred = if n_tools > 0 {
        infer_tool_calls_without_generation(
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        )
    } else {
        Vec::new()
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_now();
    if !inferred.is_empty() {
        eprintln!(
            "[serve] inferred tool_calls without generation: {:?}",
            inferred
                .iter()
                .map(|c| format!("{} {}", c.function.name, c.function.arguments))
                .collect::<Vec<_>>()
        );
        if req.stream {
            return Ok(
                stream_tool_calls_only(id, created, model_name, inferred, include_usage, ids.len())
                    .into_response(),
            );
        }
        return Ok(Json(completion_with_tools(
            id,
            created,
            model_name,
            inferred,
            ids.len(),
            0,
        ))
        .into_response());
    }

    let seed = (id_seed() ^ ids.len() as u32) | 1;
    let (ev_tx, ev_rx) = tokio_mpsc::unbounded_channel();
    let job = InferenceRequest {
        prompt_ids: ids,
        max_tokens,
        stop,
        temperature: req.temperature,
        min_p: req.min_p,
        seed,
        events: ev_tx,
    };
    match st.jobs.try_send(job) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return Err(api_err(
                StatusCode::TOO_MANY_REQUESTS,
                "queue_full",
                "inference queue is full",
            ));
        }
        Err(TrySendError::Disconnected(_)) => {
            return Err(api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "scheduler thread is gone",
            ));
        }
    }

    let hold_for_tools = n_tools > 0 && awaits_tool_call(&req.messages);
    if req.stream {
        Ok(stream_sse(
            ev_rx,
            id,
            created,
            model_name,
            include_usage,
            st.tokenizer,
            hold_for_tools,
            req.messages,
            req.tools,
            req.tool_choice,
        )
        .into_response())
    } else {
        collect_sync(
            ev_rx,
            id,
            created,
            model_name,
            st.tokenizer,
            req.messages,
            req.tools,
            req.tool_choice,
        )
        .await
    }
}

async fn collect_sync(
    mut ev_rx: tokio_mpsc::UnboundedReceiver<StreamEvent>,
    id: String,
    created: u64,
    model: String,
    tokenizer: Arc<Tokenizer>,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<Value>,
) -> Result<Response, ApiError> {
    let mut tokens = Vec::new();
    let mut finish_reason = "stop".to_string();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    while let Some(ev) = ev_rx.recv().await {
        match ev {
            StreamEvent::Token(t) => tokens.push(t),
            StreamEvent::Done {
                finish_reason: fr,
                prompt_tokens: p,
                completion_tokens: c,
            } => {
                finish_reason = fr;
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            StreamEvent::Error(e) => {
                return Err(api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    e,
                ));
            }
        }
    }
    let raw = decode_tokens(&tokenizer, &tokens);
    let tool_calls = if tools.as_ref().is_some_and(|t| !t.is_empty()) {
        resolve_tool_calls(&raw, &messages, tools.as_deref(), tool_choice.as_ref())
    } else {
        Vec::new()
    };
    if !tool_calls.is_empty() {
        eprintln!(
            "[serve] parsed tool_calls from generation: {:?}",
            tool_calls
                .iter()
                .map(|c| format!("{} {}", c.function.name, c.function.arguments))
                .collect::<Vec<_>>()
        );
        return Ok(Json(completion_with_tools(
            id,
            created,
            model,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        ))
        .into_response());
    }
    let (reasoning, content) = split_reasoning_and_content(&raw);
    let reasoning_content = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };
    let body = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: Some(content),
                reasoning_content,
                tool_calls: None,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    Ok(Json(body).into_response())
}

fn completion_with_tools(
    id: String,
    created: u64,
    model: String,
    tool_calls: Vec<ToolCall>,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: None,
                reasoning_content: None,
                tool_calls: Some(tool_calls),
            },
            finish_reason: "tool_calls".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

fn stream_tool_calls_only(
    id: String,
    created: u64,
    model: String,
    tool_calls: Vec<ToolCall>,
    include_usage: bool,
    prompt_tokens: usize,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>> + Send> {
    let mut events = vec![Event::default().data(chunk_json(
        &id,
        created,
        &model,
        Some("assistant"),
        None,
        None,
        None,
        None,
    ))];
    events.extend(tool_call_sse_events(&id, created, &model, &tool_calls));
    let usage = if include_usage {
        Some(Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        })
    } else {
        None
    };
    events.push(Event::default().data(chunk_json(
        &id,
        created,
        &model,
        None,
        None,
        Some("tool_calls"),
        usage.as_ref(),
        None,
    )));
    events.push(Event::default().data("[DONE]"));
    Sse::new(stream::iter(events.into_iter().map(Ok)))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn tool_call_sse_events(id: &str, created: u64, model: &str, tool_calls: &[ToolCall]) -> Vec<Event> {
    let mut events = Vec::new();
    for (index, tc) in tool_calls.iter().enumerate() {
        events.push(Event::default().data(chunk_json(
            id,
            created,
            model,
            None,
            None,
            None,
            None,
            Some(serde_json::json!([{
                "index": index,
                "id": tc.id,
                "type": tc.call_type,
                "function": { "name": tc.function.name, "arguments": "" }
            }])),
        )));
        events.push(Event::default().data(chunk_json(
            id,
            created,
            model,
            None,
            None,
            None,
            None,
            Some(serde_json::json!([{
                "index": index,
                "function": { "arguments": tc.function.arguments }
            }])),
        )));
    }
    events
}

fn stream_sse(
    ev_rx: tokio_mpsc::UnboundedReceiver<StreamEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    tokenizer: Arc<Tokenizer>,
    hold_for_tools: bool,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<Value>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>> + Send> {
    let role_json = chunk_json(
        &id, created, &model, Some("assistant"), None, None, None, None,
    );
    let first = stream::once(async move { Ok(Event::default().data(role_json)) });
    let done = stream::once(async { Ok(Event::default().data("[DONE]")) });
    let boxed: Pin<Box<dyn futures_util::Stream<Item = Result<Event, Infallible>> + Send>> =
        if hold_for_tools {
            let buffered = stream::once(async move {
                let mut tokens = Vec::new();
                let mut prompt_tokens = 0usize;
                let mut completion_tokens = 0usize;
                let mut err: Option<String> = None;
                let mut rx = ev_rx;
                while let Some(ev) = rx.recv().await {
                    match ev {
                        StreamEvent::Token(t) => tokens.push(t),
                        StreamEvent::Done {
                            prompt_tokens: p,
                            completion_tokens: c,
                            ..
                        } => {
                            prompt_tokens = p;
                            completion_tokens = c;
                            break;
                        }
                        StreamEvent::Error(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = err {
                    return vec![Event::default().data(format!(
                    "{{\"error\":{{\"message\":{},\"type\":\"internal_error\",\"code\":\"internal_error\"}}}}",
                    serde_json::to_string(&e).unwrap_or_else(|_| "\"error\"".into())
                ))];
                }
                let text = decode_tokens(&tokenizer, &tokens);
                let calls =
                    resolve_tool_calls(&text, &messages, tools.as_deref(), tool_choice.as_ref());
                let visible = if calls.is_empty() {
                    split_reasoning_and_content(&text).1
                } else {
                    String::new()
                };
                let usage = if include_usage {
                    Some(Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    })
                } else {
                    None
                };
                let mut events = Vec::new();
                if !calls.is_empty() {
                    events.extend(tool_call_sse_events(&id, created, &model, &calls));
                    events.push(Event::default().data(chunk_json(
                        &id,
                        created,
                        &model,
                        None,
                        None,
                        Some("tool_calls"),
                        usage.as_ref(),
                        None,
                    )));
                } else {
                    if !visible.is_empty() {
                        events.push(Event::default().data(chunk_json(
                            &id,
                            created,
                            &model,
                            None,
                            Some(&visible),
                            None,
                            None,
                            None,
                        )));
                    }
                    events.push(Event::default().data(chunk_json(
                        &id,
                        created,
                        &model,
                        None,
                        None,
                        Some("stop"),
                        usage.as_ref(),
                        None,
                    )));
                }
                events
            })
            .map(|events| stream::iter(events.into_iter().map(Ok)))
            .flatten();
            Box::pin(first.chain(buffered).chain(done))
        } else {
            let rest = UnboundedReceiverStream::new(ev_rx).map(move |ev| {
                let ev = match ev {
                    StreamEvent::Token(t) => {
                        let piece = decode_tokens(&tokenizer, &[t]);
                        if piece.is_empty() {
                            return Ok(Event::default().comment("skip"));
                        }
                        Event::default().data(chunk_json(
                            &id, created, &model, None, Some(&piece), None, None, None,
                        ))
                    }
                    StreamEvent::Done {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                    } => {
                        let usage = if include_usage {
                            Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            })
                        } else {
                            None
                        };
                        Event::default().data(chunk_json(
                            &id,
                            created,
                            &model,
                            None,
                            None,
                            Some(&finish_reason),
                            usage.as_ref(),
                            None,
                        ))
                    }
                    StreamEvent::Error(e) => Event::default().data(format!(
                        "{{\"error\":{{\"message\":{},\"type\":\"internal_error\",\"code\":\"internal_error\"}}}}",
                        serde_json::to_string(&e).unwrap_or_else(|_| "\"error\"".into())
                    )),
                };
                Ok(ev)
            });
            Box::pin(first.chain(rest).chain(done))
        };
    Sse::new(boxed).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<&str>,
    finish: Option<&str>,
    usage: Option<&Usage>,
    tool_calls: Option<Value>,
) -> String {
    let body = StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role,
                content: content.map(|s| s.to_string()),
                tool_calls,
            },
            finish_reason: finish.map(|s| s.to_string()),
        }],
        usage: usage.cloned(),
    };
    serde_json::to_string(&body).unwrap_or_else(|_| "{}".into())
}

/// Gemma 4 channel / tool specials must survive (`skip_special=false`, oracle).
fn decode_tokens(tok: &Tokenizer, ids: &[u32]) -> String {
    let mut out = String::new();
    for &id in ids {
        let piece = tok.decode(&[id], false).unwrap_or_default();
        if !piece.is_empty() {
            out.push_str(&piece);
            continue;
        }
        let Some(name) = tok.id_to_token(id) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix('\u{2581}') {
            if !out.is_empty() && !out.ends_with(char::is_whitespace) {
                out.push(' ');
            }
            out.push_str(rest);
        } else {
            out.push_str(&name);
        }
    }
    out
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn id_seed() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.as_nanos() as u32).rotate_left(7) ^ (d.as_micros() as u32))
        .unwrap_or(1)
}

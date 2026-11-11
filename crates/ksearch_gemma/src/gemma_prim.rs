//! Thesis A Gemma path: Graph→AST→MSL. Q4_K weights stay packed (`Load(Q4K)` expand);
//! other quant types dequant → F16. Activations stay F16.

use crate::mtp::MtpDraft;
use crate::sample::sample_softcap_min_p;
use crate::{GemmaConfig, GenerateStats, KvPool, LayerMeta, LayerNorms, SlotId};
use anyhow::{anyhow, bail, Result};
use ksearch_compute::{Buffer, Context};
use ksearch_gguf::{dequant_row_to_f32, f32_to_f16, ggml_type, quantize_f32_to_q4k, Gguf};
use ksearch_ir::{q40_nbytes, q40_row_bytes, DType};
use ksearch_kernels::Eng;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

static PREFILL_PROF_SEEN: AtomicU32 = AtomicU32::new(0);

const PREFILL_CHUNK: usize = 256;
/// id_last + drafts in one verify forward (metal-llm-server `MAX_MTP_VERIFY_SEQ`).
const MAX_MTP_VERIFY: usize = 8;

fn mtp_draft_steps() -> usize {
    std::env::var("KSEARCH_MTP_DRAFT_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0 && v < MAX_MTP_VERIFY)
        // K=1: verify batch=2 is the only draft depth that doesn't lose to
        // baseline today (longer chains: verify grows faster than E).
        .unwrap_or(1)
}

/// Tokens per prefill chunk (also the server scheduler's default slice).
pub const PREFILL_CHUNK_SIZE: usize = PREFILL_CHUNK;

enum WeightBuf {
    F16(Buffer),
    Q4K(Buffer),
    Q5K(Buffer),
    Q6K(Buffer),
}

fn is_packed(d: DType) -> bool {
    matches!(d, DType::Q4K | DType::Q6K)
}

/// Host-side RoPE table: for each pos, `hd` F16s = cos[0..half) || sin[0..half).
fn precompute_rope_table(max_seq: usize, hd: usize, theta: f32, rope_angles: usize) -> Vec<u8> {
    let half = hd / 2;
    let rope_angles = rope_angles.min(half);
    let mut out = vec![0f32; max_seq * hd];
    for pos in 0..max_seq {
        let base = pos * hd;
        for i in 0..half {
            if i < rope_angles {
                let freq = 1.0 / (theta as f64).powf((2 * i) as f64 / hd as f64) as f32;
                let ang = pos as f32 * freq;
                out[base + i] = ang.cos();
                out[base + half + i] = ang.sin();
            } else {
                out[base + i] = 1.0;
                out[base + half + i] = 0.0;
            }
        }
    }
    let mut bytes = Vec::with_capacity(out.len() * 4);
    for v in out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}


impl WeightBuf {
    fn dtype(&self) -> DType {
        match self {
            WeightBuf::F16(_) => DType::F16,
            WeightBuf::Q4K(_) => DType::Q4K,
            WeightBuf::Q5K(_) => DType::Q5K,
            WeightBuf::Q6K(_) => DType::Q6K,
        }
    }

    fn buf(&self) -> &Buffer {
        match self {
            WeightBuf::F16(b) | WeightBuf::Q4K(b) | WeightBuf::Q5K(b) | WeightBuf::Q6K(b) => b,
        }
    }
}

pub struct GemmaPrimModel {
    pub cfg: GemmaConfig,
    pub vocab: Option<ksearch_gguf::Vocab>,
    gguf: Gguf,
    /// MTP draft head + acceptance-probe state (KSEARCH_MTP / --mtp).
    mtp: Option<MtpDraft>,
    mtp_hits: usize,
    mtp_tried: usize,
    mtp_pending: Option<u32>,
    mtp_hseed: Option<Buffer>,
    mtp_verify_logits: Option<Buffer>,
    mtp_verify_argmax: Option<Buffer>,
    ctx: Context,
    /// Tied lm_head / embed (Q4_K packed when GGUF type is Q4_K, else F16).
    token_embd: WeightBuf,
    output_norm: Buffer,
    ple_proj_norm: Buffer,
    layer_norms: Vec<LayerNorms>,
    layers: Vec<LayerMeta>,
    x: Buffer,
    x2: Buffer,
    tmp_q: Buffer,
    tmp_k: Buffer,
    tmp_v: Buffer,
    tmp_o: Buffer,
    /// F32 MWG SDPA scratch: `n_heads * NWG * (max_hd + 2)`.
    tmp_attn_mwg: Buffer,
    /// FFN intermediates (gate/up) when packings differ and fusion cannot share one weight dtype.
    tmp_ff1: Buffer,
    tmp_ff2: Buffer,
    tmp_ff3: Buffer,
    logits: Buffer,
    /// Ping-pong F32 token ids: decode step i reads `[i&1]`, writes `[(i+1)&1]`.
    tok_idx: [Buffer; 2],
    /// Prefill chunk token ids (F32, length `PREFILL_CHUNK`).
    chunk_tok: Buffer,
    ple_tok: Buffer,
    /// Packed `per_layer_token_embd` for GPU row gather (Q5_K Load expand).
    ple_embd: Option<WeightBuf>,
    /// Double-buffer host→GPU PLE token rows (prefill / fallback).
    ple_stage: [Buffer; 2],
    ple_ctx: Buffer,
    ple_tmp: Buffer,
    ple_gate: Buffer,
    ple_u: Buffer,
    ple_proj: Buffer,
    meta: Vec<Buffer>,
    rope_swa: Buffer,
    rope_full: Buffer,
    hd_swa: usize,
    hd_full: usize,
    /// Device Q4_0 KV [max_seq × hd] per owning layer (Load(Q40) expand in SDPA).
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
    eng: Eng,
    weight_bufs: HashMap<String, WeightBuf>,
    max_seq: usize,
    pub pos: usize,
    decode_src: usize,
    return_token_sync: bool,
    /// Serve-path sampling; 0 = greedy GPU argmax. Cleared after each sampled step.
    sample_temperature: f32,
    sample_min_p: f32,
    sample_seed: u32,
}

impl GemmaPrimModel {
    pub fn load(path: impl AsRef<Path>, max_seq: usize) -> Result<Self> {
        let path = path.as_ref();
        eprintln!("[Thesis A / prim] Loading GGUF {} …", path.display());
        let g = Gguf::open(path);
        let cfg = GemmaConfig::from_gguf(&g)?;
        let vocab = ksearch_gguf::Vocab::from_gguf(&g);
        eprintln!(
            "[prim] layers={} hidden={} heads={}/{} vocab={} (Q4_K weights packed; acts F16)",
            cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv, cfg.vocab
        );

        let ctx = Context::new()?;
        eprintln!("GPU: {} ({})", ctx.device_name(), ctx.backend().as_str());

        let embd_name = "token_embd.weight";
        let token_embd = if g.tensor_type(embd_name) == ggml_type::Q4_K {
            let raw = g.tensor_raw(embd_name);
            eprintln!(
                "Upload token_embd Q4_K ({:.1} MB) — lm_head uses Load(Q4K); embed dequants one row",
                raw.len() as f64 / 1e6
            );
            WeightBuf::Q4K(ctx.buffer_bytes(raw))
        } else {
            eprintln!("Dequant token_embd → F16…");
            let embd_f16 = g.dequant_to_f16_bytes(embd_name);
            eprintln!("  token_embd F16: {:.1} MB", embd_f16.len() as f64 / 1e6);
            WeightBuf::F16(ctx.buffer_bytes(&embd_f16))
        };

        let output_norm = ctx.buffer_bytes(&g.dequant_to_f16_bytes("output_norm.weight"));
        let ple_proj_norm = ctx.buffer_bytes(&g.dequant_to_f16_bytes("per_layer_proj_norm.weight"));

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut layer_norms = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            eprint!("\rMeta layer {i}/{} …", cfg.n_layers);
            let hd = cfg.head_dim(i);
            let pref = format!("blk.{i}.");
            layer_norms.push(LayerNorms {
                attn_norm: ctx.buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}attn_norm.weight"))),
                q_norm: ctx.buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}attn_q_norm.weight"))),
                k_norm: ctx.buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}attn_k_norm.weight"))),
                post_attn_norm: ctx
                    .buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}post_attention_norm.weight"))),
                ffn_norm: ctx.buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}ffn_norm.weight"))),
                post_ffw_norm: ctx
                    .buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}post_ffw_norm.weight"))),
                post_norm: ctx.buffer_bytes(&g.dequant_to_f16_bytes(&format!("{pref}post_norm.weight"))),
                layer_scale: g.dequant_to_f32(&format!("{pref}layer_output_scale.weight"))[0],
            });
            layers.push(LayerMeta {
                q_rows: cfg.n_heads * hd,
                kv_rows: cfg.n_kv * hd,
                o_in: cfg.n_heads * hd,
                ffn_inter: cfg.ffn[i],
                hd,
            });
        }
        eprintln!();

        let max_hd = cfg.head_dim_full.max(cfg.head_dim_swa);
        let max_ff = cfg.ffn.iter().copied().max().unwrap_or(6144);
        let ple_total = cfg.ple_total();

        let n_kv_owners = cfg.n_layers - cfg.shared_kv_layers;
        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        for i in 0..n_kv_owners {
            let hd = cfg.head_dim(i);
            // Q4_0 packed KV (Thesis A Load(Q40) expand in SDPA).
            kv_k.push(ctx.buffer_empty_bytes(q40_nbytes(max_seq * cfg.n_kv, hd)));
            kv_v.push(ctx.buffer_empty_bytes(q40_nbytes(max_seq * cfg.n_kv, hd)));
        }

        let mut meta = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            meta.push(
                ctx.buffer_empty_bytes(32),
            );
        }
        let rope_swa = {
            let hd = cfg.head_dim_swa;
            let bytes = precompute_rope_table(max_seq, hd, cfg.rope_theta_swa, hd / 2);
            ctx.buffer_bytes(&bytes)
        };
        let rope_full = {
            let hd = cfg.head_dim_full;
            let angles = ((hd as f32) * cfg.partial_rotary) as usize / 2;
            let bytes = precompute_rope_table(max_seq, hd, cfg.rope_theta_full, angles);
            ctx.buffer_bytes(&bytes)
        };

        let t = PREFILL_CHUNK;
        let x = ctx.buffer_empty_f16(t * cfg.hidden);
        let x2 = ctx.buffer_empty_f16(t * cfg.hidden);
        let tmp_q = ctx.buffer_empty_f16(t * cfg.n_heads * max_hd);
        let tmp_k = ctx.buffer_empty_f16(t * cfg.n_kv * max_hd);
        let tmp_v = ctx.buffer_empty_f16(t * cfg.n_kv * max_hd);
        let tmp_o = ctx.buffer_empty_f16(t * cfg.n_heads * max_hd);
        let tmp_attn_mwg =
            ctx.buffer_empty_f32(t * cfg.n_heads * Eng::SDPA_MWG_NWG * (max_hd + 2));
        let tmp_ff1 = ctx.buffer_empty_f16(t * max_ff);
        let tmp_ff2 = ctx.buffer_empty_f16(t * max_ff);
        let tmp_ff3 = ctx.buffer_empty_f16(t * max_ff.max(ple_total));
        let logits = ctx.buffer_empty_f16(cfg.vocab);
        let tok_idx = [ctx.buffer_empty_f32(1), ctx.buffer_empty_f32(1)];
        let chunk_tok = ctx.buffer_empty_f32(t);
        let ple_tok = ctx.buffer_empty_f16(t * ple_total);
        let ple_stage = [
            ctx.buffer_empty_f16(ple_total),
            ctx.buffer_empty_f16(ple_total),
        ];
        let ple_ctx = ctx.buffer_empty_f16(t * ple_total);
        let ple_tmp = ctx.buffer_empty_f16(t * ple_total);
        let ple_gate = ctx.buffer_empty_f16(cfg.ple_dim);
        let ple_u = ctx.buffer_empty_f16(t * cfg.ple_dim);
        let ple_proj = ctx.buffer_empty_f16(t * cfg.hidden);

        let mut eng = Eng::new();
        {
            use std::collections::BTreeSet;
            let mut shapes = BTreeSet::new();
            let h = cfg.hidden;
            for layer in 0..cfg.n_layers {
                let hd = cfg.head_dim(layer);
                shapes.insert((cfg.n_heads * hd, h));
                shapes.insert((h, cfg.n_heads * hd));
                shapes.insert((cfg.n_kv * hd, h));
                shapes.insert((cfg.ffn[layer], h));
                shapes.insert((h, cfg.ffn[layer]));
                shapes.insert((cfg.ple_dim, h));
                shapes.insert((h, cfg.ple_dim));
            }
            shapes.insert((cfg.ple_total(), h));
            let shapes: Vec<_> = shapes.into_iter().collect();
            if std::env::var_os("KSEARCH_SKIP_BEAM").is_none() {
                eprintln!(
                    "[beam] warm {} mid-size Q4K matvec plans (chip={}) …",
                    shapes.len(),
                    ctx.device_name()
                );
                eng.warm_matvec_plans(&ctx, DType::Q4K, &shapes)?;
                // lm_head / tied embd
                eng.beam_matvec(
                    &ctx,
                    cfg.vocab,
                    h,
                    token_embd.dtype(),
                    token_embd.buf(),
                )?;
                ctx.synchronize()?;
            } else {
                eprintln!("[beam] skipped (KSEARCH_SKIP_BEAM)");
            }
        }

        eprintln!("[prim] load complete");
        let mut model = Self {
            cfg: cfg.clone(),
            vocab,
            gguf: g,
            mtp: None,
            mtp_hits: 0,
            mtp_tried: 0,
            mtp_pending: None,
            mtp_hseed: None,
            mtp_verify_logits: None,
            mtp_verify_argmax: None,
            ctx,
            token_embd,
            output_norm,
            ple_proj_norm,
            layer_norms,
            layers,
            x,
            x2,
            tmp_q,
            tmp_k,
            tmp_v,
            tmp_o,
            tmp_attn_mwg,
            tmp_ff1,
            tmp_ff2,
            tmp_ff3,
            logits,
            tok_idx,
            chunk_tok,
            ple_tok,
            ple_embd: None,
            ple_stage,
            ple_ctx,
            ple_tmp,
            ple_gate,
            ple_u,
            ple_proj,
            meta,
            rope_swa,
            rope_full,
            hd_swa: cfg.head_dim_swa,
            hd_full: cfg.head_dim_full,
            kv_k,
            kv_v,
            eng,
            weight_bufs: HashMap::new(),
            max_seq,
            pos: 0,
            decode_src: 0,
            return_token_sync: true,
            sample_temperature: 0.0,
            sample_min_p: 0.05,
            sample_seed: 1,
        };
        if model.gguf.has_tensor("per_layer_model_proj.weight") {
            let scale = 1.0 / (model.cfg.hidden as f32).sqrt();
            let mut f = model.gguf.dequant_to_f32("per_layer_model_proj.weight");
            assert!(f.len() % 256 == 0);
            for v in &mut f { *v *= scale; }
            let q4 = quantize_f32_to_q4k(&f);
            eprintln!("[prim] per_layer_model_proj → Q4_K ({:.1} MB)", q4.len() as f64 / 1e6);
            model.weight_bufs.insert(
                "per_layer_model_proj.weight".into(),
                WeightBuf::Q4K(model.ctx.buffer_bytes(&q4)),
            );
        }
        for layer in 0..model.cfg.n_layers {
            let pref = format!("blk.{layer}.");
            for name in [
                "attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight",
                "ffn_gate.weight","ffn_up.weight","ffn_down.weight",
                "inp_gate.weight","proj.weight",
            ] {
                let full = format!("{pref}{name}");
                if model.gguf.has_tensor(&full) {
                    model.ensure_weight(&full)?;
                }
            }
        }
        if model.gguf.has_tensor("per_layer_token_embd.weight") {
            let ty = model.gguf.tensor_type("per_layer_token_embd.weight");
            if ty == ggml_type::Q5_K {
                let raw = model.gguf.tensor_raw("per_layer_token_embd.weight");
                eprintln!(
                    "[prim] per_layer_token_embd Q5_K ({:.1} MB) — Load(Q5K) row gather",
                    raw.len() as f64 / 1e6
                );
                model.ple_embd = Some(WeightBuf::Q5K(model.ctx.buffer_bytes(raw)));
            }
        }
        if let Some(ple) = model.ple_embd.as_ref() {
            let h = model.cfg.hidden;
            let idx = model.tok_idx[0].clone();
            let x = model.x.clone();
            let ple_tok = model.ple_tok.clone();
            let ple_dt = ple.dtype();
            let ple_buf = ple.buf().clone();
            let scale_e = (h as f32).sqrt();
            let scale_p = (model.cfg.ple_dim as f32).sqrt();
            match &model.token_embd {
                WeightBuf::F16(embd) => {
                    let e = embd.clone();
                    model.eng.copy_scale_indexed_wd(
                        &model.ctx, h, scale_e, DType::F16, &e, &idx, &x,
                    )?;
                }
                WeightBuf::Q4K(embd) => {
                    let e = embd.clone();
                    model.eng.copy_scale_indexed_wd(
                        &model.ctx, h, scale_e, DType::Q4K, &e, &idx, &x,
                    )?;
                }
                _ => {}
            }
            model.eng.copy_scale_indexed_wd(
                &model.ctx,
                model.cfg.ple_total(),
                scale_p,
                ple_dt,
                &ple_buf,
                &idx,
                &ple_tok,
            )?;
            model.ctx.synchronize()?;
        }
        Ok(model)
    }

    fn ensure_weight(&mut self, name: &str) -> Result<()> {
        if self.weight_bufs.contains_key(name) {
            return Ok(());
        }
        let ty = self.gguf.tensor_type(name);
        let buf = if ty == ggml_type::Q4_K {
            WeightBuf::Q4K(self.ctx.buffer_bytes(self.gguf.tensor_raw(name)))
        } else if ty == ggml_type::Q6_K
            && (name.ends_with("ffn_down.weight") || name.ends_with("attn_v.weight"))
        {
            // Host-requant Q6→Q4 for faster coop matvec (ffn_down + attn_v).
            let f = self.gguf.dequant_to_f32(name);
            assert!(f.len() % 256 == 0, "{name}: elems {} not multiple of 256", f.len());
            let q4 = quantize_f32_to_q4k(&f);
            eprintln!(
                "[prim] {name} Q6_K → Q4_K host-requant ({:.1} MB)",
                q4.len() as f64 / 1e6
            );
            WeightBuf::Q4K(self.ctx.buffer_bytes(&q4))
        } else if ty == ggml_type::Q6_K {
            eprintln!("[prim] {name} Q6_K native");
            WeightBuf::Q6K(self.ctx.buffer_bytes(self.gguf.tensor_raw(name)))
        } else {
            // PLE F32/BF16 etc.: host-quantize to Q4_K for coop path.
            let f = self.gguf.dequant_to_f32(name);
            assert!(f.len() % 256 == 0, "{name}: elems {} not multiple of 256", f.len());
            let q4 = quantize_f32_to_q4k(&f);
            eprintln!(
                "[prim] {name} {} → Q4_K ({:.1} MB)",
                ksearch_gguf::ggml_type_name(ty),
                q4.len() as f64 / 1e6
            );
            WeightBuf::Q4K(self.ctx.buffer_bytes(&q4))
        };
        self.weight_bufs.insert(name.to_string(), buf);
        Ok(())
    }

    fn matvec_w(
        &mut self,
        rows: usize,
        cols: usize,
        name: &str,
        x: Buffer,
        y: Buffer,
    ) -> Result<()> {
        self.ensure_weight(name)?;
        let w = self.weight_bufs.get(name).unwrap();
        let wd = w.dtype();
        let w = w.buf().clone();
        self.eng
            .matvec_wd(&self.ctx, rows, cols, wd, &w, &x, &y)
    }

    fn matvec_w_batch(
        &mut self,
        rows: usize,
        cols: usize,
        name: &str,
        x: Buffer,
        y: Buffer,
        batch: usize,
    ) -> Result<()> {
        self.ensure_weight(name)?;
        let w = self.weight_bufs.get(name).unwrap();
        let wd = w.dtype();
        let w = w.buf().clone();
        self.eng
            .matvec_batch(&self.ctx, rows, cols, batch, wd, &w, &x, &y)
    }

    fn rmsnorm_matvec_w(
        &mut self,
        rows: usize,
        cols: usize,
        eps: f32,
        w_name: &str,
        x: Buffer,
        w_norm: &Buffer,
        y: Buffer,
    ) -> Result<()> {
        self.ensure_weight(w_name)?;
        let w = self.weight_bufs.get(w_name).unwrap();
        let wd = w.dtype();
        let w = w.buf().clone();
        if is_packed(wd) {
            self.eng.rmsnorm(&self.ctx, cols, eps, &x, w_norm, &self.x2)?;
            return self.eng.matvec_wd(&self.ctx, rows, cols, wd, &w, &self.x2, &y);
        }
        self.eng
            .rmsnorm_matvec_wd(&self.ctx, rows, cols, eps, wd, &x, w_norm, &w, &y)
    }

    fn rmsnorm_matvec_qkv_w(
        &mut self,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        eps: f32,
        q_name: &str,
        k_name: &str,
        v_name: &str,
        x: Buffer,
        w_norm: &Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
    ) -> Result<()> {
        self.ensure_weight(q_name)?;
        self.ensure_weight(k_name)?;
        self.ensure_weight(v_name)?;
        let dq = self.weight_bufs.get(q_name).unwrap().dtype();
        let dk = self.weight_bufs.get(k_name).unwrap().dtype();
        let dv = self.weight_bufs.get(v_name).unwrap().dtype();
        let wq = self.weight_bufs.get(q_name).unwrap().buf().clone();
        let wk = self.weight_bufs.get(k_name).unwrap().buf().clone();
        let wv = self.weight_bufs.get(v_name).unwrap().buf().clone();
        if dq == dk && dk == dv && dq == DType::Q4K {
            // Tiny rms, then one device-x coop QKV (do not fuse rms: LOCAL x_hat
            // regresses vs streaming device activations).
            self.eng
                .rmsnorm(&self.ctx, cols, eps, &x, w_norm, &self.x2)?;
            return self.eng.matvec_qkv_wd(
                &self.ctx,
                q_rows,
                kv_rows,
                cols,
                dq,
                &wq,
                &wk,
                &wv,
                &self.x2,
                &q,
                &k,
                &v,
            );
        }
        if dq == dk && dk == dv && !is_packed(dq) {
            return self.eng.rmsnorm_matvec_qkv_wd(
                &self.ctx,
                q_rows,
                kv_rows,
                cols,
                eps,
                dq,
                &x,
                w_norm,
                &wq,
                &wk,
                &wv,
                &q,
                &k,
                &v,
            );
        }
        // Mixed packings: rms once, then three matvecs.
        self.eng
            .rmsnorm(&self.ctx, cols, eps, &x, w_norm, &self.x2)?;
        self.eng
            .matvec_wd(&self.ctx, q_rows, cols, dq, &wq, &self.x2, &q)?;
        self.eng
            .matvec_wd(&self.ctx, kv_rows, cols, dk, &wk, &self.x2, &k)?;
        self.eng
            .matvec_wd(&self.ctx, kv_rows, cols, dv, &wv, &self.x2, &v)?;
        Ok(())
    }

    fn rmsnorm_matvec_gate_up_gelu_w(
        &mut self,
        rows: usize,
        cols: usize,
        eps: f32,
        gate_name: &str,
        up_name: &str,
        x: Buffer,
        w_norm: &Buffer,
        y: Buffer,
    ) -> Result<()> {
        self.ensure_weight(gate_name)?;
        self.ensure_weight(up_name)?;
        let dg = self.weight_bufs.get(gate_name).unwrap().dtype();
        let du = self.weight_bufs.get(up_name).unwrap().dtype();
        let wg = self.weight_bufs.get(gate_name).unwrap().buf().clone();
        let wu = self.weight_bufs.get(up_name).unwrap().buf().clone();
        if dg == du && is_packed(dg) {
            self.eng
                .rmsnorm(&self.ctx, cols, eps, &x, w_norm, &self.x2)?;
            return self.eng.matvec_gate_up_gelu_wd(
                &self.ctx, rows, cols, dg, &wg, &wu, &self.x2, &y,
            );
        }
        if dg == du {
            return self.eng.rmsnorm_matvec_gate_up_gelu_wd(
                &self.ctx,
                rows,
                cols,
                eps,
                dg,
                &x,
                w_norm,
                &wg,
                &wu,
                &y,
            );
        }
        // Mixed packings: rms once, separate matvecs, then gelu*mul.
        self.eng
            .rmsnorm(&self.ctx, cols, eps, &x, w_norm, &self.x2)?;
        self.eng
            .matvec_wd(&self.ctx, rows, cols, dg, &wg, &self.x2, &self.tmp_ff1)?;
        self.eng
            .matvec_wd(&self.ctx, rows, cols, du, &wu, &self.x2, &self.tmp_ff2)?;
        self.eng
            .gelu_mul(&self.ctx, rows, &self.tmp_ff1, &self.tmp_ff2, &y)?;
        Ok(())
    }

    fn write_meta(&self, layer: usize, tlen: u32, start: u32) {
        self.write_meta_win(layer, tlen, start, 0);
    }

    fn write_meta_win(&self, layer: usize, tlen: u32, start: u32, window: u32) {
        // f32: integers stay exact through max_seq (F16 ULP is 2+ past 2048).
        self.ctx
            .write_f32_at(&self.meta[layer], 0, &[tlen as f32, start as f32, window as f32]);
    }


    fn embed_token(&mut self, token: u32) -> Result<()> {
        let h = self.cfg.hidden;
        let scale = (h as f32).sqrt();
        match &self.token_embd {
            WeightBuf::F16(embd) => {
                self.eng.copy_scale(
                    &self.ctx,
                    h,
                    scale,
                    embd,
                    token as usize * h,
                    &self.x,
                    0,
                )?;
            }
            WeightBuf::Q4K(embd) => {
                let row_bytes = (h / 256) * 144;
                self.eng.copy_scale_wd(
                    &self.ctx,
                    h,
                    scale,
                    DType::Q4K,
                    embd,
                    token as usize * row_bytes,
                    &self.x,
                    0,
                )?;
            }
            WeightBuf::Q5K(_) | WeightBuf::Q6K(_) => bail!("token_embd Q5/Q6 embed not wired"),
        }
        Ok(())
    }

    /// GPU-resident embed: row index from F32 token-id buffer.
    fn embed_from_idx(&mut self, idx: &Buffer) -> Result<()> {
        let h = self.cfg.hidden;
        let scale = (h as f32).sqrt();
        match &self.token_embd {
            WeightBuf::F16(embd) => {
                self.eng.copy_scale_indexed_wd(
                    &self.ctx,
                    h,
                    scale,
                    DType::F16,
                    embd,
                    idx,
                    &self.x,
                )?;
            }
            WeightBuf::Q4K(embd) => {
                self.eng.copy_scale_indexed_wd(
                    &self.ctx,
                    h,
                    scale,
                    DType::Q4K,
                    embd,
                    idx,
                    &self.x,
                )?;
            }
            WeightBuf::Q5K(_) | WeightBuf::Q6K(_) => bail!("token_embd Q5/Q6 embed not wired"),
        }
        Ok(())
    }

    fn embed_from_idx_batch(&mut self, idx: &Buffer, batch: usize) -> Result<()> {
        let h = self.cfg.hidden;
        let scale = (h as f32).sqrt();
        match &self.token_embd {
            WeightBuf::F16(embd) => {
                self.eng.copy_scale_indexed_batch_wd(
                    &self.ctx,
                    h,
                    batch,
                    scale,
                    DType::F16,
                    embd,
                    idx,
                    &self.x,
                )?;
            }
            WeightBuf::Q4K(embd) => {
                self.eng.copy_scale_indexed_batch_wd(
                    &self.ctx,
                    h,
                    batch,
                    scale,
                    DType::Q4K,
                    embd,
                    idx,
                    &self.x,
                )?;
            }
            WeightBuf::Q5K(_) | WeightBuf::Q6K(_) => bail!("token_embd Q5/Q6 embed not wired"),
        }
        Ok(())
    }

    /// GPU-resident PLE row gather from F32 token-id buffer.
    fn ple_from_idx(&mut self, idx: &Buffer) -> Result<()> {
        let Some(ple) = self.ple_embd.as_ref() else {
            bail!("ple_from_idx: packed PLE table not loaded");
        };
        let scale = (self.cfg.ple_dim as f32).sqrt();
        self.eng.copy_scale_indexed_wd(
            &self.ctx,
            self.cfg.ple_total(),
            scale,
            ple.dtype(),
            ple.buf(),
            idx,
            &self.ple_tok,
        )?;
        Ok(())
    }

    fn ple_from_idx_batch(&mut self, idx: &Buffer, batch: usize) -> Result<()> {
        let Some(ple) = self.ple_embd.as_ref() else {
            bail!("ple_from_idx_batch: packed PLE table not loaded");
        };
        let scale = (self.cfg.ple_dim as f32).sqrt();
        self.eng.copy_scale_indexed_batch_wd(
            &self.ctx,
            self.cfg.ple_total(),
            batch,
            scale,
            ple.dtype(),
            ple.buf(),
            idx,
            &self.ple_tok,
        )?;
        Ok(())
    }

    fn ple_prepass_batch(&mut self, batch: usize) -> Result<()> {
        let h = self.cfg.hidden;
        let ple_total = self.cfg.ple_total();
        let n_layers = self.cfg.n_layers;
        let ple_dim = self.cfg.ple_dim;
        let eps = self.cfg.rms_eps;
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

        self.matvec_w_batch(
            ple_total,
            h,
            "per_layer_model_proj.weight",
            self.x.clone(),
            self.ple_tmp.clone(),
            batch,
        )?;
        self.eng.rmsnorm_per_head(
            &self.ctx,
            batch * n_layers,
            ple_dim,
            eps,
            &self.ple_tmp,
            &self.ple_proj_norm,
            &self.ple_ctx,
        )?;
        self.eng.add_scale(
            &self.ctx,
            batch * ple_total,
            inv_sqrt2,
            &self.ple_ctx,
            &self.ple_tok,
            &self.ple_ctx,
        )?;
        Ok(())
    }

    fn load_ple_token(&mut self, token: u32) -> Result<()> {
        let scale = (self.cfg.ple_dim as f32).sqrt();
        let mut row = self
            .gguf
            .dequant_row("per_layer_token_embd.weight", token as usize);
        for v in &mut row {
            *v *= scale;
        }
        let row_h: Vec<u16> = row.iter().map(|&v| f32_to_f16(v)).collect();
        if self.ctx.has_gpu_work() {
            self.ctx.wait_inflight_at_most(1);
            let stage = &self.ple_stage[self.pos & 1];
            self.ctx.write_u16s_nosync(stage, &row_h);
            self.eng.copy_scale(
                &self.ctx,
                self.cfg.ple_total(),
                1.0,
                stage,
                0,
                &self.ple_tok,
                0,
            )?;
        } else {
            self.ctx.write_u16s_nosync(&self.ple_tok, &row_h);
        }
        Ok(())
    }

    fn ple_prepass(&mut self) -> Result<()> {
        let h = self.cfg.hidden;
        let ple_total = self.cfg.ple_total();
        let n_layers = self.cfg.n_layers;
        let ple_dim = self.cfg.ple_dim;
        let eps = self.cfg.rms_eps;
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

        self.matvec_w(
            ple_total,
            h,
            "per_layer_model_proj.weight",
            self.x.clone(),
            self.ple_tmp.clone(),
        )?;
        self.eng.rmsnorm_per_head(
            &self.ctx,
            n_layers,
            ple_dim,
            eps,
            &self.ple_tmp,
            &self.ple_proj_norm,
            &self.ple_ctx,
        )?;
        self.eng.add_scale(
            &self.ctx,
            ple_total,
            inv_sqrt2,
            &self.ple_ctx,
            &self.ple_tok,
            &self.ple_ctx,
        )?;
        Ok(())
    }


    fn forward_token(&mut self, token: u32, want_logits: bool) -> Result<Option<u32>> {
        if self.pos >= self.max_seq {
            bail!("max_seq exceeded");
        }
        let profile = std::env::var_os("KSEARCH_PROFILE").is_some();
        let gphase = std::env::var("KSEARCH_PHASE_GPUTIME")
            .map(|v| v != "0")
            .unwrap_or(false);
        // Ablation profiling: KSEARCH_ABLATE=noattn|nomlp|nople|nohead skips a
        // component end-to-end; Δ vs baseline attributes real token time.
        let abl = std::env::var("KSEARCH_ABLATE").unwrap_or_default();
        let a_attn = abl == "noattn";
        let a_mlp = abl == "nomlp";
        let a_ple = abl == "nople";
        let a_head = abl == "nohead";
        if gphase {
            let _ = self.ctx.init_phase_counters(4096);
            self.ctx.phase_mark();
        }
        let t0 = Instant::now();
        // GPU-resident token-id gather is the ping-pong decode pipe. MTP
        // forces a synced host `token` (the probe embeds `best` + this
        // hidden); reading stale `tok_idx` here produced garbage drafts.
        let gpu_idx = self.ple_embd.is_some() && !(want_logits && self.mtp.is_some());
        if gpu_idx && !want_logits {
            // Ping-pong tok_idx so the next gather can overlap the in-flight token.
            self.ctx.wait_inflight_at_most(1);
            self.decode_src ^= 1;
            self.ctx
                .write_buffer_nosync(&self.tok_idx[self.decode_src], &[token as f32]);
        }
        if gpu_idx {
            let src = self.tok_idx[self.decode_src].clone();
            self.embed_from_idx(&src)?;
        } else {
            self.embed_token(token)?;
        }
        let t_embed = t0.elapsed();
        if gpu_idx {
            let src = self.tok_idx[self.decode_src].clone();
            self.ple_from_idx(&src)?;
        } else {
            self.load_ple_token(token)?;
        }
        let t_ple = t0.elapsed();
        if !a_ple {
            self.ple_prepass()?;
        }
        if gphase {
            self.ctx.phase_mark();
        }
        let t_pre = t0.elapsed();

        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let n_layers = self.cfg.n_layers;
        let n_heads = self.cfg.n_heads;
        let max_seq = self.max_seq;
        let window = self.cfg.sliding_window;
        let ple_dim = self.cfg.ple_dim;
        let mut ms_attn = 0.0f64;
        let mut ms_mlp = 0.0f64;
        let mut ms_ple_l = 0.0f64;

        for layer in 0..n_layers {
            let t_section = if profile {
                self.ctx.synchronize()?;
                Some(Instant::now())
            } else {
                None
            };
            let (q_rows, kv_rows, o_in, ffn_inter, hd) = {
                let lw = &self.layers[layer];
                (lw.q_rows, lw.kv_rows, lw.o_in, lw.ffn_inter, lw.hd)
            };
            let is_swa = self.cfg.is_swa(layer);
            let rope_hd = if is_swa { self.hd_swa } else { self.hd_full };
            let rope_buf = if is_swa {
                self.rope_swa.clone()
            } else {
                self.rope_full.clone()
            };
            let rope_off = self.pos * rope_hd;
            let pref = format!("blk.{layer}.");
            let owns_kv = self.cfg.owns_kv(layer);
            let kv_src = self.cfg.kv_source(layer);

            if !a_attn {
                if owns_kv {
                    let attn_norm = self.layer_norms[layer].attn_norm.clone();
                    self.rmsnorm_matvec_qkv_w(
                        q_rows,
                        kv_rows,
                        h,
                        eps,
                        &format!("{pref}attn_q.weight"),
                        &format!("{pref}attn_k.weight"),
                        &format!("{pref}attn_v.weight"),
                        self.x.clone(),
                        &attn_norm,
                        self.tmp_q.clone(),
                        self.tmp_k.clone(),
                        self.tmp_v.clone(),
                    )?;
                } else {
                    let attn_norm = self.layer_norms[layer].attn_norm.clone();
                    self.rmsnorm_matvec_w(
                        q_rows,
                        h,
                        eps,
                        &format!("{pref}attn_q.weight"),
                        self.x.clone(),
                        &attn_norm,
                        self.tmp_q.clone(),
                    )?;
                }

            if owns_kv {
                let kv_off = self.pos * self.cfg.n_kv * q40_row_bytes(hd);
                self.eng.rmsnorm_per_head_qkv_q40_off(
                    &self.ctx,
                    n_heads,
                    self.cfg.n_kv,
                    hd,
                    eps,
                    &self.tmp_q,
                    &self.layer_norms[layer].q_norm,
                    &rope_buf,
                    rope_off,
                    &self.tmp_k,
                    &self.layer_norms[layer].k_norm,
                    &self.tmp_v,
                    &self.tmp_q,
                    &self.kv_k[kv_src],
                    &self.kv_v[kv_src],
                    kv_off,
                )?;
            } else {
                self.eng.rmsnorm_per_head_rope_off(
                    &self.ctx,
                    n_heads,
                    hd,
                    eps,
                    &self.tmp_q,
                    &self.layer_norms[layer].q_norm,
                    &rope_buf,
                    rope_off,
                    &self.tmp_q,
                    0,
                )?;
            }

            let kv_len = self.pos + 1;
            let (attn_t, attn_start) = if is_swa {
                let start = kv_len.saturating_sub(window);
                ((kv_len - start) as u32, start as u32)
            } else {
                (kv_len as u32, 0u32)
            };
            self.write_meta(layer, attn_t, attn_start);
            self.eng.sdpa_hybrid_kv(
                &self.ctx,
                n_heads,
                self.cfg.n_kv,
                hd,
                max_seq,
                attn_t,
                DType::Q40,
                &self.tmp_q,
                &self.kv_k[kv_src],
                &self.kv_v[kv_src],
                &self.meta[layer],
                &self.tmp_attn_mwg,
                &self.tmp_o,
            )?;

            self.matvec_w(h, o_in, &format!("{pref}attn_output.weight"), self.tmp_o.clone(), self.x2.clone())?;
            } // !a_attn

            let gate_name = format!("{pref}ffn_gate.weight");
            let up_name = format!("{pref}ffn_up.weight");
            self.ensure_weight(&gate_name)?;
            self.ensure_weight(&up_name)?;
            let dg = self.weight_bufs.get(&gate_name).unwrap().dtype();
            let du = self.weight_bufs.get(&up_name).unwrap().dtype();
            let packed_gate = dg == du && is_packed(dg);
            let ffn_norm = self.layer_norms[layer].ffn_norm.clone();

            if !a_mlp {
            if packed_gate {
                // Residual+post-attn rms and ffn rms in one TG; gate_up stays
                // device-x coop (LOCAL x_hat in the fused rms+gate_up path is slower).
                if let Some(ts) = t_section {
                    self.ctx.synchronize()?;
                    ms_attn += ts.elapsed().as_secs_f64() * 1e3;
                }
                let t_mlp = if profile {
                    Some(Instant::now())
                } else {
                    None
                };
                self.eng.rmsnorm_add_then_rmsnorm(
                    &self.ctx,
                    h,
                    eps,
                    &self.x2,
                    &self.layer_norms[layer].post_attn_norm,
                    &self.x,
                    &ffn_norm,
                    &self.x,
                    &self.x2,
                )?;
                let wg = self.weight_bufs.get(&gate_name).unwrap().buf().clone();
                let wu = self.weight_bufs.get(&up_name).unwrap().buf().clone();
                self.eng.matvec_gate_up_gelu_wd(
                    &self.ctx,
                    ffn_inter,
                    h,
                    dg,
                    &wg,
                    &wu,
                    &self.x2,
                    &self.tmp_ff3,
                )?;
                self.matvec_w(
                    h,
                    ffn_inter,
                    &format!("{pref}ffn_down.weight"),
                    self.tmp_ff3.clone(),
                    self.x2.clone(),
                )?;
                self.eng.rmsnorm_add(
                    &self.ctx,
                    h,
                    eps,
                    &self.x2,
                    &self.layer_norms[layer].post_ffw_norm,
                    &self.x,
                    &self.x,
                )?;
                if let Some(ts) = t_mlp {
                    self.ctx.synchronize()?;
                    ms_mlp += ts.elapsed().as_secs_f64() * 1e3;
                }
            } else {
                self.eng.rmsnorm_add(
                    &self.ctx,
                    h,
                    eps,
                    &self.x2,
                    &self.layer_norms[layer].post_attn_norm,
                    &self.x,
                    &self.x,
                )?;
                if let Some(ts) = t_section {
                    self.ctx.synchronize()?;
                    ms_attn += ts.elapsed().as_secs_f64() * 1e3;
                }
                let t_mlp = if profile {
                    Some(Instant::now())
                } else {
                    None
                };

                // MLP: fused rms → gate/up matvec+gelu → down
                self.rmsnorm_matvec_gate_up_gelu_w(
                    ffn_inter,
                    h,
                    eps,
                    &gate_name,
                    &up_name,
                    self.x.clone(),
                    &ffn_norm,
                    self.tmp_ff3.clone(),
                )?;
                self.matvec_w(
                    h,
                    ffn_inter,
                    &format!("{pref}ffn_down.weight"),
                    self.tmp_ff3.clone(),
                    self.x2.clone(),
                )?;
                self.eng.rmsnorm_add(
                    &self.ctx,
                    h,
                    eps,
                    &self.x2,
                    &self.layer_norms[layer].post_ffw_norm,
                    &self.x,
                    &self.x,
                )?;
                if let Some(ts) = t_mlp {
                    self.ctx.synchronize()?;
                    ms_mlp += ts.elapsed().as_secs_f64() * 1e3;
                }
            }
            } // !a_mlp
            let t_ple_l = if profile {
                Some(Instant::now())
            } else {
                None
            };

            // PLE: fuse gate matvec + gelu*ctx, then proj matvec + rmsnorm_add_scale (sequenced).
            if !a_ple {
            let gate_name = format!("{pref}inp_gate.weight");
            self.ensure_weight(&gate_name)?;
            let dg = self.weight_bufs.get(&gate_name).unwrap().dtype();
            let wg = self.weight_bufs.get(&gate_name).unwrap().buf().clone();
            self.eng.matvec_gelu_mul_at(
                &self.ctx,
                ple_dim,
                h,
                dg,
                &wg,
                &self.x,
                &self.ple_ctx,
                layer * ple_dim,
                &self.ple_u,
            )?;
            let proj_name = format!("{pref}proj.weight");
            self.ensure_weight(&proj_name)?;
            let dp = self.weight_bufs.get(&proj_name).unwrap().dtype();
            let wp = self.weight_bufs.get(&proj_name).unwrap().buf().clone();
            self.eng.matvec_rmsnorm_add_scale(
                &self.ctx,
                h,
                ple_dim,
                eps,
                self.layer_norms[layer].layer_scale,
                dp,
                &wp,
                &self.ple_u,
                &self.layer_norms[layer].post_norm,
                &self.x,
                &self.x,
            )?;
            } // !a_ple
            if let Some(ts) = t_ple_l {
                self.ctx.synchronize()?;
                ms_ple_l += ts.elapsed().as_secs_f64() * 1e3;
            }
            // Decode: keep one encoder/CB for the whole token (llama.cpp-style).
            // Prefill still commits at token end so the next PLE gather can overlap.
        }

        let t_layers = t0.elapsed();
        if gphase {
            self.ctx.phase_mark();
        }
        self.pos += 1;
        if !want_logits {
            // Keep GPU pipeline full — host only syncs when reading argmax / generate end.
            if profile {
                self.ctx.synchronize()?;
                eprintln!(
                    "[profile] prefill pos={} embed={:.1}ms ple_load={:.1}ms prepass={:.1}ms layers={:.1}ms total={:.1}ms",
                    self.pos - 1,
                    t_embed.as_secs_f64() * 1e3,
                    (t_ple - t_embed).as_secs_f64() * 1e3,
                    (t_pre - t_ple).as_secs_f64() * 1e3,
                    (t_layers - t_pre).as_secs_f64() * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            } else {
                self.ctx.flush_async();
            }
            return Ok(None);
        }

        // Thesis-A fused output head: RmsNormMatvec(output_norm, token_embd)
        // MTP probe note: while probing we fall through to the synced branch
        // below regardless of return_token_sync.
        // then parallel softcap_argmax — Q4_K embd uses Load(Q4K) expand.
        let vocab = self.cfg.vocab;
        let cap = self.cfg.softcap;
        let embd_dt = self.token_embd.dtype();
        let embd = self.token_embd.buf().clone();
        let dst = self.tok_idx[self.decode_src ^ 1].clone();
        let sampling = self.sample_temperature >= 1e-6;
        if !a_head {
        // MTP needs the post-final-norm hidden (`x2`) as h_nextn. The fused
        // F16 rmsnorm+lm_head path never materializes it.
        if is_packed(embd_dt) || self.mtp.is_some() {
            self.eng
                .rmsnorm(&self.ctx, h, eps, &self.x, &self.output_norm, &self.x2)?;
            self.eng.matvec_wd(
                &self.ctx, vocab, h, embd_dt, &embd, &self.x2, &self.logits,
            )?;
        } else {
            self.eng.rmsnorm_matvec_wd(
                &self.ctx, vocab, h, eps, embd_dt, &self.x, &self.output_norm, &embd, &self.logits,
            )?;
        }
        if !sampling {
            self.eng
                .softcap_argmax(&self.ctx, vocab, cap, &self.logits, &dst)?;
        }
        } // !a_head
        let t_logits = t0.elapsed();
        if sampling {
            let logits = self.ctx.read_u16(&self.logits, vocab);
            let best = sample_softcap_min_p(
                &logits,
                cap,
                self.sample_temperature,
                self.sample_min_p,
                self.sample_seed,
            );
            self.sample_temperature = 0.0;
            return Ok(Some(best));
        }
        let mtp_probe = self.mtp.is_some();
        if !self.return_token_sync && !mtp_probe {
            // Keep the GPU pipeline full; host reads the argmax later.
            if profile || gphase {
                if gphase {
                    self.ctx.phase_mark();
                    self.ctx.synchronize()?;
                } else {
                    // One sync so the printed segments reflect real GPU time.
                    let _ = self.ctx.read_f32(&dst, 1)[0] as u32;
                }
                let (dispatches, enqueue_s) = ksearch_compute::take_telemetry();
                if gphase {
                    let ph = self.ctx.resolve_phase_times_ms();
                    let gpu_busy: f64 = ph.iter().sum();
                    let wall = t0.elapsed().as_secs_f64() * 1e3;
                    let phase_str = ph
                        .iter()
                        .map(|v| format!("{v:.2}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    eprintln!(
                        "[gputime] pos={} phases=[{phase_str}]ms gpu_busy={wall:.2}->{gpu_busy:.2}ms dispatches={dispatches} cpu_enq={:.2}ms",
                        self.pos - 1,
                        enqueue_s * 1e3,
                    );
                }
                if profile {
                    eprintln!(
                        "[profile] decode pos={} embed={:.1}ms ple_load={:.1}ms prepass={:.1}ms attn={:.1}ms mlp={:.1}ms ple={:.1}ms logits+sync={:.1}ms total={:.1}ms",
                        self.pos - 1,
                        t_embed.as_secs_f64() * 1e3,
                        (t_ple - t_embed).as_secs_f64() * 1e3,
                        (t_pre - t_ple).as_secs_f64() * 1e3,
                        ms_attn,
                        ms_mlp,
                        ms_ple_l,
                        (t0.elapsed() - t_logits).as_secs_f64() * 1e3
                            + (t_logits - t_layers).as_secs_f64() * 1e3,
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            return Ok(None);
        }
        let best = self.ctx.read_f32(&dst, 1)[0] as u32;

        // ---- MTP acceptance probe ------------------------------------
        // Truth check: did the previous draft step predict `best`?
        // Probe-only: KSEARCH_MTP_PROBE=1 drafts one token after each decode
        // (acceptance measurement without spec-dec). Default --mtp uses
        // generate_timed speculative verify instead.
        if mtp_probe && std::env::var_os("KSEARCH_MTP_PROBE").is_some() {
            if let Some(pred) = self.mtp_pending.take() {
                self.mtp_tried += 1;
                if pred == best as u32 {
                    self.mtp_hits += 1;
                }
                if self.mtp_tried % 16 == 0 {
                    eprintln!(
                        "[mtp] tried={} pred={pred} actual={best}",
                        self.mtp_tried + 1,
                    );
                }
                if self.mtp_tried % 64 == 0 {
                    eprintln!(
                        "[mtp-accept] {}/{} = {:.1}%",
                        self.mtp_hits,
                        self.mtp_tried,
                        100.0 * self.mtp_hits as f64 / self.mtp_tried.max(1) as f64
                    );
                }
            }
            // Contract (llama.cpp gemma4-assistant / metal-llm-server draft_chain):
            //   xh = concat(embed(best) * √hb, h_post_norm)
            // `best` is the token just sampled; `x2` is this step's post-final-norm
            // hidden. RoPE pos = KV length = index of `best` (not yet written).
            let h_seed: Vec<f32> = self
                .ctx
                .read_u16(&self.x2, h)
                .iter()
                .map(|&v| ksearch_gguf::f16_to_f32(v))
                .collect();
            let scale = (h as f32).sqrt();
            let mut emb_row = self.mtp_embedding_row(best)?;
            for v in emb_row.iter_mut() {
                *v *= scale;
            }
            let ks = self.kv_k.clone();
            let vs = self.kv_v.clone();
            let seq_len = self.pos;
            let capacity = self.max_seq;
            let (pred, _hnext) = {
                let draft = self.mtp.as_mut().expect("mtp");
                draft.forward_step(
                    &self.ctx,
                    &emb_row,
                    &h_seed,
                    seq_len,
                    seq_len,
                    capacity,
                    &ks,
                    &vs,
                )?
            };
            if std::env::var("MTP_DEBUG").is_ok() {
                let lg: Vec<f32> = self
                    .ctx
                    .read_u16(self.mtp.as_ref().unwrap().debug_logits(), self.cfg.vocab)
                    .iter()
                    .map(|&v| ksearch_gguf::f16_to_f32(v))
                    .collect();
                let mut idx: Vec<(usize, f32)> = lg.iter().cloned().enumerate().collect();
                idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                eprintln!(
                    "[mtp-top] actual={best} draft={pred} pred_top3={:?}",
                    &idx[..3]
                        .iter()
                        .map(|(i, v)| format!("{i}:{v:.1}"))
                        .collect::<Vec<_>>()
                );
            }
            self.mtp_pending = Some(pred);
        }
        // ---------------------------------------------------------------

        if profile {
            eprintln!(
                "[profile] decode pos={} embed={:.1}ms ple_load={:.1}ms prepass={:.1}ms attn={:.1}ms mlp={:.1}ms ple={:.1}ms logits+sync={:.1}ms total={:.1}ms",
                self.pos - 1,
                t_embed.as_secs_f64() * 1e3,
                (t_ple - t_embed).as_secs_f64() * 1e3,
                (t_pre - t_ple).as_secs_f64() * 1e3,
                ms_attn,
                ms_mlp,
                ms_ple_l,
                (t0.elapsed() - t_logits).as_secs_f64() * 1e3
                    + (t_logits - t_layers).as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(Some(best))
    }

    /// Load the MTP draft head and enable the acceptance-rate probe.
    pub fn load_mtp(&mut self, path: &Path) -> Result<()> {
        // Last non-shared layer of each attention type (metal-llm-server
        // `mtp_kv_source_layer`). Shared tail layers have empty KV slots.
        let n_own = self
            .cfg
            .n_layers
            .saturating_sub(self.cfg.shared_kv_layers)
            .max(1);
        let kv_map_swa = (0..n_own)
            .rev()
            .find(|&i| self.cfg.is_swa(i))
            .unwrap_or(0);
        let kv_map_full = (0..n_own)
            .rev()
            .find(|&i| !self.cfg.is_swa(i))
            .unwrap_or(kv_map_swa);
        eprintln!(
            "[mtp] kv maps: swa→{kv_map_swa} full→{kv_map_full} (last non-shared owners)"
        );
        let draft = MtpDraft::load(
            Eng::new(),
            &self.ctx,
            path,
            &self.cfg,
            self.max_seq,
            kv_map_swa,
            kv_map_full,
        )?;
        eprintln!(
            "[mtp] draft loaded: vocab={} hb={} (kv maps: swa→{} full→{})",
            draft.vocab,
            draft.hb,
            kv_map_swa,
            kv_map_full
        );
        let h = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        self.mtp_hseed = Some(self.ctx.buffer_empty_f16(h));
        self.mtp_verify_logits = Some(self.ctx.buffer_empty_f16(MAX_MTP_VERIFY * vocab));
        self.mtp_verify_argmax = Some(self.ctx.buffer_empty_f32(MAX_MTP_VERIFY));
        self.mtp = Some(draft);
        // Compile verify (mul_mm batch) + one draft step before generate timing.
        let n = mtp_draft_steps() + 1;
        let dummy = vec![0u32; n];
        let _ = self.forward_verify_batch(&dummy)?;
        {
            let ks = self.kv_k.clone();
            let vs = self.kv_v.clone();
            let cap = self.max_seq;
            let xh = self.mtp.as_ref().unwrap().xh_buf().clone();
            let z = vec![0u16; h * 2];
            self.ctx.write_u16s(&xh, &z);
            let draft = self.mtp.as_mut().unwrap();
            draft.forward_from_xh(&self.ctx, 0, n.max(1), cap, &ks, &vs)?;
            self.ctx.synchronize()?;
        }
        self.reset();
        self.mtp_hits = 0;
        self.mtp_tried = 0;
        eprintln!(
            "[mtp] spec-dec ready: draft_steps={} verify_batch={}",
            mtp_draft_steps(),
            mtp_draft_steps() + 1
        );
        Ok(())
    }

    pub fn mtp_stats(&self) -> (usize, usize) {
        (self.mtp_hits, self.mtp_tried)
    }

    fn mtp_embedding_row(&mut self, token: u32) -> Result<Vec<f32>> {
        // Same GPU dequant path as `embed_token` (Q4_K Load expand).
        let h = self.cfg.hidden;
        let dt = self.token_embd.dtype();
        let buf = self.token_embd.buf().clone();
        match dt {
            DType::Q4K => {
                let row_bytes = (h / 256) * 144;
                self.eng.copy_scale_wd(
                    &self.ctx,
                    h,
                    1.0,
                    DType::Q4K,
                    &buf,
                    token as usize * row_bytes,
                    &self.tmp_ff3,
                    0,
                )?;
            }
            DType::F16 => {
                self.eng.copy_scale(
                    &self.ctx,
                    h,
                    1.0,
                    &buf,
                    token as usize * h,
                    &self.tmp_ff3,
                    0,
                )?;
            }
            _ => {
                let row_bytes = match dt {
                    DType::Q5K => h / 256 * 176,
                    DType::Q6K => h / 256 * 210,
                    _ => h * 2,
                };
                return ctx_read_row(
                    self,
                    &buf,
                    token as usize * row_bytes,
                    row_bytes,
                    h,
                    dt,
                );
            }
        }
        self.ctx.synchronize()?;
        Ok(self
            .ctx
            .read_u16(&self.tmp_ff3, h)
            .iter()
            .map(|&v| ksearch_gguf::f16_to_f32(v))
            .collect())
    }

    /// Bind this sequence's K/V packs and `pos` to a pool slot (serving).
    pub fn bind_slot(&mut self, pool: &KvPool, slot: SlotId) -> Result<()> {
        if self.kv_k.len() != pool.n_kv_layers {
            bail!(
                "kv layer mismatch: model={} pool={}",
                self.kv_k.len(),
                pool.n_kv_layers
            );
        }
        for i in 0..self.kv_k.len() {
            self.kv_k[i] = pool.k_buf(slot, i)?.clone();
            self.kv_v[i] = pool.v_buf(slot, i)?.clone();
        }
        self.pos = pool.seq_len(slot)?;
        Ok(())
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Prefill one token (KV write, no logits).
    pub fn prefill_token(&mut self, token: u32) -> Result<()> {
        let _ = self.forward_token(token, false)?;
        Ok(())
    }

    /// Seq-parallel prefill for `tokens` (any length). Remainder of 1 uses the decode kernel.
    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            if chunk.len() == 1 {
                let _ = self.forward_token(chunk[0], false)?;
            } else {
                self.forward_seq_tokens(chunk, false)?;
            }
        }
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Batched target forward for MTP verify: writes KV, returns greedy id after each row.
    fn forward_verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if tokens.len() > MAX_MTP_VERIFY {
            bail!(
                "mtp verify batch {} > MAX_MTP_VERIFY {}",
                tokens.len(),
                MAX_MTP_VERIFY
            );
        }
        self.forward_seq_tokens(tokens, true)
    }

    /// Verify using tokens already packed in `chunk_tok` (GPU draft handoff).
    fn forward_verify_from_chunk(&mut self, t: usize) -> Result<Vec<u32>> {
        if t == 0 {
            return Ok(Vec::new());
        }
        if t > MAX_MTP_VERIFY {
            bail!("mtp verify batch {t} > MAX_MTP_VERIFY {MAX_MTP_VERIFY}");
        }
        // Dummy slice — `forward_seq_tokens` rewrites chunk_tok; use a path that
        // skips the rewrite when tokens is empty sentinel... instead inline skip:
        self.forward_seq_tokens_prepacked(t, true)
    }

    fn forward_seq_tokens(&mut self, tokens: &[u32], want_head: bool) -> Result<Vec<u32>> {
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if self.pos + t > self.max_seq {
            bail!("max_seq exceeded");
        }
        let ids: Vec<f32> = tokens.iter().map(|&tok| tok as f32).collect();
        self.ctx.write_buffer_nosync(&self.chunk_tok, &ids);
        self.forward_seq_tokens_prepacked(t, want_head)
    }

    fn forward_seq_tokens_prepacked(&mut self, t: usize, want_head: bool) -> Result<Vec<u32>> {
        if t == 0 {
            return Ok(Vec::new());
        }
        if self.pos + t > self.max_seq {
            bail!("max_seq exceeded");
        }
        let vprof = want_head && std::env::var_os("KSEARCH_VERIFY_PROFILE").is_some();
        let mut vp_embed = 0.0f64;
        let mut vp_attn = 0.0f64;
        let mut vp_mlp = 0.0f64;
        let mut vp_ple = 0.0f64;
        let mut vp_head = 0.0f64;
        let vp_t0 = Instant::now();

        let gpu_idx = self.ple_embd.is_some();
        if gpu_idx {
            let src = self.chunk_tok.clone();
            self.embed_from_idx_batch(&src, t)?;
            self.ple_from_idx_batch(&src, t)?;
        } else {
            // CPU gather fallback: read packed ids.
            let ids = self.ctx.read_f32(&self.chunk_tok, t);
            for (i, &tok_f) in ids.iter().enumerate() {
                let tok = tok_f as u32;
                self.embed_token(tok)?;
                self.eng.copy_slice(
                    &self.ctx,
                    self.cfg.hidden,
                    &self.x,
                    0,
                    &self.x,
                    i * self.cfg.hidden,
                )?;
                self.load_ple_token(tok)?;
                self.eng.copy_slice(
                    &self.ctx,
                    self.cfg.ple_total(),
                    &self.ple_tok,
                    0,
                    &self.ple_tok,
                    i * self.cfg.ple_total(),
                )?;
            }
        }
        self.ple_prepass_batch(t)?;
        if vprof {
            self.ctx.synchronize()?;
            vp_embed = vp_t0.elapsed().as_secs_f64() * 1e3;
        }

        // First large prefill pays for nvcc. Profile the next one, which is
        // steady-state GPU time, and leave later repeats unsynchronized.
        let pprof = std::env::var_os("KSEARCH_PREFILL_PROF").is_some()
            && t >= 32
            && PREFILL_PROF_SEEN.fetch_add(1, Ordering::Relaxed) == 1;
        let mut ms_qkv = 0.0f64;
        let mut ms_attn = 0.0f64;
        let mut ms_o = 0.0f64;
        let mut ms_ffn = 0.0f64;
        let mut ms_ple = 0.0f64;
        let mut tick = Instant::now();

        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let n_layers = self.cfg.n_layers;
        let n_heads = self.cfg.n_heads;
        let max_seq = self.max_seq;
        let window = self.cfg.sliding_window;
        let ple_dim = self.cfg.ple_dim;
        let pos0 = self.pos;

        // Coarse phase syncs only (not per-layer): otherwise profile cost
        // dominates and numbers are useless for optimization.
        let mut vp_attn_t = if vprof {
            Some(Instant::now())
        } else {
            None
        };
        if pprof {
            self.ctx.synchronize()?;
            tick = Instant::now();
        }
        for layer in 0..n_layers {
            let (q_rows, kv_rows, o_in, ffn_inter, hd) = {
                let lw = &self.layers[layer];
                (lw.q_rows, lw.kv_rows, lw.o_in, lw.ffn_inter, lw.hd)
            };
            let is_swa = self.cfg.is_swa(layer);
            let rope_hd = if is_swa { self.hd_swa } else { self.hd_full };
            let rope_buf = if is_swa {
                self.rope_swa.clone()
            } else {
                self.rope_full.clone()
            };
            let pref = format!("blk.{layer}.");
            let owns_kv = self.cfg.owns_kv(layer);
            let kv_src = self.cfg.kv_source(layer);

            let attn_norm = self.layer_norms[layer].attn_norm.clone();
            self.eng
                .rmsnorm_rows(&self.ctx, h, t, eps, &self.x, &attn_norm, &self.x2)?;
            if owns_kv {
                self.ensure_weight(&format!("{pref}attn_q.weight"))?;
                self.ensure_weight(&format!("{pref}attn_k.weight"))?;
                self.ensure_weight(&format!("{pref}attn_v.weight"))?;
                self.matvec_w_batch(
                    q_rows,
                    h,
                    &format!("{pref}attn_q.weight"),
                    self.x2.clone(),
                    self.tmp_q.clone(),
                    t,
                )?;
                self.matvec_w_batch(
                    kv_rows,
                    h,
                    &format!("{pref}attn_k.weight"),
                    self.x2.clone(),
                    self.tmp_k.clone(),
                    t,
                )?;
                self.matvec_w_batch(
                    kv_rows,
                    h,
                    &format!("{pref}attn_v.weight"),
                    self.x2.clone(),
                    self.tmp_v.clone(),
                    t,
                )?;
            } else {
                self.matvec_w_batch(
                    q_rows,
                    h,
                    &format!("{pref}attn_q.weight"),
                    self.x2.clone(),
                    self.tmp_q.clone(),
                    t,
                )?;
            }
            if pprof {
                self.ctx.synchronize()?;
                ms_qkv += tick.elapsed().as_secs_f64() * 1e3;
                tick = Instant::now();
            }

            if owns_kv {
                let rope_off = pos0 * rope_hd;
                let kv_off = pos0 * self.cfg.n_kv * q40_row_bytes(hd);
                self.eng.rmsnorm_per_head_qkv_q40_batch(
                    &self.ctx,
                    n_heads,
                    self.cfg.n_kv,
                    hd,
                    t,
                    eps,
                    &self.tmp_q,
                    &self.layer_norms[layer].q_norm,
                    &rope_buf,
                    rope_off,
                    &self.tmp_k,
                    &self.layer_norms[layer].k_norm,
                    &self.tmp_v,
                    &self.tmp_q,
                    &self.kv_k[kv_src],
                    &self.kv_v[kv_src],
                    kv_off,
                )?;
            } else {
                let rope_off = pos0 * rope_hd;
                self.eng.rmsnorm_per_head_rope_batch(
                    &self.ctx,
                    n_heads,
                    hd,
                    t,
                    eps,
                    &self.tmp_q,
                    &self.layer_norms[layer].q_norm,
                    &rope_buf,
                    rope_off,
                    &self.tmp_q,
                )?;
            }

            let last_tlen = pos0 + t;
            let (attn_t, win) = if is_swa {
                (
                    last_tlen.min(window) as u32,
                    window as u32,
                )
            } else {
                (last_tlen as u32, max_seq as u32)
            };
            let tlen0 = (pos0 + 1) as u32;
            self.write_meta_win(layer, tlen0, 0, win);
            self.eng.sdpa_hybrid_kv_batch(
                &self.ctx,
                n_heads,
                self.cfg.n_kv,
                t,
                hd,
                max_seq,
                attn_t,
                DType::Q40,
                &self.tmp_q,
                &self.kv_k[kv_src],
                &self.kv_v[kv_src],
                &self.meta[layer],
                &self.tmp_attn_mwg,
                &self.tmp_o,
            )?;
            if pprof {
                self.ctx.synchronize()?;
                ms_attn += tick.elapsed().as_secs_f64() * 1e3;
                tick = Instant::now();
            }

            self.matvec_w_batch(
                h,
                o_in,
                &format!("{pref}attn_output.weight"),
                self.tmp_o.clone(),
                self.x2.clone(),
                t,
            )?;

            if layer == 0 && vp_attn_t.is_some() {
                self.ctx.synchronize()?;
                if let Some(ts) = vp_attn_t.take() {
                    // Estimate full-stack attn as first-layer sample × n_layers.
                    vp_attn = ts.elapsed().as_secs_f64() * 1e3 * n_layers as f64;
                }
                // Start MLP timing after first-layer attn sample.
            }
            if pprof {
                self.ctx.synchronize()?;
                ms_o += tick.elapsed().as_secs_f64() * 1e3;
                tick = Instant::now();
            }

            let gate_name = format!("{pref}ffn_gate.weight");
            let up_name = format!("{pref}ffn_up.weight");
            self.ensure_weight(&gate_name)?;
            self.ensure_weight(&up_name)?;
            let ffn_norm = self.layer_norms[layer].ffn_norm.clone();
            self.eng.rmsnorm_add_then_rmsnorm_rows(
                &self.ctx,
                h,
                t,
                eps,
                &self.x2,
                &self.layer_norms[layer].post_attn_norm,
                &self.x,
                &ffn_norm,
                &self.x,
                &self.x2,
            )?;
            // Fused gate∥up+GeLU ext matches decode float gelu (correct vs target).
            // Separate F16-rounds gate/up before gelu — slightly higher draft accept
            // but wrong vs decode. Default fused; opt out: KSEARCH_MTP_FUSE_GATE=0.
            let fuse_gate = std::env::var("KSEARCH_MTP_FUSE_GATE").ok().as_deref() != Some("0");
            let dg = self.weight_bufs.get(&gate_name).unwrap().dtype();
            let du = self.weight_bufs.get(&up_name).unwrap().dtype();
            if fuse_gate && dg == du && is_packed(dg) && (2..=8).contains(&t) {
                let wg = self.weight_bufs.get(&gate_name).unwrap().buf().clone();
                let wu = self.weight_bufs.get(&up_name).unwrap().buf().clone();
                self.eng.matvec_gate_up_gelu_batch_wd(
                    &self.ctx,
                    ffn_inter,
                    h,
                    t,
                    dg,
                    &wg,
                    &wu,
                    &self.x2,
                    &self.tmp_ff3,
                )?;
            } else {
                self.matvec_w_batch(
                    ffn_inter,
                    h,
                    &gate_name,
                    self.x2.clone(),
                    self.tmp_ff1.clone(),
                    t,
                )?;
                self.matvec_w_batch(
                    ffn_inter,
                    h,
                    &up_name,
                    self.x2.clone(),
                    self.tmp_ff2.clone(),
                    t,
                )?;
                self.eng.gelu_mul(
                    &self.ctx,
                    t * ffn_inter,
                    &self.tmp_ff1,
                    &self.tmp_ff2,
                    &self.tmp_ff3,
                )?;
            }
            self.matvec_w_batch(
                h,
                ffn_inter,
                &format!("{pref}ffn_down.weight"),
                self.tmp_ff3.clone(),
                self.x2.clone(),
                t,
            )?;
            self.eng.rmsnorm_add_rows(
                &self.ctx,
                h,
                t,
                eps,
                &self.x2,
                &self.layer_norms[layer].post_ffw_norm,
                &self.x,
                &self.x,
            )?;
            if pprof {
                self.ctx.synchronize()?;
                ms_ffn += tick.elapsed().as_secs_f64() * 1e3;
                tick = Instant::now();
            }

            let gate_name = format!("{pref}inp_gate.weight");
            self.ensure_weight(&gate_name)?;
            let proj_name = format!("{pref}proj.weight");
            self.ensure_weight(&proj_name)?;
            let layer_scale = self.layer_norms[layer].layer_scale;
            self.matvec_w_batch(
                ple_dim,
                h,
                &gate_name,
                self.x.clone(),
                self.ple_u.clone(),
                t,
            )?;
            self.eng.gelu_mul_strided(
                &self.ctx,
                ple_dim,
                t,
                &self.ple_u,
                &self.ple_ctx,
                layer * ple_dim,
                self.cfg.ple_total(),
                &self.ple_u,
            )?;
            self.matvec_w_batch(
                h,
                ple_dim,
                &proj_name,
                self.ple_u.clone(),
                self.ple_proj.clone(),
                t,
            )?;
            self.eng.rmsnorm_add_scale_rows(
                &self.ctx,
                h,
                t,
                eps,
                layer_scale,
                &self.ple_proj,
                &self.layer_norms[layer].post_norm,
                &self.x,
                &self.x,
            )?;
            if pprof {
                self.ctx.synchronize()?;
                ms_ple += tick.elapsed().as_secs_f64() * 1e3;
                tick = Instant::now();
            }
        }
        if pprof {
            eprintln!(
                "[prefill-prof] t={t} qkv={ms_qkv:.1} attn={ms_attn:.1} o={ms_o:.1} ffn={ms_ffn:.1} ple={ms_ple:.1} ms"
            );
        }
        if vprof {
            self.ctx.synchronize()?;
            // Rough: remaining time after attn estimate is mlp+ple+glue.
            let after_layers = vp_t0.elapsed().as_secs_f64() * 1e3;
            vp_mlp = (after_layers - vp_embed - vp_attn).max(0.0);
            vp_ple = 0.0; // folded into mlp bucket above for coarse profile
        }

        self.pos += t;
        if want_head {
            let vp_head_t = if vprof {
                Some(Instant::now())
            } else {
                None
            };
            let vocab = self.cfg.vocab;
            let cap = self.cfg.softcap;
            let embd_dt = self.token_embd.dtype();
            let embd = self.token_embd.buf().clone();
            let logits = self
                .mtp_verify_logits
                .as_ref()
                .ok_or_else(|| anyhow!("mtp verify logits missing"))?
                .clone();
            let argmax = self
                .mtp_verify_argmax
                .as_ref()
                .ok_or_else(|| anyhow!("mtp verify argmax missing"))?
                .clone();
            self.eng
                .rmsnorm_rows(&self.ctx, h, t, eps, &self.x, &self.output_norm, &self.x2)?;
            self.eng.matvec_batch(
                &self.ctx,
                vocab,
                h,
                t,
                embd_dt,
                &embd,
                &self.x2,
                &logits,
            )?;
            self.eng
                .softcap_argmax_rows(&self.ctx, vocab, t, cap, &logits, &argmax)?;
            self.ctx.synchronize()?;
            if let Some(ts) = vp_head_t {
                vp_head = ts.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[verify-profile] t={t} embed={vp_embed:.1} attn={vp_attn:.1} mlp={vp_mlp:.1} ple={vp_ple:.1} head={vp_head:.1} total={:.1}ms",
                    vp_t0.elapsed().as_secs_f64() * 1e3
                );
            }
            let ids = self.ctx.read_f32(&argmax, t);
            return Ok(ids.iter().map(|&v| v as u32).collect());
        }
        // Must wait: next chunk reuses x/meta/chunk_tok. flush_async raced the
        // CPU overwrite with in-flight GPU and corrupted early-prompt KV.
        self.ctx.synchronize()?;
        Ok(Vec::new())
    }

    /// Decode one token: embed/PLE from `token`, return next greedy id.
    pub fn decode_token(&mut self, token: u32) -> Result<u32> {
        self.decode_token_sampled(token, 0.0, 0.0, 1)
    }

    /// Decode one token with oracle-style temperature + min-p (CPU, after softcap).
    /// `temperature < 1e-6` is greedy GPU argmax.
    pub fn decode_token_sampled(
        &mut self,
        token: u32,
        temperature: f32,
        min_p: f32,
        seed: u32,
    ) -> Result<u32> {
        self.decode_src = 0;
        self.return_token_sync = true;
        self.sample_temperature = temperature;
        self.sample_min_p = min_p;
        self.sample_seed = seed;
        if self.ple_embd.is_some() {
            self.ctx.synchronize()?;
            self.ctx
                .write_buffer_nosync(&self.tok_idx[0], &[token as f32]);
        }
        self.forward_token(token, true)?
            .ok_or_else(|| anyhow!("decode expected logits"))
    }

    /// F32 KV pool for multi-stream decode (P5 serving contract).
    /// Zero a slot's KV packs (see [`KvPool::clear_slot`]); the pool has no
    /// context of its own, so the model lends its own.
    pub fn clear_kv_slot(&self, pool: &KvPool, slot: SlotId) -> Result<()> {
        pool.clear_slot(&self.ctx, slot)
    }

    pub fn make_kv_pool(&self, max_batch: usize) -> Result<KvPool> {
        let n_kv_layers = self.cfg.n_kv_owners();
        let hd = self.cfg.head_dim_full.max(self.cfg.head_dim_swa);
        KvPool::new(&self.ctx, max_batch, self.max_seq, n_kv_layers, hd, self.cfg.n_kv)
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.decode_src = 0;
        self.return_token_sync = true;
        self.sample_temperature = 0.0;
        self.ctx.synchronize().ok();
        for b in self.kv_k.iter().chain(self.kv_v.iter()) {
            self.ctx.zero(b);
        }
    }

    fn mtp_copy_hseed(&mut self, row: usize) -> Result<()> {
        let h = self.cfg.hidden;
        let dst = self
            .mtp_hseed
            .as_ref()
            .ok_or_else(|| anyhow!("mtp hseed missing"))?
            .clone();
        self.eng.copy_scale_wd(
            &self.ctx,
            h,
            1.0,
            DType::F16,
            &self.x2,
            row * h * DType::F16.size_bytes(),
            &dst,
            0,
        )?;
        self.ctx.encoder_barrier();
        Ok(())
    }

    /// Draft `n` tokens from `id_last` + `mtp_hseed`. All steps share the same
    /// RoPE pos / KV (metal-llm-server `draft_chain`). Leaves `draft_ids` on GPU
    /// — caller syncs once after verify (no mid-cycle CPU round-trip).
    fn mtp_draft_chain(&mut self, id_last: u32, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let h = self.cfg.hidden;
        let scale = (h as f32).sqrt();
        let dt = self.token_embd.dtype();
        let embd = self.token_embd.buf().clone();
        let hseed = self
            .mtp_hseed
            .as_ref()
            .ok_or_else(|| anyhow!("mtp hseed missing"))?
            .clone();
        let xh = self.mtp.as_ref().unwrap().xh_buf().clone();
        let argmax = self.mtp.as_ref().unwrap().argmax_buf().clone();
        let hnext = self.mtp.as_ref().unwrap().hnext_buf().clone();
        let draft_ids = self
            .mtp_verify_argmax
            .as_ref()
            .ok_or_else(|| anyhow!("mtp draft ids missing"))?
            .clone();
        let pos = self.pos;
        let cap = self.max_seq;
        let ks = self.kv_k.clone();
        let vs = self.kv_v.clone();

        self.ctx.write_buffer_nosync(&self.tok_idx[0], &[id_last as f32]);
        for step in 0..n {
            let idx = if step == 0 {
                &self.tok_idx[0]
            } else {
                &argmax
            };
            self.eng
                .copy_scale_indexed_wd(&self.ctx, h, scale, dt, &embd, idx, &xh)?;
            self.ctx.encoder_barrier();
            let hsrc = if step == 0 { &hseed } else { &hnext };
            self.eng
                .copy_scale_wd(&self.ctx, h, 1.0, DType::F16, hsrc, 0, &xh, h * 2)?;
            self.ctx.encoder_barrier();
            {
                let draft = self.mtp.as_mut().expect("mtp");
                draft.forward_from_xh(&self.ctx, pos, pos, cap, &ks, &vs)?;
            }
            self.ctx.blit_copy_bytes(
                &argmax,
                0,
                &draft_ids,
                (step * std::mem::size_of::<f32>()) as u64,
                std::mem::size_of::<f32>() as u64,
            );
        }
        Ok(())
    }

    /// Build verify token buffer on GPU: `chunk_tok = [id_last] ++ draft_ids[0..n_draft]`.
    fn mtp_pack_verify_toks(&mut self, id_last: u32, n_draft: usize) -> Result<()> {
        let draft_ids = self
            .mtp_verify_argmax
            .as_ref()
            .ok_or_else(|| anyhow!("mtp draft ids missing"))?
            .clone();
        self.ctx.write_buffer_nosync(&self.chunk_tok, &[id_last as f32]);
        if n_draft > 0 {
            self.ctx.encoder_barrier();
            self.ctx.blit_copy_bytes(
                &draft_ids,
                0,
                &self.chunk_tok,
                std::mem::size_of::<f32>() as u64,
                (n_draft * std::mem::size_of::<f32>()) as u64,
            );
        }
        self.ctx.encoder_barrier();
        Ok(())
    }

    fn generate_mtp(
        &mut self,
        mut id_last: u32,
        n_new: usize,
        verbose: bool,
        eos_stop: bool,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let max_draft = mtp_draft_steps();
        let mut n_cycles = 0usize;
        let mut draft_ms = 0.0f64;
        let mut verify_ms = 0.0f64;
        while out.len() < n_new {
            let remain = n_new - out.len();
            let n_draft = max_draft.min(remain).max(1);
            let pos_before = self.pos;
            let t0 = Instant::now();
            self.mtp_draft_chain(id_last, n_draft)?;
            self.mtp_pack_verify_toks(id_last, n_draft)?;
            draft_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.mtp_tried += n_draft;

            let t1 = Instant::now();
            let preds = self.forward_verify_from_chunk(n_draft + 1)?;
            // Drafts live in chunk_tok[1..] (argmax was overwritten by verify).
            let packed = self.ctx.read_f32(&self.chunk_tok, n_draft + 1);
            let drafted: Vec<u32> = packed[1..].iter().map(|&v| v as u32).collect();
            verify_ms += t1.elapsed().as_secs_f64() * 1e3;
            n_cycles += 1;

            let mut ids = Vec::new();
            let mut n_acc = 0usize;
            for i in 0..drafted.len() {
                ids.push(preds[i]);
                if preds[i] != drafted[i] {
                    break;
                }
                n_acc += 1;
            }
            if n_acc == drafted.len() {
                ids.push(preds[drafted.len()]);
            }
            self.mtp_hits += n_acc;

            let rewind = drafted.len() - n_acc;
            self.pos = pos_before + (n_draft + 1) - rewind;
            let i_h = n_acc.min(n_draft);
            self.mtp_copy_hseed(i_h)?;

            for &tok in &ids {
                if out.len() >= n_new {
                    break;
                }
                out.push(tok);
                if verbose {
                    let piece = self
                        .vocab
                        .as_ref()
                        .map(|v| v.decode(&[tok], false))
                        .unwrap_or_default();
                    eprintln!("  token[{}] = {tok} {piece:?}", out.len() - 1);
                }
                if eos_stop && (tok == 1 || tok == 106) {
                    return Ok(());
                }
            }
            id_last = *ids.last().unwrap();
        }
        if n_cycles > 0 {
            eprintln!(
                "[mtp-time] cycles={} draft={:.1}ms/cyc verify={:.1}ms/cyc",
                n_cycles,
                draft_ms / n_cycles as f64,
                verify_ms / n_cycles as f64
            );
        }
        Ok(())
    }

    pub fn generate_timed(
        &mut self,
        prompt_tokens: &[u32],
        n_new: usize,
        verbose: bool,
        eos_stop: bool,
    ) -> Result<GenerateStats> {
        self.reset();
        if prompt_tokens.is_empty() {
            bail!("empty prompt");
        }
        let mut out = Vec::new();
        let t_prefill = Instant::now();
        let prefill_toks = &prompt_tokens[..prompt_tokens.len().saturating_sub(1)];
        self.prefill_chunk(prefill_toks)?;
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let prefill_tokens = prefill_toks.len();

        let mut tok = *prompt_tokens.last().unwrap();
        // MTP spec-dec cannot use the ping-pong GPU pipe (needs `best` + `x2`).
        let gpu_pipe = self.ple_embd.is_some() && self.mtp.is_none();
        let t_decode = Instant::now();
        if self.mtp.is_some() {
            let next = self
                .forward_token(tok, true)?
                .ok_or_else(|| anyhow!("decode expected logits"))?;
            out.push(next);
            if verbose {
                let piece = self
                    .vocab
                    .as_ref()
                    .map(|v| v.decode(&[next], false))
                    .unwrap_or_default();
                eprintln!("  token[0] = {next} {piece:?}");
            }
            if (!eos_stop || (next != 1 && next != 106)) && out.len() < n_new {
                self.mtp_copy_hseed(0)?;
                self.generate_mtp(next, n_new, verbose, eos_stop, &mut out)?;
            }
        } else if gpu_pipe {
            self.ctx.synchronize()?;
            self.ctx
                .write_buffer_nosync(&self.tok_idx[0], &[tok as f32]);
            self.return_token_sync = false;
            let mut stopped = false;
            for i in 0..n_new {
                let step_t0 = Instant::now();
                self.decode_src = i & 1;
                let _ = self.forward_token(tok, true)?;
                self.ctx.flush_async();
                if i > 0 {
                    self.ctx.wait_inflight_at_most(1);
                    let next = self.ctx.read_f32_nosync(&self.tok_idx[i & 1], 1)[0] as u32;
                    out.push(next);
                    tok = next;
                    if verbose {
                        let piece = self
                            .vocab
                            .as_ref()
                            .map(|v| v.decode(&[next], false))
                            .unwrap_or_default();
                        eprintln!(
                            "  token[{}] = {next} {piece:?}  ({:.0}ms)",
                            out.len() - 1,
                            step_t0.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    if eos_stop && (next == 1 || next == 106) {
                        stopped = true;
                        break;
                    }
                }
            }
            if !stopped && n_new > 0 {
                self.ctx.wait_inflight_at_most(0);
                let next = self.ctx.read_f32_nosync(&self.tok_idx[n_new & 1], 1)[0] as u32;
                out.push(next);
                if verbose {
                    let piece = self
                        .vocab
                        .as_ref()
                        .map(|v| v.decode(&[next], false))
                        .unwrap_or_default();
                    eprintln!("  token[{}] = {next} {piece:?}", out.len() - 1);
                }
            }
            self.return_token_sync = true;
        } else {
            for i in 0..n_new {
                let step_t0 = Instant::now();
                let next = self
                    .forward_token(tok, true)?
                    .ok_or_else(|| anyhow!("decode expected logits"))?;
                out.push(next);
                tok = next;
                if verbose {
                    let piece = self
                        .vocab
                        .as_ref()
                        .map(|v| v.decode(&[next], false))
                        .unwrap_or_default();
                    eprintln!(
                        "  token[{i}] = {next} {piece:?}  ({:.0}ms)",
                        step_t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                if eos_stop && (next == 1 || next == 106) {
                    break;
                }
            }
        }
        let decode_s = t_decode.elapsed().as_secs_f64();
        if verbose {
            eprintln!(
                "prefill: {} tokens in {:.2}s ({:.1} tok/s) | decode: {} new in {:.2}s ({:.1} tok/s)",
                prefill_tokens,
                prefill_s,
                prefill_tokens as f64 / prefill_s.max(1e-6),
                out.len(),
                decode_s,
                out.len() as f64 / decode_s.max(1e-6)
            );
        }
        Ok(GenerateStats {
            tokens: out,
            prefill_tokens,
            prefill_s,
            decode_s,
        })
    }

    pub fn generate(&mut self, prompt_tokens: &[u32], n_new: usize) -> Result<Vec<u32>> {
        Ok(self.generate_timed(prompt_tokens, n_new, true, true)?.tokens)
    }
}


/// Read one weight row off the GPU and dequantize to f32 (probe helper).
fn ctx_read_row(
    model: &mut GemmaPrimModel,
    buf: &Buffer,
    off: usize,
    len: usize,
    h: usize,
    dt: DType,
) -> Result<Vec<f32>> {
    let raw = model.ctx.read_bytes(buf, off, len);
    let mut out = vec![0f32; h];
    let gt = match dt {
        DType::Q4K => ggml_type::Q4_K,
        DType::Q5K => ggml_type::Q5_K,
        DType::Q6K => ggml_type::Q6_K,
        _ => ggml_type::F16,
    };
    dequant_row_to_f32(gt, &raw, h, &mut out);
    Ok(out)
}

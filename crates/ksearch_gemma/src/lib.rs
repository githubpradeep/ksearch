//! Gemma4 full forward on generated Metal kernels (Thesis A).
//!
//! Includes PLE, embed√H, V-norm, SWA windowing, and metal-matched RoPE freqs.
//! Weights: native Q4_K/Q6_K on Metal via IR-lowered matvecs (no gen_* farm).

mod kv_pool;

pub use kv_pool::{KvPool, KvSlot, SlotId};

use anyhow::{anyhow, bail, Result};
use ksearch_gguf::Gguf;
use ksearch_ir::q40_nbytes;
use ksearch_kernels::Eng;
use ksearch_metal::MetalContext;
use metal::Buffer;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Clone)]
pub struct GemmaConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim_swa: usize,
    pub head_dim_full: usize,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub ffn: Vec<usize>,
    pub swa_pattern: Vec<bool>,
    pub rms_eps: f32,
    pub softcap: f32,
    pub vocab: usize,
    pub ple_dim: usize,
    pub rope_theta_full: f32,
    pub rope_theta_swa: f32,
    pub partial_rotary: f32,
}

impl GemmaConfig {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let n_layers = g.get_u32("gemma4.block_count").ok_or_else(|| anyhow!("block_count"))? as usize;
        let hidden = g.get_u32("gemma4.embedding_length").unwrap() as usize;
        let n_heads = g.get_u32("gemma4.attention.head_count").unwrap() as usize;
        let n_kv = g.get_u32("gemma4.attention.head_count_kv").unwrap_or(1) as usize;
        let head_dim_full = g.get_u32("gemma4.attention.key_length").unwrap_or(512) as usize;
        let head_dim_swa = g.get_u32("gemma4.attention.key_length_swa").unwrap_or(256) as usize;
        let sliding_window = g.get_u32("gemma4.attention.sliding_window").unwrap_or(512) as usize;
        let shared_kv_layers = g.get_u32("gemma4.attention.shared_kv_layers").unwrap_or(0) as usize;
        let rms_eps = g.get_f32("gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let softcap = g.get_f32("gemma4.final_logit_softcapping").unwrap_or(30.0);
        let ple_dim = g
            .get_u32("gemma4.embedding_length_per_layer_input")
            .unwrap_or(256) as usize;
        let rope_theta_full = g.get_f32("gemma4.rope.freq_base").unwrap_or(1_000_000.0);
        let rope_theta_swa = g.get_f32("gemma4.rope.freq_base_swa").unwrap_or(10_000.0);

        let ffn = g
            .get_usize_list("gemma4.feed_forward_length")
            .unwrap_or_else(|| {
                vec![g.get_u32("gemma4.feed_forward_length").unwrap_or(6144) as usize; n_layers]
            });
        let ffn = if ffn.len() == 1 {
            vec![ffn[0]; n_layers]
        } else if ffn.len() == n_layers {
            ffn
        } else {
            vec![6144; n_layers]
        };
        let swa_pattern = g
            .get_arr_bool("gemma4.attention.sliding_window_pattern")
            .map(|v| v.to_vec())
            .unwrap_or_else(|| (0..n_layers).map(|i| (i + 1) % 6 != 0).collect());

        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.n_rows())
            .unwrap_or(262144);

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv,
            head_dim_swa,
            head_dim_full,
            sliding_window,
            shared_kv_layers,
            ffn,
            swa_pattern,
            rms_eps,
            softcap,
            vocab,
            ple_dim,
            rope_theta_full,
            rope_theta_swa,
            partial_rotary: 0.25,
        })
    }

    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa_pattern.get(layer).copied().unwrap_or(true)
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_swa(layer) {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    pub fn owns_kv(&self, layer: usize) -> bool {
        layer < self.n_layers.saturating_sub(self.shared_kv_layers)
    }

    pub fn kv_source(&self, layer: usize) -> usize {
        if self.owns_kv(layer) {
            layer
        } else {
            let want_swa = self.is_swa(layer);
            (0..layer)
                .rev()
                .find(|&i| self.owns_kv(i) && self.is_swa(i) == want_swa)
                .unwrap_or(0)
        }
    }

    pub fn ple_total(&self) -> usize {
        self.n_layers * self.ple_dim
    }
}

struct LayerMeta {
    q_rows: usize,
    kv_rows: usize,
    o_in: usize,
    ffn_inter: usize,
    hd: usize,
}

struct LayerNorms {
    attn_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    post_attn_norm: Buffer,
    ffn_norm: Buffer,
    post_ffw_norm: Buffer,
    post_norm: Buffer,
    layer_scale: f32,
}

pub struct GemmaModel {
    pub cfg: GemmaConfig,
    pub vocab: Option<ksearch_gguf::Vocab>,
    gguf: Gguf,
    ctx: MetalContext,
    /// Tied embed / lm_head on Metal [vocab × hidden].
    token_embd_gpu: Buffer,
    /// PLE token embd Q5_K [vocab × (n_layers * ple_dim)].
    ple_embd_gpu: Buffer,
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
    /// MWG flash-Q4 partials: `n_heads * 32 * (max_hd + 2)`.
    attn_partials: Buffer,
    tmp_ff1: Buffer,
    tmp_ff2: Buffer,
    tmp_ff3: Buffer,
    logits: Buffer,
    argmax_out: Buffer,
    token_row: Buffer,
    // PLE
    ple_tok: Buffer,
    ple_ctx: Buffer,
    ple_tmp: Buffer,
    ple_gate: Buffer,
    ple_u: Buffer,
    ple_proj: Buffer,
    ple_slice: Buffer,
    meta: Vec<Buffer>,
    cos_sin: Vec<Buffer>,
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
    /// u32[1] token index for Q4_0 KV append (shared; constant within a decode step).
    kv_pos: Buffer,
    /// Scalar inv-rms for fused MLP/attn matvecs.
    inv_rms: Buffer,
    eng: Eng,
    /// Projection weights resident on Metal (Q4_K packed or dequant f32).
    weight_bufs: HashMap<String, WeightGpu>,
    max_seq: usize,
    pub pos: usize,
}

enum WeightGpu {
    Q4K(Buffer),
    Q6K(Buffer),
    F32(Buffer),
}

impl GemmaModel {
    pub fn load(path: impl AsRef<Path>, max_seq: usize) -> Result<Self> {
        let path = path.as_ref();
        eprintln!("Loading GGUF {} …", path.display());
        let g = Gguf::open(path);
        let cfg = GemmaConfig::from_gguf(&g)?;
        let vocab = ksearch_gguf::Vocab::from_gguf(&g);
        if let Some(ref v) = vocab {
            eprintln!("Vocab: {} pieces (from GGUF)", v.len());
        }
        eprintln!(
            "Gemma4: layers={} hidden={} heads={} vocab={} swa={} shared_kv={} ple={}",
            cfg.n_layers,
            cfg.hidden,
            cfg.n_heads,
            cfg.vocab,
            cfg.sliding_window,
            cfg.shared_kv_layers,
            cfg.ple_dim
        );

        let ctx = MetalContext::new()?;
        eprintln!("Metal: {}", ctx.device_name());

        eprintln!("Upload token_embd Q4_K → Metal…");
        let embd_ty = g.tensor_type("token_embd.weight");
        assert_eq!(
            embd_ty,
            ksearch_gguf::ggml_type::Q4_K,
            "expected Q4_K token_embd, got {}",
            ksearch_gguf::ggml_type_name(embd_ty)
        );
        let embd_raw = g.tensor_raw("token_embd.weight");
        let (epb, bpb) = ksearch_gguf::block_spec(embd_ty);
        let expect = (cfg.vocab * cfg.hidden / epb) * bpb;
        assert_eq!(embd_raw.len(), expect, "token_embd Q4_K byte length");
        let token_embd_gpu = ctx.buffer_bytes(embd_raw);
        eprintln!(
            "  token_embd on Metal: {:.1} MB (native Q4_K)",
            embd_raw.len() as f64 / 1e6
        );

        eprintln!("Upload per_layer_token_embd Q5_K → Metal…");
        let ple_ty = g.tensor_type("per_layer_token_embd.weight");
        assert_eq!(
            ple_ty,
            ksearch_gguf::ggml_type::Q5_K,
            "expected Q5_K PLE embd, got {}",
            ksearch_gguf::ggml_type_name(ple_ty)
        );
        let ple_raw = g.tensor_raw("per_layer_token_embd.weight");
        let ple_row_elems = cfg.ple_total();
        let (epb_p, bpb_p) = ksearch_gguf::block_spec(ple_ty);
        let expect_ple = (cfg.vocab * ple_row_elems / epb_p) * bpb_p;
        assert_eq!(ple_raw.len(), expect_ple, "PLE embd Q5_K byte length");
        let ple_embd_gpu = ctx.buffer_bytes(ple_raw);
        eprintln!(
            "  PLE embd on Metal: {:.1} MB (native Q5_K)",
            ple_raw.len() as f64 / 1e6
        );

        let output_norm = ctx.buffer_f32(&g.dequant_to_f32("output_norm.weight"));
        let ple_proj_norm_cpu = g.dequant_to_f32("per_layer_proj_norm.weight");
        assert_eq!(
            ple_proj_norm_cpu.len(),
            cfg.ple_dim,
            "per_layer_proj_norm should be ple_dim (shared across layers)"
        );
        let ple_proj_norm = ctx.buffer_f32(&ple_proj_norm_cpu);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut layer_norms = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            eprint!("\rMeta layer {i}/{} …", cfg.n_layers);
            let hd = cfg.head_dim(i);
            let inter = cfg.ffn[i];
            let q_rows = cfg.n_heads * hd;
            let kv_rows = cfg.n_kv * hd;
            let pref = format!("blk.{i}.");
            layer_norms.push(LayerNorms {
                attn_norm: ctx.buffer_f32(&g.dequant_to_f32(&format!("{pref}attn_norm.weight"))),
                q_norm: ctx.buffer_f32(&g.dequant_to_f32(&format!("{pref}attn_q_norm.weight"))),
                k_norm: ctx.buffer_f32(&g.dequant_to_f32(&format!("{pref}attn_k_norm.weight"))),
                post_attn_norm: ctx
                    .buffer_f32(&g.dequant_to_f32(&format!("{pref}post_attention_norm.weight"))),
                ffn_norm: ctx.buffer_f32(&g.dequant_to_f32(&format!("{pref}ffn_norm.weight"))),
                post_ffw_norm: ctx
                    .buffer_f32(&g.dequant_to_f32(&format!("{pref}post_ffw_norm.weight"))),
                post_norm: ctx.buffer_f32(&g.dequant_to_f32(&format!("{pref}post_norm.weight"))),
                layer_scale: g.dequant_to_f32(&format!("{pref}layer_output_scale.weight"))[0],
            });
            layers.push(LayerMeta {
                q_rows,
                kv_rows,
                o_in: q_rows,
                ffn_inter: inter,
                hd,
            });
        }
        let max_proj = layers
            .iter()
            .map(|l| {
                (l.q_rows * cfg.hidden)
                    .max(l.kv_rows * cfg.hidden)
                    .max(cfg.hidden * l.o_in)
                    .max(l.ffn_inter * cfg.hidden)
                    .max(cfg.hidden * l.ffn_inter)
                    .max(cfg.ple_dim * cfg.hidden)
                    .max(cfg.hidden * cfg.ple_dim)
            })
            .max()
            .unwrap_or(1)
            .max(cfg.ple_total() * cfg.hidden);
        eprintln!(
            "\nWeights: native Q4_K on Metal (largest logical matvec {} elems)",
            max_proj
        );

        let max_hd = cfg.head_dim_full.max(cfg.head_dim_swa);
        let max_ff = cfg.ffn.iter().copied().max().unwrap_or(6144);
        let ple_total = cfg.ple_total();
        let x = ctx.buffer_empty_f32(cfg.hidden);
        let x2 = ctx.buffer_empty_f32(cfg.hidden);
        let tmp_q = ctx.buffer_empty_f32(cfg.n_heads * max_hd);
        let tmp_k = ctx.buffer_empty_f32(cfg.n_kv * max_hd);
        let tmp_v = ctx.buffer_empty_f32(cfg.n_kv * max_hd);
        let tmp_o = ctx.buffer_empty_f32(cfg.n_heads * max_hd);
        let attn_partials = ctx.buffer_empty_f32(cfg.n_heads * 32 * (max_hd + 2));
        let tmp_ff1 = ctx.buffer_empty_f32(max_ff.max(ple_total));
        let tmp_ff2 = ctx.buffer_empty_f32(max_ff.max(ple_total));
        let tmp_ff3 = ctx.buffer_empty_f32(max_ff.max(ple_total));
        let logits = ctx.buffer_empty_f32(cfg.vocab);
        let argmax_out = ctx.buffer_empty_f32(1);
        let token_row = ctx.buffer_empty_f32(1);
        let ple_tok = ctx.buffer_empty_f32(ple_total);
        let ple_ctx = ctx.buffer_empty_f32(ple_total);
        let ple_tmp = ctx.buffer_empty_f32(ple_total);
        let ple_gate = ctx.buffer_empty_f32(cfg.ple_dim);
        let ple_u = ctx.buffer_empty_f32(cfg.ple_dim);
        let ple_proj = ctx.buffer_empty_f32(cfg.hidden);
        let ple_slice = ctx.buffer_empty_f32(cfg.ple_dim);
        let mut meta = Vec::with_capacity(cfg.n_layers);
        let mut cos_sin = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            meta.push(
                ctx.device
                    .new_buffer(32, metal::MTLResourceOptions::StorageModeShared),
            );
            cos_sin.push(ctx.buffer_empty_f32(max_hd));
        }
        let n_kv_owners = cfg.n_layers - cfg.shared_kv_layers;
        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        for i in 0..n_kv_owners {
            let hd = cfg.head_dim(i);
            let nbytes = q40_nbytes(max_seq, hd);
            kv_k.push(ctx.buffer_empty_bytes(nbytes));
            kv_v.push(ctx.buffer_empty_bytes(nbytes));
        }
        let kv_pos = ctx
            .device
            .new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        let inv_rms = ctx.buffer_empty_f32(1);

        let mut model = Self {
            cfg: cfg.clone(),
            vocab,
            gguf: g,
            ctx,
            token_embd_gpu,
            ple_embd_gpu,
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
            attn_partials,
            tmp_ff1,
            tmp_ff2,
            tmp_ff3,
            logits,
            argmax_out,
            token_row,
            ple_tok,
            ple_ctx,
            ple_tmp,
            ple_gate,
            ple_u,
            ple_proj,
            ple_slice,
            meta,
            cos_sin,
            kv_k,
            kv_v,
            kv_pos,
            inv_rms,
            eng: Eng::new(),
            weight_bufs: HashMap::new(),
            max_seq,
            pos: 0,
        };
        eprintln!("Preload layer weights → Metal…");
        for i in 0..cfg.n_layers {
            eprint!("\rPreload {i}/{} …", cfg.n_layers);
            let pref = format!("blk.{i}.");
            for s in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "inp_gate.weight",
                "proj.weight",
            ] {
                let _ = model.ensure_weight(&format!("{pref}{s}"));
            }
        }
        let _ = model.ensure_weight("per_layer_model_proj.weight");
        eprintln!("\nPreload done ({} tensors)", model.weight_bufs.len());
        // Compile MWG flash pipelines for both head dims so first long-T token
        // doesn't pay Metal compile latency inside the decode timer.
        for &hd in &[cfg.head_dim_swa, cfg.head_dim_full] {
            model
                .eng
                .ensure_attn_gqa_q4_mwg(&model.ctx, cfg.n_heads, hd, max_seq)?;
        }
        Ok(model)
    }

    /// Upload weight once onto Metal (Q4_K/Q6_K packed, else dequant→f32 once).
    fn ensure_weight(&mut self, name: &str) -> Result<()> {
        if self.weight_bufs.contains_key(name) {
            return Ok(());
        }
        let ty = self.gguf.tensor_type(name);
        let gpu = match ty {
            ksearch_gguf::ggml_type::Q4_K => {
                WeightGpu::Q4K(self.ctx.buffer_bytes(self.gguf.tensor_raw(name)))
            }
            ksearch_gguf::ggml_type::Q6_K => {
                WeightGpu::Q6K(self.ctx.buffer_bytes(self.gguf.tensor_raw(name)))
            }
            _ => {
                let w = self.gguf.dequant_to_f32(name);
                WeightGpu::F32(self.ctx.buffer_f32(&w))
            }
        };
        self.weight_bufs.insert(name.to_string(), gpu);
        Ok(())
    }

    /// Generated matvec on Metal: Q4_K/Q6_K fused when native; else f32.
    fn matvec_w(
        &mut self,
        rows: usize,
        cols: usize,
        name: &str,
        x: Buffer,
        y: Buffer,
    ) -> Result<()> {
        self.ensure_weight(name)?;
        match self.weight_bufs.get(name).unwrap() {
            WeightGpu::Q4K(w) => {
                let w = w.clone();
                self.eng.matvec_q4k(&self.ctx, rows, cols, &w, &x, &y)
            }
            WeightGpu::Q6K(w) => {
                let w = w.clone();
                self.eng.matvec_q6k(&self.ctx, rows, cols, &w, &x, &y)
            }
            WeightGpu::F32(w) => {
                let w = w.clone();
                self.eng.matvec(&self.ctx, rows, cols, &w, &x, &y)
            }
        }
    }

    /// Q4_K matvec with fused rms via precomputed `inv_rms`; falls back to plain matvec.
    #[allow(dead_code)]
    fn matvec_w_rms(
        &mut self,
        rows: usize,
        cols: usize,
        name: &str,
        x: &Buffer,
        nw: &Buffer,
        inv: &Buffer,
        eps: f32,
        y: &Buffer,
    ) -> Result<()> {
        self.ensure_weight(name)?;
        match self.weight_bufs.get(name).unwrap() {
            WeightGpu::Q4K(w) => {
                let w = w.clone();
                self.eng
                    .matvec_q4k_rms(&self.ctx, rows, cols, eps, &w, x, nw, inv, y)
            }
            _ => {
                self.eng.rmsnorm(&self.ctx, cols, eps, x, nw, &self.x2)?;
                self.matvec_w(rows, cols, name, self.x2.clone(), y.clone())
            }
        }
    }

    /// Fused Q4 gate∥up + GeLU when both weights are Q4_K; else fall back.
    fn matvec_gate_up_gelu(
        &mut self,
        rows: usize,
        cols: usize,
        gate_name: &str,
        up_name: &str,
        x: Buffer,
        y: Buffer,
        ffn_norm: &Buffer,
        eps: f32,
        fuse_rms: bool,
    ) -> Result<()> {
        self.ensure_weight(gate_name)?;
        self.ensure_weight(up_name)?;
        let both_q4 = matches!(
            (
                self.weight_bufs.get(gate_name).unwrap(),
                self.weight_bufs.get(up_name).unwrap()
            ),
            (WeightGpu::Q4K(_), WeightGpu::Q4K(_))
        );
        if both_q4 && fuse_rms {
            let wg = match self.weight_bufs.get(gate_name).unwrap() {
                WeightGpu::Q4K(w) => w.clone(),
                _ => unreachable!(),
            };
            let wu = match self.weight_bufs.get(up_name).unwrap() {
                WeightGpu::Q4K(w) => w.clone(),
                _ => unreachable!(),
            };
            self.eng
                .inv_rms(&self.ctx, cols, eps, &x, &self.inv_rms)?;
            return self.eng.matvec_q4k_rms_gate_up_gelu(
                &self.ctx,
                rows,
                cols,
                eps,
                &wg,
                &wu,
                &x,
                ffn_norm,
                &self.inv_rms,
                &y,
            );
        }
        if both_q4 {
            let wg = match self.weight_bufs.get(gate_name).unwrap() {
                WeightGpu::Q4K(w) => w.clone(),
                _ => unreachable!(),
            };
            let wu = match self.weight_bufs.get(up_name).unwrap() {
                WeightGpu::Q4K(w) => w.clone(),
                _ => unreachable!(),
            };
            return self
                .eng
                .matvec_q4k_gate_up_gelu(&self.ctx, rows, cols, &wg, &wu, &x, &y);
        }
        self.matvec_w(rows, cols, gate_name, x.clone(), self.tmp_ff1.clone())?;
        self.matvec_w(rows, cols, up_name, x, self.tmp_ff2.clone())?;
        self.eng
            .gelu_mul(&self.ctx, rows, &self.tmp_ff1, &self.tmp_ff2, &y)
    }

    fn write_meta(&self, layer: usize, tlen: u32, start: u32) {
        // Per-layer buffer: no sync — GPU may still be reading other layers' meta.
        let ptr = self.meta[layer].contents() as *mut u32;
        unsafe {
            *ptr = tlen;
            *ptr.add(1) = start;
        }
    }

    fn write_kv_pos(&self, pos: usize) {
        let ptr = self.kv_pos.contents() as *mut u32;
        unsafe {
            *ptr = pos as u32;
        }
    }

    /// RoPE cos/sin matching metal-llm-server: freq uses full head_dim in exponent;
    /// only the first `rope_angles` pairs rotate (partial on full layers).
    fn fill_cos_sin(&self, layer: usize, pos: usize, hd: usize, theta: f32, rope_angles: usize) {
        // Per-layer buffer: host fill without flushing pending CB.
        let ptr = self.cos_sin[layer].contents() as *mut f32;
        let half = hd / 2;
        let rope_angles = rope_angles.min(half);
        unsafe {
            for i in 0..half {
                if i < rope_angles {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / hd as f32);
                    let ang = pos as f32 * freq;
                    *ptr.add(i) = ang.cos();
                    *ptr.add(half + i) = ang.sin();
                } else {
                    *ptr.add(i) = 1.0;
                    *ptr.add(half + i) = 0.0;
                }
            }
        }
    }

    pub fn embed_token(&mut self, token: u32) -> Result<()> {
        let h = self.cfg.hidden;
        let scale = (h as f32).sqrt();
        // Previous decode_step ended with sync; safe nosync write of row index.
        self.ctx
            .write_buffer_nosync(&self.token_row, &[token as f32]);
        self.eng.gather_q4k_row(
            &self.ctx,
            self.cfg.vocab,
            h,
            scale,
            &self.token_embd_gpu,
            &self.token_row,
            &self.x,
        )
    }

    fn load_ple_token(&mut self, _token: u32) -> Result<()> {
        let ple_total = self.cfg.ple_total();
        let scale = (self.cfg.ple_dim as f32).sqrt();
        // Reuse token_row (already written in embed_token) — same token index.
        self.eng.gather_q5k_row(
            &self.ctx,
            self.cfg.vocab,
            ple_total,
            scale,
            &self.ple_embd_gpu,
            &self.token_row,
            &self.ple_tok,
        )
    }

    fn ple_prepass(&mut self) -> Result<()> {
        let h = self.cfg.hidden;
        let ple_total = self.cfg.ple_total();
        let n_layers = self.cfg.n_layers;
        let ple_dim = self.cfg.ple_dim;
        let eps = self.cfg.rms_eps;
        let inv_sqrt_h = 1.0 / (h as f32).sqrt();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

        self.matvec_w(
            ple_total,
            h,
            "per_layer_model_proj.weight",
            self.x.clone(),
            self.ple_tmp.clone(),
        )?;
        self.eng
            .scale_const(&self.ctx, ple_total, inv_sqrt_h, &self.ple_tmp, &self.ple_ctx)?;

        self.eng.rmsnorm_per_head(
            &self.ctx,
            n_layers,
            ple_dim,
            eps,
            &self.ple_ctx,
            &self.ple_proj_norm,
            &self.ple_tmp,
        )?;
        self.eng
            .add(&self.ctx, ple_total, &self.ple_tmp, &self.ple_tok, &self.ple_ctx)?;
        // (ctx + tok) * 1/√2 → ple_ctx (in-place via tmp then overwrite avoided: scale into ple_ctx)
        self.eng.scale_const(
            &self.ctx,
            ple_total,
            inv_sqrt2,
            &self.ple_ctx,
            &self.ple_ctx,
        )?;
        Ok(())
    }

    /// One decode step at `self.pos`. Returns next token id (argmax).
    pub fn decode_step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.max_seq {
            bail!("max_seq exceeded");
        }
        self.embed_token(token)?;
        self.load_ple_token(token)?;
        self.ple_prepass()?;

        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let n_layers = self.cfg.n_layers;
        let n_heads = self.cfg.n_heads;
        let max_seq = self.max_seq;
        let window = self.cfg.sliding_window;
        let ple_dim = self.cfg.ple_dim;

        for layer in 0..n_layers {
            let (q_rows, kv_rows, o_in, ffn_inter, hd) = {
                let lw = &self.layers[layer];
                (lw.q_rows, lw.kv_rows, lw.o_in, lw.ffn_inter, lw.hd)
            };
            let is_swa = self.cfg.is_swa(layer);
            let theta = if is_swa {
                self.cfg.rope_theta_swa
            } else {
                self.cfg.rope_theta_full
            };
            // metal: rope_angles = rotary_dim/2; rotary_dim = hd (SWA) or hd*0.25 (full)
            let rotary_dim = if is_swa {
                hd
            } else {
                ((hd as f32) * self.cfg.partial_rotary) as usize
            };
            let rope_angles = rotary_dim / 2;
            let pref = format!("blk.{layer}.");
            let owns_kv = self.cfg.owns_kv(layer);
            let kv_src = self.cfg.kv_source(layer);

            // ── Attention ──
            self.eng.rmsnorm(
                &self.ctx,
                h,
                eps,
                &self.x,
                &self.layer_norms[layer].attn_norm,
                &self.x2,
            )?;
            self.matvec_w(
                q_rows,
                h,
                &format!("{pref}attn_q.weight"),
                self.x2.clone(),
                self.tmp_q.clone(),
            )?;
            if owns_kv {
                self.matvec_w(
                    kv_rows,
                    h,
                    &format!("{pref}attn_k.weight"),
                    self.x2.clone(),
                    self.tmp_k.clone(),
                )?;
                self.matvec_w(
                    kv_rows,
                    h,
                    &format!("{pref}attn_v.weight"),
                    self.x2.clone(),
                    self.tmp_v.clone(),
                )?;
            }

            self.fill_cos_sin(layer, self.pos, hd, theta, rope_angles);
            self.eng.rmsnorm_per_head(
                &self.ctx,
                n_heads,
                hd,
                eps,
                &self.tmp_q,
                &self.layer_norms[layer].q_norm,
                &self.tmp_q,
            )?;
            self.eng.rope(
                &self.ctx,
                n_heads,
                hd,
                &self.tmp_q,
                &self.cos_sin[layer],
                &self.tmp_q,
            )?;
            if owns_kv {
                self.eng.rmsnorm_per_head(
                    &self.ctx,
                    self.cfg.n_kv,
                    hd,
                    eps,
                    &self.tmp_k,
                    &self.layer_norms[layer].k_norm,
                    &self.tmp_k,
                )?;
                self.eng.rmsnorm_noweight(
                    &self.ctx,
                    self.cfg.n_kv,
                    hd,
                    eps,
                    &self.tmp_v,
                    &self.tmp_v,
                )?;
                self.eng.rope(
                    &self.ctx,
                    1,
                    hd,
                    &self.tmp_k,
                    &self.cos_sin[layer],
                    &self.tmp_k,
                )?;
                self.write_kv_pos(self.pos);
            }

            let kv_len = self.pos + 1;
            let (attn_t, attn_start) = if is_swa {
                let start = kv_len.saturating_sub(window);
                ((kv_len - start) as u32, start as u32)
            } else {
                (kv_len as u32, 0u32)
            };
            self.write_meta(layer, attn_t, attn_start);
            if attn_t >= 64 {
                // Long T: split-KV MWG flash (append first when we own the cache).
                // Threshold 64 (vs ggml auto's 128) wins on M1 Pro essay decode.
                if owns_kv {
                    self.eng.kv_append_q4(
                        &self.ctx,
                        hd,
                        max_seq,
                        &self.tmp_k,
                        &self.kv_pos,
                        &self.kv_k[kv_src],
                    )?;
                    self.eng.kv_append_q4(
                        &self.ctx,
                        hd,
                        max_seq,
                        &self.tmp_v,
                        &self.kv_pos,
                        &self.kv_v[kv_src],
                    )?;
                }
                self.eng.attn_gqa_q4_mwg(
                    &self.ctx,
                    n_heads,
                    hd,
                    max_seq,
                    &self.tmp_q,
                    &self.kv_k[kv_src],
                    &self.kv_v[kv_src],
                    &self.meta[layer],
                    &self.attn_partials,
                    &self.tmp_o,
                )?;
            } else if owns_kv {
                self.eng.attn_gqa_q4_fused(
                    &self.ctx,
                    n_heads,
                    hd,
                    max_seq,
                    &self.tmp_q,
                    &self.kv_k[kv_src],
                    &self.kv_v[kv_src],
                    &self.tmp_k,
                    &self.tmp_v,
                    &self.meta[layer],
                    &self.kv_pos,
                    &self.tmp_o,
                )?;
            } else {
                self.eng.attn_gqa_q4(
                    &self.ctx,
                    n_heads,
                    hd,
                    max_seq,
                    &self.tmp_q,
                    &self.kv_k[kv_src],
                    &self.kv_v[kv_src],
                    &self.meta[layer],
                    &self.tmp_o,
                )?;
            }

            self.matvec_w(
                h,
                o_in,
                &format!("{pref}attn_output.weight"),
                self.tmp_o.clone(),
                self.x2.clone(),
            )?;
            self.eng.rmsnorm_add(
                &self.ctx,
                h,
                eps,
                &self.x2,
                &self.layer_norms[layer].post_attn_norm,
                &self.x,
                &self.x,
            )?;

            // ── MLP (fused inv_rms + Q4 gate∥up+GeLU when both Q4_K) ──
            let gate_name = format!("{pref}ffn_gate.weight");
            let up_name = format!("{pref}ffn_up.weight");
            self.ensure_weight(&gate_name)?;
            self.ensure_weight(&up_name)?;
            let both_q4 = matches!(
                (
                    self.weight_bufs.get(&gate_name).unwrap(),
                    self.weight_bufs.get(&up_name).unwrap()
                ),
                (WeightGpu::Q4K(_), WeightGpu::Q4K(_))
            );
            if both_q4 {
                let ffn_norm = self.layer_norms[layer].ffn_norm.clone();
                self.matvec_gate_up_gelu(
                    ffn_inter,
                    h,
                    &gate_name,
                    &up_name,
                    self.x.clone(),
                    self.tmp_ff3.clone(),
                    &ffn_norm,
                    eps,
                    true,
                )?;
            } else {
                self.eng.rmsnorm(
                    &self.ctx,
                    h,
                    eps,
                    &self.x,
                    &self.layer_norms[layer].ffn_norm,
                    &self.x2,
                )?;
                let ffn_norm = self.layer_norms[layer].ffn_norm.clone();
                self.matvec_gate_up_gelu(
                    ffn_inter,
                    h,
                    &gate_name,
                    &up_name,
                    self.x2.clone(),
                    self.tmp_ff3.clone(),
                    &ffn_norm,
                    eps,
                    false,
                )?;
            }
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

            // ── PLE per-layer (after MLP, before layer_scale) ──
            self.matvec_w(
                ple_dim,
                h,
                &format!("{pref}inp_gate.weight"),
                self.x.clone(),
                self.ple_gate.clone(),
            )?;
            self.eng.gelu_mul_at(
                &self.ctx,
                ple_dim,
                &self.ple_gate,
                &self.ple_ctx,
                layer * ple_dim,
                &self.ple_u,
            )?;
            self.matvec_w(
                h,
                ple_dim,
                &format!("{pref}proj.weight"),
                self.ple_u.clone(),
                self.ple_proj.clone(),
            )?;
            self.eng.rmsnorm_add_scale(
                &self.ctx,
                h,
                eps,
                self.layer_norms[layer].layer_scale,
                &self.ple_proj,
                &self.layer_norms[layer].post_norm,
                &self.x,
                &self.x,
            )?;
        }

        self.eng
            .rmsnorm(&self.ctx, h, eps, &self.x, &self.output_norm, &self.x2)?;

        // Metal lm_head: Q4_K matvec + fused softcap/argmax
        let vocab = self.cfg.vocab;
        let cap = self.cfg.softcap;
        self.eng.matvec_q4k(
            &self.ctx,
            vocab,
            h,
            &self.token_embd_gpu,
            &self.x2,
            &self.logits,
        )?;
        self.eng
            .softcap_argmax(&self.ctx, vocab, cap, &self.logits, &self.argmax_out)?;
        let best = self.ctx.read_f32(&self.argmax_out, 1)[0] as u32;

        self.pos += 1;
        Ok(best)
    }

    /// Build a KvPool sized for this model (shared-KV layer count × head_dim).
    pub fn make_kv_pool(&self, max_batch: usize) -> Result<KvPool> {
        let n_kv_layers = (0..self.cfg.n_layers)
            .filter(|&i| self.cfg.owns_kv(i))
            .count()
            .max(1);
        let hd = self.cfg.head_dim_full.max(self.cfg.head_dim_swa);
        KvPool::new(&self.ctx, max_batch, self.max_seq, n_kv_layers, hd)
    }

    pub fn generate(&mut self, prompt_tokens: &[u32], n_new: usize) -> Result<Vec<u32>> {
        self.pos = 0;
        let mut out = Vec::new();
        if prompt_tokens.is_empty() {
            bail!("empty prompt");
        }
        let t_prefill = Instant::now();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            if i + 1 == prompt_tokens.len() {
                break;
            }
            let _ = self.decode_step(tok)?;
        }
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let mut tok = *prompt_tokens.last().unwrap();
        let t_decode = Instant::now();
        for i in 0..n_new {
            let step_t0 = Instant::now();
            let next = self.decode_step(tok)?;
            out.push(next);
            tok = next;
            let piece = self
                .vocab
                .as_ref()
                .map(|v| v.decode(&[next], false))
                .unwrap_or_default();
            eprintln!(
                "  token[{i}] = {next} {piece:?}  ({:.0}ms)",
                step_t0.elapsed().as_secs_f64() * 1e3
            );
            if next == 1 || next == 106 {
                // <eos> or <turn|>
                break;
            }
        }
        let decode_s = t_decode.elapsed().as_secs_f64();
        eprintln!(
            "prefill: {} tokens in {:.2}s | decode: {} new in {:.2}s ({:.1} tok/s)",
            prompt_tokens.len().saturating_sub(1),
            prefill_s,
            out.len(),
            decode_s,
            out.len() as f64 / decode_s.max(1e-6)
        );
        Ok(out)
    }
}

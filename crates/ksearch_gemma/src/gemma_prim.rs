//! Thesis A Gemma path: Graph→AST→MSL. Q4_K weights stay packed (`Load(Q4K)` expand);
//! other quant types dequant → F16. Activations stay F16.

use crate::{GemmaConfig, GenerateStats, LayerMeta, LayerNorms, KvPool};
use anyhow::{anyhow, bail, Result};
use ksearch_gguf::{f32_to_f16, ggml_type, Gguf};
use ksearch_ir::DType;
use ksearch_kernels::Eng;
use ksearch_metal::MetalContext;
use metal::Buffer;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const PREFILL_CHUNK: usize = 32;

enum WeightBuf {
    F16(Buffer),
    Q4K(Buffer),
}

impl WeightBuf {
    fn dtype(&self) -> DType {
        match self {
            WeightBuf::F16(_) => DType::F16,
            WeightBuf::Q4K(_) => DType::Q4K,
        }
    }

    fn buf(&self) -> &Buffer {
        match self {
            WeightBuf::F16(b) | WeightBuf::Q4K(b) => b,
        }
    }
}

pub struct GemmaPrimModel {
    pub cfg: GemmaConfig,
    pub vocab: Option<ksearch_gguf::Vocab>,
    gguf: Gguf,
    ctx: MetalContext,
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
    /// FFN intermediates (gate/up) when packings differ and fusion cannot share one weight dtype.
    tmp_ff1: Buffer,
    tmp_ff2: Buffer,
    tmp_ff3: Buffer,
    logits: Buffer,
    argmax_out: Buffer,
    ple_tok: Buffer,
    ple_ctx: Buffer,
    ple_tmp: Buffer,
    ple_gate: Buffer,
    ple_u: Buffer,
    ple_proj: Buffer,
    meta: Vec<Buffer>,
    cos_sin: Vec<Buffer>,
    /// F16 KV caches [max_seq × hd] per owning layer.
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
    eng: Eng,
    weight_bufs: HashMap<String, WeightBuf>,
    max_seq: usize,
    pub pos: usize,
}

impl GemmaPrimModel {
    pub fn load(path: impl AsRef<Path>, max_seq: usize) -> Result<Self> {
        let path = path.as_ref();
        eprintln!("[Thesis A / prim] Loading GGUF {} …", path.display());
        let g = Gguf::open(path);
        let cfg = GemmaConfig::from_gguf(&g)?;
        let vocab = ksearch_gguf::Vocab::from_gguf(&g);
        eprintln!(
            "[prim] layers={} hidden={} heads={} vocab={} (Q4_K weights packed; acts F16)",
            cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.vocab
        );

        let ctx = MetalContext::new()?;
        eprintln!("Metal: {}", ctx.device_name());

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
            kv_k.push(ctx.buffer_empty_f16(max_seq * hd));
            kv_v.push(ctx.buffer_empty_f16(max_seq * hd));
        }

        let mut meta = Vec::with_capacity(cfg.n_layers);
        let mut cos_sin = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            meta.push(
                ctx.device
                    .new_buffer(32, metal::MTLResourceOptions::StorageModeShared),
            );
            cos_sin.push(ctx.buffer_empty_f16(max_hd));
        }

        let x = ctx.buffer_empty_f16(cfg.hidden);
        let x2 = ctx.buffer_empty_f16(cfg.hidden);
        let tmp_q = ctx.buffer_empty_f16(cfg.n_heads * max_hd);
        let tmp_k = ctx.buffer_empty_f16(cfg.n_kv * max_hd);
        let tmp_v = ctx.buffer_empty_f16(cfg.n_kv * max_hd);
        let tmp_o = ctx.buffer_empty_f16(cfg.n_heads * max_hd);
        let tmp_ff1 = ctx.buffer_empty_f16(max_ff);
        let tmp_ff2 = ctx.buffer_empty_f16(max_ff);
        let tmp_ff3 = ctx.buffer_empty_f16(max_ff.max(ple_total));
        let logits = ctx.buffer_empty_f16(cfg.vocab);
        let argmax_out = ctx.buffer_empty_f32(1);
        let ple_tok = ctx.buffer_empty_f16(ple_total);
        let ple_ctx = ctx.buffer_empty_f16(ple_total);
        let ple_tmp = ctx.buffer_empty_f16(ple_total);
        let ple_gate = ctx.buffer_empty_f16(cfg.ple_dim);
        let ple_u = ctx.buffer_empty_f16(cfg.ple_dim);
        let ple_proj = ctx.buffer_empty_f16(cfg.hidden);

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
        Ok(Self {
            cfg: cfg.clone(),
            vocab,
            gguf: g,
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
            tmp_ff1,
            tmp_ff2,
            tmp_ff3,
            logits,
            argmax_out,
            ple_tok,
            ple_ctx,
            ple_tmp,
            ple_gate,
            ple_u,
            ple_proj,
            meta,
            cos_sin,
            kv_k,
            kv_v,
            eng,
            weight_bufs: HashMap::new(),
            max_seq,
            pos: 0,
        })
    }

    fn ensure_weight(&mut self, name: &str) -> Result<()> {
        if self.weight_bufs.contains_key(name) {
            return Ok(());
        }
        let buf = if self.gguf.tensor_type(name) == ggml_type::Q4_K {
            WeightBuf::Q4K(self.ctx.buffer_bytes(self.gguf.tensor_raw(name)))
        } else {
            // Non-Q4: dequant → F16 (tinygrad ggml_data_to_tensor → .half()).
            WeightBuf::F16(self.ctx.buffer_bytes(&self.gguf.dequant_to_f16_bytes(name)))
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
        if dq == dk && dk == dv {
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
        let ptr = self.meta[layer].contents() as *mut u16;
        unsafe {
            *ptr = f32_to_f16(tlen as f32);
            *ptr.add(1) = f32_to_f16(start as f32);
        }
    }

    fn fill_cos_sin(&self, layer: usize, pos: usize, hd: usize, theta: f32, rope_angles: usize) {
        let ptr = self.cos_sin[layer].contents() as *mut u16;
        let half = hd / 2;
        let rope_angles = rope_angles.min(half);
        unsafe {
            for i in 0..half {
                if i < rope_angles {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / hd as f32);
                    let ang = pos as f32 * freq;
                    *ptr.add(i) = f32_to_f16(ang.cos());
                    *ptr.add(half + i) = f32_to_f16(ang.sin());
                } else {
                    *ptr.add(i) = f32_to_f16(1.0);
                    *ptr.add(half + i) = f32_to_f16(0.0);
                }
            }
        }
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
            WeightBuf::Q4K(_) => {
                // Row lookup: host-dequant one Q4_K row → F16 activation (acts stay F16).
                let mut row = self
                    .gguf
                    .dequant_row("token_embd.weight", token as usize);
                for v in &mut row {
                    *v *= scale;
                }
                let row_h: Vec<u16> = row.iter().map(|&v| f32_to_f16(v)).collect();
                if self.ctx.has_gpu_work() {
                    self.ctx.synchronize()?;
                }
                self.ctx.write_u16s_nosync(&self.x, &row_h);
            }
        }
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
        // Avoid a full GPU sync on every token: decode already synced on prior argmax read;
        // during prefill, flush+wait only if work is in flight.
        if self.ctx.has_gpu_work() {
            self.ctx.synchronize()?;
        }
        self.ctx.write_u16s_nosync(&self.ple_tok, &row_h);
        Ok(())
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
        self.eng.scale_const(
            &self.ctx,
            ple_total,
            inv_sqrt_h,
            &self.ple_tmp,
            &self.ple_ctx,
        )?;
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
        self.eng.scale_const(
            &self.ctx,
            ple_total,
            inv_sqrt2,
            &self.ple_ctx,
            &self.ple_ctx,
        )?;
        Ok(())
    }

    fn kv_append_f16(&mut self, kv_src: usize, hd: usize) -> Result<()> {
        let pos = self.pos;
        self.eng.copy_slice(
            &self.ctx,
            hd,
            &self.tmp_k,
            0,
            &self.kv_k[kv_src],
            pos * hd,
        )?;
        self.eng.copy_slice(
            &self.ctx,
            hd,
            &self.tmp_v,
            0,
            &self.kv_v[kv_src],
            pos * hd,
        )?;
        Ok(())
    }

    fn forward_token(&mut self, token: u32, want_logits: bool) -> Result<Option<u32>> {
        if self.pos >= self.max_seq {
            bail!("max_seq exceeded");
        }
        let profile = std::env::var_os("KSEARCH_PROFILE").is_some();
        let t0 = Instant::now();
        self.embed_token(token)?;
        let t_embed = t0.elapsed();
        self.load_ple_token(token)?;
        let t_ple = t0.elapsed();
        self.ple_prepass()?;
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
            let theta = if is_swa {
                self.cfg.rope_theta_swa
            } else {
                self.cfg.rope_theta_full
            };
            let rotary_dim = if is_swa {
                hd
            } else {
                ((hd as f32) * self.cfg.partial_rotary) as usize
            };
            let rope_angles = rotary_dim / 2;
            let pref = format!("blk.{layer}.");
            let owns_kv = self.cfg.owns_kv(layer);
            let kv_src = self.cfg.kv_source(layer);

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

            self.fill_cos_sin(layer, self.pos, hd, theta, rope_angles);
            self.eng.rmsnorm_per_head_rope(
                &self.ctx,
                n_heads,
                hd,
                eps,
                &self.tmp_q,
                &self.layer_norms[layer].q_norm,
                &self.cos_sin[layer],
                &self.tmp_q,
            )?;
            if owns_kv {
                self.eng.rmsnorm_per_head_rope(
                    &self.ctx,
                    self.cfg.n_kv,
                    hd,
                    eps,
                    &self.tmp_k,
                    &self.layer_norms[layer].k_norm,
                    &self.cos_sin[layer],
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
                self.kv_append_f16(kv_src, hd)?;
            }

            let kv_len = self.pos + 1;
            let (attn_t, attn_start) = if is_swa {
                let start = kv_len.saturating_sub(window);
                ((kv_len - start) as u32, start as u32)
            } else {
                (kv_len as u32, 0u32)
            };
            self.write_meta(layer, attn_t, attn_start);
            self.eng.sdpa_naive(
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

            self.matvec_w(h, o_in, &format!("{pref}attn_output.weight"), self.tmp_o.clone(), self.x2.clone())?;
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
            let ffn_norm = self.layer_norms[layer].ffn_norm.clone();
            self.rmsnorm_matvec_gate_up_gelu_w(
                ffn_inter,
                h,
                eps,
                &format!("{pref}ffn_gate.weight"),
                &format!("{pref}ffn_up.weight"),
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
            let t_ple_l = if profile {
                Some(Instant::now())
            } else {
                None
            };

            // PLE
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
            if let Some(ts) = t_ple_l {
                self.ctx.synchronize()?;
                ms_ple_l += ts.elapsed().as_secs_f64() * 1e3;
            }
        }

        let t_layers = t0.elapsed();
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
            }
            return Ok(None);
        }

        // Thesis-A fused output head: RmsNormMatvec(output_norm, token_embd)
        // then parallel softcap_argmax — Q4_K embd uses Load(Q4K) expand.
        let vocab = self.cfg.vocab;
        let cap = self.cfg.softcap;
        let embd_dt = self.token_embd.dtype();
        let embd = self.token_embd.buf().clone();
        self.eng.rmsnorm_matvec_wd(
            &self.ctx,
            vocab,
            h,
            eps,
            embd_dt,
            &self.x,
            &self.output_norm,
            &embd,
            &self.logits,
        )?;
        self.eng
            .softcap_argmax(&self.ctx, vocab, cap, &self.logits, &self.argmax_out)?;
        let t_logits = t0.elapsed();
        let best = self.ctx.read_f32(&self.argmax_out, 1)[0] as u32;
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
                (t0.elapsed() - t_logits).as_secs_f64() * 1e3 + (t_logits - t_layers).as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(Some(best))
    }

    /// F32 KV pool for multi-stream decode (P5 serving contract).
    pub fn make_kv_pool(&self, max_batch: usize) -> Result<KvPool> {
        let n_kv_layers = (0..self.cfg.n_layers)
            .filter(|&i| self.cfg.owns_kv(i))
            .count()
            .max(1);
        let hd = self.cfg.head_dim_full.max(self.cfg.head_dim_swa);
        KvPool::new_f16(&self.ctx, max_batch, self.max_seq, n_kv_layers, hd)
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.ctx.synchronize().ok();
        for b in self.kv_k.iter().chain(self.kv_v.iter()) {
            let n = b.length() as usize;
            unsafe {
                std::ptr::write_bytes(b.contents() as *mut u8, 0, n);
            }
        }
    }

    pub fn generate_timed(
        &mut self,
        prompt_tokens: &[u32],
        n_new: usize,
        verbose: bool,
    ) -> Result<GenerateStats> {
        self.reset();
        if prompt_tokens.is_empty() {
            bail!("empty prompt");
        }
        let mut out = Vec::new();
        let t_prefill = Instant::now();
        let prefill_toks = &prompt_tokens[..prompt_tokens.len().saturating_sub(1)];
        // P5: chunked prefill (oracle order later; for now chunked serial advances).
        for chunk in prefill_toks.chunks(PREFILL_CHUNK) {
            for &tok in chunk {
                let _ = self.forward_token(tok, false)?;
            }
        }
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let prefill_tokens = prefill_toks.len();

        let mut tok = *prompt_tokens.last().unwrap();
        let t_decode = Instant::now();
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
            if next == 1 || next == 106 {
                break;
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
        Ok(self.generate_timed(prompt_tokens, n_new, true)?.tokens)
    }
}

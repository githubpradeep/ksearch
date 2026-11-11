//! MTP draft head ("gemma4-assistant" GGUF): a 4-layer, 256-dim assistant
//! model that reads the base model's Q4_0 KV cache and proposes next tokens
//! for speculative verify. Semantics mirror
//! `reference/metal-llm-server/src/speculative.rs` (`forward_draft_step`)
//! and llama.cpp `gemma4-assistant.cpp`:
//!
//! ```text
//! xh   = [emb(best)*√hb ; h_post_norm]       (2*hb)
//!        best = token just sampled from the target; h is target post-final-norm
//! cur  = pre_proj · xh                        (hh)
//! ×N layers:
//!   a    = rmsnorm(cur, attn_norm)
//!   q    = w_q · a;  q = rmsnorm_per_head(q, q_norm) ⊙ rope(pos)
//!   att  = sdpa(q, base_kv[mapped_layer])      (GQA read, no KV write)
//!   cur  = rmsnorm(w_o · att, post_attention)
//!   aout = cur + h                             (residual)
//!   ffin = rmsnorm(aout, ffn_norm)
//!   h'   = down(gelu(gate·ffin) ⊙ up·ffin) + aout, scaled by layer_output_scale
//! fnorm = rmsnorm(h, output_norm)
//! token = argmax(softcap(tok_embd · fnorm))    (draft's own tied head)
//! hnext = post_proj · fnorm                    (seed for next draft step)
//! ```

use crate::GemmaConfig;
use anyhow::{anyhow, Context, Result};
use ksearch_compute::{
    Buffer, Context as GpuContext, KernelSource as MetalKernelSource,
    Pipeline as ComputePipelineState,
};
use ksearch_gguf::{f16_to_f32, f32_to_f16, Gguf};
use ksearch_ir::{DType, Graph, Shape};
use ksearch_kernels::Eng;
use std::collections::HashMap;

#[derive(Clone)]
pub struct DraftLayer {
    attn_norm: Buffer,
    q_w: Buffer,
    q_norm: Buffer,
    out_w: Buffer,
    post_attn: Buffer,
    ffn_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    post_ffw: Buffer,
    out_scale: f32,
    is_full: bool,
    nh: usize,
    hd: usize,
    ffn_inter: usize,
    kv_map: usize,
}

pub struct MtpDraft {
    eng: Eng,
    layers: Vec<DraftLayer>,
    tok_embd: Buffer,  // F16 [hh, vocab] — draft's own tied output head
    pre_proj: Buffer,  // F16 [hb*2 -> hh]
    post_proj: Buffer, // F16 [hh -> hb]
    output_norm: Buffer,
    pub vocab: usize,
    pub hb: usize, // base hidden dim (1536)
    hh: usize,     // draft hidden dim (256)
    softcap: f32,
    rms_eps: f32,
    sliding_window: usize,
    base_n_kv: usize,
    rope_swa: Buffer,
    rope_full: Buffer,
    meta_swa: Buffer,
    meta_full: Buffer,
    tmp_swa: Buffer,
    tmp_full: Buffer,
    argmax: Buffer,  // F32[1] — argmax output
    logits: Buffer,  // F16[vocab] — draft head output
    hnext16: Buffer, // F16[hb] — post_proj output seed
    // scratch
    xh: Buffer,
    h: Buffer,
    a: Buffer,
    q: Buffer,
    att: Buffer,
    cur: Buffer,
    aout: Buffer,
    ffin: Buffer,
    gate: Buffer,
    up: Buffer,
    inter: Buffer,
    fdown: Buffer,
    fnorm: Buffer,
    att_out: Buffer,
    cache: HashMap<String, (MetalKernelSource, ComputePipelineState)>,
}

fn rope_table(
    max_seq: usize,
    hd: usize,
    theta: f64,
    angles: usize,
    factors: Option<&[f32]>,
) -> Vec<u8> {
    let half = hd / 2;
    let angles = angles.min(half);
    let mut out = vec![0f32; max_seq * hd];
    for pos in 0..max_seq {
        let base = pos * hd;
        for i in 0..half {
            if i < angles {
                let mut freq = 1.0 / theta.powf((2 * i) as f64 / hd as f64) as f32;
                if let Some(f) = factors {
                    freq /= f.get(i).copied().unwrap_or(1.0).max(1e-9);
                }
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

fn dump_vec_local(ctx: &GpuContext, b: &Buffer, n: usize) -> Vec<f32> {
    ctx.read_u16(b, n).iter().map(|&x| f16_to_f32(x)).collect()
}

impl MtpDraft {
    /// `kv_map_swa` / `kv_map_full`: base-model KV array indices whose cached
    /// K/V the draft's SWA / full attention layers read.
    pub fn load(
        eng: Eng,
        ctx: &GpuContext,
        path: &std::path::Path,
        cfg: &GemmaConfig,
        max_seq: usize,
        kv_map_swa: usize,
        kv_map_full: usize,
    ) -> Result<Self> {
        let g = Gguf::open(path);
        let f16b = |name: &str| -> Result<Buffer> {
            let t = g
                .tensor(name)
                .ok_or_else(|| anyhow!("mtp: missing tensor {name}"))?;
            if t.ggml_type != 1 {
                return Err(anyhow!("mtp: {name} is not F16 (type {})", t.ggml_type));
            }
            let raw = g.tensor_raw(name);
            let n = t.num_elements() * 2; // F16
            if raw.len() < n {
                return Err(anyhow!("mtp: {name} short read"));
            }
            Ok(ctx.buffer_bytes(&raw[..n]))
        };
        // Eng RMSNorm kernels load F16 weights (same as the base model).
        let rms_w = |name: &str| -> Result<Buffer> {
            if g.tensor(name).is_none() {
                return Err(anyhow!("mtp: missing tensor {name}"));
            }
            Ok(ctx.buffer_bytes(&g.dequant_to_f16_bytes(name)))
        };
        let f32v = |name: &str| -> Result<Vec<f32>> {
            let t = g
                .tensor(name)
                .ok_or_else(|| anyhow!("mtp: missing tensor {name}"))?;
            if t.ggml_type != 0 {
                return Err(anyhow!("mtp: {name} is not F32"));
            }
            let raw = g.tensor_raw(name);
            let n = t.num_elements() * 4; // F32
            if raw.len() < n {
                return Err(anyhow!("mtp: {name} short read"));
            }
            Ok(raw[..n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        let n_layers = g
            .get_u32("gemma4-assistant.block_count")
            .or_else(|| g.get_u32("gemma4-assistant.nextn_predict_layers"))
            .unwrap_or(4) as usize;
        let hh = g
            .get_u32("gemma4-assistant.embedding_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                *g.tensor("output_norm.weight")
                    .and_then(|t| t.dims.first())
                    .unwrap_or(&256) as usize
            });
        let vocab = g
            .get_u32("gemma4-assistant.vocab_size")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                *g.tensor("token_embd.weight")
                    .and_then(|t| t.dims.get(1))
                    .unwrap_or(&262144) as usize
            });
        let hidden_backbone = g
            .get_u32("gemma4-assistant.embedding_length_out")
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden);
        let full_theta = g
            .get_f32("gemma4-assistant.rope.freq_base")
            .unwrap_or(cfg.rope_theta_full) as f64;
        let swa_theta = g
            .get_f32("gemma4-assistant.rope.freq_base_swa")
            .unwrap_or(cfg.rope_theta_swa) as f64;
        let hd_full = g
            .get_u32("gemma4-assistant.attention.key_length")
            .unwrap_or(cfg.head_dim_full as u32) as usize;
        let hd_swa = g
            .get_u32("gemma4-assistant.attention.key_length_swa")
            .unwrap_or(cfg.head_dim_swa as u32) as usize;
        let sliding_window = g
            .get_u32("gemma4-assistant.attention.sliding_window")
            .unwrap_or(cfg.sliding_window as u32) as usize;
        let rms_eps = g
            .get_f32("gemma4-assistant.attention.layer_norm_rms_epsilon")
            .unwrap_or(cfg.rms_eps);
        let softcap = g
            .get_f32("gemma4-assistant.final_logit_softcapping")
            .unwrap_or(cfg.softcap);
        // GGUF `rope.dimension_count` is the RoPE *op* width (== head_dim), not
        // the rotated width. Full layers use p-RoPE 0.25 (same as the target).
        let swa_pattern: Vec<bool> = g
            .get_arr_bool("gemma4-assistant.attention.sliding_window_pattern")
            .map(|p| p.to_vec())
            .unwrap_or_else(|| (0..n_layers).map(|i| (i % 4) != 3).collect());
        if swa_pattern.len() != n_layers {
            return Err(anyhow!(
                "mtp: sliding_window_pattern length {} != n_layers {}",
                swa_pattern.len(),
                n_layers
            ));
        }

        let tok_embd = f16b("token_embd.weight")?;
        let pre_proj = f16b("nextn.pre_projection.weight")?;
        let post_proj = f16b("nextn.post_projection.weight")?;
        let output_norm = rms_w("output_norm.weight")?;
        let rope_factors = f32v("rope_freqs.weight").unwrap_or_default();

        let mut layers = Vec::new();
        let mut max_q = 0usize;
        let mut max_ffn = 0usize;
        let mut max_nh = 1usize;
        for li in 0..n_layers {
            let pfx = format!("blk.{li}.");
            if g.tensor(&format!("{pfx}attn_q.weight")).is_none() {
                break;
            }
            let q_dims = g
                .tensor(&format!("{pfx}attn_q.weight"))
                .ok_or_else(|| anyhow!("mtp: missing {pfx}attn_q"))?
                .dims
                .clone();
            let q_cols = *q_dims.get(1).unwrap_or(&0) as usize;
            let hd = g
                .tensor(&format!("{pfx}attn_q_norm.weight"))
                .and_then(|t| t.dims.first().copied())
                .unwrap_or(q_cols as u64) as usize;
            if hd == 0 || q_cols % hd != 0 {
                return Err(anyhow!("mtp: bad attn_q dims {q_dims:?}"));
            }
            let nh = q_cols / hd;
            let ffn_inter = g
                .tensor(&format!("{pfx}ffn_down.weight"))
                .and_then(|t| t.dims.first().copied())
                .unwrap_or(2048) as usize;
            let is_full = !swa_pattern.get(li).copied().unwrap_or(true);
            let scale_t =
                f32v(&format!("{pfx}layer_output_scale.weight")).unwrap_or_else(|_| vec![1.0]);
            max_q = max_q.max(nh * hd);
            max_ffn = max_ffn.max(ffn_inter);
            max_nh = max_nh.max(nh);
            layers.push(DraftLayer {
                attn_norm: rms_w(&format!("{pfx}attn_norm.weight"))?,
                q_w: f16b(&format!("{pfx}attn_q.weight"))?,
                q_norm: rms_w(&format!("{pfx}attn_q_norm.weight"))?,
                out_w: f16b(&format!("{pfx}attn_output.weight"))?,
                post_attn: rms_w(&format!("{pfx}post_attention_norm.weight"))?,
                ffn_norm: rms_w(&format!("{pfx}ffn_norm.weight"))?,
                gate: f16b(&format!("{pfx}ffn_gate.weight"))?,
                up: f16b(&format!("{pfx}ffn_up.weight"))?,
                down: f16b(&format!("{pfx}ffn_down.weight"))?,
                post_ffw: rms_w(&format!("{pfx}post_ffw_norm.weight"))?,
                out_scale: {
                    let v = *scale_t.first().unwrap_or(&1.0);
                    eprintln!("[mtp-load] {pfx}out_scale = {v:.6e} is_full={is_full} nh={nh} hd={hd} ffn={ffn_inter}");
                    v
                },
                is_full,
                nh,
                hd,
                ffn_inter,
                kv_map: if is_full { kv_map_full } else { kv_map_swa },
            });
        }
        if layers.is_empty() {
            return Err(anyhow!("mtp: no draft layers found"));
        }

        let full_angles = ((hd_full as f32) * cfg.partial_rotary) as usize / 2;
        let swa_angles = hd_swa / 2;
        let nwg = Eng::SDPA_MWG_NWG;
        eprintln!(
            "[mtp] layers={} hh={} hb={} n_kv={} swa_pattern={:?} rope_full_angles={} rope_freqs={}",
            layers.len(),
            hh,
            hidden_backbone,
            cfg.n_kv,
            swa_pattern,
            full_angles,
            rope_factors.len()
        );

        Ok(Self {
            eng,
            layers,
            tok_embd,
            pre_proj,
            post_proj,
            output_norm,
            vocab,
            hb: hidden_backbone,
            hh,
            softcap,
            rms_eps,
            sliding_window,
            base_n_kv: cfg.n_kv.max(1),
            rope_swa: ctx.buffer_bytes(&rope_table(
                max_seq,
                hd_swa,
                swa_theta,
                swa_angles,
                None,
            )),
            rope_full: ctx.buffer_bytes(&rope_table(
                max_seq,
                hd_full,
                full_theta,
                full_angles,
                Some(&rope_factors),
            )),
            meta_swa: ctx.buffer_f32(&[0.0; 3]),
            meta_full: ctx.buffer_f32(&[0.0; 3]),
            tmp_swa: ctx.buffer_empty_f32(max_nh * nwg * (hd_swa + 2)),
            tmp_full: ctx.buffer_empty_f32(max_nh * nwg * (hd_full + 2)),
            argmax: ctx.buffer_empty_f32(1),
            logits: ctx.buffer_empty_f16(vocab),
            hnext16: ctx.buffer_empty_f16(hidden_backbone),
            xh: ctx.buffer_empty_f16(hidden_backbone * 2),
            h: ctx.buffer_empty_f16(hh),
            a: ctx.buffer_empty_f16(hh),
            q: ctx.buffer_empty_f16(max_q.max(1)),
            att: ctx.buffer_empty_f16(max_q.max(1)),
            cur: ctx.buffer_empty_f16(hh),
            aout: ctx.buffer_empty_f16(hh),
            ffin: ctx.buffer_empty_f16(hh),
            gate: ctx.buffer_empty_f16(max_ffn.max(1)),
            up: ctx.buffer_empty_f16(max_ffn.max(1)),
            inter: ctx.buffer_empty_f16(max_ffn.max(1)),
            fdown: ctx.buffer_empty_f16(hh),
            fnorm: ctx.buffer_empty_f16(hh),
            att_out: ctx.buffer_empty_f16(max_q.max(1)),
            cache: HashMap::new(),
        })
    }

    fn tiny(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        build: impl FnOnce(&mut Graph) -> Result<ksearch_ir::TensorId>,
    ) -> Result<()> {
        if self.cache.contains_key(key) {
            return Ok(());
        }
        let mut g = Graph::new();
        let o = build(&mut g)?;
        let src = ctx.lower(&g, o)?;
        let pipe = ctx.compile(&src)?;
        if std::env::var("MTP_DEBUG").is_ok() && key.starts_with("scale") {
            eprintln!("=== tiny kernel {key} source:\n{}", src.source);
        }
        self.cache.insert(key.to_string(), (src, pipe));
        Ok(())
    }

    fn run_tiny(
        &self,
        ctx: &GpuContext,
        key: &str,
        ins: &[&Buffer],
        out: &Buffer,
    ) -> Result<()> {
        let (src, pipe) = self
            .cache
            .get(key)
            .ok_or_else(|| anyhow!("tiny kernel {key} missing"))?
            .clone();
        ctx.encode_multi(&pipe, &src, ins, &[out], 256)
    }

    pub fn xh_buf(&self) -> &Buffer {
        &self.xh
    }

    pub fn hnext_buf(&self) -> &Buffer {
        &self.hnext16
    }

    pub fn argmax_buf(&self) -> &Buffer {
        &self.argmax
    }

    /// One draft step.
    ///
    /// `emb_scaled` — √hb-scaled base embedding of the *just-sampled* token
    /// (len hb). `h_seed` — target post-final-norm hidden that produced that
    /// token (step 0) or the previous draft `h_next` (later steps). Attention
    /// reads the base KV cache at the mapped layer; `kv_seq` is the number of
    /// tokens already written (the sampled token is not in KV yet) and
    /// `pos` is the RoPE index of that sampled token (= `kv_seq`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
        &mut self,
        ctx: &GpuContext,
        emb_scaled: &[f32],
        h_seed: &[f32],
        pos: usize,
        seq_len: usize,
        capacity: usize,
        base_k: &[Buffer],
        base_v: &[Buffer],
    ) -> Result<(u32, Vec<f32>)> {
        if emb_scaled.len() != self.hb || h_seed.len() != self.hb {
            return Err(anyhow!(
                "mtp dims: emb={} seed={} want {}",
                emb_scaled.len(),
                h_seed.len(),
                self.hb
            ));
        }
        let mut xh16: Vec<u16> = Vec::with_capacity(self.hb * 2);
        for &v in emb_scaled {
            xh16.push(f32_to_f16(v));
        }
        for &v in h_seed {
            xh16.push(f32_to_f16(v));
        }
        ctx.write_u16s(&self.xh, &xh16);
        self.forward_from_xh(ctx, pos, seq_len, capacity, base_k, base_v)?;
        ctx.synchronize()?;
        let best = ctx.read_f32(&self.argmax, 1)[0] as u32;
        let h_next: Vec<f32> = ctx
            .read_u16(&self.hnext16, self.hb)
            .iter()
            .map(|&v| f16_to_f32(v))
            .collect();
        Ok((best, h_next))
    }

    /// Draft step with `xh` already filled as concat(embed*√hb, h_seed).
    /// Does **not** synchronize; caller reads `argmax_buf` / `hnext_buf`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_from_xh(
        &mut self,
        ctx: &GpuContext,
        pos: usize,
        seq_len: usize,
        capacity: usize,
        base_k: &[Buffer],
        base_v: &[Buffer],
    ) -> Result<()> {
        // Serialize producer/consumer kernels. Draft is not the hot path;
        // in-place scale(h→h) in particular races without a barrier.
        let barrier = |ctx: &GpuContext| ctx.encoder_barrier();

        // cur = pre_proj · xh
        self.eng.matvec_wd(
            ctx,
            self.hh,
            self.hb * 2,
            DType::F16,
            &self.pre_proj,
            &self.xh,
            &self.h,
        )?;
        barrier(ctx);

        let dbg = std::env::var("MTP_DEBUG").is_ok();
        let dump_all = std::env::var("MTP_DUMP_ALL").is_ok();
        let mut dump_seq: Vec<(String, Vec<f32>)> = Vec::new();
        let stat = |ctx: &GpuContext, b: &Buffer, n: usize, tag: &str| {
            if !dbg {
                return;
            }
            let v: Vec<f32> = ctx
                .read_u16(b, n)
                .iter()
                .map(|&x| f16_to_f32(x))
                .collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[mtp-stat] {tag}: mean={mean:.4} std={std:.4} max={mx:.3}");
        };
        stat(ctx, &self.xh, self.hb * 2, "xh");
        stat(ctx, &self.h, self.hh, "h_after_preproj");
        if dbg {
            let dump = |ctx: &GpuContext, b: &Buffer, n: usize| -> Vec<f32> {
                ctx.read_u16(b, n).iter().map(|&x| f16_to_f32(x)).collect()
            };
            let _ = std::fs::write(
                "/tmp/opencode/xh.json",
                format!("{:?}", dump(ctx, &self.xh, self.hb * 2)),
            );
            let _ = std::fs::write(
                "/tmp/opencode/h_preproj.json",
                format!("{:?}", dump(ctx, &self.h, self.hh)),
            );
        }

        let hh = self.hh;
        let eps = self.rms_eps;
        let sliding_window = self.sliding_window;
        let layers = self.layers.clone();
        for (lidx, layer) in layers.iter().enumerate() {
            self.eng.rmsnorm(ctx, hh, eps, &self.h, &layer.attn_norm, &self.a)?;

            let qelems = layer.nh * layer.hd;
            self.eng.matvec_wd(ctx, qelems, hh, DType::F16, &layer.q_w, &self.a, &self.q)?;

            // fused per-head norm + rope at pos
            let (rope_buf, rope_off) = if layer.is_full {
                (&self.rope_full, pos * layer.hd)
            } else {
                (&self.rope_swa, pos * layer.hd)
            };
            self.eng.rmsnorm_per_head_rope_off(
                ctx,
                layer.nh,
                layer.hd,
                eps,
                &self.q,
                &layer.q_norm,
                rope_buf,
                rope_off,
                &self.att,
                0,
            )?;
            barrier(ctx);

            // attention over base KV (Q4_0), GQA-style; no KV write.
            // KSEARCH_MTP_NOATTN=1 zeroes it (reference DRAFT_NO_ATTN ablation).
            let no_attn = std::env::var("KSEARCH_MTP_NOATTN").is_ok();
            if no_attn {
                let zkey = format!("zero_q{qelems}");
                self.tiny(ctx, &zkey, |g| {
                    let ix = g.input(Shape(vec![qelems]), DType::F16);
                    g.scale_const(ix, 0.0).map_err(|e| anyhow!("{e}"))
                })
                .context("zero graph")?;
                self.run_tiny(ctx, &zkey, &[&self.att_out], &self.att_out)?;
            } else {
                let kv_idx = layer.kv_map;
                if kv_idx >= base_k.len() {
                    return Err(anyhow!(
                        "mtp: kv_map {kv_idx} out of range ({})",
                        base_k.len()
                    ));
                }
                let (meta, tmp, tlen, start) = if layer.is_full {
                    (
                        &self.meta_full,
                        &self.tmp_full,
                        seq_len.max(1),
                        0usize,
                    )
                } else {
                    let tlen = seq_len.min(sliding_window).max(1);
                    let start = seq_len.saturating_sub(sliding_window);
                    (&self.meta_swa, &self.tmp_swa, tlen, start)
                };
                ctx.write_buffer(meta, &[tlen as f32, start as f32, sliding_window as f32]);
                let cap = capacity.max(tlen);
                // Draft Q attends GQA-style into the target's KV. Clamp n_kv to
                // the draft's query-head count (same as metal-llm-server).
                let n_kv = self.base_n_kv.max(1).min(layer.nh);
                self.eng.sdpa_hybrid_kv(
                    ctx,
                    layer.nh,
                    n_kv,
                    layer.hd,
                    cap,
                    tlen as u32,
                    DType::Q40,
                    &self.att,
                    &base_k[kv_idx],
                    &base_v[kv_idx],
                    meta,
                    tmp,
                    &self.att_out,
                )?;
                barrier(ctx);
            }

            self.eng
                .matvec_wd(ctx, hh, qelems, DType::F16, &layer.out_w, &self.att_out, &self.a)?;
            barrier(ctx);
            if dump_all {
                dump_seq.push((format!("L{}.a", lidx), dump_vec_local(ctx, &self.a, hh)));
                dump_seq.push((
                    format!("L{}.att", lidx),
                    dump_vec_local(ctx, &self.att, qelems),
                ));
            }
            self.eng.rmsnorm(ctx, hh, eps, &self.a, &layer.post_attn, &self.cur)?;

            // aout = cur + h ; h = scale(down-path + aout)
            self.tiny(ctx, "add_hh", |g| {
                let ia = g.input(Shape(vec![hh]), DType::F16);
                let ib = g.input(Shape(vec![hh]), DType::F16);
                g.add(ia, ib).map_err(|e| anyhow!("{e}"))
            })
            .context("add graph")?;
            self.run_tiny(ctx, "add_hh", &[&self.cur, &self.h], &self.aout)?;

            let fi = layer.ffn_inter;
            self.eng.rmsnorm(ctx, hh, eps, &self.aout, &layer.ffn_norm, &self.ffin)?;
            self.eng.matvec_wd(ctx, fi, hh, DType::F16, &layer.gate, &self.ffin, &self.gate)?;
            self.eng.matvec_wd(ctx, fi, hh, DType::F16, &layer.up, &self.ffin, &self.up)?;
            self.eng.gelu_mul(ctx, fi, &self.gate, &self.up, &self.inter)?;
            self.eng.matvec_wd(ctx, hh, fi, DType::F16, &layer.down, &self.inter, &self.fdown)?;
            self.eng.rmsnorm(ctx, hh, eps, &self.fdown, &layer.post_ffw, &self.cur)?;
            self.run_tiny(ctx, "add_hh", &[&self.cur, &self.aout], &self.h)?;
            barrier(ctx);

            let skey = format!("scale_{hh}_{:08x}", layer.out_scale.to_bits());
            let sc = layer.out_scale;
            self.tiny(ctx, &skey, move |g| {
                let ix = g.input(Shape(vec![hh]), DType::F16);
                g.scale_const(ix, sc).map_err(|e| anyhow!("{e}"))
            })
            .context("scale graph")?;
            self.run_tiny(ctx, &skey, &[&self.h], &self.h)?;
            barrier(ctx);
            if dump_all {
                dump_seq.push((format!("L{}.h", lidx), dump_vec_local(ctx, &self.h, hh)));
            }
        }

        if dbg {
            stat(ctx, &self.h, self.hh, "h_after_layers");
        }
        self.eng.rmsnorm(ctx, hh, eps, &self.h, &self.output_norm, &self.fnorm)?;

        // logits via the draft's own tied table + argmax; next seed via post_proj
        self.eng.matvec_wd(
            ctx,
            self.vocab,
            hh,
            DType::F16,
            &self.tok_embd,
            &self.fnorm,
            &self.logits,
        )?;
        self.eng.matvec_wd(
            ctx,
            self.hb,
            hh,
            DType::F16,
            &self.post_proj,
            &self.fnorm,
            &self.hnext16,
        )?;
        self.eng.softcap_argmax(ctx, self.vocab, self.softcap, &self.logits, &self.argmax)?;
        if dbg {
            ctx.synchronize()?;
            let lg: Vec<f32> = ctx
                .read_u16(&self.logits, self.vocab)
                .iter()
                .map(|&v| f16_to_f32(v))
                .collect();
            let mut idx: Vec<(usize, f32)> = lg.iter().cloned().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let lmax = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[draft] argmax={} lmax={lmax:.2} top5={:?}",
                ctx.read_f32(&self.argmax, 1)[0] as u32,
                &idx[..5]
                    .iter()
                    .map(|(i, v)| format!("{i}:{v:.1}"))
                    .collect::<Vec<_>>()
            );
        }

        if dump_all {
            let _ = std::fs::write(
                "/tmp/opencode/seq.json",
                serde_json::to_string(&dump_seq).unwrap(),
            );
        }
        Ok(())
    }
    pub fn debug_logits(&self) -> &Buffer {
        &self.logits
    }
}

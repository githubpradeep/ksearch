//! Eng: build Graph (primitives + sugar) → lower_to_metal → Metal.
//! No direct KirBody / catalog Op shortcuts.

use anyhow::Result;
use ksearch_codegen::{lower_to_metal_chip, LaunchHint, MetalKernelSource};
use ksearch_ir::{DType, Graph, Shape};
use ksearch_metal::MetalContext;
use metal::*;
use std::collections::HashMap;

pub struct Eng {
    cache: HashMap<String, (MetalKernelSource, ComputePipelineState)>,
    /// Scratch for sequenced fuses (matvec → rmsnorm) when out aliases residual.
    fuse_scratch: HashMap<usize, Buffer>,
}

impl Eng {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            fuse_scratch: HashMap::new(),
        }
    }

    fn scratch_f16(&mut self, ctx: &MetalContext, n: usize) -> Buffer {
        self.fuse_scratch
            .entry(n)
            .or_insert_with(|| ctx.buffer_empty_f16(n))
            .clone()
    }

    fn ensure(
        &mut self,
        ctx: &MetalContext,
        key: &str,
        src: MetalKernelSource,
    ) -> Result<()> {
        if !self.cache.contains_key(key) {
            let p = ctx.compile(&src)?;
            self.cache.insert(key.to_string(), (src, p));
        }
        Ok(())
    }

    fn tg_for(src: &MetalKernelSource) -> u64 {
        match src.launch {
            LaunchHint::Elementwise { .. } | LaunchHint::Rows { .. } => 256,
            LaunchHint::RowsParallel { tg, .. } => tg,
            LaunchHint::RowsParallelSg { nsg, .. } => nsg * 32,
            LaunchHint::RowsParallel2D { tg, .. } => tg,
            LaunchHint::MulMm { tw, nsg, .. } => tw * nsg,
        }
    }

    fn run(
        &self,
        ctx: &MetalContext,
        key: &str,
        inputs: &[&Buffer],
        output: &Buffer,
    ) -> Result<()> {
        self.run_multi(ctx, key, inputs, &[output])
    }

    fn run_multi(
        &self,
        ctx: &MetalContext,
        key: &str,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
    ) -> Result<()> {
        let (src, pipe) = self.cache.get(key).expect("ensure first");
        let tg = Self::tg_for(src);
        ctx.encode_multi(pipe, src, inputs, outputs, tg)?;
        Ok(())
    }

    fn run_offsets(
        &self,
        ctx: &MetalContext,
        key: &str,
        inputs: &[&Buffer],
        input_byte_offsets: &[u64],
        output: &Buffer,
        output_byte_offset: u64,
    ) -> Result<()> {
        let (src, pipe) = self.cache.get(key).expect("ensure first");
        let tg = Self::tg_for(src);
        ctx.encode_offsets(
            pipe,
            src,
            inputs,
            input_byte_offsets,
            output,
            output_byte_offset,
            tg,
        )?;
        Ok(())
    }

    pub fn matvec(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.matvec_wd(ctx, rows, cols, DType::F16, a, x, y)
    }

    /// Matvec with explicit weight dtype (`F16` or `Q4K`; activations always F16).
    pub fn matvec_wd(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        weight_dtype: DType,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        let key = format!("mv_{tag}_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), weight_dtype);
            let v = g.input(Shape(vec![cols]), DType::F16);
            let out = g.matvec_prim(w, v)?;
            let src = lower_to_metal_chip(&g, out, &ctx.device_name()).map_err(|e| {
                anyhow::anyhow!("matvec_wd {tag} {rows}x{cols}: {e}")
            })?;
            self.ensure(ctx, &key, src)?;
        }
        self.run(ctx, &key, &[a, x], y)
    }

    /// Fused Q/K/V matvecs (one launch; LOCAL-stages `x`; three outputs).
    pub fn matvec_qkv(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        x: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        self.matvec_qkv_wd(
            ctx,
            q_rows,
            kv_rows,
            cols,
            DType::F16,
            wq,
            wk,
            wv,
            x,
            q,
            k,
            v,
        )
    }

    pub fn matvec_qkv_wd(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        weight_dtype: DType,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        x: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        self.matvec_qkv_wds(
            ctx,
            q_rows,
            kv_rows,
            cols,
            weight_dtype,
            weight_dtype,
            weight_dtype,
            wq,
            wk,
            wv,
            x,
            q,
            k,
            v,
        )
    }

    /// Fused Q/K/V with per-buffer weight dtypes (e.g. Q4K/Q4K/Q6K).
    pub fn matvec_qkv_wds(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        wq_dtype: DType,
        wk_dtype: DType,
        wv_dtype: DType,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        x: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        let tag = format!(
            "{}_{}_{}",
            weight_cache_tag(wq_dtype),
            weight_cache_tag(wk_dtype),
            weight_cache_tag(wv_dtype)
        );
        let key = format!("mv_qkv_{tag}_{q_rows}x{kv_rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wq_i = g.input(Shape(vec![q_rows, cols]), wq_dtype);
            let wk_i = g.input(Shape(vec![kv_rows, cols]), wk_dtype);
            let wv_i = g.input(Shape(vec![kv_rows, cols]), wv_dtype);
            let x_i = g.input(Shape(vec![cols]), DType::F16);
            let out = g.matvec_qkv(wq_i, wk_i, wv_i, x_i)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_multi(ctx, &key, &[wq, wk, wv, x], &[q, k, v])
    }

    /// RMSNorm(x,w_norm) fused into dense matvec (LOCAL-stages `x_hat`).
    pub fn rmsnorm_matvec(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        x: &Buffer,
        w_norm: &Buffer,
        w_mat: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_matvec_wd(ctx, rows, cols, eps, DType::F16, x, w_norm, w_mat, y)
    }

    pub fn rmsnorm_matvec_wd(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        weight_dtype: DType,
        x: &Buffer,
        w_norm: &Buffer,
        w_mat: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        let key = format!("rms_mv_{tag}_{rows}x{cols}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wm = g.input(Shape(vec![rows, cols]), weight_dtype);
            let xi = g.input(Shape(vec![cols]), DType::F16);
            let wn = g.input(Shape(vec![cols]), DType::F16);
            let out = g.rmsnorm_matvec(wm, xi, wn, eps)?;
            let src = lower_to_metal_chip(&g, out, &ctx.device_name()).map_err(|e| {
                anyhow::anyhow!("rmsnorm_matvec_wd {tag} {rows}x{cols}: {e}")
            })?;
            self.ensure(ctx, &key, src)?;
        }
        self.run(ctx, &key, &[w_mat, x, w_norm], y)
    }

    /// RMSNorm into LOCAL `x_hat`, then fused Q/K/V matvecs (3 outputs).
    pub fn rmsnorm_matvec_qkv(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        eps: f32,
        x: &Buffer,
        w_norm: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_matvec_qkv_wd(
            ctx,
            q_rows,
            kv_rows,
            cols,
            eps,
            DType::F16,
            x,
            w_norm,
            wq,
            wk,
            wv,
            q,
            k,
            v,
        )
    }

    pub fn rmsnorm_matvec_qkv_wd(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        eps: f32,
        weight_dtype: DType,
        x: &Buffer,
        w_norm: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_matvec_qkv_wds(
            ctx,
            q_rows,
            kv_rows,
            cols,
            eps,
            weight_dtype,
            weight_dtype,
            weight_dtype,
            x,
            w_norm,
            wq,
            wk,
            wv,
            q,
            k,
            v,
        )
    }

    /// RMSNorm + fused Q/K/V with per-buffer weight dtypes (mixed Q4K/Q6K ok).
    pub fn rmsnorm_matvec_qkv_wds(
        &mut self,
        ctx: &MetalContext,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        eps: f32,
        wq_dtype: DType,
        wk_dtype: DType,
        wv_dtype: DType,
        x: &Buffer,
        w_norm: &Buffer,
        wq: &Buffer,
        wk: &Buffer,
        wv: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
    ) -> Result<()> {
        let tag = format!(
            "{}_{}_{}",
            weight_cache_tag(wq_dtype),
            weight_cache_tag(wk_dtype),
            weight_cache_tag(wv_dtype)
        );
        let key = format!("rms_mv_qkv_{tag}_{q_rows}x{kv_rows}x{cols}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wq_i = g.input(Shape(vec![q_rows, cols]), wq_dtype);
            let wk_i = g.input(Shape(vec![kv_rows, cols]), wk_dtype);
            let wv_i = g.input(Shape(vec![kv_rows, cols]), wv_dtype);
            let x_i = g.input(Shape(vec![cols]), DType::F16);
            let wn = g.input(Shape(vec![cols]), DType::F16);
            let out = g.rmsnorm_matvec_qkv(wq_i, wk_i, wv_i, x_i, wn, eps)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_multi(ctx, &key, &[wq, wk, wv, x, w_norm], &[q, k, v])
    }

    /// Thin Q4_K weight matvec (activations F16); Graph uses `DType::Q4K` → generic Load expand.
    pub fn matvec_q4k_prim(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.matvec_wd(ctx, rows, cols, DType::Q4K, a, x, y)
    }

    pub fn matvec_q4k_prim_at(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        x_off_elems: usize,
        y: &Buffer,
        y_off_elems: usize,
    ) -> Result<()> {
        let tag = weight_cache_tag(DType::Q4K);
        let key = format!("mv_{tag}_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F16);
            let out = g.matvec_prim(w, v)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let xb = (x_off_elems * DType::F16.size_bytes()) as u64;
        let yb = (y_off_elems * DType::F16.size_bytes()) as u64;
        self.run_offsets(ctx, &key, &[a, x], &[0, xb], y, yb)
    }

    pub fn rmsnorm(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_at(ctx, n, eps, x, 0, w, y, 0)
    }

    pub fn rmsnorm_at(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        x: &Buffer,
        x_off_elems: usize,
        w: &Buffer,
        y: &Buffer,
        y_off_elems: usize,
    ) -> Result<()> {
        let key = format!("rms_f16_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let wi = g.input(Shape(vec![n]), DType::F16);
            let out = g.rmsnorm_expand(xi, wi, eps)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let xb = (x_off_elems * DType::F16.size_bytes()) as u64;
        let yb = (y_off_elems * DType::F16.size_bytes()) as u64;
        self.run_offsets(ctx, &key, &[x, w], &[xb, 0], y, yb)
    }

    pub fn rmsnorm_add(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        residual: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("rms_f16_add_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let wi = g.input(Shape(vec![n]), DType::F16);
            let ri = g.input(Shape(vec![n]), DType::F16);
            let out = g.rmsnorm_add_expand(xi, wi, ri, eps)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x, w, residual], y)
    }

    pub fn rmsnorm_add_scale(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        scale: f32,
        x: &Buffer,
        w: &Buffer,
        residual: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("rms_f16_add_sc_{n}_{}_{}", eps.to_bits(), scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let wi = g.input(Shape(vec![n]), DType::F16);
            let ri = g.input(Shape(vec![n]), DType::F16);
            let out = g.rmsnorm_add_scale_expand(xi, wi, ri, eps, scale)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x, w, residual], y)
    }

    /// Fused post-attn residual RMS + FFN RMS: `out_x = residual + rms(y)*w_post`,
    /// `out_x2 = rms(out_x)*w_ffn` (one launch, two outputs).
    pub fn rmsnorm_add_then_rmsnorm(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        y: &Buffer,
        w_post: &Buffer,
        residual: &Buffer,
        w_ffn: &Buffer,
        out_x: &Buffer,
        out_x2: &Buffer,
    ) -> Result<()> {
        let key = format!("rms_f16_add_then_rms_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let yi = g.input(Shape(vec![n]), DType::F16);
            let wp = g.input(Shape(vec![n]), DType::F16);
            let ri = g.input(Shape(vec![n]), DType::F16);
            let wf = g.input(Shape(vec![n]), DType::F16);
            let out = g.rmsnorm_add_then_rmsnorm(yi, wp, ri, wf, eps)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_multi(ctx, &key, &[y, w_post, residual, w_ffn], &[out_x, out_x2])
    }

    pub fn rmsnorm_per_head(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("rms_f16_ph_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let wi = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rmsnorm_per_head(xi, wi, n_heads, hd, eps, true)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x, w], y)
    }

    /// Fused per-head RMSNorm + RoPE (one launch).
    pub fn rmsnorm_per_head_rope(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        cos_sin: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_per_head_rope_off(ctx, n_heads, hd, eps, x, w, cos_sin, 0, y, 0)
    }

    /// Like [`rmsnorm_per_head_rope`] with element offsets into `cos_sin` / `y` (F16 elems).
    pub fn rmsnorm_per_head_rope_off(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        cos_sin: &Buffer,
        cos_sin_off_elems: usize,
        y: &Buffer,
        y_off_elems: usize,
    ) -> Result<()> {
        let key = format!("rms_f16_ph_rope_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let wi = g.input(Shape(vec![hd]), DType::F16);
            let ci = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rmsnorm_per_head_rope(xi, wi, ci, n_heads, hd, eps, true)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let b = DType::F16.size_bytes() as u64;
        self.run_offsets(
            ctx,
            &key,
            &[x, w, cos_sin],
            &[0, 0, cos_sin_off_elems as u64 * b],
            y,
            y_off_elems as u64 * b,
        )
    }

    pub fn rmsnorm_noweight(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_noweight_off(ctx, n_heads, hd, eps, x, y, 0)
    }

    pub fn rmsnorm_noweight_off(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        y: &Buffer,
        y_off_elems: usize,
    ) -> Result<()> {
        let key = format!("rms_f16_nw_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let wi = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rmsnorm_per_head(xi, wi, n_heads, hd, eps, false)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let b = DType::F16.size_bytes() as u64;
        self.run_offsets(
            ctx,
            &key,
            &[x, x],
            &[0, 0],
            y,
            y_off_elems as u64 * b,
        )
    }

    pub fn add(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("add_f16_{n}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let a = g.input(Shape(vec![n]), DType::F16);
            let b = g.input(Shape(vec![n]), DType::F16);
            let out = g.add(a, b)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[a, b], y)
    }

    pub fn gelu_mul(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.gelu_mul_at(ctx, n, gate, up, 0, y)
    }

    pub fn gelu_mul_at(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        gate: &Buffer,
        up: &Buffer,
        up_off: usize,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("gelu_f16_{n}_{up_off}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let gate_i = g.input(Shape(vec![n]), DType::F16);
            let up_i = g.input(Shape(vec![up_off + n]), DType::F16);
            let out = g.gelu_mul_at(gate_i, up_i, up_off)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[gate, up], y)
    }

    /// PLE gate: fused matvec + `gelu(acc)*ctx[i]` (one launch). Use `ctx` byte offset for layer slice.
    pub fn matvec_gelu_mul_at(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        weight_dtype: DType,
        w: &Buffer,
        x: &Buffer,
        ctx_buf: &Buffer,
        ctx_off_elems: usize,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        // ctx_off is applied via Metal buffer offset — one pipeline for all layers.
        let key = format!("mv_gelu_mul_{tag}_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wi = g.input(Shape(vec![rows, cols]), weight_dtype);
            let xi = g.input(Shape(vec![cols]), DType::F16);
            let ci = g.input(Shape(vec![rows]), DType::F16);
            let out = g.matvec_gelu_mul(wi, xi, ci, 0)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let b = DType::F16.size_bytes() as u64;
        self.run_offsets(
            ctx,
            &key,
            &[w, x, ctx_buf],
            &[0, 0, ctx_off_elems as u64 * b],
            y,
            0,
        )
    }

    /// PLE: fused gate gelu*ctx + proj + rmsnorm_add_scale (one launch).
    pub fn matvec_gelu_mul_proj_rms_add_scale_at(
        &mut self,
        ctx: &MetalContext,
        gate_rows: usize,
        cols: usize,
        proj_rows: usize,
        weight_dtype: DType,
        w_gate: &Buffer,
        x: &Buffer,
        ctx_buf: &Buffer,
        ctx_off_elems: usize,
        w_proj: &Buffer,
        w_norm: &Buffer,
        residual: &Buffer,
        eps: f32,
        scale: f32,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        let key = format!(
            "mv_gelu_proj_rms_{tag}_{gate_rows}x{cols}_{proj_rows}_{}_{}",
            eps.to_bits(),
            scale.to_bits()
        );
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wg = g.input(Shape(vec![gate_rows, cols]), weight_dtype);
            let xi = g.input(Shape(vec![cols]), DType::F16);
            let ci = g.input(Shape(vec![gate_rows]), DType::F16);
            let wp = g.input(Shape(vec![proj_rows, gate_rows]), weight_dtype);
            let wn = g.input(Shape(vec![proj_rows]), DType::F16);
            let ri = g.input(Shape(vec![proj_rows]), DType::F16);
            let out = g.matvec_gelu_mul_proj_rms_add_scale(
                wg, xi, ci, 0, wp, wn, ri, eps, scale,
            )?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let b = DType::F16.size_bytes() as u64;
        self.run_offsets(
            ctx,
            &key,
            &[w_gate, x, ctx_buf, w_proj, w_norm, residual],
            &[0, 0, ctx_off_elems as u64 * b, 0, 0, 0],
            y,
            0,
        )
    }

    /// F16 short-K: matvec + rmsnorm_add in one Eng CALL (sequenced: fast matvec then rms).
    pub fn matvec_rmsnorm_add(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        weight_dtype: DType,
        w: &Buffer,
        x: &Buffer,
        w_norm: &Buffer,
        residual: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let scratch = self.scratch_f16(ctx, rows);
        self.matvec_wd(ctx, rows, cols, weight_dtype, w, x, &scratch)?;
        ctx.encoder_barrier();
        self.rmsnorm_add(ctx, rows, eps, &scratch, w_norm, residual, y)
    }

    /// PLE proj tail: one Eng CALL — Q4 matvec then rmsnorm_add_scale (sequenced dispatches).
    /// Graph FuseHint exists for schedule; runtime uses fast coop matvec + rms (single-TG LOCAL
    /// fuse loses to multi-TG coop on 1536×256).
    pub fn matvec_rmsnorm_add_scale(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        scale: f32,
        weight_dtype: DType,
        w: &Buffer,
        x: &Buffer,
        w_norm: &Buffer,
        residual: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let scratch = self.scratch_f16(ctx, rows);
        self.matvec_wd(ctx, rows, cols, weight_dtype, w, x, &scratch)?;
        ctx.encoder_barrier();
        self.rmsnorm_add_scale(ctx, rows, eps, scale, &scratch, w_norm, residual, y)
    }

    /// `y = scale * (a + b)` as one elementwise launch.
    pub fn add_scale(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        scale: f32,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("add_sc_f16_{n}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let ai = g.input(Shape(vec![n]), DType::F16);
            let bi = g.input(Shape(vec![n]), DType::F16);
            let s = g.add(ai, bi)?;
            let out = g.scale_const(s, scale)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[a, b], y)
    }

    /// Fused gate/up matvecs + GELU*mul (one launch; LOCAL-stages `x`).
    pub fn matvec_gate_up_gelu(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.matvec_gate_up_gelu_wd(ctx, rows, cols, DType::F16, gate, up, x, y)
    }

    pub fn matvec_gate_up_gelu_wd(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        weight_dtype: DType,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        let key = format!("mv_gate_up_gelu_{tag}_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wg = g.input(Shape(vec![rows, cols]), weight_dtype);
            let wu = g.input(Shape(vec![rows, cols]), weight_dtype);
            let v = g.input(Shape(vec![cols]), DType::F16);
            let out = g.matvec_gate_up_gelu(wg, wu, v)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[gate, up, x], y)
    }

    /// RMSNorm into LOCAL `x_hat`, then fused gate/up matvecs + GELU*mul.
    pub fn rmsnorm_matvec_gate_up_gelu(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        x: &Buffer,
        w_norm: &Buffer,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.rmsnorm_matvec_gate_up_gelu_wd(
            ctx,
            rows,
            cols,
            eps,
            DType::F16,
            x,
            w_norm,
            gate,
            up,
            y,
        )
    }

    pub fn rmsnorm_matvec_gate_up_gelu_wd(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        weight_dtype: DType,
        x: &Buffer,
        w_norm: &Buffer,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let tag = weight_cache_tag(weight_dtype);
        let key = format!("rms_mv_gate_up_gelu_{tag}_{rows}x{cols}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wg = g.input(Shape(vec![rows, cols]), weight_dtype);
            let wu = g.input(Shape(vec![rows, cols]), weight_dtype);
            let v = g.input(Shape(vec![cols]), DType::F16);
            let wn = g.input(Shape(vec![cols]), DType::F16);
            let out = g.rmsnorm_matvec_gate_up_gelu(wg, wu, v, wn, eps)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[gate, up, x, w_norm], y)
    }

    pub fn scale_const(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        scale: f32,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("sc_f16_{n}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let out = g.scale_const(xi, scale)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x], y)
    }

    pub fn rope(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        x: &Buffer,
        cos_sin: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("rope_f16_{n_heads}_{hd}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let ci = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rope(xi, ci, n_heads, hd)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x, cos_sin], y)
    }

    pub fn sdpa_naive(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        self.sdpa_naive_kv(ctx, n_q, hd, max_t, DType::F16, q, k, v, meta, out)
    }

    /// SDPA with K/V dtype `kv_dtype` (F16 or Q40 Load expand).
    pub fn sdpa_naive_kv(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        kv_dtype: DType,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("sdpa_f16_{n_q}_{hd}_{max_t}_{kv_dtype:?}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F16);
            let ki = g.input(Shape(vec![max_t * hd]), kv_dtype);
            let vi = g.input(Shape(vec![max_t * hd]), kv_dtype);
            let mi = g.input(Shape(vec![2]), DType::F16);
            let o = g.sdpa_naive(qi, ki, vi, mi, n_q, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, o, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[q, k, v, meta], out)
    }

    /// Oracle-shaped hybrid: shared-head online for short KV; partitioned MWG for long.
    pub const SDPA_MWG_NWG: usize = 16;
    pub const SDPA_MWG_THRESHOLD: u32 = 128;

    /// Partitioned MWG SDPA (pass1 + reduce). `tmp` is F32 `[n_q * NWG * (hd + 2)]`.
    pub fn sdpa_mwg_kv(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        kv_dtype: DType,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        tmp: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let nwg = Self::SDPA_MWG_NWG;
        let part_key = format!("sdpa_mwg_part_{n_q}_{hd}_{max_t}_{nwg}_{kv_dtype:?}");
        if !self.cache.contains_key(&part_key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F16);
            let ki = g.input(Shape(vec![max_t * hd]), kv_dtype);
            let vi = g.input(Shape(vec![max_t * hd]), kv_dtype);
            let mi = g.input(Shape(vec![2]), DType::F16);
            let o = g.sdpa_mwg_part(qi, ki, vi, mi, n_q, hd, max_t, nwg)?;
            self.ensure(ctx, &part_key, lower_to_metal_chip(&g, o, &ctx.device_name())?)?;
        }
        let red_key = format!("sdpa_mwg_reduce_{n_q}_{hd}_{nwg}");
        if !self.cache.contains_key(&red_key) {
            let mut g = Graph::new();
            let ti = g.input(Shape(vec![n_q * nwg * (hd + 2)]), DType::F32);
            let o = g.sdpa_mwg_reduce(ti, n_q, hd, nwg)?;
            self.ensure(ctx, &red_key, lower_to_metal_chip(&g, o, &ctx.device_name())?)?;
        }
        self.run(ctx, &part_key, &[q, k, v, meta], tmp)?;
        self.run(ctx, &red_key, &[tmp], out)
    }

    /// Hybrid SDPA: online shared-head if `attn_t < THRESHOLD`, else MWG.
    pub fn sdpa_hybrid_kv(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        attn_t: u32,
        kv_dtype: DType,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        tmp: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        if attn_t >= Self::SDPA_MWG_THRESHOLD {
            self.sdpa_mwg_kv(ctx, n_q, hd, max_t, kv_dtype, q, k, v, meta, tmp, out)
        } else {
            self.sdpa_naive_kv(ctx, n_q, hd, max_t, kv_dtype, q, k, v, meta, out)
        }
    }

    /// Pack `n` F16 elems (n%32==0) into Q4_0 at `dst` (+ byte offset).
    pub fn quantize_q40(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        src: &Buffer,
        dst: &Buffer,
        dst_byte_off: usize,
    ) -> Result<()> {
        let key = format!("q40_pack_{n}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let out = g.quantize_q40(xi, n)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_offsets(ctx, &key, &[src], &[0], dst, dst_byte_off as u64)
    }


    /// RMSNorm + RoPE + Q4_0 pack into `dst` (+ byte offset). One launch for KV-K append.
    pub fn rmsnorm_per_head_rope_q40_off(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        w: &Buffer,
        cos_sin: &Buffer,
        cos_sin_off_elems: usize,
        dst: &Buffer,
        dst_byte_off: usize,
    ) -> Result<()> {
        let key = format!("rms_ph_rope_q40_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let wi = g.input(Shape(vec![hd]), DType::F16);
            let ci = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rmsnorm_per_head_rope_q40(xi, wi, ci, n_heads, hd, eps, true)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        let b = DType::F16.size_bytes() as u64;
        self.run_offsets(
            ctx,
            &key,
            &[x, w, cos_sin],
            &[0, 0, cos_sin_off_elems as u64 * b],
            dst,
            dst_byte_off as u64,
        )
    }

    /// RMSNorm (no weight) + Q4_0 pack into `dst` (+ byte offset). One launch for KV-V append.
    pub fn rmsnorm_per_head_q40_off(
        &mut self,
        ctx: &MetalContext,
        n_heads: usize,
        hd: usize,
        eps: f32,
        x: &Buffer,
        dst: &Buffer,
        dst_byte_off: usize,
    ) -> Result<()> {
        let key = format!("rms_ph_q40_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F16);
            let wi = g.input(Shape(vec![hd]), DType::F16);
            let out = g.rmsnorm_per_head_q40(xi, wi, n_heads, hd, eps, false)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_offsets(ctx, &key, &[x, x], &[0, 0], dst, dst_byte_off as u64)
    }

    pub fn copy_slice(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        src: &Buffer,
        src_off: usize,
        dst: &Buffer,
        dst_off: usize,
    ) -> Result<()> {
        let key = format!("csl_f16_{n}_{src_off}_{dst_off}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![src_off + n]), DType::F16);
            let out = g.copy_slice(xi, src_off, dst_off, n)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[src], dst)
    }

    pub fn copy_scale(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        scale: f32,
        src: &Buffer,
        src_off: usize,
        dst: &Buffer,
        dst_off: usize,
    ) -> Result<()> {
        self.copy_scale_wd(ctx, n, scale, DType::F16, src, src_off * 2, dst, dst_off * 2)
    }

    /// Copy `n` logical elems from `src` (+byte offset) with scale. `dtype` may be F16 or
    /// Q4K/Q6K (Load expand → F16). Offsets are **bytes** into the buffers.
    pub fn copy_scale_wd(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        scale: f32,
        dtype: DType,
        src: &Buffer,
        src_byte_off: usize,
        dst: &Buffer,
        dst_byte_off: usize,
    ) -> Result<()> {
        let tag = weight_cache_tag(dtype);
        let key = format!("csl_sc_{tag}_{n}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            // Logical length only — row selection is via Metal buffer byte offset.
            let xi = g.input(Shape(vec![n]), dtype);
            let out = g.copy_scale(xi, 0, 0, n, scale)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, out, &ctx.device_name())?)?;
        }
        self.run_offsets(
            ctx,
            &key,
            &[src],
            &[src_byte_off as u64],
            dst,
            dst_byte_off as u64,
        )
    }

    pub fn softcap_argmax(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        cap: f32,
        x: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("sca_f16_{n}_{}", cap.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F16);
            let o = g.softcap_argmax(xi, cap)?;
            self.ensure(ctx, &key, lower_to_metal_chip(&g, o, &ctx.device_name())?)?;
        }
        self.run(ctx, &key, &[x], out)
    }

    /// BEAM-search one matvec; `weight` must match `weight_dtype` packing.
    pub fn beam_matvec(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        weight_dtype: DType,
        weight: &Buffer,
    ) -> Result<()> {
        use ksearch_codegen::{beam_search_matvec, load_plan};
        let chip = ctx.device_name();
        let plan_kind = match weight_dtype {
            DType::Q4K => "matvec_q4k",
            DType::Q6K => "matvec_q6k",
            DType::F16 => "matvec_f16_nr",
            _ => "matvec_f32",
        };
        let force = std::env::var("KSEARCH_BEAM_FORCE").is_ok();
        if !force && load_plan(plan_kind, &[rows, cols], &chip).is_some() {
            return Ok(());
        }
        eprintln!(
            "[beam] searching {} matvec {rows}x{cols} …",
            weight_cache_tag(weight_dtype)
        );
        let mut g = Graph::new();
        let w = g.input(Shape(vec![rows, cols]), weight_dtype);
        let v = g.input(Shape(vec![cols]), DType::F16);
        let y = g.matvec_prim(w, v)?;
        let bx = ctx.buffer_empty_f16(cols);
        let by = ctx.buffer_empty_f16(rows);
        let time_one = |kernel: &MetalKernelSource| -> Result<f64, ksearch_codegen::CodegenError> {
            let pipe = ctx
                .compile(kernel)
                .map_err(|e| ksearch_codegen::CodegenError::Msg(e.to_string()))?;
            let tg = Self::tg_for(kernel);
            let _ = ctx
                .run(&pipe, kernel, &[weight, &bx], &by, tg)
                .map_err(|e| ksearch_codegen::CodegenError::Msg(e.to_string()))?;
            let mut samples = [0f64; 3];
            for s in &mut samples {
                *s = ctx
                    .run(&pipe, kernel, &[weight, &bx], &by, tg)
                    .map_err(|e| ksearch_codegen::CodegenError::Msg(e.to_string()))?;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(samples[1])
        };
        let result = beam_search_matvec(&g, y, &chip, time_one).map_err(|e| anyhow::anyhow!(e))?;
        eprintln!(
            "[beam] {rows}x{cols} → tg={} vec={} unroll={} nr0={} ({:.3} ms{})",
            result.schedule.tg,
            result.schedule.vec,
            result.schedule.unroll,
            result.schedule.nr0,
            result.ms,
            if result.from_cache { ", cache" } else { "" }
        );
        Ok(())
    }

    /// BEAM-search one F16 matvec; `weight` must hold `rows*cols` halfs (may be product embd).
    pub fn beam_f16_matvec(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        weight: &Buffer,
    ) -> Result<()> {
        self.beam_matvec(ctx, rows, cols, DType::F16, weight)
    }

    /// Warm plan cache for mid-size matvecs (allocates scratch weights).
    pub fn warm_matvec_plans(
        &mut self,
        ctx: &MetalContext,
        weight_dtype: DType,
        shapes: &[(usize, usize)],
    ) -> Result<()> {
        const MAX_ELEMS: usize = 14_000_000; // include PLE prepass 8960×1536
        for &(rows, cols) in shapes {
            let n = rows.saturating_mul(cols);
            if rows == 0 || cols == 0 || n > MAX_ELEMS {
                continue;
            }
            if matches!(weight_dtype, DType::Q4K | DType::Q6K) && n % 256 != 0 {
                continue;
            }
            let w = match weight_dtype {
                DType::Q4K => ctx.buffer_empty_bytes(ksearch_ir::q4k_nbytes(n)),
                DType::Q6K => ctx.buffer_empty_bytes(ksearch_ir::q6k_nbytes(n)),
                DType::F16 => ctx.buffer_empty_f16(n),
                DType::F32 => ctx.buffer_empty_f32(n),
                _ => continue,
            };
            self.beam_matvec(ctx, rows, cols, weight_dtype, &w)?;
        }
        Ok(())
    }

    /// Warm plan cache for mid-size F16 matvecs (allocates scratch weights).
    pub fn warm_f16_matvec_plans(
        &mut self,
        ctx: &MetalContext,
        shapes: &[(usize, usize)],
    ) -> Result<()> {
        self.warm_matvec_plans(ctx, DType::F16, shapes)
    }
}

fn weight_cache_tag(d: DType) -> &'static str {
    match d {
        DType::Q4K => "q4k",
        DType::Q6K => "q6k",
        DType::F16 => "f16",
        DType::F32 => "f32",
        _ => "other",
    }
}

impl Default for Eng {
    fn default() -> Self {
        Self::new()
    }
}

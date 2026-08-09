//! Eng: Graph → lower_to_metal → Metal. No hand `gen_*` templates.

use anyhow::Result;
use ksearch_codegen::{lower_to_metal, MetalKernelSource};
use ksearch_ir::{DType, Graph, Shape};
use ksearch_metal::MetalContext;
use metal::*;
use std::collections::HashMap;

pub struct Eng {
    cache: HashMap<String, (MetalKernelSource, ComputePipelineState)>,
}

impl Eng {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
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

    fn run(
        &self,
        ctx: &MetalContext,
        key: &str,
        inputs: &[&Buffer],
        output: &Buffer,
        tg: u64,
    ) -> Result<()> {
        let (src, pipe) = self.cache.get(key).expect("ensure first");
        // Coalesce into pending CB; host wait happens on synchronize/read.
        ctx.encode(pipe, src, inputs, output, tg)?;
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
        let key = format!("mv_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::F32);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let m = g.mul_broadcast_row(w, v)?;
            let out = g.sum_reduce(m, 1)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        // Launch TG baked into RowsParallel.
        self.run(ctx, &key, &[a, x], y, 32)
    }

    #[allow(dead_code)]
    pub fn matvec_bf16(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_bf16_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::BF16);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_bf16(w, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[a, x], y, 32)
    }

    pub fn matvec_q4k(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_q4k_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_q4k(w, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        // Launch TG baked into RowsParallel hint.
        self.run(ctx, &key, &[a, x], y, 64)
    }

    /// Fused Q4_K gate∥up + GeLU(gate)*up.
    pub fn matvec_q4k_gate_up_gelu(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_q4k_gug_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wg = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let wu = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_q4k_gate_up_gelu(wg, wu, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[gate, up, x], y, 64)
    }

    pub fn inv_rms(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        eps: f32,
        x: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("inv_rms_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let o = g.inv_rms(xi, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(ctx, &key, &[x], out, 256)
    }

    /// Precomputed inv_rms fused into Q4 gate∥up + GeLU.
    pub fn matvec_q4k_rms_gate_up_gelu(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        nw: &Buffer,
        inv: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_q4k_rms_gug_{rows}x{cols}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wg = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let wu = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let nwi = g.input(Shape(vec![cols]), DType::F32);
            let invi = g.input(Shape(vec![1]), DType::F32);
            let out = g.matvec_q4k_rms_gate_up_gelu(wg, wu, v, nwi, invi, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[gate, up, x, nw, inv], y, 64)
    }

    /// Precomputed inv_rms fused into Q4_K matvec.
    pub fn matvec_q4k_rms(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        eps: f32,
        a: &Buffer,
        x: &Buffer,
        nw: &Buffer,
        inv: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_q4k_rms_{rows}x{cols}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let nwi = g.input(Shape(vec![cols]), DType::F32);
            let invi = g.input(Shape(vec![1]), DType::F32);
            let out = g.matvec_q4k_rms(w, v, nwi, invi, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[a, x, nw, inv], y, 64)
    }

    pub fn matvec_q6k(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("mv_q6k_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::Q6K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_q6k(w, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        // Launch TG baked into RowsParallel hint.
        self.run(ctx, &key, &[a, x], y, 64)
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
        let key = format!("rms_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let wi = g.input(Shape(vec![n]), DType::F32);
            let out = g.rmsnorm(xi, wi, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, w], y, 256)
    }

    /// `y = rmsnorm(x,w) + residual`.
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
        let key = format!("rms_add_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let wi = g.input(Shape(vec![n]), DType::F32);
            let ri = g.input(Shape(vec![n]), DType::F32);
            let out = g.rmsnorm_add(xi, wi, ri, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, w, residual], y, 256)
    }

    /// `y = scale * (rmsnorm(x,w) + residual)`.
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
        let key = format!("rms_add_sc_{n}_{}_{}", eps.to_bits(), scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let wi = g.input(Shape(vec![n]), DType::F32);
            let ri = g.input(Shape(vec![n]), DType::F32);
            let out = g.rmsnorm_add_scale(xi, wi, ri, eps, scale)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, w, residual], y, 256)
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
        let key = format!("rms_ph_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F32);
            let wi = g.input(Shape(vec![hd]), DType::F32);
            let out = g.rmsnorm_per_head(xi, wi, n_heads, hd, eps, true)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, w], y, n_heads.max(1) as u64)
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
        let key = format!("rms_nw_{n_heads}_{hd}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F32);
            let wi = g.input(Shape(vec![hd]), DType::F32); // unused binding
            let out = g.rmsnorm_per_head(xi, wi, n_heads, hd, eps, false)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, x], y, n_heads.max(1) as u64)
    }

    pub fn add(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("add_{n}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let a = g.input(Shape(vec![n]), DType::F32);
            let b = g.input(Shape(vec![n]), DType::F32);
            let out = g.add(a, b)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[a, b], y, 256)
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
        let key = format!("gelu_{n}_{up_off}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let gate_i = g.input(Shape(vec![n]), DType::F32);
            let up_i = g.input(Shape(vec![up_off + n]), DType::F32);
            let out = g.gelu_mul_at(gate_i, up_i, up_off)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[gate, up], y, 256)
    }

    pub fn scale_const(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        scale: f32,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("sc_{n}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let out = g.scale_const(xi, scale)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x], y, 256)
    }

    pub fn scale(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        x: &Buffer,
        s: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("scale_{n}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let si = g.input(Shape(vec![1]), DType::F32);
            let out = g.scale_buf(xi, si)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, s], y, 256)
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
        let key = format!("rope_{n_heads}_{hd}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n_heads * hd]), DType::F32);
            let ci = g.input(Shape(vec![hd]), DType::F32);
            let out = g.rope(xi, ci, n_heads, hd)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x, cos_sin], y, n_heads.max(1) as u64)
    }

    pub fn attn_gqa(
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
        let key = format!("attn_flash_{n_q}_{hd}_{max_t}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let ki = g.input(Shape(vec![max_t * hd]), DType::F32);
            let vi = g.input(Shape(vec![max_t * hd]), DType::F32);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let o = g.attn_gqa(qi, ki, vi, mi, n_q, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(ctx, &key, &[q, k, v, meta], out, n_q.max(1) as u64)
    }

    /// Quantize f32 row into Q4_0 KV cache at token index in `pos` (u32[1]).
    pub fn kv_append_q4(
        &mut self,
        ctx: &MetalContext,
        hd: usize,
        max_t: usize,
        src: &Buffer,
        pos: &Buffer,
        cache: &Buffer,
    ) -> Result<()> {
        let key = format!("kvq4_{hd}_{max_t}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let si = g.input(Shape(vec![hd]), DType::F32);
            let pi = g.input(Shape(vec![1]), DType::F32);
            let out = g.kv_append_q4(si, pi, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[src, pos], cache, 32)
    }

    /// Flash decode attention over Q4_0 KV caches.
    pub fn attn_gqa_q4(
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
        let key = format!("attn_flash_q4_t32tg256_{n_q}_{hd}_{max_t}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let ki = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let vi = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let o = g.attn_gqa_q4(qi, ki, vi, mi, n_q, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(ctx, &key, &[q, k, v, meta], out, n_q.max(1) as u64)
    }

    /// MWG-style split-KV flash-Q4: split across `nwg` WGs then reduce.
    /// Caller provides `partials` sized `n_q * nwg * (hd + 2)` floats. Uses nwg=32.
    pub fn attn_gqa_q4_mwg(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        meta: &Buffer,
        partials: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        const NWG: usize = 32;
        self.ensure_attn_gqa_q4_mwg(ctx, n_q, hd, max_t)?;
        let split_key = format!("attn_flash_q4_mwg_split_tg32_{NWG}_{n_q}_{hd}_{max_t}");
        self.run(
            ctx,
            &split_key,
            &[q, k, v, meta],
            partials,
            (n_q * NWG).max(1) as u64,
        )?;
        let reduce_key = format!("attn_flash_q4_mwg_reduce_tg32_{NWG}_{n_q}_{hd}");
        self.run(ctx, &reduce_key, &[partials], out, n_q.max(1) as u64)
    }

    /// Compile MWG split+reduce pipelines (no dispatch).
    pub fn ensure_attn_gqa_q4_mwg(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
    ) -> Result<()> {
        const NWG: usize = 32;
        let split_key = format!("attn_flash_q4_mwg_split_tg32_{NWG}_{n_q}_{hd}_{max_t}");
        if !self.cache.contains_key(&split_key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let ki = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let vi = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let o = g.attn_gqa_q4_split(qi, ki, vi, mi, n_q, hd, max_t, NWG)?;
            self.ensure(ctx, &split_key, lower_to_metal(&g, o)?)?;
        }
        let reduce_key = format!("attn_flash_q4_mwg_reduce_tg32_{NWG}_{n_q}_{hd}");
        if !self.cache.contains_key(&reduce_key) {
            let mut g = Graph::new();
            let pi = g.input(Shape(vec![n_q * NWG * (hd + 2)]), DType::F32);
            let o = g.attn_gqa_q4_reduce(pi, n_q, hd, NWG)?;
            self.ensure(ctx, &reduce_key, lower_to_metal(&g, o)?)?;
        }
        Ok(())
    }

    /// Fused flash-Q4 attn + KV append (current token from f32; caches updated in-kernel).
    pub fn attn_gqa_q4_fused(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        k_new: &Buffer,
        v_new: &Buffer,
        meta: &Buffer,
        pos: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("attn_flash_q4f_t32tg256_{n_q}_{hd}_{max_t}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let ki = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let vi = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let kn = g.input(Shape(vec![hd]), DType::F32);
            let vn = g.input(Shape(vec![hd]), DType::F32);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let pi = g.input(Shape(vec![1]), DType::F32);
            let o = g.attn_gqa_q4_fused(qi, ki, vi, kn, vn, mi, pi, n_q, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(
            ctx,
            &key,
            &[q, k, v, k_new, v_new, meta, pos],
            out,
            n_q.max(1) as u64,
        )
    }

    /// Q-only fused MQA flash for shared-KV layers (Q-norm+RoPE + flash, no append).
    pub fn attn_gqa_q4_q_fused(
        &mut self,
        ctx: &MetalContext,
        n_q: usize,
        hd: usize,
        max_t: usize,
        eps: f32,
        q: &Buffer,
        q_norm: &Buffer,
        cos_sin: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        meta: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("attn_flash_q4qf_mqa_{n_q}_{hd}_{max_t}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let qn = g.input(Shape(vec![hd]), DType::F32);
            let cs = g.input(Shape(vec![hd]), DType::F32);
            let kc = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let vc = g.input(Shape(vec![max_t * hd]), DType::Q40);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let o = g.attn_gqa_q4_q_fused(qi, qn, cs, kc, vc, mi, n_q, hd, max_t, eps)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(
            ctx,
            &key,
            &[q, q_norm, cos_sin, k_cache, v_cache, meta],
            out,
            n_q.max(1) as u64,
        )
    }

    pub fn softcap(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        cap: f32,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("softcap_{n}_{}", cap.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let out = g.softcap(xi, cap)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x], y, 256)
    }

    pub fn argmax(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        x: &Buffer,
        out_idx: &Buffer,
    ) -> Result<()> {
        let key = format!("argmax_{n}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let out = g.argmax(xi)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x], out_idx, 1)
    }

    pub fn copy(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.scale_const(ctx, n, 1.0, x, y)
    }

    /// GPU copy of `n` f32s with offsets (avoids host sync + memcpy).
    pub fn copy_slice(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        src: &Buffer,
        src_off: usize,
        dst: &Buffer,
        dst_off: usize,
    ) -> Result<()> {
        let key = format!("csl_{n}_{src_off}_{dst_off}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            // Placeholder shape large enough for offset+n checks in IR.
            let xi = g.input(Shape(vec![src_off + n]), DType::F32);
            let out = g.copy_slice(xi, src_off, dst_off, n)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[src], dst, 256)
    }

    /// Gather Q4_K row `token` into `y`, scaled by `scale`.
    pub fn gather_q4k_row(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        scale: f32,
        w: &Buffer,
        row_idx: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("gq4k_{rows}x{cols}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wi = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let ri = g.input(Shape(vec![1]), DType::F32);
            let out = g.gather_q4k_row(wi, ri, scale)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[w, row_idx], y, 256)
    }

    /// Gather Q5_K row `token` into `y`, scaled by `scale`.
    pub fn gather_q5k_row(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        scale: f32,
        w: &Buffer,
        row_idx: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        let key = format!("gq5k_{rows}x{cols}_{}", scale.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let wi = g.input(Shape(vec![rows, cols]), DType::Q5K);
            let ri = g.input(Shape(vec![1]), DType::F32);
            let out = g.gather_q5k_row(wi, ri, scale)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[w, row_idx], y, 256)
    }

    /// Fused softcap + argmax (IR SoftcapArgmax → MSL).
    pub fn softcap_argmax(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        cap: f32,
        x: &Buffer,
        out_idx: &Buffer,
    ) -> Result<()> {
        let key = format!("sca_{n}_{}", cap.to_bits());
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let xi = g.input(Shape(vec![n]), DType::F32);
            let out = g.softcap_argmax(xi, cap)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[x], out_idx, 1)
    }
}

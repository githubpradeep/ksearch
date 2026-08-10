//! Eng: Graph/Kernel IR → render_msl → Metal. Thesis A only (no fused catalog Ops).

use anyhow::Result;
use ksearch_codegen::{
    layer, lower_to_metal, MetalKernelSource,
};
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
        ctx.encode(pipe, src, inputs, output, tg)?;
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
        tg: u64,
    ) -> Result<()> {
        let (src, pipe) = self.cache.get(key).expect("ensure first");
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
        let key = format!("mv_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::F32);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_prim(w, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        self.run(ctx, &key, &[a, x], y, 32)
    }

    /// Q4_K weights via `matvec_prim` — dtype fusion at Kernel IR render.
    pub fn matvec_q4k_prim(
        &mut self,
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        a: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        self.matvec_q4k_prim_at(ctx, rows, cols, a, x, 0, y, 0)
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
        let key = format!("mv_q4k_prim_{rows}x{cols}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let w = g.input(Shape(vec![rows, cols]), DType::Q4K);
            let v = g.input(Shape(vec![cols]), DType::F32);
            let out = g.matvec_prim(w, v)?;
            self.ensure(ctx, &key, lower_to_metal(&g, out)?)?;
        }
        let xb = (x_off_elems * 4) as u64;
        let yb = (y_off_elems * 4) as u64;
        self.run_offsets(ctx, &key, &[a, x], &[0, xb], y, yb, 64)
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
        let key = format!("rms_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            self.ensure(ctx, &key, layer::render_rmsnorm(n, eps)?)?;
        }
        let xb = (x_off_elems * 4) as u64;
        let yb = (y_off_elems * 4) as u64;
        self.run_offsets(ctx, &key, &[x, w], &[xb, 0], y, yb, 256)
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
        let key = format!("rms_add_{n}_{}", eps.to_bits());
        if !self.cache.contains_key(&key) {
            self.ensure(ctx, &key, layer::render_rmsnorm_add(n, eps)?)?;
        }
        self.run(ctx, &key, &[x, w, residual], y, 256)
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
        let key = format!("rms_add_sc_{n}_{}_{}", eps.to_bits(), scale.to_bits());
        if !self.cache.contains_key(&key) {
            self.ensure(ctx, &key, layer::render_rmsnorm_add_scale(n, eps, scale)?)?;
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
            self.ensure(
                ctx,
                &key,
                layer::render_rmsnorm_per_head(n_heads, hd, eps, true)?,
            )?;
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
            self.ensure(
                ctx,
                &key,
                layer::render_rmsnorm_per_head(n_heads, hd, eps, false)?,
            )?;
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
            self.ensure(ctx, &key, layer::render_gelu_mul(n, up_off)?)?;
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
            self.ensure(ctx, &key, layer::render_rope(n_heads, hd)?)?;
        }
        self.run(ctx, &key, &[x, cos_sin], y, n_heads.max(1) as u64)
    }

    /// Naive MQA SDPA (Graph → schedule → Kernel IR → MSL).
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
        let key = format!("sdpa_naive_{n_q}_{hd}_{max_t}");
        if !self.cache.contains_key(&key) {
            let mut g = Graph::new();
            let qi = g.input(Shape(vec![n_q * hd]), DType::F32);
            let ki = g.input(Shape(vec![max_t * hd]), DType::F32);
            let vi = g.input(Shape(vec![max_t * hd]), DType::F32);
            let mi = g.input(Shape(vec![2]), DType::F32);
            let o = g.sdpa_naive(qi, ki, vi, mi, n_q, hd, max_t)?;
            self.ensure(ctx, &key, lower_to_metal(&g, o)?)?;
        }
        self.run(ctx, &key, &[q, k, v, meta], out, 256)
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
        let key = format!("csl_{n}_{src_off}_{dst_off}");
        if !self.cache.contains_key(&key) {
            self.ensure(ctx, &key, layer::render_copy_slice(src_off, dst_off, n)?)?;
        }
        self.run(ctx, &key, &[src], dst, 256)
    }

    pub fn softcap_argmax(
        &mut self,
        ctx: &MetalContext,
        n: usize,
        cap: f32,
        x: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let key = format!("sca_{n}_{}", cap.to_bits());
        if !self.cache.contains_key(&key) {
            self.ensure(ctx, &key, layer::render_softcap_argmax(n, cap)?)?;
        }
        self.run(ctx, &key, &[x], out, 256)
    }
}

impl Default for Eng {
    fn default() -> Self {
        Self::new()
    }
}

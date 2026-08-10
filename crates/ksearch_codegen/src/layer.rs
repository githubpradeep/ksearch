//! Layer sugar helpers: build Graph + FuseHint → schedule → MSL (same path as Eng).

use crate::{lower_to_metal, CodegenError, MetalKernelSource};
use ksearch_ir::{DType, Graph, Shape};

pub fn render_rmsnorm(n: usize, eps: f32) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n]), DType::F32);
    let w = g.input(Shape(vec![n]), DType::F32);
    let out = g.rmsnorm_expand(x, w, eps)?;
    lower_to_metal(&g, out)
}

pub fn render_rmsnorm_add(n: usize, eps: f32) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n]), DType::F32);
    let w = g.input(Shape(vec![n]), DType::F32);
    let r = g.input(Shape(vec![n]), DType::F32);
    let out = g.rmsnorm_add_expand(x, w, r, eps)?;
    lower_to_metal(&g, out)
}

pub fn render_rmsnorm_add_scale(
    n: usize,
    eps: f32,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n]), DType::F32);
    let w = g.input(Shape(vec![n]), DType::F32);
    let r = g.input(Shape(vec![n]), DType::F32);
    let out = g.rmsnorm_add_scale_expand(x, w, r, eps, scale)?;
    lower_to_metal(&g, out)
}

pub fn render_rmsnorm_per_head(
    n_heads: usize,
    hd: usize,
    eps: f32,
    with_weight: bool,
) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n_heads * hd]), DType::F32);
    let w = g.input(Shape(vec![hd]), DType::F32);
    let out = g.rmsnorm_per_head(x, w, n_heads, hd, eps, with_weight)?;
    lower_to_metal(&g, out)
}

pub fn render_rope(n_heads: usize, hd: usize) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n_heads * hd]), DType::F32);
    let c = g.input(Shape(vec![hd]), DType::F32);
    let out = g.rope(x, c, n_heads, hd)?;
    lower_to_metal(&g, out)
}

pub fn render_gelu_mul(n: usize, up_off: usize) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let gate = g.input(Shape(vec![n]), DType::F32);
    let up = g.input(Shape(vec![up_off + n]), DType::F32);
    let out = g.gelu_mul_at(gate, up, up_off)?;
    lower_to_metal(&g, out)
}

pub fn render_copy_slice(
    src_off: usize,
    dst_off: usize,
    n: usize,
) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![src_off + n]), DType::F32);
    let out = g.copy_slice(x, src_off, dst_off, n)?;
    lower_to_metal(&g, out)
}

pub fn render_softcap_argmax(n: usize, cap: f32) -> Result<MetalKernelSource, CodegenError> {
    let mut g = Graph::new();
    let x = g.input(Shape(vec![n]), DType::F32);
    let out = g.softcap_argmax(x, cap)?;
    lower_to_metal(&g, out)
}

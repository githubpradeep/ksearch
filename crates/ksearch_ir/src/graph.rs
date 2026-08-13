//! Tinygrad-shaped Graph: primitives + CALL regions; sugar expands / hints fusion.

use crate::{DType, FuseHint, IrError, Shape, TensorId};
use std::collections::HashMap;

/// Out dtype for matvec-like FuseHints: float weights match act, or Q4K/Q6K×F16 → F16.
fn matvec_weight_act_out(weight: DType, act: DType, cols: usize) -> Result<DType, IrError> {
    match (weight, act) {
        (d, a) if d.is_float() && a == d => Ok(d),
        (DType::Q4K | DType::Q6K, DType::F16) if cols % 256 == 0 => Ok(DType::F16),
        _ => Err(IrError::ShapeMismatch),
    }
}

fn packed_k_quant(d: DType) -> bool {
    matches!(d, DType::Q4K | DType::Q6K)
}

/// QKV weights: all same float, or any mix of packed K-quants (shared F16 out).
fn qkv_weight_dtypes_ok(dq: DType, dk: DType, dv: DType, dx: DType, cols: usize) -> Result<DType, IrError> {
    let oq = matvec_weight_act_out(dq, dx, cols)?;
    let ok = matvec_weight_act_out(dk, dx, cols)?;
    let ov = matvec_weight_act_out(dv, dx, cols)?;
    if oq != ok || ok != ov {
        return Err(IrError::ShapeMismatch);
    }
    let same = dq == dk && dk == dv;
    let all_packed = packed_k_quant(dq) && packed_k_quant(dk) && packed_k_quant(dv);
    if same || all_packed {
        Ok(oq)
    } else {
        Err(IrError::ShapeMismatch)
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Input { shape: Shape, dtype: DType },
    Const { value: f32, shape: Shape, dtype: DType },
    Add { a: TensorId, b: TensorId },
    Mul { a: TensorId, b: TensorId },
    ScaleConst { x: TensorId, scale: f32 },
    Rsqrt { x: TensorId },
    Tanh { x: TensorId },
    Exp { x: TensorId },
    SumReduce { inp: TensorId, axis: usize },
    MaxReduce { inp: TensorId, axis: usize },
    Expand { inp: TensorId, shape: Shape },
    Reshape { inp: TensorId, shape: Shape },
    Permute { inp: TensorId, axes: Vec<usize> },
    MulBroadcastRow { left: TensorId, row: TensorId },
    CopySlice {
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
    },
    /// Tinygrad-like CALL: one scheduled kernel over `inputs` (algorithm via [`FuseHint`]).
    Call { inputs: Vec<TensorId> },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// Sugar / scheduler fusion hints keyed by output tensor id.
    pub fuse_hints: HashMap<u32, FuseHint>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, op: Op, shape: Shape, dtype: DType) -> TensorId {
        let id = TensorId(self.nodes.len() as u32);
        self.nodes.push(Node { op, shape, dtype });
        id
    }

    pub fn hint(&mut self, id: TensorId, hint: FuseHint) {
        self.fuse_hints.insert(id.0, hint);
    }

    pub fn fuse_hint(&self, id: TensorId) -> Option<&FuseHint> {
        self.fuse_hints.get(&id.0)
    }

    pub fn input(&mut self, shape: Shape, dtype: DType) -> TensorId {
        self.push(Op::Input { shape: shape.clone(), dtype }, shape, dtype)
    }

    pub fn const_f32(&mut self, value: f32, shape: Shape) -> TensorId {
        self.const_val(value, shape, DType::F32)
    }

    pub fn const_val(&mut self, value: f32, shape: Shape, dtype: DType) -> TensorId {
        self.push(
            Op::Const {
                value,
                shape: shape.clone(),
                dtype,
            },
            shape,
            dtype,
        )
    }

    pub fn call(
        &mut self,
        inputs: Vec<TensorId>,
        shape: Shape,
        dtype: DType,
        hint: FuseHint,
    ) -> TensorId {
        let id = self.push(Op::Call { inputs }, shape, dtype);
        self.hint(id, hint);
        id
    }

    pub fn shape_dtype(&self, id: TensorId) -> Result<(Shape, DType), IrError> {
        let n = self.nodes.get(id.0 as usize).ok_or(IrError::BadTensorId)?;
        Ok((n.shape.clone(), n.dtype))
    }

    pub fn node(&self, id: TensorId) -> Result<&Node, IrError> {
        self.nodes.get(id.0 as usize).ok_or(IrError::BadTensorId)
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, IrError> {
        let (sa, da) = self.shape_dtype(a)?;
        let (sb, db) = self.shape_dtype(b)?;
        if sa != sb || da != db {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::Add { a, b }, sa, da))
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, IrError> {
        let (sa, da) = self.shape_dtype(a)?;
        let (sb, db) = self.shape_dtype(b)?;
        if sa != sb || da != db {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::Mul { a, b }, sa, da))
    }

    pub fn scale_const(&mut self, x: TensorId, scale: f32) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        Ok(self.push(Op::ScaleConst { x, scale }, s, d))
    }

    pub fn rsqrt(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        Ok(self.push(Op::Rsqrt { x }, s, d))
    }

    pub fn tanh(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        Ok(self.push(Op::Tanh { x }, s, d))
    }

    pub fn exp(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        Ok(self.push(Op::Exp { x }, s, d))
    }

    pub fn mul_broadcast_row(&mut self, left: TensorId, row: TensorId) -> Result<TensorId, IrError> {
        let (sl, dl) = self.shape_dtype(left)?;
        let (sr, dr) = self.shape_dtype(row)?;
        if sl.rank() != 2 || sr.rank() != 1 || sl.0[1] != sr.0[0] {
            return Err(IrError::ShapeMismatch);
        }
        let out_dtype = match (dl, dr) {
            (DType::F32, DType::F32) => DType::F32,
            (DType::F16, DType::F16) => DType::F16,
            (DType::Q4K | DType::Q6K, DType::F32) if sl.0[1] % 256 == 0 => DType::F32,
            (DType::Q4K | DType::Q6K, DType::F16) if sl.0[1] % 256 == 0 => DType::F16,
            _ => return Err(IrError::ShapeMismatch),
        };
        Ok(self.push(
            Op::MulBroadcastRow { left, row },
            sl.clone(),
            out_dtype,
        ))
    }

    pub fn matvec_prim(&mut self, w: TensorId, x: TensorId) -> Result<TensorId, IrError> {
        let m = self.mul_broadcast_row(w, x)?;
        self.sum_reduce(m, 1)
    }

    pub fn sum_reduce(&mut self, inp: TensorId, axis: usize) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(inp)?;
        if axis >= s.rank() {
            return Err(IrError::BadAxis);
        }
        let mut out = s.0.clone();
        out.remove(axis);
        if out.is_empty() {
            out.push(1);
        }
        Ok(self.push(Op::SumReduce { inp, axis }, Shape(out), d))
    }

    pub fn max_reduce(&mut self, inp: TensorId, axis: usize) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(inp)?;
        if axis >= s.rank() {
            return Err(IrError::BadAxis);
        }
        let mut out = s.0.clone();
        out.remove(axis);
        if out.is_empty() {
            out.push(1);
        }
        Ok(self.push(Op::MaxReduce { inp, axis }, Shape(out), d))
    }

    pub fn expand(&mut self, inp: TensorId, shape: Shape) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(inp)?;
        let src_n = s.numel();
        let dst_n = shape.numel();
        if src_n == 0 || dst_n % src_n != 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::Expand { inp, shape: shape.clone() }, shape, d))
    }

    pub fn reshape(&mut self, inp: TensorId, shape: Shape) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(inp)?;
        if s.numel() != shape.numel() {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::Reshape { inp, shape: shape.clone() }, shape, d))
    }

    pub fn permute(&mut self, inp: TensorId, axes: Vec<usize>) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(inp)?;
        if axes.len() != s.rank() {
            return Err(IrError::ShapeMismatch);
        }
        let mut seen = vec![false; axes.len()];
        let mut out = Vec::with_capacity(axes.len());
        for &a in &axes {
            if a >= s.rank() || seen[a] {
                return Err(IrError::ShapeMismatch);
            }
            seen[a] = true;
            out.push(s.0[a]);
        }
        Ok(self.push(Op::Permute { inp, axes }, Shape(out), d))
    }

    pub fn copy_slice(
        &mut self,
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) -> Result<TensorId, IrError> {
        let (_, d) = self.shape_dtype(src)?;
        Ok(self.push(
            Op::CopySlice {
                src,
                src_off,
                dst_off,
                n,
            },
            Shape(vec![n]),
            d,
        ))
    }

    pub fn square(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        self.mul(x, x)
    }

    pub fn softcap(&mut self, x: TensorId, cap: f32) -> Result<TensorId, IrError> {
        let scaled = self.scale_const(x, 1.0 / cap)?;
        let t = self.tanh(scaled)?;
        self.scale_const(t, cap)
    }

    pub fn gelu_tanh(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        let x2 = self.mul(x, x)?;
        let x3 = self.mul(x2, x)?;
        let c044 = self.scale_const(x3, 0.044715)?;
        let inner = self.add(x, c044)?;
        let u = self.scale_const(inner, 0.79788456)?;
        let t = self.tanh(u)?;
        let one = self.const_val(1.0, s.clone(), d);
        let one_plus = self.add(one, t)?;
        let half_x = self.scale_const(x, 0.5)?;
        self.mul(half_x, one_plus)
    }

    /// Tinygrad RMSNorm expand + fuse hint for one-kernel schedule.
    pub fn rmsnorm_expand(
        &mut self,
        x: TensorId,
        w: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        if sx != sw || !dx.is_float() || dx != dw || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        let n = sx.0[0];
        let sq = self.square(x)?;
        let sum = self.sum_reduce(sq, 0)?;
        let mean = self.scale_const(sum, 1.0 / n as f32)?;
        let eps_t = self.const_val(eps, Shape(vec![1]), dx);
        let mean_eps = self.add(mean, eps_t)?;
        let inv = self.rsqrt(mean_eps)?;
        let inv_b = self.expand(inv, sx.clone())?;
        let xn = self.mul(x, inv_b)?;
        let out = self.mul(xn, w)?;
        self.hint(
            out,
            FuseHint::RmsNorm {
                n,
                eps,
                x,
                w,
            },
        );
        Ok(out)
    }

    pub fn rmsnorm_add_expand(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let yn = self.rmsnorm_expand(x, w, eps)?;
        let out = self.add(yn, residual)?;
        let n = self.shape_dtype(x)?.0.numel();
        self.hint(
            out,
            FuseHint::RmsNormAdd {
                n,
                eps,
                x,
                w,
                residual,
            },
        );
        Ok(out)
    }

    pub fn rmsnorm_add_scale_expand(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        eps: f32,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let added = self.rmsnorm_add_expand(x, w, residual, eps)?;
        let out = self.scale_const(added, scale)?;
        let n = self.shape_dtype(x)?.0.numel();
        self.hint(
            out,
            FuseHint::RmsNormAddScale {
                n,
                eps,
                scale,
                x,
                w,
                residual,
            },
        );
        Ok(out)
    }

    /// Fused attn→MLP: `out_x = residual + rms(y)*w_post`, `out_x2 = rms(out_x)*w_ffn`.
    /// Call root shape is `out_x`; Metal emits 2 outputs (like MatvecQkv).
    pub fn rmsnorm_add_then_rmsnorm(
        &mut self,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sy, dy) = self.shape_dtype(y)?;
        let (sp, dp) = self.shape_dtype(w_post)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        let (sf, df) = self.shape_dtype(w_ffn)?;
        if sy != sp || sy != sr || sy != sf || sy.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if !dy.is_float() || dy != dp || dy != dr || dy != df {
            return Err(IrError::ShapeMismatch);
        }
        let n = sy.0[0];
        Ok(self.call(
            vec![y, w_post, residual, w_ffn],
            sy,
            dy,
            FuseHint::RmsNormAddThenRmsNorm {
                n,
                eps,
                y,
                w_post,
                residual,
                w_ffn,
            },
        ))
    }

    /// Per-head RMSNorm as CALL (movement+reduce fused by schedule).
    pub fn rmsnorm_per_head(
        &mut self,
        x: TensorId,
        w: TensorId,
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.numel() != n_heads * hd {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w],
            Shape(vec![n_heads * hd]),
            dx,
            FuseHint::RmsNormPerHead {
                n_heads,
                hd,
                eps,
                with_weight,
                x,
                w,
            },
        ))
    }

    /// Per-head RMSNorm + RoPE as one CALL (decode hot path).
    pub fn rmsnorm_per_head_rope(
        &mut self,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.numel() != n_heads * hd {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w, cos_sin],
            Shape(vec![n_heads * hd]),
            dx,
            FuseHint::RmsNormPerHeadRope {
                n_heads,
                hd,
                eps,
                with_weight,
                n_tok: 1,
                x,
                w,
                cos_sin,
            },
        ))
    }


    /// Per-head RMSNorm + RoPE, pack Q4_0 (KV K append; one CALL).
    pub fn rmsnorm_per_head_rope_q40(
        &mut self,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.numel() != n_heads * hd || hd % 32 != 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w, cos_sin],
            Shape(vec![n_heads * hd]),
            DType::Q40,
            FuseHint::RmsNormPerHeadRopeQ40 {
                n_heads,
                hd,
                eps,
                with_weight,
                x,
                w,
                cos_sin,
            },
        ))
    }

    /// Per-head RMSNorm, pack Q4_0 (KV V append; one CALL).
    pub fn rmsnorm_per_head_q40(
        &mut self,
        x: TensorId,
        w: TensorId,
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.numel() != n_heads * hd || hd % 32 != 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w],
            Shape(vec![n_heads * hd]),
            DType::Q40,
            FuseHint::RmsNormPerHeadQ40 {
                n_heads,
                hd,
                eps,
                with_weight,
                x,
                w,
            },
        ))
    }

    /// Q rms+RoPE (F16) + K rms+RoPE+Q40 + V rms+Q40 as one CALL (decode KV owners).
    pub fn rmsnorm_per_head_qkv_q40(
        &mut self,
        q: TensorId,
        qw: TensorId,
        cos_sin: TensorId,
        k: TensorId,
        kw: TensorId,
        v: TensorId,
        n_q: usize,
        n_kv: usize,
        hd: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        if !dq.is_float()
            || dq != dk
            || dq != dv
            || sq.numel() != n_q * hd
            || sk.numel() != n_kv * hd
            || sv.numel() != n_kv * hd
            || hd % 32 != 0
            || n_kv == 0
            || n_q < n_kv
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, qw, cos_sin, k, kw, v],
            Shape(vec![n_q * hd]),
            dq,
            FuseHint::RmsNormPerHeadQkvQ40 {
                n_q,
                n_kv,
                hd,
                eps,
                n_tok: 1,
                q,
                qw,
                cos_sin,
                k,
                kw,
                v,
            },
        ))
    }

    /// Prefill: `n_tok` independent Q heads; K/V packed at `tok * n_kv + head`.
    pub fn rmsnorm_per_head_qkv_q40_batch(
        &mut self,
        q: TensorId,
        qw: TensorId,
        cos_sin: TensorId,
        k: TensorId,
        kw: TensorId,
        v: TensorId,
        n_q: usize,
        n_kv: usize,
        hd: usize,
        n_tok: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        if n_tok == 0
            || !dq.is_float()
            || dq != dk
            || dq != dv
            || sq.numel() != n_tok * n_q * hd
            || sk.numel() != n_tok * n_kv * hd
            || sv.numel() != n_tok * n_kv * hd
            || hd % 32 != 0
            || n_kv == 0
            || n_q < n_kv
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, qw, cos_sin, k, kw, v],
            Shape(vec![n_tok * n_q * hd]),
            dq,
            FuseHint::RmsNormPerHeadQkvQ40 {
                n_q,
                n_kv,
                hd,
                eps,
                n_tok,
                q,
                qw,
                cos_sin,
                k,
                kw,
                v,
            },
        ))
    }

    /// Prefill: `n_tok * n_heads` TGs; rope is `[n_tok, hd]` (caller offsets pos0).
    pub fn rmsnorm_per_head_rope_batch(
        &mut self,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
        n_heads: usize,
        hd: usize,
        n_tok: usize,
        eps: f32,
        with_weight: bool,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if n_tok == 0 || !dx.is_float() || sx.numel() != n_tok * n_heads * hd {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w, cos_sin],
            Shape(vec![n_tok * n_heads * hd]),
            dx,
            FuseHint::RmsNormPerHeadRope {
                n_heads,
                hd,
                eps,
                with_weight,
                n_tok,
                x,
                w,
                cos_sin,
            },
        ))
    }

    pub fn copy_scale(
        &mut self,
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(src)?;
        if s.numel() < src_off + n {
            return Err(IrError::ShapeMismatch);
        }
        // Float: same dtype. Packed K-quant: Load expand → F16 acts (Thesis A).
        let out_dt = if d.is_float() {
            d
        } else if matches!(d, DType::Q4K | DType::Q5K | DType::Q6K) {
            DType::F16
        } else {
            return Err(IrError::ShapeMismatch);
        };
        Ok(self.call(
            vec![src],
            Shape(vec![n]),
            out_dt,
            FuseHint::CopyScale {
                src_off,
                dst_off,
                n,
                scale,
                src,
                src_dtype: d,
            },
        ))
    }

    /// Gather one packed/F16 row: `out[i] = scale * src[uint(idx[0]) * n + i]`.
    pub fn copy_scale_indexed(
        &mut self,
        src: TensorId,
        idx: TensorId,
        n: usize,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (_, d) = self.shape_dtype(src)?;
        let (is, idt) = self.shape_dtype(idx)?;
        if is.numel() < 1 || idt != DType::F32 {
            return Err(IrError::ShapeMismatch);
        }
        let out_dt = if d.is_float() {
            d
        } else if matches!(d, DType::Q4K | DType::Q5K | DType::Q6K) {
            DType::F16
        } else {
            return Err(IrError::ShapeMismatch);
        };
        Ok(self.call(
            vec![src, idx],
            Shape(vec![n]),
            out_dt,
            FuseHint::CopyScaleIndexed {
                n,
                scale,
                src,
                idx,
                src_dtype: d,
            },
        ))
    }

    /// Prefill: gather `batch` rows; `idx` is F32 `[batch]`.
    pub fn copy_scale_indexed_batch(
        &mut self,
        src: TensorId,
        idx: TensorId,
        n: usize,
        batch: usize,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (_, d) = self.shape_dtype(src)?;
        let (is, idt) = self.shape_dtype(idx)?;
        if batch == 0 || n == 0 || is.numel() < batch || idt != DType::F32 {
            return Err(IrError::ShapeMismatch);
        }
        let out_dt = if d.is_float() {
            d
        } else if matches!(d, DType::Q4K | DType::Q5K | DType::Q6K) {
            DType::F16
        } else {
            return Err(IrError::ShapeMismatch);
        };
        Ok(self.call(
            vec![src, idx],
            Shape(vec![batch * n]),
            out_dt,
            FuseHint::CopyScaleIndexedBatch {
                n,
                batch,
                scale,
                src,
                idx,
                src_dtype: d,
            },
        ))
    }

    /// Prefill: `Y = W @ X` for `batch` token rows (`x` is `[batch, cols]`).
    pub fn matvec_batch(
        &mut self,
        w: TensorId,
        x: TensorId,
        batch: usize,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if sw.rank() != 2 || batch == 0 || dx != DType::F16 {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sw.0[0];
        let cols = sw.0[1];
        if sx.numel() < batch * cols {
            return Err(IrError::ShapeMismatch);
        }
        let out_dt = matvec_weight_act_out(dw, dx, cols)?;
        Ok(self.call(
            vec![w, x],
            Shape(vec![batch * rows]),
            out_dt,
            FuseHint::MatvecBatch {
                rows,
                cols,
                batch,
                w,
                x,
            },
        ))
    }

    pub fn rmsnorm_rows(
        &mut self,
        x: TensorId,
        w: TensorId,
        n: usize,
        rows: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        if rows == 0 || n == 0 || dx != DType::F16 || dw != DType::F16 {
            return Err(IrError::ShapeMismatch);
        }
        if sx.numel() < rows * n || sw.numel() < n {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w],
            Shape(vec![rows * n]),
            DType::F16,
            FuseHint::RmsNormRows {
                n,
                rows,
                eps,
                x,
                w,
            },
        ))
    }

    pub fn rmsnorm_add_rows(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        n: usize,
        rows: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        self.rmsnorm_add_scale_rows(x, w, residual, n, rows, eps, 1.0)
    }

    /// `out[t, i] = scale * (residual[t, i] + rmsnorm(x[t])*w[i])`.
    pub fn rmsnorm_add_scale_rows(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        n: usize,
        rows: usize,
        eps: f32,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        if rows == 0 || n == 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, w, residual],
            Shape(vec![rows * n]),
            DType::F16,
            FuseHint::RmsNormAddRows {
                n,
                rows,
                eps,
                scale,
                x,
                w,
                residual,
            },
        ))
    }

    pub fn rmsnorm_add_then_rmsnorm_rows(
        &mut self,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
        n: usize,
        rows: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        Ok(self.call(
            vec![y, w_post, residual, w_ffn],
            Shape(vec![rows * n]),
            DType::F16,
            FuseHint::RmsNormAddThenRmsNormRows {
                n,
                rows,
                eps,
                y,
                w_post,
                residual,
                w_ffn,
            },
        ))
    }

    pub fn rope(
        &mut self,
        x: TensorId,
        cos_sin: TensorId,
        n_heads: usize,
        hd: usize,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.numel() != n_heads * hd {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x, cos_sin],
            Shape(vec![n_heads * hd]),
            dx,
            FuseHint::Rope {
                n_heads,
                hd,
                x,
                cos_sin,
            },
        ))
    }

    /// SDPA sugar → CALL (Q@Kᵀ→softmax→@V fused by schedule; not a Graph catalog Op).
    /// K/V may be F16 or Q40 (Load expand → float in the generic renderer).
    pub fn sdpa_naive(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        let kv_ok = matches!(dk, DType::F16 | DType::Q40) && dk == dv;
        if !dq.is_float()
            || !kv_ok
            || sq.numel() != n_q * hd
            || sk.numel() != max_t * hd
            || sv.numel() != max_t * hd
            || n_q == 0
            || hd == 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, k, v, meta],
            Shape(vec![n_q * hd]),
            dq,
            FuseHint::SdpaNaive {
                n_q,
                hd,
                max_t,
                q,
                k,
                v,
                meta,
                kv_dtype: dk,
            },
        ))
    }

    /// MWG pass1: `n_q * nwg` TGs write F32 partials `(m, l, O[hd])` per (head, part).
    pub fn sdpa_mwg_part(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        let kv_ok = matches!(dk, DType::F16 | DType::Q40) && dk == dv;
        if !dq.is_float()
            || !kv_ok
            || sq.numel() != n_q * hd
            || sk.numel() != max_t * hd
            || sv.numel() != max_t * hd
            || n_q == 0
            || hd == 0
            || max_t == 0
            || nwg == 0
            || hd % 32 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, k, v, meta],
            Shape(vec![n_q * nwg * (hd + 2)]),
            DType::F32,
            FuseHint::SdpaMwgPart {
                n_q,
                hd,
                max_t,
                nwg,
                q,
                k,
                v,
                meta,
                kv_dtype: dk,
            },
        ))
    }

    /// MWG pass2: merge NWG F32 partials → F16 attention output.
    pub fn sdpa_mwg_reduce(
        &mut self,
        tmp: TensorId,
        n_q: usize,
        hd: usize,
        nwg: usize,
    ) -> Result<TensorId, IrError> {
        let (st, dt) = self.shape_dtype(tmp)?;
        if dt != DType::F32
            || st.numel() != n_q * nwg * (hd + 2)
            || n_q == 0
            || hd == 0
            || nwg == 0
            || hd % 32 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![tmp],
            Shape(vec![n_q * hd]),
            DType::F16,
            FuseHint::SdpaMwgReduce {
                n_q,
                hd,
                nwg,
                tmp,
            },
        ))
    }

    /// Prefill SDPA: `n_tok` queries packed as `[n_tok, n_q, hd]`; causal tlen = meta_tlen + tok.
    pub fn sdpa_naive_batch(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        let kv_ok = matches!(dk, DType::F16 | DType::Q40) && dk == dv;
        if !dq.is_float()
            || !kv_ok
            || sq.numel() != n_tok * n_q * hd
            || sk.numel() != max_t * hd
            || sv.numel() != max_t * hd
            || n_q == 0
            || n_tok == 0
            || hd == 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, k, v, meta],
            Shape(vec![n_tok * n_q * hd]),
            dq,
            FuseHint::SdpaNaiveBatch {
                n_q,
                n_tok,
                hd,
                max_t,
                q,
                k,
                v,
                meta,
                kv_dtype: dk,
            },
        ))
    }

    pub fn sdpa_mwg_part_batch(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sk, dk) = self.shape_dtype(k)?;
        let (sv, dv) = self.shape_dtype(v)?;
        let kv_ok = matches!(dk, DType::F16 | DType::Q40) && dk == dv;
        if !dq.is_float()
            || !kv_ok
            || sq.numel() != n_tok * n_q * hd
            || sk.numel() != max_t * hd
            || sv.numel() != max_t * hd
            || n_q == 0
            || n_tok == 0
            || hd == 0
            || max_t == 0
            || nwg == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![q, k, v, meta],
            Shape(vec![n_tok * n_q * nwg * (hd + 2)]),
            DType::F32,
            FuseHint::SdpaMwgPartBatch {
                n_q,
                n_tok,
                hd,
                max_t,
                nwg,
                q,
                k,
                v,
                meta,
                kv_dtype: dk,
            },
        ))
    }

    pub fn sdpa_mwg_reduce_batch(
        &mut self,
        tmp: TensorId,
        n_q: usize,
        n_tok: usize,
        hd: usize,
        nwg: usize,
    ) -> Result<TensorId, IrError> {
        let (st, dt) = self.shape_dtype(tmp)?;
        if dt != DType::F32
            || st.numel() != n_tok * n_q * nwg * (hd + 2)
            || n_q == 0
            || n_tok == 0
            || hd == 0
            || nwg == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![tmp],
            Shape(vec![n_tok * n_q * hd]),
            DType::F16,
            FuseHint::SdpaMwgReduceBatch {
                n_q,
                n_tok,
                hd,
                nwg,
                tmp,
            },
        ))
    }

    /// Pack `n` F16 elems (n%32==0) into Q4_0 blocks (logical shape stays `n`).
    pub fn quantize_q40(&mut self, x: TensorId, n: usize) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if dx != DType::F16 || sx.numel() < n || n == 0 || n % 32 != 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.call(
            vec![x],
            Shape(vec![n]),
            DType::Q40,
            FuseHint::QuantizeQ40 { n, src: x },
        ))
    }

    pub fn gelu_mul_at(
        &mut self,
        gate: TensorId,
        up: TensorId,
        up_off: usize,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        if !dg.is_float() || du != dg || sg.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        let n = sg.0[0];
        if su.numel() < up_off + n {
            return Err(IrError::ShapeMismatch);
        }
        let g = self.gelu_tanh(gate)?;
        let up_s = self.copy_slice(up, up_off, 0, n)?;
        let out = self.mul(g, up_s)?;
        self.hint(
            out,
            FuseHint::GeluMul {
                n,
                up_off,
                inner: 0,
                up_stride: 0,
                gate,
                up,
            },
        );
        Ok(out)
    }

    /// `out[t*inner + i] = gelu(gate[t*inner + i]) * up[up_off + t*up_stride + i]`.
    pub fn gelu_mul_strided(
        &mut self,
        gate: TensorId,
        up: TensorId,
        inner: usize,
        n_tok: usize,
        up_off: usize,
        up_stride: usize,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        let n = inner.saturating_mul(n_tok);
        if inner == 0
            || n_tok == 0
            || !dg.is_float()
            || du != dg
            || sg.numel() != n
            || su.numel() < up_off + (n_tok - 1) * up_stride + inner
        {
            return Err(IrError::ShapeMismatch);
        }
        let g = self.gelu_tanh(gate)?;
        let out = self.mul(g, gate)?;
        self.hint(
            out,
            FuseHint::GeluMul {
                n,
                up_off,
                inner,
                up_stride,
                gate,
                up,
            },
        );
        Ok(out)
    }

    /// Fused gate/up matvecs + GELU*mul: `out[i] = gelu(W_gate[i]·x) * (W_up[i]·x)`.
    /// Weights may be F16/F32 (matching act) or Q4K with F16 activations (out F16).
    pub fn matvec_gate_up_gelu(
        &mut self,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if sg.rank() != 2 || su != sg || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if du != dg || sx.0[0] != sg.0[1] {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sg.0[0];
        let cols = sg.0[1];
        let out_dt = matvec_weight_act_out(dg, dx, cols)?;
        Ok(self.call(
            vec![gate, up, x],
            Shape(vec![rows]),
            out_dt,
            FuseHint::MatvecGateUpGelu {
                rows,
                cols,
                gate,
                up,
                x,
            },
        ))
    }

    /// PLE gate: `out[i] = gelu(W[i]·x) * ctx[ctx_off + i]` (FUSED_MLP_PLE-shaped, one CALL).
    pub fn matvec_gelu_mul(
        &mut self,
        w: TensorId,
        x: TensorId,
        ctx: TensorId,
        ctx_off: usize,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sc, dc) = self.shape_dtype(ctx)?;
        if sw.rank() != 2 || sx.rank() != 1 || !dc.is_float() {
            return Err(IrError::ShapeMismatch);
        }
        if sx.0[0] != sw.0[1] {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sw.0[0];
        let cols = sw.0[1];
        if sc.numel() < ctx_off + rows {
            return Err(IrError::ShapeMismatch);
        }
        let out_dt = matvec_weight_act_out(dw, dx, cols)?;
        Ok(self.call(
            vec![w, x, ctx],
            Shape(vec![rows]),
            out_dt,
            FuseHint::MatvecGeluMul {
                rows,
                cols,
                ctx_off,
                w,
                x,
                ctx,
            },
        ))
    }

    /// PLE: `u = gelu(W_gate@x)*ctx`; `y = W_proj@u`; `out = scale*(residual + rms(y)*w_norm)`.
    pub fn matvec_gelu_mul_proj_rms_add_scale(
        &mut self,
        w_gate: TensorId,
        x: TensorId,
        ctx: TensorId,
        ctx_off: usize,
        w_proj: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        eps: f32,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(w_gate)?;
        let (sp, dp) = self.shape_dtype(w_proj)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sc, dc) = self.shape_dtype(ctx)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        if sg.rank() != 2 || sp.rank() != 2 || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if !dc.is_float() || dn != dx || dr != dx || dg != dp {
            return Err(IrError::ShapeMismatch);
        }
        let gate_rows = sg.0[0];
        let cols = sg.0[1];
        let proj_rows = sp.0[0];
        if sx.0[0] != cols || sp.0[1] != gate_rows {
            return Err(IrError::ShapeMismatch);
        }
        if sn.0[0] != proj_rows || sr.0[0] != proj_rows {
            return Err(IrError::ShapeMismatch);
        }
        if sc.numel() < ctx_off + gate_rows {
            return Err(IrError::ShapeMismatch);
        }
        let out_dt = matvec_weight_act_out(dg, dx, cols)?;
        Ok(self.call(
            vec![w_gate, x, ctx, w_proj, w_norm, residual],
            Shape(vec![proj_rows]),
            out_dt,
            FuseHint::MatvecGeluMulProjRmsAddScale {
                gate_rows,
                cols,
                proj_rows,
                ctx_off,
                eps,
                scale,
                w_gate,
                x,
                ctx,
                w_proj,
                w_norm,
                residual,
            },
        ))
    }

    /// `y = W@x` then `out = residual + rmsnorm(y)*w_norm`.
    pub fn matvec_rmsnorm_add(
        &mut self,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w_mat)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        if sw.rank() != 2 || sx.rank() != 1 || sn.rank() != 1 || sr.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if dn != dx || dr != dx || sx.0[0] != sw.0[1] || sn.0[0] != sw.0[0] || sr.0[0] != sw.0[0]
        {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sw.0[0];
        let cols = sw.0[1];
        let out_dt = matvec_weight_act_out(dw, dx, cols)?;
        Ok(self.call(
            vec![w_mat, x, w_norm, residual],
            Shape(vec![rows]),
            out_dt,
            FuseHint::MatvecRmsNormAdd {
                rows,
                cols,
                eps,
                w_mat,
                x,
                w_norm,
                residual,
            },
        ))
    }

    /// `y = W@x` then `out = scale * (residual + rmsnorm(y)*w_norm)`.
    pub fn matvec_rmsnorm_add_scale(
        &mut self,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        eps: f32,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w_mat)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        if sw.rank() != 2 || sx.rank() != 1 || sn.rank() != 1 || sr.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if dn != dx || dr != dx || sx.0[0] != sw.0[1] || sn.0[0] != sw.0[0] || sr.0[0] != sw.0[0]
        {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sw.0[0];
        let cols = sw.0[1];
        let out_dt = matvec_weight_act_out(dw, dx, cols)?;
        Ok(self.call(
            vec![w_mat, x, w_norm, residual],
            Shape(vec![rows]),
            out_dt,
            FuseHint::MatvecRmsNormAddScale {
                rows,
                cols,
                eps,
                scale,
                w_mat,
                x,
                w_norm,
                residual,
            },
        ))
    }

    /// Fused Q/K/V matvecs sharing `x`: Metal emits 3 outputs; Call root shape is Q.
    /// Weights may be F16/F32 (matching act) or Q4K with F16 activations (out F16).
    pub fn matvec_qkv(
        &mut self,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        x: TensorId,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(wq)?;
        let (sk, dk) = self.shape_dtype(wk)?;
        let (sv, dv) = self.shape_dtype(wv)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if sq.rank() != 2 || sk.rank() != 2 || sv.rank() != 2 || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if sk != sv || sq.0[1] != sk.0[1] || sx.0[0] != sq.0[1] {
            return Err(IrError::ShapeMismatch);
        }
        let q_rows = sq.0[0];
        let kv_rows = sk.0[0];
        let cols = sq.0[1];
        let out_dt = qkv_weight_dtypes_ok(dq, dk, dv, dx, cols)?;
        Ok(self.call(
            vec![wq, wk, wv, x],
            Shape(vec![q_rows]),
            out_dt,
            FuseHint::MatvecQkv {
                q_rows,
                kv_rows,
                cols,
                wq,
                wk,
                wv,
                x,
            },
        ))
    }

    /// RMSNorm(x,w_norm) fused into dense matvec: LOCAL `x_hat`, `y = W @ x_hat`.
    /// Weight may be Q4K with F16 act/norm (norm stays F16); out is F16.
    pub fn rmsnorm_matvec(
        &mut self,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w_mat)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        if sw.rank() != 2 || sx.rank() != 1 || sn.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if dn != dx {
            return Err(IrError::ShapeMismatch);
        }
        if sx.0[0] != sw.0[1] || sn.0[0] != sx.0[0] {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sw.0[0];
        let cols = sw.0[1];
        let out_dt = matvec_weight_act_out(dw, dx, cols)?;
        Ok(self.call(
            vec![w_mat, x, w_norm],
            Shape(vec![rows]),
            out_dt,
            FuseHint::RmsNormMatvec {
                n: cols,
                eps,
                rows,
                cols,
                x,
                w_norm,
                w_mat,
            },
        ))
    }

    /// RMSNorm into LOCAL `x_hat`, then fused gate/up matvec+GELU.
    pub fn rmsnorm_matvec_gate_up_gelu(
        &mut self,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
        w_norm: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        if sg.rank() != 2 || su != sg || sx.rank() != 1 || sn.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        if du != dg || dn != dx {
            return Err(IrError::ShapeMismatch);
        }
        if sx.0[0] != sg.0[1] || sn.0[0] != sx.0[0] {
            return Err(IrError::ShapeMismatch);
        }
        let rows = sg.0[0];
        let cols = sg.0[1];
        let out_dt = matvec_weight_act_out(dg, dx, cols)?;
        Ok(self.call(
            vec![gate, up, x, w_norm],
            Shape(vec![rows]),
            out_dt,
            FuseHint::RmsNormMatvecGateUpGelu {
                n: cols,
                eps,
                rows,
                cols,
                x,
                w_norm,
                gate,
                up,
            },
        ))
    }

    /// RMSNorm into LOCAL `x_hat`, then fused Q/K/V matvecs (3 outputs).
    pub fn rmsnorm_matvec_qkv(
        &mut self,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        x: TensorId,
        w_norm: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(wq)?;
        let (sk, dk) = self.shape_dtype(wk)?;
        let (sv, dv) = self.shape_dtype(wv)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(w_norm)?;
        if sq.rank() != 2 || sk.rank() != 2 || sv.rank() != 2 || sx.rank() != 1 || sn.rank() != 1
        {
            return Err(IrError::ShapeMismatch);
        }
        if sk != sv || sq.0[1] != sk.0[1] || sx.0[0] != sq.0[1] || sn.0[0] != sx.0[0] {
            return Err(IrError::ShapeMismatch);
        }
        if dn != dx {
            return Err(IrError::ShapeMismatch);
        }
        let q_rows = sq.0[0];
        let kv_rows = sk.0[0];
        let cols = sq.0[1];
        let out_dt = qkv_weight_dtypes_ok(dq, dk, dv, dx, cols)?;
        Ok(self.call(
            vec![wq, wk, wv, x, w_norm],
            Shape(vec![q_rows]),
            out_dt,
            FuseHint::RmsNormMatvecQkv {
                n: cols,
                eps,
                q_rows,
                kv_rows,
                cols,
                x,
                w_norm,
                wq,
                wk,
                wv,
            },
        ))
    }

    pub fn softcap_argmax(&mut self, x: TensorId, cap: f32) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if !dx.is_float() || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        let n = sx.0[0];
        Ok(self.call(
            vec![x],
            Shape(vec![1]),
            DType::F32, // index must stay F32 — half cannot represent vocab ids
            FuseHint::SoftcapArgmax { n, cap, x },
        ))
    }
}

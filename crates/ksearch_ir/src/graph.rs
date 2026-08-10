//! Tinygrad-shaped Graph: primitives only (ALU + movement + reduce).

use crate::{DType, IrError, Shape, TensorId};

#[derive(Clone, Debug)]
pub enum Op {
    Input { shape: Shape, dtype: DType },
    /// Tinygrad-style CONST leaf (broadcastable scalar / filled tensor).
    Const { value: f32, shape: Shape, dtype: DType },
    Add { a: TensorId, b: TensorId },
    Mul { a: TensorId, b: TensorId },
    ScaleConst { x: TensorId, scale: f32 },
    Rsqrt { x: TensorId },
    Tanh { x: TensorId },
    Exp { x: TensorId },
    SumReduce { inp: TensorId, axis: usize },
    MaxReduce { inp: TensorId, axis: usize },
    /// Tinygrad EXPAND: broadcast `inp` to `shape` (numel must divide).
    Expand { inp: TensorId, shape: Shape },
    MulBroadcastRow { left: TensorId, row: TensorId },
    CopySlice {
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
    },
    /// Composed SDPA sugar materialized as one scheduled kernel (Q@Kᵀ→softmax→@V).
    SdpaNaive {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    },
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

    pub fn input(&mut self, shape: Shape, dtype: DType) -> TensorId {
        self.push(Op::Input { shape: shape.clone(), dtype }, shape, dtype)
    }

    pub fn const_f32(&mut self, value: f32, shape: Shape) -> TensorId {
        self.push(
            Op::Const {
                value,
                shape: shape.clone(),
                dtype: DType::F32,
            },
            shape,
            DType::F32,
        )
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
            (DType::Q4K, DType::F32) if sl.0[1] % 256 == 0 => DType::F32,
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
        let (_, dk) = self.shape_dtype(k)?;
        let (_, dv) = self.shape_dtype(v)?;
        if dq != DType::F32
            || dk != DType::F32
            || dv != DType::F32
            || sq.numel() != n_q * hd
            || n_q == 0
            || hd == 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::SdpaNaive {
                q,
                k,
                v,
                meta,
                n_q,
                hd,
                max_t,
            },
            Shape(vec![n_q * hd]),
            DType::F32,
        ))
    }

    /// Tinygrad `x.square()`.
    pub fn square(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        self.mul(x, x)
    }

    /// Tinygrad softcap sugar: `cap * tanh(x / cap)`.
    pub fn softcap(&mut self, x: TensorId, cap: f32) -> Result<TensorId, IrError> {
        let scaled = self.scale_const(x, 1.0 / cap)?;
        let t = self.tanh(scaled)?;
        self.scale_const(t, cap)
    }

    /// Tinygrad gelu (tanh approx): `0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))`.
    pub fn gelu_tanh(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, _) = self.shape_dtype(x)?;
        let x2 = self.mul(x, x)?;
        let x3 = self.mul(x2, x)?;
        let c044 = self.scale_const(x3, 0.044715)?;
        let inner = self.add(x, c044)?;
        let u = self.scale_const(inner, 0.79788456)?;
        let t = self.tanh(u)?;
        let one = self.const_f32(1.0, s);
        let one_plus = self.add(one, t)?;
        let half_x = self.scale_const(x, 0.5)?;
        self.mul(half_x, one_plus)
    }

    /// Tinygrad `nn.RMSNorm`: `x * rsqrt(mean(x^2)+eps) * weight` (rank-1).
    pub fn rmsnorm_expand(
        &mut self,
        x: TensorId,
        w: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        if sx != sw || dx != DType::F32 || dw != DType::F32 || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        let n = sx.0[0];
        let sq = self.square(x)?;
        let sum = self.sum_reduce(sq, 0)?;
        let mean = self.scale_const(sum, 1.0 / n as f32)?;
        let eps_t = self.const_f32(eps, Shape(vec![1]));
        let mean_eps = self.add(mean, eps_t)?;
        let inv = self.rsqrt(mean_eps)?;
        let inv_b = self.expand(inv, sx.clone())?;
        let xn = self.mul(x, inv_b)?;
        self.mul(xn, w)
    }

    /// `gelu(gate) * up[up_off .. up_off+n]` (tinygrad-style expand + slice).
    pub fn gelu_mul_at(
        &mut self,
        gate: TensorId,
        up: TensorId,
        up_off: usize,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        if dg != DType::F32 || du != DType::F32 || sg.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        let n = sg.0[0];
        if su.numel() < up_off + n {
            return Err(IrError::ShapeMismatch);
        }
        let g = self.gelu_tanh(gate)?;
        let up_s = self.copy_slice(up, up_off, 0, n)?;
        self.mul(g, up_s)
    }
}

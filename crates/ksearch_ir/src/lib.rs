//! Graph / kernel IR for Thesis A: ops lower to generated Metal, not hand shaders.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    /// GGML Q4_K packed blocks (256 elems / 144 bytes). Logical shape is still element counts.
    Q4K,
    /// GGML Q5_K packed blocks (256 elems / 176 bytes).
    Q5K,
    /// GGML Q6_K packed blocks (256 elems / 210 bytes).
    Q6K,
    /// GGML Q4_0 packed groups (32 elems / 18 bytes). Used for KV cache rows.
    Q40,
    /// Brain float16 (2 bytes/elem).
    BF16,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::Q4K | DType::Q5K | DType::Q6K | DType::Q40 => 0,
        }
    }

    pub fn msl(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            DType::Q4K | DType::Q5K | DType::Q6K | DType::Q40 => "uchar",
        }
    }
}

/// Bytes per Q4_0 row of `hd` elements (`hd` must be a multiple of 32).
pub fn q40_row_bytes(hd: usize) -> usize {
    assert!(hd % 32 == 0, "Q4_0 hd must be multiple of 32");
    (hd / 32) * 18
}

/// Total bytes for a Q4_0 KV buffer of `max_t` tokens × `hd` elems.
pub fn q40_nbytes(max_t: usize, hd: usize) -> usize {
    max_t * q40_row_bytes(hd)
}

/// Byte length of a Q4_K tensor with `nelem` logical elements (must be multiple of 256).
pub fn q4k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q4_K nelem must be multiple of 256");
    (nelem / 256) * 144
}

/// Byte length of a Q5_K tensor with `nelem` logical elements (must be multiple of 256).
pub fn q5k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q5_K nelem must be multiple of 256");
    (nelem / 256) * 176
}

/// Byte length of a Q6_K tensor with `nelem` logical elements (must be multiple of 256).
pub fn q6k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q6_K nelem must be multiple of 256");
    (nelem / 256) * 210
}

/// Shape as a list of dimensions (concrete for P0; symbolic later).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    /// Device buffer input (binding index assigned at lower).
    Input { shape: Shape, dtype: DType },
    Add { a: TensorId, b: TensorId },
    Mul { a: TensorId, b: TensorId },
    /// Sum-reduce last axis: `[M, K] -> [M]` (P0 matvec building block).
    SumReduce { inp: TensorId, axis: usize },
    /// Broadcast vector `[K]` against rows of `[M, K]` via mul — used with SumReduce for matvec.
    MulBroadcastRow { left: TensorId, row: TensorId },
    /// Fused Q4_K dequant + matvec: `w[M,K] @ x[K] -> y[M]`.
    MatVecQ4K { w: TensorId, x: TensorId },
    /// Fused Q6_K dequant + matvec.
    MatVecQ6K { w: TensorId, x: TensorId },
    /// BF16 weights × F32 vector → F32.
    MatVecBF16 { w: TensorId, x: TensorId },
    /// Fused Q4_K gate∥up matvec + GeLU(gate)*up → [M].
    MatVecQ4KGateUpGelu { gate: TensorId, up: TensorId, x: TensorId },
    /// RMSNorm(x,w) fused into Q4 gate∥up + GeLU.
    MatVecQ4KRmsGateUpGelu { gate: TensorId, up: TensorId, x: TensorId, w: TensorId, inv: TensorId, eps: f32 },
    /// RMSNorm(x,w) fused into Q4_K matvec.
    MatVecQ4KRms { w: TensorId, x: TensorId, nw: TensorId, inv: TensorId, eps: f32 },
    /// `y[i] = x[i] * scale` (scale baked into lowered MSL).
    ScaleConst { x: TensorId, scale: f32 },
    /// `y[i] = x[i] * s[0]` (scalar in buffer).
    ScaleBuf { x: TensorId, s: TensorId },
    /// `y[i] = cap * tanh(x[i] / cap)`.
    Softcap { x: TensorId, cap: f32 },
    /// GeLU(gate) * up (Gemma MLP / PLE). `up_off` indexes into `up`.
    GeluMul {
        gate: TensorId,
        up: TensorId,
        up_off: usize,
    },
    /// Argmax over 1-D; out shape `[1]` as f32 index.
    ArgMax { x: TensorId },
    /// RMSNorm: `y = x * rsqrt(mean(x^2)+eps) * w`.
    RmsNorm { x: TensorId, w: TensorId, eps: f32 },
    /// `out[0] = rsqrt(mean(x^2)+eps)`.
    InvRms { x: TensorId, eps: f32 },
    /// `y = rmsnorm(x,w) + residual` (fused residual).
    RmsNormAdd { x: TensorId, w: TensorId, residual: TensorId, eps: f32 },
    /// `y = scale * (rmsnorm(x,w) + residual)`.
    RmsNormAddScale { x: TensorId, w: TensorId, residual: TensorId, eps: f32, scale: f32 },
    /// Per-head RMSNorm over last dim `hd`; `with_weight` uses `w[hd]`.
    RmsNormPerHead {
        x: TensorId,
        w: TensorId,
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    },
    /// RoPE on `[n_heads, hd]` with cos_sin `[hd]` (cos then sin halves).
    Rope {
        x: TensorId,
        cos_sin: TensorId,
        n_heads: usize,
        hd: usize,
    },
    /// Tiled flash decode attention (MQA/GQA); meta = `[T, start]` u32 viewed as float buffer binding.
    AttnGqa {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    },
    /// Flash decode over Q4_0 KV caches; meta = `[T, start]` (same as AttnGqa).
    AttnGqaQ4 {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    },
    /// Sequence-parallel split flash-Q4 (MWG-style). Output is partials
    /// `n_q * nwg * (hd + 2)` = `[O…, M, L]` per (head, wg).
    AttnGqaQ4Split {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
    },
    /// Reduce MWG partials → final O `[n_q * hd]`.
    AttnGqaQ4Reduce {
        partials: TensorId,
        n_q: usize,
        hd: usize,
        nwg: usize,
    },
    /// Fused flash-Q4 attn + append: score `pos` from f32 `k_new`/`v_new`, prior from Q4 caches,
    /// then quantize-append into caches (one writer TG for MQA). meta = `[T, start]`.
    AttnGqaQ4Fused {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        k_new: TensorId,
        v_new: TensorId,
        meta: TensorId,
        pos: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    },
    /// Q-only fused MQA flash-Q4 for shared-KV layers (no K/V prep, no append).
    /// Bind order: q_raw, q_norm, cos_sin, k_cache, v_cache, meta.
    AttnGqaQ4QFused {
        q: TensorId,
        q_norm: TensorId,
        cos_sin: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
        eps: f32,
    },
    /// Quantize f32 row `src[hd]` to Q4_0 and write at token `pos` into the output cache.
    /// `pos` is a U32[1] buffer (bound as F32[1] input).
    KvAppendQ4 {
        src: TensorId,
        pos: TensorId,
        hd: usize,
        max_t: usize,
    },
    /// Copy `n` f32 elements: `dst[dst_off + i] = src[src_off + i]`.
    CopySlice {
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
    },
    /// Gather one Q4_K row: `y[0..cols] = dequant(w[row]) * scale`.
    GatherQ4KRow {
        w: TensorId,
        row_idx: TensorId,
        cols: usize,
        scale: f32,
    },
    /// Gather one Q5_K row: `y[0..cols] = dequant(w[row]) * scale`.
    GatherQ5KRow {
        w: TensorId,
        row_idx: TensorId,
        cols: usize,
        scale: f32,
    },
    /// Fused softcap + argmax over 1-D.
    SoftcapArgmax { x: TensorId, cap: f32 },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Default, Debug)]
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

    pub fn mul_broadcast_row(&mut self, left: TensorId, row: TensorId) -> Result<TensorId, IrError> {
        let (sl, dl) = self.shape_dtype(left)?;
        let (sr, dr) = self.shape_dtype(row)?;
        if dl != dr || sl.rank() != 2 || sr.rank() != 1 || sl.0[1] != sr.0[0] {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MulBroadcastRow { left, row },
            sl.clone(),
            dl,
        ))
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

    pub fn matvec_q4k(&mut self, w: TensorId, x: TensorId) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if dw != DType::Q4K
            || dx != DType::F32
            || sw.rank() != 2
            || sx.rank() != 1
            || sw.0[1] != sx.0[0]
            || sw.0[1] % 256 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecQ4K { w, x },
            Shape(vec![sw.0[0]]),
            DType::F32,
        ))
    }

    pub fn matvec_q6k(&mut self, w: TensorId, x: TensorId) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if dw != DType::Q6K
            || dx != DType::F32
            || sw.rank() != 2
            || sx.rank() != 1
            || sw.0[1] != sx.0[0]
            || sw.0[1] % 256 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecQ6K { w, x },
            Shape(vec![sw.0[0]]),
            DType::F32,
        ))
    }

    pub fn matvec_bf16(&mut self, w: TensorId, x: TensorId) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if dw != DType::BF16
            || dx != DType::F32
            || sw.rank() != 2
            || sx.rank() != 1
            || sw.0[1] != sx.0[0]
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecBF16 { w, x },
            Shape(vec![sw.0[0]]),
            DType::F32,
        ))
    }

    pub fn matvec_q4k_gate_up_gelu(
        &mut self,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        let (sx, dx) = self.shape_dtype(x)?;
        if dg != DType::Q4K
            || du != DType::Q4K
            || dx != DType::F32
            || sg != su
            || sg.rank() != 2
            || sx.rank() != 1
            || sg.0[1] != sx.0[0]
            || sg.0[1] % 256 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecQ4KGateUpGelu { gate, up, x },
            Shape(vec![sg.0[0]]),
            DType::F32,
        ))
    }


    pub fn matvec_q4k_rms_gate_up_gelu(
        &mut self,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
        w: TensorId,
        inv: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        let (si, di) = self.shape_dtype(inv)?;
        if dg != DType::Q4K
            || du != DType::Q4K
            || dx != DType::F32
            || dw != DType::F32
            || di != DType::F32
            || sg != su
            || sg.rank() != 2
            || sx.rank() != 1
            || sw.numel() != sx.0[0]
            || si.numel() < 1
            || sg.0[1] != sx.0[0]
            || sg.0[1] % 256 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecQ4KRmsGateUpGelu {
                gate,
                up,
                x,
                w,
                inv,
                eps,
            },
            Shape(vec![sg.0[0]]),
            DType::F32,
        ))
    }

    pub fn matvec_q4k_rms(
        &mut self,
        w: TensorId,
        x: TensorId,
        nw: TensorId,
        inv: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sx, dx) = self.shape_dtype(x)?;
        let (sn, dn) = self.shape_dtype(nw)?;
        let (si, di) = self.shape_dtype(inv)?;
        if dw != DType::Q4K
            || dx != DType::F32
            || dn != DType::F32
            || di != DType::F32
            || sw.rank() != 2
            || sx.rank() != 1
            || sn.numel() != sx.0[0]
            || si.numel() < 1
            || sw.0[1] != sx.0[0]
            || sw.0[1] % 256 != 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::MatVecQ4KRms { w, x, nw, inv, eps },
            Shape(vec![sw.0[0]]),
            DType::F32,
        ))
    }

    pub fn scale_const(&mut self, x: TensorId, scale: f32) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        if d != DType::F32 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::ScaleConst { x, scale }, s, d))
    }

    pub fn scale_buf(&mut self, x: TensorId, s: TensorId) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (ss, ds) = self.shape_dtype(s)?;
        if dx != DType::F32 || ds != DType::F32 || ss.numel() < 1 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::ScaleBuf { x, s }, sx, dx))
    }

    pub fn softcap(&mut self, x: TensorId, cap: f32) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        if d != DType::F32 || cap == 0.0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::Softcap { x, cap }, s, d))
    }

    pub fn gelu_mul(&mut self, gate: TensorId, up: TensorId) -> Result<TensorId, IrError> {
        self.gelu_mul_at(gate, up, 0)
    }

    pub fn gelu_mul_at(
        &mut self,
        gate: TensorId,
        up: TensorId,
        up_off: usize,
    ) -> Result<TensorId, IrError> {
        let (sg, dg) = self.shape_dtype(gate)?;
        let (su, du) = self.shape_dtype(up)?;
        if dg != DType::F32
            || du != DType::F32
            || su.numel() < up_off.saturating_add(sg.numel())
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::GeluMul { gate, up, up_off }, sg, dg))
    }

    pub fn argmax(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        if d != DType::F32 || s.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::ArgMax { x }, Shape(vec![1]), DType::F32))
    }

    pub fn rmsnorm(&mut self, x: TensorId, w: TensorId, eps: f32) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        if dx != DType::F32 || dw != DType::F32 || sx != sw {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::RmsNorm { x, w, eps }, sx, dx))
    }

    pub fn inv_rms(&mut self, x: TensorId, eps: f32) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        if dx != DType::F32 || sx.rank() != 1 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(Op::InvRms { x, eps }, Shape(vec![1]), DType::F32))
    }

    pub fn rmsnorm_add(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        if dx != DType::F32 || dw != DType::F32 || dr != DType::F32 || sx != sw || sx != sr {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::RmsNormAdd {
                x,
                w,
                residual,
                eps,
            },
            sx,
            dx,
        ))
    }

    pub fn rmsnorm_add_scale(
        &mut self,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
        eps: f32,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (sx, dx) = self.shape_dtype(x)?;
        let (sw, dw) = self.shape_dtype(w)?;
        let (sr, dr) = self.shape_dtype(residual)?;
        if dx != DType::F32 || dw != DType::F32 || dr != DType::F32 || sx != sw || sx != sr {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::RmsNormAddScale {
                x,
                w,
                residual,
                eps,
                scale,
            },
            sx,
            dx,
        ))
    }

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
        let (_, dw) = self.shape_dtype(w)?;
        if dx != DType::F32
            || dw != DType::F32
            || sx.numel() != n_heads * hd
            || n_heads == 0
            || hd == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::RmsNormPerHead {
                x,
                w,
                n_heads,
                hd,
                eps,
                with_weight,
            },
            sx,
            dx,
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
        let (sc, dc) = self.shape_dtype(cos_sin)?;
        if dx != DType::F32
            || dc != DType::F32
            || sx.numel() != n_heads * hd
            || sc.numel() != hd
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::Rope {
                x,
                cos_sin,
                n_heads,
                hd,
            },
            sx,
            dx,
        ))
    }

    pub fn attn_gqa(
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
        let (_, dm) = self.shape_dtype(meta)?;
        if dq != DType::F32
            || dk != DType::F32
            || dv != DType::F32
            || dm != DType::F32
            || sq.numel() != n_q * hd
            || n_q == 0
            || hd == 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqa {
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

    pub fn attn_gqa_q4(
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
        let (_, dm) = self.shape_dtype(meta)?;
        if dq != DType::F32
            || dk != DType::Q40
            || dv != DType::Q40
            || dm != DType::F32
            || sq.numel() != n_q * hd
            || n_q == 0
            || hd == 0
            || hd % 32 != 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqaQ4 {
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

    /// Split-KV flash-Q4: online softmax over a sequence chunk per workgroup.
    /// Output placeholder sized `n_q * nwg * (hd + 2)` (O, M, L per partition).
    pub fn attn_gqa_q4_split(
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
        let (_, dk) = self.shape_dtype(k)?;
        let (_, dv) = self.shape_dtype(v)?;
        let (_, dm) = self.shape_dtype(meta)?;
        if dq != DType::F32
            || dk != DType::Q40
            || dv != DType::Q40
            || dm != DType::F32
            || sq.numel() != n_q * hd
            || n_q == 0
            || hd == 0
            || hd % 32 != 0
            || max_t == 0
            || nwg == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqaQ4Split {
                q,
                k,
                v,
                meta,
                n_q,
                hd,
                max_t,
                nwg,
            },
            Shape(vec![n_q * nwg * (hd + 2)]),
            DType::F32,
        ))
    }

    /// Merge MWG partials into final attention output `[n_q * hd]`.
    pub fn attn_gqa_q4_reduce(
        &mut self,
        partials: TensorId,
        n_q: usize,
        hd: usize,
        nwg: usize,
    ) -> Result<TensorId, IrError> {
        let (sp, dp) = self.shape_dtype(partials)?;
        if dp != DType::F32
            || sp.numel() != n_q * nwg * (hd + 2)
            || n_q == 0
            || hd == 0
            || hd % 32 != 0
            || nwg == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqaQ4Reduce {
                partials,
                n_q,
                hd,
                nwg,
            },
            Shape(vec![n_q * hd]),
            DType::F32,
        ))
    }

    /// Full-fused MQA: Q/K rmsnorm+RoPE, V rmsnorm, flash over Q4, append at `pos`.
    pub fn attn_gqa_q4_fused(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        k_new: TensorId,
        v_new: TensorId,
        meta: TensorId,
        pos: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (_, dk) = self.shape_dtype(k)?;
        let (_, dv) = self.shape_dtype(v)?;
        let (skn, dkn) = self.shape_dtype(k_new)?;
        let (svn, dvn) = self.shape_dtype(v_new)?;
        let (_, dm) = self.shape_dtype(meta)?;
        let (sp, dp) = self.shape_dtype(pos)?;
        if dq != DType::F32
            || dk != DType::Q40
            || dv != DType::Q40
            || dkn != DType::F32
            || dvn != DType::F32
            || dm != DType::F32
            || dp != DType::F32
            || sq.numel() != n_q * hd
            || skn.numel() < hd
            || svn.numel() < hd
            || sp.numel() < 1
            || n_q == 0
            || hd == 0
            || hd % 32 != 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqaQ4Fused {
                q,
                k,
                v,
                k_new,
                v_new,
                meta,
                pos,
                n_q,
                hd,
                max_t,
            },
            Shape(vec![n_q * hd]),
            DType::F32,
        ))
    }

    /// Q-only fused MQA flash for shared-KV layers.
    pub fn attn_gqa_q4_q_fused(
        &mut self,
        q: TensorId,
        q_norm: TensorId,
        cos_sin: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        meta: TensorId,
        n_q: usize,
        hd: usize,
        max_t: usize,
        eps: f32,
    ) -> Result<TensorId, IrError> {
        let (sq, dq) = self.shape_dtype(q)?;
        let (sqn, dqn) = self.shape_dtype(q_norm)?;
        let (scs, dcs) = self.shape_dtype(cos_sin)?;
        let (_, dkc) = self.shape_dtype(k_cache)?;
        let (_, dvc) = self.shape_dtype(v_cache)?;
        let (_, dm) = self.shape_dtype(meta)?;
        if dq != DType::F32
            || dqn != DType::F32
            || dcs != DType::F32
            || dkc != DType::Q40
            || dvc != DType::Q40
            || dm != DType::F32
            || sq.numel() != n_q * hd
            || sqn.numel() < hd
            || scs.numel() < hd
            || n_q == 0
            || hd == 0
            || hd % 32 != 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::AttnGqaQ4QFused {
                q,
                q_norm,
                cos_sin,
                k_cache,
                v_cache,
                meta,
                n_q,
                hd,
                max_t,
                eps,
            },
            Shape(vec![n_q * hd]),
            DType::F32,
        ))
    }

    /// Quantize `src[hd]` to Q4_0 at token index from `pos` (U32[1] as F32 buffer).
    /// Output is a placeholder; runtime binds the full Q4_0 cache.
    pub fn kv_append_q4(
        &mut self,
        src: TensorId,
        pos: TensorId,
        hd: usize,
        max_t: usize,
    ) -> Result<TensorId, IrError> {
        let (ss, ds) = self.shape_dtype(src)?;
        let (sp, dp) = self.shape_dtype(pos)?;
        if ds != DType::F32
            || dp != DType::F32
            || ss.numel() < hd
            || sp.numel() < 1
            || hd == 0
            || hd % 32 != 0
            || max_t == 0
        {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::KvAppendQ4 {
                src,
                pos,
                hd,
                max_t,
            },
            Shape(vec![max_t * hd]),
            DType::Q40,
        ))
    }

    /// Copy `n` elems from `src` at `src_off` into the output buffer at `dst_off`.
    /// Output shape is a placeholder `[n]` (runtime binds the full destination buffer).
    pub fn copy_slice(
        &mut self,
        src: TensorId,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(src)?;
        if d != DType::F32 || src_off + n > s.numel() || n == 0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::CopySlice {
                src,
                src_off,
                dst_off,
                n,
            },
            Shape(vec![n]),
            DType::F32,
        ))
    }

    /// `w` is Q4_K `[vocab, cols]`; `row_idx` is F32[1] with row as float; out is F32[cols].
    pub fn gather_q4k_row(
        &mut self,
        w: TensorId,
        row_idx: TensorId,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sr, dr) = self.shape_dtype(row_idx)?;
        if dw != DType::Q4K
            || dr != DType::F32
            || sw.rank() != 2
            || sw.0[1] % 256 != 0
            || sr.numel() < 1
        {
            return Err(IrError::ShapeMismatch);
        }
        let cols = sw.0[1];
        Ok(self.push(
            Op::GatherQ4KRow {
                w,
                row_idx,
                cols,
                scale,
            },
            Shape(vec![cols]),
            DType::F32,
        ))
    }

    /// `w` is Q5_K `[vocab, cols]`; `row_idx` is F32[1]; out is F32[cols].
    pub fn gather_q5k_row(
        &mut self,
        w: TensorId,
        row_idx: TensorId,
        scale: f32,
    ) -> Result<TensorId, IrError> {
        let (sw, dw) = self.shape_dtype(w)?;
        let (sr, dr) = self.shape_dtype(row_idx)?;
        if dw != DType::Q5K
            || dr != DType::F32
            || sw.rank() != 2
            || sw.0[1] % 256 != 0
            || sr.numel() < 1
        {
            return Err(IrError::ShapeMismatch);
        }
        let cols = sw.0[1];
        Ok(self.push(
            Op::GatherQ5KRow {
                w,
                row_idx,
                cols,
                scale,
            },
            Shape(vec![cols]),
            DType::F32,
        ))
    }

    pub fn softcap_argmax(&mut self, x: TensorId, cap: f32) -> Result<TensorId, IrError> {
        let (s, d) = self.shape_dtype(x)?;
        if d != DType::F32 || s.rank() != 1 || cap == 0.0 {
            return Err(IrError::ShapeMismatch);
        }
        Ok(self.push(
            Op::SoftcapArgmax { x, cap },
            Shape(vec![1]),
            DType::F32,
        ))
    }

    pub fn shape_dtype(&self, id: TensorId) -> Result<(Shape, DType), IrError> {
        let n = self
            .nodes
            .get(id.0 as usize)
            .ok_or(IrError::BadTensorId)?;
        Ok((n.shape.clone(), n.dtype))
    }

    pub fn node(&self, id: TensorId) -> Result<&Node, IrError> {
        self.nodes.get(id.0 as usize).ok_or(IrError::BadTensorId)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IrError {
    #[error("bad tensor id")]
    BadTensorId,
    #[error("shape / dtype mismatch")]
    ShapeMismatch,
    #[error("bad reduce axis")]
    BadAxis,
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

//! Kernel IR (UOp-lite): axis loops + loads/stores/ALU after scheduling.
//! Thesis A: Graph primitives → schedule → KernelIr → BEAM OptOps → MSL render.

use crate::{DType, Shape, TensorId};

/// Buffer binding in a scheduled kernel (input or output).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufId(pub u32);

/// SSA value inside a kernel body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValId(pub u32);

/// Discrete schedule choices searched by BEAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptSchedule {
    /// Threads per threadgroup (row-parallel F32 matvec / elemwise TG size).
    pub tg: u64,
    /// Vector width for K loads (1, 2, or 4) — F32 path.
    pub vec: u32,
    /// Unroll factor along the K loop (1, 2, 4, 8) — F32 path.
    pub unroll: u32,
    /// Q4_K: simdgroups per threadgroup (1, 2, 4). Ignored for F32.
    pub nsg: u32,
    /// Q4_K: rows per simdgroup (2, 4, 8). Ignored for F32.
    pub nr0: u32,
}

impl Default for OptSchedule {
    fn default() -> Self {
        Self {
            tg: 32,
            vec: 4,
            unroll: 1,
            nsg: 2,
            nr0: 4,
        }
    }
}

impl OptSchedule {
    /// Deliberately weak schedule for BEAM baseline comparison.
    pub fn untuned() -> Self {
        Self {
            tg: 32,
            vec: 1,
            unroll: 1,
            nsg: 1,
            nr0: 2,
        }
    }

    /// Strong default for Q4_K matvec (ggml-style).
    pub fn q4k_default() -> Self {
        Self {
            tg: 64,
            vec: 1,
            unroll: 1,
            nsg: 2,
            nr0: 4,
        }
    }
}

/// One scheduled Metal kernel (before OptOps applied at render).
#[derive(Clone, Debug)]
pub struct ScheduledKernel {
    pub name: String,
    /// Graph inputs in Metal binding order (0..n-1); output is separate.
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub kind: KernelKind,
}

/// Algorithmic shape of a kernel — still *composed* of primops, not a fused catalog op.
#[derive(Clone, Debug)]
pub enum KernelKind {
    /// `out[i] = f(inputs…)` elementwise, fused Add/Mul chains.
    Elementwise {
        n: usize,
        /// Root expression built from inputs (Add/Mul tree).
        expr: ElemExpr,
    },
    /// Dense matvec: `y[r] = sum_k A[r,k]*x[k]` from MulBroadcastRow+SumReduce.
    /// `weight_dtype` may be `F32` or `Q4K` (dequant fused at render — dtype fusion).
    Matvec {
        rows: usize,
        cols: usize,
        matrix: TensorId,
        vector: TensorId,
        weight_dtype: DType,
    },
    /// `out[r] = sum_k inp[r,k]`.
    SumLast {
        rows: usize,
        cols: usize,
        inp: TensorId,
    },
    /// Naive MQA SDPA: Q@Kᵀ → softmax → @V (meta = T, start).
    SdpaNaive {
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
    },
    /// Scheduled fusion of tinygrad RMSNorm expand (square→mean→rsqrt→mul).
    RmsNorm {
        n: usize,
        eps: f32,
        x: TensorId,
        w: TensorId,
    },
    RmsNormAdd {
        n: usize,
        eps: f32,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
    },
    RmsNormAddScale {
        n: usize,
        eps: f32,
        scale: f32,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
    },
    RmsNormPerHead {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
    },
    Rope {
        n_heads: usize,
        hd: usize,
        x: TensorId,
        cos_sin: TensorId,
    },
    GeluMul {
        n: usize,
        up_off: usize,
        gate: TensorId,
        up: TensorId,
    },
    CopySlice {
        src_off: usize,
        dst_off: usize,
        n: usize,
        src: TensorId,
    },
    SoftcapArgmax {
        n: usize,
        cap: f32,
        x: TensorId,
    },
}

/// Elementwise expression over bound inputs (by binding index).
#[derive(Clone, Debug)]
pub enum ElemExpr {
    /// Load input binding `bi` at linear index `gid`.
    Load(usize),
    Add(Box<ElemExpr>, Box<ElemExpr>),
    Mul(Box<ElemExpr>, Box<ElemExpr>),
    /// `scale * inner` (folded ScaleConst).
    Scale(Box<ElemExpr>, f32),
}

/// Fully lowered kernel IR ready for MSL render (+ OptSchedule).
#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub n_inputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub body: KirBody,
}

#[derive(Clone, Debug)]
pub enum KirBody {
    Elementwise { n: usize, expr: ElemExpr },
    Matvec {
        rows: usize,
        cols: usize,
        weight_dtype: DType,
    },
    SumLast { rows: usize, cols: usize },
    SdpaNaive {
        n_q: usize,
        hd: usize,
        max_t: usize,
    },
    /// Fused schedule of tinygrad RMSNorm expand: square→mean→rsqrt→mul.
    RmsNorm { n: usize, eps: f32 },
    RmsNormAdd { n: usize, eps: f32 },
    RmsNormAddScale { n: usize, eps: f32, scale: f32 },
    RmsNormPerHead {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
    },
    Rope { n_heads: usize, hd: usize },
    GeluMul { n: usize, up_off: usize },
    CopySlice {
        src_off: usize,
        dst_off: usize,
        n: usize,
    },
    SoftcapArgmax { n: usize, cap: f32 },
}

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
    pub tg: u64,
    pub vec: u32,
    pub unroll: u32,
    pub nsg: u32,
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
    pub fn untuned() -> Self {
        Self {
            tg: 32,
            vec: 1,
            unroll: 1,
            nsg: 1,
            nr0: 2,
        }
    }

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

/// Sugar / pattern fusion hint (tinygrad CALL region metadata — not a Graph Op catalog).
#[derive(Clone, Debug)]
pub enum FuseHint {
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
    SdpaNaive {
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
    },
    SoftcapArgmax {
        n: usize,
        cap: f32,
        x: TensorId,
    },
}

#[derive(Clone, Debug)]
pub struct ScheduledKernel {
    pub name: String,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub kind: KernelKind,
}

#[derive(Clone, Debug)]
pub enum KernelKind {
    Elementwise { n: usize, expr: ElemExpr },
    Matvec {
        rows: usize,
        cols: usize,
        matrix: TensorId,
        vector: TensorId,
        weight_dtype: DType,
    },
    SumLast {
        rows: usize,
        cols: usize,
        inp: TensorId,
    },
    SdpaNaive {
        n_q: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
    },
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

#[derive(Clone, Debug)]
pub enum ElemExpr {
    Load(usize),
    Add(Box<ElemExpr>, Box<ElemExpr>),
    Mul(Box<ElemExpr>, Box<ElemExpr>),
    Scale(Box<ElemExpr>, f32),
}

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

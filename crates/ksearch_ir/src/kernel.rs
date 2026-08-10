//! Kernel IR (tinygrad UOp-lite): CALL body = stmts/exprs. No named hand kernels.

use crate::{DType, Shape, TensorId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValId(pub u32);

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
            tg: 1,
            vec: 1,
            unroll: 1,
            nsg: 1,
            nr0: 2,
        }
    }
}

impl OptSchedule {
    pub fn untuned() -> Self {
        Self::default()
    }
    pub fn q4k_default() -> Self {
        Self::default()
    }
}

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
pub enum KirLaunch {
    Elementwise { n: usize },
    Rows { rows: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Neg,
    Exp,
    Tanh,
    Rsqrt,
    Sqrt,
    Floor,
}

#[derive(Clone, Debug)]
pub enum KirExpr {
    ConstF32(f32),
    ConstU32(u32),
    Gid,
    ForVar(u32),
    Var(u32),
    /// Logical element load. `Q4K` means packed weight buffer; renderer expands dtype.
    Load {
        buf: u32,
        idx: Box<KirExpr>,
        dtype: DType,
    },
    Bin {
        op: BinOp,
        a: Box<KirExpr>,
        b: Box<KirExpr>,
    },
    Unary {
        op: UnaryOp,
        a: Box<KirExpr>,
    },
    CmpGt {
        a: Box<KirExpr>,
        b: Box<KirExpr>,
    },
}

#[derive(Clone, Debug)]
pub enum KirStmt {
    For {
        id: u32,
        n: usize,
        body: Vec<KirStmt>,
    },
    Let {
        id: u32,
        expr: KirExpr,
    },
    Assign {
        id: u32,
        expr: KirExpr,
    },
    Store {
        buf: u32,
        idx: KirExpr,
        val: KirExpr,
    },
    If {
        cond: KirExpr,
        body: Vec<KirStmt>,
    },
}

#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub n_inputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: KirLaunch,
    pub body: Vec<KirStmt>,
    pub next_id: u32,
}

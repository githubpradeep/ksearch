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
    /// Dense F32 matvec default (LOCAL≈TG, UPCAST≈VEC).
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
    /// Same TG, scalar K (BEAM baseline).
    pub fn untuned() -> Self {
        Self {
            tg: 32,
            vec: 1,
            unroll: 1,
            nsg: 2,
            nr0: 4,
        }
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
    RmsNormPerHeadRope {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    Rope {
        n_heads: usize,
        hd: usize,
        x: TensorId,
        cos_sin: TensorId,
    },
    CopyScale {
        src_off: usize,
        dst_off: usize,
        n: usize,
        scale: f32,
        src: TensorId,
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
    RmsNormPerHeadRope {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    Rope {
        n_heads: usize,
        hd: usize,
        x: TensorId,
        cos_sin: TensorId,
    },
    CopyScale {
        src_off: usize,
        dst_off: usize,
        n: usize,
        scale: f32,
        src: TensorId,
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
    /// One thread per row (`thread_position_in_grid`).
    Rows { rows: usize },
    /// One threadgroup per row; `gid`=row, `lid`=lane (`OptSchedule.tg`).
    RowsParallel { rows: usize, tg: u64 },
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
    Lid,
    ForVar(u32),
    Var(u32),
    /// Mutable uint (thread-local), e.g. K index across ForRange loops.
    UVar(u32),
    /// Logical element load (F32 after dequant).
    Load {
        buf: u32,
        idx: Box<KirExpr>,
        dtype: DType,
    },
    /// `dot(floatN(a[idx_a..]), floatN(b[idx_b..]))` for width 1/2/4 (promote half→float).
    VecMulSum {
        a_buf: u32,
        a_idx: Box<KirExpr>,
        b_buf: u32,
        b_idx: Box<KirExpr>,
        width: u32,
        dtype: DType,
    },
    SimdSum(Box<KirExpr>),
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
    CmpEq {
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
    LetU32 {
        id: u32,
        expr: KirExpr,
    },
    Assign {
        id: u32,
        expr: KirExpr,
    },
    /// `for (; id + limit_off < bound; id += step)` over existing `LetU32` var.
    ForRange {
        id: u32,
        limit_off: KirExpr,
        bound: KirExpr,
        step: KirExpr,
        body: Vec<KirStmt>,
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
    /// Tree reduce `acc` across threadgroup (`tg` lanes). Uses simd_sum when tg≤32.
    ThreadgroupReduce { acc_id: u32, tg: u64 },
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

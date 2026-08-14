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
        // Seed TG; lower clamps to cols/256 (or /32). nr0 amortizes LOCAL x (lm_head).
        Self {
            tg: 32,
            vec: 1,
            unroll: 1,
            nsg: 2,
            nr0: 16,
        }
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
    /// Attn→MLP boundary: `out_x = residual + rms(y)*w_post`, `out_x2 = rms(out_x)*w_ffn` (2 outs).
    RmsNormAddThenRmsNorm {
        n: usize,
        eps: f32,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
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
        n_tok: usize,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    /// Per-head RMSNorm (+ optional RoPE) then pack Q4_0 into KV (one CALL).
    RmsNormPerHeadRopeQ40 {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    /// Per-head RMSNorm (no RoPE) then pack Q4_0 into KV (V append).
    RmsNormPerHeadQ40 {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
    },
    /// Q rms+RoPE (F16) + K rms+RoPE+Q40 + V rms+Q40 in one 32-wide launch.
    RmsNormPerHeadQkvQ40 {
        n_q: usize,
        n_kv: usize,
        hd: usize,
        eps: f32,
        n_tok: usize,
        q: TensorId,
        qw: TensorId,
        cos_sin: TensorId,
        k: TensorId,
        kw: TensorId,
        v: TensorId,
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
        /// Logical load dtype (F16 or Q4K/Q5K/Q6K); store uses Call out dtype (F16 for quant).
        src_dtype: DType,
    },
    /// `out[gid] = scale * src[uint(idx[0]) * n + gid]` (GPU-resident row gather).
    CopyScaleIndexed {
        n: usize,
        scale: f32,
        src: TensorId,
        idx: TensorId,
        src_dtype: DType,
    },
    /// Prefill: `out[t, i] = scale * src[uint(idx[t]) * n + i]` for `t in 0..batch`.
    CopyScaleIndexedBatch {
        n: usize,
        batch: usize,
        scale: f32,
        src: TensorId,
        idx: TensorId,
        src_dtype: DType,
    },
    /// Prefill: `Y = W @ X` for `batch` tokens. `batch>8` lowers as GEMM (weights reused).
    MatvecBatch {
        rows: usize,
        cols: usize,
        batch: usize,
        w: TensorId,
        x: TensorId,
    },
    /// Prefill RMSNorm over `rows` independent vectors of length `n`.
    RmsNormRows {
        n: usize,
        rows: usize,
        eps: f32,
        x: TensorId,
        w: TensorId,
    },
    RmsNormAddRows {
        n: usize,
        rows: usize,
        eps: f32,
        scale: f32,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
    },
    /// Two outputs, `rows` independent tokens.
    RmsNormAddThenRmsNormRows {
        n: usize,
        rows: usize,
        eps: f32,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
    },
    /// Prefill SDPA: `n_tok` query tokens, causal `tlen = meta_tlen + tok`.
    /// K/V logical shape `[max_t, n_kv, hd]` (MQA: `n_kv=1`).
    SdpaNaiveBatch {
        n_q: usize,
        n_kv: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgPartBatch {
        n_q: usize,
        n_kv: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgReduceBatch {
        n_q: usize,
        n_tok: usize,
        hd: usize,
        nwg: usize,
        tmp: TensorId,
    },
    GeluMul {
        n: usize,
        up_off: usize,
        /// 0 = contiguous `up[up_off + gid]`. Else `gid = tok*inner + i`.
        inner: usize,
        up_stride: usize,
        gate: TensorId,
        up: TensorId,
    },
    /// Fused MLP gate/up matvecs + GELU*mul: `out[i] = gelu(dot(W_gate[i],x)) * dot(W_up[i],x)`.
    MatvecGateUpGelu {
        rows: usize,
        cols: usize,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
    },
    /// PLE-style: `out[i] = gelu(W[i]·x) * ctx[ctx_off + i]` (one launch; structure of FUSED_MLP_PLE).
    MatvecGeluMul {
        rows: usize,
        cols: usize,
        ctx_off: usize,
        w: TensorId,
        x: TensorId,
        ctx: TensorId,
    },
    /// Extend MatvecGeluMul with proj + residual rms: `u=gelu(Wg@x)*ctx`; `y=Wp@u`;
    /// `out = scale * (residual + rmsnorm(y)*w_norm)`. One CALL when `u` fits LOCAL.
    MatvecGeluMulProjRmsAddScale {
        gate_rows: usize,
        cols: usize,
        proj_rows: usize,
        ctx_off: usize,
        eps: f32,
        scale: f32,
        w_gate: TensorId,
        x: TensorId,
        ctx: TensorId,
        w_proj: TensorId,
        w_norm: TensorId,
        residual: TensorId,
    },
    /// `y = W@x` then `out = residual + rmsnorm(y)*w` (safe when single-TG / short-K).
    MatvecRmsNormAdd {
        rows: usize,
        cols: usize,
        eps: f32,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
    },
    /// Like MatvecRmsNormAdd with `out = scale * (residual + rmsnorm(y)*w)`.
    MatvecRmsNormAddScale {
        rows: usize,
        cols: usize,
        eps: f32,
        scale: f32,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
    },
    /// Fused Q/K/V matvecs sharing one LOCAL `x`: `Q=Wq@x`, `K=Wk@x`, `V=Wv@x` (3 outputs).
    MatvecQkv {
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        x: TensorId,
    },
    /// RMSNorm(x,w_norm) then dense matvec: LOCAL-stage `x_hat`, `y = W @ x_hat`.
    RmsNormMatvec {
        n: usize,
        eps: f32,
        rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        w_mat: TensorId,
    },
    /// RMSNorm into LOCAL `x_hat`, then fused gate/up matvec+GELU.
    RmsNormMatvecGateUpGelu {
        n: usize,
        eps: f32,
        rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        gate: TensorId,
        up: TensorId,
    },
    /// RMSNorm into LOCAL `x_hat`, then fused Q/K/V matvecs.
    RmsNormMatvecQkv {
        n: usize,
        eps: f32,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
    },
    SdpaNaive {
        n_q: usize,
        n_kv: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        /// K/V Load dtype (F16 or Q40 dequant-at-load).
        kv_dtype: DType,
    },
    /// Partitioned MWG SDPA pass1: per (head, part) online softmax → F32 tmp `(m,l,O)`.
    SdpaMwgPart {
        n_q: usize,
        n_kv: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    /// Partitioned MWG SDPA pass2: log-sum-exp merge of NWG partials → F16 O.
    SdpaMwgReduce {
        n_q: usize,
        hd: usize,
        nwg: usize,
        tmp: TensorId,
    },
    /// Pack F16 activations into Q4_0 blocks (KV append).
    QuantizeQ40 {
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
        n_kv: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgPart {
        n_q: usize,
        n_kv: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgReduce {
        n_q: usize,
        hd: usize,
        nwg: usize,
        tmp: TensorId,
    },
    QuantizeQ40 {
        n: usize,
        src: TensorId,
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
    /// Two outputs: residual stream + ffn-normalized activations.
    RmsNormAddThenRmsNorm {
        n: usize,
        eps: f32,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
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
        n_tok: usize,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    RmsNormPerHeadRopeQ40 {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
        cos_sin: TensorId,
    },
    RmsNormPerHeadQ40 {
        n_heads: usize,
        hd: usize,
        eps: f32,
        with_weight: bool,
        x: TensorId,
        w: TensorId,
    },
    /// Three outputs: Q F16, K Q40, V Q40.
    RmsNormPerHeadQkvQ40 {
        n_q: usize,
        n_kv: usize,
        hd: usize,
        eps: f32,
        n_tok: usize,
        q: TensorId,
        qw: TensorId,
        cos_sin: TensorId,
        k: TensorId,
        kw: TensorId,
        v: TensorId,
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
        /// Logical load dtype (F16 or Q4K/Q5K/Q6K); store uses Call out dtype (F16 for quant).
        src_dtype: DType,
    },
    CopyScaleIndexed {
        n: usize,
        scale: f32,
        src: TensorId,
        idx: TensorId,
        src_dtype: DType,
    },
    CopyScaleIndexedBatch {
        n: usize,
        batch: usize,
        scale: f32,
        src: TensorId,
        idx: TensorId,
        src_dtype: DType,
    },
    MatvecBatch {
        rows: usize,
        cols: usize,
        batch: usize,
        matrix: TensorId,
        vector: TensorId,
        weight_dtype: DType,
    },
    RmsNormRows {
        n: usize,
        rows: usize,
        eps: f32,
        x: TensorId,
        w: TensorId,
    },
    RmsNormAddRows {
        n: usize,
        rows: usize,
        eps: f32,
        scale: f32,
        x: TensorId,
        w: TensorId,
        residual: TensorId,
    },
    RmsNormAddThenRmsNormRows {
        n: usize,
        rows: usize,
        eps: f32,
        y: TensorId,
        w_post: TensorId,
        residual: TensorId,
        w_ffn: TensorId,
    },
    SdpaNaiveBatch {
        n_q: usize,
        n_kv: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgPartBatch {
        n_q: usize,
        n_kv: usize,
        n_tok: usize,
        hd: usize,
        max_t: usize,
        nwg: usize,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        meta: TensorId,
        kv_dtype: DType,
    },
    SdpaMwgReduceBatch {
        n_q: usize,
        n_tok: usize,
        hd: usize,
        nwg: usize,
        tmp: TensorId,
    },
    GeluMul {
        n: usize,
        up_off: usize,
        /// 0 = contiguous `up[up_off + gid]`. Else `gid = tok*inner + i`.
        inner: usize,
        up_stride: usize,
        gate: TensorId,
        up: TensorId,
    },
    MatvecGateUpGelu {
        rows: usize,
        cols: usize,
        gate: TensorId,
        up: TensorId,
        x: TensorId,
        weight_dtype: DType,
    },
    /// `out[i] = gelu(W[i]·x) * ctx[ctx_off + i]`.
    MatvecGeluMul {
        rows: usize,
        cols: usize,
        ctx_off: usize,
        w: TensorId,
        x: TensorId,
        ctx: TensorId,
        weight_dtype: DType,
    },
    /// PLE: gate gelu*ctx → proj → rmsnorm_add_scale (one launch).
    MatvecGeluMulProjRmsAddScale {
        gate_rows: usize,
        cols: usize,
        proj_rows: usize,
        ctx_off: usize,
        eps: f32,
        scale: f32,
        w_gate: TensorId,
        x: TensorId,
        ctx: TensorId,
        w_proj: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        weight_dtype: DType,
    },
    MatvecRmsNormAdd {
        rows: usize,
        cols: usize,
        eps: f32,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        weight_dtype: DType,
    },
    MatvecRmsNormAddScale {
        rows: usize,
        cols: usize,
        eps: f32,
        scale: f32,
        w_mat: TensorId,
        x: TensorId,
        w_norm: TensorId,
        residual: TensorId,
        weight_dtype: DType,
    },
    MatvecQkv {
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        x: TensorId,
        wq_dtype: DType,
        wk_dtype: DType,
        wv_dtype: DType,
    },
    RmsNormMatvec {
        n: usize,
        eps: f32,
        rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        w_mat: TensorId,
        weight_dtype: DType,
    },
    RmsNormMatvecGateUpGelu {
        n: usize,
        eps: f32,
        rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        gate: TensorId,
        up: TensorId,
        weight_dtype: DType,
    },
    RmsNormMatvecQkv {
        n: usize,
        eps: f32,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        x: TensorId,
        w_norm: TensorId,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        wq_dtype: DType,
        wk_dtype: DType,
        wv_dtype: DType,
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
    /// `rows` threadgroups; each TG may cover `OptSchedule.nr0` output rows (`gid`=TG index).
    RowsParallel { rows: usize, tg: u64 },
    /// ggml mul_vec layout: threadsPerThreadgroup=(32, nsg).
    RowsParallelSg { rows: usize, nsg: u64 },
    /// ggml mul_mm layout: grid (ceil(N/32), ceil(M/64)), threads (32, 4).
    MulMm {
        tg_x: u64,
        tg_y: u64,
        tw: u64,
        nsg: u64,
        smem: u64,
    },
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
    /// When `b_from_tg` is `Some(id)`, B is loaded from `threadgroup float tg{id}` (already float).
    VecMulSum {
        a_buf: u32,
        a_idx: Box<KirExpr>,
        b_buf: u32,
        b_idx: Box<KirExpr>,
        width: u32,
        dtype: DType,
        b_from_tg: Option<u32>,
    },
    /// Load from `threadgroup float tg{id}[idx]` (LOCAL staging).
    TgLoad {
        id: u32,
        idx: Box<KirExpr>,
    },
    /// Load from `thread float th{id}[idx]` (per-lane regs).
    ThreadLoad {
        id: u32,
        idx: Box<KirExpr>,
    },
    SimdSum(Box<KirExpr>),
    /// One simdgroup-lane partial for a Q4_K superblock (ggml `mul_vec_q4_K` layout).
    /// `row_base` = row * cols (element index); `ib` = superblock index; `lane` = lid%32.
    /// `b_from_tg=None` → device `x_buf` half activations (oracle-style, no TG staging).
    Q4kCoopFrag {
        w_buf: u32,
        row_base: Box<KirExpr>,
        cols: u32,
        ib: Box<KirExpr>,
        b_from_tg: Option<u32>,
        x_buf: u32,
        /// Added to device `x` index (0 for decode; `tok * cols` for batched prefill).
        x_off: Box<KirExpr>,
        lane: Box<KirExpr>,
    },
    /// One simdgroup-lane partial for a Q6_K superblock (ggml `mul_vec_q6_K` layout).
    /// When `b_from_tg` is `None`, activations are read from device `x_buf` (half).
    /// `x_off` is subtracted from `ib*256` when indexing TG `x` (tiled LOCAL staging).
    Q6kCoopFrag {
        w_buf: u32,
        row_base: Box<KirExpr>,
        cols: u32,
        ib: Box<KirExpr>,
        b_from_tg: Option<u32>,
        x_buf: u32,
        x_off: Box<KirExpr>,
        lane: Box<KirExpr>,
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
    CmpEq {
        a: Box<KirExpr>,
        b: Box<KirExpr>,
    },
    /// `float(uint_expr)` for storing indices in float tg/regs.
    CastU32ToF32(Box<KirExpr>),
    /// `uint(float_expr)` — e.g. SDPA meta tlen/start from F16 buffers.
    CastF32ToU32(Box<KirExpr>),
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
    /// `threadgroup float tg{id}[n];` (static LOCAL; Apple needs no set_threadgroup_memory_length).
    TgDeclF32 { id: u32, n: usize },
    /// Cooperative store into LOCAL: `tg{id}[idx] = val`.
    TgStore {
        id: u32,
        idx: KirExpr,
        val: KirExpr,
    },
    /// `thread float th{id}[n];` — per-lane register file array.
    ThreadDeclF32 { id: u32, n: usize },
    ThreadStore {
        id: u32,
        idx: KirExpr,
        val: KirExpr,
    },
    /// `threadgroup_barrier(mem_flags::mem_threadgroup)`.
    Barrier,
    If {
        cond: KirExpr,
        body: Vec<KirStmt>,
    },
    /// Tree reduce `acc` across threadgroup (`tg` lanes). Uses simd_sum when tg≤32.
    ThreadgroupReduce { acc_id: u32, tg: u64 },
    /// Parallel argmax: `val_id`/`idx_id` become the TG-wide max and its index (as f32).
    ThreadgroupArgmax { val_id: u32, idx_id: u32, tg: u64 },
    /// ggml-style Q4_K: one y-load, accumulate 4 consecutive rows into `acc_ids`.
    /// `b_from_tg=None` → device `x_buf` half activations.
    /// `n_tok==1` writes `acc_ids` (decode). `n_tok>1` loads W once and writes
    /// `thread float th{acc_th}[t*4+r]` (prefill GEMM).
    Q4kCoopNr4 {
        w_buf: u32,
        row0_base: KirExpr,
        cols: u32,
        ib: KirExpr,
        b_from_tg: Option<u32>,
        x_buf: u32,
        x_off: KirExpr,
        lane: KirExpr,
        acc_ids: [u32; 4],
        n_tok: u32,
        x_stride: u32,
        acc_th: Option<u32>,
    },
    /// Dual gate∥up: one y-load, 4 rows × two weight streams.
    Q4kCoopNr4Dual {
        row0_base: KirExpr,
        cols: u32,
        ib: KirExpr,
        b_from_tg: Option<u32>,
        x_buf: u32,
        x_off: KirExpr,
        lane: KirExpr,
        acc_g: [u32; 4],
        acc_u: [u32; 4],
    },
    /// ggml-style Q6_K: one y-load, accumulate 4 consecutive rows into `acc_ids`.
    /// `b_from_tg=None` → device `x_buf` activations (large-K path).
    /// `x_off` subtracts from `ib*256` for tiled TG staging.
    Q6kCoopNr4 {
        w_buf: u32,
        row0_base: KirExpr,
        cols: u32,
        ib: KirExpr,
        b_from_tg: Option<u32>,
        x_buf: u32,
        x_off: KirExpr,
        lane: KirExpr,
        acc_ids: [u32; 4],
        n_tok: u32,
        x_stride: u32,
        acc_th: Option<u32>,
    },
    /// Prefill Q4_K GEMM: Load-expand a 64×32 K-tile, simdgroup MMA (oracle mul_mm tiling).
    Q4kMulMm {
        rows: u32,
        cols: u32,
        batch: u32,
        w_buf: u32,
        x_buf: u32,
        out_buf: u32,
    },
    /// Prefill Q6_K GEMM: same 64×32×32 simdgroup MMA tile as Q4_K.
    Q6kMulMm {
        rows: u32,
        cols: u32,
        batch: u32,
        w_buf: u32,
        x_buf: u32,
        out_buf: u32,
    },
    /// Pack one Q4_0 block (32 elems → 18 bytes) into `dst_buf` at `block` index.
    Q40PackBlock {
        dst_buf: u32,
        block: KirExpr,
        src_buf: u32,
        src_elem: KirExpr,
    },
    /// Pack one Q4_0 block from `thread float th{th_id}[th_off .. +32]`.
    Q40PackFromThread {
        dst_buf: u32,
        block: KirExpr,
        th_id: u32,
        th_off: KirExpr,
    },
    /// Dequant one Q4_0 block into `threadgroup float tg{tg_id}[tg_off .. +32]`.
    Q40DequantToTg {
        tg_id: u32,
        tg_off: KirExpr,
        src_buf: u32,
        src_elem: KirExpr,
    },
    /// Store float4 from Load expand into `tg{tg_id}[tg_off .. +4]` (idx multiple of 4).
    TgStoreF4FromLoad {
        tg_id: u32,
        tg_off: KirExpr,
        src_buf: u32,
        src_elem: KirExpr,
        dtype: DType,
    },
}

#[derive(Clone, Debug)]
pub struct KernelIr {
    pub name: String,
    pub n_inputs: usize,
    /// Device output buffers (`out` / `out0..`). Default 1.
    pub n_outputs: usize,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub launch: KirLaunch,
    pub body: Vec<KirStmt>,
    pub next_id: u32,
}

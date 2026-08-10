//! Scheduled fusion builders: tinygrad sugar → Kernel IR (not Graph catalog Ops).
//!
//! RMSNorm / GeLU / RoPE / softcap are composed of ALU+reduce in sugar; these helpers
//! emit the **fused** Kernel IR region the scheduler would invent (one launch).

use crate::{render_msl, CodegenError, MetalKernelSource};
use ksearch_ir::{DType, KirBody, KernelIr, OptSchedule, Shape};

pub fn render_rmsnorm(n: usize, eps: f32) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_rmsnorm_{n}"),
            n_inputs: 2,
            out_shape: Shape(vec![n]),
            out_dtype: DType::F32,
            body: KirBody::RmsNorm { n, eps },
        },
        OptSchedule::default(),
    )
}

pub fn render_rmsnorm_add(n: usize, eps: f32) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_rms_add_{n}"),
            n_inputs: 3,
            out_shape: Shape(vec![n]),
            out_dtype: DType::F32,
            body: KirBody::RmsNormAdd { n, eps },
        },
        OptSchedule::default(),
    )
}

pub fn render_rmsnorm_add_scale(
    n: usize,
    eps: f32,
    scale: f32,
) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_rms_add_sc_{n}"),
            n_inputs: 3,
            out_shape: Shape(vec![n]),
            out_dtype: DType::F32,
            body: KirBody::RmsNormAddScale { n, eps, scale },
        },
        OptSchedule::default(),
    )
}

pub fn render_rmsnorm_per_head(
    n_heads: usize,
    hd: usize,
    eps: f32,
    with_weight: bool,
) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_rms_ph_{n_heads}_{hd}_{}", with_weight as u8),
            n_inputs: 2,
            out_shape: Shape(vec![n_heads * hd]),
            out_dtype: DType::F32,
            body: KirBody::RmsNormPerHead {
                n_heads,
                hd,
                eps,
                with_weight,
            },
        },
        OptSchedule::default(),
    )
}

pub fn render_rope(n_heads: usize, hd: usize) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_rope_{n_heads}_{hd}"),
            n_inputs: 2,
            out_shape: Shape(vec![n_heads * hd]),
            out_dtype: DType::F32,
            body: KirBody::Rope { n_heads, hd },
        },
        OptSchedule::default(),
    )
}

pub fn render_gelu_mul(n: usize, up_off: usize) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_gelu_mul_{n}_{up_off}"),
            n_inputs: 2,
            out_shape: Shape(vec![n]),
            out_dtype: DType::F32,
            body: KirBody::GeluMul { n, up_off },
        },
        OptSchedule::default(),
    )
}

pub fn render_copy_slice(
    src_off: usize,
    dst_off: usize,
    n: usize,
) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_copy_slice_{n}_{src_off}_{dst_off}"),
            n_inputs: 1,
            out_shape: Shape(vec![n]),
            out_dtype: DType::F32,
            body: KirBody::CopySlice {
                src_off,
                dst_off,
                n,
            },
        },
        OptSchedule::default(),
    )
}

pub fn render_softcap_argmax(n: usize, cap: f32) -> Result<MetalKernelSource, CodegenError> {
    render_msl(
        &KernelIr {
            name: format!("k_sca_{n}"),
            n_inputs: 1,
            out_shape: Shape(vec![1]),
            out_dtype: DType::F32,
            body: KirBody::SoftcapArgmax { n, cap },
        },
        OptSchedule::default(),
    )
}

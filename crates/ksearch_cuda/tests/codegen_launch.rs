//! End-to-end wiring: `ksearch_codegen` → `ksearch_cuda`.
//!
//! The codegen crate emits CUDA source plus a `LaunchHint`; the runtime crate
//! turns that pair into a grid/block/smem configuration and a positional
//! argument list. These tests pin that contract on every kernel in the shared
//! zoo, and run a real device round-trip when a CUDA GPU + `nvcc` are present
//! (otherwise they print a skip notice and pass, so CI stays green on macOS).

use ksearch_codegen::{render_cuda, smoke, OptSchedule};
use ksearch_cuda::{launch_config, CudaContext, KernelArgs, LaunchConfig, DEFAULT_TG};

/// Grid/block coverage and limits for every hint the renderer can produce.
#[test]
fn every_smoke_kernel_maps_to_a_valid_launch() {
    for k in smoke::smoke_kernels() {
        let rendered = render_cuda(&k, OptSchedule::default()).expect("render cuda");
        let cfg = launch_config(&rendered.launch, DEFAULT_TG);

        assert!(
            cfg.threads() >= 1 && cfg.threads() <= 1024,
            "{}: {} threads per block is out of range",
            k.name,
            cfg.threads()
        );
        assert!(cfg.blocks() >= 1, "{}: empty grid", k.name);
        assert_eq!(cfg.grid.2, 1, "{}: 3-D grid is never needed", k.name);

        // The block size must cover the whole output for the flat hints.
        match rendered.launch {
            ksearch_codegen::LaunchHint::Elementwise { n } => {
                assert!(
                    cfg.blocks() * cfg.threads() as u64 >= n as u64,
                    "{}: {n} elements not covered by {:?}",
                    k.name,
                    cfg.grid
                );
            }
            ksearch_codegen::LaunchHint::RowsParallel2D { batch, .. } => {
                assert_eq!(cfg.grid.1 as usize, batch, "{}: batch dim lost", k.name);
            }
            _ => {}
        }

        // One argument per buffer, in `in0..inN, out` order.
        let mut args = KernelArgs::new();
        for _ in 0..rendered.n_inputs {
            args.ptr(0);
        }
        for _ in 0..rendered.n_outputs {
            args.ptr(0);
        }
        assert_eq!(
            args.len(),
            rendered.n_inputs + rendered.n_outputs,
            "{}: argument count",
            k.name
        );
    }
}

/// The tiled GEMM hint keeps its tile shape and shared-memory request.
#[test]
fn mul_mm_launch_keeps_tile_and_smem() {
    let k = smoke::smoke_kernel("k_q4k_mul_mm").expect("smoke kernel");
    let rendered = render_cuda(&k, OptSchedule::default()).expect("render cuda");
    let cfg = launch_config(&rendered.launch, DEFAULT_TG);
    assert_eq!(cfg.block, (256, 1, 1));
    assert_eq!(cfg.smem, 8192);
    assert_eq!(cfg.grid, (1, 1, 1), "64x64 tile over rows=64, batch=64");
}

/// The two-dimensional-block hints must line up with the emitted prologue
/// (`tiisg = threadIdx.x`, `sgitg = threadIdx.y`).
#[test]
fn simdgroup_hint_uses_a_two_dimensional_block() {
    let k = smoke::smoke_kernel("k_q4k_coop").expect("smoke kernel");
    let rendered = render_cuda(&k, OptSchedule::default()).expect("render cuda");
    let cfg = launch_config(&rendered.launch, DEFAULT_TG);
    assert_eq!(cfg.block.0, 32, "warp lanes must be threadIdx.x");
    assert_eq!(cfg.block.1, 4, "simdgroup id must be threadIdx.y");
    assert_eq!(cfg.grid.0, 64, "one block per row");
    assert!(rendered.source.contains("blockIdx.x"));
    assert!(rendered.source.contains("threadIdx.y"));
}

/// Real device round-trip. Runs only where CUDA is usable.
#[test]
fn gpu_copy_round_trip_when_available() {
    let mut ctx = match CudaContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    if ctx.nvcc().is_none() {
        eprintln!("skipped: nvcc not found (install the CUDA toolkit)");
        return;
    }
    eprintln!(
        "CUDA device: {} ({}, {} threads/block max)",
        ctx.info().name,
        ctx.info().sm_arch(),
        ctx.info().max_threads_per_block
    );

    let k = smoke::smoke_kernel("k_copy").expect("smoke kernel");
    let rendered = render_cuda(&k, OptSchedule::default()).expect("render cuda");
    let n = 256usize;
    let bytes_per = 2; // f16
    let f = ctx.load(&rendered).expect("compile + load k_copy");

    let src = ctx.alloc(n * bytes_per).expect("alloc in");
    let dst = ctx.alloc(n * bytes_per).expect("alloc out");

    // `k_copy` is a bit-for-bit copy, so arbitrary f16 patterns are fine.
    let host_in: Vec<u8> = (0..n * bytes_per)
        .map(|i| (i as u8).wrapping_mul(7))
        .collect();
    ctx.copy_to(&src, &host_in).expect("upload");
    ctx.copy_to(&dst, &vec![0u8; n * bytes_per])
        .expect("zero out");

    let mut args = KernelArgs::new();
    args.ptr(src.as_ptr());
    args.ptr(dst.as_ptr());
    ctx.launch(f, &rendered.launch, &mut args).expect("launch");
    ctx.synchronize().expect("sync");

    let mut host_out = vec![0u8; n * bytes_per];
    ctx.copy_from(&dst, &mut host_out).expect("download");
    assert_eq!(host_out, host_in, "k_copy corrupted data");

    // The elementwise hint must cover every element.
    let cfg: LaunchConfig = launch_config(&rendered.launch, DEFAULT_TG);
    assert!(cfg.blocks() * cfg.threads() as u64 >= n as u64);
}

/// The graph → CUDA path the model actually uses (`lower_to_cuda_chip`), not
/// just the hand-built smoke kernels. This is the lowering the CLI's
/// `matvec --backend cuda` and any future `Eng` CUDA path go through.
#[test]
fn lowered_matvec_graph_is_launchable() {
    use ksearch_codegen::lower_to_cuda_chip;
    use ksearch_ir::{DType, Graph, Shape};

    for (dtype, cols) in [(DType::F16, 512usize), (DType::Q4K, 512)] {
        let rows = 256usize;
        let mut g = Graph::new();
        let a = g.input(Shape(vec![rows, cols]), dtype);
        let x = g.input(Shape(vec![cols]), DType::F16);
        let ax = g.mul_broadcast_row(a, x).expect("mul_broadcast_row");
        let y = g.sum_reduce(ax, 1).expect("sum_reduce");

        let src = lower_to_cuda_chip(&g, y, "test-chip").expect("lower to cuda");
        assert!(
            src.source.contains("KS_KERNEL void") && !src.source.contains("[[buffer("),
            "matvec {dtype:?} did not lower to CUDA"
        );
        if dtype == DType::Q4K {
            assert!(
                src.source.contains("ksearch_load_q4k"),
                "Q4K matvec must pull in the Q4K dequant helper"
            );
        }
        assert_eq!(src.n_inputs + src.n_outputs, 3, "matvec takes W, x and out");

        let cfg = launch_config(&src.launch, DEFAULT_TG);
        assert!(cfg.threads() <= 1024, "{dtype:?}: block too large");
        assert!(cfg.blocks() >= 1);
        assert!(
            cfg.grid.0 as usize >= rows / 256,
            "{dtype:?}: grid too small"
        );

        // The runtime marshals `W, x, out` positionally.
        let mut args = KernelArgs::new();
        for _ in 0..(src.n_inputs + src.n_outputs) {
            args.ptr(0);
        }
        assert_eq!(args.pointers().len(), 3);
    }
}

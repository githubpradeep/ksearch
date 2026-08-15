//! Microbench Q4K/Q6K gate_up_gelu and plain matvec on Metal.
use anyhow::Result;
use ksearch_codegen::lower_to_metal_chip;
use ksearch_ir::{DType, Graph, Shape};
use ksearch_metal::MetalContext;

fn time_kernel(
    ctx: &MetalContext,
    name: &str,
    rows: usize,
    cols: usize,
    dual: bool,
    wd: DType,
) -> Result<()> {
    let mut g = Graph::new();
    let (k, n_in) = if dual {
        let wg = g.input(Shape(vec![rows, cols]), wd);
        let wu = g.input(Shape(vec![rows, cols]), wd);
        let x = g.input(Shape(vec![cols]), DType::F16);
        let out = g.matvec_gate_up_gelu(wg, wu, x)?;
        (lower_to_metal_chip(&g, out, &ctx.device_name())?, 3)
    } else {
        let w = g.input(Shape(vec![rows, cols]), wd);
        let x = g.input(Shape(vec![cols]), DType::F16);
        let out = g.matvec_prim(w, x)?;
        (lower_to_metal_chip(&g, out, &ctx.device_name())?, 2)
    };
    println!("{name}: launch={:?}", k.launch);
    let pipe = ctx.compile(&k)?;
    let nbytes = match wd {
        DType::Q4K => ksearch_ir::q4k_nbytes(rows * cols),
        DType::Q6K => ksearch_ir::q6k_nbytes(rows * cols),
        _ => rows * cols * 2,
    };
    let zeros = vec![0u8; nbytes];
    let w0 = ctx.buffer_bytes(&zeros);
    let w1 = ctx.buffer_bytes(&zeros);
    let bx = ctx.buffer_empty_f16(cols);
    let by = ctx.buffer_empty_f16(rows);
    let tg = match &k.launch {
        ksearch_codegen::LaunchHint::RowsParallel { tg, .. } => *tg,
        ksearch_codegen::LaunchHint::RowsParallelSg { nsg, .. } => nsg * 32,
        _ => 64,
    };
    let inputs: Vec<&metal::Buffer> = if dual {
        vec![&w0, &w1, &bx]
    } else {
        vec![&w0, &bx]
    };
    assert_eq!(inputs.len(), n_in);
    for _ in 0..5 {
        let _ = ctx.run(&pipe, &k, &inputs, &by, tg)?;
    }
    let mut samples = Vec::new();
    for _ in 0..20 {
        let ms = ctx.run(&pipe, &k, &inputs, &by, tg)?;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  p10={:.3}ms p50={:.3}ms p90={:.3}ms",
        samples[2], samples[10], samples[18]
    );
    Ok(())
}

fn main() -> Result<()> {
    let ctx = MetalContext::new()?;
    println!("device={}", ctx.device_name());
    // MLP gate/up
    time_kernel(&ctx, "gate_up 6144x1536", 6144, 1536, true, DType::Q4K)?;
    // MLP down (Gemma Q4_K_M uses Q6_K); include large-K 12288 (tiled TG).
    time_kernel(&ctx, "matvec Q6K 1536x6144", 1536, 6144, false, DType::Q6K)?;
    time_kernel(&ctx, "matvec Q6K 1536x12288", 1536, 12288, false, DType::Q6K)?;
    time_kernel(&ctx, "matvec Q4K 1536x6144", 1536, 6144, false, DType::Q4K)?;
    // Attn o_proj-ish
    time_kernel(&ctx, "matvec Q4K 1536x1536", 1536, 1536, false, DType::Q4K)?;
    Ok(())
}

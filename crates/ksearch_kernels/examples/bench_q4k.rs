//! Microbench Q4K gate_up_gelu and plain matvec on Metal.
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
) -> Result<()> {
    let mut g = Graph::new();
    let (k, n_in) = if dual {
        let wg = g.input(Shape(vec![rows, cols]), DType::Q4K);
        let wu = g.input(Shape(vec![rows, cols]), DType::Q4K);
        let x = g.input(Shape(vec![cols]), DType::F16);
        let out = g.matvec_gate_up_gelu(wg, wu, x)?;
        (lower_to_metal_chip(&g, out, &ctx.device_name())?, 3)
    } else {
        let w = g.input(Shape(vec![rows, cols]), DType::Q4K);
        let x = g.input(Shape(vec![cols]), DType::F16);
        let out = g.matvec_prim(w, x)?;
        (lower_to_metal_chip(&g, out, &ctx.device_name())?, 2)
    };
    println!("{name}: launch={:?}", k.launch);
    let pipe = ctx.compile(&k)?;
    let bpr = cols / 256;
    let nbytes = rows * bpr * 144;
    let zeros = vec![0u8; nbytes];
    let w0 = ctx.buffer_bytes(&zeros);
    let w1 = ctx.buffer_bytes(&zeros);
    let bx = ctx.buffer_empty_f16(cols);
    let by = ctx.buffer_empty_f16(rows);
    let tg = match k.launch {
        ksearch_codegen::LaunchHint::RowsParallel { tg, .. } => tg,
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
    time_kernel(&ctx, "gate_up 6144x1536", 6144, 1536, true)?;
    // MLP down
    time_kernel(&ctx, "matvec 1536x6144", 1536, 6144, false)?;
    // Attn o_proj-ish
    time_kernel(&ctx, "matvec 1536x1536", 1536, 1536, false)?;
    Ok(())
}

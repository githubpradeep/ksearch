use anyhow::Result;
use ksearch_compute::Context;
use ksearch_gguf::{f32_to_f16, quantize_f32_to_q4k};
use ksearch_ir::DType;
use ksearch_kernels::Eng;
use std::time::Instant;

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect()
}

fn main() -> Result<()> {
    let ctx = Context::new()?;
    let mut eng = Eng::new();
    let cols = 1536usize;
    let batch = 2usize;
    // Don't fill all - too big. Use random q4 from small pattern repeated - quantize needs full.
    // Use smaller proxy: 64k rows
    let rows = 65536usize;
    let mut w = vec![0f32; rows * cols];
    for i in 0..w.len() {
        w[i] = ((i % 200) as f32 - 100.) * 0.01;
    }
    let mut x = vec![0f32; batch * cols];
    for i in 0..x.len() {
        x[i] = ((i % 50) as f32 - 25.) * 0.04;
    }
    let bw = ctx.buffer_bytes(&quantize_f32_to_q4k(&w));
    let bx = ctx.buffer_bytes(&f16_bytes(&x));
    let by = ctx.buffer_empty_f16(batch * rows);
    let by1 = ctx.buffer_empty_f16(rows);
    for _ in 0..2 {
        eng.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bw, &bx, &by)?;
        ctx.synchronize()?;
    }
    let t0 = Instant::now();
    for _ in 0..10 {
        eng.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bw, &bx, &by)?;
        ctx.synchronize()?;
    }
    println!(
        "ext batch=2 65k: {:.2} ms",
        t0.elapsed().as_secs_f64() * 1e3 / 10.
    );
    let t0 = Instant::now();
    for _ in 0..10 {
        for t in 0..batch {
            eng.matvec_wd_at(&ctx, rows, cols, DType::Q4K, &bw, &bx, t * cols, &by1, 0)?;
        }
        ctx.synchronize()?;
    }
    println!(
        "2x coop 65k: {:.2} ms",
        t0.elapsed().as_secs_f64() * 1e3 / 10.
    );
    Ok(())
}

//! Compare fused gate∥up+GeLU ext vs separate matvec_batch + gelu_mul (batch 2).
use anyhow::Result;
use ksearch_compute::{Buffer, Context};
use ksearch_gguf::{f16_to_f32, f32_to_f16, quantize_f32_to_q4k};
use ksearch_ir::DType;
use ksearch_kernels::Eng;

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect()
}

fn read_f16(ctx: &Context, buf: &Buffer, n: usize) -> Vec<f32> {
    ctx.synchronize().ok();
    ctx.read_u16(buf, n)
        .into_iter()
        .map(f16_to_f32)
        .collect()
}

fn stats(name: &str, a: &[f32], b: &[f32]) {
    let mut max = 0.0f32;
    let mut rms = 0.0f64;
    let mut n_bad = 0usize;
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        if e > max {
            max = e;
        }
        if e > 0.05 {
            n_bad += 1;
        }
        rms += (e as f64) * (e as f64);
    }
    rms = (rms / a.len().max(1) as f64).sqrt();
    println!(
        "  {name}: n={} max|err|={max:.4e} rms={rms:.4e} n>|0.05|={n_bad}",
        a.len()
    );
}

fn main() -> Result<()> {
    let ctx = Context::new()?;
    let mut eng = Eng::new();
    let rows = 2048usize;
    let cols = 1536usize;
    let batch = 2usize;
    let mut wg = vec![0f32; rows * cols];
    let mut wu = vec![0f32; rows * cols];
    let mut x = vec![0f32; batch * cols];
    for r in 0..rows {
        for c in 0..cols {
            wg[r * cols + c] = (((r * 17 + c * 13) % 200) as f32 - 100.0) * 0.02;
            wu[r * cols + c] = (((r * 19 + c * 11) % 200) as f32 - 100.0) * 0.02;
        }
    }
    for t in 0..batch {
        for c in 0..cols {
            x[t * cols + c] = (((t * 11 + c * 7) % 50) as f32 - 25.0) * 0.04;
        }
    }
    let wqg = quantize_f32_to_q4k(&wg);
    let wqu = quantize_f32_to_q4k(&wu);
    let bwg = ctx.buffer_bytes(&wqg);
    let bwu = ctx.buffer_bytes(&wqu);
    let bx = ctx.buffer_bytes(&f16_bytes(&x));
    let by_f = ctx.buffer_empty_f16(batch * rows);
    let byg = ctx.buffer_empty_f16(batch * rows);
    let byu = ctx.buffer_empty_f16(batch * rows);
    let by_s = ctx.buffer_empty_f16(batch * rows);

    eng.matvec_gate_up_gelu_batch_wd(
        &ctx, rows, cols, batch, DType::Q4K, &bwg, &bwu, &bx, &by_f,
    )?;
    eng.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bwg, &bx, &byg)?;
    eng.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bwu, &bx, &byu)?;
    eng.gelu_mul(&ctx, batch * rows, &byg, &byu, &by_s)?;
    ctx.synchronize()?;
    let fused = read_f16(&ctx, &by_f, batch * rows);
    let sep = read_f16(&ctx, &by_s, batch * rows);
    stats("fused vs separate (F16 gate/up then gelu)", &fused, &sep);

    let bx1 = ctx.buffer_empty_f16(cols);
    let by1 = ctx.buffer_empty_f16(rows);
    let mut coop = vec![0f32; batch * rows];
    for t in 0..batch {
        eng.copy_scale_wd(
            &ctx,
            cols,
            1.0,
            DType::F16,
            &bx,
            t * cols * DType::F16.size_bytes(),
            &bx1,
            0,
        )?;
        eng.matvec_gate_up_gelu_wd(&ctx, rows, cols, DType::Q4K, &bwg, &bwu, &bx1, &by1)?;
        ctx.synchronize()?;
        let row = read_f16(&ctx, &by1, rows);
        coop[t * rows..(t + 1) * rows].copy_from_slice(&row);
    }
    stats("fused vs decode-style coop", &fused, &coop);
    stats("separate vs decode-style coop", &sep, &coop);
    Ok(())
}

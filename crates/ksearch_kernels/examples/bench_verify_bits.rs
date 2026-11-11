use anyhow::Result;
use ksearch_compute::{Buffer, Context};
use ksearch_gguf::{f32_to_f16, quantize_f32_to_q4k};
use ksearch_ir::DType;
use ksearch_kernels::Eng;
use std::time::Instant;

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes()).collect()
}
fn rand_f16(n: usize, seed: u32) -> Vec<u8> {
    let mut v = vec![0f32; n];
    let mut s = seed;
    for x in &mut v {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = ((s >> 16) as f32 / 65535.0) * 0.4 - 0.2;
    }
    f16_bytes(&v)
}
fn write_meta(ctx: &Context, buf: &Buffer, tlen: u32, start: u32, win: u32) {
    // [tlen, start, window] in f32 at element offset 0 (the model's per-layer
    // attention metadata layout).
    ctx.write_f32_at(buf, 0, &[tlen as f32, start as f32, win as f32]);
}

fn main() -> Result<()> {
    let ctx = Context::new()?;
    let mut eng = Eng::new();
    let n_q = 8usize;
    let hd = 256usize;
    let n_tok = 2usize;
    let max_t = 64usize;
    let pos0 = 32usize;
    let qn = n_tok * n_q * hd;
    let kn = max_t * hd;
    let bq = ctx.buffer_bytes(&rand_f16(qn, 11));
    let bk_f = ctx.buffer_bytes(&rand_f16(kn, 12));
    let bv_f = ctx.buffer_bytes(&rand_f16(kn, 13));
    let bk = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(max_t, hd));
    let bv = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(max_t, hd));
    eng.quantize_q40(&ctx, kn, &bk_f, &bk, 0)?;
    eng.quantize_q40(&ctx, kn, &bv_f, &bv, 0)?;
    let meta = ctx.buffer_empty_bytes(32);
    let tmp = ctx.buffer_empty_f32(n_tok * n_q * Eng::SDPA_MWG_NWG * (hd + 2));
    let o = ctx.buffer_empty_f16(qn);
    let q1 = ctx.buffer_empty_f16(n_q * hd);
    let o1 = ctx.buffer_empty_f16(n_q * hd);
    write_meta(&ctx, &meta, (pos0 + 1) as u32, 0, max_t as u32);
    let last = (pos0 + n_tok) as u32;
    // warmup
    for _ in 0..5 {
        eng.sdpa_hybrid_kv_batch(&ctx, n_q, 1, n_tok, hd, max_t, last.min(max_t as u32), DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o)?;
        ctx.synchronize()?;
    }
    let n = 50;
    let t0 = Instant::now();
    for _ in 0..n {
        eng.sdpa_hybrid_kv_batch(&ctx, n_q, 1, n_tok, hd, max_t, last.min(max_t as u32), DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o)?;
        ctx.synchronize()?;
    }
    println!("sdpa batch=2 shared: {:.3} ms", t0.elapsed().as_secs_f64()*1e3/n as f64);

    std::env::set_var("KSEARCH_SDPA_SHARED", "0");
    // force new key by using different eng cache - need new Eng
    let mut eng2 = Eng::new();
    for _ in 0..5 {
        eng2.sdpa_hybrid_kv_batch(&ctx, n_q, 1, n_tok, hd, max_t, last.min(max_t as u32), DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o)?;
        ctx.synchronize()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        eng2.sdpa_hybrid_kv_batch(&ctx, n_q, 1, n_tok, hd, max_t, last.min(max_t as u32), DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o)?;
        ctx.synchronize()?;
    }
    println!("sdpa batch=2 noshare: {:.3} ms", t0.elapsed().as_secs_f64()*1e3/n as f64);

    // serial 2x decode sdpa
    let mut eng3 = Eng::new();
    std::env::remove_var("KSEARCH_SDPA_SHARED");
    for _ in 0..5 {
        for t in 0..n_tok {
            let abs = (pos0 + t + 1) as u32;
            write_meta(&ctx, &meta, abs.min(max_t as u32), abs.saturating_sub(max_t as u32), 0);
            eng3.copy_slice(&ctx, n_q*hd, &bq, t*n_q*hd, &q1, 0)?;
            eng3.sdpa_hybrid_kv(&ctx, n_q, 1, hd, max_t, abs.min(max_t as u32), DType::Q40, &q1, &bk, &bv, &meta, &tmp, &o1)?;
        }
        ctx.synchronize()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        for t in 0..n_tok {
            let abs = (pos0 + t + 1) as u32;
            write_meta(&ctx, &meta, abs.min(max_t as u32), abs.saturating_sub(max_t as u32), 0);
            eng3.copy_slice(&ctx, n_q*hd, &bq, t*n_q*hd, &q1, 0)?;
            eng3.sdpa_hybrid_kv(&ctx, n_q, 1, hd, max_t, abs.min(max_t as u32), DType::Q40, &q1, &bk, &bv, &meta, &tmp, &o1)?;
        }
        ctx.synchronize()?;
    }
    println!("sdpa 2x serial decode: {:.3} ms", t0.elapsed().as_secs_f64()*1e3/n as f64);

    // FFN gate_up
    let rows = 2048usize;
    let cols = 1536usize;
    let batch = 2usize;
    let mut wg = vec![0f32; rows*cols];
    let mut wu = vec![0f32; rows*cols];
    for i in 0..wg.len() { wg[i] = ((i%200) as f32 - 100.)*0.02; wu[i] = (((i*3)%200) as f32 - 100.)*0.02; }
    let mut x = vec![0f32; batch*cols];
    for i in 0..x.len() { x[i] = ((i%50) as f32 - 25.)*0.04; }
    let bwg = ctx.buffer_bytes(&quantize_f32_to_q4k(&wg));
    let bwu = ctx.buffer_bytes(&quantize_f32_to_q4k(&wu));
    let bx = ctx.buffer_bytes(&f16_bytes(&x));
    let by = ctx.buffer_empty_f16(batch*rows);
    let byg = ctx.buffer_empty_f16(batch*rows);
    let byu = ctx.buffer_empty_f16(batch*rows);
    let bys = ctx.buffer_empty_f16(batch*rows);
    let mut eng4 = Eng::new();
    for _ in 0..3 {
        eng4.matvec_gate_up_gelu_batch_wd(&ctx, rows, cols, batch, DType::Q4K, &bwg, &bwu, &bx, &by)?;
        ctx.synchronize()?;
    }
    let t0 = Instant::now();
    for _ in 0..20 {
        eng4.matvec_gate_up_gelu_batch_wd(&ctx, rows, cols, batch, DType::Q4K, &bwg, &bwu, &bx, &by)?;
        ctx.synchronize()?;
    }
    println!("gate_up fused batch2: {:.3} ms", t0.elapsed().as_secs_f64()*1e3/20.0);
    let t0 = Instant::now();
    for _ in 0..20 {
        eng4.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bwg, &bx, &byg)?;
        eng4.matvec_batch(&ctx, rows, cols, batch, DType::Q4K, &bwu, &bx, &byu)?;
        eng4.gelu_mul(&ctx, batch*rows, &byg, &byu, &bys)?;
        ctx.synchronize()?;
    }
    println!("gate+up+gelu separate: {:.3} ms", t0.elapsed().as_secs_f64()*1e3/20.0);
    Ok(())
}

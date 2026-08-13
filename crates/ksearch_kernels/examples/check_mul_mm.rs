//! Compare prefill Q4_K mul_mm (`matvec_batch` with batch>8) against decode GEMV.
use anyhow::Result;
use ksearch_gguf::{dequant_type_to_f32, f16_to_f32, f32_to_f16, ggml_type, quantize_f32_to_q4k};
use ksearch_ir::DType;
use ksearch_kernels::Eng;
use ksearch_metal::MetalContext;

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f16(ctx: &MetalContext, buf: &metal::Buffer, n: usize) -> Vec<f32> {
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

fn check(ctx: &MetalContext, eng: &mut Eng, rows: usize, cols: usize, batch: usize) -> Result<()> {
    let mut w = vec![0f32; rows * cols];
    let mut x = vec![0f32; batch * cols];
    for r in 0..rows {
        for c in 0..cols {
            w[r * cols + c] = (((r * 17 + c * 13) % 200) as f32 - 100.0) * 0.02;
        }
    }
    for t in 0..batch {
        for c in 0..cols {
            x[t * cols + c] = (((t * 11 + c * 7) % 50) as f32 - 25.0) * 0.04;
        }
    }
    let wq = quantize_f32_to_q4k(&w);
    let wd = dequant_type_to_f32(ggml_type::Q4_K, &wq, w.len());
    let mut cpu = vec![0f32; batch * rows];
    for t in 0..batch {
        for r in 0..rows {
            let mut s = 0.0f32;
            for c in 0..cols {
                s += wd[r * cols + c] * x[t * cols + c];
            }
            cpu[t * rows + r] = s;
        }
    }

    let bw = ctx.buffer_bytes(&wq);
    let bx = ctx.buffer_bytes(&f16_bytes(&x));
    let by_mm = ctx.buffer_empty_f16(batch * rows);
    let by_mv = ctx.buffer_empty_f16(batch * rows);

    eng.matvec_batch(ctx, rows, cols, batch, DType::Q4K, &bw, &bx, &by_mm)?;
    for t in 0..batch {
        eng.matvec_wd_at(
            ctx,
            rows,
            cols,
            DType::Q4K,
            &bw,
            &bx,
            t * cols,
            &by_mv,
            t * rows,
        )?;
    }
    ctx.synchronize()?;

    let mm = read_f16(ctx, &by_mm, batch * rows);
    let mv = read_f16(ctx, &by_mv, batch * rows);
    println!("shape {rows}x{cols} batch={batch}");
    stats("mul_mm vs cpu", &mm, &cpu);
    stats("gemv   vs cpu", &mv, &cpu);
    stats("mul_mm vs gemv", &mm, &mv);
    if !mm.is_empty() {
        println!(
            "  sample mm[0..4]={:?}  mv[0..4]={:?}  cpu[0..4]={:?}",
            &mm[..4.min(mm.len())],
            &mv[..4.min(mv.len())],
            &cpu[..4.min(cpu.len())]
        );
    }
    Ok(())
}

fn ones_f16(n: usize) -> Vec<u8> {
    f16_bytes(&vec![1.0f32; n])
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

fn write_meta(buf: &metal::Buffer, tlen: u32, start: u32, win: u32) {
    let ptr = buf.contents() as *mut f32;
    unsafe {
        *ptr = tlen as f32;
        *ptr.add(1) = start as f32;
        *ptr.add(2) = win as f32;
    }
}

fn check_qkv(ctx: &MetalContext, eng: &mut Eng, n_q: usize, n_kv: usize, hd: usize, n_tok: usize) -> Result<()> {
    let eps = 1e-6f32;
    let qn = n_tok * n_q * hd;
    let kn = n_tok * n_kv * hd;
    let bq = ctx.buffer_bytes(&rand_f16(qn, 1));
    let bk = ctx.buffer_bytes(&rand_f16(kn, 2));
    let bv = ctx.buffer_bytes(&rand_f16(kn, 3));
    let qw = ctx.buffer_bytes(&ones_f16(hd));
    let kw = ctx.buffer_bytes(&ones_f16(hd));
    let mut rope = vec![0f32; n_tok * hd];
    let half = hd / 2;
    for t in 0..n_tok {
        for i in 0..half {
            rope[t * hd + i] = 1.0;
            rope[t * hd + half + i] = 0.0;
        }
    }
    let br = ctx.buffer_bytes(&f32_bytes(&rope));
    let q_b = ctx.buffer_empty_f16(qn);
    let q_s = ctx.buffer_empty_f16(qn);
    let kv_bytes = ksearch_ir::q40_nbytes(n_tok, hd);
    let kk_b = ctx.buffer_empty_bytes(kv_bytes);
    let kv_b = ctx.buffer_empty_bytes(kv_bytes);
    let kk_s = ctx.buffer_empty_bytes(kv_bytes);
    let kv_s = ctx.buffer_empty_bytes(kv_bytes);

    eng.rmsnorm_per_head_qkv_q40_batch(
        ctx, n_q, n_kv, hd, n_tok, eps, &bq, &qw, &br, 0, &bk, &kw, &bv, &q_b, &kk_b, &kv_b, 0,
    )?;
    for t in 0..n_tok {
        let q_off = t * n_q * hd;
        let k_off = t * n_kv * hd;
        let kv_off = t * ksearch_ir::q40_row_bytes(hd);
        eng.rmsnorm_per_head_qkv_q40_row(
            ctx, n_q, n_kv, hd, eps, &bq, q_off, &qw, &br, t * hd, &bk, k_off, &kw, &bv, k_off,
            &q_s, q_off, &kk_s, &kv_s, kv_off,
        )?;
    }
    ctx.synchronize()?;
    let qb = read_f16(ctx, &q_b, qn);
    let qs = read_f16(ctx, &q_s, qn);
    println!("qkv n_q={n_q} n_kv={n_kv} hd={hd} n_tok={n_tok}");
    stats("Q batch vs serial", &qb, &qs);

    let kb = unsafe { std::slice::from_raw_parts(kk_b.contents() as *const u8, kv_bytes) };
    let ks = unsafe { std::slice::from_raw_parts(kk_s.contents() as *const u8, kv_bytes) };
    let ndiff = kb.iter().zip(ks).filter(|(a, b)| a != b).count();
    println!("  K Q40 bytes differ={ndiff}/{kv_bytes}");
    let vb = unsafe { std::slice::from_raw_parts(kv_b.contents() as *const u8, kv_bytes) };
    let vs = unsafe { std::slice::from_raw_parts(kv_s.contents() as *const u8, kv_bytes) };
    let ndiff = vb.iter().zip(vs).filter(|(a, b)| a != b).count();
    println!("  V Q40 bytes differ={ndiff}/{kv_bytes}");
    Ok(())
}

fn check_sdpa(
    ctx: &MetalContext,
    eng: &mut Eng,
    n_q: usize,
    hd: usize,
    n_tok: usize,
    max_t: usize,
    pos0: usize,
    win: u32,
) -> Result<()> {
    let qn = n_tok * n_q * hd;
    let kn = max_t * hd;
    let bq = ctx.buffer_bytes(&rand_f16(qn, 11));
    let bk_f = ctx.buffer_bytes(&rand_f16(kn, 12));
    let bv_f = ctx.buffer_bytes(&rand_f16(kn, 13));
    let bk = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(max_t, hd));
    let bv = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(max_t, hd));
    eng.quantize_q40(ctx, kn, &bk_f, &bk, 0)?;
    eng.quantize_q40(ctx, kn, &bv_f, &bv, 0)?;
    let meta = ctx.device.new_buffer(32, metal::MTLResourceOptions::StorageModeShared);
    let tmp = ctx.buffer_empty_f32(n_tok * n_q * Eng::SDPA_MWG_NWG * (hd + 2));
    let o_b = ctx.buffer_empty_f16(qn);
    let o_s = ctx.buffer_empty_f16(qn);
    let q1 = ctx.buffer_empty_f16(n_q * hd);
    let o1 = ctx.buffer_empty_f16(n_q * hd);

    ctx.synchronize()?;
    write_meta(&meta, (pos0 + 1) as u32, 0, win);
    let last_tlen = (pos0 + n_tok) as u32;
    eng.sdpa_hybrid_kv_batch(
        ctx, n_q, n_tok, hd, max_t, last_tlen.min(win), DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o_b,
    )?;

    for t in 0..n_tok {
        let abs = (pos0 + t + 1) as u32;
        let tlen = abs.min(win);
        let start = abs - tlen;
        ctx.synchronize()?;
        write_meta(&meta, tlen, start, 0);
        eng.copy_slice(ctx, n_q * hd, &bq, t * n_q * hd, &q1, 0)?;
        eng.sdpa_hybrid_kv(
            ctx, n_q, hd, max_t, tlen, DType::Q40, &q1, &bk, &bv, &meta, &tmp, &o1,
        )?;
        eng.copy_slice(ctx, n_q * hd, &o1, 0, &o_s, t * n_q * hd)?;
    }
    ctx.synchronize()?;
    let ob = read_f16(ctx, &o_b, qn);
    let os = read_f16(ctx, &o_s, qn);
    println!("sdpa n_q={n_q} hd={hd} n_tok={n_tok} pos0={pos0} max_t={max_t} win={win} last_tlen={last_tlen}");
    stats("O batch vs serial", &ob, &os);
    if !ob.is_empty() {
        println!(
            "  sample b[0..4]={:?} s[0..4]={:?}",
            &ob[..4.min(ob.len())],
            &os[..4.min(os.len())]
        );
    }
    Ok(())
}

/// Needle at KV pos 0: K[0]=Q, V[0]=1, other V=0. O should stay ~1 if early keys survive.
fn check_needle(
    ctx: &MetalContext,
    eng: &mut Eng,
    n_q: usize,
    hd: usize,
    tlen: usize,
) -> Result<()> {
    let qn = n_q * hd;
    let kn = tlen * hd;
    let mut q = vec![0f32; qn];
    let mut k = vec![0f32; kn];
    let mut v = vec![0f32; kn];
    for i in 0..qn {
        q[i] = 1.0;
    }
    for i in 0..hd {
        k[i] = 1.0;
        v[i] = 1.0;
    }
    let bq = ctx.buffer_bytes(&f16_bytes(&q));
    let bk_f = ctx.buffer_bytes(&f16_bytes(&k));
    let bv_f = ctx.buffer_bytes(&f16_bytes(&v));
    let bk = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(tlen, hd));
    let bv = ctx.buffer_empty_bytes(ksearch_ir::q40_nbytes(tlen, hd));
    eng.quantize_q40(ctx, kn, &bk_f, &bk, 0)?;
    eng.quantize_q40(ctx, kn, &bv_f, &bv, 0)?;
    let meta = ctx
        .device
        .new_buffer(32, metal::MTLResourceOptions::StorageModeShared);
    write_meta(&meta, tlen as u32, 0, tlen as u32);
    let tmp = ctx.buffer_empty_f32(n_q * Eng::SDPA_MWG_NWG * (hd + 2));
    let o = ctx.buffer_empty_f16(qn);
    eng.sdpa_hybrid_kv(
        ctx, n_q, hd, tlen, tlen as u32, DType::Q40, &bq, &bk, &bv, &meta, &tmp, &o,
    )?;
    ctx.synchronize()?;
    let out = read_f16(ctx, &o, qn);
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    let min = out.iter().copied().fold(f32::INFINITY, f32::min);
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "needle tlen={tlen} hd={hd}: O mean={mean:.4} min={min:.4} max={max:.4} (want ~1 if pos0 retrieved)"
    );
    Ok(())
}

fn main() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut eng = Eng::new();
    println!("device={}", ctx.device_name());
    check(&ctx, &mut eng, 64, 256, 16)?;
    check(&ctx, &mut eng, 128, 256, 9)?;
    check(&ctx, &mut eng, 2048, 1536, 256)?;
    check(&ctx, &mut eng, 4096, 1536, 256)?;
    check(&ctx, &mut eng, 4096, 1536, 55)?;
    check_qkv(&ctx, &mut eng, 8, 1, 256, 16)?;
    check_qkv(&ctx, &mut eng, 8, 1, 512, 16)?;
    check_sdpa(&ctx, &mut eng, 8, 256, 16, 64, 0, 64)?;
    check_sdpa(&ctx, &mut eng, 8, 512, 16, 64, 0, 64)?;
    check_sdpa(&ctx, &mut eng, 8, 256, 16, 256, 0, 256)?;
    check_sdpa(&ctx, &mut eng, 8, 512, 16, 256, 120, 256)?;
    check_sdpa(&ctx, &mut eng, 8, 256, 16, 256, 200, 256)?;
    check_needle(&ctx, &mut eng, 8, 256, 64)?;
    check_needle(&ctx, &mut eng, 8, 512, 64)?;
    check_needle(&ctx, &mut eng, 8, 256, 512)?;
    check_needle(&ctx, &mut eng, 8, 512, 512)?;
    check_needle(&ctx, &mut eng, 8, 512, 2048)?;
    check_needle(&ctx, &mut eng, 8, 512, 8192)?;
    Ok(())
}

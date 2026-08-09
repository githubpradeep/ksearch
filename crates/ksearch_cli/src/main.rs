//! ksearch CLI — Thesis A compiler path toward Gemma generate.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use ksearch_codegen::{beam_tg_candidates, lower_to_metal};
use ksearch_gemma::GemmaModel;
use ksearch_gguf::{
    build_tokenizer_from_gguf, encode_prompt, gemma4_chat_prompt, Gguf,
};
use ksearch_ir::{DType, Graph, Shape};
use ksearch_metal::MetalContext;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ksearch", about = "Metal kernel compiler (Thesis A)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    ElemAdd {
        #[arg(long, default_value_t = 1_048_576)]
        n: usize,
    },
    Matvec {
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 4096)]
        cols: usize,
        #[arg(long, default_value_t = false)]
        beam: bool,
    },
    /// Load Gemma4 GGUF and generate (prints decoded text).
    Generate {
        #[arg(long)]
        gguf: PathBuf,
        /// User text; wrapped in Gemma4 chat template and BPE-encoded.
        #[arg(long)]
        prompt: Option<String>,
        /// Raw comma-separated token ids (used if --prompt is omitted).
        #[arg(long)]
        tokens: Option<String>,
        #[arg(long, default_value_t = 32)]
        n_predict: usize,
        #[arg(long, default_value_t = 512)]
        max_seq: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ElemAdd { n } => elem_add(n),
        Cmd::Matvec { rows, cols, beam } => matvec(rows, cols, beam),
        Cmd::Generate {
            gguf,
            prompt,
            tokens,
            n_predict,
            max_seq,
        } => generate(gguf, prompt, tokens, n_predict, max_seq),
    }
}

fn generate(
    gguf: PathBuf,
    prompt: Option<String>,
    tokens: Option<String>,
    n_predict: usize,
    max_seq: usize,
) -> Result<()> {
    let prompt_ids = if let Some(text) = prompt {
        let g = Gguf::open(&gguf);
        let tok = build_tokenizer_from_gguf(&g).map_err(|e| anyhow::anyhow!(e))?;
        let chat = gemma4_chat_prompt(&text);
        eprintln!("prompt: {chat:?}");
        encode_prompt(&tok, &chat, true).map_err(|e| anyhow::anyhow!(e))?
    } else if let Some(tokens) = tokens {
        tokens
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        bail!("provide --prompt TEXT or --tokens id,id,...");
    };

    let mut model = GemmaModel::load(&gguf, max_seq)?;
    let out = model.generate(&prompt_ids, n_predict)?;

    println!("prompt tokens: {prompt_ids:?}");
    println!("generated ids: {out:?}");
    if let Some(ref vocab) = model.vocab {
        let text = vocab.decode(&out, true);
        println!("---");
        println!("{text}");
    } else {
        eprintln!("(no tokenizer.ggml.tokens in GGUF — cannot decode to text)");
    }
    Ok(())
}

fn elem_add(n: usize) -> Result<()> {
    let mut g = Graph::new();
    let a = g.input(Shape(vec![n]), DType::F32);
    let b = g.input(Shape(vec![n]), DType::F32);
    let out = g.add(a, b)?;

    let kernel = lower_to_metal(&g, out)?;
    println!("=== generated MSL ({}) ===\n{}", kernel.name, kernel.source);

    let ctx = MetalContext::new()?;
    println!("Metal device: {}", ctx.device_name());

    let mut va = vec![0f32; n];
    let mut vb = vec![0f32; n];
    for i in 0..n {
        va[i] = (i % 100) as f32 * 0.01;
        vb[i] = (i % 77) as f32 * 0.02;
    }
    let mut expect = vec![0f32; n];
    for i in 0..n {
        expect[i] = va[i] + vb[i];
    }

    let ba = ctx.buffer_f32(&va);
    let bb = ctx.buffer_f32(&vb);
    let bout = ctx.buffer_empty_f32(n);
    let pipe = ctx.compile(&kernel)?;
    let ms = ctx.run(&pipe, &kernel, &[&ba, &bb], &bout, 256)?;
    let got = ctx.read_f32(&bout, n);

    let max_err = expect
        .iter()
        .zip(got.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0f32, f32::max);
    println!("elem_add n={n}: wall={ms:.3} ms  max|err|={max_err:.3e}");
    if max_err > 1e-5 {
        anyhow::bail!("CPU/GPU mismatch");
    }
    println!("OK");
    Ok(())
}

fn matvec(rows: usize, cols: usize, beam: bool) -> Result<()> {
    let mut g = Graph::new();
    let a = g.input(Shape(vec![rows, cols]), DType::F32);
    let x = g.input(Shape(vec![cols]), DType::F32);
    let ax = g.mul_broadcast_row(a, x)?;
    let y = g.sum_reduce(ax, 1)?;

    let base = lower_to_metal(&g, y)?;
    let ctx = MetalContext::new()?;
    println!("Metal device: {}", ctx.device_name());

    let mut va = vec![0f32; rows * cols];
    let mut vx = vec![0f32; cols];
    for i in 0..rows * cols {
        va[i] = ((i * 17) % 100) as f32 * 0.01;
    }
    for i in 0..cols {
        vx[i] = ((i * 13) % 50) as f32 * 0.02;
    }
    let mut expect = vec![0f32; rows];
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            s += va[r * cols + c] * vx[c];
        }
        expect[r] = s;
    }

    let ba = ctx.buffer_f32(&va);
    let bx = ctx.buffer_f32(&vx);
    let by = ctx.buffer_empty_f32(rows);

    let candidates = if beam {
        beam_tg_candidates(rows)
    } else {
        vec![128]
    };
    let mut best_ms = f64::INFINITY;
    let mut best_tg = candidates[0];
    for &tg in &candidates {
        let pipe = ctx.compile(&base)?;
        let _ = ctx.run(&pipe, &base, &[&ba, &bx], &by, tg)?;
        let ms = ctx.run(&pipe, &base, &[&ba, &bx], &by, tg)?;
        let gflops = (2.0 * rows as f64 * cols as f64) / (ms * 1e6);
        println!("matvec {rows}x{cols} tg={tg}: {ms:.3} ms  {gflops:.1} GFLOP/s");
        if ms < best_ms {
            best_ms = ms;
            best_tg = tg;
        }
    }
    let pipe = ctx.compile(&base)?;
    let _ = ctx.run(&pipe, &base, &[&ba, &bx], &by, best_tg)?;
    let got = ctx.read_f32(&by, rows);
    let max_err = expect
        .iter()
        .zip(got.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0f32, f32::max);
    println!("best tg={best_tg}  max|err|={max_err:.3e}");
    if max_err > 1e-2 {
        anyhow::bail!("CPU/GPU mismatch");
    }
    println!("OK");
    Ok(())
}

// temporary - no, use a small bin

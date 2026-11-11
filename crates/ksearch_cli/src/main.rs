//! ksearch CLI — Thesis A compiler path toward Gemma generate.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
#[cfg(feature = "metal")]
use ksearch_codegen::{
    beam_search_matvec, lower_to_metal, lower_with_schedule, CodegenError, Target,
};
use ksearch_gemma::GemmaPrimModel;
use ksearch_gguf::{build_tokenizer_from_gguf, encode_prompt, gemma4_chat_prompt, Gguf};
use ksearch_ir::{DType, Graph, OptSchedule, Shape};
#[cfg(feature = "metal")]
use ksearch_metal::MetalContext;
use std::path::{Path, PathBuf};

mod gemma4_chat;
mod scheduler;
mod serve;

const DEFAULT_GGUF: &str = "~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf";
const ESSAY_PROMPT: &str =
    "Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conclusion.";

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
        /// `f16` (default) or `q4k` (requires cols % 256 == 0).
        #[arg(long)]
        dtype: Option<String>,
        /// Verify every unique candidate kernel against the CPU reference.
        #[arg(long, default_value_t = false)]
        verify_all: bool,
        /// `metal` (default) or `cuda` (NVIDIA driver + nvcc required).
        #[arg(long, default_value = "metal")]
        backend: String,
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
        /// MTP draft head GGUF: speculative decode (draft + batched verify).
        #[arg(long)]
        mtp: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        n_predict: usize,
        #[arg(long, default_value_t = 512)]
        max_seq: usize,
        /// GPU backend: `auto` (default), `metal`, or `cuda` (H100 / RTX 50-series).
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Regression bench: Hi gate + essay decode/prefill tok/s.
    Bench {
        #[arg(long, default_value = DEFAULT_GGUF)]
        gguf: String,
        #[arg(long, default_value_t = 32)]
        n_predict_hi: usize,
        #[arg(long, default_value_t = 512)]
        n_predict_essay: usize,
        #[arg(long, default_value_t = 1024)]
        max_seq: usize,
        /// Fixed prefill length. With `--tg`, this is the bench (llama-bench `-p`), not the Hi sample.
        #[arg(long)]
        pp: Option<usize>,
        /// Fixed decode length. With `--pp`, generate this many tokens and do not stop on end-of-turn.
        #[arg(long)]
        tg: Option<usize>,
        /// Timed repeats after one discarded warmup. Used with `--pp` and `--tg`.
        #[arg(long, default_value_t = 3)]
        repeat: usize,
        /// GPU backend: `auto` (default), `metal`, or `cuda` (H100 / RTX 50-series).
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// OpenAI-compatible HTTP server (`/v1/chat/completions`).
    Serve {
        #[arg(long, default_value = DEFAULT_GGUF)]
        gguf: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// KV slot capacity (tokens). Default 16k like metal-llm-server; model ctx is ~128k.
        #[arg(long, default_value_t = 16384)]
        max_seq: usize,
        /// Concurrent KV slots (llama.cpp `--parallel`).
        #[arg(long, default_value_t = 4)]
        slots: usize,
        /// GPU backend: `auto` (default), `metal`, or `cuda` (H100 / RTX 50-series).
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Head-to-head MSL variants for the Q4K coop GEMV (codegen R&D).
    Q4kFragBench {
        #[arg(long, default_value_t = 8960)]
        rows: usize,
        #[arg(long, default_value_t = 1536)]
        cols: usize,
    },
    /// CUDA backend: render the kernel zoo, and compile it with `nvcc` when asked.
    Cuda {
        /// Directory to write the generated `.cu` sources into.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Compile every kernel with `nvcc` (needs a CUDA device to pick `sm_XX`).
        #[arg(long, default_value_t = false)]
        compile: bool,
        /// CUDA device ordinal used for `--compile`.
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// Block size for the flat (elementwise / per-row) launch hints.
        #[arg(long, default_value_t = 256)]
        tg_size: u64,
    },
    /// Train/eval the learned cost model on collected kernel timings.
    Tune {
        /// Restrict to this chip (default: every chip in the dataset).
        #[arg(long)]
        chip: Option<String>,
        /// Min timed configs per shape group to count toward ranking metrics.
        #[arg(long, default_value_t = 5)]
        min_group: usize,
        /// Replay guided search on a live rows×cols F16 matvec (GPU ground truth).
        #[arg(long)]
        live_rows: Option<usize>,
        #[arg(long)]
        live_cols: Option<usize>,
        /// Keep only the newest N dataset samples first.
        #[arg(long)]
        vacuum: Option<usize>,
    },
}

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

/// Resolve `--backend` for the model commands (generate / bench / serve).
///
/// `auto` keeps the host default (`KSEARCH_BACKEND`, else Metal on Apple hosts
/// and CUDA everywhere else — so an H100 or RTX 5090 box runs CUDA without any
/// flag). An explicit spelling is exported as `KSEARCH_BACKEND`, which is what
/// `ksearch_compute::Context::new` consults when the model opens its GPU.
fn select_backend(sel: &str) -> Result<()> {
    if sel.eq_ignore_ascii_case("auto") {
        return Ok(());
    }
    let backend = ksearch_compute::Backend::parse(sel).ok_or_else(|| {
        anyhow::anyhow!("--backend must be `auto`, `metal` or `cuda` (got `{sel}`)")
    })?;
    // Fail fast when the binary was built without the requested backend.
    #[cfg(not(feature = "metal"))]
    if backend == ksearch_compute::Backend::Metal {
        bail!("this build has no Metal backend: rebuild with `--features metal`");
    }
    #[cfg(not(feature = "cuda"))]
    if backend == ksearch_compute::Backend::Cuda {
        bail!("this build has no CUDA backend: rebuild with `--features cuda`");
    }
    std::env::set_var("KSEARCH_BACKEND", backend.as_str());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ElemAdd { n } => elem_add(n),
        Cmd::Matvec {
            rows,
            cols,
            beam,
            dtype,
            verify_all,
            backend,
        } => match backend.as_str() {
            "metal" => matvec(rows, cols, beam, dtype.as_deref(), verify_all),
            "cuda" => matvec_cuda(rows, cols, beam, dtype.as_deref(), verify_all),
            other => bail!("--backend must be `metal` or `cuda` (got `{other}`)"),
        },
        Cmd::Generate {
            gguf,
            prompt,
            tokens,
            mtp,
            n_predict,
            max_seq,
            backend,
        } => {
            select_backend(&backend)?;
            generate(gguf, prompt, tokens, n_predict, max_seq, mtp)
        }
        Cmd::Bench {
            gguf,
            n_predict_hi,
            n_predict_essay,
            max_seq,
            pp,
            tg,
            repeat,
            backend,
        } => {
            select_backend(&backend)?;
            bench(
                expand_home(&gguf),
                n_predict_hi,
                n_predict_essay,
                max_seq,
                pp,
                tg,
                repeat,
            )
        }
        Cmd::Serve {
            gguf,
            port,
            max_seq,
            slots,
            backend,
        } => {
            select_backend(&backend)?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(serve::run_server(serve::ServeArgs {
                gguf: expand_home(&gguf),
                port,
                max_seq,
                slots,
            }))
        }
        Cmd::Q4kFragBench { rows, cols } => q4k_frag_bench(rows, cols),
        Cmd::Cuda {
            out,
            compile,
            device,
            tg_size,
        } => cuda(out, compile, device, tg_size),
        Cmd::Tune {
            chip,
            min_group,
            live_rows,
            live_cols,
            vacuum,
        } => {
            if let Some(keep) = vacuum {
                let dropped = ksearch_tuner::dataset::truncate_dataset(keep)?;
                println!("vacuum: dropped {dropped} old samples");
            }
            if let (Some(rows), Some(cols)) = (live_rows, live_cols) {
                tune_live(rows, cols)
            } else {
                tune_train_eval(chip.as_deref(), min_group)
            }
        }
    }
}

fn tune_train_eval(chip: Option<&str>, min_group: usize) -> Result<()> {
    use ksearch_tuner::gbdt::GbdtConfig;
    use ksearch_tuner::model::{evaluate, CostModel};
    use std::collections::{BTreeSet, HashMap};

    let all = ksearch_tuner::dataset::load_samples(None);
    if all.is_empty() {
        bail!(
            "no samples in {} — collect some first: KSEARCH_BEAM_FORCE=1 \\
             cargo run -p ksearch_cli --release -- matvec --rows 4096 --cols 4096 --beam",
            ksearch_tuner::dataset::dataset_path().display()
        );
    }
    let mut chips: BTreeSet<String> = all.iter().map(|s| s.chip.clone()).collect();
    if let Some(c) = chip {
        chips.retain(|x| x == c);
        if chips.is_empty() {
            bail!("no samples for chip {c}");
        }
    }
    for c in &chips {
        let samples = ksearch_tuner::dataset::load_samples(Some(c));
        // Honest metrics: hold out every 5th sample within each shape group.
        let mut groups: HashMap<(String, usize, usize), Vec<&ksearch_tuner::Sample>> =
            HashMap::new();
        for s in &samples {
            groups
                .entry((s.kind.clone(), s.rows, s.cols))
                .or_default()
                .push(s);
        }
        let mut train: Vec<ksearch_tuner::Sample> = Vec::new();
        let mut holdout: Vec<ksearch_tuner::Sample> = Vec::new();
        for grp in groups.values() {
            for (i, s) in grp.iter().enumerate() {
                if i % 5 == 0 {
                    holdout.push((*s).clone());
                } else {
                    train.push((*s).clone());
                }
            }
        }
        let fit =
            |data: &[ksearch_tuner::Sample]| CostModel::train(data, c, &GbdtConfig::default());

        if holdout.len() >= 30 && train.len() >= 64 {
            let m = fit(&train);
            let ev = evaluate(&m, &holdout, 3);
            println!(
                "\nchip {c}: {} samples ({} train / {} holdout)",
                samples.len(),
                train.len(),
                holdout.len()
            );
            println!(
                "  holdout: {} groups  mean spearman {:.3}  top1 {:.1}%  mean regret {:.2}%",
                ev.n_groups,
                ev.mean_spearman,
                100.0 * ev.top1_rate,
                100.0 * ev.mean_regret
            );
            let mut worst = ev.groups.clone();
            worst.sort_by(|a, b| {
                b.regret
                    .partial_cmp(&a.regret)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for g in worst.iter().take(5) {
                println!(
                    "    {} {}x{}: n={} ρ={:.3} top1={} regret={:.2}%",
                    g.kind,
                    g.rows,
                    g.cols,
                    g.n,
                    g.spearman,
                    g.top1_hit,
                    100.0 * g.regret
                );
            }
        } else {
            let m = fit(&samples);
            let ev = evaluate(&m, &samples, min_group);
            println!(
                "\nchip {c}: {} samples (in-sample, too few for holdout)",
                samples.len()
            );
            println!(
                "  groups≥{min_group}: {}  mean spearman {:.3}  top1 {:.1}%  mean regret {:.2}%",
                ev.n_groups,
                ev.mean_spearman,
                100.0 * ev.top1_rate,
                100.0 * ev.mean_regret
            );
        }

        // Final model always trains on everything.
        let path = fit(&samples).save()?;
        println!("  saved {}", path.display());
    }
    Ok(())
}

/// Time every candidate once (ground truth), then replay the guided policy
/// against those measurements: how close does budgeted search get to
/// exhaustive search on this shape, right now?
#[cfg(not(feature = "metal"))]
fn tune_live(_rows: usize, _cols: usize) -> Result<()> {
    bail!("`tune-live` needs the Metal backend: rebuild with `--features metal`")
}

#[cfg(feature = "metal")]
fn tune_live(rows: usize, cols: usize) -> Result<()> {
    use ksearch_codegen::{beam_matvec_candidates, MetalKernelSource};
    use ksearch_tuner::{GuidedSearch, Sched, SearchCtx, TunerConfig};
    use std::collections::HashSet;
    use std::hash::{DefaultHasher, Hash, Hasher};

    let ctx = MetalContext::new()?;
    let chip = ctx.device_name();
    println!("Metal device: {chip}");

    let mut g = Graph::new();
    let w = g.input(Shape(vec![rows, cols]), DType::F16);
    let v = g.input(Shape(vec![cols]), DType::F16);
    let ax = g.mul_broadcast_row(w, v)?;
    let y = g.sum_reduce(ax, 1)?;

    let mut va = vec![0f32; rows * cols];
    let mut vx = vec![0f32; cols];
    for (i, x) in va.iter_mut().enumerate() {
        *x = ((i * 17) % 100) as f32 * 0.01;
    }
    for (i, x) in vx.iter_mut().enumerate() {
        *x = ((i * 13) % 50) as f32 * 0.02;
    }
    let f16_bytes = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| ksearch_gguf::f32_to_f16(v).to_le_bytes())
            .collect()
    };
    let bw = ctx.buffer_bytes(&f16_bytes(&va));
    let bx = ctx.buffer_bytes(&f16_bytes(&vx));
    let by = ctx.buffer_empty_f16(rows);

    let time_one = |kernel: &MetalKernelSource| -> Result<f64, CodegenError> {
        let pipe = ctx
            .compile(kernel)
            .map_err(|e| CodegenError::Msg(e.to_string()))?;
        let _ = ctx
            .run(&pipe, kernel, &[&bw, &bx], &by, kernel_tg(kernel))
            .map_err(|e| CodegenError::Msg(e.to_string()))?;
        let mut samples = [0f64; 3];
        for s in &mut samples {
            *s = ctx
                .run(&pipe, kernel, &[&bw, &bx], &by, kernel_tg(kernel))
                .map_err(|e| CodegenError::Msg(e.to_string()))?;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(samples[1])
    };

    // Lower + dedup exactly like beam_search_matvec does.
    let mut uniq: Vec<(OptSchedule, MetalKernelSource)> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for sched in beam_matvec_candidates() {
        let kernel = lower_with_schedule(&g, y, sched)?;
        let mut h = DefaultHasher::new();
        kernel.source.hash(&mut h);
        sched.hash(&mut h);
        if seen.insert(h.finish()) {
            uniq.push((sched, kernel));
        }
    }
    let scheds: Vec<Sched> = uniq.iter().map(|(s, _)| (*s).into()).collect();

    println!("timing all {} unique kernels (ground truth)…", uniq.len());
    let mut times = Vec::with_capacity(uniq.len());
    for (_, k) in &uniq {
        times.push(time_one(k)?);
    }
    let (best_i, best_ms) = times
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let cfg = TunerConfig {
        enabled: true,
        ..Default::default()
    };
    let mut gs = GuidedSearch::new(
        cfg,
        SearchCtx {
            kind: "matvec_f16_nr",
            rows,
            cols,
            chip: &chip,
        },
        &scheds,
    );
    if !gs.should_guide() {
        bail!(
            "cost model not ready for {chip} — collect samples first (KSEARCH_BEAM_FORCE=1 matvec --beam), then `ksearch tune`"
        );
    }
    while !gs.should_stop() {
        let batch = gs.next_batch();
        if batch.is_empty() {
            break;
        }
        for i in batch {
            gs.record(i, times[i]);
        }
    }
    let (gi, gm) = gs.best().unwrap_or((best_i, *best_ms));
    println!(
        "exhaustive: {} kernels → best {:.3} ms (tg={} nr0={})",
        uniq.len(),
        best_ms,
        uniq[best_i].0.tg,
        uniq[best_i].0.nr0
    );
    println!(
        "guided:     {} timings ({:.0}% of grid) → {:.3} ms (tg={} nr0={})  +{:.3} ms vs best",
        gs.n_measured(),
        100.0 * gs.n_measured() as f64 / uniq.len() as f64,
        gm,
        uniq[gi].0.tg,
        uniq[gi].0.nr0,
        gm - best_ms
    );
    Ok(())
}

fn encode_user_prompt(gguf: &Path, text: &str) -> Result<Vec<u32>> {
    let g = Gguf::open(gguf);
    let tok = build_tokenizer_from_gguf(&g).map_err(|e| anyhow::anyhow!(e))?;
    let chat = gemma4_chat_prompt(text);
    encode_prompt(&tok, &chat, true).map_err(|e| anyhow::anyhow!(e))
}

fn bench(
    gguf: PathBuf,
    n_predict_hi: usize,
    n_predict_essay: usize,
    max_seq: usize,
    pp: Option<usize>,
    tg: Option<usize>,
    repeat: usize,
) -> Result<()> {
    eprintln!("bench gguf={}", gguf.display());
    if let (Some(pp), Some(tg)) = (pp, tg) {
        return bench_fixed(gguf, max_seq, pp, tg, repeat);
    }
    let hi_ids = encode_user_prompt(&gguf, "Hi")?;
    let essay_ids = encode_user_prompt(&gguf, ESSAY_PROMPT)?;

    eprintln!("path: Thesis A GemmaPrim (tinygrad-shaped IR)");
    let mut model = GemmaPrimModel::load(&gguf, max_seq)?;
    let hi = model.generate_timed(&hi_ids, n_predict_hi, false, true)?;
    let hi_text = model
        .vocab
        .as_ref()
        .map(|v| v.decode(&hi.tokens, true))
        .unwrap_or_default();
    let hi_pass = hi_text.contains("Hi!") && hi_text.contains("help") && !hi.tokens.is_empty();
    println!(
        "hi:     prefill={:.1} tok/s  decode={:.1} tok/s  pass={}  text={:?}",
        hi.prefill_tok_s(),
        hi.decode_tok_s(),
        hi_pass,
        hi_text.trim()
    );

    let essay = model.generate_timed(&essay_ids, n_predict_essay, false, true)?;
    println!(
        "essay:  prefill={:.1} tok/s  decode={:.1} tok/s  tokens={}",
        essay.prefill_tok_s(),
        essay.decode_tok_s(),
        essay.tokens.len()
    );

    if !hi_pass {
        bail!("Hi gate failed: {hi_text:?}");
    }
    Ok(())
}

/// llama-bench shape: one discarded warmup, then `repeat` runs of exactly `pp`
/// prefill tokens and `tg` decode tokens. End-of-turn does not cut the run short.
fn bench_fixed(gguf: PathBuf, max_seq: usize, pp: usize, tg: usize, repeat: usize) -> Result<()> {
    if pp == 0 || tg == 0 {
        bail!("--pp and --tg must be > 0");
    }
    if pp + tg > max_seq {
        bail!("--max-seq {max_seq} is smaller than --pp {pp} + --tg {tg}");
    }
    let repeat = repeat.max(1);
    // prefill_chunk times every token except the last, which is the first decode step.
    let prompt = vec![100u32; pp + 1];
    eprintln!("path: Thesis A GemmaPrim (fixed pp={pp} tg={tg} repeat={repeat})");
    let mut model = GemmaPrimModel::load(&gguf, max_seq)?;
    let hi_ids = encode_user_prompt(&gguf, "Hi")?;
    let hi = model.generate_timed(&hi_ids, 16, false, true)?;
    let hi_text = model
        .vocab
        .as_ref()
        .map(|v| v.decode(&hi.tokens, true))
        .unwrap_or_default();
    println!("check: {:?}", hi_text.trim());
    if !(hi_text.contains("Hi!") && hi_text.contains("help")) {
        bail!("prefill check failed: {hi_text:?}");
    }
    eprintln!("warmup (not timed)");
    let warm = model.generate_timed(&prompt, tg, false, false)?;
    eprintln!(
        "warmup prefill={} decode={}",
        warm.prefill_tokens,
        warm.tokens.len()
    );

    let mut pp_rates = Vec::with_capacity(repeat);
    let mut tg_rates = Vec::with_capacity(repeat);
    for i in 0..repeat {
        let stats = model.generate_timed(&prompt, tg, false, false)?;
        if stats.prefill_tokens != pp || stats.tokens.len() != tg {
            bail!(
                "run {i} measured prefill={} decode={}, wanted {pp}/{tg}",
                stats.prefill_tokens,
                stats.tokens.len()
            );
        }
        pp_rates.push(stats.prefill_tok_s());
        tg_rates.push(stats.decode_tok_s());
        println!(
            "pp{pp}  {:.2} tok/s   tg{tg}  {:.2} tok/s",
            pp_rates[i], tg_rates[i]
        );
    }
    println!(
        "pp{pp}  {:.2} ± {:.2} tok/s",
        mean(&pp_rates),
        stdev(&pp_rates)
    );
    println!(
        "tg{tg}  {:.2} ± {:.2} tok/s",
        mean(&tg_rates),
        stdev(&tg_rates)
    );
    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stdev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64;
    v.sqrt()
}

fn generate(
    gguf: PathBuf,
    prompt: Option<String>,
    tokens: Option<String>,
    n_predict: usize,
    max_seq: usize,
    mtp: Option<PathBuf>,
) -> Result<()> {
    let prompt_ids = if let Some(text) = prompt {
        let ids = encode_user_prompt(&gguf, &text)?;
        eprintln!("prompt tokens: {}", ids.len());
        ids
    } else if let Some(tokens) = tokens {
        tokens
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        bail!("provide --prompt TEXT or --tokens id,id,...");
    };

    eprintln!("path: Thesis A GemmaPrim");
    let mut model = GemmaPrimModel::load(&gguf, max_seq)?;
    if let Some(ref mtp_path) = mtp {
        model.load_mtp(mtp_path)?;
    }
    let out = model.generate(&prompt_ids, n_predict)?;
    let (hits, tried) = model.mtp_stats();
    if tried > 0 {
        println!(
            "[mtp-accept] final: {}/{} = {:.1}%",
            hits,
            tried,
            100.0 * hits as f64 / tried as f64
        );
    }
    println!("prompt tokens: {prompt_ids:?}");
    println!("generated ids: {out:?}");
    if let Some(ref vocab) = model.vocab {
        println!("---\n{}", vocab.decode(&out, true));
    } else {
        eprintln!("(no tokenizer.ggml.tokens in GGUF — cannot decode to text)");
    }
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn elem_add(_n: usize) -> Result<()> {
    bail!("`elem-add` needs the Metal backend: rebuild with `--features metal`")
}

#[cfg(feature = "metal")]
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

#[cfg(not(feature = "metal"))]
fn matvec(
    _rows: usize,
    _cols: usize,
    _beam: bool,
    _dtype: Option<&str>,
    _verify_all: bool,
) -> Result<()> {
    bail!("`matvec --backend metal` needs the Metal backend: rebuild with `--features metal`")
}

#[cfg(feature = "metal")]
fn matvec(
    rows: usize,
    cols: usize,
    beam: bool,
    dtype: Option<&str>,
    verify_all: bool,
) -> Result<()> {
    let q4k = matches!(dtype, Some("q4k" | "Q4K"));
    if q4k && cols % 256 != 0 {
        anyhow::bail!("--dtype q4k needs cols % 256 == 0");
    }
    let w_dtype = if q4k { DType::Q4K } else { DType::F16 };
    let mut g = Graph::new();
    let a = g.input(Shape(vec![rows, cols]), w_dtype);
    let x = g.input(Shape(vec![cols]), DType::F16);
    let ax = g.mul_broadcast_row(a, x)?;
    let y = g.sum_reduce(ax, 1)?;

    let ctx = MetalContext::new()?;
    println!("Metal device: {}", ctx.device_name());
    println!(
        "path: Graph → schedule → Kernel IR → MSL ({})",
        if q4k { "Q4K" } else { "F16" }
    );

    let (va, vx, expect) = matvec_inputs(rows, cols, q4k);

    let f16_bytes = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| ksearch_gguf::f32_to_f16(v).to_le_bytes())
            .collect()
    };
    let ba = if q4k {
        ctx.buffer_bytes(&ksearch_gguf::quantize_f32_to_q4k(&va))
    } else {
        ctx.buffer_bytes(&f16_bytes(&va))
    };
    let bx = ctx.buffer_bytes(&f16_bytes(&vx));
    let by = ctx.buffer_empty_f16(rows);

    let time_one = |kernel: &ksearch_codegen::MetalKernelSource| -> Result<f64, CodegenError> {
        let pipe = ctx
            .compile(kernel)
            .map_err(|e| CodegenError::Msg(e.to_string()))?;
        let _ = ctx
            .run(&pipe, kernel, &[&ba, &bx], &by, kernel_tg(kernel))
            .map_err(|e| CodegenError::Msg(e.to_string()))?;
        let mut samples = [0f64; 3];
        for s in &mut samples {
            *s = ctx
                .run(&pipe, kernel, &[&ba, &bx], &by, kernel_tg(kernel))
                .map_err(|e| CodegenError::Msg(e.to_string()))?;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(samples[1])
    };

    let check = |kernel: &ksearch_codegen::MetalKernelSource| -> Result<f32> {
        let pipe = ctx.compile(kernel)?;
        ctx.run(&pipe, kernel, &[&ba, &bx], &by, kernel_tg(kernel))?;
        let got: Vec<f32> = ctx
            .read_u16(&by, rows)
            .into_iter()
            .map(ksearch_gguf::f16_to_f32)
            .collect();
        Ok(expect
            .iter()
            .zip(got.iter())
            .map(|(e, o)| (e - o).abs())
            .fold(0f32, f32::max))
    };

    if beam && verify_all {
        // Numerical sweep over every unique candidate kernel.
        let mut seen = std::collections::HashSet::new();
        let mut n_ok = 0usize;
        let mut worst = 0f32;
        for sched in std::iter::once(OptSchedule::untuned()).chain(match w_dtype {
            DType::Q4K => ksearch_codegen::beam_matvec_q4k_candidates(),
            _ => ksearch_codegen::beam_matvec_candidates(),
        }) {
            let kernel = lower_with_schedule(&g, y, sched)?;
            let mut h = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            kernel.source.hash(&mut h);
            sched.hash(&mut h);
            if !seen.insert(h.finish()) {
                continue;
            }
            let err = check(&kernel)?;
            worst = worst.max(err);
            // F16 accumulation error grows ~linearly with K; the fixed 0.05
            // gate only fits cols<=512. Scale with K (0.05 @ 256).
            let tol = 0.05f32 * (cols as f32 / 256.0).max(1.0);
            if err > tol {
                anyhow::bail!(
                    "CPU/GPU mismatch at nsg={} nr0={} unroll={} unroll_tol={tol:.3}: max|err|={err:.3e}",
                    sched.nsg,
                    sched.nr0,
                    sched.unroll
                );
            }
            n_ok += 1;
        }
        println!("verify-all: {n_ok} unique kernels OK (worst max|err|={worst:.3e})");
        return Ok(());
    }

    let plan_kind = if q4k {
        "matvec_q4k_v2"
    } else {
        "matvec_f16_nr"
    };
    let (best_kernel, best_ms, label) = if beam {
        let untuned = lower_with_schedule(&g, y, OptSchedule::untuned())?;
        let base_ms = time_one(&untuned)?;
        let base_gflops = (2.0 * rows as f64 * cols as f64) / (base_ms * 1e6);
        println!(
            "untuned TG={} VEC={} UNROLL={}: {base_ms:.3} ms  {base_gflops:.1} GFLOP/s",
            OptSchedule::untuned().tg,
            OptSchedule::untuned().vec,
            OptSchedule::untuned().unroll
        );

        let chip = ctx.device_name();
        if std::env::var_os("KSEARCH_BEAM_FORCE").is_some() {
            let key = ksearch_codegen::plan_key(plan_kind, &[rows, cols], &chip);
            let _ =
                std::fs::remove_file(ksearch_codegen::beam_cache_dir().join(format!("{key}.txt")));
        }
        let result = beam_search_matvec(&g, y, &chip, Target::Metal, time_one)?;
        let gflops = (2.0 * rows as f64 * cols as f64) / (result.ms * 1e6);
        let src = if result.from_cache { "cache" } else { "search" };
        println!(
            "beam ({src}) TG={} VEC={} UNROLL={} NSG={} NR0={}: {:.3} ms  {gflops:.1} GFLOP/s  ({:.2}x vs untuned)",
            result.schedule.tg,
            result.schedule.vec,
            result.schedule.unroll,
            result.schedule.nsg,
            result.schedule.nr0,
            result.ms,
            base_ms / result.ms
        );
        (
            result.kernel,
            result.ms,
            format!(
                "tg={} vec={} unroll={} nsg={} nr0={}",
                result.schedule.tg,
                result.schedule.vec,
                result.schedule.unroll,
                result.schedule.nsg,
                result.schedule.nr0
            ),
        )
    } else {
        let kernel = lower_to_metal(&g, y)?;
        let ms = time_one(&kernel)?;
        let gflops = (2.0 * rows as f64 * cols as f64) / (ms * 1e6);
        println!("matvec {rows}x{cols}: {ms:.3} ms  {gflops:.1} GFLOP/s");
        (kernel, ms, "default".into())
    };

    let pipe = ctx.compile(&best_kernel)?;
    if std::env::var_os("KSEARCH_DUMP_MSL").is_some() {
        println!(
            "=== generated MSL ({}) ===\n{}",
            best_kernel.name, best_kernel.source
        );
    }
    let _ = ctx.run(
        &pipe,
        &best_kernel,
        &[&ba, &bx],
        &by,
        kernel_tg(&best_kernel),
    )?;
    let got: Vec<f32> = ctx
        .read_u16(&by, rows)
        .into_iter()
        .map(ksearch_gguf::f16_to_f32)
        .collect();
    let max_err = expect
        .iter()
        .zip(got.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0f32, f32::max);
    println!("best ({label})  {best_ms:.3} ms  max|err|={max_err:.3e}");
    // F16 accumulate vs F32 CPU ref — error grows ~linearly with K
    // (0.05 @ cols=256, matching the historical gate at small K).
    let tol = 0.05f32 * (cols as f32 / 256.0).max(1.0);
    if max_err > tol {
        anyhow::bail!("CPU/GPU mismatch (tol={tol:.3})");
    }
    println!("OK");
    Ok(())
}

/// Deterministic matvec inputs and the CPU reference, shared by the Metal and
/// CUDA gates.
///
/// For Q4K the GPU multiplies *dequantized* weights, so the reference
/// dequantizes identically and f16 rounding is the only error source.
fn matvec_inputs(rows: usize, cols: usize, q4k: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut va = vec![0f32; rows * cols];
    let mut vx = vec![0f32; cols];
    for (i, v) in va.iter_mut().enumerate() {
        *v = ((i * 17) % 100) as f32 * 0.01;
    }
    for (i, v) in vx.iter_mut().enumerate() {
        *v = ((i * 13) % 50) as f32 * 0.02;
    }

    let expect: Vec<f32> = if q4k {
        let packed = ksearch_gguf::quantize_f32_to_q4k(&va);
        let deq =
            ksearch_gguf::dequant_type_to_f32(ksearch_gguf::ggml_type::Q4_K, &packed, rows * cols);
        (0..rows)
            .map(|r| (0..cols).map(|c| deq[r * cols + c] * vx[c]).sum())
            .collect()
    } else {
        (0..rows)
            .map(|r| (0..cols).map(|c| va[r * cols + c] * vx[c]).sum())
            .collect()
    };
    (va, vx, expect)
}

/// `matvec --backend cuda`: the same graph, lowered to CUDA and run through the
/// driver API, checked against the CPU reference.
///
/// This is the on-device gate for H100 / RTX 40-50-series: it exercises the
/// whole CUDA path (codegen → `nvcc --cubin` for the local `sm_XX` → driver
/// launch) and reports GFLOP/s.
fn matvec_cuda(
    rows: usize,
    cols: usize,
    beam: bool,
    dtype: Option<&str>,
    verify_all: bool,
) -> Result<()> {
    use ksearch_codegen::{lower_to_cuda_chip, lower_with_schedule_target, Target};
    use ksearch_cuda::{launch_config, CudaContext, KernelArgs};
    use std::collections::HashSet;
    use std::time::Instant;

    let q4k = matches!(dtype, Some("q4k" | "Q4K"));
    if q4k && !cols.is_multiple_of(256) {
        bail!("--dtype q4k needs cols % 256 == 0");
    }
    let w_dtype = if q4k { DType::Q4K } else { DType::F16 };
    let mut g = Graph::new();
    let a = g.input(Shape(vec![rows, cols]), w_dtype);
    let x = g.input(Shape(vec![cols]), DType::F16);
    let ax = g.mul_broadcast_row(a, x)?;
    let y = g.sum_reduce(ax, 1)?;

    let mut ctx = CudaContext::with_options(0, 256).map_err(|e| anyhow::anyhow!("{e}"))?;
    let chip = ctx.info().name.clone();
    println!("CUDA device: {chip} ({})", ctx.info().sm_arch());
    println!(
        "nvcc: {}",
        ctx.nvcc()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found".into())
    );
    println!(
        "path: Graph → schedule → Kernel IR → CUDA ({})",
        if q4k { "Q4K" } else { "F16" }
    );

    let (va, vx, expect) = matvec_inputs(rows, cols, q4k);
    let f16_bytes = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| ksearch_gguf::f32_to_f16(v).to_le_bytes())
            .collect()
    };
    let wbytes = if q4k {
        ksearch_gguf::quantize_f32_to_q4k(&va)
    } else {
        f16_bytes(&va)
    };
    let ba = ctx
        .alloc(wbytes.len())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let bx = ctx
        .alloc(vx.len() * 2)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let by = ctx.alloc(rows * 2).map_err(|e| anyhow::anyhow!("{e}"))?;
    ctx.copy_to(&ba, &wbytes)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ctx.copy_to(&bx, &f16_bytes(&vx))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // One launch per kernel; `synchronize` gives the wall time of the whole
    // dispatch, which is what matters for a single matvec.
    let time_one = |ctx: &CudaContext,
                    f: ksearch_cuda::Kernel,
                    kernel: &ksearch_codegen::KernelSource|
     -> Result<f64> {
        let launch = || -> Result<()> {
            let mut args = KernelArgs::new();
            args.ptr(ba.as_ptr());
            args.ptr(bx.as_ptr());
            args.ptr(by.as_ptr());
            ctx.launch(f, &kernel.launch, &mut args)
                .map_err(|e| anyhow::anyhow!("{e}"))
        };
        launch()?;
        ctx.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            launch()?;
            ctx.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(best)
    };

    let check = |ctx: &CudaContext,
                 f: ksearch_cuda::Kernel,
                 kernel: &ksearch_codegen::KernelSource|
     -> Result<f32> {
        let mut args = KernelArgs::new();
        args.ptr(ba.as_ptr());
        args.ptr(bx.as_ptr());
        args.ptr(by.as_ptr());
        ctx.launch(f, &kernel.launch, &mut args)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ctx.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut raw = vec![0u8; rows * 2];
        ctx.copy_from(&by, &mut raw)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let max_err = raw
            .chunks_exact(2)
            .zip(expect.iter())
            .map(|(c, e)| {
                let got = ksearch_gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                (e - got).abs()
            })
            .fold(0f32, f32::max);
        Ok(max_err)
    };

    // F16 accumulate vs F32 CPU ref — error grows ~linearly with K.
    let tol = 0.05f32 * (cols as f32 / 256.0).max(1.0);
    let gflops = |ms: f64| (2.0 * rows as f64 * cols as f64) / (ms * 1e6);

    let scheds: Vec<OptSchedule> = if beam {
        std::iter::once(OptSchedule::untuned())
            .chain(match w_dtype {
                DType::Q4K => ksearch_codegen::beam_matvec_q4k_candidates(),
                _ => ksearch_codegen::beam_matvec_candidates(),
            })
            .collect()
    } else {
        vec![OptSchedule::untuned()]
    };

    fn source_hash(src: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        h.finish()
    }

    let mut seen = HashSet::new();
    let mut best: Option<(f64, String, f32)> = None;
    let mut n_ok = 0usize;
    let mut worst = 0f32;
    for sched in scheds {
        let kernel = if beam {
            lower_with_schedule_target(&g, y, sched, Target::Cuda)?
        } else {
            lower_to_cuda_chip(&g, y, &chip)?
        };
        if !seen.insert(source_hash(&kernel.source)) {
            continue;
        }
        let f = ctx
            .load(&kernel)
            .map_err(|e| anyhow::anyhow!("nvcc/load failed: {e}"))?;
        let err = check(&ctx, f, &kernel)?;
        worst = worst.max(err);
        if verify_all && err > tol {
            bail!(
                "CPU/GPU mismatch (tg={} vec={} nsg={}): max|err|={err:.3e} > tol={tol:.3}",
                sched.tg,
                sched.vec,
                sched.nsg
            );
        }
        n_ok += 1;

        let ms = time_one(&ctx, f, &kernel)?;
        let cfg = launch_config(&kernel.launch, 256);
        let label = format!(
            "tg={} vec={} unroll={} nsg={} nr0={}",
            sched.tg, sched.vec, sched.unroll, sched.nsg, sched.nr0
        );
        println!(
            "{label}  grid=({},{},{}) block=({},{},{}) smem={}  {ms:.3} ms  {:.1} GFLOP/s  max|err|={err:.3e}",
            cfg.grid.0,
            cfg.grid.1,
            cfg.grid.2,
            cfg.block.0,
            cfg.block.1,
            cfg.block.2,
            cfg.smem,
            gflops(ms)
        );
        if best.as_ref().is_none_or(|(bms, _, _)| ms < *bms) {
            best = Some((ms, label, err));
        }
    }

    if verify_all {
        println!("verify-all: {n_ok} unique kernels OK (worst max|err|={worst:.3e})");
    }
    let (ms, label, err) = best.ok_or_else(|| anyhow::anyhow!("no CUDA kernel produced"))?;
    println!(
        "best ({label})  {ms:.3} ms  {:.1} GFLOP/s  max|err|={err:.3e}",
        gflops(ms)
    );
    if err > tol {
        bail!("CPU/GPU mismatch (tol={tol:.3})");
    }
    println!("OK");
    Ok(())
}

#[cfg(feature = "metal")]
fn kernel_tg(kernel: &ksearch_codegen::MetalKernelSource) -> u64 {
    match &kernel.launch {
        ksearch_codegen::LaunchHint::RowsParallel { tg, .. } => *tg,
        ksearch_codegen::LaunchHint::Elementwise { .. } => 256,
        _ => 32,
    }
}

// ---------------------------------------------------------------------------
// Q4K GEMV variant bench: hand-written MSL layouts vs the pipeline kernel.
// Finds the memory-access pattern that recovers DRAM bandwidth; the winner
// gets ported back into the renderer's Load(Q4K) expand.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
const FRAG_HELPERS: &str = r#"#include <metal_stdlib>
using namespace metal;
inline void ksearch_get_scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {
  if (j < 4u) { d = q[j] & 63u; m = q[j + 4u] & 63u; }
  else { d = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u); m = (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u); }
}
// One 32-nibble sub-chunk of a Q4_K superblock, packed uchar4 weight loads,
// x streamed from device half memory. j = nibble offset within the superblock.
inline float ksearch_bench_dot32(device const uchar* blk, uint j, device const half* x, uint xidx) {
  float d = float(*(device const half*)(blk + 0));
  float dmin = float(*(device const half*)(blk + 2));
  device const uchar* scales = blk + 4;
  device const uchar* qs = blk + 16;
  uint is = (j / 64u) * 2u;
  uint qoff = (j / 64u) * 32u;
  uint jl = j % 64u;
  float4 x0 = float4(*(device const half4*)(x + xidx + 0u));
  float4 x1 = float4(*(device const half4*)(x + xidx + 4u));
  float4 x2 = float4(*(device const half4*)(x + xidx + 8u));
  float4 x3 = float4(*(device const half4*)(x + xidx + 12u));
  float4 x4 = float4(*(device const half4*)(x + xidx + 16u));
  float4 x5 = float4(*(device const half4*)(x + xidx + 20u));
  float4 x6 = float4(*(device const half4*)(x + xidx + 24u));
  float4 x7 = float4(*(device const half4*)(x + xidx + 28u));
  packed_uchar4 qp0 = *(device const packed_uchar4*)(qs + qoff + 0u);
  packed_uchar4 qp1 = *(device const packed_uchar4*)(qs + qoff + 4u);
  packed_uchar4 qp2 = *(device const packed_uchar4*)(qs + qoff + 8u);
  packed_uchar4 qp3 = *(device const packed_uchar4*)(qs + qoff + 12u);
  packed_uchar4 qp4 = *(device const packed_uchar4*)(qs + qoff + 16u);
  packed_uchar4 qp5 = *(device const packed_uchar4*)(qs + qoff + 20u);
  packed_uchar4 qp6 = *(device const packed_uchar4*)(qs + qoff + 24u);
  packed_uchar4 qp7 = *(device const packed_uchar4*)(qs + qoff + 28u);
  uint4 q0 = uint4(qp0[0], qp0[1], qp0[2], qp0[3]);
  uint4 q1 = uint4(qp1[0], qp1[1], qp1[2], qp1[3]);
  uint4 q2 = uint4(qp2[0], qp2[1], qp2[2], qp2[3]);
  uint4 q3 = uint4(qp3[0], qp3[1], qp3[2], qp3[3]);
  uint4 q4 = uint4(qp4[0], qp4[1], qp4[2], qp4[3]);
  uint4 q5 = uint4(qp5[0], qp5[1], qp5[2], qp5[3]);
  uint4 q6 = uint4(qp6[0], qp6[1], qp6[2], qp6[3]);
  uint4 q7 = uint4(qp7[0], qp7[1], qp7[2], qp7[3]);
  float4 acc4 = float4(0.0f);
  uchar sc, mn;
  if (jl < 32u) {
    ksearch_get_scale_min_k4(is, scales, sc, mn);
    float d1 = d * float(sc);
    float4 m = float4(dmin * float(mn));
    acc4 += (d1 * float4(int4(q0 & 0xFu)) - m) * x0;
    acc4 += (d1 * float4(int4(q1 & 0xFu)) - m) * x1;
    acc4 += (d1 * float4(int4(q2 & 0xFu)) - m) * x2;
    acc4 += (d1 * float4(int4(q3 & 0xFu)) - m) * x3;
    acc4 += (d1 * float4(int4(q4 & 0xFu)) - m) * x4;
    acc4 += (d1 * float4(int4(q5 & 0xFu)) - m) * x5;
    acc4 += (d1 * float4(int4(q6 & 0xFu)) - m) * x6;
    acc4 += (d1 * float4(int4(q7 & 0xFu)) - m) * x7;
  } else {
    ksearch_get_scale_min_k4(is + 1u, scales, sc, mn);
    float d2 = d * float(sc);
    float4 m = float4(dmin * float(mn));
    acc4 += (d2 * float4(int4(q0 >> 4u)) - m) * x0;
    acc4 += (d2 * float4(int4(q1 >> 4u)) - m) * x1;
    acc4 += (d2 * float4(int4(q2 >> 4u)) - m) * x2;
    acc4 += (d2 * float4(int4(q3 >> 4u)) - m) * x3;
    acc4 += (d2 * float4(int4(q4 >> 4u)) - m) * x4;
    acc4 += (d2 * float4(int4(q5 >> 4u)) - m) * x5;
    acc4 += (d2 * float4(int4(q6 >> 4u)) - m) * x6;
    acc4 += (d2 * float4(int4(q7 >> 4u)) - m) * x7;
  }
  return acc4.x + acc4.y + acc4.z + acc4.w;
}
"#;

#[cfg(feature = "metal")]
const KERNEL_ROWLANE: &str = r#"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]])
{
  uint row = gid * 128u + lid;
  if (row >= @ROWS@u) { return; }
  uint bpr = @COLS@u / 256u;
  device const uchar* roww = w + row * bpr * 144u;
  float acc = 0.f;
  for (uint ib = 0u; ib < @NB@u; ib++) {
    device const uchar* blk = roww + ib * 144u;
    for (uint c = 0u; c < 8u; c++) {
      acc += ksearch_bench_dot32(blk, c * 32u, x, ib * 256u + c * 32u);
    }
  }
  out[row] = half(acc);
}
"#;

#[cfg(feature = "metal")]
const KERNEL_COALESCED: &str = r#"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint tiisg [[thread_index_in_simdgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]])
{
  uint row0 = (gid * @NSG@u + sgitg) * 4u;
  if (row0 + 3u >= @ROWS@u) { return; }
  uint bpr = @COLS@u / 256u;
  // 32 lanes = 4 superblocks x 8 chunks; consecutive lanes touch consecutive
  // 32B of qs memory -> coalesced transactions, zero inter-lane redundancy.
  uint sb = tiisg / 8u;
  uint c  = tiisg % 8u;
  uint j  = c * 32u;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (uint ib0 = 0u; ib0 < @NB@u; ib0 += 4u) {
    uint ib = ib0 + sb;
    if (ib < @NB@u) {
      uint xi = ib * 256u + j;
      acc0 += ksearch_bench_dot32(w + ((row0 + 0u) * bpr + ib) * 144u, j, x, xi);
      acc1 += ksearch_bench_dot32(w + ((row0 + 1u) * bpr + ib) * 144u, j, x, xi);
      acc2 += ksearch_bench_dot32(w + ((row0 + 2u) * bpr + ib) * 144u, j, x, xi);
      acc3 += ksearch_bench_dot32(w + ((row0 + 3u) * bpr + ib) * 144u, j, x, xi);
    }
  }
  acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
  if (tiisg == 0u) {
    out[row0] = half(acc0); out[row0 + 1u] = half(acc1);
    out[row0 + 2u] = half(acc2); out[row0 + 3u] = half(acc3);
  }
}
"#;

#[cfg(feature = "metal")]
const KERNEL_LOADPROBE: &str = r#"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]])
{
  uint lane = gid * 128u + lid;
  uint n = @ROWS@u * (@COLS@u / 256u) * 144u;
  uint n4 = n / 16u;
  uint ntot = @NTG@u * 128u;
  device const uchar4* p = (device const uchar4*)w;
  uint s = 0u;
  for (uint i = lane; i < n4; i += ntot) {
    s += uint(p[i][0]) + uint(p[i][1]);
  }
  if (s == 1u) { out[lane % @ROWS@u] = half(1.0f); }
}
"#;

// Verbatim llama.cpp mul_vec_q4_K_f32_impl recipe (reference/metal-llm-server
// ggml_mul_mv_q4.metal), adapted: baked dims, half activations, half dst.
#[cfg(feature = "metal")]
const KERNEL_GGMLKQ: &str = r##"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint tiisg [[thread_index_in_simdgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]])
{
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;
  const short NSG = @NSG@;
  const short nr0 = 4;
  const short ix = tiisg/8;
  const short it = tiisg%8;
  const short iq = it/4;
  const short ir = it%4;
  const int nb = @COLS@u/256u;
  const int bpr = @COLS@u/256u;
  const int first_row = (gid * NSG + sgitg) * nr0;
  if (first_row + nr0 > @ROWS@u) { return; }
  float yl[16];
  float yh[16];
  float sumf[4];
  for (short row = 0; row < nr0; ++row) sumf[row] = 0.f;
  // row stride in uint16 units
  const int rb16 = (bpr * 144u) / 2u;
  device const uint16_t* rowq = (device const uint16_t*)w + first_row * rb16;
  device const half* y = x;
  device const half* y4 = y + ix * 256u + 64u * iq + 8u * ir;
  uint16_t sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  for (int ib = ix; ib < nb; ib += 4) {
    float4 sumy = {0.f, 0.f, 0.f, 0.f};
    for (short i = 0; i < 8; ++i) {
      yl[i+0] = float(y4[i+0]);  sumy[0] += yl[i+0];
      yl[i+8] = float(y4[i+32]); sumy[1] += yl[i+8];
      yh[i+0] = float(y4[i+128]); sumy[2] += yh[i+0];
      yh[i+8] = float(y4[i+160]); sumy[3] += yh[i+8];
    }
    device const uint16_t* sc = (device const uint16_t*)((device const uchar*)(rowq + ib * 72u) + 4) + iq;
    device const uint16_t* q1 = rowq + ib * 72u + 8u + 16u * iq + 4u * ir;
    device const half* dh = (device const half*)(rowq + ib * 72u);
    for (short row = 0; row < nr0; row++) {
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      device const uint16_t* q2 = q1 + 32;
      float4 acc1 = {0.f, 0.f, 0.f, 0.f};
      float4 acc2 = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (short i = 0; i < 4; ++i) {
        acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
        acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
        acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
        acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
        acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
        acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
        acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
        acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
      }
      sumf[row] += float(dh[0]) * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                    (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                    (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                    (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                   float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
      q1 += rb16;
      sc += rb16;
      dh += rb16;
    }
    y4 += 4u * 256u;
  }
  for (int row = 0; row < nr0 && first_row + row < @ROWS@u; ++row) {
    float sum_all = simd_sum(sumf[row]);
    if (tiisg == 0) {
      out[first_row + row] = half(sum_all);
    }
  }
}
"##;

// Saturating DRAM read probe: many threads, 4 independent 16B loads in
// flight per thread per iteration. Establishes the machine's real ceiling.
#[cfg(feature = "metal")]
const KERNEL_BWPROBE: &str = r#"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]])
{
  uint lane = gid * 256u + lid;
  uint n4 = (@ROWS@u * (@COLS@u / 256u) * 144u) / 16u;
  uint ntot = @NTG@u * 256u;
  device const uint4* p = (device const uint4*)w;
  uint4 s = uint4(0u);
  for (uint i = lane; i + ntot * 3u < n4; i += ntot * 4u) {
    s ^= p[i];
    s ^= p[i + ntot];
    s ^= p[i + ntot * 2u];
    s ^= p[i + ntot * 3u];
  }
  if ((s.x ^ s.y ^ s.z ^ s.w) == 1u) { out[lane % @ROWS@u] = half(1.0f); }
}
"#;

// Split-K partial reduce: one thread per row sums KS f32 partials.
#[cfg(feature = "metal")]
const KERNEL_SPLITREDUCE: &str = r#"
kernel void k_bench(
  device const float* partial [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]])
{
  uint r = gid * 256u + lid;
  if (r >= @ROWS@u) { return; }
  float s = 0.f;
  for (uint k = 0u; k < @KS@u; k++) { s += partial[k * @ROWS@u + r]; }
  out[r] = half(s);
}
"#;

// Empty kernel: measures dispatch+sync round-trip floor.
#[cfg(feature = "metal")]
const KERNEL_EMPTY: &str = r#"
kernel void k_bench(
  device const uchar* w [[buffer(0)]],
  device const half* x [[buffer(1)]],
  device half* out [[buffer(2)]],
  uint gid [[threadgroup_position_in_grid]])
{
  if (gid == 0u) { out[0] = half(1.0f); }
}
"#;

#[cfg(not(feature = "metal"))]
fn q4k_frag_bench(_rows: usize, _cols: usize) -> Result<()> {
    bail!("`q4k-frag-bench` needs the Metal backend: rebuild with `--features metal`")
}

#[cfg(feature = "metal")]
fn q4k_frag_bench(rows: usize, cols: usize) -> Result<()> {
    use ksearch_codegen::{LaunchHint, MetalKernelSource};
    if cols % 256 != 0 || rows % 16 != 0 {
        anyhow::bail!("need cols%256==0 and rows%16==0");
    }
    let ctx = MetalContext::new()?;
    println!("Metal device: {}", ctx.device_name());

    let mut va = vec![0f32; rows * cols];
    let mut vx = vec![0f32; cols];
    for (i, v) in va.iter_mut().enumerate() {
        *v = ((i * 17) % 100) as f32 * 0.01;
    }
    for (i, v) in vx.iter_mut().enumerate() {
        *v = ((i * 13) % 50) as f32 * 0.02;
    }
    let packed = ksearch_gguf::quantize_f32_to_q4k(&va);
    let deq =
        ksearch_gguf::dequant_type_to_f32(ksearch_gguf::ggml_type::Q4_K, &packed, rows * cols);
    let mut expect = vec![0f32; rows];
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            s += deq[r * cols + c] * vx[c];
        }
        expect[r] = s;
    }
    let f16_bytes = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| ksearch_gguf::f32_to_f16(v).to_le_bytes())
            .collect()
    };
    let bw = ctx.buffer_bytes(&packed);
    let bx = ctx.buffer_bytes(&f16_bytes(&vx));
    let by = ctx.buffer_empty_f16(rows);

    let nb = cols / 256;
    let subst = |s: &str, nsg: u64| -> String {
        s.replace("@NSG@", &nsg.to_string())
            .replace("@ROWS@", &rows.to_string())
            .replace("@COLS@", &cols.to_string())
            .replace("@NB@", &nb.to_string())
    };
    let mk = |src: String, launch: LaunchHint| MetalKernelSource {
        name: "k_bench".to_string(),
        source: format!("{FRAG_HELPERS}\n{src}"),
        n_inputs: 2,
        n_outputs: 1,
        out_shape: ksearch_ir::Shape(vec![rows]),
        out_dtype: DType::F16,
        launch,
    };

    let variants: Vec<(String, MetalKernelSource)> = vec![
        (
            "coalesced".into(),
            mk(
                subst(KERNEL_COALESCED, 4),
                LaunchHint::RowsParallelSg {
                    rows: rows / 16,
                    nsg: 4,
                },
            ),
        ),
        (
            "bwprobe_ntg256".into(),
            mk(
                KERNEL_BWPROBE
                    .replace("@ROWS@", &rows.to_string())
                    .replace("@COLS@", &cols.to_string())
                    .replace("@NTG@", "256"),
                LaunchHint::RowsParallel { rows: 256, tg: 256 },
            ),
        ),
        (
            "bwprobe_ntg1024".into(),
            mk(
                KERNEL_BWPROBE
                    .replace("@ROWS@", &rows.to_string())
                    .replace("@COLS@", &cols.to_string())
                    .replace("@NTG@", "1024"),
                LaunchHint::RowsParallel {
                    rows: 1024,
                    tg: 256,
                },
            ),
        ),
        (
            "loadprobe".into(),
            mk(
                KERNEL_LOADPROBE
                    .replace("@ROWS@", &rows.to_string())
                    .replace("@COLS@", &cols.to_string())
                    .replace("@NTG@", &(rows.div_ceil(128) * 2).to_string()),
                LaunchHint::RowsParallel {
                    rows: rows.div_ceil(128) * 2,
                    tg: 128,
                },
            ),
        ),
        (
            "ggmlkq_nsg2".into(),
            mk(
                KERNEL_GGMLKQ
                    .replace("@ROWS@", &rows.to_string())
                    .replace("@COLS@", &cols.to_string())
                    .replace("@NSG@", "2"),
                LaunchHint::RowsParallelSg {
                    rows: rows.div_ceil(8),
                    nsg: 2,
                },
            ),
        ),
        (
            "ggmlkq_nsg4".into(),
            mk(
                KERNEL_GGMLKQ
                    .replace("@ROWS@", &rows.to_string())
                    .replace("@COLS@", &cols.to_string())
                    .replace("@NSG@", "4"),
                LaunchHint::RowsParallelSg {
                    rows: rows.div_ceil(16),
                    nsg: 4,
                },
            ),
        ),
        (
            "rowlane".into(),
            mk(
                subst(KERNEL_ROWLANE, 0),
                LaunchHint::RowsParallel {
                    rows: rows.div_ceil(128),
                    tg: 128,
                },
            ),
        ),
    ];

    // Pipeline baseline (current winner schedule).
    let mut g = Graph::new();
    let w = g.input(ksearch_ir::Shape(vec![rows, cols]), DType::Q4K);
    let xv = g.input(ksearch_ir::Shape(vec![cols]), DType::F16);
    let ax = g.mul_broadcast_row(w, xv)?;
    let y = g.sum_reduce(ax, 1)?;
    let winner = OptSchedule {
        tg: 128,
        vec: 1,
        unroll: 1,
        nsg: 4,
        nr0: 16,
    };
    let base = lower_with_schedule(&g, y, winner)?;

    // SLC scrubber: 64MB read between reps forces cold-DRAM conditions
    // (decode reads each weight matrix once, cold — this is the regime that
    // matters; warm-rep numbers are SLC replay artifacts).
    const SCRUB_BYTES: usize = 64 * 1024 * 1024;
    let scrub_buf = ctx.buffer_bytes(&vec![0u8; SCRUB_BYTES]);
    let scrub_rows = SCRUB_BYTES / 144;
    let scrub_src = mk(
        KERNEL_BWPROBE
            .replace("@ROWS@", &scrub_rows.to_string())
            .replace("@COLS@", "256")
            .replace("@NTG@", "512"),
        LaunchHint::RowsParallel { rows: 512, tg: 256 },
    );
    let scrub_pipe = ctx.compile(&scrub_src)?;
    let do_scrub = |by: &ksearch_metal::BufferAlias| -> Result<()> {
        ctx.run(&scrub_pipe, &scrub_src, &[&scrub_buf, &bx], by, 256)?;
        Ok(())
    };

    let time_src = |src: &MetalKernelSource, scrub: bool| -> Result<(f64, f32)> {
        let pipe = ctx.compile(src)?;
        let tg = match src.launch {
            LaunchHint::RowsParallelSg { nsg, .. } => nsg * 32,
            LaunchHint::RowsParallel { tg, .. } => tg,
            _ => 128,
        };
        ctx.run(&pipe, src, &[&bw, &bx], &by, tg)?;
        let got: Vec<f32> = ctx
            .read_u16(&by, rows)
            .into_iter()
            .map(ksearch_gguf::f16_to_f32)
            .collect();
        let err = expect
            .iter()
            .zip(got.iter())
            .map(|(e, o)| (e - o).abs())
            .fold(0f32, f32::max);
        let reps = if scrub { 7 } else { 15 };
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            if scrub {
                do_scrub(&by)?;
                samples.push(ctx.run(&pipe, src, &[&bw, &bx], &by, tg)?);
            } else {
                samples.push(ctx.run(&pipe, src, &[&bw, &bx], &by, tg)?);
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok((samples[samples.len() / 2], err))
    };

    let mb = (rows * cols * 144 / 256 + cols * 2 + rows * 2) as f64 / 1e6;
    println!(
        "shape {rows}x{cols}  traffic≈{mb:.1}MB  ({nb} superblocks/row)  [warm = SLC-resident, cold = SLC-scrubbed]"
    );

    // Split-K prototype: partition superblocks across KS× more threadgroups;
    // f32 partials + tiny reduce. Attacks small-matrix DRAM underutilization.
    let splitk = |ks: usize, scrub: bool| -> Result<(f64, f32)> {
        let rtg = rows.div_ceil(8);
        // Patch the ggmlkq source into split-K form.
        let mut src = KERNEL_GGMLKQ.to_string();
        src = src.replace(
            "const int first_row = (gid * NSG + sgitg) * nr0;",
            "const int first_row = (r0g * NSG + sgitg) * nr0;",
        );
        src = src.replace(
            "const int nb = @COLS@u/256u;",
            &format!(
                "const int nb = @COLS@u/256u;\n  const int rtgc = {rtg}u;\n  const int part = gid / rtgc;\n  const int r0g = gid % rtgc;\n  const int per = (nb + {ks} - 1) / {ks};\n  const int ib_lo = part * per;\n  const int ib_hi = min(ib_lo + per, nb);"
            ),
        );
        src = src.replace(
            "for (int ib = ix; ib < nb; ib += 4)",
            "for (int ib = ib_lo + ix; ib < ib_hi; ib += 4)",
        );
        src = src.replace(
            "device half* out [[buffer(2)]],",
            "device float* partial [[buffer(2)]],",
        );
        src = src.replace(
            "out[first_row + row] = half(sum_all);",
            "partial[part * @ROWS@u + first_row + row] = sum_all;",
        );
        src = src
            .replace("@ROWS@", &rows.to_string())
            .replace("@COLS@", &cols.to_string())
            .replace("@NSG@", "2");
        let main_k = MetalKernelSource {
            name: "k_bench".into(),
            source: format!("{}\n{}", FRAG_HELPERS, src),
            n_inputs: 2,
            n_outputs: 1,
            out_shape: ksearch_ir::Shape(vec![rows]),
            out_dtype: DType::F32,
            launch: LaunchHint::RowsParallel {
                rows: rtg * ks,
                tg: 64,
            },
        };
        let red_src = mk(
            KERNEL_SPLITREDUCE
                .replace("@ROWS@", &rows.to_string())
                .replace("@KS@", &ks.to_string()),
            LaunchHint::RowsParallel {
                rows: rows.div_ceil(256),
                tg: 256,
            },
        );
        let main_pipe = ctx.compile(&main_k)?;
        let red_pipe = ctx.compile(&red_src)?;
        let partial = ctx.buffer_empty_f32(rows * ks);

        // correctness pass
        ctx.run(&main_pipe, &main_k, &[&bw, &bx], &partial, 64)?;
        ctx.run(&red_pipe, &red_src, &[&partial, &bx], &by, 256)?;
        let got: Vec<f32> = ctx
            .read_u16(&by, rows)
            .into_iter()
            .map(ksearch_gguf::f16_to_f32)
            .collect();
        let err = expect
            .iter()
            .zip(got.iter())
            .map(|(e, o)| (e - o).abs())
            .fold(0f32, f32::max);

        let reps = if scrub { 7 } else { 15 };
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            if scrub {
                do_scrub(&by)?;
            }
            let t0 = std::time::Instant::now();
            ctx.run(&main_pipe, &main_k, &[&bw, &bx], &partial, 64)?;
            ctx.run(&red_pipe, &red_src, &[&partial, &bx], &by, 256)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok((samples[samples.len() / 2], err))
    };

    // Dispatch+sync floor: how much of each rep is NOT kernel work?
    let empty_src = mk(
        KERNEL_EMPTY.to_string(),
        LaunchHint::RowsParallel { rows: 8, tg: 64 },
    );
    let empty_pipe = ctx.compile(&empty_src)?;
    let mut es = Vec::with_capacity(30);
    for _ in 0..30 {
        es.push(ctx.run(&empty_pipe, &empty_src, &[&bw, &bx], &by, 64)?);
    }
    es.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[floor] empty kernel dispatch+sync median = {:.3} ms",
        es[es.len() / 2]
    );

    struct Row {
        name: &'static str,
        warm: Option<(f64, f32)>,
        cold: Option<(f64, f32)>,
    }
    let mut table: Vec<Row> = Vec::new();

    // Warm + cold rounds for each candidate.
    let cases: Vec<(&str, &MetalKernelSource)> = vec![
        ("pipeline", &base),
        (
            "ggmlkq_nsg2",
            variants
                .iter()
                .find(|(n, _)| n == "ggmlkq_nsg2")
                .map(|(_, s)| s)
                .unwrap(),
        ),
    ];
    for round in 0..2 {
        for (name, src) in &cases {
            match time_src(src, round == 1) {
                Ok((ms, err)) => {
                    let slot = if round == 0 { 0 } else { 1 };
                    if let Some(r) = table.iter_mut().find(|r| r.name == *name) {
                        if slot == 0 {
                            r.warm = Some((ms, err));
                        } else {
                            r.cold = Some((ms, err));
                        }
                    } else {
                        let (warm, cold) = if slot == 0 {
                            (Some((ms, err)), None)
                        } else {
                            (None, Some((ms, err)))
                        };
                        table.push(Row {
                            name: name,
                            warm,
                            cold,
                        });
                    }
                }
                Err(e) => println!("{name}: FAILED {e}"),
            }
        }
        match splitk(2, round == 1) {
            Ok((ms, err)) => {
                let e = table.iter_mut().find(|r| r.name == "splitk2");
                if round == 0 {
                    table.push(Row {
                        name: "splitk2",
                        warm: Some((ms, err)),
                        cold: None,
                    });
                } else if let Some(r) = e {
                    r.cold = Some((ms, err));
                }
            }
            Err(e) => println!("splitk2: FAILED {e}"),
        }
        match splitk(4, round == 1) {
            Ok((ms, err)) => {
                let e = table.iter_mut().find(|r| r.name == "splitk4");
                if round == 0 {
                    table.push(Row {
                        name: "splitk4",
                        warm: Some((ms, err)),
                        cold: None,
                    });
                } else if let Some(r) = e {
                    r.cold = Some((ms, err));
                }
            }
            Err(e) => println!("splitk4: FAILED {e}"),
        }
    }

    println!(
        "{:^17} {:^18} {:^18}",
        "kernel", "warm (SLC)", "cold (DRAM)"
    );
    for r in &table {
        let fmt = |v: &Option<(f64, f32)>| match v {
            Some((ms, _)) => format!("{:7.3} ms {:5.1}", ms, mb / ms),
            None => "—".to_string(),
        };
        println!("{:^17} {:^18} {:^18}", r.name, fmt(&r.warm), fmt(&r.cold));
    }
    Ok(())
}

/// `ksearch cuda` — inspect / validate the CUDA backend without a GPU.
///
/// Renders every kernel in [`ksearch_codegen::smoke`] for the CUDA target and
/// prints the grid/block/shared-memory configuration the runtime would use.
/// With `--compile` each source is additionally passed through `nvcc --cubin`
/// for the compute capability of the device in the machine, which is the
/// end-to-end check on an H100 / RTX 50-series box.
fn cuda(out: Option<PathBuf>, compile: bool, device: usize, tg_size: u64) -> Result<()> {
    use ksearch_codegen::smoke;
    use ksearch_codegen::{render_cuda, OptSchedule};
    use ksearch_cuda::{launch_config, CudaContext};

    let ctx = match CudaContext::with_options(device, tg_size) {
        Ok(ctx) => {
            let i = ctx.info();
            println!(
                "CUDA device {device}: {} ({}, {} threads/block max, {} KiB shared max)",
                i.name,
                i.sm_arch(),
                i.max_threads_per_block,
                i.max_shared_per_block / 1024
            );
            match ctx.nvcc() {
                Some(p) => println!("nvcc: {}", p.display()),
                None => println!("nvcc: not found (install the CUDA toolkit to compile)"),
            }
            Some(ctx)
        }
        Err(e) => {
            println!("CUDA device {device}: unavailable ({e})");
            None
        }
    };

    if compile && ctx.is_none() {
        bail!("--compile needs a CUDA device so nvcc can target its compute capability");
    }

    let mut failures = 0usize;
    for k in smoke::smoke_kernels() {
        let src = render_cuda(&k, OptSchedule::default())?;
        let cfg = launch_config(&src.launch, tg_size);
        println!(
            "{:<16} grid=({},{},{}) block=({},{},{}) smem={}  in={} out={}  {} bytes",
            src.name,
            cfg.grid.0,
            cfg.grid.1,
            cfg.grid.2,
            cfg.block.0,
            cfg.block.1,
            cfg.block.2,
            cfg.smem,
            src.n_inputs,
            src.n_outputs,
            src.source.len()
        );
        if let Some(dir) = &out {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!("{}.cu", src.name));
            std::fs::write(&path, &src.source)?;
            println!("  wrote {}", path.display());
        }
        if let Some(ctx) = &ctx {
            if compile {
                match ctx.compile(&src.source, &src.name) {
                    Ok(cubin) => println!("  nvcc: ok ({} bytes cubin)", cubin.len()),
                    Err(e) => {
                        failures += 1;
                        eprintln!("  nvcc: FAILED\n{e}");
                    }
                }
            }
        }
    }

    if failures > 0 {
        bail!("{failures} kernel(s) failed to compile");
    }
    Ok(())
}

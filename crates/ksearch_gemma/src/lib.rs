//! Gemma4 forward — Thesis A [`GemmaPrimModel`] only.

mod gemma_prim;
mod kv_pool;
mod sample;

pub use gemma_prim::{GemmaPrimModel, PREFILL_CHUNK_SIZE};
pub use sample::sample_softcap_min_p;
pub use kv_pool::{KvPool, KvSlot, SlotId};

use anyhow::{anyhow, Result};
use ksearch_gguf::Gguf;
use ksearch_ir::q40_nbytes;
use metal::Buffer;

#[derive(Clone)]
pub struct GemmaConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim_swa: usize,
    pub head_dim_full: usize,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub ffn: Vec<usize>,
    pub swa_pattern: Vec<bool>,
    pub rms_eps: f32,
    pub softcap: f32,
    pub vocab: usize,
    pub ple_dim: usize,
    pub rope_theta_full: f32,
    pub rope_theta_swa: f32,
    pub partial_rotary: f32,
    pub context_length: usize,
}

impl GemmaConfig {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        if let Some(arch) = g.get_str("general.architecture") {
            if arch != "gemma4" {
                return Err(anyhow!("unsupported GGUF architecture `{arch}` (expected gemma4)"));
            }
        }
        if g.get_u32("gemma4.expert_count").unwrap_or(0) > 0 {
            return Err(anyhow!("Gemma 4 MoE (A4B) is not supported"));
        }
        let n_layers = g
            .get_u32("gemma4.block_count")
            .ok_or_else(|| anyhow!("block_count"))? as usize;
        let hidden = g.get_u32("gemma4.embedding_length").unwrap() as usize;
        let n_heads = g.get_u32("gemma4.attention.head_count").unwrap() as usize;
        let n_kv_list = g
            .get_usize_list("gemma4.attention.head_count_kv")
            .unwrap_or_else(|| vec![1]);
        if n_kv_list.len() > 1 {
            return Err(anyhow!(
                "per-layer head_count_kv is not supported (A4B/MoE)"
            ));
        }
        let n_kv = n_kv_list.first().copied().unwrap_or(1).max(1);
        if n_heads % n_kv != 0 {
            return Err(anyhow!("head_count {n_heads} not divisible by head_count_kv {n_kv}"));
        }
        let head_dim_full = g.get_u32("gemma4.attention.key_length").unwrap_or(512) as usize;
        let head_dim_swa = g.get_u32("gemma4.attention.key_length_swa").unwrap_or(256) as usize;
        let sliding_window = g.get_u32("gemma4.attention.sliding_window").unwrap_or(512) as usize;
        let shared_kv_layers = g.get_u32("gemma4.attention.shared_kv_layers").unwrap_or(0) as usize;
        let rms_eps = g
            .get_f32("gemma4.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);
        let softcap = g.get_f32("gemma4.final_logit_softcapping").unwrap_or(30.0);
        let ple_dim = g
            .get_u32("gemma4.embedding_length_per_layer_input")
            .unwrap_or(256) as usize;
        let rope_theta_full = g.get_f32("gemma4.rope.freq_base").unwrap_or(1_000_000.0);
        let rope_theta_swa = g.get_f32("gemma4.rope.freq_base_swa").unwrap_or(10_000.0);

        let ffn = g
            .get_usize_list("gemma4.feed_forward_length")
            .unwrap_or_else(|| {
                vec![g.get_u32("gemma4.feed_forward_length").unwrap_or(6144) as usize; n_layers]
            });
        let ffn = if ffn.len() == 1 {
            vec![ffn[0]; n_layers]
        } else if ffn.len() == n_layers {
            ffn
        } else {
            vec![6144; n_layers]
        };
        let swa_pattern = g
            .get_arr_bool("gemma4.attention.sliding_window_pattern")
            .map(|v| v.to_vec())
            .unwrap_or_else(|| (0..n_layers).map(|i| (i + 1) % 6 != 0).collect());

        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.n_rows())
            .unwrap_or(262144);
        let context_length = g.get_u32("gemma4.context_length").unwrap_or(131072) as usize;

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv,
            head_dim_swa,
            head_dim_full,
            sliding_window,
            shared_kv_layers,
            ffn,
            swa_pattern,
            rms_eps,
            softcap,
            vocab,
            ple_dim,
            rope_theta_full,
            rope_theta_swa,
            partial_rotary: 0.25,
            context_length,
        })
    }

    pub fn n_kv_owners(&self) -> usize {
        (0..self.n_layers)
            .filter(|&i| self.owns_kv(i))
            .count()
            .max(1)
    }

    /// Q4_0 K+V bytes for `slots` sequences of `max_seq` (pool uses max head dim).
    pub fn kv_q40_bytes(&self, max_seq: usize, slots: usize) -> usize {
        let hd = self.head_dim_full.max(self.head_dim_swa);
        slots * self.n_kv_owners() * 2 * q40_nbytes(max_seq * self.n_kv, hd)
    }

    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa_pattern.get(layer).copied().unwrap_or(true)
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_swa(layer) {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    pub fn owns_kv(&self, layer: usize) -> bool {
        layer < self.n_layers.saturating_sub(self.shared_kv_layers)
    }

    pub fn kv_source(&self, layer: usize) -> usize {
        if self.owns_kv(layer) {
            layer
        } else {
            let want_swa = self.is_swa(layer);
            (0..layer)
                .rev()
                .find(|&i| self.owns_kv(i) && self.is_swa(i) == want_swa)
                .unwrap_or(0)
        }
    }

    pub fn ple_total(&self) -> usize {
        self.n_layers * self.ple_dim
    }
}

pub(crate) struct LayerMeta {
    pub q_rows: usize,
    pub kv_rows: usize,
    pub o_in: usize,
    pub ffn_inter: usize,
    pub hd: usize,
}

pub(crate) struct LayerNorms {
    pub attn_norm: Buffer,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub post_attn_norm: Buffer,
    pub ffn_norm: Buffer,
    pub post_ffw_norm: Buffer,
    pub post_norm: Buffer,
    pub layer_scale: f32,
}

/// Timing + tokens from generate_timed.
pub struct GenerateStats {
    pub tokens: Vec<u32>,
    pub prefill_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
}

impl GenerateStats {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill_s.max(1e-6)
    }

    pub fn decode_tok_s(&self) -> f64 {
        self.tokens.len() as f64 / self.decode_s.max(1e-6)
    }
}

//! KvPool: continuous-batching style KV slots for B≥1 decode (Thesis A Phase E).
//!
//! Metal buffers are owned by the pool; slots track occupancy and sequence length.
//! Full multi-stream decode wiring lands on top of this; the pool itself is the
//! serving contract (SlotId + capacity) called out in DESIGN.md.

use anyhow::{bail, Result};
use ksearch_ir::q40_nbytes;
use ksearch_metal::MetalContext;
use metal::Buffer;

/// Opaque slot handle for a resident sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(pub u32);

pub struct KvSlot {
    pub id: SlotId,
    pub occupied: bool,
    pub seq_len: usize,
    pub max_seq: usize,
}

/// Pool of KV cache rows for `n_kv_layers` physical KV layers (shared-KV aware).
pub struct KvPool {
    pub slots: Vec<KvSlot>,
    pub max_batch: usize,
    pub max_seq: usize,
    pub n_kv_layers: usize,
    pub hd: usize,
    /// KV heads per token (GQA; 1 = MQA).
    pub n_kv: usize,
    /// True when buffers are float elems `[max_seq * hd]` (F16 or F32); false = Q4_0 packs.
    pub f32_kv: bool,
    /// Element size when `f32_kv` (2=F16, 4=F32).
    pub kv_elem_bytes: usize,
    /// Per slot, per kv-layer: K and V caches.
    pub k: Vec<Vec<Buffer>>,
    pub v: Vec<Vec<Buffer>>,
}

impl KvPool {
    /// Q4_0 packed KV (debt / bandwidth path).
    pub fn new(
        ctx: &MetalContext,
        max_batch: usize,
        max_seq: usize,
        n_kv_layers: usize,
        hd: usize,
        n_kv: usize,
    ) -> Result<Self> {
        Self::new_inner(ctx, max_batch, max_seq, n_kv_layers, hd, n_kv, false, 0)
    }

    /// F32 KV (legacy).
    pub fn new_f32(
        ctx: &MetalContext,
        max_batch: usize,
        max_seq: usize,
        n_kv_layers: usize,
        hd: usize,
        n_kv: usize,
    ) -> Result<Self> {
        Self::new_inner(ctx, max_batch, max_seq, n_kv_layers, hd, n_kv, true, 4)
    }

    /// F16 KV for Thesis A prim path (tinygrad `.half()`).
    pub fn new_f16(
        ctx: &MetalContext,
        max_batch: usize,
        max_seq: usize,
        n_kv_layers: usize,
        hd: usize,
        n_kv: usize,
    ) -> Result<Self> {
        Self::new_inner(ctx, max_batch, max_seq, n_kv_layers, hd, n_kv, true, 2)
    }

    fn new_inner(
        ctx: &MetalContext,
        max_batch: usize,
        max_seq: usize,
        n_kv_layers: usize,
        hd: usize,
        n_kv: usize,
        float_kv: bool,
        elem_bytes: usize,
    ) -> Result<Self> {
        if max_batch == 0 || max_seq == 0 || n_kv_layers == 0 || hd == 0 || n_kv == 0 {
            bail!("KvPool dims must be non-zero");
        }
        let mut slots = Vec::with_capacity(max_batch);
        let mut k = Vec::with_capacity(max_batch);
        let mut v = Vec::with_capacity(max_batch);
        for i in 0..max_batch {
            slots.push(KvSlot {
                id: SlotId(i as u32),
                occupied: false,
                seq_len: 0,
                max_seq,
            });
            let mut ks = Vec::with_capacity(n_kv_layers);
            let mut vs = Vec::with_capacity(n_kv_layers);
            for _ in 0..n_kv_layers {
                if float_kv {
                    let n = max_seq * n_kv * hd;
                    if elem_bytes == 2 {
                        ks.push(ctx.buffer_empty_f16(n));
                        vs.push(ctx.buffer_empty_f16(n));
                    } else {
                        ks.push(ctx.buffer_empty_f32(n));
                        vs.push(ctx.buffer_empty_f32(n));
                    }
                } else {
                    let nbytes = q40_nbytes(max_seq * n_kv, hd);
                    ks.push(ctx.buffer_empty_bytes(nbytes));
                    vs.push(ctx.buffer_empty_bytes(nbytes));
                }
            }
            k.push(ks);
            v.push(vs);
        }
        Ok(Self {
            slots,
            max_batch,
            max_seq,
            n_kv_layers,
            hd,
            n_kv,
            f32_kv: float_kv,
            kv_elem_bytes: elem_bytes,
            k,
            v,
        })
    }

    pub fn alloc(&mut self) -> Result<SlotId> {
        for s in &mut self.slots {
            if !s.occupied {
                s.occupied = true;
                s.seq_len = 0;
                return Ok(s.id);
            }
        }
        bail!("KvPool full (max_batch={})", self.max_batch)
    }

    /// Zero K/V packs for a slot (call after [`alloc`] before first prefill).
    pub fn clear_slot(&self, id: SlotId) -> Result<()> {
        let i = id.0 as usize;
        if i >= self.max_batch {
            bail!("bad SlotId");
        }
        for b in self.k[i].iter().chain(self.v[i].iter()) {
            let n = b.length() as usize;
            unsafe {
                std::ptr::write_bytes(b.contents() as *mut u8, 0, n);
            }
        }
        Ok(())
    }

    pub fn seq_len(&self, id: SlotId) -> Result<usize> {
        self.slots
            .get(id.0 as usize)
            .map(|s| s.seq_len)
            .ok_or_else(|| anyhow::anyhow!("bad SlotId"))
    }

    pub fn free(&mut self, id: SlotId) -> Result<()> {
        let s = self
            .slots
            .get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("bad SlotId"))?;
        s.occupied = false;
        s.seq_len = 0;
        Ok(())
    }

    pub fn active_batch(&self) -> Vec<SlotId> {
        self.slots
            .iter()
            .filter(|s| s.occupied)
            .map(|s| s.id)
            .collect()
    }

    pub fn k_buf(&self, slot: SlotId, layer: usize) -> Result<&Buffer> {
        self.k
            .get(slot.0 as usize)
            .and_then(|layers| layers.get(layer))
            .ok_or_else(|| anyhow::anyhow!("bad kv index"))
    }

    pub fn v_buf(&self, slot: SlotId, layer: usize) -> Result<&Buffer> {
        self.v
            .get(slot.0 as usize)
            .and_then(|layers| layers.get(layer))
            .ok_or_else(|| anyhow::anyhow!("bad kv index"))
    }

    pub fn bump_len(&mut self, slot: SlotId) -> Result<usize> {
        let s = self
            .slots
            .get_mut(slot.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("bad SlotId"))?;
        if !s.occupied {
            bail!("slot not occupied");
        }
        if s.seq_len >= s.max_seq {
            bail!("slot seq full");
        }
        s.seq_len += 1;
        Ok(s.seq_len)
    }

    pub fn bump_len_by(&mut self, slot: SlotId, n: usize) -> Result<usize> {
        let s = self
            .slots
            .get_mut(slot.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("bad SlotId"))?;
        if !s.occupied {
            bail!("slot not occupied");
        }
        if s.seq_len + n > s.max_seq {
            bail!("slot seq full");
        }
        s.seq_len += n;
        Ok(s.seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_batch() {
        let ctx = MetalContext::new().expect("metal");
        let mut pool = KvPool::new(&ctx, 4, 64, 2, 256, 1).expect("pool");
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_ne!(a, b);
        assert_eq!(pool.active_batch().len(), 2);
        pool.free(a).unwrap();
        assert_eq!(pool.active_batch().len(), 1);
        let c = pool.alloc().unwrap();
        assert_eq!(c, a); // reuses freed slot
    }
}

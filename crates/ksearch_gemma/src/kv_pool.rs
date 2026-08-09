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

/// Pool of KV cache rows for `n_layers_kv` physical KV layers (shared-KV aware).
pub struct KvPool {
    pub slots: Vec<KvSlot>,
    pub max_batch: usize,
    pub max_seq: usize,
    pub n_kv_layers: usize,
    pub hd: usize,
    /// Per slot, per kv-layer: K and V Q4_0 caches `[max_seq * row_bytes]`.
    pub k: Vec<Vec<Buffer>>,
    pub v: Vec<Vec<Buffer>>,
}

impl KvPool {
    pub fn new(
        ctx: &MetalContext,
        max_batch: usize,
        max_seq: usize,
        n_kv_layers: usize,
        hd: usize,
    ) -> Result<Self> {
        if max_batch == 0 || max_seq == 0 || n_kv_layers == 0 || hd == 0 {
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
                let nbytes = q40_nbytes(max_seq, hd);
                ks.push(ctx.buffer_empty_bytes(nbytes));
                vs.push(ctx.buffer_empty_bytes(nbytes));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_batch() {
        let ctx = MetalContext::new().expect("metal");
        let mut pool = KvPool::new(&ctx, 4, 64, 2, 256).expect("pool");
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

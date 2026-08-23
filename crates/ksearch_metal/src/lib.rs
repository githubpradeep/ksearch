//! Metal device runtime for generated kernels.

use anyhow::{anyhow, Context, Result};
use ksearch_codegen::{LaunchHint, MetalKernelSource};
use metal::*;
use objc::rc::autoreleasepool;
use std::cell::RefCell;
use std::ffi::c_void;
use std::time::Instant;

struct Pending {
    cmd: CommandBuffer,
    enc: ComputeCommandEncoder,
}

/// Dispatch telemetry (decode-cost audit): count of kernel dispatches and
/// total host CPU nanos spent inside encode. Read-and-reset via [`take_telemetry`].
static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENQUEUE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (dispatches, host-enqueue seconds) since last call; resets both counters.
pub fn take_telemetry() -> (u64, f64) {
    use std::sync::atomic::Ordering;
    (
        DISPATCH_COUNT.swap(0, Ordering::Relaxed),
        ENQUEUE_NANOS.swap(0, Ordering::Relaxed) as f64 / 1e9,
    )
}

/// Re-export for host-side type hints.
pub type BufferAlias = Buffer;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    /// Open CB + compute encoder (many dispatches, one encoder).
    pending: RefCell<Option<Pending>>,
    /// GPU timestamp counter buffer for phase profiling (KSEARCH_PHASE_GPUTIME).
    phase_csb: RefCell<Option<CounterSampleBuffer>>,
    phase_index: std::cell::Cell<usize>,
    /// Committed CBs not yet waited on (retain until synchronize).
    inflight: RefCell<Vec<CommandBuffer>>,
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pending: RefCell::new(None),
            phase_csb: RefCell::new(None),
            phase_index: std::cell::Cell::new(0),
            inflight: RefCell::new(Vec::new()),
        })
    }

    /// Enable GPU-timestamp sampling (Apple "timestamp" counter set).
    /// Idempotent; returns Err if the device lacks the counter set.
    pub fn init_phase_counters(&self, max_samples: u64) -> Result<()> {
        if self.phase_csb.borrow().is_some() {
            return Ok(());
        }
        let sets = self.device.counter_sets();
        let ts = sets
            .iter()
            .find(|s| s.name().to_ascii_lowercase().contains("timestamp"))
            .ok_or_else(|| anyhow!("device has no timestamp counter set"))?;
        let desc = CounterSampleBufferDescriptor::new();
        desc.set_counter_set(ts);
        desc.set_sample_count(max_samples);
        desc.set_storage_mode(MTLStorageMode::Private);
        let csb = self
            .device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .map_err(|e| anyhow!("counter sample buffer: {e}"))?;
        self.phase_index.set(0);
        *self.phase_csb.borrow_mut() = Some(csb);
        Ok(())
    }

    /// Sample one GPU timestamp at this point in the open encoder
    /// (`with_barrier = false` keeps execution pipelined).
    pub fn phase_mark(&self) {
        if let Some(csb) = self.phase_csb.borrow().as_ref() {
            if let Some(p) = self.pending.borrow_mut().as_mut() {
                let i = self.phase_index.get();
                p.enc
                    .sample_counters_in_buffer(csb, i as NSUInteger, false);
                self.phase_index.set(i + 1);
            }
        }
    }

    /// Per-interval GPU times in ms between consecutive marks (len = marks-1).
    pub fn resolve_phase_times_ms(&self) -> Vec<f64> {
        let n = self.phase_index.get();
        if n < 2 {
            return Vec::new();
        }
        let vals = {
            let borrow = self.phase_csb.borrow();
            let csb = borrow.as_ref().expect("init_phase_counters first");
            csb.resolve_counter_range(NSRange {
                location: 0,
                length: n as NSUInteger,
            })
        };
        vals.windows(2)
            .map(|w| w[1].saturating_sub(w[0]) as f64 / 1e6)
            .collect()
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn compile(&self, kernel: &MetalKernelSource) -> Result<ComputePipelineState> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = self
            .device
            .new_library_with_source(&kernel.source, &options)
            .map_err(|e| anyhow!("MSL compile failed: {e}\n{}", kernel.source))?;
        let function = library
            .get_function(&kernel.name, None)
            .map_err(|e| anyhow!("function {}: {e}", kernel.name))?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("pipeline: {e}"))
            .context("pipeline")
    }

    pub fn buffer_f32(&self, data: &[f32]) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = std::mem::size_of_val(data) as u64;
        self.device
            .new_buffer_with_data(ptr, len, MTLResourceOptions::StorageModeShared)
    }

    pub fn buffer_bytes(&self, data: &[u8]) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = data.len() as u64;
        self.device
            .new_buffer_with_data(ptr, len, MTLResourceOptions::StorageModeShared)
    }

    pub fn buffer_empty_f32(&self, n: usize) -> Buffer {
        self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer_empty_f16(&self, n: usize) -> Buffer {
        self.buffer_empty_bytes(n * 2)
    }

    pub fn buffer_empty_bytes(&self, n: usize) -> Buffer {
        self.device
            .new_buffer(n as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Raw byte range read from a buffer (forces sync).
    pub fn read_bytes(&self, buf: &Buffer, offset_bytes: usize, len_bytes: usize) -> Vec<u8> {
        unsafe {
            let ptr = (buf.contents() as *const u8).add(offset_bytes);
            std::slice::from_raw_parts(ptr, len_bytes).to_vec()
        }
    }

    pub fn read_f32(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        self.synchronize().ok();
        self.read_f32_nosync(buf, n)
    }

    /// Host read without flushing GPU. Caller must ensure `buf` is not a pending GPU write.
    pub fn read_f32_nosync(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    pub fn read_u16(&self, buf: &Buffer, n: usize) -> Vec<u16> {
        self.synchronize().ok();
        let ptr = buf.contents() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    pub fn write_buffer(&self, buf: &Buffer, data: &[f32]) {
        self.synchronize().ok();
        self.write_buffer_nosync(buf, data);
    }

    /// Host write without flushing GPU. Caller must ensure `buf` is not a pending GPU write target.
    pub fn write_buffer_nosync(&self, buf: &Buffer, data: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn write_u16s(&self, buf: &Buffer, data: &[u16]) {
        self.synchronize().ok();
        self.write_u16s_nosync(buf, data);
    }

    /// Host write without flushing GPU. Caller must ensure `buf` is not a pending GPU read target
    /// (e.g. double-buffer + [`wait_inflight_at_most`]).
    pub fn write_u16s_nosync(&self, buf: &Buffer, data: &[u16]) {
        let ptr = buf.contents() as *mut u16;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn ensure_pending(&self) -> std::cell::RefMut<'_, Option<Pending>> {
        let mut slot = self.pending.borrow_mut();
        if slot.is_none() {
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            *slot = Some(Pending { cmd, enc });
        }
        slot
    }

    fn end_pending(pending: Pending, wait: bool) -> CommandBuffer {
        pending.enc.end_encoding();
        pending.cmd.commit();
        if wait {
            pending.cmd.wait_until_completed();
        }
        pending.cmd
    }

    /// True if a compute encoder is open or command buffers are in flight.
    pub fn has_gpu_work(&self) -> bool {
        self.pending.borrow().is_some() || !self.inflight.borrow().is_empty()
    }

    /// Commit pending work without waiting (overlap CPU encode with GPU).
    pub fn flush_async(&self) {
        autoreleasepool(|| {
            if let Some(p) = self.pending.borrow_mut().take() {
                let cmd = Self::end_pending(p, false);
                self.inflight.borrow_mut().push(cmd);
            }
        });
    }

    /// Block until at most `max_inflight` command buffers remain (drop completed from the front).
    /// Use with double-buffered host writes: `max_inflight=1` keeps one CB in flight.
    pub fn wait_inflight_at_most(&self, max_inflight: usize) {
        let mut inflight = self.inflight.borrow_mut();
        while inflight.len() > max_inflight {
            if let Some(cmd) = inflight.first() {
                cmd.wait_until_completed();
            }
            inflight.remove(0);
        }
        // Also reap any already-completed CBs at the front.
        while let Some(cmd) = inflight.first() {
            // status 4 == Completed in metal-rs; fall back to not spinning — only peek via wait with timeout unavailable.
            // If still running, stop reaping.
            use metal::MTLCommandBufferStatus;
            if cmd.status() == MTLCommandBufferStatus::Completed {
                inflight.remove(0);
            } else {
                break;
            }
        }
    }

    /// End the current compute encoder and start a new one on the same CB.
    /// Serializes prior dispatches vs subsequent ones without a CPU wait.
    pub fn encoder_barrier(&self) {
        let mut slot = self.pending.borrow_mut();
        if let Some(p) = slot.as_mut() {
            p.enc.end_encoding();
            p.enc = p.cmd.new_compute_command_encoder().to_owned();
        }
    }

    /// GPU buffer→buffer copy on the open command buffer (no CPU wait).
    /// Ends any open compute encoder, blits, then opens a fresh compute encoder.
    pub fn blit_copy_bytes(
        &self,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        nbytes: u64,
    ) {
        autoreleasepool(|| {
            let mut slot = self.pending.borrow_mut();
            if slot.is_none() {
                let cmd = self.queue.new_command_buffer().to_owned();
                *slot = Some(Pending {
                    cmd: cmd.clone(),
                    enc: cmd.new_compute_command_encoder().to_owned(),
                });
            }
            let p = slot.as_mut().unwrap();
            p.enc.end_encoding();
            let blit = p.cmd.new_blit_command_encoder();
            blit.copy_from_buffer(src, src_off, dst, dst_off, nbytes);
            blit.end_encoding();
            p.enc = p.cmd.new_compute_command_encoder().to_owned();
        });
    }

    /// Commit pending work and wait for all in-flight CBs (call before host reads).
    pub fn synchronize(&self) -> Result<()> {
        autoreleasepool(|| {
            if let Some(p) = self.pending.borrow_mut().take() {
                let cmd = Self::end_pending(p, false);
                self.inflight.borrow_mut().push(cmd);
            }
            let mut inflight = self.inflight.borrow_mut();
            if let Some(last) = inflight.last() {
                last.wait_until_completed();
            }
            inflight.clear();
            Ok(())
        })
    }

    /// Encode into the open compute encoder (no wait). Many dispatches share one encoder.
    pub fn encode(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        output: &Buffer,
        tg_size: u64,
    ) -> Result<()> {
        self.encode_multi(pipeline, kernel, inputs, &[output], tg_size)
    }

    /// Like [`encode`] with one or more output buffers (`n_outputs`).
    pub fn encode_multi(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        tg_size: u64,
    ) -> Result<()> {
        let in_offs: Vec<u64> = inputs.iter().map(|_| 0u64).collect();
        let out_offs: Vec<u64> = outputs.iter().map(|_| 0u64).collect();
        self.encode_offsets_multi(pipeline, kernel, inputs, &in_offs, outputs, &out_offs, tg_size)
    }

    /// Like [`encode`], but each input/output may start at a byte offset into its buffer.
    pub fn encode_offsets(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        input_byte_offsets: &[u64],
        output: &Buffer,
        output_byte_offset: u64,
        tg_size: u64,
    ) -> Result<()> {
        self.encode_offsets_multi(
            pipeline,
            kernel,
            inputs,
            input_byte_offsets,
            &[output],
            &[output_byte_offset],
            tg_size,
        )
    }

    /// Multi-output encode with per-buffer byte offsets.
    pub fn encode_offsets_multi(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        input_byte_offsets: &[u64],
        outputs: &[&Buffer],
        output_byte_offsets: &[u64],
        tg_size: u64,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        let r = self.encode_offsets_multi_inner(
            pipeline,
            kernel,
            inputs,
            input_byte_offsets,
            outputs,
            output_byte_offsets,
            tg_size,
        );
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ENQUEUE_NANOS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        r
    }

    fn encode_offsets_multi_inner(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        input_byte_offsets: &[u64],
        outputs: &[&Buffer],
        output_byte_offsets: &[u64],
        tg_size: u64,
    ) -> Result<()> {
        let n_out = kernel.n_outputs.max(1);
        if inputs.len() != kernel.n_inputs {
            return Err(anyhow!(
                "expected {} inputs, got {}",
                kernel.n_inputs,
                inputs.len()
            ));
        }
        if input_byte_offsets.len() != inputs.len() {
            return Err(anyhow!(
                "expected {} input offsets, got {}",
                inputs.len(),
                input_byte_offsets.len()
            ));
        }
        if outputs.len() != n_out {
            return Err(anyhow!(
                "expected {} outputs, got {}",
                n_out,
                outputs.len()
            ));
        }
        if output_byte_offsets.len() != outputs.len() {
            return Err(anyhow!(
                "expected {} output offsets, got {}",
                outputs.len(),
                output_byte_offsets.len()
            ));
        }
        // One encoder/CB per token: do not open an autorelease pool per dispatch.
        let mut slot = self.ensure_pending();
        let pending = slot.as_mut().unwrap();
        let enc = &pending.enc;
        enc.set_compute_pipeline_state(pipeline);
        for (i, b) in inputs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), input_byte_offsets[i]);
        }
        for (o, b) in outputs.iter().enumerate() {
            enc.set_buffer(
                (kernel.n_inputs + o) as u64,
                Some(b),
                output_byte_offsets[o],
            );
        }

        let (n_tg, tg) = match &kernel.launch {
            LaunchHint::Elementwise { n } => {
                let tg = tg_size.min(*n as u64).max(1);
                let n_tg = (*n as u64 + tg - 1) / tg;
                (n_tg, tg)
            }
            LaunchHint::Rows { rows, .. } => {
                let tg = tg_size.min(*rows as u64).max(1);
                let n_tg = (*rows as u64 + tg - 1) / tg;
                (n_tg, tg)
            }
            LaunchHint::RowsParallel { rows, tg } => (*rows as u64, *tg),
            LaunchHint::RowsParallelSg { rows, nsg } => (*rows as u64, nsg * 32),
            LaunchHint::RowsParallel2D { rows, batch, tg } => {
                let _ = (rows, batch, tg);
                (*rows as u64 * *batch as u64, *tg)
            }
            LaunchHint::MulMm { .. } => (1, 1),
        };
        match &kernel.launch {
            LaunchHint::RowsParallel2D { rows, batch, tg } => {
                enc.dispatch_thread_groups(
                    MTLSize::new(*rows as u64, *batch as u64, 1),
                    MTLSize::new(*tg, 1, 1),
                );
            }
            LaunchHint::RowsParallelSg { rows, nsg } => {
                enc.dispatch_thread_groups(
                    MTLSize::new(*rows as u64, 1, 1),
                    MTLSize::new(32, *nsg, 1),
                );
            }
            LaunchHint::MulMm {
                tg_x,
                tg_y,
                tw,
                nsg,
                smem,
            } => {
                enc.set_threadgroup_memory_length(0, *smem);
                enc.dispatch_thread_groups(
                    MTLSize::new(*tg_x, *tg_y, 1),
                    MTLSize::new(*tw, *nsg, 1),
                );
            }
            _ => {
                enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            }
        }
        Ok(())
    }

    /// Run once with immediate wait; returns wall ms (BEAM timing).
    pub fn run(
        &self,
        pipeline: &ComputePipelineState,
        kernel: &MetalKernelSource,
        inputs: &[&Buffer],
        output: &Buffer,
        tg_size: u64,
    ) -> Result<f64> {
        self.synchronize()?;
        if inputs.len() != kernel.n_inputs {
            return Err(anyhow!(
                "expected {} inputs, got {}",
                kernel.n_inputs,
                inputs.len()
            ));
        }
        autoreleasepool(|| {
            let t0 = Instant::now();
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            for (i, b) in inputs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_buffer(kernel.n_inputs as u64, Some(output), 0);

            let (n_tg, tg) = match &kernel.launch {
                LaunchHint::Elementwise { n } => {
                    let tg = tg_size.min(*n as u64).max(1);
                    let n_tg = (*n as u64 + tg - 1) / tg;
                    (n_tg, tg)
                }
                LaunchHint::Rows { rows, .. } => {
                    let tg = tg_size.min(*rows as u64).max(1);
                    let n_tg = (*rows as u64 + tg - 1) / tg;
                    (n_tg, tg)
                }
                LaunchHint::RowsParallel { rows, tg } => (*rows as u64, *tg),
                LaunchHint::RowsParallelSg { rows, nsg } => (*rows as u64, nsg * 32),
                LaunchHint::RowsParallel2D { rows, batch, tg } => {
                    (*rows as u64 * *batch as u64, *tg)
                }
                LaunchHint::MulMm { .. } => (1, 1),
            };
            match &kernel.launch {
                LaunchHint::RowsParallel2D { rows, batch, tg } => {
                    enc.dispatch_thread_groups(
                        MTLSize::new(*rows as u64, *batch as u64, 1),
                        MTLSize::new(*tg, 1, 1),
                    );
                }
                LaunchHint::RowsParallelSg { rows, nsg } => {
                    enc.dispatch_thread_groups(
                        MTLSize::new(*rows as u64, 1, 1),
                        MTLSize::new(32, *nsg, 1),
                    );
                }
                LaunchHint::MulMm {
                    tg_x,
                    tg_y,
                    tw,
                    nsg,
                    smem,
                } => {
                    enc.set_threadgroup_memory_length(0, *smem);
                    enc.dispatch_thread_groups(
                        MTLSize::new(*tg_x, *tg_y, 1),
                        MTLSize::new(*tw, *nsg, 1),
                    );
                }
                _ => {
                    enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                }
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            Ok(t0.elapsed().as_secs_f64() * 1e3)
        })
    }
}

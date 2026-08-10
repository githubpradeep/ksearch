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

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    /// Open CB + compute encoder (many dispatches, one encoder).
    pending: RefCell<Option<Pending>>,
    /// Committed CBs not yet waited on (retain until synchronize).
    inflight: RefCell<Vec<CommandBuffer>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pending: RefCell::new(None),
            inflight: RefCell::new(Vec::new()),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn compile(&self, kernel: &MetalKernelSource) -> Result<ComputePipelineState> {
        let options = CompileOptions::new();
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

    pub fn read_f32(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        self.synchronize().ok();
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
        if let Some(p) = self.pending.borrow_mut().take() {
            let cmd = Self::end_pending(p, false);
            self.inflight.borrow_mut().push(cmd);
        }
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

    /// Commit pending work and wait for all in-flight CBs (call before host reads).
    pub fn synchronize(&self) -> Result<()> {
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
        autoreleasepool(|| {
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
                LaunchHint::RowsParallel2D { rows, batch, tg } => {
                    // Encode as flattened 1D for the common path; 2D set below.
                    let _ = (rows, batch, tg);
                    (*rows as u64 * *batch as u64, *tg)
                }
                LaunchHint::MulMm { .. } => (1, 1), // unused; MulMm branch below
            };
            match &kernel.launch {
                LaunchHint::RowsParallel2D { rows, batch, tg } => {
                    enc.dispatch_thread_groups(
                        MTLSize::new(*rows as u64, *batch as u64, 1),
                        MTLSize::new(*tg, 1, 1),
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
        })
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

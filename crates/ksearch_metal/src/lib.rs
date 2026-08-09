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
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pending: RefCell::new(None),
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

    pub fn buffer_empty_bytes(&self, n: usize) -> Buffer {
        self.device
            .new_buffer(n as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_f32(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        self.synchronize().ok();
        let ptr = buf.contents() as *const f32;
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

    fn ensure_pending(&self) -> std::cell::RefMut<'_, Option<Pending>> {
        let mut slot = self.pending.borrow_mut();
        if slot.is_none() {
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            *slot = Some(Pending { cmd, enc });
        }
        slot
    }

    fn end_pending(pending: Pending, wait: bool) {
        pending.enc.end_encoding();
        pending.cmd.commit();
        if wait {
            pending.cmd.wait_until_completed();
        }
    }

    /// Commit pending work without waiting (overlap CPU encode with GPU).
    pub fn flush_async(&self) {
        if let Some(p) = self.pending.borrow_mut().take() {
            Self::end_pending(p, false);
        }
    }

    /// Commit pending work and wait (call before host reads).
    pub fn synchronize(&self) -> Result<()> {
        if let Some(p) = self.pending.borrow_mut().take() {
            Self::end_pending(p, true);
        }
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
        if inputs.len() != kernel.n_inputs {
            return Err(anyhow!(
                "expected {} inputs, got {}",
                kernel.n_inputs,
                inputs.len()
            ));
        }
        autoreleasepool(|| {
            let mut slot = self.ensure_pending();
            let pending = slot.as_mut().unwrap();
            let enc = &pending.enc;
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
            };
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
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
            };
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            Ok(t0.elapsed().as_secs_f64() * 1e3)
        })
    }
}

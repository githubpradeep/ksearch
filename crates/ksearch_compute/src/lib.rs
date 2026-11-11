//! Backend-neutral GPU execution for the model layer.
//!
//! `ksearch_kernels::Eng`, `ksearch_gemma` and the CLI are written against this
//! facade instead of against Metal directly, so the same model code runs on an
//! Apple GPU (Metal) or an NVIDIA GPU (CUDA — H100, A100, RTX 40/50-series).
//!
//! The KV cache, weights and activations are [`Buffer`]s; kernels are lowered
//! from the shared Kernel IR by [`Context::lower`], which picks the MSL or CUDA
//! emitter for the active backend. Both backends accept the same
//! [`KernelSource`], so `Eng`'s per-op code is backend-independent:
//!
//! ```ignore
//! let src = ctx.lower(&graph, out)?;             // MSL or CUDA C++
//! let pipe = ctx.compile(&src)?;                 // MTLComputePipelineState or cubin
//! ctx.encode_multi(&pipe, &src, &[a, x], &[y], 256)?;
//! ```
//!
//! Backend choice: `KSEARCH_BACKEND=metal|cuda` (or [`Context::new_backend`]).
//! With neither set, an Apple GPU prefers Metal and everything else prefers
//! CUDA, so the same binary is fast on a Mac and on an H100.

use anyhow::{anyhow, Result};
#[cfg(feature = "cuda")]
use ksearch_codegen::lower_to_cuda_chip;
#[cfg(feature = "metal")]
use ksearch_codegen::lower_to_metal_chip;
use ksearch_ir::{Graph, TensorId};
use std::cell::RefCell;
use std::time::Instant;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
compile_error!(
    "ksearch_compute needs at least one backend: enable the `metal` and/or `cuda` feature \
     (for an NVIDIA-only box: --no-default-features --features cuda)"
);

/// Which GPU API to drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cuda,
}

impl Backend {
    /// Parse the `KSEARCH_BACKEND` / `--backend` spelling.
    pub fn parse(s: &str) -> Option<Backend> {
        match s.to_ascii_lowercase().as_str() {
            "metal" | "mtl" => Some(Backend::Metal),
            "cuda" | "nvidia" => Some(Backend::Cuda),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
        }
    }
}

/// Device memory: a Metal buffer or a CUDA allocation.
///
/// Cloning shares the underlying storage on both backends, matching the Metal
/// semantics the model layer was written against.
#[derive(Debug)]
pub enum Buffer {
    #[cfg(feature = "metal")]
    Metal(metal::Buffer),
    #[cfg(feature = "cuda")]
    Cuda(ksearch_cuda::Buffer),
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "metal")]
            Buffer::Metal(b) => Buffer::Metal(b.clone()),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(b) => Buffer::Cuda(b.clone()),
        }
    }
}

/// Back-compat alias: the name the Metal code used for a buffer handle.
pub type BufferAlias = Buffer;

/// A compiled kernel: a Metal pipeline state or a loaded cubin entry point.
#[derive(Clone, Debug)]
pub enum Pipeline {
    #[cfg(feature = "metal")]
    Metal(metal::ComputePipelineState),
    #[cfg(feature = "cuda")]
    Cuda(ksearch_cuda::Kernel),
}

/// A GPU context driving one [`Backend`].
pub struct Context {
    backend: Backend,
    #[cfg(feature = "metal")]
    metal: Option<ksearch_metal::MetalContext>,
    #[cfg(feature = "cuda")]
    cuda: Option<RefCell<ksearch_cuda::CudaContext>>,
    /// Host-side phase timers (the CUDA path has no GPU counters).
    marks: RefCell<Vec<Instant>>,
    max_marks: RefCell<usize>,
}

// The CUDA context is `Send` and re-binds itself per call; the Metal context
// holds `RefCell`s and stays on its owning thread, exactly as before.
impl Context {
    /// Open a context on the preferred backend.
    ///
    /// `KSEARCH_BACKEND=metal|cuda` wins; otherwise an Apple host opens Metal
    /// and every other host opens CUDA, falling back to Metal.
    pub fn new() -> Result<Self> {
        if let Some(b) = std::env::var("KSEARCH_BACKEND")
            .ok()
            .and_then(|v| Backend::parse(&v))
        {
            return Self::new_backend(b);
        }
        let mut errs: Vec<String> = Vec::new();
        #[cfg(feature = "metal")]
        {
            if cfg!(target_vendor = "apple") {
                match Self::open_strict(Backend::Metal) {
                    Ok(ctx) => return Ok(ctx),
                    Err(e) => errs.push(format!("metal: {e}")),
                }
            }
        }
        #[cfg(feature = "cuda")]
        match Self::open_strict(Backend::Cuda) {
            Ok(ctx) => return Ok(ctx),
            Err(e) => errs.push(format!("cuda: {e}")),
        }
        #[cfg(feature = "metal")]
        {
            if !cfg!(target_vendor = "apple") {
                match Self::open_strict(Backend::Metal) {
                    Ok(ctx) => return Ok(ctx),
                    Err(e) => errs.push(format!("metal: {e}")),
                }
            }
        }
        let _ = &errs;
        Err(anyhow!(
            "no usable GPU backend (set KSEARCH_BACKEND=metal|cuda): {}",
            errs.join("; ")
        ))
    }

    /// Open exactly this backend (ignores `KSEARCH_BACKEND`).
    pub fn new_backend(backend: Backend) -> Result<Self> {
        Self::open_strict(backend)
    }

    /// Open exactly this backend, or fail with the backend's own error.
    pub fn open_strict(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Metal => {
                #[allow(clippy::needless_return)]
                #[cfg(feature = "metal")]
                {
                    let ctx = ksearch_metal::MetalContext::new()?;
                    return Ok(Self::from_metal(ctx));
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(anyhow!(
                        "this build has no Metal backend (feature `metal` off)"
                    ))
                }
            }
            Backend::Cuda => {
                #[allow(clippy::needless_return)]
                #[cfg(feature = "cuda")]
                {
                    let device = std::env::var("KSEARCH_CUDA_DEVICE")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    let tg = std::env::var("KSEARCH_TG")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(ksearch_cuda::DEFAULT_TG);
                    let ctx = ksearch_cuda::CudaContext::with_options(device, tg)?;
                    return Ok(Self::from_cuda(ctx));
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow!(
                        "this build has no CUDA backend (feature `cuda` off)"
                    ))
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    fn from_metal(ctx: ksearch_metal::MetalContext) -> Self {
        ACTIVE_BACKEND.store(BACKEND_METAL, std::sync::atomic::Ordering::Relaxed);
        Self {
            backend: Backend::Metal,
            metal: Some(ctx),
            #[cfg(feature = "cuda")]
            cuda: None,
            marks: RefCell::new(Vec::new()),
            max_marks: RefCell::new(0),
        }
    }

    #[cfg(feature = "cuda")]
    fn from_cuda(ctx: ksearch_cuda::CudaContext) -> Self {
        ACTIVE_BACKEND.store(BACKEND_CUDA, std::sync::atomic::Ordering::Relaxed);
        Self {
            backend: Backend::Cuda,
            #[cfg(feature = "metal")]
            metal: None,
            cuda: Some(RefCell::new(ctx)),
            marks: RefCell::new(Vec::new()),
            max_marks: RefCell::new(0),
        }
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// `true` when a context could be opened on this backend.
    pub fn available(backend: Backend) -> bool {
        Self::open_strict(backend).is_ok()
    }

    /// Human-readable device name; also the plan-cache "chip" key.
    pub fn device_name(&self) -> String {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().device_name(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => self.cuda_ctx().info().name.clone(),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_ctx(&self) -> &ksearch_metal::MetalContext {
        self.metal.as_ref().expect("metal context")
    }

    #[cfg(feature = "cuda")]
    fn cuda_ctx(&self) -> std::cell::Ref<'_, ksearch_cuda::CudaContext> {
        self.cuda.as_ref().expect("cuda context").borrow()
    }

    #[cfg(feature = "cuda")]
    fn cuda_ctx_mut(&self) -> std::cell::RefMut<'_, ksearch_cuda::CudaContext> {
        self.cuda.as_ref().expect("cuda context").borrow_mut()
    }

    // ---------------------------------------------------------------- lowering

    /// Lower a primitive Graph region for this backend.
    ///
    /// Metal emits MSL, CUDA emits CUDA C++; the returned [`KernelSource`] (with
    /// its [`LaunchHint`]) is what [`Context::compile`] consumes.
    pub fn lower(&self, graph: &Graph, out: TensorId) -> Result<KernelSource> {
        let chip = self.device_name();
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Ok(lower_to_metal_chip(graph, out, &chip)?),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(lower_to_cuda_chip(graph, out, &chip)?),
        }
    }

    /// Lower with an explicit chip key (bypasses [`Context::device_name`]).
    pub fn lower_chip(&self, graph: &Graph, out: TensorId, chip: &str) -> Result<KernelSource> {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Ok(lower_to_metal_chip(graph, out, chip)?),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(lower_to_cuda_chip(graph, out, chip)?),
        }
    }

    /// Compile a lowered kernel for this backend.
    pub fn compile(&self, kernel: &KernelSource) -> Result<Pipeline> {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Ok(Pipeline::Metal(self.metal_ctx().compile(kernel)?)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(Pipeline::Cuda(self.cuda_ctx_mut().load(kernel)?)),
        }
    }

    // ---------------------------------------------------------------- buffers

    pub fn buffer_f32(&self, data: &[f32]) -> Buffer {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Buffer::Metal(self.metal_ctx().buffer_f32(data)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let buf = self.cuda_ctx().alloc(data.len() * 4).expect("alloc f32");
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                self.cuda_ctx().copy_to(&buf, bytes).expect("upload f32");
                Buffer::Cuda(buf)
            }
        }
    }

    pub fn buffer_bytes(&self, data: &[u8]) -> Buffer {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Buffer::Metal(self.metal_ctx().buffer_bytes(data)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let buf = self
                    .cuda_ctx()
                    .alloc(data.len().max(1))
                    .expect("alloc bytes");
                self.cuda_ctx().copy_to(&buf, data).expect("upload bytes");
                Buffer::Cuda(buf)
            }
        }
    }

    pub fn buffer_empty_f32(&self, n: usize) -> Buffer {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Buffer::Metal(self.metal_ctx().buffer_empty_f32(n)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Buffer::Cuda(
                self.cuda_ctx()
                    .alloc(n * 4)
                    .expect("alloc empty f32 buffer"),
            ),
        }
    }

    pub fn buffer_empty_f16(&self, n: usize) -> Buffer {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Buffer::Metal(self.metal_ctx().buffer_empty_f16(n)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Buffer::Cuda(
                self.cuda_ctx()
                    .alloc(n * 2)
                    .expect("alloc empty f16 buffer"),
            ),
        }
    }

    pub fn buffer_empty_bytes(&self, n: usize) -> Buffer {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => Buffer::Metal(self.metal_ctx().buffer_empty_bytes(n)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Buffer::Cuda(self.cuda_ctx().alloc(n).expect("alloc empty buffer")),
        }
    }

    /// Size of a buffer in bytes.
    pub fn buffer_len(&self, buf: &Buffer) -> usize {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => b.length() as usize,
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => b.len(),
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    /// Zero a whole buffer.
    pub fn zero(&self, buf: &Buffer) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => {
                let n = b.length() as usize;
                unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, n) };
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                self.cuda_ctx().memset(b, 0).expect("memset");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    // ------------------------------------------------------------------- reads

    pub fn read_bytes(&self, buf: &Buffer, offset_bytes: usize, len_bytes: usize) -> Vec<u8> {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => {
                self.metal_ctx().read_bytes(b, offset_bytes, len_bytes)
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                self.synchronize().ok();
                let mut out = vec![0u8; len_bytes];
                self.cuda_ctx()
                    .copy_from_offset(b, offset_bytes, &mut out)
                    .expect("download bytes");
                out
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    pub fn read_f32(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().read_f32(b, n),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                self.synchronize().ok();
                let mut out = vec![0f32; n];
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
                self.cuda_ctx().copy_from(b, bytes).expect("download f32");
                out
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    /// Same as [`Context::read_f32`]; the CUDA path always synchronizes (reads
    /// are the only place the host needs the data, so the explicit "nosync"
    /// variant is a pure throughput hint).
    pub fn read_f32_nosync(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        self.read_f32(buf, n)
    }

    pub fn read_u16(&self, buf: &Buffer, n: usize) -> Vec<u16> {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().read_u16(b, n),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                self.synchronize().ok();
                let mut out = vec![0u16; n];
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 2) };
                self.cuda_ctx().copy_from(b, bytes).expect("download u16");
                out
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    // ------------------------------------------------------------------ writes

    pub fn write_buffer(&self, buf: &Buffer, data: &[f32]) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().write_buffer(b, data),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                self.cuda_ctx().copy_to(b, bytes).expect("upload f32");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    pub fn write_buffer_nosync(&self, buf: &Buffer, data: &[f32]) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().write_buffer_nosync(b, data),
            #[cfg(feature = "cuda")]
            _ => self.write_buffer(buf, data),
        }
    }

    pub fn write_u16s(&self, buf: &Buffer, data: &[u16]) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().write_u16s(b, data),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
                };
                self.cuda_ctx().copy_to(b, bytes).expect("upload u16");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    pub fn write_u16s_nosync(&self, buf: &Buffer, data: &[u16]) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => self.metal_ctx().write_u16s_nosync(b, data),
            #[cfg(feature = "cuda")]
            _ => self.write_u16s(buf, data),
        }
    }

    /// Write `vals` at element offset `elem_off` (used for the per-layer
    /// attention metadata, which the model updates between launches).
    pub fn write_f32_at(&self, buf: &Buffer, elem_off: usize, vals: &[f32]) {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(b)) => unsafe {
                let p = (b.contents() as *mut f32).add(elem_off);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v;
                }
            },
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
                };
                self.cuda_ctx()
                    .copy_to_at(b, elem_off * 4, bytes)
                    .expect("upload f32 at offset");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffer of a different backend"),
        }
    }

    // ------------------------------------------------------------------- sync

    pub fn has_gpu_work(&self) -> bool {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().has_gpu_work(),
            // CUDA launches run on the default stream; the host only observes
            // them through an explicit synchronize.
            #[cfg(feature = "cuda")]
            Backend::Cuda => false,
        }
    }

    pub fn flush_async(&self) {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().flush_async(),
            // Nothing to commit: a CUDA launch is already queued.
            #[cfg(feature = "cuda")]
            Backend::Cuda => {}
        }
    }

    pub fn wait_inflight_at_most(&self, max_inflight: usize) {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().wait_inflight_at_most(max_inflight),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                if max_inflight == 0 {
                    self.synchronize().ok();
                }
            }
        }
    }

    pub fn encoder_barrier(&self) {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().encoder_barrier(),
            // Launches are already ordered within the stream.
            #[cfg(feature = "cuda")]
            Backend::Cuda => {}
        }
    }

    /// Device-to-device copy (Metal blit / `cuMemcpyDtoD`).
    pub fn blit_copy_bytes(
        &self,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        nbytes: u64,
    ) {
        match (self.backend, src, dst) {
            #[cfg(feature = "metal")]
            (Backend::Metal, Buffer::Metal(s), Buffer::Metal(d)) => self
                .metal_ctx()
                .blit_copy_bytes(s, src_off, d, dst_off, nbytes),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(s), Buffer::Cuda(d)) => {
                self.cuda_ctx()
                    .copy_device_to_device(
                        d,
                        dst_off as usize,
                        s,
                        src_off as usize,
                        nbytes as usize,
                    )
                    .expect("device to device copy");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("buffers of a different backend"),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().synchronize(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(self.cuda_ctx().synchronize()?),
        }
    }

    // -------------------------------------------------------------- dispatch

    /// Encode one dispatch with a single output (no host wait).
    pub fn encode(
        &self,
        pipeline: &Pipeline,
        kernel: &KernelSource,
        inputs: &[&Buffer],
        output: &Buffer,
        tg_size: u64,
    ) -> Result<()> {
        self.encode_multi(pipeline, kernel, inputs, &[output], tg_size)
    }

    /// Encode one dispatch with `kernel.n_outputs` outputs (no host wait).
    pub fn encode_multi(
        &self,
        pipeline: &Pipeline,
        kernel: &KernelSource,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        tg_size: u64,
    ) -> Result<()> {
        let in_offs = vec![0u64; inputs.len()];
        let out_offs = vec![0u64; outputs.len()];
        self.encode_offsets_multi(
            pipeline, kernel, inputs, &in_offs, outputs, &out_offs, tg_size,
        )
    }

    /// Encode one dispatch with per-buffer byte offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_offsets(
        &self,
        pipeline: &Pipeline,
        kernel: &KernelSource,
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

    /// Encode one dispatch with per-buffer byte offsets and multiple outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_offsets_multi(
        &self,
        pipeline: &Pipeline,
        kernel: &KernelSource,
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
            return Err(anyhow!("expected {} outputs, got {}", n_out, outputs.len()));
        }
        if output_byte_offsets.len() != outputs.len() {
            return Err(anyhow!(
                "expected {} output offsets, got {}",
                outputs.len(),
                output_byte_offsets.len()
            ));
        }

        match (self.backend, pipeline) {
            #[cfg(feature = "metal")]
            (Backend::Metal, Pipeline::Metal(p)) => {
                let ins = Self::metal_bufs(inputs)?;
                let outs = Self::metal_bufs(outputs)?;
                self.metal_ctx().encode_offsets_multi(
                    p,
                    kernel,
                    &ins,
                    input_byte_offsets,
                    &outs,
                    output_byte_offsets,
                    tg_size,
                )
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Pipeline::Cuda(f)) => {
                let mut args = ksearch_cuda::KernelArgs::new();
                for (i, b) in inputs.iter().enumerate() {
                    args.ptr(self.device_ptr(b)? + input_byte_offsets[i]);
                }
                for (o, b) in outputs.iter().enumerate() {
                    args.ptr(self.device_ptr(b)? + output_byte_offsets[o]);
                }
                let cfg = ksearch_cuda::launch_config(&kernel.launch, tg_size);
                let ctx = self.cuda_ctx();
                ctx.launch_with(*f, cfg, &mut args)?;
                CUDA_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => Err(anyhow!("pipeline of a different backend")),
        }
    }

    /// Run once and wait; returns wall milliseconds.
    pub fn run(
        &self,
        pipeline: &Pipeline,
        kernel: &KernelSource,
        inputs: &[&Buffer],
        output: &Buffer,
        tg_size: u64,
    ) -> Result<f64> {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => {
                let Pipeline::Metal(p) = pipeline else {
                    return Err(anyhow!("pipeline of a different backend"));
                };
                let ins = Self::metal_bufs(inputs)?;
                let out = Self::metal_buf(output)?;
                self.metal_ctx().run(p, kernel, &ins, out, tg_size)
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                self.synchronize()?;
                let t0 = Instant::now();
                self.encode(pipeline, kernel, inputs, output, tg_size)?;
                let enqueue_ns = t0.elapsed().as_nanos() as u64;
                CUDA_ENQUEUE_NANOS.fetch_add(enqueue_ns, std::sync::atomic::Ordering::Relaxed);
                self.synchronize()?;
                Ok((t0.elapsed().as_secs_f64() - enqueue_ns as f64 / 1e9) * 1e3)
            }
        }
    }

    /// Borrow the Metal handles behind backend-neutral buffers.
    #[cfg(feature = "metal")]
    fn metal_buf(buf: &Buffer) -> Result<&metal::Buffer> {
        match buf {
            Buffer::Metal(b) => Ok(b),
            #[allow(unreachable_patterns)]
            _ => Err(anyhow!("buffer of a different backend")),
        }
    }

    /// Borrow a slice of Metal handles behind backend-neutral buffers.
    #[cfg(feature = "metal")]
    fn metal_bufs<'a>(bufs: &[&'a Buffer]) -> Result<Vec<&'a metal::Buffer>> {
        bufs.iter().map(|b| Self::metal_buf(b)).collect()
    }

    /// Raw device pointer of a buffer of this context.
    #[cfg(feature = "cuda")]
    fn device_ptr(&self, buf: &Buffer) -> Result<ksearch_cuda::CUdeviceptr> {
        match (self.backend, buf) {
            #[cfg(not(feature = "metal"))]
            (Backend::Metal, _) => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            (Backend::Cuda, _) => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            (Backend::Metal, _) => Err(anyhow!("device pointers are a CUDA-only detail")),
            #[cfg(feature = "cuda")]
            (Backend::Cuda, Buffer::Cuda(b)) => Ok(b.as_ptr()),
            #[allow(unreachable_patterns)]
            _ => Err(anyhow!("buffer of a different backend")),
        }
    }

    // ------------------------------------------------------------ telemetry

    /// `(dispatches, enqueue_seconds)` since the last call.
    pub fn take_telemetry(&self) -> (u64, f64) {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => ksearch_metal::take_telemetry(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use std::sync::atomic::Ordering;
                let n = CUDA_DISPATCHES.swap(0, Ordering::Relaxed);
                let ns = CUDA_ENQUEUE_NANOS.swap(0, Ordering::Relaxed);
                (n, ns as f64 / 1e9)
            }
        }
    }

    /// Begin phase timing. Metal uses its GPU counters; CUDA records host
    /// instants, which is the honest thing to do without GPU timestamps.
    pub fn init_phase_counters(&self, max_samples: u64) -> Result<()> {
        *self.max_marks.borrow_mut() = max_samples as usize;
        self.marks.borrow_mut().clear();
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().init_phase_counters(max_samples),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok(()),
        }
    }

    pub fn phase_mark(&self) {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().phase_mark(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let mut marks = self.marks.borrow_mut();
                let max = *self.max_marks.borrow();
                if max == 0 || marks.len() < max + 1 {
                    marks.push(Instant::now());
                }
            }
        }
    }

    pub fn resolve_phase_times_ms(&self) -> Vec<f64> {
        match self.backend {
            #[cfg(not(feature = "metal"))]
            Backend::Metal => unreachable!("this build has no `metal` feature"),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("this build has no `cuda` feature"),
            #[cfg(feature = "metal")]
            Backend::Metal => self.metal_ctx().resolve_phase_times_ms(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let marks = self.marks.borrow();
                marks
                    .windows(2)
                    .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1e3)
                    .collect()
            }
        }
    }
}

/// Dispatches since the last [`Context::take_telemetry`] (CUDA only).
#[cfg(feature = "cuda")]
static CUDA_DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Launch overhead in nanoseconds (CUDA only).
#[cfg(feature = "cuda")]
static CUDA_ENQUEUE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const BACKEND_NONE: u8 = 0;
#[allow(dead_code)]
const BACKEND_METAL: u8 = 1;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
const BACKEND_CUDA: u8 = 2;

/// Last backend a [`Context`] was opened on, so the free functions
/// ([`take_telemetry`]) can follow the model layer.
static ACTIVE_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(BACKEND_NONE);

/// `(dispatches, enqueue_seconds)` for the process's active backend.
pub fn take_telemetry() -> (u64, f64) {
    use std::sync::atomic::Ordering;
    let active = ACTIVE_BACKEND.load(Ordering::Relaxed);
    #[cfg(feature = "metal")]
    if active == BACKEND_METAL {
        return ksearch_metal::take_telemetry();
    }
    #[cfg(feature = "cuda")]
    if active == BACKEND_CUDA {
        let n = CUDA_DISPATCHES.swap(0, Ordering::Relaxed);
        let ns = CUDA_ENQUEUE_NANOS.swap(0, Ordering::Relaxed);
        return (n, ns as f64 / 1e9);
    }
    (0, 0.0)
}

/// `true` when the process has a usable CUDA device (driver + device).
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        ksearch_cuda::CudaContext::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// `true` when the process has a usable Metal device.
pub fn metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        ksearch_metal::MetalContext::new().is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

pub use ksearch_codegen::{KernelSource, LaunchHint};

#[cfg(test)]
mod tests {
    use super::*;
    use ksearch_ir::{DType, Graph, Shape};

    #[test]
    fn backend_names_round_trip() {
        assert_eq!(Backend::parse("cuda"), Some(Backend::Cuda));
        assert_eq!(Backend::parse("CUDA"), Some(Backend::Cuda));
        assert_eq!(Backend::parse("metaL"), Some(Backend::Metal));
        assert_eq!(Backend::parse("tpu"), None);
        assert_eq!(Backend::Cuda.as_str(), "cuda");
    }

    #[test]
    fn lowering_targets_follow_the_backend() {
        // The same Graph region must lower to MSL for Metal and to CUDA C++ for
        // CUDA, which is what makes `Eng` backend-independent.
        let mut g = Graph::new();
        let x = g.input(Shape(vec![256]), DType::F16);
        let y = g.input(Shape(vec![256]), DType::F16);
        let out = g.add(x, y).unwrap();

        let chip = "ksearch_compute_test";
        let msl = ksearch_codegen::lower_to_metal_chip(&g, out, chip).unwrap();
        let cuda = ksearch_codegen::lower_to_cuda_chip(&g, out, chip).unwrap();
        assert!(
            msl.source.contains("kernel void"),
            "expected an MSL entry point"
        );
        assert!(
            cuda.source.contains("__global__") && cuda.source.contains("blockIdx"),
            "expected a CUDA entry point"
        );
        assert_eq!(cuda.n_inputs, 2);
        assert_eq!(cuda.n_outputs, 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_reports_driver_state_gracefully() {
        // Without an NVIDIA driver this must be a readable error; on a CUDA
        // machine it must open a context. Either way: no panic.
        match Context::new_backend(Backend::Cuda) {
            Ok(ctx) => {
                assert_eq!(ctx.backend(), Backend::Cuda);
                assert!(!ctx.device_name().is_empty());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("CUDA") || msg.contains("device"),
                    "unexpected failure: {msg}"
                );
            }
        }
    }
}

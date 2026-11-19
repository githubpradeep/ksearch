//! CUDA backend runtime for NVIDIA GPUs: H100 (`sm_90`), A100, RTX 40/50-series
//! (`sm_120`), and anything else the installed toolkit supports.
//!
//! The driver is `dlopen`ed at run time and kernels are compiled to cubins with
//! `nvcc --cubin`, so this crate still builds and links on machines that have no
//! CUDA install (macOS CI, CPU-only Linux): opening a context reports
//! [`CudaError::DriverUnavailable`] instead of failing to link.
//!
//! Kernels come from [`ksearch_codegen`], which emits CUDA C++ from the shared
//! Kernel IR, so the same graph region that lowers to MSL on a Mac lowers to
//! CUDA here:
//!
//! ```no_run
//! # use ksearch_cuda::*;
//! # fn run(src: ksearch_codegen::KernelSource, a: &[u8], b: &[u8]) -> Result<(), CudaError> {
//! let mut ctx = CudaContext::with_options(0, DEFAULT_TG)?;
//! let k = ctx.load(&src)?;                      // nvcc --cubin, then cuModuleLoadData
//! let ba = ctx.alloc(a.len())?;
//! let bb = ctx.alloc(b.len())?;
//! let bout = ctx.alloc(256)?;
//! ctx.copy_to(&ba, a)?;
//! ctx.copy_to(&bb, b)?;
//! let mut args = KernelArgs::new();
//! args.ptr(ba.as_ptr()).ptr(bb.as_ptr()).ptr(bout.as_ptr());
//! ctx.launch(k, &src.launch, &mut args)?;       // grid/block from the LaunchHint
//! ctx.synchronize()?;
//! # Ok(()) }
//! ```
//!
//! Backends are chosen a level up, in `ksearch_compute`: `KSEARCH_BACKEND=cuda`,
//! `KSEARCH_CUDA_DEVICE=<n>`, `KSEARCH_TG=<threads/block>`, and `KSEARCH_NVCC`
//! when `nvcc` is not on `PATH`.

pub mod ffi;
pub mod launch;

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use ffi::{CUdeviceptr, CudaError, DRIVER_LIB_NAMES};
pub use launch::{launch_config, LaunchConfig};
/// Default threads per block for flat (`Elementwise` / `Rows`) kernels.
pub const DEFAULT_TG: u64 = 256;
/// Above this, a kernel must opt in to its dynamic shared memory size.
const DEFAULT_MAX_SMEM: u32 = 48 * 1024;
/// At or above this dynamic shared size, ask for a max shared-memory carveout
/// (hint) so more than one block fits per SM.
const SMEM_CARVEOUT_THRESHOLD: u32 = 24 * 1024;
/// Environment override for the `nvcc` binary.
const NVCC_ENV: &str = "KSEARCH_NVCC";

/// What the driver reports about the device.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    pub name: String,
    /// Compute capability, e.g. `9.0` on an H100, `12.0` on an RTX 5090.
    pub major: i32,
    pub minor: i32,
    pub max_threads_per_block: i32,
    pub max_shared_per_block: i32,
    pub multiprocessor_count: i32,
    pub total_mem_bytes: u64,
}

impl DeviceInfo {
    /// `nvcc -arch` value for this device, e.g. `sm_90` or `sm_120`.
    pub fn sm_arch(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }
}

/// Driver, context and device facts shared by the context and every buffer.
struct Shared {
    drv: ffi::Driver,
    ctx: ffi::CUcontext,
    info: DeviceInfo,
    nvcc: Option<PathBuf>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                (self.drv.cuCtxDestroy)(self.ctx);
            }
            self.ctx = std::ptr::null_mut();
        }
    }
}

// Driver calls re-bind the context with `cuCtxSetCurrent` before every launch and
// copy, and the context outlives every `Arc<Shared>` handle.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A CUDA context on one device.
///
/// The context is `Send`: launches and copies re-bind it on the calling thread.
pub struct CudaContext {
    shared: Arc<Shared>,
    /// cubins loaded into this context, keyed by a hash of the source.
    modules: HashMap<u64, ffi::CUmodule>,
    /// Fill / increment / stash kernels for GPU-resident decode state.
    runtime: Option<RuntimeKernels>,
    /// Non-default stream used to capture and replay a decode token.
    stream: ffi::CUstream,
    capturing: bool,
    /// Even and odd token graphs. `None` until capture succeeds.
    decode_exec: [Option<ffi::CUgraphExec>; 2],
    /// Threads per block used by [`launch_config`] for flat kernels.
    tg: u64,
}

impl CudaContext {
    /// `true` when this process can open a CUDA device (driver + a GPU).
    ///
    /// Used by `ksearch_compute::cuda_available()` and the CLI's backend picker;
    /// never panics, and is cheap enough to call at start-up.
    pub fn is_available() -> bool {
        let Ok(drv) = ffi::Driver::load() else {
            return false;
        };
        unsafe {
            if drv.check("cuInit", (drv.cuInit)(0)).is_err() {
                return false;
            }
            let mut count: i32 = 0;
            drv.check("cuDeviceGetCount", (drv.cuDeviceGetCount)(&mut count))
                .is_ok()
                && count > 0
        }
    }

    /// Open device 0 with [`DEFAULT_TG`].
    pub fn new() -> Result<Self, CudaError> {
        Self::with_options(0, DEFAULT_TG)
    }

    /// Open `device` with the default block size.
    pub fn with_device(device: usize) -> Result<Self, CudaError> {
        Self::with_options(device, DEFAULT_TG)
    }

    /// Open `device` and remember `tg` as the default threads-per-block.
    ///
    /// `device` is indexed in the driver's own order (0 = first GPU, typically
    /// the one CUDA_VISIBLE_DEVICES exposes).
    pub fn with_options(device: usize, tg: u64) -> Result<Self, CudaError> {
        let drv = ffi::Driver::load()?;
        let (ctx, info) = unsafe {
            drv.check("cuInit", (drv.cuInit)(0))?;

            let mut count: i32 = 0;
            drv.check("cuDeviceGetCount", (drv.cuDeviceGetCount)(&mut count))?;
            if count <= 0 {
                return Err(CudaError::NoDevice);
            }
            if device >= count as usize {
                return Err(CudaError::BadDevice(device, count));
            }

            let mut dev: ffi::CUdevice = 0;
            drv.check("cuDeviceGet", (drv.cuDeviceGet)(&mut dev, device as i32))?;

            let mut info = DeviceInfo::default();
            let mut buf = [0i8; 256];
            drv.check(
                "cuDeviceGetName",
                (drv.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, dev),
            )?;
            info.name = std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let attr = |what: i32| -> Result<i32, CudaError> {
                let mut v: i32 = 0;
                drv.check(
                    "cuDeviceGetAttribute",
                    (drv.cuDeviceGetAttribute)(&mut v, what, dev),
                )?;
                Ok(v)
            };
            info.major = attr(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            info.minor = attr(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            info.max_threads_per_block = attr(ffi::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
            info.max_shared_per_block = attr(ffi::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
            info.multiprocessor_count = attr(ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;

            let mut mem: usize = 0;
            if drv
                .check("cuDeviceTotalMem", (drv.cuDeviceTotalMem)(&mut mem, dev))
                .is_ok()
            {
                info.total_mem_bytes = mem as u64;
            }

            let mut ctx: ffi::CUcontext = std::ptr::null_mut();
            drv.check("cuCtxCreate", (drv.cuCtxCreate)(&mut ctx, 0, dev))?;
            (ctx, info)
        };

        Ok(Self {
            shared: Arc::new(Shared {
                drv,
                ctx,
                info,
                nvcc: find_nvcc(),
            }),
            modules: HashMap::new(),
            runtime: None,
            stream: std::ptr::null_mut(),
            capturing: false,
            decode_exec: [None, None],
            tg: tg.max(1),
        })
    }

    /// Device facts (name, compute capability, limits).
    pub fn info(&self) -> &DeviceInfo {
        &self.shared.info
    }

    /// Device name — also the plan-cache "chip" key.
    pub fn device_name(&self) -> &str {
        &self.shared.info.name
    }

    /// `nvcc -arch` value for this device, e.g. `sm_90` / `sm_120`.
    pub fn arch(&self) -> String {
        self.shared.info.sm_arch()
    }

    /// Default threads per block for flat kernels.
    pub fn tg_size(&self) -> u64 {
        self.tg
    }

    /// The `nvcc` used to build kernels, when one was found.
    pub fn nvcc(&self) -> Option<&Path> {
        self.shared.nvcc.as_deref()
    }

    // ------------------------------------------------------------------ memory

    /// Allocate `bytes` of device memory.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer, CudaError> {
        let bytes = bytes.max(1);
        let mut dptr: ffi::CUdeviceptr = 0;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check("cuMemAlloc", (drv.cuMemAlloc)(&mut dptr, bytes))?;
        }
        Ok(Buffer {
            alloc: Arc::new(Alloc {
                shared: Arc::clone(&self.shared),
                dptr,
                bytes,
            }),
        })
    }

    /// Host → device copy into the start of `buf`.
    pub fn copy_to(&self, buf: &Buffer, data: &[u8]) -> Result<(), CudaError> {
        self.copy_to_at(buf, 0, data)
    }

    /// Host → device copy at a byte offset.
    pub fn copy_to_at(&self, buf: &Buffer, off: usize, data: &[u8]) -> Result<(), CudaError> {
        buf.check_range(off, data.len(), "copy_to")?;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check(
                "cuMemcpyHtoD",
                (drv.cuMemcpyHtoD)(
                    buf.dptr() + off as ffi::CUdeviceptr,
                    data.as_ptr() as *const _,
                    data.len(),
                ),
            )
        }
    }

    /// Device → host copy from the start of `buf`.
    pub fn copy_from(&self, buf: &Buffer, out: &mut [u8]) -> Result<(), CudaError> {
        self.copy_from_offset(buf, 0, out)
    }

    /// Device → host copy from a byte offset.
    pub fn copy_from_offset(
        &self,
        buf: &Buffer,
        off: usize,
        out: &mut [u8],
    ) -> Result<(), CudaError> {
        buf.check_range(off, out.len(), "copy_from")?;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check(
                "cuMemcpyDtoH",
                (drv.cuMemcpyDtoH)(
                    out.as_mut_ptr() as *mut _,
                    buf.dptr() + off as ffi::CUdeviceptr,
                    out.len(),
                ),
            )
        }
    }

    /// Device → device copy (the Metal `blit_copy_bytes` equivalent).
    pub fn copy_device_to_device(
        &self,
        dst: &Buffer,
        dst_off: usize,
        src: &Buffer,
        src_off: usize,
        nbytes: usize,
    ) -> Result<(), CudaError> {
        src.check_range(src_off, nbytes, "blit src")?;
        dst.check_range(dst_off, nbytes, "blit dst")?;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check(
                "cuMemcpyDtoD",
                (drv.cuMemcpyDtoD)(
                    dst.dptr() + dst_off as ffi::CUdeviceptr,
                    src.dptr() + src_off as ffi::CUdeviceptr,
                    nbytes,
                ),
            )
        }
    }

    /// Fill `buf` with a byte value (used to zero the KV cache / activations).
    pub fn memset(&self, buf: &Buffer, value: i32) -> Result<(), CudaError> {
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check(
                "cuMemsetD8",
                (drv.cuMemsetD8)(buf.dptr(), value as u8, buf.len()),
            )
        }
    }

    // ----------------------------------------------------------------- kernels

    /// Compile + load a generated kernel.
    ///
    /// The cubin is cached per context, so lowering the same source twice costs
    /// one `cuModuleGetFunction`. Requires `nvcc` (see [`CudaContext::nvcc`]).
    pub fn load(&mut self, src: &ksearch_codegen::KernelSource) -> Result<Kernel, CudaError> {
        self.load_source(&src.name, &src.source)
    }

    /// Compile + load raw CUDA C++ (the entry point is `name`).
    pub fn load_source(&mut self, name: &str, source: &str) -> Result<Kernel, CudaError> {
        let key = source_key(source);
        let module = match self.modules.get(&key).copied() {
            Some(m) => m,
            None => {
                let cubin = self.compile(source, name)?;
                let mut module: ffi::CUmodule = std::ptr::null_mut();
                let mut image = cubin;
                // The driver expects a NUL-terminated image.
                image.push(0);
                unsafe {
                    let drv = &self.shared.drv;
                    drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
                    drv.check(
                        "cuModuleLoadData",
                        (drv.cuModuleLoadData)(&mut module, image.as_ptr() as *const _),
                    )?;
                }
                self.modules.insert(key, module);
                module
            }
        };

        let cname = CString::new(name)
            .map_err(|_| CudaError::Other(format!("invalid kernel name `{name}`")))?;
        let mut func: ffi::CUfunction = std::ptr::null_mut();
        unsafe {
            let drv = &self.shared.drv;
            drv.check(
                "cuModuleGetFunction",
                (drv.cuModuleGetFunction)(&mut func, module, cname.as_ptr()),
            )?;
        }
        Ok(Kernel { func })
    }

    fn runtime_kernels(&mut self) -> Result<RuntimeKernels, CudaError> {
        if let Some(k) = self.runtime {
            return Ok(k);
        }
        let src = RUNTIME_SRC;
        let k = RuntimeKernels {
            fill: self.load_source("ks_fill_attn_meta", src)?,
            inc: self.load_source("ks_inc_u32", src)?,
            scatter: self.load_source("ks_scatter_meta", src)?,
            stash: self.load_source("ks_stash_f32", src)?,
            stash_inc: self.load_source("ks_stash_f32_inc", src)?,
            deq_q4k: self.load_source("ks_dequant_q4k_f16", DEQUANT_SRC)?,
            deq_q6k: self.load_source("ks_dequant_q6k_f16", DEQUANT_SRC)?,
            mm_reduce: self.load_source("ks_f16_mm_reduce", DEQUANT_SRC)?,
        };
        self.runtime = Some(k);
        Ok(k)
    }

    /// Write every layer's attention metadata from the device position.
    ///
    /// `table` holds one device pointer per layer (the 32-byte meta slot).
    /// `pos` is a single `f32` in device memory. `swa_mask` bit `i` marks a
    /// sliding-window layer. One launch, no host copy.
    pub fn fill_attn_meta(
        &mut self,
        table: &Buffer,
        pos: &Buffer,
        swa_mask: u64,
        n_layers: u32,
        window: u32,
    ) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(table.as_ptr())
            .ptr(pos.as_ptr())
            .u64(swa_mask)
            .u32(n_layers)
            .u32(window);
        let block = 64u32;
        let grid = (n_layers.max(1) + block - 1) / block;
        self.launch_with(k.fill, LaunchConfig::new((grid.max(1), 1), (block, 1), 0), &mut args)
    }

    /// `*pos += 1` on the device, ordered with the surrounding launches.
    pub fn inc_u32(&mut self, pos: &Buffer) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(pos.as_ptr());
        self.launch_with(k.inc, LaunchConfig::new((1, 1), (1, 1), 0), &mut args)
    }

    /// Scatter `n_layers` records of 4 floats from `src` into the per-layer slots.
    pub fn scatter_meta(
        &mut self,
        table: &Buffer,
        src: &Buffer,
        n_layers: u32,
    ) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(table.as_ptr()).ptr(src.as_ptr()).u32(n_layers);
        let block = 64u32;
        let grid = (n_layers.max(1) + block - 1) / block;
        self.launch_with(
            k.scatter,
            LaunchConfig::new((grid.max(1), 1), (block, 1), 0),
            &mut args,
        )
    }

    /// `dst[index] = src[0]`, queued on the default stream.
    pub fn stash_f32(
        &mut self,
        src: &Buffer,
        dst: &Buffer,
        index: u32,
    ) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(src.as_ptr()).ptr(dst.as_ptr()).u32(index);
        self.launch_with(k.stash, LaunchConfig::new((1, 1), (1, 1), 0), &mut args)
    }

    /// `dst[*index] = src[0]; *index += 1`, so a replayed graph advances the slot.
    pub fn stash_f32_inc(
        &mut self,
        src: &Buffer,
        dst: &Buffer,
        index: &Buffer,
    ) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let mut args = KernelArgs::new();
        args.ptr(src.as_ptr()).ptr(dst.as_ptr()).ptr(index.as_ptr());
        self.launch_with(
            k.stash_inc,
            LaunchConfig::new((1, 1), (1, 1), 0),
            &mut args,
        )
    }

    /// Dequantize `n` packed weights from `src` into F16 `dst`, once.
    ///
    /// `q6` selects the Q6_K layout; otherwise `src` is Q4_K. `n` must be a
    /// multiple of 256 (the super-block size). Queued on the default stream
    /// so a following GEMM launch is ordered after it.
    pub fn dequant_f16(
        &mut self,
        src: &Buffer,
        dst: &Buffer,
        n: usize,
        q6: bool,
    ) -> Result<(), CudaError> {
        if !n.is_multiple_of(256) {
            return Err(CudaError::Other(format!(
                "dequant_f16: {n} elements is not a multiple of the 256-element block"
            )));
        }
        let k = self.runtime_kernels()?;
        let block = 256u32;
        let (kernel, units) = if q6 {
            (k.deq_q6k, n as u64)
        } else {
            (k.deq_q4k, (n / 16) as u64)
        };
        let grid = (units.div_ceil(block as u64)).min(u32::MAX as u64) as u32;
        let mut args = KernelArgs::new();
        args.ptr(src.as_ptr()).ptr(dst.as_ptr()).u64(units);
        self.launch_with(
            kernel,
            LaunchConfig::new((grid.max(1), 1), (block, 1), 0),
            &mut args,
        )
    }

    /// Sum `splits` F32 partial planes of `nm` elements into the F16 `out`.
    ///
    /// The tail of the split-K prefill GEMM: `parts[s * nm + i]`, summed
    /// over `s`. Queued on the default stream, ordered after the GEMM.
    pub fn f16_mm_reduce(
        &mut self,
        parts: &Buffer,
        out: &Buffer,
        nm: usize,
        splits: usize,
    ) -> Result<(), CudaError> {
        let k = self.runtime_kernels()?;
        let block = 256u32;
        let grid = ((nm as u64).div_ceil(block as u64)).min(u32::MAX as u64) as u32;
        let mut args = KernelArgs::new();
        args.ptr(parts.as_ptr()).ptr(out.as_ptr()).u64(nm as u64).u32(splits as u32);
        self.launch_with(
            k.mm_reduce,
            LaunchConfig::new((grid.max(1), 1), (block, 1), 0),
            &mut args,
        )
    }

    // ---------------------------------------------------------------- dispatch

    /// Launch with the grid/block the [`ksearch_codegen::LaunchHint`] asks for.
    pub fn launch(
        &self,
        k: Kernel,
        hint: &ksearch_codegen::LaunchHint,
        args: &mut KernelArgs,
    ) -> Result<(), CudaError> {
        self.launch_with(k, launch_config(hint, self.tg), args)
    }

    /// Launch with an explicit configuration (e.g. a re-tiled grid from the tuner).
    pub fn launch_with(
        &self,
        k: Kernel,
        cfg: LaunchConfig,
        args: &mut KernelArgs,
    ) -> Result<(), CudaError> {
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
        }
        if cfg.smem > DEFAULT_MAX_SMEM {
            // Above 48 KiB a kernel must opt in to its dynamic shared memory.
            unsafe {
                drv.check(
                    "cuFuncSetAttribute",
                    (drv.cuFuncSetAttribute)(
                        k.func,
                        ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        cfg.smem as i32,
                    ),
                )?;
            }
        } else if cfg.smem >= SMEM_CARVEOUT_THRESHOLD {
            // Large-but-fitting tiles (the F16 GEMM's 30 KB) ask for a shared
            // memory carveout so two blocks are resident per SM; the driver
            // treats this as a hint.
            unsafe {
                let _ = drv.check(
                    "cuFuncSetAttribute",
                    (drv.cuFuncSetAttribute)(
                        k.func,
                        ffi::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                        100,
                    ),
                );
            }
        }
        let mut ptrs = args.pointers();
        unsafe {
            drv.check(
                "cuLaunchKernel",
                (drv.cuLaunchKernel)(
                    k.func,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.smem,
                    if self.capturing {
                        self.stream
                    } else {
                        std::ptr::null_mut()
                    },
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }

    fn ensure_stream(&mut self) -> Result<(), CudaError> {
        if !self.stream.is_null() {
            return Ok(());
        }
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            // CU_STREAM_NON_BLOCKING
            drv.check(
                "cuStreamCreate",
                (drv.cuStreamCreate)(&mut self.stream, 1),
            )?;
        }
        Ok(())
    }

    /// `true` after both parity graphs have been instantiated.
    pub fn decode_graphs_ready(&self) -> bool {
        self.decode_exec[0].is_some() && self.decode_exec[1].is_some()
    }

    /// Record subsequent launches on the decode stream. Does not execute them.
    pub fn begin_decode_capture(&mut self) -> Result<(), CudaError> {
        if self.capturing {
            return Err(CudaError::Other("decode capture already active".into()));
        }
        self.ensure_stream()?;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check(
                "cuStreamBeginCapture",
                (drv.cuStreamBeginCapture)(self.stream, 0),
            )?;
        }
        self.capturing = true;
        Ok(())
    }

    /// Finish the capture and keep the executable graph for `parity` (0 or 1).
    pub fn end_decode_capture(&mut self, parity: usize) -> Result<(), CudaError> {
        if !self.capturing {
            return Err(CudaError::Other("decode capture is not active".into()));
        }
        self.capturing = false;
        let drv = &self.shared.drv;
        let mut graph: ffi::CUgraph = std::ptr::null_mut();
        unsafe {
            drv.check(
                "cuStreamEndCapture",
                (drv.cuStreamEndCapture)(self.stream, &mut graph),
            )?;
        }
        let mut exec: ffi::CUgraphExec = std::ptr::null_mut();
        let instantiated = unsafe {
            drv.check(
                "cuGraphInstantiateWithFlags",
                (drv.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
            )
        };
        unsafe {
            (drv.cuGraphDestroy)(graph);
        }
        instantiated?;
        if let Some(old) = self.decode_exec[parity].replace(exec) {
            unsafe { (drv.cuGraphExecDestroy)(old) };
        }
        Ok(())
    }

    /// Leave capture mode after a failed token recording.
    pub fn abort_decode_capture(&mut self) {
        if !self.capturing {
            return;
        }
        self.capturing = false;
        let drv = &self.shared.drv;
        let mut graph: ffi::CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = (drv.cuStreamEndCapture)(self.stream, &mut graph);
            if !graph.is_null() {
                (drv.cuGraphDestroy)(graph);
            }
        }
    }

    /// Replay one captured token on the decode stream.
    pub fn launch_decode_graph(&mut self, parity: usize) -> Result<(), CudaError> {
        let Some(exec) = self.decode_exec[parity] else {
            return Err(CudaError::Other("decode graph is not captured".into()));
        };
        self.ensure_stream()?;
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check("cuGraphLaunch", (drv.cuGraphLaunch)(exec, self.stream))
        }
    }

    // ------------------------------------------------------------------ events

    /// Record a phase event and return its handle.
    ///
    /// `on_decode_stream` picks the stream the bracketed work was enqueued on:
    /// the capture stream for replayed decode tokens, the default stream for
    /// eager launches. Recording during an active capture is an error (the
    /// event would become a node of the replayed graph).
    pub fn record_phase_event(&self, on_decode_stream: bool) -> Result<PhaseEvent, CudaError> {
        if self.capturing {
            return Err(CudaError::Other(
                "cannot record a phase event during decode capture".into(),
            ));
        }
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            let mut ev: ffi::CUevent = std::ptr::null_mut();
            drv.check("cuEventCreate", (drv.cuEventCreate)(&mut ev, 0))?;
            let stream = if on_decode_stream && !self.stream.is_null() {
                self.stream
            } else {
                std::ptr::null_mut()
            };
            drv.check("cuEventRecord", (drv.cuEventRecord)(ev, stream))?;
            Ok(PhaseEvent { ev })
        }
    }

    /// GPU milliseconds from `a` to `b` (waits for `b` to complete).
    pub fn phase_event_elapsed_ms(&self, a: PhaseEvent, b: PhaseEvent) -> Result<f64, CudaError> {
        let drv = &self.shared.drv;
        unsafe {
            drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            drv.check("cuEventSynchronize", (drv.cuEventSynchronize)(b.ev))?;
            let mut ms: f32 = 0.0;
            drv.check(
                "cuEventElapsedTime",
                (drv.cuEventElapsedTime)(&mut ms, a.ev, b.ev),
            )?;
            Ok(ms as f64)
        }
    }

    /// Free a phase event.
    pub fn phase_event_destroy(&self, ev: PhaseEvent) {
        if ev.ev.is_null() {
            return;
        }
        let drv = &self.shared.drv;
        unsafe {
            let _ = (drv.cuCtxSetCurrent)(self.shared.ctx);
            let _ = (drv.cuEventDestroy)(ev.ev);
        }
    }

    /// Block until every queued launch and copy has completed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        let drv = &self.shared.drv;
        unsafe { drv.check("cuCtxSynchronize", (drv.cuCtxSynchronize)()) }
    }

    // -------------------------------------------------------------------- nvcc

    /// `nvcc --cubin` `source` for this device's compute capability, returning
    /// the cubin image.
    ///
    /// Public so `ksearch cuda --compile` can validate every rendered kernel
    /// against the real toolkit without loading it.
    pub fn compile(&self, source: &str, name: &str) -> Result<Vec<u8>, CudaError> {
        let nvcc = self.shared.nvcc.clone().ok_or_else(|| {
            CudaError::Other(format!(
                "no `nvcc` found on PATH; install the CUDA toolkit or set {NVCC_ENV}"
            ))
        })?;
        nvcc_cubin(&nvcc, &self.arch(), source, name)
    }
}

/// `nvcc --cubin` argument list for one kernel: the `-arch` value is what makes
/// an H100 build `sm_90` and an RTX 5090 build `sm_120`.
fn nvcc_cubin_args(arch: &str, src: &Path, out: &Path) -> Vec<std::ffi::OsString> {
    vec![
        "--cubin".into(),
        "-std=c++17".into(),
        "-O3".into(),
        format!("-arch={arch}").into(),
        "-o".into(),
        out.into(),
        src.into(),
    ]
}

/// Build `source` (a `ksearch_codegen` CUDA kernel) into a cubin with `nvcc`
/// for `arch`, e.g. `sm_90`.
///
/// Split out of [`CudaContext::compile`] so the exact targeting of the H100 /
/// RTX 50-series build can be tested with a stub `nvcc` on a machine that has
/// no NVIDIA GPU.
pub fn nvcc_cubin(nvcc: &Path, arch: &str, source: &str, name: &str) -> Result<Vec<u8>, CudaError> {
    let dir = std::env::temp_dir().join(format!("ksearch_cuda_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| CudaError::Other(e.to_string()))?;
    let cu = dir.join(format!("{name}.cu"));
    let cubin = dir.join(format!("{name}.{}.cubin", source_key(source)));
    std::fs::write(&cu, source).map_err(|e| CudaError::Other(e.to_string()))?;

    let out = std::process::Command::new(nvcc)
        .args(nvcc_cubin_args(arch, &cu, &cubin))
        .output()
        .map_err(|e| CudaError::Other(format!("failed to run `{}`: {e}", nvcc.display())))?;
    if !out.status.success() {
        return Err(CudaError::Other(format!(
            "nvcc failed for `{name}` ({}):\n{}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        )));
    }
    std::fs::read(&cubin).map_err(|e| CudaError::Other(e.to_string()))
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.abort_decode_capture();
        // Graphs and the stream reference the context, so they go before modules.
        unsafe {
            let drv = &self.shared.drv;
            let _ = drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx));
            for slot in &mut self.decode_exec {
                if let Some(exec) = slot.take() {
                    (drv.cuGraphExecDestroy)(exec);
                }
            }
            if !self.stream.is_null() {
                (drv.cuStreamDestroy)(self.stream);
                self.stream = std::ptr::null_mut();
            }
            for module in self.modules.values() {
                (drv.cuModuleUnload)(*module);
            }
        }
        self.modules.clear();
    }
}

/// Device memory. Cloning shares the same allocation, matching the Metal buffer
/// semantics the model layer was written against.
pub struct Buffer {
    alloc: Arc<Alloc>,
}

struct Alloc {
    shared: Arc<Shared>,
    dptr: ffi::CUdeviceptr,
    bytes: usize,
}

impl Drop for Alloc {
    fn drop(&mut self) {
        if self.dptr != 0 {
            unsafe {
                let drv = &self.shared.drv;
                let _ = (drv.cuCtxSetCurrent)(self.shared.ctx);
                (drv.cuMemFree)(self.dptr);
            }
            self.dptr = 0;
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            alloc: Arc::clone(&self.alloc),
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("ptr", &format_args!("{:#x}", self.alloc.dptr))
            .field("bytes", &self.alloc.bytes)
            .finish()
    }
}

impl Buffer {
    /// Device pointer (`cuMemAlloc` handle).
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.alloc.dptr
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.alloc.bytes
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    fn dptr(&self) -> CUdeviceptr {
        self.alloc.dptr
    }

    fn check_range(&self, off: usize, len: usize, what: &str) -> Result<(), CudaError> {
        if off + len > self.alloc.bytes {
            return Err(CudaError::Other(format!(
                "{what}: [{off}, {}) is outside a {} byte buffer",
                off + len,
                self.alloc.bytes
            )));
        }
        Ok(())
    }
}

/// Decode-state kernels compiled once per context.
#[derive(Clone, Copy)]
struct RuntimeKernels {
    fill: Kernel,
    inc: Kernel,
    scatter: Kernel,
    stash: Kernel,
    stash_inc: Kernel,
    deq_q4k: Kernel,
    deq_q6k: Kernel,
    mm_reduce: Kernel,
}

const RUNTIME_SRC: &str = r#"
extern "C" __global__ void ks_fill_attn_meta(
    const unsigned long long* metas,
    const float* pos,
    unsigned long long swa_mask,
    unsigned n_layers,
    unsigned window) {
  unsigned layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer >= n_layers || layer >= 64u) return;
  unsigned kv = (unsigned)pos[0] + 1u;
  unsigned start = 0u;
  unsigned tlen = kv;
  if ((swa_mask >> layer) & 1ull) {
    start = kv > window ? kv - window : 0u;
    tlen = kv - start;
  }
  float* dst = (float*)metas[layer];
  dst[0] = (float)tlen;
  dst[1] = (float)start;
  dst[2] = 0.0f;
}

extern "C" __global__ void ks_inc_u32(float* pos) { pos[0] += 1.0f; }

extern "C" __global__ void ks_scatter_meta(
    const unsigned long long* metas, const float* src, unsigned n_layers) {
  unsigned layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer >= n_layers) return;
  float* dst = (float*)metas[layer];
  const float* s = src + layer * 4u;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
}

extern "C" __global__ void ks_stash_f32(const float* src, float* dst, unsigned index) {
  dst[index] = src[0];
}

extern "C" __global__ void ks_stash_f32_inc(const float* src, float* dst, unsigned* index) {
  unsigned i = index[0];
  dst[i] = src[0];
  index[0] = i + 1u;
}
"#;

/// One-time weight dequantization into the F16 prefill cache.
///
/// These kernels are the `ksearch_load_q4k` / `ksearch_load_q6k` expanders
/// written out as plain CUDA (the runtime kernels are compiled standalone,
/// without the MSL-dialect prelude). Each Q4_K thread dequantizes one 16
/// element chunk — the same recipe the WMMA tile proved on device — and the
/// Q6_K kernel maps one thread per element.
const DEQUANT_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ks_dequant_q4k_f16(
    const unsigned char* src, half* dst, unsigned long long n16) {
  unsigned long long c = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  if (c >= n16) return;
  unsigned long long e = c * 16ull;
  const unsigned char* blk = src + (e >> 8) * 144ull;
  const float d = __half2float(*((const half*)blk));
  const float dmin = __half2float(*((const half*)(blk + 2)));
  const unsigned char* scales = blk + 4;
  const unsigned char* qs = blk + 16;
  const unsigned j = (unsigned)(e & 255ull);
  const unsigned group = j >> 6;
  const unsigned jl = j & 63u;
  const unsigned is = (group << 1) | (jl >= 32u ? 1u : 0u);
  const unsigned qbase = (group << 5) | (jl & 16u);
  unsigned char sc, mn;
  if (is < 4u) {
    sc = (unsigned char)(scales[is] & 63u);
    mn = (unsigned char)(scales[is + 4u] & 63u);
  } else {
    sc = (unsigned char)((scales[is + 4u] & 0x0Fu) | ((scales[is - 4u] & 0xC0u) >> 2));
    mn = (unsigned char)((scales[is + 4u] >> 4) | ((scales[is] & 0xC0u) >> 2));
  }
  const float scale = d * (float)sc;
  const float m = dmin * (float)mn;
  const bool hi = jl >= 32u;
  half* out = dst + e;
#pragma unroll
  for (unsigned l = 0u; l < 16u; l += 4u) {
    const unsigned bits = *(const unsigned*)(qs + qbase + l);
    float n0, n1, n2, n3;
    if (!hi) {
      n0 = (float)(bits & 0xFu);
      n1 = (float)((bits >> 8) & 0xFu);
      n2 = (float)((bits >> 16) & 0xFu);
      n3 = (float)((bits >> 24) & 0xFu);
    } else {
      n0 = (float)((bits >> 4) & 0xFu);
      n1 = (float)((bits >> 12) & 0xFu);
      n2 = (float)((bits >> 20) & 0xFu);
      n3 = (float)((bits >> 28) & 0xFu);
    }
    out[l] = __float2half(scale * n0 - m);
    out[l + 1u] = __float2half(scale * n1 - m);
    out[l + 2u] = __float2half(scale * n2 - m);
    out[l + 3u] = __float2half(scale * n3 - m);
  }
}

static __device__ float ks_q6k_at(const unsigned char* blk, unsigned j) {
  const unsigned hpart = j >> 7;
  const unsigned char* ql = blk + hpart * 64u;
  const unsigned char* qh = blk + 128u + hpart * 32u;
  const signed char* sc = (const signed char*)(blk + 192u) + hpart * 8u;
  const float d = __half2float(*((const half*)(blk + 208)));
  const unsigned jl = j & 127u;
  const unsigned l = jl & 31u;
  const unsigned is = l >> 4;
  int q;
  if (jl < 32u) {
    q = (int)((ql[l] & 0x0Fu) | (((qh[l] >> 0) & 3) << 4)) - 32;
    return d * (float)sc[is] * (float)q;
  } else if (jl < 64u) {
    q = (int)((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3) << 4)) - 32;
    return d * (float)sc[is + 2] * (float)q;
  } else if (jl < 96u) {
    q = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    return d * (float)sc[is + 4] * (float)q;
  } else {
    q = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * (float)sc[is + 6] * (float)q;
  }
}

extern "C" __global__ void ks_dequant_q6k_f16(
    const unsigned char* src, half* dst, unsigned long long n) {
  unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half(ks_q6k_at(src + (i >> 8) * 210ull, (unsigned)(i & 255ull)));
}

/* Sum split-K F32 partials into the F16 GEMM output:
   out[i] = sum_s parts[s * nm + i]. */
extern "C" __global__ void ks_f16_mm_reduce(
    const float* parts, half* out, unsigned long long nm, unsigned splits) {
  unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  if (i >= nm) return;
  float acc = 0.0f;
  for (unsigned s = 0u; s < splits; ++s) {
    acc += parts[(unsigned long long)s * nm + i];
  }
  out[i] = __float2half(acc);
}
"#;

/// A loaded kernel entry point (a `CUfunction` belonging to its context's module).
///
/// `Copy` because it is a bare handle: the module is owned by the
/// [`CudaContext`] that loaded it.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    func: ffi::CUfunction,
}

/// Opaque driver event handle for phase timing. Pass it back to the
/// [`CudaContext`] methods; it is `Copy` and freed with
/// [`CudaContext::phase_event_destroy`].
#[derive(Clone, Copy, Debug)]
pub struct PhaseEvent {
    ev: ffi::CUevent,
}

impl PhaseEvent {
    /// Raw `CUevent`, for the backend-neutral wrapper in `ksearch_compute`.
    pub fn raw(&self) -> ffi::CUevent {
        self.ev
    }

    /// Wrap a raw `CUevent` produced by this crate's context.
    pub fn from_raw(ev: ffi::CUevent) -> Self {
        PhaseEvent { ev }
    }
}

impl Kernel {
    /// Raw `CUfunction`, for a caller driving `cuLaunchKernel` itself.
    pub fn raw(&self) -> ffi::CUfunction {
        self.func
    }
}

/// Type-erased kernel argument list.
///
/// `cuLaunchKernel` takes `void**` — pointers to the argument values — so the
/// values must stay put until the launch. They live in one growable byte store,
/// and the pointer array is only computed once the list is complete.
#[derive(Default)]
pub struct KernelArgs {
    store: Vec<u8>,
    at: Vec<usize>,
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, bytes: &[u8], align: usize) {
        while !self.store.len().is_multiple_of(align) {
            self.store.push(0);
        }
        self.at.push(self.store.len());
        self.store.extend_from_slice(bytes);
    }

    /// Device pointer argument (buffers are `CUdeviceptr` in the generated code).
    pub fn ptr(&mut self, p: CUdeviceptr) -> &mut Self {
        self.push(&p.to_le_bytes(), 8);
        self
    }

    /// `uint` / `int` scalar.
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.push(&v.to_le_bytes(), 4);
        self
    }

    /// `float` scalar.
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.push(&v.to_le_bytes(), 4);
        self
    }

    /// 64-bit scalar (`size_t` / `ulonglong`).
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.push(&v.to_le_bytes(), 8);
        self
    }

    pub fn len(&self) -> usize {
        self.at.len()
    }

    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// Argument pointer array, valid until the next mutation.
    ///
    /// Public so tests can pin the positional contract (`in0..inN, out`) without
    /// needing a device.
    pub fn pointers(&mut self) -> Vec<*mut std::ffi::c_void> {
        let base = self.store.as_mut_ptr();
        self.at
            .iter()
            .map(|off| unsafe { base.add(*off) } as *mut std::ffi::c_void)
            .collect()
    }
}

/// `nvcc` from `KSEARCH_NVCC`, else `nvcc` on `PATH`, else `None`.
fn find_nvcc() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(NVCC_ENV) {
        return Some(PathBuf::from(p));
    }
    let ok = std::process::Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then(|| PathBuf::from("nvcc"))
}

/// Stable hash of a kernel source, used as the module-cache key.
fn source_key(src: &str) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut h = DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel the driver can run on any compute capability.
    const ADD_F32: &str = r#"
extern "C" __global__ void ks_add_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

    #[test]
    fn arch_names_follow_compute_capability() {
        let h100 = DeviceInfo {
            major: 9,
            minor: 0,
            ..Default::default()
        };
        let rtx5090 = DeviceInfo {
            major: 12,
            minor: 0,
            ..Default::default()
        };
        assert_eq!(h100.sm_arch(), "sm_90");
        assert_eq!(rtx5090.sm_arch(), "sm_120");
    }

    #[test]
    fn availability_check_never_panics() {
        // `false` on this machine, `true` on a CUDA box.
        let _ = CudaContext::is_available();
    }

    #[test]
    fn missing_driver_is_a_clean_error() {
        match CudaContext::with_options(0, DEFAULT_TG) {
            Ok(ctx) => {
                assert!(!ctx.device_name().is_empty());
                assert!(ctx.arch().starts_with("sm_"));
                assert!(ctx.info().max_threads_per_block > 0);
            }
            Err(CudaError::DriverUnavailable(msg)) => {
                assert!(msg.contains("libcuda"), "name the libraries tried: {msg}");
            }
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn args_get_stable_aligned_addresses() {
        let mut a = KernelArgs::new();
        a.ptr(0xdead_beef).u32(7).f32(1.5).u64(9);
        assert_eq!(a.len(), 4);
        let p = a.pointers();
        let (lo, hi) = (
            a.store.as_ptr() as usize,
            a.store.as_ptr() as usize + a.store.len(),
        );
        for q in &p {
            let v = *q as usize;
            assert!(v >= lo && v < hi, "argument pointer outside the store");
        }
        // The `ptr` and `u64` arguments must be 8-byte aligned.
        assert_eq!(a.store.len() % 8, 0);
    }

    /// End-to-end: compile with nvcc, load the cubin, launch, read back.
    ///
    /// Skipped when the machine has no driver; this is the real check on an
    /// H100 / RTX 50-series box (`cargo test -p ksearch_cuda`).
    #[test]
    fn gpu_round_trip_when_available() {
        if !CudaContext::is_available() {
            eprintln!("no CUDA device: skipping round-trip");
            return;
        }
        let mut ctx = CudaContext::with_options(0, DEFAULT_TG).expect("open device");
        if ctx.nvcc().is_none() {
            eprintln!("no nvcc: skipping round-trip");
            return;
        }
        let k = ctx.load_source("ks_add_f32", ADD_F32).expect("load kernel");
        assert!(!k.raw().is_null());

        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i * 2) as f32).collect();
        let (ab, bb) = (as_bytes(&a), as_bytes(&b));

        let ba = ctx.alloc(ab.len()).unwrap();
        let bb_buf = ctx.alloc(bb.len()).unwrap();
        let bout = ctx.alloc(ab.len()).unwrap();
        ctx.copy_to(&ba, &ab).unwrap();
        ctx.copy_to(&bb_buf, &bb).unwrap();

        let mut args = KernelArgs::new();
        args.ptr(ba.as_ptr())
            .ptr(bb_buf.as_ptr())
            .ptr(bout.as_ptr())
            .u32(64);
        let hint = ksearch_codegen::LaunchHint::Elementwise { n: 64 };
        ctx.launch(k, &hint, &mut args).unwrap();
        ctx.synchronize().unwrap();

        let mut out = vec![0u8; ab.len()];
        ctx.copy_from(&bout, &mut out).unwrap();
        for (i, got) in from_bytes(&out).iter().enumerate() {
            assert!((got - (a[i] + b[i])).abs() < 1e-6, "elem {i}: {got}");
        }
    }

    fn as_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn from_bytes(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ------------------------------------------------------------- nvcc targeting
    //
    // These run on any machine (no NVIDIA driver needed): they drive the real
    // `nvcc_cubin` entry point with a stub `nvcc`, which is what proves the
    // H100 / RTX 50-series `-arch` selection and the error paths.

    #[cfg(unix)]
    fn stub_nvcc(tag: &str, body: &str) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let dir =
            std::env::temp_dir().join(format!("ksearch_nvcc_stub_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("argv.log");
        let script = dir.join("nvcc");
        let log_q = log.to_string_lossy().replace('\'', "'\\''");
        let script_body = format!("#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{log_q}'\n{body}\n");
        std::fs::write(&script, script_body).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        (script, log)
    }

    /// `-o` / `<file>` pairs from a recorded argv log.
    #[cfg(unix)]
    fn out_path(log: &Path) -> PathBuf {
        let argv = std::fs::read_to_string(log).unwrap();
        let mut it = argv.lines();
        while let Some(a) = it.next() {
            if a == "-o" {
                return PathBuf::from(it.next().expect("-o value"));
            }
        }
        panic!("no -o in nvcc argv:\n{argv}");
    }

    #[cfg(unix)]
    #[test]
    fn nvcc_targets_the_arch_of_the_card() {
        // The stub writes a fake cubin where `-o` points, like a real nvcc.
        let (nvcc, log) = stub_nvcc("arch", "out=\"\"\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then out=\"$2\"; fi\n  shift\ndone\nprintf 'CUBIN' > \"$out\"");
        for arch in ["sm_90", "sm_120"] {
            let name = format!("ks_probe_{arch}");
            let cubin = nvcc_cubin(&nvcc, arch, ADD_F32, &name).unwrap();
            assert_eq!(cubin, b"CUBIN", "stub cubin for {arch}");
            let argv = std::fs::read_to_string(&log).unwrap();
            assert!(
                argv.contains(&format!("-arch={arch}")),
                "argv for {arch}:\n{argv}"
            );
            assert!(argv.contains("--cubin"), "argv:\n{argv}");
            assert!(argv.contains("-std=c++17"), "argv:\n{argv}");
            let cu = argv.lines().find(|l| l.ends_with(&format!("{name}.cu")));
            assert!(cu.is_some(), "source path missing from argv:\n{argv}");
        }
        assert_eq!(
            out_path(&log).extension().unwrap(),
            "cubin",
            "cubin written where nvcc was told"
        );
    }

    #[cfg(unix)]
    #[test]
    fn nvcc_failure_surfaces_the_compiler_output() {
        let (nvcc, _) = stub_nvcc("fail", "echo 'ks_add_f32: error: boom' >&2\nexit 1");
        let err = nvcc_cubin(&nvcc, "sm_90", ADD_F32, "ks_bad").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nvcc failed for `ks_bad`"), "{msg}");
        assert!(
            msg.contains("boom"),
            "compiler stderr must be reported: {msg}"
        );
    }

    #[test]
    fn missing_nvcc_binary_is_a_clean_error() {
        let missing = std::env::temp_dir().join("ksearch_no_such_nvcc_12345");
        let err = nvcc_cubin(&missing, "sm_90", ADD_F32, "ks_missing").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("failed to run"), "{msg}");
        assert!(msg.contains("ksearch_no_such_nvcc_12345"), "{msg}");
    }
}

//! CUDA driver API bindings, resolved at run time through `dlopen`.
//!
//! The driver is loaded lazily so the crate builds and links on machines with
//! no CUDA install (macOS CI, CPU-only Linux): [`Driver::load`] then reports
//! `CudaError::DriverUnavailable`. Nothing here needs `libcuda` at link time
//! and no external crate is used to reach it.
#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
/// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 39;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

const RTLD_NOW: c_int = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// Everything that can go wrong reaching the driver or a device.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// No `libcuda`/`nvcuda` on this machine (CI, macOS, CPU-only Linux).
    #[error("CUDA driver library not found (tried {0})")]
    DriverUnavailable(String),
    #[error("CUDA driver is missing symbol `{0}`")]
    MissingSymbol(&'static str),
    #[error("{call} failed: {name} ({code})")]
    Driver {
        call: &'static str,
        name: String,
        code: CUresult,
    },
    #[error("no CUDA devices found")]
    NoDevice,
    #[error("device {0} out of range (0..{1})")]
    BadDevice(usize, c_int),
    #[error("{0}")]
    Other(String),
}

fn dlerror_string() -> String {
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Resolved entry points of `libcuda`.
pub struct Driver {
    /// Kept open for the lifetime of the table.
    _lib: *mut c_void,
    error_name: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    pub cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult,
    pub cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    #[allow(clippy::type_complexity)]
    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    pub cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemsetD8: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    pub cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuMemcpyDtoD: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
}

/// Names to try, Linux first (H100 / RTX 50-series), then Windows / macOS.
pub const DRIVER_LIB_NAMES: [&str; 4] =
    ["libcuda.so.1", "libcuda.so", "nvcuda.dll", "libcuda.dylib"];

impl Driver {
    /// `dlopen` the CUDA driver and resolve the entry points used here.
    pub fn load() -> Result<Self, CudaError> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        for n in DRIVER_LIB_NAMES {
            let cs = CString::new(n).expect("no NUL");
            handle = unsafe { dlopen(cs.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            return Err(CudaError::DriverUnavailable(format!(
                "{} ({})",
                DRIVER_LIB_NAMES.join(", "),
                dlerror_string()
            )));
        }

        macro_rules! sym {
            ($name:literal, $t:ty) => {{
                let cs = CString::new($name).expect("no NUL");
                let p = unsafe { dlsym(handle, cs.as_ptr()) };
                if p.is_null() {
                    unsafe { dlclose(handle) };
                    return Err(CudaError::MissingSymbol($name));
                }
                unsafe { std::mem::transmute::<*mut c_void, $t>(p) }
            }};
        }

        Ok(Driver {
            _lib: handle,
            error_name: sym!(
                "cuGetErrorName",
                unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult
            ),
            cuInit: sym!("cuInit", unsafe extern "C" fn(c_uint) -> CUresult),
            cuDeviceGetCount: sym!(
                "cuDeviceGetCount",
                unsafe extern "C" fn(*mut c_int) -> CUresult
            ),
            cuDeviceGet: sym!(
                "cuDeviceGet",
                unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult
            ),
            cuDeviceGetName: sym!(
                "cuDeviceGetName",
                unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult
            ),
            cuDeviceGetAttribute: sym!(
                "cuDeviceGetAttribute",
                unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult
            ),
            cuDeviceTotalMem: sym!(
                "cuDeviceTotalMem_v2",
                unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult
            ),
            cuCtxCreate: sym!(
                "cuCtxCreate_v2",
                unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult
            ),
            cuCtxDestroy: sym!(
                "cuCtxDestroy_v2",
                unsafe extern "C" fn(CUcontext) -> CUresult
            ),
            cuCtxSetCurrent: sym!(
                "cuCtxSetCurrent",
                unsafe extern "C" fn(CUcontext) -> CUresult
            ),
            cuCtxSynchronize: sym!("cuCtxSynchronize", unsafe extern "C" fn() -> CUresult),
            cuModuleLoadData: sym!(
                "cuModuleLoadData",
                unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult
            ),
            cuModuleUnload: sym!("cuModuleUnload", unsafe extern "C" fn(CUmodule) -> CUresult),
            cuModuleGetFunction: sym!(
                "cuModuleGetFunction",
                unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult
            ),
            cuFuncSetAttribute: sym!(
                "cuFuncSetAttribute",
                unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult
            ),
            cuLaunchKernel: sym!(
                "cuLaunchKernel",
                unsafe extern "C" fn(
                    CUfunction,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    CUstream,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> CUresult
            ),
            cuMemAlloc: sym!(
                "cuMemAlloc_v2",
                unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult
            ),
            cuMemFree: sym!(
                "cuMemFree_v2",
                unsafe extern "C" fn(CUdeviceptr) -> CUresult
            ),
            cuMemsetD8: sym!(
                "cuMemsetD8_v2",
                unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult
            ),
            cuMemcpyHtoD: sym!(
                "cuMemcpyHtoD_v2",
                unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult
            ),
            cuMemcpyDtoH: sym!(
                "cuMemcpyDtoH_v2",
                unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult
            ),
            cuMemcpyDtoD: sym!(
                "cuMemcpyDtoD_v2",
                unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult
            ),
        })
    }

    /// Human-readable name of a driver status code.
    pub fn error_name(&self, code: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        let rc = unsafe { (self.error_name)(code, &mut p) };
        if rc != CUDA_SUCCESS || p.is_null() {
            format!("CUresult {code}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    /// `?`-able wrapper around a driver call.
    pub fn check(&self, call: &'static str, code: CUresult) -> Result<(), CudaError> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Driver {
                call,
                name: self.error_name(code),
                code,
            })
        }
    }
}

// The table is immutable after construction and points into a process-global
// library, so it can be shared between threads.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

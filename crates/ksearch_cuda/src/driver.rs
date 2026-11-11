//! Minimal CUDA *driver API* binding (no toolkit needed at build time).
//!
//! `libcuda` is opened with `dlopen` and every entry point is resolved lazily,
//! so this crate builds and links on machines without a CUDA installation
//! (macOS CI, CPU-only build hosts). [`Driver::load`] fails with a readable
//! error when no driver is present.

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

/// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`
pub const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
pub const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;

/// Shared memory a kernel may request without an explicit opt-in.
pub const DEFAULT_MAX_DYNAMIC_SHARED: u32 = 48 * 1024;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
extern "C" {}

const RTLD_NOW: c_int = 2;

/// Driver library names, most specific first.
const LIB_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so", "libcuda.dylib", "nvcuda.dll"];

macro_rules! driver_fn {
    ($($field:ident: $ty:ty),* $(,)?) => {
        /// Resolved driver entry points.
        pub struct Driver {
            /// Keeps the library mapped for the lifetime of the table.
            handle: *mut c_void,
            $(pub $field: $ty,)*
        }
    };
}

driver_fn! {
    cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    cuCtxDestroy_v2: unsafe extern "C" fn(CUcontext) -> CUresult,
    cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint, c_uint, c_uint,
        c_uint, c_uint, c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cuMemcpyDtoD_v2: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    cuMemsetD8_v2: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    cuGetErrorName: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

impl Driver {
    /// Open `libcuda` and resolve every entry point.
    pub fn load() -> Result<Driver, String> {
        let mut handle = std::ptr::null_mut();
        for name in LIB_NAMES {
            let cs = CString::new(*name).expect("static name");
            handle = unsafe { dlopen(cs.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            return Err(format!(
                "CUDA driver library not found (tried {}). Install the NVIDIA driver or run on a CUDA machine.",
                LIB_NAMES.join(", ")
            ));
        }

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let cs = CString::new($name).expect("static name");
                let p = unsafe { dlsym(handle, cs.as_ptr()) };
                if p.is_null() {
                    return Err(format!("CUDA driver symbol `{}` not found", $name));
                }
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }

        Ok(Driver {
            handle,
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
            cuCtxCreate_v2: sym!(
                "cuCtxCreate_v2",
                unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult
            ),
            cuCtxDestroy_v2: sym!(
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
            cuMemAlloc_v2: sym!(
                "cuMemAlloc_v2",
                unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult
            ),
            cuMemFree_v2: sym!(
                "cuMemFree_v2",
                unsafe extern "C" fn(CUdeviceptr) -> CUresult
            ),
            cuMemcpyHtoD_v2: sym!(
                "cuMemcpyHtoD_v2",
                unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult
            ),
            cuMemcpyDtoH_v2: sym!(
                "cuMemcpyDtoH_v2",
                unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult
            ),
            cuMemcpyDtoD_v2: sym!(
                "cuMemcpyDtoD_v2",
                unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult
            ),
            cuMemsetD8_v2: sym!(
                "cuMemsetD8_v2",
                unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult
            ),
            cuGetErrorName: sym!(
                "cuGetErrorName",
                unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult
            ),
        })
    }

    /// `CUresult` → readable error string, including the symbolic name.
    pub fn error(&self, code: CUresult) -> String {
        let mut name: *const c_char = std::ptr::null();
        let sym = unsafe { (self.cuGetErrorName)(code, &mut name) };
        if sym == CUDA_SUCCESS && !name.is_null() {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUresult({code})")
        }
    }

    /// Release the driver table.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every context, module and device allocation
    /// created through this driver has already been released, and that no other
    /// thread can still use it.
    pub unsafe fn close(self) {
        // `dlclose` is deliberately not called: CUDA keeps background threads
        // and unloading the driver mid-process is not supported. The mapping
        // simply lives until the process exits.
        let _ = self.handle;
    }
}

// The table is immutable after load; every field is a plain function pointer.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

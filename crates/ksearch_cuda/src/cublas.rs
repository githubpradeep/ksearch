//! cuBLAS (dlopen) for F16 prefill GEMM — faster than the hand WMMA tile on A10.
//!
//! Layout (row-major): `Y[n, m] = sum_k W[m, k] * X[n, k]` with `W: M×K`,
//! `X: N×K`, `Y: N×M`, all F16. Uses `cublasGemmEx` with FP32 accumulate.

use crate::ffi::{CUdeviceptr, CUstream, CudaError};
use crate::Buffer;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

type CublasStatus = c_int;
type CublasHandle = *mut c_void;
type CudaDataType = c_int;
type CublasComputeType = c_int;
type CublasOperation = c_int;
type CublasGemmAlgo = c_int;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: CublasOperation = 0;
const CUBLAS_OP_T: CublasOperation = 1;
const CUDA_R_16F: CudaDataType = 2;
const CUBLAS_COMPUTE_32F: CublasComputeType = 68;
const CUBLAS_GEMM_DEFAULT: CublasGemmAlgo = -1;
const RTLD_NOW: c_int = 2;

const LIB_NAMES: &[&str] = &[
    "libcublas.so.12",
    "libcublas.so.11",
    "libcublas.so",
];

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

struct Api {
    _lib: *mut c_void,
    create: unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus,
    destroy: unsafe extern "C" fn(CublasHandle) -> CublasStatus,
    set_stream: unsafe extern "C" fn(CublasHandle, CUstream) -> CublasStatus,
    gemm_ex: unsafe extern "C" fn(
        CublasHandle,
        CublasOperation,
        CublasOperation,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        CudaDataType,
        c_int,
        *const c_void,
        CudaDataType,
        c_int,
        *const c_void,
        *mut c_void,
        CudaDataType,
        c_int,
        CublasComputeType,
        CublasGemmAlgo,
    ) -> CublasStatus,
}

// Safety: symbols are immutable for process life; handle is per-context.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn load_api() -> Result<&'static Api, CudaError> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    match API.get_or_init(|| {
        let mut handle = std::ptr::null_mut();
        for n in LIB_NAMES {
            let cs = CString::new(*n).expect("lib name");
            handle = unsafe { dlopen(cs.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "cublas not found".into()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(err);
        }
        macro_rules! sym {
            ($name:literal, $t:ty) => {{
                let cs = CString::new($name).expect("sym");
                let p = unsafe { dlsym(handle, cs.as_ptr()) };
                if p.is_null() {
                    unsafe { dlclose(handle) };
                    return Err(format!("missing cublas symbol {}", $name));
                }
                unsafe { std::mem::transmute::<*mut c_void, $t>(p) }
            }};
        }
        Ok(Api {
            _lib: handle,
            create: sym!(
                "cublasCreate_v2",
                unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus
            ),
            destroy: sym!(
                "cublasDestroy_v2",
                unsafe extern "C" fn(CublasHandle) -> CublasStatus
            ),
            set_stream: sym!(
                "cublasSetStream_v2",
                unsafe extern "C" fn(CublasHandle, CUstream) -> CublasStatus
            ),
            gemm_ex: sym!(
                "cublasGemmEx",
                unsafe extern "C" fn(
                    CublasHandle,
                    CublasOperation,
                    CublasOperation,
                    c_int,
                    c_int,
                    c_int,
                    *const c_void,
                    *const c_void,
                    CudaDataType,
                    c_int,
                    *const c_void,
                    CudaDataType,
                    c_int,
                    *const c_void,
                    *mut c_void,
                    CudaDataType,
                    c_int,
                    CublasComputeType,
                    CublasGemmAlgo,
                ) -> CublasStatus
            ),
        })
    }) {
        Ok(api) => Ok(api),
        Err(e) => Err(CudaError::Other(format!("cublas: {e}"))),
    }
}

fn check(call: &'static str, st: CublasStatus) -> Result<(), CudaError> {
    if st == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Other(format!("{call} failed: status={st}")))
    }
}

/// Per-context cuBLAS handle.
pub struct Cublas {
    handle: CublasHandle,
}

impl Drop for Cublas {
    fn drop(&mut self) {
        if let Ok(api) = load_api() {
            if !self.handle.is_null() {
                unsafe {
                    (api.destroy)(self.handle);
                }
                self.handle = std::ptr::null_mut();
            }
        }
    }
}

impl Cublas {
    pub fn new() -> Result<Self, CudaError> {
        let api = load_api()?;
        let mut handle = std::ptr::null_mut();
        check("cublasCreate", unsafe { (api.create)(&mut handle) })?;
        Ok(Self { handle })
    }

    /// `Y[n,m] = sum_k W[m,k] * X[n,k]` (all F16, row-major).
    pub fn gemm_f16_nt(
        &self,
        stream: CUstream,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        let api = load_api()?;
        check("cublasSetStream", unsafe { (api.set_stream)(self.handle, stream) })?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Col-major view of row-major buffers:
        //   W_rm M×K ld=K  ↔  W_cm as K×M with op=T → M×K
        //   X_rm N×K ld=K  ↔  X_cm as K×N with op=N → K×N
        //   Y_rm N×M ld=M  ↔  Y_cm as M×N ld=M
        // Y_cm = W^T * X  =>  op(A)=T, op(B)=N, A=W, B=X, C=Y
        check(
            "cublasGemmEx",
            unsafe {
                (api.gemm_ex)(
                    self.handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha as *const f32 as *const c_void,
                    w as *const c_void,
                    CUDA_R_16F,
                    k,
                    x as *const c_void,
                    CUDA_R_16F,
                    k,
                    &beta as *const f32 as *const c_void,
                    y as *mut c_void,
                    CUDA_R_16F,
                    m,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            },
        )
    }
}

impl super::CudaContext {
    fn cublas(&mut self) -> Result<&Cublas, CudaError> {
        if self.cublas.is_none() {
            let drv = &self.shared.drv;
            unsafe {
                drv.check("cuCtxSetCurrent", (drv.cuCtxSetCurrent)(self.shared.ctx))?;
            }
            self.cublas = Some(Cublas::new()?);
        }
        Ok(self.cublas.as_ref().unwrap())
    }

    /// Prefill F16 GEMM via cuBLAS: `Y[n,m] = sum_k W[m,k] * X[n,k]`.
    pub fn gemm_f16_cublas(
        &mut self,
        w: &Buffer,
        x: &Buffer,
        y: &Buffer,
        rows: u32,
        cols: u32,
        batch: u32,
    ) -> Result<(), CudaError> {
        let stream = self.stream;
        let blas = self.cublas()?;
        blas.gemm_f16_nt(
            stream,
            w.as_ptr(),
            x.as_ptr(),
            y.as_ptr(),
            rows as i32,
            batch as i32,
            cols as i32,
        )
    }

    /// `true` when libcublas can be loaded.
    pub fn cublas_available() -> bool {
        load_api().is_ok()
    }
}

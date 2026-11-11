//! Kernel argument marshalling for `cuLaunchKernel`.
//!
//! `cuLaunchKernel` takes an array of pointers to argument values, so the
//! arguments must outlive the launch. [`KernelArgs`] owns the storage and hands
//! out stable pointers, which removes the classic "pointer to a temporary" bug
//! of hand-written `void**` arrays.

use std::ffi::c_void;

use crate::driver::CUdeviceptr;

/// Owned argument blob for one kernel launch.
#[derive(Default, Debug)]
pub struct KernelArgs {
    store: Vec<u8>,
    at: Vec<usize>,
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of arguments pushed so far.
    pub fn len(&self) -> usize {
        self.at.len()
    }

    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    fn push8(&mut self, bytes: &[u8]) {
        // Keep every slot 8-byte aligned so a device pointer or `f64` scalar can
        // never end up on a 4-byte boundary.
        while !self.store.len().is_multiple_of(8) {
            self.store.push(0);
        }
        self.at.push(self.store.len());
        self.store.extend_from_slice(bytes);
    }

    /// Push a device pointer argument (`T*` in the kernel signature).
    pub fn ptr(&mut self, p: CUdeviceptr) {
        self.push8(&p.to_ne_bytes());
    }

    /// Push a 64-bit scalar.
    pub fn u64(&mut self, v: u64) {
        self.push8(&v.to_ne_bytes());
    }

    /// Push a 32-bit scalar.
    pub fn u32(&mut self, v: u32) {
        self.push8(&v.to_ne_bytes());
    }

    /// Push a 32-bit float scalar.
    pub fn f32(&mut self, v: f32) {
        self.push8(&v.to_ne_bytes());
    }

    /// Pointers to pass as `kernelParams`. Valid until the next mutation.
    pub fn pointers(&mut self) -> Vec<*mut c_void> {
        let base = self.store.as_mut_ptr();
        self.at
            .iter()
            .map(|off| unsafe { base.add(*off) } as *mut c_void)
            .collect()
    }

    /// Raw pointer slots for `kernelParams` (alias of [`Self::pointers`]).
    pub fn as_ptrs(&mut self) -> Vec<*mut c_void> {
        self.pointers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_eight_byte_aligned() {
        let mut a = KernelArgs::new();
        a.u32(1);
        a.ptr(0xdead_beef);
        a.f32(0.5);
        assert_eq!(a.len(), 3);
        let ptrs = a.pointers();
        for p in &ptrs {
            assert_eq!(*p as usize % 8, 0, "argument slot not 8-byte aligned");
        }
    }

    #[test]
    fn pointers_stay_valid_across_pushes() {
        let mut a = KernelArgs::new();
        a.ptr(7);
        // Force the backing store to grow/reallocate.
        for i in 0..64 {
            a.u32(i);
        }
        let ptrs = a.pointers();
        let first = ptrs[0] as *const u64;
        assert_eq!(unsafe { *first }, 7);
    }
}
